I rechecked the updated repository by source review, not by running the benchmark locally. The reported gate result is consistent with the current implementation.

## Recheck verdict

The update moved `pgturbohybrid` in the right direction. The decisive signal in your report is not just that `document_nodes` returns `good`; it is that **`exact_token_scan` still returns `spike_1` and does not admit `good`**. That means the old token-node strategy is structurally lossy even when ANN error is removed. The repo’s audit now explicitly describes this distinction: `exact_token_scan` scores every token node but still feeds the same token-hit → document-accumulator → doc-candidate pipeline, so if that oracle still misses, the problem is token-top-k admission rather than graph ANN recall.

The new document-level paths solve the synthetic top-1 failure in the intended way: they score candidate documents with full document-level MaxSim before final heap rerank. The implementation now has `multivector_graph = token_nodes | document_nodes`; `document_nodes` is opt-in, stores one graph node per heap document, persists a float32 multivector sidecar, builds edges with symmetrized document MaxSim, and scans either by document-graph traversal with sidecar MaxSim or by exact sidecar scan when the search becomes exhaustive.

I would **not** read the `top10 admission recall = 0.200` for `plain_fallback`, `exact_doc_scan`, `doc_graph_prototype`, and `document_nodes` as a failure of those paths in this gate. The benchmark README says this deterministic recall gate is primarily a synthetic many-moderate top-1 admission gate; it fails unless the document-level exact paths return and admit the exact top-1. DBpedia admission and broad recall remain separate opt-in checks through normal benchmark runs plus `--admission-debug`.

## What looks good now

The implementation now has the right diagnostic surface. `TqDenseCandidateStats` includes candidate source, graph mode, exact-token scan counts, plain-fallback fields, document-graph stats, reservoir stats, BM25-injection stats, admission trace fields, and token stats. This should make DBpedia failures much easier to classify.

The document-node build path stores the original multivector for each document in build state and creates one graph node using a representative vector. More importantly, build-time graph distance for `document_nodes` switches to symmetrized full MaxSim over the stored document multivectors: `0.5 * (MaxSim(a,b)/count(a) + MaxSim(b,a)/count(b))`, returned as a distance by negation. That is the right graph-topology direction.

The document-node scan path now loads the document-vector sidecar, scores visited graph nodes with full `TqMultiVectorMaxSim(query, doc)`, and then exact-reranks heap multivectors. This is much closer to Qdrant’s point-level multivector model, where MaxSim returns a combined score per point rather than per subvector.   Qdrant’s public docs describe `max_sim` as a sum of maximum similarities between the matrices and emphasize that MaxSim returns a single combined score per point, not per subvector. ([Qdrant][1])

The fallback work also looks useful. `multivector_plain_fallback` now supports `off | auto | force` and can switch to exact heap MaxSim when estimated cardinality is small or candidate/rerank budgets are near-exhaustive. That is the correct safety net for the “10k corpus requires 6400+ candidates” case.

## Gaps I would fix next

The biggest code-level concern is **incremental insert for `document_nodes`**. Build-time document-node graph edges use symmetrized full MaxSim, but the insert path creates an averaged representative vector and calls the normal single-vector graph insert path. The sidecar is appended afterward. That means freshly inserted document-node graph connectivity may be based on average-vector geometry, not document-level MaxSim geometry.

That should be a P0 follow-up: implement a document-node-specific insert path that links the new document using the same full/symmetrized MaxSim objective as build. The new document’s full multivector can be kept in memory during insertion; existing neighbor candidates can be scored from the loaded document sidecar. Without this, bulk-built `document_nodes` indexes and incrementally updated `document_nodes` indexes may drift in quality.

The second concern is cost. The current document-node scan is correct but uses a float32 multivector sidecar and full MaxSim for graph traversal. The stats even report `multivectorDocGraphQuantizedScores = 0` and `doc_graph_f32`, and the design doc marks compact quantized document scoring as future work.   This is acceptable as a correctness-first production slice, but it will become the next bottleneck on DBpedia-scale or larger corpora.

The third concern is budget semantics. In `document_nodes`, `candidateLimit` is roughly bounded by `min(doc_candidate_k, docLimit * 2, doc_count)`, and `searchEf` is at least that candidate limit.  For real recall runs, you probably want separate knobs:

```text
turbohybrid.multivector_doc_graph_search_ef
turbohybrid.multivector_doc_graph_oversampling
turbohybrid.multivector_doc_graph_exact_rescore_k
```

Do not overload `doc_candidate_k` as both admission cap and graph traversal quality knob. Qdrant-style systems separate candidate-generation breadth, oversampling, and exact rescore.

The fourth concern is stats/estimation polish. Plain fallback estimates document count from relation estimates or metadata; in token-node indexes, `tqNodeCount` is token count, not document count, so fallback decisions should prefer `tqMultivectorDocCount` when a docmap exists.  Also, plain fallback currently labels graph mode with the default graph mode rather than necessarily the actual index graph mode, because it does not carry `meta` into that stats assignment. That is not a ranking bug, but it can confuse benchmark reports.

## Other optimizations worth integrating

### 1. Compact document-node MaxSim scoring

This is the highest-impact next step. Keep the current `document_nodes` architecture, but replace traversal-time float32 sidecar scoring with compact sidecar scoring:

```text
multivector_doc_storage = f32 | f16 | sq8 | pq | binary
multivector_doc_graph_oversampling = 2..20
multivector_doc_graph_rescore = sidecar_f32 | heap_exact
```

The search path becomes:

```text
document HNSW traversal
  -> approximate full MaxSim from compact document sidecar
  -> oversample candidate docs
  -> exact MaxSim rerank from f32 sidecar or heap
```

This directly mirrors the Qdrant lesson: quantization should approximate point-level MaxSim scoring, not change the retrieval identity from documents to token postings. Qdrant’s multivector API exposes MaxSim as a point-level comparator, and its hybrid query docs also model multi-stage retrieval as prefetch candidates followed by a main query/rerank over those candidates. ([Qdrant][1])

The simplest implementation order would be `f16` first, then per-dimension scalar int8, then maybe product quantization or binary quantization. ColBERTv2 is a useful reference here because it reduced late-interaction storage with residual compression while preserving retrieval quality. ([arXiv][2])

### 2. PLAID-lite as a legacy token-node rescue path

PLAID is still relevant even though `document_nodes` is now the better architecture. PLAID accelerates late interaction by treating each passage as a lightweight bag of centroids, using centroid interaction and centroid pruning to eliminate low-scoring passages before exact late-interaction scoring. The paper reports large CPU/GPU speedups without quality loss relative to vanilla ColBERTv2. ([arXiv][3])

For `pgturbohybrid`, PLAID-lite would be useful in two places:

```text
Old token_nodes indexes:
  centroid postings -> document candidates -> exact MaxSim rerank

Hybrid retrieval:
  BM25 candidates ∪ PLAID-lite candidates ∪ document_nodes candidates
  -> exact MaxSim rerank or RRF
```

I would not prioritize this above document-node compact scoring, but it is the best way to make the old token-node layout less pathological without requiring immediate reindexing into `document_nodes`.

### 3. MUVERA-style fixed-dimensional proxy branch

MUVERA reduces multi-vector retrieval to single-vector MIPS by generating fixed-dimensional encodings whose inner product approximates multi-vector similarity. The paper reports similar recall with fewer candidates and lower latency than prior multi-vector search heuristics. ([arXiv][4])

This fits `pgturbohybrid` especially well because you already have a mature single-vector graph path. Add a document-level proxy vector sidecar or second dense key:

```text
doc multivector -> proxy vector
query multivector -> proxy vector
existing TurboQuant dense graph retrieves proxy candidates
exact MaxSim reranks candidates
```

For hybrid retrieval, this becomes another prefetch branch:

```text
BM25 branch
proxy-vector branch
document-node MaxSim branch
optional exact MaxSim rerank over union
```

This is lower risk than implementing a completely new centroid engine, because it reuses existing dense graph infrastructure.

### 4. Learned sparse candidate generation: SPLATE/SPLADE

SPLATE is particularly relevant because it maps frozen ColBERTv2 token embeddings into a sparse vocabulary space, uses traditional sparse retrieval for candidate generation, and then reranks with ColBERTv2. The paper claims it matches PLAID effectiveness by reranking only 50 documents retrieved under 10 ms, making it attractive for CPU environments. ([arXiv][5])

SPLADE/SPLADEv2 is the broader learned-sparse family: sparse lexical/expansion vectors are compatible with inverted indexes and can compete strongly with dense and sparse methods. ([arXiv][6])

For `pgturbohybrid`, the integration path is:

```text
Add a learned_sparse_tsvector or weighted sparse key
Use BM25/impact/WAND-like infrastructure for sparse neural candidates
Inject sparse candidates into exact MaxSim rerank
Optionally fuse sparse branch and document-node branch with RRF
```

This is the strongest hybrid-retrieval extension because it turns “BM25 injection” into “semantic sparse injection.” BM25 remains useful, but learned sparse retrieval can recover semantic lexical variants that raw BM25 misses.

### 5. WARP-style and Col-Bandit-style MaxSim rerank pruning

Once candidate admission is fixed, the next cost center is exact MaxSim over many candidates. WARP targets multi-vector retrieval efficiency with dynamic similarity imputation, implicit decompression, and two-stage reduction; it reports major latency reductions relative to XTR/PLAID-style baselines. ([arXiv][7])

Col-Bandit is newer and directly attacks query-time MaxSim cost: it treats reranking as a top-K identification problem and adaptively reveals only the document/query-token MaxSim entries needed to decide the top results, reporting up to 5× MaxSim FLOP reductions. ([arXiv][8])

These are not candidate-generation fixes; they are **rerank cost reducers**. They become very attractive once hybrid retrieval starts producing larger candidate unions:

```text
BM25 candidates
+ learned sparse candidates
+ document_nodes candidates
+ proxy vector candidates
= larger union
-> pruned/adaptive exact MaxSim
```

That lets you improve admission recall without linearly increasing exact-rerank cost.

### 6. DESSERT-style vector-set retrieval

DESSERT is a general vector-set search algorithm with theoretical guarantees, and its ColBERT integration reports 2–5× speedups with minimal recall loss. ([arXiv][9])

I would treat DESSERT as a research branch rather than the next production path. It is attractive if `document_nodes` graph traversal with compact MaxSim still costs too much, but it is a bigger indexing departure than MUVERA-style proxy vectors or compact document-node scoring.

## Hybrid retrieval integration

The current state is much better for hybrid retrieval than the old token-node design because `document_nodes` makes the dense branch document-keyed from the start. The design doc already states that the dense document-node candidate list should behave as the dense branch in hybrid search, while BM25 remains document-keyed and RRF operates over document ranks.

I would define three hybrid modes clearly:

```text
1. dense_only_with_injection
   BM25 / sparse / proxy branches are candidate admission only.
   Final rank = exact MaxSim.

2. rrf
   Dense document-node branch produces ranks.
   BM25 / learned sparse branch produces ranks.
   Final rank = RRF over document IDs.

3. normalized_score_fusion
   Dense score = MaxSim / query_token_count.
   Sparse score = normalized BM25 or learned sparse score.
   Final rank = calibrated weighted sum or DBSF-like query-local normalization.
```

RRF should remain the safe default. Qdrant’s hybrid docs make the same point: RRF fuses by rank, while score fusion requires normalization because dense and sparse scores live on different scales. Qdrant’s DBSF normalizes each retriever’s returned-score distribution before summing, and its docs warn that raw alpha-weighted dense/sparse score combinations are unreliable without normalization. ([Qdrant][10])

For `pgturbohybrid`, the most practical hybrid pipeline is:

```text
Prefetch/admission:
  A. document_nodes MaxSim graph candidates
  B. BM25 or learned sparse candidates
  C. optional MUVERA/proxy-vector candidates

Candidate union:
  document-keyed dedupe
  preserve branch ranks and branch scores
  keep admission provenance stats

Scoring:
  dense-only mode -> exact MaxSim over union
  RRF mode -> RRF over branch ranks
  score-fusion mode -> normalize scores, then combine

Final:
  optional exact MaxSim rerank or MaxSim-aware diversity
```

The existing benchmark already has flags for BM25 candidate injection, candidate reservoirs, exact doc scan, doc graph prototype, document nodes, token ablation, and admission debug. The README documents `--multivector-bm25-candidate-injection off|hybrid_only|dense_with_text`, where `dense_with_text` uses BM25 as admission safety rather than final scorer. That is exactly the right hybrid semantics.

## Recommended next engineering prompts

### P0: fix document-node incremental insert geometry

```text
Implement document-node-specific incremental insert for multivector_graph=document_nodes.

Do not link new document nodes using only the averaged representative vector.
Use the same symmetrized full MaxSim objective used by bulk build:
  0.5 * (MaxSim(new, existing)/count(new) + MaxSim(existing, new)/count(existing))

During insert:
- keep the new document multivector in memory,
- load existing document vectors from the sidecar/cache,
- score candidate neighbors with full document MaxSim,
- use the same score for neighbor selection and reciprocal pruning,
- append sidecar records safely,
- add tests comparing bulk build vs insert-after-build topology/recall on the many-moderate corpus.
```

### P1: add document-node oversampling and DBpedia admission gate

```text
Add:
  turbohybrid.multivector_doc_graph_search_ef
  turbohybrid.multivector_doc_graph_oversampling
  turbohybrid.multivector_doc_graph_rescore_k

Make document_nodes candidateLimit independent from final_k * 2.
Benchmark DBpedia admission with:
  exact_top1_admission
  exact_top10_admission_recall
  latency
  docs_scored
  graph_edges_visited
  exact_rerank_docs

Acceptance:
  document_nodes should admit exact top-1/top-10 at far lower budgets than token_graph and exact_token_scan.
```

### P1: compact document-node sidecar scoring

```text
Add compact document-node sidecar storage:
  multivector_doc_storage = f32 | f16 | sq8

Search:
  graph traversal scores compact sidecar MaxSim,
  oversamples candidates,
  exact-reranks from f32 sidecar or heap.

Stats:
  multivector_doc_graph_quantized_scores
  multivector_doc_graph_storage_kind
  multivector_doc_graph_oversampling
  multivector_doc_graph_rescore_source

Keep f32 as correctness reference.
```

### P2: hybrid branch scheduler

```text
Implement a branch-aware hybrid admission scheduler for multivector queries.

Branches:
  dense_document_nodes
  bm25
  learned_sparse_future
  proxy_vector_future

Modes:
  dense_only_with_injection -> final exact MaxSim
  rrf -> branch-rank fusion
  dbsf_or_normalized_score -> explicit normalized score fusion only

Use admission-debug stats to adapt branch budgets.
Do not allow raw BM25 + raw MaxSim alpha fusion without normalization.
```

### P2: MUVERA-style proxy branch prototype

```text
Prototype a fixed-dimensional proxy vector for each document multivector.

Use existing single-vector TurboQuant graph as candidate generator:
  proxy candidates -> exact MaxSim rerank

Expose as:
  turbohybrid.multivector_candidate_source = proxy_vector | document_nodes | graph

Hybrid:
  allow proxy branch as another prefetch/admission source.
```

## Bottom line

The current update fixes the core conceptual bug for new `document_nodes` indexes: candidate generation can now operate at document level and use full MaxSim for visited candidates. The synthetic gate result is exactly what I would expect if that fix is working.

The next work should not be more token-node reservoir tuning. Keep the token path for compatibility and diagnostics, but focus on:

```text
1. document-node insert correctness,
2. document-node oversampling/search-ef tuning,
3. compact document MaxSim scoring,
4. hybrid branch scheduling with RRF or normalized score fusion,
5. rerank pruning so larger hybrid candidate unions stay cheap.
```

That is the path most consistent with Qdrant’s point-level MaxSim design, ColBERT/PLAID-style late-interaction efficiency work, and practical hybrid retrieval.

[1]: https://qdrant.tech/documentation/concepts/vectors/ "Vectors - Qdrant"
[2]: https://arxiv.org/abs/2112.01488?utm_source=chatgpt.com "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction"
[3]: https://arxiv.org/abs/2205.09707?utm_source=chatgpt.com "PLAID: An Efficient Engine for Late Interaction Retrieval"
[4]: https://arxiv.org/abs/2405.19504?utm_source=chatgpt.com "MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings"
[5]: https://arxiv.org/abs/2404.13950?utm_source=chatgpt.com "SPLATE: Sparse Late Interaction Retrieval"
[6]: https://arxiv.org/abs/2109.10086?utm_source=chatgpt.com "SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval"
[7]: https://arxiv.org/abs/2501.17788?utm_source=chatgpt.com "WARP: An Efficient Engine for Multi-Vector Retrieval"
[8]: https://arxiv.org/abs/2602.02827?utm_source=chatgpt.com "Col-Bandit: Zero-Shot Query-Time Pruning for Late-Interaction Retrieval"
[9]: https://arxiv.org/abs/2210.15748?utm_source=chatgpt.com "DESSERT: An Efficient Algorithm for Vector Set Search with Vector Set Queries"
[10]: https://qdrant.tech/documentation/concepts/hybrid-queries/ "Hybrid Queries - Qdrant"
