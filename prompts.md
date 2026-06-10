## Current `pgturbohybrid` state to build from

`pgturbohybrid` is no longer only a token-node ColBERT approximation. The current docs show two paths: the legacy `token_nodes` path, where query tokens retrieve token/subvector graph hits and then aggregate to documents, and the new explicit `document_nodes` path, where one graph node represents one heap document/tuple version and search scoring is intended to be full document-level MaxSim.

The current benchmark docs also show that Prompt 11 has already started a production document-node path: `multivector_graph = token_nodes | document_nodes`, one graph node per heap document, a versioned index-resident multivector sidecar, symmetrized document MaxSim for build-time edge selection, document-sidecar scoring before heap rerank, and storage modes `f32 | f16 | sq8`. It also now has `proxy_vector`, document-graph knobs, RRF, normalized score fusion, and adaptive hybrid budgeting.

So the next best ideas are **not** “make token-node candidate generation wider.” The strongest direction is to make `document_nodes` fast, robust, hybrid-aware, and model-aware, while keeping `token_nodes` only as compatibility/debug mode.

---

## What modern systems and papers are doing

### Qdrant: point-level MaxSim + prefetch pipelines

Qdrant’s multivectors store a variable number of same-shaped dense vectors per point, and its MaxSim comparator returns a **single combined score per point**, not per subvector. Qdrant explicitly defines `max_sim` as the sum of maximum similarities between vectors in the matrices, which matches the semantic direction `pgturbohybrid` is moving toward with `document_nodes`. ([Qdrant][1])

Qdrant’s docs also recommend using ColBERT as a reranker over a small candidate set for large-scale retrieval, because ColBERT improves semantic granularity but costs more memory and speed than single-vector retrieval. They suggest dense retrieval for roughly 100–500 initial candidates followed by ColBERT reranking. ([Qdrant][2])

The most copyable Qdrant API idea is its **prefetch/multi-stage query model**. It can run cheap first-stage retrieval, then rescore with a larger or multivector representation, and it supports nested prefetches such as byte-vector prefetch → full dense vector prefetch → ColBERT multivector final query. ([Qdrant][3])

For hybrid, Qdrant exposes both RRF and DBSF. RRF fuses by ranks; DBSF keeps raw scores but normalizes each retriever’s returned-score distribution before summing, and Qdrant warns that no fusion method dominates universally, so an eval set should choose. ([Qdrant][3])

**Copy:** nested branch/prefetch planning, explicit branch budgets, point-level MaxSim scoring, oversampling before exact rerank, DBSF-like score fusion.

---

### Vespa: tensor-native ColBERT, long context, int8 compression, phased ranking

Vespa’s long-context ColBERT implementation stores token vectors as tensors such as `tensor<int8>(context{}, token{}, v[16])`, with `context` representing sliding context windows and `token` representing token position. It uses compressed token vectors and can store them as paged attributes, allowing OS paging for large token tensors. ([Vespa Blog][4])

Vespa expresses MaxSim directly in rank profiles using tensor operations. It supports context-level and cross-context MaxSim over long documents, and explicitly discusses using ColBERT as a second-phase ranking expression after a first-stage retrieval operator. ([Vespa Blog][4])

Vespa also notes that structured fields can either be concatenated or stored as several ColBERT tensors, letting ranking expressions weight multiple MaxSim calculations per field. ([Vespa Blog][4])

**Copy:** context-window-aware multivectors, field-specific MaxSim, paged sidecar storage, phased ranking, int8 token compression, and “ColBERT as second phase over constrained candidates” as an explicit planner mode.

---

### Milvus: hybrid branch model + sparse WAND/MaxScore + normalized weighted fusion

Milvus’ “multi-vector hybrid search” is not ColBERT MaxSim; it is multiple vector fields searched simultaneously, such as dense text, sparse BM25/SPLADE-like text, and image dense vectors. It then reranks/fuses the result sets. ([Milvus][5])

Milvus’ sparse index supports BM25 with `DAAT_MAXSCORE`, and its docs list `DAAT_MAXSCORE`, `DAAT_WAND`, and `TAAT_NAIVE` as inverted-index algorithm choices. ([Milvus][5])

Milvus’ `WeightedRanker` normalizes route scores into `[0,1]`, using an arctan transform because IP, L2, and other metrics live on different scales; its `RRFRanker` fuses by rank, typically with smoothing constant 60. ([Milvus][6])

**Copy:** make hybrid retrieval branch-native, preserve per-branch score/rank provenance, add a calibrated route-normalization mode, and expose sparse branch algorithm choices similar to WAND/MaxScore.

---

### PLAID / ColBERTv2: centroid interaction and progressive pruning

PLAID accelerates ColBERTv2 by treating each passage as a lightweight bag of centroids, using centroid interaction and centroid pruning before exact scoring. The paper reports up to 7× GPU and 45× CPU speedups versus vanilla ColBERTv2 without quality loss. ([arXiv][7])

A 2024 PLAID reproducibility study found that PLAID’s Pareto frontier depends on carefully balancing parameters, and that lexical reranking can be very competitive at low-latency points, but cannot reach peak exhaustive ColBERT effectiveness because lexical candidate recall is limited. ([arXiv][8])

**Copy:** a PLAID-lite centroid sidecar for `token_nodes` compatibility and as an optional `document_nodes` candidate prefilter, but only after the current document-node path has DBpedia-scale admission data.

---

### SPLATE / SLIM / learned sparse: inverted-index candidate generation for late interaction

SPLATE maps frozen ColBERTv2 token embeddings into sparse vocabulary space so traditional sparse retrieval can generate candidates, then reranks with ColBERT MaxSim. The paper says it matches PLAID ColBERTv2 effectiveness by reranking 50 documents retrieved under 10 ms. ([arXiv][9])

SLIM similarly maps contextual token vectors to sparse lexical space and uses inverted-index retrieval plus refinement, explicitly targeting compatibility with off-the-shelf lexical search libraries such as Lucene. ([arXiv][10])

**Copy:** learned-sparse candidate injection using existing BM25/WAND infrastructure, with exact MaxSim final ranking. This is especially attractive for hybrid retrieval because it upgrades BM25 injection into semantic sparse injection.

---

### MUVERA / LEMUR: reduce multivector search to single-vector ANN

MUVERA generates fixed-dimensional encodings of query and document multivectors whose inner product approximates multivector similarity, allowing off-the-shelf MIPS/ANN. The paper reports 2–5× fewer retrieved candidates and 90% lower latency with improved average recall in its evaluations. ([arXiv][11])

LEMUR, published in 2026, also reduces multivector similarity to single-vector search through a learned latent-space reduction and reports order-of-magnitude speedups on ColBERTv2 and modern multivector models. ([arXiv][12])

**Copy:** upgrade the current `proxy_vector` prototype into a real learned/provable proxy branch: first heuristic representative vectors, then MUVERA-style fixed-dimensional encodings, then learned LEMUR-like projection if benchmarked.

---

### WARP / Col-Bandit: exact MaxSim rerank cost reduction

WARP targets multi-vector scoring speed with dynamic similarity imputation, implicit decompression, and two-stage reduction; it reports 41× speedup over an XTR reference implementation and 3× speedup over official PLAID while preserving quality. ([arXiv][13])

Col-Bandit, a 2026 paper, reduces query-time MaxSim FLOPs by adaptively revealing only the document/query-token interactions needed to identify top-K, without index modifications or retraining. It reports up to 5× MaxSim FLOP reductions. ([arXiv][14])

**Copy:** after admission recall is fixed, reduce exact rerank cost with adaptive per-query-token MaxSim pruning and two-stage MaxSim reductions.

---

### Token pooling / ColBERTSaR: reduce storage and gather cost

Token pooling clusters document token vectors during indexing to reduce the number of vectors stored. The 2024 paper reports a 50% footprint reduction with virtually no retrieval degradation, and further 66–75% reductions with degradation under 5% on most datasets. ([arXiv][15])

A very recent 2026 paper, ColBERTSaR, proposes turning a ColBERT index into a true inverted index via embedding quantization and reports 50–70% smaller indexes than one-bit PLAID while retaining effectiveness. This is new enough that I would treat it as an experimental branch, not an immediate production dependency. ([arXiv][16])

**Copy:** token pooling should be near-term because it is model-agnostic and simple. ColBERTSaR-style quantized inverted indexes are worth a research prototype later.

---

### ModernColBERT / Jina-ColBERT-v2 / PyLate model compatibility

Jina-ColBERT-v2 is a multilingual late-interaction retriever with architectural/training improvements for broader multilingual retrieval. ([arXiv][17])

PyLate adds late-interaction support on top of Sentence Transformers and has enabled GTE-ModernColBERT and Reason-ModernColBERT. ([arXiv][18])

Qdrant’s FastEmbed docs list supported late-interaction models such as `colbert-ir/colbertv2.0` and `answerdotai/answerai-colbert-small-v1`, with dimensions 128 and 96 respectively. ([Qdrant][2])

**Copy:** model-metadata-driven multivector ingestion: dimensions, token limits, token masks, query/doc role, pooling policy, storage kind, normalization, and per-token weights must not be hardcoded to one ColBERT variant.

---

# What `pgturbohybrid` should copy next

## Highest priority

1. **Document-node insert correctness.** Current build uses symmetrized MaxSim, but incremental insert must be audited so it does not link document nodes using only representative-vector geometry. This matters because document-node graph quality will drift after inserts if insert geometry differs from build geometry.

2. **DBpedia-scale document-node admission/latency benchmark.** The synthetic gate proves top-1 admission on a constructed case, but the benchmark docs explicitly say DBpedia admission and recall quality are separate opt-in checks.

3. **Qdrant-style branch/prefetch planner.** `pgturbohybrid` already has branches; make them explicit, nested, budgeted, and visible in stats.

4. **Vespa-style long-context and field-aware MaxSim.** This is the clearest route for ModernColBERT and enterprise documents: title/body/section fields, context windows, and paged storage.

5. **Rerank cost pruning.** As hybrid admission improves, candidate unions grow; copy WARP/Col-Bandit ideas to keep exact MaxSim affordable.

## Medium priority

6. **MUVERA/LEMUR proxy branch.** `proxy_vector` exists; make it a real branch with FDE/learned projections and admission benchmarks.

7. **Token pooling.** It reduces storage and MaxSim cost without changing query-time logic.

8. **Learned sparse branch.** Upgrade BM25 injection into SPLATE/SLIM-style semantic sparse injection.

## Research priority

9. **PLAID-lite centroids.** Useful for legacy `token_nodes` and possibly as a fast document-node sidecar prefilter.

10. **ColBERTSaR-style quantized inverted index.** Promising but too new to make core yet.

---

# Codex prompts

Run these one at a time. They assume the current repository state with `document_nodes`, `f32|f16|sq8` sidecar storage, `proxy_vector`, normalized hybrid fusion, and admission-debug infrastructure.

---

## Prompt 1 — Audit and fix document-node incremental insert geometry

```text
You are working in agentxagi/pgturbohybrid.

Goal:
Ensure incremental insert/update for multivector_graph=document_nodes uses the same document-level MaxSim graph geometry as bulk build.

Context:
Bulk build for document_nodes uses one graph node per heap document and symmetrized document MaxSim for graph edge selection. Incremental insert must not link a new document node using only an averaged representative vector if that changes the graph topology semantics.

Tasks:
1. Audit PgturbohybridGraphInsertMultiVectorBatchInPlace and related insert helpers.
2. If document_nodes insert still uses a representative vector for graph neighbor search/link selection, implement a document-node-specific insert path.
3. During insert:
   - keep the inserted PgturbohybridMultiVector in memory;
   - load existing document multivectors from the document-node sidecar/cache;
   - score candidate neighbors with the same symmetrized MaxSim used by bulk build:
     0.5 * (MaxSim(new, existing)/count(new) + MaxSim(existing, new)/count(existing));
   - use that score for entry search, neighbor selection, and reciprocal pruning;
   - append the document-node sidecar after graph insertion with WAL-safe metadata updates.
4. Add stats:
   - multivector_doc_graph_insert_full_maxsim_edges
   - multivector_doc_graph_insert_representative_fallbacks
   - multivector_doc_graph_insert_pairs_scored
5. Add tests:
   - build document_nodes index, insert a new many-moderate document, verify it is retrievable;
   - compare bulk-build vs insert-after-build recall on the synthetic many-moderate corpus;
   - update/delete/vacuum visibility still passes.
6. Update docs/dev/multivector-document-graph-design.md and benchmarks/README.md.

Acceptance:
- document_nodes insert never silently uses representative-vector geometry for graph links unless a clearly named debug fallback is enabled.
- bulk and incremental paths use the same documented document-level scorer.
```

---

## Prompt 2 — Add DBpedia document-node admission benchmark gate

```text
You are working in agentxagi/pgturbohybrid.

Goal:
Turn the current synthetic recall gate into a DBpedia-scale admission gate for document_nodes.

Context:
The synthetic gate proves exact top-1 admission on a constructed case. We now need DBpedia evidence for:
- token_nodes,
- exact_token_scan,
- document_nodes f32/f16/sq8,
- proxy_vector,
- plain fallback,
- exact_doc_scan.

Tasks:
1. Extend benchmarks/dbpedia_colbert_multivector.py with:
   --document-node-admission-grid
   --document-node-storage-grid f32,f16,sq8
   --document-node-ef-grid 50,100,200,400,800
   --document-node-oversampling-grid 1,2,4,8
2. For each query and mode, collect:
   - exact_top1_admitted
   - exact_top10_admission_recall
   - exact_top1_rank
   - final NDCG/Recall/MRR where qrels exist
   - latency p50/p95
   - docs_scored
   - edges_visited
   - exact_rerank_docs
   - sidecar bytes read/cache hit stats
   - storage kind
3. Emit JSON and Markdown summaries.
4. Add a benchmark README section with a required 10k DBpedia command and
   optional 100k/1M scale-up commands.
5. Do not require external DBpedia data in normal CI.

Acceptance:
- The report can show whether document_nodes beats token_nodes and exact_token_scan on admission at lower budgets.
- f16/sq8 quality loss is visible against f32.
- 10k DBpedia evidence is sufficient for this prompt's acceptance gate; 100k
  and 1M runs are optional scale checks, not required completion evidence.
```

---

## Prompt 3 — Implement Qdrant-style nested prefetch branch planner

```text
You are working in agentxagi/pgturbohybrid.

Goal:
Implement a branch-aware, nested prefetch planner for multivector/hybrid retrieval.

Inspiration:
Qdrant supports prefetch pipelines where cheap candidates feed a more expensive scorer, and hybrid branches can be fused with RRF or DBSF.

Current state:
pgturbohybrid already has dense multivector, BM25, proxy_vector, document_nodes, RRF, normalized score fusion, and adaptive hybrid budgeting.

Tasks:
1. Add an internal branch plan struct:
   - branch kind: bm25, dense_single, proxy_vector, document_nodes, token_nodes, exact_doc_scan
   - candidate_limit
   - rescore_limit
   - branch_rank
   - branch_score
   - branch_source flags
2. Add a GUC:
   turbohybrid.multivector_branch_plan = auto | dense_only | qdrant_like
3. In qdrant_like mode support nested plans:
   - proxy_vector -> document_nodes MaxSim -> exact heap MaxSim
   - BM25/sparse -> exact MaxSim
   - dense_single/proxy + BM25 -> RRF/DBSF/normalized fusion
4. Preserve existing SQL API initially; this is internal planning.
5. Add scan stats:
   - branch_count
   - branch_kinds
   - branch_candidate_counts
   - branch_truncated_flags
   - branch_latency_us
   - branch_fusion_mode
6. Add tests:
   - branch dedupe is heap/document keyed;
   - dense-only with BM25 injection still ranks by exact MaxSim;
   - RRF path preserves branch ranks;
   - normalized fusion never combines raw BM25 and raw MaxSim without normalization.

Acceptance:
- Branch plans are deterministic and visible in turbohybrid_last_scan_stats().
- Existing behavior remains default unless qdrant_like is selected.
```

---

## Prompt 4 — Add DBSF-style distribution-based score fusion

```text
You are working in agentxagi/pgturbohybrid.

Goal:
Add Qdrant-style DBSF fusion for multivector hybrid retrieval.

Context:
Qdrant DBSF normalizes each retriever’s returned score distribution before combining. pgturbohybrid already supports normalized score fusion modes, but we need an explicit DBSF mode with branch-local diagnostics.

Tasks:
1. Add fusion mode:
   fusion => 'dbsf'
2. For each branch:
   - collect raw scores from the branch candidate set;
   - compute mean and sample standard deviation;
   - normalize using clipped 3-sigma endpoints or a documented robust alternative;
   - handle degenerate stddev=0 safely.
3. Combine normalized branch scores by sum or configurable weights.
4. Add GUCs:
   turbohybrid.dbsf_sigma = 3.0
   turbohybrid.dbsf_min_branch_candidates = 10
   turbohybrid.dbsf_robust = off | mad
5. Add stats:
   - dbsf_enabled
   - dbsf_branch_mean/stddev
   - dbsf_branch_min/max
   - dbsf_degenerate_branches
6. Tests:
   - score scale mismatch: BM25 large values + MaxSim small values;
   - degenerate branch with identical scores;
   - RRF and calibrated modes unchanged.

Acceptance:
- DBSF is never used silently.
- Fusion remains document-keyed.
- Docs explain when RRF is safer than DBSF.
```

---

## Prompt 5 — Add Vespa-style long-context multivector support

```text
You are working in agentxagi/pgturbohybrid.

Goal:
Support long-context ColBERT/ModernColBERT documents without forcing users to split each context window into a separate SQL row.

Inspiration:
Vespa stores ColBERT as tensors with context and token dimensions and supports context-level and cross-context MaxSim.

Tasks:
1. Extend turbohybrid_multivector metadata or add a new internal layout that can represent:
   - context/window ordinal
   - token ordinal within context
   - field id or section id
2. Add builder function:
   turbohybrid_multivector_from_contexts(raw_values real[], dim int, context_offsets int[])
3. Add scoring modes:
   - cross_context_maxsim: current global MaxSim across all doc tokens
   - context_level_maxsim: score each context independently, then max or top-N aggregate
   - field_weighted_maxsim: weighted sum of MaxSim across named fields/sections
4. Add index options:
   multivector_context_mode = flat | context_level
   multivector_field_mode = off | weighted
5. Add docs and examples for title/body/section fields.
6. Add tests:
   - cross-context equals current MaxSim when all tokens are flat;
   - context-level max chooses the best window;
   - field weights affect rank deterministically;
   - MVCC and exact rerank still use heap/document identity.

Acceptance:
- Long documents can remain one SQL result row.
- Context/field modes are explicit and benchmarkable.
```

---

## Prompt 6 — Add paged/cold document multivector sidecar mode

```text
You are working in agentxagi/pgturbohybrid.

Goal:
Add a Vespa-inspired cold/paged sidecar mode for large document-node multivector storage.

Context:
document_nodes currently uses index-resident sidecar storage. For large corpora, f32/f16/sq8 sidecar memory can dominate. Vespa uses paged attributes to let the OS page large tensors.

Tasks:
1. Design and implement:
   multivector_doc_storage_cache = resident | paged | auto
2. In paged mode:
   - keep document-node graph adjacency and compact metadata resident;
   - memory-map or page-load document sidecar chunks on demand;
   - track page/cache misses separately from graph code pages.
3. Add stats:
   - multivector_doc_sidecar_cache_mode
   - multivector_doc_sidecar_pages_read
   - multivector_doc_sidecar_cache_hits
   - multivector_doc_sidecar_bytes_touched
4. Add fallback:
   - for low latency profile, prefer resident if under memory cap;
   - for large corpora, auto chooses paged.
5. Add benchmark support for cache-cold and cache-warm document_nodes scans.

Acceptance:
- Large document_nodes indexes can run without loading all multivectors into backend memory.
- Stats make random access cost visible.
```

---

## Prompt 7 — Implement token pooling for multivector storage reduction

```text
You are working in agentxagi/pgturbohybrid.

Goal:
Implement model-agnostic token pooling to reduce document multivector size before indexing.

Inspiration:
Recent token pooling work reports ~50% ColBERT index footprint reduction with minimal retrieval degradation.

Tasks:
1. Add optional index-time document token pooling:
   multivector_token_pooling = off | kmeans | greedy_cosine
   multivector_token_pooling_target_ratio = 0.5
   multivector_token_pooling_min_tokens = 16
2. Pool only document tokens, not query tokens.
3. Store original token count and pooled token count in doc sidecar metadata.
4. Exact rerank options:
   - rerank over pooled sidecar;
   - optionally heap exact over original multivector if heap value is available.
5. Add stats:
   - multivector_tokens_original
   - multivector_tokens_pooled
   - multivector_token_pooling_ratio
6. Add benchmark grid:
   ratios 1.0, 0.75, 0.5, 0.33
   storage f32/f16/sq8
7. Tests:
   - deterministic pooling on small fixtures;
   - pooled exact rerank remains stable;
   - invalid ratios rejected.

Acceptance:
- Pooling is opt-in.
- Benchmark shows storage/latency/recall tradeoff.
```

---

## Prompt 8 — Upgrade proxy_vector toward MUVERA-style fixed-dimensional encodings

```text
You are working in agentxagi/pgturbohybrid.

Goal:
Turn the current proxy_vector prototype into a serious fixed-dimensional multivector proxy branch.

Inspiration:
MUVERA reduces multivector retrieval to single-vector MIPS using fixed-dimensional encodings. LEMUR similarly reduces multivector search to a learned latent single-vector search.

Current state:
proxy_vector currently uses the existing single-vector TurboQuant graph over document representative vectors for admission, then exact MaxSim rerank.

Tasks:
1. Add a proxy encoder abstraction:
   - mean_pool
   - max_pool
   - random_projection_fde
   - learned_projection_placeholder
2. Store proxy vector per document node as a normal dense graph key or sidecar.
3. Add query proxy generation for the same encoder.
4. Branch plan:
   proxy_vector -> exact MaxSim rerank
   proxy_vector -> document_nodes MaxSim -> exact heap MaxSim
5. Add stats:
   - proxy_encoder_kind
   - proxy_candidates
   - proxy_top1_admission
   - proxy_exact_rerank_docs
6. Add benchmark:
   - compare proxy encoders against document_nodes and exact_doc_scan;
   - report candidates required to admit exact top-1/top-10.

Acceptance:
- Existing representative proxy remains available.
- New proxy encoders are pluggable and benchmarked.
```

---

## Prompt 9 — Add learned sparse multivector candidate injection

```text
You are working in agentxagi/pgturbohybrid.

Goal:
Add a learned-sparse branch for ColBERT/ModernColBERT candidate generation, inspired by SPLATE/SLIM.

Context:
pgturbohybrid already has BM25, sparse/lexical infrastructure, BM25 candidate injection, and hybrid fusion. Learned sparse candidate generation can improve admission recall while keeping final exact MaxSim.

Tasks:
1. Define a sparse sidecar/input format:
   - document id
   - sparse term ids
   - weights
   - optional field id
2. Add SQL ingestion helper:
   turbohybrid_sparse_vector_from_arrays(term_ids int[], weights real[])
3. Add branch:
   multivector_sparse_candidate_source = off | bm25 | learned_sparse
4. Use existing sparse/BM25/WAND/impact infrastructure where possible.
5. For dense-only-with-text:
   - learned_sparse candidates are admission-only;
   - final rank remains exact MaxSim.
6. For hybrid:
   - learned_sparse can participate in RRF/DBSF/calibrated fusion.
7. Add stats:
   - learned_sparse_candidates
   - learned_sparse_retained_for_maxsim
   - learned_sparse_branch_latency_us
8. Add benchmark hooks for SPLADE/SPLATE-style exported sparse vectors.

Acceptance:
- No model training is required inside PostgreSQL.
- Candidate injection is document-keyed and exact-MaxSim-reranked.
```

---

## Prompt 10 — Implement adaptive MaxSim rerank pruning

```text
You are working in agentxagi/pgturbohybrid.

Goal:
Reduce exact MaxSim rerank cost after candidate admission is fixed.

Inspiration:
WARP and Col-Bandit reduce MaxSim computation by avoiding unnecessary full interaction matrix computation.

Tasks:
1. Add exact rerank mode:
   turbohybrid.multivector_exact_rerank = off | topk | adaptive
2. Adaptive mode:
   - compute cheap upper/lower bounds per candidate;
   - process query tokens in an importance order;
   - maintain top-K threshold;
   - stop scoring a candidate once it cannot enter top-K;
   - always allow exact/full mode for parity.
3. Start with safe deterministic bounds:
   - max possible remaining contribution from query-token norms;
   - precomputed per-document token norm maxima;
   - query-token IDF/importance order if available.
4. Add stats:
   - exact_rerank_candidates
   - exact_rerank_tokens_evaluated
   - exact_rerank_tokens_skipped
   - exact_rerank_pairs_saved
   - adaptive_rerank_topk_changed_vs_full
5. Add tests:
   - adaptive result equals full exact on deterministic fixtures;
   - fallback to full exact when bounds are unsafe;
   - benchmark DBpedia pair savings.

Acceptance:
- Adaptive mode must be exact by default.
- Any approximate relaxation needs a separate explicit GUC.
```

---

## Prompt 11 — Add query-token importance and masking

```text
You are working in agentxagi/pgturbohybrid.

Goal:
Support model/query-token importance for ModernColBERT and faster MaxSim.

Inspiration:
Token-importance work improves late-interaction scoring by weighting query-token contributions. Vespa/Qdrant-style ColBERT deployments also benefit from token masking for punctuation/special/noisy tokens.

Tasks:
1. Extend turbohybrid_query multivector payload to optionally carry:
   - query_token_weights real[]
   - query_token_mask bool[]
2. Extend MaxSim:
   score(Q,D) = sum_i weight_i * max_j sim(q_i, d_j)
3. Exact rerank and document-node sidecar scoring must both support weights/masks.
4. Candidate generation:
   - skip masked tokens in token_nodes;
   - use weights to order adaptive exact rerank;
   - document_nodes scoring uses weighted MaxSim.
5. Add model metadata defaults:
   - special token masking
   - punctuation token masking
   - IDF weighting hook
6. Add tests:
   - weights reproduce unweighted when all weights = 1;
   - masked tokens do not affect exact score;
   - weighted and unweighted rankings differ deterministically on fixture.

Acceptance:
- ModernColBERT-specific token weighting is supported without hardcoding a model.
```

---

## Prompt 12 — Add ModernColBERT / ColPali model metadata registry

```text
You are working in agentxagi/pgturbohybrid.

Goal:
Make pgturbohybrid robust to modern late-interaction models with different dimensions, query/doc token behavior, and pooling/masking rules.

Context:
Current and emerging models include ColBERTv2, Jina-ColBERT-v2, AnswerAI ColBERT, GTE-ModernColBERT, Reason-ModernColBERT, and ColPali-style visual-document multivectors.

Tasks:
1. Add a model metadata table or extension config:
   - model_name
   - dim
   - default_query_max_tokens
   - default_doc_max_tokens
   - distance mode
   - normalized tokens yes/no
   - recommended storage kind f32/f16/sq8
   - token mask policy
   - optional field/context policy
2. Add SQL helper:
   turbohybrid_multivector_model_info(model_name text)
3. Add validation:
   - reject wrong dimensions with model-aware hints;
   - warn on suspicious token counts;
   - expose model metadata in index stats.
4. Add benchmark support:
   --colbert-model-name
   --expected-dim auto
5. Add docs:
   - ColBERTv2
   - AnswerAI ColBERT small
   - Jina-ColBERT-v2
   - GTE/Reason ModernColBERT placeholders
   - ColPali-like visual multivectors

Acceptance:
- No hardcoded assumption that ColBERT vectors are always 128 dimensions.
- Model metadata improves error messages and benchmark reproducibility.
```

---

## Prompt 13 — PLAID-lite centroid sidecar for compatibility mode

```text
You are working in agentxagi/pgturbohybrid.

Goal:
Add a PLAID-lite centroid sidecar as an optional candidate generator, mainly for token_nodes compatibility and for fast prefiltering.

Inspiration:
PLAID uses centroid interaction and centroid pruning before exact ColBERT scoring.

Tasks:
1. Add index option:
   multivector_centroids = off | kmeans
   multivector_centroid_count = auto | integer
2. Build:
   - cluster document token vectors into centroids;
   - store per-document centroid ids and residual summary;
   - preserve original document multivectors for exact rerank.
3. Search:
   - map query tokens to nearest centroids;
   - collect documents from centroid postings;
   - compute centroid-interaction approximate MaxSim;
   - exact-rerank top documents.
4. Stats:
   - centroid_lists_visited
   - centroid_docs_touched
   - centroid_pruned_docs
   - centroid_candidates
5. Benchmarks:
   - token_nodes vs centroid_lite vs document_nodes;
   - DBpedia admission and latency.

Acceptance:
- This must not replace document_nodes as the primary path.
- It must be opt-in and documented as compatibility/experimental.
```

---

## Prompt 14 — ColBERTSaR-style quantized inverted-index research branch

```text
You are working in agentxagi/pgturbohybrid.

Goal:
Create a research-only branch for quantized inverted-index ColBERT candidate generation.

Inspiration:
Recent ColBERTSaR work suggests embedding quantization can turn ColBERT indexing into a true inverted index and shrink indexes relative to one-bit PLAID.

Tasks:
1. Create docs/dev/multivector-colbertsar-research.md.
2. Prototype only under:
   turbohybrid.multivector_candidate_source = quantized_inverted_experimental
3. Quantize token embeddings into learned/codebook terms.
4. Build postings:
   codeword -> docId, tokenOrdinal, quantized residual/score payload
5. Query:
   quantize query tokens;
   retrieve postings;
   approximate score;
   exact MaxSim rerank.
6. Add warnings:
   - experimental;
   - not default;
   - storage format unstable.
7. Benchmark against learned_sparse, PLAID-lite, token_nodes, and document_nodes.

Acceptance:
- No production on-disk compatibility promise.
- Useful enough to compare storage size and admission recall.
```

---

## Prompt 15 — End-to-end hybrid evaluation harness

```text
You are working in agentxagi/pgturbohybrid.

Goal:
Add an end-to-end hybrid evaluation harness for document_nodes + BM25 + learned_sparse/proxy branches.

Tasks:
1. Extend benchmarks/dbpedia_colbert_multivector.py to run these modes:
   - exact_scan
   - document_nodes
   - document_nodes + BM25 admission-only
   - document_nodes + BM25 RRF
   - document_nodes + BM25 DBSF
   - proxy_vector -> document_nodes -> exact MaxSim
   - learned_sparse -> exact MaxSim, if available
2. For each mode report:
   - BEIR metrics
   - admission recall
   - latency p50/p95
   - branch latency
   - branch candidates
   - exact MaxSim pairs
   - memory/sidecar bytes
3. Add Markdown comparison output:
   - best quality
   - best latency under quality floor
   - candidate admission failures
   - recommended default GUC profile
4. Add no external-data CI dependency.

Acceptance:
- The harness can justify default recommendations for latency, balanced, quality, and high_recall profiles.
- 10k DBpedia evidence is sufficient for this prompt's acceptance gate; 100k
  and 1M runs are optional scale checks, not required completion evidence.
```

---

## Suggested execution order

Start with correctness and evidence:

```text
1 -> 2 -> 3 -> 4
```

Then make document-node scalable:

```text
6 -> 7 -> 10
```

Then improve hybrid recall:

```text
8 -> 9 -> 15
```

Then broaden model coverage:

```text
5 -> 11 -> 12
```

Then explore research branches:

```text
13 -> 14
```

The single most valuable next PR is probably **Prompt 2 plus Prompt 15**: a DBpedia document-node admission and hybrid benchmark. Without that, it is hard to know whether `f16`, `sq8`, `proxy_vector`, RRF, DBSF, and BM25 injection are actually improving the real failure case rather than only passing the synthetic top-1 gate.

[1]: https://qdrant.tech/documentation/concepts/vectors/ "Vectors - Qdrant"
[2]: https://qdrant.tech/documentation/fastembed/fastembed-colbert/ "Working with ColBERT - Qdrant"
[3]: https://qdrant.tech/documentation/concepts/hybrid-queries/ "Hybrid Queries - Qdrant"
[4]: https://blog.vespa.ai/announcing-long-context-colbert-in-vespa/ "Announcing Vespa Long-Context ColBERT | Vespa Blog"
[5]: https://milvus.io/docs/multi-vector-search.md "Multi-Vector Hybrid Search | Milvus Documentation"
[6]: https://milvus.io/docs/reranking.md "Reranking | Milvus Documentation"
[7]: https://arxiv.org/abs/2205.09707?utm_source=chatgpt.com "PLAID: An Efficient Engine for Late Interaction Retrieval"
[8]: https://arxiv.org/abs/2404.14989?utm_source=chatgpt.com "A Reproducibility Study of PLAID"
[9]: https://arxiv.org/abs/2404.13950?utm_source=chatgpt.com "SPLATE: Sparse Late Interaction Retrieval"
[10]: https://arxiv.org/abs/2302.06587?utm_source=chatgpt.com "SLIM: Sparsified Late Interaction for Multi-Vector Retrieval with Inverted Indexes"
[11]: https://arxiv.org/abs/2405.19504?utm_source=chatgpt.com "MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings"
[12]: https://arxiv.org/abs/2601.21853?utm_source=chatgpt.com "LEMUR: Learned Multi-Vector Retrieval"
[13]: https://arxiv.org/abs/2501.17788?utm_source=chatgpt.com "WARP: An Efficient Engine for Multi-Vector Retrieval"
[14]: https://arxiv.org/abs/2602.02827?utm_source=chatgpt.com "Col-Bandit: Zero-Shot Query-Time Pruning for Late-Interaction Retrieval"
[15]: https://arxiv.org/abs/2409.14683?utm_source=chatgpt.com "Reducing the Footprint of Multi-Vector Retrieval with Minimal Performance Impact via Token Pooling"
[16]: https://arxiv.org/abs/2606.05568?utm_source=chatgpt.com "ColBERTSaR: Sparsified ColBERT Index via Product Quantization"
[17]: https://arxiv.org/abs/2408.16672?utm_source=chatgpt.com "Jina-ColBERT-v2: A General-Purpose Multilingual Late Interaction Retriever"
[18]: https://arxiv.org/abs/2508.03555?utm_source=chatgpt.com "PyLate: Flexible Training and Retrieval for Late Interaction Models"
