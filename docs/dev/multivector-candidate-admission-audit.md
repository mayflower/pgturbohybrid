# Multivector Candidate Admission Audit

This audit maps the current multivector candidate-generation path before
changing runtime behavior. The current path is a token-node candidate generator:
one graph node represents one document subvector/token, and result admission is
reconstructed into document candidates after per-query-token graph searches.

## File And Function Map

### Build-Time Token-Node Layout

- [src/pgturbohybrid_multivector.h](/Volumes/CrucialMusic/src/pgturbohybrid/src/pgturbohybrid_multivector.h:24)
  defines the multivector identity contract:
  - `nodeId` is one document token/subvector.
  - `docId` is the document-level identity used for result aggregation.
  - `heaptid` is the heap tuple resolved from the document map.
  - `tokenOrdinal` is the subvector position inside the document.
- [src/pgturbohybrid_quant.c](/Volumes/CrucialMusic/src/pgturbohybrid/src/pgturbohybrid_quant.c:1281)
  `PgturbohybridGraphIndexIsMultiVector()` detects multivector indexes from the
  dense key type.
- [src/pgturbohybrid_quant.c](/Volumes/CrucialMusic/src/pgturbohybrid/src/pgturbohybrid_quant.c:1311)
  `PgturbohybridGraphAppendBuildMultiVector()` expands each heap tuple into one
  graph node per document subvector. It assigns one `docId`, stores the heap TID
  in `TqMultiVectorDocMapEntry`, and writes `nodeId -> (docId, tokenOrdinal)` in
  `TqMultiVectorNodeMapEntry`.
- [src/pgturbohybrid_quant.h](/Volumes/CrucialMusic/src/pgturbohybrid/src/pgturbohybrid_quant.h:213)
  records build-state sidecar arrays for the node map and document map.
- [src/pgturbohybrid_quant.h](/Volumes/CrucialMusic/src/pgturbohybrid/src/pgturbohybrid_quant.h:460)
  records scan-time loaded docmap state.

### Query-Time Candidate Path

- [src/pgturbohybrid_quant.c](/Volumes/CrucialMusic/src/pgturbohybrid/src/pgturbohybrid_quant.c:9290)
  `PgturbohybridGraphCollectMultiVectorDenseCandidates()` is the main
  multivector dense candidate collection function.
- [src/pgturbohybrid_quant.c](/Volumes/CrucialMusic/src/pgturbohybrid/src/pgturbohybrid_quant.c:9396)
  computes the per-token raw-hit target and adaptive widening bounds from the
  multivector GUCs, `targetK`, and index node count.
- [src/pgturbohybrid_quant.c](/Volumes/CrucialMusic/src/pgturbohybrid/src/pgturbohybrid_quant.c:9433)
  estimates and checks accumulator capacity before allocating the document hash
  and slab arrays.
- [src/pgturbohybrid_quant.c](/Volumes/CrucialMusic/src/pgturbohybrid/src/pgturbohybrid_quant.c:9474)
  chooses the persistent docmap sidecar or a heap-TID hash fallback for resolving
  token-node hits into document IDs.
- [src/pgturbohybrid_quant.c](/Volumes/CrucialMusic/src/pgturbohybrid/src/pgturbohybrid_quant.c:9509)
  iterates query subvectors.
- [src/pgturbohybrid_quant.c](/Volumes/CrucialMusic/src/pgturbohybrid/src/pgturbohybrid_quant.c:9546)
  calls `PgturbohybridGraphRunTraversalPass()` once for the current query
  subvector/token.
- [src/pgturbohybrid_quant.c](/Volumes/CrucialMusic/src/pgturbohybrid/src/pgturbohybrid_quant.c:9434)
  `PgturbohybridMultiVectorExactTokenScan()` is the developer oracle for
  `turbohybrid.multivector_candidate_source = 'exact_token_scan'`. It scans all
  stored graph token nodes for each query token, batch-scores them through the
  existing quantized/exact graph scorer, retains the same bounded top raw hits,
  and then feeds the normal document accumulator.
- [src/pgturbohybrid_quant.c](/Volumes/CrucialMusic/src/pgturbohybrid/src/pgturbohybrid_quant.c:9561)
  resolves `hits[i].nodeId` through the sidecar node map when available; the
  heap-TID hash fallback assigns a scan-local `docId` for older indexes.
- [src/pgturbohybrid_quant.c](/Volumes/CrucialMusic/src/pgturbohybrid/src/pgturbohybrid_quant.c:9594)
  scans token hits, counts raw subvector hits, resolves doc IDs and heap TIDs,
  deduplicates per query token, and stops the token after
  `multivector_unique_docs_per_token` unique documents.
- [src/pgturbohybrid_quant.c](/Volumes/CrucialMusic/src/pgturbohybrid/src/pgturbohybrid_quant.c:9233)
  `PgturbohybridMultiVectorAccumulateDoc()` updates the document-level
  approximate MaxSim accumulator.
- [src/pgturbohybrid_quant.c](/Volumes/CrucialMusic/src/pgturbohybrid/src/pgturbohybrid_quant.c:9658)
  applies the final document candidate cap before exact rerank.
- [src/pgturbohybrid_quant.c](/Volumes/CrucialMusic/src/pgturbohybrid/src/pgturbohybrid_quant.c:9680)
  exact-reranks the retained prefix with heap float32 MaxSim.
- [src/pgturbohybrid_quant.c](/Volumes/CrucialMusic/src/pgturbohybrid/src/pgturbohybrid_quant.c:8741)
  `PgturbohybridMultiVectorExactHeapRerank()` fetches visible heap tuples and
  computes `distance = -TqMultiVectorMaxSim(query, doc)`.

### GUC And Stats Plumbing

- [src/pgturbohybrid_am.c](/Volumes/CrucialMusic/src/pgturbohybrid/src/pgturbohybrid_am.c:135)
  defines current multivector GUC defaults.
- [src/pgturbohybrid_am.c](/Volumes/CrucialMusic/src/pgturbohybrid/src/pgturbohybrid_am.c:5419)
  registers the SQL-visible GUCs.
- [src/pgturbohybrid_quant.h](/Volumes/CrucialMusic/src/pgturbohybrid/src/pgturbohybrid_quant.h:334)
  defines `TqDenseCandidateStats`, including multivector fields.
- [src/pgturbohybrid_stats.c](/Volumes/CrucialMusic/src/pgturbohybrid/src/pgturbohybrid_stats.c:2834)
  serializes multivector scan stats into `turbohybrid_last_scan_stats()`.

## Current Control Flow

1. The query path extracts the `PgturbohybridMultiVector` query from
   `turbohybrid_query`.
2. The scan validates query dimensions and query-token count against
   `turbohybrid.multivector_max_dim` and
   `turbohybrid.multivector_max_query_vectors`.
3. The graph metapage is read. For a multivector index, `tqNodeCount` is token
   node count, not document count.
4. `turbohybrid.multivector_plain_fallback` is evaluated. `force` always
   chooses an exact heap MaxSim scan. `auto` chooses it for small estimated
   document counts or when the requested/effective document-candidate or exact
   rerank budget is near exhaustive. When selected, the scan heap-fetches
   visible tuples through PostgreSQL table AM, scores exact float32 MaxSim, and
   returns the top document candidates without entering the token-node path.
5. If fallback is not selected, the code computes `initialRawTarget` and
   `maxRawTarget`:
   - adaptive mode starts at `multivector_subvector_k` and can widen toward the
     larger of `multivector_unique_docs_per_token`, `targetK`, and the raw-hit
     cap;
   - non-adaptive mode starts at the max of `multivector_subvector_k`,
     `multivector_unique_docs_per_token`, and `targetK`;
   - both modes clamp to `multivector_max_raw_hits_per_token` and index node
     count.
6. The accumulator capacity is estimated as
   `query_vectors * min(maxRawTarget, multivector_unique_docs_per_token)`, plus
   hash and slab overhead. The scan errors if the estimate exceeds
   `multivector_max_accumulator_mb`.
7. The docmap sidecar is loaded when available. Otherwise the scan creates a
   heap-TID hash so token hits from the same heap tuple get one scan-local
   `docId`.
8. For each query subvector `qi`:
   - the query subvector is copied into a single-vector `vector`;
   - the TurboQuant query is prepared;
   - in the default `graph` candidate source,
     `PgturbohybridGraphRunTraversalPass()` runs an approximate graph search
     over token nodes and returns up to `tokenRawTarget` token-node hits;
   - in `exact_token_scan` developer mode,
     `PgturbohybridMultiVectorExactTokenScan()` scores every live token node
     against the query token and keeps the top raw hits under the same
     raw-hit/unique-doc caps;
   - if adaptive widening is enabled in graph mode and the token yielded too
     few unique documents, the token search widens and repeats until the
     unique-doc target or maximum raw target is reached.
9. The final hit list for that token is processed in graph-hit order:
   - each hit is resolved to `docId`;
   - duplicates for the same token/doc are counted;
   - `PgturbohybridMultiVectorAccumulateDoc()` keeps one max similarity per
     query token per document and sums those maxima as approximate MaxSim;
   - token processing stops after
     `multivector_unique_docs_per_token` unique documents.
10. After all query tokens, the document hash is reduced to a bounded candidate
   array. `docLimit = min(targetK or multivector_doc_candidate_k,
   multivector_doc_candidate_k)`. With
   `turbohybrid.multivector_candidate_reservoirs = 'off'`, only the best
   approximate-score documents are retained. With reservoirs enabled, the
   bounded candidate set is a document-keyed union of approximate-score,
   coverage, mean seen-similarity, and per-query-token document reservoirs.
11. If exact rerank is enabled, a bounded prefix is heap-fetched and exact
    float32 MaxSim is recomputed. The limit is
    `min(docCount, multivector_doc_candidate_k, multivector_exact_rerank_k)`.
12. The candidate array is sorted again after exact rerank and returned as
    document/heap candidates. `nodeId` remains only the best contributing token
    node recorded on the candidate, not the SQL result identity.

## Current GUCs

| GUC | Default | Range / values | Admission effect |
| --- | ---: | --- | --- |
| `turbohybrid.multivector_max_doc_vectors` | `PGTURBOHYBRID_MULTIVECTOR_DEFAULT_MAX_DOC_VECTORS` | `1..PGTURBOHYBRID_MAX_MULTIVECTOR_DOC_VECTORS` | Build/query validation cap for document token count. |
| `turbohybrid.multivector_max_query_vectors` | `PGTURBOHYBRID_MULTIVECTOR_DEFAULT_MAX_QUERY_VECTORS` | `1..PGTURBOHYBRID_MAX_MULTIVECTOR_QUERY_VECTORS` | Query validation cap and accumulator multiplier. |
| `turbohybrid.multivector_max_dim` | `PGTURBOHYBRID_MULTIVECTOR_MAX_DIM` | `1..PGTURBOHYBRID_MULTIVECTOR_MAX_DIM` | Query/doc subvector dimension cap. |
| `turbohybrid.multivector_subvector_k` | `100` | `1..100000` | Initial per-query-token ANN hit count in adaptive mode. |
| `turbohybrid.multivector_unique_docs_per_token` | `100` | `1..100000` | Stops each query token after this many unique documents. |
| `turbohybrid.multivector_max_raw_hits_per_token` | `400` | `1..100000` | Hard cap for adaptive raw token hits. |
| `turbohybrid.multivector_adaptive_widening` | `auto` | `off`, `auto`, `on` | Allows repeated per-token searches with larger raw targets when unique docs underfill. |
| `turbohybrid.multivector_candidate_source` | `graph` | `graph`, `document_nodes`, `exact_token_scan`, `exact_doc_scan`, `doc_graph_prototype`, `proxy_vector` | Developer candidate-source switch. `graph` uses the index graph mode; `document_nodes` requires a document-node index; `exact_token_scan` scores all token nodes per query token before normal document aggregation; `exact_doc_scan` heap-scans exact document MaxSim; `doc_graph_prototype` is the heap-backed Prompt 10 document-graph validation hook; `proxy_vector` uses a document-node representative-vector graph for admission and exact MaxSim rerank. |
| `turbohybrid.multivector_plain_fallback` | `auto` | `auto`, `off`, `force` | Exact heap MaxSim fallback for small or near-exhaustive multivector scans. |
| `turbohybrid.multivector_plain_fallback_max_docs` | `1000` | `0..10000000` | Auto fallback threshold for estimated live documents. |
| `turbohybrid.multivector_plain_fallback_candidate_fraction` | `0.5` | `0.0..1.0` | Auto fallback threshold when candidate/rerank budgets exceed this fraction of estimated documents. |
| `turbohybrid.multivector_candidate_reservoirs` | `conservative` | `off`, `conservative`, `balanced` | Selects single score-top-K truncation or bounded multi-reservoir document candidate retention before exact rerank. |
| `turbohybrid.multivector_per_token_doc_reservoir_k` | `1` | `0..100000` | Per-query-token document reservoir width when reservoirs are enabled. |
| `turbohybrid.multivector_coverage_reservoir_k` | `10` | `0..100000` | Width for coverage and mean seen-similarity reservoirs. |
| `turbohybrid.multivector_bm25_candidate_injection` | `off` | `off`, `hybrid_only`, `dense_with_text` | Injects lexical BM25 candidates into exact MaxSim rerank for hybrid or text-backed dense runs. |
| `turbohybrid.multivector_docmap` | `auto` | `off`, `auto`, `require` | Chooses persistent node/doc sidecar or heap-TID hash fallback. |
| `turbohybrid.multivector_doc_candidate_k` | `100` | `1..100000` | Caps document candidates before exact rerank. |
| `turbohybrid.multivector_exact_rerank` | `topk` | `off`, `topk` | Enables heap float32 MaxSim rerank of retained documents. |
| `turbohybrid.multivector_exact_rerank_k` | `100` | `1..100000` | Caps exact heap-reranked document prefix. |
| `turbohybrid.multivector_max_accumulator_mb` | `64` | `1..PGTURBOHYBRID_MAX_MULTIVECTOR_ACCUMULATOR_MB` | Rejects scans whose document accumulator estimate is too large. |
| `turbohybrid.multivector_debug_admission` | `off` | `off`, `summary`, `trace` | Enables debug-only admission summary counters or bounded per-document trace. |
| `turbohybrid.multivector_debug_trace_limit` | `1000` | `0..1000` | Hard cap for trace entries emitted in `trace` mode. |
| `turbohybrid.multivector_debug_skip_query_tokens` | empty string | comma-separated token ordinals | Developer-only token ablation hook that skips selected query tokens during candidate generation while exact rerank still sees the full query. |

## Current Index Options

| Option | Default | Values | Admission effect |
| --- | --- | --- | --- |
| `multivector_graph` | `token_nodes` | `token_nodes`, `document_nodes` | `token_nodes` is the compatible token/subvector-node graph. `document_nodes` is explicit opt-in and stores one graph node per heap document plus a versioned float32 multivector sidecar; build edge selection uses symmetrized document MaxSim, non-exhaustive scans traverse document graph adjacency with f32/f16/sq8 sidecar scoring, and near-exhaustive scans use the exact sidecar scan. |

## Current Scan Stats

`turbohybrid_last_scan_stats()` currently exposes these multivector fields:

| Field | Source | Meaning |
| --- | --- | --- |
| `multivector_enabled` | candidate collection | Whether the multivector path ran. |
| `multivector_query_vectors` | query multivector | Query subvector count. |
| `multivector_doc_vectors_limit` | GUC | Configured max document vectors. |
| `multivector_subvector_searches` | query multivector | Number of non-skipped per-token searches in token candidate modes; `0` for exact/plain document fallback paths. |
| `multivector_raw_subvector_hits` | token-hit loop | Raw token-node hits processed after final widening. |
| `multivector_adaptive_widening_triggered` | token loop | Whether any query token widened beyond the initial target. |
| `multivector_adaptive_initial_raw_target` | target setup | Initial raw token target after clamping. |
| `multivector_adaptive_final_raw_target` | token loop | Maximum final raw target used by any token. |
| `multivector_candidate_source` | candidate source path | `graph`, `document_nodes`, `exact_token_scan`, `exact_doc_scan`, `doc_graph_prototype`, `proxy_vector`, or `plain_fallback` for this scan. |
| `multivector_exact_token_scan_enabled` | candidate source GUC | Whether the exact-token oracle path was active. |
| `multivector_exact_token_scan_nodes_scored` | exact-token oracle | Count of live token nodes scored across all query tokens. |
| `multivector_plain_fallback_used` | fallback decision | Whether exact heap MaxSim fallback bypassed token candidate generation. |
| `multivector_plain_fallback_reason` | fallback decision | `force`, `small_estimated_docs`, `doc_candidate_fraction`, `exact_rerank_fraction`, `exact_doc_scan`, `doc_graph_prototype_heap_scan`, or `not_applicable`. |
| `multivector_plain_fallback_docs_scored` | fallback heap scan | Visible heap multivector documents scored exactly. |
| `multivector_plain_fallback_pairs` | fallback heap scan | Query-token by document-token comparisons used by exact MaxSim fallback. |
| `multivector_doc_graph_prototype_enabled` | doc graph prototype | Whether the Prompt 10 document-level prototype source was active. |
| `multivector_doc_graph_nodes` | document graph | Document graph nodes available for `document_nodes`; `0` for token-node scans. |
| `multivector_doc_graph_docs_scored` | document graph | Documents scored by approximate/full document MaxSim; exact heap MaxSim in prototype mode, exact float32 sidecar MaxSim for `f32`, or compact sidecar MaxSim for `f16`/`sq8`. |
| `multivector_doc_graph_edges_visited` | document graph | Graph edges traversed; `0` for the heap-backed prototype, real edge visits for document-node traversal, or the scored-document count for near-exhaustive sidecar scans. |
| `multivector_doc_graph_candidates` | document graph | Candidate documents retained by the prototype or `document_nodes` source. |
| `multivector_doc_graph_quantized_scores` | document graph | Approximate compact-code document scores. `f16` and `sq8` document-node scans increment this; `f32` scans report `0`. |
| `multivector_doc_graph_heap_fetches` | document graph | Heap document fetches used by prototype/fallback paths or exact rerank. |
| `multivector_doc_graph_exact_rerank_docs` | document graph | Document graph candidates reranked against heap float32 multivectors. |
| `multivector_doc_graph_warning` | document graph | `prototype_heap_scan_no_index_resident_doc_graph` for the heap-backed prototype, `document_node_f32_sidecar_graph_traversal`, `document_node_f16_sidecar_graph_traversal`, `document_node_sq8_sidecar_graph_traversal`, `document_node_proxy_vector_graph_traversal`, or `document_node_f32_sidecar_exact_scan` when the document-node budget covers the sidecar. |
| `multivector_reservoirs_enabled` | candidate reduction | Whether the bounded multi-reservoir candidate selector replaced score-only truncation. |
| `multivector_reservoir_score_docs` | candidate reduction | Documents retained from the approximate MaxSim-sum reservoir. |
| `multivector_reservoir_coverage_docs` | candidate reduction | Documents retained from the query-token coverage reservoir. |
| `multivector_reservoir_mean_docs` | candidate reduction | Documents retained from the mean seen-similarity reservoir. |
| `multivector_reservoir_per_token_docs` | candidate reduction | Documents retained from the combined per-query-token document reservoirs. |
| `multivector_reservoir_bm25_docs` | candidate reduction | BM25-injected documents retained in the reservoir union when BM25 candidate injection is active. |
| `multivector_reservoir_union_docs` | candidate reduction | Final document-keyed union size before exact rerank. |
| `multivector_reservoir_duplicates` | candidate reduction | Documents proposed by multiple reservoirs and deduplicated by doc ID. |
| `multivector_docmap_source` | docmap setup | `sidecar`, `heap_tid_hash`, or `none`. |
| `multivector_docmap_bytes` | docmap setup | Loaded sidecar bytes. |
| `multivector_unique_docs` | token-hit loop | Sum of token-local unique document hits retained. |
| `multivector_duplicate_doc_hits` | token-hit loop | Token hits whose document was already seen for that query token. |
| `multivector_maxsim_updates` | accumulator | Count of query-token max updates across documents. |
| `multivector_doc_candidates` | candidate truncation | Retained document candidates after `doc_candidate_k` truncation. |
| `multivector_exact_rerank_enabled` | GUC | Whether exact rerank is configured on. |
| `multivector_exact_rerank_docs` | exact rerank | Number of candidates actually exact-reranked. |
| `multivector_exact_rerank_pairs` | exact rerank | Query-token by document-token comparisons used by exact MaxSim. |
| `multivector_exact_kernel` | exact rerank | MaxSim kernel name when exact rerank ran. |
| `multivector_accumulator_kind` | accumulator | Currently `docid_hash_slab`. |
| `multivector_memory_bytes_estimate` | accumulator setup | Estimated accumulator memory for this scan. |
| `multivector_admission_debug_enabled` | debug GUC | Whether `summary` or `trace` mode was active for this scan. |
| `multivector_admission_candidates_before_rerank` | document accumulator | Accumulated document count before `doc_candidate_k` truncation. |
| `multivector_admission_candidates_after_truncation` | candidate truncation | Document candidates retained for possible exact rerank. |
| `multivector_admission_exact_rerank_docs` | exact rerank | Number of retained candidates exact-reranked. |
| `multivector_admission_truncated_by_doc_candidate_k` | candidate truncation | Whether accumulated documents exceeded retained candidates. |
| `multivector_admission_truncated_by_accumulator_memory` | accumulator setup | Currently false because memory overflow errors instead of truncating. |
| `multivector_admission_trace_available` | debug trace | Whether a bounded per-document trace payload is present. |

In `turbohybrid.multivector_debug_admission = 'trace'` mode,
`turbohybrid_last_scan_stats()` also includes
`multivector_admission_trace`, bounded by
`turbohybrid.multivector_debug_trace_limit`. Entries are document-keyed and
include doc ID / heap TID, approximate score before exact rerank, token coverage,
raw and duplicate hit counts, pre-truncation candidate rank, exact-rerank
retention, and exact rerank score when available.

The generic graph stats are also relevant while diagnosing admission:
`graph_effective_result_target`, `graph_effective_search_ef`,
`graph_visited_nodes`, `graph_scored_codes`, `graph_candidate_count`,
`heap_rescore_count`, and heap fetch/rescore timing fields. They describe token
graph traversal and heap rerank cost, but they do not currently identify exact
top-K admission.

## Where Exact Top-K Can Be Dropped

### ANN / Token Miss

The first loss point is the per-query-token graph search. A document can be
exact top-K by full MaxSim and still have none of its token nodes returned by
`PgturbohybridGraphRunTraversalPass()` for any query token. In that case it is
never resolved to a `docId`, never enters the accumulator, and cannot be exact
reranked.

This miss can be caused by graph traversal recall, quantized token scoring, low
`ef_search`, low `multivector_subvector_k`, or
`multivector_max_raw_hits_per_token`.

Use `turbohybrid.multivector_candidate_source = 'exact_token_scan'` to isolate
this class of miss. If exact-token mode admits a document that graph mode misses
under the same raw-hit and document caps, the loss is in graph/quantized token
candidate generation. If exact-token mode still misses it, the loss is
structural to token-top-K admission or later truncation.

Use `turbohybrid.multivector_plain_fallback = 'force'` as the exact document
oracle inside the index scan path. It heap-scans MVCC-visible multivectors,
scores full MaxSim, and returns the same top document candidates as
`turbohybrid_multivector_maxsim_distance` for the retained `LIMIT`/`final_k`.
`auto` selects the same path for small or near-exhaustive cases; `off` keeps the
token candidate path available for diagnostics.

### Document Aggregation Miss

The second loss point is token-local unique-document stopping. The hit loop
processes raw token hits in returned order and breaks after
`multivector_unique_docs_per_token` unique documents for that query token. A
document whose token node is present later in the raw hit list can be skipped
before it contributes to the accumulator.

This is distinct from an ANN miss: the token node may have been returned by the
graph, but not admitted into the document accumulator.

### Doc-Candidate Truncation Miss

The third loss point is document-candidate reduction. Every accumulated document
is converted to a candidate with `distance = -entry->score`, but only
`docLimit` documents survive. `docLimit` is bounded by
`multivector_doc_candidate_k` and the dense branch target. A true exact top-K
document that is accumulated but has weak pre-rerank evidence can be dropped
here.

This is the structural ColBERT failure mode: a document can be globally strong
because many query tokens match moderately, while spiky documents occupy the
token-level and approximate-score reservoirs. The multi-reservoir selector
mitigates this by preserving a bounded document-keyed union from score,
coverage, mean seen-similarity, and per-token reservoirs. It still cannot admit
a document that never reached the accumulator, and it cannot infer hidden full
MaxSim for documents that are indistinguishable before heap rerank.

### Exact-Rerank Miss

The fourth loss point is the exact-rerank prefix. Exact rerank only runs over
`min(docCount, multivector_doc_candidate_k, multivector_exact_rerank_k)`
documents. If a true exact top-K document survived document-candidate truncation
but sits beyond `multivector_exact_rerank_k` in approximate order, it remains
approximately scored and may still rank incorrectly.

If `multivector_exact_rerank = off`, every retained document keeps approximate
ordering and exact top-K correctness is not guaranteed even after admission.

### Accumulator Memory Rejection

The accumulator memory check does not silently drop documents. It raises an
error when the estimated hash/slab state exceeds
`multivector_max_accumulator_mb`. For benchmark diagnostics, this still matters:
candidate budget sweeps can fail at larger budgets unless the memory cap is
raised.

## Low-Risk Instrumentation Points

These locations can collect admission diagnostics with minimal behavioral risk:

1. After target setup and accumulator memory check, record effective
   `initialRawTarget`, `maxRawTarget`, estimated document capacity, and memory
   estimate. Most of these already exist as aggregate stats.
2. Inside the adaptive widening loop, record per-query-token raw target,
   search EF, hit count, and unique-doc count before deciding whether to widen.
3. In the final token-hit loop, record document-keyed trace entries from
   `docId`/heap TID, approximate score, query-token coverage, raw hit count, and
   duplicate hit count. This is the best place to distinguish token returned vs
   token retained.
4. Before `PgturbohybridMultiVectorCandidateHeapOffer()`, count total
   accumulated documents. After heap reduction, compare retained candidates with
   the pre-truncation count to detect `doc_candidate_k` truncation.
5. Immediately before and after `PgturbohybridMultiVectorExactHeapRerank()`,
   mark which retained document-keyed candidates were exact-reranked and record
   exact score where available.
6. In benchmark-only code, compare exact scan top-K heap TIDs against the
   document-keyed trace or retained candidate set to classify each miss as:
   ANN/token miss, document aggregation miss, doc-candidate truncation miss, or
   exact-rerank miss.

Trace payloads must remain bounded and should be disabled by default. Any trace
should be document-keyed by heap TID or `docId`; token `nodeId` should only be
stored as evidence for the best contributing token, never as result identity.

## Known Tradeoffs

- Token-node graph search is not MaxSim-aligned at document admission time. It
  can miss many-moderate-match documents that win only after summing MaxSim
  over multiple query tokens.
- Exact/plain fallback is the safest current path for small or near-exhaustive
  multivector scans because it scores MVCC-visible documents with full float32
  MaxSim before returning candidates.
- Exact token scan isolates graph/quantization recall from structural token
  top-K admission loss, but it remains a debug oracle and still feeds the same
  lossy document accumulator and truncation stages.
- Multi-reservoir retention and BM25 candidate injection can improve candidate
  retention after a document reaches the accumulator or lexical branch. They
  cannot recover documents that no candidate source admitted.
- Document-level graph storage is the intended MaxSim-aligned candidate
  generation direction. `document_nodes` now provides the first production
  slice with document-keyed graph nodes and exact float32 sidecar MaxSim
  candidate scoring; `doc_graph_prototype` remains heap-backed validation for
  benchmark comparison, and compact quantized traversal remains future work.
- Exact heap rerank remains the semantic final scorer; all approximate paths
  should be judged by whether they admit the documents that exact MaxSim would
  rank highly.
