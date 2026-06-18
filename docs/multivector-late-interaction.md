# Multivector Late Interaction

`pgturbohybrid` includes an experimental `turbohybrid_multivector` type for
ColBERT-style late-interaction retrieval. A row stores multiple same-dimensional
token vectors, the TurboHybrid graph indexes each token vector as a subnode, and
SQL results are still PostgreSQL heap rows/documents.

The key invariant is:

- graph `nodeId` = one subvector/token
- SQL result = document heap tuple
- document score = MaxSim over all query/document token vectors

## Build Values

Construct a multivector from a `vector[]` array. Every element must be non-null,
finite, and have the same dimension:

```sql
SELECT turbohybrid_multivector(ARRAY[
  '[1,0,0]'::vector,
  '[0,1,0]'::vector
]);

SELECT turbohybrid_multivector_dims(colbert),
       turbohybrid_multivector_count(colbert)
FROM passages;
```

Embedding extensions that already have flat row-major float data can avoid
building intermediate pgvector values:

```sql
SELECT turbohybrid_multivector_from_float4(
  ARRAY[1,0,0,1]::real[],
  2
);
```

`turbohybrid_multivector_from_float4(raw_values, dim)` requires `dim > 0`, a
non-empty `real[]`, an array length divisible by `dim`, no null elements, and
finite values. It stores values exactly in the same multivector layout used by
the `vector[]` constructor.

Long documents can stay in one SQL row while carrying context/window metadata:

```sql
SELECT turbohybrid_multivector_from_contexts(
  ARRAY[1,0, 0,1, 0.2,0.2, 0.1,0.1]::real[],
  2,
  ARRAY[0,2]::int[]
);
```

`context_offsets` are zero-based document-token start offsets. They must start
with `0`, be strictly increasing, and point inside the stored token sequence.
The last context runs to the end of the multivector. For section-aware scoring,
store one non-negative field id per context:

```sql
SELECT turbohybrid_multivector_from_contexts_and_fields(
  ARRAY[1,0, 0,1, 0.2,0.2, 0.1,0.1]::real[],
  2,
  ARRAY[0,2]::int[],
  ARRAY[1,2]::int[] -- title/body, for example
);
```

Metadata accessors are available for validation and debugging:
`turbohybrid_multivector_context_count(mv)`,
`turbohybrid_multivector_context_offsets(mv)`, and
`turbohybrid_multivector_field_ids(mv)`.

Exact reference scoring is available without an index:

```sql
SELECT turbohybrid_multivector_maxsim($query_mv, $doc_mv) AS score,
       turbohybrid_multivector_maxsim_distance($query_mv, $doc_mv) AS distance;
```

Flat MaxSim is the default and treats all document tokens as one global token
set. Context-level MaxSim scores each stored context independently and returns
the best context score:

```sql
SELECT turbohybrid_multivector_context_maxsim($query_mv, $doc_mv);
```

Field-weighted MaxSim is explicit because weights are query/workload policy:

```sql
SELECT turbohybrid_multivector_field_weighted_maxsim(
  $query_mv,
  $doc_mv,
  ARRAY[1,2]::int[],
  ARRAY[2.0,1.0]::real[]
);
```

MaxSim score is larger-is-better. SQL index ordering remains distance-oriented,
so the exposed distance is:

```text
distance = -maxsim(query, document)
```

## Dense-Only Index

Load data first, then build the index:

```sql
CREATE TABLE passages (
  id bigint PRIMARY KEY,
  colbert turbohybrid_multivector NOT NULL
);

INSERT INTO passages VALUES
  (1, turbohybrid_multivector(ARRAY[
    '[1,0,0]'::vector,
    '[0,1,0]'::vector
  ])),
  (2, turbohybrid_multivector(ARRAY[
    '[0,0,1]'::vector,
    '[0.7,0.3,0]'::vector
  ]));

CREATE INDEX passages_colbert_idx
ON passages USING turbohybrid (
  colbert multivector_maxsim_ip_turbohybrid_ops
);
```

For new indexes, prefer `multivector_maxsim_ip_turbohybrid_ops`. It names the
actual raw dot-product MaxSim semantics. The older
`multivector_cosine_turbohybrid_ops` remains available for compatibility and
assumes stored and query token vectors are already normalized before indexing
or querying.

Query with a multivector payload:

```sql
SELECT id
FROM passages
ORDER BY colbert <~> turbohybrid_query(
  multivector_query => turbohybrid_multivector(ARRAY[
    '[1,0,0]'::vector,
    '[0,1,0]'::vector
  ]),
  dense_k => 100,
  final_k => 10
)
LIMIT 10;
```

Model/query-token importance can be supplied without changing the stored
document multivectors:

```sql
ORDER BY colbert <~> turbohybrid_query(
  multivector_query => $1,
  query_token_weights => ARRAY[1.0, 0.4, 2.0]::real[],
  query_token_mask => ARRAY[false, true, false]
)
```

`query_token_weights` and `query_token_mask` must match the query multivector
token count. A `true` mask entry removes that query token from token-node
candidate generation and exact scoring. The weighted score is:

```text
score(Q, D) = sum_i weight_i * max_j dot(q_i, d_j)
```

If weights are omitted, every unmasked query token has weight `1.0`. The
adaptive exact reranker orders tokens by weighted query-token norm, and
document-node sidecar scoring uses the same weighted MaxSim as heap rerank.
Model-specific special-token masking, punctuation masking, and IDF weighting
should be applied by the query/model metadata layer that builds these arrays;
the scorer accepts the arrays and does not hardcode one ColBERT variant.

The graph returns subvector hits internally, but the access method aggregates
them by heap tuple/document. A document with many matching token vectors should
still appear only once in the result list.

## Hybrid With BM25

Multivector indexes can include the normal BM25 `tsvector` key:

```sql
CREATE TABLE hybrid_passages (
  id bigint PRIMARY KEY,
  colbert turbohybrid_multivector NOT NULL,
  body text NOT NULL,
  body_tsv tsvector NOT NULL
);

CREATE INDEX hybrid_passages_idx
ON hybrid_passages USING turbohybrid (
  colbert multivector_maxsim_ip_turbohybrid_ops,
  body_tsv bm25_tsvector_turbohybrid_ops
);
```

Use RRF fusion for the supported hybrid multivector path:

```sql
SELECT id
FROM hybrid_passages
ORDER BY colbert <~> turbohybrid_query(
  multivector_query => $1,
  text_query => websearch_to_tsquery('english', $2),
  fusion => 'rrf',
  dense_k => 100,
  bm25_k => 100,
  final_k => 10
)
LIMIT 10;
```

Fusion is document-level: dense MaxSim candidates are keyed by document/heap
tuple, BM25 candidates are keyed by heap tuple, and RRF combines document ranks.
Weighted score-level fusion is also document-keyed and combines normalized
MaxSim (`maxsim / query_vector_count`) with the existing normalized BM25 score.
`fast_weighted` and `calibrated` are supported only as normalized score-fusion
modes: dense MaxSim uses logistic normalization and BM25 uses saturating
normalization. Raw BM25 plus raw MaxSim alpha fusion is intentionally rejected.

With `turbohybrid.hybrid_budget_policy = 'adaptive'`, multivector hybrid scans
use dense admission stats after the multivector branch has run to refine the
defaulted BM25 branch budget before fetching BM25 candidates. Document-level
sources such as `document_nodes`, `proxy_vector`, `exact_doc_scan`, and
`doc_graph_prototype` can reduce a defaulted BM25 budget when dense admission is
not truncated and enough documents were exact-reranked. Token-node admission
truncation or dense underfill keeps the BM25 branch wide. The decision is
reported in `hybrid_budget_reason`, for example
`admission_document_dense_reduce_bm25`,
`admission_exact_dense_reduce_bm25`, or
`admission_truncated_keep_bm25`.

## ColBERT Reranking For Vector + BM25 Hybrid

The native multivector hybrid path above uses ColBERT/MaxSim as the dense
candidate branch. A different and often useful shape is:

1. run the existing single-vector + BM25 hybrid index scan;
2. keep a bounded candidate window;
3. exact-rerank only those heap rows with ColBERT MaxSim.

Use a materialized candidate CTE for that shape:

```sql
WITH q AS (
  SELECT
    turbohybrid_query(
      vector_query => $1::vector,
      text_query => websearch_to_tsquery('english', $2),
      dense_k => 200,
      bm25_k => 200,
      final_k => 200
    ) AS hybrid_query,
    $3::turbohybrid_multivector AS colbert_query
),
candidates AS MATERIALIZED (
  SELECT p.id, p.colbert
  FROM passages p, q
  ORDER BY p.embedding <~> q.hybrid_query
  LIMIT 200
)
SELECT c.id
FROM candidates c, q
ORDER BY turbohybrid_multivector_maxsim(q.colbert_query, c.colbert) DESC
LIMIT 10;
```

This keeps the first-stage index payload unambiguous: `vector_query` and
`multivector_query` are still mutually exclusive in one `turbohybrid_query`.
The ColBERT reranker query is passed to the scalar MaxSim function and therefore
does not change the vector+BM25 fusion semantics. The final SQL ordering in the
outer query is exact heap MaxSim over the retained candidate window.

The candidate window is the quality/latency knob. Larger `dense_k`, `bm25_k`,
`final_k`, and CTE `LIMIT` values give the reranker more documents to rescue but
increase heap fetch and MaxSim work.

## Tuning Knobs

Approximate candidate collection:

```sql
SET turbohybrid.multivector_subvector_k = 100;
SET turbohybrid.multivector_unique_docs_per_token = 100;
SET turbohybrid.multivector_max_raw_hits_per_token = 400;
SET turbohybrid.multivector_adaptive_widening = 'auto'; -- off | auto | on
SET turbohybrid.multivector_doc_candidate_k = 100;
SET turbohybrid.multivector_candidate_source = 'graph'; -- graph | document_nodes | exact_token_scan | exact_doc_scan | doc_graph_prototype | proxy_vector | centroid_lite | quantized_inverted_experimental
SET turbohybrid.multivector_proxy_encoder = 'normalized_mean'; -- normalized_mean | mean | mean_pool | first_token | max_abs_mean | centroid_mean | max_pool | random_projection_fde | learned_projection_placeholder | learned_projection_v1
SET turbohybrid.multivector_plain_fallback = 'auto'; -- auto | off | force
SET turbohybrid.multivector_plain_fallback_max_docs = 1000;
SET turbohybrid.multivector_plain_fallback_candidate_fraction = 0.5;
SET turbohybrid.multivector_candidate_reservoirs = 'conservative'; -- off | conservative | balanced
SET turbohybrid.multivector_per_token_doc_reservoir_k = 1;
SET turbohybrid.multivector_coverage_reservoir_k = 10;
SET turbohybrid.multivector_bm25_candidate_injection = 'off'; -- off | hybrid_only | dense_with_text
SET turbohybrid.multivector_sparse_candidate_source = 'off'; -- off | bm25 | learned_sparse
SET turbohybrid.multivector_docmap = 'auto'; -- off | auto | require
```

Adaptive widening starts each query token at
`turbohybrid.multivector_subvector_k`. If that traversal finds fewer
token-local unique documents than
`turbohybrid.multivector_unique_docs_per_token`, it widens and retries that
token up to the hard `turbohybrid.multivector_max_raw_hits_per_token` cap.

`turbohybrid.multivector_docmap` controls the persistent node-to-document
sidecar used by multivector indexes. `auto` prefers the sidecar and falls back
to the heap-TID hash path for old indexes that do not have it. `off` forces the
heap-TID hash path. `require` fails with REINDEX guidance if a sidecar is
missing or malformed.

`turbohybrid.multivector_candidate_source` defaults to `graph`. For
`multivector_graph = token_nodes`, that is the production token-node graph
path. For `multivector_graph = document_nodes`, it uses the explicit
document-node index path described below. `document_nodes` is an explicit
candidate-source alias that requires a document-node index. `proxy_vector` also
requires `multivector_graph = document_nodes`; it uses the index-persisted
fixed-dimensional proxy encoder as the single-vector graph key for admission
and exact-reranks admitted documents with full MaxSim. `normalized_mean` is the
default document proxy. `first_token`, `max_pool`, and
`random_projection_fde` are additional pluggable encoders for DBpedia admission
comparison. `learned_projection_placeholder` remains an explicit unsupported
sentinel, while `learned_projection_v1` is an opt-in file-backed first slice
that requires `turbohybrid.multivector_learned_projection_path` and keeps the
projected proxy dimension equal to the multivector dimension to avoid an index
format change. See [multivector-learned-projection-proxy.md](dev/multivector-learned-projection-proxy.md).
`centroid_mean` is an optional quality mode that keeps one graph node per document, requires
`multivector_centroids = kmeans`, stores compact centroids in the sidecar, uses
the normalized mean of those centroids as the graph proxy vector, and runs a
bounded centroid MaxSim pre-rerank before the unchanged full-token exact MaxSim
rerank. `multivector_centroid_count = 0` means auto. Scan stats expose
`proxy_encoder_kind`, `proxy_candidates`, `proxy_top1_admission`,
`proxy_exact_rerank_docs`, `multivector_centroid_count`,
`multivector_centroid_prerank_docs`, and
`multivector_full_maxsim_rerank_docs`.
`centroid_lite` is an experimental PLAID-inspired candidate source for indexes
built with `multivector_centroids = kmeans`. On document-node indexes, build
and incremental insert persist deterministic document-local k-means centroid
vectors, a mean residual summary, and codeword posting tuples in the
multivector docmap sidecar. Scans load those persisted posting lists, keep the
original multivector sidecar for exact heap MaxSim rerank, validate centroid
sidecar entries lazily for documents reached through postings, and report
`centroid_lists_visited`, `centroid_docs_touched`, `centroid_pruned_docs`,
`centroid_postings_touched`, `centroid_postings_skipped`,
`centroid_posting_limit_per_token`, and `centroid_candidates`. The optional
`turbohybrid.multivector_centroid_lite_max_postings_per_token` GUC defaults to
`0`, which preserves full posting-list admission. With
`turbohybrid.multivector_centroid_lite_posting_selection = uniform_stride`,
positive values cap each query-token posting list for benchmark tradeoff
experiments before the exact MaxSim rerank. Positive uniform caps use
deterministic midpoint-spaced sampling across the posting list, not first-N
truncation, and report `centroid_posting_cap_strategy = uniform_stride`.
The experimental
`turbohybrid.multivector_centroid_lite_posting_selection = score_topk` mode
instead uses newly persisted centroid posting payloads to keep the strongest
bounded postings from the probed centroid lists, scores retained candidates
with compact query-centroid MaxSim, and keeps the best document pool for exact
heap MaxSim rerank. Existing centroid indexes built before posting payloads
remain readable but need REINDEX/rebuild before `score_topk` can use the
payload-sorted posting order. The scan-time
`turbohybrid.multivector_centroid_lite_candidate_scoring` GUC defaults to
`posting_payload`; the experimental `codeword_maxsim` value accumulates a
PLAID-style approximate MaxSim over the selected centroid/codeword matches
without loading document-centroid vectors, and `doc_centroid_maxsim` re-ranks
touched documents by approximate MaxSim over the persisted document-centroid
sidecar before the exact heap rerank. These are admission-only scorers:
retained documents are always exact heap MaxSim-reranked. The companion
`turbohybrid.multivector_centroid_lite_probe_centroids_per_token` GUC expands
each query token to multiple deterministic centroid probes, and
`turbohybrid.multivector_centroid_lite_score_threshold` can skip weak centroid
lists before broad document touching. The relative
`turbohybrid.multivector_centroid_lite_score_drop_from_best` filter keeps
only probed lists close to the best centroid score for each query token. The
benchmark applies these scan-time filters to both `codeword_maxsim` and
`doc_centroid_maxsim` rows so the compact-code scorer can be measured before
the slower document-centroid rescoring path. Treat uniform caps as diagnostic
negative controls: they cannot be promoted unless admission recall improves
while `centroid_docs_touched / doc_count` also drops. On token-node indexes it
is a compatibility mode that uses exact token-scan admission before exact heap
MaxSim rerank and reports
`multivector_doc_graph_warning =
token_node_centroid_lite_exact_token_prefilter`. It is opt-in benchmark
plumbing, not the primary document-node path. Plain fallback is bypassed when
`centroid_lite` is selected, so missing centroid configuration fails explicitly.
For document-node serving checks, interpret `centroid_lite` latency by phase:
the bounded exact MaxSim rerank must respect
`turbohybrid.multivector_exact_rerank_k`, while high
`centroid_docs_touched` or `docs_scored_near_table_size` means the centroid
posting prefilter itself is near-exhaustive. That is a candidate-source
admission problem, not a sidecar or SIMD rerank problem. The guarded
`turbohybrid.multivector_centroid_lite_bitset_prefilter = experimental` mode
currently builds a scan-local posting-union bitset from selected posting lists.
It is off by default, does not change persisted storage, and can require a
document to appear in at least
`turbohybrid.multivector_centroid_lite_bitset_min_token_matches` selected
query-token lists before the document reaches the candidate heap. It reports
`centroid_bitset_prefilter_enabled`,
`centroid_bitset_lists_used`, `centroid_bitset_docs_set`,
`centroid_bitset_docs_after_threshold`,
`centroid_bitset_candidates`, `centroid_bitset_time_us`
(`centroid_bitset_prefilter_time_us` is kept as a compatibility alias), and
`centroid_bitset_memory_bytes`. This is a scan-local prototype only: there is
no persisted bitset or posting-block layout in the stable index format.
The guarded `turbohybrid.multivector_centroid_lite_pruning =
safe_upper_bound` prototype is also off by default. It only drops a
centroid-lite candidate after the current candidate band is full and the
persisted centroid interaction is a proven safe bound for that document. The
current sidecar stores a mean residual summary rather than a radius, so
nonzero residual summaries are treated as unsafe and the document is retained.
Stats expose `centroid_upper_bound_enabled`,
`centroid_upper_bound_docs_checked`, `centroid_upper_bound_docs_pruned`,
`centroid_upper_bound_time_us` (`centroid_upper_bound_prune_time_us` is kept
as a compatibility alias), `centroid_upper_bound_prune_ratio`,
`centroid_upper_bound_unsafe_fallbacks`,
`centroid_candidates_before_bound`, and
`centroid_candidates_after_bound`. Final retained documents are still ranked by
exact MaxSim.
See `docs/dev/multivector-centroid-bitset-prefilter.md` for the non-production
design direction.
`quantized_inverted_experimental` is a guarded research-only ColBERTSaR-style
candidate source for document-node indexes. The current prototype uses
persisted experimental deterministic codeword postings from the multivector
docmap sidecar, validates document vectors lazily for touched postings, then
exact-reranks admitted heap documents with MaxSim. It supports bounded
multi-codeword probing and `score_topk` posting selection using the same
compact deterministic codeword scorer used for measurement in centroid-lite.
Token-node indexes fail explicitly, plain fallback is bypassed when the source
is selected, and there is no production on-disk compatibility promise; see
`docs/dev/multivector-colbertsar-research.md`.
For pure-ColBERT candidate-source focus runs with an explicit external
codebook, named profiles encode the codebook source, posting caps, probe count,
compact scoring mode, and exact rerank budget. These profiles are benchmark
configurations only. They do not change SQL defaults, and they remain
experimental until an acceptance report proves admission, qrel quality, latency,
and work-counter gates on the target corpus.
`exact_token_scan`, `exact_doc_scan`, and
`doc_graph_prototype`
are developer validation modes for separating token graph recall, token-top-K
admission loss, exact document MaxSim, and document-level graph behavior. They
are useful for benchmark diagnosis, not normal serving defaults.

### ColBERT candidate-source metrics

Every pure-ColBERT candidate-source row should be read in two layers:
candidate admission first, exact heap MaxSim rerank second. A candidate source
may be useful only if it admits relevant documents cheaply enough for the final
exact rerank band. The final SQL order is still exact MaxSim over retained heap
tuples; approximate proxy, centroid, or codeword scores are admission signals.

Use these metric groups when comparing implementations:

| Area | Fields |
| --- | --- |
| Latency | `p50_ms`, `p95_ms`, `p99_ms`, `qps`, `phase_total_time_us` |
| Qrel quality | `recall@10`, `ndcg@10`, `mrr@10` |
| Exact-oracle admission | `exact_top1_admission_rate`, `exact_top10_admission_recall`, `admission_evidence` |
| Proxy graph admission | `proxy_candidates_returned`, `proxy_candidate_limit_effective`, `proxy_candidate_limit_source`, `proxy_graph_nodes_visited`, `proxy_graph_edges_visited`, `proxy_vector_score_time_us` |
| Centroid-lite admission | `centroid_lists_visited`, `centroid_postings_touched`, `centroid_postings_selected`, `centroid_docs_touched`, `centroid_candidates`, `centroid_candidate_scoring` |
| Centroid-lite filters | `centroid_bitset_docs_after_threshold`, `centroid_bitset_memory_bytes`, `centroid_upper_bound_docs_pruned`, `centroid_upper_bound_unsafe_fallbacks` |
| Quantized-inverted admission | `quantized_inverted_postings_touched`, `quantized_inverted_docs_scored`, `quantized_inverted_candidates`, `quantized_inverted_precompact_pruned_docs` |
| Compact code scoring | `quantized_inverted_compact_kernel`, `quantized_inverted_compact_score_us`, `quantized_inverted_compact_docs_scored`, `quantized_inverted_compact_pairs_evaluated`, `quantized_inverted_compact_pairs_skipped` |
| Exact rerank | `multivector_exact_rerank_docs`, `multivector_exact_rerank_pairs`, `multivector_exact_kernel`, `multivector_exact_maxsim_rerank_time_us`, `multivector_exact_heap_fetch_time_us` |
| Storage and cache | `multivector_doc_storage_kind`, `proxy_only_index`, `centroid_only_index`, `full_multivector_sidecar_available`, `multivector_doc_sidecar_bytes_touched`, `multivector_doc_sidecar_pages_read` |

Interpret zero or near-zero `recall@10`/`ndcg@10` profiles as negative controls
when qrels are available. Interpret missing exact admission separately: BEIR
quality-only runs intentionally skip `exact_doc_scan`, while sampled-oracle
runs label admission as `sampled_exact_oracle` and must report how many oracle
queries were covered.

`turbohybrid.multivector_plain_fallback = auto` switches to exact heap MaxSim
when the estimated corpus is small enough or the requested candidate/rerank
budget is near-exhaustive. `force` is the exact document oracle inside the
access method; `off` keeps the token-node candidate path active for diagnostics.

Multivector indexes also accept:

```sql
WITH (
  multivector_graph = token_nodes,
  multivector_centroids = off,
  multivector_centroid_count = 0,
  multivector_proxy_encoder = normalized_mean,
  multivector_context_mode = flat,
  multivector_field_mode = off
)
```

`token_nodes` is the default compatibility storage mode. It preserves existing
behavior and reports `multivector_graph_mode = token_nodes` in
`turbohybrid_index_stats()`. `multivector_context_mode = flat` keeps current
cross-context/global MaxSim semantics. `multivector_context_mode =
context_level` is explicit opt-in for context-window scoring in document-node
sidecar and exact heap rerank paths; long documents still return one SQL row.
`multivector_field_mode = weighted` records that field metadata is expected and
is visible in index stats, but query-specific field weights still come from the
explicit `turbohybrid_multivector_field_weighted_maxsim()` scorer until the
query payload carries field weights.

Explicit `document_nodes` indexes store one graph node per heap document and a
document-node storage tier. The full-sidecar tiers `f32`, `f16`, and `sq8`
persist document multivectors in the docmap sidecar; `proxy_only` persists only
the proxy graph/docmap data; `centroid_only` persists proxy graph/docmap data
plus k-means centroid/posting sidecars. When an external
`quantized_inverted_experimental` codebook is selected at build time,
`centroid_only` can also persist compact quantized posting/codeword payloads for
the explicit compact experimental path. Production builds use a fixed-dimensional
proxy vector for graph topology and reserve exact MaxSim for bounded final
rerank. The diagnostic `multivector_doc_build_scorer = exact_symmetric` mode
uses symmetrized document MaxSim during graph construction and is guarded by
`turbohybrid.multivector_exact_symmetric_build_max_docs` unless
`turbohybrid.multivector_allow_exact_symmetric_build = on`. Non-exhaustive
scans traverse document graph adjacency. `proxy_vector` candidate generation
scores graph nodes with the fixed-dimensional proxy vector and only touches full
document multivectors for bounded exact rerank or explicitly selected full
document-node sidecar modes. Non-proxy document-node scans score visited
candidates with the selected `turbohybrid.multivector_doc_storage = f32 | f16 |
sq8` sidecar. `proxy_only` and `centroid_only` do not silently substitute a
full-sidecar path: unsupported candidate sources fail with REINDEX guidance, and
bounded exact rerank fetches heap multivectors. `centroid_only` supports
`centroid_lite` and the guarded compact `quantized_inverted_experimental` path
when `quantized_inverted_sidecar_available = true`. Near-exhaustive scans use
the exact float32 sidecar scan only when that sidecar exists.
`turbohybrid_index_stats()` reports
`multivector_context_mode` and `multivector_field_mode` for benchmark
provenance, plus storage capabilities:
`multivector_doc_storage_kind`, `proxy_only_index`, `centroid_only_index`,
`full_multivector_sidecar_available`, `centroid_sidecar_available`,
`quantized_inverted_sidecar_available`, and
`exact_rerank_source_supported`. `multivector_doc_graph_warning` reports
`document_node_f32_sidecar_graph_traversal`,
`document_node_f16_sidecar_graph_traversal`,
`document_node_sq8_sidecar_graph_traversal`, or
`document_node_f32_sidecar_exact_scan`.

Large document-node corpora can choose the sidecar cache policy explicitly:

```sql
SET turbohybrid.multivector_doc_storage_cache = 'auto';     -- auto | resident | paged
```

`auto` keeps plain `proxy_vector` document-node scans from loading the full
sidecar for graph admission. On full-sidecar indexes, full multivectors are
touched only for bounded exact rerank or explicit sidecar-scoring modes. On
`proxy_only` and `centroid_only` indexes, bounded exact rerank uses heap
multivectors because `full_multivector_sidecar_available = false`. Other
document-node scan modes may keep the document sidecar resident for
latency-profile scans when it fits
`turbohybrid.native_cache_max_mb`; explicit `resident` still forces loaded
native sidecar storage, while `paged` keeps graph adjacency resident and
loads/touches sidecar pages for the scan. Last-scan stats expose
`multivector_doc_sidecar_cache_mode`,
`multivector_doc_sidecar_pages_read`,
`multivector_doc_sidecar_cache_hits`,
`multivector_doc_sidecar_cache_misses`, and
`multivector_doc_sidecar_bytes_touched`. These aggregate sidecar counters are
kept for compatibility and can include first-scan cache construction. For
document-node proxy scans, `sidecar_cache_build_bytes`,
`sidecar_cache_build_pages_read`, and `sidecar_cache_build_time_us` separate
native cache construction from per-query sidecar materialization/touches
reported as
`sidecar_query_bytes_touched`, `sidecar_query_pages_read`,
`sidecar_query_vectors_loaded`, and `sidecar_query_time_us`. The proxy path also
reports `proxy_candidate_limit_effective` and `proxy_candidate_limit_source`, so
benchmarks can distinguish a requested candidate budget from an EF-limited graph
traversal. Paged document-node scans also expose
`multivector_doc_sidecar_vectors_loaded`, which counts document multivectors
reconstructed from sidecar chunks during the scan.

Document-node indexes can optionally pool document tokens at index build and
incremental insert time:

```sql
CREATE INDEX passages_colbert_doc_pool_idx
ON passages USING turbohybrid
  (colbert multivector_maxsim_ip_turbohybrid_ops)
WITH (
  multivector_graph = document_nodes,
  multivector_token_pooling = greedy_cosine, -- off | kmeans | greedy_cosine
  multivector_token_pooling_target_ratio = 0.5,
  multivector_token_pooling_min_tokens = 16
);
```

Pooling is opt-in and applies only to stored document tokens. Query
multivectors are not pooled. The document sidecar stores the original and pooled
token counts so scans can report `multivector_tokens_original`,
`multivector_tokens_pooled`, and `multivector_token_pooling_ratio`. Graph
traversal scores the pooled sidecar; exact heap rerank can still score the
original heap `turbohybrid_multivector` when the heap value is available.
Context-aware multivectors currently require
`multivector_token_pooling = off`, because pooling rewrites the document-token
sequence and context-aware pooling must preserve or recompute window metadata.

Document-node indexes also accept experimental PLAID-lite centroid options:

```sql
CREATE INDEX passages_colbert_centroid_idx
ON passages USING turbohybrid
  (colbert multivector_maxsim_ip_turbohybrid_ops)
WITH (
  multivector_graph = document_nodes,
  multivector_centroids = kmeans,        -- off | kmeans
  multivector_centroid_count = 0         -- 0 means auto
);
```

Use them with `SET turbohybrid.multivector_candidate_source = 'centroid_lite'`.
The current implementation is deliberately conservative: centroid interaction
is only an admission approximation and final ranking still uses exact heap
MaxSim over the original document multivectors.

### x00k Serving Selection Benchmark

Use the DBpedia ColBERT serving grid when selecting a document-node profile for
10k to x00k corpora:

```sh
python benchmarks/dbpedia_colbert_multivector.py \
  --precomputed-dataset johannhartmann/pgturbohybrid_dbpedia_colbert \
  --max-docs 10000 \
  --max-queries 100 \
  --reuse-data \
  --document-node-serving-grid \
  --admission-budget-sweep 50,100,200,400,800 \
  --output .nix-dev/tmp/dbpedia-colbert-serving-grid-10k.json \
  --markdown-output .nix-dev/tmp/dbpedia-colbert-serving-grid-10k.md
```

For x00k evaluation, raise `--max-docs` to `100000`, `300000`, or the corpus
size and use `--max-queries 500` or `1000`. Leave graph knobs at the extension
defaults unless the run is explicitly a smoke test. The report ranks the named
document-node profiles with `best_latency_safe`, `best_quality`, and
`best_balanced`; choose `best_latency_safe` when it passes admission and qrel
quality thresholds, and choose `best_quality` when relevance dominates. Treat
`centroid_lite` as PLAID-inspired admission with exact final MaxSim, and treat
`quantized_inverted_experimental` as opt-in research. Generated JSON,
Markdown, local logs, and datasets should stay under `.nix-dev/tmp/` or another
ignored path and must not be committed.

The default serving grid stays compact. Use
`--document-node-serving-grid-include-proxy-encoders` only for focused admission
experiments that need the additional `proxy_max_pool_f16` and
`proxy_random_projection_fde_f16` profile names; pair it with
`--document-node-serving-grid-profiles` to avoid widening unrelated runs.
Use `--document-node-serving-grid-pure-dense-proxy-focus` when evaluating a
replacement for the weak `normalized_mean` proxy admission baseline without
BM25, learned sparse, or hybrid rescue. This mode compares
`proxy_normalized_mean_proxy_only`, `proxy_max_pool_proxy_only`, and
`proxy_random_projection_fde_proxy_only` over candidate budgets `800,1600,3200`
and exact rerank `k = 100,400,800`; if
`--multivector-learned-projection-path` is supplied, it also includes
`proxy_learned_projection_v1_proxy_only`. Missing learned-projection weights
fail before index build, and final retained candidates remain exact MaxSim
ranked. Treat normalized mean as a latency baseline in this report, not as the
main short-term ColBERT admission path when a stronger pure-dense proxy has
nonzero admission at acceptable latency.
Use `--document-node-serving-grid-include-bm25-rescue` only when admission loss
is the question and the benchmark should build a BM25 key for lexical rescue
profiles. The added `*_bm25_rescue` profiles are admission experiments: BM25
can add candidate documents, but retained documents are still ordered by exact
MaxSim. The benchmark sends `text_query` for these profiles so the rescue
branch is active even though the profile remains dense-only for final ranking.
Use `--document-node-serving-grid-include-centroid-lite-caps` only for focused
centroid-lite pruning experiments. It adds capped profile rows such as
`centroid_lite_f16_cap_016`, `centroid_lite_f16_cap_032`, and
`centroid_lite_f16_cap_064`, driven by the scan-time
`turbohybrid.multivector_centroid_lite_max_postings_per_token` GUC. It also
adds guarded rows such as `centroid_lite_f16_prune_safe_upper_bound` and
`centroid_lite_f16_cap_032_prune_safe_upper_bound`, driven by
`turbohybrid.multivector_centroid_lite_pruning = safe_upper_bound`. These rows
reuse the same persisted kmeans centroid sidecar as uncapped `centroid_lite_f16`
and remain experimental admission evidence; final retained candidates are still
ordered by exact MaxSim. Positive caps use deterministic uniform-stride posting
sampling and expose `centroid_posting_cap_strategy` so reports distinguish
uncapped full-list admission from capped sampling. Uniform cap rows are
diagnostic only and should stay rejected unless they both improve admission
recall and reduce documents touched; zero-quality cap rows are negative
controls, not serving candidates. Safe upper-bound pruning only
drops candidates when the persisted residual summary proves the centroid score
is exact for that document; otherwise it keeps the candidate and reports
`centroid_upper_bound_unsafe_fallbacks`.
Use `--document-node-serving-grid-centroid-lite-focus` for a compact
centroid-lite-only comparison of uncapped, capped `016/032/064`, guarded
`safe_upper_bound` pruning, scan-local bitset-prefilter measurement rows,
pooled `centroid_lite_f16_pool_050`, and `centroid_mean_f16` baseline rows.
Use `--document-node-serving-grid-token-pooling-memory-focus` when the next
decision is whether document-node token pooling is the safest storage reduction
path for 1M ColBERT indexes. The legacy
`--document-node-serving-grid-token-pooling-focus` flag is an alias. The focus
grid compares pure-ColBERT rows only: `proxy_max_pool_proxy_only`,
`proxy_random_projection_fde_proxy_only`, and
`document_nodes_sq8_paged_maxsim`, each with pooling `off` plus
`greedy_cosine` target ratios `0.75` and `0.50`.
`document_nodes_sq8_paged_maxsim` uses document-node graph admission with `sq8`
storage, paged document sidecar access, approximate full MaxSim graph scoring,
and exact heap MaxSim rerank for retained candidates. It excludes BM25 rescue,
learned-sparse rescue, and centroid-lite cap rows. Reports include
index/docmap/sidecar/graph bytes, memory gain versus the unpooled family
baseline, pooled token counts, pooling ratio, exact rerank pairs, build time,
latency, admission, `compact_maxsim_score_us`, `compact_maxsim_pairs`, and qrel
metrics when available, plus
`best_memory_safe_pooling`, `best_latency_safe_pooling`,
`best_pooling_quality_safe`, and rejected pooling profiles. Pooling changes the
persisted document multivectors, so pooled rows are separate index builds and
final retained candidates are still ranked by exact MaxSim.
If `--document-node-serving-grid-include-proxy-encoders` is also set, the grid
adds `proxy_max_pool_f16_bm25_rescue` as a focused comparison for the stronger
`max_pool` proxy baseline; the default BM25 rescue set remains unchanged.
The rescue report records the effective rescue candidate limit, combined pool
size, limit reason, retained candidates, and exact-reranked rescue count so a
run can distinguish lexical underfill from dense/rerank cap truncation. The
same aggregate rescue summaries are preserved in the serving recommendation
rows, rejected-profile rows, and Markdown "why this profile won" text.
Rejected rows also carry deterministic `admission_improvement_hints` so the
next run can distinguish an EF cap, document-candidate cap, rerank cap, lexical
underfill, learned-sparse fixture gap, or a broader candidate-source quality
bottleneck. If exact rerank exhausts the admitted band and top-10 admission is
still low, the hints point to the next focused candidate-quality experiment:
`try_max_pool_or_centroid_mean_proxy`, `try_centroid_mean_proxy`,
`try_sparse_rescue_or_centroid_lite`,
`try_balanced_candidate_reservoirs`, or
`try_bm25_or_learned_sparse_rescue`.
Use `--document-node-serving-grid-include-reservoirs` only for focused
candidate-admission experiments. Reservoir rows set
`turbohybrid.multivector_candidate_reservoirs = balanced`. For explicit
document-node `proxy_vector` scans, reservoir mode selects the bounded exact
rerank band from proxy-ranked candidates using a score-prefix plus deterministic
rank-spread sample. `conservative` keeps most of the proxy-ranked prefix;
`balanced` allocates more of the band to rank-spread exploration. This is still
a scan-time candidate-admission experiment, not a default serving profile:
final ordering remains exact MaxSim over retained document candidates. A row is
valid evidence only when
`turbohybrid_last_scan_stats()` reports `multivector_reservoirs_enabled = true`
and nonzero reservoir union docs; otherwise the benchmark marks it as
`candidate_reservoirs_not_executed` and the row cannot become a safe serving
recommendation. If proxy-encoder variants are also enabled, the grid can expose
`proxy_max_pool_f16_reservoir_balanced`; interpret those rows through the same
execution stats.
Use `--document-node-serving-grid-include-learned-sparse-rescue` only when
external learned-sparse document/query features are available. It adds
`proxy_normalized_mean_f16_learned_sparse_rescue` and
`centroid_mean_f16_learned_sparse_rescue`, requires
`--learned-sparse-doc-jsonl` plus `--learned-sparse-query-jsonl`, and uses the
learned-sparse postings branch only as an admission source before exact MaxSim
final ordering. If proxy-encoder variants are also enabled, the grid adds
`proxy_max_pool_f16_learned_sparse_rescue` as the matching focused comparison
profile. Mixed BM25 and learned-sparse serving-grid runs use separate
lexical index keys: BM25 rescue builds `body_tsv`, while learned-sparse rescue
builds `learned_sparse_tsv`. Benchmark output includes learned-sparse
document/query coverage ratios; partial JSONL fixture coverage is explicitly
flagged and should not be treated as production serving evidence.

Production readiness for learned-sparse rescue requires this checklist before a
profile is promoted from benchmark evidence to serving guidance:

- Document JSONL coverage is 100% for the served corpus, or partial coverage is
  explicitly accepted and recorded as a limitation.
- Query JSONL coverage is 100% for the benchmark query set.
- The `term_id` vocabulary is stable across document and query feature files.
- The feature-generator version is recorded.
- The feature-generator model name and checksum are recorded when available.
- `learned_sparse_tsv` is built and indexed; learned-sparse rescue must not
  silently use the `body_tsv` BM25 key.
- The query path uses the learned sparse tsquery generated from
  `q.learned_sparse`, not the web-search text fallback.
- Final SQL ranking remains exact MaxSim over retained multivector candidates.
- Rescue cap accounting is present: learned-sparse candidates,
  retained-for-MaxSim counts, branch latency, and exact rerank counts.
- p95 latency and admission thresholds pass on the target corpus, not only on a
  tiny smoke run.

Any future SQL-visible `turbohybrid.multivector_serving_profile` mapping must
first pass the benchmark gate:
`SERVING_GRID_JSON=... python benchmarks/dbpedia_colbert_multivector.py --validate-serving-profile-guc-evidence`.
The gate is expected to reject smoke-only or undersized artifacts, incomplete
learned-sparse coverage, experimental profiles, missing qrel quality metrics,
or slow-path warnings unless the prompt explicitly opts into
`ALLOW_UNSAFE_PROFILE=1`.

Use `--document-node-serving-grid-include-entry-sidecar` only for focused graph
entry admission experiments. It adds `proxy_normalized_mean_f16_entry_sidecar`
and `centroid_mean_f16_entry_sidecar`, persists `entry_sidecar = on` graph entry
representatives with 128 `hybrid_level_covering` entries by default, and reports
the entry-sidecar settings plus `graph_entry_sidecar_*` scan counters in the
serving-grid JSON and Markdown. These rows have distinct physical index
signatures from the plain proxy profiles and should be read as candidate-quality
evidence, not as default serving profiles. Final ranking remains exact MaxSim
over the retained document candidates. If the proxy-encoder comparison switch is
also enabled, `proxy_max_pool_f16_entry_sidecar` is added as the matching
`max_pool` graph-entry admission experiment.
For scan-time entry sampling experiments on an existing document-node graph, use
`turbohybrid.multivector_doc_graph_entry_sample_count` or the benchmark flag
`--multivector-doc-graph-entry-sample-count`. The default `0` preserves the
compiled entry-sample heuristic. A positive value scores that many deterministic
document proxy entry seeds, bounded by document count, before base-layer graph
traversal. Last-scan stats report `graph_entry_sample_configured`,
`graph_entry_sample_effective`, `graph_entry_sample_scored`, and the
`multivector_doc_graph_entry_sample_*` aliases. This is a candidate-admission
measurement knob only; it does not change persisted index format or final exact
MaxSim ordering.
Use `--document-node-serving-grid-include-entry-samples` for paired serving-grid
evidence. It adds scan-time variants such as
`proxy_normalized_mean_f16_entry_sample_032` and
`centroid_mean_f16_entry_sample_032` using the counts from
`--document-node-serving-grid-entry-sample-counts` (default `32,128`). These
rows reuse the same physical document-node index as their baseline profile and
only change graph-entry seeding during retrieval. The serving-grid JSON,
Markdown, recommendation rows, and `candidate_source_deltas` report the
effective entry-sample count and compare the variant against the matching
baseline at the same EF, oversampling, and candidate budget.
The serving-grid report also emits `candidate_source_deltas` for known paired
variants: entry sampling, entry-sidecar, candidate reservoirs, BM25 rescue,
learned-sparse rescue, centroid-lite caps, and proxy-encoder variants. Each delta compares
against the matching plain baseline at the same EF, oversampling, and executed
candidate budget, so candidate-source changes can be judged before widening a
smoke into a larger x00k run. Delta rows also preserve compact experiment
evidence such as entry-sample counts, entry-sidecar representative work,
reservoir union/duplicate summaries, rescue-candidate summaries, and
centroid-lite posting caps.
The serving recommendation repeats the strongest deltas in
`candidate_source_delta_summary`: best top-10 admission gain, best NDCG gain,
best p95-latency gain, and best admission gain per comparison family. These
fields are benchmark evidence only; they do not change SQL candidate generation
or final exact MaxSim ordering.
The first 2k DBpedia comparison with these opt-in proxy profiles kept
`centroid_mean_f16` as the best quality/balanced safe profile. `max_pool`
improved admission over `normalized_mean` but still trailed `centroid_mean`,
while `random_projection_fde` was fast but too weak for admission.

After the proxy sidecar fix, the decisive serving evidence should separate
exact rerank cost from candidate-source cost. A centroid-lite run with
`candidate_k = 800` and `exact_rerank_k = 100` should report
`multivector_exact_rerank_docs = 100`; if latency is still high and
`centroid_docs_touched` is near the document count, the next optimization is
posting-list admission/pruning rather than sidecar caching or MaxSim SIMD.

The serving-grid report includes explicit cost accounting. At the top level,
`total_elapsed_ms`, `index_build_elapsed_ms_total`,
`exact_baseline_elapsed_ms_total`, `retrieval_elapsed_ms_total`,
`exact_baseline_query_count`, and `retrieval_query_count` show whether the run
spent time building indexes, recomputing exact admission baselines, or executing
indexed retrieval. The same split is emitted per profile and per
profile/EF/oversampling row.
Build diagnostics for document-node centroid/proxy indexes are exposed through
`turbohybrid_last_build_stats()`: `multivector_centroid_build_us`,
`multivector_centroid_cluster_us`, `multivector_centroid_residual_us`,
`multivector_centroid_build_docs`, `multivector_centroid_build_vectors`,
`multivector_proxy_build_us`, `multivector_doc_sidecar_write_us`,
`multivector_centroid_sidecar_write_us`,
`multivector_centroid_posting_write_us`, and
`multivector_centroid_posting_count`. Use these fields when a 10k/x00k
serving-grid run spends most of its time in `CREATE INDEX`; they distinguish
centroid construction, proxy construction, sidecar serialization, and centroid
posting serialization from graph topology work.
The benchmark harness exposes the same fields through
`--document-node-serving-build-only`, which builds the selected serving profiles
and skips retrieval/admission so build stalls can be diagnosed directly. The
report derives `dominant_build_phase`, `build_phase_known_ms`,
`build_phase_unattributed_ms`, preserves generic aliases such as
`build_edges_us`, `write_pages_us`, `wal_us`, and `total_us`, and classifies
slow build symptoms with labels including `centroid_kmeans_dominates_build`,
`centroid_posting_write_dominates_build`, `proxy_build_dominates_build`,
`graph_edges_dominates_build`, `build_unattributed_high`, and
`index_rebuild_not_reused`. Use the unattributed bucket as the signal for
graph/topology build work or another phase not covered by the multivector
timers.

Missing or malformed document-node sidecar metadata fails with REINDEX guidance
instead of silently falling back to token-node storage.

`turbohybrid.multivector_candidate_reservoirs` can replace score-only document
truncation with a bounded union of score, coverage, mean seen-similarity, and
per-query-token document reservoirs before exact rerank. This mitigates
token-node admission loss when useful pre-rerank evidence reached the
accumulator, but it is not a substitute for exact/plain fallback or a future
document-level MaxSim graph.

`turbohybrid.multivector_bm25_candidate_injection` can use lexical BM25 hits as
an admission safety net for multivector MaxSim when the index has a BM25 key and
the query supplies `text_query`. `hybrid_only` applies this to hybrid
multivector/text queries. `dense_with_text` also allows text-backed dense-only
MaxSim runs where BM25 admits candidates, but exact MaxSim remains the dense
score. RRF, when requested, still fuses document-level ranks.

`turbohybrid.multivector_sparse_candidate_source` is the explicit sparse
candidate branch selector. `off` keeps the legacy behavior, `bm25` uses lexical
postings, and `learned_sparse` is the SPLATE/SLIM-style admission path for
externally exported learned sparse vectors. PostgreSQL does not train the sparse
model; callers ingest term IDs and weights with
`turbohybrid_sparse_vector_from_arrays(term_ids int[], weights real[])`, convert
document-side features with `turbohybrid_sparse_vector_to_tsvector(...)`, and
convert query features with `turbohybrid_sparse_vector_to_tsquery(...)`. The
branch reuses the sparse/BM25 candidate machinery. In dense-only-with-text
queries, learned-sparse candidates are admission-only and final ranking remains
exact MaxSim. In hybrid queries, the branch is document-keyed and can contribute
to rank/score fusion modes that are explicitly enabled for multivector hybrid
search.

Exact heap rerank:

```sql
SET turbohybrid.multivector_exact_rerank = 'topk'; -- off | topk | adaptive
SET turbohybrid.multivector_exact_rerank_k = 100;
```

Safety caps:

```sql
SET turbohybrid.multivector_max_doc_vectors = 256;
SET turbohybrid.multivector_max_query_vectors = 64;
SET turbohybrid.multivector_max_dim = 4096;
SET turbohybrid.multivector_max_accumulator_mb = 64;
```

Model metadata:

```sql
SELECT turbohybrid_multivector_model_info('colbert-ir/colbertv2.0');
SET turbohybrid.multivector_model_name = 'jinaai/jina-colbert-v2-96';
```

`turbohybrid_multivector_model_info(model_name text)` returns a JSON profile
with `dim`, default query/document token caps, distance mode, token
normalization, recommended document-node sidecar storage, token mask policy, and
field/context policy. When `turbohybrid.multivector_model_name` is set to a
registered profile, build, insert, and query validation reject dimension
mismatches with model-aware hints. Token counts above the model profile defaults
emit warnings while still respecting the explicit GUC caps.
`turbohybrid_index_stats()` exposes the configured model name, known/unknown
status, dimension, recommended storage, mask policy, and context policy for
benchmark provenance.

Built-in profiles intentionally avoid assuming every ColBERT-like model is
128-dimensional:

| Model profile | Dim | Default query/doc tokens | Notes |
| --- | ---: | --- | --- |
| `colbert-ir/colbertv2.0` | 128 | 32 / 180 | Classic ColBERTv2 text profile. |
| `answerdotai/answerai-colbert-small-v1` | 96 | 32 / 512 | Small multilingual ColBERT profile. |
| `jinaai/jina-colbert-v2` | 128 | 32 / 8192 | Long-context default Matryoshka variant. |
| `jinaai/jina-colbert-v2-96` | 96 | 32 / 8192 | Jina 96-dimensional variant. |
| `jinaai/jina-colbert-v2-64` | 64 | 32 / 8192 | Jina 64-dimensional variant. |
| `lightonai/GTE-ModernColBERT-v1` | 128 | 48 / 300 | PyLate ModernColBERT profile; document length can be raised for long-context runs. |
| `chadboyda/Reason-ModernColBERT` | 128 | 48 / 300 | Reasoning-tuned ModernColBERT profile. |
| `VAGOsolutions/SauerkrautLM-Multi-ColBERT-15m` | 128 | 32 / 256 | PyLate validation model. |
| `johannhartmann/SauerkrautLM-Multi-ColBERT-15m-GGUF` | 128 | 32 / 256 | GGUF validation model for `pg_colbert_llama`. |
| `vidore/colpali-v1.2` / `colpali-like-visual` | 128 | 64 / model-specific | ColPali-style visual-document multivectors; document patch counts depend on processor settings. |

The exact rerank path fetches the original `turbohybrid_multivector` heap value
for a bounded document prefix and computes f32 MaxSim. It does not store a full
float copy of each subvector in the index by default.

`adaptive` preserves the exact returned top-K result while reducing MaxSim work
when a safe threshold exists. It processes query tokens in deterministic
importance order and bounds the remaining contribution with weighted
query-token norms and the candidate document's maximum token norm. Masked and
zero-weight query tokens do not contribute to the bound. Candidates are stopped
only when their upper bound cannot enter top-K. If no smaller final top-K
threshold exists, or if bounds are unsafe, the scan uses the full exact scorer.
Approximate relaxations require a separate explicit GUC.

## Performance Model

Use these variables when sizing or benchmarking:

- `D`: documents
- `L`: token vectors per document
- `N`: graph subnodes, `N = D * L`
- `Q`: query token vectors
- `d`: vector dimension
- `Ks`: raw subvector hits per query vector
- `C`: unique candidate documents
- `R`: exact rerank documents

Expected work:

- build expansion: `O(D * L)`
- approximate query collection: `O(Q * ANN + Q * Ks)`
- document accumulator memory: `O(C * Q)`, not `O(D * Q)` or `O(N * Q)`
- exact rerank: `O(R * Q * L * d)`

For local slope checks, run:

```sh
psql -d "$PGDATABASE" -f benchmarks/dev/multivector_late_interaction.sql
```

## Diagnostics

Inspect the last scan:

```sql
SELECT turbohybrid_last_scan_stats();
```

Useful fields include:

- `multivector_enabled`
- `multivector_query_vectors`
- `multivector_subvector_searches`
- `multivector_raw_subvector_hits`
- `multivector_adaptive_widening_triggered`
- `multivector_adaptive_initial_raw_target`
- `multivector_adaptive_final_raw_target`
- `multivector_unique_docs`
- `multivector_duplicate_doc_hits`
- `multivector_maxsim_updates`
- `multivector_doc_candidates`
- `multivector_docmap_source`
- `multivector_docmap_bytes`
- `multivector_reservoirs_enabled`
- `multivector_reservoir_score_docs`
- `multivector_reservoir_coverage_docs`
- `multivector_reservoir_mean_docs`
- `multivector_reservoir_per_token_docs`
- `multivector_reservoir_bm25_docs`
- `multivector_reservoir_union_docs`
- `multivector_reservoir_duplicates`
- `multivector_bm25_injection_enabled`
- `multivector_bm25_injection_candidates`
- `multivector_bm25_injection_retained`
- `multivector_bm25_injection_exact_reranked`
- `learned_sparse_candidates`
- `learned_sparse_retained_for_maxsim`
- `learned_sparse_branch_latency_us`
- `multivector_doc_graph_prototype_enabled`
- `multivector_doc_graph_nodes`
- `multivector_doc_graph_docs_scored`
- `multivector_doc_graph_edges_visited`
- `multivector_doc_graph_candidates`
- `multivector_doc_graph_quantized_scores`
- `multivector_doc_graph_heap_fetches`
- `multivector_doc_graph_exact_rerank_docs`
- `multivector_doc_graph_warning`
- `multivector_tokens_original`
- `multivector_tokens_pooled`
- `multivector_token_pooling_ratio`
- `multivector_exact_rerank_enabled`
- `multivector_exact_rerank_docs`
- `multivector_exact_rerank_pairs`
- `exact_rerank_candidates`
- `exact_rerank_tokens_evaluated`
- `exact_rerank_tokens_skipped`
- `exact_rerank_pairs_saved`
- `adaptive_rerank_topk_changed_vs_full`
- `multivector_exact_kernel`
- `multivector_accumulator_kind`
- `multivector_memory_bytes_estimate`

For resident-cache sizing, `turbohybrid_estimate_memory(index)` reports
`native.multivector_docmap_bytes` and includes those bytes in
`native.estimated_total_bytes` when the index has a valid sidecar.

## Known Tradeoffs

- Token-node candidate generation can miss many-moderate-match documents. A
  document may be the exact MaxSim top result because many query tokens match
  moderately, while no single document token is admitted early enough by
  per-token top-K search.
- Exact/plain fallback is safer for small or near-exhaustive cases. It scores
  MVCC-visible documents with full float32 MaxSim and avoids lossy token-node
  admission.
- Multi-reservoir retention and BM25 candidate injection are mitigations. They
  can preserve better document candidates after evidence reaches the
  accumulator, but they cannot recover a document that no candidate source
  admitted.
- A document-level graph is the MaxSim-aligned candidate-generation direction.
  `document_nodes` is explicit opt-in today and avoids token admission loss by
  scoring document candidates with full sidecar MaxSim. Its current warning
  marks that this first slice is an exact float32 sidecar scan, not yet the
  final compact quantized HNSW traversal.
- Exact rerank remains the semantic final scorer. Approximate candidates and
  admission sources should only decide which documents are eligible for exact
  MaxSim scoring.

## Limitations

- This is an alpha feature and the storage format is not a compatibility
  promise.
- Indexes built before the multivector docmap sidecar continue to work in
  `auto` mode through the heap-TID hash fallback. Use `REINDEX` to build the
  sidecar, or set `turbohybrid.multivector_docmap = 'require'` to enforce it.
- Prefer `multivector_maxsim_ip_turbohybrid_ops` for new indexes. The legacy
  `multivector_cosine_turbohybrid_ops` name is compatibility-only and assumes
  normalized token vectors while still using dot-product MaxSim internally.
- `turbohybrid_query` cannot contain both `vector_query` and
  `multivector_query`.
- Incremental insert/update into a token-node multivector TurboHybrid index
  expands the new tuple into one graph subnode per stored token vector.
  Document-node indexes store one graph node per document and the selected
  proxy, centroid, or sidecar payloads. Hybrid indexes append one BM25 delta for
  the inserted document when the lexical key is non-null.
- Hybrid multivector search is document-keyed. RRF is supported, and the
  normalized score-fusion modes `weighted`, `fast_weighted`, `calibrated`, and
  `dbsf` are supported where documented. Raw BM25 plus raw MaxSim alpha fusion
  remains rejected.
- SIMD MaxSim kernels are optional; scalar is always available and defines the
  semantics. SIMD-enabled builds may dispatch to AVX-512F, AVX2, or NEON exact
  rerank kernels.
