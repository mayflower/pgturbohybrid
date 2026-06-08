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

## Tuning Knobs

Approximate candidate collection:

```sql
SET turbohybrid.multivector_subvector_k = 100;
SET turbohybrid.multivector_unique_docs_per_token = 100;
SET turbohybrid.multivector_max_raw_hits_per_token = 400;
SET turbohybrid.multivector_adaptive_widening = 'auto'; -- off | auto | on
SET turbohybrid.multivector_doc_candidate_k = 100;
SET turbohybrid.multivector_candidate_source = 'graph'; -- graph | document_nodes | exact_token_scan | exact_doc_scan | doc_graph_prototype | proxy_vector | centroid_lite | quantized_inverted_experimental
SET turbohybrid.multivector_proxy_encoder = 'normalized_mean'; -- normalized_mean | first_token | centroid_mean | max_pool | random_projection_fde | learned_projection_placeholder
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
comparison, while `learned_projection_placeholder` fails explicitly until
learned projection weights are configured. `centroid_mean` is an optional
quality mode that keeps one graph node per document, requires
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
original multivector sidecar for exact heap MaxSim rerank, and report
`centroid_lists_visited`, `centroid_docs_touched`, `centroid_pruned_docs`, and
`centroid_candidates`. On token-node indexes it is a compatibility mode that
uses exact token-scan admission before exact heap MaxSim rerank and reports
`multivector_doc_graph_warning =
token_node_centroid_lite_exact_token_prefilter`. It is opt-in benchmark
plumbing, not the primary document-node path. Plain fallback is bypassed when
`centroid_lite` is selected, so missing centroid configuration fails explicitly.
`quantized_inverted_experimental` is a guarded research-only ColBERTSaR-style
candidate source for document-node indexes. The current prototype uses
persisted experimental deterministic codeword postings from the multivector
docmap sidecar, then exact-reranks admitted heap documents with MaxSim.
Token-node indexes fail explicitly, plain fallback is bypassed when the source
is selected, and there is no production on-disk compatibility promise; see
`docs/dev/multivector-colbertsar-research.md`.
`exact_token_scan`, `exact_doc_scan`, and
`doc_graph_prototype`
are developer validation modes for separating token graph recall, token-top-K
admission loss, exact document MaxSim, and document-level graph behavior. They
are useful for benchmark diagnosis, not normal serving defaults.

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
versioned float32 multivector sidecar. Production builds use a fixed-dimensional
proxy vector for graph topology and reserve exact MaxSim for bounded final
rerank. The diagnostic `multivector_doc_build_scorer = exact_symmetric` mode
uses symmetrized document MaxSim during graph construction and is guarded by
`turbohybrid.multivector_exact_symmetric_build_max_docs` unless
`turbohybrid.multivector_allow_exact_symmetric_build = on`. Non-exhaustive
scans traverse document graph adjacency, score visited candidates with the selected
`turbohybrid.multivector_doc_storage = f32 | f16 | sq8` sidecar, and
exact-rerank heap multivectors. Near-exhaustive scans use the exact float32
sidecar scan. `turbohybrid_index_stats()` reports
`multivector_context_mode` and `multivector_field_mode` for benchmark
provenance. `multivector_doc_graph_warning` reports
`document_node_f32_sidecar_graph_traversal`,
`document_node_f16_sidecar_graph_traversal`,
`document_node_sq8_sidecar_graph_traversal`, or
`document_node_f32_sidecar_exact_scan`.

Large document-node corpora can choose the sidecar cache policy explicitly:

```sql
SET turbohybrid.multivector_doc_storage_cache = 'auto';     -- auto | resident | paged
```

`auto` keeps the document sidecar resident for latency-profile scans when it
fits `turbohybrid.native_cache_max_mb`, otherwise it uses paged sidecar access.
`resident` prefers a loaded native sidecar, while `paged` keeps graph adjacency
resident and loads/touches sidecar pages for the scan. Last-scan stats expose
`multivector_doc_sidecar_cache_mode`,
`multivector_doc_sidecar_pages_read`,
`multivector_doc_sidecar_cache_hits`,
`multivector_doc_sidecar_cache_misses`, and
`multivector_doc_sidecar_bytes_touched`. Paged document-node scans also expose
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
- Incremental insert/update into a multivector TurboHybrid index expands the new
  tuple into one graph subnode per document vector. Hybrid indexes append one
  BM25 delta for the inserted document when the lexical key is non-null.
- Hybrid multivector search currently supports document-level RRF. Treat other
  fusion modes as unsupported unless tests and docs for that mode say otherwise.
- SIMD MaxSim kernels are optional; scalar is always available and defines the
  semantics. SIMD-enabled builds may dispatch to AVX-512F, AVX2, or NEON exact
  rerank kernels.
