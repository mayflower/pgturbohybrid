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

Exact reference scoring is available without an index:

```sql
SELECT turbohybrid_multivector_maxsim($query_mv, $doc_mv) AS score,
       turbohybrid_multivector_maxsim_distance($query_mv, $doc_mv) AS distance;
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
SET turbohybrid.multivector_candidate_source = 'graph'; -- graph | document_nodes | exact_token_scan | exact_doc_scan | doc_graph_prototype | proxy_vector
SET turbohybrid.multivector_plain_fallback = 'auto'; -- auto | off | force
SET turbohybrid.multivector_plain_fallback_max_docs = 1000;
SET turbohybrid.multivector_plain_fallback_candidate_fraction = 0.5;
SET turbohybrid.multivector_candidate_reservoirs = 'conservative'; -- off | conservative | balanced
SET turbohybrid.multivector_per_token_doc_reservoir_k = 1;
SET turbohybrid.multivector_coverage_reservoir_k = 10;
SET turbohybrid.multivector_bm25_candidate_injection = 'off'; -- off | hybrid_only | dense_with_text
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
requires `multivector_graph = document_nodes`; it uses the document
representative single-vector graph for admission and exact-reranks admitted
documents with full MaxSim. `exact_token_scan`, `exact_doc_scan`, and
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
WITH (multivector_graph = token_nodes)
```

`token_nodes` is the default compatibility storage mode. It preserves existing
behavior and reports `multivector_graph_mode = token_nodes` in
`turbohybrid_index_stats()`. Explicit `document_nodes` indexes store one graph
node per heap document and a versioned float32 multivector sidecar. Build-time
edge selection uses symmetrized document MaxSim. Non-exhaustive scans traverse
document graph adjacency, score visited candidates by exact float32 sidecar
MaxSim, and exact-rerank heap multivectors. Near-exhaustive scans use the exact
sidecar scan. `multivector_doc_graph_warning` reports
`document_node_f32_sidecar_graph_traversal` or
`document_node_f32_sidecar_exact_scan`; compact quantized document scoring is
still future work. Missing or malformed document-node sidecar metadata fails
with REINDEX guidance instead of silently falling back to token-node storage.

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

Exact heap rerank:

```sql
SET turbohybrid.multivector_exact_rerank = 'topk'; -- off | topk
SET turbohybrid.multivector_exact_rerank_k = 100;
```

Safety caps:

```sql
SET turbohybrid.multivector_max_doc_vectors = 256;
SET turbohybrid.multivector_max_query_vectors = 64;
SET turbohybrid.multivector_max_dim = 4096;
SET turbohybrid.multivector_max_accumulator_mb = 64;
```

The exact rerank path fetches the original `turbohybrid_multivector` heap value
for a bounded document prefix and computes f32 MaxSim. It does not store a full
float copy of each subvector in the index by default.

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
- `multivector_doc_graph_prototype_enabled`
- `multivector_doc_graph_nodes`
- `multivector_doc_graph_docs_scored`
- `multivector_doc_graph_edges_visited`
- `multivector_doc_graph_candidates`
- `multivector_doc_graph_quantized_scores`
- `multivector_doc_graph_heap_fetches`
- `multivector_doc_graph_exact_rerank_docs`
- `multivector_doc_graph_warning`
- `multivector_exact_rerank_enabled`
- `multivector_exact_rerank_docs`
- `multivector_exact_rerank_pairs`
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
