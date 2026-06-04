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
  colbert multivector_cosine_turbohybrid_ops
);
```

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
  colbert multivector_cosine_turbohybrid_ops,
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
Score-level fusion modes for multivector are not the supported path today.

## Tuning Knobs

Approximate candidate collection:

```sql
SET turbohybrid.multivector_subvector_k = 100;
SET turbohybrid.multivector_unique_docs_per_token = 100;
SET turbohybrid.multivector_max_raw_hits_per_token = 400;
SET turbohybrid.multivector_doc_candidate_k = 100;
```

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
- `multivector_unique_docs`
- `multivector_duplicate_doc_hits`
- `multivector_maxsim_updates`
- `multivector_doc_candidates`
- `multivector_exact_rerank_enabled`
- `multivector_exact_rerank_docs`
- `multivector_exact_rerank_pairs`
- `multivector_exact_kernel`
- `multivector_accumulator_kind`
- `multivector_memory_bytes_estimate`

## Limitations

- This is an alpha feature and the storage format is not a compatibility
  promise.
- Only the cosine multivector operator class is exposed today, using dot-product
  MaxSim semantics over the stored token vectors.
- `turbohybrid_query` cannot contain both `vector_query` and
  `multivector_query`.
- Incremental insert/update into a multivector TurboHybrid index expands the new
  tuple into one graph subnode per document vector. Hybrid indexes append one
  BM25 delta for the inserted document when the lexical key is non-null.
- Hybrid multivector search currently supports document-level RRF. Treat other
  fusion modes as unsupported unless tests and docs for that mode say otherwise.
- SIMD MaxSim kernels are optional; scalar is always available and defines the
  semantics.
