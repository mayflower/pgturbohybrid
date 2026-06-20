# llama_embed Examples

This page shows the two common ways to use `llama_embed` output:

- dense embeddings returned as `vector`, stored and searched with pgvector;
- token or late-interaction embeddings stored in `multivector` columns,
  stored and searched with `pgturbohybrid`.

The examples assume the model alias `sauerkraut-modern` resolves to an
administrator-installed GGUF under `pg_colbert_llama.model_dir`. SQL callers pass
model aliases, not filesystem paths.

## Install Extensions

Dense pgvector search needs `vector` and `llama_embed`:

```sql
CREATE EXTENSION vector;
CREATE EXTENSION llama_embed;
```

Multivector search additionally needs `pgturbohybrid`:

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pgturbohybrid;
CREATE EXTENSION llama_embed;
```

The extension name for new SQL is `llama_embed`. The internal GUC namespace is
currently still `pg_colbert_llama.*`:

```sql
SET pg_colbert_llama.model_dir = '/var/lib/postgresql/embedding-models';
```

## Check The Dense Output Dimension

pgvector columns are usually declared as `vector(n)`, so check the model's dense
output dimension before creating the table:

```sql
SELECT (llama_embed(
  'sauerkraut-modern',
  'dimension check',
  '{"mode": "dense", "pooling": "mean"}'::jsonb
)->>'dim')::int AS dense_dim;
```

Use that value in the `vector(n)` column below.

## Dense Embeddings With pgvector

Use `llama_embed_vector()` when a model should produce one embedding per input
text. The returned value is a normal pgvector `vector` and can be indexed with
pgvector's HNSW or IVFFlat indexes.

```sql
CREATE TABLE dense_documents (
  id        bigserial PRIMARY KEY,
  body      text NOT NULL,
  embedding vector(768) NOT NULL -- replace 768 with your model's dense_dim
);

INSERT INTO dense_documents (body, embedding)
VALUES
  (
    'PostgreSQL stores structured data and supports extensions.',
    llama_embed_vector(
      'sauerkraut-modern',
      'PostgreSQL stores structured data and supports extensions.',
      '{"mode": "dense", "pooling": "mean"}'::jsonb
    )
  ),
  (
    'llama.cpp can run local embedding models from GGUF files.',
    llama_embed_vector(
      'sauerkraut-modern',
      'llama.cpp can run local embedding models from GGUF files.',
      '{"mode": "dense", "pooling": "mean"}'::jsonb
    )
  );

CREATE INDEX dense_documents_embedding_hnsw
ON dense_documents USING hnsw (embedding vector_cosine_ops);

SELECT id, body
FROM dense_documents
ORDER BY embedding <=> llama_embed_vector(
  'sauerkraut-modern',
  'local PostgreSQL extension search',
  '{"mode": "dense", "pooling": "mean"}'::jsonb
)
LIMIT 10;
```

Notes:

- `llama_embed_vector()` is the direct pgvector path.
- Use pgvector operators and opclasses such as `<=>`, `vector_cosine_ops`,
  `vector_l2_ops`, or `vector_ip_ops` according to the model and metric.
- Dense mode rejects `pooling = "none"` because one output vector is required.

## Batch Dense Ingest

For larger imports, stage text first and update embeddings in batches:

```sql
CREATE TABLE dense_staging (
  id   bigserial PRIMARY KEY,
  body text NOT NULL
);

CREATE TABLE dense_indexed (
  id        bigint PRIMARY KEY,
  body      text NOT NULL,
  embedding vector(768) NOT NULL
);

INSERT INTO dense_indexed (id, body, embedding)
SELECT id,
       body,
       llama_embed_vector(
         'sauerkraut-modern',
         body,
         '{"mode": "dense", "pooling": "mean"}'::jsonb
       )
FROM dense_staging
ORDER BY id;
```

`llama_embed_vector_batch(model, text[], options)` is also available when an
application wants to batch inputs explicitly and unpack the returned `vector[]`.

## Multivector Embeddings With pgturbohybrid

Use `llama_embed_mv()` when a ColBERT-style or late-interaction model should
produce multiple token vectors per input. Store those values in the public
`multivector` column type, not a pgvector `vector`.

```sql
CREATE TABLE passages (
  id       bigserial PRIMARY KEY,
  doc_id   bigint NOT NULL,
  chunk_no int NOT NULL,
  body     text NOT NULL,
  colbert  multivector NOT NULL
);

INSERT INTO passages (doc_id, chunk_no, body, colbert)
VALUES
  (
    1,
    0,
    'PostgreSQL extensions can add new index access methods.',
    llama_embed_mv(
      'sauerkraut-modern',
      'PostgreSQL extensions can add new index access methods.',
      '{"mode": "tokens", "prefix": "[D] "}'::jsonb
    )
  ),
  (
    2,
    0,
    'ColBERT uses MaxSim over token-level vectors.',
    llama_embed_mv(
      'sauerkraut-modern',
      'ColBERT uses MaxSim over token-level vectors.',
      '{"mode": "tokens", "prefix": "[D] "}'::jsonb
    )
  );

CREATE INDEX passages_colbert_idx
ON passages USING turbohybrid (
  colbert multivector_maxsim_ip_turbohybrid_ops
);

SELECT id, doc_id, chunk_no, body
FROM passages
ORDER BY colbert <~> turbohybrid_query(
  multivector_query => llama_embed_mv(
    'sauerkraut-modern',
    'how does ColBERT search token vectors?',
    '{"mode": "tokens", "prefix": "[Q] "}'::jsonb
  ),
  dense_k => 100,
  final_k => 10
)
LIMIT 10;
```

Notes:

- `llama_embed_mv()` is the direct `pgturbohybrid` multivector path.
- Prefer `multivector_maxsim_ip_turbohybrid_ops` for new ColBERT-style indexes.
- The final result ordering remains document-level MaxSim over retained
  candidates.
- Token mode rejects dense pooling options because it returns one vector per
  retained token.

## Multivector Plus BM25 In One pgturbohybrid Index

`pgturbohybrid` can combine a multivector key with a BM25 `tsvector` key. This is
still different from storing dense `vector` output in pgvector; the column below
stores `multivector`.

```sql
CREATE TABLE hybrid_passages (
  id       bigserial PRIMARY KEY,
  body     text NOT NULL,
  body_tsv tsvector GENERATED ALWAYS AS (
    to_tsvector('english', body)
  ) STORED,
  colbert  multivector NOT NULL
);

INSERT INTO hybrid_passages (body, colbert)
SELECT body,
       llama_embed_mv(
         'sauerkraut-modern',
         body,
         '{"mode": "tokens", "prefix": "[D] "}'::jsonb
       )
FROM (
  VALUES
    ('PostgreSQL vector search and BM25 can be fused.'),
    ('Late interaction reranking uses exact MaxSim over token vectors.')
) AS input(body);

CREATE INDEX hybrid_passages_idx
ON hybrid_passages USING turbohybrid (
  colbert  multivector_maxsim_ip_turbohybrid_ops,
  body_tsv bm25_tsvector_turbohybrid_ops
);

SELECT id, body
FROM hybrid_passages
ORDER BY colbert <~> turbohybrid_query(
  multivector_query => llama_embed_mv(
    'sauerkraut-modern',
    'postgres hybrid colbert search',
    '{"mode": "tokens", "prefix": "[Q] "}'::jsonb
  ),
  text_query => websearch_to_tsquery('english', 'postgres hybrid colbert search'),
  fusion     => 'rrf',
  dense_k    => 100,
  bm25_k     => 100,
  final_k    => 10
)
LIMIT 10;
```

## Which Function Should I Use?

| Use case | Function | SQL type | Index/search path |
| --- | --- | --- | --- |
| One dense embedding per text | `llama_embed_vector()` | `vector` | pgvector HNSW/IVFFlat |
| Dense batch embedding | `llama_embed_vector_batch()` | `vector[]` | unpack into pgvector rows |
| Token vectors for inspection | `llama_embed_tokens()` | `vector[]` | application-side processing |
| ColBERT/late-interaction search | `llama_embed_mv()` | `multivector` column | `pgturbohybrid` multivector |
| Batch ColBERT ingest | `llama_embed_mv_batch()` | `multivector` columns | unpack into `pgturbohybrid` rows |
