# llama_embed Companion Extension

`llama_embed` is a companion PostgreSQL extension for generating dense and
multivector embeddings with llama.cpp/libllama and searching them with
`pgturbohybrid`. It started as the ColBERT-only `pg_colbert_llama` extension;
that extension name and the `colbert_*` SQL functions remain compatibility
entry points, but the public extension for new installs is `llama_embed`.

It is intentionally separate from the `pgturbohybrid` access method:

```text
llama_embed
  -> loads GGUF models through llama.cpp/libllama
  -> tokenizes text
  -> runs embedding models
  -> returns vector, vector[], jsonb, or multivector-compatible values

pgturbohybrid
  -> stores multivector columns
  -> expands token vectors into graph subnodes
  -> runs approximate MaxSim search
  -> exact-reranks bounded document candidates
  -> optionally fuses dense MaxSim with BM25
```

Keeping the split avoids making the index AM depend on llama.cpp. Model loading,
GGUF conversion, hardware backends, and embedding policy can evolve in
`llama_embed` while `pgturbohybrid` stays focused on storage and search.

For copyable SQL examples that show dense embeddings with pgvector and
multivector embeddings with `pgturbohybrid`, see
[`extensions/pg_colbert_llama/examples/README.md`](../extensions/pg_colbert_llama/examples/README.md).

## Runtime Requirements

Install the dependencies first:

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pgturbohybrid;
CREATE EXTENSION llama_embed;
```

`llama_embed` exposes:

```sql
llama_embed(model text, input text, options jsonb DEFAULT '{}')              RETURNS jsonb
llama_embed_vector(model text, input text, options jsonb DEFAULT '{}')       RETURNS vector
llama_embed_vector_batch(model text, inputs text[], options jsonb DEFAULT '{}') RETURNS vector[]
llama_embed_tokens(model text, input text, options jsonb DEFAULT '{}')       RETURNS vector[]
llama_embed_mv(model text, input text, options jsonb DEFAULT '{}')           RETURNS multivector
llama_embed_mv_batch(model text, inputs text[], options jsonb DEFAULT '{}')  RETURNS multivector[]
llama_embed_model_info(model text)                                          RETURNS jsonb
```

The legacy `pg_colbert_llama` extension exposes compatibility functions:

```sql
colbert(model text, input text)         RETURNS jsonb
colbert_vectors(model text, input text) RETURNS vector[]
colbert_float4(model text, input text)  RETURNS real[]
colbert_dim(model text, input text)     RETURNS int4
colbert_mv(model text, input text)      RETURNS turbohybrid_multivector
colbert_model_info(model text)          RETURNS jsonb
```

The public `llama_embed_mv` and compatibility `colbert_mv` paths use
`turbohybrid_multivector_from_float4(real[], dim int)` when available and falls
back to `turbohybrid_multivector(vector[])` for older `pgturbohybrid` builds.
Store the result in the public `multivector` column type for new schemas.

## Model Strings

SQL callers pass aliases, not arbitrary filesystem paths. The generic
`llama_embed_*` API accepts plain aliases:

```sql
SELECT llama_embed_vector('sauerkraut-modern', 'Was ist PostgreSQL?');
SELECT llama_embed_mv('sauerkraut-modern', body, '{"mode": "tokens"}'::jsonb);
```

The ColBERT compatibility API accepts role-qualified aliases:

```text
<alias>:query
<alias>:doc
```

Aliases may contain only letters, digits, `_`, `-`, and `.`. They cannot start
with `.`, contain `..`, or contain path separators. The resolved model path is:

```text
${pg_colbert_llama.model_dir}/${alias}.gguf
```

Optional `pg_colbert_llama.allowed_models` restricts aliases to an
administrator-provided comma-separated allowlist. The GUC namespace remains
`pg_colbert_llama.*` in this compatibility slice.

## Modes And Pooling

`llama_embed` supports two output modes:

```text
mode = dense   -- one pgvector vector, pooled by llama.cpp
mode = tokens  -- one vector per retained token/subvector
```

Dense mode defaults to mean pooling and rejects `pooling = none`. Token mode
uses per-token embeddings and rejects dense pooling. Supported pooling options
are `mean`, `cls`, `last`, and `rank`, where backend support depends on the
loaded llama.cpp model.

## GUCs

```text
pg_colbert_llama.model_dir = '/var/lib/postgresql/colbert-models'
pg_colbert_llama.threads = 4
pg_colbert_llama.n_ctx = 512
pg_colbert_llama.n_batch = 512
pg_colbert_llama.n_gpu_layers = 0
pg_colbert_llama.cache_size = 2
pg_colbert_llama.query_prefix = '[Q] '
pg_colbert_llama.document_prefix = '[D] '
pg_colbert_llama.max_query_vectors = 64
pg_colbert_llama.max_doc_vectors = 256
pg_colbert_llama.require_normalized = on
pg_colbert_llama.expected_dim = 128
pg_colbert_llama.allowed_models = ''
```

For ModernColBERT GGUFs, set `pg_colbert_llama.expected_dim` from the
registered pgturbohybrid model profile or pass an explicit dimension for an
unregistered export. `turbohybrid_multivector_model_info(model_name)` exposes
the extension-side profiles used by DBpedia benchmark `--expected-dim auto`;
current GTE/Reason ModernColBERT and Sauerkraut validation profiles are 128d,
while other ColBERT-style models such as AnswerAI small and Jina 96/64 variants
are not. The GGUF must include the ColBERT projection, not only the ModernBERT
hidden states. Use the small 15m validation pair by default:
`VAGOsolutions/SauerkrautLM-Multi-ColBERT-15m` for PyLate parity and
`johannhartmann/SauerkrautLM-Multi-ColBERT-15m-GGUF` for the PostgreSQL
llama.cpp path. Run
`extensions/pg_colbert_llama/tools/canonicalize_pg_colbert_gguf.py` before
using the GGUF with the llama engine. The canonicalizer rewrites the
repository's `pg_colbert_v1` GGUF metadata and tensors into llama.cpp BERT
format, emits CPU-compatible `f32` tensors by default, and writes the ColBERT
projection to a sidecar named `<model>.gguf.colbert_proj`.

`require_normalized` defaults to `on` because
`multivector_maxsim_ip_turbohybrid_ops` uses raw dot-product MaxSim. The engine
must return L2-normalized token vectors for cosine-like ColBERT scoring.

## Table And Ingest

Use one row per passage or chunk:

```sql
CREATE TABLE passages (
  id       bigserial PRIMARY KEY,
  doc_id   bigint NOT NULL,
  chunk_no int NOT NULL,
  body     text NOT NULL,
  body_tsv tsvector GENERATED ALWAYS AS (
    to_tsvector('simple', body)
  ) STORED,
  colbert  multivector NOT NULL
);
```

Ingest should be explicit, not a generated column, because model inference
depends on model bytes, llama.cpp version, GUCs, hardware backend, and extension
version:

```sql
INSERT INTO passages (doc_id, chunk_no, body, colbert)
SELECT doc_id, chunk_no, body, llama_embed_mv(
  'sauerkraut-modern',
  body,
  '{"mode": "tokens", "prefix": "[D] "}'::jsonb
)
FROM staging_passages;
```

## Indexes

Dense-only ColBERT MaxSim:

```sql
CREATE INDEX passages_colbert_idx
ON passages USING turbohybrid (
  colbert multivector_maxsim_ip_turbohybrid_ops
);
```

Hybrid ColBERT + BM25:

```sql
CREATE INDEX passages_hybrid_idx
ON passages USING turbohybrid (
  colbert  multivector_maxsim_ip_turbohybrid_ops,
  body_tsv bm25_tsvector_turbohybrid_ops
);
```

Prefer `multivector_maxsim_ip_turbohybrid_ops` for new indexes. It names the
raw dot-product MaxSim semantics. Stored and query token vectors must be
L2-normalized when the model expects cosine-like ColBERT scoring.

## Query

```sql
SELECT id, doc_id, chunk_no, body
FROM passages
ORDER BY colbert <~> turbohybrid_query(
  multivector_query => llama_embed_mv(
    'sauerkraut-modern',
    $1,
    '{"mode": "tokens", "prefix": "[Q] "}'::jsonb
  ),
  text_query        => websearch_to_tsquery('simple', $1),
  fusion            => 'rrf',
  dense_k           => 300,
  bm25_k            => 300,
  final_k           => 20
)
LIMIT 20;
```

The hybrid multivector path is document-level: dense MaxSim candidates are keyed
by heap tuple, BM25 candidates are keyed by heap tuple, and RRF combines
document ranks.

## llama.cpp Engine Contract

The llama engine should:

1. Resolve the alias to an admin-installed local GGUF.
2. Load the model with `llama_model_load_from_file`.
3. Create a context with embeddings enabled.
4. Use a pooling type matching the requested `llama_embed` mode and options.
5. Use non-causal attention for encoder/embedding models.
6. Tokenize `query_prefix || input` or `document_prefix || input` with
   `parse_special = true`.
7. Run the model and read either pooled sequence embeddings for dense mode or
   per-token embeddings for token mode.
8. Require output dimension `pg_colbert_llama.expected_dim` for compatibility
   ColBERT calls or when `expected_dim` is passed explicitly.
9. L2-normalize retained vectors when normalization is requested.
10. Drop special/pad tokens where llama vocab APIs expose them safely.
11. Cap query/doc vectors with the role-specific GUCs.

The default build uses a deterministic stub engine for CI. Build the optional
llama target with:

```sh
make -C extensions/pg_colbert_llama \
  PG_COLBERT_LLAMA_ENGINE=llama \
  LLAMA_CPP_INCLUDE=/path/to/llama.cpp/include \
  LLAMA_CPP_LIB=/path/to/llama.cpp/lib
```

Live tests are gated by `PG_COLBERT_LLAMA_TEST_MODEL` so normal CI never
downloads or requires a GGUF.

Inside the Nix dev shell, use:

```sh
th-colbert-build-stub
th-colbert-test-stub
th-colbert-build-llama
hf download johannhartmann/SauerkrautLM-Multi-ColBERT-15m-GGUF \
  sauerkraut-modern.gguf \
  sauerkraut-modern.gguf.colbert_proj \
  --local-dir .nix-dev/models/colbert-15m

PG_COLBERT_LLAMA_TEST_MODEL="$PWD/.nix-dev/models/colbert-15m/sauerkraut-modern.gguf" \
  th-colbert-live-test
```

The external parity checker compares PostgreSQL output against PyLate and is
also kept out of default CI:

```sh
python extensions/pg_colbert_llama/tools/compare_pylate.py \
  --pg-dsn "$DATABASE_URL" \
  --pg-model-alias sauerkraut-modern:query \
  --model-name-or-path VAGOsolutions/SauerkrautLM-Multi-ColBERT-15m \
  --texts-file texts.txt \
  --role query \
  --model-dir /var/lib/postgresql/colbert-models \
  --ranking-query "red planet" \
  --ranking-docs-file ranking-docs.txt \
  --ranking-expected-index 0
```

For the ranking smoke, put the obviously relevant passage first in
`ranking-docs.txt`. The tool encodes the query with `alias:query`, encodes
candidate passages with `alias:doc`, computes ColBERT MaxSim for both
PostgreSQL and PyLate output, and exits nonzero unless both systems rank the
expected passage first.
