## Strategy

Build a new companion extension:

```text
pg_colbert_llama
```

and use `pgturbohybrid` as the index/search engine.

Do **not** put llama.cpp directly into the pgturbohybrid access method yet. Keep the split:

```text
pg_colbert_llama
  → loads GGUF with llama.cpp/libllama
  → tokenizes text
  → runs ModernBERT/ColBERT embedding
  → returns turbohybrid_multivector

pgturbohybrid
  → stores turbohybrid_multivector
  → expands token vectors into graph subnodes
  → runs approximate MaxSim search
  → exact-reranks bounded candidates
  → optionally fuses with BM25
```

This fits the current pgturbohybrid direction very well. The current docs say `turbohybrid_multivector` is specifically for ColBERT-style late interaction: one row stores multiple token vectors, the graph indexes each token vector as a subnode, and SQL results are aggregated back to document heap rows using MaxSim.

The current docs also now recommend `multivector_maxsim_ip_turbohybrid_ops` for new indexes because it names the actual raw dot-product MaxSim semantics; `multivector_cosine_turbohybrid_ops` remains compatibility-only and assumes vectors are normalized before indexing/querying.

That means the embedding extension must output **L2-normalized 128-dimensional ColBERT token vectors**.

---

# Extension concept

## Extension name

```text
pg_colbert_llama
```

## Dependencies

Runtime:

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pgturbohybrid;
CREATE EXTENSION pg_colbert_llama;
```

Build-time:

```text
PostgreSQL server headers
pgvector headers
pgturbohybrid installed in database
llama.cpp/libllama headers and libraries
```

The current pgturbohybrid Makefile is a normal PGXS extension build, already discovers pgvector headers, and has regression tests wired through `REGRESS`; the companion extension should copy that style instead of changing pgturbohybrid’s build system.

---

# SQL API

Expose three layers.

## 1. Native retrieval function

This is the main function:

```sql
CREATE FUNCTION colbert_mv(model text, input text)
RETURNS turbohybrid_multivector;
```

Usage:

```sql
SELECT colbert_mv('sauerkraut-modern:query', 'Was ist PostgreSQL?');
SELECT colbert_mv('sauerkraut-modern:doc',   body);
```

## 2. Debug JSON function

Keep your original shape:

```sql
CREATE FUNCTION colbert(model text, input text)
RETURNS jsonb;
```

Example output:

```json
{
  "model": "sauerkraut-modern",
  "role": "query",
  "dim": 128,
  "count": 37,
  "normalized": true,
  "token_ids": [123, 456, 789],
  "vectors": [[0.0123, -0.0441, 0.0912]]
}
```

## 3. Compatibility vector-array function

Useful for tests and for constructing `turbohybrid_multivector` through the existing public pgturbohybrid constructor:

```sql
CREATE FUNCTION colbert_vectors(model text, input text)
RETURNS vector[];
```

Then the first implementation of `colbert_mv` can simply be:

```sql
CREATE FUNCTION colbert_mv(model text, input text)
RETURNS turbohybrid_multivector
LANGUAGE sql
STABLE
STRICT
PARALLEL RESTRICTED
AS $$
  SELECT turbohybrid_multivector(colbert_vectors(model, input));
$$;
```

That avoids depending on pgturbohybrid’s private C struct ABI in v0.1.

The current pgturbohybrid SQL surface already exposes `turbohybrid_multivector(vector[])`, `turbohybrid_multivector_dims`, `turbohybrid_multivector_count`, `turbohybrid_multivector_subvector`, `turbohybrid_multivector_to_vector_array`, and MaxSim helpers.

---

# How to use it with pgturbohybrid

## Table

Use one row per passage/chunk, not one row per giant document.

```sql
CREATE TABLE passages (
  id       bigserial PRIMARY KEY,
  doc_id   bigint NOT NULL,
  chunk_no int NOT NULL,
  body     text NOT NULL,
  body_tsv tsvector GENERATED ALWAYS AS (
    to_tsvector('simple', body)
  ) STORED,
  colbert  turbohybrid_multivector NOT NULL
);
```

## Ingest

```sql
INSERT INTO passages (doc_id, chunk_no, body, colbert)
SELECT
  doc_id,
  chunk_no,
  body,
  colbert_mv('sauerkraut-modern:doc', body)
FROM staging_passages;
```

Do not use a generated column for model inference. Model inference depends on GGUF bytes, llama.cpp version, GUCs, hardware backend, and extension version, so it should be an explicit ingest step or controlled trigger.

## Dense-only ColBERT index

```sql
CREATE INDEX passages_colbert_idx
ON passages USING turbohybrid (
  colbert multivector_maxsim_ip_turbohybrid_ops
);
```

## Hybrid ColBERT + BM25 index

```sql
CREATE INDEX passages_hybrid_idx
ON passages USING turbohybrid (
  colbert  multivector_maxsim_ip_turbohybrid_ops,
  body_tsv bm25_tsvector_turbohybrid_ops
);
```

## Query

```sql
SELECT id, doc_id, chunk_no, body
FROM passages
ORDER BY colbert <~> turbohybrid_query(
  multivector_query => colbert_mv('sauerkraut-modern:query', $1),
  text_query        => websearch_to_tsquery('simple', $1),
  fusion            => 'rrf',
  dense_k           => 300,
  bm25_k            => 300,
  final_k           => 20
)
LIMIT 20;
```

The current multivector docs say hybrid multivector + BM25 is document-level: dense MaxSim candidates are keyed by heap tuple, BM25 candidates are keyed by heap tuple, and RRF combines document ranks. They also document `weighted` support for normalized MaxSim plus BM25, while `fast_weighted` and `calibrated` are not supported for multivector hybrid scans yet.

---

# llama.cpp engine concept

Use **libllama** as the BERT/ModernBERT GGUF runtime.

The engine should:

```text
1. Resolve model alias.
2. Load local GGUF through llama_model_load_from_file.
3. Create llama_context with embeddings=true.
4. Use pooling_type = LLAMA_POOLING_TYPE_NONE.
5. Use non-causal attention for encoder/embedding models.
6. Tokenize "[Q] " + input or "[D] " + input with parse_special=true.
7. Run the model.
8. Read per-token embeddings with llama_get_embeddings_ith.
9. Validate dim == 128.
10. L2-normalize each token vector.
11. Drop special/pad/skiplist tokens.
12. Cap query/doc vector counts.
13. Return vector[] or turbohybrid_multivector.
```

The current llama C API has model/context loading through `llama_model_load_from_file` and `llama_init_from_model`, exposes `llama_context_params.embeddings`, `pooling_type`, and `attention_type`, and reports output dimensionality via `llama_model_n_embd_out`. ([GitHub][1])

For tokenization, use `llama_tokenize(..., add_special, parse_special)`. The API explicitly documents `parse_special` for tokenizing special/control tokens instead of treating them as plaintext, which matters for `[Q]` and `[D]`-style ColBERT prefixes. ([GitHub][1])

For token-vector extraction, llama.cpp exposes `llama_get_embeddings_ith`; the header says token embeddings are available when pooling is `LLAMA_POOLING_TYPE_NONE`, and the embedding example uses `llama_get_embeddings_ith` in that mode. ([GitHub][1]) The official embedding example also normalizes embeddings before output, which matches the pgturbohybrid MaxSim requirement for normalized token vectors when using dot-product MaxSim. ([GitHub][2])

For conversion, make sure the GGUF includes the ColBERT/SentenceTransformers dense projection. The llama.cpp converter has `--sentence-transformers-dense-modules`, and the help text says dense modules are not included by default. ([GitHub][3])

The underlying Sauerkraut ModernColBERT model family is PyLate/ColBERT late interaction, maps sentences/paragraphs to sequences of 128-dimensional vectors, and uses MaxSim; its model card shows `ModernBertModel` followed by `Dense(768 -> 128 dim, no bias)`. ([huggingface.co][4])

---

# Model aliasing

Use model aliases, not arbitrary filesystem paths from SQL.

```sql
SELECT colbert_mv('sauerkraut-modern:query', '...');
SELECT colbert_mv('sauerkraut-modern:doc', '...');
```

Suggested GUCs:

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
```

Model string grammar:

```text
<alias>:query
<alias>:doc
```

Alias resolution:

```text
sauerkraut-modern
  → ${pg_colbert_llama.model_dir}/sauerkraut-modern.gguf
```

Reject:

```text
model strings with ../
absolute paths from non-superusers
missing role suffix
GGUF output dim != 128
non-finite embeddings
zero retained token vectors
```

---

# pgturbohybrid limits to respect

Current pgturbohybrid hard/default multivector constants are:

```text
PGTURBOHYBRID_MULTIVECTOR_MAX_COUNT = 4096
default max doc vectors             = 256
default max query vectors           = 64
```



So the extension should default to:

```text
query:    retain <= 64 token vectors
document: retain <= 256 token vectors
dim:      128
```

pgturbohybrid also exposes the relevant runtime caps:

```sql
SET turbohybrid.multivector_max_doc_vectors = 256;
SET turbohybrid.multivector_max_query_vectors = 64;
SET turbohybrid.multivector_max_dim = 4096;
SET turbohybrid.multivector_max_accumulator_mb = 64;
```



---

# Recommended implementation sequence

## Phase 1: separate companion extension, public constructor path

```text
pg_colbert_llama
  colbert_vectors(...) RETURNS vector[]
  colbert_mv(...)      RETURNS turbohybrid_multivector via SQL wrapper
  colbert(...)         RETURNS jsonb
```

This is slower than direct native construction, but it uses only pgturbohybrid’s public SQL API.

## Phase 2: direct native multivector construction

Add one stable pgturbohybrid function:

```sql
turbohybrid_multivector_from_float4(values real[], dim int)
```

or:

```sql
turbohybrid_multivector_from_bytea(values bytea, count int, dim int)
```

Then `pg_colbert_llama` can avoid building hundreds of intermediate pgvector values.

## Phase 3: optional integration into pgturbohybrid tree

Only after the extension is stable, add optional docs or a contrib-style integration in the pgturbohybrid repo. I would still keep the llama.cpp dependency optional.

---

# Codex prompts

Below are copy-pasteable prompts for stepwise implementation. Run them one at a time.

---

## Prompt 1 — inspect current pgturbohybrid state and write design doc

```text
You are working in the mayflower/pgturbohybrid repository on branch main.

Goal:
Create a design document for a companion PostgreSQL extension named pg_colbert_llama that embeds text with llama.cpp/libllama and returns pgturbohybrid-compatible ColBERT multivectors.

Tasks:
1. Inspect the current repository, especially:
   - README.md
   - docs/multivector-late-interaction.md
   - sql/pgturbohybrid--0.1.0.sql
   - src/pgturbohybrid_multivector.h
   - src/pgturbohybrid_multivector.c
   - test/sql/pgturbohybrid_multivector.sql
   - Makefile
2. Add docs/colbert-llama-extension.md.
3. The doc must explain:
   - pg_colbert_llama is a companion extension, not part of the index AM.
   - It depends on vector and pgturbohybrid at runtime.
   - It links llama.cpp/libllama at build/runtime.
   - It exposes:
     colbert(model text, input text) RETURNS jsonb
     colbert_vectors(model text, input text) RETURNS vector[]
     colbert_mv(model text, input text) RETURNS turbohybrid_multivector
   - It uses model aliases like sauerkraut-modern:query and sauerkraut-modern:doc.
   - It returns 128-dimensional L2-normalized token vectors.
   - pgturbohybrid indexes should use multivector_maxsim_ip_turbohybrid_ops.
   - The first implementation should use turbohybrid_multivector(colbert_vectors(...)).
   - Later direct construction can avoid vector[] overhead.
4. Include SQL usage examples:
   - table schema
   - ingest
   - dense-only multivector index
   - hybrid multivector + BM25 index
   - query using turbohybrid_query(multivector_query => colbert_mv(...)).
5. Add a short README.md link to this doc under the Multivector section if appropriate.

Tests:
- Documentation-only change; run markdown lint if available.
- Do not change C code yet.
```

---

## Prompt 2 — add companion extension skeleton

```text
You are working in mayflower/pgturbohybrid.

Goal:
Add a separate PGXS companion extension under extensions/pg_colbert_llama without changing pgturbohybrid behavior.

Create:
extensions/pg_colbert_llama/Makefile
extensions/pg_colbert_llama/pg_colbert_llama.control
extensions/pg_colbert_llama/sql/pg_colbert_llama--0.1.0.sql
extensions/pg_colbert_llama/src/pg_colbert_llama.c
extensions/pg_colbert_llama/src/colbert_engine.h
extensions/pg_colbert_llama/src/colbert_engine_stub.c
extensions/pg_colbert_llama/test/sql/pg_colbert_llama.sql
extensions/pg_colbert_llama/test/expected/pg_colbert_llama.out
extensions/pg_colbert_llama/README.md

Requirements:
1. The extension must require PostgreSQL 14+ in SQL, mirroring the pgturbohybrid style.
2. The SQL install script must check that vector and pgturbohybrid are installed.
3. Expose these C functions initially backed by a deterministic stub engine:
   - colbert(model text, input text) RETURNS jsonb
   - colbert_vectors(model text, input text) RETURNS vector[]
   - colbert_model_info(model text) RETURNS jsonb
4. Expose this SQL wrapper:
   - colbert_mv(model text, input text) RETURNS turbohybrid_multivector
     implemented as SELECT turbohybrid_multivector(colbert_vectors(model, input)).
5. Stub behavior:
   - For model ending in :query, return 2 vectors of dim 4.
   - For model ending in :doc, return 3 vectors of dim 4.
   - Values must be deterministic, finite, and L2-normalized.
   - Reject model strings without :query or :doc.
6. The Makefile should copy pgturbohybrid's PGXS style and pgvector header discovery style where useful.
7. Add regression tests that:
   - CREATE EXTENSION vector;
   - CREATE EXTENSION pgturbohybrid;
   - CREATE EXTENSION pg_colbert_llama;
   - Verify colbert_model_info returns JSON with engine='stub'.
   - Verify colbert_vectors returns vector[].
   - Verify colbert_mv returns turbohybrid_multivector.
   - Verify turbohybrid_multivector_dims(colbert_mv(...)) = 4.
   - Verify query count = 2 and doc count = 3.
   - Verify turbohybrid_multivector_maxsim(query_mv, doc_mv) runs.

Do not link llama.cpp yet.
Keep code simple and warning-clean.
```

---

## Prompt 3 — add model string parser and GUCs

```text
Goal:
Implement robust model alias parsing and pg_colbert_llama GUCs.

Files:
extensions/pg_colbert_llama/src/pg_colbert_llama.c
extensions/pg_colbert_llama/src/colbert_engine.h
extensions/pg_colbert_llama/src/colbert_engine_stub.c
extensions/pg_colbert_llama/test/sql/pg_colbert_llama.sql
extensions/pg_colbert_llama/test/expected/pg_colbert_llama.out

Tasks:
1. Add _PG_init() for pg_colbert_llama.
2. Define GUCs:
   - pg_colbert_llama.model_dir, string, default '/var/lib/postgresql/colbert-models'
   - pg_colbert_llama.threads, int, default 4, min 1
   - pg_colbert_llama.n_ctx, int, default 512
   - pg_colbert_llama.n_batch, int, default 512
   - pg_colbert_llama.n_gpu_layers, int, default 0
   - pg_colbert_llama.max_query_vectors, int, default 64
   - pg_colbert_llama.max_doc_vectors, int, default 256
   - pg_colbert_llama.query_prefix, string, default '[Q] '
   - pg_colbert_llama.document_prefix, string, default '[D] '
3. Implement a parser:
   input: 'alias:query' or 'alias:doc'
   output:
     alias
     role enum
     max_vectors from role GUC
     prefix from role GUC
4. Reject:
   - missing role
   - unknown role
   - empty alias
   - alias containing '/'
   - alias containing '..'
   - alias longer than a safe fixed limit
5. Update stub engine to use the parser output.
6. Add tests for invalid model names and GUC changes.
7. All SQL functions should be STRICT where appropriate and PARALLEL RESTRICTED.

Do not add llama.cpp yet.
```

---

## Prompt 4 — add vector[] construction using pgvector directly

```text
Goal:
Make colbert_vectors construct real pgvector vector[] values efficiently instead of going through text.

Tasks:
1. Inspect pgturbohybrid's src/pgturbohybrid_vector_compat.h and src/pgturbohybrid_vector_compat.c.
2. In pg_colbert_llama, add a minimal pgvector compatibility layer:
   - Find vector type OID.
   - Construct Vector varlena values directly.
   - Respect vector dimensions and finite values.
3. Implement colbert_vectors(model text, input text) using the engine output:
   engine output:
     int32 count
     int32 dim
     float *values row-major
   SQL output:
     vector[]
4. Keep stub deterministic.
5. Add regression tests:
   - array_length(colbert_vectors(...), 1)
   - colbert_vectors(...)[1]::text matches expected vector text format approximately
   - turbohybrid_multivector(colbert_vectors(...)) roundtrips through turbohybrid_multivector_to_vector_array.
6. Avoid depending on pgturbohybrid private headers from the companion extension.
7. Keep memory allocated in PostgreSQL memory contexts.

Do not link llama.cpp yet.
```

---

## Prompt 5 — add optional llama.cpp build path

```text
Goal:
Add optional llama.cpp/libllama integration behind a Makefile flag while keeping the stub build as the default for CI.

Requirements:
1. In extensions/pg_colbert_llama/Makefile add:
   PG_COLBERT_LLAMA_ENGINE ?= stub
   Supported values:
     stub
     llama
2. When PG_COLBERT_LLAMA_ENGINE=stub:
   build colbert_engine_stub.c only.
3. When PG_COLBERT_LLAMA_ENGINE=llama:
   build colbert_engine_llama.cpp and link against libllama.
4. Add variables:
   LLAMA_CPP_INCLUDE ?=
   LLAMA_CPP_LIB ?=
   LLAMA_CPP_LDFLAGS ?=
5. For llama build:
   - add -I$(LLAMA_CPP_INCLUDE)
   - add -L$(LLAMA_CPP_LIB)
   - link -lllama and any required ggml libraries
   - link C++ standard library correctly
6. Add a compile-time macro:
   -DPG_COLBERT_LLAMA_ENGINE_LLAMA=1 or -DPG_COLBERT_LLAMA_ENGINE_STUB=1
7. Add README build examples:
   make
   make PG_COLBERT_LLAMA_ENGINE=llama LLAMA_CPP_INCLUDE=/path/include LLAMA_CPP_LIB=/path/lib
8. Stub regression tests must remain default.

Do not implement llama inference yet; colbert_engine_llama.cpp may return a clear "llama engine not implemented" error if selected.
```

---

## Prompt 6 — implement llama model loading and cache

```text
Goal:
Implement llama.cpp model loading for PG_COLBERT_LLAMA_ENGINE=llama.

Files:
extensions/pg_colbert_llama/src/colbert_engine_llama.cpp
extensions/pg_colbert_llama/src/colbert_engine.h
extensions/pg_colbert_llama/src/pg_colbert_llama.c

Tasks:
1. Use llama_backend_init() once per backend process.
2. Resolve alias to:
   ${pg_colbert_llama.model_dir}/${alias}.gguf
3. Load model with llama_model_load_from_file.
4. Create context with llama_init_from_model.
5. Context params:
   embeddings = true
   pooling_type = LLAMA_POOLING_TYPE_NONE
   attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL
   n_ctx = pg_colbert_llama.n_ctx
   n_batch = pg_colbert_llama.n_batch
   n_ubatch = pg_colbert_llama.n_batch
   n_threads and n_threads_batch from pg_colbert_llama.threads
6. Model params:
   use_mmap = true where supported
   n_gpu_layers from pg_colbert_llama.n_gpu_layers
7. Maintain a backend-local cache keyed by alias plus resolved path.
8. Cache size comes from pg_colbert_llama.cache_size if already added; otherwise add that GUC now.
9. On cache eviction, free llama_context and llama_model.
10. Do not allow C++ exceptions to cross PostgreSQL C frames.
11. Convert engine errors into PostgreSQL ERROR messages in the C wrapper.
12. Add colbert_model_info(model) fields for llama engine:
   engine
   alias
   role
   path
   n_ctx_train
   n_embd_out
   n_layer
   n_head
   loaded_from_cache
13. Add tests that still pass in stub mode.
14. Add a TAP or manual test script that only runs with PG_COLBERT_LLAMA_TEST_MODEL set.

Do not implement actual embedding yet.
```

---

## Prompt 7 — implement llama tokenization and embedding extraction

```text
Goal:
Implement actual llama.cpp embedding for encoder/embedding GGUF models.

Tasks:
1. In colbert_engine_llama.cpp, implement encode(model_role, input).
2. Build text as:
   role_prefix + input
   where role_prefix is from GUC:
     query -> pg_colbert_llama.query_prefix
     doc   -> pg_colbert_llama.document_prefix
3. Tokenize with llama_tokenize using:
   add_special = true
   parse_special = true
4. Enforce:
   tokens not empty
   tokens <= n_ctx/n_batch
5. Create llama_batch with one sequence.
6. Mark all retained token positions as output-enabled.
7. Clear previous llama memory before each encode.
8. Run the model.
   Prefer llama_encode for encoder-only models if the available llama API/model indicates encoder mode.
   Otherwise follow llama.cpp's embedding example pattern.
9. Read token embeddings with llama_get_embeddings_ith for pooling_type NONE.
10. Use llama_model_n_embd_out(model) as output dimension.
11. For v0.1, hard-fail unless n_embd_out == 128.
12. Copy embeddings into engine-owned output memory.
13. L2-normalize each token vector after copying.
14. Drop obvious special/pad tokens if llama vocab APIs expose them safely; otherwise keep a TODO and expose token ids in JSON for validation.
15. Cap output count:
   query -> pg_colbert_llama.max_query_vectors
   doc   -> pg_colbert_llama.max_doc_vectors
16. Return:
   count
   dim
   token_ids
   row-major float values
17. colbert(...) JSON should include:
   engine
   alias
   role
   dim
   count
   normalized
   token_ids
   vectors
18. Add live TAP/manual test gated by PG_COLBERT_LLAMA_TEST_MODEL:
   - CREATE EXTENSION vector;
   - CREATE EXTENSION pgturbohybrid;
   - CREATE EXTENSION pg_colbert_llama;
   - SET pg_colbert_llama.model_dir to the directory of PG_COLBERT_LLAMA_TEST_MODEL;
   - SELECT turbohybrid_multivector_dims(colbert_mv('alias:query','test')) = 128;
   - SELECT turbohybrid_multivector_count(...) > 0;
   - Verify all vectors are finite using colbert(...) JSON or colbert_vectors.
19. Stub tests must still pass by default.
```

---

## Prompt 8 — validate ColBERT projection and GGUF metadata

```text
Goal:
Make the extension fail clearly when the GGUF is not a usable ColBERT GGUF.

Tasks:
1. In llama engine model load:
   - check llama_model_n_embd_out(model)
   - require 128 by default
2. Add GUC:
   pg_colbert_llama.expected_dim = 128
3. colbert_model_info must report:
   - n_embd_out
   - expected_dim
   - projection_status:
     'ok'
     'missing_or_unexpected_dim'
4. If n_embd_out != expected_dim:
   ERROR with hint:
   "The GGUF must include the ColBERT dense projection. Reconvert with llama.cpp convert_hf_to_gguf.py --sentence-transformers-dense-modules or provide a GGUF whose embedding output is 128-dimensional."
5. Add documentation explaining why this matters for ModernColBERT:
   ModernBERT hidden states are not enough; ColBERT needs Dense(768 -> 128) output.
6. Add tests in stub mode by allowing expected_dim=4.
7. Live test should assert expected_dim=128 for real models.
```

---

## Prompt 9 — add exact pgturbohybrid integration test

```text
Goal:
Add a live integration test that proves pg_colbert_llama output can be indexed and queried by pgturbohybrid.

Create:
extensions/pg_colbert_llama/test/t/002_live_pgturbohybrid_multivector.pl

Requirements:
1. Skip unless PG_COLBERT_LLAMA_TEST_MODEL is set.
2. Create database and extensions:
   vector
   pgturbohybrid
   pg_colbert_llama
3. Configure model_dir and other GUCs.
4. Create table:
   passages(id int primary key, body text, body_tsv tsvector, colbert turbohybrid_multivector)
5. Insert 3 small documents using colbert_mv('alias:doc', body).
6. Create index:
   CREATE INDEX ... USING turbohybrid
   (colbert multivector_maxsim_ip_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops);
7. Query:
   ORDER BY colbert <~> turbohybrid_query(
     multivector_query => colbert_mv('alias:query', 'red planet'),
     text_query => websearch_to_tsquery('simple', 'red planet'),
     fusion => 'rrf',
     dense_k => 20,
     bm25_k => 20,
     final_k => 3
   )
8. Assert:
   - query returns rows
   - turbohybrid_last_scan_stats()->>'multivector_enabled' = 'true'
   - stats show dense branch used
   - stats show index_used true when enable_seqscan = off
9. Keep this test out of normal make installcheck unless the env var is set.
```

---

## Prompt 10 — add benchmark script

```text
Goal:
Add a small benchmark script for local measurement of embedding and search latency.

Create:
extensions/pg_colbert_llama/bench/colbert_ingest_search.sql
extensions/pg_colbert_llama/bench/README.md

Benchmark must measure:
1. document encoding throughput:
   SELECT count(*), avg elapsed per colbert_mv('alias:doc', body)
2. query encoding latency:
   SELECT colbert_mv('alias:query', query)
3. dense-only pgturbohybrid multivector search:
   ORDER BY colbert <~> turbohybrid_query(multivector_query => ...)
4. hybrid pgturbohybrid search:
   multivector_query + text_query + fusion='rrf'
5. Diagnostics:
   SELECT turbohybrid_last_scan_stats();
   SELECT turbohybrid_estimate_memory(index);
6. Include recommended GUC sweep:
   turbohybrid.multivector_subvector_k
   turbohybrid.multivector_unique_docs_per_token
   turbohybrid.multivector_max_raw_hits_per_token
   turbohybrid.multivector_adaptive_widening
   turbohybrid.multivector_doc_candidate_k
   turbohybrid.multivector_exact_rerank_k

Do not commit benchmark output.
```

---

## Prompt 11 — add safe filesystem/model policy

```text
Goal:
Harden model path handling.

Tasks:
1. Model argument must be alias:role only.
2. Alias may contain only [A-Za-z0-9_.-].
3. Alias must not start with '.'.
4. Alias must not contain slash or backslash.
5. Resolved path is model_dir + '/' + alias + '.gguf'.
6. Use canonicalization where safe.
7. Reject paths outside model_dir.
8. Add optional GUC:
   pg_colbert_llama.allowed_models = ''
   If non-empty, it is a comma-separated allowlist of aliases.
9. Add tests:
   - '../x:query' rejected
   - '/tmp/x:query' rejected
   - 'x/y:query' rejected
   - '.hidden:query' rejected
   - allowed_models blocks unknown aliases
10. Update docs with security model:
   - no model download from SQL
   - no arbitrary paths
   - model files must be installed by admin
```

---

## Prompt 12 — add direct pgturbohybrid raw-float constructor

This one changes pgturbohybrid itself and is optional, but it is the right performance bridge.

```text
Goal:
Add a stable pgturbohybrid constructor for multivectors from raw float data, so embedding extensions do not need to build vector[].

Implement in pgturbohybrid core:
CREATE FUNCTION turbohybrid_multivector_from_float4(
  values real[],
  dim int4
) RETURNS turbohybrid_multivector
AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_from_float4'
LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

Semantics:
1. values is a flat row-major real[].
2. dim must be > 0.
3. array length must be divisible by dim.
4. count = array_length / dim.
5. count must pass existing multivector count checks.
6. dim must pass existing multivector dim checks.
7. every value must be finite.
8. Store values exactly in PgturbohybridMultiVector.
9. Reuse existing PgturbohybridMultiVectorSize and validation helpers.

Tests:
1. turbohybrid_multivector_from_float4(ARRAY[1,0,0,1]::real[], 2)
   has dim 2, count 2.
2. It equals or roundtrips similarly to:
   turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector])
3. Reject:
   - dim 0
   - length not divisible by dim
   - empty array
   - NULL element
   - NaN/infinity
4. Use existing test/sql/pgturbohybrid_multivector.sql style.
5. Update docs/multivector-late-interaction.md with this constructor as the preferred extension interop path.

Do not remove turbohybrid_multivector(vector[]).
```

---

## Prompt 13 — switch pg_colbert_llama to direct constructor when available

```text
Goal:
Use turbohybrid_multivector_from_float4 when pgturbohybrid provides it, while keeping vector[] fallback.

Tasks:
1. Add SQL function:
   colbert_float4(model text, input text) RETURNS real[]
   returning flat row-major values.
2. Add:
   colbert_dim(model text, input text) RETURNS int4
   or include dim in a composite type if cleaner.
3. Update colbert_mv:
   If turbohybrid_multivector_from_float4 exists, use it.
   Otherwise fallback to turbohybrid_multivector(colbert_vectors(...)).
4. Because SQL cannot easily branch on function existence inside immutable definitions, consider:
   - two SQL scripts, or
   - a plpgsql wrapper marked STABLE/PARALLEL RESTRICTED, or
   - document the fast path for pgturbohybrid >= version containing constructor.
5. Add regression tests for fallback.
6. Add live test for fast path if pgturbohybrid has the constructor.
```

---

## Prompt 14 — add parity tool against PyLate

```text
Goal:
Add an external parity checker that compares pg_colbert_llama output against PyLate for the same model.

Create:
extensions/pg_colbert_llama/tools/compare_pylate.py

Behavior:
1. Inputs:
   --model-name-or-path
   --pg-dsn
   --pg-model-alias
   --texts-file
   --role query|doc
2. For each text:
   - Get PyLate vectors with model.encode(..., is_query=role=='query')
   - Get PostgreSQL vectors from colbert(...) JSON or colbert_vectors(...)
   - Compare:
     dim
     count
     max_abs_error
     mean_abs_error
     max norm deviation from 1.0
3. Print summary.
4. Exit nonzero if thresholds fail.
5. Add README instructions.
6. Do not run this in default CI because PyLate/model downloads are external.
```

---

## Prompt 15 — Nix/dev-shell integration

```text
Goal:
Add optional Nix support for developing pg_colbert_llama with llama.cpp.

Tasks:
1. Inspect the repository flake.nix.
2. Add a dev shell or package variant that includes:
   - PostgreSQL
   - pgvector
   - pgturbohybrid
   - llama.cpp/libllama headers and shared libraries
3. Add shell commands:
   th-colbert-build-stub
   th-colbert-test-stub
   th-colbert-build-llama
   th-colbert-live-test
4. Keep the default flake check cheap:
   - build pgturbohybrid
   - build pg_colbert_llama stub
   - run stub regression tests
5. Do not require downloading GGUF models in flake check.
6. Document how to set PG_COLBERT_LLAMA_TEST_MODEL for live tests.
```

---

# Key test matrix

Use this as the acceptance checklist.

```text
Stub CI:
  extension installs
  invalid model strings rejected
  colbert JSON shape stable
  colbert_vectors returns vector[]
  colbert_mv returns turbohybrid_multivector
  dimensions/counts correct
  MaxSim works
  pgturbohybrid index can be built on stub multivectors

Llama live:
  GGUF loads
  model info reports n_embd_out = 128
  query/doc embeddings differ because prefixes differ
  all vectors finite
  all vector norms ≈ 1.0
  count within caps
  pgturbohybrid dense multivector scan uses index
  pgturbohybrid hybrid multivector+BM25 scan uses index
  turbohybrid_last_scan_stats shows multivector_enabled=true

Parity:
  PyLate versus PostgreSQL dim/count match
  max_abs_error acceptable for chosen GGUF precision
  ranking smoke test puts obvious relevant passage first
```

---

# Final recommendation

Implement **`pg_colbert_llama` as a companion extension first**, using llama.cpp/libllama internally and pgturbohybrid’s public `turbohybrid_multivector(vector[])` constructor externally.

Then add one small pgturbohybrid enhancement:

```sql
turbohybrid_multivector_from_float4(real[], dim int)
```

That gives you a clean fast path without making pgturbohybrid depend on llama.cpp. The search engine remains lean, while the embedding engine can evolve independently with llama.cpp API changes, GGUF conversion changes, and ColBERT-specific validation.

[1]: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/include/llama.h "raw.githubusercontent.com"
[2]: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/examples/embedding/embedding.cpp "raw.githubusercontent.com"
[3]: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_hf_to_gguf.py "raw.githubusercontent.com"
[4]: https://huggingface.co/VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT "VAGOsolutions/SauerkrautLM-Multi-Reason-ModernColBERT · Hugging Face"
