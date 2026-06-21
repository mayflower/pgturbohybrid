<p align="center">
  <img src="docs/assets/logo.png" alt="pgturbohybrid logo with a compact hot hatchback theme" width="720">
</p>

# pgturbohybrid

This README helps you understand what `pgturbohybrid` does, when hybrid search
is useful, how to install it, how to create your first index, and how to check
whether the fast path is working.

> The hot hatch of pgvector hybrid search: practical PostgreSQL retrieval with
> a surprising turn of speed.

[![build](https://github.com/mayflower/pgturbohybrid/actions/workflows/build.yml/badge.svg)](https://github.com/mayflower/pgturbohybrid/actions/workflows/build.yml)
![Status: alpha](https://img.shields.io/badge/status-alpha-orange)
![PostgreSQL 14–19](https://img.shields.io/badge/PostgreSQL-14--19-336791)
![PostgreSQL extension](https://img.shields.io/badge/PostgreSQL-extension-336791)

`pgturbohybrid` is a PostgreSQL extension for hybrid dense-vector + lexical
retrieval on top of pgvector. It combines pgvector dense nearest-neighbor search
with PostgreSQL text search using BM25-style ranking. BM25, short for Best
Matching 25, is a classic keyword-ranking method that rewards exact term
matches without requiring embeddings.

It is a standalone companion extension for pgvector: install pgvector first,
then install `pgturbohybrid` alongside it. The logo nods to hot hatchbacks:
compact, practical, quick, and daily-driver friendly. Not a million-euro
hypercar, but still has room for groceries.

## Status

`pgturbohybrid` is alpha software.

- APIs and on-disk index formats may change.
- Benchmarks are early and should be repeated on your own data.
- It is a good fit for evaluation, prototypes, and controlled experiments.
- Treat production use as something to validate carefully, not assume.

Per-feature maturity (stable public, experimental, research-only, diagnostic) is
tracked in the [feature & maturity matrix](docs/feature-matrix.md).

## Benchmarks

Dense (vector-only) retrieval on **dbpedia-openai-1M** (1,000,000 × 1536-d,
cosine), top-10, run with the [vector-db-benchmark][vdbb] harness at
`parallel=8` on a single node (AWS c6i, Intel Xeon Platinum 8375C). Every engine
is measured at **steady state**: an untimed warm-up pass precedes the timed run
so each engine's cache/buffers are hot. (Without this, engines that keep their
working set in a separate cache populated on first access — including
pgturbohybrid's native scan cache — are unfairly penalized against engines whose
index is already warm in shared buffers from the build.)

| engine | recall@10 | queries/s | mean latency |
|---|---:|---:|---:|
| **pgturbohybrid `dense`** | 0.836 | **5739** | 1.27 ms |
| **pgturbohybrid `high_recall`** | **0.983** | **2800** | 2.71 ms |
| weaviate | 0.977 | 2633 | 2.90 ms |
| pgvector (HNSW) | 0.979 | 1770 | 4.37 ms |
| qdrant | 0.986 | 853 | 9.24 ms |
| milvus | 0.988 | 750 | 10.39 ms |

- `dense` is the speed profile (4-bit, no rescore): the highest throughput here,
  ~3.2× pgvector, at recall 0.84 — use it when approximate recall is acceptable.
- `high_recall` is the exact-free high-recall profile (4-bit + heap-band
  rescore): **0.983 recall at 2800 q/s** — the best recall-per-throughput in this
  set. It beats pgvector on both recall *and* throughput, and delivers ~3× the
  throughput of qdrant/milvus at near-equal recall.

Single machine, single dataset — repeat on your own data and hardware. 4-bit
quantization is strongest on cosine / inner-product embeddings; high-dimensional
L2 (e.g. GIST-960) is a weaker case where recall holds up but latency does not.

[vdbb]: https://github.com/johannhartmann/vector-db-benchmark

## What It Does

TurboHybrid, the feature provided by `pgturbohybrid`, aims to make hybrid search
feel like a normal PostgreSQL index:

- combines pgvector dense retrieval with PostgreSQL text search/BM25-style
  retrieval
- fuses dense and lexical candidates inside one `turbohybrid` index access
  method
- uses reciprocal-rank fusion, or RRF, by default. RRF combines ranked result
  lists by position instead of trying to compare unlike score scales directly.
- aims to avoid the overhead of SQL-level two-index RRF plans for common
  top-k queries
- keeps pgvector unmodified

The basic idea: dense search helps with meaning, lexical search helps with exact
words, product names, IDs, and the weird little terms users actually type.
RRF is the fusion method that combines the two ranked candidate lists.
For score-level experiments, `turbohybrid_query(fusion => 'calibrated')` uses
monotonic dense/BM25 score normalization, chooses a dense alpha from query shape
when `alpha` is omitted, and can add a small bonus for candidates that appear in
both branches. This is a separate score-fusion mode; it does not preserve RRF
semantics and does not enable the `fast_weighted` BM25 score-bound pruning path.

```sql
SET turbohybrid.calibrated_fusion_both_match_bonus = 0.06;
SET turbohybrid.calibrated_fusion_identifier_bm25_alpha = 0.35;
SET turbohybrid.calibrated_fusion_broad_dense_alpha = 0.70;
SET turbohybrid.calibrated_fusion_default_alpha = 0.50;

SELECT id
FROM documents
ORDER BY embedding <~> turbohybrid_query(
  vector_query => $1,
  text_query => $2,
  fusion => 'calibrated'
)
LIMIT 10;
```

Inspect `calibrated_fusion_enabled`,
`calibrated_fusion_query_shape`, `calibrated_fusion_alpha_effective`,
`calibrated_fusion_both_match_bonus`,
`calibrated_fusion_dense_norm_mode`, and
`calibrated_fusion_bm25_norm_mode` in `turbohybrid_last_scan_stats()`.

## Multivector Late Interaction

`pgturbohybrid` includes a public `multivector` column type for
late-interaction retrieval models such as ColBERT-style MaxSim. A multivector
stores several same-dimensional token vectors for one document row. The native
graph build expands those token vectors into graph subnodes, while query output
is aggregated back to heap rows so the same document is not returned multiple
times.

```sql
CREATE TABLE passages (
  id bigint PRIMARY KEY,
  colbert multivector
);

INSERT INTO passages VALUES
  (1, turbohybrid_multivector(ARRAY[
    '[1,0,0]'::vector,
    '[0,1,0]'::vector
  ]));

CREATE INDEX passages_colbert_idx
ON passages USING turbohybrid (
  colbert multivector_cosine_turbohybrid_ops
);

SELECT id
FROM passages
ORDER BY colbert <~> turbohybrid_query(
  multivector_query => turbohybrid_multivector(ARRAY[
    '[1,0,0]'::vector
  ])
)
LIMIT 10;
```

Current multivector contract:

- dense-only multivector scans are supported with `multivector_query`;
- `vector_query` and `multivector_query` are mutually exclusive in one
  `turbohybrid_query`;
- hybrid multivector + `text_query` search is supported on two-key indexes with
  a `tsvector` key. RRF and the normalized score-fusion modes `weighted`,
  `fast_weighted`, `calibrated`, and `dbsf` are document-keyed; raw BM25 plus
  raw MaxSim alpha fusion is rejected;
- token-node indexes expand each row into one graph subnode per stored token
  vector. Document-node indexes store one graph node per document and attach the
  selected proxy, centroid, or sidecar payloads. Incremental insert/update
  follows the same storage mode and appends BM25 delta data when a lexical key
  is present;
- textual literal input for the underlying multivector value remains
  intentionally unsupported. Construct values from `vector[]`,
  `turbohybrid_multivector_from_float4(...)`, or the context/field
  constructors.

Candidate collection is approximate: each query token searches the TurboQuant
graph, then results are accumulated with document-level MaxSim. Tune the bounded
candidate budgets with:

```sql
SET turbohybrid.multivector_subvector_k = 100;
SET turbohybrid.multivector_unique_docs_per_token = 100;
SET turbohybrid.multivector_max_raw_hits_per_token = 400;
SET turbohybrid.multivector_doc_candidate_k = 100;
SET turbohybrid.multivector_exact_rerank = 'topk'; -- or 'off'
SET turbohybrid.multivector_exact_rerank_k = 100;
SET turbohybrid.multivector_max_accumulator_mb = 64;
```

By default, multivector scans exact-rerank a bounded top document prefix from
the heap. The index still stores compact TurboQuant subvector nodes; exact
rerank fetches the original `turbohybrid_multivector` heap value and computes
portable f32 MaxSim for at most `multivector_exact_rerank_k` retained document
candidates. Set `turbohybrid.multivector_exact_rerank = 'off'` to inspect the
raw approximate ordering.

Exact MaxSim always has a scalar fallback. When the extension is built with
SIMD support and `turbohybrid.simd` is enabled, the exact dot-product kernel may
dispatch to AVX2 on x86 or NEON on ARM for the bounded rerank work; portable and
`SIMD_BUILD=none` builds continue to use the scalar path.

### Native ColBERT candidate generation

For ColBERT-style models, `pgturbohybrid` can build document-node indexes that
separate candidate generation from final ranking: an approximate candidate
source (`proxy_vector`, `document_nodes`, experimental `centroid_lite`, or
research-only `quantized_inverted_experimental`; `exact_doc_scan` /
`exact_token_scan` are diagnostic oracles) chooses a bounded document set via the
`multivector_doc_storage` tier (`f32` / `f16` / `sq8` / `proxy_only` /
`centroid_only`), and the returned SQL order stays exact heap MaxSim over that
set. Inspect candidate-source health and admission via
`turbohybrid_last_scan_stats()`, and bound work with the
`turbohybrid.multivector_max_*` caps. Full storage-tier, candidate-source,
diagnostic-field, and tuning detail is in the doc below.

See [`docs/multivector-late-interaction.md`](docs/multivector-late-interaction.md)
for dense-only and hybrid examples, tuning guidance, diagnostics, and current
limitations.

For local ColBERT embedding inside PostgreSQL, see
[`docs/colbert-llama-extension.md`](docs/colbert-llama-extension.md). It
describes the companion `llama_embed` extension, which keeps llama.cpp model
loading separate from the `pgturbohybrid` index AM and returns dense `vector`,
token-level `vector[]`, and multivector-compatible values through the public
SQL API. Store late-interaction outputs in `multivector` columns. The
implementation still ships from the `pg_colbert_llama` source
directory for compatibility, and the legacy `CREATE EXTENSION
pg_colbert_llama` / `colbert_*` API remains available for existing ColBERT
callers.
The standalone examples in
[`extensions/pg_colbert_llama/examples/README.md`](extensions/pg_colbert_llama/examples/README.md)
show dense `llama_embed_vector()` output stored in pgvector and multivector
`llama_embed_mv()` output stored in `pgturbohybrid`.

## Native Sparse Retrieval (alpha)

> **Alpha / experimental.** The on-disk sparse format and SQL surface may change;
> version mismatches fail clearly and recommend `REINDEX`.

`pgturbohybrid` stores and searches learned-sparse (SPLADE-style) vectors
natively via the `turbohybrid_sparse_vector` type, the `<~*>` distance operator,
and the `sparse_ip_turbohybrid_ops` opclass. Sparse keys work alongside a dense
or multivector graph, or stand alone (sparse-only / sparse+BM25), and fuse with
dense/BM25 via RRF. Postings can be exact (f32) or quantized (q16/q8) with an
exact top-band rerank.

```sql
CREATE INDEX ON docs USING turbohybrid (s sparse_ip_turbohybrid_ops);
SELECT id FROM docs
ORDER BY s <~*> turbohybrid_query(sparse_query => q.s, final_k => 10)
LIMIT 10;
```

See [docs/sparse-embeddings.md](docs/sparse-embeddings.md) for the full type,
index shapes, quantization, fusion, GUCs, and stats, and
[the `llama_embed` sparse API](docs/colbert-llama-extension.md#sparse-splade-output--alpha)
for generating sparse vectors.

## When It Is Useful

Try `pgturbohybrid` when you are evaluating:

- RAG, or retrieval-augmented generation, over documents, tickets, support
  content, or knowledge bases
- semantic + lexical product or documentation search
- queries where dense-only retrieval misses exact terms
- queries where full-text search misses paraphrases
- PostgreSQL-first systems that want hybrid retrieval without adding another
  search service on day one

## When Not To Use It Yet

Wait, or isolate it carefully, if you need:

- stable on-disk index compatibility across releases
- official pgvector support
- production workloads without your own benchmark and relevance validation
- a mature operational story for every PostgreSQL and pgvector combination

Alpha means the paint is dry enough to touch, not enough to take through a car
wash.

## Install

### Nix development shell

For a reproducible local development environment with PostgreSQL, pgvector, and
`pgturbohybrid` packaged together:

```sh
nix develop
th-pg-init
th-smoke
th-psql
```

The default shell uses PostgreSQL 17 and pgvector v0.8.2. It keeps the local
database cluster under `.nix-dev/pg17-pgvector-v0.8.2/`, so it does not install
extensions into Homebrew, apt, or another system PostgreSQL prefix. If your Nix
install does not enable flakes globally, run commands with:

```sh
nix --extra-experimental-features 'nix-command flakes' develop
```

Useful commands inside the shell:

```sh
th-pg-start             # start the local PostgreSQL cluster
th-pg-stop              # stop it
th-pg-reset             # recreate it from scratch
th-installcheck         # run SQL regression tests
th-prove-installcheck   # run TAP tests when local TAP dependencies are available
th-test                 # run smoke + SQL regression tests
```

To test against the pinned pgvector `master` input instead of v0.8.2:

```sh
nix develop .#pgvector-master
th-pg-reset
th-smoke
```

After changing extension C or SQL files, re-enter the shell or run the relevant
`nix develop ... -c ...` command again so PostgreSQL sees the rebuilt extension
package. The Nix workflow intentionally uses an isolated wrapped PostgreSQL
rather than `make install` into a mutable system prefix.

For deterministic local quality checks, use the benchmark commands from the
default shell. They create synthetic data in the local Nix-managed database and
print result tables; do not commit captured output.

```sh
th-bench-retrieval-quality
th-bench-profile-grid
th-bench-tune-profile
```

For Python and real-data benchmark work, use the benchmark shell. It includes
`uv` and common Python data packages while keeping the default development shell
small.

```sh
nix develop .#bench
th-bench-concurrent-dense --help
FIQA_DATASET=/path/to/fiqa th-bench-fiqa-quick
```

`th-bench-fiqa-quick` defaults to the separate
`pgturbohybrid_fiqa_quick` database so it does not recreate the normal
`pgturbohybrid_dev` database. Set `FIQA_PGDATABASE=...` for a different
benchmark database, or set `PGDATABASE=...` explicitly when you want full
control.

The flake keeps `nix flake check` intentionally cheap: it builds the extension,
the wrapped PostgreSQL package, the pgvector-master variant, and a scalar
`SIMD_BUILD=none` variant. Long-running benchmarks, external datasets, and
host-specific result files stay outside `flake check`.

The flake inputs pin both nixpkgs and pgvector. Update them explicitly when
testing newer dependencies:

```sh
nix flake lock --update-input pgvector-master
```

### Manual install

Install pgvector first:

```sh
git clone --depth 1 --branch v0.8.2 https://github.com/pgvector/pgvector.git ../pgvector
make -C ../pgvector
make -C ../pgvector install
```

Then build and install `pgturbohybrid`:

```sh
git clone https://github.com/mayflower/pgturbohybrid.git
cd pgturbohybrid
make
make install
```

Create both extensions in your database:

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pgturbohybrid;
```

For a repeatable local setup, you can also use:

```sh
PG_CONFIG=pg_config PGVECTOR_REF=v0.8.2 scripts/dev-install.sh
```

## Quick Start

### Dense-Only Vector Search

Use a one-column `turbohybrid` index when you only need vector retrieval:

```sql
CREATE TABLE documents (
    id bigserial PRIMARY KEY,
    embedding vector(3) NOT NULL,
    body text NOT NULL
);

CREATE INDEX documents_dense_idx ON documents
USING turbohybrid (embedding vector_cosine_turbohybrid_ops);

SELECT id, body
FROM documents
ORDER BY embedding <~> turbohybrid_query(vector_query => $1::vector)
LIMIT 10;
```

For the common shapes there are shorter convenience wrappers that forward to
`turbohybrid_query(...)`: `turbohybrid_dense_query($1)`,
`turbohybrid_hybrid_query($1, $2)`, `turbohybrid_sparse_query($1)`, and
`turbohybrid_multivector_query($1)` (each takes optional `final_k` / per-branch
budgets). See [docs/how-it-works.md](docs/how-it-works.md#convenience-query-constructors).

### Hybrid Vector + Text Search

Add a `tsvector` key when queries use `text_query`:

```sql
CREATE TABLE documents (
    id bigserial PRIMARY KEY,
    embedding vector(3) NOT NULL,
    body text NOT NULL,
    body_tsv tsvector GENERATED ALWAYS AS (
        to_tsvector('english', body)
    ) STORED
);

INSERT INTO documents (embedding, body)
VALUES
    ('[1,0,0]', 'postgres vector search'),
    ('[1,1,0]', 'hybrid search with bm25'),
    ('[0,1,0]', 'lexical search in postgres'),
    ('[0,0,1]', 'unrelated document');

CREATE INDEX documents_turbohybrid_idx ON documents
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
);

ANALYZE documents;

SELECT id, body
FROM documents
ORDER BY embedding <~> turbohybrid_query(
    vector_query => '[1,0,0]'::vector,
    text_query => websearch_to_tsquery('english', 'postgres hybrid search')
)
LIMIT 10;
```

For tables that also store a ColBERT document column, use ColBERT as a reranker
for a dense-vector + BM25 hybrid candidate set by keeping the first-stage hybrid
query on the vector index and reranking only the bounded heap rows:

```sql
WITH q AS (
    SELECT
        turbohybrid_query(
            vector_query => '[1,0,0]'::vector,
            text_query => websearch_to_tsquery('english', 'postgres hybrid search'),
            dense_k => 200,
            bm25_k => 200,
            final_k => 200
        ) AS hybrid_query,
        turbohybrid_multivector(ARRAY[
            '[1,0,0]'::vector,
            '[0,1,0]'::vector
        ]) AS colbert_query
),
candidates AS MATERIALIZED (
    SELECT d.id, d.body, d.colbert
    FROM documents d, q
    ORDER BY d.embedding <~> q.hybrid_query
    LIMIT 200
)
SELECT c.id, c.body
FROM candidates c, q
ORDER BY turbohybrid_multivector_maxsim(q.colbert_query, c.colbert) DESC
LIMIT 10;
```

This is the supported shape for ColBERT reranking a vector+BM25 hybrid today:
`vector_query` and `multivector_query` still remain mutually exclusive inside
one `turbohybrid_query`, so the reranker query is passed to the scalar MaxSim
function instead of being mixed into the first-stage index payload.

Current DBpedia ColBERT benchmark evidence for this pattern is positive but
still bounded by the first-stage candidate window. With a dense+BM25 RRF
first-stage window of 200 candidates and exact ColBERT MaxSim reranking over
that window, top-10 quality changed as follows:

| corpus | stage | recall@10 | ndcg@10 | mrr@10 | map@10 |
|---|---|---:|---:|---:|---:|
| 50k docs / 25 queries | RRF first stage | 0.188000 | 0.135688 | 0.240000 | - |
| 50k docs / 25 queries | exact ColBERT rerank | 0.308000 | 0.251833 | 0.460000 | - |
| 1M docs / 381 queries | RRF first stage | 0.098838 | 0.072247 | 0.120932 | 0.041474 |
| 1M docs / 381 queries | exact ColBERT rerank | 0.128778 | 0.132511 | 0.241557 | 0.106777 |

On the 1M run, exact ColBERT reranking improved recall@10 by `30.3%`,
ndcg@10 by `83.4%`, mrr@10 by `99.7%`, and map@10 by `157.5%` relative to
the RRF candidate ordering. The measured full-path latency for RRF retrieval
plus exact ColBERT rerank over 200 candidates was p50 `30.745 ms`, p95
`148.354 ms`, and p99 `367.001 ms` over 381 queries. The corresponding 50k
run measured p50 `63.966 ms`, p95 `156.849 ms`, and p99 `483.225 ms` over 25
queries.

Treat these numbers as benchmark evidence for the rerank workflow, not as a
default serving profile: this mode computes BEIR/qrel quality only and does not
run a full exact MaxSim admission oracle. Recall is also limited by the RRF
candidate window, so larger windows should be benchmarked when higher recall is
the target.

Do not read this as evidence for a three-branch dense+BM25+ColBERT proxy
retriever. On the same 1M DBpedia corpus, the current proxy-only ColBERT branch
was a fast but effectively dead candidate source (`recall@10 = 0.000262`,
`ndcg@10 = 0.000364`), and naive RRF over dense, BM25, and that ColBERT branch
reduced quality (`recall@10 = 0.010892`, `ndcg@10 = 0.005273`) compared with
dense+BM25 RRF alone. Until native ColBERT candidate generation has stronger
admission evidence, use ColBERT as the bounded exact reranker shown above.

`text_query` requires a turbohybrid index with a `tsvector` key. A dense-only
index accepts vector queries and rejects text or vector+text queries with a clear
error.

Use `<~->` for L2, `<~#>` for negative inner product, and `<~>` for cosine
hybrid ordering. A longer copy-paste walkthrough lives in
[examples/fast_start.sql](examples/fast_start.sql) and
[docs/fast_setup.md](docs/fast_setup.md).

Migration note: existing two-key hybrid indexes remain valid. Dense-only users
can now create smaller one-key indexes. To change an index from hybrid to
dense-only, or from dense-only to hybrid, rebuild it with `DROP INDEX` /
`CREATE INDEX` or `REINDEX` after changing the index definition.

## Fast Defaults

Fresh sessions use the `latency` profile:

```sql
SHOW turbohybrid.profile;
```

The default path uses a 4-bit quantized index, exact vector storage off,
adaptive dense widening off, dense and BM25 candidate budgets of 100, RRF
constant 60, and the SQL `LIMIT` as the final result target when possible. In
plain terms: create the default index, query with
`ORDER BY ... turbohybrid_query(...) LIMIT n`, then inspect the scan stats
below.

Public candidate and cache settings are intentionally capped in this alpha so a
user cannot set runaway per-query budgets in a shared PostgreSQL server. If you
hit a cap during evaluation, please open an issue with the dataset size, query
shape, and the settings you tried.

For the normal fast path, keep the query simple:

```sql
SELECT id, body
FROM documents
ORDER BY embedding <~> turbohybrid_query(
    vector_query => $1::vector,
    text_query => websearch_to_tsquery('english', $2)
)
LIMIT 10;
```

That is the "daily driver" mode: compact settings, fewer knobs, enough speed to
be interesting, and no need to pack a racing helmet.

## Profile choice

Pick a profile by what the query workload needs; all are compact 4-bit,
exact-free by default. Always validate on your own data — the guidance below is
qualitative, and any numbers in `benchmarks/` are local synthetic examples, not
portable performance claims.

- **`latency`** (default): fastest. A good fit for easy corpora and
  latency-sensitive serving where approximate recall is acceptable.
- **`matched_recall`**: the compact comparison profile, intended to approximate
  full-vector HNSW recall (pgvector/Qdrant) without exact storage. Treat it as a
  comparison baseline and **validate its recall on your real workload** before
  relying on it.
- **`high_recall`**: use when hard or ambiguous dense recall matters and you have
  latency headroom. It recovers recall on hard queries by using wider
  `graph_ef_search` / `graph_oversampling` (and heuristic build) — i.e. by
  searching more, at higher per-query latency. The recall gain comes from those
  wider search windows, not from the opt-in features below. (A local synthetic
  hard case is documented in `benchmarks/README.md`.)
- **`quality`**: relevance-oriented (stronger, slower). **Benchmark it before
  making it a default** — its extra cost is only worth it if your data shows a
  relevance gain.

The newer retrieval features — residual rerank, dense uncertainty retry, BM25
heap-tsvector rerank, and final diversity — are **opt-in or profile-gated**, off
in the default profiles, and should be **benchmarked separately** on your data
before enabling. They change behavior independently of the profile's graph/search
windows, so measure them one at a time.

These are guidelines, not defaults to change: no profile's compiled defaults
should be retuned from synthetic benchmarks alone.

## Deep Profile Tuning

Beyond the profile choice above, TurboHybrid exposes `matched_recall`,
`high_recall`, and `quality` modes plus `quantization_bits`, `exact_storage`,
heap rescore, residual rerank, adaptive widening, uncertainty retry, entry
sidecars, payload seeding, final diversity, and segment controls. The full
tuning guide -- with worked examples and the `turbohybrid_last_scan_stats()`
keys to inspect for each knob -- is in
**[docs/profile-tuning.md](docs/profile-tuning.md)**.

No profile's compiled defaults should be retuned from synthetic benchmarks alone.

## Diagnostics

The stable vs experimental vs diagnostic-only keys of
`turbohybrid_last_scan_stats()` are documented in
[docs/diagnostics-schema.md](docs/diagnostics-schema.md).

After a query, check whether PostgreSQL used the expected TurboHybrid path:

```sql
SELECT turbohybrid_last_scan_stats();
SELECT turbohybrid_index_stats('documents_turbohybrid_idx'::regclass);
SELECT turbohybrid_estimate_memory('documents_turbohybrid_idx'::regclass);
SELECT turbohybrid_graph_repair_dry_run('documents_turbohybrid_idx'::regclass);
SELECT turbohybrid_simd_capabilities();
```

`turbohybrid_graph_repair_dry_run(index, sample_nodes, search_ef,
candidate_limit)` is an opt-in read-only graph neighborhood diagnostic for
native graph indexes. It samples node IDs deterministically, compares each
sample's existing level-0 neighborhood with a stronger bounded local candidate
pool, and reports `avg_overlap`, `weak_nodes`, `missed_neighbor_count`, and
`suggested_edges`. The function never writes index pages, never emits WAL, and
uses `AccessShareLock`; it is meant to decide whether a future write-capable
repair pass is worth building.

Use `turbohybrid_estimate_memory(index)` before a query or prewarm step when
sizing native graph cache memory. It reports native code/adjacency/exact-vector
cache estimates, the current native cache policy, shared-cache per-backend view
bytes, and BM25 reader-cache arrays without building the native graph cache or
loading BM25 doc/liveness caches.

Native cache scope matters under concurrency:

- `per_backend` stores the resident native graph cache in each active PostgreSQL
  backend. A 400 MB native cache is roughly 400 MB at 1 backend, 4 GB at 10
  backends, and 40 GB at 100 backends before BM25 reader caches and other
  backend memory.
- `shared` stores the large immutable native graph arenas once in the mmap-backed
  shared cache, but each backend still allocates view/scratch metadata and BM25
  reader caches.
- `off` avoids resident native graph cache memory, but scans load graph pages
  through PostgreSQL shared buffers and can do more per-scan I/O/CPU work.

Use the estimator to compute concrete 1/10/100-backend projections for an index:

```sql
WITH estimate AS (
  SELECT turbohybrid_estimate_memory('documents_turbohybrid_idx'::regclass) AS m
),
backends(n) AS (VALUES (1), (10), (100))
SELECT
  n AS active_backends,
  pg_size_pretty(
    n * (
      (m->'concurrency'->>'per_backend_total_bytes_per_backend')::bigint +
      (m->'concurrency'->>'bm25_total_bytes_per_backend')::bigint
    )
  ) AS per_backend_scope_total,
  pg_size_pretty(
    (m->'concurrency'->>'shared_cache_total_bytes')::bigint +
    n * (
      (m->'concurrency'->>'shared_backend_view_bytes_per_backend')::bigint +
      (m->'concurrency'->>'bm25_total_bytes_per_backend')::bigint
    )
  ) AS shared_scope_total
FROM estimate, backends
ORDER BY n;
```

`turbohybrid.native_cache_warn_mb` emits a non-fatal DEBUG1 message when a
per-backend native cache build crosses the configured resident-size threshold.
Set it to `0` to disable the warning.

For a slow query, `turbohybrid_last_scan_diagnosis()` reduces the full stats
JSON to the key dense hot-path fields, a few derived ratios, and a single
`diagnosis` label (for example `healthy_u8_x4`, `traversal_dominated`,
`rescore_dominated`, or `scalar_lut_fallback`):

```sql
SELECT turbohybrid_last_scan_diagnosis();
SELECT turbohybrid_last_scan_diagnosis() ->> 'diagnosis';
```

Useful fields include:

- `profile`
- `index_used`
- `scan_orchestration`
- `quantization_bits`
- `exact_storage`
- SIMD fields such as `runtime_avx2`, `enabled_avx512vnni`, and
  `simd_force`
- `dense_candidates_effective`
- `bm25_candidates_effective`
- `final_k_source`
- `elapsed_us`

For troubleshooting examples, see [docs/fast_setup.md](docs/fast_setup.md).

## More Benchmark Snapshots

The headline dbpedia-1M dense comparison is in [Benchmarks](#benchmarks) above.
Additional **local snapshots** -- FIQA/OpenAI hybrid, DBPedia 1M hybrid, and a
dense-only Turbovec reference, with full settings, baselines, and tables -- live
in **[docs/benchmarks/local-snapshots.md](docs/benchmarks/local-snapshots.md)**.
They are local snapshots, not global claims; repeat them on your own hardware
and query mix. Reproduction notes:
[fiqa-openai.md](docs/benchmarks/fiqa-openai.md),
[dbpedia_openai3_large.md](benchmarks/dbpedia_openai3_large.md),
[benchmarks/README.md](benchmarks/README.md), and
[bring-your-own-rag.md](docs/benchmarks/bring-your-own-rag.md).

## How It Works, Short Version

`pgturbohybrid` defines a `turbohybrid` PostgreSQL index access method over:

- one pgvector `vector` column for dense retrieval
- optionally, one `tsvector` column for lexical/BM25 retrieval

Dense-only indexes run the dense branch. Hybrid indexes can gather dense and
BM25-style lexical candidates, fuse them with reciprocal-rank fusion, and return
rows through PostgreSQL's normal `ORDER BY ... LIMIT` index-scan shape.

It depends on pgvector's SQL `vector` type but does not require pgvector to be
patched. Some graph/index code is derived from pgvector's HNSW implementation;
see [NOTICE](NOTICE) and [docs/architecture.md](docs/architecture.md).

## Documentation

- [Feature & maturity matrix](docs/feature-matrix.md)
- [How TurboHybrid works](docs/how-it-works.md)
- [Easy fast setup](docs/fast_setup.md)
- [Diagnostics schema](docs/diagnostics-schema.md)
- [Sparse (SPLADE) embeddings](docs/sparse-embeddings.md)
- [Multivector late interaction](docs/multivector-late-interaction.md)
- [FIQA/OpenAI benchmark snapshot](docs/benchmarks/fiqa-openai.md)
- [Bring-your-own RAG benchmark](docs/benchmarks/bring-your-own-rag.md)
- [DBPedia OpenAI3-large benchmark spec](benchmarks/dbpedia_openai3_large.md)
- [Benchmark methodology](benchmarks/README.md)
- [Compatibility notes](docs/compatibility.md)
- [Architecture notes](docs/architecture.md)
- [Release policy](RELEASE.md)
- [v0.1.0-alpha.2 release notes](docs/release-notes/v0.1.0-alpha.2.md)
- [Release hygiene summary](RELEASE_HYGIENE.md)

## Compatibility

The release target is PostgreSQL 14 through 19 (CI builds and runs the
regression tests on every version; PostgreSQL 19 is tested against pgvector
`master`). The pgvector compatibility target is pgvector 0.8.2 through current
pgvector `master`.

See [docs/compatibility.md](docs/compatibility.md) for the tested matrix and
boundary notes. If pgvector changes its internal vector layout, `pgturbohybrid`
should fail directly rather than silently reading malformed data.

## Contributing

Contributions are welcome, especially:

- real benchmark reports with complete context
- compatibility results across PostgreSQL, pgvector, CPU, and OS versions
- bug reports with `EXPLAIN` output and diagnostic JSON
- documentation fixes that make the alpha easier to evaluate safely

Start with [CONTRIBUTING.md](CONTRIBUTING.md). For community expectations,
see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), [SUPPORT.md](SUPPORT.md), and
[SECURITY.md](SECURITY.md).

## Attribution

`pgturbohybrid` depends on pgvector and contains code derived from pgvector's
HNSW implementation. pgvector is an excellent PostgreSQL vector search
extension; `pgturbohybrid` is a separate experimental companion project, not an
official pgvector project.

`pgturbohybrid` is distributed under the PostgreSQL License. See
[LICENSE](LICENSE) for the license text and [NOTICE](NOTICE) for pgvector
attribution.
