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

Contract, in brief: `vector_query` and `multivector_query` are mutually
exclusive; hybrid multivector + `text_query` fuses document-keyed (RRF /
`weighted` / `fast_weighted` / `calibrated` / `dbsf`); scans exact-rerank a
bounded top-document prefix from the heap by default (no full f32 vectors stored
in the index); and textual literal input is intentionally unsupported (construct
from `vector[]` or `turbohybrid_multivector_from_float4(...)`).

Candidate budgets, exact-rerank policy, document-node storage tiers
(`f32`/`f16`/`sq8`/`proxy_only`/`centroid_only`), the native ColBERT candidate
sources (`proxy_vector`, `document_nodes`, experimental `centroid_lite`,
research-only `quantized_inverted_experimental`), and tuning live in
**[docs/multivector-late-interaction.md](docs/multivector-late-interaction.md)**.
For generating embeddings locally, the companion `llama_embed` extension is
documented in
[docs/colbert-llama-extension.md](docs/colbert-llama-extension.md).

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

A reproducible **Nix development shell** (PostgreSQL 17 + pgvector + the
extension, with `th-*` helper commands and benchmark shells) is documented in
[CONTRIBUTING.md](CONTRIBUTING.md#nix-development-shell). For a system install,
use the manual steps below.

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

After a query, check whether PostgreSQL used the expected TurboHybrid path:

```sql
SELECT turbohybrid_last_scan_stats();
SELECT turbohybrid_last_scan_diagnosis();   -- single bottleneck label + key fields
SELECT turbohybrid_index_stats('documents_turbohybrid_idx'::regclass);
SELECT turbohybrid_simd_capabilities();
```

`turbohybrid_last_scan_diagnosis()` reduces the full stats JSON to the key dense
hot-path fields and a single `diagnosis` label (for example `healthy_u8_x4`,
`traversal_dominated`, `rescore_dominated`, or `scalar_lut_fallback`):

```sql
SELECT turbohybrid_last_scan_diagnosis() ->> 'diagnosis';
```

The stable vs experimental vs diagnostic-only keys of
`turbohybrid_last_scan_stats()` are documented in
[docs/diagnostics-schema.md](docs/diagnostics-schema.md). For cache-scope
sizing, per-backend memory multiplication, `turbohybrid_estimate_memory()`
projections, the read-only `turbohybrid_graph_repair_dry_run()` diagnostic, and
VACUUM/REINDEX/upgrade guidance, see [docs/operations.md](docs/operations.md).

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
- [Beta scope](docs/beta-scope.md)
- [Roadmap](ROADMAP.md)
- [How TurboHybrid works](docs/how-it-works.md)
- [Easy fast setup](docs/fast_setup.md)
- [Diagnostics schema](docs/diagnostics-schema.md)
- [Operations guide](docs/operations.md)
- [Storage format](docs/storage-format.md)
- [Profile tuning](docs/profile-tuning.md)
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
