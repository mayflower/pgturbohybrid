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

`pgturbohybrid` includes an experimental `turbohybrid_multivector` type for
late-interaction retrieval models such as ColBERT-style MaxSim. A multivector
stores several same-dimensional token vectors for one document row. The native
graph build expands those token vectors into graph subnodes, while query output
is aggregated back to heap rows so the same document is not returned multiple
times.

```sql
CREATE TABLE passages (
  id bigint PRIMARY KEY,
  colbert turbohybrid_multivector
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
- textual literal input for `turbohybrid_multivector` remains intentionally
  unsupported. Construct values from `vector[]`,
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
separate candidate generation from final ranking. Candidate sources are allowed
to use approximate proxy, centroid, or compact-code scores to choose a bounded
document set. The returned SQL order remains exact heap MaxSim over that
retained set.

The document-node storage tiers are selected with the `multivector_doc_storage`
index option:

- `f32`, `f16`, and `sq8` store a full document multivector sidecar for
  document-node scoring experiments.
- `proxy_only` stores only document IDs, heap TIDs, graph adjacency, and the
  fixed-dimensional proxy vector. It is useful for low-memory proxy admission
  and exact heap rerank.
- `centroid_only` stores centroid and posting payloads without the full
  multivector sidecar. It is intended for centroid and quantized-inverted
  candidate-source experiments.

The main pure-ColBERT candidate sources are:

- `proxy_vector`: graph traversal over one proxy vector per document.
- `document_nodes`: document-node graph scoring over the configured
  document-node storage.
- `centroid_lite`: experimental PLAID-style centroid posting admission.
- `quantized_inverted_experimental`: research-only codeword/posting admission.
- `exact_doc_scan` and `exact_token_scan`: diagnostic oracles, not serving
  paths.

Use `turbohybrid_last_scan_stats()` or the DBpedia ColBERT benchmark JSON to
check whether a candidate source is healthy. The most useful fields are:
latency (`p50_ms`, `p95_ms`, `p99_ms`), qrel quality (`recall@10`, `ndcg@10`,
`mrr@10`), exact-oracle admission when available
(`exact_top1_admission_rate`, `exact_top10_admission_recall`), candidate work
(`proxy_candidates_returned`, `centroid_docs_touched`,
`quantized_inverted_docs_scored`), and exact rerank cost
(`multivector_exact_rerank_docs`, `multivector_exact_rerank_pairs`,
`multivector_exact_maxsim_rerank_time_us`).

Safety caps are controlled by `turbohybrid.multivector_max_doc_vectors`,
`turbohybrid.multivector_max_query_vectors`, and
`turbohybrid.multivector_max_dim`.

See [`docs/multivector-late-interaction.md`](docs/multivector-late-interaction.md)
for dense-only and hybrid examples, tuning guidance, diagnostics, and current
limitations.

For local ColBERT embedding inside PostgreSQL, see
[`docs/colbert-llama-extension.md`](docs/colbert-llama-extension.md). It
describes the companion `pg_colbert_llama` extension, which keeps llama.cpp
model loading separate from the `pgturbohybrid` index AM and returns
`turbohybrid_multivector` values through the public SQL API.

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

## Matched-Recall And Quality Modes

Use `matched_recall` when you want a compact 4-bit, exact-free index but need a
single named setting for pgvector/Qdrant-like dense recall comparisons:

```sql
SET turbohybrid.profile = 'matched_recall';
```

This profile keeps `quantization_bits = 4` and `exact_storage = off`, but new
low-dimensional indexes use heuristic graph neighbor selection, exact build
distances during graph construction, `ef_construction = 192`,
`ef_search = 128`, `graph_oversampling = 8`, final top-k heap rescore, and one
native segment unless `native_segments` is explicitly set. Exact build
distances are build-time only; they do not store full vectors in the index.
Changing build-time reloptions such as neighbor selection, build distance,
residual sketches, entry sidecar, graph windows, or segment count for an
existing index requires `REINDEX`. Query-time GUCs such as heap rescore,
adaptive widening, uncertainty retry, residual mode, and fusion can be compared
without rebuilding unless they depend on build-time index contents.

`quantization_bits = 8` is available as an opt-in scalar-safe prototype for
recall experiments:

```sql
CREATE INDEX documents_turbohybrid_q8_idx ON documents
USING turbohybrid (embedding vector_cosine_turbohybrid_ops)
WITH (quantization_bits = 8, exact_storage = off);
```

The 8-bit path stores one byte per dimension and uses scalar code scoring. It
does not reuse the optimized 4-bit split/LUT SIMD kernels, is not the default,
and should be benchmarked against 4-bit, 4-bit plus residual rerank, and
`exact_storage = on` before adopting it for a workload. `turbohybrid_last_scan_stats()`
reports `quantization_bits = 8` and `dense_scorer = scalar_8bit` when this path
is active.

Use `high_recall` when you want near-exact dense recall from a compact 4-bit,
**exact-free** index and have latency headroom to spend:

```sql
SET turbohybrid.profile = 'high_recall';
```

`high_recall` keeps `quantization_bits = 4` and `exact_storage = off`, reuses
`matched_recall`'s candidate budgets, and additionally resolves
`dense_heap_rescore` to `band` at scan time (exact rescore of the full final
candidate band by re-reading vectors from the heap), turns
`dense_adaptive_widening` off, and defaults new indexes to
`ef_construction = 256`, `ef_search = 192`, `graph_oversampling = 12`. Band
rescore can recover ordering quality when 4-bit code scoring finds good
candidates but mis-ranks them, without storing full vectors in the index. Pair
it with a heuristic build for best results, then verify the recall/p95 tradeoff
with the retrieval-quality grid or your external benchmark before adopting it:

```sql
SET turbohybrid.profile = 'high_recall';
SET turbohybrid.dense_build_neighbor_select = 'heuristic';
SET turbohybrid.dense_build_distance = 'code';

CREATE INDEX documents_turbohybrid_high_recall_idx ON documents
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (
    quantization_bits = 4,
    exact_storage = off,
    graph_ef_construction = 256,
    graph_ef_search = 192,
    graph_oversampling = 12,
    native_segments = 1
);
```

Use `quality` when relevance matters more than lowest latency:

```sql
SET turbohybrid.profile = 'quality';
```

Quality mode is stronger and slower: it uses larger dense and lexical candidate
budgets, conservative BM25 paths, higher default graph search windows, and
heuristic dense-neighbor selection for new indexes. It does not automatically
enable calibrated fusion, phrase/proximity heap lexical rerank, or the bounded
uncertainty retry; those remain explicit benchmark knobs until the grid shows a
quality win at acceptable p95/p99. It usually costs more build CPU and query CPU
than `latency` or `matched_recall`, so compare it at matched recall/precision
instead of only comparing raw p95. For quality-sensitive production evaluation,
benchmark an exact-storage index too:

```sql
CREATE INDEX documents_turbohybrid_quality_idx ON documents
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (exact_storage = on);
```

For exact-free 4-bit indexes that need a precision check without storing a full
copy of every vector in the index, heap-backed exact rescore can rescore only
the final top-k from the heap:

```sql
SET turbohybrid.dense_heap_rescore = 'topk'; -- or 'band'
```

`topk` fetches heap tuples for the final top-k and recomputes exact vector
distance with the same SIMD kernels used by index-exact rescore. `band` rescans
the full final candidate band. Both add heap I/O. The `latency` profile keeps
heap rescore off; the `high_recall` profile defaults it to `band`; `balanced`,
`matched_recall`, and `quality` may enable `topk` automatically for
low-dimensional exact-free indexes, and an explicit
`SET turbohybrid.dense_heap_rescore = 'off'|'topk'|'band'` always wins over the
profile default; set it to `auto` to return to profile-driven behavior.
Residual rerank is an opt-in, lower-I/O refinement (off in every profile),
built into the index with
`WITH (residual_rerank = on, residual_rerank_bytes = 16|32|64)`. It reorders a
narrow top band from residual sketches stored in the index, so it is **not a
replacement for heap-band rescore**: it recovers recall only when the true
neighbours already sit in that narrow band, whereas heap-band rescore recovers
neighbours from the wider candidate band. Larger `residual_rerank_bytes` reranks
more precisely but grows the index — a build-time storage tradeoff. Residual
sketches are build-time index contents; the scan-time adjustment is controlled by
GUCs:

```sql
SET turbohybrid.dense_residual_rerank_mode = 'calibrated'; -- off | fixed | calibrated
SET turbohybrid.dense_residual_rerank_weight = -1.0;       -- auto
SET turbohybrid.dense_residual_rerank_max_adjust_ratio = 0.15;
```

`fixed` preserves the original hardcoded residual adjustment. `calibrated`
scales the sketch adjustment by the observed final-band distance spread and
clamps it, which makes the adjustment comparable across queries. Inspect
`residual_rerank_mode`, `residual_rerank_weight_effective`,
`residual_rerank_band`, `residual_rerank_max_adjustment`,
`residual_rerank_reordered_count`, and `residual_rerank_topk_changed` in
`turbohybrid_last_scan_stats()`.
Because residual sketches are stored in the index, enabling or changing
`residual_rerank_bytes` for an existing index requires `REINDEX`. Switching
between `off`, `fixed`, and `calibrated` residual adjustment modes is
query-time only, but the mode has no effect unless the index was built with
residual sketches.

For phrase/proximity-like text queries, BM25 can optionally rerank a bounded
candidate prefix by fetching the indexed heap `tsvector` and applying PostgreSQL
text-search ranking semantics:

```sql
SET turbohybrid.bm25_heap_tsvector_rerank = 'auto'; -- off | topk | band | auto
SET turbohybrid.bm25_heap_tsvector_rerank_multiplier = 4;
SET turbohybrid.bm25_heap_tsvector_rerank_weight = 0.10;
```

The default is `off`. `topk` fetches only the final-k BM25/hybrid candidates;
`band` fetches up to `final_k * multiplier`, capped by the BM25 candidate
count; `auto` enables the same bounded band for phrase tsqueries. The adjustment
uses `ts_rank_cd` when heap `tsvector` positions are present and `ts_rank`
otherwise. This does not add positional postings or change the index format,
and it does not run for dense-only queries. Inspect
`bm25_heap_tsvector_rerank_mode`, `bm25_heap_tsvector_rerank_count`,
`bm25_heap_tsvector_rerank_fetch_us`,
`bm25_heap_tsvector_rerank_score_us`, and
`bm25_heap_tsvector_rerank_topk_changed` in
`turbohybrid_last_scan_stats()`.

Do not treat any profile as universally best. Measure latency and relevance on
your dataset. Adaptive dense widening stays off for the `latency` profile, but
`balanced`, `matched_recall`, and `quality` can use a conservative `auto` mode
on low-dimensional exact-free 4-bit indexes when `final_k` is small and the
score boundary looks ambiguous. The separate bounded uncertainty retry stays
off in named production profiles until benchmark evidence shows the p95/p99 cost
is low; test it explicitly:

```sql
SET turbohybrid.dense_uncertainty_retry = 'auto'; -- off | auto | on
SET turbohybrid.dense_uncertainty_retry_max_passes = 1;
SET turbohybrid.dense_uncertainty_retry_multiplier = 1.5;
```

`off` preserves the single traversal path. `auto` retries only when the first
candidate band is underfilled or has flat top-k/boundary gaps, or when sidecar,
payload-filter, residual-rerank, or heap-rescore evidence suggests the band was
uncertain. `on` forces the retry whenever the wider target is bounded by the
node count and `turbohybrid.max_scan_tuples`. Inspect
`dense_uncertainty_retry_triggered`, `dense_uncertainty_retry_reason`,
`dense_uncertainty_final_target`, and `dense_uncertainty_final_ef` in
`turbohybrid_last_scan_stats()`.

Local expansion, entry sidecars, uncertainty retry, calibrated hybrid fusion,
phrase/proximity heap lexical rerank, and residual rerank remain opt-in knobs
for benchmark work unless a named profile explicitly documents otherwise. Entry
sidecars keep using
the fixed metapage node-id array; `entry_sidecar_strategy` controls which
representatives are chosen at build time:

```sql
WITH (
  entry_sidecar = on,
  entry_sidecar_representatives = 128,
  entry_sidecar_strategy = 'hash' -- hash | farthest_code | level_covering | hybrid_level_covering
)
```

`hash` is the default and preserves existing behavior. `farthest_code` uses a
deterministic farthest-point selection in code-distance space, `level_covering`
prefers high-level graph nodes while diversifying by code hash bucket, and
`hybrid_level_covering` combines both. Because the selected node IDs are stored
when the index is built, changing the strategy for an existing index requires
`REINDEX` to change the sidecar contents.
For payload-filtered dense scans over `INCLUDE` int4 columns, the scan path can
also seed graph traversal from the existing payload-ref range:

```sql
SET turbohybrid.payload_entry_seeding = 'auto'; -- off | auto | on
SET turbohybrid.payload_entry_seed_count = 8;   -- max 64
```

This is scan-time only: it does not add payload-routing pages or change the
index format. `auto` is the default and only affects scans with an active int4
payload equality filter; if the payload value has no ref range, traversal falls
back to the normal global/sidecar entry points. Inspect
`payload_entry_seeding_hit`, `payload_entry_seed_count`, and
`payload_entry_seed_range_count` in `turbohybrid_last_scan_stats()`.

To reduce near-duplicate final results from the same document or chunk group,
enable scan-time diversity over an existing int4 `INCLUDE` payload slot:

```sql
SET turbohybrid.final_diversity = 'group_payload'; -- off | group_payload
SET turbohybrid.final_diversity_payload_slot = 0;  -- INCLUDE payload slot
SET turbohybrid.final_diversity_lambda = 0.75;     -- relevance/diversity mix
SET turbohybrid.final_diversity_pool_multiplier = 3;
```

This is off by default and does not fetch heap rows or change the index format.
The payload slot is zero-based in `INCLUDE` column order. If the slot is invalid
or a candidate has no payload value, the scan falls back to normal ranking.
Inspect `final_diversity_mode`, `final_diversity_pool_size`,
`final_diversity_selected`, and
`final_diversity_duplicate_groups_suppressed` in
`turbohybrid_last_scan_stats()`.
`native_segments` is a build/concurrency lever rather than a free quality knob:
the default is one segment, quality/exact-build auto segmenting resolves to one
segment, and multi-segment indexes should be benchmarked with
`turbohybrid.native_segment_budget = sqrt|linear` before using them for
quality-sensitive comparisons.

For a deterministic synthetic dense/hybrid profile sweep, use
`benchmarks/dev/retrieval_quality_grid.sql`. It reports recall or overlap at K,
duplicate groups, elapsed milliseconds, index settings, and selected scan stats
for `latency`, `matched_recall`, `high_recall`, `quality`, residual rerank,
heap rescore, uncertainty retry, entry sidecar, calibrated fusion, lexical heap
rerank, and diversity configurations. Use it as the before/after methodology
for deciding whether a new query-time feature should become a profile default.
For a dense-only profile sweep against a glove-like workload, use
`benchmarks/glove100_recall_latency_grid.sql`.

For an existing workload with known expected ids, use
`benchmarks/dev/tune_retrieval_profile.sql` as a practical autotuning harness.
It consumes an `eval_queries` table, sweeps query-time profiles, candidate
budgets, fusion, residual rerank mode, and heap rescore mode against an existing
TurboHybrid index, then prints all trials, a Pareto frontier, and an optional
recommendation under a p95 latency budget. This is a developer benchmark script;
it does not add a SQL-visible autotuner or change production defaults.

## Diagnostics

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

## Benchmark Snapshot

These are local benchmark snapshots, not global claims. Results vary by
dataset, hardware, PostgreSQL settings, cache state, and query workload. nDCG
means normalized discounted cumulative gain, a relevance metric for ranked
results. HNSW means Hierarchical Navigable Small World, pgvector's graph index
type for approximate nearest-neighbor vector search.

On this FIQA/OpenAI setup, the run used 57,638 corpus rows, 648 qrels-backed
queries, 1,536-dimensional OpenAI `text-embedding-3-small` embeddings, the
`latency` profile, 100 dense candidates, 100 BM25 candidates, `final_k = 10`,
three warmup passes, and one measured pass. The TurboHybrid index used 4-bit
quantization with `exact_storage = off`.

| Method | Settings | p95 | nDCG@10 |
| --- | --- | ---: | ---: |
| pgturbohybrid default | default 4-bit exact-free index, adaptive widening off, LIMIT-inferred `final_k` | 1.628 ms | 0.415535 |
| pgturbohybrid adaptive auto 2.0 | explicit diagnostic setting, adaptive dense widening opt-in | 1.552 ms | 0.421465 |
| SQL RRF baseline | pgvector HNSW plus PostgreSQL GIN full-text search, 100/100 candidates | 2.009 ms | 0.423430 |
| pgvector dense-only reference | pgvector HNSW, no lexical branch | 1.412 ms | 0.442786 |

On a larger DBPedia/OpenAI3-large 1M qdrant-self setup, the run used 1,000,000
rows, 1,000 sampled self-queries, 3,072-dimensional
`text-embedding-3-large` embeddings, the `latency` profile, 100 dense
candidates, 100 BM25 candidates, `final_k = 10`, one warmup pass, and three
measured passes.

| Method | p95 | p99 | nDCG@10 | recall@10 | Index size |
| --- | ---: | ---: | ---: | ---: | ---: |
| pgturbohybrid | 3.978 ms | 7.386 ms | 0.940439 | 0.980 | 2.38 GB |
| pgvector halfvec HNSW + GIN FTS SQL RRF | 79.799 ms | 219.576 ms | 0.971086 | 0.992 | 8.41 GB |

That DBPedia result is a tradeoff, not a victory lap: TurboHybrid was much
faster and smaller on this machine, while the pgvector + FTS SQL RRF baseline
kept higher nDCG@10 and recall@10. The package default latency profile keeps
adaptive dense widening off; adaptive widening variants remain available in the
benchmark harness for controlled experiments. Benchmark details, baselines, and
reproduction notes are in
[docs/benchmarks/fiqa-openai.md](docs/benchmarks/fiqa-openai.md),
[benchmarks/dbpedia_openai3_large.md](benchmarks/dbpedia_openai3_large.md), and
[benchmarks/README.md](benchmarks/README.md).

The same DBPedia/OpenAI3-large corpus can also be used as a dense-only systems
comparison. This is not a hybrid-search benchmark: it uses the dataset's
existing 3,072-dimensional embeddings, no BM25 branch, no full-text search, and
no SQL RRF fusion. Turbovec is an in-process dense vector library that runs a
flat (brute-force) 4-bit scan over all rows, so treat this as a useful
reference point rather than a PostgreSQL access-method comparison. The run below
is a fresh local snapshot (Ice Lake Xeon, current build, 200 qdrant-self
queries, one warmup pass plus measured passes); the pgturbohybrid percentiles
are end-to-end SQL latency while the Turbovec percentiles are in-process,
single-threaded `index.search()` timings.

| Dense-only method | p50 | p95 | p99 | nDCG@10 | recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| pgturbohybrid dense-only | 1.848 ms | 2.287 ms | 2.443 ms | 0.965 | 0.965 |
| Turbovec TurboQuant 4-bit | 174.176 ms | 181.312 ms | 190.260 ms | 1.000 | 1.000 |

In this qdrant-self setup each query has a single positive qrel, so recall@10 is
"is the source document in the top 10." Turbovec's flat scan is exact over the
4-bit codes and recovers that document for every query (recall@10 = 1.000), but
pays O(n) latency (~174 ms p50). `pgturbohybrid` uses a sub-linear graph index:
~94x lower latency (1.848 ms p50) while still recovering the source document for
96.5% of queries. The top-10 overlap between the two runs was 0.922 — i.e. the
graph reproduces ~92% of Turbovec's exact-over-codes top-10. As with the hybrid
numbers above, repeat this on your own hardware and query mix before drawing
conclusions.

If you already have a PostgreSQL RAG database, the bring-your-own benchmark
compares TurboHybrid with your existing retrieval SQL on your own rows and
query embeddings. See
[docs/benchmarks/bring-your-own-rag.md](docs/benchmarks/bring-your-own-rag.md).

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

- [How TurboHybrid works](docs/how-it-works.md)
- [Easy fast setup](docs/fast_setup.md)
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
