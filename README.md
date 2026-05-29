<p align="center">
  <img src="docs/assets/logo.png" alt="pgturbohybrid logo with a compact hot hatchback theme" width="720">
</p>

# pgturbohybrid

This README helps you understand what `pgturbohybrid` does, when hybrid search
is useful, how to install it, how to create your first index, and how to check
whether the fast path is working.

> The hot hatch of pgvector hybrid search: practical PostgreSQL retrieval with
> a surprising turn of speed.

![Status: alpha](https://img.shields.io/badge/status-alpha-orange)
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

Use `<~->` for L2, `<~#>` for negative inner product, and `<~>` for cosine
hybrid ordering. A longer copy-paste walkthrough lives in
[examples/fast_start.sql](examples/fast_start.sql) and
[docs/fast_setup.md](docs/fast_setup.md).

## Fast Defaults

Fresh sessions use the `latency` profile:

```sql
SHOW turbohybrid.profile;
```

The default path uses a 4-bit quantized index, exact vector storage off, dense
and BM25 candidate budgets of 100, RRF constant 60, and the SQL `LIMIT` as the
final result target when possible. In plain terms: create the default index,
query with `ORDER BY ... turbohybrid_query(...) LIMIT n`, then inspect the scan
stats below.

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

## Quality Mode

Use `quality` when relevance matters more than lowest latency:

```sql
SET turbohybrid.profile = 'quality';
```

Quality mode uses larger dense and lexical candidate budgets and conservative
BM25 paths. For quality-sensitive production evaluation, benchmark an
exact-storage index too:

```sql
CREATE INDEX documents_turbohybrid_quality_idx ON documents
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (exact_storage = on);
```

Do not treat either profile as universally best. Measure latency and relevance
on your dataset.

## Diagnostics

After a query, check whether PostgreSQL used the expected TurboHybrid path:

```sql
SELECT turbohybrid_last_scan_stats();
SELECT turbohybrid_index_stats('documents_turbohybrid_idx'::regclass);
SELECT turbohybrid_simd_capabilities();
```

Useful fields include:

- `profile`
- `index_used`
- `scan_orchestration`
- `quantization_bits`
- `exact_storage`
- `dense_candidates_effective`
- `bm25_candidates_effective`
- `final_k_source`
- `elapsed_us`

For troubleshooting examples, see [docs/fast_setup.md](docs/fast_setup.md).

## Benchmark Snapshot

On this FIQA/OpenAI setup, the public `v0.1.0-alpha.2` snapshot used 57,638
corpus rows, 648 qrels-backed queries, 1,536-dimensional OpenAI
`text-embedding-3-small` embeddings, one warmup pass, one measured pass, and the
`latency` profile. The default TurboHybrid index used 4-bit quantization with
`exact_storage = off`; both hybrid rows used 100 dense candidates, 100 BM25
candidates, and nDCG@10 as the quality metric. nDCG means normalized discounted
cumulative gain, a relevance metric for ranked results.

HNSW means Hierarchical Navigable Small World, pgvector's graph index type for
approximate nearest-neighbor vector search.

| Method | Settings | p95 | nDCG@10 |
| --- | --- | ---: | ---: |
| pgturbohybrid default | default 4-bit exact-free index, LIMIT-inferred `final_k` | 0.910 ms | 0.421540 |
| SQL RRF baseline | pgvector HNSW default reloptions plus PostgreSQL GIN (Generalized Inverted Index) full-text search, 100/100 candidates | 3.254 ms | 0.423341 |

There is also a larger Qdrant DBPedia OpenAI3-large 1M developer benchmark for
systems testing with the dataset's existing 3,072-dimensional embeddings. It is
useful for build time, memory, index-size, and retrieval-path experiments, but
the default self-query setup is not an external relevance benchmark. We keep
those results out of the README until the quality story is backed by a
reproducible, externally labeled run.

These are specific benchmark snapshots, not global pgvector comparisons.
Results vary by dataset, hardware, PostgreSQL settings, and query workload.
Benchmark details, baselines, and reproduction notes are in
[docs/benchmarks/fiqa-openai.md](docs/benchmarks/fiqa-openai.md),
[benchmarks/dbpedia_openai3_large.md](benchmarks/dbpedia_openai3_large.md), and
[benchmarks/README.md](benchmarks/README.md).

If you already have a PostgreSQL RAG database, the bring-your-own benchmark
compares TurboHybrid with your existing retrieval SQL on your own rows and
query embeddings. See
[docs/benchmarks/bring-your-own-rag.md](docs/benchmarks/bring-your-own-rag.md).

## How It Works, Short Version

`pgturbohybrid` defines a `turbohybrid` PostgreSQL index access method over:

- a pgvector `vector` column for dense retrieval
- a generated `tsvector` column for lexical retrieval

At query time, it gathers dense and BM25-style lexical candidates, fuses them
with reciprocal-rank fusion, and returns rows through PostgreSQL's normal
`ORDER BY ... LIMIT` index-scan shape.

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

The release target is PostgreSQL 14 through 18, with PostgreSQL 19 included
when the CI setup action provides it. The pgvector compatibility target is
pgvector 0.8.2 through current pgvector `master`.

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
