<p align="center">
  <img src="logo.png" alt="pgturbohybrid" width="720">
</p>

# pgturbohybrid

`pgturbohybrid` is an experimental PostgreSQL extension for dense vector and
BM25 hybrid retrieval on top of pgvector's `vector` type. It installs as its
own extension, builds its own shared library, and requires an already installed
unmodified `vector` extension.

## Status

This project is alpha software. The SQL API, on-disk index format, and
performance profile may change before a stable release. Use it for evaluation
and controlled experiments until the compatibility and recovery matrix is
green on your target PostgreSQL and pgvector versions.

## Requirements

- PostgreSQL 14 through 18 are tested in CI. PostgreSQL 19 is included when the
  setup action provides it.
- pgvector 0.8.2 and pgvector `master` are tested in CI.
- PGXS build tooling for the target PostgreSQL installation.
- A C compiler supported by PostgreSQL. Portable SIMD is the default; the build
  does not use `-march=native` unless requested.

`pgturbohybrid` uses pgvector's SQL `vector` type but does not link against
private pgvector C symbols.

## Why This Exists

`pgturbohybrid` is a companion extension for pgvector. It is not a fork of
pgvector and is not an official pgvector project. The goal is to experiment with
hybrid dense-vector and BM25 retrieval while keeping pgvector itself
unmodified.

## Installation

Install pgvector first:

```sh
git clone --depth 1 --branch v0.8.2 https://github.com/pgvector/pgvector.git /tmp/pgvector
make -C /tmp/pgvector
make -C /tmp/pgvector install
```

Then build and install `pgturbohybrid`:

```sh
git clone https://github.com/mayflower/pgturbohybrid.git
cd pgturbohybrid
make
make install
```

Create both extensions in the database:

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pgturbohybrid;
```

For a repeatable local setup:

```sh
PG_CONFIG=pg_config PGVECTOR_REF=v0.8.2 scripts/dev-install.sh
```

## Minimal Example

```sql
CREATE TABLE documents (
    id bigserial PRIMARY KEY,
    embedding vector(3) NOT NULL,
    body text NOT NULL,
    body_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', body)) STORED
);

INSERT INTO documents (embedding, body)
VALUES
    ('[1,0,0]', 'postgres vector search'),
    ('[1,1,0]', 'hybrid search with bm25'),
    ('[0,1,0]', 'lexical search in postgres');

CREATE INDEX documents_pgturbohybrid_idx ON documents
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
);

SELECT id, body
FROM documents
ORDER BY embedding <~> turbohybrid_query(
    vector_query => '[1,0,0]'::vector,
    text_query => websearch_to_tsquery('english', 'postgres hybrid search'),
    dense_k => 10,
    bm25_k => 10,
    final_k => 3
)
LIMIT 3;
```

Use `<~->` for L2, `<~#>` for negative inner product, and `<~>` for cosine
hybrid ordering.

## Index Options

The public index options are:

- `graph_m`: maximum graph connections.
- `graph_ef_construction`: graph candidate list size during build.
- `graph_ef_search`: graph candidate list size during scans.
- `graph_oversampling`: graph candidate oversampling multiplier.
- `quantization_bits`: quantized dense-vector code width.
- `exact_storage`: store exact vectors in the index for final exact rescoring.
- `routing`: dense routing mode.

## Query Parameters

`turbohybrid_query(...)` accepts:

- `vector_query`: dense pgvector query vector.
- `text_query`: PostgreSQL `tsquery` for BM25 retrieval.
- `fusion`: currently `rrf` for reciprocal-rank fusion.
- `dense_weight` and `bm25_weight`: branch weights for score fusion.
- `rrf_k`: RRF constant.
- `dense_k`: dense branch candidate budget.
- `bm25_k`: BM25 branch candidate budget.
- `final_k`: final result target.
- `require_bm25_match`: only return rows with a BM25 lexical match.

## Settings

The public GUCs are intentionally small:

- `turbohybrid.default_dense_k`
- `turbohybrid.default_bm25_k`
- `turbohybrid.default_rrf_k`
- `turbohybrid.enable_wand`
- `turbohybrid.max_union_candidates`
- `turbohybrid.simd`

## Diagnostics

The installed diagnostic API is limited to:

- `turbohybrid_index_stats(regclass)`
- `turbohybrid_last_scan_stats()`
- `turbohybrid_simd_capabilities()`

These return stable JSONB summaries for index metadata, the last scan in the
current backend, and build/host SIMD capability information.

## WAL And Recovery

`pgturbohybrid` uses PostgreSQL generic WAL for index page changes. It does not
register a custom resource manager and does not require
`shared_preload_libraries = 'pgturbohybrid'`.

## Benchmarks

Benchmark methodology and reproducibility requirements live in
[benchmarks/README.md](benchmarks/README.md). Performance claims should use
real embedding datasets with relevance labels, not synthetic vector generators.
Generated result JSON and Markdown files should be written outside the
repository or under ignored benchmark output directories.

## Compatibility Notes

`pgturbohybrid` requires pgvector at build and install time. It refers to
pgvector's `vector` type through the extension dependency search path, so
pgvector can be installed outside `public` on supported PostgreSQL versions.

The compatibility layer validates that the `vector` extension is installed and
that vector dimensions are valid before accessing vector payloads. If pgvector's
type layout changes, the build or runtime checks should fail with a direct
compatibility error rather than silently reading malformed data.

See [COMPATIBILITY.md](COMPATIBILITY.md) for the tested PostgreSQL and pgvector
matrix.

## Known Limitations

- Alpha status: SQL APIs and the on-disk index format may change before a
  stable release.
- On-disk format: incompatible storage changes require `REINDEX`; see
  [RELEASE.md](RELEASE.md) for the current policy.
- pgvector ABI: the extension depends on pgvector's SQL `vector` type and uses
  a private compatibility copy of its varlena layout. Tested compatibility
  starts at pgvector 0.8.2.
- Dimensions: the compatibility layer supports pgvector vector payloads up to
  16000 dimensions; index build paths validate their supported limits before
  writing index pages.
- Text-aware scalar fallback: hybrid queries with a `text_query` are intended
  for indexed `ORDER BY ... LIMIT` retrieval. Scalar projection without an
  index path fails with a `feature_not_supported` error.
- Production support: treat this as evaluation software until the release
  checklist is green for your target platform and PostgreSQL version.

## Attribution

This project depends on pgvector and contains code derived from pgvector's HNSW
implementation. pgvector is an excellent PostgreSQL vector search extension;
`pgturbohybrid` is a separate experimental extension built on top of it and is
not an official pgvector project.

The license is preserved in [LICENSE](LICENSE), with attribution notes in
[NOTICE](NOTICE).
