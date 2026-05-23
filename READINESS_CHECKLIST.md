# pgturbohybrid Standalone Readiness Checklist

This checklist records the local release gate for the standalone
`pgturbohybrid` extension.

## Checklist

- [x] repository is no longer a pgvector fork structurally
- [x] extension name is `pgturbohybrid`
- [x] shared library is `pgturbohybrid`
- [x] control file is `pgturbohybrid.control`
- [x] SQL install script is `pgturbohybrid--0.1.0.sql`
- [x] control file has `requires = 'vector'`
- [x] no pgvector control or SQL files are modified
- [x] no pgvector types are created
- [x] no pgvector operators/opclasses are replaced
- [x] all public SQL objects are owned by `pgturbohybrid` and exposed under the
  `turbohybrid` feature surface
- [x] all GUCs use `turbohybrid.*`
- [x] Makefile builds only the `pgturbohybrid` shared library
- [x] Makefile.win is scoped to `pgturbohybrid.dll`
- [x] CI installs unmodified pgvector first
- [x] CI tests pgvector `v0.8.2` and pgvector `master`
- [x] `CREATE EXTENSION vector; CREATE EXTENSION pgturbohybrid;` works locally
- [x] non-public schema install is covered by regression tests
- [x] `DROP EXTENSION pgturbohybrid` leaves pgvector installed and usable
- [x] SQL regression tests pass locally
- [ ] TAP tests pass locally
- [ ] restart/WAL tests pass locally
- [ ] Windows build passes
- [ ] macOS matrix passes
- [x] local macOS build passes
- [ ] Linux matrix passes
- [ ] valgrind/UBSan passes where available
- [x] README is standalone and does not claim to be official pgvector
- [x] benchmark results are reproducible but not committed as generated artifacts
- [x] compatibility with pgvector versions is documented
- [x] license and pgvector attribution are documented

## Remaining Blockers

- Local TAP tests are present but skipped because this PostgreSQL PGXS
  installation does not include PostgreSQL TAP Perl modules.
- Windows, Linux matrix, and valgrind/UBSan results still need to be proven by
  GitHub Actions or equivalent platform runners.
- Restart/WAL TAP behavior is implemented as a test target but is not locally
  proven until TAP modules are available.

## Remaining Risks

- `pgturbohybrid` uses a private compatibility copy of pgvector's `Vector`
  layout. The install script requires pgvector 0.8.2 or newer and runtime checks
  validate dimensions and extension presence, but pgvector ABI drift remains the
  primary compatibility risk.
- The current smoke query forces index usage with `SET enable_seqscan = off`
  because scalar hybrid text distance intentionally rejects text-aware fallback
  evaluation.
- The source archive target packages the current non-ignored working tree. A
  tagged release should be built from a clean committed tree.

## Release Readiness Verdict

Release is blocked until CI proves TAP/restart, Windows, Linux, and
valgrind/UBSan coverage. The local PostgreSQL 16/macOS source build,
installation, regression suite, benchmark smoke, package archive, and standalone
SQL smoke all pass.

## Exact Release Tag Name

`v0.1.0`

## Exact README Installation Snippet

```sh
git clone --depth 1 --branch v0.8.2 https://github.com/pgvector/pgvector.git /tmp/pgvector
make -C /tmp/pgvector
make -C /tmp/pgvector install

make
make install
```

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pgturbohybrid;
```

## Exact Smoke-Test SQL

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pgturbohybrid;
SET enable_seqscan = off;

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

DROP INDEX documents_pgturbohybrid_idx;
DROP EXTENSION pgturbohybrid;
SELECT '[1,2,3]'::vector AS vector_still_available;
DROP TABLE documents;
DROP EXTENSION vector;
```

## Local Validation Run

- `PGVECTOR_REF=v0.8.2 ./scripts/install-pgvector.sh`: passed.
- `make clean && make && make install && make installcheck`: passed.
- `make prove_installcheck`: TAP tests discovered but skipped with `NOTESTS`
  because PostgreSQL TAP modules are unavailable in this local PGXS install.
- Standalone smoke SQL in a clean database: passed.
- `python3 benchmarks/suite.py run-system-synthetic --rows 100 --dimensions 8
  --runs 1 --warmup 0`: passed.
- `make dist`: passed; archive sweep found no pgvector control, SQL upgrade, or
  pgvector source files in the generated source archives.
