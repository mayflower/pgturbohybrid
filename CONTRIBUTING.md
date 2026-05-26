# Contributing

This file helps you set up a local development checkout, run the expected
checks, and send useful changes to `pgturbohybrid`.

`pgturbohybrid` is developed as a standalone PostgreSQL extension that depends
on an unmodified pgvector installation.

## Local Setup

Install PostgreSQL development tools first, then run:

```sh
PG_CONFIG=/path/to/pg_config PGVECTOR_REF=v0.8.2 scripts/dev-install.sh
```

The script clones upstream pgvector into `.deps/`, builds and installs it, then
builds and installs `pgturbohybrid` from this checkout. It creates a fresh
`pgturbohybrid_dev` database and verifies:

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pgturbohybrid;
```

Use `PGVECTOR_REF=master` to test against pgvector development head. Use
`PGVECTOR_REPO` only when intentionally testing another pgvector remote; do not
edit the pgvector checkout as part of pgturbohybrid development.

## Useful Commands

```sh
scripts/build-pgvector.sh
scripts/install-pgvector.sh
scripts/dev-install.sh
scripts/run-tests.sh
scripts/run-tap.sh
scripts/clean-dev-db.sh
```

All scripts accept `PG_CONFIG`. Scripts that touch pgvector also accept
`PGVECTOR_REF`.

## Test Expectations

Run SQL regression tests before sending changes:

```sh
make installcheck
```

Run TAP tests when PostgreSQL TAP modules are available:

```sh
make prove_installcheck
```

The tests should validate stable user-visible behavior: extension install/drop,
index creation, query ordering, restart/recovery behavior, and compatibility
with pgvector as a required extension.

If your change includes a benchmark claim, include the dataset, embedding
dimensions, query count, index settings, candidate budgets, baseline, and
quality metrics. Generated benchmark JSON, CSV, and Markdown outputs should
stay out of the repository.

Default builds use `MATH_MODE=strict` so floating-point validation and ranking
behavior stay conservative. Use `MATH_MODE=fast` only for explicit performance
experiments, and state that build setting in any benchmark claim.

The `hardening` GitHub Actions workflow is manual/nightly. Use it for release
preparation or memory-safety work; it covers a strict SIMD-disabled build, gcc,
and clang static analysis without slowing every push.
