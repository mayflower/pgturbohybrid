# Contributing

This file helps you set up a local development checkout, run the expected
checks, and send useful changes to `pgturbohybrid`.

`pgturbohybrid` is developed as a standalone PostgreSQL extension that depends
on an unmodified pgvector installation.

## Reporting Issues And Opening PRs

Use the GitHub templates so reports carry enough context to reproduce:

- [Bug report](.github/ISSUE_TEMPLATE/bug_report.yml) — PostgreSQL/pgvector/pgturbohybrid versions, OS/arch, `CREATE INDEX`, query, and `EXPLAIN`.
- [Feature request](.github/ISSUE_TEMPLATE/feature_request.yml).

Pull requests use [the PR template](.github/PULL_REQUEST_TEMPLATE.md) checklist
(tests, docs, release notes, benchmark claims, and compatibility/`REINDEX`
impact).

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

## Nix Development Shell

For a reproducible local environment with PostgreSQL, pgvector, and
`pgturbohybrid` packaged together:

```sh
nix develop          # add --extra-experimental-features 'nix-command flakes' if needed
th-pg-init           # init + start the local PG17 cluster, install all extensions
th-smoke             # minimal vector + pgturbohybrid query
th-psql              # connect to pgturbohybrid_dev
```

The default shell uses PostgreSQL 17 and pgvector v0.8.2 and keeps its cluster
under `.nix-dev/pg17-pgvector-v0.8.2/`, so it never touches a system PostgreSQL.
Useful in-shell commands:

```sh
th-pg-start / th-pg-stop / th-pg-reset   # manage the local cluster
th-installcheck                          # SQL regression tests
th-prove-installcheck                    # TAP tests (needs IPC::Run)
th-test                                  # smoke + regression
```

Test against pinned pgvector `master` with `nix develop .#pgvector-master`
(then `th-pg-reset`). After changing extension C/SQL, re-enter the shell (or
re-run `nix develop ... -c ...`) so PostgreSQL sees the rebuilt package.

`nix flake check` is intentionally cheap: it builds the extension, the wrapped
PostgreSQL, the pgvector-master variant, and a scalar `SIMD_BUILD=none` variant.
Update pins explicitly with `nix flake lock --update-input pgvector-master`.

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
