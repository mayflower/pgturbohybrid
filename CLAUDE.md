# CLAUDE.md

Guidance for working in this repository — the `pgturbohybrid` PostgreSQL
extension (a `turbohybrid` index access method over pgvector: dense-vector +
BM25 + multivector/ColBERT retrieval).

## Development environment: an open `nix develop` (PostgreSQL 17)

The Nix flake (`flake.nix`) dev shell is the **canonical build/test
environment**: PostgreSQL **17**, a pinned pgvector, and the extensions built
from the working tree, plus a local throwaway cluster and `th-*` helper
commands. The flake builds the extensions from `./.`, so the helper commands
rebuild from current source and test against PG17 — this is the reproducible
path, and the source of truth for behavior, recall, and benchmarks.

**Assume a `nix develop` shell is already open and active. Do not start another
one yourself.** The first evaluation/realization is slow, and a second
shell/cluster only duplicates what is already there. Each tool/Bash call you
make is a *fresh* shell **without** the nix env, so reach the open environment
one of these ways:

- **Interactive or long-running steps — ask the user to run it in their open dev
  shell.** In Claude Code they can run it inline with the `! <command>` prefix
  (e.g. `! th-installcheck`). This uses the shell that's already realized.
- **Programmatically from a tool shell —** prefix with `nix develop -c
  <command>` (the `nix` binary is at `/nix/var/nix/profiles/default/bin/nix`
  when not on PATH). It reuses the already-realized dev shell and the same local
  cluster, but **may rebuild from source and can be slow** the first time after
  an edit — prefer the user's open shell for anything lengthy.

### What the dev shell provides

Env vars: `PG_CONFIG` → the nix PG17 `pg_config`; `PGPORT=55432`,
`PGDATABASE=pgturbohybrid_dev`, `PGUSER=postgres`. `PGDATA`/`PGHOST` live under
`.nix-dev/$TH_ENV_NAME/` (`TH_ENV_NAME` is `pg17-pgvector-v0.8.2` for the stable
shell, `pg17-pgvector-master` for the master one), with the unix socket at
`.nix-dev/$TH_ENV_NAME/run`.

Helper commands (run in the shell, or via `nix develop -c`):

| command | purpose |
| --- | --- |
| `th-pg-init` | init + start the local PG17 cluster and install all four extensions (vector, pgturbohybrid, llama_embed, pg_colbert_llama) |
| `th-pg-start` / `th-pg-stop` / `th-pg-reset` | manage / recreate the local cluster |
| `th-psql` | connect to `pgturbohybrid_dev` |
| `th-smoke` | minimal vector + pgturbohybrid query |
| `th-test` | smoke + SQL regression |
| `th-installcheck` | SQL regression (`pg_regress`) |
| `th-prove-installcheck` | TAP tests |
| `th-colbert-build-stub` / `th-colbert-test-stub` | ColBERT/`llama_embed` with the **stub** engine (no llama.cpp) |
| `th-colbert-build-llama` / `th-colbert-live-test` | build against real **llama.cpp** + gated live tests |
| `th-bench-*` | deterministic benchmark helpers |

### Talking to the running dev cluster from a tool shell

Once `th-pg-init` has been run (in the open shell), the PG17 server persists and
is reachable directly — no nix env needed:

```sh
psql -h "$PWD/.nix-dev/pg17-pgvector-v0.8.2/run" -p 55432 -U postgres -d pgturbohybrid_dev
```

(Adjust the `.nix-dev/...` suffix if the master shell is in use.)

## System PostgreSQL 18 is *not* the dev env

There is also a **system PostgreSQL 18** (`/usr/bin/pg_config`, port 5432, socket
auth as `ubuntu`). It is handy for a quick compile check (`make all`), but it is
a **different major version** than the nix env. Do not present system-PG18
results as the dev environment's, and do not mix the two. For anything
behavioral (SQL regression, recall, benchmarks) use the nix PG17 env.

## Build / test quick reference

- Build the main extension: **`make all`** — the bare `make` default goal only
  builds the first object, so always say `make all`. SIMD matrix (amd64 sanity):
  `make all SIMD_BUILD={native,portable,none}`.
- Regression (canonical, PG17): `th-installcheck` / `th-test`. Ad-hoc against
  system PG18: `sudo make install && make installcheck` (you manage the install).
- `pg_colbert_llama` is a separate extension in `extensions/pg_colbert_llama`
  (own Makefile; `PG_COLBERT_LLAMA_ENGINE=stub` by default, `=llama` needs a
  llama.cpp backend). Its regression suite shares one DB, so run `llama_embed`
  before `pg_colbert_llama` only with that ordering in mind.
- **Do not run `make` / `make clean` in the working tree while a nix
  `th-installcheck` / `pg_regress` run is in progress.** `pg_regress` writes to
  `./results/` in the working tree, and `make clean` deletes that directory —
  the collision shows up as a spurious `cannot create …/results/…out.diff:
  Directory nonexistent` bail-out, not a real test failure. Run one at a time.

## Naming conventions

All SQL-visible names use the `turbohybrid.*` (GUCs), `turbohybrid_*`
(functions), and `pgturbohybrid_*` (C symbols) prefixes. Never use `vector_*`,
`hnsw_*`, `tq_*`, or generic `hybrid_*` for exported/SQL-visible names — that
namespace belongs to pgvector and the AM internals.
