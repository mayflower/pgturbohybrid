# Developer Benchmark Tools

The scripts in this directory are developer tools. They are useful for local
profiling, regression checks, and smoke tests, but their output is not a public
performance claim by itself. Public claims should be written up in
`docs/benchmarks/` with dataset, hardware, profile, index settings, budgets,
baselines, and quality metrics.

Generated JSON, CSV, and Markdown outputs are ignored. Set `OUTPUT` when you
want a specific artifact path; otherwise scripts write under
`benchmarks/results/`.

## Environment

Common variables:

```sh
export PGDATABASE=pgturbohybrid_fiqa
export FIQA_DATASET=/path/to/beir/fiqa
export OUTPUT=benchmarks/results/fiqa-openai.json
export PG_CONFIG=pg_config
export PGVECTOR_REF=v0.8.2
```

`PGDATABASE`, `FIQA_DATASET`, and `OUTPUT` control the database, dataset, and
result artifact. `PG_CONFIG` and `PGVECTOR_REF` are used by scripts that build
or install pgvector or `pgturbohybrid`.

`compare_old_branch.sh` runs two databases. Use `PGDATABASE` or
`CURRENT_PGDATABASE` for the current worktree database, and `OLD_PGDATABASE`
for the optional old patched-branch database.

## Perf Smoke

Use the quick FIQA runner for a small real-data check:

```sh
FIQA_DATASET=/path/to/beir/fiqa \
PGDATABASE=pgturbohybrid_fiqa_quick \
OUTPUT=benchmarks/results/fiqa-quick.json \
benchmarks/dev/run_fiqa_quick.sh
```

Set `MAX_DOCS` and `MAX_QUERIES` to tune the smoke size. Set
`INSTALL_PGVECTOR=1` or `BUILD_PGTURBOHYBRID=1` when you want the script to
prepare dependencies before the run.

## Full FIQA Benchmark

For a full FIQA/OpenAI matrix:

```sh
FIQA_DATASET=/path/to/beir/fiqa \
PGDATABASE=pgturbohybrid_fiqa \
OUTPUT=benchmarks/results/fiqa-openai-full.json \
benchmarks/dev/run_perf_matrix.sh
```

This writes the main JSON artifact to `OUTPUT` and a companion category summary
beside it. Treat the result as developer evidence until it is reviewed and
summarized in `docs/benchmarks/fiqa-openai.md`.

Set `DEV_DIAGNOSTICS=1` only when you also provide a valid
`DEV_DIAGNOSTICS_SQL` file for local diagnostic objects. The default path does
not require developer-only SQL objects.

## Old-Branch Comparison

`compare_old_branch.sh` is for development comparisons against an older patched
pgvector branch. It is not part of public release claims.
