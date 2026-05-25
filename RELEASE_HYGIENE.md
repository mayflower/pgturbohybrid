# Release Hygiene

This scrub prepares the tree for the `v0.1.0-alpha.2` public alpha.

## Moved Or Removed

- Moved architecture, compatibility, attribution-supporting, migration, and
  benchmark material under `docs/`.
- Added polished release-facing benchmark notes in
  `docs/benchmarks/fiqa-openai.md`.
- Added release notes in `docs/release-notes/v0.1.0-alpha.2.md`.
- Removed root-level implementation-history reports for fast defaults and
  performance recovery.
- Moved the logo into `docs/assets/` and updated `README.md`.

## Path Hygiene

Benchmark documentation now uses explicit environment variables:

```sh
FIQA_DATASET=/path/to/fiqa
PGDATABASE=pgturbohybrid_fiqa
OUTPUT=benchmarks/results/pgturbohybrid-fiqa.json
```

The benchmark runner reads `FIQA_DATASET`, `PGDATABASE`, and `OUTPUT`, and no
release-facing docs retain developer machine paths.

## Generated Artifacts

Generated full benchmark outputs are not tracked. The repository keeps only
small static fixtures and benchmark configuration files required for examples,
tests, and validation:

- `benchmarks/config/*.json`
- `benchmarks/examples/*.trec`
- `test/expected/*.out`

Generated benchmark JSON, CSV, and Markdown outputs are ignored under benchmark
result directories.

## Ignore Coverage

`.gitignore` covers:

- benchmark result and output directories
- generated benchmark JSON, CSV, and Markdown outputs
- local pgvector dependency checkouts
- test clusters and regression output
- build products
- Python bytecode
- editor and OS files

## Verification

`git status --ignored` was run after the scrub. Ignored generated artifacts do
not appear as tracked files.
