# Benchmarks

Run benchmarks from the repository's Nix environment. Benchmark results are
local evidence, not committed release policy; keep generated data and reports
under `.nix-dev/`.

## Dense and BM25

The deterministic retrieval grid exercises the supported dense and BM25 paths
against a live PostgreSQL instance:

```sh
nix --extra-experimental-features 'nix-command flakes' develop
th-pg-init
th-bench-retrieval-quality
```

Use `benchmarks/fiqa_openai.py` when a real FIQA/OpenAI comparison is needed.
Its command-line help documents the dataset and connection arguments:

```sh
nix --extra-experimental-features 'nix-command flakes' develop .#bench
uv run benchmarks/fiqa_openai.py --help
```

Reports should include dataset and model provenance, PostgreSQL and extension
versions, the compared configurations, recall or nDCG at the requested K, and
p50/p95 latency. Compare runs at equivalent quality.

## ColBERT multivector

The DBpedia benchmark uses the small validation model configured by the flake
and a separate llama-backed PostgreSQL cluster. Keep the GGUF under
`.nix-dev/models/colbert-15m/`.

```sh
nix --extra-experimental-features 'nix-command flakes' develop .#bench
DBPEDIA_DATASET=/path/to/qdrant-dbpedia \
BEIR_DBPEDIA_DATASET=/path/to/beir-dbpedia \
PG_COLBERT_LLAMA_TEST_MODEL=.nix-dev/models/colbert-15m/model.gguf \
  th-bench-dbpedia-colbert
```

The wrapper defaults to smoke-sized input. Inspect its current arguments before
a larger run:

```sh
th-bench-dbpedia-colbert --help
```

The supported serving contract is document-level candidate generation followed
by bounded exact float32 MaxSim reranking. Exact scans are benchmark oracles,
not selectable production strategies.

## Developer microbenchmarks

Focused SQL and Python programs under `benchmarks/dev/` and this directory may
be used to diagnose a measured regression. They are not a parallel acceptance
framework and do not define release thresholds.
