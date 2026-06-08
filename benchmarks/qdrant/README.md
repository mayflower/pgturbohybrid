# DBpedia ColBERT Qdrant Comparison

This benchmark compares `pgturbohybrid` and Qdrant with the same precomputed
DBpedia ColBERT multivectors.

The PostgreSQL side imports the compact DBpedia ColBERT parquet export, builds a
document-node index, and refuses to continue unless the index is using the
scalable proxy build path:

```sql
WITH (
  multivector_graph = document_nodes,
  multivector_doc_build_scorer = proxy,
  multivector_proxy_encoder = normalized_mean,
  exact_storage = off,
  quantization_bits = 4
)
```

The Qdrant side creates one multivector point per document:

```python
VectorParams(
    size=128,
    distance=Distance.COSINE,
    multivector_config=MultiVectorConfig(comparator=MultiVectorComparator.MAX_SIM),
)
```

## Dependencies

Use the repository benchmark shell when possible:

```sh
nix --extra-experimental-features 'nix-command flakes' develop .#bench
```

If you run outside Nix, install:

```sh
python -m pip install -r benchmarks/qdrant/requirements.txt
```

Start Qdrant before running the full comparison:

```sh
docker run --rm -p 6333:6333 qdrant/qdrant:v1.14.1
```

## Data

The script expects the compact DBpedia ColBERT export produced by
`benchmarks/dbpedia_colbert_hf_dataset.py`. By default it first looks at:

```text
.nix-dev/hf-datasets/dbpedia-colbert-multivector-1m-f16
```

If that path does not exist, it falls back to the Hugging Face dataset repo:

```text
johannhartmann/pgturbohybrid_dbpedia_colbert
```

Override this with `--precomputed-dataset /path/or/repo`.

## Commands

Run a 1k smoke comparison:

```sh
nix --extra-experimental-features 'nix-command flakes' develop .#bench --command \
  python benchmarks/qdrant/dbpedia_colbert_qdrant_compare.py \
    --create-database \
    --max-docs 1000 \
    --max-queries 32
```

Run the 10k comparison:

```sh
nix --extra-experimental-features 'nix-command flakes' develop .#bench --command \
  python benchmarks/qdrant/dbpedia_colbert_qdrant_compare.py \
    --create-database \
    --max-docs 10000 \
    --max-queries 64
```

For a PostgreSQL-only validation run, pass `--pg-only`. Without `--pg-only`, the
script refuses to run if Qdrant is not reachable at `--qdrant-url`.

## Output

The script prints a comparison table with:

- build/index time;
- insert/load time;
- index/storage size where available;
- recall@10 against brute-force exact MaxSim on the configured subset;
- p50/p95/p99 query latency;
- pgturbohybrid exact rerank docs and pairs;
- pgturbohybrid index/build/scan stats JSON in the result file;
- Qdrant collection metadata and client-side timings.

JSON results are written under `benchmarks/qdrant/results/` by default. Generated
results and local Qdrant storage directories are ignored by git.

## Safety Checks

The pgturbohybrid run stops before reporting benchmark numbers if:

- `turbohybrid_index_stats()` reports
  `multivector_doc_build_scorer != proxy`;
- `turbohybrid_last_build_stats()` reports
  `multivector_doc_exact_build_distance_calls > 0`;
- `turbohybrid_index_stats()` reports `build_fast_edges = false`;
- `turbohybrid_index_stats()` reports a `node_count` different from the loaded
  document count;
- the index is not in `multivector_graph_mode = document_nodes`.

This prevents accidental comparisons against the diagnostic exact symmetric
document-document MaxSim topology path or a slow heuristic edge build.
