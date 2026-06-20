# llama_embed Benchmarks

`colbert_ingest_search.sql` is a small local benchmark harness for measuring:

- document encoding throughput with `llama_embed_mv('alias', body, '{"mode": "tokens", "prefix": "[D] "}'::jsonb)`
- query encoding latency with `llama_embed_mv('alias', query, '{"mode": "tokens", "prefix": "[Q] "}'::jsonb)`
- dense-only pgturbohybrid multivector search
- hybrid multivector + BM25 search using `fusion = 'rrf'`
- scan diagnostics through `turbohybrid_last_scan_stats()`
- index memory estimates through `turbohybrid_estimate_memory(index)`

Run it in a database where `vector`, `pgturbohybrid`, and `llama_embed` are
installed:

```sh
psql -v MODEL_ALIAS=sauerkraut-modern -f extensions/pg_colbert_llama/bench/colbert_ingest_search.sql
```

For a stub-engine syntax check, pass `-v EXPECTED_DIM=4`. For live ColBERT
models, leave it unset or pass `-v EXPECTED_DIM=128`.

Do not commit benchmark output.
