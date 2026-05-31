# pgturbohybrid Benchmarks

Benchmark comparisons for `pgturbohybrid` must use real embedding datasets and
documented relevance metrics. The benchmark setup treats pgvector as an
upstream dependency and installs `pgturbohybrid` as a separate extension. It
does not require a patched pgvector checkout.

Public benchmark explanations live under `docs/benchmarks/`. This directory is
for reproducible tooling, acceptance thresholds, and developer benchmark
helpers. Developer-only workflows are documented in `benchmarks/dev/README.md`.

If you already have a PostgreSQL RAG database and want a quick local comparison,
use `benchmarks/rag_existing.py`. It compares TurboHybrid with your own
retrieval SQL and is documented in
[`docs/benchmarks/bring-your-own-rag.md`](../docs/benchmarks/bring-your-own-rag.md).

The DBPedia OpenAI3-large benchmark spec lives in
[`dbpedia_openai3_large.md`](dbpedia_openai3_large.md). It covers the
1M-row Qdrant DBPedia corpus, BEIR DBPedia queries/qrels, the native pgvector
`halfvec` + PostgreSQL full-text SQL RRF baseline, the TurboHybrid hybrid runs,
and a dense-only default comparison between pgvector HNSW and TurboHybrid.
Internal dense-only experiment decisions from the prompt-pack work live in
[`docs/internal/dbpedia-dense-quality-decision.md`](../docs/internal/dbpedia-dense-quality-decision.md).

## Baselines

Every publishable run should include at least:

- `postgres_sql_rrf`: pgvector HNSW over `vector`, PostgreSQL full-text search
  over `tsvector`, and SQL reciprocal-rank fusion.
- `pgturbohybrid`: one pgturbohybrid index over the same `vector` and
  `tsvector` columns.

When relevant, also compare `pgturbohybrid_recovered_explicit` to document the
latency, storage, and quality tradeoff from the recovered fast settings: 4-bit,
`exact_storage = off`, 100/100/60, and `final_k = 10`. The package default keeps
adaptive dense widening off; the harness includes explicit adaptive variants for
controlled recovery experiments. The harness still accepts
`pgturbohybrid_exact_storage_off` as a legacy alias for older artifacts.

## Profile Matrix

Publishable FIQA/OpenAI results should include all of:

- `pgturbohybrid`: latency profile, default index options, omitted query
  budgets, and LIMIT-inferred `final_k`.
- `pgturbohybrid_recovered_explicit`: latency profile, 4-bit index,
  `exact_storage = off`, effective `dense_k = 100`, `bm25_k = 100`, and
  `final_k = 10`.
- `pgturbohybrid_adaptive_auto_2_0`: same FIQA/OpenAI latency-profile index
  and budgets, with adaptive dense widening explicitly enabled in `auto` mode
  at multiplier `2.0`. Use this as an opt-in quality recovery diagnostic, not
  as the package default.
- `pgturbohybrid_quality`: quality profile, effective `dense_k = 400`,
  `bm25_k = 400`, exact-safe BM25 paths, SIMD enabled, and a documented
  `exact_storage` choice. Prefer `exact_storage = on` when evaluating final
  quality-sensitive settings.
- `postgres_sql_rrf`: pgvector HNSW plus PostgreSQL full-text search fused in
  SQL.

The result summary must compare quality profile against latency profile for
both relevance and latency. Do not infer quality-profile relevance from the
latency-profile artifact.

Validate a generated matrix artifact with:

```sh
FIQA_DATASET=/path/to/fiqa
PGDATABASE=pgturbohybrid_fiqa
OUTPUT=benchmarks/results/pgturbohybrid-fiqa.json
python3 benchmarks/tools/check_acceptance.py \
  "$OUTPUT" \
  --suite fiqa_openai_profile_matrix
```

## Publishable Run Metadata

Do not commit generated benchmark outputs. Store JSON/Markdown in an external
artifact store or ignored directories such as `benchmarks/results/`.

Record the following with any published result:

- hardware and CPU governor
- operating system
- PostgreSQL version and settings
- pgvector ref or release
- pgturbohybrid commit
- dataset and embedding model
- corpus size, query count, and embedding dimensions
- retrieval profile
- index settings and reloptions
- candidate budgets, fusion settings, and final result target
- exact commands
- warmup policy and measured run count
- p50, p95, p99, QPS
- index size, build time, and WAL generated
- build provenance for fresh, shared, and existing-index runs
- exact-build provenance from TurboHybrid index stats when applicable
- quality metrics such as recall, nDCG, MRR, MAP, or overlap
- baseline definitions and index options
- note that results vary by dataset and hardware

## Benchmarking dense kernels

The dense scoring kernel is auto-selected per host (the best available SIMD tier
plus the u8 x4 batch path). To compare kernels apples-to-apples, pin the
implementation with the developer/benchmark GUCs (the "Developer / benchmark"
category in
[`docs/architecture.md`](../docs/architecture.md#gucs-and-reloptions)) and
confirm the kernel that actually ran with `turbohybrid_last_scan_stats()`. All
combinations below return identical results; only the kernel that produces them
changes -- so a difference in timing is a kernel difference, not a recall one.

`dense_query_split_impl` and `dense_u8_split` default to `auto`; set them only to
override the auto choice. Apply the `SET`s per session, then run the workload.

| Target kernel | GUCs to set |
| --- | --- |
| Host best (default) | leave all unset (`auto` -> best tier + u8 x4) |
| AVX-512 VNNI, u8 x4 | `dense_graph_avx512vnni=on; dense_graph_avxvnni=on; dense_u8_split=auto; dense_u8_batch_x4=on` |
| AVX2, u8 x4 | `dense_graph_avx512vnni=off; dense_graph_avxvnni=off; dense_u8_split=auto; dense_u8_batch_x4=on` |
| u8 x4 off (four single-node u8 passes) | same as the chosen tier, but `dense_u8_batch_x4=off` |
| Signed split (no u8) | `dense_u8_split=off` (signed-codebook split at the host's tier) |
| Scalar / LUT fallback | `turbohybrid.simd=off` (the non-SIMD reference) |

Forcing AVX-512 off always selects the AVX2 kernels on any AVX2 host; on a host
without AVX-512 VNNI the AVX-512-on row resolves to those same AVX2 kernels.

Verify the active kernel after a scan:

```sql
SELECT turbohybrid_last_scan_stats() ->> 'dense_scorer';      -- e.g. unsigned_split_avx512vnni
SELECT turbohybrid_last_scan_stats() -> 'dense' -> 'kernels'; -- u8_batch_mode, u8_kernel_single/_batch
```

Ready-made harnesses live next to this README:

- `native_hotpath_bench.sql` -- live native-scan latency, u8 x4 on vs off.
- `u8_x4_kernel_microbench.sql`, `u8_split_microbench.sql` -- kernel-level
  ns/code microbenchmarks.
- `native_scan_kernel_stats.sql` -- per-bucket kernel attribution from a real
  scan.
- `rescore_band_latency.sql` -- exact-rescore (`dense_rescore_band`) latency.
- `concurrency_dense_bench.sql` / `concurrency_dense_bench.sh` -- concurrent-client
  scaling diagnostics (see below).

## Concurrent-client scaling diagnostics

`concurrency_dense_bench.sql` (driven by `concurrency_dense_bench.sh`) explains
why dense-default throughput can scale *down* with concurrent clients on
glove-100-angular (observed ~325 RPS / p95 4.4 ms at 1 client collapsing to
~127 RPS / p95 179 ms at 8 clients) while pgvector and Qdrant scale up. It does
not change query behaviour -- it only measures, so the cause is known before any
algorithm is touched.

It drives the same fixed query set at 1/2/4/8/16 concurrent backends (each a real
backend opened via `dblink`, so each holds its own per-backend native cache --
the thing under test) across three native cache caps (`native_cache_max_mb` ∈
{0 = uncached, 512 = default, high}) and two prewarm modes (A = cold, B = cache
prebuilt), and attributes the p95 explosion to one of four causes using
per-backend instrumentation from `turbohybrid_last_scan_stats()`:

| Suspected cause | Signal the harness reads |
| --- | --- |
| Cold per-backend cache build | `native_cache_built_this_scan` / `native_cache_build_us` on the cold query; removed by prewarm mode B |
| Cache duplication / memory bandwidth | `native_cache_bytes` per backend × clients (`total_cache_bytes`); warm, 0 page reads, `graph_total_us` rising with clients; cache ON collapses while OFF scales |
| Lock waits | `pg_stat_activity` `wait_event_type='Lock'/'LWLock'` sampled during the run; ungranted `pg_locks` on the index |
| Traversal / scoring CPU | `graph_batch_us` / `graph_traverse_us` per query rising with clients, no waits, ~0 page reads |

The `native_cache_*` keys it relies on are emitted by
`turbohybrid_last_scan_stats()` (flat keys and under `dense.cache`):
`native_cache_mode` (`per_backend` / `uncached` / `none`),
`native_cache_built_this_scan`, `native_cache_build_us`, and
`native_cache_bytes` with a `code`/`adj`/`exact` breakdown.

```bash
# Against the loaded glove-100-angular dataset (real collapse):
benchmarks/concurrency_dense_bench.sh pgturbohybrid_benchmark

# Or build + run a synthetic glove-sized dataset in a throwaway DB:
CCB_NROWS=1183514 CCB_DIMS=100 \
  benchmarks/concurrency_dense_bench.sh pgturbohybrid_ccbench
```

It prints three tables: the **scaling curve** (RPS + p50/p95/p99 vs clients), a
**cause-attribution** table (cold build µs, wait %, code pages, scoring µs, cache
bytes per backend × clients), and a heuristic **verdict** per config. It measures
server-side latency via `dblink` (no client round-trip), so absolute RPS differs
from the Python `vector-db-benchmark` numbers; the scaling *shape* and the
per-backend cause signals are the point. Reproducing the per-backend cache
collapse needs a glove-sized index (~1.18M × 100) -- a small synthetic dataset
stays cache-resident and scales fine, which the harness correctly shows.

## Acceptance Checks

Fast defaults must pass the FIQA/OpenAI quality gate before they are used for a
published claim. The gate is configured in
`config/acceptance_thresholds.json` and is intended for a full manual or
nightly FIQA run, not for per-PR perf smoke.

Run it against the generated full benchmark artifact:

```sh
OUTPUT=benchmarks/results/pgturbohybrid-fiqa.json
python3 benchmarks/tools/check_acceptance.py \
  "$OUTPUT" \
  --suite fiqa_openai_fast_defaults
```

The artifact must include `pgturbohybrid_recovered_explicit` latency-profile
results and the SQL RRF baseline with `nDCG@10`, `MRR@10`, `p95_ms`, and either
`Recall@10` or `overlap@10` versus SQL RRF. If the gate fails, evaluate
`quality` profile, `exact_storage = on`, or larger dense/BM25 budgets before
publishing the fast default result.

## Dataset Notes

`config/datasets.json` lists intended quality and systems datasets. FIQA, BEIR,
MS MARCO, MIRACL, LoTTE, and RAG sets should be run as reproducible
experiments with committed commands and external result artifacts.

For 3,072-dimensional DBPedia/OpenAI3-large runs, use
`benchmarks/dbpedia_openai3_large.py` rather than the FIQA harness. The native
PostgreSQL hybrid baseline uses pgvector `halfvec(3072)` HNSW plus PostgreSQL
full-text search because standard pgvector `vector` HNSW is not the intended
ANN path for this dimensionality.

For the DBPedia dense-only default comparison, use
`--methods pgvector_halfvec_dense_only,pgturbohybrid_dense_only,pgturbohybrid_dense_adaptive_auto_1_25,pgturbohybrid_dense_exact_storage_on`.
That run does not pass a text query to TurboHybrid and should not be reported as
hybrid retrieval. The adaptive row is opt-in diagnostic behavior, and the
exact-storage row is an upper-bound reference, not a compact default candidate.

For an optional external-library reference, use
`benchmarks/dbpedia_turbovec.py` against the same loaded DBPedia query set. It
builds a Turbovec `TurboQuantIndex(dim=3072, bit_width=4)` from the Qdrant
Parquet embeddings and reports dense-only latency and quality metrics. Keep
that row separate from PostgreSQL-native comparisons because Turbovec runs
in-process and does not exercise PostgreSQL storage, MVCC, indexing, or SQL
execution.

Synthetic vector generators are intentionally not part of the benchmark suite.
They are too far from real retrieval workloads for project performance claims.
The bring-your-own RAG benchmark is different: it is a local evaluation helper
for existing user data, not a source of public project benchmark claims.
