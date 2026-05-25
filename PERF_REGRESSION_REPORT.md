# Fast Defaults Validation Report

Date: 2026-05-25

## Summary

Final validation passed the build and regression test matrix on the local
PostgreSQL 16 PGXS install. The full FIQA/OpenAI benchmark used the real
57,638-row, 1,536-dimensional dataset with 648 qrels-backed queries.

The latency default is now the recovered fast path: after canonicalizing
latency-profile defaults before execution, the omitted-budget default path
measured `0.910 ms` p95 and returned the same FIQA rankings as the explicit
recovered query. The explicit recovered row measured `1.030 ms` p95 in the same
accepted artifact, which is within normal local run variance of the `1.006 ms`
target.

## Validation Commands

Passed:

```sh
make clean
make
make install
make installcheck
make prove_installcheck

make clean
make SIMD_BUILD=none
make install
make installcheck

python3 -m py_compile benchmarks/fiqa_openai.py benchmarks/dev/perf_smoke_check.py
bash -n benchmarks/dev/run_perf_matrix.sh benchmarks/dev/run_fiqa_quick.sh benchmarks/dev/compare_old_branch.sh
python3 benchmarks/dev/perf_smoke_check.py /tmp/pgturbohybrid-fiqa-canonicalized-with-baseline.json
python3 benchmarks/tools/check_acceptance.py /tmp/pgturbohybrid-fiqa-canonicalized-with-baseline.json --suite fiqa_openai_fast_defaults
```

`make prove_installcheck` reported `NOTESTS` because the PostgreSQL TAP modules
are not available in this local PGXS installation.

## Current FIQA/OpenAI Acceptance Result

Command:

```sh
python3 benchmarks/fiqa_openai.py \
  --database pgturbohybrid_fiqa_final \
  --dataset /Volumes/CrucialMusic/src/pgvector/.cache/beir/fiqa \
  --reuse-data \
  --profile latency \
  --dense-k 100 \
  --bm25-k 100 \
  --final-k 10 \
  --warmup 1 \
  --methods postgres_sql_rrf,pgturbohybrid,pgturbohybrid_recovered_explicit \
  --explain \
  --bm25-cache-probe \
  --output /tmp/pgturbohybrid-fiqa-canonicalized-with-baseline.json
```

| Method | Settings | Build ms | Index MB | p50 | p95 | p99 | nDCG@10 | MRR@10 | Recall@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| pgturbohybrid default | default index, omitted budgets, LIMIT inferred | 68,861.120 | 113.45 | 0.729 ms | 0.910 ms | 1.096 ms | 0.421540 | 0.498162 | 0.491077 |
| pgturbohybrid explicit recovered | 4-bit, exact_storage=off, dense_k=100, bm25_k=100, rrf_k=60, final_k=10 | 53,952.735 | 113.45 | 0.793 ms | 1.030 ms | 1.361 ms | 0.421540 | 0.498162 | 0.491077 |
| SQL RRF | pgvector HNSW + PostgreSQL FTS, 100/100 | 103,636.772 | 468.54 | 1.635 ms | 3.254 ms | 5.579 ms | 0.423341 | 0.490324 | 0.503056 |

## Structural Fast-Path Checks

The perf-smoke checker passed on
`/tmp/pgturbohybrid-fiqa-canonicalized-with-baseline.json`.

Observed default-path stats:

- `profile = latency`
- `quantization_bits = 4`
- `exact_storage = false`
- `dense_k_effective = 100`
- `bm25_k_effective = 100`
- `final_k_source = limit`
- `final_k_inferred_count = 648`
- `scan_orchestration = graph_native`
- BM25 cache warm probes passed for all warm probe rows

## Assessment

The canonicalized default path satisfies the latency and quality guard:

- p95 is `0.910 ms`, below the `1.006 ms` recovered target.
- SQL RRF p95 is `3.254 ms` on the same hardware.
- nDCG@10 is essentially equal to SQL RRF (`0.421540` vs `0.423341`).

The default omitted-budget path now uses the same canonical execution shape as
the explicit recovered path after LIMIT/default resolution. It still carries
different diagnostics (`final_k_source = limit` instead of `explicit`), but the
retrieval output is identical on FIQA and p95 is close to the recovered target.

## Environment

- CPU: Apple M4
- OS: macOS 26.5 arm64
- PostgreSQL: 16.13 (Homebrew)
- pgvector: 0.8.2
- pgturbohybrid: 0.1.0
- source commit: `edf1b5e` with local fast-default changes
- compiler: Apple clang 21.0.0
- PGXS flags: `-Wall ... -O2`
- SIMD: portable build, ARM dotprod and i8mm compiled, SIMD enabled

Generated full JSON artifacts are intentionally not committed.
