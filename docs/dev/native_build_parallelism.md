# Native Build Parallelism Baseline

This note defines the serial baseline to capture before changing native graph
build parallelism. The build path already emits `DEBUG1` phase logs, but those
logs are not easy to compare across runs. Use `turbohybrid_last_build_stats()`
for machine-readable timing and distance-path counters from the most recent
native graph build in the current backend.

## Serial Phase Structure

Native graph builds currently run these phases:

1. `fit_correction_scan_us`: optional heap scan for streaming correction fit
   when code-only cosine or inner-product build needs correction statistics.
2. `scan_us`: heap scan, vector detoast/copy, and append into build storage.
3. `fit_correction_us`: correction fitting for non-code-only builds after scan.
4. `encode_us`: quantized code encoding for non-code-only builds after fitting.
5. `build_edges_us`: graph edge construction.
6. `free_exact_vectors_us`: release temporary exact vectors after edge build.
7. `reorder_nodes_us`: node locality reorder before page write.
8. `connect_backbone_us`: optional level-0 backbone connection.
9. `entry_sidecar_us`: routing entries and entry-sidecar construction.
10. `write_pages_us`: index page layout and writes.
11. `wal_us`: WAL new-page logging.
12. `total_us`: wall-clock build time for the native graph build.

The JSON also records index parameters, build distance-path counters, index
shape, and `worker_count`. For a strict serial baseline, run with
`turbohybrid.native_build_workers = '0'`.

`turbohybrid.native_build_workers` accepts `auto`, `0`, `1`, `2`, `4`, or `8`.
Explicit numeric values request that many workers, capped by
`max_parallel_maintenance_workers`. `auto` uses PostgreSQL's parallel
`CREATE INDEX` worker choice.

## Reading Build Stats

After building a native index in the same session:

```sql
SELECT jsonb_pretty(turbohybrid_last_build_stats());
```

The result is backend-local and only describes the most recent native graph
build in that backend. It is diagnostic state; it does not change build output.

Important fields:

- `relation_name`, `relid`, `index_shape`
- `node_count`, `dimensions`, `quantization_bits`, `m`, `ef_construction`
- `exact_storage`, `build_code_only`, `build_fast_edges`
- `build_distance_*` counters for build-time scoring paths
- phase timings in microseconds
- `worker_count`
- `native_build_workers_requested`, `native_build_workers_launched`
- `parallel_fit_enabled`, `parallel_scan_enabled`,
  `parallel_encode_enabled`
- `worker_merge_us`, `worker_scan_us`

Parallel build currently applies only to native code-only builds, where workers
can scan heap ranges, collect TIDs and payload values, build residual sketches,
and encode quantized codes before the serial edge build. Builds that keep exact
vectors (`exact_storage = on`) or use quantile/P-square correction fitting fall
back to the serial pre-edge path. That fallback is intentional because merging
quantile estimators is not exact enough for this baseline patch.

Parallel workers produce records keyed by heap TID. The leader sorts records by
heap TID before assigning node IDs, so merged node order is deterministic and
independent of worker completion order. `PgturbohybridGraphBuildEdges()` still
runs serially after the merge.

## Benchmark

Run the baseline benchmark:

```sh
psql "$DATABASE_URL" -f benchmarks/native_build_phase_bench.sql
```

Override dimensions or row count with psql variables:

```sh
psql "$DATABASE_URL" \
  -v ROWS=100000 \
  -v DIMS=100 \
  -v EF_CONSTRUCTION=192 \
  -f benchmarks/native_build_phase_bench.sql
```

The script builds a dense-only native TurboHybrid index with
`native_build_workers` set to `0`, `2`, `4`, and `8`. For each variant it
prints `turbohybrid_last_build_stats()`, index size, launched-worker counts,
parallel phase flags, merge timing, per-participant scan timing, and a phase
table in microseconds and milliseconds.

## Parallelization Priority

Use the phase table to choose the first target. If `scan_us`,
`fit_correction_scan_us`, or `encode_us` are material, parallelize heap scan,
correction statistics, and encoding first because they are independent per row
and easier to merge. If `build_edges_us` dominates after those phases are
measured, segmented or parallel edge construction should be evaluated as the
next build-time bottleneck.
