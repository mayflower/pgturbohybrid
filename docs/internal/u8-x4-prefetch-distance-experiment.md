# Experiment: deeper prefetch distance in the U8 x4 scoring path

Status: **completed — negative result, not shipped.**
Date: 2026-05-31. Host: 32-core x86-64 (AVX-512 VNNI, Ice Lake-class), 61 GB RAM,
PostgreSQL 18.4.

## Hypothesis

Whole-code prefetch is already gated by `graphLargeCodeArena` (code working set
> 64 MB) and the batch scorer already prefetches an 8-code lookahead window. The
x4 u8 kernel overlaps four scattered code streams (memory-level parallelism). The
open question: on large (1M+) scattered-code arenas that are RAM-latency-bound,
does prefetching codes *further* ahead than the existing 8-code window hide more
latency and improve dense-scan p50/p95?

## What was tried

A developer GUC `turbohybrid.dev.u8_x4_prefetch_distance` (int, default `0` =
off). When `> 0`, and only when `graphLargeCodeArena` is true and the u8 split
kernel is active, `PgturbohybridGraphScoreNodeBatch()` prefetches that many
additional *whole* codes beyond the existing 8-code window. The default-off path
is byte-identical to the shipped code, and the block is gated so cache-resident
and non-u8 scans can never reach it (they are also memory-cheap, so deeper
prefetch is pure overhead there).

`distance` is in codes, so 8 ≈ "one more batch ahead", 16 ≈ "two more batches",
32/64 = deeper. This is a strictly *more aggressive* superset of the
"next batch / next two batches" variants in the task brief.

The experimental patch (reverted from `src/` after measurement; reapply to
reproduce):

```c
/* src/pgturbohybrid_quant_score.c, inside PgturbohybridGraphScoreNodeBatch(),
   immediately after the existing 8-code prefetch window loop: */
if (pgturbohybrid_dev_u8_x4_prefetch_distance > 0 &&
    so->graphLargeCodeArena && so->tq.u8.enabled)
{
    int deepEnd = (int) Min((int64) i + 8 + pgturbohybrid_dev_u8_x4_prefetch_distance,
                            (int64) nodeCount);
    for (int j = i + 8; j < deepEnd; j++)
    {
        PgturbohybridGraphScanNode *node = &storage->nodes[nodeIds[j]];
        if (node->code != NULL)
            for (Size off = 0; off < so->tq.codeBytes; off += 64)
                PGTURBOHYBRID_GRAPH_PREFETCH_READ(node->code + off);
    }
}
```
plus the GUC (`pgturbohybrid_dev_u8_x4_prefetch_distance`, default 0) in
`pgturbohybrid_graph.c` / `pgturbohybrid.h` / `pgturbohybrid_am.c`.

A "chunk-level prefetch inside the x4 kernel" variant was *not* separately
implemented: for a large arena the driver already issues whole-code prefetch for
the lookahead window, so in-kernel chunk prefetch of those same codes is
redundant; and the driver-level deeper lookahead measured here is a strictly more
aggressive prefetch, so its flat result already bounds the chunk-level upside.

## Method

Two synthetic beds (1536-dim, 4-bit, code-only; native graph scans; warm
per-backend code arena):

- **10k cache-resident control** — code arena ≈ 7.7 MB (`graphLargeCodeArena =
  false`); the deeper-prefetch block is gated off here by construction.
- **200k memory-bound** — code arena ≈ 153 MB (`graphLargeCodeArena = true`,
  several× L3; scattered code loads miss cache → RAM-bound). Index 219 MB.
- **600k memory-bound** — code arena ≈ 460 MB. Index ~660 MB.

Metric: internal scan timers from `turbohybrid_last_scan_stats()` —
`graph_batch_us` (the SIMD scoring phase the prefetch targets) and
`graph_total_us` (dense-scan latency). Internal timers exclude planning /
plpgsql, isolating the scan. 150 queries per setting, distinct query vectors, 30
warm-up queries first; three interleaved rounds (`0 → 8 → 16 → 32 → 64`) to
absorb thermal/temporal drift. Harness: `pf_bench()` (below).

## Results

### 200k bed (153 MB arena, RAM-bound, ~391 codes scored/query)

`graph_batch_us` (µs), p50 / p95 by round:

| distance | r1 p50 | r2 p50 | r3 p50 | r1 p95 | r2 p95 | r3 p95 |
|---------:|-------:|-------:|-------:|-------:|-------:|-------:|
| 0 (off)  | 73     | 82     | 82.5   | 106.6  | 110.2  | 110.0  |
| 8        | 72     | 79.5   | 78.5   | 111.7  | 122.6  | 122.1  |
| 16       | 71     | 82     | 85     | 101.0  | 108.6  | 125.6  |
| 32       | 63.5   | 83     | 85.5   | 100.6  | 126.6  | 113.6  |
| 64       | 81.5   | 82     | 82.5   | 112.0  | 111.6  | 113.0  |

`graph_total_us` (µs) p50 / p95 tracked the same pattern (p50 ≈ 256–273, p95 ≈
285–349 across *all* distances including off).

Observations:
- Run-to-run variance for a *fixed* distance (±~13% on p50) is larger than any
  difference *between* distances. Round 1 was uniformly faster than rounds 2–3
  across every distance (drift), not a distance effect.
- No distance gives a consistent p50 or p95 improvement; the smallest deeper
  distance (8) was marginally *worse* on p95 in 2/3 rounds.
- `graph_batch_us` (~80 µs) is only ~30% of `graph_total_us` (~260 µs), so even
  a large batch-scoring win could not move total latency much; traversal / heap
  / sort dominate.

### 600k bed (460 MB arena, RAM-bound, ~407 codes scored/query)

Methodological note: the 600k working set (codes + nodes + adjacency ≈ 656 MB)
exceeds the default `native_cache_max_mb = 512`, so by default the index falls
back to per-scan page loading — `graph_total_us` ≈ **308 ms**/query, dominated by
page loading, not scoring (a different bottleneck; deeper prefetch was also no
help there). To test the intended regime — a large, RAM-resident, *contiguous*
scattered arena — the run below sets `native_cache_max_mb = 2048` so the 460 MB
code arena is fully resident.

`graph_batch_us` (µs) p50 / p95 by round (arena-resident, cache = 2 GB):

| distance | r1 p50 | r2 p50 | r3 p50 | r1 p95 | r2 p95 | r3 p95 |
|---------:|-------:|-------:|-------:|-------:|-------:|-------:|
| 0 (off)  | 68.5   | 58     | 63.5   | 91.1   | 85.1   | 86.0   |
| 16       | 63.5   | 64.5   | 55.5   | 93.1   | 92.0   | 90.0   |
| 32       | 69     | 74     | 72.5   | 95.1   | 99.1   | 98.1   |
| 64       | 72.5   | 72     | 55     | 96.1   | 94.0   | 86.1   |

`graph_total_us` p50 ≈ 231–257 / p95 ≈ 274–314 across all distances including
off. Same conclusion as 200k: no p50/p95 improvement; the larger distance (32)
trends slightly *worse* on both batch and total p95 (prefetch-instruction
overhead and cache pressure); the smallest (16) is break-even within noise.
`scored_med` ≈ 407 — only marginally above the 200k bed's 391, because HNSW
scored-count grows ~logarithmically; a true 1.2M index scores a similar number
per query, so the per-query memory regime here is representative of 1M+.

### 10k control (7.7 MB arena, cache-resident)

`large_code_arena = false`; distance 16 vs 0 was a no-op (total p50 247 vs 242,
within noise) — confirms the gate: deeper prefetch never fires on small/
cache-resident indexes, so small-index latency is unaffected.

## Conclusion / decision

Deeper x4 prefetch (8 / 16 / 32 / 64 codes beyond the existing window) does **not**
clearly improve dense-scan p50 or p95 on any tested regime — 153 MB and 460 MB
RAM-resident arenas, or the 656 MB page-load regime. In the arena-resident cases
the between-distance differences sit inside the run-to-run noise band, and the
larger distances trend slightly *worse* on p95.

Why this is expected, not a measurement artifact:
- The batch scorer already prefetches an 8-code whole-code window, the x4 kernel
  already overlaps four scattered code streams (MLP), and the hardware
  prefetchers + OoO execution cover the rest. For the ~400 codes scored per
  query there is little un-hidden memory latency left for *more* software
  prefetch to recover.
- `graph_batch_us` (the only phase prefetch touches) is just ~25–30% of
  `graph_total_us`; traversal / heap / sort dominate, capping any possible
  total-latency win.
- Extra prefetch instructions and cache pressure are pure overhead once the
  latency is already hidden, which is why larger distances nudge p95 up.

**Decision: do not ship.** The experimental GUC and prefetch block were
**reverted** from `src/`, so the default hot path is byte-identical to before
the experiment (no default regression, by construction; small/cache-resident
indexes were never reachable anyway via the `graphLargeCodeArena` gate). This
file preserves the approach, the patch, and the data so the negative result is
recorded and the experiment can be re-run (e.g. on real 1M+ embedding data or
different hardware) without rediscovering it.

If revisited, the more promising next lever is *layout* (reordering codes so a
batch's four scattered loads land on fewer pages / closer addresses), not deeper
prefetch.

## Reproduce

1. Reapply the experimental patch above (GUC + prefetch block); rebuild/install.
2. Build the beds (1536-dim, 4-bit). For a bed whose working set (≈ rows ×
   codeBytes, plus nodes/adjacency) approaches `native_cache_max_mb`, raise that
   GUC above the working set first — otherwise the index falls back to per-scan
   page loading and you measure page-load latency, not the arena scoring path.
3. Run the interleaved sweep over `{0, 8, 16, 32, 64}` and read the internal
   timers. For large beds prefer the static-table harness variant (`PERFORM id
   FROM <tbl> ...` with the table name fixed, so plpgsql caches the plan) over
   the `EXECUTE format(...)` form below, which re-plans each query and dominates the
   wall-clock on a large index (it does not affect the internal timers).

`pf_bench()` harness:

```sql
CREATE OR REPLACE FUNCTION pf_bench(tbl text, dist int, nq int, warm int DEFAULT 30)
RETURNS TABLE(distance int, batch_p50 numeric, batch_p95 numeric,
              total_p50 numeric, total_p95 numeric, scored_med numeric,
              large_arena bool, u8mode text)
LANGUAGE plpgsql AS $$
DECLARE
  qv vector; i int; s jsonb;
  bus bigint[] := '{}'; tus bigint[] := '{}'; scd bigint[] := '{}';
  la bool; um text;
BEGIN
  PERFORM set_config('turbohybrid.dev.u8_x4_prefetch_distance', dist::text, false);
  PERFORM set_config('jit','off',false);
  PERFORM set_config('enable_seqscan','off',false);
  FOR i IN 1..warm LOOP
    qv := (SELECT array_agg((sin(i*0.13+g*0.017)+0.2*cos(i*0.001+g*0.03))::real)
           FROM generate_series(1,1536) g)::vector;
    EXECUTE format('SELECT id FROM %s ORDER BY embedding <~> turbohybrid_query(vector_query => $1) LIMIT 10', tbl) USING qv;
  END LOOP;
  FOR i IN 1..nq LOOP
    qv := (SELECT array_agg((sin((i+9000)*0.131+g*0.017)+0.2*cos(i*0.0011+g*0.029))::real)
           FROM generate_series(1,1536) g)::vector;
    EXECUTE format('SELECT id FROM %s ORDER BY embedding <~> turbohybrid_query(vector_query => $1) LIMIT 10', tbl) USING qv;
    s := turbohybrid_last_scan_stats();
    bus := array_append(bus, (s->>'graph_batch_us')::bigint);
    tus := array_append(tus, (s->>'graph_total_us')::bigint);
    scd := array_append(scd, (s->>'graph_scored_codes')::bigint);
  END LOOP;
  la := (s->>'graph_large_code_arena')::bool; um := s->>'graph_u8_batch_mode';
  RETURN QUERY
  WITH b AS (SELECT unnest(bus) v), t AS (SELECT unnest(tus) v), c AS (SELECT unnest(scd) v)
  SELECT dist,
    (SELECT percentile_cont(0.5)  WITHIN GROUP (ORDER BY v) FROM b)::numeric,
    (SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY v) FROM b)::numeric,
    (SELECT percentile_cont(0.5)  WITHIN GROUP (ORDER BY v) FROM t)::numeric,
    (SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY v) FROM t)::numeric,
    (SELECT percentile_cont(0.5)  WITHIN GROUP (ORDER BY v) FROM c)::numeric,
    la, um;
END$$;
```
