-- ns/code microbenchmark of the dense 4-bit scoring kernels (Prompt 2).
--
-- turbohybrid_scorer_bench(query, doc, bits, ncodes, iters) times a tight loop
-- that scores `ncodes` cache/RAM-resident 4-bit codes under each kernel and
-- reports the minimum ns/code over `iters` passes, isolating raw per-code
-- compute from the scan machinery.  Codes are visited in a scattered order
-- (large coprime stride) so every kernel sees the access pattern of a real
-- native scan -- four scattered graph neighbours per batch -- rather than a
-- contiguous sweep that would hand the single-node path a fake sequential
-- prefetch advantage.
--
-- Kernels reported:
--   scalar_lut    -- per-dim float LUT gather (the slow reference)
--   signed_split  -- signed-codebook query split (the fallback kernel)
--   u8_single     -- unsigned-codebook split, one code per kernel call
--   u8_x4         -- unsigned-codebook split, true 4-candidate batch (the
--                    default production batch; shares the query load across the
--                    four codes and overlaps their scattered memory latency)
--
-- Two regimes are measured at 1536 dims:
--   10k codes   (~7.5 MB)  -- cache-resident, compute-bound
--   1.2M codes  (~920 MB)  -- spills L3, memory-bound (mirrors the 1M index)
-- and two SIMD tiers (AVX-512 VNNI and AVX2) via the dense_graph_avx* knobs.
--
-- Expectation on amd64 (Ice Lake VNNI): u8 beats scalar/LUT by ~20-30x and
-- beats signed split; u8_x4 is ~tied with u8_single compute-bound and markedly
-- faster (≈1.4x) memory-bound, because sharing the query load and issuing the
-- four scattered code loads together exposes memory-level parallelism that the
-- serialized single-node path cannot.  Run:
--   psql -d <db> -f benchmarks/u8_x4_kernel_microbench.sql

\set ON_ERROR_STOP on
\pset pager off

CREATE EXTENSION IF NOT EXISTS pgturbohybrid CASCADE;
SET jit = off;

-- Deterministic 1536-dim query/doc spanning many codebook bins.
DROP TABLE IF EXISTS u8x4_qd;
CREATE TEMP TABLE u8x4_qd AS
SELECT (SELECT array_agg((sin(g * 0.013) + 0.3 * cos(g * 0.05))::real ORDER BY g)
        FROM generate_series(1, 1536) g)::real[]::vector AS q,
       (SELECT array_agg((cos(g * 0.017) * sin(g * 0.003) + 0.5 * sin(g * 0.001))::real ORDER BY g)
        FROM generate_series(1, 1536) g)::real[]::vector AS d;

\echo '== AVX-512 VNNI tier =='
SET turbohybrid.dense_graph_avx512vnni = on;
SET turbohybrid.dense_graph_avxvnni = on;
SELECT '10k  compute-bound' AS regime,
       jsonb_pretty(turbohybrid_scorer_bench(q, d, 4, 10000, 40)) AS bench FROM u8x4_qd;
SELECT '1.2M memory-bound' AS regime,
       jsonb_pretty(turbohybrid_scorer_bench(q, d, 4, 1200000, 6)) AS bench FROM u8x4_qd;

\echo '== AVX2 tier =='
SET turbohybrid.dense_graph_avx512vnni = off;
SET turbohybrid.dense_graph_avxvnni = off;
SELECT '10k  compute-bound' AS regime,
       jsonb_pretty(turbohybrid_scorer_bench(q, d, 4, 10000, 40)) AS bench FROM u8x4_qd;
SELECT '1.2M memory-bound' AS regime,
       jsonb_pretty(turbohybrid_scorer_bench(q, d, 4, 1200000, 6)) AS bench FROM u8x4_qd;

RESET turbohybrid.dense_graph_avx512vnni;
RESET turbohybrid.dense_graph_avxvnni;
DROP TABLE u8x4_qd;
RESET jit;
