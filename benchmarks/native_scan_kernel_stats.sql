-- Native-scan scoring-kernel diagnostics for pgturbohybrid.
--
-- Proves which SIMD kernel the native scan actually runs inside
-- PgturbohybridGraphScoreNodeBatch() / PgturbohybridGraphScoreNode() for a
-- 1536-dim 4-bit index, and surfaces the batch sizing and the configuration
-- that selects the kernel.  Use this to answer "is the hot dense path the u8
-- split, the signed split, or the scalar/LUT fallback?" and, when it is the
-- scalar/LUT path unexpectedly, to see why (score_mode / bits / u8_split_enabled
-- / query_split_enabled / dense_simd_force).
--
--   psql -d <db> -f benchmarks/native_scan_kernel_stats.sql
--
-- Reads turbohybrid_last_scan_stats(); the keys of interest are:
--   graph_score_kernels  -- {kernel: {nodes, calls}}, nodes sum == graph_scored_codes
--   graph_batch_calls / graph_batch_nodes / graph_avg_batch_size
--   score_mode / dimensions / quantization_bits / u8_split_enabled /
--   query_split_enabled / dense_simd_force / dense_scorer
--   exact_rescore_count / graph_code_pages_read / graph_adj_pages_read

\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS pgturbohybrid CASCADE;

DROP TABLE IF EXISTS th_kernel_probe;
CREATE TABLE th_kernel_probe (id int, embedding vector(1536), body_tsv tsvector);

-- Synthetic 1536-dim corpus large enough that graph traversal does real
-- batch-of-4 scoring (not just a handful of single-node tail scores).
INSERT INTO th_kernel_probe
SELECT g,
       (SELECT array_agg((sin(g * 0.001 + d * 0.017) + 0.05 * sin(g * 0.7 + d))::real
                         ORDER BY d)
        FROM generate_series(1, 1536) AS d)::real[]::vector,
       to_tsvector('english', 'document number ' || g)
FROM generate_series(1, 5000) AS g;

CREATE INDEX th_kernel_probe_idx ON th_kernel_probe
  USING turbohybrid (embedding vector_cosine_turbohybrid_ops,
                     body_tsv bm25_tsvector_turbohybrid_ops)
  WITH (quantization_bits = 4);

ANALYZE th_kernel_probe;

SET enable_seqscan = off;
SET max_parallel_workers_per_gather = 0;

-- Run a representative nearest-neighbor scan so last-scan stats reflect it.
SELECT id
FROM th_kernel_probe
ORDER BY embedding <~> turbohybrid_query(
           vector_query => (SELECT embedding FROM th_kernel_probe WHERE id = 1234))
LIMIT 10;

\echo '== per-kernel scoring attribution (nodes sum == graph_scored_codes) =='
SELECT jsonb_pretty(turbohybrid_last_scan_stats() -> 'graph_score_kernels')
       AS graph_score_kernels;

\echo '== batch sizing + kernel-selection config =='
SELECT key, value
FROM jsonb_each_text(turbohybrid_last_scan_stats())
WHERE key IN (
        'scan_orchestration', 'graph_storage_kind', 'score_mode', 'dimensions',
        'quantization_bits', 'query_split_enabled', 'u8_split_enabled',
        'dense_simd_force', 'dense_scorer', 'dense_scoring_kernel', 'dense_batch_kernel',
        'graph_scored_codes', 'graph_simd_scored_codes', 'graph_scalar_scored_codes',
        'graph_batch_calls', 'graph_batch_nodes', 'graph_avg_batch_size',
        'exact_rescore_count', 'graph_code_pages_read', 'graph_adj_pages_read')
ORDER BY key;

\echo '== EXPLAIN (ANALYZE, BUFFERS, SETTINGS) of the stats accessor =='
EXPLAIN (ANALYZE, BUFFERS, SETTINGS) SELECT turbohybrid_last_scan_stats();

-- Clean up (comment out to keep the probe table for further inspection).
DROP TABLE th_kernel_probe;
