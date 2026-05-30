-- Native base-layer traversal profile + warm latency.
--
-- Reports the graph_base_layer counters (frontier push/pop, nearest offers,
-- visited checks vs duplicate skips, per-pop batch calls/nodes, peak frontier)
-- and the internal traversal-CPU timers (graph_traverse_us / graph_total_us,
-- averaged over many queries -- drift-robust, unlike sub-ms wall-clock) plus
-- warm wall-clock p50/p95/p99, on a 1536-dim 4-bit native index.
--
--   psql -d <db> -f benchmarks/base_layer_traversal.sql
--
-- Notes:
--   * graph_heap_us is ~0: base-layer heap ops are sub-microsecond; the former
--     per-op INSTR_TIME timing was removed (it measured nothing but paid the
--     clock_gettime cost in the hot loop -- ~20us/query off graph_traverse_us).
--   * duplicate_skips is typically ~90% of visited_checks: dense HNSW neighbors
--     re-encounter visited nodes; the O(1) visitedGeneration check handles them.

\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS pgturbohybrid CASCADE;

DROP TABLE IF EXISTS blt_docs;
CREATE TABLE blt_docs (id int, embedding vector(1536), body_tsv tsvector);
INSERT INTO blt_docs
SELECT g,
       (SELECT array_agg((sin(g * 0.001 + d * 0.017) + 0.05 * sin(g * 0.7 + d))::real ORDER BY d)
        FROM generate_series(1, 1536) AS d)::real[]::vector,
       to_tsvector('english', 'document number ' || g)
FROM generate_series(1, 10000) AS g;
CREATE INDEX blt_idx ON blt_docs
  USING turbohybrid (embedding vector_cosine_turbohybrid_ops,
                     body_tsv bm25_tsvector_turbohybrid_ops)
  WITH (quantization_bits = 4);
ANALYZE blt_docs;

SET enable_seqscan = off;
SET max_parallel_workers_per_gather = 0;
SET jit = off;

SELECT id FROM blt_docs
  ORDER BY embedding <~> turbohybrid_query(vector_query => (SELECT embedding FROM blt_docs WHERE id = 1234))
  LIMIT 10;

\echo '== base-layer traversal counters (one query) =='
SELECT jsonb_pretty(turbohybrid_last_scan_stats() -> 'graph_base_layer') AS graph_base_layer;

CREATE OR REPLACE FUNCTION blt_pctl(arr float8[], q float8) RETURNS float8
LANGUAGE sql AS $f$ SELECT percentile_cont(q) WITHIN GROUP (ORDER BY x) FROM unnest(arr) x $f$;

DO $$
DECLARE
    qv vector; t0 timestamptz; k int;
    d float8[] := '{}'; trav bigint := 0; tot bigint := 0; n int := 0;
BEGIN
    FOR k IN 1..60 LOOP
        PERFORM id FROM blt_docs ORDER BY embedding <~> turbohybrid_query(
          vector_query => (SELECT embedding FROM blt_docs WHERE id = (k * 131) % 10000 + 1)) LIMIT 10;
    END LOOP;
    FOR k IN 1..600 LOOP
        qv := (SELECT embedding FROM blt_docs WHERE id = (k * 131) % 10000 + 1);
        t0 := clock_timestamp();
        PERFORM id FROM blt_docs ORDER BY embedding <~> turbohybrid_query(vector_query => qv) LIMIT 10;
        d := array_append(d, extract(epoch FROM clock_timestamp() - t0) * 1000);
        trav := trav + (turbohybrid_last_scan_stats() ->> 'graph_traverse_us')::bigint;
        tot := tot + (turbohybrid_last_scan_stats() ->> 'graph_total_us')::bigint;
        n := n + 1;
    END LOOP;
    RAISE NOTICE 'internal (avg us): traverse=%  total=%', trav / n, tot / n;
    RAISE NOTICE 'wall (ms): p50=%  p95=%  p99=%',
        round(blt_pctl(d,0.5)::numeric,4), round(blt_pctl(d,0.95)::numeric,4),
        round(blt_pctl(d,0.99)::numeric,4);
END $$;

DROP FUNCTION blt_pctl(float8[], float8);
DROP TABLE blt_docs;
RESET enable_seqscan;
RESET max_parallel_workers_per_gather;
RESET jit;
