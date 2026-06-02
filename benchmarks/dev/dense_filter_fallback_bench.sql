-- Scaling smoke for dense full-candidate/fill fallback paths.
--
-- Builds one dense-only native graph over deterministic synthetic vectors.  The
-- index includes grp as a graph-owned payload column, but heap_bucket remains a
-- normal heap filter.  Compare:
--
--   payload_filter       WHERE grp = ...         (payload can be evaluated by the graph)
--   unmapped_heap_filter WHERE heap_bucket = ... (planner/runtime may widen dense work)
--
-- Expected trend: the unmapped heap-filter case should report
-- dense_filter_unmapped=true and, when the widened target is large enough,
-- dense_linear_fallback_warning=true with a high dense_linear_fallback_ratio.
-- The exact elapsed time is machine dependent; use the stats counters, not a
-- fixed latency threshold.
--
-- Usage:
--   psql -d <db> -f benchmarks/dev/dense_filter_fallback_bench.sql
--   psql -d <db> -v NROWS=20000 -v DIMS=16 -v DENSE_K=50 -v FINAL_K=10 \
--        -f benchmarks/dev/dense_filter_fallback_bench.sql

\set ON_ERROR_STOP on
\pset pager off
\x auto
\timing on

\if :{?NROWS}
\else
  \set NROWS 4000
\endif
\if :{?DIMS}
\else
  \set DIMS 8
\endif
\if :{?DENSE_K}
\else
  \set DENSE_K 20
\endif
\if :{?FINAL_K}
\else
  \set FINAL_K 8
\endif
\if :{?FILTER_KEEP_PCT}
\else
  \set FILTER_KEEP_PCT 90
\endif

\echo 'dense_filter_fallback_bench config:'
\echo '  NROWS           =' :NROWS
\echo '  DIMS            =' :DIMS
\echo '  DENSE_K         =' :DENSE_K
\echo '  FINAL_K         =' :FINAL_K
\echo '  FILTER_KEEP_PCT =' :FILTER_KEEP_PCT

CREATE EXTENSION IF NOT EXISTS pgturbohybrid CASCADE;

DROP TABLE IF EXISTS th_dense_filter_fallback_docs;
CREATE TABLE th_dense_filter_fallback_docs (
    id int PRIMARY KEY,
    grp int NOT NULL,
    heap_bucket int NOT NULL,
    embedding vector(:DIMS) NOT NULL
);

INSERT INTO th_dense_filter_fallback_docs
SELECT g,
       g % 16,
       CASE WHEN g <= (:NROWS * :FILTER_KEEP_PCT / 100) THEN 1 ELSE 0 END,
       (
           SELECT array_agg(
                      (sin(g * 0.013 + d * 0.17) +
                       cos(g * 0.031 + d * 0.07))::real
                      ORDER BY d)
           FROM generate_series(1, :DIMS) AS d
       )::real[]::vector
FROM generate_series(1, :NROWS) AS g;

CREATE INDEX th_dense_filter_fallback_idx ON th_dense_filter_fallback_docs
USING turbohybrid (embedding vector_cosine_turbohybrid_ops) INCLUDE (grp)
WITH (quantization_bits = 4, exact_storage = off);

ANALYZE th_dense_filter_fallback_docs;

SET enable_seqscan = off;
SET max_parallel_workers_per_gather = 0;
SET jit = off;
SET turbohybrid.warn_linear_fallback = off;
SET turbohybrid.linear_fallback_notice_threshold_ratio = 0.25;

CREATE TEMP TABLE th_dense_filter_fallback_config AS
SELECT :DENSE_K::int AS dense_k,
       :FINAL_K::int AS final_k;

CREATE TEMP TABLE th_dense_filter_fallback_results (
    scenario text NOT NULL,
    elapsed_ms float8 NOT NULL,
    rows_returned int NOT NULL,
    stats jsonb NOT NULL
);

DO $$
DECLARE
    qv vector;
    t0 timestamptz;
    n int;
    st jsonb;
    dense_k int;
    final_k int;
BEGIN
    SELECT c.dense_k, c.final_k INTO dense_k, final_k
    FROM th_dense_filter_fallback_config c;

    SELECT embedding INTO qv
    FROM th_dense_filter_fallback_docs
    WHERE id = 1;

    t0 := clock_timestamp();
    SELECT count(*) INTO n
    FROM (
        SELECT id
        FROM th_dense_filter_fallback_docs
        WHERE grp = 1
        ORDER BY embedding <~> turbohybrid_query(
            vector_query => qv,
            dense_k => dense_k,
            final_k => final_k
        )
        LIMIT final_k
    ) s;
    st := turbohybrid_last_scan_stats();
    INSERT INTO th_dense_filter_fallback_results
    VALUES (
        'payload_filter',
        extract(epoch FROM clock_timestamp() - t0) * 1000.0,
        n,
        jsonb_build_object(
            'scan_orchestration', st->'scan_orchestration',
            'graph_widening_reason', st->'graph_widening_reason',
            'graph_effective_result_target', st->'graph_effective_result_target',
            'graph_effective_search_ef', st->'graph_effective_search_ef',
            'graph_scored_codes', st->'graph_scored_codes',
            'graph_visited_nodes', st->'graph_visited_nodes',
            'graph_fill_candidate_band_calls', st->'graph_fill_candidate_band_calls',
            'graph_fill_candidate_band_reason', st->'graph_fill_candidate_band_reason',
            'graph_fill_candidate_band_visited', st->'graph_fill_candidate_band_visited',
            'graph_fill_candidate_band_scored', st->'graph_fill_candidate_band_scored',
            'graph_fill_candidate_band_target', st->'graph_fill_candidate_band_target',
            'graph_fill_candidate_band_used_payload_refs', st->'graph_fill_candidate_band_used_payload_refs',
            'graph_fill_candidate_band_payload_ref_count', st->'graph_fill_candidate_band_payload_ref_count',
            'dense_filter_unmapped', st->'dense_filter_unmapped',
            'dense_linear_fallback_warning', st->'dense_linear_fallback_warning',
            'dense_linear_fallback_ratio', st->'dense_linear_fallback_ratio',
            'dense_elapsed_us', st->'dense_elapsed_us'
        )
    );

    t0 := clock_timestamp();
    SELECT count(*) INTO n
    FROM (
        SELECT id
        FROM th_dense_filter_fallback_docs
        WHERE heap_bucket = 1
        ORDER BY embedding <~> turbohybrid_query(
            vector_query => qv,
            dense_k => dense_k,
            final_k => final_k
        )
        LIMIT final_k
    ) s;
    st := turbohybrid_last_scan_stats();
    INSERT INTO th_dense_filter_fallback_results
    VALUES (
        'unmapped_heap_filter',
        extract(epoch FROM clock_timestamp() - t0) * 1000.0,
        n,
        jsonb_build_object(
            'scan_orchestration', st->'scan_orchestration',
            'graph_widening_reason', st->'graph_widening_reason',
            'graph_effective_result_target', st->'graph_effective_result_target',
            'graph_effective_search_ef', st->'graph_effective_search_ef',
            'graph_scored_codes', st->'graph_scored_codes',
            'graph_visited_nodes', st->'graph_visited_nodes',
            'graph_fill_candidate_band_calls', st->'graph_fill_candidate_band_calls',
            'graph_fill_candidate_band_reason', st->'graph_fill_candidate_band_reason',
            'graph_fill_candidate_band_visited', st->'graph_fill_candidate_band_visited',
            'graph_fill_candidate_band_scored', st->'graph_fill_candidate_band_scored',
            'graph_fill_candidate_band_target', st->'graph_fill_candidate_band_target',
            'graph_fill_candidate_band_used_payload_refs', st->'graph_fill_candidate_band_used_payload_refs',
            'graph_fill_candidate_band_payload_ref_count', st->'graph_fill_candidate_band_payload_ref_count',
            'dense_filter_unmapped', st->'dense_filter_unmapped',
            'dense_linear_fallback_warning', st->'dense_linear_fallback_warning',
            'dense_linear_fallback_ratio', st->'dense_linear_fallback_ratio',
            'dense_elapsed_us', st->'dense_elapsed_us'
        )
    );
END
$$;

SELECT scenario,
       round(elapsed_ms::numeric, 3) AS elapsed_ms,
       rows_returned,
       jsonb_pretty(stats) AS scan_stats
FROM th_dense_filter_fallback_results
ORDER BY scenario;

EXPLAIN (ANALYZE, BUFFERS, SETTINGS)
SELECT id
FROM th_dense_filter_fallback_docs
WHERE heap_bucket = 1
ORDER BY embedding <~> turbohybrid_query(
    vector_query => (SELECT embedding FROM th_dense_filter_fallback_docs WHERE id = 1),
    dense_k => :DENSE_K,
    final_k => :FINAL_K
)
LIMIT :FINAL_K;

SELECT jsonb_pretty(turbohybrid_last_scan_stats()) AS last_unmapped_filter_stats;

RESET turbohybrid.warn_linear_fallback;
RESET turbohybrid.linear_fallback_notice_threshold_ratio;
RESET enable_seqscan;
RESET max_parallel_workers_per_gather;
RESET jit;
