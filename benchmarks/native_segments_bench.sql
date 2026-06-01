-- Native graph segment-count benchmark.
--
-- Run:
--   psql -d <db> -f benchmarks/native_segments_bench.sql
--
-- Optional psql variables:
--   NROWS       synthetic/source row count       (default 10000)
--   DIMS        embedding dimensions             (default 100)
--   QSET        query-vector count               (default 64)
--   WARM        warmup queries per segment count (default 20)
--   TIMED       timed queries per segment count  (default 100)
--   FINAL_K     recall/precision target          (default 10)
--   DENSE_K     dense candidate budget           (default 100)
--   BITS        quantization bits                (default 4)
--   SEGMENTS    comma-separated segment counts   (default 1,2,4,8)
--   BUDGETS     comma-separated budget modes     (default off,sqrt,linear)
--
-- This measures the tradeoff of partitioning one native graph into independent
-- graph segments.  More segments reduce per-segment edge-construction work and
-- can improve build time, but query search must scale the search budget across
-- segments or recall can drop. Compare off (old behavior), sqrt, and linear
-- budget modes together before choosing a segment count.

\set ON_ERROR_STOP on
\pset pager off

\if :{?NROWS}
\else
  \set NROWS 10000
\endif
\if :{?DIMS}
\else
  \set DIMS 100
\endif
\if :{?QSET}
\else
  \set QSET 64
\endif
\if :{?WARM}
\else
  \set WARM 20
\endif
\if :{?TIMED}
\else
  \set TIMED 100
\endif
\if :{?FINAL_K}
\else
  \set FINAL_K 10
\endif
\if :{?DENSE_K}
\else
  \set DENSE_K 100
\endif
\if :{?BITS}
\else
  \set BITS 4
\endif
\if :{?SEGMENTS}
\else
  \set SEGMENTS 1,2,4,8
\endif
\if :{?BUDGETS}
\else
  \set BUDGETS off,sqrt,linear
\endif

CREATE EXTENSION IF NOT EXISTS pgturbohybrid CASCADE;

SET enable_seqscan = off;
SET max_parallel_workers_per_gather = 0;
SET jit = off;

\echo '== native segment benchmark =='
\echo 'rows=' :NROWS ' dims=' :DIMS ' qset=' :QSET ' warm=' :WARM ' timed=' :TIMED ' final_k=' :FINAL_K ' dense_k=' :DENSE_K ' segments=' :SEGMENTS ' budgets=' :BUDGETS

DROP TABLE IF EXISTS th_segments_docs CASCADE;
DROP TABLE IF EXISTS th_segments_queries CASCADE;
DROP TABLE IF EXISTS th_segments_config CASCADE;
DROP TABLE IF EXISTS th_segments_truth CASCADE;
DROP TABLE IF EXISTS th_segments_results CASCADE;

CREATE TABLE th_segments_docs (
    id int PRIMARY KEY,
    embedding vector(:DIMS) NOT NULL
);

INSERT INTO th_segments_docs(id, embedding)
SELECT g,
       (SELECT array_agg(
                   (sin(g * 0.011 + d * 0.017) +
                    0.05 * cos(g * 0.037 + d * 0.013))::real
                   ORDER BY d)
        FROM generate_series(1, :DIMS) AS d)::real[]::vector(:DIMS)
FROM generate_series(1, :NROWS) AS g;

CREATE TABLE th_segments_queries AS
SELECT row_number() OVER ()::int AS qid,
       embedding AS query_embedding
FROM th_segments_docs
ORDER BY id
LIMIT :QSET;

CREATE TEMP TABLE th_segments_config(segment_count int, budget_mode text);
INSERT INTO th_segments_config(segment_count, budget_mode)
SELECT s.segment_count, b.budget_mode
FROM (
    SELECT trim(value)::int AS segment_count
    FROM regexp_split_to_table(:'SEGMENTS', ',') AS value
    WHERE trim(value) <> ''
) s
CROSS JOIN (
    SELECT trim(value)::text AS budget_mode
    FROM regexp_split_to_table(:'BUDGETS', ',') AS value
    WHERE trim(value) <> ''
) b;

ALTER TABLE th_segments_config
    ADD PRIMARY KEY (segment_count, budget_mode);

DELETE FROM th_segments_config
WHERE budget_mode NOT IN ('auto', 'off', 'sqrt', 'linear');

CREATE TEMP TABLE th_segments_builds AS
SELECT DISTINCT segment_count
FROM th_segments_config;

CREATE TEMP TABLE th_segments_truth AS
SELECT q.qid, array_agg(d.id ORDER BY d.embedding <=> q.query_embedding, d.id) AS truth_ids
FROM th_segments_queries q
CROSS JOIN LATERAL (
    SELECT id, embedding
    FROM th_segments_docs
    ORDER BY embedding <=> q.query_embedding, id
    LIMIT :FINAL_K
) d
GROUP BY q.qid;

CREATE TEMP TABLE th_segments_results (
    segment_count int,
    budget_mode text,
    build_ms float8,
    index_size_bytes bigint,
    precision_at_k float8,
    p50_ms float8,
    p95_ms float8,
    avg_segment_count float8,
    avg_segments_searched float8,
    avg_entry_points float8,
    avg_effective_search_ef_before_scaling float8,
    avg_effective_search_ef_after_scaling float8,
    avg_scored_codes float8,
    avg_visited_nodes float8,
    avg_code_pages_read float8,
    avg_adj_pages_read float8,
    index_stats jsonb,
    last_scan_stats jsonb
);

CREATE OR REPLACE FUNCTION pg_temp.th_segments_run(p_segment_count int,
                                                   p_budget_mode text,
                                                   p_warm int,
                                                   p_timed int,
                                                   p_dense_k int,
                                                   p_final_k int,
                                                   p_bits int,
                                                   p_qset int)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    started timestamptz;
    build_ms float8;
    query_started timestamptz;
    q record;
    iter int;
    ids int[];
    latencies float8[] := ARRAY[]::float8[];
    precision_sum float8 := 0.0;
    stats jsonb := '{}'::jsonb;
    segment_count_sum float8 := 0.0;
    segments_searched_sum float8 := 0.0;
    entry_points_sum float8 := 0.0;
    search_ef_before_sum float8 := 0.0;
    search_ef_after_sum float8 := 0.0;
    scored_sum float8 := 0.0;
    visited_sum float8 := 0.0;
    code_pages_sum float8 := 0.0;
    adj_pages_sum float8 := 0.0;
BEGIN
    PERFORM set_config('turbohybrid.native_segment_budget', p_budget_mode, false);
    IF to_regclass('th_segments_idx') IS NULL THEN
        DROP INDEX IF EXISTS th_segments_idx;
        started := clock_timestamp();
        EXECUTE format(
            'CREATE INDEX th_segments_idx ON th_segments_docs ' ||
            'USING turbohybrid (embedding vector_cosine_turbohybrid_ops) ' ||
            'WITH (native_segments = %s, quantization_bits = %s)',
            p_segment_count,
            p_bits
        );
        build_ms := extract(epoch FROM clock_timestamp() - started) * 1000.0;
    ELSE
        SELECT r.build_ms INTO build_ms
        FROM th_segments_results r
        WHERE r.segment_count = p_segment_count
          AND r.build_ms IS NOT NULL
        ORDER BY r.budget_mode
        LIMIT 1;
    END IF;

    FOR iter IN 1..p_warm LOOP
        FOR q IN SELECT * FROM th_segments_queries ORDER BY qid LOOP
            PERFORM id
            FROM th_segments_docs
            ORDER BY embedding <~> turbohybrid_query(
                vector_query => q.query_embedding,
                dense_k => p_dense_k,
                final_k => p_final_k
            )
            LIMIT p_final_k;
        END LOOP;
    END LOOP;

    FOR iter IN 1..p_timed LOOP
        FOR q IN SELECT * FROM th_segments_queries ORDER BY qid LOOP
            query_started := clock_timestamp();
            SELECT array_agg(id ORDER BY ord) INTO ids
            FROM (
                SELECT id, row_number() OVER () AS ord
                FROM (
                    SELECT id
                    FROM th_segments_docs
                    ORDER BY embedding <~> turbohybrid_query(
                        vector_query => q.query_embedding,
                        dense_k => p_dense_k,
                        final_k => p_final_k
                    )
                    LIMIT p_final_k
                ) s
            ) ranked;
            latencies := latencies || (extract(epoch FROM clock_timestamp() - query_started) * 1000.0);
            stats := turbohybrid_last_scan_stats();
            precision_sum := precision_sum + (
                SELECT count(*)::float8 / p_final_k
                FROM unnest(ids) got(id)
                JOIN th_segments_truth truth ON truth.qid = q.qid
                WHERE got.id = ANY(truth.truth_ids)
            );
            segment_count_sum := segment_count_sum + coalesce((stats->>'graph_segment_count')::float8, 0);
            segments_searched_sum := segments_searched_sum + coalesce((stats->>'graph_segments_searched')::float8, 0);
            entry_points_sum := entry_points_sum + coalesce((stats->>'graph_entry_point_count')::float8, 0);
            search_ef_before_sum := search_ef_before_sum + coalesce((stats->>'effective_search_ef_before_segment_scaling')::float8, 0);
            search_ef_after_sum := search_ef_after_sum + coalesce((stats->>'effective_search_ef_after_segment_scaling')::float8, 0);
            scored_sum := scored_sum + coalesce((stats->>'graph_scored_codes')::float8, 0);
            visited_sum := visited_sum + coalesce((stats->>'graph_visited_nodes')::float8, 0);
            code_pages_sum := code_pages_sum + coalesce((stats->>'graph_code_pages_read')::float8, 0);
            adj_pages_sum := adj_pages_sum + coalesce((stats->>'graph_adj_pages_read')::float8, 0);
        END LOOP;
    END LOOP;

    INSERT INTO th_segments_results
    SELECT p_segment_count,
           p_budget_mode,
           build_ms,
           pg_relation_size('th_segments_idx'),
           precision_sum / (p_timed * p_qset),
           percentile_cont(0.50) WITHIN GROUP (ORDER BY latency_ms),
           percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms),
           segment_count_sum / (p_timed * p_qset),
           segments_searched_sum / (p_timed * p_qset),
           entry_points_sum / (p_timed * p_qset),
           search_ef_before_sum / (p_timed * p_qset),
           search_ef_after_sum / (p_timed * p_qset),
           scored_sum / (p_timed * p_qset),
           visited_sum / (p_timed * p_qset),
           code_pages_sum / (p_timed * p_qset),
           adj_pages_sum / (p_timed * p_qset),
           turbohybrid_index_stats('th_segments_idx'::regclass),
           stats
    FROM unnest(latencies) AS latency_ms;
END;
$$;

CREATE OR REPLACE FUNCTION pg_temp.th_segments_run_group(p_segment_count int,
                                                         p_warm int,
                                                         p_timed int,
                                                         p_dense_k int,
                                                         p_final_k int,
                                                         p_bits int,
                                                         p_qset int)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    b record;
BEGIN
    DROP INDEX IF EXISTS th_segments_idx;
    FOR b IN
        SELECT budget_mode
        FROM th_segments_config
        WHERE segment_count = p_segment_count
        ORDER BY CASE budget_mode
            WHEN 'off' THEN 1
            WHEN 'sqrt' THEN 2
            WHEN 'linear' THEN 3
            ELSE 4
        END
    LOOP
        PERFORM pg_temp.th_segments_run(p_segment_count, b.budget_mode, p_warm,
                                        p_timed, p_dense_k, p_final_k, p_bits,
                                        p_qset);
    END LOOP;
END;
$$;

SELECT pg_temp.th_segments_run_group(segment_count, :WARM, :TIMED, :DENSE_K,
                                     :FINAL_K, :BITS, :QSET)
FROM th_segments_builds
ORDER BY segment_count;

\echo '== summary =='
SELECT segment_count,
       budget_mode,
       round(build_ms::numeric, 1) AS build_ms,
       pg_size_pretty(index_size_bytes) AS index_size,
       round(precision_at_k::numeric, 4) AS precision_at_k,
       round(p50_ms::numeric, 3) AS p50_ms,
       round(p95_ms::numeric, 3) AS p95_ms,
       round(avg_segment_count::numeric, 2) AS avg_segment_count,
       round(avg_segments_searched::numeric, 2) AS avg_segments_searched,
       round(avg_entry_points::numeric, 2) AS avg_entry_points,
       round(avg_effective_search_ef_before_scaling::numeric, 1) AS avg_ef_before,
       round(avg_effective_search_ef_after_scaling::numeric, 1) AS avg_ef_after,
       round(avg_scored_codes::numeric, 1) AS avg_scored_codes,
       round(avg_visited_nodes::numeric, 1) AS avg_visited_nodes,
       round(avg_code_pages_read::numeric, 1) AS avg_code_pages_read,
       round(avg_adj_pages_read::numeric, 1) AS avg_adj_pages_read
FROM th_segments_results
ORDER BY segment_count, budget_mode;

\echo '== raw stats =='
SELECT segment_count, budget_mode, index_stats, last_scan_stats
FROM th_segments_results
ORDER BY segment_count, budget_mode;
