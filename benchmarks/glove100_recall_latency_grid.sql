-- Dense recall/latency profile grid for glove-like vector workloads.
--
-- Run:
--   psql -d <db> -f benchmarks/glove100_recall_latency_grid.sql
--
-- Optional psql variables:
--   NROWS         synthetic/source row count           (default 10000)
--   DIMS          embedding dimensions                 (default 100)
--   QSET          query-vector count                   (default 64)
--   WARM          warmup queries per config            (default 20)
--   TIMED         timed queries per config             (default 100)
--   FINAL_K       recall/precision target              (default 10)
--   SOURCE_TABLE  existing table to copy embeddings from
--   SOURCE_ID     source id column                     (default id)
--   SOURCE_VECTOR source vector column                 (default embedding)
--
-- The grid compares latency-oriented defaults with balanced/quality profile
-- choices plus explicit exact, residual, and heap-rescore variants. Precision@K
-- is measured against exact pgvector ordering over the copied benchmark table.

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
\if :{?SOURCE_ID}
\else
  \set SOURCE_ID id
\endif
\if :{?SOURCE_VECTOR}
\else
  \set SOURCE_VECTOR embedding
\endif

CREATE EXTENSION IF NOT EXISTS pgturbohybrid CASCADE;

SET max_parallel_workers_per_gather = 0;
SET jit = off;

\echo '== pgturbohybrid glove-like recall/latency grid =='
\echo 'rows=' :NROWS ' dims=' :DIMS ' qset=' :QSET ' warm=' :WARM ' timed=' :TIMED ' final_k=' :FINAL_K

DROP TABLE IF EXISTS th_glove_grid_docs CASCADE;
DROP TABLE IF EXISTS th_glove_grid_queries CASCADE;
DROP TABLE IF EXISTS th_glove_grid_configs CASCADE;
DROP TABLE IF EXISTS th_glove_grid_results CASCADE;

CREATE TABLE th_glove_grid_docs (
    id int PRIMARY KEY,
    embedding vector(:DIMS) NOT NULL
);

\if :{?SOURCE_TABLE}
\echo '== loading embeddings from source table ' :SOURCE_TABLE ' =='
INSERT INTO th_glove_grid_docs(id, embedding)
SELECT :SOURCE_ID::int,
       :SOURCE_VECTOR::vector(:DIMS)
FROM :SOURCE_TABLE
WHERE :SOURCE_VECTOR IS NOT NULL
ORDER BY :SOURCE_ID
LIMIT :NROWS;
\else
\echo '== loading synthetic glove-like fallback embeddings =='
INSERT INTO th_glove_grid_docs(id, embedding)
SELECT g,
       (SELECT array_agg(
                   (sin(g * 0.007 + d * 0.019) +
                    0.04 * cos(g * 0.031 + d * 0.011))::real
                   ORDER BY d)
        FROM generate_series(1, :DIMS) AS d)::real[]::vector(:DIMS)
FROM generate_series(1, :NROWS) AS g;
\endif

ANALYZE th_glove_grid_docs;

CREATE TABLE th_glove_grid_queries (
    seq int PRIMARY KEY,
    qvec vector(:DIMS) NOT NULL,
    exact_ids int[] NOT NULL
);

\echo '== preparing exact baseline =='
WITH sampled AS (
    SELECT row_number() OVER (ORDER BY id) - 1 AS seq,
           embedding AS qvec
    FROM th_glove_grid_docs
    ORDER BY id
    LIMIT :QSET
)
INSERT INTO th_glove_grid_queries(seq, qvec, exact_ids)
SELECT s.seq,
       s.qvec,
       (
           SELECT array_agg(id ORDER BY dist, id)
           FROM (
               SELECT d.id,
                      d.embedding <=> s.qvec AS dist
               FROM th_glove_grid_docs d
               ORDER BY d.embedding <=> s.qvec, d.id
               LIMIT :FINAL_K
           ) exact
       ) AS exact_ids
FROM sampled s;

CREATE TABLE th_glove_grid_configs (
    label text PRIMARY KEY,
    profile text NOT NULL,
    index_options text NOT NULL,
    dense_rescore_band text NOT NULL DEFAULT 'auto',
    dense_heap_rescore text NOT NULL DEFAULT 'off'
);

INSERT INTO th_glove_grid_configs(label, profile, index_options, dense_rescore_band, dense_heap_rescore)
VALUES
    ('default', 'latency',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 128, graph_ef_search = 64, graph_oversampling = 4',
     'auto', 'off'),
    ('balanced', 'balanced',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 192, graph_ef_search = 96, graph_oversampling = 4',
     'auto', 'off'),
    ('quality', 'quality',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 256, graph_ef_search = 192, graph_oversampling = 8',
     'limited', 'off'),
    ('exact_storage', 'quality',
     'quantization_bits = 4, exact_storage = on, graph_ef_construction = 256, graph_ef_search = 192, graph_oversampling = 8',
     'limited', 'off'),
    ('residual_rerank', 'quality',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 256, graph_ef_search = 192, graph_oversampling = 8, residual_rerank = on, residual_rerank_bytes = 64',
     'limited', 'off'),
    ('heap_rescore_topk', 'quality',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 256, graph_ef_search = 192, graph_oversampling = 8',
     'limited', 'topk'),
    ('heap_rescore_band', 'quality',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 256, graph_ef_search = 192, graph_oversampling = 8',
     'limited', 'band');

CREATE TABLE th_glove_grid_results (
    label text PRIMARY KEY,
    profile text NOT NULL,
    index_options text NOT NULL,
    build_ms float8 NOT NULL,
    index_bytes bigint NOT NULL,
    precision_at_k_avg float8 NOT NULL,
    precision_at_k_min float8 NOT NULL,
    p50_ms float8 NOT NULL,
    p95_ms float8 NOT NULL,
    p99_ms float8 NOT NULL,
    index_stats jsonb NOT NULL,
    last_scan_stats jsonb NOT NULL
);

CREATE OR REPLACE FUNCTION pg_temp.th_glove_grid_overlap(a int[], b int[])
RETURNS int
LANGUAGE sql
AS $$
    SELECT count(*)::int
    FROM (
        SELECT unnest(a)
        INTERSECT
        SELECT unnest(b)
    ) s
$$;

CREATE OR REPLACE FUNCTION pg_temp.th_glove_grid_pctl(arr float8[], q float8)
RETURNS float8
LANGUAGE sql
AS $$
    SELECT percentile_cont(q) WITHIN GROUP (ORDER BY x)
    FROM unnest(arr) AS x
$$;

CREATE OR REPLACE FUNCTION pg_temp.th_glove_grid_run_config(p_label text,
	                                                            p_profile text,
	                                                            p_index_options text,
	                                                            p_rescore_band text,
	                                                            p_heap_rescore text,
	                                                            p_warm int,
                                                            p_timed int,
                                                            p_final_k int)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    build_started timestamptz;
    qrec record;
    nqueries int;
    i int;
    ids int[];
    elapsed_ms float8;
    durations float8[] := '{}';
    precision float8;
    precision_sum float8 := 0.0;
    precision_min float8 := NULL;
    build_ms float8;
    last_stats jsonb;
    index_stats jsonb;
BEGIN
    EXECUTE 'DROP INDEX IF EXISTS th_glove_grid_idx';
    PERFORM set_config('turbohybrid.profile', p_profile, false);
    PERFORM set_config('turbohybrid.dense_build_neighbor_select', 'auto', false);
    PERFORM set_config('turbohybrid.dense_rescore_band', p_rescore_band, false);
    PERFORM set_config('turbohybrid.dense_heap_rescore', p_heap_rescore, false);

    build_started := clock_timestamp();
    EXECUTE format(
        'CREATE INDEX th_glove_grid_idx ON th_glove_grid_docs USING turbohybrid ' ||
        '(embedding vector_cosine_turbohybrid_ops) WITH (%s)',
        p_index_options);
    build_ms := extract(epoch FROM clock_timestamp() - build_started) * 1000.0;
    ANALYZE th_glove_grid_docs;

    PERFORM set_config('enable_seqscan', 'off', false);

    SELECT count(*) INTO nqueries FROM th_glove_grid_queries;
    IF p_warm > 0 THEN
        FOR i IN 0..p_warm - 1 LOOP
            SELECT * INTO qrec
            FROM th_glove_grid_queries
            WHERE seq = i % nqueries;

            SELECT array_agg(id ORDER BY dist, id) INTO ids
            FROM (
                SELECT d.id,
                       d.embedding <~> turbohybrid_query(vector_query => qrec.qvec,
                                                         final_k => p_final_k) AS dist
                FROM th_glove_grid_docs d
                ORDER BY d.embedding <~> turbohybrid_query(vector_query => qrec.qvec,
                                                           final_k => p_final_k),
                         d.id
                LIMIT p_final_k
            ) ranked;
        END LOOP;
    END IF;

    IF p_timed > 0 THEN
        FOR i IN 0..p_timed - 1 LOOP
            SELECT * INTO qrec
            FROM th_glove_grid_queries
            WHERE seq = i % nqueries;

            build_started := clock_timestamp();
            SELECT array_agg(id ORDER BY dist, id) INTO ids
            FROM (
                SELECT d.id,
                       d.embedding <~> turbohybrid_query(vector_query => qrec.qvec,
                                                         final_k => p_final_k) AS dist
                FROM th_glove_grid_docs d
                ORDER BY d.embedding <~> turbohybrid_query(vector_query => qrec.qvec,
                                                           final_k => p_final_k),
                         d.id
                LIMIT p_final_k
            ) ranked;
            elapsed_ms := extract(epoch FROM clock_timestamp() - build_started) * 1000.0;
            durations := array_append(durations, elapsed_ms);

            precision := pg_temp.th_glove_grid_overlap(ids, qrec.exact_ids)::float8 /
                         GREATEST(p_final_k, 1);
            precision_sum := precision_sum + precision;
            precision_min := CASE
                WHEN precision_min IS NULL THEN precision
                ELSE LEAST(precision_min, precision)
            END;
        END LOOP;
    END IF;

    last_stats := turbohybrid_last_scan_stats();
    index_stats := turbohybrid_index_stats('th_glove_grid_idx'::regclass);

    INSERT INTO th_glove_grid_results(label, profile, index_options, build_ms,
                                      index_bytes, precision_at_k_avg,
                                      precision_at_k_min, p50_ms, p95_ms,
                                      p99_ms, index_stats, last_scan_stats)
    VALUES (
        p_label,
        p_profile,
        p_index_options,
        build_ms,
        pg_relation_size('th_glove_grid_idx'::regclass),
        precision_sum / GREATEST(p_timed, 1),
        COALESCE(precision_min, 0.0),
        COALESCE(pg_temp.th_glove_grid_pctl(durations, 0.50), 0.0),
        COALESCE(pg_temp.th_glove_grid_pctl(durations, 0.95), 0.0),
        COALESCE(pg_temp.th_glove_grid_pctl(durations, 0.99), 0.0),
        index_stats,
        last_stats
    )
    ON CONFLICT (label) DO UPDATE
    SET profile = EXCLUDED.profile,
        index_options = EXCLUDED.index_options,
        build_ms = EXCLUDED.build_ms,
        index_bytes = EXCLUDED.index_bytes,
        precision_at_k_avg = EXCLUDED.precision_at_k_avg,
        precision_at_k_min = EXCLUDED.precision_at_k_min,
        p50_ms = EXCLUDED.p50_ms,
        p95_ms = EXCLUDED.p95_ms,
        p99_ms = EXCLUDED.p99_ms,
        index_stats = EXCLUDED.index_stats,
        last_scan_stats = EXCLUDED.last_scan_stats;
END;
$$;

\echo '== running grid =='
SELECT pg_temp.th_glove_grid_run_config(label, profile, index_options,
                                        dense_rescore_band, dense_heap_rescore,
                                        :WARM, :TIMED,
                                        :FINAL_K)
FROM th_glove_grid_configs
ORDER BY CASE label
    WHEN 'default' THEN 1
    WHEN 'balanced' THEN 2
    WHEN 'quality' THEN 3
    WHEN 'exact_storage' THEN 4
    WHEN 'residual_rerank' THEN 5
    WHEN 'heap_rescore_topk' THEN 6
    WHEN 'heap_rescore_band' THEN 7
    ELSE 100
END;

\echo '== summary =='
SELECT label,
       profile,
       round(build_ms::numeric, 1) AS build_ms,
       pg_size_pretty(index_bytes) AS index_size,
       round(precision_at_k_avg::numeric, 4) AS precision_at_k_avg,
       round(precision_at_k_min::numeric, 4) AS precision_at_k_min,
       round(p50_ms::numeric, 3) AS p50_ms,
       round(p95_ms::numeric, 3) AS p95_ms,
       round(p99_ms::numeric, 3) AS p99_ms,
       index_stats->>'build_neighbor_select' AS build_neighbor_select,
       index_stats->>'build_fast_edges' AS build_fast_edges,
       index_stats->>'graph_ef_construction' AS graph_ef_construction,
       index_stats->>'graph_ef_search' AS graph_ef_search,
       index_stats->>'graph_oversampling' AS graph_oversampling,
	       index_stats->>'exact_storage' AS exact_storage,
	       index_stats->>'residual_rerank_bytes' AS residual_rerank_bytes,
	       last_scan_stats->>'dense_heap_rescore' AS dense_heap_rescore,
	       last_scan_stats->>'graph_scored_codes' AS graph_scored_codes,
	       last_scan_stats->>'graph_effective_search_ef' AS graph_effective_search_ef,
	       last_scan_stats->>'exact_rescore_count' AS exact_rescore_count,
	       last_scan_stats->>'heap_rescore_count' AS heap_rescore_count,
	       last_scan_stats->>'heap_fetch_us' AS heap_fetch_us,
	       last_scan_stats->>'heap_rescore_us' AS heap_rescore_us,
	       last_scan_stats->>'exact_rescore_source' AS exact_rescore_source
FROM th_glove_grid_results
ORDER BY CASE label
    WHEN 'default' THEN 1
    WHEN 'balanced' THEN 2
    WHEN 'quality' THEN 3
    WHEN 'exact_storage' THEN 4
    WHEN 'residual_rerank' THEN 5
    WHEN 'heap_rescore_topk' THEN 6
    WHEN 'heap_rescore_band' THEN 7
    ELSE 100
END;
