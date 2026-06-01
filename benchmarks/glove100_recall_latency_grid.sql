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
-- The grid compares latency-oriented defaults with balanced/matched_recall/
-- quality profile choices plus isolated single-lever variants. Treat this
-- benchmark as the
-- authority for deciding whether any quality lever is worth promoting into a
-- default: it records the precision gain, p95 cost, build cost, index size, and
-- the scan/build stats needed to decide whether quality loss comes from graph
-- topology, search budget, or final ranking.  Precision is measured against
-- exact pgvector ordering over the copied benchmark table.  Keep FINAL_K=10
-- when comparing against the glove100 external benchmark numbers.

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
    dense_build_neighbor_select text NOT NULL DEFAULT 'auto',
    dense_build_distance text NOT NULL DEFAULT 'auto',
    dense_rescore_band text NOT NULL DEFAULT 'auto',
    dense_heap_rescore text NOT NULL DEFAULT 'auto',
    dense_adaptive_widening text NOT NULL DEFAULT 'auto'
);

INSERT INTO th_glove_grid_configs(label, profile, index_options,
                                  dense_build_neighbor_select,
                                  dense_build_distance,
                                  dense_rescore_band, dense_heap_rescore,
                                  dense_adaptive_widening)
VALUES
    ('default_latency', 'latency',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 128, graph_ef_search = 64, graph_oversampling = 4, native_segments = 1',
     'auto', 'auto', 'auto', 'auto', 'auto'),
    ('adaptive_off', 'latency',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 128, graph_ef_search = 64, graph_oversampling = 4, native_segments = 1',
     'auto', 'auto', 'auto', 'auto', 'off'),
    ('code_build_fast_edges', 'latency',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 128, graph_ef_search = 64, graph_oversampling = 4, native_segments = 1',
     'fast', 'code', 'auto', 'off', 'off'),
    ('code_build_heuristic_edges', 'latency',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 128, graph_ef_search = 64, graph_oversampling = 4, native_segments = 1',
     'heuristic', 'code', 'auto', 'off', 'off'),
    ('exact_build_heuristic_edges', 'latency',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 128, graph_ef_search = 64, graph_oversampling = 4, native_segments = 1',
     'heuristic', 'exact', 'auto', 'off', 'off'),
    ('heuristic_edges_only', 'latency',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 128, graph_ef_search = 64, graph_oversampling = 4, native_segments = 1',
     'heuristic', 'auto', 'auto', 'off', 'off'),
    ('ef96_only', 'latency',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 128, graph_ef_search = 96, graph_oversampling = 4, native_segments = 1',
     'auto', 'auto', 'auto', 'off', 'off'),
    ('ef128_only', 'latency',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 128, graph_ef_search = 128, graph_oversampling = 4, native_segments = 1',
     'auto', 'auto', 'auto', 'off', 'off'),
    ('oversampling8_only', 'latency',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 128, graph_ef_search = 64, graph_oversampling = 8, native_segments = 1',
     'auto', 'auto', 'auto', 'off', 'off'),
    ('heuristic_plus_ef96', 'latency',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 128, graph_ef_search = 96, graph_oversampling = 4, native_segments = 1',
     'heuristic', 'auto', 'auto', 'off', 'off'),
    ('heuristic_plus_ef128', 'latency',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 128, graph_ef_search = 128, graph_oversampling = 4, native_segments = 1',
     'heuristic', 'auto', 'auto', 'off', 'off'),
    ('heap_rescore_topk_only', 'latency',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 128, graph_ef_search = 64, graph_oversampling = 4, native_segments = 1',
     'auto', 'auto', 'auto', 'topk', 'off'),
    ('heap_rescore_band_only', 'latency',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 128, graph_ef_search = 64, graph_oversampling = 4, native_segments = 1',
     'auto', 'auto', 'auto', 'band', 'off'),
    ('exact_build_distances_only', 'latency',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 128, graph_ef_search = 64, graph_oversampling = 4, native_segments = 1',
     'auto', 'exact', 'auto', 'off', 'off'),
    ('exact_build_plus_heuristic', 'latency',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 128, graph_ef_search = 64, graph_oversampling = 4, native_segments = 1',
     'heuristic', 'exact', 'auto', 'off', 'off'),
    ('balanced', 'balanced',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 192, graph_ef_search = 96, graph_oversampling = 4, native_segments = 1',
     'auto', 'auto', 'auto', 'auto', 'auto'),
    ('adaptive_balanced', 'balanced',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 192, graph_ef_search = 96, graph_oversampling = 4, native_segments = 1',
     'auto', 'auto', 'auto', 'auto', 'auto'),
    ('balanced_no_heap_rescore', 'balanced',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 192, graph_ef_search = 96, graph_oversampling = 4, native_segments = 1',
     'auto', 'auto', 'auto', 'off', 'auto'),
    ('balanced_topk_heap_rescore', 'balanced',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 192, graph_ef_search = 96, graph_oversampling = 4, native_segments = 1',
     'auto', 'auto', 'auto', 'topk', 'auto'),
    ('matched_recall', 'matched_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = auto',
     'auto', 'auto', 'auto', 'auto', 'auto'),
    ('quality', 'quality',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 256, graph_ef_search = 192, graph_oversampling = 8, native_segments = 1',
     'auto', 'auto', 'limited', 'auto', 'auto'),
    ('adaptive_quality', 'quality',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 256, graph_ef_search = 192, graph_oversampling = 8, native_segments = 1',
     'auto', 'auto', 'limited', 'auto', 'auto'),
    ('quality_topk_heap_rescore', 'quality',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 256, graph_ef_search = 192, graph_oversampling = 8, native_segments = 1',
     'auto', 'auto', 'limited', 'topk', 'auto'),
    ('quality_band_heap_rescore', 'quality',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 256, graph_ef_search = 192, graph_oversampling = 8, native_segments = 1',
     'auto', 'auto', 'limited', 'band', 'auto'),
    ('exact_storage', 'quality',
     'quantization_bits = 4, exact_storage = on, graph_ef_construction = 256, graph_ef_search = 192, graph_oversampling = 8, native_segments = 1',
     'auto', 'auto', 'limited', 'auto', 'auto'),
    ('residual_rerank', 'quality',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 256, graph_ef_search = 192, graph_oversampling = 8, native_segments = 1, residual_rerank = on, residual_rerank_bytes = 64',
     'auto', 'auto', 'limited', 'off', 'auto'),
    ('heap_rescore_topk', 'quality',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 256, graph_ef_search = 192, graph_oversampling = 8, native_segments = 1',
     'auto', 'auto', 'limited', 'topk', 'auto'),
    ('heap_rescore_band', 'quality',
     'quantization_bits = 4, exact_storage = off, graph_ef_construction = 256, graph_ef_search = 192, graph_oversampling = 8, native_segments = 1',
     'auto', 'auto', 'limited', 'band', 'auto');

CREATE TABLE th_glove_grid_results (
    label text PRIMARY KEY,
    profile text NOT NULL,
    index_options text NOT NULL,
    build_ms float8 NOT NULL,
    index_bytes bigint NOT NULL,
    precision_at_10 float8 NOT NULL,
    precision_at_10_min float8 NOT NULL,
    p50_ms float8 NOT NULL,
    p95_ms float8 NOT NULL,
    p99_ms float8 NOT NULL,
    graph_m int,
    graph_ef_construction int,
    graph_ef_search int,
    graph_oversampling int,
    quantization_bits int,
    exact_storage boolean,
    dense_build_exact_distances boolean,
    dense_build_distance_mode text,
    build_neighbor_select text,
    build_fast_edges boolean,
    native_segments int,
    residual_rerank_bytes int,
    heap_rescore_count bigint,
    heap_rescore_us bigint,
    heap_rescore_auto_enabled boolean,
    heap_rescore_reason text,
    exact_rescore_source text,
    graph_scored_codes bigint,
    graph_effective_search_ef bigint,
    graph_effective_result_target bigint,
    adaptive_widening_triggered boolean,
    adaptive_widening_reason text,
    adaptive_initial_result_target bigint,
    adaptive_final_result_target bigint,
    adaptive_initial_search_ef bigint,
    adaptive_final_search_ef bigint,
    adaptive_gap_top10 float8,
    adaptive_gap_boundary float8,
    graph_batch_us bigint,
    graph_base_us bigint,
    graph_traverse_us bigint,
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
                                                            p_neighbor_select text,
                                                            p_build_distance text,
	                                                            p_rescore_band text,
	                                                            p_heap_rescore text,
	                                                            p_adaptive_widening text,
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
    PERFORM set_config('turbohybrid.dense_build_neighbor_select', p_neighbor_select, false);
    EXECUTE 'RESET turbohybrid.dense_build_exact_distances';
    PERFORM set_config('turbohybrid.dense_build_distance', p_build_distance, false);
    PERFORM set_config('turbohybrid.dense_rescore_band', p_rescore_band, false);
    PERFORM set_config('turbohybrid.dense_heap_rescore', p_heap_rescore, false);
    PERFORM set_config('turbohybrid.dense_adaptive_widening', p_adaptive_widening, false);

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
                                                           final_k => p_final_k)
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
                                                           final_k => p_final_k)
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
                                      index_bytes, precision_at_10,
                                      precision_at_10_min, p50_ms, p95_ms,
                                      p99_ms, graph_m, graph_ef_construction,
                                      graph_ef_search, graph_oversampling,
                                      quantization_bits, exact_storage,
                                      dense_build_exact_distances,
                                      dense_build_distance_mode,
                                      build_neighbor_select, build_fast_edges,
                                      native_segments, residual_rerank_bytes,
                                      heap_rescore_count, heap_rescore_us,
                                      heap_rescore_auto_enabled,
                                      heap_rescore_reason,
                                      exact_rescore_source, graph_scored_codes,
                                      graph_effective_search_ef,
                                      graph_effective_result_target,
                                      adaptive_widening_triggered,
                                      adaptive_widening_reason,
                                      adaptive_initial_result_target,
                                      adaptive_final_result_target,
                                      adaptive_initial_search_ef,
                                      adaptive_final_search_ef,
                                      adaptive_gap_top10,
                                      adaptive_gap_boundary,
                                      graph_batch_us, graph_base_us,
                                      graph_traverse_us, index_stats,
                                      last_scan_stats)
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
        (index_stats->>'graph_m')::int,
        (index_stats->>'graph_ef_construction')::int,
        (index_stats->>'graph_ef_search')::int,
        (index_stats->>'graph_oversampling')::int,
        (index_stats->>'quantization_bits')::int,
        (index_stats->>'exact_storage')::boolean,
        (index_stats->>'dense_build_exact_distances')::boolean,
        index_stats->>'dense_build_distance_mode',
        index_stats->>'build_neighbor_select',
        (index_stats->>'build_fast_edges')::boolean,
        (index_stats->>'native_segments')::int,
        (index_stats->>'residual_rerank_bytes')::int,
        COALESCE((last_stats->>'heap_rescore_count')::bigint, 0),
        COALESCE((last_stats->>'heap_rescore_us')::bigint, 0),
        COALESCE((last_stats->>'heap_rescore_auto_enabled')::boolean, false),
        last_stats->>'heap_rescore_reason',
        last_stats->>'exact_rescore_source',
        COALESCE((last_stats->>'graph_scored_codes')::bigint, 0),
        COALESCE((last_stats->>'graph_effective_search_ef')::bigint, 0),
        COALESCE((last_stats->>'graph_effective_result_target')::bigint, 0),
        COALESCE((last_stats->>'adaptive_widening_triggered')::boolean, false),
        last_stats->>'adaptive_widening_reason',
        COALESCE((last_stats->>'adaptive_initial_result_target')::bigint, 0),
        COALESCE((last_stats->>'adaptive_final_result_target')::bigint, 0),
        COALESCE((last_stats->>'adaptive_initial_search_ef')::bigint, 0),
        COALESCE((last_stats->>'adaptive_final_search_ef')::bigint, 0),
        COALESCE((last_stats->>'adaptive_gap_top10')::float8, 0.0),
        COALESCE((last_stats->>'adaptive_gap_boundary')::float8, 0.0),
        COALESCE((last_stats->>'graph_batch_us')::bigint, 0),
        COALESCE((last_stats->'dense'->'timing_us'->>'base')::bigint,
                 (last_stats->>'graph_base_us')::bigint,
                 0),
        COALESCE((last_stats->>'graph_traverse_us')::bigint, 0),
        index_stats,
        last_stats
    )
    ON CONFLICT (label) DO UPDATE
    SET profile = EXCLUDED.profile,
        index_options = EXCLUDED.index_options,
        build_ms = EXCLUDED.build_ms,
        index_bytes = EXCLUDED.index_bytes,
        precision_at_10 = EXCLUDED.precision_at_10,
        precision_at_10_min = EXCLUDED.precision_at_10_min,
        p50_ms = EXCLUDED.p50_ms,
        p95_ms = EXCLUDED.p95_ms,
        p99_ms = EXCLUDED.p99_ms,
        graph_m = EXCLUDED.graph_m,
        graph_ef_construction = EXCLUDED.graph_ef_construction,
        graph_ef_search = EXCLUDED.graph_ef_search,
        graph_oversampling = EXCLUDED.graph_oversampling,
        quantization_bits = EXCLUDED.quantization_bits,
        exact_storage = EXCLUDED.exact_storage,
        dense_build_exact_distances = EXCLUDED.dense_build_exact_distances,
        dense_build_distance_mode = EXCLUDED.dense_build_distance_mode,
        build_neighbor_select = EXCLUDED.build_neighbor_select,
        build_fast_edges = EXCLUDED.build_fast_edges,
        native_segments = EXCLUDED.native_segments,
        residual_rerank_bytes = EXCLUDED.residual_rerank_bytes,
        heap_rescore_count = EXCLUDED.heap_rescore_count,
        heap_rescore_us = EXCLUDED.heap_rescore_us,
        heap_rescore_auto_enabled = EXCLUDED.heap_rescore_auto_enabled,
        heap_rescore_reason = EXCLUDED.heap_rescore_reason,
        exact_rescore_source = EXCLUDED.exact_rescore_source,
        graph_scored_codes = EXCLUDED.graph_scored_codes,
        graph_effective_search_ef = EXCLUDED.graph_effective_search_ef,
        graph_effective_result_target = EXCLUDED.graph_effective_result_target,
        adaptive_widening_triggered = EXCLUDED.adaptive_widening_triggered,
        adaptive_widening_reason = EXCLUDED.adaptive_widening_reason,
        adaptive_initial_result_target = EXCLUDED.adaptive_initial_result_target,
        adaptive_final_result_target = EXCLUDED.adaptive_final_result_target,
        adaptive_initial_search_ef = EXCLUDED.adaptive_initial_search_ef,
        adaptive_final_search_ef = EXCLUDED.adaptive_final_search_ef,
        adaptive_gap_top10 = EXCLUDED.adaptive_gap_top10,
        adaptive_gap_boundary = EXCLUDED.adaptive_gap_boundary,
        graph_batch_us = EXCLUDED.graph_batch_us,
        graph_base_us = EXCLUDED.graph_base_us,
        graph_traverse_us = EXCLUDED.graph_traverse_us,
        index_stats = EXCLUDED.index_stats,
        last_scan_stats = EXCLUDED.last_scan_stats;
END;
$$;

\echo '== running grid =='
SELECT pg_temp.th_glove_grid_run_config(label, profile, index_options,
                                        dense_build_neighbor_select,
                                        dense_build_distance,
                                        dense_rescore_band, dense_heap_rescore,
                                        dense_adaptive_widening,
                                        :WARM, :TIMED,
                                        :FINAL_K)
FROM th_glove_grid_configs
ORDER BY CASE label
    WHEN 'default_latency' THEN 1
    WHEN 'adaptive_off' THEN 2
    WHEN 'code_build_fast_edges' THEN 3
    WHEN 'code_build_heuristic_edges' THEN 4
    WHEN 'exact_build_heuristic_edges' THEN 5
    WHEN 'heuristic_edges_only' THEN 6
    WHEN 'ef96_only' THEN 7
    WHEN 'ef128_only' THEN 8
    WHEN 'oversampling8_only' THEN 9
    WHEN 'heuristic_plus_ef96' THEN 10
    WHEN 'heuristic_plus_ef128' THEN 11
    WHEN 'heap_rescore_topk_only' THEN 12
    WHEN 'heap_rescore_band_only' THEN 13
    WHEN 'exact_build_distances_only' THEN 14
    WHEN 'exact_build_plus_heuristic' THEN 15
    WHEN 'balanced' THEN 20
    WHEN 'adaptive_balanced' THEN 21
    WHEN 'balanced_no_heap_rescore' THEN 22
    WHEN 'balanced_topk_heap_rescore' THEN 23
    WHEN 'matched_recall' THEN 25
    WHEN 'quality' THEN 30
    WHEN 'adaptive_quality' THEN 31
    WHEN 'quality_topk_heap_rescore' THEN 32
    WHEN 'quality_band_heap_rescore' THEN 33
    WHEN 'exact_storage' THEN 40
    WHEN 'residual_rerank' THEN 41
    WHEN 'heap_rescore_topk' THEN 42
    WHEN 'heap_rescore_band' THEN 43
    ELSE 100
END;

\echo '== full result rows =='
SELECT label,
       profile,
       round(build_ms::numeric, 1) AS build_ms,
       pg_size_pretty(index_bytes) AS index_size,
       round(precision_at_10::numeric, 4) AS precision_at_10,
       round(precision_at_10_min::numeric, 4) AS precision_at_10_min,
       round(p50_ms::numeric, 3) AS p50_ms,
       round(p95_ms::numeric, 3) AS p95_ms,
       round(p99_ms::numeric, 3) AS p99_ms,
       graph_m,
       graph_ef_construction,
       graph_ef_search,
       graph_oversampling,
       quantization_bits,
       exact_storage,
       dense_build_exact_distances,
       dense_build_distance_mode,
       build_neighbor_select,
       build_fast_edges,
       native_segments,
       residual_rerank_bytes,
       last_scan_stats->>'dense_heap_rescore' AS dense_heap_rescore,
       heap_rescore_count,
       heap_rescore_us,
       heap_rescore_auto_enabled,
       heap_rescore_reason,
       exact_rescore_source,
       graph_scored_codes,
       graph_effective_search_ef,
       graph_effective_result_target,
       adaptive_widening_triggered,
       adaptive_widening_reason,
       adaptive_initial_result_target,
       adaptive_final_result_target,
       adaptive_initial_search_ef,
       adaptive_final_search_ef,
       round(adaptive_gap_top10::numeric, 6) AS adaptive_gap_top10,
       round(adaptive_gap_boundary::numeric, 6) AS adaptive_gap_boundary,
       graph_batch_us,
       graph_base_us,
       graph_traverse_us
FROM th_glove_grid_results
ORDER BY CASE label
    WHEN 'default_latency' THEN 1
    WHEN 'adaptive_off' THEN 2
    WHEN 'code_build_fast_edges' THEN 3
    WHEN 'code_build_heuristic_edges' THEN 4
    WHEN 'exact_build_heuristic_edges' THEN 5
    WHEN 'heuristic_edges_only' THEN 6
    WHEN 'ef96_only' THEN 7
    WHEN 'ef128_only' THEN 8
    WHEN 'oversampling8_only' THEN 9
    WHEN 'heuristic_plus_ef96' THEN 10
    WHEN 'heuristic_plus_ef128' THEN 11
    WHEN 'heap_rescore_topk_only' THEN 12
    WHEN 'heap_rescore_band_only' THEN 13
    WHEN 'exact_build_distances_only' THEN 14
    WHEN 'exact_build_plus_heuristic' THEN 15
    WHEN 'balanced' THEN 20
    WHEN 'adaptive_balanced' THEN 21
    WHEN 'balanced_no_heap_rescore' THEN 22
    WHEN 'balanced_topk_heap_rescore' THEN 23
    WHEN 'matched_recall' THEN 25
    WHEN 'quality' THEN 30
    WHEN 'adaptive_quality' THEN 31
    WHEN 'quality_topk_heap_rescore' THEN 32
    WHEN 'quality_band_heap_rescore' THEN 33
    WHEN 'exact_storage' THEN 40
    WHEN 'residual_rerank' THEN 41
    WHEN 'heap_rescore_topk' THEN 42
    WHEN 'heap_rescore_band' THEN 43
    ELSE 100
END;

\echo '== isolated lever deltas vs default_latency =='
WITH base AS (
    SELECT *
    FROM th_glove_grid_results
    WHERE label = 'default_latency'
),
single_levers AS (
    SELECT r.*
    FROM th_glove_grid_results r
    WHERE r.label IN (
        'heuristic_edges_only',
        'code_build_fast_edges',
        'code_build_heuristic_edges',
        'exact_build_heuristic_edges',
        'ef96_only',
        'ef128_only',
        'oversampling8_only',
        'heap_rescore_topk_only',
        'heap_rescore_band_only',
        'exact_build_distances_only',
        'adaptive_off',
        'adaptive_balanced',
        'adaptive_quality',
        'balanced_no_heap_rescore',
        'balanced_topk_heap_rescore',
        'quality_topk_heap_rescore',
        'quality_band_heap_rescore'
    )
)
SELECT s.label,
       round(s.precision_at_10::numeric, 4) AS precision_at_10,
       round((s.precision_at_10 - b.precision_at_10)::numeric, 4) AS precision_gain,
       round(s.p95_ms::numeric, 3) AS p95_ms,
       round((s.p95_ms / NULLIF(b.p95_ms, 0))::numeric, 3) AS p95_x_default,
       round((s.build_ms / NULLIF(b.build_ms, 0))::numeric, 3) AS build_x_default,
       pg_size_pretty(s.index_bytes) AS index_size,
       s.build_neighbor_select,
       s.graph_ef_search,
       s.graph_oversampling,
       s.dense_build_exact_distances,
       s.dense_build_distance_mode,
       s.heap_rescore_count,
       s.heap_rescore_auto_enabled,
       s.heap_rescore_reason,
       s.exact_rescore_source,
       s.adaptive_widening_triggered,
       s.adaptive_widening_reason,
       s.adaptive_initial_result_target,
       s.adaptive_final_result_target,
       s.adaptive_initial_search_ef,
       s.adaptive_final_search_ef,
       s.graph_scored_codes
FROM single_levers s
CROSS JOIN base b
ORDER BY (s.precision_at_10 - b.precision_at_10) / NULLIF(s.p95_ms / NULLIF(b.p95_ms, 0), 0) DESC NULLS LAST,
         s.p95_ms,
         s.label;

\echo '== decision summary =='
WITH base AS (
    SELECT p95_ms AS default_p95_ms
    FROM th_glove_grid_results
    WHERE label = 'default_latency'
),
targets(kind, threshold, p95_multiplier) AS (
    VALUES
        ('fastest config with precision >= 0.70', 0.70::float8, NULL::float8),
        ('fastest config with precision >= 0.74', 0.74::float8, NULL::float8),
        ('fastest config with precision >= 0.80', 0.80::float8, NULL::float8),
        ('best precision under 1.25x default p95', NULL::float8, 1.25::float8),
        ('best precision under 1.50x default p95', NULL::float8, 1.50::float8)
),
ranked AS (
    SELECT t.kind,
           r.*,
           b.default_p95_ms,
           row_number() OVER (
               PARTITION BY t.kind
               ORDER BY
                   CASE WHEN t.threshold IS NOT NULL THEN r.p95_ms END ASC NULLS LAST,
                   CASE WHEN t.threshold IS NULL THEN r.precision_at_10 END DESC NULLS LAST,
                   r.p95_ms ASC,
                   r.label
           ) AS rn
    FROM targets t
    CROSS JOIN base b
    JOIN th_glove_grid_results r
      ON (t.threshold IS NOT NULL AND r.precision_at_10 >= t.threshold)
      OR (t.p95_multiplier IS NOT NULL AND r.p95_ms <= b.default_p95_ms * t.p95_multiplier)
)
SELECT t.kind,
       COALESCE(r.label, '(no qualifying config)') AS label,
       round(r.precision_at_10::numeric, 4) AS precision_at_10,
       round(r.p95_ms::numeric, 3) AS p95_ms,
       round((r.p95_ms / NULLIF(r.default_p95_ms, 0))::numeric, 3) AS p95_x_default,
       round(r.build_ms::numeric, 1) AS build_ms,
       pg_size_pretty(r.index_bytes) AS index_size,
       r.build_neighbor_select,
       r.graph_ef_search,
       r.graph_oversampling,
       r.dense_build_exact_distances,
       r.dense_build_distance_mode,
       r.residual_rerank_bytes,
       r.heap_rescore_count,
       r.heap_rescore_auto_enabled,
       r.heap_rescore_reason,
       r.exact_rescore_source,
       r.adaptive_widening_triggered,
       r.adaptive_widening_reason,
       r.adaptive_initial_result_target,
       r.adaptive_final_result_target,
       r.graph_scored_codes
FROM targets t
LEFT JOIN ranked r
  ON r.kind = t.kind
 AND r.rn = 1
ORDER BY CASE t.kind
    WHEN 'fastest config with precision >= 0.70' THEN 1
    WHEN 'fastest config with precision >= 0.74' THEN 2
    WHEN 'fastest config with precision >= 0.80' THEN 3
    WHEN 'best precision under 1.25x default p95' THEN 4
    WHEN 'best precision under 1.50x default p95' THEN 5
    ELSE 100
END;
