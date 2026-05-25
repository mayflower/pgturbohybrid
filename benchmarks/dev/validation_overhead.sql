\set ON_ERROR_STOP on

\if :{?dense_small}
\else
    \set dense_small 50
\endif
\if :{?dense_large}
\else
    \set dense_large 400
\endif
\if :{?final_k}
\else
    \set final_k 10
\endif

DO $$
BEGIN
    IF to_regclass('fiqa_docs') IS NULL THEN
        RAISE EXCEPTION 'validation_overhead.sql requires a real fiqa_docs table loaded from FIQA/OpenAI files';
    END IF;
    IF to_regprocedure('turbohybrid_debug_last_scan_stats()') IS NULL THEN
        RAISE EXCEPTION 'validation_overhead.sql requires developer diagnostics; run make dev-diagnostics-install and load sql/pgturbohybrid_dev_diagnostics.sql';
    END IF;
END $$;

CREATE TEMP TABLE validation_overhead_results (
    scenario text PRIMARY KEY,
    dense_k int,
    exact_storage bool,
    strict_delta bigint NOT NULL,
    fast_delta bigint NOT NULL,
    cache_miss_delta bigint NOT NULL,
    cache_hit_delta bigint NOT NULL
);

CREATE TEMP TABLE validation_overhead_settings AS
SELECT
    :dense_small::int AS dense_small,
    :dense_large::int AS dense_large,
    :final_k::int AS final_k;

CREATE OR REPLACE FUNCTION pg_temp.validation_counter_delta(
    before_stats jsonb,
    after_stats jsonb,
    key text
) RETURNS bigint
LANGUAGE sql
AS $$
    SELECT COALESCE((after_stats->>key)::bigint, 0) -
           COALESCE((before_stats->>key)::bigint, 0)
$$;

DO $$
DECLARE
    before_stats jsonb;
    after_stats jsonb;
    query_vector text;
    direct_distance double precision;
    row_count int;
    small_k int;
    large_k int;
    result_k int;
BEGIN
    SELECT dense_small, dense_large, final_k
    INTO small_k, large_k, result_k
    FROM validation_overhead_settings;

    SELECT embedding::text INTO query_vector
    FROM fiqa_docs
    ORDER BY doc_id
    LIMIT 1;

    IF query_vector IS NULL THEN
        RAISE EXCEPTION 'validation_overhead.sql requires non-empty real fiqa_docs data';
    END IF;

    PERFORM set_config('enable_seqscan', 'off', true);

    SELECT turbohybrid_debug_last_scan_stats() INTO before_stats;
    SELECT sum(turbohybrid_vector_l2_distance(d.embedding, q.embedding))
    INTO direct_distance
    FROM (
        SELECT embedding
        FROM fiqa_docs
        ORDER BY doc_id
        LIMIT 100
    ) d
    CROSS JOIN LATERAL (
        SELECT embedding
        FROM fiqa_docs
        ORDER BY doc_id DESC
        LIMIT 1
    ) q;
    SELECT turbohybrid_debug_last_scan_stats() INTO after_stats;

    INSERT INTO validation_overhead_results
    SELECT
        'direct_l2_distance',
        NULL,
        NULL,
        pg_temp.validation_counter_delta(before_stats, after_stats, 'vector_strict_validations'),
        pg_temp.validation_counter_delta(before_stats, after_stats, 'vector_fast_checks'),
        pg_temp.validation_counter_delta(before_stats, after_stats, 'vector_type_cache_misses'),
        pg_temp.validation_counter_delta(before_stats, after_stats, 'vector_type_cache_hits');

    DROP INDEX IF EXISTS fiqa_validation_turbohybrid_idx;
    CREATE INDEX fiqa_validation_turbohybrid_idx ON fiqa_docs
    USING turbohybrid (
        embedding vector_cosine_turbohybrid_ops,
        body_tsv bm25_tsvector_turbohybrid_ops
    )
    WITH (quantization_bits = 4, exact_storage = on);
    ANALYZE fiqa_docs;

    FOREACH row_count IN ARRAY ARRAY[small_k, large_k] LOOP
        SELECT turbohybrid_debug_last_scan_stats() INTO before_stats;
        PERFORM count(*)
        FROM (
            SELECT doc_id
            FROM fiqa_docs
            ORDER BY embedding <~> turbohybrid_query(
                vector_query => query_vector::vector,
                dense_k => row_count,
                bm25_k => 0,
                final_k => result_k
            )
            LIMIT result_k
        ) s;
        SELECT turbohybrid_debug_last_scan_stats() INTO after_stats;

        INSERT INTO validation_overhead_results
        SELECT
            'indexed_exact_on_' || row_count,
            row_count,
            true,
            pg_temp.validation_counter_delta(before_stats, after_stats, 'vector_strict_validations'),
            pg_temp.validation_counter_delta(before_stats, after_stats, 'vector_fast_checks'),
            pg_temp.validation_counter_delta(before_stats, after_stats, 'vector_type_cache_misses'),
            pg_temp.validation_counter_delta(before_stats, after_stats, 'vector_type_cache_hits');
    END LOOP;

    DROP INDEX fiqa_validation_turbohybrid_idx;
    CREATE INDEX fiqa_validation_turbohybrid_idx ON fiqa_docs
    USING turbohybrid (
        embedding vector_cosine_turbohybrid_ops,
        body_tsv bm25_tsvector_turbohybrid_ops
    )
    WITH (quantization_bits = 4, exact_storage = off);
    ANALYZE fiqa_docs;

    FOREACH row_count IN ARRAY ARRAY[small_k, large_k] LOOP
        SELECT turbohybrid_debug_last_scan_stats() INTO before_stats;
        PERFORM count(*)
        FROM (
            SELECT doc_id
            FROM fiqa_docs
            ORDER BY embedding <~> turbohybrid_query(
                vector_query => query_vector::vector,
                dense_k => row_count,
                bm25_k => 0,
                final_k => result_k
            )
            LIMIT result_k
        ) s;
        SELECT turbohybrid_debug_last_scan_stats() INTO after_stats;

        INSERT INTO validation_overhead_results
        SELECT
            'indexed_exact_off_' || row_count,
            row_count,
            false,
            pg_temp.validation_counter_delta(before_stats, after_stats, 'vector_strict_validations'),
            pg_temp.validation_counter_delta(before_stats, after_stats, 'vector_fast_checks'),
            pg_temp.validation_counter_delta(before_stats, after_stats, 'vector_type_cache_misses'),
            pg_temp.validation_counter_delta(before_stats, after_stats, 'vector_type_cache_hits');
    END LOOP;
END $$;

DO $$
DECLARE
    exact_on_small bigint;
    exact_on_large bigint;
    exact_off_small bigint;
    exact_off_large bigint;
    scan_cache_misses bigint;
    small_k int;
    large_k int;
BEGIN
    SELECT dense_small, dense_large
    INTO small_k, large_k
    FROM validation_overhead_settings;

    SELECT strict_delta INTO exact_on_small
    FROM validation_overhead_results
    WHERE scenario = 'indexed_exact_on_' || small_k::text;

    SELECT strict_delta INTO exact_on_large
    FROM validation_overhead_results
    WHERE scenario = 'indexed_exact_on_' || large_k::text;

    SELECT strict_delta INTO exact_off_small
    FROM validation_overhead_results
    WHERE scenario = 'indexed_exact_off_' || small_k::text;

    SELECT strict_delta INTO exact_off_large
    FROM validation_overhead_results
    WHERE scenario = 'indexed_exact_off_' || large_k::text;

    SELECT COALESCE(sum(cache_miss_delta), 0) INTO scan_cache_misses
    FROM validation_overhead_results
    WHERE scenario LIKE 'indexed_%';

    IF exact_on_large > exact_on_small + 2 THEN
        RAISE EXCEPTION
            'strict vector validations grew with exact-storage-on candidate count: % at dense_k %, % at dense_k %',
            exact_on_small, small_k, exact_on_large, large_k;
    END IF;

    IF exact_off_large > exact_off_small + 2 THEN
        RAISE EXCEPTION
            'strict vector validations grew with exact-storage-off candidate count: % at dense_k %, % at dense_k %',
            exact_off_small, small_k, exact_off_large, large_k;
    END IF;

    IF scan_cache_misses > 1 THEN
        RAISE EXCEPTION
            'vector type cache misses grew during indexed scans: % misses',
            scan_cache_misses;
    END IF;
END $$;

SELECT *
FROM validation_overhead_results
ORDER BY scenario;
