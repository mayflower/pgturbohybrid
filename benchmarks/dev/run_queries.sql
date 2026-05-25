\set ON_ERROR_STOP on
\i benchmarks/dev/assert_index_plan.sql

\if :{?label}
\else
    \set label 'local'
\endif
\if :{?dataset}
\else
    \set dataset 'fiqa-openai'
\endif
\if :{?dense_k}
\else
    \set dense_k 100
\endif
\if :{?bm25_k}
\else
    \set bm25_k 100
\endif
\if :{?rrf_k}
\else
    \set rrf_k 60
\endif
\if :{?final_k}
\else
    \set final_k 10
\endif
\if :{?warmup}
\else
    \set warmup 1
\endif
\if :{?measure}
\else
    \set measure 5
\endif
\if :{?planner_debug}
\else
    \set planner_debug 1
\endif
\if :{?force_turbohybrid_index}
\else
    \set force_turbohybrid_index 0
\endif
\if :{?profile}
\else
    \set profile 'balanced'
\endif

SELECT set_config('turbohybrid.profile', :'profile', false);
\if :{?bm25_hot_postings_cache_mb}
SELECT set_config('turbohybrid.bm25_hot_postings_cache_mb', :'bm25_hot_postings_cache_mb', false);
\endif
\if :{?bm25_strategy}
SELECT set_config('turbohybrid.bm25_strategy', :'bm25_strategy', false);
\endif
\if :{?bm25_impact_or_mode}
SELECT set_config('turbohybrid.bm25_impact_or_mode', :'bm25_impact_or_mode', false);
\endif
\if :{?bm25_hybrid_bound}
SELECT set_config('turbohybrid.bm25_hybrid_bound', :'bm25_hybrid_bound', false);
\endif
\if :{?enable_wand}
SELECT set_config('turbohybrid.enable_wand', :'enable_wand', false);
\endif
\if :{?bm25_force_full_sort}
SELECT set_config('turbohybrid.bm25_force_full_sort', :'bm25_force_full_sort', false);
\endif
\if :{?bm25_accumulator_mode}
SELECT set_config('turbohybrid.bm25_accumulator_mode', :'bm25_accumulator_mode', false);
\endif

INSERT INTO perf_matrix_runs(label, dataset, dense_k, bm25_k, rrf_k, final_k, notes)
VALUES (:'label', :'dataset', :dense_k, :bm25_k, :rrf_k, :final_k,
        'real FIQA/OpenAI pgturbohybrid category matrix; profile=' ||
        current_setting('turbohybrid.profile'))
RETURNING run_id AS perf_run_id \gset

CREATE TEMP TABLE perf_matrix_current_run AS
SELECT
    :perf_run_id::bigint AS run_id,
    :dense_k::int AS dense_k,
    :bm25_k::int AS bm25_k,
    :rrf_k::int AS rrf_k,
    :final_k::int AS final_k,
    :warmup::int AS warmup,
    :measure::int AS measure,
    (:planner_debug::int <> 0) AS planner_debug,
    (:force_turbohybrid_index::int <> 0) AS force_turbohybrid_index;

DO $$
DECLARE
    c record;
    p record;
    sql text;
    plan jsonb;
    forced_plan jsonb;
    planner_validation jsonb;
    started timestamptz;
    elapsed double precision;
    forced_elapsed double precision;
    i int;
    stats jsonb;
    debug_stats jsonb;
    index_json jsonb;
    simd_json jsonb;
    used_index bool;
    cost_regression bool;
BEGIN
    SELECT * INTO p FROM perf_matrix_current_run;
    IF p.force_turbohybrid_index THEN
        EXECUTE 'SET LOCAL enable_seqscan = off';
    END IF;

    FOR c IN SELECT * FROM perf_matrix_query_cases ORDER BY category LOOP
        IF c.category = 'dense_only' THEN
            sql := format(
                'SELECT doc_id FROM fiqa_docs ORDER BY embedding <~> turbohybrid_query(vector_query => %L::vector, dense_k => %s, bm25_k => %s, rrf_k => %s, final_k => %s) LIMIT %s',
                c.vector_query, p.dense_k, p.bm25_k, p.rrf_k, p.final_k, p.final_k);
        ELSIF c.category IN ('bm25_or', 'bm25_and') THEN
            sql := format(
                'SELECT doc_id FROM fiqa_docs ORDER BY embedding <~> turbohybrid_query(text_query => to_tsquery(''english'', %L), dense_k => %s, bm25_k => %s, rrf_k => %s, final_k => %s) LIMIT %s',
                c.text_query, p.dense_k, p.bm25_k, p.rrf_k, p.final_k, p.final_k);
        ELSIF c.category LIKE 'bm25_%' THEN
            sql := format(
                'SELECT doc_id FROM fiqa_docs ORDER BY embedding <~> turbohybrid_query(text_query => plainto_tsquery(''english'', %L), dense_k => %s, bm25_k => %s, rrf_k => %s, final_k => %s) LIMIT %s',
                c.text_query, p.dense_k, p.bm25_k, p.rrf_k, p.final_k, p.final_k);
        ELSIF c.category IN ('hybrid_or', 'hybrid_and') THEN
            sql := format(
                'SELECT doc_id FROM fiqa_docs ORDER BY embedding <~> turbohybrid_query(vector_query => %L::vector, text_query => to_tsquery(''english'', %L), dense_k => %s, bm25_k => %s, rrf_k => %s, final_k => %s) LIMIT %s',
                c.vector_query, c.text_query, p.dense_k, p.bm25_k, p.rrf_k, p.final_k, p.final_k);
        ELSE
            sql := format(
                'SELECT doc_id FROM fiqa_docs ORDER BY embedding <~> turbohybrid_query(vector_query => %L::vector, text_query => plainto_tsquery(''english'', %L), dense_k => %s, bm25_k => %s, rrf_k => %s, final_k => %s) LIMIT %s',
                c.vector_query, c.text_query, p.dense_k, p.bm25_k, p.rrf_k, p.final_k, p.final_k);
        END IF;

        EXECUTE 'EXPLAIN (ANALYZE, BUFFERS, SETTINGS, VERBOSE, FORMAT JSON) ' || sql INTO plan;
        planner_validation := pg_temp.perf_assert_turbohybrid_index_plan(
            plan, 'fiqa_turbohybrid_idx'::regclass, c.category);
        used_index := (planner_validation->>'valid')::bool;
        forced_plan := NULL;
        forced_elapsed := NULL;

        IF p.planner_debug THEN
            EXECUTE 'SET LOCAL enable_seqscan = off';
            EXECUTE 'EXPLAIN (ANALYZE, BUFFERS, SETTINGS, VERBOSE, FORMAT JSON) ' || sql INTO forced_plan;
            PERFORM pg_temp.perf_assert_turbohybrid_index_plan(
                forced_plan, 'fiqa_turbohybrid_idx'::regclass, c.category || ' forced');
            started := clock_timestamp();
            EXECUTE sql;
            forced_elapsed := EXTRACT(EPOCH FROM clock_timestamp() - started) * 1000.0;
            IF p.force_turbohybrid_index THEN
                EXECUTE 'SET LOCAL enable_seqscan = off';
            ELSE
                EXECUTE 'RESET enable_seqscan';
            END IF;
        END IF;

        FOR i IN 1..p.warmup LOOP
            EXECUTE sql;
        END LOOP;

        FOR i IN 1..p.measure LOOP
            started := clock_timestamp();
            EXECUTE sql;
            elapsed := EXTRACT(EPOCH FROM clock_timestamp() - started) * 1000.0;
            stats := turbohybrid_last_scan_stats();
            cost_regression := forced_elapsed IS NOT NULL AND
                forced_elapsed > 0.0 AND elapsed > forced_elapsed * 2.0;
            IF to_regprocedure('turbohybrid_debug_last_scan_stats()') IS NOT NULL THEN
                BEGIN
                    EXECUTE 'SELECT turbohybrid_debug_last_scan_stats()' INTO debug_stats;
                EXCEPTION WHEN undefined_function THEN
                    debug_stats := NULL;
                END;
            ELSE
                debug_stats := NULL;
            END IF;
            index_json := turbohybrid_index_stats('fiqa_turbohybrid_idx'::regclass);
            simd_json := turbohybrid_simd_capabilities();
            INSERT INTO perf_matrix_query_results(
                run_id, category, query_sql, plan, forced_plan, planner_validation,
                last_scan_stats, debug_scan_stats, index_stats, simd_capabilities,
                elapsed_ms, forced_elapsed_ms, cost_model_regression, used_turbohybrid_index
            )
            VALUES (
                p.run_id, c.category, sql, plan, forced_plan, planner_validation,
                stats, debug_stats, index_json, simd_json, elapsed, forced_elapsed,
                cost_regression, used_index
            );
        END LOOP;
    END LOOP;
END $$;

SELECT
    category,
    count(*) AS measured_queries,
    round(min(elapsed_ms)::numeric, 3) AS min_ms,
    round(avg(elapsed_ms)::numeric, 3) AS mean_ms,
    round(percentile_cont(0.50) WITHIN GROUP (ORDER BY elapsed_ms)::numeric, 3) AS p50_ms,
    round(percentile_cont(0.95) WITHIN GROUP (ORDER BY elapsed_ms)::numeric, 3) AS p95_ms,
    round(percentile_cont(0.99) WITHIN GROUP (ORDER BY elapsed_ms)::numeric, 3) AS p99_ms,
    round(max(elapsed_ms)::numeric, 3) AS max_ms,
    bool_and(used_turbohybrid_index) AS used_turbohybrid_index,
    bool_or(cost_model_regression) AS cost_model_regression
FROM perf_matrix_query_results
WHERE run_id = :perf_run_id
GROUP BY category
ORDER BY category;
