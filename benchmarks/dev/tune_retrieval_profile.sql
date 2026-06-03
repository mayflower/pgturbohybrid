\set ON_ERROR_STOP on
\pset pager off
\timing off

-- Practical retrieval-profile tuner for an existing TurboHybrid index.
--
-- Required input:
--   eval_queries(id int, vector_query vector null, tsquery tsquery null,
--                expected_ids bigint[] not null)
--
-- The script also accepts retrieval_quality_grid.sql-style query tables with
-- qvec/text_query/expected_ids columns.
--
-- Required psql variable:
--   INDEX_NAME        TurboHybrid index regclass
--
-- Optional psql variables:
--   TABLE_NAME        Heap table regclass. Defaults to the index table.
--   ID_COLUMN         Heap id column. Defaults to id.
--   LIMIT_K           Final result count and metric cutoff. Defaults to 10.
--   MAX_TRIALS        Trial cap for the practical grid. Defaults to 96.
--   EVAL_QUERY_TABLE  Query table or temp table. Defaults to eval_queries.
--   LATENCY_BUDGET_MS Optional p95 budget for the recommendation table.
--
-- Example:
--   psql -d "$PGDATABASE" \
--     -v INDEX_NAME=documents_turbohybrid_idx \
--     -v TABLE_NAME=documents \
--     -v ID_COLUMN=id \
--     -v LIMIT_K=10 \
--     -v MAX_TRIALS=96 \
--     -v EVAL_QUERY_TABLE=eval_queries \
--     -v LATENCY_BUDGET_MS=20 \
--     -f benchmarks/dev/tune_retrieval_profile.sql

\if :{?INDEX_NAME}
\else
\echo 'ERROR: set INDEX_NAME, for example -v INDEX_NAME=documents_turbohybrid_idx'
\quit 1
\endif

\if :{?TABLE_NAME}
\else
\set TABLE_NAME ''
\endif

\if :{?ID_COLUMN}
\else
\set ID_COLUMN id
\endif

\if :{?LIMIT_K}
\else
\set LIMIT_K 10
\endif

\if :{?MAX_TRIALS}
\else
\set MAX_TRIALS 96
\endif

\if :{?EVAL_QUERY_TABLE}
\else
\set EVAL_QUERY_TABLE eval_queries
\endif

\if :{?LATENCY_BUDGET_MS}
\else
\set LATENCY_BUDGET_MS NULL
\endif

DROP TABLE IF EXISTS pg_temp.th_tune_trials;
DROP TABLE IF EXISTS pg_temp.th_tune_eval_results;
DROP TABLE IF EXISTS pg_temp.th_tune_trial_summary;
DROP TABLE IF EXISTS pg_temp.th_tune_pareto_frontier;
DROP TABLE IF EXISTS pg_temp.th_tune_recommendation;
DROP TABLE IF EXISTS pg_temp.th_tune_args;

CREATE TEMP TABLE th_tune_args AS
SELECT :'INDEX_NAME'::text AS index_name,
       NULLIF(:'TABLE_NAME', '') AS table_name,
       :'ID_COLUMN'::text AS id_column,
       :LIMIT_K::int AS limit_k,
       :MAX_TRIALS::int AS max_trials,
       :'EVAL_QUERY_TABLE'::text AS eval_query_table,
       :LATENCY_BUDGET_MS::numeric AS latency_budget_ms;

CREATE TEMP TABLE th_tune_trials (
    trial_id int GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    profile text NOT NULL,
    dense_k int NOT NULL,
    bm25_k int NOT NULL,
    fusion text NOT NULL,
    residual_mode text NOT NULL,
    heap_rescore text NOT NULL,
    entry_sidecar_strategy text NOT NULL
);

CREATE TEMP TABLE th_tune_eval_results (
    trial_id int NOT NULL,
    query_id int NOT NULL,
    returned_ids bigint[] NOT NULL DEFAULT '{}',
    expected_ids bigint[] NOT NULL,
    overlap_count int NOT NULL DEFAULT 0,
    expected_count int NOT NULL DEFAULT 0,
    recall_at_k numeric NOT NULL DEFAULT 0,
    overlap_at_k numeric NOT NULL DEFAULT 0,
    elapsed_ms numeric,
    scan_stats jsonb NOT NULL DEFAULT '{}'::jsonb,
    error_text text
);

DO $$
DECLARE
    args record;
    index_reg regclass;
    table_reg regclass;
    index_table_reg regclass;
    eval_reg regclass;
    id_col name;
    vector_col name;
    eval_vector_col name;
    eval_text_col name;
    limit_k int;
    max_trials int;
    latency_budget_ms numeric;
    index_stats jsonb := '{}'::jsonb;
    sidecar_strategy text := 'current_index';
    has_heap_rescore_guc bool := current_setting('turbohybrid.dense_heap_rescore', true) IS NOT NULL;
    has_residual_guc bool := current_setting('turbohybrid.dense_residual_rerank_mode', true) IS NOT NULL;
    profiles text[] := ARRAY['latency', 'matched_recall', 'high_recall', 'quality'];
    dense_ks int[] := ARRAY[50, 100, 200, 400];
    bm25_ks int[] := ARRAY[50, 100, 200, 400];
    fusions text[] := ARRAY['rrf', 'fast_weighted', 'calibrated'];
    residual_modes text[] := ARRAY['default'];
    heap_rescores text[] := ARRAY['default'];
    fusion_name text;
    ok_fusions text[] := ARRAY[]::text[];
    trial record;
    query_row record;
    started_at timestamptz;
    elapsed numeric;
    returned bigint[];
    overlap int;
    expected_n int;
    stats jsonb;
    query_sql text;
    err text;
BEGIN
    SELECT *
    INTO args
    FROM th_tune_args;

    index_reg := args.index_name::regclass;
    eval_reg := args.eval_query_table::regclass;
    id_col := args.id_column::name;
    limit_k := args.limit_k;
    max_trials := args.max_trials;
    latency_budget_ms := args.latency_budget_ms;

    IF limit_k <= 0 THEN
        RAISE EXCEPTION 'LIMIT_K must be positive, got %', limit_k;
    END IF;
    IF max_trials <= 0 THEN
        RAISE EXCEPTION 'MAX_TRIALS must be positive, got %', max_trials;
    END IF;

    SELECT i.indrelid::regclass,
           a.attname
    INTO index_table_reg, vector_col
    FROM pg_index i
    JOIN pg_attribute a
      ON a.attrelid = i.indrelid
     AND a.attnum = i.indkey[0]
    WHERE i.indexrelid = index_reg;

    IF index_table_reg IS NULL OR vector_col IS NULL THEN
        RAISE EXCEPTION 'could not inspect index %', index_reg;
    END IF;

    IF args.table_name IS NULL THEN
        table_reg := index_table_reg;
    ELSE
        table_reg := args.table_name::regclass;
    END IF;

    IF table_reg <> index_table_reg THEN
        RAISE EXCEPTION 'TABLE_NAME % does not match index % table %',
            table_reg, index_reg, index_table_reg;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_attribute
        WHERE attrelid = table_reg
          AND attname = id_col
          AND NOT attisdropped
    ) THEN
        RAISE EXCEPTION 'ID_COLUMN %.% does not exist', table_reg, id_col;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_attribute
        WHERE attrelid = eval_reg
          AND attname = 'vector_query'
          AND NOT attisdropped
    ) THEN
        eval_vector_col := 'vector_query';
    ELSIF EXISTS (
        SELECT 1
        FROM pg_attribute
        WHERE attrelid = eval_reg
          AND attname = 'qvec'
          AND NOT attisdropped
    ) THEN
        eval_vector_col := 'qvec';
    ELSE
        RAISE EXCEPTION '% must have vector_query or qvec column', eval_reg;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_attribute
        WHERE attrelid = eval_reg
          AND attname = 'tsquery'
          AND NOT attisdropped
    ) THEN
        eval_text_col := 'tsquery';
    ELSIF EXISTS (
        SELECT 1
        FROM pg_attribute
        WHERE attrelid = eval_reg
          AND attname = 'text_query'
          AND NOT attisdropped
    ) THEN
        eval_text_col := 'text_query';
    ELSE
        RAISE EXCEPTION '% must have tsquery or text_query column', eval_reg;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_attribute
        WHERE attrelid = eval_reg
          AND attname = 'expected_ids'
          AND NOT attisdropped
    ) THEN
        RAISE EXCEPTION '% must have expected_ids bigint[] column', eval_reg;
    END IF;

    BEGIN
        EXECUTE 'SELECT turbohybrid_index_stats($1)' INTO index_stats USING index_reg;
        sidecar_strategy := COALESCE(index_stats->>'entry_sidecar_strategy',
                                     index_stats->>'graph_entry_sidecar_strategy',
                                     'current_index');
    EXCEPTION WHEN undefined_function THEN
        index_stats := '{}'::jsonb;
        sidecar_strategy := 'current_index';
    END;

    FOREACH fusion_name IN ARRAY fusions LOOP
        BEGIN
            PERFORM turbohybrid_query(vector_query => '[1]'::vector,
                                      fusion => fusion_name);
            ok_fusions := ok_fusions || fusion_name;
        EXCEPTION WHEN OTHERS THEN
            RAISE NOTICE 'skipping unavailable fusion mode %: %', fusion_name, SQLERRM;
        END;
    END LOOP;

    IF cardinality(ok_fusions) = 0 THEN
        RAISE EXCEPTION 'no requested fusion modes are available';
    END IF;

    IF has_residual_guc THEN
        residual_modes := ARRAY['off', 'fixed', 'calibrated'];
    END IF;
    IF has_heap_rescore_guc THEN
        heap_rescores := ARRAY['off', 'topk', 'band'];
    END IF;

    INSERT INTO th_tune_trials(profile, dense_k, bm25_k, fusion,
                               residual_mode, heap_rescore,
                               entry_sidecar_strategy)
    SELECT profile,
           GREATEST(limit_k, dense_k) AS dense_k,
           GREATEST(limit_k, bm25_k) AS bm25_k,
           fusion,
           residual_mode,
           heap_rescore,
           sidecar_strategy
    FROM unnest(profiles) WITH ORDINALITY AS p(profile, p_ord)
    CROSS JOIN unnest(dense_ks) WITH ORDINALITY AS dk(dense_k, dk_ord)
    CROSS JOIN unnest(bm25_ks) WITH ORDINALITY AS bk(bm25_k, bk_ord)
    CROSS JOIN unnest(ok_fusions) WITH ORDINALITY AS f(fusion, f_ord)
    CROSS JOIN unnest(residual_modes) WITH ORDINALITY AS r(residual_mode, r_ord)
    CROSS JOIN unnest(heap_rescores) WITH ORDINALITY AS h(heap_rescore, h_ord)
    ORDER BY p_ord, dk_ord, bk_ord, f_ord, r_ord, h_ord
    LIMIT max_trials;

    RAISE NOTICE 'tuning index %, table %, vector key %, eval table %, limit_k %, max_trials %, trials %',
        index_reg, table_reg, vector_col, eval_reg, limit_k, max_trials,
        (SELECT count(*) FROM th_tune_trials);
    RAISE NOTICE 'index sidecar strategy is %, index stats profile is %',
        sidecar_strategy, COALESCE(index_stats->>'profile', 'unknown');

    -- This harness evaluates TurboHybrid retrieval settings for the supplied
    -- index. Small synthetic eval tables are especially prone to sequential
    -- plans, so keep the benchmark focused on the index path.
    PERFORM set_config('enable_seqscan', 'off', true);

    query_sql := format(
        'SELECT COALESCE(array_agg(%1$I::bigint), ARRAY[]::bigint[])
           FROM (
             SELECT %1$I
             FROM %2$s
             ORDER BY %3$I <~> turbohybrid_query(
                 vector_query => $1,
                 text_query => $2,
                 fusion => $3,
                 dense_k => $4,
                 bm25_k => $5,
                 final_k => $6)
             LIMIT $6
           ) s',
        id_col, table_reg, vector_col);

    FOR trial IN
        SELECT *
        FROM th_tune_trials
        ORDER BY trial_id
    LOOP
        PERFORM set_config('turbohybrid.profile', trial.profile, true);
        IF has_residual_guc AND trial.residual_mode <> 'default' THEN
            PERFORM set_config('turbohybrid.dense_residual_rerank_mode',
                               trial.residual_mode, true);
        END IF;
        IF has_heap_rescore_guc AND trial.heap_rescore <> 'default' THEN
            PERFORM set_config('turbohybrid.dense_heap_rescore',
                               trial.heap_rescore, true);
        END IF;

        FOR query_row IN EXECUTE format(
            'SELECT id,
                    %1$I AS vector_query,
                    %2$I AS text_query,
                    expected_ids
             FROM %3$s
             WHERE expected_ids IS NOT NULL
             ORDER BY id',
            eval_vector_col, eval_text_col, eval_reg)
        LOOP
            returned := ARRAY[]::bigint[];
            stats := '{}'::jsonb;
            err := NULL;
            started_at := clock_timestamp();
            BEGIN
                EXECUTE query_sql
                INTO returned
                USING query_row.vector_query,
                      query_row.text_query,
                      trial.fusion,
                      trial.dense_k,
                      trial.bm25_k,
                      limit_k;
                elapsed := EXTRACT(epoch FROM clock_timestamp() - started_at) * 1000.0;
                stats := turbohybrid_last_scan_stats();
            EXCEPTION WHEN OTHERS THEN
                elapsed := EXTRACT(epoch FROM clock_timestamp() - started_at) * 1000.0;
                err := SQLERRM;
                returned := ARRAY[]::bigint[];
                stats := jsonb_build_object('error', err);
            END;

            SELECT count(*)::int
            INTO overlap
            FROM (
                SELECT DISTINCT x
                FROM unnest(returned) AS r(x)
                INTERSECT
                SELECT DISTINCT y
                FROM unnest(query_row.expected_ids[1:limit_k]) AS e(y)
            ) hits;

            expected_n := LEAST(limit_k, COALESCE(array_length(query_row.expected_ids, 1), 0));

            INSERT INTO th_tune_eval_results(trial_id, query_id, returned_ids,
                                             expected_ids, overlap_count,
                                             expected_count, recall_at_k,
                                             overlap_at_k, elapsed_ms,
                                             scan_stats, error_text)
            VALUES (
                trial.trial_id,
                query_row.id,
                COALESCE(returned, ARRAY[]::bigint[]),
                query_row.expected_ids,
                COALESCE(overlap, 0),
                expected_n,
                CASE WHEN expected_n > 0 THEN COALESCE(overlap, 0)::numeric / expected_n ELSE 0 END,
                COALESCE(overlap, 0)::numeric / limit_k,
                elapsed,
                COALESCE(stats, '{}'::jsonb),
                err
            );
        END LOOP;
    END LOOP;

    CREATE TEMP TABLE th_tune_trial_summary AS
    SELECT t.trial_id,
           t.profile,
           t.dense_k,
           t.bm25_k,
           t.fusion,
           t.residual_mode,
           t.heap_rescore,
           t.entry_sidecar_strategy,
           count(*) AS query_count,
           count(*) FILTER (WHERE r.error_text IS NOT NULL) AS failed_query_count,
           round(avg(r.recall_at_k), 4) AS avg_recall_at_k,
           round(avg(r.overlap_at_k), 4) AS avg_overlap_at_k,
           round(avg(r.elapsed_ms), 3) AS mean_ms,
           round(percentile_cont(0.50) WITHIN GROUP (ORDER BY r.elapsed_ms)::numeric, 3) AS p50_ms,
           round(percentile_cont(0.95) WITHIN GROUP (ORDER BY r.elapsed_ms)::numeric, 3) AS p95_ms,
           round(percentile_cont(0.99) WITHIN GROUP (ORDER BY r.elapsed_ms)::numeric, 3) AS p99_ms,
           max(NULLIF(r.scan_stats->>'graph_effective_search_ef', '')::int) AS graph_effective_search_ef,
           max(NULLIF(r.scan_stats->>'graph_effective_result_target', '')::int) AS graph_effective_result_target,
           max(NULLIF(r.scan_stats->>'graph_scored_codes', '')::bigint) AS graph_scored_codes_max,
           max(NULLIF(r.scan_stats->>'heap_rescore_count', '')::int) AS heap_rescore_count_max,
           max(r.scan_stats->>'exact_rescore_source') AS exact_rescore_source,
           max(r.scan_stats->>'dense_scorer') AS dense_scorer,
           bool_or(COALESCE((r.scan_stats->>'index_used')::bool, false)) AS any_index_used,
           max(r.error_text) FILTER (WHERE r.error_text IS NOT NULL) AS sample_error
    FROM th_tune_trials t
    JOIN th_tune_eval_results r USING (trial_id)
    GROUP BY t.trial_id, t.profile, t.dense_k, t.bm25_k, t.fusion,
             t.residual_mode, t.heap_rescore, t.entry_sidecar_strategy
    ORDER BY t.trial_id;

    CREATE TEMP TABLE th_tune_pareto_frontier AS
    SELECT s.*
    FROM th_tune_trial_summary s
    WHERE s.failed_query_count = 0
      AND NOT EXISTS (
          SELECT 1
          FROM th_tune_trial_summary better
          WHERE better.failed_query_count = 0
            AND better.avg_recall_at_k >= s.avg_recall_at_k
            AND better.p95_ms <= s.p95_ms
            AND (better.avg_recall_at_k > s.avg_recall_at_k
                 OR better.p95_ms < s.p95_ms)
      );

    CREATE TEMP TABLE th_tune_recommendation AS
    SELECT *
    FROM th_tune_pareto_frontier
    WHERE latency_budget_ms IS NOT NULL
      AND p95_ms <= latency_budget_ms
    ORDER BY avg_recall_at_k DESC, p95_ms ASC, mean_ms ASC, trial_id ASC
    LIMIT 1;
END $$;

\echo ''
\echo '=== all trials ==='
SELECT trial_id, profile, dense_k, bm25_k, fusion, residual_mode,
       heap_rescore, entry_sidecar_strategy, query_count, failed_query_count,
       avg_recall_at_k, avg_overlap_at_k, mean_ms, p50_ms, p95_ms, p99_ms,
       graph_effective_search_ef, graph_effective_result_target,
       graph_scored_codes_max, heap_rescore_count_max, exact_rescore_source,
       dense_scorer, any_index_used, sample_error
FROM th_tune_trial_summary
ORDER BY trial_id;

\echo ''
\echo '=== pareto frontier: maximize recall@k, minimize p95 ==='
SELECT trial_id, profile, dense_k, bm25_k, fusion, residual_mode,
       heap_rescore, entry_sidecar_strategy, avg_recall_at_k,
       avg_overlap_at_k, mean_ms, p50_ms, p95_ms, p99_ms,
       graph_effective_search_ef, graph_effective_result_target,
       graph_scored_codes_max, heap_rescore_count_max, exact_rescore_source,
       dense_scorer
FROM th_tune_pareto_frontier
ORDER BY p95_ms ASC, avg_recall_at_k DESC, trial_id ASC;

\echo ''
\echo '=== recommendation under LATENCY_BUDGET_MS, if provided ==='
SELECT :LATENCY_BUDGET_MS::numeric AS latency_budget_ms,
       trial_id, profile, dense_k, bm25_k, fusion, residual_mode,
       heap_rescore, entry_sidecar_strategy, avg_recall_at_k,
       avg_overlap_at_k, p95_ms, p99_ms
FROM th_tune_recommendation
CROSS JOIN th_tune_args;

SELECT 'no LATENCY_BUDGET_MS provided; use -v LATENCY_BUDGET_MS=20 to print a budgeted recommendation' AS note
FROM th_tune_args
WHERE latency_budget_ms IS NULL;
