-- Shared EXPLAIN helpers for turbohybrid index plan guards.
-- Included by scalar-fallback-guard.sql and regression tests.

CREATE OR REPLACE FUNCTION pg_temp.turbohybrid_plan_root(plan_doc jsonb)
RETURNS jsonb
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT CASE
        WHEN jsonb_typeof(plan_doc) = 'array' THEN plan_doc->0->'Plan'
        WHEN plan_doc ? 'Plan' THEN plan_doc->'Plan'
        ELSE plan_doc
    END
$$;

CREATE OR REPLACE FUNCTION pg_temp.turbohybrid_plan_nodes(plan_node jsonb)
RETURNS SETOF jsonb
LANGUAGE plpgsql
AS $$
DECLARE
    child jsonb;
BEGIN
    RETURN NEXT plan_node;

    FOR child IN
        SELECT value
        FROM jsonb_array_elements(COALESCE(plan_node->'Plans', '[]'::jsonb))
    LOOP
        RETURN QUERY SELECT * FROM pg_temp.turbohybrid_plan_nodes(child);
    END LOOP;
END;
$$;

CREATE OR REPLACE FUNCTION pg_temp.turbohybrid_plan_has_seq_scan(plan_node jsonb)
RETURNS bool
LANGUAGE sql
AS $$
    SELECT EXISTS (
        SELECT 1
        FROM pg_temp.turbohybrid_plan_nodes(plan_node) AS n(node)
        WHERE n.node->>'Node Type' = 'Seq Scan'
    )
$$;

CREATE OR REPLACE FUNCTION pg_temp.turbohybrid_plan_has_top_sort_over_seq_scan(plan_node jsonb)
RETURNS bool
LANGUAGE plpgsql
AS $$
DECLARE
    child jsonb;
    node_type text;
BEGIN
    node_type := plan_node->>'Node Type';

    IF node_type IN ('Sort', 'Incremental Sort') THEN
        RETURN pg_temp.turbohybrid_plan_has_seq_scan(plan_node);
    END IF;

    IF node_type IN ('Limit', 'Result', 'Gather', 'Gather Merge') THEN
        FOR child IN
            SELECT value
            FROM jsonb_array_elements(COALESCE(plan_node->'Plans', '[]'::jsonb))
        LOOP
            IF pg_temp.turbohybrid_plan_has_top_sort_over_seq_scan(child) THEN
                RETURN true;
            END IF;
        END LOOP;
    END IF;

    RETURN false;
END;
$$;

CREATE OR REPLACE FUNCTION pg_temp.turbohybrid_assert_index_plan(
    plan_doc jsonb,
    expected_index regclass,
    query_label text DEFAULT 'turbohybrid query')
RETURNS jsonb
LANGUAGE plpgsql
AS $$
DECLARE
    root jsonb;
    expected_index_name text;
    expected_am_name text;
    has_expected_index_scan bool;
    top_sort_seq bool;
BEGIN
    SELECT c.relname, am.amname
    INTO expected_index_name, expected_am_name
    FROM pg_class c
    JOIN pg_am am ON am.oid = c.relam
    WHERE c.oid = expected_index;

    IF expected_index_name IS NULL THEN
        RAISE EXCEPTION USING
            ERRCODE = '23514',
            MESSAGE = format('expected turbohybrid index does not exist for %s', query_label);
    END IF;

    IF expected_am_name <> 'turbohybrid' THEN
        RAISE EXCEPTION USING
            ERRCODE = '23514',
            MESSAGE = format('expected index %s uses access method %s, not turbohybrid',
                             expected_index_name, expected_am_name);
    END IF;

    root := pg_temp.turbohybrid_plan_root(plan_doc);
    IF root IS NULL THEN
        RAISE EXCEPTION USING
            ERRCODE = '23514',
            MESSAGE = format('EXPLAIN JSON did not contain a plan for %s', query_label);
    END IF;

    SELECT EXISTS (
        SELECT 1
        FROM pg_temp.turbohybrid_plan_nodes(root) AS n(node)
        WHERE n.node->>'Node Type' = 'Index Scan'
          AND n.node->>'Index Name' = expected_index_name
    )
    INTO has_expected_index_scan;

    top_sort_seq := pg_temp.turbohybrid_plan_has_top_sort_over_seq_scan(root);

    IF NOT has_expected_index_scan THEN
        RAISE EXCEPTION USING
            ERRCODE = '23514',
            MESSAGE = format('query %s did not use Index Scan on %s (scalar fallback risk)',
                             query_label, expected_index_name),
            DETAIL = left(plan_doc::text, 4000);
    END IF;

    IF top_sort_seq THEN
        RAISE EXCEPTION USING
            ERRCODE = '23514',
            MESSAGE = format('query %s used a top-level Sort over Seq Scan', query_label),
            DETAIL = left(plan_doc::text, 4000);
    END IF;

    RETURN jsonb_build_object(
        'valid', true,
        'query_label', query_label,
        'expected_index', expected_index_name,
        'access_method', expected_am_name,
        'has_expected_index_scan', has_expected_index_scan,
        'top_level_sort_over_seq_scan', top_sort_seq
    );
END;
$$;