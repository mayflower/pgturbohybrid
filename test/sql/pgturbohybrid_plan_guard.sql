\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;

DROP TABLE IF EXISTS tqh_plan_guard_docs;
CREATE TABLE tqh_plan_guard_docs (
	id int PRIMARY KEY,
	embedding vector(3) NOT NULL,
	body_tsv tsvector NOT NULL
);

INSERT INTO tqh_plan_guard_docs (id, embedding, body_tsv) VALUES
	(1, '[1,0,0]', to_tsvector('english', 'postgres hybrid search vector')),
	(2, '[0.9,0.1,0]', to_tsvector('english', 'hybrid bm25 postgres')),
	(3, '[0.8,0.2,0]', to_tsvector('english', 'search retrieval common'));

CREATE INDEX tqh_plan_guard_idx ON tqh_plan_guard_docs
USING turbohybrid (
	embedding vector_cosine_turbohybrid_ops,
	body_tsv bm25_tsvector_turbohybrid_ops
);

ANALYZE tqh_plan_guard_docs;
SET enable_seqscan = off;

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
        RAISE EXCEPTION 'expected turbohybrid index does not exist for %', query_label;
    END IF;

    IF expected_am_name <> 'turbohybrid' THEN
        RAISE EXCEPTION 'expected index % uses %, not turbohybrid',
            expected_index_name, expected_am_name;
    END IF;

    root := pg_temp.turbohybrid_plan_root(plan_doc);
    SELECT EXISTS (
        SELECT 1
        FROM pg_temp.turbohybrid_plan_nodes(root) AS n(node)
        WHERE n.node->>'Node Type' = 'Index Scan'
          AND n.node->>'Index Name' = expected_index_name
    )
    INTO has_expected_index_scan;

    top_sort_seq := pg_temp.turbohybrid_plan_has_top_sort_over_seq_scan(root);

    IF NOT has_expected_index_scan THEN
        RAISE EXCEPTION 'query % did not use Index Scan on %',
            query_label, expected_index_name;
    END IF;

    IF top_sort_seq THEN
        RAISE EXCEPTION 'query % used Sort over Seq Scan', query_label;
    END IF;

    RETURN jsonb_build_object(
        'valid', true,
        'query_label', query_label,
        'expected_index', expected_index_name
    );
END;
$$;

DO $$
DECLARE
    plan_json jsonb;
    guard_result jsonb;
    scan_stats jsonb;
BEGIN
    EXECUTE $plan$
        EXPLAIN (FORMAT JSON)
        SELECT id
        FROM tqh_plan_guard_docs
        ORDER BY embedding <~> turbohybrid_query(
            vector_query => '[1,0,0]'::vector,
            text_query => websearch_to_tsquery('english', 'postgres hybrid search')
        )
        LIMIT 5
    $plan$
    INTO plan_json;

    guard_result := pg_temp.turbohybrid_assert_index_plan(
        plan_json,
        'tqh_plan_guard_idx'::regclass,
        'regression hybrid plan guard');

    PERFORM id
    FROM tqh_plan_guard_docs
    ORDER BY embedding <~> turbohybrid_query(
        vector_query => '[1,0,0]'::vector,
        text_query => websearch_to_tsquery('english', 'postgres hybrid search')
    )
    LIMIT 5;

    scan_stats := turbohybrid_last_scan_stats();

    IF scan_stats->>'index_shape' IS DISTINCT FROM 'hybrid' THEN
        RAISE EXCEPTION 'expected hybrid index_shape, got %', scan_stats->>'index_shape';
    END IF;

    RAISE NOTICE 'plan guard ok: %', guard_result;
END;
$$;

SELECT 'plan_guard_pass' AS status;