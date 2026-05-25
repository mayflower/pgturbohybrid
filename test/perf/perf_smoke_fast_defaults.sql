\set ON_ERROR_STOP on
SET client_min_messages = warning;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;

DROP TABLE IF EXISTS perf_smoke_docs;
CREATE TABLE perf_smoke_docs (
	id bigserial PRIMARY KEY,
	embedding vector(3) NOT NULL,
	body text NOT NULL,
	body_tsv tsvector GENERATED ALWAYS AS (
		to_tsvector('english', body)
	) STORED
);

INSERT INTO perf_smoke_docs (embedding, body)
SELECT
	format('[%s,%s,%s]',
		   1.0 - (g % 17)::float8 / 100.0,
		   (g % 31)::float8 / 100.0,
		   (g % 43)::float8 / 100.0)::vector(3),
	CASE
		WHEN g % 7 = 0 THEN 'postgres hybrid search bm25 vector latency common'
		WHEN g % 5 = 0 THEN 'postgres hybrid search vector common'
		ELSE 'postgres hybrid search bm25 common'
	END
FROM generate_series(1, 1500) AS g;

CREATE INDEX perf_smoke_docs_idx ON perf_smoke_docs
USING turbohybrid (
	embedding vector_cosine_turbohybrid_ops,
	body_tsv bm25_tsvector_turbohybrid_ops
);

ANALYZE perf_smoke_docs;

-- Keep the smoke structural. The purpose is to catch regressions where the
-- default index path is unavailable or no longer fast-path shaped, not to test
-- PostgreSQL's cost model on a tiny CI table.
SET enable_seqscan = off;

CREATE TEMP TABLE perf_smoke_output (result jsonb);

DO $$
DECLARE
	index_stats jsonb;
	warmup_stats jsonb;
	measured_stats jsonb;
	guard_before jsonb;
	guard_after jsonb;
	simd_stats jsonb;
	plan_json jsonb;
	index_used bool;
	seq_scan_seen bool;
	sort_seen bool;
	execution_ms float8;
	strict_validation_delta bigint;
	fast_check_delta bigint;
	row_count int;
BEGIN
	index_stats := turbohybrid_index_stats('perf_smoke_docs_idx'::regclass);
	simd_stats := turbohybrid_simd_capabilities();

	IF index_stats->>'profile' <> 'latency' THEN
		RAISE EXCEPTION 'expected latency profile, got %', index_stats->>'profile';
	END IF;

	IF (index_stats->>'quantization_bits')::int <> 4 THEN
		RAISE EXCEPTION 'expected 4-bit default index, got %',
			index_stats->>'quantization_bits';
	END IF;

	IF (index_stats->>'exact_storage')::bool THEN
		RAISE EXCEPTION 'expected exact_storage=false default index';
	END IF;

	IF (simd_stats->>'simd_build_disabled')::bool THEN
		RAISE EXCEPTION 'SIMD was unexpectedly disabled in perf smoke build';
	END IF;

	SELECT count(*) INTO row_count
	FROM (
		SELECT id
		FROM perf_smoke_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			text_query => websearch_to_tsquery('english', 'postgres hybrid search')
		)
		LIMIT 10
	) AS warmup;

	IF row_count <> 10 THEN
		RAISE EXCEPTION 'expected warmup LIMIT 10 to return 10 rows, got %',
			row_count;
	END IF;

	warmup_stats := turbohybrid_last_scan_stats();

	EXECUTE $query$
		EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON)
		SELECT id
		FROM perf_smoke_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			text_query => websearch_to_tsquery('english', 'postgres hybrid search')
		)
		LIMIT 10
	$query$ INTO plan_json;

	measured_stats := turbohybrid_last_scan_stats();
	execution_ms := ((plan_json->0)->>'Execution Time')::float8;

	WITH RECURSIVE plan_nodes(node) AS (
		SELECT plan_json->0->'Plan'
		UNION ALL
		SELECT children.child
		FROM plan_nodes,
			 LATERAL jsonb_array_elements(COALESCE(node->'Plans', '[]'::jsonb))
				AS children(child)
	)
	SELECT
		bool_or(node->>'Index Name' = 'perf_smoke_docs_idx'),
		bool_or(node->>'Node Type' = 'Seq Scan'),
		bool_or(node->>'Node Type' = 'Sort')
	INTO index_used, seq_scan_seen, sort_seen
	FROM plan_nodes;

	IF NOT COALESCE(index_used, false) THEN
		RAISE EXCEPTION 'perf smoke query did not use perf_smoke_docs_idx';
	END IF;

	IF COALESCE(seq_scan_seen, false) OR COALESCE(sort_seen, false) THEN
		RAISE EXCEPTION 'perf smoke query used seq scan or sort: seq_scan=%, sort=%',
			seq_scan_seen, sort_seen;
	END IF;

	IF measured_stats->>'profile' <> 'latency' THEN
		RAISE EXCEPTION 'expected scan profile latency, got %',
			measured_stats->>'profile';
	END IF;

	IF (measured_stats->>'dense_k_effective')::int <> 100 THEN
		RAISE EXCEPTION 'expected dense_k_effective=100, got %',
			measured_stats->>'dense_k_effective';
	END IF;

	IF (measured_stats->>'bm25_k_effective')::int <> 100 THEN
		RAISE EXCEPTION 'expected bm25_k_effective=100, got %',
			measured_stats->>'bm25_k_effective';
	END IF;

	IF (measured_stats->>'final_k_effective')::int <> 10 OR
	   (measured_stats->>'detected_sql_limit')::int <> 10 OR
	   NOT (measured_stats->>'final_k_inferred')::bool THEN
		RAISE EXCEPTION 'expected final_k inferred from LIMIT 10, got %',
			measured_stats;
	END IF;

	IF NOT (measured_stats->>'bm25_cache_hit')::bool THEN
		RAISE EXCEPTION 'expected BM25 cache hit after warmup, got %',
			measured_stats;
	END IF;

	IF execution_ms > 1000.0 THEN
		RAISE EXCEPTION 'perf smoke exceeded loose latency guard: % ms',
			execution_ms;
	END IF;

	guard_before := turbohybrid_last_scan_stats();

	SELECT count(*) INTO row_count
	FROM (
		SELECT id
		FROM perf_smoke_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			text_query => websearch_to_tsquery('english', 'postgres hybrid search')
		)
		LIMIT 100
	) AS validation_guard;

	IF row_count <> 100 THEN
		RAISE EXCEPTION 'expected validation guard LIMIT 100 to return 100 rows, got %',
			row_count;
	END IF;

	guard_after := turbohybrid_last_scan_stats();
	strict_validation_delta :=
		(guard_after->>'strict_vector_validations')::bigint -
		(guard_before->>'strict_vector_validations')::bigint;
	fast_check_delta :=
		(guard_after->>'fast_vector_checks')::bigint -
		(guard_before->>'fast_vector_checks')::bigint;

	IF strict_validation_delta >= 100 THEN
		RAISE EXCEPTION 'strict vector validations scaled with candidates: %',
			strict_validation_delta;
	END IF;

	INSERT INTO perf_smoke_output
	SELECT jsonb_build_object(
		'status', 'pass',
		'profile', measured_stats->>'profile',
		'measured_execution_ms', execution_ms,
		'index_used', index_used,
		'seq_scan_seen', COALESCE(seq_scan_seen, false),
		'sort_seen', COALESCE(sort_seen, false),
		'index_stats', index_stats,
		'warmup_scan_stats', warmup_stats,
		'measured_scan_stats', measured_stats,
		'validation_guard', jsonb_build_object(
			'strict_vector_validation_delta', strict_validation_delta,
			'fast_vector_check_delta', fast_check_delta
		),
		'simd_capabilities', simd_stats,
		'plan', plan_json
	);
END
$$;

SELECT jsonb_pretty(result)
FROM perf_smoke_output;
