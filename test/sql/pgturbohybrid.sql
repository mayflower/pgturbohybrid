DO $$
BEGIN
	IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pgturbohybrid') THEN
		EXECUTE 'DROP EXTENSION pgturbohybrid CASCADE';
	END IF;
	IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
		EXECUTE 'DROP EXTENSION vector CASCADE';
	END IF;
END
$$;

CREATE EXTENSION vector;
CREATE EXTENSION pgturbohybrid;

SET enable_seqscan = off;

DO $$
BEGIN
	IF current_setting('turbohybrid.profile') <> 'latency' THEN
		RAISE EXCEPTION 'unexpected profile default: %',
			current_setting('turbohybrid.profile');
	END IF;

	IF current_setting('turbohybrid.dense_adaptive_widening') <> 'off' THEN
		RAISE EXCEPTION 'unexpected dense_adaptive_widening default: %',
			current_setting('turbohybrid.dense_adaptive_widening');
	END IF;

	IF current_setting('turbohybrid.dense_local_expansion') <> 'off' THEN
		RAISE EXCEPTION 'unexpected dense_local_expansion default: %',
			current_setting('turbohybrid.dense_local_expansion');
	END IF;

	IF current_setting('turbohybrid.dense_uncertainty_retry') <> 'off' THEN
		RAISE EXCEPTION 'unexpected dense_uncertainty_retry default: %',
			current_setting('turbohybrid.dense_uncertainty_retry');
	END IF;

	IF current_setting('turbohybrid.dense_build_neighbor_select') <> 'auto' THEN
		RAISE EXCEPTION 'unexpected dense_build_neighbor_select default: %',
			current_setting('turbohybrid.dense_build_neighbor_select');
	END IF;

	IF current_setting('turbohybrid.hybrid_budget_policy') <> 'fixed' THEN
		RAISE EXCEPTION 'unexpected hybrid_budget_policy default: %',
			current_setting('turbohybrid.hybrid_budget_policy');
	END IF;

	IF current_setting('turbohybrid.bm25_heap_tsvector_rerank') <> 'off' THEN
		RAISE EXCEPTION 'unexpected bm25_heap_tsvector_rerank default: %',
			current_setting('turbohybrid.bm25_heap_tsvector_rerank');
	END IF;

	IF current_setting('turbohybrid.warn_linear_fallback') <> 'on' THEN
		RAISE EXCEPTION 'unexpected warn_linear_fallback default: %',
			current_setting('turbohybrid.warn_linear_fallback');
	END IF;

	IF current_setting('turbohybrid.linear_fallback_notice_threshold_ratio') <> '0.25' THEN
		RAISE EXCEPTION 'unexpected linear_fallback_notice_threshold_ratio default: %',
			current_setting('turbohybrid.linear_fallback_notice_threshold_ratio');
	END IF;
END
$$;

CREATE TABLE tqh_docs (
	id int,
	embedding vector(3),
	body_tsv tsvector
);

INSERT INTO tqh_docs VALUES
	(1, '[1,0,0]', to_tsvector('english', 'postgres vector search')),
	(2, '[1,1,0]', to_tsvector('english', 'hybrid bm25 search')),
	(3, '[0,1,0]', to_tsvector('english', 'lexical search')),
	(4, '[0,0,1]', to_tsvector('english', 'unrelated document'));

CREATE INDEX tqh_docs_idx ON tqh_docs
USING turbohybrid (
	embedding vector_cosine_turbohybrid_ops,
	body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (quantization_bits = 4, exact_storage = on);

CREATE TABLE tqh_default_docs (
	id int,
	embedding vector(3),
	body_tsv tsvector
);

INSERT INTO tqh_default_docs VALUES
	(1, '[1,0,0]', to_tsvector('english', 'postgres vector search')),
	(2, '[1,1,0]', to_tsvector('english', 'hybrid bm25 search')),
	(3, '[0,1,0]', to_tsvector('english', 'lexical search')),
	(4, '[0,0,1]', to_tsvector('english', 'unrelated document'));

CREATE INDEX tqh_default_docs_idx ON tqh_default_docs
USING turbohybrid (
	embedding vector_cosine_turbohybrid_ops,
	body_tsv bm25_tsvector_turbohybrid_ops
);

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_index_stats('tqh_default_docs_idx'::regclass);

	IF stats->>'profile' <> 'latency' OR
		(stats->>'graph_m')::int <> 16 OR
		(stats->>'graph_ef_construction')::int <> 128 OR
		(stats->>'graph_ef_search')::int <> 64 OR
		(stats->>'graph_oversampling')::int <> 4 OR
		stats->>'routing' <> 'auto' OR
		stats->>'storage_kind' <> 'pgturbohybrid_graph_native' OR
		stats->>'index_shape' <> 'hybrid' OR
		(stats->>'bm25_branch_available')::boolean IS DISTINCT FROM true OR
		(stats->>'bm25_document_count')::int <> 4 OR
		(stats->>'quantization_bits')::int <> 4 OR
		stats->>'dense_build_distance_mode' <> 'code' OR
		stats->>'build_neighbor_select' <> 'heuristic' OR
		stats->>'build_neighbor_select_reason' <> 'auto_lowdim' OR
		(stats->>'build_fast_edges')::boolean IS DISTINCT FROM false OR
		(stats->>'entry_sidecar_count')::int <> 0 OR
		(stats->>'residual_rerank_bytes')::int <> 0 OR
		(stats->>'dense_build_exact_distances')::boolean IS DISTINCT FROM false OR
		(stats->>'native_segments')::int <> 1 OR
		(stats->>'exact_storage')::boolean IS DISTINCT FROM false THEN
		RAISE EXCEPTION 'unexpected default index stats: %', stats;
	END IF;
END
$$;

CREATE TABLE tqh_segment_docs (
	id int,
	embedding vector(3)
);

INSERT INTO tqh_segment_docs VALUES
	(1, '[1,0,0]'),
	(2, '[0.9,0.1,0]'),
	(3, '[0,1,0]'),
	(4, '[0,0.9,0.1]'),
	(5, '[0,0,1]'),
	(6, '[0.1,0,0.9]');

CREATE INDEX tqh_segment_docs_idx ON tqh_segment_docs
USING turbohybrid (embedding vector_cosine_turbohybrid_ops)
WITH (native_segments = 2, quantization_bits = 4, exact_storage = off);

DO $$
DECLARE
	stats jsonb;
	result_count int;
BEGIN
	stats := turbohybrid_index_stats('tqh_segment_docs_idx'::regclass);
	IF (stats->>'native_segments')::int <> 2 OR
		(stats->>'native_segment_bytes')::int <> 64 OR
		stats->>'index_shape' <> 'dense_only' THEN
		RAISE EXCEPTION 'unexpected segmented native index stats: %', stats;
	END IF;

	SELECT count(*) INTO result_count
	FROM (
		SELECT id
		FROM tqh_segment_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			dense_k => 4,
			final_k => 4
		)
		LIMIT 4
	) s;

	IF result_count <> 4 THEN
		RAISE EXCEPTION 'segmented native dense query returned % rows', result_count;
	END IF;

	stats := turbohybrid_last_scan_stats();
	IF (stats->>'graph_segment_count')::int <> 2 OR
		(stats->>'native_segments')::int <> 2 OR
		stats->>'per_segment_budget_mode' <> 'sqrt' OR
		(stats->>'effective_search_ef_after_segment_scaling')::int <
			(stats->>'effective_search_ef_before_segment_scaling')::int OR
		(stats->>'graph_segments_searched')::int <> 2 THEN
		RAISE EXCEPTION 'unexpected segmented scan stats: %', stats;
	END IF;
END
$$;

CREATE INDEX tqh_docs_id_btree_idx ON tqh_docs (id);

DO $$
DECLARE
	dense_stats jsonb;
	hybrid_stats jsonb;
	btree_stats jsonb;
BEGIN
	dense_stats := turbohybrid_estimate_memory('tqh_segment_docs_idx'::regclass);
	hybrid_stats := turbohybrid_estimate_memory('tqh_default_docs_idx'::regclass);
	btree_stats := turbohybrid_estimate_memory('tqh_docs_id_btree_idx'::regclass);

	IF dense_stats->>'index' <> 'tqh_segment_docs_idx' OR
		(dense_stats->>'node_count')::int <> 6 OR
		(dense_stats->>'dimensions')::int <> 3 OR
		(dense_stats->>'quantization_bits')::int <> 4 OR
		(dense_stats->'native'->>'available')::boolean IS DISTINCT FROM true OR
		(dense_stats->'native'->>'code_bytes')::bigint <= 0 OR
		(dense_stats->'native'->>'adjacency_bytes')::bigint <= 0 OR
		(dense_stats->'native'->>'node_bytes')::bigint <= 0 OR
		(dense_stats->'native'->>'visited_generation_bytes')::bigint <= 0 OR
		(dense_stats->'native'->>'page_map_bytes')::bigint <= 0 OR
		(dense_stats->'native'->>'estimated_total_bytes')::bigint <=
			(dense_stats->'native'->>'code_bytes')::bigint OR
		(dense_stats->'bm25'->>'available')::boolean IS DISTINCT FROM false THEN
		RAISE EXCEPTION 'unexpected dense memory estimate: %', dense_stats;
	END IF;

	IF hybrid_stats->>'index' <> 'tqh_default_docs_idx' OR
		(hybrid_stats->>'node_count')::int <> 4 OR
		(hybrid_stats->'native'->>'available')::boolean IS DISTINCT FROM true OR
		(hybrid_stats->'native'->>'cache_policy') IS NULL OR
		(hybrid_stats->'bm25'->>'available')::boolean IS DISTINCT FROM true OR
		(hybrid_stats->'bm25'->>'doc_lens_bytes')::bigint <= 0 OR
		(hybrid_stats->'bm25'->>'heap_tids_bytes')::bigint <= 0 OR
		(hybrid_stats->'bm25'->>'live_nodes_bytes')::bigint <= 0 OR
		(hybrid_stats->'bm25'->>'lexicon_entries')::int <= 0 OR
		(hybrid_stats->'bm25'->>'estimated_base_cache_bytes')::bigint <=
			(hybrid_stats->'bm25'->>'doc_lens_bytes')::bigint OR
		(hybrid_stats->'concurrency'->>'per_backend_total_bytes_per_backend')::bigint <= 0 OR
		(hybrid_stats->'concurrency'->>'bm25_total_bytes_per_backend')::bigint <= 0 THEN
		RAISE EXCEPTION 'unexpected hybrid memory estimate: %', hybrid_stats;
	END IF;

	IF btree_stats->>'index' <> 'tqh_docs_id_btree_idx' OR
		(btree_stats->>'node_count')::int <> 0 OR
		(btree_stats->>'dimensions')::int <> 0 OR
		(btree_stats->'native'->>'available')::boolean IS DISTINCT FROM false OR
		(btree_stats->'bm25'->>'available')::boolean IS DISTINCT FROM false OR
		(btree_stats->'concurrency'->>'per_backend_total_bytes_per_backend')::bigint <> 0 THEN
		RAISE EXCEPTION 'unexpected btree memory estimate: %', btree_stats;
	END IF;
END
$$;

DROP INDEX tqh_docs_id_btree_idx;

CREATE TABLE tqh_fill_band_docs (
	id int,
	grp int,
	embedding vector(3)
);

INSERT INTO tqh_fill_band_docs
SELECT g,
	CASE WHEN g <= 1100 THEN 1 ELSE 2 END,
	('[' || (g % 7)::text || ',' || (g % 11)::text || ',' ||
		(g % 13)::text || ']')::vector
FROM generate_series(1, 1200) AS g;

CREATE INDEX tqh_fill_band_docs_idx ON tqh_fill_band_docs
USING turbohybrid (embedding vector_cosine_turbohybrid_ops) INCLUDE (grp)
WITH (quantization_bits = 4, exact_storage = off);

ANALYZE tqh_fill_band_docs;

DO $$
DECLARE
	stats jsonb;
	result_count int;
BEGIN
	PERFORM set_config('turbohybrid.warn_linear_fallback', 'off', true);
	PERFORM set_config('turbohybrid.linear_fallback_notice_threshold_ratio', '0.25', true);
	PERFORM set_config('turbohybrid.payload_entry_seeding', 'off', true);

	SELECT count(*) INTO result_count
	FROM (
		SELECT id
		FROM tqh_fill_band_docs
		WHERE grp = 1
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			dense_k => 20,
			final_k => 8
		)
		LIMIT 8
	) s;

	stats := turbohybrid_last_scan_stats();
	IF result_count <> 8 OR
		(stats->>'graph_fill_candidate_band_calls')::int <= 0 OR
		stats->>'graph_fill_candidate_band_reason' <> 'payload_exact_band_miss' OR
		(stats->>'graph_fill_candidate_band_visited')::int <= 0 OR
		(stats->>'graph_fill_candidate_band_scored')::int <= 0 OR
		(stats->>'graph_fill_candidate_band_selected_before')::int >=
			(stats->>'graph_fill_candidate_band_selected_after')::int OR
		(stats->>'graph_fill_candidate_band_target')::int <= 0 OR
		(stats->>'graph_fill_candidate_band_used_payload_refs')::boolean IS DISTINCT FROM true OR
		(stats->>'graph_fill_candidate_band_payload_ref_count')::int <= 1024 OR
		(stats->'dense'->'traversal'->>'fill_candidate_band_calls')::int <>
			(stats->>'graph_fill_candidate_band_calls')::int OR
		stats->'dense'->'traversal'->>'fill_candidate_band_reason' <>
			stats->>'graph_fill_candidate_band_reason' THEN
		RAISE EXCEPTION 'unexpected fill candidate band stats: %', stats;
	END IF;
END
$$;

DO $$
DECLARE
	stats jsonb;
	result_count int;
BEGIN
	PERFORM set_config('turbohybrid.payload_entry_seeding', 'auto', true);
	PERFORM set_config('turbohybrid.payload_entry_seed_count', '6', true);

	SELECT count(*) INTO result_count
	FROM (
		SELECT id
		FROM tqh_fill_band_docs
		WHERE grp = 1
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[2,0,0]'::vector,
			dense_k => 32,
			final_k => 4
		)
		LIMIT 4
	) s;

	stats := turbohybrid_last_scan_stats();
	IF result_count <> 4 OR
		stats->>'payload_entry_seeding_mode' <> 'auto' OR
		(stats->>'payload_entry_seeding_hit')::boolean IS DISTINCT FROM true OR
		(stats->>'payload_entry_seed_count')::int <= 0 OR
		(stats->>'payload_entry_seed_count')::int > 6 OR
		(stats->>'payload_entry_seed_payload_slot')::int <> 0 OR
		(stats->>'payload_entry_seed_range_count')::int <> 1099 OR
		(stats->'dense'->'traversal'->>'payload_entry_seed_count')::int <>
			(stats->>'payload_entry_seed_count')::int THEN
		RAISE EXCEPTION 'unexpected payload entry seed hit stats: %', stats;
	END IF;

	PERFORM id
	FROM tqh_fill_band_docs
	WHERE grp = 99999
	ORDER BY embedding <~> turbohybrid_query(
		vector_query => '[2,0,0]'::vector,
		dense_k => 32,
		final_k => 4
	)
	LIMIT 4;

	stats := turbohybrid_last_scan_stats();
	IF stats->>'payload_entry_seeding_mode' <> 'auto' OR
		(stats->>'payload_entry_seeding_hit')::boolean IS DISTINCT FROM false OR
		(stats->>'payload_entry_seed_count')::int <> 0 OR
		(stats->>'payload_entry_seed_payload_slot')::int <> 0 OR
		(stats->>'payload_entry_seed_range_count')::int <> 0 THEN
		RAISE EXCEPTION 'unexpected payload entry seed miss stats: %', stats;
	END IF;
END
$$;

DROP TABLE tqh_fill_band_docs;

CREATE TABLE tqh_final_diversity_docs (
	id int,
	group_id int,
	embedding vector(3)
);

INSERT INTO tqh_final_diversity_docs VALUES
	(1, 1, '[1,0,0]'),
	(2, 1, '[0.99,0.01,0]'),
	(3, 1, '[0.98,0.02,0]'),
	(4, 2, '[0.97,0.03,0]'),
	(5, 3, '[0.96,0.04,0]'),
	(6, 4, '[0.95,0.05,0]'),
	(7, 5, '[0.70,0.30,0]'),
	(8, 6, '[0.60,0.40,0]');

CREATE INDEX tqh_final_diversity_docs_idx ON tqh_final_diversity_docs
USING turbohybrid (embedding vector_cosine_turbohybrid_ops) INCLUDE (group_id)
WITH (quantization_bits = 4, exact_storage = off, graph_ef_search = 32,
	  graph_oversampling = 4);

ANALYZE tqh_final_diversity_docs;

DO $$
DECLARE
	off_ids int[];
	on_ids int[];
	off_duplicates int;
	on_duplicates int;
	stats jsonb;
BEGIN
	PERFORM set_config('enable_seqscan', 'off', true);
	PERFORM set_config('turbohybrid.final_diversity', 'off', true);
	PERFORM set_config('turbohybrid.final_diversity_payload_slot', '0', true);

	SELECT array_agg(id ORDER BY ord) INTO off_ids
	FROM (
		SELECT id, row_number() OVER () AS ord
		FROM (
			SELECT id
			FROM tqh_final_diversity_docs
			ORDER BY embedding <~> turbohybrid_query(
				vector_query => '[1,0,0]'::vector,
				dense_k => 8,
				final_k => 4
			)
			LIMIT 4
		) s
	) ranked;

	stats := turbohybrid_last_scan_stats();
	IF stats->>'final_diversity_mode' <> 'off' OR
		(stats->>'final_diversity_selected')::int <> 0 THEN
		RAISE EXCEPTION 'unexpected final diversity off stats: %', stats;
	END IF;

	SELECT count(*)::int - count(DISTINCT d.group_id)::int
	INTO off_duplicates
	FROM unnest(off_ids) AS r(id)
	JOIN tqh_final_diversity_docs d USING (id);

	PERFORM set_config('turbohybrid.final_diversity', 'group_payload', true);
	PERFORM set_config('turbohybrid.final_diversity_payload_slot', '0', true);
	PERFORM set_config('turbohybrid.final_diversity_lambda', '0', true);
	PERFORM set_config('turbohybrid.final_diversity_pool_multiplier', '3', true);

	SELECT array_agg(id ORDER BY ord) INTO on_ids
	FROM (
		SELECT id, row_number() OVER () AS ord
		FROM (
			SELECT id
			FROM tqh_final_diversity_docs
			ORDER BY embedding <~> turbohybrid_query(
				vector_query => '[1,0,0]'::vector,
				dense_k => 8,
				final_k => 4
			)
			LIMIT 4
		) s
	) ranked;

	stats := turbohybrid_last_scan_stats();
	SELECT count(*)::int - count(DISTINCT d.group_id)::int
	INTO on_duplicates
	FROM unnest(on_ids) AS r(id)
	JOIN tqh_final_diversity_docs d USING (id);

	IF off_duplicates <= 0 OR
		on_duplicates >= off_duplicates OR
		stats->>'final_diversity_mode' <> 'group_payload' OR
		(stats->>'final_diversity_payload_slot')::int <> 0 OR
		(stats->>'final_diversity_pool_size')::int < 4 OR
		(stats->>'final_diversity_selected')::int <> 4 OR
		(stats->>'final_diversity_duplicate_groups_suppressed')::int <= 0 OR
		(stats->>'final_diversity_us')::int < 0 THEN
		RAISE EXCEPTION 'unexpected final diversity result: off ids %, on ids %, stats %',
			off_ids, on_ids, stats;
	END IF;

	PERFORM set_config('turbohybrid.final_diversity', 'group_payload', true);
	PERFORM set_config('turbohybrid.final_diversity_payload_slot', '1', true);
	PERFORM id
	FROM tqh_final_diversity_docs
	ORDER BY embedding <~> turbohybrid_query(
		vector_query => '[1,0,0]'::vector,
		dense_k => 8,
		final_k => 4
	)
	LIMIT 4;

	stats := turbohybrid_last_scan_stats();
	IF stats->>'final_diversity_mode' <> 'group_payload' OR
		(stats->>'final_diversity_payload_slot')::int <> 1 OR
		(stats->>'final_diversity_selected')::int <> 0 THEN
		RAISE EXCEPTION 'unexpected final diversity invalid-slot fallback stats: %',
			stats;
	END IF;
END
$$;

DROP TABLE tqh_final_diversity_docs;

CREATE TABLE tqh_graph_repair_docs (
	id int,
	embedding vector(3)
);

INSERT INTO tqh_graph_repair_docs
SELECT g,
	('[' || (g % 17)::text || ',' || (g % 19)::text || ',' ||
		(g % 23)::text || ']')::vector
FROM generate_series(1, 96) AS g;

CREATE INDEX tqh_graph_repair_docs_idx ON tqh_graph_repair_docs
USING turbohybrid (embedding vector_cosine_turbohybrid_ops)
WITH (quantization_bits = 4, exact_storage = off, graph_ef_search = 32);

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_graph_repair_dry_run(
		'tqh_graph_repair_docs_idx'::regclass, 12, 64, 24);
	IF (stats->>'dry_run')::boolean IS DISTINCT FROM true OR
		(stats->>'writes_index_pages')::boolean IS DISTINCT FROM false OR
		(stats->>'requires_access_exclusive_lock')::boolean IS DISTINCT FROM false OR
		(stats->>'sampled_nodes')::int <= 0 OR
		(stats->>'sampled_nodes')::int > 12 OR
		(stats->>'avg_overlap')::float8 < 0 OR
		(stats->>'avg_overlap')::float8 > 1 OR
		(stats->>'weak_nodes')::int < 0 OR
		(stats->>'suggested_edges')::int < 0 OR
		(stats->'parameters'->>'sample_nodes')::int <> 12 OR
		(stats->'parameters'->>'search_ef')::int <> 64 OR
		(stats->'parameters'->>'candidate_limit')::int <> 24 THEN
		RAISE EXCEPTION 'unexpected graph repair dry-run stats: %', stats;
	END IF;
END
$$;

CREATE TABLE tqh_graph_repair_empty_docs (
	id int,
	embedding vector(3)
);

CREATE INDEX tqh_graph_repair_empty_docs_idx ON tqh_graph_repair_empty_docs
USING turbohybrid (embedding vector_cosine_turbohybrid_ops)
WITH (quantization_bits = 4, exact_storage = off);

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_graph_repair_dry_run(
		'tqh_graph_repair_empty_docs_idx'::regclass, 10, 40, 20);
	IF (stats->>'node_count')::int <> 0 OR
		(stats->>'sampled_nodes')::int <> 0 OR
		(stats->>'suggested_edges')::int <> 0 THEN
		RAISE EXCEPTION 'unexpected empty graph repair dry-run stats: %', stats;
	END IF;
END
$$;

CREATE INDEX tqh_graph_repair_docs_btree_idx ON tqh_graph_repair_docs (id);

DO $$
BEGIN
	BEGIN
		PERFORM turbohybrid_graph_repair_dry_run(
			'tqh_graph_repair_docs_btree_idx'::regclass, 10, 40, 20);
		RAISE EXCEPTION 'expected non-native graph repair dry-run error';
	EXCEPTION WHEN feature_not_supported THEN
		NULL;
	END;
END
$$;

DROP TABLE tqh_graph_repair_empty_docs;
DROP TABLE tqh_graph_repair_docs;

CREATE TABLE tqh_unmapped_filter_docs (
	id int,
	embedding vector(3)
);

INSERT INTO tqh_unmapped_filter_docs
SELECT g,
	('[' || (g % 7)::text || ',' || (g % 11)::text || ',' ||
		(g % 13)::text || ']')::vector
FROM generate_series(1, 1200) AS g;

CREATE INDEX tqh_unmapped_filter_docs_idx ON tqh_unmapped_filter_docs
USING turbohybrid (embedding vector_cosine_turbohybrid_ops)
WITH (quantization_bits = 4, exact_storage = off);

ANALYZE tqh_unmapped_filter_docs;

DO $$
DECLARE
	simple_plan json;
	filter_plan json;
	simple_cost float8;
	filter_cost float8;
BEGIN
	PERFORM set_config('enable_seqscan', 'off', true);

	EXECUTE $explain$
		EXPLAIN (FORMAT JSON, COSTS TRUE)
		SELECT id
		FROM tqh_unmapped_filter_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			dense_k => 20,
			final_k => 8
		)
		LIMIT 8
	$explain$ INTO simple_plan;

	EXECUTE $explain$
		EXPLAIN (FORMAT JSON, COSTS TRUE)
		SELECT id
		FROM tqh_unmapped_filter_docs
		WHERE id <= 1100
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			dense_k => 20,
			final_k => 8
		)
		LIMIT 8
	$explain$ INTO filter_plan;

	simple_cost := (simple_plan->0->'Plan'->>'Total Cost')::float8;
	filter_cost := (filter_plan->0->'Plan'->>'Total Cost')::float8;

	IF simple_cost <= 0 OR filter_cost <= simple_cost * 2.0 THEN
		RAISE EXCEPTION 'unmapped heap filter should cost more than simple LIMIT query: simple %, filter %, plans % / %',
			simple_cost, filter_cost, simple_plan, filter_plan;
	END IF;
END
$$;

DO $$
DECLARE
	stats jsonb;
	result_count int;
BEGIN
	PERFORM set_config('turbohybrid.warn_linear_fallback', 'off', true);
	PERFORM set_config('turbohybrid.linear_fallback_notice_threshold_ratio', '0.25', true);

	SELECT count(*) INTO result_count
	FROM (
		SELECT id
		FROM tqh_unmapped_filter_docs
		WHERE id <= 1100
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			dense_k => 20,
			final_k => 8
		)
		LIMIT 8
	) s;

	stats := turbohybrid_last_scan_stats();
	IF result_count <> 8 OR
		stats->>'graph_widening_reason' <> 'filter' OR
		(stats->>'dense_filter_unmapped')::boolean IS DISTINCT FROM true OR
		(stats->>'dense_linear_fallback_warning')::boolean IS DISTINCT FROM true OR
		(stats->>'dense_linear_fallback_ratio')::float8 < 0.25 OR
		(stats->'dense'->>'filter_unmapped')::boolean IS DISTINCT FROM true OR
		(stats->'dense'->>'linear_fallback_warning')::boolean IS DISTINCT FROM true OR
		(stats->'dense'->>'linear_fallback_ratio')::float8 < 0.25 THEN
		RAISE EXCEPTION 'unexpected unmapped filter fallback stats: %', stats;
	END IF;
END
$$;

DROP TABLE tqh_unmapped_filter_docs;

DELETE FROM tqh_segment_docs WHERE id = 6;
VACUUM tqh_segment_docs;

DO $$
DECLARE
	result_count int;
BEGIN
	SELECT count(*) INTO result_count
	FROM (
		SELECT id
		FROM tqh_segment_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			dense_k => 4,
			final_k => 4
		)
		LIMIT 4
	) s;

	IF result_count <> 4 THEN
		RAISE EXCEPTION 'segmented dense query after vacuum returned % rows',
			result_count;
	END IF;
END
$$;

DROP INDEX tqh_segment_docs_idx;

CREATE INDEX tqh_segment_docs_idx ON tqh_segment_docs
USING turbohybrid (embedding vector_cosine_turbohybrid_ops)
WITH (native_segments = 4, quantization_bits = 4, exact_storage = off);

DO $$
DECLARE
	stats jsonb;
	result_count int;
BEGIN
	stats := turbohybrid_index_stats('tqh_segment_docs_idx'::regclass);
	IF (stats->>'native_segments')::int <> 4 THEN
		RAISE EXCEPTION 'unexpected native_segments=4 index stats: %', stats;
	END IF;

	SELECT count(*) INTO result_count
	FROM (
		SELECT id
		FROM tqh_segment_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			dense_k => 4,
			final_k => 4
		)
		LIMIT 4
	) s;

	IF result_count <> 4 THEN
		RAISE EXCEPTION 'segmented native_segments=4 query returned % rows', result_count;
	END IF;

	stats := turbohybrid_last_scan_stats();
	IF (stats->>'graph_segment_count')::int <> 4 THEN
		RAISE EXCEPTION 'unexpected native_segments=4 scan stats: %', stats;
	END IF;
END
$$;

DO $$
DECLARE
	latency_stats jsonb;
	balanced_stats jsonb;
	matched_recall_stats jsonb;
	quality_stats jsonb;
	forced_fast_stats jsonb;
	forced_heuristic_stats jsonb;
	highdim_stats jsonb;
	scan_count int;
BEGIN
	EXECUTE 'CREATE TEMP TABLE tqh_profile_docs (id int, embedding vector(3)) ON COMMIT DROP';
	EXECUTE $SQL$
		INSERT INTO tqh_profile_docs VALUES
			(1, '[1,0,0]'::vector),
			(2, '[0.9,0.1,0]'::vector),
			(3, '[0,1,0]'::vector),
			(4, '[0,0,1]'::vector)
	$SQL$;

	PERFORM set_config('turbohybrid.dense_build_neighbor_select', 'auto', true);

	PERFORM set_config('turbohybrid.profile', 'latency', true);
	EXECUTE 'CREATE INDEX tqh_profile_latency_idx ON tqh_profile_docs USING turbohybrid (embedding vector_cosine_turbohybrid_ops)';
	EXECUTE 'SELECT turbohybrid_index_stats(''tqh_profile_latency_idx''::regclass)' INTO latency_stats;

	IF latency_stats->>'build_neighbor_select' <> 'heuristic' OR
		latency_stats->>'build_neighbor_select_reason' <> 'auto_lowdim' OR
		(latency_stats->>'build_fast_edges')::boolean IS DISTINCT FROM false OR
		(latency_stats->>'dense_build_exact_distances')::boolean IS DISTINCT FROM false OR
		latency_stats->>'dense_build_distance_mode' <> 'code' OR
		(latency_stats->>'graph_ef_construction')::int <> 128 OR
		(latency_stats->>'graph_ef_search')::int <> 64 OR
		(latency_stats->>'graph_oversampling')::int <> 4 THEN
		RAISE EXCEPTION 'unexpected latency profile build stats: %', latency_stats;
	END IF;

	PERFORM set_config('turbohybrid.profile', 'balanced', true);
	EXECUTE 'CREATE INDEX tqh_profile_balanced_idx ON tqh_profile_docs USING turbohybrid (embedding vector_cosine_turbohybrid_ops)';
	EXECUTE 'SELECT turbohybrid_index_stats(''tqh_profile_balanced_idx''::regclass)' INTO balanced_stats;

	IF balanced_stats->>'build_neighbor_select' <> 'heuristic' OR
		balanced_stats->>'build_neighbor_select_reason' <> 'auto_balanced' OR
		(balanced_stats->>'build_fast_edges')::boolean IS DISTINCT FROM false OR
		(balanced_stats->>'dense_build_exact_distances')::boolean IS DISTINCT FROM true OR
		balanced_stats->>'dense_build_distance_mode' <> 'exact' OR
		(balanced_stats->>'graph_ef_construction')::int <> 192 OR
		(balanced_stats->>'graph_ef_search')::int <> 96 OR
		(balanced_stats->>'graph_oversampling')::int <> 4 THEN
		RAISE EXCEPTION 'unexpected balanced profile build stats: %', balanced_stats;
	END IF;

	PERFORM set_config('turbohybrid.profile', 'matched_recall', true);
	EXECUTE 'CREATE INDEX tqh_profile_matched_recall_idx ON tqh_profile_docs USING turbohybrid (embedding vector_cosine_turbohybrid_ops)';
	EXECUTE 'SELECT turbohybrid_index_stats(''tqh_profile_matched_recall_idx''::regclass)' INTO matched_recall_stats;

	IF matched_recall_stats->>'profile' <> 'matched_recall' OR
		matched_recall_stats->>'build_neighbor_select' <> 'heuristic' OR
		matched_recall_stats->>'build_neighbor_select_reason' <> 'auto_matched_recall' OR
		(matched_recall_stats->>'build_fast_edges')::boolean IS DISTINCT FROM false OR
		(matched_recall_stats->>'dense_build_exact_distances')::boolean IS DISTINCT FROM true OR
		matched_recall_stats->>'dense_build_distance_mode' <> 'exact' OR
		(matched_recall_stats->>'graph_ef_construction')::int <> 192 OR
		(matched_recall_stats->>'graph_ef_search')::int <> 128 OR
		(matched_recall_stats->>'graph_oversampling')::int <> 8 OR
		(matched_recall_stats->>'native_segments')::int <> 1 THEN
		RAISE EXCEPTION 'unexpected matched_recall profile build stats: %', matched_recall_stats;
	END IF;

	EXECUTE 'DROP INDEX tqh_profile_latency_idx';
	EXECUTE 'DROP INDEX tqh_profile_balanced_idx';
	PERFORM set_config('enable_seqscan', 'off', true);
	EXECUTE $SQL$
		SELECT count(*) FROM (
			SELECT id
			FROM tqh_profile_docs
			ORDER BY embedding <~> turbohybrid_query(vector_query => '[1,0,0]'::vector, final_k => 2)
			LIMIT 2
		) q
	$SQL$ INTO scan_count;
	IF scan_count <> 2 THEN
		RAISE EXCEPTION 'matched_recall scan returned % rows', scan_count;
	END IF;
	EXECUTE 'SELECT turbohybrid_last_scan_stats()' INTO matched_recall_stats;
	IF (matched_recall_stats->>'index_used')::boolean IS DISTINCT FROM true OR
		matched_recall_stats->>'profile' <> 'matched_recall' OR
		(matched_recall_stats->>'dense_build_exact_distances')::boolean IS DISTINCT FROM true OR
		matched_recall_stats->>'dense_build_distance_mode' <> 'exact' OR
		matched_recall_stats->>'build_neighbor_select' <> 'heuristic' OR
		matched_recall_stats->>'build_neighbor_select_reason' <> 'auto_matched_recall' OR
		(matched_recall_stats->>'build_fast_edges')::boolean IS DISTINCT FROM false OR
		(matched_recall_stats->>'graph_ef_construction')::int <> 192 OR
		(matched_recall_stats->>'graph_ef_search')::int <> 128 OR
		(matched_recall_stats->>'graph_oversampling')::int <> 8 THEN
		RAISE EXCEPTION 'unexpected matched_recall profile scan stats: %', matched_recall_stats;
	END IF;

	EXECUTE 'CREATE INDEX tqh_profile_matched_recall_auto_segments_idx ON tqh_profile_docs USING turbohybrid (embedding vector_cosine_turbohybrid_ops) WITH (native_segments = auto)';
	EXECUTE 'SELECT turbohybrid_index_stats(''tqh_profile_matched_recall_auto_segments_idx''::regclass)' INTO matched_recall_stats;

	IF (matched_recall_stats->>'native_segments')::int <> 1 OR
		(matched_recall_stats->>'graph_ef_construction')::int <> 192 OR
		(matched_recall_stats->>'graph_ef_search')::int <> 128 OR
		(matched_recall_stats->>'graph_oversampling')::int <> 8 OR
		matched_recall_stats->>'dense_build_distance_mode' <> 'exact' OR
		(matched_recall_stats->>'build_neighbor_select') <> 'heuristic' THEN
		RAISE EXCEPTION 'matched_recall native_segments=auto should resolve to one segment: %', matched_recall_stats;
	END IF;

	PERFORM set_config('turbohybrid.profile', 'quality', true);
	EXECUTE 'CREATE INDEX tqh_profile_quality_idx ON tqh_profile_docs USING turbohybrid (embedding vector_cosine_turbohybrid_ops)';
	EXECUTE 'SELECT turbohybrid_index_stats(''tqh_profile_quality_idx''::regclass)' INTO quality_stats;

	IF quality_stats->>'build_neighbor_select' <> 'heuristic' OR
		quality_stats->>'build_neighbor_select_reason' <> 'auto_quality' OR
		(quality_stats->>'build_fast_edges')::boolean IS DISTINCT FROM false OR
		(quality_stats->>'dense_build_exact_distances')::boolean IS DISTINCT FROM true OR
		quality_stats->>'dense_build_distance_mode' <> 'exact' OR
		(quality_stats->>'graph_ef_construction')::int <> 256 OR
		(quality_stats->>'graph_ef_search')::int <> 192 OR
		(quality_stats->>'graph_oversampling')::int <> 8 THEN
		RAISE EXCEPTION 'unexpected quality profile build stats: %', quality_stats;
	END IF;

	EXECUTE 'CREATE INDEX tqh_profile_quality_auto_segments_idx ON tqh_profile_docs USING turbohybrid (embedding vector_cosine_turbohybrid_ops) WITH (native_segments = auto)';
	EXECUTE 'SELECT turbohybrid_index_stats(''tqh_profile_quality_auto_segments_idx''::regclass)' INTO quality_stats;

	IF (quality_stats->>'native_segments')::int <> 1 THEN
		RAISE EXCEPTION 'quality native_segments=auto should resolve to one segment: %', quality_stats;
	END IF;

	PERFORM set_config('turbohybrid.profile', 'latency', true);
	PERFORM set_config('turbohybrid.dense_build_distance', 'code', true);
	PERFORM set_config('turbohybrid.dense_build_neighbor_select', 'fast', true);
	EXECUTE 'CREATE INDEX tqh_profile_forced_fast_idx ON tqh_profile_docs USING turbohybrid (embedding vector_cosine_turbohybrid_ops)';
	EXECUTE 'SELECT turbohybrid_index_stats(''tqh_profile_forced_fast_idx''::regclass)' INTO forced_fast_stats;

	IF forced_fast_stats->>'build_neighbor_select' <> 'fast' OR
		forced_fast_stats->>'build_neighbor_select_reason' <> 'explicit_fast' OR
		(forced_fast_stats->>'build_fast_edges')::boolean IS DISTINCT FROM true THEN
		RAISE EXCEPTION 'unexpected forced-fast profile build stats: %', forced_fast_stats;
	END IF;

	PERFORM set_config('turbohybrid.profile', 'latency', true);
	PERFORM set_config('turbohybrid.dense_build_distance', 'code', true);
	PERFORM set_config('turbohybrid.dense_build_neighbor_select', 'heuristic', true);
	EXECUTE 'CREATE INDEX tqh_profile_forced_heuristic_idx ON tqh_profile_docs USING turbohybrid (embedding vector_cosine_turbohybrid_ops)';
	EXECUTE 'SELECT turbohybrid_index_stats(''tqh_profile_forced_heuristic_idx''::regclass)' INTO forced_heuristic_stats;

	IF forced_heuristic_stats->>'build_neighbor_select' <> 'heuristic' OR
		forced_heuristic_stats->>'build_neighbor_select_reason' <> 'explicit_heuristic' OR
		(forced_heuristic_stats->>'build_fast_edges')::boolean IS DISTINCT FROM false THEN
		RAISE EXCEPTION 'unexpected forced-heuristic profile build stats: %', forced_heuristic_stats;
	END IF;

	EXECUTE 'CREATE TEMP TABLE tqh_profile_highdim_docs (id int, embedding vector(300)) ON COMMIT DROP';
	EXECUTE $SQL$
		INSERT INTO tqh_profile_highdim_docs
		SELECT g,
			('[' || (
				SELECT string_agg(CASE WHEN d = g THEN '1' ELSE '0' END, ',' ORDER BY d)
				FROM generate_series(1, 300) AS d
			) || ']')::vector
		FROM generate_series(1, 4) AS g
	$SQL$;

	PERFORM set_config('turbohybrid.dense_build_neighbor_select', 'auto', true);
	PERFORM set_config('turbohybrid.dense_build_distance', 'auto', true);
	EXECUTE 'CREATE INDEX tqh_profile_highdim_idx ON tqh_profile_highdim_docs USING turbohybrid (embedding vector_cosine_turbohybrid_ops)';
	EXECUTE 'SELECT turbohybrid_index_stats(''tqh_profile_highdim_idx''::regclass)' INTO highdim_stats;

	IF highdim_stats->>'build_neighbor_select' <> 'fast' OR
		highdim_stats->>'build_neighbor_select_reason' <> 'auto_latency_highdim' OR
		(highdim_stats->>'build_fast_edges')::boolean IS DISTINCT FROM true THEN
		RAISE EXCEPTION 'unexpected high-dimensional latency build stats: %', highdim_stats;
	END IF;
END
$$;

DO $$
DECLARE
	stats jsonb;
	top_id int;
BEGIN
	SELECT id INTO top_id
	FROM tqh_default_docs
	ORDER BY embedding <~> turbohybrid_query(
		vector_query => '[1,0,0]'::vector,
		dense_k => 3,
		final_k => 1
	)
	LIMIT 1;

	IF top_id <> 1 THEN
		RAISE EXCEPTION 'unexpected default dense-only result: %', top_id;
	END IF;

	stats := turbohybrid_last_scan_stats();

	IF stats->>'dense_adaptive_widening_mode' <> 'off' OR
		stats->>'index_shape' <> 'hybrid' OR
		(stats->>'bm25_branch_available')::boolean IS DISTINCT FROM true OR
		(stats->>'dense_branch_used')::boolean IS DISTINCT FROM true OR
		(stats->>'bm25_branch_used')::boolean IS DISTINCT FROM false OR
		(stats->>'dense_adaptive_triggered')::boolean IS DISTINCT FROM false OR
		stats->>'dense_local_expansion_mode' <> 'off' OR
		(stats->>'dense_local_expansion_triggered')::boolean IS DISTINCT FROM false OR
		(stats->>'dense_adaptive_initial_result_target')::int <>
			(stats->>'dense_adaptive_final_result_target')::int OR
		(stats->>'dense_adaptive_initial_search_ef')::int <>
			(stats->>'dense_adaptive_final_search_ef')::int THEN
		RAISE EXCEPTION 'unexpected default dense-only scan stats: %', stats;
	END IF;
END
$$;

CREATE TABLE tqh_dense_only_docs (
	id int PRIMARY KEY,
	embedding vector(3),
	category text
);

INSERT INTO tqh_dense_only_docs VALUES
	(1, '[1,0,0]', 'keep'),
	(2, NULL, 'keep'),
	(3, '[0,1,0]', 'other');

CREATE INDEX tqh_dense_only_docs_idx ON tqh_dense_only_docs
USING turbohybrid (embedding vector_cosine_turbohybrid_ops);

DO $$
DECLARE
	ids int[];
	stats jsonb;
BEGIN
	stats := turbohybrid_index_stats('tqh_dense_only_docs_idx'::regclass);
	IF stats->>'index_shape' <> 'dense_only' OR
		(stats->>'bm25_branch_available')::boolean IS DISTINCT FROM false OR
		(stats->>'hybrid')::boolean IS DISTINCT FROM false OR
		(stats->>'bm25_document_count')::int <> 0 THEN
		RAISE EXCEPTION 'unexpected dense-only index stats: %', stats;
	END IF;

	stats := turbohybrid_last_build_stats();
	IF stats->>'available' <> 'true' OR
		stats->>'relation_name' <> 'tqh_dense_only_docs_idx' OR
		stats->>'index_shape' <> 'dense_only' OR
		(stats->>'node_count')::int <> 2 OR
		(stats->>'dimensions')::int <> 3 OR
		(stats->>'worker_count')::int < 0 OR
		NOT (stats ? 'build_edges_us') OR
		NOT (stats ? 'build_distance_calls') THEN
		RAISE EXCEPTION 'unexpected dense-only build stats: %', stats;
	END IF;

	SELECT array_agg(id) INTO ids
	FROM (
		SELECT id
		FROM tqh_dense_only_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			dense_k => 4,
			final_k => 4
		)
		LIMIT 4
	) s;

	IF ids <> ARRAY[1,3] THEN
		RAISE EXCEPTION 'unexpected dense-only build results: %', ids;
	END IF;

	stats := turbohybrid_last_scan_stats();
	IF stats->>'index_shape' <> 'dense_only' OR
		(stats->>'bm25_branch_available')::boolean IS DISTINCT FROM false OR
		(stats->>'dense_branch_used')::boolean IS DISTINCT FROM true OR
		(stats->>'bm25_branch_used')::boolean IS DISTINCT FROM false THEN
		RAISE EXCEPTION 'unexpected dense-only vector scan stats: %', stats;
	END IF;

	BEGIN
		PERFORM id
		FROM tqh_dense_only_docs
		ORDER BY embedding <~> turbohybrid_query(
			text_query => websearch_to_tsquery('simple', 'alpha'),
			bm25_k => 4,
			final_k => 4
		)
		LIMIT 1;
		RAISE EXCEPTION 'dense-only text query unexpectedly succeeded';
	EXCEPTION WHEN OTHERS THEN
		IF SQLERRM <> 'text_query requires a turbohybrid index with a tsvector key' THEN
			RAISE;
		END IF;
	END;

	BEGIN
		PERFORM id
		FROM tqh_dense_only_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			text_query => websearch_to_tsquery('simple', 'alpha'),
			dense_k => 4,
			bm25_k => 4,
			final_k => 4
		)
		LIMIT 1;
		RAISE EXCEPTION 'dense-only hybrid query unexpectedly succeeded';
	EXCEPTION WHEN OTHERS THEN
		IF SQLERRM <> 'text_query requires a turbohybrid index with a tsvector key' THEN
			RAISE;
		END IF;
	END;
END
$$;

CREATE TABLE tqh_insert_o1_empty_docs (
	id int PRIMARY KEY,
	embedding vector(3)
);

CREATE INDEX tqh_insert_o1_empty_idx ON tqh_insert_o1_empty_docs
USING turbohybrid (embedding vector_cosine_turbohybrid_ops)
WITH (quantization_bits = 4, exact_storage = off);

INSERT INTO tqh_insert_o1_empty_docs VALUES
	(1, '[1,0,0]'),
	(2, '[0,1,0]');

DO $$
DECLARE
	top_id int;
	stats jsonb;
BEGIN
	SELECT id INTO top_id
	FROM tqh_insert_o1_empty_docs
	ORDER BY embedding <~> turbohybrid_query(
		vector_query => '[0,1,0]'::vector,
		dense_k => 4,
		final_k => 1
	)
	LIMIT 1;

	IF top_id <> 2 THEN
		RAISE EXCEPTION 'exact-free inserted row was not visible: %', top_id;
	END IF;

	stats := turbohybrid_index_stats('tqh_insert_o1_empty_idx'::regclass);
	IF (stats->>'node_count')::int <> 2 OR
		(stats->>'exact_storage')::boolean IS DISTINCT FROM false THEN
		RAISE EXCEPTION 'unexpected exact-free insert stats: %', stats;
	END IF;
END
$$;

CREATE TABLE tqh_insert_o1_exact_docs (
	id int PRIMARY KEY,
	embedding vector(3)
);

INSERT INTO tqh_insert_o1_exact_docs VALUES (1, '[1,0,0]');

CREATE INDEX tqh_insert_o1_exact_idx ON tqh_insert_o1_exact_docs
USING turbohybrid (embedding vector_cosine_turbohybrid_ops)
WITH (quantization_bits = 4, exact_storage = on);

INSERT INTO tqh_insert_o1_exact_docs VALUES (2, '[0,1,0]');

DO $$
DECLARE
	top_id int;
	stats jsonb;
BEGIN
	SELECT id INTO top_id
	FROM tqh_insert_o1_exact_docs
	ORDER BY embedding <~> turbohybrid_query(
		vector_query => '[0,1,0]'::vector,
		dense_k => 4,
		final_k => 1
	)
	LIMIT 1;

	IF top_id <> 2 THEN
		RAISE EXCEPTION 'exact-storage inserted row was not visible: %', top_id;
	END IF;

	stats := turbohybrid_index_stats('tqh_insert_o1_exact_idx'::regclass);
	IF (stats->>'node_count')::int <> 2 OR
		(stats->>'exact_storage')::boolean IS DISTINCT FROM true THEN
		RAISE EXCEPTION 'unexpected exact-storage insert stats: %', stats;
	END IF;
END
$$;

CREATE TABLE tqh_insert_o1_payload_docs (
	id int PRIMARY KEY,
	grp int,
	embedding vector(3)
);

INSERT INTO tqh_insert_o1_payload_docs VALUES (1, 3, '[1,0,0]');

CREATE INDEX tqh_insert_o1_payload_idx ON tqh_insert_o1_payload_docs
USING turbohybrid (embedding vector_cosine_turbohybrid_ops) INCLUDE (grp)
WITH (
	quantization_bits = 4,
	exact_storage = off,
	residual_rerank = on,
	residual_rerank_bytes = 16
);

INSERT INTO tqh_insert_o1_payload_docs VALUES (2, 7, '[0,1,0]');

DO $$
DECLARE
	top_id int;
	stats jsonb;
BEGIN
	SELECT id INTO top_id
	FROM tqh_insert_o1_payload_docs
	WHERE grp = 7
	ORDER BY embedding <~> turbohybrid_query(
		vector_query => '[0,1,0]'::vector,
		dense_k => 4,
		final_k => 1
	)
	LIMIT 1;

	IF top_id <> 2 THEN
		RAISE EXCEPTION 'payload/residual inserted row was not visible: %', top_id;
	END IF;

	stats := turbohybrid_index_stats('tqh_insert_o1_payload_idx'::regclass);
	IF (stats->>'node_count')::int <> 2 OR
		(stats->>'payload_count')::int <> 1 OR
		(stats->>'residual_rerank_bytes')::int <> 16 THEN
		RAISE EXCEPTION 'unexpected payload/residual insert stats: %', stats;
	END IF;
END
$$;

DROP TABLE tqh_insert_o1_payload_docs;
DROP TABLE tqh_insert_o1_exact_docs;
DROP TABLE tqh_insert_o1_empty_docs;

SET max_parallel_maintenance_workers = 2;

CREATE TABLE tqh_parallel_build_docs (
	id int PRIMARY KEY,
	embedding vector(3),
	body text,
	body_tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', body)) STORED
);

INSERT INTO tqh_parallel_build_docs (id, embedding, body)
SELECT i,
	('[' ||
		((i % 7)::float8 / 7.0)::text || ',' ||
		((i % 11)::float8 / 11.0)::text || ',' ||
		((i % 13)::float8 / 13.0)::text || ']')::vector(3),
	CASE WHEN i % 5 = 0 THEN 'alpha identifier' ELSE 'beta natural text' END
FROM generate_series(1, 200) AS i;

CREATE INDEX tqh_parallel_dense_idx ON tqh_parallel_build_docs
USING turbohybrid (embedding vector_cosine_turbohybrid_ops)
WITH (quantization_bits = 4, exact_storage = off);

DO $$
DECLARE
	ids int[];
	stats jsonb;
BEGIN
	stats := turbohybrid_last_build_stats();
	IF stats->>'relation_name' <> 'tqh_parallel_dense_idx' OR
		stats->>'index_shape' <> 'dense_only' OR
		(stats->>'native_build_workers_requested')::int <> 2 OR
		(stats->>'native_build_workers_launched')::int < 1 OR
		(stats->>'parallel_fit_enabled')::boolean IS DISTINCT FROM true OR
		(stats->>'parallel_scan_enabled')::boolean IS DISTINCT FROM true OR
		(stats->>'parallel_encode_enabled')::boolean IS DISTINCT FROM true OR
		(stats->>'parallel_edge_build_enabled')::boolean IS DISTINCT FROM true OR
		stats->>'segment_build_mode' <> 'parallel_batch' OR
		(stats->>'parallel_edge_workers_launched')::int < 1 OR
		(stats->>'parallel_edge_segments')::int < 2 OR
		(stats->>'node_count')::int <> 200 OR
		jsonb_array_length(stats->'worker_scan_us') < 2 THEN
		RAISE EXCEPTION 'unexpected parallel dense build stats: %', stats;
	END IF;

	SELECT array_agg(id) INTO ids
	FROM (
		SELECT id
		FROM tqh_parallel_build_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[0.1,0.2,0.3]'::vector,
			dense_k => 20,
			final_k => 5
		)
		LIMIT 5
	) s;

	IF cardinality(ids) <> 5 THEN
		RAISE EXCEPTION 'unexpected parallel dense query results: %', ids;
	END IF;
END
$$;

CREATE INDEX tqh_parallel_hybrid_idx ON tqh_parallel_build_docs
USING turbohybrid (
	embedding vector_cosine_turbohybrid_ops,
	body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (quantization_bits = 4, exact_storage = off);

DO $$
DECLARE
	row_count int;
	stats jsonb;
BEGIN
	stats := turbohybrid_last_build_stats();
	IF stats->>'relation_name' <> 'tqh_parallel_hybrid_idx' OR
		stats->>'index_shape' <> 'hybrid' OR
		(stats->>'native_build_workers_requested')::int <> 2 OR
		(stats->>'node_count')::int <> 200 OR
		(stats->>'native_build_workers_launched')::int < 1 THEN
		RAISE EXCEPTION 'unexpected parallel hybrid build stats: %', stats;
	END IF;

	SELECT count(*) INTO row_count
	FROM (
		SELECT id
		FROM tqh_parallel_build_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[0.1,0.2,0.3]'::vector,
			text_query => to_tsquery('simple', 'alpha'),
			dense_k => 20,
			bm25_k => 20,
			final_k => 5
		)
		LIMIT 5
	) s;

	IF row_count <> 5 THEN
		RAISE EXCEPTION 'unexpected parallel hybrid query count: %', row_count;
	END IF;
END
$$;

CREATE INDEX tqh_parallel_lowbit_idx ON tqh_parallel_build_docs
USING turbohybrid (embedding vector_cosine_turbohybrid_ops)
WITH (quantization_bits = 2, exact_storage = off);

DO $$
DECLARE
	row_count int;
	stats jsonb;
BEGIN
	stats := turbohybrid_last_build_stats();
	IF stats->>'relation_name' <> 'tqh_parallel_lowbit_idx' OR
		(stats->>'quantization_bits')::int <> 2 OR
		(stats->>'node_count')::int <> 200 OR
		(stats->>'native_build_workers_launched')::int < 1 THEN
		RAISE EXCEPTION 'unexpected parallel low-bit build stats: %', stats;
	END IF;

	SELECT count(*) INTO row_count
	FROM (
		SELECT id
		FROM tqh_parallel_build_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[0.1,0.2,0.3]'::vector,
			dense_k => 20,
			final_k => 5
		)
		LIMIT 5
	) s;

	IF row_count <> 5 THEN
		RAISE EXCEPTION 'unexpected parallel low-bit query count: %', row_count;
	END IF;
END
$$;

CREATE INDEX tqh_parallel_exact_idx ON tqh_parallel_build_docs
USING turbohybrid (embedding vector_cosine_turbohybrid_ops)
WITH (quantization_bits = 4, exact_storage = on);

DO $$
DECLARE
	row_count int;
	stats jsonb;
BEGIN
	stats := turbohybrid_last_build_stats();
	IF stats->>'relation_name' <> 'tqh_parallel_exact_idx' OR
		(stats->>'exact_storage')::boolean IS DISTINCT FROM true OR
		(stats->>'native_build_workers_requested')::int <> 2 OR
		(stats->>'native_build_workers_launched')::int <> 0 OR
		(stats->>'parallel_scan_enabled')::boolean IS DISTINCT FROM false THEN
		RAISE EXCEPTION 'unexpected exact-storage serial fallback stats: %', stats;
	END IF;

	SELECT count(*) INTO row_count
	FROM (
		SELECT id
		FROM tqh_parallel_build_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[0.1,0.2,0.3]'::vector,
			dense_k => 20,
			final_k => 5
		)
		LIMIT 5
	) s;

	IF row_count <> 5 THEN
		RAISE EXCEPTION 'unexpected exact-storage fallback query count: %', row_count;
	END IF;
END
$$;

CREATE TABLE tqh_parallel_stride_docs (
	id int PRIMARY KEY,
	embedding vector(100)
);

INSERT INTO tqh_parallel_stride_docs (id, embedding)
SELECT i,
	(
		SELECT ('[' || string_agg((((i + d) % 29)::float8 / 29.0)::text, ',' ORDER BY d) || ']')::vector(100)
		FROM generate_series(1, 100) AS d
	)
FROM generate_series(1, 96) AS i;

CREATE INDEX tqh_parallel_stride_idx ON tqh_parallel_stride_docs
USING turbohybrid (embedding vector_cosine_turbohybrid_ops)
WITH (quantization_bits = 4, exact_storage = off);

DO $$
DECLARE
	row_count int;
	stats jsonb;
BEGIN
	stats := turbohybrid_last_build_stats();
	IF stats->>'relation_name' <> 'tqh_parallel_stride_idx' OR
		(stats->>'dimensions')::int <> 100 OR
		(stats->>'quantization_bits')::int <> 4 OR
		(stats->>'node_count')::int <> 96 OR
		(stats->>'native_build_workers_launched')::int < 1 THEN
		RAISE EXCEPTION 'unexpected parallel 100-dim build stats: %', stats;
	END IF;

	SELECT count(*) INTO row_count
	FROM (
		SELECT id
		FROM tqh_parallel_stride_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => (
				SELECT ('[' || string_agg('0.1', ',' ORDER BY d) || ']')::vector
				FROM generate_series(1, 100) AS d
			),
			dense_k => 20,
			final_k => 5
		)
		LIMIT 5
	) s;

	IF row_count <> 5 THEN
		RAISE EXCEPTION 'unexpected parallel 100-dim query count: %', row_count;
	END IF;
END
$$;

DROP TABLE tqh_parallel_stride_docs CASCADE;

RESET turbohybrid.native_build_workers;
RESET max_parallel_maintenance_workers;
DROP TABLE tqh_parallel_build_docs CASCADE;

INSERT INTO tqh_dense_only_docs VALUES (4, '[0.9,0.1,0]', 'keep');
INSERT INTO tqh_dense_only_docs VALUES (5, NULL, 'keep');
UPDATE tqh_dense_only_docs SET embedding = '[0,0,1]' WHERE id = 3;

DO $$
DECLARE
	ids int[];
	stats jsonb;
BEGIN
	SELECT array_agg(id) INTO ids
	FROM (
		SELECT id
		FROM tqh_dense_only_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[0,0,1]'::vector,
			dense_k => 5,
			final_k => 5
		)
		LIMIT 5
	) s;

	IF cardinality(ids) <> 3 OR ids[1] <> 3 OR 5 = ANY(ids) THEN
		RAISE EXCEPTION 'unexpected dense-only update/null-vector results: %', ids;
	END IF;

	SELECT array_agg(id) INTO ids
	FROM (
		SELECT id
		FROM tqh_dense_only_docs
		WHERE category = 'keep'
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			dense_k => 5,
			final_k => 5
		)
		LIMIT 2
	) s;

	IF ids <> ARRAY[1,4] THEN
		RAISE EXCEPTION 'unexpected dense-only filtered results: %', ids;
	END IF;

	SELECT array_agg(id) INTO ids
	FROM (
		SELECT id
		FROM tqh_dense_only_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			dense_k => 5,
			final_k => 5
		)
		LIMIT 2
	) s;

	IF ids <> ARRAY[1,4] THEN
		RAISE EXCEPTION 'unexpected dense-only LIMIT results: %', ids;
	END IF;
END
$$;

CREATE TABLE tqh_delta_docs (
	id int PRIMARY KEY,
	embedding vector(3),
	body_tsv tsvector
);

INSERT INTO tqh_delta_docs VALUES
	(1, '[1,0,0]', to_tsvector('simple', 'basealpha'));

CREATE INDEX tqh_delta_docs_idx ON tqh_delta_docs
USING turbohybrid (
	embedding vector_cosine_turbohybrid_ops,
	body_tsv bm25_tsvector_turbohybrid_ops
);

INSERT INTO tqh_delta_docs VALUES
	(2, '[0,1,0]', to_tsvector('simple', 'freshalpha')),
	(3, '[0,0,1]', NULL);

DO $$
DECLARE
	ids int[];
	stats jsonb;
BEGIN
	SELECT array_agg(id) INTO ids
	FROM (
		SELECT id
		FROM tqh_delta_docs
		ORDER BY embedding <~> turbohybrid_query(
			text_query => websearch_to_tsquery('simple', 'freshalpha'),
			dense_k => 0,
			bm25_k => 4,
			final_k => 4
		)
		LIMIT 4
	) s;

	IF ids <> ARRAY[2] THEN
		RAISE EXCEPTION 'unexpected hybrid BM25 delta results: %', ids;
	END IF;

	stats := turbohybrid_last_scan_stats();
	IF stats->>'index_shape' <> 'hybrid' OR
		(stats->>'bm25_branch_available')::boolean IS DISTINCT FROM true OR
		(stats->>'dense_branch_used')::boolean IS DISTINCT FROM false OR
		(stats->>'bm25_branch_used')::boolean IS DISTINCT FROM true THEN
		RAISE EXCEPTION 'unexpected hybrid text-only scan stats: %', stats;
	END IF;

	SELECT array_agg(id) INTO ids
	FROM (
		SELECT id
		FROM tqh_delta_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[0,1,0]'::vector,
			text_query => websearch_to_tsquery('simple', 'freshalpha'),
			dense_k => 4,
			bm25_k => 4,
			final_k => 4
		)
		LIMIT 4
	) s;

	IF ids IS NULL OR ids[1] <> 2 THEN
		RAISE EXCEPTION 'unexpected hybrid vector+text results: %', ids;
	END IF;

	stats := turbohybrid_last_scan_stats();
	IF (stats->>'dense_branch_used')::boolean IS DISTINCT FROM true OR
		(stats->>'bm25_branch_used')::boolean IS DISTINCT FROM true THEN
		RAISE EXCEPTION 'unexpected hybrid vector+text scan stats: %', stats;
	END IF;
END
$$;

CREATE TABLE tqh_fast_weighted_docs (
	id int PRIMARY KEY,
	embedding vector(3),
	body_tsv tsvector
);

INSERT INTO tqh_fast_weighted_docs VALUES
	(1, '[1,0,0]', to_tsvector('simple', 'alpha alpha dense')),
	(2, '[0.95,0.05,0]', to_tsvector('simple', 'alpha dense')),
	(3, '[0.1,0.9,0]', to_tsvector('simple', 'alpha alpha alpha lexical')),
	(4, '[0,0,1]', to_tsvector('simple', 'beta')),
	(5, '[0.8,0.2,0]', NULL),
	(6, '[0.2,0.8,0]', to_tsvector('simple', 'alpha lexical')),
	(7, '[1,0,0]', NULL);

CREATE INDEX tqh_fast_weighted_docs_idx ON tqh_fast_weighted_docs
USING turbohybrid (
	embedding vector_cosine_turbohybrid_ops,
	body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (quantization_bits = 4, exact_storage = on);

DO $$
DECLARE
	ids_pruning_on int[];
	ids_pruning_off int[];
	ids_alpha0 int[];
	ids_alpha1 int[];
	stats jsonb;
BEGIN
	PERFORM set_config('turbohybrid.bm25_strategy', 'wand', true);
	PERFORM set_config('turbohybrid.enable_wand', 'on', true);
	PERFORM set_config('turbohybrid.fast_weighted_score_bound_pruning', 'off', true);

	SELECT array_agg(id) INTO ids_pruning_off
	FROM (
		SELECT id
		FROM tqh_fast_weighted_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			text_query => websearch_to_tsquery('simple', 'alpha'),
			fusion => 'fast_weighted',
			alpha => 0.6,
			dense_k => 6,
			bm25_k => 6,
			final_k => 4
		)
		LIMIT 4
	) s;

	PERFORM set_config('turbohybrid.fast_weighted_score_bound_pruning', 'on', true);

	SELECT array_agg(id) INTO ids_pruning_on
	FROM (
		SELECT id
		FROM tqh_fast_weighted_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			text_query => websearch_to_tsquery('simple', 'alpha'),
			fusion => 'fast_weighted',
			alpha => 0.6,
			dense_k => 6,
			bm25_k => 6,
			final_k => 4
		)
		LIMIT 4
	) s;

	IF ids_pruning_on IS DISTINCT FROM ids_pruning_off THEN
		RAISE EXCEPTION 'fast_weighted pruning changed results: on %, off %',
			ids_pruning_on, ids_pruning_off;
	END IF;

	stats := turbohybrid_last_scan_stats();
	IF (stats->>'fast_weighted_enabled')::boolean IS DISTINCT FROM true OR
		(stats->>'fast_weighted_alpha')::float8 <> 0.6 OR
		stats->>'bm25_norm_mode' <> 'saturating' OR
		stats->>'dense_norm_mode' <> 'logistic' THEN
		RAISE EXCEPTION 'unexpected fast_weighted stats: %', stats;
	END IF;

	SELECT array_agg(id) INTO ids_alpha0
	FROM (
		SELECT id
		FROM tqh_fast_weighted_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			text_query => websearch_to_tsquery('simple', 'alpha'),
			fusion => 'fast_weighted',
			alpha => 0,
			dense_k => 6,
			bm25_k => 6,
			final_k => 3
		)
		LIMIT 3
	) s;

	SELECT array_agg(id) INTO ids_alpha1
	FROM (
		SELECT id
		FROM tqh_fast_weighted_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			text_query => websearch_to_tsquery('simple', 'alpha'),
			fusion => 'fast_weighted',
			alpha => 1,
			dense_k => 6,
			bm25_k => 6,
			final_k => 3
		)
		LIMIT 3
	) s;

	IF cardinality(ids_alpha0) <> 3 OR cardinality(ids_alpha1) <> 3 THEN
		RAISE EXCEPTION 'fast_weighted alpha edge case returned too few rows: alpha0 %, alpha1 %',
			ids_alpha0, ids_alpha1;
	END IF;

	PERFORM set_config('turbohybrid.fast_weighted_score_bound_pruning', 'on', true);

	PERFORM id
	FROM tqh_fast_weighted_docs
	ORDER BY embedding <~> turbohybrid_query(
		vector_query => '[1,0,0]'::vector,
		text_query => websearch_to_tsquery('simple', 'alpha'),
		fusion => 'rrf',
		dense_k => 6,
		bm25_k => 6,
		final_k => 4
	)
	LIMIT 4;

	stats := turbohybrid_last_scan_stats();
	IF (stats->>'fast_weighted_enabled')::boolean IS DISTINCT FROM false OR
		(stats->>'bm25_blocks_pruned_by_fused_score_bound')::bigint <> 0 OR
		(stats->>'bm25_candidates_pruned_by_fused_score_bound')::bigint <> 0 THEN
		RAISE EXCEPTION 'RRF used fast_weighted score-bound pruning path: %', stats;
	END IF;

	PERFORM set_config('turbohybrid.calibrated_fusion_both_match_bonus', '0.25', true);

	SELECT array_agg(id) INTO ids_alpha1
	FROM (
		SELECT id
		FROM tqh_fast_weighted_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			text_query => websearch_to_tsquery('simple', 'alpha'),
			fusion => 'calibrated',
			alpha => 1,
			dense_k => 7,
			bm25_k => 7,
			final_k => 2
		)
		LIMIT 2
	) s;

	stats := turbohybrid_last_scan_stats();
	IF ids_alpha1[1] <> 1 OR
		(stats->>'calibrated_fusion_enabled')::boolean IS DISTINCT FROM true OR
		(stats->>'calibrated_fusion_alpha_effective')::float8 <> 1 OR
		(stats->>'calibrated_fusion_both_match_bonus')::float8 <> 0.25 OR
		stats->>'calibrated_fusion_dense_norm_mode' <> 'logistic' OR
		stats->>'calibrated_fusion_bm25_norm_mode' <> 'saturating' OR
		stats->>'bm25_norm_mode' <> 'saturating' OR
		stats->>'dense_norm_mode' <> 'logistic' THEN
		RAISE EXCEPTION 'unexpected calibrated fusion results %, stats %',
			ids_alpha1, stats;
	END IF;
END
$$;

DROP TABLE tqh_fast_weighted_docs;

CREATE TABLE tqh_adaptive_budget_docs (
	id int PRIMARY KEY,
	embedding vector(3),
	body_tsv tsvector
);

INSERT INTO tqh_adaptive_budget_docs VALUES
	(1, '[1,0,0]', to_tsvector('simple', 'common shared sku123abc')),
	(2, '[0.98,0.02,0]', to_tsvector('simple', 'common shared')),
	(3, '[0.9,0.1,0]', to_tsvector('simple', 'common shared')),
	(4, '[0.8,0.2,0]', to_tsvector('simple', 'common shared')),
	(5, '[0.2,0.8,0]', to_tsvector('simple', 'common shared lexical')),
	(6, '[0.1,0.9,0]', to_tsvector('simple', 'common shared lexical')),
	(7, '[0,1,0]', to_tsvector('simple', 'common shared lexical')),
	(8, '[0,0,1]', to_tsvector('simple', 'common shared'));

CREATE INDEX tqh_adaptive_budget_docs_idx ON tqh_adaptive_budget_docs
USING turbohybrid (
	embedding vector_cosine_turbohybrid_ops,
	body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (quantization_bits = 4, exact_storage = on);

DO $$
DECLARE
	fixed_ids int[];
	adaptive_ids int[];
	overlap int;
	fixed_dense_k int;
	fixed_bm25_k int;
	adaptive_dense_k int;
	adaptive_bm25_k int;
	fixed_latency_metric_available boolean;
	adaptive_latency_metric_available boolean;
	stats jsonb;
BEGIN
	PERFORM set_config('turbohybrid.hybrid_budget_policy', 'fixed', true);

	SELECT array_agg(id) INTO fixed_ids
	FROM (
		SELECT id
		FROM tqh_adaptive_budget_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			text_query => websearch_to_tsquery('simple', 'sku123abc'),
			final_k => 3
		)
		LIMIT 3
	) s;

	stats := turbohybrid_last_scan_stats();
	fixed_dense_k := (stats->>'hybrid_dense_k_chosen')::int;
	fixed_bm25_k := (stats->>'hybrid_bm25_k_chosen')::int;
	fixed_latency_metric_available := stats ? 'elapsed_us';
	IF stats->>'hybrid_budget_policy' <> 'fixed' THEN
		RAISE EXCEPTION 'expected fixed hybrid budget policy: %', stats;
	END IF;

	PERFORM set_config('turbohybrid.hybrid_budget_policy', 'adaptive', true);

	SELECT array_agg(id) INTO adaptive_ids
	FROM (
		SELECT id
		FROM tqh_adaptive_budget_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			text_query => websearch_to_tsquery('simple', 'sku123abc'),
			final_k => 3
		)
		LIMIT 3
	) s;

	stats := turbohybrid_last_scan_stats();
	adaptive_dense_k := (stats->>'hybrid_dense_k_chosen')::int;
	adaptive_latency_metric_available := stats ? 'elapsed_us';
	IF stats->>'hybrid_budget_policy' <> 'adaptive' OR
		stats->>'hybrid_query_shape' <> 'rare_identifier' OR
		stats->>'hybrid_budget_reason' !~ '^approx_' OR
		adaptive_dense_k >= fixed_dense_k THEN
		RAISE EXCEPTION 'unexpected rare identifier adaptive budget stats: % fixed dense %',
			stats, fixed_dense_k;
	END IF;

	SELECT count(*) INTO overlap
	FROM unnest(fixed_ids) f(id)
	JOIN unnest(adaptive_ids) a(id) USING (id);
	IF overlap < 1 THEN
		RAISE EXCEPTION 'adaptive rare identifier overlap@3 unexpectedly empty: fixed %, adaptive %',
			fixed_ids, adaptive_ids;
	END IF;

	RAISE NOTICE 'adaptive_budget_benchmark shape=% overlap@3=% fixed_dense_k=% adaptive_dense_k=% fixed_bm25_k=% adaptive_bm25_k=% latency_metric_available=% ndcg=%',
		'rare_identifier', overlap, fixed_dense_k, adaptive_dense_k,
		fixed_bm25_k, (stats->>'hybrid_bm25_k_chosen')::int,
		fixed_latency_metric_available AND adaptive_latency_metric_available,
		'not_available';

	SELECT array_agg(id) INTO adaptive_ids
	FROM (
		SELECT id
		FROM tqh_adaptive_budget_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			text_query => websearch_to_tsquery('simple', 'common shared'),
			final_k => 3
		)
		LIMIT 3
	) s;

	stats := turbohybrid_last_scan_stats();
	adaptive_bm25_k := (stats->>'hybrid_bm25_k_chosen')::int;
	IF stats->>'hybrid_query_shape' <> 'broad_natural_language' OR
		adaptive_bm25_k >= fixed_bm25_k THEN
		RAISE EXCEPTION 'unexpected broad adaptive budget stats: % fixed bm25 %',
			stats, fixed_bm25_k;
	END IF;
END
$$;

DROP TABLE tqh_adaptive_budget_docs;

CREATE TABLE tqh_rrf_fusion_docs (
	id int PRIMARY KEY,
	embedding vector(3),
	body_tsv tsvector
);

INSERT INTO tqh_rrf_fusion_docs
SELECT g,
	('[1,' || (g::float8 / 1000.0)::text || ',0]')::vector(3),
	to_tsvector('simple',
		CASE WHEN g BETWEEN 41 AND 120
			THEN 'gamma alpha'
			ELSE 'gamma'
		END)
FROM generate_series(1, 160) AS g;

CREATE INDEX tqh_rrf_fusion_docs_idx ON tqh_rrf_fusion_docs
USING turbohybrid (
	embedding vector_cosine_turbohybrid_ops,
	body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (quantization_bits = 4, exact_storage = on);

DO $$
DECLARE
	dense_full_ids int[];
	dense_top_ids int[];
	bm25_full_ids int[];
	bm25_top_ids int[];
	bm25_alpha_ids int[];
	reference_ids int[];
	hybrid_ids int[];
	stats jsonb;
BEGIN
	PERFORM set_config('turbohybrid.hybrid_budget_policy', 'fixed', true);

	SELECT array_agg(id) INTO dense_full_ids
	FROM (
		SELECT id
		FROM tqh_rrf_fusion_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			fusion => 'rrf',
			dense_k => 140,
			final_k => 140
		)
	) s;

	SELECT array_agg(id) INTO dense_top_ids
	FROM (
		SELECT id
		FROM tqh_rrf_fusion_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			fusion => 'rrf',
			dense_k => 140,
			final_k => 10
		)
	) s;

	stats := turbohybrid_last_scan_stats();
	IF stats->>'fusion_strategy' <> 'generation_array' OR
		(stats->>'fusion_candidates_seen')::int < 128 THEN
		RAISE EXCEPTION 'dense-only RRF did not use generation-array fusion: %', stats;
	END IF;

	IF dense_top_ids <> dense_full_ids[1:10] THEN
		RAISE EXCEPTION 'dense-only RRF top-k changed: top %, full %',
			dense_top_ids, dense_full_ids[1:10];
	END IF;

	SELECT array_agg(id) INTO bm25_full_ids
	FROM (
		SELECT id
		FROM tqh_rrf_fusion_docs
		ORDER BY embedding <~> turbohybrid_query(
			text_query => websearch_to_tsquery('simple', 'gamma'),
			fusion => 'rrf',
			bm25_k => 140,
			final_k => 140
		)
	) s;

	SELECT array_agg(id) INTO bm25_top_ids
	FROM (
		SELECT id
		FROM tqh_rrf_fusion_docs
		ORDER BY embedding <~> turbohybrid_query(
			text_query => websearch_to_tsquery('simple', 'gamma'),
			fusion => 'rrf',
			bm25_k => 140,
			final_k => 10
		)
	) s;

	stats := turbohybrid_last_scan_stats();
	IF stats->>'fusion_strategy' <> 'generation_array' OR
		(stats->>'fusion_candidates_seen')::int < 128 THEN
		RAISE EXCEPTION 'BM25-only RRF did not use generation-array fusion: %', stats;
	END IF;

	IF bm25_top_ids <> bm25_full_ids[1:10] THEN
		RAISE EXCEPTION 'BM25-only RRF top-k changed: top %, full %',
			bm25_top_ids, bm25_full_ids[1:10];
	END IF;

	SELECT array_agg(id) INTO bm25_alpha_ids
	FROM (
		SELECT id
		FROM tqh_rrf_fusion_docs
		ORDER BY embedding <~> turbohybrid_query(
			text_query => websearch_to_tsquery('simple', 'alpha'),
			fusion => 'rrf',
			bm25_k => 140,
			final_k => 140
		)
		LIMIT 140
	) s;

	WITH dense_branch AS (
		SELECT id, dense_rank
		FROM unnest(dense_full_ids) WITH ORDINALITY AS d(id, dense_rank)
	),
	bm25_branch AS (
		SELECT id, bm25_rank
		FROM unnest(bm25_alpha_ids) WITH ORDINALITY AS b(id, bm25_rank)
	),
	fused AS (
		SELECT
			coalesce(d.id, b.id) AS id,
			d.dense_rank,
			b.bm25_rank,
			((CASE WHEN d.dense_rank IS NULL THEN 0
				ELSE 1.0 / (60 + d.dense_rank) END) +
			 (CASE WHEN b.bm25_rank IS NULL THEN 0
				ELSE 1.0 / (60 + b.bm25_rank) END)) AS score
		FROM dense_branch d
		FULL JOIN bm25_branch b USING (id)
	)
	SELECT array_agg(id) INTO reference_ids
	FROM (
		SELECT id
		FROM fused
		ORDER BY score DESC,
			(dense_rank IS NOT NULL AND bm25_rank IS NOT NULL) DESC,
			coalesce(dense_rank, 2147483647),
			coalesce(bm25_rank, 2147483647),
			id
		LIMIT 20
	) r;

	SELECT array_agg(id) INTO hybrid_ids
	FROM (
		SELECT id
		FROM tqh_rrf_fusion_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			text_query => websearch_to_tsquery('simple', 'alpha'),
			fusion => 'rrf',
			dense_k => 140,
			bm25_k => 140,
			final_k => 20
		)
	) h;

	stats := turbohybrid_last_scan_stats();
	IF stats->>'fusion_strategy' <> 'generation_array' OR
		(stats->>'fusion_duplicates')::int <= 0 OR
		(stats->>'fusion_heap_replacements')::int <= 0 THEN
		RAISE EXCEPTION 'overlap RRF did not report expected generation-array stats: %', stats;
	END IF;

	IF hybrid_ids <> reference_ids THEN
		RAISE EXCEPTION 'overlap RRF generation-array order changed: hybrid %, reference %',
			hybrid_ids, reference_ids;
	END IF;
END
$$;

DROP TABLE tqh_rrf_fusion_docs;

DROP TABLE tqh_segment_docs;
DROP TABLE tqh_dense_only_docs;
DROP TABLE tqh_delta_docs;

-- A normal vector-order SQL scan on a (vector + tsvector) turbohybrid index must
-- run the NATIVE quantized graph path (tqgraphgettuple ->
-- PgturbohybridGraphCollectResults), never the legacy graph_hnsw
-- PgturbohybridGraphSearchLayer scan path.  scan_orchestration = 'graph_native'
-- is emitted only when the native storage path records the scan; the legacy
-- full-vector element-tuple path would report 'graph_hnsw' and a flat index
-- 'flat'.  This guards against treating SearchLayer as the native hot path.
DO $$
DECLARE
	stats jsonb;
	top_id int;
BEGIN
	SELECT id INTO top_id
	FROM tqh_default_docs
	ORDER BY embedding <~> turbohybrid_query(vector_query => '[1,0,0]'::vector)
	LIMIT 1;

	stats := turbohybrid_last_scan_stats();

	IF stats->>'scan_orchestration' <> 'graph_native' THEN
		RAISE EXCEPTION 'expected native graph scan path (tqgraphgettuple/CollectResults), got scan_orchestration=%: %',
			stats->>'scan_orchestration', stats;
	END IF;

	IF stats->>'graph_storage_kind' <> 'pgturbohybrid_graph_native' THEN
		RAISE EXCEPTION 'expected native graph storage kind, got %: %',
			stats->>'graph_storage_kind', stats;
	END IF;

	IF (stats->>'graph_visited_nodes')::bigint <= 0 THEN
		RAISE EXCEPTION 'native graph scan visited no nodes: %', stats;
	END IF;
END
$$;

DROP INDEX tqh_default_docs_idx;

SET turbohybrid.dense_build_distance = exact;

CREATE INDEX tqh_default_docs_idx ON tqh_default_docs
USING turbohybrid (
	embedding vector_cosine_turbohybrid_ops,
	body_tsv bm25_tsvector_turbohybrid_ops
);

RESET turbohybrid.dense_build_distance;

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_index_stats('tqh_default_docs_idx'::regclass);

	IF (stats->>'dense_build_exact_distances')::boolean IS DISTINCT FROM true OR
		stats->>'dense_build_distance_mode' <> 'exact' OR
		(stats->>'exact_storage')::boolean IS DISTINCT FROM false THEN
		RAISE EXCEPTION 'unexpected exact-build stats: %', stats;
	END IF;
END
$$;

DO $$
DECLARE
	exact_build_size bigint;
	exact_storage_size bigint;
	stats jsonb;
BEGIN
	EXECUTE 'CREATE TEMP TABLE tqh_exact_build_size_docs (id int, embedding vector(64)) ON COMMIT DROP';
	EXECUTE $SQL$
		INSERT INTO tqh_exact_build_size_docs
		SELECT g,
			('[' || (
				SELECT string_agg((((g * 31 + d * 17) % 100)::float8 / 100.0)::text, ',' ORDER BY d)
				FROM generate_series(1, 64) AS d
			) || ']')::vector
		FROM generate_series(1, 256) AS g
	$SQL$;

	PERFORM set_config('turbohybrid.dense_build_distance', 'exact', true);
	EXECUTE 'CREATE INDEX tqh_exact_build_size_exact_idx ON tqh_exact_build_size_docs USING turbohybrid (embedding vector_cosine_turbohybrid_ops) WITH (quantization_bits = 4, exact_storage = off)';
	EXECUTE 'SELECT pg_relation_size(''tqh_exact_build_size_exact_idx''::regclass)' INTO exact_build_size;
	EXECUTE 'SELECT turbohybrid_index_stats(''tqh_exact_build_size_exact_idx''::regclass)' INTO stats;

	IF (stats->>'dense_build_exact_distances')::boolean IS DISTINCT FROM true OR
		stats->>'dense_build_distance_mode' <> 'exact' OR
		(stats->>'exact_storage')::boolean IS DISTINCT FROM false THEN
		RAISE EXCEPTION 'unexpected exact-build compact stats: %', stats;
	END IF;

	PERFORM set_config('turbohybrid.dense_build_distance', 'code', true);
	EXECUTE 'CREATE INDEX tqh_exact_build_size_storage_idx ON tqh_exact_build_size_docs USING turbohybrid (embedding vector_cosine_turbohybrid_ops) WITH (quantization_bits = 4, exact_storage = on)';
	EXECUTE 'SELECT pg_relation_size(''tqh_exact_build_size_storage_idx''::regclass)' INTO exact_storage_size;

	IF exact_build_size >= exact_storage_size THEN
		RAISE EXCEPTION 'exact-build compact index size % should be smaller than exact_storage size %',
			exact_build_size, exact_storage_size;
	END IF;

	PERFORM set_config('turbohybrid.dense_build_distance', 'auto', true);
END
$$;

DROP INDEX tqh_default_docs_idx;

-- 8-bit is an opt-in scalar-safe prototype path.  It must build, scan, insert,
-- and report scalar_8bit without using the packed 4-bit SIMD scorers.
DROP TABLE IF EXISTS tqh_quant8_docs;
CREATE TABLE tqh_quant8_docs (id int PRIMARY KEY, embedding vector(4));
INSERT INTO tqh_quant8_docs
SELECT i,
	   ARRAY[
		   ((i * 17) % 101)::real / 100.0,
		   ((i * 31) % 101)::real / 100.0,
		   ((i * 47) % 101)::real / 100.0,
		   ((i * 59) % 101)::real / 100.0
	   ]::real[]::vector
FROM generate_series(1, 64) AS i;

CREATE INDEX tqh_quant8_docs_idx ON tqh_quant8_docs
USING turbohybrid (embedding vector_cosine_turbohybrid_ops)
WITH (quantization_bits = 8, exact_storage = off, graph_ef_search = 64);
ANALYZE tqh_quant8_docs;

DO $$
DECLARE
	q vector := ARRAY[0.25, 0.50, 0.75, 1.0]::real[]::vector;
	stats jsonb;
	scorer jsonb;
	exact_ids int[];
	approx_ids int[];
	overlap_count int;
BEGIN
	scorer := turbohybrid_scorer_distances(q, q, 8);
	IF abs((scorer->>'selected')::float8 - (scorer->>'scalar_lut')::float8) > 1e-9 OR
		scorer->>'selected_kernel' <> 'scalar_8bit' THEN
		RAISE EXCEPTION 'unexpected 8-bit scorer parity stats: %', scorer;
	END IF;

	stats := turbohybrid_index_stats('tqh_quant8_docs_idx'::regclass);
	IF (stats->>'quantization_bits')::int <> 8 OR
		(stats->>'exact_storage')::boolean IS DISTINCT FROM false THEN
		RAISE EXCEPTION 'unexpected 8-bit index stats: %', stats;
	END IF;

	SELECT array_agg(id ORDER BY exact_distance, id)
	INTO exact_ids
	FROM (
		SELECT id, embedding <=> q AS exact_distance
		FROM tqh_quant8_docs
		ORDER BY embedding <=> q, id
		LIMIT 5
	) s;

	SELECT array_agg(id)
	INTO approx_ids
	FROM (
		SELECT id
		FROM tqh_quant8_docs
		ORDER BY embedding <~> turbohybrid_query(vector_query => q, dense_k => 64, final_k => 5)
		LIMIT 5
	) s;

	stats := turbohybrid_last_scan_stats();
	IF (stats->>'quantization_bits')::int <> 8 OR
		stats->>'dense_scorer' <> 'scalar_8bit' OR
		stats->>'dense_scoring_kernel' <> 'scalar_8bit' THEN
		RAISE EXCEPTION 'unexpected 8-bit scan stats: %', stats;
	END IF;

	SELECT count(*) INTO overlap_count
	FROM unnest(exact_ids) AS e(id)
	JOIN unnest(approx_ids) AS a(id) USING (id);

	IF overlap_count < 3 THEN
		RAISE EXCEPTION '8-bit approximate results too far from exact order: exact %, approx %',
			exact_ids, approx_ids;
	END IF;
END
$$;

INSERT INTO tqh_quant8_docs VALUES (65, ARRAY[0.25, 0.50, 0.75, 1.0]::real[]::vector);
VACUUM tqh_quant8_docs;

DO $$
DECLARE
	q vector := ARRAY[0.25, 0.50, 0.75, 1.0]::real[]::vector;
	rows_seen int;
	stats jsonb;
BEGIN
	SELECT count(*) INTO rows_seen
	FROM (
		SELECT id
		FROM tqh_quant8_docs
		ORDER BY embedding <~> turbohybrid_query(vector_query => q, dense_k => 64, final_k => 5)
		LIMIT 5
	) s;

	stats := turbohybrid_last_scan_stats();
	IF rows_seen <> 5 OR (stats->>'quantization_bits')::int <> 8 OR
		stats->>'dense_scorer' <> 'scalar_8bit' THEN
		RAISE EXCEPTION 'unexpected 8-bit insert/vacuum scan result rows=%, stats=%',
			rows_seen, stats;
	END IF;
END
$$;

DROP TABLE tqh_quant8_docs;

CREATE INDEX tqh_default_docs_idx ON tqh_default_docs
USING turbohybrid (
	embedding vector_cosine_turbohybrid_ops,
	body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (exact_storage = on);

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_index_stats('tqh_default_docs_idx'::regclass);

	IF (stats->>'exact_storage')::boolean IS DISTINCT FROM true OR
		(stats->>'quantization_bits')::int <> 4 THEN
		RAISE EXCEPTION 'unexpected exact-storage override stats: %', stats;
	END IF;
END
$$;

DROP INDEX tqh_default_docs_idx;

CREATE INDEX tqh_default_docs_idx ON tqh_default_docs
USING turbohybrid (
	embedding vector_cosine_turbohybrid_ops,
	body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (quantization_bits = 2);

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_index_stats('tqh_default_docs_idx'::regclass);

	IF (stats->>'quantization_bits')::int <> 2 OR
		(stats->>'exact_storage')::boolean IS DISTINCT FROM false THEN
		RAISE EXCEPTION 'unexpected quantization override stats: %', stats;
	END IF;
END
$$;

DROP INDEX tqh_default_docs_idx;

CREATE INDEX tqh_default_docs_idx ON tqh_default_docs
USING turbohybrid (
	embedding vector_cosine_turbohybrid_ops,
	body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (
	entry_sidecar = on,
	entry_sidecar_representatives = 8,
	graph_backbone = on,
	residual_rerank = on,
	residual_rerank_bytes = 16
);

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_index_stats('tqh_default_docs_idx'::regclass);

	IF (stats->>'entry_sidecar_count')::int <= 0 OR
		(stats->>'entry_sidecar_count')::int > 8 OR
		(stats->>'entry_sidecar_bytes')::int <>
			(stats->>'entry_sidecar_count')::int * 4 OR
		(stats->>'entry_sidecar_representatives_configured')::int <> 8 OR
		stats->>'entry_sidecar_strategy' <> 'hash' OR
		(stats->>'graph_backbone')::boolean IS DISTINCT FROM true OR
		(stats->>'residual_rerank_bytes')::int <> 16 OR
		(stats->>'residual_rerank_storage_bytes')::int <> 16 * 4 THEN
		RAISE EXCEPTION 'unexpected sidecar stats: %', stats;
	END IF;
END
$$;

DO $$
BEGIN
	BEGIN
		EXECUTE $cmd$
			CREATE INDEX tqh_default_docs_bad_sidecar_idx ON tqh_default_docs
			USING turbohybrid (
				embedding vector_cosine_turbohybrid_ops,
				body_tsv bm25_tsvector_turbohybrid_ops
			)
			WITH (entry_sidecar = on, entry_sidecar_strategy = 'not_a_strategy')
		$cmd$;
		RAISE EXCEPTION 'invalid entry_sidecar_strategy was accepted';
	EXCEPTION WHEN invalid_parameter_value THEN
		NULL;
	END;
END
$$;

DO $$
DECLARE
	strategy text;
	stats jsonb;
	scan_stats jsonb;
BEGIN
	FOREACH strategy IN ARRAY ARRAY[
		'hash',
		'farthest_code',
		'level_covering',
		'hybrid_level_covering'
	]
	LOOP
		DROP INDEX IF EXISTS tqh_default_docs_idx;
		EXECUTE format($cmd$
			CREATE INDEX tqh_default_docs_idx ON tqh_default_docs
			USING turbohybrid (
				embedding vector_cosine_turbohybrid_ops,
				body_tsv bm25_tsvector_turbohybrid_ops
			)
			WITH (
				quantization_bits = 4,
				exact_storage = off,
				entry_sidecar = on,
				entry_sidecar_representatives = 8,
				entry_sidecar_strategy = %L
			)
		$cmd$, strategy);

		stats := turbohybrid_index_stats('tqh_default_docs_idx'::regclass);
		IF stats->>'entry_sidecar_strategy' <> strategy OR
			(stats->>'entry_sidecar_representatives_configured')::int <> 8 OR
			(stats->>'entry_sidecar_count')::int <= 0 OR
			(stats->>'entry_sidecar_count')::int > 8 THEN
			RAISE EXCEPTION 'unexpected % sidecar index stats: %', strategy, stats;
		END IF;

		PERFORM id
		FROM tqh_default_docs
		ORDER BY embedding <~> turbohybrid_query(vector_query => '[1,0,0]'::vector,
												 dense_k => 8,
												 final_k => 2)
		LIMIT 2;

		scan_stats := turbohybrid_last_scan_stats();
		IF scan_stats->>'graph_entry_sidecar_strategy' <> strategy OR
			(scan_stats->>'graph_entry_sidecar_representatives_configured')::int <> 8 OR
			(scan_stats->>'graph_entry_sidecar_count')::int <= 0 THEN
			RAISE EXCEPTION 'unexpected % sidecar scan stats: %', strategy, scan_stats;
		END IF;
	END LOOP;
END
$$;

DROP TABLE tqh_default_docs;

DO $$
DECLARE
	opclasses text[];
BEGIN
	SELECT array_agg(opcname ORDER BY opcname) INTO opclasses
	FROM pg_opclass
	WHERE opcmethod = (SELECT oid FROM pg_am WHERE amname = 'turbohybrid');

	IF opclasses <> ARRAY[
		'bm25_tsvector_turbohybrid_ops',
		'vector_cosine_turbohybrid_ops',
		'vector_ip_turbohybrid_ops',
		'vector_l2_turbohybrid_ops'
	] THEN
		RAISE EXCEPTION 'unexpected pgturbohybrid opclasses: %', opclasses;
	END IF;
END
$$;

CREATE INDEX tqh_wrong_column_idx ON tqh_docs
USING turbohybrid (body_tsv vector_cosine_turbohybrid_ops);

CREATE INDEX tqh_wrong_opclass_idx ON tqh_docs
USING turbohybrid (embedding bm25_tsvector_turbohybrid_ops);

CREATE INDEX tqh_docs_hnsw_idx ON tqh_docs
USING hnsw (embedding vector_l2_ops);

DO $$
DECLARE
	hnsw_top_id int;
	pgturbohybrid_top_id int;
BEGIN
	SELECT id INTO hnsw_top_id
	FROM tqh_docs
	ORDER BY embedding <-> '[1,0,0]'::vector
	LIMIT 1;

	SELECT id INTO pgturbohybrid_top_id
	FROM tqh_docs
	ORDER BY embedding <~> turbohybrid_query(
		vector_query => '[1,0,0]'::vector,
		dense_k => 4,
		final_k => 1
	)
	LIMIT 1;

	IF hnsw_top_id <> 1 OR pgturbohybrid_top_id <> 1 THEN
		RAISE EXCEPTION 'unexpected coexistence results: hnsw %, pgturbohybrid %',
			hnsw_top_id, pgturbohybrid_top_id;
	END IF;
END
$$;

DROP INDEX tqh_docs_idx;

DO $$
DECLARE
	top_id int;
BEGIN
	SELECT id INTO top_id
	FROM tqh_docs
	ORDER BY embedding <-> '[1,0,0]'::vector
	LIMIT 1;

	IF top_id <> 1 THEN
		RAISE EXCEPTION 'pgvector HNSW index changed after dropping pgturbohybrid index: %', top_id;
	END IF;
END
$$;

CREATE INDEX tqh_docs_idx ON tqh_docs
USING turbohybrid (
	embedding vector_cosine_turbohybrid_ops,
	body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (quantization_bits = 4, exact_storage = on);

DROP INDEX tqh_docs_hnsw_idx;

DO $$
DECLARE
	top_id int;
BEGIN
	SELECT id INTO top_id
	FROM tqh_docs
	ORDER BY embedding <~> turbohybrid_query(
		vector_query => '[1,0,0]'::vector,
		dense_k => 4,
		final_k => 1
	)
	LIMIT 1;

	IF top_id <> 1 THEN
		RAISE EXCEPTION 'pgturbohybrid index changed after dropping pgvector HNSW index: %', top_id;
	END IF;
END
$$;

DO $$
DECLARE
	ids int[];
BEGIN
	SELECT array_agg(id) INTO ids
	FROM (
		SELECT id
		FROM tqh_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			dense_k => 4,
			final_k => 3
		)
		LIMIT 3
	) s;

	IF ids <> ARRAY[1,2,3] THEN
		RAISE EXCEPTION 'unexpected dense-only results: %', ids;
	END IF;
END
$$;

DO $$
DECLARE
	ids int[];
BEGIN
	SELECT array_agg(id) INTO ids
	FROM (
		SELECT id
		FROM tqh_docs
		ORDER BY embedding <~> turbohybrid_query(
			text_query => websearch_to_tsquery('english', 'lexical search'),
			dense_k => 0,
			bm25_k => 4,
			final_k => 2
		)
		LIMIT 2
	) s;

	IF ids <> ARRAY[3] THEN
		RAISE EXCEPTION 'unexpected BM25-only results: %', ids;
	END IF;
END
$$;

DO $$
DECLARE
	ids int[];
BEGIN
	SELECT array_agg(id) INTO ids
	FROM (
		SELECT id
		FROM tqh_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			text_query => websearch_to_tsquery('english', 'postgres search'),
			dense_k => 4,
			bm25_k => 4,
			final_k => 3
		)
		LIMIT 3
	) s;

	IF ids <> ARRAY[1,2,3] THEN
		RAISE EXCEPTION 'unexpected hybrid results: %', ids;
	END IF;
END
$$;

CREATE TABLE tqh_limit_docs (
	id int,
	embedding vector(3),
	body_tsv tsvector
);

INSERT INTO tqh_limit_docs VALUES
	(1, '[1,0,0]', to_tsvector('english', 'postgres hybrid search')),
	(2, '[1,0.1,0]', to_tsvector('english', 'postgres hybrid search')),
	(3, '[1,0.2,0]', to_tsvector('english', 'postgres hybrid search')),
	(4, '[1,0.3,0]', to_tsvector('english', 'postgres hybrid search')),
	(5, '[1,0.4,0]', to_tsvector('english', 'postgres hybrid search')),
	(6, '[1,0.5,0]', to_tsvector('english', 'postgres hybrid search')),
	(7, '[1,0.6,0]', to_tsvector('english', 'postgres hybrid search')),
	(8, '[1,0.7,0]', to_tsvector('english', 'postgres hybrid search')),
	(9, '[1,0.8,0]', to_tsvector('english', 'postgres hybrid search')),
	(10, '[1,0.9,0]', to_tsvector('english', 'postgres hybrid search')),
	(11, '[1,1,0]', to_tsvector('english', 'postgres hybrid search')),
	(12, '[1,1.1,0]', to_tsvector('english', 'postgres hybrid search'));

CREATE INDEX tqh_limit_docs_idx ON tqh_limit_docs
USING turbohybrid (
	embedding vector_cosine_turbohybrid_ops,
	body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (quantization_bits = 4, exact_storage = on);

DO $$
DECLARE
	result_count int;
	stats jsonb;
BEGIN
	SELECT count(*) INTO result_count
	FROM (
		SELECT id
		FROM tqh_limit_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector
		)
		LIMIT 10
	) s;

	stats := turbohybrid_last_scan_stats();
	IF result_count <> 10 OR
		(stats->>'final_k_requested')::int <> 0 OR
		(stats->>'final_k_effective')::int <> 10 OR
		(stats->>'detected_sql_limit')::int <> 10 OR
		(stats->>'final_k_inferred')::boolean IS DISTINCT FROM true OR
		stats->>'final_k_source' <> 'limit' THEN
		RAISE EXCEPTION 'unexpected LIMIT 10 inferred final_k: count %, stats %',
			result_count, stats;
	END IF;
END
$$;

DO $$
DECLARE
	result_count int;
	stats jsonb;
BEGIN
	SELECT count(*) INTO result_count
	FROM (
		SELECT id
		FROM tqh_limit_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector
		)
	) s;

	stats := turbohybrid_last_scan_stats();
	IF result_count <> 10 OR
		(stats->>'final_k_requested')::int <> 0 OR
		(stats->>'final_k_effective')::int <> 10 OR
		(stats->>'detected_sql_limit')::int <> 0 OR
		(stats->>'final_k_inferred')::boolean IS DISTINCT FROM false OR
		stats->>'final_k_source' <> 'default' THEN
		RAISE EXCEPTION 'unexpected dense no-LIMIT default final_k: count %, stats %',
			result_count, stats;
	END IF;
END
$$;

DO $$
DECLARE
	result_count int;
	stats jsonb;
BEGIN
	SELECT count(*) INTO result_count
	FROM (
		SELECT id
		FROM tqh_limit_docs
		ORDER BY embedding <~> turbohybrid_query(
			text_query => websearch_to_tsquery('english', 'postgres')
		)
		LIMIT 3
	) s;

	stats := turbohybrid_last_scan_stats();
	IF result_count <> 3 OR
		(stats->>'final_k_requested')::int <> 0 OR
		(stats->>'final_k_effective')::int <> 3 OR
		(stats->>'detected_sql_limit')::int <> 3 OR
		(stats->>'final_k_inferred')::boolean IS DISTINCT FROM true OR
		stats->>'final_k_source' <> 'limit' THEN
		RAISE EXCEPTION 'unexpected LIMIT 3 inferred final_k: count %, stats %',
			result_count, stats;
	END IF;
END
$$;

DO $$
DECLARE
	result_count int;
	stats jsonb;
BEGIN
	SELECT count(*) INTO result_count
	FROM (
		SELECT id
		FROM tqh_limit_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			text_query => websearch_to_tsquery('english', 'postgres')
		)
	) s;

	stats := turbohybrid_last_scan_stats();
	IF result_count <> 10 OR
		(stats->>'final_k_requested')::int <> 0 OR
		(stats->>'final_k_effective')::int <> 10 OR
		(stats->>'detected_sql_limit')::int <> 0 OR
		(stats->>'final_k_inferred')::boolean IS DISTINCT FROM false OR
		stats->>'final_k_source' <> 'default' THEN
		RAISE EXCEPTION 'unexpected no-LIMIT default final_k: count %, stats %',
			result_count, stats;
	END IF;
END
$$;

DO $$
DECLARE
	result_count int;
	stats jsonb;
BEGIN
	SELECT count(*) INTO result_count
	FROM (
		SELECT id
		FROM tqh_limit_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			text_query => websearch_to_tsquery('english', 'postgres'),
			final_k => 20
		)
		LIMIT 10
	) s;

	stats := turbohybrid_last_scan_stats();
	IF result_count <> 10 OR
		(stats->>'final_k_requested')::int <> 20 OR
		(stats->>'final_k_effective')::int <> 20 OR
		(stats->>'detected_sql_limit')::int <> 10 OR
		(stats->>'final_k_inferred')::boolean IS DISTINCT FROM false OR
		stats->>'final_k_source' <> 'explicit' THEN
		RAISE EXCEPTION 'unexpected explicit final_k with LIMIT: count %, stats %',
			result_count, stats;
	END IF;
END
$$;

DROP TABLE tqh_limit_docs;

CREATE TABLE tqh_validation_docs (
	id int,
	embedding vector(3),
	body_tsv tsvector
);

INSERT INTO tqh_validation_docs
SELECT g,
	('[' || 1 || ',' || (g::float8 / 1000.0) || ',0]')::vector,
	to_tsvector('english', 'validation overhead')
FROM generate_series(1, 150) g;

CREATE INDEX tqh_validation_docs_idx ON tqh_validation_docs
USING turbohybrid (
	embedding vector_cosine_turbohybrid_ops,
	body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (quantization_bits = 4, exact_storage = off);

SET enable_seqscan = off;

DO $$
DECLARE
	before_stats jsonb;
	after_stats jsonb;
	result_count int;
	strict_delta bigint;
	fast_delta bigint;
BEGIN
	before_stats := turbohybrid_last_scan_stats();

	SELECT count(*) INTO result_count
	FROM (
		SELECT id
		FROM tqh_validation_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			dense_k => 100,
			final_k => 100
		)
		LIMIT 100
	) s;

	after_stats := turbohybrid_last_scan_stats();
	strict_delta := (after_stats->>'strict_vector_validations')::bigint -
		(before_stats->>'strict_vector_validations')::bigint;
	fast_delta := (after_stats->>'fast_vector_checks')::bigint -
		(before_stats->>'fast_vector_checks')::bigint;

	IF result_count <> 100 THEN
		RAISE EXCEPTION 'unexpected validation guard result count: %', result_count;
	END IF;

	IF strict_delta >= 100 THEN
		RAISE EXCEPTION 'strict vector validations scaled with candidates: before %, after %, delta %',
			before_stats, after_stats, strict_delta;
	END IF;

	IF fast_delta <= 0 THEN
		RAISE EXCEPTION 'expected fast vector checks during indexed scan: before %, after %',
			before_stats, after_stats;
	END IF;
END
$$;

DROP TABLE tqh_validation_docs;

DO $$
DECLARE
	top_id int;
BEGIN
	EXECUTE 'PREPARE tqh_prepared(vector) AS
		SELECT id
		FROM tqh_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => $1,
			dense_k => 4,
			final_k => 1
		)
		LIMIT 1';
	EXECUTE 'EXECUTE tqh_prepared(''[1,0,0]''::vector)' INTO top_id;
	EXECUTE 'DEALLOCATE tqh_prepared';

	IF top_id <> 1 THEN
		RAISE EXCEPTION 'unexpected prepared statement result: %', top_id;
	END IF;
END
$$;

DO $$
DECLARE
	top_id int;
BEGIN
	WITH q AS (
		SELECT turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			dense_k => 4,
			final_k => 1
		) AS query
	)
	SELECT id INTO top_id
	FROM tqh_docs, q
	ORDER BY embedding <~> q.query
	LIMIT 1;

	IF top_id <> 1 THEN
		RAISE EXCEPTION 'unexpected CTE query result: %', top_id;
	END IF;
END
$$;

DO $$
DECLARE
	top_id int;
BEGIN
	SELECT id INTO top_id
	FROM (
		SELECT *
		FROM tqh_docs
	) d
	ORDER BY d.embedding <~> turbohybrid_query(
		vector_query => '[1,0,0]'::vector,
		dense_k => 4,
		final_k => 1
	)
	LIMIT 1;

	IF top_id <> 1 THEN
		RAISE EXCEPTION 'unexpected subquery result: %', top_id;
	END IF;
END
$$;

CREATE TEMP TABLE tqh_filter (id int PRIMARY KEY);
INSERT INTO tqh_filter VALUES (1), (3);

DO $$
DECLARE
	top_id int;
BEGIN
	SELECT d.id INTO top_id
	FROM tqh_docs d
	JOIN tqh_filter f USING (id)
	ORDER BY d.embedding <~> turbohybrid_query(
		vector_query => '[1,0,0]'::vector,
		dense_k => 4,
		final_k => 1
	)
	LIMIT 1;

	IF top_id <> 1 THEN
		RAISE EXCEPTION 'unexpected join query result: %', top_id;
	END IF;
END
$$;

DO $$
DECLARE
	distance float8;
BEGIN
	SELECT embedding <~-> turbohybrid_query(
		vector_query => '[1,0,0]'::vector
	) INTO distance
	FROM tqh_docs
	WHERE id = 1;

	IF distance <> 0 THEN
		RAISE EXCEPTION 'unexpected projected dense distance: %', distance;
	END IF;
END
$$;

DO $$
BEGIN
	PERFORM embedding <~> turbohybrid_query(
		vector_query => '[1,0,0]'::vector,
		text_query => websearch_to_tsquery('english', 'postgres')
	)
	FROM tqh_docs
	WHERE id = 1;
	RAISE EXCEPTION 'expected text-aware scalar projection rejection';
EXCEPTION WHEN feature_not_supported THEN
END
$$;

INSERT INTO tqh_docs VALUES
	(5, '[1,0,1]', to_tsvector('english', 'fresh delta term'));

DO $$
DECLARE
	top_id int;
BEGIN
	SELECT id INTO top_id
	FROM tqh_docs
	ORDER BY embedding <~> turbohybrid_query(
		text_query => websearch_to_tsquery('english', 'fresh delta'),
		dense_k => 0,
		bm25_k => 4,
		final_k => 1
	)
	LIMIT 1;

	IF top_id <> 5 THEN
		RAISE EXCEPTION 'unexpected delta BM25 result: %', top_id;
	END IF;
END
$$;

DO $$
DECLARE
	ids int[];
BEGIN
	SELECT array_agg(id) INTO ids
	FROM (
		SELECT id
		FROM tqh_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			text_query => websearch_to_tsquery('english', 'hybrid'),
			dense_k => 4,
			bm25_k => 4,
			final_k => 2,
			require_bm25_match => true
		)
		LIMIT 2
	) s;

	IF ids <> ARRAY[2] THEN
		RAISE EXCEPTION 'unexpected require_bm25_match results: %', ids;
	END IF;

	SELECT array_agg(id) INTO ids
	FROM (
		SELECT id
		FROM tqh_docs
		WHERE id IN (1, 2)
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			text_query => websearch_to_tsquery('english', 'hybrid'),
			dense_k => 4,
			bm25_k => 4,
			final_k => 4
		)
		LIMIT 2
	) s;

	IF cardinality(ids) <> 2 OR NOT ARRAY[1,2] @> ids THEN
		RAISE EXCEPTION 'unexpected hybrid filtered results: %', ids;
	END IF;
END
$$;

DO $$
BEGIN
	PERFORM turbohybrid_index_stats('tqh_docs_idx'::regclass);
	PERFORM turbohybrid_prewarm('tqh_docs_idx'::regclass);
	PERFORM turbohybrid_last_scan_stats();
	PERFORM turbohybrid_simd_capabilities();
END
$$;

CREATE TABLE tqh_empty_docs (
	id int,
	embedding vector(3),
	body_tsv tsvector
);

CREATE INDEX tqh_empty_docs_idx ON tqh_empty_docs
USING turbohybrid (
	embedding vector_l2_turbohybrid_ops,
	body_tsv bm25_tsvector_turbohybrid_ops
);

DO $$
DECLARE
	result_count int;
BEGIN
	SELECT count(*) INTO result_count
	FROM (
		SELECT id
		FROM tqh_empty_docs
		ORDER BY embedding <~-> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			dense_k => 4,
			final_k => 1
		)
		LIMIT 1
	) s;

	IF result_count <> 0 THEN
		RAISE EXCEPTION 'unexpected empty index result count: %', result_count;
	END IF;
END
$$;

DROP TABLE tqh_empty_docs;

CREATE TABLE tqh_highdim_docs (
	id int,
	embedding vector(3072),
	body_tsv tsvector
);

INSERT INTO tqh_highdim_docs
SELECT g,
	('[' || string_agg(((((g * 10) + d) % 17)::float4 / 17)::text, ',' ORDER BY d) || ']')::vector(3072),
	to_tsvector('english', 'dbpedia entity ' || g)
FROM generate_series(1, 3) AS g
CROSS JOIN generate_series(1, 3072) AS d
GROUP BY g;

CREATE INDEX tqh_highdim_docs_idx ON tqh_highdim_docs
USING turbohybrid (
	embedding vector_cosine_turbohybrid_ops,
	body_tsv bm25_tsvector_turbohybrid_ops
);

DO $$
DECLARE
	query_embedding tqh_highdim_docs.embedding%TYPE;
	top_id int;
	stats jsonb;
BEGIN
	SELECT embedding INTO query_embedding
	FROM tqh_highdim_docs
	WHERE id = 1;

	SELECT id INTO top_id
	FROM tqh_highdim_docs
	ORDER BY embedding <~> turbohybrid_query(
		vector_query => query_embedding,
		text_query => websearch_to_tsquery('english', 'dbpedia entity 1'),
		dense_k => 3,
		bm25_k => 3,
		final_k => 1
	)
	LIMIT 1;

	IF top_id <> 1 THEN
		RAISE EXCEPTION 'unexpected high-dimensional result: %', top_id;
	END IF;

	stats := turbohybrid_last_scan_stats();

	IF (stats->>'index_used')::boolean IS DISTINCT FROM true THEN
		RAISE EXCEPTION 'expected high-dimensional query to use pgturbohybrid index: %', stats;
	END IF;
END
$$;

DROP TABLE tqh_highdim_docs;

DO $$
DECLARE
	simd_on_ids int[];
	simd_off_ids int[];
BEGIN
	SET LOCAL turbohybrid.simd = on;
	SELECT array_agg(id) INTO simd_on_ids
	FROM (
		SELECT id
		FROM tqh_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			dense_k => 4,
			final_k => 3
		)
		LIMIT 3
	) s;

	SET LOCAL turbohybrid.simd = off;
	SELECT array_agg(id) INTO simd_off_ids
	FROM (
		SELECT id
		FROM tqh_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			dense_k => 4,
			final_k => 3
		)
		LIMIT 3
	) s;

	IF simd_on_ids <> simd_off_ids THEN
		RAISE EXCEPTION 'SIMD result parity failed: on %, off %',
			simd_on_ids, simd_off_ids;
	END IF;
END
$$;

DO $$
DECLARE
	funcs text[];
BEGIN
	SELECT array_agg(p.proname ORDER BY p.proname) INTO funcs
	FROM pg_proc p
	JOIN pg_depend d ON d.objid = p.oid AND d.deptype = 'e'
	JOIN pg_extension e ON e.oid = d.refobjid
	WHERE e.extname = 'pgturbohybrid'
		AND p.proname IN (
			'turbohybrid_index_stats',
			'turbohybrid_estimate_memory',
			'turbohybrid_last_build_stats',
			'turbohybrid_last_scan_diagnosis',
			'turbohybrid_last_scan_stats',
			'turbohybrid_prewarm',
			'turbohybrid_simd_capabilities'
		);

	IF funcs <> ARRAY[
		'turbohybrid_estimate_memory',
		'turbohybrid_index_stats',
		'turbohybrid_last_build_stats',
		'turbohybrid_last_scan_diagnosis',
		'turbohybrid_last_scan_stats',
		'turbohybrid_prewarm',
		'turbohybrid_simd_capabilities'
	] THEN
		RAISE EXCEPTION 'unexpected pgturbohybrid diagnostic functions: %', funcs;
	END IF;

	IF EXISTS (
		SELECT 1
		FROM pg_proc p
		JOIN pg_depend d ON d.objid = p.oid AND d.deptype = 'e'
		JOIN pg_extension e ON e.oid = d.refobjid
		WHERE e.extname = 'pgturbohybrid'
			AND p.proname LIKE '%debug%'
	) THEN
		RAISE EXCEPTION 'pgturbohybrid extension should not install debug functions';
	END IF;
END
$$;

DO $$
DECLARE
	keys text[];
BEGIN
	SELECT array_agg(key ORDER BY key) INTO keys
	FROM jsonb_object_keys(turbohybrid_index_stats('tqh_docs_idx'::regclass)) AS key;

	IF keys <> ARRAY[
		'blocks',
		'bm25_average_document_length',
		'bm25_branch_available',
		'bm25_document_count',
		'build_correction_us',
		'build_edge_us',
		'build_encode_us',
		'build_fast_edges',
		'build_neighbor_select',
		'build_neighbor_select_reason',
		'build_scan_us',
		'build_worker_count',
		'build_write_us',
		'correction_us',
		'dense_build_distance_mode',
		'dense_build_exact_distances',
		'edge_us',
		'encode_us',
		'entry_sidecar_bytes',
		'entry_sidecar_count',
		'entry_sidecar_representatives_configured',
		'entry_sidecar_strategy',
		'exact_storage',
		'graph_backbone',
		'graph_ef_construction',
		'graph_ef_search',
		'graph_m',
		'graph_oversampling',
		'hybrid',
		'index_shape',
		'native_segment_bytes',
		'native_segments',
		'profile',
		'quantization_bits',
		'residual_rerank_bytes',
		'residual_rerank_storage_bytes',
		'routing',
		'routing_entry_bytes',
		'routing_entry_count',
		'scan_us',
		'storage_kind',
		'version',
		'worker_count',
		'write_us'
	] THEN
		RAISE EXCEPTION 'unexpected pgturbohybrid index stats keys: %', keys;
	END IF;

	SELECT array_agg(key ORDER BY key) INTO keys
	FROM jsonb_object_keys(turbohybrid_last_build_stats()) AS key;

	IF keys <> ARRAY[
		'available',
		'build_code_only',
		'build_distance_calls',
		'build_distance_code_code',
		'build_distance_exact',
		'build_distance_fallback',
		'build_distance_packed',
		'build_distance_query_split',
		'build_distance_weighted',
		'build_edges_add_neighbor_us',
		'build_edges_average_nearest_count',
		'build_edges_distance_calls',
		'build_edges_entry_update_us',
		'build_edges_max_frontier_size',
		'build_edges_prune_neighbor_us',
		'build_edges_search_layer_us',
		'build_edges_select_neighbor_us',
		'build_edges_us',
		'build_fast_edges',
		'build_neighbor_select',
		'build_neighbor_select_reason',
		'connect_backbone_us',
		'dense_build_distance_mode',
		'dimensions',
		'ef_construction',
		'encode_us',
		'entry_sidecar_us',
		'exact_storage',
		'fit_correction_scan_us',
		'fit_correction_us',
		'free_exact_vectors_us',
		'index_shape',
		'm',
		'native_build_workers_launched',
		'native_build_workers_requested',
		'native_segment_bytes',
		'native_segments',
		'node_count',
		'parallel_edge_build_enabled',
		'parallel_edge_repair_us',
		'parallel_edge_segments',
		'parallel_edge_workers_launched',
		'parallel_encode_enabled',
		'parallel_fit_enabled',
		'parallel_scan_enabled',
		'parallel_segment_build_enabled',
		'quantization_bits',
		'relation_name',
		'relid',
		'reorder_nodes_us',
		'scan_us',
		'segment_build_mode',
		'total_us',
		'version',
		'wal_us',
		'worker_count',
		'worker_merge_us',
		'worker_scan_us',
		'write_pages_us'
	] THEN
		RAISE EXCEPTION 'unexpected pgturbohybrid build stats keys: %', keys;
	END IF;

	SELECT array_agg(key ORDER BY key) INTO keys
	FROM jsonb_object_keys(turbohybrid_last_scan_stats()) AS key;

	IF keys <> ARRAY[
		'adaptive_final_result_target',
		'adaptive_final_search_ef',
		'adaptive_gap_boundary',
		'adaptive_gap_top10',
		'adaptive_initial_result_target',
		'adaptive_initial_search_ef',
		'adaptive_widening_reason',
		'adaptive_widening_triggered',
		'adj_buffer_lock_wait_us',
		'auto_budget',
		'bm25',
		'bm25_accumulator_mode',
		'bm25_blocks_pruned_by_fused_score_bound',
		'bm25_branch_available',
		'bm25_branch_used',
		'bm25_cache_build_us',
		'bm25_cache_hit',
		'bm25_candidates_effective',
		'bm25_candidates_pruned_by_fused_score_bound',
		'bm25_cold_cache_o_n_work',
		'bm25_common_term_fallback',
		'bm25_docstats_bytes',
		'bm25_docstats_loaded_this_query',
		'bm25_elapsed_us',
		'bm25_heap_tsvector_rerank',
		'bm25_heap_tsvector_rerank_count',
		'bm25_heap_tsvector_rerank_fetch_us',
		'bm25_heap_tsvector_rerank_mode',
		'bm25_heap_tsvector_rerank_multiplier',
		'bm25_heap_tsvector_rerank_score_us',
		'bm25_heap_tsvector_rerank_topk_changed',
		'bm25_heap_tsvector_rerank_weight',
		'bm25_hot_postings_cache_hit',
		'bm25_hot_postings_cache_hits',
		'bm25_hot_postings_cache_mb',
		'bm25_hot_postings_cache_misses',
		'bm25_hybrid_bound',
		'bm25_impact_or_mode',
		'bm25_k_defaulted',
		'bm25_k_effective',
		'bm25_liveness_bytes',
		'bm25_liveness_loaded_this_query',
		'bm25_norm_mode',
		'bm25_postings_decode_ratio',
		'bm25_strategy',
		'bm25_wand_pruned',
		'build_fast_edges',
		'build_neighbor_select',
		'build_neighbor_select_reason',
		'calibrated_fusion_alpha_effective',
		'calibrated_fusion_bm25_norm_mode',
		'calibrated_fusion_both_match_bonus',
		'calibrated_fusion_dense_norm_mode',
		'calibrated_fusion_enabled',
		'calibrated_fusion_query_shape',
		'candidate_objects_allocated',
		'code_buffer_lock_wait_us',
		'dense',
		'dense_adaptive_final_result_target',
		'dense_adaptive_final_search_ef',
		'dense_adaptive_gap_boundary',
		'dense_adaptive_gap_top10',
		'dense_adaptive_initial_result_target',
		'dense_adaptive_initial_search_ef',
		'dense_adaptive_trigger_reason',
		'dense_adaptive_triggered',
		'dense_adaptive_widening_mode',
		'dense_batch_kernel',
		'dense_branch_used',
		'dense_build_distance_mode',
		'dense_build_exact_distances',
		'dense_candidates_effective',
			'dense_elapsed_us',
			'dense_exact_kernel',
			'dense_filter_unmapped',
			'dense_heap_rescore',
			'dense_k_defaulted',
		'dense_k_effective',
		'dense_linear_fallback_ratio',
		'dense_linear_fallback_warning',
		'dense_local_expansion_candidates_added',
		'dense_local_expansion_mode',
		'dense_local_expansion_neighbors_scored',
		'dense_local_expansion_seed_count',
		'dense_local_expansion_triggered',
		'dense_local_expansion_us',
		'dense_norm_mode',
		'dense_residual_rerank_bytes',
		'dense_residual_rerank_count',
		'dense_residual_rerank_us',
		'dense_scalar_fallback_kernel',
		'dense_scorer',
		'dense_scoring_kernel',
		'dense_simd_force',
		'dense_u8_batch_x4_enabled',
		'dense_uncertainty_final_ef',
		'dense_uncertainty_final_target',
		'dense_uncertainty_gap_boundary',
		'dense_uncertainty_gap_top10',
		'dense_uncertainty_initial_ef',
		'dense_uncertainty_initial_target',
		'dense_uncertainty_retry_mode',
		'dense_uncertainty_retry_passes',
		'dense_uncertainty_retry_reason',
		'dense_uncertainty_retry_triggered',
		'detected_sql_limit',
		'dimensions',
		'effective_search_ef_after_segment_scaling',
		'effective_search_ef_before_segment_scaling',
		'elapsed_us',
			'exact_free',
			'exact_rescore_count',
			'exact_rescore_for_bm25_only',
			'exact_rescore_source',
			'exact_storage',
		'fast_vector_checks',
		'fast_weighted_alpha',
		'fast_weighted_enabled',
		'final_diversity',
		'final_diversity_duplicate_groups_suppressed',
		'final_diversity_lambda',
		'final_diversity_mode',
		'final_diversity_payload_slot',
		'final_diversity_pool_multiplier',
		'final_diversity_pool_size',
		'final_diversity_selected',
		'final_diversity_us',
		'final_k_effective',
		'final_k_inferred',
		'final_k_requested',
		'final_k_source',
		'fusion',
		'fusion_candidates_seen',
		'fusion_duplicates',
		'fusion_elapsed_us',
		'fusion_generation_array_reset',
		'fusion_generation_array_reused',
		'fusion_heap_replacements',
		'fusion_strategy',
		'graph_adj_pages_read',
		'graph_avg_batch_size',
		'graph_base_layer',
		'graph_batch_calls',
		'graph_batch_nodes',
		'graph_batch_scored_codes',
		'graph_batch_us',
		'graph_candidate_count',
		'graph_code_arena_estimated_bytes',
		'graph_code_bytes',
		'graph_code_pages',
		'graph_code_pages_read',
		'graph_dense_budget_policy',
		'graph_dense_requested_k',
		'graph_ef_construction',
		'graph_ef_search',
		'graph_effective_rescore_band',
		'graph_effective_result_target',
		'graph_effective_search_ef',
		'graph_entry_point_count',
		'graph_entry_sidecar_count',
		'graph_entry_sidecar_representatives_configured',
		'graph_entry_sidecar_scored',
		'graph_entry_sidecar_selected',
		'graph_entry_sidecar_strategy',
		'graph_entry_sidecar_us',
		'graph_exact_cache',
		'graph_exact_cache_active',
		'graph_fill_candidate_band_calls',
		'graph_fill_candidate_band_payload_ref_count',
		'graph_fill_candidate_band_reason',
		'graph_fill_candidate_band_scored',
		'graph_fill_candidate_band_selected_after',
		'graph_fill_candidate_band_selected_before',
		'graph_fill_candidate_band_target',
		'graph_fill_candidate_band_used_payload_refs',
		'graph_fill_candidate_band_visited',
		'graph_heap_us',
		'graph_highdim_widening_multiplier',
		'graph_large_code_arena',
		'graph_m',
		'graph_oversampling',
		'graph_prepare_us',
		'graph_rescore_band',
		'graph_rescore_band_active',
		'graph_rescore_band_policy',
		'graph_rescore_count',
		'graph_rescore_pages',
		'graph_rescore_us',
		'graph_scalar_scored_codes',
		'graph_scan_lock_wait_us',
		'graph_score_kernels',
		'graph_scored_codes',
		'graph_segment_count',
		'graph_segments_searched',
		'graph_simd_scored_codes',
		'graph_sort_us',
		'graph_storage_kind',
		'graph_total_us',
		'graph_traverse_us',
		'graph_u8_batch_mode',
		'graph_visited_nodes',
			'graph_whole_code_prefetch_active',
			'graph_widening_reason',
			'heap_fetch_us',
			'heap_rescore_auto_enabled',
			'heap_rescore_count',
			'heap_rescore_reason',
			'heap_rescore_us',
			'heap_tuples_returned',
		'hybrid_bm25_k_chosen',
		'hybrid_budget_policy',
		'hybrid_budget_reason',
		'hybrid_dense_k_chosen',
		'hybrid_query_shape',
		'index_shape',
		'index_used',
		'native_cache_adj_bytes',
		'native_cache_attach_us',
		'native_cache_build_us',
		'native_cache_built_this_scan',
		'native_cache_bytes',
		'native_cache_code_bytes',
		'native_cache_exact_bytes',
		'native_cache_mode',
		'native_cache_policy',
		'native_cache_reason',
		'native_cache_refcount',
		'native_cache_reused',
		'native_cache_scope',
		'native_cache_used',
		'native_cache_wait_us',
		'native_cache_warning',
		'native_cache_warning_reason',
		'native_segments',
		'payload_entry_seed_count',
		'payload_entry_seed_payload_slot',
		'payload_entry_seed_range_count',
		'payload_entry_seed_us',
		'payload_entry_seeding_hit',
		'payload_entry_seeding_mode',
		'per_segment_budget_mode',
		'profile',
		'quantization_bits',
		'query',
		'query_split_enabled',
		'residual_rerank_active',
		'residual_rerank_band',
		'residual_rerank_max_adjustment',
		'residual_rerank_mode',
		'residual_rerank_reordered_count',
		'residual_rerank_topk_changed',
		'residual_rerank_weight_effective',
		'scan_orchestration',
		'score_mode',
		'strict_vector_validations',
		'u8_kernel_batch',
		'u8_kernel_single',
		'u8_split_enabled',
		'vector_type_cache_hits',
		'vector_type_cache_misses',
		'version'
	] THEN
		RAISE EXCEPTION 'unexpected pgturbohybrid last scan stats keys: %', keys;
	END IF;

	IF turbohybrid_last_scan_stats()->>'profile' != 'latency' THEN
		RAISE EXCEPTION 'unexpected pgturbohybrid last scan profile: %',
			turbohybrid_last_scan_stats()->>'profile';
	END IF;

	SELECT array_agg(key ORDER BY key) INTO keys
	FROM jsonb_object_keys(turbohybrid_last_scan_stats()->'bm25') AS key
	WHERE key IN (
		'cold_cache_o_n_work',
		'common_term_fallback',
		'docstats_bytes',
		'docstats_loaded_this_query',
		'heap_tsvector_rerank_count',
		'heap_tsvector_rerank_fetch_us',
		'heap_tsvector_rerank_mode',
		'heap_tsvector_rerank_score_us',
		'heap_tsvector_rerank_topk_changed',
		'liveness_bytes',
		'liveness_loaded_this_query',
		'postings_decode_ratio',
		'wand_pruned'
	);

	IF keys <> ARRAY[
		'cold_cache_o_n_work',
		'common_term_fallback',
		'docstats_bytes',
		'docstats_loaded_this_query',
		'heap_tsvector_rerank_count',
		'heap_tsvector_rerank_fetch_us',
		'heap_tsvector_rerank_mode',
		'heap_tsvector_rerank_score_us',
		'heap_tsvector_rerank_topk_changed',
		'liveness_bytes',
		'liveness_loaded_this_query',
		'postings_decode_ratio',
		'wand_pruned'
	] THEN
		RAISE EXCEPTION 'unexpected nested BM25 diagnostic keys: %', keys;
	END IF;

	SELECT array_agg(key ORDER BY key) INTO keys
	FROM jsonb_object_keys(turbohybrid_simd_capabilities()) AS key;

	IF keys <> ARRAY[
		'architecture',
		'avx512_weighted_policy',
		'compile_arm_dotprod',
		'compile_arm_i8mm',
		'compile_avx2',
		'compile_avx512_weighted',
		'compile_avx512vnni',
		'compile_avx512vpopcntdq',
		'compile_avxvnni',
		'enabled_avx512_weighted',
		'enabled_avx512vnni',
		'enabled_avx512vpopcntdq',
		'runtime_avx2',
		'runtime_avx512_weighted',
		'runtime_avx512vnni',
		'runtime_avx512vpopcntdq',
		'simd_build_disabled',
		'simd_force',
		'version'
	] THEN
		RAISE EXCEPTION 'unexpected pgturbohybrid SIMD capability keys: %', keys;
	END IF;
END
$$;

DO $$
BEGIN
	PERFORM set_config('turbohybrid.dense_build_exact_distances', 'on', true);
	IF current_setting('turbohybrid.dense_build_exact_distances') <> 'on' THEN
		RAISE EXCEPTION 'dense_build_exact_distances GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_build_distance', 'exact', true);
	IF current_setting('turbohybrid.dense_build_distance') <> 'exact' THEN
		RAISE EXCEPTION 'dense_build_distance exact GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_build_distance', 'code', true);
	IF current_setting('turbohybrid.dense_build_distance') <> 'code' THEN
		RAISE EXCEPTION 'dense_build_distance code GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_build_distance', 'auto', true);
	IF current_setting('turbohybrid.dense_build_distance') <> 'auto' THEN
		RAISE EXCEPTION 'dense_build_distance auto GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_build_neighbor_select', 'heuristic', true);
	IF current_setting('turbohybrid.dense_build_neighbor_select') <> 'heuristic' THEN
		RAISE EXCEPTION 'dense_build_neighbor_select heuristic GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_build_neighbor_select', 'fast', true);
	IF current_setting('turbohybrid.dense_build_neighbor_select') <> 'fast' THEN
		RAISE EXCEPTION 'dense_build_neighbor_select fast GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_build_neighbor_select', 'auto', true);
	IF current_setting('turbohybrid.dense_build_neighbor_select') <> 'auto' THEN
		RAISE EXCEPTION 'dense_build_neighbor_select auto GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_adaptive_widening', 'auto', true);
	IF current_setting('turbohybrid.dense_adaptive_widening') <> 'auto' THEN
		RAISE EXCEPTION 'dense_adaptive_widening auto GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_adaptive_widening', 'on', true);
	IF current_setting('turbohybrid.dense_adaptive_widening') <> 'on' THEN
		RAISE EXCEPTION 'dense_adaptive_widening on GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_adaptive_widening', 'off', true);
	IF current_setting('turbohybrid.dense_adaptive_widening') <> 'off' THEN
		RAISE EXCEPTION 'dense_adaptive_widening off GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_adaptive_widening_multiplier', '1.5', true);
	IF current_setting('turbohybrid.dense_adaptive_widening_multiplier') <> '1.5' THEN
		RAISE EXCEPTION 'dense_adaptive_widening_multiplier GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_adaptive_widening_max_multiplier', '3.0', true);
	IF current_setting('turbohybrid.dense_adaptive_widening_max_multiplier') <> '3' THEN
		RAISE EXCEPTION 'dense_adaptive_widening_max_multiplier GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_adaptive_min_gap', '0.05', true);
	IF current_setting('turbohybrid.dense_adaptive_min_gap') <> '0.05' THEN
		RAISE EXCEPTION 'dense_adaptive_min_gap GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_uncertainty_retry', 'auto', true);
	IF current_setting('turbohybrid.dense_uncertainty_retry') <> 'auto' THEN
		RAISE EXCEPTION 'dense_uncertainty_retry auto GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_uncertainty_retry', 'on', true);
	IF current_setting('turbohybrid.dense_uncertainty_retry') <> 'on' THEN
		RAISE EXCEPTION 'dense_uncertainty_retry on GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_uncertainty_retry', 'off', true);
	IF current_setting('turbohybrid.dense_uncertainty_retry') <> 'off' THEN
		RAISE EXCEPTION 'dense_uncertainty_retry off GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_uncertainty_retry_max_passes', '2', true);
	IF current_setting('turbohybrid.dense_uncertainty_retry_max_passes') <> '2' THEN
		RAISE EXCEPTION 'dense_uncertainty_retry_max_passes GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_uncertainty_retry_multiplier', '1.5', true);
	IF current_setting('turbohybrid.dense_uncertainty_retry_multiplier') <> '1.5' THEN
		RAISE EXCEPTION 'dense_uncertainty_retry_multiplier GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_uncertainty_min_gap', '0.03', true);
	IF current_setting('turbohybrid.dense_uncertainty_min_gap') <> '0.03' THEN
		RAISE EXCEPTION 'dense_uncertainty_min_gap GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_residual_rerank_mode', 'off', true);
	IF current_setting('turbohybrid.dense_residual_rerank_mode') <> 'off' THEN
		RAISE EXCEPTION 'dense_residual_rerank_mode off GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_residual_rerank_mode', 'fixed', true);
	IF current_setting('turbohybrid.dense_residual_rerank_mode') <> 'fixed' THEN
		RAISE EXCEPTION 'dense_residual_rerank_mode fixed GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_residual_rerank_mode', 'calibrated', true);
	IF current_setting('turbohybrid.dense_residual_rerank_mode') <> 'calibrated' THEN
		RAISE EXCEPTION 'dense_residual_rerank_mode calibrated GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_residual_rerank_weight', '0.5', true);
	IF current_setting('turbohybrid.dense_residual_rerank_weight') <> '0.5' THEN
		RAISE EXCEPTION 'dense_residual_rerank_weight GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_residual_rerank_max_adjust_ratio', '0.25', true);
	IF current_setting('turbohybrid.dense_residual_rerank_max_adjust_ratio') <> '0.25' THEN
		RAISE EXCEPTION 'dense_residual_rerank_max_adjust_ratio GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.warn_linear_fallback', 'off', true);
	IF current_setting('turbohybrid.warn_linear_fallback') <> 'off' THEN
		RAISE EXCEPTION 'warn_linear_fallback off GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.warn_linear_fallback', 'on', true);
	IF current_setting('turbohybrid.warn_linear_fallback') <> 'on' THEN
		RAISE EXCEPTION 'warn_linear_fallback on GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.linear_fallback_notice_threshold_ratio', '0.5', true);
	IF current_setting('turbohybrid.linear_fallback_notice_threshold_ratio') <> '0.5' THEN
		RAISE EXCEPTION 'linear_fallback_notice_threshold_ratio GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_local_expansion', 'auto', true);
	IF current_setting('turbohybrid.dense_local_expansion') <> 'auto' THEN
		RAISE EXCEPTION 'dense_local_expansion auto GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_local_expansion', 'on', true);
	IF current_setting('turbohybrid.dense_local_expansion') <> 'on' THEN
		RAISE EXCEPTION 'dense_local_expansion on GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_local_expansion', 'off', true);
	IF current_setting('turbohybrid.dense_local_expansion') <> 'off' THEN
		RAISE EXCEPTION 'dense_local_expansion off GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_local_expansion_topn', '8', true);
	IF current_setting('turbohybrid.dense_local_expansion_topn') <> '8' THEN
		RAISE EXCEPTION 'dense_local_expansion_topn GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.dense_local_expansion_max_neighbors', '128', true);
	IF current_setting('turbohybrid.dense_local_expansion_max_neighbors') <> '128' THEN
		RAISE EXCEPTION 'dense_local_expansion_max_neighbors GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.bm25_heap_tsvector_rerank', 'topk', true);
	IF current_setting('turbohybrid.bm25_heap_tsvector_rerank') <> 'topk' THEN
		RAISE EXCEPTION 'bm25_heap_tsvector_rerank topk GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.bm25_heap_tsvector_rerank', 'band', true);
	IF current_setting('turbohybrid.bm25_heap_tsvector_rerank') <> 'band' THEN
		RAISE EXCEPTION 'bm25_heap_tsvector_rerank band GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.bm25_heap_tsvector_rerank', 'auto', true);
	IF current_setting('turbohybrid.bm25_heap_tsvector_rerank') <> 'auto' THEN
		RAISE EXCEPTION 'bm25_heap_tsvector_rerank auto GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.bm25_heap_tsvector_rerank', 'off', true);
	IF current_setting('turbohybrid.bm25_heap_tsvector_rerank') <> 'off' THEN
		RAISE EXCEPTION 'bm25_heap_tsvector_rerank off GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.bm25_heap_tsvector_rerank_multiplier', '4', true);
	IF current_setting('turbohybrid.bm25_heap_tsvector_rerank_multiplier') <> '4' THEN
		RAISE EXCEPTION 'bm25_heap_tsvector_rerank_multiplier GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.bm25_heap_tsvector_rerank_weight', '0.10', true);
	IF current_setting('turbohybrid.bm25_heap_tsvector_rerank_weight') <> '0.1' THEN
		RAISE EXCEPTION 'bm25_heap_tsvector_rerank_weight GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.native_cache_policy', 'auto', true);
	IF current_setting('turbohybrid.native_cache_policy') <> 'auto' THEN
		RAISE EXCEPTION 'native_cache_policy auto GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.native_cache_policy', 'per_backend', true);
	IF current_setting('turbohybrid.native_cache_policy') <> 'per_backend' THEN
		RAISE EXCEPTION 'native_cache_policy per_backend GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.native_cache_policy', 'off', true);
	IF current_setting('turbohybrid.native_cache_policy') <> 'off' THEN
		RAISE EXCEPTION 'native_cache_policy off GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.native_cache_scope', 'shared', true);
	IF current_setting('turbohybrid.native_cache_scope') <> 'shared' THEN
		RAISE EXCEPTION 'native_cache_scope shared GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.native_cache_warn_mb', '0', true);
	IF current_setting('turbohybrid.native_cache_warn_mb') <> '0' THEN
		RAISE EXCEPTION 'native_cache_warn_mb 0 GUC did not stick';
	END IF;

	PERFORM set_config('turbohybrid.native_cache_warn_mb', '512', true);
	IF current_setting('turbohybrid.native_cache_warn_mb') <> '512MB' THEN
		RAISE EXCEPTION 'native_cache_warn_mb 512 GUC did not stick';
	END IF;
END
$$;

DO $$
DECLARE
	stats jsonb;
	row_count int;
BEGIN
	SELECT count(*) INTO row_count
	FROM (
		SELECT id
		FROM tqh_docs
		ORDER BY embedding <~> turbohybrid_query(
			text_query => to_tsquery('english', 'hybrid <-> bm25'),
			bm25_k => 4,
			final_k => 2
		)
		LIMIT 2
	) s;

	IF row_count = 0 THEN
		RAISE EXCEPTION 'expected phrase BM25 query to return rows';
	END IF;

	stats := turbohybrid_last_scan_stats();
	IF stats->>'bm25_heap_tsvector_rerank_mode' <> 'off' OR
		(stats->>'bm25_heap_tsvector_rerank_count')::int <> 0 THEN
		RAISE EXCEPTION 'default BM25 heap tsvector rerank changed behavior: %',
			stats;
	END IF;

	SET LOCAL turbohybrid.bm25_heap_tsvector_rerank = band;
	SET LOCAL turbohybrid.bm25_heap_tsvector_rerank_multiplier = 4;
	SET LOCAL turbohybrid.bm25_heap_tsvector_rerank_weight = 0.10;

	SELECT count(*) INTO row_count
	FROM (
		SELECT id
		FROM tqh_docs
		ORDER BY embedding <~> turbohybrid_query(
			text_query => to_tsquery('english', 'hybrid <-> bm25'),
			bm25_k => 4,
			final_k => 2
		)
		LIMIT 2
	) s;

	stats := turbohybrid_last_scan_stats();
	IF row_count = 0 OR
		stats->>'bm25_heap_tsvector_rerank_mode' <> 'band' OR
		(stats->>'bm25_heap_tsvector_rerank_count')::int <= 0 OR
		NOT (stats ? 'bm25_heap_tsvector_rerank_fetch_us') OR
		NOT (stats ? 'bm25_heap_tsvector_rerank_score_us') OR
		NOT (stats ? 'bm25_heap_tsvector_rerank_topk_changed') THEN
		RAISE EXCEPTION 'expected BM25 heap tsvector rerank stats: %',
			stats;
	END IF;
END
$$;

DO $$
DECLARE
	per_backend_ids int[];
	shared_ids int[];
	off_ids int[];
	stats jsonb;
BEGIN
	SET LOCAL turbohybrid.native_cache_policy = per_backend;
	SET LOCAL turbohybrid.native_cache_max_mb = 512;

	SELECT array_agg(id ORDER BY distance, id) INTO per_backend_ids
	FROM (
		SELECT id,
		       embedding <~> turbohybrid_query(
			       vector_query => '[1,0,0]'::vector,
			       dense_k => 10,
			       final_k => 5
		       ) AS distance
		FROM tqh_docs
		ORDER BY distance
		LIMIT 5
	) s;

	SET LOCAL turbohybrid.native_cache_policy = off;

	SELECT array_agg(id ORDER BY distance, id) INTO off_ids
	FROM (
		SELECT id,
		       embedding <~> turbohybrid_query(
			       vector_query => '[1,0,0]'::vector,
			       dense_k => 10,
			       final_k => 5
		       ) AS distance
		FROM tqh_docs
		ORDER BY distance
		LIMIT 5
	) s;

	IF off_ids <> per_backend_ids THEN
		RAISE EXCEPTION 'native_cache_policy=off changed dense results: off=% per_backend=%',
			off_ids, per_backend_ids;
	END IF;

	stats := turbohybrid_last_scan_stats();
	IF stats->>'native_cache_policy' <> 'off' OR
	   (stats->>'native_cache_used')::boolean OR
	   stats->>'native_cache_reason' <> 'policy_off' OR
	   NOT (stats ? 'native_cache_warning') OR
	   NOT (stats ? 'native_cache_warning_reason') THEN
		RAISE EXCEPTION 'native_cache_policy=off not reflected in stats: %', stats;
	END IF;

	SET LOCAL turbohybrid.native_cache_scope = shared;

	SELECT array_agg(id ORDER BY distance, id) INTO shared_ids
	FROM (
		SELECT id,
		       embedding <~> turbohybrid_query(
			       vector_query => '[1,0,0]'::vector,
			       dense_k => 10,
			       final_k => 5
		       ) AS distance
		FROM tqh_docs
		ORDER BY distance
		LIMIT 5
	) s;

	IF shared_ids <> per_backend_ids THEN
		RAISE EXCEPTION 'native_cache_scope=shared changed dense results: shared=% per_backend=%',
			shared_ids, per_backend_ids;
	END IF;

	stats := turbohybrid_last_scan_stats();
	IF stats->>'native_cache_policy' <> 'shared' OR
	   NOT (stats ? 'native_cache_warning') OR
	   NOT (stats ? 'native_cache_warning_reason') OR
	   NOT (
		   (stats->>'native_cache_scope' = 'shared' AND
		    (stats->>'native_cache_used')::boolean AND
		    stats->>'native_cache_reason' = 'shared_fits_max_mb') OR
		   (stats->>'native_cache_scope' = 'per_scan' AND
		    NOT (stats->>'native_cache_used')::boolean AND
		    stats->>'native_cache_reason' IN ('shared_attach_failed',
											  'shared_build_timeout',
											  'exceeds_max_mb'))
	   ) THEN
		RAISE EXCEPTION 'native_cache_scope=shared not reflected in stats: %', stats;
	END IF;
END
$$;

DO $$
DECLARE
	stats jsonb;
BEGIN
	SET LOCAL turbohybrid.native_cache_max_mb = 512;

	stats := turbohybrid_prewarm('tqh_docs_idx'::regclass);
	IF stats->>'native_cache_scope' <> 'shared' OR
	   NOT (stats ? 'native_cache_build_us') OR
	   NOT (stats ? 'native_cache_attach_us') OR
	   NOT (stats ? 'native_cache_bytes') THEN
		RAISE EXCEPTION 'turbohybrid_prewarm omitted expected shared-cache fields: %',
			stats;
	END IF;

	IF NOT (stats->>'native_cache_used')::boolean AND
	   stats->>'native_cache_reason' NOT IN ('shared_attach_failed', 'shared_build_timeout', 'exceeds_max_mb') THEN
		RAISE EXCEPTION 'turbohybrid_prewarm failed with unexpected reason: %',
			stats;
	END IF;
END
$$;

DO $$
DECLARE
	stats jsonb;
	top_id int;
BEGIN
	SET LOCAL turbohybrid.dense_adaptive_widening = on;
	SET LOCAL turbohybrid.dense_local_expansion = off;

	SELECT id INTO top_id
	FROM tqh_docs
	ORDER BY embedding <~> turbohybrid_query(
		vector_query => '[1,0,0]'::vector,
		dense_k => 3,
		final_k => 1
	)
	LIMIT 1;

	IF top_id <> 1 THEN
		RAISE EXCEPTION 'unexpected explicit adaptive dense-only result: %', top_id;
	END IF;

	stats := turbohybrid_last_scan_stats();

	IF stats->>'dense_adaptive_widening_mode' <> 'on' THEN
		RAISE EXCEPTION 'expected explicit adaptive mode in scan stats: %', stats;
	END IF;
END
$$;

DROP TABLE tqh_docs;
