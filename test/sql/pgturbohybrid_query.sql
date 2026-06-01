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

DO $$
DECLARE
	settings text[];
BEGIN
	SELECT array_agg(name ORDER BY name) INTO settings
	FROM pg_settings
	WHERE name LIKE 'turbohybrid.%';

	IF settings <> ARRAY[
		'turbohybrid.bm25_accumulator_mode',
		'turbohybrid.bm25_hot_postings_cache_mb',
		'turbohybrid.bm25_hot_postings_cache_min_df',
		'turbohybrid.bm25_hybrid_bound',
		'turbohybrid.bm25_impact_or_mode',
		'turbohybrid.bm25_strategy',
		'turbohybrid.default_bm25_k',
		'turbohybrid.default_dense_k',
		'turbohybrid.default_rrf_k',
		'turbohybrid.dense_adaptive_min_gap',
		'turbohybrid.dense_adaptive_widening',
		'turbohybrid.dense_adaptive_widening_max_multiplier',
		'turbohybrid.dense_adaptive_widening_multiplier',
		'turbohybrid.dense_build_distance',
		'turbohybrid.dense_build_exact_distances',
		'turbohybrid.dense_build_neighbor_select',
		'turbohybrid.dense_graph_avx512vnni',
			'turbohybrid.dense_graph_avxvnni',
			'turbohybrid.dense_heap_rescore',
			'turbohybrid.dense_local_expansion',
		'turbohybrid.dense_local_expansion_max_neighbors',
		'turbohybrid.dense_local_expansion_topn',
		'turbohybrid.dense_query_split_impl',
		'turbohybrid.dense_rescore_band',
		'turbohybrid.dense_u8_batch_x4',
		'turbohybrid.dense_u8_split',
		'turbohybrid.enable_wand',
		'turbohybrid.fast_weighted_score_bound_pruning',
		'turbohybrid.hybrid_budget_policy',
		'turbohybrid.max_union_candidates',
		'turbohybrid.native_build_workers',
		'turbohybrid.native_cache_max_mb',
		'turbohybrid.native_cache_policy',
		'turbohybrid.native_cache_scope',
		'turbohybrid.native_segment_budget',
		'turbohybrid.profile',
		'turbohybrid.simd'
	] THEN
		RAISE EXCEPTION 'unexpected pgturbohybrid settings: %', settings;
	END IF;

	IF current_setting('turbohybrid.profile') != 'latency' THEN
		RAISE EXCEPTION 'unexpected default turbohybrid profile: %',
			current_setting('turbohybrid.profile');
	END IF;
	IF current_setting('turbohybrid.native_build_workers') != '2' THEN
		RAISE EXCEPTION 'unexpected default native build workers: %',
			current_setting('turbohybrid.native_build_workers');
	END IF;

		IF current_setting('turbohybrid.dense_adaptive_widening') != 'off' OR
			current_setting('turbohybrid.dense_heap_rescore') != 'off' OR
			current_setting('turbohybrid.dense_local_expansion') != 'off' OR
			current_setting('turbohybrid.native_segment_budget') != 'auto' OR
			current_setting('turbohybrid.dense_build_distance') != 'auto' OR
			current_setting('turbohybrid.dense_build_exact_distances') != 'off' THEN
			RAISE EXCEPTION 'unexpected latency dense defaults: adaptive %, heap %, local %, segment budget %, build distance %, exact build %',
				current_setting('turbohybrid.dense_adaptive_widening'),
				current_setting('turbohybrid.dense_heap_rescore'),
				current_setting('turbohybrid.dense_local_expansion'),
				current_setting('turbohybrid.native_segment_budget'),
				current_setting('turbohybrid.dense_build_distance'),
				current_setting('turbohybrid.dense_build_exact_distances');
		END IF;

	IF current_setting('turbohybrid.bm25_strategy') != 'auto' OR
		current_setting('turbohybrid.bm25_impact_or_mode') != 'approx' OR
		current_setting('turbohybrid.bm25_hot_postings_cache_mb') != '32' OR
		current_setting('turbohybrid.bm25_hot_postings_cache_min_df') != '1024' OR
		current_setting('turbohybrid.bm25_hybrid_bound') != 'approx' OR
		current_setting('turbohybrid.bm25_accumulator_mode') != 'auto' THEN
		RAISE EXCEPTION 'unexpected latency BM25 defaults: strategy %, impact_or %, cache %, min_df %, bound %, accumulator %',
			current_setting('turbohybrid.bm25_strategy'),
			current_setting('turbohybrid.bm25_impact_or_mode'),
			current_setting('turbohybrid.bm25_hot_postings_cache_mb'),
			current_setting('turbohybrid.bm25_hot_postings_cache_min_df'),
			current_setting('turbohybrid.bm25_hybrid_bound'),
			current_setting('turbohybrid.bm25_accumulator_mode');
	END IF;
END
$$;

SET turbohybrid.enable_wand = off;
SET turbohybrid.bm25_strategy = wand;
SET turbohybrid.bm25_impact_or_mode = off;
SET turbohybrid.bm25_hot_postings_cache_mb = 7;
SET turbohybrid.bm25_hot_postings_cache_min_df = 2048;
SET turbohybrid.bm25_hybrid_bound = off;
SET turbohybrid.bm25_accumulator_mode = hash;
SET turbohybrid.max_union_candidates = 64;
SET turbohybrid.simd = off;
SET turbohybrid.native_build_workers = '0';
SET turbohybrid.native_build_workers = '1';
SET turbohybrid.native_build_workers = '2';
SET turbohybrid.native_build_workers = '4';
SET turbohybrid.native_build_workers = '8';
SET turbohybrid.native_build_workers = 'auto';
RESET turbohybrid.native_build_workers;
RESET turbohybrid.enable_wand;
RESET turbohybrid.bm25_strategy;
RESET turbohybrid.bm25_impact_or_mode;
RESET turbohybrid.bm25_hot_postings_cache_mb;
RESET turbohybrid.bm25_hot_postings_cache_min_df;
RESET turbohybrid.bm25_hybrid_bound;
RESET turbohybrid.bm25_accumulator_mode;
RESET turbohybrid.max_union_candidates;
RESET turbohybrid.simd;

DO $$
BEGIN
	BEGIN
		EXECUTE 'SET turbohybrid.default_dense_k = 2147483647';
		RAISE EXCEPTION 'expected default_dense_k cap error';
	EXCEPTION WHEN invalid_parameter_value THEN
	END;

	BEGIN
		EXECUTE 'SET turbohybrid.default_bm25_k = 2147483647';
		RAISE EXCEPTION 'expected default_bm25_k cap error';
	EXCEPTION WHEN invalid_parameter_value THEN
	END;

	BEGIN
		EXECUTE 'SET turbohybrid.default_rrf_k = 2147483647';
		RAISE EXCEPTION 'expected default_rrf_k cap error';
	EXCEPTION WHEN invalid_parameter_value THEN
	END;

	BEGIN
		EXECUTE 'SET turbohybrid.max_union_candidates = 2147483647';
		RAISE EXCEPTION 'expected max_union_candidates cap error';
	EXCEPTION WHEN invalid_parameter_value THEN
	END;

	BEGIN
		EXECUTE 'SET turbohybrid.bm25_hot_postings_cache_mb = 2147483647';
		RAISE EXCEPTION 'expected bm25_hot_postings_cache_mb cap error';
	EXCEPTION WHEN invalid_parameter_value THEN
	END;

	BEGIN
		EXECUTE 'SET turbohybrid.native_build_workers = ''3''';
		RAISE EXCEPTION 'expected native_build_workers value error';
	EXCEPTION WHEN invalid_parameter_value THEN
	END;
END
$$;

DO $$
BEGIN
	IF turbohybrid_query_out(turbohybrid_query(
		vector_query => '[1,0,0]'::vector,
		text_query => websearch_to_tsquery('english', 'postgres'),
		fusion => 'fast_weighted',
		alpha => 0.75,
		final_k => 5
	))::text != 'turbohybrid_query(fusion=fast_weighted,vector=true,tsquery=true,dense_weight=1,bm25_weight=1,alpha=0.75,rrf_k=60,dense_k=100,bm25_k=100,final_k=5,require_bm25_match=false)' THEN
		RAISE EXCEPTION 'unexpected fast_weighted turbohybrid_query output';
	END IF;
END
$$;

DO $$
BEGIN
	IF turbohybrid_query_out(turbohybrid_query(
		vector_query => '[1,0,0]'::vector,
		text_query => websearch_to_tsquery('english', 'postgres vector search'),
		fusion => 'rrf',
		dense_k => 12,
		bm25_k => 34,
		rrf_k => 60
	))::text != 'turbohybrid_query(fusion=rrf,vector=true,tsquery=true,dense_weight=1,bm25_weight=1,alpha=null,rrf_k=60,dense_k=12,bm25_k=34,final_k=null,require_bm25_match=false)' THEN
		RAISE EXCEPTION 'unexpected turbohybrid_query output';
	END IF;
END
$$;

DO $$
BEGIN
	IF turbohybrid_query_out(turbohybrid_query(
		text_query => websearch_to_tsquery('english', 'postgres'),
		fusion => 'weighted',
		alpha => 0.25,
		final_k => 10,
		require_bm25_match => true
	))::text != 'turbohybrid_query(fusion=weighted,vector=false,tsquery=true,dense_weight=1,bm25_weight=1,alpha=0.25,rrf_k=60,dense_k=100,bm25_k=100,final_k=10,require_bm25_match=true)' THEN
		RAISE EXCEPTION 'unexpected weighted turbohybrid_query output';
	END IF;
END
$$;

DO $$
BEGIN
	IF turbohybrid_query_out(turbohybrid_query(
		vector_query => '[1,0,0]'::vector,
		final_k => NULL
	))::text != 'turbohybrid_query(fusion=rrf,vector=true,tsquery=false,dense_weight=1,bm25_weight=1,alpha=null,rrf_k=60,dense_k=100,bm25_k=100,final_k=null,require_bm25_match=false)' THEN
		RAISE EXCEPTION 'unexpected explicit NULL final_k output';
	END IF;
END
$$;

DO $$
BEGIN
	IF turbohybrid_query_out(turbohybrid_query(
		vector_query => '[1,0,0]'::vector,
		text_query => websearch_to_tsquery('english', 'postgres')
	))::text != 'turbohybrid_query(fusion=rrf,vector=true,tsquery=true,dense_weight=1,bm25_weight=1,alpha=null,rrf_k=60,dense_k=100,bm25_k=100,final_k=null,require_bm25_match=false)' THEN
		RAISE EXCEPTION 'unexpected default hybrid query output';
	END IF;
END
$$;

SET turbohybrid.profile = balanced;

DO $$
BEGIN
	IF turbohybrid_query_out(turbohybrid_query(
		vector_query => '[1,0,0]'::vector
	))::text != 'turbohybrid_query(fusion=rrf,vector=true,tsquery=false,dense_weight=1,bm25_weight=1,alpha=null,rrf_k=60,dense_k=200,bm25_k=200,final_k=null,require_bm25_match=false)' THEN
		RAISE EXCEPTION 'unexpected balanced profile defaults';
	END IF;

	IF current_setting('turbohybrid.bm25_strategy') != 'auto' OR
		current_setting('turbohybrid.bm25_impact_or_mode') != 'exact_only' OR
		current_setting('turbohybrid.bm25_hot_postings_cache_mb') != '16' OR
		current_setting('turbohybrid.bm25_hybrid_bound') != 'safe' OR
		current_setting('turbohybrid.bm25_accumulator_mode') != 'auto' THEN
		RAISE EXCEPTION 'unexpected balanced BM25 defaults';
	END IF;
END
$$;

SET turbohybrid.profile = quality;

DO $$
BEGIN
	IF turbohybrid_query_out(turbohybrid_query(
		vector_query => '[1,0,0]'::vector
	))::text != 'turbohybrid_query(fusion=rrf,vector=true,tsquery=false,dense_weight=1,bm25_weight=1,alpha=null,rrf_k=60,dense_k=400,bm25_k=400,final_k=null,require_bm25_match=false)' THEN
		RAISE EXCEPTION 'unexpected quality profile defaults';
	END IF;

	IF current_setting('turbohybrid.bm25_strategy') != 'auto' OR
		current_setting('turbohybrid.bm25_impact_or_mode') != 'exact_only' OR
		current_setting('turbohybrid.bm25_hot_postings_cache_mb') != '16' OR
		current_setting('turbohybrid.bm25_hot_postings_cache_min_df') != '1024' OR
		current_setting('turbohybrid.bm25_hybrid_bound') != 'safe' OR
		current_setting('turbohybrid.bm25_accumulator_mode') != 'auto' OR
		current_setting('turbohybrid.enable_wand') != 'on' OR
		current_setting('turbohybrid.simd') != 'on' THEN
		RAISE EXCEPTION 'unexpected quality BM25 defaults';
	END IF;

	IF turbohybrid_last_scan_stats()->>'exact_rescore_for_bm25_only' != 'true' OR
		turbohybrid_last_scan_stats()->>'auto_budget' != 'false' THEN
		RAISE EXCEPTION 'unexpected quality internal defaults: %',
			turbohybrid_last_scan_stats();
	END IF;
END
$$;

SET turbohybrid.profile = matched_recall;

DO $$
BEGIN
	IF turbohybrid_query_out(turbohybrid_query(
		vector_query => '[1,0,0]'::vector
	))::text != 'turbohybrid_query(fusion=rrf,vector=true,tsquery=false,dense_weight=1,bm25_weight=1,alpha=null,rrf_k=60,dense_k=200,bm25_k=200,final_k=null,require_bm25_match=false)' THEN
		RAISE EXCEPTION 'unexpected matched_recall profile defaults';
	END IF;

	IF current_setting('turbohybrid.bm25_strategy') != 'auto' OR
		current_setting('turbohybrid.bm25_impact_or_mode') != 'exact_only' OR
		current_setting('turbohybrid.bm25_hot_postings_cache_mb') != '16' OR
		current_setting('turbohybrid.bm25_hybrid_bound') != 'safe' OR
		current_setting('turbohybrid.bm25_accumulator_mode') != 'auto' OR
		current_setting('turbohybrid.dense_adaptive_widening') != 'auto' OR
		current_setting('turbohybrid.dense_adaptive_widening_max_multiplier') != '1.5' THEN
		RAISE EXCEPTION 'unexpected matched_recall defaults';
	END IF;
END
$$;

SET turbohybrid.default_dense_k = 7;
SET turbohybrid.default_bm25_k = 8;
SET turbohybrid.default_rrf_k = 9;
SET turbohybrid.bm25_strategy = wand;
SET turbohybrid.bm25_impact_or_mode = off;
SET turbohybrid.bm25_hot_postings_cache_mb = 7;
SET turbohybrid.bm25_hybrid_bound = off;
SET turbohybrid.bm25_accumulator_mode = hash;
SET turbohybrid.simd = off;
SET turbohybrid.profile = quality;

DO $$
BEGIN
	IF turbohybrid_query_out(turbohybrid_query(
		vector_query => '[1,0,0]'::vector
	))::text != 'turbohybrid_query(fusion=rrf,vector=true,tsquery=false,dense_weight=1,bm25_weight=1,alpha=null,rrf_k=9,dense_k=7,bm25_k=8,final_k=null,require_bm25_match=false)' THEN
		RAISE EXCEPTION 'explicit budget GUCs did not override profile default';
	END IF;

	IF current_setting('turbohybrid.bm25_strategy') != 'wand' OR
		current_setting('turbohybrid.bm25_impact_or_mode') != 'off' OR
		current_setting('turbohybrid.bm25_hot_postings_cache_mb') != '7' OR
		current_setting('turbohybrid.bm25_hybrid_bound') != 'off' OR
		current_setting('turbohybrid.bm25_accumulator_mode') != 'hash' OR
		current_setting('turbohybrid.simd') != 'off' THEN
		RAISE EXCEPTION 'explicit BM25 GUCs did not override profile defaults';
	END IF;
END
$$;

RESET turbohybrid.default_dense_k;
RESET turbohybrid.default_bm25_k;
RESET turbohybrid.default_rrf_k;
RESET turbohybrid.bm25_strategy;
RESET turbohybrid.bm25_impact_or_mode;
RESET turbohybrid.bm25_hot_postings_cache_mb;
RESET turbohybrid.bm25_hybrid_bound;
RESET turbohybrid.bm25_accumulator_mode;
RESET turbohybrid.simd;
RESET turbohybrid.profile;

SET turbohybrid.default_dense_k = 7;
SET turbohybrid.default_bm25_k = 8;
SET turbohybrid.default_rrf_k = 9;

DO $$
BEGIN
	IF turbohybrid_query_out(turbohybrid_query(
		vector_query => '[1,0,0]'::vector,
		rrf_k => NULL,
		dense_k => NULL,
		bm25_k => NULL
	))::text != 'turbohybrid_query(fusion=rrf,vector=true,tsquery=false,dense_weight=1,bm25_weight=1,alpha=null,rrf_k=9,dense_k=7,bm25_k=8,final_k=null,require_bm25_match=false)' THEN
		RAISE EXCEPTION 'unexpected explicit NULL GUC default output';
	END IF;
END
$$;

DO $$
BEGIN
	IF turbohybrid_query_out(turbohybrid_query(
		vector_query => '[1,0,0]'::vector
	))::text != 'turbohybrid_query(fusion=rrf,vector=true,tsquery=false,dense_weight=1,bm25_weight=1,alpha=null,rrf_k=9,dense_k=7,bm25_k=8,final_k=null,require_bm25_match=false)' THEN
		RAISE EXCEPTION 'unexpected omitted GUC default output';
	END IF;
END
$$;

RESET turbohybrid.default_dense_k;
RESET turbohybrid.default_bm25_k;
RESET turbohybrid.default_rrf_k;

CREATE TEMP TABLE tq_constructor_cache_results (
	id int GENERATED ALWAYS AS IDENTITY,
	value text
);

PREPARE tq_constructor_cache AS
INSERT INTO tq_constructor_cache_results (value)
SELECT turbohybrid_query_out(turbohybrid_query(vector_query => '[1]'::vector))::text;

EXECUTE tq_constructor_cache;
SET turbohybrid.default_dense_k = 7;
EXECUTE tq_constructor_cache;

DO $$
DECLARE
	values text[];
BEGIN
	SELECT array_agg(value ORDER BY id) INTO values
	FROM tq_constructor_cache_results;

	IF values[1] NOT LIKE '%dense_k=100%' THEN
		RAISE EXCEPTION 'prepared query did not use original dense_k default: %', values[1];
	END IF;

	IF values[2] NOT LIKE '%dense_k=7%' THEN
		RAISE EXCEPTION 'prepared query reused stale dense_k default after GUC change: %', values[2];
	END IF;
END
$$;

DEALLOCATE tq_constructor_cache;
DROP TABLE tq_constructor_cache_results;
RESET turbohybrid.default_dense_k;

CREATE TABLE hq_items (id int, val vector(3));
INSERT INTO hq_items VALUES
	(1, '[1,0,0]'),
	(2, '[1,1,0]'),
	(3, '[0,1,0]');

DO $$
DECLARE
	ids int[];
BEGIN
	SELECT array_agg(id ORDER BY val <~> turbohybrid_query(vector_query => '[1,0,0]'::vector)) INTO ids
	FROM hq_items;

	IF ids != ARRAY[1, 2, 3] THEN
		RAISE EXCEPTION 'unexpected vector-only hybrid ordering: %', ids;
	END IF;
END
$$;

DO $$
BEGIN
	IF NOT (
		('[2,0,0]'::vector <~-> turbohybrid_query(vector_query => '[1,0,0]'::vector)) = 1 AND
		('[2,0,0]'::vector <~#> turbohybrid_query(vector_query => '[1,0,0]'::vector)) = -2 AND
		('[2,0,0]'::vector <~> turbohybrid_query(vector_query => '[1,0,0]'::vector)) = 0
	) THEN
		RAISE EXCEPTION 'unexpected hybrid distance operator results';
	END IF;
END
$$;

DO $$
BEGIN
	IF NOT (
		turbohybrid_vector_l2_squared_distance('[2,0,0]'::vector, '[1,0,0]'::vector) = 1 AND
		turbohybrid_vector_l2_distance('[2,0,0]'::vector, '[1,0,0]'::vector) = 1 AND
		turbohybrid_vector_negative_inner_product('[2,0,0]'::vector, '[1,0,0]'::vector) = -2 AND
		turbohybrid_vector_cosine_distance('[2,0,0]'::vector, '[1,0,0]'::vector) = 0 AND
		turbohybrid_vector_norm('[3,4,0]'::vector) = 5
	) THEN
		RAISE EXCEPTION 'unexpected pgturbohybrid vector support function results';
	END IF;
END
$$;

DO $$
BEGIN
	PERFORM turbohybrid_vector_l2_distance('[1,0]'::vector, '[1,0,0]'::vector);
	RAISE EXCEPTION 'expected vector dimension mismatch error';
EXCEPTION WHEN data_exception THEN
END
$$;

DO $$
DECLARE
	ok bool;
BEGIN
	SELECT bool_and(provolatile = 's' AND proparallel = 'r') INTO ok
	FROM pg_proc
	WHERE oid IN (
		'turbohybrid_distance(vector,turbohybrid_query)'::regprocedure,
		'turbohybrid_l2_distance(vector,turbohybrid_query)'::regprocedure,
		'turbohybrid_negative_inner_product(vector,turbohybrid_query)'::regprocedure,
		'turbohybrid_cosine_distance(vector,turbohybrid_query)'::regprocedure
	);

	IF NOT ok THEN
		RAISE EXCEPTION 'pgturbohybrid distance functions should be stable and parallel restricted';
	END IF;
END
$$;

DO $$
BEGIN
	IF NOT EXISTS (
		SELECT 1
		FROM pg_proc
		WHERE oid = 'turbohybrid_last_scan_stats()'::regprocedure
			AND proparallel = 'r'
	) THEN
		RAISE EXCEPTION 'turbohybrid_last_scan_stats should be parallel restricted';
	END IF;

	IF NOT EXISTS (
		SELECT 1
		FROM pg_proc
		WHERE oid = 'turbohybrid_last_build_stats()'::regprocedure
			AND proparallel = 'r'
	) THEN
		RAISE EXCEPTION 'turbohybrid_last_build_stats should be parallel restricted';
	END IF;
END
$$;

DO $$
DECLARE
	ok bool;
BEGIN
	SELECT bool_and(provolatile = 'i' AND proparallel = 's') INTO ok
	FROM pg_proc
	WHERE oid IN (
		'turbohybrid_vector_l2_squared_distance(vector,vector)'::regprocedure,
		'turbohybrid_vector_l2_distance(vector,vector)'::regprocedure,
		'turbohybrid_vector_negative_inner_product(vector,vector)'::regprocedure,
		'turbohybrid_vector_cosine_distance(vector,vector)'::regprocedure,
		'turbohybrid_vector_norm(vector)'::regprocedure
	);

	IF NOT ok THEN
		RAISE EXCEPTION 'pgturbohybrid vector support functions should be immutable and parallel safe';
	END IF;
END
$$;

DO $$
BEGIN
	PERFORM turbohybrid_distance('[1,0,0]'::vector, turbohybrid_query(
		text_query => websearch_to_tsquery('english', 'postgres')
	));
	RAISE EXCEPTION 'expected hybrid text fallback rejection';
EXCEPTION WHEN feature_not_supported THEN
END
$$;

DO $$
BEGIN
	PERFORM id
	FROM hq_items
	ORDER BY val <~> turbohybrid_query(
		vector_query => '[1,0,0]'::vector,
		text_query => websearch_to_tsquery('english', 'postgres')
	)
	LIMIT 1;
	RAISE EXCEPTION 'expected scalar hybrid ordering fallback rejection';
EXCEPTION WHEN feature_not_supported THEN
END
$$;

DO $$
BEGIN
	PERFORM turbohybrid_query(vector_query => '[1]'::vector, dense_k => -1);
	RAISE EXCEPTION 'expected dense_k validation error';
EXCEPTION WHEN invalid_parameter_value THEN
END
$$;

DO $$
BEGIN
	PERFORM turbohybrid_query(vector_query => '[1]'::vector, fusion => 'bogus');
	RAISE EXCEPTION 'expected fusion validation error';
EXCEPTION WHEN invalid_parameter_value THEN
END
$$;

DO $$
BEGIN
	PERFORM turbohybrid_query(vector_query => '[1]'::vector, alpha => 1.5);
	RAISE EXCEPTION 'expected alpha validation error';
EXCEPTION WHEN invalid_parameter_value THEN
END
$$;

DO $$
BEGIN
	PERFORM turbohybrid_query(vector_query => '[1]'::vector, rrf_k => 0);
	RAISE EXCEPTION 'expected rrf_k validation error';
EXCEPTION WHEN invalid_parameter_value THEN
END
$$;

DO $$
BEGIN
	PERFORM 'anything'::turbohybrid_query;
	RAISE EXCEPTION 'expected turbohybrid_query input rejection';
EXCEPTION WHEN feature_not_supported THEN
END
$$;

DROP TABLE hq_items;
