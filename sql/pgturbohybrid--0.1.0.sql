\echo Use "CREATE EXTENSION pgturbohybrid" to load this file. \quit

DO $pgturbohybrid_compat$
DECLARE
	vector_version pg_catalog.text;
	vector_parts pg_catalog.text[];
	vector_major pg_catalog.int4;
	vector_minor pg_catalog.int4;
	vector_patch pg_catalog.int4;
BEGIN
	IF pg_catalog.current_setting('server_version_num')::pg_catalog.int4 < 140000 THEN
		RAISE EXCEPTION 'pgturbohybrid requires PostgreSQL 14 or newer'
			USING ERRCODE = '0A000';
	END IF;

	SELECT extversion INTO vector_version
	FROM pg_catalog.pg_extension
	WHERE extname = 'vector';

	IF vector_version IS NULL THEN
		RAISE EXCEPTION 'pgturbohybrid requires the vector extension'
			USING ERRCODE = '0A000',
				  HINT = 'Run CREATE EXTENSION vector before CREATE EXTENSION pgturbohybrid.';
	END IF;

	vector_parts := pg_catalog.regexp_match(vector_version, '^([0-9]+)\.([0-9]+)\.([0-9]+)');
	IF vector_parts IS NULL THEN
		RAISE EXCEPTION 'unsupported vector extension version %', vector_version
			USING ERRCODE = '0A000',
				  HINT = 'pgturbohybrid currently supports pgvector 0.8.2 and newer compatible releases.';
	END IF;

	vector_major := vector_parts[1]::pg_catalog.int4;
	vector_minor := vector_parts[2]::pg_catalog.int4;
	vector_patch := vector_parts[3]::pg_catalog.int4;

	IF vector_major = 0 AND (vector_minor < 8 OR (vector_minor = 8 AND vector_patch < 2)) THEN
		RAISE EXCEPTION 'unsupported vector extension version %', vector_version
			USING ERRCODE = '0A000',
				  HINT = 'Install pgvector 0.8.2 or newer before installing pgturbohybrid.';
	END IF;
END
$pgturbohybrid_compat$;

CREATE TYPE turbohybrid_query;

CREATE FUNCTION turbohybrid_query_in(pg_catalog.cstring) RETURNS turbohybrid_query
	AS 'MODULE_PATHNAME', 'pgturbohybrid_query_in'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_query_out(turbohybrid_query) RETURNS pg_catalog.cstring
	AS 'MODULE_PATHNAME', 'pgturbohybrid_query_out'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE TYPE turbohybrid_query (
	INPUT = turbohybrid_query_in,
	OUTPUT = turbohybrid_query_out,
	INTERNALLENGTH = variable,
	STORAGE = extended,
	ALIGNMENT = double
);

CREATE TYPE turbohybrid_sparse_vector;

CREATE FUNCTION turbohybrid_sparse_vector_in(pg_catalog.cstring) RETURNS turbohybrid_sparse_vector
	AS 'MODULE_PATHNAME', 'pgturbohybrid_sparse_vector_in'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_sparse_vector_out(turbohybrid_sparse_vector) RETURNS pg_catalog.cstring
	AS 'MODULE_PATHNAME', 'pgturbohybrid_sparse_vector_out'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE TYPE turbohybrid_sparse_vector (
	INPUT = turbohybrid_sparse_vector_in,
	OUTPUT = turbohybrid_sparse_vector_out,
	INTERNALLENGTH = variable,
	STORAGE = extended,
	ALIGNMENT = double
);

CREATE FUNCTION turbohybrid_sparse_vector_from_arrays(
	term_ids pg_catalog.int4[],
	weights pg_catalog.float4[]
) RETURNS turbohybrid_sparse_vector
	AS 'MODULE_PATHNAME', 'pgturbohybrid_sparse_vector_from_arrays'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_sparse_vector_terms(
	sparse turbohybrid_sparse_vector
) RETURNS pg_catalog.text
	AS 'MODULE_PATHNAME', 'pgturbohybrid_sparse_vector_terms'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_sparse_vector_query_terms(
	sparse turbohybrid_sparse_vector
) RETURNS pg_catalog.text
	AS 'MODULE_PATHNAME', 'pgturbohybrid_sparse_vector_query_terms'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_sparse_vector_to_tsvector(
	sparse turbohybrid_sparse_vector
) RETURNS pg_catalog.tsvector
	LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE
	RETURN pg_catalog.to_tsvector('pg_catalog.simple'::pg_catalog.regconfig,
								  turbohybrid_sparse_vector_terms(sparse));

CREATE FUNCTION turbohybrid_sparse_vector_to_tsquery(
	sparse turbohybrid_sparse_vector
) RETURNS pg_catalog.tsquery
	LANGUAGE SQL IMMUTABLE STRICT PARALLEL SAFE
	RETURN pg_catalog.to_tsquery('pg_catalog.simple'::pg_catalog.regconfig,
								 turbohybrid_sparse_vector_query_terms(sparse));

CREATE TYPE turbohybrid_multivector;

CREATE FUNCTION turbohybrid_multivector_in(pg_catalog.cstring) RETURNS turbohybrid_multivector
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_in'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_multivector_out(turbohybrid_multivector) RETURNS pg_catalog.cstring
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_out'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_multivector_recv(pg_catalog.internal) RETURNS turbohybrid_multivector
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_recv'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_multivector_send(turbohybrid_multivector) RETURNS pg_catalog.bytea
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_send'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE TYPE turbohybrid_multivector (
	INPUT = turbohybrid_multivector_in,
	OUTPUT = turbohybrid_multivector_out,
	RECEIVE = turbohybrid_multivector_recv,
	SEND = turbohybrid_multivector_send,
	INTERNALLENGTH = variable,
	STORAGE = extended,
	ALIGNMENT = double
);

CREATE FUNCTION turbohybrid_multivector(vector[]) RETURNS turbohybrid_multivector
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_constructor'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_multivector_from_float4(
	raw_values pg_catalog.float4[],
	dim pg_catalog.int4
) RETURNS turbohybrid_multivector
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_from_float4'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_multivector_from_contexts(
	raw_values pg_catalog.float4[],
	dim pg_catalog.int4,
	context_offsets pg_catalog.int4[]
) RETURNS turbohybrid_multivector
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_from_contexts'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_multivector_from_contexts_and_fields(
	raw_values pg_catalog.float4[],
	dim pg_catalog.int4,
	context_offsets pg_catalog.int4[],
	field_ids pg_catalog.int4[]
) RETURNS turbohybrid_multivector
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_from_contexts_and_fields'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_multivector_dims(turbohybrid_multivector) RETURNS pg_catalog.int4
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_dims'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_multivector_count(turbohybrid_multivector) RETURNS pg_catalog.int4
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_count'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_multivector_model_info(model_name pg_catalog.text) RETURNS pg_catalog.jsonb
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_model_info'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_multivector_context_count(turbohybrid_multivector) RETURNS pg_catalog.int4
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_context_count'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_multivector_context_offsets(turbohybrid_multivector) RETURNS pg_catalog.int4[]
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_context_offsets'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_multivector_field_ids(turbohybrid_multivector) RETURNS pg_catalog.int4[]
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_field_ids'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_multivector_subvector(
	mv turbohybrid_multivector,
	ordinal pg_catalog.int4
) RETURNS vector
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_subvector'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_multivector_to_vector_array(
	mv turbohybrid_multivector
) RETURNS vector[]
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_to_vector_array'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_multivector_maxsim(
	query turbohybrid_multivector,
	doc turbohybrid_multivector
) RETURNS pg_catalog.float8
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_maxsim'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_multivector_context_maxsim(
	query turbohybrid_multivector,
	doc turbohybrid_multivector
) RETURNS pg_catalog.float8
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_context_maxsim'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_multivector_field_weighted_maxsim(
	query turbohybrid_multivector,
	doc turbohybrid_multivector,
	field_ids pg_catalog.int4[],
	weights pg_catalog.float4[]
) RETURNS pg_catalog.float8
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_field_weighted_maxsim'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_multivector_maxsim_scalar(
	query turbohybrid_multivector,
	doc turbohybrid_multivector
) RETURNS pg_catalog.float8
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_maxsim_scalar'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_multivector_maxsim_blocked_scalar(
	query turbohybrid_multivector,
	doc turbohybrid_multivector
) RETURNS pg_catalog.float8
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_maxsim_blocked_scalar'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_experimental_compact_code_score(
	query_codes pg_catalog.int2[],
	doc_codes pg_catalog.int2[],
	experimental pg_catalog.bool DEFAULT false,
	force_kernel pg_catalog.text DEFAULT 'auto'
) RETURNS pg_catalog.jsonb
	AS 'MODULE_PATHNAME', 'pgturbohybrid_experimental_compact_code_score'
	LANGUAGE C VOLATILE PARALLEL SAFE;

CREATE FUNCTION turbohybrid_multivector_maxsim_distance(
	query turbohybrid_multivector,
	doc turbohybrid_multivector
) RETURNS pg_catalog.float8
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_maxsim_distance'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_query(
	vector_query vector DEFAULT NULL,
	text_query pg_catalog.tsquery DEFAULT NULL,
	fusion pg_catalog.text DEFAULT 'rrf',
	dense_weight pg_catalog.float8 DEFAULT 1.0,
	bm25_weight pg_catalog.float8 DEFAULT 1.0,
	alpha pg_catalog.float8 DEFAULT NULL,
	rrf_k pg_catalog.int4 DEFAULT NULL,
	dense_k pg_catalog.int4 DEFAULT NULL,
	bm25_k pg_catalog.int4 DEFAULT NULL,
	final_k pg_catalog.int4 DEFAULT NULL,
	require_bm25_match pg_catalog.bool DEFAULT false,
	multivector_query turbohybrid_multivector DEFAULT NULL,
	query_token_weights pg_catalog.float4[] DEFAULT NULL,
	query_token_mask pg_catalog.bool[] DEFAULT NULL
) RETURNS turbohybrid_query
	AS 'MODULE_PATHNAME', 'pgturbohybrid_query_constructor'
	LANGUAGE C STABLE PARALLEL SAFE;

CREATE FUNCTION turbohybrid_distance(vector, turbohybrid_query) RETURNS pg_catalog.float8
	AS 'MODULE_PATHNAME', 'pgturbohybrid_distance'
	LANGUAGE C STABLE STRICT PARALLEL RESTRICTED;

CREATE FUNCTION turbohybrid_l2_distance(vector, turbohybrid_query) RETURNS pg_catalog.float8
	AS 'MODULE_PATHNAME', 'pgturbohybrid_l2_distance'
	LANGUAGE C STABLE STRICT PARALLEL RESTRICTED;

CREATE FUNCTION turbohybrid_negative_inner_product(vector, turbohybrid_query) RETURNS pg_catalog.float8
	AS 'MODULE_PATHNAME', 'pgturbohybrid_negative_inner_product'
	LANGUAGE C STABLE STRICT PARALLEL RESTRICTED;

CREATE FUNCTION turbohybrid_cosine_distance(vector, turbohybrid_query) RETURNS pg_catalog.float8
	AS 'MODULE_PATHNAME', 'pgturbohybrid_cosine_distance'
	LANGUAGE C STABLE STRICT PARALLEL RESTRICTED;

CREATE FUNCTION turbohybrid_multivector_distance(turbohybrid_multivector, turbohybrid_query) RETURNS pg_catalog.float8
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_query_distance'
	LANGUAGE C STABLE STRICT PARALLEL RESTRICTED;

CREATE FUNCTION turbohybrid_vector_l2_squared_distance(vector, vector) RETURNS pg_catalog.float8
	AS 'MODULE_PATHNAME', 'pgturbohybrid_vector_l2_squared_distance'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_vector_l2_distance(vector, vector) RETURNS pg_catalog.float8
	AS 'MODULE_PATHNAME', 'pgturbohybrid_vector_l2_distance'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_vector_negative_inner_product(vector, vector) RETURNS pg_catalog.float8
	AS 'MODULE_PATHNAME', 'pgturbohybrid_vector_negative_inner_product'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_vector_cosine_distance(vector, vector) RETURNS pg_catalog.float8
	AS 'MODULE_PATHNAME', 'pgturbohybrid_vector_cosine_distance'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_vector_norm(vector) RETURNS pg_catalog.float8
	AS 'MODULE_PATHNAME', 'pgturbohybrid_vector_norm'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE OPERATOR <~-> (
	LEFTARG = vector, RIGHTARG = turbohybrid_query, PROCEDURE = turbohybrid_l2_distance
);

CREATE OPERATOR <~#> (
	LEFTARG = vector, RIGHTARG = turbohybrid_query, PROCEDURE = turbohybrid_negative_inner_product
);

CREATE OPERATOR <~> (
	LEFTARG = vector, RIGHTARG = turbohybrid_query, PROCEDURE = turbohybrid_cosine_distance
);

CREATE OPERATOR <~> (
	LEFTARG = turbohybrid_multivector, RIGHTARG = turbohybrid_query, PROCEDURE = turbohybrid_multivector_distance
);

CREATE FUNCTION turbohybrid_handler(pg_catalog.internal) RETURNS pg_catalog.index_am_handler
	AS 'MODULE_PATHNAME', 'pgturbohybrid_handler'
	LANGUAGE C;

CREATE ACCESS METHOD turbohybrid TYPE INDEX HANDLER turbohybrid_handler;

CREATE OPERATOR CLASS vector_l2_turbohybrid_ops
	DEFAULT FOR TYPE vector USING turbohybrid AS
	OPERATOR 1 <~-> (vector, turbohybrid_query) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 turbohybrid_vector_l2_squared_distance(vector, vector),
	FUNCTION 3 turbohybrid_vector_l2_distance(vector, vector);

CREATE OPERATOR CLASS vector_ip_turbohybrid_ops
	FOR TYPE vector USING turbohybrid AS
	OPERATOR 1 <~#> (vector, turbohybrid_query) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 turbohybrid_vector_negative_inner_product(vector, vector),
	FUNCTION 4 turbohybrid_vector_norm(vector);

CREATE OPERATOR CLASS vector_cosine_turbohybrid_ops
	FOR TYPE vector USING turbohybrid AS
	OPERATOR 1 <~> (vector, turbohybrid_query) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 turbohybrid_vector_negative_inner_product(vector, vector),
	FUNCTION 2 turbohybrid_vector_norm(vector),
	FUNCTION 4 turbohybrid_vector_norm(vector);

CREATE OPERATOR CLASS multivector_cosine_turbohybrid_ops
	FOR TYPE turbohybrid_multivector USING turbohybrid AS
	OPERATOR 1 <~> (turbohybrid_multivector, turbohybrid_query) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 turbohybrid_vector_negative_inner_product(vector, vector),
	FUNCTION 2 turbohybrid_vector_norm(vector),
	FUNCTION 4 turbohybrid_vector_norm(vector);

CREATE OPERATOR CLASS multivector_maxsim_ip_turbohybrid_ops
	FOR TYPE turbohybrid_multivector USING turbohybrid AS
	OPERATOR 1 <~> (turbohybrid_multivector, turbohybrid_query) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 turbohybrid_vector_negative_inner_product(vector, vector),
	FUNCTION 2 turbohybrid_vector_norm(vector),
	FUNCTION 4 turbohybrid_vector_norm(vector);

CREATE OPERATOR CLASS bm25_tsvector_turbohybrid_ops
	FOR TYPE pg_catalog.tsvector USING turbohybrid AS
	STORAGE pg_catalog.tsvector;

CREATE FUNCTION turbohybrid_index_stats(pg_catalog.regclass) RETURNS pg_catalog.jsonb
	AS 'MODULE_PATHNAME', 'pgturbohybrid_index_stats'
	LANGUAGE C STABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_estimate_memory(pg_catalog.regclass) RETURNS pg_catalog.jsonb
	AS 'MODULE_PATHNAME', 'pgturbohybrid_estimate_memory'
	LANGUAGE C STABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_prewarm(pg_catalog.regclass) RETURNS pg_catalog.jsonb
	AS 'MODULE_PATHNAME', 'pgturbohybrid_prewarm'
	LANGUAGE C STRICT PARALLEL RESTRICTED;

CREATE FUNCTION turbohybrid_multivector_proxy_diagnostics(
	index pg_catalog.regclass,
	sample_docs pg_catalog.int4 DEFAULT 100,
	query_count pg_catalog.int4 DEFAULT 20
) RETURNS pg_catalog.jsonb
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_proxy_diagnostics'
	LANGUAGE C STABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_last_scan_stats() RETURNS pg_catalog.jsonb
	AS 'MODULE_PATHNAME', 'pgturbohybrid_last_scan_stats'
	LANGUAGE C PARALLEL RESTRICTED;

CREATE FUNCTION turbohybrid_last_build_stats() RETURNS pg_catalog.jsonb
	AS 'MODULE_PATHNAME', 'pgturbohybrid_last_build_stats'
	LANGUAGE C PARALLEL RESTRICTED;

CREATE FUNCTION turbohybrid_graph_repair_dry_run(
	index pg_catalog.regclass,
	sample_nodes pg_catalog.int4 DEFAULT 1000,
	search_ef pg_catalog.int4 DEFAULT 400,
	candidate_limit pg_catalog.int4 DEFAULT 200
) RETURNS pg_catalog.jsonb
	AS 'MODULE_PATHNAME', 'pgturbohybrid_graph_repair_dry_run'
	LANGUAGE C STABLE STRICT PARALLEL RESTRICTED;

-- Compact bottleneck diagnosis for the last TurboHybrid scan.  Pulls the key
-- dense hot-path fields out of turbohybrid_last_scan_stats(), adds a few derived
-- ratios, and reduces them to a single 'diagnosis' label, so one SELECT explains
-- a slow query without parsing the full stats JSON.  Strictly read-only: it only
-- reads the stats the previous scan already recorded and never alters scan
-- execution (hence VOLATILE / PARALLEL RESTRICTED, matching the underlying
-- backend-local scan state).
--
-- Timer note: graph_traverse_us is the umbrella phase that already contains
-- graph_base_us, and base contains graph_batch_us and graph_heap_us.  The
-- traversal-overhead test therefore compares the useful SIMD scoring time
-- (graph_batch_us) against the rest of the traverse (graph_traverse_us minus
-- graph_batch_us); summing the nested timers would double-count batch scoring.
-- graph_base_us has no flat stats key, so it is read from the nested
-- dense.timing_us.base section.
CREATE FUNCTION turbohybrid_last_scan_diagnosis() RETURNS pg_catalog.jsonb
	LANGUAGE plpgsql PARALLEL RESTRICTED AS $turbohybrid_diag$
DECLARE
	s             jsonb := turbohybrid_last_scan_stats();
	orchestration text   := s ->> 'scan_orchestration';
	u8_mode       text   := s ->> 'graph_u8_batch_mode';
	scored        bigint := COALESCE((s ->> 'graph_scored_codes')::bigint, 0);
	scalar_codes  bigint := COALESCE((s ->> 'graph_scalar_scored_codes')::bigint, 0);
	visited       bigint := COALESCE((s ->> 'graph_visited_nodes')::bigint, 0);
	batch_us      bigint := COALESCE((s ->> 'graph_batch_us')::bigint, 0);
	heap_us       bigint := COALESCE((s ->> 'graph_heap_us')::bigint, 0);
	base_us       bigint := COALESCE((s -> 'dense' -> 'timing_us' ->> 'base')::bigint, 0);
	traverse_us   bigint := COALESCE((s ->> 'graph_traverse_us')::bigint, 0);
	rescore_us    bigint := COALESCE((s ->> 'graph_rescore_us')::bigint, 0);
	native_scope  text   := s ->> 'native_cache_scope';
	native_reason text   := s ->> 'native_cache_reason';
	native_built  boolean := COALESCE((s ->> 'native_cache_built_this_scan')::boolean, false);
	native_build_us bigint := COALESCE((s ->> 'native_cache_build_us')::bigint, 0);
	native_attach_us bigint := COALESCE((s ->> 'native_cache_attach_us')::bigint, 0);
	native_wait_us bigint := COALESCE((s ->> 'native_cache_wait_us')::bigint, 0);
	native_bytes bigint := COALESCE((s ->> 'native_cache_bytes')::bigint, 0);
	ef            bigint := COALESCE((s ->> 'graph_effective_search_ef')::bigint, 0);
	result_target bigint := COALESCE((s ->> 'graph_effective_result_target')::bigint, 0);
	large_arena   boolean := COALESCE((s ->> 'graph_large_code_arena')::boolean, false);
	hit_rate      float8 := COALESCE((s -> 'graph_code_pages' ->> 'hit_rate')::float8, 0);
	target_floor  bigint := GREATEST(result_target, 1);
	batch_per_code   float8 := CASE WHEN scored  > 0 THEN batch_us::float8 / scored  ELSE 0 END;
	heap_per_visited float8 := CASE WHEN visited > 0 THEN heap_us::float8  / visited ELSE 0 END;
	overhead_us   bigint := GREATEST(traverse_us - batch_us, 0);  -- traverse work that is not SIMD scoring
	diagnosis     text;
BEGIN
	-- Priority-ordered single label: configuration red flags first, then the
	-- dominant time component, then the healthy/expected case.
	IF orchestration IS DISTINCT FROM 'graph_native' OR (scored = 0 AND traverse_us = 0) THEN
		diagnosis := 'no_native_dense_scan';
	ELSIF native_built AND native_build_us > GREATEST(traverse_us, batch_us, rescore_us, 1000) THEN
		-- The first scan in this backend paid the native cache materialization cost.
		diagnosis := 'native_cache_cold_build';
	ELSIF scalar_codes > 0 AND scalar_codes * 10 >= scored THEN
		-- >=10% of scored codes took the scalar/LUT path: the SIMD scorer did not engage.
		diagnosis := 'scalar_lut_fallback';
	ELSIF u8_mode = 'single' THEN
		-- u8 batch path ran but not as x4 (dense_u8_batch_x4 off, or x4 unavailable).
		diagnosis := 'u8_x4_disabled';
	ELSIF rescore_us > 0 AND rescore_us > batch_us AND rescore_us > overhead_us THEN
		-- exact f32 rescore is the largest of {batch scoring, traversal, rescore}.
		diagnosis := 'rescore_dominated';
	ELSIF large_arena AND batch_per_code > 0.5 THEN
		-- code working set exceeds CPU cache and per-code scoring is slow: RAM-bound.
		diagnosis := 'memory_bound';
	ELSIF overhead_us::float8 > batch_us * 1.5 AND traverse_us > 0 THEN
		-- graph walk / heap / frontier bookkeeping clearly outweighs SIMD scoring.
		diagnosis := 'traversal_dominated';
	ELSIF (ef > 16 * target_floor AND ef > 256)
		  OR (scored > 64 * target_floor AND scored > 4096) THEN
		-- candidate budget far exceeds the requested result target.
		diagnosis := 'candidate_budget_high';
	ELSIF u8_mode = 'x4' THEN
		-- u8 x4 batch active and SIMD scoring dominates within the expected range.
		diagnosis := 'healthy_u8_x4';
	ELSE
		diagnosis := 'ok';
	END IF;

	RETURN jsonb_build_object(
		'scan_orchestration', orchestration,
		'graph_storage_kind', s ->> 'graph_storage_kind',
		'dimensions', s -> 'dimensions',
		'quantization_bits', s -> 'quantization_bits',
		'score_mode', s ->> 'score_mode',
		'dense_scorer', s ->> 'dense_scorer',
		'graph_u8_batch_mode', u8_mode,
		'u8_kernel_batch', s ->> 'u8_kernel_batch',
		'dense_u8_batch_x4_enabled', s -> 'dense_u8_batch_x4_enabled',
		'graph_large_code_arena', s -> 'graph_large_code_arena',
		'graph_whole_code_prefetch_active', s -> 'graph_whole_code_prefetch_active',
		'native_cache_policy', s ->> 'native_cache_policy',
		'native_cache_scope', native_scope,
		'native_cache_reason', native_reason,
		'native_cache_used', s -> 'native_cache_used',
		'native_cache_reused', s -> 'native_cache_reused',
		'native_cache_built_this_scan', native_built,
		'native_cache_build_us', native_build_us,
		'native_cache_attach_us', native_attach_us,
		'native_cache_wait_us', native_wait_us,
		'native_cache_bytes', native_bytes,
		'graph_scored_codes', scored,
		'graph_batch_us', batch_us,
		'graph_heap_us', heap_us,
		'graph_base_us', base_us,
		'graph_traverse_us', traverse_us,
		'graph_rescore_us', rescore_us,
		'graph_effective_search_ef', ef,
		'graph_effective_result_target', result_target,
		'graph_effective_rescore_band', s -> 'graph_effective_rescore_band',
		'code_page_hit_rate', round(hit_rate::numeric, 4),
		'batch_us_per_code', round(batch_per_code::numeric, 4),
		'heap_us_per_visited_node', round(heap_per_visited::numeric, 4),
		'diagnosis', diagnosis
	);
END;
$turbohybrid_diag$;

CREATE FUNCTION turbohybrid_simd_capabilities() RETURNS pg_catalog.jsonb
	AS 'MODULE_PATHNAME', 'pgturbohybrid_simd_capabilities'
	LANGUAGE C STABLE PARALLEL SAFE;

-- Diagnostic: score one (query, doc) pair under each dense approximate scorer
-- (scalar/LUT and signed query split) plus a linear/uniform-quantizer
-- reference, so operators can inspect quantization error and tests can prove
-- the SIMD path uses the non-uniform codebook rather than raw nibbles.
CREATE FUNCTION turbohybrid_scorer_distances(vector, vector) RETURNS pg_catalog.jsonb
	AS 'MODULE_PATHNAME', 'turbohybrid_scorer_distances'
	LANGUAGE C STABLE STRICT PARALLEL SAFE;

-- Same diagnostic at a chosen quantization bit width (2 or 4).
CREATE FUNCTION turbohybrid_scorer_distances(vector, vector, integer) RETURNS pg_catalog.jsonb
	AS 'MODULE_PATHNAME', 'turbohybrid_scorer_distances'
	LANGUAGE C STABLE STRICT PARALLEL SAFE;

-- Exposes the shared low/high query-split quantization (qScale fixed at 1, so the
-- value argument is the already-scaled value) for regression-pinning the split
-- boundaries.  high_coef = 256 is the signed split, 128 the x86 u8 split.
CREATE FUNCTION turbohybrid_query_split_probe(float8, float8, integer) RETURNS pg_catalog.jsonb
	AS 'MODULE_PATHNAME', 'turbohybrid_query_split_probe'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Diagnostic that scores four DISTINCT docs both through the x4 u8 batch kernel
-- (one pass) and through four single-node u8 kernel calls, returning both result
-- arrays and a bit-exact match flag.  Proves the 4-candidate batch equals four
-- single-code calls for distinct codes -- the case the native scorer runs.
CREATE FUNCTION turbohybrid_scorer_x4_batch_parity(vector, vector, vector, vector, vector, integer) RETURNS pg_catalog.jsonb
	AS 'MODULE_PATHNAME', 'turbohybrid_scorer_x4_batch_parity'
	LANGUAGE C STABLE CALLED ON NULL INPUT PARALLEL SAFE;

-- Tight-loop ns/code microbenchmark of the dense 4-bit scoring kernels
-- (scalar/LUT, signed split, single-node u8 split, x4 u8 split batch).  Times
-- scoring `ncodes` cache-resident codes over `iters` passes, reporting the
-- minimum ns/code per kernel -- isolating raw per-code compute from the scan
-- machinery and the memory wall.  Non-strict so ncodes/iters may default.
CREATE FUNCTION turbohybrid_scorer_bench(vector, vector, integer, integer, integer) RETURNS pg_catalog.jsonb
	AS 'MODULE_PATHNAME', 'turbohybrid_scorer_bench'
	LANGUAGE C VOLATILE CALLED ON NULL INPUT PARALLEL SAFE;

COMMENT ON EXTENSION pgturbohybrid IS 'TurboHybrid dense vector and BM25 search extension for pgvector';
COMMENT ON ACCESS METHOD turbohybrid IS 'TurboHybrid dense vector and BM25 hybrid index access method';
COMMENT ON TYPE turbohybrid_query IS 'TurboHybrid query payload for dense vector and BM25 hybrid search';

COMMENT ON FUNCTION turbohybrid_query_in(pg_catalog.cstring) IS 'Input function for turbohybrid_query';
COMMENT ON FUNCTION turbohybrid_query_out(turbohybrid_query) IS 'Output function for turbohybrid_query';
COMMENT ON FUNCTION turbohybrid_query(vector, pg_catalog.tsquery, pg_catalog.text, pg_catalog.float8, pg_catalog.float8, pg_catalog.float8, pg_catalog.int4, pg_catalog.int4, pg_catalog.int4, pg_catalog.int4, pg_catalog.bool, turbohybrid_multivector, pg_catalog.float4[], pg_catalog.bool[]) IS 'Constructs a TurboHybrid query payload';
COMMENT ON FUNCTION turbohybrid_distance(vector, turbohybrid_query) IS 'Default TurboHybrid distance between a vector and a query';
COMMENT ON FUNCTION turbohybrid_l2_distance(vector, turbohybrid_query) IS 'L2 TurboHybrid distance between a vector and a query';
COMMENT ON FUNCTION turbohybrid_negative_inner_product(vector, turbohybrid_query) IS 'Negative inner product TurboHybrid distance between a vector and a query';
COMMENT ON FUNCTION turbohybrid_cosine_distance(vector, turbohybrid_query) IS 'Cosine TurboHybrid distance between a vector and a query';
COMMENT ON FUNCTION turbohybrid_multivector_distance(turbohybrid_multivector, turbohybrid_query) IS 'Exact MaxSim distance between a multivector and a TurboHybrid query';
COMMENT ON FUNCTION turbohybrid_multivector_from_contexts(pg_catalog.float4[], pg_catalog.int4, pg_catalog.int4[]) IS 'Construct a context-aware turbohybrid_multivector from flat float values and zero-based context start token offsets';
COMMENT ON FUNCTION turbohybrid_multivector_from_contexts_and_fields(pg_catalog.float4[], pg_catalog.int4, pg_catalog.int4[], pg_catalog.int4[]) IS 'Construct a context-aware turbohybrid_multivector with one non-negative field id per context';
COMMENT ON FUNCTION turbohybrid_multivector_model_info(pg_catalog.text) IS 'Return registered late-interaction model metadata for multivector validation and benchmark provenance';
COMMENT ON FUNCTION turbohybrid_multivector_context_count(turbohybrid_multivector) IS 'Return the number of context windows stored in a turbohybrid_multivector';
COMMENT ON FUNCTION turbohybrid_multivector_context_offsets(turbohybrid_multivector) IS 'Return zero-based context start token offsets from a turbohybrid_multivector';
COMMENT ON FUNCTION turbohybrid_multivector_field_ids(turbohybrid_multivector) IS 'Return one field id per context window from a turbohybrid_multivector, defaulting to 0 for flat values';
COMMENT ON FUNCTION turbohybrid_multivector_context_maxsim(turbohybrid_multivector, turbohybrid_multivector) IS 'Exact context-level MaxSim: score each document context independently and return the best context score';
COMMENT ON FUNCTION turbohybrid_multivector_field_weighted_maxsim(turbohybrid_multivector, turbohybrid_multivector, pg_catalog.int4[], pg_catalog.float4[]) IS 'Exact field-weighted MaxSim over context field ids, using one weight per requested field';
COMMENT ON FUNCTION turbohybrid_multivector_subvector(turbohybrid_multivector, pg_catalog.int4) IS 'Return one 1-based subvector from a turbohybrid_multivector as vector';
COMMENT ON FUNCTION turbohybrid_multivector_to_vector_array(turbohybrid_multivector) IS 'Return all subvectors from a turbohybrid_multivector as vector[]';
COMMENT ON FUNCTION turbohybrid_vector_l2_squared_distance(vector, vector) IS 'Squared L2 support function used by TurboHybrid vector opclasses';
COMMENT ON FUNCTION turbohybrid_vector_l2_distance(vector, vector) IS 'L2 support function used by TurboHybrid vector opclasses';
COMMENT ON FUNCTION turbohybrid_vector_negative_inner_product(vector, vector) IS 'Negative inner product support function used by TurboHybrid vector opclasses';
COMMENT ON FUNCTION turbohybrid_vector_cosine_distance(vector, vector) IS 'Cosine support function used by TurboHybrid vector opclasses';
COMMENT ON FUNCTION turbohybrid_vector_norm(vector) IS 'Vector norm support function used by TurboHybrid vector opclasses';
COMMENT ON FUNCTION turbohybrid_handler(pg_catalog.internal) IS 'Index access method handler for TurboHybrid';
COMMENT ON FUNCTION turbohybrid_index_stats(pg_catalog.regclass) IS 'Return stable TurboHybrid index metadata as jsonb';
COMMENT ON FUNCTION turbohybrid_estimate_memory(pg_catalog.regclass) IS 'Estimate native graph cache and BM25 reader-cache memory for a TurboHybrid index without building or loading those caches';
COMMENT ON FUNCTION turbohybrid_prewarm(pg_catalog.regclass) IS 'Build or attach the shared native graph cache for a TurboHybrid native graph index and return cache diagnostics as jsonb';
COMMENT ON FUNCTION turbohybrid_multivector_proxy_diagnostics(pg_catalog.regclass, pg_catalog.int4, pg_catalog.int4) IS 'Read-only bounded diagnostic comparing document proxy nearest-neighbor order with exact multivector MaxSim order for sampled document-node multivector indexes';
COMMENT ON FUNCTION turbohybrid_last_scan_stats() IS 'Return backend-local summary information for the last TurboHybrid scan as jsonb; parallel restricted because it reads mutable scan state';
COMMENT ON FUNCTION turbohybrid_last_build_stats() IS 'Return backend-local summary information for the last native TurboHybrid graph build as jsonb; parallel restricted because it reads mutable build state';
COMMENT ON FUNCTION turbohybrid_graph_repair_dry_run(pg_catalog.regclass, pg_catalog.int4, pg_catalog.int4, pg_catalog.int4) IS 'Read-only native graph repair diagnostic; samples graph neighborhoods and reports weak-neighborhood and suggested-edge counts without modifying the index';
COMMENT ON FUNCTION turbohybrid_last_scan_diagnosis() IS 'Return a compact bottleneck diagnosis of the last TurboHybrid scan as jsonb: key dense hot-path fields, derived ratios, and a single diagnosis label; read-only over turbohybrid_last_scan_stats()';
COMMENT ON FUNCTION turbohybrid_simd_capabilities() IS 'Return pgturbohybrid build and architecture SIMD capability information as jsonb';

COMMENT ON OPERATOR <~-> (vector, turbohybrid_query) IS 'L2 distance operator for TurboHybrid vector queries';
COMMENT ON OPERATOR <~#> (vector, turbohybrid_query) IS 'Negative inner product distance operator for TurboHybrid vector queries';
COMMENT ON OPERATOR <~> (vector, turbohybrid_query) IS 'Cosine distance operator for TurboHybrid vector queries';

COMMENT ON OPERATOR CLASS vector_l2_turbohybrid_ops USING turbohybrid IS 'TurboHybrid L2 vector operator class';
COMMENT ON OPERATOR CLASS vector_ip_turbohybrid_ops USING turbohybrid IS 'TurboHybrid inner product vector operator class';
COMMENT ON OPERATOR CLASS vector_cosine_turbohybrid_ops USING turbohybrid IS 'TurboHybrid cosine vector operator class';
COMMENT ON OPERATOR CLASS multivector_cosine_turbohybrid_ops USING turbohybrid IS 'Compatibility multivector operator class for normalized token vectors; uses dot-product MaxSim internally';
COMMENT ON OPERATOR CLASS multivector_maxsim_ip_turbohybrid_ops USING turbohybrid IS 'TurboHybrid multivector operator class exposing raw dot-product MaxSim semantics';
COMMENT ON OPERATOR CLASS bm25_tsvector_turbohybrid_ops USING turbohybrid IS 'TurboHybrid BM25 tsvector operator class';

COMMENT ON OPERATOR FAMILY vector_l2_turbohybrid_ops USING turbohybrid IS 'TurboHybrid L2 vector operator family';
COMMENT ON OPERATOR FAMILY vector_ip_turbohybrid_ops USING turbohybrid IS 'TurboHybrid inner product vector operator family';
COMMENT ON OPERATOR FAMILY vector_cosine_turbohybrid_ops USING turbohybrid IS 'TurboHybrid cosine vector operator family';
COMMENT ON OPERATOR FAMILY multivector_cosine_turbohybrid_ops USING turbohybrid IS 'Compatibility multivector operator family for normalized token vectors; uses dot-product MaxSim internally';
COMMENT ON OPERATOR FAMILY multivector_maxsim_ip_turbohybrid_ops USING turbohybrid IS 'TurboHybrid multivector raw dot-product MaxSim operator family';
COMMENT ON OPERATOR FAMILY bm25_tsvector_turbohybrid_ops USING turbohybrid IS 'TurboHybrid BM25 tsvector operator family';
