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

-- Canonical builder: typed-args C worker (not STRICT; top_k may be NULL).
CREATE FUNCTION turbohybrid_sparse_vector_build(
	term_ids pg_catalog.int4[],
	weights pg_catalog.float4[],
	drop_non_positive pg_catalog.bool,
	deduplicate pg_catalog.text,
	sort pg_catalog.bool,
	top_k pg_catalog.int4,
	min_weight pg_catalog.float8,
	normalize_mode pg_catalog.text
) RETURNS turbohybrid_sparse_vector
	AS 'MODULE_PATHNAME', 'pgturbohybrid_sparse_vector_build'
	LANGUAGE C IMMUTABLE PARALLEL SAFE;

-- Public jsonb-options form: extracts options (with defaults) and calls the worker.
CREATE FUNCTION turbohybrid_sparse_vector_build(
	term_ids pg_catalog.int4[],
	weights pg_catalog.float4[],
	options pg_catalog.jsonb DEFAULT '{}'::pg_catalog.jsonb
) RETURNS turbohybrid_sparse_vector
	LANGUAGE SQL IMMUTABLE PARALLEL SAFE
	RETURN turbohybrid_sparse_vector_build(
		term_ids,
		weights,
		COALESCE((options ->> 'drop_non_positive')::pg_catalog.bool, true),
		COALESCE(options ->> 'deduplicate', 'sum'),
		COALESCE((options ->> 'sort')::pg_catalog.bool, true),
		(options ->> 'top_k')::pg_catalog.int4,
		COALESCE((options ->> 'min_weight')::pg_catalog.float8, 0.0),
		COALESCE(options ->> 'normalize', 'none'));

CREATE FUNCTION turbohybrid_sparse_vector_term_ids(
	sparse turbohybrid_sparse_vector
) RETURNS pg_catalog.int4[]
	AS 'MODULE_PATHNAME', 'pgturbohybrid_sparse_vector_term_ids'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_sparse_vector_weights(
	sparse turbohybrid_sparse_vector
) RETURNS pg_catalog.float4[]
	AS 'MODULE_PATHNAME', 'pgturbohybrid_sparse_vector_weights'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_sparse_vector_count(
	sparse turbohybrid_sparse_vector
) RETURNS pg_catalog.int4
	AS 'MODULE_PATHNAME', 'pgturbohybrid_sparse_vector_count'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

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

CREATE FUNCTION turbohybrid_experimental_query(
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
	query_token_mask pg_catalog.bool[] DEFAULT NULL,
	multivector_weight pg_catalog.float8 DEFAULT 1.0,
	multivector_k pg_catalog.int4 DEFAULT NULL,
	sparse_query turbohybrid_sparse_vector DEFAULT NULL,
	sparse_weight pg_catalog.float8 DEFAULT 1.0,
	sparse_k pg_catalog.int4 DEFAULT NULL,
	require_sparse_match pg_catalog.bool DEFAULT false
) RETURNS turbohybrid_query
	AS '$libdir/pgturbohybrid', 'pgturbohybrid_query_constructor'
	LANGUAGE C STABLE PARALLEL SAFE;
CREATE FUNCTION turbohybrid_sparse_query(
	sparse_query turbohybrid_sparse_vector,
	final_k pg_catalog.int4 DEFAULT NULL,
	sparse_k pg_catalog.int4 DEFAULT NULL
) RETURNS turbohybrid_query
	LANGUAGE SQL STABLE PARALLEL SAFE
	RETURN turbohybrid_experimental_query(
		sparse_query => sparse_query,
		final_k => final_k,
		sparse_k => sparse_k);

CREATE FUNCTION turbohybrid_multivector_query(
	multivector_query turbohybrid_multivector,
	final_k pg_catalog.int4 DEFAULT NULL,
	multivector_k pg_catalog.int4 DEFAULT NULL
) RETURNS turbohybrid_query
	LANGUAGE SQL STABLE PARALLEL SAFE
	RETURN turbohybrid_experimental_query(
		multivector_query => multivector_query,
		final_k => final_k,
		multivector_k => multivector_k);

CREATE FUNCTION turbohybrid_sparse_inner_product_distance(
	turbohybrid_sparse_vector, turbohybrid_query
) RETURNS pg_catalog.float8
	AS 'MODULE_PATHNAME', 'pgturbohybrid_sparse_inner_product_distance'
	LANGUAGE C STABLE STRICT PARALLEL RESTRICTED;

CREATE OPERATOR <~*> (
	LEFTARG = turbohybrid_sparse_vector, RIGHTARG = turbohybrid_query,
	PROCEDURE = turbohybrid_sparse_inner_product_distance
);
CREATE FUNCTION turbohybrid_multivector_distance(turbohybrid_multivector, turbohybrid_query) RETURNS pg_catalog.float8
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_query_distance'
	LANGUAGE C STABLE STRICT PARALLEL RESTRICTED;

CREATE OPERATOR <~> (
	LEFTARG = turbohybrid_multivector, RIGHTARG = turbohybrid_query, PROCEDURE = turbohybrid_multivector_distance
);
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
-- turbohybrid_sparse_vector.  Native sparse index scan is implemented (see
-- src/pgturbohybrid_sparse_*.c): a turbohybrid index keyed on this opclass
-- builds a native sparse postings store and serves ORDER BY ... <~*> scans,
-- standalone (sparse-only / sparse+BM25) or alongside a dense/multivector graph.
-- Maturity: experimental public (see docs/feature-matrix.md and
-- docs/sparse-embeddings.md).
CREATE OPERATOR CLASS sparse_ip_turbohybrid_ops
	FOR TYPE turbohybrid_sparse_vector USING turbohybrid AS
	OPERATOR 1 <~*> (turbohybrid_sparse_vector, turbohybrid_query) FOR ORDER BY pg_catalog.float_ops,
	STORAGE turbohybrid_sparse_vector;
CREATE FUNCTION turbohybrid_sparse_compact(pg_catalog.regclass) RETURNS pg_catalog.bool
	AS 'MODULE_PATHNAME', 'pgturbohybrid_sparse_compact'
	LANGUAGE C VOLATILE STRICT PARALLEL UNSAFE;

CREATE FUNCTION turbohybrid_multivector_proxy_diagnostics(
	index pg_catalog.regclass,
	sample_docs pg_catalog.int4 DEFAULT 100,
	query_count pg_catalog.int4 DEFAULT 20
) RETURNS pg_catalog.jsonb
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_proxy_diagnostics'
	LANGUAGE C STABLE STRICT PARALLEL SAFE;

COMMENT ON TYPE turbohybrid_sparse_vector IS 'Learned-sparse (SPLADE-style) vector value: sorted (term_id, weight) pairs. [experimental public]';
COMMENT ON TYPE turbohybrid_multivector IS 'Late-interaction multivector: several same-dimensional token vectors for one row (ColBERT-style MaxSim). [experimental public]';
COMMENT ON FUNCTION turbohybrid_sparse_vector_in(pg_catalog.cstring) IS 'Input function for turbohybrid_sparse_vector. [internal helper]';
COMMENT ON FUNCTION turbohybrid_sparse_vector_out(turbohybrid_sparse_vector) IS 'Output function for turbohybrid_sparse_vector. [internal helper]';
COMMENT ON FUNCTION turbohybrid_multivector_in(pg_catalog.cstring) IS 'Input function for turbohybrid_multivector. [internal helper]';
COMMENT ON FUNCTION turbohybrid_multivector_out(turbohybrid_multivector) IS 'Output function for turbohybrid_multivector. [internal helper]';
COMMENT ON FUNCTION turbohybrid_multivector_recv(pg_catalog.internal) IS 'Binary receive function for turbohybrid_multivector. [internal helper]';
COMMENT ON FUNCTION turbohybrid_multivector_send(turbohybrid_multivector) IS 'Binary send function for turbohybrid_multivector. [internal helper]';
COMMENT ON FUNCTION turbohybrid_experimental_query(vector, pg_catalog.tsquery, pg_catalog.text, pg_catalog.float8, pg_catalog.float8, pg_catalog.float8, pg_catalog.int4, pg_catalog.int4, pg_catalog.int4, pg_catalog.int4, pg_catalog.bool, turbohybrid_multivector, pg_catalog.float4[], pg_catalog.bool[], pg_catalog.float8, pg_catalog.int4, turbohybrid_sparse_vector, pg_catalog.float8, pg_catalog.int4, pg_catalog.bool) IS 'Construct a TurboHybrid query payload for dense, BM25, sparse, and/or multivector retrieval with fusion. [stable public]';
COMMENT ON FUNCTION turbohybrid_sparse_query(turbohybrid_sparse_vector, pg_catalog.int4, pg_catalog.int4) IS 'Convenience wrapper: sparse-only TurboHybrid query payload (forwards to turbohybrid_query). [experimental public]';
COMMENT ON FUNCTION turbohybrid_multivector_query(turbohybrid_multivector, pg_catalog.int4, pg_catalog.int4) IS 'Convenience wrapper: multivector-only TurboHybrid query payload (forwards to turbohybrid_query). [experimental public]';
COMMENT ON FUNCTION turbohybrid_sparse_vector_from_arrays(pg_catalog.int4[], pg_catalog.float4[]) IS 'Build a turbohybrid_sparse_vector from parallel term_id/weight arrays. [experimental public]';
COMMENT ON FUNCTION turbohybrid_sparse_vector_build(pg_catalog.int4[], pg_catalog.float4[], pg_catalog.bool, pg_catalog.text, pg_catalog.bool, pg_catalog.int4, pg_catalog.float8, pg_catalog.text) IS 'Typed-args sparse-vector builder worker (deduplicate/sort/top_k/min_weight/normalize). [internal helper]';
COMMENT ON FUNCTION turbohybrid_sparse_vector_build(pg_catalog.int4[], pg_catalog.float4[], pg_catalog.jsonb) IS 'Build a turbohybrid_sparse_vector with jsonb options (drop_non_positive, deduplicate, sort, top_k, min_weight, normalize). [experimental public]';
COMMENT ON FUNCTION turbohybrid_sparse_vector_terms(turbohybrid_sparse_vector) IS 'Render sparse terms as a weighted simple-text string. [experimental public]';
COMMENT ON FUNCTION turbohybrid_sparse_vector_query_terms(turbohybrid_sparse_vector) IS 'Render sparse terms as an OR-joined query string. [experimental public]';
COMMENT ON FUNCTION turbohybrid_sparse_vector_to_tsvector(turbohybrid_sparse_vector) IS 'Convert a sparse vector to tsvector via the simple text configuration. [experimental public]';
COMMENT ON FUNCTION turbohybrid_sparse_vector_to_tsquery(turbohybrid_sparse_vector) IS 'Convert a sparse vector to an OR tsquery via the simple text configuration. [experimental public]';
COMMENT ON FUNCTION turbohybrid_sparse_vector_term_ids(turbohybrid_sparse_vector) IS 'Return the term ids of a sparse vector. [experimental public]';
COMMENT ON FUNCTION turbohybrid_sparse_vector_weights(turbohybrid_sparse_vector) IS 'Return the weights of a sparse vector. [experimental public]';
COMMENT ON FUNCTION turbohybrid_sparse_vector_count(turbohybrid_sparse_vector) IS 'Return the nonzero term count of a sparse vector. [experimental public]';
COMMENT ON FUNCTION turbohybrid_multivector(vector[]) IS 'Construct a turbohybrid_multivector from an array of same-dimensional vectors. [experimental public]';
COMMENT ON FUNCTION turbohybrid_multivector_from_float4(pg_catalog.float4[], pg_catalog.int4) IS 'Construct a turbohybrid_multivector from flat float values and a token dimension. [experimental public]';
COMMENT ON FUNCTION turbohybrid_multivector_from_contexts(pg_catalog.float4[], pg_catalog.int4, pg_catalog.int4[]) IS 'Construct a context-aware turbohybrid_multivector from flat float values and zero-based context start token offsets. [experimental public]';
COMMENT ON FUNCTION turbohybrid_multivector_from_contexts_and_fields(pg_catalog.float4[], pg_catalog.int4, pg_catalog.int4[], pg_catalog.int4[]) IS 'Construct a context-aware turbohybrid_multivector with one non-negative field id per context. [experimental public]';
COMMENT ON FUNCTION turbohybrid_multivector_dims(turbohybrid_multivector) IS 'Return the per-token dimension of a turbohybrid_multivector. [experimental public]';
COMMENT ON FUNCTION turbohybrid_multivector_count(turbohybrid_multivector) IS 'Return the number of token vectors in a turbohybrid_multivector. [experimental public]';
COMMENT ON FUNCTION turbohybrid_multivector_model_info(pg_catalog.text) IS 'Return registered late-interaction model metadata for multivector validation and benchmark provenance. [diagnostic]';
COMMENT ON FUNCTION turbohybrid_multivector_context_count(turbohybrid_multivector) IS 'Return the number of context windows stored in a turbohybrid_multivector. [experimental public]';
COMMENT ON FUNCTION turbohybrid_multivector_context_offsets(turbohybrid_multivector) IS 'Return zero-based context start token offsets from a turbohybrid_multivector. [experimental public]';
COMMENT ON FUNCTION turbohybrid_multivector_field_ids(turbohybrid_multivector) IS 'Return one field id per context window from a turbohybrid_multivector, defaulting to 0 for flat values. [experimental public]';
COMMENT ON FUNCTION turbohybrid_multivector_subvector(turbohybrid_multivector, pg_catalog.int4) IS 'Return one 1-based subvector from a turbohybrid_multivector as vector. [experimental public]';
COMMENT ON FUNCTION turbohybrid_multivector_to_vector_array(turbohybrid_multivector) IS 'Return all subvectors from a turbohybrid_multivector as vector[]. [experimental public]';
COMMENT ON FUNCTION turbohybrid_multivector_maxsim(turbohybrid_multivector, turbohybrid_multivector) IS 'Exact MaxSim score between a query and document multivector. [experimental public]';
COMMENT ON FUNCTION turbohybrid_multivector_context_maxsim(turbohybrid_multivector, turbohybrid_multivector) IS 'Exact context-level MaxSim: score each document context independently and return the best context score. [experimental public]';
COMMENT ON FUNCTION turbohybrid_multivector_field_weighted_maxsim(turbohybrid_multivector, turbohybrid_multivector, pg_catalog.int4[], pg_catalog.float4[]) IS 'Exact field-weighted MaxSim over context field ids, using one weight per requested field. [experimental public]';
COMMENT ON FUNCTION turbohybrid_multivector_maxsim_scalar(turbohybrid_multivector, turbohybrid_multivector) IS 'Scalar reference MaxSim used to verify SIMD parity. [diagnostic]';
COMMENT ON FUNCTION turbohybrid_multivector_maxsim_blocked_scalar(turbohybrid_multivector, turbohybrid_multivector) IS 'Blocked scalar reference MaxSim used to verify SIMD parity. [diagnostic]';
COMMENT ON FUNCTION turbohybrid_multivector_maxsim_distance(turbohybrid_multivector, turbohybrid_multivector) IS 'Exact MaxSim as a smaller-is-better distance (-MaxSim). [experimental public]';
COMMENT ON FUNCTION turbohybrid_experimental_compact_code_score(pg_catalog.int2[], pg_catalog.int2[], pg_catalog.bool, pg_catalog.text) IS 'Compact-code scoring probe for kernel experiments; not a serving path. [research-only]';
COMMENT ON FUNCTION turbohybrid_multivector_distance(turbohybrid_multivector, turbohybrid_query) IS 'Exact MaxSim distance backing the multivector <~> operator. [internal helper]';
COMMENT ON FUNCTION turbohybrid_sparse_inner_product_distance(turbohybrid_sparse_vector, turbohybrid_query) IS 'Sparse inner-product distance backing the <~*> operator. [internal helper]';
COMMENT ON FUNCTION turbohybrid_sparse_compact(pg_catalog.regclass) IS 'Compact a sparse TurboHybrid index in place, reclaiming dead postings. [experimental public]';
COMMENT ON FUNCTION turbohybrid_multivector_proxy_diagnostics(pg_catalog.regclass, pg_catalog.int4, pg_catalog.int4) IS 'Read-only bounded diagnostic comparing document proxy nearest-neighbor order with exact multivector MaxSim order for sampled document-node multivector indexes. [diagnostic]';
COMMENT ON OPERATOR <~> (turbohybrid_multivector, turbohybrid_query) IS 'MaxSim distance operator for TurboHybrid multivector queries. [experimental public]';
COMMENT ON OPERATOR <~*> (turbohybrid_sparse_vector, turbohybrid_query) IS 'Inner-product distance operator for TurboHybrid sparse queries. [experimental public]';
COMMENT ON OPERATOR CLASS multivector_cosine_turbohybrid_ops USING turbohybrid IS 'Compatibility multivector operator class for normalized token vectors; uses dot-product MaxSim internally. [experimental public]';
COMMENT ON OPERATOR CLASS multivector_maxsim_ip_turbohybrid_ops USING turbohybrid IS 'TurboHybrid multivector operator class exposing raw dot-product MaxSim semantics. [experimental public]';
COMMENT ON OPERATOR CLASS sparse_ip_turbohybrid_ops USING turbohybrid IS 'TurboHybrid learned-sparse inner-product operator class. [experimental public]';
COMMENT ON OPERATOR FAMILY multivector_cosine_turbohybrid_ops USING turbohybrid IS 'Compatibility multivector operator family for normalized token vectors; uses dot-product MaxSim internally. [experimental public]';
COMMENT ON OPERATOR FAMILY multivector_maxsim_ip_turbohybrid_ops USING turbohybrid IS 'TurboHybrid multivector raw dot-product MaxSim operator family. [experimental public]';
COMMENT ON OPERATOR FAMILY sparse_ip_turbohybrid_ops USING turbohybrid IS 'TurboHybrid learned-sparse inner-product operator family. [experimental public]';
COMMENT ON EXTENSION pgturbohybrid_experimental IS 'Experimental learned-sparse and multivector features for pgturbohybrid. [experimental public]';
COMMENT ON FUNCTION turbohybrid_experimental_query(vector, pg_catalog.tsquery, pg_catalog.text, pg_catalog.float8, pg_catalog.float8, pg_catalog.float8, pg_catalog.int4, pg_catalog.int4, pg_catalog.int4, pg_catalog.int4, pg_catalog.bool, turbohybrid_multivector, pg_catalog.float4[], pg_catalog.bool[], pg_catalog.float8, pg_catalog.int4, turbohybrid_sparse_vector, pg_catalog.float8, pg_catalog.int4, pg_catalog.bool) IS 'Construct an experimental TurboHybrid query payload with sparse and multivector retrieval. [experimental public]';
