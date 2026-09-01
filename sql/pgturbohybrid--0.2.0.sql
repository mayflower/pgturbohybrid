\echo Use "CREATE EXTENSION pgturbohybrid" to load this file. \quit

DO $pgturbohybrid_compat$
DECLARE
	vector_version pg_catalog.text;
	vector_parts pg_catalog.text[];
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
	IF vector_parts IS NULL OR
	   (vector_parts[1]::pg_catalog.int4 = 0 AND
	    (vector_parts[2]::pg_catalog.int4 < 8 OR
	     (vector_parts[2]::pg_catalog.int4 = 8 AND vector_parts[3]::pg_catalog.int4 < 2))) THEN
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

CREATE FUNCTION turbohybrid_multivector_from_float4(pg_catalog.float4[], pg_catalog.int4)
RETURNS turbohybrid_multivector
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_from_float4'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_multivector_dims(turbohybrid_multivector) RETURNS pg_catalog.int4
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_dims'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_multivector_count(turbohybrid_multivector) RETURNS pg_catalog.int4
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_count'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_multivector_subvector(turbohybrid_multivector, pg_catalog.int4)
RETURNS vector
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_subvector'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_multivector_to_vector_array(turbohybrid_multivector) RETURNS vector[]
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_to_vector_array'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_multivector_maxsim(turbohybrid_multivector, turbohybrid_multivector)
RETURNS pg_catalog.float8
	AS 'MODULE_PATHNAME', 'pgturbohybrid_multivector_maxsim'
	LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_multivector_maxsim_distance(turbohybrid_multivector, turbohybrid_multivector)
RETURNS pg_catalog.float8
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
	query_token_mask pg_catalog.bool[] DEFAULT NULL,
	multivector_weight pg_catalog.float8 DEFAULT 1.0,
	multivector_k pg_catalog.int4 DEFAULT NULL
) RETURNS turbohybrid_query
	AS 'MODULE_PATHNAME', 'pgturbohybrid_query_constructor'
	LANGUAGE C STABLE PARALLEL SAFE;

CREATE FUNCTION turbohybrid_dense_query(
	vector_query vector,
	final_k pg_catalog.int4 DEFAULT NULL,
	dense_k pg_catalog.int4 DEFAULT NULL
) RETURNS turbohybrid_query
	LANGUAGE SQL STABLE PARALLEL SAFE
	RETURN turbohybrid_query(vector_query => vector_query, final_k => final_k, dense_k => dense_k);

CREATE FUNCTION turbohybrid_hybrid_query(
	vector_query vector,
	text_query pg_catalog.tsquery,
	final_k pg_catalog.int4 DEFAULT NULL,
	dense_k pg_catalog.int4 DEFAULT NULL,
	bm25_k pg_catalog.int4 DEFAULT NULL
) RETURNS turbohybrid_query
	LANGUAGE SQL STABLE PARALLEL SAFE
	RETURN turbohybrid_query(
		vector_query => vector_query,
		text_query => text_query,
		final_k => final_k,
		dense_k => dense_k,
		bm25_k => bm25_k);

CREATE FUNCTION turbohybrid_multivector_query(
	multivector_query turbohybrid_multivector,
	final_k pg_catalog.int4 DEFAULT NULL,
	multivector_k pg_catalog.int4 DEFAULT NULL
) RETURNS turbohybrid_query
	LANGUAGE SQL STABLE PARALLEL SAFE
	RETURN turbohybrid_query(
		multivector_query => multivector_query,
		final_k => final_k,
		multivector_k => multivector_k);

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

CREATE FUNCTION turbohybrid_multivector_distance(turbohybrid_multivector, turbohybrid_query)
RETURNS pg_catalog.float8
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
	LEFTARG = turbohybrid_multivector,
	RIGHTARG = turbohybrid_query,
	PROCEDURE = turbohybrid_multivector_distance
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

CREATE OPERATOR CLASS bm25_tsvector_turbohybrid_ops
	FOR TYPE pg_catalog.tsvector USING turbohybrid AS
	STORAGE pg_catalog.tsvector;

CREATE OPERATOR CLASS multivector_maxsim_ip_turbohybrid_ops
	DEFAULT FOR TYPE turbohybrid_multivector USING turbohybrid AS
	OPERATOR 1 <~> (turbohybrid_multivector, turbohybrid_query) FOR ORDER BY pg_catalog.float_ops,
	FUNCTION 1 turbohybrid_vector_negative_inner_product(vector, vector),
	FUNCTION 2 turbohybrid_vector_norm(vector),
	FUNCTION 4 turbohybrid_vector_norm(vector);

CREATE FUNCTION turbohybrid_index_stats(pg_catalog.regclass) RETURNS pg_catalog.jsonb
	AS 'MODULE_PATHNAME', 'pgturbohybrid_index_stats'
	LANGUAGE C STABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_estimate_memory(pg_catalog.regclass) RETURNS pg_catalog.jsonb
	AS 'MODULE_PATHNAME', 'pgturbohybrid_estimate_memory'
	LANGUAGE C STABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_prewarm(pg_catalog.regclass) RETURNS pg_catalog.jsonb
	AS 'MODULE_PATHNAME', 'pgturbohybrid_prewarm'
	LANGUAGE C STRICT PARALLEL RESTRICTED;

CREATE FUNCTION turbohybrid_validate_index(index pg_catalog.regclass, deep pg_catalog.bool DEFAULT false)
RETURNS pg_catalog.jsonb
	AS 'MODULE_PATHNAME', 'pgturbohybrid_validate_index'
	LANGUAGE C STRICT STABLE PARALLEL UNSAFE;

COMMENT ON EXTENSION pgturbohybrid IS
	'Fast dense, BM25, and late-interaction retrieval in one PostgreSQL index access method.';
COMMENT ON TYPE turbohybrid_multivector IS
	'Token vectors for exact MaxSim late-interaction ranking.';
COMMENT ON FUNCTION turbohybrid_multivector_maxsim(turbohybrid_multivector, turbohybrid_multivector) IS
	'Exact MaxSim similarity; larger values rank higher.';
COMMENT ON FUNCTION turbohybrid_multivector_maxsim_distance(turbohybrid_multivector, turbohybrid_multivector) IS
	'Exact MaxSim distance (-MaxSim); smaller values rank first.';
