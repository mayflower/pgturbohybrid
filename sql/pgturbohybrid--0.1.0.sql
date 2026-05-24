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
	require_bm25_match pg_catalog.bool DEFAULT false
) RETURNS turbohybrid_query
	AS 'MODULE_PATHNAME', 'pgturbohybrid_query_constructor'
	LANGUAGE C STABLE PARALLEL SAFE;

CREATE FUNCTION turbohybrid_distance(vector, turbohybrid_query) RETURNS pg_catalog.float8
	AS 'MODULE_PATHNAME', 'pgturbohybrid_distance'
	LANGUAGE C STABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_l2_distance(vector, turbohybrid_query) RETURNS pg_catalog.float8
	AS 'MODULE_PATHNAME', 'pgturbohybrid_l2_distance'
	LANGUAGE C STABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_negative_inner_product(vector, turbohybrid_query) RETURNS pg_catalog.float8
	AS 'MODULE_PATHNAME', 'pgturbohybrid_negative_inner_product'
	LANGUAGE C STABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_cosine_distance(vector, turbohybrid_query) RETURNS pg_catalog.float8
	AS 'MODULE_PATHNAME', 'pgturbohybrid_cosine_distance'
	LANGUAGE C STABLE STRICT PARALLEL SAFE;

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

CREATE FUNCTION turbohybrid_index_stats(pg_catalog.regclass) RETURNS pg_catalog.jsonb
	AS 'MODULE_PATHNAME', 'pgturbohybrid_index_stats'
	LANGUAGE C STABLE STRICT PARALLEL SAFE;

CREATE FUNCTION turbohybrid_last_scan_stats() RETURNS pg_catalog.jsonb
	AS 'MODULE_PATHNAME', 'pgturbohybrid_last_scan_stats'
	LANGUAGE C PARALLEL SAFE;

CREATE FUNCTION turbohybrid_simd_capabilities() RETURNS pg_catalog.jsonb
	AS 'MODULE_PATHNAME', 'pgturbohybrid_simd_capabilities'
	LANGUAGE C STABLE PARALLEL SAFE;

COMMENT ON EXTENSION pgturbohybrid IS 'TurboHybrid dense vector and BM25 search extension for pgvector';
COMMENT ON ACCESS METHOD turbohybrid IS 'TurboHybrid dense vector and BM25 hybrid index access method';
COMMENT ON TYPE turbohybrid_query IS 'TurboHybrid query payload for dense vector and BM25 hybrid search';

COMMENT ON FUNCTION turbohybrid_query_in(pg_catalog.cstring) IS 'Input function for turbohybrid_query';
COMMENT ON FUNCTION turbohybrid_query_out(turbohybrid_query) IS 'Output function for turbohybrid_query';
COMMENT ON FUNCTION turbohybrid_query(vector, pg_catalog.tsquery, pg_catalog.text, pg_catalog.float8, pg_catalog.float8, pg_catalog.float8, pg_catalog.int4, pg_catalog.int4, pg_catalog.int4, pg_catalog.int4, pg_catalog.bool) IS 'Constructs a TurboHybrid query payload';
COMMENT ON FUNCTION turbohybrid_distance(vector, turbohybrid_query) IS 'Default TurboHybrid distance between a vector and a query';
COMMENT ON FUNCTION turbohybrid_l2_distance(vector, turbohybrid_query) IS 'L2 TurboHybrid distance between a vector and a query';
COMMENT ON FUNCTION turbohybrid_negative_inner_product(vector, turbohybrid_query) IS 'Negative inner product TurboHybrid distance between a vector and a query';
COMMENT ON FUNCTION turbohybrid_cosine_distance(vector, turbohybrid_query) IS 'Cosine TurboHybrid distance between a vector and a query';
COMMENT ON FUNCTION turbohybrid_vector_l2_squared_distance(vector, vector) IS 'Squared L2 support function used by TurboHybrid vector opclasses';
COMMENT ON FUNCTION turbohybrid_vector_l2_distance(vector, vector) IS 'L2 support function used by TurboHybrid vector opclasses';
COMMENT ON FUNCTION turbohybrid_vector_negative_inner_product(vector, vector) IS 'Negative inner product support function used by TurboHybrid vector opclasses';
COMMENT ON FUNCTION turbohybrid_vector_cosine_distance(vector, vector) IS 'Cosine support function used by TurboHybrid vector opclasses';
COMMENT ON FUNCTION turbohybrid_vector_norm(vector) IS 'Vector norm support function used by TurboHybrid vector opclasses';
COMMENT ON FUNCTION turbohybrid_handler(pg_catalog.internal) IS 'Index access method handler for TurboHybrid';
COMMENT ON FUNCTION turbohybrid_index_stats(pg_catalog.regclass) IS 'Return stable TurboHybrid index metadata as jsonb';
COMMENT ON FUNCTION turbohybrid_last_scan_stats() IS 'Return stable summary information for the last TurboHybrid scan in this backend as jsonb';
COMMENT ON FUNCTION turbohybrid_simd_capabilities() IS 'Return pgturbohybrid build and architecture SIMD capability information as jsonb';

COMMENT ON OPERATOR <~-> (vector, turbohybrid_query) IS 'L2 distance operator for TurboHybrid vector queries';
COMMENT ON OPERATOR <~#> (vector, turbohybrid_query) IS 'Negative inner product distance operator for TurboHybrid vector queries';
COMMENT ON OPERATOR <~> (vector, turbohybrid_query) IS 'Cosine distance operator for TurboHybrid vector queries';

COMMENT ON OPERATOR CLASS vector_l2_turbohybrid_ops USING turbohybrid IS 'TurboHybrid L2 vector operator class';
COMMENT ON OPERATOR CLASS vector_ip_turbohybrid_ops USING turbohybrid IS 'TurboHybrid inner product vector operator class';
COMMENT ON OPERATOR CLASS vector_cosine_turbohybrid_ops USING turbohybrid IS 'TurboHybrid cosine vector operator class';
COMMENT ON OPERATOR CLASS bm25_tsvector_turbohybrid_ops USING turbohybrid IS 'TurboHybrid BM25 tsvector operator class';

COMMENT ON OPERATOR FAMILY vector_l2_turbohybrid_ops USING turbohybrid IS 'TurboHybrid L2 vector operator family';
COMMENT ON OPERATOR FAMILY vector_ip_turbohybrid_ops USING turbohybrid IS 'TurboHybrid inner product vector operator family';
COMMENT ON OPERATOR FAMILY vector_cosine_turbohybrid_ops USING turbohybrid IS 'TurboHybrid cosine vector operator family';
COMMENT ON OPERATOR FAMILY bm25_tsvector_turbohybrid_ops USING turbohybrid IS 'TurboHybrid BM25 tsvector operator family';
