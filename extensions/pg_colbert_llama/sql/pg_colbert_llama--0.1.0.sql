\echo Use "CREATE EXTENSION pg_colbert_llama" to load this file. \quit

DO $pg_colbert_llama_compat$
BEGIN
	IF pg_catalog.current_setting('server_version_num')::pg_catalog.int4 < 140000 THEN
		RAISE EXCEPTION 'pg_colbert_llama requires PostgreSQL 14 or newer'
			USING ERRCODE = '0A000';
	END IF;

	IF NOT EXISTS (
		SELECT 1 FROM pg_catalog.pg_extension WHERE extname = 'vector'
	) THEN
		RAISE EXCEPTION 'pg_colbert_llama requires the vector extension'
			USING ERRCODE = '0A000',
				  HINT = 'Run CREATE EXTENSION vector before CREATE EXTENSION pg_colbert_llama.';
	END IF;

	IF NOT EXISTS (
		SELECT 1 FROM pg_catalog.pg_extension WHERE extname = 'pgturbohybrid'
	) THEN
		RAISE EXCEPTION 'pg_colbert_llama requires the pgturbohybrid extension'
			USING ERRCODE = '0A000',
				  HINT = 'Run CREATE EXTENSION pgturbohybrid before CREATE EXTENSION pg_colbert_llama.';
	END IF;
END
$pg_colbert_llama_compat$;

CREATE FUNCTION colbert(model pg_catalog.text, input pg_catalog.text)
RETURNS pg_catalog.jsonb
AS 'MODULE_PATHNAME', 'pg_colbert_llama_colbert'
LANGUAGE C STABLE STRICT PARALLEL RESTRICTED;

CREATE FUNCTION colbert_vectors(model pg_catalog.text, input pg_catalog.text)
RETURNS vector[]
AS 'MODULE_PATHNAME', 'pg_colbert_llama_colbert_vectors'
LANGUAGE C STABLE STRICT PARALLEL RESTRICTED;

CREATE FUNCTION colbert_float4(model pg_catalog.text, input pg_catalog.text)
RETURNS pg_catalog.float4[]
AS 'MODULE_PATHNAME', 'pg_colbert_llama_colbert_float4'
LANGUAGE C STABLE STRICT PARALLEL RESTRICTED;

CREATE FUNCTION colbert_dim(model pg_catalog.text, input pg_catalog.text)
RETURNS pg_catalog.int4
AS 'MODULE_PATHNAME', 'pg_colbert_llama_colbert_dim'
LANGUAGE C STABLE STRICT PARALLEL RESTRICTED;

CREATE FUNCTION colbert_model_info(model pg_catalog.text)
RETURNS pg_catalog.jsonb
AS 'MODULE_PATHNAME', 'pg_colbert_llama_colbert_model_info'
LANGUAGE C STABLE STRICT PARALLEL RESTRICTED;

CREATE FUNCTION colbert_mv(model pg_catalog.text, input pg_catalog.text)
RETURNS turbohybrid_multivector
AS 'MODULE_PATHNAME', 'pg_colbert_llama_colbert_mv'
LANGUAGE C STABLE STRICT PARALLEL RESTRICTED;
