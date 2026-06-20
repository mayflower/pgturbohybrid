\echo Use "CREATE EXTENSION llama_embed" to load this file. \quit

DO $llama_embed_compat$
BEGIN
	IF pg_catalog.current_setting('server_version_num')::pg_catalog.int4 < 140000 THEN
		RAISE EXCEPTION 'llama_embed requires PostgreSQL 14 or newer'
			USING ERRCODE = '0A000';
	END IF;

	IF NOT EXISTS (
		SELECT 1 FROM pg_catalog.pg_extension WHERE extname = 'vector'
	) THEN
		RAISE EXCEPTION 'llama_embed requires the vector extension'
			USING ERRCODE = '0A000',
				  HINT = 'Run CREATE EXTENSION vector before CREATE EXTENSION llama_embed.';
	END IF;

	IF NOT EXISTS (
		SELECT 1 FROM pg_catalog.pg_extension WHERE extname = 'pgturbohybrid'
	) THEN
		RAISE EXCEPTION 'llama_embed requires the pgturbohybrid extension'
			USING ERRCODE = '0A000',
				  HINT = 'Run CREATE EXTENSION pgturbohybrid before CREATE EXTENSION llama_embed.';
	END IF;
END
$llama_embed_compat$;

CREATE FUNCTION llama_embed(model pg_catalog.text, input pg_catalog.text, options pg_catalog.jsonb DEFAULT '{}'::pg_catalog.jsonb)
RETURNS pg_catalog.jsonb
AS 'MODULE_PATHNAME', 'pg_colbert_llama_llama_embed'
LANGUAGE C STABLE STRICT PARALLEL RESTRICTED;

CREATE FUNCTION llama_embed_vector(model pg_catalog.text, input pg_catalog.text, options pg_catalog.jsonb DEFAULT '{}'::pg_catalog.jsonb)
RETURNS vector
AS 'MODULE_PATHNAME', 'pg_colbert_llama_llama_embed_vector'
LANGUAGE C STABLE STRICT PARALLEL RESTRICTED;

CREATE FUNCTION llama_embed_vector_batch(model pg_catalog.text, inputs pg_catalog.text[], options pg_catalog.jsonb DEFAULT '{}'::pg_catalog.jsonb)
RETURNS vector[]
AS 'MODULE_PATHNAME', 'pg_colbert_llama_llama_embed_vector_batch'
LANGUAGE C STABLE STRICT PARALLEL RESTRICTED;

CREATE FUNCTION llama_embed_tokens(model pg_catalog.text, input pg_catalog.text, options pg_catalog.jsonb DEFAULT '{}'::pg_catalog.jsonb)
RETURNS vector[]
AS 'MODULE_PATHNAME', 'pg_colbert_llama_llama_embed_tokens'
LANGUAGE C STABLE STRICT PARALLEL RESTRICTED;

CREATE FUNCTION llama_embed_mv(model pg_catalog.text, input pg_catalog.text, options pg_catalog.jsonb DEFAULT '{}'::pg_catalog.jsonb)
RETURNS multivector
AS 'MODULE_PATHNAME', 'pg_colbert_llama_llama_embed_mv'
LANGUAGE C STABLE STRICT PARALLEL RESTRICTED;

CREATE FUNCTION llama_embed_mv_batch(model pg_catalog.text, inputs pg_catalog.text[], options pg_catalog.jsonb DEFAULT '{}'::pg_catalog.jsonb)
RETURNS multivector[]
AS 'MODULE_PATHNAME', 'pg_colbert_llama_llama_embed_mv_batch'
LANGUAGE C STABLE STRICT PARALLEL RESTRICTED;

CREATE FUNCTION llama_embed_sparse(model pg_catalog.text, input pg_catalog.text, options pg_catalog.jsonb DEFAULT '{}'::pg_catalog.jsonb)
RETURNS turbohybrid_sparse_vector
AS 'MODULE_PATHNAME', 'pg_colbert_llama_llama_embed_sparse'
LANGUAGE C STABLE STRICT PARALLEL RESTRICTED;

CREATE FUNCTION llama_embed_sparse_batch(model pg_catalog.text, inputs pg_catalog.text[], options pg_catalog.jsonb DEFAULT '{}'::pg_catalog.jsonb)
RETURNS turbohybrid_sparse_vector[]
AS 'MODULE_PATHNAME', 'pg_colbert_llama_llama_embed_sparse_batch'
LANGUAGE C STABLE STRICT PARALLEL RESTRICTED;

CREATE FUNCTION llama_embed_model_info(model pg_catalog.text)
RETURNS pg_catalog.jsonb
AS 'MODULE_PATHNAME', 'pg_colbert_llama_llama_embed_model_info'
LANGUAGE C STABLE STRICT PARALLEL RESTRICTED;
