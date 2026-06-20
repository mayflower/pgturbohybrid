DO $$
BEGIN
	IF EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'pg_colbert_llama') THEN
		EXECUTE 'DROP EXTENSION pg_colbert_llama CASCADE';
	END IF;
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
CREATE EXTENSION pg_colbert_llama;
\pset format unaligned

SET pg_colbert_llama.expected_dim = 4;

SELECT colbert_model_info('sauerkraut-modern:query')->>'engine' AS engine,
       colbert_model_info('sauerkraut-modern:query')->>'projection_status' AS projection_status;

SELECT colbert('sauerkraut-modern:query', 'test')->>'role' AS role,
       colbert('sauerkraut-modern:query', 'test')->>'dim' AS dim,
       colbert('sauerkraut-modern:query', 'test')->>'count' AS query_count;

SELECT array_length(colbert_vectors('sauerkraut-modern:doc', 'test'), 1) AS doc_vectors,
       (colbert_vectors('sauerkraut-modern:doc', 'test'))[1]::text AS first_vector;

SELECT turbohybrid_multivector_dims(colbert_mv('sauerkraut-modern:query', 'test')) AS query_dims,
       turbohybrid_multivector_count(colbert_mv('sauerkraut-modern:query', 'test')) AS query_count,
       turbohybrid_multivector_count(colbert_mv('sauerkraut-modern:doc', 'test')) AS doc_count;

SELECT array_length(colbert_mv_batch('sauerkraut-modern:doc', ARRAY['alpha', 'beta']), 1) AS batch_count,
       turbohybrid_multivector_dims((colbert_mv_batch('sauerkraut-modern:doc', ARRAY['alpha', 'beta']))[1]) AS batch_dims,
       turbohybrid_multivector_count((colbert_mv_batch('sauerkraut-modern:doc', ARRAY['alpha', 'beta']))[2]) AS second_doc_count;

SELECT turbohybrid_multivector_maxsim(
  colbert_mv('sauerkraut-modern:query', 'test'),
  colbert_mv('sauerkraut-modern:doc', 'test')
) AS maxsim;

SELECT turbohybrid_multivector_from_float4(
  colbert_float4('sauerkraut-modern:query', 'test'),
  colbert_dim('sauerkraut-modern:query', 'test')
)::text AS direct_mv;

SELECT turbohybrid_multivector(
  turbohybrid_multivector_to_vector_array(colbert_mv('sauerkraut-modern:query', 'test'))
)::text AS mv_roundtrip;

CREATE SCHEMA colbert_llama_shadow;
CREATE FUNCTION colbert_llama_shadow.turbohybrid_multivector_from_float4(
  pg_catalog.float4[],
  pg_catalog.int4
) RETURNS turbohybrid_multivector
LANGUAGE plpgsql
AS $$
BEGIN
  RAISE EXCEPTION 'shadow function used';
END
$$;
SET search_path = colbert_llama_shadow, public;
SELECT turbohybrid_multivector_count(colbert_mv('sauerkraut-modern:query', 'test')) AS search_path_safe_count;
RESET search_path;
SET client_min_messages = warning;
DROP SCHEMA colbert_llama_shadow CASCADE;
RESET client_min_messages;

SET pg_colbert_llama.max_query_vectors = 1;
SELECT turbohybrid_multivector_count(colbert_mv('sauerkraut-modern:query', 'test')) AS capped_query_count;
RESET pg_colbert_llama.max_query_vectors;

DO $invalid_models$
DECLARE
	spec record;
BEGIN
	FOR spec IN
		SELECT 'missing_role' AS name, $$SELECT colbert_model_info('sauerkraut-modern')$$ AS sql
		UNION ALL SELECT 'unknown_role', $$SELECT colbert_model_info('sauerkraut-modern:passage')$$
		UNION ALL SELECT 'empty_alias', $$SELECT colbert_model_info(':query')$$
		UNION ALL SELECT 'slash', $$SELECT colbert_model_info('x/y:query')$$
		UNION ALL SELECT 'backslash', $$SELECT colbert_model_info('x\y:query')$$
		UNION ALL SELECT 'dotdot', $$SELECT colbert_model_info('x..y:query')$$
		UNION ALL SELECT 'hidden', $$SELECT colbert_model_info('.hidden:query')$$
	LOOP
		BEGIN
			EXECUTE spec.sql;
			RAISE EXCEPTION 'expected invalid model to fail: %', spec.name;
		EXCEPTION WHEN invalid_parameter_value THEN
			NULL;
		END;
	END LOOP;
END
$invalid_models$;

SET pg_colbert_llama.allowed_models = 'sauerkraut-modern';
SELECT colbert_model_info('sauerkraut-modern:query')->>'alias' AS allowed_alias;

DO $allowed_models$
BEGIN
	BEGIN
		PERFORM colbert_model_info('other-model:query');
		RAISE EXCEPTION 'expected allowed_models to reject other-model';
	EXCEPTION WHEN insufficient_privilege THEN
		NULL;
	END;
END
$allowed_models$;
RESET pg_colbert_llama.allowed_models;

CREATE TABLE pg_colbert_llama_passages (
  id int PRIMARY KEY,
  body text NOT NULL,
  body_tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', body)) STORED,
  colbert turbohybrid_multivector NOT NULL
);

INSERT INTO pg_colbert_llama_passages (id, body, colbert) VALUES
  (1, 'alpha beta', colbert_mv('sauerkraut-modern:doc', 'alpha beta')),
  (2, 'gamma delta', colbert_mv('sauerkraut-modern:doc', 'gamma delta')),
  (3, 'alpha gamma', colbert_mv('sauerkraut-modern:doc', 'alpha gamma'));

CREATE INDEX pg_colbert_llama_passages_idx
ON pg_colbert_llama_passages USING turbohybrid (
  colbert multivector_maxsim_ip_turbohybrid_ops,
  body_tsv bm25_tsvector_turbohybrid_ops
);

SET enable_seqscan = off;
SELECT id AS indexed_id
FROM pg_colbert_llama_passages
ORDER BY colbert <~> turbohybrid_query(
  multivector_query => colbert_mv('sauerkraut-modern:query', 'alpha'),
  text_query => websearch_to_tsquery('simple', 'alpha'),
  fusion => 'rrf',
  dense_k => 20,
  bm25_k => 20,
  final_k => 3
)
LIMIT 3;

SELECT turbohybrid_last_scan_stats()->>'multivector_enabled' AS multivector_enabled,
       turbohybrid_last_scan_stats()->>'index_used' AS index_used;
RESET enable_seqscan;

DROP TABLE pg_colbert_llama_passages;
