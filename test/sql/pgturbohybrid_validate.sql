SET client_min_messages = warning;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;
RESET client_min_messages;

CREATE TABLE validate_dense (id int, embedding vector(3));
INSERT INTO validate_dense
SELECT g, ARRAY[g::real, (g + 1)::real, (g + 2)::real]::vector
FROM generate_series(1, 24) AS g;
CREATE INDEX validate_dense_idx ON validate_dense
USING turbohybrid (embedding vector_l2_turbohybrid_ops)
WITH (exact_storage = on);

DO $$
DECLARE
	shallow jsonb := turbohybrid_validate_index('validate_dense_idx'::regclass);
	deep_result jsonb := turbohybrid_validate_index('validate_dense_idx'::regclass, true);
BEGIN
	IF NOT (shallow->>'ok')::boolean OR shallow->>'scope' <> 'sampled' OR
	   (shallow->>'error_count')::int <> 0 OR
	   shallow->'branches'->'dense'->>'status' <> 'valid' THEN
		RAISE EXCEPTION 'unexpected shallow validation: %', shallow;
	END IF;
	IF NOT (deep_result->>'ok')::boolean OR deep_result->>'scope' <> 'deep' OR
	   (deep_result->>'reachable_live_nodes')::int <> 24 OR
	   (deep_result->>'unreachable_live_nodes')::int <> 0 THEN
		RAISE EXCEPTION 'unexpected deep validation: %', deep_result;
	END IF;
END
$$;

DROP TABLE validate_dense;
