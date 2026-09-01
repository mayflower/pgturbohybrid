SET client_min_messages = warning;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;
RESET client_min_messages;

WITH values AS (
	SELECT turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector]) AS q,
	       turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector]) AS d
)
SELECT turbohybrid_multivector_dims(q) AS dims,
	   turbohybrid_multivector_count(q) AS tokens,
	   turbohybrid_multivector_subvector(q, 1) AS second_token,
	   turbohybrid_multivector_maxsim(q, d) AS maxsim,
	   turbohybrid_multivector_maxsim_distance(q, d) AS distance
FROM values;

SELECT turbohybrid_multivector_from_float4(
	ARRAY[1,0,0,1]::float4[], 2)::text AS from_float4;

CREATE TABLE passages (
	id integer PRIMARY KEY,
	tokens turbohybrid_multivector NOT NULL
);

INSERT INTO passages VALUES
	(1, turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector])),
	(2, turbohybrid_multivector(ARRAY['[0.8,0.2]'::vector, '[0.2,0.8]'::vector])),
	(3, turbohybrid_multivector(ARRAY['[-1,0]'::vector, '[0,-1]'::vector]));

CREATE INDEX passages_idx ON passages USING turbohybrid
	(tokens multivector_maxsim_ip_turbohybrid_ops)
	WITH (exact_storage = on);

SET enable_seqscan = off;
SELECT array_agg(id ORDER BY distance, id) AS maxsim_order
FROM (
	SELECT id, tokens <~> turbohybrid_multivector_query(
		turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector]), 3, 3) AS distance
	FROM passages
	ORDER BY distance
	LIMIT 3
) ranked;
RESET enable_seqscan;

DROP TABLE passages;
