SET client_min_messages = warning;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;
RESET client_min_messages;

SELECT ('[1,0]'::vector <~> turbohybrid_dense_query('[1,0]'::vector, 10, 20))
	 = ('[1,0]'::vector <~> turbohybrid_query(
		vector_query => '[1,0]'::vector, final_k => 10, dense_k => 20)) AS dense_wrapper;

SELECT turbohybrid_hybrid_query(
		'[1,0]'::vector, to_tsquery('simple', 'postgres'), 10, 20, 30)::text
	 = turbohybrid_query(
		vector_query => '[1,0]'::vector,
		text_query => to_tsquery('simple', 'postgres'),
		final_k => 10, dense_k => 20, bm25_k => 30)::text AS hybrid_wrapper;

WITH q AS (
	SELECT turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector]) AS value
)
SELECT (value <~> turbohybrid_multivector_query(value, 10, 20))
	 = (value <~> turbohybrid_query(
		multivector_query => value, final_k => 10, multivector_k => 20)) AS multivector_wrapper
FROM q;
