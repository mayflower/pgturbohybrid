DO $$
BEGIN
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
\pset format unaligned

SELECT turbohybrid_multivector_dims(
  turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector])
) AS dims;

SELECT turbohybrid_multivector_count(
  turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector])
) AS count;

SELECT turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector])::text AS mv_out;

SELECT turbohybrid_multivector_maxsim(
  turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector]),
  turbohybrid_multivector(ARRAY['[1,0]'::vector, '[1,1]'::vector])
) AS maxsim;

SELECT turbohybrid_multivector_maxsim_distance(
  turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector]),
  turbohybrid_multivector(ARRAY['[1,0]'::vector, '[1,1]'::vector])
) AS distance;

SELECT turbohybrid_multivector(ARRAY['[1,0]'::vector, '[1,1]'::vector]) <~>
  turbohybrid_query(multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector]))
  AS operator_distance;

SELECT turbohybrid_multivector_maxsim(
  turbohybrid_multivector(ARRAY['[-1,0]'::vector]),
  turbohybrid_multivector(ARRAY['[2,0]'::vector, '[3,0]'::vector])
) AS negative_maxsim;

CREATE TABLE mv_docs (
  id int,
  colbert turbohybrid_multivector,
  body_tsv tsvector
);

INSERT INTO mv_docs VALUES
  (1, turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0.8,0.2]'::vector]), to_tsvector('alpha beta')),
  (2, turbohybrid_multivector(ARRAY['[0,1]'::vector, '[0.2,0.8]'::vector]), to_tsvector('gamma delta')),
  (3, turbohybrid_multivector(ARRAY['[0.95,0.05]'::vector, '[0,1]'::vector]), to_tsvector('alpha gamma'));

\set VERBOSITY terse

CREATE INDEX mv_docs_colbert_idx ON mv_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops);

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_index_stats('mv_docs_colbert_idx'::regclass);
	IF (stats->>'node_count')::int <> 6 THEN
		RAISE EXCEPTION 'expected multivector build to create 6 graph nodes, got %',
			stats->>'node_count';
	END IF;
END
$$;

SELECT id FROM mv_docs
  ORDER BY colbert <~> turbohybrid_query(
    multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector])
  )
  LIMIT 1;

SET enable_seqscan = off;
SELECT id FROM mv_docs
  ORDER BY colbert <~> turbohybrid_query(
    multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector])
  )
  LIMIT 1;

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_last_scan_stats();
	IF stats->>'index_used' <> 'true' OR
		stats->>'dense_branch_used' <> 'true' OR
		(stats->>'dense_candidates')::int < 1 THEN
		RAISE EXCEPTION 'expected indexed multivector dense scan, got %', stats;
	END IF;
END
$$;

SELECT COUNT(*) AS result_count,
       COUNT(DISTINCT id) AS distinct_docs
FROM (
  SELECT id FROM mv_docs
    ORDER BY colbert <~> turbohybrid_query(
      multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector]),
      dense_k => 6,
      final_k => 3
    )
    LIMIT 3
) s;
RESET enable_seqscan;

SET turbohybrid.multivector_max_doc_vectors = 1;
CREATE INDEX mv_docs_colbert_limited_idx ON mv_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops);
RESET turbohybrid.multivector_max_doc_vectors;

CREATE INDEX mv_docs_bad_order_idx ON mv_docs USING turbohybrid
  (body_tsv bm25_tsvector_turbohybrid_ops, colbert multivector_cosine_turbohybrid_ops);

SELECT turbohybrid_multivector(ARRAY['[1,0]'::vector]) <~>
  turbohybrid_query(vector_query => '[1,0]'::vector);

DROP TABLE mv_docs;

\set VERBOSITY terse

SELECT turbohybrid_multivector(ARRAY[]::vector[]);

SELECT turbohybrid_multivector(ARRAY['[1,0]'::vector, NULL::vector]);

SELECT turbohybrid_multivector(ARRAY['[1,0]'::vector, '[1,0,0]'::vector]);

SELECT turbohybrid_multivector_maxsim(
  turbohybrid_multivector(ARRAY['[1,0]'::vector]),
  turbohybrid_multivector(ARRAY['[1,0,0]'::vector])
);

SELECT 'turbohybrid_multivector(dim=2,count=1,values=[[1,0]])'::turbohybrid_multivector;
