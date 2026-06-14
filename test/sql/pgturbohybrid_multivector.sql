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
SET turbohybrid.multivector_plain_fallback = off;

SELECT turbohybrid_multivector_dims(
  turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector])
) AS dims;

SELECT turbohybrid_multivector_count(
  turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector])
) AS count;

SELECT turbohybrid_multivector_subvector(
  turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector]),
  1
)::text AS subvector_1;

SELECT turbohybrid_multivector_subvector(
  turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector]),
  2
)::text AS subvector_2;

SELECT turbohybrid_multivector(
  turbohybrid_multivector_to_vector_array(
    turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector])
  )
)::text AS mv_array_roundtrip;

SELECT turbohybrid_multivector_from_float4(ARRAY[1,0,0,1]::real[], 2)::text
  AS mv_float4;

SELECT turbohybrid_multivector(
  turbohybrid_multivector_to_vector_array(
    turbohybrid_multivector_from_float4(ARRAY[1,0,0,1]::real[], 2)
  )
)::text AS mv_float4_roundtrip;

DO $mv_float4$
DECLARE
	spec record;
BEGIN
	FOR spec IN
		SELECT 'dim_zero' AS name,
			   'SELECT turbohybrid_multivector_from_float4(ARRAY[1,0]::real[], 0)' AS sql
		UNION ALL
		SELECT 'length_not_divisible',
			   'SELECT turbohybrid_multivector_from_float4(ARRAY[1,0,1]::real[], 2)'
		UNION ALL
		SELECT 'empty_array',
			   'SELECT turbohybrid_multivector_from_float4(ARRAY[]::real[], 2)'
		UNION ALL
		SELECT 'null_element',
			   'SELECT turbohybrid_multivector_from_float4(ARRAY[1,NULL]::real[], 2)'
		UNION ALL
		SELECT 'nan_element',
			   $$SELECT turbohybrid_multivector_from_float4(ARRAY['NaN'::real,0]::real[], 2)$$
		UNION ALL
		SELECT 'infinite_element',
			   $$SELECT turbohybrid_multivector_from_float4(ARRAY['Infinity'::real,0]::real[], 2)$$
	LOOP
		BEGIN
			EXECUTE spec.sql;
			RAISE EXCEPTION 'expected turbohybrid_multivector_from_float4 case to fail: %',
				spec.name;
		EXCEPTION
			WHEN invalid_parameter_value OR data_exception OR null_value_not_allowed THEN
				NULL;
		END;
	END LOOP;
END
$mv_float4$;

DO $$
DECLARE
	ordinal int;
BEGIN
	FOREACH ordinal IN ARRAY ARRAY[0, 3]
	LOOP
		BEGIN
			PERFORM turbohybrid_multivector_subvector(
				turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector]),
				ordinal
			);
			RAISE EXCEPTION 'expected multivector subvector ordinal to fail: %',
				ordinal;
		EXCEPTION WHEN array_subscript_error THEN
			NULL;
		END;
	END LOOP;
END
$$;

SELECT turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector])::text AS mv_out;

SELECT turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector])::text::turbohybrid_multivector::text
  AS mv_roundtrip;

SELECT t.typsend::regproc::text AS mv_send,
       t.typreceive::regproc::text AS mv_recv
FROM pg_type t
WHERE t.typname = 'turbohybrid_multivector';

SELECT pg_catalog.length(
  turbohybrid_multivector_send(
    turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector])
  )
) AS mv_binary_len;

DO $long_context$
DECLARE
	q turbohybrid_multivector :=
		turbohybrid_multivector_from_float4(ARRAY[1,0,0,1]::real[], 2);
	flat_doc turbohybrid_multivector :=
		turbohybrid_multivector_from_float4(ARRAY[1,0,0,1,0.2,0.2,0.2,0.2]::real[], 2);
	one_context_doc turbohybrid_multivector :=
		turbohybrid_multivector_from_contexts(ARRAY[1,0,0,1,0.2,0.2,0.2,0.2]::real[], 2, ARRAY[0]::int[]);
	two_context_doc turbohybrid_multivector :=
		turbohybrid_multivector_from_contexts(ARRAY[0.2,0,0,0.2,1,0,0,1]::real[], 2, ARRAY[0,2]::int[]);
	field_doc_a turbohybrid_multivector :=
		turbohybrid_multivector_from_contexts_and_fields(ARRAY[1,0,0.8,0]::real[], 2, ARRAY[0,1]::int[], ARRAY[1,2]::int[]);
	field_doc_b turbohybrid_multivector :=
		turbohybrid_multivector_from_contexts_and_fields(ARRAY[0.8,0,1,0]::real[], 2, ARRAY[0,1]::int[], ARRAY[1,2]::int[]);
	roundtrip turbohybrid_multivector;
BEGIN
	IF turbohybrid_multivector_context_count(two_context_doc) <> 2 OR
	   turbohybrid_multivector_context_offsets(two_context_doc) <> ARRAY[0,2]::int[] THEN
		RAISE EXCEPTION 'context metadata was not stored correctly';
	END IF;
	IF turbohybrid_multivector_field_ids(field_doc_a) <> ARRAY[1,2]::int[] THEN
		RAISE EXCEPTION 'field metadata was not stored correctly';
	END IF;
	IF turbohybrid_multivector_context_maxsim(q, one_context_doc) <>
	   turbohybrid_multivector_maxsim(q, flat_doc) THEN
		RAISE EXCEPTION 'single context MaxSim must match flat global MaxSim';
	END IF;
	IF turbohybrid_multivector_context_maxsim(q, two_context_doc) <> 2.0 THEN
		RAISE EXCEPTION 'context-level MaxSim did not choose the best window';
	END IF;
	IF turbohybrid_multivector_field_weighted_maxsim(
		   turbohybrid_multivector_from_float4(ARRAY[1,0]::real[], 2),
		   field_doc_a, ARRAY[1,2]::int[], ARRAY[2,1]::real[]
	   ) <= turbohybrid_multivector_field_weighted_maxsim(
		   turbohybrid_multivector_from_float4(ARRAY[1,0]::real[], 2),
		   field_doc_b, ARRAY[1,2]::int[], ARRAY[2,1]::real[]
	   ) THEN
		RAISE EXCEPTION 'field weights did not affect rank deterministically';
	END IF;
	roundtrip := two_context_doc::text::turbohybrid_multivector;
	IF turbohybrid_multivector_context_offsets(roundtrip) <> ARRAY[0,2]::int[] THEN
		RAISE EXCEPTION 'context metadata text roundtrip failed';
	END IF;
END
$long_context$;

DO $long_context_errors$
DECLARE
	spec record;
BEGIN
	FOR spec IN
		SELECT 'missing_zero' AS name,
			   'SELECT turbohybrid_multivector_from_contexts(ARRAY[1,0,0,1]::real[], 2, ARRAY[1]::int[])' AS sql
		UNION ALL
		SELECT 'duplicate_offset',
			   'SELECT turbohybrid_multivector_from_contexts(ARRAY[1,0,0,1]::real[], 2, ARRAY[0,0]::int[])'
		UNION ALL
		SELECT 'field_count_mismatch',
			   'SELECT turbohybrid_multivector_from_contexts_and_fields(ARRAY[1,0,0,1]::real[], 2, ARRAY[0,1]::int[], ARRAY[1]::int[])'
	LOOP
		BEGIN
			EXECUTE spec.sql;
			RAISE EXCEPTION 'expected context constructor case to fail: %',
				spec.name;
		EXCEPTION
			WHEN invalid_parameter_value OR data_exception OR array_subscript_error THEN
				NULL;
		END;
	END LOOP;
END
$long_context_errors$;

CREATE TEMP TABLE th_mv_context_docs (
	id int,
	embedding turbohybrid_multivector
);
INSERT INTO th_mv_context_docs VALUES
	(1, turbohybrid_multivector_from_contexts_and_fields(ARRAY[1,0,0,1]::real[], 2, ARRAY[0,1]::int[], ARRAY[1,2]::int[])),
	(2, turbohybrid_multivector_from_contexts_and_fields(ARRAY[0.8,0,0,0.8]::real[], 2, ARRAY[0,1]::int[], ARRAY[1,2]::int[]));
CREATE INDEX th_mv_context_docs_idx ON th_mv_context_docs
	USING turbohybrid (embedding multivector_maxsim_ip_turbohybrid_ops)
	WITH (multivector_graph = 'document_nodes',
		  multivector_context_mode = 'context_level',
		  multivector_field_mode = 'weighted');

DO $long_context_index$
DECLARE
	stats jsonb := turbohybrid_index_stats('th_mv_context_docs_idx'::regclass);
	top_id int;
BEGIN
	IF stats->>'multivector_context_mode' <> 'context_level' OR
	   stats->>'multivector_field_mode' <> 'weighted' THEN
		RAISE EXCEPTION 'context/field reloptions missing from stats: %', stats;
	END IF;
	SET LOCAL turbohybrid.multivector_candidate_source = 'exact_doc_scan';
	SELECT id INTO top_id
	FROM th_mv_context_docs
	ORDER BY embedding <~> turbohybrid_query(
		multivector_query => turbohybrid_multivector_from_float4(ARRAY[0,1]::real[], 2),
		final_k => 1
	)
	LIMIT 1;
	IF top_id <> 1 THEN
		RAISE EXCEPTION 'context-level indexed rerank returned %, expected 1', top_id;
	END IF;
END
$long_context_index$;

DO $long_context_pooling$
BEGIN
	BEGIN
		EXECUTE $sql$
			CREATE INDEX th_mv_context_docs_pool_idx ON th_mv_context_docs
				USING turbohybrid (embedding multivector_maxsim_ip_turbohybrid_ops)
				WITH (multivector_graph = 'document_nodes',
					  multivector_context_mode = 'context_level',
					  multivector_token_pooling = 'greedy_cosine',
					  multivector_token_pooling_target_ratio = 0.5,
					  multivector_token_pooling_min_tokens = 1)
		$sql$;
		RAISE EXCEPTION 'expected context-aware token pooling to fail';
	EXCEPTION
		WHEN feature_not_supported THEN
			NULL;
	END;
END
$long_context_pooling$;

CREATE TEMP TABLE th_mv_context_mvcc_docs (
	id int,
	embedding turbohybrid_multivector
);
INSERT INTO th_mv_context_mvcc_docs VALUES
	(1, turbohybrid_multivector_from_contexts(ARRAY[1,0,0.1,0]::real[], 2, ARRAY[0,1]::int[])),
	(2, turbohybrid_multivector_from_contexts(ARRAY[0.1,0,0,1]::real[], 2, ARRAY[0,1]::int[])),
	(3, turbohybrid_multivector_from_contexts(ARRAY[0.2,0,0,0.8]::real[], 2, ARRAY[0,1]::int[]));
CREATE INDEX th_mv_context_mvcc_docs_idx ON th_mv_context_mvcc_docs
	USING turbohybrid (embedding multivector_maxsim_ip_turbohybrid_ops)
	WITH (multivector_graph = 'document_nodes',
		  multivector_context_mode = 'context_level');
SET enable_seqscan = off;
SET turbohybrid.multivector_candidate_source = 'exact_doc_scan';
DO $long_context_mvcc_before$
DECLARE
	top_id int;
BEGIN
	SELECT id INTO top_id
	FROM th_mv_context_mvcc_docs
	ORDER BY embedding <~> turbohybrid_query(
		multivector_query => turbohybrid_multivector_from_float4(ARRAY[0,1]::real[], 2),
		final_k => 1
	)
	LIMIT 1;
	IF top_id <> 2 THEN
		RAISE EXCEPTION 'context-level MVCC precheck returned %, expected 2', top_id;
	END IF;
END
$long_context_mvcc_before$;
UPDATE th_mv_context_mvcc_docs
SET embedding = turbohybrid_multivector_from_contexts(ARRAY[1,0,0.1,0]::real[], 2, ARRAY[0,1]::int[])
WHERE id = 2;
DELETE FROM th_mv_context_mvcc_docs WHERE id = 1;
VACUUM th_mv_context_mvcc_docs;
DO $long_context_mvcc_after$
DECLARE
	top_id int;
BEGIN
	SELECT id INTO top_id
	FROM th_mv_context_mvcc_docs
	ORDER BY embedding <~> turbohybrid_query(
		multivector_query => turbohybrid_multivector_from_float4(ARRAY[0,1]::real[], 2),
		final_k => 1
	)
	LIMIT 1;
	IF top_id <> 3 THEN
		RAISE EXCEPTION 'context-level MVCC postcheck returned %, expected 3', top_id;
	END IF;
END
$long_context_mvcc_after$;
RESET turbohybrid.multivector_candidate_source;
RESET enable_seqscan;

SELECT turbohybrid_multivector_dims(
  'turbohybrid_multivector(dim=2,count=1,values=[[1,0]])'::turbohybrid_multivector
) AS literal_dims;

SELECT ' turbohybrid_multivector ( dim = 2 , count = 1 , values = [ [ 1 , 0 ] ] ) '::turbohybrid_multivector::text
  AS mv_whitespace;

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

DO $$
DECLARE
	message text;
	detail text;
BEGIN
	BEGIN
		PERFORM turbohybrid_multivector(ARRAY['[1,0]'::vector]) <~>
			turbohybrid_query(
				multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
				text_query => to_tsquery('alpha')
			);
		RAISE EXCEPTION 'expected scalar multivector text fallback error';
	EXCEPTION WHEN feature_not_supported THEN
		GET STACKED DIAGNOSTICS message = MESSAGE_TEXT,
								detail = PG_EXCEPTION_DETAIL;
		IF message <> 'hybrid text queries require a turbohybrid index scan' OR
			detail <> 'Scalar multivector MaxSim can only evaluate the multivector payload.' THEN
			RAISE EXCEPTION 'unexpected scalar multivector fallback diagnostics: %, %',
				message, detail;
		END IF;
	END;
END
$$;

SELECT turbohybrid_multivector_maxsim(
  turbohybrid_multivector(ARRAY['[-1,0]'::vector]),
  turbohybrid_multivector(ARRAY['[2,0]'::vector, '[3,0]'::vector])
) AS negative_maxsim;

CREATE TABLE sv_docs (
  id int,
  embedding vector(2)
);

INSERT INTO sv_docs VALUES
  (1, '[1,0]'::vector),
  (2, '[0,1]'::vector);

CREATE INDEX sv_docs_embedding_idx ON sv_docs USING turbohybrid
  (embedding vector_cosine_turbohybrid_ops);

SET enable_seqscan = off;
SELECT id FROM sv_docs
  ORDER BY embedding <~> turbohybrid_query(vector_query => '[1,0]'::vector)
  LIMIT 1;

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_last_scan_stats();
	IF stats->>'multivector_enabled' <> 'false' OR
		(stats->>'multivector_query_vectors')::int <> 0 OR
		(stats->>'multivector_subvector_searches')::int <> 0 OR
		stats->>'multivector_exact_kernel' IS NOT NULL OR
		stats->>'multivector_accumulator_kind' IS NOT NULL THEN
		RAISE EXCEPTION 'expected single-vector scan to clear multivector stats, got %', stats;
	END IF;
END
$$;
RESET enable_seqscan;
DROP TABLE sv_docs;

CREATE TABLE mv_token_limit_docs (
  id int,
  colbert turbohybrid_multivector
);

INSERT INTO mv_token_limit_docs VALUES
  (1, turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector]));

CREATE INDEX mv_token_limit_docs_idx ON mv_token_limit_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops);

SET enable_seqscan = off;
SET turbohybrid.multivector_unique_docs_per_token = 1;
SET turbohybrid.multivector_max_raw_hits_per_token = 10;
SELECT COUNT(*) AS result_count,
       COUNT(DISTINCT id) AS distinct_docs
FROM (
  SELECT id FROM mv_token_limit_docs
    ORDER BY colbert <~> turbohybrid_query(
      multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector]),
      dense_k => 4,
      final_k => 4
    )
    LIMIT 4
) s;

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_last_scan_stats();
	IF (stats->>'multivector_subvector_searches')::int <> 2 OR
		(stats->>'multivector_proxy_graph_searches')::int <> 0 OR
		(stats->>'multivector_unique_docs')::int <> 2 OR
		(stats->>'multivector_doc_candidates')::int <> 1 OR
		stats->>'multivector_admission_debug_enabled' <> 'false' OR
		stats ? 'multivector_admission_trace' THEN
		RAISE EXCEPTION 'expected per-token unique doc accounting, got %', stats;
	END IF;
END
$$;

DO $$
DECLARE
	stats jsonb;
BEGIN
	PERFORM set_config('turbohybrid.multivector_debug_admission', 'summary', false);
	PERFORM id FROM mv_token_limit_docs
	  ORDER BY colbert <~> turbohybrid_query(
	    multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector]),
	    dense_k => 4,
	    final_k => 4
	  )
	  LIMIT 4;
	stats := turbohybrid_last_scan_stats();
	IF stats->>'multivector_admission_debug_enabled' <> 'true' OR
		(stats->>'multivector_admission_candidates_before_rerank')::int <> 1 OR
		(stats->>'multivector_admission_candidates_after_truncation')::int <> 1 OR
		(stats->>'multivector_admission_exact_rerank_docs')::int <> 1 OR
		stats->>'multivector_admission_truncated_by_doc_candidate_k' <> 'false' OR
		stats->>'multivector_admission_truncated_by_accumulator_memory' <> 'false' OR
		stats->>'multivector_admission_trace_available' <> 'false' OR
		stats ? 'multivector_admission_trace' THEN
		RAISE EXCEPTION 'expected summary admission diagnostics without trace, got %', stats;
	END IF;
	PERFORM set_config('turbohybrid.multivector_debug_admission', 'off', false);
END
$$;

DO $$
DECLARE
	stats jsonb;
	trace jsonb;
	first_entry jsonb;
BEGIN
	PERFORM set_config('turbohybrid.multivector_debug_admission', 'trace', false);
	PERFORM set_config('turbohybrid.multivector_debug_trace_limit', '1', false);
	PERFORM id FROM mv_token_limit_docs
	  ORDER BY colbert <~> turbohybrid_query(
	    multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector]),
	    dense_k => 4,
	    final_k => 4
	  )
	  LIMIT 4;
	stats := turbohybrid_last_scan_stats();
	trace := stats->'multivector_admission_trace';
	first_entry := trace->0;
	IF stats->>'multivector_admission_debug_enabled' <> 'true' OR
		stats->>'multivector_admission_trace_available' <> 'true' OR
		jsonb_typeof(trace) <> 'array' OR
		jsonb_array_length(trace) <> 1 OR
		NOT first_entry ? 'doc_id' OR
		NOT first_entry ? 'heap_tid' OR
		(first_entry->>'query_token_coverage_count')::int <> 2 OR
		(first_entry->>'raw_hit_count')::int <> 2 OR
		(first_entry->>'retained_for_exact_rerank') <> 'true' OR
		first_entry->>'exact_rerank_score' IS NULL THEN
		RAISE EXCEPTION 'expected bounded document-keyed admission trace, got %', stats;
	END IF;
	PERFORM set_config('turbohybrid.multivector_debug_trace_limit', '1000', false);
	PERFORM set_config('turbohybrid.multivector_debug_admission', 'off', false);
END
$$;
RESET turbohybrid.multivector_max_raw_hits_per_token;
RESET turbohybrid.multivector_unique_docs_per_token;
RESET enable_seqscan;
DROP TABLE mv_token_limit_docs;

CREATE TABLE mv_adaptive_docs (
  id int,
  colbert turbohybrid_multivector
);

INSERT INTO mv_adaptive_docs VALUES
  (1, turbohybrid_multivector(ARRAY['[1,0]'::vector])),
  (2, turbohybrid_multivector(ARRAY['[0.95,0.05]'::vector]));

CREATE INDEX mv_adaptive_docs_idx ON mv_adaptive_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops);

SET enable_seqscan = off;
SET turbohybrid.multivector_subvector_k = 1;
SET turbohybrid.multivector_unique_docs_per_token = 2;
SET turbohybrid.multivector_max_raw_hits_per_token = 2;
SET turbohybrid.multivector_adaptive_widening = on;
SELECT COUNT(*) AS adaptive_result_count
FROM (
  SELECT id FROM mv_adaptive_docs
    ORDER BY colbert <~> turbohybrid_query(
      multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
      dense_k => 2,
      final_k => 2
    )
    LIMIT 2
) s;

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_last_scan_stats();
	IF stats->>'multivector_adaptive_widening_triggered' <> 'true' OR
		(stats->>'multivector_adaptive_initial_raw_target')::int <> 1 OR
		(stats->>'multivector_adaptive_final_raw_target')::int <> 2 OR
		(stats->>'multivector_raw_subvector_hits')::int > 2 THEN
		RAISE EXCEPTION 'expected adaptive multivector widening to reach cap, got %', stats;
	END IF;
END
$$;

SET turbohybrid.multivector_adaptive_widening = off;
SELECT COUNT(*) AS fixed_result_count
FROM (
  SELECT id FROM mv_adaptive_docs
    ORDER BY colbert <~> turbohybrid_query(
      multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
      dense_k => 2,
      final_k => 2
    )
    LIMIT 2
) s;

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_last_scan_stats();
	IF stats->>'multivector_adaptive_widening_triggered' <> 'false' OR
		(stats->>'multivector_adaptive_initial_raw_target')::int <> 2 OR
		(stats->>'multivector_adaptive_final_raw_target')::int <> 2 THEN
		RAISE EXCEPTION 'expected fixed multivector raw target when adaptive widening is off, got %', stats;
	END IF;
END
$$;
RESET turbohybrid.multivector_adaptive_widening;
RESET turbohybrid.multivector_max_raw_hits_per_token;
RESET turbohybrid.multivector_unique_docs_per_token;
RESET turbohybrid.multivector_subvector_k;
RESET enable_seqscan;
DROP TABLE mv_adaptive_docs;

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
		(stats->>'dense_candidates')::int < 1 OR
		stats->>'exact_rescore_source' <> 'heap' OR
		(stats->>'heap_rescore_count')::int < 1 OR
		stats->>'multivector_enabled' <> 'true' OR
		(stats->>'multivector_query_vectors')::int <> 1 OR
		(stats->>'multivector_doc_vectors_limit')::int < 1 OR
		(stats->>'multivector_subvector_searches')::int <> 1 OR
		(stats->>'multivector_raw_subvector_hits')::int < 1 OR
		(stats->>'multivector_unique_docs')::int < 1 OR
		(stats->>'multivector_maxsim_updates')::int < 1 OR
		(stats->>'multivector_doc_candidates')::int < 1 OR
		stats->>'multivector_exact_rerank_enabled' <> 'true' OR
		(stats->>'multivector_exact_rerank_docs')::int < 1 OR
		(stats->>'multivector_exact_rerank_pairs')::int < 1 OR
		stats->>'multivector_exact_kernel' IS NULL OR
		stats->>'multivector_accumulator_kind' <> 'docid_hash_slab' OR
		stats->>'multivector_docmap_source' <> 'sidecar' OR
		(stats->>'multivector_docmap_bytes')::bigint < 1 OR
		(stats->>'multivector_memory_bytes_estimate')::bigint < 1 OR
		(stats->>'multivector_memory_bytes_estimate')::bigint > 1048576 THEN
		RAISE EXCEPTION 'expected indexed multivector dense scan, got %', stats;
	END IF;
END
$$;

DO $$
DECLARE
	memory_stats jsonb;
BEGIN
	memory_stats := turbohybrid_estimate_memory('mv_docs_colbert_idx'::regclass);
	IF (memory_stats->'native'->>'multivector_docmap_bytes')::bigint < 1 THEN
		RAISE EXCEPTION 'expected multivector docmap bytes in memory estimate, got %',
			memory_stats;
	END IF;
END
$$;

SET turbohybrid.multivector_docmap = off;
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
	IF stats->>'multivector_docmap_source' <> 'heap_tid_hash' OR
		(stats->>'multivector_docmap_bytes')::bigint <> 0 THEN
		RAISE EXCEPTION 'expected forced heap-TID docmap fallback, got %', stats;
	END IF;
END
$$;

SET turbohybrid.multivector_docmap = require;
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
	IF stats->>'multivector_docmap_source' <> 'sidecar' OR
		(stats->>'multivector_docmap_bytes')::bigint < 1 THEN
		RAISE EXCEPTION 'expected required multivector docmap sidecar, got %',
			stats;
	END IF;
END
$$;
RESET turbohybrid.multivector_docmap;

SET turbohybrid.default_dense_k = 7;
SELECT id FROM mv_docs
  ORDER BY colbert <~> turbohybrid_query(
    multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
    dense_k => NULL
  )
  LIMIT 1;

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_last_scan_stats();
	IF stats->>'multivector_enabled' <> 'true' OR
		(stats->>'dense_candidates_effective')::int <> 7 THEN
		RAISE EXCEPTION 'expected multivector default dense_k to use dense budget, got %',
			stats;
	END IF;
END
$$;
RESET turbohybrid.default_dense_k;

SET turbohybrid.multivector_exact_rerank = off;
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
		stats->>'exact_rescore_source' <> 'none' OR
		(stats->>'heap_rescore_count')::int <> 0 OR
		stats->>'multivector_exact_rerank_enabled' <> 'false' OR
		(stats->>'multivector_exact_rerank_docs')::int <> 0 OR
		(stats->>'multivector_exact_rerank_pairs')::int <> 0 OR
		stats->>'multivector_exact_kernel' IS NOT NULL THEN
		RAISE EXCEPTION 'expected multivector exact rerank off stats, got %', stats;
	END IF;
END
$$;
RESET turbohybrid.multivector_exact_rerank;

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

SET turbohybrid.multivector_doc_candidate_k = 2;
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

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_last_scan_stats();
	IF (stats->>'multivector_doc_candidates')::int <> 2 THEN
		RAISE EXCEPTION 'expected bounded multivector candidate heap to retain 2 docs, got %',
			stats;
	END IF;
END
$$;
RESET turbohybrid.multivector_doc_candidate_k;
RESET enable_seqscan;

CREATE TABLE mv_alias_cosine_docs (
  id int,
  colbert turbohybrid_multivector
);

CREATE TABLE mv_alias_ip_docs (
  id int,
  colbert turbohybrid_multivector
);

INSERT INTO mv_alias_cosine_docs VALUES
  (1, turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0.8,0.2]'::vector])),
  (2, turbohybrid_multivector(ARRAY['[0,1]'::vector, '[0.2,0.8]'::vector])),
  (3, turbohybrid_multivector(ARRAY['[0.95,0.05]'::vector, '[0,1]'::vector]));

INSERT INTO mv_alias_ip_docs SELECT * FROM mv_alias_cosine_docs;

CREATE INDEX mv_alias_cosine_idx ON mv_alias_cosine_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops);

CREATE INDEX mv_alias_ip_idx ON mv_alias_ip_docs USING turbohybrid
  (colbert multivector_maxsim_ip_turbohybrid_ops);

SET enable_seqscan = off;
WITH cosine_results AS (
  SELECT pg_catalog.array_agg(id) AS ids
  FROM (
    SELECT id FROM mv_alias_cosine_docs
      ORDER BY colbert <~> turbohybrid_query(
        multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector]),
        dense_k => 6,
        final_k => 3
      )
      LIMIT 3
  ) s
),
ip_results AS (
  SELECT pg_catalog.array_agg(id) AS ids
  FROM (
    SELECT id FROM mv_alias_ip_docs
      ORDER BY colbert <~> turbohybrid_query(
        multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector]),
        dense_k => 6,
        final_k => 3
      )
      LIMIT 3
  ) s
)
SELECT cosine_results.ids = ip_results.ids AS maxsim_ip_alias_same_order
FROM cosine_results, ip_results;
RESET enable_seqscan;
DROP TABLE mv_alias_cosine_docs;
DROP TABLE mv_alias_ip_docs;

CREATE TABLE mv_recall_docs (
  id int,
  colbert turbohybrid_multivector
);

INSERT INTO mv_recall_docs VALUES
  (1, turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector])),
  (2, turbohybrid_multivector(ARRAY['[0.9,0.1]'::vector, '[0.1,0.9]'::vector])),
  (3, turbohybrid_multivector(ARRAY['[0.8,0.2]'::vector, '[0.3,0.7]'::vector])),
  (4, turbohybrid_multivector(ARRAY['[0.6,0.4]'::vector, '[0.2,0.6]'::vector])),
  (5, turbohybrid_multivector(ARRAY['[-1,0]'::vector, '[0,-1]'::vector])),
  (6, turbohybrid_multivector(ARRAY['[0.4,0.6]'::vector, '[0.5,0.5]'::vector]));

CREATE INDEX mv_recall_docs_idx ON mv_recall_docs USING turbohybrid
  (colbert multivector_maxsim_ip_turbohybrid_ops);

SET enable_seqscan = off;
SET turbohybrid.multivector_doc_candidate_k = 8;
SET turbohybrid.multivector_unique_docs_per_token = 8;
SET turbohybrid.multivector_max_raw_hits_per_token = 64;

WITH q AS (
  SELECT turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector]) AS mv
),
brute AS (
  SELECT pg_catalog.array_agg(id ORDER BY dist, id) AS ids
  FROM (
    SELECT id, turbohybrid_multivector_maxsim_distance(q.mv, colbert) AS dist
    FROM mv_recall_docs, q
    ORDER BY dist, id
    LIMIT 5
  ) s
),
indexed AS (
  SELECT pg_catalog.array_agg(id ORDER BY ord) AS ids
  FROM (
    SELECT id, row_number() OVER () AS ord
    FROM mv_recall_docs, q
    ORDER BY colbert <~> turbohybrid_query(
      multivector_query => q.mv,
      dense_k => 12,
      final_k => 5
    )
    LIMIT 5
  ) s
)
SELECT brute.ids = indexed.ids AS multivector_index_matches_bruteforce
FROM brute, indexed;

RESET turbohybrid.multivector_max_raw_hits_per_token;
RESET turbohybrid.multivector_unique_docs_per_token;
RESET turbohybrid.multivector_doc_candidate_k;
RESET enable_seqscan;
DROP TABLE mv_recall_docs;

CREATE TABLE mv_insert_batch_dense_docs (
  id int,
  colbert turbohybrid_multivector
);

INSERT INTO mv_insert_batch_dense_docs VALUES
  (1, turbohybrid_multivector(ARRAY['[0,1]'::vector]));

CREATE INDEX mv_insert_batch_dense_idx ON mv_insert_batch_dense_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops);

INSERT INTO mv_insert_batch_dense_docs VALUES
  (2, turbohybrid_multivector(ARRAY[
    '[1,0]'::vector,
    '[0.8,0.2]'::vector,
    '[0.6,0.4]'::vector
  ]));

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_index_stats('mv_insert_batch_dense_idx'::regclass);
	IF (stats->>'node_count')::int <> 4 THEN
		RAISE EXCEPTION 'expected 3-token multivector insert to append three dense nodes, got %',
			stats->>'node_count';
	END IF;
END
$$;

SET enable_seqscan = off;
SELECT id FROM mv_insert_batch_dense_docs
  ORDER BY colbert <~> turbohybrid_query(
    multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
    dense_k => 4,
    final_k => 1
  )
  LIMIT 1;
RESET enable_seqscan;
DROP TABLE mv_insert_batch_dense_docs;

CREATE TABLE mv_insert_batch_hybrid_docs (
  id int,
  colbert turbohybrid_multivector,
  body_tsv tsvector
);

INSERT INTO mv_insert_batch_hybrid_docs VALUES
  (1, turbohybrid_multivector(ARRAY['[0,1]'::vector]), to_tsvector('alpha'));

CREATE INDEX mv_insert_batch_hybrid_idx ON mv_insert_batch_hybrid_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops,
   body_tsv bm25_tsvector_turbohybrid_ops);

INSERT INTO mv_insert_batch_hybrid_docs VALUES
  (2, turbohybrid_multivector(ARRAY[
    '[1,0]'::vector,
    '[0.8,0.2]'::vector,
    '[0.6,0.4]'::vector
  ]), to_tsvector('batchterm'));

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_index_stats('mv_insert_batch_hybrid_idx'::regclass);
	IF (stats->>'node_count')::int <> 4 THEN
		RAISE EXCEPTION 'expected 3-token hybrid multivector insert to append three dense nodes, got %',
			stats->>'node_count';
	END IF;
END
$$;

SET enable_seqscan = off;
SELECT id FROM mv_insert_batch_hybrid_docs
	ORDER BY colbert <~> turbohybrid_query(
		multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
		text_query => to_tsquery('batchterm'),
		dense_k => 4,
		bm25_k => 4,
		final_k => 4,
		fusion => 'rrf'
	)
	LIMIT 4;
RESET enable_seqscan;
DROP TABLE mv_insert_batch_hybrid_docs;

CREATE TABLE mv_bm25_injection_docs (
  id text PRIMARY KEY,
  colbert turbohybrid_multivector,
  body_tsv tsvector
);

INSERT INTO mv_bm25_injection_docs VALUES
  ('spike', turbohybrid_multivector(ARRAY['[1,0]'::vector]), to_tsvector('simple', 'spikeonly')),
  ('spike_y', turbohybrid_multivector(ARRAY['[0,1]'::vector]), to_tsvector('simple', 'spikeonly')),
  ('good', turbohybrid_multivector(ARRAY[
    '[0.8,0.2]'::vector,
    '[0.2,0.8]'::vector
  ]), to_tsvector('simple', 'needle')),
  ('weak_needle', turbohybrid_multivector(ARRAY['[-1,0]'::vector]), to_tsvector('simple', 'needle'));

CREATE INDEX mv_bm25_injection_idx ON mv_bm25_injection_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops,
   body_tsv bm25_tsvector_turbohybrid_ops)
  WITH (graph_m = 4, graph_ef_construction = 8, graph_ef_search = 8);
ANALYZE mv_bm25_injection_docs;

SET enable_seqscan = off;
SET turbohybrid.multivector_plain_fallback = off;
SET turbohybrid.multivector_candidate_source = exact_token_scan;
SET turbohybrid.multivector_candidate_reservoirs = off;
SET turbohybrid.multivector_subvector_k = 1;
SET turbohybrid.multivector_unique_docs_per_token = 1;
SET turbohybrid.multivector_max_raw_hits_per_token = 1;
SET turbohybrid.multivector_doc_candidate_k = 1;
SET turbohybrid.multivector_exact_rerank_k = 1;

DO $$
DECLARE
	q turbohybrid_multivector := turbohybrid_multivector(ARRAY[
		'[1,0]'::vector,
		'[0,1]'::vector
	]);
	top_id text;
	stats jsonb;
BEGIN
	SET LOCAL turbohybrid.multivector_bm25_candidate_injection = off;
	SELECT id INTO top_id
	FROM mv_bm25_injection_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  text_query => to_tsquery('simple', 'needle'),
	  dense_k => 1,
	  bm25_k => 2,
	  final_k => 1,
	  bm25_weight => 0
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();
	IF top_id NOT IN ('spike', 'spike_y') OR
		stats->>'multivector_bm25_injection_enabled' <> 'false' THEN
		RAISE EXCEPTION 'expected BM25 injection off to leave dense miss in place, top %, stats %',
			top_id, stats;
	END IF;

	SET LOCAL turbohybrid.multivector_bm25_candidate_injection = dense_with_text;
	SET LOCAL turbohybrid.multivector_branch_plan = dense_only;
	SELECT id INTO top_id
	FROM mv_bm25_injection_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  text_query => to_tsquery('simple', 'needle'),
	  dense_k => 1,
	  bm25_k => 2,
	  final_k => 1,
	  bm25_weight => 0
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();
	IF top_id <> 'good' OR
		stats->>'multivector_bm25_injection_enabled' <> 'true' OR
		(stats->>'multivector_bm25_injection_candidates')::int < 1 OR
		(stats->>'multivector_bm25_injection_retained')::int < 1 OR
		(stats->>'multivector_bm25_injection_exact_reranked')::int < 1 OR
		stats->>'multivector_branch_plan' <> 'dense_only' OR
		stats->>'branch_fusion_mode' <> 'dense_only' OR
		(stats->>'branch_count')::int <> 1 OR
		stats->'branch_kinds' <> '["token_nodes"]'::jsonb OR
		(stats->'branch_source_flags'->>0)::int <> 7 THEN
		RAISE EXCEPTION 'expected dense_with_text BM25 injection to admit exact-MaxSim winner, top %, stats %',
			top_id, stats;
	END IF;
	IF stats->>'fusion_strategy' NOT IN ('sorted_merge_doc', 'hash_doc') THEN
		RAISE EXCEPTION 'expected document-keyed fusion after injection, got %',
			stats;
	END IF;

	SET LOCAL turbohybrid.multivector_bm25_candidate_injection = hybrid_only;
	SET LOCAL turbohybrid.multivector_branch_plan = qdrant_like;
	SELECT id INTO top_id
	FROM mv_bm25_injection_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  text_query => to_tsquery('simple', 'needle'),
	  dense_k => 1,
	  bm25_k => 2,
	  final_k => 2,
	  fusion => 'rrf'
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();
	IF stats->>'multivector_bm25_injection_enabled' <> 'true' OR
		stats->>'fusion_strategy' NOT IN ('sorted_merge_doc', 'hash_doc') OR
		stats->>'multivector_branch_plan' <> 'qdrant_like' OR
		stats->>'branch_fusion_mode' <> 'qdrant_like_rrf' OR
		(stats->>'branch_count')::int <> 3 OR
		stats->'branch_kinds' <> '["token_nodes", "bm25", "exact_doc_scan"]'::jsonb OR
		stats->'branch_ranks' <> '[1, 2, 3]'::jsonb OR
		(stats->'branch_candidate_counts'->>2)::int < 1 THEN
		RAISE EXCEPTION 'expected hybrid_only injection to preserve document-keyed RRF, top %, stats %',
			top_id, stats;
	END IF;

	SET LOCAL turbohybrid.multivector_bm25_candidate_injection = off;
	SET LOCAL turbohybrid.multivector_sparse_candidate_source = learned_sparse;
	SET LOCAL turbohybrid.multivector_branch_plan = dense_only;
	SELECT id INTO top_id
	FROM mv_bm25_injection_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  text_query => to_tsquery('simple', 'needle'),
	  dense_k => 1,
	  bm25_k => 2,
	  final_k => 1,
	  bm25_weight => 0
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();
	IF top_id <> 'good' OR
		stats->>'multivector_bm25_injection_enabled' <> 'true' OR
		(stats->>'learned_sparse_candidates')::int < 1 OR
		(stats->>'learned_sparse_retained_for_maxsim')::int < 1 OR
		(stats->>'learned_sparse_branch_latency_us')::int < 0 OR
		(stats->>'multivector_bm25_injection_exact_reranked')::int < 1 THEN
		RAISE EXCEPTION 'expected learned_sparse injection to admit exact-MaxSim winner, top %, stats %',
			top_id, stats;
	END IF;
	SET LOCAL turbohybrid.multivector_sparse_candidate_source = off;
END
$$;

DO $$
DECLARE
	q turbohybrid_multivector := turbohybrid_multivector(ARRAY[
		'[1,0]'::vector,
		'[0,1]'::vector
	]);
	topk_ids text[];
	adaptive_ids text[];
	stats jsonb;
BEGIN
	SET LOCAL turbohybrid.multivector_candidate_source = exact_token_scan;
	SET LOCAL turbohybrid.multivector_candidate_reservoirs = off;
	SET LOCAL turbohybrid.multivector_subvector_k = 4;
	SET LOCAL turbohybrid.multivector_unique_docs_per_token = 4;
	SET LOCAL turbohybrid.multivector_max_raw_hits_per_token = 8;
	SET LOCAL turbohybrid.multivector_doc_candidate_k = 4;
	SET LOCAL turbohybrid.multivector_exact_rerank_k = 4;

	SET LOCAL turbohybrid.multivector_exact_rerank = topk;
	SELECT array_agg(id) INTO topk_ids
	FROM (
		SELECT id
		FROM mv_bm25_injection_docs
		ORDER BY colbert <~> turbohybrid_query(
		  multivector_query => q,
		  dense_k => 4,
		  final_k => 1
		)
		LIMIT 3
	) ranked;

	SET LOCAL turbohybrid.multivector_exact_rerank = adaptive;
	SELECT array_agg(id) INTO adaptive_ids
	FROM (
		SELECT id
		FROM mv_bm25_injection_docs
		ORDER BY colbert <~> turbohybrid_query(
		  multivector_query => q,
		  dense_k => 4,
		  final_k => 1
		)
		LIMIT 3
	) ranked;
	stats := turbohybrid_last_scan_stats();
	IF adaptive_ids IS DISTINCT FROM topk_ids OR
		stats->>'adaptive_rerank_topk_changed_vs_full' <> 'false' OR
		(stats->>'exact_rerank_candidates')::int < 1 OR
		(stats->>'exact_rerank_tokens_evaluated')::int < 1 OR
		(stats->>'exact_rerank_tokens_skipped')::int < 1 OR
		(stats->>'exact_rerank_pairs_saved')::int < 1 THEN
		RAISE EXCEPTION 'expected adaptive rerank to preserve topk result with saved work, topk %, adaptive %, stats %',
			topk_ids, adaptive_ids, stats;
	END IF;

	SET LOCAL turbohybrid.multivector_exact_rerank = adaptive;
	SET LOCAL turbohybrid.multivector_exact_rerank_k = 1;
	SELECT array_agg(id) INTO adaptive_ids
	FROM (
		SELECT id
		FROM mv_bm25_injection_docs
		ORDER BY colbert <~> turbohybrid_query(
		  multivector_query => q,
		  dense_k => 1,
		  final_k => 1
		)
		LIMIT 1
	) ranked;
	stats := turbohybrid_last_scan_stats();
	IF (stats->>'exact_rerank_candidates')::int <> 1 OR
		(stats->>'exact_rerank_tokens_evaluated')::int <> 2 OR
		(stats->>'exact_rerank_tokens_skipped')::int <> 0 OR
		(stats->>'exact_rerank_pairs_saved')::int <> 0 THEN
		RAISE EXCEPTION 'expected adaptive rerank to fall back to full scoring when topK equals rerank limit, stats %',
			stats;
	END IF;
END
$$;

DO $$
DECLARE
	q turbohybrid_multivector := turbohybrid_multivector(ARRAY[
		'[1,0]'::vector,
		'[0,1]'::vector
	]);
	top_id text;
	stats jsonb;
BEGIN
	SET LOCAL turbohybrid.multivector_candidate_source = exact_doc_scan;
	SELECT id INTO top_id
	FROM mv_bm25_injection_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  dense_k => 1,
	  final_k => 1
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();
	IF top_id <> 'good' OR
		stats->>'multivector_candidate_source' <> 'exact_doc_scan' OR
		stats->>'multivector_plain_fallback_reason' <> 'exact_doc_scan' OR
		stats->>'multivector_doc_graph_prototype_enabled' <> 'false' THEN
		RAISE EXCEPTION 'expected exact_doc_scan to exact-score documents, top %, stats %',
			top_id, stats;
	END IF;

	SET LOCAL turbohybrid.multivector_candidate_source = doc_graph_prototype;
	SELECT id INTO top_id
	FROM mv_bm25_injection_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  dense_k => 1,
	  final_k => 1
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();
	IF top_id <> 'good' OR
		stats->>'multivector_candidate_source' <> 'doc_graph_prototype' OR
		stats->>'multivector_doc_graph_prototype_enabled' <> 'true' OR
		(stats->>'multivector_doc_graph_docs_scored')::int <> 4 OR
		(stats->>'multivector_doc_graph_candidates')::int <> 1 OR
		(stats->>'multivector_doc_graph_heap_fetches')::int <> 4 OR
		stats->>'multivector_doc_graph_warning' <>
			'prototype_heap_scan_no_index_resident_doc_graph' THEN
		RAISE EXCEPTION 'expected doc_graph_prototype heap-scan stats, top %, stats %',
			top_id, stats;
	END IF;
END
$$;

DO $$
BEGIN
	BEGIN
		SET LOCAL turbohybrid.multivector_candidate_source = document_nodes;
		PERFORM id
		FROM mv_bm25_injection_docs
		ORDER BY colbert <~> turbohybrid_query(
		  multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
		  dense_k => 1,
		  final_k => 1
		)
		LIMIT 1;
		RAISE EXCEPTION 'expected document_nodes source on token-node index to fail';
	EXCEPTION
		WHEN feature_not_supported THEN
			IF SQLERRM NOT LIKE '%requires multivector_graph = document_nodes%' THEN
				RAISE EXCEPTION 'unexpected document_nodes token-node error: %', SQLERRM;
			END IF;
	END;
END
$$;

DO $$
BEGIN
	BEGIN
		PERFORM id
		FROM mv_docs
		ORDER BY colbert <~> turbohybrid_query(
		  multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
		  text_query => to_tsquery('simple', 'needle'),
		  dense_k => 1,
		  bm25_k => 1,
		  final_k => 1
		)
		LIMIT 1;
		RAISE EXCEPTION 'expected text_query on non-lexical multivector index to fail';
	EXCEPTION
		WHEN feature_not_supported THEN
			NULL;
	END;
END
$$;

DROP TABLE mv_bm25_injection_docs;
RESET turbohybrid.multivector_bm25_candidate_injection;
RESET turbohybrid.multivector_sparse_candidate_source;
RESET turbohybrid.multivector_exact_rerank;
RESET turbohybrid.multivector_exact_rerank_k;
RESET turbohybrid.multivector_candidate_source;
RESET turbohybrid.multivector_candidate_reservoirs;
SET turbohybrid.multivector_plain_fallback = off;
RESET turbohybrid.multivector_doc_candidate_k;
RESET turbohybrid.multivector_max_raw_hits_per_token;
RESET turbohybrid.multivector_unique_docs_per_token;
RESET turbohybrid.multivector_subvector_k;
RESET enable_seqscan;

DROP INDEX mv_docs_colbert_idx;
CREATE INDEX mv_docs_hybrid_idx ON mv_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops,
   body_tsv bm25_tsvector_turbohybrid_ops);

INSERT INTO mv_docs VALUES
  (4, turbohybrid_multivector(ARRAY['[-1,-1]'::vector, '[-0.8,-1.2]'::vector]), to_tsvector('epsilon'));

SET enable_seqscan = off;
DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_index_stats('mv_docs_hybrid_idx'::regclass);
	IF (stats->>'node_count')::int <> 8 THEN
		RAISE EXCEPTION 'expected multivector insert to append two graph nodes, got %',
			stats->>'node_count';
	END IF;
END
$$;

SELECT id FROM mv_docs
  ORDER BY colbert <~> turbohybrid_query(
    multivector_query => turbohybrid_multivector(ARRAY['[-1,-1]'::vector]),
    dense_k => 8,
    final_k => 1
  )
  LIMIT 1;

DO $$
DECLARE
	stats jsonb;
	memory_stats jsonb;
BEGIN
	stats := turbohybrid_last_scan_stats();
	memory_stats := turbohybrid_estimate_memory('mv_docs_hybrid_idx'::regclass);
	IF stats->>'multivector_docmap_source' <> 'sidecar' OR
		(stats->>'multivector_docmap_bytes')::bigint < 1 OR
		(memory_stats->'native'->>'multivector_docmap_bytes')::bigint < 1 THEN
		RAISE EXCEPTION 'expected inserted multivector docmap sidecar, scan %, memory %',
			stats, memory_stats;
	END IF;
END
$$;

SELECT id FROM mv_docs
  ORDER BY colbert <~> turbohybrid_query(
    multivector_query => turbohybrid_multivector(ARRAY['[-1,-1]'::vector]),
    text_query => to_tsquery('epsilon'),
    dense_k => 8,
    bm25_k => 3,
    final_k => 1,
    fusion => 'rrf'
  )
  LIMIT 1;

SET turbohybrid.multivector_branch_plan = qdrant_like;

SELECT id FROM mv_docs
  ORDER BY colbert <~> turbohybrid_query(
    multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
    text_query => to_tsquery('gamma'),
    dense_k => 6,
    bm25_k => 3,
    final_k => 3,
    fusion => 'rrf'
  )
  LIMIT 1;

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_last_scan_stats();
	IF stats->>'index_used' <> 'true' OR
		stats->>'dense_branch_used' <> 'true' OR
		stats->>'bm25_branch_used' <> 'true' OR
		(stats->>'both_match')::int < 1 OR
		stats->>'fusion_strategy' NOT IN ('sorted_merge_doc', 'hash_doc') THEN
		RAISE EXCEPTION 'expected indexed multivector hybrid scan, got %', stats;
	END IF;
END
$$;

SELECT id FROM mv_docs
  ORDER BY colbert <~> turbohybrid_query(
    multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector]),
    text_query => to_tsquery('alpha | gamma'),
    dense_k => 6,
    bm25_k => 3,
    final_k => 3,
    fusion => 'rrf'
  )
  LIMIT 3;

SELECT id FROM mv_docs
  ORDER BY colbert <~> turbohybrid_query(
    multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
    dense_k => 6,
    final_k => 1,
    fusion => 'weighted'
  )
  LIMIT 1;

SELECT id FROM mv_docs
  ORDER BY colbert <~> turbohybrid_query(
    multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
    text_query => to_tsquery('gamma'),
    fusion => 'weighted'
  )
  LIMIT 1;

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_last_scan_stats();
	IF stats->>'fusion_strategy' NOT IN ('sorted_merge_doc', 'hash_doc') OR
		(stats->>'both_match')::int < 1 THEN
		RAISE EXCEPTION 'expected weighted multivector hybrid fusion, got %',
			stats;
	END IF;
END
$$;

SELECT id FROM mv_docs
  ORDER BY colbert <~> turbohybrid_query(
    multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
    text_query => to_tsquery('gamma'),
    fusion => 'fast_weighted'
  )
  LIMIT 1;

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_last_scan_stats();
	IF stats->>'fusion_strategy' NOT IN ('sorted_merge_doc', 'hash_doc') OR
		(stats->>'fast_weighted_enabled')::boolean IS DISTINCT FROM true OR
		stats->>'dense_norm_mode' <> 'logistic' OR
		stats->>'bm25_norm_mode' <> 'saturating' OR
		stats->>'multivector_branch_plan' <> 'qdrant_like' OR
		stats->>'branch_fusion_mode' <> 'qdrant_like_normalized' OR
		stats->'branch_kinds' <> '["token_nodes", "bm25", "exact_doc_scan"]'::jsonb THEN
		RAISE EXCEPTION 'expected normalized fast_weighted multivector hybrid fusion, got %',
			stats;
	END IF;
END
$$;

RESET turbohybrid.multivector_branch_plan;

SELECT id FROM mv_docs
  ORDER BY colbert <~> turbohybrid_query(
    multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
    text_query => to_tsquery('gamma'),
    fusion => 'calibrated'
  )
  LIMIT 1;

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_last_scan_stats();
	IF stats->>'fusion_strategy' NOT IN ('sorted_merge_doc', 'hash_doc') OR
		(stats->>'calibrated_fusion_enabled')::boolean IS DISTINCT FROM true OR
		stats->>'calibrated_fusion_dense_norm_mode' <> 'logistic' OR
		stats->>'calibrated_fusion_bm25_norm_mode' <> 'saturating' THEN
		RAISE EXCEPTION 'expected calibrated multivector hybrid fusion, got %',
			stats;
	END IF;
END
$$;

SET turbohybrid.multivector_branch_plan = qdrant_like;
SET turbohybrid.dbsf_min_branch_candidates = 1;

SELECT id FROM mv_docs
  ORDER BY colbert <~> turbohybrid_query(
    multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
    text_query => to_tsquery('alpha | gamma'),
    dense_k => 6,
    bm25_k => 3,
    final_k => 3,
    fusion => 'dbsf'
  )
  LIMIT 1;

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_last_scan_stats();
	IF stats->>'fusion_strategy' NOT IN ('sorted_merge_doc', 'hash_doc') OR
		(stats->>'dbsf_enabled')::boolean IS DISTINCT FROM true OR
		stats->>'dense_norm_mode' <> 'dbsf' OR
		stats->>'bm25_norm_mode' <> 'dbsf' OR
		stats->>'multivector_branch_plan' <> 'qdrant_like' OR
		stats->>'branch_fusion_mode' <> 'qdrant_like_dbsf' OR
		stats->'branch_kinds' <> '["token_nodes", "bm25", "exact_doc_scan"]'::jsonb OR
		jsonb_array_length(stats->'dbsf_branch_mean') <> 2 OR
		jsonb_array_length(stats->'dbsf_branch_stddev') <> 2 OR
		(stats->'dbsf_branch_stddev'->>0)::float8 <= 0 OR
		(stats->'dbsf_branch_stddev'->>1)::float8 <= 0 OR
		(stats->>'dbsf_degenerate_branches')::int <> 0 THEN
		RAISE EXCEPTION 'expected document-keyed DBSF multivector hybrid fusion, got %',
			stats;
	END IF;
END
$$;

CREATE TABLE mv_dbsf_scale_docs (
  id int,
  colbert turbohybrid_multivector,
  body_tsv tsvector
);

INSERT INTO mv_dbsf_scale_docs VALUES
  (1, turbohybrid_multivector(ARRAY['[0.99,0.01]'::vector]), to_tsvector('simple', repeat('needle ', 200))),
  (2, turbohybrid_multivector(ARRAY['[0.90,0.10]'::vector]), to_tsvector('simple', repeat('needle ', 80))),
  (3, turbohybrid_multivector(ARRAY['[0.80,0.20]'::vector]), to_tsvector('simple', repeat('needle ', 20))),
  (4, turbohybrid_multivector(ARRAY['[0.70,0.30]'::vector]), to_tsvector('simple', 'needle')),
  (5, turbohybrid_multivector(ARRAY['[0.60,0.40]'::vector]), to_tsvector('simple', 'background one')),
  (6, turbohybrid_multivector(ARRAY['[0.50,0.50]'::vector]), to_tsvector('simple', 'background two')),
  (7, turbohybrid_multivector(ARRAY['[0.40,0.60]'::vector]), to_tsvector('simple', 'background three')),
  (8, turbohybrid_multivector(ARRAY['[0.30,0.70]'::vector]), to_tsvector('simple', 'background four')),
  (9, turbohybrid_multivector(ARRAY['[0.20,0.80]'::vector]), to_tsvector('simple', 'background five')),
  (10, turbohybrid_multivector(ARRAY['[0.10,0.90]'::vector]), to_tsvector('simple', 'background six')),
  (11, turbohybrid_multivector(ARRAY['[0.05,0.95]'::vector]), to_tsvector('simple', 'background seven')),
  (12, turbohybrid_multivector(ARRAY['[-1,0]'::vector]), to_tsvector('simple', 'background eight'));

CREATE INDEX mv_dbsf_scale_idx ON mv_dbsf_scale_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops,
   body_tsv bm25_tsvector_turbohybrid_ops);
ANALYZE mv_dbsf_scale_docs;

DO $$
DECLARE
	stats jsonb;
	top_id int;
BEGIN
	SELECT id INTO top_id
	FROM mv_dbsf_scale_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
	  text_query => to_tsquery('simple', 'needle'),
	  dense_k => 8,
	  bm25_k => 4,
	  final_k => 4,
	  fusion => 'dbsf'
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();
	IF top_id IS NULL OR
		(stats->>'dbsf_enabled')::boolean IS DISTINCT FROM true OR
		(stats->'dbsf_branch_max'->>0)::float8 <= (stats->'dbsf_branch_min'->>0)::float8 OR
		(stats->'dbsf_branch_max'->>1)::float8 <= (stats->'dbsf_branch_min'->>1)::float8 OR
		(stats->'dbsf_branch_stddev'->>0)::float8 <= 0 OR
		(stats->'dbsf_branch_stddev'->>1)::float8 <= 0 OR
		(stats->>'dbsf_degenerate_branches')::int <> 0 THEN
		RAISE EXCEPTION 'expected DBSF to normalize mismatched BM25/MaxSim branch scales, top %, stats %',
			top_id, stats;
	END IF;
END
$$;

DROP TABLE mv_dbsf_scale_docs;

CREATE TABLE mv_dbsf_degenerate_docs (
  id int,
  colbert turbohybrid_multivector,
  body_tsv tsvector
);

INSERT INTO mv_dbsf_degenerate_docs VALUES
  (1, turbohybrid_multivector(ARRAY['[1,0]'::vector]), to_tsvector('simple', 'same')),
  (2, turbohybrid_multivector(ARRAY['[1,0]'::vector]), to_tsvector('simple', 'same')),
  (3, turbohybrid_multivector(ARRAY['[1,0]'::vector]), to_tsvector('simple', 'same'));

CREATE INDEX mv_dbsf_degenerate_idx ON mv_dbsf_degenerate_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops,
   body_tsv bm25_tsvector_turbohybrid_ops);
ANALYZE mv_dbsf_degenerate_docs;

DO $$
DECLARE
	stats jsonb;
BEGIN
	PERFORM id
	FROM mv_dbsf_degenerate_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
	  text_query => to_tsquery('simple', 'same'),
	  dense_k => 3,
	  bm25_k => 3,
	  final_k => 3,
	  fusion => 'dbsf'
	)
	LIMIT 3;
	stats := turbohybrid_last_scan_stats();
	IF (stats->>'dbsf_enabled')::boolean IS DISTINCT FROM true OR
		(stats->>'dbsf_degenerate_branches')::int < 1 THEN
		RAISE EXCEPTION 'expected DBSF degenerate branch diagnostics for identical scores, got %',
			stats;
	END IF;
END
$$;

DROP TABLE mv_dbsf_degenerate_docs;
RESET turbohybrid.dbsf_min_branch_candidates;
RESET turbohybrid.multivector_branch_plan;

CREATE TABLE mv_docnode_hybrid_scheduler_docs (
  id text PRIMARY KEY,
  colbert turbohybrid_multivector,
  body_tsv tsvector
);

INSERT INTO mv_docnode_hybrid_scheduler_docs VALUES
  ('dense_a', turbohybrid_multivector(ARRAY[
    '[1,0]'::vector,
    '[0,1]'::vector
  ]), to_tsvector('simple', 'common alpha')),
  ('dense_b', turbohybrid_multivector(ARRAY[
    '[0.9,0.1]'::vector,
    '[0.1,0.9]'::vector
  ]), to_tsvector('simple', 'common beta')),
  ('lexical_only', turbohybrid_multivector(ARRAY[
    '[-1,0]'::vector,
    '[0,-1]'::vector
  ]), to_tsvector('simple', 'common needle')),
  ('mixed', turbohybrid_multivector(ARRAY[
    '[0.8,0.2]'::vector,
    '[0.2,0.8]'::vector
  ]), to_tsvector('simple', 'common needle'));

CREATE INDEX mv_docnode_hybrid_scheduler_idx
  ON mv_docnode_hybrid_scheduler_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops,
   body_tsv bm25_tsvector_turbohybrid_ops)
  WITH (multivector_graph = document_nodes,
        graph_m = 4,
        graph_ef_construction = 8,
        graph_ef_search = 4);
ANALYZE mv_docnode_hybrid_scheduler_docs;

DO $$
DECLARE
	stats jsonb;
	top_id text;
BEGIN
	PERFORM set_config('turbohybrid.hybrid_budget_policy', 'adaptive', true);
	PERFORM set_config('turbohybrid.multivector_candidate_source', 'document_nodes', true);
	PERFORM set_config('turbohybrid.default_dense_k', '64', true);
	PERFORM set_config('turbohybrid.default_bm25_k', '64', true);

	SELECT id INTO top_id
	FROM mv_docnode_hybrid_scheduler_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => turbohybrid_multivector(ARRAY[
	    '[1,0]'::vector,
	    '[0,1]'::vector
	  ]),
	  text_query => to_tsquery('simple', 'common | needle'),
	  fusion => 'rrf'
	)
	LIMIT 1;

	stats := turbohybrid_last_scan_stats();
	IF top_id IS NULL OR
		stats->>'hybrid_budget_policy' <> 'adaptive' OR
		stats->>'multivector_candidate_source' <> 'document_nodes' OR
		stats->>'multivector_graph_mode' <> 'document_nodes' OR
		stats->>'hybrid_budget_reason' <>
			'admission_document_dense_reduce_bm25' OR
		(stats->>'hybrid_bm25_k_chosen')::int >= 64 OR
		stats->>'fusion_strategy' NOT IN ('sorted_merge_doc', 'hash_doc') THEN
		RAISE EXCEPTION 'expected branch-aware document-node hybrid scheduler stats, top %, stats %',
			top_id, stats;
	END IF;
END
$$;

DROP TABLE mv_docnode_hybrid_scheduler_docs;

CREATE TABLE mv_weighted_docs (
  id int,
  colbert turbohybrid_multivector,
  body_tsv tsvector
);

INSERT INTO mv_weighted_docs VALUES
  (1, turbohybrid_multivector(ARRAY['[1,0]'::vector]), to_tsvector('alpha')),
  (2, turbohybrid_multivector(ARRAY['[1,0]'::vector]), to_tsvector('beta')),
  (3, turbohybrid_multivector(ARRAY['[-1,0]'::vector]), to_tsvector('alpha'));

CREATE INDEX mv_weighted_docs_idx ON mv_weighted_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops,
   body_tsv bm25_tsvector_turbohybrid_ops);

DO $$
DECLARE
	top_id int;
BEGIN
	SELECT id INTO top_id FROM mv_weighted_docs
		ORDER BY colbert <~> turbohybrid_query(
			multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
			text_query => to_tsquery('alpha'),
			dense_k => 2,
			bm25_k => 2,
			final_k => 3,
			fusion => 'weighted',
			alpha => 0.5
		)
		LIMIT 1;
	IF top_id <> 1 THEN
		RAISE EXCEPTION 'expected weighted fusion to rank both-evidence doc first, got %',
			top_id;
	END IF;
END
$$;

DO $$
DECLARE
	result_count int;
	result_id int;
	seen_ids int[] := ARRAY[]::int[];
BEGIN
	result_count := 0;
	FOR result_id IN
		SELECT id FROM mv_weighted_docs
			ORDER BY colbert <~> turbohybrid_query(
				multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
				text_query => to_tsquery('alpha'),
				dense_k => 2,
				bm25_k => 2,
				final_k => 3,
				fusion => 'weighted',
				alpha => 0.5
			)
			LIMIT 3
	LOOP
		result_count := result_count + 1;
		IF result_id = ANY(seen_ids) THEN
			RAISE EXCEPTION 'expected weighted fusion to deduplicate docs, saw duplicate doc %',
				result_id;
		END IF;
		seen_ids := seen_ids || result_id;
	END LOOP;
	IF result_count < 1 THEN
		RAISE EXCEPTION 'expected weighted fusion to return at least one doc';
	END IF;
END
$$;

SELECT bool_and(id IN (1, 3)) AS weighted_require_bm25_match
FROM (
  SELECT id FROM mv_weighted_docs
    ORDER BY colbert <~> turbohybrid_query(
      multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
      text_query => to_tsquery('alpha'),
      dense_k => 2,
      bm25_k => 2,
      final_k => 3,
      fusion => 'weighted',
      alpha => 0.5,
      require_bm25_match => true
    )
    LIMIT 3
) s;

DROP TABLE mv_weighted_docs;

SET turbohybrid.multivector_max_query_vectors = 1;
SELECT id FROM mv_docs
  ORDER BY colbert <~> turbohybrid_query(
    multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector]),
    dense_k => 6,
    final_k => 1
  )
  LIMIT 1;
RESET turbohybrid.multivector_max_query_vectors;

SET turbohybrid.multivector_max_dim = 1;
SELECT id FROM mv_docs
  ORDER BY colbert <~> turbohybrid_query(
    multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
    dense_k => 6,
    final_k => 1
  )
  LIMIT 1;
RESET turbohybrid.multivector_max_dim;

SELECT turbohybrid_multivector_model_info('colbert-ir/colbertv2.0')->>'dim'
  AS colbertv2_dim;
SELECT turbohybrid_multivector_model_info('answerdotai/answerai-colbert-small-v1')->>'dim'
  AS answerai_small_dim;
SELECT turbohybrid_multivector_model_info('jinaai/jina-colbert-v2-64')->>'dim'
  AS jina_colbert_v2_64_dim;
SELECT turbohybrid_multivector_model_info('johannhartmann/SauerkrautLM-Multi-ColBERT-15m-GGUF')->>'status'
  AS sauerkraut_gguf_status;
SELECT turbohybrid_multivector_model_info('colbert-ir/colbertv2.0')->>'token_mask_policy'
  AS colbertv2_mask_policy;
SELECT turbohybrid_multivector_model_info('vidore/colpali-v1.2')->>'field_context_policy'
  AS colpali_field_context_policy;
SELECT turbohybrid_multivector_model_info('not/a-real-model')->>'known'
  AS unknown_model_known;

SET turbohybrid.multivector_model_name = 'answerdotai/answerai-colbert-small-v1';
SELECT id FROM mv_docs
  ORDER BY colbert <~> turbohybrid_query(
    multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
    dense_k => 6,
    final_k => 1
  )
  LIMIT 1;
RESET turbohybrid.multivector_model_name;

SET turbohybrid.multivector_model_name = 'jinaai/jina-colbert-v2-64';
SELECT turbohybrid_index_stats('mv_docs_hybrid_idx'::regclass)->>'multivector_model_name'
  AS stats_model_name,
       turbohybrid_index_stats('mv_docs_hybrid_idx'::regclass)->>'multivector_model_dim'
  AS stats_model_dim,
       turbohybrid_index_stats('mv_docs_hybrid_idx'::regclass)->>'multivector_model_recommended_storage_kind'
  AS stats_model_storage,
       turbohybrid_index_stats('mv_docs_hybrid_idx'::regclass)->>'multivector_model_token_mask_policy'
  AS stats_model_mask_policy,
       turbohybrid_index_stats('mv_docs_hybrid_idx'::regclass)->>'multivector_model_field_context_policy'
  AS stats_model_field_context_policy;
RESET turbohybrid.multivector_model_name;

CREATE TEMP TABLE mv_model_warning_docs (
	id int,
	colbert turbohybrid_multivector
);
INSERT INTO mv_model_warning_docs
SELECT 1,
       turbohybrid_multivector_from_float4(
         ARRAY(
           SELECT CASE WHEN gs = 1 THEN 1 ELSE 0 END::real
           FROM generate_series(1, 128) gs
         ),
         128
       );
CREATE INDEX mv_model_warning_docs_idx ON mv_model_warning_docs
USING turbohybrid (colbert multivector_maxsim_ip_turbohybrid_ops);
SET enable_seqscan = off;
SET turbohybrid.multivector_model_name = 'colbert-ir/colbertv2.0';
SELECT id FROM mv_model_warning_docs
  ORDER BY colbert <~> turbohybrid_query(
    multivector_query => turbohybrid_multivector_from_float4(
      ARRAY(
        SELECT CASE WHEN (gs - 1) % 128 = 0 THEN 1 ELSE 0 END::real
        FROM generate_series(1, 128 * 33) gs
      ),
      128
    ),
    dense_k => 1,
    final_k => 1
  )
  LIMIT 1;
RESET turbohybrid.multivector_model_name;
SET enable_seqscan = off;
DROP TABLE mv_model_warning_docs;

SET turbohybrid.multivector_max_raw_hits_per_token = 1;
SELECT id FROM mv_docs
  ORDER BY colbert <~> turbohybrid_query(
    multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
    dense_k => 6,
    final_k => 1
  )
  LIMIT 1;

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_last_scan_stats();
	IF (stats->>'multivector_raw_subvector_hits')::int > 1 THEN
		RAISE EXCEPTION 'expected raw-hit cap to be respected, got %', stats;
	END IF;
END
$$;
RESET turbohybrid.multivector_max_raw_hits_per_token;

DELETE FROM mv_docs WHERE id = 4;
VACUUM mv_docs;
SELECT COALESCE(bool_or(id = 4), false) AS deleted_doc_returned
FROM (
  SELECT id FROM mv_docs
    ORDER BY colbert <~> turbohybrid_query(
      multivector_query => turbohybrid_multivector(ARRAY['[-1,-1]'::vector]),
      dense_k => 8,
      final_k => 3
    )
    LIMIT 3
) s;

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_last_scan_stats();
	IF stats->>'multivector_docmap_source' <> 'sidecar' THEN
		RAISE EXCEPTION 'expected post-delete scan to keep using sidecar, got %',
			stats;
	END IF;
END
$$;

CREATE TABLE mv_lifecycle_docs (
  id int PRIMARY KEY,
  colbert turbohybrid_multivector,
  body_tsv tsvector
);

INSERT INTO mv_lifecycle_docs VALUES
  (1, turbohybrid_multivector(ARRAY['[1,0]'::vector]), to_tsvector('delete_me')),
  (2, turbohybrid_multivector(ARRAY['[-1,0]'::vector]), to_tsvector('old_update')),
  (3, turbohybrid_multivector(ARRAY['[-1,0]'::vector]), to_tsvector('hybrid_old'));

CREATE INDEX mv_lifecycle_idx ON mv_lifecycle_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops,
   body_tsv bm25_tsvector_turbohybrid_ops);

SET enable_seqscan = off;
DELETE FROM mv_lifecycle_docs WHERE id = 1;
SELECT COALESCE(bool_or(id = 1), false) AS lifecycle_deleted_before_vacuum
FROM (
  SELECT id FROM mv_lifecycle_docs
    ORDER BY colbert <~> turbohybrid_query(
      multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
      dense_k => 6,
      final_k => 3
    )
    LIMIT 3
) s;

VACUUM mv_lifecycle_docs;
SELECT COALESCE(bool_or(id = 1), false) AS lifecycle_deleted_after_vacuum
FROM (
  SELECT id FROM mv_lifecycle_docs
    ORDER BY colbert <~> turbohybrid_query(
      multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
      dense_k => 6,
      final_k => 3
    )
    LIMIT 3
) s;

UPDATE mv_lifecycle_docs
SET colbert = turbohybrid_multivector(ARRAY['[1,0]'::vector]),
    body_tsv = to_tsvector('updated_dense')
WHERE id = 2;

SELECT id AS lifecycle_updated_dense_top FROM mv_lifecycle_docs
  ORDER BY colbert <~> turbohybrid_query(
    multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
    dense_k => 6,
    final_k => 1
  )
  LIMIT 1;

SELECT id AS lifecycle_old_update_top FROM mv_lifecycle_docs
  ORDER BY colbert <~> turbohybrid_query(
    multivector_query => turbohybrid_multivector(ARRAY['[-1,0]'::vector]),
    dense_k => 6,
    final_k => 1
  )
  LIMIT 1;

UPDATE mv_lifecycle_docs
SET colbert = turbohybrid_multivector(ARRAY['[0,1]'::vector]),
    body_tsv = to_tsvector('hybrid_new')
WHERE id = 3;

SELECT id AS lifecycle_hybrid_update_top FROM mv_lifecycle_docs
  ORDER BY colbert <~> turbohybrid_query(
    multivector_query => turbohybrid_multivector(ARRAY['[0,1]'::vector]),
    text_query => to_tsquery('hybrid_new'),
    dense_k => 6,
    bm25_k => 6,
    final_k => 1,
    fusion => 'rrf'
  )
  LIMIT 1;

CREATE TEMP TABLE mv_lifecycle_before_insert AS
SELECT (turbohybrid_index_stats('mv_lifecycle_idx'::regclass)->>'node_count')::int AS node_count;

INSERT INTO mv_lifecycle_docs VALUES
  (4, turbohybrid_multivector(ARRAY[
    '[-0.2,-0.8]'::vector,
    '[-0.4,-0.6]'::vector,
    '[-0.6,-0.4]'::vector
  ]), to_tsvector('insert_after_vacuum'));

DO $$
DECLARE
	before_count int;
	after_count int;
BEGIN
	SELECT node_count INTO before_count FROM mv_lifecycle_before_insert;
	after_count := (turbohybrid_index_stats('mv_lifecycle_idx'::regclass)->>'node_count')::int;
	IF after_count <= before_count THEN
		RAISE EXCEPTION 'expected post-vacuum multivector insert to increase node_count, before %, after %',
			before_count, after_count;
	END IF;
END
$$;

SELECT id AS lifecycle_insert_after_vacuum_top FROM mv_lifecycle_docs
  ORDER BY colbert <~> turbohybrid_query(
    multivector_query => turbohybrid_multivector(ARRAY['[-0.2,-0.8]'::vector]),
    dense_k => 8,
    final_k => 1
  )
  LIMIT 1;

DROP TABLE mv_lifecycle_docs;
DROP TABLE mv_lifecycle_before_insert;

SELECT id AS rrf_text_only_top FROM mv_docs
  ORDER BY colbert <~> turbohybrid_query(
    text_query => to_tsquery('gamma'),
    bm25_k => 3,
    final_k => 1,
    fusion => 'rrf'
  )
  LIMIT 1;

SELECT bool_and(id IN (2, 3)) AS rrf_require_bm25_match
FROM (
  SELECT id FROM mv_docs
    ORDER BY colbert <~> turbohybrid_query(
      multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
      text_query => to_tsquery('gamma'),
      dense_k => 6,
      bm25_k => 3,
      final_k => 3,
      fusion => 'rrf',
      require_bm25_match => true
    )
    LIMIT 3
) s;

EXPLAIN (COSTS OFF)
SELECT id FROM mv_docs
  ORDER BY colbert <~> turbohybrid_query(
    multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
    dense_k => 6,
    final_k => 1
  )
  LIMIT 1;
RESET enable_seqscan;

CREATE TABLE mv_usability_dense_only_docs (
  id int,
  colbert turbohybrid_multivector
);

INSERT INTO mv_usability_dense_only_docs VALUES
  (1, turbohybrid_multivector(ARRAY['[1,0]'::vector]));

CREATE INDEX mv_usability_dense_only_idx ON mv_usability_dense_only_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops);

SET enable_seqscan = off;
SELECT id FROM mv_usability_dense_only_docs
  ORDER BY colbert <~> turbohybrid_query(
    text_query => to_tsquery('alpha'),
    bm25_k => 1,
    final_k => 1,
    fusion => 'rrf'
  )
  LIMIT 1;
RESET enable_seqscan;
DROP TABLE mv_usability_dense_only_docs;

CREATE TABLE mv_usability_vector_docs (
  id int,
  embedding vector(2)
);

INSERT INTO mv_usability_vector_docs VALUES
  (1, '[1,0]'::vector);

CREATE INDEX mv_usability_vector_idx ON mv_usability_vector_docs USING turbohybrid
  (embedding vector_cosine_turbohybrid_ops);

SET enable_seqscan = off;
SELECT id FROM mv_usability_vector_docs
  ORDER BY embedding <~> turbohybrid_query(
    multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
    dense_k => 1,
    final_k => 1
  )
  LIMIT 1;
RESET enable_seqscan;
DROP TABLE mv_usability_vector_docs;

CREATE INDEX mv_docs_expr_idx ON mv_docs USING turbohybrid
  ((coalesce(colbert, colbert)) multivector_cosine_turbohybrid_ops);

SET turbohybrid.multivector_max_doc_vectors = 1;
CREATE INDEX mv_docs_colbert_limited_idx ON mv_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops);
RESET turbohybrid.multivector_max_doc_vectors;

CREATE INDEX mv_docs_bad_order_idx ON mv_docs USING turbohybrid
  (body_tsv bm25_tsvector_turbohybrid_ops, colbert multivector_cosine_turbohybrid_ops);

SELECT turbohybrid_index_stats('mv_docs_hybrid_idx'::regclass)->>'multivector_graph_mode'
  AS default_multivector_graph_mode;

CREATE INDEX mv_docs_graph_mode_token_idx ON mv_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops)
  WITH (multivector_graph = token_nodes);

SELECT turbohybrid_index_stats('mv_docs_graph_mode_token_idx'::regclass)->>'multivector_graph_mode'
  AS explicit_multivector_graph_mode;

CREATE TABLE mv_document_node_docs (
  id int PRIMARY KEY,
  colbert turbohybrid_multivector
);

INSERT INTO mv_document_node_docs VALUES
  (1, turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0.8,0.2]'::vector])),
  (2, turbohybrid_multivector(ARRAY['[0,1]'::vector, '[0.2,0.8]'::vector])),
  (3, turbohybrid_multivector(ARRAY['[0.95,0.05]'::vector, '[0,1]'::vector]));

CREATE INDEX mv_document_node_docs_token_nodes_idx ON mv_document_node_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops)
  WITH (multivector_graph = token_nodes);

SELECT turbohybrid_index_stats('mv_document_node_docs_token_nodes_idx'::regclass)->>'multivector_graph_mode'
  AS token_node_graph_mode,
       (turbohybrid_index_stats('mv_document_node_docs_token_nodes_idx'::regclass)->>'node_count')::int
  AS token_node_count;

SET turbohybrid.dense_build_neighbor_select = auto;

CREATE INDEX mv_document_node_docs_idx ON mv_document_node_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops)
  WITH (multivector_graph = document_nodes, graph_ef_search = 2);

SELECT (turbohybrid_last_build_stats()->>'multivector_doc_exact_build_distance_calls')::bigint
  AS document_node_exact_build_calls;

SELECT turbohybrid_index_stats('mv_document_node_docs_idx'::regclass)->>'multivector_graph_mode'
  AS document_node_graph_mode,
       turbohybrid_index_stats('mv_document_node_docs_idx'::regclass)->>'multivector_doc_build_scorer'
  AS document_node_build_scorer,
       turbohybrid_index_stats('mv_document_node_docs_idx'::regclass)->>'multivector_proxy_encoder'
  AS document_node_proxy_encoder,
       (turbohybrid_index_stats('mv_document_node_docs_idx'::regclass)->>'dimensions')::int
  AS document_node_proxy_dimensions,
       (turbohybrid_index_stats('mv_document_node_docs_idx'::regclass)->>'node_count')::int
  AS document_node_count,
       (turbohybrid_index_stats('mv_document_node_docs_idx'::regclass)->>'build_fast_edges')::boolean
  AS document_node_build_fast_edges,
       turbohybrid_index_stats('mv_document_node_docs_idx'::regclass)->>'build_neighbor_select_reason'
  AS document_node_build_neighbor_select_reason;

SET turbohybrid.dense_build_neighbor_select = heuristic;

CREATE INDEX mv_document_node_docs_explicit_heuristic_idx ON mv_document_node_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops)
  WITH (multivector_graph = document_nodes, graph_ef_search = 2);

SELECT turbohybrid_index_stats('mv_document_node_docs_explicit_heuristic_idx'::regclass)->>'build_neighbor_select'
  AS document_node_explicit_neighbor_select,
       (turbohybrid_index_stats('mv_document_node_docs_explicit_heuristic_idx'::regclass)->>'build_fast_edges')::boolean
  AS document_node_explicit_build_fast_edges,
       turbohybrid_index_stats('mv_document_node_docs_explicit_heuristic_idx'::regclass)->>'build_neighbor_select_reason'
  AS document_node_explicit_build_neighbor_select_reason;

SET turbohybrid.dense_build_neighbor_select = auto;

CREATE INDEX mv_document_node_docs_exact_symmetric_idx ON mv_document_node_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops)
  WITH (multivector_graph = document_nodes,
        multivector_doc_build_scorer = exact_symmetric,
        graph_ef_search = 2);

SELECT (turbohybrid_last_build_stats()->>'multivector_doc_exact_build_distance_calls')::bigint > 0
  AS exact_document_node_exact_build_calls_positive,
       (turbohybrid_last_build_stats()->>'multivector_doc_exact_build_distance_us')::bigint >= 0
  AS exact_document_node_exact_build_us_recorded;

SELECT turbohybrid_index_stats('mv_document_node_docs_exact_symmetric_idx'::regclass)->>'multivector_doc_build_scorer'
  AS explicit_document_node_build_scorer,
       turbohybrid_index_stats('mv_document_node_docs_exact_symmetric_idx'::regclass)->>'build_neighbor_select'
  AS exact_document_node_neighbor_select,
       (turbohybrid_index_stats('mv_document_node_docs_exact_symmetric_idx'::regclass)->>'build_fast_edges')::boolean
  AS exact_document_node_build_fast_edges,
       turbohybrid_index_stats('mv_document_node_docs_exact_symmetric_idx'::regclass)->>'build_neighbor_select_reason'
  AS exact_document_node_build_neighbor_select_reason;

SET turbohybrid.multivector_exact_symmetric_build_max_docs = 2;

DO $$
DECLARE
	err_message text;
	err_hint text;
BEGIN
	BEGIN
		EXECUTE '
			CREATE INDEX mv_document_node_docs_exact_symmetric_guard_idx
			ON mv_document_node_docs USING turbohybrid
			(colbert multivector_cosine_turbohybrid_ops)
			WITH (multivector_graph = document_nodes,
			      multivector_doc_build_scorer = exact_symmetric,
			      graph_ef_search = 2)';
		RAISE EXCEPTION 'expected exact_symmetric scale guard to fail';
	EXCEPTION WHEN feature_not_supported THEN
		GET STACKED DIAGNOSTICS
			err_message = MESSAGE_TEXT,
			err_hint = PG_EXCEPTION_HINT;
		IF err_message <> 'exact symmetric multivector document graph build is not allowed at this scale' THEN
			RAISE EXCEPTION 'unexpected exact_symmetric scale guard message: %', err_message;
		END IF;
		IF err_hint <> 'Use multivector_doc_build_scorer = proxy for production builds, or set turbohybrid.multivector_allow_exact_symmetric_build = on for diagnostic experiments.' THEN
			RAISE EXCEPTION 'unexpected exact_symmetric scale guard hint: %', err_hint;
		END IF;
	END;
END
$$;

CREATE INDEX mv_document_node_docs_proxy_low_threshold_idx ON mv_document_node_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops)
  WITH (multivector_graph = document_nodes,
        multivector_doc_build_scorer = proxy,
        graph_ef_search = 2);

RESET turbohybrid.multivector_exact_symmetric_build_max_docs;

SET max_parallel_maintenance_workers = 2;
SET turbohybrid.native_build_workers = '2';
SET turbohybrid.native_parallel_edge_build = auto;

CREATE TABLE mv_document_node_parallel_docs (
  id int PRIMARY KEY,
  colbert turbohybrid_multivector
);

INSERT INTO mv_document_node_parallel_docs
SELECT i,
       turbohybrid_multivector(ARRAY[
         ('[' || ((i % 7)::float8 / 7.0)::text || ',' ||
                 ((i % 11)::float8 / 11.0)::text || ',' ||
                 ((i % 13)::float8 / 13.0)::text || ']')::vector(3),
         ('[' || (((i + 3) % 7)::float8 / 7.0)::text || ',' ||
                 (((i + 5) % 11)::float8 / 11.0)::text || ',' ||
                 (((i + 7) % 13)::float8 / 13.0)::text || ']')::vector(3)
       ])
FROM generate_series(1, 200) AS i;

CREATE INDEX mv_document_node_parallel_proxy_idx ON mv_document_node_parallel_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops)
  WITH (multivector_graph = document_nodes, exact_storage = off, graph_ef_search = 2);

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_last_build_stats();
	IF stats->>'relation_name' <> 'mv_document_node_parallel_proxy_idx' OR
		(stats->>'node_count')::int <> 200 OR
		turbohybrid_index_stats('mv_document_node_parallel_proxy_idx'::regclass)->>'multivector_doc_build_scorer' <> 'proxy' OR
		(stats->>'native_build_workers_requested')::int <> 2 OR
		(stats->>'parallel_scan_enabled')::boolean IS DISTINCT FROM false OR
		(stats->>'parallel_encode_enabled')::boolean IS DISTINCT FROM false THEN
		RAISE EXCEPTION 'unexpected document-node proxy worker stats: %', stats;
	END IF;

	IF (stats->>'parallel_edge_build_enabled')::boolean THEN
		IF stats->>'parallel_edge_build_disabled_reason' <> 'none' OR
			(stats->>'parallel_edge_workers_launched')::int < 1 THEN
			RAISE EXCEPTION 'unexpected enabled document-node proxy edge stats: %', stats;
		END IF;
	ELSE
		IF stats->>'parallel_edge_build_disabled_reason' = 'none' THEN
			RAISE EXCEPTION 'missing document-node proxy edge disable reason: %', stats;
		END IF;
	END IF;
END
$$;

CREATE TABLE mv_document_node_proxy_regression_docs (
  id int PRIMARY KEY,
  colbert turbohybrid_multivector
);

INSERT INTO mv_document_node_proxy_regression_docs
SELECT doc_id,
       turbohybrid_multivector(array_agg(v ORDER BY token_ordinal))
FROM (
  SELECT doc_id,
         token_ordinal,
         ('[' || string_agg(
           (((doc_id * 131 + token_ordinal * 17 + dim * 7) % 2000)::float8 / 1000.0 - 1.0)::text,
           ',' ORDER BY dim
         ) || ']')::vector(128) AS v
  FROM generate_series(1, 200) AS doc_ids(doc_id)
  CROSS JOIN LATERAL generate_series(1, 32 + (doc_id % 33)) AS tokens(token_ordinal)
  CROSS JOIN LATERAL generate_series(1, 128) AS dims(dim)
  GROUP BY doc_id, token_ordinal
) AS vectors
GROUP BY doc_id;

CREATE INDEX mv_document_node_proxy_regression_idx ON mv_document_node_proxy_regression_docs USING turbohybrid
  (colbert multivector_maxsim_ip_turbohybrid_ops)
  WITH (quantization_bits = 4,
        exact_storage = off,
        multivector_graph = document_nodes,
        multivector_doc_build_scorer = proxy);

DO $$
DECLARE
	build_stats jsonb;
	index_stats jsonb;
	distance_calls bigint;
BEGIN
	build_stats := turbohybrid_last_build_stats();
	index_stats := turbohybrid_index_stats('mv_document_node_proxy_regression_idx'::regclass);
	distance_calls := (build_stats->>'build_edges_distance_calls')::bigint;

	IF (index_stats->>'node_count')::int <> 200 OR
		(build_stats->>'node_count')::int <> 200 THEN
		RAISE EXCEPTION 'document-node proxy regression node count mismatch, build %, index %',
			build_stats, index_stats;
	END IF;

	IF index_stats->>'multivector_doc_build_scorer' <> 'proxy' OR
		build_stats->>'multivector_doc_build_scorer' <> 'proxy' THEN
		RAISE EXCEPTION 'document-node proxy regression used unexpected build scorer, build %, index %',
			build_stats, index_stats;
	END IF;

	IF (build_stats->>'multivector_doc_exact_build_distance_calls')::bigint <> 0 THEN
		RAISE EXCEPTION 'document-node proxy regression used exact document build distance: %',
			build_stats;
	END IF;

	IF (index_stats->>'build_fast_edges')::boolean IS DISTINCT FROM true OR
		(build_stats->>'build_fast_edges')::boolean IS DISTINCT FROM true THEN
		RAISE EXCEPTION 'document-node proxy regression did not use fast edges by default, build %, index %',
			build_stats, index_stats;
	END IF;

	IF distance_calls <= 0 OR distance_calls > 100000 THEN
		RAISE EXCEPTION 'document-node proxy regression distance call count outside ceiling: %',
			build_stats;
	END IF;

	IF NOT (build_stats ? 'parallel_edge_build_enabled') AND
		NOT (index_stats ? 'build_worker_count') THEN
		RAISE EXCEPTION 'document-node proxy regression missing worker/parallel stats, build %, index %',
			build_stats, index_stats;
	END IF;
END
$$;

SET enable_seqscan = off;
SET turbohybrid.multivector_plain_fallback = off;
SET turbohybrid.multivector_candidate_source = graph;
SET turbohybrid.multivector_doc_candidate_k = 200;
SET turbohybrid.multivector_exact_rerank_k = 200;
SET turbohybrid.multivector_doc_graph_search_ef = 200;
SET turbohybrid.multivector_doc_graph_rescore_k = 200;
SET turbohybrid.multivector_doc_graph_oversampling = 1;

DO $$
DECLARE
	q turbohybrid_multivector;
	index_ids int[];
	brute_ids int[];
	stats jsonb;
BEGIN
	SELECT colbert INTO q
	FROM mv_document_node_proxy_regression_docs
	WHERE id = 73;

	SELECT array_agg(id ORDER BY distance, id)
	INTO index_ids
	FROM (
		SELECT id,
		       colbert <~> turbohybrid_query(
		         multivector_query => q,
		         dense_k => 200,
		         final_k => 10
		       ) AS distance
		FROM mv_document_node_proxy_regression_docs
		ORDER BY distance, id
		LIMIT 10
	) AS ranked;

	SELECT array_agg(id ORDER BY distance, id)
	INTO brute_ids
	FROM (
		SELECT id,
		       turbohybrid_multivector_maxsim_distance(q, colbert) AS distance
		FROM mv_document_node_proxy_regression_docs
		ORDER BY distance, id
		LIMIT 10
	) AS ranked;

	stats := turbohybrid_last_scan_stats();
	IF index_ids <> brute_ids THEN
		RAISE EXCEPTION 'document-node proxy regression rerank disagrees with brute force, index %, brute %, stats %',
			index_ids, brute_ids, stats;
	END IF;

	IF stats->>'multivector_graph_mode' <> 'document_nodes' OR
		stats->>'multivector_candidate_source' <> 'graph' OR
		stats->>'multivector_candidate_path' NOT IN ('proxy_graph', 'exact_doc_scan') OR
		(stats->>'multivector_subvector_searches')::int <> 0 OR
		stats->>'multivector_exact_rerank_source' <> 'sidecar' OR
		(stats->>'multivector_exact_rerank_heap_fetches')::int <> 0 OR
		(stats->>'multivector_doc_graph_exact_rerank_docs')::int < 10 THEN
		RAISE EXCEPTION 'document-node proxy regression used unexpected query path: %',
			stats;
	END IF;
END
$$;

RESET turbohybrid.multivector_doc_graph_oversampling;
RESET turbohybrid.multivector_doc_graph_rescore_k;
RESET turbohybrid.multivector_doc_graph_search_ef;
RESET turbohybrid.multivector_exact_rerank_k;
RESET turbohybrid.multivector_doc_candidate_k;
RESET turbohybrid.multivector_candidate_source;
RESET turbohybrid.multivector_plain_fallback;
RESET enable_seqscan;
DROP TABLE mv_document_node_proxy_regression_docs;

CREATE INDEX mv_document_node_parallel_exact_idx ON mv_document_node_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops)
  WITH (multivector_graph = document_nodes,
        multivector_doc_build_scorer = exact_symmetric,
        exact_storage = off,
        graph_ef_search = 2);

SELECT turbohybrid_last_build_stats()->>'parallel_edge_build_disabled_reason'
  AS exact_document_node_parallel_disabled_reason;

CREATE INDEX mv_document_node_parallel_token_idx ON mv_document_node_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops)
  WITH (multivector_graph = token_nodes, exact_storage = off);

SELECT turbohybrid_last_build_stats()->>'parallel_edge_build_disabled_reason'
  AS token_node_parallel_disabled_reason;

RESET turbohybrid.native_parallel_edge_build;
RESET turbohybrid.native_build_workers;
RESET max_parallel_maintenance_workers;
DROP TABLE mv_document_node_parallel_docs;

DO $$
BEGIN
	BEGIN
		EXECUTE '
			CREATE INDEX mv_document_node_docs_bad_build_scorer_idx
			ON mv_document_node_docs USING turbohybrid
			(colbert multivector_cosine_turbohybrid_ops)
			WITH (multivector_graph = document_nodes,
			      multivector_doc_build_scorer = nope)';
		RAISE EXCEPTION 'expected invalid multivector_doc_build_scorer to fail';
	EXCEPTION WHEN invalid_parameter_value THEN
		NULL;
	END;
END
$$;

SET enable_seqscan = off;
SET turbohybrid.multivector_plain_fallback = off;
SELECT id AS document_node_top1
FROM mv_document_node_docs
ORDER BY colbert <~> turbohybrid_query(
  multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
  dense_k => 3,
  final_k => 1
)
LIMIT 1;

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_last_scan_stats();
	IF stats->>'multivector_graph_mode' <> 'document_nodes' OR
		stats->>'multivector_candidate_source' <> 'graph' OR
		stats->>'multivector_candidate_path' <> 'proxy_graph' OR
		stats->>'multivector_docmap_source' <> 'sidecar' OR
		(stats->>'multivector_proxy_graph_searches')::int <> 1 OR
		(stats->>'multivector_subvector_searches')::int <> 0 OR
		(stats->>'multivector_doc_graph_nodes')::int <> 3 OR
		(stats->>'multivector_doc_graph_search_ef')::int <> 2 OR
		(stats->>'multivector_doc_graph_candidates')::int <> 3 OR
		(stats->>'multivector_doc_graph_quantized_scores')::int <> 0 OR
		(stats->>'multivector_doc_graph_exact_rerank_docs')::int <> 3 OR
		stats->>'multivector_doc_graph_rescore_source' <> 'sidecar' OR
		stats->>'multivector_exact_rerank_source' <> 'sidecar' OR
		(stats->>'multivector_doc_graph_heap_fetches')::int <> 0 OR
		(stats->>'multivector_exact_rerank_heap_fetches')::int <> 0 OR
		stats->>'multivector_doc_graph_warning' <>
			'document_node_proxy_vector_graph_traversal' THEN
		RAISE EXCEPTION 'expected document-node proxy graph scan stats, got %',
			stats;
	END IF;
END
$$;

DO $$
DECLARE
	q turbohybrid_multivector := turbohybrid_multivector(ARRAY[
		'[1,0]'::vector,
		'[0,1]'::vector
	]);
	index_ids int[];
	brute_ids int[];
	result_count int;
	distinct_count int;
BEGIN
	PERFORM set_config('turbohybrid.multivector_doc_graph_search_ef', '3', true);
	SELECT array_agg(id ORDER BY id), count(*), count(DISTINCT id)
	INTO index_ids, result_count, distinct_count
	FROM (
		SELECT id
		FROM mv_document_node_docs
		ORDER BY colbert <~> turbohybrid_query(
			multivector_query => q,
			dense_k => 3,
			final_k => 3
		)
		LIMIT 3
	) s;

	SELECT array_agg(id ORDER BY id)
	INTO brute_ids
	FROM (
		SELECT id
		FROM mv_document_node_docs
		ORDER BY turbohybrid_multivector_maxsim_distance(q, colbert), id
		LIMIT 3
	) s;

	IF result_count <> distinct_count THEN
		RAISE EXCEPTION 'document-node proxy graph returned duplicate docs: %',
			index_ids;
	END IF;
	IF index_ids <> brute_ids THEN
		RAISE EXCEPTION 'document-node proxy graph rerank disagrees with brute force, index %, brute %',
			index_ids, brute_ids;
	END IF;
	PERFORM set_config('turbohybrid.multivector_doc_graph_search_ef', '0', true);
END
$$;

SET turbohybrid.multivector_doc_graph_rescore_k = 1;
SELECT id AS document_node_graph_traversal_probe
FROM mv_document_node_docs
ORDER BY colbert <~> turbohybrid_query(
  multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
  dense_k => 1,
  final_k => 1
)
LIMIT 1;

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_last_scan_stats();
	IF stats->>'multivector_graph_mode' <> 'document_nodes' OR
		stats->>'multivector_candidate_path' <> 'proxy_graph' OR
		stats->>'multivector_doc_graph_warning' <>
			'document_node_proxy_vector_graph_traversal' OR
		(stats->>'multivector_proxy_graph_searches')::int <> 1 OR
		(stats->>'multivector_subvector_searches')::int <> 0 OR
		(stats->>'multivector_doc_graph_nodes')::int <> 3 OR
		(stats->>'multivector_doc_graph_candidates')::int < 1 OR
		(stats->>'multivector_doc_graph_exact_rerank_docs')::int < 1 OR
		stats->>'multivector_doc_graph_rescore_source' <> 'sidecar' OR
		stats->>'multivector_exact_rerank_source' <> 'sidecar' OR
		(stats->>'multivector_doc_graph_heap_fetches')::int <> 0 OR
		(stats->>'multivector_exact_rerank_heap_fetches')::int <> 0 OR
		(stats->>'multivector_exact_rerank_sidecar_reads')::int < 1 OR
		(stats->>'multivector_exact_rerank_sidecar_bytes')::int < 1 THEN
		RAISE EXCEPTION 'expected non-exhaustive document-node graph traversal stats, got %',
			stats;
	END IF;
END
$$;
RESET turbohybrid.multivector_doc_graph_rescore_k;

SET turbohybrid.multivector_plain_fallback = auto;
SET turbohybrid.multivector_plain_fallback_max_docs = 0;
SET turbohybrid.multivector_plain_fallback_candidate_fraction = 0.5;
SET turbohybrid.multivector_doc_graph_rescore_k = 2;
SET turbohybrid.multivector_doc_graph_oversampling = 2;
SELECT id AS document_node_auto_full_candidate_fallback
FROM mv_document_node_docs
ORDER BY colbert <~> turbohybrid_query(
  multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
  dense_k => 1,
  final_k => 1
)
LIMIT 1;

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_last_scan_stats();
	IF stats->>'multivector_plain_fallback_used' <> 'true' OR
		stats->>'multivector_plain_fallback_reason' <> 'document_node_candidate_fraction' OR
		(stats->>'multivector_plain_fallback_docs_scored')::int <> 3 OR
		(stats->>'multivector_doc_graph_edges_visited')::int <> 0 THEN
		RAISE EXCEPTION 'expected document-node expanded candidate fallback stats, got %',
			stats;
	END IF;
END
$$;
RESET turbohybrid.multivector_doc_graph_oversampling;
RESET turbohybrid.multivector_doc_graph_rescore_k;
RESET turbohybrid.multivector_plain_fallback_candidate_fraction;
RESET turbohybrid.multivector_plain_fallback_max_docs;
SET turbohybrid.multivector_plain_fallback = off;
RESET enable_seqscan;

INSERT INTO mv_document_node_docs VALUES
  (4, turbohybrid_multivector(ARRAY['[-1,0]'::vector, '[-0.8,-0.2]'::vector]));

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_index_stats('mv_document_node_docs_idx'::regclass);
	IF (stats->>'node_count')::int <> 4 OR
		stats->>'multivector_graph_mode' <> 'document_nodes' THEN
		RAISE EXCEPTION 'expected document-node insert to append one graph node, got %',
			stats;
	END IF;
END
$$;

UPDATE mv_document_node_docs
SET colbert = turbohybrid_multivector(ARRAY['[0,1]'::vector, '[0.1,0.9]'::vector])
WHERE id = 4;
DELETE FROM mv_document_node_docs WHERE id = 2;
VACUUM mv_document_node_docs;

SET enable_seqscan = off;
SELECT id <> 2 AS document_node_post_vacuum_visible
FROM mv_document_node_docs
ORDER BY colbert <~> turbohybrid_query(
  multivector_query => turbohybrid_multivector(ARRAY['[0,1]'::vector]),
  dense_k => 4,
  final_k => 1
)
LIMIT 1;

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_last_scan_stats();
	IF stats->>'multivector_graph_mode' <> 'document_nodes' OR
		stats->>'multivector_docmap_source' <> 'sidecar' OR
		(stats->>'multivector_doc_graph_candidates')::int < 1 OR
		(stats->>'multivector_doc_graph_exact_rerank_docs')::int < 1 THEN
		RAISE EXCEPTION 'expected document-node post-vacuum visibility to use sidecar graph path, got %',
			stats;
	END IF;
END
$$;
RESET enable_seqscan;
RESET turbohybrid.multivector_plain_fallback;

-- Regression: multivector docmap sidecar must survive VACUUM with heavy churn.
-- Dead-node compaction is skipped for docmap-backed indexes (nodeId remapping
-- would invalidate the sidecar).  Page-bloat-only compaction (no dead nodes)
-- is safe because nodeIdMap is identity.  This test exercises delete + update
-- + VACUUM and confirms the index remains queryable without "docmap sidecar
-- is invalid".
DELETE FROM mv_document_node_docs WHERE id = 1;
UPDATE mv_document_node_docs
SET colbert = turbohybrid_multivector(ARRAY['[-0.5,0.5]'::vector])
WHERE id = 3;
VACUUM mv_document_node_docs;

SET enable_seqscan = off;
SELECT id <> 1 AS document_node_compaction_guard_survived
FROM mv_document_node_docs
ORDER BY colbert <~> turbohybrid_query(
  multivector_query => turbohybrid_multivector(ARRAY['[-0.5,0.5]'::vector]),
  dense_k => 4,
  final_k => 1
)
LIMIT 1;
RESET enable_seqscan;

-- Regression: dead_node_ratio must be visible after VACUUM on document-node
-- indexes.  The stability-check.sh bloat monitor reads this stat to schedule
-- REINDEX before p95 degrades.  After deleting id=1 and updating id=3, at
-- least one node must be marked dead and counted.
DO $$
DECLARE
	stats jsonb;
	dead_ratio numeric;
	dead_count int;
	live_count int;
	node_count int;
BEGIN
	stats := turbohybrid_index_stats('mv_document_node_docs_idx'::regclass);
	dead_ratio := COALESCE((stats->>'dead_node_ratio')::numeric, -1);
	dead_count := COALESCE((stats->>'dead_node_count')::int, -1);
	live_count := COALESCE((stats->>'live_node_count')::int, -1);
	node_count := (stats->>'node_count')::int;

	IF dead_count < 1 THEN
		RAISE EXCEPTION 'expected dead_node_count >= 1 after VACUUM churn, got % (stats=%)',
			dead_count, stats;
	END IF;

	IF dead_ratio <= 0 THEN
		RAISE EXCEPTION 'expected dead_node_ratio > 0 after VACUUM churn, got % (stats=%)',
			dead_ratio, stats;
	END IF;

	IF live_count + dead_count <> node_count THEN
		RAISE EXCEPTION 'live_node_count + dead_node_count <> node_count: % + % <> % (stats=%)',
			live_count, dead_count, node_count, stats;
	END IF;
END
$$;

DROP TABLE mv_document_node_docs;

CREATE TABLE mv_document_node_pool_docs (
  id int PRIMARY KEY,
  colbert turbohybrid_multivector
);

INSERT INTO mv_document_node_pool_docs VALUES
  (1, turbohybrid_multivector(ARRAY[
    '[1,0,0,0]'::vector,
    '[0.98,0.02,0,0]'::vector,
    '[0,1,0,0]'::vector,
    '[0.02,0.98,0,0]'::vector,
    '[0,0,1,0]'::vector,
    '[0,0,0.98,0.02]'::vector,
    '[0,0,0,1]'::vector,
    '[0,0,0.02,0.98]'::vector
  ])),
  (2, turbohybrid_multivector(ARRAY[
    '[0,0,1,0]'::vector,
    '[0,0,0.98,0.02]'::vector,
    '[0,0,0,1]'::vector,
    '[0,0,0.02,0.98]'::vector,
    '[-1,0,0,0]'::vector,
    '[-0.98,-0.02,0,0]'::vector,
    '[0,-1,0,0]'::vector,
    '[-0.02,-0.98,0,0]'::vector
  ]));

CREATE INDEX mv_document_node_pool_docs_idx ON mv_document_node_pool_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops)
  WITH (multivector_graph = document_nodes,
        multivector_token_pooling = greedy_cosine,
        multivector_token_pooling_target_ratio = 0.5,
        multivector_token_pooling_min_tokens = 4,
        graph_ef_search = 2);

SET enable_seqscan = off;
SET turbohybrid.multivector_plain_fallback = off;
SELECT id AS document_node_pooling_top1
FROM mv_document_node_pool_docs
ORDER BY colbert <~> turbohybrid_query(
  multivector_query => turbohybrid_multivector(ARRAY[
    '[1,0,0,0]'::vector,
    '[0,1,0,0]'::vector,
    '[0,0,1,0]'::vector,
    '[0,0,0,1]'::vector
  ]),
  dense_k => 2,
  final_k => 1
)
LIMIT 1;

DO $$
DECLARE
	stats jsonb;
	original_tokens int;
	pooled_tokens int;
	pooling_ratio numeric;
BEGIN
	stats := turbohybrid_last_scan_stats();
	original_tokens := (stats->>'multivector_tokens_original')::int;
	pooled_tokens := (stats->>'multivector_tokens_pooled')::int;
	pooling_ratio := (stats->>'multivector_token_pooling_ratio')::numeric;

	IF stats->>'multivector_graph_mode' <> 'document_nodes' OR
		original_tokens <> 16 OR
		pooled_tokens <> 8 OR
		pooling_ratio < 0.49 OR
		pooling_ratio > 0.51 OR
		(stats->>'multivector_doc_graph_exact_rerank_docs')::int < 1 THEN
		RAISE EXCEPTION 'expected pooled document-node scan stats, got %',
			stats;
	END IF;
END
$$;
RESET enable_seqscan;
RESET turbohybrid.multivector_plain_fallback;

DROP INDEX mv_document_node_pool_docs_idx;

CREATE INDEX mv_document_node_pool_docs_kmeans_idx ON mv_document_node_pool_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops)
  WITH (multivector_graph = document_nodes,
        multivector_token_pooling = kmeans,
        multivector_token_pooling_target_ratio = 0.5,
        multivector_token_pooling_min_tokens = 4,
        graph_ef_search = 2);

SET enable_seqscan = off;
SET turbohybrid.multivector_plain_fallback = off;
SELECT id AS document_node_kmeans_pooling_top1
FROM mv_document_node_pool_docs
ORDER BY colbert <~> turbohybrid_query(
  multivector_query => turbohybrid_multivector(ARRAY[
    '[1,0,0,0]'::vector,
    '[0,1,0,0]'::vector,
    '[0,0,1,0]'::vector,
    '[0,0,0,1]'::vector
  ]),
  dense_k => 2,
  final_k => 1
)
LIMIT 1;

DO $$
DECLARE
	stats jsonb;
	original_tokens int;
	pooled_tokens int;
	pooling_ratio numeric;
BEGIN
	stats := turbohybrid_last_scan_stats();
	original_tokens := (stats->>'multivector_tokens_original')::int;
	pooled_tokens := (stats->>'multivector_tokens_pooled')::int;
	pooling_ratio := (stats->>'multivector_token_pooling_ratio')::numeric;

	IF stats->>'multivector_graph_mode' <> 'document_nodes' OR
		original_tokens <> 16 OR
		pooled_tokens <> 8 OR
		pooling_ratio < 0.49 OR
		pooling_ratio > 0.51 OR
		(stats->>'multivector_doc_graph_exact_rerank_docs')::int < 1 THEN
		RAISE EXCEPTION 'expected kmeans pooled document-node scan stats, got %',
			stats;
	END IF;
END
$$;
RESET enable_seqscan;
RESET turbohybrid.multivector_plain_fallback;

DO $$
BEGIN
	BEGIN
		EXECUTE '
			CREATE INDEX mv_document_node_pool_bad_ratio_idx
			ON mv_document_node_pool_docs USING turbohybrid
			(colbert multivector_cosine_turbohybrid_ops)
			WITH (multivector_graph = document_nodes,
			      multivector_token_pooling = greedy_cosine,
			      multivector_token_pooling_target_ratio = 0)';
		RAISE EXCEPTION 'expected invalid multivector_token_pooling_target_ratio to fail';
	EXCEPTION WHEN invalid_parameter_value THEN
		NULL;
	END;
END
$$;

DROP TABLE mv_document_node_pool_docs;

SELECT turbohybrid_multivector(ARRAY['[1,0]'::vector]) <~>
  turbohybrid_query(vector_query => '[1,0]'::vector);

SET enable_seqscan = off;
SELECT id FROM mv_docs
  ORDER BY colbert <~> turbohybrid_query(
    vector_query => '[1,0]'::vector,
    dense_k => 1,
    final_k => 1
  )
  LIMIT 1;
RESET enable_seqscan;

DROP TABLE mv_docs;

DO $$
DECLARE
	doc turbohybrid_multivector :=
		turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector]);
	query_two turbohybrid_multivector :=
		turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector]);
	query_one turbohybrid_multivector :=
		turbohybrid_multivector(ARRAY['[1,0]'::vector]);
	unweighted float8;
	weighted_ones float8;
	masked float8;
	one_token float8;
	unweighted_top int;
	weighted_top int;
BEGIN
	unweighted := turbohybrid_multivector_distance(
		doc,
		turbohybrid_query(multivector_query => query_two));
	weighted_ones := turbohybrid_multivector_distance(
		doc,
		turbohybrid_query(
			multivector_query => query_two,
			query_token_weights => ARRAY[1,1]::real[]));
	masked := turbohybrid_multivector_distance(
		doc,
		turbohybrid_query(
			multivector_query => query_two,
			query_token_mask => ARRAY[false,true]));
	one_token := turbohybrid_multivector_distance(
		doc,
		turbohybrid_query(multivector_query => query_one));

	IF abs(unweighted - weighted_ones) > 0.000001 THEN
		RAISE EXCEPTION 'all-one query token weights changed score: % vs %',
			unweighted, weighted_ones;
	END IF;
	IF abs(masked - one_token) > 0.000001 THEN
		RAISE EXCEPTION 'masked query token affected score: % vs %',
			masked, one_token;
	END IF;

	WITH docs(id, mv) AS (
		VALUES
			(1, turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0.5,0.5]'::vector])),
			(2, turbohybrid_multivector(ARRAY['[0.4,0]'::vector, '[0,1]'::vector]))
	)
	SELECT id INTO unweighted_top
	FROM docs
	ORDER BY mv <~> turbohybrid_query(multivector_query => query_two)
	LIMIT 1;

	WITH docs(id, mv) AS (
		VALUES
			(1, turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0.5,0.5]'::vector])),
			(2, turbohybrid_multivector(ARRAY['[0.4,0]'::vector, '[0,1]'::vector]))
	)
	SELECT id INTO weighted_top
	FROM docs
	ORDER BY mv <~> turbohybrid_query(
		multivector_query => query_two,
		query_token_weights => ARRAY[1,3]::real[])
	LIMIT 1;

	IF unweighted_top <> 1 OR weighted_top <> 2 THEN
		RAISE EXCEPTION 'expected query token weights to change ranking, got unweighted %, weighted %',
			unweighted_top, weighted_top;
	END IF;
END
$$;

CREATE TABLE mv_query_token_mask_docs (
	id int,
	colbert turbohybrid_multivector
);

INSERT INTO mv_query_token_mask_docs VALUES
	(1, turbohybrid_multivector(ARRAY['[1,0]'::vector])),
	(2, turbohybrid_multivector(ARRAY['[0,1]'::vector]));

CREATE INDEX mv_query_token_mask_docs_idx ON mv_query_token_mask_docs
USING turbohybrid (colbert multivector_cosine_turbohybrid_ops)
WITH (multivector_graph = token_nodes, graph_ef_search = 2);

SET enable_seqscan = off;
DO $$
DECLARE
	top_id int;
BEGIN
	SELECT id INTO top_id
	FROM mv_query_token_mask_docs
	ORDER BY colbert <~> turbohybrid_query(
		multivector_query => turbohybrid_multivector(ARRAY[
			'[1,0]'::vector,
			'[0,1]'::vector
		]),
		query_token_mask => ARRAY[false,true],
		dense_k => 1,
		final_k => 1
	)
	LIMIT 1;

	IF top_id <> 1 THEN
		RAISE EXCEPTION 'masked token-node query returned %, expected 1',
			top_id;
	END IF;
END
$$;
RESET enable_seqscan;

DROP TABLE mv_query_token_mask_docs;

CREATE TABLE mv_query_token_weight_docnodes (
	id int,
	colbert turbohybrid_multivector
);

INSERT INTO mv_query_token_weight_docnodes VALUES
	(1, turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0.5,0.5]'::vector])),
	(2, turbohybrid_multivector(ARRAY['[0.4,0]'::vector, '[0,1]'::vector]));

CREATE INDEX mv_query_token_weight_docnodes_idx ON mv_query_token_weight_docnodes
USING turbohybrid (colbert multivector_maxsim_ip_turbohybrid_ops)
WITH (multivector_graph = document_nodes, graph_ef_search = 4);

SET enable_seqscan = off;
SET turbohybrid.multivector_candidate_source = 'document_nodes';
SET turbohybrid.multivector_exact_rerank = off;
DO $$
DECLARE
	query_two turbohybrid_multivector :=
		turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector]);
	unweighted_top int;
	weighted_top int;
	masked_top int;
BEGIN
	SELECT id INTO unweighted_top
	FROM mv_query_token_weight_docnodes
	ORDER BY colbert <~> turbohybrid_query(
		multivector_query => query_two,
		final_k => 1
	)
	LIMIT 1;

	SELECT id INTO weighted_top
	FROM mv_query_token_weight_docnodes
	ORDER BY colbert <~> turbohybrid_query(
		multivector_query => query_two,
		query_token_weights => ARRAY[1,3]::real[],
		final_k => 1
	)
	LIMIT 1;

	SELECT id INTO masked_top
	FROM mv_query_token_weight_docnodes
	ORDER BY colbert <~> turbohybrid_query(
		multivector_query => query_two,
		query_token_mask => ARRAY[false,true],
		final_k => 1
	)
	LIMIT 1;

	IF unweighted_top <> 1 OR weighted_top <> 2 OR masked_top <> 1 THEN
		RAISE EXCEPTION 'expected document-node query token sidecar ranking 1/2/1, got %/%/%',
			unweighted_top, weighted_top, masked_top;
	END IF;
END
$$;
RESET turbohybrid.multivector_exact_rerank;
RESET turbohybrid.multivector_candidate_source;
RESET enable_seqscan;

DROP TABLE mv_query_token_weight_docnodes;

CREATE TABLE mv_centroid_lite_docs (
	id int,
	colbert turbohybrid_multivector
);

INSERT INTO mv_centroid_lite_docs VALUES
	(1, turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0.8,0.2]'::vector, '[0,1]'::vector, '[0.2,0.8]'::vector])),
	(2, turbohybrid_multivector(ARRAY['[0,1]'::vector, '[0.1,0.9]'::vector, '[-1,0]'::vector, '[-0.8,0.2]'::vector])),
	(3, turbohybrid_multivector(ARRAY['[-1,0]'::vector, '[-0.9,0.1]'::vector, '[0,-1]'::vector, '[0.1,-0.9]'::vector]));

CREATE INDEX mv_centroid_lite_docs_idx ON mv_centroid_lite_docs
USING turbohybrid (colbert multivector_cosine_turbohybrid_ops)
WITH (multivector_graph = document_nodes,
      multivector_centroids = kmeans,
      multivector_centroid_count = 2,
      graph_ef_search = 2);

SET enable_seqscan = off;
DO $$
DECLARE
	top_id int;
	stats jsonb;
BEGIN
	PERFORM set_config('turbohybrid.multivector_candidate_source', 'centroid_lite', true);
	PERFORM set_config('turbohybrid.multivector_plain_fallback', 'off', true);
	PERFORM set_config('turbohybrid.multivector_doc_candidate_k', '3', true);
	SELECT id INTO top_id
	FROM mv_centroid_lite_docs
	ORDER BY colbert <~> turbohybrid_query(
		multivector_query => turbohybrid_multivector(ARRAY[
			'[1,0]'::vector,
			'[0,1]'::vector
		]),
		dense_k => 3,
		final_k => 1
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();

	IF top_id <> 1 OR
		stats->>'multivector_candidate_source' <> 'centroid_lite' OR
		stats->>'multivector_doc_graph_warning' <> 'document_node_centroid_lite_prefilter' OR
		(stats->>'centroid_docs_touched')::int <= 0 OR
		(stats->>'centroid_lists_visited')::int <= 0 OR
		NOT (stats ? 'centroid_pruned_docs') OR
		(stats->>'centroid_candidates')::int <= 0 THEN
		RAISE EXCEPTION 'centroid_lite scan failed, top %, stats %',
			top_id, stats;
	END IF;
END
$$;
RESET enable_seqscan;

INSERT INTO mv_centroid_lite_docs VALUES
	(4, turbohybrid_multivector(ARRAY['[0.6,0.8]'::vector, '[0.6,0.8]'::vector, '[0.5,0.86]'::vector, '[0.55,0.83]'::vector]));

SET enable_seqscan = off;
DO $$
DECLARE
	top_id int;
	stats jsonb;
BEGIN
	PERFORM set_config('turbohybrid.multivector_candidate_source', 'centroid_lite', true);
	PERFORM set_config('turbohybrid.multivector_plain_fallback', 'off', true);
	PERFORM set_config('turbohybrid.multivector_doc_candidate_k', '4', true);
	SELECT id INTO top_id
	FROM mv_centroid_lite_docs
	ORDER BY colbert <~> turbohybrid_query(
		multivector_query => turbohybrid_multivector(ARRAY[
			'[0.6,0.8]'::vector,
			'[0.6,0.8]'::vector
		]),
		dense_k => 4,
		final_k => 1
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();

	IF top_id <> 4 OR
		stats->>'multivector_candidate_source' <> 'centroid_lite' OR
		stats->>'multivector_doc_graph_warning' <> 'document_node_centroid_lite_prefilter' OR
		(stats->>'centroid_docs_touched')::int <= 0 THEN
		RAISE EXCEPTION 'centroid_lite insert sidecar scan failed, top %, stats %',
			top_id, stats;
	END IF;
END
$$;
RESET enable_seqscan;

DO $$
BEGIN
	CREATE TEMP TABLE mv_centroid_lite_bad_docs (
		id int,
		colbert turbohybrid_multivector
	);
	INSERT INTO mv_centroid_lite_bad_docs VALUES
		(1, turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector])),
		(2, turbohybrid_multivector(ARRAY['[-1,0]'::vector, '[0,-1]'::vector]));
	CREATE INDEX mv_centroid_lite_bad_docs_idx ON mv_centroid_lite_bad_docs
	USING turbohybrid (colbert multivector_cosine_turbohybrid_ops)
	WITH (multivector_graph = document_nodes);

	PERFORM set_config('enable_seqscan', 'off', true);
	PERFORM set_config('turbohybrid.multivector_candidate_source', 'centroid_lite', true);
	PERFORM set_config('turbohybrid.multivector_plain_fallback', 'force', true);
	PERFORM id
	FROM mv_centroid_lite_bad_docs
	ORDER BY colbert <~> turbohybrid_query(
		multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
		dense_k => 2,
		final_k => 1
	)
	LIMIT 1;
	RAISE EXCEPTION 'centroid_lite without centroid option unexpectedly executed';
EXCEPTION
	WHEN feature_not_supported THEN
		IF SQLERRM NOT LIKE 'centroid_lite multivector candidate source requires multivector_centroids = kmeans%' THEN
			RAISE EXCEPTION 'unexpected centroid_lite missing-centroid error: %', SQLERRM;
		END IF;
END
$$;

DO $$
DECLARE
	top_id int;
	stats jsonb;
BEGIN
	CREATE TEMP TABLE mv_centroid_lite_token_docs (
		id int,
		colbert turbohybrid_multivector
	);
	INSERT INTO mv_centroid_lite_token_docs VALUES
		(1, turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector])),
		(2, turbohybrid_multivector(ARRAY['[-1,0]'::vector, '[0,-1]'::vector]));
	CREATE INDEX mv_centroid_lite_token_docs_idx ON mv_centroid_lite_token_docs
	USING turbohybrid (colbert multivector_cosine_turbohybrid_ops)
	WITH (multivector_graph = token_nodes,
	      multivector_centroids = kmeans);

	PERFORM set_config('enable_seqscan', 'off', true);
	PERFORM set_config('turbohybrid.multivector_candidate_source', 'centroid_lite', true);
	PERFORM set_config('turbohybrid.multivector_plain_fallback', 'off', true);
	SELECT id INTO top_id
	FROM mv_centroid_lite_token_docs
	ORDER BY colbert <~> turbohybrid_query(
		multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
		dense_k => 2,
		final_k => 1
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();

	IF top_id <> 1 OR
		stats->>'multivector_candidate_source' <> 'centroid_lite' OR
		stats->>'multivector_doc_graph_warning' <> 'token_node_centroid_lite_exact_token_prefilter' OR
		(stats->>'centroid_lists_visited')::int <= 0 OR
		(stats->>'centroid_candidates')::int <= 0 THEN
		RAISE EXCEPTION 'token-node centroid_lite scan failed, top %, stats %',
			top_id, stats;
	END IF;
END
$$;

DO $$
DECLARE
	top_id int;
	stats jsonb;
BEGIN
	PERFORM set_config('enable_seqscan', 'off', true);
	PERFORM set_config('turbohybrid.multivector_candidate_source', 'quantized_inverted_experimental', true);
	PERFORM set_config('turbohybrid.multivector_plain_fallback', 'force', true);
	PERFORM set_config('turbohybrid.multivector_doc_candidate_k', '3', true);
	SELECT id INTO top_id
	FROM mv_centroid_lite_docs
	ORDER BY colbert <~> turbohybrid_query(
		multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
		dense_k => 3,
		final_k => 1
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();

	IF top_id <> 1 OR
		stats->>'multivector_candidate_source' <> 'quantized_inverted_experimental' OR
		stats->>'multivector_accumulator_kind' <> 'quantized_inverted_experimental' OR
		stats->>'multivector_doc_graph_warning' <> 'quantized_inverted_experimental_persisted_postings' OR
		(stats->>'quantized_inverted_lists_visited')::int <> 1 OR
		(stats->>'quantized_inverted_postings_touched')::int <= 0 OR
		(stats->>'quantized_inverted_docs_scored')::int <= 0 OR
		(stats->>'quantized_inverted_candidates')::int <= 0 OR
		(stats->>'quantized_inverted_exact_rerank_docs')::int <= 0 OR
		(stats->>'quantized_inverted_codebook_size')::int <> 4 THEN
		RAISE EXCEPTION 'quantized_inverted_experimental scan failed, top %, stats %',
			top_id, stats;
	END IF;
END
$$;

DO $$
BEGIN
	CREATE TEMP TABLE mv_quantized_inverted_token_docs (
		id int,
		colbert turbohybrid_multivector
	);
	INSERT INTO mv_quantized_inverted_token_docs VALUES
		(1, turbohybrid_multivector(ARRAY['[1,0]'::vector])),
		(2, turbohybrid_multivector(ARRAY['[-1,0]'::vector]));
	CREATE INDEX mv_quantized_inverted_token_docs_idx ON mv_quantized_inverted_token_docs
	USING turbohybrid (colbert multivector_cosine_turbohybrid_ops)
	WITH (multivector_graph = token_nodes);

	PERFORM set_config('enable_seqscan', 'off', true);
	PERFORM set_config('turbohybrid.multivector_candidate_source', 'quantized_inverted_experimental', true);
	PERFORM set_config('turbohybrid.multivector_plain_fallback', 'force', true);
	PERFORM id
	FROM mv_quantized_inverted_token_docs
	ORDER BY colbert <~> turbohybrid_query(
		multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
		dense_k => 2,
		final_k => 1
	)
	LIMIT 1;
	RAISE EXCEPTION 'quantized_inverted_experimental token-node scan unexpectedly executed';
EXCEPTION
	WHEN feature_not_supported THEN
		IF SQLERRM NOT LIKE 'quantized_inverted_experimental multivector candidate source requires multivector_graph = document_nodes%' THEN
			RAISE EXCEPTION 'unexpected quantized_inverted_experimental error: %', SQLERRM;
		END IF;
END
$$;

CREATE TEMP TABLE mv_doc_proxy_budget_docs (
	id int,
	colbert turbohybrid_multivector
);

INSERT INTO mv_doc_proxy_budget_docs
SELECT g,
	CASE
		WHEN g = 1 THEN
			turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector])
		WHEN g BETWEEN 2 AND 5 THEN
			turbohybrid_multivector(ARRAY['[0.9,0.1]'::vector, '[0.9,0.1]'::vector])
		ELSE
			turbohybrid_multivector(ARRAY['[-1,0]'::vector, '[-1,0]'::vector])
	END
FROM generate_series(1, 128) AS g;

CREATE INDEX mv_doc_proxy_budget_docs_idx ON mv_doc_proxy_budget_docs
USING turbohybrid (colbert multivector_cosine_turbohybrid_ops)
WITH (
	quantization_bits = 4,
	exact_storage = off,
	multivector_graph = document_nodes,
	multivector_doc_build_scorer = proxy,
	multivector_proxy_encoder = normalized_mean,
	graph_ef_search = 16,
	graph_oversampling = 4
);

DO $$
DECLARE
	stats jsonb;
	latency_target int;
	quality_target int;
	override_target int;
	capped_target int;
BEGIN
	PERFORM set_config('enable_seqscan', 'off', true);
	PERFORM set_config('turbohybrid.multivector_candidate_source', 'graph', true);
	PERFORM set_config('turbohybrid.multivector_plain_fallback', 'off', true);
	PERFORM set_config('turbohybrid.multivector_doc_candidate_k', '128', true);
	PERFORM set_config('turbohybrid.multivector_exact_rerank_k', '128', true);

	PERFORM set_config('turbohybrid.profile', 'latency', true);
	PERFORM id
	FROM mv_doc_proxy_budget_docs
	ORDER BY colbert <~> turbohybrid_query(
		multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
		final_k => 10
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();
	latency_target := (stats->>'multivector_proxy_candidate_target')::int;
	IF latency_target <> 100 OR
		(stats->>'multivector_candidate_path') <> 'proxy_graph' OR
		(stats->>'multivector_proxy_graph_searches')::int <> 1 OR
		(stats->>'multivector_exact_rerank_k_effective')::int >
			(stats->>'multivector_proxy_candidates_returned')::int THEN
		RAISE EXCEPTION 'unexpected latency proxy budget stats: %', stats;
	END IF;

	PERFORM set_config('turbohybrid.profile', 'quality', true);
	PERFORM id
	FROM mv_doc_proxy_budget_docs
	ORDER BY colbert <~> turbohybrid_query(
		multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
		final_k => 10
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();
	quality_target := (stats->>'multivector_proxy_candidate_target')::int;
	IF quality_target <= latency_target OR quality_target <> 128 THEN
		RAISE EXCEPTION 'quality profile did not widen proxy candidate target: latency %, quality %, stats %',
			latency_target, quality_target, stats;
	END IF;

	PERFORM set_config('turbohybrid.profile', 'latency', true);
	PERFORM set_config('turbohybrid.multivector_doc_candidate_k', '100', true);
	PERFORM id
	FROM mv_doc_proxy_budget_docs
	ORDER BY colbert <~> turbohybrid_query(
		multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
		dense_k => 20,
		final_k => 15
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();
	override_target := (stats->>'multivector_proxy_candidate_target')::int;
	IF override_target <> 60 THEN
		RAISE EXCEPTION 'dense_k/final_k proxy budget override failed: %', stats;
	END IF;

	PERFORM set_config('turbohybrid.multivector_doc_candidate_k', '12', true);
	PERFORM id
	FROM mv_doc_proxy_budget_docs
	ORDER BY colbert <~> turbohybrid_query(
		multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
		dense_k => 40,
		final_k => 10
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();
	capped_target := (stats->>'multivector_proxy_candidate_target')::int;
	IF capped_target <> 12 OR
		(stats->>'multivector_exact_rerank_k_effective')::int > capped_target THEN
		RAISE EXCEPTION 'proxy candidate cap failed: %', stats;
	END IF;
END
$$;

DO $$
DECLARE
	query_mv turbohybrid_multivector :=
		turbohybrid_multivector(ARRAY['[1,0]'::vector]);
	brute_ids int[];
	low_ids int[];
	high_ids int[];
	low_recall_hits int;
	high_recall_hits int;
	low_stats jsonb;
	high_stats jsonb;
BEGIN
	SELECT array_agg(id ORDER BY score DESC, id) INTO brute_ids
	FROM (
		SELECT id, turbohybrid_multivector_maxsim(query_mv, colbert) AS score
		FROM mv_doc_proxy_budget_docs
		ORDER BY score DESC, id
		LIMIT 10
	) AS brute;

	PERFORM set_config('enable_seqscan', 'off', true);
	PERFORM set_config('turbohybrid.profile', 'latency', true);
	PERFORM set_config('turbohybrid.multivector_candidate_source', 'graph', true);
	PERFORM set_config('turbohybrid.multivector_plain_fallback', 'off', true);
	PERFORM set_config('turbohybrid.multivector_doc_candidate_k', '1', true);
	PERFORM set_config('turbohybrid.multivector_exact_rerank_k', '1', true);

	SELECT array_agg(id ORDER BY distance, id) INTO low_ids
	FROM (
		SELECT id,
			colbert <~> turbohybrid_query(
				multivector_query => query_mv,
				dense_k => 1,
				final_k => 10
			) AS distance
		FROM mv_doc_proxy_budget_docs
		ORDER BY distance, id
		LIMIT 10
	) AS low_results;
	low_stats := turbohybrid_last_scan_stats();

	PERFORM set_config('turbohybrid.multivector_doc_candidate_k', '12', true);
	PERFORM set_config('turbohybrid.multivector_exact_rerank_k', '12', true);
	SELECT array_agg(id ORDER BY distance, id) INTO high_ids
	FROM (
		SELECT id,
			colbert <~> turbohybrid_query(
				multivector_query => query_mv,
				dense_k => 12,
				final_k => 10
			) AS distance
		FROM mv_doc_proxy_budget_docs
		ORDER BY distance, id
		LIMIT 10
	) AS high_results;
	high_stats := turbohybrid_last_scan_stats();

	SELECT count(*) INTO low_recall_hits
	FROM unnest(low_ids) AS hit(id)
	WHERE hit.id = ANY (brute_ids);
	SELECT count(*) INTO high_recall_hits
	FROM unnest(high_ids) AS hit(id)
	WHERE hit.id = ANY (brute_ids);

	IF high_recall_hits <= low_recall_hits OR
		(high_stats->>'multivector_proxy_candidate_target')::int <=
			(low_stats->>'multivector_proxy_candidate_target')::int THEN
		RAISE EXCEPTION 'proxy budget recall@10 did not improve: brute %, low % hits %, high % hits %, low stats %, high stats %',
			brute_ids, low_ids, low_recall_hits, high_ids, high_recall_hits,
			low_stats, high_stats;
	END IF;
END
$$;

DO $$
DECLARE
	before_stats jsonb;
	after_stats jsonb;
	diag jsonb;
BEGIN
	before_stats := turbohybrid_index_stats('mv_doc_proxy_budget_docs_idx'::regclass);
	diag := turbohybrid_multivector_proxy_diagnostics(
		'mv_doc_proxy_budget_docs_idx'::regclass,
		32,
		5
	);
	after_stats := turbohybrid_index_stats('mv_doc_proxy_budget_docs_idx'::regclass);

	IF diag->>'read_only' <> 'true' OR
		diag->>'sample_limited' <> 'true' OR
		(diag->>'sample_docs')::int <> 32 OR
		(diag->>'query_count')::int <> 5 OR
		diag->>'proxy_encoder' <> 'normalized_mean' OR
		NOT (diag ? 'recall_at_10_proxy_to_exact') OR
		NOT (diag ? 'avg_proxy_exact_rank_correlation') OR
		(diag->>'recommended_doc_candidate_k')::int < 10 THEN
		RAISE EXCEPTION 'unexpected proxy diagnostics JSON: %', diag;
	END IF;

	IF before_stats <> after_stats THEN
		RAISE EXCEPTION 'proxy diagnostics mutated index stats: before %, after %',
			before_stats, after_stats;
	END IF;
END
$$;

CREATE TEMP TABLE mv_proxy_diag_dense_docs (
	id int,
	embedding vector(2)
);
INSERT INTO mv_proxy_diag_dense_docs VALUES
	(1, '[1,0]'),
	(2, '[0,1]');
CREATE INDEX mv_proxy_diag_dense_docs_idx ON mv_proxy_diag_dense_docs
USING turbohybrid (embedding vector_cosine_turbohybrid_ops);

DO $$
DECLARE
	errmsg text;
BEGIN
	BEGIN
		PERFORM turbohybrid_multivector_proxy_diagnostics(
			'mv_proxy_diag_dense_docs_idx'::regclass,
			2,
			1
		);
		RAISE EXCEPTION 'expected proxy diagnostics to reject non-multivector index';
	EXCEPTION
		WHEN feature_not_supported THEN
			GET STACKED DIAGNOSTICS errmsg = MESSAGE_TEXT;
			IF errmsg NOT LIKE '%document-node multivector index%' THEN
				RAISE EXCEPTION 'unexpected proxy diagnostics error: %', errmsg;
			END IF;
	END;
END
$$;

DROP TABLE mv_proxy_diag_dense_docs;
DROP TABLE mv_doc_proxy_budget_docs;
DROP TABLE mv_centroid_lite_docs;

\set VERBOSITY terse

SELECT turbohybrid_multivector(ARRAY[]::vector[]);

SELECT turbohybrid_multivector(ARRAY['[1,0]'::vector, NULL::vector]);

SELECT turbohybrid_multivector(ARRAY['[1,0]'::vector, '[1,0,0]'::vector]);

SELECT turbohybrid_multivector_maxsim(
  turbohybrid_multivector(ARRAY['[1,0]'::vector]),
  turbohybrid_multivector(ARRAY['[1,0,0]'::vector])
);

DO $$
DECLARE
	literal text;
BEGIN
	FOREACH literal IN ARRAY ARRAY[
		'turbohybrid_multivector(dim=2,count=2,values=[[1,0]])',
		'turbohybrid_multivector(dim=2,count=1,values=[[1]])',
		'turbohybrid_multivector(dim=2,count=1,values=[1,0])',
		'turbohybrid_multivector(dim=2,count=1,values=[[NaN,0]])',
		'turbohybrid_multivector(dim=2,count=1,values=[[Inf,0]])',
		'turbohybrid_multivector(dim=2,count=1,values=[[1,0]]) trailing'
	]
	LOOP
		BEGIN
			EXECUTE format('SELECT %L::turbohybrid_multivector', literal);
			RAISE EXCEPTION 'expected malformed multivector literal to fail: %', literal;
		EXCEPTION WHEN invalid_text_representation OR data_exception THEN
			NULL;
		END;
	END LOOP;
END
$$;
