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

CREATE TABLE mv_many_moderate_docs (
  id text PRIMARY KEY,
  colbert turbohybrid_multivector
);

-- This models the DBpedia/ColBERT failure mode: one document is best by full
-- MaxSim because it covers all query tokens moderately, while each spike
-- document wins exactly one query-token candidate list.
INSERT INTO mv_many_moderate_docs VALUES
  ('spike_1', turbohybrid_multivector(ARRAY['[1,0,0,0]'::vector])),
  ('spike_2', turbohybrid_multivector(ARRAY['[0,1,0,0]'::vector])),
  ('spike_3', turbohybrid_multivector(ARRAY['[0,0,1,0]'::vector])),
  ('spike_4', turbohybrid_multivector(ARRAY['[0,0,0,1]'::vector])),
  ('good', turbohybrid_multivector(ARRAY[
    '[0.8,0.6,0,0]'::vector,
    '[0.6,0.8,0,0]'::vector,
    '[0,0,0.8,0.6]'::vector,
    '[0,0,0.6,0.8]'::vector
  ]));

CREATE INDEX mv_many_moderate_docs_idx ON mv_many_moderate_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops)
  WITH (graph_m = 4, graph_ef_construction = 8, graph_ef_search = 8);
ANALYZE mv_many_moderate_docs;

WITH q AS (
  SELECT turbohybrid_multivector(ARRAY[
    '[1,0,0,0]'::vector,
    '[0,1,0,0]'::vector,
    '[0,0,1,0]'::vector,
    '[0,0,0,1]'::vector
  ]) AS mv
)
SELECT id,
       round((-turbohybrid_multivector_maxsim_distance(q.mv, colbert))::numeric, 3) AS exact_maxsim
FROM mv_many_moderate_docs, q
ORDER BY turbohybrid_multivector_maxsim_distance(q.mv, colbert), id;

SET enable_seqscan = off;
SET turbohybrid.multivector_plain_fallback = off;
SET turbohybrid.multivector_subvector_k = 1;
SET turbohybrid.multivector_unique_docs_per_token = 1;
SET turbohybrid.multivector_max_raw_hits_per_token = 1;
SET turbohybrid.multivector_doc_candidate_k = 1;
SET turbohybrid.multivector_adaptive_widening = off;
SET turbohybrid.multivector_debug_admission = trace;
SET turbohybrid.multivector_debug_trace_limit = 10;

WITH q AS (
  SELECT turbohybrid_multivector(ARRAY[
    '[1,0,0,0]'::vector,
    '[0,1,0,0]'::vector,
    '[0,0,1,0]'::vector,
    '[0,0,0,1]'::vector
  ]) AS mv
)
SELECT id AS low_budget_index_top1
FROM mv_many_moderate_docs, q
ORDER BY colbert <~> turbohybrid_query(
  multivector_query => q.mv,
  dense_k => 1,
  final_k => 1
)
LIMIT 1;

DO $$
DECLARE
	q turbohybrid_multivector := turbohybrid_multivector(ARRAY[
		'[1,0,0,0]'::vector,
		'[0,1,0,0]'::vector,
		'[0,0,1,0]'::vector,
		'[0,0,0,1]'::vector
	]);
	exact_top1 text;
	index_top1 text;
	stats jsonb;
	trace jsonb;
	good_tid text;
	good_block int;
	good_offset int;
	good_admitted boolean;
BEGIN
	SELECT id INTO exact_top1
	FROM mv_many_moderate_docs
	ORDER BY turbohybrid_multivector_maxsim_distance(q, colbert), id
	LIMIT 1;
	IF exact_top1 <> 'good' THEN
		RAISE EXCEPTION 'expected exact MaxSim top-1 to be good, got %',
			exact_top1;
	END IF;

	SELECT id INTO index_top1
	FROM mv_many_moderate_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  dense_k => 1,
	  final_k => 1
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();
	trace := stats->'multivector_admission_trace';

	IF index_top1 IS NULL OR index_top1 = 'good' THEN
		RAISE EXCEPTION 'expected low-budget index path to miss good, got %',
			index_top1;
	END IF;
	IF (stats->>'multivector_admission_candidates_before_rerank')::int <> 4 OR
		(stats->>'multivector_admission_candidates_after_truncation')::int <> 1 OR
		stats->>'multivector_admission_truncated_by_doc_candidate_k' <> 'true' OR
		stats->>'multivector_plain_fallback_used' <> 'false' OR
		jsonb_array_length(trace) <> 4 THEN
		RAISE EXCEPTION 'expected low-budget admission truncation trace, got %',
			stats;
	END IF;

	SELECT ctid::text INTO good_tid
	FROM mv_many_moderate_docs
	WHERE id = 'good';
	good_block := split_part(trim(both '()' from good_tid), ',', 1)::int;
	good_offset := split_part(trim(both '()' from good_tid), ',', 2)::int;

	SELECT EXISTS (
		SELECT 1
		FROM jsonb_array_elements(trace) AS e(entry)
		WHERE (entry->>'heap_block')::int = good_block
		  AND (entry->>'heap_offset')::int = good_offset
	) INTO good_admitted;
	IF good_admitted THEN
		RAISE EXCEPTION 'expected good document to be absent from admission trace, got %',
			stats;
	END IF;
END
$$;

SELECT (turbohybrid_last_scan_stats()->>'multivector_admission_candidates_before_rerank')::int
  AS low_budget_candidates_before_rerank,
       (turbohybrid_last_scan_stats()->>'multivector_admission_candidates_after_truncation')::int
  AS low_budget_candidates_after_truncation,
       turbohybrid_last_scan_stats()->>'multivector_admission_truncated_by_doc_candidate_k'
  AS truncated_by_doc_candidate_k;

SET turbohybrid.multivector_candidate_source = exact_token_scan;

DO $$
DECLARE
	q turbohybrid_multivector := turbohybrid_multivector(ARRAY[
		'[1,0,0,0]'::vector,
		'[0,1,0,0]'::vector,
		'[0,0,1,0]'::vector,
		'[0,0,0,1]'::vector
	]);
	graph_candidates_before int :=
		(turbohybrid_last_scan_stats()->>'multivector_admission_candidates_before_rerank')::int;
	exact_source_top1 text;
	stats jsonb;
BEGIN
	SELECT id INTO exact_source_top1
	FROM mv_many_moderate_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  dense_k => 1,
	  final_k => 1
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();

	IF stats->>'multivector_candidate_source' <> 'exact_token_scan' OR
		stats->>'multivector_exact_token_scan_enabled' <> 'true' THEN
		RAISE EXCEPTION 'expected exact_token_scan stats, got %', stats;
	END IF;
	IF (stats->>'multivector_exact_token_scan_nodes_scored')::int <> 32 THEN
		RAISE EXCEPTION 'expected exact token scan to score every query-token/node pair, got %',
			stats;
	END IF;
	IF (stats->>'multivector_admission_candidates_before_rerank')::int <
		graph_candidates_before THEN
		RAISE EXCEPTION 'exact token scan admitted fewer docs than graph under same caps: graph %, exact %, stats %',
			graph_candidates_before,
			stats->>'multivector_admission_candidates_before_rerank',
			stats;
	END IF;
	IF exact_source_top1 IS NULL THEN
		RAISE EXCEPTION 'expected exact token scan to return a document';
	END IF;
END
$$;

SELECT turbohybrid_last_scan_stats()->>'multivector_candidate_source'
  AS exact_token_candidate_source,
       (turbohybrid_last_scan_stats()->>'multivector_exact_token_scan_nodes_scored')::int
  AS exact_token_nodes_scored;

CREATE TABLE mv_reservoir_docs (
  id text PRIMARY KEY,
  colbert turbohybrid_multivector
);

-- This isolates Prompt 6's reservoir mitigation from graph ANN recall: the
-- exact-token source admits the globally good document through one query token,
-- but score-only doc_candidate_k truncation drops it before exact rerank.
INSERT INTO mv_reservoir_docs VALUES
  ('strong_1', turbohybrid_multivector(ARRAY['[1,0,0]'::vector])),
  ('strong_2a', turbohybrid_multivector(ARRAY['[0,1,0]'::vector])),
  ('strong_2b', turbohybrid_multivector(ARRAY['[0,1,0]'::vector])),
  ('strong_3a', turbohybrid_multivector(ARRAY['[0,0,1]'::vector])),
  ('strong_3b', turbohybrid_multivector(ARRAY['[0,0,1]'::vector])),
  ('good', turbohybrid_multivector(ARRAY[
    '[1,0,0]'::vector,
    '[0,1,0]'::vector,
    '[0,0,1]'::vector
  ]));

CREATE INDEX mv_reservoir_docs_idx ON mv_reservoir_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops)
  WITH (graph_m = 4, graph_ef_construction = 8, graph_ef_search = 8);
ANALYZE mv_reservoir_docs;

SET turbohybrid.multivector_candidate_source = exact_token_scan;
SET turbohybrid.multivector_subvector_k = 2;
SET turbohybrid.multivector_unique_docs_per_token = 2;
SET turbohybrid.multivector_max_raw_hits_per_token = 2;
SET turbohybrid.multivector_doc_candidate_k = 3;
SET turbohybrid.multivector_exact_rerank_k = 3;
SET turbohybrid.multivector_coverage_reservoir_k = 0;
SET turbohybrid.multivector_per_token_doc_reservoir_k = 2;
SET turbohybrid.multivector_candidate_reservoirs = off;

DO $$
DECLARE
	q turbohybrid_multivector := turbohybrid_multivector(ARRAY[
		'[1,0,0]'::vector,
		'[0,1,0]'::vector,
		'[0,0,1]'::vector
	]);
	exact_top1 text;
	score_only_top1 text;
	stats jsonb;
BEGIN
	SELECT id INTO exact_top1
	FROM mv_reservoir_docs
	ORDER BY turbohybrid_multivector_maxsim_distance(q, colbert), id
	LIMIT 1;
	IF exact_top1 <> 'good' THEN
		RAISE EXCEPTION 'expected reservoir exact top-1 to be good, got %',
			exact_top1;
	END IF;

	SELECT id INTO score_only_top1
	FROM mv_reservoir_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  dense_k => 3,
	  final_k => 1
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();
	IF score_only_top1 = 'good' OR
		stats->>'multivector_reservoirs_enabled' <> 'false' THEN
		RAISE EXCEPTION 'expected score-only truncation to miss good, top %, stats %',
			score_only_top1, stats;
	END IF;
END
$$;

SET turbohybrid.multivector_candidate_reservoirs = balanced;

DO $$
DECLARE
	q turbohybrid_multivector := turbohybrid_multivector(ARRAY[
		'[1,0,0]'::vector,
		'[0,1,0]'::vector,
		'[0,0,1]'::vector
	]);
	reservoir_top1 text;
	stats jsonb;
BEGIN
	SELECT id INTO reservoir_top1
	FROM mv_reservoir_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  dense_k => 3,
	  final_k => 1
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();
	IF reservoir_top1 <> 'good' OR
		stats->>'multivector_reservoirs_enabled' <> 'true' OR
		(stats->>'multivector_reservoir_union_docs')::int <> 3 OR
		(stats->>'multivector_reservoir_per_token_docs')::int < 1 OR
		(stats->>'multivector_exact_rerank_docs')::int <> 3 THEN
		RAISE EXCEPTION 'expected reservoir path to retain and rerank good, top %, stats %',
			reservoir_top1, stats;
	END IF;
END
$$;

SELECT turbohybrid_last_scan_stats()->>'multivector_reservoirs_enabled'
  AS reservoir_enabled,
       (turbohybrid_last_scan_stats()->>'multivector_reservoir_union_docs')::int
  AS reservoir_union_docs,
       (turbohybrid_last_scan_stats()->>'multivector_reservoir_per_token_docs')::int
  AS reservoir_per_token_docs;

DROP TABLE mv_reservoir_docs;
RESET turbohybrid.multivector_candidate_source;
RESET turbohybrid.multivector_candidate_reservoirs;
RESET turbohybrid.multivector_coverage_reservoir_k;
RESET turbohybrid.multivector_per_token_doc_reservoir_k;

SET turbohybrid.multivector_plain_fallback = force;

DO $$
DECLARE
	q turbohybrid_multivector := turbohybrid_multivector(ARRAY[
		'[1,0,0,0]'::vector,
		'[0,1,0,0]'::vector,
		'[0,0,1,0]'::vector,
		'[0,0,0,1]'::vector
	]);
	exact_top1 text;
	fallback_top1 text;
	stats jsonb;
BEGIN
	SELECT id INTO exact_top1
	FROM mv_many_moderate_docs
	ORDER BY turbohybrid_multivector_maxsim_distance(q, colbert), id
	LIMIT 1;

	SELECT id INTO fallback_top1
	FROM mv_many_moderate_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  dense_k => 1,
	  final_k => 1
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();

	IF fallback_top1 <> exact_top1 OR fallback_top1 <> 'good' THEN
		RAISE EXCEPTION 'expected force fallback top1 % to match exact top1 %, stats %',
			fallback_top1, exact_top1, stats;
	END IF;
	IF stats->>'multivector_candidate_source' <> 'plain_fallback' OR
		stats->>'multivector_plain_fallback_used' <> 'true' OR
		stats->>'multivector_plain_fallback_reason' <> 'force' OR
		(stats->>'multivector_plain_fallback_docs_scored')::int <> 5 OR
		(stats->>'multivector_plain_fallback_pairs')::int <> 32 THEN
		RAISE EXCEPTION 'expected force fallback stats, got %', stats;
	END IF;
END
$$;

SELECT turbohybrid_last_scan_stats()->>'multivector_plain_fallback_reason'
  AS force_fallback_reason,
       (turbohybrid_last_scan_stats()->>'multivector_plain_fallback_docs_scored')::int
  AS force_fallback_docs_scored,
       (turbohybrid_last_scan_stats()->>'multivector_plain_fallback_pairs')::int
  AS force_fallback_pairs;

SET turbohybrid.multivector_plain_fallback = auto;

DO $$
DECLARE
	q turbohybrid_multivector := turbohybrid_multivector(ARRAY[
		'[1,0,0,0]'::vector,
		'[0,1,0,0]'::vector,
		'[0,0,1,0]'::vector,
		'[0,0,0,1]'::vector
	]);
	auto_top1 text;
	stats jsonb;
BEGIN
	SELECT id INTO auto_top1
	FROM mv_many_moderate_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  dense_k => 1,
	  final_k => 1
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();

	IF auto_top1 <> 'good' OR
		stats->>'multivector_plain_fallback_used' <> 'true' OR
		stats->>'multivector_plain_fallback_reason' <> 'small_estimated_docs' THEN
		RAISE EXCEPTION 'expected auto small-table fallback, top1 %, stats %',
			auto_top1, stats;
	END IF;
END
$$;

SELECT turbohybrid_last_scan_stats()->>'multivector_plain_fallback_reason'
  AS auto_fallback_reason;

CREATE TABLE mv_plain_fallback_mvcc (
  id text PRIMARY KEY,
  colbert turbohybrid_multivector
);
INSERT INTO mv_plain_fallback_mvcc VALUES
  ('deleted_best', turbohybrid_multivector(ARRAY['[1,0]'::vector])),
  ('keep', turbohybrid_multivector(ARRAY['[0,1]'::vector]));
CREATE INDEX mv_plain_fallback_mvcc_idx ON mv_plain_fallback_mvcc USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops);
ANALYZE mv_plain_fallback_mvcc;
DELETE FROM mv_plain_fallback_mvcc WHERE id = 'deleted_best';
UPDATE mv_plain_fallback_mvcc
SET colbert = turbohybrid_multivector(ARRAY['[1,0]'::vector])
WHERE id = 'keep';

SET turbohybrid.multivector_plain_fallback = force;

DO $$
DECLARE
	mvcc_top1 text;
	stats jsonb;
BEGIN
	SELECT id INTO mvcc_top1
	FROM mv_plain_fallback_mvcc
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => turbohybrid_multivector(ARRAY['[1,0]'::vector]),
	  dense_k => 1,
	  final_k => 1
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();

	IF mvcc_top1 <> 'keep' OR
		stats->>'multivector_plain_fallback_used' <> 'true' OR
		(stats->>'multivector_plain_fallback_docs_scored')::int <> 1 OR
		(stats->>'multivector_plain_fallback_pairs')::int <> 1 THEN
		RAISE EXCEPTION 'expected fallback heap scan to respect MVCC, top1 %, stats %',
			mvcc_top1, stats;
	END IF;
END
$$;

SELECT turbohybrid_last_scan_stats()->>'multivector_plain_fallback_reason'
  AS mvcc_fallback_reason,
       (turbohybrid_last_scan_stats()->>'multivector_plain_fallback_docs_scored')::int
  AS mvcc_docs_scored;

DROP TABLE mv_plain_fallback_mvcc;

SET turbohybrid.multivector_candidate_source = exact_token_scan;
SET turbohybrid.multivector_debug_admission = summary;
SET turbohybrid.multivector_subvector_k = 4;
SET turbohybrid.multivector_unique_docs_per_token = 4;
SET turbohybrid.multivector_max_raw_hits_per_token = 4;
SET turbohybrid.multivector_doc_candidate_k = 4;
SET turbohybrid.multivector_exact_rerank_k = 4;
SET turbohybrid.multivector_candidate_reservoirs = off;
SET turbohybrid.multivector_plain_fallback = off;
SET turbohybrid.multivector_debug_skip_query_tokens = '';

DO $$
DECLARE
	q turbohybrid_multivector := turbohybrid_multivector(ARRAY[
		'[1,0,0,0]'::vector,
		'[0,1,0,0]'::vector,
		'[0,0,1,0]'::vector,
		'[0,0,0,1]'::vector
	]);
	stats jsonb;
	debug_top1 text;
BEGIN
	SELECT id INTO debug_top1
	FROM mv_many_moderate_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  dense_k => 4,
	  final_k => 4
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();

	IF stats->>'multivector_query_token_stats_available' <> 'true' OR
		jsonb_array_length(stats->'multivector_query_token_stats') <> 4 THEN
		RAISE EXCEPTION 'expected four query-token diagnostic entries, got %',
			stats;
	END IF;
	IF NOT EXISTS (
		SELECT 1
		FROM jsonb_array_elements(stats->'multivector_query_token_stats') AS e(entry)
		WHERE (entry->>'query_token_ordinal')::int = 0
		  AND (entry->>'raw_hits')::int >= 1
		  AND (entry->>'unique_docs')::int >= 1
		  AND entry->>'top_hit_similarity' IS NOT NULL
		  AND (entry->>'candidate_docs_retained_from_token')::int >= 1
		  AND entry->>'skipped' = 'false'
	) THEN
		RAISE EXCEPTION 'expected token 0 contribution diagnostics, got %',
			stats;
	END IF;

	PERFORM set_config('turbohybrid.multivector_debug_skip_query_tokens',
					   '0, 2', false);
	SELECT id INTO debug_top1
	FROM mv_many_moderate_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  dense_k => 4,
	  final_k => 4
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();

	IF (stats->>'multivector_subvector_searches')::int <> 2 OR
		(stats->>'multivector_exact_rerank_pairs')::int <= 0 THEN
		RAISE EXCEPTION 'expected skip to affect candidate generation only, got %',
			stats;
	END IF;
	IF NOT EXISTS (
		SELECT 1
		FROM jsonb_array_elements(stats->'multivector_query_token_stats') AS e(entry)
		WHERE (entry->>'query_token_ordinal')::int = 0
		  AND entry->>'skipped' = 'true'
		  AND (entry->>'raw_hits')::int = 0
		  AND entry->>'top_hit_similarity' IS NULL
	) OR NOT EXISTS (
		SELECT 1
		FROM jsonb_array_elements(stats->'multivector_query_token_stats') AS e(entry)
		WHERE (entry->>'query_token_ordinal')::int = 1
		  AND entry->>'skipped' = 'false'
		  AND (entry->>'raw_hits')::int >= 1
	) THEN
		RAISE EXCEPTION 'expected skipped and unskipped token diagnostics, got %',
			stats;
	END IF;

	BEGIN
		PERFORM set_config('turbohybrid.multivector_debug_skip_query_tokens',
						   'bad', false);
		PERFORM id
		FROM mv_many_moderate_docs
		ORDER BY colbert <~> turbohybrid_query(
		  multivector_query => q,
		  dense_k => 4,
		  final_k => 4
		)
		LIMIT 1;
		RAISE EXCEPTION 'expected invalid skip token list error';
	EXCEPTION
		WHEN invalid_parameter_value THEN
			NULL;
	END;
	PERFORM set_config('turbohybrid.multivector_debug_skip_query_tokens',
					   '', false);
END
$$;

SELECT jsonb_array_length(turbohybrid_last_scan_stats()->'multivector_query_token_stats')
  AS query_token_stats_entries,
       turbohybrid_last_scan_stats()->>'multivector_query_token_stats_available'
  AS query_token_stats_available;

DROP INDEX mv_many_moderate_docs_idx;
CREATE INDEX mv_many_moderate_docs_doc_nodes_idx ON mv_many_moderate_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops)
  WITH (multivector_graph = document_nodes,
        graph_m = 4,
        graph_ef_construction = 8,
        graph_ef_search = 8);

RESET turbohybrid.multivector_candidate_source;
SET turbohybrid.multivector_candidate_reservoirs = off;
SET turbohybrid.multivector_plain_fallback = off;
SET turbohybrid.multivector_doc_candidate_k = 1;
SET turbohybrid.multivector_exact_rerank_k = 1;
SET turbohybrid.multivector_debug_admission = summary;
SET turbohybrid.multivector_debug_skip_query_tokens = '';

DO $$
DECLARE
	q turbohybrid_multivector := turbohybrid_multivector(ARRAY[
		'[1,0,0,0]'::vector,
		'[0,1,0,0]'::vector,
		'[0,0,1,0]'::vector,
		'[0,0,0,1]'::vector
	]);
	doc_node_top1 text;
	stats jsonb;
BEGIN
	SELECT id INTO doc_node_top1
	FROM mv_many_moderate_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  dense_k => 1,
	  final_k => 1
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();

	IF doc_node_top1 <> 'good' OR
		stats->>'multivector_graph_mode' <> 'document_nodes' OR
		stats->>'multivector_candidate_source' <> 'graph' OR
		(stats->>'multivector_doc_graph_nodes')::int <> 5 OR
		(stats->>'multivector_doc_graph_candidates')::int <> 1 OR
		(stats->>'multivector_doc_graph_exact_rerank_docs')::int <> 1 OR
		(stats->>'multivector_doc_graph_quantized_scores')::int <> 0 OR
		stats->>'multivector_doc_graph_storage_kind' <> 'f32' OR
		stats->>'multivector_doc_graph_rescore_source' <> 'sidecar' OR
		stats->>'multivector_exact_rerank_source' <> 'sidecar' OR
		(stats->>'multivector_doc_graph_heap_fetches')::int <> 0 OR
		(stats->>'multivector_exact_rerank_heap_fetches')::int <> 0 OR
		(stats->>'multivector_exact_rerank_sidecar_reads')::int <> 1 OR
		stats->>'multivector_doc_graph_warning' <> 'document_node_f32_sidecar_exact_scan' THEN
		RAISE EXCEPTION 'expected document-node graph to admit many-moderate good doc, top %, stats %',
			doc_node_top1, stats;
	END IF;
END
$$;

SELECT turbohybrid_last_scan_stats()->>'multivector_graph_mode'
  AS document_node_many_moderate_mode,
       turbohybrid_last_scan_stats()->>'multivector_doc_graph_warning'
  AS document_node_many_moderate_warning,
       (turbohybrid_last_scan_stats()->>'multivector_doc_graph_nodes')::int
  AS document_node_many_moderate_nodes;

SET turbohybrid.native_cache_scope = off;
SET turbohybrid.multivector_doc_storage_cache = 'paged';

DO $$
DECLARE
	q turbohybrid_multivector := turbohybrid_multivector(ARRAY[
		'[1,0,0,0]'::vector,
		'[0,1,0,0]'::vector,
		'[0,0,1,0]'::vector,
		'[0,0,0,1]'::vector
	]);
	doc_node_top1 text;
	stats jsonb;
BEGIN
	SELECT id INTO doc_node_top1
	FROM mv_many_moderate_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  dense_k => 1,
	  final_k => 1
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();

	IF doc_node_top1 <> 'good' OR
		stats->>'multivector_doc_sidecar_cache_mode' <> 'paged' OR
		COALESCE((stats->>'multivector_doc_sidecar_pages_read')::int, 0) <= 0 OR
		COALESCE((stats->>'multivector_doc_sidecar_cache_misses')::int, 0) <= 0 OR
		COALESCE((stats->>'multivector_doc_sidecar_vectors_loaded')::int, 0) <= 0 OR
		COALESCE((stats->>'multivector_doc_sidecar_bytes_touched')::int, 0) <= 0 THEN
		RAISE EXCEPTION 'expected paged document-node sidecar stats with correct top doc, top %, stats %',
			doc_node_top1, stats;
	END IF;
END
$$;

SELECT turbohybrid_last_scan_stats()->>'multivector_doc_sidecar_cache_mode'
  AS document_node_sidecar_cache_mode,
       (turbohybrid_last_scan_stats()->>'multivector_doc_sidecar_pages_read')::int > 0
  AS document_node_sidecar_pages_read,
       (turbohybrid_last_scan_stats()->>'multivector_doc_sidecar_vectors_loaded')::int > 0
  AS document_node_sidecar_vectors_loaded,
       (turbohybrid_last_scan_stats()->>'multivector_doc_sidecar_bytes_touched')::int > 0
  AS document_node_sidecar_bytes_touched;

RESET turbohybrid.native_cache_scope;
RESET turbohybrid.multivector_doc_storage_cache;

SET turbohybrid.multivector_doc_graph_search_ef = 4;
SET turbohybrid.multivector_doc_graph_oversampling = 2;
SET turbohybrid.multivector_doc_graph_rescore_k = 2;
SET turbohybrid.multivector_doc_storage = 'f16';

DO $$
DECLARE
	q turbohybrid_multivector := turbohybrid_multivector(ARRAY[
		'[1,0,0,0]'::vector,
		'[0,1,0,0]'::vector,
		'[0,0,1,0]'::vector,
		'[0,0,0,1]'::vector
	]);
	doc_node_top1 text;
	stats jsonb;
BEGIN
	SELECT id INTO doc_node_top1
	FROM mv_many_moderate_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  dense_k => 1,
	  final_k => 1
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();

	IF doc_node_top1 <> 'good' OR
		stats->>'multivector_candidate_path' <> 'proxy_graph' OR
		stats->>'multivector_doc_graph_warning' <>
			'document_node_proxy_vector_graph_traversal' OR
		(stats->>'multivector_proxy_graph_searches')::int <> 1 OR
		(stats->>'multivector_subvector_searches')::int <> 0 OR
		(stats->>'multivector_doc_graph_search_ef')::int <> 4 OR
		(stats->>'multivector_doc_graph_oversampling')::int <> 2 OR
		(stats->>'multivector_doc_graph_rescore_k')::int <> 1 OR
		(stats->>'multivector_doc_graph_candidates')::int <> 4 OR
		(stats->>'multivector_doc_graph_exact_rerank_docs')::int <> 1 OR
		(stats->>'multivector_doc_graph_quantized_scores')::int <> 0 OR
		stats->>'multivector_doc_graph_storage_kind' <> 'f16' OR
		stats->>'multivector_doc_graph_rescore_source' <> 'sidecar' OR
		stats->>'multivector_exact_rerank_source' <> 'sidecar' OR
		(stats->>'multivector_doc_graph_heap_fetches')::int <> 0 OR
		(stats->>'multivector_exact_rerank_heap_fetches')::int <> 0 OR
		(stats->>'multivector_exact_rerank_sidecar_reads')::int <> 1 OR
		stats->>'multivector_accumulator_kind' <> 'doc_proxy_graph' THEN
		RAISE EXCEPTION 'expected document-node graph f16 knobs to widen candidates independently of final_k, top %, stats %',
			doc_node_top1, stats;
	END IF;
END
$$;

SELECT (turbohybrid_last_scan_stats()->>'multivector_doc_graph_search_ef')::int
  AS document_node_knob_search_ef,
       (turbohybrid_last_scan_stats()->>'multivector_doc_graph_oversampling')::int
  AS document_node_knob_oversampling,
       (turbohybrid_last_scan_stats()->>'multivector_doc_graph_rescore_k')::int
  AS document_node_knob_rescore_k,
       (turbohybrid_last_scan_stats()->>'multivector_doc_graph_candidates')::int
  AS document_node_knob_candidates,
       (turbohybrid_last_scan_stats()->>'multivector_doc_graph_exact_rerank_docs')::int
  AS document_node_knob_exact_rerank_docs,
       turbohybrid_last_scan_stats()->>'multivector_doc_graph_storage_kind'
  AS document_node_knob_storage_kind,
       turbohybrid_last_scan_stats()->>'multivector_doc_graph_rescore_source'
  AS document_node_knob_rescore_source;

SET turbohybrid.multivector_doc_candidate_k = 10;
SET turbohybrid.multivector_exact_rerank_k = 10;
SET turbohybrid.multivector_doc_graph_search_ef = 2;
SET turbohybrid.multivector_doc_graph_rescore_k = 10;

DO $$
DECLARE
	q turbohybrid_multivector := turbohybrid_multivector(ARRAY[
		'[1,0,0,0]'::vector,
		'[0,1,0,0]'::vector,
		'[0,0,1,0]'::vector,
		'[0,0,0,1]'::vector
	]);
	stats jsonb;
BEGIN
	PERFORM id
	FROM mv_many_moderate_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  dense_k => 1,
	  final_k => 1
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();

	IF (stats->>'multivector_doc_graph_search_ef')::int <> 2 OR
		stats->>'multivector_candidate_path' <> 'proxy_graph' OR
		(stats->>'multivector_proxy_graph_searches')::int <> 1 OR
		(stats->>'multivector_subvector_searches')::int <> 0 OR
		(stats->>'multivector_doc_graph_candidates')::int <> 5 OR
		(stats->>'multivector_doc_graph_exact_rerank_docs')::int <> 5 THEN
		RAISE EXCEPTION 'expected explicit document-node EF with proxy candidate-band fill, stats %',
			stats;
	END IF;
END
$$;

SELECT (turbohybrid_last_scan_stats()->>'multivector_doc_graph_search_ef')::int
  AS document_node_explicit_ef,
       (turbohybrid_last_scan_stats()->>'multivector_doc_graph_candidates')::int = 5
  AS document_node_explicit_ef_fills_candidate_band,
       (turbohybrid_last_scan_stats()->>'multivector_doc_graph_exact_rerank_docs')::int = 5
  AS document_node_explicit_ef_reranks_candidate_band;

SET turbohybrid.multivector_doc_graph_search_ef = 0;

DO $$
DECLARE
	q turbohybrid_multivector := turbohybrid_multivector(ARRAY[
		'[1,0,0,0]'::vector,
		'[0,1,0,0]'::vector,
		'[0,0,1,0]'::vector,
		'[0,0,0,1]'::vector
	]);
	stats jsonb;
BEGIN
	PERFORM id
	FROM mv_many_moderate_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  dense_k => 1,
	  final_k => 1
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();

		IF (stats->>'multivector_doc_graph_search_ef')::int <> 5 OR
			(stats->>'multivector_doc_graph_candidates')::int > 5 OR
			(stats->>'multivector_doc_graph_exact_rerank_docs')::int > 5 THEN
			RAISE EXCEPTION 'expected default document-node EF to use graph_ef_search independently of candidate budget, stats %',
				stats;
		END IF;
END
$$;

SELECT (turbohybrid_last_scan_stats()->>'multivector_doc_graph_search_ef')::int
  AS document_node_default_ef,
       (turbohybrid_last_scan_stats()->>'multivector_doc_graph_candidates')::int <= 5
  AS document_node_default_ef_caps_candidates,
       (turbohybrid_last_scan_stats()->>'multivector_doc_graph_exact_rerank_docs')::int <= 5
  AS document_node_default_ef_caps_rerank;

SET turbohybrid.multivector_doc_candidate_k = 1;
SET turbohybrid.multivector_exact_rerank_k = 1;
SET turbohybrid.multivector_doc_graph_search_ef = 4;
SET turbohybrid.multivector_doc_graph_rescore_k = 2;
SET turbohybrid.multivector_doc_storage = 'sq8';

DO $$
DECLARE
	q turbohybrid_multivector := turbohybrid_multivector(ARRAY[
		'[1,0,0,0]'::vector,
		'[0,1,0,0]'::vector,
		'[0,0,1,0]'::vector,
		'[0,0,0,1]'::vector
	]);
	doc_node_top1 text;
	stats jsonb;
BEGIN
	SELECT id INTO doc_node_top1
	FROM mv_many_moderate_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  dense_k => 1,
	  final_k => 1
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();

	IF doc_node_top1 <> 'good' OR
		stats->>'multivector_candidate_path' <> 'proxy_graph' OR
		stats->>'multivector_doc_graph_warning' <>
			'document_node_proxy_vector_graph_traversal' OR
		(stats->>'multivector_proxy_graph_searches')::int <> 1 OR
		(stats->>'multivector_subvector_searches')::int <> 0 OR
		(stats->>'multivector_doc_graph_quantized_scores')::int <> 0 OR
		stats->>'multivector_doc_graph_storage_kind' <> 'sq8' OR
		stats->>'multivector_doc_graph_rescore_source' <> 'sidecar' OR
		stats->>'multivector_exact_rerank_source' <> 'sidecar' OR
		(stats->>'multivector_doc_graph_heap_fetches')::int <> 0 OR
		(stats->>'multivector_exact_rerank_heap_fetches')::int <> 0 OR
		(stats->>'multivector_exact_rerank_sidecar_reads')::int < 1 OR
		stats->>'multivector_accumulator_kind' <> 'doc_proxy_graph' THEN
		RAISE EXCEPTION 'expected document-node graph sq8 compact scoring to admit many-moderate good doc, top %, stats %',
			doc_node_top1, stats;
	END IF;
END
$$;

SELECT turbohybrid_last_scan_stats()->>'multivector_doc_graph_storage_kind'
  AS document_node_sq8_storage_kind,
       turbohybrid_last_scan_stats()->>'multivector_doc_graph_rescore_source'
  AS document_node_sq8_rescore_source,
       (turbohybrid_last_scan_stats()->>'multivector_doc_graph_quantized_scores')::int > 0
  AS document_node_sq8_quantized_scores;

SET turbohybrid.multivector_doc_storage = 'f32';
SET turbohybrid.multivector_candidate_source = document_nodes;

DO $$
DECLARE
	q turbohybrid_multivector := turbohybrid_multivector(ARRAY[
		'[1,0,0,0]'::vector,
		'[0,1,0,0]'::vector,
		'[0,0,1,0]'::vector,
		'[0,0,0,1]'::vector
	]);
	doc_node_top1 text;
	stats jsonb;
BEGIN
	SELECT id INTO doc_node_top1
	FROM mv_many_moderate_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  dense_k => 1,
	  final_k => 1
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();

	IF doc_node_top1 <> 'good' OR
		stats->>'multivector_candidate_source' <> 'document_nodes' OR
		stats->>'multivector_graph_mode' <> 'document_nodes' OR
		stats->>'multivector_doc_graph_warning' <>
			'document_node_f32_sidecar_graph_traversal' THEN
		RAISE EXCEPTION 'expected explicit document_nodes candidate source to use document-node graph, top %, stats %',
			doc_node_top1, stats;
	END IF;
END
$$;

SELECT turbohybrid_last_scan_stats()->>'multivector_candidate_source'
  AS document_node_source,
       turbohybrid_last_scan_stats()->>'multivector_graph_mode'
  AS document_node_graph_mode;

SET turbohybrid.multivector_candidate_source = proxy_vector;

DO $$
DECLARE
	q turbohybrid_multivector := turbohybrid_multivector(ARRAY[
		'[1,0,0,0]'::vector,
		'[0,1,0,0]'::vector,
		'[0,0,1,0]'::vector,
		'[0,0,0,1]'::vector
	]);
	doc_node_top1 text;
	stats jsonb;
BEGIN
	SELECT id INTO doc_node_top1
	FROM mv_many_moderate_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  dense_k => 1,
	  final_k => 1
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();

	IF doc_node_top1 <> 'good' OR
		stats->>'multivector_candidate_source' <> 'proxy_vector' OR
		stats->>'multivector_doc_graph_warning' <>
			'document_node_proxy_vector_graph_traversal' OR
		stats->>'multivector_accumulator_kind' <>
			'doc_proxy_graph' OR
		stats->>'proxy_encoder_kind' <> 'normalized_mean' OR
		(stats->>'multivector_doc_graph_candidates')::int <> 4 OR
		(stats->>'multivector_doc_graph_exact_rerank_docs')::int <> 1 OR
		(stats->>'proxy_candidates')::int <> 4 OR
		(stats->>'proxy_exact_rerank_docs')::int <> 1 OR
		NOT (stats ? 'proxy_top1_admission') OR
		(stats->>'multivector_doc_graph_quantized_scores')::int <> 0 OR
		stats->>'multivector_doc_graph_rescore_source' <> 'sidecar' OR
		stats->>'multivector_exact_rerank_source' <> 'sidecar' OR
		(stats->>'multivector_doc_graph_heap_fetches')::int <> 0 OR
		(stats->>'multivector_exact_rerank_heap_fetches')::int <> 0 OR
		(stats->>'multivector_exact_rerank_sidecar_reads')::int <> 1 THEN
		RAISE EXCEPTION 'expected proxy-vector document-node graph admission with exact MaxSim rerank, top %, stats %',
			doc_node_top1, stats;
	END IF;
END
$$;

SELECT turbohybrid_last_scan_stats()->>'multivector_candidate_source'
  AS document_node_proxy_source,
       turbohybrid_last_scan_stats()->>'multivector_doc_graph_warning'
  AS document_node_proxy_warning,
       turbohybrid_last_scan_stats()->>'proxy_encoder_kind'
  AS document_node_proxy_encoder,
       (turbohybrid_last_scan_stats()->>'proxy_candidates')::int
  AS document_node_proxy_candidates,
       (turbohybrid_last_scan_stats()->>'multivector_doc_graph_exact_rerank_docs')::int
  AS document_node_proxy_exact_rerank_docs;

CREATE TABLE mv_proxy_encoder_docs (
	id text PRIMARY KEY,
	colbert turbohybrid_multivector
);

INSERT INTO mv_proxy_encoder_docs VALUES
('alpha', turbohybrid_multivector(ARRAY[
	'[1,0,0,0]'::vector,
	'[0,1,0,0]'::vector
])),
('beta', turbohybrid_multivector(ARRAY[
	'[0,0,1,0]'::vector,
	'[0,0,0,1]'::vector
]));

CREATE INDEX mv_proxy_encoder_docs_max_idx ON mv_proxy_encoder_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops)
  WITH (multivector_graph = document_nodes, multivector_proxy_encoder = max_pool);

DO $$
DECLARE
	q turbohybrid_multivector := turbohybrid_multivector(ARRAY[
		'[1,0,0,0]'::vector,
		'[0,1,0,0]'::vector
	]);
	stats jsonb;
BEGIN
	PERFORM id
	FROM mv_proxy_encoder_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  dense_k => 2,
	  final_k => 1
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();

	IF stats->>'proxy_encoder_kind' <> 'max_pool' OR
		(stats->>'proxy_candidates')::int <= 0 OR
		(stats->>'proxy_exact_rerank_docs')::int <= 0 THEN
		RAISE EXCEPTION 'expected max_pool proxy stats, got %', stats;
	END IF;
END
$$;

SELECT turbohybrid_last_scan_stats()->>'proxy_encoder_kind'
  AS document_node_proxy_max_encoder;

DROP INDEX mv_proxy_encoder_docs_max_idx;

CREATE INDEX mv_proxy_encoder_docs_fde_idx ON mv_proxy_encoder_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops)
  WITH (multivector_graph = document_nodes, multivector_proxy_encoder = random_projection_fde);

DO $$
DECLARE
	q turbohybrid_multivector := turbohybrid_multivector(ARRAY[
		'[1,0,0,0]'::vector,
		'[0,1,0,0]'::vector
	]);
	stats jsonb;
BEGIN
	PERFORM id
	FROM mv_proxy_encoder_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  dense_k => 2,
	  final_k => 1
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();

	IF stats->>'proxy_encoder_kind' <> 'random_projection_fde' OR
		(stats->>'proxy_candidates')::int <= 0 OR
		(stats->>'proxy_exact_rerank_docs')::int <= 0 THEN
		RAISE EXCEPTION 'expected random_projection_fde proxy stats, got %', stats;
	END IF;
END
$$;

SELECT turbohybrid_last_scan_stats()->>'proxy_encoder_kind'
  AS document_node_proxy_fde_encoder;

DROP INDEX mv_proxy_encoder_docs_fde_idx;

DO $$
BEGIN
	EXECUTE 'CREATE INDEX mv_proxy_encoder_docs_learned_idx ON mv_proxy_encoder_docs USING turbohybrid ' ||
		'(colbert multivector_cosine_turbohybrid_ops) WITH (multivector_graph = document_nodes, ' ||
		'multivector_proxy_encoder = learned_projection_placeholder)';
	RAISE EXCEPTION 'learned projection placeholder index build unexpectedly succeeded';
EXCEPTION WHEN feature_not_supported THEN
	IF SQLERRM NOT LIKE '%learned multivector proxy projection is not configured%' THEN
		RAISE;
	END IF;
END
$$;

DROP TABLE mv_proxy_encoder_docs;

SET turbohybrid.multivector_branch_plan = qdrant_like;

DO $$
DECLARE
	q turbohybrid_multivector := turbohybrid_multivector(ARRAY[
		'[1,0,0,0]'::vector,
		'[0,1,0,0]'::vector,
		'[0,0,1,0]'::vector,
		'[0,0,0,1]'::vector
	]);
	stats jsonb;
BEGIN
	PERFORM set_config('turbohybrid.multivector_doc_graph_rescore_k', '2', true);
	PERFORM id
	FROM mv_many_moderate_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  fusion => 'rrf',
	  dense_k => 1,
	  final_k => 1
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();

	IF stats->>'multivector_branch_plan' <> 'qdrant_like' OR
		stats->>'branch_fusion_mode' <> 'qdrant_like_rrf' OR
		(stats->>'branch_count')::int <> 3 OR
		stats->'branch_kinds' <> '["proxy_vector", "document_nodes", "exact_doc_scan"]'::jsonb OR
		stats->'branch_ranks' <> '[1, 2, 3]'::jsonb OR
		(stats->'branch_candidate_counts'->>0)::int <> 4 OR
		(stats->'branch_candidate_counts'->>1)::int <> 4 OR
		(stats->'branch_candidate_counts'->>2)::int <> 2 OR
		(stats->>'multivector_doc_graph_exact_rerank_docs')::int <> 2 OR
		(stats->>'proxy_exact_rerank_docs')::int <> 2 THEN
		RAISE EXCEPTION 'unexpected qdrant-like proxy branch plan stats: %', stats;
	END IF;
END
$$;

SELECT turbohybrid_last_scan_stats()->>'multivector_branch_plan'
  AS qdrant_like_proxy_plan,
       turbohybrid_last_scan_stats()->'branch_kinds'
  AS qdrant_like_proxy_branch_kinds,
       turbohybrid_last_scan_stats()->'branch_candidate_counts'
  AS qdrant_like_proxy_candidate_counts,
       (turbohybrid_last_scan_stats()->>'proxy_exact_rerank_docs')::int
  AS qdrant_like_proxy_exact_rerank_docs;

SELECT turbohybrid_sparse_vector_terms(
         turbohybrid_sparse_vector_from_arrays(ARRAY[7, 9], ARRAY[2.0, 1.0]::real[])
       ) AS learned_sparse_terms,
       turbohybrid_sparse_vector_query_terms(
         turbohybrid_sparse_vector_from_arrays(ARRAY[7, 9], ARRAY[2.0, 1.0]::real[])
       ) AS learned_sparse_query_terms,
       turbohybrid_sparse_vector_to_tsvector(
         turbohybrid_sparse_vector_from_arrays(ARRAY[7, 9], ARRAY[2.0, 1.0]::real[])
       ) AS learned_sparse_tsvector,
       turbohybrid_sparse_vector_to_tsquery(
         turbohybrid_sparse_vector_from_arrays(ARRAY[7, 9], ARRAY[2.0, 1.0]::real[])
       ) AS learned_sparse_tsquery;

CREATE TABLE mv_learned_sparse_docs (
	id text PRIMARY KEY,
	colbert turbohybrid_multivector,
	learned_sparse turbohybrid_sparse_vector,
	learned_sparse_tsv tsvector GENERATED ALWAYS AS
	  (turbohybrid_sparse_vector_to_tsvector(learned_sparse)) STORED
);

INSERT INTO mv_learned_sparse_docs (id, colbert, learned_sparse) VALUES
('alpha', turbohybrid_multivector(ARRAY['[1,0,0,0]'::vector]),
          turbohybrid_sparse_vector_from_arrays(ARRAY[7], ARRAY[1.0]::real[])),
('beta', turbohybrid_multivector(ARRAY['[0.8,0.6,0,0]'::vector]),
         turbohybrid_sparse_vector_from_arrays(ARRAY[42], ARRAY[1.0]::real[]));

CREATE INDEX mv_learned_sparse_docs_idx
  ON mv_learned_sparse_docs USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops,
   learned_sparse_tsv bm25_tsvector_turbohybrid_ops)
  WITH (multivector_graph = document_nodes,
        graph_ef_search = 1);

SET turbohybrid.multivector_sparse_candidate_source = learned_sparse;
SET turbohybrid.multivector_bm25_candidate_injection = off;

DO $$
DECLARE
	q turbohybrid_multivector := turbohybrid_multivector(ARRAY['[1,0,0,0]'::vector]);
	sq turbohybrid_sparse_vector :=
		turbohybrid_sparse_vector_from_arrays(ARRAY[42], ARRAY[1.0]::real[]);
	stats jsonb;
	top1 text;
BEGIN
	SELECT id INTO top1
	FROM mv_learned_sparse_docs
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  text_query => turbohybrid_sparse_vector_to_tsquery(sq),
	  dense_k => 1,
	  bm25_k => 2,
	  bm25_weight => 0,
	  final_k => 2
	)
	LIMIT 1;
	stats := turbohybrid_last_scan_stats();

	IF top1 <> 'alpha' OR
		(stats->>'learned_sparse_candidates')::int <= 0 OR
		(stats->>'learned_sparse_retained_for_maxsim')::int < 0 OR
		(stats->>'learned_sparse_branch_latency_us')::bigint < 0 THEN
		RAISE EXCEPTION 'unexpected learned sparse exact-MaxSim admission result %, stats %',
			top1, stats;
	END IF;
END
$$;

SELECT turbohybrid_last_scan_stats()->>'learned_sparse_candidates'
  AS learned_sparse_candidates,
       turbohybrid_last_scan_stats()->>'learned_sparse_retained_for_maxsim'
  AS learned_sparse_retained_for_maxsim;

RESET turbohybrid.multivector_sparse_candidate_source;
RESET turbohybrid.multivector_bm25_candidate_injection;
DROP TABLE mv_learned_sparse_docs;

RESET turbohybrid.multivector_candidate_source;
RESET turbohybrid.multivector_branch_plan;
RESET turbohybrid.multivector_doc_graph_search_ef;
RESET turbohybrid.multivector_doc_graph_oversampling;
RESET turbohybrid.multivector_doc_graph_rescore_k;
RESET turbohybrid.multivector_doc_storage;

CREATE TABLE mv_doc_node_insert_many_moderate_bulk (
  id text PRIMARY KEY,
  colbert turbohybrid_multivector
);

CREATE TABLE mv_doc_node_insert_many_moderate_inc (
  id text PRIMARY KEY,
  colbert turbohybrid_multivector
);

INSERT INTO mv_doc_node_insert_many_moderate_bulk VALUES
  ('spike_1', turbohybrid_multivector(ARRAY['[1,0,0,0]'::vector])),
  ('spike_2', turbohybrid_multivector(ARRAY['[0,1,0,0]'::vector])),
  ('spike_3', turbohybrid_multivector(ARRAY['[0,0,1,0]'::vector])),
  ('spike_4', turbohybrid_multivector(ARRAY['[0,0,0,1]'::vector])),
  ('good', turbohybrid_multivector(ARRAY[
    '[0.8,0.6,0,0]'::vector,
    '[0.6,0.8,0,0]'::vector,
    '[0,0,0.8,0.6]'::vector,
    '[0,0,0.6,0.8]'::vector
  ]));

INSERT INTO mv_doc_node_insert_many_moderate_inc
SELECT *
FROM mv_doc_node_insert_many_moderate_bulk
WHERE id <> 'good';

CREATE INDEX mv_doc_node_insert_many_moderate_bulk_idx
  ON mv_doc_node_insert_many_moderate_bulk USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops)
  WITH (multivector_graph = document_nodes,
        graph_m = 4,
        graph_ef_construction = 8,
        graph_ef_search = 1);

CREATE INDEX mv_doc_node_insert_many_moderate_inc_idx
  ON mv_doc_node_insert_many_moderate_inc USING turbohybrid
  (colbert multivector_cosine_turbohybrid_ops)
  WITH (multivector_graph = document_nodes,
        graph_m = 4,
        graph_ef_construction = 8,
        graph_ef_search = 1);

INSERT INTO mv_doc_node_insert_many_moderate_inc VALUES
  ('good', turbohybrid_multivector(ARRAY[
    '[0.8,0.6,0,0]'::vector,
    '[0.6,0.8,0,0]'::vector,
    '[0,0,0.8,0.6]'::vector,
    '[0,0,0.6,0.8]'::vector
  ]));

DO $$
DECLARE
	q turbohybrid_multivector := turbohybrid_multivector(ARRAY[
		'[1,0,0,0]'::vector,
		'[0,1,0,0]'::vector,
		'[0,0,1,0]'::vector,
		'[0,0,0,1]'::vector
	]);
	bulk_top1 text;
	inc_top1 text;
	bulk_stats jsonb;
	inc_stats jsonb;
BEGIN
	SELECT id INTO bulk_top1
	FROM mv_doc_node_insert_many_moderate_bulk
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  dense_k => 1,
	  final_k => 1
	)
	LIMIT 1;
	bulk_stats := turbohybrid_last_scan_stats();

	SELECT id INTO inc_top1
	FROM mv_doc_node_insert_many_moderate_inc
	ORDER BY colbert <~> turbohybrid_query(
	  multivector_query => q,
	  dense_k => 1,
	  final_k => 1
	)
	LIMIT 1;
	inc_stats := turbohybrid_last_scan_stats();

	IF bulk_top1 <> 'good' OR inc_top1 <> 'good' OR
		bulk_stats->>'multivector_doc_graph_warning' <>
			'document_node_proxy_vector_graph_traversal' OR
		inc_stats->>'multivector_doc_graph_warning' <>
			'document_node_proxy_vector_graph_traversal' OR
		(bulk_stats->>'multivector_doc_graph_candidates')::int <> 1 OR
		(inc_stats->>'multivector_doc_graph_candidates')::int <> 1 OR
		(bulk_stats->>'graph_entry_point_count')::int <= 1 OR
		(inc_stats->>'graph_entry_point_count')::int <= 1 OR
		(inc_stats->>'multivector_doc_graph_nodes')::int <> 5 OR
		(inc_stats->>'multivector_doc_graph_insert_full_maxsim_edges')::int <= 0 OR
		(inc_stats->>'multivector_doc_graph_insert_representative_fallbacks')::int <> 0 OR
		(inc_stats->>'multivector_doc_graph_insert_pairs_scored')::int <= 0 THEN
		RAISE EXCEPTION 'expected bulk and incremental document-node graph recall parity, bulk top %, inc top %, bulk stats %, inc stats %',
			bulk_top1, inc_top1, bulk_stats, inc_stats;
	END IF;
END
$$;

SELECT turbohybrid_last_scan_stats()->>'multivector_doc_graph_warning'
  AS document_node_insert_many_moderate_warning,
       (turbohybrid_last_scan_stats()->>'multivector_doc_graph_nodes')::int
  AS document_node_insert_many_moderate_nodes,
       (turbohybrid_last_scan_stats()->>'multivector_doc_graph_candidates')::int
  AS document_node_insert_many_moderate_candidates,
       (turbohybrid_last_scan_stats()->>'graph_entry_point_count')::int > 1
  AS document_node_insert_sampled_entries,
       (turbohybrid_last_scan_stats()->>'multivector_doc_graph_insert_full_maxsim_edges')::int > 0
  AS document_node_insert_full_maxsim_edges,
       (turbohybrid_last_scan_stats()->>'multivector_doc_graph_insert_representative_fallbacks')::int
  AS document_node_insert_representative_fallbacks,
       (turbohybrid_last_scan_stats()->>'multivector_doc_graph_insert_pairs_scored')::int > 0
  AS document_node_insert_pairs_scored;

DROP TABLE mv_doc_node_insert_many_moderate_inc;
DROP TABLE mv_doc_node_insert_many_moderate_bulk;

DROP TABLE mv_many_moderate_docs;
RESET turbohybrid.multivector_candidate_source;
RESET turbohybrid.multivector_candidate_reservoirs;
RESET turbohybrid.multivector_debug_skip_query_tokens;
RESET turbohybrid.multivector_plain_fallback;
RESET turbohybrid.multivector_debug_trace_limit;
RESET turbohybrid.multivector_debug_admission;
RESET turbohybrid.multivector_adaptive_widening;
RESET turbohybrid.multivector_doc_candidate_k;
RESET turbohybrid.multivector_max_raw_hits_per_token;
RESET turbohybrid.multivector_unique_docs_per_token;
RESET turbohybrid.multivector_subvector_k;
RESET enable_seqscan;
