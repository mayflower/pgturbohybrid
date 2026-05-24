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

SET enable_seqscan = off;

CREATE TABLE tqh_docs (
	id int,
	embedding vector(3),
	body_tsv tsvector
);

INSERT INTO tqh_docs VALUES
	(1, '[1,0,0]', to_tsvector('english', 'postgres vector search')),
	(2, '[1,1,0]', to_tsvector('english', 'hybrid bm25 search')),
	(3, '[0,1,0]', to_tsvector('english', 'lexical search')),
	(4, '[0,0,1]', to_tsvector('english', 'unrelated document'));

CREATE INDEX tqh_docs_idx ON tqh_docs
USING turbohybrid (
	embedding vector_cosine_turbohybrid_ops,
	body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (quantization_bits = 4, exact_storage = on);

DO $$
DECLARE
	opclasses text[];
BEGIN
	SELECT array_agg(opcname ORDER BY opcname) INTO opclasses
	FROM pg_opclass
	WHERE opcmethod = (SELECT oid FROM pg_am WHERE amname = 'turbohybrid');

	IF opclasses <> ARRAY[
		'bm25_tsvector_turbohybrid_ops',
		'vector_cosine_turbohybrid_ops',
		'vector_ip_turbohybrid_ops',
		'vector_l2_turbohybrid_ops'
	] THEN
		RAISE EXCEPTION 'unexpected pgturbohybrid opclasses: %', opclasses;
	END IF;
END
$$;

CREATE INDEX tqh_wrong_column_idx ON tqh_docs
USING turbohybrid (body_tsv vector_cosine_turbohybrid_ops);

CREATE INDEX tqh_wrong_opclass_idx ON tqh_docs
USING turbohybrid (embedding bm25_tsvector_turbohybrid_ops);

CREATE INDEX tqh_docs_hnsw_idx ON tqh_docs
USING hnsw (embedding vector_l2_ops);

DO $$
DECLARE
	hnsw_top_id int;
	pgturbohybrid_top_id int;
BEGIN
	SELECT id INTO hnsw_top_id
	FROM tqh_docs
	ORDER BY embedding <-> '[1,0,0]'::vector
	LIMIT 1;

	SELECT id INTO pgturbohybrid_top_id
	FROM tqh_docs
	ORDER BY embedding <~> turbohybrid_query(
		vector_query => '[1,0,0]'::vector,
		dense_k => 4,
		final_k => 1
	)
	LIMIT 1;

	IF hnsw_top_id <> 1 OR pgturbohybrid_top_id <> 1 THEN
		RAISE EXCEPTION 'unexpected coexistence results: hnsw %, pgturbohybrid %',
			hnsw_top_id, pgturbohybrid_top_id;
	END IF;
END
$$;

DROP INDEX tqh_docs_idx;

DO $$
DECLARE
	top_id int;
BEGIN
	SELECT id INTO top_id
	FROM tqh_docs
	ORDER BY embedding <-> '[1,0,0]'::vector
	LIMIT 1;

	IF top_id <> 1 THEN
		RAISE EXCEPTION 'pgvector HNSW index changed after dropping pgturbohybrid index: %', top_id;
	END IF;
END
$$;

CREATE INDEX tqh_docs_idx ON tqh_docs
USING turbohybrid (
	embedding vector_cosine_turbohybrid_ops,
	body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (quantization_bits = 4, exact_storage = on);

DROP INDEX tqh_docs_hnsw_idx;

DO $$
DECLARE
	top_id int;
BEGIN
	SELECT id INTO top_id
	FROM tqh_docs
	ORDER BY embedding <~> turbohybrid_query(
		vector_query => '[1,0,0]'::vector,
		dense_k => 4,
		final_k => 1
	)
	LIMIT 1;

	IF top_id <> 1 THEN
		RAISE EXCEPTION 'pgturbohybrid index changed after dropping pgvector HNSW index: %', top_id;
	END IF;
END
$$;

DO $$
DECLARE
	ids int[];
BEGIN
	SELECT array_agg(id) INTO ids
	FROM (
		SELECT id
		FROM tqh_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			dense_k => 4,
			final_k => 3
		)
		LIMIT 3
	) s;

	IF ids <> ARRAY[1,2,3] THEN
		RAISE EXCEPTION 'unexpected dense-only results: %', ids;
	END IF;
END
$$;

DO $$
DECLARE
	ids int[];
BEGIN
	SELECT array_agg(id) INTO ids
	FROM (
		SELECT id
		FROM tqh_docs
		ORDER BY embedding <~> turbohybrid_query(
			text_query => websearch_to_tsquery('english', 'lexical search'),
			dense_k => 0,
			bm25_k => 4,
			final_k => 2
		)
		LIMIT 2
	) s;

	IF ids <> ARRAY[3] THEN
		RAISE EXCEPTION 'unexpected BM25-only results: %', ids;
	END IF;
END
$$;

DO $$
DECLARE
	ids int[];
BEGIN
	SELECT array_agg(id) INTO ids
	FROM (
		SELECT id
		FROM tqh_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			text_query => websearch_to_tsquery('english', 'postgres search'),
			dense_k => 4,
			bm25_k => 4,
			final_k => 3
		)
		LIMIT 3
	) s;

	IF ids <> ARRAY[1,2,3] THEN
		RAISE EXCEPTION 'unexpected hybrid results: %', ids;
	END IF;
END
$$;

DO $$
DECLARE
	top_id int;
BEGIN
	EXECUTE 'PREPARE tqh_prepared(vector) AS
		SELECT id
		FROM tqh_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => $1,
			dense_k => 4,
			final_k => 1
		)
		LIMIT 1';
	EXECUTE 'EXECUTE tqh_prepared(''[1,0,0]''::vector)' INTO top_id;
	EXECUTE 'DEALLOCATE tqh_prepared';

	IF top_id <> 1 THEN
		RAISE EXCEPTION 'unexpected prepared statement result: %', top_id;
	END IF;
END
$$;

DO $$
DECLARE
	top_id int;
BEGIN
	WITH q AS (
		SELECT turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			dense_k => 4,
			final_k => 1
		) AS query
	)
	SELECT id INTO top_id
	FROM tqh_docs, q
	ORDER BY embedding <~> q.query
	LIMIT 1;

	IF top_id <> 1 THEN
		RAISE EXCEPTION 'unexpected CTE query result: %', top_id;
	END IF;
END
$$;

DO $$
DECLARE
	top_id int;
BEGIN
	SELECT id INTO top_id
	FROM (
		SELECT *
		FROM tqh_docs
	) d
	ORDER BY d.embedding <~> turbohybrid_query(
		vector_query => '[1,0,0]'::vector,
		dense_k => 4,
		final_k => 1
	)
	LIMIT 1;

	IF top_id <> 1 THEN
		RAISE EXCEPTION 'unexpected subquery result: %', top_id;
	END IF;
END
$$;

CREATE TEMP TABLE tqh_filter (id int PRIMARY KEY);
INSERT INTO tqh_filter VALUES (1), (3);

DO $$
DECLARE
	top_id int;
BEGIN
	SELECT d.id INTO top_id
	FROM tqh_docs d
	JOIN tqh_filter f USING (id)
	ORDER BY d.embedding <~> turbohybrid_query(
		vector_query => '[1,0,0]'::vector,
		dense_k => 4,
		final_k => 1
	)
	LIMIT 1;

	IF top_id <> 1 THEN
		RAISE EXCEPTION 'unexpected join query result: %', top_id;
	END IF;
END
$$;

DO $$
DECLARE
	distance float8;
BEGIN
	SELECT embedding <~-> turbohybrid_query(
		vector_query => '[1,0,0]'::vector
	) INTO distance
	FROM tqh_docs
	WHERE id = 1;

	IF distance <> 0 THEN
		RAISE EXCEPTION 'unexpected projected dense distance: %', distance;
	END IF;
END
$$;

DO $$
BEGIN
	PERFORM embedding <~> turbohybrid_query(
		vector_query => '[1,0,0]'::vector,
		text_query => websearch_to_tsquery('english', 'postgres')
	)
	FROM tqh_docs
	WHERE id = 1;
	RAISE EXCEPTION 'expected text-aware scalar projection rejection';
EXCEPTION WHEN feature_not_supported THEN
END
$$;

INSERT INTO tqh_docs VALUES
	(5, '[1,0,1]', to_tsvector('english', 'fresh delta term'));

DO $$
DECLARE
	top_id int;
BEGIN
	SELECT id INTO top_id
	FROM tqh_docs
	ORDER BY embedding <~> turbohybrid_query(
		text_query => websearch_to_tsquery('english', 'fresh delta'),
		dense_k => 0,
		bm25_k => 4,
		final_k => 1
	)
	LIMIT 1;

	IF top_id <> 5 THEN
		RAISE EXCEPTION 'unexpected delta BM25 result: %', top_id;
	END IF;
END
$$;

DO $$
DECLARE
	ids int[];
BEGIN
	SELECT array_agg(id) INTO ids
	FROM (
		SELECT id
		FROM tqh_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			text_query => websearch_to_tsquery('english', 'hybrid'),
			dense_k => 4,
			bm25_k => 4,
			final_k => 2,
			require_bm25_match => true
		)
		LIMIT 2
	) s;

	IF ids <> ARRAY[2] THEN
		RAISE EXCEPTION 'unexpected require_bm25_match results: %', ids;
	END IF;
END
$$;

DO $$
BEGIN
	PERFORM turbohybrid_index_stats('tqh_docs_idx'::regclass);
	PERFORM turbohybrid_last_scan_stats();
	PERFORM turbohybrid_simd_capabilities();
END
$$;

CREATE TABLE tqh_empty_docs (
	id int,
	embedding vector(3),
	body_tsv tsvector
);

CREATE INDEX tqh_empty_docs_idx ON tqh_empty_docs
USING turbohybrid (
	embedding vector_l2_turbohybrid_ops,
	body_tsv bm25_tsvector_turbohybrid_ops
);

DO $$
DECLARE
	result_count int;
BEGIN
	SELECT count(*) INTO result_count
	FROM (
		SELECT id
		FROM tqh_empty_docs
		ORDER BY embedding <~-> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			dense_k => 4,
			final_k => 1
		)
		LIMIT 1
	) s;

	IF result_count <> 0 THEN
		RAISE EXCEPTION 'unexpected empty index result count: %', result_count;
	END IF;
END
$$;

DROP TABLE tqh_empty_docs;

DO $$
DECLARE
	simd_on_ids int[];
	simd_off_ids int[];
BEGIN
	SET LOCAL turbohybrid.simd = on;
	SELECT array_agg(id) INTO simd_on_ids
	FROM (
		SELECT id
		FROM tqh_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			dense_k => 4,
			final_k => 3
		)
		LIMIT 3
	) s;

	SET LOCAL turbohybrid.simd = off;
	SELECT array_agg(id) INTO simd_off_ids
	FROM (
		SELECT id
		FROM tqh_docs
		ORDER BY embedding <~> turbohybrid_query(
			vector_query => '[1,0,0]'::vector,
			dense_k => 4,
			final_k => 3
		)
		LIMIT 3
	) s;

	IF simd_on_ids <> simd_off_ids THEN
		RAISE EXCEPTION 'SIMD result parity failed: on %, off %',
			simd_on_ids, simd_off_ids;
	END IF;
END
$$;

DO $$
DECLARE
	funcs text[];
BEGIN
	SELECT array_agg(p.proname ORDER BY p.proname) INTO funcs
	FROM pg_proc p
	JOIN pg_depend d ON d.objid = p.oid AND d.deptype = 'e'
	JOIN pg_extension e ON e.oid = d.refobjid
	WHERE e.extname = 'pgturbohybrid'
		AND p.proname IN (
			'turbohybrid_index_stats',
			'turbohybrid_last_scan_stats',
			'turbohybrid_simd_capabilities'
		);

	IF funcs <> ARRAY[
		'turbohybrid_index_stats',
		'turbohybrid_last_scan_stats',
		'turbohybrid_simd_capabilities'
	] THEN
		RAISE EXCEPTION 'unexpected pgturbohybrid diagnostic functions: %', funcs;
	END IF;

	IF EXISTS (
		SELECT 1
		FROM pg_proc p
		JOIN pg_depend d ON d.objid = p.oid AND d.deptype = 'e'
		JOIN pg_extension e ON e.oid = d.refobjid
		WHERE e.extname = 'pgturbohybrid'
			AND p.proname LIKE '%debug%'
	) THEN
		RAISE EXCEPTION 'pgturbohybrid extension should not install debug functions';
	END IF;
END
$$;

DO $$
DECLARE
	keys text[];
BEGIN
	SELECT array_agg(key ORDER BY key) INTO keys
	FROM jsonb_object_keys(turbohybrid_index_stats('tqh_docs_idx'::regclass)) AS key;

	IF keys <> ARRAY[
		'blocks',
		'bm25_average_document_length',
		'bm25_document_count',
		'exact_storage',
		'graph_ef_construction',
		'graph_ef_search',
		'graph_m',
		'graph_oversampling',
		'hybrid',
		'quantization_bits',
		'storage_kind',
		'version'
	] THEN
		RAISE EXCEPTION 'unexpected pgturbohybrid index stats keys: %', keys;
	END IF;

	SELECT array_agg(key ORDER BY key) INTO keys
	FROM jsonb_object_keys(turbohybrid_last_scan_stats()) AS key;

	IF keys <> ARRAY[
		'graph_candidate_count',
		'graph_dense_budget_policy',
		'graph_dense_requested_k',
		'graph_effective_rescore_band',
		'graph_effective_result_target',
		'graph_effective_search_ef',
		'graph_highdim_widening_multiplier',
		'graph_rescore_band_policy',
		'graph_rescore_count',
		'graph_storage_kind',
		'graph_widening_reason',
		'quantization_bits',
		'scan_orchestration',
		'version'
	] THEN
		RAISE EXCEPTION 'unexpected pgturbohybrid last scan stats keys: %', keys;
	END IF;

	SELECT array_agg(key ORDER BY key) INTO keys
	FROM jsonb_object_keys(turbohybrid_simd_capabilities()) AS key;

	IF keys <> ARRAY[
		'architecture',
		'compile_arm_dotprod',
		'compile_arm_i8mm',
		'compile_avx2',
		'compile_avx512_weighted',
		'compile_avx512vnni',
		'compile_avx512vpopcntdq',
		'compile_avxvnni',
		'simd_build_disabled',
		'version'
	] THEN
		RAISE EXCEPTION 'unexpected pgturbohybrid SIMD capability keys: %', keys;
	END IF;
END
$$;

DROP TABLE tqh_docs;
