DO $$
BEGIN
	IF hybrid_query_out(hybrid_query(
		vector_query => '[1,0,0]'::vector,
		text_query => websearch_to_tsquery('english', 'postgres vector search'),
		fusion => 'rrf',
		dense_k => 12,
		bm25_k => 34,
		rrf_k => 60
	))::text != 'hybrid_query(fusion=rrf,vector=true,tsquery=true,dense_weight=1,bm25_weight=1,alpha=null,rrf_k=60,dense_k=12,bm25_k=34,final_k=null,require_bm25_match=false)' THEN
		RAISE EXCEPTION 'unexpected hybrid_query output';
	END IF;
END
$$;

DO $$
BEGIN
	IF hybrid_query_out(hybrid_query(
		text_query => websearch_to_tsquery('english', 'postgres'),
		fusion => 'weighted',
		alpha => 0.25,
		final_k => 10,
		require_bm25_match => true
	))::text != 'hybrid_query(fusion=weighted,vector=false,tsquery=true,dense_weight=1,bm25_weight=1,alpha=0.25,rrf_k=60,dense_k=400,bm25_k=400,final_k=10,require_bm25_match=true)' THEN
		RAISE EXCEPTION 'unexpected weighted hybrid_query output';
	END IF;
END
$$;

DO $$
BEGIN
	IF hybrid_query_out(hybrid_query(
		vector_query => '[1,0,0]'::vector,
		final_k => NULL
	))::text != 'hybrid_query(fusion=rrf,vector=true,tsquery=false,dense_weight=1,bm25_weight=1,alpha=null,rrf_k=60,dense_k=400,bm25_k=400,final_k=null,require_bm25_match=false)' THEN
		RAISE EXCEPTION 'unexpected explicit NULL final_k output';
	END IF;
END
$$;

SET hybrid.default_dense_k = 7;
SET hybrid.default_bm25_k = 8;
SET hybrid.default_rrf_k = 9;

DO $$
BEGIN
	IF hybrid_query_out(hybrid_query(
		vector_query => '[1,0,0]'::vector,
		rrf_k => NULL,
		dense_k => NULL,
		bm25_k => NULL
	))::text != 'hybrid_query(fusion=rrf,vector=true,tsquery=false,dense_weight=1,bm25_weight=1,alpha=null,rrf_k=9,dense_k=7,bm25_k=8,final_k=null,require_bm25_match=false)' THEN
		RAISE EXCEPTION 'unexpected explicit NULL GUC default output';
	END IF;
END
$$;

DO $$
BEGIN
	IF hybrid_query_out(hybrid_query(
		vector_query => '[1,0,0]'::vector
	))::text != 'hybrid_query(fusion=rrf,vector=true,tsquery=false,dense_weight=1,bm25_weight=1,alpha=null,rrf_k=9,dense_k=7,bm25_k=8,final_k=null,require_bm25_match=false)' THEN
		RAISE EXCEPTION 'unexpected omitted GUC default output';
	END IF;
END
$$;

RESET hybrid.default_dense_k;
RESET hybrid.default_bm25_k;
RESET hybrid.default_rrf_k;

CREATE TABLE hq_items (id int, val vector(3));
INSERT INTO hq_items VALUES
	(1, '[1,0,0]'),
	(2, '[1,1,0]'),
	(3, '[0,1,0]');

DO $$
DECLARE
	ids int[];
BEGIN
	SELECT array_agg(id ORDER BY val <~> hybrid_query(vector_query => '[1,0,0]'::vector)) INTO ids
	FROM hq_items;

	IF ids != ARRAY[1, 2, 3] THEN
		RAISE EXCEPTION 'unexpected vector-only hybrid ordering: %', ids;
	END IF;
END
$$;

DO $$
BEGIN
	IF NOT (
		('[2,0,0]'::vector <~-> hybrid_query(vector_query => '[1,0,0]'::vector)) = 1 AND
		('[2,0,0]'::vector <~#> hybrid_query(vector_query => '[1,0,0]'::vector)) = -2 AND
		('[2,0,0]'::vector <~> hybrid_query(vector_query => '[1,0,0]'::vector)) = 0
	) THEN
		RAISE EXCEPTION 'unexpected hybrid distance operator results';
	END IF;
END
$$;

DO $$
DECLARE
	ok bool;
BEGIN
	SELECT bool_and(provolatile = 's' AND proparallel = 'u') INTO ok
	FROM pg_proc
	WHERE proname IN (
		'hybrid_distance',
		'hybrid_l2_distance',
		'hybrid_negative_inner_product',
		'hybrid_cosine_distance'
	);

	IF NOT ok THEN
		RAISE EXCEPTION 'hybrid distance functions should be plan-context aware';
	END IF;
END
$$;

DO $$
BEGIN
	PERFORM hybrid_distance('[1,0,0]'::vector, hybrid_query(
		text_query => websearch_to_tsquery('english', 'postgres')
	));
	RAISE EXCEPTION 'expected hybrid text fallback rejection';
EXCEPTION WHEN feature_not_supported THEN
END
$$;

DO $$
BEGIN
	PERFORM id
	FROM hq_items
	ORDER BY val <~> hybrid_query(
		vector_query => '[1,0,0]'::vector,
		text_query => websearch_to_tsquery('english', 'postgres')
	)
	LIMIT 1;
	RAISE EXCEPTION 'expected scalar hybrid ordering fallback rejection';
EXCEPTION WHEN feature_not_supported THEN
END
$$;

DO $$
BEGIN
	PERFORM hybrid_query(vector_query => '[1]'::vector, dense_k => -1);
	RAISE EXCEPTION 'expected dense_k validation error';
EXCEPTION WHEN invalid_parameter_value THEN
END
$$;

DO $$
BEGIN
	PERFORM hybrid_query(vector_query => '[1]'::vector, fusion => 'bogus');
	RAISE EXCEPTION 'expected fusion validation error';
EXCEPTION WHEN invalid_parameter_value THEN
END
$$;

DO $$
BEGIN
	PERFORM hybrid_query(vector_query => '[1]'::vector, alpha => 1.5);
	RAISE EXCEPTION 'expected alpha validation error';
EXCEPTION WHEN invalid_parameter_value THEN
END
$$;

DO $$
BEGIN
	PERFORM hybrid_query(vector_query => '[1]'::vector, rrf_k => 0);
	RAISE EXCEPTION 'expected rrf_k validation error';
EXCEPTION WHEN invalid_parameter_value THEN
END
$$;

DO $$
BEGIN
	PERFORM 'anything'::hybrid_query;
	RAISE EXCEPTION 'expected hybrid_query input rejection';
EXCEPTION WHEN feature_not_supported THEN
END
$$;

DROP TABLE hq_items;
