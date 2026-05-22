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
	embedding vector_cosine_hybrid_ops,
	body_tsv bm25_tsvector_ops
)
WITH (tq_exact_storage = on);

DO $$
DECLARE
	ids int[];
BEGIN
	SELECT array_agg(id) INTO ids
	FROM (
		SELECT id
		FROM tqh_docs
		ORDER BY embedding <~> hybrid_query(
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
		ORDER BY embedding <~> hybrid_query(
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
		ORDER BY embedding <~> hybrid_query(
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

INSERT INTO tqh_docs VALUES
	(5, '[1,0,1]', to_tsvector('english', 'fresh delta term'));

DO $$
DECLARE
	top_id int;
BEGIN
	SELECT id INTO top_id
	FROM tqh_docs
	ORDER BY embedding <~> hybrid_query(
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
		ORDER BY embedding <~> hybrid_query(
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

DROP TABLE tqh_docs;
