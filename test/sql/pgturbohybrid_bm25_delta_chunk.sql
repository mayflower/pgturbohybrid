\set ON_ERROR_STOP on
SET client_min_messages = warning;

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;

DROP TABLE IF EXISTS tqh_delta_chunk_docs;
CREATE TABLE tqh_delta_chunk_docs (
	id int PRIMARY KEY,
	embedding vector(3),
	body_tsv tsvector
);

INSERT INTO tqh_delta_chunk_docs (id, embedding, body_tsv) VALUES
	(1, '[1,0,0]', to_tsvector('simple', 'seedalpha'));

CREATE INDEX tqh_delta_chunk_docs_idx ON tqh_delta_chunk_docs
USING turbohybrid (
	embedding vector_cosine_turbohybrid_ops,
	body_tsv bm25_tsvector_turbohybrid_ops
);

DO $$
DECLARE
	big_tsv tsvector;
	term_count int;
BEGIN
	SELECT to_tsvector(
		'simple',
		(SELECT string_agg('chunkterm' || g::text, ' ')
		 FROM generate_series(1, 500) AS g)
	) INTO big_tsv;

	term_count := array_length(tsvector_to_array(big_tsv), 1);
	IF term_count IS NULL OR term_count < 400 THEN
		RAISE EXCEPTION 'expected large tsvector for chunking test, got % terms', term_count;
	END IF;

	INSERT INTO tqh_delta_chunk_docs (id, embedding, body_tsv)
	VALUES (2, '[0.9,0.1,0]', big_tsv);
END;
$$;

ANALYZE tqh_delta_chunk_docs;
SET enable_seqscan = off;
SET enable_bitmapscan = off;

SELECT id
FROM tqh_delta_chunk_docs
ORDER BY embedding <~> turbohybrid_query(
	text_query => websearch_to_tsquery('simple', 'chunkterm250'),
	dense_k => 0,
	bm25_k => 8,
	final_k => 4
)
LIMIT 1;

DO $$
DECLARE
	stats jsonb;
BEGIN
	stats := turbohybrid_last_scan_stats();
	IF (stats->>'bm25_branch_used')::boolean IS DISTINCT FROM true THEN
		RAISE EXCEPTION 'expected BM25 branch after chunked delta insert: %', stats;
	END IF;
END;
$$;

SELECT 'bm25_delta_chunk_pass' AS status;