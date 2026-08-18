-- BM25 reader cache insert-tolerance contract:
--   1. a document inserted AFTER the per-backend cache was built must be
--      findable by a BM25 query in the SAME backend (base cache reused,
--      delta entries rebuilt -- the old code rebuilt everything, the new
--      code must not lose the new doc);
--   2. a document deleted and vacuumed must stop matching (liveness reload
--      is keyed by graphFlags, which vacuum bumps);
--   3. corpus statistics stay consistent through reuse.
\set ON_ERROR_STOP on
SET client_min_messages = warning;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;
RESET client_min_messages;

CREATE TABLE tq_cache_reuse_docs (
	id int PRIMARY KEY,
	embedding vector(3),
	body_tsv tsvector
);
INSERT INTO tq_cache_reuse_docs VALUES
	(1, '[0.1,0.2,0.3]', to_tsvector('simple', 'alpha beta gamma')),
	(2, '[0.2,0.3,0.1]', to_tsvector('simple', 'alpha delta epsilon')),
	(3, '[0.3,0.1,0.2]', to_tsvector('simple', 'beta zeta alpha'));
CREATE INDEX tq_cache_reuse_idx ON tq_cache_reuse_docs
USING turbohybrid (
	embedding vector_cosine_turbohybrid_ops,
	body_tsv bm25_tsvector_turbohybrid_ops
);

-- Build the reader cache in this backend, then check the marker term is
-- unknown (forces lexicon + delta lookups through the cache).
SET enable_sort = off;
SELECT id AS pre_insert_ids FROM tq_cache_reuse_docs
ORDER BY embedding <~> turbohybrid_query(
	vector_query => '[0.2,0.2,0.2]'::vector,
	text_query => websearch_to_tsquery('simple', 'zzqqxxmarker'))
LIMIT 5;

-- Insert after the cache exists: same backend must find it on the next query.
INSERT INTO tq_cache_reuse_docs VALUES
	(4, '[0.25,0.25,0.25]', to_tsvector('simple', 'zzqqxxmarker fresh doc'));
SELECT id FROM tq_cache_reuse_docs
ORDER BY embedding <~> turbohybrid_query(
	vector_query => '[0.2,0.2,0.2]'::vector,
	text_query => websearch_to_tsquery('simple', 'zzqqxxmarker'))
LIMIT 5;

-- The cached base still answers the old vocabulary.
SELECT id AS old_vocab_ids FROM tq_cache_reuse_docs
ORDER BY embedding <~> turbohybrid_query(
	vector_query => '[0.2,0.2,0.2]'::vector,
	text_query => websearch_to_tsquery('simple', 'delta'))
LIMIT 5;

-- Delete + vacuum: liveness must drop the dead doc even with a warm cache.
DELETE FROM tq_cache_reuse_docs WHERE id = 4;
VACUUM tq_cache_reuse_docs;
SELECT id AS post_vacuum_ids FROM tq_cache_reuse_docs
ORDER BY embedding <~> turbohybrid_query(
	vector_query => '[0.2,0.2,0.2]'::vector,
	text_query => websearch_to_tsquery('simple', 'zzqqxxmarker'))
LIMIT 5;

-- Hybrid path through the same reused cache.
SELECT id AS hybrid_ids FROM tq_cache_reuse_docs
ORDER BY embedding <~> turbohybrid_query(
	vector_query => '[0.2,0.2,0.2]'::vector,
	text_query => websearch_to_tsquery('simple', 'alpha beta'))
LIMIT 5;

DROP TABLE tq_cache_reuse_docs;
