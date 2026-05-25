-- pgturbohybrid fast-start example
--
-- Run after installing pgvector and pgturbohybrid:
--   psql -d your_database -f examples/fast_start.sql

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;

DROP TABLE IF EXISTS turbohybrid_fast_start_docs;

CREATE TABLE turbohybrid_fast_start_docs (
	id bigserial PRIMARY KEY,
	embedding vector(3) NOT NULL,
	body text NOT NULL,
	body_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', body)) STORED
);

INSERT INTO turbohybrid_fast_start_docs (embedding, body) VALUES
	('[1.00,0.00,0.00]', 'PostgreSQL vector search with hybrid full text ranking'),
	('[0.98,0.10,0.00]', 'Fast pgvector retrieval using TurboHybrid indexes'),
	('[0.95,0.05,0.05]', 'Hybrid search combines semantic vectors and BM25 terms'),
	('[0.90,0.20,0.00]', 'Latency focused retrieval for RAG applications'),
	('[0.80,0.30,0.10]', 'Dense embeddings and lexical matching in PostgreSQL'),
	('[0.70,0.40,0.10]', 'BM25 helps exact words such as postgres and indexes'),
	('[0.20,0.90,0.10]', 'Vacuum maintenance and relational database operations'),
	('[0.10,0.95,0.10]', 'SQL joins filters and transactions in application data'),
	('[0.00,1.00,0.20]', 'Analytics dashboards over structured business records'),
	('[0.10,0.10,0.95]', 'Image metadata and media asset catalog search'),
	('[0.05,0.15,0.90]', 'Audio archive retrieval with tags and descriptions'),
	('[0.30,0.20,0.80]', 'Document storage with generated text search columns');

CREATE INDEX turbohybrid_fast_start_docs_idx
ON turbohybrid_fast_start_docs
USING turbohybrid (
	embedding vector_cosine_turbohybrid_ops,
	body_tsv bm25_tsvector_turbohybrid_ops
);

ANALYZE turbohybrid_fast_start_docs;

SHOW turbohybrid.profile;
SHOW turbohybrid.default_dense_k;
SHOW turbohybrid.default_bm25_k;

-- This demo table is tiny, so make the planner show the index-backed path.
-- Do not set this in normal application sessions.
SET enable_seqscan = off;

EXPLAIN (ANALYZE, BUFFERS)
SELECT id, body
FROM turbohybrid_fast_start_docs
ORDER BY embedding <~> turbohybrid_query(
	vector_query => '[1,0,0]'::vector,
	text_query => websearch_to_tsquery('english', 'postgres vector hybrid search')
)
LIMIT 10;

SELECT id, body
FROM turbohybrid_fast_start_docs
ORDER BY embedding <~> turbohybrid_query(
	vector_query => '[1,0,0]'::vector,
	text_query => websearch_to_tsquery('english', 'postgres vector hybrid search')
)
LIMIT 10;

RESET enable_seqscan;

SELECT turbohybrid_last_scan_stats();
SELECT turbohybrid_index_stats('turbohybrid_fast_start_docs_idx'::regclass);
SELECT turbohybrid_simd_capabilities();

-- Cleanup:
-- DROP TABLE IF EXISTS turbohybrid_fast_start_docs;
