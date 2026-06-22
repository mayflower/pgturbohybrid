-- Index key map: type-based key discovery, cap lift, sparse recognition.
SET client_min_messages = warning;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;
RESET client_min_messages;

CREATE TABLE km (
  id int,
  v vector(4),
  tsv tsvector,
  s turbohybrid_sparse_vector
);
INSERT INTO km VALUES
  (1, '[1,0,0,0]', to_tsvector('simple','alpha beta'),
      turbohybrid_sparse_vector_build(ARRAY[1,2]::int4[], ARRAY[1.0,2.0]::float4[])),
  (2, '[0,1,0,0]', to_tsvector('simple','gamma delta'),
      turbohybrid_sparse_vector_build(ARRAY[2,3]::int4[], ARRAY[1.0,1.0]::float4[]));

-- accepted shapes (regression anchors): dense-only and dense+bm25 build + scan
CREATE INDEX km_dense ON km USING turbohybrid (v vector_cosine_turbohybrid_ops);
SET enable_seqscan = off;
SELECT id FROM km ORDER BY v <~> turbohybrid_query(vector_query => '[1,0,0,0]'::vector, dense_k => 10, final_k => 2) LIMIT 2;
RESET enable_seqscan;
DROP INDEX km_dense;

CREATE INDEX km_dense_bm25 ON km USING turbohybrid (
  v vector_cosine_turbohybrid_ops, tsv bm25_tsvector_turbohybrid_ops);
SET enable_seqscan = off;
SELECT id FROM km ORDER BY v <~> turbohybrid_query(
  vector_query => '[1,0,0,0]'::vector,
  text_query => websearch_to_tsquery('simple','alpha'),
  fusion => 'rrf', dense_k => 10, bm25_k => 10, final_k => 2) LIMIT 2;
RESET enable_seqscan;
DROP INDEX km_dense_bm25;

-- dense+sparse builds (dense-present sparse branch; see pgturbohybrid_sparse_scan)
CREATE INDEX km_dense_sparse ON km USING turbohybrid (v vector_cosine_turbohybrid_ops, s sparse_ip_turbohybrid_ops);
DROP INDEX km_dense_sparse;
-- sparse-only builds (sparse-primary node space owns node identity)
CREATE INDEX km_sparse_only ON km USING turbohybrid (s sparse_ip_turbohybrid_ops);
DROP INDEX km_sparse_only;
-- 3-key dense+sparse+bm25: the 2-key cap is lifted and all three branches build
CREATE INDEX km_dense_sparse_bm25 ON km USING turbohybrid (
  v vector_cosine_turbohybrid_ops, s sparse_ip_turbohybrid_ops, tsv bm25_tsvector_turbohybrid_ops);
DROP INDEX km_dense_sparse_bm25;
-- sparse+bm25 (no dense): sparse-primary node space also serves the bm25 branch
CREATE INDEX km_sparse_bm25 ON km USING turbohybrid (
  s sparse_ip_turbohybrid_ops, tsv bm25_tsvector_turbohybrid_ops);
DROP INDEX km_sparse_bm25;
-- bm25-only rejected: no vector/multivector graph or sparse key
CREATE INDEX ON km USING turbohybrid (tsv bm25_tsvector_turbohybrid_ops);
-- reordered: graph key must be the first index column
CREATE INDEX ON km USING turbohybrid (tsv bm25_tsvector_turbohybrid_ops, v vector_cosine_turbohybrid_ops);

-- duplicate dense keys rejected
CREATE TABLE km2 (id int, v1 vector(4), v2 vector(4));
CREATE INDEX ON km2 USING turbohybrid (
  v1 vector_cosine_turbohybrid_ops, v2 vector_cosine_turbohybrid_ops);
DROP TABLE km2;

-- duplicate bm25 keys rejected
CREATE TABLE km3 (id int, v vector(4), t1 tsvector, t2 tsvector);
CREATE INDEX ON km3 USING turbohybrid (
  v vector_cosine_turbohybrid_ops, t1 bm25_tsvector_turbohybrid_ops, t2 bm25_tsvector_turbohybrid_ops);
DROP TABLE km3;

-- duplicate sparse keys rejected (during key-map classification)
CREATE TABLE km4 (id int, v vector(4), s1 turbohybrid_sparse_vector, s2 turbohybrid_sparse_vector);
CREATE INDEX ON km4 USING turbohybrid (
  v vector_cosine_turbohybrid_ops, s1 sparse_ip_turbohybrid_ops, s2 sparse_ip_turbohybrid_ops);
DROP TABLE km4;

DROP TABLE km;
