-- sparse_query payload + <~*> exact scalar distance.
SET client_min_messages = warning;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;
RESET client_min_messages;

-- round-trip: sparse_query through turbohybrid_query output
SELECT turbohybrid_query(
  sparse_query => turbohybrid_sparse_vector_build(ARRAY[1,2,3]::int4[], ARRAY[1.0,2.0,3.0]::float4[]),
  sparse_weight => 2.0, sparse_k => 50)::text AS with_sparse;

-- defaults: sparse_k null -> "null", require_sparse_match false
SELECT turbohybrid_query(
  sparse_query => turbohybrid_sparse_vector_build(ARRAY[1]::int4[], ARRAY[1.0]::float4[]))::text AS sparse_defaults;

-- require_sparse_match round-trips
SELECT turbohybrid_query(
  sparse_query => turbohybrid_sparse_vector_build(ARRAY[1]::int4[], ARRAY[1.0]::float4[]),
  require_sparse_match => true)::text AS require_sparse;

-- no sparse_query -> sparse=false
SELECT turbohybrid_query(vector_query => '[1,2,3]'::vector)::text AS no_sparse;

-- exact scalar dot ordering (seqscan, no index): query term 2 weight 1.0
CREATE TABLE sp_docs (id int, s turbohybrid_sparse_vector);
INSERT INTO sp_docs VALUES
  (1, turbohybrid_sparse_vector_build(ARRAY[1,2]::int4[], ARRAY[1.0,1.0]::float4[])),
  (2, turbohybrid_sparse_vector_build(ARRAY[2,3]::int4[], ARRAY[5.0,1.0]::float4[])),
  (3, turbohybrid_sparse_vector_build(ARRAY[4]::int4[],   ARRAY[9.0]::float4[]));
SELECT id,
       round((s <~*> turbohybrid_query(
         sparse_query => turbohybrid_sparse_vector_build(ARRAY[2]::int4[], ARRAY[1.0]::float4[])))::numeric, 2) AS dist
FROM sp_docs
ORDER BY s <~*> turbohybrid_query(
  sparse_query => turbohybrid_sparse_vector_build(ARRAY[2]::int4[], ARRAY[1.0]::float4[])), id;

-- unsorted (from_arrays) doc vector scores identically (order-independent dot)
SELECT round((turbohybrid_sparse_vector_from_arrays(ARRAY[3,2,1]::int4[], ARRAY[1.0,5.0,1.0]::float4[])
       <~*> turbohybrid_query(
         sparse_query => turbohybrid_sparse_vector_build(ARRAY[2]::int4[], ARRAY[1.0]::float4[])))::numeric, 2) AS unsorted_dist;

-- invalid sparse_weight / sparse_k
SELECT turbohybrid_query(
  sparse_query => turbohybrid_sparse_vector_build(ARRAY[1]::int4[], ARRAY[1.0]::float4[]),
  sparse_weight => -1.0);
SELECT turbohybrid_query(
  sparse_query => turbohybrid_sparse_vector_build(ARRAY[1]::int4[], ARRAY[1.0]::float4[]),
  sparse_k => -5);

-- require_sparse_match without sparse_query is rejected
SELECT turbohybrid_query(require_sparse_match => true);

-- distance function on a query lacking sparse_query is rejected
SELECT turbohybrid_sparse_vector_build(ARRAY[1]::int4[], ARRAY[1.0]::float4[])
       <~*> turbohybrid_query(vector_query => '[1]'::vector);

DROP TABLE sp_docs;

-- A sparse-only index (no dense/multivector graph key) builds via the
-- sparse-primary node space: node identity comes from a dedicated
-- node-map chain.  Build/scan/DML are exercised in pgturbohybrid_sparse_primary;
-- here we only confirm the create path is accepted.
CREATE TABLE sp_idx (id int, s turbohybrid_sparse_vector);
CREATE INDEX ON sp_idx USING turbohybrid (s sparse_ip_turbohybrid_ops);
DROP TABLE sp_idx;
