-- Backend-local sparse cache + memory estimator (Prompt 10).
SET client_min_messages = warning;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;
RESET client_min_messages;

CREATE TABLE scache (id int, embedding vector(3), s turbohybrid_sparse_vector);
INSERT INTO scache
SELECT g, '[1,0,0]', turbohybrid_sparse_vector_build(ARRAY[1]::int4[], ARRAY[g::float4])
FROM generate_series(1, 60) g;
CREATE INDEX scache_idx ON scache USING turbohybrid
  (embedding vector_cosine_turbohybrid_ops, s sparse_ip_turbohybrid_ops)
  WITH (sparse_quant_bits = 8, sparse_block_size = 8);

-- Estimator: a sparse index reports sparse.* byte fields and a positive total.
SELECT turbohybrid_estimate_memory('scache_idx'::regclass)->'sparse'->>'available' AS sparse_available,
       (turbohybrid_estimate_memory('scache_idx'::regclass)->'sparse'->>'sparse_total_bytes_per_backend')::bigint > 0 AS total_positive,
       (turbohybrid_estimate_memory('scache_idx'::regclass)->'sparse'->>'sparse_lexicon_bytes')::bigint > 0 AS lexicon_positive,
       (turbohybrid_estimate_memory('scache_idx'::regclass)->'concurrency'->>'sparse_total_bytes_per_backend')::bigint > 0 AS concurrency_total;

-- A dense-only index reports sparse unavailable.
CREATE TABLE donly (id int, embedding vector(3));
INSERT INTO donly SELECT g, '[1,0,0]' FROM generate_series(1, 5) g;
CREATE INDEX donly_idx ON donly USING turbohybrid (embedding vector_cosine_turbohybrid_ops);
SELECT turbohybrid_estimate_memory('donly_idx'::regclass)->'sparse'->>'available' AS dense_sparse_available;
DROP TABLE donly;

SET enable_seqscan = off;
SET turbohybrid.sparse_hot_postings_cache_min_df = 4;

-- Reader cache: first query builds it (miss), the second hits; results identical.
DO $$
DECLARE
  ids1 int[];
  ids2 int[];
  st jsonb;
BEGIN
  SET LOCAL enable_seqscan = off;
  SELECT array_agg(id) INTO ids1 FROM (
    SELECT id FROM scache ORDER BY s <~*> turbohybrid_query(
      sparse_query => turbohybrid_sparse_vector_build(ARRAY[1]::int4[], ARRAY[1.0]::float4[]),
      sparse_k => 5) LIMIT 5) q;
  st := turbohybrid_last_scan_stats();
  IF st->>'sparse_cache_hit' != 'false' THEN
    RAISE EXCEPTION 'first query should build the cache (miss): %', st->>'sparse_cache_hit';
  END IF;

  SELECT array_agg(id) INTO ids2 FROM (
    SELECT id FROM scache ORDER BY s <~*> turbohybrid_query(
      sparse_query => turbohybrid_sparse_vector_build(ARRAY[1]::int4[], ARRAY[1.0]::float4[]),
      sparse_k => 5) LIMIT 5) q;
  st := turbohybrid_last_scan_stats();
  IF st->>'sparse_cache_hit' != 'true' THEN
    RAISE EXCEPTION 'second query should hit the reader cache: %', st->>'sparse_cache_hit';
  END IF;
  IF (st->>'sparse_cache_bytes')::bigint <= 0 THEN
    RAISE EXCEPTION 'cache bytes should be positive: %', st;
  END IF;
  IF ids1 IS DISTINCT FROM ids2 THEN
    RAISE EXCEPTION 'cache changed results: % vs %', ids1, ids2;
  END IF;
END $$;

-- REINDEX (new relfilenumber) invalidates the cache: the next query rebuilds.
REINDEX INDEX scache_idx;
DO $$
DECLARE st jsonb;
BEGIN
  SET LOCAL enable_seqscan = off;
  PERFORM id FROM scache ORDER BY s <~*> turbohybrid_query(
    sparse_query => turbohybrid_sparse_vector_build(ARRAY[1]::int4[], ARRAY[1.0]::float4[]),
    sparse_k => 5) LIMIT 5;
  st := turbohybrid_last_scan_stats();
  IF st->>'sparse_cache_hit' != 'false' THEN
    RAISE EXCEPTION 'REINDEX should invalidate the cache: %', st->>'sparse_cache_hit';
  END IF;
END $$;

-- Hot-postings cache: repeated WAND queries on a high-df term hit the cache and
-- stay within the size cap.
DO $$
DECLARE st jsonb;
BEGIN
  SET LOCAL enable_seqscan = off;
  SET LOCAL turbohybrid.sparse_hot_postings_cache_min_df = 4;
  SET LOCAL turbohybrid.sparse_hot_postings_cache_mb = 16;
  -- prime
  PERFORM id FROM scache ORDER BY s <~*> turbohybrid_query(
    sparse_query => turbohybrid_sparse_vector_build(ARRAY[1]::int4[], ARRAY[1.0]::float4[])) LIMIT 5;
  -- repeat: should hit the hot cache
  PERFORM id FROM scache ORDER BY s <~*> turbohybrid_query(
    sparse_query => turbohybrid_sparse_vector_build(ARRAY[1]::int4[], ARRAY[1.0]::float4[])) LIMIT 5;
  st := turbohybrid_last_scan_stats();
  IF (st->>'sparse_hot_postings_cache_hits')::int < 1 THEN
    RAISE EXCEPTION 'expected hot-postings cache hits on repeat: %', st;
  END IF;
  IF (st->>'sparse_hot_postings_cache_bytes')::bigint > 16::bigint * 1024 * 1024 THEN
    RAISE EXCEPTION 'hot cache exceeded its size cap: %', st;
  END IF;
END $$;

-- Hot-postings cache disabled (mb=0): the hot cache is not consulted (no hits),
-- even though entries cached by an earlier mb>0 query may remain resident.
DO $$
DECLARE st jsonb;
BEGIN
  SET LOCAL enable_seqscan = off;
  SET LOCAL turbohybrid.sparse_hot_postings_cache_mb = 0;
  PERFORM id FROM scache ORDER BY s <~*> turbohybrid_query(
    sparse_query => turbohybrid_sparse_vector_build(ARRAY[1]::int4[], ARRAY[1.0]::float4[])) LIMIT 5;
  st := turbohybrid_last_scan_stats();
  IF (st->>'sparse_hot_postings_cache_hits')::int != 0 THEN
    RAISE EXCEPTION 'mb=0 should disable hot-cache serving: %', st;
  END IF;
END $$;

RESET turbohybrid.sparse_hot_postings_cache_mb;
RESET turbohybrid.sparse_hot_postings_cache_min_df;
RESET enable_seqscan;
DROP TABLE scache;
