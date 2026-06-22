-- Native dense+sparse index build and sparse-as-sole-ORDER-BY scan:
-- exact float32 OR-accumulation over the inverted index, ranked by inner
-- product, with MVCC node liveness.
SET client_min_messages = warning;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;
RESET client_min_messages;

CREATE TABLE sp_scan (id int, embedding vector(4), s turbohybrid_sparse_vector);
INSERT INTO sp_scan VALUES
  (1, '[1,0,0,0]', turbohybrid_sparse_vector_build(ARRAY[1,2]::int4[], ARRAY[1.0,1.0]::float4[])),
  (2, '[0,1,0,0]', turbohybrid_sparse_vector_build(ARRAY[2,3]::int4[], ARRAY[5.0,1.0]::float4[])),
  (3, '[0,0,1,0]', turbohybrid_sparse_vector_build(ARRAY[4]::int4[],   ARRAY[9.0]::float4[])),
  (4, '[0,0,0,1]', turbohybrid_sparse_vector_build(ARRAY[2,5]::int4[], ARRAY[2.0,3.0]::float4[]));

-- Dense+sparse index now builds (the dense-present gate is lifted).
-- Pin exact f32 postings (sparse_quant_bits=0) so the distances below are exact;
-- quantized (q8/q16) postings are exercised in pgturbohybrid_sparse_quant.
CREATE INDEX sp_scan_idx ON sp_scan
  USING turbohybrid (embedding vector_cosine_turbohybrid_ops, s sparse_ip_turbohybrid_ops)
  WITH (sparse_quant_bits = 0);

SET enable_seqscan = off;

-- The planner uses the turbohybrid index for a sparse <~*> ORDER BY.  (The plan
-- text is checked via LIKE so it is not sensitive to turbohybrid_query deparse.)
DO $$
DECLARE
  line text;
  plan text := '';
BEGIN
  SET LOCAL enable_seqscan = off;
  FOR line IN
    EXECUTE 'EXPLAIN (COSTS OFF) SELECT id FROM sp_scan ORDER BY s <~*> '
            'turbohybrid_query(sparse_query => '
            'turbohybrid_sparse_vector_build(ARRAY[2]::int4[], ARRAY[1.0]::float4[])) LIMIT 10'
  LOOP
    plan := plan || line || E'\n';
  END LOOP;
  IF plan NOT LIKE '%Index Scan using sp_scan_idx%' THEN
    RAISE EXCEPTION 'sparse ORDER BY did not use the index:%', E'\n' || plan;
  END IF;
END $$;

-- Query term {2:1.0}: doc2 dot=5, doc4 dot=2, doc1 dot=1; doc3 (no matching
-- term) is not a candidate.  Index returns matching docs by descending IP.
SELECT id, round((s <~*> turbohybrid_query(
  sparse_query => turbohybrid_sparse_vector_build(ARRAY[2]::int4[], ARRAY[1.0]::float4[])))::numeric, 2) AS dist
FROM sp_scan
ORDER BY s <~*> turbohybrid_query(
  sparse_query => turbohybrid_sparse_vector_build(ARRAY[2]::int4[], ARRAY[1.0]::float4[]))
LIMIT 10;

-- Two-term query {2:1.0, 5:2.0}: doc4 dot=2*1+3*2=8, doc2 dot=5, doc1 dot=1.
SELECT id, round((s <~*> turbohybrid_query(
  sparse_query => turbohybrid_sparse_vector_build(ARRAY[2,5]::int4[], ARRAY[1.0,2.0]::float4[])))::numeric, 2) AS dist
FROM sp_scan
ORDER BY s <~*> turbohybrid_query(
  sparse_query => turbohybrid_sparse_vector_build(ARRAY[2,5]::int4[], ARRAY[1.0,2.0]::float4[]))
LIMIT 10;

-- Single rare term {3:2.0}: only doc2 matches (dot=2).
SELECT id, round((s <~*> turbohybrid_query(
  sparse_query => turbohybrid_sparse_vector_build(ARRAY[3]::int4[], ARRAY[2.0]::float4[])))::numeric, 2) AS dist
FROM sp_scan
ORDER BY s <~*> turbohybrid_query(
  sparse_query => turbohybrid_sparse_vector_build(ARRAY[3]::int4[], ARRAY[2.0]::float4[]))
LIMIT 10;

-- Unknown query term {99:1.0}: no postings -> empty result.
SELECT id FROM sp_scan
ORDER BY s <~*> turbohybrid_query(
  sparse_query => turbohybrid_sparse_vector_build(ARRAY[99]::int4[], ARRAY[1.0]::float4[]))
LIMIT 10;

-- The index branch agrees with the brute-force seqscan order over matching docs.
DO $$
DECLARE
  idx_ids int[];
  seq_ids int[];
BEGIN
  SET LOCAL enable_seqscan = off;
  SELECT array_agg(id) INTO idx_ids FROM (
    SELECT id FROM sp_scan
    ORDER BY s <~*> turbohybrid_query(
      sparse_query => turbohybrid_sparse_vector_build(ARRAY[2,5]::int4[], ARRAY[1.0,2.0]::float4[]))
    LIMIT 10) q;

  SET LOCAL enable_seqscan = on;
  SELECT array_agg(id) INTO seq_ids FROM (
    SELECT id FROM sp_scan
    WHERE (s <~*> turbohybrid_query(
             sparse_query => turbohybrid_sparse_vector_build(ARRAY[2,5]::int4[], ARRAY[1.0,2.0]::float4[]))) < 0
    ORDER BY s <~*> turbohybrid_query(
      sparse_query => turbohybrid_sparse_vector_build(ARRAY[2,5]::int4[], ARRAY[1.0,2.0]::float4[]))) q;

  IF idx_ids IS DISTINCT FROM seq_ids THEN
    RAISE EXCEPTION 'index order % disagrees with seqscan order %', idx_ids, seq_ids;
  END IF;
END $$;

-- Scan stats report the sparse branch (available, used, resolved terms).
DO $$
DECLARE
  st jsonb;
BEGIN
  SET LOCAL enable_seqscan = off;
  PERFORM id FROM sp_scan
    ORDER BY s <~*> turbohybrid_query(
      sparse_query => turbohybrid_sparse_vector_build(ARRAY[2,3]::int4[], ARRAY[1.0,1.0]::float4[]))
    LIMIT 10;
  st := turbohybrid_last_scan_stats();
  IF st->>'sparse_branch_available' != 'true' THEN
    RAISE EXCEPTION 'sparse_branch_available not true: %', st;
  END IF;
  IF st->>'sparse_branch_used' != 'true' THEN
    RAISE EXCEPTION 'sparse_branch_used not true: %', st;
  END IF;
  IF st->>'sparse_terms' != '2' THEN
    RAISE EXCEPTION 'unexpected sparse_terms: %', st;
  END IF;
  IF st->>'sparse_resolved_terms' != '2' THEN
    RAISE EXCEPTION 'unexpected sparse_resolved_terms: %', st;
  END IF;
  IF (st->>'sparse_candidates_scored')::int < 1 THEN
    RAISE EXCEPTION 'unexpected sparse_candidates_scored: %', st;
  END IF;
  IF st->>'index_shape' != 'sparse' THEN
    RAISE EXCEPTION 'unexpected index_shape: %', st;
  END IF;
END $$;

-- A query term present in the lexicon but absent from the query resolves to 0.
DO $$
DECLARE
  st jsonb;
BEGIN
  SET LOCAL enable_seqscan = off;
  PERFORM id FROM sp_scan
    ORDER BY s <~*> turbohybrid_query(
      sparse_query => turbohybrid_sparse_vector_build(ARRAY[2,99]::int4[], ARRAY[1.0,1.0]::float4[]))
    LIMIT 10;
  st := turbohybrid_last_scan_stats();
  IF st->>'sparse_resolved_terms' != '1' THEN
    RAISE EXCEPTION 'expected one resolved term (term 99 unknown): %', st;
  END IF;
END $$;

RESET enable_seqscan;
DROP TABLE sp_scan;
