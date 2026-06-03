-- residual_rerank_grid.sql
--
-- Decision benchmark: on exact-free (exact_storage=off) 4-bit indexes, is the
-- in-index calibrated residual rerank worth recommending versus the heap-band
-- exact rescore? Developer harness, NOT a publishable result; prints a report,
-- writes no files, changes NO defaults and NO runtime behavior.
--
-- Two ways to correct 4-bit quantization ranking error on an exact-free index:
--   * residual rerank  -- a few residual bytes stored IN the index
--     (residual_rerank=on, residual_rerank_bytes=N), applied at query time via
--     dense_residual_rerank_mode = fixed | calibrated. No heap fetch.
--   * heap-band rescore -- dense_heap_rescore = topk | band, which fetches the
--     exact vectors from the HEAP table to rescore. Exact, but pays heap I/O.
-- This grid measures recall@k and latency for both so you can decide whether the
-- cheaper residual rerank recovers enough recall to recommend over heap rescore.
--
-- Run:
--   createdb resrr ; psql -d resrr -f benchmarks/dev/residual_rerank_grid.sql ; dropdb resrr
-- psql variables (optional): DIMS (default 768), NROWS (default 5000), K (10), ITERS (60).
--
-- How to interpret / decision rule:
--   * Baseline = no rescore (recall ceiling set by the 4-bit scan).
--   * heap-band is the exact-rescore reference (upper bound on recall recovery).
--   * Recommend calibrated residual rerank when, on YOUR data, it recovers most
--     of the heap-band recall gain at materially lower p95 and acceptable index
--     size/memory growth. fixed vs calibrated shows whether calibration helps.
--     A single synthetic grid does not change any profile/index default.

\set ON_ERROR_STOP on
\pset pager off
CREATE EXTENSION IF NOT EXISTS pgturbohybrid CASCADE;
\if :{?DIMS}
\else
  \set DIMS 768
\endif
\if :{?NROWS}
\else
  \set NROWS 5000
\endif
\if :{?K}
\else
  \set K 10
\endif
\if :{?ITERS}
\else
  \set ITERS 60
\endif
\echo 'residual_rerank_grid: DIMS=' :DIMS ' NROWS=' :NROWS ' K=' :K ' ITERS=' :ITERS

-- Deterministic clustered corpus (50 clusters x ~100 rows) with moderate within-
-- cluster jitter, so a query's true top-k are present in the candidate pool but
-- the 4-bit scan mis-ranks the close members -> rescore (residual or heap) has
-- recall to recover. No random(): reproducible.
DROP TABLE IF EXISTS rr_docs CASCADE;
CREATE TABLE rr_docs (id int PRIMARY KEY, embedding vector(:DIMS) NOT NULL);
INSERT INTO rr_docs
SELECT g,
  (SELECT array_agg((sin((g % 50 + 1) * 0.29 + d * 0.011)
                   + cos((g % 50 + 1) * 0.15 + d * 0.006)
                   + 0.12 * sin(g * 0.37 + d * 0.041))::real ORDER BY d)
   FROM generate_series(1, :DIMS) d)::real[]::vector
FROM generate_series(1, :NROWS) g;
ANALYZE rr_docs;

-- Query = a member of cluster 7 (id 357); exact KNN ground truth via seqscan <=>.
DROP TABLE IF EXISTS rr_q CASCADE;
CREATE TABLE rr_q (name text PRIMARY KEY, qvec vector(:DIMS), exact_ids int[]);
INSERT INTO rr_q(name) VALUES ('dense');
UPDATE rr_q SET qvec=(SELECT embedding FROM rr_docs WHERE id=357);
UPDATE rr_q q SET exact_ids=(SELECT array_agg(id ORDER BY embedding <=> q.qvec)
  FROM (SELECT id, embedding FROM rr_docs ORDER BY embedding <=> q.qvec LIMIT :K) s);

DROP TABLE IF EXISTS rr_res CASCADE;
CREATE TABLE rr_res (arm text, index_variant text, mode text, recall_at_k float8, p50_ms float8, p95_ms float8,
  residual_band int, residual_reordered int, residual_topk_changed bool, residual_us bigint,
  heap_rescore_count int, heap_rescore_us bigint, index_bytes bigint, est_memory jsonb);

CREATE OR REPLACE FUNCTION pg_temp.ov(a int[],b int[]) RETURNS int LANGUAGE sql AS
$$ SELECT count(*)::int FROM (SELECT unnest(a) INTERSECT SELECT unnest(b)) s $$;

\echo '== running residual rerank vs heap rescore grid =='
DROP TABLE IF EXISTS rr_params;
CREATE TEMP TABLE rr_params AS SELECT :K::int AS p_k, :ITERS::int AS p_iters;
DO $$
DECLARE
  q record; ids int[]; st jsonb; sql text; qexpr text;
  k int; dk int := 200; iters int; t0 timestamptz; i int; lat float8[]; ov float8;
  ibytes bigint; emem jsonb;
  variant record; rmode text; hmode text;
  -- (variant_label, residual reloptions)
  variants text[][] := ARRAY[
    ARRAY['off', 'residual_rerank = off'],
    ARRAY['16',  'residual_rerank = on, residual_rerank_bytes = 16'],
    ARRAY['32',  'residual_rerank = on, residual_rerank_bytes = 32'],
    ARRAY['64',  'residual_rerank = on, residual_rerank_bytes = 64']];
  v text[];
BEGIN
  SELECT p_k, p_iters INTO k, iters FROM rr_params;
  PERFORM set_config('enable_seqscan','off',false); PERFORM set_config('jit','off',false);
  PERFORM set_config('turbohybrid.profile','matched_recall',false);
  SELECT * INTO q FROM rr_q WHERE name='dense';
  FOREACH v SLICE 1 IN ARRAY variants LOOP
    EXECUTE 'DROP INDEX IF EXISTS rr_idx';
    EXECUTE format('CREATE INDEX rr_idx ON rr_docs USING turbohybrid (embedding vector_cosine_turbohybrid_ops) WITH (quantization_bits = 4, exact_storage = off, native_segments = 1, %s)', v[2]);
    ANALYZE rr_docs;
    ibytes := pg_relation_size('rr_idx'); emem := turbohybrid_estimate_memory('rr_idx'::regclass);

    -- Residual-rerank arm: heap rescore off, sweep residual mode.
    PERFORM set_config('turbohybrid.dense_heap_rescore','off',false);
    FOREACH rmode IN ARRAY ARRAY['off','fixed','calibrated'] LOOP
      PERFORM set_config('turbohybrid.dense_residual_rerank_mode', rmode, false);
      qexpr := format('turbohybrid_query(vector_query=>%L::vector, dense_k=>%s, final_k=>%s)', q.qvec::text, dk, k);
      sql := format('SELECT array_agg(id) FROM (SELECT id FROM rr_docs ORDER BY embedding <~> %s LIMIT %s) s', qexpr, k);
      EXECUTE sql INTO ids; lat := '{}';
      FOR i IN 1..iters LOOP t0:=clock_timestamp(); EXECUTE sql INTO ids; lat := lat||(extract(epoch FROM clock_timestamp()-t0)*1000); END LOOP;
      st := turbohybrid_last_scan_stats();
      ov := pg_temp.ov(ids,(q.exact_ids)[1:k])::float8/k;
      INSERT INTO rr_res VALUES ('residual', v[1], rmode, ov,
        (SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY x) FROM unnest(lat) x),
        (SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY x) FROM unnest(lat) x),
        (st->>'residual_rerank_band')::int, (st->>'residual_rerank_reordered_count')::int,
        (st->>'residual_rerank_topk_changed')::bool, (st->>'dense_residual_rerank_us')::bigint,
        (st->>'heap_rescore_count')::int, (st->>'heap_rescore_us')::bigint, ibytes, emem);
    END LOOP;

    -- Heap-rescore arm (only on the residual=off index): residual off, sweep heap mode.
    IF v[1] = 'off' THEN
      PERFORM set_config('turbohybrid.dense_residual_rerank_mode','off',false);
      FOREACH hmode IN ARRAY ARRAY['off','topk','band'] LOOP
        PERFORM set_config('turbohybrid.dense_heap_rescore', hmode, false);
        qexpr := format('turbohybrid_query(vector_query=>%L::vector, dense_k=>%s, final_k=>%s)', q.qvec::text, dk, k);
        sql := format('SELECT array_agg(id) FROM (SELECT id FROM rr_docs ORDER BY embedding <~> %s LIMIT %s) s', qexpr, k);
        EXECUTE sql INTO ids; lat := '{}';
        FOR i IN 1..iters LOOP t0:=clock_timestamp(); EXECUTE sql INTO ids; lat := lat||(extract(epoch FROM clock_timestamp()-t0)*1000); END LOOP;
        st := turbohybrid_last_scan_stats();
        ov := pg_temp.ov(ids,(q.exact_ids)[1:k])::float8/k;
        INSERT INTO rr_res VALUES ('heap', v[1], hmode, ov,
          (SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY x) FROM unnest(lat) x),
          (SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY x) FROM unnest(lat) x),
          NULL, NULL, NULL, NULL,
          (st->>'heap_rescore_count')::int, (st->>'heap_rescore_us')::bigint, ibytes, emem);
      END LOOP;
      PERFORM set_config('turbohybrid.dense_heap_rescore','off',false);
    END IF;
  END LOOP;
END $$;

\echo '== recall@k + latency: residual rerank arm (heap rescore off) =='
SELECT index_variant AS residual_bytes, mode AS residual_mode,
  round(recall_at_k::numeric,3) AS recall, round(p50_ms::numeric,3) AS p50_ms, round(p95_ms::numeric,3) AS p95_ms,
  residual_band, residual_reordered AS reordered, residual_topk_changed AS topk_chg, residual_us
FROM rr_res WHERE arm='residual'
ORDER BY array_position(ARRAY['off','16','32','64'], index_variant),
         array_position(ARRAY['off','fixed','calibrated'], mode);

\echo '== recall@k + latency: heap rescore arm (residual off, on the residual=off index) =='
SELECT mode AS heap_rescore, round(recall_at_k::numeric,3) AS recall,
  round(p50_ms::numeric,3) AS p50_ms, round(p95_ms::numeric,3) AS p95_ms,
  heap_rescore_count AS heap_count, heap_rescore_us AS heap_us
FROM rr_res WHERE arm='heap'
ORDER BY array_position(ARRAY['off','topk','band'], mode);

\echo '== index size + memory estimate by residual_rerank_bytes =='
SELECT index_variant AS residual_bytes,
  pg_size_pretty(max(index_bytes)) AS index_size,
  max(est_memory->'native'->>'estimated_total_mb') AS est_total_mb,
  max(est_memory->'native'->>'residual_bytes') AS est_residual_bytes
FROM rr_res GROUP BY index_variant
ORDER BY array_position(ARRAY['off','16','32','64'], index_variant);

\echo '== DONE residual_rerank_grid =='
