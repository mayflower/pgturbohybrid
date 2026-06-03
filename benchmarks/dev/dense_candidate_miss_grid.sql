-- dense_candidate_miss_grid.sql
--
-- Decision harness that ATTRIBUTES a dense recall miss to its cause, so you know
-- which lever fixes it: widen graph search (ef_search/oversampling), residual
-- rerank, or heap-band rescore. Developer harness, NOT a publishable result;
-- prints a report, writes no files, changes NO defaults and NO runtime behavior.
--
-- For each query it computes the exact dense top-k (seqscan <=>) and asks:
--   1. candidate_pool_contains_exact_topk : are the exact neighbours even in the
--      raw graph candidate pool (top dense_k by 4-bit, no rescore)?  If not, the
--      graph search did not reach them -> widen ef_search / oversampling.
--   2. residual_band_contains_exact_topk  : are they in the narrow residual
--      rerank band (top ~2*final_k by 4-bit)?  If yes, residual rerank can fix it.
--   3. heap_band_contains_exact_topk      : does heap-band rescore (exact, over
--      the whole pool) recover them?
-- and prints a decision label per case.
--
-- Run:
--   createdb cmiss ; psql -d cmiss -f benchmarks/dev/dense_candidate_miss_grid.sql ; dropdb cmiss
-- psql variables (optional): DIMS (default 1536), NROWS (default 2500), K (10), ITERS (40).
--
-- How to read it / decision rule:
--   candidate_generation_miss     -> exact top-k not in pool; widen ef_search/oversampling.
--   residual_band_too_narrow      -> in pool, not in residual band; residual rerank won't help, use heap-band.
--   quantized_misorder_fixed_by_heap -> in pool and band, base recall<1; heap-band (or residual) re-ranks it in.
--   graph_search_sufficient       -> base recall already 1.0; no rescore needed.
--   payload_filter_underfilled    -> filtered candidate pool smaller than k.
--   The treatment sweep then confirms which lever actually recovers recall. As
--   with the other dev grids, no default changes from a synthetic run alone.

\set ON_ERROR_STOP on
\pset pager off
CREATE EXTENSION IF NOT EXISTS pgturbohybrid CASCADE;
\if :{?DIMS}
\else
  \set DIMS 1536
\endif
\if :{?NROWS}
\else
  \set NROWS 2500
\endif
\if :{?K}
\else
  \set K 10
\endif
\if :{?ITERS}
\else
  \set ITERS 40
\endif
\echo 'dense_candidate_miss_grid: DIMS=' :DIMS ' NROWS=' :NROWS ' K=' :K ' ITERS=' :ITERS

-- Deterministic corpus with three dense regimes + a payload column:
--   ids 1..300    region E (easy): a smooth 1-D-ish manifold -> the 4-bit order
--                 matches the exact order, recall saturates.
--   ids 301..360  region M (misorder): one MODERATE 60-doc cluster -> the exact
--                 top-k all land in the candidate pool/band but 4-bit mis-ranks
--                 them, so heap-band (or residual) re-ranks them in.
--   ids 361..N    region S (spread): many small clusters -> an out-of-distribution
--                 query's true neighbours are scattered.
-- category_id = id%5, but category 5 is rare (only ids%500=0) for the underfill case.
DROP TABLE IF EXISTS cm_docs CASCADE;
CREATE TABLE cm_docs (id int PRIMARY KEY, region text, embedding vector(:DIMS) NOT NULL, category_id int);
INSERT INTO cm_docs
SELECT g,
  CASE WHEN g<=300 THEN 'E' WHEN g<=360 THEN 'M' ELSE 'S' END,
  (SELECT array_agg((CASE
     WHEN g<=10  THEN 80.0*sin(d*0.5) + 0.02*sin(g*3.0 + d*0.7)                                   -- E: exactly-10-doc isolated micro-cluster (its members ARE the unambiguous top-10, no 11th nearby) -> recall saturates
     WHEN g<=300 THEN sin(g*0.020 + d*0.018) + 0.30*cos(d*0.004)                                  -- E filler manifold
     WHEN g<=360 THEN sin(2.0 + d*0.011) + cos(1.3 + d*0.007) + 0.08*sin(g*0.7 + d*0.05)          -- M: moderate 60-doc cluster
     ELSE sin((g%120+1)*0.41 + d*0.013) + cos((g%120+1)*0.23 + d*0.009) + 0.5*sin(g*0.05 + d*0.017) END)::real ORDER BY d) -- S: spread
   FROM generate_series(1,:DIMS) d)::real[]::vector,
  CASE WHEN g%500=0 THEN 5 ELSE g%5 END
FROM generate_series(1,:NROWS) g;
ANALYZE cm_docs;

-- Query cases + exact dense top-k (seqscan <=>, optionally category-filtered).
DROP TABLE IF EXISTS cm_q CASCADE;
CREATE TABLE cm_q (name text PRIMARY KEY, qvec vector(:DIMS), filter_cat int, exact_ids int[]);
INSERT INTO cm_q(name,filter_cat) VALUES ('easy',NULL),('hard_misorder',NULL),('hard_spread',NULL),('payload_filtered',5);
UPDATE cm_q SET qvec=(SELECT embedding FROM cm_docs WHERE id=5) WHERE name='easy';             -- region E isolated micro-cluster (ids 1..10) -> top-10 unambiguous
UPDATE cm_q SET qvec=(SELECT embedding FROM cm_docs WHERE id=330) WHERE name='hard_misorder';  -- region M (moderate cluster)
UPDATE cm_q SET qvec=(SELECT a.embedding+b.embedding+c.embedding FROM cm_docs a JOIN cm_docs b ON b.id=1700 JOIN cm_docs c ON c.id=2300 WHERE a.id=900) WHERE name='hard_spread';  -- OOD across S
UPDATE cm_q SET qvec=(SELECT embedding FROM cm_docs WHERE id=330) WHERE name='payload_filtered';
UPDATE cm_q q SET exact_ids=(SELECT array_agg(id ORDER BY embedding <=> q.qvec)
  FROM (SELECT id,embedding FROM cm_docs d WHERE q.filter_cat IS NULL OR d.category_id=q.filter_cat
        ORDER BY embedding <=> q.qvec LIMIT :K) s);

DROP TABLE IF EXISTS cm_diag CASCADE;
CREATE TABLE cm_diag (case_name text, exact_topk int, base_recall float8, heap_band_recall float8, wide_ef_recall float8,
  candidate_pool_contains bool, residual_band_contains bool, heap_band_contains bool, decision_label text);
DROP TABLE IF EXISTS cm_sweep CASCADE;
CREATE TABLE cm_sweep (case_name text, treatment text, value text, recall_at_k float8, pool_contains bool,
  graph_candidate_count int, residual_band int, heap_rescore_count int, graph_visited int, graph_scored int,
  graph_total_us int, graph_batch_us int, heap_rescore_us bigint, residual_rerank_us bigint, p50_ms float8, p95_ms float8);

CREATE OR REPLACE FUNCTION pg_temp.ov(a int[],b int[]) RETURNS int LANGUAGE sql AS
$$ SELECT count(*)::int FROM (SELECT unnest(a) INTERSECT SELECT unnest(b)) s $$;
-- result ids for the CURRENT index + current GUCs, at a given final_k.
CREATE OR REPLACE FUNCTION pg_temp.qids(p_qvec vector, p_filter int, p_dense_k int, p_final_k int) RETURNS int[]
LANGUAGE plpgsql AS $$
DECLARE res int[];
BEGIN
  EXECUTE format('SELECT array_agg(id) FROM (SELECT d.id FROM cm_docs d WHERE %s ORDER BY d.embedding <~> turbohybrid_query(vector_query=>%L::vector, dense_k=>%s, final_k=>%s) LIMIT %s) s',
    CASE WHEN p_filter IS NULL THEN 'TRUE' ELSE format('d.category_id=%s',p_filter) END, p_qvec::text, p_dense_k, p_final_k, p_final_k) INTO res;
  RETURN res;
END $$;

\echo '== building indexes + running cases =='
DROP TABLE IF EXISTS cm_params;
CREATE TEMP TABLE cm_params AS SELECT :K::int AS p_k, :ITERS::int AS p_iters, :NROWS::int AS p_n;
DO $$
DECLARE
  q record; cfg record; ids int[]; pool int[]; band int[]; st jsonb; sql text; qexpr text;
  k int; iters int; nrows int; dk int := 200; t0 timestamptz; i int; lat float8[]; band_w int;
  poolc bool; recall float8;
  -- index configs: (treatment, value, reloptions-or-PROFILE)
  cfgs text[][] := ARRAY[
    ARRAY['profile','latency','PROFILE:latency'], ARRAY['profile','matched_recall','PROFILE:matched_recall'],
    ARRAY['profile','high_recall','PROFILE:high_recall'], ARRAY['profile','quality','PROFILE:quality'],
    ARRAY['ef_search','64','graph_ef_search=64, graph_oversampling=4'],
    ARRAY['ef_search','96','graph_ef_search=96, graph_oversampling=4'],
    ARRAY['ef_search','128','graph_ef_search=128, graph_oversampling=4'],
    ARRAY['ef_search','192','graph_ef_search=192, graph_oversampling=4'],
    ARRAY['ef_search','256','graph_ef_search=256, graph_oversampling=4'],
    ARRAY['ef_search','384','graph_ef_search=384, graph_oversampling=4'],
    ARRAY['oversampling','8','graph_ef_search=128, graph_oversampling=8'],
    ARRAY['oversampling','12','graph_ef_search=128, graph_oversampling=12'],
    ARRAY['oversampling','16','graph_ef_search=128, graph_oversampling=16'],
    ARRAY['base','residual32','graph_ef_search=128, graph_oversampling=4, residual_rerank=on, residual_rerank_bytes=32']];
  c text[]; mode text;
BEGIN
  SELECT p_k, p_iters, p_n INTO k, iters, nrows FROM cm_params;
  band_w := 2*k;  -- residual rerank band approx (top ~2*final_k)
  PERFORM set_config('enable_seqscan','off',false); PERFORM set_config('jit','off',false);

  FOREACH c SLICE 1 IN ARRAY cfgs LOOP
    EXECUTE 'DROP INDEX IF EXISTS cm_idx';
    IF c[3] LIKE 'PROFILE:%' THEN
      PERFORM set_config('turbohybrid.profile', split_part(c[3],':',2), false);
      EXECUTE 'CREATE INDEX cm_idx ON cm_docs USING turbohybrid (embedding vector_cosine_turbohybrid_ops)';
    ELSE
      PERFORM set_config('turbohybrid.profile','matched_recall',false);
      EXECUTE format('CREATE INDEX cm_idx ON cm_docs USING turbohybrid (embedding vector_cosine_turbohybrid_ops) WITH (quantization_bits=4, exact_storage=off, native_segments=1, %s)', c[3]);
    END IF;
    ANALYZE cm_docs;

    FOR q IN SELECT * FROM cm_q ORDER BY name LOOP
      -- raw 4-bit candidate pool (no rescore): does it contain the exact top-k?
      PERFORM set_config('turbohybrid.dense_residual_rerank_mode','off',false);
      PERFORM set_config('turbohybrid.dense_heap_rescore','off',false);
      pool := pg_temp.qids(q.qvec, q.filter_cat, dk, LEAST(dk, GREATEST(k, 200)));
      poolc := (SELECT bool_and(e = ANY(pool)) FROM unnest((q.exact_ids)[1:k]) e);

      -- measured recall at final_k=k (default rescore off) + timing + stats
      sql := format('SELECT array_agg(id) FROM (SELECT d.id FROM cm_docs d WHERE %s ORDER BY d.embedding <~> turbohybrid_query(vector_query=>%L::vector, dense_k=>%s, final_k=>%s) LIMIT %s) s',
        CASE WHEN q.filter_cat IS NULL THEN 'TRUE' ELSE format('d.category_id=%s',q.filter_cat) END, q.qvec::text, dk, k, k);
      EXECUTE sql INTO ids; lat := '{}';
      FOR i IN 1..iters LOOP t0:=clock_timestamp(); EXECUTE sql INTO ids; lat := lat||(extract(epoch FROM clock_timestamp()-t0)*1000); END LOOP;
      st := turbohybrid_last_scan_stats();
      recall := pg_temp.ov(ids,(q.exact_ids)[1:k])::float8/GREATEST(1,COALESCE(array_length((q.exact_ids)[1:k],1),0));
      INSERT INTO cm_sweep VALUES (q.name, c[1], c[2], recall, poolc,
        (st->>'graph_candidate_count')::int, (st->>'residual_rerank_band')::int, (st->>'heap_rescore_count')::int,
        (st->>'graph_visited_nodes')::int, (st->>'graph_scored_codes')::int,
        (st->>'graph_total_us')::int, (st->>'graph_batch_us')::int,
        (st->>'heap_rescore_us')::bigint, (st->>'dense_residual_rerank_us')::bigint,
        (SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY x) FROM unnest(lat) x),
        (SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY x) FROM unnest(lat) x));

      -- On the base residual index: residual + heap sub-sweeps and the per-case diagnosis.
      IF c[1]='base' THEN
        FOREACH mode IN ARRAY ARRAY['fixed','calibrated'] LOOP
          PERFORM set_config('turbohybrid.dense_residual_rerank_mode',mode,false);
          PERFORM set_config('turbohybrid.dense_heap_rescore','off',false);
          ids := pg_temp.qids(q.qvec,q.filter_cat,dk,k); st := turbohybrid_last_scan_stats();
          INSERT INTO cm_sweep VALUES (q.name,'residual',mode, pg_temp.ov(ids,(q.exact_ids)[1:k])::float8/GREATEST(1,COALESCE(array_length((q.exact_ids)[1:k],1),0)), poolc,
            (st->>'graph_candidate_count')::int,(st->>'residual_rerank_band')::int,(st->>'heap_rescore_count')::int,
            (st->>'graph_visited_nodes')::int,(st->>'graph_scored_codes')::int,(st->>'graph_total_us')::int,(st->>'graph_batch_us')::int,
            (st->>'heap_rescore_us')::bigint,(st->>'dense_residual_rerank_us')::bigint, NULL, NULL);
        END LOOP;
        PERFORM set_config('turbohybrid.dense_residual_rerank_mode','off',false);
        FOREACH mode IN ARRAY ARRAY['topk','band'] LOOP
          PERFORM set_config('turbohybrid.dense_heap_rescore',mode,false);
          ids := pg_temp.qids(q.qvec,q.filter_cat,dk,k); st := turbohybrid_last_scan_stats();
          INSERT INTO cm_sweep VALUES (q.name,'heap',mode, pg_temp.ov(ids,(q.exact_ids)[1:k])::float8/GREATEST(1,COALESCE(array_length((q.exact_ids)[1:k],1),0)), poolc,
            (st->>'graph_candidate_count')::int,(st->>'residual_rerank_band')::int,(st->>'heap_rescore_count')::int,
            (st->>'graph_visited_nodes')::int,(st->>'graph_scored_codes')::int,(st->>'graph_total_us')::int,(st->>'graph_batch_us')::int,
            (st->>'heap_rescore_us')::bigint,(st->>'dense_residual_rerank_us')::bigint, NULL, NULL);
        END LOOP;
        PERFORM set_config('turbohybrid.dense_heap_rescore','off',false);

        -- diagnosis: containment + label (base index)
        DECLARE base_recall float8; heap_recall float8; band_contains bool; heap_contains bool; cnt int; lbl text;
        BEGIN
          cnt := COALESCE(array_length((q.exact_ids)[1:k],1),0);
          band := pg_temp.qids(q.qvec,q.filter_cat,dk,band_w);
          band_contains := (SELECT bool_and(e = ANY(band)) FROM unnest((q.exact_ids)[1:k]) e);
          ids := pg_temp.qids(q.qvec,q.filter_cat,dk,k);
          base_recall := pg_temp.ov(ids,(q.exact_ids)[1:k])::float8/GREATEST(1,cnt);
          PERFORM set_config('turbohybrid.dense_heap_rescore','band',false);
          ids := pg_temp.qids(q.qvec,q.filter_cat,dk,k);
          heap_recall := pg_temp.ov(ids,(q.exact_ids)[1:k])::float8/GREATEST(1,cnt);
          heap_contains := (heap_recall >= 0.999);
          PERFORM set_config('turbohybrid.dense_heap_rescore','off',false);
          lbl := CASE
            WHEN q.filter_cat IS NOT NULL AND cnt < k THEN 'payload_filter_underfilled'
            WHEN base_recall >= 0.999 THEN 'graph_search_sufficient'
            WHEN NOT poolc THEN 'candidate_generation_miss'
            WHEN NOT band_contains THEN 'residual_band_too_narrow'
            ELSE 'quantized_misorder_fixed_by_heap' END;
          INSERT INTO cm_diag VALUES (q.name, cnt, round(base_recall::numeric,3), round(heap_recall::numeric,3),
            NULL, poolc, band_contains, heap_contains, lbl);
        END;
      END IF;
    END LOOP;
  END LOOP;

  -- fill wide_ef_recall in diagnosis from the ef_search=384 sweep rows
  UPDATE cm_diag d SET wide_ef_recall = (SELECT round(recall_at_k::numeric,3) FROM cm_sweep s
     WHERE s.case_name=d.case_name AND s.treatment='ef_search' AND s.value='384');
END $$;

\echo '== DIAGNOSIS: where does each case miss, and the decision label =='
SELECT case_name, exact_topk, base_recall, heap_band_recall, wide_ef_recall,
  candidate_pool_contains AS pool_has_topk, residual_band_contains AS band_has_topk, heap_band_contains AS heap_recovers,
  decision_label
FROM cm_diag ORDER BY array_position(ARRAY['easy','hard_misorder','hard_spread','payload_filtered'], case_name);

\echo '== TREATMENT SWEEP: recall@k by case x treatment (which lever recovers?) =='
SELECT case_name, treatment, value, round(recall_at_k::numeric,3) AS recall, pool_contains AS pool_has_topk,
  graph_candidate_count AS cands, residual_band AS res_band, heap_rescore_count AS heap_cnt,
  graph_visited AS visited, graph_scored AS scored,
  round(p50_ms::numeric,3) AS p50_ms, round(p95_ms::numeric,3) AS p95_ms
FROM cm_sweep
ORDER BY array_position(ARRAY['easy','hard_misorder','hard_spread','payload_filtered'], case_name),
  array_position(ARRAY['profile','ef_search','oversampling','base','residual','heap'], treatment),
  value;

\echo '== timing detail (us): graph_total/batch, heap_rescore_us, residual_rerank_us =='
SELECT case_name, treatment, value, graph_total_us, graph_batch_us, heap_rescore_us, residual_rerank_us
FROM cm_sweep WHERE treatment IN ('base','heap','residual','ef_search')
ORDER BY array_position(ARRAY['easy','hard_misorder','hard_spread','payload_filtered'], case_name), treatment, value;

\echo '== DONE dense_candidate_miss_grid =='
