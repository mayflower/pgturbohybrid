-- bm25_phrase_rerank_grid.sql
--
-- Decision benchmark: does turbohybrid.bm25_heap_tsvector_rerank help enough on
-- phrase/proximity queries to justify ever making `auto` a quality-profile
-- default? It is a developer harness, NOT a publishable result; it prints a
-- report and leaves no files. It changes NO profile defaults and NO runtime
-- behavior (only SETs GUCs for the session and builds throwaway objects).
--
-- The heap-tsvector rerank rescans the top BM25 candidates with ts_rank_cd
-- (cover density / proximity-aware) and blends it into the BM25 score. It should
-- help when the BM25 (bag-of-words, tf-driven) ordering puts proximity-wrong
-- docs above proximity-correct ones. This corpus is built to create exactly that
-- tension so the rerank has something to fix.
--
-- Run:
--   createdb bm25_phrase ; psql -d bm25_phrase -f benchmarks/dev/bm25_phrase_rerank_grid.sql ; dropdb bm25_phrase
--
-- psql variables (optional): DIMS (default 768), NROWS (default 3000), K (default 10), ITERS (default 60).
--
-- How to interpret (and the decision rule):
--   * overlap@k is measured against the EXACT-PHRASE docs (the proximity-correct
--     answer). Compare off vs topk/band/auto per query/fusion.
--   * `auto` only engages when the tsquery has a phrase operator (phraseto_tsquery
--     / <-> / <N>); on a bag query it is a no-op (rerank_count=0) by design.
--   * topk_changed=true means the rerank actually reordered the final top-k.
--   * Recommend enabling bm25_heap_tsvector_rerank=auto in the quality profile
--     ONLY if, on real phrase/proximity workloads, it raises overlap@k materially
--     while p95 stays acceptable AND rrf/calibrated both benefit. A single
--     synthetic grid is not sufficient to change a shipped profile default.

\set ON_ERROR_STOP on
\pset pager off
CREATE EXTENSION IF NOT EXISTS pgturbohybrid CASCADE;
\if :{?DIMS}
\else
  \set DIMS 768
\endif
\if :{?NROWS}
\else
  \set NROWS 3000
\endif
\if :{?K}
\else
  \set K 10
\endif
\if :{?ITERS}
\else
  \set ITERS 60
\endif
\echo 'bm25_phrase_rerank_grid: DIMS=' :DIMS ' NROWS=' :NROWS ' K=' :K ' ITERS=' :ITERS

-- Deterministic corpus. The first 50 ids share one dense cluster (semantically
-- similar) and carry the phrase terms in different shapes; the rest are dense-
-- spread distractors without the phrase terms. No random(): reproducible.
--   ids 1..10  exact_phrase     : "alpha beta gamma" adjacent (EXPECTED answer)
--   ids 11..20 reordered        : same terms, shuffled order
--   ids 21..30 far_apart        : terms separated by fillers
--   ids 31..40 high_tf_scattered: terms repeated (high BM25 tf), poor proximity
--   ids 41..50 rare_identifier  : a unique rareid<n>, no phrase terms
--   ids 51..   common distractor: 'common corpus' + cluster/filler only
DROP TABLE IF EXISTS bpr_docs CASCADE;
CREATE TABLE bpr_docs (id int PRIMARY KEY, doc_kind text, embedding vector(:DIMS) NOT NULL,
                      body text NOT NULL, body_tsv tsvector NOT NULL, group_id int, category_id int);
INSERT INTO bpr_docs
SELECT g,
  CASE WHEN g BETWEEN 1 AND 10 THEN 'exact_phrase'
       WHEN g BETWEEN 11 AND 20 THEN 'reordered'
       WHEN g BETWEEN 21 AND 30 THEN 'far_apart'
       WHEN g BETWEEN 31 AND 40 THEN 'high_tf_scattered'
       WHEN g BETWEEN 41 AND 50 THEN 'rare_identifier'
       ELSE 'distractor' END,
  -- ids 1..50 share ONE identical vector (the query vector): the dense branch
  -- cannot order them, so the BM25 ranking -- and therefore the heap-tsvector
  -- rerank -- determines the fused top-k. This is what gives the rerank a fair
  -- chance to change results; distractors are dense-spread elsewhere.
  (SELECT array_agg((CASE WHEN g<=50
       THEN sin(5.0 + d*0.012) + cos(3.0 + d*0.006)
       ELSE sin((g%37+1)*0.31 + d*0.012) + cos((g%37+1)*0.17 + d*0.006) + 0.4*sin(g*0.05 + d*0.013) END)::real ORDER BY d)
   FROM generate_series(1,:DIMS) d)::real[]::vector,
  CASE
    WHEN g BETWEEN 1 AND 10  THEN 'common corpus clustera alpha beta gamma tail'
    WHEN g BETWEEN 11 AND 20 THEN 'common corpus clustera gamma alpha beta tail'
    WHEN g BETWEEN 21 AND 30 THEN 'common corpus clustera alpha f1 f2 f3 f4 f5 f6 f7 f8 f9 beta f10 f11 f12 f13 f14 f15 f16 f17 f18 gamma'
    WHEN g BETWEEN 31 AND 40 THEN 'common corpus clustera alpha gamma beta alpha beta gamma beta alpha gamma'
    WHEN g BETWEEN 41 AND 50 THEN 'common corpus clustera rareid'||g
    ELSE 'common corpus cluster'||(g%37)||' filler'||(g%50) END,
  to_tsvector('simple', CASE
    WHEN g BETWEEN 1 AND 10  THEN 'common corpus clustera alpha beta gamma tail'
    WHEN g BETWEEN 11 AND 20 THEN 'common corpus clustera gamma alpha beta tail'
    WHEN g BETWEEN 21 AND 30 THEN 'common corpus clustera alpha f1 f2 f3 f4 f5 f6 f7 f8 f9 beta f10 f11 f12 f13 f14 f15 f16 f17 f18 gamma'
    WHEN g BETWEEN 31 AND 40 THEN 'common corpus clustera alpha gamma beta alpha beta gamma beta alpha gamma'
    WHEN g BETWEEN 41 AND 50 THEN 'common corpus clustera rareid'||g
    ELSE 'common corpus cluster'||(g%37)||' filler'||(g%50) END),
  (g/10), (g%6)
FROM generate_series(1,:NROWS) g;
ANALYZE bpr_docs;

CREATE INDEX bpr_idx ON bpr_docs USING turbohybrid (embedding vector_cosine_turbohybrid_ops, body_tsv bm25_tsvector_turbohybrid_ops)
  INCLUDE (group_id, category_id)
  WITH (quantization_bits = 4, exact_storage = off, native_segments = 1);
ANALYZE bpr_docs;

-- Queries all target the phrase "alpha beta gamma"; the dense vector is the
-- shared cluster centre of ids 1..50. EXPECTED answer = the exact-phrase docs.
DROP TABLE IF EXISTS bpr_q CASCADE;
CREATE TABLE bpr_q (name text PRIMARY KEY, qvec vector(:DIMS), tsq tsquery, expected_ids int[]);
INSERT INTO bpr_q(name,tsq) VALUES
 ('bag',       plainto_tsquery('simple','alpha beta gamma')),          -- no phrase -> auto is a no-op
 ('phrase',    phraseto_tsquery('simple','alpha beta gamma')),         -- phrase -> auto engages
 ('proximity', to_tsquery('simple','alpha <-> beta'));                 -- partial-phrase proximity
UPDATE bpr_q SET qvec=(SELECT embedding FROM bpr_docs WHERE id=5);      -- inside the shared cluster
UPDATE bpr_q SET expected_ids=(SELECT array_agg(id ORDER BY id) FROM bpr_docs WHERE doc_kind='exact_phrase');

DROP TABLE IF EXISTS bpr_res CASCADE;
CREATE TABLE bpr_res (query_name text, rerank_mode text, fusion text,
  overlap_at_k float8, p50_ms float8, p95_ms float8,
  rerank_count int, rerank_fetch_us bigint, rerank_score_us bigint, rerank_topk_changed bool,
  fusion_candidates_seen int, fusion_duplicates int, calibrated_shape text, calibrated_alpha float8,
  result_ids int[]);

CREATE OR REPLACE FUNCTION pg_temp.ov(a int[],b int[]) RETURNS int LANGUAGE sql AS
$$ SELECT count(*)::int FROM (SELECT unnest(a) INTERSECT SELECT unnest(b)) s $$;

\echo '== running bm25 phrase/proximity rerank grid =='
DROP TABLE IF EXISTS bpr_params;
CREATE TEMP TABLE bpr_params AS SELECT :K::int AS p_k, :ITERS::int AS p_iters;
DO $$
DECLARE
  q record; m text; fmode text; ids int[]; st jsonb; f jsonb; sql text; qexpr text;
  k int; dk int := 200; bk int := 200; iters int; t0 timestamptz; i int; lat float8[]; ov float8;
BEGIN
  SELECT p_k, p_iters INTO k, iters FROM bpr_params;
  PERFORM set_config('enable_seqscan','off',false); PERFORM set_config('jit','off',false);
  PERFORM set_config('turbohybrid.profile','matched_recall',false);
  FOREACH m IN ARRAY ARRAY['off','topk','band','auto'] LOOP
    PERFORM set_config('turbohybrid.bm25_heap_tsvector_rerank', m, false);
    FOREACH fmode IN ARRAY ARRAY['rrf','calibrated'] LOOP
      FOR q IN SELECT * FROM bpr_q ORDER BY name LOOP
        qexpr := format('turbohybrid_query(vector_query=>%L::vector, text_query=>%L::tsquery, fusion=>%L, dense_k=>%s, bm25_k=>%s, final_k=>%s)',
          q.qvec::text, q.tsq::text, fmode, dk, bk, k);
        sql := format('SELECT array_agg(id) FROM (SELECT id FROM bpr_docs ORDER BY embedding <~> %s LIMIT %s) s', qexpr, k);
        EXECUTE sql INTO ids;  -- warm
        lat := '{}';
        FOR i IN 1..iters LOOP t0:=clock_timestamp(); EXECUTE sql INTO ids; lat := lat || (extract(epoch FROM clock_timestamp()-t0)*1000); END LOOP;
        st := turbohybrid_last_scan_stats(); f := st->'fusion';
        ov := pg_temp.ov(ids, q.expected_ids)::float8 / GREATEST(1, COALESCE(array_length(q.expected_ids,1),0));
        INSERT INTO bpr_res VALUES (q.name, m, fmode, ov,
          (SELECT percentile_cont(0.5)  WITHIN GROUP (ORDER BY x) FROM unnest(lat) x),
          (SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY x) FROM unnest(lat) x),
          (st->>'bm25_heap_tsvector_rerank_count')::int, (st->>'bm25_heap_tsvector_rerank_fetch_us')::bigint,
          (st->>'bm25_heap_tsvector_rerank_score_us')::bigint, (st->>'bm25_heap_tsvector_rerank_topk_changed')::bool,
          (f->>'candidates_seen')::int, (f->>'duplicates')::int,
          f->>'calibrated_fusion_query_shape', (f->>'calibrated_fusion_alpha_effective')::float8,
          ids);
      END LOOP;
    END LOOP;
  END LOOP;
END $$;

\echo '== overlap@k vs exact-phrase docs, by query / fusion / rerank mode =='
SELECT query_name, fusion,
  round(max(overlap_at_k) FILTER (WHERE rerank_mode='off')::numeric,3)  AS off,
  round(max(overlap_at_k) FILTER (WHERE rerank_mode='topk')::numeric,3) AS topk,
  round(max(overlap_at_k) FILTER (WHERE rerank_mode='band')::numeric,3) AS band,
  round(max(overlap_at_k) FILTER (WHERE rerank_mode='auto')::numeric,3) AS auto
FROM bpr_res GROUP BY query_name, fusion ORDER BY query_name, fusion;

\echo '== rerank engagement + latency per cell (count / topk_changed / fetch_us / score_us / p50 / p95) =='
SELECT query_name, rerank_mode, fusion, rerank_count AS cnt, rerank_topk_changed AS topk_chg,
  rerank_fetch_us AS fetch_us, rerank_score_us AS score_us,
  round(p50_ms::numeric,3) AS p50_ms, round(p95_ms::numeric,3) AS p95_ms
FROM bpr_res ORDER BY query_name,
  array_position(ARRAY['off','topk','band','auto'], rerank_mode), fusion;

\echo '== fusion stats per cell (candidates_seen / duplicates / calibrated shape+alpha) =='
SELECT query_name, rerank_mode, fusion, fusion_candidates_seen AS cand_seen, fusion_duplicates AS dups,
  calibrated_shape, round(calibrated_alpha::numeric,3) AS calibrated_alpha
FROM bpr_res ORDER BY query_name,
  array_position(ARRAY['off','topk','band','auto'], rerank_mode), fusion;

\echo '== DONE bm25_phrase_rerank_grid =='
