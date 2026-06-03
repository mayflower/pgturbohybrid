-- profile_recall_latency_grid.sql
--
-- Deterministic AMD64/x86_64 profile recall/latency grid for the named retrieval
-- profiles. Developer harness, NOT a publishable result: it prints a report and
-- leaves no result files. DO NOT commit its output (see benchmarks/README.md).
--
-- It changes NO profile defaults and NO runtime behavior. It only SETs
-- turbohybrid.profile / turbohybrid.simd for the duration of the session and
-- builds throwaway indexes on throwaway tables.
--
-- Run:
--   psql -d <db> -f benchmarks/dev/profile_recall_latency_grid.sql
--   psql -d <db> -v SIMD_MODE=off -f benchmarks/dev/profile_recall_latency_grid.sql
--
-- psql variables (all optional):
--   SIMD_MODE   on | off | both   (default both) -- turbohybrid.simd settings to sweep
--   DIMS        embedding dims                    (default 1536; >=1024 exercises the u8 SIMD path)
--   NROWS_EASY  easy doc/chunk corpus rows        (default 5000)
--   NROWS_HARD  hard spread corpus rows           (default 6000)
--   K           result cutoff                     (default 10)
--   ITERS       timed iterations per latency cell (default 60)

\set ON_ERROR_STOP on
\pset pager off
CREATE EXTENSION IF NOT EXISTS pgturbohybrid CASCADE;

\if :{?SIMD_MODE}
\else
  \set SIMD_MODE both
\endif
\if :{?DIMS}
\else
  \set DIMS 1536
\endif
\if :{?NROWS_EASY}
\else
  \set NROWS_EASY 5000
\endif
\if :{?NROWS_HARD}
\else
  \set NROWS_HARD 6000
\endif
\if :{?K}
\else
  \set K 10
\endif
\if :{?ITERS}
\else
  \set ITERS 60
\endif
\echo 'profile_recall_latency_grid: SIMD_MODE=' :SIMD_MODE ' DIMS=' :DIMS ' EASY=' :NROWS_EASY ' HARD=' :NROWS_HARD ' K=' :K ' ITERS=' :ITERS

-- ---------------------------------------------------------------------------
-- Corpus 1 (easy): "documents" x near-duplicate "chunks". A query near one
-- document returns many of its chunks -> easy dense recall, grouped/duplicate
-- case, payload filter, lexical, hybrid. Deterministic (sin/cos, no random()).
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS prg_easy CASCADE;
CREATE TABLE prg_easy (id int PRIMARY KEY, embedding vector(:DIMS) NOT NULL, body text NOT NULL,
                       body_tsv tsvector NOT NULL, group_id int, category_id int);
INSERT INTO prg_easy
SELECT g,
  (SELECT array_agg((sin(((g/10)+1)*0.21+d*0.012)+cos(((g/10)+1)*0.13+d*0.006)+0.05*sin(g*0.05+d*0.013))::real ORDER BY d)
   FROM generate_series(1,:DIMS) d)::real[]::vector,
  concat_ws(' ','doc'||(g/10),'clusterterm'||(g%40),'cat'||(g%6),'common corpus shared'),
  to_tsvector('simple', concat_ws(' ','doc'||(g/10),'clusterterm'||(g%40),'cat'||(g%6),'common corpus shared')),
  (g/10), (g%6)
FROM generate_series(1,:NROWS_EASY) g;
ANALYZE prg_easy;

-- ---------------------------------------------------------------------------
-- Corpus 2 (hard): spread/unique vectors. The query is OUT OF DISTRIBUTION
-- (sum of three distant docs) so its true neighbours are scattered -> low
-- ef_search profiles miss them; high_recall recovers them.
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS prg_hard CASCADE;
CREATE TABLE prg_hard (id int PRIMARY KEY, embedding vector(:DIMS) NOT NULL);
INSERT INTO prg_hard
SELECT g,
  (SELECT array_agg((sin((g%40+1)*0.37+d*0.011)+cos((g%40+1)*0.19+d*0.007)+0.55*sin(g*0.05+d*0.013))::real ORDER BY d)
   FROM generate_series(1,:DIMS) d)::real[]::vector
FROM generate_series(1,:NROWS_HARD) g;
ANALYZE prg_hard;

-- Query vectors + exact ground truth (seqscan cosine <=>; lexical via ts_rank_cd).
DROP TABLE IF EXISTS prg_q CASCADE;
CREATE TABLE prg_q (name text PRIMARY KEY, corpus text, kind text, qvec vector(:DIMS), tsq tsquery, filter_cat int, exact_ids int[]);
INSERT INTO prg_q(name,corpus,kind,tsq,filter_cat) VALUES
 ('dense_easy',    'easy','dense',   NULL, NULL),
 ('lexical',       'easy','lexical', plainto_tsquery('simple','clusterterm7'), NULL),
 ('hybrid',        'easy','hybrid',  plainto_tsquery('simple','clusterterm7'), NULL),
 ('payload_filter','easy','dense',   NULL, 3),
 ('grouped',       'easy','dense',   NULL, NULL),
 ('hard_dense',    'hard','dense',   NULL, NULL);
UPDATE prg_q SET qvec=(SELECT embedding FROM prg_easy WHERE id=807) WHERE name IN ('dense_easy','hybrid','payload_filter');
UPDATE prg_q SET qvec=(SELECT embedding FROM prg_easy WHERE id=500) WHERE name='grouped';
UPDATE prg_q SET qvec=(SELECT a.embedding+b.embedding+c.embedding FROM prg_hard a JOIN prg_hard b ON b.id=2011 JOIN prg_hard c ON c.id=5003 WHERE a.id=807) WHERE name='hard_dense';
UPDATE prg_q q SET exact_ids=(SELECT array_agg(id ORDER BY embedding <=> q.qvec)
  FROM (SELECT id,embedding FROM prg_easy d WHERE q.filter_cat IS NULL OR d.category_id=q.filter_cat ORDER BY embedding <=> q.qvec LIMIT :K) s)
  WHERE corpus='easy' AND kind IN ('dense','hybrid');
UPDATE prg_q q SET exact_ids=(SELECT array_agg(id ORDER BY ts_rank_cd(body_tsv,q.tsq) DESC)
  FROM (SELECT id,body_tsv FROM prg_easy d WHERE d.body_tsv @@ q.tsq ORDER BY ts_rank_cd(body_tsv,q.tsq) DESC LIMIT :K) s)
  WHERE kind='lexical';
UPDATE prg_q q SET exact_ids=(SELECT array_agg(id ORDER BY embedding <=> q.qvec)
  FROM (SELECT id,embedding FROM prg_hard d ORDER BY embedding <=> q.qvec LIMIT :K) s)
  WHERE corpus='hard';

DROP TABLE IF EXISTS prg_report CASCADE;
CREATE TABLE prg_report (
  profile text, simd text,
  quantization_bits int, graph_m int, graph_ef_construction int, graph_ef_search int, graph_oversampling int,
  build_neighbor_select text, retry_mode text, residual_mode text, bm25_heap_rerank_mode text, final_diversity_mode text,
  dense_scorer text, native_cache_reason text, scalar_scored int, simd_scored int,
  easy_dense_recall float8, hard_dense_recall float8, payload_recall float8, hybrid_overlap float8,
  grouped_dup_groups int, hard_p50_ms float8, hard_p95_ms float8, lexical_p50_ms float8);

CREATE OR REPLACE FUNCTION pg_temp.ov(a int[],b int[]) RETURNS int LANGUAGE sql AS
$$ SELECT count(*)::int FROM (SELECT unnest(a) INTERSECT SELECT unnest(b)) s $$;
CREATE OR REPLACE FUNCTION pg_temp.dupg(ids int[]) RETURNS int LANGUAGE sql AS
$$ SELECT COALESCE(sum(c-1),0)::int FROM (SELECT count(*) c FROM unnest(ids) u JOIN prg_easy d ON d.id=u GROUP BY d.group_id HAVING count(*)>1) s $$;

-- psql :variables are not substituted inside DO $$ ... $$ blocks; pass them via
-- a temp table the block reads.
DROP TABLE IF EXISTS prg_params;
CREATE TEMP TABLE prg_params AS SELECT :K::int AS p_k, :ITERS::int AS p_iters, lower(:'SIMD_MODE') AS p_simd;

\echo '== running profile recall/latency grid =='
DO $$
DECLARE
  prof text; simd text; q record; ids int[]; st jsonb; idxst jsonb; sql text; qexpr text;
  k int; dk int := 200; iters int; t0 timestamptz; i int; lat float8[];
  simd_mode text; simdlist text[];
  profiles text[] := ARRAY['latency','balanced','matched_recall','high_recall','quality','debug'];
  v_easy_dense float8; v_hard_dense float8; v_payload float8; v_hybrid float8; v_dup int;
  v_hardp50 float8; v_hardp95 float8; v_lexp50 float8;
  v_scorer text; v_cache text; v_scalar int; v_simd int;
BEGIN
  SELECT p_k, p_iters, p_simd INTO k, iters, simd_mode FROM prg_params;
  simdlist := CASE simd_mode WHEN 'on' THEN ARRAY['on'] WHEN 'off' THEN ARRAY['off'] ELSE ARRAY['on','off'] END;
  PERFORM set_config('enable_seqscan','off',false); PERFORM set_config('jit','off',false);
  FOREACH prof IN ARRAY profiles LOOP
    PERFORM set_config('turbohybrid.profile', prof, false);
    EXECUTE 'DROP INDEX IF EXISTS prg_easy_idx';
    EXECUTE 'CREATE INDEX prg_easy_idx ON prg_easy USING turbohybrid (embedding vector_cosine_turbohybrid_ops, body_tsv bm25_tsvector_turbohybrid_ops) INCLUDE (group_id, category_id)';
    EXECUTE 'DROP INDEX IF EXISTS prg_hard_idx';
    EXECUTE 'CREATE INDEX prg_hard_idx ON prg_hard USING turbohybrid (embedding vector_cosine_turbohybrid_ops)';
    ANALYZE prg_easy; ANALYZE prg_hard;
    idxst := turbohybrid_index_stats('prg_easy_idx'::regclass);
    FOREACH simd IN ARRAY simdlist LOOP
      PERFORM set_config('turbohybrid.simd', simd, false);
      v_easy_dense:=NULL; v_hard_dense:=NULL; v_payload:=NULL; v_hybrid:=NULL; v_dup:=NULL;
      v_hardp50:=NULL; v_hardp95:=NULL; v_lexp50:=NULL; v_scorer:=NULL; v_cache:=NULL; v_scalar:=NULL; v_simd:=NULL;
      FOR q IN SELECT * FROM prg_q ORDER BY name LOOP
        qexpr := format('turbohybrid_query(vector_query=>%s, text_query=>%s, dense_k=>%s, bm25_k=>200, final_k=>%s)',
          CASE WHEN q.qvec IS NULL THEN 'NULL::vector' ELSE quote_literal(q.qvec::text)||'::vector' END,
          CASE WHEN q.tsq IS NULL THEN 'NULL::tsquery' ELSE quote_literal(q.tsq::text)||'::tsquery' END,
          CASE WHEN q.qvec IS NULL THEN 0 ELSE dk END, k);
        sql := format('SELECT array_agg(id) FROM (SELECT d.id FROM %I d WHERE %s ORDER BY d.embedding <~> %s LIMIT %s) s',
          CASE WHEN q.corpus='hard' THEN 'prg_hard' ELSE 'prg_easy' END,
          CASE WHEN q.filter_cat IS NULL THEN 'TRUE' ELSE format('d.category_id=%s',q.filter_cat) END, qexpr, k);
        EXECUTE sql INTO ids;  -- warm
        -- time the latency-relevant cases
        IF q.name IN ('hard_dense','lexical') THEN
          lat := '{}';
          FOR i IN 1..iters LOOP t0:=clock_timestamp(); EXECUTE sql INTO ids; lat := lat || (extract(epoch FROM clock_timestamp()-t0)*1000); END LOOP;
        ELSE
          EXECUTE sql INTO ids;
        END IF;
        st := turbohybrid_last_scan_stats();
        IF q.name='dense_easy' THEN
          v_easy_dense := pg_temp.ov(ids,(q.exact_ids)[1:k])::float8/k;
          v_scorer := st->>'dense_scorer'; v_cache := st->>'native_cache_reason';
          v_scalar := (st->>'graph_scalar_scored_codes')::int; v_simd := (st->>'graph_simd_scored_codes')::int;
        ELSIF q.name='hard_dense' THEN
          v_hard_dense := pg_temp.ov(ids,(q.exact_ids)[1:k])::float8/k;
          v_hardp50 := (SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY x) FROM unnest(lat) x);
          v_hardp95 := (SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY x) FROM unnest(lat) x);
        ELSIF q.name='payload_filter' THEN v_payload := pg_temp.ov(ids,(q.exact_ids)[1:k])::float8/k;
        ELSIF q.name='hybrid' THEN v_hybrid := pg_temp.ov(ids,(q.exact_ids)[1:k])::float8/k;
        ELSIF q.name='grouped' THEN v_dup := pg_temp.dupg(ids);
        ELSIF q.name='lexical' THEN v_lexp50 := (SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY x) FROM unnest(lat) x);
        END IF;
      END LOOP;
      INSERT INTO prg_report VALUES (prof, simd,
        (idxst->>'quantization_bits')::int, (idxst->>'graph_m')::int, (idxst->>'graph_ef_construction')::int,
        (idxst->>'graph_ef_search')::int, (idxst->>'graph_oversampling')::int, idxst->>'build_neighbor_select',
        st->>'dense_uncertainty_retry_mode', st->>'residual_rerank_mode', st->>'bm25_heap_tsvector_rerank_mode', st->>'final_diversity_mode',
        v_scorer, v_cache, v_scalar, v_simd,
        round(v_easy_dense::numeric,3), round(v_hard_dense::numeric,3), round(v_payload::numeric,3), round(v_hybrid::numeric,3),
        v_dup, round(v_hardp50::numeric,3), round(v_hardp95::numeric,3), round(v_lexp50::numeric,3));
    END LOOP;
  END LOOP;
END $$;

\echo '== profile grid: build reloptions + effective feature modes =='
SELECT profile, simd, quantization_bits AS bits, graph_m AS m, graph_ef_construction AS efc,
       graph_ef_search AS efs, graph_oversampling AS ovs, build_neighbor_select AS nsel,
       retry_mode AS retry, residual_mode AS residual, bm25_heap_rerank_mode AS bm25_rerank, final_diversity_mode AS final_div,
       dense_scorer, native_cache_reason AS cache, scalar_scored, simd_scored
FROM prg_report
ORDER BY array_position(ARRAY['latency','balanced','matched_recall','high_recall','quality','debug'], profile), simd;

\echo '== profile grid: recall / overlap / latency =='
SELECT profile, simd, easy_dense_recall AS easy_dense, hard_dense_recall AS hard_dense,
       payload_recall, hybrid_overlap, grouped_dup_groups AS grouped_dups,
       hard_p50_ms, hard_p95_ms, lexical_p50_ms
FROM prg_report
ORDER BY array_position(ARRAY['latency','balanced','matched_recall','high_recall','quality','debug'], profile), simd;

\echo '== SIMD on-vs-off parity (recall/overlap/grouping identical beyond candidate-order ties) =='
SELECT count(*) profiles_compared,
  count(*) FILTER (WHERE a.easy_dense_recall IS DISTINCT FROM b.easy_dense_recall
                      OR a.hard_dense_recall IS DISTINCT FROM b.hard_dense_recall
                      OR a.payload_recall   IS DISTINCT FROM b.payload_recall
                      OR a.hybrid_overlap   IS DISTINCT FROM b.hybrid_overlap
                      OR a.grouped_dup_groups <> b.grouped_dup_groups) AS recall_or_grouping_diffs
FROM prg_report a JOIN prg_report b USING (profile) WHERE a.simd='on' AND b.simd='off';

\echo '== DONE profile_recall_latency_grid =='
