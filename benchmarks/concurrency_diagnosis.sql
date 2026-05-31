-- Per-query dense native concurrency diagnosis fields.
--
-- This is a lightweight companion to concurrency_dense_bench.sql.  It runs a
-- fixed vector query set in one backend and prints the native cache / lock /
-- buffer-wait / traversal fields after every query.  Use it to confirm that the
-- instrumentation is present, and use pgbench or the dblink matrix harness for
-- real concurrent-client measurements.
--
-- Run:
--   psql -d <db> -f benchmarks/concurrency_diagnosis.sql
--
-- Optional psql variables:
--   TBL      existing table to use, else synthetic table is built (default items)
--   VCOL     vector column                                   (default embedding)
--   NROWS    synthetic rows                                  (default 10000)
--   DIMS     synthetic dimensions                            (default 100)
--   QSET     fixed query-vector count                        (default 8)
--   WARM     warmup queries printed before timed phase       (default 2)
--   TIMED    timed queries printed                           (default 8)
--   DENSE_K  turbohybrid_query dense_k                       (default 100)
--   FINAL_K  query LIMIT / final_k                           (default 10)
--   POLICY   turbohybrid.native_cache_scope                  (default auto)
--   CACHE_MB turbohybrid.native_cache_max_mb                 (default 512)

\set ON_ERROR_STOP on
\pset pager off

\if :{?TBL}
\else
  \set TBL items
\endif
\if :{?VCOL}
\else
  \set VCOL embedding
\endif
\if :{?NROWS}
\else
  \set NROWS 10000
\endif
\if :{?DIMS}
\else
  \set DIMS 100
\endif
\if :{?QSET}
\else
  \set QSET 8
\endif
\if :{?WARM}
\else
  \set WARM 2
\endif
\if :{?TIMED}
\else
  \set TIMED 8
\endif
\if :{?DENSE_K}
\else
  \set DENSE_K 100
\endif
\if :{?FINAL_K}
\else
  \set FINAL_K 10
\endif
\if :{?POLICY}
\else
  \set POLICY auto
\endif
\if :{?CACHE_MB}
\else
  \set CACHE_MB 512
\endif

CREATE EXTENSION IF NOT EXISTS pgturbohybrid CASCADE;

SET enable_seqscan = off;
SET max_parallel_workers_per_gather = 0;
SET jit = off;
SET turbohybrid.native_cache_scope = :'POLICY';
SET turbohybrid.native_cache_max_mb = :CACHE_MB;

SELECT to_regclass(:'TBL') IS NOT NULL AS "TBL_EXISTS" \gset
\if :TBL_EXISTS
  \echo 'concurrency_diagnosis: using existing table' :TBL
\else
  \echo 'concurrency_diagnosis: building synthetic table th_diag_items (' :NROWS 'x' :DIMS ')'
  DROP TABLE IF EXISTS th_diag_items;
  CREATE TABLE th_diag_items (
      id int PRIMARY KEY,
      embedding vector(:DIMS) NOT NULL
  );
  INSERT INTO th_diag_items(id, embedding)
  SELECT g,
         (SELECT array_agg(
                     (sin(g * 0.011 + d * 0.017) +
                      0.05 * cos(g * 0.037 + d * 0.013))::real
                     ORDER BY d)
          FROM generate_series(1, :DIMS) AS d)::real[]::vector(:DIMS)
  FROM generate_series(1, :NROWS) AS g;
  CREATE INDEX th_diag_items_idx ON th_diag_items
    USING turbohybrid (embedding vector_cosine_turbohybrid_ops)
    WITH (quantization_bits = 4);
  ANALYZE th_diag_items;
  \set TBL th_diag_items
  \set VCOL embedding
\endif

DROP TABLE IF EXISTS th_diag_queries;
CREATE TEMP TABLE th_diag_queries (seq int PRIMARY KEY, qvec vector);
INSERT INTO th_diag_queries(seq, qvec)
SELECT row_number() OVER (ORDER BY id) - 1, :VCOL::vector
FROM :TBL
ORDER BY id
LIMIT :QSET;

DROP TABLE IF EXISTS th_diag_results;
CREATE TEMP TABLE th_diag_results (
    phase text NOT NULL,
    query_no int NOT NULL,
    elapsed_ms float8 NOT NULL,
    stats jsonb NOT NULL
);

CREATE OR REPLACE FUNCTION pg_temp.th_diag_run_queries(p_tbl regclass,
                                                       p_vcol name,
                                                       p_warm int,
                                                       p_timed int,
                                                       p_dense_k int,
                                                       p_final_k int)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    qtext text;
    qv vector;
    nq int;
    k int;
    started timestamptz;
    elapsed float8;
BEGIN
    SELECT count(*) INTO nq FROM th_diag_queries;
    qtext := format(
        'SELECT count(*) FROM (' ||
        'SELECT id FROM %s ' ||
        'ORDER BY %I <~> turbohybrid_query(' ||
        'vector_query => $1, dense_k => $2, final_k => $3) ' ||
        'LIMIT $3' ||
        ') s',
        p_tbl,
        p_vcol);

    FOR k IN 0 .. p_warm - 1 LOOP
        SELECT qvec INTO qv FROM th_diag_queries WHERE seq = (k % nq);
        started := clock_timestamp();
        EXECUTE qtext USING qv, p_dense_k, p_final_k;
        elapsed := extract(epoch FROM clock_timestamp() - started) * 1000.0;
        INSERT INTO th_diag_results
        VALUES ('warm', k + 1, elapsed, turbohybrid_last_scan_stats());
    END LOOP;

    FOR k IN 0 .. p_timed - 1 LOOP
        SELECT qvec INTO qv FROM th_diag_queries WHERE seq = (k % nq);
        started := clock_timestamp();
        EXECUTE qtext USING qv, p_dense_k, p_final_k;
        elapsed := extract(epoch FROM clock_timestamp() - started) * 1000.0;
        INSERT INTO th_diag_results
        VALUES ('timed', k + 1, elapsed, turbohybrid_last_scan_stats());
    END LOOP;
END;
$$;

\echo '== per-query native dense diagnosis =='
\echo 'table=' :TBL ' vector_col=' :VCOL ' policy=' :POLICY ' cache_mb=' :CACHE_MB ' warm=' :WARM ' timed=' :TIMED
SELECT pg_temp.th_diag_run_queries(:'TBL'::regclass, :'VCOL'::name,
                                   :WARM, :TIMED, :DENSE_K, :FINAL_K);

SELECT phase,
       query_no,
       round(elapsed_ms::numeric, 3) AS elapsed_ms,
       stats->>'scan_orchestration' AS scan_orchestration,
       stats->>'native_cache_policy' AS native_cache_policy,
       stats->>'native_cache_scope' AS native_cache_scope,
       (stats->>'native_cache_used')::boolean AS native_cache_used,
       stats->>'native_cache_reason' AS native_cache_reason,
       (stats->>'native_cache_reused')::boolean AS native_cache_reused,
       (stats->>'native_cache_built_this_scan')::boolean AS native_cache_built_this_scan,
       (stats->>'native_cache_attach_us')::bigint AS native_cache_attach_us,
       (stats->>'native_cache_build_us')::bigint AS native_cache_build_us,
       (stats->>'native_cache_wait_us')::bigint AS native_cache_wait_us,
       (stats->>'native_cache_refcount')::bigint AS native_cache_refcount,
       pg_size_pretty((stats->>'native_cache_bytes')::bigint) AS native_cache_bytes,
       pg_size_pretty((stats->>'native_cache_code_bytes')::bigint) AS native_cache_code_bytes,
       pg_size_pretty((stats->>'native_cache_adj_bytes')::bigint) AS native_cache_adj_bytes,
       pg_size_pretty((stats->>'native_cache_exact_bytes')::bigint) AS native_cache_exact_bytes,
       (stats->>'graph_scan_lock_wait_us')::bigint AS graph_scan_lock_wait_us,
       (stats->>'code_buffer_lock_wait_us')::bigint AS code_buffer_lock_wait_us,
       (stats->>'adj_buffer_lock_wait_us')::bigint AS adj_buffer_lock_wait_us,
       (stats->>'graph_code_pages_read')::bigint AS graph_code_pages_read,
       (stats->>'graph_adj_pages_read')::bigint AS graph_adj_pages_read,
       (stats->>'graph_traverse_us')::bigint AS graph_traverse_us,
       (stats->>'graph_total_us')::bigint AS graph_total_us
FROM th_diag_results
ORDER BY phase = 'timed', query_no;

\echo '== interpretation =='
\echo 'Cold cache build: native_cache_built_this_scan=true and native_cache_build_us is large.'
\echo 'Per-backend reuse: native_cache_scope=per_backend, native_cache_used=true, and native_cache_reused=true after warmup.'
\echo 'Shared cache comparison: rerun with -v POLICY=shared; attach/build/wait stats should show mmap build or attach.'
\echo 'Policy-off comparison: rerun with -v POLICY=off; native_cache_used=false and native_cache_reason=policy_off should appear.'
\echo 'Scan-lock contention: graph_scan_lock_wait_us grows under concurrent external clients.'
\echo 'Buffer-lock/page loading: code/adj_buffer_lock_wait_us and graph_*_pages_read grow, especially with CACHE_MB=0.'
\echo 'Steady-state traversal: graph_traverse_us/graph_total_us grow while lock/page counters stay near zero.'

DROP FUNCTION pg_temp.th_diag_run_queries(regclass, name, int, int, int, int);
DROP TABLE IF EXISTS th_diag_queries;
DROP TABLE IF EXISTS th_diag_results;

RESET enable_seqscan;
RESET max_parallel_workers_per_gather;
RESET jit;
RESET turbohybrid.native_cache_scope;
RESET turbohybrid.native_cache_max_mb;
