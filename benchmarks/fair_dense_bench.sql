-- Fair dense-only vs hybrid-shape benchmark harness.
--
-- Run:
--   psql -d <db> -f benchmarks/fair_dense_bench.sql
--
-- Optional psql variables:
--   NROWS   synthetic document count              (default 10000)
--   DIMS    embedding dimensions                  (default 384)
--   QSET    fixed query-vector count              (default 64)
--   BITS    quantization bits                     (default 4)
--   WARM    warmup queries per client per shape   (default 20)
--   TIMED   timed queries per client per shape    (default 100)
--   FINAL_K query LIMIT / final_k                 (default 10)
--   DENSE_K dense candidate budget                (default 100)
--
-- This benchmark builds two indexes over equivalent embeddings:
--   A. dense-only: one vector key, no BM25 metadata
--   B. hybrid: vector key plus populated body_tsv, matching the old
--      fake-tsvector dense benchmark shape
--
-- It runs the same vector-only query set against both shapes with 1 and 8
-- concurrent dblink backends.  Precision@K is measured against exact pgvector
-- cosine ordering over the dense table.  Scan stats are sampled from each
-- backend's final timed query.

\set ON_ERROR_STOP on
\pset pager off

\if :{?NROWS}
\else
  \set NROWS 10000
\endif
\if :{?DIMS}
\else
  \set DIMS 384
\endif
\if :{?QSET}
\else
  \set QSET 64
\endif
\if :{?BITS}
\else
  \set BITS 4
\endif
\if :{?WARM}
\else
  \set WARM 20
\endif
\if :{?TIMED}
\else
  \set TIMED 100
\endif
\if :{?FINAL_K}
\else
  \set FINAL_K 10
\endif
\if :{?DENSE_K}
\else
  \set DENSE_K 100
\endif

CREATE EXTENSION IF NOT EXISTS pgturbohybrid CASCADE;
CREATE EXTENSION IF NOT EXISTS dblink;

SET max_parallel_workers_per_gather = 0;
SET jit = off;

\echo '== fair dense benchmark: dense-only vs hybrid shape =='
\echo 'rows=' :NROWS ' dims=' :DIMS ' qset=' :QSET ' bits=' :BITS ' warm=' :WARM ' timed=' :TIMED ' final_k=' :FINAL_K ' dense_k=' :DENSE_K

DROP TABLE IF EXISTS th_fair_dense_docs CASCADE;
DROP TABLE IF EXISTS th_fair_hybrid_docs CASCADE;
DROP TABLE IF EXISTS th_fair_dense_queries CASCADE;
DROP TABLE IF EXISTS th_fair_build_result;
DROP TABLE IF EXISTS th_fair_results;

CREATE TABLE th_fair_dense_docs (
    id int PRIMARY KEY,
    embedding vector(:DIMS) NOT NULL
);

CREATE TABLE th_fair_hybrid_docs (
    id int PRIMARY KEY,
    embedding vector(:DIMS) NOT NULL,
    body_tsv tsvector NOT NULL
);

\echo '== loading equivalent embeddings =='
INSERT INTO th_fair_dense_docs(id, embedding)
SELECT g,
       (SELECT array_agg(
                   (sin(g * 0.011 + d * 0.017) +
                    0.05 * cos(g * 0.037 + d * 0.013))::real
                   ORDER BY d)
        FROM generate_series(1, :DIMS) AS d)::real[]::vector(:DIMS)
FROM generate_series(1, :NROWS) AS g;

INSERT INTO th_fair_hybrid_docs(id, embedding, body_tsv)
SELECT id,
       embedding,
       to_tsvector('english',
                   'document number ' || id || ' dense benchmark common token')
FROM th_fair_dense_docs;

ANALYZE th_fair_dense_docs;
ANALYZE th_fair_hybrid_docs;

CREATE TABLE th_fair_build_result (
    label text PRIMARY KEY,
    build_ms float8 NOT NULL
);

CREATE OR REPLACE FUNCTION pg_temp.th_fair_create_index(p_label text,
                                                        p_sql text)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    started timestamptz;
BEGIN
    started := clock_timestamp();
    EXECUTE p_sql;
    INSERT INTO th_fair_build_result(label, build_ms)
    VALUES (p_label, extract(epoch FROM clock_timestamp() - started) * 1000.0);
END;
$$;

\echo '== building indexes =='
SELECT pg_temp.th_fair_create_index(
    'dense_only',
    'CREATE INDEX th_fair_dense_idx ON th_fair_dense_docs ' ||
    'USING turbohybrid (embedding vector_cosine_turbohybrid_ops) ' ||
    'WITH (quantization_bits = ' || :BITS || ')'
);

SELECT pg_temp.th_fair_create_index(
    'hybrid',
    'CREATE INDEX th_fair_hybrid_idx ON th_fair_hybrid_docs ' ||
    'USING turbohybrid (embedding vector_cosine_turbohybrid_ops, ' ||
    'body_tsv bm25_tsvector_turbohybrid_ops) ' ||
    'WITH (quantization_bits = ' || :BITS || ')'
);

ANALYZE th_fair_dense_docs;
ANALYZE th_fair_hybrid_docs;

\echo '== preparing fixed query set and exact baseline =='
CREATE TABLE th_fair_dense_queries (
    seq int PRIMARY KEY,
    qvec vector(:DIMS) NOT NULL,
    exact_ids int[] NOT NULL
);

WITH sampled AS (
    SELECT row_number() OVER (ORDER BY id) - 1 AS seq,
           embedding AS qvec
    FROM th_fair_dense_docs
    ORDER BY id
    LIMIT :QSET
)
INSERT INTO th_fair_dense_queries(seq, qvec, exact_ids)
SELECT s.seq,
       s.qvec,
       (
           SELECT array_agg(id ORDER BY dist, id)
           FROM (
               SELECT d.id,
                      d.embedding <=> s.qvec AS dist
               FROM th_fair_dense_docs d
               ORDER BY d.embedding <=> s.qvec, d.id
               LIMIT :FINAL_K
           ) exact
       ) AS exact_ids
FROM sampled s;

CREATE TABLE th_fair_results (
    label text NOT NULL,
    clients int NOT NULL,
    metrics jsonb NOT NULL,
    PRIMARY KEY (label, clients)
);

CREATE OR REPLACE FUNCTION th_fair_overlap(a int[], b int[])
RETURNS int
LANGUAGE sql
AS $$
    SELECT count(*)::int
    FROM (
        SELECT unnest(a)
        INTERSECT
        SELECT unnest(b)
    ) s
$$;

-- Runs inside each dblink backend.  Functions and the fixed query table are
-- regular database objects because dblink sessions cannot see pg_temp objects
-- from the driver backend.
CREATE OR REPLACE FUNCTION th_fair_dense_client_run(p_tbl text,
                                                    p_warm int,
                                                    p_timed int,
                                                    p_dense_k int,
                                                    p_final_k int)
RETURNS jsonb
LANGUAGE plpgsql
AS $fn$
DECLARE
    qtext text;
    qrec record;
    nq int;
    k int;
    t0 timestamptz;
    wall0 timestamptz;
    ids int[];
    st jsonb;
    durs float8[] := '{}';
    precision_sum float8 := 0.0;
    precision_min float8 := NULL;
    precision float8;
BEGIN
    PERFORM set_config('enable_seqscan', 'off', false);
    PERFORM set_config('jit', 'off', false);
    PERFORM set_config('max_parallel_workers_per_gather', '0', false);

    SELECT count(*) INTO nq FROM th_fair_dense_queries;
    qtext := format(
        'SELECT array_agg(id ORDER BY ord) FROM (' ||
        'SELECT id, row_number() OVER () AS ord ' ||
        'FROM (SELECT id FROM %I ' ||
        'ORDER BY embedding <~> turbohybrid_query(' ||
        'vector_query => $1, dense_k => $2, final_k => $3) ' ||
        'LIMIT $3) ranked' ||
        ') s',
        p_tbl);

    FOR k IN 0 .. p_warm - 1 LOOP
        SELECT qvec, exact_ids INTO qrec
        FROM th_fair_dense_queries
        WHERE seq = (k % nq);
        EXECUTE qtext INTO ids USING qrec.qvec, p_dense_k, p_final_k;
    END LOOP;

    wall0 := clock_timestamp();
    FOR k IN 0 .. p_timed - 1 LOOP
        SELECT qvec, exact_ids INTO qrec
        FROM th_fair_dense_queries
        WHERE seq = (k % nq);

        t0 := clock_timestamp();
        EXECUTE qtext INTO ids USING qrec.qvec, p_dense_k, p_final_k;
        durs := array_append(durs,
                             extract(epoch FROM clock_timestamp() - t0) * 1000.0);

        precision := th_fair_overlap(ids, qrec.exact_ids)::float8 / p_final_k;
        precision_sum := precision_sum + precision;
        precision_min := least(coalesce(precision_min, precision), precision);
    END LOOP;

    st := turbohybrid_last_scan_stats();

    RETURN jsonb_build_object(
        'pid', pg_backend_pid(),
        'n', p_timed,
        'wall_ms', extract(epoch FROM clock_timestamp() - wall0) * 1000.0,
        'durations', to_jsonb(durs),
        'precision_at_k_avg', precision_sum / NULLIF(p_timed, 0),
        'precision_at_k_min', precision_min,
        'last_scan_stats', st,
        'index_shape', st->>'index_shape',
        'bm25_branch_available', (st->>'bm25_branch_available')::boolean,
        'dense_branch_used', (st->>'dense_branch_used')::boolean,
        'bm25_branch_used', (st->>'bm25_branch_used')::boolean,
        'scan_orchestration', st->>'scan_orchestration',
        'native_cache_bytes', NULLIF(st->>'native_cache_bytes', '')::bigint,
        'dense_elapsed_us', NULLIF(st->>'dense_elapsed_us', '')::bigint,
        'bm25_elapsed_us', NULLIF(st->>'bm25_elapsed_us', '')::bigint,
        'graph_total_us', NULLIF(st->>'graph_total_us', '')::bigint
    );
END $fn$;

CREATE OR REPLACE FUNCTION th_fair_dense_run_config(p_label text,
                                                    p_tbl text,
                                                    p_clients int,
                                                    p_warm int,
                                                    p_timed int,
                                                    p_dense_k int,
                                                    p_final_k int)
RETURNS jsonb
LANGUAGE plpgsql
AS $fn$
DECLARE
    connstr text;
    cn text;
    sql text;
    res jsonb;
    results jsonb[] := '{}';
    i int;
    busy int;
    total_n bigint;
    max_wall_ms float8;
    rps float8;
    p50 float8;
    p95 float8;
    p99 float8;
BEGIN
    connstr := format('dbname=%s port=%s application_name=th_fair_dense_client',
                      current_database(),
                      coalesce(inet_server_port()::text, '5432'));

    FOR i IN 0 .. p_clients - 1 LOOP
        cn := 'th_fair_dense_' || i;
        PERFORM dblink_connect(cn, connstr);
        sql := format('SELECT th_fair_dense_client_run(%L,%s,%s,%s,%s)',
                      p_tbl, p_warm, p_timed, p_dense_k, p_final_k);
        PERFORM dblink_send_query(cn, sql);
    END LOOP;

    LOOP
        busy := 0;
        FOR i IN 0 .. p_clients - 1 LOOP
            busy := busy + dblink_is_busy('th_fair_dense_' || i);
        END LOOP;
        EXIT WHEN busy = 0;
        PERFORM pg_sleep(0.01);
    END LOOP;

    FOR i IN 0 .. p_clients - 1 LOOP
        cn := 'th_fair_dense_' || i;
        SELECT r INTO res FROM dblink_get_result(cn) AS t(r jsonb);
        IF res IS NOT NULL THEN
            results := array_append(results, res);
        END IF;
        PERFORM 1 FROM dblink_get_result(cn) AS t(r jsonb);
        PERFORM dblink_disconnect(cn);
    END LOOP;

    SELECT sum((r->>'n')::bigint), max((r->>'wall_ms')::float8)
    INTO total_n, max_wall_ms
    FROM unnest(results) r;

    rps := CASE
        WHEN max_wall_ms > 0 THEN total_n / (max_wall_ms / 1000.0)
    END;

    SELECT percentile_cont(0.50) WITHIN GROUP (ORDER BY d),
           percentile_cont(0.95) WITHIN GROUP (ORDER BY d),
           percentile_cont(0.99) WITHIN GROUP (ORDER BY d)
    INTO p50, p95, p99
    FROM (
        SELECT (jsonb_array_elements_text(r->'durations'))::float8 AS d
        FROM unnest(results) r
    ) q;

    RETURN jsonb_build_object(
        'label', p_label,
        'clients', p_clients,
        'distinct_pids', (SELECT count(DISTINCT (r->>'pid')) FROM unnest(results) r),
        'queries', total_n,
        'rps', round(rps::numeric, 1),
        'p50_ms', round(p50::numeric, 4),
        'p95_ms', round(p95::numeric, 4),
        'p99_ms', round(p99::numeric, 4),
        'precision_at_k_avg',
            (SELECT round(avg((r->>'precision_at_k_avg')::float8)::numeric, 4)
             FROM unnest(results) r),
        'precision_at_k_min',
            (SELECT round(min((r->>'precision_at_k_min')::float8)::numeric, 4)
             FROM unnest(results) r),
        'index_shape', (SELECT max(r->>'index_shape') FROM unnest(results) r),
        'bm25_branch_available',
            (SELECT bool_or((r->>'bm25_branch_available')::boolean)
             FROM unnest(results) r),
        'dense_branch_used',
            (SELECT bool_or((r->>'dense_branch_used')::boolean)
             FROM unnest(results) r),
        'bm25_branch_used',
            (SELECT bool_or((r->>'bm25_branch_used')::boolean)
             FROM unnest(results) r),
        'scan_orchestration',
            (SELECT max(r->>'scan_orchestration') FROM unnest(results) r),
        'native_cache_bytes',
            (SELECT max((r->>'native_cache_bytes')::bigint) FROM unnest(results) r),
        'dense_elapsed_us',
            (SELECT round(avg((r->>'dense_elapsed_us')::bigint), 1)
             FROM unnest(results) r),
        'bm25_elapsed_us',
            (SELECT round(avg((r->>'bm25_elapsed_us')::bigint), 1)
             FROM unnest(results) r),
        'graph_total_us',
            (SELECT round(avg((r->>'graph_total_us')::bigint), 1)
             FROM unnest(results) r)
    );
END $fn$;

\echo '== running fair matrix with true separate backends =='
INSERT INTO th_fair_results(label, clients, metrics)
VALUES
    ('dense_only', 1, th_fair_dense_run_config('dense_only', 'th_fair_dense_docs',
                                               1, :WARM, :TIMED, :DENSE_K, :FINAL_K)),
    ('dense_only', 8, th_fair_dense_run_config('dense_only', 'th_fair_dense_docs',
                                               8, :WARM, :TIMED, :DENSE_K, :FINAL_K)),
    ('hybrid', 1, th_fair_dense_run_config('hybrid', 'th_fair_hybrid_docs',
                                           1, :WARM, :TIMED, :DENSE_K, :FINAL_K)),
    ('hybrid', 8, th_fair_dense_run_config('hybrid', 'th_fair_hybrid_docs',
                                           8, :WARM, :TIMED, :DENSE_K, :FINAL_K));

\echo '== build time and index size =='
WITH index_sizes AS (
    SELECT 'dense_only'::text AS label,
           pg_relation_size('th_fair_dense_idx'::regclass) AS index_bytes,
           turbohybrid_index_stats('th_fair_dense_idx'::regclass) AS index_stats
    UNION ALL
    SELECT 'hybrid',
           pg_relation_size('th_fair_hybrid_idx'::regclass),
           turbohybrid_index_stats('th_fair_hybrid_idx'::regclass)
)
SELECT s.label,
       round(b.build_ms::numeric, 3) AS build_ms,
       pg_size_pretty(s.index_bytes) AS index_size,
       s.index_bytes,
       s.index_stats->>'index_shape' AS index_shape,
       (s.index_stats->>'bm25_branch_available')::boolean AS bm25_branch_available,
       (s.index_stats->>'bm25_document_count')::int AS bm25_document_count
FROM index_sizes s
JOIN th_fair_build_result b USING (label)
ORDER BY s.label;

\echo '== dense latency, throughput, and exact precision =='
SELECT label,
       clients,
       (metrics->>'queries')::int AS queries,
       (metrics->>'distinct_pids')::int AS distinct_pids,
       (metrics->>'rps')::numeric AS rps,
       (metrics->>'p50_ms')::numeric AS p50_ms,
       (metrics->>'p95_ms')::numeric AS p95_ms,
       (metrics->>'p99_ms')::numeric AS p99_ms,
       (metrics->>'precision_at_k_avg')::numeric AS precision_at_k_avg,
       (metrics->>'precision_at_k_min')::numeric AS precision_at_k_min
FROM th_fair_results
ORDER BY label, clients;

\echo '== last scan stats aggregated across client backends =='
SELECT label,
       clients,
       metrics->>'index_shape' AS index_shape,
       (metrics->>'bm25_branch_available')::boolean AS bm25_branch_available,
       (metrics->>'dense_branch_used')::boolean AS dense_branch_used,
       (metrics->>'bm25_branch_used')::boolean AS bm25_branch_used,
       metrics->>'scan_orchestration' AS scan_orchestration,
       pg_size_pretty((metrics->>'native_cache_bytes')::bigint) AS native_cache_bytes,
       (metrics->>'dense_elapsed_us')::numeric AS dense_elapsed_us_avg,
       (metrics->>'bm25_elapsed_us')::numeric AS bm25_elapsed_us_avg,
       (metrics->>'graph_total_us')::numeric AS graph_total_us_avg
FROM th_fair_results
ORDER BY label, clients;

\echo '== dense-only vs hybrid ratios =='
WITH build AS (
    SELECT max(build_ms) FILTER (WHERE label = 'dense_only') AS dense_build_ms,
           max(build_ms) FILTER (WHERE label = 'hybrid') AS hybrid_build_ms,
           pg_relation_size('th_fair_dense_idx'::regclass) AS dense_bytes,
           pg_relation_size('th_fair_hybrid_idx'::regclass) AS hybrid_bytes
    FROM th_fair_build_result
),
lat AS (
    SELECT clients,
           max((metrics->>'p95_ms')::numeric) FILTER (WHERE label = 'dense_only') AS dense_p95_ms,
           max((metrics->>'p95_ms')::numeric) FILTER (WHERE label = 'hybrid') AS hybrid_p95_ms,
           max((metrics->>'precision_at_k_avg')::numeric) FILTER (WHERE label = 'dense_only') AS dense_precision,
           max((metrics->>'precision_at_k_avg')::numeric) FILTER (WHERE label = 'hybrid') AS hybrid_precision
    FROM th_fair_results
    GROUP BY clients
)
SELECT l.clients,
       round(b.dense_build_ms::numeric, 3) AS dense_build_ms,
       round(b.hybrid_build_ms::numeric, 3) AS hybrid_build_ms,
       round((b.hybrid_build_ms / NULLIF(b.dense_build_ms, 0))::numeric, 3) AS hybrid_build_x_dense,
       pg_size_pretty(b.dense_bytes) AS dense_index_size,
       pg_size_pretty(b.hybrid_bytes) AS hybrid_index_size,
       round((b.hybrid_bytes::numeric / NULLIF(b.dense_bytes, 0))::numeric, 3) AS hybrid_size_x_dense,
       l.dense_p95_ms,
       l.hybrid_p95_ms,
       round((l.hybrid_p95_ms / NULLIF(l.dense_p95_ms, 0))::numeric, 3) AS hybrid_p95_x_dense,
       l.dense_precision,
       l.hybrid_precision
FROM lat l
CROSS JOIN build b
ORDER BY l.clients;

\echo '== interpretation =='
\echo 'dense_only should report index_shape=dense_only and bm25_branch_available=false.'
\echo 'hybrid should report index_shape=hybrid and bm25_branch_available=true, but bm25_branch_used=false for vector-only queries.'
\echo 'Precision@K is overlap with exact pgvector cosine top-K over the same vectors.'
\echo 'The 1-client and 8-client rows use separate dblink backends, so per-backend cache effects are visible in scan stats.'

DROP FUNCTION IF EXISTS th_fair_dense_run_config(text, text, int, int, int, int, int);
DROP FUNCTION IF EXISTS th_fair_dense_client_run(text, int, int, int, int);
DROP FUNCTION IF EXISTS th_fair_overlap(int[], int[]);
DROP FUNCTION IF EXISTS pg_temp.th_fair_create_index(text, text);

DROP TABLE IF EXISTS th_fair_results;
DROP TABLE IF EXISTS th_fair_build_result;
DROP TABLE IF EXISTS th_fair_dense_queries;
DROP TABLE IF EXISTS th_fair_dense_docs;
DROP TABLE IF EXISTS th_fair_hybrid_docs;

RESET enable_seqscan;
RESET max_parallel_workers_per_gather;
RESET jit;
