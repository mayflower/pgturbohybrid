-- Concurrent-client dense scaling diagnostics for the native TurboHybrid path.
--
-- WHY: pgturbohybrid dense-default scales DOWN, not up, with concurrent clients
-- on glove-100-angular (observed: 1 client ~325 RPS / p95 4.4 ms; 8 clients
-- ~127 RPS / p95 179 ms), while pgvector and Qdrant scale up.  This harness
-- exists to EXPLAIN that collapse before any algorithm is changed: it drives the
-- same fixed query set at 1/2/4/8/16 concurrent backends under three native
-- cache caps and two prewarm modes, and attributes the p95 explosion to one of
-- four causes by reading per-backend instrumentation from each client:
--
--   1. COLD PER-BACKEND CACHE BUILD
--        The native scan cache is process-local: every backend builds its own
--        copy on its first scan.  Signal: native_cache_built_this_scan / build_us
--        on the cold query.  If prewarm mode B (build once before timing) removes
--        the collapse, the cost was the cold build.
--   2. MEMORY BANDWIDTH / CACHE DUPLICATION
--        With a per-backend cache, N clients hold N independent copies of the
--        whole code+adjacency arena.  Signal: native_cache_bytes per backend x
--        clients (total_cache_bytes), with warm scans, 0 page reads, and rising
--        graph_total_us as clients grow.  Compare cache ON (per_backend) vs OFF
--        (native_cache_max_mb=0 -> uncached, shared buffers): if OFF scales and
--        ON collapses, duplication/bandwidth is the cause.
--   3. LOCK WAITS
--        Signal: pg_stat_activity wait_event_type='Lock'/'LWLock' sampled during
--        the run, and ungranted pg_locks on the index (the PGTURBOHYBRID_GRAPH_
--        SCAN_LOCK ShareLock).  High waiting% that grows with clients => lock
--        bound.
--   4. TRAVERSAL / SCORING CPU
--        Signal: graph_batch_us / graph_traverse_us per query rising with client
--        count, no lock waits, ~0 page reads => CPU/core contention.
--
-- This does NOT change any query behaviour -- it only measures.
--
-- TRUE CONCURRENCY MODEL: each "client" is a separate backend opened via dblink
-- (contrib).  Because the native cache is per-backend, N dblink connections give
-- N independent per-backend caches -- exactly the thing under test.  Server-side
-- latency is measured (no client/network round-trip), so absolute RPS differs
-- from the Python vector-db-benchmark numbers; the SCALING SHAPE and the
-- per-backend cause signals are what this harness is for.  For authoritative
-- end-to-end throughput, drive the same query with pgbench -c N (see README).
--
-- USAGE (targets an existing `items` + turbohybrid index if present, else builds
-- a synthetic glove-100-shaped dataset):
--
--   psql -d pgturbohybrid_benchmark -f benchmarks/concurrency_dense_bench.sql
--
-- Override with -v NAME=VALUE:
--   TBL          base table (default items)            -- used if it exists
--   VCOL         vector column (default embedding)
--   NROWS        synthetic rows if TBL is built here    (default 200000)
--   DIMS         synthetic vector dims                  (default 100)
--   QSET         fixed query-vector count               (default 64)
--   WARM         prewarm queries per client (mode B)    (default 40)
--   TIMED        timed queries per client               (default 200)
--   MAX_CLIENTS  cap the client sweep                   (default 16)
--   HIGH_CACHE_MB high native cache cap (MB)            (default 4096)
--
-- WARNING: reproducing the real collapse needs a glove-sized index (~1.18M x
-- 100).  The synthetic default (200k) validates the harness and the signals but
-- may stay cache-resident enough that bandwidth duplication does not yet bite;
-- point -v TBL at the loaded glove table for the real picture.

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
  \set NROWS 200000
\endif
\if :{?DIMS}
\else
  \set DIMS 100
\endif
\if :{?QSET}
\else
  \set QSET 64
\endif
\if :{?WARM}
\else
  \set WARM 40
\endif
\if :{?TIMED}
\else
  \set TIMED 200
\endif
\if :{?MAX_CLIENTS}
\else
  \set MAX_CLIENTS 16
\endif
\if :{?HIGH_CACHE_MB}
\else
  \set HIGH_CACHE_MB 4096
\endif

CREATE EXTENSION IF NOT EXISTS dblink;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid CASCADE;

-- Resolve the target table: reuse :TBL if it exists, else build a synthetic
-- glove-100-shaped dataset (regular table -- dblink client backends must see it).
SELECT to_regclass(:'TBL') IS NOT NULL AS tbl_exists \gset
\if :tbl_exists
  \echo 'concurrency_dense_bench: using existing table' :TBL
\else
  \echo 'concurrency_dense_bench: building synthetic table ccbench_items (' :NROWS 'x' :DIMS ')'
  DROP TABLE IF EXISTS ccbench_items;
  CREATE TABLE ccbench_items (
      id int PRIMARY KEY,
      embedding vector(:DIMS),
      text text DEFAULT '',
      text_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
  );
  INSERT INTO ccbench_items(id, embedding)
  SELECT g,
         (SELECT array_agg((sin(g * 0.001 + d * 0.017) + 0.05 * sin(g * 0.7 + d))::real ORDER BY d)
          FROM generate_series(1, :DIMS) AS d)::real[]::vector
  FROM generate_series(1, :NROWS) AS g;
  CREATE INDEX ccbench_items_idx ON ccbench_items
    USING turbohybrid (embedding vector_cosine_turbohybrid_ops,
                       text_tsv bm25_tsvector_turbohybrid_ops)
    WITH (quantization_bits = 4);
  ANALYZE ccbench_items;
  \set TBL ccbench_items
  \set VCOL embedding
\endif

-- Discover the turbohybrid index on the target table.  Quote the alias so psql
-- keeps the uppercase variable name (unquoted identifiers fold to lowercase).
SELECT c.relname AS "IDX"
FROM pg_index i
JOIN pg_class c ON c.oid = i.indexrelid
JOIN pg_am am ON am.oid = c.relam
WHERE i.indrelid = :'TBL'::regclass AND am.amname = 'turbohybrid'
ORDER BY c.relname
LIMIT 1 \gset
\if :{?IDX}
\else
  \echo 'ERROR: no turbohybrid index found on' :TBL
  \quit
\endif
\echo 'concurrency_dense_bench: table=' :TBL ' vcol=' :VCOL ' index=' :IDX

-- Fixed query set (regular table so dblink client backends can read it): QSET
-- vectors sampled deterministically from the data.  Every client runs the same
-- queries so latency is comparable across clients and configs.
DROP TABLE IF EXISTS ccbench_queries;
CREATE TABLE ccbench_queries (seq int PRIMARY KEY, qvec vector);
INSERT INTO ccbench_queries(seq, qvec)
SELECT row_number() OVER (ORDER BY id) - 1, embedding
FROM (
    SELECT id, :VCOL::vector AS embedding
    FROM :TBL
    ORDER BY id
    LIMIT :QSET
) s;

-- ---------------------------------------------------------------------------
-- Per-client worker (runs inside each dblink backend).  Drives warm + timed
-- loops over the fixed query set and returns its per-backend metrics as one
-- jsonb blob: PID, timed count, wall time, all per-query latencies (for global
-- percentiles), the cold-build signal (from the first timed query) and the
-- steady-state cache/timing signals (from the last timed query).
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION ccbench_client_run(p_tbl text, p_vcol text,
                                              p_cache_mb int, p_prewarm boolean,
                                              p_warm int, p_timed int)
RETURNS jsonb LANGUAGE plpgsql AS $fn$
DECLARE
    qtext text;
    qv    vector;
    k     int;
    nq    int;
    t0    timestamptz;
    wall0 timestamptz;
    durs  float8[] := '{}';
    st    jsonb;
    cold  jsonb := NULL;
BEGIN
    PERFORM set_config('enable_seqscan', 'off', false);
    PERFORM set_config('jit', 'off', false);
    PERFORM set_config('max_parallel_workers_per_gather', '0', false);
    PERFORM set_config('turbohybrid.native_cache_max_mb', p_cache_mb::text, false);

    SELECT count(*) INTO nq FROM ccbench_queries;
    -- %I-quoted identifiers; per-query vector is bound, never interpolated.
    qtext := format('SELECT id FROM %I ORDER BY %I <~> turbohybrid_query('
                    'vector_query => $1, dense_k => 100, final_k => 10) LIMIT 10',
                    p_tbl, p_vcol);

    IF p_prewarm THEN
        FOR k IN 0 .. p_warm - 1 LOOP
            SELECT qvec INTO qv FROM ccbench_queries WHERE seq = (k % nq);
            EXECUTE qtext USING qv;
        END LOOP;
    END IF;

    wall0 := clock_timestamp();
    FOR k IN 0 .. p_timed - 1 LOOP
        SELECT qvec INTO qv FROM ccbench_queries WHERE seq = (k % nq);
        t0 := clock_timestamp();
        EXECUTE qtext USING qv;
        durs := array_append(durs, extract(epoch FROM clock_timestamp() - t0) * 1000);
        IF k = 0 THEN
            cold := turbohybrid_last_scan_stats();
        END IF;
    END LOOP;
    st := turbohybrid_last_scan_stats();    -- steady state (last timed query)

    RETURN jsonb_build_object(
        'pid', pg_backend_pid(),
        'n', p_timed,
        'wall_ms', extract(epoch FROM clock_timestamp() - wall0) * 1000,
        'durations', to_jsonb(durs),
        'orchestration', st->>'scan_orchestration',
        'native_cache_mode', st->>'native_cache_mode',
        'native_cache_bytes', (st->>'native_cache_bytes')::bigint,
        'native_cache_code_bytes', (st->>'native_cache_code_bytes')::bigint,
        'native_cache_adj_bytes', (st->>'native_cache_adj_bytes')::bigint,
        'cold_built', (cold->>'native_cache_built_this_scan')::boolean,
        'cold_build_us', (cold->>'native_cache_build_us')::bigint,
        'code_pages_read', (st->>'graph_code_pages_read')::bigint,
        'batch_us', (st->>'graph_batch_us')::bigint,
        'traverse_us', (st->>'graph_traverse_us')::bigint,
        'heap_us', (st->>'graph_heap_us')::bigint,
        'total_us', (st->>'graph_total_us')::bigint,
        'large_code_arena', (st->>'graph_large_code_arena')::boolean
    );
END $fn$;

-- ---------------------------------------------------------------------------
-- Orchestrator: run ONE (clients, cache_mb, prewarm) config with true
-- concurrency.  Opens p_clients FRESH dblink backends (fresh => mode A measures a
-- genuine cold per-backend build), fires ccbench_client_run on all of them
-- asynchronously, samples pg_stat_activity wait events + pg_locks on the index
-- while they run, then aggregates throughput, global latency percentiles and the
-- per-backend cause signals into one jsonb result.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION ccbench_run_config(p_tbl text, p_vcol text, p_idx text,
                                              p_clients int, p_cache_mb int,
                                              p_prewarm boolean, p_warm int,
                                              p_timed int)
RETURNS jsonb LANGUAGE plpgsql AS $fn$
DECLARE
    connstr   text;
    cn        text;
    i         int;
    busy      int;
    sql       text;
    res       jsonb;
    results   jsonb[] := '{}';
    -- sampling accumulators
    n_samples int := 0;
    n_active  int := 0;
    n_waiting int := 0;
    n_lock    int := 0;
    n_lwlock  int := 0;
    n_io      int := 0;
    n_other   int := 0;
    n_idxlock int := 0;
    v         int;
    wt        text;
    -- aggregates
    total_n     bigint;
    max_wall_ms float8;
    rps         float8;
    p50 float8; p95 float8; p99 float8;
    idx_oid oid := p_idx::regclass::oid;
BEGIN
    connstr := format('dbname=%s port=%s application_name=ccbench_client',
                      current_database(),
                      coalesce(inet_server_port()::text, '5432'));

    -- Open fresh backends and fire the timed run asynchronously on each.
    FOR i IN 0 .. p_clients - 1 LOOP
        cn := 'ccbench_' || i;
        PERFORM dblink_connect(cn, connstr);
        sql := format('SELECT ccbench_client_run(%L,%L,%s,%L,%s,%s)',
                      p_tbl, p_vcol, p_cache_mb, p_prewarm, p_warm, p_timed);
        PERFORM dblink_send_query(cn, sql);
    END LOOP;

    -- Sample wait events + index locks while the clients are in flight.
    LOOP
        busy := 0;
        FOR i IN 0 .. p_clients - 1 LOOP
            busy := busy + dblink_is_busy('ccbench_' || i);
        END LOOP;
        EXIT WHEN busy = 0;

        FOR wt IN
            SELECT wait_event_type
            FROM pg_stat_activity
            WHERE backend_type = 'client backend'
              AND application_name = 'ccbench_client'
              AND pid <> pg_backend_pid()
              AND state = 'active'
        LOOP
            n_active := n_active + 1;
            IF wt IS NOT NULL THEN
                n_waiting := n_waiting + 1;
                IF wt = 'Lock' THEN n_lock := n_lock + 1;
                ELSIF wt = 'LWLock' THEN n_lwlock := n_lwlock + 1;
                ELSIF wt IN ('IO', 'BufferPin') THEN n_io := n_io + 1;
                ELSE n_other := n_other + 1;
                END IF;
            END IF;
        END LOOP;

        SELECT count(*) INTO v FROM pg_locks
        WHERE relation = idx_oid AND NOT granted;
        n_idxlock := n_idxlock + v;

        n_samples := n_samples + 1;
        PERFORM pg_sleep(0.02);
    END LOOP;

    -- Collect each client's result blob, then disconnect.
    FOR i IN 0 .. p_clients - 1 LOOP
        cn := 'ccbench_' || i;
        SELECT r INTO res FROM dblink_get_result(cn) AS t(r jsonb);
        IF res IS NOT NULL THEN
            results := array_append(results, res);
        END IF;
        PERFORM 1 FROM dblink_get_result(cn) AS t(r jsonb);  -- drain
        PERFORM dblink_disconnect(cn);
    END LOOP;

    -- Aggregate throughput + global latency percentiles.
    SELECT sum((r->>'n')::bigint), max((r->>'wall_ms')::float8)
    INTO total_n, max_wall_ms
    FROM unnest(results) r;

    rps := CASE WHEN max_wall_ms > 0 THEN total_n / (max_wall_ms / 1000.0) END;

    SELECT percentile_cont(0.50) WITHIN GROUP (ORDER BY d),
           percentile_cont(0.95) WITHIN GROUP (ORDER BY d),
           percentile_cont(0.99) WITHIN GROUP (ORDER BY d)
    INTO p50, p95, p99
    FROM (
        SELECT (jsonb_array_elements_text(r->'durations'))::float8 AS d
        FROM unnest(results) r
    ) q;

    RETURN jsonb_build_object(
        'clients', p_clients,
        'cache_mb', p_cache_mb,
        'prewarm', p_prewarm,
        'distinct_pids', (SELECT count(DISTINCT (r->>'pid')) FROM unnest(results) r),
        'rps', round(rps::numeric, 1),
        'p50_ms', round(p50::numeric, 3),
        'p95_ms', round(p95::numeric, 3),
        'p99_ms', round(p99::numeric, 3),
        'native_cache_mode', (SELECT max(r->>'native_cache_mode') FROM unnest(results) r),
        'cache_bytes_per_backend', (SELECT max((r->>'native_cache_bytes')::bigint) FROM unnest(results) r),
        'total_cache_bytes', (SELECT sum((r->>'native_cache_bytes')::bigint) FROM unnest(results) r),
        'cold_built_any', (SELECT bool_or((r->>'cold_built')::boolean) FROM unnest(results) r),
        'max_cold_build_us', (SELECT max((r->>'cold_build_us')::bigint) FROM unnest(results) r),
        'avg_code_pages_read', (SELECT round(avg((r->>'code_pages_read')::bigint), 1) FROM unnest(results) r),
        'avg_batch_us', (SELECT round(avg((r->>'batch_us')::bigint), 1) FROM unnest(results) r),
        'avg_traverse_us', (SELECT round(avg((r->>'traverse_us')::bigint), 1) FROM unnest(results) r),
        'avg_heap_us', (SELECT round(avg((r->>'heap_us')::bigint), 1) FROM unnest(results) r),
        'avg_total_us', (SELECT round(avg((r->>'total_us')::bigint), 1) FROM unnest(results) r),
        'non_native', (SELECT bool_or(coalesce(r->>'orchestration','') <> 'graph_native') FROM unnest(results) r),
        -- sampling summary
        'samples', n_samples,
        'active_obs', n_active,
        'waiting_pct', CASE WHEN n_active > 0 THEN round(100.0 * n_waiting / n_active, 1) ELSE 0 END,
        'lock_obs', n_lock,
        'lwlock_obs', n_lwlock,
        'io_obs', n_io,
        'other_wait_obs', n_other,
        'index_lock_waits', n_idxlock
    );
END $fn$;

-- ---------------------------------------------------------------------------
-- Drive the full matrix: clients {1,2,4,8,16} x cache_mb {0, default 512, high}
-- x prewarm {A=off, B=on}.  cache_mb=0 forces uncached per-scan loading; 512 is
-- the shipped default (per-backend cache); HIGH gives headroom for large arenas.
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS ccbench_results;
CREATE TEMP TABLE ccbench_results (cache_mb int, prewarm boolean, clients int, m jsonb);

-- psql does NOT interpolate :vars inside dollar-quoted blocks, so the matrix
-- parameters travel through a config row the driver reads.
DROP TABLE IF EXISTS ccbench_config;
CREATE TEMP TABLE ccbench_config AS
SELECT :'TBL'::text       AS tbl,
       :'VCOL'::text      AS vcol,
       :'IDX'::text       AS idx,
       :HIGH_CACHE_MB::int AS high_cache_mb,
       :MAX_CLIENTS::int   AS max_clients,
       :WARM::int          AS warm,
       :TIMED::int         AS timed;

\echo ''
\echo '== running matrix (this opens up to MAX_CLIENTS concurrent backends per config) =='
DO $driver$
DECLARE
    cfg     ccbench_config%ROWTYPE;
    clients int[] := ARRAY[1, 2, 4, 8, 16];
    caches  int[];
    c  int;
    cm int;
    pw boolean;
    res jsonb;
BEGIN
    SELECT * INTO cfg FROM ccbench_config;
    caches := ARRAY[0, 512, cfg.high_cache_mb];
    FOREACH cm IN ARRAY caches LOOP
        FOREACH pw IN ARRAY ARRAY[false, true] LOOP
            FOREACH c IN ARRAY clients LOOP
                CONTINUE WHEN c > cfg.max_clients;
                res := ccbench_run_config(cfg.tbl, cfg.vcol, cfg.idx, c, cm, pw,
                                          cfg.warm, cfg.timed);
                INSERT INTO ccbench_results VALUES (cm, pw, c, res);
                RAISE NOTICE 'cache_mb=% prewarm=% clients=% -> rps=% p95=% p99=% mode=% waiting%%=%',
                    cm, pw, c,
                    res->>'rps', res->>'p95_ms', res->>'p99_ms',
                    res->>'native_cache_mode', res->>'waiting_pct';
            END LOOP;
        END LOOP;
    END LOOP;
END $driver$;

-- ---------------------------------------------------------------------------
-- DIAGNOSIS OUTPUT
-- ---------------------------------------------------------------------------
\echo ''
\echo '== 1) SCALING CURVE: RPS + latency vs clients (the collapse) =='
\echo '   cache_mb=0 is uncached (shared buffers); 512 default + HIGH are per-backend.'
\echo '   prewarm f=mode A (cold build counted), t=mode B (cache prebuilt).'
SELECT cache_mb,
       prewarm,
       clients,
       (m->>'native_cache_mode')        AS mode,
       (m->>'rps')::numeric             AS rps,
       (m->>'p50_ms')::numeric          AS p50_ms,
       (m->>'p95_ms')::numeric          AS p95_ms,
       (m->>'p99_ms')::numeric          AS p99_ms,
       (m->>'distinct_pids')::int       AS pids
FROM ccbench_results
ORDER BY cache_mb, prewarm, clients;

\echo ''
\echo '== 2) CAUSE ATTRIBUTION per config =='
\echo '   build_us: one-time cold per-backend build (mode A only).'
\echo '   waiting%/lock/lwlock: share of sampled active backends stalled on a wait.'
\echo '   code_pages: per-query code pages read (high => uncached page loading).'
\echo '   batch_us/total_us: per-query scoring/total CPU (rises => CPU contention).'
\echo '   cache/backend x clients = total_cache (memory duplicated across clients).'
SELECT cache_mb,
       prewarm,
       clients,
       (m->>'max_cold_build_us')::bigint                       AS cold_build_us,
       (m->>'waiting_pct')::numeric                            AS waiting_pct,
       (m->>'lock_obs')::int                                   AS lock,
       (m->>'lwlock_obs')::int                                 AS lwlock,
       (m->>'index_lock_waits')::int                           AS idx_lockw,
       (m->>'avg_code_pages_read')::numeric                    AS code_pages,
       (m->>'avg_batch_us')::numeric                           AS batch_us,
       (m->>'avg_total_us')::numeric                           AS total_us,
       pg_size_pretty((m->>'cache_bytes_per_backend')::bigint) AS cache_per_backend,
       pg_size_pretty((m->>'total_cache_bytes')::bigint)       AS total_cache
FROM ccbench_results
ORDER BY cache_mb, prewarm, clients;

\echo ''
\echo '== 3) VERDICT: dominant suspected cause per config (heuristic) =='
\echo '   Read against the 1-client baseline in the same cache/prewarm group.'
WITH base AS (
    SELECT cache_mb, prewarm,
           max((m->>'p95_ms')::numeric) FILTER (WHERE clients = 1) AS p95_1
    FROM ccbench_results GROUP BY cache_mb, prewarm
)
SELECT r.cache_mb,
       r.prewarm,
       r.clients,
       round((r.m->>'p95_ms')::numeric / NULLIF(b.p95_1, 0), 1) AS p95_blowup_x,
       CASE
         WHEN (r.m->>'non_native')::boolean THEN 'NOT native path (check plan)'
         WHEN r.clients = 1 THEN 'baseline'
         WHEN (r.m->>'waiting_pct')::numeric >= 25
              AND ((r.m->>'lock_obs')::int + (r.m->>'lwlock_obs')::int) > 0
              THEN 'lock/lwlock waits'
         WHEN NOT r.prewarm AND (r.m->>'max_cold_build_us')::bigint > 0
              AND (r.m->>'p95_ms')::numeric > 2 * COALESCE(b.p95_1, 0)
              THEN 'cold per-backend cache build'
         WHEN r.cache_mb > 0 AND (r.m->>'avg_code_pages_read')::numeric < 1
              AND (r.m->>'p95_ms')::numeric > 2 * COALESCE(b.p95_1, 0)
              THEN 'cache duplication / memory bandwidth'
         WHEN (r.m->>'avg_code_pages_read')::numeric >= 1
              AND (r.m->>'p95_ms')::numeric > 2 * COALESCE(b.p95_1, 0)
              THEN 'page loading (uncached)'
         WHEN (r.m->>'p95_ms')::numeric > 2 * COALESCE(b.p95_1, 0)
              THEN 'traversal/scoring CPU contention'
         ELSE 'scales OK'
       END AS suspected_cause
FROM ccbench_results r
JOIN base b ON b.cache_mb = r.cache_mb AND b.prewarm = r.prewarm
ORDER BY r.cache_mb, r.prewarm, r.clients;

-- Cleanup (helper functions + fixed-query table; synthetic data kept for reuse).
DROP FUNCTION IF EXISTS ccbench_run_config(text, text, text, int, int, boolean, int, int);
DROP FUNCTION IF EXISTS ccbench_client_run(text, text, int, boolean, int, int);
DROP TABLE IF EXISTS ccbench_queries;
\echo ''
\echo 'concurrency_dense_bench: done. (synthetic table ccbench_items, if built, is left for reuse;'
\echo ' DROP TABLE ccbench_items to remove it.)'
