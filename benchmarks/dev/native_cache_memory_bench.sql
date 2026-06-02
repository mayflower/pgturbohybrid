-- Scaling smoke for native graph cache memory and cold/warm cache behavior.
--
-- Builds a deterministic dense-only index, prints turbohybrid_estimate_memory(),
-- and compares the first per-backend scan against a second warm scan in the
-- same backend.  It also runs one uncached scan for comparison.
--
-- Expected trend: the first per_backend scan should show native_cache_used=true
-- and native_cache_built_this_scan=true; the second should show reuse.  The
-- off scan avoids resident native cache but may perform more page work.  Timing
-- depends on hardware and cache state, so use native_cache_* fields to diagnose.
--
-- Usage:
--   psql -d <db> -f benchmarks/dev/native_cache_memory_bench.sql
--   psql -d <db> -v NROWS=50000 -v DIMS=32 \
--        -f benchmarks/dev/native_cache_memory_bench.sql

\set ON_ERROR_STOP on
\pset pager off
\x auto
\timing on

\if :{?NROWS}
\else
  \set NROWS 5000
\endif
\if :{?DIMS}
\else
  \set DIMS 32
\endif
\if :{?DENSE_K}
\else
  \set DENSE_K 100
\endif
\if :{?FINAL_K}
\else
  \set FINAL_K 10
\endif

\echo 'native_cache_memory_bench config:'
\echo '  NROWS   =' :NROWS
\echo '  DIMS    =' :DIMS
\echo '  DENSE_K =' :DENSE_K
\echo '  FINAL_K =' :FINAL_K

CREATE EXTENSION IF NOT EXISTS pgturbohybrid CASCADE;

DROP TABLE IF EXISTS th_native_cache_memory_docs;
CREATE TABLE th_native_cache_memory_docs (
    id int PRIMARY KEY,
    embedding vector(:DIMS) NOT NULL
);

INSERT INTO th_native_cache_memory_docs
SELECT g,
       (
           SELECT array_agg(
                      (sin(g * 0.011 + d * 0.023) +
                       0.25 * cos(g * 0.005 + d * 0.19))::real
                      ORDER BY d)
           FROM generate_series(1, :DIMS) AS d
       )::real[]::vector
FROM generate_series(1, :NROWS) AS g;

CREATE INDEX th_native_cache_memory_idx ON th_native_cache_memory_docs
USING turbohybrid (embedding vector_cosine_turbohybrid_ops)
WITH (quantization_bits = 4, exact_storage = off);

ANALYZE th_native_cache_memory_docs;

SELECT jsonb_pretty(turbohybrid_estimate_memory('th_native_cache_memory_idx'::regclass))
    AS estimated_memory;

SET enable_seqscan = off;
SET max_parallel_workers_per_gather = 0;
SET jit = off;

CREATE TEMP TABLE th_native_cache_memory_config AS
SELECT :DENSE_K::int AS dense_k,
       :FINAL_K::int AS final_k;

CREATE TEMP TABLE th_native_cache_memory_results (
    scenario text NOT NULL,
    elapsed_ms float8 NOT NULL,
    rows_returned int NOT NULL,
    stats jsonb NOT NULL
);

DO $$
DECLARE
    qv vector;
    t0 timestamptz;
    n int;
    st jsonb;
    dense_k int;
    final_k int;
BEGIN
    SELECT c.dense_k, c.final_k INTO dense_k, final_k
    FROM th_native_cache_memory_config c;

    SELECT embedding INTO qv
    FROM th_native_cache_memory_docs
    WHERE id = 1;

    PERFORM set_config('turbohybrid.native_cache_scope', 'per_backend', true);
    t0 := clock_timestamp();
    SELECT count(*) INTO n
    FROM (
        SELECT id
        FROM th_native_cache_memory_docs
        ORDER BY embedding <~> turbohybrid_query(
            vector_query => qv,
            dense_k => dense_k,
            final_k => final_k
        )
        LIMIT final_k
    ) s;
    st := turbohybrid_last_scan_stats();
    INSERT INTO th_native_cache_memory_results
    VALUES (
        'cold_per_backend',
        extract(epoch FROM clock_timestamp() - t0) * 1000.0,
        n,
        jsonb_build_object(
            'native_cache_policy', st->'native_cache_policy',
            'native_cache_scope', st->'native_cache_scope',
            'native_cache_mode', st->'native_cache_mode',
            'native_cache_used', st->'native_cache_used',
            'native_cache_reused', st->'native_cache_reused',
            'native_cache_built_this_scan', st->'native_cache_built_this_scan',
            'native_cache_build_us', st->'native_cache_build_us',
            'native_cache_attach_us', st->'native_cache_attach_us',
            'native_cache_wait_us', st->'native_cache_wait_us',
            'native_cache_refcount', st->'native_cache_refcount',
            'native_cache_bytes', st->'native_cache_bytes',
            'native_cache_code_bytes', st->'native_cache_code_bytes',
            'native_cache_adj_bytes', st->'native_cache_adj_bytes',
            'native_cache_exact_bytes', st->'native_cache_exact_bytes',
            'native_cache_warning', st->'native_cache_warning',
            'native_cache_warning_reason', st->'native_cache_warning_reason',
            'graph_code_pages_read', st->'graph_code_pages_read',
            'graph_adj_pages_read', st->'graph_adj_pages_read',
            'graph_scored_codes', st->'graph_scored_codes',
            'dense_elapsed_us', st->'dense_elapsed_us'
        )
    );

    t0 := clock_timestamp();
    SELECT count(*) INTO n
    FROM (
        SELECT id
        FROM th_native_cache_memory_docs
        ORDER BY embedding <~> turbohybrid_query(
            vector_query => qv,
            dense_k => dense_k,
            final_k => final_k
        )
        LIMIT final_k
    ) s;
    st := turbohybrid_last_scan_stats();
    INSERT INTO th_native_cache_memory_results
    VALUES (
        'warm_per_backend',
        extract(epoch FROM clock_timestamp() - t0) * 1000.0,
        n,
        jsonb_build_object(
            'native_cache_policy', st->'native_cache_policy',
            'native_cache_scope', st->'native_cache_scope',
            'native_cache_mode', st->'native_cache_mode',
            'native_cache_used', st->'native_cache_used',
            'native_cache_reused', st->'native_cache_reused',
            'native_cache_built_this_scan', st->'native_cache_built_this_scan',
            'native_cache_build_us', st->'native_cache_build_us',
            'native_cache_attach_us', st->'native_cache_attach_us',
            'native_cache_wait_us', st->'native_cache_wait_us',
            'native_cache_refcount', st->'native_cache_refcount',
            'native_cache_bytes', st->'native_cache_bytes',
            'native_cache_code_bytes', st->'native_cache_code_bytes',
            'native_cache_adj_bytes', st->'native_cache_adj_bytes',
            'native_cache_exact_bytes', st->'native_cache_exact_bytes',
            'native_cache_warning', st->'native_cache_warning',
            'native_cache_warning_reason', st->'native_cache_warning_reason',
            'graph_code_pages_read', st->'graph_code_pages_read',
            'graph_adj_pages_read', st->'graph_adj_pages_read',
            'graph_scored_codes', st->'graph_scored_codes',
            'dense_elapsed_us', st->'dense_elapsed_us'
        )
    );

    PERFORM set_config('turbohybrid.native_cache_scope', 'off', true);
    t0 := clock_timestamp();
    SELECT count(*) INTO n
    FROM (
        SELECT id
        FROM th_native_cache_memory_docs
        ORDER BY embedding <~> turbohybrid_query(
            vector_query => qv,
            dense_k => dense_k,
            final_k => final_k
        )
        LIMIT final_k
    ) s;
    st := turbohybrid_last_scan_stats();
    INSERT INTO th_native_cache_memory_results
    VALUES (
        'native_cache_off',
        extract(epoch FROM clock_timestamp() - t0) * 1000.0,
        n,
        jsonb_build_object(
            'native_cache_policy', st->'native_cache_policy',
            'native_cache_scope', st->'native_cache_scope',
            'native_cache_mode', st->'native_cache_mode',
            'native_cache_used', st->'native_cache_used',
            'native_cache_reused', st->'native_cache_reused',
            'native_cache_built_this_scan', st->'native_cache_built_this_scan',
            'native_cache_build_us', st->'native_cache_build_us',
            'native_cache_attach_us', st->'native_cache_attach_us',
            'native_cache_wait_us', st->'native_cache_wait_us',
            'native_cache_refcount', st->'native_cache_refcount',
            'native_cache_bytes', st->'native_cache_bytes',
            'native_cache_code_bytes', st->'native_cache_code_bytes',
            'native_cache_adj_bytes', st->'native_cache_adj_bytes',
            'native_cache_exact_bytes', st->'native_cache_exact_bytes',
            'native_cache_warning', st->'native_cache_warning',
            'native_cache_warning_reason', st->'native_cache_warning_reason',
            'graph_code_pages_read', st->'graph_code_pages_read',
            'graph_adj_pages_read', st->'graph_adj_pages_read',
            'graph_scored_codes', st->'graph_scored_codes',
            'dense_elapsed_us', st->'dense_elapsed_us'
        )
    );
END
$$;

SELECT scenario,
       round(elapsed_ms::numeric, 3) AS elapsed_ms,
       rows_returned,
       jsonb_pretty(stats) AS scan_stats
FROM th_native_cache_memory_results
ORDER BY scenario;

RESET turbohybrid.native_cache_scope;
RESET enable_seqscan;
RESET max_parallel_workers_per_gather;
RESET jit;
