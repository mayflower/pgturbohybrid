-- Scaling smoke for single-row insert and reciprocal-neighbor update cost.
--
-- Builds a deterministic native graph, then inserts several small batches while
-- reporting elapsed time and index stats after each batch.  If the extension is
-- built with DEBUG1 reciprocal-neighbor instrumentation, set DEBUG_INSERT=1 to
-- show those logs in the psql stream.
--
-- Expected trend: per-row insert cost should stay roughly stable as the index
-- grows over these small batches.  A sharp slope change suggests reciprocal
-- adjacency updates fell back to page-chain scanning or another O(N) path.
--
-- Usage:
--   psql -d <db> -f benchmarks/dev/insert_scaling_bench.sql
--   psql -d <db> -v BASE_ROWS=10000 -v BATCH_ROWS=100 -v BATCHES=5 \
--        -f benchmarks/dev/insert_scaling_bench.sql
--   psql -d <db> -v DEBUG_INSERT=1 -f benchmarks/dev/insert_scaling_bench.sql

\set ON_ERROR_STOP on
\pset pager off
\x auto
\timing on

\if :{?BASE_ROWS}
\else
  \set BASE_ROWS 3000
\endif
\if :{?BATCH_ROWS}
\else
  \set BATCH_ROWS 50
\endif
\if :{?BATCHES}
\else
  \set BATCHES 4
\endif
\if :{?DIMS}
\else
  \set DIMS 16
\endif
\if :{?DEBUG_INSERT}
\else
  \set DEBUG_INSERT 0
\endif

\echo 'insert_scaling_bench config:'
\echo '  BASE_ROWS    =' :BASE_ROWS
\echo '  BATCH_ROWS   =' :BATCH_ROWS
\echo '  BATCHES      =' :BATCHES
\echo '  DIMS         =' :DIMS
\echo '  DEBUG_INSERT =' :DEBUG_INSERT

CREATE EXTENSION IF NOT EXISTS pgturbohybrid CASCADE;

DROP TABLE IF EXISTS th_insert_scaling_docs;
CREATE TABLE th_insert_scaling_docs (
    id int PRIMARY KEY,
    grp int NOT NULL,
    embedding vector(:DIMS) NOT NULL
);

INSERT INTO th_insert_scaling_docs
SELECT g,
       g % 64,
       (
           SELECT array_agg(
                      (sin(g * 0.019 + d * 0.037) +
                       0.15 * cos(g * 0.003 + d * 0.29))::real
                      ORDER BY d)
           FROM generate_series(1, :DIMS) AS d
       )::real[]::vector
FROM generate_series(1, :BASE_ROWS) AS g;

CREATE INDEX th_insert_scaling_idx ON th_insert_scaling_docs
USING turbohybrid (embedding vector_cosine_turbohybrid_ops) INCLUDE (grp)
WITH (quantization_bits = 4, exact_storage = off);

ANALYZE th_insert_scaling_docs;

SET enable_seqscan = off;
SET max_parallel_workers_per_gather = 0;
SET jit = off;

\if :DEBUG_INSERT
SET client_min_messages = debug1;
\endif

SELECT jsonb_pretty(turbohybrid_index_stats('th_insert_scaling_idx'::regclass))
    AS initial_index_stats;

CREATE TEMP TABLE th_insert_scaling_config AS
SELECT :BASE_ROWS::int AS base_rows,
       :BATCH_ROWS::int AS batch_rows,
       :BATCHES::int AS batches,
       :DIMS::int AS dims;

CREATE TEMP TABLE th_insert_scaling_results (
    batch_no int NOT NULL,
    rows_inserted int NOT NULL,
    elapsed_ms float8 NOT NULL,
    ms_per_row float8 NOT NULL,
    total_rows int NOT NULL,
    index_stats jsonb NOT NULL,
    probe_stats jsonb NOT NULL
);

DO $$
DECLARE
    b int;
    first_id int;
    last_id int;
    t0 timestamptz;
    elapsed float8;
    total int;
    st jsonb;
    scan_st jsonb;
    qv vector;
    base_rows int;
    batch_rows int;
    batches int;
    dims int;
BEGIN
    SELECT c.base_rows, c.batch_rows, c.batches, c.dims
    INTO base_rows, batch_rows, batches, dims
    FROM th_insert_scaling_config c;

    FOR b IN 1..batches LOOP
        first_id := base_rows + ((b - 1) * batch_rows) + 1;
        last_id := base_rows + (b * batch_rows);
        t0 := clock_timestamp();

        INSERT INTO th_insert_scaling_docs
        SELECT g,
               g % 64,
               (
                   SELECT array_agg(
                              (sin(g * 0.019 + d * 0.037) +
                               0.15 * cos(g * 0.003 + d * 0.29))::real
                              ORDER BY d)
                   FROM generate_series(1, dims) AS d
               )::real[]::vector
        FROM generate_series(first_id, last_id) AS g;

        elapsed := extract(epoch FROM clock_timestamp() - t0) * 1000.0;
        SELECT count(*) INTO total FROM th_insert_scaling_docs;
        st := turbohybrid_index_stats('th_insert_scaling_idx'::regclass);

        SELECT embedding INTO qv
        FROM th_insert_scaling_docs
        WHERE id = first_id;

        PERFORM id
        FROM th_insert_scaling_docs
        ORDER BY embedding <~> turbohybrid_query(
            vector_query => qv,
            dense_k => 20,
            final_k => 5
        )
        LIMIT 5;

        scan_st := turbohybrid_last_scan_stats();

        INSERT INTO th_insert_scaling_results
        VALUES (
            b,
            batch_rows,
            elapsed,
            elapsed / GREATEST(batch_rows, 1),
            total,
            jsonb_build_object(
                'node_count', COALESCE(st->'node_count', to_jsonb(total)),
                'graph_m', st->'graph_m',
                'graph_ef_construction', st->'graph_ef_construction',
                'quantization_bits', st->'quantization_bits',
                'exact_storage', st->'exact_storage',
                'index_shape', st->'index_shape',
                'native_segments', st->'native_segments',
                'storage_kind', st->'storage_kind',
                'build_neighbor_select', st->'build_neighbor_select'
            ),
            jsonb_build_object(
                'scan_orchestration', scan_st->'scan_orchestration',
                'graph_scored_codes', scan_st->'graph_scored_codes',
                'graph_visited_nodes', scan_st->'graph_visited_nodes',
                'native_cache_used', scan_st->'native_cache_used',
                'native_cache_reused', scan_st->'native_cache_reused',
                'dense_elapsed_us', scan_st->'dense_elapsed_us'
            )
        );
    END LOOP;
END
$$;

SELECT batch_no,
       rows_inserted,
       total_rows,
       round(elapsed_ms::numeric, 3) AS elapsed_ms,
       round(ms_per_row::numeric, 3) AS ms_per_row,
       jsonb_pretty(index_stats) AS index_stats,
       jsonb_pretty(probe_stats) AS post_insert_probe_stats
FROM th_insert_scaling_results
ORDER BY batch_no;

RESET enable_seqscan;
RESET max_parallel_workers_per_gather;
RESET jit;
\if :DEBUG_INSERT
RESET client_min_messages;
\endif
