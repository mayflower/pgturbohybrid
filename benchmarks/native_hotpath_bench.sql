-- Repeatable benchmark for the native turbohybrid AM hot path.
--
-- Builds realistic vector + tsvector indexes (native quantized storage),
-- confirms each scan goes through the native path (tqgraphgettuple /
-- PgturbohybridGraphCollectResults, i.e. scan_orchestration='graph_native'),
-- and times the scoring/traversal/rescore variants so CPU costs are visible.
-- The native-path guard makes it impossible to "optimize" a dormant path: any
-- benchmarked scan that is not graph_native aborts the run.
--
--   psql -d <db> -f benchmarks/native_hotpath_bench.sql
--
-- Configurable psql variables (override with -v NAME=VALUE):
--   NROWS          rows in each index            (default 10000)
--   DIMS           vector dimensions             (default 1536)
--   TIMED_QUERIES  timed queries per variant     (default 200)
--   WARM_QUERIES   warm-up queries per variant   (default 40)
--
-- Variants: dense-only, hybrid, U8-split on/off, the true x4 u8 batch vs four
-- single-node u8 passes (turbohybrid.dense_u8_batch_x4 on/off), signed split
-- forced, scalar/LUT forced, and exact-rescore off/auto/exact (on an
-- exact_storage index so the rescore band actually does work).
--
-- Per variant: wall p50/p95/p99; graph_scored_codes; the scoring-kernel bucket
-- counts; graph_batch_us (CPU scoring); graph_heap_us; graph_rescore_us;
-- graph_code_pages_read; graph_adj_pages_read; graph_visited_nodes; and the new
-- batch/arena diagnostics from turbohybrid_last_scan_stats():
--   graph_u8_batch_mode               -- x4 / single / none (which batch path ran)
--   dense_u8_batch_x4_enabled         -- the GUC value for this scan
--   graph_large_code_arena            -- code working set > 64 MB (RAM-bound)
--   graph_whole_code_prefetch_active  -- whole-code prefetch actually fired
--   graph_code_bytes                  -- per-code width
--   graph_code_arena_estimated_bytes  -- tqNodeCount * codeBytes
--
-- Derived metrics printed at the end:
--   batch_us_per_code        -- CPU scoring time per scored code
--   heap_us_per_visited_node -- heap fetch time per visited graph node
--   batch_nodes_per_call     -- average batch size fed to the batch scorer
--   code_arena_MB            -- estimated code working set in MB
--
-- LARGE-INDEX MODE (cache-resident vs large-arena/prefetch behaviour):
--   The default 10k/1536 run is cache-resident (graph_large_code_arena=false,
--   whole-code prefetch inactive) and stays quick for routine development.  To
--   exercise the large-arena / whole-code-prefetch path, build a multi-GB index
--   whose code working set exceeds the 64 MB threshold, e.g. 1M rows x 3072 dims:
--
--     psql -d <db> -v NROWS=1000000 -v DIMS=3072 \
--          -v TIMED_QUERIES=50 -v WARM_QUERIES=10 \
--          -f benchmarks/native_hotpath_bench.sql
--
--   WARNING: large mode is SLOW and HEAVY.  1M x 3072 builds two indexes (the
--   exact_storage copy also stores f32 vectors) and can take many minutes plus
--   tens of GB of disk and memory.  Run it deliberately -- never in CI or a
--   routine dev loop.  At that size graph_large_code_arena flips to true and
--   graph_whole_code_prefetch_active reports whether whole-code prefetch ran.

\set ON_ERROR_STOP on
\pset pager off

-- Defaults (applied only when not supplied on the command line via -v).
\if :{?NROWS}
\else
  \set NROWS 10000
\endif
\if :{?DIMS}
\else
  \set DIMS 1536
\endif
\if :{?TIMED_QUERIES}
\else
  \set TIMED_QUERIES 200
\endif
\if :{?WARM_QUERIES}
\else
  \set WARM_QUERIES 40
\endif

\echo 'native_hotpath_bench config:'
\echo '  NROWS         =' :NROWS
\echo '  DIMS          =' :DIMS
\echo '  TIMED_QUERIES =' :TIMED_QUERIES
\echo '  WARM_QUERIES  =' :WARM_QUERIES

CREATE EXTENSION IF NOT EXISTS pgturbohybrid CASCADE;

-- Exact-free (default) index: code-only native storage.
DROP TABLE IF EXISTS nhp_docs;
CREATE TABLE nhp_docs (id int PRIMARY KEY, embedding vector(:DIMS), body_tsv tsvector);
INSERT INTO nhp_docs
SELECT g,
       (SELECT array_agg((sin(g * 0.001 + d * 0.017) + 0.05 * sin(g * 0.7 + d))::real ORDER BY d)
        FROM generate_series(1, :DIMS) AS d)::real[]::vector,
       to_tsvector('english', 'document number ' || g)
FROM generate_series(1, :NROWS) AS g;
CREATE INDEX nhp_idx ON nhp_docs
  USING turbohybrid (embedding vector_cosine_turbohybrid_ops,
                     body_tsv bm25_tsvector_turbohybrid_ops)
  WITH (quantization_bits = 4);

-- exact_storage index: stores f32 vectors so the exact rescore band has work.
DROP TABLE IF EXISTS nhp_exact;
CREATE TABLE nhp_exact (id int PRIMARY KEY, embedding vector(:DIMS), body_tsv tsvector);
INSERT INTO nhp_exact SELECT id, embedding, body_tsv FROM nhp_docs;
CREATE INDEX nhp_exact_idx ON nhp_exact
  USING turbohybrid (embedding vector_cosine_turbohybrid_ops,
                     body_tsv bm25_tsvector_turbohybrid_ops)
  WITH (quantization_bits = 4, exact_storage = on);

ANALYZE nhp_docs;
ANALYZE nhp_exact;

SET enable_seqscan = off;
SET max_parallel_workers_per_gather = 0;
SET jit = off;

-- Confirm both indexes use native quantized storage before benchmarking.
DO $$
DECLARE st jsonb; t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['nhp_docs','nhp_exact'] LOOP
        EXECUTE format(
          'SELECT id FROM %I ORDER BY embedding <~> turbohybrid_query(vector_query => $1) LIMIT 10', t)
          USING (SELECT embedding FROM nhp_docs WHERE id = 1);
        st := turbohybrid_last_scan_stats();
        IF st->>'scan_orchestration' <> 'graph_native'
           OR st->>'graph_storage_kind' <> 'pgturbohybrid_graph_native' THEN
            RAISE EXCEPTION '% is not native quantized storage (orchestration=%, storage=%)',
                t, st->>'scan_orchestration', st->>'graph_storage_kind';
        END IF;
    END LOOP;
    RAISE NOTICE 'both indexes confirmed native (graph_native)';
END $$;

CREATE TEMP TABLE nhp_results (
    seq int, variant text, scan_path text,
    p50_ms float8, p95_ms float8, p99_ms float8,
    scored_codes bigint, hot_kernel text,
    batch_us bigint, heap_us bigint, rescore_us bigint,
    code_pages_read bigint, adj_pages_read bigint, visited_nodes bigint,
    -- U8 batch + whole-code prefetch diagnostics (turbohybrid_last_scan_stats()).
    u8_batch_mode text, u8_batch_x4_enabled bool,
    large_code_arena bool, whole_code_prefetch_active bool,
    code_bytes bigint, code_arena_estimated_bytes bigint,
    -- For the batch_nodes_per_call derived metric.
    batch_calls bigint, batch_nodes bigint,
    kernels jsonb
);

CREATE OR REPLACE FUNCTION nhp_bench(p_seq int, p_label text, p_tbl text, p_hybrid bool,
                                     p_simd text, p_u8 text, p_impl text, p_rescore text,
                                     p_x4 text, p_nrows int, p_warm int, p_timed int) RETURNS void
LANGUAGE plpgsql AS $$
DECLARE
    qv vector; t0 timestamptz; k int;
    durs float8[] := '{}';
    qtext text; st jsonb; ks jsonb;
    hot_kernel text := 'none'; hot_nodes bigint := -1; kname text;
BEGIN
    PERFORM set_config('turbohybrid.simd', p_simd, false);
    PERFORM set_config('turbohybrid.dense_u8_split', p_u8, false);
    PERFORM set_config('turbohybrid.dense_query_split_impl', p_impl, false);
    PERFORM set_config('turbohybrid.dense_rescore_band', p_rescore, false);
    PERFORM set_config('turbohybrid.dense_u8_batch_x4', p_x4, false);

    IF p_hybrid THEN
        qtext := format('SELECT id FROM %I ORDER BY embedding <~> turbohybrid_query('
                        'vector_query => $1, text_query => to_tsquery(''english'',''document'')) LIMIT 10', p_tbl);
    ELSE
        qtext := format('SELECT id FROM %I ORDER BY embedding <~> turbohybrid_query('
                        'vector_query => $1) LIMIT 10', p_tbl);
    END IF;

    FOR k IN 1..p_warm LOOP   -- warm
        EXECUTE format('SELECT embedding FROM %I WHERE id = $1', p_tbl)
            INTO qv USING (k * 131) % p_nrows + 1;
        EXECUTE qtext USING qv;
    END LOOP;

    FOR k IN 1..p_timed LOOP  -- timed
        EXECUTE format('SELECT embedding FROM %I WHERE id = $1', p_tbl)
            INTO qv USING (k * 131) % p_nrows + 1;
        t0 := clock_timestamp();
        EXECUTE qtext USING qv;
        durs := array_append(durs, extract(epoch FROM clock_timestamp() - t0) * 1000);
    END LOOP;

    st := turbohybrid_last_scan_stats();

    -- Native-path guard: refuse to record a dormant-path measurement.
    IF st->>'scan_orchestration' <> 'graph_native' THEN
        RAISE EXCEPTION '[%] scan did NOT use the native path (scan_orchestration=%) -- refusing to benchmark a dormant path',
            p_label, st->>'scan_orchestration';
    END IF;

    ks := st->'graph_score_kernels';
    FOR kname IN SELECT jsonb_object_keys(ks) LOOP
        IF (ks->kname->>'nodes')::bigint > hot_nodes THEN
            hot_nodes := (ks->kname->>'nodes')::bigint;
            hot_kernel := kname;
        END IF;
    END LOOP;

    INSERT INTO nhp_results VALUES (
        p_seq, p_label, st->>'scan_orchestration',
        (SELECT percentile_cont(0.5)  WITHIN GROUP (ORDER BY x) FROM unnest(durs) x),
        (SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY x) FROM unnest(durs) x),
        (SELECT percentile_cont(0.99) WITHIN GROUP (ORDER BY x) FROM unnest(durs) x),
        (st->>'graph_scored_codes')::bigint,
        hot_kernel || ':' || hot_nodes,
        (st->>'graph_batch_us')::bigint,
        (st->>'graph_heap_us')::bigint,
        (st->>'graph_rescore_us')::bigint,
        (st->>'graph_code_pages_read')::bigint,
        (st->>'graph_adj_pages_read')::bigint,
        (st->>'graph_visited_nodes')::bigint,
        st->>'graph_u8_batch_mode',
        (st->>'dense_u8_batch_x4_enabled')::bool,
        (st->>'graph_large_code_arena')::bool,
        (st->>'graph_whole_code_prefetch_active')::bool,
        (st->>'graph_code_bytes')::bigint,
        (st->>'graph_code_arena_estimated_bytes')::bigint,
        (st->>'graph_batch_calls')::bigint,
        (st->>'graph_batch_nodes')::bigint,
        ks);
END $$;

-- p_seq, label, table, hybrid, simd, u8, impl, rescore, x4, nrows, warm, timed
SELECT nhp_bench(1,  'dense_only',    'nhp_docs',  false, 'on',  'auto', 'auto',   'auto',  'on',  :NROWS, :WARM_QUERIES, :TIMED_QUERIES);
SELECT nhp_bench(2,  'hybrid',        'nhp_docs',  true,  'on',  'auto', 'auto',   'auto',  'on',  :NROWS, :WARM_QUERIES, :TIMED_QUERIES);
SELECT nhp_bench(3,  'u8_split_on',   'nhp_docs',  false, 'on',  'on',   'auto',   'auto',  'on',  :NROWS, :WARM_QUERIES, :TIMED_QUERIES);
SELECT nhp_bench(4,  'u8_split_off',  'nhp_docs',  false, 'on',  'off',  'auto',   'auto',  'on',  :NROWS, :WARM_QUERIES, :TIMED_QUERIES);
-- x4 batch vs four single-node u8 passes (u8 forced on so the batch scorer runs).
SELECT nhp_bench(5,  'u8_x4_on',      'nhp_docs',  false, 'on',  'on',   'auto',   'auto',  'on',  :NROWS, :WARM_QUERIES, :TIMED_QUERIES);
SELECT nhp_bench(6,  'u8_x4_off',     'nhp_docs',  false, 'on',  'on',   'auto',   'auto',  'off', :NROWS, :WARM_QUERIES, :TIMED_QUERIES);
SELECT nhp_bench(7,  'signed_forced', 'nhp_docs',  false, 'on',  'auto', 'signed', 'auto',  'on',  :NROWS, :WARM_QUERIES, :TIMED_QUERIES);
SELECT nhp_bench(8,  'scalar_lut',    'nhp_docs',  false, 'off', 'auto', 'auto',   'auto',  'on',  :NROWS, :WARM_QUERIES, :TIMED_QUERIES);
SELECT nhp_bench(9,  'rescore_off',   'nhp_exact', false, 'on',  'auto', 'auto',   'off',   'on',  :NROWS, :WARM_QUERIES, :TIMED_QUERIES);
SELECT nhp_bench(10, 'rescore_auto',  'nhp_exact', false, 'on',  'auto', 'auto',   'auto',  'on',  :NROWS, :WARM_QUERIES, :TIMED_QUERIES);
SELECT nhp_bench(11, 'rescore_exact', 'nhp_exact', false, 'on',  'auto', 'auto',   'exact', 'on',  :NROWS, :WARM_QUERIES, :TIMED_QUERIES);

RESET turbohybrid.simd;
RESET turbohybrid.dense_u8_split;
RESET turbohybrid.dense_query_split_impl;
RESET turbohybrid.dense_rescore_band;
RESET turbohybrid.dense_u8_batch_x4;

\echo '== latency + traversal/scoring/rescore CPU per variant (all scan_path must be graph_native) =='
SELECT variant, scan_path,
       round(p50_ms::numeric,4) p50, round(p95_ms::numeric,4) p95, round(p99_ms::numeric,4) p99,
       scored_codes scored, batch_us, heap_us, rescore_us,
       code_pages_read cpr, adj_pages_read apr, visited_nodes visited
FROM nhp_results ORDER BY seq;

\echo '== U8 batch mode + code-arena / whole-code-prefetch state per variant =='
SELECT variant,
       u8_batch_mode AS batch_mode, u8_batch_x4_enabled AS x4_enabled,
       large_code_arena AS large_arena, whole_code_prefetch_active AS prefetch,
       code_bytes,
       round(code_arena_estimated_bytes / 1048576.0, 2) AS code_arena_mb
FROM nhp_results ORDER BY seq;

\echo '== derived per-code / per-node / per-call metrics =='
SELECT variant,
       round(CASE WHEN scored_codes > 0 THEN batch_us::numeric / scored_codes END, 5) AS batch_us_per_code,
       round(CASE WHEN visited_nodes > 0 THEN heap_us::numeric / visited_nodes END, 5) AS heap_us_per_visited_node,
       round(CASE WHEN batch_calls > 0 THEN batch_nodes::numeric / batch_calls END, 3) AS batch_nodes_per_call,
       round(code_arena_estimated_bytes / 1048576.0, 2) AS code_arena_mb
FROM nhp_results ORDER BY seq;

\echo '== dominant scoring kernel per variant =='
SELECT variant, hot_kernel FROM nhp_results ORDER BY seq;

\echo '== full scoring-kernel bucket counts (nonzero) per variant =='
SELECT r.variant, k.key AS kernel,
       (r.kernels->k.key->>'nodes')::bigint AS nodes,
       (r.kernels->k.key->>'calls')::bigint AS calls
FROM nhp_results r, LATERAL jsonb_object_keys(r.kernels) k(key)
WHERE (r.kernels->k.key->>'nodes')::bigint > 0
ORDER BY r.seq, nodes DESC;

DROP FUNCTION nhp_bench(int, text, text, bool, text, text, text, text, text, int, int, int);
DROP TABLE nhp_docs;
DROP TABLE nhp_exact;
RESET enable_seqscan;
RESET max_parallel_workers_per_gather;
RESET jit;
