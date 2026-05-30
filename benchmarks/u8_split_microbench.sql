-- Microbenchmark for the native 4-bit dense scoring kernels.
--
-- Builds a 10k-row 1536-dim 4-bit index and times warm nearest-neighbor scans
-- under each scoring path, interleaved per query to cancel drift:
--   scalar_lut   -- turbohybrid.simd = off  (per-dim float LUT gather; baseline)
--   signed_split -- signed-codebook query split  (the fallback / baseline kernel)
--   u8_avx2      -- unsigned-codebook split, AVX2 maddubs + madd
--   u8_avx512    -- unsigned-codebook split, AVX-512 VNNI (dpbusd), where available
--
-- Reports warm p50/p95 per mode and the speedups, plus codes scored per query
-- (turbohybrid_last_scan_stats()->graph_scored_codes) so the kernel work is
-- visible.  Run:  psql -d <db> -f benchmarks/u8_split_microbench.sql
--
-- Acceptance reference: u8 should beat scalar/LUT by a wide margin and be at
-- least as fast as the signed split (faster on AVX2-only hosts; on AVX-512-VNNI
-- both signed and u8 use dpbusd and are close).

\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS pgturbohybrid CASCADE;

DROP TABLE IF EXISTS u8mb_docs;
CREATE TABLE u8mb_docs (id int, embedding vector(1536), body_tsv tsvector);
INSERT INTO u8mb_docs
SELECT g,
       (SELECT array_agg((sin(g * 0.001 + d * 0.017) + 0.05 * sin(g * 0.7 + d))::real ORDER BY d)
        FROM generate_series(1, 1536) AS d)::real[]::vector,
       to_tsvector('english', 'document number ' || g)
FROM generate_series(1, 10000) AS g;

CREATE INDEX u8mb_idx ON u8mb_docs
  USING turbohybrid (embedding vector_cosine_turbohybrid_ops,
                     body_tsv bm25_tsvector_turbohybrid_ops)
  WITH (quantization_bits = 4);
ANALYZE u8mb_docs;

SET enable_seqscan = off;
SET max_parallel_workers_per_gather = 0;
SET jit = off;

CREATE OR REPLACE FUNCTION u8mb_pctl(arr float8[], q float8) RETURNS float8
LANGUAGE sql AS $f$ SELECT percentile_cont(q) WITHIN GROUP (ORDER BY x) FROM unnest(arr) x $f$;

DO $$
DECLARE
    qv vector; t0 timestamptz; k int; c int;
    scl float8[] := '{}';        -- scalar/LUT
    sa float8[] := '{}';         -- signed split, AVX2
    ua float8[] := '{}';         -- u8 split, AVX2
    sv float8[] := '{}';         -- signed split, AVX-512 VNNI
    uv float8[] := '{}';         -- u8 split, AVX-512 VNNI
    scored bigint := 0;
BEGIN
    -- Warm pages/cache.
    FOR k IN 1..50 LOOP
        PERFORM id FROM u8mb_docs
          ORDER BY embedding <~> turbohybrid_query(
            vector_query => (SELECT embedding FROM u8mb_docs WHERE id = (k * 131) % 10000 + 1)) LIMIT 10;
    END LOOP;

    FOR k IN 1..400 LOOP
        c := (k * 131) % 10000 + 1;
        qv := (SELECT embedding FROM u8mb_docs WHERE id = c);

        -- scalar/LUT
        PERFORM set_config('turbohybrid.simd', 'off', true);
        t0 := clock_timestamp();
        PERFORM id FROM u8mb_docs ORDER BY embedding <~> turbohybrid_query(vector_query => qv) LIMIT 10;
        scl := array_append(scl, extract(epoch FROM clock_timestamp() - t0) * 1000);
        PERFORM set_config('turbohybrid.simd', 'on', true);

        -- AVX2 tier: disable VNNI/AVX-VNNI so both kernels run on AVX2.
        PERFORM set_config('turbohybrid.dense_graph_avx512vnni', 'off', true);
        PERFORM set_config('turbohybrid.dense_graph_avxvnni', 'off', true);

        PERFORM set_config('turbohybrid.dense_u8_split', 'off', true);
        PERFORM set_config('turbohybrid.dense_query_split_impl', 'signed', true);
        t0 := clock_timestamp();
        PERFORM id FROM u8mb_docs ORDER BY embedding <~> turbohybrid_query(vector_query => qv) LIMIT 10;
        sa := array_append(sa, extract(epoch FROM clock_timestamp() - t0) * 1000);

        PERFORM set_config('turbohybrid.dense_query_split_impl', 'auto', true);
        PERFORM set_config('turbohybrid.dense_u8_split', 'on', true);
        t0 := clock_timestamp();
        PERFORM id FROM u8mb_docs ORDER BY embedding <~> turbohybrid_query(vector_query => qv) LIMIT 10;
        ua := array_append(ua, extract(epoch FROM clock_timestamp() - t0) * 1000);

        -- AVX-512 VNNI tier (if available): both kernels use dpbusd.
        PERFORM set_config('turbohybrid.dense_graph_avx512vnni', 'on', true);
        PERFORM set_config('turbohybrid.dense_graph_avxvnni', 'on', true);

        PERFORM set_config('turbohybrid.dense_u8_split', 'off', true);
        PERFORM set_config('turbohybrid.dense_query_split_impl', 'signed', true);
        t0 := clock_timestamp();
        PERFORM id FROM u8mb_docs ORDER BY embedding <~> turbohybrid_query(vector_query => qv) LIMIT 10;
        sv := array_append(sv, extract(epoch FROM clock_timestamp() - t0) * 1000);

        PERFORM set_config('turbohybrid.dense_query_split_impl', 'auto', true);
        PERFORM set_config('turbohybrid.dense_u8_split', 'on', true);
        t0 := clock_timestamp();
        PERFORM id FROM u8mb_docs ORDER BY embedding <~> turbohybrid_query(vector_query => qv) LIMIT 10;
        uv := array_append(uv, extract(epoch FROM clock_timestamp() - t0) * 1000);
        scored := scored + (turbohybrid_last_scan_stats() ->> 'graph_scored_codes')::bigint;
    END LOOP;

    RESET turbohybrid.dense_u8_split;
    RESET turbohybrid.dense_query_split_impl;
    RESET turbohybrid.dense_graph_avx512vnni;
    RESET turbohybrid.dense_graph_avxvnni;

    RAISE NOTICE 'queries=400, avg codes scored/query=%', scored / 400;
    RAISE NOTICE 'p50 ms: scalar_lut=%  | AVX2: signed=% u8=%  | VNNI: signed=% u8=%',
        round(u8mb_pctl(scl,0.5)::numeric,4),
        round(u8mb_pctl(sa,0.5)::numeric,4), round(u8mb_pctl(ua,0.5)::numeric,4),
        round(u8mb_pctl(sv,0.5)::numeric,4), round(u8mb_pctl(uv,0.5)::numeric,4);
    RAISE NOTICE 'p95 ms: scalar_lut=%  | AVX2: signed=% u8=%  | VNNI: signed=% u8=%',
        round(u8mb_pctl(scl,0.95)::numeric,4),
        round(u8mb_pctl(sa,0.95)::numeric,4), round(u8mb_pctl(ua,0.95)::numeric,4),
        round(u8mb_pctl(sv,0.95)::numeric,4), round(u8mb_pctl(uv,0.95)::numeric,4);
    RAISE NOTICE 'p50: u8_avx2 vs signed_avx2=%x | u8_vnni vs signed_vnni=%x | u8(best) vs scalar/LUT=%x',
        round((u8mb_pctl(sa,0.5)/u8mb_pctl(ua,0.5))::numeric,2),
        round((u8mb_pctl(sv,0.5)/u8mb_pctl(uv,0.5))::numeric,2),
        round((u8mb_pctl(scl,0.5)/LEAST(u8mb_pctl(ua,0.5),u8mb_pctl(uv,0.5)))::numeric,2);
END $$;

DROP FUNCTION u8mb_pctl(float8[], float8);
DROP TABLE u8mb_docs;
RESET enable_seqscan;
RESET max_parallel_workers_per_gather;
RESET jit;
