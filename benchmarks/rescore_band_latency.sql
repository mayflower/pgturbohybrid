-- Latency cost of exact f32 dense rescore, by band, on a native graph index.
--
-- Builds an exact_storage 1536-dim 4-bit index (so exact rescore has real
-- work) and times warm scans under turbohybrid.dense_rescore_band = off / auto
-- / exact, interleaved per query to cancel drift.  Reports warm p50/p95 and the
-- rescore activity (graph_effective_rescore_band / graph_rescore_count) per
-- band so the cost is visible.
--
--   psql -d <db> -f benchmarks/rescore_band_latency.sql
--
-- Takeaway: 'off' pays no exact-rescore cost; 'exact' (and 'auto' here, since a
-- high-dim cosine exact_storage index resolves AUTO to a full rescore) pays the
-- f32 reread.  A DEFAULT exact-free (code-only) index -- what the latency
-- profile builds unless exact_storage is requested -- pays 0 regardless; this
-- script uses exact_storage purely to make the rescore cost measurable.

\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS pgturbohybrid CASCADE;

DROP TABLE IF EXISTS rbb_docs;
CREATE TABLE rbb_docs (id int, embedding vector(1536), body_tsv tsvector);
INSERT INTO rbb_docs
SELECT g,
       (SELECT array_agg((sin(g * 0.001 + d * 0.017) + 0.05 * sin(g * 0.7 + d))::real ORDER BY d)
        FROM generate_series(1, 1536) AS d)::real[]::vector,
       to_tsvector('english', 'document number ' || g)
FROM generate_series(1, 5000) AS g;

CREATE INDEX rbb_idx ON rbb_docs
  USING turbohybrid (embedding vector_cosine_turbohybrid_ops,
                     body_tsv bm25_tsvector_turbohybrid_ops)
  WITH (quantization_bits = 4, exact_storage = on);
ANALYZE rbb_docs;

SET enable_seqscan = off;
SET max_parallel_workers_per_gather = 0;
SET jit = off;

CREATE OR REPLACE FUNCTION rbb_pctl(arr float8[], q float8) RETURNS float8
LANGUAGE sql AS $f$ SELECT percentile_cont(q) WITHIN GROUP (ORDER BY x) FROM unnest(arr) x $f$;

DO $$
DECLARE
    qv vector; t0 timestamptz; k int; c int;
    d_off float8[] := '{}'; d_auto float8[] := '{}'; d_exact float8[] := '{}';
    eff_off bigint := 0; eff_auto bigint := 0; eff_exact bigint := 0;
BEGIN
    FOR k IN 1..50 LOOP   -- warm
        PERFORM id FROM rbb_docs ORDER BY embedding <~> turbohybrid_query(
          vector_query => (SELECT embedding FROM rbb_docs WHERE id = (k * 131) % 5000 + 1)) LIMIT 10;
    END LOOP;

    FOR k IN 1..250 LOOP
        c := (k * 131) % 5000 + 1;
        qv := (SELECT embedding FROM rbb_docs WHERE id = c);

        PERFORM set_config('turbohybrid.dense_rescore_band', 'off', true);
        t0 := clock_timestamp();
        PERFORM id FROM rbb_docs ORDER BY embedding <~> turbohybrid_query(vector_query => qv) LIMIT 10;
        d_off := array_append(d_off, extract(epoch FROM clock_timestamp() - t0) * 1000);
        eff_off := eff_off + (turbohybrid_last_scan_stats() ->> 'graph_effective_rescore_band')::bigint;

        PERFORM set_config('turbohybrid.dense_rescore_band', 'auto', true);
        t0 := clock_timestamp();
        PERFORM id FROM rbb_docs ORDER BY embedding <~> turbohybrid_query(vector_query => qv) LIMIT 10;
        d_auto := array_append(d_auto, extract(epoch FROM clock_timestamp() - t0) * 1000);
        eff_auto := eff_auto + (turbohybrid_last_scan_stats() ->> 'graph_effective_rescore_band')::bigint;

        PERFORM set_config('turbohybrid.dense_rescore_band', 'exact', true);
        t0 := clock_timestamp();
        PERFORM id FROM rbb_docs ORDER BY embedding <~> turbohybrid_query(vector_query => qv) LIMIT 10;
        d_exact := array_append(d_exact, extract(epoch FROM clock_timestamp() - t0) * 1000);
        eff_exact := eff_exact + (turbohybrid_last_scan_stats() ->> 'graph_effective_rescore_band')::bigint;
    END LOOP;
    RESET turbohybrid.dense_rescore_band;

    RAISE NOTICE 'avg effective rescore band/query: off=%  auto=%  exact=%',
        eff_off / 250, eff_auto / 250, eff_exact / 250;
    RAISE NOTICE 'p50 ms: off=%  auto=%  exact=%',
        round(rbb_pctl(d_off,0.5)::numeric,4), round(rbb_pctl(d_auto,0.5)::numeric,4),
        round(rbb_pctl(d_exact,0.5)::numeric,4);
    RAISE NOTICE 'p95 ms: off=%  auto=%  exact=%',
        round(rbb_pctl(d_off,0.95)::numeric,4), round(rbb_pctl(d_auto,0.95)::numeric,4),
        round(rbb_pctl(d_exact,0.95)::numeric,4);
    RAISE NOTICE 'p50 exact-rescore overhead: exact vs off = %x',
        round((rbb_pctl(d_exact,0.5)/rbb_pctl(d_off,0.5))::numeric,2);
END $$;

DROP FUNCTION rbb_pctl(float8[], float8);
DROP TABLE rbb_docs;
RESET enable_seqscan;
RESET max_parallel_workers_per_gather;
RESET jit;
