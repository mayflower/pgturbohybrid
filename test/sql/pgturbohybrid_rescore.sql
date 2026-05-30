-- Exact-rescore policy for native graph scans.
--
-- The latency profile must not pay exact f32 rescore costs unless the
-- index/query configuration requires it:
--   * an exact-free (code-only) 4-bit native index never exact-rescores --
--     graph_effective_rescore_band stays 0 even when the band is forced to
--     'exact', because there are no stored f32 vectors to rescore;
--   * an exact_storage index honors the band (exact rescores, off does not);
--   * hybrid (vector + bm25) scans do not force exact dense rescore under the
--     latency profile (exact_rescore_for_bm25_only is off).
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;

SET turbohybrid.profile = latency;
SET enable_seqscan = off;
SET jit = off;

-- Exact-free (default) and exact_storage 4-bit native indexes over the same data.
DROP TABLE IF EXISTS rb_free;
CREATE TABLE rb_free (id int PRIMARY KEY, embedding vector(64), body_tsv tsvector);
INSERT INTO rb_free(id, embedding, body_tsv)
SELECT i,
       (SELECT array_agg(sin(i * 0.1 + g * 0.3)::real ORDER BY g) FROM generate_series(1, 64) g)::real[]::vector,
       to_tsvector('english', 'document ' || i)
FROM generate_series(1, 400) AS i;
CREATE INDEX rb_free_idx ON rb_free
    USING turbohybrid (embedding vector_cosine_turbohybrid_ops, body_tsv bm25_tsvector_turbohybrid_ops)
    WITH (quantization_bits = 4);

DROP TABLE IF EXISTS rb_exact;
CREATE TABLE rb_exact (id int PRIMARY KEY, embedding vector(64), body_tsv tsvector);
INSERT INTO rb_exact(id, embedding, body_tsv)
SELECT i,
       (SELECT array_agg(sin(i * 0.1 + g * 0.3)::real ORDER BY g) FROM generate_series(1, 64) g)::real[]::vector,
       to_tsvector('english', 'document ' || i)
FROM generate_series(1, 400) AS i;
CREATE INDEX rb_exact_idx ON rb_exact
    USING turbohybrid (embedding vector_cosine_turbohybrid_ops, body_tsv bm25_tsvector_turbohybrid_ops)
    WITH (quantization_bits = 4, exact_storage = on);

ANALYZE rb_free;
ANALYZE rb_exact;

DO $$
DECLARE
    qv vector;
    st jsonb;
BEGIN
    qv := (SELECT embedding FROM rb_free WHERE id = 42);

    -- (1) Latency profile, exact-free index, default band: no exact rescore.
    PERFORM id FROM rb_free ORDER BY embedding <~> turbohybrid_query(vector_query => qv) LIMIT 10;
    st := turbohybrid_last_scan_stats();
    IF st->>'scan_orchestration' <> 'graph_native' THEN
        RAISE EXCEPTION 'expected native scan, got %', st->>'scan_orchestration';
    END IF;
    IF (st->>'exact_free')::bool IS DISTINCT FROM true THEN
        RAISE EXCEPTION 'exact-free index not reported exact_free: %', st;
    END IF;
    IF (st->>'exact_storage')::bool IS DISTINCT FROM false THEN
        RAISE EXCEPTION 'exact-free index reported exact_storage: %', st;
    END IF;
    IF st->>'graph_rescore_band' <> 'auto' THEN
        RAISE EXCEPTION 'configured band not auto: %', st->>'graph_rescore_band';
    END IF;
    IF (st->>'graph_effective_rescore_band')::int <> 0 THEN
        RAISE EXCEPTION 'latency exact-free effective rescore band <> 0: %', st;
    END IF;
    IF (st->>'graph_rescore_count')::int <> 0 OR (st->>'graph_rescore_pages')::int <> 0 THEN
        RAISE EXCEPTION 'latency exact-free rescored (count/pages <> 0): %', st;
    END IF;
    IF (st->>'graph_rescore_band_active')::bool IS DISTINCT FROM false THEN
        RAISE EXCEPTION 'latency exact-free reports band active: %', st;
    END IF;

    -- (2) Forcing band = exact must NOT make a code-only index exact-rescore.
    SET turbohybrid.dense_rescore_band = exact;
    PERFORM id FROM rb_free ORDER BY embedding <~> turbohybrid_query(vector_query => qv) LIMIT 10;
    st := turbohybrid_last_scan_stats();
    IF (st->>'graph_rescore_band_policy') <> 'exact' THEN
        RAISE EXCEPTION 'dense_rescore_band GUC not reflected: %', st->>'graph_rescore_band_policy';
    END IF;
    IF (st->>'graph_effective_rescore_band')::int <> 0 THEN
        RAISE EXCEPTION 'exact-free index exact-rescored under band=exact (must stay 0): %', st;
    END IF;
    RESET turbohybrid.dense_rescore_band;

    -- (3) exact_storage index: band = exact rescores (quality behavior intact).
    SET turbohybrid.dense_rescore_band = exact;
    PERFORM id FROM rb_exact ORDER BY embedding <~> turbohybrid_query(vector_query => qv) LIMIT 10;
    st := turbohybrid_last_scan_stats();
    IF (st->>'exact_free')::bool IS DISTINCT FROM false THEN
        RAISE EXCEPTION 'exact_storage index reported exact_free: %', st;
    END IF;
    IF (st->>'exact_storage')::bool IS DISTINCT FROM true THEN
        RAISE EXCEPTION 'exact_storage index not reported exact_storage: %', st;
    END IF;
    IF (st->>'graph_effective_rescore_band')::int <= 0 OR (st->>'graph_rescore_count')::int <= 0 THEN
        RAISE EXCEPTION 'exact_storage band=exact did not rescore: %', st;
    END IF;

    -- (4) exact_storage index: band = off disables exact rescore.
    SET turbohybrid.dense_rescore_band = off;
    PERFORM id FROM rb_exact ORDER BY embedding <~> turbohybrid_query(vector_query => qv) LIMIT 10;
    st := turbohybrid_last_scan_stats();
    IF (st->>'graph_effective_rescore_band')::int <> 0 THEN
        RAISE EXCEPTION 'exact_storage band=off still rescored: %', st;
    END IF;
    RESET turbohybrid.dense_rescore_band;

    -- (5) Hybrid (vector + bm25) scan on the exact-free index under the latency
    -- profile must not force exact dense rescore.
    PERFORM id FROM rb_free
      ORDER BY embedding <~> turbohybrid_query(vector_query => qv,
                                               text_query => to_tsquery('english', 'document'))
      LIMIT 10;
    st := turbohybrid_last_scan_stats();
    IF (st->>'exact_rescore_for_bm25_only')::bool IS DISTINCT FROM false THEN
        RAISE EXCEPTION 'latency profile enabled exact_rescore_for_bm25_only: %', st;
    END IF;
    IF (st->>'exact_free')::bool IS DISTINCT FROM true OR (st->>'graph_effective_rescore_band')::int <> 0 THEN
        RAISE EXCEPTION 'latency hybrid exact-free scan exact-rescored: %', st;
    END IF;
END;
$$;

SELECT 'rescore band ok' AS result;

DROP INDEX rb_free_idx;
DROP INDEX rb_exact_idx;
DROP TABLE rb_free;
DROP TABLE rb_exact;
RESET turbohybrid.profile;
RESET enable_seqscan;
RESET jit;
