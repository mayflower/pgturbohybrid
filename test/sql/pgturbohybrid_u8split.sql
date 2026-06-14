SET client_min_messages = warning;
-- Unsigned-codebook (u8) 4-bit query-split scorer: parity, codebook
-- correctness, and default selection.
--
-- The u8 scorer feeds _mm256_maddubs_epi16 / _mm512_dpbusd_epi32 an unsigned
-- codebook (the signed codebook shifted by +128, so the un-shifted values equal
-- the centres the codes were encoded with) and signed 7-bit query halves, then
-- removes the +128 bias.  It must:
--   * agree with its own scalar reference bit-for-bit (integer-derived);
--   * stay very close to the scalar/LUT reference (codebook-correct);
--   * be far from a raw 0..15 nibble interpretation (adversarial control);
--   * be selected by default on amd64 when the SIMD is available.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;

SET enable_seqscan = off;
SET jit = off;

-- (1) Per-kernel parity on deliberately non-linear 1536-dim pairs.
CREATE OR REPLACE FUNCTION u8_check(p_label text) RETURNS void
LANGUAGE plpgsql AS $$
DECLARE
    q vector;
    d vector;
    r jsonb;
    lut float8;
    sgn float8;
    u8s float8;
    u8v float8;
    lin float8;
    i int;
BEGIN
    FOR i IN 1..6 LOOP
        q := (SELECT array_agg(sin(i * 0.7 + g * 0.013))::real[]::vector
              FROM generate_series(1, 1536) g);
        d := (SELECT array_agg(cos(i * 1.1 + g * 0.017))::real[]::vector
              FROM generate_series(1, 1536) g);
        r := turbohybrid_scorer_distances(q, d);
        lut := (r ->> 'scalar_lut')::float8;
        sgn := (r ->> 'signed_split')::float8;
        u8s := (r ->> 'unsigned_split_scalar')::float8;
        u8v := (r ->> 'unsigned_split_simd')::float8;
        lin := (r ->> 'linear_reference')::float8;

        -- u8 SIMD must equal its scalar reference exactly (integer derived).
        IF (r ->> 'unsigned_split_simd') <> (r ->> 'unsigned_split_scalar') THEN
            RAISE EXCEPTION '%: pair % u8 simd=% != u8 scalar=% (kernel %)',
                p_label, i, u8v, u8s, r ->> 'unsigned_split_kernel';
        END IF;

        -- u8 must reproduce the scalar/LUT codebook distance within the
        -- query-quantization tolerance (7-bit halves -> ~few %).
        IF abs(u8v - lut) / (abs(lut) + abs(u8v) + 0.01) > 0.05 THEN
            RAISE EXCEPTION '%: pair % unsigned_split=% diverges from scalar_lut=%',
                p_label, i, u8v, lut;
        END IF;
        -- signed split likewise (sanity that both representations track LUT).
        IF abs(sgn - lut) / (abs(sgn) + abs(lut) + 0.01) > 0.05 THEN
            RAISE EXCEPTION '%: pair % signed_split=% diverges from scalar_lut=%',
                p_label, i, sgn, lut;
        END IF;

        -- Adversarial: a raw 0..15 nibble interpretation must be grossly wrong,
        -- and u8 must be much closer to the LUT reference than raw nibbles are.
        IF abs(lin - lut) < 0.5 * (abs(lut) + 1.0) THEN
            RAISE EXCEPTION '%: pair % linear_reference=% too close to scalar_lut=% (no adversarial signal)',
                p_label, i, lin, lut;
        END IF;
        IF abs(u8v - lut) >= abs(lin - lut) THEN
            RAISE EXCEPTION '%: pair % u8 (%) not closer to scalar_lut (%) than raw nibble (%)',
                p_label, i, u8v, lut, lin;
        END IF;
    END LOOP;
END;
$$;

-- Best available u8 kernel (AVX-512 VNNI on capable amd64).
SET turbohybrid.dense_graph_avx512vnni = on;
SET turbohybrid.dense_graph_avxvnni = on;
SELECT u8_check('u8_auto');

-- Force the AVX2 maddubs u8 kernel (disable AVX-512 VNNI).
SET turbohybrid.dense_graph_avx512vnni = off;
SELECT u8_check('u8_avx2');

RESET turbohybrid.dense_graph_avx512vnni;
RESET turbohybrid.dense_graph_avxvnni;

-- (2) Selection: a real 1536-dim 4-bit dense scan must use the u8 scorer by
-- default on amd64 (never the LUT gather), and honor the impl GUC.
DROP TABLE IF EXISTS u8_docs;
CREATE TABLE u8_docs (id int PRIMARY KEY, embedding vector(1536));
INSERT INTO u8_docs(id, embedding)
SELECT i,
       (SELECT array_agg(sin(i * 0.29 + g * 0.011))::real[]::vector FROM generate_series(1, 1536) g)
FROM generate_series(1, 1500) AS i;  -- enough nodes that batch-of-4 scoring fires
CREATE INDEX u8_idx ON u8_docs
    USING turbohybrid (embedding vector_cosine_turbohybrid_ops)
    WITH (quantization_bits = 4);
ANALYZE u8_docs;

DO $$
DECLARE
    qv vector;
    scorer text;
    scalar_codes int;
    simd_avail bool;
    ks jsonb;
    u8_batch bigint;
    signed_batch bigint;
    scalar_batch bigint;
BEGIN
    qv := (SELECT array_agg(cos(g * 0.017))::real[]::vector FROM generate_series(1, 1536) g);
    -- Does an unsigned-codebook SIMD kernel actually run on this build/CPU?
    simd_avail := (turbohybrid_scorer_distances(qv, qv) ->> 'unsigned_split_used')::bool;

    -- Default (auto): u8 selected when available; never the scalar/LUT gather.
    SET turbohybrid.dense_query_split_impl = auto;
    SET turbohybrid.dense_u8_split = auto;
    PERFORM id FROM u8_docs ORDER BY embedding <~> turbohybrid_query(vector_query => qv) LIMIT 10;
    scorer := turbohybrid_last_scan_stats() ->> 'dense_scorer';
    scalar_codes := (turbohybrid_last_scan_stats() ->> 'graph_scalar_scored_codes')::int;
    ks := turbohybrid_last_scan_stats() -> 'graph_score_kernels';
    IF simd_avail THEN
        IF scorer NOT LIKE 'unsigned_split_%' THEN
            RAISE EXCEPTION 'auto: 1536-dim 4-bit scan used % (expected unsigned_split_*)', scorer;
        END IF;
        IF scalar_codes <> 0 THEN
            RAISE EXCEPTION 'auto: % codes used the scalar/LUT gather', scalar_codes;
        END IF;

        -- Acceptance: the hot native batch scorer reports a u8-split kernel
        -- bucket (batch_u8_split_avx2 or _avx512vnni), and no batch nodes go
        -- through the signed split or scalar/LUT.
        u8_batch := (ks->'batch_u8_split_avx2'->>'nodes')::bigint
                  + (ks->'batch_u8_split_avx512vnni'->>'nodes')::bigint;
        signed_batch := (ks->'batch_signed_split_avx2'->>'nodes')::bigint
                      + (ks->'batch_signed_split_avxvnni'->>'nodes')::bigint
                      + (ks->'batch_signed_split_avx512vnni'->>'nodes')::bigint;
        scalar_batch := (ks->'batch_scalar_or_lut'->>'nodes')::bigint;
        IF u8_batch <= 0 THEN
            RAISE EXCEPTION 'auto: native batch scorer reported no u8-split nodes: %', ks;
        END IF;
        IF signed_batch <> 0 OR scalar_batch <> 0 THEN
            RAISE EXCEPTION 'auto: native batch used signed/scalar nodes (signed=%, scalar=%): %',
                signed_batch, scalar_batch, ks;
        END IF;
    END IF;

    -- Forced signed fallback.
    SET turbohybrid.dense_query_split_impl = signed;
    PERFORM id FROM u8_docs ORDER BY embedding <~> turbohybrid_query(vector_query => qv) LIMIT 10;
    scorer := turbohybrid_last_scan_stats() ->> 'dense_scorer';
    IF simd_avail AND scorer NOT LIKE 'signed_split_%' THEN
        RAISE EXCEPTION 'signed: scan used % (expected signed_split_*)', scorer;
    END IF;

    -- Forced unsigned.
    SET turbohybrid.dense_query_split_impl = unsigned;
    PERFORM id FROM u8_docs ORDER BY embedding <~> turbohybrid_query(vector_query => qv) LIMIT 10;
    scorer := turbohybrid_last_scan_stats() ->> 'dense_scorer';
    IF simd_avail AND scorer NOT LIKE 'unsigned_split_%' THEN
        RAISE EXCEPTION 'unsigned: scan used % (expected unsigned_split_*)', scorer;
    END IF;
    RESET turbohybrid.dense_query_split_impl;

    -- (3) Dedicated dense_u8_split control (for controlled benchmarking).
    -- off: disable the u8 split even under impl=auto -> signed split runs.
    SET turbohybrid.dense_u8_split = off;
    PERFORM id FROM u8_docs ORDER BY embedding <~> turbohybrid_query(vector_query => qv) LIMIT 10;
    scorer := turbohybrid_last_scan_stats() ->> 'dense_scorer';
    IF simd_avail AND scorer LIKE 'unsigned_split_%' THEN
        RAISE EXCEPTION 'dense_u8_split=off: scan still used u8 scorer %', scorer;
    END IF;

    -- on: force the u8 split even when impl asks for the signed split.
    SET turbohybrid.dense_query_split_impl = signed;
    SET turbohybrid.dense_u8_split = on;
    PERFORM id FROM u8_docs ORDER BY embedding <~> turbohybrid_query(vector_query => qv) LIMIT 10;
    scorer := turbohybrid_last_scan_stats() ->> 'dense_scorer';
    IF simd_avail AND scorer NOT LIKE 'unsigned_split_%' THEN
        RAISE EXCEPTION 'dense_u8_split=on: scan used % (expected unsigned_split_*)', scorer;
    END IF;

    RESET turbohybrid.dense_query_split_impl;
    RESET turbohybrid.dense_u8_split;
END;
$$;

SELECT 'u8 split parity ok' AS result;

DROP FUNCTION u8_check(text);
DROP INDEX u8_idx;
DROP TABLE u8_docs;
RESET enable_seqscan;
RESET jit;
