-- Guard against regressions where packed 4-bit (or 2-bit) codes are scored as
-- linear nibbles 0..15 instead of through the non-uniform codebook.
--
-- For a fixed, deterministic (query, doc) pair that quantizes across the whole
-- codebook (low, centre, and high bins), turbohybrid_scorer_distances() returns
-- three scores at a chosen bit width:
--   A  scalar_lut        -- reference via TqCodeCenters / the per-dim LUT
--   B  signed_split      -- codebook-shuffle SIMD (and unsigned_split_simd at 4-bit)
--   C  linear_reference  -- intentional raw 0..15 nibble interpretation
-- A correct codebook-shuffle SIMD scorer matches A; a raw-nibble scorer would
-- match C.  We assert A~=B (and A~=u8 SIMD at 4-bit) and that C is far from A
-- (so the check is meaningful) across 4-bit/2-bit and tail (non-multiple-of-16)
-- and exact (1536) dimensions.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;

SET jit = off;

-- Deterministic query (mixed signs + magnitudes) and doc (wide range so the
-- code uses many low/centre/high codebook bins) at an arbitrary dimension.
CREATE OR REPLACE FUNCTION ng_query(p_dim int) RETURNS vector
LANGUAGE sql AS $$
    SELECT array_agg((sin(g * 0.013) + 0.3 * cos(g * 0.05))::double precision)::real[]::vector
    FROM generate_series(1, p_dim) g;
$$;
CREATE OR REPLACE FUNCTION ng_doc(p_dim int) RETURNS vector
LANGUAGE sql AS $$
    SELECT array_agg((cos(g * 0.017) * sin(g * 0.003) + 0.5 * sin(g * 0.001))::double precision)::real[]::vector
    FROM generate_series(1, p_dim) g;
$$;

CREATE OR REPLACE FUNCTION ng_check(p_label text, p_bits int, p_dim int) RETURNS void
LANGUAGE plpgsql AS $$
DECLARE
    r jsonb;
    a float8;          -- scalar/LUT reference
    b float8;          -- codebook-shuffle SIMD (signed split)
    u8 float8;         -- codebook-shuffle SIMD (unsigned split, 4-bit)
    c float8;          -- raw-nibble linear reference
    distinct_n int;
    min_bins int;
BEGIN
    r := turbohybrid_scorer_distances(ng_query(p_dim), ng_doc(p_dim), p_bits);
    a := (r ->> 'scalar_lut')::float8;
    b := (r ->> 'signed_split')::float8;
    u8 := (r ->> 'unsigned_split_simd')::float8;
    c := (r ->> 'linear_reference')::float8;
    distinct_n := (r ->> 'distinct_nibbles')::int;

    -- The code must exercise many codebook bins, else raw vs codebook would not
    -- diverge and the guard would be vacuous.
    min_bins := CASE WHEN p_bits = 4 THEN 12 ELSE 4 END;
    IF distinct_n < min_bins THEN
        RAISE EXCEPTION '%: bits=% dim=% only % distinct code bins (< %)',
            p_label, p_bits, p_dim, distinct_n, min_bins;
    END IF;

    -- A ~= B: the codebook-shuffle SIMD scorer must match the scalar/LUT
    -- reference within query-quantization tolerance.  A raw-nibble SIMD scorer
    -- would instead land near C and fail here.
    IF abs(b - a) / (abs(a) + 1.0) > 0.02 THEN
        RAISE EXCEPTION '%: bits=% dim=% signed_split=% diverges from scalar_lut=% (kernel %)',
            p_label, p_bits, p_dim, b, a, r ->> 'split_kernel';
    END IF;
    IF p_bits = 4 AND abs(u8 - a) / (abs(a) + 1.0) > 0.02 THEN
        RAISE EXCEPTION '%: bits=% dim=% unsigned_split_simd=% diverges from scalar_lut=% (kernel %)',
            p_label, p_bits, p_dim, u8, a, r ->> 'unsigned_split_kernel';
    END IF;
    -- The x4 batch kernel must reproduce the single-node SIMD result bit-for-bit.
    -- Both are formatted as %.9g of the underlying double, so identical doubles
    -- yield identical text; any divergence means the 4-candidate decode/accumulate
    -- drifted from the proven single-node kernel.  Checked for both the AVX2 and
    -- AVX-512 VNNI x4 kernels via the kernel-forcing GUCs in ng_check_all().
    IF p_bits = 4 AND (r ->> 'unsigned_split_x4') IS DISTINCT FROM (r ->> 'unsigned_split_simd') THEN
        RAISE EXCEPTION '%: bits=% dim=% unsigned_split_x4=% != unsigned_split_simd=% (x4 batch parity)',
            p_label, p_bits, p_dim, r ->> 'unsigned_split_x4', r ->> 'unsigned_split_simd';
    END IF;

    -- C far from A: a linear raw-nibble score is grossly wrong, proving the
    -- guard is meaningful, and it must diverge far more than the SIMD scorer.
    IF abs(c - a) / (abs(a) + 1.0) < 0.05 THEN
        RAISE EXCEPTION '%: bits=% dim=% raw-nibble=% too close to scalar_lut=% (guard not meaningful)',
            p_label, p_bits, p_dim, c, a;
    END IF;
    IF abs(c - a) < 10.0 * abs(b - a) THEN
        RAISE EXCEPTION '%: bits=% dim=% raw-nibble gap (%) not >> codebook gap (%)',
            p_label, p_bits, p_dim, abs(c - a), abs(b - a);
    END IF;
END;
$$;

CREATE OR REPLACE FUNCTION ng_check_all(p_label text) RETURNS void
LANGUAGE plpgsql AS $$
BEGIN
    PERFORM ng_check(p_label, 4, 1536);   -- 4-bit, exact multiple of 16
    PERFORM ng_check(p_label, 4, 1540);   -- 4-bit, tail of 4 dims
    PERFORM ng_check(p_label, 2, 1536);   -- 2-bit, exact multiple of 16
    PERFORM ng_check(p_label, 2, 1540);   -- 2-bit, tail of 4 dims
END;
$$;

-- Best available SIMD (AVX-512 VNNI on capable amd64).
SET turbohybrid.dense_graph_avx512vnni = on;
SET turbohybrid.dense_graph_avxvnni = on;
SELECT ng_check_all('best');

-- Force the AVX2 kernels (signed cvtepi8/madd and unsigned maddubs).
SET turbohybrid.dense_graph_avx512vnni = off;
SET turbohybrid.dense_graph_avxvnni = off;
SELECT ng_check_all('avx2');

RESET turbohybrid.dense_graph_avx512vnni;
RESET turbohybrid.dense_graph_avxvnni;

SELECT 'nibble guard ok' AS result;

DROP FUNCTION ng_check_all(text);
DROP FUNCTION ng_check(text, int, int);
DROP FUNCTION ng_query(int);
DROP FUNCTION ng_doc(int);
RESET jit;
