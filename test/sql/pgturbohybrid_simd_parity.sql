-- SIMD/scalar result-parity guardrail for native dense scans.
--
-- Asserts that a native turbohybrid dense scan returns the SAME result-ID set
-- regardless of the dense scoring kernel selected by turbohybrid.simd and
-- turbohybrid.dense_u8_batch_x4, and that the SIMD kernel stat fields are
-- present. This is a correctness guardrail, NOT a latency benchmark -- it makes
-- no timing assertions.
--
-- Architecture-neutral: result-set parity holds on any backend (x86 SIMD, ARM
-- SIMD, or the scalar fallback). The one x86-specific expectation (that an AVX2
-- host with simd=on does not fall back to the scalar kernel) is guarded behind
-- turbohybrid_simd_capabilities() so it is simply skipped on non-x86 / non-SIMD
-- builds rather than failing.
--
-- ID sets are compared sorted (order-independent): the documented behavior is
-- that SIMD vs scalar may reorder tied/near-equal candidates, but must never
-- change WHICH ids are returned.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;

SET enable_seqscan = off;
SET jit = off;

-- Deterministic, well-separated 1536-dim corpus (>=1024 dims exercises the 4-bit
-- u8 SIMD path). No random(): reproducible.
CREATE TABLE simd_parity_docs (id int PRIMARY KEY, embedding vector(1536));
INSERT INTO simd_parity_docs
SELECT g,
  (SELECT array_agg((sin((g % 32 + 1) * 0.30 + d * 0.010)
                   + cos((g % 32 + 1) * 0.17 + d * 0.007)
                   + 0.50 * sin(g * 0.05 + d * 0.013))::real ORDER BY d)
   FROM generate_series(1, 1536) d)::real[]::vector
FROM generate_series(1, 1500) g;

CREATE INDEX simd_parity_idx ON simd_parity_docs
  USING turbohybrid (embedding vector_cosine_turbohybrid_ops)
  WITH (quantization_bits = 4, exact_storage = off);
ANALYZE simd_parity_docs;

-- Sorted top-k ID set for the current turbohybrid.simd / dense_u8_batch_x4 GUCs.
CREATE FUNCTION simd_parity_ids() RETURNS bigint[]
LANGUAGE plpgsql AS $$
DECLARE qv vector; res bigint[];
BEGIN
    SELECT embedding INTO qv FROM simd_parity_docs WHERE id = 100;
    SELECT array_agg(id ORDER BY id) INTO res FROM (
        SELECT id FROM simd_parity_docs
        ORDER BY embedding <~> turbohybrid_query(vector_query => qv, dense_k => 100, final_k => 10)
        LIMIT 10) t;
    RETURN res;
END $$;

DO $$
DECLARE base bigint[]; got bigint[]; st jsonb; caps jsonb;
BEGIN
    -- baseline: SIMD on, x4 batch on
    PERFORM set_config('turbohybrid.simd', 'on', false);
    PERFORM set_config('turbohybrid.dense_u8_batch_x4', 'on', false);
    base := simd_parity_ids();

    -- (a) SIMD kernel stat fields present (presence only, no arch-specific values)
    st := turbohybrid_last_scan_stats();
    IF st->>'dense_scorer' IS NULL
       OR NOT (st ? 'graph_score_kernels')
       OR st->>'graph_scored_codes' IS NULL THEN
        RAISE EXCEPTION 'simd_parity: missing kernel stat fields (dense_scorer/graph_score_kernels/graph_scored_codes)';
    END IF;

    -- (b) capability-guarded x86 expectation: AVX2 host + simd=on must not use the
    --     scalar fallback. Skipped cleanly when SIMD is unavailable.
    caps := turbohybrid_simd_capabilities();
    IF COALESCE((caps->>'runtime_avx2')::bool, false) THEN
        IF st->>'dense_scorer' = 'scalar_lut' THEN
            RAISE EXCEPTION 'simd_parity: runtime_avx2 and simd=on but dense_scorer=scalar_lut';
        END IF;
    END IF;

    -- (c) parity: x4 batch off returns the same ID set
    PERFORM set_config('turbohybrid.dense_u8_batch_x4', 'off', false);
    got := simd_parity_ids();
    IF got IS DISTINCT FROM base THEN
        RAISE EXCEPTION 'simd_parity: dense_u8_batch_x4=off changed the result set (% vs %)', got, base;
    END IF;

    -- (d) parity: SIMD off (scalar) returns the same ID set
    PERFORM set_config('turbohybrid.simd', 'off', false);
    got := simd_parity_ids();
    IF got IS DISTINCT FROM base THEN
        RAISE EXCEPTION 'simd_parity: turbohybrid.simd=off changed the result set (% vs %)', got, base;
    END IF;

    RESET turbohybrid.simd;
    RESET turbohybrid.dense_u8_batch_x4;
END $$;

SELECT 'simd_parity_ok' AS result;

DROP FUNCTION simd_parity_ids();
DROP TABLE simd_parity_docs;
