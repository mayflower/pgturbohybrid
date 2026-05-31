-- Parity of the integer query-split dense scorers (AVX2 / AVX-VNNI / AVX-512
-- VNNI / NEON) against each other and against the scalar/LUT reference path.
--
-- The query-split kernels all compute the same integer dot product, so the
-- distinct kernels that actually run must return identical top-K orderings.
-- The scalar/LUT path is a different approximation (it does not quantize the
-- query) and steers the greedy graph walk slightly differently, so instead of
-- requiring scalar and query split to match each other we require both to
-- recover the known nearest neighbours of a structured corpus.
--
-- All comparisons run inside function bodies that raise on mismatch, so the
-- regression output stays stable.  The test adapts to the platform: if no
-- query-split SIMD kernel is available (e.g. SIMD disabled) every config falls
-- back to scalar and the parity holds trivially.
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;

SET enable_seqscan = off;
SET jit = off;
SELECT setseed(0.42);

-- 1536-dim corpus (>= 1024 so query split is eligible) with an unambiguous
-- nearest-neighbour structure that survives 4-bit/2-bit quantization: docs
-- 1..20 each bump a *distinct* dimension of the query q by a graded magnitude
-- (so they have distinct codes and distinct distances and are clearly the 20
-- nearest), while docs 21..500 are random and far.  turbohybrid indexes need a
-- vector column plus a tsvector column even for dense-only scans; the dense
-- query below passes no text_query.
DROP TABLE IF EXISTS qs_basis;
CREATE TABLE qs_basis (q real[]);
INSERT INTO qs_basis SELECT array_agg(random() * 2 - 1) FROM generate_series(1, 1536);

DROP TABLE IF EXISTS qs_docs;
CREATE TABLE qs_docs (id int PRIMARY KEY, embedding vector(1536), body_tsv tsvector);
INSERT INTO qs_docs(id, embedding, body_tsv)
SELECT i,
       (SELECT array_agg(CASE WHEN ord = 1 + (i * 70) % 1536
                              THEN b.q[ord] + (5.0 + i * 0.5)
                              ELSE b.q[ord] END ORDER BY ord)
        FROM generate_series(1, 1536) AS ord)::real[]::vector,
       to_tsvector('english', 'document number ' || i)
FROM qs_basis b, generate_series(1, 20) AS i;
INSERT INTO qs_docs(id, embedding, body_tsv)
SELECT i,
       (SELECT array_agg(random() * 2 - 1)::real[]::vector FROM generate_series(1, 1536)),
       to_tsvector('english', 'document number ' || i)
FROM generate_series(21, 500) AS i;

DROP TABLE IF EXISTS qs_query;
CREATE TABLE qs_query (embedding vector(1536));
INSERT INTO qs_query SELECT (q)::vector FROM qs_basis;

DROP TABLE IF EXISTS qs_results;
CREATE TABLE qs_results (label text, bits int, rank int, id int);
DROP TABLE IF EXISTS qs_kernel;
CREATE TABLE qs_kernel (label text, bits int, kernel text);

-- Run one dense top-20 query under the current GUCs (near-exhaustive dense_k so
-- graph recall is not the variable under test), recording the ordered ids and
-- the dense scorer that actually ran (read right after the scan).
CREATE OR REPLACE FUNCTION qs_collect(p_label text, p_bits int) RETURNS void
LANGUAGE plpgsql AS $$
DECLARE
    qv vector(1536);
BEGIN
    SELECT embedding INTO qv FROM qs_query;
    INSERT INTO qs_results(label, bits, rank, id)
    SELECT p_label, p_bits, row_number() OVER (), t.id
    FROM (
        SELECT id
        FROM qs_docs
        ORDER BY embedding <~> turbohybrid_query(vector_query => qv, dense_k => 600, final_k => 20)
        LIMIT 20
    ) t;
    INSERT INTO qs_kernel(label, bits, kernel)
    VALUES (p_label, p_bits, turbohybrid_last_scan_stats() ->> 'dense_scoring_kernel');
END;
$$;

CREATE OR REPLACE FUNCTION qs_run_all(p_bits int) RETURNS void
LANGUAGE plpgsql AS $$
BEGIN
    SET turbohybrid.simd = off;                       -- scalar / LUT reference
    PERFORM qs_collect('scalar', p_bits);

    SET turbohybrid.simd = on;                        -- best available query split
    SET turbohybrid.dense_graph_avx512vnni = on;
    SET turbohybrid.dense_graph_avxvnni = on;
    PERFORM qs_collect('auto', p_bits);

    SET turbohybrid.dense_graph_avx512vnni = off;     -- drop AVX-512 VNNI
    SET turbohybrid.dense_graph_avxvnni = on;
    PERFORM qs_collect('no512', p_bits);

    SET turbohybrid.dense_graph_avx512vnni = off;     -- force AVX2 query split
    SET turbohybrid.dense_graph_avxvnni = off;
    PERFORM qs_collect('avx2only', p_bits);

    RESET turbohybrid.simd;
    RESET turbohybrid.dense_graph_avx512vnni;
    RESET turbohybrid.dense_graph_avxvnni;
END;
$$;

-- Recall of a config's top-20 against the known nearest neighbours (ids 1..20).
CREATE OR REPLACE FUNCTION qs_recall(p_label text, p_bits int) RETURNS int
LANGUAGE sql AS $$
    SELECT count(*)::int
    FROM (SELECT id FROM qs_results WHERE label = p_label AND bits = p_bits
          INTERSECT SELECT generate_series(1, 20)) x;
$$;

CREATE OR REPLACE FUNCTION qs_check(p_bits int) RETURNS void
LANGUAGE plpgsql AS $$
DECLARE
    base_kernel text;
    auto_kernel text;
    rec record;
    ref_ids int[];
    cmp_ids int[];
BEGIN
    SELECT kernel INTO base_kernel FROM qs_kernel WHERE label = 'scalar' AND bits = p_bits;
    SELECT kernel INTO auto_kernel FROM qs_kernel WHERE label = 'auto' AND bits = p_bits;

    -- turbohybrid.simd = off must always select the scalar/LUT path.
    IF base_kernel <> 'scalar' THEN
        RAISE EXCEPTION 'bits %: simd=off used kernel %, expected scalar', p_bits, base_kernel;
    END IF;

    -- Both the scalar reference and the default scan must recover the known
    -- nearest neighbours (shared ground-truth parity).
    IF qs_recall('scalar', p_bits) < 18 THEN
        RAISE EXCEPTION 'bits %: scalar recall %/20 (< 18)', p_bits, qs_recall('scalar', p_bits);
    END IF;
    IF qs_recall('auto', p_bits) < 18 THEN
        RAISE EXCEPTION 'bits %: query-split recall %/20 (< 18)', p_bits, qs_recall('auto', p_bits);
    END IF;

    IF auto_kernel = 'scalar' THEN
        RETURN;                 -- no query-split SIMD on this platform
    END IF;

    -- Default scan must use a query-split kernel, never the scalar/LUT gather.
    IF auto_kernel NOT IN ('query_split_avx2', 'avxvnni', 'avx512vnni', 'neon', 'arm_i8mm') THEN
        RAISE EXCEPTION 'bits %: auto used unexpected kernel %', p_bits, auto_kernel;
    END IF;

    -- Every query-split kernel that actually ran must return the exact same
    -- top-20 ordering as auto (identical integer dot product), and none may
    -- silently fall back to the scalar/LUT gather while query split is on.
    SELECT array_agg(id ORDER BY rank) INTO ref_ids
    FROM qs_results WHERE label = 'auto' AND bits = p_bits;

    FOR rec IN
        SELECT label, kernel FROM qs_kernel WHERE bits = p_bits AND label <> 'scalar'
    LOOP
        IF rec.kernel = 'scalar' THEN
            RAISE EXCEPTION 'bits %: config % fell back to scalar/LUT despite query split', p_bits, rec.label;
        END IF;
        SELECT array_agg(id ORDER BY rank) INTO cmp_ids
        FROM qs_results WHERE label = rec.label AND bits = p_bits;
        IF cmp_ids <> ref_ids THEN
            RAISE EXCEPTION 'bits %: query-split kernel % (config %) differs from auto',
                p_bits, rec.kernel, rec.label;
        END IF;
    END LOOP;
END;
$$;

-- 4-bit index parity.
DROP INDEX IF EXISTS qs_idx;
CREATE INDEX qs_idx ON qs_docs USING turbohybrid (embedding vector_cosine_turbohybrid_ops, body_tsv bm25_tsvector_turbohybrid_ops)
    WITH (quantization_bits = 4);
ANALYZE qs_docs;
SELECT qs_run_all(4);
SELECT qs_check(4);

-- 2-bit index parity.
DROP INDEX qs_idx;
CREATE INDEX qs_idx ON qs_docs USING turbohybrid (embedding vector_cosine_turbohybrid_ops, body_tsv bm25_tsvector_turbohybrid_ops)
    WITH (quantization_bits = 2);
ANALYZE qs_docs;
SELECT qs_run_all(2);
SELECT qs_check(2);

SELECT 'querysplit parity ok' AS result;

-- Targeted boundary check for the shared low/high split math
-- (TqQuerySplitValue, exposed via turbohybrid_query_split_probe).  Pins the
-- low/high values around negatives, positives, the +/- HIGH_COEF/2 boundary,
-- and the saturation limits, for both the signed split (K=256, absMax=32639)
-- and the x86 u8 split (K=128, absMax=8127).  These are the canonical values of
-- the unchanged algorithm; the refactor must keep them bit-identical.
CREATE OR REPLACE FUNCTION qsp_check(p_val float8, p_absmax float8, p_hc int,
									 p_qs int, p_low int, p_high int) RETURNS void
LANGUAGE plpgsql AS $$
DECLARE
	r jsonb;
BEGIN
	r := turbohybrid_query_split_probe(p_val, p_absmax, p_hc);
	IF (r ->> 'qsigned')::int <> p_qs OR (r ->> 'low')::int <> p_low
		OR (r ->> 'high')::int <> p_high THEN
		RAISE EXCEPTION 'split(%, absmax=%, K=%) = {qsigned=%, low=%, high=%}, expected {%, %, %}',
			p_val, p_absmax, p_hc, r ->> 'qsigned', r ->> 'low', r ->> 'high', p_qs, p_low, p_high;
	END IF;
	-- Reconstruction invariant: qsigned = K*high + low.
	IF (r ->> 'qsigned')::int <> p_hc * (r ->> 'high')::int + (r ->> 'low')::int THEN
		RAISE EXCEPTION 'split invariant broken at %: qsigned=% != %*high(%)+low(%)',
			p_val, r ->> 'qsigned', p_hc, r ->> 'high', r ->> 'low';
	END IF;
END;
$$;

DO $$
BEGIN
	-- Signed split: HIGH_COEF = 256, absMax = 32639.
	PERFORM qsp_check(300, 32639, 256, 300, 44, 1);        -- positive
	PERFORM qsp_check(130, 32639, 256, 130, -126, 1);      -- positive, low wraps past +K/2
	PERFORM qsp_check(128, 32639, 256, 128, -128, 1);      -- exactly +K/2 -> low = -K/2
	PERFORM qsp_check(127, 32639, 256, 127, 127, 0);       -- just below +K/2
	PERFORM qsp_check(1, 32639, 256, 1, 1, 0);
	PERFORM qsp_check(0, 32639, 256, 0, 0, 0);
	PERFORM qsp_check(-1, 32639, 256, -1, -1, 0);
	PERFORM qsp_check(-127, 32639, 256, -127, -127, 0);
	PERFORM qsp_check(-128, 32639, 256, -128, -128, 0);    -- -K/2
	PERFORM qsp_check(-129, 32639, 256, -129, 127, -1);    -- just past -K/2
	PERFORM qsp_check(-130, 32639, 256, -130, 126, -1);    -- negative
	PERFORM qsp_check(-300, 32639, 256, -300, -44, -1);    -- negative
	PERFORM qsp_check(40000, 32639, 256, 32639, 127, 127); -- +saturation
	PERFORM qsp_check(-40000, 32639, 256, -32639, -127, -127); -- -saturation

	-- x86 u8 split: HIGH_COEF = 128, absMax = 8127.
	PERFORM qsp_check(200, 8127, 128, 200, -56, 2);        -- positive
	PERFORM qsp_check(70, 8127, 128, 70, -58, 1);          -- positive, low wraps past +K/2
	PERFORM qsp_check(64, 8127, 128, 64, -64, 1);          -- +K/2
	PERFORM qsp_check(63, 8127, 128, 63, 63, 0);           -- just below +K/2
	PERFORM qsp_check(-64, 8127, 128, -64, -64, 0);        -- -K/2
	PERFORM qsp_check(-65, 8127, 128, -65, 63, -1);        -- just past -K/2
	PERFORM qsp_check(-200, 8127, 128, -200, 56, -2);      -- negative
	PERFORM qsp_check(9000, 8127, 128, 8127, 63, 63);      -- +saturation
	PERFORM qsp_check(-9000, 8127, 128, -8127, -63, -63);  -- -saturation
END $$;

SELECT 'query split boundary values ok' AS result;

DROP FUNCTION qsp_check(float8, float8, int, int, int, int);

DROP FUNCTION qs_collect(text, int);
DROP FUNCTION qs_run_all(int);
DROP FUNCTION qs_recall(text, int);
DROP FUNCTION qs_check(int);
DROP TABLE qs_results;
DROP TABLE qs_kernel;
DROP TABLE qs_query;
DROP TABLE qs_docs;
DROP TABLE qs_basis;
RESET enable_seqscan;
RESET jit;
