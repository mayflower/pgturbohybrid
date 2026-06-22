-- pgturbohybrid_fuzz: malformed / hostile user input must produce clean
-- PostgreSQL ERRORs (never a crash, assertion failure, or silently-wrong
-- result), and the session must stay usable afterwards. See SECURITY_CODE_AUDIT.md.
\set VERBOSITY terse
SET client_min_messages = warning;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;
RESET client_min_messages;

-- --- opaque types reject text input cleanly ------------------------------
SELECT 'garbage'::turbohybrid_query;
SELECT 'garbage'::turbohybrid_sparse_vector;
SELECT '{1:0.5}'::turbohybrid_sparse_vector;
SELECT '[[1,2],[3,4]]'::turbohybrid_multivector;

-- --- sparse-vector constructor validation --------------------------------
SELECT turbohybrid_sparse_vector_from_arrays(ARRAY[1, 2], ARRAY[1.0]::float4[]);          -- length mismatch
SELECT turbohybrid_sparse_vector_from_arrays(ARRAY[1], ARRAY['NaN'::float4]);             -- non-finite weight
SELECT turbohybrid_sparse_vector_from_arrays(ARRAY[1], ARRAY['Infinity'::float4]);        -- non-finite weight
SELECT turbohybrid_sparse_vector_build(ARRAY[1]::int4[], ARRAY[1.0]::float4[],
                                       '{"normalize":"bogus"}'::jsonb);                    -- bad option

-- --- multivector constructor validation ----------------------------------
SELECT turbohybrid_multivector(ARRAY[]::vector[]);                                        -- empty
SELECT turbohybrid_multivector_from_float4(ARRAY[1, 2, 3]::float4[], 2);                   -- not divisible by dim
SELECT turbohybrid_multivector_from_float4(ARRAY[1, 2]::float4[], 0);                      -- dim 0
SELECT turbohybrid_multivector_from_float4(ARRAY[1, 2]::float4[], -1);                     -- negative dim
SELECT turbohybrid_multivector_from_contexts(ARRAY[1, 2, 3, 4]::float4[], 2,
                                             ARRAY[5]::int4[]);                            -- offsets must start at 0

-- --- scan-time query-vector validation -----------------------------------
CREATE TABLE fz (id int, e vector(3));
INSERT INTO fz SELECT g, ARRAY[g, g + 1, g + 2]::float4[]::vector FROM generate_series(1, 50) g;
CREATE INDEX ON fz USING turbohybrid (e vector_cosine_turbohybrid_ops);
SET enable_seqscan = off;

SELECT id FROM fz ORDER BY e <~> turbohybrid_query(vector_query => '[NaN,1,2]'::vector) LIMIT 1;       -- NaN
SELECT id FROM fz ORDER BY e <~> turbohybrid_query(vector_query => '[Infinity,1,2]'::vector) LIMIT 1;  -- Inf
SELECT id FROM fz ORDER BY e <~> turbohybrid_query(vector_query => '[1,2,3,4]'::vector) LIMIT 1;       -- wrong dim

-- --- the session is still healthy after all of the handled errors --------
SELECT count(*) AS still_usable FROM (
  SELECT id FROM fz ORDER BY e <~> turbohybrid_query(vector_query => '[1,2,3]'::vector) LIMIT 5
) q;

RESET enable_seqscan;
DROP TABLE fz;
