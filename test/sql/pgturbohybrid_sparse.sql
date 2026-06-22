-- Canonical sparse-vector builder + inspection helpers.
SET client_min_messages = warning;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;
RESET client_min_messages;

-- Build with default options: dedup=sum, sort=true, drop_non_positive=true.
-- term 5 appears twice (1.0 + 3.0 = 4.0); output sorted by term_id.
SELECT turbohybrid_sparse_vector_count(
  turbohybrid_sparse_vector_build(ARRAY[5,1,5,3]::int4[], ARRAY[1.0,2.0,3.0,4.0]::float4[])) AS cnt;
SELECT turbohybrid_sparse_vector_term_ids(
  turbohybrid_sparse_vector_build(ARRAY[5,1,5,3]::int4[], ARRAY[1.0,2.0,3.0,4.0]::float4[])) AS term_ids;
SELECT turbohybrid_sparse_vector_weights(
  turbohybrid_sparse_vector_build(ARRAY[5,1,5,3]::int4[], ARRAY[1.0,2.0,3.0,4.0]::float4[])) AS weights;

-- dedup = max
SELECT turbohybrid_sparse_vector_weights(
  turbohybrid_sparse_vector_build(ARRAY[5,5]::int4[], ARRAY[1.0,3.0]::float4[],
                                  '{"deduplicate":"max"}')) AS dedup_max;

-- dedup = error (duplicate term -> error)
SELECT turbohybrid_sparse_vector_build(ARRAY[5,5]::int4[], ARRAY[1.0,3.0]::float4[],
                                       '{"deduplicate":"error"}');

-- top_k keeps the highest-weight terms, output re-sorted by term_id
SELECT turbohybrid_sparse_vector_term_ids(
  turbohybrid_sparse_vector_build(ARRAY[1,2,3]::int4[], ARRAY[0.5,2.0,1.0]::float4[],
                                  '{"top_k":2}')) AS top_k_ids;

-- drop_non_positive (default): weight <= 0 dropped
SELECT turbohybrid_sparse_vector_term_ids(
  turbohybrid_sparse_vector_build(ARRAY[1,2,3]::int4[], ARRAY[-1.0,0.0,2.0]::float4[])) AS dropped_ids;

-- min_weight raises the drop threshold
SELECT turbohybrid_sparse_vector_term_ids(
  turbohybrid_sparse_vector_build(ARRAY[1,2,3]::int4[], ARRAY[0.5,1.5,2.0]::float4[],
                                  '{"min_weight":1.0}')) AS min_weight_ids;

-- drop_non_positive=false keeps negative/zero weights
SELECT turbohybrid_sparse_vector_weights(
  turbohybrid_sparse_vector_build(ARRAY[1,2]::int4[], ARRAY[-1.0,2.0]::float4[],
                                  '{"drop_non_positive":false}')) AS keep_negative;

-- sort=false preserves first-seen order
SELECT turbohybrid_sparse_vector_term_ids(
  turbohybrid_sparse_vector_build(ARRAY[5,1,3]::int4[], ARRAY[1.0,1.0,1.0]::float4[],
                                  '{"sort":false}')) AS unsorted_ids;

-- empty inputs / all-dropped -> count 0
SELECT turbohybrid_sparse_vector_count(
  turbohybrid_sparse_vector_build(ARRAY[]::int4[], ARRAY[]::float4[])) AS empty_cnt;
SELECT turbohybrid_sparse_vector_count(
  turbohybrid_sparse_vector_build(ARRAY[1]::int4[], ARRAY[-1.0]::float4[])) AS all_dropped_cnt;

-- invalid inputs
SELECT turbohybrid_sparse_vector_build(ARRAY[1,2]::int4[], ARRAY[1.0]::float4[]);          -- length mismatch
SELECT turbohybrid_sparse_vector_build(ARRAY[-1]::int4[], ARRAY[1.0]::float4[]);           -- negative term id
SELECT turbohybrid_sparse_vector_build(ARRAY[1]::int4[], ARRAY['NaN'::float4]);            -- non-finite weight
SELECT turbohybrid_sparse_vector_build(ARRAY[1]::int4[], ARRAY[1.0]::float4[],
                                       '{"normalize":"l2"}');                              -- unsupported normalize
SELECT turbohybrid_sparse_vector_build(ARRAY[1]::int4[], ARRAY[1.0]::float4[],
                                       '{"deduplicate":"avg"}');                           -- unsupported dedup
SELECT turbohybrid_sparse_vector_build(ARRAY[1]::int4[], ARRAY[1.0]::float4[],
                                       '{"top_k":-1}');                                    -- negative top_k

-- existing constructor is unchanged
SELECT turbohybrid_sparse_vector_count(
  turbohybrid_sparse_vector_from_arrays(ARRAY[1,2]::int4[], ARRAY[1.0,2.0]::float4[])) AS from_arrays_cnt;

-- inspectors round-trip a built vector
WITH v AS (
  SELECT turbohybrid_sparse_vector_build(ARRAY[9,2,9]::int4[], ARRAY[1.0,2.0,2.0]::float4[]) AS s
)
SELECT turbohybrid_sparse_vector_term_ids(s) AS ids,
       turbohybrid_sparse_vector_weights(s) AS w,
       turbohybrid_sparse_vector_count(s)   AS c
FROM v;
