-- Deterministic dense/hybrid retrieval-quality grid.
--
-- Run:
--   psql -d <db> -f benchmarks/dev/retrieval_quality_grid.sql
--
-- Optional psql variables:
--   NROWS             synthetic corpus rows              (default 10000)
--   DIMS              embedding dimensions               (default 64)
--   FINAL_K           result/quality cutoff               (default 10)
--   DENSE_K           turbohybrid dense candidate budget  (default 200)
--   BM25_K            turbohybrid BM25 candidate budget   (default 200)
--   EXACT_BASELINE_K  exact dense baseline window         (default 100)
--
-- This script is a developer harness, not a publishable benchmark result. It
-- creates a synthetic corpus with clustered vectors, common terms, rare
-- identifier-like terms, phrase-like text, and INCLUDE payload columns. It then
-- rebuilds one hybrid turbohybrid index per config and reports recall/overlap,
-- duplicate groups, elapsed milliseconds, and selected last-scan stats. Do not
-- commit generated output.

\set ON_ERROR_STOP on
\pset pager off
\x off

\if :{?NROWS}
\else
  \set NROWS 10000
\endif
\if :{?DIMS}
\else
  \set DIMS 64
\endif
\if :{?FINAL_K}
\else
  \set FINAL_K 10
\endif
\if :{?DENSE_K}
\else
  \set DENSE_K 200
\endif
\if :{?BM25_K}
\else
  \set BM25_K 200
\endif
\if :{?EXACT_BASELINE_K}
\else
  \set EXACT_BASELINE_K 100
\endif

CREATE EXTENSION IF NOT EXISTS pgturbohybrid CASCADE;

SET max_parallel_workers_per_gather = 0;
SET jit = off;

\echo '== pgturbohybrid retrieval quality grid =='
\echo 'rows=' :NROWS ' dims=' :DIMS ' final_k=' :FINAL_K ' dense_k=' :DENSE_K ' bm25_k=' :BM25_K ' exact_baseline_k=' :EXACT_BASELINE_K

DROP TABLE IF EXISTS th_rq_docs CASCADE;
DROP TABLE IF EXISTS th_rq_cases CASCADE;
DROP TABLE IF EXISTS th_rq_configs CASCADE;
DROP TABLE IF EXISTS th_rq_results CASCADE;

CREATE TABLE th_rq_docs (
    id bigint PRIMARY KEY,
    embedding vector(:DIMS) NOT NULL,
    body text NOT NULL,
    body_tsv tsvector NOT NULL,
    group_id int NOT NULL,
    category_id int NOT NULL
);

\echo '== loading deterministic clustered corpus =='
WITH source AS (
    SELECT g::bigint AS id,
           ((g - 1) % 64)::int AS group_id,
           (((g - 1) / 64) % 8)::int AS category_id,
           ((g - 1) % 16)::int AS cluster_id,
           ((g % 7) + 1)::int AS common_weight
    FROM generate_series(1, :NROWS) AS g
),
vectors AS (
    SELECT s.*,
           (
               SELECT array_agg(
                          (
                              sin((s.cluster_id + 1) * 0.73 + d * 0.113) +
                              cos((s.category_id + 1) * 0.41 + d * 0.071) +
                              0.035 * sin(s.id * 0.017 + d * 0.029)
                          )::real
                          ORDER BY d)
               FROM generate_series(1, :DIMS) AS d
           )::real[]::vector AS embedding
    FROM source s
),
docs AS (
    SELECT v.id,
           v.embedding,
           v.group_id,
           v.category_id,
           concat_ws(' ',
               repeat('commonneedle ', v.common_weight),
               'shared broad corpus',
               'category' || v.category_id,
               'group' || v.group_id,
               'topic' || v.cluster_id,
               CASE WHEN v.id % 11 = 0 THEN 'phrasealpha phrasebeta phrasegamma' ELSE 'phrasealpha filler phrasebeta' END,
               'sku' || lpad(v.id::text, 6, '0') || 'abc',
               CASE WHEN v.id % 37 = 0 THEN 'rarebatch37' ELSE 'ordinarybatch' END
           ) AS body
    FROM vectors v
)
INSERT INTO th_rq_docs(id, embedding, body, body_tsv, group_id, category_id)
SELECT id,
       embedding::vector(:DIMS),
       body,
       to_tsvector('simple', body),
       group_id,
       category_id
FROM docs
ORDER BY id;

ANALYZE th_rq_docs;

CREATE TABLE th_rq_cases (
    case_name text PRIMARY KEY,
    case_kind text NOT NULL,
    qvec vector(:DIMS),
    text_query tsquery,
    filter_group_id int,
    filter_category_id int,
    require_bm25_match boolean NOT NULL DEFAULT false,
    exact_ids bigint[],
    expected_ids bigint[],
    notes text
);

\echo '== preparing deterministic query cases and baselines =='
WITH params AS (
    SELECT :FINAL_K::int AS final_k,
           :EXACT_BASELINE_K::int AS exact_baseline_k,
           greatest(1, least(:NROWS::int, :NROWS::int / 2))::bigint AS rare_id,
           LEAST(5, (SELECT max(category_id) FROM th_rq_docs))::int AS payload_category_id
),
case_seed AS (
    SELECT 'dense_nearest'::text AS case_name,
           'dense'::text AS case_kind,
           (SELECT embedding FROM th_rq_docs WHERE id = least(17, :NROWS)::bigint) AS qvec,
           NULL::tsquery AS text_query,
           NULL::int AS filter_group_id,
           NULL::int AS filter_category_id,
           false AS require_bm25_match,
           'exact nearest-neighbor query from an existing row'::text AS notes
    UNION ALL
    SELECT 'dense_ambiguous_cluster',
           'dense',
           (
               SELECT (
                   SELECT array_agg(
                              (
                                  sin((3 + 1) * 0.73 + d * 0.113) +
                                  cos((2 + 1) * 0.41 + d * 0.071)
                              )::real
                              ORDER BY d)
                   FROM generate_series(1, :DIMS) AS d
               )::real[]::vector(:DIMS)
           ),
           NULL::tsquery,
           NULL::int,
           NULL::int,
           false,
           'cluster-center query without a matching heap row'
    UNION ALL
    SELECT 'payload_filtered_dense',
           'dense',
           (SELECT embedding
            FROM th_rq_docs
            WHERE category_id = (SELECT payload_category_id FROM params)
            ORDER BY id
            LIMIT 1),
           NULL::tsquery,
           NULL::int,
           (SELECT payload_category_id FROM params),
           false,
           'dense query with graph-owned INCLUDE category_id filter'
    UNION ALL
    SELECT 'rare_lexical_identifier',
           'lexical',
           NULL::vector(:DIMS),
           to_tsquery('simple', 'sku' || lpad((SELECT rare_id FROM params)::text, 6, '0') || 'abc'),
           NULL::int,
           NULL::int,
           true,
           'single rare identifier-like BM25 term'
    UNION ALL
    SELECT 'common_lexical_category',
           'lexical',
           NULL::vector(:DIMS),
           to_tsquery('simple', 'commonneedle & category3'),
           NULL::int,
           NULL::int,
           true,
           'broad/common term constrained by a deterministic category token'
    UNION ALL
    SELECT 'phrase_lexical',
           'lexical',
           NULL::vector(:DIMS),
           to_tsquery('simple', 'phrasealpha <-> phrasebeta'),
           NULL::int,
           NULL::int,
           true,
           'phrase-like lexical query'
    UNION ALL
    SELECT 'hybrid_semantic_lexical',
           'hybrid',
           (SELECT embedding FROM th_rq_docs WHERE category_id = 2 AND group_id = 10 ORDER BY id LIMIT 1),
           to_tsquery('simple', 'topic10 & category2'),
           NULL::int,
           2,
           false,
           'semantic query anchored by matching lexical topic/category terms'
),
exact_dense AS (
    SELECT c.case_name,
           (
               SELECT array_agg(d.id ORDER BY d.embedding <=> c.qvec, d.id)
               FROM (
                   SELECT d.id, d.embedding
                   FROM th_rq_docs d
                   WHERE c.qvec IS NOT NULL
                     AND (c.filter_group_id IS NULL OR d.group_id = c.filter_group_id)
                     AND (c.filter_category_id IS NULL OR d.category_id = c.filter_category_id)
                   ORDER BY d.embedding <=> c.qvec, d.id
                   LIMIT (SELECT exact_baseline_k FROM params)
               ) d
           ) AS exact_ids
    FROM case_seed c
),
expected AS (
    SELECT c.case_name,
           CASE c.case_name
               WHEN 'rare_lexical_identifier' THEN
                   ARRAY[(SELECT rare_id FROM params)]::bigint[]
               WHEN 'common_lexical_category' THEN
                   (
                       SELECT array_agg(id ORDER BY common_weight DESC, id)
                       FROM (
                           SELECT d.id, ((d.id % 7) + 1)::int AS common_weight
                           FROM th_rq_docs d
                           WHERE d.body_tsv @@ to_tsquery('simple', 'commonneedle & category3')
                           ORDER BY ((d.id % 7) + 1)::int DESC, d.id
                           LIMIT (SELECT final_k FROM params)
                       ) s
                   )
               WHEN 'phrase_lexical' THEN
                   (
                       SELECT array_agg(id ORDER BY id)
                       FROM (
                           SELECT d.id
                           FROM th_rq_docs d
                           WHERE d.body_tsv @@ to_tsquery('simple', 'phrasealpha <-> phrasebeta')
                           ORDER BY d.id
                           LIMIT (SELECT final_k FROM params)
                       ) s
                   )
               WHEN 'hybrid_semantic_lexical' THEN
                   (
                       SELECT array_agg(id ORDER BY dist, id)
                       FROM (
                           SELECT d.id,
                                  d.embedding <=> c.qvec AS dist
                           FROM th_rq_docs d
                           WHERE d.body_tsv @@ c.text_query
                             AND d.category_id = 2
                           ORDER BY d.embedding <=> c.qvec, d.id
                           LIMIT (SELECT final_k FROM params)
                       ) s
                   )
               ELSE
                   (
                       SELECT (exact_ids)[1:(SELECT final_k FROM params)]
                       FROM exact_dense e
                       WHERE e.case_name = c.case_name
                   )
           END AS expected_ids
    FROM case_seed c
)
INSERT INTO th_rq_cases(case_name, case_kind, qvec, text_query, filter_group_id,
                        filter_category_id, require_bm25_match, exact_ids,
                        expected_ids, notes)
SELECT c.case_name,
       c.case_kind,
       c.qvec,
       c.text_query,
       c.filter_group_id,
       c.filter_category_id,
       c.require_bm25_match,
       e.exact_ids,
       x.expected_ids,
       c.notes
FROM case_seed c
LEFT JOIN exact_dense e USING (case_name)
LEFT JOIN expected x USING (case_name)
ORDER BY c.case_name;

CREATE TABLE th_rq_configs (
    config_order int PRIMARY KEY,
    label text NOT NULL UNIQUE,
    profile text NOT NULL,
    index_options text NOT NULL,
    fusion text NOT NULL DEFAULT 'rrf',
    alpha float8,
    dense_heap_rescore text NOT NULL DEFAULT 'auto',
    dense_residual_rerank_mode text NOT NULL DEFAULT 'calibrated',
    dense_adaptive_widening text NOT NULL DEFAULT 'auto',
    dense_uncertainty_retry text NOT NULL DEFAULT 'off',
    dense_uncertainty_retry_max_passes int NOT NULL DEFAULT 1,
    payload_entry_seeding text NOT NULL DEFAULT 'auto',
    payload_entry_seed_count int NOT NULL DEFAULT 8,
    final_diversity text NOT NULL DEFAULT 'off',
    final_diversity_payload_slot int NOT NULL DEFAULT -1,
    final_diversity_lambda float8 NOT NULL DEFAULT 0.75,
    final_diversity_pool_multiplier int NOT NULL DEFAULT 3,
    bm25_heap_tsvector_rerank text NOT NULL DEFAULT 'off',
    bm25_heap_tsvector_rerank_multiplier int NOT NULL DEFAULT 4,
    bm25_heap_tsvector_rerank_weight float8 NOT NULL DEFAULT 0.10
);

-- Promotion gate for named profile defaults:
--   * matched_recall uncertainty retry: compare rows 150/160/170 and promote
--     only if auto improves recall/overlap with low p95/p99 cost.
--   * high_recall residual rerank: compare rows 30/32/34/56 and promote
--     calibrated residual behavior only if it reduces the need for heap-band
--     rescoring on indexes built with residual sketches.
--   * quality calibrated fusion: compare rows 40/112/114 on hybrid cases and
--     promote only if hybrid overlap improves at acceptable latency.
--   * heap lexical rerank: compare rows 180/190/200 on phrase-like lexical and
--     hybrid cases; keep it opt-in unless phrase-shaped queries clearly win.
-- Generated result tables are benchmark artifacts and should not be committed.
INSERT INTO th_rq_configs(config_order, label, profile, index_options, fusion,
                          alpha, dense_heap_rescore, dense_residual_rerank_mode,
                          dense_adaptive_widening,
                          dense_uncertainty_retry, dense_uncertainty_retry_max_passes,
                          payload_entry_seeding, payload_entry_seed_count,
                          final_diversity, final_diversity_payload_slot,
                          final_diversity_lambda,
                          final_diversity_pool_multiplier,
                          bm25_heap_tsvector_rerank,
                          bm25_heap_tsvector_rerank_multiplier,
                          bm25_heap_tsvector_rerank_weight)
VALUES
    (10, 'latency', 'latency',
     'quantization_bits = 4, exact_storage = off, native_segments = 1',
     'rrf', NULL, 'auto', 'calibrated', 'auto', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (20, 'matched_recall', 'matched_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = 1',
     'rrf', NULL, 'auto', 'calibrated', 'auto', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (22, 'matched_recall_4bit_baseline', 'matched_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = 1',
     'rrf', NULL, 'auto', 'calibrated', 'auto', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (24, 'matched_recall_4bit_residual32', 'matched_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = 1, residual_rerank = on, residual_rerank_bytes = 32',
     'rrf', NULL, 'auto', 'calibrated', 'auto', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (26, 'matched_recall_8bit_scalar', 'matched_recall',
     'quantization_bits = 8, exact_storage = off, native_segments = 1',
     'rrf', NULL, 'auto', 'calibrated', 'auto', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (28, 'matched_recall_exact_storage', 'matched_recall',
     'quantization_bits = 4, exact_storage = on, native_segments = 1',
     'rrf', NULL, 'auto', 'calibrated', 'auto', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (30, 'high_recall', 'high_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = 1',
     'rrf', NULL, 'auto', 'calibrated', 'auto', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (32, 'high_recall_residual32_calibrated_heap_off', 'high_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = 1, residual_rerank = on, residual_rerank_bytes = 32',
     'rrf', NULL, 'off', 'calibrated', 'auto', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (34, 'high_recall_residual32_fixed_heap_off', 'high_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = 1, residual_rerank = on, residual_rerank_bytes = 32',
     'rrf', NULL, 'off', 'fixed', 'auto', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (40, 'quality', 'quality',
     'quantization_bits = 4, exact_storage = off, native_segments = 1',
     'rrf', NULL, 'auto', 'calibrated', 'auto', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (50, 'matched_recall_residual32_calibrated', 'matched_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = 1, residual_rerank = on, residual_rerank_bytes = 32',
     'rrf', NULL, 'auto', 'calibrated', 'auto', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (52, 'matched_recall_residual32_fixed', 'matched_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = 1, residual_rerank = on, residual_rerank_bytes = 32',
     'rrf', NULL, 'auto', 'fixed', 'auto', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (54, 'matched_recall_residual32_off', 'matched_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = 1, residual_rerank = on, residual_rerank_bytes = 32',
     'rrf', NULL, 'auto', 'off', 'auto', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (56, 'matched_recall_residual32_heap_band', 'matched_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = 1, residual_rerank = on, residual_rerank_bytes = 32',
     'rrf', NULL, 'band', 'calibrated', 'auto', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (60, 'matched_recall_entry_sidecar_off', 'matched_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = 1, entry_sidecar = off',
     'rrf', NULL, 'auto', 'calibrated', 'auto', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (70, 'matched_recall_entry_sidecar_hash128', 'matched_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = 1, entry_sidecar = on, entry_sidecar_representatives = 128, entry_sidecar_strategy = hash',
     'rrf', NULL, 'auto', 'calibrated', 'auto', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (80, 'matched_recall_entry_sidecar_farthest128', 'matched_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = 1, entry_sidecar = on, entry_sidecar_representatives = 128, entry_sidecar_strategy = farthest_code',
     'rrf', NULL, 'auto', 'calibrated', 'auto', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (90, 'matched_recall_entry_sidecar_hybrid128', 'matched_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = 1, entry_sidecar = on, entry_sidecar_representatives = 128, entry_sidecar_strategy = hybrid_level_covering',
     'rrf', NULL, 'auto', 'calibrated', 'auto', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (100, 'quality_residual32_entry_sidecar128', 'quality',
     'quantization_bits = 4, exact_storage = off, native_segments = 1, residual_rerank = on, residual_rerank_bytes = 32, entry_sidecar = on, entry_sidecar_representatives = 128, entry_sidecar_strategy = hybrid_level_covering',
     'rrf', NULL, 'auto', 'calibrated', 'auto', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (110, 'fast_weighted_alpha_0_50', 'matched_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = 1',
     'fast_weighted', 0.50, 'auto', 'calibrated', 'auto', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (112, 'calibrated_fusion', 'matched_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = 1',
     'calibrated', NULL, 'auto', 'calibrated', 'auto', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (114, 'quality_calibrated_fusion', 'quality',
     'quantization_bits = 4, exact_storage = off, native_segments = 1',
     'calibrated', NULL, 'auto', 'calibrated', 'auto', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (120, 'matched_recall_payload_seed_off', 'matched_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = 1',
     'rrf', NULL, 'auto', 'calibrated', 'auto', 'off', 1, 'off', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (130, 'matched_recall_payload_seed_auto8', 'matched_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = 1',
     'rrf', NULL, 'auto', 'calibrated', 'auto', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (140, 'matched_recall_payload_seed_on16', 'matched_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = 1',
     'rrf', NULL, 'auto', 'calibrated', 'auto', 'off', 1, 'on', 16, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (150, 'matched_recall_uncertainty_retry_off', 'matched_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = 1',
     'rrf', NULL, 'auto', 'calibrated', 'off', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (160, 'matched_recall_uncertainty_retry_auto', 'matched_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = 1',
     'rrf', NULL, 'auto', 'calibrated', 'off', 'auto', 1, 'auto', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (170, 'matched_recall_uncertainty_retry_on', 'matched_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = 1',
     'rrf', NULL, 'auto', 'calibrated', 'off', 'on', 1, 'auto', 8, 'off', -1, 0.75, 3, 'off', 4, 0.10),
    (180, 'matched_recall_bm25_heap_tsvector_topk', 'matched_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = 1',
     'rrf', NULL, 'auto', 'calibrated', 'auto', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'topk', 4, 0.10),
    (190, 'matched_recall_bm25_heap_tsvector_band', 'matched_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = 1',
     'rrf', NULL, 'auto', 'calibrated', 'auto', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'band', 4, 0.10),
    (200, 'matched_recall_bm25_heap_tsvector_auto', 'matched_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = 1',
     'rrf', NULL, 'auto', 'calibrated', 'auto', 'off', 1, 'auto', 8, 'off', -1, 0.75, 3, 'auto', 4, 0.10),
    (210, 'matched_recall_final_diversity_group_payload', 'matched_recall',
     'quantization_bits = 4, exact_storage = off, native_segments = 1',
     'rrf', NULL, 'auto', 'calibrated', 'auto', 'off', 1, 'auto', 8, 'group_payload', 0, 0.50, 3, 'off', 4, 0.10);

CREATE TABLE th_rq_results (
    config_order int NOT NULL,
    label text NOT NULL,
    case_name text NOT NULL,
    case_kind text NOT NULL,
    profile text NOT NULL,
    fusion text NOT NULL,
    alpha float8,
    recall_at_k float8,
    overlap_at_k float8,
    duplicate_group_count int NOT NULL,
    elapsed_ms float8 NOT NULL,
    rows_returned int NOT NULL,
    result_ids bigint[],
    expected_ids bigint[],
    scan_stats jsonb NOT NULL,
    index_stats jsonb NOT NULL,
    repair_stats jsonb NOT NULL,
    PRIMARY KEY (label, case_name)
);

CREATE OR REPLACE FUNCTION pg_temp.th_rq_overlap(a bigint[], b bigint[])
RETURNS int
LANGUAGE sql
AS $$
    SELECT count(*)::int
    FROM (
        SELECT unnest(COALESCE(a, '{}'::bigint[]))
        INTERSECT
        SELECT unnest(COALESCE(b, '{}'::bigint[]))
    ) s
$$;

CREATE OR REPLACE FUNCTION pg_temp.th_rq_duplicate_groups(ids bigint[])
RETURNS int
LANGUAGE sql
AS $$
    SELECT COALESCE(sum(group_rows - 1), 0)::int
    FROM (
        SELECT d.group_id, count(*) AS group_rows
        FROM unnest(COALESCE(ids, '{}'::bigint[])) AS r(id)
        JOIN th_rq_docs d USING (id)
        GROUP BY d.group_id
        HAVING count(*) > 1
    ) s
$$;

CREATE TEMP TABLE th_rq_params AS
SELECT :DENSE_K::int AS dense_k,
       :BM25_K::int AS bm25_k,
       :FINAL_K::int AS final_k;

\echo '== running retrieval quality grid =='
DO $$
DECLARE
    cfg record;
    q record;
    t0 timestamptz;
    ids bigint[];
    elapsed_ms float8;
    st jsonb;
    idxst jsonb;
    repairst jsonb;
    selected_st jsonb;
    expected_at_k bigint[];
    exact_at_k bigint[];
    warm_ids bigint[];
    expected_count int;
    exact_count int;
    effective_dense_k int;
    p_dense_k int := 200;
    p_bm25_k int := 200;
    p_final_k int := 10;
    alpha_sql text;
    filter_sql text;
    query_expr text;
    query_sql text;
    text_sql text;
    vector_sql text;
BEGIN
    SELECT dense_k, bm25_k, final_k
    INTO p_dense_k, p_bm25_k, p_final_k
    FROM th_rq_params;

    FOR cfg IN SELECT * FROM th_rq_configs ORDER BY config_order LOOP
        EXECUTE 'DROP INDEX IF EXISTS th_rq_idx';
        PERFORM set_config('turbohybrid.profile', cfg.profile, false);
        PERFORM set_config('turbohybrid.dense_heap_rescore', cfg.dense_heap_rescore, false);
        PERFORM set_config('turbohybrid.dense_residual_rerank_mode', cfg.dense_residual_rerank_mode, false);
        PERFORM set_config('turbohybrid.dense_adaptive_widening', cfg.dense_adaptive_widening, false);
        PERFORM set_config('turbohybrid.dense_uncertainty_retry', cfg.dense_uncertainty_retry, false);
        PERFORM set_config('turbohybrid.dense_uncertainty_retry_max_passes', cfg.dense_uncertainty_retry_max_passes::text, false);
        PERFORM set_config('turbohybrid.payload_entry_seeding', cfg.payload_entry_seeding, false);
        PERFORM set_config('turbohybrid.payload_entry_seed_count', cfg.payload_entry_seed_count::text, false);
        PERFORM set_config('turbohybrid.final_diversity', cfg.final_diversity, false);
        PERFORM set_config('turbohybrid.final_diversity_payload_slot', cfg.final_diversity_payload_slot::text, false);
        PERFORM set_config('turbohybrid.final_diversity_lambda', cfg.final_diversity_lambda::text, false);
        PERFORM set_config('turbohybrid.final_diversity_pool_multiplier', cfg.final_diversity_pool_multiplier::text, false);
        PERFORM set_config('turbohybrid.bm25_heap_tsvector_rerank', cfg.bm25_heap_tsvector_rerank, false);
        PERFORM set_config('turbohybrid.bm25_heap_tsvector_rerank_multiplier', cfg.bm25_heap_tsvector_rerank_multiplier::text, false);
        PERFORM set_config('turbohybrid.bm25_heap_tsvector_rerank_weight', cfg.bm25_heap_tsvector_rerank_weight::text, false);

        RAISE NOTICE 'building config % profile=% residual_mode=% uncertainty_retry=% payload_entry_seeding=% seed_count=% final_diversity=% bm25_heap_tsvector=% options=(%)',
            cfg.label, cfg.profile, cfg.dense_residual_rerank_mode,
            cfg.dense_uncertainty_retry, cfg.payload_entry_seeding,
            cfg.payload_entry_seed_count, cfg.final_diversity,
            cfg.bm25_heap_tsvector_rerank, cfg.index_options;
        EXECUTE format(
            'CREATE INDEX th_rq_idx ON th_rq_docs USING turbohybrid ' ||
            '(embedding vector_cosine_turbohybrid_ops, body_tsv bm25_tsvector_turbohybrid_ops) ' ||
            'INCLUDE (group_id, category_id) WITH (%s)',
            cfg.index_options);
        ANALYZE th_rq_docs;
	    idxst := turbohybrid_index_stats('th_rq_idx'::regclass);
	    repairst := turbohybrid_graph_repair_dry_run(
	        'th_rq_idx'::regclass,
	        LEAST(128, GREATEST(1, (SELECT count(*)::int FROM th_rq_docs) / 100)),
	        GREATEST(64, LEAST(400, p_dense_k * 2)),
	        GREATEST(32, LEAST(200, p_dense_k)));

        PERFORM set_config('enable_seqscan', 'off', false);

        FOR q IN SELECT * FROM th_rq_cases ORDER BY case_name LOOP
            vector_sql := CASE
                WHEN q.qvec IS NULL THEN 'NULL::vector'
                ELSE quote_literal(q.qvec::text) || '::vector'
            END;
            text_sql := CASE
                WHEN q.text_query IS NULL THEN 'NULL::tsquery'
                ELSE quote_literal(q.text_query::text) || '::tsquery'
            END;
            alpha_sql := CASE
                WHEN cfg.alpha IS NULL THEN 'NULL::float8'
                ELSE cfg.alpha::text || '::float8'
            END;
            filter_sql := 'TRUE';
            IF q.filter_group_id IS NOT NULL THEN
                filter_sql := filter_sql || format(' AND d.group_id = %s', q.filter_group_id);
            END IF;
            IF q.filter_category_id IS NOT NULL THEN
                filter_sql := filter_sql || format(' AND d.category_id = %s', q.filter_category_id);
            END IF;
            effective_dense_k := CASE WHEN q.qvec IS NULL THEN 0 ELSE p_dense_k END;
            query_expr := format(
                'turbohybrid_query(vector_query => %s, text_query => %s, fusion => %L, alpha => %s, dense_k => %s, bm25_k => %s, final_k => %s, require_bm25_match => %L)',
                vector_sql,
                text_sql,
                cfg.fusion,
                alpha_sql,
                effective_dense_k,
                p_bm25_k,
                p_final_k,
                q.require_bm25_match);
            query_sql := format(
                'SELECT array_agg(id) FROM (SELECT d.id FROM th_rq_docs d WHERE %s ORDER BY d.embedding <~> %s LIMIT %s) s',
                filter_sql,
                query_expr,
                p_final_k);

            -- One warm query keeps per-config cold attach/build from dominating
            -- every first case while still recording stats for the measured
            -- query below.
            EXECUTE query_sql INTO warm_ids;

            t0 := clock_timestamp();
            EXECUTE query_sql INTO ids;
            elapsed_ms := extract(epoch FROM clock_timestamp() - t0) * 1000.0;
            st := turbohybrid_last_scan_stats();

            expected_at_k := (q.expected_ids)[1:LEAST(p_final_k, COALESCE(array_length(q.expected_ids, 1), 0))];
            exact_at_k := (q.exact_ids)[1:LEAST(p_final_k, COALESCE(array_length(q.exact_ids, 1), 0))];
            expected_count := GREATEST(1, COALESCE(array_length(expected_at_k, 1), 0));
            exact_count := GREATEST(1, COALESCE(array_length(exact_at_k, 1), 0));

            selected_st := jsonb_build_object(
                'index_used', st->>'index_used',
                'scan_orchestration', st->>'scan_orchestration',
                'profile', st->>'profile',
                'fusion', st->>'fusion',
                'dense_k_effective', st->>'dense_k_effective',
                'bm25_k_effective', st->>'bm25_k_effective',
                'final_k_effective', st->>'final_k_effective',
                'dense_candidates_effective', st->>'dense_candidates_effective',
                'bm25_candidates_effective', st->>'bm25_candidates_effective',
                'graph_effective_search_ef', st->>'graph_effective_search_ef',
                'graph_effective_result_target', st->>'graph_effective_result_target',
                'graph_scored_codes', st->>'graph_scored_codes',
                'graph_batch_us', st->>'graph_batch_us',
                'graph_base_us', st->>'graph_base_us',
                'graph_traverse_us', st->>'graph_traverse_us',
                'heap_rescore_count', st->>'heap_rescore_count',
                'heap_fetch_us', st->>'heap_fetch_us',
                'heap_rescore_us', st->>'heap_rescore_us',
                'exact_rescore_source', st->>'exact_rescore_source',
                'dense_residual_rerank_count', st->>'dense_residual_rerank_count',
                'dense_residual_rerank_bytes', st->>'dense_residual_rerank_bytes',
                'dense_residual_rerank_us', st->>'dense_residual_rerank_us',
                'residual_rerank_mode', st->>'residual_rerank_mode',
                'residual_rerank_weight_effective', st->>'residual_rerank_weight_effective',
                'residual_rerank_band', st->>'residual_rerank_band',
                'residual_rerank_max_adjustment', st->>'residual_rerank_max_adjustment',
                'residual_rerank_reordered_count', st->>'residual_rerank_reordered_count',
                'residual_rerank_topk_changed', st->>'residual_rerank_topk_changed',
                'calibrated_fusion_enabled', st->>'calibrated_fusion_enabled',
                'calibrated_fusion_query_shape', st->>'calibrated_fusion_query_shape',
                'calibrated_fusion_alpha_effective', st->>'calibrated_fusion_alpha_effective',
                'calibrated_fusion_both_match_bonus', st->>'calibrated_fusion_both_match_bonus',
                'calibrated_fusion_dense_norm_mode', st->>'calibrated_fusion_dense_norm_mode',
                'calibrated_fusion_bm25_norm_mode', st->>'calibrated_fusion_bm25_norm_mode'
            ) || jsonb_build_object(
                'dense_uncertainty_retry_mode', st->>'dense_uncertainty_retry_mode',
                'dense_uncertainty_retry_triggered', st->>'dense_uncertainty_retry_triggered',
                'dense_uncertainty_retry_reason', st->>'dense_uncertainty_retry_reason',
                'dense_uncertainty_retry_passes', st->>'dense_uncertainty_retry_passes',
                'dense_uncertainty_initial_target', st->>'dense_uncertainty_initial_target',
                'dense_uncertainty_final_target', st->>'dense_uncertainty_final_target',
                'dense_uncertainty_initial_ef', st->>'dense_uncertainty_initial_ef',
                'dense_uncertainty_final_ef', st->>'dense_uncertainty_final_ef',
                'dense_uncertainty_gap_top10', st->>'dense_uncertainty_gap_top10',
                'dense_uncertainty_gap_boundary', st->>'dense_uncertainty_gap_boundary',
                'graph_entry_sidecar_count', st->>'graph_entry_sidecar_count',
                'graph_entry_sidecar_scored', st->>'graph_entry_sidecar_scored',
                'graph_entry_sidecar_selected', st->>'graph_entry_sidecar_selected',
                'graph_entry_sidecar_representatives_configured', st->>'graph_entry_sidecar_representatives_configured',
                'graph_entry_sidecar_strategy', st->>'graph_entry_sidecar_strategy',
                'graph_entry_sidecar_us', st->>'graph_entry_sidecar_us',
                'payload_entry_seeding_mode', st->>'payload_entry_seeding_mode',
                'payload_entry_seeding_hit', st->>'payload_entry_seeding_hit',
                'payload_entry_seed_count', st->>'payload_entry_seed_count',
                'payload_entry_seed_payload_slot', st->>'payload_entry_seed_payload_slot',
                'payload_entry_seed_range_count', st->>'payload_entry_seed_range_count',
                'payload_entry_seed_us', st->>'payload_entry_seed_us',
                'final_diversity_mode', st->>'final_diversity_mode',
                'final_diversity_payload_slot', st->>'final_diversity_payload_slot',
                'final_diversity_pool_size', st->>'final_diversity_pool_size',
                'final_diversity_selected', st->>'final_diversity_selected',
                'final_diversity_duplicate_groups_suppressed', st->>'final_diversity_duplicate_groups_suppressed',
                'final_diversity_us', st->>'final_diversity_us',
                'bm25_query_shape', st->>'bm25_query_shape',
                'bm25_strategy_selected', st->>'bm25_strategy_selected',
                'bm25_postings_visited', st->>'bm25_postings_visited',
                'bm25_blocks_skipped', st->>'bm25_blocks_skipped',
                'bm25_heap_tsvector_rerank_mode', st->>'bm25_heap_tsvector_rerank_mode',
                'bm25_heap_tsvector_rerank_count', st->>'bm25_heap_tsvector_rerank_count',
                'bm25_heap_tsvector_rerank_fetch_us', st->>'bm25_heap_tsvector_rerank_fetch_us',
                'bm25_heap_tsvector_rerank_score_us', st->>'bm25_heap_tsvector_rerank_score_us',
                'bm25_heap_tsvector_rerank_topk_changed', st->>'bm25_heap_tsvector_rerank_topk_changed',
                'native_cache_scope', st->>'native_cache_scope',
                'native_cache_used', st->>'native_cache_used'
            );

            INSERT INTO th_rq_results(config_order, label, case_name, case_kind,
                                      profile, fusion, alpha, recall_at_k,
                                      overlap_at_k, duplicate_group_count,
                                      elapsed_ms, rows_returned, result_ids,
                                      expected_ids, scan_stats, index_stats,
                                      repair_stats)
            VALUES (
                cfg.config_order,
                cfg.label,
                q.case_name,
                q.case_kind,
                cfg.profile,
                cfg.fusion,
                cfg.alpha,
                CASE WHEN q.case_kind = 'dense'
                     THEN pg_temp.th_rq_overlap(ids, exact_at_k)::float8 / exact_count
                     ELSE NULL
                END,
                pg_temp.th_rq_overlap(ids, expected_at_k)::float8 / expected_count,
                pg_temp.th_rq_duplicate_groups(ids),
                elapsed_ms,
                COALESCE(array_length(ids, 1), 0),
                COALESCE(ids, '{}'::bigint[]),
                COALESCE(expected_at_k, '{}'::bigint[]),
                selected_st,
                idxst,
                repairst
            );
        END LOOP;
    END LOOP;
END
$$;

\echo '== compact result table =='
SELECT label,
       case_name,
       case_kind,
       profile,
       fusion,
       round(recall_at_k::numeric, 4) AS recall_at_k,
       round(overlap_at_k::numeric, 4) AS overlap_at_k,
       duplicate_group_count AS dup_groups,
       round(elapsed_ms::numeric, 3) AS elapsed_ms,
       rows_returned AS rows,
       scan_stats->>'index_used' AS index_used,
       scan_stats->>'scan_orchestration' AS orchestration,
       scan_stats->>'graph_effective_search_ef' AS graph_ef,
       scan_stats->>'graph_effective_result_target' AS result_target,
       scan_stats->>'graph_scored_codes' AS scored_codes,
       scan_stats->>'heap_rescore_count' AS heap_rescore,
       scan_stats->>'exact_rescore_source' AS rescore_source,
       scan_stats->>'residual_rerank_mode' AS residual_mode,
       scan_stats->>'residual_rerank_band' AS residual_band,
       scan_stats->>'residual_rerank_topk_changed' AS residual_topk_changed,
       scan_stats->>'calibrated_fusion_alpha_effective' AS calibrated_alpha,
       scan_stats->>'dense_uncertainty_retry_mode' AS retry_mode,
       scan_stats->>'dense_uncertainty_retry_triggered' AS retry_hit,
       scan_stats->>'dense_uncertainty_retry_reason' AS retry_reason,
       scan_stats->>'dense_uncertainty_retry_passes' AS retry_passes,
       scan_stats->>'graph_entry_sidecar_strategy' AS sidecar_strategy,
       scan_stats->>'graph_entry_sidecar_scored' AS sidecar_scored,
       scan_stats->>'payload_entry_seeding_mode' AS payload_seed_mode,
       scan_stats->>'payload_entry_seeding_hit' AS payload_seed_hit,
       scan_stats->>'payload_entry_seed_count' AS payload_seeds,
       scan_stats->>'payload_entry_seed_range_count' AS payload_seed_range,
       scan_stats->>'final_diversity_mode' AS final_diversity,
       scan_stats->>'final_diversity_selected' AS diversity_selected,
       scan_stats->>'final_diversity_duplicate_groups_suppressed' AS diversity_suppressed,
       scan_stats->>'final_diversity_us' AS diversity_us,
       scan_stats->>'bm25_query_shape' AS bm25_shape,
       scan_stats->>'bm25_strategy_selected' AS bm25_strategy,
       scan_stats->>'bm25_heap_tsvector_rerank_mode' AS bm25_heap_tsv_mode,
       scan_stats->>'bm25_heap_tsvector_rerank_count' AS bm25_heap_tsv_count,
       scan_stats->>'bm25_heap_tsvector_rerank_topk_changed' AS bm25_heap_tsv_topk_changed,
       repair_stats->>'avg_overlap' AS repair_avg_overlap,
       repair_stats->>'weak_nodes' AS repair_weak_nodes,
       repair_stats->>'suggested_edges' AS repair_suggested_edges,
       repair_stats->>'elapsed_ms' AS repair_elapsed_ms
FROM th_rq_results
ORDER BY config_order, case_name;

\echo '== config summary =='
SELECT label,
       profile,
       round(avg(recall_at_k) FILTER (WHERE case_kind = 'dense')::numeric, 4) AS avg_dense_recall_at_k,
       round(avg(overlap_at_k) FILTER (WHERE case_kind IN ('lexical', 'hybrid'))::numeric, 4) AS avg_lexical_hybrid_overlap_at_k,
       round(avg(elapsed_ms)::numeric, 3) AS avg_elapsed_ms,
       round(percentile_cont(0.95) WITHIN GROUP (ORDER BY elapsed_ms)::numeric, 3) AS p95_elapsed_ms,
       sum(duplicate_group_count) AS duplicate_groups_total
FROM th_rq_results
GROUP BY config_order, label, profile
ORDER BY config_order;

\echo '== selected index settings by config =='
SELECT DISTINCT ON (config_order)
       label,
       index_stats->>'profile' AS index_profile,
       index_stats->>'index_shape' AS index_shape,
       index_stats->>'quantization_bits' AS quantization_bits,
       index_stats->>'exact_storage' AS exact_storage,
       index_stats->>'dense_build_exact_distances' AS dense_build_exact_distances,
       index_stats->>'dense_build_distance_mode' AS dense_build_distance_mode,
       index_stats->>'build_neighbor_select' AS build_neighbor_select,
       index_stats->>'build_neighbor_select_reason' AS build_neighbor_select_reason,
       index_stats->>'graph_ef_construction' AS ef_construction,
       index_stats->>'graph_ef_search' AS ef_search,
       index_stats->>'graph_oversampling' AS oversampling,
       index_stats->>'native_segments' AS native_segments,
       index_stats->>'residual_rerank_bytes' AS residual_rerank_bytes,
       index_stats->>'entry_sidecar_count' AS entry_sidecar_count,
       index_stats->>'entry_sidecar_representatives_configured' AS entry_sidecar_representatives,
       index_stats->>'entry_sidecar_strategy' AS entry_sidecar_strategy,
       repair_stats->>'sampled_nodes' AS repair_sampled_nodes,
       repair_stats->>'avg_overlap' AS repair_avg_overlap,
       repair_stats->>'weak_nodes' AS repair_weak_nodes,
       repair_stats->>'suggested_edges' AS repair_suggested_edges
FROM th_rq_results
ORDER BY config_order, label;
