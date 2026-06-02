-- Scaling smoke for BM25 cold/warm caches, common-term decoding, and large
-- hybrid fusion candidate budgets.
--
-- Builds a deterministic hybrid index with one very common token and several
-- sparse tokens.  Runs common-term text-only scans twice, a rare-term scan, and
-- a hybrid scan with deliberately large dense_k/bm25_k budgets.
--
-- Expected trend: the first text scan can show docstats/liveness loaded in this
-- query and cold-cache O(N) work.  The second scan should be warm.  Common-term
-- scans should decode far more postings than rare-term scans, and the large
-- hybrid scan should expose fusion_generation_array_* / fusion_* stats.
--
-- Usage:
--   psql -d <db> -f benchmarks/dev/bm25_cold_warm_bench.sql
--   psql -d <db> -v NROWS=20000 -v COMMON_BM25_K=1000 -v HYBRID_K=2000 \
--        -f benchmarks/dev/bm25_cold_warm_bench.sql

\set ON_ERROR_STOP on
\pset pager off
\x auto
\timing on

\if :{?NROWS}
\else
  \set NROWS 5000
\endif
\if :{?DIMS}
\else
  \set DIMS 16
\endif
\if :{?COMMON_BM25_K}
\else
  \set COMMON_BM25_K 500
\endif
\if :{?RARE_BM25_K}
\else
  \set RARE_BM25_K 50
\endif
\if :{?HYBRID_K}
\else
  \set HYBRID_K 1000
\endif
\if :{?FINAL_K}
\else
  \set FINAL_K 10
\endif

\echo 'bm25_cold_warm_bench config:'
\echo '  NROWS         =' :NROWS
\echo '  DIMS          =' :DIMS
\echo '  COMMON_BM25_K =' :COMMON_BM25_K
\echo '  RARE_BM25_K   =' :RARE_BM25_K
\echo '  HYBRID_K      =' :HYBRID_K
\echo '  FINAL_K       =' :FINAL_K

CREATE EXTENSION IF NOT EXISTS pgturbohybrid CASCADE;

DROP TABLE IF EXISTS th_bm25_cold_warm_docs;
CREATE TABLE th_bm25_cold_warm_docs (
    id int PRIMARY KEY,
    embedding vector(:DIMS) NOT NULL,
    body text NOT NULL,
    body_tsv tsvector NOT NULL
);

INSERT INTO th_bm25_cold_warm_docs
SELECT g,
       (
           SELECT array_agg(
                      (sin(g * 0.009 + d * 0.041) +
                       0.10 * cos(g * 0.071 + d * 0.011))::real
                      ORDER BY d)
           FROM generate_series(1, :DIMS) AS d
       )::real[]::vector,
       concat_ws(' ',
           'common',
           'common',
           'topic' || (g % 32),
           CASE WHEN g % 250 = 0 THEN 'rareneedle' ELSE 'background' END,
           CASE WHEN g % 17 = 0 THEN 'mediumterm' ELSE 'filler' END,
           'doc' || g
       ) AS body,
       to_tsvector('simple',
           concat_ws(' ',
               'common',
               'common',
               'topic' || (g % 32),
               CASE WHEN g % 250 = 0 THEN 'rareneedle' ELSE 'background' END,
               CASE WHEN g % 17 = 0 THEN 'mediumterm' ELSE 'filler' END,
               'doc' || g
           )
       ) AS body_tsv
FROM generate_series(1, :NROWS) AS g;

CREATE INDEX th_bm25_cold_warm_idx ON th_bm25_cold_warm_docs
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (quantization_bits = 4, exact_storage = off);

ANALYZE th_bm25_cold_warm_docs;

SET enable_seqscan = off;
SET max_parallel_workers_per_gather = 0;
SET jit = off;
SET turbohybrid.enable_wand = on;
SET turbohybrid.bm25_strategy = wand;
-- Lower the diagnostic fallback threshold so the small synthetic corpus can
-- expose bm25_common_term_fallback for the common token.
SET turbohybrid.bm25_common_term_fallback_min_postings = 100;

CREATE TEMP TABLE th_bm25_cold_warm_config AS
SELECT :COMMON_BM25_K::int AS common_bm25_k,
       :RARE_BM25_K::int AS rare_bm25_k,
       :HYBRID_K::int AS hybrid_k,
       :FINAL_K::int AS final_k;

CREATE TEMP TABLE th_bm25_cold_warm_results (
    scenario text NOT NULL,
    elapsed_ms float8 NOT NULL,
    rows_returned int NOT NULL,
    stats jsonb NOT NULL
);

DO $$
DECLARE
    qv vector;
    t0 timestamptz;
    n int;
    st jsonb;
    common_bm25_k int;
    rare_bm25_k int;
    hybrid_k int;
    final_k int;
BEGIN
    SELECT c.common_bm25_k, c.rare_bm25_k, c.hybrid_k, c.final_k
    INTO common_bm25_k, rare_bm25_k, hybrid_k, final_k
    FROM th_bm25_cold_warm_config c;

    SELECT embedding INTO qv
    FROM th_bm25_cold_warm_docs
    WHERE id = 1;

    t0 := clock_timestamp();
    SELECT count(*) INTO n
    FROM (
        SELECT id
        FROM th_bm25_cold_warm_docs
        ORDER BY embedding <~> turbohybrid_query(
            text_query => websearch_to_tsquery('simple', 'common'),
            dense_k => 0,
            bm25_k => common_bm25_k,
            final_k => final_k
        )
        LIMIT final_k
    ) s;
    st := turbohybrid_last_scan_stats();
    INSERT INTO th_bm25_cold_warm_results
    VALUES ('common_text_cold',
            extract(epoch FROM clock_timestamp() - t0) * 1000.0,
            n,
            jsonb_build_object(
                'index_shape', st->'index_shape',
                'dense_branch_used', st->'dense_branch_used',
                'bm25_branch_used', st->'bm25_branch_used',
                'bm25_docstats_loaded_this_query', st->'bm25_docstats_loaded_this_query',
                'bm25_liveness_loaded_this_query', st->'bm25_liveness_loaded_this_query',
                'bm25_docstats_bytes', st->'bm25_docstats_bytes',
                'bm25_liveness_bytes', st->'bm25_liveness_bytes',
                'bm25_cold_cache_o_n_work', st->'bm25_cold_cache_o_n_work',
                'bm25_cache_hit', st->'bm25_cache_hit',
                'bm25_cache_build_us', st->'bm25_cache_build_us',
                'bm25_postings_decoded', st->'bm25_postings_decoded',
                'bm25_postings_decode_ratio', st->'bm25_postings_decode_ratio',
                'bm25_common_term_fallback', st->'bm25_common_term_fallback',
                'bm25_wand_pruned', st->'bm25_wand_pruned',
                'bm25_elapsed_us', st->'bm25_elapsed_us',
                'bm25', st->'bm25'
            ));

    PERFORM set_config('turbohybrid.enable_wand', 'off', true);
    t0 := clock_timestamp();
    SELECT count(*) INTO n
    FROM (
        SELECT id
        FROM th_bm25_cold_warm_docs
        ORDER BY embedding <~> turbohybrid_query(
            text_query => websearch_to_tsquery('simple', 'common'),
            dense_k => 0,
            bm25_k => common_bm25_k,
            final_k => final_k
        )
        LIMIT final_k
    ) s;
    st := turbohybrid_last_scan_stats();
    INSERT INTO th_bm25_cold_warm_results
    VALUES ('common_text_no_wand',
            extract(epoch FROM clock_timestamp() - t0) * 1000.0,
            n,
            jsonb_build_object(
                'index_shape', st->'index_shape',
                'dense_branch_used', st->'dense_branch_used',
                'bm25_branch_used', st->'bm25_branch_used',
                'bm25_docstats_loaded_this_query', st->'bm25_docstats_loaded_this_query',
                'bm25_liveness_loaded_this_query', st->'bm25_liveness_loaded_this_query',
                'bm25_docstats_bytes', st->'bm25_docstats_bytes',
                'bm25_liveness_bytes', st->'bm25_liveness_bytes',
                'bm25_cold_cache_o_n_work', st->'bm25_cold_cache_o_n_work',
                'bm25_cache_hit', st->'bm25_cache_hit',
                'bm25_cache_build_us', st->'bm25_cache_build_us',
                'bm25_postings_decoded', st->'bm25_postings_decoded',
                'bm25_postings_decode_ratio', st->'bm25_postings_decode_ratio',
                'bm25_common_term_fallback', st->'bm25_common_term_fallback',
                'bm25_wand_pruned', st->'bm25_wand_pruned',
                'bm25_elapsed_us', st->'bm25_elapsed_us',
                'bm25', st->'bm25'
            ));
    PERFORM set_config('turbohybrid.enable_wand', 'on', true);

    t0 := clock_timestamp();
    SELECT count(*) INTO n
    FROM (
        SELECT id
        FROM th_bm25_cold_warm_docs
        ORDER BY embedding <~> turbohybrid_query(
            text_query => websearch_to_tsquery('simple', 'common'),
            dense_k => 0,
            bm25_k => common_bm25_k,
            final_k => final_k
        )
        LIMIT final_k
    ) s;
    st := turbohybrid_last_scan_stats();
    INSERT INTO th_bm25_cold_warm_results
    VALUES ('common_text_warm',
            extract(epoch FROM clock_timestamp() - t0) * 1000.0,
            n,
            jsonb_build_object(
                'index_shape', st->'index_shape',
                'dense_branch_used', st->'dense_branch_used',
                'bm25_branch_used', st->'bm25_branch_used',
                'bm25_docstats_loaded_this_query', st->'bm25_docstats_loaded_this_query',
                'bm25_liveness_loaded_this_query', st->'bm25_liveness_loaded_this_query',
                'bm25_docstats_bytes', st->'bm25_docstats_bytes',
                'bm25_liveness_bytes', st->'bm25_liveness_bytes',
                'bm25_cold_cache_o_n_work', st->'bm25_cold_cache_o_n_work',
                'bm25_cache_hit', st->'bm25_cache_hit',
                'bm25_cache_build_us', st->'bm25_cache_build_us',
                'bm25_postings_decoded', st->'bm25_postings_decoded',
                'bm25_postings_decode_ratio', st->'bm25_postings_decode_ratio',
                'bm25_common_term_fallback', st->'bm25_common_term_fallback',
                'bm25_wand_pruned', st->'bm25_wand_pruned',
                'bm25_elapsed_us', st->'bm25_elapsed_us',
                'bm25', st->'bm25'
            ));

    t0 := clock_timestamp();
    SELECT count(*) INTO n
    FROM (
        SELECT id
        FROM th_bm25_cold_warm_docs
        ORDER BY embedding <~> turbohybrid_query(
            text_query => websearch_to_tsquery('simple', 'rareneedle'),
            dense_k => 0,
            bm25_k => rare_bm25_k,
            final_k => final_k
        )
        LIMIT final_k
    ) s;
    st := turbohybrid_last_scan_stats();
    INSERT INTO th_bm25_cold_warm_results
    VALUES ('rare_text_warm',
            extract(epoch FROM clock_timestamp() - t0) * 1000.0,
            n,
            jsonb_build_object(
                'index_shape', st->'index_shape',
                'dense_branch_used', st->'dense_branch_used',
                'bm25_branch_used', st->'bm25_branch_used',
                'bm25_docstats_loaded_this_query', st->'bm25_docstats_loaded_this_query',
                'bm25_liveness_loaded_this_query', st->'bm25_liveness_loaded_this_query',
                'bm25_docstats_bytes', st->'bm25_docstats_bytes',
                'bm25_liveness_bytes', st->'bm25_liveness_bytes',
                'bm25_cold_cache_o_n_work', st->'bm25_cold_cache_o_n_work',
                'bm25_cache_hit', st->'bm25_cache_hit',
                'bm25_cache_build_us', st->'bm25_cache_build_us',
                'bm25_postings_decoded', st->'bm25_postings_decoded',
                'bm25_postings_decode_ratio', st->'bm25_postings_decode_ratio',
                'bm25_common_term_fallback', st->'bm25_common_term_fallback',
                'bm25_wand_pruned', st->'bm25_wand_pruned',
                'bm25_elapsed_us', st->'bm25_elapsed_us',
                'bm25', st->'bm25'
            ));

    t0 := clock_timestamp();
    SELECT count(*) INTO n
    FROM (
        SELECT id
        FROM th_bm25_cold_warm_docs
        ORDER BY embedding <~> turbohybrid_query(
            vector_query => qv,
            text_query => websearch_to_tsquery('simple', 'common'),
            dense_k => hybrid_k,
            bm25_k => hybrid_k,
            final_k => final_k
        )
        LIMIT final_k
    ) s;
    st := turbohybrid_last_scan_stats();
    INSERT INTO th_bm25_cold_warm_results
    VALUES ('hybrid_large_candidates',
            extract(epoch FROM clock_timestamp() - t0) * 1000.0,
            n,
            jsonb_build_object(
                'dense_branch_used', st->'dense_branch_used',
                'bm25_branch_used', st->'bm25_branch_used',
                'dense_k_effective', st->'dense_k_effective',
                'bm25_k_effective', st->'bm25_k_effective',
                'final_k_effective', st->'final_k_effective',
                'fusion_strategy', st->'fusion_strategy',
                'fusion_candidates_seen', st->'fusion_candidates_seen',
                'fusion_duplicates', st->'fusion_duplicates',
                'fusion_elapsed_us', st->'fusion_elapsed_us',
                'fusion_generation_array_reused', st->'fusion_generation_array_reused',
                'fusion_generation_array_reset', st->'fusion_generation_array_reset',
                'bm25_postings_decoded', st->'bm25_postings_decoded',
                'bm25_common_term_fallback', st->'bm25_common_term_fallback',
                'bm25_wand_pruned', st->'bm25_wand_pruned',
                'graph_scored_codes', st->'graph_scored_codes',
                'dense_elapsed_us', st->'dense_elapsed_us',
                'bm25_elapsed_us', st->'bm25_elapsed_us',
                'fusion', st->'fusion'
            ));
END
$$;

SELECT scenario,
       round(elapsed_ms::numeric, 3) AS elapsed_ms,
       rows_returned,
       jsonb_pretty(stats) AS scan_stats
FROM th_bm25_cold_warm_results
ORDER BY scenario;

RESET turbohybrid.bm25_common_term_fallback_min_postings;
RESET turbohybrid.bm25_strategy;
RESET turbohybrid.enable_wand;
RESET enable_seqscan;
RESET max_parallel_workers_per_gather;
RESET jit;
