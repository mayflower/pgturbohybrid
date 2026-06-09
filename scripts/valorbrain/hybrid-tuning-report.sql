\set ON_ERROR_STOP on

-- Snapshot hybrid GUC defaults + one canonical query's scan stats for tuning Kd/Kb/rrfK.
-- Does not change settings; use output to adjust reloptions or session GUCs.

\echo '=== turbohybrid hybrid GUCs (session + defaults) ==='
SELECT name,
       setting,
       unit,
       short_desc
FROM pg_settings
WHERE name IN (
    'turbohybrid.profile',
    'turbohybrid.default_dense_k',
    'turbohybrid.default_bm25_k',
    'turbohybrid.default_rrf_k',
    'turbohybrid.bm25_hybrid_bound',
    'turbohybrid.hybrid_budget_policy',
    'turbohybrid.native_cache_scope',
    'turbohybrid.native_cache_disk_max_mb'
)
ORDER BY name;

\echo '=== index stats ==='
SELECT turbohybrid_index_stats('idx_documents_fts_hybrid'::regclass)
WHERE to_regclass('public.idx_documents_fts_hybrid') IS NOT NULL;

\echo '=== sample hybrid query (warmup) ==='
DO $$
DECLARE
    sample_embedding text;
    row_count int;
BEGIN
    IF to_regclass('public.idx_documents_fts_hybrid') IS NULL THEN
        RAISE EXCEPTION 'idx_documents_fts_hybrid not found';
    END IF;

    SELECT embedding::text INTO sample_embedding
    FROM documents_fts
    WHERE embedding IS NOT NULL
    ORDER BY doc_id
    LIMIT 1;

    IF sample_embedding IS NULL THEN
        RAISE EXCEPTION 'no hybrid embeddings in documents_fts';
    END IF;

    SELECT count(*) INTO row_count
    FROM (
        SELECT d.id
        FROM documents_fts fts
        JOIN documents d ON d.id = fts.doc_id
        WHERE fts.embedding IS NOT NULL
          AND d.active = 1
        ORDER BY fts.embedding <~> turbohybrid_query(
            vector_query => sample_embedding::vector,
            text_query => websearch_to_tsquery('english', 'postgres hybrid search')
        )
        LIMIT 10
    ) q;

    IF row_count <> 10 THEN
        RAISE EXCEPTION 'warmup returned % rows, expected 10', row_count;
    END IF;
END;
$$;

\echo '=== last_scan_stats after warmup ==='
SELECT jsonb_pretty(turbohybrid_last_scan_stats());

\echo '=== cache + table health ==='
SELECT relname,
       n_live_tup,
       n_dead_tup,
       last_autovacuum,
       last_autoanalyze
FROM pg_stat_user_tables
WHERE relname IN ('documents_fts', 'documents', 'content_vectors')
ORDER BY relname;