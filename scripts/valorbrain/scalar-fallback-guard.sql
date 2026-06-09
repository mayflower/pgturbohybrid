\set ON_ERROR_STOP on

-- ValorBrain production guard: hybrid /search must use idx_documents_fts_hybrid.
-- Fails fast when the planner falls back to seq scan + sort (10-100x slower).
--
-- Usage:
--   psql -h 127.0.0.1 -p 5433 -U postgres -d valorbrain -f scripts/valorbrain/scalar-fallback-guard.sql
-- Or via scripts/valorbrain/scalar-fallback-guard.sh

\ir plan-guard-lib.sql

DO $$
DECLARE
    hybrid_idx regclass;
    sample_embedding text;
    plan_json jsonb;
    guard_result jsonb;
    hybrid_rows bigint;
BEGIN
    SELECT to_regclass('public.idx_documents_fts_hybrid') INTO hybrid_idx;
    IF hybrid_idx IS NULL THEN
        RAISE EXCEPTION 'idx_documents_fts_hybrid is missing; hybrid search cannot use turbohybrid';
    END IF;

    SELECT count(*) INTO hybrid_rows
    FROM documents_fts
    WHERE embedding IS NOT NULL;

    IF hybrid_rows < 1 THEN
        RAISE EXCEPTION 'documents_fts has no hybrid embeddings; guard needs at least one row with embedding IS NOT NULL';
    END IF;

    SELECT embedding::text INTO sample_embedding
    FROM documents_fts
    WHERE embedding IS NOT NULL
    ORDER BY doc_id
    LIMIT 1;

    EXECUTE $plan$
        EXPLAIN (FORMAT JSON)
        SELECT d.id
        FROM documents_fts fts
        JOIN documents d ON d.id = fts.doc_id
        WHERE fts.embedding IS NOT NULL
          AND d.active = 1
          AND (d.invalidated_at IS NULL OR d.invalidated_at = '')
        ORDER BY fts.embedding <~> turbohybrid_query(
            vector_query => $1::vector,
            text_query => websearch_to_tsquery('english', $2)
        )
        LIMIT 10
    $plan$
    USING sample_embedding, 'postgres hybrid search'
    INTO plan_json;

    guard_result := pg_temp.turbohybrid_assert_index_plan(
        plan_json,
        hybrid_idx,
        'valorbrain hybrid /search canonical');

    RAISE NOTICE 'scalar fallback guard passed: %', guard_result;
END;
$$;

-- Dense-only shape used by content_vectors (vectorSearch still aggregates; this
-- documents the index-backed ORDER BY pattern operators should prefer).
DO $$
DECLARE
    dense_idx regclass;
    sample_embedding text;
    plan_json jsonb;
    guard_result jsonb;
BEGIN
    SELECT to_regclass('public.idx_content_vectors_turbo') INTO dense_idx;
    IF dense_idx IS NULL THEN
        RAISE NOTICE 'idx_content_vectors_turbo missing; skipping dense guard';
        RETURN;
    END IF;

    SELECT embedding::text INTO sample_embedding
    FROM content_vectors
    WHERE embedding IS NOT NULL
    ORDER BY hash, seq
    LIMIT 1;

    IF sample_embedding IS NULL THEN
        RAISE NOTICE 'content_vectors has no embeddings; skipping dense guard';
        RETURN;
    END IF;

    EXECUTE $plan$
        EXPLAIN (FORMAT JSON)
        SELECT cv.hash
        FROM content_vectors cv
        WHERE cv.embedding IS NOT NULL
        ORDER BY cv.embedding <~> turbohybrid_query(vector_query => $1::vector)
        LIMIT 10
    $plan$
    USING sample_embedding
    INTO plan_json;

    guard_result := pg_temp.turbohybrid_assert_index_plan(
        plan_json,
        dense_idx,
        'valorbrain dense vector index-backed ORDER BY');

    RAISE NOTICE 'dense index guard passed: %', guard_result;
END;
$$;