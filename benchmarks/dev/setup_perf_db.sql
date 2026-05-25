\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;

CREATE TABLE IF NOT EXISTS perf_matrix_runs (
    run_id bigserial PRIMARY KEY,
    started_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    label text NOT NULL,
    dataset text NOT NULL,
    dense_k int NOT NULL,
    bm25_k int NOT NULL,
    rrf_k int NOT NULL,
    final_k int NOT NULL,
    notes text
);

CREATE TABLE IF NOT EXISTS perf_matrix_query_cases (
    category text PRIMARY KEY,
    query_id text,
    vector_query text,
    text_query text,
    description text NOT NULL
);

CREATE TABLE IF NOT EXISTS perf_matrix_query_results (
    run_id bigint NOT NULL REFERENCES perf_matrix_runs(run_id),
    category text NOT NULL,
    query_sql text NOT NULL,
    plan jsonb NOT NULL,
    forced_plan jsonb,
    planner_validation jsonb,
    last_scan_stats jsonb,
    debug_scan_stats jsonb,
    index_stats jsonb,
    simd_capabilities jsonb,
    elapsed_ms double precision NOT NULL,
    forced_elapsed_ms double precision,
    cost_model_regression bool NOT NULL DEFAULT false,
    used_turbohybrid_index bool NOT NULL,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);

ALTER TABLE perf_matrix_query_results
    ADD COLUMN IF NOT EXISTS debug_scan_stats jsonb,
    ADD COLUMN IF NOT EXISTS forced_plan jsonb,
    ADD COLUMN IF NOT EXISTS planner_validation jsonb,
    ADD COLUMN IF NOT EXISTS forced_elapsed_ms double precision,
    ADD COLUMN IF NOT EXISTS cost_model_regression bool NOT NULL DEFAULT false;

TRUNCATE perf_matrix_query_cases;

INSERT INTO perf_matrix_query_cases(category, query_id, vector_query, text_query, description)
SELECT 'dense_only', query_id, embedding::text, NULL,
       'Dense-only pgturbohybrid query using a real FIQA query vector'
FROM fiqa_queries
ORDER BY query_id
LIMIT 1;

CREATE TEMP TABLE perf_matrix_term_cases AS
WITH doc_count AS (
    SELECT count(*)::float8 AS n FROM fiqa_docs
),
term_stats AS MATERIALIZED (
    SELECT word, ndoc
    FROM ts_stat('SELECT body_tsv FROM fiqa_docs')
    WHERE word ~ '^[a-z0-9][a-z0-9_-]*$'
      AND length(word) >= 3
),
qrel_term_stats AS MATERIALIZED (
    SELECT word, ndoc
    FROM ts_stat(
        'SELECT d.body_tsv FROM fiqa_docs d ' ||
        'JOIN fiqa_qrels r ON r.doc_id = d.doc_id ' ||
        'WHERE r.relevance > 0'
    )
    WHERE word ~ '^[a-z0-9][a-z0-9_-]*$'
      AND length(word) >= 3
),
and_doc AS (
    SELECT d.doc_id
    FROM fiqa_docs d
    JOIN fiqa_qrels r ON r.doc_id = d.doc_id
    WHERE r.relevance > 0
    GROUP BY d.doc_id
    ORDER BY count(*) DESC, d.doc_id
    LIMIT 1
)
SELECT 'rare' AS case_name, word AS tsquery_expr
FROM qrel_term_stats
ORDER BY ndoc ASC, word
LIMIT 1;

INSERT INTO perf_matrix_term_cases(case_name, tsquery_expr)
SELECT 'common', word
FROM (
    SELECT word
    FROM ts_stat('SELECT body_tsv FROM fiqa_docs'), (SELECT count(*)::float8 AS n FROM fiqa_docs) doc_count
    WHERE word ~ '^[a-z0-9][a-z0-9_-]*$'
      AND length(word) >= 3
      AND ndoc < doc_count.n
    ORDER BY ndoc DESC, word
    LIMIT 1
) s;

INSERT INTO perf_matrix_term_cases(case_name, tsquery_expr)
SELECT 'or', string_agg(word, ' | ' ORDER BY ndoc DESC, word)
FROM (
    SELECT word, ndoc
    FROM ts_stat('SELECT body_tsv FROM fiqa_docs'), (SELECT count(*)::float8 AS n FROM fiqa_docs) doc_count
    WHERE word ~ '^[a-z0-9][a-z0-9_-]*$'
      AND length(word) >= 3
      AND ndoc < doc_count.n
    ORDER BY ndoc DESC, word
    LIMIT 2
) s;

INSERT INTO perf_matrix_term_cases(case_name, tsquery_expr)
SELECT 'and', string_agg(word, ' & ' ORDER BY word)
FROM (
    SELECT s.word
    FROM (
        SELECT d.doc_id
        FROM fiqa_docs d
        JOIN fiqa_qrels r ON r.doc_id = d.doc_id
        WHERE r.relevance > 0
        GROUP BY d.doc_id
        ORDER BY count(*) DESC, d.doc_id
        LIMIT 1
    ) d
    CROSS JOIN LATERAL ts_stat(
        format('SELECT body_tsv FROM fiqa_docs WHERE doc_id = %L', d.doc_id)
    ) AS s
    WHERE s.word ~ '^[a-z0-9][a-z0-9_-]*$'
      AND length(s.word) >= 3
    ORDER BY s.nentry DESC, s.word
    LIMIT 2
) s;

DO $$
DECLARE
    missing text[];
BEGIN
    SELECT array_agg(expected)
    INTO missing
    FROM unnest(ARRAY['rare', 'common', 'or', 'and']) AS expected
    WHERE NOT EXISTS (
        SELECT 1
        FROM perf_matrix_term_cases c
        WHERE c.case_name = expected
          AND c.tsquery_expr IS NOT NULL
          AND c.tsquery_expr <> ''
    );

    IF missing IS NOT NULL THEN
        RAISE EXCEPTION 'could not derive real FIQA BM25 query cases: %', missing;
    END IF;
END $$;

INSERT INTO perf_matrix_query_cases(category, query_id, vector_query, text_query, description)
SELECT 'bm25_rare', NULL, NULL, tsquery_expr,
       'BM25-only rare term derived from real FIQA qrels/corpus statistics'
FROM perf_matrix_term_cases
WHERE case_name = 'rare';

INSERT INTO perf_matrix_query_cases(category, query_id, vector_query, text_query, description)
SELECT 'bm25_common', NULL, NULL, tsquery_expr,
       'BM25-only common term derived from real FIQA corpus statistics'
FROM perf_matrix_term_cases
WHERE case_name = 'common';

INSERT INTO perf_matrix_query_cases(category, query_id, vector_query, text_query, description)
SELECT 'bm25_or', NULL, NULL, tsquery_expr,
       'BM25-only OR query derived from real FIQA corpus statistics'
FROM perf_matrix_term_cases
WHERE case_name = 'or';

INSERT INTO perf_matrix_query_cases(category, query_id, vector_query, text_query, description)
SELECT 'bm25_and', NULL, NULL, tsquery_expr,
       'BM25-only AND query derived from real FIQA qrels/corpus statistics'
FROM perf_matrix_term_cases
WHERE case_name = 'and';

INSERT INTO perf_matrix_query_cases(category, query_id, vector_query, text_query, description)
SELECT 'hybrid_rare', q.query_id, q.embedding::text, c.tsquery_expr,
       'Hybrid query combining a real FIQA query vector with real rare term'
FROM fiqa_queries q
JOIN perf_matrix_term_cases c ON c.case_name = 'rare'
ORDER BY q.query_id
LIMIT 1;

INSERT INTO perf_matrix_query_cases(category, query_id, vector_query, text_query, description)
SELECT 'hybrid_common', q.query_id, q.embedding::text, c.tsquery_expr,
       'Hybrid query combining a real FIQA query vector with real common term'
FROM fiqa_queries q
JOIN perf_matrix_term_cases c ON c.case_name = 'common'
ORDER BY q.query_id
LIMIT 1;

INSERT INTO perf_matrix_query_cases(category, query_id, vector_query, text_query, description)
SELECT 'hybrid_or', q.query_id, q.embedding::text, c.tsquery_expr,
       'Hybrid OR query combining a real FIQA query vector with real corpus terms'
FROM fiqa_queries q
JOIN perf_matrix_term_cases c ON c.case_name = 'or'
ORDER BY query_id
LIMIT 1;

INSERT INTO perf_matrix_query_cases(category, query_id, vector_query, text_query, description)
SELECT 'hybrid_and', q.query_id, q.embedding::text, c.tsquery_expr,
       'Hybrid AND query combining a real FIQA query vector with real qrel document terms'
FROM fiqa_queries q
JOIN perf_matrix_term_cases c ON c.case_name = 'and'
ORDER BY query_id
LIMIT 1;

INSERT INTO perf_matrix_query_cases(category, query_id, vector_query, text_query, description)
SELECT 'hybrid_no_lexical_match', q.query_id, q.embedding::text, q.query_text,
       'Hybrid query using a real FIQA query whose lexical branch has no matching loaded document'
FROM fiqa_queries q
WHERE NOT EXISTS (
    SELECT 1
    FROM fiqa_docs d
    WHERE d.body_tsv @@ plainto_tsquery('english', q.query_text)
)
ORDER BY q.query_id
LIMIT 1;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM perf_matrix_query_cases
        WHERE category = 'hybrid_no_lexical_match'
    ) THEN
        RAISE EXCEPTION 'could not derive hybrid_no_lexical_match from real FIQA queries and corpus';
    END IF;
END $$;
