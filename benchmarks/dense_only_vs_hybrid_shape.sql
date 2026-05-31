-- Compare true dense-only turbohybrid indexes with the previous fake-hybrid
-- pattern that carried a populated tsvector key even for vector-only queries.
--
-- Run:
--   psql -d <db> -f benchmarks/dense_only_vs_hybrid_shape.sql
--
-- Optional psql variables:
--   NROWS       synthetic document count      (default 10000)
--   DIMS        embedding dimensions          (default 384)
--   BITS        quantization bits             (default 4)
--   WARM        warmup queries per shape      (default 30)
--   TIMED       timed queries per shape       (default 200)
--   FINAL_K     query LIMIT / final_k         (default 10)
--   DENSE_K     dense candidate budget        (default 100)
--   MIN_OVERLAP minimum overlap@FINAL_K       (default 8)
--
-- The benchmark builds two equivalent corpora:
--   A. dense-only: one vector key, no BM25 metadata
--   B. fake hybrid: vector key plus populated body_tsv, matching the old tests
--
-- It reports build time, index size, warm dense-only query p50/p95, last-scan
-- stats, BM25 branch availability/usage, and overlap between result sets.  Build
-- timing is most meaningful at larger NROWS; tiny smoke runs are dominated by
-- fixed PostgreSQL/index initialization cost and cache order effects.

\set ON_ERROR_STOP on
\pset pager off

\if :{?NROWS}
\else
  \set NROWS 10000
\endif
\if :{?DIMS}
\else
  \set DIMS 384
\endif
\if :{?BITS}
\else
  \set BITS 4
\endif
\if :{?WARM}
\else
  \set WARM 30
\endif
\if :{?TIMED}
\else
  \set TIMED 200
\endif
\if :{?FINAL_K}
\else
  \set FINAL_K 10
\endif
\if :{?DENSE_K}
\else
  \set DENSE_K 100
\endif
\if :{?MIN_OVERLAP}
\else
  \set MIN_OVERLAP 8
\endif

CREATE EXTENSION IF NOT EXISTS pgturbohybrid CASCADE;

SET enable_seqscan = off;
SET max_parallel_workers_per_gather = 0;
SET jit = off;

\echo '== dense-only vs fake-hybrid index-shape benchmark =='
\echo 'rows=' :NROWS ' dims=' :DIMS ' bits=' :BITS ' warm=' :WARM ' timed=' :TIMED ' final_k=' :FINAL_K ' dense_k=' :DENSE_K

DROP TABLE IF EXISTS th_shape_dense_docs CASCADE;
DROP TABLE IF EXISTS th_shape_fake_hybrid_docs CASCADE;

CREATE TABLE th_shape_dense_docs (
    id int PRIMARY KEY,
    embedding vector(:DIMS) NOT NULL
);

CREATE TABLE th_shape_fake_hybrid_docs (
    id int PRIMARY KEY,
    embedding vector(:DIMS) NOT NULL,
    body_tsv tsvector NOT NULL
);

\echo '== loading equivalent embeddings =='
INSERT INTO th_shape_dense_docs(id, embedding)
SELECT g,
       (SELECT array_agg(
                   (sin(g * 0.011 + d * 0.017) +
                    0.05 * cos(g * 0.037 + d * 0.013))::real
                   ORDER BY d)
        FROM generate_series(1, :DIMS) AS d)::real[]::vector(:DIMS)
FROM generate_series(1, :NROWS) AS g;

INSERT INTO th_shape_fake_hybrid_docs(id, embedding, body_tsv)
SELECT id,
       embedding,
       to_tsvector('english',
                   'document number ' || id || ' dense benchmark common token')
FROM th_shape_dense_docs;

ANALYZE th_shape_dense_docs;
ANALYZE th_shape_fake_hybrid_docs;

CREATE TEMP TABLE th_shape_build_result (
    label text PRIMARY KEY,
    build_ms float8 NOT NULL
);

CREATE OR REPLACE FUNCTION pg_temp.th_shape_create_index(p_label text,
                                                         p_sql text)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    started timestamptz;
BEGIN
    started := clock_timestamp();
    EXECUTE p_sql;
    INSERT INTO th_shape_build_result(label, build_ms)
    VALUES (p_label, extract(epoch FROM clock_timestamp() - started) * 1000.0);
END;
$$;

\echo '== building indexes =='
SELECT pg_temp.th_shape_create_index(
    'dense_only',
    'CREATE INDEX th_shape_dense_idx ON th_shape_dense_docs ' ||
    'USING turbohybrid (embedding vector_cosine_turbohybrid_ops) ' ||
    'WITH (quantization_bits = ' || :BITS || ')'
);

SELECT pg_temp.th_shape_create_index(
    'fake_hybrid',
    'CREATE INDEX th_shape_fake_hybrid_idx ON th_shape_fake_hybrid_docs ' ||
    'USING turbohybrid (embedding vector_cosine_turbohybrid_ops, ' ||
    'body_tsv bm25_tsvector_turbohybrid_ops) ' ||
    'WITH (quantization_bits = ' || :BITS || ')'
);

ANALYZE th_shape_dense_docs;
ANALYZE th_shape_fake_hybrid_docs;

CREATE TEMP TABLE th_shape_timings (
    label text NOT NULL,
    seq int NOT NULL,
    elapsed_ms float8 NOT NULL,
    ids int[] NOT NULL,
    last_scan_stats jsonb NOT NULL,
    PRIMARY KEY (label, seq)
);

CREATE OR REPLACE FUNCTION pg_temp.th_shape_pctl(arr float8[], q float8)
RETURNS float8
LANGUAGE sql
AS $$
    SELECT percentile_cont(q) WITHIN GROUP (ORDER BY x)
    FROM unnest(arr) AS x
$$;

CREATE OR REPLACE FUNCTION pg_temp.th_shape_overlap(a int[], b int[])
RETURNS int
LANGUAGE sql
AS $$
    SELECT count(*)::int
    FROM (
        SELECT unnest(a)
        INTERSECT
        SELECT unnest(b)
    ) s
$$;

CREATE OR REPLACE FUNCTION pg_temp.th_shape_run_dense_queries(
    p_label text,
    p_table regclass,
    p_nrows int,
    p_warm int,
    p_timed int,
    p_dense_k int,
    p_final_k int)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    qv vector;
    ids int[];
    started timestamptz;
    k int;
    doc_id int;
    elapsed float8;
    query_sql text;
BEGIN
    query_sql := format(
        'SELECT array_agg(id) FROM (' ||
        'SELECT id FROM %s ' ||
        'ORDER BY embedding <~> turbohybrid_query(' ||
        'vector_query => $1, dense_k => $2, final_k => $3) ' ||
        'LIMIT $3' ||
        ') s',
        p_table);

    FOR k IN 1..p_warm LOOP
        doc_id := (k * 131) % p_nrows + 1;
        EXECUTE format('SELECT embedding FROM %s WHERE id = $1', p_table)
            INTO qv
            USING doc_id;
        EXECUTE query_sql INTO ids USING qv, p_dense_k, p_final_k;
    END LOOP;

    FOR k IN 1..p_timed LOOP
        doc_id := (k * 131) % p_nrows + 1;
        EXECUTE format('SELECT embedding FROM %s WHERE id = $1', p_table)
            INTO qv
            USING doc_id;

        started := clock_timestamp();
        EXECUTE query_sql INTO ids USING qv, p_dense_k, p_final_k;
        elapsed := extract(epoch FROM clock_timestamp() - started) * 1000.0;

        INSERT INTO th_shape_timings(label, seq, elapsed_ms, ids, last_scan_stats)
        VALUES (p_label, k, elapsed, ids, turbohybrid_last_scan_stats());
    END LOOP;
END;
$$;

\echo '== timing warm vector-only queries =='
SELECT pg_temp.th_shape_run_dense_queries(
    'dense_only',
    'th_shape_dense_docs'::regclass,
    :NROWS,
    :WARM,
    :TIMED,
    :DENSE_K,
    :FINAL_K
);

SELECT pg_temp.th_shape_run_dense_queries(
    'fake_hybrid',
    'th_shape_fake_hybrid_docs'::regclass,
    :NROWS,
    :WARM,
    :TIMED,
    :DENSE_K,
    :FINAL_K
);

\echo '== build time and index size =='
WITH index_sizes AS (
    SELECT 'dense_only'::text AS label,
           pg_relation_size('th_shape_dense_idx'::regclass) AS index_bytes,
           turbohybrid_index_stats('th_shape_dense_idx'::regclass) AS index_stats
    UNION ALL
    SELECT 'fake_hybrid',
           pg_relation_size('th_shape_fake_hybrid_idx'::regclass),
           turbohybrid_index_stats('th_shape_fake_hybrid_idx'::regclass)
),
joined AS (
    SELECT s.label,
           b.build_ms,
           s.index_bytes,
           s.index_stats
    FROM index_sizes s
    JOIN th_shape_build_result b USING (label)
)
SELECT label,
       round(build_ms::numeric, 3) AS build_ms,
       pg_size_pretty(index_bytes) AS index_size,
       index_bytes,
       index_stats->>'index_shape' AS index_shape,
       (index_stats->>'bm25_branch_available')::boolean AS bm25_branch_available,
       (index_stats->>'bm25_document_count')::int AS bm25_document_count
FROM joined
ORDER BY label;

\echo '== relative build and size =='
WITH metrics AS (
    SELECT b.label,
           b.build_ms,
           CASE b.label
               WHEN 'dense_only' THEN pg_relation_size('th_shape_dense_idx'::regclass)
               ELSE pg_relation_size('th_shape_fake_hybrid_idx'::regclass)
           END AS index_bytes
    FROM th_shape_build_result b
),
pivot AS (
    SELECT max(build_ms) FILTER (WHERE label = 'dense_only') AS dense_build_ms,
           max(build_ms) FILTER (WHERE label = 'fake_hybrid') AS hybrid_build_ms,
           max(index_bytes) FILTER (WHERE label = 'dense_only') AS dense_bytes,
           max(index_bytes) FILTER (WHERE label = 'fake_hybrid') AS hybrid_bytes
    FROM metrics
)
SELECT dense_bytes < hybrid_bytes AS dense_index_smaller,
       round((hybrid_bytes::numeric / NULLIF(dense_bytes, 0))::numeric, 3) AS hybrid_size_x_dense,
       round(dense_build_ms::numeric, 3) AS dense_build_ms,
       round(hybrid_build_ms::numeric, 3) AS fake_hybrid_build_ms,
       round((hybrid_build_ms / NULLIF(dense_build_ms, 0))::numeric, 3) AS hybrid_build_time_x_dense
FROM pivot;

\echo '== vector-only query latency =='
SELECT label,
       count(*) AS measured_queries,
       round(pg_temp.th_shape_pctl(array_agg(elapsed_ms), 0.50)::numeric, 4) AS p50_ms,
       round(pg_temp.th_shape_pctl(array_agg(elapsed_ms), 0.95)::numeric, 4) AS p95_ms,
       round(avg(elapsed_ms)::numeric, 4) AS mean_ms
FROM th_shape_timings
GROUP BY label
ORDER BY label;

\echo '== last scan stats from the final timed query for each shape =='
WITH last_stats AS (
    SELECT DISTINCT ON (label)
           label,
           last_scan_stats
    FROM th_shape_timings
    ORDER BY label, seq DESC
)
SELECT label,
       last_scan_stats->>'index_shape' AS index_shape,
       (last_scan_stats->>'bm25_branch_available')::boolean AS bm25_branch_available,
       (last_scan_stats->>'dense_branch_used')::boolean AS dense_branch_used,
       (last_scan_stats->>'bm25_branch_used')::boolean AS bm25_branch_used,
       last_scan_stats->>'scan_orchestration' AS scan_orchestration,
       (last_scan_stats->>'dense_elapsed_us')::bigint AS dense_elapsed_us,
       (last_scan_stats->>'bm25_elapsed_us')::bigint AS bm25_elapsed_us
FROM last_stats
ORDER BY label;

\echo '== dense result overlap between shapes =='
WITH paired AS (
    SELECT d.seq,
           pg_temp.th_shape_overlap(d.ids, h.ids) AS overlap_at_k
    FROM th_shape_timings d
    JOIN th_shape_timings h USING (seq)
    WHERE d.label = 'dense_only'
      AND h.label = 'fake_hybrid'
)
SELECT count(*) AS compared_queries,
       min(overlap_at_k) AS min_overlap_at_k,
       round(avg(overlap_at_k)::numeric, 3) AS avg_overlap_at_k,
       :FINAL_K AS final_k,
       min(overlap_at_k) >= :MIN_OVERLAP AS passes_overlap_tolerance
FROM paired;

\echo '== interpretation =='
\echo 'dense_only should report index_shape=dense_only, bm25_branch_available=false, bm25_branch_used=false.'
\echo 'fake_hybrid should report index_shape=hybrid, bm25_branch_available=true, bm25_branch_used=false for vector-only queries.'
\echo 'A smaller dense_only index shows the storage cost of fake BM25 metadata.'
\echo 'On nontrivial row counts, build_ms shows whether skipping BM25 collection improves build time on this host.'

DROP FUNCTION pg_temp.th_shape_create_index(text, text);
DROP FUNCTION pg_temp.th_shape_pctl(float8[], float8);
DROP FUNCTION pg_temp.th_shape_overlap(int[], int[]);
DROP FUNCTION pg_temp.th_shape_run_dense_queries(text, regclass, int, int, int, int, int);

DROP TABLE th_shape_dense_docs;
DROP TABLE th_shape_fake_hybrid_docs;

RESET enable_seqscan;
RESET max_parallel_workers_per_gather;
RESET jit;
