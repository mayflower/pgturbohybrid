-- multivector_late_interaction.sql
--
-- Deterministic developer benchmark for ColBERT/Late-Interaction scaling
-- slopes. This is NOT a publishable benchmark result: it prints local timing
-- and stats only, writes no output files, and uses synthetic data.
--
-- Run:
--   psql -d <db> -f benchmarks/dev/multivector_late_interaction.sql
--
-- Optional psql variables:
--   DOCS       document count D                       (default 256)
--   DIMS       vector dimensions d                    (default 32)
--   FINAL_K    final result cutoff                    (default 10)
--   ITERS      timed iterations per cell              (default 3)
--
-- The slope grids make O-complexity visible:
--   * D constant, doc token vectors L = 8, 32, 128
--   * L constant, query token vectors Q = 4, 16, 64
--   * d constant unless the dimensions grid varies d = 32, 96, 128
--   * exact rerank R = off, 25, 100
--   * subvector_k Ks = 16, 64, 256
--
-- Expected slopes:
--   * Build stores one graph node per subvector: ~ O(D * L)
--   * Approx query work scales with Q graph searches plus Q * Ks raw hits
--   * Exact heap rerank work scales with R * Q * avg_L * d

\set ON_ERROR_STOP on
\pset pager off
\x off

\if :{?DOCS}
\else
  \set DOCS 256
\endif
\if :{?DIMS}
\else
  \set DIMS 32
\endif
\if :{?FINAL_K}
\else
  \set FINAL_K 10
\endif
\if :{?ITERS}
\else
  \set ITERS 3
\endif

CREATE EXTENSION IF NOT EXISTS pgturbohybrid CASCADE;

SET jit = off;
SET enable_seqscan = off;
SET max_parallel_workers_per_gather = 0;

\echo '== pgturbohybrid multivector late-interaction slope grid =='
\echo 'docs=' :DOCS ' dims=' :DIMS ' final_k=' :FINAL_K ' iters=' :ITERS

DROP TABLE IF EXISTS mli_report;
DROP TABLE IF EXISTS mli_docs CASCADE;
DROP TABLE IF EXISTS mli_params;

CREATE TEMP TABLE mli_params AS
SELECT :DOCS::int AS docs,
       :DIMS::int AS dims,
       :FINAL_K::int AS final_k,
       :ITERS::int AS iters;

CREATE TEMP TABLE mli_report (
    experiment text NOT NULL,
    variant text NOT NULL,
    doc_count int NOT NULL,
    dimensions int NOT NULL,
    doc_vectors int NOT NULL,
    query_vectors int NOT NULL,
    subvector_k int NOT NULL,
    exact_rerank text NOT NULL,
    exact_rerank_k int NOT NULL,
    build_ms float8 NOT NULL,
    index_bytes bigint NOT NULL,
    node_count int,
    p50_ms float8 NOT NULL,
    p95_ms float8 NOT NULL,
    recall_at_k float8 NOT NULL,
    raw_subvector_hits int,
    unique_docs int,
    doc_candidates int,
    exact_rerank_pairs bigint,
    accumulator_memory_bytes bigint,
    fusion_strategy text,
    notes text
);

CREATE OR REPLACE FUNCTION pg_temp.mli_vector(seed_doc int, token_ordinal int, dims int)
RETURNS vector
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT array_agg(
               (
                   sin(seed_doc * 0.031 + token_ordinal * 0.173 + d * 0.119) +
                   cos((seed_doc % 17 + 1) * 0.071 + d * 0.047) +
                   0.07 * sin((seed_doc + token_ordinal) * 0.013 + d * 0.193)
               )::real
               ORDER BY d
           )::real[]::vector
    FROM generate_series(1, dims) AS d;
$$;

CREATE OR REPLACE FUNCTION pg_temp.mli_multivector(seed_doc int, token_count int, dims int)
RETURNS turbohybrid_multivector
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT turbohybrid_multivector(array_agg(pg_temp.mli_vector(seed_doc, t, dims) ORDER BY t))
    FROM generate_series(1, token_count) AS t;
$$;

CREATE OR REPLACE FUNCTION pg_temp.mli_overlap(a int[], b int[])
RETURNS int
LANGUAGE sql
IMMUTABLE
AS $$
    SELECT count(*)::int
    FROM (SELECT unnest(a) INTERSECT SELECT unnest(b)) s;
$$;

CREATE OR REPLACE FUNCTION pg_temp.mli_run_case(
    p_experiment text,
    p_variant text,
    p_docs int,
    p_doc_vectors int,
    p_query_vectors int,
    p_subvector_k int,
    p_exact_rerank_k int,
    p_dims int DEFAULT NULL,
    p_notes text DEFAULT NULL
)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    dims int;
    final_k int;
    iters int;
    build_start timestamptz;
    query_start timestamptz;
    build_ms float8;
    latencies float8[] := '{}';
    q turbohybrid_multivector;
    ids int[];
    exact_ids int[];
    stats jsonb;
    index_stats jsonb;
    sql text;
    i int;
    doc_candidate_k int;
    unique_docs_per_token int;
    raw_hits_per_token int;
BEGIN
    SELECT COALESCE(p_dims, m.dims), m.final_k, m.iters
    INTO dims, final_k, iters
    FROM mli_params m;

    doc_candidate_k := greatest(final_k, p_exact_rerank_k, least(p_docs, p_subvector_k * p_query_vectors));
    unique_docs_per_token := greatest(final_k, least(p_docs, p_subvector_k));
    raw_hits_per_token := greatest(final_k, p_subvector_k);

    PERFORM set_config('turbohybrid.multivector_max_doc_vectors', greatest(p_doc_vectors, 1)::text, false);
    PERFORM set_config('turbohybrid.multivector_max_query_vectors', greatest(p_query_vectors, 1)::text, false);
    PERFORM set_config('turbohybrid.multivector_max_dim', dims::text, false);
    PERFORM set_config('turbohybrid.multivector_subvector_k', p_subvector_k::text, false);
    PERFORM set_config('turbohybrid.multivector_unique_docs_per_token', unique_docs_per_token::text, false);
    PERFORM set_config('turbohybrid.multivector_max_raw_hits_per_token', raw_hits_per_token::text, false);
    PERFORM set_config('turbohybrid.multivector_doc_candidate_k', doc_candidate_k::text, false);
    IF p_exact_rerank_k > 0 THEN
        PERFORM set_config('turbohybrid.multivector_exact_rerank', 'topk', false);
        PERFORM set_config('turbohybrid.multivector_exact_rerank_k', p_exact_rerank_k::text, false);
    ELSE
        PERFORM set_config('turbohybrid.multivector_exact_rerank', 'off', false);
        PERFORM set_config('turbohybrid.multivector_exact_rerank_k', final_k::text, false);
    END IF;

    DROP TABLE IF EXISTS mli_docs;
    CREATE TEMP TABLE mli_docs (
        id int PRIMARY KEY,
        colbert turbohybrid_multivector NOT NULL
    );

    INSERT INTO mli_docs(id, colbert)
    SELECT g, pg_temp.mli_multivector(g, p_doc_vectors, dims)
    FROM generate_series(1, p_docs) AS g;
    ANALYZE mli_docs;

    build_start := clock_timestamp();
    CREATE INDEX mli_docs_colbert_idx ON mli_docs USING turbohybrid
        (colbert multivector_cosine_turbohybrid_ops)
        WITH (quantization_bits = 4, exact_storage = off);
    build_ms := extract(epoch FROM clock_timestamp() - build_start) * 1000.0;
    index_stats := turbohybrid_index_stats('mli_docs_colbert_idx'::regclass);

    q := pg_temp.mli_multivector(1, p_query_vectors, dims);
    SELECT array_agg(e.id ORDER BY e.score DESC, e.id)
    INTO exact_ids
    FROM (
        SELECT d.id, turbohybrid_multivector_maxsim(q, d.colbert) AS score
        FROM mli_docs d
        ORDER BY score DESC, d.id
        LIMIT final_k
    ) e;

    sql := format(
        'SELECT array_agg(id) FROM (
             SELECT id
             FROM mli_docs
             ORDER BY colbert <~> turbohybrid_query(multivector_query => $1, dense_k => %s, final_k => %s)
             LIMIT %s
         ) s',
        doc_candidate_k,
        final_k,
        final_k);

    EXECUTE sql USING q INTO ids; -- warm
    FOR i IN 1..iters LOOP
        query_start := clock_timestamp();
        EXECUTE sql USING q INTO ids;
        latencies := latencies || (extract(epoch FROM clock_timestamp() - query_start) * 1000.0);
    END LOOP;
    stats := turbohybrid_last_scan_stats();

    INSERT INTO mli_report(
        experiment, variant, doc_count, dimensions, doc_vectors, query_vectors, subvector_k,
        exact_rerank, exact_rerank_k, build_ms, index_bytes, node_count,
        p50_ms, p95_ms, recall_at_k, raw_subvector_hits, unique_docs,
        doc_candidates, exact_rerank_pairs, accumulator_memory_bytes,
        fusion_strategy, notes)
    VALUES (
        p_experiment,
        p_variant,
        p_docs,
        dims,
        p_doc_vectors,
        p_query_vectors,
        p_subvector_k,
        CASE WHEN p_exact_rerank_k > 0 THEN 'topk' ELSE 'off' END,
        p_exact_rerank_k,
        build_ms,
        pg_relation_size('mli_docs_colbert_idx'),
        COALESCE(NULLIF(index_stats->>'node_count', '')::int, p_docs * p_doc_vectors),
        (SELECT percentile_cont(0.50) WITHIN GROUP (ORDER BY x) FROM unnest(latencies) AS x),
        (SELECT percentile_cont(0.95) WITHIN GROUP (ORDER BY x) FROM unnest(latencies) AS x),
        pg_temp.mli_overlap(ids, exact_ids)::float8 / greatest(1, coalesce(array_length(exact_ids, 1), 0)),
        NULLIF(stats->>'multivector_raw_subvector_hits', '')::int,
        NULLIF(stats->>'multivector_unique_docs', '')::int,
        NULLIF(stats->>'multivector_doc_candidates', '')::int,
        NULLIF(stats->>'multivector_exact_rerank_pairs', '')::bigint,
        NULLIF(stats->>'multivector_memory_bytes_estimate', '')::bigint,
        COALESCE(stats->>'fusion_strategy', 'dense_only'),
        p_notes);
END;
$$;

\echo '== running multivector slope cases =='
DO $$
DECLARE
    docs int;
    final_k int;
    l int;
    q int;
    d int;
    r int;
    ks int;
BEGIN
    SELECT m.docs, m.final_k INTO docs, final_k FROM mli_params m;

    FOREACH l IN ARRAY ARRAY[8, 32, 128] LOOP
        PERFORM pg_temp.mli_run_case(
            'vary_doc_vectors_l',
            'L=' || l,
            docs, l, 4, 64, 25,
            NULL,
            'D constant; expected build nodes and exact pairs scale with L');
    END LOOP;

    FOREACH q IN ARRAY ARRAY[4, 16, 64] LOOP
        PERFORM pg_temp.mli_run_case(
            'vary_query_vectors_q',
            'Q=' || q,
            docs, 32, q, 64, 25,
            NULL,
            'L constant; expected raw hits/searches and exact pairs scale with Q');
    END LOOP;

    FOREACH d IN ARRAY ARRAY[32, 96, 128] LOOP
        PERFORM pg_temp.mli_run_case(
            'vary_dimensions_d',
            'd=' || d,
            docs, 32, 16, 64, 25,
            d,
            'D/L/Q fixed; exact MaxSim work should scale with vector dimension d');
    END LOOP;

    FOREACH r IN ARRAY ARRAY[0, 25, 100] LOOP
        PERFORM pg_temp.mli_run_case(
            'vary_exact_rerank_r',
            CASE WHEN r = 0 THEN 'R=off' ELSE 'R=' || r END,
            docs, 32, 16, 64, r,
            NULL,
            'Approx candidate collection fixed; exact pairs should track rerank docs * Q * L');
    END LOOP;

    FOREACH ks IN ARRAY ARRAY[16, 64, 256] LOOP
        PERFORM pg_temp.mli_run_case(
            'vary_subvector_k_ks',
            'Ks=' || ks,
            docs, 32, 16, ks, 25,
            NULL,
            'Expected raw hits and unique docs grow with per-token subvector budget');
    END LOOP;
END $$;

\echo '== multivector O-slope report =='
SELECT experiment,
       variant,
       doc_count AS docs,
       dimensions AS d,
       doc_vectors AS l,
       query_vectors AS q,
       subvector_k AS ks,
       exact_rerank AS rerank,
       exact_rerank_k AS r,
       node_count AS graph_nodes,
       pg_size_pretty(index_bytes) AS index_size,
       round(build_ms::numeric, 1) AS build_ms,
       round(p50_ms::numeric, 3) AS p50_ms,
       round(p95_ms::numeric, 3) AS p95_ms,
       round(recall_at_k::numeric, 3) AS recall_at_k,
       raw_subvector_hits AS raw_hits,
       unique_docs,
       doc_candidates,
       exact_rerank_pairs AS exact_pairs,
       accumulator_memory_bytes AS accum_bytes,
       fusion_strategy
FROM mli_report
ORDER BY
    CASE experiment
        WHEN 'vary_doc_vectors_l' THEN 1
        WHEN 'vary_query_vectors_q' THEN 2
        WHEN 'vary_dimensions_d' THEN 3
        WHEN 'vary_exact_rerank_r' THEN 4
        WHEN 'vary_subvector_k_ks' THEN 5
        ELSE 99
    END,
    dimensions,
    doc_vectors,
    query_vectors,
    exact_rerank_k,
    subvector_k;

\echo '== expected slope sanity checks =='
SELECT experiment,
       variant,
       node_count AS graph_nodes,
       doc_count * doc_vectors AS expected_subnodes,
       (node_count = doc_count * doc_vectors) AS graph_nodes_match,
       dimensions AS d,
       fusion_strategy,
       exact_rerank_pairs AS observed_exact_pairs,
       CASE
           WHEN exact_rerank = 'off' THEN 0
           ELSE exact_rerank_pairs
       END AS bounded_exact_work,
       notes
FROM mli_report
ORDER BY
    CASE experiment
        WHEN 'vary_doc_vectors_l' THEN 1
        WHEN 'vary_query_vectors_q' THEN 2
        WHEN 'vary_dimensions_d' THEN 3
        WHEN 'vary_exact_rerank_r' THEN 4
        WHEN 'vary_subvector_k_ks' THEN 5
        ELSE 99
    END,
    dimensions,
    doc_vectors,
    query_vectors,
    exact_rerank_k,
    subvector_k;

\echo '== DONE multivector_late_interaction =='
