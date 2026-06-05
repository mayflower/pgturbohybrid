\set ON_ERROR_STOP 1
\set model_alias `echo "${MODEL_ALIAS:-sauerkraut-modern}"`

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgturbohybrid;
CREATE EXTENSION IF NOT EXISTS pg_colbert_llama;

\if :{?EXPECTED_DIM}
SET pg_colbert_llama.expected_dim = :EXPECTED_DIM;
\endif

SELECT set_config('pg_colbert_llama_bench.model_alias', :'model_alias', false);

DROP TABLE IF EXISTS pg_colbert_llama_bench_passages;
CREATE TABLE pg_colbert_llama_bench_passages (
  id int PRIMARY KEY,
  body text NOT NULL,
  body_tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', body)) STORED,
  colbert turbohybrid_multivector
);

INSERT INTO pg_colbert_llama_bench_passages (id, body)
SELECT g,
       format('passage %s about alpha beta gamma delta topic %s', g, g % 17)
FROM generate_series(1, 1000) AS g;

\timing on

DROP TABLE IF EXISTS pg_colbert_llama_bench_encode_metrics;
CREATE TEMP TABLE pg_colbert_llama_bench_encode_metrics (
  id int PRIMARY KEY,
  elapsed_ms double precision NOT NULL
);

DO $$
DECLARE
  passage record;
  started_at timestamptz;
BEGIN
  FOR passage IN
    SELECT id, body
    FROM pg_colbert_llama_bench_passages
    ORDER BY id
  LOOP
    started_at := clock_timestamp();
    UPDATE pg_colbert_llama_bench_passages
    SET colbert = colbert_mv(
      current_setting('pg_colbert_llama_bench.model_alias') || ':doc',
      passage.body
    )
    WHERE id = passage.id;

    INSERT INTO pg_colbert_llama_bench_encode_metrics (id, elapsed_ms)
    VALUES (
      passage.id,
      EXTRACT(epoch FROM clock_timestamp() - started_at) * 1000.0
    );
  END LOOP;
END
$$;

SELECT count(*) AS encoded_docs,
       avg(elapsed_ms) AS avg_doc_encode_ms,
       min(elapsed_ms) AS min_doc_encode_ms,
       max(elapsed_ms) AS max_doc_encode_ms
FROM pg_colbert_llama_bench_encode_metrics;

SELECT colbert_mv(:'model_alias' || ':query', 'alpha beta topic');

CREATE INDEX pg_colbert_llama_bench_dense_idx
ON pg_colbert_llama_bench_passages USING turbohybrid (
  colbert multivector_maxsim_ip_turbohybrid_ops
);

SET enable_seqscan = off;

SELECT id
FROM pg_colbert_llama_bench_passages
ORDER BY colbert <~> turbohybrid_query(
  multivector_query => colbert_mv(:'model_alias' || ':query', 'alpha beta topic'),
  dense_k => 100,
  final_k => 10
)
LIMIT 10;

SELECT turbohybrid_last_scan_stats();
SELECT turbohybrid_estimate_memory('pg_colbert_llama_bench_dense_idx'::regclass);

DROP INDEX pg_colbert_llama_bench_dense_idx;

CREATE INDEX pg_colbert_llama_bench_hybrid_idx
ON pg_colbert_llama_bench_passages USING turbohybrid (
  colbert multivector_maxsim_ip_turbohybrid_ops,
  body_tsv bm25_tsvector_turbohybrid_ops
);

SELECT id
FROM pg_colbert_llama_bench_passages
ORDER BY colbert <~> turbohybrid_query(
  multivector_query => colbert_mv(:'model_alias' || ':query', 'alpha beta topic'),
  text_query => websearch_to_tsquery('simple', 'alpha beta topic'),
  fusion => 'rrf',
  dense_k => 100,
  bm25_k => 100,
  final_k => 10
)
LIMIT 10;

SELECT turbohybrid_last_scan_stats();
SELECT turbohybrid_estimate_memory('pg_colbert_llama_bench_hybrid_idx'::regclass);

RESET enable_seqscan;

-- Suggested GUC sweep:
-- SET turbohybrid.multivector_subvector_k = 100;
-- SET turbohybrid.multivector_unique_docs_per_token = 100;
-- SET turbohybrid.multivector_max_raw_hits_per_token = 400;
-- SET turbohybrid.multivector_adaptive_widening = 'auto';
-- SET turbohybrid.multivector_doc_candidate_k = 100;
-- SET turbohybrid.multivector_exact_rerank_k = 100;
