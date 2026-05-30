-- Dense TurboHybrid scan diagnostics
--
-- Explains WHERE a dense TurboHybrid query spends time so you can tell SIMD
-- scoring cost apart from page/tuple/heap overhead, and confirm exactly which
-- scoring kernel ran (e.g. avx512vnni vs the avx2_lut_gather scalar fallback).
-- Useful when comparing TurboHybrid dense latency against pgvector HNSW.
--
-- Usage (defaults target the DBPedia benchmark schema in
-- benchmarks/dbpedia_openai3_large.py; override -v as needed):
--
--   psql -d pgturbohybrid_dbpedia_1m \
--        -v tbl=dbpedia_docs -v vcol=embedding \
--        -f benchmarks/dev/dense_scan_diagnostics.sql
--
-- A query vector is self-sourced from the first table row unless you pass
-- -v qvec='[...]'.  dense_k/final_k can be overridden with -v dense_k / -v final_k.

\set ON_ERROR_STOP on

\if :{?tbl}
\else
  \set tbl dbpedia_docs
\endif
\if :{?vcol}
\else
  \set vcol embedding
\endif
\if :{?dense_k}
\else
  \set dense_k 100
\endif
\if :{?final_k}
\else
  \set final_k 10
\endif
\if :{?qvec}
\else
  -- Self-source a query vector from an arbitrary table row (cheap: no sort).
  SELECT :vcol::text AS qvec FROM :tbl LIMIT 1 \gset
\endif

-- On small tables the planner may prefer a seq scan; force the index path so
-- the diagnostic reflects the TurboHybrid graph scan.  jit=off removes the
-- one-off JIT cost that would otherwise mask steady-state numbers.
SET enable_seqscan = off;
SET jit = off;

-- Warm the per-backend scan state.  The FIRST graph scan in a backend pays a
-- one-time allocation/setup cost (visible as a large graph_prepare_us); run
-- once so the measured query below reflects steady state, not cold start.
SELECT 1
FROM :tbl
ORDER BY :vcol <~> turbohybrid_query(vector_query => :'qvec'::vector,
                                     dense_k => :dense_k, final_k => :final_k)
LIMIT :final_k;

-- 1) Planner / buffer / settings view.  Confirms the index is used and splits
--    "shared hit" (cached pages) from "read" (page-fetch overhead), plus the
--    runtime GUCs in effect.
EXPLAIN (ANALYZE, BUFFERS, SETTINGS)
SELECT doc_id
FROM :tbl
ORDER BY :vcol <~> turbohybrid_query(vector_query => :'qvec'::vector,
                                     dense_k => :dense_k, final_k => :final_k)
LIMIT :final_k;

-- 2) Kernel + per-phase scan diagnostics for the query just run (backend-local).
--    How to read it to separate SIMD scoring from page/tuple overhead:
--
--      WHICH KERNEL RAN
--        dense_scoring_kernel           selected SIMD code scorer (e.g. avx512vnni,
--                                       avxvnni, query_split_avx2, neon, scalar)
--        dense_scalar_fallback_kernel   scorer for codes that miss the query-split
--                                       path (avx2_lut_gather on x86, else scalar)
--        dense_exact_kernel             exact distance kernel used for rescoring
--        query_split_enabled / dimensions / quantization_bits
--
--      SIMD SCORING WORK  (the compute cost)
--        graph_scored_codes             total approximate code scores
--        graph_simd_scored_codes        scored via fast SIMD (batch + single-node)
--        graph_scalar_scored_codes      scored via the slow scalar/LUT fallback
--        graph_batch_scored_codes       subset scored via the batch-of-4 kernel
--        graph_batch_us                 CPU time inside the SIMD scoring kernels
--
--      PAGE / TUPLE / HEAP OVERHEAD  (the not-SIMD cost)
--        graph_code_pages_read          element (code) pages read/locked
--        graph_adj_pages_read           neighbor (adjacency) pages read/locked
--        heap_tuples_returned           heap tuples returned by the scan
--        candidate_objects_allocated    candidate objects materialized
--        graph_traverse_us / graph_heap_us / graph_sort_us / graph_prepare_us
--
--      EXACT RESCORE / BUDGETS / TOGGLES
--        graph_rescore_count            items rescored with the exact distance
--        graph_effective_search_ef / graph_oversampling / dense_candidates_effective
--        detected_sql_limit
--        exact_storage / residual_rerank_active / graph_rescore_band_active /
--        graph_exact_cache_active
--
--    Rule of thumb: if graph_batch_us dominates dense_elapsed_us with
--    graph_*_pages_read ~ 0, the query is SIMD/compute bound; if page reads or
--    heap_tuples_returned are large, it is I/O / tuple bound; a nonzero
--    graph_scalar_scored_codes means the slow avx2_lut_gather path ran.
SELECT jsonb_pretty(turbohybrid_last_scan_stats());

-- The literal probe named in the diagnostics task.  NOTE: this EXPLAINs the
-- stats-function call itself (a trivial scalar evaluation) -- it does NOT run a
-- dense scan.  The dense query in step (1) is what populates the stats read by
-- turbohybrid_last_scan_stats().
EXPLAIN (ANALYZE, BUFFERS, SETTINGS)
SELECT turbohybrid_last_scan_stats();

RESET enable_seqscan;
RESET jit;
