-- Native code-page cache locality diagnostic.
--
-- Reports turbohybrid_last_scan_stats()->graph_code_pages on a warm 1536-dim
-- 4-bit scan to confirm scored candidates are served from already-loaded code
-- pages / the in-memory codeArena rather than re-read per candidate.
--
--   psql -d <db> -f benchmarks/code_page_locality.sql
--
-- Interpreting graph_code_pages:
--   load_attempts          calls to PgturbohybridGraphLoadCodePage this scan
--   cache_hits / cache_misses, hit_rate
--   code_pages_read        code pages actually read+copied this scan
--   code_tuples_copied     code tuples copied into storage this scan
--   arena_used_bytes       code copied into storage this scan (0 when fully
--                          served from the cross-scan native cache)
--   arena_allocated_bytes  contiguous code-arena size (0 when the index exceeds
--                          the 512MB native-cache cap -> per-node code buffers)
--   pages_read_per_scored_code  code pages read / graph_scored_codes (low = good locality)
--
-- Expected on a cacheable index (<=512MB resident): hit_rate=1.0, misses=0,
-- code_pages_read=0 -- the whole index's codes are resident in codeArena, so no
-- per-candidate page cache is needed.  On indexes exceeding the cap (uncached,
-- fresh storage per scan) intra-scan page locality still serves several scored
-- candidates per page read (codePagesLoaded); the lever to reduce the remaining
-- reads is code-page layout / reordering (cluster co-visited nodes onto shared
-- pages) -- not per-candidate / element-tuple block-grouping.

\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS pgturbohybrid CASCADE;

DROP TABLE IF EXISTS cpl_docs;
CREATE TABLE cpl_docs (id int, embedding vector(1536), body_tsv tsvector);
INSERT INTO cpl_docs
SELECT g,
       (SELECT array_agg((sin(g * 0.001 + d * 0.017) + 0.05 * sin(g * 0.7 + d))::real ORDER BY d)
        FROM generate_series(1, 1536) AS d)::real[]::vector,
       to_tsvector('english', 'document number ' || g)
FROM generate_series(1, 10000) AS g;
CREATE INDEX cpl_idx ON cpl_docs
  USING turbohybrid (embedding vector_cosine_turbohybrid_ops,
                     body_tsv bm25_tsvector_turbohybrid_ops)
  WITH (quantization_bits = 4);
ANALYZE cpl_docs;

SET enable_seqscan = off;
SET max_parallel_workers_per_gather = 0;
SET jit = off;

DO $$
DECLARE k int;
BEGIN
    FOR k IN 1..50 LOOP
        PERFORM id FROM cpl_docs ORDER BY embedding <~> turbohybrid_query(
          vector_query => (SELECT embedding FROM cpl_docs WHERE id = (k * 131) % 10000 + 1)) LIMIT 10;
    END LOOP;
END $$;

SELECT id FROM cpl_docs
  ORDER BY embedding <~> turbohybrid_query(vector_query => (SELECT embedding FROM cpl_docs WHERE id = 4242))
  LIMIT 10;

\echo '== warm code-page locality =='
SELECT jsonb_pretty(turbohybrid_last_scan_stats() -> 'graph_code_pages') AS graph_code_pages;
SELECT (turbohybrid_last_scan_stats() ->> 'graph_scored_codes') AS graph_scored_codes,
       (turbohybrid_last_scan_stats() ->> 'graph_code_pages_read') AS graph_code_pages_read;

DROP TABLE cpl_docs;
RESET enable_seqscan;
RESET max_parallel_workers_per_gather;
RESET jit;
