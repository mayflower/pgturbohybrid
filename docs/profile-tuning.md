# Profile Tuning

Deep tuning guidance for `pgturbohybrid` retrieval profiles and the
query-time / build-time knobs around them. This was moved out of the README
to keep that a concise landing page; see the README "Profile choice" and
"Deep Profile Tuning" sections for the short version.

## Matched-Recall And Quality Modes

Use `matched_recall` when you want a compact 4-bit, exact-free index but need a
single named setting for pgvector/Qdrant-like dense recall comparisons:

```sql
SET turbohybrid.profile = 'matched_recall';
```

This profile keeps `quantization_bits = 4` and `exact_storage = off`, but new
low-dimensional indexes use heuristic graph neighbor selection, exact build
distances during graph construction, `ef_construction = 192`,
`ef_search = 128`, `graph_oversampling = 8`, final top-k heap rescore, and one
native segment unless `native_segments` is explicitly set. Exact build
distances are build-time only; they do not store full vectors in the index.
Changing build-time reloptions such as neighbor selection, build distance,
residual sketches, entry sidecar, graph windows, or segment count for an
existing index requires `REINDEX`. Query-time GUCs such as heap rescore,
adaptive widening, uncertainty retry, residual mode, and fusion can be compared
without rebuilding unless they depend on build-time index contents.

`quantization_bits = 8` is available as an opt-in scalar-safe prototype for
recall experiments:

```sql
CREATE INDEX documents_turbohybrid_q8_idx ON documents
USING turbohybrid (embedding vector_cosine_turbohybrid_ops)
WITH (quantization_bits = 8, exact_storage = off);
```

The 8-bit path stores one byte per dimension and uses scalar code scoring. It
does not reuse the optimized 4-bit split/LUT SIMD kernels, is not the default,
and should be benchmarked against 4-bit, 4-bit plus residual rerank, and
`exact_storage = on` before adopting it for a workload. `turbohybrid_last_scan_stats()`
reports `quantization_bits = 8` and `dense_scorer = scalar_8bit` when this path
is active.

Use `high_recall` when you want near-exact dense recall from a compact 4-bit,
**exact-free** index and have latency headroom to spend:

```sql
SET turbohybrid.profile = 'high_recall';
```

`high_recall` keeps `quantization_bits = 4` and `exact_storage = off`, reuses
`matched_recall`'s candidate budgets, and additionally resolves
`dense_heap_rescore` to `band` at scan time (exact rescore of the full final
candidate band by re-reading vectors from the heap), turns
`dense_adaptive_widening` off, and defaults new indexes to
`ef_construction = 256`, `ef_search = 192`, `graph_oversampling = 12`. Band
rescore can recover ordering quality when 4-bit code scoring finds good
candidates but mis-ranks them, without storing full vectors in the index. Pair
it with a heuristic build for best results, then verify the recall/p95 tradeoff
with the retrieval-quality grid or your external benchmark before adopting it:

```sql
SET turbohybrid.profile = 'high_recall';
SET turbohybrid.dense_build_neighbor_select = 'heuristic';
SET turbohybrid.dense_build_distance = 'code';

CREATE INDEX documents_turbohybrid_high_recall_idx ON documents
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (
    quantization_bits = 4,
    exact_storage = off,
    graph_ef_construction = 256,
    graph_ef_search = 192,
    graph_oversampling = 12,
    native_segments = 1
);
```

Use `quality` when relevance matters more than lowest latency:

```sql
SET turbohybrid.profile = 'quality';
```

Quality mode is stronger and slower: it uses larger dense and lexical candidate
budgets, conservative BM25 paths, higher default graph search windows, and
heuristic dense-neighbor selection for new indexes. It does not automatically
enable calibrated fusion, phrase/proximity heap lexical rerank, or the bounded
uncertainty retry; those remain explicit benchmark knobs until the grid shows a
quality win at acceptable p95/p99. It usually costs more build CPU and query CPU
than `latency` or `matched_recall`, so compare it at matched recall/precision
instead of only comparing raw p95. For quality-sensitive production evaluation,
benchmark an exact-storage index too:

```sql
CREATE INDEX documents_turbohybrid_quality_idx ON documents
USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    body_tsv bm25_tsvector_turbohybrid_ops
)
WITH (exact_storage = on);
```

For exact-free 4-bit indexes that need a precision check without storing a full
copy of every vector in the index, heap-backed exact rescore can rescore only
the final top-k from the heap:

```sql
SET turbohybrid.dense_heap_rescore = 'topk'; -- or 'band'
```

`topk` fetches heap tuples for the final top-k and recomputes exact vector
distance with the same SIMD kernels used by index-exact rescore. `band` rescans
the full final candidate band. Both add heap I/O. The `latency` profile keeps
heap rescore off; the `high_recall` profile defaults it to `band`; `balanced`,
`matched_recall`, and `quality` may enable `topk` automatically for
low-dimensional exact-free indexes, and an explicit
`SET turbohybrid.dense_heap_rescore = 'off'|'topk'|'band'` always wins over the
profile default; set it to `auto` to return to profile-driven behavior.
Residual rerank is an opt-in, lower-I/O refinement (off in every profile),
built into the index with
`WITH (residual_rerank = on, residual_rerank_bytes = 16|32|64)`. It reorders a
narrow top band from residual sketches stored in the index, so it is **not a
replacement for heap-band rescore**: it recovers recall only when the true
neighbours already sit in that narrow band, whereas heap-band rescore recovers
neighbours from the wider candidate band. Larger `residual_rerank_bytes` reranks
more precisely but grows the index — a build-time storage tradeoff. Residual
sketches are build-time index contents; the scan-time adjustment is controlled by
GUCs:

```sql
SET turbohybrid.dense_residual_rerank_mode = 'calibrated'; -- off | fixed | calibrated
SET turbohybrid.dense_residual_rerank_weight = -1.0;       -- auto
SET turbohybrid.dense_residual_rerank_max_adjust_ratio = 0.15;
```

`fixed` preserves the original hardcoded residual adjustment. `calibrated`
scales the sketch adjustment by the observed final-band distance spread and
clamps it, which makes the adjustment comparable across queries. Inspect
`residual_rerank_mode`, `residual_rerank_weight_effective`,
`residual_rerank_band`, `residual_rerank_max_adjustment`,
`residual_rerank_reordered_count`, and `residual_rerank_topk_changed` in
`turbohybrid_last_scan_stats()`.
Because residual sketches are stored in the index, enabling or changing
`residual_rerank_bytes` for an existing index requires `REINDEX`. Switching
between `off`, `fixed`, and `calibrated` residual adjustment modes is
query-time only, but the mode has no effect unless the index was built with
residual sketches.

For phrase/proximity-like text queries, BM25 can optionally rerank a bounded
candidate prefix by fetching the indexed heap `tsvector` and applying PostgreSQL
text-search ranking semantics:

```sql
SET turbohybrid.bm25_heap_tsvector_rerank = 'auto'; -- off | topk | band | auto
SET turbohybrid.bm25_heap_tsvector_rerank_multiplier = 4;
SET turbohybrid.bm25_heap_tsvector_rerank_weight = 0.10;
```

The default is `off`. `topk` fetches only the final-k BM25/hybrid candidates;
`band` fetches up to `final_k * multiplier`, capped by the BM25 candidate
count; `auto` enables the same bounded band for phrase tsqueries. The adjustment
uses `ts_rank_cd` when heap `tsvector` positions are present and `ts_rank`
otherwise. This does not add positional postings or change the index format,
and it does not run for dense-only queries. Inspect
`bm25_heap_tsvector_rerank_mode`, `bm25_heap_tsvector_rerank_count`,
`bm25_heap_tsvector_rerank_fetch_us`,
`bm25_heap_tsvector_rerank_score_us`, and
`bm25_heap_tsvector_rerank_topk_changed` in
`turbohybrid_last_scan_stats()`.

Do not treat any profile as universally best. Measure latency and relevance on
your dataset. Adaptive dense widening stays off for the `latency` profile, but
`balanced`, `matched_recall`, and `quality` can use a conservative `auto` mode
on low-dimensional exact-free 4-bit indexes when `final_k` is small and the
score boundary looks ambiguous. The separate bounded uncertainty retry stays
off in named production profiles until benchmark evidence shows the p95/p99 cost
is low; test it explicitly:

```sql
SET turbohybrid.dense_uncertainty_retry = 'auto'; -- off | auto | on
SET turbohybrid.dense_uncertainty_retry_max_passes = 1;
SET turbohybrid.dense_uncertainty_retry_multiplier = 1.5;
```

`off` preserves the single traversal path. `auto` retries only when the first
candidate band is underfilled or has flat top-k/boundary gaps, or when sidecar,
payload-filter, residual-rerank, or heap-rescore evidence suggests the band was
uncertain. `on` forces the retry whenever the wider target is bounded by the
node count and `turbohybrid.max_scan_tuples`. Inspect
`dense_uncertainty_retry_triggered`, `dense_uncertainty_retry_reason`,
`dense_uncertainty_final_target`, and `dense_uncertainty_final_ef` in
`turbohybrid_last_scan_stats()`.

Local expansion, entry sidecars, uncertainty retry, calibrated hybrid fusion,
phrase/proximity heap lexical rerank, and residual rerank remain opt-in knobs
for benchmark work unless a named profile explicitly documents otherwise. Entry
sidecars keep using
the fixed metapage node-id array; `entry_sidecar_strategy` controls which
representatives are chosen at build time:

```sql
WITH (
  entry_sidecar = on,
  entry_sidecar_representatives = 128,
  entry_sidecar_strategy = 'hash' -- hash | farthest_code | level_covering | hybrid_level_covering
)
```

`hash` is the default and preserves existing behavior. `farthest_code` uses a
deterministic farthest-point selection in code-distance space, `level_covering`
prefers high-level graph nodes while diversifying by code hash bucket, and
`hybrid_level_covering` combines both. Because the selected node IDs are stored
when the index is built, changing the strategy for an existing index requires
`REINDEX` to change the sidecar contents.
For payload-filtered dense scans over `INCLUDE` int4 columns, the scan path can
also seed graph traversal from the existing payload-ref range:

```sql
SET turbohybrid.payload_entry_seeding = 'auto'; -- off | auto | on
SET turbohybrid.payload_entry_seed_count = 8;   -- max 64
```

This is scan-time only: it does not add payload-routing pages or change the
index format. `auto` is the default and only affects scans with an active int4
payload equality filter; if the payload value has no ref range, traversal falls
back to the normal global/sidecar entry points. Inspect
`payload_entry_seeding_hit`, `payload_entry_seed_count`, and
`payload_entry_seed_range_count` in `turbohybrid_last_scan_stats()`.

To reduce near-duplicate final results from the same document or chunk group,
enable scan-time diversity over an existing int4 `INCLUDE` payload slot:

```sql
SET turbohybrid.final_diversity = 'group_payload'; -- off | group_payload
SET turbohybrid.final_diversity_payload_slot = 0;  -- INCLUDE payload slot
SET turbohybrid.final_diversity_lambda = 0.75;     -- relevance/diversity mix
SET turbohybrid.final_diversity_pool_multiplier = 3;
```

This is off by default and does not fetch heap rows or change the index format.
The payload slot is zero-based in `INCLUDE` column order. If the slot is invalid
or a candidate has no payload value, the scan falls back to normal ranking.
Inspect `final_diversity_mode`, `final_diversity_pool_size`,
`final_diversity_selected`, and
`final_diversity_duplicate_groups_suppressed` in
`turbohybrid_last_scan_stats()`.
`native_segments` is a build/concurrency lever rather than a free quality knob:
the default is one segment, quality/exact-build auto segmenting resolves to one
segment, and multi-segment indexes should be benchmarked with
`turbohybrid.native_segment_budget = sqrt|linear` before using them for
quality-sensitive comparisons.

For a deterministic synthetic dense/hybrid profile sweep, use
`benchmarks/dev/retrieval_quality_grid.sql`. It reports recall or overlap at K,
duplicate groups, elapsed milliseconds, index settings, and selected scan stats
for `latency`, `matched_recall`, `high_recall`, `quality`, residual rerank,
heap rescore, uncertainty retry, entry sidecar, calibrated fusion, lexical heap
rerank, and diversity configurations. Use it as the before/after methodology
for deciding whether a new query-time feature should become a profile default.
For a dense-only profile sweep against a glove-like workload, use
`benchmarks/glove100_recall_latency_grid.sql`.

For an existing workload with known expected ids, use
`benchmarks/dev/tune_retrieval_profile.sql` as a practical autotuning harness.
It consumes an `eval_queries` table, sweeps query-time profiles, candidate
budgets, fusion, residual rerank mode, and heap rescore mode against an existing
TurboHybrid index, then prints all trials, a Pareto frontier, and an optional
recommendation under a p95 latency budget. This is a developer benchmark script;
it does not add a SQL-visible autotuner or change production defaults.

