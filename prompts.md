
Current verified state:

```text
The proxy_vector sidecar issue is fixed in the current build.

10k latency-only, proxy_normalized_mean_f16, f16/auto, EF=100,
candidate_k=800, exact_rerank_k=100:
- p50 ~= 24 ms
- p95 ~= 33 ms
- proxy_lazy_sidecar_vectors = true
- proxy_vector_uses_full_sidecar_for_graph = false
- proxy_vector_near_exhaustive_sidecar_touch = false
- sidecar vectors loaded/query = 100
- sidecar bytes touched/query ~= 3.65 MB, not ~=258 MB
- exact kernel = blocked_neon

The remaining issue is admission quality, not sidecar scan cost.
proxy_candidates is capped by search EF:
- EF=100 -> proxy_candidates=100, limit_source=search_ef
- EF=200 -> proxy_candidates=200, limit_source=search_ef
- EF=800 -> proxy_candidates=800, limit_source=candidate_target

Focused 10k admission/relevance validation for proxy_normalized_mean_f16
with EF 50,100,200, oversampling 1,2, budgets 50,100,200,400,800:
- best row EF=200/os=1, budget=800
- p95 ~= 34 ms
- top1 admission ~= 0.12
- top10 admission recall ~= 0.124
- recall@10 ~= 0.149
- ndcg@10 ~= 0.132

Focused 2k proxy-vs-centroid admission/relevance validation after the
sidecar fix:
- proxy_normalized_mean_f16 best row EF=200/os=1, budget=800:
  p95 ~= 17 ms, top10 admission ~= 0.188, recall@10 ~= 0.249,
  ndcg@10 ~= 0.226
- centroid_mean_f16 best row EF=200/os=2, budget=800:
  p95 ~= 20 ms, top10 admission ~= 0.404, recall@10 ~= 0.442,
  ndcg@10 ~= 0.395

Focused 2k proxy-vs-centroid-lite validation showed centroid_lite has better
qrels but a serving-cost bug:
- centroid_lite_f16 best-quality row EF=100/os=1, budget=800:
  p95 ~= 626 ms, recall@10 ~= 0.542, ndcg@10 ~= 0.483
- it reranked 800 docs even when serving_exact_rerank_k was 100
- fix applied: document-node exact rerank now honors
  turbohybrid.multivector_exact_rerank_k for centroid_lite/document-node
  paths, and regression coverage asserts centroid_lite caps exact rerank docs

Post-fix 2k centroid_lite latency-only smoke with candidate_k=800,
exact_rerank_k=100, EF=100, oversampling=1:
- p50 ~= 459 ms, p95 ~= 485 ms
- exact rerank docs p50/p95 = 100/100
- exact MaxSim rerank p95 ~= 5.5 ms, exact heap fetch p95 ~= 1.5 ms
- candidate-source time p95 ~= 469 ms
- centroid_docs_touched p50 ~= 1987 of 2000 documents
- slow-path warnings are now near-exhaustive docs-scored warnings, not
  exact-rerank-budget warnings

Next work should target candidate admission quality for centroid/proxy paths,
not MaxSim SIMD or broad sidecar/cache work. `centroid_mean` is the first
candidate-source variant with clearly better latency/quality evidence. After
the rerank-cap fix, centroid_lite remains too slow because its posting-list
prefilter touches nearly the whole corpus; optimize or replace centroid_lite
admission before considering it for a serving profile. The benchmark harness
now has an opt-in proxy encoder comparison switch:
`--document-node-serving-grid-include-proxy-encoders`, which exposes
`proxy_max_pool_f16` and `proxy_random_projection_fde_f16` for focused
candidate-admission experiments without widening the default serving grid.

Focused 2k proxy encoder comparison with `proxy_normalized_mean_f16`,
`proxy_max_pool_f16`, `proxy_random_projection_fde_f16`, and
`centroid_mean_f16`:
- best_latency_safe remained empty; no profile met the configured safety
  thresholds.
- `centroid_mean_f16` remained best quality/balanced:
  p95 ~= 19 ms, top10 admission ~= 0.404, recall@10 ~= 0.442,
  ndcg@10 ~= 0.395, mrr@10 ~= 0.813.
- `proxy_max_pool_f16` improved over normalized mean but still trailed
  centroid_mean:
  p95 ~= 17 ms, top10 admission ~= 0.308, recall@10 ~= 0.338,
  ndcg@10 ~= 0.315.
- `proxy_normalized_mean_f16` stayed fastest among viable non-experimental
  proxy baselines with weak admission:
  p95 ~= 15 ms, top10 admission ~= 0.188, recall@10 ~= 0.249,
  ndcg@10 ~= 0.226.
- `proxy_random_projection_fde_f16` was very fast but too weak for admission:
  p95 ~= 7 ms, top10 admission ~= 0.048, recall@10 ~= 0.108,
  ndcg@10 ~= 0.115.

Attempted 10k candidate-source validation for
`proxy_normalized_mean_f16`, `proxy_max_pool_f16`, and `centroid_mean_f16`
was stopped because it spent more than six minutes in the first `CREATE INDEX`
before producing retrieval evidence. A narrower 10k `centroid_mean_f16` run was
also stopped after more than six minutes in `CREATE INDEX`. No partial JSON or
Markdown artifact was produced. This means the next actionable blocker for
promoting `centroid_mean` from 2k evidence to 10k/x00k evidence is build-cost
accounting and optimization for centroid sidecar/proxy indexes, not another
broad serving-grid rerun. Build-cost accounting now needs to be read from
`turbohybrid_last_build_stats()`: `multivector_centroid_build_us`,
`multivector_centroid_build_docs`, `multivector_centroid_build_vectors`,
`multivector_proxy_build_us`, `multivector_doc_sidecar_write_us`,
`multivector_centroid_sidecar_write_us`,
`multivector_centroid_posting_write_us`, and
`multivector_centroid_posting_count`. Centroid subphase attribution is exposed
as `multivector_centroid_cluster_us` and
`multivector_centroid_residual_us`.

Build-cost accounting is now wired through the benchmark build-only path. A
fresh-cluster 200-document `centroid_mean_f16` build-only proof showed:
- index build ~= 13.4 s
- centroid build ~= 6.53 s
- centroid cluster subphase ~= 6.38 s
- centroid residual subphase ~= 0.15 s
- centroid docs = 200
- centroid vectors/postings = 1819

So the current 10k build blocker is specifically k-means centroid clustering
cost, not residual scoring, sidecar writing, graph edge distance calls, or exact
MaxSim rerank.

After caching nearest-selected seed similarities during farthest-first centroid
initialization, the same fresh-cluster 200-document `centroid_mean_f16`
build-only proof showed:
- index build ~= 3.65 s
- centroid build ~= 1.68 s
- centroid cluster subphase ~= 1.53 s
- centroid residual subphase ~= 0.15 s
- centroid docs = 200
- centroid vectors/postings = 1819

This is a material build-time improvement, but centroid clustering remains the
dominant known build phase. The next proof should scale this same build-only
measurement to a larger document count before retrying broad 10k serving-grid
admission validation.

Scaled build-only proofs after the same optimization:
- 1k docs: index build ~= 16.7 s, centroid build ~= 7.70 s,
  centroid cluster ~= 7.01 s, centroid residual ~= 0.69 s,
  centroid vectors/postings = 8709
- 5k docs: index build ~= 47.1 s, centroid build ~= 21.9 s,
  centroid cluster ~= 19.8 s, centroid residual ~= 2.04 s,
  centroid vectors/postings = 35778
- 10k docs: index build ~= 91.7 s, centroid build ~= 41.1 s,
  centroid cluster ~= 37.1 s, centroid residual ~= 3.91 s,
  centroid vectors/postings = 70314

The 10k centroid build is now slow but bounded enough for a focused admission
validation. Do not restart the broad exhaustive grid first; run a narrow
10k candidate-source validation for the profiles with current evidence:
`proxy_normalized_mean_f16`, `proxy_max_pool_f16`, and `centroid_mean_f16`.

Focused 10k candidate-source validation after the centroid build fix:
- command shape: 10k docs, 50 queries, profiles
  `proxy_normalized_mean_f16`, `proxy_max_pool_f16`, `centroid_mean_f16`,
  EF grid 50/100/200, oversampling 1/2, executed budget 800,
  serving exact rerank k 100.
- total elapsed ~= 175 s
- index builds ~= 133 s total for three profiles
- exact baseline ~= 29.6 s for 50 queries
- retrieval ~= 11.8 s for 900 retrieval queries
- no profile met `best_latency_safe`; the default top10 admission threshold is
  still 0.80.
- best quality/balanced row was `centroid_mean_f16`, EF=200, oversampling=2:
  p95 ~= 18.9 ms, top1 admission ~= 0.26, top10 admission ~= 0.174,
  recall@10 ~= 0.214, ndcg@10 ~= 0.224, mrr@10 ~= 0.637.
- `proxy_max_pool_f16`, EF=200, oversampling=1:
  p95 ~= 19.1 ms, top1 admission ~= 0.26, top10 admission ~= 0.120,
  recall@10 ~= 0.144, ndcg@10 ~= 0.151.
- `proxy_normalized_mean_f16`, EF=100, oversampling=2:
  p95 ~= 17.8 ms, top1 admission ~= 0.16, top10 admission ~= 0.078,
  recall@10 ~= 0.129, ndcg@10 ~= 0.125.

The next candidate-quality step should test whether higher EF/candidate graph
admission improves `centroid_mean_f16` enough before adding another algorithm.
The serving-grid harness currently sweeps EF only 50/100/200, while prior
stats showed proxy candidates are capped by search EF. Add a bounded way to run
serving-grid EF values such as 400/800 for focused evidence, or use an existing
single-profile admission path if it can produce the same recommendation rows.

The serving-grid harness now has focused overrides:
- `--document-node-serving-ef-grid`
- `--document-node-serving-oversampling-grid`

Focused 10k high-EF validation for `centroid_mean_f16` only:
- 10k docs, 50 queries, EF 400/800, oversampling 1/2, budget 800,
  serving exact rerank k 100.
- total elapsed ~= 133 s
- index build ~= 101 s
- exact baseline ~= 27.7 s
- retrieval ~= 4.5 s for 200 retrieval queries
- best row EF=800, oversampling=1:
  p50 ~= 23.9 ms, p95 ~= 25.9 ms, top1 admission ~= 0.52,
  top10 admission ~= 0.356, recall@10 ~= 0.367, ndcg@10 ~= 0.354,
  mrr@10 ~= 0.764.
- EF=400 improved over EF=200 but plateaued below EF=800:
  top10 admission ~= 0.234, ndcg@10 ~= 0.273.

This proves candidate quality improves with EF, but the serving-mode metric is
still bounded by exact rerank k=100. Before changing algorithms, run a focused
diagnostic with `serving_exact_rerank_mode = admission_exhaustive` for
`centroid_mean_f16`, EF=800, budget=800, to separate graph admission quality
from the serving rerank cap.

Focused 10k admission-exhaustive diagnostic for `centroid_mean_f16`:
- 10k docs, 50 queries, EF=800, oversampling=1, budget=800,
  `serving_exact_rerank_mode = admission_exhaustive`,
  `multivector_exact_rerank_k = 800`.
- total elapsed ~= 129 s
- index build ~= 96.7 s
- exact baseline ~= 27.7 s
- retrieval ~= 4.9 s for 50 retrieval queries
- p50 ~= 98.6 ms, p95 ~= 113.5 ms
- top1 admission ~= 0.56
- top10 admission ~= 0.388
- recall@10 ~= 0.397
- ndcg@10 ~= 0.369
- mrr@10 ~= 0.774
- exact rerank docs p50/p95 = 800/800
- sidecar vectors loaded/query = 800
- sidecar bytes touched/query ~= 25.2 MB
- exact kernel = blocked_neon

The exhaustive diagnostic only improves top10 admission from ~=0.356 to
~=0.388 at the same EF/budget. That means the dominant quality bottleneck is
candidate admission into the 800-doc graph/proxy band, not the serving
exact-rerank cap of 100. Do not spend the next slice on MaxSim SIMD, heap
fetches, or sidecar I/O for this quality issue. The next algorithmic work
should improve candidate generation itself: multi-proxy document admission,
centroid-lite pruning that does not touch nearly the whole corpus, BM25 or
learned-sparse rescue, or a better graph entry/search strategy.
```

Historical sweep that motivated the sidecar prompts before the sidecar fix:

```text
rerank_k 25  -> p95 ~55 ms, exact rerank p95 ~0.9 ms
rerank_k 50  -> p95 ~58 ms, exact rerank p95 ~1.8 ms
rerank_k 100 -> p95 ~59 ms, exact rerank p95 ~3.8 ms
```

So even at `rerank_k=100`, exact MaxSim is only a few milliseconds. The query is slow because the `proxy_vector` path still does roughly:

```text
~258 MB sidecar bytes touched/query
~34k sidecar pages read/query
~51–53 ms sidecar load/query
```

That means the next Codex prompts should be **not** “optimize MaxSim” and **not** “change rerank defaults,” but:

```text
1. identify why proxy_vector touches near-full sidecar
2. separate cache warmup from per-query sidecar reads
3. make proxy_vector admission use only proxy graph data
4. touch full multivectors only for bounded exact rerank
```

The existing stats plumbing already exposes graph timing, heap-rescore timing, sidecar counters, exact rerank counters and scan stats; the missing piece is now path-specific attribution and then eliminating the sidecar touch.

## Prompt 1 — Trace why `proxy_vector` touches 258 MB/query

```text
You are working in github.com/mayflower/pgturbohybrid.

We ran a latency-only rerank sweep for document-node proxy_vector.

Findings:
- candidate_source = proxy_vector
- storage = f16
- cache = auto
- ef = 100
- oversampling = 1
- candidate_k = 800
- requested exact rerank k = 25/50/100/200
- runtime exact rerank docs:
  - 25 -> 25 docs
  - 50 -> 50 docs
  - 100 -> 100 docs
  - 200 -> still 100 docs because proxy_candidates=100
- exact kernel = blocked_neon
- exact rerank p95 at k=100 ~= 3.8 ms
- total query p95 at k=100 ~= 59 ms
- sidecar load p50/p95 ~= 51/53 ms
- sidecar bytes touched ~= 257,899,824 bytes/query
- sidecar pages read ~= 34,072 pages/query

Conclusion:
Exact MaxSim rerank is not the bottleneck. Sidecar access is.

Do not change code in this prompt.

Task:
Trace the current proxy_vector document-node scan path and produce a code-level diagnosis.

Inspect the C code paths for:
- candidate_source = proxy_vector
- multivector_graph = document_nodes
- document-node graph traversal
- proxy vector graph scoring
- document-node sidecar loading
- f16/sq8/f32 sidecar access
- exact heap MaxSim rerank
- native cache build/reuse
- stats increments for sidecar bytes/pages/vectors loaded

Answer these questions with file/function names:

1. During proxy_vector graph traversal, does the scan score graph nodes using:
   a. only the fixed-dimensional proxy vector, or
   b. full document multivector sidecar MaxSim?

2. When are multivector_doc_sidecar_bytes_touched and pages_read incremented?
   - resident cache build?
   - per-query sidecar page reads?
   - per-candidate vector reconstruction?
   - exact rerank?
   - near-exhaustive fallback?

3. Are the 258 MB touched per query a one-time cache warmup being reported per scan, or repeated per-query work?

4. Why does proxy_candidates=100 when candidate_k=800?
   - ef limit?
   - oversampling?
   - doc_graph_rescore_k?
   - final_k?
   - graph traversal candidate cap?
   - proxy admission cap?

5. Does proxy_vector accidentally take the same path as document_nodes/full-MaxSim scoring anywhere?

6. Are there existing warnings/stats that distinguish:
   - document_node_proxy_vector_graph_traversal
   - document_node_f16_sidecar_graph_traversal
   - document_node_f16_sidecar_exact_scan
   - near-exhaustive sidecar scan
   - resident cache build

7. Is exact rerank using heap multivectors or sidecar multivectors in this run?

Deliverable:
A short diagnosis with:
- suspected exact function causing the sidecar touch
- whether this is a stats-accounting issue or actual query work
- safest next code change
- tests required

Do not optimize yet.
Do not change SQL semantics.
Do not change persisted formats.
```

---

## Prompt 2 — Add precise stats for proxy sidecar-touch attribution

Run this after Prompt 1.

```text
You are working in github.com/mayflower/pgturbohybrid.

We need to distinguish these cases for document_nodes + proxy_vector:

A. proxy graph traversal uses only proxy vectors and sidecar bytes are cache warmup/accounting.
B. proxy graph traversal accidentally loads full multivectors for every scored document.
C. sidecar is read only for bounded exact rerank candidates.
D. near-exhaustive fallback or exact sidecar scan is happening unintentionally.

Goal:
Add precise scan stats for proxy_vector sidecar attribution.

Do not change query results.
Do not change persisted formats.
Do not optimize yet.

Add stats to turbohybrid_last_scan_stats() for document-node multivector scans:

Proxy graph/admission:
- proxy_graph_nodes_visited
- proxy_graph_edges_visited
- proxy_graph_candidates_seen
- proxy_candidates_returned
- proxy_vector_scores_computed
- proxy_vector_score_time_us

Full multivector sidecar during proxy path:
- proxy_full_sidecar_vectors_loaded
- proxy_full_sidecar_bytes_touched
- proxy_full_sidecar_pages_read
- proxy_full_sidecar_load_time_us
- proxy_full_sidecar_reconstruct_time_us

Exact rerank:
- proxy_exact_rerank_docs
- proxy_exact_rerank_heap_fetches
- proxy_exact_rerank_sidecar_fetches
- proxy_exact_rerank_bytes_touched
- proxy_exact_rerank_time_us

Cache/accounting:
- sidecar_cache_build_this_query
- sidecar_cache_build_bytes
- sidecar_cache_build_time_us
- sidecar_query_bytes_touched
- sidecar_query_pages_read
- sidecar_query_vectors_loaded
- sidecar_query_load_time_us

Slow-path booleans:
- proxy_vector_uses_full_sidecar_for_graph
- proxy_vector_near_exhaustive_sidecar_touch
- proxy_vector_sidecar_touch_reason

Definitions:
- sidecar_cache_build_* counts one-time resident/native cache build work.
- sidecar_query_* counts per-query work after cache is available.
- proxy_full_sidecar_* counts full multivector sidecar accesses before exact final rerank.
- proxy_exact_rerank_* counts only final exact rerank work.
- Do not double-count cache build bytes as per-query bytes touched.

Add benchmark extraction:
- include all new fields in latency-only JSON
- include p50/p95 summaries where the latency-only benchmark aggregates scan stats
- include them in markdown

Tests:
- Python extraction self-check preserves new fields.
- SQL regression/TAP with tiny document_nodes proxy_vector query checks fields exist or zero consistently.
- Tiny proxy_vector query with candidate_k=2 should not report proxy_vector_near_exhaustive_sidecar_touch.
- exact_doc_scan is allowed to touch all docs and must not be mislabeled as proxy_vector near-exhaustive.

Validation:
Run:
  nix --extra-experimental-features 'nix-command flakes' develop --command th-installcheck
  nix --extra-experimental-features 'nix-command flakes' develop --command th-smoke
  nix --extra-experimental-features 'nix-command flakes' develop --command python benchmarks/dbpedia_colbert_multivector.py --help
```

---

## Prompt 3 — Fix proxy_vector if it loads full sidecar during graph admission

Run this only if Prompt 1 or 2 proves `proxy_vector_uses_full_sidecar_for_graph = true`.

```text
You are working in github.com/mayflower/pgturbohybrid.

Precondition:
Stats/diagnosis show that document_nodes + proxy_vector loads or reconstructs full document multivectors during proxy graph traversal or candidate admission.

Goal:
Make proxy_vector admission use only the persisted fixed-dimensional proxy vector for graph traversal and candidate admission.

Do not change final ranking semantics.
Do not change persisted formats unless absolutely necessary.
Do not use approximate proxy scores for final SQL ordering.

Required behavior:
- proxy_vector graph traversal scores candidates using only the persisted proxy vector.
- Full document multivectors are touched only for:
  1. bounded exact rerank, or
  2. explicit exact_doc_scan/document_nodes full-MaxSim modes.
- candidate_source = proxy_vector must not perform near-exhaustive full sidecar scans.
- final SQL order remains exact MaxSim over original multivectors for retained candidates.

Implementation tasks:
1. Locate where proxy_vector candidate source enters document-node scan.
2. Split proxy admission from full sidecar MaxSim scoring if they are currently coupled.
3. Ensure graph traversal scorer for proxy_vector uses the fixed-dimensional proxy key.
4. Ensure exact rerank still fetches original heap multivectors or the correct exact source.
5. Keep all MVCC/visibility behavior unchanged.
6. Preserve document/heap tuple identity; never rank SQL results by graph node ID.

Stats:
After the fix, proxy_vector with candidate_k=800, ef=100, oversampling=1 should report:
- proxy_vector_uses_full_sidecar_for_graph = false
- proxy_full_sidecar_vectors_loaded near 0 before exact rerank
- proxy_exact_rerank_docs <= effective exact rerank k
- sidecar_query_bytes_touched much lower than 258 MB/query, unless it is clearly labeled cache build
- proxy_candidates_returned visible
- exact rerank time still reported

Tests:
- Tiny deterministic document_nodes proxy_vector query returns same top-k as before when candidate/rerank budget is exhaustive.
- proxy_vector candidate_k=10 does not load full sidecar for all documents.
- document_nodes full-MaxSim mode may still use full sidecar and should be labeled differently.
- exact_doc_scan remains exact oracle.
- No silent fallback to exact_doc_scan/plain fallback.

Validation:
Run:
  nix --extra-experimental-features 'nix-command flakes' develop --command th-installcheck
  nix --extra-experimental-features 'nix-command flakes' develop --command th-smoke

Then rerun latency-only 10k proxy_vector benchmark:
- same 25 queries
- candidate_k=800
- exact_rerank_k=100
- ef=100
- oversampling=1
- f16
- auto cache

Report before/after:
- p50/p95/p99
- qps
- sidecar bytes/pages
- sidecar load time
- proxy full-sidecar vectors loaded
- exact rerank docs/time
```

---

## Prompt 4 — Fix accounting if 258 MB is cache-build, not per-query work

Run this if Prompt 1 or 2 shows sidecar bytes are mostly resident/native cache build accounted on every scan.

```text
You are working in github.com/mayflower/pgturbohybrid.

Precondition:
The large sidecar bytes/pages for proxy_vector are primarily cache build or cache load accounting, not repeated per-query sidecar reads.

Goal:
Separate one-time cache build accounting from per-query sidecar touch accounting.

Do not change query results.
Do not change persisted formats.
Do not change candidate generation.

Tasks:
1. Audit sidecar/native cache counters:
   - multivector_doc_sidecar_bytes_touched
   - multivector_doc_sidecar_pages_read
   - multivector_doc_sidecar_vectors_loaded
   - native_cache_built_this_scan
   - native_cache_reused
   - native_cache_bytes
   - native_cache_exact_bytes

2. Add clear separate counters:
   - sidecar_cache_build_bytes
   - sidecar_cache_build_pages_read
   - sidecar_cache_build_time_us
   - sidecar_query_bytes_touched
   - sidecar_query_pages_read
   - sidecar_query_vectors_loaded
   - sidecar_query_time_us

3. Keep old fields for backward compatibility, but document whether they include cache build or query work.

4. Ensure latency-only benchmark reports:
   - cache build warmup excluded/included
   - first-query stats separately from warm-query stats
   - native cache reuse rate

5. Add warmup option if missing:
   --serving-latency-warmup-queries INT
Default:
   1 or enough to build resident cache once for latency-only runs.

6. In latency-only results, report:
   - cold p50/p95
   - warm p50/p95
   - cache_build_queries
   - warm_queries
   - cache_reused_on_warm_queries

Tests:
- First query with cache build reports cache_build bytes.
- Subsequent query reports cache_reused and low query bytes.
- Old fields still exist.
- Markdown distinguishes cold vs warm latency.

Validation:
Run:
  nix --extra-experimental-features 'nix-command flakes' develop --command th-installcheck
  nix --extra-experimental-features 'nix-command flakes' develop --command th-smoke
```

---

## Prompt 5 — Fix auto cache policy if `auto` chooses paged or rebuilds repeatedly

Run this if `auto` is choosing a slow mode or repeatedly rebuilding cache.

```text
You are working in github.com/mayflower/pgturbohybrid.

Precondition:
Stats show proxy_vector f16 auto cache is slow because:
- auto selected paged when resident would fit, or
- native/resident cache is rebuilt repeatedly, or
- native cache is not reused between queries.

Goal:
Fix cache policy/reuse, not retrieval semantics.

Tasks:
1. Inspect auto policy for:
   turbohybrid.multivector_doc_storage_cache = auto

2. Inspect:
   turbohybrid.native_cache_max_mb
   turbohybrid_estimate_memory(index)
   native cache key/lifetime
   relation/index invalidation
   storage kind f32/f16/sq8
   document-node sidecar bytes
   resident cache eligibility

3. Add diagnostics if missing:
   - auto_cache_decision
   - auto_cache_reason
   - sidecar_estimated_bytes
   - native_cache_max_bytes
   - resident_cache_allowed
   - resident_cache_rejected_reason
   - native_cache_key
   - native_cache_reuse_blocked_reason

4. If auto rejects resident although sidecar fits:
   - fix decision logic
   - add regression coverage.

5. If native cache rebuilds every query:
   - fix cache reuse key/lifetime if safe
   - add test proving reuse across two queries in one backend/session.

6. If native_cache_max_mb is simply too low:
   - do not hide it.
   - add benchmark slow-path warning:
     "auto selected paged because estimated sidecar bytes exceed native_cache_max_mb"
   - update docs with recommended setting.

Tests:
- auto chooses resident when sidecar estimate fits.
- auto chooses paged when estimate exceeds limit.
- clear rejected reason is emitted.
- repeated query reuses resident cache in same backend.
- no persisted format changes.

Validation:
Run:
  nix --extra-experimental-features 'nix-command flakes' develop --command th-installcheck
  nix --extra-experimental-features 'nix-command flakes' develop --command th-smoke
```

---

## Prompt 6 — Reduce proxy candidate sidecar touch if proxy_candidates is capped at 100

Your sweep found `proxy_candidates=100` even when `candidate_k=800`. That deserves its own investigation.

```text
You are working in github.com/mayflower/pgturbohybrid.

Observation:
In latency-only proxy_vector run:
- requested candidate_k = 800
- requested exact_rerank_k = 200 in one run
- runtime exact rerank docs = 100
- proxy_candidates = 100
- multivector_doc_graph_docs_scored = 100
- ef = 100
- oversampling = 1

Question:
Why does proxy_vector produce only 100 candidates when dense/candidate budget is 800?

Do not change code initially.

Task:
Trace effective candidate limit calculation for document_nodes + proxy_vector.

Inspect:
- dense_k / candidate_k plumbing from benchmark SQL into turbohybrid_query
- multivector_doc_candidate_k
- multivector_exact_rerank_k
- multivector_doc_graph_search_ef
- multivector_doc_graph_oversampling
- multivector_doc_graph_rescore_k
- final_k
- effective_result_target
- proxy candidate cap
- graph traversal EF cap

Deliverable:
Explain whether proxy_candidates=100 comes from:
- EF=100
- a hard default cap
- final_k
- exact_rerank_k
- doc_candidate_k not being applied
- candidate source bug
- benchmark not setting the intended GUC

If the limit is intentional:
- add stats showing the limiting reason:
  proxy_candidate_limit_source
  proxy_candidate_limit_effective

If the benchmark intended 800 candidates but extension used 100 due to plumbing:
- fix benchmark GUC/SQL plumbing.
- add test that candidate_k=800 reaches the extension as multivector_doc_candidate_k=800.

If extension caps proxy candidates at EF and ignores oversampling/candidate_k:
- decide whether that is correct.
- If not correct, adjust proxy traversal to collect up to min(candidate_k, ef * oversampling or explicit cap) without touching full sidecar.
- Add tests.

Validation:
Run Python/self-check tests and th-installcheck if C changes are made.
```

This is important because fixing sidecar access may reveal that candidate count is too low for relevance. The system currently looks fast-ish at 100 candidates but may not be testing the intended `candidate_k=800`.

---

## Prompt 7 — Re-run the decisive benchmark after sidecar/proxy fixes

```text
You are working in github.com/mayflower/pgturbohybrid.

Precondition:
One of the sidecar/proxy fixes has landed:
- proxy_vector no longer touches full sidecar during graph admission, or
- sidecar cache accounting is separated, or
- auto cache reuse/policy is fixed, or
- proxy candidate cap plumbing is fixed.

Run the same latency-only benchmark as before:

- 10k docs
- 25 queries
- candidate_source = proxy_vector
- profile = proxy_normalized_mean_f16
- storage = f16
- cache = auto
- ef = 100
- oversampling = 1
- candidate_k = 800
- exact_rerank_k = 100
- no admission baseline
- no trace

Output:
  .nix-dev/tmp/dbpedia-colbert-serving-latency-10k-after-proxy-sidecar.json
  .nix-dev/tmp/dbpedia-colbert-serving-latency-10k-after-proxy-sidecar.md

Compare against baseline:
- p50 ~57 ms
- p95 ~59 ms
- exact rerank p95 ~3.8 ms
- sidecar load p95 ~53 ms
- sidecar bytes touched ~258 MB/query
- sidecar pages read ~34,072/query
- proxy_candidates=100

Report:
- p50/p95/p99/qps before/after
- proxy_candidates before/after
- exact_rerank_docs before/after
- exact rerank time before/after
- sidecar bytes/pages before/after
- sidecar query bytes vs cache build bytes
- sidecar load time before/after
- graph traversal time before/after
- native cache reused
- slow path warnings

If p95 is still above 25 ms:
- identify new dominant phase.
- do not guess.
```

---

## Prompt 8 — Only after sidecar fix: relevance/admission validation

After reducing sidecar work, validate whether the faster proxy path still admits enough good docs.

```text
You are working in github.com/mayflower/pgturbohybrid.

Precondition:
Latency is improved for proxy_vector and sidecar touch is understood.

Goal:
Run a small admission/relevance check for the optimized proxy path.

Use:
- 10k docs
- max queries 25 or 50
- candidate_source = proxy_vector
- profile = proxy_normalized_mean_f16
- storage = f16
- cache = auto or resident, whichever is selected after sidecar fix
- ef values: 50,100,200
- oversampling: 1,2
- candidate budgets: 100,200,400,800
- exact_rerank_k: 100
- admission_debug_mode: summary unless trace is required for exact admission

Report:
- top1 admission
- top10 admission recall
- recall@10 / ndcg@10 / mrr@10 if qrels are present
- p50/p95 latency
- proxy_candidates
- exact_rerank_docs
- sidecar query bytes/time
- graph traversal time

Compare:
- proxy_normalized_mean_f16
- centroid_mean_f16
- centroid_lite_f16 only if its sidecar behavior is understood

Do not include quantized_inverted_experimental by default.

Output:
  .nix-dev/tmp/dbpedia-colbert-serving-admission-after-sidecar.json
  .nix-dev/tmp/dbpedia-colbert-serving-admission-after-sidecar.md

Do not commit generated files.
```

---

## Short conclusion

The sidecar/cache prompts above are implemented and validated for the current
proxy_vector fast path. Exact rerank is not the current bottleneck, and
proxy_vector no longer touches the full document sidecar during graph
admission.

The next real target is:

```text
candidate admission quality for document_nodes proxy and centroid paths
```

Specifically, the current 10k validation shows:

```text
proxy_normalized_mean_f16 is fast enough for smoke serving latency, but its
top10 admission recall is far below the 0.80 safety threshold.
```

Continue with candidate-source quality work: centroid_mean/centroid_lite
comparison only where sidecar behavior is bounded and understood, larger EF
only when the benchmark is meant to test larger proxy candidate bands, and
additional proxy/centroid admission strategies before any serving-profile GUC.

Current candidate-quality harness update:
- the document-node serving grid now has an opt-in BM25 rescue switch,
  `--document-node-serving-grid-include-bm25-rescue`
- the opt-in rescue profiles are:
  `proxy_normalized_mean_f16_bm25_rescue` and
  `centroid_mean_f16_bm25_rescue`
- when BM25 rescue and proxy-encoder variants are both enabled, the harness
  also exposes `proxy_max_pool_f16_bm25_rescue` for a focused comparison of the
  stronger `max_pool` proxy baseline without changing the default rescue set
- these profiles use `multivector_branch_plan = dense_only`,
  `multivector_bm25_candidate_injection = dense_with_text`, and pass
  `text_query` into `turbohybrid_query`, so the extension's BM25 injection
  branch is exercised without enabling the qdrant-like proxy full-sidecar
  rescore path
- benchmark rows now aggregate BM25 injection evidence across all queries:
  `bm25_injection_enabled_queries`, `bm25_injection_candidates`,
  `bm25_injection_retained`, and `bm25_injection_exact_reranked`
- a tiny local 200-doc/5-query smoke against the precomputed DBpedia dataset
  validated the plumbing:
  `proxy_normalized_mean_f16_bm25_rescue`, EF=50, budget=50, injected BM25
  on 2/5 queries; candidates summary max=18, mean=4.4; retained summary
  max=11, mean=2.8
- a focused 10k run with the rescue profile accidentally set to
  `multivector_branch_plan = qdrant_like` improved admission but was not a
  valid serving result: p95 ~= 446 ms, top10 admission ~= 0.548, recall@10
  ~= 0.464, ndcg@10 ~= 0.398, and stats showed
  `proxy_vector_near_exhaustive_sidecar_touch = true` with resident-cache
  materialization of all 10k docs. This proves BM25 rescue can improve
  admission, but also proves the profile must remain `dense_only` unless the
  qdrant-like sidecar rescore path is separately fixed.
- focused 10k A/B validation with `branch_plan = dense_only`, 25 queries,
  EF=800, candidate_k=800, exact_rerank_k=100:
  - baseline `proxy_normalized_mean_f16`: p50 ~= 18.6 ms, p95 ~= 21.1 ms,
    top1 admission ~= 0.20, top10 admission ~= 0.14, recall@10 ~= 0.151,
    ndcg@10 ~= 0.143, mrr@10 ~= 0.361
  - `proxy_normalized_mean_f16_bm25_rescue`: p50 ~= 18.2 ms,
    p95 ~= 22.2 ms, top1 admission ~= 0.44, top10 admission ~= 0.34,
    recall@10 ~= 0.351, ndcg@10 ~= 0.271, mrr@10 ~= 0.477
  - BM25 injection fired on 11/25 queries; candidates max=21, mean=4.52;
    retained max=16, mean=3.0
  - both runs stayed on the bounded serving path:
    `proxy_vector_near_exhaustive_sidecar_touch = false`,
    sidecar vectors loaded per query = 100, bytes touched ~= 3.65 MB,
    pages read ~= 947, exact kernel = `blocked_neon`
- this is still below the 0.80 top10 admission safety threshold, but it is a
  useful, bounded candidate-quality improvement and should be compared against
  centroid_mean/centroid_lite and learned-sparse rescue next
- the serving grid now also has an opt-in learned-sparse rescue switch,
  `--document-node-serving-grid-include-learned-sparse-rescue`
- the opt-in learned-sparse profiles are:
  `proxy_normalized_mean_f16_learned_sparse_rescue` and
  `centroid_mean_f16_learned_sparse_rescue`
- when learned-sparse rescue and proxy-encoder variants are both enabled, the
  harness also exposes `proxy_max_pool_f16_learned_sparse_rescue` for the same
  focused proxy-quality comparison
- these profiles use `multivector_branch_plan = dense_only`,
  `multivector_sparse_candidate_source = learned_sparse`, and the existing
  learned-sparse `text_query` conversion, so learned-sparse postings are used
  only as candidate admission before exact MaxSim final ranking
- the flag is intentionally separate from BM25 rescue and requires
  `--learned-sparse-doc-jsonl` plus `--learned-sparse-query-jsonl` for real
  runs
- a tiny local smoke with existing `.nix-dev/tmp` learned-sparse JSONL fixtures
  validated the profile and row reporting: 50 docs, 2 queries,
  `proxy_normalized_mean_f16_learned_sparse_rescue`, EF=50, budget=50,
  p50 ~= 13.7 ms, p95 ~= 14.2 ms, top10 admission ~= 0.90; row summaries
  preserved learned-sparse candidates max=31/mean=18 and retained
  max=31/mean=18; `proxy_vector_near_exhaustive_sidecar_touch = false`
- the benchmark index signature now records the concrete lexical index column,
  so BM25 rescue and learned-sparse rescue cannot accidentally share a physical
  index. A mixed 50-doc/2-query smoke confirmed two index groups:
  BM25 uses `body_tsv`, learned-sparse uses `learned_sparse_tsv`. This smoke is
  only a plumbing check; because the table is tiny, near-table-size slow-path
  warnings are expected and are not serving evidence.
- learned-sparse JSONL import now records loaded document/query coverage,
  coverage ratios, partial-coverage warnings, and Markdown reporting. This
  prevents the local partial fixture from being mistaken for production serving
  evidence while keeping the opt-in rescue path runnable for plumbing checks.
- BM25/learned-sparse rescue stats now expose the effective rescue candidate
  limit, combined pool size, limit reason, retained count, and exact-reranked
  rescue count. This is accounting only; it does not change final exact MaxSim
  ordering or candidate admission behavior.
- serving-grid recommendation rows now preserve those aggregate rescue fields
  and Markdown "why this profile won" text reports them. This prevents
  best/rejected profile explanations from depending only on one sampled scan.
- rejected serving-grid recommendation rows now include
  `admission_improvement_hints`, including EF caps, document-candidate caps,
  rerank caps, BM25 lexical underfill, learned-sparse fixture gaps, and generic
  candidate-source quality bottlenecks. This is recommendation/reporting only.
- when a row exhausts the admitted candidate band but still misses the top-10
  admission threshold, those hints now identify the next focused experiment:
  `try_max_pool_or_centroid_mean_proxy`, `try_centroid_mean_proxy`,
  `try_sparse_rescue_or_centroid_lite`,
  `try_balanced_candidate_reservoirs`, or
  `try_bm25_or_learned_sparse_rescue`
- centroid_lite now has an opt-in bounded posting-list experiment knob:
  `turbohybrid.multivector_centroid_lite_max_postings_per_token` and benchmark
  flag `--multivector-centroid-lite-max-postings-per-token`. The default `0`
  preserves current full posting-list admission. Positive values cap each
  query-token centroid posting list before the final exact MaxSim rerank using
  deterministic uniform-stride sampling across the posting list, not first-N
  truncation. Scans report `centroid_postings_touched`,
  `centroid_postings_skipped`, `centroid_posting_limit_per_token`, and
  `centroid_posting_cap_strategy`. This is a measurement/quality tradeoff knob
  for the still-experimental `centroid_lite` path, not a serving default.
- the document-node serving grid now has an opt-in capped centroid-lite profile
  sweep via `--document-node-serving-grid-include-centroid-lite-caps` and
  `--document-node-serving-grid-centroid-lite-posting-caps`. It adds named rows
  such as `centroid_lite_f16_cap_016` while reusing the same physical kmeans
  centroid index as uncapped `centroid_lite_f16`; this is benchmark evidence
  plumbing only, not a default serving-profile change.
- centroid_lite candidate-source scans no longer pre-validate every document's
  vector and centroid on every query. They validate centroid sidecar entries
  lazily for documents reached through posting lists, so capped posting-list
  experiments measure the bounded posting work more directly. This does not
  change final exact MaxSim rerank semantics or persisted formats.
- `quantized_inverted_experimental` scans no longer pre-validate every document
  vector on every query. They validate document vectors lazily for touched
  postings before exact heap MaxSim rerank, keeping the branch explicitly
  experimental while making its candidate-source timing less polluted by
  whole-sidecar validation.
- the document-node serving grid now has an opt-in graph entry sidecar profile
  sweep via `--document-node-serving-grid-include-entry-sidecar`. It adds
  `proxy_normalized_mean_f16_entry_sidecar` and
  `centroid_mean_f16_entry_sidecar`, builds physical indexes with
  `entry_sidecar = on`, 128 `hybrid_level_covering` representatives by default,
  and reports the entry-sidecar reloptions plus `graph_entry_sidecar_*` scan
  counters in JSON and Markdown. This is candidate-admission evidence only; it
  does not change default serving profiles or final exact MaxSim ordering.
- when entry-sidecar experiments and proxy-encoder variants are both enabled,
  the harness also exposes `proxy_max_pool_f16_entry_sidecar` so the stronger
  `max_pool` proxy baseline can be tested with the same graph-entry
  representative sidecar without changing the default compact serving grid.
- a 200-document local DBpedia build-only smoke passed for the entry-sidecar
  physical-index variants using
  `.nix-dev/hf-datasets/dbpedia-colbert-multivector-1m-f16` as the local
  precomputed dataset. It built
  `proxy_normalized_mean_f16_entry_sidecar`,
  `proxy_max_pool_f16_entry_sidecar`, and
  `centroid_mean_f16_entry_sidecar` with three separate physical index
  signatures. Each index reported `entry_sidecar = on`, 128
  `hybrid_level_covering` representatives, and `entry_sidecar_count = 128`.
  Build-only timings on the tiny smoke were approximately 666 ms for
  `proxy_normalized_mean_f16_entry_sidecar`, 453 ms for
  `proxy_max_pool_f16_entry_sidecar`, and 3830 ms for
  `centroid_mean_f16_entry_sidecar`; the centroid row was dominated by
  centroid construction. This validates the entry-sidecar build/reporting
  plumbing only; it is not retrieval/admission evidence.
- the next entry-sidecar step should be a narrow retrieval/admission smoke that
  compares plain vs entry-sidecar variants for normalized_mean, max_pool, and
  centroid_mean at a small doc count first, then 10k only if the small run shows
  the graph-entry sidecar changes candidate admission. Do not widen this into
  the full serving grid until there is evidence that entry selection is the
  bottleneck being improved.
- the first narrow retrieval/admission smoke was run on 200 documents and 5
  queries with EF=50, candidate budget=50, and exact rerank K=50. It compared
  plain and entry-sidecar variants for normalized_mean, max_pool, and
  centroid_mean. Entry-sidecar scans reported the representative sidecar path
  (`graph_entry_sidecar_count = 128`, about 115 representatives scored, and
  7-10 selected depending on profile), but top-10 admission did not change in
  this tiny run:
  - normalized_mean: 0.42 plain and 0.42 entry-sidecar
  - max_pool: 0.66 plain and 0.66 entry-sidecar
  - centroid_mean: 0.54 plain and 0.54 entry-sidecar
  `proxy_max_pool_f16_entry_sidecar` was the best quality/balanced row in this
  small smoke, with p95 around 9.8 ms, top-1 admission 0.80, top-10 admission
  0.66, and NDCG@10 around 0.665. No row met the 0.80 top-10 admission
  threshold. Treat this as a plumbing and small-corpus candidate-signal check,
  not x00k serving evidence.
- the serving-grid report now includes `candidate_source_deltas` in JSON and a
  Markdown table. It compares known paired variants (entry-sidecar, BM25 rescue,
  learned-sparse rescue, centroid-lite caps, document-node/centroid-lite source
  variants, and proxy-encoder variants) against the matching plain baseline at
  the same EF, oversampling, and executed candidate budget. This is
  recommendation/reporting only and does not change SQL/index/search behavior.
- the serving-grid recommendation now also includes
  `candidate_source_delta_summary`, which surfaces the best admission, quality,
  and latency deltas plus the best admission delta per comparison family. This
  keeps the next focused candidate-source experiment visible even when all rows
  still fail the serving safety threshold; it is reporting-only and does not
  change SQL/index/search behavior.
- document-node proxy graph entry sampling is now tunable for scan-time
  admission diagnostics with
  `turbohybrid.multivector_doc_graph_entry_sample_count` and the benchmark flag
  `--multivector-doc-graph-entry-sample-count`. Default `0` preserves the
  compiled sampler. Positive values score that many deterministic document
  proxy entry seeds, bounded by document count, and expose
  `graph_entry_sample_*` plus `multivector_doc_graph_entry_sample_*` scan stats.
  This is not a persisted-format change and does not change final exact MaxSim
  ranking.
- Regression coverage now asserts both the default entry-sampling stats and a
  positive `multivector_doc_graph_entry_sample_count = 2` scan on a
  deterministic document-node proxy index. This proves the scan-time diagnostic
  knob is visible to SQL and reflected in `turbohybrid_last_scan_stats()`.
- The document-node serving grid now has an opt-in scan-time entry-sample
  profile sweep via `--document-node-serving-grid-include-entry-samples` and
  `--document-node-serving-grid-entry-sample-counts` (default `32,128`). It
  adds paired rows such as `proxy_normalized_mean_f16_entry_sample_032` and
  `centroid_mean_f16_entry_sample_032`, reuses the same physical index as the
  baseline, reports the effective entry-sample count in JSON/Markdown and
  recommendation rows, and includes an `entry_sample` family in
  `candidate_source_deltas`. This is reporting/evidence plumbing for candidate
  admission quality only; it does not change SQL/index/search behavior,
  persisted format, or final exact MaxSim ordering.
- The document-node serving grid now has an opt-in bounded reservoir profile
  sweep via `--document-node-serving-grid-include-reservoirs`, and explicit
  document-node `proxy_vector` scans with
  `turbohybrid.multivector_candidate_reservoirs = balanced` now execute a
  proxy-rank plus deterministic rank-spread selector for the bounded exact
  rerank band. This does not change the default path, persisted format, or final
  exact MaxSim ordering. The row is still treated as valid evidence only when
  scan stats report `multivector_reservoirs_enabled = true` and nonzero
  reservoir union docs; otherwise it remains rejected as
  `candidate_reservoirs_not_executed`.
- Rejected serving-grid recommendation hints now know about the reservoir
  profiles directly. Candidate-source bottleneck rows that have not tried
  reservoirs now suggest `try_balanced_candidate_reservoirs`; rows that already
  tried balanced reservoirs can report
  `candidate_reservoirs_no_union`,
  `candidate_reservoirs_no_extra_docs`, or
  `candidate_reservoirs_high_duplicates`. The docs and benchmark README now
  list candidate reservoirs as a first-class paired-delta family. This is
  reporting-only and does not change SQL/index/search behavior.
- The paired candidate-source delta report now gates reservoir deltas on the
  same execution evidence. Rows with requested reservoirs but no
  `multivector_reservoirs_enabled` evidence remain visible with
  `candidate_reservoirs_not_executed`, but they are excluded from
  `candidate_source_delta_summary` best-improvement picks. This prevents a
  document-node proxy reservoir no-op from looking like the next candidate
  quality winner.
- Candidate-source delta rows now preserve compact experiment-specific evidence
  details, including entry-sample work, entry-sidecar representative selection,
  reservoir union/duplicate summaries, rescue candidate counts, and
  centroid-lite posting caps. This is JSON/Markdown reporting only and does not
  change SQL/index/search behavior.
- Explicit document-node `proxy_vector` scans now execute both SQL-visible
  reservoir modes. `conservative` keeps most of the proxy-ranked prefix with a
  small deterministic rank-spread sample, while `balanced` uses a larger
  rank-spread share. The serving-grid opt-in reservoir profiles still use
  `balanced`; this fix prevents `conservative` from being an advertised but
  silent no-op. Regression coverage asserts both modes expose reservoir stats
  and keep final ranking in the exact MaxSim rerank over retained documents.
