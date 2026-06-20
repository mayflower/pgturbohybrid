Below is a **copy-paste prompt pack** ordered for implementation. It keeps the current constraint: **no BM25 rescue, no learned-sparse rescue, final SQL ordering remains exact MaxSim**.

Run these in order.

---

# Prompt 0 — Baseline audit for quantized-inverted hot path

```text
You are working in github.com/mayflower/pgturbohybrid.

Goal:
Audit the current quantized_inverted_experimental implementation before changing behavior.

Background:
The current best default-quality row is:

  quantized_inverted_external_centroid_only_compact_topk_128_probe_016_topm_01_score_bound
  budget 8192

Observed quality:
- top1 admission: 0.96
- top10 admission: 0.832
- recall@10: 0.539333
- ndcg@10: 0.458704
- p95 latency: 1914.798 ms

Observed p95 timing:
- candidate source / quantized inverted path: ~1503 ms
- compact scoring: ~1063 ms
- query-codeword scoring: ~329 ms
- exact MaxSim rerank: ~398 ms
- heap fetch: ~183 ms
- final sort: ~12 ms
- sidecar I/O: effectively 0

Observed work:
- postings touched: ~60k
- docs compact-scored: ~23.4k
- compact payload touched: ~6.8 MB
- exact rerank docs: 512
- score-bound pruning prunes only about 24 / 23k docs

Repository facts to verify:
- quantized_inverted_experimental is research-only and final ranking remains exact heap MaxSim.
- Current compact scoring uses temporary scorePayload-like compact payloads and is admission-only.
- Current stats already include postings touched/selected/skipped, docs scored, compact score time, query-codeword time, token coverage, and score-bound pruning counters.
- top_m > 1 is rejected unless there is an explicitly versioned posting format.

Read:
- docs/dev/multivector-colbertsar-research.md
- docs/dev/multivector-document-graph-design.md
- src/pgturbohybrid_am.h
- src/pgturbohybrid_am.c
- src/pgturbohybrid_stats.c
- src/pgturbohybrid_quant.h
- src/pgturbohybrid_quant_scan_cache.c
- test/sql/pgturbohybrid_multivector.sql
- test/expected/pgturbohybrid_multivector.out
- benchmarks/config/result_schema.json
- benchmarks/README.md

Do not change files.

Deliverable:
1. A short implementation map:
   - where quantized inverted postings are scanned
   - where query-codeword scores are computed
   - where document compact scores are computed
   - where exact MaxSim rerank is invoked
   - where score-bound pruning currently runs
   - where token coverage filtering currently runs
   - where last_scan_stats are populated

2. A bottleneck table:
   - phase
   - current stat fields
   - likely time cost
   - memory/work driver
   - whether it should be optimized in Prompt 1, 2, or 3

3. Confirm whether the following fields already exist in last_scan_stats and benchmark JSON:
   - quantized_inverted_postings_touched
   - quantized_inverted_postings_selected
   - quantized_inverted_postings_skipped
   - quantized_inverted_docs_scored
   - quantized_inverted_candidates
   - quantized_inverted_query_codeword_score_us
   - quantized_inverted_compact_score_us
   - quantized_inverted_compact_docs_scored
   - quantized_inverted_compact_payload_bytes
   - quantized_inverted_token_coverage_mode
   - quantized_inverted_active_query_tokens
   - quantized_inverted_token_matches_total
   - quantized_inverted_min_token_matches
   - quantized_inverted_token_match_filtered_docs
   - quantized_inverted_score_bound_pruning_enabled
   - quantized_inverted_score_bound_docs_checked
   - quantized_inverted_score_bound_docs_pruned
   - quantized_inverted_candidates_before_bound
   - quantized_inverted_candidates_after_bound

4. Recommend exact file/function insertion points for Prompt 1.

Constraints:
- No BM25 rescue.
- No learned-sparse rescue.
- Do not change SQL behavior.
- Do not change on-disk format.
- Do not modify benchmark outputs.
```

---

# Prompt 1 — Implement precompact document gate

```text
You are working in github.com/mayflower/pgturbohybrid.

Goal:
Reduce quantized_inverted_experimental latency by adding a PLAID/NextPlaid-style precompact document gate before compact code scoring.

Core idea:
Keep broad probe/cap settings for quality, but do not compact-score every touched document. First rank touched documents with a cheap centroid/codeword-level MaxSim approximation, retain a bounded document set, then run the existing compact scorer only on that set.

Background:
The current bottleneck is:
- ~23.4k docs compact-scored per query
- ~1063 ms compact scoring at p95
- score-bound pruning only prunes about 24 docs, so it is ineffective
- sidecar I/O is effectively zero, so this is a CPU/candidate-volume problem

Constraints:
- Do not use BM25 rescue.
- Do not use learned-sparse rescue.
- Keep final SQL ordering exact MaxSim.
- Keep quantized_inverted_experimental opt-in.
- Do not change stable on-disk formats.
- Do not silently substitute another candidate source.
- Do not change top_m > 1 behavior.
- Approximate scores remain admission-only.

Add GUC:
  turbohybrid.multivector_quantized_inverted_precompact =
    off | centroid_maxsim_topk | centroid_maxsim_reservoir

Default:
  off

Add GUCs:
  turbohybrid.multivector_quantized_inverted_precompact_score_k = 4096
  turbohybrid.multivector_quantized_inverted_precompact_coverage_k = 512
  turbohybrid.multivector_quantized_inverted_precompact_per_token_k = 16
  turbohybrid.multivector_quantized_inverted_compact_max_docs = 6144

Behavior:

1. off
   - Preserve current behavior exactly.
   - All currently compact-scored docs remain compact-scored.

2. centroid_maxsim_topk
   - During posting accumulation, maintain a cheap per-document score:
       cheap_score(doc) = sum over query token i of best query_codeword_score(i, matched_codeword)
   - Use existing query-token x codeword scores; do not decode full document vectors.
   - Retain only top precompact_score_k documents for compact scoring.
   - If precompact_score_k <= 0, treat as no limit.
   - Exact MaxSim final rerank remains unchanged.

3. centroid_maxsim_reservoir
   - Retain a document-keyed union of:
       a. top precompact_score_k by cheap_score
       b. top precompact_coverage_k by token coverage / token match count
       c. per-query-token reservoirs of width precompact_per_token_k
   - Deduplicate by docId.
   - Clamp final precompact union to compact_max_docs if compact_max_docs > 0.
   - Prefer score first, then coverage, then deterministic docId order for ties.
   - This is the intended benchmark mode.

Implementation requirements:
- Avoid per-document heap fetches in the precompact stage.
- Use PostgreSQL memory contexts and overflow-checked allocation sizes.
- Do not allocate O(total_docs * query_tokens) unless already bounded and justified.
- Prefer sparse/touched-doc accumulators.
- Keep deterministic ordering for tests.
- Do not regress current off-mode stats or results.

Add stats:
  quantized_inverted_precompact_enabled
  quantized_inverted_precompact_mode
  quantized_inverted_docs_touched_before_precompact
  quantized_inverted_precompact_score_k
  quantized_inverted_precompact_coverage_k
  quantized_inverted_precompact_per_token_k
  quantized_inverted_compact_max_docs
  quantized_inverted_precompact_score_docs
  quantized_inverted_precompact_coverage_docs
  quantized_inverted_precompact_per_token_docs
  quantized_inverted_precompact_union_docs
  quantized_inverted_precompact_duplicates
  quantized_inverted_precompact_pruned_docs
  quantized_inverted_precompact_us
  quantized_inverted_compact_docs_skipped_by_precompact

Stats invariants:
- compact_docs_skipped_by_precompact =
    docs_touched_before_precompact - compact_docs_scored
  when precompact is enabled and positive.
- precompact_pruned_docs must never be negative.
- off mode must report precompact_enabled = false and zero precompact pruning.

Tests:
1. SQL/C regression:
   - off mode produces the same result order and same compact_docs_scored as before.
   - centroid_maxsim_topk limits compact scoring on a synthetic dataset.
   - centroid_maxsim_reservoir includes score, coverage, and per-token candidates.
   - exhaustive precompact budget matches off mode.
   - final SQL ordering remains exact MaxSim after rerank.

2. Determinism:
   - Tie cases sort deterministically by score, coverage, docId.

3. Safety:
   - Invalid GUC values fail or clamp clearly.
   - No silent fallback to exact_doc_scan, proxy_vector, BM25, or learned_sparse.
   - top_m > 1 behavior is unchanged.

Validation:
Run:
  nix --extra-experimental-features 'nix-command flakes' develop --command th-installcheck
  nix --extra-experimental-features 'nix-command flakes' develop --command th-smoke

Do not commit generated benchmark output.
```

---

# Prompt 2 — Benchmark harness and recommendation gate for precompact

```text
You are working in github.com/mayflower/pgturbohybrid.

Goal:
Add a focused benchmark grid for the new quantized_inverted precompact modes and make the recommendation logic choose only quality-safe latency improvements.

Precondition:
Prompt 1 implemented:
- turbohybrid.multivector_quantized_inverted_precompact
- precompact stats
- SQL regression tests

Background baseline:
Current best default-quality row:
  quantized_inverted_external_centroid_only_compact_topk_128_probe_016_topm_01_score_bound
  budget 8192

Baseline quality:
- top1 admission: 0.96
- top10 admission: 0.832
- recall@10: 0.539333
- ndcg@10: 0.458704
- p95 latency: 1914.798 ms

Target:
Find a row that preserves the quality gate while reducing compact_docs_scored from ~23.4k to <= 6k and p95 latency by at least 25%.

Constraints:
- Do not use BM25 rescue.
- Do not use learned-sparse rescue.
- Keep final SQL ordering exact MaxSim.
- Keep generated outputs under .nix-dev/tmp/.
- Do not change C behavior in this prompt unless needed only for stats plumbing.

Add benchmark mode:
  --document-node-colbert-quantized-inverted-precompact-focus

Default fixed settings:
- candidate source: quantized_inverted_experimental
- profile base: quantized_inverted_external_centroid_only_compact_topk_128_probe_016_topm_01_score_bound
- budget: 8192
- exact_rerank_k: 512
- posting cap: 128
- probe codewords per token: 16
- top_m: 1
- compact scoring: experimental
- score bound: enabled if currently used by the baseline
- codebook path: use --multivector-quantized-inverted-codebook-path

Rows:
1. precompact_off
2. precompact_topk_2048
3. precompact_topk_4096
4. precompact_topk_6144
5. precompact_topk_8192
6. precompact_reservoir_2048
7. precompact_reservoir_4096
8. precompact_reservoir_6144
9. precompact_reservoir_8192

For reservoir rows:
- coverage_k = 512
- per_token_k = 16
- compact_max_docs = max(score_k + coverage_k + active_query_tokens * per_token_k, score_k)
- allow CLI overrides:
    --quantized-inverted-precompact-score-grid
    --quantized-inverted-precompact-coverage-k
    --quantized-inverted-precompact-per-token-k
    --quantized-inverted-compact-max-docs

Report per row:
- profile
- precompact mode
- precompact score_k
- precompact coverage_k
- precompact per_token_k
- compact_max_docs
- top1 admission
- top10 admission
- recall@10
- ndcg@10
- mrr@10
- p50 / p95 / p99
- quantized_inverted_postings_touched
- quantized_inverted_postings_selected
- quantized_inverted_docs_touched_before_precompact
- quantized_inverted_precompact_union_docs
- quantized_inverted_precompact_pruned_docs
- quantized_inverted_compact_docs_scored
- quantized_inverted_compact_docs_skipped_by_precompact
- quantized_inverted_query_codeword_score_us
- quantized_inverted_precompact_us
- quantized_inverted_compact_score_us
- exact_rerank_docs
- exact_rerank_us
- heap_fetch_us
- final_sort_us
- sidecar page/read stats

Recommendation gates:
A row may be promoted as best_quantized_inverted_precompact only if:
- top10 admission >= 0.80
- top1 admission >= 0.94
- recall@10 >= baseline_recall@10 - 0.01
- ndcg@10 >= baseline_ndcg@10 - 0.01
- p95 <= baseline_p95 * 0.75
- compact_docs_scored <= 6000
- final ranking source is exact MaxSim
- candidate source is quantized_inverted_experimental
- BM25 and learned_sparse are not active

Reject reasons:
- rejected_top10_admission_below_gate
- rejected_top1_admission_below_gate
- rejected_recall_drop
- rejected_ndcg_drop
- rejected_latency_not_improved
- rejected_compact_docs_too_high
- rejected_wrong_candidate_source
- rejected_sparse_or_bm25_rescue_active
- rejected_final_ranking_not_exact_maxsim

Markdown:
Add a section:
  Quantized inverted precompact focus

Include:
- baseline row
- best accepted row
- rejected rows table
- latency breakdown table
- compact-doc reduction table
- recommendation and next action

Tests:
- profile expansion deterministic
- CLI grid parsing deterministic
- recommendation gate rejects low-admission fast rows
- recommendation gate rejects rows with compact_docs_scored > 6000
- recommendation gate accepts a synthetic quality-safe faster row
- JSON/Markdown contain all new precompact fields
- qrels-less mode can still rank by admission + p95 if admission metrics are present

Validation:
Run:
  nix --extra-experimental-features 'nix-command flakes' develop --command \
    python benchmarks/dbpedia_colbert_multivector.py --help

Run Python/self-check tests.

Do not commit generated outputs.
```

---

# Prompt 3 — Query-codeword scoring kernel and top-probe selection

```text
You are working in github.com/mayflower/pgturbohybrid.

Goal:
Reduce quantized_inverted query-codeword scoring time and avoid unnecessary full query-token × codebook materialization.

Background:
Current timing shows query-codeword scoring at about 329 ms p95, which is a major part of the quantized inverted candidate-source cost. The codebook routing stage should be cheap and batched.

Precondition:
Prompt 1 and Prompt 2 are either implemented or their insertion points are known.

Constraints:
- Do not use BM25 rescue.
- Do not use learned-sparse rescue.
- Keep final SQL ordering exact MaxSim.
- Keep quantized_inverted_experimental opt-in.
- Do not change stable on-disk formats.
- Do not change codebook checksum semantics.
- top_m > 1 behavior remains unchanged unless a separate prompt explicitly versions the posting format.

Tasks:
1. Audit current query-codeword scoring:
   - how external codebook vectors are loaded
   - whether they are normalized
   - whether they are row-major or transposed
   - whether full Q × C scores are materialized
   - how top probe codewords per query token are selected
   - whether masked/zero-weight query tokens are skipped early

2. Add implementation mode:
   turbohybrid.multivector_quantized_inverted_query_codeword_kernel =
     auto | scalar | blocked

Default:
   auto

3. Implement blocked scalar first:
   - aligned codebook access
   - batch query tokens against codebook blocks
   - maintain top probe codewords per query token while scanning
   - avoid retaining full Q × C matrix unless compact/precompact scoring requires it
   - if compact/precompact scoring requires query-token/codeword lookup later, store only the needed codeword scores for probed/touched codewords, or use a compact score table keyed by codeword.

4. Optional SIMD:
   Add AVX2/AVX512/NEON only if the scalar blocked reference is correct and tested.
   It is acceptable for this prompt to stop after blocked scalar if that is the safe slice.

5. Add stats:
   - quantized_inverted_query_codeword_kernel
   - quantized_inverted_query_codeword_scores_computed
   - quantized_inverted_query_codeword_blocks
   - quantized_inverted_query_codeword_topk_us
   - quantized_inverted_query_codeword_full_matrix_materialized
   - quantized_inverted_query_codeword_active_query_tokens
   - quantized_inverted_query_codeword_skipped_query_tokens

6. Preserve existing stat:
   - quantized_inverted_query_codeword_score_us

7. Tests:
   - blocked scalar returns same top probe codewords as existing scalar on deterministic vectors
   - masked query tokens are skipped
   - zero-weight query tokens are skipped if existing scoring semantics allow it
   - auto selects blocked scalar or existing scalar deterministically
   - full-matrix and streaming-topk modes produce identical selected codewords
   - final SQL ordering remains exact MaxSim

8. Benchmark:
   Extend precompact focus rows with:
   - query_codeword_kernel = scalar
   - query_codeword_kernel = blocked
   only for the best two precompact rows from Prompt 2.

Acceptance:
- Query-codeword scoring p95 improves by at least 30% on the focused grid, or the report explains why not.
- Admission metrics must not change except for deterministic tie effects.
- Final exact ranking must not change for exhaustive candidate tests.

Validation:
Run th-installcheck and th-smoke.
Run focused DBpedia precompact benchmark.
Do not commit generated outputs.
```

---

# Prompt 4 — Compact scoring layout audit and docId-ordered scorer

```text
You are working in github.com/mayflower/pgturbohybrid.

Goal:
Reduce per-document compact scoring overhead after precompact has reduced the number of compact-scored documents.

Background:
Qdrant-style multivector scoring is document/point-level: docId maps to a start/count range in flattened inner-vector storage, and the scorer scores the whole point. pgturbohybrid’s own design doc describes this shape for quantized multivector scoring: per-point offsets into flattened storage and score_point-like whole-document scoring.

Current bottleneck before Prompt 1:
- compact scoring ~1063 ms p95
- compact docs scored ~23.4k
- compact payload touched ~6.8 MB

After Prompt 1:
Expected compact docs scored should fall to ~2k–6k. This prompt optimizes per-doc compact scoring cost.

Constraints:
- Do not use BM25 rescue.
- Do not use learned-sparse rescue.
- Keep final SQL ordering exact MaxSim.
- Keep quantized_inverted_experimental opt-in.
- Do not change stable on-disk format in this prompt.
- If a better persisted layout is needed, produce a design doc and stop before changing format.

Tasks:
1. Audit compact scoring data access:
   - docId to compact payload lookup
   - token/codeword payload layout
   - whether compact docs are scored in arbitrary order
   - per-doc allocation or hash lookup inside scoring
   - query-codeword score lookup pattern
   - branch-heavy inner loops
   - prefetch opportunities

2. Implement scan-local docId ordering:
   - after precompact selects docs, sort compact-scoring docIds ascending
   - score docs in docId order to improve locality
   - preserve final ranking by sorting candidates after scoring
   - expose whether docId ordering changed compact score order

3. Implement no-allocation compact scorer path:
   - no palloc per doc
   - reuse one query-token maxima buffer
   - clear only active query-token slots
   - skip masked query tokens
   - use compact score table from Prompt 3 if available

4. Add stats:
   - quantized_inverted_compact_doc_order = original | docid
   - quantized_inverted_compact_inner_allocations
   - quantized_inverted_compact_active_query_tokens
   - quantized_inverted_compact_pairs_evaluated
   - quantized_inverted_compact_pairs_skipped
   - quantized_inverted_compact_prefetches
   - quantized_inverted_compact_avg_doc_tokens
   - quantized_inverted_compact_us_per_doc
   - quantized_inverted_compact_payload_bytes_per_doc

5. Tests:
   - docId-ordered compact scoring returns same top candidates as original order
   - no-allocation path matches scalar reference
   - masked query tokens are skipped
   - final SQL ordering remains exact MaxSim
   - compact_topk_changed_vs_scalar remains false or is reported accurately

6. Benchmark:
   Run only on:
   - best precompact row from Prompt 2
   - second-best precompact row from Prompt 2
   Compare:
   - compact_doc_order original
   - compact_doc_order docid
   - no-allocation on/off if exposed

Acceptance:
- Compact score p95 improves by at least 15% on the best precompact row, or stats show compact scoring is no longer a top-two bottleneck.
- Admission and qrels metrics do not regress beyond deterministic tie noise.
- Final exact ranking remains unchanged for exhaustive tests.

Validation:
Run th-installcheck and th-smoke.
Run focused DBpedia benchmark.
Do not commit generated outputs.
```

---

# Prompt 5 — Exact rerank secondary optimization after candidate-source fix

```text
You are working in github.com/mayflower/pgturbohybrid.

Goal:
Optimize exact MaxSim rerank only after quantized inverted candidate-source work is reduced.

Background:
Current exact rerank is material but secondary:
- exact MaxSim rerank ~398 ms p95
- heap fetch ~183 ms
- exact rerank docs: 512

The primary bottleneck is compact scoring. This prompt should only run after precompact reduces compact scoring enough that exact rerank becomes top-two latency again.

Constraints:
- Do not use BM25 rescue.
- Do not use learned-sparse rescue.
- Final SQL ordering must remain exact MaxSim over retained candidates.
- Do not lower exact_rerank_k as a hidden quality change.
- Any adaptive rerank must preserve final top-k equivalence against full exact rerank for tested cases.

Tasks:
1. Add exact rerank phase breakdown if missing:
   - heap fetch us
   - multivector decode us
   - MaxSim compute us
   - adaptive pruning us
   - tokens evaluated
   - tokens skipped
   - pairs saved
   - exact kernel
   - docs reranked

2. Add focused exact rerank grid:
   - exact_rerank_k 256,384,512
   - adaptive exact rerank off/on
   - only for best precompact profile
   - no BM25/learned sparse rows

3. Recommendation gate:
   Do not promote lower exact_rerank_k unless:
   - top10 admission remains >= 0.80
   - recall@10 and ndcg@10 remain within 0.005 of the 512 baseline
   - exact top1 admission remains >= 0.94
   - final ranking is exact MaxSim for retained docs

4. Tests:
   - adaptive rerank equals full exact rerank on deterministic synthetic cases
   - lower exact_rerank_k is reported as quality-risk if it changes top-k
   - final SQL ordering remains exact MaxSim within retained docs
   - stats distinguish heap fetch vs MaxSim compute

Validation:
Run th-installcheck and th-smoke.
Run focused DBpedia exact-rerank grid.
```

---

# Prompt 6 — End-to-end acceptance and default-quality profile update

```text
You are working in github.com/mayflower/pgturbohybrid.

Goal:
Choose the new best quantized_inverted default-quality experimental profile only after evidence from Prompts 1–5.

Required input:
  PRECOMPACT_GRID_JSON=.nix-dev/tmp/<precompact-grid>.json

Optional input:
  QUERY_CODEWORD_GRID_JSON=.nix-dev/tmp/<query-codeword-grid>.json
  COMPACT_LAYOUT_GRID_JSON=.nix-dev/tmp/<compact-layout-grid>.json
  EXACT_RERANK_GRID_JSON=.nix-dev/tmp/<exact-rerank-grid>.json

Constraints:
- Do not use BM25 rescue.
- Do not use learned-sparse rescue.
- Keep quantized_inverted_experimental opt-in.
- Do not make experimental branch a production default.
- Do not change final SQL ordering semantics.
- Do not change on-disk formats.
- Do not commit generated outputs.

Tasks:
1. Load benchmark JSONs and identify:
   - current baseline row
   - best precompact row
   - best query-codeword kernel row
   - best compact-layout row
   - best exact-rerank row, if safe

2. Apply hard gates:
   - candidate_source == quantized_inverted_experimental
   - final ranking exact MaxSim
   - BM25 inactive
   - learned_sparse inactive
   - top10 admission >= 0.80
   - top1 admission >= 0.94
   - recall@10 >= baseline - 0.01
   - ndcg@10 >= baseline - 0.01
   - p95 <= baseline * 0.75
   - compact_docs_scored <= 6000, unless compact scoring is no longer top-two bottleneck and the report explains why
   - generated output path under .nix-dev/tmp/

3. Update benchmark profile naming only if gates pass:
   New suggested profile name:
     quantized_inverted_external_centroid_only_precompact_reservoir_4096_probe_016_cap_128_topm_01

   Adjust score_k in name if a different value wins:
     precompact_reservoir_2048
     precompact_reservoir_6144
     precompact_reservoir_8192

4. Do not remove old profiles.
   Mark old default-quality row as:
     previous_default_quality
   Mark new row as:
     experimental_default_quality_candidate

5. Update docs:
   - benchmarks/README.md
   - docs/dev/multivector-colbertsar-research.md

   Document:
   - why precompact exists
   - why score-bound alone was not enough
   - which stats prove the bottleneck moved
   - exact MaxSim final ranking contract
   - no BM25/learned-sparse rescue used

6. Add recommendation summary:
   - before p95
   - after p95
   - compact docs before/after
   - compact score us before/after
   - query-codeword us before/after
   - exact rerank us before/after
   - quality before/after
   - rejection reasons for unsafe faster rows

7. Tests:
   - profile registry includes new profile only when explicitly selected
   - old profile remains available
   - recommendation refuses to promote rows that violate gates
   - docs mention experimental status
   - JSON/Markdown generated under .nix-dev/tmp/

Validation:
Run:
  nix --extra-experimental-features 'nix-command flakes' develop --command \
    python benchmarks/dbpedia_colbert_multivector.py --help

Run Python/self-check tests.
Run th-installcheck if profile/GUC behavior changed.

Do not commit generated benchmark outputs.
```

---

# Prompt 7 — Optional design-only prompt for future persisted layout

```text
You are working in github.com/mayflower/pgturbohybrid.

Goal:
Create a design doc for a future versioned persisted compact multivector layout, but do not implement it.

Background:
If precompact and scan-local compact scoring still leave compact scoring as the bottleneck, the next step may require a Qdrant-like flattened point layout:
  docId -> start,count
  flat compact token/codeword payloads
  docId-ordered scoring
  optional SIMD/PQ/residual payloads

Constraints:
- Design only.
- Do not change code.
- Do not change on-disk format.
- Do not make compatibility promises.
- Keep final SQL ordering exact MaxSim.

Add:
  docs/dev/multivector-quantized-inverted-compact-layout.md

Cover:
1. Current layout and bottleneck.
2. Proposed layout:
   - magic/version
   - doc offset table
   - flat compact codeword/token payload
   - optional residual/PQ payload
   - codebook metadata/checksum
   - MVCC/heap TID mapping
   - REINDEX requirements
3. Query path:
   - broad postings
   - precompact document gate
   - docId-ordered compact scoring
   - exact heap MaxSim rerank
4. Stats:
   - layout version
   - offset bytes
   - payload bytes
   - residual bytes
   - docs compact-scored
   - payload bytes touched
   - compact us per doc
5. Compatibility:
   - old experimental layout remains readable or fails clearly
   - mismatch gives REINDEX guidance
   - top_m > 1 requires explicit version
6. Benchmark gates:
   - no quality regression
   - p95 improvement target
   - index byte target
   - final exact MaxSim contract

Validation:
Run markdown checks if available.
Do not modify C code.
```

---

## Suggested execution order

```text
0  audit current implementation
1  implement precompact gate
2  add focused benchmark/recommendation gate
3  optimize query-codeword scoring
4  optimize compact scoring order/layout in scan-local form
5  optimize exact rerank only if it becomes top-two bottleneck
6  update experimental default-quality profile after evidence
7  design future persisted compact layout only if still needed
```

The key implementation sprint is **Prompts 1–2**. They directly target the current failure mode: too many documents survive into compact scoring.

