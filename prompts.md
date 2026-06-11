# Prompt 0 — Baseline audit after latest GitHub update

```text
You are working in github.com/mayflower/pgturbohybrid.

Goal:
Produce a current implementation audit before changing code.

Read:
- AGENTS.md
- docs/multivector-late-interaction.md
- benchmarks/README.md
- benchmarks/dbpedia_colbert_multivector.py
- src/pgturbohybrid_am.h
- src/pgturbohybrid_quant.h
- src/pgturbohybrid_quant.c
- src/pgturbohybrid_stats.c
- docs/dev/multivector-document-graph-design.md
- docs/dev/multivector-colbertsar-research.md

Do not change files.

Confirm current status of:

Index build:
1. document_nodes build mode.
2. proxy-vector graph topology build.
3. centroid_mean build requirements.
4. centroid_lite kmeans centroid sidecar and posting sidecar.
5. quantized_inverted_experimental sidecar.
6. token pooling.
7. entry_sidecar physical index variant.
8. physical index signature grouping in benchmark.
9. build-only benchmark and build phase attribution.

Retrieval:
1. proxy_vector uses fixed-dimensional proxy graph admission.
2. proxy_vector touches full multivectors only for bounded exact rerank.
3. proxy_candidate_limit_effective and proxy_candidate_limit_source.
4. entry_sample scan-time candidate admission experiments.
5. entry_sidecar candidate admission experiments.
6. BM25 rescue.
7. learned_sparse rescue.
8. candidate reservoirs.
9. centroid_lite posting caps.
10. exact rerank phase timing and SIMD kernel reporting.
11. sidecar cache-build vs sidecar-query counters.

Deliverable:
A short table:
- feature
- current status
- files/functions
- missing tests
- whether it affects build, retrieval latency, admission quality, or reporting

Then recommend the next 3 prompts to run.

Constraints:
- No PyLate integration.
- Do not change final SQL ordering semantics.
- Final retained candidates must still be exact MaxSim-ranked unless a prompt explicitly says otherwise.
- Do not introduce silent on-disk format changes.
```

---

# Prompt 1 — Build-only benchmark hardening and acceptance gates

```text
You are working in github.com/mayflower/pgturbohybrid.

Goal:
Make index-build cost measurable and comparable before running expensive retrieval grids.

Focus:
- --document-node-serving-build-only
- build phase attribution
- physical index signature reuse
- build stats from turbohybrid_last_build_stats()

Tasks:

1. Audit current build-only benchmark output.
   Ensure it reports per physical index signature:
   - profile names using this signature
   - reloptions
   - index_bytes
   - elapsed_ms
   - multivector_proxy_build_us
   - multivector_centroid_build_us
   - multivector_centroid_cluster_us
   - multivector_centroid_residual_us
   - multivector_doc_sidecar_write_us
   - multivector_centroid_sidecar_write_us
   - multivector_centroid_posting_write_us
   - multivector_centroid_posting_count
   - build_edges_us
   - write_pages_us
   - wal_us
   - total_us
   - dominant_build_phase
   - build_phase_known_ms
   - build_phase_unattributed_ms

2. Add missing build stats extraction if these fields already exist in turbohybrid_last_build_stats().
   Do not invent fake timings.

3. Add a build acceptance summary:
   - fastest index signature
   - slowest index signature
   - dominant phase per signature
   - profiles sharing the same physical index
   - profiles requiring separate rebuilds and why

4. Add slow build warnings:
   - centroid_kmeans_dominates_build
   - centroid_posting_write_dominates_build
   - proxy_build_dominates_build
   - graph_edges_dominates_build
   - build_unattributed_high
   - index_rebuild_not_reused

5. Markdown:
   Add a compact "Document-node build cost" table.

6. Tests:
   Add pure-Python tests for:
   - build phase summary calculation
   - dominant_build_phase selection
   - physical signature grouping
   - missing build fields tolerated
   - profiles sharing index signatures are grouped deterministically

Validation:
Run:
  nix --extra-experimental-features 'nix-command flakes' develop --command \
    python benchmarks/dbpedia_colbert_multivector.py --help

Run the Python/self-check tests.

Do not change C search/index behavior.
Do not commit generated benchmark output.
```

---

# Prompt 2 — Fast build-only 10k smoke and report interpretation

```text
You are working in github.com/mayflower/pgturbohybrid.

Goal:
Add or improve a fast build-only smoke command and docs for diagnosing CREATE INDEX stalls.

Tasks:

1. Update benchmarks/README.md with a "Build-only first" section.
   Explain that 10k/x00k document-node experiments should run build-only before retrieval/admission grids.

2. Provide a command like:

   nix develop .#bench
   python benchmarks/dbpedia_colbert_multivector.py \
     --database pgturbohybrid_dbpedia_colbert_10k_build \
     --precomputed-dataset johannhartmann/pgturbohybrid_dbpedia_colbert \
     --max-docs 10000 \
     --max-queries 25 \
     --reuse-data \
     --document-node-serving-build-only \
     --document-node-serving-grid-include-proxy-encoders \
     --document-node-serving-grid-profiles centroid_mean_f16,proxy_max_pool_f16,proxy_normalized_mean_f16 \
     --output .nix-dev/tmp/dbpedia-colbert-serving-build-10k.json \
     --markdown-output .nix-dev/tmp/dbpedia-colbert-serving-build-10k.md

3. Add interpretation guidance:
   - If centroid build dominates: reduce centroid count, test token pooling, or avoid centroid_lite.
   - If graph build dominates: test cheaper proxy encoders and entry-sidecar separately.
   - If sidecar write dominates: compare f16/sq8 and token pooling.
   - If build_phase_unattributed_ms is high: instrument graph/topology build before guessing.

4. Ensure docs warn:
   - exact_symmetric build scorer is diagnostic only for small corpora.
   - x00k production build should use proxy graph topology unless benchmarks prove otherwise.

Validation:
Run markdown checks if available.
Run:
  nix --extra-experimental-features 'nix-command flakes' develop --command \
    python benchmarks/dbpedia_colbert_multivector.py --help

Do not commit generated outputs.
```

---

# Prompt 3 — Retrieval latency guard: make proxy_vector sidecar behavior a regression gate

```text
You are working in github.com/mayflower/pgturbohybrid.

Goal:
Turn the recent proxy_vector sidecar fixes/stats into hard regression checks.

Background:
proxy_vector should use the persisted fixed-dimensional proxy vector for graph admission and touch full document multivectors only for bounded exact rerank or explicit full document-node modes.

Tasks:

1. Add SQL/TAP regression coverage for a tiny document_nodes proxy_vector index:
   - Create synthetic turbohybrid_multivector rows.
   - Build index with multivector_graph = document_nodes.
   - Query with candidate_source = proxy_vector.
   - Assert last_scan_stats shows:
     - multivector_candidate_source = proxy_vector
     - proxy_vector_uses_full_sidecar_for_graph = false
     - proxy_full_sidecar_vectors_loaded = 0 or near 0 before exact rerank
     - proxy_exact_rerank_docs <= multivector_exact_rerank_k_effective
     - proxy_candidate_limit_effective exists
     - proxy_candidate_limit_source exists
     - final result identity is heap/document based, not nodeId

2. Add a near-exhaustive warning regression:
   - candidate_k small, table small.
   - proxy_vector_near_exhaustive_sidecar_touch must be false unless explicit exact/near-exhaustive mode is selected.

3. Add exact oracle regression:
   - When candidate/rerank budget covers all visible docs, proxy_vector final top-k matches exact_doc_scan.
   - Final ordering must be exact MaxSim over retained candidates.

4. Add benchmark slow-path warning if not already present:
   - proxy_vector_near_exhaustive_sidecar_touch
   - proxy_vector_uses_full_sidecar_for_graph
   - sidecar_query_bytes_touched high relative to exact_rerank_docs
   - proxy_candidates_returned << requested candidate_k

Validation:
Run:
  nix --extra-experimental-features 'nix-command flakes' develop --command th-installcheck
  nix --extra-experimental-features 'nix-command flakes' develop --command th-smoke

Do not change query semantics.
Do not change persisted formats.
```

---

# Prompt 4 — Candidate-limit and underfill diagnostics

```text
You are working in github.com/mayflower/pgturbohybrid.

Goal:
Make proxy/document-node candidate underfill actionable.

Problem:
Users may request candidate_k=800 but the graph path may return fewer candidates because EF, oversampling, final_k, doc_candidate_k, entry selection, or graph traversal limits bind.

Tasks:

1. Audit current candidate limit stats:
   - dense_k requested
   - multivector_doc_candidate_k
   - proxy_candidate_limit_effective
   - proxy_candidate_limit_source
   - proxy_candidates_returned
   - multivector_proxy_candidate_target
   - multivector_doc_graph_search_ef
   - multivector_doc_graph_oversampling
   - multivector_exact_rerank_k_effective
   - final_k
   - effective_result_target

2. Add missing stats if the extension already knows the values.

3. Add benchmark-derived fields:
   - requested_candidate_k
   - effective_candidate_k
   - candidate_underfill
   - candidate_underfill_ratio
   - candidate_underfill_reason
   - next_admission_hint

4. next_admission_hint rules:
   - if candidate capped by EF: try higher EF or entry_sample.
   - if capped by final_k * oversampling: increase oversampling.
   - if capped by doc_candidate_k: increase multivector_doc_candidate_k.
   - if candidate source returns few docs despite high EF: try entry_sidecar or stronger proxy.
   - if exact_rerank_k is limiting rescue candidates: increase serving_exact_rerank_k or candidate reservoir.

5. Recommendation:
   Rejected profiles should include these hints.

6. Tests:
   Add synthetic tests for each candidate_underfill_reason and hint.

Validation:
Run Python/self-check tests.
Run:
  nix --extra-experimental-features 'nix-command flakes' develop --command \
    python benchmarks/dbpedia_colbert_multivector.py --help
```

---

# Prompt 5 — Focused proxy/entry-admission grid

```text
You are working in github.com/mayflower/pgturbohybrid.

Goal:
Add a focused serving-grid mode for document-node proxy admission, avoiding the full grid.

Add CLI:
  --document-node-serving-grid-proxy-admission-focus

This mode should compare only:

Profiles:
- proxy_normalized_mean_f16
- proxy_max_pool_f16
- centroid_mean_f16

Entry experiments:
- baseline
- entry_sample_032
- entry_sample_128
- entry_sidecar_128
- entry_sidecar_256

EF / oversampling:
- EF: 100,200,400
- oversampling: 1,2

Budgets:
- largest_only candidate budget, default 800 unless overridden

Exact rerank:
- serving exact rerank k default 100

Output:
For each row:
- profile
- proxy_encoder
- entry_sample_count
- entry_sidecar_enabled
- entry_sidecar_representatives
- ef
- oversampling
- p50/p95/p99
- proxy_candidates_returned
- candidate_underfill_reason
- top10 admission recall if admission is enabled
- ndcg@10 if qrels exist
- graph_entry_sample_scored
- graph_entry_sidecar_scored
- graph_entry_sidecar_selected
- graph traversal time
- exact rerank time
- sidecar query bytes/time

Delta report:
Compare each entry variant against its matching baseline at the same EF/oversampling.
Report:
- admission delta
- ndcg delta
- p95 delta
- candidate count delta

Tests:
- profile expansion deterministic.
- entry-sample variants reuse physical index signature.
- entry-sidecar variants have separate physical index signature.
- delta calculation correct.

Validation:
Run Python/self-check tests.
Run --help.

Do not change C behavior in this prompt.
```

---

# Prompt 6 — Entry-sidecar build and retrieval validation

```text
You are working in github.com/mayflower/pgturbohybrid.

Precondition:
The focused proxy/entry-admission grid shows that entry_sidecar improves admission or latency at acceptable build cost.

Goal:
Harden entry_sidecar as a document-node admission feature.

Tasks:

1. Audit entry_sidecar build:
   - reloptions
   - representative count
   - representative strategy
   - build stats
   - index_stats
   - last_scan_stats

2. Add or improve stats:
   - graph_entry_sidecar_representatives_configured
   - graph_entry_sidecar_strategy
   - graph_entry_sidecar_count
   - graph_entry_sidecar_scored
   - graph_entry_sidecar_selected
   - graph_entry_sidecar_us
   - entry_sidecar_build_us if missing
   - entry_sidecar_bytes if missing

3. Add regression tests:
   - index_stats reports entry_sidecar mode.
   - query stats report entry_sidecar counters.
   - final ranking remains exact MaxSim over retained candidates.
   - entry_sidecar build does not change heap/document identity.

4. Add benchmark warning:
   - entry_sidecar_build_cost_high
   - entry_sidecar_no_admission_gain
   - entry_sidecar_latency_regression

Validation:
Run:
  nix --extra-experimental-features 'nix-command flakes' develop --command th-installcheck
  nix --extra-experimental-features 'nix-command flakes' develop --command th-smoke
```

---

# Prompt 7 — BM25 rescue focused implementation and cap accounting

```text
You are working in github.com/mayflower/pgturbohybrid.

Goal:
Harden BM25 rescue as a CPU-friendly admission safety net for proxy_vector and centroid_mean.

Current benchmark docs describe BM25 rescue profiles and rescue cap accounting. Ensure implementation and tests match the docs.

Tasks:

1. Audit profiles:
   - proxy_normalized_mean_f16_bm25_rescue
   - centroid_mean_f16_bm25_rescue
   - proxy_max_pool_f16_bm25_rescue when proxy encoder variants are enabled

2. Verify benchmark query passes text_query for rescue profiles.
   BM25 rescue must not silently run without text_query.

3. Verify index build includes the BM25 key only for rescue/hybrid profiles that need it.
   Do not force lexical key on dense-only profiles.

4. Add/verify stats:
   - multivector_bm25_injection_enabled
   - multivector_bm25_injection_candidates
   - multivector_bm25_injection_candidate_limit
   - multivector_bm25_injection_pool_size
   - multivector_bm25_injection_limit_reason
   - multivector_bm25_injection_retained
   - multivector_bm25_injection_exact_reranked
   - reservoir_bm25_docs if reservoirs are used

5. Recommendation:
   Add rejection/hint reasons:
   - bm25_rescue_missing_text_query
   - bm25_rescue_underfilled
   - bm25_rescue_limited_by_exact_rerank_k
   - bm25_rescue_no_admission_gain
   - bm25_rescue_latency_regression

6. Tests:
   - rescue profile builds lexical key.
   - dense-only profile does not build lexical key.
   - rescue profile passes text_query.
   - missing text_query is detected.
   - cap accounting survives JSON/Markdown.

Validation:
Run Python/self-check tests.
Run th-installcheck if SQL behavior changes.
```

This follows the low-latency rerank lesson from PLAID reproducibility work: lexical reranking baselines can be very competitive in low-latency regimes, but must be reported honestly as candidate-source evidence. ([arXiv][2])

---

# Prompt 8 — Learned-sparse rescue plumbing

```text
You are working in github.com/mayflower/pgturbohybrid.

Goal:
Make learned_sparse rescue robust enough for SPLATE/SLIM-style experiments.

Do not train sparse models inside PostgreSQL.

Tasks:

1. Audit JSONL import:
   - --learned-sparse-doc-jsonl
   - --learned-sparse-query-jsonl
   - coverage reporting
   - partial coverage warnings

2. Ensure benchmark profiles:
   - proxy_normalized_mean_f16_learned_sparse_rescue
   - centroid_mean_f16_learned_sparse_rescue
   - proxy_max_pool_f16_learned_sparse_rescue when proxy variants are enabled

3. Ensure learned-sparse rescue builds the correct sparse lexical key:
   - learned_sparse_tsv
   - not body_tsv

4. Ensure query path uses learned sparse query features, not websearch_to_tsquery text fallback.

5. Stats:
   - learned_sparse_candidates
   - learned_sparse_retained_for_maxsim
   - learned_sparse_branch_latency_us
   - learned_sparse_doc_coverage
   - learned_sparse_query_coverage
   - learned_sparse_partial_coverage warning

6. Recommendation:
   Reject learned-sparse profiles as production evidence if coverage is partial.
   Still report them as plumbing/admission evidence.

7. Tests:
   - JSONL parser handles valid rows.
   - missing query/doc JSONL fails clearly.
   - partial coverage flagged.
   - learned_sparse_tsv key selected for learned sparse.
   - final ranking remains exact MaxSim.

Validation:
Run Python/self-check tests.
Run th-installcheck if SQL changes are made.
```

---

# Prompt 9 — Centroid-lite focused caps and upper-bound preparation

```text
You are working in github.com/mayflower/pgturbohybrid.

Goal:
Improve centroid_lite as a PLAID-inspired admission path without changing final ranking semantics.

Current state:
centroid_lite persists document-local kmeans centroids, residual summaries, and codeword posting tuples. It supports max postings per token caps using deterministic uniform-stride sampling. It still exact-reranks retained original multivectors.

Tasks:

1. Audit current centroid_lite stats:
   - centroid_lists_visited
   - centroid_docs_touched
   - centroid_pruned_docs
   - centroid_postings_touched
   - centroid_postings_skipped
   - centroid_posting_limit_per_token
   - centroid_posting_cap_strategy
   - centroid_candidates

2. Add derived warnings:
   - centroid_lite_near_exhaustive_docs_touched
   - centroid_lite_postings_near_exhaustive
   - centroid_lite_cap_too_aggressive
   - centroid_lite_no_admission_gain
   - centroid_lite_latency_regression

3. Add focused benchmark mode:
   --document-node-serving-grid-centroid-lite-focus

Profiles:
- centroid_lite_f16
- centroid_lite_f16_cap_016
- centroid_lite_f16_cap_032
- centroid_lite_f16_cap_064
- centroid_lite_f16_pool_050
- centroid_mean_f16 as safe baseline

Metrics:
- p50/p95
- top10 admission recall
- ndcg@10 if qrels exist
- centroid_docs_touched/doc_count
- centroid_postings_touched
- centroid_candidates
- exact rerank docs/time

4. Do not implement new pruning yet.
   This prompt is for making cap tradeoffs measurable.

Tests:
- cap profile expansion deterministic.
- centroid cap GUC applied.
- uniform_stride cap strategy visible in stats.
- near-exhaustive warning works.

Validation:
Run Python/self-check tests.
Run th-installcheck if C stats change.
```

---

# Prompt 10 — Centroid-lite upper-bound pruning prototype

```text
You are working in github.com/mayflower/pgturbohybrid.

Precondition:
centroid_lite focused benchmark shows uncapped or uniformly capped centroid_lite either touches too many docs or loses too much admission.

Goal:
Prototype a guarded centroid_lite upper-bound pruning mode.

Do not change final ranking.
Do not make this default.
Do not change persisted format unless explicitly versioned and guarded.

Add GUC:
  turbohybrid.multivector_centroid_lite_pruning = off | safe_upper_bound

Default:
  off

safe_upper_bound behavior:
- Use query-centroid interaction to estimate an upper bound per document before admitting it.
- Drop a document only if the upper bound proves it cannot enter the current candidate band.
- If no safe bound is available, keep the document.
- Exact MaxSim rerank over retained candidates remains final.

Stats:
- centroid_upper_bound_enabled
- centroid_upper_bound_docs_checked
- centroid_upper_bound_docs_pruned
- centroid_upper_bound_prune_time_us
- centroid_upper_bound_unsafe_fallbacks
- centroid_candidates_before_bound
- centroid_candidates_after_bound

Tests:
- deterministic tiny case where bound prunes nothing.
- deterministic case where a clearly impossible doc is pruned.
- exhaustive candidate budget with pruning off/on yields same final top-k.
- unsafe bound fallback keeps candidates.
- final ordering remains exact MaxSim.

Benchmark:
Add capped/pruned rows to centroid-lite focus grid.

Validation:
Run:
  nix --extra-experimental-features 'nix-command flakes' develop --command th-installcheck
  nix --extra-experimental-features 'nix-command flakes' develop --command th-smoke
```

This is the first real PLAID-like step: use centroid interaction/pruning for admission, exact MaxSim final. PLAID’s core speedup comes from centroid interaction and pruning before exact scoring. ([arXiv][1])

---

# Prompt 11 — EMVB-style bitset prefilter design doc and minimal prototype

```text
You are working in github.com/mayflower/pgturbohybrid.

Goal:
Create a design doc and minimal guarded prototype for EMVB-style bitset prefiltering over centroid_lite postings.

Do not make it production.
Do not change final ranking.
Do not replace centroid_lite default behavior.

Step 1: Design doc
Add docs/dev/multivector-centroid-bitset-prefilter.md.

Cover:
- current centroid_lite posting layout
- candidate docs touched per query
- proposed bitset layout:
  - codeword/centroid -> compressed doc bitset or block bitset
  - optional per-block count/popcount
  - optional doc frequency / IDF for centroid lists
- query-time:
  - choose top centroid/codeword lists per query token
  - OR/AND/score bitsets
  - keep bounded doc candidate set
  - exact MaxSim rerank retained docs
- memory and on-disk format versioning
- MVCC visibility implications
- stats and benchmark requirements

Step 2: Minimal experimental prototype
Add GUC:
  turbohybrid.multivector_centroid_lite_bitset_prefilter = off | experimental

Default:
  off

Prototype may use scan-local bitsets first.
No persisted format change in first slice.

Stats:
- centroid_bitset_prefilter_enabled
- centroid_bitset_lists_used
- centroid_bitset_docs_set
- centroid_bitset_docs_after_threshold
- centroid_bitset_prefilter_time_us
- centroid_bitset_memory_bytes

Tests:
- off preserves current behavior.
- experimental path returns same final top-k as uncapped centroid_lite when candidate budget is exhaustive.
- bitset memory cap respected.
- final ordering exact MaxSim.

Validation:
Run th-installcheck and th-smoke.
```

EMVB’s reported gains come from bit-vector prefiltering, SIMD column-wise centroid interaction and PQ; this prompt only takes the first safe step. ([arXiv][3])

---

# Prompt 12 — Learned projection / FDE proxy implementation plan

```text
You are working in github.com/mayflower/pgturbohybrid.

Goal:
Plan and implement the first safe slice of learned_projection_v1 / FDE-style document proxy.

Background:
The repo has random_projection_fde and learned_projection_placeholder. MUVERA-style FDE reduces multi-vector retrieval to single-vector retrieval and then exact-reranks candidates.

Do not implement model training inside PostgreSQL.
Do not change final ranking.
Do not make this default.

Step 1: Design doc
Add docs/dev/multivector-learned-projection-proxy.md.

Cover:
- model profile name
- projection dimension
- query/document asymmetric projection support
- storage of projection weights:
  - admin-installed file
  - SQL table
  - extension GUC path
  - not embedded in index unless versioned
- build-time doc proxy generation
- query-time query proxy generation
- validation:
  - dimension
  - model name
  - checksum/version
- final exact MaxSim rerank

Step 2: Minimal implementation
Add proxy encoder:
  learned_projection_v1

Behavior:
- If no projection weights are configured, fail explicitly.
- If weights configured, encode document multivector into proxy vector at index build.
- Encode query multivector into compatible proxy vector at scan time.
- Use proxy graph admission.
- Exact MaxSim final rerank unchanged.

Stats:
- proxy_encoder_kind = learned_projection_v1
- learned_projection_loaded
- learned_projection_dim
- learned_projection_weight_bytes
- learned_projection_checksum
- learned_projection_query_encode_us
- learned_projection_doc_encode_build_us

Tests:
- missing weights fails explicitly.
- dimension mismatch fails.
- deterministic tiny weights produce deterministic proxy.
- exhaustive candidate budget final top-k matches exact_doc_scan.
- index_stats exposes model/checksum/dim.

Benchmark:
Add optional serving-grid profile:
  proxy_learned_projection_v1_f16

Validation:
Run th-installcheck and th-smoke.
```

This is the MUVERA/FDE direction: better single-vector admission while retaining exact MaxSim final ranking. ([arXiv][4])

---

# Prompt 13 — Token pooling build/retrieval tradeoff gate

```text
You are working in github.com/mayflower/pgturbohybrid.

Goal:
Make token pooling decisions evidence-based for build time, index size, latency, and quality.

Current implementation supports document-node token pooling with greedy_cosine/kmeans and target ratios.

Tasks:

1. Add focused benchmark mode:
   --document-node-serving-grid-token-pooling-focus

Profiles:
- proxy_normalized_mean_f16 no pooling
- proxy_normalized_mean_f16 greedy_cosine 0.75
- proxy_normalized_mean_f16 greedy_cosine 0.50
- proxy_normalized_mean_f16 greedy_cosine 0.33
- centroid_mean_f16 no pooling
- centroid_mean_f16 greedy_cosine 0.75
- centroid_mean_f16 greedy_cosine 0.50
- centroid_mean_f16 greedy_cosine 0.33

Metrics:
- build time
- index bytes
- tokens original
- tokens pooled
- pooling ratio
- p50/p95 latency
- exact rerank pairs
- exact rerank time
- top10 admission recall
- ndcg@10 if qrels exist

2. Add recommendation:
   - best_pooling_latency_safe
   - best_pooling_quality_safe
   - rejected_pooling_profiles with reasons

3. Tests:
   - profile expansion deterministic.
   - pooling ratio reported.
   - no-pooling and pooling profiles have different index signatures.
   - qrel-less mode still recommends by admission/latency.

Validation:
Run Python/self-check tests.
```

Token pooling is a practical way to reduce memory and MaxSim work; published work reports large token-count reductions with limited retrieval degradation. ([arXiv][5])

---

# Prompt 14 — Quantized inverted: replace deterministic codeword with pluggable codebook

```text
You are working in github.com/mayflower/pgturbohybrid.

Goal:
Move quantized_inverted_experimental one step closer to ColBERTSaR-style retrieval by replacing the temporary deterministic largest-magnitude codeword assignment with a pluggable experimental codebook.

Keep it experimental.
Do not make it default.
Do not promise on-disk compatibility.
Final ranking remains exact MaxSim.

Step 1: Design
Update docs/dev/multivector-colbertsar-research.md:
- current deterministic codeword assignment
- target codebook assignment
- codebook source:
  - external file
  - SQL table
  - model profile
- codebook version/magic/checksum
- rebuild requirements
- stats and benchmark gates

Step 2: Implementation
Add experimental GUCs:
- turbohybrid.multivector_quantized_inverted_codebook = deterministic | external
- turbohybrid.multivector_quantized_inverted_codebook_path = ''
- turbohybrid.multivector_quantized_inverted_codebook_top_m = 1

Behavior:
- deterministic preserves current behavior.
- external loads codebook only when explicit.
- missing/invalid codebook fails.
- build stores codebook metadata/checksum in experimental sidecar/index stats.
- query uses same codebook metadata.
- mismatch fails with REINDEX guidance.

Stats:
- quantized_inverted_codebook_source
- quantized_inverted_codebook_size
- quantized_inverted_codebook_dim
- quantized_inverted_codebook_checksum
- quantized_inverted_codebook_top_m
- quantized_inverted_assignment_us
- quantized_inverted_postings_touched
- quantized_inverted_docs_scored
- quantized_inverted_candidates

Tests:
- deterministic still works.
- external missing file fails.
- dimension mismatch fails.
- checksum mismatch fails.
- token-node index rejects branch.
- exhaustive candidate budget final top-k matches exact_doc_scan.

Validation:
Run th-installcheck and th-smoke.
```

This follows the ColBERTSaR/quantized-inverted direction, but keeps the feature explicitly experimental.

---

# Prompt 15 — Compact-code fused scoring prototype

```text
You are working in github.com/mayflower/pgturbohybrid.

Precondition:
quantized_inverted_experimental with codebook metadata exists, and benchmark evidence shows posting traversal / compact candidate scoring is a bottleneck.

Goal:
Prototype WARP-style fused compact-code scoring for experimental quantized inverted candidates.

Do not change final SQL ordering.
Do not make it default.
Do not change stable on-disk formats.

Tasks:

1. Add scalar reference compact scorer:
   - input: query token vectors, codeword/residual payloads
   - output: approximate document score for admission only
   - no final ordering use

2. Add optional SIMD kernels only after scalar reference tests:
   - AVX2
   - AVX512
   - NEON

3. Add runtime dispatch stats:
   - quantized_inverted_compact_kernel = scalar|avx2|avx512|neon
   - quantized_inverted_compact_score_us
   - quantized_inverted_compact_docs_scored
   - quantized_inverted_compact_payload_bytes
   - quantized_inverted_compact_topk_changed_vs_scalar

4. Tests:
   - scalar deterministic.
   - SIMD matches scalar within documented tolerance.
   - final exact MaxSim ordering unchanged.
   - approximate scores are admission-only.

5. Benchmark:
   Add opt-in comparison:
   - quantized_inverted_experimental scalar
   - quantized_inverted_experimental compact SIMD
   - centroid_lite
   - centroid_mean proxy

Validation:
Run th-installcheck and th-smoke.
```

WARP’s key lesson is to avoid full decompression/reconstruction in the inner loop and combine compact scoring with efficient reduction. ([arXiv][6])

---

# Prompt 16 — Build topology instrumentation if build remains unattributed

```text
You are working in github.com/mayflower/pgturbohybrid.

Precondition:
build-only report shows build_phase_unattributed_ms is high.

Goal:
Instrument document-node graph/topology build phases.

Do not change build algorithm yet.

Add build stats:
- multivector_graph_node_assignment_us
- multivector_graph_entry_search_us
- multivector_graph_neighbor_search_us
- multivector_graph_neighbor_select_us
- multivector_graph_link_insert_us
- multivector_graph_reciprocal_prune_us
- multivector_graph_segment_write_us
- multivector_graph_wal_us if separable
- multivector_graph_build_distance_proxy_calls
- multivector_graph_build_distance_exact_calls
- multivector_graph_build_distance_cache_hits
- multivector_graph_build_distance_cache_misses

Update:
- turbohybrid_last_build_stats()
- benchmark extraction
- build-only markdown

Tests:
- build stats object contains new fields after document_nodes index build.
- missing fields tolerated in old JSON parsing.

Validation:
Run th-installcheck and th-smoke.
```

---

# Prompt 17 — Build algorithm optimization only after attribution

```text
You are working in github.com/mayflower/pgturbohybrid.

Precondition:
A build-only JSON is provided:

  BUILD_ONLY_JSON=.nix-dev/tmp/dbpedia-colbert-serving-build-10k.json

It contains enough build phase attribution to identify the dominant build phase.

Goal:
Choose one build optimization target from evidence.

Do not guess.

Decision:
1. If centroid kmeans dominates:
   - reduce auto centroid count
   - test token pooling before centroiding
   - consider global codebook path
2. If proxy build dominates:
   - optimize proxy encoder loops
   - avoid repeated normalization/allocation
3. If sidecar write dominates:
   - batch writes
   - reduce per-doc metadata overhead
   - compare f16/sq8
4. If graph neighbor search/select dominates:
   - tune m/ef_construction
   - improve entry selection
   - add bounded fast edge selector only if quality gate passes
5. If WAL/write dominates:
   - batch page writes if safe
   - reduce sidecar payload size

Output:
- chosen target
- evidence
- proposed minimal code change
- tests
- benchmark to rerun

Do not implement unless the prompt includes:
  IMPLEMENT_BUILD_OPTIMIZATION=1
```

---

# Prompt 18 — Serving-profile GUC only after evidence

```text
You are working in github.com/mayflower/pgturbohybrid.

Goal:
Add or update SQL-visible multivector serving profiles only after explicit benchmark evidence.

Required input:
  SERVING_GRID_JSON=/path/to/serving-grid.json

Preconditions:
- JSON contains document_node_serving_recommendation.
- best_latency_safe is non-null.
- best_balanced is non-null.
- selected profiles are non-experimental.
- selected profiles do not use quantized_inverted_experimental.
- selected profiles pass admission/quality thresholds.
- slow-path warnings are absent or explicitly accepted.

If any precondition fails:
- stop with no changes.

If all pass:
Add/update:
  turbohybrid.multivector_serving_profile

Values:
- off
- x00k_latency
- x00k_balanced
- x00k_quality

Rules:
- off preserves existing behavior.
- explicit user-set GUCs override profile defaults.
- no experimental branch in non-experimental profile.
- final ranking exact MaxSim.
- do not change existing defaults unless explicitly requested.

Tests:
- profile applies expected candidate source/proxy/storage/cache settings.
- explicit GUC override wins.
- experimental profile rejected.
- last_scan_stats exposes effective profile/candidate source.

Validation:
Run th-installcheck and th-smoke.
```

---

# Prompt 19 — Docs: “how to choose next experiment” decision tree

```text
You are working in github.com/mayflower/pgturbohybrid.

Goal:
Add a concise decision tree to benchmarks/README.md for choosing the next experiment.

Include:

1. If CREATE INDEX is slow:
   - run build-only
   - inspect dominant_build_phase
   - do not run full serving grid yet

2. If query latency is slow:
   - run latency-only
   - inspect phase timing
   - exact rerank vs graph traversal vs sidecar/cache vs SQL overhead

3. If proxy_vector is fast but admission weak:
   - try centroid_mean
   - try max_pool
   - try entry_sample
   - try entry_sidecar
   - try BM25/learned sparse rescue

4. If centroid_lite is slow:
   - inspect centroid_docs_touched/doc_count
   - try posting caps
   - try upper-bound pruning
   - do not tune exact MaxSim first

5. If exact rerank dominates:
   - lower serving_exact_rerank_k
   - test token pooling
   - verify SIMD kernel
   - consider adaptive rerank

6. If sparse rescue helps:
   - compare BM25 vs learned sparse
   - check rescue cap accounting
   - keep final exact MaxSim

7. If all proxy variants fail admission:
   - run centroid_lite focus
   - run quantized_inverted experimental
   - consider learned_projection_v1

Add command snippets for:
- build-only
- latency-only
- proxy-admission focus
- centroid-lite focus
- token-pooling focus

Validation:
Run markdown checks if available.
Run --help.
```

---

# Empfohlene Reihenfolge

```text
0   baseline audit
1   build-only benchmark hardening
2   build-only docs
3   proxy_vector sidecar regression gates
4   candidate underfill diagnostics
5   proxy/entry-admission focus grid
6   entry-sidecar hardening if evidence positive
7   BM25 rescue hardening
8   learned-sparse rescue hardening
9   centroid-lite focus caps
10  centroid-lite upper-bound pruning
11  EMVB-style bitset prefilter design/prototype
12  learned_projection/FDE proxy
13  token pooling focus
14  quantized inverted codebook
15  compact-code fused scoring
16  build topology instrumentation
17  build optimization with evidence
18  serving profile GUC only after evidence
19  docs decision tree
```

# Praktischer Startpunkt

Ich würde Codex **jetzt sofort** mit diesen vier Prompts starten:

```text
0  baseline audit
1  build-only benchmark hardening
3  proxy_vector sidecar regression gates
4  candidate underfill diagnostics
```

Danach entscheidet der Benchmark, ob der nächste Sprint eher:

```text
proxy/entry admission
centroid_lite pruning
BM25/learned sparse rescue
learned_projection/FDE
oder build graph topology
```

sein sollte.

[1]: https://arxiv.org/abs/2205.09707?utm_source=chatgpt.com "PLAID: An Efficient Engine for Late Interaction Retrieval"
[2]: https://arxiv.org/abs/2404.14989?utm_source=chatgpt.com "A Reproducibility Study of PLAID"
[3]: https://arxiv.org/abs/2404.02805?utm_source=chatgpt.com "Efficient Multi-Vector Dense Retrieval Using Bit Vectors"
[4]: https://arxiv.org/abs/2405.19504?utm_source=chatgpt.com "MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings"
[5]: https://arxiv.org/abs/2409.14683?utm_source=chatgpt.com "Reducing the Footprint of Multi-Vector Retrieval with Minimal Performance Impact via Token Pooling"
[6]: https://arxiv.org/abs/2501.17788?utm_source=chatgpt.com "WARP: An Efficient Engine for Multi-Vector Retrieval"
