## Prompt 0 — Baseline audit: locate the lossy admission path

```text
You are working in the pgturbohybrid repository.

Goal:
Audit the current multivector candidate-generation path and produce a precise code map before making changes.

Context:
The current multivector path appears to:
1. Run approximate ANN searches per query subvector/token.
2. Aggregate raw subvector hits into document candidates.
3. Apply document candidate caps.
4. Exact-rerank only the surviving candidates with full MaxSim.

This is lossy for ColBERT-style late interaction because a document can be globally strong by sum-of-token MaxSim while never ranking highly enough for any individual query token.

Tasks:
1. Locate the code that:
   - iterates query subvectors,
   - calls the graph/ANN search per subvector,
   - resolves nodeId -> document/heap tuple,
   - updates the MaxSim accumulator,
   - applies raw-hit, unique-doc, doc-candidate, and exact-rerank caps,
   - computes exact heap MaxSim rerank.
2. Produce a new Markdown document:
   docs/dev/multivector-candidate-admission-audit.md
3. In that doc, include:
   - file/function map,
   - current control flow,
   - all relevant GUCs and their defaults,
   - all current scan stats,
   - where a true exact top-K document can be dropped,
   - where we can add instrumentation with minimal risk.
4. Do not modify runtime behavior.
5. Run formatting/tests that are standard for this repo, or explain why they cannot be run.

Acceptance criteria:
- The doc identifies the exact functions and structs involved.
- The doc explicitly distinguishes:
  - ANN/token miss,
  - document aggregation miss,
  - doc-candidate truncation miss,
  - exact-rerank miss.
- No production behavior changes.
```

---

## Prompt 1 — Add admission-recall diagnostics to scan stats

```text
You are working in the pgturbohybrid repository.

Goal:
Add debug-only multivector admission diagnostics so we can prove whether exact top-K documents are missing before exact rerank.

Context:
Exact rerank cannot recover documents that never enter the candidate set. We need scan-level and optional per-document diagnostics that answer:
- Was the exact top-1/top-10 admitted before rerank?
- At what candidate budget was it first admitted?
- Was it dropped by token raw-hit caps, unique-doc caps, doc-candidate caps, or accumulator memory?

Tasks:
1. Add a new GUC:
   turbohybrid.multivector_debug_admission = off | summary | trace
   Default: off.
2. In summary mode, extend turbohybrid_last_scan_stats() with fields:
   - multivector_admission_debug_enabled
   - multivector_admission_candidates_before_rerank
   - multivector_admission_candidates_after_truncation
   - multivector_admission_exact_rerank_docs
   - multivector_admission_truncated_by_doc_candidate_k
   - multivector_admission_truncated_by_accumulator_memory
   - multivector_admission_trace_available
3. In trace mode, collect a bounded per-scan debug trace in memory, exposed as either:
   - a JSON field in turbohybrid_last_scan_stats(), or
   - a new function turbohybrid_last_multivector_admission_trace().
4. Trace entries should be document-keyed, not subvector-keyed, and include at least:
   - docId or heap TID
   - approximate accumulated score before exact rerank
   - query-token coverage count
   - raw hit count
   - duplicate hit count
   - candidate rank before truncation
   - retained_for_exact_rerank boolean
   - exact_rerank_score when available
5. Keep memory bounded. Add a hard trace limit GUC if necessary:
   turbohybrid.multivector_debug_trace_limit, default 1000.
6. Update docs/dev/multivector-late-interaction.md or a new debug doc.
7. Add regression tests proving:
   - default stats do not include large trace payloads,
   - summary mode reports counters,
   - trace mode returns bounded entries,
   - trace is document-keyed.

Acceptance criteria:
- Default behavior and performance are unchanged when debug mode is off.
- Debug trace cannot allocate unbounded memory.
- Existing tests pass.
```

---

## Prompt 2 — Add a deterministic synthetic regression for “many moderate matches”

```text
You are working in the pgturbohybrid repository.

Goal:
Create a small deterministic SQL regression test that reproduces the structural failure mode:
a document is exact-MaxSim top-1 because it matches many query tokens moderately, but it is not the top subvector hit for any single query token.

Context:
This is the core ColBERT-style failure:
- per-token top-k candidate generation can miss a document,
- exact rerank only sees candidates admitted earlier,
- raising k fixes it only near-exhaustively.

Tasks:
1. Add a SQL test under test/sql/ and expected output under test/expected/.
2. Use hand-constructed turbohybrid_multivector values with small dimensions.
3. Create:
   - one query multivector Q with multiple query tokens,
   - one “globally good” document D_good that has moderate best matches for many query tokens,
   - several “spiky” documents D_spike_* that each match one query token strongly but have poor total MaxSim.
4. Verify exact scan ranking:
   ORDER BY turbohybrid_multivector_maxsim_distance(query, doc)
   puts D_good at top-1.
5. Build a turbohybrid multivector index.
6. Run index search under intentionally low candidate budgets.
7. Assert or at least expose that D_good is missing before the fix.
8. Mark the failing assertion in a way compatible with this repo’s test style:
   - either expected output documents the current failure,
   - or add it as a developer-only benchmark/test script if normal regression cannot include failing behavior.
9. Add comments explaining why this test models the DBpedia failure.

Acceptance criteria:
- The test is deterministic and does not require external models or datasets.
- Exact MaxSim top-1 is D_good.
- The index path with low budgets demonstrates admission loss or records debug stats showing why it would be lost.
```

---

## Prompt 3 — Extend DBpedia benchmark with admission-recall reporting

```text
You are working in the pgturbohybrid repository.

Goal:
Extend benchmarks/dbpedia_colbert_multivector.py so it reports candidate-admission recall, not only final retrieval metrics.

Context:
The current benchmark already has:
- pgturbohybrid_colbert_multivector_query_only,
- pgturbohybrid_colbert_multivector_rrf,
- pgturbohybrid_colbert_multivector_exact_scan.

We need to compare exact top-K against the candidate set before exact rerank.

Tasks:
1. Add benchmark options:
   --admission-debug
   --admission-k, default 10
   --admission-budget-sweep, comma-separated candidate budgets such as 100,200,400,800,1600,3200,6400,10000
2. For each query:
   - run exact scan top admission_k,
   - run index search under each candidate budget,
   - collect turbohybrid_last_scan_stats(),
   - collect debug admission trace if available.
3. Report:
   - exact_top1_admitted_before_rerank
   - exact_top10_admission_recall
   - exact_top1_first_budget_admitted
   - exact_top1_candidate_rank_before_rerank if available
   - exact_top1_exact_rerank_rank if available
   - raw_subvector_hits
   - unique_docs
   - maxsim_updates
   - doc_candidates
   - exact_rerank_docs
   - memory estimate
   - latency
4. Output JSON should include a per-query section and an aggregate section.
5. Keep normal benchmark behavior unchanged unless --admission-debug is passed.
6. Update benchmark docs/comments.

Acceptance criteria:
- Running without --admission-debug produces the same shape as before.
- Running with --admission-debug clearly distinguishes:
  - not admitted,
  - admitted before rerank but ranked low,
  - admitted and exact reranked correctly.
```

---

## Prompt 4 — Add an exact per-token oracle mode

```text
You are working in the pgturbohybrid repository.

Goal:
Add a developer/debug mode that replaces approximate per-token ANN with exact per-token top-k over all subvector nodes. This isolates graph ANN miss from structural token-top-k admission loss.

Context:
We need three comparable modes:
A. current graph ANN token hits -> document aggregation -> exact rerank
B. exact per-token top-k over all token nodes -> document aggregation -> exact rerank
C. exact document MaxSim scan

If B is still bad at ordinary budgets, the token-candidate strategy is structurally too lossy.
If B is good and A is bad, graph/quantized token ANN recall is the main problem.

Tasks:
1. Add a developer GUC:
   turbohybrid.multivector_candidate_source = graph | exact_token_scan
   Default: graph.
2. Implement exact_token_scan only for multivector scans.
3. exact_token_scan should:
   - iterate all graph subnodes or available exact/quantized subvector storage,
   - score each node against the current query token,
   - produce the top raw hits per query token using the same raw-hit and unique-doc caps as graph mode,
   - feed the existing document accumulator unchanged.
4. Add scan stats:
   - multivector_candidate_source
   - multivector_exact_token_scan_nodes_scored
   - multivector_exact_token_scan_enabled
5. Add tests on a tiny multivector table proving exact_token_scan returns the same or better candidate admission than graph mode under the same candidate caps.
6. Update docs/dev/multivector-candidate-admission-audit.md.

Important constraints:
- This is a debug/developer path, not a production optimization.
- Keep it guarded by the GUC.
- Do not alter default behavior.

Acceptance criteria:
- The benchmark can run A/B/C comparisons.
- The implementation reuses the existing accumulator and exact rerank code.
```

---

## Prompt 5 — Implement exact/plain MaxSim fallback for near-exhaustive cases

```text
You are working in the pgturbohybrid repository.

Goal:
Add a safe exact/plain fallback for multivector search when the approximate candidate generator would be near-exhaustive or when the corpus/filter cardinality is small.

Context:
Qdrant avoids forcing small or exact cases through lossy HNSW; pgturbohybrid should do the same for multivector MaxSim. If DBpedia-10k needs 6400+ or 10000 document candidates, an exact MaxSim scan is probably a better plan than lossy candidate generation plus rerank.

Tasks:
1. Add GUCs:
   turbohybrid.multivector_plain_fallback = auto | off | force
   turbohybrid.multivector_plain_fallback_max_docs = integer, default choose conservative value
   turbohybrid.multivector_plain_fallback_candidate_fraction = float, default e.g. 0.5
2. In auto mode, choose exact/plain fallback when:
   - estimated live document count is <= max_docs, or
   - requested/effective doc_candidate_k exceeds candidate_fraction * estimated_docs, or
   - exact rerank k exceeds candidate_fraction * estimated_docs.
3. Implement fallback as exact document-level MaxSim over heap multivectors, respecting MVCC visibility and SQL LIMIT/final_k semantics.
4. Add scan stats:
   - multivector_plain_fallback_used
   - multivector_plain_fallback_reason
   - multivector_plain_fallback_docs_scored
   - multivector_plain_fallback_pairs
5. Ensure hybrid behavior is explicit:
   - dense-only multivector may use plain fallback,
   - hybrid RRF should either use dense plain fallback as dense branch or stay on existing path with clear stats,
   - unsupported score-level fusion must remain rejected as before.
6. Add regression tests for:
   - force mode,
   - off mode,
   - auto mode on small table,
   - MVCC visibility after delete/update where applicable.
7. Update user docs.

Acceptance criteria:
- Exact/plain fallback returns the same top-K as turbohybrid_multivector_maxsim_distance exact scan.
- The fallback is never silently used for unsupported fusion semantics.
- Existing index path remains default for larger non-near-exhaustive cases.
```

---

## Prompt 6 — Improve current token path with multi-reservoir retention

```text
You are working in the pgturbohybrid repository.

Goal:
Make the current token/subvector candidate path less lossy before exact rerank by retaining document candidates through multiple reservoirs, not only a single approximate accumulated-score top-K.

Context:
A document can be globally strong by matching many query tokens moderately. A single partial-score top-K can drop such documents too early. We need candidate retention that preserves:
- high partial MaxSim score,
- broad query-token coverage,
- strong per-token evidence,
- BM25/RRF evidence when available.

Tasks:
1. Add an internal multivector candidate selection stage that builds the final exact-rerank candidate set as a union of reservoirs:
   - top_by_approx_maxsim_sum
   - top_by_query_token_coverage
   - top_by_mean_seen_similarity
   - per_query_token_top_docs
   - optional bm25_injected_docs when hybrid/text branch exists
2. Add GUCs:
   turbohybrid.multivector_candidate_reservoirs = off | conservative | balanced
   Default: conservative once tests pass, otherwise off during initial implementation.
   turbohybrid.multivector_per_token_doc_reservoir_k = integer
   turbohybrid.multivector_coverage_reservoir_k = integer
3. Use the existing document accumulator. Do not duplicate exact MaxSim logic.
4. Ensure final exact-rerank candidate count remains bounded by:
   - multivector_doc_candidate_k,
   - multivector_exact_rerank_k,
   - max accumulator memory.
5. Add stats:
   - multivector_reservoirs_enabled
   - multivector_reservoir_score_docs
   - multivector_reservoir_coverage_docs
   - multivector_reservoir_per_token_docs
   - multivector_reservoir_bm25_docs
   - multivector_reservoir_union_docs
   - multivector_reservoir_duplicates
6. Add deterministic tests using the synthetic “many moderate matches” setup.
7. Update docs with explanation that this is a mitigation, not the final Qdrant-style architecture.

Acceptance criteria:
- The synthetic many-moderate-matches case improves admission without requiring near-exhaustive candidate_k.
- Candidate counts and memory remain bounded.
- Results are still document-keyed, never subvector-keyed.
```

---

## Prompt 7 — Add query-token contribution diagnostics and token masking experiment

```text
You are working in the pgturbohybrid repository.

Goal:
Add diagnostics to identify noisy query tokens that flood candidate generation and suppress useful documents.

Context:
Late-interaction candidate generation can be harmed by query tokens that produce many high-scoring but irrelevant token hits. We need per-query-token stats and an optional debug mode to skip low-value tokens.

Tasks:
1. Add per-query-token summary stats in debug mode:
   - query_token_ordinal
   - raw_hits
   - unique_docs
   - duplicate_doc_hits
   - top_hit_similarity
   - contribution_to_top_candidates
   - candidate_docs_retained_from_token
2. Expose the summary through turbohybrid_last_scan_stats() or a dedicated debug function.
3. Add a debug GUC:
   turbohybrid.multivector_debug_skip_query_tokens = comma-separated ordinals
   Default empty.
4. When set, skip those query token ordinals during candidate generation only.
5. Exact rerank must still use the full query multivector unless another explicit debug GUC says otherwise.
6. Extend DBpedia benchmark to run optional token-ablation experiments for a single query.

Acceptance criteria:
- Default behavior unchanged.
- Token skip is clearly marked debug-only.
- Stats are sufficient to identify tokens that contribute many raw hits but few final candidates.
```

---

## Prompt 8 — Add dense-only BM25 candidate injection for MaxSim rerank

```text
You are working in the pgturbohybrid repository.

Goal:
Use the existing BM25 branch as an admission-recall safety net for multivector MaxSim rerank when text is available.

Context:
Learned sparse systems such as SPLATE use sparse retrieval as candidate generation and then rerank with ColBERT/MaxSim. pgturbohybrid already has BM25/RRF infrastructure. For multivector queries with text available, BM25 candidates can help admit documents that token ANN misses.

Tasks:
1. Add an option:
   turbohybrid.multivector_bm25_candidate_injection = off | hybrid_only | dense_with_text
   Default off initially.
2. For dense multivector queries where the table/index has a BM25 key and the query has text_query:
   - collect BM25 candidates,
   - inject them into the exact MaxSim rerank candidate set,
   - final ranking for dense-only mode must remain dense MaxSim, not RRF, unless fusion='rrf' was requested.
3. Add stats:
   - multivector_bm25_injection_enabled
   - multivector_bm25_injection_candidates
   - multivector_bm25_injection_retained
   - multivector_bm25_injection_exact_reranked
4. Preserve existing multivector fusion validation:
   - RRF remains document-level,
   - unsupported score-level fusion stays rejected.
5. Add tests:
   - dense-only multivector plus text_query with injection ranks by exact MaxSim after admission,
   - RRF behavior unchanged,
   - no lexical key -> clear error or no injection depending on existing semantics.
6. Update docs.

Acceptance criteria:
- Injection improves admission opportunities but does not alter final dense-only scoring semantics.
- No result is deduplicated by subvector nodeId.
```

---

## Prompt 9 — Design Qdrant-style document-level MaxSim graph

```text
You are working in the pgturbohybrid repository.

Goal:
Write a design document for a Qdrant-style document-level multivector graph, where candidate generation operates on document/point nodes and every candidate point is scored by approximate full MaxSim.

Context:
The current pgturbohybrid multivector graph indexes one node per document token. Qdrant’s multivector path scores point IDs using full MaxSim over the point’s stored multivector. The practical fix direction is:
- graph node == document/point,
- graph traversal scorer == approximate full MaxSim(query_mv, doc_mv),
- graph build/link scorer == document-level multivector similarity,
- quantization/oversampling/exact rerank are approximation layers, not a token-admission filter.

Tasks:
1. Create docs/dev/multivector-document-graph-design.md.
2. Include:
   - problem statement,
   - current token-node architecture,
   - proposed document-node architecture,
   - storage layout options,
   - build algorithm,
   - search algorithm,
   - exact rerank integration,
   - MVCC/dead tuple handling,
   - incremental insert/update handling,
   - compatibility strategy,
   - GUC/index option names,
   - migration/reindex guidance,
   - testing plan.
3. Address scoring:
   - query-vs-document MaxSim is directional and natural,
   - document-vs-document graph build needs either symmetric scoring or a documented directional choice,
   - propose a symmetrized score such as 0.5 * (MaxSim(A,B)/count(A) + MaxSim(B,A)/count(B)) for graph edge construction if needed.
4. Address storage:
   - do not heap-fetch during graph traversal,
   - store compact per-doc multivector codes or references in index-resident/cache-resident storage,
   - reuse or extend the existing multivector docmap sidecar where appropriate.
5. Include a staged implementation plan:
   - Phase 1: in-memory/dev prototype,
   - Phase 2: index-resident compact storage,
   - Phase 3: quantized scorer + oversampling,
   - Phase 4: exact heap rerank and production safeguards.
6. Do not implement code in this prompt.

Acceptance criteria:
- The design is detailed enough for implementation prompts.
- It explicitly explains why this aligns candidate generation with MaxSim.
- It lists all on-disk compatibility risks.
```

---

## Prompt 10 — Prototype document-level MaxSim search behind a dev-only GUC

```text
You are working in the pgturbohybrid repository.

Goal:
Prototype document-level MaxSim candidate generation behind a dev-only GUC, without changing on-disk format.

Prerequisite:
Read docs/dev/multivector-document-graph-design.md first. If it does not exist, stop and write it before coding.

Context:
This prototype should prove whether document-level candidate generation solves the DBpedia admission problem. It does not need to be the final storage format.

Tasks:
1. Add a dev-only candidate source:
   turbohybrid.multivector_candidate_source = graph | exact_token_scan | exact_doc_scan | doc_graph_prototype
2. For doc_graph_prototype:
   - build or derive an in-memory document list for the scan from existing index/docmap/heap-accessible data,
   - perform a simple HNSW-like or beam-search-like traversal over document IDs if feasible,
   - score candidate documents with approximate full MaxSim using index-resident quantized token data where available,
   - otherwise use heap multivectors only in prototype mode with clear stats and warnings.
3. Exact rerank still uses existing heap float32 MaxSim.
4. Add stats:
   - multivector_doc_graph_prototype_enabled
   - multivector_doc_graph_docs_scored
   - multivector_doc_graph_edges_visited if graph-like
   - multivector_doc_graph_candidates
   - multivector_doc_graph_heap_fetches
   - multivector_doc_graph_warning
5. Keep this explicitly non-default.
6. Extend DBpedia benchmark to include this candidate source.
7. Compare:
   - graph token path,
   - exact token scan,
   - exact doc scan,
   - doc graph prototype.
8. Add docs warning that this is a validation prototype.

Acceptance criteria:
- No on-disk format change.
- Default behavior unchanged.
- Prototype can be benchmarked against exact scan and token candidate generation.
- It reports whether exact top-1/top-10 are admitted at much lower candidate counts.
```

---

## Prompt 11 — Implement production document-level graph storage plan

```text
You are working in the pgturbohybrid repository.

Goal:
Implement the first production slice of document-level multivector graph storage, using a versioned on-disk format and clear REINDEX requirements.

Prerequisite:
Only start this after the doc_graph_prototype benchmark shows materially better admission recall than the token-node path.

Tasks:
1. Add an index option:
   multivector_graph = token_nodes | document_nodes
   Default: token_nodes for compatibility.
2. For document_nodes:
   - assign one graph node per heap document,
   - store docId -> heap TID,
   - store compact per-document multivector code blocks,
   - store token count and dimensions,
   - preserve MVCC visibility semantics.
3. Add versioned sidecar/page structures with magic/version fields.
4. Add metapage fields for document-node multivector storage.
5. Implement build-time document graph construction:
   - graph edge scorer uses document-level multivector similarity,
   - use approximate/quantized MaxSim where possible,
   - document-vs-document scoring must match the design doc.
6. Implement search:
   - graph traversal visits document nodes,
   - scores candidates using approximate full MaxSim,
   - oversamples,
   - exact-reranks heap multivectors.
7. Add scan/index stats:
   - multivector_graph_mode
   - multivector_doc_graph_nodes
   - multivector_doc_graph_candidates
   - multivector_doc_graph_quantized_scores
   - multivector_doc_graph_exact_rerank_docs
8. Add tests:
   - build,
   - scan,
   - insert,
   - update/delete/vacuum,
   - old index compatibility,
   - REINDEX guidance for unsupported/missing sidecar,
   - deterministic recall test from synthetic many-moderate-matches case.
9. Update docs.

Acceptance criteria:
- Existing token-node indexes still work.
- New document-node indexes require explicit opt-in.
- No silent fallback on corrupt sidecar metadata.
- Candidate generation is document-keyed and MaxSim-aligned.
```

---

## Prompt 12 — Add acceptance benchmark gate for multivector recall

```text
You are working in the pgturbohybrid repository.

Goal:
Add a developer benchmark gate that prevents future regressions in multivector candidate admission.

Tasks:
1. Add a benchmark/report mode that runs:
   - exact scan baseline,
   - current token graph,
   - exact token scan debug mode,
   - multi-reservoir token mode if implemented,
   - plain fallback,
   - doc graph prototype or document_nodes mode if implemented.
2. On the deterministic synthetic test:
   - exact top-1 must be admitted by the chosen fixed path at small candidate budgets.
3. On DBpedia when data is available:
   - report exact_top1_admission_rate,
   - exact_top10_admission_recall,
   - latency p50/p95,
   - candidate count,
   - exact rerank docs,
   - memory estimate.
4. Add a Markdown output summary suitable for pasting into PR descriptions.
5. Do not make DBpedia external data required for normal CI.
6. Add documentation explaining how to run the gate locally.

Acceptance criteria:
- CI remains self-contained.
- Developers can reproduce admission-recall claims with one command when DBpedia data is available.
- The report makes it obvious whether a change improves admission, reranking, or only final metric luck.
```

---

## Prompt 13 — Cleanup and docs after mitigation/fix

```text
You are working in the pgturbohybrid repository.

Goal:
Clean up the multivector candidate-generation code and documentation after the debug/fix work.

Tasks:
1. Review all multivector paths for document-keying invariants:
   - SQL results must be heap/document keyed,
   - subvector node IDs are only evidence,
   - final exact MaxSim ranks documents.
2. Remove or clearly mark temporary debug-only code.
3. Ensure all new GUCs are documented with defaults and safety limits.
4. Update:
   - docs/multivector-late-interaction.md
   - docs/dev/multivector-late-interaction.md
   - docs/dev/multivector-candidate-admission-audit.md
   - benchmark docs
5. Add a “Known tradeoffs” section:
   - token-node candidate generation can miss many-moderate-match docs,
   - exact/plain fallback is safer for small or near-exhaustive cases,
   - document-level graph aligns candidate generation with MaxSim,
   - exact rerank remains the semantic final scorer.
6. Run the full relevant test set.

Acceptance criteria:
- Docs match actual GUC names and stats fields.
- No stale references to old debug field names.
- Tests pass.
```

---

## Recommended order to run them

Run these first because they are low-risk and will tell you exactly where the quality is lost:

```text
0 -> 1 -> 2 -> 3 -> 4
```

Then land the safest production improvements:

```text
5 -> 6 -> 7 -> 8
```

Then move to the architectural fix:

```text
9 -> 10 -> 11
```

Finally add benchmark gates and cleanup:

```text
12 -> 13
```

The most valuable first PR is probably **Prompts 1–4 together or separately**: admission diagnostics, deterministic regression, DBpedia admission reporting, and exact-token oracle mode. Those will turn the DBpedia result from “candidate generation seems lossy” into a precise failure classification: ANN miss, structural token-top-k miss, truncation miss, or exact-rerank issue.

[1]: https://github.com/qdrant/qdrant/blob/master/lib/segment/src/vector_storage/query_scorer/mod.rs "qdrant/lib/segment/src/vector_storage/query_scorer/mod.rs at master · qdrant/qdrant · GitHub"
