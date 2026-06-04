# Codex Prompt Plan: Fix and Optimize Multivector Late Interaction in `mayflower/pgturbohybrid`

## Prompt 00 — Baseline audit and safety checklist

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Create a baseline audit for the current multivector implementation before changing behavior.

Tasks:
1. Read:
   - AGENTS.md
   - docs/multivector-late-interaction.md
   - src/pgturbohybrid_multivector.c
   - src/pgturbohybrid_query.c
   - src/pgturbohybrid_quant.c
   - src/pgturbohybrid_am.c
   - test/sql/pgturbohybrid_multivector.sql
2. Create docs/dev/multivector-fix-plan.md.
3. In that document, summarize:
   - SQL I/O gap for turbohybrid_multivector text input.
   - scalar multivector operator currently ignoring text_query.
   - remaining HAS_VECTOR checks that should use denseKind/HAS_DENSE.
   - unique_docs_per_token currently needing per-token accounting.
   - candidate top-k selection needing bounded heap.
   - future docId sidecar optimization.
   - exact MaxSim SIMD/blocking optimization.
   - insert batching optimization.

Acceptance criteria:
- Only documentation changes.
- No C or SQL behavior changes.
- The document has a prioritized checklist and identifies which changes require on-disk compatibility care.
```

---

## Prompt 01 — Add shared dense-query helper functions

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Replace ad-hoc HAS_VECTOR/HAS_MULTIVECTOR checks with predictable helper functions.

Implement in src/pgturbohybrid_query.h / src/pgturbohybrid_query.c:

1. Add helper functions or static inline functions with these names:
   - PgturbohybridQueryHasVector(const PgturbohybridQueryHeader *query)
   - PgturbohybridQueryHasMultiVector(const PgturbohybridQueryHeader *query)
   - PgturbohybridQueryHasDense(const PgturbohybridQueryHeader *query)
   - PgturbohybridQueryHasText(const PgturbohybridQueryHeader *query)
   - PgturbohybridQueryDenseKind(const PgturbohybridQueryHeader *query)

2. The helpers must validate flags consistently with denseKind:
   - vector => HAS_DENSE + HAS_VECTOR + denseKind VECTOR
   - multivector => HAS_DENSE + HAS_MULTIVECTOR + denseKind MULTIVECTOR
   - none => no dense flags + denseKind NONE

3. Replace obvious reads in:
   - src/pgturbohybrid_am.c
   - src/pgturbohybrid_query.c
   - src/pgturbohybrid_multivector.c
   with the helper functions where doing so does not change behavior.

4. Do not change SQL behavior yet except through equivalent refactoring.

Tests:
- Run existing regression tests.
- Ensure pgturbohybrid_query and pgturbohybrid_multivector tests still pass.

Acceptance criteria:
- No new behavior except clearer helper usage.
- No direct flag logic removed where it is validating malformed serialized input; validators may still check raw flags.
- No silent ABI or on-disk changes.
```

---

## Prompt 02 — Fix scalar multivector operator behavior with `text_query`

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Prevent scalar multivector distance from silently ignoring text_query.

Current problem:
`turbohybrid_multivector_distance(turbohybrid_multivector, turbohybrid_query)` computes exact MaxSim when multivector_query exists. If the query also contains text_query and the expression is evaluated outside an index scan, the text branch is ignored.

Implement:
1. Add a helper mirroring vector behavior:
   - static void PgturbohybridMultiVectorRejectTextFallback(void)
   or reuse existing query rejection helper if accessible and appropriate.

2. In pgturbohybrid_multivector_query_distance:
   - validate query.
   - if query has text_query, raise FEATURE_NOT_SUPPORTED.
   - error message should be consistent with vector path:
     "hybrid text queries require a turbohybrid index scan"
   - detail should say scalar multivector MaxSim can only evaluate the multivector payload.

3. Preserve indexed hybrid multivector RRF behavior.

Tests:
Add regression cases to test/sql/pgturbohybrid_multivector.sql:
1. Scalar exact multivector distance without text_query still works:
   SELECT mv <~> turbohybrid_query(multivector_query => qmv);
2. Scalar exact multivector distance with text_query errors:
   SELECT mv <~> turbohybrid_query(multivector_query => qmv, text_query => to_tsquery('alpha'));
3. Indexed hybrid multivector RRF still works.

Acceptance criteria:
- No silent ignoring of text_query in scalar multivector distance.
- Existing vector behavior unchanged.
- Error is deterministic and covered in expected output.
```

---

## Prompt 03 — Implement parseable text input for `turbohybrid_multivector`

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Make turbohybrid_multivector dump/restore and manual literals usable by implementing text input.

Current problem:
turbohybrid_multivector_out emits:
  turbohybrid_multivector(dim=2,count=2,values=[[1,0],[0,1]])
but turbohybrid_multivector_in always raises FEATURE_NOT_SUPPORTED.

Implement:
1. Implement pgturbohybrid_multivector_in(cstring).
2. The input parser should accept the exact output format:
   turbohybrid_multivector(dim=<int>,count=<int>,values=[[...],[...]])
3. Also accept optional whitespace around separators.
4. Reject:
   - dim <= 0
   - count <= 0
   - count mismatch
   - value count mismatch
   - NaN/Inf
   - malformed brackets
   - trailing junk
   - overflow in count * dim
5. Use existing helpers:
   - PgturbohybridMultiVectorSize
   - PgturbohybridCheckMultiVector
6. Keep output format stable unless tests are updated intentionally.

Implementation pattern:
- Add small parser helpers in src/pgturbohybrid_multivector.c:
  - TqMvSkipSpaces
  - TqMvConsumeLiteral
  - TqMvParseInt32
  - TqMvParseFloat4
  - TqMvExpectChar
- Keep parser local to pgturbohybrid_multivector.c.
- Use strtod or PostgreSQL-compatible float parsing; reject non-finite values.

Tests:
Add regression coverage:
1. Round trip:
   SELECT turbohybrid_multivector(ARRAY['[1,0]'::vector, '[0,1]'::vector])::text::turbohybrid_multivector::text;
2. Manual literal:
   SELECT turbohybrid_multivector_dims('turbohybrid_multivector(dim=2,count=1,values=[[1,0]])'::turbohybrid_multivector);
3. Whitespace-tolerant literal.
4. Malformed literal errors:
   - wrong count
   - wrong dim
   - missing brackets
   - NaN
   - Inf
   - trailing junk

Acceptance criteria:
- Existing constructor still works.
- Text round-trip works.
- Regression output is deterministic.
- No binary send/recv required in this prompt.
```

---

## Prompt 04 — Add binary send/recv for `turbohybrid_multivector`

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Add binary I/O for turbohybrid_multivector so COPY BINARY and binary client protocols have a stable path.

Implement:
1. Add SQL functions:
   - turbohybrid_multivector_send(turbohybrid_multivector) returns bytea
   - turbohybrid_multivector_recv(internal) returns turbohybrid_multivector
2. Update CREATE TYPE turbohybrid_multivector with:
   - SEND = turbohybrid_multivector_send
   - RECEIVE = turbohybrid_multivector_recv
3. Binary format:
   - int32 format_version = 1
   - int32 dim
   - int32 count
   - uint32 flags
   - count * dim float4 values in network/binary-safe order
4. Validate on receive using existing helpers.
5. Reject unsupported format_version.

Implementation pattern:
- Use pq_begintypsend / pq_sendint32 / pq_sendfloat4 if available in this codebase/PostgreSQL version.
- Use StringInfo receive helpers.
- Keep all binary I/O in src/pgturbohybrid_multivector.c.

Tests:
1. Add SQL-level smoke tests if feasible.
2. Add a regression test that round-trips through send/recv indirectly if a helper is practical.
3. At minimum, ensure CREATE EXTENSION succeeds and type metadata has send/receive functions.

Acceptance criteria:
- Text I/O from Prompt 03 still passes.
- Binary receive validates malformed size/version where feasible.
- No changes to internal varlena layout.
```

---

## Prompt 05 — Fix per-token `unique_docs_per_token` accounting

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Make turbohybrid.multivector_unique_docs_per_token mean what it says: unique document hits per query token, not unique documents globally.

Current problem:
In PgturbohybridGraphCollectMultiVectorDenseCandidates, the per-token uniqueDocs counter checks whether the document exists in the global doc accumulator. This undercounts docs for later query tokens.

Implement:
1. Inside each query-token loop, create/reset a token-local seen-doc set.
2. Use this token-local set to count unique docs for the current query token.
3. Keep the global docHash for MaxSim aggregation.
4. Keep docIdHash for heap TID -> scan-local docId resolution unless a later prompt replaces it.
5. Update stats:
   - multivector_raw_subvector_hits = raw hits consumed.
   - multivector_unique_docs = total token-local unique doc hits across all query tokens, or document this field clearly.
   - multivector_duplicate_doc_hits = duplicate hits within a token and/or duplicates already present globally; choose one and document it.
6. Prefer predictable helper names:
   - PgturbohybridMultiVectorTokenSeenCreate
   - PgturbohybridMultiVectorTokenSeenReset
   - PgturbohybridMultiVectorTokenSeenAdd

Implementation guidance:
- For simplicity, use an HTAB in tokenCtx keyed by TqDocId for the first implementation.
- Reset tokenCtx each query token.
- Do not allocate O(D) memory.
- Do not change result semantics except the stopping criterion.

Tests:
Add a regression case with:
- two query tokens
- documents that appear for both tokens
- SET turbohybrid.multivector_unique_docs_per_token = 1
- SET turbohybrid.multivector_max_raw_hits_per_token high enough
Assert:
- each query token can collect one unique doc independently
- multivector_subvector_searches equals query vector count
- no duplicate SQL result rows
- stats make sense and do not regress existing tests

Acceptance criteria:
- unique_docs_per_token is per-token.
- Accumulator remains O(C * Q), not O(D * Q).
- Existing multivector dense and hybrid tests still pass.
```

---

## Prompt 06 — Replace multivector candidate qsort loop with bounded heap

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Optimize document candidate selection in PgturbohybridGraphCollectMultiVectorDenseCandidates.

Current problem:
When docCount reaches docLimit, the implementation repeatedly qsorts the candidate array before deciding whether to replace the worst candidate. This can become O(C * K log K).

Implement:
1. Add a small bounded heap for TqDenseCandidate or reuse an existing generic top-N pattern where appropriate.
2. Use predictable helper names:
   - PgturbohybridMultiVectorCandidateWorse
   - PgturbohybridMultiVectorCandidateHeapSiftUp
   - PgturbohybridMultiVectorCandidateHeapSiftDown
   - PgturbohybridMultiVectorCandidateHeapOffer
   - PgturbohybridMultiVectorCandidateHeapSort
3. Keep final output ordering identical to current:
   - best distance first
   - tie-break by heap TID
   - tie-break by nodeId
4. Track a stat if convenient:
   - multivectorCandidateHeapReplacements
   but do not add SQL-visible stats unless you also update tests and docs.

Tests:
1. Existing order-sensitive multivector tests must pass.
2. Add a test with more candidate documents than doc_candidate_k:
   - SET turbohybrid.multivector_doc_candidate_k = 2
   - dense_k larger than 2
   - verify exactly two dense documents are returned before final LIMIT.
3. Confirm no duplicate docs.

Acceptance criteria:
- Candidate selection is O(C log K).
- Final ordering matches previous comparator.
- No behavior change except performance.
```

---

## Prompt 07 — Make dense auto-budget dense-kind aware

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Use denseKind/HAS_DENSE for default dense budget decisions instead of HAS_VECTOR-only checks.

Current problem:
PgturbohybridEffectiveQuery uses HAS_VECTOR to decide whether dense auto-budget applies. That excludes multivector dense queries.

Implement:
1. In PgturbohybridEffectiveQuery:
   - replace hasVector with hasDense where appropriate.
   - hasDense should be true for vector_query or multivector_query.
2. Do not enable adaptive hybrid shape logic for multivector yet unless it is intentionally supported.
3. Add explicit comments:
   - dense default budget applies to both vector and multivector.
   - adaptive hybrid shape/probe currently remains vector-only unless implemented in a later prompt.

Tests:
1. Add regression test:
   - SET turbohybrid.default_dense_k to a known value.
   - use turbohybrid_query(multivector_query => ..., dense_k => NULL).
   - verify last_scan_stats denseCandidatesEffective reflects expected auto-budget/default behavior.
2. Existing vector auto-budget tests must pass.

Acceptance criteria:
- Dense-only multivector default dense_k behaves like dense query budget.
- Adaptive vector-only behavior remains unchanged.
- No unsupported multivector adaptive hybrid path is accidentally enabled.
```

---

## Prompt 08 — Make unsupported multivector fusion modes explicit everywhere

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Ensure unsupported multivector hybrid fusion modes fail early and consistently.

Current behavior:
Hybrid multivector + text supports RRF only. Some checks exist in rescan and collect paths.

Implement:
1. Add one shared helper:
   - PgturbohybridQueryValidateMultiVectorFusionSupport(Relation index, PgturbohybridQueryHeader *query)
2. It should:
   - return OK for dense-only multivector.
   - return OK for multivector + text_query + fusion RRF.
   - ERROR FEATURE_NOT_SUPPORTED for multivector + text_query + weighted / fast_weighted / calibrated.
3. Call it from:
   - pgturbohybridamrescan
   - PgturbohybridCollectScanResults
   - any other relevant hybrid path
4. Use one deterministic error message:
   "hybrid multivector fusion currently supports only fusion => 'rrf'"

Tests:
1. Existing fast_weighted failure remains.
2. Add weighted and calibrated failure cases.
3. Add dense-only multivector with fusion => 'weighted' if currently allowed; decide and document:
   - either allow because fusion is irrelevant without BM25
   - or reject to avoid confusion
   The behavior must be documented and tested.

Acceptance criteria:
- One source of truth for multivector fusion support.
- No duplicate inconsistent error messages.
- RRF hybrid still works.
```

---

## Prompt 09 — Clarify metric naming and add dot-product MaxSim alias

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Reduce SQL confusion around multivector_cosine_turbohybrid_ops using dot-product MaxSim.

Implement:
1. Keep existing `multivector_cosine_turbohybrid_ops` for compatibility.
2. Add an alias opclass:
   - `multivector_maxsim_ip_turbohybrid_ops`
3. It should use the same operator and support functions as the current multivector opclass.
4. Add comments:
   - multivector_cosine_turbohybrid_ops assumes stored/query token vectors are already normalized.
   - multivector_maxsim_ip_turbohybrid_ops exposes the actual raw dot-product MaxSim semantics.
5. Update docs/multivector-late-interaction.md to recommend the new alias for new users.

Tests:
1. CREATE INDEX using multivector_maxsim_ip_turbohybrid_ops.
2. Query returns same result order as multivector_cosine_turbohybrid_ops on a tiny dataset.
3. Existing multivector_cosine_turbohybrid_ops tests still pass.

Acceptance criteria:
- No breaking SQL change.
- Documentation is explicit about dot vs cosine semantics.
```

---

## Prompt 10 — Add `turbohybrid_multivector_subvector` and value inspection helpers

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Improve SQL debuggability for turbohybrid_multivector values.

Implement SQL functions:
1. turbohybrid_multivector_subvector(mv turbohybrid_multivector, ordinal int4) returns vector
2. turbohybrid_multivector_to_vector_array(mv turbohybrid_multivector) returns vector[]
3. Optional:
   - turbohybrid_multivector_l2_norms(mv turbohybrid_multivector) returns float8[]

Semantics:
- Use 1-based SQL ordinal for subvector access.
- Error on ordinal < 1 or ordinal > count.
- Returned vector dimensions must match mv->dim.
- Use existing pgvector-compatible Vector layout helpers.

Tests:
1. Construct a multivector and extract first/second vector.
2. Out-of-range ordinal errors.
3. Round-trip to vector[] and back preserves dims/count/text output.

Acceptance criteria:
- Helpful for debugging.
- No changes to index behavior.
- No performance impact on scan hot path.
```

---

## Prompt 11 — Add brute-force exact recall regression helper

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Add test coverage proving indexed multivector results agree with brute-force exact MaxSim on small deterministic datasets.

Implement:
1. Add a regression test section that:
   - creates a small table with 5-10 multivector documents.
   - creates a multivector turbohybrid index.
   - defines a query multivector.
2. Compare:
   - brute-force order:
     ORDER BY turbohybrid_multivector_maxsim_distance(q, colbert)
   - indexed order:
     ORDER BY colbert <~> turbohybrid_query(multivector_query => q, dense_k => ..., final_k => ...)
3. Use exact rerank on.
4. Use enough dense_k/doc_candidate_k so approximate candidate generation should include all docs.
5. Assert top-k IDs match exactly.

Tests:
- Add to test/sql/pgturbohybrid_multivector.sql or a new test/sql/pgturbohybrid_multivector_recall.sql.
- Update expected output.
- Update Makefile REGRESS if new test file.

Acceptance criteria:
- Small-corpus exact correctness is pinned.
- Test is deterministic.
- Does not rely on timing or random values.
```

---

## Prompt 12 — Add pg_dump / restore coverage for `turbohybrid_multivector`

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Prove turbohybrid_multivector can survive dump/restore after text input is implemented.

Implement:
1. Add a Perl TAP test if the repo’s existing TAP tests are suitable.
2. Test flow:
   - create extension vector
   - create extension pgturbohybrid
   - create table with turbohybrid_multivector column
   - insert a few values
   - pg_dump the database/table
   - restore into a fresh database
   - verify dims/count/text output and exact MaxSim.
3. If TAP is too heavy, add a documented dev test script under test/t or test/sql with comments.

Acceptance criteria:
- Dump/restore for table data works.
- This must fail before Prompt 03 and pass after Prompt 03.
- Do not commit generated dump files.
```

---

## Prompt 13 — Optimize multivector token-to-Vector conversion with reusable buffers

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Reduce allocation churn from converting each multivector token into a temporary pgvector Vector.

Current behavior:
Both build/insert and scan paths allocate a new Vector object for each token.

Implement:
1. Add helper:
   - PgturbohybridMultiVectorCopySubvectorToVector(const PgturbohybridMultiVector *mv, int32 ordinal, Vector *dst)
2. Add size helper:
   - PgturbohybridMultiVectorSubvectorSize(const PgturbohybridMultiVector *mv)
3. In scan path:
   - allocate one reusable Vector buffer in tokenCtx or resultCtx for each token loop.
   - overwrite it for each qi instead of allocating a new Vector every time.
4. In build path:
   - use buildTupleCtx or a reusable per-tuple Vector buffer where safe.
5. Keep old helper if useful, but route through the reusable implementation.

Tests:
- Existing multivector build/scan/insert tests pass.
- Add a regression test with query vector count > 1 and doc vector count > 1 to ensure no buffer overwrite leaks into later tokens.

Acceptance criteria:
- No semantic change.
- Fewer pallocs in Q-token scan loop.
- Memory contexts remain correct.
```

---

## Prompt 14 — Improve accumulator memory layout with slab allocation

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Reduce per-document palloc overhead in multivector MaxSim aggregation.

Current behavior:
Each new document accumulator allocates:
- maxsim[queryCount]
- seen[queryCount]

Implement:
1. Add a scan-local accumulator arena struct:
   - PgturbohybridMultiVectorAccumulatorArena
2. Use predictable helper names:
   - PgturbohybridMultiVectorAccumulatorArenaInit
   - PgturbohybridMultiVectorAccumulatorArenaAllocDoc
   - PgturbohybridMultiVectorAccumulatorArenaEstimatedBytes
3. Allocate maxsim/seen arrays in larger chunks/slabs rather than one palloc per document.
4. Keep semantics:
   - maxsim initialized to -INFINITY
   - seen initialized false
   - memory remains O(C * Q)
5. Keep memory cap enforcement:
   - turbohybrid.multivector_max_accumulator_mb
6. Preserve current HTAB for doc lookup unless a later prompt replaces it.

Tests:
1. Existing multivector tests pass.
2. Add a stress-ish regression with enough docs to touch multiple accumulator entries.
3. Verify multivector_memory_bytes_estimate remains plausible and under cap.

Acceptance criteria:
- Result order unchanged.
- Fewer allocations per document.
- No O(D * Q) memory behavior introduced.
```

---

## Prompt 15 — Add exact MaxSim blocked scalar kernel

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Prepare exact MaxSim for better SIMD by adding a blocked scalar kernel first.

Implement:
1. Keep existing TqMultiVectorMaxSimScalar as the semantic reference.
2. Add:
   - TqMultiVectorMaxSimBlockedScalar
3. Block over query tokens:
   - process small blocks, e.g. 4 or 8 query vectors.
   - stream document tokens.
   - maintain maxsim for the query-token block.
4. Dispatch:
   - initially keep default as existing scalar/SIMD path.
   - add a GUC or internal switch only if the repo already has a suitable pattern.
5. Add a test-only diagnostic function if existing SIMD parity tests need it:
   - compare scalar vs blocked scalar.

Tests:
1. Extend pgturbohybrid_simd_parity or pgturbohybrid_multivector tests.
2. Compare scalar and blocked scalar for:
   - dims 1, 2, 3, 8, 15, 16, 31, 32, 96, 128
   - negative values
   - different query/doc counts
3. Tolerance should be strict enough but not depend on fast-math.

Acceptance criteria:
- Blocked scalar matches reference.
- No change to default query result order unless explicitly enabled.
- Sets up later AVX2/AVX512/NEON block kernels.
```

---

## Prompt 16 — Add blocked AVX2 / NEON exact MaxSim kernels behind existing gates

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Optimize exact heap rerank for multivector MaxSim using blocked SIMD kernels.

Prerequisite:
Prompt 15 implemented blocked scalar reference.

Implement:
1. Add optional blocked SIMD kernels:
   - TqMultiVectorMaxSimBlockedAvx2
   - TqMultiVectorMaxSimBlockedNeon
2. Use existing SIMD style:
   - scalar fallback always available
   - respect PGTURBOHYBRID_DISABLE_SIMD
   - respect turbohybrid.dense_exact_simd_force where applicable
3. Do not require -march=native for portable builds.
4. Dispatch:
   - prefer blocked AVX2 on x86 when available.
   - prefer blocked NEON on ARM when available.
   - fallback to blocked scalar or existing scalar.
5. Keep exact scoring semantics:
   - dot product
   - max over doc tokens
   - sum over query tokens
   - negative values handled correctly

Tests:
1. SIMD parity:
   - scalar vs blocked scalar vs AVX2/NEON where available.
2. Build modes:
   - SIMD_BUILD=none must compile and pass.
   - SIMD_BUILD=portable must compile.
3. Update turbohybrid_simd_capabilities or stats if appropriate:
   - multivector_exact_kernel should distinguish scalar / avx2 / neon / blocked_avx2 / blocked_neon.

Acceptance criteria:
- No portability regression.
- Exact rerank results match scalar.
- multivector_exact_kernel stats remain meaningful.
```

---

## Prompt 17 — Add optional adaptive raw-hit widening per query token

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Improve recall/latency tradeoff for multivector candidate generation.

Current behavior:
rawTarget is computed once from subvector_k, unique_docs_per_token, targetK, and max_raw_hits_per_token.

Implement:
1. Add GUC:
   - turbohybrid.multivector_adaptive_widening = on/off/auto
   If adding a GUC is too much, implement auto behavior behind existing knobs.
2. For each query token:
   - start with rawTarget = multivector_subvector_k
   - run traversal.
   - if token-local unique docs < unique_docs_per_token and rawTarget < max_raw_hits_per_token:
     widen rawTarget and rerun or continue collecting if traversal API supports it.
3. Keep max_raw_hits_per_token as hard cap.
4. Add stats:
   - multivector_adaptive_widening_triggered
   - multivector_adaptive_initial_raw_target
   - multivector_adaptive_final_raw_target
   only if you also update JSON stats and docs.

Tests:
1. Set subvector_k low and unique_docs_per_token higher.
2. Verify raw hits grow but stay <= max_raw_hits_per_token.
3. Verify disabling adaptive widening restores fixed behavior.
4. Existing tests remain stable.

Acceptance criteria:
- Better recall behavior for duplicate-heavy subvector hits.
- Hard caps respected.
- No unbounded repeated graph traversals.
```

---

## Prompt 18 — Add persistent multivector doc map sidecar design document

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Design the on-disk docId sidecar before implementing it.

Reason:
Current multivector scans derive scan-local docId by hashing heap TIDs from subvector hits. That is correct but slower and repeats heap TID metadata per subnode.

Create docs/dev/multivector-docmap-sidecar.md.

Design must cover:
1. On-disk page kind(s), e.g.:
   - PGTURBOHYBRID_GRAPH_PAGE_KIND_MULTIVECTOR_DOCMAP
2. Version/magic:
   - explicit sidecar version
   - compatibility behavior for indexes without sidecar
   - REINDEX guidance if needed
3. Layout:
   - nodeId -> docId
   - docId -> heaptid
   - docId -> firstNodeId
   - docId -> tokenCount
4. Build-time writing.
5. Insert/update behavior.
6. Vacuum/dead tuple behavior.
7. Native cache integration.
8. Memory estimate integration.
9. Scan fallback behavior:
   - if sidecar unavailable, use existing heap-TID hash path.
10. Tests required.

Acceptance criteria:
- Documentation only.
- No on-disk changes yet.
- The design explicitly states how old indexes are handled safely.
```

---

## Prompt 19 — Implement optional persistent multivector doc map sidecar

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Implement the multivector doc map sidecar designed in docs/dev/multivector-docmap-sidecar.md.

Safety:
This prompt changes persisted index format. Do not silently break old indexes.
Use explicit page kind/version/magic and fallback behavior.

Implement:
1. Add new page kind:
   - PGTURBOHYBRID_GRAPH_PAGE_KIND_MULTIVECTOR_DOCMAP
2. Add sidecar metadata to graph metapage or a versioned discoverable chain.
3. Write doc map at build time:
   - nodeId -> docId
   - docId -> heaptid
   - docId -> firstNodeId
   - docId -> tokenCount
4. Load sidecar during multivector scan:
   - prefer sidecar mapping
   - fallback to heap-TID hash path for old indexes or unavailable sidecar
5. Add native cache support:
   - arrays for nodeToDoc and docMap
   - memory estimate fields
6. Update insert path:
   - append doc map entries for new multivector rows
   - update sidecar metadata safely
7. Update stats:
   - multivector_docmap_source = sidecar | heap_tid_hash
   - multivector_docmap_bytes
8. Keep existing tests passing.

Tests:
1. CREATE INDEX then scan uses sidecar.
2. INSERT into multivector index updates sidecar and scan sees new doc.
3. Old/no-sidecar fallback path can be tested with a forced GUC if needed:
   - turbohybrid.multivector_docmap = off/auto
4. VACUUM/dead tuple behavior does not return deleted rows.
5. Memory estimate includes sidecar bytes.

Acceptance criteria:
- No silent incompatibility.
- Old indexes either work through fallback or fail with clear REINDEX guidance.
- Hot path avoids heap-TID hash lookup when sidecar is available.
```

---

## Prompt 20 — Add BM25-only exact MaxSim scoring for future score fusion

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Prepare multivector hybrid score-level fusion by adding exact dense scores for BM25-only candidates.

Current behavior:
PgturbohybridApplyBm25OnlyExactRescore is vector-only and requires HAS_VECTOR.

Implement:
1. Add a multivector-specific function:
   - PgturbohybridApplyBm25OnlyMultiVectorExactRescore
2. Only call it when:
   - query has multivector_query
   - query has text_query
   - index has multivector dense key
   - a future-supported score fusion path needs dense scores
3. For now, do not enable weighted/fast_weighted/calibrated fusion unless fully implemented.
4. Function should:
   - fetch heap multivector for BM25-only candidates
   - compute exact MaxSim
   - assign denseDistance = -maxsim
   - denseSimilarity = maxsim / query_count
   - assign synthetic denseRank deterministically by comparing against dense candidate distances
5. Add comments saying this is a prerequisite for score-level fusion.

Tests:
1. Add unit/regression coverage through a diagnostic function or gated path if possible.
2. Ensure RRF behavior remains unchanged.
3. Ensure unsupported score fusion modes still error.

Acceptance criteria:
- No user-visible unsupported fusion mode becomes accidentally enabled.
- Code path exists and is testable for future fusion work.
```

---

## Prompt 21 — Implement weighted multivector hybrid fusion

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Support `fusion => 'weighted'` for multivector + BM25 hybrid search.

Prerequisites:
- Prompt 20 exists.
- Dense score normalization is defined and tested.

Semantics:
- dense raw score = exact or approximate MaxSim
- dense similarity = maxsim / query_vector_count
- BM25 score normalization should follow existing weighted fusion behavior
- Fusion key is heap tuple/document, never nodeId.

Implement:
1. Remove unsupported error for weighted only after implementation is complete.
2. For dense candidates:
   - use exact rerank score when available
   - otherwise use approximate MaxSim
3. For BM25-only candidates:
   - compute exact MaxSim if score-level fusion needs dense contribution
   - or explicitly define missing dense contribution as 0 and document quality tradeoff
4. Reuse existing weighted fusion patterns where possible.
5. Do not enable fast_weighted or calibrated in this prompt.

Tests:
1. Hybrid multivector weighted returns no duplicate docs.
2. A document with both dense and BM25 evidence ranks above dense-only/BM25-only in a controlled dataset.
3. `require_bm25_match` works.
4. Existing RRF tests still pass.
5. fast_weighted/calibrated still error.

Acceptance criteria:
- weighted support is documented.
- Fusion remains document-keyed.
- Score normalization is deterministic and tested.
```

---

## Prompt 22 — Optimize multivector incremental insert batching

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Reduce per-document insert cost for multivector indexes.

Current behavior:
PgturbohybridGraphInsertMultiVectorInPlace loops over mv->count and calls PgturbohybridGraphInsertValueInPlace once per token vector.

Implement:
1. Add a batched insert function:
   - PgturbohybridGraphInsertMultiVectorBatchInPlace
2. Keep the old function as wrapper or fallback.
3. Batch opportunities:
   - detoast once
   - validate once
   - copy payload values once
   - load correction once
   - initialize insert storage/cache once
   - update metapage once if safe
   - invalidate/update cache once if safe
4. Preserve correctness:
   - every subvector still becomes a graph node
   - all nodes share same heap TID
   - BM25 delta still appended once for the document, not once per subvector
5. If fully batching graph neighbor insertion is too large:
   - at least eliminate repeated detoast/correction/payload/cache setup
   - document remaining O(L * graph_insert) work.

Tests:
1. Existing insert expansion test still passes.
2. Insert a document with 3+ subvectors into dense-only and hybrid indexes.
3. Check node_count increments by token count.
4. Check BM25 hybrid query returns inserted doc only once.
5. Check WAL/restart TAP tests if present.

Acceptance criteria:
- Behavior unchanged.
- Fewer repeated setup operations per multivector insert.
- No unsafe metapage/cache updates.
```

---

## Prompt 23 — Add update/delete/vacuum multivector regression coverage

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Test MVCC lifecycle behavior for multivector indexes.

Add tests for:
1. DELETE:
   - create multivector index
   - delete nearest doc
   - query with LIMIT
   - ensure deleted doc is not returned
   - run VACUUM
   - query again
2. UPDATE:
   - update multivector value from far to near
   - query returns updated row
   - old version not returned
3. Hybrid update:
   - update tsvector and multivector
   - hybrid RRF sees new state
4. Insert after vacuum:
   - insert new multivector doc
   - node_count increases appropriately
   - query returns it

Implementation:
- Add to regression if deterministic and fast.
- Use TAP if VACUUM/MVCC timing is more reliable there.

Acceptance criteria:
- Lifecycle behavior is covered.
- No duplicate heap rows after update.
- Existing tests remain deterministic.
```

---

## Prompt 24 — Add planner and SQL usability tests

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Cover common SQL shapes and ensure unsupported shapes fail predictably.

Add tests for:
1. Dense-only multivector with:
   - explicit LIMIT
   - explicit final_k
   - no final_k
   - dense_k defaulted
2. Hybrid RRF with:
   - text_query only on hybrid index
   - multivector_query + text_query
   - require_bm25_match true
3. Unsupported:
   - vector_query against multivector index
   - multivector_query against vector index
   - text_query against dense-only multivector index
   - expression index
   - wrong key order
   - fusion weighted/fast_weighted/calibrated until implemented
4. Scalar:
   - exact MaxSim without index
   - scalar text_query fallback error
5. EXPLAIN:
   - optionally check the plan uses turbohybrid index when enable_seqscan=off.

Acceptance criteria:
- SQL behavior matrix is pinned.
- Unsupported behavior has clear error messages.
- No flaky timing assertions.
```

---

## Prompt 25 — Add multivector benchmark slope harness

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Add a reproducible benchmark harness that measures scaling slopes, not just one latency number.

Create or update:
- benchmarks/dev/multivector_late_interaction.sql
- benchmarks/dev/multivector_late_interaction.md

Benchmark axes:
1. D docs: small/medium configurable
2. L doc token vectors: 8, 32, 128
3. Q query vectors: 4, 16, 64
4. d dimensions: 32, 96, 128
5. exact_rerank_k: off, 25, 100
6. subvector_k: 16, 64, 256

Report:
- build time
- index size
- node_count
- query latency
- multivector_raw_subvector_hits
- multivector_unique_docs
- multivector_doc_candidates
- multivector_exact_rerank_pairs
- multivector_exact_kernel
- fusion strategy for hybrid case

Rules:
- Do not commit generated results.
- Keep benchmark optional and not part of normal installcheck.
- Use deterministic random seed where possible.

Acceptance criteria:
- Developers can validate:
  - build grows with D * L
  - approximate query grows with Q * ANN
  - exact rerank grows with R * Q * L * d
- Docs explain how to run and interpret the benchmark.
```

---

## Prompt 26 — Final integration cleanup

```markdown
Work in repository: mayflower/pgturbohybrid.

Goal:
Perform final cleanup after multivector fixes and optimizations.

Checklist:
1. Search for direct HAS_VECTOR checks.
   - Replace with PgturbohybridQueryHasDense or PgturbohybridQueryDenseKind where appropriate.
   - Keep raw flag checks only inside validators.
2. Search for multivector result deduplication by nodeId.
   - Ensure SQL results are heap/document-keyed.
3. Search for qsort inside per-document candidate accumulation loops.
   - Ensure bounded heap is used.
4. Search for text_query scalar fallback paths.
   - Ensure multivector and vector behavior are consistent.
5. Search for unsupported fusion messages.
   - Ensure they are deterministic and centralized.
6. Search for new on-disk changes.
   - Ensure version/magic/fallback/REINDEX guidance exists.
7. Update docs:
   - docs/multivector-late-interaction.md
   - docs/how-it-works.md if needed
   - docs/dev/multivector-fix-plan.md
8. Run:
   - make
   - make installcheck
   - make SIMD_BUILD=none
   - relevant TAP tests if available

Acceptance criteria:
- All tests pass.
- Docs match actual implementation.
- No generated artifacts committed.
- Final PR summary lists:
  - fixed SQL support
  - fixed correctness issues
  - performance optimizations
  - remaining limitations
```

---

# Recommended implementation order

```text
00 baseline doc
01 dense-query helpers
02 scalar multivector text_query error
03 text input
04 binary send/recv
05 per-token unique docs
06 bounded heap top-k
07 dense auto-budget for multivector
08 centralized unsupported fusion checks
09 metric/opclass alias
10 SQL inspection helpers
11 brute-force recall tests
12 dump/restore test
13 reusable token Vector buffer
14 accumulator slab
15 blocked scalar exact MaxSim
16 blocked SIMD exact MaxSim
17 adaptive raw-hit widening
18 docmap sidecar design
19 docmap sidecar implementation
20 BM25-only exact MaxSim prep
21 weighted multivector fusion
22 batched multivector insert
23 update/delete/vacuum tests
24 planner/SQL usability matrix
25 benchmark slope harness
26 final cleanup
```
