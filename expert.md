## Current-code verdict

The current repo is no longer in the “obviously using the wrong dormant path” state. It now has a native hot-path benchmark script that explicitly guards that scans use `scan_orchestration='graph_native'`, and it records latency, scoring-kernel buckets, `graph_batch_us`, `graph_heap_us`, `graph_rescore_us`, code/adj page reads, and visited nodes. 

The shipped SQL path is still the native path: `tqgraphgettuple()` calls `PgturbohybridGraphCollectResults()` on the first tuple, and dense candidate collection also calls `PgturbohybridGraphCollectResults()` if the native graph results are not already populated.  The native code-page cache is also real: `PgturbohybridGraphLoadCodePage()` loads a whole quant-code page, copies codes/payloads/residuals into scan storage/arenas, marks `codePagesLoaded[pageNo]`, and serves later node lookups from memory. 

So if amd64 is still much slower than ARM, I would **not** start with page grouping or the old `PgturbohybridGraphSearchLayer`. The likely causes are now narrower:

1. amd64 is not actually using the intended U8 split kernel.
2. amd64 is using U8 split, but the current “batch” U8 scorer is not a true 4-candidate SIMD batch.
3. AVX-512/VNNI is being selected and downclocking or losing to AVX2/AVX-VNNI.
4. The scan is scoring many more candidates than before or than ARM.
5. Traversal/heap overhead dominates, not scoring.
6. Rescore/local expansion/adaptive widening is active unexpectedly.

## The most suspicious current-code issue

The current U8 path exists and is correctly prioritized in `PgturbohybridGraphScoreNodeBatch()`: batch U8 split is tried first, then signed 4-bit split, signed 2-bit split, packed/LUT fallback, and low-bit fallbacks. 

But `PgturbohybridGraphScoreNodeBatchU8Split()` is only a **control-flow batch**, not a real multi-candidate SIMD batch. It checks four nodes, then loops `j = 0..3` and calls `PgturbohybridGraphQuerySplitU8RawAvx2()` or `...Avx512Vnni()` once per node. That removes some branch and sqrt overhead, but it still walks the full query/code loop separately for each candidate. 

That is probably the biggest reason amd64 can still lag ARM. ARM’s signed split path uses NEON table lookup plus SDOT/I8MM-style dot operations in a tight per-code loop.  On x86, the U8 AVX2 raw scorer is good, but it is still called independently per candidate; the current batch wrapper does not reuse query loads across four candidates or maintain four accumulators inside one dimension loop. 

In other words: **the code has U8 split, but not the full “Qdrant-style hot loop shape” for batched native graph traversal.**

## First debug run: use the new benchmark script

Run the included script on the same dataset/CPU class where you see regression:

```bash
make clean
make SIMD_BUILD=native MATH_MODE=fast
make install

psql -d "$DB" -f benchmarks/native_hotpath_bench.sql
```

The script already benchmarks `dense_only`, `hybrid`, `u8_split_on`, `u8_split_off`, `signed_forced`, `scalar_lut`, and rescore variants, and it aborts if the scan is not `graph_native`. 

The first thing to compare between amd64 and ARM is not wall time alone. Compare:

```sql
SELECT turbohybrid_last_scan_stats();
```

Look especially at:

```text
scan_orchestration
graph_storage_kind
graph_score_kernels
graph_scored_codes
graph_batch_scored_codes
graph_scalar_scored_codes
graph_batch_calls
graph_batch_nodes
graph_batch_us
graph_heap_us
graph_base_us
graph_traverse_us
graph_rescore_us
graph_sort_us
graph_effective_search_ef
graph_effective_result_target
graph_effective_rescore_band
graph_code_pages_read
graph_adj_pages_read
graph_visited_nodes
```

The current stats code records kernel buckets, batch calls/nodes, traversal counters, score mode, SIMD force, query split activity, U8 split usage, rescore data, and code-page cache counters.  

## How to interpret the result

### Case A: amd64 shows `batch_scalar_or_lut` or lots of `single_scalar_or_lut`

Then the U8/signed split path is not active. That is a dispatch/config/build problem.

U8 split requires 4-bit, dimension ≥ 1024, non-L1 mode, SIMD not forced scalar, and AVX2 available; policy-wise, `dense_u8_split=off` disables it and `dense_query_split_impl=signed` prevents auto U8 selection. 

Debug with:

```sql
SET turbohybrid.profile = 'latency';
SET turbohybrid.simd = 'on';
SET turbohybrid.dense_u8_split = 'on';
SET turbohybrid.dense_query_split_impl = 'unsigned';
SET turbohybrid.dense_rescore_band = 'off';

-- run dense query
SELECT turbohybrid_last_scan_stats();
```

Expected hot kernel for 1536-dim 4-bit amd64 should be `batch_u8_split_avx2` or `batch_u8_split_avx512vnni`. If not, inspect `u8_split_enabled`, `graph_query_split_active`, `dense_simd_force`, quantization bits, dimensions, and score mode in the stats.

### Case B: amd64 shows `batch_u8_split_avx512vnni` but is slower than AVX2 or ARM

This may be AVX-512 downclocking or a poor CPU-specific choice. The dispatcher prefers AVX-512 VNNI over AVX-VNNI and AVX2 when available.  The U8 batch scorer also chooses AVX-512 VNNI first when available, otherwise AVX2. 

Add/force benchmark variants for:

```text
simd auto
simd avx2
simd avxvnni
simd avx512vnni
```

If there is no public GUC to force that exact family, add one or temporarily patch the dispatcher. This is important: on some Intel client/server CPUs, AVX-512 can win a microkernel but lose end-to-end latency due to frequency throttling.

### Case C: `u8_split_on` is slower than `signed_forced`

Then the U8 split implementation is not paying off on that CPU. That can happen because the current U8 batch scorer is four single-node calls, while the signed path and ARM path are already tight integer loops.

The next code optimization should be a real `x4` U8 batch raw kernel:

```text
PgturbohybridGraphQuerySplitU8RawAvx2x4(tq, code0, code1, code2, code3, raw[4])
```

It should loop over query chunks once, load `[low|high]` once, decode four candidate code chunks, and maintain four low/high accumulators. The current implementation loops over the entire query four times. 

### Case D: `graph_batch_us` is small but `graph_heap_us` / `graph_base_us` / `graph_traverse_us` is large

Then the dense scorer is not the bottleneck. Focus on `PgturbohybridGraphSearchBaseLayer`: frontier pushes/pops, nearest offers, visited checks, duplicate skips, batch size, and max frontier. The current scan opaque already tracks these native base-layer counters. 

If ARM has the same `graph_scored_codes` but much lower `graph_heap_us`, look for branch prediction/cache-layout differences rather than SIMD scoring.

### Case E: amd64 scores many more nodes

If amd64 has higher `graph_scored_codes`, `graph_effective_search_ef`, `graph_effective_result_target`, local expansion, adaptive widening, or rescore band than ARM or earlier builds, the latency issue is not an amd64 kernel issue.

The latency profile canonicalizes default queries to explicit dense/BM25/final-k defaults, and profile defaults are set dynamically.  But payload filters, exact policy, widening, and local expansion can still change the candidate budget. Compare those fields first before optimizing SIMD.

### Case F: `graph_rescore_us` or `graph_effective_rescore_band` is nonzero on exact-free latency scans

That would be a bug or wrong profile/config. The current commit history says exact-free native indexes should never exact-rescore in latency mode, but verify with stats. 

## Specific code-level suspects

### 1. U8 batch is not a true multi-candidate SIMD batch

The function name/comment says “Batch (groups of 4) unsigned-codebook scorer,” but the implementation calls one raw kernel per node.  This is the top candidate for “still slower than ARM.”

### 2. AVX-512 is selected too eagerly

The dispatcher selects AVX-512 VNNI first when available.  The U8 scorer also prefers AVX-512 VNNI.  For latency, you need empirical CPU-family gating, not “widest vector wins.”

### 3. U8 codebook is not Qdrant’s exact unsigned table

The current U8 codebook is `signed_i8 + 128`: `[1,32,53,70,84,97,110,122,134,146,159,172,186,203,224,255]`, not Qdrant’s `[0,31,52,69,84,97,110,122,134,146,159,172,187,204,225,255]`. The code comment says this is intentional for signed-path parity.  This probably does not explain latency, but it means the “Qdrant-style” path is not exactly Qdrant-style numerically.

### 4. Build defaults are conservative

The Makefile defaults to `SIMD_BUILD=portable` and `MATH_MODE=strict`; only `SIMD_BUILD=native` adds `-march=native`.  The target-attribute SIMD functions should still exist in portable builds, but for benchmarking you should remove build ambiguity with `SIMD_BUILD=native MATH_MODE=fast` and confirm the selected kernel in stats.

## Concrete Claude Code prompts

### Prompt 1 — Add CPU-family SIMD variants to the native benchmark

```text
You are working in mayflower/pgturbohybrid.

Goal:
Find why amd64 native scans are slower than ARM by separating AVX2, AVX-VNNI, AVX-512 VNNI, signed split, U8 split, traversal, and rescore costs.

Start from benchmarks/native_hotpath_bench.sql.

Tasks:
1. Add amd64 variants that force:
   - auto
   - avx2
   - avxvnni
   - avx512vnni
   - scalar
   If the exact SIMD-family force GUC is not exposed, add a temporary developer GUC or benchmark-only macro path.
2. For each variant record:
   - p50/p95/p99
   - graph_score_kernels
   - graph_batch_us
   - graph_heap_us
   - graph_base_us
   - graph_traverse_us
   - graph_scored_codes
   - graph_batch_calls
   - graph_batch_nodes
   - graph_effective_search_ef
   - graph_effective_result_target
   - graph_effective_rescore_band
   - graph_rescore_us
3. Abort if scan_orchestration != graph_native.
4. Print a compact diagnosis:
   - hot kernel
   - avg batch size
   - batch_us per scored code
   - heap_us per visited node
   - total_us breakdown percentages

Acceptance:
- We can tell whether amd64 is losing because AVX-512 is selected, U8 split is slower than signed split, scalar/LUT is active, traversal dominates, or candidate count changed.
```

### Prompt 2 — Implement a real x4 U8 split batch kernel

```text
You are working in mayflower/pgturbohybrid.

Goal:
Replace the current PgturbohybridGraphScoreNodeBatchU8Split implementation, which calls the single-node U8 raw kernel four times, with a true 4-candidate batch raw kernel.

Current issue:
PgturbohybridGraphScoreNodeBatchU8Split() loops j=0..3 and calls PgturbohybridGraphQuerySplitU8RawAvx2/Avx512Vnni once per node. This is not a real batch kernel; it repeats query loads and loop overhead four times.

Tasks:
1. Add:
   PgturbohybridGraphQuerySplitU8RawAvx2x4(tq, code0, code1, code2, code3, raw[4])
2. For each 16-dim chunk:
   - load tq->u8SplitData [low|high] once
   - decode code0..code3 nibbles through PgturbohybridGraphCodebookU8
   - maintain four accumulators
   - avoid repeated query loads
3. Add an AVX-512 VNNI x4 version if profitable; otherwise gate AVX-512 off for this path until measured.
4. Update PgturbohybridGraphScoreNodeBatchU8Split() to use the x4 raw kernel.
5. Keep the single-node kernel for tail/scalar fallback.
6. Add parity tests:
   - x4 raw equals four single-node raw results
   - U8 scalar equals U8 SIMD
   - U8 distance matches signed split / LUT within documented tolerance
7. Add a microbenchmark that scores 10k 1536-dim 4-bit codes and reports ns/code for:
   - scalar/LUT
   - signed split
   - current single-node U8
   - new x4 U8

Acceptance:
- batch_u8_split_avx2 becomes faster than signed split on the target amd64 CPU.
- Native warm p50/p95 improves.
- No score parity regression.
```

### Prompt 3 — Add AVX-512 downclock guard

```text
You are working in mayflower/pgturbohybrid.

Goal:
Prevent AVX-512 VNNI from hurting warm latency on CPUs where it downclocks or loses end-to-end despite a faster-looking microkernel.

Tasks:
1. Add a policy GUC:
   turbohybrid.dense_avx512_policy = auto | off | on
2. In auto mode, only choose AVX-512 VNNI for U8 split when:
   - CPU family is known good, or
   - a startup microbenchmark shows it beats AVX2/AVX-VNNI by a configured margin.
3. Add stats:
   - selected dense SIMD family
   - avx512 policy
   - whether AVX-512 was disabled by policy
4. Update native_hotpath_bench.sql to compare AVX2 vs AVX-512 VNNI.

Acceptance:
- On CPUs where AVX-512 hurts p50/p95, auto selects AVX2/AVX-VNNI.
- On CPUs where AVX-512 helps, auto keeps it.
```

### Prompt 4 — Add a one-query debug view

```text
You are working in mayflower/pgturbohybrid.

Goal:
Make it easy to diagnose one slow query without reading the whole JSON blob.

Add a SQL function:
  turbohybrid_last_scan_diagnosis()

Return a compact record or JSON with:
- scan_orchestration
- graph_storage_kind
- score_mode
- dimensions
- quantization_bits
- simd_force
- u8_split_enabled
- query_split_active
- hot_scoring_kernel
- graph_scored_codes
- graph_batch_calls
- graph_avg_batch_size
- graph_batch_us
- graph_heap_us
- graph_base_us
- graph_traverse_us
- graph_rescore_us
- graph_effective_search_ef
- graph_effective_result_target
- graph_effective_rescore_band
- code_page_hit_rate
- diagnosis string:
  "scalar/LUT fallback", "AVX512 selected", "traversal dominated", "rescore dominated", "candidate budget high", "U8 split hot", etc.

Acceptance:
- One SELECT after a slow query gives the likely bottleneck.
```

## What I would do next

Run `benchmarks/native_hotpath_bench.sql` on amd64 and ARM and compare only the rows for `dense_only`, `u8_split_on`, `u8_split_off`, `signed_forced`, and `scalar_lut`.

The key table is:

```text
variant        hot_kernel                  batch_us  heap_us  scored_codes  visited_nodes  p95
u8_split_on    batch_u8_split_avx2:...      ...
signed_forced  batch_signed_split_avx2:...  ...
scalar_lut     batch_scalar_or_lut:...      ...
```

If `u8_split_on` is hot but not faster than `signed_forced`, implement the true x4 U8 batch kernel. If `batch_us` is not the largest component, stop touching SIMD and debug traversal/candidate-budget fields instead.

