# Diagnostics Schema

`turbohybrid_last_scan_stats()` returns a `jsonb` summary of the most recent
TurboHybrid scan in the current backend. It is a **diagnostic** surface: it
emits *hundreds* of flat keys plus a grouped nested view, and most of those keys
are internal counters that exist for debugging and benchmarking. This document
defines which keys are **stable** (safe for tools to depend on) versus
**experimental** or **diagnostic-only**.

The stable key names are also defined once as C constants in
`src/pgturbohybrid_diagnostics.h` and emitted from
`src/pgturbohybrid_diagnostics.c`, so a rename is caught at compile time. The
`pgturbohybrid_diagnostics` regression test pins their presence and type.

## Maturity

| Label | Meaning |
| --- | --- |
| **stable** | Name, JSON type, and meaning are not expected to change without a release note. Tools may depend on these. |
| **experimental** | Real, useful, but the name/type/semantics may still change. |
| **diagnostic-only** | Internal counters/timers for debugging and benchmarking. Numerous, may appear/disappear/rename freely between alpha releases. Do not build tooling on them. |

Backward compatibility applies to **stable** keys only. The hundreds of flat
`graph_*`, `bm25_*`, `sparse_*`, `multivector_*`, `adaptive_*`, and `*_us`
timing/counter keys not listed below are **diagnostic-only** unless explicitly
promoted here.

## Structure

Alongside the flat keys, the JSON also exposes a **grouped** view under nested
objects so bottleneck diagnosis reads top-down. The nested sections are:

| Section | Contents |
| --- | --- |
| `dense` | dense branch: `dense.kernels`, `dense.cache`, `dense.traversal`, `dense.timing_us` |
| `bm25` | BM25 branch counters and timing |
| `fusion` | how the per-branch candidate lists were combined |
| `query` | resolved query shape (dimensions, budgets, final-k) |
| `graph_base_layer` | base-layer traversal counters |
| `graph_code_pages` | code-page cache locality counters |
| `graph_score_kernels` | which dense scoring kernel ran, per bucket |

The nested values are built from the same consolidated struct as the flat keys,
so they always agree. The flat keys remain for backward compatibility.

## Stable keys

### General / query shape

| Key | Type | Meaning |
| --- | --- | --- |
| `scan_orchestration` | string | how the scan ran (e.g. `graph_native`) — **stable** |
| `score_mode` | string | scoring mode used — **stable** |
| `dimensions` | number | indexed vector dimensionality — **stable** |
| `quantization_bits` | number | dense quantization code width — **stable** |
| `final_k_effective` | number | resolved final top-k target — **stable** |

### Dense branch

| Key | Type | Meaning |
| --- | --- | --- |
| `dense_branch_used` | boolean | dense branch participated — **stable** |
| `dense_scorer` | string | dense scoring kernel family — **stable** |
| `graph_storage_kind` | string | dense graph storage kind — **stable** |
| `dense_k_effective` | number | resolved dense candidate budget — **stable** |
| `dense_k_defaulted` | boolean | dense_k was inferred vs explicit — **stable** |
| `graph_visited_nodes` | number | graph nodes visited — **stable** |
| `graph_scored_codes` | number | compact codes scored — **stable** |

### BM25 branch

| Key | Type | Meaning |
| --- | --- | --- |
| `bm25_branch_available` | boolean | index has a BM25 key — **stable** |
| `bm25_branch_used` | boolean | BM25 branch participated — **stable** |
| `bm25_k_effective` | number | resolved BM25 candidate budget — **stable** |
| `bm25_k_defaulted` | boolean | bm25_k inferred vs explicit — **stable** |

### Sparse branch

| Key | Type | Meaning |
| --- | --- | --- |
| `sparse_branch_available` | boolean | index has a sparse key — **stable** |
| `sparse_branch_used` | boolean | sparse branch participated — **stable** |
| `sparse_candidates_effective` | number | sparse candidates retained — **experimental** |
| `sparse_rerank_mode` | string | sparse exact-rerank policy — **experimental** |

### Multivector branch

| Key | Type | Meaning |
| --- | --- | --- |
| `multivector_branch_used` | boolean | multivector branch participated — **stable** |
| `multivector_k_effective` | number | resolved multivector candidate budget — **experimental** |
| `multivector_doc_graph_storage_kind` | string | document-node storage kind — **experimental** |

### Fusion

| Key | Type | Meaning |
| --- | --- | --- |
| `branch_count` | number | number of retrieval branches fused — **stable** |
| `branch_fusion_mode` | string | fusion mode (e.g. `rrf`) — **stable** |
| `branch_kinds` | array | the branch kinds fused — **experimental** |

### Native cache

| Key | Type | Meaning |
| --- | --- | --- |
| `native_cache_policy` | string | configured cache policy — **stable** |
| `native_cache_scope` | string | resolved cache scope (`off`/`per_backend`/`shared`) — **stable** |
| `native_cache_used` | boolean | a native cache served this scan — **stable** |
| `native_cache_bytes` | number | resident native cache bytes — **experimental** |

## Related functions

- `turbohybrid_last_build_stats()` — backend-local summary of the last native
  graph build (**diagnostic**).
- `turbohybrid_last_scan_diagnosis()` — derives a single bottleneck `diagnosis`
  label from the stable dense hot-path keys above (**diagnostic**).
- `turbohybrid_simd_capabilities()` — build/architecture SIMD capabilities
  (**diagnostic**).

See the [feature & maturity matrix](feature-matrix.md) for where diagnostics sit
in the overall surface.
