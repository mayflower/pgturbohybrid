# Native sparse (SPLADE) retrieval

> **Status: alpha / experimental.** Native sparse retrieval is new. The on-disk
> sparse format, GUC names, and SQL surface may change between releases. A format
> or version mismatch fails clearly and recommends `REINDEX`. The bundled
> embedding backend is a deterministic **stub**, not a trained model (see
> [Generating sparse vectors](#generating-sparse-vectors)).

`pgturbohybrid` can store and search learned-sparse (SPLADE-style) vectors — high
dimensional, mostly-zero term/weight bags — natively in a `turbohybrid` index,
alongside dense, multivector (ColBERT), and BM25 retrieval.

## The sparse vector type

`turbohybrid_sparse_vector` is a varlena holding a validated, sorted set of
`(term_id, weight)` entries. Builders and inspectors:

```sql
-- From parallel arrays (deduplicates, sorts):
turbohybrid_sparse_vector_from_arrays(term_ids int4[], weights float4[])
    RETURNS turbohybrid_sparse_vector

-- Canonical builder with post-processing options:
turbohybrid_sparse_vector_build(term_ids int4[], weights float4[],
                                options jsonb DEFAULT '{}')
    RETURNS turbohybrid_sparse_vector
-- options: drop_non_positive (bool, default true), deduplicate
-- ('sum'|'max'|'error', default 'sum'), sort (bool, default true), top_k (int),
-- min_weight (float8, default 0.0), normalize ('none', default 'none').

-- Inspectors:
turbohybrid_sparse_vector_term_ids(v)  RETURNS int4[]
turbohybrid_sparse_vector_weights(v)   RETURNS float4[]
turbohybrid_sparse_vector_terms(v)     RETURNS text       -- tsvector-style tokens
turbohybrid_sparse_vector_to_tsvector(v) RETURNS tsvector
turbohybrid_sparse_vector_to_tsquery(v)  RETURNS tsquery
```

Payloads are validated on every entry/scan: term ids must be non-negative,
weights finite and non-negative, and the varlena length must match
`count * sizeof(entry)`. Allocation sizes are checked against `MaxAllocSize`, so a
malformed or hostile payload raises `malformed turbohybrid_sparse_vector payload`
rather than over-reading or overflowing.

## Querying: the `<~*>` operator and `sparse_query`

Sparse retrieval rides the same `turbohybrid_query(...)` payload as the other
branches, ordered by the sparse distance operator `<~*>` (negative inner
product — smaller is better):

```sql
SELECT id
FROM docs
ORDER BY s <~*> turbohybrid_query(
  sparse_query => turbohybrid_sparse_vector_build(ARRAY[2,7]::int4[], ARRAY[1.0,0.5]::float4[]),
  sparse_k => 100,
  final_k => 10)
LIMIT 10;
```

`sparse_query` on an index without a `turbohybrid_sparse_vector` key is rejected
with a clear error. Without an index, `<~*>` still evaluates exactly over a
sequential scan.

## Index shapes

A sparse key is added as an index column with the `sparse_ip_turbohybrid_ops`
opclass. Two shapes are supported:

- **Dense-present sparse** (a sparse key *with* a dense/multivector graph key):
  the sparse inverted index is keyed on the dense graph's node identity.
  ```sql
  CREATE INDEX ON docs USING turbohybrid (
    embedding vector_cosine_turbohybrid_ops,
    s sparse_ip_turbohybrid_ops);
  ```
- **Sparse-primary** (a sparse key with *no* dense graph — sparse-only or
  sparse+BM25): node identity comes from a dedicated node-map chain, and node
  liveness is delegated to heap-tuple MVCC visibility (the executor filters dead
  rows). This is a newer path; an index built before it existed has no node map
  and fails clearly recommending `REINDEX`.
  ```sql
  CREATE INDEX ON docs USING turbohybrid (s sparse_ip_turbohybrid_ops);
  CREATE INDEX ON docs USING turbohybrid (
    s sparse_ip_turbohybrid_ops,
    body bm25_tsvector_turbohybrid_ops);
  ```

A `tsvector`-only index (no sparse or dense key) remains unsupported.

## Quantization and exact rerank

Postings can be stored exactly (f32) or quantized per-term to q16 / q8 to shrink
the index and speed up scoring. q16/q8 are **approximate**; the exact f32 rerank
re-scores the top band from the heap to recover quality:

| reloption | values | meaning |
| --- | --- | --- |
| `sparse_quant_bits` | `0` (f32), `16`, `8` | postings quantization |
| `sparse_quant_mode` | `per_term_linear` | per-term scalar quantization |
| `sparse_postings_encoding` | SoA / varint variants | on-disk posting layout |
| `sparse_block_size` | int | WAND block-max block size |

```sql
CREATE INDEX ON docs USING turbohybrid (s sparse_ip_turbohybrid_ops)
  WITH (sparse_quant_bits = 8);
```

The exact rerank is controlled by `turbohybrid.sparse_rerank`
(`off`/`topk`/`band`/`auto`) and `turbohybrid.sparse_rerank_k`. With f32 postings
the rerank is a no-op (already exact).

## Pruning, caching, and fusion

- **Block-max WAND** (`turbohybrid.enable_sparse_wand`, on by default) prunes
  postings via a block-max directory; it is proven to return the same top-k as an
  exhaustive scan. Pending delta documents force the exact (non-WAND) path until
  compaction.
- **Hot-postings cache** (`turbohybrid.sparse_hot_postings_cache_mb`,
  `turbohybrid.sparse_hot_postings_cache_min_df`) keeps decoded chunks for
  high-df terms backend-locally.
- **Fusion**: sparse fuses with dense and/or BM25 via reciprocal-rank fusion
  (`fusion => 'rrf'`); other fusion modes are dense+BM25/maxsim only.
- **AVX2 / NEON** scatter kernels accelerate scoring (gated by
  `turbohybrid.simd`); they are bit-parity with the scalar kernel.

## Inserts, updates, deletes, and compaction

Inserts/updates append to a sparse **delta** chain so new rows are immediately
searchable (the exact path merges the base + delta). `turbohybrid_sparse_compact(regclass)`
(or auto-compaction past `turbohybrid.sparse_delta_compaction_threshold`) rebuilds
the quantized base from the live heap, clears the delta, and restores WAND.
Deletes are handled by MVCC: vacuumed/updated rows stop matching. Sparse tuple
writes use the same GenericXLog page/WAL patterns as the dense and BM25 branches.

## Scan statistics

`turbohybrid_last_scan_stats()` exposes sparse fields after every scan:
`sparse_branch_used`, `sparse_quant_bits`, `sparse_quant_mode`,
`sparse_score_kernel`, `sparse_used_wand`, `sparse_postings_touched`,
`sparse_candidates_scored`, `sparse_wand_pruned`, `sparse_exact_rerank_count`,
and the hot-cache / delta counters. When the sparse branch is not used they are
zero / false, so a dense-only scan reports no spurious sparse activity.

## GUCs

| GUC | purpose |
| --- | --- |
| `turbohybrid.default_sparse_k` | default sparse candidate budget |
| `turbohybrid.sparse_rerank` | exact rerank mode (`off`/`topk`/`band`/`auto`) |
| `turbohybrid.sparse_rerank_k` | exact rerank band size |
| `turbohybrid.enable_sparse_wand` | block-max WAND pruning |
| `turbohybrid.sparse_hot_postings_cache_mb` | hot-postings cache cap |
| `turbohybrid.sparse_hot_postings_cache_min_df` | min df to cache a term |
| `turbohybrid.sparse_delta_compaction_threshold` | auto-compaction trigger |

## Generating sparse vectors

Sparse vectors are model-agnostic — import any `(term_id, weight)` bag. The
companion `llama_embed` extension provides a sparse output API backed by a
deterministic **stub** (no model download); see
[colbert-llama-extension.md](colbert-llama-extension.md#sparse-splade-output--alpha):

```sql
SELECT llama_embed_sparse('stub', 'red apple', '{"top_k": 64}');
```

For real SPLADE vectors, export `(id, term_ids, weights)` JSONL with
`benchmarks/export_learned_sparse_jsonl.py` (its `external_command` adapter pipes
text to your model) and load it via `turbohybrid_sparse_vector_from_arrays`.

## Benchmarking

The DBpedia ColBERT harness exposes sparse retrieval methods
(`sparse_f32`/`q16`/`q8`/`q8_rerank`, `dense_sparse_rrf`, `dense_sparse_bm25_rrf`)
via `--sparse-benchmark`; see
[benchmarks/README.md](../benchmarks/README.md#native-sparse-splade-retrieval) for
the q8/q16/f32 comparison and a small local (no-download) reproduction.
