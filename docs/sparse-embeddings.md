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
| `sparse_postings_encoding` | `auto`, `offset16_soa`, `varint`, `bitpacked` | on-disk posting layout |
| `sparse_block_size` | int | WAND block-max block size |

### Postings encoding

`sparse_postings_encoding` controls how node-ids are stored within a per-term
chunk; it changes only the physical layout, never the scores (all encodings
return bit-identical rankings and distances):

- **`offset16_soa`** (default via `auto`) — fixed 16-bit offsets in a
  structure-of-arrays layout; the weights array feeds the AVX2/NEON scatter
  scorer directly.
- **`varint`** — LEB128 node deltas; compact, but the node decode and scoring are
  scalar.
- **`bitpacked`** — fixed-width bit-packed node deltas (width = the chunk's
  largest delta, ≤16 bits). The most compact layout, and it still feeds the SIMD
  scatter scorer (the deltas unpack into 16-bit offsets first). On a real
  SPLADE++ NFCorpus q8 index, `bitpacked` was ~21% smaller than `offset16_soa`
  and slightly smaller than `varint`, while retaining SIMD scoring — a good
  default for large, memory-bound indexes.

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

Sparse vectors are model-agnostic — `pgturbohybrid` stores any `(term_id, weight)`
bag and never assumes a particular vocabulary. Three ways to produce them:

1. **A learned-sparse model (SPLADE, uniCOIL, …).** This is the intended use.
2. **The `llama_embed` companion extension**, backed by a deterministic **stub**
   (no model download) — handy for pipeline tests; see
   [colbert-llama-extension.md](colbert-llama-extension.md#sparse-splade-output--alpha):
   ```sql
   SELECT llama_embed_sparse('stub', 'red apple', '{"top_k": 64}');
   ```
3. **Importing JSONL** exported by `benchmarks/export_learned_sparse_jsonl.py`
   (its `external_command` adapter pipes text to your model), loaded with
   `turbohybrid_sparse_vector_from_arrays`.

### Worked example: SPLADE end to end

The key fact that makes SPLADE a drop-in fit: a SPLADE vector is a weight over the
model's **WordPiece vocabulary**, so each non-zero entry's vocabulary id *is* the
`term_id` — no remapping, no shared dictionary to maintain. SPLADE++ uses the
`bert-base-uncased` vocab (ids `0..30521`), exactly the `int4` `term_id` space.

Encode text with the standard SPLADE pooling and load it directly:

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

MODEL = "naver/splade-cocondenser-ensembledistil"   # SPLADE++ CoCondenser-EnsembleDistil
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForMaskedLM.from_pretrained(MODEL).eval()

@torch.no_grad()
def splade(text: str):
    enc = tok(text, truncation=True, max_length=512, return_tensors="pt")
    logits = model(**enc).logits                       # [1, L, V]
    relu = torch.log1p(torch.relu(logits))             # SPLADE activation
    vec = torch.max(relu * enc["attention_mask"].unsqueeze(-1), dim=1).values[0]
    nz = torch.nonzero(vec).squeeze(-1)
    return nz.tolist(), vec[nz].tolist()               # (term_ids, weights)

term_ids, weights = splade("what causes diabetes")
```

```sql
-- term_ids / weights are the SPLADE output above (vocabulary id = term_id).
INSERT INTO docs (doc_id, s)
VALUES ('d1', turbohybrid_sparse_vector_from_arrays(:term_ids::int4[], :weights::float4[]));

CREATE INDEX ON docs USING turbohybrid (s sparse_ip_turbohybrid_ops)
  WITH (sparse_quant_bits = 8);          -- 8-bit postings; 0 = exact f32

-- Retrieve: encode the query the same way and order by the sparse distance.
SELECT doc_id
FROM docs
ORDER BY s <~*> turbohybrid_query(
  sparse_query => turbohybrid_sparse_vector_from_arrays(:q_term_ids::int4[], :q_weights::float4[]),
  sparse_k => 1000, final_k => 10)
LIMIT 10;
```

That is the entire pipeline: encode → `from_arrays` → index → `<~*>`. The inner
product `<~*>` computes is exactly SPLADE's scoring function (`Σ q_t · d_t` over
shared terms).

## Validation on real data (BEIR NFCorpus)

The native sparse path was validated against SPLADE++ CoCondenser-EnsembleDistil
on **BEIR NFCorpus** (3,633 documents, 323 test queries) — the same model behind
pyserini's `beir-v1.0.0-nfcorpus.splade-pp-ed` prebuilt index. Documents and
queries were encoded as above, loaded into a sparse-primary index, and retrieved
through `<~*>`. Results were checked against (a) an exact brute-force SPLADE
dot-product ranking and (b) the published nDCG@10:

| method | nDCG@10 | recall@100 | top-10 overlap vs exact SPLADE |
| --- | --- | --- | --- |
| brute-force SPLADE (reference) | 0.3549 | 0.2891 | — |
| `sparse_f32` (exact) | 0.3546 | 0.2891 | 0.9997 |
| `sparse_q16` | 0.3549 | 0.2891 | 1.000 |
| `sparse_q8` | 0.3549 | 0.2891 | 1.000 |
| `sparse_q8` + exact rerank | 0.3549 | 0.2891 | 1.000 |
| *published SPLADE++ ED (pyserini/Anserini)* | *0.3473* | — | — |

The f32 scan reproduces the exact SPLADE ranking (top-10 overlap 0.9997; the
0.03% is one tie-broken document), q16/q8 are lossless on this dataset, and
nDCG@10 lands on the published reference (the ~0.008 gap is our own encoding plus
nDCG-formula nuances, not a retrieval difference). NFCorpus's low recall@10 is
expected — it averages ~50 relevant documents per query, so nDCG@10 is the
headline metric.

> The pyserini/Anserini prebuilt indexes themselves are Lucene *impact* indexes
> (a JVM is required to read them) and the only pre-encoded corpus is a single
> ~45 GB tar for all of BEIR, so this validation reproduces the SPLADE++ ED
> pipeline directly on the 2.4 MB NFCorpus rather than consuming the prebuilt
> index.

## Benchmarking

The DBpedia ColBERT harness exposes sparse retrieval methods
(`sparse_f32`/`q16`/`q8`/`q8_rerank`, `dense_sparse_rrf`, `dense_sparse_bm25_rrf`)
via `--sparse-benchmark` (or by naming them in `--methods`); see
[benchmarks/README.md](../benchmarks/README.md#native-sparse-splade-retrieval) for
the q8/q16/f32 comparison and a small local (no-download) reproduction.
