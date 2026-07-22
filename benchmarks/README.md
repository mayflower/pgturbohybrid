# pgturbohybrid Benchmarks

Benchmark comparisons for `pgturbohybrid` must use real embedding datasets and
documented relevance metrics. The benchmark setup treats pgvector as an
upstream dependency and installs `pgturbohybrid` as a separate extension. It
does not require a patched pgvector checkout.

Public benchmark explanations live under `docs/benchmarks/`. This directory is
for reproducible tooling, acceptance thresholds, and developer benchmark
helpers. Developer-only workflows are documented in `benchmarks/dev/README.md`.

## Reporting contract

Quality is the first release gate. Every comparative report publishes the fixed
seed and dataset/query provenance, recall@k and nDCG@k, latency p50/p95/p99,
QPS, cold and warm runs at 1/8/32 clients, build time, index size, peak memory,
and WAL bytes. Raw per-query samples remain CI artifacts; summaries use the
median of repeated runs. A named baseline identifies its version, commit,
configuration, hardware, PostgreSQL version, and cache state. Do not publish a
speed claim without the corresponding quality and provenance fields.

`benchmarks/tools/check_recall_gate.py` is the deterministic PR gate. It checks
dense and high-recall quality, native-index/kernel use, bounded candidates, and
absence of dead results or linear fallback. It deliberately has no wall-clock
threshold; scheduled performance jobs own repeated latency regression analysis.

## Nix integration

The Nix flake separates deterministic development checks from benchmark
experiments:

- `nix flake check` stays small and build-oriented. It builds the extension, the
  wrapped PostgreSQL package, the pgvector-master variant, and a scalar
  `SIMD_BUILD=none` variant.
- `nix develop` provides the local PostgreSQL cluster, SQL regression commands,
  and deterministic synthetic benchmark helpers.
- `nix develop .#bench` adds `uv` and common Python data packages for real-data
  benchmark scripts.

Useful commands:

```sh
nix develop
th-test
th-bench-retrieval-quality
th-bench-profile-grid
th-bench-tune-profile

nix develop .#bench
th-bench-concurrent-dense --help
th-bench-dbpedia-colbert --help
FIQA_DATASET=/path/to/fiqa th-bench-fiqa-quick
```

The Nix `th-bench-fiqa-quick` wrapper defaults to the separate
`pgturbohybrid_fiqa_quick` database. Use `FIQA_PGDATABASE=...` to choose another
benchmark database, or set `PGDATABASE=...` explicitly when you want the wrapped
script to use that database.

The Nix `th-bench-dbpedia-colbert` wrapper defaults to the separate
`pgturbohybrid_dbpedia_colbert` database and starts a separate llama-backed
PostgreSQL cluster under `.nix-dev/pg17-pgvector-v0.8.2-colbert-llama/` on
port `55433` by default. It uses the smaller
`johannhartmann/SauerkrautLM-Multi-ColBERT-15m-GGUF` validation model by
default when `--model-path` or `PG_COLBERT_LLAMA_TEST_MODEL` points at the GGUF
file. The default run is smoke-sized (`--max-docs 1000 --max-queries 32`) and
measures PostgreSQL multivector generation, persisted generated multivector
insertion/storage, pgturbohybrid index build, serial retrieval, 8x parallel
retrieval, and BEIR DBpedia qrels metrics for
`pgturbohybrid_colbert_multivector_query_only`:

```sh
nix develop .#bench
DBPEDIA_DATASET=/path/to/qdrant-dbpedia \
BEIR_DBPEDIA_DATASET=/path/to/beir-dbpedia \
PG_COLBERT_LLAMA_TEST_MODEL=/path/to/sauerkraut-modern.gguf \
  th-bench-dbpedia-colbert
```

The benchmark records `--colbert-model-name` in the JSON `model` section and
sets `turbohybrid.multivector_model_name` for PostgreSQL-side validation and
index-stat provenance. `--expected-dim` defaults to `auto`, resolving through
the same known late-interaction profiles used by the extension registry:
ColBERTv2 is 128d, AnswerAI ColBERT small is 96d, Jina-ColBERT-v2 supports
128/96/64d variants, GTE/Reason ModernColBERT profiles are 128d, the
Sauerkraut validation pair is 128d, and ColPali-like visual profiles default to
128d with processor-specific patch counts. For an unregistered export, pass
`--expected-dim <n>` explicitly and use the report metadata to preserve
reproducibility.

Use `--methods pgturbohybrid_colbert_multivector_query_only,pgturbohybrid_colbert_multivector_rrf`
to include BM25 RRF fusion, and use `--max-docs 0 --max-queries 0 --final-k 100
--quality-k 100` for an opt-in full-scale recall@100 run.

Add `--admission-debug` to run an exact-vs-candidate admission report after the
normal retrieval methods. The report exact-scans each encoded query for
`--admission-k` documents, sweeps query-only multivector retrieval across
`--admission-budget-sweep` document candidate budgets, enables bounded
`turbohybrid.multivector_debug_admission = trace`, and writes a top-level
`admission_debug` JSON section with per-query and aggregate fields for exact
top-1 admission, exact top-10 admission recall, raw subvector hits, unique docs,
MaxSim updates, retained candidates, exact rerank docs, memory estimate, and
latency. The aggregate `admission_by_budget` records use the DBpedia gate field
names `exact_top1_admitted`, `exact_top10_admission_recall`, `exact_top1_rank`,
`latency`, `docs_scored`, `graph_edges_visited`, and `exact_rerank_docs`.
`exact_top1_admission` remains as a backward-compatible alias. Normal runs
without `--admission-debug` keep the existing JSON shape.

Use `--multivector-candidate-source graph` for the normal token-node ANN path
and `--multivector-candidate-source exact_token_scan` for the developer oracle
that scores every stored token node per query token before the same document
aggregation and exact rerank. Include
`--methods pgturbohybrid_colbert_multivector_query_only,pgturbohybrid_colbert_multivector_exact_scan`
to compare token-candidate retrieval against exact document MaxSim scan; rerun
with `--multivector-candidate-source exact_token_scan --admission-debug` to
separate graph/token ANN miss from structural token-top-K admission loss. The
admission budget records include `candidate_source` and
`exact_token_scan_nodes_scored` when the oracle is active.

Two document-level validation modes are available:
`--multivector-candidate-source exact_doc_scan` and
`--multivector-candidate-source doc_graph_prototype`. `exact_doc_scan` is the
exact document MaxSim oracle. `doc_graph_prototype` is intentionally
heap-backed until index-resident document graph storage exists; its admission
records include `doc_graph_prototype_enabled`, `doc_graph_docs_scored`,
`doc_graph_edges_visited`, `doc_graph_candidates`,
`doc_graph_heap_fetches`, and `doc_graph_warning`. Some document-level paths do
not emit per-document trace entries. Admission records set
`admission_inferred_from_result_docs` when `exact_doc_scan`,
`doc_graph_prototype`, forced/plain fallback, production `document_nodes`,
`proxy_vector`, or `quantized_inverted_experimental` infer admission from exact
top documents that appear in the final exact-reranked result list. This proves
the document was admitted and retained for exact rerank, but it does not expose
the pre-rerank candidate rank; that field remains `null` unless a trace entry
provides it.

In `turbohybrid.multivector_plain_fallback = auto`, document-node indexes also
consider the expanded graph candidate limit
`rescore_k * multivector_doc_graph_oversampling` when deciding whether a scan
has become near-exhaustive. If that expanded limit crosses
`turbohybrid.multivector_plain_fallback_candidate_fraction`, the benchmark
reports `plain_fallback_reason = document_node_candidate_fraction` and uses the
single-pass heap exact scan instead of scoring every sidecar document and then
heap-reranking a large prefix. Explicit `proxy_vector`, `centroid_lite`, and
`quantized_inverted_experimental` sources keep their own candidate paths so the
grid can still measure those branches directly.

The production document-node path adds the persisted
index option `multivector_graph = token_nodes | document_nodes` and the
`turbohybrid_index_stats()` field `multivector_graph_mode`. Explicit
`document_nodes` indexes store one graph node per heap document plus an explicit
document-node storage tier. `f32`, `f16`, and `sq8` include a full multivector
sidecar. `proxy_only` includes proxy graph/docmap data without the full
multivector sidecar. `centroid_only` includes proxy graph/docmap data plus
k-means centroid/posting payloads without the full multivector sidecar; when an
external `quantized_inverted_experimental` codebook is selected at build time it
can also persist compact quantized posting/codeword payloads for the explicit
compact experimental path.
Build-time edge selection uses
symmetrized document MaxSim. Incremental document-node inserts use the same
document-level scorer for candidate collection, neighbor selection, and
reciprocal pruning; the representative vector stored in the dense graph is not
used as a silent graph-link fallback. Non-exhaustive scans traverse document graph
adjacency and score visited candidates with the selected document sidecar scoring
storage before heap rerank; near-exhaustive scans use the exact float32 sidecar
scan as the correctness reference. Set
`turbohybrid.multivector_doc_storage = f32 | f16 | sq8 | proxy_only |
centroid_only` or pass
`--multivector-doc-storage f32|f16|sq8|proxy_only|centroid_only` in the DBpedia
benchmark. Control
whether the document sidecar is reused as resident cache or loaded through
shared-buffer page reads with
`turbohybrid.multivector_doc_storage_cache = auto | resident | paged` or
`--multivector-doc-storage-cache auto|resident|paged`. For plain `proxy_vector`
document-node serving, graph admission stays on fixed-dimensional proxy data.
Full-sidecar indexes touch full document multivectors only for bounded exact
rerank or explicit sidecar-scoring modes. `proxy_only` and `centroid_only`
indexes have `full_multivector_sidecar_available = false`; their bounded exact
rerank fetches heap multivectors. `centroid_only` supports `centroid_lite` and
the guarded compact `quantized_inverted_experimental` path when
`quantized_inverted_sidecar_available = true`; unsupported full-sidecar
candidate sources fail with REINDEX guidance instead of falling back. Other
document-node scan
modes may keep low-latency profile scans resident when the sidecar fits
`turbohybrid.native_cache_max_mb`; explicit `resident` still forces loaded
sidecar storage, and explicit `paged` is useful for cold-sidecar measurements
because graph adjacency can remain native-cached while document sidecar page
reads are counted per scan. The warning is
`document_node_f32_sidecar_graph_traversal`,
`document_node_f16_sidecar_graph_traversal`,
`document_node_sq8_sidecar_graph_traversal`, or
`document_node_f32_sidecar_exact_scan`. Scan stats report
`multivector_doc_graph_storage_kind`, `multivector_doc_storage_kind`,
`proxy_only_index`, `centroid_only_index`,
`full_multivector_sidecar_available`, `centroid_sidecar_available`,
`quantized_inverted_sidecar_available`, `exact_rerank_source_supported`,
`multivector_doc_graph_quantized_scores`, and
`multivector_doc_graph_rescore_source`; sidecar cache stats report
`multivector_doc_sidecar_cache_mode`,
`multivector_doc_sidecar_pages_read`,
`multivector_doc_sidecar_cache_hits`,
`multivector_doc_sidecar_cache_misses`, and
`multivector_doc_sidecar_bytes_touched`; paged-access benchmarks also summarize
`multivector_doc_sidecar_vectors_loaded` to show how many document multivectors
were reconstructed on demand. Compatibility sidecar byte/page totals can include
cache construction. Document-node proxy scans additionally report
`sidecar_cache_build_bytes`, `sidecar_cache_build_pages_read`,
`sidecar_cache_build_time_us`, `sidecar_query_bytes_touched`,
`sidecar_query_pages_read`, `sidecar_query_vectors_loaded`, and
`sidecar_query_time_us` to separate native cache construction from per-query
sidecar materialization/touches after the cache is available. Proxy candidate
reports include `proxy_candidate_limit_effective` and
`proxy_candidate_limit_source`, which commonly identifies `graph_ef_search` as
the reason a requested candidate budget is not fully returned. Exact rerank uses
the original multivectors for final SQL ordering.
Incremental insert diagnostics are available through
`turbohybrid_last_scan_stats()`:
`multivector_doc_graph_insert_full_maxsim_edges`,
`multivector_doc_graph_insert_representative_fallbacks`, and
`multivector_doc_graph_insert_pairs_scored`.
Keep `doc_graph_prototype` in benchmark grids when you need the heap-backed
validation mode for comparison.

Document-node indexes can reduce stored document-token count with opt-in
index-time pooling:

```sh
python benchmarks/dbpedia_colbert_multivector.py \
  --multivector-graph document_nodes \
  --multivector-token-pooling greedy_cosine \
  --multivector-token-pooling-target-ratio 0.5 \
  --multivector-token-pooling-min-tokens 16 \
  --multivector-doc-storage f16
```

Pooling applies only to document tokens before they are stored in the
document-node sidecar; query tokens remain unpooled. Last-scan stats expose
`multivector_tokens_original`, `multivector_tokens_pooled`, and
`multivector_token_pooling_ratio` so storage/latency/quality runs can verify
the effective reduction. The admission grid rebuilds document-node indexes for
`--document-node-pooling-grid`, whose default is
`off:1.0,greedy_cosine:0.75,greedy_cosine:0.5,greedy_cosine:0.33`, and combines
those ratios with `--document-node-storage-grid f32,f16,sq8`.

`--multivector-candidate-source document_nodes` is an explicit alias for the
normal document-node graph path and requires `--multivector-graph document_nodes`.

`--multivector-candidate-source proxy_vector` is a document-node prototype that
uses the persisted fixed-dimensional proxy encoder as the single-vector
TurboQuant graph key for admission, then exact-reranks admitted documents with
full MaxSim. Use it with `--multivector-graph document_nodes`.
`--multivector-proxy-encoder normalized_mean|first_token|centroid_mean|mean_pool|max_pool|random_projection_fde|learned_projection_v1`
selects the proxy encoder for the index build; `normalized_mean` is the default
document proxy, `centroid_mean` requires `--multivector-centroids kmeans`,
`mean_pool` remains a compatibility encoder, and `learned_projection_v1` requires
`--multivector-learned-projection-path`. The admission grid can compare
practical built-in encoders with
`--document-node-proxy-encoder-grid normalized_mean,centroid_mean,max_pool,random_projection_fde`.
Proxy scans report `proxy_encoder_kind`, `proxy_candidates`,
`proxy_top1_admission`, and `proxy_exact_rerank_docs`.

`--multivector-candidate-source centroid_lite` is the experimental PLAID-lite
admission path. Build the index with `--multivector-centroids kmeans`; it can
run on `token_nodes` as a compatibility prefilter and on `document_nodes` as a
document-local centroid prefilter backed by persisted centroid sidecar tuples.
Leave `--multivector-centroid-count auto` or set an integer per-document
centroid count. The benchmark reports
`centroid_lists_visited`, `centroid_docs_touched`, `centroid_pruned_docs`,
`centroid_postings_touched`, `centroid_postings_skipped`,
`centroid_posting_limit_per_token`, `centroid_posting_cap_strategy`, and
`centroid_candidates`. Final ranking still uses exact MaxSim rerank over the
admitted original multivectors. This
branch persists per-document centroid vectors, residual summaries, and codeword
posting tuples in the multivector docmap sidecar; scans load those posting
lists, validate centroid sidecar entries only for touched posting documents,
and avoid per-query posting-list rebuilds. For opt-in bounded admission
experiments, set `--multivector-centroid-lite-max-postings-per-token <n>`.
The default `0` preserves full posting-list admission. Positive caps use
deterministic midpoint-spaced sampling across each posting list and report
`centroid_posting_cap_strategy = uniform_stride`.
Set `--multivector-centroid-lite-posting-selection score_topk` to use
payload-sorted posting prefixes on freshly rebuilt centroid indexes. The
benchmark also exposes `_score_topk_codeword_maxsim` profile rows, which set
`turbohybrid.multivector_centroid_lite_candidate_scoring = codeword_maxsim`
for PLAID-style codeword MaxSim over selected posting matches without loading
document-centroid vectors. `_score_topk_docmaxsim` profile rows set
`turbohybrid.multivector_centroid_lite_candidate_scoring = doc_centroid_maxsim`
so touched documents are pre-ranked by approximate MaxSim over persisted
document centroids before the bounded exact heap MaxSim rerank. Both modes are
admission experiments only; final SQL ordering remains exact MaxSim over
retained heap documents. The candidate-source focus grid can generate
`_threshold_NNN` and `_drop_NNN` rows for both `_score_topk_codeword_maxsim`
and `_score_topk_docmaxsim`; these rows reuse the same physical index and only
change scan-time posting-list filtering.
Current 2k DBpedia smoke evidence shows the exact rerank cap is honored
(`exact_rerank_docs = 100` for `--serving-exact-rerank-k 100`), but
`centroid_lite` still visits one posting list per query token and can touch
nearly every document. Treat high `centroid_docs_touched` or
`docs_scored_near_table_size` warnings as candidate-source work, not MaxSim
rerank cost.
Plain fallback is bypassed when this explicit source is selected, so missing
`--multivector-centroids kmeans` fails instead of producing
substitute fallback numbers.

`--multivector-candidate-source quantized_inverted_experimental` names the
research-only ColBERTSaR-style branch. It currently requires
`--multivector-graph document_nodes` and uses persisted experimental codeword
posting tuples before exact MaxSim rerank. Scans validate document vectors only
for touched postings, so latency numbers are not inflated by an up-front
all-document validation pass. Plain fallback is bypassed when this source is
selected so the reported numbers cannot come from a substitute candidate path.
It remains outside normal CI and default single-run retrieval because the
codebook/posting sidecar has no production compatibility promise; the opt-in
document-node admission grid includes it so research comparisons produce
explicit experimental stats without falling through to another candidate source.
The pure-ColBERT candidate-source focus can also run the explicit external
codebook variant. Build a sampled experimental codebook from the already
imported ColBERT document-token vectors first:

```bash
python benchmarks/dbpedia_colbert_multivector.py \
  --database pgturbohybrid_dbpedia_colbert_1m_colbert \
  --reuse-data \
  --document-node-colbert-quantized-codebook-build-only \
  --quantized-inverted-codebook-output \
    .nix-dev/tmp/dbpedia-colbert-1m-quantized-codebook.txt \
  --quantized-inverted-codebook-size 256 \
  --quantized-inverted-codebook-sample-docs 10000 \
  --quantized-inverted-codebook-sample-tokens 100000 \
  --quantized-inverted-codebook-kmeans-iterations 20 \
  --output .nix-dev/tmp/dbpedia-colbert-1m-quantized-codebook.json \
  --markdown-output .nix-dev/tmp/dbpedia-colbert-1m-quantized-codebook.md
```

Then point the experimental candidate-source focus at that artifact:

```bash
python benchmarks/dbpedia_colbert_multivector.py \
  --database pgturbohybrid_dbpedia_colbert_1m_colbert \
  --reuse-data \
  --document-node-colbert-candidate-source-focus \
  --include-quantized-inverted-experimental \
  --multivector-quantized-inverted-codebook-path \
    .nix-dev/tmp/dbpedia-colbert-1m-quantized-codebook.txt
```

With an external codebook and `--include-quantized-inverted-experimental`, the
candidate-source focus can select the experimental preset
`quantized_inverted_external_centroid_only_precompact_topk_8192_docid_rk512_topk`,
which uses candidate budget `8192` and exact heap MaxSim rerank K `512`. This
profile is still experimental. It uses the external codebook only when a real
codebook file is supplied, includes the codebook source/path/top-m in the
physical index signature, keeps final SQL ordering as exact heap MaxSim over
the retained candidates, and enables precompact top-k/docId-order/blocked
query-codeword scoring. The `centroid_only` compact variant writes centroids plus
quantized posting/codeword payloads and does not store a full document
multivector sidecar; use explicit
`--document-node-colbert-candidate-source-profiles` and
`--quantized-inverted-posting-caps` / `--quantized-inverted-probes` only when
running diagnostics instead of the locked comparison path. Score-bound pruning
remains an explicit experimental diagnostic; it is not part of the default
candidate-source focus profile.
The sampled codebook builder does not build an index, run retrieval, or train
inside PostgreSQL; it is an offline benchmark utility for producing the current
`pgturbohybrid_quantized_inverted_codebook_v1` text format.

#### Quantized-inverted acceptance gate

Use the acceptance report before renaming or promoting an opt-in
`quantized_inverted_experimental` profile as the current default-quality
candidate. The report reads existing benchmark artifacts only; it does not
connect to PostgreSQL, build indexes, or run retrieval:

```bash
python benchmarks/dbpedia_colbert_multivector.py \
  --quantized-inverted-default-quality-acceptance-report \
  --precompact-grid-json \
    .nix-dev/tmp/dbpedia-colbert-50k-precompact-docorder-25q.json \
  --query-codeword-grid-json \
    .nix-dev/tmp/dbpedia-colbert-50k-precompact-kernel-25q.json \
  --compact-layout-grid-json \
    .nix-dev/tmp/dbpedia-colbert-50k-precompact-docorder-25q.json \
  --exact-rerank-grid-json \
    .nix-dev/tmp/dbpedia-colbert-50k-exact-rerank-focus-25q.json \
  --output .nix-dev/tmp/dbpedia-colbert-50k-quantized-acceptance-report.json \
  --markdown-output \
    .nix-dev/tmp/dbpedia-colbert-50k-quantized-acceptance-report.md
```

The same paths can be supplied through `PRECOMPACT_GRID_JSON`,
`QUERY_CODEWORD_GRID_JSON`, `COMPACT_LAYOUT_GRID_JSON`, and
`EXACT_RERANK_GRID_JSON`. The hard gates keep this path experimental and
pure-ColBERT: the candidate source must be
`quantized_inverted_experimental`, final ranking must be exact heap MaxSim,
BM25 and learned sparse must be inactive, top-10 exact admission must be at
least `0.80`, top-1 exact admission at least `0.94`, Recall@10 and NDCG@10 may
drop by at most `0.01` from the baseline row, p95 must be at most `0.75x` the
baseline p95, compact scoring must normally touch at most `6000` documents,
and generated acceptance artifacts must stay under `.nix-dev/tmp/`.

If the hard gates fail, keep the row as benchmark evidence only. Do not promote
it as a safe serving profile just because it is faster than another candidate
source on one artifact.

#### Pure-ColBERT benchmark contract

Pure-ColBERT benchmark modes must keep BM25, learned sparse, SPLADE/SPLATE,
and hybrid fusion out of the profile set. They may compare proxy, centroid, and
quantized-inverted candidate sources, but all rows must keep final SQL ordering
as exact heap MaxSim over retained documents.

Use this checklist when reading a ColBERT artifact:

- `pure_colbert_only = true` means the run did not intentionally enable lexical
  or sparse rescue.
- `candidate_source` identifies the admission branch:
  `proxy_vector`, `document_nodes`, `centroid_lite`, or
  `quantized_inverted_experimental`.
- `profile` and `physical_index_signature` identify whether the row uses
  `proxy_only`, `centroid_only`, full `f16`/`sq8` sidecar storage, pooling, or
  an external codebook.
- `recall@10`, `ndcg@10`, and `mrr@10` are qrel quality metrics. They do not
  prove exact MaxSim admission unless an oracle is present.
- `exact_top1_admission_rate` and `exact_top10_admission_recall` are available
  only for exact-admission runs or sampled exact-oracle runs. Artifacts must
  label the evidence as full or sampled.
- `exact_admission_available = false` is expected for BEIR-quality-only and
  candidate-source-focus runs without `--oracle-input`.
- Zero `recall@10` or zero `ndcg@10` rows are negative controls when qrels are
  available; they should not be recommended as serving profiles even when p95
  is low.

Candidate-source work counters explain whether time is spent in candidate
admission or in exact rerank:

| Branch | Key counters |
| --- | --- |
| Proxy | `proxy_candidates_returned`, `proxy_candidate_limit_effective`, `proxy_candidate_limit_source`, `proxy_graph_nodes_visited`, `proxy_graph_edges_visited`, `proxy_vector_score_time_us` |
| Centroid-lite | `centroid_lists_visited`, `centroid_postings_touched`, `centroid_docs_touched`, `centroid_docs_touched_ratio`, `centroid_candidates`, `centroid_candidate_scoring` |
| Quantized inverted | `quantized_inverted_lists_visited`, `quantized_inverted_postings_touched`, `quantized_inverted_docs_scored`, `quantized_inverted_candidates`, `quantized_inverted_precompact_pruned_docs` |
| Compact scoring | `quantized_inverted_compact_kernel`, `quantized_inverted_compact_score_us`, `quantized_inverted_compact_docs_scored`, `quantized_inverted_compact_pairs_evaluated`, `quantized_inverted_compact_pairs_skipped` |
| Exact rerank | `multivector_exact_rerank_docs`, `multivector_exact_rerank_pairs`, `multivector_exact_kernel`, `multivector_exact_maxsim_rerank_time_us`, `multivector_exact_heap_fetch_time_us` |
| Storage/cache | `multivector_doc_storage_kind`, `proxy_only_index`, `centroid_only_index`, `full_multivector_sidecar_available`, `multivector_doc_sidecar_bytes_touched`, `multivector_doc_sidecar_pages_read` |

Interpretation rules:

- A fast proxy-only row with weak qrel quality is a latency baseline, not a
  quality candidate.
- A centroid-lite row that touches a large fraction of documents is still too
  broad, even if exact rerank time is small.
- A quantized-inverted row is experimental unless it passes the admission,
  quality, latency, and work-counter gates in the acceptance report.
- A qrel-quality run without exact-oracle admission can reject bad profiles but
  cannot prove safe admission.

### x00k document-node serving selection

Use `--document-node-serving-grid` when the goal is to choose a practical
document-node serving profile for 10k to x00k ColBERT corpora. This preset is a
compact production-oriented grid, not the exhaustive admission research grid. It
compares named profiles for `proxy_vector`, the explicit `document_nodes`
source, `normalized_mean`, `centroid_mean`, `centroid_lite`, token pooling, and
`sq8` sidecar storage. It leaves `quantized_inverted_experimental` out by
default; add `--document-node-serving-grid-include-experimental` only for
research runs. Add `--document-node-serving-grid-include-proxy-encoders` only
when explicitly comparing additional proxy encoders; it adds
`proxy_max_pool_f16` and `proxy_random_projection_fde_f16` to the known profile
set without changing the default compact grid. Use
`--document-node-serving-grid-pure-dense-proxy-focus` when the comparison should
stop treating `normalized_mean` as the main short-term ColBERT proxy admission
path and focus only on stronger pure-dense proxy encoders. That mode compares
`proxy_normalized_mean_proxy_only`, `proxy_max_pool_proxy_only`, and
`proxy_random_projection_fde_proxy_only` over candidate budgets `800,1600,3200`
and exact rerank `k = 100,400,800`; if
`--multivector-learned-projection-path` is supplied, it also includes
`proxy_learned_projection_v1_proxy_only`. Missing learned-projection weights
fail before index build. The mode excludes BM25 rescue, learned-sparse rescue,
hybrid fusion, and in-PostgreSQL training; final retained candidates remain
exact MaxSim ranked. Add
`--document-node-serving-grid-include-bm25-rescue` only for focused
candidate-admission experiments that should compare lexical rescue against the
dense-only profiles; it adds `proxy_normalized_mean_f16_bm25_rescue` and
`centroid_mean_f16_bm25_rescue`, builds the required BM25 key, and still uses
exact MaxSim for final dense ordering. When both BM25 rescue and proxy-encoder
variants are enabled, it also adds `proxy_max_pool_f16_bm25_rescue` so the
stronger `max_pool` proxy baseline can be tested with the same lexical rescue
without widening the default BM25 rescue set. Those rescue profiles also pass
`text_query` in the otherwise dense-only benchmark query so the BM25 admission
branch is actually exercised. The report preserves rescue cap accounting:
effective candidate limit, combined dense/rescue pool size, limit reason,
retained count, and exact-reranked rescue candidates. The serving
recommendation carries those aggregates forward instead of relying only on a
single sampled scan, so rejected rescue profiles can show whether they were
limited by rerank depth, document candidate caps, or lexical underfill.
Rejected serving profiles also include deterministic
`admission_improvement_hints`, for example `proxy_candidates_capped_by_search_ef`,
`bm25_rescue_limited_by_exact_rerank_k`, or
`learned_sparse_partial_coverage`, so the next benchmark can target the actual
admission bottleneck instead of rerunning the whole grid blindly. When a row
exhausts the admitted band and still misses the top-10 admission threshold, the
report also labels the next likely direction: `try_max_pool_or_centroid_mean_proxy`,
`try_centroid_mean_proxy`, `try_entry_sample_sweep`,
`try_sparse_rescue_or_centroid_lite`,
`try_balanced_candidate_reservoirs`, or
`try_bm25_or_learned_sparse_rescue`.
Add `--document-node-serving-grid-include-entry-samples` only for focused
scan-time graph-entry admission experiments. It adds paired profile rows such
as `proxy_normalized_mean_f16_entry_sample_032` and
`centroid_mean_f16_entry_sample_032`, using counts from
`--document-node-serving-grid-entry-sample-counts` (default `32,128`). These
rows reuse the same physical document-node index as the baseline and only set
`turbohybrid.multivector_doc_graph_entry_sample_count` for retrieval. The
serving-grid JSON, Markdown, recommendation tables, and
`candidate_source_deltas` preserve the effective entry-sample count so this can
be judged as candidate-admission evidence before changing index formats or
defaults.
Add `--document-node-serving-grid-include-reservoirs` only for focused
candidate-admission experiments. Reservoir rows set
`turbohybrid.multivector_candidate_reservoirs = balanced`. For explicit
document-node `proxy_vector` scans, reservoir mode chooses the exact rerank band
from the proxy candidate list using a score-prefix plus deterministic
rank-spread sample: `conservative` keeps most of the proxy-ranked prefix, while
`balanced` gives more of the band to rank-spread exploration. These rows are
scan-time candidate-selection experiments, not default serving profiles, and
final ranking remains exact MaxSim over retained candidates. Document-node
proxy rows are valid evidence only when scan stats report
`multivector_reservoirs_enabled = true` and nonzero reservoir union docs;
otherwise the report marks
`candidate_reservoirs_not_executed`, rejects the row as safe serving evidence,
and keeps it out of the recommendation winners.
Add `--document-node-serving-grid-include-learned-sparse-rescue` only when
external learned-sparse features are available through
`--learned-sparse-doc-jsonl` and `--learned-sparse-query-jsonl`; it adds
`proxy_normalized_mean_f16_learned_sparse_rescue` and
`centroid_mean_f16_learned_sparse_rescue`, builds the sparse lexical key, and
uses learned-sparse postings only for admission before exact MaxSim final
ordering. When proxy-encoder variants are also enabled, it adds
`proxy_max_pool_f16_learned_sparse_rescue` for the same focused proxy-quality
comparison. When learned-sparse rescue, proxy-encoder variants, and reservoirs
are all enabled, the grid also adds
`proxy_max_pool_f16_reservoir_balanced_learned_sparse_rescue` to test whether the
fast `max_pool` proxy can become safe with both rescue mechanisms. Use
`--document-node-serving-grid-learned-sparse-focus` for the compact comparison
set around this question. Mixed BM25 and learned-sparse runs build separate
physical index groups: BM25 rescue uses `body_tsv`, while learned-sparse rescue
uses `learned_sparse_tsv`. The report records learned-sparse document/query
coverage after JSONL import; partial coverage is flagged and should be treated
as candidate-source plumbing evidence, not production serving evidence.
JSONL files whose filename or metadata indicates `hash`, `toy`, `sample`, or
`plumbing_only` are also classified as plumbing-only even when their row counts
match the loaded docs and queries. The local hash sparse JSONL fixtures validate
benchmark wiring only; they must not be used to promote learned-sparse rescue to
safe serving evidence. A real learned-sparse artifact should include either a
metadata header row or sidecar manifest with stable provenance, for example:

```json
{
  "kind": "metadata",
  "feature_source": "splade",
  "feature_version": "feature-generator-v1",
  "model_name": "example/sparse-model",
  "model_checksum": "sha256:...",
  "plumbing_only": false,
  "expected_doc_count": 10000,
  "expected_query_count": 100
}
```

If no metadata is present but coverage is complete, the run remains reportable
and eligible as benchmark evidence, but the report adds
`feature_provenance_unknown`. Treat that as a follow-up requirement before
turning the profile into serving guidance.

Production readiness for learned-sparse rescue requires all of the following
evidence before it is used as a serving profile:

- Document JSONL coverage is 100% for the served corpus.
- Query JSONL coverage is 100% for the benchmark queries.
- The JSONL provenance is real learned-sparse evidence, not hash/toy/sample
  plumbing.
- The `term_id` vocabulary is stable across document and query feature files.
- The feature-generator version is recorded in the benchmark artifact.
- The feature-generator model name and checksum are recorded when available.
- `learned_sparse_tsv` is built and indexed; learned-sparse profiles must not
  silently use `body_tsv`.
- The query path uses `turbohybrid_sparse_vector_to_tsquery(q.learned_sparse)`,
  not the web-search text fallback.
- Final ranking remains exact MaxSim over retained multivector candidates.
- Rescue cap accounting is present, including learned-sparse candidates,
  retained-for-MaxSim counts, and branch latency.
- p95 latency and admission thresholds pass on the target corpus, not only on a
  tiny plumbing smoke.

Before adding or updating any SQL-visible
`turbohybrid.multivector_serving_profile` mapping, validate the evidence
artifact explicitly:

```bash
SERVING_GRID_JSON=.nix-dev/tmp/dbpedia-colbert-serving-validation-10k.json \
nix --extra-experimental-features 'nix-command flakes' develop --command \
  python benchmarks/dbpedia_colbert_multivector.py \
    --validate-serving-profile-guc-evidence
```

The gate refuses smoke-only artifacts, undersized runs, incomplete
learned-sparse coverage, experimental profiles, missing admission or qrel
quality metrics, and slow-path warnings. `ALLOW_UNSAFE_PROFILE=1` is the only
override path and must be present explicitly in the prompt that requests the
GUC change.

Add `--document-node-serving-grid-include-centroid-lite-caps` only for focused
centroid-lite pruning experiments. It appends capped profile names such as
`centroid_lite_f16_cap_016`, `centroid_lite_f16_cap_032`, and
`centroid_lite_f16_cap_064` using the scan-time
`turbohybrid.multivector_centroid_lite_max_postings_per_token` GUC. Override
the cap list with `--document-node-serving-grid-centroid-lite-posting-caps`.
It also adds guarded pruning rows such as
`centroid_lite_f16_prune_safe_upper_bound` and
`centroid_lite_f16_cap_032_prune_safe_upper_bound`, driven by the scan-time
`turbohybrid.multivector_centroid_lite_pruning = safe_upper_bound` GUC. These
rows reuse the same physical kmeans centroid index as uncapped
`centroid_lite_f16`; they are not default serving profiles and should be read
as admission/latency tradeoff evidence for the experimental centroid-lite path.
Positive caps use deterministic uniform-stride posting sampling and expose
`centroid_posting_cap_strategy` in scan stats. Uniform cap rows are diagnostic
negative controls: do not promote them unless admission recall improves and
`centroid_docs_touched / doc_count` drops. `score_topk` rows require a freshly
built centroid index to benefit from payload-sorted centroid postings; older
centroid-only artifacts remain readable but need REINDEX/rebuild before this
bounded posting order is meaningful. Safe upper-bound pruning only drops
candidates when the persisted residual summary proves the centroid score is
exact for that document; otherwise it keeps the candidate and reports
`centroid_upper_bound_unsafe_fallbacks`.
Use `--document-node-serving-grid-centroid-lite-focus` when the only question is
the centroid-lite cap/pruning tradeoff. It restricts the grid to uncapped
`centroid_lite_f16`, capped `016/032/064` variants, the corresponding
`safe_upper_bound` pruning rows, scan-local bitset-prefilter measurement rows,
`centroid_lite_f16_pool_050`, and the `centroid_mean_f16` baseline with the
largest candidate budget by default.
Use `--document-node-serving-grid-token-pooling-memory-focus` when the question
is whether greedy token pooling can make 1M document-node storage/builds
tractable before tuning centroid-lite caps. The legacy
`--document-node-serving-grid-token-pooling-focus` flag is an alias. The focus
grid compares pure-ColBERT storage-reduction rows only:
`proxy_max_pool_proxy_only`, `proxy_random_projection_fde_proxy_only`, and
`document_nodes_sq8_paged_maxsim`, each with pooling `off` plus
`greedy_cosine` target ratios `0.75` and `0.50`.
`document_nodes_sq8_paged_maxsim` builds a document-node `sq8` index, keeps the
document sidecar paged during graph admission, scores graph candidates with
approximate full MaxSim, and exact-reranks retained candidates from heap
multivectors. It does not include BM25 rescue, learned-sparse rescue, or
centroid-lite cap rows. The JSON adds
`document_node_token_pooling_recommendation` with
`best_memory_safe_pooling`, `best_latency_safe_pooling`,
`best_pooling_quality_safe`, rejected pooling profiles, index bytes, docmap
bytes, sidecar bytes, graph bytes, pooled-token counts, pooling ratio, exact
pair counts, `compact_maxsim_score_us`, `compact_maxsim_pairs`, build time,
latency, admission, and qrel metrics when available.
Pooling changes the physical index signature, so pooled rows are separate build
variants rather than scan-time GUC-only comparisons. Final retained candidates
are still ranked by exact MaxSim.
Add `--document-node-serving-grid-include-entry-sidecar` only for focused graph
entry admission experiments. It appends `proxy_normalized_mean_f16_entry_sidecar`
and `centroid_mean_f16_entry_sidecar`, builds `entry_sidecar = on` indexes with
128 `hybrid_level_covering` representatives by default, and reports the
entry-sidecar reloptions plus `graph_entry_sidecar_*` scan counters in JSON and
Markdown. These rows are physical index variants, not scan-time GUC variants,
so they do not share the plain proxy index signature. Treat them as evidence
for whether graph entry selection is the admission bottleneck; final retained
candidates are still ordered by exact MaxSim. When proxy-encoder variants are
also enabled, the grid adds `proxy_max_pool_f16_entry_sidecar` so the stronger
`max_pool` proxy baseline can be compared with the same graph-entry
representative sidecar. Candidate-source deltas label sidecar rows with
`entry_sidecar_build_cost_high`, `entry_sidecar_no_admission_gain`, or
`entry_sidecar_latency_regression` when the paired baseline shows that the
extra physical index cost, admission result, or query latency is not justified.
Use `--multivector-doc-graph-entry-sample-count` for scan-time entry sampling
experiments without rebuilding the index. The default `0` keeps the compiled
document-graph entry sampler. Positive values score that many deterministic
document proxy entry seeds, bounded by document count, and report
`graph_entry_sample_configured`, `graph_entry_sample_effective`,
`graph_entry_sample_scored`, plus the
`multivector_doc_graph_entry_sample_*` aliases in scan stats. This is an
admission-diagnostics knob only; it does not change persisted formats or final
exact MaxSim ranking.

The serving-grid JSON and Markdown also include `candidate_source_deltas` for
known paired variants. These rows compare entry-sidecar, BM25 rescue,
learned-sparse rescue, candidate reservoirs, centroid-lite caps, and
proxy-encoder variants against their matching plain baseline at the same EF,
oversampling, and executed candidate budget. Use the deltas to decide which
admission change actually improved top-10 admission, NDCG, or p95 latency
before widening a run. Delta rows keep compact evidence details for the
experiment family, including entry-sample work, entry-sidecar representative
selection, reservoir union/duplicate summaries, rescue candidate counts, and
centroid-lite posting caps.
The recommendation block also carries a compact
`candidate_source_delta_summary` with the best admission, quality, and latency
delta overall and the best admission delta per comparison family. This keeps
the next focused experiment visible even when no row meets the serving safety
threshold.

Current 2k DBpedia evidence with the opt-in proxy encoder profiles shows
`centroid_mean_f16` as the best quality/balanced candidate among the tested safe
profiles. `proxy_max_pool_f16` improves admission over `normalized_mean`, but
still trails `centroid_mean`; `proxy_random_projection_fde_f16` is fast but too
weak for admission in that smoke. Treat these as profile-selection evidence, not
as hard defaults.

Quick 10k smoke:

```sh
nix develop .#bench
python benchmarks/dbpedia_colbert_multivector.py \
  --database pgturbohybrid_dbpedia_colbert_10k \
  --precomputed-dataset johannhartmann/pgturbohybrid_dbpedia_colbert \
  --max-docs 10000 \
  --max-queries 100 \
  --reuse-data \
  --document-node-serving-grid \
  --document-node-serving-grid-smoke \
  --output .nix-dev/tmp/dbpedia-colbert-serving-grid-10k.json \
  --markdown-output .nix-dev/tmp/dbpedia-colbert-serving-grid-10k.md
```

Smoke mode runs only `proxy_normalized_mean_f16`, `centroid_mean_f16`, and
`centroid_lite_f16` over EF `50,100`, oversampling `1`, and budgets `200,800`
unless `--admission-budget-sweep` is explicit. If more than 25 queries are
loaded, the smoke run uses the first 25 and records `query_subset_used = true`.
Treat this as a harness/runtime check, not serving evidence.

#### Build-only first

For 10k/x00k document-node experiments, run the build-only diagnostic before a
full retrieval/admission grid whenever `CREATE INDEX` is slow or unmeasured.
The build-only mode answers whether the bottleneck is index construction,
sidecar serialization, centroid construction, or graph topology; a full serving
grid should come after this evidence, not before it.

If a 10k/x00k run stalls in `CREATE INDEX`, measure the build path directly
before running admission or retrieval:

```sh
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
```

The build-only report groups profiles by physical index signature and emits
`turbohybrid_last_build_stats()` fields for centroid construction, proxy
construction, document-sidecar writes, centroid-sidecar writes, and centroid
posting writes. It also preserves the generic build timer aliases
`build_edges_us`, `write_pages_us`, `wal_us`, and `total_us`, derives
`dominant_build_phase`, `build_phase_known_ms`, and
`build_phase_unattributed_ms`, and emits a build acceptance summary with
slow-build warning labels such as `centroid_kmeans_dominates_build`,
`centroid_posting_write_dominates_build`, `proxy_build_dominates_build`,
`graph_edges_dominates_build`, `build_unattributed_high`, and
`index_rebuild_not_reused`. A large unattributed bucket means the stall is
likely normal graph/topology build work or another phase that is not yet
covered by the multivector-specific timers. It intentionally skips retrieval
and exact admission baselines, so it is a build-cost diagnostic rather than
serving-quality evidence.

Interpret the build-only report before changing query knobs:

- If centroid build dominates, reduce the centroid count, test token pooling
  before centroiding, or avoid `centroid_lite` for the current serving target.
- If graph build dominates, compare cheaper proxy encoders first and test
  `entry_sidecar` as a separate physical index variant rather than mixing it
  into the same run.
- If sidecar write dominates, compare `f16`/`sq8` document storage and token
  pooling before tuning graph traversal.
- If `build_phase_unattributed_ms` is high, instrument or inspect
  graph/topology build before guessing at retrieval-side optimizations.

Keep `multivector_doc_build_scorer = exact_symmetric` diagnostic-only for tiny
corpora. x00k production builds should use proxy graph topology unless build
and quality evidence explicitly prove another path is safe.

For focused candidate-admission follow-up after build cost is bounded, keep the
profile set narrow and override only the serving-grid EF/oversampling values you
need to test. This avoids editing benchmark constants or switching to the
broader document-node admission grid:

```sh
nix develop .#bench
python benchmarks/dbpedia_colbert_multivector.py \
  --database pgturbohybrid_dbpedia_colbert_10k \
  --precomputed-dataset johannhartmann/pgturbohybrid_dbpedia_colbert \
  --max-docs 10000 \
  --max-queries 50 \
  --document-node-serving-grid \
  --document-node-serving-grid-include-proxy-encoders \
  --document-node-serving-grid-profiles centroid_mean_f16,proxy_max_pool_f16,proxy_normalized_mean_f16 \
  --document-node-serving-ef-grid 400,800 \
  --document-node-serving-oversampling-grid 1,2 \
  --admission-budget-sweep 800 \
  --output .nix-dev/tmp/dbpedia-colbert-serving-grid-10k-focused.json \
  --markdown-output .nix-dev/tmp/dbpedia-colbert-serving-grid-10k-focused.md
```

x00k evaluation template:

```sh
nix develop .#bench
python benchmarks/dbpedia_colbert_multivector.py \
  --database pgturbohybrid_dbpedia_colbert_x00k \
  --precomputed-dataset johannhartmann/pgturbohybrid_dbpedia_colbert \
  --max-docs 300000 \
  --max-queries 1000 \
  --reuse-data \
  --document-node-serving-grid \
  --serving-min-top10-admission 0.80 \
  --serving-min-ndcg-ratio-vs-exact 0.95 \
  --output .nix-dev/tmp/dbpedia-colbert-serving-grid-x00k.json \
  --markdown-output .nix-dev/tmp/dbpedia-colbert-serving-grid-x00k.md
```

Set `--max-docs` to `100000`, `300000`, or the corpus size you want to serve.
Use `--max-queries 500` for a shorter selection run and `1000` or more when
the qrel coverage is large enough to justify the extra time. Generated JSON,
Markdown, local logs, downloaded datasets, and other benchmark artifacts belong
under `.nix-dev/tmp/` or another ignored work directory and must not be
committed.

Interpret the serving-grid report as follows:

- Choose `best_latency_safe` when it passes the admission and quality
  thresholds. This is the first profile to try for production serving.
- Choose `best_quality` when relevance dominates latency or storage cost.
- Treat `centroid_lite` as PLAID-inspired admission followed by exact final
  MaxSim, not as an approximate final ranking mode.
- Treat `quantized_inverted_experimental` as opt-in research, not production.
- If `exact_doc_scan` wins relevance but is too slow, inspect admission loss by
  candidate source before tuning rerank kernels.
- If `proxy_vector` wins latency but fails admission, try `centroid_mean`,
  `centroid_lite`, opt-in BM25 rescue
  (`--document-node-serving-grid-include-bm25-rescue`), opt-in learned-sparse
  rescue (`--document-node-serving-grid-include-learned-sparse-rescue` with
  sparse JSONL inputs), or larger admission budgets.
- If exact rerank dominates latency, test `greedy_cosine` token pooling and
  `f16` or `sq8` document sidecar storage.

The report writes `document_node_serving_grid` plus
`document_node_serving_recommendation`. The recommendation includes
`best_latency_safe`, `best_quality`, `best_balanced`, the Pareto frontier, and
rejected profiles with reasons. Final SQL result ordering remains exact MaxSim
unless the candidate source is explicitly experimental.

The serving-grid JSON and Markdown also include cost accounting. Use
`total_elapsed_ms`, `index_build_elapsed_ms_total`,
`exact_baseline_elapsed_ms_total`, `retrieval_elapsed_ms_total`,
`profiles_run`, `index_builds`, `exact_baseline_query_count`, and
`retrieval_query_count` to see whether a run is dominated by repeated exact
admission baselines, index rebuilds, or indexed retrieval. Each profile summary
and each profile/EF/oversampling row repeats the same timing split so slow
profiles can be diagnosed without opening the full per-query trace.
When a serving-grid run stalls during `CREATE INDEX`, inspect
`turbohybrid_last_build_stats()` before widening the benchmark again. The
document-node build stats include `multivector_centroid_build_us`,
`multivector_centroid_cluster_us`, `multivector_centroid_residual_us`,
`multivector_centroid_build_docs`, `multivector_centroid_build_vectors`,
`multivector_proxy_build_us`, `multivector_doc_sidecar_write_us`,
`multivector_centroid_sidecar_write_us`,
`multivector_centroid_posting_write_us`, and
`multivector_centroid_posting_count` so centroid construction, proxy-vector
construction, sidecar writes, and centroid posting writes can be separated from
normal graph edge construction. The benchmark derives
`dominant_build_phase`, `build_phase_known_ms`, and
`build_phase_unattributed_ms` from these counters so a 10k stall can be routed
to centroid work, sidecar/posting serialization, or uninstrumented graph build
work before adding another optimization.

### Choosing the next x00k experiment

Use this decision tree before widening a document-node benchmark:

- If `CREATE INDEX` is slow, run build-only first and inspect
  `dominant_build_phase`. Do not run the full serving grid until build cost is
  attributed.
- If query latency is slow, run latency-only and inspect phase timing before
  changing C code. Separate exact rerank, graph traversal, sidecar/cache, and
  SQL/Python harness overhead.
- If `proxy_vector` is fast but admission is weak, try `centroid_mean`,
  `max_pool`, entry sampling, entry sidecar, BM25 rescue, or learned-sparse
  rescue as focused admission experiments.
- If `centroid_lite` is slow, inspect
  `centroid_docs_touched / doc_count`, posting counts, and posting-cap
  warnings. Treat uniform caps as diagnostics only; use upper-bound pruning or
  bitset-prefilter evidence before tuning exact MaxSim.
- If exact rerank dominates latency, lower `--serving-exact-rerank-k`, test
  token pooling, verify the SIMD kernel, and then consider adaptive rerank
  changes.
- If sparse rescue helps, compare BM25 and learned sparse with cap accounting
  and keep final exact MaxSim ranking.
- If all proxy variants fail admission, run centroid-lite focus, optionally run
  the quantized-inverted research branch, and consider the learned-projection
  proxy path.

Build-only:

```sh
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
```

Latency-only:

```sh
nix develop .#bench
python benchmarks/dbpedia_colbert_multivector.py \
  --database pgturbohybrid_dbpedia_colbert_10k_latency \
  --precomputed-dataset johannhartmann/pgturbohybrid_dbpedia_colbert \
  --max-docs 10000 \
  --max-queries 25 \
  --reuse-data \
  --document-node-serving-latency-only \
  --serving-profile-name proxy_normalized_mean_f16 \
  --serving-candidate-k 800 \
  --serving-exact-rerank-k 100 \
  --serving-ef 100 \
  --serving-oversampling 1 \
  --serving-storage f16 \
  --serving-cache auto \
  --output .nix-dev/tmp/dbpedia-colbert-serving-latency-10k.json \
  --markdown-output .nix-dev/tmp/dbpedia-colbert-serving-latency-10k.md
```

Proxy/entry admission focus:

```sh
nix develop .#bench
python benchmarks/dbpedia_colbert_multivector.py \
  --database pgturbohybrid_dbpedia_colbert_10k_proxy_focus \
  --precomputed-dataset johannhartmann/pgturbohybrid_dbpedia_colbert \
  --max-docs 10000 \
  --max-queries 50 \
  --reuse-data \
  --document-node-serving-grid-proxy-admission-focus \
  --output .nix-dev/tmp/dbpedia-colbert-proxy-admission-focus-10k.json \
  --markdown-output .nix-dev/tmp/dbpedia-colbert-proxy-admission-focus-10k.md
```

Centroid-lite focus:

```sh
nix develop .#bench
python benchmarks/dbpedia_colbert_multivector.py \
  --database pgturbohybrid_dbpedia_colbert_10k_centroid_focus \
  --precomputed-dataset johannhartmann/pgturbohybrid_dbpedia_colbert \
  --max-docs 10000 \
  --max-queries 50 \
  --reuse-data \
  --document-node-serving-grid-centroid-lite-focus \
  --output .nix-dev/tmp/dbpedia-colbert-centroid-lite-focus-10k.json \
  --markdown-output .nix-dev/tmp/dbpedia-colbert-centroid-lite-focus-10k.md
```

Token-pooling focus:

```sh
nix develop .#bench
python benchmarks/dbpedia_colbert_multivector.py \
  --database pgturbohybrid_dbpedia_colbert_10k_pooling_focus \
  --precomputed-dataset johannhartmann/pgturbohybrid_dbpedia_colbert \
  --max-docs 10000 \
  --max-queries 50 \
  --reuse-data \
  --document-node-serving-grid-token-pooling-memory-focus \
  --output .nix-dev/tmp/dbpedia-colbert-token-pooling-memory-focus-10k.json \
  --markdown-output .nix-dev/tmp/dbpedia-colbert-token-pooling-memory-focus-10k.md
```

For SPLADE/SPLATE-style exported sparse vectors, use the explicit sparse
candidate-source switch:

```sh
python benchmarks/dbpedia_colbert_multivector.py \
  --multivector-sparse-candidate-source learned_sparse \
  --learned-sparse-doc-jsonl .nix-dev/tmp/dbpedia-splate-docs.jsonl \
  --learned-sparse-query-jsonl .nix-dev/tmp/dbpedia-splate-queries.jsonl \
  --multivector-bm25-candidate-injection off \
  --methods pgturbohybrid_colbert_multivector_query_only
```

The PostgreSQL ingestion helper is
`turbohybrid_sparse_vector_from_arrays(term_ids int[], weights real[])`.
Convert exported document features into the indexed sparse key with
`turbohybrid_sparse_vector_to_tsvector(...)`, and convert exported query
features with `turbohybrid_sparse_vector_to_tsquery(...)`. The current
implementation reuses the existing sparse/BM25 postings branch for candidate
admission; it does not train sparse weights inside PostgreSQL and it does not
change the final dense ranking contract. Admitted documents are still reranked
with exact MaxSim. Reports include
`learned_sparse_candidates`, `learned_sparse_retained_for_maxsim`, and
`learned_sparse_branch_latency_us`.

The benchmark JSONL hooks expect one object per row:

```json
{"doc_id":"<dbpedia-doc-id>","term_ids":[42,777],"weights":[2.0,1.0]}
{"query_id":"<dbpedia-query-id>","term_ids":[42,991],"weights":[1.0,0.5]}
```

The document and query JSONL files must be supplied together. When loaded, the
benchmark stores the vectors in `learned_sparse`, indexes
`learned_sparse_tsv`, and the `learned_sparse_exact_maxsim` hybrid mode uses
the query sparse vector instead of deriving a `tsquery` from text.

For DBpedia document-node admission checks, use `--admission-debug` for one
specific configuration or `--document-node-admission-grid` for the full
DBpedia-scale gate. The grid rebuilds the benchmark index once for token-node
baselines and once for document-node modes, then reports token graph,
`exact_token_scan`, forced plain fallback, `exact_doc_scan`, document-node
`f32|f16|sq8`, `proxy_vector`, token-node `centroid_lite`, document-node
`centroid_lite`, and `quantized_inverted_experimental` admission. It sweeps
`--document-node-storage-grid`, `--document-node-pooling-grid`,
`--document-node-cache-grid`, `--document-node-proxy-encoder-grid`,
`--document-node-ef-grid`, and
`--document-node-oversampling-grid`; the existing
`--admission-budget-sweep` controls the candidate budgets. Leaving
`--multivector-doc-graph-rescore-k` at `0` makes each admission budget drive
the rescore budget through `turbohybrid.multivector_doc_candidate_k`.

Query-only and admission-only runs build a single-column multivector
TurboHybrid index. The benchmark adds the lexical `body_tsv` or
`learned_sparse_tsv` index column only when the selected methods, sparse
candidate source, BM25 injection, learned-sparse input, or
`--hybrid-evaluation-harness` need it. This keeps admission sweeps from
paying hybrid index-build cost unless the run is explicitly measuring
hybrid behavior.

When the document-node grid includes the default physical layout
(`--multivector-proxy-encoder` plus `off:1.0` pooling), the harness runs
`proxy_vector`, document-index `exact_doc_scan`, and
`quantized_inverted_experimental` while that index is still resident. The JSON
field `default_document_modes_reused_grid_index` records whether this avoided a
separate default document-node rebuild.

Document-node index construction uses the same symmetric document MaxSim
objective for bulk and incremental graph edges. The build scorer computes both
directions in one pairwise token pass, so DBpedia build timings should be
compared against this fused-scorer path rather than older runs that computed
`MaxSim(A,B)` and `MaxSim(B,A)` separately.

Fast 10k document-node admission smoke:

```sh
nix develop .#bench
python benchmarks/dbpedia_colbert_multivector.py \
  --database pgturbohybrid_dbpedia_colbert_10k \
  --precomputed-dataset johannhartmann/pgturbohybrid_dbpedia_colbert \
  --max-docs 10000 \
  --max-queries 100 \
  --reuse-data \
  --document-node-admission-grid \
  --document-node-storage-grid f32,f16,sq8 \
  --document-node-pooling-grid off:1.0,greedy_cosine:0.75,greedy_cosine:0.5,greedy_cosine:0.33 \
  --document-node-cache-grid auto,paged \
  --document-node-ef-grid 50,100,200 \
  --document-node-oversampling-grid 1,2,4 \
  --index-graph-m 4 \
  --index-graph-ef-construction 8 \
  --index-graph-ef-search 8 \
  --admission-budget-sweep 50,100,200,400,800 \
  --output .nix-dev/tmp/dbpedia-colbert-admission-10k.json \
  --markdown-output .nix-dev/tmp/dbpedia-colbert-admission-10k.md
```

The 10k command uses small index-build graph knobs so it is a fast harness
smoke. Leave `--index-graph-m`, `--index-graph-ef-construction`, and
`--index-graph-ef-search` at `0` or omit them for the extension defaults when
collecting quality-profile evidence.

Latest qrel-backed 10k smoke evidence for one document-node configuration
(`f32`, `auto`, no pooling, `graph_m = 4`, `graph_ef_construction = 8`,
`graph_ef_search = 8`, budgets `50,800,1600`, three queries) retained `126`
qrels from the precomputed dataset. It produced `recall@10 = 0.166667`,
`ndcg@10 = 0.207257`, exact top-1 admission `0.333333`, and exact top-10
admission recall `0.400000` at budget `1600`. Treat this as a qrel-loader and
single-profile admission smoke; the admission acceptance gate is satisfied by
10k evidence only once the grid comparison covers token baselines, exact scans,
storage modes, and proxy branches.

A narrow 10k qrel-backed admission-grid smoke with the same three queries,
`f32` storage only, no pooling, EF `50`, oversampling `1`, and budgets `50,800`
confirmed that the comparison report is populated. At budget `800`, token nodes
scored `recall@10 = 0.666667`, `ndcg@10 = 0.600137`, exact top-1 admission
`0.666667`, and top-10 admission `0.466667`; the tested document-node profile
scored `recall@10 = 0.300000`, `ndcg@10 = 0.375685`, exact top-1 admission
`0.333333`, and top-10 admission `0.300000`; exact document scan and plain
fallback both reached `recall@10 = 0.800000` and full admission. This is a
comparison-smoke result, not full acceptance evidence, because `f16`, `sq8`,
wider EF/oversampling settings, and the normal 10k query set were intentionally
omitted.

A wider 10k qrel-backed run used `10` selected DBpedia queries, `382`
loaded qrels, no pooling, `auto` cache, storage `f32,f16,sq8`, EF
`50,100,200,400,800`, oversampling `1,2,4,8`, and budgets `50,800,1600`.
The report emitted `69` comparison rows. At budget `1600`, exact document scan
and forced plain fallback reached `recall@10 = 0.667143`, `ndcg@10 = 0.591325`,
and full top-1/top-10 admission. Document-node rows with oversampling `8`
matched that exact-scan quality and full admission for all three storage modes,
but did so by scoring the full `10000` documents on this slice. Lower
oversampling exposed the quality/cost tradeoff: oversampling `4` reached
top-10 admission `0.490000` and `recall@10 = 0.498571`; oversampling `1`
reached top-10 admission `0.420000` and `recall@10 = 0.448571`. The fastest
tested document-node exact-quality rows were `f32` around `424..446 ms` p50 and
`sq8` around `430..440 ms` p50; `f16` was slower on this host. `proxy_vector`
was faster at `138.726 ms` p50 but only reached `recall@10 = 0.398571` and
top-10 admission `0.360000`. The experimental quantized-inverted branch reached
`recall@10 = 0.657143` and top-10 admission `0.720000`, but took
`35266.755 ms` p50. This satisfies the 10k storage/EF/oversampling comparison
shape; the 100k and 1M commands below remain opt-in scale proof.

A 100k reuse-index probe before the document-node entry-seeding fix exposed the
scale failure mode rather than a usable quality result: document-node recall@10
was `0.000000` at budgets `100`, `1600`, and `10000`, while one budget-10000
query touched about `3.57 GB` of document sidecar data and exact-reranked
`10000` full multivector documents. That combination can overload a developer
machine when followed by the 8-client throughput phase. The candidate path now
scores a bounded deterministic spread of document nodes as MaxSim entry seeds,
matching the single-vector graph's multi-entry strategy instead of relying only
on the global, segment, and routing entries. The index/default `graph_ef_search`
and explicit `turbohybrid.multivector_doc_graph_search_ef` settings remain
traversal caps: a large admission/rerank budget no longer inflates the graph
walk beyond the configured EF. Serving-grid reports expose this with
`proxy_candidate_limit_source = search_ef` and
`proxy_candidate_limit_effective`; set EF at least as high as the intended
proxy candidate band when the benchmark is meant to test a large candidate
budget. Use `--document-node-serving-ef-grid` and
`--document-node-serving-oversampling-grid` for focused serving-grid runs that
need EF values outside the compact default `50,100,200`. Use
`--document-node-serving-grid-profiles` to run a focused subset such as
`proxy_normalized_mean_f16` before paying for every physical profile index
build. For additional proxy encoder evidence, combine it with
`--document-node-serving-grid-include-proxy-encoders` and request
`proxy_max_pool_f16` or `proxy_random_projection_fde_f16` by name. Treat
learned projection as separate opt-in evidence: add
`--document-node-serving-grid-include-learned-projection`, configure
`--multivector-learned-projection-path`, and request
`proxy_learned_projection_v1_f16`. Treat any future 100k/1M run as a fresh
validation of that candidate-admission fix; do not use the zero-recall probe as
performance evidence.

The normal benchmark path now runs a serial probe before the 8-client throughput
phase. If the serial run already exceeds configured scan-work limits
(`--parallel-safety-max-docs-scored`,
`--parallel-safety-max-exact-rerank-docs`,
`--parallel-safety-max-exact-pairs`,
`--parallel-safety-max-sidecar-bytes`, or
`--parallel-safety-max-serial-p95-ms`), the JSON and Markdown reports mark the
parallel phase as skipped and include the observed counters. Use
`--skip-parallel-retrieval` for the first 100k/1M admission probe when you only
want serial responsiveness and scan-work counters. Use
`--force-parallel-retrieval` only after the serial admission report shows that
the candidate path is no longer expanding into near-exact MaxSim over thousands
of full document multivectors.

100k admission run:

```sh
nix develop .#bench
python benchmarks/dbpedia_colbert_multivector.py \
  --database pgturbohybrid_dbpedia_colbert_100k \
  --precomputed-dataset johannhartmann/pgturbohybrid_dbpedia_colbert \
  --max-docs 100000 \
  --max-queries 0 \
  --reuse-data \
  --document-node-admission-grid \
  --document-node-storage-grid f32,f16,sq8 \
  --document-node-pooling-grid off:1.0,greedy_cosine:0.75,greedy_cosine:0.5,greedy_cosine:0.33 \
  --document-node-cache-grid auto,paged \
  --document-node-ef-grid 50,100,200,400,800 \
  --document-node-oversampling-grid 1,2,4,8 \
  --index-graph-m 0 \
  --index-graph-ef-construction 0 \
  --index-graph-ef-search 0 \
  --index-native-segments 8 \
  --multivector-exact-rerank adaptive \
  --skip-parallel-retrieval \
  --admission-budget-sweep 100,200,400,800,1600,3200,6400,10000 \
  --output .nix-dev/tmp/dbpedia-colbert-admission-100k.json \
  --markdown-output .nix-dev/tmp/dbpedia-colbert-admission-100k.md
```

1M admission run:

```sh
nix develop .#bench
python benchmarks/dbpedia_colbert_multivector.py \
  --database pgturbohybrid_dbpedia_colbert_1m \
  --precomputed-dataset johannhartmann/pgturbohybrid_dbpedia_colbert \
  --max-docs 1000000 \
  --max-queries 0 \
  --reuse-data \
  --document-node-admission-grid \
  --document-node-storage-grid f32,f16,sq8 \
  --document-node-pooling-grid off:1.0,greedy_cosine:0.75,greedy_cosine:0.5,greedy_cosine:0.33 \
  --document-node-cache-grid auto,paged \
  --document-node-ef-grid 50,100,200,400,800 \
  --document-node-oversampling-grid 1,2,4,8 \
  --index-graph-m 0 \
  --index-graph-ef-construction 0 \
  --index-graph-ef-search 0 \
  --index-native-segments 8 \
  --skip-parallel-retrieval \
  --admission-budget-sweep 100,200,400,800,1600,3200,6400,10000 \
  --output .nix-dev/tmp/dbpedia-colbert-admission-1m.json \
  --markdown-output .nix-dev/tmp/dbpedia-colbert-admission-1m.md
```

For document-node indexes this is segmented exact-MaxSim build work, not
PostgreSQL parallel worker build work. The extension keeps document-node graph
construction serial per segment because parallel code-only workers do not carry
the document multivector sidecar required for MaxSim-aligned edge geometry.

The grid is not part of normal CI and requires DBpedia text or the precomputed
ColBERT multivector dataset. JSON output includes per-query admission records
with `exact_top1_admitted`, `exact_top10_admission_recall`, `exact_top1_rank`,
latency p50/p95, scored-doc/edge/rerank counters, storage kind, sidecar
page/cache counters, `admission_inferred_from_result_docs`, per-exact-top
`admission_evidence`, and `turbohybrid_last_scan_stats()` snapshots. Markdown
output includes a
comparison table with admission, BEIR metrics when qrels are loaded, latency,
documents scored, graph edges visited, exact rerank docs, adaptive rerank pair
savings, native cache exact-byte counters, and document-sidecar page/cache
counters. Compare
`auto`/`resident` rows against `paged` rows to separate warm resident sidecar
latency from cold/random sidecar access cost.

Precomputed bounded slices use qrel prioritization by default, matching the raw
DBpedia loader: the selected query IDs are read first, their judged document IDs
fill the `--max-docs` budget first, and the remaining document budget is filled
from the corpus shards. Use `--no-prioritize-qrels` only when intentionally
measuring first-row corpus slices. If a precomputed slice still has no loaded
qrels, the benchmark leaves BEIR metric objects empty; treat that as a broken or
non-quality slice, not retrieval-quality evidence.

For the end-to-end hybrid comparison, add
`--hybrid-evaluation-harness`. The harness rebuilds a document-node index and
runs these modes over the same loaded DBpedia queries:

- `exact_scan`
- `document_nodes`
- `document_nodes_bm25_admission`
- `document_nodes_bm25_rrf`
- `document_nodes_bm25_dbsf`
- `proxy_vector_document_nodes`
- `learned_sparse_exact_maxsim`

For research comparisons, add
`--hybrid-evaluation-modes exact_scan,document_nodes,learned_sparse_exact_maxsim,quantized_inverted_experimental`
to compare the persisted quantized-inverted branch against the learned-sparse
and exact/document-node baselines in the same harness. This mode is supported
only when named explicitly; it is not part of the default recommendation set.

The BM25 admission-only and learned-sparse modes pass `text_query` with
`bm25_weight => 0`, so sparse candidates are used for admission but final
ranking remains exact MaxSim. The RRF and DBSF modes run the sparse branch as a
hybrid branch and report branch candidates, branch latency, exact MaxSim work,
sidecar bytes, BEIR metrics, candidate admission failures, and a recommended
default profile. JSON and Markdown output also include profile-specific
recommendations for `latency`, `balanced`, `quality`, and `high_recall`, each
with the selected mode and the concrete GUCs needed to reproduce that profile.

A 10k qrel-backed hybrid run on the same `10` queries and `382` qrels
compared `exact_scan`, `document_nodes`, BM25 admission, BM25 RRF, BM25 DBSF,
and `proxy_vector_document_nodes`. `exact_scan` remained the quality, balanced,
and high-recall recommendation with `recall@10 = 0.667143`, `ndcg@10 =
0.591325`, full admission, and `358.157 ms` p50. Raw `document_nodes` was much
faster at `75.096 ms` p50 but only reached `recall@10 = 0.300000`, top-10
admission `0.250000`, and had candidate failures for all `10` queries. BM25
admission improved recall to `0.420000` and top-10 admission to `0.500000`
around `73..74 ms` p50, but still had failures on `8..9` queries. RRF had the
best BM25-fused NDCG at `0.372541`; DBSF was lower at `0.343564`.
`proxy_vector_document_nodes` was the latency recommendation at `61.388 ms`
p50, but had `recall@10 = 0.227143` and top-10 admission `0.140000`. No
learned-sparse JSONL inputs were present for this local run, so that mode
remains a conditional harness path until exported sparse vectors are supplied.

The learned-sparse branch was validated separately on a 1k qrel-backed slice
using deterministic text-hash sparse JSONL fixtures under `.nix-dev/tmp`
(`1867` document rows read, `1000` loaded documents updated, and `10` query
rows updated). This is a code-path validation for the hybrid harness, not a
SPLADE/SPLATE quality claim. On `1000` docs, `10` queries, and `382` loaded
qrels, `learned_sparse_exact_maxsim` produced nonzero sparse candidates
(`p50 = 68`, mean `55.4`), reached `recall@10 = 0.681429`,
`ndcg@10 = 0.591116`, top-10 admission `0.910000`, and `18.756 ms` p50. Exact
scan reached `recall@10 = 0.677143`, `ndcg@10 = 0.593917`, and full admission
at `46.206 ms` p50. Raw document nodes reached `recall@10 = 0.597143` and
top-10 admission `0.690000`. Treat this as proof that externally supplied
learned-sparse vectors are ingested, indexed, reported, and used for admission;
real learned-sparse model exports are still required before recommending this
mode for quality.

## NextPlaid reference on the same DBpedia corpus

Use `--next-plaid-beir-quality-only` to evaluate an external NextPlaid index
against the same DBpedia corpus, query text, and qrels already imported into
PostgreSQL. This is a reference benchmark, not a native pgturbohybrid path: it
does not build a pgturbohybrid index, does not run exact MaxSim admission
scans, and does not use BM25, learned sparse, or hybrid fusion.

Start NextPlaid separately from the sibling checkout, for example with the
project's Docker instructions in `../next-plaid`. Then run the benchmark from
this repository's bench shell:

```sh
nix develop .#bench
python benchmarks/dbpedia_colbert_multivector.py \
  --database pgturbohybrid_dbpedia_colbert_1m_colbert \
  --reuse-data \
  --next-plaid-beir-quality-only \
  --next-plaid-url http://localhost:8080 \
  --next-plaid-index-name dbpedia_colbert_1m \
  --next-plaid-top-k 100 \
  --next-plaid-n-ivf-probe 8 \
  --next-plaid-n-full-scores 4096 \
  --max-queries 381 \
  --output .nix-dev/tmp/dbpedia-colbert-1m-next-plaid-beir-quality.json \
  --markdown-output .nix-dev/tmp/dbpedia-colbert-1m-next-plaid-beir-quality.md
```

If the NextPlaid index does not already exist, add
`--next-plaid-create-index --next-plaid-add-documents --next-plaid-wait-for-index`.
The benchmark uploads document text from `dbpedia_colbert_docs` and stores
metadata `doc_id` for every document. That metadata is required because
NextPlaid result IDs are engine-local IDs; qrel evaluation maps results back to
`dbpedia_colbert_qrels.doc_id` through `metadata.doc_id`. If the metadata is
missing, the JSON and Markdown mark `docid_mapping_suspect = true` and the qrel
metrics should not be treated as valid.

The output key is `next_plaid_beir_quality`. It reports p50/p95/p99 latency,
QPS, recall@10, ndcg@10, mrr@10, qrel coverage, NextPlaid index metadata, and
the exact search parameters. Exact admission remains explicitly unavailable in
this mode; use the sampled exact oracle modes when candidate-admission evidence
is needed.

For long-context ColBERT/ModernColBERT fixtures, keep each RAG chunk as one
database row and build document-node indexes with explicit context mode:

```sql
CREATE INDEX passages_colbert_context_idx
ON passages USING turbohybrid
  (colbert multivector_maxsim_ip_turbohybrid_ops)
WITH (
  multivector_graph = document_nodes,
  multivector_doc_build_scorer = proxy,
  exact_storage = off,
  multivector_context_mode = context_level,
  multivector_field_mode = weighted
);
```

The benchmark input should construct document embeddings with
`turbohybrid_multivector_from_contexts(raw_values, dim, context_offsets)` or
`turbohybrid_multivector_from_contexts_and_fields(raw_values, dim,
context_offsets, field_ids)`. Offsets are zero-based context-window token
starts. `context_level` changes document-node sidecar scoring and exact heap
rerank to best-context MaxSim; `flat` keeps global cross-context MaxSim.
`multivector_field_mode = weighted` is currently provenance/validation for
field-aware datasets. Query-specific title/body/section weights are evaluated
with `turbohybrid_multivector_field_weighted_maxsim()` until query payloads
carry field weights. Do not combine context-aware document embeddings with
`--multivector-token-pooling` yet; context-aware indexes require pooling `off`
until pooling can preserve or rebuild context-window metadata.

Fast 10k hybrid harness:

```sh
nix develop .#bench
python benchmarks/dbpedia_colbert_multivector.py \
  --database pgturbohybrid_dbpedia_colbert_10k \
  --precomputed-dataset johannhartmann/pgturbohybrid_dbpedia_colbert \
  --max-docs 10000 \
  --max-queries 100 \
  --reuse-data \
  --hybrid-evaluation-harness \
  --multivector-doc-storage f32 \
  --multivector-doc-storage-cache auto \
  --multivector-doc-graph-search-ef 200 \
  --multivector-doc-graph-oversampling 4 \
  --multivector-exact-rerank adaptive \
  --output .nix-dev/tmp/dbpedia-colbert-hybrid-10k.json \
  --markdown-output .nix-dev/tmp/dbpedia-colbert-hybrid-10k.md
```

For 100k and 1M runs, keep the same command shape, increase `--max-docs`, and
use larger `--max-queries` only when the qrels and runtime budget justify it.
The harness is opt-in and is not part of normal CI.

For current DBpedia 1M serving-quality experiments, prefer ColBERT as a bounded
reranker over vector+BM25 candidates instead of as an independent retrieval
branch. The current proxy-only ColBERT branch is useful as a negative control,
but 1M qrel evidence showed near-zero quality and naive three-branch RRF made
the dense+BM25 result worse. Use `--colbert-hybrid-rerank` for the supported
setup:

```sh
nix develop .#bench
python benchmarks/dbpedia_colbert_multivector.py \
  --database pgturbohybrid_dbpedia_colbert_1m_hybrid_rerank \
  --precomputed-dataset johannhartmann/pgturbohybrid_dbpedia_colbert \
  --max-docs 1000000 \
  --max-queries 381 \
  --reuse-data \
  --colbert-hybrid-rerank \
  --colbert-hybrid-rerank-window 200 \
  --colbert-hybrid-dense-k 800 \
  --colbert-hybrid-bm25-k 800 \
  --rrf-k 60 \
  --final-k 10 \
  --quality-k 10 \
  --reuse-index \
  --output .nix-dev/tmp/dbpedia-colbert-1m-hybrid-rerank.json \
  --markdown-output .nix-dev/tmp/dbpedia-colbert-1m-hybrid-rerank.md
```

This mode materializes a benchmark-only dense proxy for the ColBERT vectors,
uses a normal vector+BM25 turbohybrid index for first-stage retrieval, and
orders the bounded candidate window with exact
`turbohybrid_multivector_maxsim_distance()`. It does not build or use a
multivector candidate-source index for the first stage.

Latest qrel-backed 10k hybrid smoke evidence, using three queries and `126`
loaded qrels, compared `exact_scan`, dense `document_nodes`,
BM25 admission-only, BM25 RRF, BM25 DBSF, and `proxy_vector_document_nodes`.
`exact_scan` was the quality ceiling (`recall@10 = 0.800000`,
`ndcg@10 = 0.736563`, p50 `350.746 ms`). Dense document nodes were faster but
missed most judged documents (`recall@10 = 0.166667`,
`ndcg@10 = 0.207257`, top-10 admission `0.133333`, p50 `74.467 ms`). BM25
admission/RRF/DBSF improved recall to `0.400000` with p50 around `69..75 ms`,
but all still had candidate-admission failures for all three queries.
`proxy_vector_document_nodes` was fastest (`58.943 ms` p50) but had
`recall@10 = 0.066667` and zero top-10 admission. The corrected
recommendation metadata emits `turbohybrid.multivector_candidate_source =
exact_doc_scan` when `exact_scan` wins a quality or balanced profile.

Multivector hybrid scans support RRF and normalized score fusion through
`weighted`, `fast_weighted`, `calibrated`, and explicit `dbsf`; raw BM25 plus
raw MaxSim alpha fusion is intentionally not exposed. DBSF normalizes each
returned branch score distribution before weighted summation and reports
`dbsf_enabled`, `dbsf_branch_mean`, `dbsf_branch_stddev`,
`dbsf_branch_min`, `dbsf_branch_max`, and `dbsf_degenerate_branches` in
`turbohybrid_last_scan_stats()`. Prefer RRF when candidate sets are tiny,
branch scores are identical, or distributions are too noisy for a stable
mean/stddev; use DBSF only when benchmark qrels show score-distribution fusion
beats rank fusion for the workload. With
`turbohybrid.hybrid_budget_policy = adaptive`, document-level multivector dense
branches use their admission stats before BM25 collection to shrink only
defaulted BM25 budgets when dense admission is not truncated. If admission is
underfilled or truncated by `doc_candidate_k`/accumulator limits, the scheduler
keeps the BM25 branch wide and reports the reason in `hybrid_budget_reason`.

Use `--multivector-recall-gate` for the deterministic recall gate that does
not require DBpedia, BEIR qrels, or a GGUF model. It builds a tiny synthetic
many-moderate corpus where exact MaxSim top-1 is `good`, while low-budget
token-node candidate generation admits only single-token spike documents. The
gate compares `exact_scan`, token graph, exact token scan, balanced reservoirs,
forced plain fallback, `exact_doc_scan`, `doc_graph_prototype`, and
`document_nodes`/`proxy_vector`, then fails unless the document-level exact
paths return and admit the exact top-1:

```sh
nix develop .#bench
python benchmarks/dbpedia_colbert_multivector.py \
  --multivector-recall-gate \
  --database pgturbohybrid_dev \
  --output .nix-dev/tmp/multivector-recall-gate.json \
  --markdown-output .nix-dev/tmp/multivector-recall-gate.md
```

The JSON includes the same admission-style counters used by DBpedia diagnostics
plus a `markdown_summary` field. The optional Markdown output is intended for
local reports or CI artifacts. DBpedia admission and recall quality remain
separate opt-in checks through the normal benchmark path and `--admission-debug`.
Pass `--markdown-output <path>` on normal DBpedia benchmark runs to write a
PR-ready summary with recall, latency, throughput, and admission-debug
aggregates.

Add `--token-ablation-query-id <qid>` to run one extra query-only retrieval with
`turbohybrid.multivector_debug_admission = summary` and emit the
`multivector_query_token_stats` array from `turbohybrid_last_scan_stats()`.
Pass `--token-ablation-skip-tokens 0,3` to run a second variant with
`turbohybrid.multivector_debug_skip_query_tokens` set for candidate generation
only; exact MaxSim rerank still uses the full query multivector. The top-level
`token_ablation` JSON section records latency, top docs, qrel hits, scan stats,
and per-token raw hits, unique docs, duplicates, and retained-candidate
contribution.

Use `--multivector-plain-fallback off|auto|force` to compare the lossy
token-node path with exact heap MaxSim fallback. `auto` is the PostgreSQL
default and uses `--multivector-plain-fallback-max-docs` plus
`--multivector-plain-fallback-candidate-fraction` to switch when the corpus or
candidate budget is small enough that exact scoring is safer. Admission records
include `plain_fallback_used`, `plain_fallback_reason`,
`plain_fallback_docs_scored`, and `plain_fallback_pairs`.

Use `--multivector-candidate-reservoirs off|conservative|balanced` to compare
score-only document truncation with bounded multi-reservoir retention before
exact MaxSim rerank. Tune
`--multivector-per-token-doc-reservoir-k` and
`--multivector-coverage-reservoir-k` to control the per-query-token, coverage,
and mean seen-similarity reservoirs. Admission records include reservoir union,
duplicate, score, coverage, mean, per-token, and BM25 reservoir counts.

Use `--multivector-bm25-candidate-injection off|hybrid_only|dense_with_text`
to compare the token-node path with BM25-backed candidate admission before exact
MaxSim rerank. `hybrid_only` injects lexical candidates for hybrid
multivector/text queries, while `dense_with_text` also permits text-backed
dense-only MaxSim runs where BM25 is used as an admission safety net rather
than the final scorer. Admission records include
`bm25_injection_enabled`, `bm25_injection_candidates`,
`bm25_injection_retained`, and `bm25_injection_exact_reranked`.

### Precomputed DBpedia ColBERT multivector dataset

After a DBpedia ColBERT run has generated and persisted document/query
multivectors in `dbpedia_colbert_docs` and `dbpedia_colbert_queries`, export the
loaded benchmark database as a Hugging Face-ready dataset:

```sh
nix develop .#bench
DBPEDIA_COLBERT_PGDATABASE=pgturbohybrid_dbpedia_colbert \
  th-dbpedia-colbert-hf-dataset export \
    --output-dir .nix-dev/hf-datasets/dbpedia-colbert-multivector-1m \
    --force
```

To create the PostgreSQL database, generate the 1M document/query multivectors,
and export the Hugging Face-ready dataset in one pass:

```sh
nix develop .#bench
th-dbpedia-colbert-generate-export \
  --database pgturbohybrid_dbpedia_colbert \
  --max-docs 1000000 \
  --max-queries 0 \
  --generation-clients 8 \
  --generation-threads 1 \
  --insert-batch-size 16 \
  --output-dir .nix-dev/hf-datasets/dbpedia-colbert-multivector-1m \
  --force-export
```

By default, this uses the repo-local benchmark inputs already used by the
ColBERT benchmark runs:

- corpus: `.deps/datasets/qdrant-dbpedia-openai3-large-1m`
- queries: `.deps/datasets/beir-dbpedia-entity/queries`
- qrels: `.deps/datasets/beir-dbpedia-entity-qrels/test-positive.tsv`
- model: `.nix-dev/models/colbert-15m/sauerkraut-modern.gguf`

Use `--resume` to continue from an existing partially encoded database, or
`--recreate-database` to drop and recreate the target database before starting.
Add `--repo-id <owner>/dbpedia-colbert-multivector-1m --upload` to publish the
exported dataset after generation.

To publish it, pass the target dataset repo id and `--upload`:

```sh
HF_TOKEN=... \
DBPEDIA_COLBERT_PGDATABASE=pgturbohybrid_dbpedia_colbert \
  th-dbpedia-colbert-hf-dataset export \
    --output-dir .nix-dev/hf-datasets/dbpedia-colbert-multivector-1m \
    --repo-id <owner>/dbpedia-colbert-multivector-1m \
    --upload \
    --force
```

The dataset contains Parquet shards under `docs/`, `queries/`, and `qrels/`.
Document and query rows store PostgreSQL `turbohybrid_multivector` text literals
so later runs can load them into PostgreSQL without llama.cpp embedding
generation:

```sh
DBPEDIA_COLBERT_PRECOMPUTED_DATASET=<owner>/dbpedia-colbert-multivector-1m \
  th-bench-dbpedia-colbert \
    --precomputed-dataset <owner>/dbpedia-colbert-multivector-1m \
    --reuse-data \
    --reuse-embeddings
```

For local smoke testing, use `--source .nix-dev/hf-datasets/dbpedia-colbert-multivector-1m`
with `th-dbpedia-colbert-hf-dataset import`, or pass the same local path to
`--precomputed-dataset`.

Benchmarks that need external datasets or produce host-specific artifacts are
not part of `flake check`. Keep generated JSON, CSV, logs, and Markdown reports
under ignored result directories such as `benchmarks/results/` or outside the
repository.

If you already have a PostgreSQL RAG database and want a quick local comparison,
use `benchmarks/rag_existing.py`. It compares TurboHybrid with your own
retrieval SQL and is documented in
[`docs/benchmarks/bring-your-own-rag.md`](../docs/benchmarks/bring-your-own-rag.md).

For deterministic local retrieval-quality checks that do not require external
data, use `benchmarks/dev/retrieval_quality_grid.sql`. It creates a synthetic
clustered dense/hybrid corpus, rebuilds TurboHybrid indexes across retrieval
profiles, and prints recall/overlap plus selected `turbohybrid_last_scan_stats()`
fields. Treat it as a developer regression harness, not a public benchmark
claim. The grid also includes payload-filtered cases that compare
`turbohybrid.payload_entry_seeding` off/auto/on using existing INCLUDE payload
references, plus `turbohybrid.dense_uncertainty_retry` off/auto/on rows for
bounded second-pass traversal experiments and
`turbohybrid.bm25_heap_tsvector_rerank` off/topk/band/auto rows for
phrase/proximity-like lexical queries. It also includes an opt-in
`turbohybrid.final_diversity = group_payload` row over an int4 `INCLUDE`
payload so duplicate group suppression is visible in the same result table, and
records `turbohybrid_graph_repair_dry_run()` overlap / weak-node /
suggested-edge diagnostics for each built index.

If you already have an eval query table for a real or synthetic workload, use
`benchmarks/dev/tune_retrieval_profile.sql` to sweep query-time retrieval
settings against an existing TurboHybrid index and print a Pareto frontier. The
script expects an `eval_queries`-compatible table with vector/text queries and
expected id arrays, records overlap/recall@K plus selected
`turbohybrid_last_scan_stats()` fields, and can recommend the highest-recall
setting under a supplied p95 latency budget. It is a developer autotuning
harness, not a SQL-visible C autotuner.

For multivector/ColBERT work, use
`benchmarks/dev/multivector_late_interaction.sql`. It builds a deterministic
synthetic `turbohybrid_multivector` corpus and varies `L`, `Q`, exact rerank
`R`, and subvector hit budget `Ks` to make build, approximate candidate
collection, exact rerank, and accumulator-memory slopes visible. The companion
developer note is
[`benchmarks/dev/multivector_late_interaction.md`](dev/multivector_late_interaction.md).

The DBPedia OpenAI3-large benchmark spec lives in
[`dbpedia_openai3_large.md`](dbpedia_openai3_large.md). It covers the
1M-row Qdrant DBPedia corpus, BEIR DBPedia queries/qrels, the native pgvector
`halfvec` + PostgreSQL full-text SQL RRF baseline, the TurboHybrid hybrid runs,
and a dense-only default comparison between pgvector HNSW and TurboHybrid.
Internal dense-only experiment decisions from the dense-only experiment work live in
[`docs/internal/dbpedia-dense-quality-decision.md`](../docs/internal/dbpedia-dense-quality-decision.md).

## Native sparse (SPLADE) retrieval

The DBpedia ColBERT harness (`dbpedia_colbert_multivector.py`) can benchmark the
native sparse (SPLADE-style) retrieval path over the `learned_sparse`
(`turbohybrid_sparse_vector`) column. Pass `--sparse-benchmark` to add a
`sparse_benchmark` section to the JSON output with six methods:

| method | index | description |
| --- | --- | --- |
| `sparse_f32` | sparse-primary, `sparse_quant_bits = 0` | exact f32 sparse inner product |
| `sparse_q16` | sparse-primary, `sparse_quant_bits = 16` | q16 quantized postings |
| `sparse_q8` | sparse-primary, `sparse_quant_bits = 8` | q8 quantized postings |
| `sparse_q8_rerank` | sparse-primary, `sparse_quant_bits = 8` | q8 postings + exact f32 top-band rerank (`turbohybrid.sparse_rerank = auto`) |
| `dense_sparse_rrf` | ColBERT multivector + sparse | RRF fusion of MaxSim + sparse |
| `dense_sparse_bm25_rrf` | ColBERT multivector + sparse + BM25 | RRF fusion of MaxSim + sparse + BM25 |

The `sparse_f32/q16/q8` methods are sparse-only and run against a turbohybrid
**sparse-primary** index (no dense graph key); the `dense_sparse_*` methods fuse
the ColBERT branch with sparse (and BM25) and require the `colbert` column.
Methods whose required columns are unpopulated are reported under `skipped`, so a
sparse-only corpus still produces the four `sparse_*` results.

Each method records, in the open result schema:

- IR quality: `recall@10`, `ndcg@10`, `mrr@10`, `map@10` (when qrels are present),
- `latency`: `mean_ms`, `p50_ms`, `p95_ms`, `p99_ms`, `qps`,
- `index_bytes`, `index_name`, `sparse_quant_bits`,
- `sparse_stats`: `sparse_quant_bits`, `sparse_quant_mode`, `sparse_score_kernel`,
  `sparse_used_wand`, `sparse_branch_used`, and summarized counters
  `sparse_postings_touched`, `sparse_candidates_scored`, `sparse_wand_pruned`,
  `sparse_exact_rerank_count` (from `turbohybrid_last_scan_stats()`).

The six methods are also accepted directly in `--methods` (e.g.
`--methods sparse_q8,dense_sparse_rrf`): sparse method names are routed to this
phase automatically (they need their own per-quantization indexes, so they run
here rather than in the single-ColBERT-index `--methods` loop), and their results
appear under `sparse_benchmark`. `--sparse-benchmark` is shorthand for "run all
six".

Flags: `--sparse-methods a,b,c` selects a subset (default: all six);
`--sparse-k N` sets the sparse candidate budget (`sparse_k`; `0` ⇒ `final_k`).

### q8 / q16 / f32 comparison

Running all four sparse-only methods on one corpus compares the
quality/latency/size trade-off of exact f32 postings vs q16/q8 quantization, and
whether the q8 + exact-rerank method (`sparse_q8_rerank`) recovers the recall lost
to q8 quantization. Compare `recall@10`/`ndcg@10` against the `index_bytes` and
`latency.p95_ms` of each method; `sparse_q8_rerank`'s `sparse_stats.sparse_exact_rerank_count`
shows how many candidates were re-scored exactly.

### Importing external SPLADE vectors (no model downloads)

`learned_sparse` data is model-agnostic: import any `(id, term_ids, weights)`
JSONL produced by [`export_learned_sparse_jsonl.py`](export_learned_sparse_jsonl.py),
which has two adapters and downloads no model:

- `hash_plumbing` — deterministic stdlib-only toy features for pipeline tests
  (marked `plumbing_only=true`; never serving evidence),
- `external_command` — pipes `{"kind","id","text"}` rows to a user-supplied SPLADE
  command that returns `{"id","term_ids","weights"}` rows.

The JSONL is loaded into `dbpedia_colbert_docs.learned_sparse` /
`dbpedia_colbert_queries.learned_sparse` via
`turbohybrid_sparse_vector_from_arrays(term_ids, weights)`.

### Reproducing a small local sparse benchmark

You do not need the full DBpedia corpus to exercise the sparse methods. Populate
`dbpedia_colbert_docs`/`dbpedia_colbert_queries` with a handful of rows whose
`learned_sparse` column is built with `turbohybrid_sparse_vector_build(term_ids,
weights)` (and, for the `dense_sparse_*` methods, a small `colbert` multivector
via `turbohybrid_multivector_from_float4(values, dim)` plus a `body_tsv`), then
call `run_sparse_retrieval_benchmark(conn, args, queries, qrels)` directly. With
`learned_sparse` present on a few docs and queries, the four `sparse_*` methods
build their sparse-primary indexes and return non-empty `sparse_stats`; with
`colbert` present too, the two `dense_sparse_*` fused methods also run. This path
runs against the stub `llama_embed_sparse` output or any imported JSONL, so it
needs no model download.

## Baselines

Every publishable run should include at least:

- `postgres_sql_rrf`: pgvector HNSW over `vector`, PostgreSQL full-text search
  over `tsvector`, and SQL reciprocal-rank fusion.
- `pgturbohybrid`: one pgturbohybrid index over the same `vector` and
  `tsvector` columns.

When relevant, also compare `pgturbohybrid_recovered_explicit` to document the
latency, storage, and quality tradeoff from the recovered fast settings: 4-bit,
`exact_storage = off`, 100/100/60, and `final_k = 10`. The package default keeps
adaptive dense widening off; the harness includes explicit adaptive variants for
controlled recovery experiments. The harness still accepts
`pgturbohybrid_exact_storage_off` as a legacy alias for older artifacts.

## Profile Matrix

Publishable FIQA/OpenAI results should include all of:

- `pgturbohybrid`: latency profile, default index options, omitted query
  budgets, and LIMIT-inferred `final_k`.
- `pgturbohybrid_recovered_explicit`: latency profile, 4-bit index,
  `exact_storage = off`, effective `dense_k = 100`, `bm25_k = 100`, and
  `final_k = 10`.
- `pgturbohybrid_adaptive_auto_2_0`: same FIQA/OpenAI latency-profile index
  and budgets, with adaptive dense widening explicitly enabled in `auto` mode
  at multiplier `2.0`. Use this as an opt-in quality recovery diagnostic, not
  as the package default.
- `pgturbohybrid_quality`: quality profile, effective `dense_k = 400`,
  `bm25_k = 400`, exact-safe BM25 paths, SIMD enabled, and a documented
  `exact_storage` choice. Prefer `exact_storage = on` when evaluating final
  quality-sensitive settings.
- `postgres_sql_rrf`: pgvector HNSW plus PostgreSQL full-text search fused in
  SQL.

The result summary must compare quality profile against latency profile for
both relevance and latency. Do not infer quality-profile relevance from the
latency-profile artifact.

Validate a generated matrix artifact with:

```sh
FIQA_DATASET=/path/to/fiqa
PGDATABASE=pgturbohybrid_fiqa
OUTPUT=benchmarks/results/pgturbohybrid-fiqa.json
python3 benchmarks/tools/check_acceptance.py \
  "$OUTPUT" \
  --suite fiqa_openai_profile_matrix
```

## Profile recall/latency grid (AMD64)

`benchmarks/dev/profile_recall_latency_grid.sql` is a deterministic developer
harness for sanity-checking the named profiles (`latency`, `balanced`,
`matched_recall`, `high_recall`, `quality`, `debug`) on a host. It generates two
throwaway corpora — an *easy* doc/chunk corpus (easy dense recall, grouped/
duplicate, payload-filter, lexical, hybrid) and a *hard* spread corpus with one
out-of-distribution query — builds one throwaway index per profile, and prints a
report. It changes no profile defaults and no runtime behavior.

### How to run (AMD64/x86_64)

```sh
createdb prof_grid
psql -d prof_grid -f benchmarks/dev/profile_recall_latency_grid.sql            # SIMD_MODE=both (default)
psql -d prof_grid -v SIMD_MODE=on  -f benchmarks/dev/profile_recall_latency_grid.sql
psql -d prof_grid -v SIMD_MODE=off -f benchmarks/dev/profile_recall_latency_grid.sql
dropdb prof_grid
```

Optional psql variables: `SIMD_MODE` (`on|off|both`, default `both`), `DIMS`
(default 1536 — keep `>=1024` so the 4-bit u8 AVX2/AVX-512-VNNI path is
exercised), `NROWS_EASY`, `NROWS_HARD`, `K`, `ITERS`. It is deterministic
(sin/cos over `generate_series`, no `random()`), so reruns reproduce the same
recall/grouping; only the latency columns vary with machine load.

The report has three blocks: build reloptions + effective feature modes
(`retry`/`residual`/`bm25_rerank`/`final_diversity`), recall/overlap/latency, and
a SIMD on-vs-off parity check.

### Do not commit the output

This script prints a report and leaves no files; do not paste its output, or any
captured tables/JSON from it, into the repo. Numbers depend on host CPU, memory,
and load, and are not project benchmark claims. (Markdown under `benchmarks/` is
gitignored for this reason.) The script and this README are the committed
artifacts — the results are not.

### Interpreting high_recall vs latency

On the *easy* corpus every profile reaches recall 1.0, so the profiles are only
separated by the *hard* out-of-distribution query. There, `latency`/`balanced`/
`matched_recall`/`quality` land around recall 0.6 while `high_recall` recovers to
~1.0, at roughly 2.4x the per-query latency. That recall win comes from
`high_recall`'s wider `graph_ef_search`/`graph_oversampling` (and heuristic build),
not from `dense_uncertainty_retry`, residual rerank, BM25 heap-tsvector rerank, or
final diversity — those stay at their defaults in this grid (they are opt-in or
profile-gated; benchmark each separately on your own data before enabling). Read
it as: pick
`latency` for throughput, `high_recall` when hard/ambiguous-query recall matters
and you can absorb the latency; `matched_recall` targets full-HNSW-matched recall
and should be compared on real data, not this synthetic.

### Why no profile default is changed

The grid only `SET`s `turbohybrid.profile`/`turbohybrid.simd` for the session and
builds throwaway indexes; it does not (and must not) edit any profile's compiled
defaults. A synthetic micro-grid is not sufficient evidence to retune a shipped
profile — any change to profile defaults must be justified by real-dataset
recall/latency runs, not by this harness. Treat its output as a smoke test and a
parity check, and record any tuning ideas as recommendations, not commits.

## Deciding residual rerank vs heap rescore

`benchmarks/dev/residual_rerank_grid.sql` is the harness to decide whether the
in-index calibrated residual rerank is worth recommending on exact-free 4-bit
indexes. It builds `residual_rerank=off / 16 / 32 / 64`-byte indexes and compares
`dense_residual_rerank_mode = off | fixed | calibrated` against
`dense_heap_rescore = off | topk | band`, reporting recall@k, p50/p95, the
residual stats (`residual_rerank_band`, `residual_rerank_reordered_count`,
`residual_rerank_topk_changed`, `dense_residual_rerank_us`), heap rescore
count/us, index size, and the `turbohybrid_estimate_memory` estimate. Use it (on
your data) to decide between the cheaper in-index residual rerank and the exact
heap-band rescore: residual rerank reorders a narrow top band at microsecond cost
but recovers recall only when the true neighbours are already in that band, while
heap-band rescores the full candidate pool from the heap (exact, higher latency)
and recovers more. As with the other dev grids, no index or profile default
should change from its synthetic output alone — validate on real data first.

### Residual rerank vs heap-band rescore

Three levers correct an exact-free 4-bit recall miss, and they act at different
points in the pipeline — pick by *where* the true neighbours are:

- **Residual rerank** (`residual_rerank = on`; `dense_residual_rerank_mode =
  fixed | calibrated`) is a *cheap top-band refinement*. It reorders the narrow
  top band (~`2 * final_k`) from a few residual bytes stored in the index, at
  microsecond cost and no heap I/O. Because it only re-ranks *within* that band,
  it recovers recall only when the true neighbours are already in it.
- **Heap-band rescore** (`dense_heap_rescore = band`) is the *stronger
  recall/ranking recovery*. It fetches exact vectors from the heap and rescores
  the full candidate band, so it recovers neighbours that are present in the
  wider heap band but were mis-ranked out of the narrow residual band — at the
  cost of heap I/O and higher latency.
- **Wider `graph_ef_search` / `graph_oversampling`** is what you need when the
  true neighbours are *absent from the candidate pool altogether*: no rescore can
  recover a neighbour the graph search never visited.

In the synthetic run that motivated this note, residual rerank (16/32/64 bytes,
both `fixed` and `calibrated`) reordered the band cheaply but left recall
unchanged, while heap-band rescore recovered the missing neighbours — because the
exact top-k sat in the wider candidate band, outside the narrow residual band. So
residual rerank is **not a substitute for heap-band rescore** when the neighbours
are outside the residual band, and neither is a substitute for widening graph
search when they are outside the pool. Index size also grows with
`residual_rerank_bytes`, so the residual sketch is a build-time storage cost
(`REINDEX` to change it). That ordering is data-dependent — re-run the grid on
your corpus; the takeaway is the decision rule, not the numbers.

## Phrase/proximity BM25 rerank

`benchmarks/dev/bm25_phrase_rerank_grid.sql` is the companion harness for the
`bm25_heap_tsvector_rerank` decision (off/topk/band/auto vs rrf/calibrated on
phrase/proximity queries). Same rule: it is a smoke/decision tool, not a source of
committed results or default changes.

## Deciding whether to widen graph search, use residual rerank, or use heap-band rescore

`benchmarks/dev/dense_candidate_miss_grid.sql` attributes a dense recall miss to
its *cause*, so you can tell which lever fixes it before reaching for one. For each
query it computes the exact dense top-k (seqscan `<=>`) and then probes three
containment questions on the actual index:

- **`candidate_pool_contains_exact_topk`** — are the exact neighbours even in the
  raw 4-bit graph candidate pool (top `dense_k`, no rescore)? If not, the graph
  search did not *reach* them.
- **`residual_band_contains_exact_topk`** — are they inside the narrow residual
  rerank band (top ~`2*final_k`)? Residual rerank can only reorder within that band.
- **`heap_band_contains_exact_topk`** — does heap-band rescore (exact, over the
  whole candidate pool) recover them?

From those it prints one decision label per case:

| label | meaning | lever |
| --- | --- | --- |
| `graph_search_sufficient` | base recall already 1.0 | none needed |
| `candidate_generation_miss` | exact top-k not in the pool at all | widen `graph_ef_search` / `graph_oversampling` |
| `residual_band_too_narrow` | in the pool but outside the residual band | residual rerank won't help — use heap-band rescore |
| `quantized_misorder_fixed_by_heap` | in pool and band, base recall < 1 | heap-band (or residual) re-ranks it in |
| `payload_filter_underfilled` | fewer category matches than `k` in the pool | widen the candidate budget or the filter, not the rescore |

The treatment sweep then *confirms* which lever actually moves recall: it rebuilds
the index across the four profiles, an `graph_ef_search` ladder (64→384), an
`graph_oversampling` ladder, and a residual-32 base index, and reports recall@k
plus `graph_visited_nodes`/`graph_scored_codes` and per-stage µs for each. The key
reading: if recall stays flat across the whole ef/oversampling ladder but heap-band
recovers it, the miss is quantization mis-ranking, not graph reach — widening graph
search wastes latency and you should rescore instead; only a `candidate_generation_miss`
(exact top-k absent from the pool) is the case widening ef/oversampling is meant to
fix. Run:

```
createdb cmiss ; psql -d cmiss -f benchmarks/dev/dense_candidate_miss_grid.sql ; dropdb cmiss
```

It is deterministic (no `random()`), prints a report, writes no files, and changes
no defaults. As with the other dev grids, validate on real data before changing any
index or profile setting.

## Publishable Run Metadata

Do not commit generated benchmark outputs. Store JSON/Markdown in an external
artifact store or ignored directories such as `benchmarks/results/`.

### Required provenance helper

Every new benchmark claim **must** carry a machine-readable provenance block so
a reader can reproduce it. Use the shared helper `benchmarks/bench_metadata.py`
rather than hand-rolling the fields:

- **Python drivers** call `bench_metadata.collect(query=..., dataset=...,
  rows=..., dimensions=..., query_count=..., warmup_passes=...,
  measured_passes=..., cache_state=..., index_reloptions=...)` and embed the
  returned dict in the result JSON (see `fiqa_openai.py`, which emits it under
  `provenance`).
- **SQL / shell benchmarks** capture it as a sidecar JSON:

  ```sh
  python3 benchmarks/bench_metadata.py --database "$PGDATABASE" \
    --suite my-bench --dataset glove-100-angular --rows 1183514 \
    --dimensions 100 --query-count 10000 --warmup-passes 1 \
    --measured-passes 3 --cache-state warm > my-bench.provenance.json
  ```

The helper fills git commit + dirty-tree status, host OS / arch / CPU model,
PostgreSQL / pgvector / pgturbohybrid versions, and any non-default
`turbohybrid.*` GUCs automatically; you supply the dataset and run fields.

Record the following with any published result (the helper covers most of it):

- hardware and CPU governor
- operating system
- PostgreSQL version and settings
- pgvector ref or release
- pgturbohybrid commit
- dataset and embedding model
- corpus size, query count, and embedding dimensions
- retrieval profile
- index settings and reloptions
- candidate budgets, fusion settings, and final result target
- exact commands
- warmup policy and measured run count. Latency-only document-node runs support
  `--serving-latency-warmup-queries` and report warmup/cold latency separately
  from the measured steady-state latency, plus cache-build and native-cache reuse
  counts.
- p50, p95, p99, QPS
- index size, build time, and WAL generated
- build provenance for fresh, shared, and existing-index runs
- exact-build provenance from TurboHybrid index stats when applicable
- quality metrics such as recall, nDCG, MRR, MAP, or overlap
- baseline definitions and index options
- note that results vary by dataset and hardware

## Benchmarking dense kernels

The dense scoring kernel is auto-selected per host (the best available SIMD tier
plus the u8 x4 batch path). To compare kernels apples-to-apples, pin the
implementation with the developer/benchmark GUCs (the "Developer / benchmark"
category in
[`docs/architecture.md`](../docs/architecture.md#gucs-and-reloptions)) and
confirm the kernel that actually ran with `turbohybrid_last_scan_stats()`. All
combinations below return identical results; only the kernel that produces them
changes -- so a difference in timing is a kernel difference, not a recall one.

`dense_query_split_impl` and `dense_u8_split` default to `auto`; set them only to
override the auto choice. Apply the `SET`s per session, then run the workload.

| Target kernel | GUCs to set |
| --- | --- |
| Host best (default) | leave all unset (`auto` -> best tier + u8 x4) |
| AVX-512 VNNI, u8 x4 | `dense_graph_avx512vnni=on; dense_graph_avxvnni=on; dense_u8_split=auto; dense_u8_batch_x4=on` |
| AVX2, u8 x4 | `dense_graph_avx512vnni=off; dense_graph_avxvnni=off; dense_u8_split=auto; dense_u8_batch_x4=on` |
| u8 x4 off (four single-node u8 passes) | same as the chosen tier, but `dense_u8_batch_x4=off` |
| Signed split (no u8) | `dense_u8_split=off` (signed-codebook split at the host's tier) |
| Scalar / LUT fallback | `turbohybrid.simd=off` (the non-SIMD reference) |

Forcing AVX-512 off always selects the AVX2 kernels on any AVX2 host; on a host
without AVX-512 VNNI the AVX-512-on row resolves to those same AVX2 kernels.

Verify the active kernel after a scan:

```sql
SELECT turbohybrid_last_scan_stats() ->> 'dense_scorer';      -- e.g. unsigned_split_avx512vnni
SELECT turbohybrid_last_scan_stats() -> 'dense' -> 'kernels'; -- u8_batch_mode, u8_kernel_single/_batch
```

Ready-made harnesses live next to this README:

- `dense_only_vs_hybrid_shape.sql` -- one-key dense-only index vs the old
  fake-`tsvector` hybrid shape: build time, index size, vector-query latency,
  BM25 branch stats, and overlap.
- `fair_dense_bench.sql` -- fair dense-only vs hybrid-shape dense retrieval
  with the same vectors and query set, 1-client and 8-client dblink runs,
  build time, index size, exact precision@K, and scan stats.
- `concurrency_diagnosis.sql` -- one-backend per-query printout of dense native
  cache scope/reuse/build, scan-lock wait, buffer-lock wait, page reads, and
  traversal timing fields.
- `native_hotpath_bench.sql` -- live native-scan latency, u8 x4 on vs off.
- `u8_x4_kernel_microbench.sql`, `u8_split_microbench.sql` -- kernel-level
  ns/code microbenchmarks.
- `native_scan_kernel_stats.sql` -- per-bucket kernel attribution from a real
  scan.
- `rescore_band_latency.sql` -- exact-rescore (`dense_rescore_band`) latency.
- `concurrent_dense_bench.py` -- client-side dense concurrency harness with
  persistent PostgreSQL connections, per-connection warmup, timed steady-state
  runs, cache-scope/cache-size/EF sweeps, optional exact precision, and CSV/JSON
  output.
- `glove100_recall_latency_grid.sql` -- dense-only recall/latency profile grid
  for glove-like 100-dimensional workloads. It compares `default`, `balanced`,
  `matched_recall`, `quality`, `exact_storage`, `residual_rerank`, and
  heap-rescore top-k/band, recording build time, index size, precision@K against
  exact pgvector ordering, p50/p95/p99, and dense scan stats such as build
  neighbor selection, graph EF, oversampling, scored codes, exact rescore count,
  heap rescore count, and exact rescore source. This benchmark is the authority
  for deciding whether `matched_recall` is the right one-knob default for a
  workload.
- `dev/retrieval_quality_grid.sql` -- deterministic synthetic dense/hybrid
  quality harness. It includes explicit matched-recall rows for 4-bit baseline,
  4-bit residual rerank, opt-in scalar 8-bit (`quantization_bits = 8`), and
  `exact_storage = on` so recall changes can be attributed to graph topology,
  quantization width, residual rerank, or exact final ranking without external
  datasets.
- `dev/tune_retrieval_profile.sql` -- practical query-time retrieval tuner for
  an existing TurboHybrid index. It consumes an `eval_queries` table, sweeps
  profiles, dense/BM25 budgets, fusion, residual rerank mode, and heap rescore
  mode when available, then prints all trials, the Pareto frontier, and an
  optional latency-budget recommendation.
- `dev/multivector_late_interaction.sql` -- deterministic multivector /
  ColBERT slope harness. It varies document token vectors `L`, query vectors
  `Q`, exact rerank docs `R`, and subvector hits `Ks`, then reports build time,
  index size, recall sanity, raw hits, unique docs, exact pairs, and accumulator
  memory estimates.
- `native_segments_bench.sql` -- native graph segment-count sweep
  (`native_segments = 1,2,4,8` by default). It records build time, index
  size, precision@K against exact ordering, p50/p95, segment count/search
  stats, and page/scoring stats so the build-speed, recall, and query-cost
  tradeoff is visible.
- `concurrency_dense_bench.sql` / `concurrency_dense_bench.sh` -- concurrent-client
  scaling diagnostics (see below).

## Concurrent-client scaling diagnostics

Use `concurrent_dense_bench.py` when the result needs end-to-end client-side
RPS/latency and machine-readable artifacts. It opens all client connections
first, warms every connection, starts the timed phase only after all connections
are warm, and writes both CSV and JSON under `benchmarks/output/`. The first
warm query records cold-cache signals such as `native_cache_built_this_scan` and
`native_cache_build_us`; timed RPS and p50/p95/p99 are measured after that
warmup, so cache build artifacts are visible but do not pollute steady-state
concurrency.

```bash
# Synthetic fallback, glove-like default shape (1,183,514 x 100):
uv run benchmarks/concurrent_dense_bench.py \
  --dsn pgturbohybrid_benchmark \
  --clients 1,8 \
  --native-cache-scopes off,per_backend,shared \
  --native-cache-max-mb 0,64,512 \
  --native-segments 1,2,4,8,10 \
  --graph-ef-search 64,96,128

# Existing glove table, copied into an isolated benchmark table:
uv run benchmarks/concurrent_dense_bench.py \
  --dsn pgturbohybrid_benchmark \
  --source-table items \
  --source-vector-column embedding \
  --rows 1183514 \
  --dimensions 100 \
  --compute-ground-truth
```

For quick smoke tests, lower `--rows`, `--query-count`, `--warm-queries`, and
`--timed-queries`. If `--compute-ground-truth` or `--ground-truth-table` is
provided, the output includes `precision_at_k_avg` and `precision_at_k_min`;
otherwise those columns are empty. The CSV columns include RPS, p50/p95/p99,
warm and timed native-cache build indicators, `graph_batch_us`,
`graph_base_us`, `graph_traverse_us`, `graph_code_pages_read`, and
`graph_adj_pages_read`; the JSON also includes per-client durations and sampled
first/last scan stats.

`concurrency_dense_bench.sql` (driven by `concurrency_dense_bench.sh`) explains
why dense-default throughput can scale *down* with concurrent clients on
glove-100-angular (observed ~325 RPS / p95 4.4 ms at 1 client collapsing to
~127 RPS / p95 179 ms at 8 clients) while pgvector and Qdrant scale up. It does
not change query behaviour -- it only measures, so the cause is known before any
algorithm is touched.

It drives the same fixed query set at 1/2/4/8/16 concurrent backends (each a real
backend opened via `dblink`) across native cache scopes (`per_backend`, `shared`,
and `off`), cache caps for cached paths, and two prewarm modes (A = cold, B =
cache prebuilt). It attributes the p95 explosion to one of four causes using
per-backend instrumentation from `turbohybrid_last_scan_stats()`:

The production default `turbohybrid.native_cache_scope=auto` resolves to the
shared mmap cache on supported platforms when the native working set fits
`turbohybrid.native_cache_max_mb`. Use `SELECT turbohybrid_prewarm('idx'::regclass)`
before a timed run or before admitting traffic to build/attach that shared cache
outside the first user query; the function returns JSON with `native_cache_built`,
`native_cache_attach_us`, `native_cache_build_us`, and resident byte counts.

| Suspected cause | Signal the harness reads |
| --- | --- |
| Cold per-backend cache build | `native_cache_built_this_scan` / `native_cache_build_us` on the cold query; removed by prewarm mode B |
| Cache duplication / memory bandwidth | `native_cache_scope='per_backend'`, `native_cache_used=true`, `native_cache_reused=true`, `native_cache_bytes` per backend × clients (`total_cache_bytes`); warm, 0 page reads, `graph_total_us` rising with clients; `per_backend` collapses while `shared` or `off` scales |
| Lock waits | `graph_scan_lock_wait_us` for the `PGTURBOHYBRID_GRAPH_SCAN_LOCK`, plus `pg_stat_activity` `wait_event_type='Lock'/'LWLock'` and ungranted `pg_locks` on the index |
| Shared cache coordination | `native_cache_attach_us`, `native_cache_wait_us`, and `native_cache_build_us`; high wait means clients are waiting for the first shared-cache builder |
| Buffer/page loading waits | `code_buffer_lock_wait_us`, `adj_buffer_lock_wait_us`, and `graph_*_pages_read`; compare `native_cache_scope=per_backend` and `shared` against `native_cache_scope=off` |
| Traversal / scoring CPU | `graph_batch_us` / `graph_traverse_us` per query rising with clients, no waits, ~0 page reads |

The `native_cache_*` keys it relies on are emitted by
`turbohybrid_last_scan_stats()` (flat keys and under `dense.cache`):
`native_cache_policy` / `native_cache_scope` (`auto` / `per_backend` / `shared` / `off`),
`native_cache_used`, `native_cache_reason`,
`native_cache_scope` (`per_backend` / `shared` / `per_scan` / `none`),
`native_cache_reused`, `native_cache_built_this_scan`,
`native_cache_attach_us`, `native_cache_build_us`, `native_cache_wait_us`,
`native_cache_refcount`, and `native_cache_bytes` with a `code`/`adj`/`exact`
breakdown. `native_cache_mode` is retained for compatibility and reports
`uncached` for the same condition that `native_cache_scope` calls `per_scan`.

For a quick single-backend instrumentation check, run:

```bash
psql -d pgturbohybrid_benchmark \
  -v TBL=items -v VCOL=embedding -v POLICY=per_backend -v CACHE_MB=512 \
  -f benchmarks/concurrency_diagnosis.sql
psql -d pgturbohybrid_benchmark \
  -v TBL=items -v VCOL=embedding -v POLICY=shared -v CACHE_MB=512 \
  -f benchmarks/concurrency_diagnosis.sql
psql -d pgturbohybrid_benchmark \
  -v TBL=items -v VCOL=embedding -v POLICY=off -v CACHE_MB=512 \
  -f benchmarks/concurrency_diagnosis.sql
```

For an external pgbench or Python concurrency benchmark, use a two-phase run:
open all client connections first, run enough warmup queries on every
connection to build that backend's cache, then reset client-side timers and run
the timed phase over the same fixed query set. Run the timed phase three ways:
`SET turbohybrid.native_cache_scope=per_backend` for the backend-local cache,
`SET turbohybrid.native_cache_scope=shared` for the mmap-backed shared cache,
and `SET turbohybrid.native_cache_scope=off` for per-scan page loading. Capture
`turbohybrid_last_scan_stats()` on each connection after its final warm query
and final timed query. If p95/p99 spikes disappear after warmup, cold
`native_cache_build_us` dominated; if `per_backend` collapses but `shared` or
`off` does not, suspect per-backend cache duplication or memory bandwidth; if
`graph_scan_lock_wait_us` grows, the page-level scan lock is visible; if
`code_buffer_lock_wait_us` / `adj_buffer_lock_wait_us` and page reads grow,
buffer/page loading is the culprit; otherwise compare `graph_traverse_us` and
`graph_total_us` for steady-state traversal CPU.

```bash
# Against the loaded glove-100-angular dataset (real collapse):
benchmarks/concurrency_dense_bench.sh pgturbohybrid_benchmark

# Or build + run a synthetic glove-sized dataset in a throwaway DB:
CCB_NROWS=1183514 CCB_DIMS=100 \
  benchmarks/concurrency_dense_bench.sh pgturbohybrid_ccbench
```

It prints three tables: the **scaling curve** (RPS + p50/p95/p99 vs clients), a
**cause-attribution** table (cold build µs, wait %, code pages, scoring µs, cache
bytes per backend × clients), and a heuristic **verdict** per config. It measures
server-side latency via `dblink` (no client round-trip), so absolute RPS differs
from the Python `vector-db-benchmark` numbers; the scaling *shape* and the
per-backend cause signals are the point. Reproducing the per-backend cache
collapse needs a glove-sized index (~1.18M × 100) -- a small synthetic dataset
stays cache-resident and scales fine, which the harness correctly shows.

## Acceptance Checks

Fast defaults must pass the FIQA/OpenAI quality gate before they are used for a
published claim. The gate is configured in
`config/acceptance_thresholds.json` and is intended for a full manual or
nightly FIQA run, not for per-PR perf smoke.

Run it against the generated full benchmark artifact:

```sh
OUTPUT=benchmarks/results/pgturbohybrid-fiqa.json
python3 benchmarks/tools/check_acceptance.py \
  "$OUTPUT" \
  --suite fiqa_openai_fast_defaults
```

The artifact must include `pgturbohybrid_recovered_explicit` latency-profile
results and the SQL RRF baseline with `nDCG@10`, `MRR@10`, `p95_ms`, and either
`Recall@10` or `overlap@10` versus SQL RRF. If the gate fails, evaluate
`quality` profile, `exact_storage = on`, or larger dense/BM25 budgets before
publishing the fast default result.

For dense-only comparisons against Qdrant or pgvector, include a matched-quality
grid rather than only the speed-first 4-bit exact-free default. At minimum,
report the `glove100_recall_latency_grid.sql` rows for `default`, `balanced`,
`matched_recall`, `quality`, `exact_storage`, `residual_rerank`, and
heap-rescore top-k/band, including build time and index size. Treat `latency` as
the compact fast default, use `matched_recall` for pgvector/Qdrant-style recall
comparisons without full-vector storage, and use the grid to show what it costs
to approach or exceed the external baseline's recall.

## Dataset Notes

`config/datasets.json` lists intended quality and systems datasets. FIQA, BEIR,
MS MARCO, MIRACL, LoTTE, and RAG sets should be run as reproducible
experiments with committed commands and external result artifacts.

For 3,072-dimensional DBPedia/OpenAI3-large runs, use
`benchmarks/dbpedia_openai3_large.py` rather than the FIQA harness. The native
PostgreSQL hybrid baseline uses pgvector `halfvec(3072)` HNSW plus PostgreSQL
full-text search because standard pgvector `vector` HNSW is not the intended
ANN path for this dimensionality.

For the DBPedia dense-only default comparison, use
`--methods pgvector_halfvec_dense_only,pgturbohybrid_dense_only,pgturbohybrid_dense_adaptive_auto_1_25,pgturbohybrid_dense_exact_storage_on`.
That run does not pass a text query to TurboHybrid and should not be reported as
hybrid retrieval. The adaptive row is opt-in diagnostic behavior, and the
exact-storage row is an upper-bound reference, not a compact default candidate.

For an optional external-library reference, use
`benchmarks/dbpedia_turbovec.py` against the same loaded DBPedia query set. It
builds a Turbovec `TurboQuantIndex(dim=3072, bit_width=4)` from the Qdrant
Parquet embeddings and reports dense-only latency and quality metrics. Keep
that row separate from PostgreSQL-native comparisons because Turbovec runs
in-process and does not exercise PostgreSQL storage, MVCC, indexing, or SQL
execution.

Synthetic vector generators are intentionally not part of the benchmark suite.
They are too far from real retrieval workloads for project performance claims.
The bring-your-own RAG benchmark is different: it is a local evaluation helper
for existing user data, not a source of public project benchmark claims.
