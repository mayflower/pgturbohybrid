# DBPedia Dense-Quality Decision Report

This file helps maintainers understand which dense-only DBPedia quality
experiments are worth keeping, tuning, or reverting before they become public
claims or release defaults.

This is an internal engineering report. It summarizes local development
artifacts from the Qdrant DBPedia OpenAI3-large 1M benchmark. It is not a
release-facing benchmark claim.

## Scope

- Dataset: `Qdrant/dbpedia-entities-openai3-text-embedding-3-large-3072-1M`
- Embedding model: OpenAI `text-embedding-3-large`
- Dimensions: 3,072
- Corpus rows: 1,000,000 for full runs unless noted
- Query mode: `qdrant-self`
- Main quality metric: self-qrel `nDCG@10` and recall/source recovery at 10
- Main latency metrics: p50, p95, p99, and mean single-query latency

The self-query setup uses an existing corpus embedding as the query embedding
and marks the same source row as relevant. It is useful for finding dense graph
reachability and reranking failures. It is not a human relevance benchmark.

## Current Decision

Keep:

- vector-only scan option handling for `turbohybrid_query(...)`
- benchmark harness improvements
- adaptive latency widening as an explicit experimental GUC and benchmark
  method, not as the dense default
- `turbohybrid.dense_build_exact_distances` as an experimental quality mode
- exact-build provenance in `turbohybrid_index_stats(...)`
- failed-method recording in benchmark artifacts
- compact default graph behavior after the graph edge-construction fixes and
  bounded routing-entry scan fix

Keep experimental or revisit:

- one-hop local expansion as a diagnostic knob
- level-0 backbone tuning through explicit `graph_backbone = on`
- data-aware entry sidecar, off by default and internal/experimental only
- residual rerank sidecar, off by default and internal/experimental only

Do not make default:

- full exact storage
- residual rerank sidecars
- entry sidecar representatives
- query-time widening that materially increases p95/p99

Do not publish:

- DBPedia dense-quality numbers as public project claims yet
- Turbovec numbers as PostgreSQL-native or hybrid retrieval results
- batch-mode throughput as an interactive RAG metric

## Main Result Summary

| Variant | Rows | Queries | nDCG@10 | Source top10 | p50 ms | p95 ms | p99 ms | Index bytes | Build ms | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| current default fast path, adaptive off | 1,000,000 | 1,000 | 0.947 | 0.947 | 1.481 | 3.842 | 8.747 | 2,377,760,768 | reused | keep as default guardrail |
| adaptive auto 2.0 opt-in recovery | 1,000,000 | 1,000 | 0.970 | 0.970 | 1.621 | 3.926 | 6.183 | 2,377,760,768 | reused | keep as opt-in diagnostic |
| adaptive auto 1.25 opt-in recovery | 1,000,000 | 1,000 | 0.962 | 0.962 | 1.557 | 4.209 | 7.490 | 2,377,760,768 | reused | keep as opt-in diagnostic |
| historical adaptive auto run | 1,000,000 | 1,000 | 0.970 | 0.970 | 1.965 | 5.491 | 8.624 | 2,377,760,768 | reused | stale default-candidate artifact |
| bounded routing full, adaptive off before scan rerun | 1,000,000 | 1,000 | 0.947 | 0.947 | 7.979 | 21.972 | 28.564 | 2,377,760,768 | 764,249 | stale tail-latency artifact |
| current default compact path, before bounded routing entries | 1,000,000 | 1,000 | 1.000 | 1.000 | 15.471 | 567.737 | 1751.226 | 2,368,987,136 | reused | stale tail-latency artifact |
| current default compact path, bounded routing smoke | 2,000 | 20 | 1.000 | 1.000 | 0.985 | 2.234 | 4.528 | 6,455,296 | 839 | kept as routing smoke |
| broad prompt-pack smoke, default compact path | 2,000 | 20 | 1.000 | 1.000 | 1.054 | 3.019 | 3.627 | 6,455,296 | 597 | kept as prompt-pack smoke |
| adaptive widening auto 1.5x | 1,000,000 | 1,000 | 1.000 | 1.000 | 13.533 | 499.052 | 1604.556 | 2,368,987,136 | reused | no default change |
| adaptive widening auto 2.0x | 1,000,000 | 1,000 | 1.000 | 1.000 | 12.916 | 293.934 | 977.213 | 2,368,987,136 | reused | no default change |
| adaptive widening forced 2.0x | 1,000,000 | 1,000 | 1.000 | 1.000 | 17.253 | 578.712 | 2011.251 | 2,368,987,136 | reused | reject as default |
| local expansion forced top 4 | 1,000,000 | 1,000 | 1.000 | 1.000 | 20.587 | 491.729 | 1386.958 | 2,368,987,136 | reused | reject as default |
| local expansion forced top 8 | 1,000,000 | 1,000 | 1.000 | 1.000 | 12.058 | 253.984 | 1087.079 | 2,368,987,136 | reused | no default change |
| local expansion forced top 16 | 1,000,000 | 1,000 | 1.000 | 1.000 | 11.908 | 229.258 | 972.602 | 2,368,987,136 | reused | no default change |
| local expansion auto top 8 | 1,000,000 | 1,000 | 1.000 | 1.000 | 12.185 | 218.267 | 910.234 | 2,368,987,136 | reused | no default change |
| earlier default compact path, pre edge-construction fix | 1,000,000 | 1,000 | 0.000 | 0.000 | 2.236 | 3.368 | 4.711 | 2,368,987,136 | 980,355 | stale failure artifact |
| earlier default compact path, `dense_k=10000`, pre fix | 1,000,000 | 1,000 | 0.007 | 0.007 | 28.115 | 638.660 | 1303.948 | 2,368,987,136 | reused | stale failure artifact |
| exact build distances, exact storage off, earlier build | 1,000,000 | 1,000 | 1.000 | 1.000 | 7.473 | 129.713 | 741.152 | 2,368,987,136 | 3,677,297 | keep experimental |
| residual rerank 16 bytes | 1,000,000 | 1,000 | 0.951 | 0.951 | 29.709 | 871.221 | 1632.056 | 2,368,978,944 | 1,988,858 | reject as default |
| residual rerank 32 bytes | 1,000,000 | 1,000 | 0.951 | 0.951 | 12.627 | 140.159 | 867.119 | 2,368,978,944 | 2,254,932 | tune or revert |
| residual rerank 64 bytes | 1,000,000 | 1,000 | 0.951 | 0.951 | 25.328 | 158.917 | 918.029 | 2,778,578,944 | 1,492,007 | reject as default |
| exact storage on | 1,000,000 | 1,000 | failed | failed | failed | failed | failed | failed | failed | not feasible on this host |

The historical adaptive-auto artifact is
`benchmarks/results/dbpedia-current-default-auto-final.json`. It reuses the
fresh bounded-routing 1M compact TurboHybrid index and compares the explicit
adaptive-off baseline with adaptive `auto`. Adaptive `auto` widened 37 of 1,000
queries, improved self-query nDCG@10 and recall@10 from `0.947000` to
`0.970000`, and kept the index at `2,377,760,768` bytes with
`exact_storage=false`. This was useful diagnostic evidence, but it is not the
release-facing default because the default latency profile should remain
simple, compact, and predictable across workloads.

The fresh bounded-routing index build artifact is
`benchmarks/results/dbpedia-current-routing-full.json`. It records
`routing_entry_count=15`, `routing_entry_bytes=60`, `exact_storage=false`,
`quantization_bits=4`, `graph_m=16`, `graph_ef_construction=128`,
`graph_ef_search=64`, p50 `7.979 ms`, p95 `21.972 ms`, p99 `28.564 ms`, and
build time `764,248.682 ms` before the adaptive-auto diagnostic rerun.

The older no-storage full artifact is
`benchmarks/results/dbpedia-openai3-large-current-no-storage-full.json`. It
reuses an existing 1M-row compact TurboHybrid index with `exact_storage=false`,
`quantization_bits=4`, `graph_m=16`, `graph_ef_construction=128`,
`graph_ef_search=64`, `graph_backbone=false`, and
`dense_build_exact_distances=false`.

That full artifact predates the bounded routing-entry scan fix. The quality
result is still useful as reachability evidence, but its p95/p99 tail latency
is not a current implementation claim. The earlier traversal collected
high-level start nodes by scanning the full graph. Current builds store up to
15 deterministic routing entries in the metapage and traversal only scores
those bounded entries. A current 2,000-row / 20-query smoke recorded
`routing_entry_count=15`, `routing_entry_bytes=60`, `nDCG@10=1.000000`, source
top-10 recovery of `1.000000`, and p95 `2.234 ms`.

The successful exact-build full artifact was produced before the current
metapage provenance bit existed. Later smoke artifacts prove that the current
code can record exact-build provenance, but the full 1M provenance rerun had to
be stopped for host resource safety. A current-code full provenance rerun on
the copied 1M DBPedia database was still inside `CREATE INDEX` after 2h31m and
was terminated. Stack samples showed active work in `tqgraphbuild`,
`PgturbohybridGraphBuildDistance`,
`PgturbohybridGraphBuildExactVectorDistance`, and
`PgturbohybridGraphSelectNeighbors`, with no PostgreSQL lock or I/O wait.
The macOS sampled physical footprint peaked at 16.3 GB and did not keep
growing, so this was a build-time cost problem rather than the previous runaway
memory failure. That timeout also exposed a cancellation-responsiveness gap in
the CPU-heavy graph build path; the current implementation now checks
PostgreSQL interrupts inside neighbor selection, build search, traversal, fill,
and local-expansion loops instead of only once per outer build row.

Current smoke coverage after the storage-compatibility pass:

- `benchmarks/results/dbpedia-pack-current-smoke.json`
- `benchmarks/results/dbpedia-pack-current-smoke.jsonl`
- `benchmarks/results/dbpedia-pack-current-smoke-index.json`
- `benchmarks/results/dbpedia-pack-current-smoke.md`

That smoke run uses a 2,000-row / 10-query deterministic subset and exercises
the default compact path, exact-build distances, explicit level-0 backbone,
adaptive widening, local expansion, entry sidecar, and 32-byte residual rerank.
It is a build/test sanity check, not a quality decision artifact.

Latest bounded-routing smoke after removing the full-graph start-node scan:

- `benchmarks/results/dbpedia-routing-smoke.json`
- `benchmarks/results/dbpedia-routing-smoke.jsonl`
- `benchmarks/results/dbpedia-routing-smoke-index.json`
- `benchmarks/results/dbpedia-routing-smoke.md`

That smoke uses 2,000 rows / 20 queries and records
`routing_entry_count=15`, `routing_entry_bytes=60`, `exact_storage=false`,
`quantization_bits=4`, `nDCG@10=1.000000`, source top-10 recovery of
`1.000000`, p50 `0.985 ms`, p95 `2.234 ms`, and p99 `4.528 ms`. It proves the
bounded routing path is active and avoids the previous O(N) start-node scan,
but it is not a substitute for a fresh full 1M rerun.

Latest broad prompt-pack smoke after the bounded routing fix:

- `benchmarks/results/dbpedia-pack-smoke2.json`
- `benchmarks/results/dbpedia-pack-smoke2.jsonl`
- `benchmarks/results/dbpedia-pack-smoke2-index.json`
- `benchmarks/results/dbpedia-pack-smoke2.md`

That smoke uses 2,000 rows / 20 queries and exercises pgvector dense-only plus
the default compact TurboHybrid path, exact-build distances, explicit
`graph_backbone`, adaptive widening variants, local-expansion variants,
entry-sidecar variants, and residual-rerank variants. All rows completed with
`status=ok`. The default compact TurboHybrid path recorded
`nDCG@10=1.000000`, source top-10 recovery of `1.000000`, p50 `1.054 ms`, p95
`3.019 ms`, p99 `3.627 ms`, `index_bytes=6455296`,
`routing_entry_count=15`, `entry_sidecar_count=0`, and
`residual_rerank_bytes=0`. The forced adaptive widening row was slower on this
smoke, so it remains experimental and off by default.

Latest full default-off versus explicit adaptive-auto recovery artifact:

- `benchmarks/results/dbpedia-current-default-off-recovered-20260529.json`
- `benchmarks/results/dbpedia-current-default-off-recovered-20260529.jsonl`
- `benchmarks/results/dbpedia-current-default-off-recovered-20260529-index.json`
- `benchmarks/results/dbpedia-current-default-off-recovered-20260529.md`

That run uses the same 1,000,000-row compact bounded-routing index for both
methods. `pgturbohybrid_dense_only` is the package default fast path with
adaptive widening off and recovered 947 of 1,000 sources in the top 10. The
explicit adaptive-auto 2.0 variant recovered 970 of 1,000 sources, widened 37
queries, and kept `entry_sidecar_count=0`, `residual_rerank_bytes=0`, and
`exact_storage=false`.

Latest smoke provenance after the default-drift audit:

- default compact path: `graph_backbone=false`, `dense_build_exact_distances=false`
- exact-build variant: `graph_backbone=false`, `dense_build_exact_distances=true`
- explicit backbone variant: `graph_backbone=true`, `dense_build_exact_distances=false`

Latest exact-build provenance smoke after the full-run timeout:

- `benchmarks/results/dbpedia-exactbuild-current-provenance-smoke.json`
- `benchmarks/results/dbpedia-exactbuild-current-provenance-smoke.jsonl`
- `benchmarks/results/dbpedia-exactbuild-current-provenance-smoke-index.json`
- `benchmarks/results/dbpedia-exactbuild-current-provenance-smoke.md`

That smoke uses 500 rows / 5 queries and records
`dense_build_exact_distances=true`, `exact_storage=false`,
`nDCG@10=1.000000`, and source top-10 recovery of `1.000000`. It is proof of
current provenance plumbing, not a full quality result.

## Failure Attribution

The useful failure split is:

- recovered in top 10
- reached by the diagnostic top-100 source probe
- missed by the diagnostic probe

The current probe does not expose every internal candidate that was reached and
then discarded before SQL-visible output, but it is enough to show the dominant
failure mode for the measured variants.

Current findings:

- After the graph edge-construction fixes and bounded routing-entry change,
  the explicit adaptive-off path recovered 947 of 1,000 sources at rank 1.
  Adaptive auto recovered 970 of 1,000 sources by widening only 37 ambiguous
  queries while keeping `exact_storage=off`.
- The previous full-run tail latency was traced to a full-graph scan used to
  collect high-level routing starts. Current builds store a bounded routing
  table on the metapage instead. The full 1M bounded-routing run proves the
  new table is present and used.
- Adaptive widening has useful same-index DBPedia evidence, but not enough
  broader workload evidence to become the dense default. It remains bounded to
  one second pass, does not change index storage, and stays opt-in.
- Forced adaptive widening and forced local expansion did not improve quality
  because the current default was already at full self-query recovery, and the
  forced variants add tail-latency risk.
- The older default compact failure and `dense_k=10000` rows are retained only
  as stale failure artifacts from before the edge-construction fixes.
- Exact build distances recovered 1,000 out of 1,000 sources in an earlier
  full run while keeping `exact_storage=off`, which remains useful as an
  experimental quality mode and topology diagnostic.
- Residual rerank recovered 951 out of 1,000 sources for 16, 32, and 64-byte
  sketches, but the remaining 49 misses were also missed by the top-100 source
  probe. That means the sidecar cannot fix the remaining failures by reranking;
  the source row is not reached by the candidate path.
- Entry sidecar and residual rerank do not have enough current evidence to
  justify release-default status.

## Exact Build Distances

`turbohybrid.dense_build_exact_distances` is the strongest quality signal from
the experiment pack.

Properties:

- explicit GUC
- default off
- `exact_storage=off`
- 4-bit compact runtime index
- exact distances used for graph edge construction
- raw build vectors freed after edge construction when exact storage is off
- construction mode recorded by `turbohybrid_index_stats(...)`

Decision:

- keep as experimental quality mode
- do not make default yet
- do not promote without build-time improvement or a successful full rerun on a
  quieter host
- use current successful 1M result only as internal engineering evidence

## Storage Compatibility Notes

The experiment pack added optional metapage fields for exact-build provenance,
entry-sidecar representatives, and residual rerank sketches. Current readers use
`PgturbohybridGraphReadMeta(...)` for scan/stat paths that need those fields.
That helper copies only the bytes present on the metapage, zero-fills absent
tail fields, and sanitizes sidecar/residual settings before callers size code
tuples or report index stats.

This keeps compact indexes built before those experimental fields from being
misread as having sidecars. It does not make sidecar-enabled indexes a stable
public format; the sidecar options remain experimental and off by default.

## Adaptive Widening And Local Expansion

Adaptive widening is the best default candidate from this prompt pack. Bounded
local expansion remains a diagnostic but not a default candidate yet.

On the fresh bounded-routing full run, adaptive auto widened 37 of 1,000
queries, recovered 23 additional source rows in top 10, and did not increase
index size. Explicit forced widening scored more codes on every query and
recovered fewer sources than auto in the same-index run, so forced widening
stays experimental. Local expansion did not improve recovery on the same index,
so it stays off by default.

Decision:

- keep adaptive widening off by default
- keep local expansion off by default
- keep scan stats and benchmark variants
- revisit adaptive widening only with broader workload evidence and an explicit
  release decision

## Why Adaptive Auto Is Not The Latency Default

The adaptive-auto DBPedia run is useful evidence that a bounded second graph
pass can recover missed dense neighbors on this self-query diagnostic. It is not
enough to make `auto` the release default. The benchmark is unlabeled
self-recovery, not human relevance, and it exercises a dense-only path rather
than the normal hybrid retrieval path.

The public latency default should be easy to reason about: one graph traversal,
compact 4-bit storage, no exact vectors, no sidecars, no residual rerank, no
local expansion, and no adaptive second pass. Adaptive widening stays available
as an explicit GUC and benchmark method until p95/p99 acceptance holds across
broader workloads.

## Level-0 Backbone

The level-0 backbone experiment forces adjacent graph edges after build-time
node reordering. It is now controlled by the `graph_backbone` reloption and is
off by default, so the compact default graph topology does not silently drift.

The DBPedia harness exposes this as `pgturbohybrid_dense_backbone`, records
`graph_backbone` and `index_graph_backbone` in build provenance, and keeps the
default build key separate from the backbone build key.

Decision:

- keep as an explicit DBPedia diagnostic variant
- do not enable by default
- record the setting through `turbohybrid_index_stats(...)` and benchmark
  build provenance

## Entry Sidecar

The entry sidecar stores a small set of representative node IDs, not vectors.
It is deterministic and visible through index stats.

Current evidence does not justify enabling it by default. The current compact
full run already recovers all DBPedia self-query sources, and smoke artifacts
prove the sidecar can be built and reported without making it a release
feature.

Decision:

- keep implemented as an off-by-default experimental reloption
- keep out of public README/release claims
- revisit only if a future workload has reachability misses that the compact
  default cannot recover
- do not enable by default

## Residual Rerank Sidecar

Residual rerank stores a tiny per-vector sketch and applies it only to the
final dense candidate band.

Measured full 1M variants:

- 16-byte sketch: 16,000,000 sketch bytes
- 32-byte sketch: 32,000,000 sketch bytes
- 64-byte sketch: 64,000,000 sketch bytes

All three recovered 951 out of 1,000 sources. The remaining 49 were not reached
by the top-100 source probe, so larger sketches do not address the remaining
candidate-generation misses.

Decision:

- keep implemented as an off-by-default experimental reloption
- keep out of public README/release claims
- use only for future reached-but-misranked failure analysis
- do not enable by default
- do not publish as a default-quality fix

## Sidecar Storage-Format Decision

The sidecar code paths add optional metapage fields and optional per-tuple
payloads, but default compact indexes keep both entry sidecar and residual
rerank disabled. Current metadata readers zero-fill missing metapage tails and
sanitize absent sidecar settings before reporting stats or sizing tuples.

Decision for this prompt-pack branch:

- keep the experimental sidecar implementations because they are explicit,
  measurable, and off by default
- do not promote either sidecar to alpha release defaults
- do not describe either sidecar as stable public storage format
- require a `REINDEX` note before any future release that publicly documents or
  supports these reloptions

## Exact Storage

Full exact storage remains an upper-bound concept only.

The full 1M exact-storage build was stopped for host resource safety during
`CREATE INDEX`. It expanded swap and system-disk pressure enough that the run
was not safe to complete on this machine.

Decision:

- do not make default
- keep smoke coverage for the code path
- use only as an upper-bound reference when a host can support the build

Current exact-storage smoke after the bounded routing fix:

- `benchmarks/results/dbpedia-exact-storage-smoke2.json`
- `benchmarks/results/dbpedia-exact-storage-smoke2.jsonl`
- `benchmarks/results/dbpedia-exact-storage-smoke2-index.json`
- `benchmarks/results/dbpedia-exact-storage-smoke2.md`

That smoke uses 500 rows / 5 queries and records `status=ok`,
`exact_storage=true`, `quantization_bits=4`, `routing_entry_count=15`,
`nDCG@10=1.000000`, source top-10 recovery of `1.000000`, p50 `1.087 ms`, p95
`5.523 ms`, and p99 `6.388 ms`. It proves the upper-bound path still builds and
queries after the metapage/routing changes, not that exact storage is a compact
default candidate.

## Harness And Artifacts

The DBPedia harness now supports:

- aggregate JSON output
- aggregate Markdown output
- per-query JSONL output
- optional index/build JSON output
- command metadata with the Python executable, argv, shell form, selected
  benchmark environment variables, and repo-relative working directory
- build provenance for fresh, shared, existing, and failed indexes
- failed rows with compact error summaries
- best-effort environment metadata so failed-method artifacts still write after
  transient PostgreSQL restart/recovery errors
- per-query source recovery and overlap summaries
- failure probe summaries
- TurboHybrid scan-stat aggregation

Generated benchmark artifacts are intentionally ignored. Keep full JSON,
JSONL, and Markdown benchmark outputs under ignored result directories or
external artifact storage.

Relevant generated artifacts from this investigation:

- `benchmarks/results/dense_quality_decision_report.md`
- `benchmarks/results/dbpedia-openai3-large-dense-exactbuild-current.json`
- `benchmarks/results/dbpedia-openai3-large-residual16-full.json`
- `benchmarks/results/dbpedia-openai3-large-residual32-full.json`
- `benchmarks/results/dbpedia-openai3-large-residual64-full.json`
- `benchmarks/results/dbpedia-openai3-large-exact-storage-full.json`
- `benchmarks/results/dbpedia-openai3-large-exactbuild-full-provenance.json`
- `benchmarks/results/dbpedia-openai3-large-current-no-storage-full.json`
- `benchmarks/results/dbpedia-openai3-large-current-no-storage-full.jsonl`
- `benchmarks/results/dbpedia-openai3-large-current-no-storage-full-index.json`
- `benchmarks/results/dbpedia-openai3-large-current-no-storage-full.md`
- `benchmarks/results/dbpedia-command-metadata-smoke.json`
- `benchmarks/results/dbpedia-command-metadata-smoke.jsonl`
- `benchmarks/results/dbpedia-command-metadata-smoke-index.json`
- `benchmarks/results/dbpedia-command-metadata-smoke.md`
- `benchmarks/results/dbpedia-exactbuild-current-provenance-smoke.json`
- `benchmarks/results/dbpedia-exactbuild-current-provenance-smoke.jsonl`
- `benchmarks/results/dbpedia-exactbuild-current-provenance-smoke-index.json`
- `benchmarks/results/dbpedia-exactbuild-current-provenance-smoke.md`
- `benchmarks/results/dbpedia-routing-smoke.json`
- `benchmarks/results/dbpedia-routing-smoke.jsonl`
- `benchmarks/results/dbpedia-routing-smoke-index.json`
- `benchmarks/results/dbpedia-routing-smoke.md`
- `benchmarks/results/dbpedia-pack-smoke2.json`
- `benchmarks/results/dbpedia-pack-smoke2.jsonl`
- `benchmarks/results/dbpedia-pack-smoke2-index.json`
- `benchmarks/results/dbpedia-pack-smoke2.md`
- `benchmarks/results/dbpedia-exact-storage-smoke2.json`
- `benchmarks/results/dbpedia-exact-storage-smoke2.jsonl`
- `benchmarks/results/dbpedia-exact-storage-smoke2-index.json`
- `benchmarks/results/dbpedia-exact-storage-smoke2.md`
- `benchmarks/results/dbpedia-current-routing-full.json`
- `benchmarks/results/dbpedia-current-routing-full.jsonl`
- `benchmarks/results/dbpedia-current-routing-full-index.json`
- `benchmarks/results/dbpedia-current-routing-full.md`
- `benchmarks/results/dbpedia-current-routing-query-variants.json`
- `benchmarks/results/dbpedia-current-routing-query-variants.jsonl`
- `benchmarks/results/dbpedia-current-routing-query-variants-index.json`
- `benchmarks/results/dbpedia-current-routing-query-variants.md`
- `benchmarks/results/dbpedia-current-default-off-recovered-20260529.json`
- `benchmarks/results/dbpedia-current-default-off-recovered-20260529.jsonl`
- `benchmarks/results/dbpedia-current-default-off-recovered-20260529-index.json`
- `benchmarks/results/dbpedia-current-default-off-recovered-20260529.md`
- `benchmarks/results/dbpedia-current-default-auto-final.json`
- `benchmarks/results/dbpedia-current-default-auto-final.jsonl`
- `benchmarks/results/dbpedia-current-default-auto-final-index.json`
- `benchmarks/results/dbpedia-current-default-auto-final.md`

## Prompt-Pack Requirement Matrix

| Prompt | Current status | Evidence | Remaining proof gap |
|---|---|---|---|
| 01 baseline failure attribution | implemented | per-query JSONL includes query/source IDs, source rank, top-10 IDs, probe rank, latency, and scan stats | the probe is SQL-visible top-k based, not a full internal reached/dropped trace |
| 02 exact build distances | implemented as explicit GUC | `turbohybrid.dense_build_exact_distances`, metapage provenance, regression test, broad prompt-pack smoke, earlier full DBPedia artifact, 2h31m full provenance timeout evidence | repeat full 1M provenance run only after build-time improvement or on a quieter host |
| 03 adaptive widening | implemented as explicit GUCs, default off | scan stats, harness variants, regression GUC checks, broad prompt-pack smoke, full 1M same-index default-vs-auto run | broader workload evidence before enabling by default |
| 04 local expansion | implemented as explicit GUCs, default off | scan stats, harness variants, regression stats checks, broad prompt-pack smoke, full 1M current-code sweep | broader workload evidence before enabling by default |
| 05 entry sidecar | implemented as explicit reloptions, default off | metapage stats, insert preservation, regression test, broad prompt-pack smoke, sidecar storage-format decision | broader workload evidence before documenting as a supported public format |
| 06 residual rerank sidecar | implemented as explicit reloptions, default off | sketch storage stats, scan stats, regression test, broad prompt-pack smoke, full residual artifacts, sidecar storage-format decision | broader workload evidence before documenting as a supported public format |
| 07 benchmark harness | implemented | aggregate JSON, Markdown, JSONL, index report, failed-row recording, command metadata, environment info, build provenance | schema validation is structural only; full statistical confidence depends on full runs |
| 08 ablation decision report | implemented | this internal report and ignored generated decision report | full exact-storage upper bound did not complete on this host |
| 09 release guardrails | implemented for touched docs | README keeps DBPedia out of public claims; path/claim grep is clean | repeat guardrail check before commit/release |

This matrix is a current implementation audit, not a release approval. The goal
is to keep the experimental knobs available for measurement while avoiding
silent default behavior changes.

## Remaining Gaps

- Improve exact-build construction time or repeat the full 1M
  exact-build-distance run with current provenance bits on a host where
  swap/system-disk pressure and runtime can be controlled.
- Re-test cancellation during a long exact-build-distance `CREATE INDEX`; the
  code now checks interrupts inside the hot loops, but the original 2h31m run
  had already been terminated before that patch.
- Compare the compact default on another query workload where the answer is not
  simply the source row reused as the query embedding.
- Run a second workload, such as FIQA/OpenAI, before generalizing any dense
  quality conclusion beyond DBPedia self-recovery.
- Add deeper candidate-state instrumentation if it becomes necessary to
  distinguish every reached-but-dropped case from probe-missed cases.

## Release Guardrails

- Keep DBPedia dense-quality results out of public README claims for now.
- If DBPedia results are mentioned, describe them as local development evidence
  for a self-query systems benchmark.
- Always include dataset, dimensions, query count, profile, index settings,
  budgets, warmup/measured passes, quality metric, and hardware caveat.
- Keep Turbovec separate from PostgreSQL-native baselines because it is an
  in-process vector-library reference.
- Do not present batch-mode throughput as an interactive RAG latency metric.
