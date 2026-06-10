# Project Guidelines

## Execution Culture

We ship. Prefer **smallest change that moves production metrics** over perfect abstractions.

- **Clone, fork, patch extensions** — `pgturbohybrid`, `pg_ripple`, `pg-trickle`, `pg_deltax` are in our control. Point fixes in C/SQL are normal; upstream PRs are optional follow-up.
- **SOTA by default** — benchmark against best open baselines (LoCoMo, hybrid retrieval, KG explainability). If we are not leading, document why and close the gap.
- **Always commit** — atomic commits with a clear message; never leave working tree dirty across sessions. Push when the change is verified.
- **One source of truth** — long plan: ValorBrain `docs/dev/enterprise-knowledge-stack-plan.md`; active queue: `docs/sprints/EXECUTION-BACKLOG.md`; ops: `docs/OPERATIONS-CHECKLIST.md` (ValorBrain repo).
- **ICE before starting** — Impact × Confidence ÷ Effort. P0 = production latency/reliability; P1 = KG/ripple; P2 = platform (deltax/trickle). Cut anything that does not move R@5, p95, or data safety.
- **Sprint batches** — work in numbered batches (VB-25xx, VB-26xx). Finish the batch or leave explicit `🔄` notes in the backlog; do not scatter new markdown.

### Stack (ValorBrain production)

```
Agents → ValorBrain engine (Bun, :7438)
           ├── Retrieval: pgturbohybrid + pgvector + BM25/RRF
           ├── KG: entity_triples → pg_ripple (shadow → hybrid read → SoT)
           ├── Embeddings: vLLM jina-embeddings-v5 (:7997)
           └── LLM: OpenRouter (hooks/HyDE); legacy Ollama optional
PostgreSQL :5433 — extensions: vector, pgturbohybrid, pg_ripple, pg_trgm
```

This repo (`pgturbohybrid`) is the **hybrid retrieval access method**. ValorBrain consumes it via SQL (`turbohybrid` engine). Changes here require `th-installcheck` before ValorBrain adopts.

---

## PostgreSQL Extension Style

- Keep C code in the existing PostgreSQL extension style: explicit memory
  contexts, PostgreSQL error reporting, overflow-checked size calculations, and
  no hidden heap or index access outside the access method rules.
- Do not introduce silent ABI or on-disk format changes. Any persisted format
  change needs an explicit version, magic, or compatibility check and clear
  REINDEX guidance.
- New SQL-visible behavior should have regression coverage before it becomes a
  dependency for later features.
- Do not commit generated benchmark output, regression output, local logs, or
  host-specific artifacts.

## Nix Development Environment

- Use this repository's Nix flake for builds, regression tests, TAP tests, and
  benchmark smoke checks. Do not default to the host Homebrew PostgreSQL or
  `pg_config` toolchain.
- Enter the dev environment with:
  `nix --extra-experimental-features 'nix-command flakes' develop`
- If running commands non-interactively, wrap them with:
  `nix --extra-experimental-features 'nix-command flakes' develop --command ...`
- Prefer the flake helper commands inside the dev shell:
  - `th-pg-init` to initialize/start the local PostgreSQL cluster.
  - `th-installcheck` for SQL regression tests.
  - `th-prove-installcheck` for TAP tests.
  - `th-smoke` for the minimal extension smoke test.
- Verify the environment before validation with `echo "$IN_NIX_SHELL"` and
  `pg_config --version`; the flake currently provides PostgreSQL 17. If TAP
  modules appear unavailable, first re-check that the command is running inside
  `nix develop` before treating it as a project or system issue.
- For live `pg_colbert_llama` development and validation, use the small 15m
  ColBERT pair: `VAGOsolutions/SauerkrautLM-Multi-ColBERT-15m` for PyLate
  parity and `johannhartmann/SauerkrautLM-Multi-ColBERT-15m-GGUF` for the
  PostgreSQL/llama.cpp GGUF path. Keep GGUFs under
  `.nix-dev/models/colbert-15m/`, which is ignored and must not be committed.

## Multivector / Late Interaction

- A multivector graph node is a subvector/token node, not a SQL result.
- Use these meanings consistently:
  - `nodeId`: subvector/token node.
  - `docId`: document-level identifier used for result aggregation.
  - `heaptid`: PostgreSQL heap tuple for visibility and final result output.
  - `tokenOrdinal`: token/subvector position inside the document.
- Never rank or deduplicate multivector SQL results by `nodeId`. Result ranking
  is document/heap-tuple based.
- MaxSim is similarity based: larger is better.
- SQL `ORDER BY` distance must remain smaller-is-better. For MaxSim use
  `distance = -maxsim`.
- TurboQuant may be used for approximate subvector candidate generation, but
  final MaxSim rerank should use exact float32 values unless a prompt explicitly
  changes that contract.

## Pre-Flight (every change)

See ValorBrain `docs/OPERATIONS-CHECKLIST.md` for the unified list. Minimum here:

1. `nix develop --command th-installcheck` after C/SQL changes.
2. No tracked generated artifacts.
3. Commit with regression evidence in the message (pass/fail counts).
4. If the change affects on-disk index format, bump extension version and document REINDEX.

## Release

Extension releases follow `RELEASE_CHECKLIST.md` in this repo. Coordinate with ValorBrain ops runbook when deploying to `:5433`.