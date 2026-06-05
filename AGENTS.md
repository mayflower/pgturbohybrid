# Project Guidelines

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
