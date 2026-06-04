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
