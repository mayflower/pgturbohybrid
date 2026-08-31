# On-Disk Storage Format

The contract for `pgturbohybrid`'s on-disk index format: the identity constants,
page kinds, metapage layout, when readers reject corrupt/incompatible data, and
which changes require `REINDEX`. It gives maintainers a checklist for detecting
accidental on-disk compatibility breaks. While the extension is **alpha**, the
on-disk format may change between tags — plan to `REINDEX` on upgrade (see
[operations.md](operations.md) and [../RELEASE.md](../RELEASE.md)).

## Identity and version constants

Defined in `src/pgturbohybrid.h` and the per-subsystem headers:

| Constant | Value | Meaning |
| --- | --- | --- |
| `PGTURBOHYBRID_MAGIC_NUMBER` | `0x54525944` | graph metapage magic |
| `PGTURBOHYBRID_PAGE_ID` | `0x5459` | special-space page id |
| `PGTURBOHYBRID_GRAPH_VERSION` | `1` | graph metapage format version |
| `PGTURBOHYBRID_BM25_VERSION` | `1` | BM25 metadata format version |
| `PGTURBOHYBRID_SPARSE_VERSION` | `2` | sparse postings/meta format version |
| `PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION` | `4` | multivector doc-map version |
| `PGTURBOHYBRID_QUERY_VERSION` | `4` | `turbohybrid_query` payload version (not on-disk) |
| `PGTURBOHYBRID_SPARSE_VECTOR_VERSION` | `1` | `turbohybrid_sparse_vector` varlena version |

## Metapage (block 0)

Block 0 is the graph metapage. It stores the access-method identity (magic,
page id, format version), `storageKind`, index dimensions, dense graph options,
quantization options (`tqBits`), bounded routing entry IDs, optional
entry-sidecar IDs, segment metadata, entry/start block pointers, and pointers to
the BM25 metadata, multivector doc-map, and sparse-primary node-map chains.
Older metapage tails are zero-filled by readers, so a default compact index is
never misread as having sidecars.

## Page kinds

- graph tuples and graph metadata
- quantized code pages; quantized adjacency pages
- optional exact-vector pages (final rescore); quantization correction pages
- optional residual-rerank bytes embedded in quantized code tuples
- BM25: metadata, document statistics, lexicon, postings, block-max, delta,
  impact, delta-term pages
- sparse: sparse metapage, per-term lexicon, postings chunks, block-max (WAND)
  directory, delta chain, and node-map pages (sparse-primary node↔TID identity)

All index page changes are WAL-logged with PostgreSQL generic WAL; no custom
resource manager and no `shared_preload_libraries` are required.

## Corruption / incompatibility rejection

Readers reject malformed or incompatible metadata with
`ERRCODE_DATA_CORRUPTED` (a REINDEX hint where applicable) rather than
misreading it:

- **Graph metapage** — bad magic or `storageKind` dispatches as "not a native
  graph" (so sparse-primary/non-native indexes are handled), but an unknown
  `version` after that gate is rejected. Sub-field counts/capacities (sidecar,
  routing, segments) are bounds-clamped.
- **BM25 metadata** — metapage pointer range, page-kind, tuple presence, and
  chain-bound checks.
- **Sparse node-map** — every chain pointer must be in range, never the
  metapage, never self-referential; a node-map tuple's TID run may not exceed
  its item line pointer.
- **Sparse postings** — a chunk's posting count may not exceed the physical
  `PGTURBOHYBRID_SPARSE_MAX_BLOCK_SIZE` ceiling; the sparse metapage version
  must match `PGTURBOHYBRID_SPARSE_VERSION`.

`test/t/004_metadata_corruption.pl` proves corrupt metadata yields a clean error
or controlled behavior, never a crash.

## When `REINDEX` is required

A format-affecting change must bump the relevant version constant above and the
release notes must require `REINDEX`. `REINDEX` is required when:

- the page layout changes,
- quantized code encoding changes,
- graph tuple layout changes,
- BM25 postings or lexicon layout changes,
- the sparse postings / node-map layout changes,
- compatibility with existing index pages cannot be proven.

`REINDEX` is **not** required for SQL-only diagnostic changes that do not affect
stored index pages.

## What upgrade scripts may and may not do

Extension SQL upgrade scripts (`sql/pgturbohybrid--<from>--<to>.sql`) may add,
change, or drop SQL objects (functions, operators, opclasses, comments). They
**may not** rewrite existing index pages or change the on-disk format in place:
an incompatible format change requires a version-constant bump and a documented
`REINDEX`, not an `ALTER EXTENSION UPDATE`.
