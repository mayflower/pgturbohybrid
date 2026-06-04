# Multivector Doc Map Sidecar Design

This document designs the persistent multivector document map sidecar for the
native TurboHybrid graph storage. It is intentionally documentation-only until
the implementation prompt changes the on-disk format.

## Problem

Multivector indexes expand each SQL heap tuple into one graph node per document
token. The existing build path already has transient maps:

- `nodeId -> docId, tokenOrdinal` as `TqMultiVectorNodeMapEntry`
- `docId -> heaptid, firstNodeId, tokenCount` as `TqMultiVectorDocMapEntry`

Those maps are not persisted. At scan time,
`PgturbohybridGraphCollectMultiVectorDenseCandidates` derives scan-local
`docId` values by hashing heap TIDs from subvector hits. That remains correct,
but it repeats heap-TID lookups and cannot use the contiguous document/token
structure known during build.

## Storage Identity

Add one graph page kind:

```c
#define PGTURBOHYBRID_GRAPH_PAGE_KIND_MULTIVECTOR_DOCMAP 15
```

The sidecar is present only for native quant graph indexes whose dense key is
`turbohybrid_multivector`. Single-vector indexes must leave all sidecar
metadata unset.

## Version And Compatibility

The sidecar needs its own explicit identity separate from
`PGTURBOHYBRID_GRAPH_VERSION`:

```c
#define PGTURBOHYBRID_MULTIVECTOR_DOCMAP_MAGIC 0x54514d56
#define PGTURBOHYBRID_MULTIVECTOR_DOCMAP_VERSION 1
```

The graph metapage should gain discoverability fields:

- `tqMultivectorDocMapStartBlkno`
- `tqMultivectorDocMapPageCount`
- `tqMultivectorDocMapVersion`
- `tqMultivectorDocCount`
- `tqMultivectorDocMapBytes`

Compatibility behavior:

- If start block is invalid or page count is zero, scans must use the existing
  heap-TID hash fallback.
- If a sidecar page exists but the magic/version is unsupported, scan startup
  must raise an error with clear REINDEX guidance.
- If metadata claims a sidecar but page checks fail, treat the index as corrupt
  and error; do not silently mix partial sidecar data with hash fallback.
- Old indexes without the metapage fields effectively read zero/invalid values
  through the existing bounded metapage copy path and remain usable through
  fallback.

## Layout

Use slotted PostgreSQL pages with
`PGTURBOHYBRID_GRAPH_PAGE_KIND_MULTIVECTOR_DOCMAP`. Each page begins with a
fixed header tuple followed by fixed-size arrays. Keeping fixed-size records
allows direct size checks and simple cache loading.

Header tuple:

```c
typedef struct TqMultiVectorDocMapPageHeaderData
{
    uint32 magic;
    uint16 version;
    uint16 flags;
    uint32 firstNodeId;
    uint32 nodeCount;
    uint32 firstDocId;
    uint32 docCount;
} TqMultiVectorDocMapPageHeaderData;
```

Node map record:

```c
typedef struct TqMultiVectorDocMapNodeRecord
{
    TqDocId docId;
    TqSubvectorOrdinal tokenOrdinal;
    uint16 reserved;
} TqMultiVectorDocMapNodeRecord;
```

Document map record:

```c
typedef struct TqMultiVectorDocMapDocRecord
{
    ItemPointerData heaptid;
    uint32 firstNodeId;
    uint16 tokenCount;
    uint16 flags;
} TqMultiVectorDocMapDocRecord;
```

The persisted maps are:

- `nodeId -> docId`
- `nodeId -> tokenOrdinal`
- `docId -> heaptid`
- `docId -> firstNodeId`
- `docId -> tokenCount`

The implementation may store node and document records on separate sidecar
pages if that keeps page packing simpler, but both page variants must use the
same page kind plus a header flag that identifies the record family.

## Build-Time Writing

During index build, `PgturbohybridGraphAppendBuildMultiVector` already assigns
contiguous `docId` values and records `firstNodeId` plus `tokenCount`. After
code, adjacency, exact, correction, and optional BM25 pages are written, write
the doc map sidecar from `state->multivectorNodeMap` and
`state->multivectorDocMap`.

Build requirements:

- Write only when `state->multivectorBuild` is true.
- Validate `nodeCount`, `docCount`, `firstNodeId + tokenCount`, and record
  sizes with overflow-checked PostgreSQL size helpers.
- Set metapage sidecar fields only after all sidecar pages are written.
- Include sidecar bytes and doc count in build/index stats.
- Preserve existing graph node order; `nodeId` remains the subvector identity.

## Insert And Update Behavior

Incremental insert must append sidecar entries for each new multivector heap
tuple:

- Allocate the next `docId` from metapage `tqMultivectorDocCount`.
- Append one doc record with the inserted heap TID, first inserted node ID, and
  token count.
- Append one node record per inserted subvector with that `docId` and
  `tokenOrdinal`.
- Update metapage doc count, page count, byte count, and generation under the
  same index locks used for graph append/metapage update.
- If append cannot be made safe for a partially filled sidecar page, append a
  new page rather than rewriting unrelated records.

PostgreSQL UPDATE creates a new heap tuple version and is handled as insert of
new graph nodes plus normal visibility checks. Old sidecar records remain as
historical mapping entries until vacuum/dead-node maintenance marks or ignores
their graph nodes.

## Vacuum And Dead Tuples

The sidecar maps identity; it must not bypass MVCC visibility. Scans still need
to return heap TIDs through the access method and let the normal index/heap
visibility path reject dead rows.

Vacuum behavior:

- Existing graph dead-node handling remains authoritative for whether a node is
  scannable.
- Sidecar `docId -> heaptid` records can stay append-only while dead nodes are
  marked in code/adj storage.
- If all nodes for a doc are dead, scans should never accumulate that doc
  because traversal should not offer dead nodes.
- Future compaction can rewrite sidecar records, but that requires a separate
  versioned design.

Tests must cover delete, vacuum, update, and insert-after-vacuum to prove the
sidecar does not return dead heap rows or duplicate old versions.

## Scan Fallback And Source Selection

Add a GUC for implementation/testing:

```sql
SET turbohybrid.multivector_docmap = 'auto'; -- off | auto | require
```

Source selection:

- `off`: use the existing heap-TID hash path.
- `auto`: use sidecar when metadata is valid and pages load; otherwise fallback
  for no-sidecar old indexes.
- `require`: require a valid sidecar and error with REINDEX guidance if absent.

When sidecar is used, a graph hit resolves `nodeId` directly to `docId` and
then to the heap TID/doc record. The fallback path remains the current
`heaptid -> scan-local docId` hash.

## Native Cache Integration

The implementation extends `PgturbohybridGraphNativeCache` and
`PgturbohybridGraphScanStorage` with optional arrays:

- `TqMultiVectorNodeMapEntry *multivectorNodeMap`
- `TqMultiVectorDocMapEntry *multivectorDocMap`
- `uint32 multivectorDocCount`
- `uint32 multivectorDocMapBytes`

Cache identity must include relfilenumber, metapage graph generation, sidecar
start block, page count, doc count, and sidecar version. A cache built before
an insert/metapage bump must not serve a later scan.

The native-cache memory estimate and scan stats should include the sidecar:

- `native.multivector_docmap_bytes` from
  `turbohybrid_estimate_memory(index)`
- `multivector_docmap_source = sidecar | heap_tid_hash | none`
- `multivector_docmap_bytes`

## Memory Estimate

Estimate resident sidecar bytes as:

```text
node_map_bytes = tqNodeCount * sizeof(TqMultiVectorNodeMapEntry)
doc_map_bytes = tqMultivectorDocCount * sizeof(TqMultiVectorDocMapEntry)
total = node_map_bytes + doc_map_bytes
```

Use `mul_size` and `add_size` for implementation. Report both on-disk bytes
from the metapage and resident bytes from the native-cache/storage estimate.

## Implementation Tests

Required tests for the implementation prompt:

- Build a multivector index and verify scan stats report
  `multivector_docmap_source = sidecar`.
- Insert into a multivector index and verify the new document is returned once.
- Force `turbohybrid.multivector_docmap = off` and verify fallback still works.
- Force `require` on an index without sidecar and verify clear REINDEX guidance.
- Delete/update/vacuum tests prove dead or old tuple versions are not returned.
- Memory/index stats include sidecar bytes.
- Dump/restore still works for table data; index sidecar is rebuilt by
  `CREATE INDEX` after restore.

## Non-Goals

- Do not change the SQL-visible `turbohybrid_multivector` varlena layout.
- Do not rank or deduplicate SQL results by `nodeId`.
- Do not use the sidecar as a visibility map.
- Do not require old indexes to be rebuilt unless the user explicitly selects
  a sidecar-required mode or the index claims a malformed sidecar.
