/*
 * pgturbohybrid_sparse_primary.c
 *
 * Sparse-primary node space: own node_id<->heap-TID identity for
 * indexes with no dense/multivector graph (sparse-only, sparse+BM25).  Builds a
 * minimal native metapage plus a dedicated node-map chain (one heap TID per
 * node_id) over the heap, then runs the sparse (and BM25) collectors, which key
 * on node_id via the generalized PgturbohybridReadNodeMap/ReadNodeStates.
 *
 * MVCC: node liveness is delegated to heap-tuple visibility -- the scan returns
 * candidate TIDs and the executor filters dead/invisible rows -- so the node map
 * marks every mapped node live (matching the dense-present sparse behavior, where
 * liveness is an optimization, not the correctness guarantee).
 */
#include "postgres.h"

#include <string.h>

#include "access/generic_xlog.h"
#include "access/genam.h"
#include "access/tableam.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/memutils.h"
#include "utils/rel.h"

#include "pgturbohybrid.h"
#include "pgturbohybrid_quant.h"
#include "pgturbohybrid_am.h"
#include "pgturbohybrid_bm25.h"
#include "pgturbohybrid_query.h"
#include "pgturbohybrid_sparse.h"

#define PGTURBOHYBRID_SPARSE_NODEMAP_PER_TUPLE 400

typedef struct PgturbohybridSparsePrimaryCollector
{
	int			primaryKey;
	ItemPointerData *tids;
	uint64		count;
	uint64		capacity;
} PgturbohybridSparsePrimaryCollector;

bool
PgturbohybridSparseIsPrimary(Relation index)
{
	PgturbohybridIndexKeyMap map;

	if (index == NULL || index->rd_index == NULL)
		return false;
	PgturbohybridBuildIndexKeyMap(index, NULL, &map);
	/*
	 * Sparse-primary = a sparse key with no dense/multivector graph (sparse-only
	 * or sparse+BM25).  BM25-only (tsvector without sparse or dense) remains
	 * unsupported; that is a separate milestone.
	 */
	return map.graphKey < 0 && map.hasSparse;
}

/* Create the sparse-primary metapage at block 0 (no dense graph or codes). */
static void
PgturbohybridSparsePrimaryCreateMeta(Relation index, ForkNumber forkNum)
{
	Buffer		buf;
	Page		page;
	PgturbohybridGraphMetaPage metap;

	buf = PgturbohybridGraphNewBuffer(index, forkNum);
	page = BufferGetPage(buf);
	PgturbohybridGraphInitPageKind(buf, page, PGTURBOHYBRID_GRAPH_PAGE_KIND_META);
	metap = PgturbohybridGraphPageGetMeta(page);

	memset(metap, 0, sizeof(PgturbohybridGraphMetaPageData));
	metap->magicNumber = PGTURBOHYBRID_GRAPH_MAGIC_NUMBER;
	metap->version = PGTURBOHYBRID_GRAPH_VERSION;
	metap->storageKind = PGTURBOHYBRID_GRAPH_STORAGE_QUANT_GRAPH_NATIVE;
	/*
	 * No dense graph, but keep m/efConstruction at their sane reloption defaults
	 * so any code that reads them (e.g. the graph-memory cost estimator) never
	 * divides by a zero degree.
	 */
	metap->m = PgturbohybridGraphGetM(index);
	metap->efConstruction = PgturbohybridGraphGetEfConstruction(index);
	metap->tqBits = PgturbohybridGraphGetTqBits(index);
	metap->dimensions = 0;		/* no dense vectors */
	metap->graphMaxLevel = 0;
	metap->entryBlkno = InvalidBlockNumber;
	metap->entryOffno = InvalidOffsetNumber;
	metap->entryLevel = -1;
	metap->insertPage = InvalidBlockNumber;
	metap->tqEntryNodeId = UINT_MAX;
	metap->tqNodeCount = 0;
	metap->tqCodeStartBlkno = InvalidBlockNumber;	/* identity comes from node map */
	metap->tqAdjStartBlkno = InvalidBlockNumber;
	metap->tqExactStartBlkno = InvalidBlockNumber;
	metap->tqCorrectionStartBlkno = InvalidBlockNumber;
	metap->tqBm25MetaStartBlkno = InvalidBlockNumber;
	metap->tqMultivectorDocMapStartBlkno = InvalidBlockNumber;
	metap->tqMultivectorGraphMode = PGTURBOHYBRID_DEFAULT_MULTIVECTOR_GRAPH_MODE;
	metap->tqSparseMetaStartBlkno = InvalidBlockNumber;
	metap->tqNodeMapStartBlkno = InvalidBlockNumber;
	((PageHeader) page)->pd_lower =
		((char *) metap + sizeof(PgturbohybridGraphMetaPageData)) - (char *) page;

	PgturbohybridGraphFinishPage(buf);
}

/* Update the metapage node-count + node-map anchor (in place, WAL'd). */
static void
PgturbohybridSparsePrimarySetNodeMap(Relation index, uint32 nodeCount,
									 BlockNumber nodeMapStart)
{
	Buffer		buf;
	Page		page;
	PgturbohybridGraphMetaPage metap;
	GenericXLogState *xlogState = NULL;

	buf = ReadBuffer(index, PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	if (RelationNeedsWAL(index))
	{
		xlogState = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(xlogState, buf, 0);
	}
	else
		page = BufferGetPage(buf);

	metap = PgturbohybridGraphPageGetMeta(page);
	metap->tqNodeCount = nodeCount;
	metap->tqNodeMapStartBlkno = nodeMapStart;
	PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_META_UPDATE);

	if (xlogState != NULL)
		GenericXLogFinish(xlogState);
	else
		MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);
	PgturbohybridGraphLogGraphWalRecord(index, MAIN_FORKNUM,
										PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO,
										PGTURBOHYBRID_GRAPH_GRAPH_OP_META_UPDATE);
}

static void
PgturbohybridSparsePrimaryCallback(Relation index, ItemPointer tid, Datum *values,
								   bool *isnull, bool tupleIsAlive, void *opaque)
{
	PgturbohybridSparsePrimaryCollector *c =
		(PgturbohybridSparsePrimaryCollector *) opaque;

	(void) tupleIsAlive;
	if (isnull[c->primaryKey])
		return;					/* rows with a NULL primary key are excluded */
	if (c->count == c->capacity)
	{
		c->capacity = c->capacity == 0 ? 1024 : c->capacity * 2;
		c->tids = c->tids == NULL ?
			(ItemPointerData *) palloc(sizeof(ItemPointerData) * c->capacity) :
			(ItemPointerData *) repalloc(c->tids, sizeof(ItemPointerData) * c->capacity);
	}
	c->tids[c->count++] = *tid;
}

/* Write the node map (node_id = position) as a chain of packed TID tuples. */
static BlockNumber
PgturbohybridSparsePrimaryWriteNodeMap(Relation index, ItemPointerData *tids,
									   uint64 count)
{
	BlockNumber start = InvalidBlockNumber;
	BlockNumber insertBlkno;
	PgturbohybridSparseNodeMapTupleData *tuple;
	uint64		i = 0;

	if (count == 0)
		return InvalidBlockNumber;

	tuple = (PgturbohybridSparseNodeMapTupleData *)
		palloc0(PgturbohybridSparseNodeMapTupleSize(PGTURBOHYBRID_SPARSE_NODEMAP_PER_TUPLE));
	while (i < count)
	{
		uint32		n = (uint32) Min((uint64) PGTURBOHYBRID_SPARSE_NODEMAP_PER_TUPLE,
									 count - i);

		tuple->type = PGTURBOHYBRID_SPARSE_NODEMAP_TUPLE_TYPE;
		tuple->reserved1 = 0;
		tuple->count = (uint16) n;
		tuple->firstNodeId = (uint32) i;
		memcpy(tuple->tids, &tids[i], sizeof(ItemPointerData) * n);
		(void) PgturbohybridGraphAppendTuple(index, MAIN_FORKNUM, &start,
											 PGTURBOHYBRID_GRAPH_PAGE_KIND_NODEMAP,
											 (Item) tuple,
											 PgturbohybridSparseNodeMapTupleSize(n),
											 PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT,
											 &insertBlkno);
		i += n;
	}
	pfree(tuple);
	return start;
}

IndexBuildResult *
PgturbohybridSparsePrimaryBuild(Relation heap, Relation index, IndexInfo *indexInfo)
{
	IndexBuildResult *result;
	PgturbohybridIndexKeyMap map;
	PgturbohybridSparsePrimaryCollector collector;
	BlockNumber nodeMapStart;
	MemoryContext ctx;
	MemoryContext oldCtx;

	PgturbohybridBuildIndexKeyMap(index, indexInfo, &map);

	ctx = AllocSetContextCreate(CurrentMemoryContext,
								"pgturbohybrid sparse-primary build",
								ALLOCSET_DEFAULT_SIZES);
	oldCtx = MemoryContextSwitchTo(ctx);

	PgturbohybridSparsePrimaryCreateMeta(index, MAIN_FORKNUM);

	memset(&collector, 0, sizeof(collector));
	collector.primaryKey = map.primaryKey;
	(void) table_index_build_scan(heap, index, indexInfo, true, true,
								  PgturbohybridSparsePrimaryCallback, &collector, NULL);

	/*
	 * Node ids are uint32, so a table with more rows than uint32 can address
	 * would silently wrap the node-id space (the narrowing casts in
	 * PgturbohybridSparsePrimaryWriteNodeMap and SetNodeMap below).  Reject it
	 * explicitly rather than corrupting the index.
	 */
	if (collector.count > PG_UINT32_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pgturbohybrid sparse index supports at most %u rows, but the table has " UINT64_FORMAT " rows",
						PG_UINT32_MAX, collector.count)));

	nodeMapStart = PgturbohybridSparsePrimaryWriteNodeMap(index, collector.tids,
														  collector.count);
	PgturbohybridSparsePrimarySetNodeMap(index, (uint32) collector.count, nodeMapStart);

	MemoryContextSwitchTo(oldCtx);
	MemoryContextDelete(ctx);

	/*
	 * The BM25 and sparse inverted indexes are built by pgturbohybridambuild's
	 * post-build collect passes (self-guarded by key type); they now read node
	 * identity from the node-map chain this build just wrote.
	 */
	result = (IndexBuildResult *) palloc0(sizeof(IndexBuildResult));
	result->heap_tuples = (double) collector.count;
	result->index_tuples = (double) collector.count;
	return result;
}

void
PgturbohybridSparsePrimaryBuildEmpty(Relation index)
{
	PgturbohybridSparsePrimaryCreateMeta(index, INIT_FORKNUM);
}

uint32
PgturbohybridSparsePrimaryInsert(Relation index, ItemPointer heaptid)
{
	PgturbohybridGraphMetaPageData meta;
	PgturbohybridSparseNodeMapTupleData *tuple;
	BlockNumber start;
	BlockNumber insertBlkno;
	uint32		nodeId;

	if (!PgturbohybridGraphReadMeta(index, &meta))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("pgturbohybrid sparse-primary metapage is missing")));

	nodeId = meta.tqNodeCount;
	start = meta.tqNodeMapStartBlkno;

	tuple = (PgturbohybridSparseNodeMapTupleData *)
		palloc0(PgturbohybridSparseNodeMapTupleSize(1));
	tuple->type = PGTURBOHYBRID_SPARSE_NODEMAP_TUPLE_TYPE;
	tuple->count = 1;
	tuple->firstNodeId = nodeId;
	tuple->tids[0] = *heaptid;
	(void) PgturbohybridGraphAppendTuple(index, MAIN_FORKNUM, &start,
										 PGTURBOHYBRID_GRAPH_PAGE_KIND_NODEMAP,
										 (Item) tuple,
										 PgturbohybridSparseNodeMapTupleSize(1),
										 PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT,
										 &insertBlkno);
	pfree(tuple);

	PgturbohybridSparsePrimarySetNodeMap(index, nodeId + 1, start);
	return nodeId;
}
