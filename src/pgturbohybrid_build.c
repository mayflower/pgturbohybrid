/*
 * The HNSW build happens in two phases:
 *
 * 1. In-memory phase
 *
 * In this first phase, the graph is held completely in memory. When the graph
 * is fully built, or we run out of memory reserved for the build (determined
 * by maintenance_work_mem), we materialize the graph to disk (see
 * FlushPages()), and switch to the on-disk phase.
 *
 * In a parallel build, a large contiguous chunk of shared memory is allocated
 * to hold the graph. Each worker process has its own PgturbohybridGraphBuildState struct in
 * private memory, which contains information that doesn't change throughout
 * the build, and pointers to the shared structs in shared memory. The shared
 * memory area is mapped to a different address in each worker process, and
 * 'PgturbohybridGraphBuildState.hnswarea' points to the beginning of the shared area in the
 * worker process's address space. All pointers used in the graph are
 * "relative pointers", stored as an offset from 'hnswarea'.
 *
 * Each element is protected by an LWLock. It must be held when reading or
 * modifying the element's neighbors or 'heaptids'.
 *
 * In a non-parallel build, the graph is held in backend-private memory. All
 * the elements are allocated in a dedicated memory context, 'graphCtx', and
 * the pointers used in the graph are regular pointers.
 *
 * 2. On-disk phase
 *
 * In the on-disk phase, the index is built by inserting each vector to the
 * index one by one, just like on INSERT. The only difference is that we don't
 * WAL-log the individual inserts. If the graph fit completely in memory and
 * was fully built in the in-memory phase, the on-disk phase is skipped.
 *
 * After we have finished building the graph, we perform one more scan through
 * the index and write all the pages to the WAL.
 */
#include "postgres.h"

#include <limits.h>

#include "access/genam.h"
#include "access/parallel.h"
#include "access/relscan.h"
#include "access/table.h"
#include "access/tableam.h"
#include "access/tupdesc.h"
#include "access/xact.h"
#include "access/xloginsert.h"
#include "catalog/index.h"
#include "catalog/pg_type_d.h"
#include "commands/progress.h"
#include "pgturbohybrid.h"
#include "miscadmin.h"
#include "nodes/execnodes.h"
#include "optimizer/optimizer.h"
#include "storage/bufmgr.h"
#include "storage/condition_variable.h"
#include "tcop/tcopprot.h"
#include "utils/datum.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/lsyscache.h"
#include "utils/snapmgr.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

#if PG_VERSION_NUM >= 140000
#include "utils/backend_progress.h"
#else
#include "pgstat.h"
#endif

#if PG_VERSION_NUM >= 140000
#include "utils/backend_status.h"
#include "utils/wait_event.h"
#endif

#define PARALLEL_KEY_PGTURBOHYBRID_SHARED		UINT64CONST(0xA000000000000001)
#define PARALLEL_KEY_PGTURBOHYBRID_AREA			UINT64CONST(0xA000000000000002)
#define PARALLEL_KEY_QUERY_TEXT			UINT64CONST(0xA000000000000003)

#define PGTURBOHYBRID_GRAPH_MAX_GRAPH_MEMORY (SIZE_MAX / 2)

/*
 * Create the metapage
 */
static void
CreateMetaPage(PgturbohybridGraphBuildState * buildstate)
{
	Relation	index = buildstate->index;
	ForkNumber	forkNum = buildstate->forkNum;
	Buffer		buf;
	Page		page;
	PgturbohybridGraphMetaPage metap;

	buf = PgturbohybridGraphNewBuffer(index, forkNum);
	page = BufferGetPage(buf);
	PgturbohybridGraphInitPageKind(buf, page, PGTURBOHYBRID_GRAPH_PAGE_KIND_META);

	/* Set metapage data */
	metap = PgturbohybridGraphPageGetMeta(page);
	metap->magicNumber = PGTURBOHYBRID_GRAPH_MAGIC_NUMBER;
	metap->version = PGTURBOHYBRID_GRAPH_VERSION;
	metap->dimensions = buildstate->dimensions;
	metap->m = buildstate->m;
	metap->efConstruction = buildstate->efConstruction;
	if (PgturbohybridGraphUseTqGraph(index))
		metap->storageKind = PGTURBOHYBRID_GRAPH_STORAGE_QUANT_GRAPH;
	else if (PgturbohybridGraphUseTqFlat(index))
		metap->storageKind = PGTURBOHYBRID_GRAPH_STORAGE_QUANT_FLAT;
	else
		metap->storageKind = PGTURBOHYBRID_GRAPH_STORAGE_GRAPH;
	metap->graphEfSearch = PgturbohybridGraphUseTqGraph(index) ? PgturbohybridGraphGetEfSearch(index) : 0;
	metap->graphOversampling = PgturbohybridGraphUseTqGraph(index) ? PgturbohybridGraphGetGraphOversampling(index) : 0;
	metap->graphRescoreBand = PgturbohybridGraphUseTqGraph(index) ? PgturbohybridGraphGetGraphRescoreBand(index) : 0;
	metap->graphMaxLevel = buildstate->maxLevel;
	metap->graphFlags = PgturbohybridGraphUseTqCodes(index) ? 1 : 0;
	metap->tqBits = PGTURBOHYBRID_DEFAULT_BITS;
	metap->entryBlkno = InvalidBlockNumber;
	metap->entryOffno = InvalidOffsetNumber;
	metap->entryLevel = -1;
	metap->insertPage = InvalidBlockNumber;
	metap->tqCodeStartBlkno = InvalidBlockNumber;
	metap->tqAdjStartBlkno = InvalidBlockNumber;
	metap->tqExactStartBlkno = InvalidBlockNumber;
	metap->tqCorrectionStartBlkno = InvalidBlockNumber;
	((PageHeader) page)->pd_lower =
		((char *) metap + sizeof(PgturbohybridGraphMetaPageData)) - (char *) page;

	MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);
}

/*
 * Add a new page
 */
static void
PgturbohybridGraphBuildAppendPage(Relation index, Buffer *buf, Page *page, ForkNumber forkNum)
{
	/* Add a new page */
	Buffer		newbuf = PgturbohybridGraphNewBuffer(index, forkNum);

	/* Update previous page */
	PgturbohybridGraphPageGetOpaque(*page)->nextblkno = BufferGetBlockNumber(newbuf);
	PgturbohybridGraphMarkPageGraphOp(*page, PGTURBOHYBRID_GRAPH_GRAPH_OP_PAGE_LINK);

	/* Commit */
	MarkBufferDirty(*buf);
	UnlockReleaseBuffer(*buf);

	/* Can take a while, so ensure we can interrupt */
	/* Needs to be called when no buffer locks are held */
	LockBuffer(newbuf, BUFFER_LOCK_UNLOCK);
	CHECK_FOR_INTERRUPTS();
	LockBuffer(newbuf, BUFFER_LOCK_EXCLUSIVE);

	/* Prepare new page */
	*buf = newbuf;
	*page = BufferGetPage(*buf);
	PgturbohybridGraphInitPage(*buf, *page);
}

/*
 * Create graph pages
 */
static void
CreateGraphPages(PgturbohybridGraphBuildState * buildstate)
{
	Relation	index = buildstate->index;
	ForkNumber	forkNum = buildstate->forkNum;
	Size		maxSize;
	PgturbohybridGraphElementTuple etup;
	PgturbohybridGraphNeighborTuple ntup;
	BlockNumber insertPage;
	PgturbohybridGraphElement entryPoint;
	Buffer		buf;
	Page		page;
	PgturbohybridGraphElementPtr iter = buildstate->graph->head;
	char	   *base = buildstate->hnswarea;

	/* Calculate sizes */
	maxSize = PGTURBOHYBRID_GRAPH_MAX_SIZE;

	/* Allocate once */
	etup = palloc0(PGTURBOHYBRID_GRAPH_TUPLE_ALLOC_SIZE);
	ntup = palloc0(PGTURBOHYBRID_GRAPH_TUPLE_ALLOC_SIZE);

	/* Prepare first page */
	buf = PgturbohybridGraphNewBuffer(index, forkNum);
	page = BufferGetPage(buf);
	PgturbohybridGraphInitPage(buf, page);

	while (!PgturbohybridGraphPtrIsNull(base, iter))
	{
		PgturbohybridGraphElement element = PgturbohybridGraphPtrAccess(base, iter);
		Size		etupSize;
		Size		ntupSize;
		Size		combinedSize;
		Pointer		valuePtr = PgturbohybridGraphPtrAccess(base, element->value);

		/* Update iterator */
		iter = element->next;

		/* Zero memory for each element */
		MemSet(etup, 0, PGTURBOHYBRID_GRAPH_TUPLE_ALLOC_SIZE);

		/* Calculate sizes */
		etupSize = PgturbohybridGraphElementTupleSize(index, valuePtr);
		ntupSize = PGTURBOHYBRID_GRAPH_NEIGHBOR_TUPLE_SIZE(element->level, buildstate->m);
		combinedSize = etupSize + ntupSize + sizeof(ItemIdData);

		/* Initial size check */
		if (etupSize > PGTURBOHYBRID_GRAPH_TUPLE_ALLOC_SIZE)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("index tuple too large")));

		PgturbohybridGraphSetElementTuple(index, base, etup, element);

		/* Keep element and neighbors on the same page if possible */
		if (PageGetFreeSpace(page) < etupSize || (combinedSize <= maxSize && PageGetFreeSpace(page) < combinedSize))
			PgturbohybridGraphBuildAppendPage(index, &buf, &page, forkNum);

		/* Calculate offsets */
		element->blkno = BufferGetBlockNumber(buf);
		element->offno = OffsetNumberNext(PageGetMaxOffsetNumber(page));
		if (combinedSize <= maxSize)
		{
			element->neighborPage = element->blkno;
			element->neighborOffno = OffsetNumberNext(element->offno);
		}
		else
		{
			element->neighborPage = element->blkno + 1;
			element->neighborOffno = FirstOffsetNumber;
		}

		ItemPointerSet(&etup->neighbortid, element->neighborPage, element->neighborOffno);

	/* Add element */
	if (PageAddItem(page, (Item) etup, etupSize, InvalidOffsetNumber, false, false) != element->offno)
		elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));
	PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT);

		/* Add new page if needed */
		if (PageGetFreeSpace(page) < ntupSize)
			PgturbohybridGraphBuildAppendPage(index, &buf, &page, forkNum);

	/* Add placeholder for neighbors */
	if (PageAddItem(page, (Item) ntup, ntupSize, InvalidOffsetNumber, false, false) != element->neighborOffno)
		elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));
	PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_NEIGHBOR_INSERT);
	}

	insertPage = BufferGetBlockNumber(buf);

	/* Commit */
	MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);

	entryPoint = PgturbohybridGraphPtrAccess(base, buildstate->graph->entryPoint);
	PgturbohybridGraphUpdateMetaPage(index, PGTURBOHYBRID_GRAPH_UPDATE_ENTRY_ALWAYS, entryPoint, insertPage, forkNum, true);

	pfree(etup);
	pfree(ntup);
}

/*
 * Write neighbor tuples
 */
static void
WriteNeighborTuples(PgturbohybridGraphBuildState * buildstate)
{
	Relation	index = buildstate->index;
	ForkNumber	forkNum = buildstate->forkNum;
	int			m = buildstate->m;
	PgturbohybridGraphElementPtr iter = buildstate->graph->head;
	char	   *base = buildstate->hnswarea;
	PgturbohybridGraphNeighborTuple ntup;

	/* Allocate once */
	ntup = palloc0(PGTURBOHYBRID_GRAPH_TUPLE_ALLOC_SIZE);

	while (!PgturbohybridGraphPtrIsNull(base, iter))
	{
		PgturbohybridGraphElement element = PgturbohybridGraphPtrAccess(base, iter);
		Buffer		buf;
		Page		page;
		Size		ntupSize = PGTURBOHYBRID_GRAPH_NEIGHBOR_TUPLE_SIZE(element->level, m);

		/* Update iterator */
		iter = element->next;

		/* Zero memory for each element */
		MemSet(ntup, 0, PGTURBOHYBRID_GRAPH_TUPLE_ALLOC_SIZE);

		/* Can take a while, so ensure we can interrupt */
		/* Needs to be called when no buffer locks are held */
		CHECK_FOR_INTERRUPTS();

		buf = ReadBufferExtended(index, forkNum, element->neighborPage, RBM_NORMAL, NULL);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
		page = BufferGetPage(buf);

		PgturbohybridGraphSetNeighborTuple(base, ntup, element, m);

		if (!PageIndexTupleOverwrite(page, element->neighborOffno, (Item) ntup, ntupSize))
			elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));
		PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_NEIGHBOR_UPDATE);

		/* Commit */
		MarkBufferDirty(buf);
		UnlockReleaseBuffer(buf);
	}

	pfree(ntup);
}

/*
 * Flush pages
 */
static void
FlushPages(PgturbohybridGraphBuildState * buildstate)
{
#ifdef PGTURBOHYBRID_GRAPH_MEMORY
	elog(INFO, "memory: %zu MB", buildstate->graph->memoryUsed / (1024 * 1024));
#endif

	CreateMetaPage(buildstate);
	CreateGraphPages(buildstate);
	WriteNeighborTuples(buildstate);

	buildstate->graph->flushed = true;
	MemoryContextReset(buildstate->graphCtx);
}

/*
 * Add a heap TID to an existing element
 */
static bool
AddDuplicateInMemory(PgturbohybridGraphElement element, PgturbohybridGraphElement dup)
{
	LWLockAcquire(&dup->lock, LW_EXCLUSIVE);

	if (dup->heaptidsLength == PGTURBOHYBRID_GRAPH_HEAPTIDS)
	{
		LWLockRelease(&dup->lock);
		return false;
	}

	PgturbohybridGraphAddHeapTid(dup, &element->heaptids[0]);

	LWLockRelease(&dup->lock);

	return true;
}

/*
 * Find duplicate element
 */
static bool
FindDuplicateInMemory(char *base, PgturbohybridGraphElement element)
{
	PgturbohybridGraphNeighborArray *neighbors = PgturbohybridGraphGetNeighbors(base, element, 0);
	Datum		value = PgturbohybridGraphGetValue(base, element);

	for (int i = 0; i < neighbors->length; i++)
	{
		PgturbohybridGraphCandidate *neighbor = &neighbors->items[i];
		PgturbohybridGraphElement neighborElement = PgturbohybridGraphPtrAccess(base, neighbor->element);
		Datum		neighborValue = PgturbohybridGraphGetValue(base, neighborElement);

		/* Exit early since ordered by distance */
		if (!datumIsEqual(value, neighborValue, false, -1))
			return false;

		/* Check for space */
		if (AddDuplicateInMemory(element, neighborElement))
			return true;
	}

	return false;
}

/*
 * Add to element list
 */
static void
AddElementInMemory(char *base, PgturbohybridGraphGraph * graph, PgturbohybridGraphElement element)
{
	SpinLockAcquire(&graph->lock);
	element->next = graph->head;
	PgturbohybridGraphPtrStore(base, graph->head, element);
	SpinLockRelease(&graph->lock);
}

/*
 * Update neighbors
 */
static void
UpdateNeighborsInMemory(char *base, PgturbohybridGraphSupport * support, PgturbohybridGraphElement e, int m)
{
	for (int lc = e->level; lc >= 0; lc--)
	{
		int			lm = PgturbohybridGraphGetLayerM(m, lc);
		Size		neighborsSize = PGTURBOHYBRID_GRAPH_NEIGHBOR_ARRAY_SIZE(lm);
		PgturbohybridGraphNeighborArray *neighbors = palloc(neighborsSize);

		/* Copy neighbors to local memory */
		LWLockAcquire(&e->lock, LW_SHARED);
		memcpy(neighbors, PgturbohybridGraphGetNeighbors(base, e, lc), neighborsSize);
		LWLockRelease(&e->lock);

		for (int i = 0; i < neighbors->length; i++)
		{
			PgturbohybridGraphCandidate *hc = &neighbors->items[i];
			PgturbohybridGraphElement neighborElement = PgturbohybridGraphPtrAccess(base, hc->element);

			/* Keep scan-build happy on Mac x86-64 */
			Assert(neighborElement);

			LWLockAcquire(&neighborElement->lock, LW_EXCLUSIVE);
			PgturbohybridGraphUpdateConnection(base, PgturbohybridGraphGetNeighbors(base, neighborElement, lc), e, hc->distance, lm, NULL, NULL, support);
			LWLockRelease(&neighborElement->lock);
		}
	}
}

/*
 * Update graph in memory
 */
static void
UpdateGraphInMemory(PgturbohybridGraphSupport * support, PgturbohybridGraphElement element, int m, PgturbohybridGraphElement entryPoint, PgturbohybridGraphBuildState * buildstate)
{
	PgturbohybridGraphGraph  *graph = buildstate->graph;
	char	   *base = buildstate->hnswarea;

	/* Look for duplicate */
	if (FindDuplicateInMemory(base, element))
		return;

	/* Add element */
	AddElementInMemory(base, graph, element);

	/* Update neighbors */
	UpdateNeighborsInMemory(base, support, element, m);

	/* Update entry point if needed (already have lock) */
	if (entryPoint == NULL || element->level > entryPoint->level)
		PgturbohybridGraphPtrStore(base, graph->entryPoint, element);
}

/*
 * Insert tuple in memory
 */
static void
InsertTupleInMemory(PgturbohybridGraphBuildState * buildstate, PgturbohybridGraphElement element)
{
	PgturbohybridGraphGraph  *graph = buildstate->graph;
	PgturbohybridGraphSupport *support = &buildstate->support;
	PgturbohybridGraphElement entryPoint;
	LWLock	   *entryLock = &graph->entryLock;
	LWLock	   *entryWaitLock = &graph->entryWaitLock;
	int			efConstruction = buildstate->efConstruction;
	int			m = buildstate->m;
	char	   *base = buildstate->hnswarea;

	/* Wait if another process needs exclusive lock on entry lock */
	LWLockAcquire(entryWaitLock, LW_EXCLUSIVE);
	LWLockRelease(entryWaitLock);

	/* Get entry point */
	LWLockAcquire(entryLock, LW_SHARED);
	entryPoint = PgturbohybridGraphPtrAccess(base, graph->entryPoint);

	/* Prevent concurrent inserts when likely updating entry point */
	if (entryPoint == NULL || element->level > entryPoint->level)
	{
		/* Release shared lock */
		LWLockRelease(entryLock);

		/* Tell other processes to wait and get exclusive lock */
		LWLockAcquire(entryWaitLock, LW_EXCLUSIVE);
		LWLockAcquire(entryLock, LW_EXCLUSIVE);
		LWLockRelease(entryWaitLock);

		/* Get latest entry point after lock is acquired */
		entryPoint = PgturbohybridGraphPtrAccess(base, graph->entryPoint);
	}

	/* Find neighbors for element */
	PgturbohybridGraphFindElementNeighbors(base, element, entryPoint, NULL, support, m, efConstruction, false);

	/* Update graph in memory */
	UpdateGraphInMemory(support, element, m, entryPoint, buildstate);

	/* Release entry lock */
	LWLockRelease(entryLock);
}

/*
 * Insert tuple
 */
static bool
InsertTuple(Relation index, Datum *values, bool *isnull, ItemPointer heaptid, PgturbohybridGraphBuildState * buildstate)
{
	PgturbohybridGraphGraph  *graph = buildstate->graph;
	PgturbohybridGraphElement element;
	PgturbohybridGraphAllocator *allocator = &buildstate->allocator;
	PgturbohybridGraphSupport *support = &buildstate->support;
	Size		valueSize;
	Pointer		valuePtr;
	LWLock	   *flushLock = &graph->flushLock;
	char	   *base = buildstate->hnswarea;
	Datum		value;
	Size		memoryMargin;

	/* Form index value */
	if (!PgturbohybridGraphFormIndexValue(&value, values, isnull, buildstate->typeInfo, support))
		return false;

	/* Get datum size */
	valueSize = VARSIZE_ANY(DatumGetPointer(value));

	/* In a parallel build, add a margin so allocations never fail */
	memoryMargin = base == NULL ? 0 : 1024 * 1024;

	/* Ensure graph not flushed when inserting */
	LWLockAcquire(flushLock, LW_SHARED);

	/* Are we in the on-disk phase? */
	if (graph->flushed)
	{
		LWLockRelease(flushLock);

		return PgturbohybridGraphInsertTupleOnDisk(index, support, value, heaptid, true);
	}

	/*
	 * In a parallel build, the PgturbohybridGraphElement is allocated from the shared
	 * memory area, so we need to coordinate with other processes.
	 */
	LWLockAcquire(&graph->allocatorLock, LW_EXCLUSIVE);

	/*
	 * Check that we have enough memory available for the new element now that
	 * we have the allocator lock, and flush pages if needed.
	 */
	if (graph->memoryUsed + memoryMargin >= graph->memoryTotal)
	{
		LWLockRelease(&graph->allocatorLock);

		LWLockRelease(flushLock);
		LWLockAcquire(flushLock, LW_EXCLUSIVE);

		if (!graph->flushed)
		{
			ereport(NOTICE,
					(errmsg("pgturbohybrid graph no longer fits into maintenance_work_mem after " INT64_FORMAT " tuples", (int64) graph->indtuples),
					 errdetail("Building will take significantly more time."),
					 errhint("Increase maintenance_work_mem to speed up builds.")));

			FlushPages(buildstate);
		}

		LWLockRelease(flushLock);

		return PgturbohybridGraphInsertTupleOnDisk(index, support, value, heaptid, true);
	}

	/* Ok, we can proceed to allocate the element */
	element = PgturbohybridGraphInitElement(base, heaptid, buildstate->m, buildstate->ml, buildstate->maxLevel, allocator);
	valuePtr = PgturbohybridGraphAlloc(allocator, valueSize);

	/*
	 * We have now allocated the space needed for the element, so we don't
	 * need the allocator lock anymore. Release it and initialize the rest of
	 * the element.
	 */
	LWLockRelease(&graph->allocatorLock);

	/* Copy the datum */
	memcpy(valuePtr, DatumGetPointer(value), valueSize);
	PgturbohybridGraphPtrStore(base, element->value, (char *) valuePtr);

	/* Create a lock for the element */
	LWLockInitialize(&element->lock, pgturbohybrid_graph_lock_tranche_id);

	/* Insert tuple */
	InsertTupleInMemory(buildstate, element);

	/* Release flush lock */
	LWLockRelease(flushLock);

	return true;
}

/*
 * Callback for table_index_build_scan
 */
static void
BuildCallback(Relation index, ItemPointer tid, Datum *values,
			  bool *isnull, bool tupleIsAlive, void *state)
{
	PgturbohybridGraphBuildState *buildstate = (PgturbohybridGraphBuildState *) state;
	PgturbohybridGraphGraph  *graph = buildstate->graph;
	MemoryContext oldCtx;

	/* Skip nulls */
	if (isnull[0])
		return;

	/* Use memory context */
	oldCtx = MemoryContextSwitchTo(buildstate->tmpCtx);

	/* Insert tuple */
	if (InsertTuple(index, values, isnull, tid, buildstate))
	{
		/* Update progress */
		SpinLockAcquire(&graph->lock);
		pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_DONE, ++graph->indtuples);
		SpinLockRelease(&graph->lock);
	}

	/* Reset memory context */
	MemoryContextSwitchTo(oldCtx);
	MemoryContextReset(buildstate->tmpCtx);
}

/*
 * Initialize the graph
 */
static void
InitGraph(PgturbohybridGraphGraph * graph, char *base, Size memoryTotal)
{
	/* Initialize the lock tranche if needed */
	PgturbohybridGraphInitLockTranche();

	PgturbohybridGraphPtrStore(base, graph->head, (PgturbohybridGraphElement) NULL);
	PgturbohybridGraphPtrStore(base, graph->entryPoint, (PgturbohybridGraphElement) NULL);
	graph->memoryUsed = 0;
	graph->memoryTotal = Min(memoryTotal, PGTURBOHYBRID_GRAPH_MAX_GRAPH_MEMORY);
	graph->flushed = false;
	graph->indtuples = 0;
	SpinLockInit(&graph->lock);
	LWLockInitialize(&graph->entryLock, pgturbohybrid_graph_lock_tranche_id);
	LWLockInitialize(&graph->entryWaitLock, pgturbohybrid_graph_lock_tranche_id);
	LWLockInitialize(&graph->allocatorLock, pgturbohybrid_graph_lock_tranche_id);
	LWLockInitialize(&graph->flushLock, pgturbohybrid_graph_lock_tranche_id);
}

/*
 * Initialize an allocator
 */
static void
InitAllocator(PgturbohybridGraphAllocator * allocator, void *(*alloc) (Size size, void *state), void *state)
{
	allocator->alloc = alloc;
	allocator->state = state;
}

/*
 * Memory context allocator
 */
static void *
PgturbohybridGraphMemoryContextAlloc(Size size, void *state)
{
	PgturbohybridGraphBuildState *buildstate = (PgturbohybridGraphBuildState *) state;
	void	   *chunk = MemoryContextAlloc(buildstate->graphCtx, size);

	buildstate->graphData.memoryUsed = MemoryContextMemAllocated(buildstate->graphCtx, false);

	return chunk;
}

/*
 * Shared memory allocator
 */
static void *
PgturbohybridGraphSharedMemoryAlloc(Size size, void *state)
{
	PgturbohybridGraphBuildState *buildstate = (PgturbohybridGraphBuildState *) state;
	Size		alignedSize = MAXALIGN(size);
	void	   *chunk;

	if (alignedSize > 1024 * 1024)
		elog(ERROR, "pgturbohybrid allocation too large");

	if (buildstate->graph->memoryUsed + alignedSize > buildstate->graph->memoryTotal)
		elog(ERROR, "pgturbohybrid allocator out of memory");

	chunk = buildstate->hnswarea + buildstate->graph->memoryUsed;
	buildstate->graph->memoryUsed += alignedSize;
	return chunk;
}

/*
 * Initialize the build state
 */
static void
InitBuildState(PgturbohybridGraphBuildState * buildstate, Relation heap, Relation index, IndexInfo *indexInfo, ForkNumber forkNum)
{
	buildstate->heap = heap;
	buildstate->index = index;
	buildstate->indexInfo = indexInfo;
	buildstate->forkNum = forkNum;
	buildstate->typeInfo = PgturbohybridGraphGetTypeInfo(index);

	buildstate->m = PgturbohybridGraphGetM(index);
	buildstate->efConstruction = PgturbohybridGraphGetEfConstruction(index);
	buildstate->dimensions = TupleDescAttr(index->rd_att, 0)->atttypmod;

	/* Disallow varbit since require fixed dimensions */
	if (TupleDescAttr(index->rd_att, 0)->atttypid == VARBITOID)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("type not supported for pgturbohybrid index")));

	/* Require column to have dimensions to be indexed */
	if (buildstate->dimensions < 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("column does not have dimensions")));

	if (buildstate->dimensions > buildstate->typeInfo->maxDimensions)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("column cannot have more than %d dimensions for pgturbohybrid index", buildstate->typeInfo->maxDimensions)));

	if (buildstate->efConstruction < 2 * buildstate->m)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("ef_construction must be greater than or equal to 2 * m")));

	buildstate->reltuples = 0;
	buildstate->indtuples = 0;

	/* Get support functions */
	PgturbohybridGraphInitSupport(&buildstate->support, index);

	InitGraph(&buildstate->graphData, NULL, (Size) maintenance_work_mem * 1024L);
	buildstate->graph = &buildstate->graphData;
	buildstate->ml = PgturbohybridGraphGetMl(buildstate->m);
	buildstate->maxLevel = PgturbohybridGraphGetMaxLevel(buildstate->m);

	buildstate->graphCtx = GenerationContextCreate(CurrentMemoryContext,
												   "PgturbohybridGraph build graph context",
#if PG_VERSION_NUM >= 150000
												   1024 * 1024, 1024 * 1024,
#endif
												   1024 * 1024);
	buildstate->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
											   "PgturbohybridGraph build temporary context",
											   ALLOCSET_DEFAULT_SIZES);

	InitAllocator(&buildstate->allocator, &PgturbohybridGraphMemoryContextAlloc, buildstate);

	buildstate->hnswleader = NULL;
	buildstate->hnswshared = NULL;
	buildstate->hnswarea = NULL;
}

/*
 * Free resources
 */
static void
FreeBuildState(PgturbohybridGraphBuildState * buildstate)
{
	MemoryContextDelete(buildstate->graphCtx);
	MemoryContextDelete(buildstate->tmpCtx);
}

/*
 * Within leader, wait for end of heap scan
 */
static double
ParallelHeapScan(PgturbohybridGraphBuildState * buildstate)
{
	PgturbohybridGraphShared *hnswshared = buildstate->hnswleader->hnswshared;
	int			nparticipanttuplesorts;
	double		reltuples;

	nparticipanttuplesorts = buildstate->hnswleader->nparticipanttuplesorts;
	for (;;)
	{
		SpinLockAcquire(&hnswshared->mutex);
		if (hnswshared->nparticipantsdone == nparticipanttuplesorts)
		{
			buildstate->graph = &hnswshared->graphData;
			buildstate->hnswarea = buildstate->hnswleader->hnswarea;
			reltuples = hnswshared->reltuples;
			SpinLockRelease(&hnswshared->mutex);
			break;
		}
		SpinLockRelease(&hnswshared->mutex);

		ConditionVariableSleep(&hnswshared->workersdonecv,
							   WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
	}

	ConditionVariableCancelSleep();

	return reltuples;
}

/*
 * Perform a worker's portion of a parallel insert
 */
static void
PgturbohybridGraphParallelScanAndInsert(Relation heapRel, Relation indexRel, PgturbohybridGraphShared * hnswshared, char *hnswarea, bool progress)
{
	PgturbohybridGraphBuildState buildstate;
	TableScanDesc scan;
	double		reltuples;
	IndexInfo  *indexInfo;

	/* Join parallel scan */
	indexInfo = BuildIndexInfo(indexRel);
	indexInfo->ii_Concurrent = hnswshared->isconcurrent;
	InitBuildState(&buildstate, heapRel, indexRel, indexInfo, MAIN_FORKNUM);
	buildstate.graph = &hnswshared->graphData;
	buildstate.hnswarea = hnswarea;
	InitAllocator(&buildstate.allocator, &PgturbohybridGraphSharedMemoryAlloc, &buildstate);
	scan = table_beginscan_parallel(heapRel,
									ParallelTableScanFromHnswShared(hnswshared)
#if PG_VERSION_NUM >= 190000
									,SO_NONE
#endif
		);
	reltuples = table_index_build_scan(heapRel, indexRel, indexInfo,
									   true, progress, BuildCallback,
									   (void *) &buildstate, scan);

	/* Record statistics */
	SpinLockAcquire(&hnswshared->mutex);
	hnswshared->nparticipantsdone++;
	hnswshared->reltuples += reltuples;
	SpinLockRelease(&hnswshared->mutex);

	/* Log statistics */
	if (progress)
		ereport(DEBUG1, (errmsg("leader processed " INT64_FORMAT " tuples", (int64) reltuples)));
	else
		ereport(DEBUG1, (errmsg("worker processed " INT64_FORMAT " tuples", (int64) reltuples)));

	/* Notify leader */
	ConditionVariableSignal(&hnswshared->workersdonecv);

	FreeBuildState(&buildstate);
}

/*
 * Perform work within a launched parallel process
 */
void
PgturbohybridParallelBuildMain(dsm_segment *seg, shm_toc *toc)
{
	char	   *sharedquery;
	PgturbohybridGraphShared *hnswshared;
	char	   *hnswarea;
	Relation	heapRel;
	Relation	indexRel;
	LOCKMODE	heapLockmode;
	LOCKMODE	indexLockmode;

	/* Set debug_query_string for individual workers first */
	sharedquery = shm_toc_lookup(toc, PARALLEL_KEY_QUERY_TEXT, true);
	debug_query_string = sharedquery;

	/* Report the query string from leader */
	pgstat_report_activity(STATE_RUNNING, debug_query_string);

	/* Look up shared state */
	hnswshared = shm_toc_lookup(toc, PARALLEL_KEY_PGTURBOHYBRID_SHARED, false);

	/* Open relations using lock modes known to be obtained by index.c */
	if (!hnswshared->isconcurrent)
	{
		heapLockmode = ShareLock;
		indexLockmode = AccessExclusiveLock;
	}
	else
	{
		heapLockmode = ShareUpdateExclusiveLock;
		indexLockmode = RowExclusiveLock;
	}

	/* Open relations within worker */
	heapRel = table_open(hnswshared->heaprelid, heapLockmode);
	indexRel = index_open(hnswshared->indexrelid, indexLockmode);

	hnswarea = shm_toc_lookup(toc, PARALLEL_KEY_PGTURBOHYBRID_AREA, false);

	/* Perform inserts */
	PgturbohybridGraphParallelScanAndInsert(heapRel, indexRel, hnswshared, hnswarea, false);

	/* Close relations within worker */
	index_close(indexRel, indexLockmode);
	table_close(heapRel, heapLockmode);
}

/*
 * End parallel build
 */
static void
PgturbohybridGraphEndParallel(PgturbohybridGraphLeader * hnswleader)
{
	/* Shutdown worker processes */
	WaitForParallelWorkersToFinish(hnswleader->pcxt);

	/* Free last reference to MVCC snapshot, if one was used */
	if (IsMVCCSnapshot(hnswleader->snapshot))
		UnregisterSnapshot(hnswleader->snapshot);
	DestroyParallelContext(hnswleader->pcxt);
	ExitParallelMode();
}

/*
 * Return size of shared memory required for parallel index build
 */
static Size
ParallelEstimateShared(Relation heap, Snapshot snapshot)
{
	return add_size(BUFFERALIGN(sizeof(PgturbohybridGraphShared)), table_parallelscan_estimate(heap, snapshot));
}

/*
 * Within leader, participate as a parallel worker
 */
static void
PgturbohybridGraphLeaderParticipateAsWorker(PgturbohybridGraphBuildState * buildstate)
{
	PgturbohybridGraphLeader *hnswleader = buildstate->hnswleader;

	/* Perform work common to all participants */
	PgturbohybridGraphParallelScanAndInsert(buildstate->heap, buildstate->index, hnswleader->hnswshared, hnswleader->hnswarea, true);
}

/*
 * Begin parallel build
 */
static void
PgturbohybridGraphBeginParallel(PgturbohybridGraphBuildState * buildstate, bool isconcurrent, int request)
{
	ParallelContext *pcxt;
	Snapshot	snapshot;
	Size		esthnswshared;
	Size		esthnswarea;
	Size		estother;
	PgturbohybridGraphShared *hnswshared;
	char	   *hnswarea;
	PgturbohybridGraphLeader *hnswleader = (PgturbohybridGraphLeader *) palloc0(sizeof(PgturbohybridGraphLeader));
	bool		leaderparticipates = true;
	int			querylen;

#ifdef DISABLE_LEADER_PARTICIPATION
	leaderparticipates = false;
#endif

	/* Enter parallel mode and create context */
	EnterParallelMode();
	Assert(request > 0);
	pcxt = CreateParallelContext("pgturbohybrid", "PgturbohybridParallelBuildMain", request);

	/* Get snapshot for table scan */
	if (!isconcurrent)
		snapshot = SnapshotAny;
	else
		snapshot = RegisterSnapshot(GetTransactionSnapshot());

	/* Estimate size of workspaces */
	esthnswshared = ParallelEstimateShared(buildstate->heap, snapshot);
	shm_toc_estimate_chunk(&pcxt->estimator, esthnswshared);

	/* Leave space for other objects in shared memory */
	/* Docker has a default limit of 64 MB for shm_size */
	/* which happens to be the default value of maintenance_work_mem */
	esthnswarea = maintenance_work_mem * 1024L;
	estother = 3 * 1024 * 1024;
	if (esthnswarea > estother)
		esthnswarea -= estother;

	esthnswarea = Min(esthnswarea, PGTURBOHYBRID_GRAPH_MAX_GRAPH_MEMORY);

	shm_toc_estimate_chunk(&pcxt->estimator, esthnswarea);
	shm_toc_estimate_keys(&pcxt->estimator, 2);

	/* Finally, estimate PARALLEL_KEY_QUERY_TEXT space */
	if (debug_query_string)
	{
		querylen = strlen(debug_query_string);
		shm_toc_estimate_chunk(&pcxt->estimator, querylen + 1);
		shm_toc_estimate_keys(&pcxt->estimator, 1);
	}
	else
		querylen = 0;			/* keep compiler quiet */

	/* Everyone's had a chance to ask for space, so now create the DSM */
	InitializeParallelDSM(pcxt);

	/* If no DSM segment was available, back out (do serial build) */
	if (pcxt->seg == NULL)
	{
		if (IsMVCCSnapshot(snapshot))
			UnregisterSnapshot(snapshot);
		DestroyParallelContext(pcxt);
		ExitParallelMode();
		return;
	}

	/* Store shared build state, for which we reserved space */
	hnswshared = (PgturbohybridGraphShared *) shm_toc_allocate(pcxt->toc, esthnswshared);
	/* Initialize immutable state */
	hnswshared->heaprelid = RelationGetRelid(buildstate->heap);
	hnswshared->indexrelid = RelationGetRelid(buildstate->index);
	hnswshared->isconcurrent = isconcurrent;
	ConditionVariableInit(&hnswshared->workersdonecv);
	SpinLockInit(&hnswshared->mutex);
	/* Initialize mutable state */
	hnswshared->nparticipantsdone = 0;
	hnswshared->reltuples = 0;
	table_parallelscan_initialize(buildstate->heap,
								  ParallelTableScanFromHnswShared(hnswshared),
								  snapshot);

	hnswarea = (char *) shm_toc_allocate(pcxt->toc, esthnswarea);
	InitGraph(&hnswshared->graphData, hnswarea, esthnswarea);

	/*
	 * Avoid base address for relptr for Postgres < 14.5
	 * https://github.com/postgres/postgres/commit/7201cd18627afc64850537806da7f22150d1a83b
	 */
#if PG_VERSION_NUM < 140005
	hnswshared->graphData.memoryUsed += MAXALIGN(1);
#endif

	shm_toc_insert(pcxt->toc, PARALLEL_KEY_PGTURBOHYBRID_SHARED, hnswshared);
	shm_toc_insert(pcxt->toc, PARALLEL_KEY_PGTURBOHYBRID_AREA, hnswarea);

	/* Store query string for workers */
	if (debug_query_string)
	{
		char	   *sharedquery;

		sharedquery = (char *) shm_toc_allocate(pcxt->toc, querylen + 1);
		memcpy(sharedquery, debug_query_string, querylen + 1);
		shm_toc_insert(pcxt->toc, PARALLEL_KEY_QUERY_TEXT, sharedquery);
	}

	/* Launch workers, saving status for leader/caller */
	LaunchParallelWorkers(pcxt);
	hnswleader->pcxt = pcxt;
	hnswleader->nparticipanttuplesorts = pcxt->nworkers_launched;
	if (leaderparticipates)
		hnswleader->nparticipanttuplesorts++;
	hnswleader->hnswshared = hnswshared;
	hnswleader->snapshot = snapshot;
	hnswleader->hnswarea = hnswarea;

	/* If no workers were successfully launched, back out (do serial build) */
	if (pcxt->nworkers_launched == 0)
	{
		PgturbohybridGraphEndParallel(hnswleader);
		return;
	}

	/* Log participants */
	ereport(DEBUG1, (errmsg("using %d parallel workers", pcxt->nworkers_launched)));

	/* Save leader state now that it's clear build will be parallel */
	buildstate->hnswleader = hnswleader;

	/* Join heap scan ourselves */
	if (leaderparticipates)
		PgturbohybridGraphLeaderParticipateAsWorker(buildstate);

	/* Wait for all launched workers */
	WaitForParallelWorkersToAttach(pcxt);
}

/*
 * Compute parallel workers
 */
static int
ComputeParallelWorkers(Relation heap, Relation index)
{
	int			parallel_workers;

	/* Make sure it's safe to use parallel workers */
	parallel_workers = plan_create_index_workers(RelationGetRelid(heap), RelationGetRelid(index));
	if (parallel_workers == 0)
		return 0;

	/* Use parallel_workers storage parameter on table if set */
	parallel_workers = RelationGetParallelWorkers(heap, -1);
	if (parallel_workers != -1)
		return Min(parallel_workers, max_parallel_maintenance_workers);

	return max_parallel_maintenance_workers;
}

/*
 * Build graph
 */
static void
BuildGraph(PgturbohybridGraphBuildState * buildstate)
{
	int			parallel_workers = 0;

	pgstat_progress_update_param(PROGRESS_CREATEIDX_SUBPHASE, PGTURBOHYBRID_PROGRESS_PHASE_LOAD);

	/* Calculate parallel workers */
	if (buildstate->heap != NULL)
		parallel_workers = ComputeParallelWorkers(buildstate->heap, buildstate->index);

	/* Attempt to launch parallel worker scan when required */
	if (parallel_workers > 0)
		PgturbohybridGraphBeginParallel(buildstate, buildstate->indexInfo->ii_Concurrent, parallel_workers);

	/* Add tuples to graph */
	if (buildstate->heap != NULL)
	{
		if (buildstate->hnswleader)
			buildstate->reltuples = ParallelHeapScan(buildstate);
		else
			buildstate->reltuples = table_index_build_scan(buildstate->heap, buildstate->index, buildstate->indexInfo,
														   true, true, BuildCallback, (void *) buildstate, NULL);

		buildstate->indtuples = buildstate->graph->indtuples;
	}

	/* Flush pages */
	if (!buildstate->graph->flushed)
		FlushPages(buildstate);

	/* End parallel build */
	if (buildstate->hnswleader)
		PgturbohybridGraphEndParallel(buildstate->hnswleader);
}

/*
 * Build the index
 */
static void
BuildIndex(Relation heap, Relation index, IndexInfo *indexInfo,
		   PgturbohybridGraphBuildState * buildstate, ForkNumber forkNum)
{
#ifdef PGTURBOHYBRID_GRAPH_MEMORY
	SeedRandom(42);
#endif

	InitBuildState(buildstate, heap, index, indexInfo, forkNum);

	BuildGraph(buildstate);

	if (RelationNeedsWAL(index) || forkNum == INIT_FORKNUM)
		log_newpage_range(index, forkNum, 0, RelationGetNumberOfBlocksInFork(index, forkNum), true);

	FreeBuildState(buildstate);
}

/*
 * Build the index for a logged table
 */
IndexBuildResult *
hnswbuild(Relation heap, Relation index, IndexInfo *indexInfo)
{
	IndexBuildResult *result;
	PgturbohybridGraphBuildState buildstate;

	BuildIndex(heap, index, indexInfo, &buildstate, MAIN_FORKNUM);

	result = (IndexBuildResult *) palloc(sizeof(IndexBuildResult));
	result->heap_tuples = buildstate.reltuples;
	result->index_tuples = buildstate.indtuples;

	return result;
}

IndexBuildResult *
pgturbohybridbuild(Relation heap, Relation index, IndexInfo *indexInfo)
{
	IndexBuildResult *result;

	if (PgturbohybridGraphUseTqNativeGraph(index))
		return tqgraphbuild(heap, index, indexInfo);

	if (!PgturbohybridGraphUseTqFlat(index))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("turbohybrid requires a native graph-compatible opclass"),
				 errhint("Use a turbohybrid vector opclass.")));

	PgturbohybridGraphSetForcepgturbohybridIndex(true);
	PG_TRY();
	{
		result = hnswbuild(heap, index, indexInfo);
	}
	PG_CATCH();
	{
		PgturbohybridGraphSetForcepgturbohybridIndex(false);
		PG_RE_THROW();
	}
	PG_END_TRY();
	PgturbohybridGraphSetForcepgturbohybridIndex(false);

	return result;
}

void
pgturbohybridbuildempty(Relation index)
{
	if (PgturbohybridGraphUseTqNativeGraph(index))
	{
		tqgraphbuildempty(index);
		return;
	}

	if (!PgturbohybridGraphUseTqFlat(index))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("turbohybrid requires a native graph-compatible opclass"),
				 errhint("Use a turbohybrid vector opclass.")));

	hnswbuildempty(index);
}

/*
 * Build the index for an unlogged table
 */
void
hnswbuildempty(Relation index)
{
	IndexInfo  *indexInfo = BuildIndexInfo(index);
	PgturbohybridGraphBuildState buildstate;

	BuildIndex(NULL, index, indexInfo, &buildstate, INIT_FORKNUM);
}
