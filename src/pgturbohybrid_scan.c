#include "postgres.h"

#include <math.h>
#include <limits.h>

#include "access/genam.h"
#include "access/relscan.h"
#include "fmgr.h"
#include "pgturbohybrid.h"
#include "lib/pairingheap.h"
#include "miscadmin.h"
#include "nodes/pg_list.h"
#include "pgstat.h"
#include "portability/instr_time.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/float.h"
#include "utils/memutils.h"
#include "utils/relcache.h"
#include "utils/snapmgr.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

#include "pgturbohybrid_vector_compat.h"
#include "pgturbohybrid_sparse.h"

static inline int64
PgturbohybridGraphScanElapsedUs(instr_time start)
{
	instr_time	elapsed;

	INSTR_TIME_SET_CURRENT(elapsed);
	INSTR_TIME_SUBTRACT(elapsed, start);
	return (int64) INSTR_TIME_GET_MICROSEC(elapsed);
}

/*
 * Get the initial iterative scan batch size
 */
static int
GetInitialIterativeBatchSize(PgturbohybridGraphScanOpaque so)
{
	return so->efSearch;
}

static int
CompareSearchCandidateDistances(const ListCell *a, const ListCell *b)
{
	PgturbohybridGraphSearchCandidate *sca = lfirst(a);
	PgturbohybridGraphSearchCandidate *scb = lfirst(b);

	if (sca->distance < scb->distance)
		return 1;

	if (sca->distance > scb->distance)
		return -1;

	return 0;
}

static int
GetGraphAutoRescoreLimit(PgturbohybridGraphScanOpaque so, List *items)
{
	/*
	 * The packed-code graph scorer is for traversal, not final ranking.
	 * Exact-rescore the complete candidate band returned by traversal so
	 * auto mode cannot drop true top-k rows due to code-domain ordering.
	 * This remains final-band-only: rescore reads only pages for returned
	 * candidates, not every visited page.
	 */
	return list_length(items);
}

static List *
SelectGraphRescoreBand(PgturbohybridGraphScanOpaque so, List *items)
{
	List	   *band = NIL;
	int			rescoreLimit;
	int			skip;
	int			pos = 0;
	ListCell   *lc;

	if (so->graphRescoreBand == PGTURBOHYBRID_GRAPH_RESCORE_BAND_EXACT)
		return items;

	rescoreLimit = GetGraphAutoRescoreLimit(so, items);
	if (rescoreLimit >= list_length(items))
		return items;

	list_sort(items, CompareSearchCandidateDistances);
	skip = list_length(items) - rescoreLimit;

	foreach(lc, items)
	{
		if (pos++ >= skip)
			band = lappend(band, lfirst(lc));
	}

	return band;
}

static void
RescoreScanItems(IndexScanDesc scan, List *items)
{
	PgturbohybridGraphScanOpaque so = (PgturbohybridGraphScanOpaque) scan->opaque;
	List	   *rescoreItems;

	if (!so->pgturbohybridGraphScan ||
		so->graphRescoreBand == PGTURBOHYBRID_GRAPH_RESCORE_BAND_NONE ||
		items == NIL)
		return;

	rescoreItems = SelectGraphRescoreBand(so, items);
	so->graphRescorePages += PgturbohybridGraphRescoreSearchCandidates(scan->indexRelation, &so->support, &so->q, rescoreItems);
	so->graphRescoreCount += list_length(rescoreItems);
	list_sort(items, CompareSearchCandidateDistances);
}

static void
RecordGraphScanBatchStats(PgturbohybridGraphScanOpaque so, int64 beforeTuples, int64 afterTuples, List *items)
{
	if (!so->pgturbohybridGraphScan)
		return;

	so->graphVisitedNodes += afterTuples - beforeTuples;
	so->graphCandidateCount += list_length(items);
	PgturbohybridGraphRecordGraphScanStats(so);
}

static inline double
GetFlatDistance(Datum value, PgturbohybridGraphElementTuple etup, PgturbohybridGraphSupport *support)
{
	if (DatumGetPointer(value) == NULL)
		return 0;

	return DatumGetFloat8(FunctionCall2Coll(support->procinfo,
											support->collation,
											value,
											PointerGetDatum(&etup->data)));
}

/*
 * Exact flat scan over element pages for explicit pgturbohybrid routing=flat.
 */
static List *
GetFlatScanItems(IndexScanDesc scan, Datum value)
{
	PgturbohybridGraphScanOpaque so = (PgturbohybridGraphScanOpaque) scan->opaque;
	Relation	index = scan->indexRelation;
	BlockNumber blkno = PGTURBOHYBRID_GRAPH_HEAD_BLKNO;
	List	   *items = NIL;
	char	   *base = NULL;

	while (BlockNumberIsValid(blkno))
	{
		Buffer		buf;
		Page		page;
		BlockNumber nextblkno;
		OffsetNumber offno;
		OffsetNumber maxoffno;

		CHECK_FOR_INTERRUPTS();

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		maxoffno = PageGetMaxOffsetNumber(page);

		for (offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
		{
			ItemId		iid = PageGetItemId(page, offno);
			PgturbohybridGraphElementTuple etup;
			PgturbohybridGraphElement element;
			PgturbohybridGraphSearchCandidate *sc;

			if (!ItemIdIsUsed(iid))
				continue;

			etup = (PgturbohybridGraphElementTuple) PageGetItem(page, iid);

			if (!PgturbohybridGraphIsElementTuple(etup) || etup->deleted ||
				!ItemPointerIsValid(&etup->heaptids[0]))
				continue;

			element = PgturbohybridGraphInitElementFromBlock(blkno, offno);
			PgturbohybridGraphLoadElementFromTuple(element, etup, true, false);

			sc = palloc(sizeof(PgturbohybridGraphSearchCandidate));
			PgturbohybridGraphPtrStore(base, sc->element, element);
			sc->distance = GetFlatDistance(value, etup, &so->support);

			items = lappend(items, sc);
			so->tuples++;
		}

		nextblkno = PgturbohybridGraphPageGetOpaque(page)->nextblkno;
		UnlockReleaseBuffer(buf);
		blkno = nextblkno;
	}

	list_sort(items, CompareSearchCandidateDistances);

	return items;
}

/*
 * Algorithm 5 from paper.
 *
 * Legacy graph_hnsw scan entry: drives PgturbohybridGraphSearchLayer over
 * full-vector element tuples.  This is NOT the native scan path -- a native
 * (quantized graph) index is dispatched to tqgraphgettuple in
 * pgturbohybridgettuple before pgturbohybrid_graph_get_tuple ever selects this
 * routine, and a flat index uses GetFlatScanItems.  Reached only by the
 * dormant legacy element-tuple storage; native scans use
 * tqgraphgettuple -> PgturbohybridGraphCollectResults.
 */
static List *
GetScanItems(IndexScanDesc scan, Datum value)
{
	PgturbohybridGraphScanOpaque so = (PgturbohybridGraphScanOpaque) scan->opaque;
	Relation	index = scan->indexRelation;
	PgturbohybridGraphSupport *support = &so->support;
	List	   *ep;
	List	   *w;
	int			m;
	PgturbohybridGraphElement entryPoint;
	char	   *base = NULL;
	PgturbohybridGraphQuery  *q = &so->q;
	int			searchEf;
	int64		beforeTuples;
	int64		afterTuples;
	List	   *items;

	/* Get m and entry point */
	PgturbohybridGraphGetMetaPageInfo(index, &m, &entryPoint);

	q->value = value;
	so->m = m;
	if (so->pgturbohybridGraphScan)
		PgturbohybridGraphPrepareTqQuery(index, support, value, &so->tq);

	if (entryPoint == NULL)
		return NIL;

	so->graphEntryPointCount = 1;
	ep = list_make1(PgturbohybridGraphEntryCandidate(base, entryPoint, q, index, support, false));

	for (int lc = entryPoint->level; lc >= 1; lc--)
	{
		w = PgturbohybridGraphSearchLayer(base, q, ep, 1, lc, index, support, m, false, NULL, NULL, NULL, true, NULL, -1, so->pgturbohybridGraphScan ? &so->graphScoredCodes : NULL, so->pgturbohybridGraphScan ? &so->tq : NULL);
		ep = w;
	}

	searchEf = GetInitialIterativeBatchSize(so);
	if (so->pgturbohybridGraphScan)
	{
		/*
		 * Keep graph_oversampling on traversal breadth as well as the final
		 * exact rescore band. Half-width traversal drops true neighbors on
		 * the smoke workload even with near-threshold exact refinement.
		 */
		searchEf = Min(searchEf * Max(so->graphOversampling, 1) * 2,
					   pgturbohybrid_max_scan_tuples);
	}

	beforeTuples = so->tuples;
	items = PgturbohybridGraphSearchLayer(base, q, ep, searchEf, 0, index, support, m, false, NULL, &so->v, pgturbohybrid_iterative_scan != PGTURBOHYBRID_GRAPH_ITERATIVE_SCAN_OFF ? &so->discarded : NULL, true, &so->tuples, -1, so->pgturbohybridGraphScan ? &so->graphScoredCodes : NULL, so->pgturbohybridGraphScan ? &so->tq : NULL);
	afterTuples = so->tuples;
	RescoreScanItems(scan, items);

	RecordGraphScanBatchStats(so, beforeTuples, afterTuples, items);

	return items;
}

/*
 * Resume scan at ground level with discarded candidates
 */
static List *
ResumeScanItems(IndexScanDesc scan)
{
	PgturbohybridGraphScanOpaque so = (PgturbohybridGraphScanOpaque) scan->opaque;
	Relation	index = scan->indexRelation;
	List	   *ep = NIL;
	char	   *base = NULL;
	int			batch_size = so->efSearch;
	int64		beforeTuples;
	int64		afterTuples;
	List	   *items;

	if (pairingheap_is_empty(so->discarded))
		return NIL;

	/* Get next batch of candidates */
	for (int i = 0; i < batch_size; i++)
	{
		PgturbohybridGraphSearchCandidate *sc;

		if (pairingheap_is_empty(so->discarded))
			break;

		sc = PgturbohybridGraphGetSearchCandidate(w_node, pairingheap_remove_first(so->discarded));

		ep = lappend(ep, sc);
	}

	beforeTuples = so->tuples;
	items = PgturbohybridGraphSearchLayer(base, &so->q, ep, batch_size, 0, index, &so->support, so->m, false, NULL, &so->v, &so->discarded, false, &so->tuples, -1, so->pgturbohybridGraphScan ? &so->graphScoredCodes : NULL, so->pgturbohybridGraphScan ? &so->tq : NULL);
	afterTuples = so->tuples;
	RescoreScanItems(scan, items);

	RecordGraphScanBatchStats(so, beforeTuples, afterTuples, items);

	return items;
}

/*
 * Get scan value
 */
static Datum
GetScanValue(IndexScanDesc scan)
{
	PgturbohybridGraphScanOpaque so = (PgturbohybridGraphScanOpaque) scan->opaque;
	Datum		value;

	if (scan->orderByData->sk_flags & SK_ISNULL)
		value = PointerGetDatum(NULL);
	else
	{
		value = scan->orderByData->sk_argument;

		/* Value should not be compressed or toasted */
		Assert(!VARATT_IS_COMPRESSED(DatumGetPointer(value)));
		Assert(!VARATT_IS_EXTENDED(DatumGetPointer(value)));

		/* Normalize if needed */
		if (so->support.normprocinfo != NULL)
		{
			if (so->typeInfo->normalize == pgturbohybrid_l2_normalize)
				value = PointerGetDatum(PgturbohybridL2NormalizeFast((Vector *) DatumGetPointer(value)));
			else
				value = PgturbohybridGraphNormValue(so->typeInfo, so->support.collation, value);
		}
	}

	return value;
}

static pg_noinline List *
PgturbohybridGraphGetInitialScanItemsLocked(IndexScanDesc scan, Datum value,
											bool flatScan)
{
	List	   *items;
	bool		scanLockHeld = false;
	PgturbohybridGraphScanOpaque so =
		(PgturbohybridGraphScanOpaque) scan->opaque;
	instr_time	lockStart;

	/*
	 * Get a shared lock. This allows vacuum to ensure no in-flight scans
	 * before marking tuples as deleted.
	 */
	INSTR_TIME_SET_CURRENT(lockStart);
	LockPage(scan->indexRelation, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ShareLock);
	if (so != NULL)
		so->graphScanLockWaitUs +=
			PgturbohybridGraphScanElapsedUs(lockStart);
	scanLockHeld = true;

	PG_TRY();
	{
		if (flatScan)
			items = GetFlatScanItems(scan, value);
		else
			items = GetScanItems(scan, value);
	}
	PG_CATCH();
	{
		if (scanLockHeld)
			UnlockPage(scan->indexRelation, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ShareLock);
		PG_RE_THROW();
	}
	PG_END_TRY();

	UnlockPage(scan->indexRelation, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ShareLock);

	return items;
}

static pg_noinline List *
PgturbohybridGraphResumeScanItemsLocked(IndexScanDesc scan)
{
	List	   *items;
	bool		scanLockHeld = false;
	PgturbohybridGraphScanOpaque so =
		(PgturbohybridGraphScanOpaque) scan->opaque;
	instr_time	lockStart;

	/*
	 * Locking ensures when neighbors are read, the elements they reference will
	 * not be deleted (and replaced) during the iteration.
	 *
	 * Elements loaded into memory on previous iterations may have been deleted
	 * (and replaced), so when reading neighbors, the element version must be
	 * checked.
	 */
	INSTR_TIME_SET_CURRENT(lockStart);
	LockPage(scan->indexRelation, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ShareLock);
	if (so != NULL)
		so->graphScanLockWaitUs +=
			PgturbohybridGraphScanElapsedUs(lockStart);
	scanLockHeld = true;

	PG_TRY();
	{
		items = ResumeScanItems(scan);
	}
	PG_CATCH();
	{
		if (scanLockHeld)
			UnlockPage(scan->indexRelation, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ShareLock);
		PG_RE_THROW();
	}
	PG_END_TRY();

	UnlockPage(scan->indexRelation, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ShareLock);

	return items;
}

#if defined(PGTURBOHYBRID_GRAPH_MEMORY)
/*
 * Show memory usage
 */
static void
ShowMemoryUsage(PgturbohybridGraphScanOpaque so)
{
	elog(INFO, "memory: %zu KB, tuples: " INT64_FORMAT, MemoryContextMemAllocated(so->tmpCtx, false) / 1024, so->tuples);
}
#endif

/*
 * Prepare for an index scan
 */
IndexScanDesc
pgturbohybrid_graph_begin_scan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	PgturbohybridGraphScanOpaque so;
	double		maxMemory;

	scan = RelationGetIndexScan(index, nkeys, norderbys);

	so = (PgturbohybridGraphScanOpaque) palloc0(sizeof(PgturbohybridGraphScanOpaqueData));
	so->typeInfo = PgturbohybridGraphGetTypeInfo(index);

	/* Set support functions */
	PgturbohybridGraphInitSupport(&so->support, index);

	/*
	 * Use a lower max allocation size than default to allow scanning more
	 * tuples for iterative search before exceeding work_mem
	 */
	so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
									   "PgturbohybridGraph scan temporary context",
									   0, 8 * 1024, 256 * 1024);

	/* Calculate max memory */
	/* Add 256 extra bytes to fill last block when close */
	maxMemory = (double) work_mem * pgturbohybrid_scan_mem_multiplier * 1024.0 + 256;
	so->maxMemory = Min(maxMemory, (double) (SIZE_MAX / 2));
	so->first = true;
	so->efSearch = PgturbohybridGraphGetEfSearch(index);
	so->graphOversampling = PgturbohybridGraphGetGraphOversampling(index);
	so->graphRescoreBand = PgturbohybridGraphGetGraphRescoreBand(index);
	so->graphExactCache = PgturbohybridGraphGetGraphExactCache(index);
	so->graphStorageKind = PgturbohybridGraphGetMetaPageStorageKind(index);
	so->pgturbohybridGraphScan = PgturbohybridGraphUseTqGraph(index);
	so->pgturbohybridFlatScan = PgturbohybridGraphUseTqFlat(index);
	so->previousDistance = -get_float8_infinity();

	scan->opaque = so;

	return scan;
}

IndexScanDesc
pgturbohybridbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	PgturbohybridGraphScanOpaque so;

	/*
	 * Sparse-primary: no dense graph, so allocate a minimal scan
	 * opaque (scan context + defaults) without the vector type info/support the
	 * native and flat paths require.  The sparse scan runs through
	 * PgturbohybridCollectScanResults' sole-sparse path, which only needs
	 * so->tmpCtx and the hybrid state amrescan attaches.
	 */
	if (PgturbohybridSparseIsPrimary(index))
	{
		scan = RelationGetIndexScan(index, nkeys, norderbys);
		so = palloc0(sizeof(PgturbohybridGraphScanOpaqueData));
		so->typeInfo = NULL;
		so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
										   "pgturbohybrid sparse-primary scan context",
										   0, 8 * 1024, 256 * 1024);
		so->efSearch = PgturbohybridGraphGetEfSearch(index);
		so->graphStorageKind = PGTURBOHYBRID_GRAPH_STORAGE_QUANT_GRAPH_NATIVE;
		so->first = true;
		so->tqHybridState = NULL;
		so->previousDistance = -get_float8_infinity();
		scan->opaque = so;
		return scan;
	}

	if (PgturbohybridGraphUseTqNativeGraph(index))
		return tqgraphbeginscan(index, nkeys, norderbys);

	if (!PgturbohybridGraphUseTqFlat(index))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("turbohybrid requires a native graph-compatible opclass"),
				 errhint("Use a turbohybrid vector opclass.")));

	PgturbohybridGraphSetForcepgturbohybridIndex(true);
	PG_TRY();
	{
		scan = pgturbohybrid_graph_begin_scan(index, nkeys, norderbys);
	}
	PG_CATCH();
	{
		PgturbohybridGraphSetForcepgturbohybridIndex(false);
		PG_RE_THROW();
	}
	PG_END_TRY();
	PgturbohybridGraphSetForcepgturbohybridIndex(false);

	so = (PgturbohybridGraphScanOpaque) scan->opaque;
	so->pgturbohybridGraphScan = PgturbohybridGraphUseTqGraph(index);
	so->pgturbohybridFlatScan = PgturbohybridGraphUseTqFlat(index);

	return scan;
}

/*
 * Start or restart an index scan
 */
void
pgturbohybrid_graph_rescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	PgturbohybridGraphScanOpaque so = (PgturbohybridGraphScanOpaque) scan->opaque;

	so->first = true;
	/* v and discarded are allocated in tmpCtx */
	so->v.tids = NULL;
	so->discarded = NULL;
	so->hasTupleTargetRows = false;
	so->hasEstimatedFilterSelectivity = false;
	so->hasInitialEffectiveEfSearch = false;
	so->returnedRows = 0;
	so->tupleTargetRows = -1;
	so->estimatedFilterSelectivity = -1.0;
	so->efSearch = PgturbohybridGraphGetEfSearch(scan->indexRelation);
	so->graphOversampling = PgturbohybridGraphGetGraphOversampling(scan->indexRelation);
	so->graphRescoreBand = PgturbohybridGraphGetGraphRescoreBand(scan->indexRelation);
	so->graphStorageKind = PgturbohybridGraphGetMetaPageStorageKind(scan->indexRelation);
	so->pgturbohybridGraphScan = PgturbohybridGraphUseTqGraph(scan->indexRelation);
	so->pgturbohybridFlatScan = PgturbohybridGraphUseTqFlat(scan->indexRelation);
	so->initialEffectiveEfSearch = so->efSearch;
	so->tuples = 0;
	so->graphVisitedNodes = 0;
	so->graphScoredCodes = 0;
	so->graphCandidateCount = 0;
	so->graphRescoreCount = 0;
	so->graphRescorePages = 0;
	so->graphEntryPointCount = 0;
	memset(&so->tq, 0, sizeof(PgturbohybridGraphTqQuery));
	so->previousDistance = -get_float8_infinity();
	MemoryContextReset(so->tmpCtx);

	if (keys && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));

	if (orderbys && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys, scan->numberOfOrderBys * sizeof(ScanKeyData));
}

void
pgturbohybridrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	PgturbohybridGraphScanOpaque so;

	if (PgturbohybridSparseIsPrimary(scan->indexRelation))
	{
		so = (PgturbohybridGraphScanOpaque) scan->opaque;
		so->first = true;
		so->returnedRows = 0;
		so->previousDistance = -get_float8_infinity();
		MemoryContextReset(so->tmpCtx);
		if (keys && scan->numberOfKeys > 0)
			memmove(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));
		if (orderbys && scan->numberOfOrderBys > 0)
			memmove(scan->orderByData, orderbys,
					scan->numberOfOrderBys * sizeof(ScanKeyData));
		return;
	}

	if (PgturbohybridGraphUseTqNativeGraph(scan->indexRelation))
	{
		tqgraphrescan(scan, keys, nkeys, orderbys, norderbys);
		return;
	}

	if (!PgturbohybridGraphUseTqFlat(scan->indexRelation))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("turbohybrid requires a native graph-compatible opclass"),
				 errhint("Use a turbohybrid vector opclass.")));

	PgturbohybridGraphSetForcepgturbohybridIndex(true);
	PG_TRY();
	{
		pgturbohybrid_graph_rescan(scan, keys, nkeys, orderbys, norderbys);
	}
	PG_CATCH();
	{
		PgturbohybridGraphSetForcepgturbohybridIndex(false);
		PG_RE_THROW();
	}
	PG_END_TRY();
	PgturbohybridGraphSetForcepgturbohybridIndex(false);

	so = (PgturbohybridGraphScanOpaque) scan->opaque;
	so->pgturbohybridGraphScan = PgturbohybridGraphUseTqGraph(scan->indexRelation);
	so->pgturbohybridFlatScan = PgturbohybridGraphUseTqFlat(scan->indexRelation);
}

/*
 * Fetch the next tuple in the given scan
 */
bool
pgturbohybrid_graph_get_tuple(IndexScanDesc scan, ScanDirection dir)
{
	PgturbohybridGraphScanOpaque so = (PgturbohybridGraphScanOpaque) scan->opaque;
	MemoryContext oldCtx;

	Assert(so != NULL);
	oldCtx = MemoryContextSwitchTo(so->tmpCtx);

	/*
	 * Index can be used to scan backward, but Postgres doesn't support
	 * backward scan on operators
	 */
	Assert(ScanDirectionIsForward(dir));

	if (so->first)
	{
		Datum		value;

		/* Count index scan for stats */
		pgstat_count_index_scan(scan->indexRelation);
#if PG_VERSION_NUM >= 180000
		if (scan->instrument)
			scan->instrument->nsearches++;
#endif

		/* Safety check */
		if (scan->orderByData == NULL)
			elog(ERROR, "cannot scan pgturbohybrid index without order");

		/* Requires MVCC-compliant snapshot as not able to maintain a pin */
		/* https://www.postgresql.org/docs/current/index-locking.html */
		if (!IsMVCCSnapshot(scan->xs_snapshot))
			elog(ERROR, "non-MVCC snapshots are not supported with pgturbohybrid");

		/* Get scan value */
		value = GetScanValue(scan);
		if (so->pgturbohybridFlatScan)
			PgturbohybridGraphRecordFlatScanStats();
		else if (!so->pgturbohybridGraphScan)
			PgturbohybridGraphRecordNonGraphScanStats();

		so->w = PgturbohybridGraphGetInitialScanItemsLocked(scan, value,
															so->pgturbohybridFlatScan);

		so->first = false;

#if defined(PGTURBOHYBRID_GRAPH_MEMORY)
		ShowMemoryUsage(so);
#endif
	}

	for (;;)
	{
		char	   *base = NULL;
		PgturbohybridGraphSearchCandidate *sc;
		PgturbohybridGraphElement element;
		ItemPointer heaptid;

		if (list_length(so->w) == 0)
		{
			if (so->pgturbohybridFlatScan)
				break;

			if (pgturbohybrid_iterative_scan == PGTURBOHYBRID_GRAPH_ITERATIVE_SCAN_OFF)
				break;

			/* Empty index */
			if (so->discarded == NULL)
				break;

			/* Reached max number of tuples or memory limit */
			if (so->tuples >= pgturbohybrid_max_scan_tuples || MemoryContextMemAllocated(so->tmpCtx, false) > so->maxMemory)
			{
				if (pairingheap_is_empty(so->discarded))
					break;

				/* Return remaining tuples */
				so->w = lappend(so->w, PgturbohybridGraphGetSearchCandidate(w_node, pairingheap_remove_first(so->discarded)));
			}
			else
			{
				so->w = PgturbohybridGraphResumeScanItemsLocked(scan);

#if defined(PGTURBOHYBRID_GRAPH_MEMORY)
				ShowMemoryUsage(so);
#endif
			}

			if (list_length(so->w) == 0)
				break;
		}

		sc = llast(so->w);
		element = PgturbohybridGraphPtrAccess(base, sc->element);

		/* Move to next element if no valid heap TIDs */
		if (element->heaptidsLength == 0)
		{
			so->w = list_delete_last(so->w);

			/* Mark memory as free for next iteration */
			if (pgturbohybrid_iterative_scan != PGTURBOHYBRID_GRAPH_ITERATIVE_SCAN_OFF)
			{
				pfree(element);
				pfree(sc);
			}

			continue;
		}

		heaptid = &element->heaptids[--element->heaptidsLength];

		if (pgturbohybrid_iterative_scan == PGTURBOHYBRID_GRAPH_ITERATIVE_SCAN_STRICT)
		{
			if (sc->distance < so->previousDistance)
				continue;

			so->previousDistance = sc->distance;
		}

		MemoryContextSwitchTo(oldCtx);

		scan->xs_heaptid = *heaptid;
		scan->xs_recheck = false;
		scan->xs_recheckorderby = false;
		return true;
	}

	MemoryContextSwitchTo(oldCtx);
	return false;
}

bool
pgturbohybridgettuple(IndexScanDesc scan, ScanDirection dir)
{
	PgturbohybridGraphScanOpaque so = (PgturbohybridGraphScanOpaque) scan->opaque;
	bool		result;

	/*
	 * Sparse-primary scans are driven entirely by the sole-sparse path in
	 * pgturbohybridamgettuple (via the attached hybrid state).  Reaching here
	 * means there was no sparse ORDER BY query to execute, so there are no rows.
	 */
	if (PgturbohybridSparseIsPrimary(scan->indexRelation))
		return false;

	if (PgturbohybridGraphUseTqNativeGraph(scan->indexRelation))
	{
		return tqgraphgettuple(scan, dir);
	}

	if (!PgturbohybridGraphUseTqFlat(scan->indexRelation))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("turbohybrid requires a native graph-compatible opclass"),
				 errhint("Use a turbohybrid vector opclass.")));

	PgturbohybridGraphSetForcepgturbohybridIndex(true);
	PG_TRY();
	{
		if (so != NULL && !so->pgturbohybridGraphScan && PgturbohybridGraphUseTqGraph(scan->indexRelation))
		{
			so->pgturbohybridGraphScan = true;
			so->pgturbohybridFlatScan = false;
			so->efSearch = PgturbohybridGraphGetEfSearch(scan->indexRelation);
			so->graphOversampling = PgturbohybridGraphGetGraphOversampling(scan->indexRelation);
			so->graphRescoreBand = PgturbohybridGraphGetGraphRescoreBand(scan->indexRelation);
			so->graphStorageKind = PgturbohybridGraphGetMetaPageStorageKind(scan->indexRelation);
			so->initialEffectiveEfSearch = so->efSearch;
		}
		else if (so != NULL)
			so->pgturbohybridFlatScan = PgturbohybridGraphUseTqFlat(scan->indexRelation);

		result = pgturbohybrid_graph_get_tuple(scan, dir);
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

/*
 * End a scan and release resources
 */
void
pgturbohybrid_graph_end_scan(IndexScanDesc scan)
{
	PgturbohybridGraphScanOpaque so = (PgturbohybridGraphScanOpaque) scan->opaque;

	MemoryContextDelete(so->tmpCtx);

	pfree(so);
	scan->opaque = NULL;
}

void
pgturbohybridendscan(IndexScanDesc scan)
{
	if (PgturbohybridSparseIsPrimary(scan->indexRelation))
	{
		PgturbohybridGraphScanOpaque so = (PgturbohybridGraphScanOpaque) scan->opaque;

		if (so != NULL)
		{
			MemoryContextDelete(so->tmpCtx);
			pfree(so);
		}
		scan->opaque = NULL;
		return;
	}

	if (PgturbohybridGraphUseTqNativeGraph(scan->indexRelation))
	{
		tqgraphendscan(scan);
		return;
	}

	if (!PgturbohybridGraphUseTqFlat(scan->indexRelation))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("turbohybrid requires a native graph-compatible opclass"),
				 errhint("Use a turbohybrid vector opclass.")));

	pgturbohybrid_graph_end_scan(scan);
}
