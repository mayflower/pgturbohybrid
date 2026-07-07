#include "postgres.h"

#include "access/genam.h"
#include "access/generic_xlog.h"
#include "commands/vacuum.h"
#include "pgturbohybrid.h"
#include "nodes/pg_list.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/memutils.h"
#include "utils/rel.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

#if PG_VERSION_NUM >= 180000
#define vacuum_delay_point() vacuum_delay_point(false)
#endif

/*
 * Check if deleted list contains an index TID
 */
static bool
DeletedContains(pgturbohybrid_tidhash_hash * deleted, ItemPointer indextid)
{
	return pgturbohybrid_tidhash_lookup(deleted, *indextid) != NULL;
}

/*
 * Remove deleted heap TIDs
 *
 * OK to remove for entry point, since always considered for searches and inserts
 */
static void
RemoveHeapTids(PgturbohybridGraphVacuumState * vacuumstate)
{
	BlockNumber blkno = PGTURBOHYBRID_GRAPH_HEAD_BLKNO;
	PgturbohybridGraphElement highestPoint = &vacuumstate->highestPoint;
	Relation	index = vacuumstate->index;
	BufferAccessStrategy bas = vacuumstate->bas;
	PgturbohybridGraphElement entryPoint = PgturbohybridGraphGetEntryPoint(vacuumstate->index);
	IndexBulkDeleteResult *stats = vacuumstate->stats;

	/* Store separately since highestPoint.level is uint8 */
	int			highestLevel = -1;

	/* Initialize highest point */
	highestPoint->blkno = InvalidBlockNumber;
	highestPoint->offno = InvalidOffsetNumber;

	while (BlockNumberIsValid(blkno))
	{
		BlockNumber currentBlkno = blkno;
		Buffer		buf;
		Page		page;
		GenericXLogState *state;
		OffsetNumber offno;
		OffsetNumber maxoffno;
		bool		updated = false;

		vacuum_delay_point();

		buf = ReadBufferExtended(index, MAIN_FORKNUM, blkno, RBM_NORMAL, bas);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
		state = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(state, buf, 0);
		maxoffno = PageGetMaxOffsetNumber(page);

		/* Iterate over nodes */
		for (offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
		{
			PgturbohybridGraphElementTuple etup = (PgturbohybridGraphElementTuple) PageGetItem(page, PageGetItemId(page, offno));
			int			idx = 0;
			bool		itemUpdated = false;

			/* Skip neighbor tuples */
			if (!PgturbohybridGraphIsElementTuple(etup))
				continue;

			if (ItemPointerIsValid(&etup->heaptids[0]))
			{
				for (int i = 0; i < PGTURBOHYBRID_GRAPH_HEAPTIDS; i++)
				{
					/* Stop at first unused */
					if (!ItemPointerIsValid(&etup->heaptids[i]))
						break;

					if (vacuumstate->callback(&etup->heaptids[i], vacuumstate->callback_state))
					{
						itemUpdated = true;
						stats->tuples_removed++;
					}
					else
					{
						/* Move to front of list */
						etup->heaptids[idx++] = etup->heaptids[i];
						stats->num_index_tuples++;
					}
				}

				if (itemUpdated)
				{
					/* Mark rest as invalid */
					for (int i = idx; i < PGTURBOHYBRID_GRAPH_HEAPTIDS; i++)
						ItemPointerSetInvalid(&etup->heaptids[i]);

					updated = true;
					PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_VACUUM_DELETE);
				}
			}

			if (!ItemPointerIsValid(&etup->heaptids[0]))
			{
				ItemPointerData ip;
				bool		found;

				/* Add to deleted list */
				ItemPointerSet(&ip, blkno, offno);

				pgturbohybrid_tidhash_insert(vacuumstate->deleted, ip, &found);
				Assert(!found);
			}
			else if (etup->level > highestLevel && !(entryPoint != NULL && blkno == entryPoint->blkno && offno == entryPoint->offno))
			{
				/* Keep track of highest non-entry point */
				highestPoint->blkno = blkno;
				highestPoint->offno = offno;
				highestPoint->level = etup->level;
				highestLevel = etup->level;
			}
		}

		blkno = PgturbohybridGraphPageGetOpaque(page)->nextblkno;

		if (updated)
			GenericXLogFinish(state);
		else
			GenericXLogAbort(state);

		UnlockReleaseBuffer(buf);
		if (updated)
			PgturbohybridGraphLogGraphWalRecord(index, MAIN_FORKNUM, currentBlkno, PGTURBOHYBRID_GRAPH_GRAPH_OP_VACUUM_DELETE);
	}
}

/*
 * Check for deleted neighbors
 */
static bool
NeedsUpdated(PgturbohybridGraphVacuumState * vacuumstate, PgturbohybridGraphElement element)
{
	Relation	index = vacuumstate->index;
	BufferAccessStrategy bas = vacuumstate->bas;
	Buffer		buf;
	Page		page;
	PgturbohybridGraphNeighborTuple ntup;
	bool		needsUpdated = false;

	buf = ReadBufferExtended(index, MAIN_FORKNUM, element->neighborPage, RBM_NORMAL, bas);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	ntup = (PgturbohybridGraphNeighborTuple) PageGetItem(page, PageGetItemId(page, element->neighborOffno));

	Assert(PgturbohybridGraphIsNeighborTuple(ntup));

	/* Check neighbors */
	for (int i = 0; i < ntup->count; i++)
	{
		ItemPointer indextid = &ntup->indextids[i];

		if (!ItemPointerIsValid(indextid))
			continue;

		/* Check if in deleted list */
		if (DeletedContains(vacuumstate->deleted, indextid))
		{
			needsUpdated = true;
			break;
		}
	}

	/* Also update if layer 0 is not full */
	/* This could indicate too many candidates being deleted during insert */
	if (!needsUpdated)
	{
		/*
		 * count is read from the page, so guard at runtime before indexing
		 * indextids[count - 1] -- an Assert compiles out under -DNDEBUG and
		 * would leave an indextids[-1] read.  A zero-count neighbor tuple is
		 * itself degenerate and worth repairing.
		 */
		needsUpdated = ntup->count == 0 ||
			!ItemPointerIsValid(&ntup->indextids[ntup->count - 1]);
	}

	UnlockReleaseBuffer(buf);

	return needsUpdated;
}

/*
 * Repair graph for a single element
 */
static void
RepairGraphElement(PgturbohybridGraphVacuumState * vacuumstate, PgturbohybridGraphElement element, PgturbohybridGraphElement entryPoint)
{
	Relation	index = vacuumstate->index;
	PgturbohybridGraphSupport *support = &vacuumstate->support;
	Buffer		buf;
	Page		page;
	GenericXLogState *state;
	int			m = vacuumstate->m;
	int			efConstruction = vacuumstate->efConstruction;
	BufferAccessStrategy bas = vacuumstate->bas;
	PgturbohybridGraphNeighborTuple ntup = vacuumstate->ntup;
	Size		ntupSize = PGTURBOHYBRID_GRAPH_NEIGHBOR_TUPLE_SIZE(element->level, m);
	BlockNumber blkno;
	char	   *base = NULL;

	/* Skip if element is entry point */
	if (entryPoint != NULL && element->blkno == entryPoint->blkno && element->offno == entryPoint->offno)
		return;

	/* Init fields */
	PgturbohybridGraphInitNeighbors(base, element, m, NULL);
	element->heaptidsLength = 0;

	/* Find neighbors for element, skipping itself */
	PgturbohybridGraphFindElementNeighbors(base, element, entryPoint, index, support, m, efConstruction, true);

	/* Zero memory for each element */
	MemSet(ntup, 0, PGTURBOHYBRID_GRAPH_TUPLE_ALLOC_SIZE);

	/* Update neighbor tuple */
	/* Do this before getting page to minimize locking */
	PgturbohybridGraphSetNeighborTuple(base, ntup, element, m);

	/* Get neighbor page */
	buf = ReadBufferExtended(index, MAIN_FORKNUM, element->neighborPage, RBM_NORMAL, bas);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	state = GenericXLogStart(index);
	page = GenericXLogRegisterBuffer(state, buf, 0);

	/* Overwrite tuple */
	if (!PageIndexTupleOverwrite(page, element->neighborOffno, (Item) ntup, ntupSize))
		elog(ERROR, "failed to add index item to \"%s\"", RelationGetRelationName(index));
	PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_VACUUM_REPAIR);
	blkno = BufferGetBlockNumber(buf);

	/* Commit */
	GenericXLogFinish(state);
	UnlockReleaseBuffer(buf);
	PgturbohybridGraphLogGraphWalRecord(index, MAIN_FORKNUM, blkno, PGTURBOHYBRID_GRAPH_GRAPH_OP_VACUUM_REPAIR);

	/* Update neighbors */
	PgturbohybridGraphUpdateNeighborsOnDisk(index, support, element, m, true, false);
}

/*
 * Repair graph entry point
 */
static void
RepairGraphEntryPoint(PgturbohybridGraphVacuumState * vacuumstate)
{
	Relation	index = vacuumstate->index;
	PgturbohybridGraphSupport *support = &vacuumstate->support;
	PgturbohybridGraphElement highestPoint = &vacuumstate->highestPoint;
	PgturbohybridGraphElement entryPoint;
	MemoryContext oldCtx = MemoryContextSwitchTo(vacuumstate->tmpCtx);

	if (!BlockNumberIsValid(highestPoint->blkno))
		highestPoint = NULL;

	/*
	 * Repair graph for highest non-entry point. Highest point may be outdated
	 * due to inserts that happen during and after RemoveHeapTids.
	 */
	if (highestPoint != NULL)
	{
		/* Get a shared lock */
		LockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, ShareLock);

		/* Load element */
		PgturbohybridGraphLoadElement(highestPoint, NULL, NULL, index, support, true, NULL);

		/* Repair if needed */
		if (NeedsUpdated(vacuumstate, highestPoint))
			RepairGraphElement(vacuumstate, highestPoint, PgturbohybridGraphGetEntryPoint(index));

		/* Release lock */
		UnlockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, ShareLock);
	}

	/* Prevent concurrent inserts when possibly updating entry point */
	LockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, ExclusiveLock);

	/* Get latest entry point */
	entryPoint = PgturbohybridGraphGetEntryPoint(index);

	if (entryPoint != NULL)
	{
		ItemPointerData epData;

		ItemPointerSet(&epData, entryPoint->blkno, entryPoint->offno);

		if (DeletedContains(vacuumstate->deleted, &epData))
		{
			/*
			 * Replace the entry point with the highest point. If highest
			 * point is outdated and empty, the entry point will be empty
			 * until an element is repaired.
			 */
			PgturbohybridGraphUpdateMetaPage(index, PGTURBOHYBRID_GRAPH_UPDATE_ENTRY_ALWAYS, highestPoint, InvalidBlockNumber, MAIN_FORKNUM, false);
		}
		else
		{
			/*
			 * Repair the entry point with the highest point. If highest point
			 * is outdated, this can remove connections at higher levels in
			 * the graph until they are repaired, but this should be fine.
			 */
			PgturbohybridGraphLoadElement(entryPoint, NULL, NULL, index, support, true, NULL);

			if (NeedsUpdated(vacuumstate, entryPoint))
			{
				/* Reset neighbors from previous update */
				if (highestPoint != NULL)
					PgturbohybridGraphPtrStore((char *) NULL, highestPoint->neighbors, (PgturbohybridGraphNeighborArrayPtr *) NULL);

				RepairGraphElement(vacuumstate, entryPoint, highestPoint);
			}
		}
	}

	/* Release lock */
	UnlockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, ExclusiveLock);

	/* Reset memory context */
	MemoryContextSwitchTo(oldCtx);
	MemoryContextReset(vacuumstate->tmpCtx);
}

/*
 * Repair graph for all elements
 */
static void
RepairGraph(PgturbohybridGraphVacuumState * vacuumstate)
{
	Relation	index = vacuumstate->index;
	BufferAccessStrategy bas = vacuumstate->bas;
	BlockNumber blkno = PGTURBOHYBRID_GRAPH_HEAD_BLKNO;

	/*
	 * Wait for inserts to complete. Inserts before this point may have
	 * neighbors about to be deleted. Inserts after this point will not.
	 */
	LockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, ExclusiveLock);
	UnlockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, ExclusiveLock);

	/* Repair entry point first */
	RepairGraphEntryPoint(vacuumstate);

	while (BlockNumberIsValid(blkno))
	{
		Buffer		buf;
		Page		page;
		OffsetNumber offno;
		OffsetNumber maxoffno;
		List	   *elements = NIL;
		ListCell   *lc2;
		MemoryContext oldCtx;

		vacuum_delay_point();

		oldCtx = MemoryContextSwitchTo(vacuumstate->tmpCtx);

		buf = ReadBufferExtended(index, MAIN_FORKNUM, blkno, RBM_NORMAL, bas);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		maxoffno = PageGetMaxOffsetNumber(page);

		/* Load items into memory to minimize locking */
		for (offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
		{
			PgturbohybridGraphElementTuple etup = (PgturbohybridGraphElementTuple) PageGetItem(page, PageGetItemId(page, offno));
			PgturbohybridGraphElement element;

			/* Skip neighbor tuples */
			if (!PgturbohybridGraphIsElementTuple(etup))
				continue;

			/* Skip updating neighbors if being deleted */
			if (!ItemPointerIsValid(&etup->heaptids[0]))
				continue;

			/* Create an element */
			element = PgturbohybridGraphInitElementFromBlock(blkno, offno);
			PgturbohybridGraphLoadElementFromTuple(element, etup, false, true);

			elements = lappend(elements, element);
		}

		blkno = PgturbohybridGraphPageGetOpaque(page)->nextblkno;

		UnlockReleaseBuffer(buf);

		/* Update neighbor pages */
		foreach(lc2, elements)
		{
			PgturbohybridGraphElement element = (PgturbohybridGraphElement) lfirst(lc2);
			PgturbohybridGraphElement entryPoint;
			LOCKMODE	lockmode = ShareLock;

			/* Check if any neighbors point to deleted values */
			if (!NeedsUpdated(vacuumstate, element))
				continue;

			/* Get a shared lock */
			LockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, lockmode);

			/* Refresh entry point for each element */
			entryPoint = PgturbohybridGraphGetEntryPoint(index);

			/* Prevent concurrent inserts when likely updating entry point */
			if (entryPoint == NULL || element->level > entryPoint->level)
			{
				/* Release shared lock */
				UnlockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, lockmode);

				/* Get exclusive lock */
				lockmode = ExclusiveLock;
				LockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, lockmode);

				/* Get latest entry point after lock is acquired */
				entryPoint = PgturbohybridGraphGetEntryPoint(index);
			}

			/* Repair connections */
			RepairGraphElement(vacuumstate, element, entryPoint);

			/*
			 * Update metapage if needed. Should only happen if entry point
			 * was replaced and highest point was outdated.
			 */
			if (entryPoint == NULL || element->level > entryPoint->level)
				PgturbohybridGraphUpdateMetaPage(index, PGTURBOHYBRID_GRAPH_UPDATE_ENTRY_GREATER, element, InvalidBlockNumber, MAIN_FORKNUM, false);

			/* Release lock */
			UnlockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, lockmode);
		}

		/* Reset memory context */
		MemoryContextSwitchTo(oldCtx);
		MemoryContextReset(vacuumstate->tmpCtx);
	}
}

/*
 * Mark items as deleted
 */
static void
MarkDeleted(PgturbohybridGraphVacuumState * vacuumstate)
{
	BlockNumber blkno = PGTURBOHYBRID_GRAPH_HEAD_BLKNO;
	BlockNumber insertPage = InvalidBlockNumber;
	Relation	index = vacuumstate->index;
	BufferAccessStrategy bas = vacuumstate->bas;

	/*
	 * Wait for index scans to complete. Scans before this point may contain
	 * tuples about to be deleted. Scans after this point will not, since the
	 * graph has been repaired.
	 */
	LockPage(index, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ExclusiveLock);
	UnlockPage(index, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ExclusiveLock);

	while (BlockNumberIsValid(blkno))
	{
		Buffer		buf;
		Page		page;
		GenericXLogState *state;
		OffsetNumber offno;
		OffsetNumber maxoffno;

		vacuum_delay_point();

		buf = ReadBufferExtended(index, MAIN_FORKNUM, blkno, RBM_NORMAL, bas);

		/*
		 * ambulkdelete cannot delete entries from pages that are pinned by
		 * other backends
		 *
		 * https://www.postgresql.org/docs/current/index-locking.html
		 */
		LockBufferForCleanup(buf);

		state = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(state, buf, 0);
		maxoffno = PageGetMaxOffsetNumber(page);

		/* Update element and neighbors together */
		for (offno = FirstOffsetNumber; offno <= maxoffno; offno = OffsetNumberNext(offno))
		{
			PgturbohybridGraphElementTuple etup = (PgturbohybridGraphElementTuple) PageGetItem(page, PageGetItemId(page, offno));
			PgturbohybridGraphNeighborTuple ntup;
			Buffer		nbuf;
			Page		npage;
			BlockNumber neighborPage;
			OffsetNumber neighborOffno;

			/* Skip neighbor tuples */
			if (!PgturbohybridGraphIsElementTuple(etup))
				continue;

			/* Skip deleted tuples */
			if (etup->deleted)
			{
				/* Set to first free page */
				if (!BlockNumberIsValid(insertPage))
					insertPage = blkno;

				continue;
			}

			/* Skip live tuples */
			if (ItemPointerIsValid(&etup->heaptids[0]))
				continue;

			/* Get neighbor page */
			neighborPage = ItemPointerGetBlockNumber(&etup->neighbortid);
			neighborOffno = ItemPointerGetOffsetNumber(&etup->neighbortid);

			if (neighborPage == blkno)
			{
				nbuf = buf;
				npage = page;
			}
			else
			{
				nbuf = ReadBufferExtended(index, MAIN_FORKNUM, neighborPage, RBM_NORMAL, bas);
				LockBuffer(nbuf, BUFFER_LOCK_EXCLUSIVE);
				npage = GenericXLogRegisterBuffer(state, nbuf, 0);
			}

			ntup = (PgturbohybridGraphNeighborTuple) PageGetItem(npage, PageGetItemId(npage, neighborOffno));

			/* Overwrite element */
			/* Use memset instead of MemSet to keep clang-tidy happy */
			etup->deleted = 1;
			memset(&etup->data, 0, VARSIZE_ANY(&etup->data));

			/* Overwrite neighbors */
			for (int i = 0; i < ntup->count; i++)
				ItemPointerSetInvalid(&ntup->indextids[i]);

			/* Increment version */
			/* This is used to avoid incorrect reads for iterative scans */
			/* Reserve some bits for future use */
			etup->version++;
			if (etup->version > 15)
				etup->version = 1;
			ntup->version = etup->version;
			PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_VACUUM_DELETE);
			PgturbohybridGraphMarkPageGraphOp(npage, PGTURBOHYBRID_GRAPH_GRAPH_OP_VACUUM_DELETE);

			/*
			 * We modified the tuples in place, no need to call
			 * PageIndexTupleOverwrite
			 */

			/* Commit */
			GenericXLogFinish(state);
			if (nbuf != buf)
				UnlockReleaseBuffer(nbuf);
			PgturbohybridGraphLogGraphWalRecord(index, MAIN_FORKNUM, blkno, PGTURBOHYBRID_GRAPH_GRAPH_OP_VACUUM_DELETE);
			if (nbuf != buf)
				PgturbohybridGraphLogGraphWalRecord(index, MAIN_FORKNUM, neighborPage, PGTURBOHYBRID_GRAPH_GRAPH_OP_VACUUM_DELETE);

			/* Set to first free page */
			if (!BlockNumberIsValid(insertPage))
				insertPage = blkno;

			/* Prepare new xlog */
			state = GenericXLogStart(index);
			page = GenericXLogRegisterBuffer(state, buf, 0);
		}

		blkno = PgturbohybridGraphPageGetOpaque(page)->nextblkno;

		GenericXLogAbort(state);
		UnlockReleaseBuffer(buf);
	}

	/* Update insert page last, after everything has been marked as deleted */
	PgturbohybridGraphUpdateMetaPage(index, 0, NULL, insertPage, MAIN_FORKNUM, false);
}

/*
 * Initialize the vacuum state
 */
static void
InitVacuumState(PgturbohybridGraphVacuumState * vacuumstate, IndexVacuumInfo *info, IndexBulkDeleteResult *stats, IndexBulkDeleteCallback callback, void *callback_state)
{
	Relation	index = info->index;

	if (stats == NULL)
		stats = (IndexBulkDeleteResult *) palloc0(sizeof(IndexBulkDeleteResult));

	vacuumstate->index = index;
	vacuumstate->stats = stats;
	vacuumstate->callback = callback;
	vacuumstate->callback_state = callback_state;
	vacuumstate->efConstruction = PgturbohybridGraphGetEfConstruction(index);
	vacuumstate->bas = GetAccessStrategy(BAS_BULKREAD);
	vacuumstate->ntup = palloc0(PGTURBOHYBRID_GRAPH_TUPLE_ALLOC_SIZE);
	vacuumstate->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
												"PgturbohybridGraph vacuum temporary context",
												ALLOCSET_DEFAULT_SIZES);

	PgturbohybridGraphInitSupport(&vacuumstate->support, index);

	/* Get m from metapage */
	PgturbohybridGraphGetMetaPageInfo(index, &vacuumstate->m, NULL);

	/* Create hash table */
	vacuumstate->deleted = pgturbohybrid_tidhash_create(CurrentMemoryContext, 256, NULL);
}

/*
 * Free resources
 */
static void
FreeVacuumState(PgturbohybridGraphVacuumState * vacuumstate)
{
	pgturbohybrid_tidhash_destroy(vacuumstate->deleted);
	FreeAccessStrategy(vacuumstate->bas);
	pfree(vacuumstate->ntup);
	MemoryContextDelete(vacuumstate->tmpCtx);
}

/*
 * Bulk delete tuples from the index
 */
IndexBulkDeleteResult *
pgturbohybrid_graph_bulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats,
			   IndexBulkDeleteCallback callback, void *callback_state)
{
	PgturbohybridGraphVacuumState vacuumstate;

	InitVacuumState(&vacuumstate, info, stats, callback, callback_state);

	/* Pass 1: Remove heap TIDs */
	RemoveHeapTids(&vacuumstate);

	/* Pass 2: Repair graph */
	RepairGraph(&vacuumstate);

	/* Pass 3: Mark as deleted */
	MarkDeleted(&vacuumstate);

	FreeVacuumState(&vacuumstate);

	return vacuumstate.stats;
}

/*
 * Clean up after a VACUUM operation
 */
IndexBulkDeleteResult *
pgturbohybrid_graph_vacuum_cleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats)
{
	Relation	rel = info->index;

	if (info->analyze_only)
		return stats;

	/* stats is NULL if ambulkdelete not called */
	/* OK to return NULL if index not changed */
	if (stats == NULL)
		return NULL;

	stats->num_pages = RelationGetNumberOfBlocks(rel);

	return stats;
}
