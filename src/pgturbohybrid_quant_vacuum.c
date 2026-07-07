/*
 * pgturbohybrid_quant_vacuum.c
 *
 * Quantized-graph VACUUM phase for the turbohybrid AM: bulk-delete, adjacency
 * repair for dead nodes, vacuum-stats collection, and cleanup.  Extracted
 * verbatim from pgturbohybrid_quant.c (no behaviour change) to shrink that
 * translation unit; the entry points remain declared in pgturbohybrid.h /
 * pgturbohybrid_quant.h and are called from pgturbohybrid_am.c.
 *
 * (Distinct from pgturbohybrid_vacuum.c, the non-quant graph vacuum.)
 */
#include "postgres.h"

#include "access/genam.h"
#include "access/generic_xlog.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/rel.h"

#include "pgturbohybrid.h"
#include "pgturbohybrid_quant.h"

static bool
PgturbohybridGraphRepairAdjacencyForDeadNodes(Relation index, PgturbohybridGraphMetaPageData *meta,
								   bool *deadNodes)
{
	int			levelCapacity;
	bool		changedAny = false;
	BlockNumber blkno;
	BlockNumber nblocks;

	if (!BlockNumberIsValid(meta->tqAdjStartBlkno) || deadNodes == NULL)
		return false;

	levelCapacity = PgturbohybridGraphLevelCapacity(meta->m);
	blkno = meta->tqAdjStartBlkno;
	nblocks = RelationGetNumberOfBlocks(index);

	while (BlockNumberIsValid(blkno) && blkno < nblocks)
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		BlockNumber nextblkno;
		OffsetNumber maxoff;
		bool		changed = false;
		GenericXLogState *xlogState = NULL;

		CHECK_FOR_INTERRUPTS();

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

		if (RelationNeedsWAL(index))
		{
			xlogState = GenericXLogStart(index);
			page = GenericXLogRegisterBuffer(xlogState, buf, 0);
		}
		else
			page = BufferGetPage(buf);

		opaque = PgturbohybridGraphPageGetOpaque(page);
		if ((opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) != PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_ADJ)
		{
			if (xlogState != NULL)
				GenericXLogAbort(xlogState);
			UnlockReleaseBuffer(buf);
			break;
		}
		nextblkno = opaque->nextblkno;

		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
		{
			ItemId		iid = PageGetItemId(page, offno);
			PgturbohybridGraphAdjTuple tuple;
			uint16		maxNeighbors;
			uint16		oldCount;
			uint16		scanCount;
			uint16		newCount = 0;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (PgturbohybridGraphAdjTuple) PageGetItem(page, iid);
			if (tuple->type != PGTURBOHYBRID_GRAPH_ADJ_TUPLE_TYPE ||
				tuple->nodeId >= meta->tqNodeCount ||
				tuple->level >= levelCapacity)
				continue;

			maxNeighbors = PgturbohybridGraphLevelM(meta->m, tuple->level);
			oldCount = tuple->count;
			scanCount = Min(oldCount, maxNeighbors);

			if (!deadNodes[tuple->nodeId])
			{
				for (int i = 0; i < scanCount; i++)
				{
					uint32		neighbor = tuple->neighbors[i];

					if (neighbor < meta->tqNodeCount && !deadNodes[neighbor])
						tuple->neighbors[newCount++] = neighbor;
				}
			}

			if (newCount != oldCount)
			{
				if (newCount < scanCount)
					memset(&tuple->neighbors[newCount], 0,
						   sizeof(uint32) * (scanCount - newCount));
				tuple->count = newCount;
				changed = true;
			}
		}

		if (changed)
			PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_VACUUM_REPAIR);

		if (xlogState != NULL)
		{
			if (changed)
				GenericXLogFinish(xlogState);
			else
				GenericXLogAbort(xlogState);
		}
		else if (changed)
			MarkBufferDirty(buf);

		UnlockReleaseBuffer(buf);

		if (changed)
		{
			changedAny = true;
			PgturbohybridGraphLogGraphWalRecord(index, MAIN_FORKNUM, blkno, PGTURBOHYBRID_GRAPH_GRAPH_OP_VACUUM_REPAIR);
		}

		if (nextblkno == blkno)
			break;
		blkno = nextblkno;
	}

	return changedAny;
}

void
PgturbohybridGraphCollectVacuumStats(Relation index, PgturbohybridGraphMetaPageData *meta,
						  int64 *liveNodes, int64 *deadNodes,
						  int64 *adjacencyRefs, int64 *deadNeighborRefs)
{
	int			codeTuplesPerPage;
	int			codePageCount;
	int			tqBits = meta->tqBits != 0 ? meta->tqBits : PGTURBOHYBRID_DEFAULT_BITS;
	int			levelCapacity;
	bool	   *deadBitmap;
	BlockNumber nblocks;
	BlockNumber blkno;

	*liveNodes = 0;
	*deadNodes = 0;
	*adjacencyRefs = 0;
	*deadNeighborRefs = 0;

	if (meta->tqNodeCount == 0 ||
		!BlockNumberIsValid(meta->tqCodeStartBlkno))
		return;

	deadBitmap = palloc0(sizeof(bool) * meta->tqNodeCount);
	nblocks = RelationGetNumberOfBlocks(index);
	codeTuplesPerPage =
		PgturbohybridGraphTuplesPerPage(PgturbohybridGraphCodeTupleSize(meta->dimensions,
												  meta->tqPayloadCount,
												  tqBits,
												  (meta->tqFlags & PGTURBOHYBRID_GRAPH_TQ_WEIGHTED) != 0,
												  meta->tqResidualRerankBytes));
	codePageCount = PgturbohybridGraphPageCount(meta->tqNodeCount, codeTuplesPerPage);
	blkno = meta->tqCodeStartBlkno;

	for (int pageNo = 0;
		 pageNo < codePageCount && BlockNumberIsValid(blkno) && blkno < nblocks;
		 pageNo++)
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		OffsetNumber maxoff;

		CHECK_FOR_INTERRUPTS();

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = PgturbohybridGraphPageGetOpaque(page);
		if ((opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) != PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_CODE)
		{
			UnlockReleaseBuffer(buf);
			break;
		}

		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
		{
			ItemId		iid = PageGetItemId(page, offno);
			PgturbohybridGraphCodeTuple tuple;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (PgturbohybridGraphCodeTuple) PageGetItem(page, iid);
			if (tuple->type != PGTURBOHYBRID_GRAPH_CODE_TUPLE_TYPE ||
				tuple->nodeId >= meta->tqNodeCount)
				continue;

			if (tuple->flags & PGTURBOHYBRID_GRAPH_NODE_DEAD)
			{
				deadBitmap[tuple->nodeId] = true;
				(*deadNodes)++;
			}
			else
				(*liveNodes)++;
		}

		blkno = opaque->nextblkno;
		UnlockReleaseBuffer(buf);
	}

	if (!BlockNumberIsValid(meta->tqAdjStartBlkno))
	{
		pfree(deadBitmap);
		return;
	}

	levelCapacity = PgturbohybridGraphLevelCapacity(meta->m);
	blkno = meta->tqAdjStartBlkno;

	while (BlockNumberIsValid(blkno) && blkno < nblocks)
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		OffsetNumber maxoff;
		BlockNumber nextblkno;

		CHECK_FOR_INTERRUPTS();

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = PgturbohybridGraphPageGetOpaque(page);
		if ((opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) != PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_ADJ)
		{
			UnlockReleaseBuffer(buf);
			break;
		}
		nextblkno = opaque->nextblkno;

		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
		{
			ItemId		iid = PageGetItemId(page, offno);
			PgturbohybridGraphAdjTuple tuple;
			uint16		maxNeighbors;
			bool		deadSource;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (PgturbohybridGraphAdjTuple) PageGetItem(page, iid);
			if (tuple->type != PGTURBOHYBRID_GRAPH_ADJ_TUPLE_TYPE ||
				tuple->nodeId >= meta->tqNodeCount ||
				tuple->level >= levelCapacity)
				continue;

			maxNeighbors = PgturbohybridGraphLevelM(meta->m, tuple->level);
			deadSource = deadBitmap[tuple->nodeId];

			for (int i = 0; i < Min(tuple->count, maxNeighbors); i++)
			{
				uint32		neighbor = tuple->neighbors[i];

				(*adjacencyRefs)++;
				if (deadSource || neighbor >= meta->tqNodeCount ||
					deadBitmap[neighbor])
					(*deadNeighborRefs)++;
			}
		}

		UnlockReleaseBuffer(buf);
		if (nextblkno == blkno)
			break;
		blkno = nextblkno;
	}

	pfree(deadBitmap);
}

typedef struct PgturbohybridGraphBulkDeleteState
{
	double		liveTuples;
	bool		changedAny;
	bool		repairAny;
	bool		hasDeadNodes;
} PgturbohybridGraphBulkDeleteState;

IndexBulkDeleteResult *
tqgraphbulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats,
				  IndexBulkDeleteCallback callback, void *callback_state)
{
	Relation	index = info->index;
	IndexBulkDeleteResult *volatile result = stats;
	PgturbohybridGraphMetaPageData meta;
	PgturbohybridGraphBulkDeleteState *deleteState;
	int			codeTuplesPerPage;
	int			codePageCount;
	int			tqBits;
	bool	   *deadNodes = NULL;

	if (result == NULL)
		result = palloc0(sizeof(IndexBulkDeleteResult));

	if (callback == NULL || !PgturbohybridGraphReadMeta(index, &meta) ||
		meta.tqNodeCount == 0 ||
		!BlockNumberIsValid(meta.tqCodeStartBlkno))
		return (IndexBulkDeleteResult *) result;

	tqBits = meta.tqBits != 0 ? meta.tqBits : PGTURBOHYBRID_DEFAULT_BITS;
	codeTuplesPerPage =
		PgturbohybridGraphTuplesPerPage(PgturbohybridGraphCodeTupleSize(meta.dimensions,
														  meta.tqPayloadCount,
														  tqBits,
														  (meta.tqFlags & PGTURBOHYBRID_GRAPH_TQ_WEIGHTED) != 0,
														  meta.tqResidualRerankBytes));
	codePageCount = PgturbohybridGraphPageCount(meta.tqNodeCount, codeTuplesPerPage);
	deleteState = palloc0(sizeof(PgturbohybridGraphBulkDeleteState));
	deadNodes = palloc0(sizeof(bool) * meta.tqNodeCount);

	LockPage(index, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ExclusiveLock);
	PG_TRY();
	{
		BlockNumber blkno = meta.tqCodeStartBlkno;
		BlockNumber nblocks = RelationGetNumberOfBlocks(index);

		for (int pageNo = 0;
			 pageNo < codePageCount && BlockNumberIsValid(blkno) && blkno < nblocks;
			 pageNo++)
		{
			Buffer		buf;
			Page		page;
			PgturbohybridGraphPageOpaque opaque;
			BlockNumber nextblkno;
			OffsetNumber maxoff;
			bool		changed = false;
			GenericXLogState *xlogState = NULL;

			CHECK_FOR_INTERRUPTS();

			buf = ReadBuffer(index, blkno);
			LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

			if (RelationNeedsWAL(index))
			{
				xlogState = GenericXLogStart(index);
				page = GenericXLogRegisterBuffer(xlogState, buf, 0);
			}
			else
				page = BufferGetPage(buf);

			opaque = PgturbohybridGraphPageGetOpaque(page);
			if ((opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) != PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_CODE)
			{
				if (xlogState != NULL)
					GenericXLogAbort(xlogState);
				UnlockReleaseBuffer(buf);
				break;
			}
			nextblkno = opaque->nextblkno;

			maxoff = PageGetMaxOffsetNumber(page);
			for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
			{
				ItemId		iid = PageGetItemId(page, offno);
				PgturbohybridGraphCodeTuple tuple;

				if (!ItemIdIsUsed(iid))
					continue;

				tuple = (PgturbohybridGraphCodeTuple) PageGetItem(page, iid);
				if (tuple->type != PGTURBOHYBRID_GRAPH_CODE_TUPLE_TYPE ||
					tuple->nodeId >= meta.tqNodeCount)
					continue;

				if (tuple->flags & PGTURBOHYBRID_GRAPH_NODE_DEAD)
				{
					deadNodes[tuple->nodeId] = true;
					deleteState->hasDeadNodes = true;
					continue;
				}

				if (callback(&tuple->heaptid, callback_state))
				{
					tuple->flags |= PGTURBOHYBRID_GRAPH_NODE_DEAD;
					deadNodes[tuple->nodeId] = true;
					deleteState->hasDeadNodes = true;
					result->tuples_removed += 1;
					changed = true;
				}
				else
					deleteState->liveTuples += 1;
			}

			if (changed)
				PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_VACUUM_DELETE);

			if (xlogState != NULL)
				GenericXLogFinish(xlogState);
			else if (changed)
				MarkBufferDirty(buf);

			UnlockReleaseBuffer(buf);

			if (changed)
			{
				deleteState->changedAny = true;
				PgturbohybridGraphLogGraphWalRecord(index, MAIN_FORKNUM, blkno, PGTURBOHYBRID_GRAPH_GRAPH_OP_VACUUM_DELETE);
			}

			blkno = nextblkno;
		}

		if (deleteState->hasDeadNodes)
			deleteState->repairAny = PgturbohybridGraphRepairAdjacencyForDeadNodes(index, &meta, deadNodes);

		if (deleteState->changedAny || deleteState->repairAny)
			PgturbohybridGraphBumpMetaGeneration(index);
	}
	PG_CATCH();
	{
		UnlockPage(index, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ExclusiveLock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	UnlockPage(index, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ExclusiveLock);
	pfree(deadNodes);
	result->num_index_tuples = deleteState->liveTuples;
	pfree(deleteState);

	result->estimated_count = false;

	return (IndexBulkDeleteResult *) result;
}

IndexBulkDeleteResult *
tqgraphvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats)
{
	(void) info;

	if (stats == NULL)
		stats = palloc0(sizeof(IndexBulkDeleteResult));

	return stats;
}
