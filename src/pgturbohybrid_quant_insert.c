#include "postgres.h"

#include <float.h>
#include <math.h>
#include <string.h>

#include "access/generic_xlog.h"
#include "catalog/pg_type_d.h"
#include "miscadmin.h"
#include "portability/instr_time.h"
#include "storage/bufmgr.h"
#include "utils/datum.h"
#include "utils/memutils.h"
#include "utils/rel.h"

#include "pgturbohybrid_quant.h"
#include "pgturbohybrid_am.h"
#include "pgturbohybrid_multivector.h"
#include "pgturbohybrid_quant_score.h"

typedef struct PgturbohybridGraphInsertAppendCursor
{
	Oid			relid;
	Oid			relfilenumber;
	BlockNumber startBlkno;
	BlockNumber tailBlkno;
	uint16		pageKind;
} PgturbohybridGraphInsertAppendCursor;

typedef struct PgturbohybridGraphInsertStats
{
	uint64		reciprocalNeighborsConsidered;
	uint64		reciprocalAdjCachedHits;
	uint64		reciprocalAdjChainScans;
	uint64		reciprocalAdjPagesScanned;
	uint64		reciprocalUpdateUs;
} PgturbohybridGraphInsertStats;

static inline int64
PgturbohybridGraphInsertElapsedUsSince(instr_time start)
{
	instr_time	elapsed;

	INSTR_TIME_SET_CURRENT(elapsed);
	INSTR_TIME_SUBTRACT(elapsed, start);
	return (int64) INSTR_TIME_GET_MICROSEC(elapsed);
}

static PgturbohybridGraphInsertAppendCursor tqGraphCodeAppendCursor = {
	InvalidOid, InvalidOid, InvalidBlockNumber, InvalidBlockNumber, 0
};
static PgturbohybridGraphInsertAppendCursor tqGraphAdjAppendCursor = {
	InvalidOid, InvalidOid, InvalidBlockNumber, InvalidBlockNumber, 0
};

static int
PgturbohybridGraphInsertResultCompare(const void *a, const void *b)
{
	const PgturbohybridGraphResult *ia = (const PgturbohybridGraphResult *) a;
	const PgturbohybridGraphResult *ib = (const PgturbohybridGraphResult *) b;

	if (ia->distance < ib->distance)
		return -1;
	if (ia->distance > ib->distance)
		return 1;
	return (ia->nodeId > ib->nodeId) - (ia->nodeId < ib->nodeId);
}

static int
PgturbohybridGraphInsertFrontierCompare(const void *a, const void *b)
{
	const PgturbohybridGraphFrontierItem *ia = (const PgturbohybridGraphFrontierItem *) a;
	const PgturbohybridGraphFrontierItem *ib = (const PgturbohybridGraphFrontierItem *) b;

	if (ia->distance < ib->distance)
		return -1;
	if (ia->distance > ib->distance)
		return 1;
	return (ia->nodeId > ib->nodeId) - (ia->nodeId < ib->nodeId);
}

static Vector *
PgturbohybridGraphInsertMultiVectorSubvector(const PgturbohybridMultiVector *mv,
											 int32 ordinal)
{
	Vector	   *vector;
	Size		vectorSize;
	const float *values;

	values = PgturbohybridMultiVectorValues(mv, ordinal);
	vectorSize = PGTURBOHYBRID_VECTOR_SIZE(mv->dim);
	vector = (Vector *) palloc(vectorSize);
	SET_VARSIZE(vector, vectorSize);
	vector->dim = (int16) mv->dim;
	vector->unused = 0;
	memcpy(vector->x, values, sizeof(float) * (Size) mv->dim);
	return vector;
}

static double
PgturbohybridGraphExactDistanceToNode(Relation index, PgturbohybridGraphMetaPageData *meta,
						   PgturbohybridGraphScanStorage *storage, PgturbohybridGraphSupport *support,
						   Vector *query, uint32 nodeId)
{
	Vector	   *value;
	double		distance;

	if (nodeId >= meta->tqNodeCount ||
		!PgturbohybridGraphLoadCodePage(index, NULL, meta, storage, nodeId) ||
		(storage->nodes[nodeId].flags & PGTURBOHYBRID_GRAPH_NODE_DEAD))
		return DBL_MAX;

	value = PgturbohybridGraphReadExactVector(index, &storage->nodes[nodeId], meta->dimensions);
	if (value == NULL)
		return DBL_MAX;

	distance = PgturbohybridGraphExactDistance(support, PointerGetDatum(query),
									PointerGetDatum(value));
	pfree(value);

	return distance;
}

static double
PgturbohybridGraphInsertQueryDistanceToNode(Relation index, PgturbohybridGraphScanOpaque so,
								 PgturbohybridGraphMetaPageData *meta,
								 PgturbohybridGraphScanStorage *storage, uint32 nodeId)
{
	if (so == NULL ||
		nodeId >= meta->tqNodeCount ||
		!PgturbohybridGraphLoadCodePage(index, so, meta, storage, nodeId) ||
		(storage->nodes[nodeId].flags & PGTURBOHYBRID_GRAPH_NODE_DEAD))
		return DBL_MAX;

	return PgturbohybridGraphScoreNode(so, &storage->nodes[nodeId]);
}

static double
PgturbohybridGraphInsertCodeDistanceToNode(Relation index, PgturbohybridGraphScanOpaque so,
								PgturbohybridGraphMetaPageData *meta,
								PgturbohybridGraphScanStorage *storage,
								PgturbohybridGraphScanNode *sourceNode,
								uint32 sourceNodeId,
								PgturbohybridGraphScanNode *newNode, uint32 newNodeId,
								uint32 nodeId)
{
	PgturbohybridGraphScanNode *targetNode;
	double		distance;

	if (sourceNode == NULL || (sourceNode->flags & PGTURBOHYBRID_GRAPH_NODE_DEAD))
		return DBL_MAX;

	if (nodeId == newNodeId)
		targetNode = newNode;
	else
	{
		if (nodeId >= meta->tqNodeCount ||
			!PgturbohybridGraphLoadCodePage(index, so, meta, storage, nodeId) ||
			(storage->nodes[nodeId].flags & PGTURBOHYBRID_GRAPH_NODE_DEAD))
			return DBL_MAX;
		targetNode = &storage->nodes[nodeId];
	}

	if (PgturbohybridGraphCodeCodeDistance(so, meta, sourceNode, targetNode, &distance))
		return distance;

	if (nodeId == newNodeId)
		return PgturbohybridGraphInsertQueryDistanceToNode(index, so, meta, storage,
												sourceNodeId);
	return PgturbohybridGraphInsertQueryDistanceToNode(index, so, meta, storage, nodeId);
}

static bool
PgturbohybridGraphLoadAdjTuple(Relation index, PgturbohybridGraphMetaPageData *meta, uint32 nodeId,
					int level, uint32 *neighbors, int *count,
					PgturbohybridGraphInsertStats *stats)
{
	BlockNumber blkno = meta->tqAdjStartBlkno;
	BlockNumber nblocks;
	bool		found = false;

	*count = 0;
	if (nodeId >= meta->tqNodeCount || level < 0 ||
		level >= PgturbohybridGraphLevelCapacity(meta->m))
		return false;
	if (!BlockNumberIsValid(blkno))
		return false;

	if (stats != NULL)
		stats->reciprocalAdjChainScans++;

	nblocks = RelationGetNumberOfBlocks(index);
	while (BlockNumberIsValid(blkno) && blkno < nblocks)
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		OffsetNumber maxoff;
		BlockNumber nextblkno;

		CHECK_FOR_INTERRUPTS();
		if (stats != NULL)
			stats->reciprocalAdjPagesScanned++;
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
			int			maxNeighbors = PgturbohybridGraphLevelM(meta->m, level);

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (PgturbohybridGraphAdjTuple) PageGetItem(page, iid);
			if (tuple->type == PGTURBOHYBRID_GRAPH_ADJ_TUPLE_TYPE &&
				tuple->nodeId == nodeId &&
				tuple->level == level)
			{
				*count = Min(tuple->count, maxNeighbors);
				memcpy(neighbors, tuple->neighbors, sizeof(uint32) * *count);
				found = true;
				break;
			}
		}
		UnlockReleaseBuffer(buf);
		if (found || nextblkno == blkno)
			break;
		blkno = nextblkno;
	}

	return found;
}

static void
PgturbohybridGraphUpdateAdjTuple(Relation index, PgturbohybridGraphMetaPageData *meta,
					  PgturbohybridGraphScanStorage *storage, uint32 nodeId,
					  int level, uint32 *neighbors, int count,
					  PgturbohybridGraphInsertStats *stats)
{
	BlockNumber blkno = meta->tqAdjStartBlkno;
	BlockNumber nblocks;
	bool		found = false;
	int			cacheSlot = -1;

	if (nodeId >= meta->tqNodeCount || level < 0 ||
		level >= PgturbohybridGraphLevelCapacity(meta->m) ||
		count > PgturbohybridGraphLevelM(meta->m, level))
		elog(ERROR, "invalid pgturbohybrid graph adjacency update");

	if (!BlockNumberIsValid(blkno))
		elog(ERROR, "missing pgturbohybrid graph adjacency page");

	if (storage != NULL && storage->cached)
	{
		cacheSlot = PgturbohybridGraphAdjSlot(meta, nodeId, level);
		if (cacheSlot >= 0 && cacheSlot < PgturbohybridGraphAdjRecordCount(meta) &&
			BlockNumberIsValid(storage->adjBlknos[cacheSlot]) &&
			OffsetNumberIsValid(storage->adjOffnos[cacheSlot]))
		{
			Buffer		buf;
			Page		page;
			PgturbohybridGraphPageOpaque opaque;
			GenericXLogState *xlogState = NULL;

			blkno = storage->adjBlknos[cacheSlot];
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
			if ((opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) == PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_ADJ &&
				storage->adjOffnos[cacheSlot] <= PageGetMaxOffsetNumber(page))
			{
				ItemId		iid = PageGetItemId(page, storage->adjOffnos[cacheSlot]);
				PgturbohybridGraphAdjTuple tuple;
				Size		tupleSize;
				int			tupleCapacity;

				if (ItemIdIsUsed(iid))
				{
					tuple = (PgturbohybridGraphAdjTuple) PageGetItem(page, iid);
					tupleSize = ItemIdGetLength(iid);
					tupleCapacity = (tupleSize - offsetof(PgturbohybridGraphAdjTupleData, neighbors)) /
						sizeof(uint32);
					if (tuple->type == PGTURBOHYBRID_GRAPH_ADJ_TUPLE_TYPE &&
						tuple->nodeId == nodeId &&
						tuple->level == level &&
						count <= tupleCapacity)
					{
						tuple->count = count;
						memcpy(tuple->neighbors, neighbors,
							   sizeof(uint32) * count);
						if (count < tupleCapacity)
							memset(&tuple->neighbors[count], 0,
								   sizeof(uint32) * (tupleCapacity - count));
						PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_NEIGHBOR_UPDATE);
						found = true;
					}
				}
			}

			if (xlogState != NULL)
			{
				if (found)
					GenericXLogFinish(xlogState);
				else
					GenericXLogAbort(xlogState);
			}
			else if (found)
				MarkBufferDirty(buf);
			UnlockReleaseBuffer(buf);

			if (found)
			{
				if (stats != NULL)
					stats->reciprocalAdjCachedHits++;
				PgturbohybridGraphLogGraphWalRecord(index, MAIN_FORKNUM, blkno,
									  PGTURBOHYBRID_GRAPH_GRAPH_OP_NEIGHBOR_UPDATE);
			}
		}
	}

	if (found)
		goto update_cache;

	if (stats != NULL)
		stats->reciprocalAdjChainScans++;

	nblocks = RelationGetNumberOfBlocks(index);
	while (BlockNumberIsValid(blkno) && blkno < nblocks)
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		OffsetNumber maxoff;
		BlockNumber nextblkno;
		GenericXLogState *xlogState = NULL;

		CHECK_FOR_INTERRUPTS();
		if (stats != NULL)
			stats->reciprocalAdjPagesScanned++;
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
			Size		tupleSize;
			int			tupleCapacity;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (PgturbohybridGraphAdjTuple) PageGetItem(page, iid);
			if (tuple->type != PGTURBOHYBRID_GRAPH_ADJ_TUPLE_TYPE ||
				tuple->nodeId != nodeId ||
				tuple->level != level)
				continue;

			tupleSize = ItemIdGetLength(iid);
			if (tupleSize < offsetof(PgturbohybridGraphAdjTupleData, neighbors))
				elog(ERROR, "corrupt pgturbohybrid graph adjacency tuple");
			tupleCapacity = (tupleSize - offsetof(PgturbohybridGraphAdjTupleData, neighbors)) /
				sizeof(uint32);
			if (count > tupleCapacity)
				elog(ERROR, "pgturbohybrid graph adjacency tuple lacks update capacity");

			tuple->count = count;
			memcpy(tuple->neighbors, neighbors, sizeof(uint32) * count);
			if (count < tupleCapacity)
				memset(&tuple->neighbors[count], 0,
					   sizeof(uint32) * (tupleCapacity - count));
			PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_NEIGHBOR_UPDATE);
			if (storage != NULL && storage->cached && cacheSlot >= 0)
			{
				storage->adjBlknos[cacheSlot] = blkno;
				storage->adjOffnos[cacheSlot] = offno;
			}
			found = true;
			break;
		}

		if (xlogState != NULL)
		{
			if (found)
				GenericXLogFinish(xlogState);
			else
				GenericXLogAbort(xlogState);
		}
		else if (found)
			MarkBufferDirty(buf);
		UnlockReleaseBuffer(buf);

		if (found)
		{
			PgturbohybridGraphLogGraphWalRecord(index, MAIN_FORKNUM, blkno,
								  PGTURBOHYBRID_GRAPH_GRAPH_OP_NEIGHBOR_UPDATE);
			break;
		}
		if (nextblkno == blkno)
			break;
		blkno = nextblkno;
	}

	if (!found)
		elog(ERROR, "missing pgturbohybrid graph adjacency tuple");

update_cache:
	if (storage != NULL && storage->cached)
	{
		int			slot = cacheSlot >= 0 ? cacheSlot :
			PgturbohybridGraphAdjSlot(meta, nodeId, level);

		if (slot >= 0 && slot < PgturbohybridGraphAdjRecordCount(meta))
		{
			if (storage->neighbors[slot] != NULL)
				pfree(storage->neighbors[slot]);
			storage->neighborCounts[slot] = count;
			if (count > 0)
			{
				storage->neighbors[slot] =
					MemoryContextAlloc(storage->ctx, sizeof(uint32) * count);
				memcpy(storage->neighbors[slot], neighbors,
					   sizeof(uint32) * count);
			}
			else
				storage->neighbors[slot] = NULL;
		}
	}
}

static bool
PgturbohybridGraphSelectedContains(uint32 *selected, int selectedCount, uint32 nodeId)
{
	for (int i = 0; i < selectedCount; i++)
	{
		if (selected[i] == nodeId)
			return true;
	}

	return false;
}

static bool
PgturbohybridGraphInsertCandidateDiverse(Relation index, PgturbohybridGraphMetaPageData *meta,
							  PgturbohybridGraphScanStorage *storage,
							  PgturbohybridGraphSupport *support, uint32 candidate,
							  double candidateDistance, uint32 *selected,
							  int selectedCount)
{
	Vector	   *candidateVector;
	bool		good = true;

	if (selectedCount == 0)
		return true;

	candidateVector = PgturbohybridGraphReadExactVector(index, &storage->nodes[candidate],
											 meta->dimensions);
	if (candidateVector == NULL)
		return true;

	for (int i = 0; i < selectedCount; i++)
	{
		double		selectedDistance;

		selectedDistance = PgturbohybridGraphExactDistanceToNode(index, meta, storage,
													  support, candidateVector,
													  selected[i]);
		if (selectedDistance < candidateDistance)
		{
			good = false;
			break;
		}
	}

	pfree(candidateVector);
	return good;
}

static void
PgturbohybridGraphSelectInsertNeighbors(Relation index, PgturbohybridGraphMetaPageData *meta,
							 PgturbohybridGraphScanStorage *storage,
							 PgturbohybridGraphSupport *support,
							 PgturbohybridGraphResult *candidates, int candidateCount,
							 int nodeLevel, uint32 **selected,
							 int *selectedCounts)
{
	qsort(candidates, candidateCount, sizeof(PgturbohybridGraphResult), PgturbohybridGraphInsertResultCompare);

	for (int level = 0; level <= nodeLevel; level++)
	{
		int			maxNeighbors = PgturbohybridGraphLevelM(meta->m, level);

		for (int i = 0; i < candidateCount &&
			 selectedCounts[level] < maxNeighbors; i++)
		{
			uint32		nodeId = candidates[i].nodeId;

			if (nodeId >= meta->tqNodeCount ||
				storage->nodes[nodeId].level < level ||
				(storage->nodes[nodeId].flags & PGTURBOHYBRID_GRAPH_NODE_DEAD))
				continue;

			if (!PgturbohybridGraphInsertCandidateDiverse(index, meta, storage, support,
											   nodeId, candidates[i].distance,
											   selected[level],
											   selectedCounts[level]))
				continue;

			selected[level][selectedCounts[level]++] = nodeId;
		}

		for (int i = 0; i < candidateCount &&
			 selectedCounts[level] < maxNeighbors; i++)
		{
			uint32		nodeId = candidates[i].nodeId;

			if (nodeId >= meta->tqNodeCount ||
				storage->nodes[nodeId].level < level ||
				(storage->nodes[nodeId].flags & PGTURBOHYBRID_GRAPH_NODE_DEAD) ||
				PgturbohybridGraphSelectedContains(selected[level],
										selectedCounts[level], nodeId))
				continue;

			selected[level][selectedCounts[level]++] = nodeId;
		}
	}
}

static void
PgturbohybridGraphUpdateReciprocalNeighbor(Relation index, PgturbohybridGraphMetaPageData *meta,
								PgturbohybridGraphScanStorage *storage,
								PgturbohybridGraphScanOpaque so, PgturbohybridGraphSupport *support,
								Vector *newVector, uint8 *newCode,
								float newScale, float newNorm,
								float newCodeNorm, float newEcCorrection,
								uint32 newNodeId, uint32 src, int level,
								PgturbohybridGraphInsertStats *stats)
{
	int			maxNeighbors = PgturbohybridGraphLevelM(meta->m, level);
	uint32	   *neighbors;
	uint32	   *pruned;
	int			count;
	bool		found = false;
	Vector	   *sourceVector = NULL;

	if (src >= meta->tqNodeCount ||
		!PgturbohybridGraphLoadCodePage(index, NULL, meta, storage, src) ||
		(storage->nodes[src].flags & PGTURBOHYBRID_GRAPH_NODE_DEAD) ||
		storage->nodes[src].level < level)
		return;

	if (stats != NULL)
		stats->reciprocalNeighborsConsidered++;

	neighbors = palloc0(sizeof(uint32) * (maxNeighbors + 1));
	pruned = palloc0(sizeof(uint32) * maxNeighbors);
	if (storage->cached)
	{
		int			slot = PgturbohybridGraphAdjSlot(meta, src, level);

		if (slot < 0 || slot >= PgturbohybridGraphAdjRecordCount(meta))
		{
			pfree(neighbors);
			pfree(pruned);
			return;
		}
		count = Min(storage->neighborCounts[slot], maxNeighbors);
		if (count > 0 && storage->neighbors[slot] != NULL)
			memcpy(neighbors, storage->neighbors[slot],
				   sizeof(uint32) * count);
	}
	else if (!PgturbohybridGraphLoadAdjTuple(index, meta, src, level, neighbors, &count,
											 stats))
	{
		pfree(neighbors);
		pfree(pruned);
		return;
	}

	for (int i = 0; i < count; i++)
	{
		if (neighbors[i] == newNodeId)
		{
			found = true;
			break;
		}
	}

	if (!found)
		neighbors[count++] = newNodeId;

	if (count > maxNeighbors)
	{
		PgturbohybridGraphFrontierItem *ranked = palloc(sizeof(PgturbohybridGraphFrontierItem) * count);
		int			prunedCount = 0;

		sourceVector = PgturbohybridGraphReadExactVector(index, &storage->nodes[src],
											  meta->dimensions);
		if (sourceVector != NULL)
		{
			for (int i = 0; i < count; i++)
			{
				ranked[i].nodeId = neighbors[i];
				if (neighbors[i] == newNodeId)
					ranked[i].distance = PgturbohybridGraphExactDistance(support,
															  PointerGetDatum(sourceVector),
															  PointerGetDatum(newVector));
				else
					ranked[i].distance = PgturbohybridGraphExactDistanceToNode(index, meta, storage,
																	support, sourceVector,
																	neighbors[i]);
			}
		}
		else
		{
			PgturbohybridGraphScanNode newNode;

			memset(&newNode, 0, sizeof(newNode));
			newNode.code = newCode;
			newNode.level = level;
			newNode.scale = newScale;
			newNode.norm = newNorm;
			newNode.codeNorm = newCodeNorm;
			newNode.ecCorrection = newEcCorrection;
			newNode.loaded = true;

			for (int i = 0; i < count; i++)
			{
				ranked[i].nodeId = neighbors[i];
				ranked[i].distance = PgturbohybridGraphInsertCodeDistanceToNode(index, so,
																	 meta, storage,
																	 &storage->nodes[src],
																	 src, &newNode,
																	 newNodeId,
																	 neighbors[i]);
			}
		}

		qsort(ranked, count, sizeof(PgturbohybridGraphFrontierItem), PgturbohybridGraphInsertFrontierCompare);
		for (int i = 0; i < count && prunedCount < maxNeighbors; i++)
		{
			if (ranked[i].distance < DBL_MAX)
				pruned[prunedCount++] = ranked[i].nodeId;
		}

		memcpy(neighbors, pruned, sizeof(uint32) * prunedCount);
		count = prunedCount;
		pfree(ranked);
		if (sourceVector != NULL)
			pfree(sourceVector);
	}

	PgturbohybridGraphUpdateAdjTuple(index, meta, storage, src, level, neighbors, count,
									 stats);
	pfree(neighbors);
	pfree(pruned);
}

static void
PgturbohybridGraphUpdateReciprocalNeighbors(Relation index, PgturbohybridGraphMetaPageData *meta,
								 PgturbohybridGraphScanStorage *storage,
								 PgturbohybridGraphScanOpaque so, PgturbohybridGraphSupport *support,
								 Vector *newVector, uint8 *newCode,
								 float newScale, float newNorm,
								 float newCodeNorm, float newEcCorrection,
								 uint32 newNodeId, int nodeLevel,
								 uint32 **selected, int *selectedCounts,
								 PgturbohybridGraphInsertStats *stats)
{
	instr_time	updateStart;

	INSTR_TIME_SET_CURRENT(updateStart);
	/*
	 * Cached insert storage already has direct adjacency block/offset metadata,
	 * so reciprocal updates are O(selected_neighbors * M).  Without that
	 * metadata, each update may fall back to scanning the adjacency page chain.
	 */
	for (int level = 0; level <= nodeLevel; level++)
	{
		for (int i = 0; i < selectedCounts[level]; i++)
			PgturbohybridGraphUpdateReciprocalNeighbor(index, meta, storage, so, support,
											newVector, newCode, newScale,
											newNorm, newCodeNorm,
											newEcCorrection, newNodeId,
											selected[level][i], level, stats);
	}
	if (stats != NULL)
		stats->reciprocalUpdateUs += PgturbohybridGraphInsertElapsedUsSince(updateStart);
}


static OffsetNumber
PgturbohybridGraphAppendTupleWithCursor(Relation index, BlockNumber *startBlkno,
							 uint16 pageKind, Item tuple, Size tupleSize,
							 uint16 graphOpKind, BlockNumber *insertBlkno,
							 PgturbohybridGraphInsertAppendCursor *cursor)
{
	BlockNumber originalStart = *startBlkno;
	BlockNumber appendStart = originalStart;
	OffsetNumber offno;

	if (cursor->relid == RelationGetRelid(index) &&
		cursor->relfilenumber == PgturbohybridGraphRelFileNumber(index) &&
		cursor->pageKind == pageKind &&
		cursor->startBlkno == originalStart &&
		BlockNumberIsValid(cursor->tailBlkno))
		appendStart = cursor->tailBlkno;

	offno = PgturbohybridGraphAppendTuple(index, MAIN_FORKNUM, &appendStart,
							   pageKind, tuple, tupleSize, graphOpKind,
							   insertBlkno);
	if (!BlockNumberIsValid(originalStart))
		*startBlkno = appendStart;

	cursor->relid = RelationGetRelid(index);
	cursor->relfilenumber = PgturbohybridGraphRelFileNumber(index);
	cursor->pageKind = pageKind;
	cursor->startBlkno = *startBlkno;
	cursor->tailBlkno = *insertBlkno;

	return offno;
}


static void
PgturbohybridGraphAppendInsertedCode(Relation index, BlockNumber *codeStart,
						  uint32 nodeId, ItemPointer heapTid, int nodeLevel,
						  Vector *vector, uint8 *code, uint8 *residualSketch,
						  float scale,
						  int32 *payloads, uint16 payloadMask, int payloadCount,
						  int bits, BlockNumber exactBlkno, OffsetNumber exactOffno,
						  bool tqWeighted, float ecCorrection, Size residualBytes)
{
	Size		payloadBytes = PgturbohybridGraphPayloadBytes(payloadCount);
	Size		tupleSize = PgturbohybridGraphCodeTupleSize(vector->dim, payloadCount,
												 bits, tqWeighted, residualBytes);
	PgturbohybridGraphCodeTuple tuple = palloc0(tupleSize);
	BlockNumber codeBlkno;

	tuple->type = PGTURBOHYBRID_GRAPH_CODE_TUPLE_TYPE;
	tuple->level = nodeLevel;
	tuple->flags = 0;
	tuple->nodeId = nodeId;
	tuple->heaptid = *heapTid;
	tuple->exactBlkno = exactBlkno;
	tuple->exactOffno = exactOffno;
	tuple->payloadMask = payloadMask;
	tuple->scale = scale;
	tuple->norm = PgturbohybridGraphVectorNorm(vector);
	tuple->correction = PgturbohybridGraphCodeNorm(code, vector->dim, bits);
	PgturbohybridGraphTupleSetEcCorrection(tuple, tqWeighted, ecCorrection);
	if (payloadBytes > 0 && payloads != NULL)
		memcpy(PgturbohybridGraphTuplePayloads(tuple, tqWeighted), payloads, payloadBytes);
	if (residualBytes > 0 && residualSketch != NULL)
		memcpy(PgturbohybridGraphTupleResidual(tuple, payloadBytes, residualBytes,
											   tqWeighted),
			   residualSketch, residualBytes);
	memcpy(PgturbohybridGraphTupleCode(tuple, payloadBytes, residualBytes, tqWeighted), code,
		   PgturbohybridGraphCodeBytesForBits(vector->dim, bits));

	(void) PgturbohybridGraphAppendTupleWithCursor(index, codeStart,
										PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_CODE,
										(Item) tuple, tupleSize,
										PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT,
										&codeBlkno,
										&tqGraphCodeAppendCursor);
	pfree(tuple);
}

static void
PgturbohybridGraphAppendInsertedAdj(Relation index, BlockNumber *adjStart, int m,
						 uint32 nodeId, int nodeLevel, uint32 **selected,
						 int *selectedCounts, BlockNumber *adjBlknos,
						 OffsetNumber *adjOffnos)
{
	Size		maxTupleSize = PgturbohybridGraphAdjTupleSize(PgturbohybridGraphLevelM(m, 0));
	PgturbohybridGraphAdjTuple tuple = palloc0(maxTupleSize);
	int			maxLevel = Min(nodeLevel, PgturbohybridGraphLevelCapacity(m) - 1);

	for (int level = 0; level <= maxLevel; level++)
	{
		BlockNumber adjBlkno;
		OffsetNumber adjOffno;
		int			count = selectedCounts[level];
		Size		tupleSize = PgturbohybridGraphAdjTupleSize(PgturbohybridGraphLevelM(m, level));

		memset(tuple, 0, maxTupleSize);
		tuple->type = PGTURBOHYBRID_GRAPH_ADJ_TUPLE_TYPE;
		tuple->level = level;
		tuple->count = count;
		tuple->nodeId = nodeId;
		for (int i = 0; i < count; i++)
			tuple->neighbors[i] = selected[level][i];

		adjOffno = PgturbohybridGraphAppendTupleWithCursor(index, adjStart,
												PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_ADJ,
												(Item) tuple, tupleSize,
												PGTURBOHYBRID_GRAPH_GRAPH_OP_NEIGHBOR_INSERT,
												&adjBlkno,
												&tqGraphAdjAppendCursor);
		if (adjBlknos != NULL)
			adjBlknos[level] = adjBlkno;
		if (adjOffnos != NULL)
			adjOffnos[level] = adjOffno;
	}

	pfree(tuple);
}

uint32
PgturbohybridGraphInsertValueInPlace(Relation index, IndexInfo *indexInfo,
						  ItemPointer heap_tid, Datum value,
						  Datum *values, bool *isnull)
{
	PgturbohybridGraphMetaPageData meta;
	PgturbohybridGraphScanStorage storage;
	PgturbohybridGraphScanOpaqueData insertSo;
	PgturbohybridGraphSupport support;
	PgturbohybridQuantMetaUpdate metaUpdate;
	PgturbohybridGraphNativeCache *insertCache = NULL;
	PgturbohybridGraphInsertStats insertStats;
	Vector	   *vector = (Vector *) DatumGetPointer(value);
	uint8	   *code;
	float		scale;
	uint32		newNodeId;
	int			nodeLevel;
	int			levelCapacity;
	uint32	  **selected;
	int		   *selectedCounts;
	BlockNumber *adjBlknos;
	OffsetNumber *adjOffnos;
	BlockNumber codeStart;
	BlockNumber adjStart;
	BlockNumber exactStart;
	BlockNumber exactBlkno;
	OffsetNumber exactOffno;
	uint32		entryNodeId;
	int			entryLevel;
	int			payloadCount;
	Size		payloadBytes;
	int32	   *payloads = NULL;
	uint16		payloadMask = 0;
	float	   *ecShift = NULL;
	float	   *ecScale = NULL;
	bool		insertTqWeighted;
	bool		insertTqRenorm;
	bool		insertExactStorage;
	float		insertXm;
	float		insertNorm;
	float		insertCodeNorm;
	uint8	   *residualSketch = NULL;

	if (!PgturbohybridGraphReadMeta(index, &meta))
		elog(ERROR, "pgturbohybrid native graph metapage is missing or invalid");
	memset(&insertStats, 0, sizeof(insertStats));

	if (meta.dimensions != 0 && meta.dimensions != vector->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("different vector dimensions are not supported in the same pgturbohybrid graph")));

	newNodeId = meta.tqNodeCount;
	nodeLevel = PgturbohybridGraphPickLevel(newNodeId, meta.m);
	levelCapacity = PgturbohybridGraphLevelCapacity(meta.m);
	payloadCount = meta.tqPayloadCount;
	payloadBytes = PgturbohybridGraphPayloadBytes(payloadCount);
	if (payloadCount > 0)
	{
		PgturbohybridQuantBuildState payloadState;

		memset(&payloadState, 0, sizeof(payloadState));
		payloadState.index = index;
		payloadState.indexInfo = indexInfo;
		payloadState.payloadCount = payloadCount;
		payloadState.payloadBytes = payloadBytes;
		payloads = palloc0(payloadBytes);
		PgturbohybridGraphCopyPayloadValues(&payloadState, payloads, &payloadMask,
								 values, isnull);
	}
	(void) PgturbohybridGraphLoadCorrection(index, vector->dim, &ecShift, &ecScale);
	code = palloc0(PgturbohybridGraphCodeBytesForBits(vector->dim, meta.tqBits));
	insertTqWeighted = (meta.tqFlags & PGTURBOHYBRID_GRAPH_TQ_WEIGHTED) != 0;
	insertTqRenorm = (meta.tqFlags & PGTURBOHYBRID_GRAPH_TQ_RENORM) != 0;
	insertExactStorage = (meta.tqFlags & PGTURBOHYBRID_GRAPH_EXACT_FREE) == 0;
	insertXm = 0.0f;
	if (ecShift != NULL && ecScale != NULL)
	{
		if (insertTqWeighted && insertTqRenorm)
		{
			float		centroidNorm;
			double		dimSqrt;

			scale = TqEncodeVectorWithCorrectionXmRenormBits(vector, code, meta.tqBits,
															 ecShift, ecScale,
															 &insertXm, &centroidNorm);
			if (scale != 0.0f && centroidNorm > 0.0f)
			{
				dimSqrt = sqrt((double) vector->dim);
				scale = (float) ((double) scale * dimSqrt / (double) centroidNorm);
			}
		}
		else if (insertTqWeighted)
			scale = TqEncodeVectorWithCorrectionAndXmBits(vector, code, meta.tqBits,
														   ecShift, ecScale, &insertXm);
		else
			scale = TqEncodeVectorWithCorrectionBits(vector, code, meta.tqBits,
													 ecShift, ecScale);
	}
	else
		scale = TqEncodeVectorBits(vector, code, meta.tqBits);
	insertNorm = PgturbohybridGraphVectorNorm(vector);
	insertCodeNorm = PgturbohybridGraphCodeNorm(code, vector->dim, meta.tqBits);
	if (meta.tqResidualRerankBytes > 0)
	{
		residualSketch = palloc0(meta.tqResidualRerankBytes);
		PgturbohybridGraphBuildResidualSketch(vector->x, vector->dim,
											  residualSketch,
											  meta.tqResidualRerankBytes);
	}
	selected = palloc0(sizeof(uint32 *) * levelCapacity);
	selectedCounts = palloc0(sizeof(int) * levelCapacity);
	adjBlknos = palloc(sizeof(BlockNumber) * levelCapacity);
	adjOffnos = palloc(sizeof(OffsetNumber) * levelCapacity);
	for (int level = 0; level < levelCapacity; level++)
	{
		selected[level] = palloc0(sizeof(uint32) * PgturbohybridGraphLevelM(meta.m, level));
		adjBlknos[level] = InvalidBlockNumber;
		adjOffnos[level] = InvalidOffsetNumber;
	}

	PgturbohybridGraphInitSupport(&support, index);

	if (meta.tqNodeCount > 0)
	{
		PgturbohybridGraphResult *candidates;
		int			resultTarget;
		int			searchEf;
		int			candidateCount;
		Datum		query = PointerGetDatum(vector);

		memset(&insertSo, 0, sizeof(insertSo));
		insertSo.support = support;
		insertSo.efSearch = Max(meta.efConstruction, PgturbohybridGraphLevelM(meta.m, 0));
		insertSo.graphOversampling = 1;
		insertSo.graphRescoreBand = insertExactStorage ?
			PGTURBOHYBRID_GRAPH_RESCORE_BAND_EXACT : PGTURBOHYBRID_GRAPH_RESCORE_BAND_NONE;
		PgturbohybridGraphPrepareTqQuery(index, &support, query, &insertSo.tq);

		insertCache = PgturbohybridGraphInitInsertStorage(index, &meta, &storage);
		resultTarget = Min(Max(meta.efConstruction, PgturbohybridGraphLevelM(meta.m, 0)),
						   (int) meta.tqNodeCount);
		searchEf = resultTarget;
		candidates = palloc0(sizeof(PgturbohybridGraphResult) * resultTarget);
		candidateCount = PgturbohybridGraphTraverse(index, &insertSo, &meta, &storage,
										  candidates, resultTarget, searchEf,
										  query, -1, 0);
		if (insertExactStorage)
			PgturbohybridGraphExactRescore(index, &insertSo, query, &meta, storage.nodes,
								candidates, candidateCount);
		PgturbohybridGraphSelectInsertNeighbors(index, &meta, &storage, &support,
									 candidates, candidateCount, nodeLevel,
									 selected, selectedCounts);
		PgturbohybridGraphUpdateReciprocalNeighbors(index, &meta, &storage, &insertSo,
										 &support, vector, code, scale,
										 insertNorm, insertCodeNorm, insertXm,
										 newNodeId, nodeLevel,
										 selected, selectedCounts, &insertStats);
		pfree(candidates);
	}

	if (insertStats.reciprocalNeighborsConsidered > 0)
		ereport(DEBUG1,
				(errmsg("pgturbohybrid native graph insert reciprocal update stats"),
				 errdetail("insert_reciprocal_neighbors_considered=" UINT64_FORMAT
						   " insert_reciprocal_adj_cached_hits=" UINT64_FORMAT
						   " insert_reciprocal_adj_chain_scans=" UINT64_FORMAT
						   " insert_reciprocal_adj_pages_scanned=" UINT64_FORMAT
						   " insert_reciprocal_update_us=" UINT64_FORMAT,
						   insertStats.reciprocalNeighborsConsidered,
						   insertStats.reciprocalAdjCachedHits,
						   insertStats.reciprocalAdjChainScans,
						   insertStats.reciprocalAdjPagesScanned,
						   insertStats.reciprocalUpdateUs)));

	codeStart = meta.tqCodeStartBlkno;
	adjStart = meta.tqAdjStartBlkno;
	exactStart = meta.tqExactStartBlkno;
	if (insertExactStorage)
		PgturbohybridGraphAppendInsertedExact(index, &exactStart, newNodeId, vector,
								   vector->dim, &exactBlkno, &exactOffno);
	else
	{
		exactBlkno = InvalidBlockNumber;
		exactOffno = InvalidOffsetNumber;
	}
	PgturbohybridGraphAppendInsertedCode(index, &codeStart, newNodeId, heap_tid, nodeLevel,
							  vector, code, residualSketch, scale, payloads, payloadMask,
							  payloadCount, meta.tqBits, exactBlkno, exactOffno,
							  insertTqWeighted, insertXm,
							  meta.tqResidualRerankBytes);
	PgturbohybridGraphAppendInsertedAdj(index, &adjStart, meta.m, newNodeId, nodeLevel,
							 selected, selectedCounts, adjBlknos, adjOffnos);

	entryNodeId = meta.tqNodeCount == 0 || nodeLevel > meta.graphMaxLevel ?
		newNodeId : meta.tqEntryNodeId;
	entryLevel = entryNodeId == newNodeId ? nodeLevel : meta.entryLevel;

	memset(&metaUpdate, 0, sizeof(metaUpdate));
	metaUpdate.forkNum = MAIN_FORKNUM;
	metaUpdate.building = false;
	metaUpdate.dimensions = vector->dim;
	metaUpdate.m = meta.m;
	metaUpdate.efConstruction = meta.efConstruction;
	metaUpdate.graphMaxLevel = Max(meta.graphMaxLevel, nodeLevel);
	metaUpdate.nodeCount = meta.tqNodeCount + 1;
	metaUpdate.entryNodeId = entryNodeId;
	metaUpdate.entryLevel = entryLevel;
	metaUpdate.tqBits = meta.tqBits;
	metaUpdate.tqPayloadCount = payloadCount;
	metaUpdate.tqPayloadBytes = payloadBytes;
	metaUpdate.tqFlags = meta.tqFlags;
	metaUpdate.tqEntrySidecarCount = meta.tqEntrySidecarCount;
	metaUpdate.tqEntrySidecarBytes = meta.tqEntrySidecarBytes;
	metaUpdate.tqResidualRerankBytes = meta.tqResidualRerankBytes;
	memcpy(metaUpdate.tqEntrySidecarNodeIds, meta.tqEntrySidecarNodeIds,
		   sizeof(metaUpdate.tqEntrySidecarNodeIds));
	metaUpdate.tqRoutingEntryCount = meta.tqRoutingEntryCount;
	metaUpdate.tqRoutingEntryBytes = meta.tqRoutingEntryBytes;
	memcpy(metaUpdate.tqRoutingEntryNodeIds, meta.tqRoutingEntryNodeIds,
		   sizeof(metaUpdate.tqRoutingEntryNodeIds));
	metaUpdate.tqSegmentCount = meta.tqSegmentCount;
	memcpy(metaUpdate.tqSegments, meta.tqSegments, sizeof(metaUpdate.tqSegments));
	metaUpdate.buildScanUs = meta.buildScanUs;
	metaUpdate.buildCorrectionUs = meta.buildCorrectionUs;
	metaUpdate.buildEncodeUs = meta.buildEncodeUs;
	metaUpdate.buildEdgeUs = meta.buildEdgeUs;
	metaUpdate.buildWriteUs = meta.buildWriteUs;
	metaUpdate.buildWorkerCount = meta.buildWorkerCount;
	/*
	 * Single-row inserts must keep the metapage update O(1) in node count.
	 * Do not construct fake build nodes here just to carry entry metadata.
	 */
	PgturbohybridQuantUpdateMetaPageFromUpdate(index, &metaUpdate, codeStart, adjStart,
											   exactStart,
											   meta.tqCorrectionStartBlkno);
	if (insertCache != NULL)
		PgturbohybridGraphAppendInsertCacheNode(insertCache, &meta, newNodeId, heap_tid,
									 nodeLevel, vector, code, residualSketch, scale,
									 insertNorm, insertCodeNorm, insertXm,
									 payloads, payloadMask, exactBlkno,
									 exactOffno, selected, selectedCounts,
									 adjBlknos, adjOffnos,
									 codeStart, adjStart, exactStart,
									 entryNodeId,
									 (uint16) Max(meta.graphMaxLevel, nodeLevel));
	for (int level = 0; level < levelCapacity; level++)
		pfree(selected[level]);
	pfree(selected);
	pfree(selectedCounts);
	pfree(adjBlknos);
	pfree(adjOffnos);
	if (payloads != NULL)
		pfree(payloads);
	if (ecShift != NULL)
		pfree(ecShift);
	if (ecScale != NULL)
		pfree(ecScale);
	if (residualSketch != NULL)
		pfree(residualSketch);
	pfree(code);

	return newNodeId;
}

uint32
PgturbohybridGraphInsertMultiVectorInPlace(Relation index, IndexInfo *indexInfo,
										   ItemPointer heap_tid, Datum value,
										   Datum *values, bool *isnull,
										   uint32 *insertedNodes)
{
	PgturbohybridMultiVector *mv;
	char	   *rawValue;
	uint32		firstNodeId = InvalidOid;
	uint32		count = 0;

	rawValue = (char *) DatumGetPointer(value);
	mv = PgturbohybridDatumGetMultiVector(value);
	PgturbohybridMultiVectorCheckDim((uint32) mv->dim,
									 (uint32) pgturbohybrid_multivector_max_dim);
	PgturbohybridMultiVectorCheckTokenCount((uint32) mv->count,
											(uint32) pgturbohybrid_multivector_max_doc_vectors);

	for (int32 i = 0; i < mv->count; i++)
	{
		Vector	   *vector;
		uint32		nodeId;

		vector = PgturbohybridGraphInsertMultiVectorSubvector(mv, i);
		nodeId = PgturbohybridGraphInsertValueInPlace(index, indexInfo, heap_tid,
													  PointerGetDatum(vector),
													  values, isnull);
		if (i == 0)
			firstNodeId = nodeId;
		count++;
		pfree(vector);
	}

	if ((char *) mv != rawValue)
		pfree(mv);
	if (insertedNodes != NULL)
		*insertedNodes = count;
	return firstNodeId;
}
