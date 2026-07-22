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

#include <float.h>
#include <unistd.h>

#include "access/genam.h"
#include "access/generic_xlog.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "storage/fd.h"
#include "storage/lmgr.h"
#include "utils/rel.h"

#include "pgturbohybrid.h"
#include "pgturbohybrid_quant.h"

#define PGTURBOHYBRID_GRAPH_VACUUM_CANDIDATE_MULTIPLIER 4
#define PGTURBOHYBRID_GRAPH_REINDEX_DEAD_RATIO 0.20
#define PGTURBOHYBRID_GRAPH_REINDEX_DEAD_REF_RATIO 0.25

typedef struct PgturbohybridGraphVacuumAdj
{
	uint32	   *neighbors;
	int			count;
} PgturbohybridGraphVacuumAdj;

typedef struct PgturbohybridGraphVacuumCandidate
{
	uint32		nodeId;
	double		distance;
} PgturbohybridGraphVacuumCandidate;

/* Test-only crash orchestration.  No SQL-visible switch or production path. */
static void
PgturbohybridGraphVacuumTestPause(const char *stage)
{
	const char *dir = getenv("PGTURBOHYBRID_TEST_VACUUM_STAGE_DIR");
	char		requestPath[MAXPGPATH];
	char		reachedPath[MAXPGPATH];
	int		fd;

	if (dir == NULL || dir[0] != '/')
		return;
	snprintf(requestPath, sizeof(requestPath), "%s/%s.request", dir, stage);
	if (access(requestPath, F_OK) != 0)
		return;
	snprintf(reachedPath, sizeof(reachedPath), "%s/%s.reached", dir, stage);
	fd = OpenTransientFile(reachedPath, O_WRONLY | O_CREAT | O_EXCL | PG_BINARY);
	if (fd >= 0)
		CloseTransientFile(fd);
	while (access(requestPath, F_OK) == 0)
	{
		CHECK_FOR_INTERRUPTS();
		pg_usleep(10000L);
	}
}

static int
PgturbohybridGraphVacuumCandidateCompare(const void *left, const void *right)
{
	const PgturbohybridGraphVacuumCandidate *a = left;
	const PgturbohybridGraphVacuumCandidate *b = right;

	if (a->distance < b->distance)
		return -1;
	if (a->distance > b->distance)
		return 1;
	return (a->nodeId > b->nodeId) - (a->nodeId < b->nodeId);
}

static double
PgturbohybridGraphVacuumCodeDistance(PgturbohybridGraphScanStorage *storage,
									  PgturbohybridGraphMetaPageData *meta,
									  uint32 a, uint32 b)
{
	double		distance = 0.0;

	if (a >= meta->tqNodeCount || b >= meta->tqNodeCount ||
		storage->nodes[a].code == NULL || storage->nodes[b].code == NULL)
		return DBL_MAX;
	for (uint16 i = 0; i < meta->tqCodeBytes; i++)
	{
		double		delta = (double) storage->nodes[a].code[i] -
			(double) storage->nodes[b].code[i];

		distance += delta * delta;
	}
	return distance;
}

static bool
PgturbohybridGraphVacuumAdjContains(PgturbohybridGraphVacuumAdj *adj,
									 uint32 nodeId)
{
	for (int i = 0; i < adj->count; i++)
		if (adj->neighbors[i] == nodeId)
			return true;
	return false;
}

static int
PgturbohybridGraphVacuumReplaceableSlot(PgturbohybridGraphVacuumAdj *adj,
									 bool *deadNodes, uint32 nodeCount,
									 int capacity)
{
	for (int i = 0; i < adj->count; i++)
		if (adj->neighbors[i] >= nodeCount || deadNodes[adj->neighbors[i]])
			return i;
	return adj->count < capacity ? adj->count : -1;
}

static bool
PgturbohybridGraphVacuumInstallReciprocal(PgturbohybridGraphVacuumAdj *allAdj,
									   bool *deadNodes, uint32 nodeCount,
									   int levelCapacity,
									   int level, int capacity, uint32 source,
									   uint32 deadNode, uint32 replacement)
{
	PgturbohybridGraphVacuumAdj *sourceAdj =
		&allAdj[(Size) source * levelCapacity + level];
	PgturbohybridGraphVacuumAdj *replacementAdj =
		&allAdj[(Size) replacement * levelCapacity + level];
	int			sourceSlot = -1;
	int			reciprocalSlot;

	for (int i = 0; i < sourceAdj->count; i++)
		if (sourceAdj->neighbors[i] == deadNode)
		{
			sourceSlot = i;
			break;
		}
	if (sourceSlot < 0 || PgturbohybridGraphVacuumAdjContains(sourceAdj, replacement))
		return false;

	if (PgturbohybridGraphVacuumAdjContains(replacementAdj, source))
		reciprocalSlot = -2;
	else
		reciprocalSlot = PgturbohybridGraphVacuumReplaceableSlot(replacementAdj,
														 deadNodes, nodeCount,
														 capacity);
	if (reciprocalSlot == -1)
		return false;

	sourceAdj->neighbors[sourceSlot] = replacement;
	if (reciprocalSlot >= 0)
	{
		replacementAdj->neighbors[reciprocalSlot] = source;
		if (reciprocalSlot == replacementAdj->count)
			replacementAdj->count++;
	}
	return true;
}

static bool
PgturbohybridGraphRepairAdjacencyForDeadNodes(Relation index, PgturbohybridGraphMetaPageData *meta,
								   bool *deadNodes, uint8 *nodeLevels,
								   uint64 *repairedEdges)
{
	PgturbohybridGraphScanStorage storage;
	PgturbohybridGraphVacuumAdj *allAdj;
	PgturbohybridGraphVacuumAdj *originalAdj;
	int			levelCapacity;
	int			maxCapacity;
	bool		changedAny = false;

	if (!BlockNumberIsValid(meta->tqAdjStartBlkno) || deadNodes == NULL)
		return false;

	levelCapacity = PgturbohybridGraphLevelCapacity(meta->m);
	maxCapacity = PgturbohybridGraphLevelM(meta->m, 0);
	allAdj = palloc0(sizeof(*allAdj) * meta->tqNodeCount * levelCapacity);
	originalAdj = palloc0(sizeof(*originalAdj) * meta->tqNodeCount * levelCapacity);
	memset(&storage, 0, sizeof(storage));
	PgturbohybridGraphInitScanStorage(index, meta, &storage, NULL);

	for (uint32 nodeId = 0; nodeId < meta->tqNodeCount; nodeId++)
	{
		CHECK_FOR_INTERRUPTS();
		if (!PgturbohybridGraphLoadCodePage(index, NULL, meta, &storage, nodeId))
			continue;
		for (int level = 0; level <= nodeLevels[nodeId] && level < levelCapacity; level++)
		{
			PgturbohybridGraphVacuumAdj *adj =
				&allAdj[(Size) nodeId * levelCapacity + level];
			PgturbohybridGraphVacuumAdj *original =
				&originalAdj[(Size) nodeId * levelCapacity + level];

			adj->neighbors = palloc0(sizeof(uint32) * maxCapacity);
			original->neighbors = palloc0(sizeof(uint32) * maxCapacity);
			if (!PgturbohybridGraphVacuumLoadAdjTuple(index, meta, nodeId, level,
												   adj->neighbors, &adj->count))
				continue;
			original->count = adj->count;
			memcpy(original->neighbors, adj->neighbors,
				   sizeof(uint32) * adj->count);
		}
	}

	/* Dead adjacency remains navigable.  Replace an incoming dead reference only
	 * when a reciprocal live edge can be installed at the same level. */
	for (uint32 deadNode = 0; deadNode < meta->tqNodeCount; deadNode++)
	{
		if (!deadNodes[deadNode])
			continue;
		for (int level = Min((int) nodeLevels[deadNode], levelCapacity - 1);
			 level >= 0; level--)
		{
			int			capacity = PgturbohybridGraphLevelM(meta->m, level);
			int			candidateLimit = capacity *
				PGTURBOHYBRID_GRAPH_VACUUM_CANDIDATE_MULTIPLIER;
			PgturbohybridGraphVacuumAdj *deadAdj =
				&allAdj[(Size) deadNode * levelCapacity + level];

			for (uint32 source = 0; source < meta->tqNodeCount; source++)
			{
				PgturbohybridGraphVacuumAdj *sourceAdj =
					&allAdj[(Size) source * levelCapacity + level];
				PgturbohybridGraphVacuumCandidate *candidates;
				int			candidateCount = 0;

				if (deadNodes[source] || nodeLevels[source] < level ||
					!PgturbohybridGraphVacuumAdjContains(sourceAdj, deadNode))
					continue;
				candidates = palloc(sizeof(*candidates) * candidateLimit);
				for (int pass = 0; pass < 2 && candidateCount < candidateLimit; pass++)
				{
					for (int i = 0; i < deadAdj->count && candidateCount < candidateLimit; i++)
					{
						uint32		seed = deadAdj->neighbors[i];
						PgturbohybridGraphVacuumAdj *pool = pass == 0 ? deadAdj :
							(seed < meta->tqNodeCount ?
							 &allAdj[(Size) seed * levelCapacity + level] : NULL);

						if (pool == NULL)
							continue;
						for (int j = pass == 0 ? i : 0;
							 j < pool->count && candidateCount < candidateLimit; j++)
						{
							uint32		candidate = pool->neighbors[j];
							bool		seen = false;

							if (candidate >= meta->tqNodeCount || candidate == source ||
								deadNodes[candidate] || nodeLevels[candidate] < level)
								continue;
							for (int k = 0; k < candidateCount; k++)
								if (candidates[k].nodeId == candidate)
									seen = true;
							if (seen)
								continue;
							candidates[candidateCount].nodeId = candidate;
							candidates[candidateCount].distance =
								PgturbohybridGraphVacuumCodeDistance(&storage, meta,
															 source, candidate);
							candidateCount++;
						}
					}
				}
				qsort(candidates, candidateCount, sizeof(*candidates),
					  PgturbohybridGraphVacuumCandidateCompare);
				for (int i = 0; i < candidateCount; i++)
				{
					bool		diverse = true;

					/* Same relative-neighbor diversity rule as construction:
					 * reject a candidate shadowed by an already selected live
					 * neighbor. */
					for (int j = 0; j < sourceAdj->count; j++)
					{
						uint32		selected = sourceAdj->neighbors[j];

						if (selected == deadNode || selected >= meta->tqNodeCount ||
							deadNodes[selected])
							continue;
						if (PgturbohybridGraphVacuumCodeDistance(&storage, meta,
										 candidates[i].nodeId, selected) <
							candidates[i].distance)
						{
							diverse = false;
							break;
						}
					}
					if (diverse && PgturbohybridGraphVacuumInstallReciprocal(allAdj,
							deadNodes, meta->tqNodeCount, levelCapacity, level,
							capacity, source,
							deadNode, candidates[i].nodeId))
					{
						*repairedEdges += 2;
						break;
					}
				}
				pfree(candidates);
			}
		}
	}

	for (uint32 nodeId = 0; nodeId < meta->tqNodeCount; nodeId++)
		for (int level = 0; level <= nodeLevels[nodeId] && level < levelCapacity; level++)
		{
			PgturbohybridGraphVacuumAdj *adj =
				&allAdj[(Size) nodeId * levelCapacity + level];
			PgturbohybridGraphVacuumAdj *original =
				&originalAdj[(Size) nodeId * levelCapacity + level];

			CHECK_FOR_INTERRUPTS();
			if (adj->count != original->count ||
				memcmp(adj->neighbors, original->neighbors,
					   sizeof(uint32) * adj->count) != 0)
			{
				PgturbohybridGraphVacuumUpdateAdjTuple(index, meta, nodeId, level,
													  adj->neighbors, adj->count);
				changedAny = true;
			}
		}

	return changedAny;
}

void
PgturbohybridGraphCollectVacuumStats(Relation index, PgturbohybridGraphMetaPageData *meta,
						  int64 *liveNodes, int64 *deadNodes,
						  int64 *adjacencyRefs, int64 *deadNeighborRefs,
						  int64 *deadBridgeNodes,
						  double *avgLiveDegreeLevel0,
						  uint32 *liveEntryNode)
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
	*deadBridgeNodes = 0;
	*avgLiveDegreeLevel0 = 0.0;
	*liveEntryNode = UINT_MAX;

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
			{
				(*liveNodes)++;
				if (*liveEntryNode == UINT_MAX)
					*liveEntryNode = tuple->nodeId;
			}
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
			bool		hasNavigableEdge = false;

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
				if (neighbor >= meta->tqNodeCount || deadBitmap[neighbor])
					(*deadNeighborRefs)++;
				if (neighbor < meta->tqNodeCount)
					hasNavigableEdge = true;
			}
			if (tuple->level == 0 && !deadSource)
				*avgLiveDegreeLevel0 += Min(tuple->count, maxNeighbors);
			if (tuple->level == 0 && deadSource && hasNavigableEdge)
				(*deadBridgeNodes)++;
		}

		UnlockReleaseBuffer(buf);
		if (nextblkno == blkno)
			break;
		blkno = nextblkno;
	}

	if (meta->tqEntryNodeId < meta->tqNodeCount &&
		!deadBitmap[meta->tqEntryNodeId])
		*liveEntryNode = meta->tqEntryNodeId;
	pfree(deadBitmap);
	if (*liveNodes > 0)
		*avgLiveDegreeLevel0 /= (double) *liveNodes;
}

typedef struct PgturbohybridGraphBulkDeleteState
{
	double		liveTuples;
	bool		changedAny;
	bool		repairAny;
	bool		hasDeadNodes;
	uint64		repairedEdges;
} PgturbohybridGraphBulkDeleteState;

static uint32
PgturbohybridGraphVacuumBestLiveNode(bool *deadNodes, uint8 *nodeLevels,
									 uint32 startNodeId, uint32 nodeCount)
{
	uint32		best = UINT_MAX;
	int			bestLevel = -1;
	uint64		endNodeId = (uint64) startNodeId + nodeCount;

	for (uint32 nodeId = startNodeId;
		 nodeId < endNodeId && nodeId < PG_UINT32_MAX; nodeId++)
	{
		if (deadNodes[nodeId])
			continue;
		if ((int) nodeLevels[nodeId] > bestLevel)
		{
			best = nodeId;
			bestLevel = nodeLevels[nodeId];
		}
	}
	return best;
}

static bool
PgturbohybridGraphVacuumRepairEntries(Relation index,
									  PgturbohybridGraphMetaPageData *meta,
									  bool *deadNodes, uint8 *nodeLevels)
{
	Buffer		buf;
	Page		page;
	PgturbohybridGraphMetaPage metap;
	GenericXLogState *xlogState = NULL;
	uint32		globalEntry;
	bool		changed = false;

	globalEntry = PgturbohybridGraphVacuumBestLiveNode(deadNodes, nodeLevels,
														 0, meta->tqNodeCount);
	if (globalEntry == UINT_MAX)
		return false;

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

	if (metap->tqEntryNodeId >= meta->tqNodeCount ||
		deadNodes[metap->tqEntryNodeId])
	{
		metap->tqEntryNodeId = globalEntry;
		metap->entryLevel = nodeLevels[globalEntry];
		changed = true;
	}
	for (uint16 i = 0; i < metap->tqSegmentCount; i++)
	{
		PgturbohybridGraphSegmentMetaData *segment = &metap->tqSegments[i];
		uint32		entry = segment->entryNodeId;

		if (entry < meta->tqNodeCount && !deadNodes[entry])
			continue;
		entry = PgturbohybridGraphVacuumBestLiveNode(deadNodes, nodeLevels,
														 segment->startNodeId,
														 segment->nodeCount);
		segment->entryNodeId = entry;
		segment->entryLevel = entry == UINT_MAX ? -1 : nodeLevels[entry];
		changed = true;
	}

	/* Routing and sidecar entries are return-oriented seeds, so replace dead
	 * values with the deterministic live entry instead of retaining them. */
	for (uint16 i = 0; i < metap->tqRoutingEntryCount; i++)
		if (deadNodes[metap->tqRoutingEntryNodeIds[i]])
		{
			metap->tqRoutingEntryNodeIds[i] = globalEntry;
			changed = true;
		}
	for (uint16 i = 0; i < metap->tqEntrySidecarCount; i++)
		if (deadNodes[metap->tqEntrySidecarNodeIds[i]])
		{
			metap->tqEntrySidecarNodeIds[i] = globalEntry;
			changed = true;
		}

	if (changed)
		PgturbohybridGraphMarkPageGraphOp(page,
										 PGTURBOHYBRID_GRAPH_GRAPH_OP_VACUUM_REPAIR);
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
		PgturbohybridGraphLogGraphWalRecord(index, MAIN_FORKNUM,
											PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO,
											PGTURBOHYBRID_GRAPH_GRAPH_OP_VACUUM_REPAIR);
	return changed;
}

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
	uint8	   *nodeLevels = NULL;

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
	nodeLevels = palloc0(sizeof(uint8) * meta.tqNodeCount);

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

				nodeLevels[tuple->nodeId] = tuple->level;
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
				PgturbohybridGraphVacuumTestPause("after_delete_page");
			}

			blkno = nextblkno;
		}

		if (deleteState->hasDeadNodes)
		{
			deleteState->repairAny =
				PgturbohybridGraphRepairAdjacencyForDeadNodes(index, &meta,
													 deadNodes, nodeLevels,
													 &deleteState->repairedEdges);
			deleteState->repairAny =
				PgturbohybridGraphVacuumRepairEntries(index, &meta, deadNodes,
												 nodeLevels) || deleteState->repairAny;
		}

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
	pfree(nodeLevels);
	result->num_index_tuples = deleteState->liveTuples;
	if (meta.tqNodeCount > 0 &&
		((meta.tqNodeCount - deleteState->liveTuples) / meta.tqNodeCount) >=
		PGTURBOHYBRID_GRAPH_REINDEX_DEAD_RATIO)
		ereport(WARNING,
				(errmsg("pgturbohybrid index \"%s\" has accumulated dead graph nodes",
						RelationGetRelationName(index)),
				 errdetail("Dead node ratio is at least %.2f; VACUUM repaired %llu directed edges but append-only node storage is not compacted.",
						   PGTURBOHYBRID_GRAPH_REINDEX_DEAD_RATIO,
						   (unsigned long long) deleteState->repairedEdges),
				 errhint("REINDEX the index to compact dead nodes.")));
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
	stats->pages_free = 0;

	return stats;
}
