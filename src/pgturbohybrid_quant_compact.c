/*
 * pgturbohybrid_quant_compact.c - AM page chain compaction for dead graph nodes
 *
 * After tqgraphbulkdelete marks dead nodes, this function optionally rewrites
 * the code, adjacency, and exact slab chains to remove dead-node bloat.
 * Old pages are abandoned (never freed) per standard PostgreSQL extension
 * behavior.
 */

#include "postgres.h"

#include <string.h>

#include "access/generic_xlog.h"
#include "access/stratnum.h"
#include "commands/vacuum.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/rel.h"

#include "pgturbohybrid.h"
#include "pgturbohybrid_quant.h"

#define PGTURBOHYBRID_COMPACT_INVALID_NID	UINT32_MAX

/*
 * Local helper: read the page_compaction_threshold reloption.
 * Returns 0 when the option is absent (disabled).
 */
static int
PgturbohybridGraphGetPageCompactionThreshold(Relation index)
{
	TqOptions *opts = (TqOptions *) index->rd_options;

	if (opts == NULL)
		return 25;			/* default when no reloptions */

	/*
	 * The field sits at the end of TqOptions.  It is added by the
	 * page_compaction_threshold reloption (add_int_reloption).
	 * When the relation was created before the option existed the struct
	 * tail may be truncated, so guard against out-of-bounds access by
	 * checking the varlena length.
	 */
	if (opts->vl_len_ < (int32) offsetof(TqOptions, pageCompactionThreshold) + sizeof(int))
		return 25;					/* sensible default for upgraded indexes */

	return opts->pageCompactionThreshold;
}

/*
 * Phase 1: Walk the code chain, count dead vs live nodes, and build the
 * deadNodes[] bitmap and nodeIdMap[] (old nodeId -> new sequential ID).
 *
 * Returns the number of live nodes (0 means nothing to compact).
 */
static uint32
PgturbohybridGraphCompactPhase1BloatCheck(Relation index,
										  PgturbohybridGraphMetaPageData *meta,
										  bool **outDeadNodes,
										  uint32 **outNodeIdMap,
										  int threshold)
{
	uint32		nodeCount = meta->tqNodeCount;
	int			tqBits = meta->tqBits != 0 ? meta->tqBits : PGTURBOHYBRID_DEFAULT_BITS;
	bool		tqWeighted = (meta->tqFlags & PGTURBOHYBRID_GRAPH_TQ_WEIGHTED) != 0;
	Size		codeTupleSize = PgturbohybridGraphCodeTupleSize(meta->dimensions,
																meta->tqPayloadCount,
																tqBits, tqWeighted,
																meta->tqResidualRerankBytes);
	int			codeTuplesPerPage = PgturbohybridGraphTuplesPerPage(codeTupleSize);
	int			codePageCount = PgturbohybridGraphPageCount(nodeCount, codeTuplesPerPage);
	BlockNumber nblocks = RelationGetNumberOfBlocks(index);
	BlockNumber blkno = meta->tqCodeStartBlkno;
	uint32		deadCount = 0;
	uint32		liveCount = 0;
	bool	   *deadNodes;
	uint32	   *nodeIdMap;
	uint32		nextLive = 0;

	deadNodes = palloc0(sizeof(bool) * nodeCount);
	nodeIdMap = palloc(sizeof(uint32) * nodeCount);

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
				tuple->nodeId >= nodeCount)
				continue;

			if (tuple->flags & PGTURBOHYBRID_GRAPH_NODE_DEAD)
			{
				deadNodes[tuple->nodeId] = true;
				deadCount++;
			}
			else
				liveCount++;
		}

		blkno = opaque->nextblkno;
		UnlockReleaseBuffer(buf);
	}

	/* Build the remap table */
	for (uint32 i = 0; i < nodeCount; i++)
		nodeIdMap[i] = deadNodes[i] ? PGTURBOHYBRID_COMPACT_INVALID_NID : nextLive++;

	*outDeadNodes = deadNodes;
	*outNodeIdMap = nodeIdMap;
	return liveCount;
}

/*
 * Phase 2: Rewrite the code chain.  For each live code tuple, copy it,
 * remap nodeId, and append to the new chain.
 *
 * Returns the new code chain start block number.  Also fills
 * exactBlknoMap[] / exactOffnoMap[] (indexed by old nodeId) so Phase 4
 * can look up where each live node's exact data was written.
 */
static BlockNumber
PgturbohybridGraphCompactPhase2RewriteCode(Relation index,
											PgturbohybridGraphMetaPageData *meta,
											bool *deadNodes,
											uint32 *nodeIdMap,
											uint32 liveCount,
											uint32 **outExactBlknoMap,
											OffsetNumber **outExactOffnoMap)
{
	uint32		nodeCount = meta->tqNodeCount;
	int			tqBits = meta->tqBits != 0 ? meta->tqBits : PGTURBOHYBRID_DEFAULT_BITS;
	bool		tqWeighted = (meta->tqFlags & PGTURBOHYBRID_GRAPH_TQ_WEIGHTED) != 0;
	Size		codeTupleSize = PgturbohybridGraphCodeTupleSize(meta->dimensions,
																meta->tqPayloadCount,
																tqBits, tqWeighted,
																meta->tqResidualRerankBytes);
	int			codeTuplesPerPage = PgturbohybridGraphTuplesPerPage(codeTupleSize);
	int			codePageCount = PgturbohybridGraphPageCount(nodeCount, codeTuplesPerPage);
	BlockNumber nblocks = RelationGetNumberOfBlocks(index);
	BlockNumber blkno = meta->tqCodeStartBlkno;
	Buffer		buf = InvalidBuffer;
	Page		page = NULL;
	BlockNumber start = InvalidBlockNumber;
	PgturbohybridGraphCodeTuple newTuple = palloc(codeTupleSize);
	uint32	   *exactBlknoMap = palloc0(sizeof(uint32) * nodeCount);
	OffsetNumber *exactOffnoMap = palloc0(sizeof(OffsetNumber) * nodeCount);
	BufferAccessStrategy strategy = GetAccessStrategy(BAS_BULKREAD);

	for (int pageNo = 0;
		 pageNo < codePageCount && BlockNumberIsValid(blkno) && blkno < nblocks;
		 pageNo++)
	{
		Buffer		oldBuf;
		Page		oldPage;
		PgturbohybridGraphPageOpaque opaque;
		OffsetNumber maxoff;

		CHECK_FOR_INTERRUPTS();
		PGTURBOHYBRID_VACUUM_DELAY_POINT();

		oldBuf = ReadBufferExtended(index, MAIN_FORKNUM, blkno, RBM_NORMAL, strategy);
		LockBuffer(oldBuf, BUFFER_LOCK_SHARE);
		oldPage = BufferGetPage(oldBuf);
		opaque = PgturbohybridGraphPageGetOpaque(oldPage);

		if ((opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) != PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_CODE)
		{
			UnlockReleaseBuffer(oldBuf);
			break;
		}

		maxoff = PageGetMaxOffsetNumber(oldPage);
		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
		{
			ItemId		iid = PageGetItemId(oldPage, offno);
			PgturbohybridGraphCodeTuple tuple;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (PgturbohybridGraphCodeTuple) PageGetItem(oldPage, iid);
			if (tuple->type != PGTURBOHYBRID_GRAPH_CODE_TUPLE_TYPE ||
				tuple->nodeId >= nodeCount)
				continue;

			if (deadNodes[tuple->nodeId])
				continue;

			/* Ensure new page has space */
			if (!BufferIsValid(buf) || PageGetFreeSpace(page) < codeTupleSize)
			{
				PgturbohybridGraphAppendPage(index, MAIN_FORKNUM, &buf, &page,
											 PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_CODE);
				if (!BlockNumberIsValid(start))
					start = BufferGetBlockNumber(buf);
			}

			/* Copy the tuple */
			memcpy(newTuple, tuple, codeTupleSize);
			newTuple->nodeId = nodeIdMap[tuple->nodeId];

			if (PageAddItem(page, (Item) newTuple, codeTupleSize,
							InvalidOffsetNumber, false, false) == InvalidOffsetNumber)
				elog(ERROR, "failed to add pgturbohybrid graph code item during compaction for \"%s\"",
					 RelationGetRelationName(index));

			PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT);

			/*
			 * Remember the exact-vector pointer for this node so Phase 4
			 * can locate it.  The exactBlkno/exactOffno will be rewritten
			 * after Phase 4 produces the new exact chain.
			 */
			exactBlknoMap[tuple->nodeId] = tuple->exactBlkno;
			exactOffnoMap[tuple->nodeId] = tuple->exactOffno;
		}

		blkno = opaque->nextblkno;
		UnlockReleaseBuffer(oldBuf);
	}

	if (BufferIsValid(buf))
		PgturbohybridGraphFinishPage(buf);

	pfree(newTuple);
	*outExactBlknoMap = exactBlknoMap;
	*outExactOffnoMap = exactOffnoMap;
	return start;
}

/*
 * Phase 3: Rewrite the adjacency chain.  For each adj tuple whose nodeId
 * maps to a live node, copy it, remap nodeId, remove dead neighbors, and
 * remap live neighbor IDs.
 */
static BlockNumber
PgturbohybridGraphCompactPhase3RewriteAdj(Relation index,
										  PgturbohybridGraphMetaPageData *meta,
										  bool *deadNodes,
										  uint32 *nodeIdMap,
										  uint32 liveCount)
{
	uint32		nodeCount = meta->tqNodeCount;
	BlockNumber nblocks = RelationGetNumberOfBlocks(index);
	BlockNumber blkno = meta->tqAdjStartBlkno;
	Buffer		buf = InvalidBuffer;
	Page		page = NULL;
	BlockNumber start = InvalidBlockNumber;
	int			maxNeighbors = PgturbohybridGraphLevelM(meta->m, 0);
	Size		maxTupleSize = PgturbohybridGraphAdjTupleSize(maxNeighbors);
	PgturbohybridGraphAdjTuple newTuple = palloc(maxTupleSize);
	BufferAccessStrategy strategy = GetAccessStrategy(BAS_BULKREAD);

	while (BlockNumberIsValid(blkno) && blkno < nblocks)
	{
		Buffer		oldBuf;
		Page		oldPage;
		PgturbohybridGraphPageOpaque opaque;
		OffsetNumber maxoff;

		CHECK_FOR_INTERRUPTS();
		PGTURBOHYBRID_VACUUM_DELAY_POINT();

		oldBuf = ReadBufferExtended(index, MAIN_FORKNUM, blkno, RBM_NORMAL, strategy);
		LockBuffer(oldBuf, BUFFER_LOCK_SHARE);
		oldPage = BufferGetPage(oldBuf);
		opaque = PgturbohybridGraphPageGetOpaque(oldPage);

		if ((opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) != PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_ADJ)
		{
			UnlockReleaseBuffer(oldBuf);
			break;
		}

		maxoff = PageGetMaxOffsetNumber(oldPage);
		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
		{
			ItemId		iid = PageGetItemId(oldPage, offno);
			PgturbohybridGraphAdjTuple tuple;
			uint32		oldNodeId;
			uint32		newNodeId;
			uint16		level;
			uint16		newCount = 0;
			Size		tupleSize;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (PgturbohybridGraphAdjTuple) PageGetItem(oldPage, iid);
			if (tuple->type != PGTURBOHYBRID_GRAPH_ADJ_TUPLE_TYPE ||
				tuple->nodeId >= nodeCount)
				continue;

			oldNodeId = tuple->nodeId;
			if (deadNodes[oldNodeId])
				continue;

			newNodeId = nodeIdMap[oldNodeId];
			level = tuple->level;

			/* Remap neighbors: drop dead, remap live */
			for (uint16 j = 0; j < tuple->count; j++)
			{
				uint32 nbr = tuple->neighbors[j];

				if (nbr >= nodeCount || deadNodes[nbr])
					continue;

				newTuple->neighbors[newCount++] = nodeIdMap[nbr];
			}

			tupleSize = PgturbohybridGraphAdjTupleSize(newCount);

			if (!BufferIsValid(buf) || PageGetFreeSpace(page) < tupleSize)
			{
				PgturbohybridGraphAppendPage(index, MAIN_FORKNUM, &buf, &page,
											 PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_ADJ);
				if (!BlockNumberIsValid(start))
					start = BufferGetBlockNumber(buf);
			}

			memset(newTuple, 0, maxTupleSize);
			newTuple->type = PGTURBOHYBRID_GRAPH_ADJ_TUPLE_TYPE;
			newTuple->level = level;
			newTuple->count = newCount;
			newTuple->nodeId = newNodeId;

			if (PageAddItem(page, (Item) newTuple, tupleSize,
							InvalidOffsetNumber, false, false) == InvalidOffsetNumber)
				elog(ERROR, "failed to add pgturbohybrid graph adj item during compaction for \"%s\"",
					 RelationGetRelationName(index));

			PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_NEIGHBOR_INSERT);
		}

		blkno = opaque->nextblkno;
		UnlockReleaseBuffer(oldBuf);
	}

	if (BufferIsValid(buf))
		PgturbohybridGraphFinishPage(buf);

	pfree(newTuple);
	return start;
}

/*
 * Phase 4: Rewrite the exact slab chain.  For each live node, copy its
 * exact vector bytes into the new slab chain, and record the new location
 * so the code chain back-pointers can be updated.
 *
 * The exact chain uses slab format: each page contains a single large item
 * (PgturbohybridGraphExactSlabPageHeaderData) with magic/used/capacity and
 * a data[] payload.
 *
 * Returns the new exact chain start.  Fills newExactBlkno[] and
 * newExactOffno[] (indexed by old nodeId) for live nodes.
 */
static BlockNumber
PgturbohybridGraphCompactPhase4RewriteExact(Relation index,
											PgturbohybridGraphMetaPageData *meta,
											bool *deadNodes,
											uint32 *nodeIdMap,
											uint32 liveCount,
											uint32 *oldExactBlknoMap,
											OffsetNumber *oldExactOffnoMap,
											uint32 *newExactBlkno,
											OffsetNumber *newExactOffno)
{
	uint32		nodeCount = meta->tqNodeCount;
	Size		vectorSize;
	Buffer		buf = InvalidBuffer;
	Page		page = NULL;
	BlockNumber start = InvalidBlockNumber;
	BufferAccessStrategy strategy = GetAccessStrategy(BAS_BULKREAD);

	/* Exact-free indexes skip this phase */
	if ((meta->tqFlags & PGTURBOHYBRID_GRAPH_EXACT_FREE) != 0 ||
		!BlockNumberIsValid(meta->tqExactStartBlkno))
		return InvalidBlockNumber;

	vectorSize = PGTURBOHYBRID_VECTOR_SIZE(meta->dimensions);

	for (uint32 oldNid = 0; oldNid < nodeCount; oldNid++)
	{
		BlockNumber	blkno;
		OffsetNumber offno;
		Size		copied = 0;

		if (deadNodes[oldNid])
			continue;

		if (!BlockNumberIsValid(oldExactBlknoMap[oldNid]) ||
			!PgturbohybridGraphExactByteOffsetIsValid(oldExactOffnoMap[oldNid]))
			continue;

		CHECK_FOR_INTERRUPTS();
		PGTURBOHYBRID_VACUUM_DELAY_POINT();

		blkno = oldExactBlknoMap[oldNid];
		offno = oldExactOffnoMap[oldNid];

		while (copied < vectorSize)
		{
			PgturbohybridGraphExactSlabPageHeader header;
			Size		capacity;
			Size		offset;
			Size		available;
			Size		chunk;
			Buffer		oldBuf;
			Page		oldPage;

			if (!BufferIsValid(blkno))
				break;

			oldBuf = ReadBufferExtended(index, MAIN_FORKNUM, blkno, RBM_NORMAL, strategy);
			LockBuffer(oldBuf, BUFFER_LOCK_SHARE);
			oldPage = BufferGetPage(oldBuf);

			if (!PgturbohybridGraphExactPageIsSlab(oldPage))
			{
				UnlockReleaseBuffer(oldBuf);
				break;
			}

			header = (PgturbohybridGraphExactSlabPageHeader)
				PageGetItem(oldPage, PageGetItemId(oldPage, FirstOffsetNumber));
			offset = (Size) offno - 1;

			if (offset >= header->used)
			{
				UnlockReleaseBuffer(oldBuf);
				break;
			}

			available = header->used - offset;
			chunk = Min(vectorSize - copied, available);

			/* Ensure new slab page is ready */
			if (!BufferIsValid(buf))
			{
				PgturbohybridGraphAppendPage(index, MAIN_FORKNUM, &buf, &page,
											 PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_EXACT);
				PgturbohybridGraphInitExactSlabPage(page);
				if (!BlockNumberIsValid(start))
					start = BufferGetBlockNumber(buf);
			}

			{
				PgturbohybridGraphExactSlabPageHeader newHeader =
					(PgturbohybridGraphExactSlabPageHeader)
					PageGetItem(page, PageGetItemId(page, FirstOffsetNumber));

				capacity = newHeader->capacity;
				if (newHeader->used >= capacity)
				{
					PgturbohybridGraphAppendPage(index, MAIN_FORKNUM, &buf, &page,
												 PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_EXACT);
					PgturbohybridGraphInitExactSlabPage(page);
					newHeader = (PgturbohybridGraphExactSlabPageHeader)
						PageGetItem(page, PageGetItemId(page, FirstOffsetNumber));
					capacity = newHeader->capacity;
				}

				if (copied == 0)
				{
					newExactBlkno[oldNid] = BufferGetBlockNumber(buf);
					newExactOffno[oldNid] = (OffsetNumber) (newHeader->used + 1);
				}

				memcpy(newHeader->data + newHeader->used,
					   header->data + offset, chunk);
				newHeader->used += chunk;
				PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT);
			}

			copied += chunk;
			offno = InvalidOffsetNumber;	/* continue from start of next page */
			blkno = ((PgturbohybridGraphPageOpaque)
					 PageGetSpecialPointer(oldPage))->nextblkno;
			UnlockReleaseBuffer(oldBuf);
		}
	}

	if (BufferIsValid(buf))
		PgturbohybridGraphFinishPage(buf);

	return start;
}

/*
 * Top-level entry point: compact graph page chains by removing dead nodes.
 *
 * Called from amvacuumcleanup after tqgraphbulkdelete has marked dead nodes.
 * The function is a no-op if nodeCount == 0, no chains exist, or the
 * page_compaction_threshold reloption is 0 (disabled).
 */
void
PgturbohybridGraphMaybeCompactPageChains(Relation index)
{
	PgturbohybridGraphMetaPageData meta;
	bool	   *deadNodes = NULL;
	uint32	   *nodeIdMap = NULL;
	uint32		liveCount;
	uint32	   *oldExactBlknoMap = NULL;
	OffsetNumber *oldExactOffnoMap = NULL;
	uint32	   *newExactBlkno = NULL;
	OffsetNumber *newExactOffno = NULL;
	BlockNumber newCodeStart;
	BlockNumber newAdjStart;
	BlockNumber newExactStart;
	int			threshold;

	/* Read metapage; bail if not a quant graph or empty */
	if (!PgturbohybridGraphReadMeta(index, &meta))
		return;

	if (meta.tqNodeCount == 0 ||
		!BlockNumberIsValid(meta.tqCodeStartBlkno))
		return;

	threshold = PgturbohybridGraphGetPageCompactionThreshold(index);
	if (threshold <= 0)
		return;

	/* ---------- Phase 1: bloat check ---------- */
	liveCount = PgturbohybridGraphCompactPhase1BloatCheck(index, &meta,
														  &deadNodes, &nodeIdMap,
														  threshold);

	{
		bool	hasDeadNodes = (liveCount < meta.tqNodeCount);
		int			tqBits = meta.tqBits != 0 ? meta.tqBits : PGTURBOHYBRID_DEFAULT_BITS;
		bool	tqWeighted = (meta.tqFlags & PGTURBOHYBRID_GRAPH_TQ_WEIGHTED) != 0;
		Size		codeTupleSize;
		int			codeTuplesPerPage;
		int			expectedPages;
		int			actualPages = 0;
		BlockNumber nblocks = RelationGetNumberOfBlocks(index);
		BlockNumber blkno = meta.tqCodeStartBlkno;
		bool	pageBloat;

		codeTupleSize = PgturbohybridGraphCodeTupleSize(meta.dimensions,
						meta.tqPayloadCount,
						tqBits, tqWeighted,
						meta.tqResidualRerankBytes);
		codeTuplesPerPage = PgturbohybridGraphTuplesPerPage(codeTupleSize);
		expectedPages = PgturbohybridGraphPageCount(liveCount, codeTuplesPerPage);

		/* Count actual code chain pages by walking the linked list */
		while (BlockNumberIsValid(blkno) && blkno < nblocks)
		{
			Buffer		buf;
			Page			page;
			PgturbohybridGraphPageOpaque opaque;

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

			actualPages++;
			blkno = opaque->nextblkno;
			UnlockReleaseBuffer(buf);
		}

		pageBloat = (actualPages >= Max(expectedPages * 3, expectedPages + 2));

		if (!hasDeadNodes && !pageBloat)
		{
			/* No dead nodes and no page bloat -- nothing to compact */
			pfree(deadNodes);
			pfree(nodeIdMap);
			return;
		}

		if (hasDeadNodes && !pageBloat &&
			meta.tqNodeCount > 0 &&
			(meta.tqNodeCount - liveCount) * 100 < meta.tqNodeCount * threshold)
		{
			/* Dead ratio below threshold and no page bloat -- skip compaction */
			pfree(deadNodes);
			pfree(nodeIdMap);
			return;
		}

		/*
		 * Multivector indexes with a docmap sidecar can only undergo
		 * page-bloat compaction.  When there are no dead nodes the
		 * nodeIdMap is an identity mapping (nodeIdMap[i] == i) and
		 * liveCount == tqNodeCount, so the docmap's nodeId references
		 * stay valid and the rewrite is safe.  Dead-node compaction
		 * would remap nodeIds and reduce tqNodeCount, corrupting the
		 * docmap's node/doc tuple entries — skip it and let REINDEX
		 * handle full compaction for these indexes.
		 */
		if (hasDeadNodes && BlockNumberIsValid(meta.tqMultivectorDocMapStartBlkno))
		{
			pfree(deadNodes);
			pfree(nodeIdMap);
			return;
		}
	}

	ereport(LOG,
			(errmsg("pgturbohybrid graph compaction: removing %u of %u nodes from \"%s\"",
					meta.tqNodeCount - liveCount, meta.tqNodeCount,
					RelationGetRelationName(index))));

	/* Acquire locks for the rewrite */
	LockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, ExclusiveLock);
	LockPage(index, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ExclusiveLock);

	PG_TRY();
	{
		/* ---------- Phase 2: rewrite code chain ---------- */
		newCodeStart = PgturbohybridGraphCompactPhase2RewriteCode(index, &meta,
																 deadNodes, nodeIdMap,
																 liveCount,
																 &oldExactBlknoMap,
																 &oldExactOffnoMap);

		/* ---------- Phase 3: rewrite adjacency chain ---------- */
		newAdjStart = PgturbohybridGraphCompactPhase3RewriteAdj(index, &meta,
																deadNodes, nodeIdMap,
																liveCount);

		/* ---------- Phase 4: rewrite exact slab chain ---------- */
		newExactBlkno = palloc0(sizeof(uint32) * meta.tqNodeCount);
		newExactOffno = palloc0(sizeof(OffsetNumber) * meta.tqNodeCount);

		newExactStart = PgturbohybridGraphCompactPhase4RewriteExact(index, &meta,
																	 deadNodes, nodeIdMap,
																	 liveCount,
																	 oldExactBlknoMap,
																	 oldExactOffnoMap,
																	 newExactBlkno,
																	 newExactOffno);

		/*
		 * ---------- Phase 5: atomic metapage swap ----------
		 *
		 * Build the PgturbohybridQuantMetaUpdate from the current metapage,
		 * then override only the fields that change.
		 */
		{
			PgturbohybridQuantMetaUpdate update;
			uint32		newEntryNodeId = PGTURBOHYBRID_COMPACT_INVALID_NID;
			int16		newEntryLevel = -1;
			uint16		newSegmentCount = 0;
			PgturbohybridGraphSegmentMetaData newSegments[PGTURBOHYBRID_GRAPH_MAX_NATIVE_SEGMENTS];
			uint16		newEntrySidecarCount = 0;
			uint32		newEntrySidecarNodeIds[PGTURBOHYBRID_GRAPH_MAX_ENTRY_SIDECAR_REPRESENTATIVES];
			uint16		newRoutingEntryCount = 0;
			uint32		newRoutingEntryNodeIds[PGTURBOHYBRID_GRAPH_MAX_ROUTING_ENTRIES];
			int			i;

			memset(&update, 0, sizeof(update));
			memset(newSegments, 0, sizeof(newSegments));
			memset(newEntrySidecarNodeIds, 0xff, sizeof(newEntrySidecarNodeIds));
			memset(newRoutingEntryNodeIds, 0xff, sizeof(newRoutingEntryNodeIds));

			/* Copy all fields from current metapage */
			update.forkNum = MAIN_FORKNUM;
			update.building = false;
			update.dimensions = meta.dimensions;
			update.m = meta.m;
			update.efConstruction = meta.efConstruction;
			update.graphMaxLevel = meta.graphMaxLevel;
			update.tqBits = meta.tqBits;
			update.tqPayloadCount = meta.tqPayloadCount;
			update.tqPayloadBytes = meta.tqPayloadBytes;
			update.tqFlags = meta.tqFlags;
			update.tqResidualRerankBytes = meta.tqResidualRerankBytes;
			update.tqMultivectorDocMapStartBlkno = meta.tqMultivectorDocMapStartBlkno;
			update.tqMultivectorDocMapPageCount = meta.tqMultivectorDocMapPageCount;
			update.tqMultivectorDocCount = meta.tqMultivectorDocCount;
			update.tqMultivectorDocMapBytes = meta.tqMultivectorDocMapBytes;
			update.tqMultivectorDocMapVersion = meta.tqMultivectorDocMapVersion;
			update.tqMultivectorDocMapFlags = meta.tqMultivectorDocMapFlags;
			update.tqMultivectorGraphMode = meta.tqMultivectorGraphMode;
			update.buildScanUs = meta.buildScanUs;
			update.buildCorrectionUs = meta.buildCorrectionUs;
			update.buildEncodeUs = meta.buildEncodeUs;
			update.buildEdgeUs = meta.buildEdgeUs;
			update.buildWriteUs = meta.buildWriteUs;
			update.buildWorkerCount = meta.buildWorkerCount;

			/* Override node count */
			update.nodeCount = liveCount;

			/* Remap entry node */
			if (meta.tqEntryNodeId < meta.tqNodeCount && !deadNodes[meta.tqEntryNodeId])
			{
				newEntryNodeId = nodeIdMap[meta.tqEntryNodeId];
				newEntryLevel = meta.entryLevel;
			}
			update.entryNodeId = newEntryNodeId;
			update.entryLevel = newEntryLevel;

			/* Remap segments */
			for (i = 0; i < meta.tqSegmentCount; i++)
			{
				PgturbohybridGraphSegmentMetaData *seg = &meta.tqSegments[i];
				uint32	segLiveCount = 0;

				/* Count live nodes in this segment */
				for (uint32 n = seg->startNodeId;
					 n < seg->startNodeId + seg->nodeCount && n < meta.tqNodeCount;
					 n++)
				{
					if (!deadNodes[n])
						segLiveCount++;
				}

				if (segLiveCount > 0)
				{
					newSegments[newSegmentCount] = *seg;
					newSegments[newSegmentCount].startNodeId =
						nodeIdMap[seg->startNodeId < meta.tqNodeCount ? seg->startNodeId : 0];
					newSegments[newSegmentCount].nodeCount = segLiveCount;

					if (seg->entryNodeId < meta.tqNodeCount && !deadNodes[seg->entryNodeId])
						newSegments[newSegmentCount].entryNodeId = nodeIdMap[seg->entryNodeId];
					else
						newSegments[newSegmentCount].entryNodeId = UINT32_MAX;

					newSegmentCount++;
				}
			}
			update.tqSegmentCount = newSegmentCount;
			memcpy(update.tqSegments, newSegments, sizeof(newSegments));

			/* Remap entry sidecar */
			newEntrySidecarCount = 0;
			for (i = 0; i < meta.tqEntrySidecarCount; i++)
			{
				uint32 nid = meta.tqEntrySidecarNodeIds[i];

				if (nid < meta.tqNodeCount && !deadNodes[nid])
					newEntrySidecarNodeIds[newEntrySidecarCount++] = nodeIdMap[nid];
			}
			update.tqEntrySidecarCount = newEntrySidecarCount;
			update.tqEntrySidecarBytes = newEntrySidecarCount * sizeof(uint32);
			memcpy(update.tqEntrySidecarNodeIds, newEntrySidecarNodeIds,
				   sizeof(newEntrySidecarNodeIds));

			/* Remap routing entries */
			newRoutingEntryCount = 0;
			for (i = 0; i < meta.tqRoutingEntryCount; i++)
			{
				uint32 nid = meta.tqRoutingEntryNodeIds[i];

				if (nid < meta.tqNodeCount && !deadNodes[nid])
					newRoutingEntryNodeIds[newRoutingEntryCount++] = nodeIdMap[nid];
			}
			update.tqRoutingEntryCount = newRoutingEntryCount;
			update.tqRoutingEntryBytes = newRoutingEntryCount * sizeof(uint32);
			memcpy(update.tqRoutingEntryNodeIds, newRoutingEntryNodeIds,
				   sizeof(newRoutingEntryNodeIds));

			/*
			 * Write the new metapage.  The correction chain is left
			 * untouched -- its startBlkno is copied as-is from the old
			 * metapage via the update struct defaults.
			 */
			PgturbohybridQuantUpdateMetaPageFromUpdate(index, &update,
														newCodeStart, newAdjStart,
														newExactStart,
														meta.tqCorrectionStartBlkno);
		}

		/* Invalidate per-backend scan caches */
		PgturbohybridGraphInvalidateCaches(index);
	}
	PG_CATCH();
	{
		UnlockPage(index, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ExclusiveLock);
		UnlockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, ExclusiveLock);
		PG_RE_THROW();
	}
	PG_END_TRY();

	UnlockPage(index, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ExclusiveLock);
	UnlockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, ExclusiveLock);

	pfree(deadNodes);
	pfree(nodeIdMap);
	if (oldExactBlknoMap)
		pfree(oldExactBlknoMap);
	if (oldExactOffnoMap)
		pfree(oldExactOffnoMap);
	if (newExactBlkno)
		pfree(newExactBlkno);
	if (newExactOffno)
		pfree(newExactOffno);
}
