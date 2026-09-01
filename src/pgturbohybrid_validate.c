/*
 * Mechanical, read-only validation of the on-disk pgturbohybrid index.
 */
#include "postgres.h"

#include <float.h>

#include "access/genam.h"
#include "access/table.h"
#include "catalog/pg_am_d.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "utils/fmgrprotos.h"
#include "utils/jsonb.h"
#include "utils/lsyscache.h"
#include "utils/numeric.h"
#include "utils/rel.h"

#include "pgturbohybrid.h"
#include "pgturbohybrid_bm25.h"
#include "pgturbohybrid_jsonb_compat.h"
#include "pgturbohybrid_quant.h"

typedef struct PgturbohybridValidationIssue
{
	const char *code;
	BlockNumber blkno;
	int64		nodeId;
	int32		tupleOffset;
} PgturbohybridValidationIssue;

typedef struct PgturbohybridValidationState
{
	Relation	index;
	BlockNumber nblocks;
	bool		deep;
	List	   *errors;
	List	   *warnings;
	uint64		checkedPages;
	uint64		checkedTuples;
	uint64		liveNodes;
	uint64		deadNodes;
	uint64		deadBridgeNodes;
	uint64		reachableLiveNodes;
	uint32		degreeMin;
	uint32		degreeMax;
	uint64		degreeSum;
	uint32		degreeCount;
	bool		entryValid;
	bool		routingValid;
	bool		segmentsValid;
	bool		bm25Present;
	bool		multivectorPresent;
	bool		densePresent;
	bool	   *nodeSeen;
	bool	   *nodeDead;
	uint32	   *degree;
	uint32	   *edgeOffsets;
	uint32	   *edges;
	uint64		edgeCount;
} PgturbohybridValidationState;

static void
PgturbohybridValidateAddIssue(PgturbohybridValidationState *state, bool warning,
							 const char *code, BlockNumber blkno,
							 int64 nodeId, int32 tupleOffset)
{
	PgturbohybridValidationIssue *issue = palloc(sizeof(*issue));

	issue->code = code;
	issue->blkno = blkno;
	issue->nodeId = nodeId;
	issue->tupleOffset = tupleOffset;
	if (warning)
		state->warnings = lappend(state->warnings, issue);
	else
		state->errors = lappend(state->errors, issue);
}

static void
PgturbohybridValidateJsonKey(PgturbohybridJsonbState *state, const char *key)
{
	JsonbValue value;

	value.type = jbvString;
	value.val.string.val = (char *) key;
	value.val.string.len = strlen(key);
	PgturbohybridJsonbPush(state, WJB_KEY, &value);
}

static void
PgturbohybridValidateJsonString(PgturbohybridJsonbState *state, const char *key,
							 const char *string)
{
	JsonbValue value;

	PgturbohybridValidateJsonKey(state, key);
	value.type = jbvString;
	value.val.string.val = (char *) string;
	value.val.string.len = strlen(string);
	PgturbohybridJsonbPush(state, WJB_VALUE, &value);
}

static void
PgturbohybridValidateJsonBool(PgturbohybridJsonbState *state, const char *key,
						   bool boolean)
{
	JsonbValue value;

	PgturbohybridValidateJsonKey(state, key);
	value.type = jbvBool;
	value.val.boolean = boolean;
	PgturbohybridJsonbPush(state, WJB_VALUE, &value);
}

static void
PgturbohybridValidateJsonInt64(PgturbohybridJsonbState *state, const char *key,
							int64 number)
{
	JsonbValue value;

	PgturbohybridValidateJsonKey(state, key);
	value.type = jbvNumeric;
	value.val.numeric = DatumGetNumeric(DirectFunctionCall1(int8_numeric,
												Int64GetDatum(number)));
	PgturbohybridJsonbPush(state, WJB_VALUE, &value);
}

static void
PgturbohybridValidateJsonFloat(PgturbohybridJsonbState *state, const char *key,
							double number)
{
	JsonbValue value;

	PgturbohybridValidateJsonKey(state, key);
	value.type = jbvNumeric;
	value.val.numeric = DatumGetNumeric(DirectFunctionCall1(float8_numeric,
												Float8GetDatum(number)));
	PgturbohybridJsonbPush(state, WJB_VALUE, &value);
}

static void
PgturbohybridValidateJsonIssues(PgturbohybridJsonbState *json, const char *key,
							 List *issues)
{
	ListCell   *cell;

	PgturbohybridValidateJsonKey(json, key);
	PgturbohybridJsonbPush(json, WJB_BEGIN_ARRAY, NULL);
	foreach(cell, issues)
	{
		PgturbohybridValidationIssue *issue = lfirst(cell);

		PgturbohybridJsonbPush(json, WJB_BEGIN_OBJECT, NULL);
		PgturbohybridValidateJsonString(json, "code", issue->code);
		if (BlockNumberIsValid(issue->blkno))
			PgturbohybridValidateJsonInt64(json, "block", issue->blkno);
		if (issue->nodeId >= 0)
			PgturbohybridValidateJsonInt64(json, "node", issue->nodeId);
		if (issue->tupleOffset >= 0)
			PgturbohybridValidateJsonInt64(json, "tuple", issue->tupleOffset);
		PgturbohybridJsonbPush(json, WJB_END_OBJECT, NULL);
	}
	PgturbohybridJsonbPush(json, WJB_END_ARRAY, NULL);
}

static bool
PgturbohybridValidateSampleNode(uint32 nodeId, uint32 nodeCount)
{
	uint32		stride = Max(1U, nodeCount / 64U);

	return nodeId == 0 || nodeId + 1 == nodeCount || nodeId % stride == 0;
}

static bool
PgturbohybridValidatePageEnvelope(PgturbohybridValidationState *state,
								BlockNumber blkno, Page page,
								uint16 *pageKind, BlockNumber *nextblkno)
{
	PgturbohybridGraphPageOpaque opaque;

	state->checkedPages++;
	if (PageIsNew(page) || PageGetSpecialSize(page) !=
		MAXALIGN(sizeof(PgturbohybridGraphPageOpaqueData)))
	{
		PgturbohybridValidateAddIssue(state, false, "invalid_page_envelope",
								 blkno, -1, -1);
		return false;
	}
	opaque = PgturbohybridGraphPageGetOpaque(page);
	*pageKind = opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK;
	*nextblkno = opaque->nextblkno;
	if (opaque->page_id != PGTURBOHYBRID_GRAPH_PAGE_ID)
		PgturbohybridValidateAddIssue(state, false, "invalid_page_id", blkno,
								 -1, -1);
	if (*pageKind < PGTURBOHYBRID_GRAPH_PAGE_KIND_GRAPH ||
		*pageKind > PGTURBOHYBRID_GRAPH_PAGE_KIND_NODEMAP)
		PgturbohybridValidateAddIssue(state, false, "invalid_page_kind", blkno,
								 -1, -1);
	if (*nextblkno == blkno)
		PgturbohybridValidateAddIssue(state, false, "chain_self_loop", blkno,
								 -1, -1);
	if (BlockNumberIsValid(*nextblkno) &&
		(*nextblkno == PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO ||
		 *nextblkno >= state->nblocks))
		PgturbohybridValidateAddIssue(state, false, "chain_pointer_out_of_range",
								 blkno, -1, -1);
	return true;
}

static void
PgturbohybridValidateAllPageChains(PgturbohybridValidationState *state)
{
	uint8	   *colors = palloc0(state->nblocks);

	for (BlockNumber origin = 1; origin < state->nblocks; origin++)
	{
		BlockNumber current = origin;

		CHECK_FOR_INTERRUPTS();
		while (BlockNumberIsValid(current) && current > 0 &&
			   current < state->nblocks && colors[current] == 0)
		{
			Buffer		buf = ReadBuffer(state->index, current);
			Page		page;
			uint16		kind = 0;
			BlockNumber next = InvalidBlockNumber;

			colors[current] = 1;
			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);
			(void) PgturbohybridValidatePageEnvelope(state, current, page,
												&kind, &next);
			UnlockReleaseBuffer(buf);
			if (BlockNumberIsValid(next) && next < state->nblocks && colors[next] == 1)
			{
				PgturbohybridValidateAddIssue(state, false, "chain_cycle", next,
										 -1, -1);
				break;
			}
			current = next;
		}
		current = origin;
		while (current > 0 && current < state->nblocks && colors[current] == 1)
		{
			Buffer		buf = ReadBuffer(state->index, current);
			Page		page;
			PgturbohybridGraphPageOpaque opaque;
			BlockNumber next;

			colors[current] = 2;
			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);
			if (PageIsNew(page) || PageGetSpecialSize(page) !=
				MAXALIGN(sizeof(PgturbohybridGraphPageOpaqueData)))
			{
				UnlockReleaseBuffer(buf);
				break;
			}
			opaque = PgturbohybridGraphPageGetOpaque(page);
			next = opaque->nextblkno;
			UnlockReleaseBuffer(buf);
			current = next;
		}
	}
	pfree(colors);
}

static void
PgturbohybridValidateCodeChain(PgturbohybridValidationState *state,
							  PgturbohybridGraphMetaPageData *meta)
{
	BlockNumber blkno = meta->tqCodeStartBlkno;
	bool		weighted = (meta->tqFlags & PGTURBOHYBRID_GRAPH_TQ_WEIGHTED) != 0;
	Size		expectedSize = PgturbohybridGraphCodeTupleSize(meta->dimensions,
		meta->tqPayloadCount,
		meta->tqBits == 0 ? PGTURBOHYBRID_DEFAULT_BITS : meta->tqBits,
		weighted,
		meta->tqResidualRerankBytes);
	bool	   *chainSeen = palloc0(state->nblocks);

	while (BlockNumberIsValid(blkno) && blkno > 0 && blkno < state->nblocks)
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		OffsetNumber maxoff;

		CHECK_FOR_INTERRUPTS();
		if (chainSeen[blkno])
			break;
		chainSeen[blkno] = true;
		buf = ReadBuffer(state->index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		if (PageIsNew(page) || PageGetSpecialSize(page) !=
			MAXALIGN(sizeof(PgturbohybridGraphPageOpaqueData)))
		{
			UnlockReleaseBuffer(buf);
			break;
		}
		opaque = PgturbohybridGraphPageGetOpaque(page);
		if ((opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) !=
			PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_CODE)
			PgturbohybridValidateAddIssue(state, false, "code_chain_page_kind",
									 blkno, -1, -1);
		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff;
			 offno = OffsetNumberNext(offno))
		{
			ItemId		iid = PageGetItemId(page, offno);
			PgturbohybridGraphCodeTuple tuple;
			Size		itemSize;

			state->checkedTuples++;
			if (!ItemIdIsUsed(iid))
			{
				PgturbohybridValidateAddIssue(state, false, "unused_code_item",
										 blkno, -1, offno);
				continue;
			}
			itemSize = ItemIdGetLength(iid);
			if (itemSize < offsetof(PgturbohybridGraphCodeTupleData, data))
			{
				PgturbohybridValidateAddIssue(state, false, "short_code_tuple",
										 blkno, -1, offno);
				continue;
			}
			tuple = (PgturbohybridGraphCodeTuple) PageGetItem(page, iid);
			if (itemSize < expectedSize)
				PgturbohybridValidateAddIssue(state, false, "code_tuple_size",
										 blkno, tuple->nodeId, offno);
			if (tuple->type != PGTURBOHYBRID_GRAPH_CODE_TUPLE_TYPE)
				PgturbohybridValidateAddIssue(state, false, "code_tuple_type",
										 blkno, tuple->nodeId, offno);
			if (tuple->nodeId >= meta->tqNodeCount)
			{
				PgturbohybridValidateAddIssue(state, false, "code_node_out_of_range",
										 blkno, tuple->nodeId, offno);
				continue;
			}
			if (state->nodeSeen[tuple->nodeId])
				PgturbohybridValidateAddIssue(state, false, "duplicate_code_node",
										 blkno, tuple->nodeId, offno);
			state->nodeSeen[tuple->nodeId] = true;
			state->nodeDead[tuple->nodeId] =
				(tuple->flags & PGTURBOHYBRID_GRAPH_NODE_DEAD) != 0;
			if (state->nodeDead[tuple->nodeId])
				state->deadNodes++;
			else
				state->liveNodes++;
			if (!ItemPointerIsValid(&tuple->heaptid))
				PgturbohybridValidateAddIssue(state, false, "invalid_heap_tid",
										 blkno, tuple->nodeId, offno);
			if (meta->tqPayloadCount < 16 &&
				(tuple->payloadMask >> meta->tqPayloadCount) != 0)
				PgturbohybridValidateAddIssue(state, false, "payload_mask_out_of_range",
										 blkno, tuple->nodeId, offno);
			if (BlockNumberIsValid(tuple->exactBlkno) &&
				(tuple->exactBlkno >= state->nblocks ||
				 !PgturbohybridGraphExactByteOffsetIsValid(tuple->exactOffno)))
				PgturbohybridValidateAddIssue(state, false, "exact_ref_out_of_range",
										 blkno, tuple->nodeId, offno);
		}
		blkno = opaque->nextblkno;
		UnlockReleaseBuffer(buf);
	}
	pfree(chainSeen);
	if (state->deep)
		for (uint32 nodeId = 0; nodeId < meta->tqNodeCount; nodeId++)
			if (!state->nodeSeen[nodeId])
				PgturbohybridValidateAddIssue(state, false, "missing_code_node",
										 InvalidBlockNumber, nodeId, -1);
}

static void
PgturbohybridValidateExactPages(PgturbohybridValidationState *state,
								PgturbohybridGraphMetaPageData *meta)
{
	BlockNumber blkno = meta->tqExactStartBlkno;
	bool	   *chainSeen = palloc0(Max((Size) 1, (Size) state->nblocks));

	while (BlockNumberIsValid(blkno) && blkno > 0 && blkno < state->nblocks)
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		OffsetNumber maxoff;

		CHECK_FOR_INTERRUPTS();
		if (chainSeen[blkno])
			break;
		chainSeen[blkno] = true;
		buf = ReadBuffer(state->index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		if (PageIsNew(page) || PageGetSpecialSize(page) !=
			MAXALIGN(sizeof(PgturbohybridGraphPageOpaqueData)))
		{
			UnlockReleaseBuffer(buf);
			break;
		}
		opaque = PgturbohybridGraphPageGetOpaque(page);
		if ((opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) !=
			PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_EXACT)
			PgturbohybridValidateAddIssue(state, false, "exact_chain_page_kind",
									 blkno, -1, -1);
		maxoff = PageGetMaxOffsetNumber(page);
		if (maxoff < FirstOffsetNumber)
			PgturbohybridValidateAddIssue(state, false, "empty_exact_page", blkno,
									 -1, -1);
		else
		{
			ItemId iid = PageGetItemId(page, FirstOffsetNumber);
			Size itemSize = ItemIdIsUsed(iid) ? ItemIdGetLength(iid) : 0;

			state->checkedTuples++;
			if (itemSize >= offsetof(PgturbohybridGraphExactSlabPageHeaderData, data))
			{
				PgturbohybridGraphExactSlabPageHeader header =
					(PgturbohybridGraphExactSlabPageHeader) PageGetItem(page, iid);

				if (header->magic == PGTURBOHYBRID_GRAPH_EXACT_SLAB_MAGIC)
				{
					Size available = itemSize -
						offsetof(PgturbohybridGraphExactSlabPageHeaderData, data);

					if (maxoff != FirstOffsetNumber || header->capacity != available ||
						header->used > header->capacity)
						PgturbohybridValidateAddIssue(state, false,
							"exact_slab_bounds", blkno, -1, FirstOffsetNumber);
				}
				else
				{
					Size vectorSize = PGTURBOHYBRID_VECTOR_SIZE(meta->dimensions);

					for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff;
						 offno = OffsetNumberNext(offno))
					{
						iid = PageGetItemId(page, offno);
						state->checkedTuples++;
						if (!ItemIdIsUsed(iid) ||
							ItemIdGetLength(iid) <
							offsetof(PgturbohybridGraphExactTupleData, vector) + vectorSize)
						{
							PgturbohybridValidateAddIssue(state, false,
								"exact_tuple_size", blkno, -1, offno);
							continue;
						}
						if (((PgturbohybridGraphExactTuple) PageGetItem(page, iid))->type !=
							PGTURBOHYBRID_GRAPH_EXACT_TUPLE_TYPE)
							PgturbohybridValidateAddIssue(state, false,
								"exact_tuple_type", blkno, -1, offno);
					}
				}
			}
			else
				PgturbohybridValidateAddIssue(state, false, "short_exact_page",
									 blkno, -1, FirstOffsetNumber);
		}
		blkno = opaque->nextblkno;
		UnlockReleaseBuffer(buf);
	}
	pfree(chainSeen);
}

static void
PgturbohybridValidateAdjChain(PgturbohybridValidationState *state,
							 PgturbohybridGraphMetaPageData *meta, bool fillEdges)
{
	BlockNumber blkno = meta->tqAdjStartBlkno;
	bool	   *chainSeen = palloc0(state->nblocks);
	uint32	   *cursor = NULL;

	if (fillEdges)
	{
		cursor = palloc(sizeof(uint32) * meta->tqNodeCount);
		memcpy(cursor, state->edgeOffsets,
			   sizeof(uint32) * meta->tqNodeCount);
	}
	while (BlockNumberIsValid(blkno) && blkno > 0 && blkno < state->nblocks)
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		OffsetNumber maxoff;

		CHECK_FOR_INTERRUPTS();
		if (chainSeen[blkno])
			break;
		chainSeen[blkno] = true;
		buf = ReadBuffer(state->index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		if (PageIsNew(page) || PageGetSpecialSize(page) !=
			MAXALIGN(sizeof(PgturbohybridGraphPageOpaqueData)))
		{
			UnlockReleaseBuffer(buf);
			break;
		}
		opaque = PgturbohybridGraphPageGetOpaque(page);
		if ((opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) !=
			PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_ADJ)
			PgturbohybridValidateAddIssue(state, false, "adj_chain_page_kind",
									 blkno, -1, -1);
		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff;
			 offno = OffsetNumberNext(offno))
		{
			ItemId		iid = PageGetItemId(page, offno);
			PgturbohybridGraphAdjTuple tuple;
			Size		itemSize;
			int		capacity;

			if (!fillEdges)
				state->checkedTuples++;
			if (!ItemIdIsUsed(iid))
			{
				if (!fillEdges)
					PgturbohybridValidateAddIssue(state, false, "unused_adj_item",
											 blkno, -1, offno);
				continue;
			}
			itemSize = ItemIdGetLength(iid);
			if (itemSize < offsetof(PgturbohybridGraphAdjTupleData, neighbors))
			{
				if (!fillEdges)
					PgturbohybridValidateAddIssue(state, false, "short_adj_tuple",
											 blkno, -1, offno);
				continue;
			}
			tuple = (PgturbohybridGraphAdjTuple) PageGetItem(page, iid);
			if (!state->deep && tuple->nodeId < meta->tqNodeCount &&
				!PgturbohybridValidateSampleNode(tuple->nodeId, meta->tqNodeCount))
				continue;
			capacity = tuple->level < PgturbohybridGraphLevelCapacity(meta->m) ?
				PgturbohybridGraphLevelM(meta->m, tuple->level) : 0;
			if (!fillEdges && tuple->type != PGTURBOHYBRID_GRAPH_ADJ_TUPLE_TYPE)
				PgturbohybridValidateAddIssue(state, false, "adj_tuple_type",
										 blkno, tuple->nodeId, offno);
			if (tuple->nodeId >= meta->tqNodeCount || capacity == 0)
			{
				if (!fillEdges)
					PgturbohybridValidateAddIssue(state, false, "adj_identity_out_of_range",
											 blkno, tuple->nodeId, offno);
				continue;
			}
			if (!fillEdges && (tuple->count > capacity ||
				itemSize < PgturbohybridGraphAdjTupleSize(Min(tuple->count, capacity))))
				PgturbohybridValidateAddIssue(state, false, "adj_count_or_size",
										 blkno, tuple->nodeId, offno);
			for (int i = 0; i < Min(tuple->count, capacity); i++)
			{
				uint32 neighbor = tuple->neighbors[i];

				if (neighbor >= meta->tqNodeCount)
				{
					if (!fillEdges)
						PgturbohybridValidateAddIssue(state, false,
							"adj_neighbor_out_of_range", blkno,
							 tuple->nodeId, offno);
					continue;
				}
				if (tuple->level == 0)
				{
					if (fillEdges)
						state->edges[cursor[tuple->nodeId]++] = neighbor;
					else
						state->degree[tuple->nodeId]++;
				}
			}
		}
		blkno = opaque->nextblkno;
		UnlockReleaseBuffer(buf);
	}
	if (cursor != NULL)
		pfree(cursor);
	pfree(chainSeen);
}

static bool
PgturbohybridValidateTupleTypeForPage(uint16 kind, uint8 type)
{
	switch (kind)
	{
		case PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_CORRECTION:
			return type == PGTURBOHYBRID_GRAPH_CORRECTION_TUPLE_TYPE;
		case PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_META:
			return type == PGTURBOHYBRID_BM25_META_TUPLE_TYPE;
		case PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_DOCSTATS:
			return type == PGTURBOHYBRID_BM25_DOCSTATS_TUPLE_TYPE;
		case PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_LEXICON:
			return type == PGTURBOHYBRID_BM25_LEXICON_TUPLE_TYPE;
		case PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_POSTINGS:
			return type == PGTURBOHYBRID_BM25_POSTINGS_TUPLE_TYPE;
		case PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_BLOCKMAX:
			return type == PGTURBOHYBRID_BM25_BLOCKMAX_TUPLE_TYPE;
		case PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_DELTA:
			return type == PGTURBOHYBRID_BM25_DELTA_TUPLE_TYPE;
		case PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_IMPACT:
			return type == PGTURBOHYBRID_BM25_IMPACT_TUPLE_TYPE;
		case PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_DELTA_TERM:
			return type == PGTURBOHYBRID_BM25_DELTA_TERM_TUPLE_TYPE ||
				type == PGTURBOHYBRID_BM25_DELTA_DIRECTORY_TUPLE_TYPE;
		default:
			return true;
	}
}

static Size
PgturbohybridValidateMinimumTupleSize(uint16 kind)
{
	switch (kind)
	{
		case PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_CORRECTION:
			return offsetof(PgturbohybridGraphCorrectionTupleData, values);
		case PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_META:
			return sizeof(PgturbohybridBm25MetaTupleData);
		case PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_DOCSTATS:
			return offsetof(PgturbohybridBm25DocStatsTupleData, docs);
		case PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_LEXICON:
			return offsetof(PgturbohybridBm25LexiconEntryData, termBytes);
		case PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_POSTINGS:
			return offsetof(PgturbohybridBm25PostingsTupleData, payload);
		case PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_BLOCKMAX:
			return sizeof(PgturbohybridBm25BlockMaxTupleData);
		case PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_DELTA:
			return offsetof(PgturbohybridBm25DeltaTupleData, terms);
		case PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_IMPACT:
			return offsetof(PgturbohybridBm25ImpactTupleData, entries);
		case PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_DELTA_TERM:
			return offsetof(PgturbohybridBm25DeltaTermTupleData, postings);
		default:
			return 1;
	}
}

static void
PgturbohybridValidateBranchTupleBounds(PgturbohybridValidationState *state,
									   PgturbohybridGraphMetaPageData *meta,
									   uint16 kind, BlockNumber blkno,
									   OffsetNumber offno, char *data,
									   Size itemSize)
{
	Size		required = 0;
	uint64	first = 0;
	uint64	count = 0;

	switch (kind)
	{
		case PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_CORRECTION:
		{
			PgturbohybridGraphCorrectionTuple tuple =
				(PgturbohybridGraphCorrectionTuple) data;

			count = tuple->count;
			first = tuple->startDim;
			required = offsetof(PgturbohybridGraphCorrectionTupleData, values) +
				(Size) count * sizeof(float);
			if (first + count > meta->dimensions)
				PgturbohybridValidateAddIssue(state, false,
					"correction_dimension_range", blkno, -1, offno);
			break;
		}
		case PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_DOCSTATS:
		{
			PgturbohybridBm25DocStatsTuple tuple =
				(PgturbohybridBm25DocStatsTuple) data;

			count = tuple->count;
			first = tuple->startNodeId;
			required = offsetof(PgturbohybridBm25DocStatsTupleData, docs) +
				(Size) count * sizeof(TqBm25DocStat);
			break;
		}
		case PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_POSTINGS:
		{
			PgturbohybridBm25PostingsTuple tuple =
				(PgturbohybridBm25PostingsTuple) data;

			count = tuple->count;
			first = tuple->firstNodeId;
			required = offsetof(PgturbohybridBm25PostingsTupleData, payload) +
				tuple->payloadBytes;
			if (count == 0 || tuple->firstNodeId > tuple->lastNodeId ||
				tuple->lastNodeId >= meta->tqNodeCount)
				PgturbohybridValidateAddIssue(state, false, "bm25_postings_range",
										 blkno, tuple->firstNodeId, offno);
			break;
		}
		case PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_IMPACT:
		{
			PgturbohybridBm25ImpactTuple tuple =
				(PgturbohybridBm25ImpactTuple) data;

			count = tuple->count;
			required = offsetof(PgturbohybridBm25ImpactTupleData, entries) +
				(Size) count * sizeof(PgturbohybridBm25ImpactTupleEntry);
			if (itemSize >= required)
				for (uint16 i = 0; i < tuple->count; i++)
					if (tuple->entries[i].nodeId >= meta->tqNodeCount)
						PgturbohybridValidateAddIssue(state, false,
							"bm25_impact_node_range", blkno,
							tuple->entries[i].nodeId, offno);
			break;
		}
		case PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_DELTA:
		{
			PgturbohybridBm25DeltaTuple tuple =
				(PgturbohybridBm25DeltaTuple) data;

			count = tuple->termCount;
			required = offsetof(PgturbohybridBm25DeltaTupleData, terms) +
				(Size) count * sizeof(PgturbohybridBm25DeltaTerm) + tuple->termBytesLen;
			if (tuple->nodeId >= meta->tqNodeCount ||
				!ItemPointerIsValid(&tuple->heaptid))
				PgturbohybridValidateAddIssue(state, false, "bm25_delta_identity",
										 blkno, tuple->nodeId, offno);
			break;
		}
		default:
			break;
	}
	if (required > 0 && itemSize < required)
		PgturbohybridValidateAddIssue(state, false, "branch_tuple_count_or_size",
									 blkno, -1, offno);
	if (count > 0 && first + count > meta->tqNodeCount &&
		kind == PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_DOCSTATS)
		PgturbohybridValidateAddIssue(state, false, "branch_node_range", blkno,
									 first, offno);
}

static bool
PgturbohybridValidateDocmapTuple(PgturbohybridValidationState *state,
								 PgturbohybridGraphMetaPageData *meta,
								 BlockNumber blkno, OffsetNumber offno,
								 const char *data, Size itemSize)
{
	uint8		type;
	uint8		version;
	uint16	count;
	uint32		magic;
	Size		required = 0;

	if (itemSize < 8)
		return false;
	type = *(const uint8 *) data;
	version = *(const uint8 *) (data + 1);
	memcpy(&count, data + 2, sizeof(count));
	memcpy(&magic, data + 4, sizeof(magic));
	if (magic != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC)
	{
		PgturbohybridValidateAddIssue(state, false, "docmap_magic", blkno,
								 -1, offno);
		return false;
	}
	if (version != PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION)
		PgturbohybridValidateAddIssue(state, false, "docmap_version", blkno,
								 -1, offno);
	switch (type)
	{
		case PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_NODE_TUPLE_TYPE:
			required = offsetof(PgturbohybridGraphMultiVectorDocMapNodeTupleData,
							entries) + (Size) count * sizeof(TqMultiVectorNodeMapEntry);
			break;
		case PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_DOC_TUPLE_TYPE:
			required = offsetof(PgturbohybridGraphMultiVectorDocMapDocTupleData,
							entries) + (Size) count * sizeof(TqMultiVectorDocMapEntry);
			break;
		case PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VECTOR_TUPLE_TYPE:
			required = offsetof(PgturbohybridGraphMultiVectorDocMapVectorTupleData,
							values) + (Size) count * sizeof(float);
			break;
		case PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_CONTEXT_TUPLE_TYPE:
			required = offsetof(PgturbohybridGraphMultiVectorDocMapContextTupleData,
							values) + (Size) count * sizeof(int32);
			break;
		case PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_CENTROID_TUPLE_TYPE:
			required = offsetof(PgturbohybridGraphMultiVectorDocMapCentroidTupleData,
							values) + (Size) count * sizeof(float);
			break;
		case PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_CENTROID_POSTING_TUPLE_TYPE:
			required = offsetof(PgturbohybridGraphMultiVectorDocMapCentroidPostingTupleData,
							entries) + (Size) count *
				sizeof(PgturbohybridGraphMultiVectorCentroidPostingEntry);
			break;
		case PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_QUANTIZED_POSTING_TUPLE_TYPE:
			required = offsetof(PgturbohybridGraphMultiVectorDocMapQuantizedPostingTupleData,
							entries) + (Size) count *
				sizeof(PgturbohybridGraphMultiVectorQuantizedPostingEntry);
			break;
		case PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_QUANTIZED_CODEBOOK_TUPLE_TYPE:
			required = sizeof(PgturbohybridGraphMultiVectorDocMapQuantizedCodebookTupleData);
			break;
		case PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_CENTROID_F16_TUPLE_TYPE:
			required = offsetof(PgturbohybridGraphMultiVectorDocMapCentroidF16TupleData,
							values) + (Size) count * sizeof(uint16);
			break;
		case PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VECTOR_F16_TUPLE_TYPE:
			required = offsetof(PgturbohybridGraphMultiVectorDocMapVectorF16TupleData,
							values) + (Size) count * sizeof(uint16);
			break;
		case PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VECTOR_SQ8_TUPLE_TYPE:
			required = offsetof(PgturbohybridGraphMultiVectorDocMapVectorSq8TupleData,
							values) + (Size) count * sizeof(int8);
			break;
		case PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_CENTROID_DOC_CODE_TUPLE_TYPE:
			required = offsetof(PgturbohybridGraphMultiVectorDocMapCentroidDocCodeTupleData,
							codes) + (Size) count * sizeof(uint32);
			break;
		default:
			PgturbohybridValidateAddIssue(state, false, "docmap_tuple_type", blkno,
									 -1, offno);
			return false;
	}
	if (itemSize < required)
		PgturbohybridValidateAddIssue(state, false, "docmap_tuple_size", blkno,
								 -1, offno);
	if (type == PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_NODE_TUPLE_TYPE &&
		itemSize >= required)
	{
		PgturbohybridGraphMultiVectorDocMapNodeTuple tuple =
			(PgturbohybridGraphMultiVectorDocMapNodeTuple) data;

		if ((uint64) tuple->firstNodeId + tuple->count > meta->tqNodeCount)
			PgturbohybridValidateAddIssue(state, false, "docmap_node_range", blkno,
								 tuple->firstNodeId, offno);
	}
	if (type == PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_DOC_TUPLE_TYPE &&
		itemSize >= required)
	{
		PgturbohybridGraphMultiVectorDocMapDocTuple tuple =
			(PgturbohybridGraphMultiVectorDocMapDocTuple) data;

		if ((uint64) tuple->firstDocId + tuple->count > meta->tqMultivectorDocCount)
			PgturbohybridValidateAddIssue(state, false, "docmap_doc_range", blkno,
								 tuple->firstDocId, offno);
	}
	return itemSize >= required;
}

static void
PgturbohybridValidateBranchTuples(PgturbohybridValidationState *state,
								 PgturbohybridGraphMetaPageData *meta)
{
	for (BlockNumber blkno = 1; blkno < state->nblocks; blkno++)
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		uint16		kind;
		OffsetNumber maxoff;

		CHECK_FOR_INTERRUPTS();
		buf = ReadBuffer(state->index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		if (PageIsNew(page) || PageGetSpecialSize(page) !=
			MAXALIGN(sizeof(PgturbohybridGraphPageOpaqueData)))
		{
			UnlockReleaseBuffer(buf);
			continue;
		}
		opaque = PgturbohybridGraphPageGetOpaque(page);
		kind = opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK;
		if (kind == PGTURBOHYBRID_GRAPH_PAGE_KIND_GRAPH ||
			kind == PGTURBOHYBRID_GRAPH_PAGE_KIND_META ||
			kind == PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_CODE ||
			kind == PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_ADJ ||
			kind == PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_EXACT)
		{
			UnlockReleaseBuffer(buf);
			continue;
		}
		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff;
			 offno = OffsetNumberNext(offno))
		{
			ItemId		iid = PageGetItemId(page, offno);
			char	   *data;
			Size		itemSize;

			state->checkedTuples++;
			if (!ItemIdIsUsed(iid) || ItemIdGetLength(iid) < 1)
			{
				PgturbohybridValidateAddIssue(state, false, "invalid_branch_item",
										 blkno, -1, offno);
				continue;
			}
			data = PageGetItem(page, iid);
			itemSize = ItemIdGetLength(iid);
			if (kind == PGTURBOHYBRID_GRAPH_PAGE_KIND_MULTIVECTOR_DOCMAP)
			{
				(void) PgturbohybridValidateDocmapTuple(state, meta, blkno, offno,
											 data, itemSize);
				continue;
			}
			if (itemSize < PgturbohybridValidateMinimumTupleSize(kind))
			{
				PgturbohybridValidateAddIssue(state, false, "short_branch_tuple",
										 blkno, -1, offno);
				continue;
			}
			if (!PgturbohybridValidateTupleTypeForPage(kind, *(uint8 *) data))
				PgturbohybridValidateAddIssue(state, false, "branch_tuple_type",
										 blkno, -1, offno);
			PgturbohybridValidateBranchTupleBounds(state, meta, kind, blkno,
										 offno, data, itemSize);
			if (kind == PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_META &&
				itemSize >= sizeof(PgturbohybridBm25MetaTupleData))
			{
				PgturbohybridBm25MetaTuple tuple =
					(PgturbohybridBm25MetaTuple) data;

				if (tuple->bm25Version != PGTURBOHYBRID_BM25_VERSION ||
					tuple->docCount > meta->tqNodeCount)
					PgturbohybridValidateAddIssue(state, false, "bm25_meta_bounds",
											 blkno, -1, offno);
			}
		}
		UnlockReleaseBuffer(buf);
	}
}

static void
PgturbohybridValidateMeta(PgturbohybridValidationState *state,
						 PgturbohybridGraphMetaPageData *meta)
{
	bool		densePresent = BlockNumberIsValid(meta->tqCodeStartBlkno);

	if (meta->magicNumber != PGTURBOHYBRID_GRAPH_MAGIC_NUMBER)
		PgturbohybridValidateAddIssue(state, false, "invalid_meta_magic", 0, -1, -1);
	if (meta->version != PGTURBOHYBRID_GRAPH_NATIVE_VERSION)
		PgturbohybridValidateAddIssue(state, false, "invalid_format_version", 0, -1, -1);
	if (meta->storageKind != PGTURBOHYBRID_GRAPH_STORAGE_QUANT_GRAPH_NATIVE)
		PgturbohybridValidateAddIssue(state, false, "invalid_storage_kind", 0, -1, -1);
	if ((densePresent &&
		 (meta->dimensions == 0 || meta->dimensions > PGTURBOHYBRID_GRAPH_MAX_DIM ||
		  meta->m < PGTURBOHYBRID_GRAPH_MIN_M || meta->m > PGTURBOHYBRID_GRAPH_MAX_M)) ||
		meta->tqPayloadCount > PGTURBOHYBRID_GRAPH_MAX_PAYLOADS ||
		meta->tqResidualRerankBytes > PGTURBOHYBRID_GRAPH_MAX_RESIDUAL_RERANK_BYTES)
		PgturbohybridValidateAddIssue(state, false, "meta_bounds", 0, -1, -1);
	if (densePresent && meta->tqNodeCount > 0 &&
		(meta->tqEntryNodeId >= meta->tqNodeCount))
	{
		state->entryValid = false;
		PgturbohybridValidateAddIssue(state, false, "entry_node_out_of_range", 0,
								 meta->tqEntryNodeId, -1);
	}
	else
		state->entryValid = true;
	state->routingValid = true;
	if (meta->tqRoutingEntryCount > PGTURBOHYBRID_GRAPH_MAX_ROUTING_ENTRIES)
		state->routingValid = false;
	for (uint16 i = 0; i < Min(meta->tqRoutingEntryCount,
								 PGTURBOHYBRID_GRAPH_MAX_ROUTING_ENTRIES); i++)
		if (meta->tqRoutingEntryNodeIds[i] >= meta->tqNodeCount)
			state->routingValid = false;
	if (!state->routingValid)
		PgturbohybridValidateAddIssue(state, false, "routing_entry_out_of_range", 0,
								 -1, -1);
	state->segmentsValid = meta->tqSegmentCount <= PGTURBOHYBRID_GRAPH_MAX_NATIVE_SEGMENTS;
	for (uint16 i = 0; state->segmentsValid && i < meta->tqSegmentCount; i++)
	{
		PgturbohybridGraphSegmentMetaData *segment = &meta->tqSegments[i];
		uint64		end = (uint64) segment->startNodeId + segment->nodeCount;

		if (segment->nodeCount == 0 || end > meta->tqNodeCount ||
			(segment->entryNodeId != UINT_MAX &&
			 (segment->entryNodeId < segment->startNodeId ||
			  segment->entryNodeId >= end)))
			state->segmentsValid = false;
	}
	if (!state->segmentsValid)
		PgturbohybridValidateAddIssue(state, false, "segment_bounds", 0, -1, -1);
	state->densePresent = densePresent;
	state->bm25Present = BlockNumberIsValid(meta->tqBm25MetaStartBlkno);
	state->multivectorPresent = BlockNumberIsValid(meta->tqMultivectorDocMapStartBlkno);
}

static void
PgturbohybridValidateReachability(PgturbohybridValidationState *state,
								 PgturbohybridGraphMetaPageData *meta)
{
	bool	   *visited;
	uint32	   *queue;
	uint32		head = 0;
	uint32		tail = 0;

	state->edgeOffsets = palloc0(sizeof(uint32) * ((Size) meta->tqNodeCount + 1));
	for (uint32 nodeId = 0; nodeId < meta->tqNodeCount; nodeId++)
	{
		state->edgeOffsets[nodeId + 1] = state->edgeOffsets[nodeId] + state->degree[nodeId];
		state->edgeCount += state->degree[nodeId];
		if (!state->nodeDead[nodeId])
		{
			state->degreeMin = state->degreeCount == 0 ? state->degree[nodeId] :
				Min(state->degreeMin, state->degree[nodeId]);
			state->degreeMax = Max(state->degreeMax, state->degree[nodeId]);
			state->degreeSum += state->degree[nodeId];
			state->degreeCount++;
		}
	}
	if (state->edgeCount > MaxAllocSize / sizeof(uint32))
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pgturbohybrid validator adjacency is too large")));
	state->edges = palloc0(Max((Size) 1, (Size) state->edgeCount) * sizeof(uint32));
	PgturbohybridValidateAdjChain(state, meta, true);
	visited = palloc0(meta->tqNodeCount);
	queue = palloc(Max((Size) 1, (Size) meta->tqNodeCount) * sizeof(uint32));
#define PGTURBOHYBRID_VALIDATE_SEED(seed_) \
	do { uint32 seed__ = (seed_); if (seed__ < meta->tqNodeCount && !visited[seed__]) \
	{ visited[seed__] = true; queue[tail++] = seed__; } } while (0)
	PGTURBOHYBRID_VALIDATE_SEED(meta->tqEntryNodeId);
	for (uint16 i = 0; i < meta->tqRoutingEntryCount; i++)
		PGTURBOHYBRID_VALIDATE_SEED(meta->tqRoutingEntryNodeIds[i]);
	for (uint16 i = 0; i < meta->tqSegmentCount; i++)
		PGTURBOHYBRID_VALIDATE_SEED(meta->tqSegments[i].entryNodeId);
	for (uint16 i = 0; i < meta->tqEntrySidecarCount; i++)
		PGTURBOHYBRID_VALIDATE_SEED(meta->tqEntrySidecarNodeIds[i]);
	while (head < tail)
	{
		uint32 nodeId = queue[head++];

		CHECK_FOR_INTERRUPTS();
		if (!state->nodeDead[nodeId])
			state->reachableLiveNodes++;
		else if (state->degree[nodeId] > 0)
			state->deadBridgeNodes++;
		for (uint32 i = state->edgeOffsets[nodeId];
			 i < state->edgeOffsets[nodeId + 1]; i++)
			PGTURBOHYBRID_VALIDATE_SEED(state->edges[i]);
	}
#undef PGTURBOHYBRID_VALIDATE_SEED
	if (state->reachableLiveNodes < state->liveNodes)
		PgturbohybridValidateAddIssue(state, true, "unreachable_live_nodes",
								 InvalidBlockNumber, -1, -1);
	pfree(queue);
	pfree(visited);
}

static void
PgturbohybridValidateBranchJson(PgturbohybridJsonbState *json, const char *key,
							 bool present, bool ok)
{
	PgturbohybridValidateJsonKey(json, key);
	PgturbohybridJsonbPush(json, WJB_BEGIN_OBJECT, NULL);
	PgturbohybridValidateJsonBool(json, "present", present);
	PgturbohybridValidateJsonString(json, "status", present ? (ok ? "valid" : "invalid") : "absent");
	PgturbohybridJsonbPush(json, WJB_END_OBJECT, NULL);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_validate_index);
FUNCTION_PREFIX Datum
pgturbohybrid_validate_index(PG_FUNCTION_ARGS)
{
	Oid			indexOid = PG_GETARG_OID(0);
	bool		deep = PG_NARGS() > 1 && !PG_ARGISNULL(1) ? PG_GETARG_BOOL(1) : false;
	Relation	index;
	PgturbohybridValidationState state;
	PgturbohybridGraphMetaPageData meta;
	Buffer		metaBuf;
	Page		metaPage;
	PgturbohybridGraphPageOpaque metaOpaque;
	PgturbohybridJsonbState json;
	Jsonb	   *result;
	double		reachableRatio;
	const char *recommendation;

	index = index_open(indexOid, AccessShareLock);
	if (index->rd_rel->relkind != RELKIND_INDEX ||
		!PgturbohybridGraphIspgturbohybridIndex(index))
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("relation \"%s\" is not a pgturbohybrid index",
						RelationGetRelationName(index))));
	memset(&state, 0, sizeof(state));
	memset(&meta, 0, sizeof(meta));
	state.index = index;
	state.deep = deep;
	state.nblocks = RelationGetNumberOfBlocks(index);
	if (state.nblocks == 0)
		PgturbohybridValidateAddIssue(&state, false, "missing_metapage",
								 InvalidBlockNumber, -1, -1);
	else
	{
		metaBuf = ReadBuffer(index, PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO);
		LockBuffer(metaBuf, BUFFER_LOCK_SHARE);
		metaPage = BufferGetPage(metaBuf);
		if (PageIsNew(metaPage) || PageGetSpecialSize(metaPage) !=
			MAXALIGN(sizeof(PgturbohybridGraphPageOpaqueData)) ||
			PageGetContents(metaPage) + sizeof(meta) >
			(char *) metaPage + BLCKSZ - PageGetSpecialSize(metaPage))
			PgturbohybridValidateAddIssue(&state, false, "invalid_metapage_envelope",
									 0, -1, -1);
		else
		{
			metaOpaque = PgturbohybridGraphPageGetOpaque(metaPage);
			if ((metaOpaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) !=
				PGTURBOHYBRID_GRAPH_PAGE_KIND_META ||
				metaOpaque->page_id != PGTURBOHYBRID_GRAPH_PAGE_ID)
				PgturbohybridValidateAddIssue(&state, false, "invalid_metapage_kind",
										 0, -1, -1);
			memcpy(&meta, PgturbohybridGraphPageGetMeta(metaPage), sizeof(meta));
			PgturbohybridValidateMeta(&state, &meta);
		}
		UnlockReleaseBuffer(metaBuf);
	}
	if (state.nblocks > 1)
		PgturbohybridValidateAllPageChains(&state);
	if (meta.magicNumber == PGTURBOHYBRID_GRAPH_MAGIC_NUMBER &&
		meta.storageKind == PGTURBOHYBRID_GRAPH_STORAGE_QUANT_GRAPH_NATIVE)
	{
		PgturbohybridValidateBranchTuples(&state, &meta);
		state.nodeSeen = palloc0(Max((Size) 1, (Size) meta.tqNodeCount));
		state.nodeDead = palloc0(Max((Size) 1, (Size) meta.tqNodeCount));
		state.degree = palloc0(Max((Size) 1,
									 sizeof(uint32) * (Size) meta.tqNodeCount));
		if (state.densePresent)
		{
			PgturbohybridValidateCodeChain(&state, &meta);
			if (BlockNumberIsValid(meta.tqExactStartBlkno))
				PgturbohybridValidateExactPages(&state, &meta);
			PgturbohybridValidateAdjChain(&state, &meta, false);
			if (deep && list_length(state.errors) == 0)
				PgturbohybridValidateReachability(&state, &meta);
		}
	}
	reachableRatio = state.liveNodes == 0 ? 1.0 :
		(double) state.reachableLiveNodes / (double) state.liveNodes;
	if (list_length(state.errors) > 0)
		recommendation = "corrupt";
	else if (state.deadNodes > 0 &&
		(double) state.deadNodes / (double) Max((uint64) 1,
											 state.liveNodes + state.deadNodes) >= 0.20)
		recommendation = "reindex_recommended";
	else if (state.deadNodes > 0)
		recommendation = "monitor_churn";
	else
		recommendation = "healthy";

	PgturbohybridJsonbStateInit(&json);
	PgturbohybridJsonbBeginObject(&json);
	PgturbohybridValidateJsonBool(&json, "ok", list_length(state.errors) == 0);
	PgturbohybridValidateJsonInt64(&json, "format_version", meta.version);
	PgturbohybridValidateJsonString(&json, "scope", deep ? "deep" : "sampled");
	PgturbohybridValidateJsonString(&json, "recommendation", recommendation);
	PgturbohybridValidateJsonKey(&json, "relation");
	PgturbohybridJsonbPush(&json, WJB_BEGIN_OBJECT, NULL);
	PgturbohybridValidateJsonInt64(&json, "oid", indexOid);
	PgturbohybridValidateJsonString(&json, "name", RelationGetRelationName(index));
	PgturbohybridValidateJsonInt64(&json, "database_oid", MyDatabaseId);
	PgturbohybridValidateJsonInt64(&json, "tablespace_oid", index->rd_rel->reltablespace);
	PgturbohybridValidateJsonInt64(&json, "relfilenode",
		PgturbohybridGraphRelFileNumber(index));
	PgturbohybridJsonbPush(&json, WJB_END_OBJECT, NULL);
	PgturbohybridValidateJsonInt64(&json, "node_count", meta.tqNodeCount);
	PgturbohybridValidateJsonInt64(&json, "live_nodes", state.liveNodes);
	PgturbohybridValidateJsonInt64(&json, "dead_nodes", state.deadNodes);
	PgturbohybridValidateJsonInt64(&json, "error_count", list_length(state.errors));
	PgturbohybridValidateJsonInt64(&json, "warning_count", list_length(state.warnings));
	PgturbohybridValidateJsonIssues(&json, "errors", state.errors);
	PgturbohybridValidateJsonIssues(&json, "warnings", state.warnings);
	PgturbohybridValidateJsonBool(&json, "entry_valid", state.entryValid);
	PgturbohybridValidateJsonBool(&json, "routing_valid", state.routingValid);
	PgturbohybridValidateJsonBool(&json, "segments_valid", state.segmentsValid);
	PgturbohybridValidateJsonInt64(&json, "reachable_live_nodes", state.reachableLiveNodes);
	PgturbohybridValidateJsonFloat(&json, "reachable_live_ratio", deep ? reachableRatio : 0.0);
	PgturbohybridValidateJsonInt64(&json, "unreachable_live_nodes",
		deep && state.densePresent ? state.liveNodes - state.reachableLiveNodes : 0);
	PgturbohybridValidateJsonInt64(&json, "dead_bridge_nodes", state.deadBridgeNodes);
	PgturbohybridValidateJsonKey(&json, "level0_degree");
	PgturbohybridJsonbPush(&json, WJB_BEGIN_OBJECT, NULL);
	PgturbohybridValidateJsonInt64(&json, "min", state.degreeMin);
	PgturbohybridValidateJsonFloat(&json, "avg", state.degreeCount == 0 ? 0.0 :
		(double) state.degreeSum / state.degreeCount);
	PgturbohybridValidateJsonInt64(&json, "max", state.degreeMax);
	PgturbohybridJsonbPush(&json, WJB_END_OBJECT, NULL);
	PgturbohybridValidateJsonInt64(&json, "checked_pages", state.checkedPages);
	PgturbohybridValidateJsonInt64(&json, "checked_tuples", state.checkedTuples);
	PgturbohybridValidateJsonKey(&json, "branches");
	PgturbohybridJsonbPush(&json, WJB_BEGIN_OBJECT, NULL);
	PgturbohybridValidateBranchJson(&json, "dense", state.densePresent,
		list_length(state.errors) == 0);
	PgturbohybridValidateBranchJson(&json, "bm25", state.bm25Present,
		list_length(state.errors) == 0);
	PgturbohybridValidateBranchJson(&json, "multivector", state.multivectorPresent,
		list_length(state.errors) == 0);
	PgturbohybridJsonbPush(&json, WJB_END_OBJECT, NULL);
	result = PgturbohybridJsonbEndObject(&json);
	index_close(index, AccessShareLock);
	PG_RETURN_JSONB_P(result);
}
