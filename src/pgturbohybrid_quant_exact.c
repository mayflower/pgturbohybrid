#include "postgres.h"

#include <string.h>

#include "access/generic_xlog.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/rel.h"

#include "pgturbohybrid_quant.h"

static Oid	tqGraphExactAppendRelid = InvalidOid;
static Oid	tqGraphExactAppendRelfilenumber = InvalidOid;
static BlockNumber tqGraphExactAppendStart = InvalidBlockNumber;
static BlockNumber tqGraphExactAppendTail = InvalidBlockNumber;

static PgturbohybridGraphExactSlabPageHeader
PgturbohybridGraphExactSlabHeader(Page page)
{
	ItemId		iid;

	if (PageGetMaxOffsetNumber(page) < FirstOffsetNumber)
		return NULL;

	iid = PageGetItemId(page, FirstOffsetNumber);
	if (!ItemIdIsValid(iid) || !ItemIdHasStorage(iid))
		return NULL;

	return (PgturbohybridGraphExactSlabPageHeader) PageGetItem(page, iid);
}

static Size
PgturbohybridGraphExactSlabCapacity(Page page)
{
	PgturbohybridGraphExactSlabPageHeader header = PgturbohybridGraphExactSlabHeader(page);

	if (header == NULL)
		return 0;

	return header->capacity;
}

static bool
PgturbohybridGraphExactPageIsSlab(Page page)
{
	PgturbohybridGraphExactSlabPageHeader header = PgturbohybridGraphExactSlabHeader(page);

	return header != NULL && header->magic == PGTURBOHYBRID_GRAPH_EXACT_SLAB_MAGIC;
}

bool
PgturbohybridGraphExactByteOffsetIsValid(OffsetNumber offno)
{
	return offno != InvalidOffsetNumber;
}

static void
PgturbohybridGraphInitExactSlabPage(Page page)
{
	Size		tupleSize = PageGetFreeSpace(page);
	PgturbohybridGraphExactSlabPageHeader tuple;

	tupleSize -= tupleSize % MAXIMUM_ALIGNOF;
	if (tupleSize <= offsetof(PgturbohybridGraphExactSlabPageHeaderData, data))
		elog(ERROR, "pgturbohybrid graph exact slab page has no tuple capacity");

	tuple = palloc0(tupleSize);
	tuple->magic = PGTURBOHYBRID_GRAPH_EXACT_SLAB_MAGIC;
	tuple->capacity = tupleSize - offsetof(PgturbohybridGraphExactSlabPageHeaderData, data);

	if (PageAddItem(page, (Item) tuple, tupleSize, InvalidOffsetNumber, false, false) == InvalidOffsetNumber)
		elog(ERROR, "failed to add pgturbohybrid graph exact slab item");

	pfree(tuple);
}


BlockNumber
PgturbohybridGraphWriteExactPages(PgturbohybridQuantBuildState *state)
{
	Buffer		buf = InvalidBuffer;
	Page		page = NULL;
	BlockNumber start = InvalidBlockNumber;
	Size		vectorSize = PGTURBOHYBRID_VECTOR_SIZE(state->dimensions);

	for (uint32 i = 0; i < state->nodeCount; i++)
	{
		PgturbohybridGraphBuildNode *node = &state->nodes[i];
		char	   *src = (char *) node->vector;
		Size		remaining = vectorSize;

		while (remaining > 0)
		{
			PgturbohybridGraphExactSlabPageHeader header;
			Size		capacity;
			Size		available;
			Size		chunk;

			if (!BufferIsValid(buf))
			{
				PgturbohybridGraphAppendPage(state->index, state->forkNum, &buf, &page, PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_EXACT);
				PgturbohybridGraphInitExactSlabPage(page);
				if (!BlockNumberIsValid(start))
					start = BufferGetBlockNumber(buf);
			}

			header = PgturbohybridGraphExactSlabHeader(page);
			capacity = PgturbohybridGraphExactSlabCapacity(page);
			if (header->used >= capacity)
			{
				PgturbohybridGraphAppendPage(state->index, state->forkNum, &buf, &page, PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_EXACT);
				PgturbohybridGraphInitExactSlabPage(page);
				header = PgturbohybridGraphExactSlabHeader(page);
				capacity = PgturbohybridGraphExactSlabCapacity(page);
			}

			available = capacity - header->used;
			chunk = Min(remaining, available);
			if (chunk == 0)
				elog(ERROR, "pgturbohybrid graph exact slab page has no capacity");

			if (remaining == vectorSize)
			{
				node->exactBlkno = BufferGetBlockNumber(buf);
				node->exactOffno = (OffsetNumber) (header->used + 1);
			}

			memcpy(header->data + header->used, src, chunk);
			header->used += chunk;
			src += chunk;
			remaining -= chunk;
			PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT);
		}
	}

	if (BufferIsValid(buf))
		PgturbohybridGraphFinishPage(buf);

	return start;
}


bool
PgturbohybridGraphReadExactVectorInto(Relation index, PgturbohybridGraphScanNode *node, int dimensions,
						   char *dest, PgturbohybridGraphScanOpaque so)
{
	Buffer		buf;
	Page		page;
	Size		vectorSize = PGTURBOHYBRID_VECTOR_SIZE(dimensions);
	Size		copied = 0;
	BlockNumber blkno;
	OffsetNumber offno;

	if (!BlockNumberIsValid(node->exactBlkno) ||
		!PgturbohybridGraphExactByteOffsetIsValid(node->exactOffno))
		return false;

	if (node->exactVector != NULL)
	{
		memcpy(dest, node->exactVector, vectorSize);
		return true;
	}

	blkno = node->exactBlkno;
	offno = node->exactOffno;
	while (copied < vectorSize)
	{
		PgturbohybridGraphPageOpaque opaque;

		if (!BlockNumberIsValid(blkno))
			return false;

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = PgturbohybridGraphPageGetOpaque(page);

		if ((opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) != PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_EXACT)
		{
			UnlockReleaseBuffer(buf);
			return false;
		}

		if (PgturbohybridGraphExactPageIsSlab(page))
		{
			PgturbohybridGraphExactSlabPageHeader header = PgturbohybridGraphExactSlabHeader(page);
			Size		offset = (Size) offno - 1;
			Size		available;
			Size		chunk;

			if (offset >= header->used)
			{
				UnlockReleaseBuffer(buf);
				return false;
			}

			available = header->used - offset;
			chunk = Min(vectorSize - copied, available);
			memcpy(dest + copied, header->data + offset, chunk);
			copied += chunk;
			blkno = opaque->nextblkno;
			offno = FirstOffsetNumber;
			if (so != NULL)
				so->graphRescorePages++;
			UnlockReleaseBuffer(buf);
			continue;
		}
		else
		{
			ItemId		iid;
			PgturbohybridGraphExactTuple tuple;

			if (node->exactOffno > PageGetMaxOffsetNumber(page))
			{
				UnlockReleaseBuffer(buf);
				return false;
			}

			iid = PageGetItemId(page, node->exactOffno);
			tuple = (PgturbohybridGraphExactTuple) PageGetItem(page, iid);
			if (tuple->type == PGTURBOHYBRID_GRAPH_EXACT_TUPLE_TYPE)
			{
				memcpy(dest, tuple->vector, vectorSize);
				if (so != NULL)
					so->graphRescorePages++;
				UnlockReleaseBuffer(buf);
				return true;
			}
		}

		UnlockReleaseBuffer(buf);
		return false;
	}

	return true;
}

Vector *
PgturbohybridGraphReadExactVector(Relation index, PgturbohybridGraphScanNode *node, int dimensions)
{
	Vector	   *vector;
	Size		vectorSize = PGTURBOHYBRID_VECTOR_SIZE(dimensions);

	vector = palloc(vectorSize);
	if (!PgturbohybridGraphReadExactVectorInto(index, node, dimensions, (char *) vector, NULL))
	{
		pfree(vector);
		return NULL;
	}

	return vector;
}


void
PgturbohybridGraphAppendInsertedExact(Relation index, BlockNumber *exactStart,
						   uint32 nodeId, Vector *vector, int dimensions,
						   BlockNumber *exactBlkno, OffsetNumber *exactOffno)
{
	char	   *src = (char *) vector;
	Size		remaining = PGTURBOHYBRID_VECTOR_SIZE(dimensions);
	BlockNumber blkno = *exactStart;
	BlockNumber originalStart = *exactStart;

	(void) nodeId;

	*exactBlkno = InvalidBlockNumber;
	*exactOffno = InvalidOffsetNumber;
	if (tqGraphExactAppendRelid == RelationGetRelid(index) &&
		tqGraphExactAppendRelfilenumber == PgturbohybridGraphRelFileNumber(index) &&
		tqGraphExactAppendStart == originalStart &&
		BlockNumberIsValid(tqGraphExactAppendTail))
		blkno = tqGraphExactAppendTail;

	while (remaining > 0)
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		PgturbohybridGraphExactSlabPageHeader header;
		Size		capacity;
		Size		available;
		Size		chunk;
		GenericXLogState *xlogState = NULL;
		bool		createdStart = false;
		BlockNumber initBlkno = InvalidBlockNumber;
		BlockNumber linkBlkno = InvalidBlockNumber;

		if (!BlockNumberIsValid(blkno))
		{
			LockRelationForExtension(index, ExclusiveLock);
			buf = PgturbohybridGraphNewBuffer(index, MAIN_FORKNUM);
			UnlockRelationForExtension(index, ExclusiveLock);
			blkno = BufferGetBlockNumber(buf);
			*exactStart = blkno;
			createdStart = true;
		}
		else
		{
			for (;;)
			{
				BlockNumber nextblkno;

				buf = ReadBuffer(index, blkno);
				LockBuffer(buf, BUFFER_LOCK_SHARE);
				page = BufferGetPage(buf);
				nextblkno = PgturbohybridGraphPageGetOpaque(page)->nextblkno;
				UnlockReleaseBuffer(buf);
				if (!BlockNumberIsValid(nextblkno))
					break;
				blkno = nextblkno;
			}

			buf = ReadBuffer(index, blkno);
			LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
		}

		if (RelationNeedsWAL(index))
		{
			xlogState = GenericXLogStart(index);
			page = GenericXLogRegisterBuffer(xlogState, buf,
											 createdStart ? GENERIC_XLOG_FULL_IMAGE : 0);
		}
		else
			page = BufferGetPage(buf);

		if (createdStart)
		{
			PgturbohybridGraphInitPageKind(buf, page, PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_EXACT);
			PgturbohybridGraphInitExactSlabPage(page);
			initBlkno = blkno;
		}

		opaque = PgturbohybridGraphPageGetOpaque(page);
		if ((opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) != PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_EXACT)
			elog(ERROR, "unexpected pgturbohybrid graph exact page kind while appending");
		if (!PgturbohybridGraphExactPageIsSlab(page))
			elog(ERROR, "cannot append to legacy pgturbohybrid graph exact tuple page");

		header = PgturbohybridGraphExactSlabHeader(page);
		capacity = PgturbohybridGraphExactSlabCapacity(page);
		available = header->used < capacity ? capacity - header->used : 0;

		if (available == 0)
		{
			Buffer		newbuf;
			Page		newpage;
			BlockNumber newblkno;

			LockRelationForExtension(index, ExclusiveLock);
			newbuf = PgturbohybridGraphNewBuffer(index, MAIN_FORKNUM);
			UnlockRelationForExtension(index, ExclusiveLock);
			newblkno = BufferGetBlockNumber(newbuf);

			if (xlogState != NULL)
				newpage = GenericXLogRegisterBuffer(xlogState, newbuf,
													GENERIC_XLOG_FULL_IMAGE);
			else
				newpage = BufferGetPage(newbuf);

			PgturbohybridGraphInitPageKind(newbuf, newpage, PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_EXACT);
			PgturbohybridGraphInitExactSlabPage(newpage);
			opaque->nextblkno = newblkno;
			PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_PAGE_LINK);
			linkBlkno = blkno;
			initBlkno = newblkno;

			if (xlogState != NULL)
				GenericXLogFinish(xlogState);
			else
			{
				MarkBufferDirty(buf);
				MarkBufferDirty(newbuf);
			}

			UnlockReleaseBuffer(buf);
			UnlockReleaseBuffer(newbuf);
			PgturbohybridGraphLogGraphWalRecord(index, MAIN_FORKNUM, linkBlkno,
								   PGTURBOHYBRID_GRAPH_GRAPH_OP_PAGE_LINK);
			PgturbohybridGraphLogGraphWalRecord(index, MAIN_FORKNUM, initBlkno,
								   PGTURBOHYBRID_GRAPH_GRAPH_OP_PAGE_INIT);
			blkno = newblkno;
			continue;
		}

		chunk = Min(remaining, available);
		if (!BlockNumberIsValid(*exactBlkno))
		{
			*exactBlkno = blkno;
			*exactOffno = (OffsetNumber) (header->used + 1);
		}

		memcpy(header->data + header->used, src, chunk);
		header->used += chunk;
		src += chunk;
		remaining -= chunk;
		PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT);

		if (xlogState != NULL)
			GenericXLogFinish(xlogState);
		else
			MarkBufferDirty(buf);

		UnlockReleaseBuffer(buf);
		if (BlockNumberIsValid(initBlkno))
			PgturbohybridGraphLogGraphWalRecord(index, MAIN_FORKNUM, initBlkno,
								   PGTURBOHYBRID_GRAPH_GRAPH_OP_PAGE_INIT);
		PgturbohybridGraphLogGraphWalRecord(index, MAIN_FORKNUM, blkno,
							   PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT);
	}

	tqGraphExactAppendRelid = RelationGetRelid(index);
	tqGraphExactAppendRelfilenumber = PgturbohybridGraphRelFileNumber(index);
	tqGraphExactAppendStart = *exactStart;
	tqGraphExactAppendTail = blkno;
}
