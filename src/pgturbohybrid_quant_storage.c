#include "postgres.h"

#include "access/generic_xlog.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/rel.h"

#include "pgturbohybrid_quant.h"

Oid
PgturbohybridGraphRelFileNumber(Relation index)
{
#if PG_VERSION_NUM >= 160000
	return index->rd_locator.relNumber;
#else
	return index->rd_node.relNode;
#endif
}

void
PgturbohybridGraphGetRelationLocator(Relation index, Oid *tablespaceOid,
									 Oid *relNumber)
{
#if PG_VERSION_NUM >= 160000
	*tablespaceOid = index->rd_locator.spcOid;
	*relNumber = index->rd_locator.relNumber;
#else
	*tablespaceOid = index->rd_node.spcNode;
	*relNumber = index->rd_node.relNode;
#endif
}

void
PgturbohybridGraphInitBlockMap(BlockNumber *blknos, int count)
{
	for (int i = 0; i < count; i++)
		blknos[i] = InvalidBlockNumber;
}

bool
PgturbohybridGraphEnsureBlockMap(Relation index, BlockNumber startBlkno, int pageCount,
					  uint16 pageKind, BlockNumber *blknos)
{
	BlockNumber blkno = startBlkno;

	if (pageCount <= 0)
		return true;

	if (BlockNumberIsValid(blknos[pageCount - 1]))
		return true;

	for (int pageNo = 0; pageNo < pageCount; pageNo++)
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;

		if (BlockNumberIsValid(blknos[pageNo]))
		{
			blkno = blknos[pageNo];
			continue;
		}

		if (!BlockNumberIsValid(blkno))
			return false;

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = PgturbohybridGraphPageGetOpaque(page);
		if ((opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) != pageKind)
		{
			UnlockReleaseBuffer(buf);
			return false;
		}

		blknos[pageNo] = blkno;
		blkno = opaque->nextblkno;
		UnlockReleaseBuffer(buf);
	}

	return true;
}

BlockNumber
PgturbohybridGraphGetChainBlockNumber(Relation index, BlockNumber startBlkno, int pageNo,
						   int pageCount, uint16 pageKind)
{
	BlockNumber *blknos;
	BlockNumber blkno;

	if (pageNo < 0 || pageNo >= pageCount)
		return InvalidBlockNumber;

	blknos = palloc(sizeof(BlockNumber) * pageCount);
	PgturbohybridGraphInitBlockMap(blknos, pageCount);
	if (!PgturbohybridGraphEnsureBlockMap(index, startBlkno, pageCount, pageKind, blknos))
		blkno = InvalidBlockNumber;
	else
		blkno = blknos[pageNo];
	pfree(blknos);

	return blkno;
}

BlockNumber
PgturbohybridGraphGetMappedBlockNumber(BlockNumber startBlkno, int pageNo, BlockNumber *blknos)
{
	if (pageNo < 0)
		return InvalidBlockNumber;
	if (blknos != NULL && BlockNumberIsValid(blknos[pageNo]))
		return blknos[pageNo];
	if (!BlockNumberIsValid(startBlkno))
		return InvalidBlockNumber;

	return startBlkno + pageNo;
}

bool
PgturbohybridGraphResolveChainBlockNumber(Relation index, BlockNumber startBlkno,
							   int pageNo, int pageCount, uint16 pageKind,
							   BlockNumber *blknos, BlockNumber *blkno)
{
	if (blkno == NULL || pageNo < 0 || pageNo >= pageCount)
		return false;

	if (blknos != NULL)
	{
		if (!BlockNumberIsValid(blknos[pageNo]) &&
			!PgturbohybridGraphEnsureBlockMap(index, startBlkno, pageCount, pageKind, blknos))
			return false;

		*blkno = blknos[pageNo];
		return BlockNumberIsValid(*blkno);
	}

	*blkno = PgturbohybridGraphGetChainBlockNumber(index, startBlkno, pageNo,
									 pageCount, pageKind);
	return BlockNumberIsValid(*blkno);
}
void
PgturbohybridGraphFinishPage(Buffer buf)
{
	MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);
}

void
PgturbohybridGraphAppendPage(Relation index, ForkNumber forkNum, Buffer *buf, Page *page, uint16 pageKind)
{
	Buffer		newbuf;

	LockRelationForExtension(index, ExclusiveLock);
	newbuf = PgturbohybridGraphNewBuffer(index, forkNum);
	UnlockRelationForExtension(index, ExclusiveLock);

	if (BufferIsValid(*buf))
	{
		PgturbohybridGraphPageGetOpaque(*page)->nextblkno = BufferGetBlockNumber(newbuf);
		PgturbohybridGraphMarkPageGraphOp(*page, PGTURBOHYBRID_GRAPH_GRAPH_OP_PAGE_LINK);
		PgturbohybridGraphFinishPage(*buf);
	}

	*buf = newbuf;
	*page = BufferGetPage(newbuf);
	PgturbohybridGraphInitPageKind(newbuf, *page, pageKind);
}

OffsetNumber
PgturbohybridGraphAppendTuple(Relation index, ForkNumber forkNum, BlockNumber *startBlkno,
				   uint16 pageKind, Item tuple, Size tupleSize,
				   uint16 graphOpKind, BlockNumber *insertBlkno)
{
	Buffer		buf = InvalidBuffer;
	Page		page = NULL;
	BlockNumber blkno = *startBlkno;
	GenericXLogState *xlogState = NULL;
	OffsetNumber offno;
	bool		createdStart = false;
	Buffer		linkbuf = InvalidBuffer;
	BlockNumber linkBlkno = InvalidBlockNumber;
	BlockNumber initBlkno = InvalidBlockNumber;

	if (!BlockNumberIsValid(blkno))
	{
		LockRelationForExtension(index, ExclusiveLock);
		buf = PgturbohybridGraphNewBuffer(index, forkNum);
		UnlockRelationForExtension(index, ExclusiveLock);
		blkno = BufferGetBlockNumber(buf);
		*startBlkno = blkno;
		createdStart = true;
	}
	else
	{
		for (;;)
		{
			BlockNumber nextblkno;

			buf = ReadBufferExtended(index, forkNum, blkno, RBM_NORMAL, NULL);
			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);
			nextblkno = PgturbohybridGraphPageGetOpaque(page)->nextblkno;
			UnlockReleaseBuffer(buf);

			if (!BlockNumberIsValid(nextblkno))
				break;
			blkno = nextblkno;
		}

		buf = ReadBufferExtended(index, forkNum, blkno, RBM_NORMAL, NULL);
	}

	if (!createdStart)
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	if (RelationNeedsWAL(index) && forkNum == MAIN_FORKNUM)
	{
		xlogState = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(xlogState, buf,
										 createdStart ? GENERIC_XLOG_FULL_IMAGE : 0);
	}
	else
		page = BufferGetPage(buf);

	if (createdStart)
	{
		PgturbohybridGraphInitPageKind(buf, page, pageKind);
		initBlkno = blkno;
	}
	else if ((PgturbohybridGraphPageGetOpaque(page)->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) != pageKind)
		elog(ERROR, "unexpected pgturbohybrid graph page kind while appending");

	if (PageGetFreeSpace(page) < tupleSize)
	{
		Buffer		newbuf;
		Page		newpage;

		LockRelationForExtension(index, ExclusiveLock);
		newbuf = PgturbohybridGraphNewBuffer(index, forkNum);
		UnlockRelationForExtension(index, ExclusiveLock);

		if (xlogState != NULL)
			newpage = GenericXLogRegisterBuffer(xlogState, newbuf,
												GENERIC_XLOG_FULL_IMAGE);
		else
			newpage = BufferGetPage(newbuf);

		PgturbohybridGraphInitPageKind(newbuf, newpage, pageKind);
		PgturbohybridGraphPageGetOpaque(page)->nextblkno = BufferGetBlockNumber(newbuf);
		PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_PAGE_LINK);
		linkbuf = buf;
		linkBlkno = blkno;
		initBlkno = BufferGetBlockNumber(newbuf);
		blkno = initBlkno;
		buf = newbuf;
		page = newpage;
	}

	offno = PageAddItem(page, tuple, tupleSize, InvalidOffsetNumber, false, false);
	if (offno == InvalidOffsetNumber)
		elog(ERROR, "failed to append pgturbohybrid graph tuple");

	PgturbohybridGraphMarkPageGraphOp(page, graphOpKind);

	if (xlogState != NULL)
		GenericXLogFinish(xlogState);
	else
	{
		if (BufferIsValid(linkbuf))
			MarkBufferDirty(linkbuf);
		MarkBufferDirty(buf);
	}

	UnlockReleaseBuffer(buf);
	if (BufferIsValid(linkbuf))
		UnlockReleaseBuffer(linkbuf);

	if (BlockNumberIsValid(linkBlkno))
		PgturbohybridGraphLogGraphWalRecord(index, forkNum, linkBlkno, PGTURBOHYBRID_GRAPH_GRAPH_OP_PAGE_LINK);
	if (BlockNumberIsValid(initBlkno))
		PgturbohybridGraphLogGraphWalRecord(index, forkNum, initBlkno, PGTURBOHYBRID_GRAPH_GRAPH_OP_PAGE_INIT);
	PgturbohybridGraphLogGraphWalRecord(index, forkNum, blkno, graphOpKind);

	*insertBlkno = blkno;
	return offno;
}
