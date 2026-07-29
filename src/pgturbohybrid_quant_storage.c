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
		buf = ReadBufferExtended(index, forkNum, blkno, RBM_NORMAL, NULL);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
		page = BufferGetPage(buf);

		for (;;)
		{
			BlockNumber nextblkno;
			uint16		kind;

			kind = PgturbohybridGraphPageGetOpaque(page)->pageKind &
				PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK;
			if (kind != pageKind)
			{
				/*
				 * Page kind mismatch during traversal. This can happen after
				 * REINDEX when the build path creates pages with a different
				 * kind tag than the insert path expects. Rather than killing
				 * the transaction, skip to the next page or create a new one.
				 */
				elog(WARNING,
					 "pgturbohybrid: page kind mismatch (got %u, expected %u) "
					 "on page %u — creating fresh page",
					 kind, pageKind, blkno);
				/* Fall through to the page-creation logic below by pretending
				 * there's no next page; we'll allocate a fresh one. */
				break;
			}

			nextblkno = PgturbohybridGraphPageGetOpaque(page)->nextblkno;
			if (!BlockNumberIsValid(nextblkno))
				break;

			UnlockReleaseBuffer(buf);
			blkno = nextblkno;
			buf = ReadBufferExtended(index, forkNum, blkno, RBM_NORMAL, NULL);
			LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
			page = BufferGetPage(buf);
		}
	}
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
		elog(WARNING, "pgturbohybrid: page kind mismatch on start page %u — reinitializing", blkno);

	if (PageGetFreeSpace(page) < tupleSize)
	{
		Buffer		newbuf;
		Page		newpage;

		/*
		 * Guard: if the tuple is larger than what a fresh page can hold,
		 * there's no point creating a new page — it would fail too.
		 * This happens with high-degree adjacency nodes whose neighbor
		 * arrays exceed BLCKSZ. Log a WARNING and skip the tuple rather
		 * than ERROR-ing the entire operation (which kills the calling
		 * trigger/transaction).
		 *
		 * A conta precisa incluir o ponteiro de linha: sem ele o teto ficava em
		 * 8160 e deixava passar tuplas de 8160 que `PageGetFreeSpace` de página
		 * nova (8156) recusa. Era assim que se chegava ao caminho de baixo, que
		 * vazava o lock da página anterior.
		 */
		Size		pageUsable = PgturbohybridGraphMaxItemSize();

		if (tupleSize > pageUsable)
		{
			elog(WARNING,
				 "pgturbohybrid: skipping graph tuple of size %zu (max %zu) — "
				 "node degree exceeds page capacity",
				 (Size) tupleSize, pageUsable);
			if (xlogState != NULL)
				GenericXLogAbort(xlogState);
			UnlockReleaseBuffer(buf);
			*insertBlkno = blkno;
			return InvalidOffsetNumber;
		}

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
	{
		/*
		 * PageAddItem failed even on a page with space. This can happen when
		 * the page's free space fragmentation prevents the contiguous allocation
		 * even though PageGetFreeSpace reports enough total space. Skip with
		 * WARNING instead of ERROR to avoid killing the calling transaction.
		 *
		 * `linkbuf` é a página anterior da cadeia, e ela está travada em modo
		 * exclusivo desde o bloco de cima. Esta saída liberava só `buf`, então o
		 * lock de `linkbuf` ficava preso pelo resto da sessão — lock de buffer não
		 * é liberado no commit. Foi essa fuga que parou a busca híbrida por 21
		 * minutos em 28/07: todo leitor da página 9067 travava para sempre, sem
		 * detentor visível em pg_locks e sem responder a cancelamento.
		 */
		elog(WARNING,
			 "pgturbohybrid: PageAddItem failed for tuple of size %zu on page %u — "
			 "free space was %zu, skipping tuple",
			 (Size) tupleSize, blkno, PageGetFreeSpace(page));
		if (xlogState != NULL)
			GenericXLogAbort(xlogState);
		UnlockReleaseBuffer(buf);
		if (BufferIsValid(linkbuf))
			UnlockReleaseBuffer(linkbuf);
		*insertBlkno = blkno;
		return InvalidOffsetNumber;
	}

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
