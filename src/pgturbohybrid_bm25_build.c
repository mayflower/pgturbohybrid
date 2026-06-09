#include "postgres.h"

#include <math.h>
#include <string.h>

#include "access/generic_xlog.h"
#include "access/genam.h"
#include "access/tableam.h"
#include "catalog/pg_type_d.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "storage/buffile.h"
#include "storage/lmgr.h"
#include "tsearch/ts_type.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "utils/rel.h"

#include "pgturbohybrid.h"
#include "pgturbohybrid_quant.h"
#include "pgturbohybrid_am.h"
#include "pgturbohybrid_bm25.h"

typedef struct PgturbohybridTidNode
{
	ItemPointerData tid;
	uint32		nodeId;
} PgturbohybridTidNode;

typedef struct PgturbohybridNodeState
{
	ItemPointerData tid;
	bool		live;
} PgturbohybridNodeState;

typedef struct PgturbohybridBm25Collector
{
	Relation	index;
	PgturbohybridTidNode *tidNodes;
	uint32		tidNodeCount;
	PgturbohybridBm25BuildDoc *docs;
	uint32		docCount;
	uint32		docCapacity;
	uint32	   *denseDocLens;
	uint32		denseDocLensCount;
	PgturbohybridBm25TermTuple *terms;
	uint32		termCount;
	uint32		termCapacity;
	uint32		totalTermCount;
	char	   *termBytes;
	uint32		termBytesUsed;
	uint32		termBytesCapacity;
	uint64		totalDocLen;
	uint32		maxDocLen;
	uint32		uniqueTerms;
	BlockNumber bm25MetaBlkno;
	BlockNumber bm25DocStatsStartBlkno;
	BlockNumber bm25LexiconStartBlkno;
	BlockNumber bm25PostingsStartBlkno;
	BlockNumber bm25BlockMaxStartBlkno;
	BlockNumber bm25ImpactStartBlkno;
	uint32		bm25PostingsPages;
	uint32		bm25BlockMaxPages;
	uint32		bm25ImpactPages;
	Size		softBudget;
	bool		allowSpill;
	bool		walLoggedWrites;
	struct PgturbohybridBm25SpillRun *spillRuns;
	uint32		spillRunCount;
	uint32		spillRunCapacity;
} PgturbohybridBm25Collector;

typedef struct PgturbohybridBm25SpillRun
{
	BufFile    *file;
	uint32		tupleCount;
} PgturbohybridBm25SpillRun;

typedef struct PgturbohybridBm25SpillCursor
{
	PgturbohybridBm25SpillRun *run;
	uint32		remaining;
	bool		valid;
	uint64		termHash;
	uint32		nodeId;
	uint16		tf;
	uint16		termLen;
	char	   *termBytes;
} PgturbohybridBm25SpillCursor;

typedef struct PgturbohybridBm25SpillHeap
{
	PgturbohybridBm25SpillCursor *cursors;
	int		   *items;
	uint32		count;
} PgturbohybridBm25SpillHeap;

static Oid	pgturbohybrid_bm25_delta_cursor_index = InvalidOid;
static Oid	pgturbohybrid_bm25_delta_cursor_relfilenumber = InvalidOid;
static BlockNumber pgturbohybrid_bm25_delta_cursor_start = InvalidBlockNumber;
static BlockNumber pgturbohybrid_bm25_delta_cursor_tail = InvalidBlockNumber;
static uint64 pgturbohybrid_bm25_delta_cursor_generation = 0;
static uint32 pgturbohybrid_bm25_delta_cursor_pages = 0;

void
PgturbohybridBm25ResetDeltaAppendCursor(Relation index)
{
	if (index != NULL &&
		pgturbohybrid_bm25_delta_cursor_index != RelationGetRelid(index))
		return;

	pgturbohybrid_bm25_delta_cursor_index = InvalidOid;
	pgturbohybrid_bm25_delta_cursor_relfilenumber = InvalidOid;
	pgturbohybrid_bm25_delta_cursor_start = InvalidBlockNumber;
	pgturbohybrid_bm25_delta_cursor_tail = InvalidBlockNumber;
	pgturbohybrid_bm25_delta_cursor_generation = 0;
	pgturbohybrid_bm25_delta_cursor_pages = 0;
}

static void
PgturbohybridBufFileReadExact(BufFile *file, void *ptr, size_t size)
{
#if PG_VERSION_NUM >= 160000
	BufFileReadExact(file, ptr, size);
#else
	if (BufFileRead(file, ptr, size) != size)
		elog(ERROR, "could not read pgturbohybrid BM25 spill file");
#endif
}

static bool PgturbohybridBm25PageIsKind(Page page, uint16 pageKind);
static bool PgturbohybridBm25ReadMeta(Relation index, PgturbohybridBm25MetaTupleData *meta,
								 BlockNumber *metaBlkno);
static bool PgturbohybridBm25ReadMetaForUpdate(Relation index, Buffer *outBuf,
										  Page *outPage,
										  PgturbohybridBm25MetaTuple *outTuple,
										  GenericXLogState **outXlogState);
static void PgturbohybridBm25SetMetaBlock(Relation index, BlockNumber metaBlkno);
static void PgturbohybridBm25EnsureWalTail(Relation index, ForkNumber forkNum,
									  Buffer *buf, Page *page,
									  BlockNumber *startBlkno,
									  uint16 pageKind);
static OffsetNumber PgturbohybridBm25AddWalItem(Relation index, ForkNumber forkNum,
										   Buffer *buf, Page *page,
										   BlockNumber *startBlkno,
										   uint16 pageKind, Item item,
										   Size itemSize, uint32 *pageCount,
										   BlockNumber *insertBlkno);
static uint32 PgturbohybridBm25CountChainPagesAndTail(Relation index,
												 BlockNumber startBlkno,
												 uint16 pageKind,
												 BlockNumber *tailBlkno);
static Size
PgturbohybridBm25DocStatsTupleSize(uint16 count)
{
	return MAXALIGN(offsetof(PgturbohybridBm25DocStatsTupleData, docs) +
					sizeof(TqBm25DocStat) * count);
}

static Size
PgturbohybridBm25PostingsTupleSize(Size payloadBytes)
{
	return MAXALIGN(offsetof(PgturbohybridBm25PostingsTupleData, payload) +
					payloadBytes);
}

static Size
PgturbohybridBm25LegacyPostingsTupleSize(uint16 count)
{
	return MAXALIGN(offsetof(PgturbohybridBm25PostingsTupleData, encoding) +
					sizeof(PgturbohybridBm25Posting) * count);
}

static Size
PgturbohybridBm25VarintLen(uint32 value)
{
	Size		len = 1;

	while (value >= 0x80)
	{
		value >>= 7;
		len++;
	}
	return len;
}

static void
PgturbohybridBm25PutVarint(char **ptr, uint32 value)
{
	while (value >= 0x80)
	{
		*(*ptr)++ = (char) ((value & 0x7f) | 0x80);
		value >>= 7;
	}
	*(*ptr)++ = (char) value;
}

static void
PgturbohybridBm25PutUint16(char **ptr, uint16 value)
{
	memcpy(*ptr, &value, sizeof(value));
	*ptr += sizeof(value);
}

static bool
PgturbohybridBm25CanUseDelta16(uint32 prevNodeId, uint32 nodeId)
{
	return nodeId >= prevNodeId && (nodeId - prevNodeId) <= PG_UINT16_MAX;
}

static bool
PgturbohybridBm25CanUseOffset16(uint32 firstNodeId, uint32 nodeId)
{
	return nodeId >= firstNodeId && (nodeId - firstNodeId) <= PG_UINT16_MAX;
}

static bool
PgturbohybridBm25PrecomputeTfNormEnabled(PgturbohybridBm25Collector *collector)
{
	PgturbohybridOptions *opts = (PgturbohybridOptions *) collector->index->rd_options;

	return opts == NULL || opts->bm25PrecomputeTfNorm;
}

static uint16
PgturbohybridBm25QuantizeTfNorm(PgturbohybridBm25Collector *collector, uint32 nodeId,
						   uint16 tf)
{
	PgturbohybridOptions *opts = (PgturbohybridOptions *) collector->index->rd_options;
	double		k1 = opts != NULL ? opts->bm25K1 : 1.2;
	double		b = opts != NULL ? opts->bm25B : 0.75;
	double		avgDocLen = collector->docCount == 0 ? 1.0 :
		(double) collector->totalDocLen / (double) collector->docCount;
	double		docLen = 1.0;
	double		norm;
	double		tfNorm;
	double		scale;
	long		quantized;

	if (collector->denseDocLens != NULL && nodeId < collector->denseDocLensCount &&
		collector->denseDocLens[nodeId] > 0)
		docLen = (double) collector->denseDocLens[nodeId];
	norm = k1 * (1.0 - b + b * docLen / Max(avgDocLen, 1.0));
	tfNorm = ((double) tf * (k1 + 1.0)) / ((double) tf + norm);
	scale = Max(k1 + 1.0, 1.0);
	quantized = lround((tfNorm / scale) * (double) PG_UINT16_MAX);
	if (quantized < 0)
		return 0;
	if (quantized > PG_UINT16_MAX)
		return PG_UINT16_MAX;
	return (uint16) quantized;
}

static bool
PgturbohybridBm25GetVarint(const char **ptr, const char *end, uint32 *value)
{
	uint32		result = 0;
	uint32		shift = 0;

	while (*ptr < end && shift <= 28)
	{
		unsigned char byte = (unsigned char) *(*ptr)++;

		result |= (uint32) (byte & 0x7f) << shift;
		if ((byte & 0x80) == 0)
		{
			*value = result;
			return true;
		}
		shift += 7;
	}
	return false;
}

static void
PgturbohybridBm25EncodePosting(char **ptr, uint32 *prevNodeId,
						  uint32 nodeId, uint16 tf)
{
	uint32		delta;

	if (nodeId < *prevNodeId)
		elog(ERROR, "pgturbohybrid BM25 postings are not sorted by node id");
	delta = nodeId - *prevNodeId;
	PgturbohybridBm25PutVarint(ptr, delta);
	PgturbohybridBm25PutVarint(ptr, tf);
	*prevNodeId = nodeId;
}

static void
PgturbohybridBm25EncodePostingDelta16(char **ptr, uint32 *prevNodeId,
								 uint32 nodeId, uint16 tf)
{
	uint16		delta;

	if (!PgturbohybridBm25CanUseDelta16(*prevNodeId, nodeId))
		elog(ERROR, "pgturbohybrid BM25 postings are not delta16 encodable");
	delta = (uint16) (nodeId - *prevNodeId);
	PgturbohybridBm25PutUint16(ptr, delta);
	PgturbohybridBm25PutUint16(ptr, tf);
	*prevNodeId = nodeId;
}

static void
PgturbohybridBm25EncodePostingOffset16(char **ptr, uint32 firstNodeId,
								  uint32 nodeId, uint16 tf)
{
	uint16		offset;

	if (!PgturbohybridBm25CanUseOffset16(firstNodeId, nodeId))
		elog(ERROR, "pgturbohybrid BM25 postings are not offset16 encodable");
	offset = (uint16) (nodeId - firstNodeId);
	PgturbohybridBm25PutUint16(ptr, offset);
	PgturbohybridBm25PutUint16(ptr, tf);
}

static bool
PgturbohybridBm25DecodePostingsTuple(PgturbohybridBm25PostingsTuple tuple,
								Size itemSize,
								PgturbohybridBm25Posting *postings)
{
	const char *ptr = tuple->payload;
	const char *end = tuple->payload + tuple->payloadBytes;
	uint32		prevNodeId = 0;
	uint16		encoding = tuple->encoding & PGTURBOHYBRID_BM25_POSTINGS_ENCODING_MASK;
	bool		hasTfNorm = (tuple->encoding & PGTURBOHYBRID_BM25_POSTINGS_ENCODING_TFNORM_Q16) != 0;
	Size		tfNormBytes = hasTfNorm ? tuple->count * sizeof(uint16) : 0;
	const char *dataEnd = end;

	if (PgturbohybridBm25PostingsTupleSize(tuple->payloadBytes) != itemSize)
	{
		if (PgturbohybridBm25LegacyPostingsTupleSize(tuple->count) == itemSize)
		{
			memcpy(postings, &tuple->encoding,
				   sizeof(PgturbohybridBm25Posting) * tuple->count);
			return true;
		}
		return false;
	}
	if (tfNormBytes > (Size) (end - ptr))
		return false;
	if (hasTfNorm)
		dataEnd = end - tfNormBytes;

	if (encoding == PGTURBOHYBRID_BM25_POSTINGS_ENCODING_OFFSET16)
	{
		uint32		prevDecoded = tuple->firstNodeId;

		if ((Size) (dataEnd - ptr) != tuple->count * sizeof(uint16) * 2)
			return false;
		for (uint16 i = 0; i < tuple->count; i++)
		{
			uint16		offset;
			uint16		tf;
			uint32		nodeId;

			memcpy(&offset, ptr, sizeof(offset));
			ptr += sizeof(offset);
			memcpy(&tf, ptr, sizeof(tf));
			ptr += sizeof(tf);
			if (i == 0 && offset != 0)
				return false;
			nodeId = tuple->firstNodeId + offset;
			if (nodeId < prevDecoded)
				return false;
			prevDecoded = nodeId;
			postings[i].nodeId = nodeId;
			postings[i].tf = tf;
			postings[i].reserved = 0;
		}
		if (ptr != dataEnd)
			return false;
		if (hasTfNorm)
		{
			for (uint16 i = 0; i < tuple->count; i++)
			{
				memcpy(&postings[i].reserved, ptr, sizeof(postings[i].reserved));
				ptr += sizeof(postings[i].reserved);
			}
		}
		return ptr == end;
	}

	if (encoding == PGTURBOHYBRID_BM25_POSTINGS_ENCODING_OFFSET16_SOA)
	{
		const char *offsetPtr = ptr;
		const char *tfPtr = ptr + tuple->count * sizeof(uint16);
		uint32		prevDecoded = tuple->firstNodeId;

		if ((Size) (dataEnd - ptr) != tuple->count * sizeof(uint16) * 2)
			return false;
		for (uint16 i = 0; i < tuple->count; i++)
		{
			uint16		offset;
			uint16		tf;
			uint32		nodeId;

			memcpy(&offset, offsetPtr, sizeof(offset));
			offsetPtr += sizeof(offset);
			memcpy(&tf, tfPtr, sizeof(tf));
			tfPtr += sizeof(tf);
			if (i == 0 && offset != 0)
				return false;
			nodeId = tuple->firstNodeId + offset;
			if (nodeId < prevDecoded)
				return false;
			prevDecoded = nodeId;
			postings[i].nodeId = nodeId;
			postings[i].tf = tf;
			postings[i].reserved = 0;
		}
		ptr = dataEnd;
		if (hasTfNorm)
		{
			for (uint16 i = 0; i < tuple->count; i++)
			{
				memcpy(&postings[i].reserved, ptr, sizeof(postings[i].reserved));
				ptr += sizeof(postings[i].reserved);
			}
		}
		return ptr == end;
	}

	if (encoding == PGTURBOHYBRID_BM25_POSTINGS_ENCODING_DELTA16)
	{
		if ((Size) (dataEnd - ptr) != tuple->count * sizeof(uint16) * 2)
			return false;
		prevNodeId = tuple->firstNodeId;
		for (uint16 i = 0; i < tuple->count; i++)
		{
			uint16		delta;
			uint16		tf;

			memcpy(&delta, ptr, sizeof(delta));
			ptr += sizeof(delta);
			memcpy(&tf, ptr, sizeof(tf));
			ptr += sizeof(tf);
			if (i == 0 && delta != 0)
				return false;
			if ((PG_UINT32_MAX - prevNodeId) < delta)
				return false;
			prevNodeId += delta;
			postings[i].nodeId = prevNodeId;
			postings[i].tf = tf;
			postings[i].reserved = 0;
		}
		if (ptr != dataEnd)
			return false;
		if (hasTfNorm)
		{
			for (uint16 i = 0; i < tuple->count; i++)
			{
				memcpy(&postings[i].reserved, ptr, sizeof(postings[i].reserved));
				ptr += sizeof(postings[i].reserved);
			}
		}
		return ptr == end;
	}

	if (encoding != PGTURBOHYBRID_BM25_POSTINGS_ENCODING_DELTA_VARINT)
		return false;

	prevNodeId = tuple->firstNodeId;
	for (uint16 i = 0; i < tuple->count; i++)
	{
		uint32		delta;
		uint32		tf;

		if (!PgturbohybridBm25GetVarint(&ptr, dataEnd, &delta) ||
			!PgturbohybridBm25GetVarint(&ptr, dataEnd, &tf) ||
			tf > PG_UINT16_MAX ||
			(PG_UINT32_MAX - prevNodeId) < delta)
			return false;

		if (i == 0 && delta != 0)
			return false;
		prevNodeId += delta;
		postings[i].nodeId = prevNodeId;
		postings[i].tf = (uint16) tf;
		postings[i].reserved = 0;
	}
	if (ptr != dataEnd)
		return false;
	if (hasTfNorm)
	{
		for (uint16 i = 0; i < tuple->count; i++)
		{
			memcpy(&postings[i].reserved, ptr, sizeof(postings[i].reserved));
			ptr += sizeof(postings[i].reserved);
		}
	}

	return ptr == end;
}

static uint16
PgturbohybridBm25MaxPostingsPerChunk(void)
{
	Size		maxItemSize = (BLCKSZ / 2) - SizeOfPageHeaderData -
		MAXALIGN(sizeof(PgturbohybridGraphPageOpaqueData));
	uint32		maxCount = (maxItemSize -
							offsetof(PgturbohybridBm25PostingsTupleData, payload)) /
		(PgturbohybridBm25VarintLen(PG_UINT32_MAX) +
		 PgturbohybridBm25VarintLen(PG_UINT16_MAX));

	maxCount = Max(maxCount, 1);
	return (uint16) Min(maxCount, (uint32) PG_UINT16_MAX);
}

static Size
PgturbohybridBm25LexiconEntrySize(uint16 termLen)
{
	return MAXALIGN(offsetof(PgturbohybridBm25LexiconEntryData, termBytes) + termLen);
}

static Size
PgturbohybridBm25ImpactTupleSize(uint16 count)
{
	return MAXALIGN(offsetof(PgturbohybridBm25ImpactTupleData, entries) +
					sizeof(PgturbohybridBm25ImpactTupleEntry) * count);
}

static Size
PgturbohybridBm25DeltaTupleSize(uint16 termCount, uint32 termBytesLen)
{
	return MAXALIGN(offsetof(PgturbohybridBm25DeltaTupleData, terms) +
					sizeof(PgturbohybridBm25DeltaTerm) * termCount +
					termBytesLen);
}

static Size
PgturbohybridBm25DeltaTermTupleSize(uint16 postingCount, uint32 termBytesLen)
{
	return MAXALIGN(offsetof(PgturbohybridBm25DeltaTermTupleData, postings) +
					sizeof(PgturbohybridBm25DeltaTermPosting) * postingCount +
					termBytesLen);
}

static char *
PgturbohybridBm25DeltaTermTupleBytes(PgturbohybridBm25DeltaTermTuple tuple)
{
	return ((char *) tuple) +
		offsetof(PgturbohybridBm25DeltaTermTupleData, postings) +
		sizeof(PgturbohybridBm25DeltaTermPosting) * tuple->postingCount;
}

static uint32
PgturbohybridBm25DeltaChunkTermCount(PgturbohybridBm25Collector *collector,
									 uint32 startTerm,
									 Size maxItemSize)
{
	uint32		count = 0;
	uint32		termBytes = 0;

	while (startTerm + count < collector->termCount)
	{
		PgturbohybridBm25TermTuple *term =
			&collector->terms[startTerm + count];
		Size		nextSize =
			PgturbohybridBm25DeltaTupleSize((uint16) (count + 1),
										   termBytes + term->termLen);

		if (count > 0 && nextSize > maxItemSize)
			break;

		if (count == 0 && nextSize > maxItemSize)
		{
			if (PgturbohybridBm25DeltaTermTupleSize(1, term->termLen) > maxItemSize)
				ereport(ERROR,
						(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
						 errmsg("pgturbohybrid BM25 delta term exceeds page size"),
						 errdetail("A single delta term for node %u is %u bytes long; the maximum index tuple size is %zu bytes.",
								   term->nodeId,
								   term->termLen, maxItemSize),
						 errhint("Shorten individual lexemes or tsvector values for a single row.")));
			return 1;
		}

		termBytes += term->termLen;
		count++;
	}

	if (count == 0)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pgturbohybrid BM25 delta chunk could not be formed")));

	return count;
}

static PgturbohybridBm25DeltaTuple
PgturbohybridBm25BuildDeltaChunk(PgturbohybridBm25Collector *collector,
								 uint32 nodeId,
								 ItemPointer heapTid,
								 uint32 docLen,
								 uint32 startTerm,
								 uint32 chunkTermCount,
								 Size *outSize)
{
	PgturbohybridBm25DeltaTuple delta;
	char	   *deltaBytes;
	uint32		termBytesUsed = 0;
	Size		deltaSize;

	for (uint32 i = 0; i < chunkTermCount; i++)
		termBytesUsed += collector->terms[startTerm + i].termLen;

	deltaSize = PgturbohybridBm25DeltaTupleSize((uint16) chunkTermCount,
												termBytesUsed);
	delta = palloc0(deltaSize);
	delta->type = PGTURBOHYBRID_BM25_DELTA_TUPLE_TYPE;
	delta->termCount = (uint16) chunkTermCount;
	delta->nodeId = nodeId;
	delta->heaptid = *heapTid;
	delta->docLen = docLen;
	delta->termBytesLen = termBytesUsed;
	deltaBytes = ((char *) delta) +
		offsetof(PgturbohybridBm25DeltaTupleData, terms) +
		sizeof(PgturbohybridBm25DeltaTerm) * chunkTermCount;

	for (uint32 i = 0; i < chunkTermCount; i++)
	{
		PgturbohybridBm25TermTuple *src = &collector->terms[startTerm + i];
		PgturbohybridBm25DeltaTerm *dst = &delta->terms[i];
		uint32		dstOffset = 0;

		for (uint32 j = 0; j < i; j++)
			dstOffset += collector->terms[startTerm + j].termLen;

		dst->termHash = src->termHash;
		dst->termOffset = dstOffset;
		dst->tf = src->tf;
		dst->termLen = src->termLen;
		memcpy(deltaBytes + dstOffset,
			   collector->termBytes + src->termOffset,
			   src->termLen);
	}

	*outSize = deltaSize;
	return delta;
}

static bool
PgturbohybridBm25ImpactHeadEnabled(PgturbohybridBm25Collector *collector,
							  uint32 *minDf, uint32 *headK)
{
	PgturbohybridOptions *opts = (PgturbohybridOptions *) collector->index->rd_options;

	if (opts != NULL)
	{
		*minDf = (uint32) Max(opts->bm25ImpactMinDf, 1);
		*headK = (uint32) Max(opts->bm25ImpactHeadK, 1);
		return opts->bm25ImpactHead;
	}
	*minDf = 1024;
	*headK = 2048;
	return true;
}

static double
PgturbohybridBm25BuildPostingScore(PgturbohybridBm25Collector *collector,
							  uint32 nodeId, uint16 tf)
{
	PgturbohybridOptions *opts = (PgturbohybridOptions *) collector->index->rd_options;
	double		k1 = opts != NULL ? opts->bm25K1 : 1.2;
	double		b = opts != NULL ? opts->bm25B : 0.75;
	double		avgDocLen = collector->docCount == 0 ? 1.0 :
		(double) collector->totalDocLen / (double) collector->docCount;
	double		docLen = 1.0;
	double		norm;

	if (collector->denseDocLens != NULL && nodeId < collector->denseDocLensCount &&
		collector->denseDocLens[nodeId] > 0)
		docLen = (double) collector->denseDocLens[nodeId];
	norm = k1 * (1.0 - b + b * docLen / Max(avgDocLen, 1.0));
	return ((double) tf * (k1 + 1.0)) / ((double) tf + norm);
}

static int
PgturbohybridBm25ImpactTupleEntryCompare(const void *a, const void *b)
{
	const PgturbohybridBm25ImpactTupleEntry *ia =
		(const PgturbohybridBm25ImpactTupleEntry *) a;
	const PgturbohybridBm25ImpactTupleEntry *ib =
		(const PgturbohybridBm25ImpactTupleEntry *) b;

	if (ia->score < ib->score)
		return 1;
	if (ia->score > ib->score)
		return -1;
	if (ia->nodeId < ib->nodeId)
		return -1;
	if (ia->nodeId > ib->nodeId)
		return 1;
	return 0;
}

static int
PgturbohybridTidNodeCompare(const void *a, const void *b)
{
	const PgturbohybridTidNode *ta = (const PgturbohybridTidNode *) a;
	const PgturbohybridTidNode *tb = (const PgturbohybridTidNode *) b;
	int			blockCmp;

	blockCmp = (ItemPointerGetBlockNumber(&ta->tid) > ItemPointerGetBlockNumber(&tb->tid)) -
		(ItemPointerGetBlockNumber(&ta->tid) < ItemPointerGetBlockNumber(&tb->tid));
	if (blockCmp != 0)
		return blockCmp;

	return (ItemPointerGetOffsetNumber(&ta->tid) > ItemPointerGetOffsetNumber(&tb->tid)) -
		(ItemPointerGetOffsetNumber(&ta->tid) < ItemPointerGetOffsetNumber(&tb->tid));
}

static int
PgturbohybridTermCompareWithBytes(const PgturbohybridBm25TermTuple *ta,
							 const PgturbohybridBm25TermTuple *tb,
							 const char *termBytes)
{
	int			cmp;

	if (ta->termHash != tb->termHash)
		return ta->termHash < tb->termHash ? -1 : 1;
	if (ta->termLen != tb->termLen)
		return ta->termLen < tb->termLen ? -1 : 1;

	cmp = memcmp(termBytes + ta->termOffset, termBytes + tb->termOffset,
				 ta->termLen);
	if (cmp != 0)
		return cmp;

	return (ta->nodeId > tb->nodeId) - (ta->nodeId < tb->nodeId);
}

static bool
PgturbohybridTermEqualIgnoringNode(const PgturbohybridBm25TermTuple *ta,
							  const PgturbohybridBm25TermTuple *tb,
							  const char *termBytes)
{
	if (ta->termHash != tb->termHash || ta->termLen != tb->termLen)
		return false;

	return memcmp(termBytes + ta->termOffset, termBytes + tb->termOffset,
				  ta->termLen) == 0;
}

static PgturbohybridBm25Collector *pgturbohybrid_active_sort_collector = NULL;

static int
PgturbohybridTermCompare(const void *a, const void *b)
{
	return PgturbohybridTermCompareWithBytes((const PgturbohybridBm25TermTuple *) a,
										(const PgturbohybridBm25TermTuple *) b,
										pgturbohybrid_active_sort_collector->termBytes);
}

static uint64
PgturbohybridHashTerm(const char *term, uint16 len)
{
	uint64		hash = UINT64CONST(1469598103934665603);

	for (uint16 i = 0; i < len; i++)
	{
		hash ^= (unsigned char) term[i];
		hash *= UINT64CONST(1099511628211);
	}

	return hash;
}

static Size
PgturbohybridBm25BuildArrayAllocSize(Size elemSize, Size count)
{
	if (elemSize != 0 && count > MaxAllocSize / elemSize)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pgturbohybrid BM25 build array is too large")));
	return elemSize * count;
}

static uint32
PgturbohybridBm25BuildGrowCapacity32(uint32 capacity, uint32 needed, Size elemSize)
{
	uint32		newCapacity = capacity == 0 ? 1 : capacity;

	while (newCapacity < needed)
	{
		if (newCapacity > PG_UINT32_MAX / 2)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("pgturbohybrid BM25 build capacity is too large")));
		newCapacity += newCapacity;
	}

	(void) PgturbohybridBm25BuildArrayAllocSize(elemSize, newCapacity);
	return newCapacity;
}

static void PgturbohybridSpillTermRun(PgturbohybridBm25Collector *collector);

static Size
PgturbohybridBm25MaintenanceWorkMemBytes(void)
{
	int			kilobytes = maintenance_work_mem;

	if (kilobytes <= 0)
	{
		const char *value = GetConfigOptionByName("maintenance_work_mem",
												  NULL, false);

		if (value == NULL ||
			!parse_int(value, &kilobytes, GUC_UNIT_KB, NULL) ||
			kilobytes <= 0)
			kilobytes = 64 * 1024;
	}

	return (Size) kilobytes * (Size) 1024;
}

static void
PgturbohybridCheckBudget(PgturbohybridBm25Collector *collector)
{
	Size		used = (Size) collector->docCapacity * sizeof(PgturbohybridBm25BuildDoc) +
		(Size) collector->termCount * sizeof(PgturbohybridBm25TermTuple) +
		collector->termBytesUsed +
		(Size) collector->tidNodeCount * sizeof(PgturbohybridTidNode);

	if (used > collector->softBudget)
	{
		if (collector->allowSpill && collector->termCount > 0)
		{
			PgturbohybridSpillTermRun(collector);
			return;
		}
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pgturbohybrid BM25 build collector exceeded maintenance_work_mem"),
				 errdetail("Doc metadata and native graph TID map require %zu bytes before BM25 term spill can help.",
						   used),
				 errhint("Increase maintenance_work_mem for this build.")));
	}
}

static void
PgturbohybridEnsureSpillRunCapacity(PgturbohybridBm25Collector *collector)
{
	if (collector->spillRunCount < collector->spillRunCapacity)
		return;

	collector->spillRunCapacity =
		PgturbohybridBm25BuildGrowCapacity32(collector->spillRunCapacity,
											 collector->spillRunCount + 1,
											 sizeof(PgturbohybridBm25SpillRun));
	if (collector->spillRuns == NULL)
		collector->spillRuns =
			palloc(PgturbohybridBm25BuildArrayAllocSize(sizeof(PgturbohybridBm25SpillRun),
														collector->spillRunCapacity));
	else
		collector->spillRuns = repalloc(collector->spillRuns,
										PgturbohybridBm25BuildArrayAllocSize(sizeof(PgturbohybridBm25SpillRun),
																			 collector->spillRunCapacity));
}

static void
PgturbohybridSortTermRun(PgturbohybridBm25Collector *collector)
{
	if (collector->termCount <= 1)
		return;

	pgturbohybrid_active_sort_collector = collector;
	qsort(collector->terms, collector->termCount,
		  sizeof(PgturbohybridBm25TermTuple), PgturbohybridTermCompare);
	pgturbohybrid_active_sort_collector = NULL;
}

static void
PgturbohybridSpillTermRun(PgturbohybridBm25Collector *collector)
{
	PgturbohybridBm25SpillRun *run;

	if (collector->termCount == 0)
		return;

	PgturbohybridSortTermRun(collector);
	PgturbohybridEnsureSpillRunCapacity(collector);
	run = &collector->spillRuns[collector->spillRunCount++];
	run->file = BufFileCreateTemp(false);
	run->tupleCount = collector->termCount;

	for (uint32 i = 0; i < collector->termCount; i++)
	{
		PgturbohybridBm25TermTuple *term = &collector->terms[i];
		char	   *bytes = collector->termBytes + term->termOffset;

		BufFileWrite(run->file, &term->termHash, sizeof(term->termHash));
		BufFileWrite(run->file, &term->nodeId, sizeof(term->nodeId));
		BufFileWrite(run->file, &term->tf, sizeof(term->tf));
		BufFileWrite(run->file, &term->termLen, sizeof(term->termLen));
		BufFileWrite(run->file, bytes, term->termLen);
	}

	if (BufFileSeek(run->file, 0, 0L, SEEK_SET) != 0)
		elog(ERROR, "failed to rewind pgturbohybrid BM25 spill run");

	collector->termCount = 0;
	collector->termBytesUsed = 0;
}

static void
PgturbohybridCloseSpillRuns(PgturbohybridBm25Collector *collector)
{
	for (uint32 i = 0; i < collector->spillRunCount; i++)
	{
		if (collector->spillRuns[i].file != NULL)
		{
			BufFileClose(collector->spillRuns[i].file);
			collector->spillRuns[i].file = NULL;
		}
	}
}

static void
PgturbohybridEnsureDocCapacity(PgturbohybridBm25Collector *collector)
{
	if (collector->docCount < collector->docCapacity)
		return;

	collector->docCapacity =
		PgturbohybridBm25BuildGrowCapacity32(collector->docCapacity == 0 ? 1024 :
											 collector->docCapacity,
											 collector->docCount + 1,
											 sizeof(PgturbohybridBm25BuildDoc));
	if (collector->docs == NULL)
		collector->docs =
			palloc(PgturbohybridBm25BuildArrayAllocSize(sizeof(PgturbohybridBm25BuildDoc),
														collector->docCapacity));
	else
		collector->docs = repalloc(collector->docs,
								   PgturbohybridBm25BuildArrayAllocSize(sizeof(PgturbohybridBm25BuildDoc),
																		collector->docCapacity));
	PgturbohybridCheckBudget(collector);
}

static void
PgturbohybridEnsureTermCapacity(PgturbohybridBm25Collector *collector)
{
	if (collector->termCount < collector->termCapacity)
		return;

	collector->termCapacity =
		PgturbohybridBm25BuildGrowCapacity32(collector->termCapacity == 0 ? 4096 :
											 collector->termCapacity,
											 collector->termCount + 1,
											 sizeof(PgturbohybridBm25TermTuple));
	if (collector->terms == NULL)
		collector->terms =
			palloc(PgturbohybridBm25BuildArrayAllocSize(sizeof(PgturbohybridBm25TermTuple),
														collector->termCapacity));
	else
		collector->terms = repalloc(collector->terms,
									PgturbohybridBm25BuildArrayAllocSize(sizeof(PgturbohybridBm25TermTuple),
																		 collector->termCapacity));
	PgturbohybridCheckBudget(collector);
}

static uint32
PgturbohybridAppendTermBytes(PgturbohybridBm25Collector *collector, const char *term,
						uint16 len)
{
	uint32		offset;

	if ((uint64) collector->termBytesUsed + len > PG_UINT32_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pgturbohybrid BM25 term byte arena exceeded 4GB")));

	while (collector->termBytesUsed + len > collector->termBytesCapacity)
	{
		uint32		needed = collector->termBytesUsed + len;

		collector->termBytesCapacity =
			PgturbohybridBm25BuildGrowCapacity32(collector->termBytesCapacity == 0 ?
												 64 * 1024 : collector->termBytesCapacity,
												 needed, 1);
		if (collector->termBytes == NULL)
			collector->termBytes = palloc(collector->termBytesCapacity);
		else
			collector->termBytes = repalloc(collector->termBytes,
											collector->termBytesCapacity);
		PgturbohybridCheckBudget(collector);
	}

	offset = collector->termBytesUsed;
	memcpy(collector->termBytes + offset, term, len);
	collector->termBytesUsed += len;

	return offset;
}

static bool
PgturbohybridLookupNodeId(PgturbohybridBm25Collector *collector, ItemPointer tid,
					 uint32 *nodeId)
{
	PgturbohybridTidNode key;
	PgturbohybridTidNode *found;

	key.tid = *tid;
	key.nodeId = 0;
	found = bsearch(&key, collector->tidNodes, collector->tidNodeCount,
					sizeof(PgturbohybridTidNode), PgturbohybridTidNodeCompare);
	if (found == NULL)
		return false;

	*nodeId = found->nodeId;
	return true;
}

static uint32
PgturbohybridDocLen(TSVector vector)
{
	uint32		docLen = 0;
	WordEntry  *entries = ARRPTR(vector);

	for (int i = 0; i < vector->size; i++)
	{
		uint16		tf = POSDATALEN(vector, &entries[i]);

		docLen += tf > 0 ? tf : 1;
	}

	return docLen;
}

static TSVector
PgturbohybridDetoastTSVector(Datum value, bool *mustFree)
{
	TSVector	vector;

	vector = (TSVector) PG_DETOAST_DATUM(value);
	*mustFree = PointerGetDatum(vector) != value;
	return vector;
}

static void
PgturbohybridValidateTSVector(TSVector vector)
{
	WordEntry  *entries = ARRPTR(vector);
	char	   *strings = STRPTR(vector);
	Size		vectorSize = VARSIZE_ANY(vector);
	Size		dataOffset = (Size) (strings - (char *) vector);
	Size		dataSize;

	if (dataOffset > vectorSize)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("pgturbohybrid BM25 tsvector payload is invalid")));
	dataSize = vectorSize - dataOffset;

	for (int i = 0; i < vector->size; i++)
	{
		WordEntry  *entry = &entries[i];
		Size		termEnd = (Size) entry->pos + entry->len;

		if (termEnd > dataSize)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("pgturbohybrid BM25 tsvector lexeme offset is invalid")));

		if (entry->haspos)
		{
			Size		posOffset = SHORTALIGN(termEnd);
			uint16		npos;
			Size		posEnd;

			if (posOffset + sizeof(uint16) > dataSize)
				ereport(ERROR,
						(errcode(ERRCODE_DATA_CORRUPTED),
						 errmsg("pgturbohybrid BM25 tsvector position offset is invalid")));
			npos = POSDATALEN(vector, entry);
			posEnd = posOffset + sizeof(uint16) +
				sizeof(WordEntryPos) * npos;
			if (posEnd > dataSize)
				ereport(ERROR,
						(errcode(ERRCODE_DATA_CORRUPTED),
						 errmsg("pgturbohybrid BM25 tsvector positions are invalid")));
		}
	}
}

static void
PgturbohybridCollectVectorTerms(PgturbohybridBm25Collector *collector, uint32 nodeId,
						   TSVector vector)
{
	WordEntry  *entries = ARRPTR(vector);
	char	   *strings = STRPTR(vector);

	for (int i = 0; i < vector->size; i++)
	{
		WordEntry  *entry = &entries[i];
		char	   *term = strings + entry->pos;
		uint16		tf = POSDATALEN(vector, entry);
		PgturbohybridBm25TermTuple *tuple;
		uint32		termOffset;

		if (entry->len <= 0)
			continue;

		PgturbohybridEnsureTermCapacity(collector);
		termOffset = PgturbohybridAppendTermBytes(collector, term, entry->len);
		tuple = &collector->terms[collector->termCount++];
		tuple->termHash = PgturbohybridHashTerm(term, entry->len);
		tuple->nodeId = nodeId;
		tuple->tf = tf > 0 ? tf : 1;
		tuple->termLen = entry->len;
		tuple->termOffset = termOffset;
		collector->totalTermCount++;
		PgturbohybridCheckBudget(collector);
	}
}

static void
PgturbohybridAppendBuildDoc(PgturbohybridBm25Collector *collector, uint32 nodeId,
					   ItemPointer heapTid, uint32 docLen)
{
	PgturbohybridBm25BuildDoc *doc;

	PgturbohybridEnsureDocCapacity(collector);
	doc = &collector->docs[collector->docCount++];
	doc->nodeId = nodeId;
	doc->heaptid = *heapTid;
	doc->docLen = docLen;
	collector->totalDocLen += docLen;
	collector->maxDocLen = Max(collector->maxDocLen, docLen);
}

static void
PgturbohybridAppendBuildTerm(PgturbohybridBm25Collector *collector, uint32 nodeId,
						const char *term, uint16 termLen, uint16 tf)
{
	PgturbohybridBm25TermTuple *tuple;
	uint32		termOffset;

	if (termLen == 0)
		return;

	PgturbohybridEnsureTermCapacity(collector);
	termOffset = PgturbohybridAppendTermBytes(collector, term, termLen);
	tuple = &collector->terms[collector->termCount++];
	tuple->termHash = PgturbohybridHashTerm(term, termLen);
	tuple->nodeId = nodeId;
	tuple->tf = tf > 0 ? tf : 1;
	tuple->termLen = termLen;
	tuple->termOffset = termOffset;
	collector->totalTermCount++;
	PgturbohybridCheckBudget(collector);
}

static void
PgturbohybridBm25BuildCallback(Relation index, ItemPointer tid, Datum *values,
						  bool *isnull, bool tupleIsAlive, void *opaque)
{
	PgturbohybridBm25Collector *collector = (PgturbohybridBm25Collector *) opaque;
	TSVector	vector;
	bool		mustFree;
	Datum		lexicalValue;
	uint32		nodeId;
	uint32		docLen;

	(void) tupleIsAlive;

	if (!PgturbohybridIndexGetLexicalDatum(index, values, isnull, &lexicalValue))
		return;

	vector = PgturbohybridDetoastTSVector(lexicalValue, &mustFree);
	PgturbohybridValidateTSVector(vector);
	if (!PgturbohybridLookupNodeId(collector, tid, &nodeId))
	{
		if (mustFree)
			pfree(vector);
		return;
	}

	docLen = PgturbohybridDocLen(vector);

	PgturbohybridAppendBuildDoc(collector, nodeId, tid, docLen);
	PgturbohybridCollectVectorTerms(collector, nodeId, vector);
	if (mustFree)
		pfree(vector);
}

static PgturbohybridTidNode *
PgturbohybridReadNodeMap(Relation index, uint32 *count)
{
	PgturbohybridGraphMetaPageData meta;
	uint32		seen = 0;
	PgturbohybridTidNode *map;
	int			codeTuplesPerPage;
	int			codePageCount;
	BlockNumber *codeBlknos;
	bool		tqWeighted;

	if (!PgturbohybridGraphReadMeta(index, &meta) ||
		meta.storageKind != PGTURBOHYBRID_GRAPH_STORAGE_QUANT_GRAPH_NATIVE ||
		meta.tqNodeCount == 0 ||
		!BlockNumberIsValid(meta.tqCodeStartBlkno))
	{
		*count = 0;
		return NULL;
	}

	map = palloc0(PgturbohybridCheckedArrayBytes(sizeof(PgturbohybridTidNode),
												 meta.tqNodeCount,
												 "pgturbohybrid BM25 TID map"));
	tqWeighted = (meta.tqFlags & PGTURBOHYBRID_GRAPH_TQ_WEIGHTED) != 0;
	codeTuplesPerPage =
		PgturbohybridGraphTuplesPerPage(PgturbohybridGraphCodeTupleSize(meta.dimensions,
												  meta.tqPayloadCount,
												  meta.tqBits,
												  tqWeighted,
												  meta.tqResidualRerankBytes));
	codePageCount = PgturbohybridGraphPageCount(meta.tqNodeCount, codeTuplesPerPage);
	codeBlknos = palloc(PgturbohybridCheckedArrayBytes(sizeof(BlockNumber),
													   codePageCount,
													   "pgturbohybrid BM25 code block map"));
	PgturbohybridGraphInitBlockMap(codeBlknos, codePageCount);

	for (int pageNo = 0; pageNo < codePageCount; pageNo++)
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		OffsetNumber maxoff;
		BlockNumber blkno;

		if (!PgturbohybridGraphResolveChainBlockNumber(index, meta.tqCodeStartBlkno,
											 pageNo, codePageCount,
											 PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_CODE,
											 codeBlknos, &blkno))
			break;

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
		for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off = OffsetNumberNext(off))
		{
			ItemId		iid = PageGetItemId(page, off);
			PgturbohybridGraphCodeTuple tuple;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (PgturbohybridGraphCodeTuple) PageGetItem(page, iid);
			if (tuple->type != PGTURBOHYBRID_GRAPH_CODE_TUPLE_TYPE ||
				tuple->nodeId >= meta.tqNodeCount)
				continue;

			map[seen].tid = tuple->heaptid;
			map[seen].nodeId = tuple->nodeId;
			seen++;
		}

		UnlockReleaseBuffer(buf);
	}

	pfree(codeBlknos);

	qsort(map, seen, sizeof(PgturbohybridTidNode), PgturbohybridTidNodeCompare);
	*count = seen;
	return map;
}

static PgturbohybridNodeState *
PgturbohybridReadNodeStates(Relation index, PgturbohybridGraphMetaPageData *meta, uint32 *count)
{
	PgturbohybridNodeState *states;
	int			codeTuplesPerPage;
	int			codePageCount;
	BlockNumber *codeBlknos;
	bool		tqWeighted;

	if (!PgturbohybridGraphReadMeta(index, meta) ||
		meta->storageKind != PGTURBOHYBRID_GRAPH_STORAGE_QUANT_GRAPH_NATIVE ||
		meta->tqNodeCount == 0 ||
		!BlockNumberIsValid(meta->tqCodeStartBlkno))
	{
		*count = 0;
		return NULL;
	}

	states = palloc0(PgturbohybridCheckedArrayBytes(sizeof(PgturbohybridNodeState),
													meta->tqNodeCount,
													"pgturbohybrid BM25 node state map"));
	tqWeighted = (meta->tqFlags & PGTURBOHYBRID_GRAPH_TQ_WEIGHTED) != 0;
	codeTuplesPerPage =
		PgturbohybridGraphTuplesPerPage(PgturbohybridGraphCodeTupleSize(meta->dimensions,
												  meta->tqPayloadCount,
												  meta->tqBits,
												  tqWeighted,
												  meta->tqResidualRerankBytes));
	codePageCount = PgturbohybridGraphPageCount(meta->tqNodeCount, codeTuplesPerPage);
	codeBlknos = palloc(PgturbohybridCheckedArrayBytes(sizeof(BlockNumber),
													   codePageCount,
													   "pgturbohybrid BM25 code block map"));
	PgturbohybridGraphInitBlockMap(codeBlknos, codePageCount);

	for (int pageNo = 0; pageNo < codePageCount; pageNo++)
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		OffsetNumber maxoff;
		BlockNumber blkno;

		if (!PgturbohybridGraphResolveChainBlockNumber(index, meta->tqCodeStartBlkno,
											 pageNo, codePageCount,
											 PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_CODE,
											 codeBlknos, &blkno))
			break;

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
		for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off = OffsetNumberNext(off))
		{
			ItemId		iid = PageGetItemId(page, off);
			PgturbohybridGraphCodeTuple tuple;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (PgturbohybridGraphCodeTuple) PageGetItem(page, iid);
			if (tuple->type != PGTURBOHYBRID_GRAPH_CODE_TUPLE_TYPE ||
				tuple->nodeId >= meta->tqNodeCount)
				continue;

			states[tuple->nodeId].tid = tuple->heaptid;
			states[tuple->nodeId].live =
				(tuple->flags & PGTURBOHYBRID_GRAPH_NODE_DEAD) == 0;
		}

		UnlockReleaseBuffer(buf);
	}

	pfree(codeBlknos);
	*count = meta->tqNodeCount;
	return states;
}

static uint32
PgturbohybridReduceUniqueTerms(PgturbohybridBm25Collector *collector)
{
	uint32		uniqueTerms = 0;

	if (collector->termCount == 0)
		return 0;

	PgturbohybridSortTermRun(collector);

	for (uint32 i = 0; i < collector->termCount;)
	{
		PgturbohybridBm25TermTuple *first = &collector->terms[i];
		uint32		df = 0;
		uint64		cf = 0;
		uint32		prevNode = PG_UINT32_MAX;

		uniqueTerms++;
		while (i < collector->termCount &&
			   PgturbohybridTermEqualIgnoringNode(first, &collector->terms[i],
											 collector->termBytes))
		{
			if (collector->terms[i].nodeId != prevNode)
			{
				df++;
				prevNode = collector->terms[i].nodeId;
			}
			cf += collector->terms[i].tf;
			i++;
		}

		(void) df;
		(void) cf;
	}

	return uniqueTerms;
}

static OffsetNumber
PgturbohybridBm25AddWalItem(Relation index, ForkNumber forkNum, Buffer *buf,
					   Page *page, BlockNumber *startBlkno,
					   uint16 pageKind, Item item, Size itemSize,
					   uint32 *pageCount, BlockNumber *insertBlkno)
{
	GenericXLogState *xlogState;
	OffsetNumber offno;
	BlockNumber blkno;
	bool		createdStart = !BlockNumberIsValid(*startBlkno);

	PgturbohybridBm25EnsureWalTail(index, forkNum, buf, page, startBlkno,
							  pageKind);
	if (createdStart && pageCount != NULL)
		(*pageCount)++;

	if (PageGetFreeSpace(*page) < itemSize)
	{
		Buffer		newbuf;
		Page		newpage;
		BlockNumber linkBlkno = BufferGetBlockNumber(*buf);
		BlockNumber initBlkno;

		xlogState = GenericXLogStart(index);
		*page = GenericXLogRegisterBuffer(xlogState, *buf, 0);

		LockRelationForExtension(index, ExclusiveLock);
		newbuf = PgturbohybridGraphNewBuffer(index, forkNum);
		UnlockRelationForExtension(index, ExclusiveLock);
		initBlkno = BufferGetBlockNumber(newbuf);
		newpage = GenericXLogRegisterBuffer(xlogState, newbuf,
											GENERIC_XLOG_FULL_IMAGE);

		PgturbohybridGraphPageGetOpaque(*page)->nextblkno = initBlkno;
		PgturbohybridGraphMarkPageGraphOp(*page, PGTURBOHYBRID_GRAPH_GRAPH_OP_PAGE_LINK);
		PgturbohybridGraphInitPageKind(newbuf, newpage, pageKind);

		offno = PageAddItem(newpage, item, itemSize, InvalidOffsetNumber,
							false, false);
		if (offno == InvalidOffsetNumber)
			elog(ERROR, "failed to append pgturbohybrid BM25 WAL item to \"%s\"",
				 RelationGetRelationName(index));
		PgturbohybridGraphMarkPageGraphOp(newpage, PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT);

		GenericXLogFinish(xlogState);
		PgturbohybridGraphLogGraphWalRecord(index, forkNum, linkBlkno,
							  PGTURBOHYBRID_GRAPH_GRAPH_OP_PAGE_LINK);
		PgturbohybridGraphLogGraphWalRecord(index, forkNum, initBlkno,
							  PGTURBOHYBRID_GRAPH_GRAPH_OP_PAGE_INIT);
		PgturbohybridGraphLogGraphWalRecord(index, forkNum, initBlkno,
							  PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT);

		UnlockReleaseBuffer(*buf);
		*buf = newbuf;
		*page = BufferGetPage(newbuf);
		blkno = initBlkno;
		if (pageCount != NULL)
			(*pageCount)++;
	}
	else
	{
		blkno = BufferGetBlockNumber(*buf);
		xlogState = GenericXLogStart(index);
		*page = GenericXLogRegisterBuffer(xlogState, *buf, 0);
		offno = PageAddItem(*page, item, itemSize, InvalidOffsetNumber,
							false, false);
		if (offno == InvalidOffsetNumber)
			elog(ERROR, "failed to append pgturbohybrid BM25 WAL item to \"%s\"",
				 RelationGetRelationName(index));
		PgturbohybridGraphMarkPageGraphOp(*page, PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT);
		GenericXLogFinish(xlogState);
		PgturbohybridGraphLogGraphWalRecord(index, forkNum, blkno,
							  PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT);
		*page = BufferGetPage(*buf);
	}

	if (insertBlkno != NULL)
		*insertBlkno = blkno;
	return offno;
}

static void
PgturbohybridBm25EnsureWalTail(Relation index, ForkNumber forkNum, Buffer *buf,
						  Page *page, BlockNumber *startBlkno,
						  uint16 pageKind)
{
	BlockNumber blkno = *startBlkno;

	if (BufferIsValid(*buf))
	{
		/*
		 * Refresh the page pointer after any prior GenericXLog update on this
		 * buffer (e.g. postings/impact chunk linking during compaction).
		 */
		*page = BufferGetPage(*buf);
		if (!PgturbohybridBm25PageIsKind(*page, pageKind))
		{
			UnlockReleaseBuffer(*buf);
			*buf = InvalidBuffer;
			*page = NULL;
		}
		else
			return;
	}

	if (!BlockNumberIsValid(blkno))
	{
		GenericXLogState *xlogState;

		LockRelationForExtension(index, ExclusiveLock);
		*buf = PgturbohybridGraphNewBuffer(index, forkNum);
		UnlockRelationForExtension(index, ExclusiveLock);
		blkno = BufferGetBlockNumber(*buf);
		*startBlkno = blkno;

		xlogState = GenericXLogStart(index);
		*page = GenericXLogRegisterBuffer(xlogState, *buf,
										  GENERIC_XLOG_FULL_IMAGE);
		PgturbohybridGraphInitPageKind(*buf, *page, pageKind);
		GenericXLogFinish(xlogState);
		PgturbohybridGraphLogGraphWalRecord(index, forkNum, blkno,
							  PGTURBOHYBRID_GRAPH_GRAPH_OP_PAGE_INIT);
		*page = BufferGetPage(*buf);
		return;
	}

	*buf = ReadBufferExtended(index, forkNum, blkno, RBM_NORMAL, NULL);
	LockBuffer(*buf, BUFFER_LOCK_EXCLUSIVE);
	*page = BufferGetPage(*buf);

	for (;;)
	{
		BlockNumber nextblkno;

		if (!PgturbohybridBm25PageIsKind(*page, pageKind))
		{
			UnlockReleaseBuffer(*buf);
			*buf = InvalidBuffer;
			*page = NULL;
			elog(ERROR, "unexpected pgturbohybrid BM25 page kind while appending");
		}

		nextblkno = PgturbohybridGraphPageGetOpaque(*page)->nextblkno;
		if (!BlockNumberIsValid(nextblkno))
			break;

		UnlockReleaseBuffer(*buf);
		blkno = nextblkno;
		*buf = ReadBufferExtended(index, forkNum, blkno, RBM_NORMAL, NULL);
		LockBuffer(*buf, BUFFER_LOCK_EXCLUSIVE);
		*page = BufferGetPage(*buf);
	}
}

static OffsetNumber
PgturbohybridBm25AddItem(Relation index, ForkNumber forkNum, Buffer *buf, Page *page,
					BlockNumber *startBlkno, uint16 pageKind, Item item,
					Size itemSize, uint32 *pageCount, bool useWal,
					BlockNumber *insertBlkno)
{
	OffsetNumber offno;
	Size		maxItemSize = BLCKSZ - SizeOfPageHeaderData -
		MAXALIGN(sizeof(PgturbohybridGraphPageOpaqueData));

	if (itemSize > maxItemSize)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pgturbohybrid BM25 tuple exceeds page size"),
				 errdetail("Postings are chunked before storage; an oversized tuple indicates a single metadata, lexicon, or delta tuple is too large for one index page."),
				 errhint("Shorten individual lexemes or tsvector values; common-term postings spill and chunk automatically.")));

	if (useWal)
		return PgturbohybridBm25AddWalItem(index, forkNum, buf, page, startBlkno,
									  pageKind, item, itemSize, pageCount,
									  insertBlkno);

	if (!BufferIsValid(*buf) || PageGetFreeSpace(*page) < itemSize)
	{
		PgturbohybridGraphAppendPage(index, forkNum, buf, page, pageKind);
		if (!BlockNumberIsValid(*startBlkno))
			*startBlkno = BufferGetBlockNumber(*buf);
		if (pageCount != NULL)
			(*pageCount)++;
	}

	offno = PageAddItem(*page, item, itemSize, InvalidOffsetNumber, false, false);
	if (offno == InvalidOffsetNumber)
		elog(ERROR, "failed to add pgturbohybrid BM25 item to \"%s\"",
			 RelationGetRelationName(index));
	PgturbohybridGraphMarkPageGraphOp(*page, PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT);
	MarkBufferDirty(*buf);

	if (insertBlkno != NULL)
		*insertBlkno = BufferGetBlockNumber(*buf);
	return offno;
}

static void
PgturbohybridWriteDocStats(PgturbohybridBm25Collector *collector)
{
	Buffer		buf = InvalidBuffer;
	Page		page = NULL;
	BlockNumber start = InvalidBlockNumber;
	TqBm25DocStat *dense;
	uint32		nodeId = 0;
	uint16		maxDocsPerTuple;

	if (collector->docCount == 0)
	{
		collector->bm25DocStatsStartBlkno = InvalidBlockNumber;
		return;
	}

	dense = palloc0(PgturbohybridCheckedArrayBytes(sizeof(TqBm25DocStat),
												   collector->tidNodeCount,
												   "pgturbohybrid BM25 dense doc stats"));
	for (uint32 i = 0; i < collector->docCount; i++)
	{
		if (collector->docs[i].nodeId >= collector->tidNodeCount)
			elog(ERROR, "pgturbohybrid BM25 doc nodeId out of range");
		dense[collector->docs[i].nodeId].docLen = collector->docs[i].docLen;
	}
	collector->denseDocLens =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
											   collector->tidNodeCount,
											   "pgturbohybrid BM25 dense doc lengths"));
	collector->denseDocLensCount = collector->tidNodeCount;
	for (uint32 i = 0; i < collector->tidNodeCount; i++)
		collector->denseDocLens[i] = dense[i].docLen;

	maxDocsPerTuple = (BLCKSZ / 2 - offsetof(PgturbohybridBm25DocStatsTupleData, docs)) /
		sizeof(TqBm25DocStat);
	while (nodeId < collector->tidNodeCount)
	{
		uint16		count = Min((uint32) maxDocsPerTuple,
								collector->tidNodeCount - nodeId);
		Size		size = PgturbohybridBm25DocStatsTupleSize(count);
		PgturbohybridBm25DocStatsTuple tuple = palloc0(size);

		tuple->type = PGTURBOHYBRID_BM25_DOCSTATS_TUPLE_TYPE;
		tuple->count = count;
		tuple->startNodeId = nodeId;
		memcpy(tuple->docs, &dense[nodeId], sizeof(TqBm25DocStat) * count);

		(void) PgturbohybridBm25AddItem(collector->index, MAIN_FORKNUM, &buf, &page,
								   &start, PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_DOCSTATS,
								   (Item) tuple, size, NULL,
								   collector->walLoggedWrites, NULL);
		pfree(tuple);
		nodeId += count;
	}

	if (BufferIsValid(buf))
		UnlockReleaseBuffer(buf);
	collector->bm25DocStatsStartBlkno = start;
	pfree(dense);
}

static OffsetNumber
PgturbohybridWritePostingsChunk(PgturbohybridBm25Collector *collector, uint32 termId,
						   uint32 chunkNo, uint32 startIndex, uint32 endIndex,
						   Buffer *buf, Page *page,
						   BlockNumber *postingsBlkno, uint32 *postingsBytes)
{
	uint32		count32 = endIndex - startIndex;
	uint16		count;
	Size		size;
	Size		payloadBytes = 0;
	PgturbohybridBm25PostingsTuple tuple;
	OffsetNumber offno;
	BlockNumber insertBlkno = InvalidBlockNumber;
	char	   *ptr;
	uint32		prevNodeId;
	bool		useOffset16 = true;
	bool		useDelta16 = true;
	bool		hasTfNorm = PgturbohybridBm25PrecomputeTfNormEnabled(collector);
	bool		useOffset16Soa;

	count = (uint16) count32;
	prevNodeId = collector->terms[startIndex].nodeId;
	for (uint32 i = 0; i < count32; i++)
	{
		PgturbohybridBm25TermTuple *term = &collector->terms[startIndex + i];

		if (!PgturbohybridBm25CanUseOffset16(collector->terms[startIndex].nodeId,
										term->nodeId))
			useOffset16 = false;
		if (!PgturbohybridBm25CanUseDelta16(prevNodeId, term->nodeId))
			useDelta16 = false;
		prevNodeId = term->nodeId;
	}
	if (useOffset16 || useDelta16)
		payloadBytes = count32 * sizeof(uint16) * 2;
	else
	{
		prevNodeId = collector->terms[startIndex].nodeId;
		for (uint32 i = 0; i < count32; i++)
		{
			PgturbohybridBm25TermTuple *term = &collector->terms[startIndex + i];
			uint32		delta = term->nodeId - prevNodeId;

			payloadBytes += PgturbohybridBm25VarintLen(delta);
			payloadBytes += PgturbohybridBm25VarintLen(term->tf);
			prevNodeId = term->nodeId;
		}
	}
	if (hasTfNorm)
		payloadBytes += count32 * sizeof(uint16);
	size = PgturbohybridBm25PostingsTupleSize(payloadBytes);
	tuple = palloc0(size);
	tuple->type = PGTURBOHYBRID_BM25_POSTINGS_TUPLE_TYPE;
	tuple->count = count;
	tuple->termId = termId;
	tuple->chunkNo = chunkNo;
	tuple->firstNodeId = collector->terms[startIndex].nodeId;
	tuple->lastNodeId = collector->terms[endIndex - 1].nodeId;
	tuple->nextBlkno = InvalidBlockNumber;
	tuple->nextOffno = InvalidOffsetNumber;
	useOffset16Soa = useOffset16 && hasTfNorm;
	tuple->encoding = useOffset16 ?
		(useOffset16Soa ?
		 PGTURBOHYBRID_BM25_POSTINGS_ENCODING_OFFSET16_SOA :
		 PGTURBOHYBRID_BM25_POSTINGS_ENCODING_OFFSET16) :
		(useDelta16 ?
		 PGTURBOHYBRID_BM25_POSTINGS_ENCODING_DELTA16 :
		 PGTURBOHYBRID_BM25_POSTINGS_ENCODING_DELTA_VARINT);
	if (hasTfNorm)
		tuple->encoding |= PGTURBOHYBRID_BM25_POSTINGS_ENCODING_TFNORM_Q16;
	tuple->payloadBytes = (uint16) payloadBytes;
	ptr = tuple->payload;
	prevNodeId = tuple->firstNodeId;
	if (useOffset16Soa)
	{
		char	   *tfPtr = ptr + count32 * sizeof(uint16);

		for (uint32 i = 0; i < count32; i++)
		{
			PgturbohybridBm25TermTuple *term = &collector->terms[startIndex + i];
			uint16		offset;

			if (!PgturbohybridBm25CanUseOffset16(tuple->firstNodeId, term->nodeId))
				elog(ERROR, "pgturbohybrid BM25 postings are not offset16 encodable");
			offset = (uint16) (term->nodeId - tuple->firstNodeId);
			tuple->maxTf = Max(tuple->maxTf, term->tf);
			PgturbohybridBm25PutUint16(&ptr, offset);
			PgturbohybridBm25PutUint16(&tfPtr, term->tf);
		}
		ptr = tfPtr;
	}
	else
	{
		for (uint32 i = 0; i < count32; i++)
		{
			PgturbohybridBm25TermTuple *term = &collector->terms[startIndex + i];

			tuple->maxTf = Max(tuple->maxTf, term->tf);
			if (useOffset16)
				PgturbohybridBm25EncodePostingOffset16(&ptr, tuple->firstNodeId,
												  term->nodeId, term->tf);
			else if (useDelta16)
				PgturbohybridBm25EncodePostingDelta16(&ptr, &prevNodeId, term->nodeId,
												 term->tf);
			else
				PgturbohybridBm25EncodePosting(&ptr, &prevNodeId, term->nodeId,
										  term->tf);
		}
	}
	if (hasTfNorm)
	{
		PgturbohybridOptions *opts = (PgturbohybridOptions *) collector->index->rd_options;
		double		k1 = opts != NULL ? opts->bm25K1 : 1.2;

		for (uint32 i = 0; i < count32; i++)
		{
			PgturbohybridBm25TermTuple *term = &collector->terms[startIndex + i];
			uint16		tfNormQ16 = PgturbohybridBm25QuantizeTfNorm(collector,
															  term->nodeId,
															  term->tf);

			tuple->maxTfNormQ16 = Max(tuple->maxTfNormQ16, tfNormQ16);
			PgturbohybridBm25PutUint16(&ptr, tfNormQ16);
		}
		tuple->maxScoreFactor =
			(float4) (Max(k1 + 1.0, 1.0) *
					  ((double) tuple->maxTfNormQ16 / (double) PG_UINT16_MAX));
	}

	offno = PgturbohybridBm25AddItem(collector->index, MAIN_FORKNUM, buf, page,
								&collector->bm25PostingsStartBlkno,
								PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_POSTINGS,
								(Item) tuple, size,
								&collector->bm25PostingsPages,
								collector->walLoggedWrites,
								&insertBlkno);
	*postingsBlkno = insertBlkno;
	*postingsBytes = size;
	pfree(tuple);
	return offno;
}

static void
PgturbohybridLinkPostingsChunk(Relation index, Buffer currentBuf, Page *currentPage,
						  BlockNumber prevBlkno, OffsetNumber prevOffno,
						  BlockNumber nextBlkno, OffsetNumber nextOffno)
{
	Buffer		buf;
	Page		page;
	ItemId		iid;
	PgturbohybridBm25PostingsTuple tuple;
	GenericXLogState *xlogState = NULL;
	bool		modified = false;
	bool		current = BufferIsValid(currentBuf) &&
		BufferGetBlockNumber(currentBuf) == prevBlkno;

	if (current)
	{
		buf = currentBuf;
		page = *currentPage;
	}
	else
	{
		buf = ReadBuffer(index, prevBlkno);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
		page = BufferGetPage(buf);
	}

	if (RelationNeedsWAL(index))
	{
		xlogState = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(xlogState, buf, 0);
	}

	if (prevOffno <= PageGetMaxOffsetNumber(page))
	{
		iid = PageGetItemId(page, prevOffno);
		if (ItemIdIsUsed(iid))
		{
			tuple = (PgturbohybridBm25PostingsTuple) PageGetItem(page, iid);
			if (tuple->type == PGTURBOHYBRID_BM25_POSTINGS_TUPLE_TYPE)
			{
				tuple->nextBlkno = nextBlkno;
				tuple->nextOffno = nextOffno;
				modified = true;
			}
		}
	}

	if (xlogState != NULL)
	{
		if (modified)
			GenericXLogFinish(xlogState);
		else
			GenericXLogAbort(xlogState);
	}
	else if (modified)
		MarkBufferDirty(buf);

	if (current)
		*currentPage = BufferGetPage(buf);
	else
		UnlockReleaseBuffer(buf);
}

static void
PgturbohybridLinkImpactChunk(Relation index, Buffer currentBuf, Page *currentPage,
						BlockNumber prevBlkno, OffsetNumber prevOffno,
						BlockNumber nextBlkno, OffsetNumber nextOffno)
{
	Buffer		buf;
	Page		page;
	ItemId		iid;
	PgturbohybridBm25ImpactTuple tuple;
	GenericXLogState *xlogState = NULL;
	bool		modified = false;
	bool		current = BufferIsValid(currentBuf) &&
		BufferGetBlockNumber(currentBuf) == prevBlkno;

	if (current)
	{
		buf = currentBuf;
		page = *currentPage;
	}
	else
	{
		buf = ReadBuffer(index, prevBlkno);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
		page = BufferGetPage(buf);
	}

	if (RelationNeedsWAL(index))
	{
		xlogState = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(xlogState, buf, 0);
	}

	if (prevOffno <= PageGetMaxOffsetNumber(page))
	{
		iid = PageGetItemId(page, prevOffno);
		if (ItemIdIsUsed(iid))
		{
			tuple = (PgturbohybridBm25ImpactTuple) PageGetItem(page, iid);
			if (tuple->type == PGTURBOHYBRID_BM25_IMPACT_TUPLE_TYPE)
			{
				tuple->nextBlkno = nextBlkno;
				tuple->nextOffno = nextOffno;
				modified = true;
			}
		}
	}

	if (xlogState != NULL)
	{
		if (modified)
			GenericXLogFinish(xlogState);
		else
			GenericXLogAbort(xlogState);
	}
	else if (modified)
		MarkBufferDirty(buf);

	if (current)
		*currentPage = BufferGetPage(buf);
	else
		UnlockReleaseBuffer(buf);
}

static OffsetNumber
PgturbohybridWritePostings(PgturbohybridBm25Collector *collector, uint32 termId,
					  uint32 startIndex, uint32 endIndex,
					  Buffer *buf, Page *page,
					  BlockNumber *postingsBlkno, uint32 *postingsBytes,
					  uint32 *postingsChunkCount)
{
	uint16		maxPerChunk = PgturbohybridBm25MaxPostingsPerChunk();
	OffsetNumber firstOffno = InvalidOffsetNumber;
	BlockNumber firstBlkno = InvalidBlockNumber;
	BlockNumber prevBlkno = InvalidBlockNumber;
	OffsetNumber prevOffno = InvalidOffsetNumber;

	*postingsBytes = 0;
	*postingsChunkCount = 0;
	for (uint32 chunkStart = startIndex; chunkStart < endIndex;)
	{
		uint32		chunkEnd = Min(chunkStart + (uint32) maxPerChunk, endIndex);
		BlockNumber chunkBlkno;
		OffsetNumber chunkOffno;
		uint32		chunkBytes;

		chunkOffno = PgturbohybridWritePostingsChunk(collector, termId,
												*postingsChunkCount,
												chunkStart, chunkEnd,
												buf, page,
												&chunkBlkno, &chunkBytes);
		if (!OffsetNumberIsValid(firstOffno))
		{
			firstOffno = chunkOffno;
			firstBlkno = chunkBlkno;
		}
		if (BlockNumberIsValid(prevBlkno))
			PgturbohybridLinkPostingsChunk(collector->index, *buf, page,
									  prevBlkno, prevOffno,
									  chunkBlkno, chunkOffno);

		prevBlkno = chunkBlkno;
		prevOffno = chunkOffno;
		*postingsBytes += chunkBytes;
		(*postingsChunkCount)++;
		chunkStart = chunkEnd;
	}

	*postingsBlkno = firstBlkno;
	return firstOffno;
}

static OffsetNumber
PgturbohybridWritePostingsChunkData(PgturbohybridBm25Collector *collector, uint32 termId,
							   uint32 chunkNo,
							   const PgturbohybridBm25Posting *postings,
							   uint16 count, Buffer *buf, Page *page,
							   BlockNumber *postingsBlkno,
							   uint32 *postingsBytes)
{
	Size		size;
	Size		payloadBytes = 0;
	PgturbohybridBm25PostingsTuple tuple;
	OffsetNumber offno;
	BlockNumber insertBlkno = InvalidBlockNumber;
	char	   *ptr;
	uint32		prevNodeId;
	bool		useOffset16 = true;
	bool		useDelta16 = true;
	bool		hasTfNorm = PgturbohybridBm25PrecomputeTfNormEnabled(collector);
	bool		useOffset16Soa;

	prevNodeId = postings[0].nodeId;
	for (uint16 i = 0; i < count; i++)
	{
		if (!PgturbohybridBm25CanUseOffset16(postings[0].nodeId,
										postings[i].nodeId))
			useOffset16 = false;
		if (!PgturbohybridBm25CanUseDelta16(prevNodeId, postings[i].nodeId))
			useDelta16 = false;
		prevNodeId = postings[i].nodeId;
	}
	if (useOffset16 || useDelta16)
		payloadBytes = count * sizeof(uint16) * 2;
	else
	{
		prevNodeId = postings[0].nodeId;
		for (uint16 i = 0; i < count; i++)
		{
			uint32		delta = postings[i].nodeId - prevNodeId;

			payloadBytes += PgturbohybridBm25VarintLen(delta);
			payloadBytes += PgturbohybridBm25VarintLen(postings[i].tf);
			prevNodeId = postings[i].nodeId;
		}
	}
	if (hasTfNorm)
		payloadBytes += count * sizeof(uint16);
	size = PgturbohybridBm25PostingsTupleSize(payloadBytes);
	tuple = palloc0(size);
	tuple->type = PGTURBOHYBRID_BM25_POSTINGS_TUPLE_TYPE;
	tuple->count = count;
	tuple->termId = termId;
	tuple->chunkNo = chunkNo;
	tuple->firstNodeId = postings[0].nodeId;
	tuple->lastNodeId = postings[count - 1].nodeId;
	tuple->nextBlkno = InvalidBlockNumber;
	tuple->nextOffno = InvalidOffsetNumber;
	useOffset16Soa = useOffset16 && hasTfNorm;
	tuple->encoding = useOffset16 ?
		(useOffset16Soa ?
		 PGTURBOHYBRID_BM25_POSTINGS_ENCODING_OFFSET16_SOA :
		 PGTURBOHYBRID_BM25_POSTINGS_ENCODING_OFFSET16) :
		(useDelta16 ?
		 PGTURBOHYBRID_BM25_POSTINGS_ENCODING_DELTA16 :
		 PGTURBOHYBRID_BM25_POSTINGS_ENCODING_DELTA_VARINT);
	if (hasTfNorm)
		tuple->encoding |= PGTURBOHYBRID_BM25_POSTINGS_ENCODING_TFNORM_Q16;
	tuple->payloadBytes = (uint16) payloadBytes;
	ptr = tuple->payload;
	prevNodeId = tuple->firstNodeId;
	if (useOffset16Soa)
	{
		char	   *tfPtr = ptr + count * sizeof(uint16);

		for (uint16 i = 0; i < count; i++)
		{
			uint16		offset;

			if (!PgturbohybridBm25CanUseOffset16(tuple->firstNodeId,
											postings[i].nodeId))
				elog(ERROR, "pgturbohybrid BM25 postings are not offset16 encodable");
			offset = (uint16) (postings[i].nodeId - tuple->firstNodeId);
			tuple->maxTf = Max(tuple->maxTf, postings[i].tf);
			PgturbohybridBm25PutUint16(&ptr, offset);
			PgturbohybridBm25PutUint16(&tfPtr, postings[i].tf);
		}
		ptr = tfPtr;
	}
	else
	{
		for (uint16 i = 0; i < count; i++)
		{
			tuple->maxTf = Max(tuple->maxTf, postings[i].tf);
			if (useOffset16)
				PgturbohybridBm25EncodePostingOffset16(&ptr, tuple->firstNodeId,
												  postings[i].nodeId,
												  postings[i].tf);
			else if (useDelta16)
				PgturbohybridBm25EncodePostingDelta16(&ptr, &prevNodeId,
												 postings[i].nodeId,
												 postings[i].tf);
			else
				PgturbohybridBm25EncodePosting(&ptr, &prevNodeId, postings[i].nodeId,
										  postings[i].tf);
		}
	}
	if (hasTfNorm)
	{
		PgturbohybridOptions *opts = (PgturbohybridOptions *) collector->index->rd_options;
		double		k1 = opts != NULL ? opts->bm25K1 : 1.2;

		for (uint16 i = 0; i < count; i++)
		{
			uint16		tfNormQ16 = PgturbohybridBm25QuantizeTfNorm(collector,
															  postings[i].nodeId,
															  postings[i].tf);

			tuple->maxTfNormQ16 = Max(tuple->maxTfNormQ16, tfNormQ16);
			PgturbohybridBm25PutUint16(&ptr, tfNormQ16);
		}
		tuple->maxScoreFactor =
			(float4) (Max(k1 + 1.0, 1.0) *
					  ((double) tuple->maxTfNormQ16 / (double) PG_UINT16_MAX));
	}

	offno = PgturbohybridBm25AddItem(collector->index, MAIN_FORKNUM, buf, page,
								&collector->bm25PostingsStartBlkno,
								PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_POSTINGS,
								(Item) tuple, size,
								&collector->bm25PostingsPages,
								collector->walLoggedWrites,
								&insertBlkno);
	*postingsBlkno = insertBlkno;
	*postingsBytes = size;
	pfree(tuple);
	return offno;
}

static OffsetNumber
PgturbohybridWriteImpactHead(PgturbohybridBm25Collector *collector, uint32 termId,
						PgturbohybridBm25ImpactTupleEntry *entries,
						uint32 entryCount, uint32 df,
						Buffer *buf, Page *page,
						BlockNumber *impactBlkno, uint16 *impactCount)
{
	uint32		minDf;
	uint32		headK;
	uint32		maxPerTuple;
	uint32		totalCount;
	uint32		written = 0;
	OffsetNumber firstOffno = InvalidOffsetNumber;
	BlockNumber firstBlkno = InvalidBlockNumber;
	BlockNumber prevBlkno = InvalidBlockNumber;
	OffsetNumber prevOffno = InvalidOffsetNumber;
	double		corpusDocCount = Max((double) collector->docCount, 1.0);
	double		idf;

	*impactBlkno = InvalidBlockNumber;
	*impactCount = 0;
	if (!PgturbohybridBm25ImpactHeadEnabled(collector, &minDf, &headK) ||
		df < minDf || entryCount == 0)
		return InvalidOffsetNumber;

	idf = log(1.0 + (corpusDocCount - (double) df + 0.5) /
			  ((double) df + 0.5));
	for (uint32 i = 0; i < entryCount; i++)
		entries[i].score *= (float4) idf;

	qsort(entries, entryCount, sizeof(PgturbohybridBm25ImpactTupleEntry),
		  PgturbohybridBm25ImpactTupleEntryCompare);
	maxPerTuple = (BLCKSZ / 2 - offsetof(PgturbohybridBm25ImpactTupleData, entries)) /
		sizeof(PgturbohybridBm25ImpactTupleEntry);
	totalCount = Min(entryCount, headK);
	totalCount = Min(totalCount, (uint32) PG_UINT16_MAX);
	while (written < totalCount)
	{
		uint16		count = (uint16) Min(maxPerTuple, totalCount - written);
		Size		size = PgturbohybridBm25ImpactTupleSize(count);
		PgturbohybridBm25ImpactTuple tuple;
		OffsetNumber offno;
		BlockNumber insertBlkno = InvalidBlockNumber;

		tuple = palloc0(size);
		tuple->type = PGTURBOHYBRID_BM25_IMPACT_TUPLE_TYPE;
		tuple->count = count;
		tuple->termId = termId;
		tuple->nextBlkno = InvalidBlockNumber;
		tuple->nextOffno = InvalidOffsetNumber;
		memcpy(tuple->entries, entries + written,
			   sizeof(PgturbohybridBm25ImpactTupleEntry) * count);

		offno = PgturbohybridBm25AddItem(collector->index, MAIN_FORKNUM, buf, page,
									&collector->bm25ImpactStartBlkno,
									PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_IMPACT,
									(Item) tuple, size,
									&collector->bm25ImpactPages,
									collector->walLoggedWrites,
									&insertBlkno);
		pfree(tuple);

		if (!OffsetNumberIsValid(firstOffno))
		{
			firstOffno = offno;
			firstBlkno = insertBlkno;
		}
		if (BlockNumberIsValid(prevBlkno))
			PgturbohybridLinkImpactChunk(collector->index, *buf, page,
									prevBlkno, prevOffno,
									insertBlkno, offno);

		prevBlkno = insertBlkno;
		prevOffno = offno;
		written += count;
	}

	*impactBlkno = firstBlkno;
	*impactCount = (uint16) totalCount;
	return firstOffno;
}

static OffsetNumber
PgturbohybridWriteBlockMax(PgturbohybridBm25Collector *collector, uint32 termId,
					  uint32 startIndex, uint32 endIndex,
					  Buffer *buf, Page *page, BlockNumber *blockMaxBlkno)
{
	PgturbohybridBm25BlockMaxTupleData tuple;
	uint16		maxTf = 0;
	OffsetNumber offno;
	BlockNumber insertBlkno = InvalidBlockNumber;

	memset(&tuple, 0, sizeof(tuple));
	tuple.type = PGTURBOHYBRID_BM25_BLOCKMAX_TUPLE_TYPE;
	tuple.count = 1;
	tuple.termId = termId;
	tuple.firstNodeId = collector->terms[startIndex].nodeId;
	tuple.lastNodeId = collector->terms[endIndex - 1].nodeId;
	for (uint32 i = startIndex; i < endIndex; i++)
		maxTf = Max(maxTf, collector->terms[i].tf);
	tuple.maxTf = maxTf;
	tuple.maxScoreUpperBound = (float4) maxTf;

	offno = PgturbohybridBm25AddItem(collector->index, MAIN_FORKNUM, buf, page,
								&collector->bm25BlockMaxStartBlkno,
								PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_BLOCKMAX,
								(Item) &tuple, MAXALIGN(sizeof(tuple)),
								&collector->bm25BlockMaxPages,
								collector->walLoggedWrites,
								&insertBlkno);
	*blockMaxBlkno = insertBlkno;
	return offno;
}

static OffsetNumber
PgturbohybridWriteBlockMaxData(PgturbohybridBm25Collector *collector, uint32 termId,
						  uint32 firstNodeId, uint32 lastNodeId, uint16 maxTf,
						  Buffer *buf, Page *page, BlockNumber *blockMaxBlkno)
{
	PgturbohybridBm25BlockMaxTupleData tuple;
	OffsetNumber offno;
	BlockNumber insertBlkno = InvalidBlockNumber;

	memset(&tuple, 0, sizeof(tuple));
	tuple.type = PGTURBOHYBRID_BM25_BLOCKMAX_TUPLE_TYPE;
	tuple.count = 1;
	tuple.termId = termId;
	tuple.firstNodeId = firstNodeId;
	tuple.lastNodeId = lastNodeId;
	tuple.maxTf = maxTf;
	tuple.maxScoreUpperBound = (float4) maxTf;

	offno = PgturbohybridBm25AddItem(collector->index, MAIN_FORKNUM, buf, page,
								&collector->bm25BlockMaxStartBlkno,
								PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_BLOCKMAX,
								(Item) &tuple, MAXALIGN(sizeof(tuple)),
								&collector->bm25BlockMaxPages,
								collector->walLoggedWrites,
								&insertBlkno);
	*blockMaxBlkno = insertBlkno;
	return offno;
}

static void
PgturbohybridSpillCursorCloseCurrent(PgturbohybridBm25SpillCursor *cursor)
{
	if (cursor->termBytes != NULL)
	{
		pfree(cursor->termBytes);
		cursor->termBytes = NULL;
	}
	cursor->valid = false;
}

static bool
PgturbohybridSpillCursorRead(PgturbohybridBm25SpillCursor *cursor)
{
	PgturbohybridSpillCursorCloseCurrent(cursor);
	if (cursor->remaining == 0)
		return false;

	PgturbohybridBufFileReadExact(cursor->run->file, &cursor->termHash,
								  sizeof(cursor->termHash));
	PgturbohybridBufFileReadExact(cursor->run->file, &cursor->nodeId,
								  sizeof(cursor->nodeId));
	PgturbohybridBufFileReadExact(cursor->run->file, &cursor->tf,
								  sizeof(cursor->tf));
	PgturbohybridBufFileReadExact(cursor->run->file, &cursor->termLen,
								  sizeof(cursor->termLen));
	cursor->termBytes = palloc(cursor->termLen);
	PgturbohybridBufFileReadExact(cursor->run->file, cursor->termBytes,
								  cursor->termLen);
	cursor->remaining--;
	cursor->valid = true;
	return true;
}

static int
PgturbohybridSpillCursorCompare(const PgturbohybridBm25SpillCursor *a,
						   const PgturbohybridBm25SpillCursor *b)
{
	int			cmp;

	if (a->termHash != b->termHash)
		return a->termHash < b->termHash ? -1 : 1;
	if (a->termLen != b->termLen)
		return a->termLen < b->termLen ? -1 : 1;
	cmp = memcmp(a->termBytes, b->termBytes, a->termLen);
	if (cmp != 0)
		return cmp;
	return (a->nodeId > b->nodeId) - (a->nodeId < b->nodeId);
}

static int
PgturbohybridSpillHeapCompare(PgturbohybridBm25SpillHeap *heap, int left, int right)
{
	return PgturbohybridSpillCursorCompare(&heap->cursors[left],
									  &heap->cursors[right]);
}

static void
PgturbohybridSpillHeapSwap(PgturbohybridBm25SpillHeap *heap, uint32 left, uint32 right)
{
	int			tmp = heap->items[left];

	heap->items[left] = heap->items[right];
	heap->items[right] = tmp;
}

static void
PgturbohybridSpillHeapSiftUp(PgturbohybridBm25SpillHeap *heap, uint32 pos)
{
	while (pos > 0)
	{
		uint32		parent = (pos - 1) / 2;

		if (PgturbohybridSpillHeapCompare(heap, heap->items[pos],
									 heap->items[parent]) >= 0)
			break;
		PgturbohybridSpillHeapSwap(heap, pos, parent);
		pos = parent;
	}
}

static void
PgturbohybridSpillHeapSiftDown(PgturbohybridBm25SpillHeap *heap, uint32 pos)
{
	for (;;)
	{
		uint32		left = pos * 2 + 1;
		uint32		right = left + 1;
		uint32		smallest = pos;

		if (left < heap->count &&
			PgturbohybridSpillHeapCompare(heap, heap->items[left],
									 heap->items[smallest]) < 0)
			smallest = left;
		if (right < heap->count &&
			PgturbohybridSpillHeapCompare(heap, heap->items[right],
									 heap->items[smallest]) < 0)
			smallest = right;
		if (smallest == pos)
			break;
		PgturbohybridSpillHeapSwap(heap, pos, smallest);
		pos = smallest;
	}
}

static void
PgturbohybridSpillHeapPush(PgturbohybridBm25SpillHeap *heap, int cursor)
{
	heap->items[heap->count] = cursor;
	PgturbohybridSpillHeapSiftUp(heap, heap->count);
	heap->count++;
}

static int
PgturbohybridSpillHeapPeek(PgturbohybridBm25SpillHeap *heap)
{
	if (heap->count == 0)
		return -1;
	return heap->items[0];
}

static int
PgturbohybridSpillHeapPop(PgturbohybridBm25SpillHeap *heap)
{
	int			best;

	if (heap->count == 0)
		return -1;

	best = heap->items[0];
	heap->count--;
	if (heap->count > 0)
	{
		heap->items[0] = heap->items[heap->count];
		PgturbohybridSpillHeapSiftDown(heap, 0);
	}

	return best;
}

static bool
PgturbohybridSpillSameTerm(PgturbohybridBm25SpillCursor *cursor,
					  uint64 termHash, const char *termBytes, uint16 termLen)
{
	return cursor->termHash == termHash &&
		cursor->termLen == termLen &&
		memcmp(cursor->termBytes, termBytes, termLen) == 0;
}

static void
PgturbohybridWriteLexiconItem(PgturbohybridBm25Collector *collector,
						 Buffer *lexBuf, Page *lexPage,
						 BlockNumber *lexStart, uint32 termId,
						 uint64 termHash, const char *termBytes,
						 uint16 termLen, uint32 df, uint32 cf,
						 BlockNumber postingsBlkno,
						 OffsetNumber postingsOffno,
						 uint32 postingsChunkCount, uint32 postingsBytes,
						 BlockNumber blockMaxBlkno,
						 OffsetNumber blockMaxOffno,
						 BlockNumber impactBlkno,
						 OffsetNumber impactOffno,
						 uint16 impactCount)
{
	Size		lexSize = PgturbohybridBm25LexiconEntrySize(termLen);
	PgturbohybridBm25LexiconEntry lex = palloc0(lexSize);

	lex->type = PGTURBOHYBRID_BM25_LEXICON_TUPLE_TYPE;
	lex->termLen = termLen;
	lex->termHash = termHash;
	lex->termId = termId;
	lex->df = df;
	lex->cf = cf;
	lex->postingsBlkno = postingsBlkno;
	lex->postingsOffno = postingsOffno;
	lex->postingsChunkCount = postingsChunkCount;
	lex->postingsBytes = postingsBytes;
	lex->blockMaxBlkno = blockMaxBlkno;
	lex->blockMaxOffno = blockMaxOffno;
	lex->impactBlkno = impactBlkno;
	lex->impactOffno = impactOffno;
	lex->impactCount = impactCount;
	memcpy(lex->termBytes, termBytes, termLen);

	(void) PgturbohybridBm25AddItem(collector->index, MAIN_FORKNUM,
							   lexBuf, lexPage, lexStart,
							   PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_LEXICON,
							   (Item) lex, lexSize, NULL,
							   collector->walLoggedWrites, NULL);
	pfree(lex);
}

static void
PgturbohybridWriteLexiconAndPostingsFromRuns(PgturbohybridBm25Collector *collector)
{
	Buffer		lexBuf = InvalidBuffer;
	Page		lexPage = NULL;
	Buffer		postingsBuf = InvalidBuffer;
	Page		postingsPage = NULL;
	Buffer		blockMaxBuf = InvalidBuffer;
	Page		blockMaxPage = NULL;
	Buffer		impactBuf = InvalidBuffer;
	Page		impactPage = NULL;
	BlockNumber lexStart = InvalidBlockNumber;
	PgturbohybridBm25SpillCursor *cursors;
	PgturbohybridBm25SpillHeap heap;
	PgturbohybridBm25Posting *chunk;
	uint16		maxPerChunk = PgturbohybridBm25MaxPostingsPerChunk();
	uint32		termId = 0;

	cursors = palloc0(PgturbohybridCheckedArrayBytes(sizeof(PgturbohybridBm25SpillCursor),
													collector->spillRunCount,
													"pgturbohybrid BM25 spill cursors"));
	heap.cursors = cursors;
	heap.items = palloc(PgturbohybridCheckedArrayBytes(sizeof(int),
													   collector->spillRunCount,
													   "pgturbohybrid BM25 spill heap"));
	heap.count = 0;
	chunk = palloc(PgturbohybridCheckedArrayBytes(sizeof(PgturbohybridBm25Posting),
												  maxPerChunk,
												  "pgturbohybrid BM25 posting chunk"));

	for (uint32 i = 0; i < collector->spillRunCount; i++)
	{
		cursors[i].run = &collector->spillRuns[i];
		cursors[i].remaining = collector->spillRuns[i].tupleCount;
		if (BufFileSeek(collector->spillRuns[i].file, 0, 0L, SEEK_SET) != 0)
			elog(ERROR, "failed to rewind pgturbohybrid BM25 spill run");
		if (PgturbohybridSpillCursorRead(&cursors[i]))
			PgturbohybridSpillHeapPush(&heap, (int) i);
	}

	for (;;)
	{
		int			best = PgturbohybridSpillHeapPeek(&heap);
		uint64		termHash;
		uint16		termLen;
		char	   *termBytes;
		uint32		df = 0;
		uint32		cf = 0;
		uint32		prevNode = PG_UINT32_MAX;
		uint16		maxTf = 0;
		uint16		chunkCount = 0;
		uint32		postingsChunkCount = 0;
		uint32		postingsBytes = 0;
		BlockNumber firstPostingsBlkno = InvalidBlockNumber;
		OffsetNumber firstPostingsOffno = InvalidOffsetNumber;
		BlockNumber prevPostingsBlkno = InvalidBlockNumber;
		OffsetNumber prevPostingsOffno = InvalidOffsetNumber;
		BlockNumber blockMaxBlkno;
		OffsetNumber blockMaxOffno;
		BlockNumber impactBlkno;
		OffsetNumber impactOffno;
		uint16		impactCount;
		PgturbohybridBm25ImpactTupleEntry *impactEntries = NULL;
		uint32		impactEntryCount = 0;
		uint32		impactEntryCapacity = 0;
		uint32		firstNodeId = PG_UINT32_MAX;
		uint32		lastNodeId = 0;

		if (best < 0)
			break;

		termHash = cursors[best].termHash;
		termLen = cursors[best].termLen;
		termBytes = palloc(termLen);
		memcpy(termBytes, cursors[best].termBytes, termLen);

		while (best >= 0 &&
			   PgturbohybridSpillSameTerm(&cursors[best], termHash, termBytes,
									 termLen))
		{
			uint32		nodeId = cursors[best].nodeId;
			uint16		tf = cursors[best].tf;
			int			cursor;

			if (nodeId != prevNode)
			{
				df++;
				prevNode = nodeId;
			}
			cf += tf;
			maxTf = Max(maxTf, tf);
			if (impactEntryCount >= impactEntryCapacity)
			{
				impactEntryCapacity =
					PgturbohybridBm25BuildGrowCapacity32(impactEntryCapacity == 0 ? 64 :
														 impactEntryCapacity,
														 impactEntryCount + 1,
														 sizeof(PgturbohybridBm25ImpactTupleEntry));
				impactEntries = impactEntries == NULL ?
					palloc(PgturbohybridBm25BuildArrayAllocSize(sizeof(PgturbohybridBm25ImpactTupleEntry),
																impactEntryCapacity)) :
					repalloc(impactEntries,
							 PgturbohybridBm25BuildArrayAllocSize(sizeof(PgturbohybridBm25ImpactTupleEntry),
																  impactEntryCapacity));
			}
			impactEntries[impactEntryCount].nodeId = nodeId;
			impactEntries[impactEntryCount].tfNormQ16 =
				PgturbohybridBm25QuantizeTfNorm(collector, nodeId, tf);
			impactEntries[impactEntryCount].reserved = 0;
			impactEntries[impactEntryCount].score =
				(float4) PgturbohybridBm25BuildPostingScore(collector, nodeId, tf);
			impactEntryCount++;
			if (firstNodeId == PG_UINT32_MAX)
				firstNodeId = nodeId;
			lastNodeId = nodeId;

			chunk[chunkCount].nodeId = nodeId;
			chunk[chunkCount].tf = tf;
			chunk[chunkCount].reserved = 0;
			chunkCount++;
			if (chunkCount == maxPerChunk)
			{
				BlockNumber chunkBlkno;
				OffsetNumber chunkOffno;
				uint32		chunkBytes;

				chunkOffno = PgturbohybridWritePostingsChunkData(collector, termId,
															postingsChunkCount,
															chunk, chunkCount,
															&postingsBuf,
															&postingsPage,
															&chunkBlkno,
															&chunkBytes);
				if (!OffsetNumberIsValid(firstPostingsOffno))
				{
					firstPostingsOffno = chunkOffno;
					firstPostingsBlkno = chunkBlkno;
				}
				if (BlockNumberIsValid(prevPostingsBlkno))
					PgturbohybridLinkPostingsChunk(collector->index, postingsBuf,
											  &postingsPage, prevPostingsBlkno,
											  prevPostingsOffno, chunkBlkno,
											  chunkOffno);
				prevPostingsBlkno = chunkBlkno;
				prevPostingsOffno = chunkOffno;
				postingsBytes += chunkBytes;
				postingsChunkCount++;
				chunkCount = 0;
			}

			cursor = PgturbohybridSpillHeapPop(&heap);
			Assert(cursor == best);
			if (PgturbohybridSpillCursorRead(&cursors[cursor]))
				PgturbohybridSpillHeapPush(&heap, cursor);
			best = PgturbohybridSpillHeapPeek(&heap);
		}

		if (chunkCount > 0)
		{
			BlockNumber chunkBlkno;
			OffsetNumber chunkOffno;
			uint32		chunkBytes;

			chunkOffno = PgturbohybridWritePostingsChunkData(collector, termId,
														postingsChunkCount,
														chunk, chunkCount,
														&postingsBuf,
														&postingsPage,
														&chunkBlkno,
														&chunkBytes);
			if (!OffsetNumberIsValid(firstPostingsOffno))
			{
				firstPostingsOffno = chunkOffno;
				firstPostingsBlkno = chunkBlkno;
			}
			if (BlockNumberIsValid(prevPostingsBlkno))
				PgturbohybridLinkPostingsChunk(collector->index, postingsBuf,
										  &postingsPage, prevPostingsBlkno,
										  prevPostingsOffno, chunkBlkno,
										  chunkOffno);
			postingsBytes += chunkBytes;
			postingsChunkCount++;
		}

		blockMaxOffno = PgturbohybridWriteBlockMaxData(collector, termId,
												  firstNodeId, lastNodeId,
												  maxTf, &blockMaxBuf,
												  &blockMaxPage,
												  &blockMaxBlkno);
		impactOffno = PgturbohybridWriteImpactHead(collector, termId,
											  impactEntries, impactEntryCount,
											  df, &impactBuf, &impactPage,
											  &impactBlkno, &impactCount);
		PgturbohybridWriteLexiconItem(collector, &lexBuf, &lexPage, &lexStart,
								 termId, termHash, termBytes, termLen, df, cf,
								 firstPostingsBlkno, firstPostingsOffno,
								 postingsChunkCount, postingsBytes,
								 blockMaxBlkno, blockMaxOffno,
								 impactBlkno, impactOffno, impactCount);
		if (impactEntries != NULL)
			pfree(impactEntries);
		pfree(termBytes);
		termId++;
		CHECK_FOR_INTERRUPTS();
	}

	for (uint32 i = 0; i < collector->spillRunCount; i++)
		PgturbohybridSpillCursorCloseCurrent(&cursors[i]);
	pfree(heap.items);
	pfree(cursors);
	pfree(chunk);

	if (BufferIsValid(lexBuf))
		UnlockReleaseBuffer(lexBuf);
	if (BufferIsValid(postingsBuf))
		UnlockReleaseBuffer(postingsBuf);
	if (BufferIsValid(blockMaxBuf))
		UnlockReleaseBuffer(blockMaxBuf);
	if (BufferIsValid(impactBuf))
		UnlockReleaseBuffer(impactBuf);

	collector->bm25LexiconStartBlkno = lexStart;
	collector->uniqueTerms = termId;
}

static void
PgturbohybridWriteLexiconAndPostings(PgturbohybridBm25Collector *collector)
{
	Buffer		lexBuf = InvalidBuffer;
	Page		lexPage = NULL;
	Buffer		postingsBuf = InvalidBuffer;
	Page		postingsPage = NULL;
	Buffer		blockMaxBuf = InvalidBuffer;
	Page		blockMaxPage = NULL;
	Buffer		impactBuf = InvalidBuffer;
	Page		impactPage = NULL;
	BlockNumber lexStart = InvalidBlockNumber;
	uint32		termId = 0;

	if (collector->spillRunCount > 0)
	{
		PgturbohybridWriteLexiconAndPostingsFromRuns(collector);
		return;
	}

	for (uint32 i = 0; i < collector->termCount;)
	{
		PgturbohybridBm25TermTuple *first = &collector->terms[i];
		uint32		startIndex = i;
		uint32		df = 0;
		uint32		cf = 0;
		uint32		prevNode = PG_UINT32_MAX;
		BlockNumber postingsBlkno;
		OffsetNumber postingsOffno;
		uint32		postingsBytes;
		uint32		postingsChunkCount;
		BlockNumber blockMaxBlkno;
		OffsetNumber blockMaxOffno;
		BlockNumber impactBlkno;
		OffsetNumber impactOffno;
		uint16		impactCount;
		PgturbohybridBm25ImpactTupleEntry *impactEntries;
		uint32		impactEntryCount;

		while (i < collector->termCount &&
			   PgturbohybridTermEqualIgnoringNode(first, &collector->terms[i],
											 collector->termBytes))
		{
			if (collector->terms[i].nodeId != prevNode)
			{
				df++;
				prevNode = collector->terms[i].nodeId;
			}
			cf += collector->terms[i].tf;
			i++;
		}

		postingsOffno = PgturbohybridWritePostings(collector, termId, startIndex, i,
											  &postingsBuf, &postingsPage,
											  &postingsBlkno, &postingsBytes,
											  &postingsChunkCount);
		blockMaxOffno = PgturbohybridWriteBlockMax(collector, termId, startIndex, i,
											  &blockMaxBuf, &blockMaxPage,
											  &blockMaxBlkno);
		impactEntryCount = i - startIndex;
		impactEntries =
			palloc(PgturbohybridCheckedArrayBytes(sizeof(PgturbohybridBm25ImpactTupleEntry),
												  Max(impactEntryCount, 1),
												  "pgturbohybrid BM25 impact entries"));
		for (uint32 j = startIndex; j < i; j++)
		{
			PgturbohybridBm25TermTuple *term = &collector->terms[j];
			PgturbohybridBm25ImpactTupleEntry *impact =
				&impactEntries[j - startIndex];

			impact->nodeId = term->nodeId;
			impact->tfNormQ16 = PgturbohybridBm25QuantizeTfNorm(collector,
														   term->nodeId,
														   term->tf);
			impact->reserved = 0;
			impact->score = (float4) PgturbohybridBm25BuildPostingScore(collector,
																	term->nodeId,
																	term->tf);
		}
		impactOffno = PgturbohybridWriteImpactHead(collector, termId,
											  impactEntries, impactEntryCount,
											  df, &impactBuf, &impactPage,
											  &impactBlkno, &impactCount);
		pfree(impactEntries);

		PgturbohybridWriteLexiconItem(collector, &lexBuf, &lexPage, &lexStart,
								 termId, first->termHash,
								 collector->termBytes + first->termOffset,
								 first->termLen, df, cf, postingsBlkno,
								 postingsOffno, postingsChunkCount,
								 postingsBytes, blockMaxBlkno, blockMaxOffno,
								 impactBlkno, impactOffno, impactCount);
		termId++;
	}

	if (BufferIsValid(lexBuf))
		UnlockReleaseBuffer(lexBuf);
	if (BufferIsValid(postingsBuf))
		UnlockReleaseBuffer(postingsBuf);
	if (BufferIsValid(blockMaxBuf))
		UnlockReleaseBuffer(blockMaxBuf);
	if (BufferIsValid(impactBuf))
		UnlockReleaseBuffer(impactBuf);

	collector->bm25LexiconStartBlkno = lexStart;
	collector->uniqueTerms = termId;
}

static void
PgturbohybridWriteMeta(PgturbohybridBm25Collector *collector)
{
	Buffer		buf = InvalidBuffer;
	Page		page = NULL;
	BlockNumber start = InvalidBlockNumber;
	PgturbohybridBm25MetaTupleData tuple;
	PgturbohybridOptions *opts = (PgturbohybridOptions *) collector->index->rd_options;

	memset(&tuple, 0, sizeof(tuple));
	tuple.type = PGTURBOHYBRID_BM25_META_TUPLE_TYPE;
	if (opts == NULL || opts->bm25PrecomputeTfNorm)
		tuple.reserved2 |= PGTURBOHYBRID_BM25_META_FLAG_TFNORM_Q16;
	if (collector->bm25ImpactPages > 0)
		tuple.reserved2 |= PGTURBOHYBRID_BM25_META_FLAG_IMPACT_HEAD;
	tuple.bm25Version = PGTURBOHYBRID_BM25_VERSION;
	tuple.docCount = collector->docCount;
	tuple.totalDocLen = collector->totalDocLen;
	tuple.avgDocLen = collector->docCount == 0 ? 0.0f :
		(float4) ((double) collector->totalDocLen / (double) collector->docCount);
	tuple.k1 = opts != NULL ? opts->bm25K1 : 1.2f;
	tuple.b = opts != NULL ? opts->bm25B : 0.75f;
	tuple.termCount = collector->uniqueTerms;
	tuple.termTupleCount = collector->totalTermCount;
	tuple.maxDocLen = collector->maxDocLen;
	tuple.docStatsStartBlkno = collector->bm25DocStatsStartBlkno;
	tuple.lexiconStartBlkno = collector->bm25LexiconStartBlkno;
	tuple.postingsStartBlkno = collector->bm25PostingsStartBlkno;
	tuple.blockMaxStartBlkno = collector->bm25BlockMaxStartBlkno;
	tuple.impactStartBlkno = collector->bm25ImpactStartBlkno;
	tuple.deltaStartBlkno = InvalidBlockNumber;
	tuple.deltaTermDirectoryBlkno = InvalidBlockNumber;
	tuple.deltaGeneration = 0;
	tuple.deltaDocCount = 0;
	tuple.deltaTotalDocLen = 0;
	tuple.deltaTermCount = 0;
	tuple.postingsPages = collector->bm25PostingsPages;
	tuple.blockMaxPages = collector->bm25BlockMaxPages;
	tuple.impactPages = collector->bm25ImpactPages;
	tuple.deltaPages = 0;
	tuple.deltaTermPages = 0;
	tuple.lastCompactionGeneration = 0;
	tuple.compactionCount = 0;

	(void) PgturbohybridBm25AddItem(collector->index, MAIN_FORKNUM, &buf, &page,
							   &start, PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_META,
							   (Item) &tuple, MAXALIGN(sizeof(tuple)), NULL,
							   false, NULL);
	UnlockReleaseBuffer(buf);
	collector->bm25MetaBlkno = start;
	PgturbohybridBm25SetMetaBlock(collector->index, start);
}

static void
PgturbohybridBm25WriteBasePages(PgturbohybridBm25Collector *collector)
{
	collector->bm25DocStatsStartBlkno = InvalidBlockNumber;
	collector->bm25LexiconStartBlkno = InvalidBlockNumber;
	collector->bm25PostingsStartBlkno = InvalidBlockNumber;
	collector->bm25BlockMaxStartBlkno = InvalidBlockNumber;
	collector->bm25ImpactStartBlkno = InvalidBlockNumber;
	collector->bm25PostingsPages = 0;
	collector->bm25BlockMaxPages = 0;
	collector->bm25ImpactPages = 0;

	PgturbohybridWriteDocStats(collector);
	PgturbohybridWriteLexiconAndPostings(collector);
}

static void
PgturbohybridBm25WriteStorage(PgturbohybridBm25Collector *collector)
{
	collector->bm25MetaBlkno = InvalidBlockNumber;

	PgturbohybridBm25WriteBasePages(collector);
	PgturbohybridWriteMeta(collector);
}

static char *
PgturbohybridBm25DeltaTermBytes(PgturbohybridBm25DeltaTuple tuple)
{
	return ((char *) tuple) + offsetof(PgturbohybridBm25DeltaTupleData, terms) +
		sizeof(PgturbohybridBm25DeltaTerm) * tuple->termCount;
}

static void
PgturbohybridBm25ReadDocLens(Relation index, const PgturbohybridBm25MetaTupleData *meta,
						uint32 nodeCount, uint32 *docLens)
{
	BlockNumber blkno = meta->docStatsStartBlkno;

	memset(docLens, 0, sizeof(uint32) * nodeCount);
	while (BlockNumberIsValid(blkno))
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		OffsetNumber maxoff;
		BlockNumber nextblkno;

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = PgturbohybridGraphPageGetOpaque(page);
		if (!PgturbohybridBm25PageIsKind(page, PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_DOCSTATS))
		{
			UnlockReleaseBuffer(buf);
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("pgturbohybrid BM25 docstats page has unexpected page kind")));
		}

		nextblkno = opaque->nextblkno;
		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off = OffsetNumberNext(off))
		{
			ItemId		iid = PageGetItemId(page, off);
			PgturbohybridBm25DocStatsTuple tuple;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (PgturbohybridBm25DocStatsTuple) PageGetItem(page, iid);
			if (tuple->type != PGTURBOHYBRID_BM25_DOCSTATS_TUPLE_TYPE)
				continue;
			for (uint16 i = 0; i < tuple->count; i++)
			{
				uint32		nodeId = tuple->startNodeId + i;

				if (nodeId < nodeCount)
					docLens[nodeId] = tuple->docs[i].docLen;
			}
		}

		UnlockReleaseBuffer(buf);
		blkno = nextblkno;
	}
}

static void
PgturbohybridBm25CollectBasePostings(Relation index,
								const PgturbohybridBm25MetaTupleData *meta,
								PgturbohybridBm25Collector *collector,
								const PgturbohybridNodeState *nodeStates,
								const uint32 *docLens, uint32 nodeCount)
{
	BlockNumber blkno = meta->lexiconStartBlkno;

	while (BlockNumberIsValid(blkno))
	{
		Buffer		lexBuf;
		Page		lexPage;
		PgturbohybridGraphPageOpaque lexOpaque;
		OffsetNumber maxoff;
		BlockNumber nextblkno;

		lexBuf = ReadBuffer(index, blkno);
		LockBuffer(lexBuf, BUFFER_LOCK_SHARE);
		lexPage = BufferGetPage(lexBuf);
		lexOpaque = PgturbohybridGraphPageGetOpaque(lexPage);
		if (!PgturbohybridBm25PageIsKind(lexPage, PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_LEXICON))
		{
			UnlockReleaseBuffer(lexBuf);
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("pgturbohybrid BM25 lexicon page has unexpected page kind")));
		}

		nextblkno = lexOpaque->nextblkno;
		maxoff = PageGetMaxOffsetNumber(lexPage);
		for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off = OffsetNumberNext(off))
			{
				ItemId		lexIid = PageGetItemId(lexPage, off);
				PgturbohybridBm25LexiconEntry lex;
				BlockNumber postBlkno;
				OffsetNumber postOffno;
				uint32		chunkLimit;

				if (!ItemIdIsUsed(lexIid))
					continue;

				lex = (PgturbohybridBm25LexiconEntry) PageGetItem(lexPage, lexIid);
				if (lex->type != PGTURBOHYBRID_BM25_LEXICON_TUPLE_TYPE)
					continue;
				if (!BlockNumberIsValid(lex->postingsBlkno) ||
					!OffsetNumberIsValid(lex->postingsOffno))
					continue;

				postBlkno = lex->postingsBlkno;
				postOffno = lex->postingsOffno;
				chunkLimit = Max(lex->postingsChunkCount, 1);
				for (uint32 chunkNo = 0;
					 chunkNo < chunkLimit && BlockNumberIsValid(postBlkno) &&
					 OffsetNumberIsValid(postOffno);
					 chunkNo++)
				{
					Buffer		postBuf;
					Page		postPage;
					ItemId		postIid;
						PgturbohybridBm25PostingsTuple postings;
						PgturbohybridBm25Posting *decoded;
						BlockNumber nextBlkno;
						OffsetNumber nextOffno;

					postBuf = ReadBuffer(index, postBlkno);
					LockBuffer(postBuf, BUFFER_LOCK_SHARE);
					postPage = BufferGetPage(postBuf);
					if (!PgturbohybridBm25PageIsKind(postPage, PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_POSTINGS) ||
						postOffno > PageGetMaxOffsetNumber(postPage))
					{
						UnlockReleaseBuffer(postBuf);
						ereport(ERROR,
								(errcode(ERRCODE_DATA_CORRUPTED),
								 errmsg("pgturbohybrid BM25 postings pointer is invalid")));
					}

					postIid = PageGetItemId(postPage, postOffno);
					if (!ItemIdIsUsed(postIid))
					{
						UnlockReleaseBuffer(postBuf);
						break;
					}

					postings = (PgturbohybridBm25PostingsTuple) PageGetItem(postPage, postIid);
					if (postings->type != PGTURBOHYBRID_BM25_POSTINGS_TUPLE_TYPE ||
						postings->termId != lex->termId)
					{
						UnlockReleaseBuffer(postBuf);
						ereport(ERROR,
								(errcode(ERRCODE_DATA_CORRUPTED),
								 errmsg("pgturbohybrid BM25 postings tuple is invalid")));
					}
						nextBlkno = postings->nextBlkno;
						nextOffno = postings->nextOffno;
						decoded =
							palloc(PgturbohybridCheckedArrayBytes(sizeof(PgturbohybridBm25Posting),
																  postings->count,
																  "pgturbohybrid BM25 decoded postings"));
						if (!PgturbohybridBm25DecodePostingsTuple(postings,
															ItemIdGetLength(postIid),
															decoded))
						{
							pfree(decoded);
							UnlockReleaseBuffer(postBuf);
							ereport(ERROR,
									(errcode(ERRCODE_DATA_CORRUPTED),
									 errmsg("pgturbohybrid BM25 postings payload is invalid")));
						}

						for (uint16 i = 0; i < postings->count; i++)
						{
							uint32		nodeId = decoded[i].nodeId;

							if (nodeId >= nodeCount || !nodeStates[nodeId].live ||
								docLens[nodeId] == 0)
								continue;

							PgturbohybridAppendBuildTerm(collector, nodeId, lex->termBytes,
													lex->termLen, decoded[i].tf);
						}

						pfree(decoded);
						UnlockReleaseBuffer(postBuf);
					postBlkno = nextBlkno;
					postOffno = nextOffno;
				}
			}

		UnlockReleaseBuffer(lexBuf);
		blkno = nextblkno;
		CHECK_FOR_INTERRUPTS();
	}
}

static void
PgturbohybridBm25CollectDelta(Relation index,
						 const PgturbohybridBm25MetaTupleData *meta,
						 PgturbohybridBm25Collector *collector,
						 const PgturbohybridNodeState *nodeStates,
						 bool *docSeen, uint32 nodeCount)
{
	BlockNumber blkno = meta->deltaStartBlkno;

	while (BlockNumberIsValid(blkno))
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		OffsetNumber maxoff;
		BlockNumber nextblkno;

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = PgturbohybridGraphPageGetOpaque(page);
		if (!PgturbohybridBm25PageIsKind(page, PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_DELTA))
		{
			UnlockReleaseBuffer(buf);
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("pgturbohybrid BM25 delta page has unexpected page kind")));
		}

		nextblkno = opaque->nextblkno;
		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off = OffsetNumberNext(off))
		{
			ItemId		iid = PageGetItemId(page, off);
			PgturbohybridBm25DeltaTuple tuple;
			char	   *termBytes;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (PgturbohybridBm25DeltaTuple) PageGetItem(page, iid);
			if (tuple->type != PGTURBOHYBRID_BM25_DELTA_TUPLE_TYPE ||
				tuple->nodeId >= nodeCount ||
				!nodeStates[tuple->nodeId].live)
				continue;

			if (!docSeen[tuple->nodeId])
			{
				PgturbohybridAppendBuildDoc(collector, tuple->nodeId,
									   &tuple->heaptid, tuple->docLen);
				docSeen[tuple->nodeId] = true;
			}

			termBytes = PgturbohybridBm25DeltaTermBytes(tuple);
			for (uint16 i = 0; i < tuple->termCount; i++)
			{
				PgturbohybridBm25DeltaTerm *term = &tuple->terms[i];

				if (term->termOffset + term->termLen > tuple->termBytesLen)
					ereport(ERROR,
							(errcode(ERRCODE_DATA_CORRUPTED),
							 errmsg("pgturbohybrid BM25 delta term offset is invalid")));

				PgturbohybridAppendBuildTerm(collector, tuple->nodeId,
										termBytes + term->termOffset,
										term->termLen, term->tf);
			}
		}

		UnlockReleaseBuffer(buf);
		blkno = nextblkno;
		CHECK_FOR_INTERRUPTS();
	}
}

static void
PgturbohybridBm25UpdateCompactedMeta(Relation index,
								PgturbohybridBm25Collector *collector,
								const PgturbohybridBm25MetaTupleData *oldMeta)
{
	Buffer		metaBuf;
	Page		metaPage;
	PgturbohybridBm25MetaTuple metaTuple;
	GenericXLogState *xlogState;
	PgturbohybridOptions *opts = (PgturbohybridOptions *) index->rd_options;

	if (!PgturbohybridBm25ReadMetaForUpdate(index, &metaBuf, &metaPage,
									   &metaTuple, &xlogState))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("pgturbohybrid BM25 metadata is missing")));

	metaTuple->docCount = collector->docCount;
	if (opts == NULL || opts->bm25PrecomputeTfNorm)
		metaTuple->reserved2 |= PGTURBOHYBRID_BM25_META_FLAG_TFNORM_Q16;
	else
		metaTuple->reserved2 &= ~PGTURBOHYBRID_BM25_META_FLAG_TFNORM_Q16;
	if (collector->bm25ImpactPages > 0)
		metaTuple->reserved2 |= PGTURBOHYBRID_BM25_META_FLAG_IMPACT_HEAD;
	else
		metaTuple->reserved2 &= ~PGTURBOHYBRID_BM25_META_FLAG_IMPACT_HEAD;
	metaTuple->totalDocLen = collector->totalDocLen;
	metaTuple->avgDocLen = collector->docCount == 0 ? 0.0f :
		(float4) ((double) collector->totalDocLen / (double) collector->docCount);
	metaTuple->termCount = collector->uniqueTerms;
	metaTuple->termTupleCount = collector->totalTermCount;
	metaTuple->maxDocLen = collector->maxDocLen;
	metaTuple->docStatsStartBlkno = collector->bm25DocStatsStartBlkno;
	metaTuple->lexiconStartBlkno = collector->bm25LexiconStartBlkno;
	metaTuple->postingsStartBlkno = collector->bm25PostingsStartBlkno;
	metaTuple->blockMaxStartBlkno = collector->bm25BlockMaxStartBlkno;
	metaTuple->impactStartBlkno = collector->bm25ImpactStartBlkno;
	metaTuple->deltaStartBlkno = InvalidBlockNumber;
	metaTuple->deltaTermDirectoryBlkno = InvalidBlockNumber;
	metaTuple->deltaGeneration = oldMeta->deltaGeneration + 1;
	metaTuple->deltaDocCount = 0;
	metaTuple->deltaTotalDocLen = 0;
	metaTuple->deltaTermCount = 0;
	metaTuple->postingsPages = collector->bm25PostingsPages;
	metaTuple->blockMaxPages = collector->bm25BlockMaxPages;
	metaTuple->impactPages = collector->bm25ImpactPages;
	metaTuple->deltaPages = 0;
	metaTuple->deltaTermPages = 0;
	metaTuple->lastCompactionGeneration = metaTuple->deltaGeneration;
	metaTuple->compactionCount = oldMeta->compactionCount + 1;

	if (xlogState != NULL)
		GenericXLogFinish(xlogState);
	else
		MarkBufferDirty(metaBuf);
	UnlockReleaseBuffer(metaBuf);
}

bool
PgturbohybridBm25MaybeCompact(Relation index)
{
	MemoryContext ctx;
	MemoryContext oldCtx;
	PgturbohybridBm25MetaTupleData oldMeta;
	PgturbohybridBm25Collector collector;
	PgturbohybridGraphMetaPageData graphMeta;
	PgturbohybridNodeState *nodeStates;
	uint32		nodeCount;
	uint32	   *docLens;
	bool	   *docSeen;
	PgturbohybridOptions *opts = (PgturbohybridOptions *) index->rd_options;
	int			threshold = opts != NULL ? opts->bm25DeltaCompactionThreshold : 25;
	uint32		uniqueTerms;
	bool		compacted = false;

	if (!PgturbohybridBm25ReadMeta(index, &oldMeta, NULL) ||
		oldMeta.deltaDocCount == 0 ||
		!BlockNumberIsValid(oldMeta.deltaStartBlkno))
		return false;

	if ((uint64) oldMeta.deltaDocCount * 100 <
		(uint64) Max(oldMeta.docCount, 1) * (uint64) threshold)
		return false;

	/*
	 * Delta compaction rewrites BM25 base pages and clears the delta chain.
	 * Serialize it with inserts via the same update lock aminsert uses.
	 */
	LockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, ExclusiveLock);
	PG_TRY();
	{
		ctx = AllocSetContextCreate(CurrentMemoryContext,
									"pgturbohybrid BM25 delta compaction",
									ALLOCSET_DEFAULT_SIZES);
		oldCtx = MemoryContextSwitchTo(ctx);

		nodeStates = PgturbohybridReadNodeStates(index, &graphMeta, &nodeCount);
		if (nodeCount == 0)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("pgturbohybrid native graph metadata is missing during BM25 compaction")));

		docLens = palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32),
														 nodeCount,
														 "pgturbohybrid BM25 compacted doc lengths"));
		docSeen = palloc0(PgturbohybridCheckedArrayBytes(sizeof(bool),
														 nodeCount,
														 "pgturbohybrid BM25 compacted doc visibility"));
		PgturbohybridBm25ReadDocLens(index, &oldMeta, nodeCount, docLens);

		memset(&collector, 0, sizeof(collector));
		collector.index = index;
		collector.softBudget = PgturbohybridBm25MaintenanceWorkMemBytes();
		collector.allowSpill = true;
		collector.walLoggedWrites = RelationNeedsWAL(index);
		collector.tidNodeCount = nodeCount;

		for (uint32 nodeId = 0; nodeId < nodeCount; nodeId++)
		{
			if (nodeStates[nodeId].live && docLens[nodeId] > 0)
			{
				PgturbohybridAppendBuildDoc(&collector, nodeId,
									   &nodeStates[nodeId].tid, docLens[nodeId]);
				docSeen[nodeId] = true;
			}
		}

		PgturbohybridBm25CollectBasePostings(index, &oldMeta, &collector,
										nodeStates, docLens, nodeCount);
		PgturbohybridBm25CollectDelta(index, &oldMeta, &collector,
								 nodeStates, docSeen, nodeCount);

		if (collector.spillRunCount > 0)
		{
			PgturbohybridSpillTermRun(&collector);
			uniqueTerms = 0;
		}
		else
			uniqueTerms = PgturbohybridReduceUniqueTerms(&collector);
		collector.uniqueTerms = uniqueTerms;
		PgturbohybridBm25WriteBasePages(&collector);
		PgturbohybridBm25UpdateCompactedMeta(index, &collector, &oldMeta);
		PgturbohybridBm25ResetDeltaAppendCursor(index);
		PgturbohybridBm25InvalidateCache(index);
		PgturbohybridGraphInvalidateCaches(index);
		PgturbohybridCloseSpillRuns(&collector);

		MemoryContextSwitchTo(oldCtx);
		MemoryContextDelete(ctx);
		compacted = true;
	}
	PG_CATCH();
	{
		UnlockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, ExclusiveLock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	UnlockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, ExclusiveLock);

	return compacted;
}

void
PgturbohybridBm25BuildEmpty(Relation index)
{
	MemoryContext ctx;
	MemoryContext oldCtx;
	PgturbohybridBm25Collector collector;

	if (!PgturbohybridIndexHasLexical(index))
		return;

	ctx = AllocSetContextCreate(CurrentMemoryContext,
								"pgturbohybrid BM25 empty build",
								ALLOCSET_DEFAULT_SIZES);
	oldCtx = MemoryContextSwitchTo(ctx);

	memset(&collector, 0, sizeof(collector));
	collector.index = index;
	collector.softBudget = PgturbohybridBm25MaintenanceWorkMemBytes();

	PgturbohybridBm25WriteStorage(&collector);

	MemoryContextSwitchTo(oldCtx);
	MemoryContextDelete(ctx);
}

void
PgturbohybridBm25BuildCollect(Relation heap, Relation index, IndexInfo *indexInfo)
{
	MemoryContext ctx;
	MemoryContext oldCtx;
	PgturbohybridBm25Collector collector;
	PgturbohybridGraphMetaPageData graphMeta;
	uint32		uniqueTerms;

	if (!PgturbohybridIndexHasLexical(index) &&
		(indexInfo == NULL ||
		 indexInfo->ii_NumIndexKeyAttrs <= PGTURBOHYBRID_LEXICAL_KEY_INDEX))
		return;

	if (heap == NULL)
		return;

	ctx = AllocSetContextCreate(CurrentMemoryContext,
								"pgturbohybrid BM25 build collector",
								ALLOCSET_DEFAULT_SIZES);
	oldCtx = MemoryContextSwitchTo(ctx);

	memset(&collector, 0, sizeof(collector));
	collector.index = index;
	collector.softBudget = PgturbohybridBm25MaintenanceWorkMemBytes();
	collector.allowSpill = true;

	if (!PgturbohybridGraphReadMeta(index, &graphMeta))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("pgturbohybrid BM25 collection requires native pgturbohybrid graph storage")));

	if (graphMeta.tqNodeCount > 0)
	{
		collector.tidNodes = PgturbohybridReadNodeMap(index, &collector.tidNodeCount);
		if (collector.tidNodeCount == 0)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("pgturbohybrid native graph storage is missing during BM25 collection")));
	}

	PgturbohybridCheckBudget(&collector);
	(void) table_index_build_scan(heap, index, indexInfo,
								  true, true, PgturbohybridBm25BuildCallback,
								  &collector, NULL);
	if (collector.spillRunCount > 0)
	{
		PgturbohybridSpillTermRun(&collector);
		uniqueTerms = 0;
	}
	else
		uniqueTerms = PgturbohybridReduceUniqueTerms(&collector);
	collector.uniqueTerms = uniqueTerms;
	PgturbohybridBm25WriteStorage(&collector);
	PgturbohybridCloseSpillRuns(&collector);

	MemoryContextSwitchTo(oldCtx);
	MemoryContextDelete(ctx);
}

static bool
PgturbohybridBm25ReadMetaForUpdate(Relation index, Buffer *outBuf, Page *outPage,
							  PgturbohybridBm25MetaTuple *outTuple,
							  GenericXLogState **outXlogState)
{
	PgturbohybridGraphMetaPageData graphMeta;
	BlockNumber metaBlkno;
	BlockNumber nblocks;
	OffsetNumber maxoff;

	*outBuf = InvalidBuffer;
	*outPage = NULL;
	*outTuple = NULL;
	*outXlogState = NULL;

	if (!PgturbohybridGraphReadMeta(index, &graphMeta) ||
		!BlockNumberIsValid(graphMeta.tqBm25MetaStartBlkno))
		return false;

	metaBlkno = graphMeta.tqBm25MetaStartBlkno;
	nblocks = RelationGetNumberOfBlocks(index);
	if (metaBlkno >= nblocks)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("pgturbohybrid BM25 metadata pointer is invalid"),
				 errdetail("Metapage points to block %u, but the index has only %u blocks.",
						   metaBlkno, nblocks)));

	*outBuf = ReadBuffer(index, metaBlkno);
	LockBuffer(*outBuf, BUFFER_LOCK_EXCLUSIVE);
	if (RelationNeedsWAL(index))
	{
		*outXlogState = GenericXLogStart(index);
		*outPage = GenericXLogRegisterBuffer(*outXlogState, *outBuf, 0);
	}
	else
		*outPage = BufferGetPage(*outBuf);

	if (!PgturbohybridBm25PageIsKind(*outPage, PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_META))
	{
		if (*outXlogState != NULL)
			GenericXLogAbort(*outXlogState);
		UnlockReleaseBuffer(*outBuf);
		*outBuf = InvalidBuffer;
		*outPage = NULL;
		*outXlogState = NULL;
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("pgturbohybrid BM25 metadata pointer is invalid"),
				 errdetail("Metapage points to block %u, which is not a BM25 metadata page.",
						   metaBlkno)));
	}

	maxoff = PageGetMaxOffsetNumber(*outPage);
	for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off = OffsetNumberNext(off))
	{
		ItemId		iid = PageGetItemId(*outPage, off);
		PgturbohybridBm25MetaTuple tuple;

		if (!ItemIdIsUsed(iid))
			continue;

		tuple = (PgturbohybridBm25MetaTuple) PageGetItem(*outPage, iid);
		if (tuple->type == PGTURBOHYBRID_BM25_META_TUPLE_TYPE)
		{
			*outTuple = tuple;
			return true;
		}
	}

	if (*outXlogState != NULL)
		GenericXLogAbort(*outXlogState);
	UnlockReleaseBuffer(*outBuf);
	*outBuf = InvalidBuffer;
	*outPage = NULL;
	*outXlogState = NULL;
	ereport(ERROR,
			(errcode(ERRCODE_DATA_CORRUPTED),
			 errmsg("pgturbohybrid BM25 metadata tuple is missing"),
			 errdetail("Metapage points to BM25 metadata block %u, but no metadata tuple was found.",
					   metaBlkno)));
}

static uint32
PgturbohybridBm25CountChainPagesAndTail(Relation index, BlockNumber startBlkno,
								   uint16 pageKind, BlockNumber *tailBlkno)
{
	uint32		count = 0;
	BlockNumber blkno = startBlkno;

	*tailBlkno = InvalidBlockNumber;
	while (BlockNumberIsValid(blkno))
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		BlockNumber nextblkno;

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = PgturbohybridGraphPageGetOpaque(page);
		if (!PgturbohybridBm25PageIsKind(page, pageKind))
		{
			UnlockReleaseBuffer(buf);
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("pgturbohybrid BM25 page chain has unexpected page kind"),
					 errdetail("Block %u in chain starting at block %u is not page kind %u.",
							   blkno, startBlkno, pageKind)));
		}

		nextblkno = opaque->nextblkno;
		UnlockReleaseBuffer(buf);
		count++;
		*tailBlkno = blkno;
		blkno = nextblkno;
	}

	return count;
}

static void
PgturbohybridBm25InitDeltaDirectory(PgturbohybridBm25DeltaDirectoryTuple directory)
{
	memset(directory, 0, sizeof(PgturbohybridBm25DeltaDirectoryTupleData));
	directory->type = PGTURBOHYBRID_BM25_DELTA_DIRECTORY_TUPLE_TYPE;
	directory->bucketCount = PGTURBOHYBRID_BM25_DELTA_TERM_BUCKETS;
	for (uint16 i = 0; i < PGTURBOHYBRID_BM25_DELTA_TERM_BUCKETS; i++)
	{
		directory->buckets[i].startBlkno = InvalidBlockNumber;
		directory->buckets[i].tailBlkno = InvalidBlockNumber;
	}
}

static bool
PgturbohybridBm25ReadDeltaDirectory(Relation index, BlockNumber blkno,
							   PgturbohybridBm25DeltaDirectoryTuple directory)
{
	Buffer		buf;
	Page		page;
	OffsetNumber maxoff;

	if (!BlockNumberIsValid(blkno))
		return false;

	buf = ReadBuffer(index, blkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	if (!PgturbohybridBm25PageIsKind(page, PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_DELTA_TERM))
	{
		UnlockReleaseBuffer(buf);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("pgturbohybrid BM25 delta term directory has unexpected page kind")));
	}

	maxoff = PageGetMaxOffsetNumber(page);
	for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off = OffsetNumberNext(off))
	{
		ItemId		iid = PageGetItemId(page, off);
		PgturbohybridBm25DeltaDirectoryTuple tuple;

		if (!ItemIdIsUsed(iid))
			continue;
		tuple = (PgturbohybridBm25DeltaDirectoryTuple) PageGetItem(page, iid);
		if (tuple->type != PGTURBOHYBRID_BM25_DELTA_DIRECTORY_TUPLE_TYPE)
			continue;
		if (tuple->bucketCount != PGTURBOHYBRID_BM25_DELTA_TERM_BUCKETS)
		{
			UnlockReleaseBuffer(buf);
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("pgturbohybrid BM25 delta term directory has invalid bucket count")));
		}
		memcpy(directory, tuple, sizeof(PgturbohybridBm25DeltaDirectoryTupleData));
		UnlockReleaseBuffer(buf);
		return true;
	}

	UnlockReleaseBuffer(buf);
	return false;
}

static uint32
PgturbohybridBm25DeltaTermBucketNo(uint64 termHash)
{
	return (uint32) (termHash & (PGTURBOHYBRID_BM25_DELTA_TERM_BUCKETS - 1));
}

static uint32
PgturbohybridBm25DeltaDirectoryTermPages(PgturbohybridBm25DeltaDirectoryTuple directory)
{
	uint32		pages = 0;

	for (uint16 i = 0; i < directory->bucketCount; i++)
		pages += directory->buckets[i].pages;
	return pages;
}

static BlockNumber
PgturbohybridBm25WriteDeltaTermDirectory(Relation index,
									PgturbohybridBm25DeltaDirectoryTuple directory)
{
	BlockNumber start = InvalidBlockNumber;
	BlockNumber insertBlkno = InvalidBlockNumber;

	(void) PgturbohybridGraphAppendTuple(index, MAIN_FORKNUM, &start,
							  PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_DELTA_TERM,
							  (Item) directory,
							  MAXALIGN(sizeof(PgturbohybridBm25DeltaDirectoryTupleData)),
							  PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT,
							  &insertBlkno);
	return start;
}

static void
PgturbohybridBm25AppendDeltaTermSegments(Relation index,
									PgturbohybridBm25DeltaTuple delta,
									PgturbohybridBm25DeltaDirectoryTuple directory,
									Size maxItemSize)
{
	char	   *termBytes = PgturbohybridBm25DeltaTermBytes(delta);

	for (uint16 i = 0; i < delta->termCount; i++)
	{
		PgturbohybridBm25DeltaTerm *term = &delta->terms[i];
		PgturbohybridBm25DeltaTermTuple termTuple;
		Size		termTupleSize;
		uint32		bucketNo;
		PgturbohybridBm25DeltaTermBucket *bucket;
		BlockNumber appendStart;
		BlockNumber oldTail;
		BlockNumber insertBlkno = InvalidBlockNumber;

		if (term->termOffset + term->termLen > delta->termBytesLen)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("pgturbohybrid BM25 delta term offset is invalid")));

		termTupleSize = PgturbohybridBm25DeltaTermTupleSize(1, term->termLen);
		if (termTupleSize > maxItemSize)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("pgturbohybrid BM25 delta term tuple exceeds page size")));

		termTuple = palloc0(termTupleSize);
		termTuple->type = PGTURBOHYBRID_BM25_DELTA_TERM_TUPLE_TYPE;
		termTuple->postingCount = 1;
		termTuple->termHash = term->termHash;
		termTuple->termBytesLen = term->termLen;
		termTuple->termLen = term->termLen;
		termTuple->postings[0].nodeId = delta->nodeId;
		termTuple->postings[0].heaptid = delta->heaptid;
		termTuple->postings[0].docLen = delta->docLen;
		termTuple->postings[0].tf = term->tf;
		memcpy(PgturbohybridBm25DeltaTermTupleBytes(termTuple),
			   termBytes + term->termOffset, term->termLen);

		bucketNo = PgturbohybridBm25DeltaTermBucketNo(term->termHash);
		bucket = &directory->buckets[bucketNo];
		oldTail = bucket->tailBlkno;
		appendStart = BlockNumberIsValid(bucket->tailBlkno) ?
			bucket->tailBlkno : bucket->startBlkno;
		(void) PgturbohybridGraphAppendTuple(index, MAIN_FORKNUM, &appendStart,
								  PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_DELTA_TERM,
								  (Item) termTuple, termTupleSize,
								  PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT,
								  &insertBlkno);
		if (!BlockNumberIsValid(bucket->startBlkno))
		{
			bucket->startBlkno = appendStart;
			bucket->pages = 1;
		}
		else if (!BlockNumberIsValid(oldTail) || insertBlkno != oldTail)
			bucket->pages++;
		bucket->tailBlkno = insertBlkno;
		pfree(termTuple);
	}
}

void
PgturbohybridBm25AppendDelta(Relation index, uint32 nodeId,
						ItemPointer heapTid, Datum tsvectorDatum)
{
	MemoryContext ctx;
	MemoryContext oldCtx;
	PgturbohybridBm25Collector collector;
	TSVector	vector;
	bool		mustFree;
	Size		maxItemSize;
	uint32		docLen;
	uint32		startTerm;
	BlockNumber deltaStart;
	BlockNumber deltaTail;
	BlockNumber deltaTermDirectoryBlkno;
	BlockNumber appendStart;
	BlockNumber deltaBlkno = InvalidBlockNumber;
	uint32		deltaPages;
	uint32		deltaTermPages;
	bool		deltaCursorHit;
	PgturbohybridBm25DeltaDirectoryTupleData deltaDirectory;
	Buffer		metaBuf;
	Page		metaPage;
	PgturbohybridBm25MetaTuple metaTuple;
	GenericXLogState *xlogState;

	(void) metaPage;

	ctx = AllocSetContextCreate(CurrentMemoryContext,
								"pgturbohybrid BM25 delta append",
								ALLOCSET_DEFAULT_SIZES);
	oldCtx = MemoryContextSwitchTo(ctx);

	memset(&collector, 0, sizeof(collector));
	collector.index = index;
	collector.softBudget = PgturbohybridBm25MaintenanceWorkMemBytes();
	vector = PgturbohybridDetoastTSVector(tsvectorDatum, &mustFree);
	PgturbohybridValidateTSVector(vector);
	PgturbohybridCollectVectorTerms(&collector, nodeId, vector);
	if (collector.termCount > PG_UINT16_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pgturbohybrid BM25 delta document has too many distinct terms"),
				 errdetail("Delta storage supports at most %u distinct terms per inserted document, but this document has %u.",
						   PG_UINT16_MAX, collector.termCount),
				 errhint("Rebuild the index after loading very large text documents, or reduce the tsvector vocabulary for a single row.")));

	maxItemSize = BLCKSZ - SizeOfPageHeaderData -
		MAXALIGN(sizeof(PgturbohybridGraphPageOpaqueData));
	docLen = PgturbohybridDocLen(vector);

	if (!PgturbohybridBm25ReadMetaForUpdate(index, &metaBuf, &metaPage,
									   &metaTuple, &xlogState))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("pgturbohybrid BM25 metadata is missing")));

	deltaStart = metaTuple->deltaStartBlkno;
	deltaTermDirectoryBlkno = metaTuple->deltaTermDirectoryBlkno;
	PgturbohybridBm25InitDeltaDirectory(&deltaDirectory);
	if (BlockNumberIsValid(deltaTermDirectoryBlkno))
		(void) PgturbohybridBm25ReadDeltaDirectory(index, deltaTermDirectoryBlkno,
											  &deltaDirectory);
	deltaCursorHit =
		pgturbohybrid_bm25_delta_cursor_index == RelationGetRelid(index) &&
		pgturbohybrid_bm25_delta_cursor_relfilenumber == PgturbohybridGraphRelFileNumber(index) &&
		pgturbohybrid_bm25_delta_cursor_generation == metaTuple->deltaGeneration &&
		pgturbohybrid_bm25_delta_cursor_start == metaTuple->deltaStartBlkno;
	if (deltaCursorHit)
	{
		deltaTail = pgturbohybrid_bm25_delta_cursor_tail;
		deltaPages = pgturbohybrid_bm25_delta_cursor_pages;
	}
	else
		deltaPages = PgturbohybridBm25CountChainPagesAndTail(index, deltaStart,
														 PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_DELTA,
														 &deltaTail);
	if (xlogState != NULL)
		GenericXLogAbort(xlogState);
	UnlockReleaseBuffer(metaBuf);

	for (startTerm = 0; startTerm < collector.termCount;)
	{
		PgturbohybridBm25DeltaTuple delta;
		Size		deltaSize;
		uint32		chunkTermCount =
			PgturbohybridBm25DeltaChunkTermCount(&collector, startTerm,
												 maxItemSize);

		delta = PgturbohybridBm25BuildDeltaChunk(&collector, nodeId, heapTid,
												 docLen, startTerm,
												 chunkTermCount, &deltaSize);
		appendStart = BlockNumberIsValid(deltaTail) ? deltaTail : deltaStart;
		(void) PgturbohybridGraphAppendTuple(index, MAIN_FORKNUM, &appendStart,
								  PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_DELTA,
								  (Item) delta, deltaSize,
								  PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT,
								  &deltaBlkno);
		if (!BlockNumberIsValid(deltaStart))
		{
			deltaStart = appendStart;
			deltaPages = 1;
		}
		else if (!BlockNumberIsValid(deltaTail))
			deltaPages = PgturbohybridBm25CountChainPagesAndTail(index, deltaStart,
																 PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_DELTA,
																 &deltaTail);
		else if (deltaBlkno != deltaTail)
			deltaPages++;
		deltaTail = deltaBlkno;

		PgturbohybridBm25AppendDeltaTermSegments(index, delta, &deltaDirectory,
											maxItemSize);
		pfree(delta);
		startTerm += chunkTermCount;
	}

	deltaTermDirectoryBlkno =
		PgturbohybridBm25WriteDeltaTermDirectory(index, &deltaDirectory);
	deltaTermPages = PgturbohybridBm25DeltaDirectoryTermPages(&deltaDirectory);

	if (!PgturbohybridBm25ReadMetaForUpdate(index, &metaBuf, &metaPage,
									   &metaTuple, &xlogState))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("pgturbohybrid BM25 metadata is missing")));

	metaTuple->deltaStartBlkno = deltaStart;
	metaTuple->deltaTermDirectoryBlkno = deltaTermDirectoryBlkno;
	metaTuple->deltaGeneration++;
	metaTuple->deltaDocCount++;
	metaTuple->deltaTotalDocLen += docLen;
	metaTuple->deltaTermCount += collector.termCount;
	metaTuple->deltaPages = deltaPages;
	metaTuple->deltaTermPages = deltaTermPages;
	pgturbohybrid_bm25_delta_cursor_index = RelationGetRelid(index);
	pgturbohybrid_bm25_delta_cursor_relfilenumber = PgturbohybridGraphRelFileNumber(index);
	pgturbohybrid_bm25_delta_cursor_start = metaTuple->deltaStartBlkno;
	pgturbohybrid_bm25_delta_cursor_tail = deltaTail;
	pgturbohybrid_bm25_delta_cursor_generation = metaTuple->deltaGeneration;
	pgturbohybrid_bm25_delta_cursor_pages = metaTuple->deltaPages;

	if (xlogState != NULL)
		GenericXLogFinish(xlogState);
	else
		MarkBufferDirty(metaBuf);
	UnlockReleaseBuffer(metaBuf);

	PgturbohybridBm25InvalidateCache(index);
	PgturbohybridGraphInvalidateCaches(index);

	if (mustFree)
		pfree(vector);
	MemoryContextSwitchTo(oldCtx);
	MemoryContextDelete(ctx);
}

static bool
PgturbohybridBm25PageIsKind(Page page, uint16 pageKind)
{
	PgturbohybridGraphPageOpaque opaque = PgturbohybridGraphPageGetOpaque(page);

	return opaque->page_id == PGTURBOHYBRID_GRAPH_PAGE_ID &&
		(opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) == pageKind;
}

static bool
PgturbohybridBm25ReadMeta(Relation index, PgturbohybridBm25MetaTupleData *meta,
					 BlockNumber *metaBlkno)
{
	PgturbohybridGraphMetaPageData graphMeta;
	BlockNumber blkno;
	BlockNumber nblocks;
	Buffer		buf;
	Page		page;
	OffsetNumber maxoff;

	if (!PgturbohybridGraphReadMeta(index, &graphMeta) ||
		!BlockNumberIsValid(graphMeta.tqBm25MetaStartBlkno))
		return false;

	blkno = graphMeta.tqBm25MetaStartBlkno;
	nblocks = RelationGetNumberOfBlocks(index);
	if (blkno >= nblocks)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("pgturbohybrid BM25 metadata pointer is invalid"),
				 errdetail("Metapage points to block %u, but the index has only %u blocks.",
						   blkno, nblocks)));

	buf = ReadBuffer(index, blkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	if (!PgturbohybridBm25PageIsKind(page, PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_META))
	{
		UnlockReleaseBuffer(buf);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("pgturbohybrid BM25 metadata pointer is invalid"),
				 errdetail("Metapage points to block %u, which is not a BM25 metadata page.",
						   blkno)));
	}

	maxoff = PageGetMaxOffsetNumber(page);
	for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off = OffsetNumberNext(off))
	{
		ItemId		iid = PageGetItemId(page, off);
		PgturbohybridBm25MetaTuple tuple;

		if (!ItemIdIsUsed(iid))
			continue;

		tuple = (PgturbohybridBm25MetaTuple) PageGetItem(page, iid);
		if (tuple->type != PGTURBOHYBRID_BM25_META_TUPLE_TYPE)
			continue;

		*meta = *tuple;
		if (metaBlkno != NULL)
			*metaBlkno = blkno;
		UnlockReleaseBuffer(buf);
		return true;
	}

	UnlockReleaseBuffer(buf);
	ereport(ERROR,
			(errcode(ERRCODE_DATA_CORRUPTED),
			 errmsg("pgturbohybrid BM25 metadata tuple is missing"),
			 errdetail("Metapage points to BM25 metadata block %u, but no metadata tuple was found.",
					   blkno)));
}

static void
PgturbohybridBm25SetMetaBlock(Relation index, BlockNumber metaBlkno)
{
	Buffer		buf;
	Page		page;
	PgturbohybridGraphMetaPage metap;
	GenericXLogState *xlogState = NULL;

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
	if (metap->magicNumber != PGTURBOHYBRID_GRAPH_MAGIC_NUMBER ||
		metap->storageKind != PGTURBOHYBRID_GRAPH_STORAGE_QUANT_GRAPH_NATIVE)
	{
		if (xlogState != NULL)
			GenericXLogAbort(xlogState);
		UnlockReleaseBuffer(buf);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("pgturbohybrid native graph metadata is missing")));
	}

	metap->tqBm25MetaStartBlkno = metaBlkno;
	PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_META_UPDATE);

	if (xlogState != NULL)
		GenericXLogFinish(xlogState);
	else
		MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);
	PgturbohybridGraphLogGraphWalRecord(index, MAIN_FORKNUM, PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO,
						  PGTURBOHYBRID_GRAPH_GRAPH_OP_META_UPDATE);
}
