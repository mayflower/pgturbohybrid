#include "postgres.h"

#include <math.h>
#include <string.h>

#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && (defined(__aarch64__) || defined(_M_ARM64))
#include <arm_neon.h>
#endif
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
#include <immintrin.h>
#if !defined(__AVX2__) && (defined(__GNUC__) || defined(__clang__))
#define PGTURBOHYBRID_BM25_AVX2_TARGET __attribute__((target("avx2")))
#else
#define PGTURBOHYBRID_BM25_AVX2_TARGET
#endif
#else
#define PGTURBOHYBRID_BM25_AVX2_TARGET
#endif

#include "access/genam.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "portability/instr_time.h"
#include "storage/bufmgr.h"
#include "tsearch/ts_type.h"
#include "utils/hsearch.h"
#include "utils/memutils.h"
#include "utils/rel.h"

#include "pgturbohybrid.h"
#include "pgturbohybrid_quant.h"
#include "pgturbohybrid_am.h"
#include "pgturbohybrid_bm25.h"

#define PGTURBOHYBRID_BM25_MAX_NODE_COUNT_FOR_CACHE 10000000U
#define PGTURBOHYBRID_BM25_MAX_TERM_COUNT_FOR_CACHE 10000000U

int			pgturbohybrid_last_bm25_decode_kernel = PGTURBOHYBRID_BM25_KERNEL_SCALAR;
int			pgturbohybrid_last_bm25_score_kernel = PGTURBOHYBRID_BM25_KERNEL_SCALAR;
uint64		pgturbohybrid_last_bm25_simd_blocks = 0;
uint64		pgturbohybrid_last_bm25_scalar_tail_postings = 0;

typedef struct PgturbohybridBm25CacheLexiconEntry PgturbohybridBm25CacheLexiconEntry;

typedef struct PgturbohybridBm25ImpactEntry
{
	uint32		nodeId;
	uint16		tfNormQ16;
	uint16		reserved;
	float4		score;
} PgturbohybridBm25ImpactEntry;

const char *
PgturbohybridBm25KernelName(int kernel)
{
	switch (kernel)
	{
		case PGTURBOHYBRID_BM25_KERNEL_NEON_SOA:
			return "neon_soa";
		case PGTURBOHYBRID_BM25_KERNEL_AVX2_SOA:
			return "avx2_soa";
		case PGTURBOHYBRID_BM25_KERNEL_NEON:
			return "neon";
		case PGTURBOHYBRID_BM25_KERNEL_AVX2:
			return "avx2";
		case PGTURBOHYBRID_BM25_KERNEL_SCALAR:
		default:
			return "scalar";
	}
}

const char *
PgturbohybridBm25DeltaLookupModeName(int mode)
{
	switch ((PgturbohybridBm25DeltaLookupMode) mode)
	{
		case PGTURBOHYBRID_BM25_DELTA_LOOKUP_TERM_SEGMENT:
			return "term_segment";
		case PGTURBOHYBRID_BM25_DELTA_LOOKUP_DOC_SCAN:
			return "doc_scan";
		case PGTURBOHYBRID_BM25_DELTA_LOOKUP_NONE:
		default:
			return "none";
	}
}

typedef struct PgturbohybridBm25QueryTerm
{
	char	   *term;
	uint16		termLen;
	uint64		termHash;
	uint64		matchBit;
	int			qitemIndex;
	bool		hasLexicon;
	uint32		baseDf;
	uint32		deltaDf;
	PgturbohybridBm25CacheLexiconEntry *cacheEntry;
	PgturbohybridBm25LexiconEntryData lexicon;
} PgturbohybridBm25QueryTerm;

typedef enum PgturbohybridBm25QueryShape
{
	PGTURBOHYBRID_BM25_QUERY_EMPTY,
	PGTURBOHYBRID_BM25_QUERY_SINGLE,
	PGTURBOHYBRID_BM25_QUERY_PURE_OR,
	PGTURBOHYBRID_BM25_QUERY_PURE_AND,
	PGTURBOHYBRID_BM25_QUERY_MIXED
} PgturbohybridBm25QueryShape;

typedef enum PgturbohybridBm25BooleanEvalMode
{
	PGTURBOHYBRID_BM25_BOOLEAN_EVAL_NONE,
	PGTURBOHYBRID_BM25_BOOLEAN_EVAL_SINGLE,
	PGTURBOHYBRID_BM25_BOOLEAN_EVAL_MASK,
	PGTURBOHYBRID_BM25_BOOLEAN_EVAL_TREE
} PgturbohybridBm25BooleanEvalMode;

typedef enum PgturbohybridBm25WandBoundType
{
	PGTURBOHYBRID_BM25_WAND_BOUND_NONE,
	PGTURBOHYBRID_BM25_WAND_BOUND_TF,
	PGTURBOHYBRID_BM25_WAND_BOUND_TFNORM_Q16
} PgturbohybridBm25WandBoundType;

typedef struct PgturbohybridBm25QueryEval
{
	int			shape;
	int			mode;
	uint64		requiredMask;
} PgturbohybridBm25QueryEval;

const char *
PgturbohybridBm25QueryShapeName(int shape)
{
	switch ((PgturbohybridBm25QueryShape) shape)
	{
		case PGTURBOHYBRID_BM25_QUERY_EMPTY:
			return "empty";
		case PGTURBOHYBRID_BM25_QUERY_SINGLE:
			return "single";
		case PGTURBOHYBRID_BM25_QUERY_PURE_OR:
			return "pure_or";
		case PGTURBOHYBRID_BM25_QUERY_PURE_AND:
			return "pure_and";
		case PGTURBOHYBRID_BM25_QUERY_MIXED:
		default:
			return "mixed";
	}
}

const char *
PgturbohybridBm25BooleanEvalModeName(int mode)
{
	switch ((PgturbohybridBm25BooleanEvalMode) mode)
	{
		case PGTURBOHYBRID_BM25_BOOLEAN_EVAL_NONE:
			return "none";
		case PGTURBOHYBRID_BM25_BOOLEAN_EVAL_SINGLE:
			return "single";
		case PGTURBOHYBRID_BM25_BOOLEAN_EVAL_MASK:
			return "mask";
		case PGTURBOHYBRID_BM25_BOOLEAN_EVAL_TREE:
		default:
			return "tree";
	}
}

const char *
PgturbohybridBm25WandBoundTypeName(int boundType)
{
	switch ((PgturbohybridBm25WandBoundType) boundType)
	{
		case PGTURBOHYBRID_BM25_WAND_BOUND_TF:
			return "tf";
		case PGTURBOHYBRID_BM25_WAND_BOUND_TFNORM_Q16:
			return "tfnorm_q16";
		case PGTURBOHYBRID_BM25_WAND_BOUND_NONE:
		default:
			return "none";
	}
}

typedef struct PgturbohybridBm25NodeScore
{
	uint32		nodeId;
	float8		score;
} PgturbohybridBm25NodeScore;

typedef struct PgturbohybridBm25AccumulatorEntry
{
	uint32		nodeId;
	float8		score;
	uint32		docLen;
	ItemPointerData heaptid;
	uint64		matchedTerms;
	bool		hasDeltaDoc;
	int32		heapIndex;
} PgturbohybridBm25AccumulatorEntry;

typedef struct PgturbohybridBm25Accumulator
{
	HTAB	   *entries;
	PgturbohybridBm25AccumulatorEntry *denseEntries;
	uint32	   *denseGenerations;
	uint32		denseGeneration;
	uint32		denseCapacity;
	int			mode;
	PgturbohybridBm25NodeScore *touched;
	uint32		touchedCount;
	uint32		touchedCapacity;
	PgturbohybridBm25AccumulatorEntry **topHeap;
	uint32		topHeapCount;
	uint32		topHeapCapacity;
	double		threshold;
	double		seedThreshold;
	MemoryContext memoryContext;
	PgturbohybridBm25QueryStats *stats;
} PgturbohybridBm25Accumulator;

typedef struct PgturbohybridBm25DenseAccumulatorStorage
{
	PgturbohybridBm25AccumulatorEntry *entries;
	uint32	   *generations;
	uint32		capacity;
	uint32		generation;
} PgturbohybridBm25DenseAccumulatorStorage;

static PgturbohybridBm25DenseAccumulatorStorage pgturbohybrid_bm25_dense_accumulator;

typedef struct PgturbohybridBm25PostingIterator
{
	Relation	index;
	struct PgturbohybridBm25Cache *cache;
	PgturbohybridBm25QueryTerm *term;
	double		idf;
	double		avgDocLen;
	const uint32 *docLens;
	const bool *liveNodes;
	uint32		nodeCount;
	uint32		chunkLimit;
	uint32		chunkNo;
	BlockNumber blkno;
	OffsetNumber offno;
	PgturbohybridBm25Posting *postings;
	uint16		postingsCapacity;
	uint16		count;
	uint16		pos;
	uint16		maxTf;
	uint16		maxTfNormQ16;
	float4		maxScoreFactor;
	uint32		lastNodeId;
	BlockNumber nextBlkno;
	OffsetNumber nextOffno;
	bool		valid;
	MemoryContext memoryContext;
	PgturbohybridBm25QueryStats *stats;
} PgturbohybridBm25PostingIterator;

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

struct PgturbohybridBm25CacheLexiconEntry
{
	uint16		termLen;
	uint64		termHash;
	uint32		termId;
	uint32		df;
	uint32		cf;
	BlockNumber postingsBlkno;
	OffsetNumber postingsOffno;
	uint32		postingsChunkCount;
	uint32		postingsBytes;
	BlockNumber blockMaxBlkno;
	OffsetNumber blockMaxOffno;
	BlockNumber impactBlkno;
	OffsetNumber impactOffno;
	uint16		storedImpactCount;
	uint32		termOffset;
	char	   *termBytes;
	bool		impactBuilt;
	bool		impactEligible;
	uint32		impactCount;
	PgturbohybridBm25ImpactEntry *impactHead;
};

typedef struct PgturbohybridBm25DeltaCachePosting
{
	uint32		nodeId;
	uint16		tf;
	uint16		reserved;
	uint32		docLen;
	ItemPointerData heaptid;
} PgturbohybridBm25DeltaCachePosting;

typedef struct PgturbohybridBm25DeltaCacheEntry
{
	uint16		termLen;
	uint64		termHash;
	uint32		df;
	uint32		postingCount;
	char	   *termBytes;
	PgturbohybridBm25DeltaCachePosting *postings;
} PgturbohybridBm25DeltaCacheEntry;

typedef struct PgturbohybridBm25DeltaBuildEntry
{
	uint16		termLen;
	uint64		termHash;
	uint32		nodeId;
	uint16		tf;
	uint16		reserved;
	uint32		docLen;
	ItemPointerData heaptid;
	char	   *termBytes;
} PgturbohybridBm25DeltaBuildEntry;

typedef struct PgturbohybridBm25HotPostingsKey
{
	uint32		termId;
	BlockNumber blkno;
	OffsetNumber offno;
} PgturbohybridBm25HotPostingsKey;

typedef struct PgturbohybridBm25HotPostingsEntry
{
	PgturbohybridBm25HotPostingsKey key;
	uint16		count;
	uint16		maxTf;
	uint16		maxTfNormQ16;
	float4		maxScoreFactor;
	uint32		lastNodeId;
	BlockNumber nextBlkno;
	OffsetNumber nextOffno;
	Size		bytes;
	uint64		lastUsed;
	PgturbohybridBm25Posting *postings;
	struct PgturbohybridBm25HotPostingsEntry *lruPrev;
	struct PgturbohybridBm25HotPostingsEntry *lruNext;
} PgturbohybridBm25HotPostingsEntry;

typedef struct PgturbohybridBm25HotPostingsHashEntry
{
	PgturbohybridBm25HotPostingsKey key;
	PgturbohybridBm25HotPostingsEntry *entry;
} PgturbohybridBm25HotPostingsHashEntry;

typedef struct PgturbohybridBm25Cache
{
	Oid			relid;
	Oid			relfilenumber;
	uint16		graphFlags;
	uint32		nodeCount;
	uint32		docCount;
	uint32		termCount;
	uint64		deltaGeneration;
	uint64		lastCompactionGeneration;
	BlockNumber docStatsStartBlkno;
	BlockNumber lexiconStartBlkno;
	BlockNumber postingsStartBlkno;
	BlockNumber blockMaxStartBlkno;
	BlockNumber deltaStartBlkno;
	BlockNumber deltaTermDirectoryBlkno;
	uint32	   *docLens;
	ItemPointerData *heapTids;
	bool	   *liveNodes;
	bool		docStatsLoaded;
	bool		livenessLoaded;
	PgturbohybridBm25CacheLexiconEntry *lexicon;
	uint32		lexiconCount;
	char	   *termBytesArena;
	uint32		termBytesArenaUsed;
	uint32		termBytesArenaCapacity;
	bool		deltaCacheBuilt;
	PgturbohybridBm25DeltaCacheEntry *deltaTerms;
	uint32		deltaTermCount;
	uint32		deltaPostingCount;
	uint64		deltaCacheBytes;
	HTAB	   *hotPostingsHash;
	PgturbohybridBm25HotPostingsEntry *hotPostingsHead;
	PgturbohybridBm25HotPostingsEntry *hotPostingsTail;
	uint64		hotPostingsBytes;
	uint64		hotPostingsEvictions;
	uint64		hotPostingsClock;
	MemoryContext ctx;
	struct PgturbohybridBm25Cache *next;
} PgturbohybridBm25Cache;

static PgturbohybridBm25Cache *pgturbohybrid_bm25_cache_list = NULL;

static int PgturbohybridBm25FindQueryTerm(PgturbohybridBm25QueryTerm *terms,
									 int termCount, const char *term,
									 uint16 termLen);

static Size
PgturbohybridBm25ArrayAllocSize(Size elemSize, Size count)
{
	if (elemSize != 0 && count > MaxAllocSize / elemSize)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pgturbohybrid BM25 allocation request is too large")));

	return elemSize * count;
}

static uint32
PgturbohybridBm25GrowCapacity32(uint32 capacity, uint32 needed, Size elemSize)
{
	uint32		newCapacity = Max(capacity, 1);

	while (newCapacity < needed)
	{
		if (newCapacity > PG_UINT32_MAX / 2)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("pgturbohybrid BM25 array capacity is too large")));
		newCapacity *= 2;
	}

	(void) PgturbohybridBm25ArrayAllocSize(elemSize, newCapacity);
	return newCapacity;
}

static void *
PgturbohybridBm25MemoryContextAllocArray(MemoryContext context,
									Size elemSize, Size count)
{
	return MemoryContextAlloc(context,
							  PgturbohybridBm25ArrayAllocSize(elemSize, count));
}

static void *
PgturbohybridBm25MemoryContextAllocZeroArray(MemoryContext context,
										Size elemSize, Size count)
{
	return MemoryContextAllocZero(context,
								  PgturbohybridBm25ArrayAllocSize(elemSize, count));
}

static void *
PgturbohybridBm25PallocArray(Size elemSize, Size count)
{
	return palloc(PgturbohybridBm25ArrayAllocSize(elemSize, count));
}

static void *
PgturbohybridBm25Palloc0Array(Size elemSize, Size count)
{
	return palloc0(PgturbohybridBm25ArrayAllocSize(elemSize, count));
}

static void *
PgturbohybridBm25RepallocArray(void *pointer, Size elemSize, Size count)
{
	return repalloc(pointer, PgturbohybridBm25ArrayAllocSize(elemSize, count));
}

static void
PgturbohybridBm25ValidateBlockPointer(const char *name, BlockNumber blkno,
								 BlockNumber nblocks)
{
	if (BlockNumberIsValid(blkno) && blkno >= nblocks)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("pgturbohybrid BM25 metadata block pointer is invalid"),
				 errdetail("%s points to block %u, but the index has only %u blocks.",
						   name, blkno, nblocks)));
}

static void
PgturbohybridBm25ValidateCacheMeta(Relation index,
							  const PgturbohybridBm25MetaTupleData *bm25Meta,
							  const PgturbohybridGraphMetaPageData *graphMeta)
{
	BlockNumber nblocks = RelationGetNumberOfBlocks(index);
	Size		maxTuplesFromBlocks = (Size) Max(nblocks, 1) * MaxOffsetNumber;

	if (graphMeta->tqNodeCount > PGTURBOHYBRID_BM25_MAX_NODE_COUNT_FOR_CACHE ||
		(Size) graphMeta->tqNodeCount > maxTuplesFromBlocks)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("pgturbohybrid BM25 metadata node count is invalid"),
				 errdetail("Metadata reports %u nodes for an index with %u blocks.",
						   graphMeta->tqNodeCount, nblocks)));

	if (bm25Meta->docCount > graphMeta->tqNodeCount)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("pgturbohybrid BM25 metadata document count is invalid"),
				 errdetail("BM25 metadata reports %u documents but graph metadata reports %u nodes.",
						   bm25Meta->docCount, graphMeta->tqNodeCount)));

	if (bm25Meta->termCount > PGTURBOHYBRID_BM25_MAX_TERM_COUNT_FOR_CACHE ||
		(Size) bm25Meta->termCount > maxTuplesFromBlocks)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("pgturbohybrid BM25 metadata term count is invalid"),
				 errdetail("BM25 metadata reports %u terms for an index with %u blocks.",
						   bm25Meta->termCount, nblocks)));

	PgturbohybridBm25ValidateBlockPointer("docStatsStartBlkno",
										  bm25Meta->docStatsStartBlkno, nblocks);
	PgturbohybridBm25ValidateBlockPointer("lexiconStartBlkno",
										  bm25Meta->lexiconStartBlkno, nblocks);
	PgturbohybridBm25ValidateBlockPointer("postingsStartBlkno",
										  bm25Meta->postingsStartBlkno, nblocks);
	PgturbohybridBm25ValidateBlockPointer("blockMaxStartBlkno",
										  bm25Meta->blockMaxStartBlkno, nblocks);
	PgturbohybridBm25ValidateBlockPointer("deltaStartBlkno",
										  bm25Meta->deltaStartBlkno, nblocks);
	PgturbohybridBm25ValidateBlockPointer("deltaTermDirectoryBlkno",
										  bm25Meta->deltaTermDirectoryBlkno, nblocks);
}

static uint64
PgturbohybridBm25ElapsedUs(instr_time start)
{
	instr_time	end;

	INSTR_TIME_SET_CURRENT(end);
	INSTR_TIME_SUBTRACT(end, start);
	return (uint64) INSTR_TIME_GET_MICROSEC(end);
}

static uint64
PgturbohybridBm25HashTerm(const char *term, uint16 len)
{
	uint64		hash = UINT64CONST(1469598103934665603);

	for (uint16 i = 0; i < len; i++)
	{
		hash ^= (unsigned char) term[i];
		hash *= UINT64CONST(1099511628211);
	}

	return hash;
}

static bool
PgturbohybridBm25PageIsKind(Page page, uint16 pageKind)
{
	PgturbohybridGraphPageOpaque opaque = PgturbohybridGraphPageGetOpaque(page);

	return opaque->page_id == PGTURBOHYBRID_GRAPH_PAGE_ID &&
		(opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) == pageKind;
}

static char *
PgturbohybridBm25DeltaTermTupleBytes(PgturbohybridBm25DeltaTermTuple tuple)
{
	return ((char *) tuple) +
		offsetof(PgturbohybridBm25DeltaTermTupleData, postings) +
		sizeof(PgturbohybridBm25DeltaTermPosting) * tuple->postingCount;
}

static uint32
PgturbohybridBm25DeltaTermBucketNo(uint64 termHash)
{
	return (uint32) (termHash & (PGTURBOHYBRID_BM25_DELTA_TERM_BUCKETS - 1));
}

static void
PgturbohybridBm25AccumulatorInit(PgturbohybridBm25Accumulator *acc,
							MemoryContext memoryContext, uint32 initialSize,
							uint32 topK, int mode, uint32 nodeCount,
							PgturbohybridBm25QueryStats *stats)
{
	HASHCTL		ctl;

	memset(acc, 0, sizeof(*acc));
	acc->mode = mode;
	if (mode == PGTURBOHYBRID_BM25_ACCUMULATOR_DENSE)
	{
		MemoryContext oldCtx;

		oldCtx = MemoryContextSwitchTo(CacheMemoryContext);
		if (pgturbohybrid_bm25_dense_accumulator.capacity < nodeCount)
		{
			uint32		oldCapacity = pgturbohybrid_bm25_dense_accumulator.capacity;

			if (pgturbohybrid_bm25_dense_accumulator.entries == NULL)
				pgturbohybrid_bm25_dense_accumulator.entries =
					PgturbohybridBm25MemoryContextAllocArray(CacheMemoryContext,
															 sizeof(PgturbohybridBm25AccumulatorEntry),
															 nodeCount);
			else
				pgturbohybrid_bm25_dense_accumulator.entries =
					PgturbohybridBm25RepallocArray(pgturbohybrid_bm25_dense_accumulator.entries,
												   sizeof(PgturbohybridBm25AccumulatorEntry),
												   nodeCount);
			if (pgturbohybrid_bm25_dense_accumulator.generations == NULL)
				pgturbohybrid_bm25_dense_accumulator.generations =
					PgturbohybridBm25MemoryContextAllocZeroArray(CacheMemoryContext,
																 sizeof(uint32),
																 nodeCount);
			else
			{
				pgturbohybrid_bm25_dense_accumulator.generations =
					PgturbohybridBm25RepallocArray(pgturbohybrid_bm25_dense_accumulator.generations,
												   sizeof(uint32),
												   nodeCount);
				memset(pgturbohybrid_bm25_dense_accumulator.generations + oldCapacity,
					   0, PgturbohybridBm25ArrayAllocSize(sizeof(uint32),
														 nodeCount - oldCapacity));
			}
			pgturbohybrid_bm25_dense_accumulator.capacity = nodeCount;
		}
		if (++pgturbohybrid_bm25_dense_accumulator.generation == 0)
		{
			memset(pgturbohybrid_bm25_dense_accumulator.generations, 0,
				   PgturbohybridBm25ArrayAllocSize(sizeof(uint32),
												   pgturbohybrid_bm25_dense_accumulator.capacity));
			pgturbohybrid_bm25_dense_accumulator.generation = 1;
		}
		MemoryContextSwitchTo(oldCtx);

		acc->denseEntries = pgturbohybrid_bm25_dense_accumulator.entries;
		acc->denseGenerations = pgturbohybrid_bm25_dense_accumulator.generations;
		acc->denseGeneration = pgturbohybrid_bm25_dense_accumulator.generation;
		acc->denseCapacity = pgturbohybrid_bm25_dense_accumulator.capacity;
	}
	else
	{
		memset(&ctl, 0, sizeof(ctl));
		ctl.keysize = sizeof(uint32);
		ctl.entrysize = sizeof(PgturbohybridBm25AccumulatorEntry);
		ctl.hcxt = memoryContext;
		acc->entries = hash_create("pgturbohybrid BM25 query accumulator",
								   Max(initialSize, 16),
								   &ctl,
								   HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
	}
	acc->touchedCapacity = Max(initialSize, 16);
	acc->touched = PgturbohybridBm25MemoryContextAllocZeroArray(memoryContext,
															   sizeof(PgturbohybridBm25NodeScore),
															   acc->touchedCapacity);
	acc->topHeapCapacity = topK;
	if (topK > 0)
		acc->topHeap = PgturbohybridBm25MemoryContextAllocZeroArray(memoryContext,
																	sizeof(PgturbohybridBm25AccumulatorEntry *),
																	topK);
	acc->memoryContext = memoryContext;
	acc->stats = stats;
	if (stats != NULL)
		stats->accumulatorMode = mode;
}

static bool
PgturbohybridBm25AccumulatorHeapLess(PgturbohybridBm25AccumulatorEntry *a,
								PgturbohybridBm25AccumulatorEntry *b)
{
	if (a->score < b->score)
		return true;
	if (a->score > b->score)
		return false;
	return a->nodeId > b->nodeId;
}

static void
PgturbohybridBm25AccumulatorHeapSwap(PgturbohybridBm25Accumulator *acc, uint32 a,
								uint32 b)
{
	PgturbohybridBm25AccumulatorEntry *tmp = acc->topHeap[a];

	acc->topHeap[a] = acc->topHeap[b];
	acc->topHeap[b] = tmp;
	acc->topHeap[a]->heapIndex = a;
	acc->topHeap[b]->heapIndex = b;
}

static void
PgturbohybridBm25AccumulatorHeapSiftUp(PgturbohybridBm25Accumulator *acc, uint32 pos)
{
	while (pos > 0)
	{
		uint32		parent = (pos - 1) / 2;

		if (!PgturbohybridBm25AccumulatorHeapLess(acc->topHeap[pos],
											 acc->topHeap[parent]))
			break;
		PgturbohybridBm25AccumulatorHeapSwap(acc, pos, parent);
		pos = parent;
	}
}

static void
PgturbohybridBm25AccumulatorHeapSiftDown(PgturbohybridBm25Accumulator *acc, uint32 pos)
{
	for (;;)
	{
		uint32		left = pos * 2 + 1;
		uint32		right = left + 1;
		uint32		smallest = pos;

		if (left < acc->topHeapCount &&
			PgturbohybridBm25AccumulatorHeapLess(acc->topHeap[left],
											acc->topHeap[smallest]))
			smallest = left;
		if (right < acc->topHeapCount &&
			PgturbohybridBm25AccumulatorHeapLess(acc->topHeap[right],
											acc->topHeap[smallest]))
			smallest = right;
		if (smallest == pos)
			break;
		PgturbohybridBm25AccumulatorHeapSwap(acc, pos, smallest);
		pos = smallest;
	}
}

static void
PgturbohybridBm25AccumulatorRefreshThreshold(PgturbohybridBm25Accumulator *acc)
{
	double		oldThreshold = acc->threshold;

	if (acc->topHeapCapacity > 0 &&
		acc->topHeapCount == acc->topHeapCapacity)
		acc->threshold = acc->topHeap[0]->score;
	else
		acc->threshold = 0.0;
	if (acc->stats != NULL &&
		acc->threshold > 0.0 &&
		acc->threshold != oldThreshold)
		acc->stats->wandThresholdUpdates++;
}

static void
PgturbohybridBm25AccumulatorUpdateTopK(PgturbohybridBm25Accumulator *acc,
								  PgturbohybridBm25AccumulatorEntry *entry)
{
	if (acc->topHeapCapacity == 0)
		return;

	if (entry->heapIndex >= 0)
	{
		/*
		 * Scores only increase.  This is a min-heap, so an existing entry can
		 * only move away from the root after a score update.
		 */
		PgturbohybridBm25AccumulatorHeapSiftDown(acc, (uint32) entry->heapIndex);
	}
	else if (acc->topHeapCount < acc->topHeapCapacity)
	{
		entry->heapIndex = acc->topHeapCount;
		acc->topHeap[acc->topHeapCount++] = entry;
		PgturbohybridBm25AccumulatorHeapSiftUp(acc, (uint32) entry->heapIndex);
	}
	else if (entry->score > acc->topHeap[0]->score)
	{
		acc->topHeap[0]->heapIndex = -1;
		entry->heapIndex = 0;
		acc->topHeap[0] = entry;
		PgturbohybridBm25AccumulatorHeapSiftDown(acc, 0);
		if (acc->stats != NULL)
			acc->stats->wandHeapReplacements++;
	}

	PgturbohybridBm25AccumulatorRefreshThreshold(acc);
}

static PgturbohybridBm25AccumulatorEntry *
PgturbohybridBm25AccumulatorLookup(PgturbohybridBm25Accumulator *acc, uint32 nodeId,
							  bool create)
{
	PgturbohybridBm25AccumulatorEntry *entry;
	bool		found;

	if (acc->mode == PGTURBOHYBRID_BM25_ACCUMULATOR_DENSE)
	{
		if (nodeId >= acc->denseCapacity)
			return NULL;
		entry = &acc->denseEntries[nodeId];
		found = acc->denseGenerations[nodeId] == acc->denseGeneration;
		if (found || !create)
			return found ? entry : NULL;

		acc->denseGenerations[nodeId] = acc->denseGeneration;
	}
	else
	{
		if (acc->stats != NULL)
			acc->stats->accumulatorHashLookups++;
		entry = hash_search(acc->entries, &nodeId,
							create ? HASH_ENTER : HASH_FIND, &found);
		if (entry == NULL || found || !create)
			return entry;
	}

	entry->nodeId = nodeId;
	entry->score = 0.0;
	entry->docLen = 0;
	ItemPointerSetInvalid(&entry->heaptid);
	entry->matchedTerms = 0;
	entry->hasDeltaDoc = false;
	entry->heapIndex = -1;

	if (acc->touchedCount >= acc->touchedCapacity)
	{
		acc->touchedCapacity =
			PgturbohybridBm25GrowCapacity32(acc->touchedCapacity,
											acc->touchedCount + 1,
											sizeof(PgturbohybridBm25NodeScore));
		acc->touched = PgturbohybridBm25RepallocArray(acc->touched,
													  sizeof(PgturbohybridBm25NodeScore),
													  acc->touchedCapacity);
	}
	acc->touched[acc->touchedCount].nodeId = nodeId;
	acc->touched[acc->touchedCount].score = 0.0;
	acc->touchedCount++;

	return entry;
}

static void
PgturbohybridBm25AccumulatorAddTermScore(PgturbohybridBm25Accumulator *acc,
									uint32 nodeId, float8 score,
									uint64 matchBit)
{
	PgturbohybridBm25AccumulatorEntry *entry;

	entry = PgturbohybridBm25AccumulatorLookup(acc, nodeId, true);
	if (entry == NULL)
		return;
	entry->score += score;
	entry->matchedTerms |= matchBit;
	if (acc->stats != NULL &&
		acc->mode == PGTURBOHYBRID_BM25_ACCUMULATOR_DENSE)
		acc->stats->accumulatorDenseUpdates++;
	PgturbohybridBm25AccumulatorUpdateTopK(acc, entry);
}

static int
PgturbohybridBm25CacheLexiconCompare(const void *a, const void *b)
{
	const PgturbohybridBm25CacheLexiconEntry *la =
		(const PgturbohybridBm25CacheLexiconEntry *) a;
	const PgturbohybridBm25CacheLexiconEntry *lb =
		(const PgturbohybridBm25CacheLexiconEntry *) b;
	int			cmp;

	if (la->termHash != lb->termHash)
		return la->termHash < lb->termHash ? -1 : 1;
	if (la->termLen != lb->termLen)
		return la->termLen < lb->termLen ? -1 : 1;

	cmp = memcmp(la->termBytes, lb->termBytes, la->termLen);
	if (cmp != 0)
		return cmp;

	return (la->termId > lb->termId) -
		(la->termId < lb->termId);
}

static int
PgturbohybridBm25DeltaBuildCompare(const void *a, const void *b)
{
	const PgturbohybridBm25DeltaBuildEntry *la =
		(const PgturbohybridBm25DeltaBuildEntry *) a;
	const PgturbohybridBm25DeltaBuildEntry *lb =
		(const PgturbohybridBm25DeltaBuildEntry *) b;
	int			cmp;

	if (la->termHash != lb->termHash)
		return la->termHash < lb->termHash ? -1 : 1;
	if (la->termLen != lb->termLen)
		return la->termLen < lb->termLen ? -1 : 1;
	cmp = memcmp(la->termBytes, lb->termBytes, la->termLen);
	if (cmp != 0)
		return cmp;
	return (la->nodeId > lb->nodeId) - (la->nodeId < lb->nodeId);
}

static int
PgturbohybridBm25DeltaCacheCompare(const void *a, const void *b)
{
	const PgturbohybridBm25DeltaCacheEntry *la =
		(const PgturbohybridBm25DeltaCacheEntry *) a;
	const PgturbohybridBm25DeltaCacheEntry *lb =
		(const PgturbohybridBm25DeltaCacheEntry *) b;
	int			cmp;

	if (la->termHash != lb->termHash)
		return la->termHash < lb->termHash ? -1 : 1;
	if (la->termLen != lb->termLen)
		return la->termLen < lb->termLen ? -1 : 1;
	cmp = memcmp(la->termBytes, lb->termBytes, la->termLen);
	if (cmp != 0)
		return cmp;
	return 0;
}

static char *
PgturbohybridBm25DeltaTermBytes(PgturbohybridBm25DeltaTuple tuple)
{
	return ((char *) tuple) + offsetof(PgturbohybridBm25DeltaTupleData, terms) +
		sizeof(PgturbohybridBm25DeltaTerm) * tuple->termCount;
}

static bool
PgturbohybridBm25ReadMeta(Relation index, PgturbohybridBm25MetaTupleData *meta)
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
		if (tuple->type == PGTURBOHYBRID_BM25_META_TUPLE_TYPE)
		{
			*meta = *tuple;
			UnlockReleaseBuffer(buf);
			return true;
		}
	}

	UnlockReleaseBuffer(buf);
	ereport(ERROR,
			(errcode(ERRCODE_DATA_CORRUPTED),
			 errmsg("pgturbohybrid BM25 metadata tuple is missing"),
			 errdetail("Metapage points to BM25 metadata block %u, but no metadata tuple was found.",
					   blkno)));
}

bool
PgturbohybridBm25GetPlanningStats(Relation index, PgturbohybridBm25PlanningStats *stats)
{
	PgturbohybridBm25MetaTupleData meta;

	memset(stats, 0, sizeof(*stats));
	if (!PgturbohybridBm25ReadMeta(index, &meta))
		return false;

	stats->docCount = meta.docCount;
	stats->termCount = meta.termCount;
	stats->termTupleCount = meta.termTupleCount;
	stats->deltaDocCount = meta.deltaDocCount;
	stats->deltaTermCount = meta.deltaTermCount;
	stats->postingsPages = meta.postingsPages;
	stats->blockMaxPages = meta.blockMaxPages;
	stats->deltaPages = meta.deltaPages;
	stats->deltaTermPages = meta.deltaTermPages;
	stats->hasBm25 = true;
	return true;
}

static void
PgturbohybridBm25LoadDocStats(Relation index, const PgturbohybridBm25MetaTupleData *meta,
						 uint32 nodeCount, uint32 *docLens)
{
	BlockNumber blkno = meta->docStatsStartBlkno;

	memset(docLens, 0, PgturbohybridBm25ArrayAllocSize(sizeof(uint32), nodeCount));
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
			return;
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
PgturbohybridBm25LoadHeapTids(Relation index, const PgturbohybridGraphMetaPageData *meta,
						 ItemPointerData *heapTids, bool *liveNodes)
{
	bool		tqWeighted = (meta->tqFlags & PGTURBOHYBRID_GRAPH_TQ_WEIGHTED) != 0;
	int			codeTuplesPerPage =
		PgturbohybridGraphTuplesPerPage(PgturbohybridGraphCodeTupleSize(meta->dimensions,
												  meta->tqPayloadCount,
												  meta->tqBits,
												  tqWeighted));
	int			codePageCount = PgturbohybridGraphPageCount(meta->tqNodeCount,
												 codeTuplesPerPage);
	BlockNumber *codeBlknos;

	memset(heapTids, 0,
		   PgturbohybridBm25ArrayAllocSize(sizeof(ItemPointerData),
										   meta->tqNodeCount));
	memset(liveNodes, 0,
		   PgturbohybridBm25ArrayAllocSize(sizeof(bool), meta->tqNodeCount));
	codeBlknos = PgturbohybridBm25PallocArray(sizeof(BlockNumber), codePageCount);
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
			if (tuple->type == PGTURBOHYBRID_GRAPH_CODE_TUPLE_TYPE &&
				tuple->nodeId < meta->tqNodeCount)
			{
				heapTids[tuple->nodeId] = tuple->heaptid;
				liveNodes[tuple->nodeId] =
					(tuple->flags & PGTURBOHYBRID_GRAPH_NODE_DEAD) == 0;
			}
		}

		UnlockReleaseBuffer(buf);
	}

	pfree(codeBlknos);
}

static bool
PgturbohybridBm25CacheMatches(PgturbohybridBm25Cache *cache, Relation index,
						 const PgturbohybridBm25MetaTupleData *bm25Meta,
						 const PgturbohybridGraphMetaPageData *graphMeta)
{
	return cache->relid == RelationGetRelid(index) &&
		cache->relfilenumber == PgturbohybridGraphRelFileNumber(index) &&
		cache->graphFlags == graphMeta->graphFlags &&
		cache->nodeCount == graphMeta->tqNodeCount &&
		cache->docCount == bm25Meta->docCount &&
		cache->termCount == bm25Meta->termCount &&
		cache->deltaGeneration == bm25Meta->deltaGeneration &&
		cache->lastCompactionGeneration == bm25Meta->lastCompactionGeneration &&
		cache->docStatsStartBlkno == bm25Meta->docStatsStartBlkno &&
		cache->lexiconStartBlkno == bm25Meta->lexiconStartBlkno &&
		cache->postingsStartBlkno == bm25Meta->postingsStartBlkno &&
		cache->blockMaxStartBlkno == bm25Meta->blockMaxStartBlkno &&
		cache->deltaStartBlkno == bm25Meta->deltaStartBlkno &&
		cache->deltaTermDirectoryBlkno == bm25Meta->deltaTermDirectoryBlkno;
}

static void
PgturbohybridBm25DropStaleCaches(Relation index,
							const PgturbohybridBm25MetaTupleData *bm25Meta,
							const PgturbohybridGraphMetaPageData *graphMeta)
{
	PgturbohybridBm25Cache **link = &pgturbohybrid_bm25_cache_list;

	while (*link != NULL)
	{
		PgturbohybridBm25Cache *cache = *link;

		if (cache->relid == RelationGetRelid(index) &&
			!PgturbohybridBm25CacheMatches(cache, index, bm25Meta, graphMeta))
		{
			*link = cache->next;
			MemoryContextDelete(cache->ctx);
			continue;
		}
		link = &cache->next;
	}
}

void
PgturbohybridBm25InvalidateCache(Relation index)
{
	PgturbohybridBm25Cache **link = &pgturbohybrid_bm25_cache_list;
	Oid			relid = RelationGetRelid(index);

	while (*link != NULL)
	{
		PgturbohybridBm25Cache *cache = *link;

		if (cache->relid == relid)
		{
			*link = cache->next;
			MemoryContextDelete(cache->ctx);
			continue;
		}
		link = &cache->next;
	}
}

static uint32
PgturbohybridBm25CacheAppendTermBytes(PgturbohybridBm25Cache *cache,
								 const char *termBytes, uint16 termLen)
{
	uint32		offset = cache->termBytesArenaUsed;
	uint32		required;

	if (offset > PG_UINT32_MAX - termLen)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pgturbohybrid BM25 term byte arena is too large")));
	required = offset + termLen;

	if (required > cache->termBytesArenaCapacity)
	{
		uint32		newCapacity = Max(cache->termBytesArenaCapacity, 1024);

		newCapacity = PgturbohybridBm25GrowCapacity32(newCapacity, required, 1);
		if (cache->termBytesArena == NULL)
			cache->termBytesArena = PgturbohybridBm25PallocArray(1, newCapacity);
		else
			cache->termBytesArena =
				PgturbohybridBm25RepallocArray(cache->termBytesArena, 1,
											   newCapacity);
		cache->termBytesArenaCapacity = newCapacity;
	}

	memcpy(cache->termBytesArena + offset, termBytes, termLen);
	cache->termBytesArenaUsed = required;
	return offset;
}

static void
PgturbohybridBm25LoadLexiconDirectory(Relation index,
								 const PgturbohybridBm25MetaTupleData *meta,
								 PgturbohybridBm25Cache *cache)
{
	BlockNumber blkno = meta->lexiconStartBlkno;

	cache->lexicon =
		PgturbohybridBm25Palloc0Array(sizeof(PgturbohybridBm25CacheLexiconEntry),
									  Max(meta->termCount, 1));
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
		if (!PgturbohybridBm25PageIsKind(page, PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_LEXICON))
		{
			UnlockReleaseBuffer(buf);
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("pgturbohybrid BM25 lexicon page has unexpected page kind")));
		}

		nextblkno = opaque->nextblkno;
		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off = OffsetNumberNext(off))
		{
			ItemId		iid = PageGetItemId(page, off);
			PgturbohybridBm25LexiconEntry tuple;
			PgturbohybridBm25CacheLexiconEntry *entry;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (PgturbohybridBm25LexiconEntry) PageGetItem(page, iid);
			if (tuple->type != PGTURBOHYBRID_BM25_LEXICON_TUPLE_TYPE)
				continue;
			if (cache->lexiconCount >= meta->termCount)
			{
				UnlockReleaseBuffer(buf);
				ereport(ERROR,
						(errcode(ERRCODE_DATA_CORRUPTED),
						 errmsg("pgturbohybrid BM25 lexicon has more entries than metadata declares")));
			}

			entry = &cache->lexicon[cache->lexiconCount++];
			entry->termLen = tuple->termLen;
			entry->termHash = tuple->termHash;
			entry->termId = tuple->termId;
			entry->df = tuple->df;
			entry->cf = tuple->cf;
			entry->postingsBlkno = tuple->postingsBlkno;
			entry->postingsOffno = tuple->postingsOffno;
			entry->postingsChunkCount = tuple->postingsChunkCount;
			entry->postingsBytes = tuple->postingsBytes;
			entry->blockMaxBlkno = tuple->blockMaxBlkno;
			entry->blockMaxOffno = tuple->blockMaxOffno;
			entry->impactBlkno = tuple->impactBlkno;
			entry->impactOffno = tuple->impactOffno;
			entry->storedImpactCount = tuple->impactCount;
			entry->termOffset =
				PgturbohybridBm25CacheAppendTermBytes(cache, tuple->termBytes,
												 tuple->termLen);
		}

		UnlockReleaseBuffer(buf);
		blkno = nextblkno;
	}

	for (uint32 i = 0; i < cache->lexiconCount; i++)
		cache->lexicon[i].termBytes =
			cache->termBytesArena + cache->lexicon[i].termOffset;

	if (cache->lexiconCount > 1)
		qsort(cache->lexicon, cache->lexiconCount,
			  sizeof(PgturbohybridBm25CacheLexiconEntry),
			  PgturbohybridBm25CacheLexiconCompare);
}

static PgturbohybridBm25Cache *
PgturbohybridBm25BuildCache(Relation index,
					   const PgturbohybridBm25MetaTupleData *bm25Meta,
					   const PgturbohybridGraphMetaPageData *graphMeta)
{
	MemoryContext cacheCtx;
	MemoryContext oldCtx;
	PgturbohybridBm25Cache *cache;

	PgturbohybridBm25ValidateCacheMeta(index, bm25Meta, graphMeta);
	cacheCtx = AllocSetContextCreate(CacheMemoryContext,
									 "pgturbohybrid BM25 reader cache",
									 ALLOCSET_DEFAULT_SIZES);
	oldCtx = MemoryContextSwitchTo(cacheCtx);

	cache = palloc0(sizeof(PgturbohybridBm25Cache));
	cache->relid = RelationGetRelid(index);
	cache->relfilenumber = PgturbohybridGraphRelFileNumber(index);
	cache->graphFlags = graphMeta->graphFlags;
	cache->nodeCount = graphMeta->tqNodeCount;
	cache->docCount = bm25Meta->docCount;
	cache->termCount = bm25Meta->termCount;
	cache->deltaGeneration = bm25Meta->deltaGeneration;
	cache->lastCompactionGeneration = bm25Meta->lastCompactionGeneration;
	cache->docStatsStartBlkno = bm25Meta->docStatsStartBlkno;
	cache->lexiconStartBlkno = bm25Meta->lexiconStartBlkno;
	cache->postingsStartBlkno = bm25Meta->postingsStartBlkno;
	cache->blockMaxStartBlkno = bm25Meta->blockMaxStartBlkno;
	cache->deltaStartBlkno = bm25Meta->deltaStartBlkno;
	cache->deltaTermDirectoryBlkno = bm25Meta->deltaTermDirectoryBlkno;
	cache->ctx = cacheCtx;
	PgturbohybridBm25LoadLexiconDirectory(index, bm25Meta, cache);

	cache->next = pgturbohybrid_bm25_cache_list;
	pgturbohybrid_bm25_cache_list = cache;

	MemoryContextSwitchTo(oldCtx);
	return cache;
}

static void
PgturbohybridBm25EnsureDocStats(Relation index, PgturbohybridBm25Cache *cache,
						   const PgturbohybridBm25MetaTupleData *bm25Meta,
						   const PgturbohybridGraphMetaPageData *graphMeta)
{
	MemoryContext oldCtx;

	if (cache->docStatsLoaded)
		return;

	oldCtx = MemoryContextSwitchTo(cache->ctx);
	cache->docLens = PgturbohybridBm25Palloc0Array(sizeof(uint32),
												   Max(cache->nodeCount, 1));
	PgturbohybridBm25LoadDocStats(index, bm25Meta, graphMeta->tqNodeCount,
							 cache->docLens);
	cache->docStatsLoaded = true;
	MemoryContextSwitchTo(oldCtx);
}

static void
PgturbohybridBm25EnsureLiveness(Relation index, PgturbohybridBm25Cache *cache,
						   const PgturbohybridGraphMetaPageData *graphMeta)
{
	MemoryContext oldCtx;

	if (cache->livenessLoaded)
		return;

	oldCtx = MemoryContextSwitchTo(cache->ctx);
	cache->heapTids = PgturbohybridBm25Palloc0Array(sizeof(ItemPointerData),
													Max(cache->nodeCount, 1));
	cache->liveNodes = PgturbohybridBm25Palloc0Array(sizeof(bool),
													 Max(cache->nodeCount, 1));
	PgturbohybridBm25LoadHeapTids(index, graphMeta, cache->heapTids,
							 cache->liveNodes);
	cache->livenessLoaded = true;
	MemoryContextSwitchTo(oldCtx);
}

static PgturbohybridBm25Cache *
PgturbohybridBm25GetCache(Relation index,
					 const PgturbohybridBm25MetaTupleData *bm25Meta,
					 const PgturbohybridGraphMetaPageData *graphMeta,
					 PgturbohybridBm25QueryStats *stats)
{
	PgturbohybridBm25Cache *cache;
	instr_time	start;

	PgturbohybridBm25DropStaleCaches(index, bm25Meta, graphMeta);
	for (cache = pgturbohybrid_bm25_cache_list; cache != NULL; cache = cache->next)
	{
		if (PgturbohybridBm25CacheMatches(cache, index, bm25Meta, graphMeta))
		{
			if (stats != NULL)
			{
				stats->cacheHit = true;
				stats->cacheDocstatsLoaded = cache->docStatsLoaded;
				stats->cacheLivenessLoaded = cache->livenessLoaded;
			}
			return cache;
		}
	}

	INSTR_TIME_SET_CURRENT(start);
	cache = PgturbohybridBm25BuildCache(index, bm25Meta, graphMeta);
	if (stats != NULL)
	{
		stats->cacheBuildUs = PgturbohybridBm25ElapsedUs(start);
		stats->cacheDocstatsLoaded = cache->docStatsLoaded;
		stats->cacheLivenessLoaded = cache->livenessLoaded;
	}
	return cache;
}

static bool
PgturbohybridBm25CacheFindLexiconEntry(PgturbohybridBm25Cache *cache,
								  PgturbohybridBm25QueryTerm *term,
								  PgturbohybridBm25LexiconEntryData *entry)
{
	uint64		termHash = term->termHash;
	int			lo = 0;
	int			hi = (int) cache->lexiconCount - 1;

	while (lo <= hi)
	{
		int			mid = lo + (hi - lo) / 2;
		PgturbohybridBm25CacheLexiconEntry *candidate = &cache->lexicon[mid];
		int			cmp;

		if (candidate->termHash != termHash)
			cmp = candidate->termHash < termHash ? -1 : 1;
		else if (candidate->termLen != term->termLen)
			cmp = candidate->termLen < term->termLen ? -1 : 1;
		else
			cmp = memcmp(candidate->termBytes, term->term, term->termLen);

		if (cmp == 0)
		{
			memset(entry, 0, sizeof(*entry));
			entry->type = PGTURBOHYBRID_BM25_LEXICON_TUPLE_TYPE;
			entry->termLen = candidate->termLen;
			entry->termHash = candidate->termHash;
			entry->termId = candidate->termId;
			entry->df = candidate->df;
			entry->cf = candidate->cf;
			entry->postingsBlkno = candidate->postingsBlkno;
			entry->postingsOffno = candidate->postingsOffno;
			entry->postingsChunkCount = candidate->postingsChunkCount;
			entry->postingsBytes = candidate->postingsBytes;
			entry->blockMaxBlkno = candidate->blockMaxBlkno;
			entry->blockMaxOffno = candidate->blockMaxOffno;
			entry->impactBlkno = candidate->impactBlkno;
			entry->impactOffno = candidate->impactOffno;
			entry->impactCount = candidate->storedImpactCount;
			term->cacheEntry = candidate;
			return true;
		}
		if (cmp < 0)
			lo = mid + 1;
		else
			hi = mid - 1;
	}

	return false;
}

static PgturbohybridBm25DeltaCacheEntry *
PgturbohybridBm25FindDeltaEntry(PgturbohybridBm25DeltaCacheEntry *deltaTerms,
						   uint32 deltaTermCount,
						   PgturbohybridBm25QueryTerm *term)
{
	uint64		termHash = term->termHash;
	int			lo = 0;
	int			hi = (int) deltaTermCount - 1;

	while (lo <= hi)
	{
		int			mid = lo + (hi - lo) / 2;
		PgturbohybridBm25DeltaCacheEntry *candidate = &deltaTerms[mid];
		int			cmp;

		if (candidate->termHash != termHash)
			cmp = candidate->termHash < termHash ? -1 : 1;
		else if (candidate->termLen != term->termLen)
			cmp = candidate->termLen < term->termLen ? -1 : 1;
		else
			cmp = memcmp(candidate->termBytes, term->term, term->termLen);

		if (cmp == 0)
			return candidate;
		if (cmp < 0)
			lo = mid + 1;
		else
			hi = mid - 1;
	}

	return NULL;
}

static bool
PgturbohybridBm25DeltaBuildSameTerm(PgturbohybridBm25DeltaBuildEntry *left,
							   PgturbohybridBm25DeltaBuildEntry *right)
{
	return left->termHash == right->termHash &&
		left->termLen == right->termLen &&
		memcmp(left->termBytes, right->termBytes, left->termLen) == 0;
}

static uint64
PgturbohybridBm25FinalizeDeltaBuildEntries(PgturbohybridBm25DeltaBuildEntry *buildEntries,
									  uint32 buildCount,
									  PgturbohybridBm25DeltaCacheEntry **outTerms,
									  uint32 *outTermCount,
									  uint32 *outPostingCount)
{
	uint64		finalBytes = 0;

	if (buildCount > 0)
	{
		uint32		termCount = 0;
		uint32		outTerm = 0;

		qsort(buildEntries, buildCount,
			  sizeof(PgturbohybridBm25DeltaBuildEntry),
			  PgturbohybridBm25DeltaBuildCompare);
		for (uint32 i = 0; i < buildCount;)
		{
			uint32		j = i + 1;

			while (j < buildCount &&
				   PgturbohybridBm25DeltaBuildSameTerm(&buildEntries[i],
												  &buildEntries[j]))
				j++;
			termCount++;
			i = j;
		}

		*outTerms =
			PgturbohybridBm25Palloc0Array(sizeof(PgturbohybridBm25DeltaCacheEntry),
										  termCount);
		finalBytes +=
			PgturbohybridBm25ArrayAllocSize(sizeof(PgturbohybridBm25DeltaCacheEntry),
											termCount);
		for (uint32 i = 0; i < buildCount;)
		{
			uint32		j = i + 1;
			PgturbohybridBm25DeltaCacheEntry *termEntry;

			while (j < buildCount &&
				   PgturbohybridBm25DeltaBuildSameTerm(&buildEntries[i],
												  &buildEntries[j]))
				j++;

			termEntry = &(*outTerms)[outTerm++];
			termEntry->termLen = buildEntries[i].termLen;
			termEntry->termHash = buildEntries[i].termHash;
			termEntry->df = j - i;
			termEntry->postingCount = j - i;
			termEntry->termBytes = palloc(termEntry->termLen);
			finalBytes += termEntry->termLen;
			memcpy(termEntry->termBytes, buildEntries[i].termBytes,
				   termEntry->termLen);
			termEntry->postings =
				PgturbohybridBm25Palloc0Array(sizeof(PgturbohybridBm25DeltaCachePosting),
											  termEntry->postingCount);
			finalBytes +=
				PgturbohybridBm25ArrayAllocSize(sizeof(PgturbohybridBm25DeltaCachePosting),
												termEntry->postingCount);
			for (uint32 k = i; k < j; k++)
			{
				PgturbohybridBm25DeltaCachePosting *posting =
					&termEntry->postings[k - i];

				posting->nodeId = buildEntries[k].nodeId;
				posting->tf = buildEntries[k].tf;
				posting->docLen = buildEntries[k].docLen;
				posting->heaptid = buildEntries[k].heaptid;
				pfree(buildEntries[k].termBytes);
			}
			*outPostingCount += termEntry->postingCount;
			i = j;
		}
		*outTermCount = outTerm;
		qsort(*outTerms, *outTermCount,
			  sizeof(PgturbohybridBm25DeltaCacheEntry),
			  PgturbohybridBm25DeltaCacheCompare);
	}

	return finalBytes;
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

static bool
PgturbohybridBm25BuildDeltaCacheFromTermSegments(Relation index,
											const PgturbohybridBm25MetaTupleData *meta,
											PgturbohybridBm25Cache *cache,
											PgturbohybridBm25QueryTerm *filterTerms,
											int filterTermCount,
											PgturbohybridBm25DeltaBuildEntry **outBuildEntries,
											uint32 *outBuildCount,
											uint32 *outBuildCapacity,
											uint64 *outTermPagesRead)
{
	PgturbohybridBm25DeltaDirectoryTupleData directory;
	bool		bucketNeeded[PGTURBOHYBRID_BM25_DELTA_TERM_BUCKETS] = {false};

	*outTermPagesRead = 0;
	if (filterTerms == NULL || filterTermCount <= 0 ||
		!PgturbohybridBm25ReadDeltaDirectory(index, meta->deltaTermDirectoryBlkno,
										&directory))
		return false;

	for (int i = 0; i < filterTermCount; i++)
		bucketNeeded[PgturbohybridBm25DeltaTermBucketNo(filterTerms[i].termHash)] = true;

	for (uint16 bucketNo = 0; bucketNo < directory.bucketCount; bucketNo++)
	{
		BlockNumber blkno;

		if (!bucketNeeded[bucketNo])
			continue;

		blkno = directory.buckets[bucketNo].startBlkno;
		while (BlockNumberIsValid(blkno))
		{
			Buffer		buf;
			Page		page;
			PgturbohybridGraphPageOpaque opaque;
			OffsetNumber maxoff;
			BlockNumber nextblkno;

			(*outTermPagesRead)++;
			buf = ReadBuffer(index, blkno);
			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);
			opaque = PgturbohybridGraphPageGetOpaque(page);
			if (!PgturbohybridBm25PageIsKind(page, PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_DELTA_TERM))
			{
				UnlockReleaseBuffer(buf);
				ereport(ERROR,
						(errcode(ERRCODE_DATA_CORRUPTED),
						 errmsg("pgturbohybrid BM25 delta term page has unexpected page kind")));
			}
			nextblkno = opaque->nextblkno;
			maxoff = PageGetMaxOffsetNumber(page);

			for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off = OffsetNumberNext(off))
			{
				ItemId		iid = PageGetItemId(page, off);
				PgturbohybridBm25DeltaTermTuple tuple;
				char	   *termBytes;

				if (!ItemIdIsUsed(iid))
					continue;
				tuple = (PgturbohybridBm25DeltaTermTuple) PageGetItem(page, iid);
				if (tuple->type != PGTURBOHYBRID_BM25_DELTA_TERM_TUPLE_TYPE)
					continue;
				if (tuple->termBytesLen < tuple->termLen)
					continue;

				termBytes = PgturbohybridBm25DeltaTermTupleBytes(tuple);
				if (PgturbohybridBm25FindQueryTerm(filterTerms, filterTermCount,
											  termBytes, tuple->termLen) < 0)
					continue;

				for (uint16 i = 0; i < tuple->postingCount; i++)
				{
					PgturbohybridBm25DeltaTermPosting *src = &tuple->postings[i];
					PgturbohybridBm25DeltaBuildEntry *entry;

					if (src->nodeId >= cache->nodeCount ||
						!cache->liveNodes[src->nodeId])
						continue;
					if (*outBuildCount >= *outBuildCapacity)
					{
						*outBuildCapacity =
							PgturbohybridBm25GrowCapacity32(*outBuildCapacity,
															*outBuildCount + 1,
															sizeof(PgturbohybridBm25DeltaBuildEntry));
						*outBuildEntries =
							PgturbohybridBm25RepallocArray(*outBuildEntries,
														   sizeof(PgturbohybridBm25DeltaBuildEntry),
														   *outBuildCapacity);
					}
					entry = &(*outBuildEntries)[(*outBuildCount)++];
					entry->termLen = tuple->termLen;
					entry->termHash = tuple->termHash;
					entry->nodeId = src->nodeId;
					entry->tf = src->tf;
					entry->docLen = src->docLen;
					entry->heaptid = src->heaptid;
					entry->termBytes = palloc(tuple->termLen);
					memcpy(entry->termBytes, termBytes, tuple->termLen);
				}
			}

			UnlockReleaseBuffer(buf);
			blkno = nextblkno;
			CHECK_FOR_INTERRUPTS();
		}
	}

	return true;
}

static void
PgturbohybridBm25BuildDeltaCacheEntries(Relation index,
								   const PgturbohybridBm25MetaTupleData *meta,
								   PgturbohybridBm25Cache *cache,
								   PgturbohybridBm25QueryTerm *filterTerms,
								   int filterTermCount,
								   MemoryContext memoryContext,
								   PgturbohybridBm25DeltaCacheEntry **outTerms,
								   uint32 *outTermCount,
								   uint32 *outPostingCount,
								   uint64 *outBytes,
								   PgturbohybridBm25QueryStats *stats)
{
	MemoryContext oldCtx;
	BlockNumber blkno = meta->deltaStartBlkno;
	PgturbohybridBm25DeltaBuildEntry *buildEntries = NULL;
	uint32		buildCount = 0;
	uint32		buildCapacity = filterTerms == NULL ?
		Max(meta->deltaTermCount, 1) : Max((uint32) filterTermCount, 1);
	uint64		blocksVisited = 0;
	uint64		termPagesRead = 0;
	uint64		finalBytes = 0;
	bool		usedTermSegments = false;

	*outTerms = NULL;
	*outTermCount = 0;
	*outPostingCount = 0;
	*outBytes = 0;
	oldCtx = MemoryContextSwitchTo(memoryContext);
	buildEntries =
		PgturbohybridBm25Palloc0Array(sizeof(PgturbohybridBm25DeltaBuildEntry),
									  buildCapacity);

	if (filterTerms != NULL && filterTermCount > 0 &&
		BlockNumberIsValid(meta->deltaTermDirectoryBlkno))
		usedTermSegments =
			PgturbohybridBm25BuildDeltaCacheFromTermSegments(index, meta, cache,
														filterTerms,
														filterTermCount,
														&buildEntries,
														&buildCount,
														&buildCapacity,
														&termPagesRead);

	while (!usedTermSegments && BlockNumberIsValid(blkno))
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		OffsetNumber maxoff;
		BlockNumber nextblkno;

		blocksVisited++;
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
				tuple->nodeId >= cache->nodeCount ||
				!cache->liveNodes[tuple->nodeId])
				continue;

			termBytes = PgturbohybridBm25DeltaTermBytes(tuple);
			for (uint16 i = 0; i < tuple->termCount; i++)
			{
				PgturbohybridBm25DeltaTerm *term = &tuple->terms[i];
				PgturbohybridBm25DeltaBuildEntry *entry;

				if (term->termOffset + term->termLen > tuple->termBytesLen)
					continue;
				if (filterTerms != NULL &&
					PgturbohybridBm25FindQueryTerm(filterTerms, filterTermCount,
											  termBytes + term->termOffset,
											  term->termLen) < 0)
					continue;
				if (buildCount >= buildCapacity)
				{
					buildCapacity =
						PgturbohybridBm25GrowCapacity32(buildCapacity,
														buildCount + 1,
														sizeof(PgturbohybridBm25DeltaBuildEntry));
					buildEntries =
						PgturbohybridBm25RepallocArray(buildEntries,
													   sizeof(PgturbohybridBm25DeltaBuildEntry),
													   buildCapacity);
				}
				entry = &buildEntries[buildCount++];
				entry->termLen = term->termLen;
				entry->termHash = term->termHash;
				entry->nodeId = tuple->nodeId;
				entry->tf = term->tf;
				entry->docLen = tuple->docLen;
				entry->heaptid = tuple->heaptid;
				entry->termBytes = palloc(term->termLen);
				memcpy(entry->termBytes, termBytes + term->termOffset,
					   term->termLen);
			}
		}

		UnlockReleaseBuffer(buf);
		blkno = nextblkno;
		CHECK_FOR_INTERRUPTS();
	}

	finalBytes =
		PgturbohybridBm25FinalizeDeltaBuildEntries(buildEntries, buildCount,
											  outTerms, outTermCount,
											  outPostingCount);
	pfree(buildEntries);
	*outBytes = finalBytes;

	if (stats != NULL)
	{
		stats->blocksVisited += blocksVisited + termPagesRead;
		stats->deltaBlocksVisited += blocksVisited + termPagesRead;
		stats->deltaPagesScanned += blocksVisited;
		stats->deltaTermPagesRead += termPagesRead;
		stats->deltaLookupMode = usedTermSegments ?
			PGTURBOHYBRID_BM25_DELTA_LOOKUP_TERM_SEGMENT :
			(BlockNumberIsValid(meta->deltaStartBlkno) ?
			 PGTURBOHYBRID_BM25_DELTA_LOOKUP_DOC_SCAN :
			 PGTURBOHYBRID_BM25_DELTA_LOOKUP_NONE);
	}
	MemoryContextSwitchTo(oldCtx);
}

static void
PgturbohybridBm25EnsureDeltaCache(Relation index,
							 const PgturbohybridBm25MetaTupleData *meta,
							 PgturbohybridBm25Cache *cache,
							 PgturbohybridBm25QueryStats *stats)
{
	if (cache->deltaCacheBuilt)
	{
		if (stats != NULL)
			stats->deltaCacheHit = true;
		return;
	}

	PgturbohybridBm25BuildDeltaCacheEntries(index, meta, cache, NULL, 0,
									   cache->ctx, &cache->deltaTerms,
									   &cache->deltaTermCount,
									   &cache->deltaPostingCount,
									   &cache->deltaCacheBytes, stats);
	cache->deltaCacheBuilt = true;
	cache->deltaCacheBytes = MemoryContextMemAllocated(cache->ctx, true);
}

static int
PgturbohybridBm25FindQueryTerm(PgturbohybridBm25QueryTerm *terms, int termCount,
						  const char *term, uint16 termLen)
{
	uint64		termHash = PgturbohybridBm25HashTerm(term, termLen);

	for (int i = 0; i < termCount; i++)
	{
		if (terms[i].termLen == termLen &&
			terms[i].termHash == termHash &&
			memcmp(terms[i].term, term, termLen) == 0)
			return i;
	}

	return -1;
}

static int
PgturbohybridBm25FindQueryTermByItem(PgturbohybridBm25QueryTerm *terms, int termCount,
								int qitemIndex)
{
	for (int i = 0; i < termCount; i++)
	{
		if (terms[i].qitemIndex == qitemIndex)
			return i;
	}

	return -1;
}

static bool
PgturbohybridBm25EvalTsQueryItem(QueryItem *items, QueryItem *item, char *operands,
							PgturbohybridBm25QueryTerm *terms, int termCount,
							uint64 matchedTerms)
{
	if (item->type == QI_VAL)
	{
		char	   *term = operands + item->qoperand.distance;
		int			termNo;

		termNo = PgturbohybridBm25FindQueryTermByItem(terms, termCount,
												 (int) (item - items));
		if (termNo < 0)
			termNo = PgturbohybridBm25FindQueryTerm(terms, termCount, term,
											   item->qoperand.length);
		return termNo >= 0 && (matchedTerms & terms[termNo].matchBit) != 0;
	}

	if (item->type == QI_OPR)
	{
		QueryItem  *right = item + 1;
		QueryItem  *left = item + item->qoperator.left;

		if (item->qoperator.oper == OP_AND)
			return PgturbohybridBm25EvalTsQueryItem(items, left, operands, terms,
											   termCount, matchedTerms) &&
				PgturbohybridBm25EvalTsQueryItem(items, right, operands, terms,
											termCount, matchedTerms);
		if (item->qoperator.oper == OP_OR)
			return PgturbohybridBm25EvalTsQueryItem(items, left, operands, terms,
											   termCount, matchedTerms) ||
				PgturbohybridBm25EvalTsQueryItem(items, right, operands, terms,
											termCount, matchedTerms);
	}

	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("pgturbohybrid BM25 supports OR/AND tsquery terms only")));
}

static bool
PgturbohybridBm25MatchedQuery(TSQuery query, PgturbohybridBm25QueryTerm *terms,
						 int termCount, const PgturbohybridBm25QueryEval *eval,
						 uint64 matchedTerms, PgturbohybridBm25QueryStats *stats)
{
	if (stats != NULL)
		stats->booleanEvalCalls++;
	if (query->size == 0)
		return false;
	if (eval != NULL)
	{
		if (eval->mode == PGTURBOHYBRID_BM25_BOOLEAN_EVAL_SINGLE)
			return (matchedTerms & eval->requiredMask) != 0;
		if (eval->mode == PGTURBOHYBRID_BM25_BOOLEAN_EVAL_MASK)
		{
			if (eval->shape == PGTURBOHYBRID_BM25_QUERY_PURE_OR)
				return (matchedTerms & eval->requiredMask) != 0;
			if (eval->shape == PGTURBOHYBRID_BM25_QUERY_PURE_AND)
				return (matchedTerms & eval->requiredMask) == eval->requiredMask;
		}
	}
	return PgturbohybridBm25EvalTsQueryItem(GETQUERY(query), GETQUERY(query),
									   GETOPERAND(query), terms, termCount,
									   matchedTerms);
}

static bool
PgturbohybridBm25QueryHasOperator(TSQuery query, int oper)
{
	QueryItem  *items = GETQUERY(query);

	for (int i = 0; i < query->size; i++)
	{
		if (items[i].type == QI_OPR && items[i].qoperator.oper == oper)
			return true;
	}
	return false;
}

static PgturbohybridBm25QueryEval
PgturbohybridBm25CompileQueryEval(TSQuery query, PgturbohybridBm25QueryTerm *terms,
							 int termCount)
{
	PgturbohybridBm25QueryEval eval;
	bool		hasOr = PgturbohybridBm25QueryHasOperator(query, OP_OR);
	bool		hasAnd = PgturbohybridBm25QueryHasOperator(query, OP_AND);

	memset(&eval, 0, sizeof(eval));
	for (int i = 0; i < termCount; i++)
		eval.requiredMask |= terms[i].matchBit;

	if (termCount == 0)
	{
		eval.shape = PGTURBOHYBRID_BM25_QUERY_EMPTY;
		eval.mode = PGTURBOHYBRID_BM25_BOOLEAN_EVAL_NONE;
	}
	else if (termCount == 1 && !hasOr && !hasAnd)
	{
		eval.shape = PGTURBOHYBRID_BM25_QUERY_SINGLE;
		eval.mode = PGTURBOHYBRID_BM25_BOOLEAN_EVAL_SINGLE;
	}
	else if (hasOr && !hasAnd)
	{
		eval.shape = PGTURBOHYBRID_BM25_QUERY_PURE_OR;
		eval.mode = PGTURBOHYBRID_BM25_BOOLEAN_EVAL_MASK;
	}
	else if (hasAnd && !hasOr)
	{
		eval.shape = PGTURBOHYBRID_BM25_QUERY_PURE_AND;
		eval.mode = PGTURBOHYBRID_BM25_BOOLEAN_EVAL_MASK;
	}
	else
	{
		eval.shape = PGTURBOHYBRID_BM25_QUERY_MIXED;
		eval.mode = PGTURBOHYBRID_BM25_BOOLEAN_EVAL_TREE;
	}

	return eval;
}

static void
PgturbohybridBm25CountDeltaDf(PgturbohybridBm25DeltaCacheEntry *deltaTerms,
						 uint32 deltaTermCount,
						 PgturbohybridBm25QueryTerm *terms, int termCount)
{
	for (int termNo = 0; termNo < termCount; termNo++)
	{
		PgturbohybridBm25DeltaCacheEntry *entry;

		entry = PgturbohybridBm25FindDeltaEntry(deltaTerms, deltaTermCount,
										   &terms[termNo]);
		if (entry != NULL)
			terms[termNo].deltaDf = entry->df;
	}
}

static int
PgturbohybridBm25ChooseAccumulatorMode(const PgturbohybridBm25MetaTupleData *meta,
								  PgturbohybridBm25QueryTerm *terms, int termCount)
{
	uint64		sumDf = 0;
	uint32		maxDf = 0;
	double		docCount = Max((double) meta->docCount +
							   (double) meta->deltaDocCount, 1.0);

	if (pgturbohybrid_bm25_accumulator_mode ==
		PGTURBOHYBRID_BM25_ACCUMULATOR_HASH)
		return PGTURBOHYBRID_BM25_ACCUMULATOR_HASH;
	if (pgturbohybrid_bm25_accumulator_mode ==
		PGTURBOHYBRID_BM25_ACCUMULATOR_DENSE)
		return PGTURBOHYBRID_BM25_ACCUMULATOR_DENSE;

	for (int termNo = 0; termNo < termCount; termNo++)
	{
		uint32		df = terms[termNo].baseDf + terms[termNo].deltaDf;

		sumDf += df;
		maxDf = Max(maxDf, df);
	}
	if (sumDf > (uint64) pgturbohybrid_bm25_dense_accumulator_threshold ||
		((double) maxDf / docCount) > pgturbohybrid_bm25_dense_accumulator_df_ratio)
		return PGTURBOHYBRID_BM25_ACCUMULATOR_DENSE;
	return PGTURBOHYBRID_BM25_ACCUMULATOR_HASH;
}

static void
PgturbohybridBm25ScoreDelta(const PgturbohybridBm25MetaTupleData *meta,
					   PgturbohybridBm25DeltaCacheEntry *deltaTerms,
					   uint32 deltaTermCount,
					   PgturbohybridBm25QueryTerm *terms, int termCount,
					   PgturbohybridBm25Accumulator *acc,
					   PgturbohybridBm25QueryStats *stats)
{
	double		corpusDocCount = Max((double) meta->docCount +
									 (double) meta->deltaDocCount, 1.0);
	double		avgDocLen = ((double) meta->totalDocLen +
							 (double) meta->deltaTotalDocLen) / corpusDocCount;

	for (int termNo = 0; termNo < termCount; termNo++)
	{
		PgturbohybridBm25QueryTerm *term = &terms[termNo];
		PgturbohybridBm25DeltaCacheEntry *deltaEntry;
		uint32		df;
		double		idf;

		deltaEntry = PgturbohybridBm25FindDeltaEntry(deltaTerms, deltaTermCount,
												term);
		if (deltaEntry == NULL)
			continue;

		df = term->baseDf + term->deltaDf;
		idf = log(1.0 + (corpusDocCount - (double) df + 0.5) /
				  ((double) df + 0.5));
		for (uint32 i = 0; i < deltaEntry->postingCount; i++)
		{
			PgturbohybridBm25DeltaCachePosting *posting = &deltaEntry->postings[i];
			double		dl;
			double		norm;
			double		tf = posting->tf;
			PgturbohybridBm25AccumulatorEntry *entry;

			dl = Max((double) posting->docLen, 1.0);
				norm = (double) meta->k1 *
					(1.0 - (double) meta->b +
					 (double) meta->b * dl / Max(avgDocLen, 1.0));
				entry = PgturbohybridBm25AccumulatorLookup(acc, posting->nodeId, true);
				if (entry == NULL)
					continue;
				entry->docLen = posting->docLen;
				entry->heaptid = posting->heaptid;
				entry->hasDeltaDoc = true;
				entry->score += idf *
					((tf * ((double) meta->k1 + 1.0)) / (tf + norm));
				entry->matchedTerms |= term->matchBit;
				if (stats != NULL &&
					acc->mode == PGTURBOHYBRID_BM25_ACCUMULATOR_DENSE)
					stats->accumulatorDenseUpdates++;
				PgturbohybridBm25AccumulatorUpdateTopK(acc, entry);
			if (stats != NULL)
			{
				stats->postingsDecoded++;
				stats->deltaPostingsDecoded++;
			}

			CHECK_FOR_INTERRUPTS();
		}
	}
}

static int
PgturbohybridBm25ExtractTerms(TSQuery query, PgturbohybridBm25QueryTerm **terms,
						 MemoryContext memoryContext)
{
	QueryItem  *items = GETQUERY(query);
	char	   *operands = GETOPERAND(query);
	int			count = 0;
	PgturbohybridBm25QueryTerm *out;

	for (int i = 0; i < query->size; i++)
	{
		QueryItem  *item = &items[i];

		if (item->type == QI_VAL)
		{
			if (item->qoperand.prefix)
				ereport(ERROR,
						(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
						 errmsg("pgturbohybrid BM25 prefix tsquery terms are not supported yet")));
			if (item->qoperand.weight != 0)
				ereport(ERROR,
						(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
						 errmsg("pgturbohybrid BM25 weighted tsquery terms are not supported yet")));
			count++;
		}
		else if (item->type == QI_OPR &&
				 (item->qoperator.oper == OP_NOT ||
				  item->qoperator.oper == OP_PHRASE))
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("pgturbohybrid BM25 supports OR/AND tsquery terms only")));
	}

	out = PgturbohybridBm25MemoryContextAllocZeroArray(memoryContext,
													   sizeof(PgturbohybridBm25QueryTerm),
													   Max(count, 1));
	count = 0;
	for (int i = 0; i < query->size; i++)
	{
		QueryItem  *item = &items[i];
		char	   *term;

		if (item->type != QI_VAL)
			continue;

		term = operands + item->qoperand.distance;
		if (PgturbohybridBm25FindQueryTerm(out, count, term,
									  item->qoperand.length) >= 0)
			continue;
		if (count >= 64)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("turbohybrid BM25 queries currently support at most 64 terms")));
		out[count].term = term;
		out[count].termLen = item->qoperand.length;
		out[count].termHash = PgturbohybridBm25HashTerm(term,
												   item->qoperand.length);
		out[count].matchBit = UINT64CONST(1) << count;
		out[count].qitemIndex = i;
		count++;
	}

	*terms = out;
	return count;
}

static int
PgturbohybridBm25ScoreCompare(const void *a, const void *b)
{
	const PgturbohybridBm25NodeScore *sa = (const PgturbohybridBm25NodeScore *) a;
	const PgturbohybridBm25NodeScore *sb = (const PgturbohybridBm25NodeScore *) b;

	if (sa->score > sb->score)
		return -1;
	if (sa->score < sb->score)
		return 1;
	return (sa->nodeId > sb->nodeId) - (sa->nodeId < sb->nodeId);
}

static int
PgturbohybridBm25ImpactCompare(const void *a, const void *b)
{
	const PgturbohybridBm25ImpactEntry *ia = (const PgturbohybridBm25ImpactEntry *) a;
	const PgturbohybridBm25ImpactEntry *ib = (const PgturbohybridBm25ImpactEntry *) b;

	if (ia->score > ib->score)
		return -1;
	if (ia->score < ib->score)
		return 1;
	return (ia->nodeId > ib->nodeId) - (ia->nodeId < ib->nodeId);
}

static bool
PgturbohybridBm25NodeScoreHeapLess(PgturbohybridBm25NodeScore a,
							  PgturbohybridBm25NodeScore b)
{
	if (a.score < b.score)
		return true;
	if (a.score > b.score)
		return false;
	return a.nodeId > b.nodeId;
}

static void
PgturbohybridBm25NodeScoreHeapSwap(PgturbohybridBm25NodeScore *heap, uint32 a, uint32 b)
{
	PgturbohybridBm25NodeScore tmp = heap[a];

	heap[a] = heap[b];
	heap[b] = tmp;
}

static void
PgturbohybridBm25NodeScoreHeapSiftUp(PgturbohybridBm25NodeScore *heap, uint32 pos)
{
	while (pos > 0)
	{
		uint32		parent = (pos - 1) / 2;

		if (!PgturbohybridBm25NodeScoreHeapLess(heap[pos], heap[parent]))
			break;
		PgturbohybridBm25NodeScoreHeapSwap(heap, pos, parent);
		pos = parent;
	}
}

static void
PgturbohybridBm25NodeScoreHeapSiftDown(PgturbohybridBm25NodeScore *heap, uint32 count,
								  uint32 pos)
{
	for (;;)
	{
		uint32		left = pos * 2 + 1;
		uint32		right = left + 1;
		uint32		smallest = pos;

		if (left < count &&
			PgturbohybridBm25NodeScoreHeapLess(heap[left], heap[smallest]))
			smallest = left;
		if (right < count &&
			PgturbohybridBm25NodeScoreHeapLess(heap[right], heap[smallest]))
			smallest = right;
		if (smallest == pos)
			break;
		PgturbohybridBm25NodeScoreHeapSwap(heap, pos, smallest);
		pos = smallest;
	}
}

static uint32
PgturbohybridBm25SelectFinalTopK(PgturbohybridBm25Accumulator *acc, TSQuery query,
							PgturbohybridBm25QueryTerm *terms, int termCount,
							const PgturbohybridBm25QueryEval *eval, uint32 k,
							MemoryContext memoryContext)
{
	PgturbohybridBm25NodeScore *heap;
	uint32		heapCount = 0;
	uint32		oldTouchedCount = acc->touchedCount;

	if (k == 0)
	{
		acc->touchedCount = 0;
		return 0;
	}

	if (pgturbohybrid_bm25_force_full_sort)
	{
		uint32		out = 0;

		for (uint32 i = 0; i < oldTouchedCount; i++)
		{
			PgturbohybridBm25AccumulatorEntry *entry;

			entry = PgturbohybridBm25AccumulatorLookup(acc, acc->touched[i].nodeId,
												  false);
			if (entry == NULL ||
				!PgturbohybridBm25MatchedQuery(query, terms, termCount,
										  eval, entry->matchedTerms,
										  acc->stats))
				continue;
			acc->touched[out] = acc->touched[i];
			acc->touched[out].score = entry->score;
			out++;
		}
		acc->touchedCount = out;
		if (acc->touchedCount > 1)
			qsort(acc->touched, acc->touchedCount,
				  sizeof(PgturbohybridBm25NodeScore), PgturbohybridBm25ScoreCompare);
		if (acc->stats != NULL)
		{
			acc->stats->finalSortedCount = acc->touchedCount;
			acc->stats->fullSortAvoided = false;
		}
		return Min(k, acc->touchedCount);
	}

	heap = PgturbohybridBm25MemoryContextAllocArray(memoryContext,
													sizeof(PgturbohybridBm25NodeScore),
													k);
	for (uint32 i = 0; i < oldTouchedCount; i++)
	{
		PgturbohybridBm25AccumulatorEntry *entry;
		PgturbohybridBm25NodeScore candidate;

		entry = PgturbohybridBm25AccumulatorLookup(acc, acc->touched[i].nodeId,
											  false);
		if (entry == NULL ||
			!PgturbohybridBm25MatchedQuery(query, terms, termCount,
									  eval, entry->matchedTerms, acc->stats))
			continue;

		candidate.nodeId = entry->nodeId;
		candidate.score = entry->score;
		if (heapCount < k)
		{
			heap[heapCount] = candidate;
			PgturbohybridBm25NodeScoreHeapSiftUp(heap, heapCount);
			heapCount++;
		}
		else if (PgturbohybridBm25ScoreCompare(&candidate, &heap[0]) < 0)
		{
			heap[0] = candidate;
			PgturbohybridBm25NodeScoreHeapSiftDown(heap, heapCount, 0);
			if (acc->stats != NULL)
				acc->stats->finalHeapReplacements++;
		}
	}

	if (heapCount > 1)
		qsort(heap, heapCount, sizeof(PgturbohybridBm25NodeScore),
			  PgturbohybridBm25ScoreCompare);
	for (uint32 i = 0; i < heapCount; i++)
		acc->touched[i] = heap[i];
	acc->touchedCount = heapCount;
	if (acc->stats != NULL)
	{
		acc->stats->finalSortedCount = heapCount;
		acc->stats->fullSortAvoided = true;
	}
	return heapCount;
}

static double
PgturbohybridBm25PostingScore(const PgturbohybridBm25MetaTupleData *meta, double idf,
						 double avgDocLen, uint16 tf, uint32 docLen)
{
	double		norm;

	norm = (double) meta->k1 *
		(1.0 - (double) meta->b +
		 (double) meta->b * Max((double) docLen, 1.0) / Max(avgDocLen, 1.0));
	return idf * (((double) tf * ((double) meta->k1 + 1.0)) /
				  ((double) tf + norm));
}

static double
PgturbohybridBm25PostingPrecomputedScore(const PgturbohybridBm25MetaTupleData *meta,
									double idf, uint16 tfNormQ16)
{
	double		scale = Max((double) meta->k1 + 1.0, 1.0);

	return idf * scale * ((double) tfNormQ16 / (double) PG_UINT16_MAX);
}

static double
PgturbohybridBm25PostingScoreDecoded(const PgturbohybridBm25MetaTupleData *meta,
								double idf, double avgDocLen,
								const PgturbohybridBm25Posting *posting,
								uint32 docLen)
{
	if ((meta->reserved2 & PGTURBOHYBRID_BM25_META_FLAG_TFNORM_Q16) != 0)
		return PgturbohybridBm25PostingPrecomputedScore(meta, idf,
												   posting->reserved);
	return PgturbohybridBm25PostingScore(meta, idf, avgDocLen, posting->tf,
									docLen);
}

static bool
PgturbohybridBm25ReloptionImpactHead(Relation index, uint32 *minDf, uint32 *headK)
{
	PgturbohybridOptions *opts = (PgturbohybridOptions *) index->rd_options;

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

static bool
PgturbohybridBm25EnsureImpactHead(Relation index,
							 const PgturbohybridBm25MetaTupleData *meta,
							 const PgturbohybridGraphMetaPageData *graphMeta,
							 PgturbohybridBm25Cache *cache,
							 PgturbohybridBm25QueryTerm *term,
							 const uint32 *docLens,
							 const bool *liveNodes,
							 PgturbohybridBm25QueryStats *stats)
{
	PgturbohybridBm25CacheLexiconEntry *entry = term->cacheEntry;
	uint32		minDf;
	uint32		headK;
	double		corpusDocCount;
	double		avgDocLen;
	double		idf;
	PgturbohybridBm25ImpactEntry *impact;
	PgturbohybridBm25Posting *scratch = NULL;
	uint16		scratchCapacity = 0;
	uint32		impactCount = 0;
	BlockNumber postingsBlkno;
	OffsetNumber postingsOffno;
	MemoryContext oldCtx;

	if (entry == NULL || !PgturbohybridBm25ReloptionImpactHead(index, &minDf, &headK))
		return false;
	if (entry->impactBuilt)
		return entry->impactEligible && entry->impactCount > 0;
	if (term->baseDf < minDf || headK == 0)
	{
		entry->impactBuilt = true;
		entry->impactEligible = false;
		return false;
	}
	if (BlockNumberIsValid(entry->impactBlkno) &&
		OffsetNumberIsValid(entry->impactOffno) &&
		entry->storedImpactCount > 0)
	{
		uint32		readCount = Min((uint32) entry->storedImpactCount, headK);
		uint32		loaded = 0;
		BlockNumber blkno = entry->impactBlkno;
		OffsetNumber offno = entry->impactOffno;
		double		currentCorpusDocCount = Max((double) meta->docCount +
												 (double) meta->deltaDocCount,
												 1.0);
		uint32		currentDf = term->baseDf + term->deltaDf;
		double		currentIdf = log(1.0 +
									 (currentCorpusDocCount -
									  (double) currentDf + 0.5) /
									 ((double) currentDf + 0.5));

		oldCtx = MemoryContextSwitchTo(cache->ctx);
		entry->impactHead = MemoryContextAlloc(cache->ctx,
											   sizeof(PgturbohybridBm25ImpactEntry) *
											   Max(readCount, 1));
		MemoryContextSwitchTo(oldCtx);

		while (loaded < readCount &&
			   BlockNumberIsValid(blkno) &&
			   OffsetNumberIsValid(offno))
		{
			Buffer		buf;
			Page		page;
			ItemId		iid;
			PgturbohybridBm25ImpactTuple tuple;
			uint32		chunkCount;
			BlockNumber nextBlkno;
			OffsetNumber nextOffno;

			buf = ReadBuffer(index, blkno);
			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);
			if (!PgturbohybridBm25PageIsKind(page, PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_IMPACT) ||
				offno > PageGetMaxOffsetNumber(page))
			{
				UnlockReleaseBuffer(buf);
				break;
			}

			iid = PageGetItemId(page, offno);
			if (!ItemIdIsUsed(iid))
			{
				UnlockReleaseBuffer(buf);
				break;
			}

			tuple = (PgturbohybridBm25ImpactTuple) PageGetItem(page, iid);
			if (tuple->type != PGTURBOHYBRID_BM25_IMPACT_TUPLE_TYPE ||
				tuple->termId != term->lexicon.termId)
			{
				UnlockReleaseBuffer(buf);
				break;
			}

			chunkCount = Min((uint32) tuple->count, readCount - loaded);
			for (uint32 i = 0; i < chunkCount; i++)
			{
				entry->impactHead[loaded + i].nodeId = tuple->entries[i].nodeId;
				entry->impactHead[loaded + i].tfNormQ16 =
					tuple->entries[i].tfNormQ16;
				entry->impactHead[loaded + i].reserved = 0;
				entry->impactHead[loaded + i].score =
					(float4) PgturbohybridBm25PostingPrecomputedScore(meta,
																 currentIdf,
																 tuple->entries[i].tfNormQ16);
			}
			loaded += chunkCount;
			nextBlkno = tuple->nextBlkno;
			nextOffno = tuple->nextOffno;
			UnlockReleaseBuffer(buf);
			blkno = nextBlkno;
			offno = nextOffno;
		}

		entry->impactCount = loaded;
		entry->impactBuilt = true;
		entry->impactEligible = loaded > 0;
		if (stats != NULL && loaded > 0)
			stats->impactLoadedFromStorage = true;
		return entry->impactEligible;
	}

	if (!pgturbohybrid_bm25_allow_lazy_impact_build)
	{
		entry->impactBuilt = true;
		entry->impactEligible = false;
		entry->impactCount = 0;
		return false;
	}

	oldCtx = MemoryContextSwitchTo(cache->ctx);
	impact =
		PgturbohybridBm25MemoryContextAllocArray(cache->ctx,
												 sizeof(PgturbohybridBm25ImpactEntry),
												 Max(term->baseDf, 1));
	scratch =
		PgturbohybridBm25MemoryContextAllocArray(cache->ctx,
												 sizeof(PgturbohybridBm25Posting),
												 1);
	scratchCapacity = 1;
	MemoryContextSwitchTo(oldCtx);

	corpusDocCount = Max((double) meta->docCount +
						 (double) meta->deltaDocCount, 1.0);
	avgDocLen = ((double) meta->totalDocLen +
				 (double) meta->deltaTotalDocLen) / corpusDocCount;
	idf = log(1.0 + (corpusDocCount -
					 (double) (term->baseDf + term->deltaDf) + 0.5) /
			  ((double) (term->baseDf + term->deltaDf) + 0.5));
	postingsBlkno = term->lexicon.postingsBlkno;
	postingsOffno = term->lexicon.postingsOffno;

	for (uint32 chunkNo = 0;
		 chunkNo < term->lexicon.postingsChunkCount &&
		 BlockNumberIsValid(postingsBlkno) &&
		 OffsetNumberIsValid(postingsOffno);
		 chunkNo++)
	{
		Buffer		buf;
		Page		page;
		ItemId		iid;
		PgturbohybridBm25PostingsTuple postings;
		BlockNumber nextBlkno;
		OffsetNumber nextOffno;

		buf = ReadBuffer(index, postingsBlkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		if (!PgturbohybridBm25PageIsKind(page, PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_POSTINGS) ||
			postingsOffno > PageGetMaxOffsetNumber(page))
		{
			UnlockReleaseBuffer(buf);
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("pgturbohybrid BM25 postings pointer is invalid")));
		}

		iid = PageGetItemId(page, postingsOffno);
		if (!ItemIdIsUsed(iid))
		{
			UnlockReleaseBuffer(buf);
			break;
		}

		postings = (PgturbohybridBm25PostingsTuple) PageGetItem(page, iid);
		if (postings->type != PGTURBOHYBRID_BM25_POSTINGS_TUPLE_TYPE ||
			postings->termId != term->lexicon.termId)
		{
			UnlockReleaseBuffer(buf);
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("pgturbohybrid BM25 postings tuple is invalid")));
		}
		nextBlkno = postings->nextBlkno;
		nextOffno = postings->nextOffno;
		if (scratchCapacity < postings->count)
		{
			oldCtx = MemoryContextSwitchTo(cache->ctx);
			scratch =
				PgturbohybridBm25RepallocArray(scratch,
											   sizeof(PgturbohybridBm25Posting),
											   postings->count);
			scratchCapacity = postings->count;
			MemoryContextSwitchTo(oldCtx);
		}
		if (!PgturbohybridBm25DecodePostingsTuple(postings, ItemIdGetLength(iid),
											 scratch))
		{
			UnlockReleaseBuffer(buf);
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("pgturbohybrid BM25 postings payload is invalid")));
		}
		if (stats != NULL)
		{
			stats->impactBuiltLazily = true;
			stats->impactLazyPostingsScanned += postings->count;
		}
		for (uint16 i = 0; i < postings->count; i++)
		{
			uint32		nodeId = scratch[i].nodeId;

			if (nodeId >= graphMeta->tqNodeCount || docLens[nodeId] == 0 ||
				!liveNodes[nodeId])
				continue;
			impact[impactCount].nodeId = nodeId;
			impact[impactCount].tfNormQ16 = scratch[i].reserved;
			impact[impactCount].reserved = 0;
			impact[impactCount].score =
				(float4) PgturbohybridBm25PostingScoreDecoded(meta, idf, avgDocLen,
														 &scratch[i],
														 docLens[nodeId]);
			impactCount++;
		}

		UnlockReleaseBuffer(buf);
		postingsBlkno = nextBlkno;
		postingsOffno = nextOffno;
		CHECK_FOR_INTERRUPTS();
	}

	if (impactCount > 1)
		qsort(impact, impactCount, sizeof(PgturbohybridBm25ImpactEntry),
			  PgturbohybridBm25ImpactCompare);
	entry->impactHead = impact;
	entry->impactCount = Min(impactCount, headK);
	entry->impactEligible = entry->impactCount > 0;
	entry->impactBuilt = true;
	if (stats != NULL)
		stats->cacheBytes = MemoryContextMemAllocated(cache->ctx, true);
	return entry->impactEligible;
}

static bool
PgturbohybridBm25ScoreImpactSingle(Relation index,
							  const PgturbohybridBm25MetaTupleData *meta,
							  const PgturbohybridGraphMetaPageData *graphMeta,
							  PgturbohybridBm25Cache *cache,
							  PgturbohybridBm25QueryTerm *terms, int termCount,
							  const uint32 *docLens, const bool *liveNodes,
							  PgturbohybridBm25Accumulator *acc, int32 k,
							  PgturbohybridBm25QueryStats *stats)
{
	PgturbohybridBm25CacheLexiconEntry *entry;
	uint32		readCount;

	if (termCount != 1 || !terms[0].hasLexicon || k <= 0)
		return false;
	if (!PgturbohybridBm25EnsureImpactHead(index, meta, graphMeta, cache, &terms[0],
									  docLens, liveNodes, stats))
		return false;

	entry = terms[0].cacheEntry;
	readCount = Min(entry->impactCount, (uint32) k);
	if (readCount < (uint32) k)
		return false;
	for (uint32 i = 0; i < readCount; i++)
		PgturbohybridBm25AccumulatorAddTermScore(acc, entry->impactHead[i].nodeId,
											entry->impactHead[i].score,
											terms[0].matchBit);
	if (stats != NULL)
	{
		stats->strategy = PGTURBOHYBRID_BM25_RUNTIME_IMPACT_SINGLE;
		stats->impactTerms = 1;
		stats->impactPostingsRead = readCount;
		stats->impactFullPostingsAvoided = true;
		stats->candidatesScored = readCount;
	}
	return true;
}

static bool
PgturbohybridBm25SeedImpactHeads(Relation index,
							const PgturbohybridBm25MetaTupleData *meta,
							const PgturbohybridGraphMetaPageData *graphMeta,
							PgturbohybridBm25Cache *cache,
							PgturbohybridBm25QueryTerm *terms, int termCount,
							const uint32 *docLens, const bool *liveNodes,
							PgturbohybridBm25Accumulator *acc, int32 k,
							PgturbohybridBm25QueryStats *stats)
{
	bool		seeded = false;
	PgturbohybridBm25NodeScore *seeds;
	uint32		seedCount = 0;
	uint32		seedCapacity;

	if (termCount <= 1 || k <= 0)
		return false;
	seedCapacity = (uint32) k * (uint32) termCount;
	seeds = MemoryContextAllocZero(acc->memoryContext,
								   sizeof(PgturbohybridBm25NodeScore) *
								   Max(seedCapacity, 1));

	for (int termNo = 0; termNo < termCount; termNo++)
	{
		PgturbohybridBm25CacheLexiconEntry *entry;
		uint32		readCount;

		if (!terms[termNo].hasLexicon)
			continue;
		if (!PgturbohybridBm25EnsureImpactHead(index, meta, graphMeta, cache,
										  &terms[termNo], docLens, liveNodes,
										  stats))
			continue;

		entry = terms[termNo].cacheEntry;
		readCount = Min(entry->impactCount, (uint32) k);
		if (readCount == 0)
			continue;
		for (uint32 i = 0; i < readCount; i++)
		{
			uint32		nodeId = entry->impactHead[i].nodeId;
			bool		found = false;

			for (uint32 j = 0; j < seedCount; j++)
			{
				if (seeds[j].nodeId == nodeId)
				{
					seeds[j].score += entry->impactHead[i].score;
					found = true;
					break;
				}
			}
			if (!found && seedCount < seedCapacity)
			{
				seeds[seedCount].nodeId = nodeId;
				seeds[seedCount].score = entry->impactHead[i].score;
				seedCount++;
			}
		}
		seeded = true;
		if (stats != NULL)
		{
			stats->impactTerms++;
			stats->impactPostingsRead += readCount;
		}
	}

	if (seedCount >= (uint32) k)
	{
		qsort(seeds, seedCount, sizeof(PgturbohybridBm25NodeScore),
			  PgturbohybridBm25ScoreCompare);
		acc->seedThreshold = seeds[k - 1].score;
	}
	return seeded;
}

static double PgturbohybridBm25KthScore(PgturbohybridBm25Accumulator *acc, int32 k);

static bool
PgturbohybridBm25ImpactORCanBeExact(Relation index,
							   const PgturbohybridBm25MetaTupleData *meta,
							   const PgturbohybridGraphMetaPageData *graphMeta,
							   PgturbohybridBm25Cache *cache,
							   PgturbohybridBm25QueryTerm *terms, int termCount,
							   const uint32 *docLens, const bool *liveNodes,
							   PgturbohybridBm25QueryStats *stats)
{
	if (termCount <= 1)
		return false;

	for (int termNo = 0; termNo < termCount; termNo++)
	{
		PgturbohybridBm25CacheLexiconEntry *entry;

		if (!terms[termNo].hasLexicon)
		{
			if (terms[termNo].deltaDf > 0)
				return false;
			continue;
		}
		if (terms[termNo].deltaDf > 0)
			return false;

		entry = terms[termNo].cacheEntry;
		if (entry == NULL)
			return false;
		if (!entry->impactBuilt)
		{
			if (!BlockNumberIsValid(entry->impactBlkno) ||
				!OffsetNumberIsValid(entry->impactOffno) ||
				entry->storedImpactCount == 0 ||
				entry->storedImpactCount < terms[termNo].baseDf)
				return false;
			if (!PgturbohybridBm25EnsureImpactHead(index, meta, graphMeta, cache,
											  &terms[termNo], docLens,
											  liveNodes, stats))
				return false;
		}
		if (entry->impactCount == 0 || entry->impactCount < terms[termNo].baseDf)
			return false;
	}

	return true;
}

static bool
PgturbohybridBm25ScoreImpactOR(Relation index,
						  const PgturbohybridBm25MetaTupleData *meta,
						  const PgturbohybridGraphMetaPageData *graphMeta,
						  PgturbohybridBm25Cache *cache,
						  PgturbohybridBm25QueryTerm *terms, int termCount,
						  const uint32 *docLens, const bool *liveNodes,
						  PgturbohybridBm25Accumulator *acc, int32 k,
						  PgturbohybridBm25QueryStats *stats)
{
	static const uint32 tierCumulativeLimits[] = {
		256,
		1024,
		3072,
		PG_UINT32_MAX
	};
	PgturbohybridBm25CacheLexiconEntry **impactTerms;
	int		   *impactTermNos;
	uint32	   *impactPositions;
	uint32		impactTermCount = 0;
	uint32		tiersRead = 0;
	bool		scored = false;
	bool		stoppedByBound = false;
	bool		tailSkipped = false;
	bool		unboundedDelta = false;
	double		remainingUpperBound = 0.0;

	if (termCount <= 1 || k <= 0)
		return false;

	impactTerms = MemoryContextAllocZero(acc->memoryContext,
										 sizeof(PgturbohybridBm25CacheLexiconEntry *) *
										 Max(termCount, 1));
	impactTermNos = MemoryContextAllocZero(acc->memoryContext,
										   sizeof(int) * Max(termCount, 1));
	impactPositions = MemoryContextAllocZero(acc->memoryContext,
											 sizeof(uint32) *
											 Max(termCount, 1));

	for (int termNo = 0; termNo < termCount; termNo++)
	{
		PgturbohybridBm25CacheLexiconEntry *entry;

		if (!terms[termNo].hasLexicon)
		{
			if (terms[termNo].deltaDf > 0)
				unboundedDelta = true;
			continue;
		}
		if (!PgturbohybridBm25EnsureImpactHead(index, meta, graphMeta, cache,
										  &terms[termNo], docLens, liveNodes,
										  stats))
		{
			if (terms[termNo].baseDf > 0 || terms[termNo].deltaDf > 0)
				tailSkipped = true;
			continue;
		}

		entry = terms[termNo].cacheEntry;
		if (entry->impactCount == 0)
		{
			if (terms[termNo].baseDf > 0 || terms[termNo].deltaDf > 0)
				tailSkipped = true;
			continue;
		}

		if (terms[termNo].deltaDf > 0)
		{
			unboundedDelta = true;
			tailSkipped = true;
		}
		if (entry->impactCount < terms[termNo].baseDf)
			tailSkipped = true;

		impactTerms[impactTermCount] = entry;
		impactTermNos[impactTermCount] = termNo;
		impactTermCount++;
	}

	for (uint32 tierNo = 0;
		 tierNo < lengthof(tierCumulativeLimits) && impactTermCount > 0;
		 tierNo++)
	{
		bool		tierRead = false;
		double		threshold;

		for (uint32 i = 0; i < impactTermCount; i++)
		{
			PgturbohybridBm25CacheLexiconEntry *entry = impactTerms[i];
			int			termNo = impactTermNos[i];
			uint32		start = impactPositions[i];
			uint32		end = Min(entry->impactCount,
								  tierCumulativeLimits[tierNo]);

			if (end <= start)
				continue;

			for (uint32 pos = start; pos < end; pos++)
				PgturbohybridBm25AccumulatorAddTermScore(acc,
													entry->impactHead[pos].nodeId,
													entry->impactHead[pos].score,
													terms[termNo].matchBit);

			impactPositions[i] = end;
			tierRead = true;
			scored = true;
			if (stats != NULL)
				stats->impactPostingsRead += end - start;
		}

		if (!tierRead)
			continue;

		tiersRead++;
		remainingUpperBound = 0.0;
		for (uint32 i = 0; i < impactTermCount; i++)
		{
			PgturbohybridBm25CacheLexiconEntry *entry = impactTerms[i];
			int			termNo = impactTermNos[i];
			uint32		pos = impactPositions[i];

			if (pos < entry->impactCount)
				remainingUpperBound += entry->impactHead[pos].score;
			else if (entry->impactCount < terms[termNo].baseDf &&
					 entry->impactCount > 0)
				remainingUpperBound +=
					entry->impactHead[entry->impactCount - 1].score;
		}

		threshold = PgturbohybridBm25KthScore(acc, k);
		if (!unboundedDelta && acc->topHeapCount >= (uint32) k &&
			remainingUpperBound < threshold)
		{
			stoppedByBound = true;
			break;
		}
	}

	if (stats != NULL && scored)
	{
		stats->strategy = PGTURBOHYBRID_BM25_RUNTIME_IMPACT_OR;
		stats->impactTerms = impactTermCount;
		stats->impactTiersRead = tiersRead;
		stats->impactRemainingUpperBound = remainingUpperBound;
		stats->impactEarlyStop = stoppedByBound || tailSkipped;
		stats->impactExactSafe = !unboundedDelta && !tailSkipped;
		stats->impactFullPostingsAvoided = tailSkipped;
		stats->candidatesScored = acc->touchedCount;
	}

	return scored;
}

static double
PgturbohybridBm25BlockUpperBound(const PgturbohybridBm25MetaTupleData *meta, double idf,
							double avgDocLen, uint16 maxTf)
{
	/*
	 * The configured BM25 b range is [0, 1], so docLen=1 gives the minimum
	 * normalization term and therefore the maximum possible contribution for
	 * any posting in the block.
	 */
	return PgturbohybridBm25PostingScore(meta, idf, avgDocLen, maxTf, 1);
}

pg_attribute_unused()
static bool
PgturbohybridBm25NeonAvailable(void)
{
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && (defined(__aarch64__) || defined(_M_ARM64))
	return pgturbohybrid_bm25_simd_force == PGTURBOHYBRID_BM25_SIMD_FORCE_AUTO ||
		pgturbohybrid_bm25_simd_force == PGTURBOHYBRID_BM25_SIMD_FORCE_NEON;
#else
	return false;
#endif
}

pg_attribute_unused()
static bool
PgturbohybridBm25Avx2Available(void)
{
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
	if (pgturbohybrid_bm25_simd_force != PGTURBOHYBRID_BM25_SIMD_FORCE_AUTO &&
		pgturbohybrid_bm25_simd_force != PGTURBOHYBRID_BM25_SIMD_FORCE_AVX2)
		return false;

#if defined(__AVX2__)
	return true;
#elif defined(__GNUC__) || defined(__clang__)
	return __builtin_cpu_supports("avx2");
#else
	return false;
#endif
#else
	return false;
#endif
}

pg_attribute_unused()
static void
PgturbohybridBm25RecordKernel(PgturbohybridBm25QueryStats *stats, int kernel)
{
	if (stats == NULL)
		return;
	if (kernel != PGTURBOHYBRID_BM25_KERNEL_SCALAR)
	{
		stats->decodeKernel = kernel;
		stats->scoreKernel = kernel;
	}
}

pg_attribute_unused()
static inline void
PgturbohybridBm25PrefetchDocState(const uint32 *docLens, const bool *liveNodes,
							 uint32 nodeId, uint32 nodeCount,
							 PgturbohybridBm25QueryStats *stats)
{
	if (nodeId >= nodeCount)
		return;

#if defined(__GNUC__) || defined(__clang__)
	__builtin_prefetch(docLens + nodeId, 0, 1);
	__builtin_prefetch(liveNodes + nodeId, 0, 1);
	if (stats != NULL)
		stats->prefetches++;
#else
	(void) docLens;
	(void) liveNodes;
	(void) stats;
#endif
}

static bool PGTURBOHYBRID_BM25_AVX2_TARGET
PgturbohybridBm25ScoreOffset16TfNormAvx2(const PgturbohybridBm25PostingsTuple postings,
									Size itemSize,
									const PgturbohybridBm25MetaTupleData *meta,
									const PgturbohybridGraphMetaPageData *graphMeta,
									float8 idf,
									const uint32 *docLens,
									const bool *liveNodes,
									PgturbohybridBm25Accumulator *acc,
									PgturbohybridBm25QueryTerm *term,
									PgturbohybridBm25QueryStats *stats)
{
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86))
	const uint16 *pairs;
	const uint16 *offsetsSoa;
	const uint16 *tfNorms;
	uint16		count = postings->count;
	Size		pairBytes = (Size) count * sizeof(uint16) * 2;
	Size		tfNormBytes = (Size) count * sizeof(uint16);
	uint16		encoding = postings->encoding & PGTURBOHYBRID_BM25_POSTINGS_ENCODING_MASK;
	__m256		factor;
	uint16		i = 0;
	bool		soa;

	if (!PgturbohybridBm25Avx2Available() ||
		(encoding != PGTURBOHYBRID_BM25_POSTINGS_ENCODING_OFFSET16 &&
		 encoding != PGTURBOHYBRID_BM25_POSTINGS_ENCODING_OFFSET16_SOA) ||
		(postings->encoding & PGTURBOHYBRID_BM25_POSTINGS_ENCODING_TFNORM_Q16) == 0 ||
		PgturbohybridBm25PostingsTupleSize(pairBytes + tfNormBytes) != itemSize ||
		postings->payloadBytes != pairBytes + tfNormBytes)
		return false;

	soa = encoding == PGTURBOHYBRID_BM25_POSTINGS_ENCODING_OFFSET16_SOA;
	pairs = (const uint16 *) postings->payload;
	offsetsSoa = pairs;
	tfNorms = (const uint16 *) (postings->payload + pairBytes);
	factor = _mm256_set1_ps((float) (idf * Max((double) meta->k1 + 1.0, 1.0) /
									 (double) PG_UINT16_MAX));
	for (; i + 8 <= count; i += 8)
	{
		__m128i		norm16 = _mm_loadu_si128((const __m128i *) (tfNorms + i));
		__m256		scoreVec = _mm256_mul_ps(
			_mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(norm16)), factor);
		uint16		offsets[8];
		float		scores[8];

		if (soa)
			memcpy(offsets, offsetsSoa + i, sizeof(offsets));
		else
		{
			for (int lane = 0; lane < 8; lane++)
				offsets[lane] = pairs[(i + lane) * 2];
		}
		for (int lane = 0; lane < 8; lane++)
			PgturbohybridBm25PrefetchDocState(docLens, liveNodes,
										 postings->firstNodeId + offsets[lane],
										 graphMeta->tqNodeCount, stats);
		_mm256_storeu_ps(scores, scoreVec);
		for (int lane = 0; lane < 8; lane++)
		{
			uint32		nodeId = postings->firstNodeId + offsets[lane];

			if (nodeId >= graphMeta->tqNodeCount || docLens[nodeId] == 0 ||
				!liveNodes[nodeId])
				continue;
			PgturbohybridBm25AccumulatorAddTermScore(acc, nodeId, scores[lane],
												term->matchBit);
			if (stats != NULL)
				stats->postingsDecoded++;
		}
		if (stats != NULL)
			stats->simdBlocks++;
	}
	for (; i < count; i++)
	{
		uint16		offset = soa ? offsetsSoa[i] : pairs[i * 2];
		uint32		nodeId = postings->firstNodeId + offset;
		double		score = PgturbohybridBm25PostingPrecomputedScore(meta, idf,
																tfNorms[i]);

		if (nodeId >= graphMeta->tqNodeCount || docLens[nodeId] == 0 ||
			!liveNodes[nodeId])
			continue;
		PgturbohybridBm25AccumulatorAddTermScore(acc, nodeId, score,
											term->matchBit);
		if (stats != NULL)
		{
			stats->postingsDecoded++;
			stats->scalarTailPostings++;
		}
	}
	PgturbohybridBm25RecordKernel(stats, soa ?
							 PGTURBOHYBRID_BM25_KERNEL_AVX2_SOA :
							 PGTURBOHYBRID_BM25_KERNEL_AVX2);
	return true;
#else
	(void) postings;
	(void) itemSize;
	(void) meta;
	(void) graphMeta;
	(void) idf;
	(void) docLens;
	(void) liveNodes;
	(void) acc;
	(void) term;
	(void) stats;
	return false;
#endif
}

static bool
PgturbohybridBm25ScoreOffset16TfNormNeon(const PgturbohybridBm25PostingsTuple postings,
									Size itemSize,
									const PgturbohybridBm25MetaTupleData *meta,
									const PgturbohybridGraphMetaPageData *graphMeta,
									float8 idf,
									const uint32 *docLens,
									const bool *liveNodes,
									PgturbohybridBm25Accumulator *acc,
									PgturbohybridBm25QueryTerm *term,
									PgturbohybridBm25QueryStats *stats)
{
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && (defined(__aarch64__) || defined(_M_ARM64))
	const uint16 *pairs;
	const uint16 *offsetsSoa;
	const uint16 *tfNorms;
	uint16		count = postings->count;
	Size		pairBytes = (Size) count * sizeof(uint16) * 2;
	Size		tfNormBytes = (Size) count * sizeof(uint16);
	uint16		encoding = postings->encoding & PGTURBOHYBRID_BM25_POSTINGS_ENCODING_MASK;
	float		factor;
	uint16		i = 0;
	bool		soa;

	if (!PgturbohybridBm25NeonAvailable() ||
		(encoding != PGTURBOHYBRID_BM25_POSTINGS_ENCODING_OFFSET16 &&
		 encoding != PGTURBOHYBRID_BM25_POSTINGS_ENCODING_OFFSET16_SOA) ||
		(postings->encoding & PGTURBOHYBRID_BM25_POSTINGS_ENCODING_TFNORM_Q16) == 0 ||
		PgturbohybridBm25PostingsTupleSize(pairBytes + tfNormBytes) != itemSize ||
		postings->payloadBytes != pairBytes + tfNormBytes)
		return false;

	soa = encoding == PGTURBOHYBRID_BM25_POSTINGS_ENCODING_OFFSET16_SOA;
	pairs = (const uint16 *) postings->payload;
	offsetsSoa = pairs;
	tfNorms = (const uint16 *) (postings->payload + pairBytes);
	factor = (float) (idf * Max((double) meta->k1 + 1.0, 1.0) /
					  (double) PG_UINT16_MAX);
	for (; i + 8 <= count; i += 8)
	{
		uint16x8x2_t pairVec;
		uint16x8_t normVec = vld1q_u16(tfNorms + i);
		uint32x4_t normLo = vmovl_u16(vget_low_u16(normVec));
		uint32x4_t normHi = vmovl_u16(vget_high_u16(normVec));
		float32x4_t scoreLo = vmulq_n_f32(vcvtq_f32_u32(normLo), factor);
		float32x4_t scoreHi = vmulq_n_f32(vcvtq_f32_u32(normHi), factor);
		uint16		offsets[8];
		float		scores[8];

		if (soa)
			vst1q_u16(offsets, vld1q_u16(offsetsSoa + i));
		else
		{
			pairVec = vld2q_u16(pairs + (i * 2));
			vst1q_u16(offsets, pairVec.val[0]);
		}
		for (int lane = 0; lane < 8; lane++)
			PgturbohybridBm25PrefetchDocState(docLens, liveNodes,
										 postings->firstNodeId + offsets[lane],
										 graphMeta->tqNodeCount, stats);
		vst1q_f32(scores, scoreLo);
		vst1q_f32(scores + 4, scoreHi);
		for (int lane = 0; lane < 8; lane++)
		{
			uint32		nodeId = postings->firstNodeId + offsets[lane];

			if (nodeId >= graphMeta->tqNodeCount || docLens[nodeId] == 0 ||
				!liveNodes[nodeId])
				continue;
			PgturbohybridBm25AccumulatorAddTermScore(acc, nodeId, scores[lane],
												term->matchBit);
			if (stats != NULL)
				stats->postingsDecoded++;
		}
		if (stats != NULL)
			stats->simdBlocks++;
	}
	for (; i < count; i++)
	{
		uint16		offset = soa ? offsetsSoa[i] : pairs[i * 2];
		uint32		nodeId = postings->firstNodeId + offset;
		double		score = PgturbohybridBm25PostingPrecomputedScore(meta, idf,
																tfNorms[i]);

		if (nodeId >= graphMeta->tqNodeCount || docLens[nodeId] == 0 ||
			!liveNodes[nodeId])
			continue;
		PgturbohybridBm25AccumulatorAddTermScore(acc, nodeId, score,
											term->matchBit);
		if (stats != NULL)
		{
			stats->postingsDecoded++;
			stats->scalarTailPostings++;
		}
	}
	PgturbohybridBm25RecordKernel(stats, soa ?
							 PGTURBOHYBRID_BM25_KERNEL_NEON_SOA :
							 PGTURBOHYBRID_BM25_KERNEL_NEON);
	return true;
#else
	(void) postings;
	(void) itemSize;
	(void) meta;
	(void) graphMeta;
	(void) idf;
	(void) docLens;
	(void) liveNodes;
	(void) acc;
	(void) term;
	(void) stats;
	return false;
#endif
}

static uint64
PgturbohybridBm25HotPostingsCacheLimitBytes(void)
{
	if (pgturbohybrid_bm25_hot_postings_cache_mb <= 0)
		return 0;
	return (uint64) pgturbohybrid_bm25_hot_postings_cache_mb * 1024ULL * 1024ULL;
}

static bool
PgturbohybridBm25HotPostingsCacheEligible(PgturbohybridBm25PostingIterator *it,
									 uint16 tupleEncoding)
{
	uint16		encoding = tupleEncoding & PGTURBOHYBRID_BM25_POSTINGS_ENCODING_MASK;

	return PgturbohybridBm25HotPostingsCacheLimitBytes() > 0 &&
		it->cache != NULL &&
		it->term->baseDf >= (uint32) pgturbohybrid_bm25_hot_postings_cache_min_df &&
		(encoding == PGTURBOHYBRID_BM25_POSTINGS_ENCODING_OFFSET16 ||
		 encoding == PGTURBOHYBRID_BM25_POSTINGS_ENCODING_OFFSET16_SOA) &&
		(tupleEncoding & PGTURBOHYBRID_BM25_POSTINGS_ENCODING_TFNORM_Q16) != 0;
}

static PgturbohybridBm25HotPostingsKey
PgturbohybridBm25HotPostingsCacheKey(PgturbohybridBm25PostingIterator *it,
								BlockNumber blkno, OffsetNumber offno)
{
	PgturbohybridBm25HotPostingsKey key;

	memset(&key, 0, sizeof(key));
	key.termId = it->term->lexicon.termId;
	key.blkno = blkno;
	key.offno = offno;
	return key;
}

static HTAB *
PgturbohybridBm25HotPostingsCacheHash(PgturbohybridBm25Cache *cache)
{
	HASHCTL		ctl;
	MemoryContext oldCtx;

	if (cache->hotPostingsHash != NULL)
		return cache->hotPostingsHash;

	memset(&ctl, 0, sizeof(ctl));
	ctl.keysize = sizeof(PgturbohybridBm25HotPostingsKey);
	ctl.entrysize = sizeof(PgturbohybridBm25HotPostingsHashEntry);
	ctl.hcxt = cache->ctx;

	oldCtx = MemoryContextSwitchTo(cache->ctx);
	cache->hotPostingsHash =
		hash_create("pgturbohybrid BM25 hot postings cache", 256, &ctl,
					HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
	MemoryContextSwitchTo(oldCtx);
	return cache->hotPostingsHash;
}

static void
PgturbohybridBm25HotPostingsCacheLruRemove(PgturbohybridBm25Cache *cache,
									  PgturbohybridBm25HotPostingsEntry *entry)
{
	if (entry->lruPrev != NULL)
		entry->lruPrev->lruNext = entry->lruNext;
	else
		cache->hotPostingsHead = entry->lruNext;

	if (entry->lruNext != NULL)
		entry->lruNext->lruPrev = entry->lruPrev;
	else
		cache->hotPostingsTail = entry->lruPrev;

	entry->lruPrev = NULL;
	entry->lruNext = NULL;
}

static void
PgturbohybridBm25HotPostingsCacheLruPushHead(PgturbohybridBm25Cache *cache,
										PgturbohybridBm25HotPostingsEntry *entry)
{
	entry->lruPrev = NULL;
	entry->lruNext = cache->hotPostingsHead;
	if (cache->hotPostingsHead != NULL)
		cache->hotPostingsHead->lruPrev = entry;
	else
		cache->hotPostingsTail = entry;
	cache->hotPostingsHead = entry;
}

static void
PgturbohybridBm25HotPostingsCacheLruTouch(PgturbohybridBm25Cache *cache,
									 PgturbohybridBm25HotPostingsEntry *entry)
{
	entry->lastUsed = ++cache->hotPostingsClock;
	if (cache->hotPostingsHead == entry)
		return;
	PgturbohybridBm25HotPostingsCacheLruRemove(cache, entry);
	PgturbohybridBm25HotPostingsCacheLruPushHead(cache, entry);
}

static bool
PgturbohybridBm25HotPostingsCacheLookup(PgturbohybridBm25PostingIterator *it,
								   BlockNumber blkno,
								   OffsetNumber offno)
{
	PgturbohybridBm25Cache *cache = it->cache;
	PgturbohybridBm25HotPostingsKey key;
	PgturbohybridBm25HotPostingsHashEntry *hashEntry;
	PgturbohybridBm25HotPostingsEntry *entry;

	if (cache == NULL || PgturbohybridBm25HotPostingsCacheLimitBytes() == 0 ||
		it->term->baseDf < (uint32) pgturbohybrid_bm25_hot_postings_cache_min_df)
		return false;

	if (cache->hotPostingsHash == NULL)
		goto miss;

	key = PgturbohybridBm25HotPostingsCacheKey(it, blkno, offno);
	hashEntry = hash_search(cache->hotPostingsHash, &key, HASH_FIND, NULL);
	if (hashEntry != NULL && hashEntry->entry != NULL)
	{
		entry = hashEntry->entry;
		if (it->postingsCapacity < entry->count)
		{
			if (it->postings == NULL)
				it->postings =
					PgturbohybridBm25MemoryContextAllocArray(it->memoryContext,
															 sizeof(PgturbohybridBm25Posting),
															 entry->count);
			else
				it->postings =
					PgturbohybridBm25RepallocArray(it->postings,
												   sizeof(PgturbohybridBm25Posting),
												   entry->count);
			it->postingsCapacity = entry->count;
		}
		memcpy(it->postings, entry->postings,
			   PgturbohybridBm25ArrayAllocSize(sizeof(PgturbohybridBm25Posting),
											   entry->count));
		it->count = entry->count;
		it->lastNodeId = entry->lastNodeId;
		it->maxTf = entry->maxTf;
		it->maxTfNormQ16 = entry->maxTfNormQ16;
		it->maxScoreFactor = entry->maxScoreFactor;
		it->nextBlkno = entry->nextBlkno;
		it->nextOffno = entry->nextOffno;
		it->blkno = entry->nextBlkno;
		it->offno = entry->nextOffno;
		it->chunkNo++;
		it->valid = it->count > 0;
		PgturbohybridBm25HotPostingsCacheLruTouch(cache, entry);
		if (it->stats != NULL)
		{
			it->stats->hotPostingsCacheHits++;
			it->stats->hotPostingsCacheBytes = cache->hotPostingsBytes;
			it->stats->hotPostingsCacheEvictions = cache->hotPostingsEvictions;
		}
		return it->valid;
	}

miss:
	if (it->stats != NULL)
	{
		it->stats->hotPostingsCacheMisses++;
		it->stats->hotPostingsCacheBytes = cache->hotPostingsBytes;
		it->stats->hotPostingsCacheEvictions = cache->hotPostingsEvictions;
	}
	return false;
}

static void
PgturbohybridBm25HotPostingsCacheFreeEntry(PgturbohybridBm25Cache *cache,
									  PgturbohybridBm25HotPostingsEntry *entry,
									  bool countEviction)
{
	if (entry == NULL)
		return;

	if (cache->hotPostingsHash != NULL)
		(void) hash_search(cache->hotPostingsHash, &entry->key,
						   HASH_REMOVE, NULL);
	PgturbohybridBm25HotPostingsCacheLruRemove(cache, entry);
	cache->hotPostingsBytes -= Min(cache->hotPostingsBytes,
								   (uint64) entry->bytes);
	if (countEviction)
		cache->hotPostingsEvictions++;
	if (entry->postings != NULL)
		pfree(entry->postings);
	pfree(entry);
}

static void
PgturbohybridBm25HotPostingsCacheEvictOne(PgturbohybridBm25Cache *cache)
{
	PgturbohybridBm25HotPostingsCacheFreeEntry(cache, cache->hotPostingsTail,
										  true);
}

static void
PgturbohybridBm25HotPostingsCacheStore(PgturbohybridBm25PostingIterator *it,
								  BlockNumber blkno,
								  OffsetNumber offno,
								  uint16 tupleEncoding)
{
	PgturbohybridBm25Cache *cache = it->cache;
	PgturbohybridBm25HotPostingsKey key;
	PgturbohybridBm25HotPostingsHashEntry *hashEntry;
	PgturbohybridBm25HotPostingsEntry *entry;
	uint64		limitBytes = PgturbohybridBm25HotPostingsCacheLimitBytes();
	Size		bytes;
	bool		found;
	MemoryContext oldCtx;

	if (!PgturbohybridBm25HotPostingsCacheEligible(it, tupleEncoding) ||
		it->count == 0)
		return;

	bytes = MAXALIGN(sizeof(PgturbohybridBm25HotPostingsEntry)) +
		MAXALIGN(PgturbohybridBm25ArrayAllocSize(sizeof(PgturbohybridBm25Posting),
												 it->count));
	if ((uint64) bytes > limitBytes)
		return;

	key = PgturbohybridBm25HotPostingsCacheKey(it, blkno, offno);
	hashEntry = hash_search(PgturbohybridBm25HotPostingsCacheHash(cache), &key,
							HASH_FIND, NULL);
	if (hashEntry != NULL && hashEntry->entry != NULL)
		PgturbohybridBm25HotPostingsCacheFreeEntry(cache, hashEntry->entry,
											  false);

	while (cache->hotPostingsBytes + (uint64) bytes > limitBytes &&
		   cache->hotPostingsTail != NULL)
		PgturbohybridBm25HotPostingsCacheEvictOne(cache);
	if (cache->hotPostingsBytes + (uint64) bytes > limitBytes)
		return;

	oldCtx = MemoryContextSwitchTo(cache->ctx);
	entry = palloc0(sizeof(PgturbohybridBm25HotPostingsEntry));
	entry->key = key;
	entry->count = it->count;
	entry->maxTf = it->maxTf;
	entry->maxTfNormQ16 = it->maxTfNormQ16;
	entry->maxScoreFactor = it->maxScoreFactor;
	entry->lastNodeId = it->lastNodeId;
	entry->nextBlkno = it->nextBlkno;
	entry->nextOffno = it->nextOffno;
	entry->bytes = bytes;
	entry->lastUsed = ++cache->hotPostingsClock;
	entry->postings =
		PgturbohybridBm25PallocArray(sizeof(PgturbohybridBm25Posting),
									 it->count);
	memcpy(entry->postings, it->postings,
		   PgturbohybridBm25ArrayAllocSize(sizeof(PgturbohybridBm25Posting),
										   it->count));
	hashEntry = hash_search(cache->hotPostingsHash, &key, HASH_ENTER, &found);
	hashEntry->entry = entry;
	PgturbohybridBm25HotPostingsCacheLruPushHead(cache, entry);
	cache->hotPostingsBytes += bytes;
	MemoryContextSwitchTo(oldCtx);

	if (it->stats != NULL)
		it->stats->hotPostingsCacheBytes = cache->hotPostingsBytes;
}

static void
PgturbohybridBm25PublishHotPostingsCacheStats(PgturbohybridBm25Cache *cache,
										 PgturbohybridBm25QueryStats *stats)
{
	if (cache == NULL || stats == NULL)
		return;

	stats->hotPostingsCacheBytes = cache->hotPostingsBytes;
	stats->hotPostingsCacheEvictions = cache->hotPostingsEvictions;
}

static bool
PgturbohybridBm25IteratorLoadChunk(PgturbohybridBm25PostingIterator *it)
{
	Buffer		buf;
	Page		page;
	ItemId		iid;
	PgturbohybridBm25PostingsTuple tuple;
	BlockNumber currentBlkno = it->blkno;
	OffsetNumber currentOffno = it->offno;
	uint16		tupleEncoding;

	it->valid = false;
	it->count = 0;
	it->pos = 0;

	if (it->chunkNo >= it->chunkLimit ||
		!BlockNumberIsValid(it->blkno) ||
		!OffsetNumberIsValid(it->offno))
		return false;

	if (PgturbohybridBm25HotPostingsCacheLookup(it, currentBlkno, currentOffno))
		return true;

	buf = ReadBuffer(it->index, currentBlkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	if (!PgturbohybridBm25PageIsKind(page, PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_POSTINGS) ||
		currentOffno > PageGetMaxOffsetNumber(page))
	{
		UnlockReleaseBuffer(buf);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("pgturbohybrid BM25 postings pointer is invalid")));
	}

	iid = PageGetItemId(page, currentOffno);
	if (!ItemIdIsUsed(iid))
	{
		UnlockReleaseBuffer(buf);
		return false;
	}

	tuple = (PgturbohybridBm25PostingsTuple) PageGetItem(page, iid);
	if (tuple->type != PGTURBOHYBRID_BM25_POSTINGS_TUPLE_TYPE ||
		tuple->termId != it->term->lexicon.termId)
	{
		UnlockReleaseBuffer(buf);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("pgturbohybrid BM25 postings tuple is invalid")));
	}

	it->count = tuple->count;
	it->lastNodeId = tuple->lastNodeId;
	it->maxTf = tuple->maxTf;
	it->maxTfNormQ16 = tuple->maxTfNormQ16;
	it->maxScoreFactor = tuple->maxScoreFactor;
	it->nextBlkno = tuple->nextBlkno;
	it->nextOffno = tuple->nextOffno;
	tupleEncoding = tuple->encoding;
	if (it->count > 0)
	{
			if (it->postingsCapacity < it->count)
			{
				if (it->postings == NULL)
					it->postings =
						PgturbohybridBm25MemoryContextAllocArray(it->memoryContext,
																 sizeof(PgturbohybridBm25Posting),
																 it->count);
				else
					it->postings =
						PgturbohybridBm25RepallocArray(it->postings,
													   sizeof(PgturbohybridBm25Posting),
													   it->count);
				it->postingsCapacity = it->count;
			}
			if (!PgturbohybridBm25DecodePostingsTuple(tuple, ItemIdGetLength(iid),
												 it->postings))
			{
				UnlockReleaseBuffer(buf);
				ereport(ERROR,
						(errcode(ERRCODE_DATA_CORRUPTED),
						 errmsg("pgturbohybrid BM25 postings payload is invalid")));
			}
			if (it->maxTf == 0)
			{
				for (uint16 i = 0; i < it->count; i++)
					it->maxTf = Max(it->maxTf, it->postings[i].tf);
			}
			if (it->maxTfNormQ16 == 0)
			{
				for (uint16 i = 0; i < it->count; i++)
					it->maxTfNormQ16 = Max(it->maxTfNormQ16,
										   it->postings[i].reserved);
			}
	}

	UnlockReleaseBuffer(buf);
	if (it->stats != NULL)
		it->stats->blocksVisited++;

	it->blkno = it->nextBlkno;
	it->offno = it->nextOffno;
	it->chunkNo++;
	it->valid = it->count > 0;
	PgturbohybridBm25HotPostingsCacheStore(it, currentBlkno, currentOffno,
									  tupleEncoding);
	return it->valid;
}

static bool
PgturbohybridBm25IteratorAdvanceValid(PgturbohybridBm25PostingIterator *it)
{
	for (;;)
	{
		while (it->valid && it->pos < it->count)
		{
			uint32		nodeId = it->postings[it->pos].nodeId;

			if (nodeId < it->nodeCount && it->docLens[nodeId] != 0 &&
				it->liveNodes[nodeId])
				return true;
			it->pos++;
			if (it->stats != NULL)
				it->stats->postingsDecoded++;
		}
		if (!PgturbohybridBm25IteratorLoadChunk(it))
			return false;
	}
}

static bool
PgturbohybridBm25IteratorInit(PgturbohybridBm25PostingIterator *it, Relation index,
						 PgturbohybridBm25Cache *cache,
						 PgturbohybridBm25QueryTerm *term, double idf,
						 double avgDocLen, const uint32 *docLens,
						 const bool *liveNodes, uint32 nodeCount,
						 MemoryContext memoryContext,
						 PgturbohybridBm25QueryStats *stats)
{
	memset(it, 0, sizeof(*it));
	it->index = index;
	it->cache = cache;
	it->term = term;
	it->idf = idf;
	it->avgDocLen = avgDocLen;
	it->docLens = docLens;
	it->liveNodes = liveNodes;
	it->nodeCount = nodeCount;
	it->chunkLimit = Max(term->lexicon.postingsChunkCount, 1);
	it->blkno = term->lexicon.postingsBlkno;
	it->offno = term->lexicon.postingsOffno;
	it->memoryContext = memoryContext;
	it->stats = stats;

	if (!PgturbohybridBm25IteratorLoadChunk(it))
		return false;
	return PgturbohybridBm25IteratorAdvanceValid(it);
}

static void
PgturbohybridBm25IteratorClose(PgturbohybridBm25PostingIterator *it)
{
	if (it->postings != NULL)
	{
		pfree(it->postings);
		it->postings = NULL;
	}
	it->postingsCapacity = 0;
	it->valid = false;
}

static uint32
PgturbohybridBm25IteratorNodeId(const PgturbohybridBm25PostingIterator *it)
{
	if (!it->valid || it->pos >= it->count)
		return PG_UINT32_MAX;
	return it->postings[it->pos].nodeId;
}

static bool
PgturbohybridBm25IteratorSeekTo(PgturbohybridBm25PostingIterator *it, uint32 target)
{
	while (it->valid)
	{
		if (it->lastNodeId < target)
		{
			if (it->stats != NULL)
				it->stats->blocksSkipped++;
			if (!PgturbohybridBm25IteratorLoadChunk(it))
				return false;
			continue;
		}

		while (it->pos < it->count &&
			   it->postings[it->pos].nodeId < target)
		{
			it->pos++;
			if (it->stats != NULL)
				it->stats->postingsDecoded++;
		}
		return PgturbohybridBm25IteratorAdvanceValid(it);
	}

	return false;
}

static bool
PgturbohybridBm25IteratorSeekExact(PgturbohybridBm25PostingIterator *it, uint32 target,
							  PgturbohybridBm25Posting **posting)
{
	if (!PgturbohybridBm25IteratorSeekTo(it, target))
		return false;
	if (PgturbohybridBm25IteratorNodeId(it) != target)
		return false;
	if (posting != NULL)
		*posting = &it->postings[it->pos];
	return true;
}

static bool
PgturbohybridBm25IteratorAdvancePast(PgturbohybridBm25PostingIterator *it,
								uint32 nodeId)
{
	if (nodeId == PG_UINT32_MAX)
		return false;
	return PgturbohybridBm25IteratorSeekTo(it, nodeId + 1);
}

static double
PgturbohybridBm25IteratorUpperBound(const PgturbohybridBm25MetaTupleData *meta,
							   const PgturbohybridBm25PostingIterator *it,
							   PgturbohybridBm25QueryStats *stats)
{
	if ((meta->reserved2 & PGTURBOHYBRID_BM25_META_FLAG_TFNORM_Q16) != 0 &&
		it->maxTfNormQ16 > 0)
	{
		if (stats != NULL)
		{
			stats->wandBoundType = PGTURBOHYBRID_BM25_WAND_BOUND_TFNORM_Q16;
			stats->wandBoundTighteningHits++;
		}
		if (it->maxScoreFactor > 0)
			return it->idf * (double) it->maxScoreFactor;
		return PgturbohybridBm25PostingPrecomputedScore(meta, it->idf,
												   it->maxTfNormQ16);
	}
	if (stats != NULL && stats->wandBoundType == PGTURBOHYBRID_BM25_WAND_BOUND_NONE)
		stats->wandBoundType = PGTURBOHYBRID_BM25_WAND_BOUND_TF;
	return PgturbohybridBm25BlockUpperBound(meta, it->idf, it->avgDocLen,
									   it->maxTf);
}

static int
PgturbohybridBm25ActiveIteratorCompare(const void *a, const void *b)
{
	const PgturbohybridBm25PostingIterator *ia =
		*((const PgturbohybridBm25PostingIterator *const *) a);
	const PgturbohybridBm25PostingIterator *ib =
		*((const PgturbohybridBm25PostingIterator *const *) b);
	uint32		nodeA = PgturbohybridBm25IteratorNodeId(ia);
	uint32		nodeB = PgturbohybridBm25IteratorNodeId(ib);

	if (nodeA < nodeB)
		return -1;
	if (nodeA > nodeB)
		return 1;
	return (ia->term->lexicon.termId > ib->term->lexicon.termId) -
		(ia->term->lexicon.termId < ib->term->lexicon.termId);
}

static void
PgturbohybridBm25SortActiveInitial(PgturbohybridBm25PostingIterator **active,
							  int activeCount,
							  PgturbohybridBm25QueryStats *stats)
{
	if (activeCount <= 1)
		return;

	qsort(active, activeCount, sizeof(PgturbohybridBm25PostingIterator *),
		  PgturbohybridBm25ActiveIteratorCompare);
	if (stats != NULL)
	{
		stats->wandActiveSorts++;
		stats->wandFullReorders++;
	}
}

static void
PgturbohybridBm25RefreshActiveOrder(PgturbohybridBm25PostingIterator **active,
							   int *activeCount,
							   PgturbohybridBm25QueryStats *stats)
{
	int			count = *activeCount;
	int			out = 0;
	bool		changed = false;

	for (int i = 0; i < count; i++)
	{
		if (active[i]->valid && PgturbohybridBm25IteratorNodeId(active[i]) != PG_UINT32_MAX)
			active[out++] = active[i];
		else
			changed = true;
	}
	count = out;

	for (int i = 1; i < count; i++)
	{
		PgturbohybridBm25PostingIterator *it = active[i];
		int			j = i;

		while (j > 0 &&
			   PgturbohybridBm25ActiveIteratorCompare(&it, &active[j - 1]) < 0)
		{
			active[j] = active[j - 1];
			j--;
		}
		if (j != i)
		{
			active[j] = it;
			changed = true;
		}
	}

	if (changed && stats != NULL)
		stats->wandHeapUpdates++;
	*activeCount = count;
}

static double
PgturbohybridBm25KthScore(PgturbohybridBm25Accumulator *acc, int32 k)
{
	double		threshold = acc->seedThreshold;

	if (k <= 0 || acc->topHeapCount < (uint32) k)
		return threshold;
	return Max(acc->threshold, threshold);
}

static bool
PgturbohybridBm25ScoreBaseWand(Relation index,
						  const PgturbohybridBm25MetaTupleData *meta,
						  const PgturbohybridGraphMetaPageData *graphMeta,
						  PgturbohybridBm25Cache *cache,
						  TSQuery query, PgturbohybridBm25QueryTerm *terms, int termCount,
						  const PgturbohybridBm25QueryEval *eval,
						  const uint32 *docLens, const bool *liveNodes,
						  PgturbohybridBm25Accumulator *acc, int32 k,
						  MemoryContext memoryContext,
						  PgturbohybridBm25QueryStats *stats)
{
	PgturbohybridBm25PostingIterator *iterators;
	PgturbohybridBm25PostingIterator **active;
	int			iteratorCount = 0;
	int			activeCount = 0;
	double		corpusDocCount = Max((double) meta->docCount +
									 (double) meta->deltaDocCount, 1.0);
	double		avgDocLen = ((double) meta->totalDocLen +
							 (double) meta->deltaTotalDocLen) / corpusDocCount;
	bool		used = false;

	iterators = MemoryContextAllocZero(memoryContext,
									   sizeof(PgturbohybridBm25PostingIterator) *
									   Max(termCount, 1));
	active = MemoryContextAllocZero(memoryContext,
									sizeof(PgturbohybridBm25PostingIterator *) *
									Max(termCount, 1));

	for (int termNo = 0; termNo < termCount; termNo++)
	{
		PgturbohybridBm25QueryTerm *term = &terms[termNo];
		uint32		df;
		double		idf;

		if (!term->hasLexicon)
			continue;

		df = term->baseDf + term->deltaDf;
		idf = log(1.0 + (corpusDocCount - (double) df + 0.5) /
				  ((double) df + 0.5));
		if (PgturbohybridBm25IteratorInit(&iterators[iteratorCount], index, cache,
									 term, idf, avgDocLen, docLens,
									 liveNodes, graphMeta->tqNodeCount,
									 memoryContext, stats))
		{
			active[activeCount++] = &iterators[iteratorCount];
			iteratorCount++;
		}
		else
			PgturbohybridBm25IteratorClose(&iterators[iteratorCount]);
	}
	PgturbohybridBm25SortActiveInitial(active, activeCount, stats);

	while (activeCount > 0)
	{
		int			pivot = -1;
		double		upperBound = 0.0;
		double		threshold = PgturbohybridBm25KthScore(acc, k);
		uint32		pivotNode;

		if (stats != NULL)
			stats->wandIterations++;

		for (int i = 0; i < activeCount; i++)
		{
			upperBound += PgturbohybridBm25IteratorUpperBound(meta, active[i],
														 stats);
			if (acc->touchedCount < (uint32) k || upperBound > threshold)
			{
				pivot = i;
				break;
			}
		}

		if (pivot < 0)
		{
			if (stats != NULL)
				stats->blocksSkipped += activeCount;
			break;
		}

		pivotNode = PgturbohybridBm25IteratorNodeId(active[pivot]);
		if (pivotNode == PG_UINT32_MAX)
			break;

		if (PgturbohybridBm25IteratorNodeId(active[0]) == pivotNode)
		{
			double		score = 0.0;
			uint64		matchedTerms = 0;
			bool		matched = false;

			for (int i = 0; i < activeCount; i++)
			{
				PgturbohybridBm25PostingIterator *it = active[i];

				if (!PgturbohybridBm25IteratorSeekTo(it, pivotNode))
					continue;
				if (PgturbohybridBm25IteratorNodeId(it) == pivotNode)
				{
					PgturbohybridBm25Posting *posting = &it->postings[it->pos];

					score += PgturbohybridBm25PostingScoreDecoded(meta, it->idf,
															 it->avgDocLen,
															 posting,
															 docLens[pivotNode]);
					matchedTerms |= it->term->matchBit;
					matched = true;
					if (stats != NULL)
						stats->postingsDecoded++;
				}
			}

			if (matched &&
				PgturbohybridBm25MatchedQuery(query, terms, termCount, eval,
										 matchedTerms, stats))
			{
				used = true;
				if (stats != NULL)
					stats->candidatesScored++;
				if (acc->touchedCount < (uint32) k || score > threshold)
					PgturbohybridBm25AccumulatorAddTermScore(acc, pivotNode, score,
														matchedTerms);
			}

			for (int i = 0; i < activeCount; i++)
			{
				PgturbohybridBm25PostingIterator *it = active[i];

				if (PgturbohybridBm25IteratorNodeId(it) == pivotNode)
					(void) PgturbohybridBm25IteratorAdvancePast(it, pivotNode);
			}
			PgturbohybridBm25RefreshActiveOrder(active, &activeCount, stats);
		}
		else
		{
			for (int i = 0; i < pivot; i++)
				(void) PgturbohybridBm25IteratorSeekTo(active[i], pivotNode);
			PgturbohybridBm25RefreshActiveOrder(active, &activeCount, stats);
		}

		CHECK_FOR_INTERRUPTS();
	}

	for (int i = 0; i < iteratorCount; i++)
		PgturbohybridBm25IteratorClose(&iterators[i]);

	return used;
}

static bool
PgturbohybridBm25ScoreBaseAndRarestDriver(Relation index,
									 const PgturbohybridBm25MetaTupleData *meta,
									 const PgturbohybridGraphMetaPageData *graphMeta,
									 PgturbohybridBm25Cache *cache,
									 PgturbohybridBm25QueryTerm *terms,
									 int termCount,
									 const PgturbohybridBm25QueryEval *eval,
									 const uint32 *docLens,
									 const bool *liveNodes,
									 PgturbohybridBm25Accumulator *acc,
									 int32 k,
									 MemoryContext memoryContext,
									 PgturbohybridBm25QueryStats *stats)
{
	PgturbohybridBm25PostingIterator *iterators;
	double	   *idfs;
	int			driver = -1;
	uint32		driverDf = PG_UINT32_MAX;
	double		corpusDocCount = Max((double) meta->docCount +
									 (double) meta->deltaDocCount, 1.0);
	double		avgDocLen = ((double) meta->totalDocLen +
							 (double) meta->deltaTotalDocLen) / corpusDocCount;
	bool		used = true;

	if (eval == NULL || eval->shape != PGTURBOHYBRID_BM25_QUERY_PURE_AND ||
		termCount <= 1)
		return false;

	for (int termNo = 0; termNo < termCount; termNo++)
	{
		uint32		df = terms[termNo].baseDf + terms[termNo].deltaDf;

		if (!terms[termNo].hasLexicon)
		{
			if (stats != NULL)
				stats->strategy = PGTURBOHYBRID_BM25_RUNTIME_AND_RAREST_DRIVER;
			return true;
		}
		if (df < driverDf)
		{
			driverDf = df;
			driver = termNo;
		}
	}
	if (driver < 0)
		return true;

	iterators =
		PgturbohybridBm25MemoryContextAllocZeroArray(memoryContext,
													 sizeof(PgturbohybridBm25PostingIterator),
													 termCount);
	idfs = PgturbohybridBm25MemoryContextAllocZeroArray(memoryContext,
														sizeof(double),
														termCount);

	for (int termNo = 0; termNo < termCount; termNo++)
	{
		uint32		df = terms[termNo].baseDf + terms[termNo].deltaDf;

		idfs[termNo] = log(1.0 + (corpusDocCount - (double) df + 0.5) /
						   ((double) df + 0.5));
		if (!PgturbohybridBm25IteratorInit(&iterators[termNo], index, cache,
									  &terms[termNo], idfs[termNo],
									  avgDocLen, docLens, liveNodes,
									  graphMeta->tqNodeCount, memoryContext,
									  stats))
		{
			used = true;
			goto cleanup;
		}
	}

	if (stats != NULL)
	{
		stats->strategy = PGTURBOHYBRID_BM25_RUNTIME_AND_RAREST_DRIVER;
		stats->andDriverDf = driverDf;
	}

	while (iterators[driver].valid)
	{
		PgturbohybridBm25Posting *driverPosting;
		uint32		nodeId = PgturbohybridBm25IteratorNodeId(&iterators[driver]);
		double		score;
		bool		verified = true;
		double		threshold = PgturbohybridBm25KthScore(acc, k);

		if (nodeId == PG_UINT32_MAX)
			break;

		driverPosting = &iterators[driver].postings[iterators[driver].pos];
		score = PgturbohybridBm25PostingScoreDecoded(meta, idfs[driver],
												avgDocLen, driverPosting,
												docLens[nodeId]);
		if (stats != NULL)
			stats->postingsDecoded++;

		for (int termNo = 0; termNo < termCount; termNo++)
		{
			PgturbohybridBm25Posting *posting = NULL;

			if (termNo == driver)
				continue;
			if (!PgturbohybridBm25IteratorSeekExact(&iterators[termNo], nodeId,
											   &posting))
			{
				verified = false;
				break;
			}
			score += PgturbohybridBm25PostingScoreDecoded(meta, idfs[termNo],
													 avgDocLen, posting,
													 docLens[nodeId]);
			if (stats != NULL)
				stats->postingsDecoded++;
		}

		if (verified)
		{
			if (stats != NULL)
			{
				stats->andVerifiedCandidates++;
				stats->candidatesScored++;
			}
			if (acc->touchedCount < (uint32) k || score > threshold)
				PgturbohybridBm25AccumulatorAddTermScore(acc, nodeId, score,
													eval->requiredMask);
		}
		else if (stats != NULL)
			stats->andRejectedCandidates++;

		if (!PgturbohybridBm25IteratorAdvancePast(&iterators[driver], nodeId))
			break;
		CHECK_FOR_INTERRUPTS();
	}

cleanup:
	for (int termNo = 0; termNo < termCount; termNo++)
		PgturbohybridBm25IteratorClose(&iterators[termNo]);
	return used;
}

int
PgturbohybridBm25TopK(Relation index, TSQuery query, int32 k, bool useWand,
				 MemoryContext memoryContext, PgturbohybridBm25Result **results,
				 PgturbohybridBm25QueryStats *stats)
{
	PgturbohybridBm25MetaTupleData bm25Meta;
	PgturbohybridGraphMetaPageData graphMeta;
	PgturbohybridBm25Cache *cache;
	const uint32 *docLens;
	const ItemPointerData *heapTids;
	const bool *liveNodes;
	PgturbohybridBm25DeltaCacheEntry *deltaTerms = NULL;
	uint32		deltaTermCount = 0;
	uint32		deltaPostingCount = 0;
	uint64		deltaCacheBytes = 0;
	PgturbohybridBm25Accumulator acc;
	PgturbohybridBm25QueryTerm *terms;
	PgturbohybridBm25QueryEval eval;
	int			termCount;
	int			resolvedTerms = 0;
	int			resultCount;
	bool		usedBaseWand = false;
	bool		seededImpactWand = false;
	PgturbohybridBm25Posting *decodedScratch = NULL;
	uint16		decodedScratchCapacity = 0;
	int			accumulatorMode;
	MemoryContext oldCtx;

	if (stats != NULL)
	{
		memset(stats, 0, sizeof(*stats));
		stats->decodeKernel = PGTURBOHYBRID_BM25_KERNEL_SCALAR;
		stats->scoreKernel = PGTURBOHYBRID_BM25_KERNEL_SCALAR;
		stats->strategy = PGTURBOHYBRID_BM25_RUNTIME_NONE;
	}
	pgturbohybrid_last_bm25_decode_kernel = PGTURBOHYBRID_BM25_KERNEL_SCALAR;
	pgturbohybrid_last_bm25_score_kernel = PGTURBOHYBRID_BM25_KERNEL_SCALAR;
	pgturbohybrid_last_bm25_simd_blocks = 0;
	pgturbohybrid_last_bm25_scalar_tail_postings = 0;
	*results = NULL;

	if (k <= 0)
		return 0;
	if (!PgturbohybridBm25ReadMeta(index, &bm25Meta) ||
		!PgturbohybridGraphReadMeta(index, &graphMeta) ||
		graphMeta.tqNodeCount == 0)
		return 0;

	oldCtx = MemoryContextSwitchTo(memoryContext);
	termCount = PgturbohybridBm25ExtractTerms(query, &terms, memoryContext);
	eval = PgturbohybridBm25CompileQueryEval(query, terms, termCount);
	if (stats != NULL)
	{
		stats->queryTerms = termCount;
		stats->usedWand = false;
		stats->queryShape = eval.shape;
		stats->booleanEvalMode = eval.mode;
	}
	if (termCount == 0)
	{
		MemoryContextSwitchTo(oldCtx);
		return 0;
	}

	cache = PgturbohybridBm25GetCache(index, &bm25Meta, &graphMeta, stats);
	if (stats != NULL)
	{
		stats->cacheBytes = MemoryContextMemAllocated(cache->ctx, true);
		stats->cacheLexiconEntries = cache->lexiconCount;
		stats->cacheDocstatsLoaded = cache->docStatsLoaded;
		stats->cacheLivenessLoaded = cache->livenessLoaded;
	}
	for (int termNo = 0; termNo < termCount; termNo++)
	{
		if (PgturbohybridBm25CacheFindLexiconEntry(cache, &terms[termNo],
											  &terms[termNo].lexicon))
		{
			terms[termNo].hasLexicon = true;
			terms[termNo].baseDf = terms[termNo].lexicon.df;
			resolvedTerms++;
		}
	}

	if (resolvedTerms == 0 && bm25Meta.deltaDocCount == 0)
	{
		if (stats != NULL)
		{
			stats->cacheBytes = MemoryContextMemAllocated(cache->ctx, true);
			PgturbohybridBm25PublishHotPostingsCacheStats(cache, stats);
		}
		MemoryContextSwitchTo(oldCtx);
		return 0;
	}

	PgturbohybridBm25EnsureDocStats(index, cache, &bm25Meta, &graphMeta);
	PgturbohybridBm25EnsureLiveness(index, cache, &graphMeta);
	docLens = cache->docLens;
	heapTids = cache->heapTids;
	liveNodes = cache->liveNodes;

	if (pgturbohybrid_bm25_cache_max_mb > 0 && !cache->deltaCacheBuilt)
	{
		PgturbohybridBm25BuildDeltaCacheEntries(index, &bm25Meta, cache,
										   terms, termCount, memoryContext,
										   &deltaTerms, &deltaTermCount,
										   &deltaPostingCount,
										   &deltaCacheBytes, stats);
	}
	else
	{
		PgturbohybridBm25EnsureDeltaCache(index, &bm25Meta, cache, stats);
		deltaTerms = cache->deltaTerms;
		deltaTermCount = cache->deltaTermCount;
		deltaPostingCount = cache->deltaPostingCount;
		deltaCacheBytes = cache->deltaCacheBytes;
	}
	PgturbohybridBm25CountDeltaDf(deltaTerms, deltaTermCount, terms, termCount);
	if (stats != NULL)
	{
		stats->deltaCacheBytes = deltaCacheBytes;
		stats->deltaCacheTerms = deltaTermCount;
		stats->cacheBytes = MemoryContextMemAllocated(cache->ctx, true);
		stats->cacheDocstatsLoaded = cache->docStatsLoaded;
		stats->cacheLivenessLoaded = cache->livenessLoaded;
	}
	(void) deltaPostingCount;
	for (int termNo = 0; termNo < termCount; termNo++)
	{
		if (!terms[termNo].hasLexicon && terms[termNo].deltaDf > 0)
			resolvedTerms++;
	}
	accumulatorMode = PgturbohybridBm25ChooseAccumulatorMode(&bm25Meta, terms,
														termCount);
	if (pgturbohybrid_bm25_strategy == PGTURBOHYBRID_BM25_STRATEGY_DAAT_HASH)
		accumulatorMode = PGTURBOHYBRID_BM25_ACCUMULATOR_HASH;
	PgturbohybridBm25AccumulatorInit(&acc, memoryContext, (uint32) (k * termCount),
								(uint32) k, accumulatorMode,
								graphMeta.tqNodeCount, stats);

	if (pgturbohybrid_bm25_strategy == PGTURBOHYBRID_BM25_STRATEGY_AUTO ||
		pgturbohybrid_bm25_strategy == PGTURBOHYBRID_BM25_STRATEGY_IMPACT)
		usedBaseWand = PgturbohybridBm25ScoreImpactSingle(index, &bm25Meta,
													  &graphMeta, cache,
													  terms, termCount,
													  docLens, liveNodes, &acc,
													  k, stats);

	if (!usedBaseWand &&
		pgturbohybrid_bm25_strategy == PGTURBOHYBRID_BM25_STRATEGY_IMPACT_OR &&
		eval.shape == PGTURBOHYBRID_BM25_QUERY_PURE_OR)
		usedBaseWand = PgturbohybridBm25ScoreImpactOR(index, &bm25Meta,
												  &graphMeta, cache,
												  terms, termCount,
												  docLens, liveNodes, &acc,
												  k, stats);

	if (!usedBaseWand &&
		pgturbohybrid_bm25_strategy == PGTURBOHYBRID_BM25_STRATEGY_AUTO &&
		pgturbohybrid_bm25_impact_or_mode != PGTURBOHYBRID_BM25_IMPACT_OR_MODE_OFF &&
		eval.shape == PGTURBOHYBRID_BM25_QUERY_PURE_OR)
	{
		bool		canUseImpactOR =
			pgturbohybrid_bm25_impact_or_mode ==
			PGTURBOHYBRID_BM25_IMPACT_OR_MODE_APPROX;

		if (!canUseImpactOR)
			canUseImpactOR = PgturbohybridBm25ImpactORCanBeExact(index,
															&bm25Meta,
															&graphMeta,
															cache, terms,
															termCount,
															docLens, liveNodes,
															stats);
		if (canUseImpactOR)
			usedBaseWand = PgturbohybridBm25ScoreImpactOR(index, &bm25Meta,
													  &graphMeta, cache,
													  terms, termCount,
													  docLens, liveNodes, &acc,
													  k, stats);
	}

	if (!usedBaseWand &&
		(pgturbohybrid_bm25_strategy == PGTURBOHYBRID_BM25_STRATEGY_AUTO ||
		 pgturbohybrid_bm25_strategy == PGTURBOHYBRID_BM25_STRATEGY_IMPACT) &&
		PgturbohybridBm25QueryHasOperator(query, OP_OR) &&
		!PgturbohybridBm25QueryHasOperator(query, OP_AND))
		seededImpactWand = PgturbohybridBm25SeedImpactHeads(index, &bm25Meta,
														&graphMeta, cache,
														terms, termCount,
														docLens, liveNodes,
														&acc, k, stats);

	if (!usedBaseWand &&
		pgturbohybrid_bm25_strategy == PGTURBOHYBRID_BM25_STRATEGY_AUTO &&
		eval.shape == PGTURBOHYBRID_BM25_QUERY_PURE_AND)
		usedBaseWand = PgturbohybridBm25ScoreBaseAndRarestDriver(index,
															&bm25Meta,
															&graphMeta, cache,
															terms, termCount,
															&eval, docLens,
															liveNodes, &acc, k,
															memoryContext,
															stats);

	if (!usedBaseWand &&
		useWand && pgturbohybrid_enable_wand &&
		pgturbohybrid_bm25_strategy != PGTURBOHYBRID_BM25_STRATEGY_DAAT_SIMD &&
		pgturbohybrid_bm25_strategy != PGTURBOHYBRID_BM25_STRATEGY_DAAT_HASH)
	{
		usedBaseWand = PgturbohybridBm25ScoreBaseWand(index, &bm25Meta,
												 &graphMeta, cache, query, terms,
												 termCount, &eval,
												 docLens, liveNodes, &acc, k,
												 memoryContext, stats);
		if (stats != NULL)
		{
			stats->usedWand = usedBaseWand;
			if (usedBaseWand)
				stats->strategy = seededImpactWand ?
					PGTURBOHYBRID_BM25_RUNTIME_IMPACT_SEEDED_WAND :
					PGTURBOHYBRID_BM25_RUNTIME_WAND;
		}
	}

	if (!usedBaseWand)
	{
		if (stats != NULL)
			stats->strategy =
				pgturbohybrid_bm25_strategy == PGTURBOHYBRID_BM25_STRATEGY_DAAT_HASH ?
				PGTURBOHYBRID_BM25_RUNTIME_DAAT_HASH :
				PGTURBOHYBRID_BM25_RUNTIME_DAAT_SIMD;
	for (int termNo = 0; termNo < termCount; termNo++)
	{
		float8		idf;
		PgturbohybridBm25QueryTerm *term = &terms[termNo];
		uint32		df;
		double		corpusDocCount;
		double		avgDocLen;
		BlockNumber postingsBlkno;
		OffsetNumber postingsOffno;
		uint32		chunkLimit;

		if (!term->hasLexicon)
			continue;

		df = term->baseDf + term->deltaDf;
		corpusDocCount = Max((double) bm25Meta.docCount +
							 (double) bm25Meta.deltaDocCount, 1.0);
		avgDocLen = ((double) bm25Meta.totalDocLen +
					 (double) bm25Meta.deltaTotalDocLen) / corpusDocCount;
		idf = log(1.0 + (corpusDocCount - (double) df + 0.5) /
				  ((double) df + 0.5));
		postingsBlkno = term->lexicon.postingsBlkno;
		postingsOffno = term->lexicon.postingsOffno;
		chunkLimit = Max(term->lexicon.postingsChunkCount, 1);

		for (uint32 chunkNo = 0;
			 chunkNo < chunkLimit && BlockNumberIsValid(postingsBlkno) &&
			 OffsetNumberIsValid(postingsOffno);
			 chunkNo++)
		{
			Buffer		buf;
			Page		page;
				ItemId		iid;
				PgturbohybridBm25PostingsTuple postings;
				BlockNumber nextBlkno;
				OffsetNumber nextOffno;

			buf = ReadBuffer(index, postingsBlkno);
			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);
			if (!PgturbohybridBm25PageIsKind(page, PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_POSTINGS) ||
				postingsOffno > PageGetMaxOffsetNumber(page))
			{
				UnlockReleaseBuffer(buf);
				ereport(ERROR,
						(errcode(ERRCODE_DATA_CORRUPTED),
						 errmsg("pgturbohybrid BM25 postings pointer is invalid")));
			}

			iid = PageGetItemId(page, postingsOffno);
			if (!ItemIdIsUsed(iid))
			{
				UnlockReleaseBuffer(buf);
				break;
			}

			postings = (PgturbohybridBm25PostingsTuple) PageGetItem(page, iid);
			if (postings->type != PGTURBOHYBRID_BM25_POSTINGS_TUPLE_TYPE ||
				postings->termId != term->lexicon.termId)
			{
				UnlockReleaseBuffer(buf);
				ereport(ERROR,
						(errcode(ERRCODE_DATA_CORRUPTED),
						 errmsg("pgturbohybrid BM25 postings tuple is invalid")));
				}
				nextBlkno = postings->nextBlkno;
				nextOffno = postings->nextOffno;
				if (stats != NULL)
					stats->blocksVisited++;
				if (PgturbohybridBm25ScoreOffset16TfNormAvx2(postings,
														 ItemIdGetLength(iid),
														 &bm25Meta, &graphMeta,
														 idf, docLens,
														 liveNodes, &acc, term,
														 stats) ||
					PgturbohybridBm25ScoreOffset16TfNormNeon(postings,
														 ItemIdGetLength(iid),
														 &bm25Meta, &graphMeta,
														 idf, docLens,
														 liveNodes, &acc, term,
														 stats))
				{
					UnlockReleaseBuffer(buf);
					postingsBlkno = nextBlkno;
					postingsOffno = nextOffno;
					CHECK_FOR_INTERRUPTS();
					continue;
				}
				if (decodedScratchCapacity < postings->count)
				{
					if (decodedScratch == NULL)
						decodedScratch =
							PgturbohybridBm25MemoryContextAllocArray(memoryContext,
																	 sizeof(PgturbohybridBm25Posting),
																	 postings->count);
					else
						decodedScratch =
							PgturbohybridBm25RepallocArray(decodedScratch,
														   sizeof(PgturbohybridBm25Posting),
														   postings->count);
					decodedScratchCapacity = postings->count;
				}
				if (!PgturbohybridBm25DecodePostingsTuple(postings,
													 ItemIdGetLength(iid),
													 decodedScratch))
				{
					UnlockReleaseBuffer(buf);
					ereport(ERROR,
							(errcode(ERRCODE_DATA_CORRUPTED),
							 errmsg("pgturbohybrid BM25 postings payload is invalid")));
				}

				{
					for (uint16 i = 0; i < postings->count; i++)
					{
						PgturbohybridBm25Posting *posting = &decodedScratch[i];
						uint32		nodeId = posting->nodeId;

					if (nodeId >= graphMeta.tqNodeCount || docLens[nodeId] == 0 ||
						!liveNodes[nodeId])
						continue;

					PgturbohybridBm25AccumulatorAddTermScore(&acc, nodeId,
														PgturbohybridBm25PostingScoreDecoded(&bm25Meta,
																						idf,
																						avgDocLen,
																						posting,
																						docLens[nodeId]),
														term->matchBit);
					if (stats != NULL)
							stats->postingsDecoded++;
					}
				}

				UnlockReleaseBuffer(buf);
			postingsBlkno = nextBlkno;
			postingsOffno = nextOffno;
			CHECK_FOR_INTERRUPTS();
		}
	}
	}

	PgturbohybridBm25ScoreDelta(&bm25Meta, deltaTerms, deltaTermCount,
						   terms, termCount, &acc, stats);

	if (resolvedTerms == 0 || acc.touchedCount == 0)
	{
		PgturbohybridBm25PublishHotPostingsCacheStats(cache, stats);
		MemoryContextSwitchTo(oldCtx);
		return 0;
	}

	resultCount = PgturbohybridBm25SelectFinalTopK(&acc, query, terms, termCount,
											  &eval, (uint32) k,
											  memoryContext);

	if (resultCount == 0)
	{
		PgturbohybridBm25PublishHotPostingsCacheStats(cache, stats);
		MemoryContextSwitchTo(oldCtx);
		return 0;
	}

	*results = MemoryContextAllocZero(memoryContext,
									  sizeof(PgturbohybridBm25Result) * resultCount);
	for (int i = 0; i < resultCount; i++)
	{
		uint32		nodeId = acc.touched[i].nodeId;
		PgturbohybridBm25AccumulatorEntry *entry;

		entry = PgturbohybridBm25AccumulatorLookup(&acc, nodeId, false);
		if (entry == NULL)
			continue;

		(*results)[i].nodeId = nodeId;
		(*results)[i].heaptid = entry->hasDeltaDoc ? entry->heaptid :
			heapTids[nodeId];
		(*results)[i].bm25Score = acc.touched[i].score;
		(*results)[i].rank = i + 1;
	}

	if (stats != NULL)
	{
		if (stats->candidatesScored == 0)
			stats->candidatesScored = acc.touchedCount;
		stats->accumulatorEntries = acc.touchedCount;
		PgturbohybridBm25PublishHotPostingsCacheStats(cache, stats);
		pgturbohybrid_last_bm25_decode_kernel = stats->decodeKernel;
		pgturbohybrid_last_bm25_score_kernel = stats->scoreKernel;
		pgturbohybrid_last_bm25_simd_blocks = stats->simdBlocks;
		pgturbohybrid_last_bm25_scalar_tail_postings = stats->scalarTailPostings;
	}

	MemoryContextSwitchTo(oldCtx);
	return resultCount;
}
