/*
 * pgturbohybrid_sparse_query.c
 *
 * Scan path for the native sparse-vector inverted index (prompt 4).  Resolves
 * the query's sparse_query terms against the lexicon, exact-OR-accumulates
 * score[node] += query_weight * doc_weight over the per-term postings, applies
 * MVCC node liveness, and returns the top-k candidates by descending score.
 * Exact float32; no quantization / SIMD / WAND / cache.
 */
#include "postgres.h"

#include <string.h>

#include "miscadmin.h"
#include "portability/instr_time.h"
#include "storage/bufmgr.h"
#include "utils/float.h"
#include "utils/memutils.h"
#include "utils/rel.h"

#include "pgturbohybrid.h"
#include "pgturbohybrid_quant.h"
#include "pgturbohybrid_am.h"
#include "pgturbohybrid_query.h"
#include "pgturbohybrid_sparse.h"

typedef struct PgturbohybridSparseLexEntry
{
	int32		termId;
	uint32		df;
	float4		scale;
	float4		maxWeight;
	BlockNumber postingsBlkno;
	OffsetNumber postingsOffno;
	BlockNumber blockMaxBlkno;
	OffsetNumber blockMaxOffno;
	uint32		blockCount;
} PgturbohybridSparseLexEntry;

typedef struct PgturbohybridSparseScored
{
	uint32		nodeId;
	double		score;
} PgturbohybridSparseScored;

static int
PgturbohybridSparseLexCompare(const void *a, const void *b)
{
	const PgturbohybridSparseLexEntry *la = (const PgturbohybridSparseLexEntry *) a;
	const PgturbohybridSparseLexEntry *lb = (const PgturbohybridSparseLexEntry *) b;

	if (la->termId != lb->termId)
		return la->termId < lb->termId ? -1 : 1;
	return 0;
}

static int
PgturbohybridSparseScoredCompare(const void *a, const void *b)
{
	const PgturbohybridSparseScored *sa = (const PgturbohybridSparseScored *) a;
	const PgturbohybridSparseScored *sb = (const PgturbohybridSparseScored *) b;

	if (sa->score != sb->score)
		return sa->score > sb->score ? -1 : 1;	/* descending */
	if (sa->nodeId != sb->nodeId)
		return sa->nodeId < sb->nodeId ? -1 : 1;	/* deterministic tiebreak */
	return 0;
}

static int
PgturbohybridSparseQEntryCompare(const void *a, const void *b)
{
	const PgturbohybridSparseVectorEntry *ea = (const PgturbohybridSparseVectorEntry *) a;
	const PgturbohybridSparseVectorEntry *eb = (const PgturbohybridSparseVectorEntry *) b;

	if (ea->termId != eb->termId)
		return ea->termId < eb->termId ? -1 : 1;
	return 0;
}

/* Read the sparse meta tuple into *out; return false if no sparse data. */
bool
PgturbohybridSparseReadMeta(Relation index, BlockNumber metaBlkno,
							PgturbohybridSparseMetaTupleData *out)
{
	Buffer		buf;
	Page		page;
	PgturbohybridSparseMetaTuple tuple;

	if (metaBlkno == InvalidBlockNumber || metaBlkno == 0)
		return false;
	buf = ReadBuffer(index, metaBlkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	if (PageGetMaxOffsetNumber(page) < FirstOffsetNumber)
	{
		UnlockReleaseBuffer(buf);
		return false;
	}
	tuple = (PgturbohybridSparseMetaTuple)
		PageGetItem(page, PageGetItemId(page, FirstOffsetNumber));
	if (tuple->type != PGTURBOHYBRID_SPARSE_META_TUPLE_TYPE)
	{
		uint8		foundType = tuple->type;

		UnlockReleaseBuffer(buf);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("pgturbohybrid sparse meta tuple is malformed"),
				 errdetail("Expected sparse meta tuple type 0x%02X but found 0x%02X.",
						   PGTURBOHYBRID_SPARSE_META_TUPLE_TYPE, foundType),
				 errhint("REINDEX the index to rebuild the sparse inverted index.")));
	}
	if (tuple->sparseVersion != PGTURBOHYBRID_SPARSE_VERSION)
	{
		uint32		foundVersion = tuple->sparseVersion;

		UnlockReleaseBuffer(buf);
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("pgturbohybrid sparse index uses unsupported on-disk format version %u (expected %u)",
						foundVersion, PGTURBOHYBRID_SPARSE_VERSION),
				 errhint("REINDEX the index to rebuild it in the current sparse format.")));
	}
	memcpy(out, tuple, sizeof(*out));
	UnlockReleaseBuffer(buf);
	return true;
}

bool
PgturbohybridSparseEstimateMemory(Relation index,
								  struct PgturbohybridGraphMetaPageData *graphMeta,
								  PgturbohybridSparseMemoryEstimate *out)
{
	PgturbohybridSparseMetaTupleData meta;

	memset(out, 0, sizeof(*out));
	if (graphMeta == NULL ||
		graphMeta->tqSparseMetaStartBlkno == InvalidBlockNumber ||
		graphMeta->tqSparseMetaStartBlkno == 0)
		return false;
	if (!PgturbohybridSparseReadMeta(index, graphMeta->tqSparseMetaStartBlkno, &meta))
		return false;

	out->available = true;
	out->termCount = meta.termCount;
	out->docCount = meta.docCount;
	out->nodeCount = graphMeta->tqNodeCount;
	out->quantBits = (int) meta.quantBits;
	out->lexiconBytes = (uint64) meta.termCount * sizeof(PgturbohybridSparseLexiconEntry);
	out->heapTidsBytes = (uint64) graphMeta->tqNodeCount * sizeof(ItemPointerData);
	out->livenessBytes = (uint64) graphMeta->tqNodeCount * sizeof(bool);
	out->hotPostingsCacheMaxBytes =
		(uint64) pgturbohybrid_sparse_hot_postings_cache_mb * 1024 * 1024;
	out->totalBytesPerBackend = out->lexiconBytes + out->heapTidsBytes +
		out->livenessBytes + out->hotPostingsCacheMaxBytes;
	return true;
}

/* Load all lexicon entries into a palloc'd, term-id-sorted array. */
static PgturbohybridSparseLexEntry *
PgturbohybridSparseLoadLexicon(Relation index, BlockNumber start, uint32 termCount,
							   uint32 *outCount)
{
	PgturbohybridSparseLexEntry *entries;
	uint32		count = 0;
	BlockNumber blkno = start;

	entries = (PgturbohybridSparseLexEntry *)
		palloc(sizeof(PgturbohybridSparseLexEntry) * Max(termCount, 1u));

	while (blkno != InvalidBlockNumber)
	{
		Buffer		buf = ReadBuffer(index, blkno);
		Page		page;
		OffsetNumber maxoff;

		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off++)
		{
			PgturbohybridSparseLexiconTuple lt = (PgturbohybridSparseLexiconTuple)
				PageGetItem(page, PageGetItemId(page, off));

			if (lt->type != PGTURBOHYBRID_SPARSE_LEXICON_TUPLE_TYPE)
				continue;
			for (uint16 e = 0; e < lt->count && count < termCount; e++)
			{
				entries[count].termId = lt->entries[e].termId;
				entries[count].df = lt->entries[e].df;
				entries[count].scale = lt->entries[e].scale;
				entries[count].maxWeight = lt->entries[e].maxWeight;
				entries[count].postingsBlkno = lt->entries[e].postingsBlkno;
				entries[count].postingsOffno = lt->entries[e].postingsOffno;
				entries[count].blockMaxBlkno = lt->entries[e].blockMaxBlkno;
				entries[count].blockMaxOffno = lt->entries[e].blockMaxOffno;
				entries[count].blockCount = lt->entries[e].blockCount;
				count++;
			}
		}
		blkno = PgturbohybridGraphPageGetOpaque(page)->nextblkno;
		UnlockReleaseBuffer(buf);
	}

	if (count > 1)
		qsort(entries, count, sizeof(PgturbohybridSparseLexEntry),
			  PgturbohybridSparseLexCompare);
	*outCount = count;
	return entries;
}

/* ---- Backend-local sparse cache (prompt 10) ---------------------------- */

#define PGTURBOHYBRID_SPARSE_HOT_BUCKETS 1024

typedef struct PgturbohybridSparseHotChunk
{
	BlockNumber blkno;			/* chunk key */
	OffsetNumber offno;
	uint32		count;
	uint32	   *nodes;			/* decoded node_ids */
	float4	   *weights;		/* decoded raw weight values (q code or f32) */
	Size		bytes;
	struct PgturbohybridSparseHotChunk *hashNext;
	struct PgturbohybridSparseHotChunk *lruPrev;
	struct PgturbohybridSparseHotChunk *lruNext;
} PgturbohybridSparseHotChunk;

/* A merged delta posting (one term of one inserted row, exact f32 weight). */
typedef struct PgturbohybridSparseDeltaPosting
{
	uint32		nodeId;
	int32		termId;
	float4		weight;
} PgturbohybridSparseDeltaPosting;

typedef struct PgturbohybridSparseCacheRel
{
	Oid			relid;
	Oid			relfilenumber;
	/* coherence keys: any change drops the cached base/state/delta data */
	BlockNumber lexiconStartBlkno;
	uint32		coherentNodeCount;
	uint32		deltaGeneration;
	bool		coherenceSet;
	/* reader cache: lexicon + node states (heap TIDs + liveness) */
	PgturbohybridSparseLexEntry *lexicon;
	uint32		lexiconCount;
	bool		lexiconLoaded;
	PgturbohybridNodeState *states;
	uint32		stateCount;
	bool		statesLoaded;
	/* cached delta postings (decoded), keyed by deltaGeneration */
	PgturbohybridSparseDeltaPosting *delta;
	uint32		deltaCount;
	uint32		deltaDocs;
	bool		deltaLoaded;
	Size		readerBytes;
	/* hot decoded-postings cache (LRU, MB-capped) */
	PgturbohybridSparseHotChunk *buckets[PGTURBOHYBRID_SPARSE_HOT_BUCKETS];
	PgturbohybridSparseHotChunk *lruHead;	/* most recently used */
	PgturbohybridSparseHotChunk *lruTail;	/* least recently used */
	Size		hotBytes;
	uint64		hotEvictions;
	MemoryContext ctx;
	struct PgturbohybridSparseCacheRel *next;
} PgturbohybridSparseCacheRel;

static PgturbohybridSparseCacheRel *pgturbohybrid_sparse_cache_list = NULL;

static void
PgturbohybridSparseCacheFree(PgturbohybridSparseCacheRel *cache)
{
	PgturbohybridSparseCacheRel **link = &pgturbohybrid_sparse_cache_list;

	while (*link != NULL && *link != cache)
		link = &(*link)->next;
	if (*link == cache)
		*link = cache->next;
	MemoryContextDelete(cache->ctx);	/* frees lexicon/states/hot chunks */
}

void
PgturbohybridSparseCacheInvalidate(Oid relid)
{
	PgturbohybridSparseCacheRel *cache = pgturbohybrid_sparse_cache_list;

	while (cache != NULL)
	{
		PgturbohybridSparseCacheRel *next = cache->next;

		if (cache->relid == relid)
			PgturbohybridSparseCacheFree(cache);
		cache = next;
	}
}

/*
 * Find (or rebuild) the per-relation cache.  Any change to relfilenumber
 * (REINDEX), node count (insert), lexicon location (compaction) or delta
 * generation (insert/compaction) drops the cached base/state/delta/hot data so
 * stale results are never served.  Sets *hit when the cache was reused as-is.
 */
static PgturbohybridSparseCacheRel *
PgturbohybridSparseCacheAcquire(Relation index, uint32 nodeCount,
								BlockNumber lexiconStartBlkno, uint32 deltaGeneration,
								bool *hit)
{
	Oid			relid = RelationGetRelid(index);
	Oid			relfile = PgturbohybridGraphRelFileNumber(index);
	PgturbohybridSparseCacheRel *cache = pgturbohybrid_sparse_cache_list;
	MemoryContext ctx;
	MemoryContext oldCtx;

	*hit = false;
	for (; cache != NULL; cache = cache->next)
	{
		if (cache->relid != relid)
			continue;
		if (cache->relfilenumber == relfile && cache->coherenceSet &&
			cache->coherentNodeCount == nodeCount &&
			cache->lexiconStartBlkno == lexiconStartBlkno &&
			cache->deltaGeneration == deltaGeneration)
		{
			*hit = cache->lexiconLoaded && cache->statesLoaded && cache->deltaLoaded;
			return cache;
		}
		PgturbohybridSparseCacheFree(cache); /* stale: rebuild below */
		break;
	}

	ctx = AllocSetContextCreate(CacheMemoryContext, "pgturbohybrid sparse cache",
								ALLOCSET_DEFAULT_SIZES);
	oldCtx = MemoryContextSwitchTo(ctx);
	cache = (PgturbohybridSparseCacheRel *) palloc0(sizeof(PgturbohybridSparseCacheRel));
	cache->relid = relid;
	cache->relfilenumber = relfile;
	cache->coherentNodeCount = nodeCount;
	cache->lexiconStartBlkno = lexiconStartBlkno;
	cache->deltaGeneration = deltaGeneration;
	cache->coherenceSet = true;
	cache->ctx = ctx;
	cache->next = pgturbohybrid_sparse_cache_list;
	pgturbohybrid_sparse_cache_list = cache;
	MemoryContextSwitchTo(oldCtx);
	return cache;
}

/* Load (and cache) the delta chain's postings into the per-relation cache. */
static void
PgturbohybridSparseCacheGetDelta(PgturbohybridSparseCacheRel *cache, Relation index,
								 BlockNumber deltaStart, uint32 deltaDocCount,
								 bool *built, uint64 *buildUs)
{
	if (cache->deltaLoaded)
		return;
	{
		MemoryContext oldCtx = MemoryContextSwitchTo(cache->ctx);
		instr_time	t0,
					t1;
		uint32		capacity = Max(deltaDocCount, 1u) * 4;
		uint32		count = 0;
		BlockNumber blk = deltaStart;
		PgturbohybridSparseDeltaPosting *arr;

		INSTR_TIME_SET_CURRENT(t0);
		arr = (PgturbohybridSparseDeltaPosting *)
			palloc(sizeof(PgturbohybridSparseDeltaPosting) * capacity);
		while (blk != InvalidBlockNumber)
		{
			Buffer		buf = ReadBuffer(index, blk);
			Page		page;
			OffsetNumber maxoff;

			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);
			maxoff = PageGetMaxOffsetNumber(page);
			for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off++)
			{
				PgturbohybridSparseDeltaTuple dt = (PgturbohybridSparseDeltaTuple)
					PageGetItem(page, PageGetItemId(page, off));

				if (dt->type != PGTURBOHYBRID_SPARSE_DELTA_TUPLE_TYPE)
					continue;
				cache->deltaDocs++;
				for (uint16 e = 0; e < dt->termCount; e++)
				{
					if (count == capacity)
					{
						capacity *= 2;
						arr = (PgturbohybridSparseDeltaPosting *)
							repalloc(arr, sizeof(PgturbohybridSparseDeltaPosting) * capacity);
					}
					arr[count].nodeId = dt->nodeId;
					arr[count].termId = dt->entries[e].termId;
					arr[count].weight = dt->entries[e].weight;
					count++;
				}
			}
			blk = PgturbohybridGraphPageGetOpaque(page)->nextblkno;
			UnlockReleaseBuffer(buf);
		}
		cache->delta = arr;
		cache->deltaCount = count;
		cache->deltaLoaded = true;
		cache->readerBytes += (Size) count * sizeof(PgturbohybridSparseDeltaPosting);
		INSTR_TIME_SET_CURRENT(t1);
		INSTR_TIME_SUBTRACT(t1, t0);
		*buildUs += (uint64) INSTR_TIME_GET_MICROSEC(t1);
		*built = true;
		MemoryContextSwitchTo(oldCtx);
	}
}

/* LRU helpers for the hot-postings cache. */
static void
PgturbohybridSparseHotUnlink(PgturbohybridSparseCacheRel *cache,
							 PgturbohybridSparseHotChunk *c)
{
	if (c->lruPrev != NULL)
		c->lruPrev->lruNext = c->lruNext;
	else
		cache->lruHead = c->lruNext;
	if (c->lruNext != NULL)
		c->lruNext->lruPrev = c->lruPrev;
	else
		cache->lruTail = c->lruPrev;
	c->lruPrev = c->lruNext = NULL;
}

static void
PgturbohybridSparseHotPushFront(PgturbohybridSparseCacheRel *cache,
								PgturbohybridSparseHotChunk *c)
{
	c->lruPrev = NULL;
	c->lruNext = cache->lruHead;
	if (cache->lruHead != NULL)
		cache->lruHead->lruPrev = c;
	cache->lruHead = c;
	if (cache->lruTail == NULL)
		cache->lruTail = c;
}

static inline uint32
PgturbohybridSparseHotBucket(BlockNumber blkno, OffsetNumber offno)
{
	uint32		h = ((uint32) blkno * 2654435761u) ^ ((uint32) offno * 40503u);

	return h & (PGTURBOHYBRID_SPARSE_HOT_BUCKETS - 1);
}

static PgturbohybridSparseHotChunk *
PgturbohybridSparseHotFind(PgturbohybridSparseCacheRel *cache, BlockNumber blkno,
						   OffsetNumber offno)
{
	PgturbohybridSparseHotChunk *c =
		cache->buckets[PgturbohybridSparseHotBucket(blkno, offno)];

	for (; c != NULL; c = c->hashNext)
		if (c->blkno == blkno && c->offno == offno)
			return c;
	return NULL;
}

/* Evict least-recently-used hot chunks until under the budget. */
static void
PgturbohybridSparseHotEvictTo(PgturbohybridSparseCacheRel *cache, Size budget)
{
	while (cache->hotBytes > budget && cache->lruTail != NULL)
	{
		PgturbohybridSparseHotChunk *victim = cache->lruTail;
		uint32		b = PgturbohybridSparseHotBucket(victim->blkno, victim->offno);
		PgturbohybridSparseHotChunk **link = &cache->buckets[b];

		while (*link != NULL && *link != victim)
			link = &(*link)->hashNext;
		if (*link == victim)
			*link = victim->hashNext;
		PgturbohybridSparseHotUnlink(cache, victim);
		cache->hotBytes -= victim->bytes;
		cache->hotEvictions++;
		pfree(victim->nodes);
		pfree(victim->weights);
		pfree(victim);
	}
}

/* Reader cache: cached lexicon (loaded once per relfilenumber). */
static PgturbohybridSparseLexEntry *
PgturbohybridSparseCacheGetLexicon(PgturbohybridSparseCacheRel *cache, Relation index,
								   PgturbohybridSparseMetaTuple meta, uint32 *count,
								   bool *built, uint64 *buildUs)
{
	if (!cache->lexiconLoaded)
	{
		MemoryContext oldCtx = MemoryContextSwitchTo(cache->ctx);
		instr_time	t0,
					t1;

		INSTR_TIME_SET_CURRENT(t0);
		cache->lexicon = PgturbohybridSparseLoadLexicon(index, meta->lexiconStartBlkno,
														meta->termCount,
														&cache->lexiconCount);
		cache->lexiconLoaded = true;
		cache->readerBytes += (Size) cache->lexiconCount *
			sizeof(PgturbohybridSparseLexEntry);
		INSTR_TIME_SET_CURRENT(t1);
		INSTR_TIME_SUBTRACT(t1, t0);
		*buildUs += (uint64) INSTR_TIME_GET_MICROSEC(t1);
		*built = true;
		MemoryContextSwitchTo(oldCtx);
	}
	*count = cache->lexiconCount;
	return cache->lexicon;
}

/* Reader cache: cached node states (heap TIDs + liveness). */
static PgturbohybridNodeState *
PgturbohybridSparseCacheGetStates(PgturbohybridSparseCacheRel *cache, Relation index,
								  PgturbohybridGraphMetaPageData *graphMeta,
								  uint32 *count, bool *built, uint64 *buildUs)
{
	if (!cache->statesLoaded)
	{
		MemoryContext oldCtx = MemoryContextSwitchTo(cache->ctx);
		instr_time	t0,
					t1;

		INSTR_TIME_SET_CURRENT(t0);
		cache->states = PgturbohybridReadNodeStates(index, graphMeta,
													&cache->stateCount);
		cache->statesLoaded = true;
		cache->readerBytes += (Size) cache->stateCount * sizeof(PgturbohybridNodeState);
		INSTR_TIME_SET_CURRENT(t1);
		INSTR_TIME_SUBTRACT(t1, t0);
		*buildUs += (uint64) INSTR_TIME_GET_MICROSEC(t1);
		*built = true;
		MemoryContextSwitchTo(oldCtx);
	}
	*count = cache->stateCount;
	return cache->states;
}

/* Dequantize+accumulate one decoded posting into scores[]. */
/*
 * Decode + accumulate a single postings chunk.  effMul is the effective
 * per-posting multiplier already folding in the dequant scale (effMul = qWeight
 * for f32, qWeight*scale for q8/q16).  SoA chunks route through the SIMD-capable
 * kernel dispatch; varint chunks decode node deltas with a scalar walk.
 */
static void
PgturbohybridSparseScoreChunk(PgturbohybridSparsePostingsTuple ct, int bits,
							  double effMul, int scoreKernel, double *scores,
							  uint32 nodeCount, uint64 *simdBlocks,
							  uint64 *scalarTail)
{
	uint32		n = ct->count;
	uint32		base = ct->baseNodeId;
	uint32		wwidth = PgturbohybridSparseWeightWidth(bits);
	const char *payload = ct->payload;

	/*
	 * A chunk physically holds at most PGTURBOHYBRID_SPARSE_MAX_BLOCK_SIZE
	 * postings (the writer caps every chunk at the index block size, which is
	 * itself <= this ceiling).  A larger count can only be corruption, and the
	 * bit-packed branch below indexes a fixed stack array of exactly this size,
	 * so reject an over-large count rather than overrunning it.  Cannot fire for
	 * a validly-built index.
	 */
	if (n > PGTURBOHYBRID_SPARSE_MAX_BLOCK_SIZE)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("pgturbohybrid sparse postings chunk has invalid posting count %u (max %u)",
						n, (uint32) PGTURBOHYBRID_SPARSE_MAX_BLOCK_SIZE),
				 errhint("REINDEX the index to rebuild the sparse inverted index.")));

	if (ct->encoding == PGTURBOHYBRID_SPARSE_ENCODING_VARINT)
	{
		uint16		deltaBytes;
		Size		weightsStart;
		const uint8 *deltas = (const uint8 *) (payload + sizeof(uint16));
		int			pos = 0;
		uint32		node = base;

		memcpy(&deltaBytes, payload, sizeof(uint16));
		weightsStart = PgturbohybridSparseAlignUp(sizeof(uint16) + (Size) deltaBytes,
												  wwidth);
		for (uint32 k = 0; k < n; k++)
		{
			int			consumed;
			const char *wptr = payload + weightsStart + (Size) k * wwidth;

			node += PgturbohybridSparseVarintDecode(deltas + pos, &consumed);
			pos += consumed;
			if (node < nodeCount)
			{
				if (bits == 0)
				{
					float4		w;

					memcpy(&w, wptr, sizeof(float4));
					scores[node] += effMul * (double) w;
				}
				else if (bits == 16)
				{
					uint16		q;

					memcpy(&q, wptr, sizeof(uint16));
					scores[node] += effMul * (double) q;
				}
				else
					scores[node] += effMul * (double) (*(const uint8 *) wptr);
			}
		}
		*scalarTail += n;		/* varint node decode is inherently scalar */
	}
	else if (ct->encoding == PGTURBOHYBRID_SPARSE_ENCODING_BITPACKED)
	{
		/* Bit-unpack node-delta offsets, then reuse the SIMD scatter scorer. */
		uint16		offsets[PGTURBOHYBRID_SPARSE_MAX_BLOCK_SIZE];
		int			numBits = (int) (uint8) payload[0];
		Size		deltaBytes = PgturbohybridSparseBitPackedBytes(n, numBits);
		Size		weightsStart = PgturbohybridSparseAlignUp(1 + deltaBytes, wwidth);

		Assert(n <= PGTURBOHYBRID_SPARSE_MAX_BLOCK_SIZE);
		PgturbohybridSparseBitUnpack((const uint8 *) payload + 1, n, numBits, offsets);
		PgturbohybridSparseScoreSoa(scoreKernel, payload + weightsStart, offsets, n,
									base, bits, effMul, scores, nodeCount, simdBlocks,
									scalarTail);
	}
	else						/* SoA */
	{
		Size		offsetsStart = PgturbohybridSparseAlignUp((Size) n * wwidth, 2);
		const uint16 *offsets = (const uint16 *) (payload + offsetsStart);

		PgturbohybridSparseScoreSoa(scoreKernel, payload, offsets, n, base, bits,
									effMul, scores, nodeCount, simdBlocks,
									scalarTail);
	}
}

/*
 * Accumulate one term's postings into scores[]: physically walk the term's
 * chunks from (blkno, offno) until df postings are consumed, dequantizing per
 * the index's bit width.  effMul folds the per-term dequant scale into the
 * query weight (= qWeight for f32, qWeight*scale otherwise).
 */
static uint64
PgturbohybridSparseAccumulateTerm(Relation index, BlockNumber blkno,
								  OffsetNumber offno, uint32 df, double qWeight,
								  double scale, int bits, int scoreKernel,
								  double *scores, uint32 nodeCount,
								  uint64 *simdBlocks, uint64 *scalarTail)
{
	uint32		remaining = df;
	uint64		touched = 0;
	double		effMul = bits == 0 ? qWeight : qWeight * scale;

	while (remaining > 0 && blkno != InvalidBlockNumber)
	{
		Buffer		buf = ReadBuffer(index, blkno);
		Page		page;
		OffsetNumber maxoff;

		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		maxoff = PageGetMaxOffsetNumber(page);

		while (remaining > 0 && offno <= maxoff)
		{
			PgturbohybridSparsePostingsTuple ct = (PgturbohybridSparsePostingsTuple)
				PageGetItem(page, PageGetItemId(page, offno));

			if (ct->type == PGTURBOHYBRID_SPARSE_POSTINGS_TUPLE_TYPE)
			{
				PgturbohybridSparseScoreChunk(ct, bits, effMul, scoreKernel,
											  scores, nodeCount, simdBlocks,
											  scalarTail);
				touched += ct->count;
				remaining -= Min(remaining, (uint32) ct->count);
			}
			offno++;
		}

		blkno = PgturbohybridGraphPageGetOpaque(page)->nextblkno;
		offno = FirstOffsetNumber;
		UnlockReleaseBuffer(buf);
	}
	return touched;
}

/* ---- Block-max WAND (prompt 9) ----------------------------------------- */

#define PGTURBOHYBRID_WAND_INF PG_UINT32_MAX

typedef struct PgturbohybridWandTerm
{
	double		effMul;			/* qWeight (f32) or qWeight*scale (quantized) */
	double		termMaxUB;		/* qWeight * lexicon maxWeight */
	int			bits;
	uint32		df;				/* term document frequency (for hot-cache gating) */
	PgturbohybridSparseCacheRel *cache; /* backend-local cache (or NULL) */
	PgturbohybridSparseBlockMax *blocks;
	uint32		nBlocks;
	uint32		curBlock;
	uint32	   *nodes;			/* decoded node_ids of the loaded block */
	float4	   *weights;		/* decoded raw weight values of the loaded block */
	double	   *contribs;		/* effMul * weights of the loaded block */
	uint32		chunkCount;
	uint32		chunkPos;
	bool		loaded;
	uint32		curNode;		/* PGTURBOHYBRID_WAND_INF when exhausted */
} PgturbohybridWandTerm;

typedef struct PgturbohybridWandHeapEntry
{
	double		score;
	uint32		nodeId;
	ItemPointerData heaptid;
} PgturbohybridWandHeapEntry;

static inline double
PgturbohybridWandWeight(const char *wptr, int bits)
{
	if (bits == 0)
	{
		float4		w;

		memcpy(&w, wptr, sizeof(float4));
		return (double) w;
	}
	if (bits == 16)
	{
		uint16		q;

		memcpy(&q, wptr, sizeof(uint16));
		return (double) q;
	}
	return (double) (*(const uint8 *) wptr);
}

/* Decode a postings chunk at (blk,off) into nodes[]/weights[] (raw weight values). */
static void
PgturbohybridWandDecodeChunk(Relation index, BlockNumber blk, OffsetNumber off,
							 int bits, uint32 *nodes, float4 *weights,
							 uint32 *outCount)
{
	Buffer		buf = ReadBuffer(index, blk);
	Page		page;
	PgturbohybridSparsePostingsTuple ct;
	uint32		n;
	uint32		base;
	uint32		wwidth = PgturbohybridSparseWeightWidth(bits);
	const char *payload;

	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	ct = (PgturbohybridSparsePostingsTuple) PageGetItem(page, PageGetItemId(page, off));
	n = ct->count;
	base = ct->baseNodeId;
	payload = ct->payload;

	/*
	 * The caller's nodes[]/weights[] buffers are sized to hold at most
	 * PGTURBOHYBRID_SPARSE_MAX_BLOCK_SIZE postings (the physical chunk ceiling),
	 * and the bit-packed branch indexes a stack array of exactly that size.  A
	 * larger count is corruption -- reject it instead of writing out of bounds.
	 * Cannot fire for a validly-built index.
	 */
	if (n > PGTURBOHYBRID_SPARSE_MAX_BLOCK_SIZE)
	{
		UnlockReleaseBuffer(buf);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("pgturbohybrid sparse postings chunk has invalid posting count %u (max %u)",
						n, (uint32) PGTURBOHYBRID_SPARSE_MAX_BLOCK_SIZE),
				 errhint("REINDEX the index to rebuild the sparse inverted index.")));
	}

	if (ct->encoding == PGTURBOHYBRID_SPARSE_ENCODING_VARINT)
	{
		uint16		deltaBytes;
		Size		weightsStart;
		const uint8 *deltas = (const uint8 *) (payload + sizeof(uint16));
		int			pos = 0;
		uint32		node = base;

		memcpy(&deltaBytes, payload, sizeof(uint16));
		weightsStart = PgturbohybridSparseAlignUp(sizeof(uint16) + (Size) deltaBytes,
												  wwidth);
		for (uint32 k = 0; k < n; k++)
		{
			int			consumed;

			node += PgturbohybridSparseVarintDecode(deltas + pos, &consumed);
			pos += consumed;
			nodes[k] = node;
			weights[k] = (float4)
				PgturbohybridWandWeight(payload + weightsStart + (Size) k * wwidth, bits);
		}
	}
	else if (ct->encoding == PGTURBOHYBRID_SPARSE_ENCODING_BITPACKED)
	{
		uint16		offsets[PGTURBOHYBRID_SPARSE_MAX_BLOCK_SIZE];
		int			numBits = (int) (uint8) payload[0];
		Size		deltaBytes = PgturbohybridSparseBitPackedBytes(n, numBits);
		Size		weightsStart = PgturbohybridSparseAlignUp(1 + deltaBytes, wwidth);

		Assert(n <= PGTURBOHYBRID_SPARSE_MAX_BLOCK_SIZE);
		PgturbohybridSparseBitUnpack((const uint8 *) payload + 1, n, numBits, offsets);
		for (uint32 k = 0; k < n; k++)
		{
			nodes[k] = base + offsets[k];
			weights[k] = (float4)
				PgturbohybridWandWeight(payload + weightsStart + (Size) k * wwidth, bits);
		}
	}
	else						/* SoA */
	{
		Size		offsetsStart = PgturbohybridSparseAlignUp((Size) n * wwidth, 2);

		for (uint32 k = 0; k < n; k++)
		{
			uint16		o;

			memcpy(&o, payload + offsetsStart + (Size) k * sizeof(uint16), sizeof(uint16));
			nodes[k] = base + o;
			weights[k] = (float4)
				PgturbohybridWandWeight(payload + (Size) k * wwidth, bits);
		}
	}
	*outCount = n;
	UnlockReleaseBuffer(buf);
}

/*
 * Fill nodes[]/weights[] for the chunk at (blk,off), serving from the hot-postings
 * cache when present.  On a miss the chunk is decoded from disk and, for terms with
 * df >= min_df and a non-zero cache budget, inserted into the LRU (evicting to the
 * MB cap).  Returns true if served from cache.
 */
static bool
PgturbohybridSparseHotLoadChunk(PgturbohybridSparseCacheRel *cache, Relation index,
								BlockNumber blk, OffsetNumber off, uint32 df, int bits,
								uint32 *nodes, float4 *weights, uint32 *count)
{
	Size		budget = (Size) pgturbohybrid_sparse_hot_postings_cache_mb * 1024 * 1024;
	PgturbohybridSparseHotChunk *c;

	if (cache != NULL && budget > 0)
	{
		c = PgturbohybridSparseHotFind(cache, blk, off);
		if (c != NULL)
		{
			memcpy(nodes, c->nodes, sizeof(uint32) * c->count);
			memcpy(weights, c->weights, sizeof(float4) * c->count);
			*count = c->count;
			PgturbohybridSparseHotUnlink(cache, c);
			PgturbohybridSparseHotPushFront(cache, c);
			return true;
		}
	}

	PgturbohybridWandDecodeChunk(index, blk, off, bits, nodes, weights, count);

	if (cache != NULL && budget > 0 &&
		df >= (uint32) pgturbohybrid_sparse_hot_postings_cache_min_df && *count > 0)
	{
		MemoryContext oldCtx = MemoryContextSwitchTo(cache->ctx);
		uint32		b = PgturbohybridSparseHotBucket(blk, off);

		c = (PgturbohybridSparseHotChunk *) palloc0(sizeof(PgturbohybridSparseHotChunk));
		c->blkno = blk;
		c->offno = off;
		c->count = *count;
		c->nodes = (uint32 *) palloc(sizeof(uint32) * *count);
		c->weights = (float4 *) palloc(sizeof(float4) * *count);
		memcpy(c->nodes, nodes, sizeof(uint32) * *count);
		memcpy(c->weights, weights, sizeof(float4) * *count);
		c->bytes = sizeof(PgturbohybridSparseHotChunk) +
			(Size) *count * (sizeof(uint32) + sizeof(float4));
		c->hashNext = cache->buckets[b];
		cache->buckets[b] = c;
		PgturbohybridSparseHotPushFront(cache, c);
		cache->hotBytes += c->bytes;
		PgturbohybridSparseHotEvictTo(cache, budget);
		MemoryContextSwitchTo(oldCtx);
	}
	return false;
}

/* Load a term's block-max directory (blockCount entries) by physical walk. */
static PgturbohybridSparseBlockMax *
PgturbohybridWandLoadBlockMax(Relation index, BlockNumber blk, OffsetNumber off,
							  uint32 count)
{
	PgturbohybridSparseBlockMax *out = (PgturbohybridSparseBlockMax *)
		palloc(sizeof(PgturbohybridSparseBlockMax) * Max(count, 1u));
	uint32		got = 0;

	while (got < count && blk != InvalidBlockNumber)
	{
		Buffer		buf = ReadBuffer(index, blk);
		Page		page;
		OffsetNumber maxoff;

		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		maxoff = PageGetMaxOffsetNumber(page);
		while (got < count && off <= maxoff)
		{
			PgturbohybridSparseBlockMaxTuple bt = (PgturbohybridSparseBlockMaxTuple)
				PageGetItem(page, PageGetItemId(page, off));

			if (bt->type == PGTURBOHYBRID_SPARSE_BLOCKMAX_TUPLE_TYPE)
			{
				uint32		take = (uint32) Min((uint64) (count - got), (uint64) bt->count);

				memcpy(out + got, bt->entries,
					   sizeof(PgturbohybridSparseBlockMax) * take);
				got += take;
			}
			off++;
		}
		blk = PgturbohybridGraphPageGetOpaque(page)->nextblkno;
		off = FirstOffsetNumber;
		UnlockReleaseBuffer(buf);
	}
	return out;
}

/*
 * Advance a term iterator to the first node_id >= target, skipping whole blocks
 * via the block-max directory (without reading their postings).  *visited and
 * *skipped accumulate chunk reads / block skips.
 */
static void
PgturbohybridWandAdvance(Relation index, PgturbohybridWandTerm *t, uint32 target,
						 int bits, uint64 *visited, uint64 *skipped,
						 uint64 *hotHits, uint64 *hotMisses)
{
	while (t->curBlock < t->nBlocks &&
		   t->blocks[t->curBlock].lastNodeId < target)
	{
		t->curBlock++;
		if (t->loaded)
			t->loaded = false;
		(*skipped)++;
	}
	if (t->curBlock >= t->nBlocks)
	{
		t->curNode = PGTURBOHYBRID_WAND_INF;
		return;
	}
	if (!t->loaded)
	{
		bool		hit = PgturbohybridSparseHotLoadChunk(t->cache, index,
														  t->blocks[t->curBlock].postingsBlkno,
														  t->blocks[t->curBlock].postingsOffno,
														  t->df, bits, t->nodes,
														  t->weights, &t->chunkCount);

		for (uint32 k = 0; k < t->chunkCount; k++)
			t->contribs[k] = t->effMul * (double) t->weights[k];
		t->chunkPos = 0;
		t->loaded = true;
		(*visited)++;
		if (hit)
			(*hotHits)++;
		else
			(*hotMisses)++;
	}
	while (t->chunkPos < t->chunkCount && t->nodes[t->chunkPos] < target)
		t->chunkPos++;
	if (t->chunkPos >= t->chunkCount)
	{
		/* Past this chunk: move to the next block and retry. */
		t->curBlock++;
		t->loaded = false;
		PgturbohybridWandAdvance(index, t, target, bits, visited, skipped,
								 hotHits, hotMisses);
		return;
	}
	t->curNode = t->nodes[t->chunkPos];
}

static void
PgturbohybridWandHeapSiftUp(PgturbohybridWandHeapEntry *heap, int i)
{
	while (i > 0)
	{
		int			parent = (i - 1) / 2;

		if (heap[parent].score <= heap[i].score)
			break;
		{
			PgturbohybridWandHeapEntry tmp = heap[parent];

			heap[parent] = heap[i];
			heap[i] = tmp;
		}
		i = parent;
	}
}

static void
PgturbohybridWandHeapSiftDown(PgturbohybridWandHeapEntry *heap, int size, int i)
{
	for (;;)
	{
		int			l = 2 * i + 1;
		int			r = 2 * i + 2;
		int			small = i;

		if (l < size && heap[l].score < heap[small].score)
			small = l;
		if (r < size && heap[r].score < heap[small].score)
			small = r;
		if (small == i)
			break;
		{
			PgturbohybridWandHeapEntry tmp = heap[small];

			heap[small] = heap[i];
			heap[i] = tmp;
		}
		i = small;
	}
}

/*
 * Block-max WAND top-k collection.  Returns the count; fills *out (palloc'd in
 * the current context) sorted by descending score.  Exact: never drops a true
 * top-k candidate (pruning only skips documents whose summed block upper bound
 * cannot exceed the current k-th score).
 */
static int
PgturbohybridSparseCollectWand(Relation index, PgturbohybridSparseMetaTuple meta,
							   PgturbohybridSparseCacheRel *cache,
							   PgturbohybridSparseLexEntry *lexicon, uint32 lexCount,
							   const PgturbohybridSparseVectorEntry *qEntries,
							   uint32 qCount, PgturbohybridNodeState *states,
							   uint32 nodeCount, int k,
							   PgturbohybridSparseCandidate **out,
							   PgturbohybridSparseScanStats *stats)
{
	PgturbohybridWandTerm *terms;
	PgturbohybridWandTerm **order;
	int			numTerms = 0;
	int			bits = (int) meta->quantBits;
	uint32		blockSize = meta->blockSize > 0 ? meta->blockSize :
		PGTURBOHYBRID_SPARSE_DEFAULT_BLOCK_SIZE;
	uint32		bufCap = Max(blockSize, (uint32) PGTURBOHYBRID_SPARSE_MAX_BLOCK_SIZE);
	PgturbohybridWandHeapEntry *heap;
	int			heapSize = 0;
	double		theta = -get_float8_infinity();
	uint64		visited = 0,
				skipped = 0,
				pruned = 0,
				iterations = 0,
				thresholdUpdates = 0,
				heapUpdates = 0,
				docsScored = 0,
				hotHits = 0,
				hotMisses = 0;
	uint32		resolved = 0;
	PgturbohybridSparseCandidate *result;
	int			resultCount;

	terms = (PgturbohybridWandTerm *)
		palloc0(sizeof(PgturbohybridWandTerm) * Max(qCount, 1u));
	for (uint32 i = 0; i < qCount; i++)
	{
		PgturbohybridSparseLexEntry key;
		PgturbohybridSparseLexEntry *found;
		PgturbohybridWandTerm *t;

		if (qEntries[i].weight == 0.0f)
			continue;
		key.termId = qEntries[i].termId;
		found = (PgturbohybridSparseLexEntry *)
			bsearch(&key, lexicon, lexCount, sizeof(PgturbohybridSparseLexEntry),
					PgturbohybridSparseLexCompare);
		if (found == NULL || found->blockCount == 0 ||
			found->blockMaxBlkno == InvalidBlockNumber)
			continue;
		resolved++;
		t = &terms[numTerms++];
		t->effMul = bits == 0 ? (double) qEntries[i].weight :
			(double) qEntries[i].weight * (double) found->scale;
		/*
		 * Term upper bound = effMul * max quantized code, which equals the exact
		 * maximum contribution any posting of this term can make (round() is
		 * monotonic, so the max-weight posting has the max code).  Using the raw
		 * maxWeight would be unsafe for quantized scores (rounding can exceed it).
		 */
		t->termMaxUB = t->effMul * (bits == 0 ? (double) found->maxWeight :
									(double) PgturbohybridSparseQuantize(found->maxWeight,
																		 found->scale, bits));
		t->bits = bits;
		t->df = found->df;
		t->cache = cache;
		t->nBlocks = found->blockCount;
		t->blocks = PgturbohybridWandLoadBlockMax(index, found->blockMaxBlkno,
												  found->blockMaxOffno,
												  found->blockCount);
		t->curBlock = 0;
		t->loaded = false;
		/*
		 * Size the decode scratch to the physical chunk ceiling, not the
		 * (on-disk, attacker-influenced) meta blockSize: PgturbohybridWandDecodeChunk
		 * only admits chunks of up to PGTURBOHYBRID_SPARSE_MAX_BLOCK_SIZE postings,
		 * so this guarantees the buffers can hold any chunk it accepts.  For a
		 * validly-built index blockSize <= this ceiling, so this never shrinks
		 * the buffers and changes no behavior.
		 */
		t->nodes = (uint32 *) palloc(sizeof(uint32) * bufCap);
		t->weights = (float4 *) palloc(sizeof(float4) * bufCap);
		t->contribs = (double *) palloc(sizeof(double) * bufCap);
		PgturbohybridWandAdvance(index, t, 0, bits, &visited, &skipped,
								 &hotHits, &hotMisses);
	}

	heap = (PgturbohybridWandHeapEntry *)
		palloc(sizeof(PgturbohybridWandHeapEntry) * Max(k, 1));
	order = (PgturbohybridWandTerm **)
		palloc(sizeof(PgturbohybridWandTerm *) * Max(numTerms, 1));
	for (int i = 0; i < numTerms; i++)
		order[i] = &terms[i];

	while (numTerms > 0)
	{
		int			pivotIdx = -1;
		uint32		pivotNode;
		double		ub = 0.0;

		iterations++;
		CHECK_FOR_INTERRUPTS();

		/* Sort iterators by current node_id (exhausted = INF, sort last). */
		for (int a = 1; a < numTerms; a++)
		{
			PgturbohybridWandTerm *key = order[a];
			int			b = a - 1;

			while (b >= 0 && order[b]->curNode > key->curNode)
			{
				order[b + 1] = order[b];
				b--;
			}
			order[b + 1] = key;
		}

		for (int i = 0; i < numTerms; i++)
		{
			if (order[i]->curNode == PGTURBOHYBRID_WAND_INF)
				break;
			ub += order[i]->termMaxUB;
			if (ub > theta)
			{
				pivotIdx = i;
				break;
			}
		}
		if (pivotIdx < 0)
			break;				/* no remaining document can exceed theta */
		pivotNode = order[pivotIdx]->curNode;

		if (order[0]->curNode == pivotNode)
		{
			double		score = 0.0;

			/*
			 * Sum every term positioned at pivotNode -- ties at the pivot can
			 * sort beyond pivotIdx, so iterate all terms (not just [0,pivotIdx]).
			 */
			for (int i = 0; i < numTerms; i++)
				if (order[i]->curNode == pivotNode)
					score += order[i]->contribs[order[i]->chunkPos];
			for (int i = 0; i < numTerms; i++)
				if (order[i]->curNode == pivotNode)
					PgturbohybridWandAdvance(index, order[i], pivotNode + 1, bits,
											 &visited, &skipped, &hotHits, &hotMisses);
			docsScored++;

			if (pivotNode < nodeCount && states[pivotNode].live)
			{
				if (heapSize < k)
				{
					heap[heapSize].score = score;
					heap[heapSize].nodeId = pivotNode;
					heap[heapSize].heaptid = states[pivotNode].tid;
					PgturbohybridWandHeapSiftUp(heap, heapSize);
					heapSize++;
					heapUpdates++;
					if (heapSize == k)
					{
						theta = heap[0].score;
						thresholdUpdates++;
					}
				}
				else if (score > heap[0].score)
				{
					heap[0].score = score;
					heap[0].nodeId = pivotNode;
					heap[0].heaptid = states[pivotNode].tid;
					PgturbohybridWandHeapSiftDown(heap, heapSize, 0);
					heapUpdates++;
					if (heap[0].score > theta)
					{
						theta = heap[0].score;
						thresholdUpdates++;
					}
				}
			}
		}
		else
		{
			/* Advance a lagging term before the pivot to >= pivotNode. */
			for (int i = 0; i < pivotIdx; i++)
				if (order[i]->curNode < pivotNode)
				{
					PgturbohybridWandAdvance(index, order[i], pivotNode, bits,
											 &visited, &skipped, &hotHits, &hotMisses);
					break;
				}
			pruned++;
		}
	}

	/* Emit the heap as candidates sorted by descending score. */
	resultCount = heapSize;
	result = resultCount > 0 ? (PgturbohybridSparseCandidate *)
		palloc(sizeof(PgturbohybridSparseCandidate) * resultCount) : NULL;
	for (int i = 0; i < heapSize; i++)
	{
		result[i].nodeId = heap[i].nodeId;
		result[i].heaptid = heap[i].heaptid;
		result[i].score = heap[i].score;
	}
	if (resultCount > 1)
	{
		/* Reuse the scored comparator (descending score, node_id tiebreak). */
		PgturbohybridSparseScored *tmp = (PgturbohybridSparseScored *)
			palloc(sizeof(PgturbohybridSparseScored) * resultCount);

		for (int i = 0; i < resultCount; i++)
		{
			tmp[i].nodeId = result[i].nodeId;
			tmp[i].score = result[i].score;
		}
		qsort(tmp, resultCount, sizeof(PgturbohybridSparseScored),
			  PgturbohybridSparseScoredCompare);
		for (int i = 0; i < resultCount; i++)
		{
			result[i].nodeId = tmp[i].nodeId;
			result[i].score = tmp[i].score;
			result[i].heaptid = states[tmp[i].nodeId].tid;
		}
		pfree(tmp);
	}

	if (stats != NULL)
	{
		stats->branchUsed = true;
		stats->usedWand = true;
		stats->terms = qCount;
		stats->resolvedTerms = resolved;
		stats->candidatesScored = docsScored;	/* documents fully evaluated */
		stats->blocksVisited = visited;
		stats->blocksSkipped = skipped;
		stats->wandPruned = pruned;
		stats->wandIterations = iterations;
		stats->wandThresholdUpdates = thresholdUpdates;
		stats->wandHeapUpdates = heapUpdates;
		stats->quantBits = bits;
		stats->quantMode = (int) meta->quantMode;
		stats->encoding = (int) meta->postingsEncoding;
		stats->scoreKernel = PGTURBOHYBRID_SPARSE_SCORE_SCALAR;
		stats->hotCacheHits = hotHits;
		stats->hotCacheMisses = hotMisses;
		if (cache != NULL)
			stats->hotCacheBytes = cache->hotBytes;
	}

	if (out != NULL)
		*out = result;
	return resultCount;
}

int
PgturbohybridSparseCollectCandidates(Relation index, PgturbohybridQueryHeader *query,
									 int k, bool simdEnabled, bool wandEnabled,
									 PgturbohybridSparseCandidate **out,
									 MemoryContext ctx, PgturbohybridSparseScanStats *stats)
{
	PgturbohybridGraphMetaPageData graphMeta;
	PgturbohybridSparseMetaTupleData meta;
	PgturbohybridSparseLexEntry *lexicon;
	uint32		lexCount = 0;
	struct varlena *querySparse;
	const PgturbohybridSparseVectorEntry *qEntries;
	uint32		qCount = 0;
	uint32		resolved = 0;
	uint64		postingsTouched = 0;
	uint64		simdBlocks = 0;
	uint64		scalarTail = 0;
	int			scoreKernel;
	PgturbohybridNodeState *states;
	uint32		nodeCount = 0;
	double	   *scores;
	PgturbohybridSparseScored *scored;
	uint64		scoredCount = 0;
	MemoryContext oldCtx;
	int			resultCount;
	PgturbohybridSparseCandidate *result;
	PgturbohybridSparseCacheRel *cache;
	bool		cacheBuilt = false;
	uint64		cacheBuildUs = 0;
	instr_time	t0,
				t1;

	if (out != NULL)
		*out = NULL;
	if (stats != NULL)
		memset(stats, 0, sizeof(*stats));
	if (k <= 0 || query == NULL || !PgturbohybridQueryHasSparse(query))
		return 0;

	INSTR_TIME_SET_CURRENT(t0);

	if (!PgturbohybridGraphReadMeta(index, &graphMeta))
		return 0;
	if (!PgturbohybridSparseReadMeta(index, graphMeta.tqSparseMetaStartBlkno, &meta))
		return 0;
	if (stats != NULL)
		stats->branchAvailable = true;
	scoreKernel = PgturbohybridSparseResolveScoreKernel((int) meta.quantBits,
														simdEnabled);

	querySparse = PgturbohybridQueryGetSparseVector(query);
	if (querySparse == NULL)
		return 0;
	qEntries = PgturbohybridSparseVectorData(querySparse, &qCount);
	if (qCount == 0)
		return 0;

	/* Reader cache: lexicon + node states + delta, keyed by relfilenumber and
	 * invalidated on node-count / lexicon-location / delta-generation changes. */
	{
		bool		readerHit;

		cache = PgturbohybridSparseCacheAcquire(index, graphMeta.tqNodeCount,
												meta.lexiconStartBlkno,
												meta.deltaGeneration, &readerHit);
		(void) readerHit;
	}

	oldCtx = MemoryContextSwitchTo(ctx);

	lexicon = PgturbohybridSparseCacheGetLexicon(cache, index, &meta, &lexCount,
												 &cacheBuilt, &cacheBuildUs);
	states = PgturbohybridSparseCacheGetStates(cache, index, &graphMeta, &nodeCount,
											   &cacheBuilt, &cacheBuildUs);
	if (meta.deltaDocCount > 0)
		PgturbohybridSparseCacheGetDelta(cache, index, meta.deltaStartBlkno,
										 meta.deltaDocCount, &cacheBuilt, &cacheBuildUs);
	if (stats != NULL)
	{
		stats->cacheHit = !cacheBuilt;
		stats->cacheBuildUs = cacheBuildUs;
		stats->cacheBytes = cache->readerBytes;
		stats->deltaGeneration = meta.deltaGeneration;
		stats->deltaCacheHit = meta.deltaDocCount > 0 && !cacheBuilt;
	}

	/*
	 * Block-max WAND path (prompt 9): exact top-k with safe pruning, when the
	 * index has a block-max directory, WAND is enabled, and there are no pending
	 * deltas (deltas are not in the block-max structure, so they force exact
	 * accumulation).  Otherwise fall back to the exact OR-accumulation.
	 */
	if (wandEnabled && meta.hasBlockMax && meta.deltaDocCount == 0)
	{
		resultCount = PgturbohybridSparseCollectWand(index, &meta, cache, lexicon,
													 lexCount, qEntries, qCount, states,
													 nodeCount, k, &result, stats);
		if (stats != NULL)
		{
			stats->cacheHit = !cacheBuilt;
			stats->cacheBuildUs = cacheBuildUs;
			stats->cacheBytes = cache->readerBytes;
			stats->hotCacheEvictions = cache->hotEvictions;
		}
		MemoryContextSwitchTo(oldCtx);
		INSTR_TIME_SET_CURRENT(t1);
		INSTR_TIME_SUBTRACT(t1, t0);
		if (stats != NULL)
			stats->elapsedUs = (uint64) INSTR_TIME_GET_MICROSEC(t1);
		if (out != NULL)
			*out = result;
		return resultCount;
	}

	scores = (double *) palloc0(sizeof(double) * Max(nodeCount, 1u));

	for (uint32 i = 0; i < qCount; i++)
	{
		PgturbohybridSparseLexEntry key;
		PgturbohybridSparseLexEntry *found;

		if (qEntries[i].weight == 0.0f)
			continue;
		key.termId = qEntries[i].termId;
		found = (PgturbohybridSparseLexEntry *)
			bsearch(&key, lexicon, lexCount, sizeof(PgturbohybridSparseLexEntry),
					PgturbohybridSparseLexCompare);
		if (found == NULL)
			continue;
		resolved++;
		postingsTouched += PgturbohybridSparseAccumulateTerm(index,
															 found->postingsBlkno,
															 found->postingsOffno,
															 found->df,
															 (double) qEntries[i].weight,
															 (double) found->scale,
															 (int) meta.quantBits,
															 scoreKernel,
															 scores, nodeCount,
															 &simdBlocks, &scalarTail);
	}

	/*
	 * Merge delta postings (prompt 11): inserted/updated rows score exact f32
	 * into the same accumulator; their node_ids are disjoint from the base
	 * (updates create new nodes; the old node is filtered by liveness).
	 */
	if (cache->deltaLoaded && cache->deltaCount > 0)
	{
		PgturbohybridSparseVectorEntry *qSorted = (PgturbohybridSparseVectorEntry *)
			palloc(sizeof(PgturbohybridSparseVectorEntry) * qCount);
		uint64		mergedDelta = 0;

		memcpy(qSorted, qEntries, sizeof(PgturbohybridSparseVectorEntry) * qCount);
		qsort(qSorted, qCount, sizeof(PgturbohybridSparseVectorEntry),
			  PgturbohybridSparseQEntryCompare);
		for (uint32 d = 0; d < cache->deltaCount; d++)
		{
			PgturbohybridSparseVectorEntry key;
			const PgturbohybridSparseVectorEntry *found;
			uint32		node = cache->delta[d].nodeId;

			key.termId = cache->delta[d].termId;
			found = (const PgturbohybridSparseVectorEntry *)
				bsearch(&key, qSorted, qCount,
						sizeof(PgturbohybridSparseVectorEntry),
						PgturbohybridSparseQEntryCompare);
			if (found == NULL || found->weight == 0.0f || node >= nodeCount)
				continue;
			scores[node] += (double) found->weight * (double) cache->delta[d].weight;
			mergedDelta++;
		}
		pfree(qSorted);
		if (stats != NULL)
		{
			stats->deltaPages = cache->deltaDocs;
			stats->deltaPostingsDecoded = cache->deltaCount;
			stats->deltaTerms = (uint32) mergedDelta;
		}
	}

	/* Collect live, non-zero-scoring nodes. */
	scored = (PgturbohybridSparseScored *)
		palloc(sizeof(PgturbohybridSparseScored) * Max(nodeCount, 1u));
	for (uint32 n = 0; n < nodeCount; n++)
	{
		if (scores[n] != 0.0 && states[n].live)
		{
			scored[scoredCount].nodeId = n;
			scored[scoredCount].score = scores[n];
			scoredCount++;
		}
	}

	if (scoredCount > 1)
		qsort(scored, scoredCount, sizeof(PgturbohybridSparseScored),
			  PgturbohybridSparseScoredCompare);

	resultCount = (int) Min((uint64) k, scoredCount);
	result = resultCount > 0 ? (PgturbohybridSparseCandidate *)
		palloc(sizeof(PgturbohybridSparseCandidate) * resultCount) : NULL;
	for (int r = 0; r < resultCount; r++)
	{
		result[r].nodeId = scored[r].nodeId;
		result[r].heaptid = states[scored[r].nodeId].tid;
		result[r].score = scored[r].score;
	}

	MemoryContextSwitchTo(oldCtx);

	INSTR_TIME_SET_CURRENT(t1);
	INSTR_TIME_SUBTRACT(t1, t0);

	if (stats != NULL)
	{
		stats->branchUsed = true;
		stats->terms = qCount;
		stats->resolvedTerms = resolved;
		stats->postingsTouched = postingsTouched;
		stats->candidatesScored = scoredCount;
		stats->elapsedUs = (uint64) INSTR_TIME_GET_MICROSEC(t1);
		stats->quantBits = (int) meta.quantBits;
		stats->quantMode = (int) meta.quantMode;
		stats->encoding = (int) meta.postingsEncoding;
		stats->scoreKernel = scoreKernel;
		stats->simdBlocks = simdBlocks;
		stats->scalarTailPostings = scalarTail;
	}

	if (out != NULL)
		*out = result;
	return resultCount;
}
