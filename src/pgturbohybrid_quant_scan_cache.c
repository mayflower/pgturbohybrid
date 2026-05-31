#include "postgres.h"

#include <string.h>

#include "miscadmin.h"
#include "portability/instr_time.h"
#include "storage/bufmgr.h"
#include "utils/memutils.h"
#include "utils/rel.h"

#include "pgturbohybrid_quant.h"

static PgturbohybridGraphNativeCache *pgturbohybridGraphCacheList = NULL;

/*
 * Per-backend in-memory native scan cache size cap, in bytes, from the
 * turbohybrid.native_cache_max_mb GUC (default 512 MB).  An index whose
 * resident working set fits under the cap is fully loaded into a per-backend
 * codeArena so warm scans read 0 code pages; larger indexes fall back to
 * per-scan page loading.  Raising the cap past ~1 GB requires the huge
 * allocations below (the code arena for 1M x 3072 dims is ~1.5 GB).  Because
 * the cache is per-backend, size the cap to host RAM and connection count.
 */
static inline Size
PgturbohybridGraphNativeCacheMaxBytes(void)
{
	return (Size) pgturbohybrid_native_cache_max_mb * 1024 * 1024;
}

static int
PgturbohybridGraphPayloadRefCompare(const void *a, const void *b)
{
	const PgturbohybridGraphPayloadRef *ra = (const PgturbohybridGraphPayloadRef *) a;
	const PgturbohybridGraphPayloadRef *rb = (const PgturbohybridGraphPayloadRef *) b;

	if (ra->payloadSlot < rb->payloadSlot)
		return -1;
	if (ra->payloadSlot > rb->payloadSlot)
		return 1;
	if (ra->payloadValue < rb->payloadValue)
		return -1;
	if (ra->payloadValue > rb->payloadValue)
		return 1;
	return (ra->nodeId > rb->nodeId) - (ra->nodeId < rb->nodeId);
}


static bool
PgturbohybridGraphShouldCacheExactVectors(Relation index, PgturbohybridGraphMetaPageData *meta)
{
	int			policy = PgturbohybridGraphGetGraphExactCache(index);
	Size		exactBytes;
	Size		totalBytes;

	if (policy == PGTURBOHYBRID_GRAPH_EXACT_CACHE_OFF ||
		meta->tqNodeCount == 0 || meta->dimensions == 0 ||
		!BlockNumberIsValid(meta->tqExactStartBlkno))
		return false;
	if (policy == PGTURBOHYBRID_GRAPH_EXACT_CACHE_ON)
		return true;

	exactBytes = PGTURBOHYBRID_VECTOR_SIZE(meta->dimensions);
	if (meta->tqNodeCount > SIZE_MAX / exactBytes)
		return false;
	totalBytes = exactBytes * (Size) meta->tqNodeCount;

	return totalBytes <= PGTURBOHYBRID_GRAPH_EXACT_CACHE_AUTO_MAX_BYTES;
}

static bool
PgturbohybridGraphArenaBytes(uint32 count, Size itemBytes, Size *totalBytes)
{
	if (totalBytes != NULL)
		*totalBytes = 0;
	if (count == 0 || itemBytes == 0)
		return true;
	if ((Size) count > SIZE_MAX / itemBytes)
		return false;
	if (totalBytes != NULL)
		*totalBytes = (Size) count * itemBytes;
	return AllocSizeIsValid((Size) count * itemBytes);
}

/*
 * Same, but allows huge (> MaxAllocSize) sizes, for the contiguous data arenas
 * (code / residual / payload) that are allocated with palloc_extended(HUGE).
 * The per-node metadata arrays keep the 1 GB AllocSize bound, which caps a
 * cacheable index at the point where those arrays would exceed it (graceful
 * fallback to uncached scanning); the code arena is the only one that grows
 * past 1 GB in the realistic range (1M x 3072 dims => ~1.5 GB).
 */
static bool
PgturbohybridGraphArenaBytesHuge(uint32 count, Size itemBytes, Size *totalBytes)
{
	if (totalBytes != NULL)
		*totalBytes = 0;
	if (count == 0 || itemBytes == 0)
		return true;
	if ((Size) count > SIZE_MAX / itemBytes)
		return false;
	if (totalBytes != NULL)
		*totalBytes = (Size) count * itemBytes;
	return AllocHugeSizeIsValid((Size) count * itemBytes);
}

static bool
PgturbohybridGraphShouldUseNativeCache(PgturbohybridGraphMetaPageData *meta, bool cacheExactVectors)
{
	Size		totalBytes = sizeof(PgturbohybridGraphNativeCache);
	Size		bytes;
	Size		baseNeighborsPerNode;

	if (!PgturbohybridGraphArenaBytes(meta->tqNodeCount,
									  sizeof(PgturbohybridGraphScanNode), &bytes))
		return false;
	totalBytes += bytes;
	if (PgturbohybridGraphArenaBytesHuge(meta->tqNodeCount,
										 meta->tqCodeBytes, &bytes) &&
		bytes <= PgturbohybridGraphNativeCacheMaxBytes())
		totalBytes += bytes;
	if (PgturbohybridGraphArenaBytesHuge(meta->tqNodeCount,
										 meta->tqPayloadBytes, &bytes) &&
		bytes <= PgturbohybridGraphNativeCacheMaxBytes())
		totalBytes += bytes;
	if (meta->tqResidualRerankBytes > 0)
	{
		if (!PgturbohybridGraphArenaBytesHuge(meta->tqNodeCount,
											  meta->tqResidualRerankBytes, &bytes) ||
			bytes > PgturbohybridGraphNativeCacheMaxBytes())
			return false;
		totalBytes += bytes;
	}
	if (cacheExactVectors && meta->dimensions > 0)
	{
		if (PgturbohybridGraphArenaBytes(meta->tqNodeCount,
										 PGTURBOHYBRID_VECTOR_SIZE(meta->dimensions),
										 &bytes) &&
			bytes <= PGTURBOHYBRID_GRAPH_EXACT_CACHE_AUTO_MAX_BYTES)
			totalBytes += bytes;
	}
	if (!PgturbohybridGraphArenaBytes(PgturbohybridGraphAdjRecordCount(meta),
									  sizeof(uint32 *), &bytes))
		return false;
	totalBytes += bytes;
	if (!PgturbohybridGraphArenaBytes(PgturbohybridGraphAdjRecordCount(meta),
									  sizeof(uint16), &bytes))
		return false;
	totalBytes += bytes;
	baseNeighborsPerNode = (Size) PgturbohybridGraphLevelM(meta->m, 0) * sizeof(uint32);
	if (!PgturbohybridGraphArenaBytes(meta->tqNodeCount,
									  baseNeighborsPerNode, &bytes))
		return false;
	totalBytes += bytes;

	return totalBytes <= PgturbohybridGraphNativeCacheMaxBytes();
}

static void
PgturbohybridGraphInitScanStorageUncached(PgturbohybridGraphMetaPageData *meta, PgturbohybridGraphScanStorage *storage,
							   bool cacheExactVectors)
{
	Size		arenaBytes;
	int			adjRecordCount;

	memset(storage, 0, sizeof(PgturbohybridGraphScanStorage));
	storage->ctx = CurrentMemoryContext;
	storage->nodes =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(PgturbohybridGraphScanNode),
											   meta->tqNodeCount,
											   "pgturbohybrid graph scan node cache"));
	if (meta->tqNodeCount > 0 && meta->tqCodeBytes > 0 &&
		PgturbohybridGraphArenaBytesHuge(meta->tqNodeCount, meta->tqCodeBytes,
										 &arenaBytes) &&
		arenaBytes <= PgturbohybridGraphNativeCacheMaxBytes())
		storage->codeArena = palloc_extended(arenaBytes, MCXT_ALLOC_HUGE | MCXT_ALLOC_ZERO);
	if (meta->tqNodeCount > 0 && meta->tqResidualRerankBytes > 0 &&
		PgturbohybridGraphArenaBytesHuge(meta->tqNodeCount, meta->tqResidualRerankBytes,
										 &arenaBytes) &&
		arenaBytes <= PgturbohybridGraphNativeCacheMaxBytes())
		storage->residualArena = palloc_extended(arenaBytes, MCXT_ALLOC_HUGE | MCXT_ALLOC_ZERO);
	if (meta->tqNodeCount > 0 && meta->tqPayloadBytes > 0 &&
		PgturbohybridGraphArenaBytesHuge(meta->tqNodeCount, meta->tqPayloadBytes,
										 &arenaBytes) &&
		arenaBytes <= PgturbohybridGraphNativeCacheMaxBytes())
		storage->payloadArena = palloc_extended(arenaBytes, MCXT_ALLOC_HUGE | MCXT_ALLOC_ZERO);
	if (cacheExactVectors && meta->tqNodeCount > 0 && meta->dimensions > 0)
	{
		storage->exactBytes = PGTURBOHYBRID_VECTOR_SIZE(meta->dimensions);
		if (PgturbohybridGraphArenaBytes(meta->tqNodeCount, storage->exactBytes,
										 &arenaBytes) &&
			arenaBytes <= PGTURBOHYBRID_GRAPH_EXACT_CACHE_AUTO_MAX_BYTES)
			storage->exactArena = palloc0(arenaBytes);
	}
	storage->levelCount = PgturbohybridGraphLevelCapacity(meta->m);
	adjRecordCount = PgturbohybridGraphAdjRecordCount(meta);
	storage->neighbors =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint32 *),
											   adjRecordCount,
											   "pgturbohybrid graph neighbor cache"));
	storage->neighborCounts =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(uint16),
											   adjRecordCount,
											   "pgturbohybrid graph neighbor count cache"));
	storage->codeTuplesPerPage =
		PgturbohybridGraphTuplesPerPage(PgturbohybridGraphCodeTupleSize(meta->dimensions,
												  meta->tqPayloadCount,
												  meta->tqBits,
												  (meta->tqFlags & PGTURBOHYBRID_GRAPH_TQ_WEIGHTED) != 0,
												  meta->tqResidualRerankBytes));
	storage->codePageCount = PgturbohybridGraphPageCount(meta->tqNodeCount, storage->codeTuplesPerPage);
	storage->adjPageCount = BlockNumberIsValid(meta->tqAdjStartBlkno) ? 1 : 0;
	storage->codePagesLoaded =
		palloc0(PgturbohybridCheckedArrayBytes(sizeof(bool),
											   storage->codePageCount,
											   "pgturbohybrid graph code page cache"));
	if (storage->adjPageCount > 0)
		storage->adjPagesLoaded =
			palloc0(PgturbohybridCheckedArrayBytes(sizeof(bool),
												   storage->adjPageCount,
												   "pgturbohybrid graph adjacency page cache"));
	storage->codeBlknos =
		palloc(PgturbohybridCheckedArrayBytes(sizeof(BlockNumber),
											  storage->codePageCount,
											  "pgturbohybrid graph code block map"));
	PgturbohybridGraphInitBlockMap(storage->codeBlknos, storage->codePageCount);
	storage->adjBlknos =
		palloc(PgturbohybridCheckedArrayBytes(sizeof(BlockNumber),
											  adjRecordCount,
											  "pgturbohybrid graph adjacency block map"));
	storage->adjOffnos =
		palloc(PgturbohybridCheckedArrayBytes(sizeof(OffsetNumber),
											  adjRecordCount,
											  "pgturbohybrid graph adjacency offset map"));
	for (int i = 0; i < adjRecordCount; i++)
	{
		storage->adjBlknos[i] = InvalidBlockNumber;
		storage->adjOffnos[i] = InvalidOffsetNumber;
	}
}

bool
PgturbohybridGraphLoadCodePage(Relation index, PgturbohybridGraphScanOpaque so, PgturbohybridGraphMetaPageData *meta,
					PgturbohybridGraphScanStorage *storage, uint32 nodeId)
{
	int			pageNo;
	BlockNumber blkno;

	if (nodeId >= meta->tqNodeCount || !BlockNumberIsValid(meta->tqCodeStartBlkno))
		return false;

	if (so != NULL)
	{
		so->graphCodePageAttempts++;
		so->graphCodeArenaAllocatedBytes = storage->codeArena != NULL ?
			(int64) meta->tqNodeCount * (int64) meta->tqCodeBytes : 0;
	}

	/* Cache hit: code already resident (cross-scan native cache or this scan). */
	if (storage->cached && storage->nodes[nodeId].loaded)
	{
		if (so != NULL)
			so->graphCodePageHits++;
		return storage->nodes[nodeId].loaded;
	}

	pageNo = nodeId / storage->codeTuplesPerPage;
	if (pageNo < 0 || pageNo >= storage->codePageCount)
		return false;

	/* Cache hit: this candidate's code page was already loaded this scan. */
	if (storage->codePagesLoaded[pageNo])
	{
		if (so != NULL)
			so->graphCodePageHits++;
		return storage->nodes[nodeId].loaded;
	}

	/* Cache miss: must read and copy the code page. */
	if (so != NULL)
		so->graphCodePageMisses++;

	blkno = PgturbohybridGraphGetMappedBlockNumber(meta->tqCodeStartBlkno, pageNo,
										 storage->codeBlknos);

	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		OffsetNumber maxoff;
		bool		fallbackTried = false;

retry:

		CHECK_FOR_INTERRUPTS();
		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = PgturbohybridGraphPageGetOpaque(page);
		if ((opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) != PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_CODE)
		{
			UnlockReleaseBuffer(buf);
			if (!fallbackTried)
			{
				fallbackTried = true;
				if (!PgturbohybridGraphResolveChainBlockNumber(index, meta->tqCodeStartBlkno,
													 pageNo, storage->codePageCount,
													 PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_CODE,
													 storage->codeBlknos, &blkno))
					return false;
				goto retry;
			}
			return false;
		}

		if (so != NULL)
			so->graphCodePagesRead++;
		maxoff = PageGetMaxOffsetNumber(page);
		{
			bool		tqWeighted = (meta->tqFlags & PGTURBOHYBRID_GRAPH_TQ_WEIGHTED) != 0;

			for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
			{
				ItemId		iid = PageGetItemId(page, offno);
				PgturbohybridGraphCodeTuple tuple = (PgturbohybridGraphCodeTuple) PageGetItem(page, iid);
				PgturbohybridGraphScanNode *node;

				if (tuple->type != PGTURBOHYBRID_GRAPH_CODE_TUPLE_TYPE ||
					tuple->nodeId >= meta->tqNodeCount)
					continue;

				node = &storage->nodes[tuple->nodeId];
				node->heaptid = tuple->heaptid;
				node->level = tuple->level;
				node->exactBlkno = tuple->exactBlkno;
				node->exactOffno = tuple->exactOffno;
				node->payloadMask = tuple->payloadMask;
				node->scale = tuple->scale;
				node->norm = tuple->norm;
				node->codeNorm = tuple->correction;
				node->ecCorrection = PgturbohybridGraphTupleEcCorrection(tuple, tqWeighted);
				node->flags = tuple->flags;
				if (meta->tqPayloadBytes > 0)
				{
					if (storage->payloadArena != NULL)
						node->payloads = (int32 *) (storage->payloadArena +
													((Size) tuple->nodeId * meta->tqPayloadBytes));
					else if (node->payloads == NULL)
						node->payloads = (int32 *) MemoryContextAlloc(storage->ctx,
																	  meta->tqPayloadBytes);
					memcpy(node->payloads, PgturbohybridGraphTuplePayloads(tuple, tqWeighted),
						   meta->tqPayloadBytes);
				}
				if (meta->tqCodeBytes > 0)
				{
					if (storage->codeArena != NULL)
						node->code = storage->codeArena + ((Size) tuple->nodeId * meta->tqCodeBytes);
					else if (node->code == NULL)
						node->code = (uint8 *) MemoryContextAlloc(storage->ctx,
																  meta->tqCodeBytes);
					memcpy(node->code, PgturbohybridGraphTupleCode(tuple, meta->tqPayloadBytes,
																   meta->tqResidualRerankBytes,
																   tqWeighted),
						   meta->tqCodeBytes);
				}
				if (meta->tqResidualRerankBytes > 0)
				{
					if (storage->residualArena != NULL)
						node->residualSketch = storage->residualArena +
							((Size) tuple->nodeId * meta->tqResidualRerankBytes);
					else if (node->residualSketch == NULL)
						node->residualSketch = (uint8 *) MemoryContextAlloc(storage->ctx,
																			meta->tqResidualRerankBytes);
					memcpy(node->residualSketch,
						   PgturbohybridGraphTupleResidual(tuple, meta->tqPayloadBytes,
														   meta->tqResidualRerankBytes,
														   tqWeighted),
						   meta->tqResidualRerankBytes);
				}
				node->loaded = true;
				if (so != NULL)
				{
					so->graphCodeTuplesCopied++;
					so->graphCodeArenaUsedBytes += (int64) meta->tqCodeBytes;
				}
			}
		}

		storage->codePagesLoaded[pageNo] = true;
		UnlockReleaseBuffer(buf);
	}

	return storage->nodes[nodeId].loaded;
}

static void
PgturbohybridGraphLoadAllAdjPages(Relation index, PgturbohybridGraphScanOpaque so, PgturbohybridGraphMetaPageData *meta,
					   PgturbohybridGraphScanStorage *storage)
{
	BlockNumber blkno = meta->tqAdjStartBlkno;
	BlockNumber nblocks;

	if (!BlockNumberIsValid(blkno) || storage->adjPageCount <= 0 ||
		storage->adjPagesLoaded == NULL || storage->adjPagesLoaded[0])
		return;

	nblocks = RelationGetNumberOfBlocks(index);
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

		if (so != NULL)
			so->graphAdjPagesRead++;
		nextblkno = opaque->nextblkno;
		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
		{
			ItemId		iid = PageGetItemId(page, offno);
			PgturbohybridGraphAdjTuple tuple;
			int			tupleSlot;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (PgturbohybridGraphAdjTuple) PageGetItem(page, iid);
			if (tuple->type != PGTURBOHYBRID_GRAPH_ADJ_TUPLE_TYPE ||
				tuple->nodeId >= meta->tqNodeCount ||
				tuple->level > meta->graphMaxLevel ||
				tuple->level >= storage->levelCount ||
				tuple->count > PgturbohybridGraphLevelM(meta->m, tuple->level))
				continue;

			tupleSlot = PgturbohybridGraphAdjSlot(meta, tuple->nodeId, tuple->level);
			storage->adjBlknos[tupleSlot] = blkno;
			storage->adjOffnos[tupleSlot] = offno;
			storage->neighborCounts[tupleSlot] = tuple->count;
			if (storage->neighbors[tupleSlot] != NULL)
			{
				pfree(storage->neighbors[tupleSlot]);
				storage->neighbors[tupleSlot] = NULL;
			}
			if (tuple->count > 0)
			{
				storage->neighbors[tupleSlot] =
					MemoryContextAlloc(storage->ctx, sizeof(uint32) * tuple->count);
				memcpy(storage->neighbors[tupleSlot], tuple->neighbors, sizeof(uint32) * tuple->count);
			}
		}

		UnlockReleaseBuffer(buf);
		if (nextblkno == blkno)
			break;
		blkno = nextblkno;
	}

	storage->adjPagesLoaded[0] = true;
}

bool
PgturbohybridGraphLoadAdjPage(Relation index, PgturbohybridGraphScanOpaque so, PgturbohybridGraphMetaPageData *meta,
				   PgturbohybridGraphScanStorage *storage, uint32 nodeId, int level)
{
	if (nodeId >= meta->tqNodeCount || level < 0 || level > meta->graphMaxLevel ||
		!BlockNumberIsValid(meta->tqAdjStartBlkno))
		return false;

	if (storage->cached)
		return true;

	PgturbohybridGraphLoadAllAdjPages(index, so, meta, storage);
	return true;
}

static bool
PgturbohybridGraphLoadExactVectors(Relation index, PgturbohybridGraphMetaPageData *meta,
						PgturbohybridGraphScanStorage *storage)
{
	if (storage->exactArena == NULL ||
		!BlockNumberIsValid(meta->tqExactStartBlkno))
		return false;

	for (uint32 nodeId = 0; nodeId < meta->tqNodeCount; nodeId++)
	{
		PgturbohybridGraphScanNode *node = &storage->nodes[nodeId];
		char	   *exactVector;

		CHECK_FOR_INTERRUPTS();
		if (!node->loaded)
			continue;

		exactVector = storage->exactArena +
			((Size) nodeId * storage->exactBytes);
		if (!PgturbohybridGraphReadExactVectorInto(index, node, meta->dimensions,
										exactVector, NULL))
			return false;
		node->exactVector = exactVector;
	}

	return true;
}

static void
PgturbohybridGraphBuildPayloadRefs(PgturbohybridGraphMetaPageData *meta, PgturbohybridGraphScanStorage *storage)
{
	uint32		refCount = 0;
	uint32		refIndex = 0;

	if (meta->tqPayloadCount == 0 || meta->tqPayloadBytes == 0)
		return;

	for (uint32 nodeId = 0; nodeId < meta->tqNodeCount; nodeId++)
	{
		PgturbohybridGraphScanNode *node = &storage->nodes[nodeId];

		if (!node->loaded || node->payloads == NULL ||
			node->flags & PGTURBOHYBRID_GRAPH_NODE_DEAD)
			continue;

		for (int slot = 0; slot < meta->tqPayloadCount; slot++)
		{
			if (node->payloadMask & (uint16) (1U << slot))
				refCount++;
		}
	}

	if (refCount == 0)
		return;

	storage->payloadRefs = palloc(sizeof(PgturbohybridGraphPayloadRef) * refCount);
	storage->payloadRefCount = refCount;

	for (uint32 nodeId = 0; nodeId < meta->tqNodeCount; nodeId++)
	{
		PgturbohybridGraphScanNode *node = &storage->nodes[nodeId];

		if (!node->loaded || node->payloads == NULL ||
			node->flags & PGTURBOHYBRID_GRAPH_NODE_DEAD)
			continue;

		for (int slot = 0; slot < meta->tqPayloadCount; slot++)
		{
			if ((node->payloadMask & (uint16) (1U << slot)) == 0)
				continue;

			storage->payloadRefs[refIndex].payloadSlot = (int16) slot;
			storage->payloadRefs[refIndex].payloadValue = node->payloads[slot];
			storage->payloadRefs[refIndex].nodeId = nodeId;
			refIndex++;
		}
	}

	qsort(storage->payloadRefs, storage->payloadRefCount,
		  sizeof(PgturbohybridGraphPayloadRef), PgturbohybridGraphPayloadRefCompare);
}

bool
PgturbohybridGraphPayloadRefRange(PgturbohybridGraphScanStorage *storage, int payloadSlot,
					   int32 payloadValue, uint32 *firstIndex,
					   uint32 *refCount)
{
	uint32		lo = 0;
	uint32		hi = storage->payloadRefCount;
	uint32		first;

	if (payloadSlot < 0 || storage->payloadRefs == NULL ||
		storage->payloadRefCount == 0)
		return false;

	while (lo < hi)
	{
		uint32		mid = lo + (hi - lo) / 2;
		PgturbohybridGraphPayloadRef *ref = &storage->payloadRefs[mid];

		if (ref->payloadSlot < payloadSlot ||
			(ref->payloadSlot == payloadSlot &&
			 ref->payloadValue < payloadValue))
			lo = mid + 1;
		else
			hi = mid;
	}

	first = lo;
	hi = storage->payloadRefCount;
	while (lo < hi)
	{
		uint32		mid = lo + (hi - lo) / 2;
		PgturbohybridGraphPayloadRef *ref = &storage->payloadRefs[mid];

		if (ref->payloadSlot > payloadSlot ||
			(ref->payloadSlot == payloadSlot &&
			 ref->payloadValue > payloadValue))
			hi = mid;
		else
			lo = mid + 1;
	}

	if (first >= lo)
		return false;

	*firstIndex = first;
	*refCount = lo - first;
	return true;
}

static bool
PgturbohybridGraphCacheMatches(PgturbohybridGraphNativeCache *cache, Relation index,
					PgturbohybridGraphMetaPageData *meta)
{
	return cache->relid == RelationGetRelid(index) &&
		cache->relfilenumber == PgturbohybridGraphRelFileNumber(index) &&
		cache->dimensions == meta->dimensions &&
		cache->m == meta->m &&
		cache->graphMaxLevel == meta->graphMaxLevel &&
		cache->graphFlags == meta->graphFlags &&
		cache->tqNodeCount == meta->tqNodeCount &&
		cache->tqEntryNodeId == meta->tqEntryNodeId &&
		cache->tqCodeBytes == meta->tqCodeBytes &&
		cache->tqBits == meta->tqBits &&
		cache->tqPayloadCount == meta->tqPayloadCount &&
		cache->tqPayloadBytes == meta->tqPayloadBytes &&
		cache->tqResidualRerankBytes == meta->tqResidualRerankBytes &&
		cache->tqCodeStartBlkno == meta->tqCodeStartBlkno &&
		cache->tqAdjStartBlkno == meta->tqAdjStartBlkno &&
		cache->tqExactStartBlkno == meta->tqExactStartBlkno &&
		cache->tqCorrectionStartBlkno == meta->tqCorrectionStartBlkno;
}

static void
PgturbohybridGraphDropStaleCaches(Relation index, PgturbohybridGraphMetaPageData *meta)
{
	PgturbohybridGraphNativeCache **link = &pgturbohybridGraphCacheList;

	while (*link != NULL)
	{
		PgturbohybridGraphNativeCache *cache = *link;

		if (cache->relid == RelationGetRelid(index) &&
			!PgturbohybridGraphCacheMatches(cache, index, meta))
		{
			*link = cache->next;
			MemoryContextDelete(cache->ctx);
			continue;
		}

		link = &cache->next;
	}
}

void
PgturbohybridGraphInvalidateCaches(Relation index)
{
	PgturbohybridGraphNativeCache **link = &pgturbohybridGraphCacheList;
	Oid			relid = RelationGetRelid(index);

	while (*link != NULL)
	{
		PgturbohybridGraphNativeCache *cache = *link;

		if (cache->relid == relid)
		{
			*link = cache->next;
			MemoryContextDelete(cache->ctx);
			continue;
		}

		link = &cache->next;
	}
}

static PgturbohybridGraphNativeCache *
PgturbohybridGraphFindCache(Relation index, PgturbohybridGraphMetaPageData *meta)
{
	PgturbohybridGraphDropStaleCaches(index, meta);

	for (PgturbohybridGraphNativeCache *cache = pgturbohybridGraphCacheList; cache != NULL; cache = cache->next)
	{
		if (PgturbohybridGraphCacheMatches(cache, index, meta))
			return cache;
	}

	return NULL;
}

/*
 * Sum the resident footprint a built per-backend cache holds: the code, exact
 * and adjacency arenas (the terms that dominate and are duplicated once per
 * backend), plus the node array and per-scan metadata.  Reported via
 * turbohybrid_last_scan_stats() (native_cache_bytes and the code/adj/exact
 * breakdown) so concurrent-client memory duplication is visible.
 */
static void
PgturbohybridGraphCacheComputeResidentBytes(PgturbohybridGraphNativeCache *cache,
											PgturbohybridGraphMetaPageData *meta)
{
	PgturbohybridGraphScanStorage *storage = &cache->storage;
	int			adjRecordCount = PgturbohybridGraphAdjRecordCount(meta);
	Size		codeBytes = 0;
	Size		adjBytes = 0;
	Size		exactBytes = 0;
	Size		otherBytes = 0;

	if (storage->codeArena != NULL)
		codeBytes = (Size) meta->tqNodeCount * (Size) meta->tqCodeBytes;
	if (storage->exactArena != NULL)
		exactBytes = (Size) meta->tqNodeCount * storage->exactBytes;

	/* Adjacency: the per-slot metadata arrays plus the actual neighbor lists. */
	adjBytes += (Size) adjRecordCount *
		(sizeof(uint32 *) + sizeof(uint16) + sizeof(BlockNumber) + sizeof(OffsetNumber));
	if (storage->neighborCounts != NULL)
	{
		for (int i = 0; i < adjRecordCount; i++)
			adjBytes += (Size) storage->neighborCounts[i] * sizeof(uint32);
	}

	/* Node array, residual/payload arenas, visited generation, code-page map. */
	otherBytes += (Size) meta->tqNodeCount * sizeof(PgturbohybridGraphScanNode);
	if (storage->residualArena != NULL)
		otherBytes += (Size) meta->tqNodeCount * (Size) meta->tqResidualRerankBytes;
	if (storage->payloadArena != NULL)
		otherBytes += (Size) meta->tqNodeCount * (Size) meta->tqPayloadBytes;
	if (storage->visitedGeneration != NULL)
		otherBytes += (Size) meta->tqNodeCount * sizeof(uint32);
	otherBytes += (Size) storage->codePageCount * (sizeof(bool) + sizeof(BlockNumber));

	cache->residentCodeBytes = codeBytes;
	cache->residentAdjBytes = adjBytes;
	cache->residentExactBytes = exactBytes;
	cache->residentTotalBytes = codeBytes + adjBytes + exactBytes + otherBytes;
}

static PgturbohybridGraphNativeCache *
PgturbohybridGraphBuildCache(Relation index, PgturbohybridGraphMetaPageData *meta)
{
	MemoryContext cacheCtx;
	MemoryContext oldCtx;
	PgturbohybridGraphNativeCache *cache;
	instr_time	buildStart;
	instr_time	buildElapsed;

	INSTR_TIME_SET_CURRENT(buildStart);

	cacheCtx = AllocSetContextCreate(CacheMemoryContext,
									 "pgturbohybrid native graph cache",
									 ALLOCSET_DEFAULT_SIZES);
	oldCtx = MemoryContextSwitchTo(cacheCtx);

	cache = palloc0(sizeof(PgturbohybridGraphNativeCache));
	cache->relid = RelationGetRelid(index);
	cache->relfilenumber = PgturbohybridGraphRelFileNumber(index);
	cache->dimensions = meta->dimensions;
	cache->m = meta->m;
	cache->graphMaxLevel = meta->graphMaxLevel;
	cache->graphFlags = meta->graphFlags;
	cache->tqNodeCount = meta->tqNodeCount;
	cache->tqEntryNodeId = meta->tqEntryNodeId;
	cache->tqCodeBytes = meta->tqCodeBytes;
	cache->tqBits = meta->tqBits;
	cache->tqPayloadCount = meta->tqPayloadCount;
	cache->tqPayloadBytes = meta->tqPayloadBytes;
	cache->tqResidualRerankBytes = meta->tqResidualRerankBytes;
	cache->tqCodeStartBlkno = meta->tqCodeStartBlkno;
	cache->tqAdjStartBlkno = meta->tqAdjStartBlkno;
	cache->tqExactStartBlkno = meta->tqExactStartBlkno;
	cache->tqCorrectionStartBlkno = meta->tqCorrectionStartBlkno;
	cache->ctx = cacheCtx;

	PgturbohybridGraphInitScanStorageUncached(meta, &cache->storage,
								   PgturbohybridGraphShouldCacheExactVectors(index, meta));
	cache->storage.ctx = cacheCtx;
	if (meta->tqNodeCount > 0)
	{
		cache->storage.visitedGeneration = palloc0(sizeof(uint32) * meta->tqNodeCount);
		cache->storage.visitGeneration = palloc0(sizeof(uint32));
	}

	if (cache->storage.codeArena != NULL)
	{
		for (uint32 nodeId = 0; nodeId < meta->tqNodeCount;
			 nodeId += cache->storage.codeTuplesPerPage)
			(void) PgturbohybridGraphLoadCodePage(index, NULL, meta, &cache->storage, nodeId);
	}

	PgturbohybridGraphBuildPayloadRefs(meta, &cache->storage);

	if (meta->tqNodeCount > 0)
		PgturbohybridGraphLoadAllAdjPages(index, NULL, meta, &cache->storage);

	if (cache->storage.exactArena != NULL)
		(void) PgturbohybridGraphLoadExactVectors(index, meta, &cache->storage);
	cache->storage.cached = true;

	PgturbohybridGraphCacheComputeResidentBytes(cache, meta);

	cache->next = pgturbohybridGraphCacheList;
	pgturbohybridGraphCacheList = cache;

	MemoryContextSwitchTo(oldCtx);

	INSTR_TIME_SET_CURRENT(buildElapsed);
	INSTR_TIME_SUBTRACT(buildElapsed, buildStart);
	cache->buildUs = (int64) INSTR_TIME_GET_MICROSEC(buildElapsed);

	return cache;
}

void
PgturbohybridGraphInitScanStorage(Relation index, PgturbohybridGraphMetaPageData *meta,
					   PgturbohybridGraphScanStorage *storage,
					   PgturbohybridGraphCacheInitInfo *info)
{
	PgturbohybridGraphNativeCache *cache;
	bool		cacheExactVectors = PgturbohybridGraphShouldCacheExactVectors(index, meta);

	if (info != NULL)
		memset(info, 0, sizeof(*info));

	if (!PgturbohybridGraphShouldUseNativeCache(meta, cacheExactVectors))
	{
		PgturbohybridGraphInitScanStorageUncached(meta, storage, cacheExactVectors);
		if (info != NULL)
			info->mode = PGTURBOHYBRID_GRAPH_NATIVE_CACHE_UNCACHED;
		return;
	}

	cache = PgturbohybridGraphFindCache(index, meta);
	if (cache == NULL)
	{
		cache = PgturbohybridGraphBuildCache(index, meta);
		if (info != NULL)
		{
			info->builtThisScan = true;
			info->buildUs = cache->buildUs;
		}
	}

	memcpy(storage, &cache->storage, sizeof(PgturbohybridGraphScanStorage));
	storage->cached = true;

	if (info != NULL)
	{
		info->mode = PGTURBOHYBRID_GRAPH_NATIVE_CACHE_PER_BACKEND;
		info->totalBytes = (int64) cache->residentTotalBytes;
		info->codeBytes = (int64) cache->residentCodeBytes;
		info->adjBytes = (int64) cache->residentAdjBytes;
		info->exactBytes = (int64) cache->residentExactBytes;
	}
}

PgturbohybridGraphNativeCache *
PgturbohybridGraphInitInsertStorage(Relation index, PgturbohybridGraphMetaPageData *meta,
						 PgturbohybridGraphScanStorage *storage)
{
	PgturbohybridGraphNativeCache *cache;
	bool		cacheExactVectors = PgturbohybridGraphShouldCacheExactVectors(index, meta);

	if (!PgturbohybridGraphShouldUseNativeCache(meta, cacheExactVectors))
	{
		PgturbohybridGraphInitScanStorageUncached(meta, storage, cacheExactVectors);
		return NULL;
	}

	cache = PgturbohybridGraphFindCache(index, meta);
	if (cache == NULL)
		cache = PgturbohybridGraphBuildCache(index, meta);

	memcpy(storage, &cache->storage, sizeof(PgturbohybridGraphScanStorage));
	storage->cached = true;
	return cache;
}

static uint32 *
PgturbohybridGraphCacheCopyNeighbors(PgturbohybridGraphScanStorage *storage, uint32 *neighbors,
						  int count)
{
	uint32	   *copy;

	if (count <= 0)
		return NULL;

	copy = MemoryContextAlloc(storage->ctx, sizeof(uint32) * count);
	memcpy(copy, neighbors, sizeof(uint32) * count);
	return copy;
}

void
PgturbohybridGraphAppendInsertCacheNode(PgturbohybridGraphNativeCache *cache, PgturbohybridGraphMetaPageData *meta,
							 uint32 nodeId, ItemPointer heapTid,
							 int nodeLevel, Vector *vector, uint8 *code,
							 uint8 *residualSketch,
							 float scale, float norm, float codeNorm,
							 float ecCorrection, int32 *payloads,
							 uint16 payloadMask, BlockNumber exactBlkno,
							 OffsetNumber exactOffno, uint32 **selected,
							 int *selectedCounts, BlockNumber *adjBlknos,
							 OffsetNumber *adjOffnos, BlockNumber codeStart,
							 BlockNumber adjStart, BlockNumber exactStart,
							 uint32 entryNodeId, uint16 graphMaxLevel)
{
	PgturbohybridGraphScanStorage *storage;
	uint32		oldNodeCount;
	uint32		newNodeCount;
	int			oldAdjRecordCount;
	int			newAdjRecordCount;
	int			oldCodePageCount;
	int			newCodePageCount;
	Size		codeBytes;
	Size		payloadBytes;
	Size		residualBytes;
	MemoryContext oldCtx;

	if (cache == NULL || nodeId != cache->tqNodeCount)
		return;

	storage = &cache->storage;
	oldNodeCount = cache->tqNodeCount;
	newNodeCount = oldNodeCount + 1;
	oldAdjRecordCount = oldNodeCount * PgturbohybridGraphLevelCapacity(meta->m);
	newAdjRecordCount = newNodeCount * PgturbohybridGraphLevelCapacity(meta->m);
	oldCodePageCount = storage->codePageCount;
	newCodePageCount = PgturbohybridGraphPageCount(newNodeCount, storage->codeTuplesPerPage);
	codeBytes = meta->tqCodeBytes;
	payloadBytes = meta->tqPayloadBytes;
	residualBytes = meta->tqResidualRerankBytes;

	oldCtx = MemoryContextSwitchTo(cache->ctx);

	storage->nodes = repalloc(storage->nodes,
							  PgturbohybridCheckedArrayBytes(sizeof(PgturbohybridGraphScanNode),
															 newNodeCount,
															 "pgturbohybrid graph scan node cache"));
	memset(&storage->nodes[nodeId], 0, sizeof(PgturbohybridGraphScanNode));
	if (codeBytes > 0)
	{
		/* repalloc_huge: the code arena may exceed 1 GB (see ArenaBytesHuge). */
		storage->codeArena = repalloc_huge(storage->codeArena,
										   (Size) codeBytes * (Size) newNodeCount);
		memcpy(storage->codeArena + ((Size) nodeId * codeBytes),
			   code, codeBytes);
	}
	if (payloadBytes > 0)
	{
		storage->payloadArena = repalloc_huge(storage->payloadArena,
											  (Size) payloadBytes * (Size) newNodeCount);
		if (payloads != NULL)
			memcpy(storage->payloadArena + ((Size) nodeId * payloadBytes),
				   payloads, payloadBytes);
		else
			memset(storage->payloadArena + ((Size) nodeId * payloadBytes),
				   0, payloadBytes);
	}
	if (residualBytes > 0)
	{
		storage->residualArena = repalloc_huge(storage->residualArena,
											   (Size) residualBytes * (Size) newNodeCount);
		if (residualSketch != NULL)
			memcpy(storage->residualArena + ((Size) nodeId * residualBytes),
				   residualSketch, residualBytes);
		else
			memset(storage->residualArena + ((Size) nodeId * residualBytes),
				   0, residualBytes);
	}
	if (storage->exactArena != NULL)
	{
		storage->exactArena = repalloc(storage->exactArena,
									   PgturbohybridCheckedArrayBytes(storage->exactBytes,
																	  newNodeCount,
																	  "pgturbohybrid graph exact-vector arena"));
		memcpy(storage->exactArena + ((Size) nodeId * storage->exactBytes),
			   vector, storage->exactBytes);
	}
	if (storage->visitedGeneration != NULL)
	{
		storage->visitedGeneration = repalloc(storage->visitedGeneration,
											  PgturbohybridCheckedArrayBytes(sizeof(uint32),
																			 newNodeCount,
																			 "pgturbohybrid graph visited generation cache"));
		storage->visitedGeneration[nodeId] = 0;
	}
	if (newCodePageCount != oldCodePageCount)
	{
		storage->codePagesLoaded = repalloc(storage->codePagesLoaded,
											PgturbohybridCheckedArrayBytes(sizeof(bool),
																		   newCodePageCount,
																		   "pgturbohybrid graph code page cache"));
		storage->codeBlknos = repalloc(storage->codeBlknos,
									   PgturbohybridCheckedArrayBytes(sizeof(BlockNumber),
																	  newCodePageCount,
																	  "pgturbohybrid graph code block map"));
		for (int i = oldCodePageCount; i < newCodePageCount; i++)
		{
			storage->codePagesLoaded[i] = true;
			storage->codeBlknos[i] = InvalidBlockNumber;
		}
		storage->codePageCount = newCodePageCount;
	}

	storage->neighbors = repalloc(storage->neighbors,
								  PgturbohybridCheckedArrayBytes(sizeof(uint32 *),
																 newAdjRecordCount,
																 "pgturbohybrid graph neighbor cache"));
	storage->neighborCounts = repalloc(storage->neighborCounts,
									   PgturbohybridCheckedArrayBytes(sizeof(uint16),
																	  newAdjRecordCount,
																	  "pgturbohybrid graph neighbor count cache"));
	storage->adjBlknos = repalloc(storage->adjBlknos,
								  PgturbohybridCheckedArrayBytes(sizeof(BlockNumber),
																 newAdjRecordCount,
																 "pgturbohybrid graph adjacency block map"));
	storage->adjOffnos = repalloc(storage->adjOffnos,
								  PgturbohybridCheckedArrayBytes(sizeof(OffsetNumber),
																 newAdjRecordCount,
																 "pgturbohybrid graph adjacency offset map"));
	memset(&storage->neighbors[oldAdjRecordCount], 0,
		   sizeof(uint32 *) * (newAdjRecordCount - oldAdjRecordCount));
	memset(&storage->neighborCounts[oldAdjRecordCount], 0,
		   sizeof(uint16) * (newAdjRecordCount - oldAdjRecordCount));
	for (int i = oldAdjRecordCount; i < newAdjRecordCount; i++)
	{
		storage->adjBlknos[i] = InvalidBlockNumber;
		storage->adjOffnos[i] = InvalidOffsetNumber;
	}

	for (int level = 0; level < PgturbohybridGraphLevelCapacity(meta->m); level++)
	{
		int			slot = PgturbohybridGraphAdjSlot(meta, nodeId, level);
		int			count = level <= nodeLevel ? selectedCounts[level] : 0;

		storage->neighborCounts[slot] = count;
		if (level <= nodeLevel)
		{
			storage->adjBlknos[slot] = adjBlknos[level];
			storage->adjOffnos[slot] = adjOffnos[level];
		}
		storage->neighbors[slot] =
			PgturbohybridGraphCacheCopyNeighbors(storage,
									  level <= nodeLevel ? selected[level] : NULL,
									  count);
	}

	storage->nodes[nodeId].heaptid = *heapTid;
	storage->nodes[nodeId].level = nodeLevel;
	storage->nodes[nodeId].exactBlkno = exactBlkno;
	storage->nodes[nodeId].exactOffno = exactOffno;
	storage->nodes[nodeId].payloadMask = payloadMask;
	storage->nodes[nodeId].scale = scale;
	storage->nodes[nodeId].norm = norm;
	storage->nodes[nodeId].codeNorm = codeNorm;
	storage->nodes[nodeId].ecCorrection = ecCorrection;
	storage->nodes[nodeId].flags = 0;
	storage->nodes[nodeId].loaded = true;
	if (codeBytes > 0)
		storage->nodes[nodeId].code = storage->codeArena + ((Size) nodeId * codeBytes);
	if (payloadBytes > 0)
		storage->nodes[nodeId].payloads =
			(int32 *) (storage->payloadArena + ((Size) nodeId * payloadBytes));
	if (residualBytes > 0)
		storage->nodes[nodeId].residualSketch =
			storage->residualArena + ((Size) nodeId * residualBytes);
	if (storage->exactArena != NULL)
		storage->nodes[nodeId].exactVector =
			storage->exactArena + ((Size) nodeId * storage->exactBytes);

	if (payloadBytes > 0 && payloads != NULL && payloadMask != 0)
	{
		uint32		addedRefs = 0;

		for (int slot = 0; slot < meta->tqPayloadCount; slot++)
		{
			if (payloadMask & (uint16) (1U << slot))
				addedRefs++;
		}
		if (addedRefs > 0)
		{
			uint32		oldRefCount = storage->payloadRefCount;
			uint32		refIndex = oldRefCount;

			if (storage->payloadRefs == NULL)
				storage->payloadRefs =
					palloc(sizeof(PgturbohybridGraphPayloadRef) * addedRefs);
			else
				storage->payloadRefs = repalloc(storage->payloadRefs,
												sizeof(PgturbohybridGraphPayloadRef) *
												(oldRefCount + addedRefs));
			for (int slot = 0; slot < meta->tqPayloadCount; slot++)
			{
				if ((payloadMask & (uint16) (1U << slot)) == 0)
					continue;
				storage->payloadRefs[refIndex].payloadSlot = (int16) slot;
				storage->payloadRefs[refIndex].payloadValue = payloads[slot];
				storage->payloadRefs[refIndex].nodeId = nodeId;
				refIndex++;
			}
			storage->payloadRefCount = oldRefCount + addedRefs;
			qsort(storage->payloadRefs, storage->payloadRefCount,
				  sizeof(PgturbohybridGraphPayloadRef), PgturbohybridGraphPayloadRefCompare);
		}
	}

	cache->tqNodeCount = newNodeCount;
	cache->tqEntryNodeId = entryNodeId;
	cache->graphMaxLevel = graphMaxLevel;
	cache->tqResidualRerankBytes = meta->tqResidualRerankBytes;
	cache->tqCodeStartBlkno = codeStart;
	cache->tqAdjStartBlkno = adjStart;
	cache->tqExactStartBlkno = exactStart;

	MemoryContextSwitchTo(oldCtx);
}
