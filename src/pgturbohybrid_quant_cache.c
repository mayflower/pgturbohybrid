#include "postgres.h"

#include <float.h>
#include <string.h>

#include "storage/bufmgr.h"
#include "utils/memutils.h"
#include "utils/rel.h"

#include "pgturbohybrid_quant.h"

static PgturbohybridGraphCorrectionCache *tqGraphCorrectionCacheList = NULL;

bool
PgturbohybridGraphLoadCorrection(Relation index, int dimensions, float **ecShift, float **ecScale)
{
	PgturbohybridGraphMetaPageData meta;
	BlockNumber blkno;
	BlockNumber nblocks;
	bool	   *shiftSeen;
	bool	   *scaleSeen;
	int			shiftMissing = dimensions;
	int			scaleMissing = dimensions;
	int			hops = 0;
	PgturbohybridGraphCorrectionCache *cache;

	*ecShift = NULL;
	*ecScale = NULL;

	if (!PgturbohybridGraphReadMeta(index, &meta) ||
		(meta.tqFlags & PGTURBOHYBRID_GRAPH_TQ_PLUS) == 0 ||
		meta.dimensions != (uint32) dimensions ||
		!BlockNumberIsValid(meta.tqCorrectionStartBlkno))
		return false;

	{
		PgturbohybridGraphCorrectionCache **link = &tqGraphCorrectionCacheList;

		while (*link != NULL)
		{
			cache = *link;

			if (cache->relid != RelationGetRelid(index))
			{
				link = &cache->next;
				continue;
			}

			if (cache->relfilenumber == PgturbohybridGraphRelFileNumber(index) &&
				cache->dimensions == meta.dimensions &&
				cache->tqFlags == meta.tqFlags &&
				cache->tqCorrectionStartBlkno == meta.tqCorrectionStartBlkno)
			{
				*ecShift = palloc(sizeof(float) * dimensions);
				*ecScale = palloc(sizeof(float) * dimensions);
				memcpy(*ecShift, cache->ecShift, sizeof(float) * dimensions);
				memcpy(*ecScale, cache->ecScale, sizeof(float) * dimensions);
				return true;
			}

			/*
			 * Same relation but a different generation: REINDEX / VACUUM FULL /
			 * TRUNCATE changed the relfilenumber or correction layout, so this
			 * entry can never match again.  Drop it instead of leaking it (and
			 * its context) for the backend's lifetime.
			 */
			*link = cache->next;
			MemoryContextDelete(cache->ctx);
			pfree(cache);
		}
	}

	*ecShift = palloc0(sizeof(float) * dimensions);
	*ecScale = palloc0(sizeof(float) * dimensions);
	shiftSeen = palloc0(sizeof(bool) * dimensions);
	scaleSeen = palloc0(sizeof(bool) * dimensions);
	blkno = meta.tqCorrectionStartBlkno;
	nblocks = RelationGetNumberOfBlocks(index);

	while (BlockNumberIsValid(blkno) && blkno < nblocks && hops++ < (int) nblocks)
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		OffsetNumber maxoff;

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = PgturbohybridGraphPageGetOpaque(page);

		if ((opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) != PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_CORRECTION)
		{
			UnlockReleaseBuffer(buf);
			break;
		}

		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
		{
			ItemId		iid = PageGetItemId(page, offno);
			PgturbohybridGraphCorrectionTuple tuple;
			float	   *target;
			bool	   *seen;
			int		   *missing;
			uint32		startDim;
			uint16		count;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (PgturbohybridGraphCorrectionTuple) PageGetItem(page, iid);
			if (tuple->type != PGTURBOHYBRID_GRAPH_CORRECTION_TUPLE_TYPE ||
				(tuple->field != 0 && tuple->field != 1))
				continue;

			startDim = tuple->startDim;
			count = tuple->count;
			if (count == 0 || startDim >= (uint32) dimensions ||
				count > (uint32) dimensions - startDim)
				continue;

			if (tuple->field == 0)
			{
				target = *ecShift;
				seen = shiftSeen;
				missing = &shiftMissing;
			}
			else
			{
				target = *ecScale;
				seen = scaleSeen;
				missing = &scaleMissing;
			}

			memcpy(target + startDim, tuple->values, sizeof(float) * count);
			for (int dim = 0; dim < count; dim++)
			{
				int			offset = startDim + dim;

				if (!seen[offset])
				{
					seen[offset] = true;
					(*missing)--;
				}
			}
		}

		blkno = opaque->nextblkno;
		UnlockReleaseBuffer(buf);

		if (shiftMissing == 0 && scaleMissing == 0)
		{
			MemoryContext oldCtx;

			cache = MemoryContextAllocZero(CacheMemoryContext,
										   sizeof(PgturbohybridGraphCorrectionCache));
			cache->ctx = AllocSetContextCreate(CacheMemoryContext,
											   "pgturbohybrid graph correction cache",
											   ALLOCSET_SMALL_SIZES);
			cache->relid = RelationGetRelid(index);
			cache->relfilenumber = PgturbohybridGraphRelFileNumber(index);
			cache->dimensions = meta.dimensions;
			cache->tqFlags = meta.tqFlags;
			cache->tqCorrectionStartBlkno = meta.tqCorrectionStartBlkno;
			oldCtx = MemoryContextSwitchTo(cache->ctx);
			cache->ecShift = palloc(sizeof(float) * dimensions);
			cache->ecScale = palloc(sizeof(float) * dimensions);
			memcpy(cache->ecShift, *ecShift, sizeof(float) * dimensions);
			memcpy(cache->ecScale, *ecScale, sizeof(float) * dimensions);
			MemoryContextSwitchTo(oldCtx);
			cache->next = tqGraphCorrectionCacheList;
			tqGraphCorrectionCacheList = cache;
			pfree(shiftSeen);
			pfree(scaleSeen);
			return true;
		}
	}

	pfree(shiftSeen);
	pfree(scaleSeen);
	pfree(*ecShift);
	pfree(*ecScale);
	*ecShift = NULL;
	*ecScale = NULL;
	return false;
}

/*
 * Drop every cached correction entry for this relation.  Called from
 * PgturbohybridGraphInvalidateCaches() so a REINDEX/insert that rewrites the
 * correction sidecar cannot leave a stale, never-matched entry (plus its
 * context) alive for the backend's lifetime.
 */
void
PgturbohybridGraphInvalidateCorrectionCache(Relation index)
{
	PgturbohybridGraphCorrectionCache **link = &tqGraphCorrectionCacheList;
	Oid			relid = RelationGetRelid(index);

	while (*link != NULL)
	{
		PgturbohybridGraphCorrectionCache *cache = *link;

		if (cache->relid == relid)
		{
			*link = cache->next;
			MemoryContextDelete(cache->ctx);
			pfree(cache);
			continue;
		}

		link = &cache->next;
	}
}
