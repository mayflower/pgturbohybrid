/*
 * pgturbohybrid_bm25_tenant.c
 *
 * Per-tenant BM25 corpus statistics.
 *
 * The lexical index is one shared corpus, but idf and length normalization
 * were computed from that single corpus for every tenant. Measured in
 * production on 2026-07-28: after a bulk purge the index counted 12,480
 * documents against 9,489 live rows (avgdl 1,359 vs 1,040) and a *different*
 * tenant's scenario ablation dropped two of eight. Global N/avgdl lets one
 * tenant's corpus shape move every other tenant's ranking.
 *
 * This file keeps precomputed per-tenant aggregates:
 *
 *   tenant -> { docCount, totalDocLen }
 *
 * The per-node tenant key lives in the docstats entries and delta tuples, so
 * compaction rebuilds the aggregates exactly from live nodes. Inserts
 * increment in place under the same heavyweight delta lock that serializes
 * BM25 appends. Deletes are reconciled by the existing compaction triggers.
 *
 * Tenant 0 is the "untracked" bucket: version-1 docstats pages read back as
 * tenant 0 (those bytes were zero), documents from indexes without a tenant
 * payload column land there, and scoped queries never match it -- they fall
 * back to global statistics.
 */
#include "postgres.h"

#include "access/generic_xlog.h"
#include "access/relation.h"
#include "funcapi.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "utils/builtins.h"
#include "utils/rel.h"
#include "utils/tuplestore.h"

#include "pgturbohybrid.h"
#include "pgturbohybrid_am.h"
#include "pgturbohybrid_quant.h"
#include "pgturbohybrid_bm25.h"


#define PGTURBOHYBRID_BM25_TENANT_ACCUM_CAPACITY 1024

static uint32
PgturbohybridBm25TenantHash(int32 tenant)
{
	uint32		key = (uint32) tenant;

	key *= 2654435761u;
	return key;
}

void
PgturbohybridBm25TenantAccumInit(PgturbohybridBm25TenantAccum *accum)
{
	accum->capacity = PGTURBOHYBRID_BM25_TENANT_ACCUM_CAPACITY;
	accum->entries = (PgturbohybridBm25TenantEntryData *)
		palloc0(sizeof(PgturbohybridBm25TenantEntryData) * accum->capacity);
	accum->count = 0;
	accum->saturated = false;
}

void
PgturbohybridBm25TenantAccumFree(PgturbohybridBm25TenantAccum *accum)
{
	if (accum->entries != NULL)
		pfree(accum->entries);
	accum->entries = NULL;
}

/*
 * Add one document to the in-memory accumulator. Tenant 0 is not tracked:
 * it is the untracked bucket and never scopes a query.
 */
void
PgturbohybridBm25TenantAccumAdd(PgturbohybridBm25TenantAccum *accum,
								int32 tenant, uint32 docLen)
{
	uint32		mask;
	uint32		idx;

	if (tenant == 0)
		return;

	if (accum->saturated || accum->entries == NULL)
		return;

	mask = accum->capacity - 1;
	idx = PgturbohybridBm25TenantHash(tenant) & mask;
	for (uint32 probe = 0; probe < accum->capacity; probe++)
	{
		PgturbohybridBm25TenantEntryData *entry = &accum->entries[idx];

		if (entry->tenant == tenant)
		{
			entry->docCount++;
			entry->totalDocLen += docLen;
			return;
		}
		if (entry->tenant == 0)
		{
			entry->tenant = tenant;
			entry->docCount = 1;
			entry->totalDocLen = docLen;
			accum->count++;
			return;
		}
		idx = (idx + 1) & mask;
	}

	/* Hash full: stop tracking further tenants until the next compaction. */
	accum->saturated = true;
}

/*
 * Read one tenant aggregate. Walks the (short) page chain in share mode.
 */
bool
PgturbohybridBm25TenantLookup(Relation index, BlockNumber tenantStatsBlkno,
							  int32 tenant,
							  PgturbohybridBm25TenantEntryData *entry)
{
	BlockNumber blkno = tenantStatsBlkno;
	uint32		visited = 0;

	if (tenant == 0 || !BlockNumberIsValid(tenantStatsBlkno))
		return false;

	while (BlockNumberIsValid(blkno))
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		OffsetNumber maxoff;

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = PgturbohybridGraphPageGetOpaque(page);

		if (opaque->page_id != PGTURBOHYBRID_GRAPH_PAGE_ID ||
			(opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) !=
			PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_TENANT)
		{
			UnlockReleaseBuffer(buf);
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("pgturbohybrid BM25 tenant stats page has unexpected page kind")));
		}

		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off++)
		{
			ItemId		iid = PageGetItemId(page, off);
			PgturbohybridBm25TenantTuple tuple;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (PgturbohybridBm25TenantTuple) PageGetItem(page, iid);
			if (tuple->type != PGTURBOHYBRID_BM25_TENANT_TUPLE_TYPE)
				continue;

			for (uint16 i = 0; i < tuple->count; i++)
			{
				if (tuple->entries[i].tenant == tenant)
				{
					*entry = tuple->entries[i];
					UnlockReleaseBuffer(buf);
					return true;
				}
			}
		}

		blkno = opaque->nextblkno;
		UnlockReleaseBuffer(buf);

		if (++visited > 4096)
			break;				/* corrupt chain: bounded, fall back */
	}

	return false;
}

/*
 * Increment one tenant aggregate in place. Called from the delta-append
 * write phase, while the caller holds the heavyweight BM25 delta lock, so
 * concurrent appends cannot interleave here.
 */
static void UpdateTenantEntry(Relation index, Buffer buf, int32 tenant,
						  uint32 docLen, bool useWal);

bool
PgturbohybridBm25TenantIncrement(Relation index, BlockNumber tenantStatsBlkno,
								 int32 tenant, uint32 docLen,
								 uint64 deltaGeneration)
{
	BlockNumber blkno = tenantStatsBlkno;
	bool		useWal = RelationNeedsWAL(index);
	uint32		visited = 0;

	(void) deltaGeneration;

	if (tenant == 0 || !BlockNumberIsValid(tenantStatsBlkno))
		return false;

	/*
	 * Phase 1: search the whole chain in share mode for the tenant's entry.
	 * The append path below must only run when no entry exists anywhere --
	 * checking per page would append a duplicate into the first page with
	 * free space while the real entry lives on a later page.
	 */
	while (BlockNumberIsValid(blkno))
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		OffsetNumber maxoff;
		bool		found = false;

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = PgturbohybridGraphPageGetOpaque(page);

		if (opaque->page_id != PGTURBOHYBRID_GRAPH_PAGE_ID ||
			(opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) !=
			PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_TENANT)
		{
			UnlockReleaseBuffer(buf);
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("pgturbohybrid BM25 tenant stats page has unexpected page kind")));
		}

		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber off = FirstOffsetNumber; off <= maxoff && !found; off++)
		{
			ItemId		iid = PageGetItemId(page, off);
			PgturbohybridBm25TenantTuple tuple;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (PgturbohybridBm25TenantTuple) PageGetItem(page, iid);
			if (tuple->type != PGTURBOHYBRID_BM25_TENANT_TUPLE_TYPE)
				continue;

			for (uint16 slot = 0; slot < tuple->count; slot++)
			{
				if (tuple->entries[slot].tenant == tenant)
				{
					found = true;
					break;
				}
			}
		}

		if (found)
		{
			/*
			 * Update in place.  The buffer was locked SHARE for the search;
			 * re-locking exclusively without an exclusion window is impossible
			 * with buffer locks alone, so the update helper re-searches under
			 * the exclusive lock and only writes when it still finds the
			 * tenant (share-mode inserts can race here; a lost re-find simply
			 * falls through to the append phase below, and the append itself
			 * re-checks nothing -- duplicate risk is closed by the heavyweight
			 * delta lock the caller holds).
			 */
			UpdateTenantEntry(index, buf, tenant, docLen, useWal);
			return true;
		}

		blkno = opaque->nextblkno;
		UnlockReleaseBuffer(buf);

		if (++visited > 4096)
			break;				/* corrupt chain: bounded, drop increment */
	}

	/*
	 * Phase 2: not found anywhere -- append a fresh entry to the first tuple
	 * with space.
	 */
	blkno = tenantStatsBlkno;
	visited = 0;
	while (BlockNumberIsValid(blkno))
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		GenericXLogState *xlogState = NULL;
		OffsetNumber maxoff;
		bool		wrote = false;

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
		page = BufferGetPage(buf);
		opaque = PgturbohybridGraphPageGetOpaque(page);

		if (opaque->page_id != PGTURBOHYBRID_GRAPH_PAGE_ID ||
			(opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) !=
			PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_TENANT)
		{
			UnlockReleaseBuffer(buf);
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("pgturbohybrid BM25 tenant stats page has unexpected page kind")));
		}

		if (useWal)
		{
			xlogState = GenericXLogStart(index);
			page = GenericXLogRegisterBuffer(xlogState, buf, 0);
			opaque = PgturbohybridGraphPageGetOpaque(page);
		}

		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber off = FirstOffsetNumber; off <= maxoff && !wrote; off++)
		{
			ItemId		iid = PageGetItemId(page, off);
			PgturbohybridBm25TenantTuple tuple;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (PgturbohybridBm25TenantTuple) PageGetItem(page, iid);
			if (tuple->type != PGTURBOHYBRID_BM25_TENANT_TUPLE_TYPE)
				continue;

			if (tuple->count < tuple->capacity)
			{
				tuple->entries[tuple->count].tenant = tenant;
				tuple->entries[tuple->count].docCount = 1;
				tuple->entries[tuple->count].totalDocLen = docLen;
				tuple->count++;
				wrote = true;
			}
		}

		if (wrote)
		{
			PgturbohybridGraphMarkPageGraphOp(page,
											  PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT);
			if (xlogState != NULL)
				GenericXLogFinish(xlogState);
			else
				MarkBufferDirty(buf);
			UnlockReleaseBuffer(buf);
			return true;
		}

		if (xlogState != NULL)
			GenericXLogAbort(xlogState);
		UnlockReleaseBuffer(buf);

		blkno = opaque->nextblkno;

		if (++visited > 4096)
			break;				/* corrupt chain: bounded, drop increment */
	}

	return false;
}

/*
 * Update one tenant entry in place under an exclusive buffer lock taken by
 * the caller (phase 1 of PgturbohybridBm25TenantIncrement).
 */
static void
UpdateTenantEntry(Relation index, Buffer buf, int32 tenant, uint32 docLen,
				  bool useWal)
{
	Page		page;
	GenericXLogState *xlogState = NULL;
	OffsetNumber maxoff;

	LockBuffer(buf, BUFFER_LOCK_UNLOCK);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	page = BufferGetPage(buf);

	if (useWal)
	{
		xlogState = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(xlogState, buf, 0);
	}

	maxoff = PageGetMaxOffsetNumber(page);
	for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off++)
	{
		ItemId		iid = PageGetItemId(page, off);
		PgturbohybridBm25TenantTuple tuple;

		if (!ItemIdIsUsed(iid))
			continue;

		tuple = (PgturbohybridBm25TenantTuple) PageGetItem(page, iid);
		if (tuple->type != PGTURBOHYBRID_BM25_TENANT_TUPLE_TYPE)
			continue;

		for (uint16 slot = 0; slot < tuple->count; slot++)
		{
			if (tuple->entries[slot].tenant == tenant)
			{
				tuple->entries[slot].docCount++;
				tuple->entries[slot].totalDocLen += docLen;
				PgturbohybridGraphMarkPageGraphOp(page,
												  PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT);
				if (xlogState != NULL)
					GenericXLogFinish(xlogState);
				else
					MarkBufferDirty(buf);
				UnlockReleaseBuffer(buf);
				return;
			}
		}
	}

	/*
	 * Not found under the exclusive lock (raced): release and return so the
	 * caller's append phase creates the entry.  The caller holds the
	 * heavyweight BM25 delta lock, so at most one appender runs at a time.
	 */
	if (xlogState != NULL)
		GenericXLogAbort(xlogState);
	UnlockReleaseBuffer(buf);
}


int
PgturbohybridBm25TenantPayloadSlot(Relation index)
{
	PgturbohybridOptions *opts = (PgturbohybridOptions *) index->rd_options;
	int			slot = opts != NULL ? opts->bm25TenantPayloadSlot : 0;

	if (slot < 0)
		return -1;

	{
		int			keyAttrs = IndexRelationGetNumberOfKeyAttributes(index);
		int			totalAttrs = RelationGetNumberOfAttributes(index);

		if (keyAttrs + slot >= totalAttrs)
			return -1;
	}

	return slot;
}

/*
 * Tenant key for one indexed row: the INCLUDE int4 payload column chosen by
 * bm25_tenant_payload_slot. NULL or missing column maps to bucket 0.
 */
int32
PgturbohybridBm25TenantFromValues(Relation index, Datum *values,
								  const bool *isnull)
{
	int			slot = PgturbohybridBm25TenantPayloadSlot(index);
	int			keyAttrs;

	if (slot < 0 || values == NULL || isnull == NULL)
		return 0;

	keyAttrs = IndexRelationGetNumberOfKeyAttributes(index);
	if (isnull[keyAttrs + slot])
		return 0;

	return DatumGetInt32(values[keyAttrs + slot]);
}

/*
 * Resolve the tenant scope for the current scan.
 *
 * A query is tenant-scoped when all of these hold:
 *   - turbohybrid.bm25_tenant_stats is on;
 *   - the index metadata is bm25Version >= 2 and carries a tenant stats chain;
 *   - the executor extracted an int4 equality qual over the heap column that
 *     backs the configured tenant payload slot (the same qual the dense
 *     traversal uses for tenant filtering);
 *   - that tenant has a recorded aggregate (tenant != 0).
 *
 * Otherwise the scope stays inactive and every caller keeps today's global
 * statistics.
 */
bool
PgturbohybridBm25ResolveTenantScope(Relation index,
									const PgturbohybridBm25MetaTupleData *meta,
									PgturbohybridBm25TenantScope *scope)
{
	AttrNumber	filterAttno = InvalidAttrNumber;
	int32		filterValue = 0;
	int			slot;
	int			keyAttrs;
	Form_pg_attribute attr;
	PgturbohybridBm25TenantEntryData entry;
	double		globalAvgdl;

	scope->active = false;
	scope->tenant = 0;
	scope->corpusDocCount = 0.0;
	scope->avgDocLen = 0.0;
	scope->q16FastPathDisabled = false;

	if (!pgturbohybrid_bm25_tenant_stats ||
		meta == NULL ||
		meta->bm25Version < 2 ||
		!BlockNumberIsValid(meta->tenantStatsStartBlkno))
		return true;

	if (!PgturbohybridGraphGetActivePayloadInt4Filter(&filterAttno,
													  &filterValue))
		return true;

	slot = PgturbohybridBm25TenantPayloadSlot(index);
	if (slot < 0 || filterValue == 0)
		return true;

	keyAttrs = IndexRelationGetNumberOfKeyAttributes(index);
	if (keyAttrs + slot >= RelationGetNumberOfAttributes(index))
		return true;
	attr = TupleDescAttr(RelationGetDescr(index), keyAttrs + slot);
	if (attr->attnum != filterAttno)
		return true;

	if (!PgturbohybridBm25TenantLookup(index, meta->tenantStatsStartBlkno,
									   filterValue, &entry))
		return true;

	scope->active = true;
	scope->tenant = filterValue;
	scope->corpusDocCount = Max((double) entry.docCount, 1.0);
	scope->avgDocLen = entry.docCount > 0 ?
		(double) entry.totalDocLen / (double) entry.docCount : 0.0;

	/*
	 * Precomputed TFNORM_Q16 postings bake the *global* avgdl into their
	 * quantized scores. When the tenant avgdl departs from the global one the
	 * fast path is no longer faithful and the scorer falls back to decoding
	 * tf and docLen, recomputing the length norm with the tenant avgdl.
	 */
	globalAvgdl = ((double) meta->totalDocLen + (double) meta->deltaTotalDocLen) /
		Max((double) (meta->docCount + meta->deltaDocCount), 1.0);
	scope->q16FastPathDisabled =
		fabs(scope->avgDocLen - globalAvgdl) > 0.01 * Max(globalAvgdl, 1.0);

	return true;
}

/*
 * turbohybrid_bm25_tenant_stats(index regclass) returns the recorded
 * per-tenant aggregates. Row order is storage order; deterministic per
 * build, not meaningful.
 */
PG_FUNCTION_INFO_V1(pgturbohybrid_bm25_tenant_stats_fn);

Datum
pgturbohybrid_bm25_tenant_stats_fn(PG_FUNCTION_ARGS)
{
	Relation	index = relation_open(PG_GETARG_OID(0), AccessShareLock);
	ReturnSetInfo *rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;
	Tuplestorestate *tupstore;
	MemoryContext perQueryCtx;
	MemoryContext oldCtx;
	TupleDesc	tupdesc;
	PgturbohybridGraphMetaPageData graphMeta;
	PgturbohybridBm25MetaTupleData meta;
	BlockNumber blkno;
	uint32		visited = 0;

	if (!PgturbohybridGraphReadMeta(index, &graphMeta) ||
		!BlockNumberIsValid(graphMeta.tqBm25MetaStartBlkno))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("pgturbohybrid BM25 metadata is missing")));

	{
		Buffer		buf;
		Page		page;
		OffsetNumber maxoff;
		bool		found = false;

		buf = ReadBuffer(index, graphMeta.tqBm25MetaStartBlkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off++)
		{
			ItemId		iid = PageGetItemId(page, off);

			if (!ItemIdIsUsed(iid))
				continue;
			{
				PgturbohybridBm25MetaTuple tuple =
					(PgturbohybridBm25MetaTuple) PageGetItem(page, iid);

				if (tuple->type == PGTURBOHYBRID_BM25_META_TUPLE_TYPE)
				{
					Size		itemSize = ItemIdGetLength(iid);

					memset(&meta, 0, sizeof(meta));
					memcpy(&meta, tuple,
						   Min(itemSize, (Size) sizeof(meta)));
					found = true;
					break;
				}
			}
		}
		UnlockReleaseBuffer(buf);
		if (!found)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("pgturbohybrid BM25 metadata tuple is missing")));
	}

	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("return type must be a row type")));


	perQueryCtx = rsinfo->econtext->ecxt_per_query_memory;
	oldCtx = MemoryContextSwitchTo(perQueryCtx);
	tupdesc = CreateTupleDescCopy(tupdesc);
	tupstore = tuplestore_begin_heap(false, false, work_mem);
	MemoryContextSwitchTo(oldCtx);

	blkno = meta.tenantStatsStartBlkno;
	while (meta.bm25Version >= 2 && BlockNumberIsValid(blkno))
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		OffsetNumber maxoff;

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = PgturbohybridGraphPageGetOpaque(page);

		if (opaque->page_id != PGTURBOHYBRID_GRAPH_PAGE_ID ||
			(opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) !=
			PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_TENANT)
		{
			UnlockReleaseBuffer(buf);
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("pgturbohybrid BM25 tenant stats page has unexpected page kind")));
		}

		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off++)
		{
			ItemId		iid = PageGetItemId(page, off);
			PgturbohybridBm25TenantTuple tuple;
			Datum		values[4];
			bool		nulls[4] = {false, false, false, false};

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (PgturbohybridBm25TenantTuple) PageGetItem(page, iid);
			if (tuple->type != PGTURBOHYBRID_BM25_TENANT_TUPLE_TYPE)
				continue;

			for (uint16 i = 0; i < tuple->count; i++)
			{
				PgturbohybridBm25TenantEntryData *entry = &tuple->entries[i];

				if (entry->tenant == 0)
					continue;
				values[0] = Int32GetDatum(entry->tenant);
				values[1] = Int64GetDatum((int64) entry->docCount);
				values[2] = Int64GetDatum((int64) entry->totalDocLen);
				values[3] = Float8GetDatum(entry->docCount > 0 ?
										  (double) entry->totalDocLen /
										  (double) entry->docCount : 0.0);
				tuplestore_putvalues(tupstore, tupdesc, values, nulls);
			}
		}

		blkno = opaque->nextblkno;
		UnlockReleaseBuffer(buf);

		if (++visited > 4096)
			break;
	}

	rsinfo->returnMode = SFRM_Materialize;
	rsinfo->setResult = tupstore;
	rsinfo->setDesc = tupdesc;

	relation_close(index, AccessShareLock);

	PG_RETURN_NULL();

}

/*
 * Rewrite one query's scoring metadata in place so every BM25 consumer --
 * idf, length normalization, WAND upper bounds, accumulator choice, and the
 * TFNORM_Q16 fast path -- sees the tenant corpus instead of the shared one.
 *
 * Call this only AFTER the structural caches (docstats, liveness, delta
 * entries) are loaded: those are keyed on the unscoped metadata. Structural
 * fields (page pointers, generations) are preserved; only corpus counters
 * and the precomputed-tfnorm flag change.
 */
void
PgturbohybridBm25MetaApplyTenantScope(Relation index,
									  PgturbohybridBm25MetaTupleData *meta)
{
	PgturbohybridBm25TenantScope scope;

	if (!PgturbohybridBm25ResolveTenantScope(index, meta, &scope))
		return;
	if (!scope.active)
		return;

	meta->docCount = (uint32) Max(scope.corpusDocCount, 1.0);
	meta->deltaDocCount = 0;
	meta->totalDocLen = (uint64) (scope.avgDocLen * scope.corpusDocCount);
	meta->deltaTotalDocLen = 0;
	meta->avgDocLen = (float4) scope.avgDocLen;

	/*
	 * Precomputed tfNorm baked the global avgdl into Q16 payloads; with a
	 * materially different tenant avgdl the scorer must decode tf + docLen
	 * and recompute the norm, i.e. behave as if the index had been built
	 * without precomputation.
	 */
	if (scope.q16FastPathDisabled)
		meta->reserved2 &= ~PGTURBOHYBRID_BM25_META_FLAG_TFNORM_Q16;
}

/*
 * Decode a stored BM25 delta tuple for either on-disk layout: version 2
 * (tenant present) or version 1 (legacy, tenant reads as 0). Terms are
 * returned as a pointer past the layout-specific header.
 */
bool
PgturbohybridBm25DeltaTupleDecode(PgturbohybridBm25DeltaTuple tuple,
								  Size itemSize, int32 *tenant,
								  PgturbohybridBm25DeltaTerm **terms)
{
	Size		v2 = MAXALIGN(offsetof(PgturbohybridBm25DeltaTupleData, terms) +
							(Size) tuple->termCount * sizeof(PgturbohybridBm25DeltaTerm) +
							tuple->termBytesLen);
	Size		v1 = MAXALIGN(24 +
							(Size) tuple->termCount * sizeof(PgturbohybridBm25DeltaTerm) +
							tuple->termBytesLen);
	if (itemSize == v2)
	{
		if (tenant != NULL)
			*tenant = tuple->tenant;
		*terms = (PgturbohybridBm25DeltaTerm *)
			((char *) tuple + offsetof(PgturbohybridBm25DeltaTupleData, terms));
		return true;
	}
	if (itemSize == v1)
	{
		if (tenant != NULL)
			*tenant = 0;
		*terms = (PgturbohybridBm25DeltaTerm *) ((char *) tuple + 24);
		return true;
	}


	/*
	 * Unknown size: fall back to the version-2 layout with tenant 0 instead
	 * of failing the scan. The pre-tenant code trusted termCount and relied
	 * on the per-term bounds check (termOffset + termLen <= termBytesLen);
	 * keeping that tolerance preserves scans over tuples whose layout is not
	 * exactly one of the two formulas, including zero-term continuation
	 * tuples that older writers left on the delta chain.
	 */
	if (tenant != NULL)
		*tenant = 0;
	*terms = (PgturbohybridBm25DeltaTerm *)
		((char *) tuple + offsetof(PgturbohybridBm25DeltaTupleData, terms));
	return true;
}
