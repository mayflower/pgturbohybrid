#include "postgres.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>

#include "access/amapi.h"
#include "access/genam.h"
#include "access/reloptions.h"
#include "access/sysattr.h"
#include "commands/progress.h"
#include "commands/vacuum.h"
#include "fmgr.h"
#include "pgturbohybrid.h"
#include "lib/stringinfo.h"
#include "miscadmin.h"
#include "nodes/bitmapset.h"
#include "nodes/pg_list.h"
#include "optimizer/cost.h"
#include "optimizer/optimizer.h"
#include "storage/bufmgr.h"
#include "storage/lwlock.h"
#include "utils/float.h"
#include "utils/fmgrprotos.h"
#include "utils/guc.h"
#include "utils/rel.h"
#include "utils/relcache.h"
#include "utils/selfuncs.h"
#include "utils/spccache.h"
#include "pgturbohybrid_am.h"
#include "pgturbohybrid_vector_compat.h"
#include "pgturbohybrid_quant.h"
#include "pgturbohybrid_bm25.h"

#if PG_VERSION_NUM < 150000
#define MarkGUCPrefixReserved(x) EmitWarningsOnPlaceholders(x)
#endif

static relopt_enum_elt_def pgturbohybrid_routing_relopt_options[] = {
	{"auto", PGTURBOHYBRID_ROUTING_AUTO},
	{"graph", PGTURBOHYBRID_ROUTING_GRAPH},
	{"flat", PGTURBOHYBRID_ROUTING_FLAT},
	{NULL, 0}
};

static const char *
PgturbohybridRoutingName(int routing)
{
	switch (routing)
	{
		case PGTURBOHYBRID_ROUTING_AUTO:
			return "auto";
		case PGTURBOHYBRID_ROUTING_GRAPH:
			return "graph";
		case PGTURBOHYBRID_ROUTING_FLAT:
			return "flat";
		case PGTURBOHYBRID_ROUTING_LEGACY_GRAPH:
			return "legacy_graph";
		default:
			return "unknown";
	}
}

int			pgturbohybrid_ef_search = PGTURBOHYBRID_GRAPH_DEFAULT_EF_SEARCH;
int			pgturbohybrid_iterative_scan = PGTURBOHYBRID_GRAPH_ITERATIVE_SCAN_OFF;
int			pgturbohybrid_max_scan_tuples = 20000;
double		pgturbohybrid_scan_mem_multiplier = 1;
bool		pgturbohybrid_dense_graph_prefetch = true;
bool		pgturbohybrid_dense_graph_stack_scratch = true;
bool		pgturbohybrid_dense_graph_lowbit_popcnt = true;
bool		pgturbohybrid_dense_graph_i8mm = false;
bool		pgturbohybrid_dense_graph_avxvnni = true;
bool		pgturbohybrid_dense_graph_avx512vnni = true;
bool		pgturbohybrid_dense_graph_avx512vpopcntdq = true;
bool		pgturbohybrid_dense_weighted = false;
bool		pgturbohybrid_dense_renorm = false;
bool		pgturbohybrid_dense_query_1bit_asymmetric = false;
int			pgturbohybrid_dense_query_1bit_asymmetric_bits = 8;
bool		pgturbohybrid_dense_build_exact_distances = false;
bool		pgturbohybrid_dense_hadamard_simd = true;
bool		pgturbohybrid_dense_exact_avx512 = false;
int			pgturbohybrid_dense_simd_force = PGTURBOHYBRID_SIMD_FORCE_AUTO;
int			pgturbohybrid_dense_exact_simd_force = PGTURBOHYBRID_EXACT_SIMD_FORCE_AUTO;
int			pgturbohybrid_dense_graph_batch_scoring = PGTURBOHYBRID_GRAPH_BATCH_AUTO;
int			pgturbohybrid_dense_graph_batch_size = 4;
int			pgturbohybrid_dense_graph_avx512_weighted = PGTURBOHYBRID_GRAPH_AVX512_WEIGHTED_OFF;
int			pgturbohybrid_dense_graph_lookahead_prefetch = PGTURBOHYBRID_GRAPH_LOOKAHEAD_AUTO;
int			pgturbohybrid_dense_graph_lookahead_threshold_kb = 24576;
int			pgturbohybrid_dense_budget_policy = PGTURBOHYBRID_DENSE_BUDGET_AUTO;
int			pgturbohybrid_dense_max_candidate_multiplier = 4;
double		pgturbohybrid_dense_latency_multiplier = 1.5;
int			pgturbohybrid_dense_max_rescore_multiplier = 2;
int			pgturbohybrid_dense_rescore_band_policy = PGTURBOHYBRID_RESCORE_BAND_POLICY_AUTO;
int			pgturbohybrid_graph_lock_tranche_id;
static relopt_kind pgturbohybrid_graph_relopt_kind;
static relopt_kind pgturbohybrid_relopt_kind;


/*
 * Assign a tranche ID for our LWLocks. This only needs to be done by one
 * backend, as the tranche ID is remembered in shared memory.
 *
 * This shared memory area is very small, so we just allocate it from the
 * "slop" that PostgreSQL reserves for small allocations like this. If
 * this grows bigger, we should use a shmem_request_hook and
 * RequestAddinShmemSpace() to pre-reserve space for this.
 */
void
PgturbohybridGraphInitLockTranche(void)
{
	int		   *tranche_ids;
	bool		found;

	LWLockAcquire(AddinShmemInitLock, LW_EXCLUSIVE);
	tranche_ids = ShmemInitStruct("pgturbohybrid_graph LWLock ids",
								  sizeof(int) * 1,
								  &found);
	if (!found)
	{
#if PG_VERSION_NUM >= 190000
		tranche_ids[0] = LWLockNewTrancheId("PgturbohybridGraphBuild");
#else
		tranche_ids[0] = LWLockNewTrancheId();
#endif
	}
	pgturbohybrid_graph_lock_tranche_id = tranche_ids[0];
	LWLockRelease(AddinShmemInitLock);

#if PG_VERSION_NUM < 190000
	/* Per-backend registration of the tranche ID */
	LWLockRegisterTranche(pgturbohybrid_graph_lock_tranche_id, "PgturbohybridGraphBuild");
#endif
}

const char *
PgturbohybridGraphGraphWalModeName(void)
{
	return "generic_xlog_page_ops";
}

void
PgturbohybridGraphLogGraphWalRecord(Relation index, ForkNumber forkNum, BlockNumber blkno, uint16 graphOpKind)
{
	(void) index;
	(void) forkNum;
	(void) blkno;
	(void) graphOpKind;
}

/*
 * Initialize index options and variables
 */
void
PgturbohybridGraphInit(void)
{
	if (!process_shared_preload_libraries_in_progress)
		PgturbohybridGraphInitLockTranche();

	pgturbohybrid_graph_relopt_kind = add_reloption_kind();
	add_int_reloption(pgturbohybrid_graph_relopt_kind, "m", "Max number of connections",
					  PGTURBOHYBRID_GRAPH_DEFAULT_M, PGTURBOHYBRID_GRAPH_MIN_M, PGTURBOHYBRID_GRAPH_MAX_M, AccessExclusiveLock);
	add_int_reloption(pgturbohybrid_graph_relopt_kind, "ef_construction", "Size of the dynamic candidate list for construction",
					  PGTURBOHYBRID_GRAPH_DEFAULT_EF_CONSTRUCTION, PGTURBOHYBRID_GRAPH_MIN_EF_CONSTRUCTION, PGTURBOHYBRID_GRAPH_MAX_EF_CONSTRUCTION, AccessExclusiveLock);

	pgturbohybrid_relopt_kind = add_reloption_kind();
	add_enum_reloption(pgturbohybrid_relopt_kind, "routing", "pgturbohybrid routing mode",
					   pgturbohybrid_routing_relopt_options, PGTURBOHYBRID_ROUTING_AUTO,
					   "Valid values are \"auto\", \"graph\", and \"flat\".",
					   AccessExclusiveLock);
	add_int_reloption(pgturbohybrid_relopt_kind, "graph_m", "Max number of graph connections",
					  PGTURBOHYBRID_DEFAULT_GRAPH_M, PGTURBOHYBRID_GRAPH_MIN_M, PGTURBOHYBRID_GRAPH_MAX_M, AccessExclusiveLock);
	add_int_reloption(pgturbohybrid_relopt_kind, "graph_ef_construction", "Size of the dynamic graph candidate list for construction",
					  PGTURBOHYBRID_DEFAULT_GRAPH_EF_CONSTRUCTION, PGTURBOHYBRID_GRAPH_MIN_EF_CONSTRUCTION, PGTURBOHYBRID_GRAPH_MAX_EF_CONSTRUCTION, AccessExclusiveLock);
	add_int_reloption(pgturbohybrid_relopt_kind, "graph_ef_search", "Size of the dynamic graph candidate list for search",
					  PGTURBOHYBRID_DEFAULT_GRAPH_EF_SEARCH, PGTURBOHYBRID_GRAPH_MIN_EF_SEARCH, PGTURBOHYBRID_GRAPH_MAX_EF_SEARCH, AccessExclusiveLock);
	add_int_reloption(pgturbohybrid_relopt_kind, "graph_oversampling", "Candidate oversampling multiplier for graph scans",
					  PGTURBOHYBRID_DEFAULT_GRAPH_OVERSAMPLING, 1, 1000, AccessExclusiveLock);
		add_int_reloption(pgturbohybrid_relopt_kind, "quantization_bits", "pgturbohybrid code bit width",
						  PGTURBOHYBRID_DEFAULT_INDEX_BITS, 1, PGTURBOHYBRID_DEFAULT_BITS, AccessExclusiveLock);
		add_bool_reloption(pgturbohybrid_relopt_kind, "exact_storage",
						   "Store exact vectors in native pgturbohybrid graph indexes for final exact rescoring. Set off for compact exact-free quantized-only storage.",
						   PGTURBOHYBRID_DEFAULT_EXACT_STORAGE, AccessExclusiveLock);

	PgturbohybridGraphControlInit();
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_index_stats);
FUNCTION_PREFIX Datum
pgturbohybrid_index_stats(PG_FUNCTION_ARGS)
{
	Oid			indexOid = PG_GETARG_OID(0);
	Relation	index;
	Buffer		buf;
	Page		page;
	PgturbohybridGraphMetaPage metap;
	PgturbohybridGraphPageOpaque opaque;
	BlockNumber nblocks;
	uint16		storageKind;
	uint16		graphM;
	uint16		graphEfConstruction;
	uint16		graphEfSearch;
	uint16		graphOversampling;
	uint16		tqFlags;
	uint16		tqBits;
	int			routing;
	BlockNumber tqBm25MetaStartBlkno;
	bool		hasBm25Meta = false;
	PgturbohybridBm25MetaTupleData bm25Meta;
	TqOptions  *opts;
	StringInfoData json;

	index = index_open(indexOid, AccessShareLock);
	opts = (TqOptions *) index->rd_options;

	nblocks = RelationGetNumberOfBlocks(index);
	buf = ReadBuffer(index, PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	metap = PgturbohybridGraphPageGetMeta(page);

	if (unlikely(metap->magicNumber != PGTURBOHYBRID_GRAPH_MAGIC_NUMBER))
		elog(ERROR, "pgturbohybrid index is not valid");

	storageKind = metap->storageKind;
	graphM = metap->m;
	graphEfConstruction = metap->efConstruction;
	graphEfSearch = metap->graphEfSearch;
	graphOversampling = metap->graphOversampling;
	tqFlags = metap->tqFlags;
	tqBits = metap->tqBits != 0 ? metap->tqBits : PGTURBOHYBRID_DEFAULT_BITS;
	routing = opts != NULL ? opts->routing : PGTURBOHYBRID_ROUTING_AUTO;
	tqBm25MetaStartBlkno = metap->tqBm25MetaStartBlkno > PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO ?
		metap->tqBm25MetaStartBlkno : InvalidBlockNumber;

	UnlockReleaseBuffer(buf);

	if (BlockNumberIsValid(tqBm25MetaStartBlkno))
	{
		bool		foundBm25Tuple = false;

		if (tqBm25MetaStartBlkno >= nblocks)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("pgturbohybrid BM25 metadata pointer is invalid"),
					 errdetail("Metapage points to block %u, but the index has only %u blocks.",
							   tqBm25MetaStartBlkno, nblocks)));

		buf = ReadBuffer(index, tqBm25MetaStartBlkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = PgturbohybridGraphPageGetOpaque(page);

		if (opaque->page_id != PGTURBOHYBRID_GRAPH_PAGE_ID ||
			(opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) != PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_META)
		{
			UnlockReleaseBuffer(buf);
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("pgturbohybrid BM25 metadata pointer is invalid"),
					 errdetail("Metapage points to block %u, which is not a BM25 metadata page.",
							   tqBm25MetaStartBlkno)));
		}

		for (OffsetNumber off = FirstOffsetNumber;
			 off <= PageGetMaxOffsetNumber(page);
			 off = OffsetNumberNext(off))
		{
			ItemId		iid = PageGetItemId(page, off);
			PgturbohybridBm25MetaTuple tuple;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (PgturbohybridBm25MetaTuple) PageGetItem(page, iid);
			if (tuple->type == PGTURBOHYBRID_BM25_META_TUPLE_TYPE)
			{
				bm25Meta = *tuple;
				hasBm25Meta = true;
				foundBm25Tuple = true;
				break;
			}
		}

		UnlockReleaseBuffer(buf);
		if (!foundBm25Tuple)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("pgturbohybrid BM25 metadata tuple is missing"),
					 errdetail("Metapage points to BM25 metadata block %u, but no metadata tuple was found.",
						   tqBm25MetaStartBlkno)));
	}

	initStringInfo(&json);
	appendStringInfo(&json,
					 "{\"version\":1,"
					 "\"profile\":\"%s\","
					 "\"storage_kind\":\"%s\","
					 "\"blocks\":%u,"
					 "\"graph_m\":%u,"
					 "\"graph_ef_construction\":%u,"
					 "\"graph_ef_search\":%u,"
					 "\"graph_oversampling\":%u,"
					 "\"routing\":\"%s\","
					 "\"quantization_bits\":%u,"
					 "\"exact_storage\":%s,"
					 "\"hybrid\":%s,"
					 "\"bm25_document_count\":%u,"
					 "\"bm25_average_document_length\":%.6g}",
					 PgturbohybridProfileName(pgturbohybrid_profile),
					 PgturbohybridGraphStorageKindName(storageKind),
					 nblocks,
					 graphM,
					 graphEfConstruction,
					 graphEfSearch,
					 graphOversampling,
					 PgturbohybridRoutingName(routing),
					 tqBits,
					 (tqFlags & PGTURBOHYBRID_GRAPH_EXACT_FREE) != 0 ? "false" : "true",
					 hasBm25Meta ? "true" : "false",
					 hasBm25Meta ? bm25Meta.docCount + bm25Meta.deltaDocCount : 0,
					 hasBm25Meta ? (double) (bm25Meta.totalDocLen + bm25Meta.deltaTotalDocLen) /
					 Max((double) (bm25Meta.docCount + bm25Meta.deltaDocCount), 1.0) : 0.0);
	index_close(index, AccessShareLock);

	PG_RETURN_DATUM(DirectFunctionCall1(jsonb_in, CStringGetDatum(json.data)));
}
