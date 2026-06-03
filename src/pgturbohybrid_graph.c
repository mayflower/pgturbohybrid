#include "postgres.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>

#include "access/amapi.h"
#include "access/genam.h"
#include "access/relation.h"
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
#include "utils/jsonb.h"
#include "utils/numeric.h"
#include "utils/rel.h"
#include "utils/relcache.h"
#include "utils/selfuncs.h"
#include "utils/spccache.h"
#include "pgturbohybrid_am.h"
#include "pgturbohybrid_vector_compat.h"
#include "pgturbohybrid_quant.h"
#include "pgturbohybrid_bm25.h"
#include "pgturbohybrid_jsonb_compat.h"

#if PG_VERSION_NUM < 150000
#define MarkGUCPrefixReserved(x) EmitWarningsOnPlaceholders(x)
#endif

static relopt_enum_elt_def pgturbohybrid_routing_relopt_options[] = {
	{"auto", PGTURBOHYBRID_ROUTING_AUTO},
	{"graph", PGTURBOHYBRID_ROUTING_GRAPH},
	{"flat", PGTURBOHYBRID_ROUTING_FLAT},
	{NULL, 0}
};

static relopt_enum_elt_def pgturbohybrid_entry_sidecar_strategy_relopt_options[] = {
	{"hash", PGTURBOHYBRID_ENTRY_SIDECAR_HASH},
	{"farthest_code", PGTURBOHYBRID_ENTRY_SIDECAR_FARTHEST_CODE},
	{"level_covering", PGTURBOHYBRID_ENTRY_SIDECAR_LEVEL_COVERING},
	{"hybrid_level_covering", PGTURBOHYBRID_ENTRY_SIDECAR_HYBRID_LEVEL_COVERING},
	{NULL, 0}
};

static void
PgturbohybridIndexStatsJsonbAddKey(PgturbohybridJsonbState *state, const char *key)
{
	JsonbValue	value;

	value.type = jbvString;
	value.val.string.val = (char *) key;
	value.val.string.len = strlen(key);
	PgturbohybridJsonbPush(state, WJB_KEY, &value);
}

static void
PgturbohybridIndexStatsJsonbAddString(PgturbohybridJsonbState *state, const char *key,
								 const char *val)
{
	JsonbValue	value;

	PgturbohybridIndexStatsJsonbAddKey(state, key);
	value.type = jbvString;
	value.val.string.val = (char *) val;
	value.val.string.len = strlen(val);
	PgturbohybridJsonbPush(state, WJB_VALUE, &value);
}

static void
PgturbohybridIndexStatsJsonbAddBool(PgturbohybridJsonbState *state, const char *key,
							   bool val)
{
	JsonbValue	value;

	PgturbohybridIndexStatsJsonbAddKey(state, key);
	value.type = jbvBool;
	value.val.boolean = val;
	PgturbohybridJsonbPush(state, WJB_VALUE, &value);
}

static void
PgturbohybridIndexStatsJsonbAddUInt32(PgturbohybridJsonbState *state, const char *key,
								 uint32 val)
{
	JsonbValue	value;

	PgturbohybridIndexStatsJsonbAddKey(state, key);
	value.type = jbvNumeric;
	value.val.numeric = DatumGetNumeric(DirectFunctionCall1(int8_numeric,
															Int64GetDatum((int64) val)));
	PgturbohybridJsonbPush(state, WJB_VALUE, &value);
}

static void
PgturbohybridIndexStatsJsonbAddUInt64(PgturbohybridJsonbState *state, const char *key,
								 uint64 val)
{
	char		buf[32];
	JsonbValue	value;

	snprintf(buf, sizeof(buf), UINT64_FORMAT, val);
	PgturbohybridIndexStatsJsonbAddKey(state, key);
	value.type = jbvNumeric;
	value.val.numeric = DatumGetNumeric(DirectFunctionCall3(numeric_in,
															CStringGetDatum(buf),
															ObjectIdGetDatum(InvalidOid),
															Int32GetDatum(-1)));
	PgturbohybridJsonbPush(state, WJB_VALUE, &value);
}

static void
PgturbohybridIndexStatsJsonbAddFloat8(PgturbohybridJsonbState *state, const char *key,
								 double val)
{
	JsonbValue	value;

	PgturbohybridIndexStatsJsonbAddKey(state, key);
	value.type = jbvNumeric;
	value.val.numeric = DatumGetNumeric(DirectFunctionCall1(float8_numeric,
															Float8GetDatum(val)));
	PgturbohybridJsonbPush(state, WJB_VALUE, &value);
}

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

static const char *
PgturbohybridMemoryCachePolicyName(int policy)
{
	switch ((PgturbohybridNativeCachePolicy) policy)
	{
		case PGTURBOHYBRID_NATIVE_CACHE_POLICY_SHARED:
			return "shared";
		case PGTURBOHYBRID_NATIVE_CACHE_POLICY_PER_BACKEND:
			return "per_backend";
		case PGTURBOHYBRID_NATIVE_CACHE_POLICY_OFF:
			return "off";
		case PGTURBOHYBRID_NATIVE_CACHE_POLICY_AUTO:
		default:
			return "auto";
	}
}

static const char *
PgturbohybridMemoryCacheReasonName(PgturbohybridGraphNativeCacheReason reason)
{
	switch (reason)
	{
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_SHARED_FITS_MAX_MB:
			return "shared_fits_max_mb";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_AUTO_FITS_MAX_MB:
			return "auto_fits_max_mb";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_PER_BACKEND_FITS_MAX_MB:
			return "per_backend_fits_max_mb";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_POLICY_OFF:
			return "policy_off";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_EXCEEDS_MAX_MB:
			return "exceeds_max_mb";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_SHARED_BUILD_TIMEOUT:
			return "shared_build_timeout";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_SHARED_ATTACH_FAILED:
			return "shared_attach_failed";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_NONE:
		default:
			return "none";
	}
}

static double
PgturbohybridBytesToMb(uint64 bytes)
{
	return (double) bytes / (1024.0 * 1024.0);
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
bool		pgturbohybrid_dense_build_exact_distances_user_set = false;
int			pgturbohybrid_dense_build_distance = PGTURBOHYBRID_DENSE_BUILD_DISTANCE_AUTO;
int			pgturbohybrid_dense_build_neighbor_select = PGTURBOHYBRID_DENSE_BUILD_NEIGHBOR_SELECT_AUTO;
bool		pgturbohybrid_dense_hadamard_simd = true;
int			pgturbohybrid_dense_simd_force = PGTURBOHYBRID_SIMD_FORCE_AUTO;
int			pgturbohybrid_dense_query_split_impl = PGTURBOHYBRID_QUERY_SPLIT_IMPL_AUTO;
int			pgturbohybrid_dense_u8_split = PGTURBOHYBRID_U8_SPLIT_AUTO;
bool		pgturbohybrid_dense_u8_batch_x4 = true;
int			pgturbohybrid_native_cache_policy = PGTURBOHYBRID_NATIVE_CACHE_POLICY_AUTO;
int			pgturbohybrid_native_cache_max_mb = 2048;
int			pgturbohybrid_native_cache_warn_mb = 512;
char	   *pgturbohybrid_native_build_workers = "2";
int			pgturbohybrid_native_parallel_edge_build = PGTURBOHYBRID_NATIVE_PARALLEL_EDGE_BUILD_AUTO;
int			pgturbohybrid_native_segment_budget = PGTURBOHYBRID_NATIVE_SEGMENT_BUDGET_AUTO;
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
int			pgturbohybrid_dense_heap_rescore = PGTURBOHYBRID_DENSE_HEAP_RESCORE_OFF;
bool		pgturbohybrid_dense_heap_rescore_user_set = false;
int			pgturbohybrid_dense_residual_rerank_mode =
	PGTURBOHYBRID_DENSE_RESIDUAL_RERANK_CALIBRATED;
double		pgturbohybrid_dense_residual_rerank_weight = -1.0;
double		pgturbohybrid_dense_residual_rerank_max_adjust_ratio = 0.15;
int			pgturbohybrid_dense_adaptive_widening = PGTURBOHYBRID_DENSE_ADAPTIVE_WIDENING_OFF;
double		pgturbohybrid_dense_adaptive_widening_multiplier = 2.0;
double		pgturbohybrid_dense_adaptive_widening_max_multiplier = 4.0;
double		pgturbohybrid_dense_adaptive_min_gap = 0.0;
int			pgturbohybrid_dense_uncertainty_retry = PGTURBOHYBRID_DENSE_UNCERTAINTY_RETRY_OFF;
int			pgturbohybrid_dense_uncertainty_retry_max_passes = 1;
double		pgturbohybrid_dense_uncertainty_retry_multiplier = 1.5;
double		pgturbohybrid_dense_uncertainty_min_gap = 0.03;
bool		pgturbohybrid_warn_linear_fallback = true;
double		pgturbohybrid_linear_fallback_notice_threshold_ratio = 0.25;
int			pgturbohybrid_dense_local_expansion = PGTURBOHYBRID_DENSE_LOCAL_EXPANSION_OFF;
int			pgturbohybrid_dense_local_expansion_topn = 8;
int			pgturbohybrid_dense_local_expansion_max_neighbors = 256;
int			pgturbohybrid_payload_entry_seeding = PGTURBOHYBRID_PAYLOAD_ENTRY_SEEDING_AUTO;
int			pgturbohybrid_payload_entry_seed_count =
	PGTURBOHYBRID_DEFAULT_PAYLOAD_ENTRY_SEED_COUNT;
int			pgturbohybrid_graph_lock_tranche_id;
static relopt_kind pgturbohybrid_graph_relopt_kind;
static relopt_kind pgturbohybrid_relopt_kind;

const char *
PgturbohybridDenseBuildNeighborSelectName(int mode)
{
	switch ((PgturbohybridDenseBuildNeighborSelect) mode)
	{
		case PGTURBOHYBRID_DENSE_BUILD_NEIGHBOR_SELECT_FAST:
			return "fast";
		case PGTURBOHYBRID_DENSE_BUILD_NEIGHBOR_SELECT_HEURISTIC:
			return "heuristic";
		case PGTURBOHYBRID_DENSE_BUILD_NEIGHBOR_SELECT_AUTO:
			return "auto";
		default:
			return "unknown";
	}
}

const char *
PgturbohybridDenseBuildDistanceName(int mode)
{
	switch ((PgturbohybridDenseBuildDistance) mode)
	{
		case PGTURBOHYBRID_DENSE_BUILD_DISTANCE_AUTO:
			return "auto";
		case PGTURBOHYBRID_DENSE_BUILD_DISTANCE_CODE:
			return "code";
		case PGTURBOHYBRID_DENSE_BUILD_DISTANCE_EXACT:
			return "exact";
		default:
			return "unknown";
	}
}

const char *
PgturbohybridEntrySidecarStrategyName(int strategy)
{
	switch ((PgturbohybridEntrySidecarStrategy) strategy)
	{
		case PGTURBOHYBRID_ENTRY_SIDECAR_HASH:
			return "hash";
		case PGTURBOHYBRID_ENTRY_SIDECAR_FARTHEST_CODE:
			return "farthest_code";
		case PGTURBOHYBRID_ENTRY_SIDECAR_LEVEL_COVERING:
			return "level_covering";
		case PGTURBOHYBRID_ENTRY_SIDECAR_HYBRID_LEVEL_COVERING:
			return "hybrid_level_covering";
		default:
			return "unknown";
	}
}

const char *
PgturbohybridPayloadEntrySeedingName(int mode)
{
	switch ((PgturbohybridPayloadEntrySeedingMode) mode)
	{
		case PGTURBOHYBRID_PAYLOAD_ENTRY_SEEDING_OFF:
			return "off";
		case PGTURBOHYBRID_PAYLOAD_ENTRY_SEEDING_ON:
			return "on";
		case PGTURBOHYBRID_PAYLOAD_ENTRY_SEEDING_AUTO:
		default:
			return "auto";
	}
}

const char *
PgturbohybridNativeSegmentBudgetName(int mode)
{
	switch ((PgturbohybridNativeSegmentBudgetMode) mode)
	{
		case PGTURBOHYBRID_NATIVE_SEGMENT_BUDGET_AUTO:
			return "auto";
		case PGTURBOHYBRID_NATIVE_SEGMENT_BUDGET_OFF:
			return "off";
		case PGTURBOHYBRID_NATIVE_SEGMENT_BUDGET_SQRT:
			return "sqrt";
		case PGTURBOHYBRID_NATIVE_SEGMENT_BUDGET_LINEAR:
			return "linear";
		default:
			return "unknown";
	}
}

const char *
PgturbohybridBuildNeighborSelectReasonName(int reason)
{
	switch ((PgturbohybridBuildNeighborSelectReason) reason)
	{
		case PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_EXPLICIT_FAST:
			return "explicit_fast";
		case PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_EXPLICIT_HEURISTIC:
			return "explicit_heuristic";
		case PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_AUTO_LOWDIM:
			return "auto_lowdim";
		case PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_AUTO_BALANCED:
			return "auto_balanced";
		case PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_AUTO_QUALITY:
			return "auto_quality";
		case PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_AUTO_MATCHED_RECALL:
			return "auto_matched_recall";
		case PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_AUTO_LATENCY_HIGHDIM:
			return "auto_latency_highdim";
		case PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_UNKNOWN:
		default:
			return "unknown";
	}
}

const char *
PgturbohybridGraphDenseResidualRerankModeName(int mode)
{
	switch ((TqDenseResidualRerankMode) mode)
	{
		case PGTURBOHYBRID_DENSE_RESIDUAL_RERANK_OFF:
			return "off";
		case PGTURBOHYBRID_DENSE_RESIDUAL_RERANK_FIXED:
			return "fixed";
		case PGTURBOHYBRID_DENSE_RESIDUAL_RERANK_CALIBRATED:
			return "calibrated";
		default:
			return "unknown";
	}
}

const char *
PgturbohybridGraphDenseHeapRescoreName(int mode)
{
	switch ((TqDenseHeapRescoreMode) mode)
	{
		case PGTURBOHYBRID_DENSE_HEAP_RESCORE_OFF:
			return "off";
		case PGTURBOHYBRID_DENSE_HEAP_RESCORE_TOPK:
			return "topk";
		case PGTURBOHYBRID_DENSE_HEAP_RESCORE_BAND:
			return "band";
		case PGTURBOHYBRID_DENSE_HEAP_RESCORE_AUTO:
			return "auto";
		default:
			return "unknown";
	}
}

const char *
PgturbohybridGraphDenseHeapRescoreReasonName(int reason)
{
	switch ((TqDenseHeapRescoreReason) reason)
	{
		case PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_EXPLICIT_OFF:
			return "explicit_off";
		case PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_EXPLICIT_TOPK:
			return "explicit_topk";
		case PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_EXPLICIT_BAND:
			return "explicit_band";
		case PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_LATENCY:
			return "profile_latency";
		case PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_BALANCED_LOWDIM:
			return "profile_balanced_lowdim";
		case PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_BALANCED_HIGHDIM:
			return "profile_balanced_highdim";
		case PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_QUALITY_LOWDIM:
			return "profile_quality_lowdim";
		case PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_QUALITY_HIGHDIM:
			return "profile_quality_highdim";
		case PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_MATCHED_RECALL_LOWDIM:
			return "profile_matched_recall_lowdim";
		case PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_MATCHED_RECALL_HIGHDIM:
			return "profile_matched_recall_highdim";
		case PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_HIGH_RECALL:
			return "profile_high_recall";
		case PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_EXACT_STORAGE:
			return "exact_storage";
		case PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_UNKNOWN:
		default:
			return "unknown";
	}
}

const char *
PgturbohybridGraphDenseUncertaintyRetryModeName(int mode)
{
	switch ((TqDenseUncertaintyRetryMode) mode)
	{
		case PGTURBOHYBRID_DENSE_UNCERTAINTY_RETRY_AUTO:
			return "auto";
		case PGTURBOHYBRID_DENSE_UNCERTAINTY_RETRY_ON:
			return "on";
		case PGTURBOHYBRID_DENSE_UNCERTAINTY_RETRY_OFF:
		default:
			return "off";
	}
}

const char *
PgturbohybridGraphDenseUncertaintyRetryReasonName(int reason)
{
	switch ((TqDenseUncertaintyRetryReason) reason)
	{
		case PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_FORCED:
			return "forced";
		case PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_UNDERFILLED:
			return "underfilled";
		case PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_FLAT_TOP10:
			return "flat_top10";
		case PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_FLAT_BOUNDARY:
			return "flat_boundary";
		case PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_SIDECAR_UNUSED:
			return "entry_sidecar_unused";
		case PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_PAYLOAD_UNDERFILLED:
			return "payload_underfilled";
		case PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_RESIDUAL_REORDERED:
			return "residual_reordered";
		case PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_HEAP_REORDERED:
			return "heap_reordered";
		case PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_NONE:
		default:
			return "none";
	}
}

const char *
PgturbohybridGraphExactRescoreSourceName(int source)
{
	switch ((TqExactRescoreSource) source)
	{
		case PGTURBOHYBRID_EXACT_RESCORE_SOURCE_INDEX_EXACT:
			return "index_exact";
		case PGTURBOHYBRID_EXACT_RESCORE_SOURCE_HEAP:
			return "heap";
		case PGTURBOHYBRID_EXACT_RESCORE_SOURCE_RESIDUAL:
			return "residual";
		case PGTURBOHYBRID_EXACT_RESCORE_SOURCE_NONE:
		default:
			return "none";
	}
}


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
	add_bool_reloption(pgturbohybrid_relopt_kind, "entry_sidecar",
					   "Store a small build-time list of data-aware representative entry node IDs.",
					   PGTURBOHYBRID_DEFAULT_ENTRY_SIDECAR, AccessExclusiveLock);
	add_int_reloption(pgturbohybrid_relopt_kind, "entry_sidecar_representatives",
					  "Maximum representative node IDs stored when entry_sidecar is enabled.",
					  PGTURBOHYBRID_DEFAULT_ENTRY_SIDECAR_REPRESENTATIVES, 0,
					  PGTURBOHYBRID_GRAPH_MAX_ENTRY_SIDECAR_REPRESENTATIVES,
					  AccessExclusiveLock);
	add_enum_reloption(pgturbohybrid_relopt_kind, "entry_sidecar_strategy",
					   "Representative selection strategy for entry_sidecar.",
					   pgturbohybrid_entry_sidecar_strategy_relopt_options,
					   PGTURBOHYBRID_DEFAULT_ENTRY_SIDECAR_STRATEGY,
					   "Valid values are \"hash\", \"farthest_code\", \"level_covering\", and \"hybrid_level_covering\".",
					   AccessExclusiveLock);
	add_bool_reloption(pgturbohybrid_relopt_kind, "graph_backbone",
					   "Force adjacent level-0 graph edges during experimental dense graph builds.",
					   PGTURBOHYBRID_DEFAULT_GRAPH_BACKBONE, AccessExclusiveLock);
	add_bool_reloption(pgturbohybrid_relopt_kind, "residual_rerank",
					   "Store tiny per-vector sketches for experimental final-band dense reranking.",
					   PGTURBOHYBRID_DEFAULT_RESIDUAL_RERANK, AccessExclusiveLock);
	add_int_reloption(pgturbohybrid_relopt_kind, "residual_rerank_bytes",
					  "Per-vector sketch bytes stored when residual_rerank is enabled.",
					  PGTURBOHYBRID_DEFAULT_RESIDUAL_RERANK_BYTES, 0,
					  PGTURBOHYBRID_GRAPH_MAX_RESIDUAL_RERANK_BYTES,
					  AccessExclusiveLock);

	PgturbohybridGraphControlInit();
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_estimate_memory);
FUNCTION_PREFIX Datum
pgturbohybrid_estimate_memory(PG_FUNCTION_ARGS)
{
	Oid			indexOid = PG_GETARG_OID(0);
	Relation	index;
	PgturbohybridGraphMetaPageData meta;
	PgturbohybridGraphMemoryEstimate nativeEstimate;
	PgturbohybridBm25MemoryEstimate bm25Estimate;
	bool		hasNative = false;
	bool		hasBm25 = false;
	BlockNumber nblocks;
	uint64		nativeBytesPerBackend = 0;
	uint64		sharedCacheTotalBytes = 0;
	PgturbohybridJsonbState jsonState;

	index = relation_open(indexOid, AccessShareLock);
	nblocks = RelationGetNumberOfBlocks(index);
	if (nblocks > PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO)
		hasNative = PgturbohybridGraphEstimateMemory(index, &meta,
													 &nativeEstimate);
	else
	{
		memset(&meta, 0, sizeof(meta));
		memset(&nativeEstimate, 0, sizeof(nativeEstimate));
	}
	if (hasNative)
		hasBm25 = PgturbohybridBm25EstimateMemory(index, &meta, &bm25Estimate);
	else
		memset(&bm25Estimate, 0, sizeof(bm25Estimate));
	if (hasNative)
	{
		if (nativeEstimate.effectiveCachePolicy == PGTURBOHYBRID_NATIVE_CACHE_POLICY_SHARED)
		{
			nativeBytesPerBackend = nativeEstimate.sharedBackendViewBytes;
			sharedCacheTotalBytes = nativeEstimate.estimatedTotalBytes;
		}
		else
			nativeBytesPerBackend = nativeEstimate.estimatedTotalBytes;
	}

	PgturbohybridJsonbStateInit(&jsonState);
	PgturbohybridJsonbBeginObject(&jsonState);
	PgturbohybridIndexStatsJsonbAddString(&jsonState, "index",
										  RelationGetRelationName(index));
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "node_count",
										  hasNative ? meta.tqNodeCount : 0);
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "dimensions",
										  hasNative ? meta.dimensions : 0);
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "quantization_bits",
										  hasNative ? nativeEstimate.quantizationBits : 0);

	PgturbohybridIndexStatsJsonbAddKey(&jsonState, "native");
	PgturbohybridJsonbBeginObject(&jsonState);
	PgturbohybridIndexStatsJsonbAddBool(&jsonState, "available", hasNative);
	if (hasNative)
	{
		PgturbohybridIndexStatsJsonbAddString(&jsonState, "cache_policy",
											  PgturbohybridMemoryCachePolicyName(nativeEstimate.effectiveCachePolicy));
		PgturbohybridIndexStatsJsonbAddString(&jsonState, "configured_cache_policy",
											  PgturbohybridMemoryCachePolicyName(nativeEstimate.cachePolicy));
		PgturbohybridIndexStatsJsonbAddString(&jsonState, "cache_reason",
											  PgturbohybridMemoryCacheReasonName(nativeEstimate.cacheReason));
		PgturbohybridIndexStatsJsonbAddBool(&jsonState, "cache_exact_vectors",
											nativeEstimate.cacheExactVectors);
		PgturbohybridIndexStatsJsonbAddBool(&jsonState, "adjacency_estimated",
											nativeEstimate.adjacencyEstimated);
		PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "level_capacity",
											  nativeEstimate.levelCapacity);
		PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "code_page_count",
											  nativeEstimate.codePageCount);
		PgturbohybridIndexStatsJsonbAddUInt64(&jsonState, "code_bytes",
											  nativeEstimate.codeBytes);
		PgturbohybridIndexStatsJsonbAddUInt64(&jsonState, "adjacency_bytes",
											  nativeEstimate.adjacencyBytes);
		PgturbohybridIndexStatsJsonbAddUInt64(&jsonState, "exact_bytes",
											  nativeEstimate.exactBytes);
		PgturbohybridIndexStatsJsonbAddUInt64(&jsonState, "node_bytes",
											  nativeEstimate.nodeBytes);
		PgturbohybridIndexStatsJsonbAddUInt64(&jsonState, "visited_generation_bytes",
											  nativeEstimate.visitedGenerationBytes);
		PgturbohybridIndexStatsJsonbAddUInt64(&jsonState, "payload_bytes",
											  nativeEstimate.payloadBytes);
		PgturbohybridIndexStatsJsonbAddUInt64(&jsonState, "residual_bytes",
											  nativeEstimate.residualBytes);
		PgturbohybridIndexStatsJsonbAddUInt64(&jsonState, "page_map_bytes",
											  nativeEstimate.pageMapBytes);
		PgturbohybridIndexStatsJsonbAddUInt64(&jsonState, "shared_backend_view_bytes",
											  nativeEstimate.sharedBackendViewBytes);
		PgturbohybridIndexStatsJsonbAddUInt64(&jsonState, "estimated_total_bytes",
											  nativeEstimate.estimatedTotalBytes);
		PgturbohybridIndexStatsJsonbAddFloat8(&jsonState, "estimated_total_mb",
											  PgturbohybridBytesToMb(nativeEstimate.estimatedTotalBytes));
	}
	PgturbohybridJsonbCloseObject(&jsonState);

	PgturbohybridIndexStatsJsonbAddKey(&jsonState, "bm25");
	PgturbohybridJsonbBeginObject(&jsonState);
	PgturbohybridIndexStatsJsonbAddBool(&jsonState, "available", hasBm25);
	if (hasBm25)
	{
		PgturbohybridIndexStatsJsonbAddUInt64(&jsonState, "doc_lens_bytes",
											  bm25Estimate.docLensBytes);
		PgturbohybridIndexStatsJsonbAddUInt64(&jsonState, "heap_tids_bytes",
											  bm25Estimate.heapTidsBytes);
		PgturbohybridIndexStatsJsonbAddUInt64(&jsonState, "live_nodes_bytes",
											  bm25Estimate.liveNodesBytes);
		PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "lexicon_entries",
											  bm25Estimate.termCount);
		PgturbohybridIndexStatsJsonbAddUInt64(&jsonState, "lexicon_bytes",
											  bm25Estimate.lexiconBytes);
		PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "document_count",
											  bm25Estimate.docCount);
		PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "delta_document_count",
											  bm25Estimate.deltaDocCount);
		PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "term_tuple_count",
											  bm25Estimate.termTupleCount);
		PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "postings_pages",
											  bm25Estimate.postingsPages);
		PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "block_max_pages",
											  bm25Estimate.blockMaxPages);
		PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "delta_pages",
											  bm25Estimate.deltaPages);
		PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "delta_term_pages",
											  bm25Estimate.deltaTermPages);
		PgturbohybridIndexStatsJsonbAddUInt64(&jsonState, "estimated_base_cache_bytes",
											  bm25Estimate.estimatedBaseCacheBytes);
		PgturbohybridIndexStatsJsonbAddFloat8(&jsonState, "estimated_base_cache_mb",
											  PgturbohybridBytesToMb(bm25Estimate.estimatedBaseCacheBytes));
	}
	PgturbohybridJsonbCloseObject(&jsonState);

	PgturbohybridIndexStatsJsonbAddKey(&jsonState, "concurrency");
	PgturbohybridJsonbBeginObject(&jsonState);
	if (!hasNative)
	{
		PgturbohybridIndexStatsJsonbAddString(&jsonState, "per_backend_warning",
											  "native graph cache is unavailable for this relation");
		PgturbohybridIndexStatsJsonbAddString(&jsonState, "shared_cache_note",
											  "no shared native graph cache applies");
	}
	else if (nativeEstimate.effectiveCachePolicy == PGTURBOHYBRID_NATIVE_CACHE_POLICY_SHARED)
	{
		PgturbohybridIndexStatsJsonbAddString(&jsonState, "per_backend_warning",
											  "BM25 reader cache remains per backend; native resident data is shared when the shared cache attaches");
		PgturbohybridIndexStatsJsonbAddString(&jsonState, "shared_cache_note",
											  "shared native cache stores resident graph data once; backend view and scratch bytes remain per backend");
	}
	else if (nativeEstimate.effectiveCachePolicy == PGTURBOHYBRID_NATIVE_CACHE_POLICY_OFF)
	{
		PgturbohybridIndexStatsJsonbAddString(&jsonState, "per_backend_warning",
											  "native cache policy is off; scans use per-scan storage");
		PgturbohybridIndexStatsJsonbAddString(&jsonState, "shared_cache_note",
											  "set turbohybrid.native_cache_policy = shared to use the mmap-backed shared native cache when supported");
	}
	else
	{
		PgturbohybridIndexStatsJsonbAddString(&jsonState, "per_backend_warning",
											  "native cache can be duplicated once per backend; multiply by active backend count");
		PgturbohybridIndexStatsJsonbAddString(&jsonState, "shared_cache_note",
											  "set turbohybrid.native_cache_policy = shared to store resident native graph data once when supported");
	}
	PgturbohybridIndexStatsJsonbAddUInt64(&jsonState, "per_backend_total_bytes_per_backend",
										  nativeBytesPerBackend);
	PgturbohybridIndexStatsJsonbAddUInt64(&jsonState, "shared_backend_view_bytes_per_backend",
										  hasNative ? nativeEstimate.sharedBackendViewBytes : 0);
	PgturbohybridIndexStatsJsonbAddUInt64(&jsonState, "shared_cache_total_bytes",
										  sharedCacheTotalBytes);
	PgturbohybridIndexStatsJsonbAddUInt64(&jsonState, "bm25_total_bytes_per_backend",
										  hasBm25 ? bm25Estimate.estimatedBaseCacheBytes : 0);
	PgturbohybridJsonbCloseObject(&jsonState);

	relation_close(index, AccessShareLock);
	PG_RETURN_JSONB_P(PgturbohybridJsonbEndObject(&jsonState));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_index_stats);
FUNCTION_PREFIX Datum
pgturbohybrid_index_stats(PG_FUNCTION_ARGS)
{
	Oid			indexOid = PG_GETARG_OID(0);
	Relation	index;
	Buffer		buf;
	Page		page;
	PgturbohybridGraphMetaPageData meta;
	PgturbohybridGraphPageOpaque opaque;
	BlockNumber nblocks;
	uint16		storageKind;
	uint16		graphM;
	uint16		graphEfConstruction;
	uint16		graphEfSearch;
	uint16		graphOversampling;
	uint16		tqFlags;
	uint16		tqBits;
	uint16		entrySidecarCount;
	uint16		entrySidecarBytes;
	uint16		routingEntryCount;
	uint16		routingEntryBytes;
	uint16		residualRerankBytes;
	int			routing;
	BlockNumber tqBm25MetaStartBlkno;
	bool		hasLexicalKey;
	bool		hasBm25Meta = false;
	PgturbohybridBm25MetaTupleData bm25Meta;
	TqOptions  *opts;
	PgturbohybridJsonbState jsonState;

	index = index_open(indexOid, AccessShareLock);
	opts = (TqOptions *) index->rd_options;

	nblocks = RelationGetNumberOfBlocks(index);
	if (!PgturbohybridGraphReadMeta(index, &meta))
		elog(ERROR, "pgturbohybrid index is not valid");

	storageKind = meta.storageKind;
	graphM = meta.m;
	graphEfConstruction = meta.efConstruction;
	graphEfSearch = meta.graphEfSearch;
	graphOversampling = meta.graphOversampling;
	tqFlags = meta.tqFlags;
	tqBits = meta.tqBits != 0 ? meta.tqBits : PGTURBOHYBRID_DEFAULT_BITS;
	entrySidecarCount = meta.tqEntrySidecarCount;
	entrySidecarBytes = meta.tqEntrySidecarBytes;
	routingEntryCount = meta.tqRoutingEntryCount;
	routingEntryBytes = meta.tqRoutingEntryBytes;
	residualRerankBytes = meta.tqResidualRerankBytes;
	routing = opts != NULL ? opts->routing : PGTURBOHYBRID_ROUTING_AUTO;
	tqBm25MetaStartBlkno = meta.tqBm25MetaStartBlkno;
	hasLexicalKey = PgturbohybridIndexHasLexical(index);

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

	PgturbohybridJsonbStateInit(&jsonState);
	PgturbohybridJsonbBeginObject(&jsonState);
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "version", 1);
	PgturbohybridIndexStatsJsonbAddString(&jsonState, "profile",
										  PgturbohybridProfileName(pgturbohybrid_profile));
	PgturbohybridIndexStatsJsonbAddString(&jsonState, "storage_kind",
										  PgturbohybridGraphStorageKindName(storageKind));
	PgturbohybridIndexStatsJsonbAddString(&jsonState, "index_shape",
										  hasLexicalKey ? "hybrid" : "dense_only");
	PgturbohybridIndexStatsJsonbAddBool(&jsonState, "bm25_branch_available",
										hasLexicalKey && hasBm25Meta);
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "blocks", nblocks);
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "graph_m", graphM);
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "graph_ef_construction",
										  graphEfConstruction);
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "graph_ef_search",
										  graphEfSearch);
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "graph_oversampling",
										  graphOversampling);
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "native_segments",
										  meta.tqSegmentCount > 0 ? meta.tqSegmentCount : 1);
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "native_segment_bytes",
										  meta.tqSegmentBytes);
	PgturbohybridIndexStatsJsonbAddString(&jsonState, "routing",
										  PgturbohybridRoutingName(routing));
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "quantization_bits", tqBits);
	PgturbohybridIndexStatsJsonbAddBool(&jsonState, "exact_storage",
										(tqFlags & PGTURBOHYBRID_GRAPH_EXACT_FREE) == 0);
	PgturbohybridIndexStatsJsonbAddBool(&jsonState, "dense_build_exact_distances",
										(tqFlags & PGTURBOHYBRID_GRAPH_TQ_EXACT_BUILD) != 0);
	PgturbohybridIndexStatsJsonbAddString(&jsonState, "dense_build_distance_mode",
										  (tqFlags & PGTURBOHYBRID_GRAPH_TQ_EXACT_BUILD) != 0 ?
										  "exact" : "code");
	PgturbohybridIndexStatsJsonbAddString(&jsonState, "build_neighbor_select",
										  (tqFlags & PGTURBOHYBRID_GRAPH_TQ_FAST_BUILD_EDGES) != 0 ?
										  "fast" : "heuristic");
	PgturbohybridIndexStatsJsonbAddString(&jsonState, "build_neighbor_select_reason",
										  PgturbohybridBuildNeighborSelectReasonName(PGTURBOHYBRID_GRAPH_TQ_BUILD_NEIGHBOR_REASON(tqFlags)));
	PgturbohybridIndexStatsJsonbAddBool(&jsonState, "build_fast_edges",
										(tqFlags & PGTURBOHYBRID_GRAPH_TQ_FAST_BUILD_EDGES) != 0);
	PgturbohybridIndexStatsJsonbAddBool(&jsonState, "graph_backbone",
										(tqFlags & PGTURBOHYBRID_GRAPH_TQ_BACKBONE) != 0);
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "entry_sidecar_count",
										  entrySidecarCount);
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "entry_sidecar_bytes",
										  entrySidecarBytes);
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState,
										  "entry_sidecar_representatives_configured",
										  PgturbohybridGraphGetEntrySidecarRepresentatives(index));
	PgturbohybridIndexStatsJsonbAddString(&jsonState, "entry_sidecar_strategy",
										  PgturbohybridEntrySidecarStrategyName(
											  PgturbohybridGraphGetEntrySidecarStrategy(index)));
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "routing_entry_count",
										  routingEntryCount);
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "routing_entry_bytes",
										  routingEntryBytes);
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "residual_rerank_bytes",
										  residualRerankBytes);
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "residual_rerank_storage_bytes",
										  (uint32) residualRerankBytes * meta.tqNodeCount);
	PgturbohybridIndexStatsJsonbAddBool(&jsonState, "hybrid", hasBm25Meta);
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "bm25_document_count",
										  hasBm25Meta ? bm25Meta.docCount + bm25Meta.deltaDocCount : 0);
	PgturbohybridIndexStatsJsonbAddFloat8(&jsonState, "bm25_average_document_length",
										  hasBm25Meta ?
										  (double) (bm25Meta.totalDocLen + bm25Meta.deltaTotalDocLen) /
										  Max((double) (bm25Meta.docCount + bm25Meta.deltaDocCount), 1.0) : 0.0);
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "build_worker_count",
										  meta.buildWorkerCount);
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "build_scan_us",
										  (uint32) Min(meta.buildScanUs, (uint64) PG_UINT32_MAX));
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "build_correction_us",
										  (uint32) Min(meta.buildCorrectionUs, (uint64) PG_UINT32_MAX));
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "build_encode_us",
										  (uint32) Min(meta.buildEncodeUs, (uint64) PG_UINT32_MAX));
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "build_edge_us",
										  (uint32) Min(meta.buildEdgeUs, (uint64) PG_UINT32_MAX));
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "build_write_us",
										  (uint32) Min(meta.buildWriteUs, (uint64) PG_UINT32_MAX));
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "scan_us",
										  (uint32) Min(meta.buildScanUs, (uint64) PG_UINT32_MAX));
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "correction_us",
										  (uint32) Min(meta.buildCorrectionUs, (uint64) PG_UINT32_MAX));
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "encode_us",
										  (uint32) Min(meta.buildEncodeUs, (uint64) PG_UINT32_MAX));
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "edge_us",
										  (uint32) Min(meta.buildEdgeUs, (uint64) PG_UINT32_MAX));
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "write_us",
										  (uint32) Min(meta.buildWriteUs, (uint64) PG_UINT32_MAX));
	PgturbohybridIndexStatsJsonbAddUInt32(&jsonState, "worker_count",
										  meta.buildWorkerCount);
	index_close(index, AccessShareLock);

	PG_RETURN_JSONB_P(PgturbohybridJsonbEndObject(&jsonState));
}
