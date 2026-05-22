#include "postgres.h"

#include <limits.h>
#include <math.h>

#include "access/amapi.h"
#include "access/relscan.h"
#include "access/reloptions.h"
#include "access/sysattr.h"
#include "access/xact.h"
#include "catalog/namespace.h"
#include "catalog/pg_type_d.h"
#include "commands/vacuum.h"
#include "executor/executor.h"
#include "fmgr.h"
#include "nodes/bitmapset.h"
#include "nodes/execnodes.h"
#include "nodes/nodeFuncs.h"
#include "optimizer/cost.h"
#include "optimizer/optimizer.h"
#include "portability/instr_time.h"
#include "storage/lmgr.h"
#include "utils/builtins.h"
#include "utils/float.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/selfuncs.h"
#include "utils/spccache.h"

#include "hnsw.h"
#include "hybrid_query.h"
#include "tqgraph.h"
#include "tqhybrid.h"
#include "tqhybrid_bm25.h"

#define TQHYBRID_DEFAULT_BM25_K1 1.2
#define TQHYBRID_DEFAULT_BM25_B 0.75
#define TQHYBRID_DEFAULT_DENSE_K 400
#define TQHYBRID_DEFAULT_BM25_K 400
#define TQHYBRID_DEFAULT_RRF_K 60

#if PG_VERSION_NUM < 150000
#define MarkGUCPrefixReserved(x) EmitWarningsOnPlaceholders(x)
#endif

static relopt_kind tqhybrid_relopt_kind;
static ExecutorStart_hook_type prev_tqhybrid_ExecutorStart_hook = NULL;
static ExecutorEnd_hook_type prev_tqhybrid_ExecutorEnd_hook = NULL;
static List *tqhybrid_plannedstmt_stack = NIL;
static PlannedStmt *tqhybrid_current_plannedstmt = NULL;

bool		tqhybrid_enable_wand = true;
bool		tqhybrid_debug_stats = false;
int			tqhybrid_max_union_candidates = 100000;
int			tqhybrid_default_dense_k = TQHYBRID_DEFAULT_DENSE_K;
int			tqhybrid_default_bm25_k = TQHYBRID_DEFAULT_BM25_K;
int			tqhybrid_default_rrf_k = TQHYBRID_DEFAULT_RRF_K;
int			tqhybrid_force_fusion = 0;
int			tqhybrid_fusion_hash_threshold = 128;
int			tqhybrid_debug_postings_chunk_size = 0;
bool		tqhybrid_enable_exact_rescore_for_bm25_only = false;
int			tqhybrid_bm25_cache_max_mb = 0;
int			tqhybrid_bm25_hot_postings_cache_mb = 0;
int			tqhybrid_bm25_hot_postings_cache_min_df = 1024;
bool		tqhybrid_bm25_allow_lazy_impact_build = false;
int			tqhybrid_bm25_simd_force = TQHYBRID_BM25_SIMD_FORCE_AUTO;
bool		tqhybrid_bm25_force_full_sort = false;
int			tqhybrid_bm25_accumulator_mode = TQHYBRID_BM25_ACCUMULATOR_AUTO;
int			tqhybrid_bm25_dense_accumulator_threshold = 4096;
double		tqhybrid_bm25_dense_accumulator_df_ratio = 0.05;
int			tqhybrid_bm25_strategy = TQHYBRID_BM25_STRATEGY_AUTO;
int			tqhybrid_bm25_impact_or_mode = TQHYBRID_BM25_IMPACT_OR_MODE_EXACT_ONLY;
bool		tqhybrid_auto_budget = true;
int			tqhybrid_auto_budget_min_dense_k = 32;
int			tqhybrid_auto_budget_min_bm25_k = 32;
int			tqhybrid_auto_budget_limit_multiplier = 8;
int			tqhybrid_auto_budget_quality_cap = 400;
bool		tqhybrid_auto_bm25_budget = true;
int			tqhybrid_auto_bm25_budget_min = 32;
int			tqhybrid_auto_bm25_budget_max = 400;
bool		tqhybrid_auto_bm25_budget_dense_confidence = true;
static int	tqhybrid_bm25_hybrid_bound = 0;

typedef enum TqHybridBm25HybridBoundMode
{
	TQHYBRID_BM25_HYBRID_BOUND_OFF,
	TQHYBRID_BM25_HYBRID_BOUND_SAFE,
	TQHYBRID_BM25_HYBRID_BOUND_APPROX
} TqHybridBm25HybridBoundMode;

static relopt_enum_elt_def tqhybrid_routing_relopt_options[] = {
	{"auto", TQ_ROUTING_AUTO},
	{"graph", TQ_ROUTING_GRAPH},
	{"flat", TQ_ROUTING_FLAT},
	{(const char *) NULL}
};

const char *
TqHybridBm25SimdForceName(int force)
{
	switch ((TqHybridBm25SimdForce) force)
	{
		case TQHYBRID_BM25_SIMD_FORCE_AUTO:
			return "auto";
		case TQHYBRID_BM25_SIMD_FORCE_SCALAR:
			return "scalar";
		case TQHYBRID_BM25_SIMD_FORCE_AVX2:
			return "avx2";
		case TQHYBRID_BM25_SIMD_FORCE_NEON:
			return "neon";
		default:
			return "unknown";
	}
}

const char *
TqHybridBm25AccumulatorModeName(int mode)
{
	switch ((TqHybridBm25AccumulatorMode) mode)
	{
		case TQHYBRID_BM25_ACCUMULATOR_HASH:
			return "hash";
		case TQHYBRID_BM25_ACCUMULATOR_DENSE:
			return "node_generation_arrays";
		case TQHYBRID_BM25_ACCUMULATOR_AUTO:
			return "auto";
		default:
			return "unknown";
	}
}

const char *
TqHybridBm25StrategyName(int strategy)
{
	switch ((TqHybridBm25Strategy) strategy)
	{
		case TQHYBRID_BM25_STRATEGY_AUTO:
			return "auto";
		case TQHYBRID_BM25_STRATEGY_IMPACT:
			return "impact";
		case TQHYBRID_BM25_STRATEGY_IMPACT_OR:
			return "impact_or";
		case TQHYBRID_BM25_STRATEGY_WAND:
			return "wand";
		case TQHYBRID_BM25_STRATEGY_DAAT_SIMD:
			return "daat_simd";
		case TQHYBRID_BM25_STRATEGY_DAAT_HASH:
			return "daat_hash";
		default:
			return "unknown";
	}
}

const char *
TqHybridBm25RuntimeStrategyName(int strategy)
{
	switch ((TqHybridBm25RuntimeStrategy) strategy)
	{
		case TQHYBRID_BM25_RUNTIME_IMPACT_SINGLE:
			return "impact_single";
		case TQHYBRID_BM25_RUNTIME_IMPACT_OR:
			return "impact_or";
		case TQHYBRID_BM25_RUNTIME_IMPACT_SEEDED_WAND:
			return "impact_seeded_wand";
		case TQHYBRID_BM25_RUNTIME_WAND:
			return "wand";
		case TQHYBRID_BM25_RUNTIME_AND_RAREST_DRIVER:
			return "and_rarest_driver";
		case TQHYBRID_BM25_RUNTIME_DAAT_SIMD:
			return "daat_simd";
		case TQHYBRID_BM25_RUNTIME_DAAT_HASH:
			return "daat_hash";
		case TQHYBRID_BM25_RUNTIME_NONE:
		default:
			return "none";
	}
}

static const char *
TqHybridBm25HybridBoundModeName(int mode)
{
	switch ((TqHybridBm25HybridBoundMode) mode)
	{
		case TQHYBRID_BM25_HYBRID_BOUND_OFF:
			return "off";
		case TQHYBRID_BM25_HYBRID_BOUND_SAFE:
			return "safe";
		case TQHYBRID_BM25_HYBRID_BOUND_APPROX:
			return "approx";
		default:
			return "unknown";
	}
}

static void TqHybridExecutorStartHook(QueryDesc *queryDesc, int eflags);
static void TqHybridExecutorEndHook(QueryDesc *queryDesc);
static void TqHybridXactCallback(XactEvent event, void *arg);
static void TqHybridSubXactCallback(SubXactEvent event, SubTransactionId mySubid,
								 SubTransactionId parentSubid, void *arg);

static IndexBuildResult *tqhybridbuild(Relation heap, Relation index, IndexInfo *indexInfo);
static void tqhybridbuildempty(Relation index);
static bool tqhybridinsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid, Relation heap, IndexUniqueCheck checkUnique
#if PG_VERSION_NUM >= 140000
						   , bool indexUnchanged
#endif
						   , IndexInfo *indexInfo);
static IndexBulkDeleteResult *tqhybridbulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats, IndexBulkDeleteCallback callback, void *callback_state);
static IndexBulkDeleteResult *tqhybridvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats);
static IndexScanDesc tqhybridbeginscan(Relation index, int nkeys, int norderbys);
static void tqhybridrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys);
static bool tqhybridgettuple(IndexScanDesc scan, ScanDirection dir);
static void tqhybridendscan(IndexScanDesc scan);
static void tqhybridcostestimate(PlannerInfo *root, IndexPath *path, double loop_count,
								 Cost *indexStartupCost, Cost *indexTotalCost,
								 Selectivity *indexSelectivity, double *indexCorrelation,
								 double *indexPages);
static bytea *tqhybridoptions(Datum reloptions, bool validate);
static bool tqhybridvalidate(Oid opclassoid);
static void TqHybridEnsureOrderByStorage(IndexScanDesc scan, MemoryContext scanCtx);
static bool TqHybridPathHasFilter(IndexPath *path);
static bool TqHybridFindConstQueryWalker(Node *node, void *context);
static HybridQueryHeader *TqHybridFindConstQuery(List *indexorderbys);
static int TqHybridEstimateTsQueryTerms(TSQuery query);
static int	TqHybridCurrentLimit(void);

typedef struct TqHybridResult
{
	uint32		nodeId;
	ItemPointerData heaptid;
	double		denseDistance;
	double		denseSimilarity;
	double		bm25Score;
	double		fusedScore;
	int32		denseRank;
	int32		bm25Rank;
	bool		hasDense;
	bool		hasBm25;
	bool		exactScored;
} TqHybridResult;

typedef struct TqHybridMergeSlot
{
	bool		used;
	TqHybridResult result;
} TqHybridMergeSlot;

typedef struct TqHybridScanState
{
	HybridQueryHeader *query;
	TqHybridResult *results;
	int			resultCount;
	int			resultIndex;
	bool		collectDone;
	bool		active;
} TqHybridScanState;

typedef struct TqHybridLastScanStats
{
	char		fusion[16];
	uint32		denseCandidatesRequested;
	uint32		denseCandidatesEffective;
	bool		denseKDefaulted;
	uint32		denseCandidates;
	uint32		denseEffectiveResultTarget;
	uint32		denseEffectiveSearchEf;
	uint32		denseEffectiveRescoreBand;
	double		denseHighdimWideningMultiplier;
	int			denseWideningReason;
	int			denseBudgetPolicy;
	int			denseRescoreBandPolicy;
	uint32		bm25CandidatesRequested;
	uint32		bm25CandidatesEffective;
	bool		bm25KDefaulted;
	uint32		bm25Candidates;
	char		bm25BudgetReason[48];
	double		bm25DenseConfidence;
	int			bm25HybridBoundMode;
	uint32		bm25HybridBoundStopRank;
	uint32		bm25HybridBoundSkippedEstimated;
	double		bm25HybridBoundThreshold;
	bool		bm25HybridBoundSafe;
	uint32		rrfKRequested;
	uint32		rrfKEffective;
	bool		rrfKDefaulted;
	uint32		autoBudgetLimit;
	uint32		unionCandidates;
	uint32		finalResults;
	char		fusionStrategy[16];
	uint32		fusionCandidatesSeen;
	uint32		fusionHeapSize;
	uint32		bothMatch;
	uint32		denseOnly;
	uint32		bm25Only;
	uint64		graphVisitedNodes;
	uint64		graphScoredCodes;
	uint64		graphExactRescoreCount;
	uint64		graphPrepareUs;
	uint64		graphTraverseUs;
	uint64		graphEntryUs;
	uint64		graphBaseUs;
	uint64		graphBatchUs;
	uint64		graphHeapUs;
	uint64		graphFillUs;
	uint64		graphRescoreUs;
	uint64		graphSortUs;
	uint32		bm25Terms;
	uint64		bm25PostingsDecoded;
	uint64		bm25BlocksVisited;
	uint64		bm25BlocksSkipped;
	uint32		bm25CandidatesScored;
	uint64		bm25CacheBytes;
	uint32		bm25CacheLexiconEntries;
	bool		bm25CacheHit;
	uint64		bm25CacheBuildUs;
	bool		bm25CacheDocstatsLoaded;
	bool		bm25CacheLivenessLoaded;
	uint64		bm25HotPostingsCacheHits;
	uint64		bm25HotPostingsCacheMisses;
	uint64		bm25HotPostingsCacheBytes;
	uint64		bm25HotPostingsCacheEvictions;
	int			bm25DeltaLookupMode;
	uint64		bm25DeltaPagesScanned;
	uint64		bm25DeltaTermPagesRead;
	uint64		bm25DeltaBlocksVisited;
	uint64		bm25DeltaPostingsDecoded;
	uint64		bm25DeltaCacheBytes;
	uint32		bm25DeltaCacheTerms;
	bool		bm25DeltaCacheHit;
	uint64		bm25WandIterations;
		uint64		bm25WandThresholdUpdates;
		uint64		bm25WandActiveSorts;
		uint64		bm25WandHeapUpdates;
		uint64		bm25WandFullReorders;
		uint64		bm25WandBoundTighteningHits;
		int			bm25WandBoundType;
		uint64		bm25WandHeapReplacements;
		int			bm25Strategy;
		uint32		bm25AndDriverDf;
		uint32		bm25AndVerifiedCandidates;
		uint32		bm25AndRejectedCandidates;
		uint32		bm25ImpactTerms;
		uint32		bm25ImpactTiersRead;
		uint64		bm25ImpactPostingsRead;
		double		bm25ImpactRemainingUpperBound;
		bool		bm25ImpactEarlyStop;
		bool		bm25ImpactExactSafe;
		bool		bm25ImpactFullPostingsAvoided;
		bool		bm25ImpactLoadedFromStorage;
		bool		bm25ImpactBuiltLazily;
		uint64		bm25ImpactLazyPostingsScanned;
		int			bm25AccumulatorMode;
	uint64		bm25AccumulatorHashLookups;
	uint64		bm25AccumulatorDenseUpdates;
	uint64		bm25FinalHeapReplacements;
	uint32		bm25FinalSortedCount;
	bool		bm25FullSortAvoided;
	int			bm25QueryShape;
	int			bm25BooleanEvalMode;
	uint64		bm25BooleanEvalCalls;
	int			bm25DecodeKernel;
	int			bm25ScoreKernel;
	uint64		bm25SimdBlocks;
	uint64		bm25ScalarTailPostings;
	uint64		bm25Prefetches;
	uint64		denseElapsedUs;
	uint64		bm25ElapsedUs;
	uint64		fusionElapsedUs;
	uint64		elapsedUs;
} TqHybridLastScanStats;

static TqHybridLastScanStats tqhybrid_last_scan_stats;

static uint64
TqHybridElapsedUs(instr_time start)
{
	instr_time	duration;

	INSTR_TIME_SET_CURRENT(duration);
	INSTR_TIME_SUBTRACT(duration, start);
	return (uint64) INSTR_TIME_GET_MICROSEC(duration);
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(hybrid_last_scan_stats);

PlannedStmt *
TqHybridCurrentPlannedStmt(void)
{
	return tqhybrid_current_plannedstmt;
}

static void
TqHybridPushPlannedStmt(PlannedStmt *plannedstmt)
{
	MemoryContext oldCtx = MemoryContextSwitchTo(TopMemoryContext);

	tqhybrid_plannedstmt_stack =
		lcons(tqhybrid_current_plannedstmt, tqhybrid_plannedstmt_stack);
	tqhybrid_current_plannedstmt = plannedstmt;

	MemoryContextSwitchTo(oldCtx);
}

static void
TqHybridPopPlannedStmt(void)
{
	if (tqhybrid_plannedstmt_stack == NIL)
	{
		tqhybrid_current_plannedstmt = NULL;
		return;
	}

	tqhybrid_current_plannedstmt =
		(PlannedStmt *) linitial(tqhybrid_plannedstmt_stack);
	tqhybrid_plannedstmt_stack = list_delete_first(tqhybrid_plannedstmt_stack);
}

static void
TqHybridClearPlannedStmtStack(void)
{
	list_free(tqhybrid_plannedstmt_stack);
	tqhybrid_plannedstmt_stack = NIL;
	tqhybrid_current_plannedstmt = NULL;
}

static void
TqHybridExecutorStartHook(QueryDesc *queryDesc, int eflags)
{
	if (prev_tqhybrid_ExecutorStart_hook)
		prev_tqhybrid_ExecutorStart_hook(queryDesc, eflags);
	else
		standard_ExecutorStart(queryDesc, eflags);

	TqHybridPushPlannedStmt(queryDesc->plannedstmt);
}

static void
TqHybridExecutorEndHook(QueryDesc *queryDesc)
{
	TqHybridPopPlannedStmt();

	if (prev_tqhybrid_ExecutorEnd_hook)
		prev_tqhybrid_ExecutorEnd_hook(queryDesc);
	else
		standard_ExecutorEnd(queryDesc);
}

static void
TqHybridXactCallback(XactEvent event, void *arg)
{
	if (event == XACT_EVENT_ABORT || event == XACT_EVENT_PARALLEL_ABORT)
		TqHybridClearPlannedStmtStack();
}

static void
TqHybridSubXactCallback(SubXactEvent event, SubTransactionId mySubid,
						SubTransactionId parentSubid, void *arg)
{
	if (event == SUBXACT_EVENT_ABORT_SUB)
		TqHybridClearPlannedStmtStack();
}

static void
TqHybridValidateIndex(Relation index, IndexInfo *indexInfo)
{
	TupleDesc	desc = RelationGetDescr(index);
	Oid			vectorOid;
	Oid			denseType;
	Oid			lexicalType;

	if (indexInfo != NULL)
	{
		if (indexInfo->ii_NumIndexKeyAttrs != 2)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("turbohybrid indexes require exactly two key columns"),
					 errdetail("Use one vector column followed by one tsvector column.")));

		if (indexInfo->ii_Expressions != NIL)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("turbohybrid expression indexes are not supported yet")));
	}
	else if (index->rd_index != NULL && index->rd_index->indnkeyatts != 2)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("turbohybrid indexes require exactly two key columns"),
				 errdetail("Use one vector column followed by one tsvector column.")));

	if (desc->natts < 2)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("turbohybrid indexes require exactly two key columns")));

	denseType = TupleDescAttr(desc, 0)->atttypid;
	lexicalType = TupleDescAttr(desc, 1)->atttypid;
	vectorOid = TypenameGetTypid("vector");

	if (!OidIsValid(vectorOid) || denseType != vectorOid)
		ereport(ERROR,
				(errcode(ERRCODE_DATATYPE_MISMATCH),
				 errmsg("turbohybrid first key must be type vector"),
				 errdetail("Found %s.", format_type_be(denseType))));

	if (lexicalType != TSVECTOROID)
		ereport(ERROR,
				(errcode(ERRCODE_DATATYPE_MISMATCH),
				 errmsg("turbohybrid second key must be type tsvector"),
				 errdetail("Found %s.", format_type_be(lexicalType))));
}

static ScanKey
TqHybridDenseOrderBys(ScanKey orderbys, int norderbys)
{
	ScanKey		denseOrderbys;

	if (orderbys == NULL || norderbys <= 0)
		return orderbys;

	denseOrderbys = palloc(sizeof(ScanKeyData) * norderbys);
	memcpy(denseOrderbys, orderbys, sizeof(ScanKeyData) * norderbys);

	for (int i = 0; i < norderbys; i++)
	{
		HybridQueryHeader *query;
		Vector	   *vectorQuery;

		if (denseOrderbys[i].sk_flags & SK_ISNULL)
			continue;

		query = DatumGetHybridQuery(denseOrderbys[i].sk_argument);
		HybridQueryValidate(query);

		vectorQuery = HybridQueryGetVector(query);
		if (vectorQuery == NULL)
			continue;

		denseOrderbys[i].sk_argument = PointerGetDatum(vectorQuery);
	}

	return denseOrderbys;
}

static void
TqHybridEnsureOrderByStorage(IndexScanDesc scan, MemoryContext scanCtx)
{
	MemoryContext oldCtx;

	if (scan->numberOfOrderBys <= 0)
		return;

	oldCtx = MemoryContextSwitchTo(scanCtx);
	scan->xs_orderbyvals = palloc0(sizeof(Datum) * scan->numberOfOrderBys);
	scan->xs_orderbynulls = palloc0(sizeof(bool) * scan->numberOfOrderBys);
	MemoryContextSwitchTo(oldCtx);
}

static int
TqHybridNodeCompare(const void *a, const void *b)
{
	const TqHybridResult *ra = (const TqHybridResult *) a;
	const TqHybridResult *rb = (const TqHybridResult *) b;

	return (ra->nodeId > rb->nodeId) - (ra->nodeId < rb->nodeId);
}

static int
TqHybridScoreCompare(const void *a, const void *b)
{
	const TqHybridResult *ra = (const TqHybridResult *) a;
	const TqHybridResult *rb = (const TqHybridResult *) b;
	int			raBoth = (ra->hasDense && ra->hasBm25) ? 1 : 0;
	int			rbBoth = (rb->hasDense && rb->hasBm25) ? 1 : 0;
	int32		raDenseRank = ra->hasDense ? ra->denseRank : INT_MAX;
	int32		rbDenseRank = rb->hasDense ? rb->denseRank : INT_MAX;
	int32		raBm25Rank = ra->hasBm25 ? ra->bm25Rank : INT_MAX;
	int32		rbBm25Rank = rb->hasBm25 ? rb->bm25Rank : INT_MAX;

	if (ra->fusedScore > rb->fusedScore)
		return -1;
	if (ra->fusedScore < rb->fusedScore)
		return 1;
	if (raBoth != rbBoth)
		return rbBoth - raBoth;
	if (raDenseRank != rbDenseRank)
		return (raDenseRank > rbDenseRank) - (raDenseRank < rbDenseRank);
	if (raBm25Rank != rbBm25Rank)
		return (raBm25Rank > rbBm25Rank) - (raBm25Rank < rbBm25Rank);
	return (ra->nodeId > rb->nodeId) - (ra->nodeId < rb->nodeId);
}

static bool
TqHybridScoreWorse(const TqHybridResult *left, const TqHybridResult *right)
{
	return TqHybridScoreCompare(left, right) > 0;
}

static uint32
TqHybridNextPowerOfTwo(uint32 value)
{
	uint32		result = 1;

	while (result < value)
		result <<= 1;
	return result;
}

static uint32
TqHybridFusionHashSlotCount(uint32 itemCount)
{
	uint32		target;

	if (itemCount == 0)
		return 16;

	/*
	 * Keep enough slack for linear probing without doubling the table for the
	 * common 100 dense + 100 BM25 case. The +1 guarantees at least one empty
	 * slot even when itemCount is already a power of two.
	 */
	target = itemCount + itemCount / 4 + 1;
	return TqHybridNextPowerOfTwo(Max(target, 16));
}

static uint32
TqHybridHashNodeId(uint32 nodeId)
{
	return nodeId * UINT32_C(2654435761);
}

static TqHybridResult *
TqHybridFindMergeSlot(TqHybridMergeSlot *slots, uint32 mask, uint32 nodeId)
{
	uint32		slotNo = TqHybridHashNodeId(nodeId) & mask;

	for (;;)
	{
		TqHybridMergeSlot *slot = &slots[slotNo];

		if (!slot->used)
		{
			slot->used = true;
			slot->result.nodeId = nodeId;
			return &slot->result;
		}
		if (slot->result.nodeId == nodeId)
			return &slot->result;
		slotNo = (slotNo + 1) & mask;
	}
}

static int
TqHybridFinalTarget(HybridQueryHeader *query, int mergedCount)
{
	int			finalCount = mergedCount;

	if (tqhybrid_max_union_candidates > 0)
		finalCount = Min(finalCount, tqhybrid_max_union_candidates);
	if ((query->flags & HYBRID_QUERY_FLAG_FINAL_K_IS_SET) != 0)
		finalCount = Min(finalCount, query->finalK);
	return finalCount;
}

static int
TqHybridConstLimitValue(Node *limitCount)
{
	Const	   *constant;
	int64		value;

	if (limitCount == NULL || !IsA(limitCount, Const))
		return 0;

	constant = castNode(Const, limitCount);
	if (constant->constisnull)
		return 0;

	switch (constant->consttype)
	{
		case INT2OID:
			value = DatumGetInt16(constant->constvalue);
			break;
		case INT4OID:
			value = DatumGetInt32(constant->constvalue);
			break;
		case INT8OID:
			value = DatumGetInt64(constant->constvalue);
			break;
		default:
			return 0;
	}

	if (value <= 0)
		return 0;
	return value > INT_MAX ? INT_MAX : (int) value;
}

static int
TqHybridFindLimitInPlan(Plan *plan)
{
	int			limit;

	if (plan == NULL)
		return 0;

	if (IsA(plan, Limit))
	{
		Limit	   *limitPlan = castNode(Limit, plan);

		limit = TqHybridConstLimitValue(limitPlan->limitCount);
		if (limit > 0)
			return limit;
	}

	limit = TqHybridFindLimitInPlan(plan->lefttree);
	if (limit > 0)
		return limit;

	return TqHybridFindLimitInPlan(plan->righttree);
}

static int
TqHybridCurrentLimit(void)
{
	PlannedStmt *plannedstmt = TqHybridCurrentPlannedStmt();

	if (plannedstmt == NULL || plannedstmt->planTree == NULL)
		return 0;

	return TqHybridFindLimitInPlan(plannedstmt->planTree);
}

static int
TqHybridApplyAutoBudget(int requested, int limit, int minBudget,
						bool defaulted, bool branchPresent)
{
	int			target;

	if (!tqhybrid_auto_budget || !defaulted || !branchPresent ||
		requested <= 0 || limit <= 0)
		return requested;

	if (limit > INT_MAX / tqhybrid_auto_budget_limit_multiplier)
		target = INT_MAX;
	else
		target = limit * tqhybrid_auto_budget_limit_multiplier;

	target = Max(target, minBudget);
	if (tqhybrid_auto_budget_quality_cap > 0)
		target = Min(target, tqhybrid_auto_budget_quality_cap);

	return Min(requested, target);
}

static HybridQueryHeader *
TqHybridEffectiveQuery(HybridQueryHeader *query, int limit,
					   MemoryContext memoryContext)
{
	HybridQueryHeader *effective;
	Size		querySize = VARSIZE_ANY(query);
	bool		hasVector =
		(query->flags & HYBRID_QUERY_FLAG_HAS_VECTOR) != 0;
	bool		hasTsQuery =
		(query->flags & HYBRID_QUERY_FLAG_HAS_TSQUERY) != 0;

	effective = MemoryContextAlloc(memoryContext, querySize);
	memcpy(effective, query, querySize);

	effective->denseK = TqHybridApplyAutoBudget(query->denseK, limit,
												tqhybrid_auto_budget_min_dense_k,
												(query->flags & HYBRID_QUERY_FLAG_DENSE_K_DEFAULTED) != 0,
												hasVector);
	effective->bm25K = TqHybridApplyAutoBudget(query->bm25K, limit,
											   tqhybrid_auto_budget_min_bm25_k,
											   (query->flags & HYBRID_QUERY_FLAG_BM25_K_DEFAULTED) != 0,
											   hasTsQuery);

	return effective;
}

static int
TqHybridBudgetFinalTarget(HybridQueryHeader *query, int limit)
{
	if ((query->flags & HYBRID_QUERY_FLAG_FINAL_K_IS_SET) != 0 &&
		query->finalK > 0)
		return query->finalK;
	if (limit > 0)
		return limit;
	return Max(Min(query->denseK, query->bm25K), 1);
}

static double
TqHybridDenseConfidence(const TqDenseCandidate *dense, int denseCount,
						int finalTarget)
{
	int			compareIndex;
	int			compareTarget;

	if (dense == NULL || denseCount <= 1)
		return 0.0;

	compareTarget = Max(finalTarget * 4, 2);
	compareIndex = Min(compareTarget, denseCount) - 1;
	return Max(dense[0].similarity - dense[compareIndex].similarity, 0.0);
}

static void
TqHybridMaybeApplyDenseBm25Budget(HybridQueryHeader *query,
								  const TqDenseCandidate *dense,
								  int denseCount, int limit,
								  char *reason, Size reasonSize,
								  double *denseConfidence)
{
	bool		hasVector =
		(query->flags & HYBRID_QUERY_FLAG_HAS_VECTOR) != 0;
	bool		hasTsQuery =
		(query->flags & HYBRID_QUERY_FLAG_HAS_TSQUERY) != 0;
	bool		bm25Defaulted =
		(query->flags & HYBRID_QUERY_FLAG_BM25_K_DEFAULTED) != 0;
	int			finalTarget;
	int			target;

	if (reason != NULL && reasonSize > 0)
		strlcpy(reason, "unchanged", reasonSize);
	if (denseConfidence != NULL)
		*denseConfidence = 0.0;

	if (!tqhybrid_auto_budget)
	{
		if (reason != NULL && reasonSize > 0)
			strlcpy(reason, "auto_budget_off", reasonSize);
		return;
	}
	if (!tqhybrid_auto_bm25_budget)
	{
		if (reason != NULL && reasonSize > 0)
			strlcpy(reason, "auto_bm25_budget_off", reasonSize);
		return;
	}
	if (!bm25Defaulted)
	{
		if (reason != NULL && reasonSize > 0)
			strlcpy(reason, "explicit_bm25_k", reasonSize);
		return;
	}
	if (!hasVector || !hasTsQuery)
	{
		if (reason != NULL && reasonSize > 0)
			strlcpy(reason, "not_hybrid", reasonSize);
		return;
	}
	if (query->fusion != HYBRID_FUSION_RRF)
	{
		if (reason != NULL && reasonSize > 0)
			strlcpy(reason, "non_rrf", reasonSize);
		return;
	}
	if (!tqhybrid_auto_bm25_budget_dense_confidence)
	{
		if (reason != NULL && reasonSize > 0)
			strlcpy(reason, "dense_confidence_off", reasonSize);
		return;
	}
	if (query->bm25K <= tqhybrid_auto_bm25_budget_min)
		return;

	finalTarget = TqHybridBudgetFinalTarget(query, limit);
	if (denseCount < finalTarget)
	{
		if (reason != NULL && reasonSize > 0)
			strlcpy(reason, "dense_insufficient", reasonSize);
		return;
	}

	if (denseConfidence != NULL)
		*denseConfidence =
			TqHybridDenseConfidence(dense, denseCount, finalTarget);
	if (denseConfidence == NULL || *denseConfidence <= 0.0)
	{
		if (reason != NULL && reasonSize > 0)
			strlcpy(reason, "dense_flat", reasonSize);
		return;
	}

	target = Max(finalTarget * 4, tqhybrid_auto_bm25_budget_min);
	if (tqhybrid_auto_bm25_budget_max > 0)
		target = Min(target, tqhybrid_auto_bm25_budget_max);
	target = Max(target, 1);

	if (target < query->bm25K)
	{
		query->bm25K = target;
		if (reason != NULL && reasonSize > 0)
			strlcpy(reason, "dense_confident_common_terms", reasonSize);
	}
}

static void
TqHybridMaybeApplyBm25HybridBound(HybridQueryHeader *query, int denseCount,
								  int limit, uint32 *stopRank,
								  uint32 *skippedEstimated,
								  double *threshold, bool *safe)
{
	bool		hasVector =
		(query->flags & HYBRID_QUERY_FLAG_HAS_VECTOR) != 0;
	bool		hasTsQuery =
		(query->flags & HYBRID_QUERY_FLAG_HAS_TSQUERY) != 0;
	bool		bm25Defaulted =
		(query->flags & HYBRID_QUERY_FLAG_BM25_K_DEFAULTED) != 0;
	int			finalTarget;
	double		currentThreshold;
	double		rawStopRank;
	uint32		capRank;

	if (stopRank != NULL)
		*stopRank = 0;
	if (skippedEstimated != NULL)
		*skippedEstimated = 0;
	if (threshold != NULL)
		*threshold = 0.0;
	if (safe != NULL)
		*safe = tqhybrid_bm25_hybrid_bound !=
			TQHYBRID_BM25_HYBRID_BOUND_APPROX;

	if (tqhybrid_bm25_hybrid_bound == TQHYBRID_BM25_HYBRID_BOUND_OFF ||
		!hasVector || !hasTsQuery ||
		query->fusion != HYBRID_FUSION_RRF ||
		query->denseWeight <= 0.0 || query->bm25Weight <= 0.0 ||
		query->bm25K <= 0 || denseCount <= 0)
		return;

	finalTarget = TqHybridBudgetFinalTarget(query, limit);
	if (finalTarget <= 0 || denseCount < finalTarget)
		return;

	currentThreshold = query->denseWeight /
		((double) query->rrfK + (double) finalTarget);
	if (threshold != NULL)
		*threshold = currentThreshold;

	if (currentThreshold <= 0.0)
		return;

	/*
	 * The RRF rank bound alone is not exact-safe unless BM25 contributions
	 * for dense candidates are also preserved.  Truncating BM25 to the bound
	 * can otherwise change the final top-k when dense candidates are tied or
	 * appear after the truncated lexical head.  Keep safe mode conservative
	 * until the BM25 branch can explicitly score those dense candidates.
	 */
	if (tqhybrid_bm25_hybrid_bound == TQHYBRID_BM25_HYBRID_BOUND_SAFE)
		return;

	rawStopRank = floor(query->bm25Weight / currentThreshold -
						(double) query->rrfK);
	if (rawStopRank < 1.0)
		capRank = 1;
	else if (rawStopRank > (double) PG_UINT32_MAX)
		capRank = PG_UINT32_MAX;
	else
		capRank = (uint32) rawStopRank;
	capRank = Max(capRank, (uint32) finalTarget);

	if (tqhybrid_bm25_hybrid_bound == TQHYBRID_BM25_HYBRID_BOUND_SAFE &&
		denseCount != finalTarget)
		return;
	if (tqhybrid_bm25_hybrid_bound == TQHYBRID_BM25_HYBRID_BOUND_APPROX &&
		!bm25Defaulted)
		return;

	if (capRank > 0 && capRank < (uint32) query->bm25K)
	{
		if (stopRank != NULL)
			*stopRank = capRank;
		if (skippedEstimated != NULL)
			*skippedEstimated = (uint32) query->bm25K - capRank;
		query->bm25K = (int32) capRank;
	}
}

static void
TqHybridTopNHeapSiftUp(TqHybridResult *heap, int index)
{
	while (index > 0)
	{
		int			parent = (index - 1) / 2;
		TqHybridResult tmp;

		if (!TqHybridScoreWorse(&heap[index], &heap[parent]))
			break;
		tmp = heap[parent];
		heap[parent] = heap[index];
		heap[index] = tmp;
		index = parent;
	}
}

static void
TqHybridTopNHeapSiftDown(TqHybridResult *heap, int count, int index)
{
	for (;;)
	{
		int			left = index * 2 + 1;
		int			right = left + 1;
		int			worst = index;
		TqHybridResult tmp;

		if (left < count && TqHybridScoreWorse(&heap[left], &heap[worst]))
			worst = left;
		if (right < count && TqHybridScoreWorse(&heap[right], &heap[worst]))
			worst = right;
		if (worst == index)
			break;
		tmp = heap[index];
		heap[index] = heap[worst];
		heap[worst] = tmp;
		index = worst;
	}
}

static int
TqHybridSelectTopN(TqHybridResult *items, int itemCount, int target,
				   TqHybridResult **topItems, MemoryContext memoryContext)
{
	TqHybridResult *heap;
	int			heapCount = 0;

	if (target <= 0 || itemCount <= 0)
	{
		*topItems = palloc0(sizeof(TqHybridResult));
		return 0;
	}

	heap = MemoryContextAllocZero(memoryContext,
								  sizeof(TqHybridResult) * target);
	for (int i = 0; i < itemCount; i++)
	{
		if (heapCount < target)
		{
			heap[heapCount] = items[i];
			TqHybridTopNHeapSiftUp(heap, heapCount);
			heapCount++;
		}
		else if (TqHybridScoreCompare(&items[i], &heap[0]) < 0)
		{
			heap[0] = items[i];
			TqHybridTopNHeapSiftDown(heap, heapCount, 0);
		}
	}

	if (heapCount > 1)
		qsort(heap, heapCount, sizeof(TqHybridResult), TqHybridScoreCompare);
	*topItems = heap;
	return heapCount;
}

static void
TqHybridAddDenseCandidate(TqHybridResult *item, TqDenseCandidate *candidate)
{
	item->nodeId = candidate->nodeId;
	item->heaptid = candidate->heaptid;
	item->denseDistance = candidate->distance;
	item->denseSimilarity = candidate->similarity;
	item->denseRank = candidate->rank;
	item->hasDense = true;
	item->exactScored = candidate->exactScored;
}

static void
TqHybridAddBm25Candidate(TqHybridResult *item, TqHybridBm25Result *candidate)
{
	item->nodeId = candidate->nodeId;
	item->heaptid = candidate->heaptid;
	item->bm25Score = candidate->bm25Score;
	item->bm25Rank = candidate->rank;
	item->hasBm25 = true;
}

static double
TqHybridNormalize(double value, double minValue, double maxValue)
{
	if (maxValue <= minValue)
		return value > 0 ? 1.0 : 0.0;
	return (value - minValue) / (maxValue - minValue);
}

static void
TqHybridCheckBm25OnlyExactRescore(TqHybridScanState *state,
								  TqHybridResult *results, int count)
{
	if (!tqhybrid_enable_exact_rescore_for_bm25_only)
		return;
	if ((state->query->flags & HYBRID_QUERY_FLAG_HAS_VECTOR) == 0)
		return;

	for (int i = 0; i < count; i++)
	{
		if (results[i].hasBm25 && !results[i].hasDense)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("exact rescoring for BM25-only turbohybrid candidates is not implemented yet"),
					 errdetail("Candidate node %u matched BM25 but was outside the dense candidate set.",
							   results[i].nodeId),
					 errhint("Increase dense_k so candidates are scored by the dense branch.")));
	}
}

static void
TqHybridScoreResults(TqHybridScanState *state, TqHybridResult *results, int count)
{
	HybridQueryHeader *query = state->query;
	double		minDense = get_float8_infinity();
	double		maxDense = -get_float8_infinity();
	double		minBm25 = get_float8_infinity();
	double		maxBm25 = -get_float8_infinity();
	double		alpha = (query->flags & HYBRID_QUERY_FLAG_ALPHA_IS_SET) != 0 ?
		query->alpha : 0.5;
	uint16		fusion = tqhybrid_force_fusion != 0 ?
		tqhybrid_force_fusion : query->fusion;

	if (fusion == HYBRID_FUSION_WEIGHTED)
	{
		for (int i = 0; i < count; i++)
		{
			if (results[i].hasDense)
			{
				minDense = Min(minDense, results[i].denseSimilarity);
				maxDense = Max(maxDense, results[i].denseSimilarity);
			}
			if (results[i].hasBm25)
			{
				minBm25 = Min(minBm25, results[i].bm25Score);
				maxBm25 = Max(maxBm25, results[i].bm25Score);
			}
		}
	}

	for (int i = 0; i < count; i++)
	{
		if (fusion == HYBRID_FUSION_WEIGHTED)
		{
			double		denseNorm = results[i].hasDense ?
				TqHybridNormalize(results[i].denseSimilarity, minDense, maxDense) : 0.0;
			double		bm25Norm = results[i].hasBm25 ?
				TqHybridNormalize(results[i].bm25Score, minBm25, maxBm25) : 0.0;

			results[i].fusedScore = alpha * denseNorm + (1.0 - alpha) * bm25Norm;
		}
		else
		{
			results[i].fusedScore =
				query->denseWeight *
				(results[i].hasDense ? 1.0 / ((double) query->rrfK + results[i].denseRank) : 0.0) +
				query->bm25Weight *
				(results[i].hasBm25 ? 1.0 / ((double) query->rrfK + results[i].bm25Rank) : 0.0);
		}
	}
}

static void
TqHybridCollectScanResults(IndexScanDesc scan, TqHybridScanState *state)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	TqDenseCandidate *dense = NULL;
	TqHybridBm25Result *bm25 = NULL;
	TqHybridResult *items = NULL;
	TqHybridResult *merged = NULL;
	TqDenseCandidateStats denseStats;
	TqHybridBm25QueryStats bm25Stats;
	int			denseCount = 0;
	int			bm25Count = 0;
	int			itemCount = 0;
	int			mergedCount = 0;
	int			finalCount;
	bool		useHashTopN;
	uint32		fusionCandidatesSeen;
	MemoryContext oldCtx;
	TqHybridLastScanStats lastStats;
	instr_time	totalStart;
	instr_time	phaseStart;
	HybridQueryHeader *originalQuery = state->query;
	HybridQueryHeader *scanQuery;
	int			autoBudgetLimit;
	char		bm25BudgetReason[48];
	double		bm25DenseConfidence = 0.0;
	uint32		bm25HybridBoundStopRank = 0;
	uint32		bm25HybridBoundSkippedEstimated = 0;
	double		bm25HybridBoundThreshold = 0.0;
	bool		bm25HybridBoundSafe = true;

	if (state->collectDone)
		return;

	INSTR_TIME_SET_CURRENT(totalStart);
	memset(&denseStats, 0, sizeof(denseStats));
	memset(&bm25Stats, 0, sizeof(bm25Stats));
	memset(&lastStats, 0, sizeof(lastStats));
	strlcpy(bm25BudgetReason, "not_evaluated", sizeof(bm25BudgetReason));
	oldCtx = MemoryContextSwitchTo(so->tmpCtx);
	autoBudgetLimit = TqHybridCurrentLimit();
	scanQuery = TqHybridEffectiveQuery(originalQuery, autoBudgetLimit, so->tmpCtx);
	state->query = scanQuery;
	if ((scanQuery->flags & HYBRID_QUERY_FLAG_HAS_VECTOR) != 0 &&
		scanQuery->denseK > 0)
	{
		INSTR_TIME_SET_CURRENT(phaseStart);
		denseCount = TqGraphCollectDenseCandidates(scan, scanQuery->denseK,
												   &dense, so->tmpCtx,
												   &denseStats);
		lastStats.denseElapsedUs = TqHybridElapsedUs(phaseStart);
	}

	TqHybridMaybeApplyDenseBm25Budget(scanQuery, dense, denseCount,
									  autoBudgetLimit, bm25BudgetReason,
									  sizeof(bm25BudgetReason),
									  &bm25DenseConfidence);
	TqHybridMaybeApplyBm25HybridBound(scanQuery, denseCount,
									  autoBudgetLimit,
									  &bm25HybridBoundStopRank,
									  &bm25HybridBoundSkippedEstimated,
									  &bm25HybridBoundThreshold,
									  &bm25HybridBoundSafe);

	if ((scanQuery->flags & HYBRID_QUERY_FLAG_HAS_TSQUERY) != 0 &&
		scanQuery->bm25K > 0)
	{
		TqHybridOptions *opts = (TqHybridOptions *) scan->indexRelation->rd_options;
		bool		useWand = tqhybrid_enable_wand &&
			(opts == NULL || opts->bm25BlockMax);

		INSTR_TIME_SET_CURRENT(phaseStart);
		bm25Count = TqHybridBm25TopK(scan->indexRelation,
									 HybridQueryGetTsQuery(scanQuery),
									 scanQuery->bm25K, useWand, so->tmpCtx,
									 &bm25, &bm25Stats);
		lastStats.bm25ElapsedUs = TqHybridElapsedUs(phaseStart);
	}

	INSTR_TIME_SET_CURRENT(phaseStart);
	fusionCandidatesSeen = denseCount + bm25Count;
	useHashTopN = tqhybrid_fusion_hash_threshold >= 0 &&
		fusionCandidatesSeen >= (uint32) tqhybrid_fusion_hash_threshold;
	if (useHashTopN)
	{
		uint32		slotCount =
			TqHybridFusionHashSlotCount(fusionCandidatesSeen);
		uint32		slotMask = slotCount - 1;
		TqHybridMergeSlot *slots =
			palloc0(sizeof(TqHybridMergeSlot) * slotCount);

		for (int i = 0; i < denseCount; i++)
		{
			TqHybridResult *item =
				TqHybridFindMergeSlot(slots, slotMask, dense[i].nodeId);

			TqHybridAddDenseCandidate(item, &dense[i]);
		}
		for (int i = 0; i < bm25Count; i++)
		{
			TqHybridResult *item =
				TqHybridFindMergeSlot(slots, slotMask, bm25[i].nodeId);

			if (!item->hasDense)
				item->heaptid = bm25[i].heaptid;
			item->nodeId = bm25[i].nodeId;
			item->bm25Score = bm25[i].bm25Score;
			item->bm25Rank = bm25[i].rank;
			item->hasBm25 = true;
		}

		merged = palloc0(sizeof(TqHybridResult) *
						 Max(fusionCandidatesSeen, 1));
		for (uint32 i = 0; i < slotCount; i++)
		{
			TqHybridResult item;

				if (!slots[i].used)
					continue;
				item = slots[i].result;
				if ((scanQuery->flags & HYBRID_QUERY_FLAG_REQUIRE_BM25_MATCH) != 0 &&
					!item.hasBm25)
					continue;
			if (item.hasDense && item.hasBm25)
				lastStats.bothMatch++;
			else if (item.hasDense)
				lastStats.denseOnly++;
			else if (item.hasBm25)
				lastStats.bm25Only++;
			merged[mergedCount++] = item;
		}

		TqHybridCheckBm25OnlyExactRescore(state, merged, mergedCount);
		TqHybridScoreResults(state, merged, mergedCount);

		finalCount = TqHybridFinalTarget(scanQuery, mergedCount);
		if (finalCount < mergedCount)
			finalCount = TqHybridSelectTopN(merged, mergedCount, finalCount,
											&merged, so->tmpCtx);
		else if (mergedCount > 1)
			qsort(merged, mergedCount, sizeof(TqHybridResult),
				  TqHybridScoreCompare);
		strlcpy(lastStats.fusionStrategy, "hash_topn",
				sizeof(lastStats.fusionStrategy));
		lastStats.fusionHeapSize = finalCount;
	}
	else
	{
		items = palloc0(sizeof(TqHybridResult) *
						Max((int) fusionCandidatesSeen, 1));
		for (int i = 0; i < denseCount; i++)
			TqHybridAddDenseCandidate(&items[itemCount++], &dense[i]);
		for (int i = 0; i < bm25Count; i++)
			TqHybridAddBm25Candidate(&items[itemCount++], &bm25[i]);

		if (itemCount > 1)
			qsort(items, itemCount, sizeof(TqHybridResult), TqHybridNodeCompare);

		merged = palloc0(sizeof(TqHybridResult) * Max(itemCount, 1));
		for (int i = 0; i < itemCount;)
		{
			TqHybridResult item = items[i++];

			while (i < itemCount && items[i].nodeId == item.nodeId)
			{
				if (items[i].hasDense)
				{
					item.hasDense = true;
					item.denseDistance = items[i].denseDistance;
					item.denseSimilarity = items[i].denseSimilarity;
					item.denseRank = items[i].denseRank;
					item.exactScored = items[i].exactScored;
					item.heaptid = items[i].heaptid;
				}
				if (items[i].hasBm25)
				{
					item.hasBm25 = true;
					item.bm25Score = items[i].bm25Score;
					item.bm25Rank = items[i].bm25Rank;
					if (!item.hasDense)
						item.heaptid = items[i].heaptid;
				}
				i++;
			}

			if ((scanQuery->flags & HYBRID_QUERY_FLAG_REQUIRE_BM25_MATCH) != 0 &&
				!item.hasBm25)
				continue;
			if (item.hasDense && item.hasBm25)
				lastStats.bothMatch++;
			else if (item.hasDense)
				lastStats.denseOnly++;
			else if (item.hasBm25)
				lastStats.bm25Only++;
			merged[mergedCount++] = item;
		}

		TqHybridCheckBm25OnlyExactRescore(state, merged, mergedCount);
		TqHybridScoreResults(state, merged, mergedCount);
		if (mergedCount > 1)
			qsort(merged, mergedCount, sizeof(TqHybridResult),
				  TqHybridScoreCompare);

		finalCount = TqHybridFinalTarget(scanQuery, mergedCount);
		strlcpy(lastStats.fusionStrategy, "sort",
				sizeof(lastStats.fusionStrategy));
		lastStats.fusionHeapSize = finalCount;
	}

	lastStats.fusionElapsedUs = TqHybridElapsedUs(phaseStart);
	state->results = merged;
	state->resultCount = finalCount;
	state->resultIndex = 0;
	state->collectDone = true;

	strlcpy(lastStats.fusion,
			HybridQueryFusionName(tqhybrid_force_fusion != 0 ?
								  tqhybrid_force_fusion : scanQuery->fusion),
			sizeof(lastStats.fusion));
	lastStats.denseCandidatesRequested = originalQuery->denseK;
	lastStats.denseCandidatesEffective = scanQuery->denseK;
	lastStats.denseKDefaulted =
		(originalQuery->flags & HYBRID_QUERY_FLAG_DENSE_K_DEFAULTED) != 0;
	lastStats.denseCandidates = denseCount;
	lastStats.denseEffectiveResultTarget = denseStats.effectiveResultTarget;
	lastStats.denseEffectiveSearchEf = denseStats.effectiveSearchEf;
	lastStats.denseEffectiveRescoreBand = denseStats.effectiveRescoreBand;
	lastStats.denseHighdimWideningMultiplier =
		denseStats.highdimWideningMultiplier;
	lastStats.denseWideningReason = denseStats.wideningReason;
	lastStats.denseBudgetPolicy = denseStats.denseBudgetPolicy;
	lastStats.denseRescoreBandPolicy = denseStats.rescoreBandPolicy;
	lastStats.bm25CandidatesRequested = originalQuery->bm25K;
	lastStats.bm25CandidatesEffective = scanQuery->bm25K;
	lastStats.bm25KDefaulted =
		(originalQuery->flags & HYBRID_QUERY_FLAG_BM25_K_DEFAULTED) != 0;
	lastStats.bm25Candidates = bm25Count;
	strlcpy(lastStats.bm25BudgetReason, bm25BudgetReason,
			sizeof(lastStats.bm25BudgetReason));
	lastStats.bm25DenseConfidence = bm25DenseConfidence;
	lastStats.bm25HybridBoundMode = tqhybrid_bm25_hybrid_bound;
	lastStats.bm25HybridBoundStopRank = bm25HybridBoundStopRank;
	lastStats.bm25HybridBoundSkippedEstimated =
		bm25HybridBoundSkippedEstimated;
	lastStats.bm25HybridBoundThreshold = bm25HybridBoundThreshold;
	lastStats.bm25HybridBoundSafe = bm25HybridBoundSafe;
	lastStats.rrfKRequested = originalQuery->rrfK;
	lastStats.rrfKEffective = scanQuery->rrfK;
	lastStats.rrfKDefaulted =
		(originalQuery->flags & HYBRID_QUERY_FLAG_RRF_K_DEFAULTED) != 0;
	lastStats.autoBudgetLimit = autoBudgetLimit;
	lastStats.unionCandidates = mergedCount;
	lastStats.finalResults = finalCount;
	lastStats.fusionCandidatesSeen = fusionCandidatesSeen;
	lastStats.graphVisitedNodes = denseStats.visitedGraphNodes;
	lastStats.graphScoredCodes = denseStats.scoredCodes;
	lastStats.graphExactRescoreCount = denseStats.exactRescoreCount;
	lastStats.graphPrepareUs = denseStats.prepareUs;
	lastStats.graphTraverseUs = denseStats.traverseUs;
	lastStats.graphEntryUs = denseStats.entryUs;
	lastStats.graphBaseUs = denseStats.baseUs;
	lastStats.graphBatchUs = denseStats.batchUs;
	lastStats.graphHeapUs = denseStats.heapUs;
	lastStats.graphFillUs = denseStats.fillUs;
	lastStats.graphRescoreUs = denseStats.rescoreUs;
	lastStats.graphSortUs = denseStats.sortUs;
	lastStats.bm25Terms = bm25Stats.queryTerms;
	lastStats.bm25PostingsDecoded = bm25Stats.postingsDecoded;
	lastStats.bm25BlocksVisited = bm25Stats.blocksVisited;
	lastStats.bm25BlocksSkipped = bm25Stats.blocksSkipped;
	lastStats.bm25CandidatesScored = bm25Stats.candidatesScored;
	lastStats.bm25CacheBytes = bm25Stats.cacheBytes;
	lastStats.bm25CacheLexiconEntries = bm25Stats.cacheLexiconEntries;
	lastStats.bm25CacheHit = bm25Stats.cacheHit;
	lastStats.bm25CacheBuildUs = bm25Stats.cacheBuildUs;
	lastStats.bm25CacheDocstatsLoaded = bm25Stats.cacheDocstatsLoaded;
	lastStats.bm25CacheLivenessLoaded = bm25Stats.cacheLivenessLoaded;
	lastStats.bm25HotPostingsCacheHits = bm25Stats.hotPostingsCacheHits;
	lastStats.bm25HotPostingsCacheMisses = bm25Stats.hotPostingsCacheMisses;
	lastStats.bm25HotPostingsCacheBytes = bm25Stats.hotPostingsCacheBytes;
	lastStats.bm25HotPostingsCacheEvictions =
		bm25Stats.hotPostingsCacheEvictions;
	lastStats.bm25DeltaLookupMode = bm25Stats.deltaLookupMode;
	lastStats.bm25DeltaPagesScanned = bm25Stats.deltaPagesScanned;
	lastStats.bm25DeltaTermPagesRead = bm25Stats.deltaTermPagesRead;
	lastStats.bm25DeltaBlocksVisited = bm25Stats.deltaBlocksVisited;
	lastStats.bm25DeltaPostingsDecoded = bm25Stats.deltaPostingsDecoded;
	lastStats.bm25DeltaCacheBytes = bm25Stats.deltaCacheBytes;
	lastStats.bm25DeltaCacheTerms = bm25Stats.deltaCacheTerms;
	lastStats.bm25DeltaCacheHit = bm25Stats.deltaCacheHit;
	lastStats.bm25WandIterations = bm25Stats.wandIterations;
	lastStats.bm25WandThresholdUpdates = bm25Stats.wandThresholdUpdates;
	lastStats.bm25WandActiveSorts = bm25Stats.wandActiveSorts;
	lastStats.bm25WandHeapUpdates = bm25Stats.wandHeapUpdates;
	lastStats.bm25WandFullReorders = bm25Stats.wandFullReorders;
	lastStats.bm25WandBoundTighteningHits = bm25Stats.wandBoundTighteningHits;
	lastStats.bm25WandBoundType = bm25Stats.wandBoundType;
	lastStats.bm25WandHeapReplacements = bm25Stats.wandHeapReplacements;
	lastStats.bm25Strategy = bm25Stats.strategy;
	lastStats.bm25AndDriverDf = bm25Stats.andDriverDf;
	lastStats.bm25AndVerifiedCandidates = bm25Stats.andVerifiedCandidates;
	lastStats.bm25AndRejectedCandidates = bm25Stats.andRejectedCandidates;
	lastStats.bm25ImpactTerms = bm25Stats.impactTerms;
	lastStats.bm25ImpactTiersRead = bm25Stats.impactTiersRead;
	lastStats.bm25ImpactPostingsRead = bm25Stats.impactPostingsRead;
	lastStats.bm25ImpactRemainingUpperBound =
		bm25Stats.impactRemainingUpperBound;
	lastStats.bm25ImpactEarlyStop = bm25Stats.impactEarlyStop;
	lastStats.bm25ImpactExactSafe = bm25Stats.impactExactSafe;
	lastStats.bm25ImpactFullPostingsAvoided =
		bm25Stats.impactFullPostingsAvoided;
	lastStats.bm25ImpactLoadedFromStorage =
		bm25Stats.impactLoadedFromStorage;
	lastStats.bm25ImpactBuiltLazily = bm25Stats.impactBuiltLazily;
	lastStats.bm25ImpactLazyPostingsScanned =
		bm25Stats.impactLazyPostingsScanned;
	lastStats.bm25AccumulatorMode = bm25Stats.accumulatorMode;
	lastStats.bm25AccumulatorHashLookups = bm25Stats.accumulatorHashLookups;
	lastStats.bm25AccumulatorDenseUpdates = bm25Stats.accumulatorDenseUpdates;
	lastStats.bm25FinalHeapReplacements = bm25Stats.finalHeapReplacements;
	lastStats.bm25FinalSortedCount = bm25Stats.finalSortedCount;
	lastStats.bm25FullSortAvoided = bm25Stats.fullSortAvoided;
	lastStats.bm25QueryShape = bm25Stats.queryShape;
	lastStats.bm25BooleanEvalMode = bm25Stats.booleanEvalMode;
	lastStats.bm25BooleanEvalCalls = bm25Stats.booleanEvalCalls;
	lastStats.bm25DecodeKernel = bm25Stats.decodeKernel;
	lastStats.bm25ScoreKernel = bm25Stats.scoreKernel;
	lastStats.bm25SimdBlocks = bm25Stats.simdBlocks;
	lastStats.bm25ScalarTailPostings = bm25Stats.scalarTailPostings;
	lastStats.bm25Prefetches = bm25Stats.prefetches;
		lastStats.elapsedUs = TqHybridElapsedUs(totalStart);
		tqhybrid_last_scan_stats = lastStats;
		state->query = originalQuery;
		MemoryContextSwitchTo(oldCtx);
	}

static IndexBuildResult *
tqhybridbuild(Relation heap, Relation index, IndexInfo *indexInfo)
{
	IndexBuildResult *result;

	TqHybridValidateIndex(index, indexInfo);
	result = turboquantbuild(heap, index, indexInfo);

	TqHybridBm25BuildCollect(heap, index, indexInfo);

	return result;
}

static void
tqhybridbuildempty(Relation index)
{
	TqHybridValidateIndex(index, NULL);
	turboquantbuildempty(index);
}

static bool
tqhybridinsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid, Relation heap, IndexUniqueCheck checkUnique
#if PG_VERSION_NUM >= 140000
			   , bool indexUnchanged
#endif
			   , IndexInfo *indexInfo)
{
	Datum		value;
	const HnswTypeInfo *typeInfo = HnswGetTypeInfo(index);
	HnswMetaPageData meta;
	HnswSupport support;
	uint32		nodeId;

	(void) heap;
	(void) checkUnique;
#if PG_VERSION_NUM >= 140000
	(void) indexUnchanged;
#endif
	TqHybridValidateIndex(index, indexInfo);
	if (isnull[0])
		return false;

	HnswInitSupport(&support, index);
	if (!HnswFormIndexValue(&value, values, isnull, typeInfo, &support))
		return false;

	LockPage(index, HNSW_UPDATE_LOCK, ExclusiveLock);
	PG_TRY();
	{
		if (!TqGraphReadMeta(index, &meta))
			elog(ERROR, "turbohybrid native graph metapage is missing or invalid");
		nodeId = meta.tqNodeCount;
		TqGraphInsertValueInPlace(index, indexInfo, heap_tid, value,
								  values, isnull);
		if (!isnull[1])
			TqHybridBm25AppendDelta(index, nodeId, heap_tid, values[1]);
	}
	PG_CATCH();
	{
		UnlockPage(index, HNSW_UPDATE_LOCK, ExclusiveLock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	UnlockPage(index, HNSW_UPDATE_LOCK, ExclusiveLock);

	return true;
}

static IndexBulkDeleteResult *
tqhybridbulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats, IndexBulkDeleteCallback callback, void *callback_state)
{
	IndexBulkDeleteResult *result;

	if (HnswUseTqNativeGraph(info->index))
	{
		result = tqgraphbulkdelete(info, stats, callback, callback_state);
		TqHybridBm25InvalidateCache(info->index);
	}
	else
		result = hnswbulkdelete(info, stats, callback, callback_state);

	return result;
}

static IndexBulkDeleteResult *
tqhybridvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats)
{
	IndexBulkDeleteResult *result;

	if (HnswUseTqNativeGraph(info->index))
	{
		result = tqgraphvacuumcleanup(info, stats);
		(void) TqHybridBm25MaybeCompact(info->index);
		TqHybridBm25InvalidateCache(info->index);
	}
	else
		result = hnswvacuumcleanup(info, stats);

	return result;
}

static IndexScanDesc
tqhybridbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;

	TqHybridValidateIndex(index, NULL);
	scan = turboquantbeginscan(index, nkeys, norderbys);

	return scan;
}

static void
tqhybridrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	ScanKey		denseOrderbys = TqHybridDenseOrderBys(orderbys, norderbys);
	HybridQueryHeader *hybridQuery = NULL;
	bool		hasTextQuery = false;
	bool		hasVectorQuery = false;

	if (orderbys != NULL && norderbys > 0 &&
		(orderbys[0].sk_flags & SK_ISNULL) == 0)
	{
		hybridQuery = (HybridQueryHeader *) PG_DETOAST_DATUM_COPY(orderbys[0].sk_argument);
		HybridQueryValidate(hybridQuery);
		hasTextQuery = (hybridQuery->flags & HYBRID_QUERY_FLAG_HAS_TSQUERY) != 0;
		hasVectorQuery = (hybridQuery->flags & HYBRID_QUERY_FLAG_HAS_VECTOR) != 0;
	}

	turboquantrescan(scan, keys, nkeys,
					 hasVectorQuery ? denseOrderbys : NULL,
					 hasVectorQuery ? norderbys : 0);

	if (hasTextQuery)
	{
		HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
		MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);
		TqHybridScanState *state = palloc0(sizeof(TqHybridScanState));
		Size		querySize = VARSIZE_ANY(hybridQuery);

		state->query = palloc(querySize);
		memcpy(state->query, hybridQuery, querySize);
		state->active = true;
		so->tqHybridState = state;
		MemoryContextSwitchTo(oldCtx);
		TqHybridEnsureOrderByStorage(scan, so->tmpCtx);
	}
	else if (scan->opaque != NULL)
	{
		((HnswScanOpaque) scan->opaque)->tqHybridState = NULL;
		scan->xs_orderbyvals = NULL;
		scan->xs_orderbynulls = NULL;
	}
}

static bool
tqhybridgettuple(IndexScanDesc scan, ScanDirection dir)
{
	HnswScanOpaque so = (HnswScanOpaque) scan->opaque;
	TqHybridScanState *state = so != NULL ?
		(TqHybridScanState *) so->tqHybridState : NULL;
	bool		result;

	if (state != NULL && state->active)
	{
		TqHybridResult *item;

		Assert(ScanDirectionIsForward(dir));
		TqHybridCollectScanResults(scan, state);
		if (state->resultIndex >= state->resultCount)
			return false;

		item = &state->results[state->resultIndex++];
		scan->xs_heaptid = item->heaptid;
		scan->xs_recheck = false;
		scan->xs_recheckorderby = false;
		if (scan->xs_orderbyvals != NULL && scan->xs_orderbynulls != NULL &&
			scan->numberOfOrderBys > 0)
		{
			scan->xs_orderbyvals[0] = Float8GetDatum(-item->fusedScore);
			scan->xs_orderbynulls[0] = false;
		}
		if (so != NULL)
			so->returnedRows++;
		return true;
	}

	result = turboquantgettuple(scan, dir);

	return result;
}

static void
tqhybridendscan(IndexScanDesc scan)
{
	turboquantendscan(scan);
}

static bool
TqHybridPathHasFilter(IndexPath *path)
{
	int			denseAttno = path->indexinfo->indexkeys[0];
	int			lexicalAttno = path->indexinfo->indexkeys[1];
	ListCell   *lc;

	foreach(lc, path->indexinfo->indrestrictinfo)
	{
		RestrictInfo *rinfo = lfirst_node(RestrictInfo, lc);
		Bitmapset  *attrs = NULL;

		pull_varattnos((Node *) rinfo->clause, path->indexinfo->rel->relid,
						&attrs);

		if (attrs == NULL)
			continue;

		if (bms_membership(attrs) != BMS_SINGLETON ||
			(!bms_is_member(denseAttno - FirstLowInvalidHeapAttributeNumber, attrs) &&
			 !bms_is_member(lexicalAttno - FirstLowInvalidHeapAttributeNumber, attrs)))
			return true;
	}

	return false;
}

static bool
TqHybridFindConstQueryWalker(Node *node, void *context)
{
	HybridQueryHeader **query = (HybridQueryHeader **) context;
	Oid			hybridQueryOid;

	if (node == NULL || *query != NULL)
		return false;

	if (IsA(node, Const))
	{
		Const	   *constant = castNode(Const, node);

		if (constant->constisnull)
			return false;

		hybridQueryOid = TypenameGetTypid("hybrid_query");
		if (OidIsValid(hybridQueryOid) && constant->consttype == hybridQueryOid)
		{
			*query = DatumGetHybridQuery(constant->constvalue);
			HybridQueryValidate(*query);
			return true;
		}
	}

	return expression_tree_walker(node, TqHybridFindConstQueryWalker, context);
}

static HybridQueryHeader *
TqHybridFindConstQuery(List *indexorderbys)
{
	HybridQueryHeader *query = NULL;
	ListCell   *lc;

	foreach(lc, indexorderbys)
	{
		if (TqHybridFindConstQueryWalker((Node *) lfirst(lc), &query))
			break;
	}

	return query;
}

static int
TqHybridEstimateTsQueryTerms(TSQuery query)
{
	QueryItem  *items;
	int			termCount = 0;

	if (query == NULL)
		return 0;

	items = GETQUERY(query);
	for (int i = 0; i < query->size; i++)
	{
		if (items[i].type == QI_VAL)
			termCount++;
	}

	return Max(termCount, 1);
}

static void
tqhybridcostestimate(PlannerInfo *root, IndexPath *path, double loop_count,
					 Cost *indexStartupCost, Cost *indexTotalCost,
					 Selectivity *indexSelectivity, double *indexCorrelation,
					 double *indexPages)
{
	GenericCosts costs;
	Relation	index;
	TqHybridOptions *opts = NULL;
	HnswMetaPageData graphMeta;
	TqHybridBm25PlanningStats bm25Stats;
	HybridQueryHeader *query;
	double		tuples;
	double		denseK;
	double		bm25K;
	double		finalK;
	double		termCount;
	double		efSearch;
	double		graphOversampling;
	double		m;
	double		denseWork = 0.0;
	double		bm25Postings = 0.0;
	double		bm25Work = 0.0;
	double		fusionWork;
	double		filterMultiplier = 1.0;
	double		estimatedPages;
	double		spc_random_page_cost;
	double		spc_seq_page_cost;
	double		pageCost;
	double		cpuCost;
	double		totalWork;

	if (path->indexorderbys == NIL)
	{
		*indexStartupCost = get_float8_infinity();
		*indexTotalCost = get_float8_infinity();
		*indexSelectivity = 0;
		*indexCorrelation = 0;
		*indexPages = 0;
#if PG_VERSION_NUM >= 180000
		path->path.disabled_nodes = 2;
#endif
		return;
	}

	MemSet(&costs, 0, sizeof(costs));
	genericcostestimate(root, path, loop_count, &costs);
	MemSet(&bm25Stats, 0, sizeof(bm25Stats));
	query = TqHybridFindConstQuery(path->indexorderbys);

	index = index_open(path->indexinfo->indexoid, NoLock);
	opts = (TqHybridOptions *) index->rd_options;
	MemSet(&graphMeta, 0, sizeof(graphMeta));
	if (!TqGraphReadMeta(index, &graphMeta))
	{
		graphMeta.m = opts != NULL ? opts->m : HNSW_DEFAULT_M;
		graphMeta.graphEfSearch = opts != NULL ?
			opts->graphEfSearch : TQ_DEFAULT_GRAPH_EF_SEARCH;
		graphMeta.graphOversampling = opts != NULL ?
			opts->graphOversampling : TQ_DEFAULT_GRAPH_OVERSAMPLING;
	}

	if (query != NULL)
	{
		denseK = HybridQueryGetVector(query) != NULL ? query->denseK : 0;
		bm25K = HybridQueryGetTsQuery(query) != NULL ? query->bm25K : 0;
		finalK = (query->flags & HYBRID_QUERY_FLAG_FINAL_K_IS_SET) != 0 ?
			query->finalK : Max(denseK + bm25K, 1);
		termCount = TqHybridEstimateTsQueryTerms(HybridQueryGetTsQuery(query));
	}
	else
	{
		denseK = opts != NULL ? opts->hybridDefaultDenseK : TQHYBRID_DEFAULT_DENSE_K;
		bm25K = opts != NULL ? opts->hybridDefaultBm25K : TQHYBRID_DEFAULT_BM25_K;
		finalK = Max(denseK + bm25K, 1);
		termCount = 2;
	}
	if (bm25K > 0)
		(void) TqHybridBm25GetPlanningStats(index, &bm25Stats);

	tuples = Max(path->indexinfo->tuples, 1.0);
	m = graphMeta.m > 0 ? graphMeta.m :
		(opts != NULL ? opts->m : HNSW_DEFAULT_M);
	efSearch = graphMeta.graphEfSearch > 0 ? graphMeta.graphEfSearch :
		(opts != NULL ? opts->graphEfSearch : TQ_DEFAULT_GRAPH_EF_SEARCH);
	graphOversampling = graphMeta.graphOversampling > 0 ?
		graphMeta.graphOversampling :
		(opts != NULL ? opts->graphOversampling : TQ_DEFAULT_GRAPH_OVERSAMPLING);
	index_close(index, NoLock);

	if (TqHybridPathHasFilter(path))
		filterMultiplier = Min(10.0, 1.0 / Max(costs.indexSelectivity, 0.01));

	if (denseK > 0)
	{
		double		layer0Work = Max(efSearch, denseK * graphOversampling);
		double		entryWork = Max(1.0, log(tuples)) * Max(m, 1.0);

		denseWork = (entryWork + layer0Work) * filterMultiplier;
	}

	if (bm25K > 0)
	{
		double		corpusDocs = bm25Stats.hasBm25 ?
			Max((double) bm25Stats.docCount + bm25Stats.deltaDocCount, 1.0) :
			tuples;
		double		avgDf = bm25Stats.hasBm25 && bm25Stats.termCount > 0 ?
			(double) bm25Stats.termTupleCount / bm25Stats.termCount :
			corpusDocs * 0.05;

		avgDf = Max(avgDf, 8.0);
		bm25Postings = Min(corpusDocs * Max(termCount, 1.0),
						   avgDf * Max(termCount, 1.0));
		if (tqhybrid_enable_wand && bm25Stats.blockMaxPages > 0)
			bm25Postings *= 0.35;

		bm25Work = (bm25Postings + bm25K * Max(termCount, 1.0)) *
			filterMultiplier;
	}

	fusionWork = Max(finalK, denseK + bm25K);
	if (fusionWork > 1)
		fusionWork *= log(fusionWork) / log(2.0);

	totalWork = denseWork + bm25Work + fusionWork;
	get_tablespace_page_costs(path->indexinfo->reltablespace,
							  &spc_random_page_cost, &spc_seq_page_cost);
	estimatedPages = Min(costs.numIndexPages,
						 2.0 + denseWork / 256.0 + bm25Work / 512.0);
	pageCost = estimatedPages *
		(spc_seq_page_cost + (spc_random_page_cost - spc_seq_page_cost) * 0.25);
	cpuCost = totalWork * cpu_operator_cost + fusionWork * cpu_tuple_cost;

	*indexStartupCost = pageCost + cpuCost;
	*indexTotalCost = *indexStartupCost +
		Max(finalK, 1.0) * cpu_tuple_cost * Max(loop_count, 1.0);
	*indexSelectivity = costs.indexSelectivity;
	*indexCorrelation = costs.indexCorrelation;
	*indexPages = estimatedPages;
}

static bytea *
tqhybridoptions(Datum reloptions, bool validate)
{
	static const relopt_parse_elt tab[] = {
		{"graph_m", RELOPT_TYPE_INT, offsetof(TqHybridOptions, m)},
		{"graph_ef_construction", RELOPT_TYPE_INT, offsetof(TqHybridOptions, efConstruction)},
		{"routing", RELOPT_TYPE_ENUM, offsetof(TqHybridOptions, routing)},
		{"graph_ef_search", RELOPT_TYPE_INT, offsetof(TqHybridOptions, graphEfSearch)},
		{"graph_oversampling", RELOPT_TYPE_INT, offsetof(TqHybridOptions, graphOversampling)},
		{"tq_bits", RELOPT_TYPE_INT, offsetof(TqHybridOptions, tqBits)},
		{"tq_exact_storage", RELOPT_TYPE_BOOL, offsetof(TqHybridOptions, tqExactStorage)},
	};
	TqHybridOptions *opts = (TqHybridOptions *) build_reloptions(reloptions, validate,
																 tqhybrid_relopt_kind,
																 sizeof(TqHybridOptions),
																 tab, lengthof(tab));

	if (opts != NULL)
	{
		opts->graphRescoreBand = TQ_GRAPH_RESCORE_BAND_AUTO;
		opts->graphExactCache = TQ_GRAPH_EXACT_CACHE_AUTO;
		opts->graphReorder = TQ_GRAPH_REORDER_OFF;
		opts->tqWeighted = false;
		opts->tqQuantileFit = false;
		opts->tqRenorm = false;
		opts->bm25K1 = TQHYBRID_DEFAULT_BM25_K1;
		opts->bm25B = TQHYBRID_DEFAULT_BM25_B;
		opts->bm25BlockMax = true;
		opts->bm25PrecomputeTfNorm = true;
		opts->bm25ImpactHead = true;
		opts->bm25ImpactMinDf = 1024;
		opts->bm25ImpactHeadK = 2048;
		opts->bm25DeltaCompactionThreshold = 25;
		opts->hybridDefaultFusion = HYBRID_FUSION_RRF;
		opts->hybridDefaultDenseK = TQHYBRID_DEFAULT_DENSE_K;
		opts->hybridDefaultBm25K = TQHYBRID_DEFAULT_BM25_K;
		opts->hybridDefaultRrfK = TQHYBRID_DEFAULT_RRF_K;
	}

	if (validate && opts != NULL &&
		opts->tqBits != 1 && opts->tqBits != 2 && opts->tqBits != TQ_DEFAULT_BITS)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid value %d for option \"tq_bits\"", opts->tqBits),
				 errdetail("Valid values are \"1\", \"2\", and \"4\".")));

	return (bytea *) opts;
}

static bool
tqhybridvalidate(Oid opclassoid)
{
	(void) opclassoid;
	return true;
}

void
TqHybridInit(void)
{
	tqhybrid_relopt_kind = add_reloption_kind();
	prev_tqhybrid_ExecutorStart_hook = ExecutorStart_hook;
	ExecutorStart_hook = TqHybridExecutorStartHook;
	prev_tqhybrid_ExecutorEnd_hook = ExecutorEnd_hook;
	ExecutorEnd_hook = TqHybridExecutorEndHook;
	RegisterXactCallback(TqHybridXactCallback, NULL);
	RegisterSubXactCallback(TqHybridSubXactCallback, NULL);

	add_enum_reloption(tqhybrid_relopt_kind, "routing", "Turbohybrid dense routing mode",
					   tqhybrid_routing_relopt_options, TQ_ROUTING_AUTO,
					   "Valid values are \"auto\", \"graph\", and \"flat\".",
					   AccessExclusiveLock);
	add_int_reloption(tqhybrid_relopt_kind, "graph_m", "Max number of graph connections",
					  TQ_DEFAULT_GRAPH_M, HNSW_MIN_M, HNSW_MAX_M, AccessExclusiveLock);
	add_int_reloption(tqhybrid_relopt_kind, "graph_ef_construction", "Size of the dynamic graph candidate list for construction",
					  TQ_DEFAULT_GRAPH_EF_CONSTRUCTION, HNSW_MIN_EF_CONSTRUCTION, HNSW_MAX_EF_CONSTRUCTION, AccessExclusiveLock);
	add_int_reloption(tqhybrid_relopt_kind, "graph_ef_search", "Size of the dynamic graph candidate list for search",
					  TQ_DEFAULT_GRAPH_EF_SEARCH, HNSW_MIN_EF_SEARCH, HNSW_MAX_EF_SEARCH, AccessExclusiveLock);
	add_int_reloption(tqhybrid_relopt_kind, "graph_oversampling", "Candidate oversampling multiplier for graph scans",
					  TQ_DEFAULT_GRAPH_OVERSAMPLING, 1, 1000, AccessExclusiveLock);
		add_int_reloption(tqhybrid_relopt_kind, "tq_bits", "Turboquant code bit width",
						  TQ_DEFAULT_INDEX_BITS, 1, TQ_DEFAULT_BITS, AccessExclusiveLock);
		add_bool_reloption(tqhybrid_relopt_kind, "tq_exact_storage",
						   "Store exact vectors in the dense TurboQuant graph for final exact rescoring.",
						   TQ_DEFAULT_EXACT_STORAGE, AccessExclusiveLock);

		DefineCustomIntVariable("hybrid.default_dense_k", "Default dense candidate budget for hybrid_query callers",
								NULL, &tqhybrid_default_dense_k,
								TQHYBRID_DEFAULT_DENSE_K, 0, INT_MAX, PGC_USERSET, 0, NULL, NULL, NULL);
		DefineCustomIntVariable("hybrid.default_bm25_k", "Default BM25 candidate budget for hybrid_query callers",
								NULL, &tqhybrid_default_bm25_k,
								TQHYBRID_DEFAULT_BM25_K, 0, INT_MAX, PGC_USERSET, 0, NULL, NULL, NULL);
		DefineCustomIntVariable("hybrid.default_rrf_k", "Default RRF constant for hybrid_query callers",
								NULL, &tqhybrid_default_rrf_k,
								TQHYBRID_DEFAULT_RRF_K, 1, INT_MAX, PGC_USERSET, 0, NULL, NULL, NULL);
		MarkGUCPrefixReserved("hybrid");
	}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(turbohybridhandler);
Datum
hybrid_last_scan_stats(PG_FUNCTION_ARGS)
{
	StringInfoData json;

	initStringInfo(&json);
	appendStringInfo(&json,
						 "{\"fusion\":\"%s\","
						 "\"dense_candidates_requested\":%u,"
						 "\"dense_candidates_effective\":%u,"
						 "\"dense_k_defaulted\":%s,"
						 "\"dense_candidates\":%u,"
					 "\"dense_effective_result_target\":%u,"
					 "\"dense_effective_search_ef\":%u,"
					 "\"dense_effective_rescore_band\":%u,"
					 "\"dense_highdim_widening_multiplier\":%.3f,"
					 "\"dense_widening_reason\":\"%s\","
						 "\"dense_budget_policy\":\"%s\","
						 "\"dense_rescore_band_policy\":\"%s\","
						 "\"bm25_candidates_requested\":%u,"
						 "\"bm25_candidates_effective\":%u,"
						 "\"bm25_k_defaulted\":%s,"
						 "\"bm25_candidates\":%u,"
						 "\"bm25_budget_reason\":\"%s\","
						 "\"bm25_dense_confidence\":%.6f,"
						 "\"bm25_hybrid_bound_mode\":\"%s\","
						 "\"bm25_hybrid_bound_stop_rank\":%u,"
						 "\"bm25_hybrid_bound_skipped_estimated\":%u,"
						 "\"bm25_hybrid_bound_threshold\":%.6f,"
						 "\"bm25_hybrid_bound_safe\":%s,"
						 "\"rrf_k_requested\":%u,"
						 "\"rrf_k_effective\":%u,"
						 "\"rrf_k_defaulted\":%s,"
						 "\"auto_budget_limit\":%u,"
						 "\"union_candidates\":%u,"
					 "\"final_results\":%u,"
					 "\"fusion_strategy\":\"%s\","
					 "\"fusion_candidates_seen\":%u,"
					 "\"fusion_heap_size\":%u,"
					 "\"both_match\":%u,"
					 "\"dense_only\":%u,"
					 "\"bm25_only\":%u,"
					 "\"graph_visited_nodes\":" UINT64_FORMAT ","
					 "\"graph_scored_codes\":" UINT64_FORMAT ","
					 "\"graph_exact_rescore_count\":" UINT64_FORMAT ","
					 "\"dense_prepare_us\":" UINT64_FORMAT ","
					 "\"dense_traverse_us\":" UINT64_FORMAT ","
					 "\"dense_entry_us\":" UINT64_FORMAT ","
					 "\"dense_base_us\":" UINT64_FORMAT ","
					 "\"dense_batch_us\":" UINT64_FORMAT ","
					 "\"dense_heap_us\":" UINT64_FORMAT ","
					 "\"dense_fill_us\":" UINT64_FORMAT ","
					 "\"dense_rescore_us\":" UINT64_FORMAT ","
					 "\"dense_sort_us\":" UINT64_FORMAT ","
					 "\"bm25_terms\":%u,"
					 "\"bm25_postings_decoded\":" UINT64_FORMAT ","
					 "\"bm25_blocks_visited\":" UINT64_FORMAT ","
					 "\"bm25_blocks_skipped\":" UINT64_FORMAT ","
					 "\"bm25_candidates_scored\":%u,"
					 "\"bm25_cache_bytes\":" UINT64_FORMAT ","
					 "\"bm25_cache_lexicon_entries\":%u,"
					 "\"bm25_cache_hit\":%s,"
					 "\"bm25_cache_build_us\":" UINT64_FORMAT ","
					 "\"bm25_cache_docstats_loaded\":%s,"
					 "\"bm25_cache_liveness_loaded\":%s,"
					 "\"bm25_hot_postings_cache_hit\":%s,"
					 "\"bm25_hot_postings_cache_hits\":" UINT64_FORMAT ","
					 "\"bm25_hot_postings_cache_misses\":" UINT64_FORMAT ","
					 "\"bm25_hot_postings_cache_bytes\":" UINT64_FORMAT ","
					 "\"bm25_hot_postings_cache_evictions\":" UINT64_FORMAT ","
					 "\"bm25_delta_lookup_mode\":\"%s\","
					 "\"bm25_delta_pages_scanned\":" UINT64_FORMAT ","
					 "\"bm25_delta_term_pages_read\":" UINT64_FORMAT ","
					 "\"bm25_delta_blocks_visited\":" UINT64_FORMAT ","
					 "\"bm25_delta_postings_decoded\":" UINT64_FORMAT ","
					 "\"bm25_delta_cache_bytes\":" UINT64_FORMAT ","
					 "\"bm25_delta_cache_terms\":%u,"
					 "\"bm25_delta_cache_hit\":%s,"
					 "\"bm25_wand_iterations\":" UINT64_FORMAT ","
					 "\"bm25_wand_threshold_updates\":" UINT64_FORMAT ","
					 "\"bm25_wand_active_sorts\":" UINT64_FORMAT ","
					 "\"bm25_wand_heap_updates\":" UINT64_FORMAT ","
					 "\"bm25_wand_full_reorders\":" UINT64_FORMAT ","
					 "\"bm25_wand_bound_type\":\"%s\","
					 "\"bm25_wand_bound_tightening_hits\":" UINT64_FORMAT ","
					 "\"bm25_wand_heap_replacements\":" UINT64_FORMAT ","
					 "\"bm25_strategy\":\"%s\","
					 "\"bm25_strategy_guc\":\"%s\","
					 "\"bm25_and_driver_df\":%u,"
					 "\"bm25_and_verified_candidates\":%u,"
					 "\"bm25_and_rejected_candidates\":%u,"
					 "\"bm25_impact_terms\":%u,"
					 "\"bm25_impact_tiers_read\":%u,"
					 "\"bm25_impact_postings_read\":" UINT64_FORMAT ","
					 "\"bm25_impact_remaining_upper_bound\":%.6f,"
					 "\"bm25_impact_early_stop\":%s,"
					 "\"bm25_impact_exact_safe\":%s,"
					 "\"bm25_impact_full_postings_avoided\":%s,"
					 "\"bm25_impact_loaded_from_storage\":%s,"
					 "\"bm25_impact_built_lazily\":%s,"
					 "\"bm25_impact_lazy_postings_scanned\":" UINT64_FORMAT ","
					 "\"bm25_accumulator_mode\":\"%s\","
					 "\"bm25_accumulator_hash_lookups\":" UINT64_FORMAT ","
					 "\"bm25_accumulator_dense_updates\":" UINT64_FORMAT ","
					 "\"bm25_final_heap_replacements\":" UINT64_FORMAT ","
					 "\"bm25_final_sorted_count\":%u,"
					 "\"bm25_full_sort_avoided\":%s,"
					 "\"bm25_query_shape\":\"%s\","
					 "\"bm25_boolean_eval_mode\":\"%s\","
					 "\"bm25_boolean_eval_calls\":" UINT64_FORMAT ","
					 "\"bm25_decode_kernel\":\"%s\","
					 "\"bm25_score_kernel\":\"%s\","
					 "\"bm25_simd_force\":\"%s\","
					 "\"bm25_simd_blocks\":" UINT64_FORMAT ","
					 "\"bm25_scalar_tail_postings\":" UINT64_FORMAT ","
					 "\"bm25_prefetches\":" UINT64_FORMAT ","
					 "\"dense_simd_force\":\"%s\","
					 "\"exact_simd_force\":\"%s\","
					 "\"dense_elapsed_us\":" UINT64_FORMAT ","
					 "\"bm25_elapsed_us\":" UINT64_FORMAT ","
					 "\"fusion_elapsed_us\":" UINT64_FORMAT ","
					 "\"elapsed_us\":" UINT64_FORMAT "}",
						 tqhybrid_last_scan_stats.fusion[0] != '\0' ?
						 tqhybrid_last_scan_stats.fusion : "none",
						 tqhybrid_last_scan_stats.denseCandidatesRequested,
						 tqhybrid_last_scan_stats.denseCandidatesEffective,
						 tqhybrid_last_scan_stats.denseKDefaulted ? "true" : "false",
						 tqhybrid_last_scan_stats.denseCandidates,
					 tqhybrid_last_scan_stats.denseEffectiveResultTarget,
					 tqhybrid_last_scan_stats.denseEffectiveSearchEf,
					 tqhybrid_last_scan_stats.denseEffectiveRescoreBand,
					 tqhybrid_last_scan_stats.denseHighdimWideningMultiplier,
					 TqGraphDenseWideningReasonName(tqhybrid_last_scan_stats.denseWideningReason),
						 TqGraphDenseBudgetPolicyNameExternal(tqhybrid_last_scan_stats.denseBudgetPolicy),
						 TqGraphRescoreBandPolicyNameExternal(tqhybrid_last_scan_stats.denseRescoreBandPolicy),
						 tqhybrid_last_scan_stats.bm25CandidatesRequested,
						 tqhybrid_last_scan_stats.bm25CandidatesEffective,
						 tqhybrid_last_scan_stats.bm25KDefaulted ? "true" : "false",
						 tqhybrid_last_scan_stats.bm25Candidates,
						 tqhybrid_last_scan_stats.bm25BudgetReason[0] != '\0' ?
						 tqhybrid_last_scan_stats.bm25BudgetReason : "unknown",
						 tqhybrid_last_scan_stats.bm25DenseConfidence,
						 TqHybridBm25HybridBoundModeName(tqhybrid_last_scan_stats.bm25HybridBoundMode),
						 tqhybrid_last_scan_stats.bm25HybridBoundStopRank,
						 tqhybrid_last_scan_stats.bm25HybridBoundSkippedEstimated,
						 tqhybrid_last_scan_stats.bm25HybridBoundThreshold,
						 tqhybrid_last_scan_stats.bm25HybridBoundSafe ? "true" : "false",
						 tqhybrid_last_scan_stats.rrfKRequested,
						 tqhybrid_last_scan_stats.rrfKEffective,
						 tqhybrid_last_scan_stats.rrfKDefaulted ? "true" : "false",
						 tqhybrid_last_scan_stats.autoBudgetLimit,
						 tqhybrid_last_scan_stats.unionCandidates,
					 tqhybrid_last_scan_stats.finalResults,
					 tqhybrid_last_scan_stats.fusionStrategy[0] != '\0' ?
					 tqhybrid_last_scan_stats.fusionStrategy : "none",
					 tqhybrid_last_scan_stats.fusionCandidatesSeen,
					 tqhybrid_last_scan_stats.fusionHeapSize,
					 tqhybrid_last_scan_stats.bothMatch,
					 tqhybrid_last_scan_stats.denseOnly,
					 tqhybrid_last_scan_stats.bm25Only,
					 tqhybrid_last_scan_stats.graphVisitedNodes,
					 tqhybrid_last_scan_stats.graphScoredCodes,
					 tqhybrid_last_scan_stats.graphExactRescoreCount,
					 tqhybrid_last_scan_stats.graphPrepareUs,
					 tqhybrid_last_scan_stats.graphTraverseUs,
					 tqhybrid_last_scan_stats.graphEntryUs,
					 tqhybrid_last_scan_stats.graphBaseUs,
					 tqhybrid_last_scan_stats.graphBatchUs,
					 tqhybrid_last_scan_stats.graphHeapUs,
					 tqhybrid_last_scan_stats.graphFillUs,
					 tqhybrid_last_scan_stats.graphRescoreUs,
					 tqhybrid_last_scan_stats.graphSortUs,
					 tqhybrid_last_scan_stats.bm25Terms,
					 tqhybrid_last_scan_stats.bm25PostingsDecoded,
					 tqhybrid_last_scan_stats.bm25BlocksVisited,
					 tqhybrid_last_scan_stats.bm25BlocksSkipped,
					 tqhybrid_last_scan_stats.bm25CandidatesScored,
					 tqhybrid_last_scan_stats.bm25CacheBytes,
					 tqhybrid_last_scan_stats.bm25CacheLexiconEntries,
					 tqhybrid_last_scan_stats.bm25CacheHit ? "true" : "false",
					 tqhybrid_last_scan_stats.bm25CacheBuildUs,
					 tqhybrid_last_scan_stats.bm25CacheDocstatsLoaded ? "true" : "false",
					 tqhybrid_last_scan_stats.bm25CacheLivenessLoaded ? "true" : "false",
					 tqhybrid_last_scan_stats.bm25HotPostingsCacheHits > 0 ?
					 "true" : "false",
					 tqhybrid_last_scan_stats.bm25HotPostingsCacheHits,
					 tqhybrid_last_scan_stats.bm25HotPostingsCacheMisses,
					 tqhybrid_last_scan_stats.bm25HotPostingsCacheBytes,
					 tqhybrid_last_scan_stats.bm25HotPostingsCacheEvictions,
					 TqHybridBm25DeltaLookupModeName(tqhybrid_last_scan_stats.bm25DeltaLookupMode),
					 tqhybrid_last_scan_stats.bm25DeltaPagesScanned,
					 tqhybrid_last_scan_stats.bm25DeltaTermPagesRead,
					 tqhybrid_last_scan_stats.bm25DeltaBlocksVisited,
					 tqhybrid_last_scan_stats.bm25DeltaPostingsDecoded,
					 tqhybrid_last_scan_stats.bm25DeltaCacheBytes,
					 tqhybrid_last_scan_stats.bm25DeltaCacheTerms,
					 tqhybrid_last_scan_stats.bm25DeltaCacheHit ? "true" : "false",
					 tqhybrid_last_scan_stats.bm25WandIterations,
					 tqhybrid_last_scan_stats.bm25WandThresholdUpdates,
					 tqhybrid_last_scan_stats.bm25WandActiveSorts,
					 tqhybrid_last_scan_stats.bm25WandHeapUpdates,
					 tqhybrid_last_scan_stats.bm25WandFullReorders,
					 TqHybridBm25WandBoundTypeName(tqhybrid_last_scan_stats.bm25WandBoundType),
					 tqhybrid_last_scan_stats.bm25WandBoundTighteningHits,
					 tqhybrid_last_scan_stats.bm25WandHeapReplacements,
					 TqHybridBm25RuntimeStrategyName(tqhybrid_last_scan_stats.bm25Strategy),
					 TqHybridBm25StrategyName(tqhybrid_bm25_strategy),
					 tqhybrid_last_scan_stats.bm25AndDriverDf,
					 tqhybrid_last_scan_stats.bm25AndVerifiedCandidates,
					 tqhybrid_last_scan_stats.bm25AndRejectedCandidates,
					 tqhybrid_last_scan_stats.bm25ImpactTerms,
					 tqhybrid_last_scan_stats.bm25ImpactTiersRead,
					 tqhybrid_last_scan_stats.bm25ImpactPostingsRead,
					 tqhybrid_last_scan_stats.bm25ImpactRemainingUpperBound,
					 tqhybrid_last_scan_stats.bm25ImpactEarlyStop ? "true" : "false",
					 tqhybrid_last_scan_stats.bm25ImpactExactSafe ? "true" : "false",
					 tqhybrid_last_scan_stats.bm25ImpactFullPostingsAvoided ? "true" : "false",
					 tqhybrid_last_scan_stats.bm25ImpactLoadedFromStorage ? "true" : "false",
					 tqhybrid_last_scan_stats.bm25ImpactBuiltLazily ? "true" : "false",
					 tqhybrid_last_scan_stats.bm25ImpactLazyPostingsScanned,
					 TqHybridBm25AccumulatorModeName(tqhybrid_last_scan_stats.bm25AccumulatorMode),
					 tqhybrid_last_scan_stats.bm25AccumulatorHashLookups,
					 tqhybrid_last_scan_stats.bm25AccumulatorDenseUpdates,
					 tqhybrid_last_scan_stats.bm25FinalHeapReplacements,
					 tqhybrid_last_scan_stats.bm25FinalSortedCount,
					 tqhybrid_last_scan_stats.bm25FullSortAvoided ? "true" : "false",
					 TqHybridBm25QueryShapeName(tqhybrid_last_scan_stats.bm25QueryShape),
					 TqHybridBm25BooleanEvalModeName(tqhybrid_last_scan_stats.bm25BooleanEvalMode),
					 tqhybrid_last_scan_stats.bm25BooleanEvalCalls,
					 TqHybridBm25KernelName(tqhybrid_last_scan_stats.bm25DecodeKernel),
					 TqHybridBm25KernelName(tqhybrid_last_scan_stats.bm25ScoreKernel),
					 TqHybridBm25SimdForceName(tqhybrid_bm25_simd_force),
					 tqhybrid_last_scan_stats.bm25SimdBlocks,
					 tqhybrid_last_scan_stats.bm25ScalarTailPostings,
					 tqhybrid_last_scan_stats.bm25Prefetches,
					 HnswTqSimdForceName(hnsw_tq_simd_force),
					 HnswTqExactSimdForceName(hnsw_tq_exact_simd_force),
					 tqhybrid_last_scan_stats.denseElapsedUs,
					 tqhybrid_last_scan_stats.bm25ElapsedUs,
					 tqhybrid_last_scan_stats.fusionElapsedUs,
					 tqhybrid_last_scan_stats.elapsedUs);

	PG_RETURN_DATUM(DirectFunctionCall1(jsonb_in, CStringGetDatum(json.data)));
}

Datum
turbohybridhandler(PG_FUNCTION_ARGS)
{
	IndexAmRoutine *amroutine = makeNode(IndexAmRoutine);

	amroutine->amstrategies = 0;
	amroutine->amsupport = 5;
	amroutine->amoptsprocnum = 0;
	amroutine->amcanorder = false;
	amroutine->amcanorderbyop = true;
#if PG_VERSION_NUM >= 180000
	amroutine->amcanhash = false;
	amroutine->amconsistentequality = false;
	amroutine->amconsistentordering = false;
#endif
	amroutine->amcanbackward = false;
	amroutine->amcanunique = false;
	amroutine->amcanmulticol = true;
	amroutine->amoptionalkey = true;
	amroutine->amsearcharray = false;
	amroutine->amsearchnulls = false;
	amroutine->amstorage = false;
	amroutine->amclusterable = false;
	amroutine->ampredlocks = false;
	amroutine->amcanparallel = false;
#if PG_VERSION_NUM >= 170000
	amroutine->amcanbuildparallel = true;
#endif
	amroutine->amcaninclude = true;
	amroutine->amusemaintenanceworkmem = false;
#if PG_VERSION_NUM >= 160000
	amroutine->amsummarizing = false;
#endif
	amroutine->amparallelvacuumoptions = VACUUM_OPTION_PARALLEL_BULKDEL;
	amroutine->amkeytype = InvalidOid;

	amroutine->ambuild = tqhybridbuild;
	amroutine->ambuildempty = tqhybridbuildempty;
	amroutine->aminsert = tqhybridinsert;
#if PG_VERSION_NUM >= 170000
	amroutine->aminsertcleanup = NULL;
#endif
	amroutine->ambulkdelete = tqhybridbulkdelete;
	amroutine->amvacuumcleanup = tqhybridvacuumcleanup;
	amroutine->amcanreturn = NULL;
	amroutine->amcostestimate = tqhybridcostestimate;
#if PG_VERSION_NUM >= 180000
	amroutine->amgettreeheight = NULL;
#endif
	amroutine->amoptions = tqhybridoptions;
	amroutine->amproperty = NULL;
	amroutine->ambuildphasename = NULL;
	amroutine->amvalidate = tqhybridvalidate;
#if PG_VERSION_NUM >= 140000
	amroutine->amadjustmembers = NULL;
#endif
	amroutine->ambeginscan = tqhybridbeginscan;
	amroutine->amrescan = tqhybridrescan;
	amroutine->amgettuple = tqhybridgettuple;
	amroutine->amgetbitmap = NULL;
	amroutine->amendscan = tqhybridendscan;
	amroutine->ammarkpos = NULL;
	amroutine->amrestrpos = NULL;
	amroutine->amestimateparallelscan = NULL;
	amroutine->aminitparallelscan = NULL;
	amroutine->amparallelrescan = NULL;
#if PG_VERSION_NUM >= 180000
	amroutine->amtranslatestrategy = NULL;
	amroutine->amtranslatecmptype = NULL;
#endif

	PG_RETURN_POINTER(amroutine);
}
