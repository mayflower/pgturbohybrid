#include "postgres.h"

#include <limits.h>
#include <math.h>

#include "access/amapi.h"
#include "access/relscan.h"
#include "access/reloptions.h"
#include "access/sysattr.h"
#include "access/xact.h"
#include "catalog/namespace.h"
#include "catalog/pg_extension.h"
#include "catalog/pg_opclass.h"
#include "catalog/pg_type.h"
#include "catalog/pg_type_d.h"
#include "commands/extension.h"
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
#include "utils/syscache.h"

#include "pgturbohybrid.h"
#include "pgturbohybrid_query.h"
#include "pgturbohybrid_quant.h"
#include "pgturbohybrid_am.h"
#include "pgturbohybrid_bm25.h"

#define PGTURBOHYBRID_DEFAULT_BM25_K1 1.2
#define PGTURBOHYBRID_DEFAULT_BM25_B 0.75
#define PGTURBOHYBRID_DEFAULT_DENSE_K 400
#define PGTURBOHYBRID_DEFAULT_BM25_K 400
#define PGTURBOHYBRID_DEFAULT_RRF_K 60

#if PG_VERSION_NUM < 150000
#define MarkGUCPrefixReserved(x) EmitWarningsOnPlaceholders(x)
#endif

static relopt_kind pgturbohybrid_relopt_kind;
static ExecutorStart_hook_type prev_tqhybrid_ExecutorStart_hook = NULL;
static ExecutorEnd_hook_type prev_tqhybrid_ExecutorEnd_hook = NULL;
static List *pgturbohybrid_plannedstmt_stack = NIL;
static PlannedStmt *pgturbohybrid_current_plannedstmt = NULL;

bool		pgturbohybrid_enable_wand = true;
int			pgturbohybrid_max_union_candidates = 100000;
int			pgturbohybrid_default_dense_k = PGTURBOHYBRID_DEFAULT_DENSE_K;
int			pgturbohybrid_default_bm25_k = PGTURBOHYBRID_DEFAULT_BM25_K;
int			pgturbohybrid_default_rrf_k = PGTURBOHYBRID_DEFAULT_RRF_K;
static bool pgturbohybrid_simd = true;
int			pgturbohybrid_force_fusion = 0;
int			pgturbohybrid_fusion_hash_threshold = 128;
bool		pgturbohybrid_enable_exact_rescore_for_bm25_only = false;
int			pgturbohybrid_bm25_cache_max_mb = 0;
int			pgturbohybrid_bm25_hot_postings_cache_mb = 0;
int			pgturbohybrid_bm25_hot_postings_cache_min_df = 1024;
bool		pgturbohybrid_bm25_allow_lazy_impact_build = false;
int			pgturbohybrid_bm25_simd_force = PGTURBOHYBRID_BM25_SIMD_FORCE_AUTO;
bool		pgturbohybrid_bm25_force_full_sort = false;
int			pgturbohybrid_bm25_accumulator_mode = PGTURBOHYBRID_BM25_ACCUMULATOR_AUTO;
int			pgturbohybrid_bm25_dense_accumulator_threshold = 4096;
double		pgturbohybrid_bm25_dense_accumulator_df_ratio = 0.05;
int			pgturbohybrid_bm25_strategy = PGTURBOHYBRID_BM25_STRATEGY_AUTO;
int			pgturbohybrid_bm25_impact_or_mode = PGTURBOHYBRID_BM25_IMPACT_OR_MODE_EXACT_ONLY;
bool		pgturbohybrid_auto_budget = true;
int			pgturbohybrid_auto_budget_min_dense_k = 32;
int			pgturbohybrid_auto_budget_min_bm25_k = 32;
int			pgturbohybrid_auto_budget_limit_multiplier = 8;
int			pgturbohybrid_auto_budget_quality_cap = 400;
bool		pgturbohybrid_auto_bm25_budget = true;
int			pgturbohybrid_auto_bm25_budget_min = 32;
int			pgturbohybrid_auto_bm25_budget_max = 400;
bool		pgturbohybrid_auto_bm25_budget_dense_confidence = true;
static int	pgturbohybrid_bm25_hybrid_bound = 0;

typedef enum PgturbohybridBm25HybridBoundMode
{
	PGTURBOHYBRID_BM25_HYBRID_BOUND_OFF,
	PGTURBOHYBRID_BM25_HYBRID_BOUND_SAFE,
	PGTURBOHYBRID_BM25_HYBRID_BOUND_APPROX
} PgturbohybridBm25HybridBoundMode;

static relopt_enum_elt_def pgturbohybrid_routing_relopt_options[] = {
	{"auto", PGTURBOHYBRID_ROUTING_AUTO},
	{"graph", PGTURBOHYBRID_ROUTING_GRAPH},
	{"flat", PGTURBOHYBRID_ROUTING_FLAT},
	{NULL, 0}
};

const char *
PgturbohybridBm25SimdForceName(int force)
{
	switch ((PgturbohybridBm25SimdForce) force)
	{
		case PGTURBOHYBRID_BM25_SIMD_FORCE_AUTO:
			return "auto";
		case PGTURBOHYBRID_BM25_SIMD_FORCE_SCALAR:
			return "scalar";
		case PGTURBOHYBRID_BM25_SIMD_FORCE_AVX2:
			return "avx2";
		case PGTURBOHYBRID_BM25_SIMD_FORCE_NEON:
			return "neon";
		default:
			return "unknown";
	}
}

const char *
PgturbohybridBm25AccumulatorModeName(int mode)
{
	switch ((PgturbohybridBm25AccumulatorMode) mode)
	{
		case PGTURBOHYBRID_BM25_ACCUMULATOR_HASH:
			return "hash";
		case PGTURBOHYBRID_BM25_ACCUMULATOR_DENSE:
			return "node_generation_arrays";
		case PGTURBOHYBRID_BM25_ACCUMULATOR_AUTO:
			return "auto";
		default:
			return "unknown";
	}
}

const char *
PgturbohybridBm25StrategyName(int strategy)
{
	switch ((PgturbohybridBm25Strategy) strategy)
	{
		case PGTURBOHYBRID_BM25_STRATEGY_AUTO:
			return "auto";
		case PGTURBOHYBRID_BM25_STRATEGY_IMPACT:
			return "impact";
		case PGTURBOHYBRID_BM25_STRATEGY_IMPACT_OR:
			return "impact_or";
		case PGTURBOHYBRID_BM25_STRATEGY_WAND:
			return "wand";
		case PGTURBOHYBRID_BM25_STRATEGY_DAAT_SIMD:
			return "daat_simd";
		case PGTURBOHYBRID_BM25_STRATEGY_DAAT_HASH:
			return "daat_hash";
		default:
			return "unknown";
	}
}

const char *
PgturbohybridBm25RuntimeStrategyName(int strategy)
{
	switch ((PgturbohybridBm25RuntimeStrategy) strategy)
	{
		case PGTURBOHYBRID_BM25_RUNTIME_IMPACT_SINGLE:
			return "impact_single";
		case PGTURBOHYBRID_BM25_RUNTIME_IMPACT_OR:
			return "impact_or";
		case PGTURBOHYBRID_BM25_RUNTIME_IMPACT_SEEDED_WAND:
			return "impact_seeded_wand";
		case PGTURBOHYBRID_BM25_RUNTIME_WAND:
			return "wand";
		case PGTURBOHYBRID_BM25_RUNTIME_AND_RAREST_DRIVER:
			return "and_rarest_driver";
		case PGTURBOHYBRID_BM25_RUNTIME_DAAT_SIMD:
			return "daat_simd";
		case PGTURBOHYBRID_BM25_RUNTIME_DAAT_HASH:
			return "daat_hash";
		case PGTURBOHYBRID_BM25_RUNTIME_NONE:
		default:
			return "none";
	}
}

static const char *
PgturbohybridBm25HybridBoundModeName(int mode)
{
	switch ((PgturbohybridBm25HybridBoundMode) mode)
	{
		case PGTURBOHYBRID_BM25_HYBRID_BOUND_OFF:
			return "off";
		case PGTURBOHYBRID_BM25_HYBRID_BOUND_SAFE:
			return "safe";
		case PGTURBOHYBRID_BM25_HYBRID_BOUND_APPROX:
			return "approx";
		default:
			return "unknown";
	}
}

static void
PgturbohybridAssignSimd(bool newval, void *extra)
{
	pgturbohybrid_dense_simd_force = newval ?
		PGTURBOHYBRID_SIMD_FORCE_AUTO : PGTURBOHYBRID_SIMD_FORCE_SCALAR;
	pgturbohybrid_dense_exact_simd_force = newval ?
		PGTURBOHYBRID_EXACT_SIMD_FORCE_AUTO : PGTURBOHYBRID_EXACT_SIMD_FORCE_SCALAR;
	pgturbohybrid_bm25_simd_force = newval ?
		PGTURBOHYBRID_BM25_SIMD_FORCE_AUTO : PGTURBOHYBRID_BM25_SIMD_FORCE_SCALAR;
}

static void PgturbohybridExecutorStartHook(QueryDesc *queryDesc, int eflags);
static void PgturbohybridExecutorEndHook(QueryDesc *queryDesc);
static void PgturbohybridXactCallback(XactEvent event, void *arg);
static void PgturbohybridSubXactCallback(SubXactEvent event, SubTransactionId mySubid,
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
static void PgturbohybridEnsureOrderByStorage(IndexScanDesc scan, MemoryContext scanCtx);
static bool PgturbohybridPathHasFilter(IndexPath *path);
static bool PgturbohybridFindConstQueryWalker(Node *node, void *context);
static PgturbohybridQueryHeader *PgturbohybridFindConstQuery(List *indexorderbys);
static int PgturbohybridEstimateTsQueryTerms(TSQuery query);
static int	PgturbohybridCurrentLimit(void);

typedef struct PgturbohybridResult
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
} PgturbohybridResult;

typedef struct PgturbohybridMergeSlot
{
	bool		used;
	PgturbohybridResult result;
} PgturbohybridMergeSlot;

typedef struct PgturbohybridScanState
{
	PgturbohybridQueryHeader *query;
	PgturbohybridResult *results;
	int			resultCount;
	int			resultIndex;
	bool		collectDone;
	bool		active;
} PgturbohybridScanState;

typedef struct PgturbohybridLastScanStats
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
} PgturbohybridLastScanStats;

static PgturbohybridLastScanStats pgturbohybrid_last_scan_state;

static uint64
PgturbohybridElapsedUs(instr_time start)
{
	instr_time	duration;

	INSTR_TIME_SET_CURRENT(duration);
	INSTR_TIME_SUBTRACT(duration, start);
	return (uint64) INSTR_TIME_GET_MICROSEC(duration);
}

static Datum PgturbohybridHybridLastScanStats(PG_FUNCTION_ARGS) pg_attribute_unused();

PlannedStmt *
PgturbohybridCurrentPlannedStmt(void)
{
	return pgturbohybrid_current_plannedstmt;
}

static void
PgturbohybridPushPlannedStmt(PlannedStmt *plannedstmt)
{
	MemoryContext oldCtx = MemoryContextSwitchTo(TopMemoryContext);

	pgturbohybrid_plannedstmt_stack =
		lcons(pgturbohybrid_current_plannedstmt, pgturbohybrid_plannedstmt_stack);
	pgturbohybrid_current_plannedstmt = plannedstmt;

	MemoryContextSwitchTo(oldCtx);
}

static void
PgturbohybridPopPlannedStmt(void)
{
	if (pgturbohybrid_plannedstmt_stack == NIL)
	{
		pgturbohybrid_current_plannedstmt = NULL;
		return;
	}

	pgturbohybrid_current_plannedstmt =
		(PlannedStmt *) linitial(pgturbohybrid_plannedstmt_stack);
	pgturbohybrid_plannedstmt_stack = list_delete_first(pgturbohybrid_plannedstmt_stack);
}

static void
PgturbohybridClearPlannedStmtStack(void)
{
	list_free(pgturbohybrid_plannedstmt_stack);
	pgturbohybrid_plannedstmt_stack = NIL;
	pgturbohybrid_current_plannedstmt = NULL;
}

static void
PgturbohybridExecutorStartHook(QueryDesc *queryDesc, int eflags)
{
	if (prev_tqhybrid_ExecutorStart_hook)
		prev_tqhybrid_ExecutorStart_hook(queryDesc, eflags);
	else
		standard_ExecutorStart(queryDesc, eflags);

	PgturbohybridPushPlannedStmt(queryDesc->plannedstmt);
}

static void
PgturbohybridExecutorEndHook(QueryDesc *queryDesc)
{
	PgturbohybridPopPlannedStmt();

	if (prev_tqhybrid_ExecutorEnd_hook)
		prev_tqhybrid_ExecutorEnd_hook(queryDesc);
	else
		standard_ExecutorEnd(queryDesc);
}

static void
PgturbohybridXactCallback(XactEvent event, void *arg)
{
	if (event == XACT_EVENT_ABORT || event == XACT_EVENT_PARALLEL_ABORT)
		PgturbohybridClearPlannedStmtStack();
}

static void
PgturbohybridSubXactCallback(SubXactEvent event, SubTransactionId mySubid,
						SubTransactionId parentSubid, void *arg)
{
	if (event == SUBXACT_EVENT_ABORT_SUB)
		PgturbohybridClearPlannedStmtStack();
}

static void
PgturbohybridValidateIndex(Relation index, IndexInfo *indexInfo)
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
					 errmsg("pgturbohybrid indexes require exactly two key columns"),
					 errdetail("Use one vector column followed by one tsvector column.")));

		if (indexInfo->ii_Expressions != NIL)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("pgturbohybrid expression indexes are not supported yet")));
	}
	else if (index->rd_index != NULL && index->rd_index->indnkeyatts != 2)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("pgturbohybrid indexes require exactly two key columns"),
				 errdetail("Use one vector column followed by one tsvector column.")));

	if (desc->natts < 2)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("pgturbohybrid indexes require exactly two key columns")));

	denseType = TupleDescAttr(desc, 0)->atttypid;
	lexicalType = TupleDescAttr(desc, 1)->atttypid;
	vectorOid = PgturbohybridVectorTypeOid();

	if (!OidIsValid(vectorOid) || denseType != vectorOid)
		ereport(ERROR,
				(errcode(ERRCODE_DATATYPE_MISMATCH),
				 errmsg("pgturbohybrid first key must be type vector"),
				 errdetail("Found %s.", format_type_be(denseType))));

	if (lexicalType != TSVECTOROID)
		ereport(ERROR,
				(errcode(ERRCODE_DATATYPE_MISMATCH),
				 errmsg("pgturbohybrid second key must be type tsvector"),
				 errdetail("Found %s.", format_type_be(lexicalType))));
}

static ScanKey
PgturbohybridDenseOrderBys(ScanKey orderbys, int norderbys)
{
	ScanKey		denseOrderbys;

	if (orderbys == NULL || norderbys <= 0)
		return orderbys;

	denseOrderbys = palloc(sizeof(ScanKeyData) * norderbys);
	memcpy(denseOrderbys, orderbys, sizeof(ScanKeyData) * norderbys);

	for (int i = 0; i < norderbys; i++)
	{
		PgturbohybridQueryHeader *query;
		Vector	   *vectorQuery;

		if (denseOrderbys[i].sk_flags & SK_ISNULL)
			continue;

		query = DatumGetHybridQuery(denseOrderbys[i].sk_argument);
		PgturbohybridQueryValidate(query);

		vectorQuery = PgturbohybridQueryGetVector(query);
		if (vectorQuery == NULL)
			continue;

		denseOrderbys[i].sk_argument = PointerGetDatum(vectorQuery);
	}

	return denseOrderbys;
}

static void
PgturbohybridEnsureOrderByStorage(IndexScanDesc scan, MemoryContext scanCtx)
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
PgturbohybridNodeCompare(const void *a, const void *b)
{
	const PgturbohybridResult *ra = (const PgturbohybridResult *) a;
	const PgturbohybridResult *rb = (const PgturbohybridResult *) b;

	return (ra->nodeId > rb->nodeId) - (ra->nodeId < rb->nodeId);
}

static int
PgturbohybridScoreCompare(const void *a, const void *b)
{
	const PgturbohybridResult *ra = (const PgturbohybridResult *) a;
	const PgturbohybridResult *rb = (const PgturbohybridResult *) b;
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
PgturbohybridScoreWorse(const PgturbohybridResult *left, const PgturbohybridResult *right)
{
	return PgturbohybridScoreCompare(left, right) > 0;
}

static uint32
PgturbohybridNextPowerOfTwo(uint32 value)
{
	uint32		result = 1;

	while (result < value)
		result <<= 1;
	return result;
}

static uint32
PgturbohybridFusionHashSlotCount(uint32 itemCount)
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
	return PgturbohybridNextPowerOfTwo(Max(target, 16));
}

static uint32
PgturbohybridHashNodeId(uint32 nodeId)
{
	return nodeId * UINT32_C(2654435761);
}

static PgturbohybridResult *
PgturbohybridFindMergeSlot(PgturbohybridMergeSlot *slots, uint32 mask, uint32 nodeId)
{
	uint32		slotNo = PgturbohybridHashNodeId(nodeId) & mask;

	for (;;)
	{
		PgturbohybridMergeSlot *slot = &slots[slotNo];

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
PgturbohybridFinalTarget(PgturbohybridQueryHeader *query, int mergedCount)
{
	int			finalCount = mergedCount;

	if (pgturbohybrid_max_union_candidates > 0)
		finalCount = Min(finalCount, pgturbohybrid_max_union_candidates);
	if ((query->flags & HYBRID_QUERY_FLAG_FINAL_K_IS_SET) != 0)
		finalCount = Min(finalCount, query->finalK);
	return finalCount;
}

static int
PgturbohybridConstLimitValue(Node *limitCount)
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
PgturbohybridFindLimitInPlan(Plan *plan)
{
	int			limit;

	if (plan == NULL)
		return 0;

	if (IsA(plan, Limit))
	{
		Limit	   *limitPlan = castNode(Limit, plan);

		limit = PgturbohybridConstLimitValue(limitPlan->limitCount);
		if (limit > 0)
			return limit;
	}

	limit = PgturbohybridFindLimitInPlan(plan->lefttree);
	if (limit > 0)
		return limit;

	return PgturbohybridFindLimitInPlan(plan->righttree);
}

static int
PgturbohybridCurrentLimit(void)
{
	PlannedStmt *plannedstmt = PgturbohybridCurrentPlannedStmt();

	if (plannedstmt == NULL || plannedstmt->planTree == NULL)
		return 0;

	return PgturbohybridFindLimitInPlan(plannedstmt->planTree);
}

static int
PgturbohybridApplyAutoBudget(int requested, int limit, int minBudget,
						bool defaulted, bool branchPresent)
{
	int			target;

	if (!pgturbohybrid_auto_budget || !defaulted || !branchPresent ||
		requested <= 0 || limit <= 0)
		return requested;

	if (limit > INT_MAX / pgturbohybrid_auto_budget_limit_multiplier)
		target = INT_MAX;
	else
		target = limit * pgturbohybrid_auto_budget_limit_multiplier;

	target = Max(target, minBudget);
	if (pgturbohybrid_auto_budget_quality_cap > 0)
		target = Min(target, pgturbohybrid_auto_budget_quality_cap);

	return Min(requested, target);
}

static PgturbohybridQueryHeader *
PgturbohybridEffectiveQuery(PgturbohybridQueryHeader *query, int limit,
					   MemoryContext memoryContext)
{
	PgturbohybridQueryHeader *effective;
	Size		querySize = VARSIZE_ANY(query);
	bool		hasVector =
		(query->flags & HYBRID_QUERY_FLAG_HAS_VECTOR) != 0;
	bool		hasTsQuery =
		(query->flags & HYBRID_QUERY_FLAG_HAS_TSQUERY) != 0;

	effective = MemoryContextAlloc(memoryContext, querySize);
	memcpy(effective, query, querySize);

	effective->denseK = PgturbohybridApplyAutoBudget(query->denseK, limit,
												pgturbohybrid_auto_budget_min_dense_k,
												(query->flags & HYBRID_QUERY_FLAG_DENSE_K_DEFAULTED) != 0,
												hasVector);
	effective->bm25K = PgturbohybridApplyAutoBudget(query->bm25K, limit,
											   pgturbohybrid_auto_budget_min_bm25_k,
											   (query->flags & HYBRID_QUERY_FLAG_BM25_K_DEFAULTED) != 0,
											   hasTsQuery);

	return effective;
}

static int
PgturbohybridBudgetFinalTarget(PgturbohybridQueryHeader *query, int limit)
{
	if ((query->flags & HYBRID_QUERY_FLAG_FINAL_K_IS_SET) != 0 &&
		query->finalK > 0)
		return query->finalK;
	if (limit > 0)
		return limit;
	return Max(Min(query->denseK, query->bm25K), 1);
}

static double
PgturbohybridDenseConfidence(const TqDenseCandidate *dense, int denseCount,
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
PgturbohybridMaybeApplyDenseBm25Budget(PgturbohybridQueryHeader *query,
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

	if (!pgturbohybrid_auto_budget)
	{
		if (reason != NULL && reasonSize > 0)
			strlcpy(reason, "auto_budget_off", reasonSize);
		return;
	}
	if (!pgturbohybrid_auto_bm25_budget)
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
	if (!pgturbohybrid_auto_bm25_budget_dense_confidence)
	{
		if (reason != NULL && reasonSize > 0)
			strlcpy(reason, "dense_confidence_off", reasonSize);
		return;
	}
	if (query->bm25K <= pgturbohybrid_auto_bm25_budget_min)
		return;

	finalTarget = PgturbohybridBudgetFinalTarget(query, limit);
	if (denseCount < finalTarget)
	{
		if (reason != NULL && reasonSize > 0)
			strlcpy(reason, "dense_insufficient", reasonSize);
		return;
	}

	if (denseConfidence != NULL)
		*denseConfidence =
			PgturbohybridDenseConfidence(dense, denseCount, finalTarget);
	if (denseConfidence == NULL || *denseConfidence <= 0.0)
	{
		if (reason != NULL && reasonSize > 0)
			strlcpy(reason, "dense_flat", reasonSize);
		return;
	}

	target = Max(finalTarget * 4, pgturbohybrid_auto_bm25_budget_min);
	if (pgturbohybrid_auto_bm25_budget_max > 0)
		target = Min(target, pgturbohybrid_auto_bm25_budget_max);
	target = Max(target, 1);

	if (target < query->bm25K)
	{
		query->bm25K = target;
		if (reason != NULL && reasonSize > 0)
			strlcpy(reason, "dense_confident_common_terms", reasonSize);
	}
}

static void
PgturbohybridMaybeApplyBm25HybridBound(PgturbohybridQueryHeader *query, int denseCount,
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
		*safe = pgturbohybrid_bm25_hybrid_bound !=
			PGTURBOHYBRID_BM25_HYBRID_BOUND_APPROX;

	if (pgturbohybrid_bm25_hybrid_bound == PGTURBOHYBRID_BM25_HYBRID_BOUND_OFF ||
		!hasVector || !hasTsQuery ||
		query->fusion != HYBRID_FUSION_RRF ||
		query->denseWeight <= 0.0 || query->bm25Weight <= 0.0 ||
		query->bm25K <= 0 || denseCount <= 0)
		return;

	finalTarget = PgturbohybridBudgetFinalTarget(query, limit);
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
	if (pgturbohybrid_bm25_hybrid_bound == PGTURBOHYBRID_BM25_HYBRID_BOUND_SAFE)
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

	if (pgturbohybrid_bm25_hybrid_bound == PGTURBOHYBRID_BM25_HYBRID_BOUND_SAFE &&
		denseCount != finalTarget)
		return;
	if (pgturbohybrid_bm25_hybrid_bound == PGTURBOHYBRID_BM25_HYBRID_BOUND_APPROX &&
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
PgturbohybridTopNHeapSiftUp(PgturbohybridResult *heap, int index)
{
	while (index > 0)
	{
		int			parent = (index - 1) / 2;
		PgturbohybridResult tmp;

		if (!PgturbohybridScoreWorse(&heap[index], &heap[parent]))
			break;
		tmp = heap[parent];
		heap[parent] = heap[index];
		heap[index] = tmp;
		index = parent;
	}
}

static void
PgturbohybridTopNHeapSiftDown(PgturbohybridResult *heap, int count, int index)
{
	for (;;)
	{
		int			left = index * 2 + 1;
		int			right = left + 1;
		int			worst = index;
		PgturbohybridResult tmp;

		if (left < count && PgturbohybridScoreWorse(&heap[left], &heap[worst]))
			worst = left;
		if (right < count && PgturbohybridScoreWorse(&heap[right], &heap[worst]))
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
PgturbohybridSelectTopN(PgturbohybridResult *items, int itemCount, int target,
				   PgturbohybridResult **topItems, MemoryContext memoryContext)
{
	PgturbohybridResult *heap;
	int			heapCount = 0;

	if (target <= 0 || itemCount <= 0)
	{
		*topItems = palloc0(sizeof(PgturbohybridResult));
		return 0;
	}

	heap = MemoryContextAllocZero(memoryContext,
								  sizeof(PgturbohybridResult) * target);
	for (int i = 0; i < itemCount; i++)
	{
		if (heapCount < target)
		{
			heap[heapCount] = items[i];
			PgturbohybridTopNHeapSiftUp(heap, heapCount);
			heapCount++;
		}
		else if (PgturbohybridScoreCompare(&items[i], &heap[0]) < 0)
		{
			heap[0] = items[i];
			PgturbohybridTopNHeapSiftDown(heap, heapCount, 0);
		}
	}

	if (heapCount > 1)
		qsort(heap, heapCount, sizeof(PgturbohybridResult), PgturbohybridScoreCompare);
	*topItems = heap;
	return heapCount;
}

static void
PgturbohybridAddDenseCandidate(PgturbohybridResult *item, TqDenseCandidate *candidate)
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
PgturbohybridAddBm25Candidate(PgturbohybridResult *item, PgturbohybridBm25Result *candidate)
{
	item->nodeId = candidate->nodeId;
	item->heaptid = candidate->heaptid;
	item->bm25Score = candidate->bm25Score;
	item->bm25Rank = candidate->rank;
	item->hasBm25 = true;
}

static double
PgturbohybridNormalize(double value, double minValue, double maxValue)
{
	if (maxValue <= minValue)
		return value > 0 ? 1.0 : 0.0;
	return (value - minValue) / (maxValue - minValue);
}

static void
PgturbohybridCheckBm25OnlyExactRescore(PgturbohybridScanState *state,
								  PgturbohybridResult *results, int count)
{
	if (!pgturbohybrid_enable_exact_rescore_for_bm25_only)
		return;
	if ((state->query->flags & HYBRID_QUERY_FLAG_HAS_VECTOR) == 0)
		return;

	for (int i = 0; i < count; i++)
	{
		if (results[i].hasBm25 && !results[i].hasDense)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("exact rescoring for BM25-only pgturbohybrid candidates is not implemented yet"),
					 errdetail("Candidate node %u matched BM25 but was outside the dense candidate set.",
							   results[i].nodeId),
					 errhint("Increase dense_k so candidates are scored by the dense branch.")));
	}
}

static void
PgturbohybridScoreResults(PgturbohybridScanState *state, PgturbohybridResult *results, int count)
{
	PgturbohybridQueryHeader *query = state->query;
	double		minDense = get_float8_infinity();
	double		maxDense = -get_float8_infinity();
	double		minBm25 = get_float8_infinity();
	double		maxBm25 = -get_float8_infinity();
	double		alpha = (query->flags & HYBRID_QUERY_FLAG_ALPHA_IS_SET) != 0 ?
		query->alpha : 0.5;
	uint16		fusion = pgturbohybrid_force_fusion != 0 ?
		pgturbohybrid_force_fusion : query->fusion;

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
				PgturbohybridNormalize(results[i].denseSimilarity, minDense, maxDense) : 0.0;
			double		bm25Norm = results[i].hasBm25 ?
				PgturbohybridNormalize(results[i].bm25Score, minBm25, maxBm25) : 0.0;

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
PgturbohybridCollectScanResults(IndexScanDesc scan, PgturbohybridScanState *state)
{
	PgturbohybridGraphScanOpaque so = (PgturbohybridGraphScanOpaque) scan->opaque;
	TqDenseCandidate *dense = NULL;
	PgturbohybridBm25Result *bm25 = NULL;
	PgturbohybridResult *items = NULL;
	PgturbohybridResult *merged = NULL;
	TqDenseCandidateStats denseStats;
	PgturbohybridBm25QueryStats bm25Stats;
	int			denseCount = 0;
	int			bm25Count = 0;
	int			itemCount = 0;
	int			mergedCount = 0;
	int			finalCount;
	bool		useHashTopN;
	uint32		fusionCandidatesSeen;
	MemoryContext oldCtx;
	PgturbohybridLastScanStats lastStats;
	instr_time	totalStart;
	instr_time	phaseStart;
	PgturbohybridQueryHeader *originalQuery = state->query;
	PgturbohybridQueryHeader *scanQuery;
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
	autoBudgetLimit = PgturbohybridCurrentLimit();
	scanQuery = PgturbohybridEffectiveQuery(originalQuery, autoBudgetLimit, so->tmpCtx);
	state->query = scanQuery;
	if ((scanQuery->flags & HYBRID_QUERY_FLAG_HAS_VECTOR) != 0 &&
		scanQuery->denseK > 0)
	{
		INSTR_TIME_SET_CURRENT(phaseStart);
		denseCount = PgturbohybridGraphCollectDenseCandidates(scan, scanQuery->denseK,
												   &dense, so->tmpCtx,
												   &denseStats);
		lastStats.denseElapsedUs = PgturbohybridElapsedUs(phaseStart);
	}

	PgturbohybridMaybeApplyDenseBm25Budget(scanQuery, dense, denseCount,
									  autoBudgetLimit, bm25BudgetReason,
									  sizeof(bm25BudgetReason),
									  &bm25DenseConfidence);
	PgturbohybridMaybeApplyBm25HybridBound(scanQuery, denseCount,
									  autoBudgetLimit,
									  &bm25HybridBoundStopRank,
									  &bm25HybridBoundSkippedEstimated,
									  &bm25HybridBoundThreshold,
									  &bm25HybridBoundSafe);

	if ((scanQuery->flags & HYBRID_QUERY_FLAG_HAS_TSQUERY) != 0 &&
		scanQuery->bm25K > 0)
	{
		PgturbohybridOptions *opts = (PgturbohybridOptions *) scan->indexRelation->rd_options;
		bool		useWand = pgturbohybrid_enable_wand &&
			(opts == NULL || opts->bm25BlockMax);

		INSTR_TIME_SET_CURRENT(phaseStart);
		bm25Count = PgturbohybridBm25TopK(scan->indexRelation,
									 PgturbohybridQueryGetTsQuery(scanQuery),
									 scanQuery->bm25K, useWand, so->tmpCtx,
									 &bm25, &bm25Stats);
		lastStats.bm25ElapsedUs = PgturbohybridElapsedUs(phaseStart);
	}

	INSTR_TIME_SET_CURRENT(phaseStart);
	fusionCandidatesSeen = denseCount + bm25Count;
	useHashTopN = pgturbohybrid_fusion_hash_threshold >= 0 &&
		fusionCandidatesSeen >= (uint32) pgturbohybrid_fusion_hash_threshold;
	if (useHashTopN)
	{
		uint32		slotCount =
			PgturbohybridFusionHashSlotCount(fusionCandidatesSeen);
		uint32		slotMask = slotCount - 1;
		PgturbohybridMergeSlot *slots =
			palloc0(sizeof(PgturbohybridMergeSlot) * slotCount);

		for (int i = 0; i < denseCount; i++)
		{
			PgturbohybridResult *item =
				PgturbohybridFindMergeSlot(slots, slotMask, dense[i].nodeId);

			PgturbohybridAddDenseCandidate(item, &dense[i]);
		}
		for (int i = 0; i < bm25Count; i++)
		{
			PgturbohybridResult *item =
				PgturbohybridFindMergeSlot(slots, slotMask, bm25[i].nodeId);

			if (!item->hasDense)
				item->heaptid = bm25[i].heaptid;
			item->nodeId = bm25[i].nodeId;
			item->bm25Score = bm25[i].bm25Score;
			item->bm25Rank = bm25[i].rank;
			item->hasBm25 = true;
		}

		merged = palloc0(sizeof(PgturbohybridResult) *
						 Max(fusionCandidatesSeen, 1));
		for (uint32 i = 0; i < slotCount; i++)
		{
			PgturbohybridResult item;

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

		PgturbohybridCheckBm25OnlyExactRescore(state, merged, mergedCount);
		PgturbohybridScoreResults(state, merged, mergedCount);

		finalCount = PgturbohybridFinalTarget(scanQuery, mergedCount);
		if (finalCount < mergedCount)
			finalCount = PgturbohybridSelectTopN(merged, mergedCount, finalCount,
											&merged, so->tmpCtx);
		else if (mergedCount > 1)
			qsort(merged, mergedCount, sizeof(PgturbohybridResult),
				  PgturbohybridScoreCompare);
		strlcpy(lastStats.fusionStrategy, "hash_topn",
				sizeof(lastStats.fusionStrategy));
		lastStats.fusionHeapSize = finalCount;
	}
	else
	{
		items = palloc0(sizeof(PgturbohybridResult) *
						Max((int) fusionCandidatesSeen, 1));
		for (int i = 0; i < denseCount; i++)
			PgturbohybridAddDenseCandidate(&items[itemCount++], &dense[i]);
		for (int i = 0; i < bm25Count; i++)
			PgturbohybridAddBm25Candidate(&items[itemCount++], &bm25[i]);

		if (itemCount > 1)
			qsort(items, itemCount, sizeof(PgturbohybridResult), PgturbohybridNodeCompare);

		merged = palloc0(sizeof(PgturbohybridResult) * Max(itemCount, 1));
		for (int i = 0; i < itemCount;)
		{
			PgturbohybridResult item = items[i++];

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

		PgturbohybridCheckBm25OnlyExactRescore(state, merged, mergedCount);
		PgturbohybridScoreResults(state, merged, mergedCount);
		if (mergedCount > 1)
			qsort(merged, mergedCount, sizeof(PgturbohybridResult),
				  PgturbohybridScoreCompare);

		finalCount = PgturbohybridFinalTarget(scanQuery, mergedCount);
		strlcpy(lastStats.fusionStrategy, "sort",
				sizeof(lastStats.fusionStrategy));
		lastStats.fusionHeapSize = finalCount;
	}

	lastStats.fusionElapsedUs = PgturbohybridElapsedUs(phaseStart);
	state->results = merged;
	state->resultCount = finalCount;
	state->resultIndex = 0;
	state->collectDone = true;

	strlcpy(lastStats.fusion,
			PgturbohybridQueryFusionName(pgturbohybrid_force_fusion != 0 ?
								  pgturbohybrid_force_fusion : scanQuery->fusion),
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
	lastStats.bm25HybridBoundMode = pgturbohybrid_bm25_hybrid_bound;
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
		lastStats.elapsedUs = PgturbohybridElapsedUs(totalStart);
		pgturbohybrid_last_scan_state = lastStats;
		state->query = originalQuery;
		MemoryContextSwitchTo(oldCtx);
	}

static IndexBuildResult *
tqhybridbuild(Relation heap, Relation index, IndexInfo *indexInfo)
{
	IndexBuildResult *result;

	PgturbohybridValidateIndex(index, indexInfo);
	result = pgturbohybridbuild(heap, index, indexInfo);

	PgturbohybridBm25BuildCollect(heap, index, indexInfo);

	return result;
}

static void
tqhybridbuildempty(Relation index)
{
	PgturbohybridValidateIndex(index, NULL);
	pgturbohybridbuildempty(index);
}

static bool
tqhybridinsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid, Relation heap, IndexUniqueCheck checkUnique
#if PG_VERSION_NUM >= 140000
			   , bool indexUnchanged
#endif
			   , IndexInfo *indexInfo)
{
	Datum		value;
	const PgturbohybridGraphTypeInfo *typeInfo = PgturbohybridGraphGetTypeInfo(index);
	PgturbohybridGraphMetaPageData meta;
	PgturbohybridGraphSupport support;
	uint32		nodeId;

	(void) heap;
	(void) checkUnique;
#if PG_VERSION_NUM >= 140000
	(void) indexUnchanged;
#endif
	PgturbohybridValidateIndex(index, indexInfo);
	if (isnull[0])
		return false;

	PgturbohybridGraphInitSupport(&support, index);
	if (!PgturbohybridGraphFormIndexValue(&value, values, isnull, typeInfo, &support))
		return false;

	LockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, ExclusiveLock);
	PG_TRY();
	{
		if (!PgturbohybridGraphReadMeta(index, &meta))
			elog(ERROR, "pgturbohybrid native graph metapage is missing or invalid");
		nodeId = meta.tqNodeCount;
		PgturbohybridGraphInsertValueInPlace(index, indexInfo, heap_tid, value,
								  values, isnull);
		if (!isnull[1])
			PgturbohybridBm25AppendDelta(index, nodeId, heap_tid, values[1]);
	}
	PG_CATCH();
	{
		UnlockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, ExclusiveLock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	UnlockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, ExclusiveLock);

	return true;
}

static IndexBulkDeleteResult *
tqhybridbulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats, IndexBulkDeleteCallback callback, void *callback_state)
{
	IndexBulkDeleteResult *result;

	if (PgturbohybridGraphUseTqNativeGraph(info->index))
	{
		result = tqgraphbulkdelete(info, stats, callback, callback_state);
		PgturbohybridBm25InvalidateCache(info->index);
	}
	else
		result = pgturbohybrid_graph_bulkdelete(info, stats, callback, callback_state);

	return result;
}

static IndexBulkDeleteResult *
tqhybridvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats)
{
	IndexBulkDeleteResult *result;

	if (PgturbohybridGraphUseTqNativeGraph(info->index))
	{
		result = tqgraphvacuumcleanup(info, stats);
		(void) PgturbohybridBm25MaybeCompact(info->index);
		PgturbohybridBm25InvalidateCache(info->index);
	}
	else
		result = pgturbohybrid_graph_vacuum_cleanup(info, stats);

	return result;
}

static IndexScanDesc
tqhybridbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;

	PgturbohybridValidateIndex(index, NULL);
	scan = pgturbohybridbeginscan(index, nkeys, norderbys);

	return scan;
}

static void
tqhybridrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	ScanKey		denseOrderbys = PgturbohybridDenseOrderBys(orderbys, norderbys);
	PgturbohybridQueryHeader *hybridQuery = NULL;
	bool		hasTextQuery = false;
	bool		hasVectorQuery = false;

	if (orderbys != NULL && norderbys > 0 &&
		(orderbys[0].sk_flags & SK_ISNULL) == 0)
	{
		hybridQuery = (PgturbohybridQueryHeader *) PG_DETOAST_DATUM_COPY(orderbys[0].sk_argument);
		PgturbohybridQueryValidate(hybridQuery);
		hasTextQuery = (hybridQuery->flags & HYBRID_QUERY_FLAG_HAS_TSQUERY) != 0;
		hasVectorQuery = (hybridQuery->flags & HYBRID_QUERY_FLAG_HAS_VECTOR) != 0;
	}

	pgturbohybridrescan(scan, keys, nkeys,
					 hasVectorQuery ? denseOrderbys : NULL,
					 hasVectorQuery ? norderbys : 0);

	if (hasTextQuery)
	{
		PgturbohybridGraphScanOpaque so = (PgturbohybridGraphScanOpaque) scan->opaque;
		MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);
		PgturbohybridScanState *state = palloc0(sizeof(PgturbohybridScanState));
		Size		querySize = VARSIZE_ANY(hybridQuery);

		state->query = palloc(querySize);
		memcpy(state->query, hybridQuery, querySize);
		state->active = true;
		so->tqHybridState = state;
		MemoryContextSwitchTo(oldCtx);
		PgturbohybridEnsureOrderByStorage(scan, so->tmpCtx);
	}
	else if (scan->opaque != NULL)
	{
		((PgturbohybridGraphScanOpaque) scan->opaque)->tqHybridState = NULL;
		scan->xs_orderbyvals = NULL;
		scan->xs_orderbynulls = NULL;
	}
}

static bool
tqhybridgettuple(IndexScanDesc scan, ScanDirection dir)
{
	PgturbohybridGraphScanOpaque so = (PgturbohybridGraphScanOpaque) scan->opaque;
	PgturbohybridScanState *state = so != NULL ?
		(PgturbohybridScanState *) so->tqHybridState : NULL;
	bool		result;

	if (state != NULL && state->active)
	{
		PgturbohybridResult *item;

		Assert(ScanDirectionIsForward(dir));
		PgturbohybridCollectScanResults(scan, state);
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

	result = pgturbohybridgettuple(scan, dir);

	return result;
}

static void
tqhybridendscan(IndexScanDesc scan)
{
	pgturbohybridendscan(scan);
}

static bool
PgturbohybridPathHasFilter(IndexPath *path)
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

static Oid
PgturbohybridExtensionSchema(Oid extensionOid)
{
	Form_pg_extension extensionForm;
	HeapTuple	tuple;
	Oid			schemaOid;

	tuple = SearchSysCache1(EXTENSIONOID, ObjectIdGetDatum(extensionOid));
	if (!HeapTupleIsValid(tuple))
		return InvalidOid;

	extensionForm = (Form_pg_extension) GETSTRUCT(tuple);
	schemaOid = extensionForm->extnamespace;
	ReleaseSysCache(tuple);

	return schemaOid;
}

static Oid
PgturbohybridQueryTypeOid(void)
{
	Oid			extensionOid;
	Oid			schemaOid;

	extensionOid = get_extension_oid("pgturbohybrid", true);
	if (!OidIsValid(extensionOid))
		return InvalidOid;

	schemaOid = PgturbohybridExtensionSchema(extensionOid);
	if (!OidIsValid(schemaOid))
		return InvalidOid;

	return GetSysCacheOid2(TYPENAMENSP, Anum_pg_type_oid,
						   CStringGetDatum("turbohybrid_query"),
						   ObjectIdGetDatum(schemaOid));
}

static bool
PgturbohybridFindConstQueryWalker(Node *node, void *context)
{
	PgturbohybridQueryHeader **query = (PgturbohybridQueryHeader **) context;
	Oid			hybridQueryOid;

	if (node == NULL || *query != NULL)
		return false;

	if (IsA(node, Const))
	{
		Const	   *constant = castNode(Const, node);

		if (constant->constisnull)
			return false;

		hybridQueryOid = PgturbohybridQueryTypeOid();
		if (OidIsValid(hybridQueryOid) && constant->consttype == hybridQueryOid)
		{
			*query = DatumGetHybridQuery(constant->constvalue);
			PgturbohybridQueryValidate(*query);
			return true;
		}
	}

	return expression_tree_walker(node, PgturbohybridFindConstQueryWalker, context);
}

static PgturbohybridQueryHeader *
PgturbohybridFindConstQuery(List *indexorderbys)
{
	PgturbohybridQueryHeader *query = NULL;
	ListCell   *lc;

	foreach(lc, indexorderbys)
	{
		if (PgturbohybridFindConstQueryWalker((Node *) lfirst(lc), &query))
			break;
	}

	return query;
}

static int
PgturbohybridEstimateTsQueryTerms(TSQuery query)
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
	PgturbohybridOptions *opts = NULL;
	PgturbohybridGraphMetaPageData graphMeta;
	PgturbohybridBm25PlanningStats bm25Stats;
	PgturbohybridQueryHeader *query;
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
	query = PgturbohybridFindConstQuery(path->indexorderbys);

	index = index_open(path->indexinfo->indexoid, NoLock);
	opts = (PgturbohybridOptions *) index->rd_options;
	MemSet(&graphMeta, 0, sizeof(graphMeta));
	if (!PgturbohybridGraphReadMeta(index, &graphMeta))
	{
		graphMeta.m = opts != NULL ? opts->m : PGTURBOHYBRID_GRAPH_DEFAULT_M;
		graphMeta.graphEfSearch = opts != NULL ?
			opts->graphEfSearch : PGTURBOHYBRID_DEFAULT_GRAPH_EF_SEARCH;
		graphMeta.graphOversampling = opts != NULL ?
			opts->graphOversampling : PGTURBOHYBRID_DEFAULT_GRAPH_OVERSAMPLING;
	}

	if (query != NULL)
	{
		denseK = PgturbohybridQueryGetVector(query) != NULL ? query->denseK : 0;
		bm25K = PgturbohybridQueryGetTsQuery(query) != NULL ? query->bm25K : 0;
		finalK = (query->flags & HYBRID_QUERY_FLAG_FINAL_K_IS_SET) != 0 ?
			query->finalK : Max(denseK + bm25K, 1);
		termCount = PgturbohybridEstimateTsQueryTerms(PgturbohybridQueryGetTsQuery(query));
	}
	else
	{
		denseK = opts != NULL ? opts->hybridDefaultDenseK : PGTURBOHYBRID_DEFAULT_DENSE_K;
		bm25K = opts != NULL ? opts->hybridDefaultBm25K : PGTURBOHYBRID_DEFAULT_BM25_K;
		finalK = Max(denseK + bm25K, 1);
		termCount = 2;
	}
	if (bm25K > 0)
		(void) PgturbohybridBm25GetPlanningStats(index, &bm25Stats);

	tuples = Max(path->indexinfo->tuples, 1.0);
	m = graphMeta.m > 0 ? graphMeta.m :
		(opts != NULL ? opts->m : PGTURBOHYBRID_GRAPH_DEFAULT_M);
	efSearch = graphMeta.graphEfSearch > 0 ? graphMeta.graphEfSearch :
		(opts != NULL ? opts->graphEfSearch : PGTURBOHYBRID_DEFAULT_GRAPH_EF_SEARCH);
	graphOversampling = graphMeta.graphOversampling > 0 ?
		graphMeta.graphOversampling :
		(opts != NULL ? opts->graphOversampling : PGTURBOHYBRID_DEFAULT_GRAPH_OVERSAMPLING);
	index_close(index, NoLock);

	if (PgturbohybridPathHasFilter(path))
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
		if (pgturbohybrid_enable_wand && bm25Stats.blockMaxPages > 0)
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
#if PG_VERSION_NUM >= 180000
#define PGTURBOHYBRID_RELOPT_PARSE(name, type, field) \
	{name, type, offsetof(PgturbohybridOptions, field), 0}
#else
#define PGTURBOHYBRID_RELOPT_PARSE(name, type, field) \
	{name, type, offsetof(PgturbohybridOptions, field)}
#endif
	static const relopt_parse_elt tab[] = {
		PGTURBOHYBRID_RELOPT_PARSE("graph_m", RELOPT_TYPE_INT, m),
		PGTURBOHYBRID_RELOPT_PARSE("graph_ef_construction", RELOPT_TYPE_INT, efConstruction),
		PGTURBOHYBRID_RELOPT_PARSE("routing", RELOPT_TYPE_ENUM, routing),
		PGTURBOHYBRID_RELOPT_PARSE("graph_ef_search", RELOPT_TYPE_INT, graphEfSearch),
		PGTURBOHYBRID_RELOPT_PARSE("graph_oversampling", RELOPT_TYPE_INT, graphOversampling),
		PGTURBOHYBRID_RELOPT_PARSE("quantization_bits", RELOPT_TYPE_INT, tqBits),
		PGTURBOHYBRID_RELOPT_PARSE("exact_storage", RELOPT_TYPE_BOOL, tqExactStorage),
	};
	PgturbohybridOptions *opts = (PgturbohybridOptions *) build_reloptions(reloptions, validate,
																 pgturbohybrid_relopt_kind,
																 sizeof(PgturbohybridOptions),
																 tab, lengthof(tab));
#undef PGTURBOHYBRID_RELOPT_PARSE

	if (opts != NULL)
	{
		opts->graphRescoreBand = PGTURBOHYBRID_GRAPH_RESCORE_BAND_AUTO;
		opts->graphExactCache = PGTURBOHYBRID_GRAPH_EXACT_CACHE_AUTO;
		opts->graphReorder = PGTURBOHYBRID_GRAPH_REORDER_OFF;
		opts->tqWeighted = false;
		opts->tqQuantileFit = false;
		opts->tqRenorm = false;
		opts->bm25K1 = PGTURBOHYBRID_DEFAULT_BM25_K1;
		opts->bm25B = PGTURBOHYBRID_DEFAULT_BM25_B;
		opts->bm25BlockMax = true;
		opts->bm25PrecomputeTfNorm = true;
		opts->bm25ImpactHead = true;
		opts->bm25ImpactMinDf = 1024;
		opts->bm25ImpactHeadK = 2048;
		opts->bm25DeltaCompactionThreshold = 25;
		opts->hybridDefaultFusion = HYBRID_FUSION_RRF;
		opts->hybridDefaultDenseK = PGTURBOHYBRID_DEFAULT_DENSE_K;
		opts->hybridDefaultBm25K = PGTURBOHYBRID_DEFAULT_BM25_K;
		opts->hybridDefaultRrfK = PGTURBOHYBRID_DEFAULT_RRF_K;
	}

	if (validate && opts != NULL &&
		opts->tqBits != 1 && opts->tqBits != 2 && opts->tqBits != PGTURBOHYBRID_DEFAULT_BITS)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid value %d for option \"quantization_bits\"", opts->tqBits),
				 errdetail("Valid values are \"1\", \"2\", and \"4\".")));

	return (bytea *) opts;
}

static bool
tqhybridvalidate(Oid opclassoid)
{
	HeapTuple	opclasstuple;
	Form_pg_opclass opclass;
	char	   *opcname;
	bool		valid = false;

	opclasstuple = SearchSysCache1(CLAOID, ObjectIdGetDatum(opclassoid));
	if (!HeapTupleIsValid(opclasstuple))
		return false;

	opclass = (Form_pg_opclass) GETSTRUCT(opclasstuple);
	opcname = NameStr(opclass->opcname);

	if (strcmp(opcname, "vector_l2_turbohybrid_ops") == 0 ||
		strcmp(opcname, "vector_ip_turbohybrid_ops") == 0 ||
		strcmp(opcname, "vector_cosine_turbohybrid_ops") == 0)
		valid = opclass->opcintype == PgturbohybridVectorTypeOid();
	else if (strcmp(opcname, "bm25_tsvector_turbohybrid_ops") == 0)
		valid = opclass->opcintype == TSVECTOROID;

	ReleaseSysCache(opclasstuple);
	return valid;
}

void
PgturbohybridInit(void)
{
	pgturbohybrid_relopt_kind = add_reloption_kind();
	prev_tqhybrid_ExecutorStart_hook = ExecutorStart_hook;
	ExecutorStart_hook = PgturbohybridExecutorStartHook;
	prev_tqhybrid_ExecutorEnd_hook = ExecutorEnd_hook;
	ExecutorEnd_hook = PgturbohybridExecutorEndHook;
	RegisterXactCallback(PgturbohybridXactCallback, NULL);
	RegisterSubXactCallback(PgturbohybridSubXactCallback, NULL);

	add_enum_reloption(pgturbohybrid_relopt_kind, "routing", "pgturbohybrid dense routing mode",
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
	add_int_reloption(pgturbohybrid_relopt_kind, "quantization_bits", "Quantized vector code bit width",
					  PGTURBOHYBRID_DEFAULT_INDEX_BITS, 1, PGTURBOHYBRID_DEFAULT_BITS, AccessExclusiveLock);
	add_bool_reloption(pgturbohybrid_relopt_kind, "exact_storage",
					   "Store exact vectors in the dense pgturbohybrid index for final exact rescoring.",
					   PGTURBOHYBRID_DEFAULT_EXACT_STORAGE, AccessExclusiveLock);

	DefineCustomIntVariable("turbohybrid.default_dense_k", "Default dense candidate budget for turbohybrid_query callers",
							NULL, &pgturbohybrid_default_dense_k,
							PGTURBOHYBRID_DEFAULT_DENSE_K, 0, INT_MAX, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.default_bm25_k", "Default BM25 candidate budget for turbohybrid_query callers",
							NULL, &pgturbohybrid_default_bm25_k,
							PGTURBOHYBRID_DEFAULT_BM25_K, 0, INT_MAX, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.default_rrf_k", "Default RRF constant for turbohybrid_query callers",
							NULL, &pgturbohybrid_default_rrf_k,
							PGTURBOHYBRID_DEFAULT_RRF_K, 1, INT_MAX, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("turbohybrid.enable_wand", "Enable WAND pruning for BM25 candidate generation",
							 NULL, &pgturbohybrid_enable_wand,
							 true, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.max_union_candidates", "Maximum candidates retained while fusing dense and BM25 branches",
							NULL, &pgturbohybrid_max_union_candidates,
							100000, 0, INT_MAX, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("turbohybrid.simd", "Enable SIMD kernels where supported by the host CPU",
							 NULL, &pgturbohybrid_simd,
							 true, PGC_USERSET, 0, NULL, PgturbohybridAssignSimd, NULL);
	MarkGUCPrefixReserved("turbohybrid");
}

static Datum
PgturbohybridHybridLastScanStats(PG_FUNCTION_ARGS)
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
						 pgturbohybrid_last_scan_state.fusion[0] != '\0' ?
						 pgturbohybrid_last_scan_state.fusion : "none",
						 pgturbohybrid_last_scan_state.denseCandidatesRequested,
						 pgturbohybrid_last_scan_state.denseCandidatesEffective,
						 pgturbohybrid_last_scan_state.denseKDefaulted ? "true" : "false",
						 pgturbohybrid_last_scan_state.denseCandidates,
					 pgturbohybrid_last_scan_state.denseEffectiveResultTarget,
					 pgturbohybrid_last_scan_state.denseEffectiveSearchEf,
					 pgturbohybrid_last_scan_state.denseEffectiveRescoreBand,
					 pgturbohybrid_last_scan_state.denseHighdimWideningMultiplier,
					 PgturbohybridGraphDenseWideningReasonName(pgturbohybrid_last_scan_state.denseWideningReason),
						 PgturbohybridGraphDenseBudgetPolicyNameExternal(pgturbohybrid_last_scan_state.denseBudgetPolicy),
						 PgturbohybridGraphRescoreBandPolicyNameExternal(pgturbohybrid_last_scan_state.denseRescoreBandPolicy),
						 pgturbohybrid_last_scan_state.bm25CandidatesRequested,
						 pgturbohybrid_last_scan_state.bm25CandidatesEffective,
						 pgturbohybrid_last_scan_state.bm25KDefaulted ? "true" : "false",
						 pgturbohybrid_last_scan_state.bm25Candidates,
						 pgturbohybrid_last_scan_state.bm25BudgetReason[0] != '\0' ?
						 pgturbohybrid_last_scan_state.bm25BudgetReason : "unknown",
						 pgturbohybrid_last_scan_state.bm25DenseConfidence,
						 PgturbohybridBm25HybridBoundModeName(pgturbohybrid_last_scan_state.bm25HybridBoundMode),
						 pgturbohybrid_last_scan_state.bm25HybridBoundStopRank,
						 pgturbohybrid_last_scan_state.bm25HybridBoundSkippedEstimated,
						 pgturbohybrid_last_scan_state.bm25HybridBoundThreshold,
						 pgturbohybrid_last_scan_state.bm25HybridBoundSafe ? "true" : "false",
						 pgturbohybrid_last_scan_state.rrfKRequested,
						 pgturbohybrid_last_scan_state.rrfKEffective,
						 pgturbohybrid_last_scan_state.rrfKDefaulted ? "true" : "false",
						 pgturbohybrid_last_scan_state.autoBudgetLimit,
						 pgturbohybrid_last_scan_state.unionCandidates,
					 pgturbohybrid_last_scan_state.finalResults,
					 pgturbohybrid_last_scan_state.fusionStrategy[0] != '\0' ?
					 pgturbohybrid_last_scan_state.fusionStrategy : "none",
					 pgturbohybrid_last_scan_state.fusionCandidatesSeen,
					 pgturbohybrid_last_scan_state.fusionHeapSize,
					 pgturbohybrid_last_scan_state.bothMatch,
					 pgturbohybrid_last_scan_state.denseOnly,
					 pgturbohybrid_last_scan_state.bm25Only,
					 pgturbohybrid_last_scan_state.graphVisitedNodes,
					 pgturbohybrid_last_scan_state.graphScoredCodes,
					 pgturbohybrid_last_scan_state.graphExactRescoreCount,
					 pgturbohybrid_last_scan_state.graphPrepareUs,
					 pgturbohybrid_last_scan_state.graphTraverseUs,
					 pgturbohybrid_last_scan_state.graphEntryUs,
					 pgturbohybrid_last_scan_state.graphBaseUs,
					 pgturbohybrid_last_scan_state.graphBatchUs,
					 pgturbohybrid_last_scan_state.graphHeapUs,
					 pgturbohybrid_last_scan_state.graphFillUs,
					 pgturbohybrid_last_scan_state.graphRescoreUs,
					 pgturbohybrid_last_scan_state.graphSortUs,
					 pgturbohybrid_last_scan_state.bm25Terms,
					 pgturbohybrid_last_scan_state.bm25PostingsDecoded,
					 pgturbohybrid_last_scan_state.bm25BlocksVisited,
					 pgturbohybrid_last_scan_state.bm25BlocksSkipped,
					 pgturbohybrid_last_scan_state.bm25CandidatesScored,
					 pgturbohybrid_last_scan_state.bm25CacheBytes,
					 pgturbohybrid_last_scan_state.bm25CacheLexiconEntries,
					 pgturbohybrid_last_scan_state.bm25CacheHit ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25CacheBuildUs,
					 pgturbohybrid_last_scan_state.bm25CacheDocstatsLoaded ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25CacheLivenessLoaded ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25HotPostingsCacheHits > 0 ?
					 "true" : "false",
					 pgturbohybrid_last_scan_state.bm25HotPostingsCacheHits,
					 pgturbohybrid_last_scan_state.bm25HotPostingsCacheMisses,
					 pgturbohybrid_last_scan_state.bm25HotPostingsCacheBytes,
					 pgturbohybrid_last_scan_state.bm25HotPostingsCacheEvictions,
					 PgturbohybridBm25DeltaLookupModeName(pgturbohybrid_last_scan_state.bm25DeltaLookupMode),
					 pgturbohybrid_last_scan_state.bm25DeltaPagesScanned,
					 pgturbohybrid_last_scan_state.bm25DeltaTermPagesRead,
					 pgturbohybrid_last_scan_state.bm25DeltaBlocksVisited,
					 pgturbohybrid_last_scan_state.bm25DeltaPostingsDecoded,
					 pgturbohybrid_last_scan_state.bm25DeltaCacheBytes,
					 pgturbohybrid_last_scan_state.bm25DeltaCacheTerms,
					 pgturbohybrid_last_scan_state.bm25DeltaCacheHit ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25WandIterations,
					 pgturbohybrid_last_scan_state.bm25WandThresholdUpdates,
					 pgturbohybrid_last_scan_state.bm25WandActiveSorts,
					 pgturbohybrid_last_scan_state.bm25WandHeapUpdates,
					 pgturbohybrid_last_scan_state.bm25WandFullReorders,
					 PgturbohybridBm25WandBoundTypeName(pgturbohybrid_last_scan_state.bm25WandBoundType),
					 pgturbohybrid_last_scan_state.bm25WandBoundTighteningHits,
					 pgturbohybrid_last_scan_state.bm25WandHeapReplacements,
					 PgturbohybridBm25RuntimeStrategyName(pgturbohybrid_last_scan_state.bm25Strategy),
					 PgturbohybridBm25StrategyName(pgturbohybrid_bm25_strategy),
					 pgturbohybrid_last_scan_state.bm25AndDriverDf,
					 pgturbohybrid_last_scan_state.bm25AndVerifiedCandidates,
					 pgturbohybrid_last_scan_state.bm25AndRejectedCandidates,
					 pgturbohybrid_last_scan_state.bm25ImpactTerms,
					 pgturbohybrid_last_scan_state.bm25ImpactTiersRead,
					 pgturbohybrid_last_scan_state.bm25ImpactPostingsRead,
					 pgturbohybrid_last_scan_state.bm25ImpactRemainingUpperBound,
					 pgturbohybrid_last_scan_state.bm25ImpactEarlyStop ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25ImpactExactSafe ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25ImpactFullPostingsAvoided ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25ImpactLoadedFromStorage ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25ImpactBuiltLazily ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25ImpactLazyPostingsScanned,
					 PgturbohybridBm25AccumulatorModeName(pgturbohybrid_last_scan_state.bm25AccumulatorMode),
					 pgturbohybrid_last_scan_state.bm25AccumulatorHashLookups,
					 pgturbohybrid_last_scan_state.bm25AccumulatorDenseUpdates,
					 pgturbohybrid_last_scan_state.bm25FinalHeapReplacements,
					 pgturbohybrid_last_scan_state.bm25FinalSortedCount,
					 pgturbohybrid_last_scan_state.bm25FullSortAvoided ? "true" : "false",
					 PgturbohybridBm25QueryShapeName(pgturbohybrid_last_scan_state.bm25QueryShape),
					 PgturbohybridBm25BooleanEvalModeName(pgturbohybrid_last_scan_state.bm25BooleanEvalMode),
					 pgturbohybrid_last_scan_state.bm25BooleanEvalCalls,
					 PgturbohybridBm25KernelName(pgturbohybrid_last_scan_state.bm25DecodeKernel),
					 PgturbohybridBm25KernelName(pgturbohybrid_last_scan_state.bm25ScoreKernel),
					 PgturbohybridBm25SimdForceName(pgturbohybrid_bm25_simd_force),
					 pgturbohybrid_last_scan_state.bm25SimdBlocks,
					 pgturbohybrid_last_scan_state.bm25ScalarTailPostings,
					 pgturbohybrid_last_scan_state.bm25Prefetches,
					 PgturbohybridGraphTqSimdForceName(pgturbohybrid_dense_simd_force),
					 PgturbohybridGraphTqExactSimdForceName(pgturbohybrid_dense_exact_simd_force),
					 pgturbohybrid_last_scan_state.denseElapsedUs,
					 pgturbohybrid_last_scan_state.bm25ElapsedUs,
					 pgturbohybrid_last_scan_state.fusionElapsedUs,
					 pgturbohybrid_last_scan_state.elapsedUs);

	PG_RETURN_DATUM(DirectFunctionCall1(jsonb_in, CStringGetDatum(json.data)));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_handler);
Datum
pgturbohybrid_handler(PG_FUNCTION_ARGS)
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
