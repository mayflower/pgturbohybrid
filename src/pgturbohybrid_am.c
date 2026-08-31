#include "postgres.h"

#include <limits.h>
#include <math.h>

#include "access/amapi.h"
#include "access/parallel.h"
#include "access/relscan.h"
#include "access/reloptions.h"
#include "access/sysattr.h"
#include "access/table.h"
#include "access/tableam.h"
#include "access/xact.h"
#include "catalog/namespace.h"
#include "catalog/pg_extension.h"
#include "catalog/pg_opclass.h"
#include "catalog/pg_type.h"
#include "catalog/pg_type_d.h"
#include "commands/extension.h"
#include "commands/vacuum.h"
#include "executor/executor.h"
#include "executor/tuptable.h"
#include "fmgr.h"
#include "miscadmin.h"
#include "nodes/bitmapset.h"
#include "nodes/execnodes.h"
#include "nodes/nodeFuncs.h"
#include "optimizer/cost.h"
#include "optimizer/optimizer.h"
#include "portability/instr_time.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/builtins.h"
#include "utils/float.h"
#include "utils/fmgrprotos.h"
#include "utils/fmgroids.h"
#include "utils/guc.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/selfuncs.h"
#include "utils/spccache.h"
#include "utils/syscache.h"

#include "pgturbohybrid.h"
#include "pgturbohybrid_multivector.h"
#include "pgturbohybrid_query.h"
#include "pgturbohybrid_quant.h"
#include "pgturbohybrid_quant_score.h"
#include "pgturbohybrid_am.h"
#include "pgturbohybrid_bm25.h"
#include "pgturbohybrid_sparse.h"
#include "pgturbohybrid_guc.h"

#define PGTURBOHYBRID_DEFAULT_BM25_K1 1.2
#define PGTURBOHYBRID_DEFAULT_BM25_B 0.75
#define PGTURBOHYBRID_DEFAULT_DENSE_K 100
#define PGTURBOHYBRID_DEFAULT_BM25_K 100
#define PGTURBOHYBRID_DEFAULT_SPARSE_K 100
#define PGTURBOHYBRID_DEFAULT_RRF_K 60
#define PGTURBOHYBRID_FUSION_GENERATION_ARRAY_MAX_BYTES (16 * 1024 * 1024)

static relopt_kind pgturbohybrid_relopt_kind;
static List *pgturbohybrid_plannedstmt_stack = NIL;
static PlannedStmt *pgturbohybrid_current_plannedstmt = NULL;
static bool pgturbohybrid_am_init_done = false;

static relopt_enum_elt_def pgturbohybrid_native_segments_relopt_options[] = {
	{"auto", 0},
	{"1", 1},
	{"2", 2},
	{"4", 4},
	{"8", 8},
	{NULL, 0}
};

/* Sparse-vector postings quantization mode. */
#define PGTURBOHYBRID_SPARSE_QUANT_MODE_F32 0
#define PGTURBOHYBRID_SPARSE_QUANT_MODE_PER_TERM_LINEAR 1
static relopt_enum_elt_def pgturbohybrid_sparse_quant_mode_relopt_options[] = {
	{"f32", PGTURBOHYBRID_SPARSE_QUANT_MODE_F32},
	{"per_term_linear", PGTURBOHYBRID_SPARSE_QUANT_MODE_PER_TERM_LINEAR},
	{NULL, 0}
};

/* Sparse-vector postings physical encoding (reloption; "auto" resolves at build). */
#define PGTURBOHYBRID_SPARSE_ENCODING_OPT_AUTO 0
#define PGTURBOHYBRID_SPARSE_ENCODING_OPT_OFFSET16_SOA 1
#define PGTURBOHYBRID_SPARSE_ENCODING_OPT_VARINT 2
#define PGTURBOHYBRID_SPARSE_ENCODING_OPT_BITPACKED 3
static relopt_enum_elt_def pgturbohybrid_sparse_encoding_relopt_options[] = {
	{"auto", PGTURBOHYBRID_SPARSE_ENCODING_OPT_AUTO},
	{"offset16_soa", PGTURBOHYBRID_SPARSE_ENCODING_OPT_OFFSET16_SOA},
	{"varint", PGTURBOHYBRID_SPARSE_ENCODING_OPT_VARINT},
	{"bitpacked", PGTURBOHYBRID_SPARSE_ENCODING_OPT_BITPACKED},
	{NULL, 0}
};

static relopt_enum_elt_def pgturbohybrid_entry_sidecar_strategy_relopt_options[] = {
	{"hash", PGTURBOHYBRID_ENTRY_SIDECAR_HASH},
	{"farthest_code", PGTURBOHYBRID_ENTRY_SIDECAR_FARTHEST_CODE},
	{"level_covering", PGTURBOHYBRID_ENTRY_SIDECAR_LEVEL_COVERING},
	{"hybrid_level_covering", PGTURBOHYBRID_ENTRY_SIDECAR_HYBRID_LEVEL_COVERING},
	{NULL, 0}
};

static relopt_enum_elt_def pgturbohybrid_multivector_graph_relopt_options[] = {
	{"token_nodes", PGTURBOHYBRID_MULTIVECTOR_GRAPH_TOKEN_NODES},
	{"document_nodes", PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES},
	{NULL, 0}
};

static relopt_enum_elt_def pgturbohybrid_multivector_doc_build_scorer_relopt_options[] = {
	{"proxy", PGTURBOHYBRID_MULTIVECTOR_DOC_BUILD_SCORER_PROXY},
	{"exact_symmetric", PGTURBOHYBRID_MULTIVECTOR_DOC_BUILD_SCORER_EXACT_SYMMETRIC},
	{NULL, 0}
};

static relopt_enum_elt_def pgturbohybrid_multivector_doc_storage_relopt_options[] = {
	{"f32", PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32},
	{"f16", PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F16},
	{"sq8", PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_SQ8},
	{"centroid_only", PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CENTROID_ONLY},
	{"proxy_only", PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_PROXY_ONLY},
	{NULL, 0}
};

static relopt_enum_elt_def pgturbohybrid_multivector_token_pooling_relopt_options[] = {
	{"off", PGTURBOHYBRID_MULTIVECTOR_TOKEN_POOLING_OFF},
	{"kmeans", PGTURBOHYBRID_MULTIVECTOR_TOKEN_POOLING_KMEANS},
	{"greedy_cosine", PGTURBOHYBRID_MULTIVECTOR_TOKEN_POOLING_GREEDY_COSINE},
	{NULL, 0}
};

static relopt_enum_elt_def pgturbohybrid_multivector_centroids_relopt_options[] = {
	{"off", PGTURBOHYBRID_MULTIVECTOR_CENTROIDS_OFF},
	{"kmeans", PGTURBOHYBRID_MULTIVECTOR_CENTROIDS_KMEANS},
	{NULL, 0}
};

static relopt_enum_elt_def pgturbohybrid_multivector_proxy_encoder_relopt_options[] = {
	{"mean", PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_MEAN},
	{"normalized_mean", PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_NORMALIZED_MEAN},
	{"first_token", PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_FIRST_TOKEN},
	{"max_abs_mean", PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_MAX_ABS_MEAN},
	{"centroid_mean", PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_CENTROID_MEAN},
	{"mean_pool", PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_MEAN},
	{"max_pool", PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_MAX_POOL},
	{"random_projection_fde", PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_RANDOM_PROJECTION_FDE},
	{"learned_projection_v1", PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_LEARNED_PROJECTION_V1},
	{NULL, 0}
};

static relopt_enum_elt_def pgturbohybrid_multivector_context_mode_relopt_options[] = {
	{"flat", PGTURBOHYBRID_MULTIVECTOR_CONTEXT_MODE_FLAT},
	{"context_level", PGTURBOHYBRID_MULTIVECTOR_CONTEXT_MODE_CONTEXT_LEVEL},
	{NULL, 0}
};

static relopt_enum_elt_def pgturbohybrid_multivector_field_mode_relopt_options[] = {
	{"off", PGTURBOHYBRID_MULTIVECTOR_FIELD_MODE_OFF},
	{"weighted", PGTURBOHYBRID_MULTIVECTOR_FIELD_MODE_WEIGHTED},
	{NULL, 0}
};

static void PgturbohybridXactCallback(XactEvent event, void *arg);
static void PgturbohybridSubXactCallback(SubXactEvent event, SubTransactionId mySubid,
								 SubTransactionId parentSubid, void *arg);

static IndexBuildResult *pgturbohybridambuild(Relation heap, Relation index, IndexInfo *indexInfo);
static void pgturbohybridambuildempty(Relation index);
static bool pgturbohybridaminsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid, Relation heap, IndexUniqueCheck checkUnique
#if PG_VERSION_NUM >= 140000
						   , bool indexUnchanged
#endif
						   , IndexInfo *indexInfo);
static IndexBulkDeleteResult *pgturbohybridambulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats, IndexBulkDeleteCallback callback, void *callback_state);
static IndexBulkDeleteResult *pgturbohybridamvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats);
static IndexScanDesc pgturbohybridambeginscan(Relation index, int nkeys, int norderbys);
static void pgturbohybridamrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys);
static void pgturbohybridamendscan(IndexScanDesc scan);
static void pgturbohybridamcostestimate(PlannerInfo *root, IndexPath *path, double loop_count,
								 Cost *indexStartupCost, Cost *indexTotalCost,
								 Selectivity *indexSelectivity, double *indexCorrelation,
								 double *indexPages);
static bytea *pgturbohybridamoptions(Datum reloptions, bool validate);
static bool pgturbohybridamvalidate(Oid opclassoid);
static void PgturbohybridEnsureOrderByStorage(IndexScanDesc scan, MemoryContext scanCtx);
static bool PgturbohybridPathHasFilter(IndexPath *path);
static bool PgturbohybridPathHasUnmappedFilter(IndexPath *path, Relation index);
static bool PgturbohybridIndexHasPayloadAttr(Relation index, AttrNumber heapAttno);
static bool PgturbohybridClauseIsPayloadInt4Equality(Node *node, Index relid,
											 AttrNumber heapAttno);
static bool PgturbohybridFindConstQueryWalker(Node *node, void *context);
static PgturbohybridQueryHeader *PgturbohybridFindConstQuery(List *indexorderbys);
static int PgturbohybridEstimateTsQueryTerms(TSQuery query);

typedef enum PgturbohybridHybridAdaptiveShape
{
	PGTURBOHYBRID_HYBRID_SHAPE_FIXED,
	PGTURBOHYBRID_HYBRID_SHAPE_NOT_HYBRID,
	PGTURBOHYBRID_HYBRID_SHAPE_RARE_IDENTIFIER,
	PGTURBOHYBRID_HYBRID_SHAPE_BROAD_NATURAL_LANGUAGE,
	PGTURBOHYBRID_HYBRID_SHAPE_MIXED
} PgturbohybridHybridAdaptiveShape;

typedef struct PgturbohybridHybridBudgetChoice
{
	bool		adaptive;
	int			queryShape;
	int			denseK;
	int			bm25K;
	int			finalK;
	int			rrfK;
	char		reason[96];
} PgturbohybridHybridBudgetChoice;

typedef struct PgturbohybridResult
{
	uint32		nodeId;
	ItemPointerData heaptid;
	double		denseDistance;
	double		denseSimilarity;
	double		multivectorDistance;
	double		multivectorSimilarity;
	double		bm25Score;
	double		sparseSimilarity;
	double		fusedScore;
	int32		denseRank;
	int32		multivectorRank;
	int32		bm25Rank;
	int32		sparseRank;
	bool		hasDense;
	bool		hasMultivector;
	bool		hasBm25;
	bool		hasSparse;
	bool		exactScored;
} PgturbohybridResult;

typedef struct PgturbohybridMergeSlot
{
	bool		used;
	PgturbohybridResult result;
} PgturbohybridMergeSlot;

typedef struct PgturbohybridFusionArrayEntry
{
	uint32		generation;
	PgturbohybridResult result;
} PgturbohybridFusionArrayEntry;

typedef struct PgturbohybridFusionGenerationArrayCache
{
	PgturbohybridFusionArrayEntry *entries;
	uint32		capacity;
	uint32		generation;
} PgturbohybridFusionGenerationArrayCache;

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
	char		indexShape[16];
	PgturbohybridBm25LastScanStats bm25;
	PgturbohybridDenseLastScanStats dense;
	PgturbohybridMultivectorStats multivector;
	PgturbohybridSparseSnapshotStats sparse;
	PgturbohybridBranchPlan branchPlan;
	char		profile[16];
	char		fusion[16];
	uint32		rrfKRequested;
	uint32		rrfKEffective;
	bool		rrfKDefaulted;
	uint32		finalKRequested;
	uint32		finalKEffective;
	uint32		detectedSqlLimit;
	bool		finalKInferred;
	uint32		autoBudgetLimit;
	uint32		unionCandidates;
	uint32		finalResults;
	PgturbohybridFusionLastScanStats fusionStats;
	PgturbohybridLearnedProjectionStats learnedProjection;
	uint64		compactMaxsimScoreUs;
	uint64		compactMaxsimPairs;
	uint64		compactMaxsimCacheHits;
	uint64		compactMaxsimCacheMisses;
	uint64		compactMaxsimBoundChecks;
	uint64		compactMaxsimDocsPruned;
	uint64		compactMaxsimTokensSkipped;
	PgturbohybridProxyStats proxy;
	PgturbohybridCentroidStats centroid;
		bool		fullMultivectorSidecarAvailable;
	PgturbohybridQuantizedInvertedStats quantizedInverted;
	PgturbohybridSidecarStats sidecar;
	PgturbohybridLearnedSparseStats learnedSparse;
	uint32		exactRerankCandidates;
	uint64		exactRerankTokensEvaluated;
	uint64		exactRerankTokensSkipped;
	uint64		exactRerankPairsSaved;
	bool		adaptiveRerankTopKChangedVsFull;
	PgturbohybridFinalDiversityLastScanStats finalDiversity;
	uint32		bothMatch;
	uint64		graphVisitedNodes;
	uint64		graphScoredCodes;
	uint64		graphExactRescoreCount;
	uint64		graphHeapRescoreCount;
	uint64		graphHeapFetchUs;
	uint64		graphHeapRescoreUs;
	bool		graphHeapRescoreAutoEnabled;
	int			graphHeapRescoreReason;
	int			graphExactRescoreSource;
	uint64		graphPrepareUs;
	uint64		graphTraverseUs;
	uint64		graphEntryUs;
	uint64		graphBaseUs;
	uint64		graphBatchUs;
	uint64		graphHeapUs;
	uint64		graphFillUs;
	uint64		graphFillCandidateBandCalls;
	int			graphFillCandidateBandReason;
	uint64		graphFillCandidateBandVisited;
	uint64		graphFillCandidateBandScored;
	uint64		graphFillCandidateBandSelectedBefore;
	uint64		graphFillCandidateBandSelectedAfter;
	uint64		graphFillCandidateBandTarget;
	bool		graphFillCandidateBandUsedPayloadRefs;
	uint64		graphFillCandidateBandPayloadRefCount;
	uint64		graphRescoreUs;
	uint64		graphSortUs;
	PgturbohybridFastWeightedStats fastWeighted;
	PgturbohybridCalibratedFusionStats calibratedFusion;
	PgturbohybridDbsfStats dbsf;
	PgturbohybridHybridBudgetStats hybrid;
	uint64		elapsedUs;
} PgturbohybridLastScanStats;

static PgturbohybridLastScanStats pgturbohybrid_last_scan_state;
static PgturbohybridFusionGenerationArrayCache pgturbohybrid_fusion_generation_array_cache;

void
PgturbohybridGetLastScanStatsSnapshot(PgturbohybridScanStatsSnapshot *stats)
{
	memset(stats, 0, sizeof(*stats));
	strlcpy(stats->indexShape,
			pgturbohybrid_last_scan_state.indexShape,
			sizeof(stats->indexShape));
	stats->bm25.branchAvailable =
		pgturbohybrid_last_scan_state.bm25.branchAvailable;
	stats->dense.branchUsed =
		pgturbohybrid_last_scan_state.dense.branchUsed;
	stats->multivector.branchUsed =
		pgturbohybrid_last_scan_state.multivector.branchUsed;
	stats->bm25.branchUsed =
		pgturbohybrid_last_scan_state.bm25.branchUsed;
	stats->sparse = pgturbohybrid_last_scan_state.sparse;
	stats->branchPlan = pgturbohybrid_last_scan_state.branchPlan;
	stats->dense.candidatesEffective =
		pgturbohybrid_last_scan_state.dense.candidatesEffective;
	stats->multivector.candidatesEffective =
		pgturbohybrid_last_scan_state.multivector.candidatesEffective;
	stats->dense.kDefaulted = pgturbohybrid_last_scan_state.dense.kDefaulted;
	stats->bm25.candidatesEffective =
		pgturbohybrid_last_scan_state.bm25.candidatesEffective;
	stats->bm25.kDefaulted = pgturbohybrid_last_scan_state.bm25.kDefaulted;
	stats->bm25.cacheHit = pgturbohybrid_last_scan_state.bm25.cacheHit;
	stats->bm25.cacheBuildUs = pgturbohybrid_last_scan_state.bm25.cacheBuildUs;
	stats->bm25.docstatsLoadedThisQuery =
		pgturbohybrid_last_scan_state.bm25.docstatsLoadedThisQuery;
	stats->bm25.livenessLoadedThisQuery =
		pgturbohybrid_last_scan_state.bm25.livenessLoadedThisQuery;
	stats->bm25.docstatsBytes =
		pgturbohybrid_last_scan_state.bm25.docstatsBytes;
	stats->bm25.livenessBytes =
		pgturbohybrid_last_scan_state.bm25.livenessBytes;
	stats->bm25.coldCacheONWork =
		pgturbohybrid_last_scan_state.bm25.coldCacheONWork;
	stats->bm25.postingsDecodeRatio =
		pgturbohybrid_last_scan_state.bm25.postingsDecodeRatio;
	stats->bm25.commonTermFallback =
		pgturbohybrid_last_scan_state.bm25.commonTermFallback;
	stats->bm25.wandPruned =
		pgturbohybrid_last_scan_state.bm25.wandPruned;
	stats->bm25.hotPostingsCacheHits =
		pgturbohybrid_last_scan_state.bm25.hotPostingsCacheHits;
	stats->bm25.hotPostingsCacheMisses =
		pgturbohybrid_last_scan_state.bm25.hotPostingsCacheMisses;
	stats->bm25.terms = pgturbohybrid_last_scan_state.bm25.terms;
	stats->bm25.fusedScoreBoundBlocksPruned =
		pgturbohybrid_last_scan_state.bm25.fusedScoreBoundBlocksPruned;
	stats->bm25.fusedScoreBoundCandidatesPruned =
		pgturbohybrid_last_scan_state.bm25.fusedScoreBoundCandidatesPruned;
	strlcpy(stats->bm25.heapTSVectorRerankMode,
			PgturbohybridBm25HeapTSVectorRerankModeName(
				pgturbohybrid_last_scan_state.bm25.heapTSVectorRerankMode),
			sizeof(stats->bm25.heapTSVectorRerankMode));
	stats->bm25.heapTSVectorRerankCount =
		pgturbohybrid_last_scan_state.bm25.heapTSVectorRerankCount;
	stats->bm25.heapTSVectorRerankFetchUs =
		pgturbohybrid_last_scan_state.bm25.heapTSVectorRerankFetchUs;
	stats->bm25.heapTSVectorRerankScoreUs =
		pgturbohybrid_last_scan_state.bm25.heapTSVectorRerankScoreUs;
	stats->bm25.heapTSVectorRerankTopKChanged =
		pgturbohybrid_last_scan_state.bm25.heapTSVectorRerankTopKChanged;
	stats->fastWeighted = pgturbohybrid_last_scan_state.fastWeighted;
	stats->calibratedFusion = pgturbohybrid_last_scan_state.calibratedFusion;
	stats->dbsf.enabled = pgturbohybrid_last_scan_state.dbsf.enabled;
	memcpy(stats->dbsf.branchMean,
		   pgturbohybrid_last_scan_state.dbsf.branchMean,
		   sizeof(stats->dbsf.branchMean));
	memcpy(stats->dbsf.branchStddev,
		   pgturbohybrid_last_scan_state.dbsf.branchStddev,
		   sizeof(stats->dbsf.branchStddev));
	memcpy(stats->dbsf.branchMin,
		   pgturbohybrid_last_scan_state.dbsf.branchMin,
		   sizeof(stats->dbsf.branchMin));
	memcpy(stats->dbsf.branchMax,
		   pgturbohybrid_last_scan_state.dbsf.branchMax,
		   sizeof(stats->dbsf.branchMax));
	stats->dbsf.degenerateBranches =
		pgturbohybrid_last_scan_state.dbsf.degenerateBranches;
	strlcpy(stats->bm25.normMode,
			pgturbohybrid_last_scan_state.bm25.normMode,
			sizeof(stats->bm25.normMode));
	strlcpy(stats->dense.normMode,
			pgturbohybrid_last_scan_state.dense.normMode,
			sizeof(stats->dense.normMode));
	stats->hybrid = pgturbohybrid_last_scan_state.hybrid;
	strlcpy(stats->fusionStats.strategy,
			pgturbohybrid_last_scan_state.fusionStats.strategy,
			sizeof(stats->fusionStats.strategy));
	stats->fusionStats.candidatesSeen =
		pgturbohybrid_last_scan_state.fusionStats.candidatesSeen;
	stats->fusionStats.duplicates =
		pgturbohybrid_last_scan_state.fusionStats.duplicates;
	stats->fusionStats.heapReplacements =
		pgturbohybrid_last_scan_state.fusionStats.heapReplacements;
	stats->fusionStats.generationArrayReused =
		pgturbohybrid_last_scan_state.fusionStats.generationArrayReused;
	stats->fusionStats.generationArrayReset =
		pgturbohybrid_last_scan_state.fusionStats.generationArrayReset;
	stats->multivector.enabled =
		pgturbohybrid_last_scan_state.multivector.enabled;
	stats->multivector.queryVectors =
		pgturbohybrid_last_scan_state.multivector.queryVectors;
	stats->multivector.docVectorsLimit =
		pgturbohybrid_last_scan_state.multivector.docVectorsLimit;
	stats->multivector.subvectorSearches =
		pgturbohybrid_last_scan_state.multivector.subvectorSearches;
	stats->multivector.rawSubvectorHits =
		pgturbohybrid_last_scan_state.multivector.rawSubvectorHits;
	stats->multivector.adaptiveWideningTriggered =
		pgturbohybrid_last_scan_state.multivector.adaptiveWideningTriggered;
	stats->multivector.adaptiveInitialRawTarget =
		pgturbohybrid_last_scan_state.multivector.adaptiveInitialRawTarget;
	stats->multivector.adaptiveFinalRawTarget =
		pgturbohybrid_last_scan_state.multivector.adaptiveFinalRawTarget;
	strlcpy(stats->multivector.docMapSource,
			pgturbohybrid_last_scan_state.multivector.docMapSource,
			sizeof(stats->multivector.docMapSource));
	strlcpy(stats->multivector.candidateSource,
			pgturbohybrid_last_scan_state.multivector.candidateSource,
			sizeof(stats->multivector.candidateSource));
	strlcpy(stats->multivector.candidatePath,
			pgturbohybrid_last_scan_state.multivector.candidatePath,
			sizeof(stats->multivector.candidatePath));
	strlcpy(stats->multivector.proxyEncoderKind,
			pgturbohybrid_last_scan_state.multivector.proxyEncoderKind,
			sizeof(stats->multivector.proxyEncoderKind));
	stats->learnedProjection = pgturbohybrid_last_scan_state.learnedProjection;
	strlcpy(stats->multivector.graphMode,
			pgturbohybrid_last_scan_state.multivector.graphMode,
			sizeof(stats->multivector.graphMode));
	stats->multivector.proxyGraphSearches =
		pgturbohybrid_last_scan_state.multivector.proxyGraphSearches;
	stats->multivector.exactTokenScanEnabled =
		pgturbohybrid_last_scan_state.multivector.exactTokenScanEnabled;
	stats->multivector.exactTokenScanNodesScored =
		pgturbohybrid_last_scan_state.multivector.exactTokenScanNodesScored;
	stats->multivector.plainFallbackUsed =
		pgturbohybrid_last_scan_state.multivector.plainFallbackUsed;
	strlcpy(stats->multivector.plainFallbackReason,
			pgturbohybrid_last_scan_state.multivector.plainFallbackReason,
			sizeof(stats->multivector.plainFallbackReason));
	stats->multivector.plainFallbackDocsScored =
		pgturbohybrid_last_scan_state.multivector.plainFallbackDocsScored;
	stats->multivector.plainFallbackPairs =
		pgturbohybrid_last_scan_state.multivector.plainFallbackPairs;
	stats->multivector.docGraphPrototypeEnabled =
		pgturbohybrid_last_scan_state.multivector.docGraphPrototypeEnabled;
	stats->multivector.docGraphNodes =
		pgturbohybrid_last_scan_state.multivector.docGraphNodes;
	stats->multivector.docGraphDocsScored =
		pgturbohybrid_last_scan_state.multivector.docGraphDocsScored;
	stats->multivector.docGraphEdgesVisited =
		pgturbohybrid_last_scan_state.multivector.docGraphEdgesVisited;
	stats->multivector.docGraphCandidates =
		pgturbohybrid_last_scan_state.multivector.docGraphCandidates;
	stats->multivector.docGraphSearchEf =
		pgturbohybrid_last_scan_state.multivector.docGraphSearchEf;
	stats->multivector.docGraphOversampling =
		pgturbohybrid_last_scan_state.multivector.docGraphOversampling;
	stats->multivector.docGraphRescoreK =
		pgturbohybrid_last_scan_state.multivector.docGraphRescoreK;
	stats->multivector.docGraphEntrySampleConfigured =
		pgturbohybrid_last_scan_state.multivector.docGraphEntrySampleConfigured;
	stats->multivector.docGraphEntrySampleEffective =
		pgturbohybrid_last_scan_state.multivector.docGraphEntrySampleEffective;
	stats->multivector.docGraphEntrySampleScored =
		pgturbohybrid_last_scan_state.multivector.docGraphEntrySampleScored;
	stats->multivector.docGraphQuantizedScores =
		pgturbohybrid_last_scan_state.multivector.docGraphQuantizedScores;
	stats->compactMaxsimScoreUs =
		pgturbohybrid_last_scan_state.compactMaxsimScoreUs;
	stats->compactMaxsimPairs =
		pgturbohybrid_last_scan_state.compactMaxsimPairs;
	stats->compactMaxsimCacheHits =
		pgturbohybrid_last_scan_state.compactMaxsimCacheHits;
	stats->compactMaxsimCacheMisses =
		pgturbohybrid_last_scan_state.compactMaxsimCacheMisses;
	stats->compactMaxsimBoundChecks =
		pgturbohybrid_last_scan_state.compactMaxsimBoundChecks;
	stats->compactMaxsimDocsPruned =
		pgturbohybrid_last_scan_state.compactMaxsimDocsPruned;
	stats->compactMaxsimTokensSkipped =
		pgturbohybrid_last_scan_state.compactMaxsimTokensSkipped;
	strlcpy(stats->multivector.docGraphStorageKind,
			pgturbohybrid_last_scan_state.multivector.docGraphStorageKind,
			sizeof(stats->multivector.docGraphStorageKind));
	stats->proxy = pgturbohybrid_last_scan_state.proxy;
	stats->centroid = pgturbohybrid_last_scan_state.centroid;
	stats->fullMultivectorSidecarAvailable =
		pgturbohybrid_last_scan_state.fullMultivectorSidecarAvailable;
	stats->quantizedInverted = pgturbohybrid_last_scan_state.quantizedInverted;
	strlcpy(stats->multivector.docGraphRescoreSource,
			pgturbohybrid_last_scan_state.multivector.docGraphRescoreSource,
			sizeof(stats->multivector.docGraphRescoreSource));
	stats->multivector.docGraphExactRerankDocs =
		pgturbohybrid_last_scan_state.multivector.docGraphExactRerankDocs;
	stats->multivector.docGraphHeapFetches =
		pgturbohybrid_last_scan_state.multivector.docGraphHeapFetches;
	strlcpy(stats->multivector.docGraphWarning,
			pgturbohybrid_last_scan_state.multivector.docGraphWarning,
			sizeof(stats->multivector.docGraphWarning));
	stats->multivector.proxyCandidateTarget =
		pgturbohybrid_last_scan_state.multivector.proxyCandidateTarget;
	stats->multivector.proxyCandidatesReturned =
		pgturbohybrid_last_scan_state.multivector.proxyCandidatesReturned;
	stats->multivector.exactRerankKEffective =
		pgturbohybrid_last_scan_state.multivector.exactRerankKEffective;
	strlcpy(stats->multivector.docStorageCacheRequested,
			pgturbohybrid_last_scan_state.multivector.docStorageCacheRequested,
			sizeof(stats->multivector.docStorageCacheRequested));
	strlcpy(stats->multivector.docStorageCacheEffective,
			pgturbohybrid_last_scan_state.multivector.docStorageCacheEffective,
			sizeof(stats->multivector.docStorageCacheEffective));
	stats->sidecar = pgturbohybrid_last_scan_state.sidecar;
	stats->multivector.centroidCount =
		pgturbohybrid_last_scan_state.multivector.centroidCount;
	stats->multivector.centroidPrerankDocs =
		pgturbohybrid_last_scan_state.multivector.centroidPrerankDocs;
	stats->multivector.fullMaxsimRerankDocs =
		pgturbohybrid_last_scan_state.multivector.fullMaxsimRerankDocs;
	strlcpy(stats->multivector.docSidecarCacheMode,
			pgturbohybrid_last_scan_state.multivector.docSidecarCacheMode,
			sizeof(stats->multivector.docSidecarCacheMode));
	stats->multivector.docSidecarPagesRead =
		pgturbohybrid_last_scan_state.multivector.docSidecarPagesRead;
	stats->multivector.docSidecarCacheHits =
		pgturbohybrid_last_scan_state.multivector.docSidecarCacheHits;
	stats->multivector.docSidecarCacheMisses =
		pgturbohybrid_last_scan_state.multivector.docSidecarCacheMisses;
	stats->multivector.docSidecarBytesTouched =
		pgturbohybrid_last_scan_state.multivector.docSidecarBytesTouched;
	stats->multivector.docSidecarVectorsLoaded =
		pgturbohybrid_last_scan_state.multivector.docSidecarVectorsLoaded;
	stats->multivector.docSidecarDocMapPagesRead =
		pgturbohybrid_last_scan_state.multivector.docSidecarDocMapPagesRead;
	stats->multivector.docSidecarDocMapBytesTouched =
		pgturbohybrid_last_scan_state.multivector.docSidecarDocMapBytesTouched;
	stats->multivector.docSidecarResidentVectorsLoaded =
		pgturbohybrid_last_scan_state.multivector.docSidecarResidentVectorsLoaded;
	stats->multivector.docSidecarResidentBytesLoaded =
		pgturbohybrid_last_scan_state.multivector.docSidecarResidentBytesLoaded;
	stats->multivector.docSidecarVectorChunkRefBytesTouched =
		pgturbohybrid_last_scan_state.multivector.docSidecarVectorChunkRefBytesTouched;
	stats->multivector.docSidecarPagedVectorPagesRead =
		pgturbohybrid_last_scan_state.multivector.docSidecarPagedVectorPagesRead;
	stats->multivector.docSidecarPagedVectorBytesTouched =
		pgturbohybrid_last_scan_state.multivector.docSidecarPagedVectorBytesTouched;
	stats->multivector.sidecarPageReadUs =
		pgturbohybrid_last_scan_state.multivector.sidecarPageReadUs;
	stats->multivector.sidecarVectorReconstructUs =
		pgturbohybrid_last_scan_state.multivector.sidecarVectorReconstructUs;
	stats->multivector.tokensOriginal =
		pgturbohybrid_last_scan_state.multivector.tokensOriginal;
	stats->multivector.tokensPooled =
		pgturbohybrid_last_scan_state.multivector.tokensPooled;
	stats->multivector.reservoirsEnabled =
		pgturbohybrid_last_scan_state.multivector.reservoirsEnabled;
	stats->multivector.reservoirScoreDocs =
		pgturbohybrid_last_scan_state.multivector.reservoirScoreDocs;
	stats->multivector.reservoirCoverageDocs =
		pgturbohybrid_last_scan_state.multivector.reservoirCoverageDocs;
	stats->multivector.reservoirMeanDocs =
		pgturbohybrid_last_scan_state.multivector.reservoirMeanDocs;
	stats->multivector.reservoirPerTokenDocs =
		pgturbohybrid_last_scan_state.multivector.reservoirPerTokenDocs;
	stats->multivector.reservoirBm25Docs =
		pgturbohybrid_last_scan_state.multivector.reservoirBm25Docs;
	stats->multivector.reservoirUnionDocs =
		pgturbohybrid_last_scan_state.multivector.reservoirUnionDocs;
	stats->multivector.reservoirDuplicates =
		pgturbohybrid_last_scan_state.multivector.reservoirDuplicates;
	stats->multivector.bm25InjectionEnabled =
		pgturbohybrid_last_scan_state.multivector.bm25InjectionEnabled;
	stats->multivector.bm25InjectionCandidates =
		pgturbohybrid_last_scan_state.multivector.bm25InjectionCandidates;
	stats->multivector.bm25InjectionCandidateLimit =
		pgturbohybrid_last_scan_state.multivector.bm25InjectionCandidateLimit;
	stats->multivector.bm25InjectionPoolSize =
		pgturbohybrid_last_scan_state.multivector.bm25InjectionPoolSize;
	strlcpy(stats->multivector.bm25InjectionLimitReason,
			pgturbohybrid_last_scan_state.multivector.bm25InjectionLimitReason,
			sizeof(stats->multivector.bm25InjectionLimitReason));
	stats->multivector.bm25InjectionRetained =
		pgturbohybrid_last_scan_state.multivector.bm25InjectionRetained;
	stats->multivector.bm25InjectionExactReranked =
		pgturbohybrid_last_scan_state.multivector.bm25InjectionExactReranked;
	stats->learnedSparse = pgturbohybrid_last_scan_state.learnedSparse;
	stats->multivector.docMapBytes =
		pgturbohybrid_last_scan_state.multivector.docMapBytes;
	stats->multivector.uniqueDocs =
		pgturbohybrid_last_scan_state.multivector.uniqueDocs;
	stats->multivector.duplicateDocHits =
		pgturbohybrid_last_scan_state.multivector.duplicateDocHits;
	stats->multivector.maxsimUpdates =
		pgturbohybrid_last_scan_state.multivector.maxsimUpdates;
	stats->multivector.docCandidates =
		pgturbohybrid_last_scan_state.multivector.docCandidates;
	stats->multivector.exactRerankEnabled =
		pgturbohybrid_last_scan_state.multivector.exactRerankEnabled;
	stats->multivector.exactRerankDocs =
		pgturbohybrid_last_scan_state.multivector.exactRerankDocs;
	stats->multivector.exactRerankPairs =
		pgturbohybrid_last_scan_state.multivector.exactRerankPairs;
	strlcpy(stats->multivector.exactRerankSource,
			pgturbohybrid_last_scan_state.multivector.exactRerankSource,
			sizeof(stats->multivector.exactRerankSource));
	stats->multivector.exactRerankHeapFetches =
		pgturbohybrid_last_scan_state.multivector.exactRerankHeapFetches;
	stats->multivector.exactRerankSidecarReads =
		pgturbohybrid_last_scan_state.multivector.exactRerankSidecarReads;
	stats->multivector.exactRerankSidecarBytes =
		pgturbohybrid_last_scan_state.multivector.exactRerankSidecarBytes;
	stats->multivector.candidateSourceUs =
		pgturbohybrid_last_scan_state.multivector.candidateSourceUs;
	stats->multivector.docGraphTraversalUs =
		pgturbohybrid_last_scan_state.multivector.docGraphTraversalUs;
	stats->multivector.proxyCandidateUs =
		pgturbohybrid_last_scan_state.multivector.proxyCandidateUs;
	stats->multivector.proxyGraphTraversalUs =
		pgturbohybrid_last_scan_state.multivector.proxyGraphTraversalUs;
	stats->multivector.proxyScoringUs =
		pgturbohybrid_last_scan_state.multivector.proxyScoringUs;
	stats->multivector.centroidLitePostingUs =
		pgturbohybrid_last_scan_state.multivector.centroidLitePostingUs;
	stats->multivector.quantizedInvertedPostingUs =
		pgturbohybrid_last_scan_state.multivector.quantizedInvertedPostingUs;
	stats->multivector.sidecarLoadUs =
		pgturbohybrid_last_scan_state.multivector.sidecarLoadUs;
	stats->multivector.heapVisibilityUs =
		pgturbohybrid_last_scan_state.multivector.heapVisibilityUs;
	stats->multivector.exactHeapFetchUs =
		pgturbohybrid_last_scan_state.multivector.exactHeapFetchUs;
	stats->multivector.exactRerankUs =
		pgturbohybrid_last_scan_state.multivector.exactRerankUs;
	stats->multivector.finalSortUs =
		pgturbohybrid_last_scan_state.multivector.finalSortUs;
	stats->exactRerankCandidates =
		pgturbohybrid_last_scan_state.exactRerankCandidates;
	stats->exactRerankTokensEvaluated =
		pgturbohybrid_last_scan_state.exactRerankTokensEvaluated;
	stats->exactRerankTokensSkipped =
		pgturbohybrid_last_scan_state.exactRerankTokensSkipped;
	stats->exactRerankPairsSaved =
		pgturbohybrid_last_scan_state.exactRerankPairsSaved;
	stats->adaptiveRerankTopKChangedVsFull =
		pgturbohybrid_last_scan_state.adaptiveRerankTopKChangedVsFull;
	strlcpy(stats->multivector.exactKernel,
			pgturbohybrid_last_scan_state.multivector.exactKernel,
			sizeof(stats->multivector.exactKernel));
	strlcpy(stats->multivector.accumulatorKind,
			pgturbohybrid_last_scan_state.multivector.accumulatorKind,
			sizeof(stats->multivector.accumulatorKind));
	stats->multivector.memoryBytesEstimate =
		pgturbohybrid_last_scan_state.multivector.memoryBytesEstimate;
	stats->multivector.admissionDebugEnabled =
		pgturbohybrid_last_scan_state.multivector.admissionDebugEnabled;
	stats->multivector.admissionCandidatesBeforeRerank =
		pgturbohybrid_last_scan_state.multivector.admissionCandidatesBeforeRerank;
	stats->multivector.admissionCandidatesAfterTruncation =
		pgturbohybrid_last_scan_state.multivector.admissionCandidatesAfterTruncation;
	stats->multivector.admissionExactRerankDocs =
		pgturbohybrid_last_scan_state.multivector.admissionExactRerankDocs;
	stats->multivector.admissionTruncatedByDocCandidateK =
		pgturbohybrid_last_scan_state.multivector.admissionTruncatedByDocCandidateK;
	stats->multivector.admissionTruncatedByAccumulatorMemory =
		pgturbohybrid_last_scan_state.multivector.admissionTruncatedByAccumulatorMemory;
	stats->multivector.admissionTraceAvailable =
		pgturbohybrid_last_scan_state.multivector.admissionTraceAvailable;
	stats->multivector.admissionTraceCount =
		pgturbohybrid_last_scan_state.multivector.admissionTraceCount;
	if (stats->multivector.admissionTraceCount >
		PGTURBOHYBRID_MULTIVECTOR_DEBUG_TRACE_LIMIT_MAX)
		stats->multivector.admissionTraceCount =
			PGTURBOHYBRID_MULTIVECTOR_DEBUG_TRACE_LIMIT_MAX;
	if (stats->multivector.admissionTraceCount > 0)
		memcpy(stats->multivector.admissionTrace,
			   pgturbohybrid_last_scan_state.multivector.admissionTrace,
			   sizeof(PgturbohybridMultiVectorAdmissionTraceEntry) *
			   stats->multivector.admissionTraceCount);
	stats->multivector.tokenStatsAvailable =
		pgturbohybrid_last_scan_state.multivector.tokenStatsAvailable;
	stats->multivector.tokenStatsCount =
		pgturbohybrid_last_scan_state.multivector.tokenStatsCount;
	if (stats->multivector.tokenStatsCount >
		PGTURBOHYBRID_MULTIVECTOR_TOKEN_STATS_LIMIT_MAX)
		stats->multivector.tokenStatsCount =
			PGTURBOHYBRID_MULTIVECTOR_TOKEN_STATS_LIMIT_MAX;
	if (stats->multivector.tokenStatsCount > 0)
		memcpy(stats->multivector.tokenStats,
			   pgturbohybrid_last_scan_state.multivector.tokenStats,
			   sizeof(PgturbohybridMultiVectorTokenStatsEntry) *
			   stats->multivector.tokenStatsCount);
	strlcpy(stats->finalDiversity.mode,
			PgturbohybridFinalDiversityName(
				pgturbohybrid_last_scan_state.finalDiversity.mode),
			sizeof(stats->finalDiversity.mode));
	stats->finalDiversity.payloadSlot =
		pgturbohybrid_last_scan_state.finalDiversity.payloadSlot;
	stats->finalDiversity.poolSize =
		pgturbohybrid_last_scan_state.finalDiversity.poolSize;
	stats->finalDiversity.selected =
		pgturbohybrid_last_scan_state.finalDiversity.selected;
	stats->finalDiversity.duplicateGroupsSuppressed =
		pgturbohybrid_last_scan_state.finalDiversity.duplicateGroupsSuppressed;
	stats->finalDiversity.us =
		pgturbohybrid_last_scan_state.finalDiversity.us;
	stats->dense.elapsedUs = pgturbohybrid_last_scan_state.dense.elapsedUs;
	stats->bm25.elapsedUs = pgturbohybrid_last_scan_state.bm25.elapsedUs;
	stats->fusionStats.elapsedUs = pgturbohybrid_last_scan_state.fusionStats.elapsedUs;
	stats->elapsedUs = pgturbohybrid_last_scan_state.elapsedUs;
}

static uint64
PgturbohybridElapsedUs(instr_time start)
{
	instr_time	duration;

	INSTR_TIME_SET_CURRENT(duration);
	INSTR_TIME_SUBTRACT(duration, start);
	return (uint64) INSTR_TIME_GET_MICROSEC(duration);
}

#ifdef PGTURBOHYBRID_DEV_DIAGNOSTICS
static Datum PgturbohybridHybridLastScanStats(PG_FUNCTION_ARGS) pg_attribute_unused();
#endif

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

void
PgturbohybridAmExecutorStart(QueryDesc *queryDesc, int eflags)
{
	(void) eflags;

	PgturbohybridPushPlannedStmt(queryDesc->plannedstmt);
}

void
PgturbohybridAmExecutorEnd(QueryDesc *queryDesc)
{
	(void) queryDesc;

	PgturbohybridPopPlannedStmt();
}

void
PgturbohybridAmExecutorAbort(void)
{
	PgturbohybridClearPlannedStmtStack();
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

/* Index-key position of the BM25 (tsvector) key, or -1.  Discovered by type so
 * it works whether bm25 is the 2nd key (dense+bm25) or a later key (e.g.
 * dense+sparse+bm25). */
static int
PgturbohybridIndexBm25KeyAttno(Relation index)
{
	PgturbohybridIndexKeyMap map;

	if (index == NULL || index->rd_index == NULL)
		return -1;
	PgturbohybridBuildIndexKeyMap(index, NULL, &map);
	return map.bm25Key;
}

bool
PgturbohybridIndexHasLexical(Relation index)
{
	return PgturbohybridIndexBm25KeyAttno(index) >= 0;
}

bool
PgturbohybridIndexGetLexicalDatum(Relation index, Datum *values, bool *isnull,
							 Datum *lexicalValue)
{
	int			bm25Key = PgturbohybridIndexBm25KeyAttno(index);

	if (bm25Key < 0 || values == NULL || isnull == NULL)
		return false;
	if (isnull[bm25Key])
		return false;

	*lexicalValue = values[bm25Key];
	return true;
}

/* Sparse-vector index value for an inserted row (false if no sparse key / NULL). */
static bool
PgturbohybridIndexGetSparseDatum(Relation index, Datum *values, bool *isnull,
								 Datum *sparseValue)
{
	PgturbohybridIndexKeyMap map;

	if (index == NULL || index->rd_index == NULL || values == NULL || isnull == NULL)
		return false;
	PgturbohybridBuildIndexKeyMap(index, NULL, &map);
	if (map.sparseKey < 0 || isnull[map.sparseKey])
		return false;
	*sparseValue = values[map.sparseKey];
	return true;
}

static bool
PgturbohybridIndexInfoHasLexical(Relation index, IndexInfo *indexInfo)
{
	PgturbohybridIndexKeyMap map;

	if (indexInfo != NULL)
	{
		PgturbohybridBuildIndexKeyMap(index, indexInfo, &map);
		return map.hasBm25;
	}

	return PgturbohybridIndexHasLexical(index);
}

static AttrNumber
PgturbohybridPathLexicalAttno(IndexPath *path)
{
	Relation	index;
	PgturbohybridIndexKeyMap map;
	int			bm25Key;

	/*
	 * Discover the bm25 (tsvector) key position by type rather than assuming the
	 * second column: with dense+sparse+bm25 indexes the
	 * bm25 key can be the third column, while the second is the sparse key.
	 */
	if (path == NULL || path->indexinfo == NULL)
		return InvalidAttrNumber;

	index = index_open(path->indexinfo->indexoid, NoLock);
	PgturbohybridBuildIndexKeyMap(index, NULL, &map);
	bm25Key = map.bm25Key;
	index_close(index, NoLock);

	if (bm25Key < 0 || bm25Key >= path->indexinfo->nkeycolumns)
		return InvalidAttrNumber;

	return path->indexinfo->indexkeys[bm25Key];
}

/*
 * Discover which index key carries each retrieval signal, by type (not fixed
 * position).  Validates structure: at most one of each kind, no unknown key
 * types, 1..MAX key columns, at least one retrieval key.  Policy (which
 * combinations are currently supported) is enforced by PgturbohybridValidateIndex.
 */
void
PgturbohybridBuildIndexKeyMap(Relation index, IndexInfo *indexInfo,
							  PgturbohybridIndexKeyMap *map)
{
	TupleDesc	desc = RelationGetDescr(index);
	Oid			vectorOid = PgturbohybridVectorTypeOid();
	int			keyCount;

	map->denseKey = map->multivectorKey = map->sparseKey = map->bm25Key = -1;
	map->graphKey = map->primaryKey = -1;
	map->hasDense = map->hasMultivector = map->hasSparse = map->hasBm25 = false;

	if (indexInfo != NULL)
	{
		keyCount = indexInfo->ii_NumIndexKeyAttrs;
		if (indexInfo->ii_Expressions != NIL)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("pgturbohybrid expression indexes are not supported yet")));
	}
	else if (index->rd_index != NULL)
		keyCount = index->rd_index->indnkeyatts;
	else
		keyCount = desc->natts;
	map->keyCount = keyCount;

	if (keyCount < 1 || keyCount > PGTURBOHYBRID_MAX_INDEX_KEYS)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("pgturbohybrid indexes support 1 to %d key columns",
						PGTURBOHYBRID_MAX_INDEX_KEYS)));

	if (desc->natts < keyCount)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("pgturbohybrid index tuple descriptor is missing key columns")));

	for (int i = 0; i < keyCount; i++)
	{
		Oid			t = TupleDescAttr(desc, i)->atttypid;

		if (OidIsValid(vectorOid) && t == vectorOid)
		{
			if (map->hasDense)
				ereport(ERROR,
						(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
						 errmsg("pgturbohybrid index supports at most one vector key")));
			map->denseKey = i;
			map->hasDense = true;
		}
		else if (PgturbohybridTypeIsMultiVector(t))
		{
			if (map->hasMultivector)
				ereport(ERROR,
						(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
						 errmsg("pgturbohybrid index supports at most one multivector key")));
			map->multivectorKey = i;
			map->hasMultivector = true;
		}
		else if (PgturbohybridTypeIsSparseVector(t))
		{
			if (map->hasSparse)
				ereport(ERROR,
						(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
						 errmsg("pgturbohybrid index supports at most one sparse-vector key")));
			map->sparseKey = i;
			map->hasSparse = true;
		}
		else if (t == TSVECTOROID)
		{
			if (map->hasBm25)
				ereport(ERROR,
						(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
						 errmsg("pgturbohybrid index supports at most one tsvector key")));
			map->bm25Key = i;
			map->hasBm25 = true;
		}
		else
			ereport(ERROR,
					(errcode(ERRCODE_DATATYPE_MISMATCH),
					 errmsg("pgturbohybrid index key %d has unsupported type %s",
							i + 1, format_type_be(t)),
					 errdetail("Supported key types: vector, multivector, turbohybrid_sparse_vector, tsvector.")));
	}

	if (map->hasDense && map->hasMultivector)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("pgturbohybrid index supports at most one vector or multivector key")));

	map->graphKey = map->hasDense ? map->denseKey :
		(map->hasMultivector ? map->multivectorKey : -1);
	map->primaryKey = map->graphKey >= 0 ? map->graphKey :
		(map->hasSparse ? map->sparseKey :
		 (map->hasBm25 ? map->bm25Key : -1));

	if (map->primaryKey < 0)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("pgturbohybrid index requires a vector, multivector, sparse-vector, or tsvector key")));
}

int
PgturbohybridIndexGraphKeyAttno(Relation index)
{
	PgturbohybridIndexKeyMap map;

	PgturbohybridBuildIndexKeyMap(index, NULL, &map);
	return map.graphKey;
}

static void
PgturbohybridValidateIndex(Relation index, IndexInfo *indexInfo)
{
	PgturbohybridIndexKeyMap map;

	PgturbohybridBuildIndexKeyMap(index, indexInfo, &map);

	/*
	 * Sparse-primary: a sparse-vector key with no dense/multivector
	 * graph (sparse-only or sparse+BM25).  Node identity then comes from the
	 * sparse-primary node-map chain instead of the dense graph, so there is no
	 * graph key to require or position.  BM25-only (tsvector without a sparse or
	 * graph key) remains unsupported and falls through to the error below.
	 */
	if (map.graphKey < 0)
	{
		if (map.hasSparse)
			return;
		ereport(ERROR,
				(errcode(ERRCODE_DATATYPE_MISMATCH),
				 errmsg("pgturbohybrid index requires a vector, multivector, or sparse-vector key"),
				 errhint("Add a vector or multivector key, or a turbohybrid_sparse_vector key, as an index column.")));
	}

	/*
	 * The graph key must be the first index column.  The dense/multivector
	 * build and scan paths read it from key position 0; secondary keys (sparse,
	 * bm25) are discovered by type and may follow in any order.  (Fully
	 * arbitrary graph-key placement is a possible future extension.)
	 */
	if (map.graphKey != 0)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("pgturbohybrid requires the vector or multivector key to be the first index column")));
}

static bool
PgturbohybridIndexIsMultiVector(Relation index)
{
	TupleDesc	desc = RelationGetDescr(index);

	return PgturbohybridTypeIsMultiVector(
		TupleDescAttr(desc, PGTURBOHYBRID_DENSE_KEY_INDEX)->atttypid);
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
		Vector	   *vectorCopy;
		Size		vectorSize;

		if (denseOrderbys[i].sk_flags & SK_ISNULL)
			continue;

		query = (PgturbohybridQueryHeader *) PG_DETOAST_DATUM_COPY(denseOrderbys[i].sk_argument);
		PgturbohybridQueryValidateFast(query);

		vectorQuery = PgturbohybridQueryGetVector(query);
		if (vectorQuery == NULL)
			continue;

		vectorSize = VARSIZE_ANY(vectorQuery);
		vectorCopy = palloc(vectorSize);
		memcpy(vectorCopy, vectorQuery, vectorSize);
		denseOrderbys[i].sk_argument = PointerGetDatum(vectorCopy);
		pfree(query);
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
PgturbohybridItemPointerDataCompare(const ItemPointerData *left,
									const ItemPointerData *right)
{
	BlockNumber leftBlock = ItemPointerGetBlockNumber(left);
	BlockNumber rightBlock = ItemPointerGetBlockNumber(right);
	OffsetNumber leftOffset = ItemPointerGetOffsetNumber(left);
	OffsetNumber rightOffset = ItemPointerGetOffsetNumber(right);

	if (leftBlock < rightBlock)
		return -1;
	if (leftBlock > rightBlock)
		return 1;
	if (leftOffset < rightOffset)
		return -1;
	if (leftOffset > rightOffset)
		return 1;
	return 0;
}

static int
PgturbohybridHeapTidCompare(const void *a, const void *b)
{
	const PgturbohybridResult *ra = (const PgturbohybridResult *) a;
	const PgturbohybridResult *rb = (const PgturbohybridResult *) b;
	int			cmp;

	cmp = PgturbohybridItemPointerDataCompare(&ra->heaptid, &rb->heaptid);
	if (cmp != 0)
		return cmp;
	return (ra->nodeId > rb->nodeId) - (ra->nodeId < rb->nodeId);
}

static int
PgturbohybridScoreCompare(const void *a, const void *b)
{
	const PgturbohybridResult *ra = (const PgturbohybridResult *) a;
	const PgturbohybridResult *rb = (const PgturbohybridResult *) b;
	bool		raDenseLike = ra->hasDense || ra->hasMultivector;
	bool		rbDenseLike = rb->hasDense || rb->hasMultivector;
	int			raBoth = (raDenseLike && ra->hasBm25) ? 1 : 0;
	int			rbBoth = (rbDenseLike && rb->hasBm25) ? 1 : 0;
	int32		raDenseRank = ra->hasDense ? ra->denseRank : INT_MAX;
	int32		rbDenseRank = rb->hasDense ? rb->denseRank : INT_MAX;
	int32		raMultiVectorRank =
		ra->hasMultivector ? ra->multivectorRank : INT_MAX;
	int32		rbMultiVectorRank =
		rb->hasMultivector ? rb->multivectorRank : INT_MAX;
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
	if (raMultiVectorRank != rbMultiVectorRank)
		return (raMultiVectorRank > rbMultiVectorRank) -
			(raMultiVectorRank < rbMultiVectorRank);
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

static uint32
PgturbohybridHashHeapTid(const ItemPointerData *heaptid)
{
	uint32		block = (uint32) ItemPointerGetBlockNumber(heaptid);
	uint32		offset = (uint32) ItemPointerGetOffsetNumber(heaptid);

	return (block * UINT32_C(2654435761)) ^ (offset * UINT32_C(2246822519));
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

static PgturbohybridResult *
PgturbohybridFindMergeSlotByHeapTid(PgturbohybridMergeSlot *slots,
									uint32 mask,
									const ItemPointerData *heaptid)
{
	uint32		slotNo = PgturbohybridHashHeapTid(heaptid) & mask;

	for (;;)
	{
		PgturbohybridMergeSlot *slot = &slots[slotNo];

		if (!slot->used)
		{
			slot->used = true;
			slot->result.heaptid = *heaptid;
			return &slot->result;
		}
		if (PgturbohybridItemPointerDataCompare(&slot->result.heaptid,
												heaptid) == 0)
			return &slot->result;
		slotNo = (slotNo + 1) & mask;
	}
}

static int
PgturbohybridEffectiveFinalK(PgturbohybridQueryHeader *query, int limit)
{
	if ((query->flags & PGTURBOHYBRID_QUERY_FLAG_FINAL_K_IS_SET) != 0)
		return query->finalK;
	if (limit > 0)
		return limit;
	return PGTURBOHYBRID_DEFAULT_FINAL_K;
}

static int
PgturbohybridFinalTarget(PgturbohybridQueryHeader *query, int mergedCount, int limit)
{
	int			finalCount = mergedCount;
	int			target = PgturbohybridEffectiveFinalK(query, limit);
	int64		activeLimit = PgturbohybridGraphGetActiveLimitTupleTarget();

	/*
	 * Resolve whether the scan sits under an executor LIMIT.  Prefer the
	 * plan-walked tuple target (set by the ExecutorStart hook), which sees a
	 * LIMIT even when the scan is nested inside a subquery/aggregate -- the
	 * 002_wal_restart query wraps the LIMIT in string_agg(... LIMIT 3), where
	 * PgturbohybridCurrentLimit() (passed as `limit`) reads 0.  Fall back to
	 * `limit` when the hook target is unset (< 0).
	 */
	if (activeLimit < 0 && limit > 0)
		activeLimit = limit;

	if (pgturbohybrid_max_union_candidates > 0)
		finalCount = Min(finalCount, pgturbohybrid_max_union_candidates);

	/*
	 * Hard-cap the emitted candidates at final_k only when the scan carries no
	 * executor LIMIT.  With a LIMIT the executor re-applies it after heap
	 * visibility checks, so the access method must over-return its (already
	 * oversampled) candidate band: a plain DELETE leaves the index node live
	 * (only VACUUM marks it dead), so the deleted nearest neighbour is still a
	 * candidate and the executor needs the candidates behind it to backfill the
	 * LIMIT.  Capping at exactly final_k/limit here silently dropped a row when
	 * the nearest tuple had been deleted (test/t/002_wal_restart.pl); 9caef97
	 * exposed it by routing dense-only queries through this fusion path.
	 * Without a LIMIT, final_k (explicit or default) is the output size, so the
	 * cap still applies.  The executor LIMIT bounds the real output either way,
	 * so over-returning costs only buffered candidates, not extra heap fetches.
	 */
	if (activeLimit <= 0)
		finalCount = Min(finalCount, target);
	return finalCount;
}

static int
PgturbohybridRequestedFinalK(PgturbohybridQueryHeader *query)
{
	if ((query->flags & PGTURBOHYBRID_QUERY_FLAG_FINAL_K_IS_SET) == 0)
		return 0;
	return query->finalK;
}

static bool
PgturbohybridFinalKInferred(PgturbohybridQueryHeader *query, int limit)
{
	return (query->flags & PGTURBOHYBRID_QUERY_FLAG_FINAL_K_IS_SET) == 0 &&
		limit > 0;
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

int
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

static void
PgturbohybridQueryValidateMultiVectorFusionSupport(Relation index,
												  PgturbohybridQueryHeader *query)
{
	uint16		requestedFusion;

	(void) index;
	if (query == NULL || !PgturbohybridQueryHasMultiVector(query))
		return;

	requestedFusion = pgturbohybrid_force_fusion != 0 ?
		pgturbohybrid_force_fusion : query->fusion;
	if (PgturbohybridQueryHasVector(query) &&
		requestedFusion != PGTURBOHYBRID_FUSION_RRF)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("vector_query plus multivector_query hybrid fusion supports rrf only"),
				 errdetail("Three-branch vector, multivector, and BM25 fusion keeps independent score spaces and currently uses rank-only reciprocal rank fusion.")));

	if (query == NULL ||
		!PgturbohybridQueryHasMultiVector(query) ||
		!PgturbohybridQueryHasText(query))
	{
		/*
		 * Dense-only multivector scans do not execute score fusion, so the
		 * fusion setting is currently ignored for that path.
		 */
		return;
	}

	if (requestedFusion != PGTURBOHYBRID_FUSION_RRF &&
		requestedFusion != PGTURBOHYBRID_FUSION_WEIGHTED &&
		requestedFusion != PGTURBOHYBRID_FUSION_FAST_WEIGHTED &&
		requestedFusion != PGTURBOHYBRID_FUSION_CALIBRATED &&
		requestedFusion != PGTURBOHYBRID_FUSION_DBSF)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("hybrid multivector fusion requires rank or normalized score fusion"),
				 errdetail("Supported multivector hybrid fusion modes are 'rrf', query-local normalized 'weighted', logistic/saturating 'fast_weighted', 'calibrated', and distribution-normalized 'dbsf'. Raw BM25 plus raw MaxSim alpha fusion is not supported.")));
}

static void
PgturbohybridCanonicalizeLatencyQuery(PgturbohybridQueryHeader *query, int limit)
{
	if (pgturbohybrid_profile != PGTURBOHYBRID_PROFILE_LATENCY)
		return;

	/*
	 * The latency profile's public defaults are chosen to match the recovered
	 * fast FIQA/OpenAI shape. Keep the original query around for diagnostics,
	 * but make the execution copy look like an explicit 100/100/60/final_k
	 * call once LIMIT/default resolution has happened. This keeps lower hot
	 * paths away from defaulted-query policy checks.
	 */
	query->finalK = PgturbohybridEffectiveFinalK(query, limit);
	query->flags |= PGTURBOHYBRID_QUERY_FLAG_FINAL_K_IS_SET;
	query->flags &= ~(PGTURBOHYBRID_QUERY_FLAG_DENSE_K_DEFAULTED |
					  PGTURBOHYBRID_QUERY_FLAG_BM25_K_DEFAULTED |
					  PGTURBOHYBRID_QUERY_FLAG_RRF_K_DEFAULTED);
}

static PgturbohybridQueryHeader *
PgturbohybridEffectiveQuery(PgturbohybridQueryHeader *query, int limit,
					   MemoryContext memoryContext)
{
	PgturbohybridQueryHeader *effective;
	Size		querySize = VARSIZE_ANY(query);
	bool		hasTsQuery =
		(query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_TSQUERY) != 0;

	effective = MemoryContextAlloc(memoryContext, querySize);
	memcpy(effective, query, querySize);

	effective->denseK = PgturbohybridApplyAutoBudget(query->denseK, limit,
												pgturbohybrid_auto_budget_min_dense_k,
												(query->flags & PGTURBOHYBRID_QUERY_FLAG_DENSE_K_DEFAULTED) != 0,
												PgturbohybridQueryHasVector(query));
	effective->multivectorK = PgturbohybridApplyAutoBudget(query->multivectorK, limit,
														   pgturbohybrid_auto_budget_min_dense_k,
														   (query->flags & PGTURBOHYBRID_QUERY_FLAG_DENSE_K_DEFAULTED) != 0,
														   PgturbohybridQueryHasMultiVector(query));
	effective->bm25K = PgturbohybridApplyAutoBudget(query->bm25K, limit,
											   pgturbohybrid_auto_budget_min_bm25_k,
											   (query->flags & PGTURBOHYBRID_QUERY_FLAG_BM25_K_DEFAULTED) != 0,
											   hasTsQuery);
	effective->sparseK = PgturbohybridApplyAutoBudget(query->sparseK, limit,
													  pgturbohybrid_auto_budget_min_bm25_k,
													  (query->flags & PGTURBOHYBRID_QUERY_FLAG_SPARSE_K_DEFAULTED) != 0,
													  PgturbohybridQueryHasSparse(query));
	PgturbohybridCanonicalizeLatencyQuery(effective, limit);

	return effective;
}

static int
PgturbohybridBudgetFinalTarget(PgturbohybridQueryHeader *query, int limit)
{
	if ((query->flags & PGTURBOHYBRID_QUERY_FLAG_FINAL_K_IS_SET) != 0 &&
		query->finalK > 0)
		return query->finalK;
	if (limit > 0)
		return limit;
	return PGTURBOHYBRID_DEFAULT_FINAL_K;
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

static const char *
PgturbohybridHybridQueryShapeName(int shape)
{
	switch ((PgturbohybridHybridAdaptiveShape) shape)
	{
		case PGTURBOHYBRID_HYBRID_SHAPE_FIXED:
			return "fixed";
		case PGTURBOHYBRID_HYBRID_SHAPE_NOT_HYBRID:
			return "not_hybrid";
		case PGTURBOHYBRID_HYBRID_SHAPE_RARE_IDENTIFIER:
			return "rare_identifier";
		case PGTURBOHYBRID_HYBRID_SHAPE_BROAD_NATURAL_LANGUAGE:
			return "broad_natural_language";
		case PGTURBOHYBRID_HYBRID_SHAPE_MIXED:
		default:
			return "mixed";
	}
}

static int
PgturbohybridAdaptiveMinBudget(int finalTarget)
{
	return Max(finalTarget * 2, 16);
}

static void
PgturbohybridInitHybridBudgetChoice(PgturbohybridHybridBudgetChoice *choice,
							   PgturbohybridQueryHeader *query)
{
	memset(choice, 0, sizeof(*choice));
	choice->queryShape = PGTURBOHYBRID_HYBRID_SHAPE_FIXED;
	choice->denseK = query->denseK;
	choice->bm25K = query->bm25K;
	choice->finalK = query->finalK;
	choice->rrfK = query->rrfK;
	strlcpy(choice->reason, "fixed_policy", sizeof(choice->reason));
}

static bool
PgturbohybridProfileAllowsAdaptiveRrfK(void)
{
	return pgturbohybrid_profile == PGTURBOHYBRID_PROFILE_LATENCY ||
		pgturbohybrid_profile == PGTURBOHYBRID_PROFILE_BALANCED ||
		pgturbohybrid_profile == PGTURBOHYBRID_PROFILE_MATCHED_RECALL;
}

static void
PgturbohybridApplyHybridBudgetChoice(PgturbohybridQueryHeader *query,
								const PgturbohybridQueryHeader *originalQuery,
								const PgturbohybridHybridBudgetChoice *choice)
{
	if (!choice->adaptive)
		return;

	if ((originalQuery->flags & PGTURBOHYBRID_QUERY_FLAG_DENSE_K_DEFAULTED) != 0)
		query->denseK = choice->denseK;
	if ((originalQuery->flags & PGTURBOHYBRID_QUERY_FLAG_BM25_K_DEFAULTED) != 0)
		query->bm25K = choice->bm25K;
	if ((originalQuery->flags & PGTURBOHYBRID_QUERY_FLAG_RRF_K_DEFAULTED) != 0 &&
		PgturbohybridProfileAllowsAdaptiveRrfK())
		query->rrfK = choice->rrfK;
	if ((originalQuery->flags & PGTURBOHYBRID_QUERY_FLAG_FINAL_K_IS_SET) == 0)
	{
		query->finalK = choice->finalK;
		query->flags |= PGTURBOHYBRID_QUERY_FLAG_FINAL_K_IS_SET;
	}
}

static void
PgturbohybridChooseAdaptiveHybridBudget(PgturbohybridQueryHeader *query,
								   const PgturbohybridQueryHeader *originalQuery,
								   const PgturbohybridBm25QuerySignals *signals,
								   double denseProbeGap, bool denseProbeAvailable,
								   int limit,
								   PgturbohybridHybridBudgetChoice *choice)
{
	bool		hasDense = PgturbohybridQueryHasDense(query);
	bool		hasTsQuery =
		(query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_TSQUERY) != 0;
	int			finalTarget;
	int			minBudget;
	bool		rare;
	bool		broad;

	PgturbohybridInitHybridBudgetChoice(choice, query);

	if (pgturbohybrid_hybrid_budget_policy != PGTURBOHYBRID_HYBRID_BUDGET_ADAPTIVE)
		return;

	choice->adaptive = true;
	strlcpy(choice->reason, "adaptive_approximate_policy", sizeof(choice->reason));

	if (!hasDense || !hasTsQuery)
	{
		choice->queryShape = PGTURBOHYBRID_HYBRID_SHAPE_NOT_HYBRID;
		strlcpy(choice->reason, "not_hybrid", sizeof(choice->reason));
		return;
	}

	finalTarget = PgturbohybridBudgetFinalTarget(query, limit);
	minBudget = PgturbohybridAdaptiveMinBudget(finalTarget);
	choice->finalK = finalTarget;

	rare = signals != NULL && signals->valid &&
		(signals->hasIdentifierToken ||
		 signals->maxIdf >= 2.8 ||
		 (signals->minPostings > 0 &&
		  signals->docCount > 0 &&
		  signals->minPostings <= Max(4U, signals->docCount / 20U)));
	broad = signals != NULL && signals->valid &&
		!signals->hasIdentifierToken &&
		signals->queryTerms >= 2 &&
		(signals->maxIdf < 1.5 ||
		 (signals->minPostings > 0 &&
		  signals->docCount > 0 &&
		  signals->minPostings >= Max(1U, signals->docCount / 2U)));

	if (denseProbeAvailable && denseProbeGap >= 0.08 && !rare)
		broad = true;

	if (rare)
	{
		choice->queryShape = PGTURBOHYBRID_HYBRID_SHAPE_RARE_IDENTIFIER;
		choice->denseK = Min(query->denseK, Max(minBudget, finalTarget * 4));
		choice->bm25K = query->bm25K;
		choice->rrfK = Min(query->rrfK, 30);
		strlcpy(choice->reason, signals != NULL && signals->hasIdentifierToken ?
				"approx_rare_identifier_reduce_dense" :
				"approx_high_idf_reduce_dense",
				sizeof(choice->reason));
	}
	else if (broad)
	{
		choice->queryShape = PGTURBOHYBRID_HYBRID_SHAPE_BROAD_NATURAL_LANGUAGE;
		choice->denseK = query->denseK;
		choice->bm25K = Min(query->bm25K, Max(minBudget, finalTarget * 4));
		choice->rrfK = Min(query->rrfK, 80);
		strlcpy(choice->reason,
				denseProbeAvailable && denseProbeGap >= 0.08 ?
				"approx_dense_confident_reduce_bm25" :
				"approx_broad_terms_reduce_bm25",
				sizeof(choice->reason));
	}
	else
	{
		choice->queryShape = PGTURBOHYBRID_HYBRID_SHAPE_MIXED;
		choice->denseK = Min(query->denseK, Max(minBudget, finalTarget * 6));
		choice->bm25K = Min(query->bm25K, Max(minBudget, finalTarget * 6));
		choice->rrfK = Min(query->rrfK, 60);
		strlcpy(choice->reason, "approx_mixed_balanced_budget",
				sizeof(choice->reason));
	}

	PgturbohybridApplyHybridBudgetChoice(query, originalQuery, choice);
}

static int
PgturbohybridHybridShapeFromSignals(PgturbohybridQueryHeader *query,
							   const PgturbohybridBm25QuerySignals *signals,
							   double denseProbeGap, bool denseProbeAvailable)
{
	bool		hasDense = PgturbohybridQueryHasDense(query);
	bool		hasTsQuery =
		(query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_TSQUERY) != 0;
	bool		rare;
	bool		broad;

	if (!hasDense || !hasTsQuery)
		return PGTURBOHYBRID_HYBRID_SHAPE_NOT_HYBRID;
	if (signals == NULL || !signals->valid)
		return PGTURBOHYBRID_HYBRID_SHAPE_MIXED;

	rare = signals->hasIdentifierToken ||
		signals->maxIdf >= 2.8 ||
		(signals->minPostings > 0 &&
		 signals->docCount > 0 &&
		 signals->minPostings <= Max(4U, signals->docCount / 20U));
	broad = !signals->hasIdentifierToken &&
		signals->queryTerms >= 2 &&
		(signals->maxIdf < 1.5 ||
		 (signals->minPostings > 0 &&
		  signals->docCount > 0 &&
		  signals->minPostings >= Max(1U, signals->docCount / 2U)));

	if (denseProbeAvailable && denseProbeGap >= 0.08 && !rare)
		broad = true;

	if (rare)
		return PGTURBOHYBRID_HYBRID_SHAPE_RARE_IDENTIFIER;
	if (broad)
		return PGTURBOHYBRID_HYBRID_SHAPE_BROAD_NATURAL_LANGUAGE;
	return PGTURBOHYBRID_HYBRID_SHAPE_MIXED;
}

static int
PgturbohybridHybridShapeFromName(const char *name)
{
	if (name == NULL || name[0] == '\0')
		return PGTURBOHYBRID_HYBRID_SHAPE_MIXED;
	if (strcmp(name, "rare_identifier") == 0)
		return PGTURBOHYBRID_HYBRID_SHAPE_RARE_IDENTIFIER;
	if (strcmp(name, "broad_natural_language") == 0)
		return PGTURBOHYBRID_HYBRID_SHAPE_BROAD_NATURAL_LANGUAGE;
	if (strcmp(name, "not_hybrid") == 0)
		return PGTURBOHYBRID_HYBRID_SHAPE_NOT_HYBRID;
	if (strcmp(name, "fixed") == 0)
		return PGTURBOHYBRID_HYBRID_SHAPE_FIXED;
	return PGTURBOHYBRID_HYBRID_SHAPE_MIXED;
}

static void
PgturbohybridMaybeApplyDenseBm25Budget(PgturbohybridQueryHeader *query,
								  const TqDenseCandidate *dense,
								  int denseCount, int limit,
								  char *reason, Size reasonSize,
								  double *denseConfidence)
{
	bool		hasDense = PgturbohybridQueryHasDense(query);
	bool		hasTsQuery =
		(query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_TSQUERY) != 0;
	bool		bm25Defaulted =
		(query->flags & PGTURBOHYBRID_QUERY_FLAG_BM25_K_DEFAULTED) != 0;
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
	if (!hasDense || !hasTsQuery)
	{
		if (reason != NULL && reasonSize > 0)
			strlcpy(reason, "not_hybrid", reasonSize);
		return;
	}
	if (query->fusion != PGTURBOHYBRID_FUSION_RRF)
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
PgturbohybridMaybeApplyMultiVectorAdmissionBudget(PgturbohybridQueryHeader *query,
										const PgturbohybridQueryHeader *originalQuery,
										const TqDenseCandidateStats *denseStats,
										int denseCount, int limit,
										int effectiveBm25Ceiling,
										PgturbohybridHybridBudgetChoice *choice)
{
	int			finalTarget;
	int			minBudget;
	int			reducedTarget;
	bool		bm25Defaulted;
	bool		admissionTruncated;
	bool		documentLevelSource;
	bool		exactDenseEvidence;

	if (query == NULL || originalQuery == NULL || denseStats == NULL ||
		choice == NULL)
		return;
	if (pgturbohybrid_hybrid_budget_policy !=
		PGTURBOHYBRID_HYBRID_BUDGET_ADAPTIVE)
		return;
	if (!PgturbohybridQueryHasMultiVector(query) ||
		!PgturbohybridQueryHasText(query))
		return;

	bm25Defaulted =
		(originalQuery->flags & PGTURBOHYBRID_QUERY_FLAG_BM25_K_DEFAULTED) != 0;
	if (!bm25Defaulted)
		return;
	if (query->bm25K <= 0)
		return;

	finalTarget = PgturbohybridBudgetFinalTarget(query, limit);
	if (finalTarget <= 0)
		finalTarget = PGTURBOHYBRID_DEFAULT_FINAL_K;
	minBudget = PgturbohybridAdaptiveMinBudget(finalTarget);
	admissionTruncated =
		denseStats->multivector.admissionTruncatedByDocCandidateK ||
		denseStats->multivector.admissionTruncatedByAccumulatorMemory;
	documentLevelSource =
		strcmp(denseStats->multivector.candidateSource, "document_nodes") == 0 ||
		strcmp(denseStats->multivector.candidateSource, "proxy_vector") == 0 ||
		strcmp(denseStats->multivector.candidateSource, "exact_doc_scan") == 0 ||
		strcmp(denseStats->multivector.candidateSource, "doc_graph_prototype") == 0;
	exactDenseEvidence =
		denseStats->multivector.admissionExactRerankDocs >= (uint32) finalTarget ||
		denseStats->multivector.exactRerankDocs >= (uint32) finalTarget ||
		denseStats->multivector.docGraphExactRerankDocs >= (uint32) finalTarget;

	choice->adaptive = true;
	choice->finalK = finalTarget;

	if (denseCount < finalTarget)
	{
		if (query->bm25K < effectiveBm25Ceiling)
			query->bm25K = effectiveBm25Ceiling;
		choice->bm25K = query->bm25K;
		strlcpy(choice->reason, "admission_underfilled_keep_bm25",
				sizeof(choice->reason));
		return;
	}

	if (admissionTruncated)
	{
		if (query->bm25K < effectiveBm25Ceiling)
			query->bm25K = effectiveBm25Ceiling;
		choice->bm25K = query->bm25K;
		strlcpy(choice->reason, "admission_truncated_keep_bm25",
				sizeof(choice->reason));
		return;
	}

	if (!documentLevelSource && !exactDenseEvidence)
	{
		choice->bm25K = query->bm25K;
		strlcpy(choice->reason, "admission_token_dense_keep_bm25",
				sizeof(choice->reason));
		return;
	}

	reducedTarget = Max(minBudget, finalTarget * 4);
	reducedTarget = Min(reducedTarget, effectiveBm25Ceiling);
	reducedTarget = Max(reducedTarget, 1);
	if (reducedTarget < query->bm25K)
		query->bm25K = reducedTarget;

	choice->bm25K = query->bm25K;
	choice->denseK = query->denseK;
	strlcpy(choice->reason,
			documentLevelSource ?
			"admission_document_dense_reduce_bm25" :
			"admission_exact_dense_reduce_bm25",
			sizeof(choice->reason));
}

static void
PgturbohybridMaybeApplyBm25HybridBound(PgturbohybridQueryHeader *query, int denseCount,
								  int limit, uint32 *stopRank,
								  uint32 *skippedEstimated,
								  double *threshold, bool *safe)
{
	bool		hasDense = PgturbohybridQueryHasDense(query);
	bool		hasTsQuery =
		(query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_TSQUERY) != 0;
	bool		bm25Defaulted =
		(query->flags & PGTURBOHYBRID_QUERY_FLAG_BM25_K_DEFAULTED) != 0;
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
		!hasDense || !hasTsQuery ||
		query->fusion != PGTURBOHYBRID_FUSION_RRF ||
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
PgturbohybridBranchPlanInit(PgturbohybridBranchPlan *plan, uint16 fusion)
{
	if (plan == NULL)
		return;
	memset(plan, 0, sizeof(*plan));
	plan->mode = pgturbohybrid_multivector_branch_plan;
	if (pgturbohybrid_multivector_branch_plan ==
		PGTURBOHYBRID_BRANCH_PLAN_QDRANT_LIKE)
	{
		if (fusion == PGTURBOHYBRID_FUSION_RRF)
			strlcpy(plan->fusionMode, "qdrant_like_rrf",
					sizeof(plan->fusionMode));
		else if (fusion == PGTURBOHYBRID_FUSION_DBSF)
			strlcpy(plan->fusionMode, "qdrant_like_dbsf",
					sizeof(plan->fusionMode));
		else if (fusion == PGTURBOHYBRID_FUSION_FAST_WEIGHTED ||
				 fusion == PGTURBOHYBRID_FUSION_CALIBRATED)
			strlcpy(plan->fusionMode, "qdrant_like_normalized",
					sizeof(plan->fusionMode));
		else
			strlcpy(plan->fusionMode, "qdrant_like_dense_only",
					sizeof(plan->fusionMode));
	}
	else if (pgturbohybrid_multivector_branch_plan ==
			 PGTURBOHYBRID_BRANCH_PLAN_DENSE_ONLY)
		strlcpy(plan->fusionMode, "dense_only", sizeof(plan->fusionMode));
	else
		strlcpy(plan->fusionMode, PgturbohybridQueryFusionName(fusion),
				sizeof(plan->fusionMode));
}

static void
PgturbohybridBranchPlanAdd(PgturbohybridBranchPlan *plan,
						   PgturbohybridBranchKind kind,
						   uint32 candidateLimit,
						   uint32 rescoreLimit,
						   uint32 branchRank,
						   double branchScore,
						   uint32 sourceFlags,
						   uint32 candidateCount,
						   bool truncated,
						   uint64 latencyUs)
{
	PgturbohybridBranchPlanItem *item;

	if (plan == NULL ||
		plan->count >= PGTURBOHYBRID_BRANCH_PLAN_MAX_BRANCHES)
		return;
	item = &plan->items[plan->count++];
	item->kind = kind;
	item->candidateLimit = candidateLimit;
	item->rescoreLimit = rescoreLimit;
	item->branchRank = branchRank;
	item->branchScore = branchScore;
	item->sourceFlags = sourceFlags;
	item->candidateCount = candidateCount;
	item->truncated = truncated;
	item->latencyUs = latencyUs;
}

static PgturbohybridBranchKind
PgturbohybridMultiVectorBranchKind(const TqDenseCandidateStats *denseStats)
{
	const char *source;

	if (denseStats == NULL)
		return PGTURBOHYBRID_BRANCH_KIND_TOKEN_NODES;
	source = denseStats->multivector.candidateSource;
	if (strcmp(source, "proxy_vector") == 0)
		return PGTURBOHYBRID_BRANCH_KIND_PROXY_VECTOR;
	if (strcmp(source, "centroid_lite") == 0)
		return PGTURBOHYBRID_BRANCH_KIND_CENTROID_LITE;
	if (strcmp(source, "quantized_inverted_experimental") == 0)
		return PGTURBOHYBRID_BRANCH_KIND_QUANTIZED_INVERTED_EXPERIMENTAL;
	if (strcmp(source, "document_nodes") == 0 ||
		strcmp(source, "doc_graph_prototype") == 0)
		return PGTURBOHYBRID_BRANCH_KIND_DOCUMENT_NODES;
	if (strcmp(source, "exact_doc_scan") == 0)
		return PGTURBOHYBRID_BRANCH_KIND_EXACT_DOC_SCAN;
	return PGTURBOHYBRID_BRANCH_KIND_TOKEN_NODES;
}

static void
PgturbohybridBuildBranchPlan(PgturbohybridBranchPlan *plan,
							 const PgturbohybridQueryHeader *query,
							 uint16 effectiveFusion,
							 const TqDenseCandidate *dense,
							 int denseCount,
							 bool denseInputIsMultiVector,
							 const PgturbohybridBm25Result *bm25,
							 int bm25Count,
							 const TqDenseCandidateStats *denseStats,
							 uint64 denseElapsedUs,
							 uint64 bm25ElapsedUs)
{
	bool		hasVector;
	bool		hasMultivector;
	bool		planDenseAsMultiVector;
	bool		denseTruncated;
	bool		bm25Injected;
	bool		emitBm25Branch;
	uint32		denseLimit;
	uint32		denseRescoreLimit;
	uint32		denseSourceFlags;
	double		denseScore = 0.0;
	double		bm25Score = 0.0;
	PgturbohybridBranchKind denseKind;

	PgturbohybridBranchPlanInit(plan, effectiveFusion);
	if (plan == NULL || query == NULL)
		return;

	hasVector =
		(query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_VECTOR) != 0;
	hasMultivector =
		(query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_MULTIVECTOR) != 0;
	planDenseAsMultiVector = hasMultivector && denseInputIsMultiVector;
	bm25Injected = denseStats != NULL &&
		denseStats->multivector.bm25InjectionEnabled;
	emitBm25Branch =
		pgturbohybrid_multivector_branch_plan !=
		PGTURBOHYBRID_BRANCH_PLAN_DENSE_ONLY;

	if (dense != NULL && denseCount > 0)
		denseScore = dense[0].similarity != 0.0 ?
			dense[0].similarity : -dense[0].distance;
	if (bm25 != NULL && bm25Count > 0)
		bm25Score = bm25[0].bm25Score;

	if ((hasVector || hasMultivector) &&
		(planDenseAsMultiVector ? query->multivectorK : query->denseK) > 0)
	{
		int32		requestedK = planDenseAsMultiVector ?
			query->multivectorK : query->denseK;

		denseLimit = (uint32) Max(requestedK, 0);
		denseRescoreLimit = denseLimit;
		denseTruncated = denseCount >= requestedK && requestedK > 0;
		denseKind = planDenseAsMultiVector ?
			PgturbohybridMultiVectorBranchKind(denseStats) :
			PGTURBOHYBRID_BRANCH_KIND_DENSE_SINGLE;

		if (denseStats != NULL && planDenseAsMultiVector)
		{
			if (denseStats->effectiveResultTarget > 0)
				denseLimit = denseStats->effectiveResultTarget;
			if (denseStats->multivector.docGraphExactRerankDocs > 0)
				denseRescoreLimit =
					denseStats->multivector.docGraphExactRerankDocs;
			else if (denseStats->multivector.exactRerankDocs > 0)
				denseRescoreLimit = denseStats->multivector.exactRerankDocs;
			denseTruncated =
				denseStats->multivector.admissionTruncatedByDocCandidateK ||
				denseStats->multivector.admissionTruncatedByAccumulatorMemory;
		}

		denseSourceFlags = PGTURBOHYBRID_BRANCH_SOURCE_DENSE |
			(planDenseAsMultiVector ?
			 PGTURBOHYBRID_BRANCH_SOURCE_MULTIVECTOR : 0) |
			(bm25Injected ? PGTURBOHYBRID_BRANCH_SOURCE_BM25 : 0);

		PgturbohybridBranchPlanAdd(plan, denseKind, denseLimit,
								   denseRescoreLimit, 1, denseScore,
								   denseSourceFlags,
								   (uint32) Max(denseCount, 0),
								   denseTruncated, denseElapsedUs);

		if (planDenseAsMultiVector &&
			pgturbohybrid_multivector_branch_plan ==
			PGTURBOHYBRID_BRANCH_PLAN_QDRANT_LIKE &&
			denseKind == PGTURBOHYBRID_BRANCH_KIND_PROXY_VECTOR &&
			denseStats != NULL)
		{
			PgturbohybridBranchPlanAdd(plan,
									   PGTURBOHYBRID_BRANCH_KIND_DOCUMENT_NODES,
									   denseStats->multivector.docGraphCandidates,
									   denseRescoreLimit, 2, denseScore,
									   PGTURBOHYBRID_BRANCH_SOURCE_DENSE |
									   PGTURBOHYBRID_BRANCH_SOURCE_MULTIVECTOR,
									   denseStats->multivector.docGraphCandidates,
									   denseTruncated, 0);
			PgturbohybridBranchPlanAdd(plan,
									   PGTURBOHYBRID_BRANCH_KIND_EXACT_DOC_SCAN,
									   denseRescoreLimit, denseRescoreLimit,
									   3, denseScore,
									   PGTURBOHYBRID_BRANCH_SOURCE_DENSE |
									   PGTURBOHYBRID_BRANCH_SOURCE_MULTIVECTOR |
									   PGTURBOHYBRID_BRANCH_SOURCE_EXACT,
									   denseStats->multivector.docGraphExactRerankDocs,
									   false, denseStats->heapRescoreUs);
		}
	}

	if (emitBm25Branch &&
		(query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_TSQUERY) != 0 &&
		query->bm25K > 0 && bm25Count > 0)
	{
		PgturbohybridBranchPlanAdd(plan, PGTURBOHYBRID_BRANCH_KIND_BM25,
								   (uint32) query->bm25K, 0,
								   plan->count + 1, bm25Score,
								   PGTURBOHYBRID_BRANCH_SOURCE_BM25,
								   (uint32) bm25Count,
								   bm25Count >= query->bm25K,
								   bm25ElapsedUs);
		if (hasMultivector &&
			pgturbohybrid_multivector_branch_plan ==
			PGTURBOHYBRID_BRANCH_PLAN_QDRANT_LIKE)
		{
			uint32		exactCount = bm25Injected ?
				denseStats->multivector.bm25InjectionExactReranked :
				(uint32) bm25Count;

			PgturbohybridBranchPlanAdd(plan,
									   PGTURBOHYBRID_BRANCH_KIND_EXACT_DOC_SCAN,
									   (uint32) bm25Count,
									   exactCount,
									   plan->count + 1,
									   denseScore,
									   PGTURBOHYBRID_BRANCH_SOURCE_BM25 |
									   PGTURBOHYBRID_BRANCH_SOURCE_MULTIVECTOR |
									   PGTURBOHYBRID_BRANCH_SOURCE_EXACT,
									   exactCount,
									   false, 0);
		}
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
						PgturbohybridResult **topItems,
						MemoryContext memoryContext,
						uint64 *heapReplacements)
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
			if (heapReplacements != NULL)
				(*heapReplacements)++;
		}
	}

	if (heapCount > 1)
		qsort(heap, heapCount, sizeof(PgturbohybridResult), PgturbohybridScoreCompare);
	*topItems = heap;
	return heapCount;
}

typedef struct PgturbohybridFinalDiversityCandidate
{
	PgturbohybridResult result;
	int32		groupValue;
	bool		selected;
} PgturbohybridFinalDiversityCandidate;

static int
PgturbohybridFinalDiversityDuplicateCount(const PgturbohybridFinalDiversityCandidate *candidates,
										  int count)
{
	int			duplicates = 0;

	for (int i = 0; i < count; i++)
	{
		bool		seen = false;

		for (int j = 0; j < i; j++)
		{
			if (candidates[j].groupValue == candidates[i].groupValue)
			{
				seen = true;
				break;
			}
		}
		if (seen)
			duplicates++;
	}

	return duplicates;
}

static bool
PgturbohybridFinalDiversityGroupSeen(const int32 *groups, int count,
									 int32 groupValue)
{
	for (int i = 0; i < count; i++)
	{
		if (groups[i] == groupValue)
			return true;
	}
	return false;
}

static bool
PgturbohybridFinalDiversityNodeSelected(const PgturbohybridFinalDiversityCandidate *selected,
										int count,
										uint32 nodeId)
{
	for (int i = 0; i < count; i++)
	{
		if (selected[i].result.nodeId == nodeId)
			return true;
	}
	return false;
}

static bool
PgturbohybridApplyFinalDiversity(IndexScanDesc scan,
								 PgturbohybridResult *merged,
								 int mergedCount,
								 int selectionCount,
								 int emitCount,
								 MemoryContext memoryContext,
								 PgturbohybridResult **finalResults,
								 uint64 *heapReplacements,
								 PgturbohybridLastScanStats *stats)
{
	PgturbohybridGraphScanOpaque so = (PgturbohybridGraphScanOpaque) scan->opaque;
	PgturbohybridGraphMetaPageData meta;
	PgturbohybridGraphScanStorage storage;
	PgturbohybridResult *poolResults = NULL;
	PgturbohybridFinalDiversityCandidate *candidates;
	PgturbohybridFinalDiversityCandidate *selectedCandidates;
	int32	   *selectedGroups;
	int			payloadSlot = pgturbohybrid_final_diversity_payload_slot;
	int			poolMultiplier = Max(pgturbohybrid_final_diversity_pool_multiplier, 1);
	int			poolCount;
	volatile int selectedCount = 0;
	double		bestScore;
	double		worstScore;
	double		scoreRange;
	double		lambda = pgturbohybrid_final_diversity_lambda;
	int			beforeDuplicates;
	int			afterDuplicates;
	instr_time	start;
	instr_time	lockStart;
	volatile bool applied = false;

	stats->finalDiversity.mode = pgturbohybrid_final_diversity;
	stats->finalDiversity.payloadSlot = payloadSlot;

	if (pgturbohybrid_final_diversity !=
		PGTURBOHYBRID_FINAL_DIVERSITY_GROUP_PAYLOAD)
		return false;
	if (selectionCount <= 0 || emitCount <= 0 || mergedCount <= selectionCount)
		return false;
	if (payloadSlot < 0 || so == NULL)
		return false;

	INSTR_TIME_SET_CURRENT(start);

	if (!PgturbohybridGraphReadMeta(scan->indexRelation, &meta) ||
		meta.tqNodeCount == 0 ||
		payloadSlot >= (int) meta.tqPayloadCount)
	{
		stats->finalDiversity.us = PgturbohybridElapsedUs(start);
		return false;
	}

	poolCount = (int) Min((int64) mergedCount,
						  (int64) selectionCount * (int64) poolMultiplier);
	if (poolCount <= selectionCount)
	{
		stats->finalDiversity.us = PgturbohybridElapsedUs(start);
		return false;
	}

	if (poolCount < mergedCount)
		poolCount = PgturbohybridSelectTopN(merged, mergedCount, poolCount,
											&poolResults, memoryContext,
											heapReplacements);
	else
	{
		if (mergedCount > 1)
			qsort(merged, mergedCount, sizeof(PgturbohybridResult),
				  PgturbohybridScoreCompare);
		poolResults = merged;
	}

	stats->finalDiversity.poolSize = poolCount;

	candidates = MemoryContextAllocZero(memoryContext,
										sizeof(*candidates) * Max(poolCount, 1));
	selectedCandidates = MemoryContextAllocZero(memoryContext,
											   sizeof(*selectedCandidates) *
											   Max(selectionCount, 1));
	selectedGroups = MemoryContextAllocZero(memoryContext,
											sizeof(int32) * Max(selectionCount, 1));

	INSTR_TIME_SET_CURRENT(lockStart);
	LockPage(scan->indexRelation, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ShareLock);
	so->graphScanLockWaitUs += PgturbohybridElapsedUs(lockStart);
	PG_TRY();
	{
		PgturbohybridGraphInitScanStorage(scan->indexRelation, &meta, &storage,
										  NULL);

		for (int i = 0; i < poolCount; i++)
		{
			int32		payloadValue;

			if (!PgturbohybridGraphLoadPayloadValue(scan->indexRelation, so,
													&meta, &storage,
													poolResults[i].nodeId,
													payloadSlot,
													&payloadValue))
				goto diversity_done;

			candidates[i].result = poolResults[i];
			candidates[i].groupValue = payloadValue;
		}

		bestScore = candidates[0].result.fusedScore;
		worstScore = candidates[0].result.fusedScore;
		for (int i = 1; i < poolCount; i++)
		{
			bestScore = Max(bestScore, candidates[i].result.fusedScore);
			worstScore = Min(worstScore, candidates[i].result.fusedScore);
		}
		scoreRange = bestScore - worstScore;
		if (scoreRange <= 0.0 || isnan(scoreRange) || isinf(scoreRange))
			scoreRange = 1.0;

		beforeDuplicates =
			PgturbohybridFinalDiversityDuplicateCount(candidates, selectionCount);

		while (selectedCount < selectionCount)
		{
			int			bestIndex = -1;
			double		bestDiversityScore = -HUGE_VAL;

			CHECK_FOR_INTERRUPTS();
			for (int i = 0; i < poolCount; i++)
			{
				double		relevance;
				double		penalty;
				double		diversityScore;

				if (candidates[i].selected)
					continue;

				relevance = (candidates[i].result.fusedScore - worstScore) /
					scoreRange;
				penalty = PgturbohybridFinalDiversityGroupSeen(selectedGroups,
															   selectedCount,
															   candidates[i].groupValue) ? 1.0 : 0.0;
				diversityScore = lambda * relevance - (1.0 - lambda) * penalty;
				if (bestIndex < 0 ||
					diversityScore > bestDiversityScore ||
					(diversityScore == bestDiversityScore &&
					 PgturbohybridScoreCompare(&candidates[i].result,
											   &candidates[bestIndex].result) < 0))
				{
					bestIndex = i;
					bestDiversityScore = diversityScore;
				}
			}

			if (bestIndex < 0)
				break;

			candidates[bestIndex].selected = true;
			selectedCandidates[selectedCount] = candidates[bestIndex];
			selectedGroups[selectedCount] = candidates[bestIndex].groupValue;
			selectedCount++;
		}

		if (selectedCount == selectionCount)
		{
			PgturbohybridResult *out;
			int			outCount = Min(emitCount, mergedCount);
			int			outIndex = 0;

			afterDuplicates =
				PgturbohybridFinalDiversityDuplicateCount(selectedCandidates,
														  selectedCount);
			out = MemoryContextAllocZero(memoryContext,
										 sizeof(PgturbohybridResult) *
										 Max(outCount, 1));
			for (int i = 0; i < selectionCount && outIndex < outCount; i++)
				out[outIndex++] = selectedCandidates[i].result;

			if (outIndex < outCount)
			{
				if (poolResults != merged && mergedCount > 1)
					qsort(merged, mergedCount, sizeof(PgturbohybridResult),
						  PgturbohybridScoreCompare);
				for (int i = 0; i < mergedCount && outIndex < outCount; i++)
				{
					if (PgturbohybridFinalDiversityNodeSelected(selectedCandidates,
																selectedCount,
																merged[i].nodeId))
						continue;
					out[outIndex++] = merged[i];
				}
			}
			*finalResults = out;

			stats->finalDiversity.selected = selectedCount;
			stats->finalDiversity.duplicateGroupsSuppressed =
				beforeDuplicates > afterDuplicates ?
				(uint64) (beforeDuplicates - afterDuplicates) : 0;
			applied = true;
		}
diversity_done:
		;
	}
	PG_CATCH();
	{
		UnlockPage(scan->indexRelation, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ShareLock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	UnlockPage(scan->indexRelation, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ShareLock);

	stats->finalDiversity.us = PgturbohybridElapsedUs(start);
	return applied;
}

static void
PgturbohybridAddDenseCandidate(PgturbohybridResult *item,
							   const TqDenseCandidate *candidate)
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
PgturbohybridAddMultiVectorCandidate(PgturbohybridResult *item,
									 const TqDenseCandidate *candidate)
{
	item->nodeId = candidate->nodeId;
	item->heaptid = candidate->heaptid;
	item->multivectorDistance = candidate->distance;
	item->multivectorSimilarity = candidate->similarity;
	item->multivectorRank = candidate->rank;
	item->hasMultivector = true;
	item->exactScored = candidate->exactScored;
}

static void
PgturbohybridAddBm25Candidate(PgturbohybridResult *item,
							  const PgturbohybridBm25Result *candidate)
{
	item->nodeId = candidate->nodeId;
	item->heaptid = candidate->heaptid;
	item->bm25Score = candidate->bm25Score;
	item->bm25Rank = candidate->rank;
	item->hasBm25 = true;
}

static void
PgturbohybridMergeBm25Candidate(PgturbohybridResult *item,
								const PgturbohybridBm25Result *candidate)
{
	if (!item->hasDense && !item->hasMultivector)
		item->heaptid = candidate->heaptid;
	item->nodeId = candidate->nodeId;
	item->bm25Score = candidate->bm25Score;
	item->bm25Rank = candidate->rank;
	item->hasBm25 = true;
}

static void
PgturbohybridMergeBm25ResultItem(PgturbohybridResult *item,
								 const PgturbohybridResult *bm25Item)
{
	if (!item->hasDense && !item->hasMultivector)
		item->heaptid = bm25Item->heaptid;
	item->hasBm25 = true;
	item->bm25Score = bm25Item->bm25Score;
	item->bm25Rank = bm25Item->bm25Rank;
}

static void
PgturbohybridAddSparseCandidate(PgturbohybridResult *item,
								const PgturbohybridSparseCandidate *candidate,
								int rank)
{
	item->nodeId = candidate->nodeId;
	item->heaptid = candidate->heaptid;
	item->sparseSimilarity = candidate->score;
	item->sparseRank = rank;
	item->hasSparse = true;
}

static void
PgturbohybridMergeSparseResultItem(PgturbohybridResult *item,
								   const PgturbohybridResult *sparseItem)
{
	if (!item->hasDense && !item->hasMultivector && !item->hasBm25)
		item->heaptid = sparseItem->heaptid;
	item->hasSparse = true;
	item->sparseSimilarity = sparseItem->sparseSimilarity;
	item->sparseRank = sparseItem->sparseRank;
}

static void
PgturbohybridRecordFusionClass(PgturbohybridLastScanStats *stats,
							   const PgturbohybridResult *item)
{
	bool		hasDenseLike = item->hasDense || item->hasMultivector;

	if (hasDenseLike && item->hasBm25)
		stats->bothMatch++;
	else if (hasDenseLike)
		stats->dense.only++;
	else if (item->hasBm25)
		stats->bm25.only++;
}

static double
PgturbohybridRrfScore(PgturbohybridQueryHeader *query,
					  const PgturbohybridResult *item)
{
	return query->denseWeight *
		(item->hasDense ? 1.0 / ((double) query->rrfK + item->denseRank) : 0.0) +
		query->multivectorWeight *
		(item->hasMultivector ? 1.0 / ((double) query->rrfK + item->multivectorRank) : 0.0) +
		query->bm25Weight *
		(item->hasBm25 ? 1.0 / ((double) query->rrfK + item->bm25Rank) : 0.0) +
		query->sparseWeight *
		(item->hasSparse ? 1.0 / ((double) query->rrfK + item->sparseRank) : 0.0);
}

static double
PgturbohybridNormalize(double value, double minValue, double maxValue)
{
	if (maxValue <= minValue)
		return value > 0 ? 1.0 : 0.0;
	return (value - minValue) / (maxValue - minValue);
}

static bool
PgturbohybridResultHasDenseLike(const PgturbohybridResult *item)
{
	return item->hasDense || item->hasMultivector;
}

static double
PgturbohybridResultDenseLikeSimilarity(const PgturbohybridResult *item)
{
	return item->hasDense ? item->denseSimilarity :
		item->multivectorSimilarity;
}

static double
PgturbohybridResultDenseLikeWeight(PgturbohybridQueryHeader *query,
								   const PgturbohybridResult *item)
{
	return item->hasMultivector && !item->hasDense ?
		query->multivectorWeight : query->denseWeight;
}

static double
PgturbohybridFastWeightedDenseNormalize(double similarity)
{
	double		expValue;

	if (similarity >= 0.0)
		return 1.0 / (1.0 + exp(-similarity));

	expValue = exp(similarity);
	return expValue / (1.0 + expValue);
}

static double
PgturbohybridFastWeightedDenseContribution(double alpha, double similarity)
{
	return alpha * PgturbohybridFastWeightedDenseNormalize(similarity);
}

static double
PgturbohybridClampUnit(double value)
{
	if (isnan(value) || isinf(value))
		return 0.5;
	if (value < 0.0)
		return 0.0;
	if (value > 1.0)
		return 1.0;
	return value;
}

typedef struct PgturbohybridDbsfBranchStats
{
	uint32		count;
	double		mean;
	double		stddev;
	double		min;
	double		max;
	bool		degenerate;
} PgturbohybridDbsfBranchStats;

static int
PgturbohybridDoubleAscendingCompare(const void *a, const void *b)
{
	double		da = *((const double *) a);
	double		db = *((const double *) b);

	if (da < db)
		return -1;
	if (da > db)
		return 1;
	return 0;
}

static double
PgturbohybridMedianSorted(const double *values, int count)
{
	int			mid;

	if (count <= 0)
		return 0.0;
	mid = count / 2;
	if ((count % 2) != 0)
		return values[mid];
	return (values[mid - 1] + values[mid]) * 0.5;
}

static void
PgturbohybridDbsfCollectBranch(const PgturbohybridResult *results, int count,
							   bool dense,
							   PgturbohybridDbsfBranchStats *branch)
{
	double	   *values;
	double		sum = 0.0;
	double		sumSq = 0.0;
	int			valueCount = 0;
	int			minBranchCandidates;

	memset(branch, 0, sizeof(*branch));
	branch->min = 0.0;
	branch->max = 0.0;
	if (results == NULL || count <= 0)
	{
		branch->degenerate = true;
		return;
	}

	values = palloc(sizeof(double) * count);
	for (int i = 0; i < count; i++)
	{
		double		score;

		if (dense)
		{
			if (!PgturbohybridResultHasDenseLike(&results[i]))
				continue;
			score = PgturbohybridResultDenseLikeSimilarity(&results[i]);
		}
		else
		{
			if (!results[i].hasBm25)
				continue;
			score = results[i].bm25Score;
		}
		if (isnan(score) || isinf(score))
			continue;
		values[valueCount++] = score;
		sum += score;
		if (valueCount == 1)
		{
			branch->min = score;
			branch->max = score;
		}
		else
		{
			branch->min = Min(branch->min, score);
			branch->max = Max(branch->max, score);
		}
	}

	branch->count = (uint32) valueCount;
	if (valueCount <= 0)
	{
		pfree(values);
		branch->degenerate = true;
		return;
	}

	branch->mean = sum / (double) valueCount;
	if (valueCount > 1)
	{
		for (int i = 0; i < valueCount; i++)
		{
			double		delta = values[i] - branch->mean;

			sumSq += delta * delta;
		}
		branch->stddev = sqrt(sumSq / (double) (valueCount - 1));
	}

	if (pgturbohybrid_dbsf_robust == PGTURBOHYBRID_DBSF_ROBUST_MAD)
	{
		double		median;

		qsort(values, valueCount, sizeof(double),
			  PgturbohybridDoubleAscendingCompare);
		median = PgturbohybridMedianSorted(values, valueCount);
		for (int i = 0; i < valueCount; i++)
			values[i] = fabs(values[i] - median);
		qsort(values, valueCount, sizeof(double),
			  PgturbohybridDoubleAscendingCompare);
		branch->mean = median;
		branch->stddev = 1.4826 *
			PgturbohybridMedianSorted(values, valueCount);
	}

	pfree(values);

	minBranchCandidates = Max(pgturbohybrid_dbsf_min_branch_candidates, 1);
	branch->degenerate =
		valueCount < minBranchCandidates ||
		branch->stddev <= 0.0 ||
		isnan(branch->stddev) ||
		isinf(branch->stddev);
}

static double
PgturbohybridDbsfNormalize(double value,
						   const PgturbohybridDbsfBranchStats *branch)
{
	double		sigma;
	double		lo;
	double		hi;

	if (branch == NULL || branch->degenerate)
		return 0.5;
	sigma = pgturbohybrid_dbsf_sigma;
	if (isnan(sigma) || isinf(sigma) || sigma <= 0.0)
		sigma = 3.0;
	lo = branch->mean - sigma * branch->stddev;
	hi = branch->mean + sigma * branch->stddev;
	if (hi <= lo)
		return 0.5;
	return PgturbohybridClampUnit((value - lo) / (hi - lo));
}

static double
PgturbohybridCalibratedFusionAlpha(PgturbohybridQueryHeader *query,
								   int queryShape)
{
	if ((query->flags & PGTURBOHYBRID_QUERY_FLAG_ALPHA_IS_SET) != 0)
		return PgturbohybridClampUnit(query->alpha);

	switch ((PgturbohybridHybridAdaptiveShape) queryShape)
	{
		case PGTURBOHYBRID_HYBRID_SHAPE_RARE_IDENTIFIER:
			return PgturbohybridClampUnit(
				pgturbohybrid_calibrated_fusion_identifier_bm25_alpha);
		case PGTURBOHYBRID_HYBRID_SHAPE_BROAD_NATURAL_LANGUAGE:
			return PgturbohybridClampUnit(
				pgturbohybrid_calibrated_fusion_broad_dense_alpha);
		case PGTURBOHYBRID_HYBRID_SHAPE_MIXED:
		case PGTURBOHYBRID_HYBRID_SHAPE_FIXED:
		case PGTURBOHYBRID_HYBRID_SHAPE_NOT_HYBRID:
		default:
			return PgturbohybridClampUnit(
				pgturbohybrid_calibrated_fusion_default_alpha);
	}
}

static int
PgturbohybridFusedScoreBoundDenseEntryCompare(const void *a, const void *b)
{
	const PgturbohybridBm25FusedScoreBoundDenseEntry *da =
		(const PgturbohybridBm25FusedScoreBoundDenseEntry *) a;
	const PgturbohybridBm25FusedScoreBoundDenseEntry *db =
		(const PgturbohybridBm25FusedScoreBoundDenseEntry *) b;

	return (da->nodeId > db->nodeId) - (da->nodeId < db->nodeId);
}

static int
PgturbohybridDoubleDescendingCompare(const void *a, const void *b)
{
	double		da = *((const double *) a);
	double		db = *((const double *) b);

	if (da > db)
		return -1;
	if (da < db)
		return 1;
	return 0;
}

static void
PgturbohybridPrepareFastWeightedBound(PgturbohybridQueryHeader *query,
									  TqDenseCandidate *dense,
									  int denseCount,
									  int limit,
									  MemoryContext memoryContext,
									  PgturbohybridBm25FusedScoreBoundContext *bound)
{
	double		alpha;
	int			finalK;
	double	   *contributions;

	memset(bound, 0, sizeof(*bound));

	if (!pgturbohybrid_fast_weighted_score_bound_pruning ||
		query->fusion != PGTURBOHYBRID_FUSION_FAST_WEIGHTED ||
		(query->flags & PGTURBOHYBRID_QUERY_FLAG_REQUIRE_BM25_MATCH) != 0 ||
		pgturbohybrid_enable_exact_rescore_for_bm25_only)
		return;

	finalK = PgturbohybridEffectiveFinalK(query, limit);
	if (finalK <= 0 || denseCount < finalK)
		return;

	alpha = (query->flags & PGTURBOHYBRID_QUERY_FLAG_ALPHA_IS_SET) != 0 ?
		query->alpha : 0.5;
	bound->dense =
		MemoryContextAlloc(memoryContext,
						   sizeof(PgturbohybridBm25FusedScoreBoundDenseEntry) *
						   denseCount);
	contributions =
		MemoryContextAlloc(memoryContext, sizeof(double) * denseCount);

	for (int i = 0; i < denseCount; i++)
	{
		double		contribution =
			PgturbohybridFastWeightedDenseContribution(alpha,
													   dense[i].similarity);

		((PgturbohybridBm25FusedScoreBoundDenseEntry *) bound->dense)[i].nodeId =
			dense[i].nodeId;
		((PgturbohybridBm25FusedScoreBoundDenseEntry *) bound->dense)[i].contribution =
			contribution;
		contributions[i] = contribution;
		bound->maxDenseContribution =
			i == 0 ? contribution : Max(bound->maxDenseContribution, contribution);
	}

	qsort((void *) bound->dense, denseCount,
		  sizeof(PgturbohybridBm25FusedScoreBoundDenseEntry),
		  PgturbohybridFusedScoreBoundDenseEntryCompare);
	qsort(contributions, denseCount, sizeof(double),
		  PgturbohybridDoubleDescendingCompare);

	bound->enabled = true;
	bound->alpha = alpha;
	bound->denseCount = (uint32) denseCount;
	bound->kthScore = contributions[finalK - 1];
}

/*
 * Give BM25-only candidates a dense score so fusion can rank them on both
 * signals.
 *
 * A candidate that matched the lexical (BM25) branch but fell outside the dense
 * graph candidate set has no dense rank, so by default it contributes nothing to
 * the dense side of fusion. When the active profile enables exact rescoring for
 * BM25-only candidates, score each such candidate's stored quantized code
 * against the already-prepared query (so->tq) and slot it into the dense ranking
 * by its distance. This recovers true dense neighbours that the approximate
 * graph search missed (an ANN recall miss) and stops BM25-only candidates from
 * aborting quality/debug-profile scans.
 *
 * The distance comes from the same quantized scorer that produced the dense
 * candidates' distances, so the values are on one comparable scale for
 * exact-free indexes (the common case). The dense rank is assigned by insertion
 * against the original dense candidate distances, leaving existing dense ranks
 * untouched.
 */
static void
PgturbohybridApplyBm25OnlyExactRescore(IndexScanDesc scan,
								  PgturbohybridScanState *state,
								  PgturbohybridResult *results, int count)
{
	PgturbohybridGraphScanOpaque so = (PgturbohybridGraphScanOpaque) scan->opaque;
	PgturbohybridGraphMetaPageData meta;
	PgturbohybridGraphScanStorage storage;
	double	   *denseDistances;
	int			denseDistanceCount = 0;
	bool		haveBm25Only = false;
	instr_time	lockStart;

	if (!pgturbohybrid_enable_exact_rescore_for_bm25_only)
		return;
	if ((state->query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_VECTOR) == 0)
		return;
	if (so == NULL || !so->tq.enabled)
		return;

	for (int i = 0; i < count; i++)
	{
		if (results[i].hasBm25 && !results[i].hasDense &&
			!results[i].hasMultivector)
		{
			haveBm25Only = true;
			break;
		}
	}
	if (!haveBm25Only)
		return;

	if (!PgturbohybridGraphReadMeta(scan->indexRelation, &meta) ||
		meta.tqNodeCount == 0 ||
		!BlockNumberIsValid(meta.tqCodeStartBlkno))
		return;

	/* Snapshot the dense candidates' distances for insertion ranking. */
	denseDistances = palloc(sizeof(double) * Max(count, 1));
	for (int i = 0; i < count; i++)
	{
		if (PgturbohybridResultHasDenseLike(&results[i]))
			denseDistances[denseDistanceCount++] =
				results[i].hasDense ? results[i].denseDistance :
				results[i].multivectorDistance;
	}

	INSTR_TIME_SET_CURRENT(lockStart);
	LockPage(scan->indexRelation, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ShareLock);
	so->graphScanLockWaitUs += PgturbohybridElapsedUs(lockStart);
	PG_TRY();
	{
		PgturbohybridGraphInitScanStorage(scan->indexRelation, &meta, &storage, NULL);

		for (int i = 0; i < count; i++)
		{
			PgturbohybridGraphScanNode *node;
			double		distance;
			int32		denseRank = 1;

			if (!(results[i].hasBm25 && !results[i].hasDense &&
				  !results[i].hasMultivector))
				continue;
			if (results[i].nodeId >= meta.tqNodeCount)
				continue;
			if (!PgturbohybridGraphLoadCodePage(scan->indexRelation, so, &meta,
												&storage, results[i].nodeId))
				continue;

			node = &storage.nodes[results[i].nodeId];
			if (node->code == NULL)
				continue;

			distance = PgturbohybridGraphScoreNode(so, node);

			for (int j = 0; j < denseDistanceCount; j++)
			{
				if (denseDistances[j] < distance)
					denseRank++;
			}

			results[i].denseDistance = distance;
			results[i].denseSimilarity = -distance;
			results[i].denseRank = denseRank;
			results[i].hasDense = true;
			results[i].exactScored = false;
		}
	}
	PG_CATCH();
	{
		UnlockPage(scan->indexRelation, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ShareLock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	UnlockPage(scan->indexRelation, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ShareLock);
}

/*
 * Future score-level multivector fusion needs exact dense scores for documents
 * that entered only through BM25.  Keep this helper separate from the vector
 * path because multivector dense scores must be exact heap MaxSim over the
 * original float32 token vectors, not a quantized graph-node score.
 *
	 * Weighted multivector hybrid fusion uses this for BM25-only documents.
	 * RRF normally does not need a BM25-only dense contribution.
	 */
static void
PgturbohybridApplyBm25OnlyMultiVectorExactRescore(IndexScanDesc scan,
												  PgturbohybridScanState *state,
												  PgturbohybridResult *results,
												  int count)
{
	Relation	heap;
	TupleDesc	desc;
	TupleTableSlot *slot;
	AttrNumber	denseAttno;
	PgturbohybridMultiVector *queryMv;
	const float4 *queryWeights;
	const bool *queryMask;
	double		queryWeightSum;
	double	   *denseDistances;
	int			denseDistanceCount = 0;
	bool		haveBm25Only = false;

	if (!pgturbohybrid_enable_exact_rescore_for_bm25_only)
		return;
	if (scan == NULL || scan->heapRelation == NULL ||
		scan->indexRelation == NULL || scan->indexRelation->rd_index == NULL ||
		state == NULL || state->query == NULL || results == NULL ||
		!PgturbohybridQueryHasMultiVector(state->query) ||
		!PgturbohybridQueryHasText(state->query) ||
		!PgturbohybridIndexIsMultiVector(scan->indexRelation))
		return;

	for (int i = 0; i < count; i++)
	{
		if (results[i].hasBm25 && !results[i].hasDense &&
			!results[i].hasMultivector)
		{
			haveBm25Only = true;
			break;
		}
	}
	if (!haveBm25Only)
		return;

	queryMv = PgturbohybridQueryGetMultiVector(state->query);
	if (queryMv == NULL || queryMv->count <= 0)
		return;
	queryWeights = PgturbohybridQueryGetTokenWeights(state->query);
	queryMask = PgturbohybridQueryGetTokenMask(state->query);
	queryWeightSum = PgturbohybridQueryMultiVectorWeightSum(state->query);

	denseAttno =
		scan->indexRelation->rd_index->indkey.values[PGTURBOHYBRID_DENSE_KEY_INDEX];
	heap = scan->heapRelation;
	desc = RelationGetDescr(heap);
	if (denseAttno <= 0 || denseAttno > desc->natts)
		return;

	denseDistances = palloc(sizeof(double) * Max(count, 1));
	for (int i = 0; i < count; i++)
	{
		if (results[i].hasDense)
			denseDistances[denseDistanceCount++] = results[i].denseDistance;
	}

	slot = table_slot_create(heap, NULL);
	for (int i = 0; i < count; i++)
	{
		Datum		value;
		bool		isnull;
		bool		visible;
		char	   *valuePtr;
		PgturbohybridMultiVector *docMv;
		double		maxsim;
		double		distance;
		int32		denseRank = 1;

		CHECK_FOR_INTERRUPTS();
		if (!(results[i].hasBm25 && !results[i].hasDense &&
			  !results[i].hasMultivector))
			continue;

		visible = table_tuple_fetch_row_version(heap, &results[i].heaptid,
												scan->xs_snapshot, slot);
		if (!visible)
		{
			ExecClearTuple(slot);
			continue;
		}

		value = slot_getattr(slot, denseAttno, &isnull);
		if (isnull)
		{
			ExecClearTuple(slot);
			continue;
		}

		valuePtr = DatumGetPointer(value);
		docMv = PgturbohybridDatumGetMultiVector(value);
		maxsim = TqMultiVectorMaxSimWeighted(queryMv, docMv,
											 queryWeights, queryMask);
		distance = -maxsim;

		for (int j = 0; j < denseDistanceCount; j++)
		{
			if (denseDistances[j] < distance)
				denseRank++;
		}

		results[i].multivectorDistance = distance;
		results[i].multivectorSimilarity =
			queryWeightSum > 0.0 ? maxsim / queryWeightSum : 0.0;
		results[i].multivectorRank = denseRank;
		results[i].hasMultivector = true;
		results[i].exactScored = true;

		if ((char *) docMv != valuePtr)
			pfree(docMv);
		ExecClearTuple(slot);
	}
	ExecDropSingleTupleTableSlot(slot);
	pfree(denseDistances);
}

static bool
PgturbohybridTSVectorHasPositions(TSVector vector)
{
	WordEntry  *entries;

	if (vector == NULL)
		return false;
	entries = ARRPTR(vector);
	for (int i = 0; i < vector->size; i++)
	{
		if (POSDATALEN(vector, &entries[i]) > 0)
			return true;
	}
	return false;
}

static int
PgturbohybridBm25ResultCompare(const void *a, const void *b)
{
	const PgturbohybridBm25Result *ra = (const PgturbohybridBm25Result *) a;
	const PgturbohybridBm25Result *rb = (const PgturbohybridBm25Result *) b;

	if (ra->bm25Score > rb->bm25Score)
		return -1;
	if (ra->bm25Score < rb->bm25Score)
		return 1;
	return (ra->nodeId > rb->nodeId) - (ra->nodeId < rb->nodeId);
}

typedef struct PgturbohybridMultiVectorBm25InjectionCandidate
{
	TqDenseCandidate candidate;
	bool		injected;
} PgturbohybridMultiVectorBm25InjectionCandidate;

static int
PgturbohybridMultiVectorBm25InjectionCompare(const void *a, const void *b)
{
	const PgturbohybridMultiVectorBm25InjectionCandidate *ra =
		(const PgturbohybridMultiVectorBm25InjectionCandidate *) a;
	const PgturbohybridMultiVectorBm25InjectionCandidate *rb =
		(const PgturbohybridMultiVectorBm25InjectionCandidate *) b;
	int			cmp;

	if (ra->candidate.distance < rb->candidate.distance)
		return -1;
	if (ra->candidate.distance > rb->candidate.distance)
		return 1;
	cmp = PgturbohybridItemPointerDataCompare(&ra->candidate.heaptid,
											  &rb->candidate.heaptid);
	if (cmp != 0)
		return cmp;
	return (ra->candidate.nodeId > rb->candidate.nodeId) -
		(ra->candidate.nodeId < rb->candidate.nodeId);
}

static bool
PgturbohybridDenseCandidatesContainHeapTid(const TqDenseCandidate *dense,
										   int denseCount,
										   const ItemPointerData *heaptid)
{
	for (int i = 0; i < denseCount; i++)
	{
		if (PgturbohybridItemPointerDataCompare(&dense[i].heaptid,
												heaptid) == 0)
			return true;
	}
	return false;
}

static bool
PgturbohybridInjectionPoolContainsHeapTid(
	PgturbohybridMultiVectorBm25InjectionCandidate *pool,
	int poolCount, const ItemPointerData *heaptid)
{
	for (int i = 0; i < poolCount; i++)
	{
		if (PgturbohybridItemPointerDataCompare(&pool[i].candidate.heaptid,
												heaptid) == 0)
			return true;
	}
	return false;
}

static bool
PgturbohybridShouldInjectMultiVectorBm25Candidates(PgturbohybridQueryHeader *query)
{
	if (pgturbohybrid_multivector_sparse_candidate_source ==
		PGTURBOHYBRID_MULTIVECTOR_SPARSE_CANDIDATE_SOURCE_BM25 ||
		pgturbohybrid_multivector_sparse_candidate_source ==
		PGTURBOHYBRID_MULTIVECTOR_SPARSE_CANDIDATE_SOURCE_LEARNED_SPARSE)
		return true;

	switch ((PgturbohybridMultiVectorBm25CandidateInjectionMode)
			pgturbohybrid_multivector_bm25_candidate_injection)
	{
		case PGTURBOHYBRID_MULTIVECTOR_BM25_CANDIDATE_INJECTION_DENSE_WITH_TEXT:
			return true;
		case PGTURBOHYBRID_MULTIVECTOR_BM25_CANDIDATE_INJECTION_HYBRID_ONLY:
			return query != NULL && query->bm25Weight > 0.0;
		case PGTURBOHYBRID_MULTIVECTOR_BM25_CANDIDATE_INJECTION_OFF:
		default:
			return false;
	}
}

static bool
PgturbohybridUsingLearnedSparseCandidateInjection(void)
{
	return pgturbohybrid_multivector_sparse_candidate_source ==
		PGTURBOHYBRID_MULTIVECTOR_SPARSE_CANDIDATE_SOURCE_LEARNED_SPARSE;
}

static void
PgturbohybridInjectMultiVectorBm25Candidates(IndexScanDesc scan,
											 PgturbohybridQueryHeader *query,
											 const PgturbohybridBm25Result *bm25,
											 int bm25Count,
											 TqDenseCandidate **dense,
											 int *denseCount,
											 int autoBudgetLimit,
											 MemoryContext memoryContext,
											 TqDenseCandidateStats *denseStats)
{
	Relation	heap;
	TupleDesc	desc;
	TupleTableSlot *slot;
	AttrNumber	denseAttno;
	PgturbohybridMultiVector *queryMv;
	const float4 *queryWeights;
	const bool *queryMask;
	double		queryWeightSum;
	PgturbohybridMultiVectorBm25InjectionCandidate *pool;
	TqDenseCandidate *newDense;
	int			poolCount;
	int			candidateLimit;
	int			newDenseCount;
	uint32		exactReranked = 0;
	uint32		retained = 0;
	uint32		learnedSparseRetained = 0;
	uint64		exactPairs = 0;
	bool		learnedSparse =
		PgturbohybridUsingLearnedSparseCandidateInjection();

	if (denseStats == NULL)
		return;
	denseStats->multivector.bm25InjectionEnabled = false;
	denseStats->multivector.bm25InjectionCandidates = 0;
	denseStats->multivector.bm25InjectionCandidateLimit = 0;
	denseStats->multivector.bm25InjectionPoolSize = 0;
	strlcpy(denseStats->multivector.bm25InjectionLimitReason, "not_applicable",
			sizeof(denseStats->multivector.bm25InjectionLimitReason));
	denseStats->multivector.bm25InjectionRetained = 0;
	denseStats->multivector.bm25InjectionExactReranked = 0;
	denseStats->learnedSparse.candidates = 0;
	denseStats->learnedSparse.retainedForMaxsim = 0;
	denseStats->learnedSparse.branchLatencyUs = 0;

	if (scan == NULL || scan->heapRelation == NULL ||
		scan->indexRelation == NULL || scan->indexRelation->rd_index == NULL ||
		query == NULL || bm25 == NULL || bm25Count <= 0 ||
		dense == NULL || denseCount == NULL ||
		!PgturbohybridShouldInjectMultiVectorBm25Candidates(query) ||
		!PgturbohybridQueryHasMultiVector(query) ||
		!PgturbohybridQueryHasText(query) ||
		!PgturbohybridIndexIsMultiVector(scan->indexRelation))
		return;

	queryMv = PgturbohybridQueryGetMultiVector(query);
	if (queryMv == NULL || queryMv->count <= 0)
		return;
	queryWeights = PgturbohybridQueryGetTokenWeights(query);
	queryMask = PgturbohybridQueryGetTokenMask(query);
	queryWeightSum = PgturbohybridQueryMultiVectorWeightSum(query);

	denseAttno =
		scan->indexRelation->rd_index->indkey.values[PGTURBOHYBRID_DENSE_KEY_INDEX];
	heap = scan->heapRelation;
	desc = RelationGetDescr(heap);
	if (denseAttno <= 0 || denseAttno > desc->natts)
		return;

	denseStats->multivector.bm25InjectionEnabled = true;
	denseStats->multivector.bm25InjectionCandidates = (uint32) bm25Count;
	if (learnedSparse)
		denseStats->learnedSparse.candidates = (uint32) bm25Count;

	candidateLimit = PgturbohybridBudgetFinalTarget(query, autoBudgetLimit);
	if (*denseCount >= candidateLimit)
	{
		candidateLimit = *denseCount;
		strlcpy(denseStats->multivector.bm25InjectionLimitReason, "dense_count",
				sizeof(denseStats->multivector.bm25InjectionLimitReason));
	}
	else
		strlcpy(denseStats->multivector.bm25InjectionLimitReason,
				"final_target",
				sizeof(denseStats->multivector.bm25InjectionLimitReason));
	if (candidateLimit <= 0)
	{
		candidateLimit = bm25Count;
		strlcpy(denseStats->multivector.bm25InjectionLimitReason, "bm25_count",
				sizeof(denseStats->multivector.bm25InjectionLimitReason));
	}
	if (pgturbohybrid_multivector_doc_candidate_k > 0)
	{
		if (candidateLimit > pgturbohybrid_multivector_doc_candidate_k)
		{
			candidateLimit = pgturbohybrid_multivector_doc_candidate_k;
			strlcpy(denseStats->multivector.bm25InjectionLimitReason,
					"doc_candidate_k",
					sizeof(denseStats->multivector.bm25InjectionLimitReason));
		}
	}
	if (pgturbohybrid_multivector_exact_rerank_k > 0)
	{
		if (candidateLimit > pgturbohybrid_multivector_exact_rerank_k)
		{
			candidateLimit = pgturbohybrid_multivector_exact_rerank_k;
			strlcpy(denseStats->multivector.bm25InjectionLimitReason,
					"exact_rerank_k",
					sizeof(denseStats->multivector.bm25InjectionLimitReason));
		}
	}
	candidateLimit = Max(candidateLimit, 1);
	denseStats->multivector.bm25InjectionCandidateLimit = (uint32) candidateLimit;

	pool = MemoryContextAllocZero(memoryContext,
								  sizeof(*pool) *
								  Max(*denseCount + bm25Count, 1));
	poolCount = 0;
	for (int i = 0; i < *denseCount; i++)
	{
		pool[poolCount].candidate = (*dense)[i];
		pool[poolCount].injected = false;
		poolCount++;
	}

	slot = table_slot_create(heap, NULL);
	for (int i = 0; i < bm25Count; i++)
	{
		Datum		value;
		bool		isnull;
		bool		visible;
		char	   *valuePtr;
		PgturbohybridMultiVector *docMv;
		double		maxsim;
		ItemPointerData heapTid;

		CHECK_FOR_INTERRUPTS();
		if (PgturbohybridDenseCandidatesContainHeapTid(*dense, *denseCount,
													   &bm25[i].heaptid) ||
			PgturbohybridInjectionPoolContainsHeapTid(pool, poolCount,
													  &bm25[i].heaptid))
			continue;

		heapTid = bm25[i].heaptid;
		visible = table_tuple_fetch_row_version(heap, &heapTid,
												scan->xs_snapshot, slot);
		if (!visible)
		{
			ExecClearTuple(slot);
			continue;
		}

		value = slot_getattr(slot, denseAttno, &isnull);
		if (isnull)
		{
			ExecClearTuple(slot);
			continue;
		}

		valuePtr = DatumGetPointer(value);
		docMv = PgturbohybridDatumGetMultiVector(value);
		maxsim = TqMultiVectorMaxSimWeighted(queryMv, docMv,
											 queryWeights, queryMask);
		pool[poolCount].candidate.nodeId = bm25[i].nodeId;
		pool[poolCount].candidate.heaptid = bm25[i].heaptid;
		pool[poolCount].candidate.distance = -maxsim;
		pool[poolCount].candidate.similarity =
			queryWeightSum > 0.0 ? maxsim / queryWeightSum : 0.0;
		pool[poolCount].candidate.rank = 0;
		pool[poolCount].candidate.exactScored = true;
		pool[poolCount].injected = true;
		poolCount++;
		exactReranked++;
		exactPairs += (uint64) queryMv->count * (uint64) docMv->count;

		if ((char *) docMv != valuePtr)
			pfree(docMv);
		ExecClearTuple(slot);
	}
	ExecDropSingleTupleTableSlot(slot);

	if (poolCount > 1)
		qsort(pool, poolCount, sizeof(*pool),
			  PgturbohybridMultiVectorBm25InjectionCompare);
	denseStats->multivector.bm25InjectionPoolSize = (uint32) poolCount;

	newDenseCount = Min(poolCount, candidateLimit);
	newDense = MemoryContextAllocZero(memoryContext,
									  sizeof(*newDense) *
									  Max(newDenseCount, 1));
	for (int i = 0; i < newDenseCount; i++)
	{
		newDense[i] = pool[i].candidate;
		newDense[i].rank = i + 1;
		if (pool[i].injected)
			retained++;
	}
	if (learnedSparse)
	{
		for (int i = 0; i < bm25Count; i++)
		{
			if (PgturbohybridDenseCandidatesContainHeapTid(newDense,
														  newDenseCount,
														  &bm25[i].heaptid))
				learnedSparseRetained++;
		}
	}

	*dense = newDense;
	*denseCount = newDenseCount;
	denseStats->dense.candidatesReturned = (uint32) newDenseCount;
	denseStats->multivector.docCandidates = (uint32) newDenseCount;
	denseStats->multivector.exactRerankDocs += exactReranked;
	denseStats->multivector.exactRerankPairs += exactPairs;
	denseStats->exactRerankCandidates += exactReranked;
	denseStats->exactRerankTokensEvaluated +=
		(uint64) exactReranked * (uint64) queryMv->count;
	denseStats->multivector.bm25InjectionRetained = retained;
	denseStats->multivector.bm25InjectionExactReranked = exactReranked;
	if (learnedSparse)
		denseStats->learnedSparse.retainedForMaxsim = learnedSparseRetained;
}

static int
PgturbohybridBm25HeapTSVectorRerankLimit(PgturbohybridQueryHeader *query,
										 TSQuery tsquery, int count,
										 int autoBudgetLimit)
{
	int			finalTarget;
	int64		limit;

	if (count <= 0 ||
		pgturbohybrid_bm25_heap_tsvector_rerank ==
		PGTURBOHYBRID_BM25_HEAP_TSVECTOR_RERANK_OFF)
		return 0;
	if (tsquery == NULL)
		return 0;

	if (pgturbohybrid_bm25_heap_tsvector_rerank ==
		PGTURBOHYBRID_BM25_HEAP_TSVECTOR_RERANK_AUTO &&
		!PgturbohybridBm25QueryHasPhrase(tsquery))
		return 0;

	finalTarget = PgturbohybridBudgetFinalTarget(query, autoBudgetLimit);
	if (finalTarget <= 0)
		finalTarget = PGTURBOHYBRID_DEFAULT_FINAL_K;

	if (pgturbohybrid_bm25_heap_tsvector_rerank ==
		PGTURBOHYBRID_BM25_HEAP_TSVECTOR_RERANK_TOPK)
		limit = finalTarget;
	else
		limit = (int64) finalTarget *
			Max(pgturbohybrid_bm25_heap_tsvector_rerank_multiplier, 1);

	return (int) Min((int64) count, limit);
}

static void
PgturbohybridBm25HeapTSVectorRerank(IndexScanDesc scan,
									PgturbohybridQueryHeader *query,
									PgturbohybridBm25Result *results,
									int count, int autoBudgetLimit,
									PgturbohybridBm25QueryStats *stats)
{
	TSQuery		tsquery;
	Relation	heap;
	TupleDesc	desc;
	TupleTableSlot *slot;
	AttrNumber	lexicalAttno;
	uint32	   *beforeTopK = NULL;
	int			beforeTopKCount = 0;
	int			limit;
	int			rescored = 0;
	instr_time	fetchStart;
	instr_time	scoreStart;

	if (stats != NULL)
		stats->heapTsvectorRerankMode =
			pgturbohybrid_bm25_heap_tsvector_rerank;

	if (scan == NULL || scan->heapRelation == NULL ||
		scan->indexRelation == NULL || scan->indexRelation->rd_index == NULL ||
		results == NULL || stats == NULL ||
		!PgturbohybridIndexHasLexical(scan->indexRelation))
		return;

	tsquery = PgturbohybridQueryGetTsQuery(query);
	limit = PgturbohybridBm25HeapTSVectorRerankLimit(query, tsquery, count,
													 autoBudgetLimit);
	if (limit <= 0)
		return;

	{
		int			bm25Key = PgturbohybridIndexBm25KeyAttno(scan->indexRelation);

		if (bm25Key < 0)
			return;
		lexicalAttno = scan->indexRelation->rd_index->indkey.values[bm25Key];
	}
	heap = scan->heapRelation;
	desc = RelationGetDescr(heap);
	if (lexicalAttno <= 0 || lexicalAttno > desc->natts)
		return;

	beforeTopKCount = Min(PgturbohybridBudgetFinalTarget(query, autoBudgetLimit),
						  count);
	if (beforeTopKCount > 0)
	{
		beforeTopK = palloc(sizeof(uint32) * beforeTopKCount);
		for (int i = 0; i < beforeTopKCount; i++)
			beforeTopK[i] = results[i].nodeId;
	}

	slot = table_slot_create(heap, NULL);
	for (int i = 0; i < limit; i++)
	{
		Datum		value;
		bool		isnull;
		bool		visible;
		char	   *valuePtr;
		TSVector	vector;
		bool		hasPositions;
		float4		rank;
		double		adjustment;

		CHECK_FOR_INTERRUPTS();

		INSTR_TIME_SET_CURRENT(fetchStart);
		visible = table_tuple_fetch_row_version(heap, &results[i].heaptid,
												scan->xs_snapshot, slot);
		stats->heapTsvectorRerankFetchUs += PgturbohybridElapsedUs(fetchStart);
		if (!visible)
		{
			ExecClearTuple(slot);
			continue;
		}

		value = slot_getattr(slot, lexicalAttno, &isnull);
		if (isnull)
		{
			ExecClearTuple(slot);
			continue;
		}

		valuePtr = DatumGetPointer(value);
		vector = DatumGetTSVector(value);
		hasPositions = PgturbohybridTSVectorHasPositions(vector);

		INSTR_TIME_SET_CURRENT(scoreStart);
		if (hasPositions)
			rank = DatumGetFloat4(DirectFunctionCall2(ts_rankcd_tt,
													  TSVectorGetDatum(vector),
													  TSQueryGetDatum(tsquery)));
		else
			rank = DatumGetFloat4(DirectFunctionCall2(ts_rank_tt,
													  TSVectorGetDatum(vector),
													  TSQueryGetDatum(tsquery)));
		stats->heapTsvectorRerankScoreUs += PgturbohybridElapsedUs(scoreStart);
		if (isfinite(rank) && rank > 0.0f)
		{
			adjustment = pgturbohybrid_bm25_heap_tsvector_rerank_weight *
				((double) rank / ((double) rank + 1.0));
			results[i].bm25Score += adjustment;
			rescored++;
		}
		if ((char *) vector != valuePtr)
			pfree(vector);
		ExecClearTuple(slot);
	}
	ExecDropSingleTupleTableSlot(slot);

	if (rescored <= 0)
	{
		if (beforeTopK != NULL)
			pfree(beforeTopK);
		return;
	}

	qsort(results, count, sizeof(PgturbohybridBm25Result),
		  PgturbohybridBm25ResultCompare);
	for (int i = 0; i < count; i++)
		results[i].rank = i + 1;

	stats->heapTsvectorRerankCount = rescored;
	if (beforeTopK != NULL)
	{
		for (int i = 0; i < beforeTopKCount; i++)
		{
			if (beforeTopK[i] != results[i].nodeId)
			{
				stats->heapTsvectorRerankTopKChanged = true;
				break;
			}
		}
		pfree(beforeTopK);
	}
}

static void
PgturbohybridScoreResults(PgturbohybridScanState *state, PgturbohybridResult *results,
						  int count, PgturbohybridLastScanStats *stats)
{
	PgturbohybridQueryHeader *query = state->query;
	double		minDense = get_float8_infinity();
	double		maxDense = -get_float8_infinity();
	double		minBm25 = get_float8_infinity();
	double		maxBm25 = -get_float8_infinity();
	double		alpha = (query->flags & PGTURBOHYBRID_QUERY_FLAG_ALPHA_IS_SET) != 0 ?
		query->alpha : 0.5;
	uint16		fusion = pgturbohybrid_force_fusion != 0 ?
		pgturbohybrid_force_fusion : query->fusion;
	PgturbohybridDbsfBranchStats dbsfDense;
	PgturbohybridDbsfBranchStats dbsfBm25;

	memset(&dbsfDense, 0, sizeof(dbsfDense));
	memset(&dbsfBm25, 0, sizeof(dbsfBm25));

	if (stats != NULL)
	{
		stats->calibratedFusion.enabled =
			fusion == PGTURBOHYBRID_FUSION_CALIBRATED;
		stats->dbsf.enabled = fusion == PGTURBOHYBRID_FUSION_DBSF;
		stats->dbsf.degenerateBranches = 0;
		memset(stats->dbsf.branchMean, 0, sizeof(stats->dbsf.branchMean));
		memset(stats->dbsf.branchStddev, 0, sizeof(stats->dbsf.branchStddev));
		memset(stats->dbsf.branchMin, 0, sizeof(stats->dbsf.branchMin));
		memset(stats->dbsf.branchMax, 0, sizeof(stats->dbsf.branchMax));
		stats->calibratedFusion.bothMatchBonus =
			stats->calibratedFusion.enabled ?
			PgturbohybridClampUnit(
				pgturbohybrid_calibrated_fusion_both_match_bonus) : 0.0;
		strlcpy(stats->calibratedFusion.denseNormMode,
				stats->calibratedFusion.enabled ? "logistic" : "none",
				sizeof(stats->calibratedFusion.denseNormMode));
		strlcpy(stats->calibratedFusion.bm25NormMode,
				stats->calibratedFusion.enabled ? "saturating" : "none",
				sizeof(stats->calibratedFusion.bm25NormMode));
		if (stats->calibratedFusion.enabled)
		{
			int			queryShape;

			if (stats->calibratedFusion.queryShape[0] == '\0')
				strlcpy(stats->calibratedFusion.queryShape, "mixed",
						sizeof(stats->calibratedFusion.queryShape));
			queryShape = PgturbohybridHybridShapeFromName(
				stats->calibratedFusion.queryShape);
			alpha = PgturbohybridCalibratedFusionAlpha(query, queryShape);
		}
		stats->calibratedFusion.alphaEffective =
			stats->calibratedFusion.enabled ? alpha : 0.0;
	}

	if (fusion == PGTURBOHYBRID_FUSION_WEIGHTED)
	{
		for (int i = 0; i < count; i++)
		{
			if (PgturbohybridResultHasDenseLike(&results[i]))
			{
				double		similarity =
					PgturbohybridResultDenseLikeSimilarity(&results[i]);

				minDense = Min(minDense, similarity);
				maxDense = Max(maxDense, similarity);
			}
			if (results[i].hasBm25)
			{
				minBm25 = Min(minBm25, results[i].bm25Score);
				maxBm25 = Max(maxBm25, results[i].bm25Score);
			}
		}
	}
	else if (fusion == PGTURBOHYBRID_FUSION_DBSF)
	{
		PgturbohybridDbsfCollectBranch(results, count, true, &dbsfDense);
		PgturbohybridDbsfCollectBranch(results, count, false, &dbsfBm25);
		if (stats != NULL)
		{
			stats->dbsf.branchMean[0] = dbsfDense.mean;
			stats->dbsf.branchMean[1] = dbsfBm25.mean;
			stats->dbsf.branchStddev[0] = dbsfDense.stddev;
			stats->dbsf.branchStddev[1] = dbsfBm25.stddev;
			stats->dbsf.branchMin[0] = dbsfDense.min;
			stats->dbsf.branchMin[1] = dbsfBm25.min;
			stats->dbsf.branchMax[0] = dbsfDense.max;
			stats->dbsf.branchMax[1] = dbsfBm25.max;
			stats->dbsf.degenerateBranches =
				(dbsfDense.degenerate ? 1 : 0) +
				(dbsfBm25.degenerate ? 1 : 0);
		}
	}

	for (int i = 0; i < count; i++)
	{
		if (fusion == PGTURBOHYBRID_FUSION_WEIGHTED)
		{
			double		denseNorm =
				PgturbohybridResultHasDenseLike(&results[i]) ?
				PgturbohybridNormalize(PgturbohybridResultDenseLikeSimilarity(&results[i]),
									   minDense, maxDense) : 0.0;
			double		bm25Norm = results[i].hasBm25 ?
				PgturbohybridNormalize(results[i].bm25Score, minBm25, maxBm25) : 0.0;

			results[i].fusedScore = alpha * denseNorm + (1.0 - alpha) * bm25Norm;
		}
		else if (fusion == PGTURBOHYBRID_FUSION_FAST_WEIGHTED)
		{
			double		denseNorm =
				PgturbohybridResultHasDenseLike(&results[i]) ?
				PgturbohybridFastWeightedDenseNormalize(
					PgturbohybridResultDenseLikeSimilarity(&results[i])) : 0.0;
			double		bm25Norm = results[i].hasBm25 ?
				PgturbohybridBm25NormalizeSaturating(results[i].bm25Score) : 0.0;

			results[i].fusedScore = alpha * denseNorm + (1.0 - alpha) * bm25Norm;
		}
		else if (fusion == PGTURBOHYBRID_FUSION_CALIBRATED)
		{
			double		denseNorm =
				PgturbohybridResultHasDenseLike(&results[i]) ?
				PgturbohybridFastWeightedDenseNormalize(
					PgturbohybridResultDenseLikeSimilarity(&results[i])) : 0.0;
			double		bm25Norm = results[i].hasBm25 ?
				PgturbohybridBm25NormalizeSaturating(results[i].bm25Score) : 0.0;
			double		bonus =
				(PgturbohybridResultHasDenseLike(&results[i]) && results[i].hasBm25) ?
				PgturbohybridClampUnit(
					pgturbohybrid_calibrated_fusion_both_match_bonus) : 0.0;

			results[i].fusedScore =
				alpha * denseNorm + (1.0 - alpha) * bm25Norm + bonus;
		}
		else if (fusion == PGTURBOHYBRID_FUSION_DBSF)
		{
			double		denseNorm =
				PgturbohybridResultHasDenseLike(&results[i]) ?
				PgturbohybridDbsfNormalize(
					PgturbohybridResultDenseLikeSimilarity(&results[i]),
					&dbsfDense) : 0.0;
			double		bm25Norm = results[i].hasBm25 ?
				PgturbohybridDbsfNormalize(results[i].bm25Score,
										   &dbsfBm25) : 0.0;

			results[i].fusedScore =
				PgturbohybridResultDenseLikeWeight(query, &results[i]) *
				denseNorm +
				query->bm25Weight * bm25Norm;
		}
		else
		{
			results[i].fusedScore =
				PgturbohybridRrfScore(query, &results[i]);
		}
	}
}

static int
PgturbohybridFinalizeFusedResults(IndexScanDesc scan,
								  PgturbohybridScanState *state,
								  PgturbohybridResult *merged,
								  int mergedCount,
								  int limit,
								  bool allowBm25OnlyExactRescore,
								  MemoryContext memoryContext,
								  PgturbohybridResult **finalResults,
								  uint64 *heapReplacements,
	PgturbohybridLastScanStats *stats)
{
	int			finalCount;

	if (allowBm25OnlyExactRescore)
	{
		uint16		fusion = pgturbohybrid_force_fusion != 0 ?
			pgturbohybrid_force_fusion : state->query->fusion;

		if (fusion != PGTURBOHYBRID_FUSION_RRF &&
			PgturbohybridQueryHasMultiVector(state->query) &&
			PgturbohybridQueryHasText(state->query) &&
			PgturbohybridIndexIsMultiVector(scan->indexRelation))
			PgturbohybridApplyBm25OnlyMultiVectorExactRescore(scan, state,
															  merged,
															  mergedCount);
		else
			PgturbohybridApplyBm25OnlyExactRescore(scan, state, merged,
												   mergedCount);
	}
	PgturbohybridScoreResults(state, merged, mergedCount, stats);

	finalCount = PgturbohybridFinalTarget(state->query, mergedCount, limit);
	if (PgturbohybridApplyFinalDiversity(scan, merged, mergedCount,
										 PgturbohybridBudgetFinalTarget(state->query,
																	   limit),
										 finalCount,
										 memoryContext, finalResults,
										 heapReplacements, stats))
		return finalCount;

	if (finalCount < mergedCount)
		finalCount = PgturbohybridSelectTopN(merged, mergedCount, finalCount,
											 finalResults, memoryContext,
											 heapReplacements);
	else
	{
		if (mergedCount > 1)
			qsort(merged, mergedCount, sizeof(PgturbohybridResult),
				  PgturbohybridScoreCompare);
		*finalResults = merged;
	}

	return finalCount;
}

static bool
PgturbohybridFusionNodeIdsFit(uint32 nodeCount,
							  const TqDenseCandidate *dense, int denseCount,
							  const PgturbohybridBm25Result *bm25, int bm25Count)
{
	for (int i = 0; i < denseCount; i++)
	{
		if (dense[i].nodeId >= nodeCount)
			return false;
	}
	for (int i = 0; i < bm25Count; i++)
	{
		if (bm25[i].nodeId >= nodeCount)
			return false;
	}
	return true;
}

static bool
PgturbohybridShouldUseGenerationArray(IndexScanDesc scan,
									  uint16 effectiveFusion,
									  uint32 fusionCandidatesSeen,
									  const TqDenseCandidate *dense,
									  int denseCount,
									  const PgturbohybridBm25Result *bm25,
									  int bm25Count,
									  uint32 *nodeCount)
{
	PgturbohybridGraphMetaPageData meta;
	Size		arrayBytes;

	if (effectiveFusion != PGTURBOHYBRID_FUSION_RRF ||
		pgturbohybrid_enable_exact_rescore_for_bm25_only ||
		fusionCandidatesSeen == 0 ||
		(pgturbohybrid_fusion_hash_threshold >= 0 &&
		 fusionCandidatesSeen < (uint32) pgturbohybrid_fusion_hash_threshold))
		return false;

	if (!PgturbohybridGraphReadMeta(scan->indexRelation, &meta) ||
		meta.tqNodeCount == 0)
		return false;

	arrayBytes = sizeof(PgturbohybridFusionArrayEntry) *
		(Size) meta.tqNodeCount;
	if (!AllocSizeIsValid(arrayBytes) ||
		arrayBytes > PGTURBOHYBRID_FUSION_GENERATION_ARRAY_MAX_BYTES)
		return false;
	if (!PgturbohybridFusionNodeIdsFit(meta.tqNodeCount, dense, denseCount,
									   bm25, bm25Count))
		return false;

	*nodeCount = meta.tqNodeCount;
	return true;
}

static PgturbohybridFusionArrayEntry *
PgturbohybridGetFusionGenerationArray(uint32 nodeCount,
									  uint32 *generation,
									  bool *reused,
									  bool *reset)
{
	PgturbohybridFusionGenerationArrayCache *cache =
		&pgturbohybrid_fusion_generation_array_cache;

	*reused = false;
	*reset = false;

	if (cache->entries == NULL || cache->capacity < nodeCount)
	{
		MemoryContext oldCtx;

		if (cache->entries != NULL)
			pfree(cache->entries);

		oldCtx = MemoryContextSwitchTo(CacheMemoryContext);
		cache->entries = MemoryContextAllocZero(CacheMemoryContext,
												sizeof(PgturbohybridFusionArrayEntry) *
												nodeCount);
		MemoryContextSwitchTo(oldCtx);
		cache->capacity = nodeCount;
		cache->generation = 1;
		*generation = cache->generation;
		*reset = true;
		return cache->entries;
	}

	*reused = true;
	if (cache->generation == PG_UINT32_MAX)
	{
		memset(cache->entries, 0,
			   sizeof(PgturbohybridFusionArrayEntry) * cache->capacity);
		cache->generation = 1;
		*reset = true;
	}
	else
		cache->generation++;

	*generation = cache->generation;
	return cache->entries;
}

static int
PgturbohybridFuseGenerationArray(IndexScanDesc scan,
								 PgturbohybridScanState *state,
								 const TqDenseCandidate *dense,
								 int denseCount,
								 const PgturbohybridBm25Result *bm25,
								 int bm25Count,
								 uint32 nodeCount,
								 int limit,
								 MemoryContext memoryContext,
								 PgturbohybridResult **finalResults,
								 int *mergedCountOut,
								 PgturbohybridLastScanStats *stats)
{
	uint32		generation;
	uint32		fusionCandidatesSeen = denseCount + bm25Count;
	PgturbohybridFusionArrayEntry *entries;
	uint32	   *touched;
	uint32		touchedCount = 0;
	PgturbohybridResult *merged;
	int			mergedCount = 0;

	entries = PgturbohybridGetFusionGenerationArray(nodeCount, &generation,
													&stats->fusionStats.generationArrayReused,
													&stats->fusionStats.generationArrayReset);
	touched = MemoryContextAlloc(memoryContext,
								 sizeof(uint32) * Max(fusionCandidatesSeen, 1));

	for (int i = 0; i < denseCount; i++)
	{
		PgturbohybridFusionArrayEntry *entry = &entries[dense[i].nodeId];

		if (entry->generation != generation)
		{
			entry->generation = generation;
			memset(&entry->result, 0, sizeof(entry->result));
			entry->result.nodeId = dense[i].nodeId;
			touched[touchedCount++] = dense[i].nodeId;
		}
		else
			stats->fusionStats.duplicates++;

		PgturbohybridAddDenseCandidate(&entry->result, &dense[i]);
	}

	for (int i = 0; i < bm25Count; i++)
	{
		PgturbohybridFusionArrayEntry *entry = &entries[bm25[i].nodeId];

		if (entry->generation != generation)
		{
			entry->generation = generation;
			memset(&entry->result, 0, sizeof(entry->result));
			entry->result.nodeId = bm25[i].nodeId;
			touched[touchedCount++] = bm25[i].nodeId;
		}
		else
			stats->fusionStats.duplicates++;

		PgturbohybridMergeBm25Candidate(&entry->result, &bm25[i]);
	}

	merged = MemoryContextAllocZero(memoryContext,
									sizeof(PgturbohybridResult) *
									Max((int) touchedCount, 1));
	for (uint32 i = 0; i < touchedCount; i++)
	{
		PgturbohybridResult item = entries[touched[i]].result;

		if ((state->query->flags & PGTURBOHYBRID_QUERY_FLAG_REQUIRE_BM25_MATCH) != 0 &&
			!item.hasBm25)
			continue;
		PgturbohybridRecordFusionClass(stats, &item);
		merged[mergedCount++] = item;
	}

	strlcpy(stats->fusionStats.strategy, "generation_array",
			sizeof(stats->fusionStats.strategy));
	if (mergedCountOut != NULL)
		*mergedCountOut = mergedCount;
	return PgturbohybridFinalizeFusedResults(scan, state, merged, mergedCount,
											 limit, false, memoryContext,
											 finalResults,
											 &stats->fusionStats.heapReplacements,
											 stats);
}

static int
PgturbohybridSparseCandidateScoreCompare(const void *a, const void *b)
{
	const PgturbohybridSparseCandidate *ca = (const PgturbohybridSparseCandidate *) a;
	const PgturbohybridSparseCandidate *cb = (const PgturbohybridSparseCandidate *) b;

	if (ca->score != cb->score)
		return ca->score > cb->score ? -1 : 1;	/* descending score */
	if (ca->nodeId != cb->nodeId)
		return ca->nodeId < cb->nodeId ? -1 : 1;	/* deterministic tiebreak */
	return 0;
}

static int
PgturbohybridSparseEntryTermCompare(const void *a, const void *b)
{
	const PgturbohybridSparseVectorEntry *ea = (const PgturbohybridSparseVectorEntry *) a;
	const PgturbohybridSparseVectorEntry *eb = (const PgturbohybridSparseVectorEntry *) b;

	if (ea->termId != eb->termId)
		return ea->termId < eb->termId ? -1 : 1;
	return 0;
}

/*
 * Exact f32 sparse rerank: for the top candidate band of a quantized
 * sparse scan, fetch the heap sparse column (MVCC-safe), recompute the exact
 * float32 sparse inner product against the query, then re-sort the candidates.
 * A no-op for f32 indexes under "auto"; mirrors PgturbohybridBm25HeapTSVectorRerank.
 */
static void
PgturbohybridSparseRerankCandidates(IndexScanDesc scan,
									PgturbohybridQueryHeader *query,
									PgturbohybridSparseCandidate *cands, int count,
									int autoBudgetLimit, int quantBits,
									PgturbohybridSparseScanStats *stats)
{
	int			mode = pgturbohybrid_sparse_rerank;
	Relation	heap;
	AttrNumber	sparseAttno;
	PgturbohybridIndexKeyMap map;
	struct varlena *qsv;
	const PgturbohybridSparseVectorEntry *qEntriesRaw;
	PgturbohybridSparseVectorEntry *qSorted;
	uint32		qCount = 0;
	int			band;
	int			finalTarget;
	int			rerankK;
	TupleTableSlot *slot;
	uint32	   *beforeTopK = NULL;
	int			beforeTopKCount = 0;
	int			rescored = 0;
	instr_time	t;

	if (stats != NULL)
		stats->rerankMode = mode;
	if (mode == PGTURBOHYBRID_SPARSE_RERANK_OFF || cands == NULL || count <= 0)
		return;
	/* "auto" reranks only quantized indexes; an f32 index is already exact. */
	if (mode == PGTURBOHYBRID_SPARSE_RERANK_AUTO && quantBits == 0)
		return;
	if (scan == NULL || scan->heapRelation == NULL ||
		scan->indexRelation == NULL || scan->indexRelation->rd_index == NULL)
		return;

	PgturbohybridBuildIndexKeyMap(scan->indexRelation, NULL, &map);
	if (map.sparseKey < 0)
		return;
	sparseAttno = scan->indexRelation->rd_index->indkey.values[map.sparseKey];
	heap = scan->heapRelation;
	if (sparseAttno <= 0 || sparseAttno > RelationGetDescr(heap)->natts)
		return;

	qsv = PgturbohybridQueryGetSparseVector(query);
	if (qsv == NULL)
		return;
	qEntriesRaw = PgturbohybridSparseVectorData(qsv, &qCount);
	if (qCount == 0)
		return;
	qSorted = (PgturbohybridSparseVectorEntry *)
		palloc(sizeof(PgturbohybridSparseVectorEntry) * qCount);
	memcpy(qSorted, qEntriesRaw, sizeof(PgturbohybridSparseVectorEntry) * qCount);
	qsort(qSorted, qCount, sizeof(PgturbohybridSparseVectorEntry),
		  PgturbohybridSparseEntryTermCompare);

	finalTarget = PgturbohybridBudgetFinalTarget(query, autoBudgetLimit);
	rerankK = pgturbohybrid_sparse_rerank_k > 0 ?
		pgturbohybrid_sparse_rerank_k : finalTarget;
	band = mode == PGTURBOHYBRID_SPARSE_RERANK_BAND ? count : Min(count, rerankK);
	if (band <= 0)
	{
		pfree(qSorted);
		return;
	}

	beforeTopKCount = Min(finalTarget, count);
	if (beforeTopKCount > 0)
	{
		beforeTopK = palloc(sizeof(uint32) * beforeTopKCount);
		for (int i = 0; i < beforeTopKCount; i++)
			beforeTopK[i] = cands[i].nodeId;
	}

	slot = table_slot_create(heap, NULL);
	for (int i = 0; i < band; i++)
	{
		Datum		value;
		bool		isnull;
		bool		visible;
		struct varlena *docDatum;
		const PgturbohybridSparseVectorEntry *docEntries;
		uint32		docN = 0;
		double		dot = 0.0;

		CHECK_FOR_INTERRUPTS();
		INSTR_TIME_SET_CURRENT(t);
		visible = table_tuple_fetch_row_version(heap, &cands[i].heaptid,
												scan->xs_snapshot, slot);
		if (stats != NULL)
			stats->exactRerankFetchUs += PgturbohybridElapsedUs(t);
		if (!visible)
		{
			ExecClearTuple(slot);
			continue;
		}
		value = slot_getattr(slot, sparseAttno, &isnull);
		if (isnull)
		{
			ExecClearTuple(slot);
			continue;
		}

		INSTR_TIME_SET_CURRENT(t);
		docDatum = (struct varlena *) PG_DETOAST_DATUM(value);
		docEntries = PgturbohybridSparseVectorData(docDatum, &docN);
		for (uint32 d = 0; d < docN; d++)
		{
			PgturbohybridSparseVectorEntry key;
			const PgturbohybridSparseVectorEntry *found;

			key.termId = docEntries[d].termId;
			found = (const PgturbohybridSparseVectorEntry *)
				bsearch(&key, qSorted, qCount,
						sizeof(PgturbohybridSparseVectorEntry),
						PgturbohybridSparseEntryTermCompare);
			if (found != NULL)
				dot += (double) found->weight * (double) docEntries[d].weight;
		}
		cands[i].score = dot;
		if (docDatum != (struct varlena *) DatumGetPointer(value))
			pfree(docDatum);
		if (stats != NULL)
			stats->exactRerankScoreUs += PgturbohybridElapsedUs(t);
		rescored++;
		ExecClearTuple(slot);
	}
	ExecDropSingleTupleTableSlot(slot);

	if (rescored > 0)
	{
		qsort(cands, count, sizeof(PgturbohybridSparseCandidate),
			  PgturbohybridSparseCandidateScoreCompare);
		for (int i = 0; i < beforeTopKCount; i++)
		{
			if (beforeTopK[i] != cands[i].nodeId)
			{
				if (stats != NULL)
					stats->exactRerankTopkChanged = true;
				break;
			}
		}
	}
	if (stats != NULL)
		stats->exactRerankCount = rescored;
	if (beforeTopK != NULL)
		pfree(beforeTopK);
	pfree(qSorted);
}

/*
 * Sparse-as-sole-ORDER-BY scan: resolve the query's sparse_query
 * against the native sparse inverted index, exact-OR-accumulate over live nodes,
 * and fill state->results sorted by descending inner product.  fusedScore is set
 * to the sparse score so amgettuple returns -score as the ORDER BY distance,
 * matching turbohybrid_sparse_inner_product_distance().
 */
static void
PgturbohybridCollectSparseOnlyResults(IndexScanDesc scan,
									  PgturbohybridScanState *state,
									  PgturbohybridQueryHeader *scanQuery,
									  PgturbohybridQueryHeader *originalQuery,
									  int autoBudgetLimit,
									  MemoryContext tmpCtx,
									  PgturbohybridLastScanStats *lastStats)
{
	PgturbohybridSparseCandidate *cands = NULL;
	PgturbohybridSparseScanStats sstats;
	PgturbohybridResult *results;
	int			finalTarget;
	int			candidateK;
	int			n;

	finalTarget = (int) PgturbohybridEffectiveFinalK(originalQuery, autoBudgetLimit);
	if (finalTarget < 1)
		finalTarget = 1;
	candidateK = scanQuery->sparseK > 0 ? scanQuery->sparseK : finalTarget;
	if (candidateK < finalTarget)
		candidateK = finalTarget;

	n = PgturbohybridSparseCollectCandidates(scan->indexRelation, scanQuery,
											 candidateK, pgturbohybrid_simd,
											 pgturbohybrid_enable_sparse_wand, &cands,
											 tmpCtx, &sstats);

	/* Exact f32 rerank of the top band; no-op for f32 indexes. */
	PgturbohybridSparseRerankCandidates(scan, scanQuery, cands, n, autoBudgetLimit,
										sstats.quantBits, &sstats);

	results = (PgturbohybridResult *) palloc0(sizeof(PgturbohybridResult) *
											  Max(n, 1));
	for (int i = 0; i < n; i++)
	{
		results[i].nodeId = cands[i].nodeId;
		results[i].heaptid = cands[i].heaptid;
		results[i].hasSparse = true;
		results[i].sparseSimilarity = cands[i].score;
		results[i].sparseRank = i + 1;
		results[i].fusedScore = cands[i].score;
	}

	state->results = results;
	state->resultCount = n;
	state->resultIndex = 0;
	state->collectDone = true;

	strlcpy(lastStats->indexShape, "sparse", sizeof(lastStats->indexShape));
	strlcpy(lastStats->profile, PgturbohybridProfileName(pgturbohybrid_profile),
			sizeof(lastStats->profile));
	strlcpy(lastStats->fusion, PgturbohybridQueryFusionName(scanQuery->fusion),
			sizeof(lastStats->fusion));
	lastStats->sparse.branchAvailable = sstats.branchAvailable;
	lastStats->sparse.branchUsed = sstats.branchUsed;
	lastStats->sparse.terms = sstats.terms;
	lastStats->sparse.resolvedTerms = sstats.resolvedTerms;
	lastStats->sparse.postingsTouched = sstats.postingsTouched;
	lastStats->sparse.candidatesScored = sstats.candidatesScored;
	lastStats->sparse.elapsedUs = sstats.elapsedUs;
	lastStats->sparse.quantBits = sstats.quantBits;
	lastStats->sparse.quantMode = sstats.quantMode;
	lastStats->sparse.encoding = sstats.encoding;
	lastStats->sparse.scalarTailPostings = sstats.scalarTailPostings;
	lastStats->sparse.rerankMode = sstats.rerankMode;
	lastStats->sparse.exactRerankCount = sstats.exactRerankCount;
	lastStats->sparse.exactRerankFetchUs = sstats.exactRerankFetchUs;
	lastStats->sparse.exactRerankScoreUs = sstats.exactRerankScoreUs;
	lastStats->sparse.exactRerankTopkChanged = sstats.exactRerankTopkChanged;
	lastStats->sparse.scoreKernel = sstats.scoreKernel;
	lastStats->sparse.simdBlocks = sstats.simdBlocks;
	lastStats->sparse.usedWand = sstats.usedWand;
	lastStats->sparse.blocksVisited = sstats.blocksVisited;
	lastStats->sparse.blocksSkipped = sstats.blocksSkipped;
	lastStats->sparse.wandPruned = sstats.wandPruned;
	lastStats->sparse.wandIterations = sstats.wandIterations;
	lastStats->sparse.wandThresholdUpdates = sstats.wandThresholdUpdates;
	lastStats->sparse.wandHeapUpdates = sstats.wandHeapUpdates;
	lastStats->sparse.cacheHit = sstats.cacheHit;
	lastStats->sparse.cacheBuildUs = sstats.cacheBuildUs;
	lastStats->sparse.cacheBytes = sstats.cacheBytes;
	lastStats->sparse.hotCacheHits = sstats.hotCacheHits;
	lastStats->sparse.hotCacheMisses = sstats.hotCacheMisses;
	lastStats->sparse.hotCacheBytes = sstats.hotCacheBytes;
	lastStats->sparse.hotCacheEvictions = sstats.hotCacheEvictions;
	lastStats->sparse.deltaPages = sstats.deltaPages;
	lastStats->sparse.deltaTerms = sstats.deltaTerms;
	lastStats->sparse.deltaPostingsDecoded = sstats.deltaPostingsDecoded;
	lastStats->sparse.deltaCacheHit = sstats.deltaCacheHit;
	lastStats->sparse.deltaGeneration = sstats.deltaGeneration;
	lastStats->sparse.candidatesRequested = originalQuery->sparseK;
	lastStats->sparse.candidatesEffective = scanQuery->sparseK;
	lastStats->sparse.kDefaulted =
		(originalQuery->flags & PGTURBOHYBRID_QUERY_FLAG_SPARSE_K_DEFAULTED) != 0;
	lastStats->sparse.candidates = n;
	lastStats->finalResults = n;
	lastStats->unionCandidates = n;
	lastStats->finalKRequested = PgturbohybridRequestedFinalK(originalQuery);
	lastStats->finalKEffective = finalTarget;
	lastStats->detectedSqlLimit = autoBudgetLimit;
	lastStats->autoBudgetLimit = autoBudgetLimit;
}

static void
PgturbohybridCollectScanResults(IndexScanDesc scan, PgturbohybridScanState *state)
{
	PgturbohybridGraphScanOpaque so = (PgturbohybridGraphScanOpaque) scan->opaque;
	TqDenseCandidate *dense = NULL;
	TqDenseCandidate *multivector = NULL;
	PgturbohybridBm25Result *bm25 = NULL;
	PgturbohybridResult *items = NULL;
	PgturbohybridResult *merged = NULL;
	TqDenseCandidateStats denseStats;
	TqDenseCandidateStats multivectorStats;
	PgturbohybridBm25QueryStats bm25Stats;
	int			denseCount = 0;
	int			multivectorCount = 0;
	int			bm25Count = 0;
	int			itemCount = 0;
	int			mergedCount = 0;
	int			finalCount;
	bool		useHashTopN;
	uint32		fusionCandidatesSeen;
	uint32		generationArrayNodeCount = 0;
	uint16		effectiveFusion;
	MemoryContext oldCtx;
	PgturbohybridLastScanStats lastStats;
	instr_time	totalStart;
	instr_time	phaseStart;
	PgturbohybridQueryHeader *originalQuery = state->query;
	PgturbohybridQueryHeader *scanQuery;
	int			autoBudgetLimit;
	char		bm25BudgetReason[48];
	double		bm25DenseConfidence = 0.0;
	PgturbohybridBm25QuerySignals hybridSignals;
	PgturbohybridHybridBudgetChoice hybridBudgetChoice;
	int			hybridEffectiveBm25Ceiling;
	TqDenseCandidate *denseProbe = NULL;
	int			denseProbeCount = 0;
	TqDenseCandidateStats denseProbeStats;
	int			denseProbeK = 0;
	bool		denseProbeReusable = false;
	bool		denseProbeAvailable = false;
	double		denseProbeGap = 0.0;
	uint64		denseProbeElapsedUs = 0;
	uint32		bm25HybridBoundStopRank = 0;
	uint32		bm25HybridBoundSkippedEstimated = 0;
	double		bm25HybridBoundThreshold = 0.0;
	bool		bm25HybridBoundSafe = true;
	int			bm25BudgetEffectiveK;
	PgturbohybridBm25FusedScoreBoundContext fusedBound;
	bool		hasLexicalKey;
	bool		denseBranchUsed = false;
	bool		bm25BranchUsed = false;
	uint16		requestedFusion;
	int			hybridQueryShapeForStats;
	bool		hasMultivectorQuery;
	bool		hasVectorQuery;
	bool		hasTextQuery;
	bool		indexIsMultiVector;
	bool		canRunVectorBranch;
	bool		useDocumentFusionKey;
	bool		multivectorBranchUsed = false;
	bool		multivectorMergedAsDense = false;
	uint64		multivectorElapsedUs = 0;
	PgturbohybridSparseCandidate *sparse = NULL;
	int			sparseCount = 0;
	PgturbohybridSparseScanStats sparseStats;
	bool		sparseBranchUsed = false;
	bool		hasSparseQuery;

	if (state->collectDone)
		return;

	INSTR_TIME_SET_CURRENT(totalStart);
	memset(&denseStats, 0, sizeof(denseStats));
	memset(&multivectorStats, 0, sizeof(multivectorStats));
	memset(&bm25Stats, 0, sizeof(bm25Stats));
	memset(&sparseStats, 0, sizeof(sparseStats));
	memset(&lastStats, 0, sizeof(lastStats));
	lastStats.finalDiversity.mode = PGTURBOHYBRID_FINAL_DIVERSITY_OFF;
	lastStats.finalDiversity.payloadSlot = -1;
	memset(&fusedBound, 0, sizeof(fusedBound));
	memset(&hybridSignals, 0, sizeof(hybridSignals));
	memset(&denseProbeStats, 0, sizeof(denseProbeStats));
	strlcpy(bm25BudgetReason, "not_evaluated", sizeof(bm25BudgetReason));
	oldCtx = MemoryContextSwitchTo(so->tmpCtx);
	autoBudgetLimit = PgturbohybridCurrentLimit();
	scanQuery = PgturbohybridEffectiveQuery(originalQuery, autoBudgetLimit, so->tmpCtx);
	state->query = scanQuery;
	requestedFusion = pgturbohybrid_force_fusion != 0 ?
		pgturbohybrid_force_fusion : scanQuery->fusion;
	hasLexicalKey = PgturbohybridIndexHasLexical(scan->indexRelation);
	so->graphFinalK = PgturbohybridBudgetFinalTarget(scanQuery, autoBudgetLimit);
	hybridEffectiveBm25Ceiling = scanQuery->bm25K;
	hasMultivectorQuery =
		(scanQuery->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_MULTIVECTOR) != 0;
	hasVectorQuery =
		(scanQuery->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_VECTOR) != 0;
	hasTextQuery =
		(scanQuery->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_TSQUERY) != 0;
	indexIsMultiVector = PgturbohybridIndexIsMultiVector(scan->indexRelation);
	canRunVectorBranch = hasVectorQuery && !indexIsMultiVector;
	useDocumentFusionKey = hasMultivectorQuery;

	/*
	 * Sparse-as-sole-ORDER-BY: the query targets only the sparse
	 * column.  Resolve it against the native sparse inverted index and return
	 * candidates ranked by exact inner product, bypassing the dense/bm25 fusion
	 * machinery.  Fusing sparse with dense/bm25 is handled by the RRF path.
	 */
	if (PgturbohybridQueryHasSparse(scanQuery) &&
		!hasVectorQuery && !hasMultivectorQuery && !hasTextQuery)
	{
		PgturbohybridCollectSparseOnlyResults(scan, state, scanQuery,
											  originalQuery, autoBudgetLimit,
											  so->tmpCtx, &lastStats);
		lastStats.elapsedUs = PgturbohybridElapsedUs(totalStart);
		pgturbohybrid_last_scan_state = lastStats;
		state->query = originalQuery;
		MemoryContextSwitchTo(oldCtx);
		return;
	}

	/*
	 * Sparse-with-dense/bm25 fusion only supports RRF; the
	 * weighted/fast_weighted/calibrated/dbsf modes are dense+bm25/maxsim only.
	 */
	hasSparseQuery = PgturbohybridQueryHasSparse(scanQuery);
	if (hasSparseQuery && requestedFusion != PGTURBOHYBRID_FUSION_RRF)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("sparse_query fusion currently supports only the rrf fusion mode"),
				 errhint("Use fusion => 'rrf' when combining sparse_query with dense or text retrieval.")));

	PgturbohybridQueryValidateMultiVectorFusionSupport(scan->indexRelation,
													   scanQuery);

	if (!hasLexicalKey && hasTextQuery)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("text_query requires a turbohybrid index with a tsvector key")));

	if ((pgturbohybrid_hybrid_budget_policy == PGTURBOHYBRID_HYBRID_BUDGET_ADAPTIVE ||
		 requestedFusion == PGTURBOHYBRID_FUSION_CALIBRATED) &&
		hasLexicalKey &&
		(scanQuery->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_TSQUERY) != 0)
		(void) PgturbohybridBm25AnalyzeQuerySignals(scan->indexRelation,
										 PgturbohybridQueryGetTsQuery(scanQuery),
										 so->tmpCtx, &hybridSignals);

	if (pgturbohybrid_hybrid_budget_policy == PGTURBOHYBRID_HYBRID_BUDGET_ADAPTIVE &&
		hasLexicalKey &&
		/* Adaptive hybrid dense probing remains vector-only for now. */
		canRunVectorBranch &&
		(scanQuery->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_TSQUERY) != 0 &&
		(originalQuery->flags & PGTURBOHYBRID_QUERY_FLAG_DENSE_K_DEFAULTED) != 0 &&
		scanQuery->denseK > PgturbohybridAdaptiveMinBudget(PgturbohybridBudgetFinalTarget(scanQuery,
																		  autoBudgetLimit)))
	{
		denseProbeK = Min(scanQuery->denseK,
						  Max(PgturbohybridBudgetFinalTarget(scanQuery,
														  autoBudgetLimit) * 2,
							  16));

		INSTR_TIME_SET_CURRENT(phaseStart);
		denseProbeCount = PgturbohybridGraphCollectDenseCandidates(scan, denseProbeK,
														 &denseProbe, so->tmpCtx,
														 &denseProbeStats);
		denseProbeElapsedUs = PgturbohybridElapsedUs(phaseStart);
		denseProbeAvailable = denseProbeCount > 1;
		denseProbeGap = PgturbohybridDenseConfidence(denseProbe, denseProbeCount,
											  PgturbohybridBudgetFinalTarget(scanQuery,
																 autoBudgetLimit));
	}

	PgturbohybridChooseAdaptiveHybridBudget(scanQuery, originalQuery,
											&hybridSignals, denseProbeGap,
											denseProbeAvailable,
											autoBudgetLimit,
											&hybridBudgetChoice);
	hybridQueryShapeForStats = hybridBudgetChoice.queryShape;
	if (!hybridBudgetChoice.adaptive &&
		requestedFusion == PGTURBOHYBRID_FUSION_CALIBRATED)
		hybridQueryShapeForStats =
			PgturbohybridHybridShapeFromSignals(scanQuery, &hybridSignals,
											denseProbeGap,
											denseProbeAvailable);
	strlcpy(lastStats.hybrid.queryShape,
			PgturbohybridHybridQueryShapeName(hybridQueryShapeForStats),
			sizeof(lastStats.hybrid.queryShape));
	strlcpy(lastStats.calibratedFusion.queryShape,
			PgturbohybridHybridQueryShapeName(hybridQueryShapeForStats),
			sizeof(lastStats.calibratedFusion.queryShape));
	if (denseProbe != NULL && denseProbeCount > 0 &&
		denseProbeK >= scanQuery->denseK)
	{
		dense = denseProbe;
		denseCount = denseProbeCount;
		denseStats = denseProbeStats;
		lastStats.dense.elapsedUs = denseProbeElapsedUs;
		denseProbeReusable = true;
	}

	if (canRunVectorBranch &&
		scanQuery->denseK > 0 && !denseProbeReusable)
	{
		denseBranchUsed = true;
		INSTR_TIME_SET_CURRENT(phaseStart);
		denseCount = PgturbohybridGraphCollectDenseCandidates(scan, scanQuery->denseK,
												   &dense, so->tmpCtx,
												   &denseStats);
		lastStats.dense.elapsedUs = denseProbeElapsedUs +
			PgturbohybridElapsedUs(phaseStart);
	}
	else if (denseProbeReusable)
		denseBranchUsed = true;

	if (hasMultivectorQuery && scanQuery->multivectorK > 0)
	{
		multivectorBranchUsed = true;
		INSTR_TIME_SET_CURRENT(phaseStart);
		multivectorCount =
			PgturbohybridGraphCollectMultiVectorDenseCandidates(scan,
																scanQuery,
																scanQuery->multivectorK,
																&multivector,
																so->tmpCtx,
																&multivectorStats);
		multivectorElapsedUs = PgturbohybridElapsedUs(phaseStart);
		if (!canRunVectorBranch)
		{
			dense = multivector;
			denseCount = multivectorCount;
			denseStats = multivectorStats;
			multivector = NULL;
			multivectorCount = 0;
			multivectorMergedAsDense = true;
		}
	}

	PgturbohybridMaybeApplyDenseBm25Budget(scanQuery, dense, denseCount,
									  autoBudgetLimit, bm25BudgetReason,
									  sizeof(bm25BudgetReason),
									  &bm25DenseConfidence);
	if (hasMultivectorQuery)
		PgturbohybridMaybeApplyMultiVectorAdmissionBudget(scanQuery, originalQuery,
														  multivectorMergedAsDense ? &denseStats : &multivectorStats,
														  multivectorMergedAsDense ? denseCount : multivectorCount,
														  autoBudgetLimit,
														  hybridEffectiveBm25Ceiling,
														  &hybridBudgetChoice);
	bm25BudgetEffectiveK = scanQuery->bm25K;
	PgturbohybridMaybeApplyBm25HybridBound(scanQuery, denseCount,
									  autoBudgetLimit,
									  &bm25HybridBoundStopRank,
									  &bm25HybridBoundSkippedEstimated,
									  &bm25HybridBoundThreshold,
									  &bm25HybridBoundSafe);
	PgturbohybridPrepareFastWeightedBound(scanQuery, dense, denseCount,
										  autoBudgetLimit, so->tmpCtx,
										  &fusedBound);

	if ((scanQuery->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_TSQUERY) != 0 &&
		hasLexicalKey &&
		scanQuery->bm25K > 0)
	{
		PgturbohybridOptions *opts = (PgturbohybridOptions *) scan->indexRelation->rd_options;
		bool		useWand = pgturbohybrid_enable_wand &&
			(opts == NULL || opts->bm25BlockMax);

		INSTR_TIME_SET_CURRENT(phaseStart);
		bm25BranchUsed = true;
		bm25Count = PgturbohybridBm25TopK(scan->indexRelation,
										 PgturbohybridQueryGetTsQuery(scanQuery),
										 scanQuery->bm25K, useWand, so->tmpCtx,
										 &bm25, &bm25Stats,
										 fusedBound.enabled ? &fusedBound : NULL);
		PgturbohybridBm25HeapTSVectorRerank(scan, scanQuery, bm25, bm25Count,
											autoBudgetLimit, &bm25Stats);
		lastStats.bm25.elapsedUs = PgturbohybridElapsedUs(phaseStart);
	}

	if (multivectorMergedAsDense)
		PgturbohybridInjectMultiVectorBm25Candidates(scan, scanQuery, bm25,
													 bm25Count, &dense,
													 &denseCount,
													 autoBudgetLimit,
													 so->tmpCtx, &denseStats);
	else
		PgturbohybridInjectMultiVectorBm25Candidates(scan, scanQuery, bm25,
													 bm25Count, &multivector,
													 &multivectorCount,
													 autoBudgetLimit,
													 so->tmpCtx, &multivectorStats);
	if (PgturbohybridUsingLearnedSparseCandidateInjection())
	{
		if (multivectorMergedAsDense)
			denseStats.learnedSparse.branchLatencyUs = lastStats.bm25.elapsedUs;
		else
			multivectorStats.learnedSparse.branchLatencyUs = lastStats.bm25.elapsedUs;
	}

	/* Sparse branch: exact f32 candidates fused via RRF below. */
	if (hasSparseQuery && scanQuery->sparseK > 0)
	{
		INSTR_TIME_SET_CURRENT(phaseStart);
		sparseBranchUsed = true;
		sparseCount = PgturbohybridSparseCollectCandidates(scan->indexRelation,
														   scanQuery,
														   scanQuery->sparseK,
														   pgturbohybrid_simd,
														   pgturbohybrid_enable_sparse_wand,
														   &sparse, so->tmpCtx,
														   &sparseStats);
		/* Exact f32 rerank before fusion so RRF ranks reflect exact scores. */
		PgturbohybridSparseRerankCandidates(scan, scanQuery, sparse, sparseCount,
											autoBudgetLimit, sparseStats.quantBits,
											&sparseStats);
		lastStats.sparse.elapsedUs = PgturbohybridElapsedUs(phaseStart);
	}

	INSTR_TIME_SET_CURRENT(phaseStart);
	fusionCandidatesSeen = denseCount + multivectorCount + bm25Count + sparseCount;
	lastStats.fusionStats.candidatesSeen = fusionCandidatesSeen;
	effectiveFusion = pgturbohybrid_force_fusion != 0 ?
		pgturbohybrid_force_fusion : scanQuery->fusion;
	useHashTopN = pgturbohybrid_fusion_hash_threshold >= 0 &&
		fusionCandidatesSeen >= (uint32) pgturbohybrid_fusion_hash_threshold;
	/*
	 * The generation-array and hash fusion fast paths do not yet thread the
	 * sparse branch; route sparse-present fusion through the general
	 * sorted-merge path (correct, just unoptimized -- later prompts extend the
	 * fast paths).
	 */
	if (hasSparseQuery)
		useHashTopN = false;
	if (!hasSparseQuery && !useDocumentFusionKey &&
		PgturbohybridShouldUseGenerationArray(scan, effectiveFusion,
											  fusionCandidatesSeen, dense,
											  denseCount, bm25, bm25Count,
											  &generationArrayNodeCount))
	{
		finalCount = PgturbohybridFuseGenerationArray(scan, state, dense,
													  denseCount, bm25,
													  bm25Count,
													  generationArrayNodeCount,
													  autoBudgetLimit,
													  so->tmpCtx, &merged,
													  &mergedCount,
													  &lastStats);
		lastStats.fusionStats.heapSize = finalCount;
	}
	else if (useHashTopN)
	{
		uint32		slotCount =
			PgturbohybridFusionHashSlotCount(fusionCandidatesSeen);
		uint32		slotMask = slotCount - 1;
		PgturbohybridMergeSlot *slots =
			palloc0(sizeof(PgturbohybridMergeSlot) * slotCount);

		for (int i = 0; i < denseCount; i++)
		{
			PgturbohybridResult *item =
				useDocumentFusionKey ?
				PgturbohybridFindMergeSlotByHeapTid(slots, slotMask,
													&dense[i].heaptid) :
				PgturbohybridFindMergeSlot(slots, slotMask, dense[i].nodeId);

			if (item->hasDense || item->hasMultivector || item->hasBm25)
				lastStats.fusionStats.duplicates++;
			if (multivectorMergedAsDense)
				PgturbohybridAddMultiVectorCandidate(item, &dense[i]);
			else
				PgturbohybridAddDenseCandidate(item, &dense[i]);
		}
		for (int i = 0; i < multivectorCount; i++)
		{
			PgturbohybridResult *item =
				useDocumentFusionKey ?
				PgturbohybridFindMergeSlotByHeapTid(slots, slotMask,
													&multivector[i].heaptid) :
				PgturbohybridFindMergeSlot(slots, slotMask,
										   multivector[i].nodeId);

			if (item->hasDense || item->hasMultivector || item->hasBm25)
				lastStats.fusionStats.duplicates++;
			PgturbohybridAddMultiVectorCandidate(item, &multivector[i]);
		}
		for (int i = 0; i < bm25Count; i++)
		{
			PgturbohybridResult *item =
				useDocumentFusionKey ?
				PgturbohybridFindMergeSlotByHeapTid(slots, slotMask,
													&bm25[i].heaptid) :
				PgturbohybridFindMergeSlot(slots, slotMask, bm25[i].nodeId);

			if (item->hasDense || item->hasMultivector || item->hasBm25)
				lastStats.fusionStats.duplicates++;
			PgturbohybridMergeBm25Candidate(item, &bm25[i]);
		}

		merged = palloc0(sizeof(PgturbohybridResult) *
						 Max(fusionCandidatesSeen, 1));
		for (uint32 i = 0; i < slotCount; i++)
		{
			PgturbohybridResult item;

			if (!slots[i].used)
				continue;
			item = slots[i].result;
			if ((scanQuery->flags & PGTURBOHYBRID_QUERY_FLAG_REQUIRE_BM25_MATCH) != 0 &&
				!item.hasBm25)
				continue;
			PgturbohybridRecordFusionClass(&lastStats, &item);
			merged[mergedCount++] = item;
		}

		finalCount = PgturbohybridFinalizeFusedResults(scan, state, merged,
													   mergedCount,
													   autoBudgetLimit, true,
													   so->tmpCtx, &merged,
													   &lastStats.fusionStats.heapReplacements,
													   &lastStats);
		strlcpy(lastStats.fusionStats.strategy,
				useDocumentFusionKey ? "hash_doc" : "hash",
				sizeof(lastStats.fusionStats.strategy));
		lastStats.fusionStats.heapSize = finalCount;
	}
	else
	{
		items = palloc0(sizeof(PgturbohybridResult) *
						Max((int) fusionCandidatesSeen, 1));
		for (int i = 0; i < denseCount; i++)
		{
			if (multivectorMergedAsDense)
				PgturbohybridAddMultiVectorCandidate(&items[itemCount++],
													 &dense[i]);
			else
				PgturbohybridAddDenseCandidate(&items[itemCount++], &dense[i]);
		}
		for (int i = 0; i < multivectorCount; i++)
			PgturbohybridAddMultiVectorCandidate(&items[itemCount++],
												 &multivector[i]);
		for (int i = 0; i < bm25Count; i++)
			PgturbohybridAddBm25Candidate(&items[itemCount++], &bm25[i]);
		for (int i = 0; i < sparseCount; i++)
			PgturbohybridAddSparseCandidate(&items[itemCount++], &sparse[i],
											i + 1);

		if (itemCount > 1)
			qsort(items, itemCount, sizeof(PgturbohybridResult),
				  useDocumentFusionKey ?
				  PgturbohybridHeapTidCompare : PgturbohybridNodeCompare);

		merged = palloc0(sizeof(PgturbohybridResult) * Max(itemCount, 1));
		for (int i = 0; i < itemCount;)
		{
			PgturbohybridResult item = items[i++];

			while (i < itemCount &&
				   (useDocumentFusionKey ?
					PgturbohybridItemPointerDataCompare(&items[i].heaptid,
														&item.heaptid) == 0 :
					items[i].nodeId == item.nodeId))
			{
				lastStats.fusionStats.duplicates++;
				if (items[i].hasDense)
				{
					item.hasDense = true;
					item.denseDistance = items[i].denseDistance;
					item.denseSimilarity = items[i].denseSimilarity;
					item.denseRank = items[i].denseRank;
					item.exactScored = items[i].exactScored;
					item.heaptid = items[i].heaptid;
				}
				if (items[i].hasMultivector)
				{
					item.hasMultivector = true;
					item.multivectorDistance = items[i].multivectorDistance;
					item.multivectorSimilarity = items[i].multivectorSimilarity;
					item.multivectorRank = items[i].multivectorRank;
					item.exactScored = items[i].exactScored;
					item.heaptid = items[i].heaptid;
				}
				if (items[i].hasBm25)
					PgturbohybridMergeBm25ResultItem(&item, &items[i]);
				if (items[i].hasSparse)
					PgturbohybridMergeSparseResultItem(&item, &items[i]);
				i++;
			}

			if ((scanQuery->flags & PGTURBOHYBRID_QUERY_FLAG_REQUIRE_BM25_MATCH) != 0 &&
				!item.hasBm25)
				continue;
			if ((scanQuery->flags & PGTURBOHYBRID_QUERY_FLAG_REQUIRE_SPARSE_MATCH) != 0 &&
				!item.hasSparse)
				continue;
			PgturbohybridRecordFusionClass(&lastStats, &item);
			merged[mergedCount++] = item;
		}

		finalCount = PgturbohybridFinalizeFusedResults(scan, state, merged,
													   mergedCount,
													   autoBudgetLimit, true,
													   so->tmpCtx, &merged,
													   &lastStats.fusionStats.heapReplacements,
													   &lastStats);
		strlcpy(lastStats.fusionStats.strategy,
				useDocumentFusionKey ? "sorted_merge_doc" : "sorted_merge",
				sizeof(lastStats.fusionStats.strategy));
		lastStats.fusionStats.heapSize = finalCount;
	}

	lastStats.fusionStats.elapsedUs = PgturbohybridElapsedUs(phaseStart);
	state->results = merged;
	state->resultCount = finalCount;
	state->resultIndex = 0;
	state->collectDone = true;

	strlcpy(lastStats.indexShape, hasLexicalKey ? "hybrid" : "dense_only",
			sizeof(lastStats.indexShape));
	lastStats.bm25.branchAvailable = hasLexicalKey;
	lastStats.dense.branchUsed = denseBranchUsed;
	lastStats.multivector.branchUsed = multivectorBranchUsed;
	lastStats.bm25.branchUsed = bm25BranchUsed;
	lastStats.sparse.branchAvailable = sparseStats.branchAvailable;
	lastStats.sparse.branchUsed = sparseBranchUsed;
	lastStats.sparse.terms = sparseStats.terms;
	lastStats.sparse.resolvedTerms = sparseStats.resolvedTerms;
	lastStats.sparse.postingsTouched = sparseStats.postingsTouched;
	lastStats.sparse.candidatesScored = sparseStats.candidatesScored;
	lastStats.sparse.quantBits = sparseStats.quantBits;
	lastStats.sparse.quantMode = sparseStats.quantMode;
	lastStats.sparse.encoding = sparseStats.encoding;
	lastStats.sparse.scalarTailPostings = sparseStats.scalarTailPostings;
	lastStats.sparse.rerankMode = sparseStats.rerankMode;
	lastStats.sparse.exactRerankCount = sparseStats.exactRerankCount;
	lastStats.sparse.exactRerankFetchUs = sparseStats.exactRerankFetchUs;
	lastStats.sparse.exactRerankScoreUs = sparseStats.exactRerankScoreUs;
	lastStats.sparse.exactRerankTopkChanged = sparseStats.exactRerankTopkChanged;
	lastStats.sparse.scoreKernel = sparseStats.scoreKernel;
	lastStats.sparse.simdBlocks = sparseStats.simdBlocks;
	lastStats.sparse.usedWand = sparseStats.usedWand;
	lastStats.sparse.blocksVisited = sparseStats.blocksVisited;
	lastStats.sparse.blocksSkipped = sparseStats.blocksSkipped;
	lastStats.sparse.wandPruned = sparseStats.wandPruned;
	lastStats.sparse.wandIterations = sparseStats.wandIterations;
	lastStats.sparse.wandThresholdUpdates = sparseStats.wandThresholdUpdates;
	lastStats.sparse.wandHeapUpdates = sparseStats.wandHeapUpdates;
	lastStats.sparse.cacheHit = sparseStats.cacheHit;
	lastStats.sparse.cacheBuildUs = sparseStats.cacheBuildUs;
	lastStats.sparse.cacheBytes = sparseStats.cacheBytes;
	lastStats.sparse.hotCacheHits = sparseStats.hotCacheHits;
	lastStats.sparse.hotCacheMisses = sparseStats.hotCacheMisses;
	lastStats.sparse.hotCacheBytes = sparseStats.hotCacheBytes;
	lastStats.sparse.hotCacheEvictions = sparseStats.hotCacheEvictions;
	lastStats.sparse.deltaPages = sparseStats.deltaPages;
	lastStats.sparse.deltaTerms = sparseStats.deltaTerms;
	lastStats.sparse.deltaPostingsDecoded = sparseStats.deltaPostingsDecoded;
	lastStats.sparse.deltaCacheHit = sparseStats.deltaCacheHit;
	lastStats.sparse.deltaGeneration = sparseStats.deltaGeneration;
	if (hasSparseQuery)
	{
		lastStats.sparse.candidatesRequested = originalQuery->sparseK;
		lastStats.sparse.candidatesEffective = scanQuery->sparseK;
		lastStats.sparse.kDefaulted =
			(originalQuery->flags & PGTURBOHYBRID_QUERY_FLAG_SPARSE_K_DEFAULTED) != 0;
		lastStats.sparse.candidates = sparseCount;
	}
	strlcpy(lastStats.profile, PgturbohybridProfileName(pgturbohybrid_profile),
			sizeof(lastStats.profile));
	strlcpy(lastStats.fusion,
			PgturbohybridQueryFusionName(pgturbohybrid_force_fusion != 0 ?
								  pgturbohybrid_force_fusion : scanQuery->fusion),
			sizeof(lastStats.fusion));
	if (denseBranchUsed)
	{
		lastStats.dense.candidatesRequested = originalQuery->denseK;
		lastStats.dense.candidatesEffective = scanQuery->denseK;
		lastStats.dense.kDefaulted =
			(originalQuery->flags & PGTURBOHYBRID_QUERY_FLAG_DENSE_K_DEFAULTED) != 0;
		lastStats.dense.candidates = denseCount;
		lastStats.dense.effectiveResultTarget = denseStats.effectiveResultTarget;
		lastStats.dense.effectiveSearchEf = denseStats.effectiveSearchEf;
		lastStats.dense.effectiveRescoreBand = denseStats.effectiveRescoreBand;
		lastStats.dense.highdimWideningMultiplier =
			denseStats.highdimWideningMultiplier;
		lastStats.dense.wideningReason = denseStats.wideningReason;
		lastStats.dense.budgetPolicy = denseStats.dense.budgetPolicy;
		lastStats.dense.rescoreBandPolicy = denseStats.rescoreBandPolicy;
	}
	lastStats.multivector.candidatesRequested = originalQuery->multivectorK;
	lastStats.multivector.candidatesEffective = scanQuery->multivectorK;
	lastStats.multivector.candidates =
		multivectorMergedAsDense ? denseCount : multivectorCount;
	if (hasMultivectorQuery && !multivectorMergedAsDense)
		denseStats = multivectorStats;
	lastStats.bm25.candidatesRequested = originalQuery->bm25K;
	lastStats.bm25.candidatesEffective = bm25BudgetEffectiveK;
	lastStats.bm25.kDefaulted =
		(originalQuery->flags & PGTURBOHYBRID_QUERY_FLAG_BM25_K_DEFAULTED) != 0;
	lastStats.bm25.candidates = bm25Count;
	strlcpy(lastStats.bm25.budgetReason, bm25BudgetReason,
			sizeof(lastStats.bm25.budgetReason));
	lastStats.bm25.denseConfidence = bm25DenseConfidence;
	lastStats.bm25.hybridBoundMode = pgturbohybrid_bm25_hybrid_bound;
	lastStats.bm25.hybridBoundStopRank = bm25HybridBoundStopRank;
	lastStats.bm25.hybridBoundSkippedEstimated =
		bm25HybridBoundSkippedEstimated;
	lastStats.bm25.hybridBoundThreshold = bm25HybridBoundThreshold;
	lastStats.bm25.hybridBoundSafe = bm25HybridBoundSafe;
	lastStats.rrfKRequested = originalQuery->rrfK;
	lastStats.rrfKEffective = scanQuery->rrfK;
	lastStats.rrfKDefaulted =
		(originalQuery->flags & PGTURBOHYBRID_QUERY_FLAG_RRF_K_DEFAULTED) != 0;
	lastStats.finalKRequested = PgturbohybridRequestedFinalK(originalQuery);
	lastStats.finalKEffective =
		PgturbohybridEffectiveFinalK(originalQuery, autoBudgetLimit);
	lastStats.detectedSqlLimit = autoBudgetLimit;
	lastStats.finalKInferred =
		PgturbohybridFinalKInferred(originalQuery, autoBudgetLimit);
	pgturbohybrid_last_final_k_requested = lastStats.finalKRequested;
	pgturbohybrid_last_final_k_effective = lastStats.finalKEffective;
	pgturbohybrid_last_sql_limit = lastStats.detectedSqlLimit;
	pgturbohybrid_last_final_k_inferred = lastStats.finalKInferred;
	lastStats.autoBudgetLimit = autoBudgetLimit;
	lastStats.unionCandidates = mergedCount;
	lastStats.finalResults = finalCount;
	lastStats.fusionStats.candidatesSeen = fusionCandidatesSeen;
	lastStats.graphVisitedNodes = denseStats.visitedGraphNodes;
	lastStats.graphScoredCodes = denseStats.scoredCodes;
	lastStats.graphExactRescoreCount = denseStats.exactRescoreCount;
	lastStats.graphHeapRescoreCount = denseStats.heapRescoreCount;
	lastStats.graphHeapFetchUs = denseStats.heapFetchUs;
	lastStats.graphHeapRescoreUs = denseStats.heapRescoreUs;
	lastStats.graphHeapRescoreAutoEnabled =
		denseStats.heapRescoreAutoEnabled;
	lastStats.graphHeapRescoreReason = denseStats.heapRescoreReason;
	lastStats.graphExactRescoreSource = denseStats.exactRescoreSource;
	lastStats.graphPrepareUs = denseStats.prepareUs;
	lastStats.graphTraverseUs = denseStats.traverseUs;
	lastStats.graphEntryUs = denseStats.entryUs;
	lastStats.graphBaseUs = denseStats.baseUs;
	lastStats.graphBatchUs = denseStats.batchUs;
	lastStats.graphHeapUs = denseStats.heapUs;
	lastStats.graphFillUs = denseStats.fillUs;
	lastStats.graphFillCandidateBandCalls = denseStats.fillCandidateBandCalls;
	lastStats.graphFillCandidateBandReason = denseStats.fillCandidateBandReason;
	lastStats.graphFillCandidateBandVisited = denseStats.fillCandidateBandVisited;
	lastStats.graphFillCandidateBandScored = denseStats.fillCandidateBandScored;
	lastStats.graphFillCandidateBandSelectedBefore =
		denseStats.fillCandidateBandSelectedBefore;
	lastStats.graphFillCandidateBandSelectedAfter =
		denseStats.fillCandidateBandSelectedAfter;
	lastStats.graphFillCandidateBandTarget = denseStats.fillCandidateBandTarget;
	lastStats.graphFillCandidateBandUsedPayloadRefs =
		denseStats.fillCandidateBandUsedPayloadRefs;
	lastStats.graphFillCandidateBandPayloadRefCount =
		denseStats.fillCandidateBandPayloadRefCount;
	lastStats.graphRescoreUs = denseStats.rescoreUs;
	lastStats.graphSortUs = denseStats.sortUs;
	lastStats.multivector.enabled = denseStats.multivector.enabled;
	lastStats.multivector.queryVectors = denseStats.multivector.queryVectors;
	lastStats.multivector.docVectorsLimit =
		denseStats.multivector.docVectorsLimit;
	lastStats.multivector.subvectorSearches =
		denseStats.multivector.subvectorSearches;
	lastStats.multivector.rawSubvectorHits =
		denseStats.multivector.rawSubvectorHits;
	lastStats.multivector.adaptiveWideningTriggered =
		denseStats.multivector.adaptiveWideningTriggered;
	lastStats.multivector.adaptiveInitialRawTarget =
		denseStats.multivector.adaptiveInitialRawTarget;
	lastStats.multivector.adaptiveFinalRawTarget =
		denseStats.multivector.adaptiveFinalRawTarget;
	strlcpy(lastStats.multivector.candidateSource,
			denseStats.multivector.candidateSource,
			sizeof(lastStats.multivector.candidateSource));
	strlcpy(lastStats.multivector.candidatePath,
			denseStats.multivector.candidatePath,
			sizeof(lastStats.multivector.candidatePath));
	strlcpy(lastStats.multivector.proxyEncoderKind,
			denseStats.multivector.proxyEncoderKind,
			sizeof(lastStats.multivector.proxyEncoderKind));
	lastStats.learnedProjection = denseStats.learnedProjection;
	strlcpy(lastStats.multivector.graphMode,
			denseStats.multivector.graphMode,
			sizeof(lastStats.multivector.graphMode));
	lastStats.multivector.proxyGraphSearches =
		denseStats.multivector.proxyGraphSearches;
	lastStats.multivector.exactTokenScanEnabled =
		denseStats.multivector.exactTokenScanEnabled;
	lastStats.multivector.exactTokenScanNodesScored =
		denseStats.multivector.exactTokenScanNodesScored;
	lastStats.multivector.plainFallbackUsed =
		denseStats.multivector.plainFallbackUsed;
	strlcpy(lastStats.multivector.plainFallbackReason,
			denseStats.multivector.plainFallbackReason,
			sizeof(lastStats.multivector.plainFallbackReason));
	lastStats.multivector.plainFallbackDocsScored =
		denseStats.multivector.plainFallbackDocsScored;
	lastStats.multivector.plainFallbackPairs =
		denseStats.multivector.plainFallbackPairs;
	lastStats.multivector.docGraphPrototypeEnabled =
		denseStats.multivector.docGraphPrototypeEnabled;
	lastStats.multivector.docGraphNodes =
		denseStats.multivector.docGraphNodes;
	lastStats.multivector.docGraphDocsScored =
		denseStats.multivector.docGraphDocsScored;
	lastStats.multivector.docGraphEdgesVisited =
		denseStats.multivector.docGraphEdgesVisited;
	lastStats.multivector.docGraphCandidates =
		denseStats.multivector.docGraphCandidates;
	lastStats.multivector.docGraphSearchEf =
		denseStats.multivector.docGraphSearchEf;
	lastStats.multivector.docGraphOversampling =
		denseStats.multivector.docGraphOversampling;
	lastStats.multivector.docGraphRescoreK =
		denseStats.multivector.docGraphRescoreK;
	lastStats.multivector.docGraphEntrySampleConfigured =
		denseStats.multivector.docGraphEntrySampleConfigured;
	lastStats.multivector.docGraphEntrySampleEffective =
		denseStats.multivector.docGraphEntrySampleEffective;
	lastStats.multivector.docGraphEntrySampleScored =
		denseStats.multivector.docGraphEntrySampleScored;
	lastStats.multivector.docGraphQuantizedScores =
		denseStats.multivector.docGraphQuantizedScores;
	lastStats.compactMaxsimScoreUs =
		denseStats.compactMaxsimScoreUs;
	lastStats.compactMaxsimPairs =
		denseStats.compactMaxsimPairs;
	lastStats.compactMaxsimCacheHits =
		denseStats.compactMaxsimCacheHits;
	lastStats.compactMaxsimCacheMisses =
		denseStats.compactMaxsimCacheMisses;
	lastStats.compactMaxsimBoundChecks =
		denseStats.compactMaxsimBoundChecks;
	lastStats.compactMaxsimDocsPruned =
		denseStats.compactMaxsimDocsPruned;
	lastStats.compactMaxsimTokensSkipped =
		denseStats.compactMaxsimTokensSkipped;
	strlcpy(lastStats.multivector.docGraphStorageKind,
			denseStats.multivector.docGraphStorageKind,
			sizeof(lastStats.multivector.docGraphStorageKind));
	lastStats.proxy = denseStats.proxy;
	lastStats.centroid = denseStats.centroid;
	lastStats.fullMultivectorSidecarAvailable =
		denseStats.fullMultivectorSidecarAvailable;
	lastStats.quantizedInverted = denseStats.quantizedInverted;
	strlcpy(lastStats.multivector.docGraphRescoreSource,
			denseStats.multivector.docGraphRescoreSource,
			sizeof(lastStats.multivector.docGraphRescoreSource));
	lastStats.multivector.docGraphExactRerankDocs =
		denseStats.multivector.docGraphExactRerankDocs;
	lastStats.multivector.docGraphHeapFetches =
		denseStats.multivector.docGraphHeapFetches;
	strlcpy(lastStats.multivector.docGraphWarning,
			denseStats.multivector.docGraphWarning,
			sizeof(lastStats.multivector.docGraphWarning));
	lastStats.multivector.proxyCandidateTarget =
		denseStats.multivector.proxyCandidateTarget;
	lastStats.multivector.proxyCandidatesReturned =
		denseStats.multivector.proxyCandidatesReturned;
	lastStats.multivector.exactRerankKEffective =
		denseStats.multivector.exactRerankKEffective;
	strlcpy(lastStats.multivector.docStorageCacheRequested,
			denseStats.multivector.docStorageCacheRequested,
			sizeof(lastStats.multivector.docStorageCacheRequested));
	strlcpy(lastStats.multivector.docStorageCacheEffective,
			denseStats.multivector.docStorageCacheEffective,
			sizeof(lastStats.multivector.docStorageCacheEffective));
	lastStats.sidecar = denseStats.sidecar;
	lastStats.multivector.centroidCount =
		denseStats.multivector.centroidCount;
	lastStats.multivector.centroidPrerankDocs =
		denseStats.multivector.centroidPrerankDocs;
	lastStats.multivector.fullMaxsimRerankDocs =
		denseStats.multivector.fullMaxsimRerankDocs;
	strlcpy(lastStats.multivector.docSidecarCacheMode,
			denseStats.multivector.docSidecarCacheMode,
			sizeof(lastStats.multivector.docSidecarCacheMode));
	lastStats.multivector.docSidecarPagesRead =
		denseStats.multivector.docSidecarPagesRead;
	lastStats.multivector.docSidecarCacheHits =
		denseStats.multivector.docSidecarCacheHits;
	lastStats.multivector.docSidecarCacheMisses =
		denseStats.multivector.docSidecarCacheMisses;
	lastStats.multivector.docSidecarBytesTouched =
		denseStats.multivector.docSidecarBytesTouched;
	lastStats.multivector.docSidecarVectorsLoaded =
		denseStats.multivector.docSidecarVectorsLoaded;
	lastStats.multivector.docSidecarDocMapPagesRead =
		denseStats.multivector.docSidecarDocMapPagesRead;
	lastStats.multivector.docSidecarDocMapBytesTouched =
		denseStats.multivector.docSidecarDocMapBytesTouched;
	lastStats.multivector.docSidecarResidentVectorsLoaded =
		denseStats.multivector.docSidecarResidentVectorsLoaded;
	lastStats.multivector.docSidecarResidentBytesLoaded =
		denseStats.multivector.docSidecarResidentBytesLoaded;
	lastStats.multivector.docSidecarVectorChunkRefBytesTouched =
		denseStats.multivector.docSidecarVectorChunkRefBytesTouched;
	lastStats.multivector.docSidecarPagedVectorPagesRead =
		denseStats.multivector.docSidecarPagedVectorPagesRead;
	lastStats.multivector.docSidecarPagedVectorBytesTouched =
		denseStats.multivector.docSidecarPagedVectorBytesTouched;
	lastStats.multivector.sidecarPageReadUs =
		denseStats.multivector.sidecarPageReadUs;
	lastStats.multivector.sidecarVectorReconstructUs =
		denseStats.multivector.sidecarVectorReconstructUs;
	lastStats.multivector.tokensOriginal =
		denseStats.multivector.tokensOriginal;
	lastStats.multivector.tokensPooled =
		denseStats.multivector.tokensPooled;
	lastStats.multivector.reservoirsEnabled =
		denseStats.multivector.reservoirsEnabled;
	lastStats.multivector.reservoirScoreDocs =
		denseStats.multivector.reservoirScoreDocs;
	lastStats.multivector.reservoirCoverageDocs =
		denseStats.multivector.reservoirCoverageDocs;
	lastStats.multivector.reservoirMeanDocs =
		denseStats.multivector.reservoirMeanDocs;
	lastStats.multivector.reservoirPerTokenDocs =
		denseStats.multivector.reservoirPerTokenDocs;
	lastStats.multivector.reservoirBm25Docs =
		denseStats.multivector.reservoirBm25Docs;
	lastStats.multivector.reservoirUnionDocs =
		denseStats.multivector.reservoirUnionDocs;
	lastStats.multivector.reservoirDuplicates =
		denseStats.multivector.reservoirDuplicates;
	lastStats.multivector.bm25InjectionEnabled =
		denseStats.multivector.bm25InjectionEnabled;
	lastStats.multivector.bm25InjectionCandidates =
		denseStats.multivector.bm25InjectionCandidates;
	lastStats.multivector.bm25InjectionCandidateLimit =
		denseStats.multivector.bm25InjectionCandidateLimit;
	lastStats.multivector.bm25InjectionPoolSize =
		denseStats.multivector.bm25InjectionPoolSize;
	strlcpy(lastStats.multivector.bm25InjectionLimitReason,
			denseStats.multivector.bm25InjectionLimitReason,
			sizeof(lastStats.multivector.bm25InjectionLimitReason));
	lastStats.multivector.bm25InjectionRetained =
		denseStats.multivector.bm25InjectionRetained;
	lastStats.multivector.bm25InjectionExactReranked =
		denseStats.multivector.bm25InjectionExactReranked;
	lastStats.learnedSparse = denseStats.learnedSparse;
	switch ((PgturbohybridMultiVectorDocMapSource) denseStats.multivector.docMapSource)
	{
		case PGTURBOHYBRID_MULTIVECTOR_DOCMAP_SOURCE_SIDECAR:
			strlcpy(lastStats.multivector.docMapSource, "sidecar",
					sizeof(lastStats.multivector.docMapSource));
			break;
		case PGTURBOHYBRID_MULTIVECTOR_DOCMAP_SOURCE_HEAP_TID_HASH:
			strlcpy(lastStats.multivector.docMapSource, "heap_tid_hash",
					sizeof(lastStats.multivector.docMapSource));
			break;
		case PGTURBOHYBRID_MULTIVECTOR_DOCMAP_SOURCE_NONE:
		default:
			strlcpy(lastStats.multivector.docMapSource, "none",
					sizeof(lastStats.multivector.docMapSource));
			break;
	}
	lastStats.multivector.docMapBytes =
		denseStats.multivector.docMapBytes;
	lastStats.multivector.uniqueDocs = denseStats.multivector.uniqueDocs;
	lastStats.multivector.duplicateDocHits =
		denseStats.multivector.duplicateDocHits;
	lastStats.multivector.maxsimUpdates =
		denseStats.multivector.maxsimUpdates;
	lastStats.multivector.docCandidates = denseStats.multivector.docCandidates;
	lastStats.multivector.exactRerankEnabled =
		denseStats.multivector.exactRerankEnabled;
	lastStats.multivector.exactRerankDocs =
		denseStats.multivector.exactRerankDocs;
	lastStats.multivector.exactRerankPairs =
		denseStats.multivector.exactRerankPairs;
	strlcpy(lastStats.multivector.exactRerankSource,
			PgturbohybridMultiVectorRerankSourceName(
				denseStats.multivector.exactRerankSource),
			sizeof(lastStats.multivector.exactRerankSource));
	lastStats.multivector.exactRerankHeapFetches =
		denseStats.multivector.exactRerankHeapFetches;
	lastStats.multivector.exactRerankSidecarReads =
		denseStats.multivector.exactRerankSidecarReads;
	lastStats.multivector.exactRerankSidecarBytes =
		denseStats.multivector.exactRerankSidecarBytes;
	lastStats.multivector.candidateSourceUs =
		denseStats.multivector.candidateSourceUs;
	lastStats.multivector.docGraphTraversalUs =
		denseStats.multivector.docGraphTraversalUs;
	lastStats.multivector.proxyCandidateUs =
		denseStats.multivector.proxyCandidateUs;
	lastStats.multivector.proxyGraphTraversalUs =
		denseStats.multivector.proxyGraphTraversalUs;
	lastStats.multivector.proxyScoringUs =
		denseStats.multivector.proxyScoringUs;
	lastStats.multivector.centroidLitePostingUs =
		denseStats.multivector.centroidLitePostingUs;
	lastStats.multivector.quantizedInvertedPostingUs =
		denseStats.multivector.quantizedInvertedPostingUs;
	lastStats.multivector.sidecarLoadUs =
		denseStats.multivector.sidecarLoadUs;
	lastStats.multivector.heapVisibilityUs =
		denseStats.multivector.heapVisibilityUs;
	lastStats.multivector.exactHeapFetchUs =
		denseStats.multivector.exactHeapFetchUs;
	lastStats.multivector.exactRerankUs =
		denseStats.multivector.exactRerankUs;
	lastStats.multivector.finalSortUs =
		denseStats.multivector.finalSortUs;
	lastStats.exactRerankCandidates = denseStats.exactRerankCandidates;
	lastStats.exactRerankTokensEvaluated =
		denseStats.exactRerankTokensEvaluated;
	lastStats.exactRerankTokensSkipped =
		denseStats.exactRerankTokensSkipped;
	lastStats.exactRerankPairsSaved = denseStats.exactRerankPairsSaved;
	lastStats.adaptiveRerankTopKChangedVsFull =
		denseStats.adaptiveRerankTopKChangedVsFull;
	strlcpy(lastStats.multivector.exactKernel,
			denseStats.multivector.exactKernel,
			sizeof(lastStats.multivector.exactKernel));
	strlcpy(lastStats.multivector.accumulatorKind,
			denseStats.multivector.accumulatorKind,
			sizeof(lastStats.multivector.accumulatorKind));
	lastStats.multivector.memoryBytesEstimate =
		denseStats.multivector.memoryBytesEstimate;
	lastStats.multivector.admissionDebugEnabled =
		denseStats.multivector.admissionDebugEnabled;
	lastStats.multivector.admissionCandidatesBeforeRerank =
		denseStats.multivector.admissionCandidatesBeforeRerank;
	lastStats.multivector.admissionCandidatesAfterTruncation =
		denseStats.multivector.admissionCandidatesAfterTruncation;
	lastStats.multivector.admissionExactRerankDocs =
		denseStats.multivector.admissionExactRerankDocs;
	lastStats.multivector.admissionTruncatedByDocCandidateK =
		denseStats.multivector.admissionTruncatedByDocCandidateK;
	lastStats.multivector.admissionTruncatedByAccumulatorMemory =
		denseStats.multivector.admissionTruncatedByAccumulatorMemory;
	lastStats.multivector.admissionTraceAvailable =
		denseStats.multivector.admissionTraceAvailable;
	lastStats.multivector.admissionTraceCount =
		denseStats.multivector.admissionTraceCount;
	if (lastStats.multivector.admissionTraceCount >
		PGTURBOHYBRID_MULTIVECTOR_DEBUG_TRACE_LIMIT_MAX)
		lastStats.multivector.admissionTraceCount =
			PGTURBOHYBRID_MULTIVECTOR_DEBUG_TRACE_LIMIT_MAX;
	if (lastStats.multivector.admissionTraceCount > 0)
		memcpy(lastStats.multivector.admissionTrace,
			   denseStats.multivector.admissionTrace,
			   sizeof(PgturbohybridMultiVectorAdmissionTraceEntry) *
			   lastStats.multivector.admissionTraceCount);
	lastStats.multivector.tokenStatsAvailable =
		denseStats.multivector.tokenStatsAvailable;
	lastStats.multivector.tokenStatsCount =
		denseStats.multivector.tokenStatsCount;
	if (lastStats.multivector.tokenStatsCount >
		PGTURBOHYBRID_MULTIVECTOR_TOKEN_STATS_LIMIT_MAX)
		lastStats.multivector.tokenStatsCount =
			PGTURBOHYBRID_MULTIVECTOR_TOKEN_STATS_LIMIT_MAX;
	if (lastStats.multivector.tokenStatsCount > 0)
		memcpy(lastStats.multivector.tokenStats,
			   denseStats.multivector.tokenStats,
			   sizeof(PgturbohybridMultiVectorTokenStatsEntry) *
			   lastStats.multivector.tokenStatsCount);
	lastStats.bm25.terms = bm25Stats.queryTerms;
	lastStats.bm25.postingsDecoded = bm25Stats.postingsDecoded;
	lastStats.bm25.blocksVisited = bm25Stats.blocksVisited;
	lastStats.bm25.blocksSkipped = bm25Stats.blocksSkipped;
	lastStats.bm25.fusedScoreBoundBlocksPruned =
		bm25Stats.fusedScoreBoundBlocksPruned;
	lastStats.bm25.fusedScoreBoundCandidatesPruned =
		bm25Stats.fusedScoreBoundCandidatesPruned;
	lastStats.bm25.candidatesScored = bm25Stats.candidatesScored;
	lastStats.bm25.heapTSVectorRerankMode = bm25Stats.heapTsvectorRerankMode;
	lastStats.bm25.heapTSVectorRerankCount =
		bm25Stats.heapTsvectorRerankCount;
	lastStats.bm25.heapTSVectorRerankFetchUs =
		bm25Stats.heapTsvectorRerankFetchUs;
	lastStats.bm25.heapTSVectorRerankScoreUs =
		bm25Stats.heapTsvectorRerankScoreUs;
	lastStats.bm25.heapTSVectorRerankTopKChanged =
		bm25Stats.heapTsvectorRerankTopKChanged;
	lastStats.bm25.cacheBytes = bm25Stats.cacheBytes;
	lastStats.bm25.cacheLexiconEntries = bm25Stats.cacheLexiconEntries;
	lastStats.bm25.cacheHit = bm25Stats.cacheHit;
	lastStats.bm25.cacheBuildUs = bm25Stats.cacheBuildUs;
	lastStats.bm25.cacheDocstatsLoaded = bm25Stats.cacheDocstatsLoaded;
	lastStats.bm25.cacheLivenessLoaded = bm25Stats.cacheLivenessLoaded;
	lastStats.bm25.docstatsLoadedThisQuery =
		bm25Stats.docstatsLoadedThisQuery;
	lastStats.bm25.livenessLoadedThisQuery =
		bm25Stats.livenessLoadedThisQuery;
	lastStats.bm25.docstatsBytes = bm25Stats.docstatsBytes;
	lastStats.bm25.livenessBytes = bm25Stats.livenessBytes;
	lastStats.bm25.coldCacheONWork = bm25Stats.coldCacheONWork;
	lastStats.bm25.postingsDecodeRatio = bm25Stats.postingsDecodeRatio;
	lastStats.bm25.commonTermFallback = bm25Stats.commonTermFallback;
	lastStats.bm25.wandPruned = bm25Stats.wandPruned;
	lastStats.bm25.hotPostingsCacheHits = bm25Stats.hotPostingsCacheHits;
	lastStats.bm25.hotPostingsCacheMisses = bm25Stats.hotPostingsCacheMisses;
	lastStats.bm25.hotPostingsCacheBytes = bm25Stats.hotPostingsCacheBytes;
	lastStats.bm25.hotPostingsCacheEvictions =
		bm25Stats.hotPostingsCacheEvictions;
	lastStats.bm25.deltaLookupMode = bm25Stats.deltaLookupMode;
	lastStats.bm25.deltaPagesScanned = bm25Stats.deltaPagesScanned;
	lastStats.bm25.deltaTermPagesRead = bm25Stats.deltaTermPagesRead;
	lastStats.bm25.deltaBlocksVisited = bm25Stats.deltaBlocksVisited;
	lastStats.bm25.deltaPostingsDecoded = bm25Stats.deltaPostingsDecoded;
	lastStats.bm25.deltaCacheBytes = bm25Stats.deltaCacheBytes;
	lastStats.bm25.deltaCacheTerms = bm25Stats.deltaCacheTerms;
	lastStats.bm25.deltaCacheHit = bm25Stats.deltaCacheHit;
	lastStats.bm25.wandIterations = bm25Stats.wandIterations;
	lastStats.bm25.wandThresholdUpdates = bm25Stats.wandThresholdUpdates;
	lastStats.bm25.wandActiveSorts = bm25Stats.wandActiveSorts;
	lastStats.bm25.wandHeapUpdates = bm25Stats.wandHeapUpdates;
	lastStats.bm25.wandFullReorders = bm25Stats.wandFullReorders;
	lastStats.bm25.wandBoundTighteningHits = bm25Stats.wandBoundTighteningHits;
	lastStats.bm25.wandBoundType = bm25Stats.wandBoundType;
	lastStats.bm25.wandHeapReplacements = bm25Stats.wandHeapReplacements;
	lastStats.bm25.strategy = bm25Stats.strategy;
	lastStats.bm25.andDriverDf = bm25Stats.andDriverDf;
	lastStats.bm25.andVerifiedCandidates = bm25Stats.andVerifiedCandidates;
	lastStats.bm25.andRejectedCandidates = bm25Stats.andRejectedCandidates;
	lastStats.bm25.impactTerms = bm25Stats.impactTerms;
	lastStats.bm25.impactTiersRead = bm25Stats.impactTiersRead;
	lastStats.bm25.impactPostingsRead = bm25Stats.impactPostingsRead;
	lastStats.bm25.impactRemainingUpperBound =
		bm25Stats.impactRemainingUpperBound;
	lastStats.bm25.impactEarlyStop = bm25Stats.impactEarlyStop;
	lastStats.bm25.impactExactSafe = bm25Stats.impactExactSafe;
	lastStats.bm25.impactFullPostingsAvoided =
		bm25Stats.impactFullPostingsAvoided;
	lastStats.bm25.impactLoadedFromStorage =
		bm25Stats.impactLoadedFromStorage;
	lastStats.bm25.impactBuiltLazily = bm25Stats.impactBuiltLazily;
	lastStats.bm25.impactLazyPostingsScanned =
		bm25Stats.impactLazyPostingsScanned;
	lastStats.bm25.accumulatorMode = bm25Stats.accumulatorMode;
	lastStats.bm25.accumulatorHashLookups = bm25Stats.accumulatorHashLookups;
	lastStats.bm25.accumulatorDenseUpdates = bm25Stats.accumulatorDenseUpdates;
	lastStats.bm25.finalHeapReplacements = bm25Stats.finalHeapReplacements;
	lastStats.bm25.finalSortedCount = bm25Stats.finalSortedCount;
	lastStats.bm25.fullSortAvoided = bm25Stats.fullSortAvoided;
	lastStats.bm25.queryShape = bm25Stats.queryShape;
	lastStats.bm25.booleanEvalMode = bm25Stats.booleanEvalMode;
	lastStats.bm25.booleanEvalCalls = bm25Stats.booleanEvalCalls;
	lastStats.bm25.decodeKernel = bm25Stats.decodeKernel;
	lastStats.bm25.scoreKernel = bm25Stats.scoreKernel;
	lastStats.bm25.simdBlocks = bm25Stats.simdBlocks;
	lastStats.bm25.scalarTailPostings = bm25Stats.scalarTailPostings;
	lastStats.bm25.prefetches = bm25Stats.prefetches;
	lastStats.fastWeighted.enabled =
		effectiveFusion == PGTURBOHYBRID_FUSION_FAST_WEIGHTED;
	lastStats.fastWeighted.alpha =
		(scanQuery->flags & PGTURBOHYBRID_QUERY_FLAG_ALPHA_IS_SET) != 0 ?
		scanQuery->alpha : 0.5;
	lastStats.calibratedFusion.enabled =
		effectiveFusion == PGTURBOHYBRID_FUSION_CALIBRATED;
	lastStats.dbsf.enabled =
		effectiveFusion == PGTURBOHYBRID_FUSION_DBSF;
	strlcpy(lastStats.bm25.normMode,
			lastStats.dbsf.enabled ? "dbsf" :
			((lastStats.fastWeighted.enabled || lastStats.calibratedFusion.enabled) ?
			 "saturating" : "none"),
			sizeof(lastStats.bm25.normMode));
	strlcpy(lastStats.dense.normMode,
			lastStats.dbsf.enabled ? "dbsf" :
			((lastStats.fastWeighted.enabled || lastStats.calibratedFusion.enabled) ?
			 "logistic" : "none"),
			sizeof(lastStats.dense.normMode));
	strlcpy(lastStats.hybrid.budgetPolicy,
			PgturbohybridHybridBudgetPolicyName(pgturbohybrid_hybrid_budget_policy),
			sizeof(lastStats.hybrid.budgetPolicy));
	strlcpy(lastStats.hybrid.queryShape,
			PgturbohybridHybridQueryShapeName(hybridQueryShapeForStats),
			sizeof(lastStats.hybrid.queryShape));
	strlcpy(lastStats.calibratedFusion.queryShape,
			PgturbohybridHybridQueryShapeName(hybridQueryShapeForStats),
			sizeof(lastStats.calibratedFusion.queryShape));
	lastStats.hybrid.denseKChosen = denseBranchUsed ? scanQuery->denseK : 0;
	lastStats.hybrid.bm25KChosen = scanQuery->bm25K;
	strlcpy(lastStats.hybrid.budgetReason, hybridBudgetChoice.reason,
			sizeof(lastStats.hybrid.budgetReason));
	PgturbohybridBuildBranchPlan(&lastStats.branchPlan, scanQuery,
								 effectiveFusion,
								 (hasMultivectorQuery && !multivectorMergedAsDense) ?
								 multivector : dense,
								 (hasMultivectorQuery && !multivectorMergedAsDense) ?
								 multivectorCount : denseCount,
								 hasMultivectorQuery,
								 bm25, bm25Count, &denseStats,
								 hasMultivectorQuery ? multivectorElapsedUs :
								 lastStats.dense.elapsedUs,
								 lastStats.bm25.elapsedUs);
	lastStats.elapsedUs = PgturbohybridElapsedUs(totalStart);
	pgturbohybrid_last_scan_state = lastStats;
	state->query = originalQuery;
	MemoryContextSwitchTo(oldCtx);
}

static IndexBuildResult *
pgturbohybridambuild(Relation heap, Relation index, IndexInfo *indexInfo)
{
	IndexBuildResult *result;

	PgturbohybridValidateIndex(index, indexInfo);
	if (PgturbohybridSparseIsPrimary(index))
		result = PgturbohybridSparsePrimaryBuild(heap, index, indexInfo);
	else
		result = tqgraphbuild(heap, index, indexInfo);

	if (PgturbohybridIndexInfoHasLexical(index, indexInfo))
		PgturbohybridBm25BuildCollect(heap, index, indexInfo);

	/* No-op unless the index has a sparse-vector key (self-guarded). */
	PgturbohybridSparseBuildCollect(heap, index, indexInfo);

	return result;
}

static void
pgturbohybridambuildempty(Relation index)
{
	PgturbohybridValidateIndex(index, NULL);
	if (PgturbohybridSparseIsPrimary(index))
		PgturbohybridSparsePrimaryBuildEmpty(index);
	else
		tqgraphbuildempty(index);
	if (PgturbohybridIndexHasLexical(index))
		PgturbohybridBm25BuildEmpty(index);
}

static bool
pgturbohybridaminsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid, Relation heap, IndexUniqueCheck checkUnique
#if PG_VERSION_NUM >= 140000
			   , bool indexUnchanged
#endif
			   , IndexInfo *indexInfo)
{
	Datum		value;
	Datum		lexicalValue;
	Datum		sparseValue;
	const PgturbohybridGraphTypeInfo *typeInfo;
	PgturbohybridGraphSupport support;
	uint32		nodeId;

	(void) heap;
	(void) checkUnique;
#if PG_VERSION_NUM >= 140000
	(void) indexUnchanged;
#endif
	PgturbohybridValidateIndex(index, indexInfo);

	/*
	 * Sparse-primary: no dense graph.  Allocate a node_id from the
	 * node-map chain (the heap TID is recorded there) and append the sparse /
	 * BM25 deltas keyed on it.  Liveness comes from heap MVCC visibility.
	 */
	if (PgturbohybridSparseIsPrimary(index))
	{
		PgturbohybridIndexKeyMap map;

		PgturbohybridBuildIndexKeyMap(index, indexInfo, &map);
		if (isnull[map.primaryKey])
			return false;

		LockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, ExclusiveLock);
		PG_TRY();
		{
			nodeId = PgturbohybridSparsePrimaryInsert(index, heap_tid);
			if (PgturbohybridIndexGetLexicalDatum(index, values, isnull, &lexicalValue))
				PgturbohybridBm25AppendDelta(index, nodeId, heap_tid, lexicalValue);
			if (PgturbohybridIndexGetSparseDatum(index, values, isnull, &sparseValue))
			{
				PgturbohybridSparseAppendDelta(index, nodeId, heap_tid, sparseValue);
				PgturbohybridSparseMaybeCompact(index, false);
			}
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

	if (isnull[0])
		return false;

	if (PgturbohybridIndexIsMultiVector(index))
	{
		uint32		insertedNodes = 0;

		LockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, ExclusiveLock);
		PG_TRY();
		{
				nodeId = PgturbohybridGraphInsertMultiVectorBatchInPlace(index, indexInfo,
																		 heap_tid,
																		 values[0],
																		 values,
																		 isnull,
																		 &insertedNodes);
			if (insertedNodes > 0 &&
				PgturbohybridIndexGetLexicalDatum(index, values, isnull, &lexicalValue))
				PgturbohybridBm25AppendDelta(index, nodeId, heap_tid, lexicalValue);
			if (insertedNodes > 0 &&
				PgturbohybridIndexGetSparseDatum(index, values, isnull, &sparseValue))
			{
				PgturbohybridSparseAppendDelta(index, nodeId, heap_tid, sparseValue);
				PgturbohybridSparseMaybeCompact(index, false);
			}
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

	typeInfo = PgturbohybridGraphGetTypeInfo(index);
	PgturbohybridGraphInitSupport(&support, index);
	if (!PgturbohybridGraphFormIndexValue(&value, values, isnull, typeInfo, &support))
		return false;

	LockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, ExclusiveLock);
	PG_TRY();
	{
		nodeId = PgturbohybridGraphInsertValueInPlace(index, indexInfo, heap_tid,
													  value, values, isnull);
		if (PgturbohybridIndexGetLexicalDatum(index, values, isnull, &lexicalValue))
			PgturbohybridBm25AppendDelta(index, nodeId, heap_tid, lexicalValue);
		if (PgturbohybridIndexGetSparseDatum(index, values, isnull, &sparseValue))
		{
			PgturbohybridSparseAppendDelta(index, nodeId, heap_tid, sparseValue);
			PgturbohybridSparseMaybeCompact(index, false);
		}
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
pgturbohybridambulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats, IndexBulkDeleteCallback callback, void *callback_state)
{
	IndexBulkDeleteResult *result;

	/*
	 * Sparse-primary indexes have no flat/graph element storage to prune.  Node
	 * liveness is delegated to heap-tuple MVCC visibility (the executor filters
	 * dead TIDs), so bulkdelete is a no-op beyond cache invalidation; dead
	 * node-map / posting entries are reclaimed by REINDEX or sparse compaction.
	 */
	if (PgturbohybridSparseIsPrimary(info->index))
	{
		if (stats == NULL)
			stats = (IndexBulkDeleteResult *) palloc0(sizeof(IndexBulkDeleteResult));
		PgturbohybridBm25InvalidateCache(info->index);
		PgturbohybridSparseCacheInvalidate(RelationGetRelid(info->index));
		return stats;
	}

	result = tqgraphbulkdelete(info, stats, callback, callback_state);
	PgturbohybridBm25InvalidateCache(info->index);
	PgturbohybridSparseCacheInvalidate(RelationGetRelid(info->index));

	return result;
}

static IndexBulkDeleteResult *
pgturbohybridamvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats)
{
	IndexBulkDeleteResult *result;

	if (PgturbohybridSparseIsPrimary(info->index))
	{
		if (stats == NULL)
			stats = (IndexBulkDeleteResult *) palloc0(sizeof(IndexBulkDeleteResult));
		if (!info->estimated_count)
			stats->num_pages = RelationGetNumberOfBlocks(info->index);
		(void) PgturbohybridBm25MaybeCompact(info->index);
		PgturbohybridBm25InvalidateCache(info->index);
		PgturbohybridSparseCacheInvalidate(RelationGetRelid(info->index));
		return stats;
	}

	result = tqgraphvacuumcleanup(info, stats);
	(void) PgturbohybridBm25MaybeCompact(info->index);
	PgturbohybridBm25InvalidateCache(info->index);
	PgturbohybridSparseCacheInvalidate(RelationGetRelid(info->index));

	return result;
}

static IndexScanDesc
pgturbohybridambeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;

	PgturbohybridValidateIndex(index, NULL);
	if (PgturbohybridSparseIsPrimary(index))
	{
		PgturbohybridGraphScanOpaque so;

		scan = RelationGetIndexScan(index, nkeys, norderbys);
		so = palloc0(sizeof(PgturbohybridGraphScanOpaqueData));
		so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
										   "pgturbohybrid sparse-primary scan context",
										   0, 8 * 1024, 256 * 1024);
		so->efSearch = PgturbohybridGraphGetEfSearch(index);
		so->graphStorageKind = PGTURBOHYBRID_GRAPH_STORAGE_QUANT_GRAPH_NATIVE;
		so->first = true;
		so->previousDistance = -get_float8_infinity();
		scan->opaque = so;
	}
	else
		scan = tqgraphbeginscan(index, nkeys, norderbys);

	return scan;
}

static void
pgturbohybridamrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	ScanKey		denseOrderbys = PgturbohybridDenseOrderBys(orderbys, norderbys);
	PgturbohybridQueryHeader *hybridQuery = NULL;
	bool		hasTextQuery = false;
	bool		hasVectorQuery = false;
	bool		hasMultiVectorQuery = false;
	bool		hasSparseQuery = false;
	bool		useScalarVectorOrderby = false;

	if (orderbys != NULL && norderbys > 0 &&
		(orderbys[0].sk_flags & SK_ISNULL) == 0)
	{
		hybridQuery = (PgturbohybridQueryHeader *) PG_DETOAST_DATUM_COPY(orderbys[0].sk_argument);
		PgturbohybridQueryValidateFast(hybridQuery);
		hasTextQuery = (hybridQuery->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_TSQUERY) != 0;
		hasVectorQuery = (hybridQuery->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_VECTOR) != 0;
		hasMultiVectorQuery =
			(hybridQuery->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_MULTIVECTOR) != 0;
		hasSparseQuery =
			(hybridQuery->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_SPARSE) != 0;
	}

	if (PgturbohybridIndexIsMultiVector(scan->indexRelation))
	{
		if (hasVectorQuery)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("vector_query requires a turbohybrid index with a vector key"),
					 errdetail("Document-node multivector indexes can execute multivector_query and text_query in one scan, but cannot run an independent scalar vector branch."),
					 errhint("Use multivector_query for ColBERT retrieval on this index, or run scalar dense retrieval against a vector-key index and fuse results outside this scan.")));
	}
	else if (hasMultiVectorQuery)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("multivector_query requires a turbohybrid index with a multivector key")));
	else
		useScalarVectorOrderby = hasVectorQuery;

	PgturbohybridQueryValidateMultiVectorFusionSupport(scan->indexRelation,
													   hybridQuery);

	if (hasTextQuery && !PgturbohybridIndexHasLexical(scan->indexRelation))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("text_query requires a turbohybrid index with a tsvector key")));

	if (hasSparseQuery)
	{
		PgturbohybridIndexKeyMap sparseKeyMap;

		PgturbohybridBuildIndexKeyMap(scan->indexRelation, NULL, &sparseKeyMap);
		if (!sparseKeyMap.hasSparse)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("sparse_query requires a turbohybrid index with a turbohybrid_sparse_vector key"),
					 errhint("Add a turbohybrid_sparse_vector key to the index, or evaluate sparse_query with turbohybrid_query(...) for exact scoring without an index.")));
	}

	if (PgturbohybridSparseIsPrimary(scan->indexRelation))
	{
		PgturbohybridGraphScanOpaque so =
			(PgturbohybridGraphScanOpaque) scan->opaque;

		so->first = true;
		so->returnedRows = 0;
		so->previousDistance = -get_float8_infinity();
		MemoryContextReset(so->tmpCtx);
		if (keys && scan->numberOfKeys > 0)
			memmove(scan->keyData, keys,
					scan->numberOfKeys * sizeof(ScanKeyData));
		if (orderbys && scan->numberOfOrderBys > 0)
			memmove(scan->orderByData, orderbys,
					scan->numberOfOrderBys * sizeof(ScanKeyData));
	}
	else
		tqgraphrescan(scan, keys, nkeys,
					  useScalarVectorOrderby ? denseOrderbys : NULL,
					  useScalarVectorOrderby ? norderbys : 0);

	if (hasTextQuery || hasVectorQuery || hasMultiVectorQuery || hasSparseQuery)
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

bool
pgturbohybridamgettuple(IndexScanDesc scan, ScanDirection dir)
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

	result = PgturbohybridSparseIsPrimary(scan->indexRelation) ?
		false : tqgraphgettuple(scan, dir);

	return result;
}

static void
pgturbohybridamendscan(IndexScanDesc scan)
{
	if (PgturbohybridSparseIsPrimary(scan->indexRelation))
	{
		PgturbohybridGraphScanOpaque so =
			(PgturbohybridGraphScanOpaque) scan->opaque;

		if (so != NULL)
		{
			MemoryContextDelete(so->tmpCtx);
			pfree(so);
		}
		scan->opaque = NULL;
	}
	else
		tqgraphendscan(scan);
}

static bool
PgturbohybridPathHasFilter(IndexPath *path)
{
	int			denseAttno = path->indexinfo->indexkeys[0];
	AttrNumber	lexicalAttno = PgturbohybridPathLexicalAttno(path);
	bool		hasLexicalAttno = AttributeNumberIsValid(lexicalAttno);
	ListCell   *lc;

	foreach(lc, path->indexinfo->indrestrictinfo)
	{
		RestrictInfo *rinfo = lfirst_node(RestrictInfo, lc);
		Bitmapset  *attrs = NULL;

		pull_varattnos((Node *) rinfo->clause, path->indexinfo->rel->relid,
						&attrs);

		if (attrs == NULL)
			continue;

		if (bms_membership(attrs) != BMS_SINGLETON)
			return true;
		if (bms_is_member(denseAttno - FirstLowInvalidHeapAttributeNumber, attrs))
			continue;
		if (hasLexicalAttno &&
			bms_is_member(lexicalAttno - FirstLowInvalidHeapAttributeNumber, attrs))
			continue;

		/* Dense-only indexes have no lexical key; anything else is a heap filter. */
		return true;
	}

	return false;
}

static bool
PgturbohybridIndexHasPayloadAttr(Relation index, AttrNumber heapAttno)
{
	int			totalAttrs = IndexRelationGetNumberOfAttributes(index);
	int			keyAttrs = IndexRelationGetNumberOfKeyAttributes(index);

	for (int i = keyAttrs; i < totalAttrs; i++)
	{
		if (index->rd_index->indkey.values[i] == heapAttno)
			return true;
	}

	return false;
}

static bool
PgturbohybridClauseIsPayloadInt4Equality(Node *node, Index relid, AttrNumber heapAttno)
{
	OpExpr	   *op;
	Node	   *left;
	Node	   *right;
	Var		   *var = NULL;
	Const	   *constant = NULL;
	Oid			opfuncid;

	if (node == NULL || !IsA(node, OpExpr))
		return false;

	op = castNode(OpExpr, node);
	if (list_length(op->args) != 2)
		return false;

	opfuncid = op->opfuncid;
	if (!OidIsValid(opfuncid))
		opfuncid = get_opcode(op->opno);
	if (opfuncid != F_INT4EQ)
		return false;

	left = linitial(op->args);
	right = lsecond(op->args);

	if (IsA(left, Var) && IsA(right, Const))
	{
		var = castNode(Var, left);
		constant = castNode(Const, right);
	}
	else if (IsA(left, Const) && IsA(right, Var))
	{
		var = castNode(Var, right);
		constant = castNode(Const, left);
	}
	else
		return false;

	return var->varno == relid &&
		var->varattno == heapAttno &&
		var->vartype == INT4OID &&
		constant->consttype == INT4OID &&
		!constant->constisnull;
}

static bool
PgturbohybridPathHasUnmappedFilter(IndexPath *path, Relation index)
{
	int			denseAttno = path->indexinfo->indexkeys[0];
	AttrNumber	lexicalAttno = PgturbohybridPathLexicalAttno(path);
	bool		hasLexicalAttno = AttributeNumberIsValid(lexicalAttno);
	Index		relid = path->indexinfo->rel->relid;
	ListCell   *lc;

	foreach(lc, path->indexinfo->indrestrictinfo)
	{
		RestrictInfo *rinfo = lfirst_node(RestrictInfo, lc);
		Bitmapset  *attrs = NULL;
		int			attrOffset;
		AttrNumber	heapAttno;

		pull_varattnos((Node *) rinfo->clause, relid, &attrs);

		if (attrs == NULL)
			continue;

		if (bms_membership(attrs) != BMS_SINGLETON)
		{
			bms_free(attrs);
			return true;
		}

		attrOffset = bms_singleton_member(attrs);
		bms_free(attrs);
		heapAttno = attrOffset + FirstLowInvalidHeapAttributeNumber;

		if (heapAttno == denseAttno ||
			(hasLexicalAttno && heapAttno == lexicalAttno))
			continue;

		/*
		 * Runtime can use graph payload refs only for simple int4 equality
		 * predicates on INCLUDE payload columns.  Everything else is costed as
		 * an unmapped heap filter because it may force dense collection toward
		 * the full scan budget before the executor can recheck the predicate.
		 */
		if (PgturbohybridIndexHasPayloadAttr(index, heapAttno) &&
			PgturbohybridClauseIsPayloadInt4Equality((Node *) rinfo->clause,
										 relid, heapAttno))
			continue;

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
			*query = DatumGetPgturbohybridQuery(constant->constvalue);
			PgturbohybridQueryValidateFast(*query);
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
pgturbohybridamcostestimate(PlannerInfo *root, IndexPath *path, double loop_count,
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
	double		nodeCount;
	double		filterSelectivity;
	double		denseCandidateWork = 0.0;
	double		denseWork = 0.0;
	double		bm25Postings = 0.0;
	double		bm25Work = 0.0;
	double		nativeCacheWork = 0.0;
	double		bm25ColdCacheWork = 0.0;
	double		rescoreWork = 0.0;
	double		fusionWork;
	double		fusionInput;
	double		estimatedPages;
	double		spc_random_page_cost;
	double		spc_seq_page_cost;
	double		pageCost;
	double		cpuCost;
	double		totalWork;
	bool		hasHeapFilter = false;
	bool		hasUnmappedFilter = false;
	bool		hasLexicalKey;
	bool		haveGraphMemory = false;
	PgturbohybridGraphMemoryEstimate graphMemory;

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
	MemSet(&graphMemory, 0, sizeof(graphMemory));
	query = PgturbohybridFindConstQuery(path->indexorderbys);

	index = index_open(path->indexinfo->indexoid, NoLock);
	hasLexicalKey = PgturbohybridIndexHasLexical(index);
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
		denseK = (PgturbohybridQueryGetVector(query) != NULL ? query->denseK : 0) +
			(PgturbohybridQueryGetMultiVector(query) != NULL ? query->multivectorK : 0);
		bm25K = hasLexicalKey && PgturbohybridQueryGetTsQuery(query) != NULL ?
			query->bm25K : 0;
		finalK = (query->flags & PGTURBOHYBRID_QUERY_FLAG_FINAL_K_IS_SET) != 0 ?
			query->finalK : Max(denseK + bm25K, 1);
		termCount = PgturbohybridEstimateTsQueryTerms(PgturbohybridQueryGetTsQuery(query));
	}
	else
	{
		denseK = opts != NULL ? opts->hybridDefaultDenseK : PGTURBOHYBRID_DEFAULT_DENSE_K;
		bm25K = 0;
		finalK = Max(denseK, 1);
		termCount = 2;
	}
	/*
	 * Sparse-primary indexes have no dense/multivector graph.  When the ORDER BY
	 * query could not be const-folded (query == NULL), denseK defaults to a
	 * positive value above; force it to zero here so the dense-graph work and
	 * memory estimation (which assume a populated graph) are skipped.
	 */
	if (PgturbohybridSparseIsPrimary(index))
		denseK = 0;
	if (bm25K > 0)
		(void) PgturbohybridBm25GetPlanningStats(index, &bm25Stats);
	hasHeapFilter = PgturbohybridPathHasFilter(path);
	if (hasHeapFilter)
		hasUnmappedFilter = PgturbohybridPathHasUnmappedFilter(path, index);
	if (denseK > 0)
		haveGraphMemory = PgturbohybridGraphEstimateMemory(index, &graphMeta,
														   &graphMemory);

	tuples = Max(path->indexinfo->tuples, 1.0);
	nodeCount = graphMeta.tqNodeCount > 0 ? graphMeta.tqNodeCount : tuples;
	m = graphMeta.m > 0 ? graphMeta.m :
		(opts != NULL ? opts->m : PGTURBOHYBRID_GRAPH_DEFAULT_M);
	efSearch = graphMeta.graphEfSearch > 0 ? graphMeta.graphEfSearch :
		(opts != NULL ? opts->graphEfSearch : PGTURBOHYBRID_DEFAULT_GRAPH_EF_SEARCH);
	graphOversampling = graphMeta.graphOversampling > 0 ?
		graphMeta.graphOversampling :
		(opts != NULL ? opts->graphOversampling : PGTURBOHYBRID_DEFAULT_GRAPH_OVERSAMPLING);
	index_close(index, NoLock);

	filterSelectivity = costs.indexSelectivity;
	if (path->path.parent != NULL && path->path.parent->tuples > 0 &&
		path->path.rows >= 0)
		filterSelectivity = path->path.rows / path->path.parent->tuples;
	filterSelectivity = Max(Min(filterSelectivity, 1.0), 0.001);

	if (denseK > 0)
	{
		double		layer0Work = Max(efSearch, denseK * graphOversampling);
		double		entryWork = Max(1.0, log(tuples)) * Max(m, 1.0);
		double		dimensions = graphMeta.dimensions > 0 ?
			graphMeta.dimensions : 128.0;
		double		rescoreBand = 0.0;

		denseCandidateWork = layer0Work;

		if (hasHeapFilter)
		{
			double		selectiveTarget =
				ceil(Max(finalK, denseK) / filterSelectivity) *
				Max(graphOversampling, 1.0);

			if (hasUnmappedFilter)
			{
				double		scanCap = pgturbohybrid_max_scan_tuples > 0 ?
					(double) pgturbohybrid_max_scan_tuples : nodeCount;

				/*
				 * Runtime PgturbohybridGraphCollectResults() widens unmapped
				 * heap filters to min(N, max_scan_tuples) because the graph
				 * cannot evaluate the predicate during traversal.
				 */
				denseCandidateWork = Max(denseCandidateWork,
										 Min(nodeCount, scanCap));
			}
			else
			{
				/*
				 * Payload-owned int4 equality filters can use graph payload
				 * refs, but result collection still scales with the filtered
				 * band size when selectivity is low.
				 */
				denseCandidateWork = Max(denseCandidateWork,
										 Min(nodeCount, selectiveTarget));
			}
		}

		denseWork = entryWork + denseCandidateWork;

		/*
		 * Exact/heap rescore over a large final band is separate work after
		 * candidate collection.  The exact policies rescore the band; limited
		 * caps the band by the configured multiplier; large widened bands get
		 * a small extra term even under auto/off to reflect sorting and tuple
		 * preparation around the fallback path.
		 */
		if (pgturbohybrid_dense_rescore_band_policy ==
			PGTURBOHYBRID_RESCORE_BAND_POLICY_EXACT ||
			pgturbohybrid_dense_heap_rescore ==
			PGTURBOHYBRID_DENSE_HEAP_RESCORE_BAND)
			rescoreBand = denseCandidateWork;
		else if (pgturbohybrid_dense_rescore_band_policy ==
				 PGTURBOHYBRID_RESCORE_BAND_POLICY_LIMITED)
			rescoreBand = Min(denseCandidateWork,
							  Max(denseK, finalK) *
							  Max(pgturbohybrid_dense_max_rescore_multiplier, 1));
		else if (pgturbohybrid_dense_heap_rescore ==
				 PGTURBOHYBRID_DENSE_HEAP_RESCORE_TOPK)
			rescoreBand = Max(finalK, 1.0);
		else if (denseCandidateWork > layer0Work * 2.0)
			rescoreBand = (denseCandidateWork - layer0Work) * 0.25;

		rescoreWork = rescoreBand * Max(dimensions / 64.0, 1.0);

		if (haveGraphMemory &&
			graphMemory.effectiveCachePolicy != PGTURBOHYBRID_NATIVE_CACHE_POLICY_OFF &&
			graphMemory.estimatedTotalBytes > 0)
		{
			/*
			 * A cold native cache build copies O(N) resident code/node/adjacency
			 * state before warm scans become cheap.  Keep this as a bounded
			 * startup surcharge so ordinary warm top-k plans stay attractive.
			 */
			nativeCacheWork = Min(nodeCount * 0.05,
								  (double) graphMemory.estimatedTotalBytes /
								  (double) BLCKSZ);
		}
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
		{
			double		dfRatio = bm25Postings / Max(corpusDocs, 1.0);

			/*
			 * Common terms make WAND/impact pruning less selective.  Use the
			 * normal discount for sparse terms, but retain more postings work
			 * when average df is a large fraction of the corpus.
			 */
			bm25Postings *= dfRatio >= 0.10 ? 0.65 : 0.35;
		}

		/*
		 * Cold BM25 queries may populate docLens/heapTids/liveNodes arrays of
		 * size N before postings scoring.  The planner cannot know cache
		 * warmth, so this is a small O(N) startup term rather than a full scan
		 * replacement.
		 */
		if (bm25Stats.hasBm25)
			bm25ColdCacheWork = corpusDocs * 0.04;

		bm25Work = bm25Postings + bm25K * Max(termCount, 1.0);
		if (hasHeapFilter)
			bm25Work *= Min(4.0, 1.0 / Max(filterSelectivity, 0.05));
	}

	fusionInput = Max(finalK, Max(denseCandidateWork, denseK) + bm25K);
	if (pgturbohybrid_max_union_candidates > 0)
		fusionInput = Min(fusionInput, (double) pgturbohybrid_max_union_candidates);
	fusionWork = fusionInput;
	if (fusionWork > 1)
		fusionWork *= log(fusionWork) / log(2.0);

	totalWork = denseWork + bm25Work + nativeCacheWork + bm25ColdCacheWork +
		rescoreWork + fusionWork;
	get_tablespace_page_costs(path->indexinfo->reltablespace,
							  &spc_random_page_cost, &spc_seq_page_cost);
	estimatedPages = Min(costs.numIndexPages,
						 2.0 + denseWork / 256.0 + bm25Work / 512.0 +
						 nativeCacheWork / 128.0 +
						 bm25ColdCacheWork / 256.0 +
						 rescoreWork / 512.0);
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
pgturbohybridamoptions(Datum reloptions, bool validate)
{
#if PG_VERSION_NUM >= 180000 && PG_VERSION_NUM < 190000
#define PGTURBOHYBRID_RELOPT_PARSE(name, type, field) \
	{name, type, offsetof(PgturbohybridOptions, field), 0}
#else
#define PGTURBOHYBRID_RELOPT_PARSE(name, type, field) \
	{name, type, offsetof(PgturbohybridOptions, field)}
#endif
	static const relopt_parse_elt tab[] = {
		PGTURBOHYBRID_RELOPT_PARSE("graph_m", RELOPT_TYPE_INT, m),
		PGTURBOHYBRID_RELOPT_PARSE("graph_ef_construction", RELOPT_TYPE_INT, efConstruction),
		PGTURBOHYBRID_RELOPT_PARSE("graph_ef_search", RELOPT_TYPE_INT, graphEfSearch),
		PGTURBOHYBRID_RELOPT_PARSE("graph_oversampling", RELOPT_TYPE_INT, graphOversampling),
		PGTURBOHYBRID_RELOPT_PARSE("native_segments", RELOPT_TYPE_ENUM, nativeSegments),
		PGTURBOHYBRID_RELOPT_PARSE("quantization_bits", RELOPT_TYPE_INT, tqBits),
		PGTURBOHYBRID_RELOPT_PARSE("exact_storage", RELOPT_TYPE_BOOL, tqExactStorage),
		PGTURBOHYBRID_RELOPT_PARSE("entry_sidecar", RELOPT_TYPE_BOOL, entrySidecar),
		PGTURBOHYBRID_RELOPT_PARSE("entry_sidecar_representatives", RELOPT_TYPE_INT, entrySidecarRepresentatives),
		PGTURBOHYBRID_RELOPT_PARSE("entry_sidecar_strategy", RELOPT_TYPE_ENUM, entrySidecarStrategy),
		PGTURBOHYBRID_RELOPT_PARSE("graph_backbone", RELOPT_TYPE_BOOL, graphBackbone),
		PGTURBOHYBRID_RELOPT_PARSE("residual_rerank", RELOPT_TYPE_BOOL, residualRerank),
		PGTURBOHYBRID_RELOPT_PARSE("residual_rerank_bytes", RELOPT_TYPE_INT, residualRerankBytes),
		PGTURBOHYBRID_RELOPT_PARSE("multivector_graph", RELOPT_TYPE_ENUM, multivectorGraphMode),
		PGTURBOHYBRID_RELOPT_PARSE("multivector_doc_build_scorer", RELOPT_TYPE_ENUM, multivectorDocBuildScorer),
		PGTURBOHYBRID_RELOPT_PARSE("multivector_doc_storage", RELOPT_TYPE_ENUM, multivectorDocStorage),
		PGTURBOHYBRID_RELOPT_PARSE("multivector_token_pooling", RELOPT_TYPE_ENUM, multivectorTokenPooling),
		PGTURBOHYBRID_RELOPT_PARSE("multivector_token_pooling_target_ratio", RELOPT_TYPE_REAL, multivectorTokenPoolingTargetRatio),
		PGTURBOHYBRID_RELOPT_PARSE("multivector_token_pooling_min_tokens", RELOPT_TYPE_INT, multivectorTokenPoolingMinTokens),
		PGTURBOHYBRID_RELOPT_PARSE("multivector_centroids", RELOPT_TYPE_ENUM, multivectorCentroids),
		PGTURBOHYBRID_RELOPT_PARSE("multivector_centroid_count", RELOPT_TYPE_INT, multivectorCentroidCount),
		PGTURBOHYBRID_RELOPT_PARSE("multivector_proxy_encoder", RELOPT_TYPE_ENUM, multivectorProxyEncoder),
		PGTURBOHYBRID_RELOPT_PARSE("multivector_context_mode", RELOPT_TYPE_ENUM, multivectorContextMode),
		PGTURBOHYBRID_RELOPT_PARSE("multivector_field_mode", RELOPT_TYPE_ENUM, multivectorFieldMode),
		PGTURBOHYBRID_RELOPT_PARSE("sparse_quant_bits", RELOPT_TYPE_INT, sparseQuantBits),
		PGTURBOHYBRID_RELOPT_PARSE("sparse_quant_mode", RELOPT_TYPE_ENUM, sparseQuantMode),
		PGTURBOHYBRID_RELOPT_PARSE("sparse_postings_encoding", RELOPT_TYPE_ENUM, sparsePostingsEncoding),
		PGTURBOHYBRID_RELOPT_PARSE("sparse_block_size", RELOPT_TYPE_INT, sparseBlockSize),
		PGTURBOHYBRID_RELOPT_PARSE("sparse_block_max", RELOPT_TYPE_BOOL, sparseBlockMax),
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
		opts->hybridDefaultFusion = PGTURBOHYBRID_FUSION_RRF;
		opts->hybridDefaultDenseK = PGTURBOHYBRID_DEFAULT_DENSE_K;
		opts->hybridDefaultBm25K = PGTURBOHYBRID_DEFAULT_BM25_K;
		opts->hybridDefaultRrfK = PGTURBOHYBRID_DEFAULT_RRF_K;
	}

	if (validate && opts != NULL &&
		opts->tqBits != 1 && opts->tqBits != 2 &&
		opts->tqBits != PGTURBOHYBRID_DEFAULT_BITS &&
		opts->tqBits != 8)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid value %d for option \"quantization_bits\"", opts->tqBits),
				 errdetail("Valid values are \"1\", \"2\", \"4\", and \"8\".")));

	if (validate && opts != NULL &&
		opts->sparseQuantBits != 0 && opts->sparseQuantBits != 8 &&
		opts->sparseQuantBits != 16)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid value %d for option \"sparse_quant_bits\"", opts->sparseQuantBits),
				 errdetail("Valid values are \"0\", \"8\", and \"16\".")));

	return (bytea *) opts;
}

static bool
pgturbohybridamvalidate(Oid opclassoid)
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
	else if (strcmp(opcname, "multivector_cosine_turbohybrid_ops") == 0 ||
			 strcmp(opcname, "multivector_maxsim_ip_turbohybrid_ops") == 0)
		valid = opclass->opcintype == PgturbohybridMultiVectorTypeOid();
	else if (strcmp(opcname, "bm25_tsvector_turbohybrid_ops") == 0)
		valid = opclass->opcintype == TSVECTOROID;
	else if (strcmp(opcname, "sparse_ip_turbohybrid_ops") == 0)
		valid = opclass->opcintype == PgturbohybridSparseVectorTypeOid();

	ReleaseSysCache(opclasstuple);
	return valid;
}

void
PgturbohybridInit(void)
{
	bool		gucsAlreadyDefined;

	if (pgturbohybrid_am_init_done)
		return;
	pgturbohybrid_am_init_done = true;

	pgturbohybrid_relopt_kind = add_reloption_kind();
	RegisterXactCallback(PgturbohybridXactCallback, NULL);
	RegisterSubXactCallback(PgturbohybridSubXactCallback, NULL);

	add_int_reloption(pgturbohybrid_relopt_kind, "graph_m", "Max number of graph connections",
					  PGTURBOHYBRID_DEFAULT_GRAPH_M, PGTURBOHYBRID_GRAPH_MIN_M, PGTURBOHYBRID_GRAPH_MAX_M, AccessExclusiveLock);
	add_int_reloption(pgturbohybrid_relopt_kind, "graph_ef_construction", "Size of the dynamic graph candidate list for construction",
					  PGTURBOHYBRID_DEFAULT_GRAPH_EF_CONSTRUCTION, PGTURBOHYBRID_GRAPH_MIN_EF_CONSTRUCTION, PGTURBOHYBRID_GRAPH_MAX_EF_CONSTRUCTION, AccessExclusiveLock);
	add_int_reloption(pgturbohybrid_relopt_kind, "graph_ef_search", "Size of the dynamic graph candidate list for search",
					  PGTURBOHYBRID_DEFAULT_GRAPH_EF_SEARCH, PGTURBOHYBRID_GRAPH_MIN_EF_SEARCH, PGTURBOHYBRID_GRAPH_MAX_EF_SEARCH, AccessExclusiveLock);
	add_int_reloption(pgturbohybrid_relopt_kind, "graph_oversampling", "Candidate oversampling multiplier for graph scans",
					  PGTURBOHYBRID_DEFAULT_GRAPH_OVERSAMPLING, 1, 1000, AccessExclusiveLock);
	add_enum_reloption(pgturbohybrid_relopt_kind, "native_segments",
					   "Native graph segment count; 1 keeps the legacy single graph.",
					   pgturbohybrid_native_segments_relopt_options, 1,
					   "Valid values are \"1\", \"2\", \"4\", \"8\", and \"auto\".",
					   AccessExclusiveLock);
	add_int_reloption(pgturbohybrid_relopt_kind, "quantization_bits", "Quantized vector code bit width",
					  PGTURBOHYBRID_DEFAULT_INDEX_BITS, 1, 8, AccessExclusiveLock);
	add_bool_reloption(pgturbohybrid_relopt_kind, "exact_storage",
					   "Store exact vectors in the dense pgturbohybrid index for final exact rescoring.",
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
	add_enum_reloption(pgturbohybrid_relopt_kind, "multivector_graph",
					   "Multivector graph node storage mode.",
					   pgturbohybrid_multivector_graph_relopt_options,
					   PGTURBOHYBRID_DEFAULT_MULTIVECTOR_GRAPH_MODE,
					   "Valid values are \"token_nodes\" and \"document_nodes\". \"document_nodes\" stores one graph node per heap document with a versioned document multivector sidecar for MaxSim-aligned candidate generation.",
					   AccessExclusiveLock);
	add_enum_reloption(pgturbohybrid_relopt_kind, "multivector_doc_build_scorer",
					   "Document-node multivector graph build distance scorer.",
					   pgturbohybrid_multivector_doc_build_scorer_relopt_options,
					   PGTURBOHYBRID_DEFAULT_MULTIVECTOR_DOC_BUILD_SCORER,
					   "Valid values are \"proxy\" and \"exact_symmetric\". Only meaningful with multivector_graph = document_nodes.",
					   AccessExclusiveLock);
	add_enum_reloption(pgturbohybrid_relopt_kind, "multivector_doc_storage",
					   "Document-node multivector sidecar storage mode.",
					   pgturbohybrid_multivector_doc_storage_relopt_options,
					   PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32,
		"Valid values are \"f32\", \"f16\", \"sq8\", experimental \"centroid_only\", and experimental \"proxy_only\". \"centroid_only\" stores centroid sidecars for centroid_lite with heap exact rerank; \"proxy_only\" stores graph proxy vectors and doc mapping only.",
					   AccessExclusiveLock);
	add_enum_reloption(pgturbohybrid_relopt_kind, "multivector_token_pooling",
					   "Index-time document-token pooling mode for multivector document_nodes indexes.",
					   pgturbohybrid_multivector_token_pooling_relopt_options,
					   PGTURBOHYBRID_MULTIVECTOR_TOKEN_POOLING_OFF,
					   "Valid values are \"off\", \"kmeans\", and \"greedy_cosine\".",
					   AccessExclusiveLock);
	add_real_reloption(pgturbohybrid_relopt_kind, "multivector_token_pooling_target_ratio",
					   "Target pooled/original document-token ratio when multivector_token_pooling is enabled.",
					   PGTURBOHYBRID_MULTIVECTOR_TOKEN_POOLING_DEFAULT_TARGET_RATIO,
					   0.01, 1.0, AccessExclusiveLock);
	add_int_reloption(pgturbohybrid_relopt_kind, "multivector_token_pooling_min_tokens",
					  "Minimum document-token count before index-time multivector token pooling is applied.",
					  PGTURBOHYBRID_MULTIVECTOR_TOKEN_POOLING_DEFAULT_MIN_TOKENS,
					  1, PGTURBOHYBRID_MULTIVECTOR_MAX_COUNT, AccessExclusiveLock);
	add_enum_reloption(pgturbohybrid_relopt_kind, "multivector_centroids",
					   "Experimental PLAID-lite document centroid mode for multivector candidate generation.",
					   pgturbohybrid_multivector_centroids_relopt_options,
					   PGTURBOHYBRID_MULTIVECTOR_CENTROIDS_OFF,
					   "Valid values are \"off\" and \"kmeans\".",
					   AccessExclusiveLock);
	add_int_reloption(pgturbohybrid_relopt_kind, "multivector_centroid_count",
					  "Per-document centroid count for multivector_centroids; 0 selects an automatic count.",
					  PGTURBOHYBRID_MULTIVECTOR_CENTROID_COUNT_AUTO,
					  PGTURBOHYBRID_MULTIVECTOR_CENTROID_COUNT_AUTO,
					  PGTURBOHYBRID_MULTIVECTOR_CENTROID_COUNT_MAX,
					  AccessExclusiveLock);
	add_enum_reloption(pgturbohybrid_relopt_kind, "multivector_proxy_encoder",
					   "Fixed-dimensional proxy encoder for document-node proxy_vector admission.",
					   pgturbohybrid_multivector_proxy_encoder_relopt_options,
					   PGTURBOHYBRID_DEFAULT_MULTIVECTOR_PROXY_ENCODER,
					   "Valid values are \"normalized_mean\", \"mean\", \"first_token\", \"max_abs_mean\", \"centroid_mean\", \"max_pool\", \"random_projection_fde\", and \"learned_projection_v1\".",
					   AccessExclusiveLock);
	add_enum_reloption(pgturbohybrid_relopt_kind, "multivector_context_mode",
					   "Long-context multivector scoring mode.",
					   pgturbohybrid_multivector_context_mode_relopt_options,
					   PGTURBOHYBRID_MULTIVECTOR_CONTEXT_MODE_FLAT,
					   "Valid values are \"flat\" and \"context_level\".",
					   AccessExclusiveLock);
	add_enum_reloption(pgturbohybrid_relopt_kind, "multivector_field_mode",
					   "Field-aware multivector scoring mode.",
					   pgturbohybrid_multivector_field_mode_relopt_options,
					   PGTURBOHYBRID_MULTIVECTOR_FIELD_MODE_OFF,
					   "Valid values are \"off\" and \"weighted\".",
					   AccessExclusiveLock);
	add_int_reloption(pgturbohybrid_relopt_kind, "sparse_quant_bits",
					  "Sparse-vector postings quantization bit width (0=f32, 8, 16).",
					  PGTURBOHYBRID_SPARSE_DEFAULT_QUANT_BITS, 0, 16, AccessExclusiveLock);
	add_enum_reloption(pgturbohybrid_relopt_kind, "sparse_quant_mode",
					   "Sparse-vector postings quantization mode.",
					   pgturbohybrid_sparse_quant_mode_relopt_options,
					   PGTURBOHYBRID_SPARSE_QUANT_MODE_PER_TERM_LINEAR,
					   "Valid values are \"f32\" and \"per_term_linear\".",
					   AccessExclusiveLock);
	add_enum_reloption(pgturbohybrid_relopt_kind, "sparse_postings_encoding",
					   "Sparse-vector postings physical encoding.",
					   pgturbohybrid_sparse_encoding_relopt_options,
					   PGTURBOHYBRID_SPARSE_ENCODING_OPT_AUTO,
					   "Valid values are \"auto\", \"offset16_soa\", and \"varint\".",
					   AccessExclusiveLock);
	add_int_reloption(pgturbohybrid_relopt_kind, "sparse_block_size",
					  "Sparse-vector postings per chunk.",
					  PGTURBOHYBRID_SPARSE_DEFAULT_BLOCK_SIZE, 1,
					  PGTURBOHYBRID_SPARSE_MAX_BLOCK_SIZE, AccessExclusiveLock);
	add_bool_reloption(pgturbohybrid_relopt_kind, "sparse_block_max",
					   "Write a per-chunk block-max directory enabling sparse WAND pruning.",
					   true, AccessExclusiveLock);

	if (IsParallelWorker())
		return;

	gucsAlreadyDefined =
		GetConfigOption("turbohybrid.default_dense_k", true, false) != NULL;
	if (gucsAlreadyDefined)
		return;

	PgturbohybridRegisterGUCs();
}

#ifdef PGTURBOHYBRID_DEV_DIAGNOSTICS
static Datum
PgturbohybridHybridLastScanStats(PG_FUNCTION_ARGS)
{
	StringInfoData json;

	initStringInfo(&json);
	appendStringInfo(&json,
						 "{\"profile\":\"%s\","
						 "\"fusion\":\"%s\","
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
						 "\"final_k_requested\":%u,"
						 "\"final_k_effective\":%u,"
						 "\"detected_sql_limit\":%u,"
						 "\"final_k_inferred\":%s,"
						 "\"auto_budget_limit\":%u,"
						 "\"union_candidates\":%u,"
					 "\"final_results\":%u,"
					 "\"fusion_strategy\":\"%s\","
					 "\"fusion_candidates_seen\":%u,"
					 "\"fusion_heap_size\":%u,"
					 "\"fusion_duplicates\":" UINT64_FORMAT ","
					 "\"fusion_heap_replacements\":" UINT64_FORMAT ","
					 "\"fusion_generation_array_reused\":%s,"
					 "\"fusion_generation_array_reset\":%s,"
					 "\"fast_weighted_enabled\":%s,"
					 "\"fast_weighted_alpha\":%.6f,"
					 "\"bm25_norm_mode\":\"%s\","
					 "\"dense_norm_mode\":\"%s\","
					 "\"hybrid_budget_policy\":\"%s\","
					 "\"hybrid_query_shape\":\"%s\","
					 "\"hybrid_dense_k_chosen\":%u,"
					 "\"hybrid_bm25_k_chosen\":%u,"
					 "\"hybrid_budget_reason\":\"%s\","
					 "\"both_match\":%u,"
					 "\"dense_only\":%u,"
					 "\"bm25_only\":%u,"
						 "\"graph_visited_nodes\":" UINT64_FORMAT ","
						 "\"graph_scored_codes\":" UINT64_FORMAT ","
						 "\"graph_exact_rescore_count\":" UINT64_FORMAT ","
						 "\"heap_rescore_count\":" UINT64_FORMAT ","
						 "\"heap_fetch_us\":" UINT64_FORMAT ","
						 "\"heap_rescore_us\":" UINT64_FORMAT ","
						 "\"heap_rescore_auto_enabled\":%s,"
						 "\"heap_rescore_reason\":\"%s\","
						 "\"exact_rescore_source\":\"%s\","
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
					 "\"bm25_blocks_pruned_by_fused_score_bound\":" UINT64_FORMAT ","
					 "\"bm25_candidates_pruned_by_fused_score_bound\":" UINT64_FORMAT ","
					 "\"bm25_candidates_scored\":%u,"
					 "\"bm25_cache_bytes\":" UINT64_FORMAT ","
					 "\"bm25_cache_lexicon_entries\":%u,"
					 "\"bm25_cache_hit\":%s,"
					 "\"bm25_cache_build_us\":" UINT64_FORMAT ","
					 "\"bm25_cache_docstats_loaded\":%s,"
					 "\"bm25_cache_liveness_loaded\":%s,"
					 "\"bm25_docstats_loaded_this_query\":%s,"
					 "\"bm25_liveness_loaded_this_query\":%s,"
					 "\"bm25_docstats_bytes\":" UINT64_FORMAT ","
					 "\"bm25_liveness_bytes\":" UINT64_FORMAT ","
					 "\"bm25_cold_cache_o_n_work\":%s,"
					 "\"bm25_postings_decode_ratio\":%.6f,"
					 "\"bm25_common_term_fallback\":%s,"
					 "\"bm25_wand_pruned\":" UINT64_FORMAT ","
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
						 pgturbohybrid_last_scan_state.profile[0] != '\0' ?
						 pgturbohybrid_last_scan_state.profile :
						 PgturbohybridProfileName(pgturbohybrid_profile),
						 pgturbohybrid_last_scan_state.fusion[0] != '\0' ?
						 pgturbohybrid_last_scan_state.fusion : "none",
						 pgturbohybrid_last_scan_state.dense.candidatesRequested,
						 pgturbohybrid_last_scan_state.dense.candidatesEffective,
						 pgturbohybrid_last_scan_state.dense.kDefaulted ? "true" : "false",
						 pgturbohybrid_last_scan_state.dense.candidates,
					 pgturbohybrid_last_scan_state.dense.effectiveResultTarget,
					 pgturbohybrid_last_scan_state.dense.effectiveSearchEf,
					 pgturbohybrid_last_scan_state.dense.effectiveRescoreBand,
					 pgturbohybrid_last_scan_state.dense.highdimWideningMultiplier,
					 PgturbohybridGraphDenseWideningReasonName(pgturbohybrid_last_scan_state.dense.wideningReason),
						 PgturbohybridGraphDenseBudgetPolicyNameExternal(pgturbohybrid_last_scan_state.dense.budgetPolicy),
						 PgturbohybridGraphRescoreBandPolicyNameExternal(pgturbohybrid_last_scan_state.dense.rescoreBandPolicy),
						 pgturbohybrid_last_scan_state.bm25.candidatesRequested,
						 pgturbohybrid_last_scan_state.bm25.candidatesEffective,
						 pgturbohybrid_last_scan_state.bm25.kDefaulted ? "true" : "false",
						 pgturbohybrid_last_scan_state.bm25.candidates,
						 pgturbohybrid_last_scan_state.bm25.budgetReason[0] != '\0' ?
						 pgturbohybrid_last_scan_state.bm25.budgetReason : "unknown",
						 pgturbohybrid_last_scan_state.bm25.denseConfidence,
						 PgturbohybridBm25HybridBoundModeName(pgturbohybrid_last_scan_state.bm25.hybridBoundMode),
						 pgturbohybrid_last_scan_state.bm25.hybridBoundStopRank,
						 pgturbohybrid_last_scan_state.bm25.hybridBoundSkippedEstimated,
						 pgturbohybrid_last_scan_state.bm25.hybridBoundThreshold,
						 pgturbohybrid_last_scan_state.bm25.hybridBoundSafe ? "true" : "false",
						 pgturbohybrid_last_scan_state.rrfKRequested,
						 pgturbohybrid_last_scan_state.rrfKEffective,
						 pgturbohybrid_last_scan_state.rrfKDefaulted ? "true" : "false",
						 pgturbohybrid_last_scan_state.finalKRequested,
						 pgturbohybrid_last_scan_state.finalKEffective,
						 pgturbohybrid_last_scan_state.detectedSqlLimit,
						 pgturbohybrid_last_scan_state.finalKInferred ? "true" : "false",
						 pgturbohybrid_last_scan_state.autoBudgetLimit,
						 pgturbohybrid_last_scan_state.unionCandidates,
					 pgturbohybrid_last_scan_state.finalResults,
					 pgturbohybrid_last_scan_state.fusionStats.strategy[0] != '\0' ?
					 pgturbohybrid_last_scan_state.fusionStats.strategy : "none",
					 pgturbohybrid_last_scan_state.fusionStats.candidatesSeen,
					 pgturbohybrid_last_scan_state.fusionStats.heapSize,
					 pgturbohybrid_last_scan_state.fusionStats.duplicates,
					 pgturbohybrid_last_scan_state.fusionStats.heapReplacements,
					 pgturbohybrid_last_scan_state.fusionStats.generationArrayReused ? "true" : "false",
					 pgturbohybrid_last_scan_state.fusionStats.generationArrayReset ? "true" : "false",
					 pgturbohybrid_last_scan_state.fastWeighted.enabled ? "true" : "false",
					 pgturbohybrid_last_scan_state.fastWeighted.alpha,
					 pgturbohybrid_last_scan_state.bm25.normMode[0] != '\0' ?
					 pgturbohybrid_last_scan_state.bm25.normMode : "none",
					 pgturbohybrid_last_scan_state.dense.normMode[0] != '\0' ?
					 pgturbohybrid_last_scan_state.dense.normMode : "none",
					 pgturbohybrid_last_scan_state.hybrid.budgetPolicy[0] != '\0' ?
					 pgturbohybrid_last_scan_state.hybrid.budgetPolicy : "fixed",
					 pgturbohybrid_last_scan_state.hybrid.queryShape[0] != '\0' ?
					 pgturbohybrid_last_scan_state.hybrid.queryShape : "fixed",
					 pgturbohybrid_last_scan_state.hybrid.denseKChosen,
					 pgturbohybrid_last_scan_state.hybrid.bm25KChosen,
					 pgturbohybrid_last_scan_state.hybrid.budgetReason[0] != '\0' ?
					 pgturbohybrid_last_scan_state.hybrid.budgetReason : "fixed_policy",
					 pgturbohybrid_last_scan_state.bothMatch,
					 pgturbohybrid_last_scan_state.dense.only,
					 pgturbohybrid_last_scan_state.bm25.only,
						 pgturbohybrid_last_scan_state.graphVisitedNodes,
						 pgturbohybrid_last_scan_state.graphScoredCodes,
						 pgturbohybrid_last_scan_state.graphExactRescoreCount,
						 pgturbohybrid_last_scan_state.graphHeapRescoreCount,
						 pgturbohybrid_last_scan_state.graphHeapFetchUs,
						 pgturbohybrid_last_scan_state.graphHeapRescoreUs,
						 pgturbohybrid_last_scan_state.graphHeapRescoreAutoEnabled ? "true" : "false",
						 PgturbohybridGraphDenseHeapRescoreReasonName(pgturbohybrid_last_scan_state.graphHeapRescoreReason),
						 PgturbohybridGraphExactRescoreSourceName(pgturbohybrid_last_scan_state.graphExactRescoreSource),
						 pgturbohybrid_last_scan_state.graphPrepareUs,
					 pgturbohybrid_last_scan_state.graphTraverseUs,
					 pgturbohybrid_last_scan_state.graphEntryUs,
					 pgturbohybrid_last_scan_state.graphBaseUs,
					 pgturbohybrid_last_scan_state.graphBatchUs,
					 pgturbohybrid_last_scan_state.graphHeapUs,
					 pgturbohybrid_last_scan_state.graphFillUs,
					 pgturbohybrid_last_scan_state.graphRescoreUs,
					 pgturbohybrid_last_scan_state.graphSortUs,
					 pgturbohybrid_last_scan_state.bm25.terms,
					 pgturbohybrid_last_scan_state.bm25.postingsDecoded,
					 pgturbohybrid_last_scan_state.bm25.blocksVisited,
					 pgturbohybrid_last_scan_state.bm25.blocksSkipped,
					 pgturbohybrid_last_scan_state.bm25.fusedScoreBoundBlocksPruned,
					 pgturbohybrid_last_scan_state.bm25.fusedScoreBoundCandidatesPruned,
					 pgturbohybrid_last_scan_state.bm25.candidatesScored,
					 pgturbohybrid_last_scan_state.bm25.cacheBytes,
					 pgturbohybrid_last_scan_state.bm25.cacheLexiconEntries,
					 pgturbohybrid_last_scan_state.bm25.cacheHit ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25.cacheBuildUs,
					 pgturbohybrid_last_scan_state.bm25.cacheDocstatsLoaded ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25.cacheLivenessLoaded ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25.docstatsLoadedThisQuery ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25.livenessLoadedThisQuery ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25.docstatsBytes,
					 pgturbohybrid_last_scan_state.bm25.livenessBytes,
					 pgturbohybrid_last_scan_state.bm25.coldCacheONWork ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25.postingsDecodeRatio,
					 pgturbohybrid_last_scan_state.bm25.commonTermFallback ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25.wandPruned,
					 pgturbohybrid_last_scan_state.bm25.hotPostingsCacheHits > 0 ?
					 "true" : "false",
					 pgturbohybrid_last_scan_state.bm25.hotPostingsCacheHits,
					 pgturbohybrid_last_scan_state.bm25.hotPostingsCacheMisses,
					 pgturbohybrid_last_scan_state.bm25.hotPostingsCacheBytes,
					 pgturbohybrid_last_scan_state.bm25.hotPostingsCacheEvictions,
					 PgturbohybridBm25DeltaLookupModeName(pgturbohybrid_last_scan_state.bm25.deltaLookupMode),
					 pgturbohybrid_last_scan_state.bm25.deltaPagesScanned,
					 pgturbohybrid_last_scan_state.bm25.deltaTermPagesRead,
					 pgturbohybrid_last_scan_state.bm25.deltaBlocksVisited,
					 pgturbohybrid_last_scan_state.bm25.deltaPostingsDecoded,
					 pgturbohybrid_last_scan_state.bm25.deltaCacheBytes,
					 pgturbohybrid_last_scan_state.bm25.deltaCacheTerms,
					 pgturbohybrid_last_scan_state.bm25.deltaCacheHit ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25.wandIterations,
					 pgturbohybrid_last_scan_state.bm25.wandThresholdUpdates,
					 pgturbohybrid_last_scan_state.bm25.wandActiveSorts,
					 pgturbohybrid_last_scan_state.bm25.wandHeapUpdates,
					 pgturbohybrid_last_scan_state.bm25.wandFullReorders,
					 PgturbohybridBm25WandBoundTypeName(pgturbohybrid_last_scan_state.bm25.wandBoundType),
					 pgturbohybrid_last_scan_state.bm25.wandBoundTighteningHits,
					 pgturbohybrid_last_scan_state.bm25.wandHeapReplacements,
					 PgturbohybridBm25RuntimeStrategyName(pgturbohybrid_last_scan_state.bm25.strategy),
					 PgturbohybridBm25StrategyName(pgturbohybrid_bm25_strategy),
					 pgturbohybrid_last_scan_state.bm25.andDriverDf,
					 pgturbohybrid_last_scan_state.bm25.andVerifiedCandidates,
					 pgturbohybrid_last_scan_state.bm25.andRejectedCandidates,
					 pgturbohybrid_last_scan_state.bm25.impactTerms,
					 pgturbohybrid_last_scan_state.bm25.impactTiersRead,
					 pgturbohybrid_last_scan_state.bm25.impactPostingsRead,
					 pgturbohybrid_last_scan_state.bm25.impactRemainingUpperBound,
					 pgturbohybrid_last_scan_state.bm25.impactEarlyStop ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25.impactExactSafe ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25.impactFullPostingsAvoided ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25.impactLoadedFromStorage ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25.impactBuiltLazily ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25.impactLazyPostingsScanned,
					 PgturbohybridBm25AccumulatorModeName(pgturbohybrid_last_scan_state.bm25.accumulatorMode),
					 pgturbohybrid_last_scan_state.bm25.accumulatorHashLookups,
					 pgturbohybrid_last_scan_state.bm25.accumulatorDenseUpdates,
					 pgturbohybrid_last_scan_state.bm25.finalHeapReplacements,
					 pgturbohybrid_last_scan_state.bm25.finalSortedCount,
					 pgturbohybrid_last_scan_state.bm25.fullSortAvoided ? "true" : "false",
					 PgturbohybridBm25QueryShapeName(pgturbohybrid_last_scan_state.bm25.queryShape),
					 PgturbohybridBm25BooleanEvalModeName(pgturbohybrid_last_scan_state.bm25.booleanEvalMode),
					 pgturbohybrid_last_scan_state.bm25.booleanEvalCalls,
					 PgturbohybridBm25KernelName(pgturbohybrid_last_scan_state.bm25.decodeKernel),
					 PgturbohybridBm25KernelName(pgturbohybrid_last_scan_state.bm25.scoreKernel),
					 PgturbohybridBm25SimdForceName(pgturbohybrid_bm25_simd_force),
					 pgturbohybrid_last_scan_state.bm25.simdBlocks,
					 pgturbohybrid_last_scan_state.bm25.scalarTailPostings,
					 pgturbohybrid_last_scan_state.bm25.prefetches,
					 PgturbohybridGraphTqSimdForceName(pgturbohybrid_dense_simd_force),
					 PgturbohybridGraphTqExactSimdForceName(pgturbohybrid_dense_exact_simd_force),
					 pgturbohybrid_last_scan_state.dense.elapsedUs,
					 pgturbohybrid_last_scan_state.bm25.elapsedUs,
					 pgturbohybrid_last_scan_state.fusionStats.elapsedUs,
					 pgturbohybrid_last_scan_state.elapsedUs);

	PG_RETURN_DATUM(DirectFunctionCall1(jsonb_in, CStringGetDatum(json.data)));
}
#endif

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

	amroutine->ambuild = pgturbohybridambuild;
	amroutine->ambuildempty = pgturbohybridambuildempty;
	amroutine->aminsert = pgturbohybridaminsert;
#if PG_VERSION_NUM >= 170000
	amroutine->aminsertcleanup = NULL;
#endif
	amroutine->ambulkdelete = pgturbohybridambulkdelete;
	amroutine->amvacuumcleanup = pgturbohybridamvacuumcleanup;
	amroutine->amcanreturn = NULL;
	amroutine->amcostestimate = pgturbohybridamcostestimate;
#if PG_VERSION_NUM >= 180000
	amroutine->amgettreeheight = NULL;
#endif
	amroutine->amoptions = pgturbohybridamoptions;
	amroutine->amproperty = NULL;
	amroutine->ambuildphasename = NULL;
	amroutine->amvalidate = pgturbohybridamvalidate;
#if PG_VERSION_NUM >= 140000
	amroutine->amadjustmembers = NULL;
#endif
	amroutine->ambeginscan = pgturbohybridambeginscan;
	amroutine->amrescan = pgturbohybridamrescan;
	amroutine->amgettuple = pgturbohybridamgettuple;
	amroutine->amgetbitmap = NULL;
	amroutine->amendscan = pgturbohybridamendscan;
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
