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

static relopt_enum_elt_def pgturbohybrid_routing_relopt_options[] = {
	{"auto", PGTURBOHYBRID_ROUTING_AUTO},
	{"graph", PGTURBOHYBRID_ROUTING_GRAPH},
	{"flat", PGTURBOHYBRID_ROUTING_FLAT},
	{NULL, 0}
};

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

/* Sparse exact-storage mode (sidecar reserved for a later prompt; only off works). */
#define PGTURBOHYBRID_SPARSE_EXACT_STORAGE_OFF 0
#define PGTURBOHYBRID_SPARSE_EXACT_STORAGE_SIDECAR 1
static relopt_enum_elt_def pgturbohybrid_sparse_exact_storage_relopt_options[] = {
	{"off", PGTURBOHYBRID_SPARSE_EXACT_STORAGE_OFF},
	{"sidecar", PGTURBOHYBRID_SPARSE_EXACT_STORAGE_SIDECAR},
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
	{"learned_projection_placeholder", PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_LEARNED_PROJECTION_PLACEHOLDER},
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
	bool		bm25BranchAvailable;
	bool		denseBranchUsed;
	bool		multivectorBranchUsed;
	bool		bm25BranchUsed;
	bool		sparseBranchAvailable;
	bool		sparseBranchUsed;
	uint32		sparseTerms;
	uint32		sparseResolvedTerms;
	uint64		sparsePostingsTouched;
	uint64		sparseCandidatesScored;
	uint64		sparseElapsedUs;
	uint32		sparseCandidatesRequested;
	uint32		sparseCandidatesEffective;
	bool		sparseKDefaulted;
	uint32		sparseCandidates;
	int			sparseQuantBits;
	int			sparseQuantMode;
	int			sparseEncoding;
	uint64		sparseScalarTailPostings;
	int			sparseRerankMode;
	uint64		sparseExactRerankCount;
	uint64		sparseExactRerankFetchUs;
	uint64		sparseExactRerankScoreUs;
	bool		sparseExactRerankTopkChanged;
	int			sparseScoreKernel;
	uint64		sparseSimdBlocks;
	bool		sparseUsedWand;
	uint64		sparseBlocksVisited;
	uint64		sparseBlocksSkipped;
	uint64		sparseWandPruned;
	uint64		sparseWandIterations;
	uint64		sparseWandThresholdUpdates;
	uint64		sparseWandHeapUpdates;
	bool		sparseCacheHit;
	uint64		sparseCacheBuildUs;
	uint64		sparseCacheBytes;
	uint64		sparseHotCacheHits;
	uint64		sparseHotCacheMisses;
	uint64		sparseHotCacheBytes;
	uint64		sparseHotCacheEvictions;
	uint32		sparseDeltaPages;
	uint32		sparseDeltaTerms;
	uint64		sparseDeltaPostingsDecoded;
	bool		sparseDeltaCacheHit;
	uint32		sparseDeltaGeneration;
	PgturbohybridBranchPlan branchPlan;
	char		profile[16];
	char		fusion[16];
	uint32		denseCandidatesRequested;
	uint32		denseCandidatesEffective;
	bool		denseKDefaulted;
	uint32		denseCandidates;
	uint32		multivectorCandidatesRequested;
	uint32		multivectorCandidatesEffective;
	uint32		multivectorCandidates;
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
	uint32		finalKRequested;
	uint32		finalKEffective;
	uint32		detectedSqlLimit;
	bool		finalKInferred;
	uint32		autoBudgetLimit;
	uint32		unionCandidates;
	uint32		finalResults;
	char		fusionStrategy[24];
	uint32		fusionCandidatesSeen;
	uint32		fusionHeapSize;
	uint64		fusionDuplicates;
	uint64		fusionHeapReplacements;
	bool		fusionGenerationArrayReused;
	bool		fusionGenerationArrayReset;
	bool		multivectorEnabled;
	uint32		multivectorQueryVectors;
	uint32		multivectorDocVectorsLimit;
	uint64		multivectorSubvectorSearches;
	uint64		multivectorRawSubvectorHits;
	bool		multivectorAdaptiveWideningTriggered;
	uint32		multivectorAdaptiveInitialRawTarget;
	uint32		multivectorAdaptiveFinalRawTarget;
	char		multivectorDocMapSource[16];
	char		multivectorCandidateSource[48];
	char		multivectorCandidatePath[48];
	char		multivectorProxyEncoderKind[32];
	bool		learnedProjectionLoaded;
	uint32		learnedProjectionDim;
	uint64		learnedProjectionWeightBytes;
	char		learnedProjectionModel[128];
	char		learnedProjectionChecksum[128];
	uint64		learnedProjectionQueryEncodeUs;
	char		multivectorGraphMode[24];
	uint64		multivectorProxyGraphSearches;
	bool		multivectorExactTokenScanEnabled;
	uint64		multivectorExactTokenScanNodesScored;
	bool		multivectorPlainFallbackUsed;
	char		multivectorPlainFallbackReason[48];
	uint64		multivectorPlainFallbackDocsScored;
	uint64		multivectorPlainFallbackPairs;
	bool		multivectorDocGraphPrototypeEnabled;
	uint64		multivectorDocGraphNodes;
	uint64		multivectorDocGraphDocsScored;
	uint64		multivectorDocGraphEdgesVisited;
	uint32		multivectorDocGraphCandidates;
	uint32		multivectorDocGraphSearchEf;
	uint32		multivectorDocGraphOversampling;
	uint32		multivectorDocGraphRescoreK;
	uint32		multivectorDocGraphEntrySampleConfigured;
	uint32		multivectorDocGraphEntrySampleEffective;
	uint32		multivectorDocGraphEntrySampleScored;
	uint64		multivectorDocGraphQuantizedScores;
	uint64		compactMaxsimScoreUs;
	uint64		compactMaxsimPairs;
	uint64		compactMaxsimCacheHits;
	uint64		compactMaxsimCacheMisses;
	uint64		compactMaxsimBoundChecks;
	uint64		compactMaxsimDocsPruned;
	uint64		compactMaxsimTokensSkipped;
	char		multivectorDocGraphStorageKind[16];
	bool		proxyOnlyIndex;
	bool		centroidOnlyIndex;
		bool		fullMultivectorSidecarAvailable;
		bool		centroidSidecarAvailable;
		bool		centroidDocCodesAvailable;
		bool		quantizedInvertedSidecarAvailable;
	char		multivectorDocGraphRescoreSource[16];
	uint32		multivectorDocGraphExactRerankDocs;
	uint64		multivectorDocGraphHeapFetches;
	char		multivectorDocGraphWarning[96];
	uint32		multivectorProxyCandidateTarget;
	uint32		multivectorProxyCandidatesReturned;
	uint32		multivectorExactRerankKEffective;
	uint32		proxyCandidateLimitEffective;
	char		proxyCandidateLimitSource[32];
	uint64		proxyGraphNodesVisited;
	uint64		proxyGraphEdgesVisited;
	uint32		proxyGraphCandidatesSeen;
	uint32		proxyCandidatesReturned;
	uint64		proxyVectorScoresComputed;
	uint64		proxyVectorScoreUs;
	uint32		proxyCandidates;
	bool		proxyLazySidecarVectors;
	char		multivectorDocStorageCacheRequested[16];
	char		multivectorDocStorageCacheEffective[16];
	bool		proxyTop1Admission;
	uint32		proxyExactRerankDocs;
	uint64		proxyFullSidecarVectorsLoaded;
	uint64		proxyFullSidecarBytesTouched;
	uint64		proxyFullSidecarPagesRead;
	uint64		proxyFullSidecarLoadUs;
	uint64		proxyFullSidecarReconstructUs;
	uint64		proxyExactRerankHeapFetches;
	uint64		proxyExactRerankSidecarFetches;
	uint64		proxyExactRerankBytesTouched;
	uint64		proxyExactRerankUs;
	bool		sidecarCacheBuildThisQuery;
	uint64		sidecarCacheBuildBytes;
	uint64		sidecarCacheBuildPagesRead;
	uint64		sidecarCacheBuildUs;
	uint64		sidecarQueryBytesTouched;
	uint64		sidecarQueryPagesRead;
	uint64		sidecarQueryVectorsLoaded;
	uint64		sidecarQueryLoadUs;
	uint64		sidecarQueryUs;
	bool		proxyVectorUsesFullSidecarForGraph;
	bool		proxyVectorNearExhaustiveSidecarTouch;
	char		proxyVectorSidecarTouchReason[64];
	uint64		centroidListsVisited;
	uint64		centroidDocsTouched;
	uint64		centroidPrunedDocs;
	uint64		centroidPostingsTouched;
	uint64		centroidPostingsSelected;
	uint64		centroidPostingsSkipped;
	uint64		centroidProbeUs;
	uint64		centroidPostingScanUs;
	uint64		centroidAccumulateUs;
	uint64		centroidCandidateHeapUs;
	uint32		centroidPostingLimitPerToken;
	uint32		centroidProbeCentroidsPerToken;
	uint32		centroidCodewordTopM;
	double		centroidScoreThreshold;
	double		centroidScoreDropFromBest;
	uint64		centroidListsSkippedByThreshold;
	char		centroidPostingCapStrategy[32];
	char		centroidCandidateScoring[32];
	uint32		centroidCandidates;
	bool		centroidBitsetPrefilterEnabled;
	uint32		centroidBitsetMinTokenMatches;
	uint32		centroidBitsetListsUsed;
	uint32		centroidBitsetDocsSet;
	uint32		centroidBitsetDocsAfterThreshold;
	uint64		centroidBitsetPrefilterUs;
	uint64		centroidBitsetMemoryBytes;
	bool		centroidUpperBoundEnabled;
	uint64		centroidUpperBoundDocsChecked;
	uint64		centroidUpperBoundDocsPruned;
	uint64		centroidUpperBoundPruneUs;
	uint64		centroidUpperBoundUnsafeFallbacks;
	uint32		centroidCandidatesBeforeBound;
	uint32		centroidCandidatesAfterBound;
	uint32		multivectorCentroidCount;
	uint32		multivectorCentroidPrerankDocs;
	uint32		multivectorFullMaxsimRerankDocs;
	uint64		quantizedInvertedListsVisited;
	uint64		quantizedInvertedPostingsTouched;
	uint64		quantizedInvertedPostingsSelected;
	uint64		quantizedInvertedPostingsSkipped;
	uint32		quantizedInvertedPostingLimitPerToken;
	uint32		quantizedInvertedProbeCodewordsPerToken;
	char		quantizedInvertedPostingCapStrategy[32];
	uint64		quantizedInvertedDocsScored;
	uint32		quantizedInvertedCandidates;
	uint32		quantizedInvertedExactRerankDocs;
	char		quantizedInvertedCodebookSource[16];
	uint32		quantizedInvertedCodebookSize;
	uint32		quantizedInvertedCodebookDim;
	char		quantizedInvertedCodebookChecksum[128];
	uint32		quantizedInvertedCodebookTopM;
	uint64		quantizedInvertedAssignmentUs;
	uint64		quantizedInvertedQueryCodewordScoreUs;
	char		quantizedInvertedQueryCodewordKernel[16];
	uint64		quantizedInvertedQueryCodewordScoresComputed;
	uint64		quantizedInvertedQueryCodewordBlocks;
	uint64		quantizedInvertedQueryCodewordTopkUs;
	bool		quantizedInvertedQueryCodewordFullMatrixMaterialized;
	uint32		quantizedInvertedQueryCodewordActiveQueryTokens;
	uint32		quantizedInvertedQueryCodewordSkippedQueryTokens;
	uint64		quantizedInvertedListOffsetBytes;
	uint64		quantizedInvertedPostingBytes;
	uint64		quantizedInvertedSidecarBytes;
	char		quantizedInvertedCompactKernel[24];
	char		quantizedInvertedCompactScoreSource[32];
	uint64		quantizedInvertedCompactScoreUs;
	uint64		quantizedInvertedCompactDocsScored;
	uint64		quantizedInvertedCompactPayloadBytes;
	char		quantizedInvertedCompactDocOrder[16];
	uint64		quantizedInvertedCompactInnerAllocations;
	uint32		quantizedInvertedCompactActiveQueryTokens;
	uint64		quantizedInvertedCompactPairsEvaluated;
	uint64		quantizedInvertedCompactPairsSkipped;
	uint64		quantizedInvertedCompactPrefetches;
	double		quantizedInvertedCompactAvgDocTokens;
	double		quantizedInvertedCompactUsPerDoc;
	double		quantizedInvertedCompactPayloadBytesPerDoc;
	bool		quantizedInvertedCompactTopKChangedVsScalar;
	bool		quantizedInvertedPrecompactEnabled;
	char		quantizedInvertedPrecompactMode[32];
	uint32		quantizedInvertedDocsTouchedBeforePrecompact;
	uint32		quantizedInvertedPrecompactScoreK;
	uint32		quantizedInvertedPrecompactCoverageK;
	uint32		quantizedInvertedPrecompactPerTokenK;
	uint32		quantizedInvertedCompactMaxDocs;
	uint32		quantizedInvertedPrecompactScoreDocs;
	uint32		quantizedInvertedPrecompactCoverageDocs;
	uint32		quantizedInvertedPrecompactPerTokenDocs;
	uint32		quantizedInvertedPrecompactUnionDocs;
	uint32		quantizedInvertedPrecompactDuplicates;
	uint32		quantizedInvertedPrecompactPrunedDocs;
	uint64		quantizedInvertedPrecompactUs;
	uint32		quantizedInvertedCompactDocsSkippedByPrecompact;
	char		quantizedInvertedTokenCoverageMode[24];
	uint32		quantizedInvertedActiveQueryTokens;
	uint64		quantizedInvertedTokenMatchesTotal;
	uint32		quantizedInvertedTokenMatchesMax;
	uint32		quantizedInvertedMinTokenMatches;
	uint64		quantizedInvertedTokenMatchFilteredDocs;
	bool		quantizedInvertedScoreBoundPruningEnabled;
	uint64		quantizedInvertedScoreBoundDocsChecked;
	uint64		quantizedInvertedScoreBoundDocsPruned;
	uint64		quantizedInvertedScoreBoundPruneUs;
	uint64		quantizedInvertedScoreBoundUnsafeFallbacks;
	uint32		quantizedInvertedCandidatesBeforeBound;
	uint32		quantizedInvertedCandidatesAfterBound;
	char		multivectorDocSidecarCacheMode[16];
	uint64		multivectorDocSidecarPagesRead;
	uint64		multivectorDocSidecarCacheHits;
	uint64		multivectorDocSidecarCacheMisses;
	uint64		multivectorDocSidecarBytesTouched;
	uint64		multivectorDocSidecarVectorsLoaded;
	uint64		multivectorDocSidecarDocMapPagesRead;
	uint64		multivectorDocSidecarDocMapBytesTouched;
	uint64		multivectorDocSidecarResidentVectorsLoaded;
	uint64		multivectorDocSidecarResidentBytesLoaded;
	uint64		multivectorDocSidecarVectorChunkRefBytesTouched;
	uint64		multivectorDocSidecarPagedVectorPagesRead;
	uint64		multivectorDocSidecarPagedVectorBytesTouched;
	uint64		multivectorSidecarPageReadUs;
	uint64		multivectorSidecarVectorReconstructUs;
	uint64		multivectorTokensOriginal;
	uint64		multivectorTokensPooled;
	bool		multivectorReservoirsEnabled;
	uint32		multivectorReservoirScoreDocs;
	uint32		multivectorReservoirCoverageDocs;
	uint32		multivectorReservoirMeanDocs;
	uint32		multivectorReservoirPerTokenDocs;
	uint32		multivectorReservoirBm25Docs;
	uint32		multivectorReservoirUnionDocs;
	uint32		multivectorReservoirDuplicates;
	bool		multivectorBm25InjectionEnabled;
	uint32		multivectorBm25InjectionCandidates;
	uint32		multivectorBm25InjectionCandidateLimit;
	uint32		multivectorBm25InjectionPoolSize;
	char		multivectorBm25InjectionLimitReason[32];
	uint32		multivectorBm25InjectionRetained;
	uint32		multivectorBm25InjectionExactReranked;
	uint32		learnedSparseCandidates;
	uint32		learnedSparseRetainedForMaxsim;
	uint64		learnedSparseBranchLatencyUs;
	uint64		multivectorDocMapBytes;
	uint64		multivectorUniqueDocs;
	uint64		multivectorDuplicateDocHits;
	uint64		multivectorMaxsimUpdates;
	uint32		multivectorDocCandidates;
	bool		multivectorExactRerankEnabled;
	uint32		multivectorExactRerankDocs;
	uint64		multivectorExactRerankPairs;
	char		multivectorExactRerankSource[16];
	uint64		multivectorExactRerankHeapFetches;
	uint64		multivectorExactRerankSidecarReads;
	uint64		multivectorExactRerankSidecarBytes;
	uint64		multivectorCandidateSourceUs;
	uint64		multivectorDocGraphTraversalUs;
	uint64		multivectorProxyCandidateUs;
	uint64		multivectorProxyGraphTraversalUs;
	uint64		multivectorProxyScoringUs;
	uint64		multivectorCentroidLitePostingUs;
	uint64		multivectorQuantizedInvertedPostingUs;
	uint64		multivectorSidecarLoadUs;
	uint64		multivectorHeapVisibilityUs;
	uint64		multivectorExactHeapFetchUs;
	uint64		multivectorExactRerankUs;
	uint64		multivectorFinalSortUs;
	uint32		exactRerankCandidates;
	uint64		exactRerankTokensEvaluated;
	uint64		exactRerankTokensSkipped;
	uint64		exactRerankPairsSaved;
	bool		adaptiveRerankTopKChangedVsFull;
	char		multivectorExactKernel[16];
	char		multivectorAccumulatorKind[48];
	uint64		multivectorMemoryBytesEstimate;
	bool		multivectorAdmissionDebugEnabled;
	uint32		multivectorAdmissionCandidatesBeforeRerank;
	uint32		multivectorAdmissionCandidatesAfterTruncation;
	uint32		multivectorAdmissionExactRerankDocs;
	bool		multivectorAdmissionTruncatedByDocCandidateK;
	bool		multivectorAdmissionTruncatedByAccumulatorMemory;
	bool		multivectorAdmissionTraceAvailable;
	uint32		multivectorAdmissionTraceCount;
	PgturbohybridMultiVectorAdmissionTraceEntry multivectorAdmissionTrace[PGTURBOHYBRID_MULTIVECTOR_DEBUG_TRACE_LIMIT_MAX];
	bool		multivectorTokenStatsAvailable;
	uint32		multivectorTokenStatsCount;
	PgturbohybridMultiVectorTokenStatsEntry multivectorTokenStats[PGTURBOHYBRID_MULTIVECTOR_TOKEN_STATS_LIMIT_MAX];
	int			finalDiversityMode;
	int32		finalDiversityPayloadSlot;
	uint32		finalDiversityPoolSize;
	uint32		finalDiversitySelected;
	uint64		finalDiversityDuplicateGroupsSuppressed;
	uint64		finalDiversityUs;
	uint32		bothMatch;
	uint32		denseOnly;
	uint32		bm25Only;
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
	uint32		bm25Terms;
	uint64		bm25PostingsDecoded;
	uint64		bm25BlocksVisited;
	uint64		bm25BlocksSkipped;
	uint64		bm25FusedScoreBoundBlocksPruned;
	uint64		bm25FusedScoreBoundCandidatesPruned;
	uint32		bm25CandidatesScored;
	int			bm25HeapTSVectorRerankMode;
	uint32		bm25HeapTSVectorRerankCount;
	uint64		bm25HeapTSVectorRerankFetchUs;
	uint64		bm25HeapTSVectorRerankScoreUs;
	bool		bm25HeapTSVectorRerankTopKChanged;
	uint64		bm25CacheBytes;
	uint32		bm25CacheLexiconEntries;
	bool		bm25CacheHit;
	uint64		bm25CacheBuildUs;
	bool		bm25CacheDocstatsLoaded;
	bool		bm25CacheLivenessLoaded;
	bool		bm25DocstatsLoadedThisQuery;
	bool		bm25LivenessLoadedThisQuery;
	uint64		bm25DocstatsBytes;
	uint64		bm25LivenessBytes;
	bool		bm25ColdCacheONWork;
	double		bm25PostingsDecodeRatio;
	bool		bm25CommonTermFallback;
	uint64		bm25WandPruned;
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
	bool		fastWeightedEnabled;
	double		fastWeightedAlpha;
	bool		calibratedFusionEnabled;
	char		calibratedFusionQueryShape[32];
	double		calibratedFusionAlphaEffective;
	double		calibratedFusionBothMatchBonus;
	char		calibratedFusionDenseNormMode[16];
	char		calibratedFusionBm25NormMode[16];
	bool		dbsfEnabled;
	double		dbsfBranchMean[2];
	double		dbsfBranchStddev[2];
	double		dbsfBranchMin[2];
	double		dbsfBranchMax[2];
	uint32		dbsfDegenerateBranches;
	char		bm25NormMode[16];
	char		denseNormMode[16];
	char		hybridBudgetPolicy[16];
	char		hybridQueryShape[32];
	uint32		hybridDenseKChosen;
	uint32		hybridBm25KChosen;
	char		hybridBudgetReason[96];
	uint64		denseElapsedUs;
	uint64		bm25ElapsedUs;
	uint64		fusionElapsedUs;
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
	stats->bm25BranchAvailable =
		pgturbohybrid_last_scan_state.bm25BranchAvailable;
	stats->denseBranchUsed =
		pgturbohybrid_last_scan_state.denseBranchUsed;
	stats->multivectorBranchUsed =
		pgturbohybrid_last_scan_state.multivectorBranchUsed;
	stats->bm25BranchUsed =
		pgturbohybrid_last_scan_state.bm25BranchUsed;
	stats->sparseBranchAvailable =
		pgturbohybrid_last_scan_state.sparseBranchAvailable;
	stats->sparseBranchUsed =
		pgturbohybrid_last_scan_state.sparseBranchUsed;
	stats->sparseTerms = pgturbohybrid_last_scan_state.sparseTerms;
	stats->sparseResolvedTerms =
		pgturbohybrid_last_scan_state.sparseResolvedTerms;
	stats->sparsePostingsTouched =
		pgturbohybrid_last_scan_state.sparsePostingsTouched;
	stats->sparseCandidatesScored =
		pgturbohybrid_last_scan_state.sparseCandidatesScored;
	stats->sparseElapsedUs = pgturbohybrid_last_scan_state.sparseElapsedUs;
	stats->sparseCandidatesRequested =
		pgturbohybrid_last_scan_state.sparseCandidatesRequested;
	stats->sparseCandidatesEffective =
		pgturbohybrid_last_scan_state.sparseCandidatesEffective;
	stats->sparseKDefaulted = pgturbohybrid_last_scan_state.sparseKDefaulted;
	stats->sparseCandidates = pgturbohybrid_last_scan_state.sparseCandidates;
	stats->sparseQuantBits = pgturbohybrid_last_scan_state.sparseQuantBits;
	stats->sparseQuantMode = pgturbohybrid_last_scan_state.sparseQuantMode;
	stats->sparseEncoding = pgturbohybrid_last_scan_state.sparseEncoding;
	stats->sparseScalarTailPostings =
		pgturbohybrid_last_scan_state.sparseScalarTailPostings;
	stats->sparseRerankMode = pgturbohybrid_last_scan_state.sparseRerankMode;
	stats->sparseExactRerankCount =
		pgturbohybrid_last_scan_state.sparseExactRerankCount;
	stats->sparseExactRerankFetchUs =
		pgturbohybrid_last_scan_state.sparseExactRerankFetchUs;
	stats->sparseExactRerankScoreUs =
		pgturbohybrid_last_scan_state.sparseExactRerankScoreUs;
	stats->sparseExactRerankTopkChanged =
		pgturbohybrid_last_scan_state.sparseExactRerankTopkChanged;
	stats->sparseScoreKernel = pgturbohybrid_last_scan_state.sparseScoreKernel;
	stats->sparseSimdBlocks = pgturbohybrid_last_scan_state.sparseSimdBlocks;
	stats->sparseUsedWand = pgturbohybrid_last_scan_state.sparseUsedWand;
	stats->sparseBlocksVisited = pgturbohybrid_last_scan_state.sparseBlocksVisited;
	stats->sparseBlocksSkipped = pgturbohybrid_last_scan_state.sparseBlocksSkipped;
	stats->sparseWandPruned = pgturbohybrid_last_scan_state.sparseWandPruned;
	stats->sparseWandIterations =
		pgturbohybrid_last_scan_state.sparseWandIterations;
	stats->sparseWandThresholdUpdates =
		pgturbohybrid_last_scan_state.sparseWandThresholdUpdates;
	stats->sparseWandHeapUpdates =
		pgturbohybrid_last_scan_state.sparseWandHeapUpdates;
	stats->sparseCacheHit = pgturbohybrid_last_scan_state.sparseCacheHit;
	stats->sparseCacheBuildUs = pgturbohybrid_last_scan_state.sparseCacheBuildUs;
	stats->sparseCacheBytes = pgturbohybrid_last_scan_state.sparseCacheBytes;
	stats->sparseHotCacheHits = pgturbohybrid_last_scan_state.sparseHotCacheHits;
	stats->sparseHotCacheMisses = pgturbohybrid_last_scan_state.sparseHotCacheMisses;
	stats->sparseHotCacheBytes = pgturbohybrid_last_scan_state.sparseHotCacheBytes;
	stats->sparseHotCacheEvictions =
		pgturbohybrid_last_scan_state.sparseHotCacheEvictions;
	stats->sparseDeltaPages = pgturbohybrid_last_scan_state.sparseDeltaPages;
	stats->sparseDeltaTerms = pgturbohybrid_last_scan_state.sparseDeltaTerms;
	stats->sparseDeltaPostingsDecoded =
		pgturbohybrid_last_scan_state.sparseDeltaPostingsDecoded;
	stats->sparseDeltaCacheHit = pgturbohybrid_last_scan_state.sparseDeltaCacheHit;
	stats->sparseDeltaGeneration =
		pgturbohybrid_last_scan_state.sparseDeltaGeneration;
	stats->branchPlan = pgturbohybrid_last_scan_state.branchPlan;
	stats->denseCandidatesEffective =
		pgturbohybrid_last_scan_state.denseCandidatesEffective;
	stats->multivectorCandidatesEffective =
		pgturbohybrid_last_scan_state.multivectorCandidatesEffective;
	stats->denseKDefaulted = pgturbohybrid_last_scan_state.denseKDefaulted;
	stats->bm25CandidatesEffective =
		pgturbohybrid_last_scan_state.bm25CandidatesEffective;
	stats->bm25KDefaulted = pgturbohybrid_last_scan_state.bm25KDefaulted;
	stats->bm25CacheHit = pgturbohybrid_last_scan_state.bm25CacheHit;
	stats->bm25CacheBuildUs = pgturbohybrid_last_scan_state.bm25CacheBuildUs;
	stats->bm25DocstatsLoadedThisQuery =
		pgturbohybrid_last_scan_state.bm25DocstatsLoadedThisQuery;
	stats->bm25LivenessLoadedThisQuery =
		pgturbohybrid_last_scan_state.bm25LivenessLoadedThisQuery;
	stats->bm25DocstatsBytes =
		pgturbohybrid_last_scan_state.bm25DocstatsBytes;
	stats->bm25LivenessBytes =
		pgturbohybrid_last_scan_state.bm25LivenessBytes;
	stats->bm25ColdCacheONWork =
		pgturbohybrid_last_scan_state.bm25ColdCacheONWork;
	stats->bm25PostingsDecodeRatio =
		pgturbohybrid_last_scan_state.bm25PostingsDecodeRatio;
	stats->bm25CommonTermFallback =
		pgturbohybrid_last_scan_state.bm25CommonTermFallback;
	stats->bm25WandPruned =
		pgturbohybrid_last_scan_state.bm25WandPruned;
	stats->bm25HotPostingsCacheHits =
		pgturbohybrid_last_scan_state.bm25HotPostingsCacheHits;
	stats->bm25HotPostingsCacheMisses =
		pgturbohybrid_last_scan_state.bm25HotPostingsCacheMisses;
	stats->bm25Terms = pgturbohybrid_last_scan_state.bm25Terms;
	stats->bm25FusedScoreBoundBlocksPruned =
		pgturbohybrid_last_scan_state.bm25FusedScoreBoundBlocksPruned;
	stats->bm25FusedScoreBoundCandidatesPruned =
		pgturbohybrid_last_scan_state.bm25FusedScoreBoundCandidatesPruned;
	strlcpy(stats->bm25HeapTSVectorRerankMode,
			PgturbohybridBm25HeapTSVectorRerankModeName(
				pgturbohybrid_last_scan_state.bm25HeapTSVectorRerankMode),
			sizeof(stats->bm25HeapTSVectorRerankMode));
	stats->bm25HeapTSVectorRerankCount =
		pgturbohybrid_last_scan_state.bm25HeapTSVectorRerankCount;
	stats->bm25HeapTSVectorRerankFetchUs =
		pgturbohybrid_last_scan_state.bm25HeapTSVectorRerankFetchUs;
	stats->bm25HeapTSVectorRerankScoreUs =
		pgturbohybrid_last_scan_state.bm25HeapTSVectorRerankScoreUs;
	stats->bm25HeapTSVectorRerankTopKChanged =
		pgturbohybrid_last_scan_state.bm25HeapTSVectorRerankTopKChanged;
	stats->fastWeightedEnabled =
		pgturbohybrid_last_scan_state.fastWeightedEnabled;
	stats->fastWeightedAlpha =
		pgturbohybrid_last_scan_state.fastWeightedAlpha;
	stats->calibratedFusionEnabled =
		pgturbohybrid_last_scan_state.calibratedFusionEnabled;
	strlcpy(stats->calibratedFusionQueryShape,
			pgturbohybrid_last_scan_state.calibratedFusionQueryShape,
			sizeof(stats->calibratedFusionQueryShape));
	stats->calibratedFusionAlphaEffective =
		pgturbohybrid_last_scan_state.calibratedFusionAlphaEffective;
	stats->calibratedFusionBothMatchBonus =
		pgturbohybrid_last_scan_state.calibratedFusionBothMatchBonus;
	strlcpy(stats->calibratedFusionDenseNormMode,
			pgturbohybrid_last_scan_state.calibratedFusionDenseNormMode,
			sizeof(stats->calibratedFusionDenseNormMode));
	strlcpy(stats->calibratedFusionBm25NormMode,
			pgturbohybrid_last_scan_state.calibratedFusionBm25NormMode,
			sizeof(stats->calibratedFusionBm25NormMode));
	stats->dbsfEnabled = pgturbohybrid_last_scan_state.dbsfEnabled;
	memcpy(stats->dbsfBranchMean,
		   pgturbohybrid_last_scan_state.dbsfBranchMean,
		   sizeof(stats->dbsfBranchMean));
	memcpy(stats->dbsfBranchStddev,
		   pgturbohybrid_last_scan_state.dbsfBranchStddev,
		   sizeof(stats->dbsfBranchStddev));
	memcpy(stats->dbsfBranchMin,
		   pgturbohybrid_last_scan_state.dbsfBranchMin,
		   sizeof(stats->dbsfBranchMin));
	memcpy(stats->dbsfBranchMax,
		   pgturbohybrid_last_scan_state.dbsfBranchMax,
		   sizeof(stats->dbsfBranchMax));
	stats->dbsfDegenerateBranches =
		pgturbohybrid_last_scan_state.dbsfDegenerateBranches;
	strlcpy(stats->bm25NormMode,
			pgturbohybrid_last_scan_state.bm25NormMode,
			sizeof(stats->bm25NormMode));
	strlcpy(stats->denseNormMode,
			pgturbohybrid_last_scan_state.denseNormMode,
			sizeof(stats->denseNormMode));
	strlcpy(stats->hybridBudgetPolicy,
			pgturbohybrid_last_scan_state.hybridBudgetPolicy,
			sizeof(stats->hybridBudgetPolicy));
	strlcpy(stats->hybridQueryShape,
			pgturbohybrid_last_scan_state.hybridQueryShape,
			sizeof(stats->hybridQueryShape));
	stats->hybridDenseKChosen =
		pgturbohybrid_last_scan_state.hybridDenseKChosen;
	stats->hybridBm25KChosen =
		pgturbohybrid_last_scan_state.hybridBm25KChosen;
	strlcpy(stats->hybridBudgetReason,
			pgturbohybrid_last_scan_state.hybridBudgetReason,
			sizeof(stats->hybridBudgetReason));
	strlcpy(stats->fusionStrategy,
			pgturbohybrid_last_scan_state.fusionStrategy,
			sizeof(stats->fusionStrategy));
	stats->fusionCandidatesSeen =
		pgturbohybrid_last_scan_state.fusionCandidatesSeen;
	stats->fusionDuplicates =
		pgturbohybrid_last_scan_state.fusionDuplicates;
	stats->fusionHeapReplacements =
		pgturbohybrid_last_scan_state.fusionHeapReplacements;
	stats->fusionGenerationArrayReused =
		pgturbohybrid_last_scan_state.fusionGenerationArrayReused;
	stats->fusionGenerationArrayReset =
		pgturbohybrid_last_scan_state.fusionGenerationArrayReset;
	stats->multivectorEnabled =
		pgturbohybrid_last_scan_state.multivectorEnabled;
	stats->multivectorQueryVectors =
		pgturbohybrid_last_scan_state.multivectorQueryVectors;
	stats->multivectorDocVectorsLimit =
		pgturbohybrid_last_scan_state.multivectorDocVectorsLimit;
	stats->multivectorSubvectorSearches =
		pgturbohybrid_last_scan_state.multivectorSubvectorSearches;
	stats->multivectorRawSubvectorHits =
		pgturbohybrid_last_scan_state.multivectorRawSubvectorHits;
	stats->multivectorAdaptiveWideningTriggered =
		pgturbohybrid_last_scan_state.multivectorAdaptiveWideningTriggered;
	stats->multivectorAdaptiveInitialRawTarget =
		pgturbohybrid_last_scan_state.multivectorAdaptiveInitialRawTarget;
	stats->multivectorAdaptiveFinalRawTarget =
		pgturbohybrid_last_scan_state.multivectorAdaptiveFinalRawTarget;
	strlcpy(stats->multivectorDocMapSource,
			pgturbohybrid_last_scan_state.multivectorDocMapSource,
			sizeof(stats->multivectorDocMapSource));
	strlcpy(stats->multivectorCandidateSource,
			pgturbohybrid_last_scan_state.multivectorCandidateSource,
			sizeof(stats->multivectorCandidateSource));
	strlcpy(stats->multivectorCandidatePath,
			pgturbohybrid_last_scan_state.multivectorCandidatePath,
			sizeof(stats->multivectorCandidatePath));
	strlcpy(stats->multivectorProxyEncoderKind,
			pgturbohybrid_last_scan_state.multivectorProxyEncoderKind,
			sizeof(stats->multivectorProxyEncoderKind));
	stats->learnedProjectionLoaded =
		pgturbohybrid_last_scan_state.learnedProjectionLoaded;
	stats->learnedProjectionDim =
		pgturbohybrid_last_scan_state.learnedProjectionDim;
	stats->learnedProjectionWeightBytes =
		pgturbohybrid_last_scan_state.learnedProjectionWeightBytes;
	strlcpy(stats->learnedProjectionModel,
			pgturbohybrid_last_scan_state.learnedProjectionModel,
			sizeof(stats->learnedProjectionModel));
	strlcpy(stats->learnedProjectionChecksum,
			pgturbohybrid_last_scan_state.learnedProjectionChecksum,
			sizeof(stats->learnedProjectionChecksum));
	stats->learnedProjectionQueryEncodeUs =
		pgturbohybrid_last_scan_state.learnedProjectionQueryEncodeUs;
	strlcpy(stats->multivectorGraphMode,
			pgturbohybrid_last_scan_state.multivectorGraphMode,
			sizeof(stats->multivectorGraphMode));
	stats->multivectorProxyGraphSearches =
		pgturbohybrid_last_scan_state.multivectorProxyGraphSearches;
	stats->multivectorExactTokenScanEnabled =
		pgturbohybrid_last_scan_state.multivectorExactTokenScanEnabled;
	stats->multivectorExactTokenScanNodesScored =
		pgturbohybrid_last_scan_state.multivectorExactTokenScanNodesScored;
	stats->multivectorPlainFallbackUsed =
		pgturbohybrid_last_scan_state.multivectorPlainFallbackUsed;
	strlcpy(stats->multivectorPlainFallbackReason,
			pgturbohybrid_last_scan_state.multivectorPlainFallbackReason,
			sizeof(stats->multivectorPlainFallbackReason));
	stats->multivectorPlainFallbackDocsScored =
		pgturbohybrid_last_scan_state.multivectorPlainFallbackDocsScored;
	stats->multivectorPlainFallbackPairs =
		pgturbohybrid_last_scan_state.multivectorPlainFallbackPairs;
	stats->multivectorDocGraphPrototypeEnabled =
		pgturbohybrid_last_scan_state.multivectorDocGraphPrototypeEnabled;
	stats->multivectorDocGraphNodes =
		pgturbohybrid_last_scan_state.multivectorDocGraphNodes;
	stats->multivectorDocGraphDocsScored =
		pgturbohybrid_last_scan_state.multivectorDocGraphDocsScored;
	stats->multivectorDocGraphEdgesVisited =
		pgturbohybrid_last_scan_state.multivectorDocGraphEdgesVisited;
	stats->multivectorDocGraphCandidates =
		pgturbohybrid_last_scan_state.multivectorDocGraphCandidates;
	stats->multivectorDocGraphSearchEf =
		pgturbohybrid_last_scan_state.multivectorDocGraphSearchEf;
	stats->multivectorDocGraphOversampling =
		pgturbohybrid_last_scan_state.multivectorDocGraphOversampling;
	stats->multivectorDocGraphRescoreK =
		pgturbohybrid_last_scan_state.multivectorDocGraphRescoreK;
	stats->multivectorDocGraphEntrySampleConfigured =
		pgturbohybrid_last_scan_state.multivectorDocGraphEntrySampleConfigured;
	stats->multivectorDocGraphEntrySampleEffective =
		pgturbohybrid_last_scan_state.multivectorDocGraphEntrySampleEffective;
	stats->multivectorDocGraphEntrySampleScored =
		pgturbohybrid_last_scan_state.multivectorDocGraphEntrySampleScored;
	stats->multivectorDocGraphQuantizedScores =
		pgturbohybrid_last_scan_state.multivectorDocGraphQuantizedScores;
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
	strlcpy(stats->multivectorDocGraphStorageKind,
			pgturbohybrid_last_scan_state.multivectorDocGraphStorageKind,
			sizeof(stats->multivectorDocGraphStorageKind));
	stats->proxyOnlyIndex =
		pgturbohybrid_last_scan_state.proxyOnlyIndex;
	stats->centroidOnlyIndex =
		pgturbohybrid_last_scan_state.centroidOnlyIndex;
	stats->fullMultivectorSidecarAvailable =
		pgturbohybrid_last_scan_state.fullMultivectorSidecarAvailable;
	stats->centroidSidecarAvailable =
		pgturbohybrid_last_scan_state.centroidSidecarAvailable;
	stats->centroidDocCodesAvailable =
		pgturbohybrid_last_scan_state.centroidDocCodesAvailable;
	stats->quantizedInvertedSidecarAvailable =
		pgturbohybrid_last_scan_state.quantizedInvertedSidecarAvailable;
	strlcpy(stats->multivectorDocGraphRescoreSource,
			pgturbohybrid_last_scan_state.multivectorDocGraphRescoreSource,
			sizeof(stats->multivectorDocGraphRescoreSource));
	stats->multivectorDocGraphExactRerankDocs =
		pgturbohybrid_last_scan_state.multivectorDocGraphExactRerankDocs;
	stats->multivectorDocGraphHeapFetches =
		pgturbohybrid_last_scan_state.multivectorDocGraphHeapFetches;
	strlcpy(stats->multivectorDocGraphWarning,
			pgturbohybrid_last_scan_state.multivectorDocGraphWarning,
			sizeof(stats->multivectorDocGraphWarning));
	stats->multivectorProxyCandidateTarget =
		pgturbohybrid_last_scan_state.multivectorProxyCandidateTarget;
	stats->multivectorProxyCandidatesReturned =
		pgturbohybrid_last_scan_state.multivectorProxyCandidatesReturned;
	stats->multivectorExactRerankKEffective =
		pgturbohybrid_last_scan_state.multivectorExactRerankKEffective;
	stats->proxyCandidateLimitEffective =
		pgturbohybrid_last_scan_state.proxyCandidateLimitEffective;
	strlcpy(stats->proxyCandidateLimitSource,
			pgturbohybrid_last_scan_state.proxyCandidateLimitSource,
			sizeof(stats->proxyCandidateLimitSource));
	stats->proxyGraphNodesVisited =
		pgturbohybrid_last_scan_state.proxyGraphNodesVisited;
	stats->proxyGraphEdgesVisited =
		pgturbohybrid_last_scan_state.proxyGraphEdgesVisited;
	stats->proxyGraphCandidatesSeen =
		pgturbohybrid_last_scan_state.proxyGraphCandidatesSeen;
	stats->proxyCandidatesReturned =
		pgturbohybrid_last_scan_state.proxyCandidatesReturned;
	stats->proxyVectorScoresComputed =
		pgturbohybrid_last_scan_state.proxyVectorScoresComputed;
	stats->proxyVectorScoreUs =
		pgturbohybrid_last_scan_state.proxyVectorScoreUs;
	stats->proxyCandidates =
		pgturbohybrid_last_scan_state.proxyCandidates;
	stats->proxyLazySidecarVectors =
		pgturbohybrid_last_scan_state.proxyLazySidecarVectors;
	strlcpy(stats->multivectorDocStorageCacheRequested,
			pgturbohybrid_last_scan_state.multivectorDocStorageCacheRequested,
			sizeof(stats->multivectorDocStorageCacheRequested));
	strlcpy(stats->multivectorDocStorageCacheEffective,
			pgturbohybrid_last_scan_state.multivectorDocStorageCacheEffective,
			sizeof(stats->multivectorDocStorageCacheEffective));
	stats->proxyTop1Admission =
		pgturbohybrid_last_scan_state.proxyTop1Admission;
	stats->proxyExactRerankDocs =
		pgturbohybrid_last_scan_state.proxyExactRerankDocs;
	stats->proxyFullSidecarVectorsLoaded =
		pgturbohybrid_last_scan_state.proxyFullSidecarVectorsLoaded;
	stats->proxyFullSidecarBytesTouched =
		pgturbohybrid_last_scan_state.proxyFullSidecarBytesTouched;
	stats->proxyFullSidecarPagesRead =
		pgturbohybrid_last_scan_state.proxyFullSidecarPagesRead;
	stats->proxyFullSidecarLoadUs =
		pgturbohybrid_last_scan_state.proxyFullSidecarLoadUs;
	stats->proxyFullSidecarReconstructUs =
		pgturbohybrid_last_scan_state.proxyFullSidecarReconstructUs;
	stats->proxyExactRerankHeapFetches =
		pgturbohybrid_last_scan_state.proxyExactRerankHeapFetches;
	stats->proxyExactRerankSidecarFetches =
		pgturbohybrid_last_scan_state.proxyExactRerankSidecarFetches;
	stats->proxyExactRerankBytesTouched =
		pgturbohybrid_last_scan_state.proxyExactRerankBytesTouched;
	stats->proxyExactRerankUs =
		pgturbohybrid_last_scan_state.proxyExactRerankUs;
	stats->sidecarCacheBuildThisQuery =
		pgturbohybrid_last_scan_state.sidecarCacheBuildThisQuery;
	stats->sidecarCacheBuildBytes =
		pgturbohybrid_last_scan_state.sidecarCacheBuildBytes;
	stats->sidecarCacheBuildPagesRead =
		pgturbohybrid_last_scan_state.sidecarCacheBuildPagesRead;
	stats->sidecarCacheBuildUs =
		pgturbohybrid_last_scan_state.sidecarCacheBuildUs;
	stats->sidecarQueryBytesTouched =
		pgturbohybrid_last_scan_state.sidecarQueryBytesTouched;
	stats->sidecarQueryPagesRead =
		pgturbohybrid_last_scan_state.sidecarQueryPagesRead;
	stats->sidecarQueryVectorsLoaded =
		pgturbohybrid_last_scan_state.sidecarQueryVectorsLoaded;
	stats->sidecarQueryLoadUs =
		pgturbohybrid_last_scan_state.sidecarQueryLoadUs;
	stats->sidecarQueryUs =
		pgturbohybrid_last_scan_state.sidecarQueryUs;
	stats->proxyVectorUsesFullSidecarForGraph =
		pgturbohybrid_last_scan_state.proxyVectorUsesFullSidecarForGraph;
	stats->proxyVectorNearExhaustiveSidecarTouch =
		pgturbohybrid_last_scan_state.proxyVectorNearExhaustiveSidecarTouch;
	strlcpy(stats->proxyVectorSidecarTouchReason,
			pgturbohybrid_last_scan_state.proxyVectorSidecarTouchReason,
			sizeof(stats->proxyVectorSidecarTouchReason));
	stats->centroidListsVisited =
		pgturbohybrid_last_scan_state.centroidListsVisited;
	stats->centroidDocsTouched =
		pgturbohybrid_last_scan_state.centroidDocsTouched;
	stats->centroidPrunedDocs =
		pgturbohybrid_last_scan_state.centroidPrunedDocs;
	stats->centroidPostingsTouched =
		pgturbohybrid_last_scan_state.centroidPostingsTouched;
	stats->centroidPostingsSelected =
		pgturbohybrid_last_scan_state.centroidPostingsSelected;
	stats->centroidPostingsSkipped =
		pgturbohybrid_last_scan_state.centroidPostingsSkipped;
	stats->centroidProbeUs =
		pgturbohybrid_last_scan_state.centroidProbeUs;
	stats->centroidPostingScanUs =
		pgturbohybrid_last_scan_state.centroidPostingScanUs;
	stats->centroidAccumulateUs =
		pgturbohybrid_last_scan_state.centroidAccumulateUs;
	stats->centroidCandidateHeapUs =
		pgturbohybrid_last_scan_state.centroidCandidateHeapUs;
	stats->centroidPostingLimitPerToken =
		pgturbohybrid_last_scan_state.centroidPostingLimitPerToken;
	stats->centroidProbeCentroidsPerToken =
		pgturbohybrid_last_scan_state.centroidProbeCentroidsPerToken;
	stats->centroidCodewordTopM =
		pgturbohybrid_last_scan_state.centroidCodewordTopM;
	stats->centroidScoreThreshold =
		pgturbohybrid_last_scan_state.centroidScoreThreshold;
	stats->centroidScoreDropFromBest =
		pgturbohybrid_last_scan_state.centroidScoreDropFromBest;
	stats->centroidListsSkippedByThreshold =
		pgturbohybrid_last_scan_state.centroidListsSkippedByThreshold;
	strlcpy(stats->centroidPostingCapStrategy,
			pgturbohybrid_last_scan_state.centroidPostingCapStrategy,
			sizeof(stats->centroidPostingCapStrategy));
	strlcpy(stats->centroidCandidateScoring,
			pgturbohybrid_last_scan_state.centroidCandidateScoring,
			sizeof(stats->centroidCandidateScoring));
	stats->centroidCandidates =
		pgturbohybrid_last_scan_state.centroidCandidates;
	stats->centroidBitsetPrefilterEnabled =
		pgturbohybrid_last_scan_state.centroidBitsetPrefilterEnabled;
	stats->centroidBitsetMinTokenMatches =
		pgturbohybrid_last_scan_state.centroidBitsetMinTokenMatches;
	stats->centroidBitsetListsUsed =
		pgturbohybrid_last_scan_state.centroidBitsetListsUsed;
	stats->centroidBitsetDocsSet =
		pgturbohybrid_last_scan_state.centroidBitsetDocsSet;
	stats->centroidBitsetDocsAfterThreshold =
		pgturbohybrid_last_scan_state.centroidBitsetDocsAfterThreshold;
	stats->centroidBitsetPrefilterUs =
		pgturbohybrid_last_scan_state.centroidBitsetPrefilterUs;
	stats->centroidBitsetMemoryBytes =
		pgturbohybrid_last_scan_state.centroidBitsetMemoryBytes;
	stats->centroidUpperBoundEnabled =
		pgturbohybrid_last_scan_state.centroidUpperBoundEnabled;
	stats->centroidUpperBoundDocsChecked =
		pgturbohybrid_last_scan_state.centroidUpperBoundDocsChecked;
	stats->centroidUpperBoundDocsPruned =
		pgturbohybrid_last_scan_state.centroidUpperBoundDocsPruned;
	stats->centroidUpperBoundPruneUs =
		pgturbohybrid_last_scan_state.centroidUpperBoundPruneUs;
	stats->centroidUpperBoundUnsafeFallbacks =
		pgturbohybrid_last_scan_state.centroidUpperBoundUnsafeFallbacks;
	stats->centroidCandidatesBeforeBound =
		pgturbohybrid_last_scan_state.centroidCandidatesBeforeBound;
	stats->centroidCandidatesAfterBound =
		pgturbohybrid_last_scan_state.centroidCandidatesAfterBound;
	stats->multivectorCentroidCount =
		pgturbohybrid_last_scan_state.multivectorCentroidCount;
	stats->multivectorCentroidPrerankDocs =
		pgturbohybrid_last_scan_state.multivectorCentroidPrerankDocs;
	stats->multivectorFullMaxsimRerankDocs =
		pgturbohybrid_last_scan_state.multivectorFullMaxsimRerankDocs;
	stats->quantizedInvertedListsVisited =
		pgturbohybrid_last_scan_state.quantizedInvertedListsVisited;
	stats->quantizedInvertedPostingsTouched =
		pgturbohybrid_last_scan_state.quantizedInvertedPostingsTouched;
	stats->quantizedInvertedPostingsSelected =
		pgturbohybrid_last_scan_state.quantizedInvertedPostingsSelected;
	stats->quantizedInvertedPostingsSkipped =
		pgturbohybrid_last_scan_state.quantizedInvertedPostingsSkipped;
	stats->quantizedInvertedPostingLimitPerToken =
		pgturbohybrid_last_scan_state.quantizedInvertedPostingLimitPerToken;
	stats->quantizedInvertedProbeCodewordsPerToken =
		pgturbohybrid_last_scan_state.quantizedInvertedProbeCodewordsPerToken;
	strlcpy(stats->quantizedInvertedPostingCapStrategy,
			pgturbohybrid_last_scan_state.quantizedInvertedPostingCapStrategy,
			sizeof(stats->quantizedInvertedPostingCapStrategy));
	stats->quantizedInvertedDocsScored =
		pgturbohybrid_last_scan_state.quantizedInvertedDocsScored;
	stats->quantizedInvertedCandidates =
		pgturbohybrid_last_scan_state.quantizedInvertedCandidates;
	stats->quantizedInvertedExactRerankDocs =
		pgturbohybrid_last_scan_state.quantizedInvertedExactRerankDocs;
	strlcpy(stats->quantizedInvertedCodebookSource,
			pgturbohybrid_last_scan_state.quantizedInvertedCodebookSource,
			sizeof(stats->quantizedInvertedCodebookSource));
	stats->quantizedInvertedCodebookSize =
		pgturbohybrid_last_scan_state.quantizedInvertedCodebookSize;
	stats->quantizedInvertedCodebookDim =
		pgturbohybrid_last_scan_state.quantizedInvertedCodebookDim;
	strlcpy(stats->quantizedInvertedCodebookChecksum,
			pgturbohybrid_last_scan_state.quantizedInvertedCodebookChecksum,
			sizeof(stats->quantizedInvertedCodebookChecksum));
	stats->quantizedInvertedCodebookTopM =
		pgturbohybrid_last_scan_state.quantizedInvertedCodebookTopM;
	stats->quantizedInvertedAssignmentUs =
		pgturbohybrid_last_scan_state.quantizedInvertedAssignmentUs;
	stats->quantizedInvertedQueryCodewordScoreUs =
		pgturbohybrid_last_scan_state.quantizedInvertedQueryCodewordScoreUs;
	strlcpy(stats->quantizedInvertedQueryCodewordKernel,
			pgturbohybrid_last_scan_state.quantizedInvertedQueryCodewordKernel,
			sizeof(stats->quantizedInvertedQueryCodewordKernel));
	stats->quantizedInvertedQueryCodewordScoresComputed =
		pgturbohybrid_last_scan_state.quantizedInvertedQueryCodewordScoresComputed;
	stats->quantizedInvertedQueryCodewordBlocks =
		pgturbohybrid_last_scan_state.quantizedInvertedQueryCodewordBlocks;
	stats->quantizedInvertedQueryCodewordTopkUs =
		pgturbohybrid_last_scan_state.quantizedInvertedQueryCodewordTopkUs;
	stats->quantizedInvertedQueryCodewordFullMatrixMaterialized =
		pgturbohybrid_last_scan_state.quantizedInvertedQueryCodewordFullMatrixMaterialized;
	stats->quantizedInvertedQueryCodewordActiveQueryTokens =
		pgturbohybrid_last_scan_state.quantizedInvertedQueryCodewordActiveQueryTokens;
	stats->quantizedInvertedQueryCodewordSkippedQueryTokens =
		pgturbohybrid_last_scan_state.quantizedInvertedQueryCodewordSkippedQueryTokens;
	stats->quantizedInvertedListOffsetBytes =
		pgturbohybrid_last_scan_state.quantizedInvertedListOffsetBytes;
	stats->quantizedInvertedPostingBytes =
		pgturbohybrid_last_scan_state.quantizedInvertedPostingBytes;
	stats->quantizedInvertedSidecarBytes =
		pgturbohybrid_last_scan_state.quantizedInvertedSidecarBytes;
	strlcpy(stats->quantizedInvertedCompactKernel,
			pgturbohybrid_last_scan_state.quantizedInvertedCompactKernel,
			sizeof(stats->quantizedInvertedCompactKernel));
	strlcpy(stats->quantizedInvertedCompactScoreSource,
			pgturbohybrid_last_scan_state.quantizedInvertedCompactScoreSource,
			sizeof(stats->quantizedInvertedCompactScoreSource));
	stats->quantizedInvertedCompactScoreUs =
		pgturbohybrid_last_scan_state.quantizedInvertedCompactScoreUs;
	stats->quantizedInvertedCompactDocsScored =
		pgturbohybrid_last_scan_state.quantizedInvertedCompactDocsScored;
	stats->quantizedInvertedCompactPayloadBytes =
		pgturbohybrid_last_scan_state.quantizedInvertedCompactPayloadBytes;
	strlcpy(stats->quantizedInvertedCompactDocOrder,
			pgturbohybrid_last_scan_state.quantizedInvertedCompactDocOrder,
			sizeof(stats->quantizedInvertedCompactDocOrder));
	stats->quantizedInvertedCompactInnerAllocations =
		pgturbohybrid_last_scan_state.quantizedInvertedCompactInnerAllocations;
	stats->quantizedInvertedCompactActiveQueryTokens =
		pgturbohybrid_last_scan_state.quantizedInvertedCompactActiveQueryTokens;
	stats->quantizedInvertedCompactPairsEvaluated =
		pgturbohybrid_last_scan_state.quantizedInvertedCompactPairsEvaluated;
	stats->quantizedInvertedCompactPairsSkipped =
		pgturbohybrid_last_scan_state.quantizedInvertedCompactPairsSkipped;
	stats->quantizedInvertedCompactPrefetches =
		pgturbohybrid_last_scan_state.quantizedInvertedCompactPrefetches;
	stats->quantizedInvertedCompactAvgDocTokens =
		pgturbohybrid_last_scan_state.quantizedInvertedCompactAvgDocTokens;
	stats->quantizedInvertedCompactUsPerDoc =
		pgturbohybrid_last_scan_state.quantizedInvertedCompactUsPerDoc;
	stats->quantizedInvertedCompactPayloadBytesPerDoc =
		pgturbohybrid_last_scan_state.quantizedInvertedCompactPayloadBytesPerDoc;
	stats->quantizedInvertedCompactTopKChangedVsScalar =
		pgturbohybrid_last_scan_state.quantizedInvertedCompactTopKChangedVsScalar;
	stats->quantizedInvertedPrecompactEnabled =
		pgturbohybrid_last_scan_state.quantizedInvertedPrecompactEnabled;
	strlcpy(stats->quantizedInvertedPrecompactMode,
			pgturbohybrid_last_scan_state.quantizedInvertedPrecompactMode,
			sizeof(stats->quantizedInvertedPrecompactMode));
	stats->quantizedInvertedDocsTouchedBeforePrecompact =
		pgturbohybrid_last_scan_state.quantizedInvertedDocsTouchedBeforePrecompact;
	stats->quantizedInvertedPrecompactScoreK =
		pgturbohybrid_last_scan_state.quantizedInvertedPrecompactScoreK;
	stats->quantizedInvertedPrecompactCoverageK =
		pgturbohybrid_last_scan_state.quantizedInvertedPrecompactCoverageK;
	stats->quantizedInvertedPrecompactPerTokenK =
		pgturbohybrid_last_scan_state.quantizedInvertedPrecompactPerTokenK;
	stats->quantizedInvertedCompactMaxDocs =
		pgturbohybrid_last_scan_state.quantizedInvertedCompactMaxDocs;
	stats->quantizedInvertedPrecompactScoreDocs =
		pgturbohybrid_last_scan_state.quantizedInvertedPrecompactScoreDocs;
	stats->quantizedInvertedPrecompactCoverageDocs =
		pgturbohybrid_last_scan_state.quantizedInvertedPrecompactCoverageDocs;
	stats->quantizedInvertedPrecompactPerTokenDocs =
		pgturbohybrid_last_scan_state.quantizedInvertedPrecompactPerTokenDocs;
	stats->quantizedInvertedPrecompactUnionDocs =
		pgturbohybrid_last_scan_state.quantizedInvertedPrecompactUnionDocs;
	stats->quantizedInvertedPrecompactDuplicates =
		pgturbohybrid_last_scan_state.quantizedInvertedPrecompactDuplicates;
	stats->quantizedInvertedPrecompactPrunedDocs =
		pgturbohybrid_last_scan_state.quantizedInvertedPrecompactPrunedDocs;
	stats->quantizedInvertedPrecompactUs =
		pgturbohybrid_last_scan_state.quantizedInvertedPrecompactUs;
	stats->quantizedInvertedCompactDocsSkippedByPrecompact =
		pgturbohybrid_last_scan_state.quantizedInvertedCompactDocsSkippedByPrecompact;
	strlcpy(stats->quantizedInvertedTokenCoverageMode,
			pgturbohybrid_last_scan_state.quantizedInvertedTokenCoverageMode,
			sizeof(stats->quantizedInvertedTokenCoverageMode));
	stats->quantizedInvertedActiveQueryTokens =
		pgturbohybrid_last_scan_state.quantizedInvertedActiveQueryTokens;
	stats->quantizedInvertedTokenMatchesTotal =
		pgturbohybrid_last_scan_state.quantizedInvertedTokenMatchesTotal;
	stats->quantizedInvertedTokenMatchesMax =
		pgturbohybrid_last_scan_state.quantizedInvertedTokenMatchesMax;
	stats->quantizedInvertedMinTokenMatches =
		pgturbohybrid_last_scan_state.quantizedInvertedMinTokenMatches;
	stats->quantizedInvertedTokenMatchFilteredDocs =
		pgturbohybrid_last_scan_state.quantizedInvertedTokenMatchFilteredDocs;
	stats->quantizedInvertedScoreBoundPruningEnabled =
		pgturbohybrid_last_scan_state.quantizedInvertedScoreBoundPruningEnabled;
	stats->quantizedInvertedScoreBoundDocsChecked =
		pgturbohybrid_last_scan_state.quantizedInvertedScoreBoundDocsChecked;
	stats->quantizedInvertedScoreBoundDocsPruned =
		pgturbohybrid_last_scan_state.quantizedInvertedScoreBoundDocsPruned;
	stats->quantizedInvertedScoreBoundPruneUs =
		pgturbohybrid_last_scan_state.quantizedInvertedScoreBoundPruneUs;
	stats->quantizedInvertedScoreBoundUnsafeFallbacks =
		pgturbohybrid_last_scan_state.quantizedInvertedScoreBoundUnsafeFallbacks;
	stats->quantizedInvertedCandidatesBeforeBound =
		pgturbohybrid_last_scan_state.quantizedInvertedCandidatesBeforeBound;
	stats->quantizedInvertedCandidatesAfterBound =
		pgturbohybrid_last_scan_state.quantizedInvertedCandidatesAfterBound;
	strlcpy(stats->multivectorDocSidecarCacheMode,
			pgturbohybrid_last_scan_state.multivectorDocSidecarCacheMode,
			sizeof(stats->multivectorDocSidecarCacheMode));
	stats->multivectorDocSidecarPagesRead =
		pgturbohybrid_last_scan_state.multivectorDocSidecarPagesRead;
	stats->multivectorDocSidecarCacheHits =
		pgturbohybrid_last_scan_state.multivectorDocSidecarCacheHits;
	stats->multivectorDocSidecarCacheMisses =
		pgturbohybrid_last_scan_state.multivectorDocSidecarCacheMisses;
	stats->multivectorDocSidecarBytesTouched =
		pgturbohybrid_last_scan_state.multivectorDocSidecarBytesTouched;
	stats->multivectorDocSidecarVectorsLoaded =
		pgturbohybrid_last_scan_state.multivectorDocSidecarVectorsLoaded;
	stats->multivectorDocSidecarDocMapPagesRead =
		pgturbohybrid_last_scan_state.multivectorDocSidecarDocMapPagesRead;
	stats->multivectorDocSidecarDocMapBytesTouched =
		pgturbohybrid_last_scan_state.multivectorDocSidecarDocMapBytesTouched;
	stats->multivectorDocSidecarResidentVectorsLoaded =
		pgturbohybrid_last_scan_state.multivectorDocSidecarResidentVectorsLoaded;
	stats->multivectorDocSidecarResidentBytesLoaded =
		pgturbohybrid_last_scan_state.multivectorDocSidecarResidentBytesLoaded;
	stats->multivectorDocSidecarVectorChunkRefBytesTouched =
		pgturbohybrid_last_scan_state.multivectorDocSidecarVectorChunkRefBytesTouched;
	stats->multivectorDocSidecarPagedVectorPagesRead =
		pgturbohybrid_last_scan_state.multivectorDocSidecarPagedVectorPagesRead;
	stats->multivectorDocSidecarPagedVectorBytesTouched =
		pgturbohybrid_last_scan_state.multivectorDocSidecarPagedVectorBytesTouched;
	stats->multivectorSidecarPageReadUs =
		pgturbohybrid_last_scan_state.multivectorSidecarPageReadUs;
	stats->multivectorSidecarVectorReconstructUs =
		pgturbohybrid_last_scan_state.multivectorSidecarVectorReconstructUs;
	stats->multivectorTokensOriginal =
		pgturbohybrid_last_scan_state.multivectorTokensOriginal;
	stats->multivectorTokensPooled =
		pgturbohybrid_last_scan_state.multivectorTokensPooled;
	stats->multivectorReservoirsEnabled =
		pgturbohybrid_last_scan_state.multivectorReservoirsEnabled;
	stats->multivectorReservoirScoreDocs =
		pgturbohybrid_last_scan_state.multivectorReservoirScoreDocs;
	stats->multivectorReservoirCoverageDocs =
		pgturbohybrid_last_scan_state.multivectorReservoirCoverageDocs;
	stats->multivectorReservoirMeanDocs =
		pgturbohybrid_last_scan_state.multivectorReservoirMeanDocs;
	stats->multivectorReservoirPerTokenDocs =
		pgturbohybrid_last_scan_state.multivectorReservoirPerTokenDocs;
	stats->multivectorReservoirBm25Docs =
		pgturbohybrid_last_scan_state.multivectorReservoirBm25Docs;
	stats->multivectorReservoirUnionDocs =
		pgturbohybrid_last_scan_state.multivectorReservoirUnionDocs;
	stats->multivectorReservoirDuplicates =
		pgturbohybrid_last_scan_state.multivectorReservoirDuplicates;
	stats->multivectorBm25InjectionEnabled =
		pgturbohybrid_last_scan_state.multivectorBm25InjectionEnabled;
	stats->multivectorBm25InjectionCandidates =
		pgturbohybrid_last_scan_state.multivectorBm25InjectionCandidates;
	stats->multivectorBm25InjectionCandidateLimit =
		pgturbohybrid_last_scan_state.multivectorBm25InjectionCandidateLimit;
	stats->multivectorBm25InjectionPoolSize =
		pgturbohybrid_last_scan_state.multivectorBm25InjectionPoolSize;
	strlcpy(stats->multivectorBm25InjectionLimitReason,
			pgturbohybrid_last_scan_state.multivectorBm25InjectionLimitReason,
			sizeof(stats->multivectorBm25InjectionLimitReason));
	stats->multivectorBm25InjectionRetained =
		pgturbohybrid_last_scan_state.multivectorBm25InjectionRetained;
	stats->multivectorBm25InjectionExactReranked =
		pgturbohybrid_last_scan_state.multivectorBm25InjectionExactReranked;
	stats->learnedSparseCandidates =
		pgturbohybrid_last_scan_state.learnedSparseCandidates;
	stats->learnedSparseRetainedForMaxsim =
		pgturbohybrid_last_scan_state.learnedSparseRetainedForMaxsim;
	stats->learnedSparseBranchLatencyUs =
		pgturbohybrid_last_scan_state.learnedSparseBranchLatencyUs;
	stats->multivectorDocMapBytes =
		pgturbohybrid_last_scan_state.multivectorDocMapBytes;
	stats->multivectorUniqueDocs =
		pgturbohybrid_last_scan_state.multivectorUniqueDocs;
	stats->multivectorDuplicateDocHits =
		pgturbohybrid_last_scan_state.multivectorDuplicateDocHits;
	stats->multivectorMaxsimUpdates =
		pgturbohybrid_last_scan_state.multivectorMaxsimUpdates;
	stats->multivectorDocCandidates =
		pgturbohybrid_last_scan_state.multivectorDocCandidates;
	stats->multivectorExactRerankEnabled =
		pgturbohybrid_last_scan_state.multivectorExactRerankEnabled;
	stats->multivectorExactRerankDocs =
		pgturbohybrid_last_scan_state.multivectorExactRerankDocs;
	stats->multivectorExactRerankPairs =
		pgturbohybrid_last_scan_state.multivectorExactRerankPairs;
	strlcpy(stats->multivectorExactRerankSource,
			pgturbohybrid_last_scan_state.multivectorExactRerankSource,
			sizeof(stats->multivectorExactRerankSource));
	stats->multivectorExactRerankHeapFetches =
		pgturbohybrid_last_scan_state.multivectorExactRerankHeapFetches;
	stats->multivectorExactRerankSidecarReads =
		pgturbohybrid_last_scan_state.multivectorExactRerankSidecarReads;
	stats->multivectorExactRerankSidecarBytes =
		pgturbohybrid_last_scan_state.multivectorExactRerankSidecarBytes;
	stats->multivectorCandidateSourceUs =
		pgturbohybrid_last_scan_state.multivectorCandidateSourceUs;
	stats->multivectorDocGraphTraversalUs =
		pgturbohybrid_last_scan_state.multivectorDocGraphTraversalUs;
	stats->multivectorProxyCandidateUs =
		pgturbohybrid_last_scan_state.multivectorProxyCandidateUs;
	stats->multivectorProxyGraphTraversalUs =
		pgturbohybrid_last_scan_state.multivectorProxyGraphTraversalUs;
	stats->multivectorProxyScoringUs =
		pgturbohybrid_last_scan_state.multivectorProxyScoringUs;
	stats->multivectorCentroidLitePostingUs =
		pgturbohybrid_last_scan_state.multivectorCentroidLitePostingUs;
	stats->multivectorQuantizedInvertedPostingUs =
		pgturbohybrid_last_scan_state.multivectorQuantizedInvertedPostingUs;
	stats->multivectorSidecarLoadUs =
		pgturbohybrid_last_scan_state.multivectorSidecarLoadUs;
	stats->multivectorHeapVisibilityUs =
		pgturbohybrid_last_scan_state.multivectorHeapVisibilityUs;
	stats->multivectorExactHeapFetchUs =
		pgturbohybrid_last_scan_state.multivectorExactHeapFetchUs;
	stats->multivectorExactRerankUs =
		pgturbohybrid_last_scan_state.multivectorExactRerankUs;
	stats->multivectorFinalSortUs =
		pgturbohybrid_last_scan_state.multivectorFinalSortUs;
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
	strlcpy(stats->multivectorExactKernel,
			pgturbohybrid_last_scan_state.multivectorExactKernel,
			sizeof(stats->multivectorExactKernel));
	strlcpy(stats->multivectorAccumulatorKind,
			pgturbohybrid_last_scan_state.multivectorAccumulatorKind,
			sizeof(stats->multivectorAccumulatorKind));
	stats->multivectorMemoryBytesEstimate =
		pgturbohybrid_last_scan_state.multivectorMemoryBytesEstimate;
	stats->multivectorAdmissionDebugEnabled =
		pgturbohybrid_last_scan_state.multivectorAdmissionDebugEnabled;
	stats->multivectorAdmissionCandidatesBeforeRerank =
		pgturbohybrid_last_scan_state.multivectorAdmissionCandidatesBeforeRerank;
	stats->multivectorAdmissionCandidatesAfterTruncation =
		pgturbohybrid_last_scan_state.multivectorAdmissionCandidatesAfterTruncation;
	stats->multivectorAdmissionExactRerankDocs =
		pgturbohybrid_last_scan_state.multivectorAdmissionExactRerankDocs;
	stats->multivectorAdmissionTruncatedByDocCandidateK =
		pgturbohybrid_last_scan_state.multivectorAdmissionTruncatedByDocCandidateK;
	stats->multivectorAdmissionTruncatedByAccumulatorMemory =
		pgturbohybrid_last_scan_state.multivectorAdmissionTruncatedByAccumulatorMemory;
	stats->multivectorAdmissionTraceAvailable =
		pgturbohybrid_last_scan_state.multivectorAdmissionTraceAvailable;
	stats->multivectorAdmissionTraceCount =
		pgturbohybrid_last_scan_state.multivectorAdmissionTraceCount;
	if (stats->multivectorAdmissionTraceCount >
		PGTURBOHYBRID_MULTIVECTOR_DEBUG_TRACE_LIMIT_MAX)
		stats->multivectorAdmissionTraceCount =
			PGTURBOHYBRID_MULTIVECTOR_DEBUG_TRACE_LIMIT_MAX;
	if (stats->multivectorAdmissionTraceCount > 0)
		memcpy(stats->multivectorAdmissionTrace,
			   pgturbohybrid_last_scan_state.multivectorAdmissionTrace,
			   sizeof(PgturbohybridMultiVectorAdmissionTraceEntry) *
			   stats->multivectorAdmissionTraceCount);
	stats->multivectorTokenStatsAvailable =
		pgturbohybrid_last_scan_state.multivectorTokenStatsAvailable;
	stats->multivectorTokenStatsCount =
		pgturbohybrid_last_scan_state.multivectorTokenStatsCount;
	if (stats->multivectorTokenStatsCount >
		PGTURBOHYBRID_MULTIVECTOR_TOKEN_STATS_LIMIT_MAX)
		stats->multivectorTokenStatsCount =
			PGTURBOHYBRID_MULTIVECTOR_TOKEN_STATS_LIMIT_MAX;
	if (stats->multivectorTokenStatsCount > 0)
		memcpy(stats->multivectorTokenStats,
			   pgturbohybrid_last_scan_state.multivectorTokenStats,
			   sizeof(PgturbohybridMultiVectorTokenStatsEntry) *
			   stats->multivectorTokenStatsCount);
	strlcpy(stats->finalDiversityMode,
			PgturbohybridFinalDiversityName(
				pgturbohybrid_last_scan_state.finalDiversityMode),
			sizeof(stats->finalDiversityMode));
	stats->finalDiversityPayloadSlot =
		pgturbohybrid_last_scan_state.finalDiversityPayloadSlot;
	stats->finalDiversityPoolSize =
		pgturbohybrid_last_scan_state.finalDiversityPoolSize;
	stats->finalDiversitySelected =
		pgturbohybrid_last_scan_state.finalDiversitySelected;
	stats->finalDiversityDuplicateGroupsSuppressed =
		pgturbohybrid_last_scan_state.finalDiversityDuplicateGroupsSuppressed;
	stats->finalDiversityUs =
		pgturbohybrid_last_scan_state.finalDiversityUs;
	stats->denseElapsedUs = pgturbohybrid_last_scan_state.denseElapsedUs;
	stats->bm25ElapsedUs = pgturbohybrid_last_scan_state.bm25ElapsedUs;
	stats->fusionElapsedUs = pgturbohybrid_last_scan_state.fusionElapsedUs;
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
		denseStats->multivectorAdmissionTruncatedByDocCandidateK ||
		denseStats->multivectorAdmissionTruncatedByAccumulatorMemory;
	documentLevelSource =
		strcmp(denseStats->multivectorCandidateSource, "document_nodes") == 0 ||
		strcmp(denseStats->multivectorCandidateSource, "proxy_vector") == 0 ||
		strcmp(denseStats->multivectorCandidateSource, "exact_doc_scan") == 0 ||
		strcmp(denseStats->multivectorCandidateSource, "doc_graph_prototype") == 0;
	exactDenseEvidence =
		denseStats->multivectorAdmissionExactRerankDocs >= (uint32) finalTarget ||
		denseStats->multivectorExactRerankDocs >= (uint32) finalTarget ||
		denseStats->multivectorDocGraphExactRerankDocs >= (uint32) finalTarget;

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
	source = denseStats->multivectorCandidateSource;
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
		denseStats->multivectorBm25InjectionEnabled;
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
			if (denseStats->multivectorDocGraphExactRerankDocs > 0)
				denseRescoreLimit =
					denseStats->multivectorDocGraphExactRerankDocs;
			else if (denseStats->multivectorExactRerankDocs > 0)
				denseRescoreLimit = denseStats->multivectorExactRerankDocs;
			denseTruncated =
				denseStats->multivectorAdmissionTruncatedByDocCandidateK ||
				denseStats->multivectorAdmissionTruncatedByAccumulatorMemory;
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
									   denseStats->multivectorDocGraphCandidates,
									   denseRescoreLimit, 2, denseScore,
									   PGTURBOHYBRID_BRANCH_SOURCE_DENSE |
									   PGTURBOHYBRID_BRANCH_SOURCE_MULTIVECTOR,
									   denseStats->multivectorDocGraphCandidates,
									   denseTruncated, 0);
			PgturbohybridBranchPlanAdd(plan,
									   PGTURBOHYBRID_BRANCH_KIND_EXACT_DOC_SCAN,
									   denseRescoreLimit, denseRescoreLimit,
									   3, denseScore,
									   PGTURBOHYBRID_BRANCH_SOURCE_DENSE |
									   PGTURBOHYBRID_BRANCH_SOURCE_MULTIVECTOR |
									   PGTURBOHYBRID_BRANCH_SOURCE_EXACT,
									   denseStats->multivectorDocGraphExactRerankDocs,
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
				denseStats->multivectorBm25InjectionExactReranked :
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

	stats->finalDiversityMode = pgturbohybrid_final_diversity;
	stats->finalDiversityPayloadSlot = payloadSlot;

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
		stats->finalDiversityUs = PgturbohybridElapsedUs(start);
		return false;
	}

	poolCount = (int) Min((int64) mergedCount,
						  (int64) selectionCount * (int64) poolMultiplier);
	if (poolCount <= selectionCount)
	{
		stats->finalDiversityUs = PgturbohybridElapsedUs(start);
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

	stats->finalDiversityPoolSize = poolCount;

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

			stats->finalDiversitySelected = selectedCount;
			stats->finalDiversityDuplicateGroupsSuppressed =
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

	stats->finalDiversityUs = PgturbohybridElapsedUs(start);
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
		stats->denseOnly++;
	else if (item->hasBm25)
		stats->bm25Only++;
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
	denseStats->multivectorBm25InjectionEnabled = false;
	denseStats->multivectorBm25InjectionCandidates = 0;
	denseStats->multivectorBm25InjectionCandidateLimit = 0;
	denseStats->multivectorBm25InjectionPoolSize = 0;
	strlcpy(denseStats->multivectorBm25InjectionLimitReason, "not_applicable",
			sizeof(denseStats->multivectorBm25InjectionLimitReason));
	denseStats->multivectorBm25InjectionRetained = 0;
	denseStats->multivectorBm25InjectionExactReranked = 0;
	denseStats->learnedSparseCandidates = 0;
	denseStats->learnedSparseRetainedForMaxsim = 0;
	denseStats->learnedSparseBranchLatencyUs = 0;

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

	denseStats->multivectorBm25InjectionEnabled = true;
	denseStats->multivectorBm25InjectionCandidates = (uint32) bm25Count;
	if (learnedSparse)
		denseStats->learnedSparseCandidates = (uint32) bm25Count;

	candidateLimit = PgturbohybridBudgetFinalTarget(query, autoBudgetLimit);
	if (*denseCount >= candidateLimit)
	{
		candidateLimit = *denseCount;
		strlcpy(denseStats->multivectorBm25InjectionLimitReason, "dense_count",
				sizeof(denseStats->multivectorBm25InjectionLimitReason));
	}
	else
		strlcpy(denseStats->multivectorBm25InjectionLimitReason,
				"final_target",
				sizeof(denseStats->multivectorBm25InjectionLimitReason));
	if (candidateLimit <= 0)
	{
		candidateLimit = bm25Count;
		strlcpy(denseStats->multivectorBm25InjectionLimitReason, "bm25_count",
				sizeof(denseStats->multivectorBm25InjectionLimitReason));
	}
	if (pgturbohybrid_multivector_doc_candidate_k > 0)
	{
		if (candidateLimit > pgturbohybrid_multivector_doc_candidate_k)
		{
			candidateLimit = pgturbohybrid_multivector_doc_candidate_k;
			strlcpy(denseStats->multivectorBm25InjectionLimitReason,
					"doc_candidate_k",
					sizeof(denseStats->multivectorBm25InjectionLimitReason));
		}
	}
	if (pgturbohybrid_multivector_exact_rerank_k > 0)
	{
		if (candidateLimit > pgturbohybrid_multivector_exact_rerank_k)
		{
			candidateLimit = pgturbohybrid_multivector_exact_rerank_k;
			strlcpy(denseStats->multivectorBm25InjectionLimitReason,
					"exact_rerank_k",
					sizeof(denseStats->multivectorBm25InjectionLimitReason));
		}
	}
	candidateLimit = Max(candidateLimit, 1);
	denseStats->multivectorBm25InjectionCandidateLimit = (uint32) candidateLimit;

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
	denseStats->multivectorBm25InjectionPoolSize = (uint32) poolCount;

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
	denseStats->denseCandidatesReturned = (uint32) newDenseCount;
	denseStats->multivectorDocCandidates = (uint32) newDenseCount;
	denseStats->multivectorExactRerankDocs += exactReranked;
	denseStats->multivectorExactRerankPairs += exactPairs;
	denseStats->exactRerankCandidates += exactReranked;
	denseStats->exactRerankTokensEvaluated +=
		(uint64) exactReranked * (uint64) queryMv->count;
	denseStats->multivectorBm25InjectionRetained = retained;
	denseStats->multivectorBm25InjectionExactReranked = exactReranked;
	if (learnedSparse)
		denseStats->learnedSparseRetainedForMaxsim = learnedSparseRetained;
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
		stats->calibratedFusionEnabled =
			fusion == PGTURBOHYBRID_FUSION_CALIBRATED;
		stats->dbsfEnabled = fusion == PGTURBOHYBRID_FUSION_DBSF;
		stats->dbsfDegenerateBranches = 0;
		memset(stats->dbsfBranchMean, 0, sizeof(stats->dbsfBranchMean));
		memset(stats->dbsfBranchStddev, 0, sizeof(stats->dbsfBranchStddev));
		memset(stats->dbsfBranchMin, 0, sizeof(stats->dbsfBranchMin));
		memset(stats->dbsfBranchMax, 0, sizeof(stats->dbsfBranchMax));
		stats->calibratedFusionBothMatchBonus =
			stats->calibratedFusionEnabled ?
			PgturbohybridClampUnit(
				pgturbohybrid_calibrated_fusion_both_match_bonus) : 0.0;
		strlcpy(stats->calibratedFusionDenseNormMode,
				stats->calibratedFusionEnabled ? "logistic" : "none",
				sizeof(stats->calibratedFusionDenseNormMode));
		strlcpy(stats->calibratedFusionBm25NormMode,
				stats->calibratedFusionEnabled ? "saturating" : "none",
				sizeof(stats->calibratedFusionBm25NormMode));
		if (stats->calibratedFusionEnabled)
		{
			int			queryShape;

			if (stats->calibratedFusionQueryShape[0] == '\0')
				strlcpy(stats->calibratedFusionQueryShape, "mixed",
						sizeof(stats->calibratedFusionQueryShape));
			queryShape = PgturbohybridHybridShapeFromName(
				stats->calibratedFusionQueryShape);
			alpha = PgturbohybridCalibratedFusionAlpha(query, queryShape);
		}
		stats->calibratedFusionAlphaEffective =
			stats->calibratedFusionEnabled ? alpha : 0.0;
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
			stats->dbsfBranchMean[0] = dbsfDense.mean;
			stats->dbsfBranchMean[1] = dbsfBm25.mean;
			stats->dbsfBranchStddev[0] = dbsfDense.stddev;
			stats->dbsfBranchStddev[1] = dbsfBm25.stddev;
			stats->dbsfBranchMin[0] = dbsfDense.min;
			stats->dbsfBranchMin[1] = dbsfBm25.min;
			stats->dbsfBranchMax[0] = dbsfDense.max;
			stats->dbsfBranchMax[1] = dbsfBm25.max;
			stats->dbsfDegenerateBranches =
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
													&stats->fusionGenerationArrayReused,
													&stats->fusionGenerationArrayReset);
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
			stats->fusionDuplicates++;

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
			stats->fusionDuplicates++;

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

	strlcpy(stats->fusionStrategy, "generation_array",
			sizeof(stats->fusionStrategy));
	if (mergedCountOut != NULL)
		*mergedCountOut = mergedCount;
	return PgturbohybridFinalizeFusedResults(scan, state, merged, mergedCount,
											 limit, false, memoryContext,
											 finalResults,
											 &stats->fusionHeapReplacements,
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
	lastStats->sparseBranchAvailable = sstats.branchAvailable;
	lastStats->sparseBranchUsed = sstats.branchUsed;
	lastStats->sparseTerms = sstats.terms;
	lastStats->sparseResolvedTerms = sstats.resolvedTerms;
	lastStats->sparsePostingsTouched = sstats.postingsTouched;
	lastStats->sparseCandidatesScored = sstats.candidatesScored;
	lastStats->sparseElapsedUs = sstats.elapsedUs;
	lastStats->sparseQuantBits = sstats.quantBits;
	lastStats->sparseQuantMode = sstats.quantMode;
	lastStats->sparseEncoding = sstats.encoding;
	lastStats->sparseScalarTailPostings = sstats.scalarTailPostings;
	lastStats->sparseRerankMode = sstats.rerankMode;
	lastStats->sparseExactRerankCount = sstats.exactRerankCount;
	lastStats->sparseExactRerankFetchUs = sstats.exactRerankFetchUs;
	lastStats->sparseExactRerankScoreUs = sstats.exactRerankScoreUs;
	lastStats->sparseExactRerankTopkChanged = sstats.exactRerankTopkChanged;
	lastStats->sparseScoreKernel = sstats.scoreKernel;
	lastStats->sparseSimdBlocks = sstats.simdBlocks;
	lastStats->sparseUsedWand = sstats.usedWand;
	lastStats->sparseBlocksVisited = sstats.blocksVisited;
	lastStats->sparseBlocksSkipped = sstats.blocksSkipped;
	lastStats->sparseWandPruned = sstats.wandPruned;
	lastStats->sparseWandIterations = sstats.wandIterations;
	lastStats->sparseWandThresholdUpdates = sstats.wandThresholdUpdates;
	lastStats->sparseWandHeapUpdates = sstats.wandHeapUpdates;
	lastStats->sparseCacheHit = sstats.cacheHit;
	lastStats->sparseCacheBuildUs = sstats.cacheBuildUs;
	lastStats->sparseCacheBytes = sstats.cacheBytes;
	lastStats->sparseHotCacheHits = sstats.hotCacheHits;
	lastStats->sparseHotCacheMisses = sstats.hotCacheMisses;
	lastStats->sparseHotCacheBytes = sstats.hotCacheBytes;
	lastStats->sparseHotCacheEvictions = sstats.hotCacheEvictions;
	lastStats->sparseDeltaPages = sstats.deltaPages;
	lastStats->sparseDeltaTerms = sstats.deltaTerms;
	lastStats->sparseDeltaPostingsDecoded = sstats.deltaPostingsDecoded;
	lastStats->sparseDeltaCacheHit = sstats.deltaCacheHit;
	lastStats->sparseDeltaGeneration = sstats.deltaGeneration;
	lastStats->sparseCandidatesRequested = originalQuery->sparseK;
	lastStats->sparseCandidatesEffective = scanQuery->sparseK;
	lastStats->sparseKDefaulted =
		(originalQuery->flags & PGTURBOHYBRID_QUERY_FLAG_SPARSE_K_DEFAULTED) != 0;
	lastStats->sparseCandidates = n;
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
	lastStats.finalDiversityMode = PGTURBOHYBRID_FINAL_DIVERSITY_OFF;
	lastStats.finalDiversityPayloadSlot = -1;
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
	strlcpy(lastStats.hybridQueryShape,
			PgturbohybridHybridQueryShapeName(hybridQueryShapeForStats),
			sizeof(lastStats.hybridQueryShape));
	strlcpy(lastStats.calibratedFusionQueryShape,
			PgturbohybridHybridQueryShapeName(hybridQueryShapeForStats),
			sizeof(lastStats.calibratedFusionQueryShape));
	if (denseProbe != NULL && denseProbeCount > 0 &&
		denseProbeK >= scanQuery->denseK)
	{
		dense = denseProbe;
		denseCount = denseProbeCount;
		denseStats = denseProbeStats;
		lastStats.denseElapsedUs = denseProbeElapsedUs;
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
		lastStats.denseElapsedUs = denseProbeElapsedUs +
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
		lastStats.bm25ElapsedUs = PgturbohybridElapsedUs(phaseStart);
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
			denseStats.learnedSparseBranchLatencyUs = lastStats.bm25ElapsedUs;
		else
			multivectorStats.learnedSparseBranchLatencyUs = lastStats.bm25ElapsedUs;
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
		lastStats.sparseElapsedUs = PgturbohybridElapsedUs(phaseStart);
	}

	INSTR_TIME_SET_CURRENT(phaseStart);
	fusionCandidatesSeen = denseCount + multivectorCount + bm25Count + sparseCount;
	lastStats.fusionCandidatesSeen = fusionCandidatesSeen;
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
		lastStats.fusionHeapSize = finalCount;
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
				lastStats.fusionDuplicates++;
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
				lastStats.fusionDuplicates++;
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
				lastStats.fusionDuplicates++;
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
													   &lastStats.fusionHeapReplacements,
													   &lastStats);
		strlcpy(lastStats.fusionStrategy,
				useDocumentFusionKey ? "hash_doc" : "hash",
				sizeof(lastStats.fusionStrategy));
		lastStats.fusionHeapSize = finalCount;
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
				lastStats.fusionDuplicates++;
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
													   &lastStats.fusionHeapReplacements,
													   &lastStats);
		strlcpy(lastStats.fusionStrategy,
				useDocumentFusionKey ? "sorted_merge_doc" : "sorted_merge",
				sizeof(lastStats.fusionStrategy));
		lastStats.fusionHeapSize = finalCount;
	}

	lastStats.fusionElapsedUs = PgturbohybridElapsedUs(phaseStart);
	state->results = merged;
	state->resultCount = finalCount;
	state->resultIndex = 0;
	state->collectDone = true;

	strlcpy(lastStats.indexShape, hasLexicalKey ? "hybrid" : "dense_only",
			sizeof(lastStats.indexShape));
	lastStats.bm25BranchAvailable = hasLexicalKey;
	lastStats.denseBranchUsed = denseBranchUsed;
	lastStats.multivectorBranchUsed = multivectorBranchUsed;
	lastStats.bm25BranchUsed = bm25BranchUsed;
	lastStats.sparseBranchAvailable = sparseStats.branchAvailable;
	lastStats.sparseBranchUsed = sparseBranchUsed;
	lastStats.sparseTerms = sparseStats.terms;
	lastStats.sparseResolvedTerms = sparseStats.resolvedTerms;
	lastStats.sparsePostingsTouched = sparseStats.postingsTouched;
	lastStats.sparseCandidatesScored = sparseStats.candidatesScored;
	lastStats.sparseQuantBits = sparseStats.quantBits;
	lastStats.sparseQuantMode = sparseStats.quantMode;
	lastStats.sparseEncoding = sparseStats.encoding;
	lastStats.sparseScalarTailPostings = sparseStats.scalarTailPostings;
	lastStats.sparseRerankMode = sparseStats.rerankMode;
	lastStats.sparseExactRerankCount = sparseStats.exactRerankCount;
	lastStats.sparseExactRerankFetchUs = sparseStats.exactRerankFetchUs;
	lastStats.sparseExactRerankScoreUs = sparseStats.exactRerankScoreUs;
	lastStats.sparseExactRerankTopkChanged = sparseStats.exactRerankTopkChanged;
	lastStats.sparseScoreKernel = sparseStats.scoreKernel;
	lastStats.sparseSimdBlocks = sparseStats.simdBlocks;
	lastStats.sparseUsedWand = sparseStats.usedWand;
	lastStats.sparseBlocksVisited = sparseStats.blocksVisited;
	lastStats.sparseBlocksSkipped = sparseStats.blocksSkipped;
	lastStats.sparseWandPruned = sparseStats.wandPruned;
	lastStats.sparseWandIterations = sparseStats.wandIterations;
	lastStats.sparseWandThresholdUpdates = sparseStats.wandThresholdUpdates;
	lastStats.sparseWandHeapUpdates = sparseStats.wandHeapUpdates;
	lastStats.sparseCacheHit = sparseStats.cacheHit;
	lastStats.sparseCacheBuildUs = sparseStats.cacheBuildUs;
	lastStats.sparseCacheBytes = sparseStats.cacheBytes;
	lastStats.sparseHotCacheHits = sparseStats.hotCacheHits;
	lastStats.sparseHotCacheMisses = sparseStats.hotCacheMisses;
	lastStats.sparseHotCacheBytes = sparseStats.hotCacheBytes;
	lastStats.sparseHotCacheEvictions = sparseStats.hotCacheEvictions;
	lastStats.sparseDeltaPages = sparseStats.deltaPages;
	lastStats.sparseDeltaTerms = sparseStats.deltaTerms;
	lastStats.sparseDeltaPostingsDecoded = sparseStats.deltaPostingsDecoded;
	lastStats.sparseDeltaCacheHit = sparseStats.deltaCacheHit;
	lastStats.sparseDeltaGeneration = sparseStats.deltaGeneration;
	if (hasSparseQuery)
	{
		lastStats.sparseCandidatesRequested = originalQuery->sparseK;
		lastStats.sparseCandidatesEffective = scanQuery->sparseK;
		lastStats.sparseKDefaulted =
			(originalQuery->flags & PGTURBOHYBRID_QUERY_FLAG_SPARSE_K_DEFAULTED) != 0;
		lastStats.sparseCandidates = sparseCount;
	}
	strlcpy(lastStats.profile, PgturbohybridProfileName(pgturbohybrid_profile),
			sizeof(lastStats.profile));
	strlcpy(lastStats.fusion,
			PgturbohybridQueryFusionName(pgturbohybrid_force_fusion != 0 ?
								  pgturbohybrid_force_fusion : scanQuery->fusion),
			sizeof(lastStats.fusion));
	if (denseBranchUsed)
	{
		lastStats.denseCandidatesRequested = originalQuery->denseK;
		lastStats.denseCandidatesEffective = scanQuery->denseK;
		lastStats.denseKDefaulted =
			(originalQuery->flags & PGTURBOHYBRID_QUERY_FLAG_DENSE_K_DEFAULTED) != 0;
		lastStats.denseCandidates = denseCount;
		lastStats.denseEffectiveResultTarget = denseStats.effectiveResultTarget;
		lastStats.denseEffectiveSearchEf = denseStats.effectiveSearchEf;
		lastStats.denseEffectiveRescoreBand = denseStats.effectiveRescoreBand;
		lastStats.denseHighdimWideningMultiplier =
			denseStats.highdimWideningMultiplier;
		lastStats.denseWideningReason = denseStats.wideningReason;
		lastStats.denseBudgetPolicy = denseStats.denseBudgetPolicy;
		lastStats.denseRescoreBandPolicy = denseStats.rescoreBandPolicy;
	}
	lastStats.multivectorCandidatesRequested = originalQuery->multivectorK;
	lastStats.multivectorCandidatesEffective = scanQuery->multivectorK;
	lastStats.multivectorCandidates =
		multivectorMergedAsDense ? denseCount : multivectorCount;
	if (hasMultivectorQuery && !multivectorMergedAsDense)
		denseStats = multivectorStats;
	lastStats.bm25CandidatesRequested = originalQuery->bm25K;
	lastStats.bm25CandidatesEffective = bm25BudgetEffectiveK;
	lastStats.bm25KDefaulted =
		(originalQuery->flags & PGTURBOHYBRID_QUERY_FLAG_BM25_K_DEFAULTED) != 0;
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
	lastStats.fusionCandidatesSeen = fusionCandidatesSeen;
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
	lastStats.multivectorEnabled = denseStats.multivectorEnabled;
	lastStats.multivectorQueryVectors = denseStats.multivectorQueryVectors;
	lastStats.multivectorDocVectorsLimit =
		denseStats.multivectorDocVectorsLimit;
	lastStats.multivectorSubvectorSearches =
		denseStats.multivectorSubvectorSearches;
	lastStats.multivectorRawSubvectorHits =
		denseStats.multivectorRawSubvectorHits;
	lastStats.multivectorAdaptiveWideningTriggered =
		denseStats.multivectorAdaptiveWideningTriggered;
	lastStats.multivectorAdaptiveInitialRawTarget =
		denseStats.multivectorAdaptiveInitialRawTarget;
	lastStats.multivectorAdaptiveFinalRawTarget =
		denseStats.multivectorAdaptiveFinalRawTarget;
	strlcpy(lastStats.multivectorCandidateSource,
			denseStats.multivectorCandidateSource,
			sizeof(lastStats.multivectorCandidateSource));
	strlcpy(lastStats.multivectorCandidatePath,
			denseStats.multivectorCandidatePath,
			sizeof(lastStats.multivectorCandidatePath));
	strlcpy(lastStats.multivectorProxyEncoderKind,
			denseStats.multivectorProxyEncoderKind,
			sizeof(lastStats.multivectorProxyEncoderKind));
	lastStats.learnedProjectionLoaded = denseStats.learnedProjectionLoaded;
	lastStats.learnedProjectionDim = denseStats.learnedProjectionDim;
	lastStats.learnedProjectionWeightBytes =
		denseStats.learnedProjectionWeightBytes;
	strlcpy(lastStats.learnedProjectionModel,
			denseStats.learnedProjectionModel,
			sizeof(lastStats.learnedProjectionModel));
	strlcpy(lastStats.learnedProjectionChecksum,
			denseStats.learnedProjectionChecksum,
			sizeof(lastStats.learnedProjectionChecksum));
	lastStats.learnedProjectionQueryEncodeUs =
		denseStats.learnedProjectionQueryEncodeUs;
	strlcpy(lastStats.multivectorGraphMode,
			denseStats.multivectorGraphMode,
			sizeof(lastStats.multivectorGraphMode));
	lastStats.multivectorProxyGraphSearches =
		denseStats.multivectorProxyGraphSearches;
	lastStats.multivectorExactTokenScanEnabled =
		denseStats.multivectorExactTokenScanEnabled;
	lastStats.multivectorExactTokenScanNodesScored =
		denseStats.multivectorExactTokenScanNodesScored;
	lastStats.multivectorPlainFallbackUsed =
		denseStats.multivectorPlainFallbackUsed;
	strlcpy(lastStats.multivectorPlainFallbackReason,
			denseStats.multivectorPlainFallbackReason,
			sizeof(lastStats.multivectorPlainFallbackReason));
	lastStats.multivectorPlainFallbackDocsScored =
		denseStats.multivectorPlainFallbackDocsScored;
	lastStats.multivectorPlainFallbackPairs =
		denseStats.multivectorPlainFallbackPairs;
	lastStats.multivectorDocGraphPrototypeEnabled =
		denseStats.multivectorDocGraphPrototypeEnabled;
	lastStats.multivectorDocGraphNodes =
		denseStats.multivectorDocGraphNodes;
	lastStats.multivectorDocGraphDocsScored =
		denseStats.multivectorDocGraphDocsScored;
	lastStats.multivectorDocGraphEdgesVisited =
		denseStats.multivectorDocGraphEdgesVisited;
	lastStats.multivectorDocGraphCandidates =
		denseStats.multivectorDocGraphCandidates;
	lastStats.multivectorDocGraphSearchEf =
		denseStats.multivectorDocGraphSearchEf;
	lastStats.multivectorDocGraphOversampling =
		denseStats.multivectorDocGraphOversampling;
	lastStats.multivectorDocGraphRescoreK =
		denseStats.multivectorDocGraphRescoreK;
	lastStats.multivectorDocGraphEntrySampleConfigured =
		denseStats.multivectorDocGraphEntrySampleConfigured;
	lastStats.multivectorDocGraphEntrySampleEffective =
		denseStats.multivectorDocGraphEntrySampleEffective;
	lastStats.multivectorDocGraphEntrySampleScored =
		denseStats.multivectorDocGraphEntrySampleScored;
	lastStats.multivectorDocGraphQuantizedScores =
		denseStats.multivectorDocGraphQuantizedScores;
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
	strlcpy(lastStats.multivectorDocGraphStorageKind,
			denseStats.multivectorDocGraphStorageKind,
			sizeof(lastStats.multivectorDocGraphStorageKind));
	lastStats.proxyOnlyIndex = denseStats.proxyOnlyIndex;
	lastStats.centroidOnlyIndex = denseStats.centroidOnlyIndex;
	lastStats.fullMultivectorSidecarAvailable =
		denseStats.fullMultivectorSidecarAvailable;
	lastStats.centroidSidecarAvailable =
		denseStats.centroidSidecarAvailable;
	lastStats.centroidDocCodesAvailable =
		denseStats.centroidDocCodesAvailable;
	lastStats.quantizedInvertedSidecarAvailable =
		denseStats.quantizedInvertedSidecarAvailable;
	strlcpy(lastStats.multivectorDocGraphRescoreSource,
			denseStats.multivectorDocGraphRescoreSource,
			sizeof(lastStats.multivectorDocGraphRescoreSource));
	lastStats.multivectorDocGraphExactRerankDocs =
		denseStats.multivectorDocGraphExactRerankDocs;
	lastStats.multivectorDocGraphHeapFetches =
		denseStats.multivectorDocGraphHeapFetches;
	strlcpy(lastStats.multivectorDocGraphWarning,
			denseStats.multivectorDocGraphWarning,
			sizeof(lastStats.multivectorDocGraphWarning));
	lastStats.multivectorProxyCandidateTarget =
		denseStats.multivectorProxyCandidateTarget;
	lastStats.multivectorProxyCandidatesReturned =
		denseStats.multivectorProxyCandidatesReturned;
	lastStats.multivectorExactRerankKEffective =
		denseStats.multivectorExactRerankKEffective;
	lastStats.proxyCandidateLimitEffective =
		denseStats.proxyCandidateLimitEffective;
	strlcpy(lastStats.proxyCandidateLimitSource,
			denseStats.proxyCandidateLimitSource,
			sizeof(lastStats.proxyCandidateLimitSource));
	lastStats.proxyGraphNodesVisited = denseStats.proxyGraphNodesVisited;
	lastStats.proxyGraphEdgesVisited = denseStats.proxyGraphEdgesVisited;
	lastStats.proxyGraphCandidatesSeen = denseStats.proxyGraphCandidatesSeen;
	lastStats.proxyCandidatesReturned = denseStats.proxyCandidatesReturned;
	lastStats.proxyVectorScoresComputed = denseStats.proxyVectorScoresComputed;
	lastStats.proxyVectorScoreUs = denseStats.proxyVectorScoreUs;
	lastStats.proxyCandidates = denseStats.proxyCandidates;
	lastStats.proxyLazySidecarVectors = denseStats.proxyLazySidecarVectors;
	strlcpy(lastStats.multivectorDocStorageCacheRequested,
			denseStats.multivectorDocStorageCacheRequested,
			sizeof(lastStats.multivectorDocStorageCacheRequested));
	strlcpy(lastStats.multivectorDocStorageCacheEffective,
			denseStats.multivectorDocStorageCacheEffective,
			sizeof(lastStats.multivectorDocStorageCacheEffective));
	lastStats.proxyTop1Admission = denseStats.proxyTop1Admission;
	lastStats.proxyExactRerankDocs = denseStats.proxyExactRerankDocs;
	lastStats.proxyFullSidecarVectorsLoaded =
		denseStats.proxyFullSidecarVectorsLoaded;
	lastStats.proxyFullSidecarBytesTouched =
		denseStats.proxyFullSidecarBytesTouched;
	lastStats.proxyFullSidecarPagesRead =
		denseStats.proxyFullSidecarPagesRead;
	lastStats.proxyFullSidecarLoadUs =
		denseStats.proxyFullSidecarLoadUs;
	lastStats.proxyFullSidecarReconstructUs =
		denseStats.proxyFullSidecarReconstructUs;
	lastStats.proxyExactRerankHeapFetches =
		denseStats.proxyExactRerankHeapFetches;
	lastStats.proxyExactRerankSidecarFetches =
		denseStats.proxyExactRerankSidecarFetches;
	lastStats.proxyExactRerankBytesTouched =
		denseStats.proxyExactRerankBytesTouched;
	lastStats.proxyExactRerankUs = denseStats.proxyExactRerankUs;
	lastStats.sidecarCacheBuildThisQuery =
		denseStats.sidecarCacheBuildThisQuery;
	lastStats.sidecarCacheBuildBytes = denseStats.sidecarCacheBuildBytes;
	lastStats.sidecarCacheBuildPagesRead =
		denseStats.sidecarCacheBuildPagesRead;
	lastStats.sidecarCacheBuildUs = denseStats.sidecarCacheBuildUs;
	lastStats.sidecarQueryBytesTouched =
		denseStats.sidecarQueryBytesTouched;
	lastStats.sidecarQueryPagesRead = denseStats.sidecarQueryPagesRead;
	lastStats.sidecarQueryVectorsLoaded =
		denseStats.sidecarQueryVectorsLoaded;
	lastStats.sidecarQueryLoadUs = denseStats.sidecarQueryLoadUs;
	lastStats.sidecarQueryUs = denseStats.sidecarQueryUs;
	lastStats.proxyVectorUsesFullSidecarForGraph =
		denseStats.proxyVectorUsesFullSidecarForGraph;
	lastStats.proxyVectorNearExhaustiveSidecarTouch =
		denseStats.proxyVectorNearExhaustiveSidecarTouch;
	strlcpy(lastStats.proxyVectorSidecarTouchReason,
			denseStats.proxyVectorSidecarTouchReason,
			sizeof(lastStats.proxyVectorSidecarTouchReason));
	lastStats.centroidListsVisited = denseStats.centroidListsVisited;
	lastStats.centroidDocsTouched = denseStats.centroidDocsTouched;
	lastStats.centroidPrunedDocs = denseStats.centroidPrunedDocs;
	lastStats.centroidPostingsTouched = denseStats.centroidPostingsTouched;
	lastStats.centroidPostingsSelected = denseStats.centroidPostingsSelected;
	lastStats.centroidPostingsSkipped = denseStats.centroidPostingsSkipped;
	lastStats.centroidProbeUs = denseStats.centroidProbeUs;
	lastStats.centroidPostingScanUs = denseStats.centroidPostingScanUs;
	lastStats.centroidAccumulateUs = denseStats.centroidAccumulateUs;
	lastStats.centroidCandidateHeapUs = denseStats.centroidCandidateHeapUs;
	lastStats.centroidPostingLimitPerToken =
		denseStats.centroidPostingLimitPerToken;
	lastStats.centroidProbeCentroidsPerToken =
		denseStats.centroidProbeCentroidsPerToken;
	lastStats.centroidCodewordTopM = denseStats.centroidCodewordTopM;
	lastStats.centroidScoreThreshold = denseStats.centroidScoreThreshold;
	lastStats.centroidScoreDropFromBest = denseStats.centroidScoreDropFromBest;
	lastStats.centroidListsSkippedByThreshold =
		denseStats.centroidListsSkippedByThreshold;
	strlcpy(lastStats.centroidPostingCapStrategy,
			denseStats.centroidPostingCapStrategy,
			sizeof(lastStats.centroidPostingCapStrategy));
	strlcpy(lastStats.centroidCandidateScoring,
			denseStats.centroidCandidateScoring,
			sizeof(lastStats.centroidCandidateScoring));
	lastStats.centroidCandidates = denseStats.centroidCandidates;
	lastStats.centroidBitsetPrefilterEnabled =
		denseStats.centroidBitsetPrefilterEnabled;
	lastStats.centroidBitsetMinTokenMatches =
		denseStats.centroidBitsetMinTokenMatches;
	lastStats.centroidBitsetListsUsed =
		denseStats.centroidBitsetListsUsed;
	lastStats.centroidBitsetDocsSet = denseStats.centroidBitsetDocsSet;
	lastStats.centroidBitsetDocsAfterThreshold =
		denseStats.centroidBitsetDocsAfterThreshold;
	lastStats.centroidBitsetPrefilterUs =
		denseStats.centroidBitsetPrefilterUs;
	lastStats.centroidBitsetMemoryBytes =
		denseStats.centroidBitsetMemoryBytes;
	lastStats.centroidUpperBoundEnabled =
		denseStats.centroidUpperBoundEnabled;
	lastStats.centroidUpperBoundDocsChecked =
		denseStats.centroidUpperBoundDocsChecked;
	lastStats.centroidUpperBoundDocsPruned =
		denseStats.centroidUpperBoundDocsPruned;
	lastStats.centroidUpperBoundPruneUs =
		denseStats.centroidUpperBoundPruneUs;
	lastStats.centroidUpperBoundUnsafeFallbacks =
		denseStats.centroidUpperBoundUnsafeFallbacks;
	lastStats.centroidCandidatesBeforeBound =
		denseStats.centroidCandidatesBeforeBound;
	lastStats.centroidCandidatesAfterBound =
		denseStats.centroidCandidatesAfterBound;
	lastStats.multivectorCentroidCount =
		denseStats.multivectorCentroidCount;
	lastStats.multivectorCentroidPrerankDocs =
		denseStats.multivectorCentroidPrerankDocs;
	lastStats.multivectorFullMaxsimRerankDocs =
		denseStats.multivectorFullMaxsimRerankDocs;
	lastStats.quantizedInvertedListsVisited =
		denseStats.quantizedInvertedListsVisited;
	lastStats.quantizedInvertedPostingsTouched =
		denseStats.quantizedInvertedPostingsTouched;
	lastStats.quantizedInvertedPostingsSelected =
		denseStats.quantizedInvertedPostingsSelected;
	lastStats.quantizedInvertedPostingsSkipped =
		denseStats.quantizedInvertedPostingsSkipped;
	lastStats.quantizedInvertedPostingLimitPerToken =
		denseStats.quantizedInvertedPostingLimitPerToken;
	lastStats.quantizedInvertedProbeCodewordsPerToken =
		denseStats.quantizedInvertedProbeCodewordsPerToken;
	strlcpy(lastStats.quantizedInvertedPostingCapStrategy,
			denseStats.quantizedInvertedPostingCapStrategy,
			sizeof(lastStats.quantizedInvertedPostingCapStrategy));
	lastStats.quantizedInvertedDocsScored =
		denseStats.quantizedInvertedDocsScored;
	lastStats.quantizedInvertedCandidates =
		denseStats.quantizedInvertedCandidates;
	lastStats.quantizedInvertedExactRerankDocs =
		denseStats.quantizedInvertedExactRerankDocs;
	strlcpy(lastStats.quantizedInvertedCodebookSource,
			denseStats.quantizedInvertedCodebookSource,
			sizeof(lastStats.quantizedInvertedCodebookSource));
	lastStats.quantizedInvertedCodebookSize =
		denseStats.quantizedInvertedCodebookSize;
	lastStats.quantizedInvertedCodebookDim =
		denseStats.quantizedInvertedCodebookDim;
	strlcpy(lastStats.quantizedInvertedCodebookChecksum,
			denseStats.quantizedInvertedCodebookChecksum,
			sizeof(lastStats.quantizedInvertedCodebookChecksum));
	lastStats.quantizedInvertedCodebookTopM =
		denseStats.quantizedInvertedCodebookTopM;
	lastStats.quantizedInvertedAssignmentUs =
		denseStats.quantizedInvertedAssignmentUs;
	lastStats.quantizedInvertedQueryCodewordScoreUs =
		denseStats.quantizedInvertedQueryCodewordScoreUs;
	strlcpy(lastStats.quantizedInvertedQueryCodewordKernel,
			denseStats.quantizedInvertedQueryCodewordKernel,
			sizeof(lastStats.quantizedInvertedQueryCodewordKernel));
	lastStats.quantizedInvertedQueryCodewordScoresComputed =
		denseStats.quantizedInvertedQueryCodewordScoresComputed;
	lastStats.quantizedInvertedQueryCodewordBlocks =
		denseStats.quantizedInvertedQueryCodewordBlocks;
	lastStats.quantizedInvertedQueryCodewordTopkUs =
		denseStats.quantizedInvertedQueryCodewordTopkUs;
	lastStats.quantizedInvertedQueryCodewordFullMatrixMaterialized =
		denseStats.quantizedInvertedQueryCodewordFullMatrixMaterialized;
	lastStats.quantizedInvertedQueryCodewordActiveQueryTokens =
		denseStats.quantizedInvertedQueryCodewordActiveQueryTokens;
	lastStats.quantizedInvertedQueryCodewordSkippedQueryTokens =
		denseStats.quantizedInvertedQueryCodewordSkippedQueryTokens;
	lastStats.quantizedInvertedListOffsetBytes =
		denseStats.quantizedInvertedListOffsetBytes;
	lastStats.quantizedInvertedPostingBytes =
		denseStats.quantizedInvertedPostingBytes;
	lastStats.quantizedInvertedSidecarBytes =
		denseStats.quantizedInvertedSidecarBytes;
	strlcpy(lastStats.quantizedInvertedCompactKernel,
			denseStats.quantizedInvertedCompactKernel,
			sizeof(lastStats.quantizedInvertedCompactKernel));
	strlcpy(lastStats.quantizedInvertedCompactScoreSource,
			denseStats.quantizedInvertedCompactScoreSource,
			sizeof(lastStats.quantizedInvertedCompactScoreSource));
	lastStats.quantizedInvertedCompactScoreUs =
		denseStats.quantizedInvertedCompactScoreUs;
	lastStats.quantizedInvertedCompactDocsScored =
		denseStats.quantizedInvertedCompactDocsScored;
	lastStats.quantizedInvertedCompactPayloadBytes =
		denseStats.quantizedInvertedCompactPayloadBytes;
	strlcpy(lastStats.quantizedInvertedCompactDocOrder,
			denseStats.quantizedInvertedCompactDocOrder,
			sizeof(lastStats.quantizedInvertedCompactDocOrder));
	lastStats.quantizedInvertedCompactInnerAllocations =
		denseStats.quantizedInvertedCompactInnerAllocations;
	lastStats.quantizedInvertedCompactActiveQueryTokens =
		denseStats.quantizedInvertedCompactActiveQueryTokens;
	lastStats.quantizedInvertedCompactPairsEvaluated =
		denseStats.quantizedInvertedCompactPairsEvaluated;
	lastStats.quantizedInvertedCompactPairsSkipped =
		denseStats.quantizedInvertedCompactPairsSkipped;
	lastStats.quantizedInvertedCompactPrefetches =
		denseStats.quantizedInvertedCompactPrefetches;
	lastStats.quantizedInvertedCompactAvgDocTokens =
		denseStats.quantizedInvertedCompactAvgDocTokens;
	lastStats.quantizedInvertedCompactUsPerDoc =
		denseStats.quantizedInvertedCompactUsPerDoc;
	lastStats.quantizedInvertedCompactPayloadBytesPerDoc =
		denseStats.quantizedInvertedCompactPayloadBytesPerDoc;
	lastStats.quantizedInvertedCompactTopKChangedVsScalar =
		denseStats.quantizedInvertedCompactTopKChangedVsScalar;
	lastStats.quantizedInvertedPrecompactEnabled =
		denseStats.quantizedInvertedPrecompactEnabled;
	strlcpy(lastStats.quantizedInvertedPrecompactMode,
			denseStats.quantizedInvertedPrecompactMode,
			sizeof(lastStats.quantizedInvertedPrecompactMode));
	lastStats.quantizedInvertedDocsTouchedBeforePrecompact =
		denseStats.quantizedInvertedDocsTouchedBeforePrecompact;
	lastStats.quantizedInvertedPrecompactScoreK =
		denseStats.quantizedInvertedPrecompactScoreK;
	lastStats.quantizedInvertedPrecompactCoverageK =
		denseStats.quantizedInvertedPrecompactCoverageK;
	lastStats.quantizedInvertedPrecompactPerTokenK =
		denseStats.quantizedInvertedPrecompactPerTokenK;
	lastStats.quantizedInvertedCompactMaxDocs =
		denseStats.quantizedInvertedCompactMaxDocs;
	lastStats.quantizedInvertedPrecompactScoreDocs =
		denseStats.quantizedInvertedPrecompactScoreDocs;
	lastStats.quantizedInvertedPrecompactCoverageDocs =
		denseStats.quantizedInvertedPrecompactCoverageDocs;
	lastStats.quantizedInvertedPrecompactPerTokenDocs =
		denseStats.quantizedInvertedPrecompactPerTokenDocs;
	lastStats.quantizedInvertedPrecompactUnionDocs =
		denseStats.quantizedInvertedPrecompactUnionDocs;
	lastStats.quantizedInvertedPrecompactDuplicates =
		denseStats.quantizedInvertedPrecompactDuplicates;
	lastStats.quantizedInvertedPrecompactPrunedDocs =
		denseStats.quantizedInvertedPrecompactPrunedDocs;
	lastStats.quantizedInvertedPrecompactUs =
		denseStats.quantizedInvertedPrecompactUs;
	lastStats.quantizedInvertedCompactDocsSkippedByPrecompact =
		denseStats.quantizedInvertedCompactDocsSkippedByPrecompact;
	strlcpy(lastStats.quantizedInvertedTokenCoverageMode,
			denseStats.quantizedInvertedTokenCoverageMode,
			sizeof(lastStats.quantizedInvertedTokenCoverageMode));
	lastStats.quantizedInvertedActiveQueryTokens =
		denseStats.quantizedInvertedActiveQueryTokens;
	lastStats.quantizedInvertedTokenMatchesTotal =
		denseStats.quantizedInvertedTokenMatchesTotal;
	lastStats.quantizedInvertedTokenMatchesMax =
		denseStats.quantizedInvertedTokenMatchesMax;
	lastStats.quantizedInvertedMinTokenMatches =
		denseStats.quantizedInvertedMinTokenMatches;
	lastStats.quantizedInvertedTokenMatchFilteredDocs =
		denseStats.quantizedInvertedTokenMatchFilteredDocs;
	lastStats.quantizedInvertedScoreBoundPruningEnabled =
		denseStats.quantizedInvertedScoreBoundPruningEnabled;
	lastStats.quantizedInvertedScoreBoundDocsChecked =
		denseStats.quantizedInvertedScoreBoundDocsChecked;
	lastStats.quantizedInvertedScoreBoundDocsPruned =
		denseStats.quantizedInvertedScoreBoundDocsPruned;
	lastStats.quantizedInvertedScoreBoundPruneUs =
		denseStats.quantizedInvertedScoreBoundPruneUs;
	lastStats.quantizedInvertedScoreBoundUnsafeFallbacks =
		denseStats.quantizedInvertedScoreBoundUnsafeFallbacks;
	lastStats.quantizedInvertedCandidatesBeforeBound =
		denseStats.quantizedInvertedCandidatesBeforeBound;
	lastStats.quantizedInvertedCandidatesAfterBound =
		denseStats.quantizedInvertedCandidatesAfterBound;
	strlcpy(lastStats.multivectorDocSidecarCacheMode,
			denseStats.multivectorDocSidecarCacheMode,
			sizeof(lastStats.multivectorDocSidecarCacheMode));
	lastStats.multivectorDocSidecarPagesRead =
		denseStats.multivectorDocSidecarPagesRead;
	lastStats.multivectorDocSidecarCacheHits =
		denseStats.multivectorDocSidecarCacheHits;
	lastStats.multivectorDocSidecarCacheMisses =
		denseStats.multivectorDocSidecarCacheMisses;
	lastStats.multivectorDocSidecarBytesTouched =
		denseStats.multivectorDocSidecarBytesTouched;
	lastStats.multivectorDocSidecarVectorsLoaded =
		denseStats.multivectorDocSidecarVectorsLoaded;
	lastStats.multivectorDocSidecarDocMapPagesRead =
		denseStats.multivectorDocSidecarDocMapPagesRead;
	lastStats.multivectorDocSidecarDocMapBytesTouched =
		denseStats.multivectorDocSidecarDocMapBytesTouched;
	lastStats.multivectorDocSidecarResidentVectorsLoaded =
		denseStats.multivectorDocSidecarResidentVectorsLoaded;
	lastStats.multivectorDocSidecarResidentBytesLoaded =
		denseStats.multivectorDocSidecarResidentBytesLoaded;
	lastStats.multivectorDocSidecarVectorChunkRefBytesTouched =
		denseStats.multivectorDocSidecarVectorChunkRefBytesTouched;
	lastStats.multivectorDocSidecarPagedVectorPagesRead =
		denseStats.multivectorDocSidecarPagedVectorPagesRead;
	lastStats.multivectorDocSidecarPagedVectorBytesTouched =
		denseStats.multivectorDocSidecarPagedVectorBytesTouched;
	lastStats.multivectorSidecarPageReadUs =
		denseStats.multivectorSidecarPageReadUs;
	lastStats.multivectorSidecarVectorReconstructUs =
		denseStats.multivectorSidecarVectorReconstructUs;
	lastStats.multivectorTokensOriginal =
		denseStats.multivectorTokensOriginal;
	lastStats.multivectorTokensPooled =
		denseStats.multivectorTokensPooled;
	lastStats.multivectorReservoirsEnabled =
		denseStats.multivectorReservoirsEnabled;
	lastStats.multivectorReservoirScoreDocs =
		denseStats.multivectorReservoirScoreDocs;
	lastStats.multivectorReservoirCoverageDocs =
		denseStats.multivectorReservoirCoverageDocs;
	lastStats.multivectorReservoirMeanDocs =
		denseStats.multivectorReservoirMeanDocs;
	lastStats.multivectorReservoirPerTokenDocs =
		denseStats.multivectorReservoirPerTokenDocs;
	lastStats.multivectorReservoirBm25Docs =
		denseStats.multivectorReservoirBm25Docs;
	lastStats.multivectorReservoirUnionDocs =
		denseStats.multivectorReservoirUnionDocs;
	lastStats.multivectorReservoirDuplicates =
		denseStats.multivectorReservoirDuplicates;
	lastStats.multivectorBm25InjectionEnabled =
		denseStats.multivectorBm25InjectionEnabled;
	lastStats.multivectorBm25InjectionCandidates =
		denseStats.multivectorBm25InjectionCandidates;
	lastStats.multivectorBm25InjectionCandidateLimit =
		denseStats.multivectorBm25InjectionCandidateLimit;
	lastStats.multivectorBm25InjectionPoolSize =
		denseStats.multivectorBm25InjectionPoolSize;
	strlcpy(lastStats.multivectorBm25InjectionLimitReason,
			denseStats.multivectorBm25InjectionLimitReason,
			sizeof(lastStats.multivectorBm25InjectionLimitReason));
	lastStats.multivectorBm25InjectionRetained =
		denseStats.multivectorBm25InjectionRetained;
	lastStats.multivectorBm25InjectionExactReranked =
		denseStats.multivectorBm25InjectionExactReranked;
	lastStats.learnedSparseCandidates = denseStats.learnedSparseCandidates;
	lastStats.learnedSparseRetainedForMaxsim =
		denseStats.learnedSparseRetainedForMaxsim;
	lastStats.learnedSparseBranchLatencyUs =
		denseStats.learnedSparseBranchLatencyUs;
	switch ((PgturbohybridMultiVectorDocMapSource) denseStats.multivectorDocMapSource)
	{
		case PGTURBOHYBRID_MULTIVECTOR_DOCMAP_SOURCE_SIDECAR:
			strlcpy(lastStats.multivectorDocMapSource, "sidecar",
					sizeof(lastStats.multivectorDocMapSource));
			break;
		case PGTURBOHYBRID_MULTIVECTOR_DOCMAP_SOURCE_HEAP_TID_HASH:
			strlcpy(lastStats.multivectorDocMapSource, "heap_tid_hash",
					sizeof(lastStats.multivectorDocMapSource));
			break;
		case PGTURBOHYBRID_MULTIVECTOR_DOCMAP_SOURCE_NONE:
		default:
			strlcpy(lastStats.multivectorDocMapSource, "none",
					sizeof(lastStats.multivectorDocMapSource));
			break;
	}
	lastStats.multivectorDocMapBytes =
		denseStats.multivectorDocMapBytes;
	lastStats.multivectorUniqueDocs = denseStats.multivectorUniqueDocs;
	lastStats.multivectorDuplicateDocHits =
		denseStats.multivectorDuplicateDocHits;
	lastStats.multivectorMaxsimUpdates =
		denseStats.multivectorMaxsimUpdates;
	lastStats.multivectorDocCandidates = denseStats.multivectorDocCandidates;
	lastStats.multivectorExactRerankEnabled =
		denseStats.multivectorExactRerankEnabled;
	lastStats.multivectorExactRerankDocs =
		denseStats.multivectorExactRerankDocs;
	lastStats.multivectorExactRerankPairs =
		denseStats.multivectorExactRerankPairs;
	strlcpy(lastStats.multivectorExactRerankSource,
			PgturbohybridMultiVectorRerankSourceName(
				denseStats.multivectorExactRerankSource),
			sizeof(lastStats.multivectorExactRerankSource));
	lastStats.multivectorExactRerankHeapFetches =
		denseStats.multivectorExactRerankHeapFetches;
	lastStats.multivectorExactRerankSidecarReads =
		denseStats.multivectorExactRerankSidecarReads;
	lastStats.multivectorExactRerankSidecarBytes =
		denseStats.multivectorExactRerankSidecarBytes;
	lastStats.multivectorCandidateSourceUs =
		denseStats.multivectorCandidateSourceUs;
	lastStats.multivectorDocGraphTraversalUs =
		denseStats.multivectorDocGraphTraversalUs;
	lastStats.multivectorProxyCandidateUs =
		denseStats.multivectorProxyCandidateUs;
	lastStats.multivectorProxyGraphTraversalUs =
		denseStats.multivectorProxyGraphTraversalUs;
	lastStats.multivectorProxyScoringUs =
		denseStats.multivectorProxyScoringUs;
	lastStats.multivectorCentroidLitePostingUs =
		denseStats.multivectorCentroidLitePostingUs;
	lastStats.multivectorQuantizedInvertedPostingUs =
		denseStats.multivectorQuantizedInvertedPostingUs;
	lastStats.multivectorSidecarLoadUs =
		denseStats.multivectorSidecarLoadUs;
	lastStats.multivectorHeapVisibilityUs =
		denseStats.multivectorHeapVisibilityUs;
	lastStats.multivectorExactHeapFetchUs =
		denseStats.multivectorExactHeapFetchUs;
	lastStats.multivectorExactRerankUs =
		denseStats.multivectorExactRerankUs;
	lastStats.multivectorFinalSortUs =
		denseStats.multivectorFinalSortUs;
	lastStats.exactRerankCandidates = denseStats.exactRerankCandidates;
	lastStats.exactRerankTokensEvaluated =
		denseStats.exactRerankTokensEvaluated;
	lastStats.exactRerankTokensSkipped =
		denseStats.exactRerankTokensSkipped;
	lastStats.exactRerankPairsSaved = denseStats.exactRerankPairsSaved;
	lastStats.adaptiveRerankTopKChangedVsFull =
		denseStats.adaptiveRerankTopKChangedVsFull;
	strlcpy(lastStats.multivectorExactKernel,
			denseStats.multivectorExactKernel,
			sizeof(lastStats.multivectorExactKernel));
	strlcpy(lastStats.multivectorAccumulatorKind,
			denseStats.multivectorAccumulatorKind,
			sizeof(lastStats.multivectorAccumulatorKind));
	lastStats.multivectorMemoryBytesEstimate =
		denseStats.multivectorMemoryBytesEstimate;
	lastStats.multivectorAdmissionDebugEnabled =
		denseStats.multivectorAdmissionDebugEnabled;
	lastStats.multivectorAdmissionCandidatesBeforeRerank =
		denseStats.multivectorAdmissionCandidatesBeforeRerank;
	lastStats.multivectorAdmissionCandidatesAfterTruncation =
		denseStats.multivectorAdmissionCandidatesAfterTruncation;
	lastStats.multivectorAdmissionExactRerankDocs =
		denseStats.multivectorAdmissionExactRerankDocs;
	lastStats.multivectorAdmissionTruncatedByDocCandidateK =
		denseStats.multivectorAdmissionTruncatedByDocCandidateK;
	lastStats.multivectorAdmissionTruncatedByAccumulatorMemory =
		denseStats.multivectorAdmissionTruncatedByAccumulatorMemory;
	lastStats.multivectorAdmissionTraceAvailable =
		denseStats.multivectorAdmissionTraceAvailable;
	lastStats.multivectorAdmissionTraceCount =
		denseStats.multivectorAdmissionTraceCount;
	if (lastStats.multivectorAdmissionTraceCount >
		PGTURBOHYBRID_MULTIVECTOR_DEBUG_TRACE_LIMIT_MAX)
		lastStats.multivectorAdmissionTraceCount =
			PGTURBOHYBRID_MULTIVECTOR_DEBUG_TRACE_LIMIT_MAX;
	if (lastStats.multivectorAdmissionTraceCount > 0)
		memcpy(lastStats.multivectorAdmissionTrace,
			   denseStats.multivectorAdmissionTrace,
			   sizeof(PgturbohybridMultiVectorAdmissionTraceEntry) *
			   lastStats.multivectorAdmissionTraceCount);
	lastStats.multivectorTokenStatsAvailable =
		denseStats.multivectorTokenStatsAvailable;
	lastStats.multivectorTokenStatsCount =
		denseStats.multivectorTokenStatsCount;
	if (lastStats.multivectorTokenStatsCount >
		PGTURBOHYBRID_MULTIVECTOR_TOKEN_STATS_LIMIT_MAX)
		lastStats.multivectorTokenStatsCount =
			PGTURBOHYBRID_MULTIVECTOR_TOKEN_STATS_LIMIT_MAX;
	if (lastStats.multivectorTokenStatsCount > 0)
		memcpy(lastStats.multivectorTokenStats,
			   denseStats.multivectorTokenStats,
			   sizeof(PgturbohybridMultiVectorTokenStatsEntry) *
			   lastStats.multivectorTokenStatsCount);
	lastStats.bm25Terms = bm25Stats.queryTerms;
	lastStats.bm25PostingsDecoded = bm25Stats.postingsDecoded;
	lastStats.bm25BlocksVisited = bm25Stats.blocksVisited;
	lastStats.bm25BlocksSkipped = bm25Stats.blocksSkipped;
	lastStats.bm25FusedScoreBoundBlocksPruned =
		bm25Stats.fusedScoreBoundBlocksPruned;
	lastStats.bm25FusedScoreBoundCandidatesPruned =
		bm25Stats.fusedScoreBoundCandidatesPruned;
	lastStats.bm25CandidatesScored = bm25Stats.candidatesScored;
	lastStats.bm25HeapTSVectorRerankMode = bm25Stats.heapTsvectorRerankMode;
	lastStats.bm25HeapTSVectorRerankCount =
		bm25Stats.heapTsvectorRerankCount;
	lastStats.bm25HeapTSVectorRerankFetchUs =
		bm25Stats.heapTsvectorRerankFetchUs;
	lastStats.bm25HeapTSVectorRerankScoreUs =
		bm25Stats.heapTsvectorRerankScoreUs;
	lastStats.bm25HeapTSVectorRerankTopKChanged =
		bm25Stats.heapTsvectorRerankTopKChanged;
	lastStats.bm25CacheBytes = bm25Stats.cacheBytes;
	lastStats.bm25CacheLexiconEntries = bm25Stats.cacheLexiconEntries;
	lastStats.bm25CacheHit = bm25Stats.cacheHit;
	lastStats.bm25CacheBuildUs = bm25Stats.cacheBuildUs;
	lastStats.bm25CacheDocstatsLoaded = bm25Stats.cacheDocstatsLoaded;
	lastStats.bm25CacheLivenessLoaded = bm25Stats.cacheLivenessLoaded;
	lastStats.bm25DocstatsLoadedThisQuery =
		bm25Stats.docstatsLoadedThisQuery;
	lastStats.bm25LivenessLoadedThisQuery =
		bm25Stats.livenessLoadedThisQuery;
	lastStats.bm25DocstatsBytes = bm25Stats.docstatsBytes;
	lastStats.bm25LivenessBytes = bm25Stats.livenessBytes;
	lastStats.bm25ColdCacheONWork = bm25Stats.coldCacheONWork;
	lastStats.bm25PostingsDecodeRatio = bm25Stats.postingsDecodeRatio;
	lastStats.bm25CommonTermFallback = bm25Stats.commonTermFallback;
	lastStats.bm25WandPruned = bm25Stats.wandPruned;
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
	lastStats.fastWeightedEnabled =
		effectiveFusion == PGTURBOHYBRID_FUSION_FAST_WEIGHTED;
	lastStats.fastWeightedAlpha =
		(scanQuery->flags & PGTURBOHYBRID_QUERY_FLAG_ALPHA_IS_SET) != 0 ?
		scanQuery->alpha : 0.5;
	lastStats.calibratedFusionEnabled =
		effectiveFusion == PGTURBOHYBRID_FUSION_CALIBRATED;
	lastStats.dbsfEnabled =
		effectiveFusion == PGTURBOHYBRID_FUSION_DBSF;
	strlcpy(lastStats.bm25NormMode,
			lastStats.dbsfEnabled ? "dbsf" :
			((lastStats.fastWeightedEnabled || lastStats.calibratedFusionEnabled) ?
			 "saturating" : "none"),
			sizeof(lastStats.bm25NormMode));
	strlcpy(lastStats.denseNormMode,
			lastStats.dbsfEnabled ? "dbsf" :
			((lastStats.fastWeightedEnabled || lastStats.calibratedFusionEnabled) ?
			 "logistic" : "none"),
			sizeof(lastStats.denseNormMode));
	strlcpy(lastStats.hybridBudgetPolicy,
			PgturbohybridHybridBudgetPolicyName(pgturbohybrid_hybrid_budget_policy),
			sizeof(lastStats.hybridBudgetPolicy));
	strlcpy(lastStats.hybridQueryShape,
			PgturbohybridHybridQueryShapeName(hybridQueryShapeForStats),
			sizeof(lastStats.hybridQueryShape));
	strlcpy(lastStats.calibratedFusionQueryShape,
			PgturbohybridHybridQueryShapeName(hybridQueryShapeForStats),
			sizeof(lastStats.calibratedFusionQueryShape));
	lastStats.hybridDenseKChosen = denseBranchUsed ? scanQuery->denseK : 0;
	lastStats.hybridBm25KChosen = scanQuery->bm25K;
	strlcpy(lastStats.hybridBudgetReason, hybridBudgetChoice.reason,
			sizeof(lastStats.hybridBudgetReason));
	PgturbohybridBuildBranchPlan(&lastStats.branchPlan, scanQuery,
								 effectiveFusion,
								 (hasMultivectorQuery && !multivectorMergedAsDense) ?
								 multivector : dense,
								 (hasMultivectorQuery && !multivectorMergedAsDense) ?
								 multivectorCount : denseCount,
								 hasMultivectorQuery,
								 bm25, bm25Count, &denseStats,
								 hasMultivectorQuery ? multivectorElapsedUs :
								 lastStats.denseElapsedUs,
								 lastStats.bm25ElapsedUs);
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
	result = pgturbohybridbuild(heap, index, indexInfo);

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
	pgturbohybridbuildempty(index);
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

	if (PgturbohybridGraphUseTqNativeGraph(info->index))
	{
		result = tqgraphbulkdelete(info, stats, callback, callback_state);
		PgturbohybridBm25InvalidateCache(info->index);
		PgturbohybridSparseCacheInvalidate(RelationGetRelid(info->index));
	}
	else
		result = pgturbohybrid_graph_bulkdelete(info, stats, callback, callback_state);

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

	if (PgturbohybridGraphUseTqNativeGraph(info->index))
	{
		result = tqgraphvacuumcleanup(info, stats);
		(void) PgturbohybridBm25MaybeCompact(info->index);
		PgturbohybridBm25InvalidateCache(info->index);
		PgturbohybridSparseCacheInvalidate(RelationGetRelid(info->index));
	}
	else
		result = pgturbohybrid_graph_vacuum_cleanup(info, stats);

	return result;
}

static IndexScanDesc
pgturbohybridambeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;

	PgturbohybridValidateIndex(index, NULL);
	scan = pgturbohybridbeginscan(index, nkeys, norderbys);

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

	pgturbohybridrescan(scan, keys, nkeys,
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

	result = pgturbohybridgettuple(scan, dir);

	return result;
}

static void
pgturbohybridamendscan(IndexScanDesc scan)
{
	pgturbohybridendscan(scan);
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
		PGTURBOHYBRID_RELOPT_PARSE("routing", RELOPT_TYPE_ENUM, routing),
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
		PGTURBOHYBRID_RELOPT_PARSE("sparse_exact_storage", RELOPT_TYPE_ENUM, sparseExactStorage),
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

	if (validate && opts != NULL &&
		opts->sparseExactStorage == PGTURBOHYBRID_SPARSE_EXACT_STORAGE_SIDECAR)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("sparse_exact_storage=sidecar is not supported yet"),
				 errhint("Exact sparse rerank fetches the heap sparse column; use sparse_exact_storage=off.")));

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
					   "Valid values are \"normalized_mean\", \"mean\", \"first_token\", \"max_abs_mean\", \"centroid_mean\", \"max_pool\", \"random_projection_fde\", \"learned_projection_placeholder\", and \"learned_projection_v1\".",
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
	add_enum_reloption(pgturbohybrid_relopt_kind, "sparse_exact_storage",
					   "Exact f32 sparse storage for rerank (only \"off\" is supported).",
					   pgturbohybrid_sparse_exact_storage_relopt_options,
					   PGTURBOHYBRID_SPARSE_EXACT_STORAGE_OFF,
					   "Valid value is \"off\"; \"sidecar\" is reserved for a future release.",
					   AccessExclusiveLock);
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
						 pgturbohybrid_last_scan_state.finalKRequested,
						 pgturbohybrid_last_scan_state.finalKEffective,
						 pgturbohybrid_last_scan_state.detectedSqlLimit,
						 pgturbohybrid_last_scan_state.finalKInferred ? "true" : "false",
						 pgturbohybrid_last_scan_state.autoBudgetLimit,
						 pgturbohybrid_last_scan_state.unionCandidates,
					 pgturbohybrid_last_scan_state.finalResults,
					 pgturbohybrid_last_scan_state.fusionStrategy[0] != '\0' ?
					 pgturbohybrid_last_scan_state.fusionStrategy : "none",
					 pgturbohybrid_last_scan_state.fusionCandidatesSeen,
					 pgturbohybrid_last_scan_state.fusionHeapSize,
					 pgturbohybrid_last_scan_state.fusionDuplicates,
					 pgturbohybrid_last_scan_state.fusionHeapReplacements,
					 pgturbohybrid_last_scan_state.fusionGenerationArrayReused ? "true" : "false",
					 pgturbohybrid_last_scan_state.fusionGenerationArrayReset ? "true" : "false",
					 pgturbohybrid_last_scan_state.fastWeightedEnabled ? "true" : "false",
					 pgturbohybrid_last_scan_state.fastWeightedAlpha,
					 pgturbohybrid_last_scan_state.bm25NormMode[0] != '\0' ?
					 pgturbohybrid_last_scan_state.bm25NormMode : "none",
					 pgturbohybrid_last_scan_state.denseNormMode[0] != '\0' ?
					 pgturbohybrid_last_scan_state.denseNormMode : "none",
					 pgturbohybrid_last_scan_state.hybridBudgetPolicy[0] != '\0' ?
					 pgturbohybrid_last_scan_state.hybridBudgetPolicy : "fixed",
					 pgturbohybrid_last_scan_state.hybridQueryShape[0] != '\0' ?
					 pgturbohybrid_last_scan_state.hybridQueryShape : "fixed",
					 pgturbohybrid_last_scan_state.hybridDenseKChosen,
					 pgturbohybrid_last_scan_state.hybridBm25KChosen,
					 pgturbohybrid_last_scan_state.hybridBudgetReason[0] != '\0' ?
					 pgturbohybrid_last_scan_state.hybridBudgetReason : "fixed_policy",
					 pgturbohybrid_last_scan_state.bothMatch,
					 pgturbohybrid_last_scan_state.denseOnly,
					 pgturbohybrid_last_scan_state.bm25Only,
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
					 pgturbohybrid_last_scan_state.bm25Terms,
					 pgturbohybrid_last_scan_state.bm25PostingsDecoded,
					 pgturbohybrid_last_scan_state.bm25BlocksVisited,
					 pgturbohybrid_last_scan_state.bm25BlocksSkipped,
					 pgturbohybrid_last_scan_state.bm25FusedScoreBoundBlocksPruned,
					 pgturbohybrid_last_scan_state.bm25FusedScoreBoundCandidatesPruned,
					 pgturbohybrid_last_scan_state.bm25CandidatesScored,
					 pgturbohybrid_last_scan_state.bm25CacheBytes,
					 pgturbohybrid_last_scan_state.bm25CacheLexiconEntries,
					 pgturbohybrid_last_scan_state.bm25CacheHit ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25CacheBuildUs,
					 pgturbohybrid_last_scan_state.bm25CacheDocstatsLoaded ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25CacheLivenessLoaded ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25DocstatsLoadedThisQuery ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25LivenessLoadedThisQuery ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25DocstatsBytes,
					 pgturbohybrid_last_scan_state.bm25LivenessBytes,
					 pgturbohybrid_last_scan_state.bm25ColdCacheONWork ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25PostingsDecodeRatio,
					 pgturbohybrid_last_scan_state.bm25CommonTermFallback ? "true" : "false",
					 pgturbohybrid_last_scan_state.bm25WandPruned,
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
