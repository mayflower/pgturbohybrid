#ifndef PGTURBOHYBRID_AM_H
#define PGTURBOHYBRID_AM_H

#include "postgres.h"

#include "access/genam.h"
#include "executor/execdesc.h"
#include "nodes/plannodes.h"
#include "nodes/pathnodes.h"
#include "utils/rel.h"

#include "pgturbohybrid_multivector.h"

#define PGTURBOHYBRID_DEFAULT_FINAL_K 10
#define PGTURBOHYBRID_MAX_DEFAULT_DENSE_K 10000
#define PGTURBOHYBRID_MAX_DEFAULT_BM25_K 10000
#define PGTURBOHYBRID_MAX_RRF_K 100000
#define PGTURBOHYBRID_MAX_UNION_CANDIDATES 1000000
#define PGTURBOHYBRID_MAX_HOT_POSTINGS_CACHE_MB 1024
#define PGTURBOHYBRID_MAX_MULTIVECTOR_DOC_VECTORS 4096
#define PGTURBOHYBRID_MAX_MULTIVECTOR_QUERY_VECTORS 4096
#define PGTURBOHYBRID_MAX_MULTIVECTOR_ACCUMULATOR_MB 1024
#define PGTURBOHYBRID_DENSE_KEY_INDEX 0
#define PGTURBOHYBRID_LEXICAL_KEY_INDEX 1

typedef enum PgturbohybridHybridBudgetPolicy
{
	PGTURBOHYBRID_HYBRID_BUDGET_FIXED,
	PGTURBOHYBRID_HYBRID_BUDGET_ADAPTIVE
}			PgturbohybridHybridBudgetPolicy;

typedef enum PgturbohybridMultiVectorDocMapSource
{
	PGTURBOHYBRID_MULTIVECTOR_DOCMAP_SOURCE_NONE,
	PGTURBOHYBRID_MULTIVECTOR_DOCMAP_SOURCE_HEAP_TID_HASH,
	PGTURBOHYBRID_MULTIVECTOR_DOCMAP_SOURCE_SIDECAR
}			PgturbohybridMultiVectorDocMapSource;

typedef struct PgturbohybridOptions
{
	int32		vl_len_;
	int			m;
	int			efConstruction;
	int			routing;
	int			graphEfSearch;
	int			graphOversampling;
	int			graphRescoreBand;
	int			graphExactCache;
	int			graphReorder;
	int			nativeSegments;
	int			tqBits;
	bool		tqWeighted;
	bool		tqQuantileFit;
	bool		tqRenorm;
	bool		tqExactStorage;
	bool		entrySidecar;
	int			entrySidecarRepresentatives;
	int			entrySidecarStrategy;
	bool		graphBackbone;
	bool		residualRerank;
	int			residualRerankBytes;
	int			multivectorGraphMode;
	int			multivectorDocBuildScorer;
	int			multivectorTokenPooling;
	float8		multivectorTokenPoolingTargetRatio;
	int			multivectorTokenPoolingMinTokens;
	int			multivectorCentroids;
	int			multivectorCentroidCount;
	int			multivectorProxyEncoder;
	int			multivectorContextMode;
	int			multivectorFieldMode;
	float8		bm25K1;
	float8		bm25B;
	bool		bm25BlockMax;
	bool		bm25PrecomputeTfNorm;
	bool		bm25ImpactHead;
	int			bm25ImpactMinDf;
	int			bm25ImpactHeadK;
	int			bm25DeltaCompactionThreshold;
	int			hybridDefaultFusion;
	int			hybridDefaultDenseK;
	int			hybridDefaultBm25K;
	int			hybridDefaultRrfK;
}			PgturbohybridOptions;

#define PGTURBOHYBRID_BRANCH_PLAN_MAX_BRANCHES 8

typedef enum PgturbohybridBranchPlanMode
{
	PGTURBOHYBRID_BRANCH_PLAN_AUTO,
	PGTURBOHYBRID_BRANCH_PLAN_DENSE_ONLY,
	PGTURBOHYBRID_BRANCH_PLAN_QDRANT_LIKE
}			PgturbohybridBranchPlanMode;

typedef enum PgturbohybridBranchKind
{
	PGTURBOHYBRID_BRANCH_KIND_BM25,
	PGTURBOHYBRID_BRANCH_KIND_DENSE_SINGLE,
	PGTURBOHYBRID_BRANCH_KIND_PROXY_VECTOR,
	PGTURBOHYBRID_BRANCH_KIND_DOCUMENT_NODES,
	PGTURBOHYBRID_BRANCH_KIND_TOKEN_NODES,
	PGTURBOHYBRID_BRANCH_KIND_EXACT_DOC_SCAN,
	PGTURBOHYBRID_BRANCH_KIND_CENTROID_LITE,
	PGTURBOHYBRID_BRANCH_KIND_QUANTIZED_INVERTED_EXPERIMENTAL
}			PgturbohybridBranchKind;

#define PGTURBOHYBRID_BRANCH_SOURCE_DENSE		(1U << 0)
#define PGTURBOHYBRID_BRANCH_SOURCE_BM25		(1U << 1)
#define PGTURBOHYBRID_BRANCH_SOURCE_MULTIVECTOR	(1U << 2)
#define PGTURBOHYBRID_BRANCH_SOURCE_EXACT		(1U << 3)

typedef struct PgturbohybridBranchPlanItem
{
	int			kind;
	uint32		candidateLimit;
	uint32		rescoreLimit;
	uint32		branchRank;
	double		branchScore;
	uint32		sourceFlags;
	uint32		candidateCount;
	bool		truncated;
	uint64		latencyUs;
}			PgturbohybridBranchPlanItem;

typedef struct PgturbohybridBranchPlan
{
	int			mode;
	uint32		count;
	char		fusionMode[32];
	PgturbohybridBranchPlanItem items[PGTURBOHYBRID_BRANCH_PLAN_MAX_BRANCHES];
}			PgturbohybridBranchPlan;

typedef struct PgturbohybridScanStatsSnapshot
{
	char		indexShape[16];
	bool		bm25BranchAvailable;
	bool		denseBranchUsed;
	bool		bm25BranchUsed;
	PgturbohybridBranchPlan branchPlan;
	uint32		denseCandidatesEffective;
	bool		denseKDefaulted;
	uint32		bm25CandidatesEffective;
	bool		bm25KDefaulted;
	bool		bm25CacheHit;
	uint64		bm25CacheBuildUs;
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
	uint32		bm25Terms;
	uint64		bm25FusedScoreBoundBlocksPruned;
	uint64		bm25FusedScoreBoundCandidatesPruned;
	char		bm25HeapTSVectorRerankMode[16];
	uint32		bm25HeapTSVectorRerankCount;
	uint64		bm25HeapTSVectorRerankFetchUs;
	uint64		bm25HeapTSVectorRerankScoreUs;
	bool		bm25HeapTSVectorRerankTopKChanged;
	bool		fastWeightedEnabled;
	double		fastWeightedAlpha;
	bool		calibratedFusionEnabled;
	char		calibratedFusionQueryShape[32];
	double		calibratedFusionAlphaEffective;
	double		calibratedFusionBothMatchBonus;
	char		calibratedFusionDenseNormMode[16];
	char		calibratedFusionBm25NormMode[16];
	char		bm25NormMode[16];
	char		denseNormMode[16];
	bool		dbsfEnabled;
	double		dbsfBranchMean[2];
	double		dbsfBranchStddev[2];
	double		dbsfBranchMin[2];
	double		dbsfBranchMax[2];
	uint32		dbsfDegenerateBranches;
	char		hybridBudgetPolicy[16];
	char		hybridQueryShape[32];
	uint32		hybridDenseKChosen;
	uint32		hybridBm25KChosen;
	char		hybridBudgetReason[96];
	char		fusionStrategy[24];
	uint32		fusionCandidatesSeen;
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
	char		multivectorDocGraphStorageKind[16];
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
	uint64		centroidPostingsSkipped;
	uint32		centroidPostingLimitPerToken;
	char		centroidPostingCapStrategy[32];
	uint32		centroidCandidates;
	bool		centroidBitsetPrefilterEnabled;
	uint32		centroidBitsetListsUsed;
	uint32		centroidBitsetDocsSet;
	uint32		centroidBitsetDocsAfterThreshold;
	uint64		centroidBitsetPrefilterUs;
	uint64		centroidBitsetMemoryBytes;
	uint32		multivectorCentroidCount;
	uint32		multivectorCentroidPrerankDocs;
	uint32		multivectorFullMaxsimRerankDocs;
	uint64		quantizedInvertedListsVisited;
	uint64		quantizedInvertedPostingsTouched;
	uint64		quantizedInvertedDocsScored;
	uint32		quantizedInvertedCandidates;
	uint32		quantizedInvertedExactRerankDocs;
	char		quantizedInvertedCodebookSource[16];
	uint32		quantizedInvertedCodebookSize;
	uint32		quantizedInvertedCodebookDim;
	char		quantizedInvertedCodebookChecksum[128];
	uint32		quantizedInvertedCodebookTopM;
	uint64		quantizedInvertedAssignmentUs;
	uint64		quantizedInvertedListOffsetBytes;
	uint64		quantizedInvertedPostingBytes;
	uint64		quantizedInvertedSidecarBytes;
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
	/* Token-local unique document hits summed across query tokens. */
	uint64		multivectorUniqueDocs;
	/* Raw hits whose document was already seen for the same query token. */
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
	char		finalDiversityMode[24];
	int32		finalDiversityPayloadSlot;
	uint32		finalDiversityPoolSize;
	uint32		finalDiversitySelected;
	uint64		finalDiversityDuplicateGroupsSuppressed;
	uint64		finalDiversityUs;
	uint64		denseElapsedUs;
	uint64		bm25ElapsedUs;
	uint64		fusionElapsedUs;
	uint64		elapsedUs;
}			PgturbohybridScanStatsSnapshot;

extern bool pgturbohybrid_enable_wand;
extern int	pgturbohybrid_max_union_candidates;
extern int	pgturbohybrid_default_dense_k;
extern int	pgturbohybrid_default_bm25_k;
extern int	pgturbohybrid_default_rrf_k;
extern uint64 pgturbohybrid_guc_generation;
extern int	pgturbohybrid_last_final_k_requested;
extern int	pgturbohybrid_last_final_k_effective;
extern int	pgturbohybrid_last_sql_limit;
extern bool pgturbohybrid_last_final_k_inferred;
extern int	pgturbohybrid_force_fusion;
extern int	pgturbohybrid_fusion_hash_threshold;
extern bool pgturbohybrid_enable_exact_rescore_for_bm25_only;
extern int	pgturbohybrid_bm25_cache_max_mb;
extern int	pgturbohybrid_bm25_hot_postings_cache_mb;
extern int	pgturbohybrid_bm25_hot_postings_cache_min_df;
extern int	pgturbohybrid_bm25_common_term_fallback_min_postings;
extern bool pgturbohybrid_bm25_allow_lazy_impact_build;
extern int	pgturbohybrid_bm25_simd_force;
extern bool pgturbohybrid_bm25_force_full_sort;
extern int	pgturbohybrid_bm25_accumulator_mode;
extern int	pgturbohybrid_bm25_dense_accumulator_threshold;
extern double pgturbohybrid_bm25_dense_accumulator_df_ratio;
extern int	pgturbohybrid_bm25_strategy;
extern int	pgturbohybrid_bm25_impact_or_mode;
extern int	pgturbohybrid_bm25_hybrid_bound;
extern int	pgturbohybrid_bm25_heap_tsvector_rerank;
extern int	pgturbohybrid_bm25_heap_tsvector_rerank_multiplier;
extern double pgturbohybrid_bm25_heap_tsvector_rerank_weight;
extern bool pgturbohybrid_auto_budget;
extern int	pgturbohybrid_auto_budget_min_dense_k;
extern int	pgturbohybrid_auto_budget_min_bm25_k;
extern int	pgturbohybrid_auto_budget_limit_multiplier;
extern int	pgturbohybrid_auto_budget_quality_cap;
extern bool pgturbohybrid_auto_bm25_budget;
extern int	pgturbohybrid_auto_bm25_budget_min;
extern int	pgturbohybrid_auto_bm25_budget_max;
extern bool pgturbohybrid_auto_bm25_budget_dense_confidence;
extern int	pgturbohybrid_hybrid_budget_policy;
extern int	pgturbohybrid_profile;
extern double pgturbohybrid_calibrated_fusion_both_match_bonus;
extern double pgturbohybrid_calibrated_fusion_identifier_bm25_alpha;
extern double pgturbohybrid_calibrated_fusion_broad_dense_alpha;
extern double pgturbohybrid_calibrated_fusion_default_alpha;
extern double pgturbohybrid_dbsf_sigma;
extern int	pgturbohybrid_dbsf_min_branch_candidates;
extern int	pgturbohybrid_dbsf_robust;
extern int	pgturbohybrid_multivector_max_doc_vectors;
extern int	pgturbohybrid_multivector_max_query_vectors;
extern int	pgturbohybrid_multivector_max_dim;
extern char *pgturbohybrid_multivector_model_name;
extern int	pgturbohybrid_multivector_subvector_k;
extern int	pgturbohybrid_multivector_unique_docs_per_token;
extern int	pgturbohybrid_multivector_max_raw_hits_per_token;
extern int	pgturbohybrid_multivector_adaptive_widening;
extern int	pgturbohybrid_multivector_docmap;
extern int	pgturbohybrid_multivector_doc_candidate_k;
extern int	pgturbohybrid_multivector_doc_graph_search_ef;
extern int	pgturbohybrid_multivector_doc_graph_oversampling;
extern int	pgturbohybrid_multivector_doc_graph_rescore_k;
extern int	pgturbohybrid_multivector_doc_graph_entry_sample_count;
extern int	pgturbohybrid_multivector_doc_storage;
extern int	pgturbohybrid_multivector_doc_storage_cache;
extern int	pgturbohybrid_multivector_exact_rerank;
extern int	pgturbohybrid_multivector_exact_rerank_k;
extern int	pgturbohybrid_multivector_proxy_encoder;
extern char *pgturbohybrid_multivector_learned_projection_path;
extern char *pgturbohybrid_multivector_learned_projection_model;
extern char *pgturbohybrid_multivector_learned_projection_checksum;
extern bool pgturbohybrid_multivector_allow_exact_symmetric_build;
extern int	pgturbohybrid_multivector_exact_symmetric_build_max_docs;
extern int	pgturbohybrid_multivector_max_accumulator_mb;
extern int	pgturbohybrid_multivector_debug_admission;
extern int	pgturbohybrid_multivector_debug_trace_limit;
extern char *pgturbohybrid_multivector_debug_skip_query_tokens;
extern int	pgturbohybrid_multivector_candidate_source;
extern int	pgturbohybrid_multivector_plain_fallback;
extern int	pgturbohybrid_multivector_plain_fallback_max_docs;
extern double pgturbohybrid_multivector_plain_fallback_candidate_fraction;
extern int	pgturbohybrid_multivector_candidate_reservoirs;
extern int	pgturbohybrid_multivector_per_token_doc_reservoir_k;
extern int	pgturbohybrid_multivector_coverage_reservoir_k;
extern int	pgturbohybrid_multivector_bm25_candidate_injection;
extern int	pgturbohybrid_multivector_sparse_candidate_source;
extern int	pgturbohybrid_multivector_branch_plan;
extern int	pgturbohybrid_multivector_centroid_lite_max_postings_per_token;
extern int	pgturbohybrid_multivector_centroid_lite_bitset_prefilter;
extern int	pgturbohybrid_multivector_quantized_inverted_codebook;
extern char *pgturbohybrid_multivector_quantized_inverted_codebook_path;
extern int	pgturbohybrid_multivector_quantized_inverted_codebook_top_m;

typedef enum PgturbohybridMultiVectorExactRerankMode
{
	PGTURBOHYBRID_MULTIVECTOR_EXACT_RERANK_OFF,
	PGTURBOHYBRID_MULTIVECTOR_EXACT_RERANK_TOPK,
	PGTURBOHYBRID_MULTIVECTOR_EXACT_RERANK_ADAPTIVE
}			PgturbohybridMultiVectorExactRerankMode;

typedef enum PgturbohybridMultiVectorCentroidLiteBitsetPrefilterMode
{
	PGTURBOHYBRID_MULTIVECTOR_CENTROID_LITE_BITSET_PREFILTER_OFF,
	PGTURBOHYBRID_MULTIVECTOR_CENTROID_LITE_BITSET_PREFILTER_EXPERIMENTAL
}			PgturbohybridMultiVectorCentroidLiteBitsetPrefilterMode;

typedef enum PgturbohybridMultiVectorAdaptiveWideningMode
{
	PGTURBOHYBRID_MULTIVECTOR_ADAPTIVE_WIDENING_OFF,
	PGTURBOHYBRID_MULTIVECTOR_ADAPTIVE_WIDENING_AUTO,
	PGTURBOHYBRID_MULTIVECTOR_ADAPTIVE_WIDENING_ON
}			PgturbohybridMultiVectorAdaptiveWideningMode;

typedef enum PgturbohybridMultiVectorDocMapMode
{
	PGTURBOHYBRID_MULTIVECTOR_DOCMAP_OFF,
	PGTURBOHYBRID_MULTIVECTOR_DOCMAP_AUTO,
	PGTURBOHYBRID_MULTIVECTOR_DOCMAP_REQUIRE
}			PgturbohybridMultiVectorDocMapMode;

typedef enum PgturbohybridMultiVectorDocStorageMode
{
	PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32,
	PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F16,
	PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_SQ8
}			PgturbohybridMultiVectorDocStorageMode;

typedef enum PgturbohybridMultiVectorDocStorageCacheMode
{
	PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_RESIDENT,
	PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_PAGED,
	PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_AUTO
}			PgturbohybridMultiVectorDocStorageCacheMode;

typedef enum PgturbohybridMultiVectorDebugAdmissionMode
{
	PGTURBOHYBRID_MULTIVECTOR_DEBUG_ADMISSION_OFF,
	PGTURBOHYBRID_MULTIVECTOR_DEBUG_ADMISSION_SUMMARY,
	PGTURBOHYBRID_MULTIVECTOR_DEBUG_ADMISSION_TRACE
}			PgturbohybridMultiVectorDebugAdmissionMode;

typedef enum PgturbohybridMultiVectorCandidateSource
{
	PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_GRAPH,
	PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_EXACT_TOKEN_SCAN,
	PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_EXACT_DOC_SCAN,
	PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_DOC_GRAPH_PROTOTYPE,
	PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_DOCUMENT_NODES,
	PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_PROXY_VECTOR,
	PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_CENTROID_LITE,
	PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_QUANTIZED_INVERTED_EXPERIMENTAL
}			PgturbohybridMultiVectorCandidateSource;

typedef enum PgturbohybridMultiVectorQuantizedInvertedCodebookSource
{
	PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_DETERMINISTIC,
	PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_EXTERNAL
}			PgturbohybridMultiVectorQuantizedInvertedCodebookSource;

typedef enum PgturbohybridMultiVectorPlainFallbackMode
{
	PGTURBOHYBRID_MULTIVECTOR_PLAIN_FALLBACK_AUTO,
	PGTURBOHYBRID_MULTIVECTOR_PLAIN_FALLBACK_OFF,
	PGTURBOHYBRID_MULTIVECTOR_PLAIN_FALLBACK_FORCE
}			PgturbohybridMultiVectorPlainFallbackMode;

typedef enum PgturbohybridMultiVectorCandidateReservoirMode
{
	PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_RESERVOIRS_OFF,
	PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_RESERVOIRS_CONSERVATIVE,
	PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_RESERVOIRS_BALANCED
}			PgturbohybridMultiVectorCandidateReservoirMode;

typedef enum PgturbohybridMultiVectorBm25CandidateInjectionMode
{
	PGTURBOHYBRID_MULTIVECTOR_BM25_CANDIDATE_INJECTION_OFF,
	PGTURBOHYBRID_MULTIVECTOR_BM25_CANDIDATE_INJECTION_HYBRID_ONLY,
	PGTURBOHYBRID_MULTIVECTOR_BM25_CANDIDATE_INJECTION_DENSE_WITH_TEXT
}			PgturbohybridMultiVectorBm25CandidateInjectionMode;

typedef enum PgturbohybridMultiVectorSparseCandidateSource
{
	PGTURBOHYBRID_MULTIVECTOR_SPARSE_CANDIDATE_SOURCE_OFF,
	PGTURBOHYBRID_MULTIVECTOR_SPARSE_CANDIDATE_SOURCE_BM25,
	PGTURBOHYBRID_MULTIVECTOR_SPARSE_CANDIDATE_SOURCE_LEARNED_SPARSE
}			PgturbohybridMultiVectorSparseCandidateSource;

typedef enum PgturbohybridProfile
{
	PGTURBOHYBRID_PROFILE_LATENCY,
	PGTURBOHYBRID_PROFILE_BALANCED,
	PGTURBOHYBRID_PROFILE_QUALITY,
	PGTURBOHYBRID_PROFILE_MATCHED_RECALL,
	PGTURBOHYBRID_PROFILE_DEBUG,
	PGTURBOHYBRID_PROFILE_HIGH_RECALL
}			PgturbohybridProfile;

typedef enum PgturbohybridBm25SimdForce
{
	PGTURBOHYBRID_BM25_SIMD_FORCE_AUTO,
	PGTURBOHYBRID_BM25_SIMD_FORCE_SCALAR,
	PGTURBOHYBRID_BM25_SIMD_FORCE_AVX2,
	PGTURBOHYBRID_BM25_SIMD_FORCE_NEON
}			PgturbohybridBm25SimdForce;

typedef enum PgturbohybridBm25AccumulatorMode
{
	PGTURBOHYBRID_BM25_ACCUMULATOR_HASH,
	PGTURBOHYBRID_BM25_ACCUMULATOR_DENSE,
	PGTURBOHYBRID_BM25_ACCUMULATOR_AUTO
}			PgturbohybridBm25AccumulatorMode;

typedef enum PgturbohybridBm25Strategy
{
	PGTURBOHYBRID_BM25_STRATEGY_AUTO,
	PGTURBOHYBRID_BM25_STRATEGY_IMPACT,
	PGTURBOHYBRID_BM25_STRATEGY_IMPACT_OR,
	PGTURBOHYBRID_BM25_STRATEGY_WAND,
	PGTURBOHYBRID_BM25_STRATEGY_DAAT_SIMD,
	PGTURBOHYBRID_BM25_STRATEGY_DAAT_HASH
}			PgturbohybridBm25Strategy;

typedef enum PgturbohybridBm25ImpactOrMode
{
	PGTURBOHYBRID_BM25_IMPACT_OR_MODE_OFF,
	PGTURBOHYBRID_BM25_IMPACT_OR_MODE_EXACT_ONLY,
	PGTURBOHYBRID_BM25_IMPACT_OR_MODE_APPROX
}			PgturbohybridBm25ImpactOrMode;

typedef enum PgturbohybridBm25RuntimeStrategy
{
	PGTURBOHYBRID_BM25_RUNTIME_NONE,
	PGTURBOHYBRID_BM25_RUNTIME_IMPACT_SINGLE,
	PGTURBOHYBRID_BM25_RUNTIME_IMPACT_OR,
	PGTURBOHYBRID_BM25_RUNTIME_IMPACT_SEEDED_WAND,
	PGTURBOHYBRID_BM25_RUNTIME_WAND,
	PGTURBOHYBRID_BM25_RUNTIME_AND_RAREST_DRIVER,
	PGTURBOHYBRID_BM25_RUNTIME_DAAT_SIMD,
	PGTURBOHYBRID_BM25_RUNTIME_DAAT_HASH
}			PgturbohybridBm25RuntimeStrategy;

const char *PgturbohybridBm25SimdForceName(int force);
const char *PgturbohybridBm25AccumulatorModeName(int mode);
const char *PgturbohybridBm25StrategyName(int strategy);
const char *PgturbohybridBm25ImpactOrModeName(int mode);
const char *PgturbohybridBm25HybridBoundModeName(int mode);
const char *PgturbohybridBm25RuntimeStrategyName(int strategy);
const char *PgturbohybridBm25HeapTSVectorRerankModeName(int mode);
const char *PgturbohybridHybridBudgetPolicyName(int policy);
const char *PgturbohybridProfileName(int profile);
void		PgturbohybridApplyProfileDefaults(void);

void		PgturbohybridInit(void);
void		PgturbohybridAmExecutorStart(QueryDesc *queryDesc, int eflags);
void		PgturbohybridAmExecutorEnd(QueryDesc *queryDesc);
void		PgturbohybridAmExecutorAbort(void);
PlannedStmt *PgturbohybridCurrentPlannedStmt(void);
int			PgturbohybridCurrentLimit(void);
void		PgturbohybridGetLastScanStatsSnapshot(PgturbohybridScanStatsSnapshot *stats);
bool		PgturbohybridIndexHasLexical(Relation index);
bool		PgturbohybridIndexGetLexicalDatum(Relation index, Datum *values,
										bool *isnull, Datum *lexicalValue);

#endif
