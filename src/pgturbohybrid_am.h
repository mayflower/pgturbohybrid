#ifndef PGTURBOHYBRID_AM_H
#define PGTURBOHYBRID_AM_H

#include "postgres.h"

#include "access/genam.h"
#include "executor/execdesc.h"
#include "nodes/plannodes.h"
#include "nodes/pathnodes.h"
#include "utils/rel.h"

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

typedef struct PgturbohybridScanStatsSnapshot
{
	char		indexShape[16];
	bool		bm25BranchAvailable;
	bool		denseBranchUsed;
	bool		bm25BranchUsed;
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
	char		multivectorExactKernel[16];
	char		multivectorAccumulatorKind[16];
	uint64		multivectorMemoryBytesEstimate;
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
extern int	pgturbohybrid_multivector_max_doc_vectors;
extern int	pgturbohybrid_multivector_max_query_vectors;
extern int	pgturbohybrid_multivector_max_dim;
extern int	pgturbohybrid_multivector_subvector_k;
extern int	pgturbohybrid_multivector_unique_docs_per_token;
extern int	pgturbohybrid_multivector_max_raw_hits_per_token;
extern int	pgturbohybrid_multivector_adaptive_widening;
extern int	pgturbohybrid_multivector_docmap;
extern int	pgturbohybrid_multivector_doc_candidate_k;
extern int	pgturbohybrid_multivector_exact_rerank;
extern int	pgturbohybrid_multivector_exact_rerank_k;
extern int	pgturbohybrid_multivector_max_accumulator_mb;

typedef enum PgturbohybridMultiVectorExactRerankMode
{
	PGTURBOHYBRID_MULTIVECTOR_EXACT_RERANK_OFF,
	PGTURBOHYBRID_MULTIVECTOR_EXACT_RERANK_TOPK
}			PgturbohybridMultiVectorExactRerankMode;

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
