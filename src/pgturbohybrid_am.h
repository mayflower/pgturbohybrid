#ifndef PGTURBOHYBRID_AM_H
#define PGTURBOHYBRID_AM_H

#include "postgres.h"

#include "access/genam.h"
#include "nodes/plannodes.h"
#include "nodes/pathnodes.h"
#include "utils/rel.h"

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
	int			tqBits;
	bool		tqWeighted;
	bool		tqQuantileFit;
	bool		tqRenorm;
	bool		tqExactStorage;
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

extern bool pgturbohybrid_enable_wand;
extern int	pgturbohybrid_max_union_candidates;
extern int	pgturbohybrid_default_dense_k;
extern int	pgturbohybrid_default_bm25_k;
extern int	pgturbohybrid_default_rrf_k;
extern int	pgturbohybrid_force_fusion;
extern int	pgturbohybrid_fusion_hash_threshold;
extern bool pgturbohybrid_enable_exact_rescore_for_bm25_only;
extern int	pgturbohybrid_bm25_cache_max_mb;
extern int	pgturbohybrid_bm25_hot_postings_cache_mb;
extern int	pgturbohybrid_bm25_hot_postings_cache_min_df;
extern bool pgturbohybrid_bm25_allow_lazy_impact_build;
extern int	pgturbohybrid_bm25_simd_force;
extern bool pgturbohybrid_bm25_force_full_sort;
extern int	pgturbohybrid_bm25_accumulator_mode;
extern int	pgturbohybrid_bm25_dense_accumulator_threshold;
extern double pgturbohybrid_bm25_dense_accumulator_df_ratio;
extern int	pgturbohybrid_bm25_strategy;
extern int	pgturbohybrid_bm25_impact_or_mode;
extern bool pgturbohybrid_auto_budget;
extern int	pgturbohybrid_auto_budget_min_dense_k;
extern int	pgturbohybrid_auto_budget_min_bm25_k;
extern int	pgturbohybrid_auto_budget_limit_multiplier;
extern int	pgturbohybrid_auto_budget_quality_cap;
extern bool pgturbohybrid_auto_bm25_budget;
extern int	pgturbohybrid_auto_bm25_budget_min;
extern int	pgturbohybrid_auto_bm25_budget_max;
extern bool pgturbohybrid_auto_bm25_budget_dense_confidence;

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
const char *PgturbohybridBm25RuntimeStrategyName(int strategy);

void		PgturbohybridInit(void);
PlannedStmt *PgturbohybridCurrentPlannedStmt(void);
Datum		pgturbohybrid_handler(PG_FUNCTION_ARGS);

#endif
