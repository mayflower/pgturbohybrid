#include "postgres.h"

#include "utils/guc.h"

#include "pgturbohybrid.h"
#include "pgturbohybrid_am.h"
#include "pgturbohybrid_bm25.h"
#include "pgturbohybrid_guc.h"
#include "pgturbohybrid_multivector.h"
#include "pgturbohybrid_quant.h"

#if PG_VERSION_NUM < 150000
#define MarkGUCPrefixReserved(x) EmitWarningsOnPlaceholders(x)
#endif

/* Public controls. Everything else is an internal fixed policy. */
int pgturbohybrid_default_dense_k = 100;
int pgturbohybrid_default_bm25_k = 100;
int pgturbohybrid_default_rrf_k = 60;
bool pgturbohybrid_enable_wand = true;
int pgturbohybrid_max_union_candidates = 100000;
bool pgturbohybrid_simd = true;
int pgturbohybrid_bm25_hot_postings_cache_mb = 0;
int pgturbohybrid_multivector_max_doc_vectors =
	PGTURBOHYBRID_MULTIVECTOR_DEFAULT_MAX_DOC_VECTORS;
int pgturbohybrid_multivector_max_query_vectors =
	PGTURBOHYBRID_MULTIVECTOR_DEFAULT_MAX_QUERY_VECTORS;
int pgturbohybrid_multivector_max_dim = PGTURBOHYBRID_MULTIVECTOR_MAX_DIM;
int pgturbohybrid_multivector_subvector_k = 100;
int pgturbohybrid_multivector_unique_docs_per_token = 100;
int pgturbohybrid_multivector_max_raw_hits_per_token = 400;
int pgturbohybrid_multivector_exact_rerank_k = 100;

/* Fixed implementation policy retained by the hot path. */
int pgturbohybrid_profile = PGTURBOHYBRID_PROFILE_LATENCY;
uint64 pgturbohybrid_guc_generation = 1;
int pgturbohybrid_force_fusion = 0;
int pgturbohybrid_fusion_hash_threshold = 128;
bool pgturbohybrid_fast_weighted_score_bound_pruning = true;
double pgturbohybrid_dbsf_sigma = 3.0;
int pgturbohybrid_dbsf_min_branch_candidates = 10;
int pgturbohybrid_dbsf_robust = 0;
bool pgturbohybrid_enable_exact_rescore_for_bm25_only = false;
int pgturbohybrid_bm25_cache_max_mb = 0;
int pgturbohybrid_bm25_hot_postings_cache_min_df = 1024;
int pgturbohybrid_bm25_common_term_fallback_min_postings = 100000;
bool pgturbohybrid_bm25_allow_lazy_impact_build = false;
int pgturbohybrid_bm25_simd_force = PGTURBOHYBRID_BM25_SIMD_FORCE_AUTO;
bool pgturbohybrid_bm25_force_full_sort = false;
int pgturbohybrid_bm25_accumulator_mode = PGTURBOHYBRID_BM25_ACCUMULATOR_AUTO;
int pgturbohybrid_bm25_dense_accumulator_threshold = 4096;
double pgturbohybrid_bm25_dense_accumulator_df_ratio = 0.05;
int pgturbohybrid_bm25_strategy = PGTURBOHYBRID_BM25_STRATEGY_AUTO;
int pgturbohybrid_bm25_impact_or_mode = PGTURBOHYBRID_BM25_IMPACT_OR_MODE_EXACT_ONLY;
int pgturbohybrid_bm25_heap_tsvector_rerank =
	PGTURBOHYBRID_BM25_HEAP_TSVECTOR_RERANK_OFF;
int pgturbohybrid_bm25_heap_tsvector_rerank_multiplier = 4;
double pgturbohybrid_bm25_heap_tsvector_rerank_weight = 0.10;
int pgturbohybrid_final_diversity = PGTURBOHYBRID_FINAL_DIVERSITY_OFF;
int pgturbohybrid_final_diversity_payload_slot = -1;
double pgturbohybrid_final_diversity_lambda =
	PGTURBOHYBRID_FINAL_DIVERSITY_DEFAULT_LAMBDA;
int pgturbohybrid_final_diversity_pool_multiplier =
	PGTURBOHYBRID_FINAL_DIVERSITY_DEFAULT_POOL_MULTIPLIER;
bool pgturbohybrid_auto_budget = true;
int pgturbohybrid_auto_budget_min_dense_k = 32;
int pgturbohybrid_auto_budget_min_bm25_k = 32;
int pgturbohybrid_auto_budget_limit_multiplier = 8;
int pgturbohybrid_auto_budget_quality_cap = 400;
bool pgturbohybrid_auto_bm25_budget = true;
int pgturbohybrid_auto_bm25_budget_min = 32;
int pgturbohybrid_auto_bm25_budget_max = 400;
bool pgturbohybrid_auto_bm25_budget_dense_confidence = true;
int pgturbohybrid_hybrid_budget_policy = PGTURBOHYBRID_HYBRID_BUDGET_FIXED;
int pgturbohybrid_bm25_hybrid_bound = PGTURBOHYBRID_BM25_HYBRID_BOUND_SAFE;
double pgturbohybrid_calibrated_fusion_both_match_bonus = 0.06;
double pgturbohybrid_calibrated_fusion_identifier_bm25_alpha = 0.35;
double pgturbohybrid_calibrated_fusion_broad_dense_alpha = 0.70;
double pgturbohybrid_calibrated_fusion_default_alpha = 0.50;
char *pgturbohybrid_multivector_model_name = "";
int pgturbohybrid_multivector_adaptive_widening =
	PGTURBOHYBRID_MULTIVECTOR_ADAPTIVE_WIDENING_AUTO;
int pgturbohybrid_multivector_docmap = PGTURBOHYBRID_MULTIVECTOR_DOCMAP_OFF;
int pgturbohybrid_multivector_doc_candidate_k = 100;
int pgturbohybrid_multivector_doc_graph_search_ef = 0;
int pgturbohybrid_multivector_doc_graph_oversampling = 1;
int pgturbohybrid_multivector_doc_graph_rescore_k = 0;
int pgturbohybrid_multivector_doc_graph_entry_sample_count = 0;
int pgturbohybrid_multivector_doc_storage = PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32;
int pgturbohybrid_multivector_doc_storage_cache =
	PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_AUTO;
int pgturbohybrid_multivector_exact_rerank =
	PGTURBOHYBRID_MULTIVECTOR_EXACT_RERANK_ADAPTIVE;
int pgturbohybrid_multivector_proxy_encoder =
	PGTURBOHYBRID_DEFAULT_MULTIVECTOR_PROXY_ENCODER;
char *pgturbohybrid_multivector_learned_projection_path = "";
char *pgturbohybrid_multivector_learned_projection_model = "";
char *pgturbohybrid_multivector_learned_projection_checksum = "";
bool pgturbohybrid_multivector_allow_exact_symmetric_build = false;
int pgturbohybrid_multivector_exact_symmetric_build_max_docs = 1000;
int pgturbohybrid_multivector_exact_symmetric_build_max_tokens = 0;
int pgturbohybrid_multivector_max_accumulator_mb = 64;
int pgturbohybrid_multivector_debug_admission =
	PGTURBOHYBRID_MULTIVECTOR_DEBUG_ADMISSION_OFF;
int pgturbohybrid_multivector_debug_trace_limit = 1000;
char *pgturbohybrid_multivector_debug_skip_query_tokens = "";
int pgturbohybrid_multivector_candidate_source =
	PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_GRAPH;
int pgturbohybrid_multivector_plain_fallback =
	PGTURBOHYBRID_MULTIVECTOR_PLAIN_FALLBACK_AUTO;
int pgturbohybrid_multivector_plain_fallback_max_docs = 1000;
double pgturbohybrid_multivector_plain_fallback_candidate_fraction = 0.5;
int pgturbohybrid_multivector_candidate_reservoirs =
	PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_RESERVOIRS_CONSERVATIVE;
int pgturbohybrid_multivector_per_token_doc_reservoir_k = 1;
int pgturbohybrid_multivector_coverage_reservoir_k = 10;
int pgturbohybrid_multivector_bm25_candidate_injection =
	PGTURBOHYBRID_MULTIVECTOR_BM25_CANDIDATE_INJECTION_OFF;
int pgturbohybrid_multivector_branch_plan = PGTURBOHYBRID_BRANCH_PLAN_AUTO;
int pgturbohybrid_multivector_centroid_lite_max_postings_per_token = 0;
int pgturbohybrid_multivector_centroid_lite_probe_centroids_per_token = 1;
int pgturbohybrid_multivector_centroid_lite_codeword_top_m = 1;
int pgturbohybrid_multivector_centroid_lite_posting_selection =
	PGTURBOHYBRID_MULTIVECTOR_POSTING_SELECTION_UNIFORM_STRIDE;
int pgturbohybrid_multivector_centroid_lite_candidate_scoring =
	PGTURBOHYBRID_MULTIVECTOR_CENTROID_LITE_CANDIDATE_SCORING_POSTING_PAYLOAD;
int pgturbohybrid_multivector_centroid_lite_bitset_prefilter =
	PGTURBOHYBRID_MULTIVECTOR_CENTROID_LITE_BITSET_PREFILTER_OFF;
int pgturbohybrid_multivector_centroid_lite_bitset_min_token_matches = 1;
int pgturbohybrid_multivector_centroid_lite_pruning =
	PGTURBOHYBRID_MULTIVECTOR_CENTROID_LITE_PRUNING_OFF;
double pgturbohybrid_multivector_centroid_lite_score_threshold = -1.0;
double pgturbohybrid_multivector_centroid_lite_score_drop_from_best = -1.0;
int pgturbohybrid_multivector_quantized_inverted_codebook =
	PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_DETERMINISTIC;
char *pgturbohybrid_multivector_quantized_inverted_codebook_path = "";
int pgturbohybrid_multivector_quantized_inverted_codebook_top_m = 1;
int pgturbohybrid_multivector_quantized_inverted_compact_scoring =
	PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_COMPACT_SCORING_OFF;
int pgturbohybrid_multivector_quantized_inverted_compact_doc_order =
	PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_COMPACT_DOC_ORDER_DOCID;
int pgturbohybrid_multivector_quantized_inverted_query_codeword_kernel =
	PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_QUERY_CODEWORD_KERNEL_AUTO;
int pgturbohybrid_multivector_quantized_inverted_precompact =
	PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_PRECOMPACT_OFF;
int pgturbohybrid_multivector_quantized_inverted_precompact_score_k = 4096;
int pgturbohybrid_multivector_quantized_inverted_precompact_coverage_k = 512;
int pgturbohybrid_multivector_quantized_inverted_precompact_per_token_k = 16;
int pgturbohybrid_multivector_quantized_inverted_compact_max_docs = 6144;
int pgturbohybrid_multivector_quantized_inverted_token_coverage =
	PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_TOKEN_COVERAGE_OFF;
int pgturbohybrid_multivector_quantized_inverted_min_token_matches = 0;
int pgturbohybrid_multivector_quantized_inverted_pruning =
	PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_PRUNING_OFF;
int pgturbohybrid_multivector_quantized_inverted_max_postings_per_token = 0;
int pgturbohybrid_multivector_quantized_inverted_probe_codewords_per_token = 1;
int pgturbohybrid_multivector_quantized_inverted_posting_selection =
	PGTURBOHYBRID_MULTIVECTOR_POSTING_SELECTION_SCORE_TOPK;

const char *
PgturbohybridHybridBudgetPolicyName(int policy)
{
	return policy == PGTURBOHYBRID_HYBRID_BUDGET_ADAPTIVE ? "adaptive" : "fixed";
}

const char *
PgturbohybridProfileName(int profile)
{
	(void) profile;
	return "default";
}

const char *
PgturbohybridBm25SimdForceName(int force)
{
	switch ((PgturbohybridBm25SimdForce) force)
	{
		case PGTURBOHYBRID_BM25_SIMD_FORCE_SCALAR: return "scalar";
		case PGTURBOHYBRID_BM25_SIMD_FORCE_AVX2: return "avx2";
		case PGTURBOHYBRID_BM25_SIMD_FORCE_NEON: return "neon";
		default: return "auto";
	}
}

const char *
PgturbohybridBm25AccumulatorModeName(int mode)
{
	switch ((PgturbohybridBm25AccumulatorMode) mode)
	{
		case PGTURBOHYBRID_BM25_ACCUMULATOR_HASH: return "hash";
		case PGTURBOHYBRID_BM25_ACCUMULATOR_DENSE: return "node_generation_arrays";
		default: return "auto";
	}
}

const char *
PgturbohybridBm25StrategyName(int strategy)
{
	switch ((PgturbohybridBm25Strategy) strategy)
	{
		case PGTURBOHYBRID_BM25_STRATEGY_IMPACT: return "impact";
		case PGTURBOHYBRID_BM25_STRATEGY_IMPACT_OR: return "impact_or";
		case PGTURBOHYBRID_BM25_STRATEGY_WAND: return "wand";
		case PGTURBOHYBRID_BM25_STRATEGY_DAAT_SIMD: return "daat_simd";
		case PGTURBOHYBRID_BM25_STRATEGY_DAAT_HASH: return "daat_hash";
		default: return "auto";
	}
}

const char *
PgturbohybridBm25RuntimeStrategyName(int strategy)
{
	switch ((PgturbohybridBm25RuntimeStrategy) strategy)
	{
		case PGTURBOHYBRID_BM25_RUNTIME_IMPACT_SINGLE: return "impact_single";
		case PGTURBOHYBRID_BM25_RUNTIME_IMPACT_OR: return "impact_or";
		case PGTURBOHYBRID_BM25_RUNTIME_IMPACT_SEEDED_WAND: return "impact_seeded_wand";
		case PGTURBOHYBRID_BM25_RUNTIME_WAND: return "wand";
		case PGTURBOHYBRID_BM25_RUNTIME_AND_RAREST_DRIVER: return "and_rarest_driver";
		case PGTURBOHYBRID_BM25_RUNTIME_DAAT_SIMD: return "daat_simd";
		case PGTURBOHYBRID_BM25_RUNTIME_DAAT_HASH: return "daat_hash";
		default: return "none";
	}
}

const char *
PgturbohybridBm25ImpactOrModeName(int mode)
{
	switch ((PgturbohybridBm25ImpactOrMode) mode)
	{
		case PGTURBOHYBRID_BM25_IMPACT_OR_MODE_EXACT_ONLY: return "exact_only";
		case PGTURBOHYBRID_BM25_IMPACT_OR_MODE_APPROX: return "approx";
		default: return "off";
	}
}

const char *
PgturbohybridBm25HybridBoundModeName(int mode)
{
	switch ((PgturbohybridBm25HybridBoundMode) mode)
	{
		case PGTURBOHYBRID_BM25_HYBRID_BOUND_SAFE: return "safe";
		case PGTURBOHYBRID_BM25_HYBRID_BOUND_APPROX: return "approx";
		default: return "off";
	}
}

const char *
PgturbohybridBm25HeapTSVectorRerankModeName(int mode)
{
	switch ((PgturbohybridBm25HeapTSVectorRerankMode) mode)
	{
		case PGTURBOHYBRID_BM25_HEAP_TSVECTOR_RERANK_TOPK: return "topk";
		case PGTURBOHYBRID_BM25_HEAP_TSVECTOR_RERANK_BAND: return "band";
		case PGTURBOHYBRID_BM25_HEAP_TSVECTOR_RERANK_AUTO: return "auto";
		default: return "off";
	}
}

const char *
PgturbohybridFinalDiversityName(int mode)
{
	return mode == PGTURBOHYBRID_FINAL_DIVERSITY_GROUP_PAYLOAD ?
		"group_payload" : "off";
}

static void
PgturbohybridAssignQuerySetting(int newval, void *extra)
{
	(void) newval;
	(void) extra;
	if (++pgturbohybrid_guc_generation == 0)
		pgturbohybrid_guc_generation = 1;
}

void
PgturbohybridRegisterGUCs(void)
{
	DefineCustomIntVariable("turbohybrid.default_dense_k",
		"Default dense candidate budget.", NULL,
		&pgturbohybrid_default_dense_k, 100, 1, 1000000,
		PGC_USERSET, 0, NULL, PgturbohybridAssignQuerySetting, NULL);
	DefineCustomIntVariable("turbohybrid.default_bm25_k",
		"Default BM25 candidate budget.", NULL,
		&pgturbohybrid_default_bm25_k, 100, 1, 1000000,
		PGC_USERSET, 0, NULL, PgturbohybridAssignQuerySetting, NULL);
	DefineCustomIntVariable("turbohybrid.default_rrf_k",
		"Default reciprocal-rank fusion constant.", NULL,
		&pgturbohybrid_default_rrf_k, 60, 1, 1000000,
		PGC_USERSET, 0, NULL, PgturbohybridAssignQuerySetting, NULL);
	DefineCustomBoolVariable("turbohybrid.enable_wand",
		"Use WAND pruning for BM25 candidates.", NULL,
		&pgturbohybrid_enable_wand, true,
		PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.max_union_candidates",
		"Maximum candidates retained during fusion.", NULL,
		&pgturbohybrid_max_union_candidates, 100000, 1, 10000000,
		PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("turbohybrid.simd",
		"Use available SIMD scoring kernels.", NULL,
		&pgturbohybrid_simd, true,
		PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.bm25_hot_postings_cache_mb",
		"Backend-local cache for repeated BM25 postings.", NULL,
		&pgturbohybrid_bm25_hot_postings_cache_mb, 0, 0, 65536,
		PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_subvector_k",
		"Graph candidates retained per query token.", NULL,
		&pgturbohybrid_multivector_subvector_k, 100, 1, 1000000,
		PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_unique_docs_per_token",
		"Unique documents retained per query token.", NULL,
		&pgturbohybrid_multivector_unique_docs_per_token, 100, 1, 1000000,
		PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_exact_rerank_k",
		"Documents reranked with exact float32 MaxSim.", NULL,
		&pgturbohybrid_multivector_exact_rerank_k, 100, 1, 1000000,
		PGC_USERSET, 0, NULL, NULL, NULL);
	MarkGUCPrefixReserved("turbohybrid");
}
