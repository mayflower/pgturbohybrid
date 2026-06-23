/*
 * pgturbohybrid_guc.c
 *		turbohybrid.* GUC variable definitions, enum option tables, profile
 *		defaults, GUC check/assign hooks, and custom GUC registration.
 *
 * This file holds the GUC/profile machinery previously embedded in
 * pgturbohybrid_am.c. Behavior is unchanged: registration order, names,
 * defaults, bounds, flags, and check/assign wiring all match the original.
 * PgturbohybridRegisterGUCs() is invoked from PgturbohybridInit() in
 * pgturbohybrid_am.c after reloption registration and the parallel-worker /
 * already-defined guards.
 */
#include "postgres.h"

#include <limits.h>

#include "miscadmin.h"
#include "utils/builtins.h"
#include "utils/guc.h"

#include "pgturbohybrid.h"
#include "pgturbohybrid_multivector.h"
#include "pgturbohybrid_quant.h"
#include "pgturbohybrid_quant_score.h"
#include "pgturbohybrid_am.h"
#include "pgturbohybrid_bm25.h"
#include "pgturbohybrid_sparse.h"
#include "pgturbohybrid_guc.h"

#define PGTURBOHYBRID_DEFAULT_DENSE_K 100
#define PGTURBOHYBRID_DEFAULT_BM25_K 100
#define PGTURBOHYBRID_DEFAULT_SPARSE_K 100
#define PGTURBOHYBRID_DEFAULT_RRF_K 60

#if PG_VERSION_NUM < 150000
#define MarkGUCPrefixReserved(x) EmitWarningsOnPlaceholders(x)
#endif

int			pgturbohybrid_profile = PGTURBOHYBRID_PROFILE_LATENCY;
bool		pgturbohybrid_enable_wand = true;
bool		pgturbohybrid_enable_sparse_wand = true;
int			pgturbohybrid_sparse_hot_postings_cache_mb = 16;
int			pgturbohybrid_sparse_hot_postings_cache_min_df = 256;
int			pgturbohybrid_sparse_delta_compaction_threshold = 1000;
int			pgturbohybrid_max_union_candidates = 100000;
int			pgturbohybrid_default_dense_k = PGTURBOHYBRID_DEFAULT_DENSE_K;
int			pgturbohybrid_default_bm25_k = PGTURBOHYBRID_DEFAULT_BM25_K;
int			pgturbohybrid_default_sparse_k = PGTURBOHYBRID_DEFAULT_SPARSE_K;
int			pgturbohybrid_default_rrf_k = PGTURBOHYBRID_DEFAULT_RRF_K;
uint64		pgturbohybrid_guc_generation = 1;
int			pgturbohybrid_last_final_k_requested = 0;
int			pgturbohybrid_last_final_k_effective = 0;
int			pgturbohybrid_last_sql_limit = 0;
bool		pgturbohybrid_last_final_k_inferred = false;
bool		pgturbohybrid_simd = true;
int			pgturbohybrid_force_fusion = 0;
int			pgturbohybrid_fusion_hash_threshold = 128;
bool		pgturbohybrid_fast_weighted_score_bound_pruning = true;
double		pgturbohybrid_dbsf_sigma = 3.0;
int			pgturbohybrid_dbsf_min_branch_candidates = 10;
int			pgturbohybrid_dbsf_robust = 0;
bool		pgturbohybrid_enable_exact_rescore_for_bm25_only = false;
int			pgturbohybrid_bm25_cache_max_mb = 0;
int			pgturbohybrid_bm25_hot_postings_cache_mb = 0;
int			pgturbohybrid_bm25_hot_postings_cache_min_df = 1024;
int			pgturbohybrid_bm25_common_term_fallback_min_postings = 100000;
bool		pgturbohybrid_bm25_allow_lazy_impact_build = false;
int			pgturbohybrid_bm25_simd_force = PGTURBOHYBRID_BM25_SIMD_FORCE_AUTO;
bool		pgturbohybrid_bm25_force_full_sort = false;
int			pgturbohybrid_bm25_accumulator_mode = PGTURBOHYBRID_BM25_ACCUMULATOR_AUTO;
int			pgturbohybrid_bm25_dense_accumulator_threshold = 4096;
double		pgturbohybrid_bm25_dense_accumulator_df_ratio = 0.05;
int			pgturbohybrid_bm25_strategy = PGTURBOHYBRID_BM25_STRATEGY_AUTO;
int			pgturbohybrid_bm25_impact_or_mode = PGTURBOHYBRID_BM25_IMPACT_OR_MODE_EXACT_ONLY;
int			pgturbohybrid_bm25_heap_tsvector_rerank =
	PGTURBOHYBRID_BM25_HEAP_TSVECTOR_RERANK_OFF;
int			pgturbohybrid_bm25_heap_tsvector_rerank_multiplier = 4;
double		pgturbohybrid_bm25_heap_tsvector_rerank_weight = 0.10;
int			pgturbohybrid_sparse_rerank = PGTURBOHYBRID_SPARSE_RERANK_AUTO;
int			pgturbohybrid_sparse_rerank_k = 0;	/* 0 = follow final_k */
int			pgturbohybrid_final_diversity =
	PGTURBOHYBRID_FINAL_DIVERSITY_OFF;
int			pgturbohybrid_final_diversity_payload_slot = -1;
double		pgturbohybrid_final_diversity_lambda =
	PGTURBOHYBRID_FINAL_DIVERSITY_DEFAULT_LAMBDA;
int			pgturbohybrid_final_diversity_pool_multiplier =
	PGTURBOHYBRID_FINAL_DIVERSITY_DEFAULT_POOL_MULTIPLIER;
bool		pgturbohybrid_auto_budget = true;
int			pgturbohybrid_auto_budget_min_dense_k = 32;
int			pgturbohybrid_auto_budget_min_bm25_k = 32;
int			pgturbohybrid_auto_budget_limit_multiplier = 8;
int			pgturbohybrid_auto_budget_quality_cap = 400;
bool		pgturbohybrid_auto_bm25_budget = true;
int			pgturbohybrid_auto_bm25_budget_min = 32;
int			pgturbohybrid_auto_bm25_budget_max = 400;
bool		pgturbohybrid_auto_bm25_budget_dense_confidence = true;
int			pgturbohybrid_hybrid_budget_policy = PGTURBOHYBRID_HYBRID_BUDGET_FIXED;
int			pgturbohybrid_bm25_hybrid_bound = PGTURBOHYBRID_BM25_HYBRID_BOUND_SAFE;
double		pgturbohybrid_calibrated_fusion_both_match_bonus = 0.06;
double		pgturbohybrid_calibrated_fusion_identifier_bm25_alpha = 0.35;
double		pgturbohybrid_calibrated_fusion_broad_dense_alpha = 0.70;
double		pgturbohybrid_calibrated_fusion_default_alpha = 0.50;
int			pgturbohybrid_multivector_max_doc_vectors =
	PGTURBOHYBRID_MULTIVECTOR_DEFAULT_MAX_DOC_VECTORS;
int			pgturbohybrid_multivector_max_query_vectors =
	PGTURBOHYBRID_MULTIVECTOR_DEFAULT_MAX_QUERY_VECTORS;
int			pgturbohybrid_multivector_max_dim = PGTURBOHYBRID_MULTIVECTOR_MAX_DIM;
char	   *pgturbohybrid_multivector_model_name = "";
int			pgturbohybrid_multivector_subvector_k = 100;
int			pgturbohybrid_multivector_unique_docs_per_token = 100;
int			pgturbohybrid_multivector_max_raw_hits_per_token = 400;
int			pgturbohybrid_multivector_adaptive_widening =
	PGTURBOHYBRID_MULTIVECTOR_ADAPTIVE_WIDENING_AUTO;
int			pgturbohybrid_multivector_docmap =
	PGTURBOHYBRID_MULTIVECTOR_DOCMAP_AUTO;
int			pgturbohybrid_multivector_doc_candidate_k = 100;
int			pgturbohybrid_multivector_doc_graph_search_ef = 0;
int			pgturbohybrid_multivector_doc_graph_oversampling = 1;
int			pgturbohybrid_multivector_doc_graph_rescore_k = 0;
int			pgturbohybrid_multivector_doc_graph_entry_sample_count = 0;
int			pgturbohybrid_multivector_doc_storage =
	PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32;
int			pgturbohybrid_multivector_doc_storage_cache =
	PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_AUTO;
int			pgturbohybrid_multivector_exact_rerank =
	PGTURBOHYBRID_MULTIVECTOR_EXACT_RERANK_ADAPTIVE;
int			pgturbohybrid_multivector_exact_rerank_k = 100;
int			pgturbohybrid_multivector_proxy_encoder =
	PGTURBOHYBRID_DEFAULT_MULTIVECTOR_PROXY_ENCODER;
char	   *pgturbohybrid_multivector_learned_projection_path = "";
char	   *pgturbohybrid_multivector_learned_projection_model = "";
char	   *pgturbohybrid_multivector_learned_projection_checksum = "";
bool		pgturbohybrid_multivector_allow_exact_symmetric_build = false;
int			pgturbohybrid_multivector_exact_symmetric_build_max_docs = 1000;
int			pgturbohybrid_multivector_max_accumulator_mb = 64;
int			pgturbohybrid_multivector_debug_admission =
	PGTURBOHYBRID_MULTIVECTOR_DEBUG_ADMISSION_OFF;
int			pgturbohybrid_multivector_debug_trace_limit = 1000;
char	   *pgturbohybrid_multivector_debug_skip_query_tokens = "";
int			pgturbohybrid_multivector_candidate_source =
	PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_GRAPH;
int			pgturbohybrid_multivector_quantized_inverted_codebook =
	PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_DETERMINISTIC;
char	   *pgturbohybrid_multivector_quantized_inverted_codebook_path = "";
int			pgturbohybrid_multivector_quantized_inverted_codebook_top_m = 1;
int			pgturbohybrid_multivector_quantized_inverted_compact_scoring =
	PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_COMPACT_SCORING_OFF;
int			pgturbohybrid_multivector_quantized_inverted_compact_doc_order =
	PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_COMPACT_DOC_ORDER_DOCID;
int			pgturbohybrid_multivector_quantized_inverted_query_codeword_kernel =
	PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_QUERY_CODEWORD_KERNEL_AUTO;
int			pgturbohybrid_multivector_quantized_inverted_precompact =
	PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_PRECOMPACT_OFF;
int			pgturbohybrid_multivector_quantized_inverted_precompact_score_k =
	4096;
int			pgturbohybrid_multivector_quantized_inverted_precompact_coverage_k =
	512;
int			pgturbohybrid_multivector_quantized_inverted_precompact_per_token_k =
	16;
int			pgturbohybrid_multivector_quantized_inverted_compact_max_docs =
	6144;
int			pgturbohybrid_multivector_quantized_inverted_token_coverage =
	PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_TOKEN_COVERAGE_OFF;
int			pgturbohybrid_multivector_quantized_inverted_min_token_matches = 0;
int			pgturbohybrid_multivector_quantized_inverted_pruning =
	PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_PRUNING_OFF;
int			pgturbohybrid_multivector_plain_fallback =
	PGTURBOHYBRID_MULTIVECTOR_PLAIN_FALLBACK_AUTO;
int			pgturbohybrid_multivector_plain_fallback_max_docs = 1000;
double		pgturbohybrid_multivector_plain_fallback_candidate_fraction = 0.5;
int			pgturbohybrid_multivector_candidate_reservoirs =
	PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_RESERVOIRS_CONSERVATIVE;
int			pgturbohybrid_multivector_per_token_doc_reservoir_k = 1;
int			pgturbohybrid_multivector_coverage_reservoir_k = 10;
int			pgturbohybrid_multivector_bm25_candidate_injection =
	PGTURBOHYBRID_MULTIVECTOR_BM25_CANDIDATE_INJECTION_OFF;
int			pgturbohybrid_multivector_sparse_candidate_source =
	PGTURBOHYBRID_MULTIVECTOR_SPARSE_CANDIDATE_SOURCE_OFF;
int			pgturbohybrid_multivector_branch_plan =
	PGTURBOHYBRID_BRANCH_PLAN_AUTO;
int			pgturbohybrid_multivector_centroid_lite_max_postings_per_token = 0;
int			pgturbohybrid_multivector_centroid_lite_probe_centroids_per_token = 1;
int			pgturbohybrid_multivector_centroid_lite_codeword_top_m = 1;
int			pgturbohybrid_multivector_centroid_lite_posting_selection =
	PGTURBOHYBRID_MULTIVECTOR_POSTING_SELECTION_UNIFORM_STRIDE;
int			pgturbohybrid_multivector_centroid_lite_candidate_scoring =
	PGTURBOHYBRID_MULTIVECTOR_CENTROID_LITE_CANDIDATE_SCORING_POSTING_PAYLOAD;
int			pgturbohybrid_multivector_centroid_lite_bitset_prefilter =
	PGTURBOHYBRID_MULTIVECTOR_CENTROID_LITE_BITSET_PREFILTER_OFF;
int			pgturbohybrid_multivector_centroid_lite_bitset_min_token_matches = 1;
int			pgturbohybrid_multivector_centroid_lite_pruning =
	PGTURBOHYBRID_MULTIVECTOR_CENTROID_LITE_PRUNING_OFF;
double		pgturbohybrid_multivector_centroid_lite_score_threshold = -1.0;
double		pgturbohybrid_multivector_centroid_lite_score_drop_from_best = -1.0;
int			pgturbohybrid_multivector_quantized_inverted_max_postings_per_token = 0;
int			pgturbohybrid_multivector_quantized_inverted_probe_codewords_per_token = 1;
int			pgturbohybrid_multivector_quantized_inverted_posting_selection =
	PGTURBOHYBRID_MULTIVECTOR_POSTING_SELECTION_SCORE_TOPK;
static bool pgturbohybrid_bm25_strategy_user_set = false;
static bool pgturbohybrid_bm25_impact_or_mode_user_set = false;
static bool pgturbohybrid_bm25_hot_postings_cache_mb_user_set = false;
static bool pgturbohybrid_bm25_hot_postings_cache_min_df_user_set = false;
static bool pgturbohybrid_bm25_hybrid_bound_user_set = false;
static bool pgturbohybrid_bm25_accumulator_mode_user_set = false;


static const struct config_enum_entry pgturbohybrid_profile_options[] = {
	{"latency", PGTURBOHYBRID_PROFILE_LATENCY, false},
	{"balanced", PGTURBOHYBRID_PROFILE_BALANCED, false},
	{"quality", PGTURBOHYBRID_PROFILE_QUALITY, false},
	{"matched_recall", PGTURBOHYBRID_PROFILE_MATCHED_RECALL, false},
	{"high_recall", PGTURBOHYBRID_PROFILE_HIGH_RECALL, false},
	{"debug", PGTURBOHYBRID_PROFILE_DEBUG, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_bm25_strategy_options[] = {
	{"auto", PGTURBOHYBRID_BM25_STRATEGY_AUTO, false},
	{"impact", PGTURBOHYBRID_BM25_STRATEGY_IMPACT, false},
	{"impact_or", PGTURBOHYBRID_BM25_STRATEGY_IMPACT_OR, false},
	{"wand", PGTURBOHYBRID_BM25_STRATEGY_WAND, false},
	{"daat_simd", PGTURBOHYBRID_BM25_STRATEGY_DAAT_SIMD, false},
	{"daat_hash", PGTURBOHYBRID_BM25_STRATEGY_DAAT_HASH, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_bm25_impact_or_mode_options[] = {
	{"off", PGTURBOHYBRID_BM25_IMPACT_OR_MODE_OFF, false},
	{"exact_only", PGTURBOHYBRID_BM25_IMPACT_OR_MODE_EXACT_ONLY, false},
	{"approx", PGTURBOHYBRID_BM25_IMPACT_OR_MODE_APPROX, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_bm25_hybrid_bound_options[] = {
	{"off", PGTURBOHYBRID_BM25_HYBRID_BOUND_OFF, false},
	{"safe", PGTURBOHYBRID_BM25_HYBRID_BOUND_SAFE, false},
	{"approx", PGTURBOHYBRID_BM25_HYBRID_BOUND_APPROX, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_bm25_accumulator_mode_options[] = {
	{"auto", PGTURBOHYBRID_BM25_ACCUMULATOR_AUTO, false},
	{"hash", PGTURBOHYBRID_BM25_ACCUMULATOR_HASH, false},
	{"node_generation_arrays", PGTURBOHYBRID_BM25_ACCUMULATOR_DENSE, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_dbsf_robust_options[] = {
	{"off", PGTURBOHYBRID_DBSF_ROBUST_OFF, false},
	{"mad", PGTURBOHYBRID_DBSF_ROBUST_MAD, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_bm25_heap_tsvector_rerank_options[] = {
	{"off", PGTURBOHYBRID_BM25_HEAP_TSVECTOR_RERANK_OFF, false},
	{"topk", PGTURBOHYBRID_BM25_HEAP_TSVECTOR_RERANK_TOPK, false},
	{"band", PGTURBOHYBRID_BM25_HEAP_TSVECTOR_RERANK_BAND, false},
	{"auto", PGTURBOHYBRID_BM25_HEAP_TSVECTOR_RERANK_AUTO, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_sparse_rerank_options[] = {
	{"off", PGTURBOHYBRID_SPARSE_RERANK_OFF, false},
	{"topk", PGTURBOHYBRID_SPARSE_RERANK_TOPK, false},
	{"band", PGTURBOHYBRID_SPARSE_RERANK_BAND, false},
	{"auto", PGTURBOHYBRID_SPARSE_RERANK_AUTO, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_hybrid_budget_policy_options[] = {
	{"fixed", PGTURBOHYBRID_HYBRID_BUDGET_FIXED, false},
	{"adaptive", PGTURBOHYBRID_HYBRID_BUDGET_ADAPTIVE, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_dense_adaptive_widening_options[] = {
	{"off", PGTURBOHYBRID_DENSE_ADAPTIVE_WIDENING_OFF, false},
	{"auto", PGTURBOHYBRID_DENSE_ADAPTIVE_WIDENING_AUTO, false},
	{"on", PGTURBOHYBRID_DENSE_ADAPTIVE_WIDENING_ON, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_multivector_adaptive_widening_options[] = {
	{"off", PGTURBOHYBRID_MULTIVECTOR_ADAPTIVE_WIDENING_OFF, false},
	{"auto", PGTURBOHYBRID_MULTIVECTOR_ADAPTIVE_WIDENING_AUTO, false},
	{"on", PGTURBOHYBRID_MULTIVECTOR_ADAPTIVE_WIDENING_ON, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_multivector_docmap_options[] = {
	{"off", PGTURBOHYBRID_MULTIVECTOR_DOCMAP_OFF, false},
	{"auto", PGTURBOHYBRID_MULTIVECTOR_DOCMAP_AUTO, false},
	{"require", PGTURBOHYBRID_MULTIVECTOR_DOCMAP_REQUIRE, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_multivector_doc_storage_options[] = {
	{"f32", PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32, false},
	{"f16", PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F16, false},
	{"sq8", PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_SQ8, false},
	{"centroid_only", PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CENTROID_ONLY, false},
	{"proxy_only", PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_PROXY_ONLY, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_multivector_doc_storage_cache_options[] = {
	{"resident", PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_RESIDENT, false},
	{"paged", PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_PAGED, false},
	{"auto", PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_AUTO, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_dense_uncertainty_retry_options[] = {
	{"off", PGTURBOHYBRID_DENSE_UNCERTAINTY_RETRY_OFF, false},
	{"auto", PGTURBOHYBRID_DENSE_UNCERTAINTY_RETRY_AUTO, false},
	{"on", PGTURBOHYBRID_DENSE_UNCERTAINTY_RETRY_ON, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_dense_query_split_impl_options[] = {
	{"auto", PGTURBOHYBRID_QUERY_SPLIT_IMPL_AUTO, false},
	{"signed", PGTURBOHYBRID_QUERY_SPLIT_IMPL_SIGNED, false},
	{"unsigned", PGTURBOHYBRID_QUERY_SPLIT_IMPL_UNSIGNED, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_dense_u8_split_options[] = {
	{"auto", PGTURBOHYBRID_U8_SPLIT_AUTO, false},
	{"on", PGTURBOHYBRID_U8_SPLIT_ON, false},
	{"off", PGTURBOHYBRID_U8_SPLIT_OFF, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_dense_build_neighbor_select_options[] = {
	{"fast", PGTURBOHYBRID_DENSE_BUILD_NEIGHBOR_SELECT_FAST, false},
	{"heuristic", PGTURBOHYBRID_DENSE_BUILD_NEIGHBOR_SELECT_HEURISTIC, false},
	{"auto", PGTURBOHYBRID_DENSE_BUILD_NEIGHBOR_SELECT_AUTO, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_dense_build_distance_options[] = {
	{"auto", PGTURBOHYBRID_DENSE_BUILD_DISTANCE_AUTO, false},
	{"code", PGTURBOHYBRID_DENSE_BUILD_DISTANCE_CODE, false},
	{"exact", PGTURBOHYBRID_DENSE_BUILD_DISTANCE_EXACT, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_native_cache_policy_options[] = {
	{"auto", PGTURBOHYBRID_NATIVE_CACHE_POLICY_AUTO, false},
	{"per_backend", PGTURBOHYBRID_NATIVE_CACHE_POLICY_PER_BACKEND, false},
	{"off", PGTURBOHYBRID_NATIVE_CACHE_POLICY_OFF, false},
	{"shared", PGTURBOHYBRID_NATIVE_CACHE_POLICY_SHARED, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_native_segment_budget_options[] = {
	{"auto", PGTURBOHYBRID_NATIVE_SEGMENT_BUDGET_AUTO, false},
	{"off", PGTURBOHYBRID_NATIVE_SEGMENT_BUDGET_OFF, false},
	{"sqrt", PGTURBOHYBRID_NATIVE_SEGMENT_BUDGET_SQRT, false},
	{"linear", PGTURBOHYBRID_NATIVE_SEGMENT_BUDGET_LINEAR, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_native_parallel_edge_build_options[] = {
	{"auto", PGTURBOHYBRID_NATIVE_PARALLEL_EDGE_BUILD_AUTO, false},
	{"off", PGTURBOHYBRID_NATIVE_PARALLEL_EDGE_BUILD_OFF, false},
	{"on", PGTURBOHYBRID_NATIVE_PARALLEL_EDGE_BUILD_ON, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_dense_rescore_band_options[] = {
	{"auto", PGTURBOHYBRID_RESCORE_BAND_POLICY_AUTO, false},
	{"off", PGTURBOHYBRID_RESCORE_BAND_POLICY_OFF, false},
	{"exact", PGTURBOHYBRID_RESCORE_BAND_POLICY_EXACT, false},
	{"limited", PGTURBOHYBRID_RESCORE_BAND_POLICY_LIMITED, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_dense_heap_rescore_options[] = {
	{"off", PGTURBOHYBRID_DENSE_HEAP_RESCORE_OFF, false},
	{"topk", PGTURBOHYBRID_DENSE_HEAP_RESCORE_TOPK, false},
	{"band", PGTURBOHYBRID_DENSE_HEAP_RESCORE_BAND, false},
	{"auto", PGTURBOHYBRID_DENSE_HEAP_RESCORE_AUTO, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_multivector_exact_rerank_options[] = {
	{"off", PGTURBOHYBRID_MULTIVECTOR_EXACT_RERANK_OFF, false},
	{"topk", PGTURBOHYBRID_MULTIVECTOR_EXACT_RERANK_TOPK, false},
	{"adaptive", PGTURBOHYBRID_MULTIVECTOR_EXACT_RERANK_ADAPTIVE, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_multivector_proxy_encoder_options[] = {
	{"mean", PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_MEAN, false},
	{"normalized_mean", PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_NORMALIZED_MEAN, false},
	{"first_token", PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_FIRST_TOKEN, false},
	{"max_abs_mean", PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_MAX_ABS_MEAN, false},
	{"centroid_mean", PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_CENTROID_MEAN, false},
	{"mean_pool", PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_MEAN, false},
	{"max_pool", PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_MAX_POOL, false},
	{"random_projection_fde", PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_RANDOM_PROJECTION_FDE, false},
	{"learned_projection_placeholder", PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_LEARNED_PROJECTION_PLACEHOLDER, false},
	{"learned_projection_v1", PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_LEARNED_PROJECTION_V1, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_multivector_debug_admission_options[] = {
	{"off", PGTURBOHYBRID_MULTIVECTOR_DEBUG_ADMISSION_OFF, false},
	{"summary", PGTURBOHYBRID_MULTIVECTOR_DEBUG_ADMISSION_SUMMARY, false},
	{"trace", PGTURBOHYBRID_MULTIVECTOR_DEBUG_ADMISSION_TRACE, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_multivector_candidate_source_options[] = {
	{"graph", PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_GRAPH, false},
	{"exact_token_scan", PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_EXACT_TOKEN_SCAN, false},
	{"exact_doc_scan", PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_EXACT_DOC_SCAN, false},
	{"doc_graph_prototype", PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_DOC_GRAPH_PROTOTYPE, false},
	{"document_nodes", PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_DOCUMENT_NODES, false},
	{"proxy_vector", PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_PROXY_VECTOR, false},
	{"centroid_lite", PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_CENTROID_LITE, false},
	{"quantized_inverted_experimental", PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_QUANTIZED_INVERTED_EXPERIMENTAL, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_multivector_quantized_inverted_codebook_options[] = {
	{"deterministic", PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_DETERMINISTIC, false},
	{"external", PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_EXTERNAL, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_multivector_quantized_inverted_compact_scoring_options[] = {
	{"off", PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_COMPACT_SCORING_OFF, false},
	{"experimental", PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_COMPACT_SCORING_EXPERIMENTAL, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_multivector_quantized_inverted_compact_doc_order_options[] = {
	{"original", PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_COMPACT_DOC_ORDER_ORIGINAL, false},
	{"docid", PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_COMPACT_DOC_ORDER_DOCID, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_multivector_quantized_inverted_query_codeword_kernel_options[] = {
	{"auto", PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_QUERY_CODEWORD_KERNEL_AUTO, false},
	{"scalar", PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_QUERY_CODEWORD_KERNEL_SCALAR, false},
	{"blocked", PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_QUERY_CODEWORD_KERNEL_BLOCKED, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_multivector_quantized_inverted_precompact_options[] = {
	{"off", PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_PRECOMPACT_OFF, false},
	{"centroid_maxsim_topk", PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_PRECOMPACT_CENTROID_MAXSIM_TOPK, false},
	{"centroid_maxsim_reservoir", PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_PRECOMPACT_CENTROID_MAXSIM_RESERVOIR, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_multivector_quantized_inverted_token_coverage_options[] = {
	{"off", PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_TOKEN_COVERAGE_OFF, false},
	{"linear", PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_TOKEN_COVERAGE_LINEAR, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_multivector_centroid_lite_bitset_prefilter_options[] = {
	{"off", PGTURBOHYBRID_MULTIVECTOR_CENTROID_LITE_BITSET_PREFILTER_OFF, false},
	{"experimental", PGTURBOHYBRID_MULTIVECTOR_CENTROID_LITE_BITSET_PREFILTER_EXPERIMENTAL, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_multivector_centroid_lite_pruning_options[] = {
	{"off", PGTURBOHYBRID_MULTIVECTOR_CENTROID_LITE_PRUNING_OFF, false},
	{"safe_upper_bound", PGTURBOHYBRID_MULTIVECTOR_CENTROID_LITE_PRUNING_SAFE_UPPER_BOUND, false},
	{"score_bound_experimental", PGTURBOHYBRID_MULTIVECTOR_CENTROID_LITE_PRUNING_SCORE_BOUND_EXPERIMENTAL, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_multivector_quantized_inverted_pruning_options[] = {
	{"off", PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_PRUNING_OFF, false},
	{"score_bound_experimental", PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_PRUNING_SCORE_BOUND_EXPERIMENTAL, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_multivector_posting_selection_options[] = {
	{"uniform_stride", PGTURBOHYBRID_MULTIVECTOR_POSTING_SELECTION_UNIFORM_STRIDE, false},
	{"score_topk", PGTURBOHYBRID_MULTIVECTOR_POSTING_SELECTION_SCORE_TOPK, false},
	{"union_score", PGTURBOHYBRID_MULTIVECTOR_POSTING_SELECTION_UNION_SCORE, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_multivector_centroid_lite_candidate_scoring_options[] = {
	{"posting_payload", PGTURBOHYBRID_MULTIVECTOR_CENTROID_LITE_CANDIDATE_SCORING_POSTING_PAYLOAD, false},
	{"codeword_maxsim", PGTURBOHYBRID_MULTIVECTOR_CENTROID_LITE_CANDIDATE_SCORING_CODEWORD_MAXSIM, false},
	{"doc_centroid_maxsim", PGTURBOHYBRID_MULTIVECTOR_CENTROID_LITE_CANDIDATE_SCORING_DOC_CENTROID_MAXSIM, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_multivector_plain_fallback_options[] = {
	{"auto", PGTURBOHYBRID_MULTIVECTOR_PLAIN_FALLBACK_AUTO, false},
	{"off", PGTURBOHYBRID_MULTIVECTOR_PLAIN_FALLBACK_OFF, false},
	{"force", PGTURBOHYBRID_MULTIVECTOR_PLAIN_FALLBACK_FORCE, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_multivector_candidate_reservoirs_options[] = {
	{"off", PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_RESERVOIRS_OFF, false},
	{"conservative", PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_RESERVOIRS_CONSERVATIVE, false},
	{"balanced", PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_RESERVOIRS_BALANCED, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_multivector_bm25_candidate_injection_options[] = {
	{"off", PGTURBOHYBRID_MULTIVECTOR_BM25_CANDIDATE_INJECTION_OFF, false},
	{"hybrid_only", PGTURBOHYBRID_MULTIVECTOR_BM25_CANDIDATE_INJECTION_HYBRID_ONLY, false},
	{"dense_with_text", PGTURBOHYBRID_MULTIVECTOR_BM25_CANDIDATE_INJECTION_DENSE_WITH_TEXT, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_multivector_sparse_candidate_source_options[] = {
	{"off", PGTURBOHYBRID_MULTIVECTOR_SPARSE_CANDIDATE_SOURCE_OFF, false},
	{"bm25", PGTURBOHYBRID_MULTIVECTOR_SPARSE_CANDIDATE_SOURCE_BM25, false},
	{"learned_sparse", PGTURBOHYBRID_MULTIVECTOR_SPARSE_CANDIDATE_SOURCE_LEARNED_SPARSE, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_multivector_branch_plan_options[] = {
	{"auto", PGTURBOHYBRID_BRANCH_PLAN_AUTO, false},
	{"dense_only", PGTURBOHYBRID_BRANCH_PLAN_DENSE_ONLY, false},
	{"qdrant_like", PGTURBOHYBRID_BRANCH_PLAN_QDRANT_LIKE, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_dense_residual_rerank_mode_options[] = {
	{"off", PGTURBOHYBRID_DENSE_RESIDUAL_RERANK_OFF, false},
	{"fixed", PGTURBOHYBRID_DENSE_RESIDUAL_RERANK_FIXED, false},
	{"calibrated", PGTURBOHYBRID_DENSE_RESIDUAL_RERANK_CALIBRATED, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_dense_local_expansion_options[] = {
	{"off", PGTURBOHYBRID_DENSE_LOCAL_EXPANSION_OFF, false},
	{"auto", PGTURBOHYBRID_DENSE_LOCAL_EXPANSION_AUTO, false},
	{"on", PGTURBOHYBRID_DENSE_LOCAL_EXPANSION_ON, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_payload_entry_seeding_options[] = {
	{"off", PGTURBOHYBRID_PAYLOAD_ENTRY_SEEDING_OFF, false},
	{"auto", PGTURBOHYBRID_PAYLOAD_ENTRY_SEEDING_AUTO, false},
	{"on", PGTURBOHYBRID_PAYLOAD_ENTRY_SEEDING_ON, false},
	{NULL, 0, false}
};

static const struct config_enum_entry pgturbohybrid_final_diversity_options[] = {
	{"off", PGTURBOHYBRID_FINAL_DIVERSITY_OFF, false},
	{"group_payload", PGTURBOHYBRID_FINAL_DIVERSITY_GROUP_PAYLOAD, false},
	{NULL, 0, false}
};

typedef struct PgturbohybridProfileDefaults
{
	int			denseK;
	int			bm25K;
	int			rrfK;
	bool		enableWand;
	bool		enableSimd;
	int			bm25Strategy;
	int			bm25ImpactOrMode;
	int			bm25HotPostingsCacheMb;
	int			bm25AccumulatorMode;
	int			bm25HybridBound;
	bool		exactRescoreForBm25Only;
	bool		autoBudget;
	int			autoBudgetMinDenseK;
	int			autoBudgetMinBm25K;
	int			autoBudgetLimitMultiplier;
	int			autoBudgetQualityCap;
	bool		autoBm25Budget;
	int			autoBm25BudgetMin;
	int			autoBm25BudgetMax;
	bool		autoBm25BudgetDenseConfidence;
	const char *denseAdaptiveWidening;
	const char *denseAdaptiveWideningMultiplier;
	const char *denseAdaptiveWideningMaxMultiplier;
	const char *denseUncertaintyRetry;
	int			multivectorDocCandidateK;
	int			multivectorExactRerankK;
} PgturbohybridProfileDefaults;

const char *
PgturbohybridHybridBudgetPolicyName(int policy)
{
	switch ((PgturbohybridHybridBudgetPolicy) policy)
	{
		case PGTURBOHYBRID_HYBRID_BUDGET_ADAPTIVE:
			return "adaptive";
		case PGTURBOHYBRID_HYBRID_BUDGET_FIXED:
		default:
			return "fixed";
	}
}

const char *
PgturbohybridProfileName(int profile)
{
	switch ((PgturbohybridProfile) profile)
	{
		case PGTURBOHYBRID_PROFILE_LATENCY:
			return "latency";
		case PGTURBOHYBRID_PROFILE_BALANCED:
			return "balanced";
		case PGTURBOHYBRID_PROFILE_QUALITY:
			return "quality";
		case PGTURBOHYBRID_PROFILE_MATCHED_RECALL:
			return "matched_recall";
		case PGTURBOHYBRID_PROFILE_HIGH_RECALL:
			return "high_recall";
		case PGTURBOHYBRID_PROFILE_DEBUG:
			return "debug";
		default:
			return "unknown";
	}
}

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

const char *
PgturbohybridBm25ImpactOrModeName(int mode)
{
	switch ((PgturbohybridBm25ImpactOrMode) mode)
	{
		case PGTURBOHYBRID_BM25_IMPACT_OR_MODE_OFF:
			return "off";
		case PGTURBOHYBRID_BM25_IMPACT_OR_MODE_EXACT_ONLY:
			return "exact_only";
		case PGTURBOHYBRID_BM25_IMPACT_OR_MODE_APPROX:
			return "approx";
		default:
			return "unknown";
	}
}

const char *
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

const char *
PgturbohybridBm25HeapTSVectorRerankModeName(int mode)
{
	switch ((PgturbohybridBm25HeapTSVectorRerankMode) mode)
	{
		case PGTURBOHYBRID_BM25_HEAP_TSVECTOR_RERANK_OFF:
			return "off";
		case PGTURBOHYBRID_BM25_HEAP_TSVECTOR_RERANK_TOPK:
			return "topk";
		case PGTURBOHYBRID_BM25_HEAP_TSVECTOR_RERANK_BAND:
			return "band";
		case PGTURBOHYBRID_BM25_HEAP_TSVECTOR_RERANK_AUTO:
			return "auto";
		default:
			return "unknown";
	}
}

const char *
PgturbohybridFinalDiversityName(int mode)
{
	switch ((PgturbohybridFinalDiversityMode) mode)
	{
		case PGTURBOHYBRID_FINAL_DIVERSITY_OFF:
			return "off";
		case PGTURBOHYBRID_FINAL_DIVERSITY_GROUP_PAYLOAD:
			return "group_payload";
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

static void
PgturbohybridSetDynamicDefaultInt(const char *name, int value)
{
	char		buf[32];

	snprintf(buf, sizeof(buf), "%d", value);
	(void) SetConfigOption(name, buf, PGC_USERSET, PGC_S_DYNAMIC_DEFAULT);
}

static void
PgturbohybridSetDynamicDefaultBool(const char *name, bool value)
{
	(void) SetConfigOption(name, value ? "true" : "false",
						   PGC_USERSET, PGC_S_DYNAMIC_DEFAULT);
}

static void
PgturbohybridSetDynamicDefaultString(const char *name, const char *value)
{
	(void) SetConfigOption(name, value, PGC_USERSET, PGC_S_DYNAMIC_DEFAULT);
}

static bool
PgturbohybridGucSourceIsExplicit(GucSource source)
{
	return source > PGC_S_DYNAMIC_DEFAULT;
}

static void
PgturbohybridBumpGucGeneration(void)
{
	if (++pgturbohybrid_guc_generation == 0)
		pgturbohybrid_guc_generation = 1;
}

static bool
PgturbohybridCheckMaxIntGuc(const char *name, int value, int maxValue)
{
	if (value > maxValue)
	{
		GUC_check_errmsg("%s must not exceed %d", name, maxValue);
		return false;
	}

	return true;
}

static bool
PgturbohybridCheckDefaultDenseK(int *newval, void **extra, GucSource source)
{
	return PgturbohybridCheckMaxIntGuc("turbohybrid.default_dense_k", *newval,
									  PGTURBOHYBRID_MAX_DEFAULT_DENSE_K);
}

static bool
PgturbohybridCheckDefaultBm25K(int *newval, void **extra, GucSource source)
{
	return PgturbohybridCheckMaxIntGuc("turbohybrid.default_bm25_k", *newval,
									  PGTURBOHYBRID_MAX_DEFAULT_BM25_K);
}

static bool
PgturbohybridCheckDefaultSparseK(int *newval, void **extra, GucSource source)
{
	return PgturbohybridCheckMaxIntGuc("turbohybrid.default_sparse_k", *newval,
									  PGTURBOHYBRID_MAX_DEFAULT_SPARSE_K);
}

static bool
PgturbohybridCheckDefaultRrfK(int *newval, void **extra, GucSource source)
{
	return PgturbohybridCheckMaxIntGuc("turbohybrid.default_rrf_k", *newval,
									  PGTURBOHYBRID_MAX_RRF_K);
}

static bool
PgturbohybridCheckMaxUnionCandidates(int *newval, void **extra, GucSource source)
{
	return PgturbohybridCheckMaxIntGuc("turbohybrid.max_union_candidates", *newval,
									  PGTURBOHYBRID_MAX_UNION_CANDIDATES);
}

static void
PgturbohybridAssignQueryDefaultInt(int newval, void *extra)
{
	PgturbohybridBumpGucGeneration();
}

static bool
PgturbohybridCheckBm25Strategy(int *newval, void **extra, GucSource source)
{
	pgturbohybrid_bm25_strategy_user_set = PgturbohybridGucSourceIsExplicit(source);
	return true;
}

static bool
PgturbohybridCheckBm25ImpactOrMode(int *newval, void **extra, GucSource source)
{
	pgturbohybrid_bm25_impact_or_mode_user_set = PgturbohybridGucSourceIsExplicit(source);
	return true;
}

static bool
PgturbohybridCheckBm25HotPostingsCacheMb(int *newval, void **extra, GucSource source)
{
	if (!PgturbohybridCheckMaxIntGuc("turbohybrid.bm25_hot_postings_cache_mb", *newval,
									PGTURBOHYBRID_MAX_HOT_POSTINGS_CACHE_MB))
		return false;

	pgturbohybrid_bm25_hot_postings_cache_mb_user_set =
		PgturbohybridGucSourceIsExplicit(source);
	return true;
}

static bool
PgturbohybridCheckBm25HotPostingsCacheMinDf(int *newval, void **extra, GucSource source)
{
	pgturbohybrid_bm25_hot_postings_cache_min_df_user_set =
		PgturbohybridGucSourceIsExplicit(source);
	return true;
}

static bool
PgturbohybridCheckBm25HybridBound(int *newval, void **extra, GucSource source)
{
	pgturbohybrid_bm25_hybrid_bound_user_set = PgturbohybridGucSourceIsExplicit(source);
	return true;
}

static bool
PgturbohybridCheckBm25AccumulatorMode(int *newval, void **extra, GucSource source)
{
	pgturbohybrid_bm25_accumulator_mode_user_set =
		PgturbohybridGucSourceIsExplicit(source);
	return true;
}

static bool
PgturbohybridCheckDenseBuildExactDistances(bool *newval, void **extra, GucSource source)
{
	pgturbohybrid_dense_build_exact_distances_user_set =
		PgturbohybridGucSourceIsExplicit(source);
	return true;
}

static bool
PgturbohybridCheckDenseHeapRescore(int *newval, void **extra, GucSource source)
{
	if (*newval == PGTURBOHYBRID_DENSE_HEAP_RESCORE_AUTO)
		pgturbohybrid_dense_heap_rescore_user_set = false;
	else
		pgturbohybrid_dense_heap_rescore_user_set =
			PgturbohybridGucSourceIsExplicit(source);
	return true;
}

static void
PgturbohybridProfileDefaultsFor(int profile, PgturbohybridProfileDefaults *defaults)
{
	memset(defaults, 0, sizeof(*defaults));
	defaults->rrfK = PGTURBOHYBRID_DEFAULT_RRF_K;
	defaults->enableWand = true;
	defaults->enableSimd = true;
	defaults->bm25Strategy = PGTURBOHYBRID_BM25_STRATEGY_AUTO;
	defaults->bm25ImpactOrMode = PGTURBOHYBRID_BM25_IMPACT_OR_MODE_EXACT_ONLY;
	defaults->bm25AccumulatorMode = PGTURBOHYBRID_BM25_ACCUMULATOR_AUTO;
	defaults->bm25HybridBound = PGTURBOHYBRID_BM25_HYBRID_BOUND_SAFE;
	defaults->autoBudget = true;
	defaults->autoBudgetLimitMultiplier = 10;
	defaults->denseAdaptiveWidening = "off";
	defaults->denseAdaptiveWideningMultiplier = "2.0";
	defaults->denseAdaptiveWideningMaxMultiplier = "4.0";
	defaults->multivectorDocCandidateK = 100;
	defaults->multivectorExactRerankK = 100;
	/*
	 * Keep bounded retry opt-in until retrieval-quality grids show a stable
	 * quality win with acceptable p95/p99 cost for the target workload.
	 */
	defaults->denseUncertaintyRetry = "off";

	switch ((PgturbohybridProfile) profile)
	{
		case PGTURBOHYBRID_PROFILE_BALANCED:
			defaults->denseK = 200;
			defaults->bm25K = 200;
			defaults->multivectorDocCandidateK = 200;
			defaults->multivectorExactRerankK = 200;
			defaults->bm25HotPostingsCacheMb = 16;
			defaults->autoBudgetMinDenseK = 64;
			defaults->autoBudgetMinBm25K = 64;
			defaults->autoBudgetQualityCap = 200;
			defaults->autoBm25Budget = true;
			defaults->autoBm25BudgetMin = 64;
			defaults->autoBm25BudgetMax = 200;
			defaults->autoBm25BudgetDenseConfidence = true;
			defaults->denseAdaptiveWidening = "auto";
			defaults->denseAdaptiveWideningMultiplier = "1.5";
			defaults->denseAdaptiveWideningMaxMultiplier = "1.5";
			break;
		case PGTURBOHYBRID_PROFILE_QUALITY:
			defaults->denseK = 400;
			defaults->bm25K = 400;
			defaults->multivectorDocCandidateK = 400;
			defaults->multivectorExactRerankK = 400;
			defaults->enableSimd = true;
			defaults->bm25Strategy = PGTURBOHYBRID_BM25_STRATEGY_AUTO;
			defaults->bm25ImpactOrMode = PGTURBOHYBRID_BM25_IMPACT_OR_MODE_EXACT_ONLY;
			defaults->bm25HotPostingsCacheMb = 16;
			defaults->bm25HybridBound = PGTURBOHYBRID_BM25_HYBRID_BOUND_SAFE;
			defaults->exactRescoreForBm25Only = true;
			defaults->autoBudget = false;
			defaults->autoBudgetMinDenseK = 100;
			defaults->autoBudgetMinBm25K = 100;
			defaults->autoBudgetQualityCap = 400;
			defaults->autoBm25Budget = false;
			defaults->autoBm25BudgetMin = 100;
			defaults->autoBm25BudgetMax = 400;
			defaults->denseAdaptiveWidening = "auto";
			defaults->denseAdaptiveWideningMultiplier = "2.0";
			defaults->denseAdaptiveWideningMaxMultiplier = "2.0";
			break;
		case PGTURBOHYBRID_PROFILE_MATCHED_RECALL:
			defaults->denseK = 200;
			defaults->bm25K = 200;
			defaults->multivectorDocCandidateK = 200;
			defaults->multivectorExactRerankK = 200;
			defaults->bm25HotPostingsCacheMb = 16;
			defaults->autoBudgetMinDenseK = 64;
			defaults->autoBudgetMinBm25K = 64;
			defaults->autoBudgetQualityCap = 200;
			defaults->autoBm25Budget = true;
			defaults->autoBm25BudgetMin = 64;
			defaults->autoBm25BudgetMax = 200;
			defaults->autoBm25BudgetDenseConfidence = true;
			defaults->denseAdaptiveWidening = "auto";
			defaults->denseAdaptiveWideningMultiplier = "1.5";
			defaults->denseAdaptiveWideningMaxMultiplier = "1.5";
			break;
		case PGTURBOHYBRID_PROFILE_HIGH_RECALL:
			/*
			 * Exact-free, high-recall point: matched_recall's candidate
			 * budgets with adaptive widening off. The defining behavior —
			 * full band heap rescore — is resolved at scan time in
			 * PgturbohybridGraphEffectiveHeapRescoreMode (profile case
			 * HIGH_RECALL -> BAND) rather than mutated here as a GUC default,
			 * because applying dense_heap_rescore via a dynamic default
			 * regressed heap-band recall. On quantized (exact_storage=off)
			 * indexes the band rescore re-ranks 4-bit code candidates against
			 * exact heap vectors, spending latency headroom rather than index
			 * size. Pair with a heuristic graph build
			 * (graph_ef_construction=256, graph_ef_search=192,
			 * graph_oversampling=12, native_segments=1), which this profile
			 * also supplies as defaults when the index does not set those
			 * reloptions explicitly.
			 */
			defaults->denseK = 200;
			defaults->bm25K = 200;
			defaults->multivectorDocCandidateK = 400;
			defaults->multivectorExactRerankK = 400;
			defaults->bm25HotPostingsCacheMb = 16;
			defaults->autoBudgetMinDenseK = 64;
			defaults->autoBudgetMinBm25K = 64;
			defaults->autoBudgetQualityCap = 200;
			defaults->autoBm25Budget = true;
			defaults->autoBm25BudgetMin = 64;
			defaults->autoBm25BudgetMax = 200;
			defaults->autoBm25BudgetDenseConfidence = true;
			defaults->denseAdaptiveWidening = "off";
			break;
		case PGTURBOHYBRID_PROFILE_DEBUG:
			defaults->denseK = 400;
			defaults->bm25K = 400;
			defaults->multivectorDocCandidateK = 400;
			defaults->multivectorExactRerankK = 400;
			defaults->enableSimd = false;
			defaults->bm25HotPostingsCacheMb = 0;
			defaults->bm25HybridBound = PGTURBOHYBRID_BM25_HYBRID_BOUND_OFF;
			defaults->exactRescoreForBm25Only = true;
			defaults->autoBudget = false;
			defaults->autoBudgetMinDenseK = 100;
			defaults->autoBudgetMinBm25K = 100;
			defaults->autoBudgetQualityCap = 400;
			defaults->autoBm25Budget = false;
			defaults->autoBm25BudgetMin = 100;
			defaults->autoBm25BudgetMax = 400;
			defaults->denseAdaptiveWidening = "auto";
			defaults->denseAdaptiveWideningMultiplier = "2.0";
			defaults->denseAdaptiveWideningMaxMultiplier = "2.0";
			defaults->denseUncertaintyRetry = "auto";
			break;
		case PGTURBOHYBRID_PROFILE_LATENCY:
		default:
			defaults->denseK = PGTURBOHYBRID_DEFAULT_DENSE_K;
			defaults->bm25K = PGTURBOHYBRID_DEFAULT_BM25_K;
			defaults->bm25ImpactOrMode = PGTURBOHYBRID_BM25_IMPACT_OR_MODE_APPROX;
			defaults->bm25HotPostingsCacheMb = 32;
			defaults->bm25HybridBound = PGTURBOHYBRID_BM25_HYBRID_BOUND_APPROX;
			defaults->autoBudgetMinDenseK = PGTURBOHYBRID_DEFAULT_DENSE_K;
			defaults->autoBudgetMinBm25K = PGTURBOHYBRID_DEFAULT_BM25_K;
			defaults->autoBudgetQualityCap = PGTURBOHYBRID_DEFAULT_DENSE_K;
			defaults->autoBm25Budget = false;
			defaults->autoBm25BudgetMin = PGTURBOHYBRID_DEFAULT_BM25_K;
			defaults->autoBm25BudgetMax = PGTURBOHYBRID_DEFAULT_BM25_K;
			break;
	}
}

void
PgturbohybridApplyProfileDefaults(void)
{
	PgturbohybridProfileDefaults defaults;

	PgturbohybridProfileDefaultsFor(pgturbohybrid_profile, &defaults);

	PgturbohybridSetDynamicDefaultInt("turbohybrid.default_dense_k",
									  defaults.denseK);
	PgturbohybridSetDynamicDefaultInt("turbohybrid.default_bm25_k",
									  defaults.bm25K);
	PgturbohybridSetDynamicDefaultInt("turbohybrid.default_sparse_k",
									  defaults.bm25K);
	PgturbohybridSetDynamicDefaultInt("turbohybrid.default_rrf_k",
									  defaults.rrfK);
	PgturbohybridSetDynamicDefaultBool("turbohybrid.enable_wand",
									   defaults.enableWand);
	PgturbohybridSetDynamicDefaultBool("turbohybrid.simd",
									   defaults.enableSimd);

	PgturbohybridSetDynamicDefaultString("turbohybrid.bm25_strategy",
										 PgturbohybridBm25StrategyName(defaults.bm25Strategy));
	PgturbohybridSetDynamicDefaultString("turbohybrid.bm25_impact_or_mode",
										 PgturbohybridBm25ImpactOrModeName(defaults.bm25ImpactOrMode));
	PgturbohybridSetDynamicDefaultInt("turbohybrid.bm25_hot_postings_cache_mb",
									  defaults.bm25HotPostingsCacheMb);
	PgturbohybridSetDynamicDefaultInt("turbohybrid.bm25_hot_postings_cache_min_df",
									  1024);
	PgturbohybridSetDynamicDefaultString("turbohybrid.bm25_accumulator_mode",
										 PgturbohybridBm25AccumulatorModeName(defaults.bm25AccumulatorMode));
	PgturbohybridSetDynamicDefaultString("turbohybrid.bm25_hybrid_bound",
										 PgturbohybridBm25HybridBoundModeName(defaults.bm25HybridBound));
	pgturbohybrid_enable_exact_rescore_for_bm25_only =
		defaults.exactRescoreForBm25Only;
	pgturbohybrid_auto_budget = defaults.autoBudget;
	pgturbohybrid_auto_budget_min_dense_k = defaults.autoBudgetMinDenseK;
	pgturbohybrid_auto_budget_min_bm25_k = defaults.autoBudgetMinBm25K;
	pgturbohybrid_auto_budget_limit_multiplier =
		defaults.autoBudgetLimitMultiplier;
	pgturbohybrid_auto_budget_quality_cap = defaults.autoBudgetQualityCap;
	pgturbohybrid_auto_bm25_budget = defaults.autoBm25Budget;
	pgturbohybrid_auto_bm25_budget_min = defaults.autoBm25BudgetMin;
	pgturbohybrid_auto_bm25_budget_max = defaults.autoBm25BudgetMax;
	pgturbohybrid_auto_bm25_budget_dense_confidence =
		defaults.autoBm25BudgetDenseConfidence;
	PgturbohybridSetDynamicDefaultString("turbohybrid.dense_adaptive_widening",
										 defaults.denseAdaptiveWidening);
	PgturbohybridSetDynamicDefaultString("turbohybrid.dense_adaptive_widening_multiplier",
										 defaults.denseAdaptiveWideningMultiplier);
	PgturbohybridSetDynamicDefaultString("turbohybrid.dense_adaptive_widening_max_multiplier",
										 defaults.denseAdaptiveWideningMaxMultiplier);
	PgturbohybridSetDynamicDefaultString("turbohybrid.dense_uncertainty_retry",
										 defaults.denseUncertaintyRetry);
	PgturbohybridSetDynamicDefaultInt("turbohybrid.multivector_doc_candidate_k",
									  defaults.multivectorDocCandidateK);
	PgturbohybridSetDynamicDefaultInt("turbohybrid.multivector_exact_rerank_k",
									  defaults.multivectorExactRerankK);
}

static void
PgturbohybridAssignProfile(int newval, void *extra)
{
	pgturbohybrid_profile = newval;
	PgturbohybridBumpGucGeneration();
	PgturbohybridApplyProfileDefaults();
}

static bool
PgturbohybridNativeBuildWorkersCheck(char **newval, void **extra, GucSource source)
{
	char	   *setting = *newval;

	(void) extra;
	(void) source;

	if (setting == NULL)
		return false;
	if (pg_strcasecmp(setting, "auto") == 0 ||
		strcmp(setting, "0") == 0 ||
		strcmp(setting, "1") == 0 ||
		strcmp(setting, "2") == 0 ||
		strcmp(setting, "4") == 0 ||
		strcmp(setting, "8") == 0)
		return true;

	GUC_check_errdetail("Use \"auto\", 0, 1, 2, 4, or 8.");
	return false;
}

static void
PgturbohybridDefineDefaultBudgetGUCs(void)
{
	DefineCustomIntVariable("turbohybrid.default_dense_k", "Default dense candidate budget for turbohybrid_query callers",
							NULL, &pgturbohybrid_default_dense_k,
							PGTURBOHYBRID_DEFAULT_DENSE_K, 0,
							PGTURBOHYBRID_MAX_DEFAULT_DENSE_K,
							PGC_USERSET, 0, PgturbohybridCheckDefaultDenseK,
							PgturbohybridAssignQueryDefaultInt, NULL);
	DefineCustomIntVariable("turbohybrid.default_bm25_k", "Default BM25 candidate budget for turbohybrid_query callers",
							NULL, &pgturbohybrid_default_bm25_k,
							PGTURBOHYBRID_DEFAULT_BM25_K, 0,
							PGTURBOHYBRID_MAX_DEFAULT_BM25_K,
							PGC_USERSET, 0, PgturbohybridCheckDefaultBm25K,
							PgturbohybridAssignQueryDefaultInt, NULL);
	DefineCustomIntVariable("turbohybrid.default_sparse_k", "Default sparse-vector candidate budget for turbohybrid_query callers",
							NULL, &pgturbohybrid_default_sparse_k,
							PGTURBOHYBRID_DEFAULT_SPARSE_K, 0,
							PGTURBOHYBRID_MAX_DEFAULT_SPARSE_K,
							PGC_USERSET, 0, PgturbohybridCheckDefaultSparseK,
							PgturbohybridAssignQueryDefaultInt, NULL);
	DefineCustomIntVariable("turbohybrid.default_rrf_k", "Default RRF constant for turbohybrid_query callers",
							NULL, &pgturbohybrid_default_rrf_k,
							PGTURBOHYBRID_DEFAULT_RRF_K, 1,
							PGTURBOHYBRID_MAX_RRF_K,
							PGC_USERSET, 0, PgturbohybridCheckDefaultRrfK,
							PgturbohybridAssignQueryDefaultInt, NULL);
}

void
PgturbohybridRegisterGUCs(void)
{
	PgturbohybridDefineDefaultBudgetGUCs();
	DefineCustomBoolVariable("turbohybrid.enable_wand", "Enable WAND pruning for BM25 candidate generation",
							 NULL, &pgturbohybrid_enable_wand,
							 true, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("turbohybrid.enable_sparse_wand", "Enable block-max WAND pruning for sparse candidate generation",
							 NULL, &pgturbohybrid_enable_sparse_wand,
							 true, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.sparse_hot_postings_cache_mb", "Backend-local sparse hot-postings cache size in MB (0 disables)",
							NULL, &pgturbohybrid_sparse_hot_postings_cache_mb,
							16, 0, 65536, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.sparse_hot_postings_cache_min_df", "Minimum term document frequency to cache decoded sparse postings",
							NULL, &pgturbohybrid_sparse_hot_postings_cache_min_df,
							256, 1, 1000000000, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.sparse_delta_compaction_threshold", "Auto-compact the sparse delta chain once it holds this many documents (0 disables auto-compaction)",
							NULL, &pgturbohybrid_sparse_delta_compaction_threshold,
							1000, 0, 1000000000, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("turbohybrid.fast_weighted_score_bound_pruning",
							 "Enable exact fused-score bound pruning for fast_weighted BM25 traversal",
							 "Only applies to turbohybrid_query(fusion => 'fast_weighted'); RRF and distribution-normalized weighted fusion do not use this pruning path.",
							 &pgturbohybrid_fast_weighted_score_bound_pruning,
							 true, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomRealVariable("turbohybrid.calibrated_fusion_both_match_bonus",
							 "Score bonus for calibrated fusion candidates that appear in both dense and BM25 branches",
							 NULL, &pgturbohybrid_calibrated_fusion_both_match_bonus,
							 0.06, 0.0, 1.0, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomRealVariable("turbohybrid.calibrated_fusion_identifier_bm25_alpha",
							 "Dense alpha used by calibrated fusion for rare identifier-like text queries",
							 "Lower values give BM25 more weight.",
							 &pgturbohybrid_calibrated_fusion_identifier_bm25_alpha,
							 0.35, 0.0, 1.0, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomRealVariable("turbohybrid.calibrated_fusion_broad_dense_alpha",
							 "Dense alpha used by calibrated fusion for broad natural-language queries",
							 "Higher values give dense similarity more weight.",
							 &pgturbohybrid_calibrated_fusion_broad_dense_alpha,
							 0.70, 0.0, 1.0, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomRealVariable("turbohybrid.calibrated_fusion_default_alpha",
							 "Dense alpha used by calibrated fusion for mixed/default query shapes",
							 NULL, &pgturbohybrid_calibrated_fusion_default_alpha,
							 0.50, 0.0, 1.0, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomRealVariable("turbohybrid.dbsf_sigma",
							 "Sigma window used by DBSF distribution-based score normalization",
							 "DBSF maps each branch score through clipped mean +/- sigma * stddev endpoints before weighted summation.",
							 &pgturbohybrid_dbsf_sigma,
							 3.0, 0.001, 100.0, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.dbsf_min_branch_candidates",
							"Minimum usable branch candidates before DBSF treats a branch as degenerate",
							"Degenerate DBSF branches normalize present scores to 0.5 and are exposed in scan stats.",
							&pgturbohybrid_dbsf_min_branch_candidates,
							10, 1, 1000000, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.dbsf_robust",
							 "Robust DBSF normalization mode",
							 "Use 'mad' to center DBSF on the branch median and scale by median absolute deviation.",
							 &pgturbohybrid_dbsf_robust,
							 PGTURBOHYBRID_DBSF_ROBUST_OFF,
							 pgturbohybrid_dbsf_robust_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_max_doc_vectors",
							"Maximum token vectors accepted for one indexed multivector document",
							"Multivector index builds use this as the per-document token cap. Single-vector indexes are unaffected.",
							&pgturbohybrid_multivector_max_doc_vectors,
							PGTURBOHYBRID_MULTIVECTOR_DEFAULT_MAX_DOC_VECTORS,
							1, PGTURBOHYBRID_MAX_MULTIVECTOR_DOC_VECTORS,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_max_query_vectors",
							"Maximum token vectors accepted for one multivector query",
							"Multivector scans use this as the per-query MaxSim accumulator cap. Single-vector scans are unaffected.",
							&pgturbohybrid_multivector_max_query_vectors,
							PGTURBOHYBRID_MULTIVECTOR_DEFAULT_MAX_QUERY_VECTORS,
							1, PGTURBOHYBRID_MAX_MULTIVECTOR_QUERY_VECTORS,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_max_dim",
							"Maximum dimensions accepted for multivector token vectors",
							"Multivector build and query paths use this cap before token-node expansion or document-node sidecar storage.",
							&pgturbohybrid_multivector_max_dim,
							PGTURBOHYBRID_MULTIVECTOR_MAX_DIM,
							1, PGTURBOHYBRID_MULTIVECTOR_MAX_DIM,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomStringVariable("turbohybrid.multivector_model_name",
							   "Late-interaction model metadata profile for multivector validation",
							   "When set to a registered model name, multivector dimensions are checked against the model profile and suspicious token counts produce warnings.",
							   &pgturbohybrid_multivector_model_name,
							   "",
							   PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_subvector_k",
							"Subvector ANN hits collected for each multivector query token",
							"Multivector MaxSim scans run one bounded graph search per query token and merge hits by heap tuple.",
							&pgturbohybrid_multivector_subvector_k,
							100, 1, 100000,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_unique_docs_per_token",
							"Unique document hits retained per multivector query token",
							"Stops each query token after this many distinct heap tuples have contributed to the MaxSim accumulator.",
							&pgturbohybrid_multivector_unique_docs_per_token,
							100, 1, 100000,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_max_raw_hits_per_token",
							"Maximum raw subvector hits scanned per multivector query token",
							"Caps duplicate subvector hits before document-level MaxSim aggregation.",
							&pgturbohybrid_multivector_max_raw_hits_per_token,
							400, 1, 100000,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.multivector_adaptive_widening",
							 "Adaptive raw-hit widening mode for multivector query tokens",
							 "When enabled, multivector scans start from multivector_subvector_k and widen per token up to multivector_max_raw_hits_per_token if too few unique documents are seen.",
							 &pgturbohybrid_multivector_adaptive_widening,
							 PGTURBOHYBRID_MULTIVECTOR_ADAPTIVE_WIDENING_AUTO,
							 pgturbohybrid_multivector_adaptive_widening_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.multivector_docmap",
							 "Persistent docmap sidecar mode for multivector scans",
							 "auto uses the sidecar when available and falls back to heap-TID hashing for old indexes; require errors if the sidecar is missing.",
							 &pgturbohybrid_multivector_docmap,
							 PGTURBOHYBRID_MULTIVECTOR_DOCMAP_AUTO,
							 pgturbohybrid_multivector_docmap_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_doc_candidate_k",
							"Document candidates retained after approximate multivector MaxSim aggregation",
							"Final multivector dense results are truncated by this document-level candidate budget and the query dense_k.",
							&pgturbohybrid_multivector_doc_candidate_k,
							100, 1, 100000,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_doc_graph_search_ef",
							"Document-node graph traversal breadth for multivector MaxSim scans",
							"Zero uses the index graph_ef_search setting as the traversal cap; positive values override traversal ef for document-node multivector graphs.",
							&pgturbohybrid_multivector_doc_graph_search_ef,
							0, 0, 100000,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_doc_graph_oversampling",
							"Document-node graph candidate oversampling multiplier for multivector MaxSim scans",
							"Document-node graph traversal retains this multiple of the exact-rescore budget before final reranking.",
							&pgturbohybrid_multivector_doc_graph_oversampling,
							1, 1, 100,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_doc_graph_rescore_k",
							"Document-node graph exact rescore prefix for multivector MaxSim scans",
							"Zero uses multivector_doc_candidate_k; positive values set the document-node exact-rescore budget independently from final_k.",
							&pgturbohybrid_multivector_doc_graph_rescore_k,
							0, 0, 100000,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.multivector_doc_storage",
							 "Document-node multivector sidecar scoring storage",
							 "f32 uses the exact float32 sidecar for traversal; f16 and sq8 build compact in-memory scoring sidecars and keep exact heap rerank for final ordering; proxy_only is an experimental index reloption for proxy graph admission with heap exact rerank.",
							 &pgturbohybrid_multivector_doc_storage,
							 PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32,
							 pgturbohybrid_multivector_doc_storage_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.multivector_doc_storage_cache",
							 "Document-node multivector sidecar cache mode",
							 "auto keeps plain proxy_vector scans on the lazy paged sidecar path and may keep other low-latency document-node scans resident when the sidecar fits native_cache_max_mb; resident explicitly loads the sidecar; paged reports per-scan sidecar page reads and bytes touched.",
							 &pgturbohybrid_multivector_doc_storage_cache,
							 PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_AUTO,
							 pgturbohybrid_multivector_doc_storage_cache_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.multivector_exact_rerank",
							 "Exact heap rerank mode for multivector MaxSim scans",
							 "topk fetches a bounded document-candidate prefix from the heap and recomputes exact f32 MaxSim; adaptive uses conservative token-norm upper bounds to skip candidates that cannot enter top-K; off keeps approximate TurboQuant ordering.",
							 &pgturbohybrid_multivector_exact_rerank,
							 PGTURBOHYBRID_MULTIVECTOR_EXACT_RERANK_ADAPTIVE,
							 pgturbohybrid_multivector_exact_rerank_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_exact_rerank_k",
							"Maximum document candidates exact-reranked for multivector MaxSim scans",
							"Bounds heap tuple fetches and exact query-token by document-token MaxSim work.",
							&pgturbohybrid_multivector_exact_rerank_k,
							100, 1, 100000,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.multivector_proxy_encoder",
							 "Default fixed-dimensional proxy encoder for new document-node indexes",
							 "Existing indexes use their persisted multivector_proxy_encoder reloption; this GUC is a session fallback when no index reloption is available.",
							 &pgturbohybrid_multivector_proxy_encoder,
							 PGTURBOHYBRID_DEFAULT_MULTIVECTOR_PROXY_ENCODER,
							 pgturbohybrid_multivector_proxy_encoder_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomStringVariable("turbohybrid.multivector_learned_projection_path",
							   "Path to learned_projection_v1 proxy weights",
							   "Empty disables learned_projection_v1. The first safe slice accepts an administrator-provided text weight file and keeps final exact MaxSim rerank unchanged.",
							   &pgturbohybrid_multivector_learned_projection_path,
							   "",
							   PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomStringVariable("turbohybrid.multivector_learned_projection_model",
							   "Expected learned_projection_v1 model profile name",
							   "When non-empty, the loaded learned projection file must declare the same model name.",
							   &pgturbohybrid_multivector_learned_projection_model,
							   "",
							   PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomStringVariable("turbohybrid.multivector_learned_projection_checksum",
							   "Expected learned_projection_v1 projection checksum",
							   "When non-empty, the loaded learned projection file must declare the same checksum string.",
							   &pgturbohybrid_multivector_learned_projection_checksum,
							   "",
							   PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("turbohybrid.multivector_allow_exact_symmetric_build",
							 "Allow exact symmetric document MaxSim graph topology builds above the diagnostic size guard",
							 "Off blocks multivector document-node exact_symmetric graph builds above turbohybrid.multivector_exact_symmetric_build_max_docs. This protects production builds and benchmarks from accidentally selecting O(doc_tokens^2 * dim) document-document MaxSim topology.",
							 &pgturbohybrid_multivector_allow_exact_symmetric_build,
							 false, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_exact_symmetric_build_max_docs",
							"Maximum document count allowed for exact symmetric multivector document graph builds without explicit override",
							"CREATE INDEX errors above this observed document count when multivector_doc_build_scorer = exact_symmetric and turbohybrid.multivector_allow_exact_symmetric_build is off.",
							&pgturbohybrid_multivector_exact_symmetric_build_max_docs,
							1000, 0, INT_MAX,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_max_accumulator_mb",
							"Maximum memory allowed for one multivector document accumulator",
							"Prevents pathological multivector queries from allocating state proportional to too many touched documents times query vectors.",
							&pgturbohybrid_multivector_max_accumulator_mb,
							64, 1, PGTURBOHYBRID_MAX_MULTIVECTOR_ACCUMULATOR_MB,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.multivector_debug_admission",
							 "Debug admission diagnostics for multivector candidate generation",
							 "summary adds bounded admission counters; trace also records a bounded document-keyed trace for the last scan.",
							 &pgturbohybrid_multivector_debug_admission,
							 PGTURBOHYBRID_MULTIVECTOR_DEBUG_ADMISSION_OFF,
							 pgturbohybrid_multivector_debug_admission_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_debug_trace_limit",
							"Maximum document trace entries recorded for multivector admission debugging",
							"Only used when turbohybrid.multivector_debug_admission is trace.",
							&pgturbohybrid_multivector_debug_trace_limit,
							1000, 0, PGTURBOHYBRID_MULTIVECTOR_DEBUG_TRACE_LIMIT_MAX,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomStringVariable("turbohybrid.multivector_debug_skip_query_tokens",
							   "Comma-separated multivector query token ordinals skipped during candidate generation",
							   "Debug-only token ablation setting; exact MaxSim rerank still uses the full query multivector.",
							   &pgturbohybrid_multivector_debug_skip_query_tokens,
							   "",
							   PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.multivector_candidate_source",
							 "Developer candidate source for multivector MaxSim scans",
							 "graph uses the index graph mode; document_nodes requires a document-node index; exact_token_scan scores all stored token nodes per query token; exact_doc_scan and doc_graph_prototype score heap documents with full MaxSim for validation; proxy_vector uses the persisted document-node proxy encoder graph for admission and exact MaxSim rerank; centroid_lite uses opt-in document-local centroids before exact MaxSim rerank; quantized_inverted_experimental is a research-only ColBERTSaR-style branch with unstable storage.",
							 &pgturbohybrid_multivector_candidate_source,
							 PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_GRAPH,
							 pgturbohybrid_multivector_candidate_source_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.multivector_quantized_inverted_codebook",
							 "Experimental quantized-inverted codebook source",
							 "deterministic preserves the temporary largest-magnitude codeword assignment; external loads an administrator-provided experimental text codebook and requires matching sidecar metadata.",
							 &pgturbohybrid_multivector_quantized_inverted_codebook,
							 PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_DETERMINISTIC,
							 pgturbohybrid_multivector_quantized_inverted_codebook_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomStringVariable("turbohybrid.multivector_quantized_inverted_codebook_path",
							   "Path to an experimental quantized-inverted external codebook",
							   "Only used when turbohybrid.multivector_quantized_inverted_codebook = external. The file header is pgturbohybrid_quantized_inverted_codebook_v1 <dim> <codebook_size> <checksum>.",
							   &pgturbohybrid_multivector_quantized_inverted_codebook_path,
							   "",
							   PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_quantized_inverted_codebook_top_m",
							"Number of codewords assigned per token for the experimental quantized-inverted branch",
							"External experimental codebooks support top_m 1-16 multi-posting assignment; deterministic codebooks remain top_m = 1.",
							&pgturbohybrid_multivector_quantized_inverted_codebook_top_m,
							1, 1, 16,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.multivector_quantized_inverted_compact_scoring",
							 "Experimental compact-code scoring mode for quantized_inverted_experimental",
							 "off preserves the existing float-token admission scorer. experimental uses compact score payloads for admission diagnostics only; final SQL ranking remains exact heap MaxSim.",
							 &pgturbohybrid_multivector_quantized_inverted_compact_scoring,
							 PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_COMPACT_SCORING_OFF,
							 pgturbohybrid_multivector_quantized_inverted_compact_scoring_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.multivector_quantized_inverted_compact_doc_order",
							 "Experimental document order for quantized-inverted compact scoring",
							 "docid scores the scan-local retained document set in ascending docId order to improve compact sidecar locality; original preserves posting-discovery order. Final SQL ranking remains exact heap MaxSim.",
							 &pgturbohybrid_multivector_quantized_inverted_compact_doc_order,
							 PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_COMPACT_DOC_ORDER_DOCID,
							 pgturbohybrid_multivector_quantized_inverted_compact_doc_order_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.multivector_quantized_inverted_query_codeword_kernel",
							 "Experimental query-codeword scoring kernel for quantized_inverted_experimental",
							 "auto uses the blocked scalar codebook path where available; scalar is a reference path; blocked batches query-token/codeword dot products while maintaining the top probe codewords. Final SQL ranking remains exact heap MaxSim.",
							 &pgturbohybrid_multivector_quantized_inverted_query_codeword_kernel,
							 PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_QUERY_CODEWORD_KERNEL_AUTO,
							 pgturbohybrid_multivector_quantized_inverted_query_codeword_kernel_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.multivector_quantized_inverted_precompact",
							 "Experimental precompact document gate for quantized_inverted_experimental",
							 "off preserves current compact scoring. centroid_maxsim_topk and centroid_maxsim_reservoir retain a bounded touched-document set using cheap query-codeword scores before full compact-code MaxSim admission; final SQL ranking remains exact heap MaxSim over retained candidates.",
							 &pgturbohybrid_multivector_quantized_inverted_precompact,
							 PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_PRECOMPACT_OFF,
							 pgturbohybrid_multivector_quantized_inverted_precompact_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_quantized_inverted_precompact_score_k",
							"Top cheap-score document count retained by quantized-inverted precompact",
							"Zero disables the score-count limit for centroid_maxsim_topk. The reservoir mode treats zero as no score-ranked contribution.",
							&pgturbohybrid_multivector_quantized_inverted_precompact_score_k,
							4096, 0, 10000000,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_quantized_inverted_precompact_coverage_k",
							"Top query-token coverage document count retained by quantized-inverted reservoir precompact",
							"Only used by centroid_maxsim_reservoir.",
							&pgturbohybrid_multivector_quantized_inverted_precompact_coverage_k,
							512, 0, 10000000,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_quantized_inverted_precompact_per_token_k",
							"Per-query-token reservoir width retained by quantized-inverted reservoir precompact",
							"Only used by centroid_maxsim_reservoir.",
							&pgturbohybrid_multivector_quantized_inverted_precompact_per_token_k,
							16, 0, 100000,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_quantized_inverted_compact_max_docs",
							"Maximum document count compact-scored after quantized-inverted reservoir precompact",
							"Zero disables the final reservoir union clamp.",
							&pgturbohybrid_multivector_quantized_inverted_compact_max_docs,
							6144, 0, 10000000,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.multivector_quantized_inverted_token_coverage",
							 "Experimental query-token coverage adjustment for quantized_inverted_experimental",
							 "off preserves the existing summed compact-code admission score. linear multiplies the approximate document score by matched query-token coverage before exact heap MaxSim rerank.",
							 &pgturbohybrid_multivector_quantized_inverted_token_coverage,
							 PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_TOKEN_COVERAGE_OFF,
							 pgturbohybrid_multivector_quantized_inverted_token_coverage_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_quantized_inverted_min_token_matches",
							"Minimum matched query tokens for quantized_inverted_experimental candidates",
							"Zero preserves existing behavior. Positive values drop approximate quantized-inverted candidates that matched fewer query tokens before bounded exact heap MaxSim rerank.",
							&pgturbohybrid_multivector_quantized_inverted_min_token_matches,
							0, 0, 1024,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.multivector_quantized_inverted_pruning",
							 "Experimental score-bound pruning for quantized_inverted_experimental",
							 "off preserves current behavior. score_bound_experimental skips compact full-doc codeword scoring when an approximate posting-score bound cannot enter the retained candidate heap; final SQL ranking remains exact heap MaxSim over retained candidates.",
							 &pgturbohybrid_multivector_quantized_inverted_pruning,
							 PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_PRUNING_OFF,
							 pgturbohybrid_multivector_quantized_inverted_pruning_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.multivector_branch_plan",
							 "Multivector branch planner mode",
							 "auto preserves existing branch execution, dense_only exposes a dense-only branch plan, and qdrant_like emits nested prefetch-style branch diagnostics for multivector/hybrid scans.",
							 &pgturbohybrid_multivector_branch_plan,
							 PGTURBOHYBRID_BRANCH_PLAN_AUTO,
							 pgturbohybrid_multivector_branch_plan_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_centroid_lite_max_postings_per_token",
							"Maximum centroid_lite postings read per query token",
							"Zero preserves the full experimental centroid_lite posting-list union; positive values cap each query-token posting list before exact MaxSim rerank so benchmarks can measure bounded admission tradeoffs.",
							&pgturbohybrid_multivector_centroid_lite_max_postings_per_token,
							0, 0, 10000000,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_centroid_lite_probe_centroids_per_token",
							"Maximum centroid_lite posting lists probed per query token",
							"One preserves existing deterministic centroid_lite behavior; larger values probe the best deterministic centroid lists for each query token before exact heap MaxSim rerank.",
							&pgturbohybrid_multivector_centroid_lite_probe_centroids_per_token,
							1, 1, 1024,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_centroid_lite_codeword_top_m",
							"Number of centroid_lite codeword posting lists assigned per document centroid",
							"One preserves existing persisted centroid_lite postings. Larger values are experimental and build denser deterministic posting lists before exact heap MaxSim rerank.",
							&pgturbohybrid_multivector_centroid_lite_codeword_top_m,
							1, 1, 16,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.multivector_centroid_lite_posting_selection",
							 "Experimental centroid_lite posting selection strategy",
							 "uniform_stride preserves diagnostic capped posting-list sampling; score_topk keeps the best payload-sorted postings per list; union_score unions probed centroid posting lists, scores documents by compact query-centroid MaxSim, and keeps the best document pool before exact heap MaxSim rerank.",
							 &pgturbohybrid_multivector_centroid_lite_posting_selection,
							 PGTURBOHYBRID_MULTIVECTOR_POSTING_SELECTION_UNIFORM_STRIDE,
							 pgturbohybrid_multivector_posting_selection_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.multivector_centroid_lite_candidate_scoring",
							 "Experimental centroid_lite candidate scoring strategy",
							 "posting_payload ranks touched documents by bounded posting payload scores; codeword_maxsim accumulates a PLAID-style approximate MaxSim over selected centroid codewords; doc_centroid_maxsim re-ranks touched documents by approximate MaxSim over persisted document centroids before exact heap MaxSim rerank.",
							 &pgturbohybrid_multivector_centroid_lite_candidate_scoring,
							 PGTURBOHYBRID_MULTIVECTOR_CENTROID_LITE_CANDIDATE_SCORING_POSTING_PAYLOAD,
							 pgturbohybrid_multivector_centroid_lite_candidate_scoring_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.multivector_centroid_lite_bitset_prefilter",
							 "Experimental centroid_lite scan-local bitset prefilter",
							 "off preserves current centroid_lite behavior; experimental builds a scan-local posting-union bitset for measurement only before exact MaxSim rerank.",
							 &pgturbohybrid_multivector_centroid_lite_bitset_prefilter,
							 PGTURBOHYBRID_MULTIVECTOR_CENTROID_LITE_BITSET_PREFILTER_OFF,
							 pgturbohybrid_multivector_centroid_lite_bitset_prefilter_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_centroid_lite_bitset_min_token_matches",
							"Minimum query-token posting-list matches for centroid_lite bitset prefilter",
							"One preserves the posting-union behavior. Larger values are experimental EMVB-style scan-local filtering before exact heap MaxSim rerank.",
							&pgturbohybrid_multivector_centroid_lite_bitset_min_token_matches,
							1, 1, 64,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomRealVariable("turbohybrid.multivector_centroid_lite_score_threshold",
							 "Minimum query-centroid score for centroid_lite posting-list expansion",
							 "-1.0 preserves existing behavior for normalized ColBERT vectors. Higher values skip weak centroid posting lists before exact heap MaxSim rerank.",
							 &pgturbohybrid_multivector_centroid_lite_score_threshold,
							 -1.0, -1.0, 1.0,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomRealVariable("turbohybrid.multivector_centroid_lite_score_drop_from_best",
							 "Maximum centroid_lite score drop from the best probed centroid per query token",
							 "-1.0 disables relative pruning. Non-negative values keep only probed centroid posting lists whose query-centroid score is at least best_score - drop before exact heap MaxSim rerank.",
							 &pgturbohybrid_multivector_centroid_lite_score_drop_from_best,
							 -1.0, -1.0, 2.0,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.multivector_centroid_lite_pruning",
							 "Experimental centroid_lite upper-bound pruning",
							 "off preserves current centroid_lite behavior; safe_upper_bound prunes only candidates with a proven safe query-centroid upper bound before exact MaxSim rerank.",
							 &pgturbohybrid_multivector_centroid_lite_pruning,
							 PGTURBOHYBRID_MULTIVECTOR_CENTROID_LITE_PRUNING_OFF,
							 pgturbohybrid_multivector_centroid_lite_pruning_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_quantized_inverted_max_postings_per_token",
							"Maximum quantized_inverted_experimental postings retained per query token",
							"Zero preserves the full experimental quantized-inverted posting-list union; positive values keep a bounded set of postings selected by the configured posting-selection strategy before exact heap MaxSim rerank.",
							&pgturbohybrid_multivector_quantized_inverted_max_postings_per_token,
							0, 0, 10000000,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_quantized_inverted_probe_codewords_per_token",
							"Maximum quantized_inverted_experimental posting lists probed per query token",
							"One preserves existing codeword assignment behavior; larger values probe the best codeword lists for each query token before exact heap MaxSim rerank.",
							&pgturbohybrid_multivector_quantized_inverted_probe_codewords_per_token,
							1, 1, 1024,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.multivector_quantized_inverted_posting_selection",
							 "Experimental quantized_inverted_experimental posting selection strategy",
							 "score_topk keeps the best capped postings by compact/code or exact token score before exact heap MaxSim rerank; uniform_stride is diagnostic compatibility sampling.",
							 &pgturbohybrid_multivector_quantized_inverted_posting_selection,
							 PGTURBOHYBRID_MULTIVECTOR_POSTING_SELECTION_SCORE_TOPK,
							 pgturbohybrid_multivector_posting_selection_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_doc_graph_entry_sample_count",
							"Document-node graph entry samples scored before traversal",
							"Zero preserves the compiled default entry-sample cap. Positive values widen the deterministic proxy-entry sample used before document-node graph traversal without changing index storage.",
							&pgturbohybrid_multivector_doc_graph_entry_sample_count,
							0, 0, 1000000,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.multivector_plain_fallback",
							 "Exact/plain heap fallback mode for multivector MaxSim scans",
							 "auto uses exact heap MaxSim for small or near-exhaustive scans; off keeps the token candidate path; force always uses exact heap MaxSim.",
							 &pgturbohybrid_multivector_plain_fallback,
							 PGTURBOHYBRID_MULTIVECTOR_PLAIN_FALLBACK_AUTO,
							 pgturbohybrid_multivector_plain_fallback_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_plain_fallback_max_docs",
							"Maximum estimated documents for automatic exact/plain multivector fallback",
							"Auto fallback uses exact heap MaxSim when the estimated live document count is at or below this threshold.",
							&pgturbohybrid_multivector_plain_fallback_max_docs,
							1000, 0, 10000000,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomRealVariable("turbohybrid.multivector_plain_fallback_candidate_fraction",
							 "Near-exhaustive candidate fraction for automatic exact/plain multivector fallback",
							 "Auto fallback uses exact heap MaxSim when doc_candidate_k or exact_rerank_k reaches this fraction of the estimated document count.",
							 &pgturbohybrid_multivector_plain_fallback_candidate_fraction,
							 0.5, 0.0, 1.0,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.multivector_candidate_reservoirs",
							 "Candidate reservoir selection mode for multivector MaxSim scans",
							 "off keeps the single approximate-score top-K; conservative and balanced retain a bounded union of score, coverage, mean-similarity, and per-token document reservoirs before exact rerank.",
							 &pgturbohybrid_multivector_candidate_reservoirs,
							 PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_RESERVOIRS_CONSERVATIVE,
							 pgturbohybrid_multivector_candidate_reservoirs_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_per_token_doc_reservoir_k",
							"Per-query-token document reservoir size for multivector candidate retention",
							"When multivector_candidate_reservoirs is enabled, each query token can contribute up to this many document candidates to the bounded union.",
							&pgturbohybrid_multivector_per_token_doc_reservoir_k,
							1, 0, 100000,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.multivector_coverage_reservoir_k",
							"Coverage reservoir size for multivector candidate retention",
							"When multivector_candidate_reservoirs is enabled, this many high query-token-coverage documents can be added to the bounded union.",
							&pgturbohybrid_multivector_coverage_reservoir_k,
							10, 0, 100000,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.multivector_bm25_candidate_injection",
							 "BM25 candidate injection mode for multivector MaxSim rerank",
							 "off disables injection; hybrid_only injects BM25 candidates for hybrid multivector text queries; dense_with_text also allows text-backed dense-only MaxSim reranking.",
							 &pgturbohybrid_multivector_bm25_candidate_injection,
							 PGTURBOHYBRID_MULTIVECTOR_BM25_CANDIDATE_INJECTION_OFF,
							 pgturbohybrid_multivector_bm25_candidate_injection_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.multivector_sparse_candidate_source",
							 "Sparse branch source for multivector candidate injection",
							 "off disables the explicit sparse branch; bm25 uses lexical postings; learned_sparse reuses the sparse postings path for exported learned sparse vectors and exact-MaxSim reranks admitted documents.",
							 &pgturbohybrid_multivector_sparse_candidate_source,
							 PGTURBOHYBRID_MULTIVECTOR_SPARSE_CANDIDATE_SOURCE_OFF,
							 pgturbohybrid_multivector_sparse_candidate_source_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.max_union_candidates", "Maximum candidates retained while fusing dense and BM25 branches",
							NULL, &pgturbohybrid_max_union_candidates,
							100000, 0,
							PGTURBOHYBRID_MAX_UNION_CANDIDATES,
							PGC_USERSET, 0, PgturbohybridCheckMaxUnionCandidates,
							NULL, NULL);
	DefineCustomBoolVariable("turbohybrid.simd", "Enable SIMD kernels where supported by the host CPU",
							 NULL, &pgturbohybrid_simd,
							 true, PGC_USERSET, 0, NULL, PgturbohybridAssignSimd, NULL);
	DefineCustomBoolVariable("turbohybrid.dense_graph_avx512vnni",
							 "(developer/benchmark) Allow the AVX-512 VNNI query-split dense scorer",
							 "Diagnostic knob: turn off to fall back to AVX-VNNI/AVX2 query split for kernel parity testing.",
							 &pgturbohybrid_dense_graph_avx512vnni,
							 true, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("turbohybrid.dense_graph_avxvnni",
							 "(developer/benchmark) Allow the AVX-VNNI query-split dense scorer",
							 "Diagnostic knob: turn off (with avx512vnni off) to force the AVX2 query-split scorer for parity testing.",
							 &pgturbohybrid_dense_graph_avxvnni,
							 true, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.dense_query_split_impl",
							 "(developer/benchmark) 4-bit dense query-split representation",
							 "signed uses the signed-codebook split; unsigned uses the x86 unsigned-codebook maddubs/VPDPBUSD split; auto picks unsigned on x86 when available.",
							 &pgturbohybrid_dense_query_split_impl,
							 PGTURBOHYBRID_QUERY_SPLIT_IMPL_AUTO,
							 pgturbohybrid_dense_query_split_impl_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.dense_u8_split",
							 "(developer/benchmark) Use the unsigned-codebook (u8) 4-bit split dense scorer",
							 "on forces the u8 maddubs/VPDPBUSD split whenever its hard requirements hold (4-bit, dim>=64, mode!=L1, AVX2+); off disables it (signed split or scalar/LUT); auto defers to dense_query_split_impl. For controlled benchmarking.",
							 &pgturbohybrid_dense_u8_split,
							 PGTURBOHYBRID_U8_SPLIT_AUTO,
							 pgturbohybrid_dense_u8_split_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("turbohybrid.dense_u8_batch_x4",
							 "(developer/benchmark) Use the 4-candidate (x4) u8 split batch kernel instead of four single-node calls",
							 "On (default) scores a batch's four scattered codes in one kernel pass: the query [low|high] data is loaded once instead of four times, and the four scattered code loads are issued together so their memory latency overlaps (memory-level parallelism). Kernel ns/code on amd64 (Ice Lake VNNI): ~tied with single-node when codes are cache-resident (compute-bound), ~1.4x faster when they stream from RAM (memory-bound, the regime that dominates the 1M index). Off forces four single-node passes for parity/benchmarking.",
							 &pgturbohybrid_dense_u8_batch_x4,
							 true, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.dense_rescore_band",
							 "Exact f32 dense rescore policy for native graph scans",
							 "auto rescores only when the budget/quality policy or exact storage requires it (latency-profile exact-free scans resolve to 0); off never exact-rescores; exact rescores the full candidate band; limited caps it. Exact-free (code-only) indexes never exact-rescore regardless. For controlled benchmarking.",
							 &pgturbohybrid_dense_rescore_band_policy,
							 PGTURBOHYBRID_RESCORE_BAND_POLICY_AUTO,
							 pgturbohybrid_dense_rescore_band_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.dense_heap_rescore",
							 "Heap-backed exact dense rescore policy for exact-free native graph scans",
							 "off keeps the low-latency code-only path; topk fetches heap tuples for the requested top-k band and computes exact vector distance; band fetches and exact-rescores the full final candidate band. This trades heap I/O and latency for precision without exact_storage.",
							 &pgturbohybrid_dense_heap_rescore,
							 PGTURBOHYBRID_DENSE_HEAP_RESCORE_OFF,
							 pgturbohybrid_dense_heap_rescore_options,
							 PGC_USERSET, 0,
							 PgturbohybridCheckDenseHeapRescore,
							 NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.dense_residual_rerank_mode",
							 "Residual-sketch dense rerank mode for exact-free native graph scans",
							 "off ignores stored residual sketches; fixed preserves the original fixed 0.03 sketch-similarity adjustment; calibrated scales and clamps the adjustment using the observed final-band distance spread.",
							 &pgturbohybrid_dense_residual_rerank_mode,
							 PGTURBOHYBRID_DENSE_RESIDUAL_RERANK_CALIBRATED,
							 pgturbohybrid_dense_residual_rerank_mode_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomRealVariable("turbohybrid.dense_residual_rerank_weight",
							 "Residual-sketch calibrated rerank weight",
							 "-1 uses the built-in auto weight; non-negative values multiply the final-band distance spread before clamping.",
							 &pgturbohybrid_dense_residual_rerank_weight,
							 -1.0, -1.0, 10.0, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomRealVariable("turbohybrid.dense_residual_rerank_max_adjust_ratio",
							 "Maximum calibrated residual-rerank adjustment as a fraction of final-band distance spread",
							 NULL,
							 &pgturbohybrid_dense_residual_rerank_max_adjust_ratio,
							 0.15, 0.0, 1.0, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.dense_residual_rerank_band_multiplier",
							"Experimental: residual rerank band width as a multiple of the final-k target",
							"Benchmark-only knob. The residual rerank band is min(count, max(final_k_target * this, 20)). The default 2 preserves current behavior; a wider band lets residual rerank reach neighbours that fall outside the narrow default band.",
							&pgturbohybrid_dense_residual_rerank_band_multiplier,
							2, 1, 16, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.native_cache_policy",
							 "Native dense graph cache policy",
							 "Compatibility alias for turbohybrid.native_cache_scope.",
							 &pgturbohybrid_native_cache_policy,
							 PGTURBOHYBRID_NATIVE_CACHE_POLICY_AUTO,
							 pgturbohybrid_native_cache_policy_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.native_cache_scope",
							 "Native dense graph cache scope",
							 "auto uses the mmap-backed shared immutable cache on supported platforms when the native working set fits native_cache_max_mb; per_backend explicitly uses the backend-local cache; shared forces the shared cache; off uses scan-local page loading through shared buffers.",
							 &pgturbohybrid_native_cache_policy,
							 PGTURBOHYBRID_NATIVE_CACHE_POLICY_AUTO,
							 pgturbohybrid_native_cache_policy_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.native_cache_max_mb",
							"Max size (MB) of the native dense scan cache",
							"An index whose resident code/node/adjacency working set fits under this cap is fully loaded into the selected native cache scope, so warm scans read 0 code pages. Larger indexes fall back to per-scan page loading. This is per-backend only when native_cache_scope=per_backend. Default 2048 (covers ~1M-row indexes at common dimensions; under the default native_cache_scope=auto the working set is held once in the mmap-backed shared cache, not per backend).",
							&pgturbohybrid_native_cache_max_mb,
							2048, 0, 1048576,
							PGC_USERSET, GUC_UNIT_MB, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.native_cache_warn_mb",
							"Warn when building a large per-backend native dense scan cache",
							"Emits DEBUG1 when a per-backend native graph cache build exceeds this resident-size threshold. The warning is non-fatal and reminds users that per_backend memory is duplicated by each active PostgreSQL backend. Set to 0 to disable. Default 512.",
							&pgturbohybrid_native_cache_warn_mb,
							512, 0, 1048576,
							PGC_USERSET, GUC_UNIT_MB, NULL, NULL, NULL);
	DefineCustomStringVariable("turbohybrid.native_build_workers",
							   "Parallel worker count for native dense graph build scan, encoding, and edge construction",
							   "Default 2 requests parallel native builds; auto uses PostgreSQL's parallel CREATE INDEX worker choice; 0 disables native parallel build; 1, 2, 4, or 8 requests that many workers.",
							   &pgturbohybrid_native_build_workers,
							   "2", PGC_USERSET, 0,
							   PgturbohybridNativeBuildWorkersCheck,
							   NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.native_parallel_edge_build",
							 "Parallelize native dense graph edge construction",
							 "auto parallelizes code-only native graph edge construction when workers are available; on requires the parallel edge path and errors if unsupported; off preserves serial edge construction.",
							 &pgturbohybrid_native_parallel_edge_build,
							 PGTURBOHYBRID_NATIVE_PARALLEL_EDGE_BUILD_AUTO,
							 pgturbohybrid_native_parallel_edge_build_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.native_segment_budget",
							 "Native dense graph segment search budget scaling",
							 "auto scales search budget for segmented native graphs (sqrt by default, linear for quality/exact-build indexes); off preserves the raw search budget; sqrt and linear force explicit scaling modes for benchmarks.",
							 &pgturbohybrid_native_segment_budget,
							 PGTURBOHYBRID_NATIVE_SEGMENT_BUDGET_AUTO,
							 pgturbohybrid_native_segment_budget_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.dense_build_distance",
							 "Dense graph build distance source",
							 "auto uses exact f32 build distances for low-dimensional balanced/matched_recall/quality builds and compact code distances for latency/high-dimensional builds; code always builds graph edges in the quantized-code domain; exact uses original vectors during build without implying exact_storage.",
							 &pgturbohybrid_dense_build_distance,
							 PGTURBOHYBRID_DENSE_BUILD_DISTANCE_AUTO,
							 pgturbohybrid_dense_build_distance_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("turbohybrid.dense_build_exact_distances",
							 "(developer/benchmark) Use exact f32 vector distances while building dense graph edges",
							 "Compatibility override for turbohybrid.dense_build_distance. When explicitly set, on forces exact and off forces code.",
							 &pgturbohybrid_dense_build_exact_distances,
							 false, PGC_USERSET, 0,
							 PgturbohybridCheckDenseBuildExactDistances,
							 NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.dense_build_neighbor_select",
							 "Dense graph neighbor selection during native graph builds",
							 "fast keeps the code-only simple edge selector; heuristic uses the diversified HNSW-style selector; auto uses fast for multivector document-node proxy builds, heuristic for low-dimensional and balanced/matched_recall/quality dense builds, and fast for high-dimensional latency builds.",
							 &pgturbohybrid_dense_build_neighbor_select,
							 PGTURBOHYBRID_DENSE_BUILD_NEIGHBOR_SELECT_AUTO,
							 pgturbohybrid_dense_build_neighbor_select_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.dense_adaptive_widening",
							 "Adaptive dense graph widening mode",
							 "Valid values are off, auto, and on. Non-off modes may run one bounded second graph pass.",
							 &pgturbohybrid_dense_adaptive_widening,
							 PGTURBOHYBRID_DENSE_ADAPTIVE_WIDENING_OFF,
							 pgturbohybrid_dense_adaptive_widening_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomRealVariable("turbohybrid.dense_adaptive_widening_multiplier",
							 "Multiplier for one adaptive dense graph widening pass",
							 NULL, &pgturbohybrid_dense_adaptive_widening_multiplier,
							 2.0, 1.0, 16.0, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomRealVariable("turbohybrid.dense_adaptive_widening_max_multiplier",
							 "Maximum multiplier cap for adaptive dense graph widening",
							 NULL, &pgturbohybrid_dense_adaptive_widening_max_multiplier,
							 4.0, 1.0, 32.0, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomRealVariable("turbohybrid.dense_adaptive_min_gap",
							 "Normalized score-gap threshold for automatic dense graph widening; 0 uses the built-in threshold",
							 NULL, &pgturbohybrid_dense_adaptive_min_gap,
							 0.0, 0.0, 1.0, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.dense_uncertainty_retry",
							 "Bounded dense graph uncertainty retry mode",
							 "Valid values are off, auto, and on. Non-off modes may run one additional wider graph traversal when the first candidate band looks uncertain.",
							 &pgturbohybrid_dense_uncertainty_retry,
							 PGTURBOHYBRID_DENSE_UNCERTAINTY_RETRY_OFF,
							 pgturbohybrid_dense_uncertainty_retry_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.dense_uncertainty_retry_max_passes",
							"Maximum bounded retry passes for dense graph uncertainty retry",
							"1 permits one bounded second traversal; 2 is accepted for future multi-pass experiments.",
							&pgturbohybrid_dense_uncertainty_retry_max_passes,
							1, 1, 2, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomRealVariable("turbohybrid.dense_uncertainty_retry_multiplier",
							 "Multiplier for dense uncertainty retry candidate target and ef_search",
							 NULL, &pgturbohybrid_dense_uncertainty_retry_multiplier,
							 1.5, 1.0, 8.0, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomRealVariable("turbohybrid.dense_uncertainty_min_gap",
							 "Normalized score-gap threshold for automatic dense uncertainty retry",
							 NULL, &pgturbohybrid_dense_uncertainty_min_gap,
							 0.03, 0.0, 1.0, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("turbohybrid.warn_linear_fallback",
							 "Warn when unmapped heap filters make dense graph scans near-linear",
							 "When enabled, emits DEBUG1 if an unmapped heap filter widens native dense candidate collection to a large fraction of the index.",
							 &pgturbohybrid_warn_linear_fallback,
							 true, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomRealVariable("turbohybrid.linear_fallback_notice_threshold_ratio",
							 "Dense graph linear-fallback warning threshold",
							 "Emit the unmapped-filter dense fallback warning when resultTarget / nodeCount is at least this ratio.",
							 &pgturbohybrid_linear_fallback_notice_threshold_ratio,
							 0.25, 0.0, 1.0, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.dense_local_expansion",
							 "Experimental bounded one-hop dense graph expansion mode",
							 "Valid values are off, auto, and on. Non-off modes score neighbors of top approximate candidates.",
							 &pgturbohybrid_dense_local_expansion,
							 PGTURBOHYBRID_DENSE_LOCAL_EXPANSION_OFF,
							 pgturbohybrid_dense_local_expansion_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.dense_local_expansion_topn",
							"Number of top approximate dense candidates to expand locally",
							NULL, &pgturbohybrid_dense_local_expansion_topn,
							8, 1, 64, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.dense_local_expansion_max_neighbors",
							"Maximum level-0 neighbors to score during local dense expansion",
							NULL, &pgturbohybrid_dense_local_expansion_max_neighbors,
							256, 1, 4096, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.payload_entry_seeding",
							 "Use INCLUDE int4 payload refs as dense graph entry seeds for payload-filtered scans",
							 "auto/on sample existing payload refs for the active int4 equality filter and offer those nodes as graph entry points; off preserves global-only entry behavior.",
							 &pgturbohybrid_payload_entry_seeding,
							 PGTURBOHYBRID_PAYLOAD_ENTRY_SEEDING_AUTO,
							 pgturbohybrid_payload_entry_seeding_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.payload_entry_seed_count",
							"Maximum payload-local graph entry seeds sampled for a payload-filtered dense scan",
							NULL, &pgturbohybrid_payload_entry_seed_count,
							PGTURBOHYBRID_DEFAULT_PAYLOAD_ENTRY_SEED_COUNT, 1,
							PGTURBOHYBRID_MAX_PAYLOAD_ENTRY_SEED_COUNT,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.final_diversity",
							 "Final result diversity mode",
							 "off preserves relevance order; group_payload greedily diversifies the final top-k using an existing int4 INCLUDE payload slot without fetching heap rows.",
							 &pgturbohybrid_final_diversity,
							 PGTURBOHYBRID_FINAL_DIVERSITY_OFF,
							 pgturbohybrid_final_diversity_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.final_diversity_payload_slot",
							"INCLUDE int4 payload slot used for final group diversity",
							"-1 disables group-payload diversity until a slot is configured. Slot 0 is the first int4 INCLUDE payload stored by the turbohybrid index.",
							&pgturbohybrid_final_diversity_payload_slot,
							-1, -1, PGTURBOHYBRID_GRAPH_MAX_PAYLOADS - 1,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomRealVariable("turbohybrid.final_diversity_lambda",
							 "Final diversity relevance weight",
							 "1.0 preserves pure relevance scoring; lower values penalize duplicate payload groups more strongly.",
							 &pgturbohybrid_final_diversity_lambda,
							 PGTURBOHYBRID_FINAL_DIVERSITY_DEFAULT_LAMBDA,
							 0.0, 1.0, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.final_diversity_pool_multiplier",
							"Final diversity candidate pool multiplier",
							"The bounded diversity pool is final_k multiplied by this value and capped by the merged candidate count.",
							&pgturbohybrid_final_diversity_pool_multiplier,
							PGTURBOHYBRID_FINAL_DIVERSITY_DEFAULT_POOL_MULTIPLIER,
							1, PGTURBOHYBRID_FINAL_DIVERSITY_MAX_POOL_MULTIPLIER,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.bm25_strategy", "BM25 candidate generation strategy",
							 "Valid values are auto, impact, impact_or, wand, daat_simd, and daat_hash.",
							 &pgturbohybrid_bm25_strategy,
							 PGTURBOHYBRID_BM25_STRATEGY_AUTO,
							 pgturbohybrid_bm25_strategy_options,
							 PGC_USERSET, 0, PgturbohybridCheckBm25Strategy, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.bm25_impact_or_mode", "Impact-head OR behavior for BM25 OR queries",
							 "Valid values are off, exact_only, and approx.",
							 &pgturbohybrid_bm25_impact_or_mode,
							 PGTURBOHYBRID_BM25_IMPACT_OR_MODE_EXACT_ONLY,
							 pgturbohybrid_bm25_impact_or_mode_options,
							 PGC_USERSET, 0, PgturbohybridCheckBm25ImpactOrMode, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.bm25_hot_postings_cache_mb", "Hot postings cache size for repeated common BM25 terms",
							NULL, &pgturbohybrid_bm25_hot_postings_cache_mb,
							0, 0, PGTURBOHYBRID_MAX_HOT_POSTINGS_CACHE_MB,
							PGC_USERSET, 0,
							PgturbohybridCheckBm25HotPostingsCacheMb, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.bm25_hot_postings_cache_min_df", "Minimum BM25 document frequency for hot postings cache entries",
							NULL, &pgturbohybrid_bm25_hot_postings_cache_min_df,
							1024, 1, INT_MAX, PGC_USERSET, 0,
							PgturbohybridCheckBm25HotPostingsCacheMinDf, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.bm25_common_term_fallback_min_postings",
							"Minimum decoded postings count for flagging common-term BM25 fallback stats",
							NULL,
							&pgturbohybrid_bm25_common_term_fallback_min_postings,
							100000, 0, INT_MAX, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.bm25_hybrid_bound", "Hybrid RRF bound mode for BM25 candidate generation",
							 "Valid values are off, safe, and approx.",
							 &pgturbohybrid_bm25_hybrid_bound,
							 PGTURBOHYBRID_BM25_HYBRID_BOUND_SAFE,
							 pgturbohybrid_bm25_hybrid_bound_options,
							 PGC_USERSET, 0, PgturbohybridCheckBm25HybridBound, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.bm25_accumulator_mode", "BM25 accumulator implementation",
							 "Valid values are auto, hash, and node_generation_arrays.",
							 &pgturbohybrid_bm25_accumulator_mode,
							 PGTURBOHYBRID_BM25_ACCUMULATOR_AUTO,
							 pgturbohybrid_bm25_accumulator_mode_options,
							 PGC_USERSET, 0, PgturbohybridCheckBm25AccumulatorMode, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.bm25_heap_tsvector_rerank",
							 "Heap-backed tsvector rerank policy for BM25 and hybrid scans",
							 "Valid values are off, topk, band, and auto. The default off preserves indexed BM25 scoring; auto applies a bounded candidate-band rerank to phrase/proximity tsqueries.",
							 &pgturbohybrid_bm25_heap_tsvector_rerank,
							 PGTURBOHYBRID_BM25_HEAP_TSVECTOR_RERANK_OFF,
							 pgturbohybrid_bm25_heap_tsvector_rerank_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.bm25_heap_tsvector_rerank_multiplier",
							"Candidate-band multiplier for BM25 heap tsvector rerank",
							"Only applies to band and auto modes; the fetch band is final_k multiplied by this value and capped by bm25_k.",
							&pgturbohybrid_bm25_heap_tsvector_rerank_multiplier,
							4, 1, 64, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomRealVariable("turbohybrid.bm25_heap_tsvector_rerank_weight",
							 "Maximum bounded BM25 score adjustment from heap tsvector ranking",
							 "The adjustment is weight * rank / (rank + 1), so it is capped by this value.",
							 &pgturbohybrid_bm25_heap_tsvector_rerank_weight,
							 0.10, 0.0, 1.0, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.sparse_rerank",
							 "Exact f32 rerank policy for quantized sparse candidate bands",
							 "Valid values are off, topk, band, and auto. auto reranks quantized (q8/q16) sparse indexes and is a no-op for exact f32 indexes.",
							 &pgturbohybrid_sparse_rerank,
							 PGTURBOHYBRID_SPARSE_RERANK_AUTO,
							 pgturbohybrid_sparse_rerank_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("turbohybrid.sparse_rerank_k",
							"Number of top sparse candidates to exact-rerank (0 = follow final_k)",
							"Applies to topk and auto modes; band mode reranks the whole returned candidate set.",
							&pgturbohybrid_sparse_rerank_k,
							0, 0, 1000000, PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.hybrid_budget_policy",
							 "Hybrid branch budget policy",
							 "Valid values are fixed and adaptive. Adaptive is approximate and may change dense_k, bm25_k, final_k, and rrf_k for defaulted hybrid queries.",
							 &pgturbohybrid_hybrid_budget_policy,
							 PGTURBOHYBRID_HYBRID_BUDGET_FIXED,
							 pgturbohybrid_hybrid_budget_policy_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomEnumVariable("turbohybrid.profile", "Default retrieval profile for pgturbohybrid",
							 "Valid values are latency, balanced, matched_recall, high_recall, quality, and debug.",
							 &pgturbohybrid_profile,
							 PGTURBOHYBRID_PROFILE_LATENCY,
							 pgturbohybrid_profile_options,
							 PGC_USERSET, 0, NULL, PgturbohybridAssignProfile, NULL);
	PgturbohybridApplyProfileDefaults();
	MarkGUCPrefixReserved("turbohybrid");
}
