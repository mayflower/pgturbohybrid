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
#include "storage/lmgr.h"
#include "storage/lwlock.h"
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
#include "pgturbohybrid_sparse.h"
#include "pgturbohybrid_quant.h"
#include "pgturbohybrid_quant_score.h"
#include "pgturbohybrid_am.h"
#include "pgturbohybrid_bm25.h"

#define PGTURBOHYBRID_DEFAULT_BM25_K1 1.2
#define PGTURBOHYBRID_DEFAULT_BM25_B 0.75
#define PGTURBOHYBRID_DEFAULT_DENSE_K 100
#define PGTURBOHYBRID_DEFAULT_BM25_K 100
#define PGTURBOHYBRID_DEFAULT_RRF_K 60
#define PGTURBOHYBRID_FUSION_GENERATION_ARRAY_MAX_BYTES (16 * 1024 * 1024)

#if PG_VERSION_NUM < 150000
#define MarkGUCPrefixReserved(x) EmitWarningsOnPlaceholders(x)
#endif

typedef enum PgturbohybridBm25HybridBoundMode
{
	PGTURBOHYBRID_BM25_HYBRID_BOUND_OFF,
	PGTURBOHYBRID_BM25_HYBRID_BOUND_SAFE,
	PGTURBOHYBRID_BM25_HYBRID_BOUND_APPROX
} PgturbohybridBm25HybridBoundMode;

typedef enum PgturbohybridBm25HeapTSVectorRerankMode
{
	PGTURBOHYBRID_BM25_HEAP_TSVECTOR_RERANK_OFF,
	PGTURBOHYBRID_BM25_HEAP_TSVECTOR_RERANK_TOPK,
	PGTURBOHYBRID_BM25_HEAP_TSVECTOR_RERANK_BAND,
	PGTURBOHYBRID_BM25_HEAP_TSVECTOR_RERANK_AUTO
} PgturbohybridBm25HeapTSVectorRerankMode;

static relopt_kind pgturbohybrid_relopt_kind;
static List *pgturbohybrid_plannedstmt_stack = NIL;
static PlannedStmt *pgturbohybrid_current_plannedstmt = NULL;
static bool pgturbohybrid_am_init_done = false;

int			pgturbohybrid_profile = PGTURBOHYBRID_PROFILE_LATENCY;
bool		pgturbohybrid_enable_wand = true;
bool		pgturbohybrid_bm25_tenant_stats = true;
int			pgturbohybrid_max_union_candidates = 100000;
int			pgturbohybrid_default_dense_k = PGTURBOHYBRID_DEFAULT_DENSE_K;
int			pgturbohybrid_default_bm25_k = PGTURBOHYBRID_DEFAULT_BM25_K;
int			pgturbohybrid_default_rrf_k = PGTURBOHYBRID_DEFAULT_RRF_K;
int			pgturbohybrid_sparse_delta_compaction_threshold = 1000;
int			pgturbohybrid_sparse_hot_postings_cache_mb = 16;
int			pgturbohybrid_sparse_hot_postings_cache_min_df = 256;
uint64		pgturbohybrid_guc_generation = 1;
int			pgturbohybrid_last_final_k_requested = 0;
int			pgturbohybrid_last_final_k_effective = 0;
int			pgturbohybrid_last_sql_limit = 0;
bool		pgturbohybrid_last_final_k_inferred = false;
static bool pgturbohybrid_simd = true;
int			pgturbohybrid_force_fusion = 0;
int			pgturbohybrid_fusion_hash_threshold = 128;
static bool pgturbohybrid_fast_weighted_score_bound_pruning = true;
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

typedef enum PgturbohybridDbsfRobustMode
{
	PGTURBOHYBRID_DBSF_ROBUST_OFF,
	PGTURBOHYBRID_DBSF_ROBUST_MAD
}			PgturbohybridDbsfRobustMode;

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

/* Sparse-vector reloptions */
#define PGTURBOHYBRID_SPARSE_QUANT_MODE_F32 0
#define PGTURBOHYBRID_SPARSE_QUANT_MODE_PER_TERM_LINEAR 1
static relopt_enum_elt_def pgturbohybrid_sparse_quant_mode_relopt_options[] = {
	{"f32", PGTURBOHYBRID_SPARSE_QUANT_MODE_F32},
	{"per_term_linear", PGTURBOHYBRID_SPARSE_QUANT_MODE_PER_TERM_LINEAR},
	{NULL, 0}
};

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

#define PGTURBOHYBRID_SPARSE_EXACT_STORAGE_OFF 0
#define PGTURBOHYBRID_SPARSE_EXACT_STORAGE_SIDECAR 1
static relopt_enum_elt_def pgturbohybrid_sparse_exact_storage_relopt_options[] = {
	{"off", PGTURBOHYBRID_SPARSE_EXACT_STORAGE_OFF},
	{"sidecar", PGTURBOHYBRID_SPARSE_EXACT_STORAGE_SIDECAR},
	{NULL, 0}
};

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
	/* Sparse branch stats */
	bool		sparseBranchAvailable;
	bool		sparseBranchUsed;
	uint32		sparseTerms;
	uint32		sparseResolvedTerms;
	uint64		sparsePostingsTouched;
	uint64		sparseCandidatesScored;
	uint32		sparseCandidates;
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

bool
PgturbohybridIndexHasLexical(Relation index)
{
	return index != NULL && index->rd_index != NULL &&
		index->rd_index->indnkeyatts > PGTURBOHYBRID_LEXICAL_KEY_INDEX;
}

bool
PgturbohybridIndexGetLexicalDatum(Relation index, Datum *values, bool *isnull,
							 Datum *lexicalValue)
{
	if (!PgturbohybridIndexHasLexical(index) || values == NULL || isnull == NULL)
		return false;
	if (isnull[PGTURBOHYBRID_LEXICAL_KEY_INDEX])
		return false;

	*lexicalValue = values[PGTURBOHYBRID_LEXICAL_KEY_INDEX];
	return true;
}

static bool
PgturbohybridIndexInfoHasLexical(Relation index, IndexInfo *indexInfo)
{
	if (indexInfo != NULL)
		return indexInfo->ii_NumIndexKeyAttrs > PGTURBOHYBRID_LEXICAL_KEY_INDEX;

	return PgturbohybridIndexHasLexical(index);
}

static AttrNumber
PgturbohybridPathLexicalAttno(IndexPath *path)
{
	if (path == NULL || path->indexinfo == NULL ||
		path->indexinfo->nkeycolumns <= PGTURBOHYBRID_LEXICAL_KEY_INDEX)
		return InvalidAttrNumber;

	return path->indexinfo->indexkeys[PGTURBOHYBRID_LEXICAL_KEY_INDEX];
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
	 * bm25) are discovered by type and may follow in any order.
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
		(item->hasBm25 ? 1.0 / ((double) query->rrfK + item->bm25Rank) : 0.0);
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

	lexicalAttno =
		scan->indexRelation->rd_index->indkey.values[PGTURBOHYBRID_LEXICAL_KEY_INDEX];
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

	if (state->collectDone)
		return;

	INSTR_TIME_SET_CURRENT(totalStart);
	memset(&denseStats, 0, sizeof(denseStats));
	memset(&multivectorStats, 0, sizeof(multivectorStats));
	memset(&bm25Stats, 0, sizeof(bm25Stats));
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
	 * Sparse-as-sole-ORDER-BY: the query targets only the sparse column.
	 * Resolve it against the native sparse inverted index and return
	 * candidates ranked by exact inner product, bypassing the dense/bm25
	 * fusion machinery.
	 */
	if (PgturbohybridQueryHasSparse(scanQuery) &&
		!hasVectorQuery && !hasMultivectorQuery && !hasTextQuery)
	{
		PgturbohybridSparseCandidate *cands = NULL;
		PgturbohybridSparseScanStats sstats;
		PgturbohybridResult *results;
		int			finalTarget;
		int			candidateK;
		int			n;

		memset(&sstats, 0, sizeof(sstats));
		finalTarget = (int) PgturbohybridEffectiveFinalK(originalQuery, autoBudgetLimit);
		if (finalTarget < 1)
			finalTarget = 1;
		candidateK = scanQuery->sparseK > 0 ? scanQuery->sparseK : finalTarget;
		if (candidateK < finalTarget)
			candidateK = finalTarget;

		n = PgturbohybridSparseCollectCandidates(scan->indexRelation, scanQuery,
												 candidateK, pgturbohybrid_simd,
												 true, &cands,
												 so->tmpCtx, &sstats);

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

		strlcpy(lastStats.indexShape, "sparse", sizeof(lastStats.indexShape));
		lastStats.sparseBranchAvailable = sstats.branchAvailable;
		lastStats.sparseBranchUsed = sstats.branchUsed;
		lastStats.sparseTerms = sstats.terms;
		lastStats.sparseResolvedTerms = sstats.resolvedTerms;
		lastStats.sparsePostingsTouched = sstats.postingsTouched;
		lastStats.sparseCandidatesScored = sstats.candidatesScored;
		lastStats.sparseCandidates = n;
		lastStats.elapsedUs = PgturbohybridElapsedUs(totalStart);
		pgturbohybrid_last_scan_state = lastStats;
		state->query = originalQuery;
		MemoryContextSwitchTo(oldCtx);
		return;
	}

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

	INSTR_TIME_SET_CURRENT(phaseStart);
	fusionCandidatesSeen = denseCount + multivectorCount + bm25Count;
	lastStats.fusionCandidatesSeen = fusionCandidatesSeen;
	effectiveFusion = pgturbohybrid_force_fusion != 0 ?
		pgturbohybrid_force_fusion : scanQuery->fusion;
	useHashTopN = pgturbohybrid_fusion_hash_threshold >= 0 &&
		fusionCandidatesSeen >= (uint32) pgturbohybrid_fusion_hash_threshold;
	if (!useDocumentFusionKey &&
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
				i++;
			}

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
	PgturbohybridIndexKeyMap map;

	PgturbohybridValidateIndex(index, indexInfo);
	result = pgturbohybridbuild(heap, index, indexInfo);

	if (PgturbohybridIndexInfoHasLexical(index, indexInfo))
		PgturbohybridBm25BuildCollect(heap, index, indexInfo);

	PgturbohybridBuildIndexKeyMap(index, indexInfo, &map);
	/*
	 * The sparse inverted index builds here.  PgturbohybridSparseBuildCollect
	 * anchors its meta chain in the graph metapage itself
	 * (tqSparseMetaStartBlkno), so no pre-initialization is needed.  Build
	 * failures degrade to a warning: the dense/bm25 halves stay usable and the
	 * sparse scan reports branch-unavailable.  Regression coverage:
	 * pgturbohybrid_sparse_scan (build + sole-ORDER-BY scan + planner).
	 */
	if (map.hasSparse)
	{
		MemoryContext oldcontext = CurrentMemoryContext;

		PG_TRY();
		{
			PgturbohybridSparseBuildCollect(heap, index, indexInfo);
		}
		PG_CATCH();
		{
			ErrorData  *edata;

			/*
			 * Swallowing an error without unwinding the transaction leaves
			 * behind whatever the failed call was holding. Buffer content locks
			 * are the dangerous part: they are LWLocks, so they have no deadlock
			 * detector, no timeout, and they are not released at commit — a
			 * single leaked exclusive lock on an index page makes every later
			 * reader of that page hang forever, and the only way out is
			 * restarting the cluster. That happened on 2026-07-28: hybrid search
			 * stopped for 21 minutes with three backends parked in
			 * LWLock/BufferContent and no lock holder visible in pg_locks.
			 *
			 * ambuild runs at statement level and holds no LWLock of its own, so
			 * releasing everything here is safe and restores the invariant that
			 * a statement ends with no LWLock held. Buffer pins are cleaned up
			 * by the resource owner at end of transaction; those only produce a
			 * refcount warning, they do not block other backends.
			 */
			MemoryContextSwitchTo(oldcontext);
			edata = CopyErrorData();
			FlushErrorState();
			LWLockReleaseAll();
			ereport(WARNING,
					(errmsg("pgturbohybrid: sparse index build skipped"),
					 errdetail("%s", edata->message)));
			FreeErrorData(edata);
		}
		PG_END_TRY();
	}

	return result;
}

static void
pgturbohybridambuildempty(Relation index)
{
	PgturbohybridIndexKeyMap map;

	PgturbohybridValidateIndex(index, NULL);
	pgturbohybridbuildempty(index);
	if (PgturbohybridIndexHasLexical(index))
		PgturbohybridBm25BuildEmpty(index);

	PgturbohybridBuildIndexKeyMap(index, NULL, &map);
	if (map.hasSparse && PgturbohybridSparseIsPrimary(index))
		PgturbohybridSparsePrimaryBuildEmpty(index);
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
	const PgturbohybridGraphTypeInfo *typeInfo;
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
			PgturbohybridBm25AppendDelta(index, nodeId, heap_tid, lexicalValue,
										 PgturbohybridBm25TenantFromValues(index, values, isnull));
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

	/*
	 * O lock de escrita no índice volta a ser exclusivo (2026-07-28).
	 *
	 * O patch de 2026-07-11 baixou para ShareLock para deixar inserções
	 * concorrerem, e o ganho de vazão era real. O custo não estava medido, e é
	 * grave. Reproduzido em `scripts/db/stress-turbohybrid-concurrency.sh` com 4
	 * escritores e 4 leitores sobre o mesmo índice:
	 *
	 *   1. Nós de grafo desaparecem. 60 linhas inseridas, 47 nós no grafo
	 *      (node_count 237 contra bm25_document_count 260). O documento fica
	 *      visível para a busca lexical e invisível para a busca vetorial, sem
	 *      erro em log nenhum. `PgturbohybridGraphInsertValueInPlace` faz
	 *      leia-modifique-escreva no metadado do grafo (contagem de nós, ponto de
	 *      entrada), e dois desses ao mesmo tempo perdem um.
	 *   2. O leitor quebra o processo. SIGSEGV em
	 *      `PgturbohybridGraphLoadCodePage`, em `storage->codePagesLoaded[15]`:
	 *      o cache nativo compartilhado é um arquivo mapeado em memória que o
	 *      escritor faz crescer, e o leitor segue o ponteiro velho. Duas quedas
	 *      do cluster inteiro em dois minutos de teste.
	 *
	 * O motivo original do patch — inserção segurando o índice por minutos —
	 * tinha outra causa, consertada no mesmo dia: `PgturbohybridBm25AppendDelta`
	 * segurava a página de metadados do BM25 em BUFFER_LOCK_EXCLUSIVE durante a
	 * caminhada inteira da cadeia de delta, o que parava toda busca. Com aquela
	 * janela fechada, serializar escrita por índice custa pouco: a escrita é
	 * milissegundos, e a leitura continua concorrente sem restrição.
	 *
	 * Voltar a concorrer aqui exige antes: metadado do grafo com atualização
	 * atômica e cache nativo com geração revalidada no leitor. Enquanto isso não
	 * existir, correção vem antes de vazão.
	 */
	LockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, ExclusiveLock);
	PG_TRY();
	{
		nodeId = PgturbohybridGraphInsertValueInPlace(index, indexInfo, heap_tid,
													  value, values, isnull);
		if (PgturbohybridIndexGetLexicalDatum(index, values, isnull, &lexicalValue))
			PgturbohybridBm25AppendDelta(index, nodeId, heap_tid, lexicalValue,
										 PgturbohybridBm25TenantFromValues(index, values, isnull));
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
pgturbohybridamvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats)
{
	IndexBulkDeleteResult *result;

	if (PgturbohybridGraphUseTqNativeGraph(info->index))
	{
		PgturbohybridBm25PlanningStats bm25Stats;
		bool		hasBm25;

		result = tqgraphvacuumcleanup(info, stats);
		(void) PgturbohybridBm25MaybeCompact(info->index);

		/*
		 * Índice com ramo de BM25 não passa pela compactação de página
		 * (2026-07-28).
		 *
		 * `PgturbohybridGraphMaybeCompactPageChains` reescreve as cadeias de
		 * página atribuindo identificadores de nó novos e densos (o `nodeIdMap`
		 * da fase 1). As estruturas do BM25 — listas de postagem, estatística de
		 * documento, teto por bloco, camadas de impacto — são todas indexadas por
		 * identificador de nó, e nenhuma delas é reescrita ali. Depois de uma
		 * compactação, a perna lexical casaria documento por documento errado.
		 *
		 * Reproduzido: tabela de 2.000 linhas com índice híbrido, `DELETE` de
		 * metade, `VACUUM`. O grafo ficou com 1.000 nós, o metadado do BM25
		 * seguiu dizendo 2.000 documentos, e a verificação de invariante passou a
		 * derrubar **toda** varredura híbrida com "BM25 metadata document count
		 * is invalid". A busca vetorial seguia funcionando; a híbrida morria até
		 * o REINDEX. Falhar alto é melhor que responder errado, mas nenhuma das
		 * duas serve.
		 *
		 * Enquanto a renumeração não souber remapear o ramo lexical, quem tem
		 * BM25 não compacta: o nó morto fica marcado, e
		 * `PgturbohybridBm25MaybeCompact` — que reconstrói a base a partir dos
		 * nós vivos mantendo o identificador de cada um — cuida da estatística.
		 * Espaço se recupera com REINDEX.
		 */
		hasBm25 = PgturbohybridBm25GetPlanningStats(info->index, &bm25Stats) &&
			bm25Stats.hasBm25;
		if (hasBm25)
			elog(DEBUG1,
				 "pgturbohybrid: compactação de página ignorada em \"%s\" — "
				 "o ramo de BM25 é indexado por identificador de nó e a "
				 "compactação renumera; use REINDEX para recuperar espaço",
				 RelationGetRelationName(info->index));
		else
			PgturbohybridGraphMaybeCompactPageChains(info->index);

		PgturbohybridBm25ValidateMetaPointer(info->index);
		PgturbohybridBm25InvalidateCache(info->index);
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
		hasSparseQuery = PgturbohybridQueryHasSparse(hybridQuery);
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

		/* Free the detoasted copy now that it has been copied into tmpCtx.
		 * Without this, every rescan in a parameterized nested-loop join
		 * leaks one PG_DETOAST_DATUM_COPY-sized allocation. */
		pfree(hybridQuery);
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
		PGTURBOHYBRID_RELOPT_PARSE("page_compaction_threshold", RELOPT_TYPE_INT, pageCompactionThreshold),
		PGTURBOHYBRID_RELOPT_PARSE("bm25_tenant_payload_slot", RELOPT_TYPE_INT, bm25TenantPayloadSlot),
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
	DefineCustomIntVariable("turbohybrid.default_rrf_k", "Default RRF constant for turbohybrid_query callers",
							NULL, &pgturbohybrid_default_rrf_k,
							PGTURBOHYBRID_DEFAULT_RRF_K, 1,
							PGTURBOHYBRID_MAX_RRF_K,
							PGC_USERSET, 0, PgturbohybridCheckDefaultRrfK,
							PgturbohybridAssignQueryDefaultInt, NULL);
}

/*
 * Discover which index key carries each retrieval signal, by type (not fixed
 * position).  Validates structure: at most one of each kind, no unknown key
 * types, 1..MAX key columns, at least one retrieval key.
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

	add_int_reloption(pgturbohybrid_relopt_kind, "page_compaction_threshold",
				  "Percentage of dead graph nodes that triggers automatic page chain compaction during VACUUM (0 = disabled)",
				  25, 0, 100, AccessExclusiveLock);

	add_int_reloption(pgturbohybrid_relopt_kind, "bm25_tenant_payload_slot",
				  "Zero-based INCLUDE payload slot carrying the int4 tenant key for per-tenant BM25 statistics (-1 disables tracking)",
				  0, -1, PGTURBOHYBRID_GRAPH_MAX_PAYLOADS - 1, AccessExclusiveLock);

	add_int_reloption(pgturbohybrid_relopt_kind, "sparse_quant_bits",
			  "Sparse postings weight quantization bits (0=f32, 8=q8, 16=q16)",
			  0, 0, 16, AccessExclusiveLock);
	add_enum_reloption(pgturbohybrid_relopt_kind, "sparse_quant_mode",
			  "Sparse weight quantization mode",
			  pgturbohybrid_sparse_quant_mode_relopt_options,
			  PGTURBOHYBRID_SPARSE_QUANT_MODE_PER_TERM_LINEAR,
			  "f32", AccessExclusiveLock);
	add_enum_reloption(pgturbohybrid_relopt_kind, "sparse_postings_encoding",
			  "Sparse postings physical encoding",
			  pgturbohybrid_sparse_encoding_relopt_options,
			  PGTURBOHYBRID_SPARSE_ENCODING_OPT_AUTO,
			  "auto", AccessExclusiveLock);
	add_int_reloption(pgturbohybrid_relopt_kind, "sparse_block_size",
			  "Sparse postings block size (terms per chunk)",
			  256, 16, 65536, AccessExclusiveLock);
	add_enum_reloption(pgturbohybrid_relopt_kind, "sparse_exact_storage",
			  "Sparse exact-storage mode (off = quantized only)",
			  pgturbohybrid_sparse_exact_storage_relopt_options,
			  PGTURBOHYBRID_SPARSE_EXACT_STORAGE_OFF,
			  "off", AccessExclusiveLock);
	add_bool_reloption(pgturbohybrid_relopt_kind, "sparse_block_max",
			   "Enable block-max WAND pruning for sparse top-k",
			   true, AccessExclusiveLock);

	if (IsParallelWorker())
		return;

	gucsAlreadyDefined =
		GetConfigOption("turbohybrid.default_dense_k", true, false) != NULL;
	if (gucsAlreadyDefined)
		return;
	PgturbohybridDefineDefaultBudgetGUCs();
	DefineCustomBoolVariable("turbohybrid.bm25_tenant_stats",
							 "Scope BM25 idf and length normalization to the tenant extracted from the query's int4 payload equality filter",
							 "Requires an index built with bm25_tenant_payload_slot and a tenant stats chain (bm25Version >= 2). Queries without the filter, or tenants without recorded aggregates, keep global statistics.",
							 &pgturbohybrid_bm25_tenant_stats,
							 true,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("turbohybrid.enable_wand", "Enable WAND pruning for BM25 candidate generation",
							 NULL, &pgturbohybrid_enable_wand,
							 true, PGC_USERSET, 0, NULL, NULL, NULL);
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
							 "on forces the u8 maddubs/VPDPBUSD split whenever its hard requirements hold (4-bit, dim>=1024, mode!=L1, AVX2+); off disables it (signed split or scalar/LUT); auto defers to dense_query_split_impl. For controlled benchmarking.",
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
	DefineCustomIntVariable("turbohybrid.native_cache_disk_max_mb",
							"Max total on-disk shared native cache under pg_turbohybrid_cache",
							"Caps the aggregate size of mmap-backed .tqcache files in the PostgreSQL data directory. When exceeded, the oldest unattached cache files are deleted automatically after each new cache build and via turbohybrid_prune_shared_cache(). Stale files from REINDEX are also removed per index. Set to 0 to disable the cap (not recommended on space-constrained hosts). Default 8192.",
							&pgturbohybrid_native_cache_disk_max_mb,
							8192, 0, 1048576,
							PGC_SIGHUP, GUC_UNIT_MB, NULL, NULL, NULL);
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
