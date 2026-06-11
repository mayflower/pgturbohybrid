#include "postgres.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>

#include "access/genam.h"
#include "catalog/pg_class.h"
#include "fmgr.h"
#include "pgturbohybrid.h"
#include "lib/stringinfo.h"
#include "pgturbohybrid_multivector.h"
#include "pgturbohybrid_quant.h"
#include "pgturbohybrid_am.h"
#include "pgturbohybrid_bm25.h"
#include "pgturbohybrid_vector_compat.h"
#include "pgturbohybrid_jsonb_compat.h"
#include "utils/fmgrprotos.h"
#include "utils/jsonb.h"
#include "utils/numeric.h"
#include "utils/rel.h"

#define PGTURBOHYBRID_SCAN_ORCHESTRATION_NONE		0
#define PGTURBOHYBRID_SCAN_ORCHESTRATION_GRAPH		1
#define PGTURBOHYBRID_SCAN_ORCHESTRATION_FLAT		2

static int64 pgturbohybrid_last_graph_visited_nodes = 0;
static int64 pgturbohybrid_last_graph_scored_codes = 0;
static int64 pgturbohybrid_last_graph_batch_scored_codes = 0;
static int64 pgturbohybrid_last_graph_scalar_scored_codes = 0;
static int pgturbohybrid_last_graph_batch_kernel = PGTURBOHYBRID_SCORING_SCALAR;
static int pgturbohybrid_last_weighted_code_code_kernel = PGTURBOHYBRID_SCORING_SCALAR;
static int64 pgturbohybrid_last_graph_candidate_count = 0;
static int pgturbohybrid_last_graph_rescore_band = PGTURBOHYBRID_GRAPH_RESCORE_BAND_AUTO;
static int64 pgturbohybrid_last_graph_rescore_count = 0;
static int64 pgturbohybrid_last_graph_rescore_pages = 0;
static int64 pgturbohybrid_last_graph_code_pages_read = 0;
static int64 pgturbohybrid_last_graph_adj_pages_read = 0;
static int64 pgturbohybrid_last_graph_segment_count = 0;
static int64 pgturbohybrid_last_graph_segments_searched = 0;
static int pgturbohybrid_last_graph_per_segment_budget_mode = PGTURBOHYBRID_NATIVE_SEGMENT_BUDGET_OFF;
static int64 pgturbohybrid_last_graph_search_ef_before_segment_scaling = 0;
static int64 pgturbohybrid_last_graph_search_ef_after_segment_scaling = 0;
static int64 pgturbohybrid_last_graph_code_page_attempts = 0;
static int64 pgturbohybrid_last_graph_code_page_hits = 0;
static int64 pgturbohybrid_last_graph_code_page_misses = 0;
static int64 pgturbohybrid_last_graph_code_tuples_copied = 0;
static int64 pgturbohybrid_last_graph_code_arena_allocated_bytes = 0;
static int64 pgturbohybrid_last_graph_code_arena_used_bytes = 0;
static int64 pgturbohybrid_last_graph_entry_point_count = 0;
static int64 pgturbohybrid_last_graph_entry_sample_configured = 0;
static int64 pgturbohybrid_last_graph_entry_sample_effective = 0;
static int64 pgturbohybrid_last_graph_entry_sample_scored = 0;
static int64 pgturbohybrid_last_graph_entry_sidecar_count = 0;
static int64 pgturbohybrid_last_graph_entry_sidecar_scored = 0;
static int64 pgturbohybrid_last_graph_entry_sidecar_selected = 0;
static int64 pgturbohybrid_last_graph_entry_sidecar_representatives_configured = 0;
static int pgturbohybrid_last_graph_entry_sidecar_strategy = PGTURBOHYBRID_DEFAULT_ENTRY_SIDECAR_STRATEGY;
static int64 pgturbohybrid_last_graph_entry_sidecar_us = 0;
static int pgturbohybrid_last_payload_entry_seeding_mode =
	PGTURBOHYBRID_PAYLOAD_ENTRY_SEEDING_AUTO;
static bool pgturbohybrid_last_payload_entry_seeding_hit = false;
static int64 pgturbohybrid_last_payload_entry_seed_count = 0;
static int pgturbohybrid_last_payload_entry_seed_payload_slot = -1;
static int64 pgturbohybrid_last_payload_entry_seed_range_count = 0;
static int64 pgturbohybrid_last_payload_entry_seed_us = 0;
static int64 pgturbohybrid_last_graph_residual_rerank_count = 0;
static int64 pgturbohybrid_last_graph_residual_rerank_bytes = 0;
static int64 pgturbohybrid_last_graph_residual_rerank_us = 0;
static int pgturbohybrid_last_graph_residual_rerank_mode =
	PGTURBOHYBRID_DENSE_RESIDUAL_RERANK_CALIBRATED;
static double pgturbohybrid_last_graph_residual_rerank_weight_effective = 0.0;
static int64 pgturbohybrid_last_graph_residual_rerank_band = 0;
static int pgturbohybrid_last_graph_residual_rerank_band_multiplier = 0;
static double pgturbohybrid_last_graph_residual_rerank_max_adjustment = 0.0;
static int64 pgturbohybrid_last_graph_residual_rerank_reordered_count = 0;
static bool pgturbohybrid_last_graph_residual_rerank_topk_changed = false;
static int64 pgturbohybrid_last_graph_heap_rescore_count = 0;
static int64 pgturbohybrid_last_graph_heap_fetch_us = 0;
static int64 pgturbohybrid_last_graph_heap_rescore_us = 0;
static int pgturbohybrid_last_graph_heap_rescore_mode = PGTURBOHYBRID_DENSE_HEAP_RESCORE_OFF;
static bool pgturbohybrid_last_graph_heap_rescore_auto_enabled = false;
static int pgturbohybrid_last_graph_heap_rescore_reason = PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_UNKNOWN;
static int64 pgturbohybrid_last_graph_prepare_us = 0;
static int64 pgturbohybrid_last_graph_traverse_us = 0;
static int64 pgturbohybrid_last_graph_entry_us = 0;
static int64 pgturbohybrid_last_graph_base_us = 0;
static int64 pgturbohybrid_last_graph_batch_us = 0;
static int64 pgturbohybrid_last_graph_heap_us = 0;
static int64 pgturbohybrid_last_graph_fill_us = 0;
static int64 pgturbohybrid_last_graph_fill_candidate_band_calls = 0;
static int pgturbohybrid_last_graph_fill_candidate_band_reason =
	PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_NONE;
static int64 pgturbohybrid_last_graph_fill_candidate_band_visited = 0;
static int64 pgturbohybrid_last_graph_fill_candidate_band_scored = 0;
static int64 pgturbohybrid_last_graph_fill_candidate_band_selected_before = 0;
static int64 pgturbohybrid_last_graph_fill_candidate_band_selected_after = 0;
static int64 pgturbohybrid_last_graph_fill_candidate_band_target = 0;
static bool pgturbohybrid_last_graph_fill_candidate_band_used_payload_refs = false;
static int64 pgturbohybrid_last_graph_fill_candidate_band_payload_ref_count = 0;
static int64 pgturbohybrid_last_graph_rescore_us = 0;
static int64 pgturbohybrid_last_graph_sort_us = 0;
static int64 pgturbohybrid_last_graph_total_us = 0;
static int64 pgturbohybrid_last_graph_dense_requested_k = 0;
static int64 pgturbohybrid_last_graph_effective_result_target = 0;
static int64 pgturbohybrid_last_graph_effective_search_ef = 0;
static int64 pgturbohybrid_last_graph_effective_rescore_band = 0;
static double pgturbohybrid_last_graph_highdim_widening_multiplier = 1.0;
static int pgturbohybrid_last_graph_widening_reason = PGTURBOHYBRID_DENSE_WIDENING_NONE;
static bool pgturbohybrid_last_dense_filter_unmapped = false;
static bool pgturbohybrid_last_dense_linear_fallback_warning = false;
static double pgturbohybrid_last_dense_linear_fallback_ratio = 0.0;
static int pgturbohybrid_last_graph_adaptive_widening_mode = PGTURBOHYBRID_DENSE_ADAPTIVE_WIDENING_OFF;
static bool pgturbohybrid_last_graph_adaptive_triggered = false;
static int pgturbohybrid_last_graph_adaptive_trigger_reason = PGTURBOHYBRID_DENSE_ADAPTIVE_REASON_NONE;
static int64 pgturbohybrid_last_graph_adaptive_initial_result_target = 0;
static int64 pgturbohybrid_last_graph_adaptive_final_result_target = 0;
static int64 pgturbohybrid_last_graph_adaptive_initial_search_ef = 0;
static int64 pgturbohybrid_last_graph_adaptive_final_search_ef = 0;
static double pgturbohybrid_last_graph_adaptive_gap_top10 = 0.0;
static double pgturbohybrid_last_graph_adaptive_gap_boundary = 0.0;
static int pgturbohybrid_last_graph_uncertainty_retry_mode = PGTURBOHYBRID_DENSE_UNCERTAINTY_RETRY_OFF;
static bool pgturbohybrid_last_graph_uncertainty_retry_triggered = false;
static int pgturbohybrid_last_graph_uncertainty_retry_reason = PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_NONE;
static int64 pgturbohybrid_last_graph_uncertainty_retry_passes = 0;
static int64 pgturbohybrid_last_graph_uncertainty_initial_result_target = 0;
static int64 pgturbohybrid_last_graph_uncertainty_final_result_target = 0;
static int64 pgturbohybrid_last_graph_uncertainty_initial_search_ef = 0;
static int64 pgturbohybrid_last_graph_uncertainty_final_search_ef = 0;
static double pgturbohybrid_last_graph_uncertainty_gap_top10 = 0.0;
static double pgturbohybrid_last_graph_uncertainty_gap_boundary = 0.0;
static int pgturbohybrid_last_graph_local_expansion_mode = PGTURBOHYBRID_DENSE_LOCAL_EXPANSION_OFF;
static bool pgturbohybrid_last_graph_local_expansion_triggered = false;
static int64 pgturbohybrid_last_graph_local_expansion_seed_count = 0;
static int64 pgturbohybrid_last_graph_local_expansion_neighbors_scored = 0;
static int64 pgturbohybrid_last_graph_local_expansion_candidates_added = 0;
static int64 pgturbohybrid_last_graph_local_expansion_us = 0;
static int pgturbohybrid_last_graph_dense_budget_policy = PGTURBOHYBRID_DENSE_BUDGET_AUTO;
static int pgturbohybrid_last_graph_rescore_band_policy = PGTURBOHYBRID_RESCORE_BAND_POLICY_AUTO;
static int pgturbohybrid_last_graph_exact_rescore_source = PGTURBOHYBRID_EXACT_RESCORE_SOURCE_NONE;
static int pgturbohybrid_last_graph_scoring_kernel = PGTURBOHYBRID_SCORING_SCALAR;
static int pgturbohybrid_last_exact_vector_kernel = PGTURBOHYBRID_EXACT_KERNEL_SCALAR;
static bool pgturbohybrid_last_exact_vector_kernel_recorded = false;
static int pgturbohybrid_last_graph_storage_kind = PGTURBOHYBRID_GRAPH_STORAGE_GRAPH;
static int pgturbohybrid_last_scan_orchestration = PGTURBOHYBRID_SCAN_ORCHESTRATION_NONE;
static int pgturbohybrid_last_graph_quantization_bits = 0;
static bool pgturbohybrid_last_graph_exact_storage = false;
static bool pgturbohybrid_last_graph_exact_storage_known = false;
static bool pgturbohybrid_last_graph_build_exact_distances = false;
static int pgturbohybrid_last_graph_build_distance_mode = PGTURBOHYBRID_DENSE_BUILD_DISTANCE_CODE;
static bool pgturbohybrid_last_graph_build_fast_edges = false;
static int pgturbohybrid_last_graph_build_neighbor_select_reason = PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_UNKNOWN;
static bool pgturbohybrid_last_graph_query_split_active = false;
static bool pgturbohybrid_last_graph_querysplit_used = false;
static bool pgturbohybrid_last_graph_u8_split_used = false;
static bool pgturbohybrid_last_dense_u8_batch_x4_enabled = true;
static int pgturbohybrid_last_graph_u8_batch_mode = PGTURBOHYBRID_U8_BATCH_NONE;
static int pgturbohybrid_last_graph_u8_kernel_single = PGTURBOHYBRID_U8_KERNEL_NONE;
static int pgturbohybrid_last_graph_u8_kernel_batch = PGTURBOHYBRID_U8_KERNEL_NONE;
static bool pgturbohybrid_last_graph_large_code_arena = false;
static bool pgturbohybrid_last_graph_whole_code_prefetch_active = false;
static int64 pgturbohybrid_last_graph_code_bytes = 0;
static int64 pgturbohybrid_last_graph_code_arena_estimated_bytes = 0;
static int pgturbohybrid_last_graph_native_cache_mode = PGTURBOHYBRID_GRAPH_NATIVE_CACHE_NONE;
static int pgturbohybrid_last_graph_native_cache_policy = PGTURBOHYBRID_NATIVE_CACHE_POLICY_AUTO;
static int pgturbohybrid_last_graph_native_cache_reason = PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_NONE;
static bool pgturbohybrid_last_graph_native_cache_used = false;
static bool pgturbohybrid_last_graph_native_cache_reused = false;
static bool pgturbohybrid_last_graph_native_cache_built_this_scan = false;
static int64 pgturbohybrid_last_graph_native_cache_attach_us = 0;
static int64 pgturbohybrid_last_graph_native_cache_build_us = 0;
static int64 pgturbohybrid_last_graph_native_cache_wait_us = 0;
static int64 pgturbohybrid_last_graph_native_cache_refcount = -1;
static int64 pgturbohybrid_last_graph_native_cache_bytes = 0;
static int64 pgturbohybrid_last_graph_native_cache_code_bytes = 0;
static int64 pgturbohybrid_last_graph_native_cache_adj_bytes = 0;
static int64 pgturbohybrid_last_graph_native_cache_exact_bytes = 0;
static bool pgturbohybrid_last_graph_native_cache_warning = false;
static const char *pgturbohybrid_last_graph_native_cache_warning_reason = "none";
static int64 pgturbohybrid_last_graph_scan_lock_wait_us = 0;
static int64 pgturbohybrid_last_graph_code_buffer_lock_wait_us = 0;
static int64 pgturbohybrid_last_graph_adj_buffer_lock_wait_us = 0;
static int64 pgturbohybrid_last_graph_dimensions = 0;
static int64 pgturbohybrid_last_graph_returned_rows = 0;
static int64 pgturbohybrid_last_graph_m = 0;
static int64 pgturbohybrid_last_graph_ef_construction = 0;
static int64 pgturbohybrid_last_graph_ef_search = 0;
static int64 pgturbohybrid_last_graph_oversampling = 0;
static int pgturbohybrid_last_graph_exact_cache = PGTURBOHYBRID_GRAPH_EXACT_CACHE_AUTO;
static int64 pgturbohybrid_last_graph_score_kernel_nodes[PGTURBOHYBRID_SCORE_KERNEL_BUCKET_COUNT];
static int64 pgturbohybrid_last_graph_score_kernel_calls[PGTURBOHYBRID_SCORE_KERNEL_BUCKET_COUNT];
static int64 pgturbohybrid_last_graph_batch_calls = 0;
static int64 pgturbohybrid_last_graph_batch_nodes = 0;
static int64 pgturbohybrid_last_graph_base_frontier_pushes = 0;
static int64 pgturbohybrid_last_graph_base_frontier_pops = 0;
static int64 pgturbohybrid_last_graph_base_nearest_offers = 0;
static int64 pgturbohybrid_last_graph_base_visited_checks = 0;
static int64 pgturbohybrid_last_graph_base_duplicate_skips = 0;
static int64 pgturbohybrid_last_graph_base_batch_calls = 0;
static int64 pgturbohybrid_last_graph_base_batch_nodes = 0;
static int64 pgturbohybrid_last_graph_base_max_frontier = 0;
static int pgturbohybrid_last_graph_score_mode = PGTURBOHYBRID_SCORE_L2;
static int pgturbohybrid_last_graph_simd_force = PGTURBOHYBRID_SIMD_FORCE_AUTO;
static PgturbohybridNativeBuildStatsSnapshot pgturbohybrid_last_native_build_stats;
static bool pgturbohybrid_last_native_build_stats_valid = false;
static PgturbohybridGraphDocInsertStats pgturbohybrid_last_doc_insert_stats;

typedef struct PgturbohybridMultiVectorProxyDiagnosticScore
{
	uint32		sampleIndex;
	double		score;
} PgturbohybridMultiVectorProxyDiagnosticScore;

typedef struct PgturbohybridMultiVectorProxyDiagnosticResult
{
	uint32		corpusDocs;
	uint32		sampleDocsRequested;
	uint32		queryCountRequested;
	uint32		sampleDocs;
	uint32		queryCount;
	uint32		dimensions;
	double		avgDocTokens;
	int			proxyEncoder;
	double		recallAt10ProxyToExact;
	double		avgProxyExactRankCorrelation;
	uint32		recommendedDocCandidateK;
	uint64		exactPairsScored;
	uint64		proxyPairsScored;
} PgturbohybridMultiVectorProxyDiagnosticResult;

static int
PgturbohybridMultiVectorProxyDiagnosticScoreCompare(const void *a,
													const void *b)
{
	const PgturbohybridMultiVectorProxyDiagnosticScore *sa =
		(const PgturbohybridMultiVectorProxyDiagnosticScore *) a;
	const PgturbohybridMultiVectorProxyDiagnosticScore *sb =
		(const PgturbohybridMultiVectorProxyDiagnosticScore *) b;

	if (sa->score > sb->score)
		return -1;
	if (sa->score < sb->score)
		return 1;
	if (sa->sampleIndex < sb->sampleIndex)
		return -1;
	if (sa->sampleIndex > sb->sampleIndex)
		return 1;
	return 0;
}

static void
PgturbohybridJsonbAddKey(PgturbohybridJsonbState *state, const char *key)
{
	JsonbValue	value;

	value.type = jbvString;
	value.val.string.val = (char *) key;
	value.val.string.len = strlen(key);
	PgturbohybridJsonbPush(state, WJB_KEY, &value);
}

static void
PgturbohybridJsonbAddString(PgturbohybridJsonbState *state, const char *key, const char *val)
{
	JsonbValue	value;

	PgturbohybridJsonbAddKey(state, key);
	value.type = jbvString;
	value.val.string.val = (char *) val;
	value.val.string.len = strlen(val);
	PgturbohybridJsonbPush(state, WJB_VALUE, &value);
}

static void
PgturbohybridJsonbAddBool(PgturbohybridJsonbState *state, const char *key, bool val)
{
	JsonbValue	value;

	PgturbohybridJsonbAddKey(state, key);
	value.type = jbvBool;
	value.val.boolean = val;
	PgturbohybridJsonbPush(state, WJB_VALUE, &value);
}

static void
PgturbohybridJsonbAddNull(PgturbohybridJsonbState *state, const char *key)
{
	JsonbValue	value;

	PgturbohybridJsonbAddKey(state, key);
	value.type = jbvNull;
	PgturbohybridJsonbPush(state, WJB_VALUE, &value);
}

static void
PgturbohybridJsonbAddInt64(PgturbohybridJsonbState *state, const char *key, int64 val)
{
	JsonbValue	value;

	PgturbohybridJsonbAddKey(state, key);
	value.type = jbvNumeric;
	value.val.numeric = DatumGetNumeric(DirectFunctionCall1(int8_numeric,
															Int64GetDatum(val)));
	PgturbohybridJsonbPush(state, WJB_VALUE, &value);
}

static void
PgturbohybridJsonbAddUint64(PgturbohybridJsonbState *state, const char *key, uint64 val)
{
	char		buf[32];
	JsonbValue	value;

	snprintf(buf, sizeof(buf), UINT64_FORMAT, val);
	PgturbohybridJsonbAddKey(state, key);
	value.type = jbvNumeric;
	value.val.numeric = DatumGetNumeric(DirectFunctionCall3(numeric_in,
															CStringGetDatum(buf),
															ObjectIdGetDatum(InvalidOid),
															Int32GetDatum(-1)));
	PgturbohybridJsonbPush(state, WJB_VALUE, &value);
}

static void
PgturbohybridJsonbAddUint64Array(PgturbohybridJsonbState *state,
								 const char *key, const uint64 *vals,
								 uint32 count)
{
	PgturbohybridJsonbAddKey(state, key);
	PgturbohybridJsonbPush(state, WJB_BEGIN_ARRAY, NULL);
	for (uint32 i = 0; i < count; i++)
	{
		char		buf[32];
		JsonbValue	value;

		snprintf(buf, sizeof(buf), UINT64_FORMAT, vals[i]);
		value.type = jbvNumeric;
		value.val.numeric = DatumGetNumeric(DirectFunctionCall3(numeric_in,
																CStringGetDatum(buf),
																ObjectIdGetDatum(InvalidOid),
																Int32GetDatum(-1)));
		PgturbohybridJsonbPush(state, WJB_ELEM, &value);
	}
	PgturbohybridJsonbPush(state, WJB_END_ARRAY, NULL);
}

static void
PgturbohybridJsonbAddUint32Array(PgturbohybridJsonbState *state,
								 const char *key, const uint32 *vals,
								 uint32 count)
{
	PgturbohybridJsonbAddKey(state, key);
	PgturbohybridJsonbPush(state, WJB_BEGIN_ARRAY, NULL);
	for (uint32 i = 0; i < count; i++)
	{
		JsonbValue	value;

		value.type = jbvNumeric;
		value.val.numeric = DatumGetNumeric(DirectFunctionCall1(int8_numeric,
																Int64GetDatum(vals[i])));
		PgturbohybridJsonbPush(state, WJB_ELEM, &value);
	}
	PgturbohybridJsonbPush(state, WJB_END_ARRAY, NULL);
}

static void
PgturbohybridJsonbAddBoolArray(PgturbohybridJsonbState *state,
							   const char *key, const bool *vals,
							   uint32 count)
{
	PgturbohybridJsonbAddKey(state, key);
	PgturbohybridJsonbPush(state, WJB_BEGIN_ARRAY, NULL);
	for (uint32 i = 0; i < count; i++)
	{
		JsonbValue	value;

		value.type = jbvBool;
		value.val.boolean = vals[i];
		PgturbohybridJsonbPush(state, WJB_ELEM, &value);
	}
	PgturbohybridJsonbPush(state, WJB_END_ARRAY, NULL);
}

static void
PgturbohybridJsonbAddStringArray(PgturbohybridJsonbState *state,
								 const char *key, const char *const *vals,
								 uint32 count)
{
	PgturbohybridJsonbAddKey(state, key);
	PgturbohybridJsonbPush(state, WJB_BEGIN_ARRAY, NULL);
	for (uint32 i = 0; i < count; i++)
	{
		JsonbValue	value;
		const char *val = vals[i] != NULL ? vals[i] : "";

		value.type = jbvString;
		value.val.string.val = (char *) val;
		value.val.string.len = strlen(val);
		PgturbohybridJsonbPush(state, WJB_ELEM, &value);
	}
	PgturbohybridJsonbPush(state, WJB_END_ARRAY, NULL);
}

static void
PgturbohybridJsonbAddFloat8Array(PgturbohybridJsonbState *state,
								 const char *key, const double *vals,
								 uint32 count)
{
	PgturbohybridJsonbAddKey(state, key);
	PgturbohybridJsonbPush(state, WJB_BEGIN_ARRAY, NULL);
	for (uint32 i = 0; i < count; i++)
	{
		JsonbValue	value;

		value.type = jbvNumeric;
		value.val.numeric = DatumGetNumeric(DirectFunctionCall1(float8_numeric,
																Float8GetDatum(vals[i])));
		PgturbohybridJsonbPush(state, WJB_ELEM, &value);
	}
	PgturbohybridJsonbPush(state, WJB_END_ARRAY, NULL);
}

static const char *
PgturbohybridBranchPlanModeName(int mode)
{
	switch (mode)
	{
		case PGTURBOHYBRID_BRANCH_PLAN_AUTO:
			return "auto";
		case PGTURBOHYBRID_BRANCH_PLAN_DENSE_ONLY:
			return "dense_only";
		case PGTURBOHYBRID_BRANCH_PLAN_QDRANT_LIKE:
			return "qdrant_like";
		default:
			return "unknown";
	}
}

static const char *
PgturbohybridBranchKindName(int kind)
{
	switch (kind)
	{
		case PGTURBOHYBRID_BRANCH_KIND_BM25:
			return "bm25";
		case PGTURBOHYBRID_BRANCH_KIND_DENSE_SINGLE:
			return "dense_single";
		case PGTURBOHYBRID_BRANCH_KIND_PROXY_VECTOR:
			return "proxy_vector";
		case PGTURBOHYBRID_BRANCH_KIND_DOCUMENT_NODES:
			return "document_nodes";
		case PGTURBOHYBRID_BRANCH_KIND_TOKEN_NODES:
			return "token_nodes";
		case PGTURBOHYBRID_BRANCH_KIND_EXACT_DOC_SCAN:
			return "exact_doc_scan";
		case PGTURBOHYBRID_BRANCH_KIND_CENTROID_LITE:
			return "centroid_lite";
		case PGTURBOHYBRID_BRANCH_KIND_QUANTIZED_INVERTED_EXPERIMENTAL:
			return "quantized_inverted_experimental";
		default:
			return "unknown";
	}
}

static void
PgturbohybridJsonbAddBranchPlan(PgturbohybridJsonbState *state,
								const PgturbohybridBranchPlan *plan)
{
	const char *kinds[PGTURBOHYBRID_BRANCH_PLAN_MAX_BRANCHES];
	uint32		candidateCounts[PGTURBOHYBRID_BRANCH_PLAN_MAX_BRANCHES];
	uint32		candidateLimits[PGTURBOHYBRID_BRANCH_PLAN_MAX_BRANCHES];
	uint32		rescoreLimits[PGTURBOHYBRID_BRANCH_PLAN_MAX_BRANCHES];
	uint32		branchRanks[PGTURBOHYBRID_BRANCH_PLAN_MAX_BRANCHES];
	uint32		sourceFlags[PGTURBOHYBRID_BRANCH_PLAN_MAX_BRANCHES];
	uint64		latencyUs[PGTURBOHYBRID_BRANCH_PLAN_MAX_BRANCHES];
	double		branchScores[PGTURBOHYBRID_BRANCH_PLAN_MAX_BRANCHES];
	bool		truncated[PGTURBOHYBRID_BRANCH_PLAN_MAX_BRANCHES];
	uint32		count;

	if (plan == NULL)
		return;
	count = Min(plan->count, (uint32) PGTURBOHYBRID_BRANCH_PLAN_MAX_BRANCHES);
	for (uint32 i = 0; i < count; i++)
	{
		const PgturbohybridBranchPlanItem *item = &plan->items[i];

		kinds[i] = PgturbohybridBranchKindName(item->kind);
		candidateCounts[i] = item->candidateCount;
		candidateLimits[i] = item->candidateLimit;
		rescoreLimits[i] = item->rescoreLimit;
		branchRanks[i] = item->branchRank;
		sourceFlags[i] = item->sourceFlags;
		latencyUs[i] = item->latencyUs;
		branchScores[i] = item->branchScore;
		truncated[i] = item->truncated;
	}

	PgturbohybridJsonbAddString(state, "multivector_branch_plan",
								PgturbohybridBranchPlanModeName(plan->mode));
	PgturbohybridJsonbAddInt64(state, "branch_count", count);
	PgturbohybridJsonbAddStringArray(state, "branch_kinds", kinds, count);
	PgturbohybridJsonbAddUint32Array(state, "branch_candidate_counts",
									 candidateCounts, count);
	PgturbohybridJsonbAddBoolArray(state, "branch_truncated_flags",
								   truncated, count);
	PgturbohybridJsonbAddUint64Array(state, "branch_latency_us",
									 latencyUs, count);
	PgturbohybridJsonbAddString(state, "branch_fusion_mode",
								plan->fusionMode[0] != '\0' ?
								plan->fusionMode : "none");
	PgturbohybridJsonbAddUint32Array(state, "branch_candidate_limits",
									 candidateLimits, count);
	PgturbohybridJsonbAddUint32Array(state, "branch_rescore_limits",
									 rescoreLimits, count);
	PgturbohybridJsonbAddUint32Array(state, "branch_ranks",
									 branchRanks, count);
	PgturbohybridJsonbAddFloat8Array(state, "branch_scores",
									 branchScores, count);
	PgturbohybridJsonbAddUint32Array(state, "branch_source_flags",
									 sourceFlags, count);
}

static void
PgturbohybridJsonbAddFloat8(PgturbohybridJsonbState *state, const char *key, double val)
{
	JsonbValue	value;

	PgturbohybridJsonbAddKey(state, key);
	value.type = jbvNumeric;
	value.val.numeric = DatumGetNumeric(DirectFunctionCall1(float8_numeric,
															Float8GetDatum(val)));
	PgturbohybridJsonbPush(state, WJB_VALUE, &value);
}

static void
PgturbohybridJsonbAddMultiVectorAdmissionTrace(PgturbohybridJsonbState *state,
											   const PgturbohybridScanStatsSnapshot *scanStats)
{
	uint32		count;

	if (scanStats == NULL || !scanStats->multivectorAdmissionTraceAvailable)
		return;

	count = Min(scanStats->multivectorAdmissionTraceCount,
				(uint32) PGTURBOHYBRID_MULTIVECTOR_DEBUG_TRACE_LIMIT_MAX);
	PgturbohybridJsonbAddKey(state, "multivector_admission_trace");
	PgturbohybridJsonbPush(state, WJB_BEGIN_ARRAY, NULL);
	for (uint32 i = 0; i < count; i++)
	{
		const PgturbohybridMultiVectorAdmissionTraceEntry *entry =
			&scanStats->multivectorAdmissionTrace[i];
		char		tidbuf[64];

		snprintf(tidbuf, sizeof(tidbuf), "(%u,%u)",
				 entry->block, entry->offset);
		PgturbohybridJsonbPush(state, WJB_BEGIN_OBJECT, NULL);
		PgturbohybridJsonbAddInt64(state, "doc_id", entry->docId);
		PgturbohybridJsonbAddString(state, "heap_tid", tidbuf);
		PgturbohybridJsonbAddInt64(state, "heap_block", entry->block);
		PgturbohybridJsonbAddInt64(state, "heap_offset", entry->offset);
		PgturbohybridJsonbAddInt64(state, "best_node_id", entry->bestNodeId);
		PgturbohybridJsonbAddFloat8(state, "approximate_score_before_rerank",
									entry->approximateScoreBeforeRerank);
		PgturbohybridJsonbAddInt64(state, "query_token_coverage_count",
								   entry->queryTokenCoverageCount);
		PgturbohybridJsonbAddInt64(state, "raw_hit_count",
								   entry->rawHitCount);
		PgturbohybridJsonbAddInt64(state, "duplicate_hit_count",
								   entry->duplicateHitCount);
		PgturbohybridJsonbAddInt64(state, "candidate_rank_before_truncation",
								   entry->candidateRankBeforeTruncation);
		PgturbohybridJsonbAddBool(state, "retained_for_exact_rerank",
								  entry->retainedForExactRerank);
		if (entry->exactRerankScoreAvailable)
			PgturbohybridJsonbAddFloat8(state, "exact_rerank_score",
										entry->exactRerankScore);
		else
			PgturbohybridJsonbAddNull(state, "exact_rerank_score");
		PgturbohybridJsonbCloseObject(state);
	}
	PgturbohybridJsonbPush(state, WJB_END_ARRAY, NULL);
}

static void
PgturbohybridJsonbAddMultiVectorTokenStats(PgturbohybridJsonbState *state,
										   const PgturbohybridScanStatsSnapshot *scanStats)
{
	uint32		count;

	if (scanStats == NULL || !scanStats->multivectorTokenStatsAvailable)
		return;

	count = Min(scanStats->multivectorTokenStatsCount,
				(uint32) PGTURBOHYBRID_MULTIVECTOR_TOKEN_STATS_LIMIT_MAX);
	PgturbohybridJsonbAddKey(state, "multivector_query_token_stats");
	PgturbohybridJsonbPush(state, WJB_BEGIN_ARRAY, NULL);
	for (uint32 i = 0; i < count; i++)
	{
		const PgturbohybridMultiVectorTokenStatsEntry *entry =
			&scanStats->multivectorTokenStats[i];

		PgturbohybridJsonbPush(state, WJB_BEGIN_OBJECT, NULL);
		PgturbohybridJsonbAddInt64(state, "query_token_ordinal",
								   entry->queryTokenOrdinal);
		PgturbohybridJsonbAddInt64(state, "raw_hits", entry->rawHits);
		PgturbohybridJsonbAddInt64(state, "unique_docs", entry->uniqueDocs);
		PgturbohybridJsonbAddInt64(state, "duplicate_doc_hits",
								   entry->duplicateDocHits);
		if (entry->topHitSimilarityAvailable)
			PgturbohybridJsonbAddFloat8(state, "top_hit_similarity",
										entry->topHitSimilarity);
		else
			PgturbohybridJsonbAddNull(state, "top_hit_similarity");
		PgturbohybridJsonbAddFloat8(state, "contribution_to_top_candidates",
									entry->contributionToTopCandidates);
		PgturbohybridJsonbAddInt64(state, "candidate_docs_retained_from_token",
								   entry->candidateDocsRetainedFromToken);
		PgturbohybridJsonbAddBool(state, "skipped", entry->skipped);
		PgturbohybridJsonbCloseObject(state);
	}
	PgturbohybridJsonbPush(state, WJB_END_ARRAY, NULL);
}

void
PgturbohybridGraphRecordNativeBuildStats(const PgturbohybridNativeBuildStatsSnapshot *stats)
{
	if (stats == NULL)
		return;

	memcpy(&pgturbohybrid_last_native_build_stats, stats,
		   sizeof(pgturbohybrid_last_native_build_stats));
	pgturbohybrid_last_native_build_stats_valid = true;
}

void
PgturbohybridGraphRecordDocInsertStats(const PgturbohybridGraphDocInsertStats *stats)
{
	if (stats == NULL)
	{
		memset(&pgturbohybrid_last_doc_insert_stats, 0,
			   sizeof(pgturbohybrid_last_doc_insert_stats));
		return;
	}

	memcpy(&pgturbohybrid_last_doc_insert_stats, stats,
		   sizeof(pgturbohybrid_last_doc_insert_stats));
}

static const char *TqScanOrchestrationName(void);
static const char *PgturbohybridFinalKSourceName(void);
static const char *TqExactKernelName(int kernel);
static const char *PgturbohybridGraphAvx512WeightedModeName(int mode);

void
PgturbohybridGraphRecordExactVectorKernel(int kernel)
{
	pgturbohybrid_last_exact_vector_kernel = kernel;
	pgturbohybrid_last_exact_vector_kernel_recorded = true;
}

void
PgturbohybridGraphRecordWeightedCodeCodeKernel(int kernel)
{
	pgturbohybrid_last_weighted_code_code_kernel = kernel;
}

/*
 * Update the heap-tuples-returned counter at end of scan.  The bulk scan stats
 * are captured when the graph search completes (before any tuple is emitted),
 * so returnedRows is still zero then; this records the final tally.
 */
void
PgturbohybridGraphRecordReturnedRows(int64 returnedRows)
{
	pgturbohybrid_last_graph_returned_rows = returnedRows;
}

static int
TqExpectedExactKernel(void)
{
	if (pgturbohybrid_dense_exact_simd_force == PGTURBOHYBRID_EXACT_SIMD_FORCE_SCALAR)
		return PGTURBOHYBRID_EXACT_KERNEL_SCALAR;
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && (defined(__aarch64__) || defined(_M_ARM64))
	return PGTURBOHYBRID_EXACT_KERNEL_NEON;
#endif
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && defined(__x86_64__)
	return PGTURBOHYBRID_EXACT_KERNEL_AVX2;
#endif
	return PGTURBOHYBRID_EXACT_KERNEL_AUTOVEC_FMA;
}

void
PgturbohybridGraphRecordGraphScanStats(PgturbohybridGraphScanOpaque so)
{
	if (so == NULL || !so->pgturbohybridGraphScan)
		return;

	pgturbohybrid_last_scan_orchestration = PGTURBOHYBRID_SCAN_ORCHESTRATION_GRAPH;
	pgturbohybrid_last_graph_visited_nodes = so->graphVisitedNodes;
	pgturbohybrid_last_graph_scored_codes = so->graphScoredCodes;
	pgturbohybrid_last_graph_batch_scored_codes = so->graphBatchScoredCodes;
	pgturbohybrid_last_graph_scalar_scored_codes = so->graphScalarScoredCodes;
	pgturbohybrid_last_graph_batch_kernel = so->graphBatchKernel;
	pgturbohybrid_last_graph_candidate_count = so->graphCandidateCount;
	pgturbohybrid_last_graph_rescore_band = so->graphRescoreBand;
	pgturbohybrid_last_graph_rescore_count = so->graphRescoreCount;
	pgturbohybrid_last_graph_rescore_pages = so->graphRescorePages;
	pgturbohybrid_last_graph_code_pages_read = so->graphCodePagesRead;
	pgturbohybrid_last_graph_adj_pages_read = so->graphAdjPagesRead;
	pgturbohybrid_last_graph_segment_count = so->graphSegmentCount;
	pgturbohybrid_last_graph_segments_searched = so->graphSegmentsSearched;
	pgturbohybrid_last_graph_per_segment_budget_mode =
		so->graphPerSegmentBudgetMode;
	pgturbohybrid_last_graph_search_ef_before_segment_scaling =
		so->graphSearchEfBeforeSegmentScaling;
	pgturbohybrid_last_graph_search_ef_after_segment_scaling =
		so->graphSearchEfAfterSegmentScaling;
	pgturbohybrid_last_graph_code_page_attempts = so->graphCodePageAttempts;
	pgturbohybrid_last_graph_code_page_hits = so->graphCodePageHits;
	pgturbohybrid_last_graph_code_page_misses = so->graphCodePageMisses;
	pgturbohybrid_last_graph_code_tuples_copied = so->graphCodeTuplesCopied;
	pgturbohybrid_last_graph_code_arena_allocated_bytes = so->graphCodeArenaAllocatedBytes;
	pgturbohybrid_last_graph_code_arena_used_bytes = so->graphCodeArenaUsedBytes;
	pgturbohybrid_last_graph_entry_point_count = so->graphEntryPointCount;
	pgturbohybrid_last_graph_entry_sample_configured =
		so->graphEntrySampleConfigured;
	pgturbohybrid_last_graph_entry_sample_effective =
		so->graphEntrySampleEffective;
	pgturbohybrid_last_graph_entry_sample_scored =
		so->graphEntrySampleScored;
	pgturbohybrid_last_graph_entry_sidecar_count = so->graphEntrySidecarCount;
	pgturbohybrid_last_graph_entry_sidecar_scored = so->graphEntrySidecarScored;
	pgturbohybrid_last_graph_entry_sidecar_selected = so->graphEntrySidecarSelected;
	pgturbohybrid_last_graph_entry_sidecar_representatives_configured =
		so->graphEntrySidecarRepresentativesConfigured;
	pgturbohybrid_last_graph_entry_sidecar_strategy = so->graphEntrySidecarStrategy;
	pgturbohybrid_last_graph_entry_sidecar_us = so->graphEntrySidecarUs;
	pgturbohybrid_last_payload_entry_seeding_mode = so->graphPayloadEntrySeedingMode;
	pgturbohybrid_last_payload_entry_seeding_hit = so->graphPayloadEntrySeedingHit;
	pgturbohybrid_last_payload_entry_seed_count = so->graphPayloadEntrySeedCount;
	pgturbohybrid_last_payload_entry_seed_payload_slot =
		so->graphPayloadEntrySeedPayloadSlot;
	pgturbohybrid_last_payload_entry_seed_range_count =
		so->graphPayloadEntrySeedRangeCount;
	pgturbohybrid_last_payload_entry_seed_us = so->graphPayloadEntrySeedUs;
	pgturbohybrid_last_graph_residual_rerank_count = so->graphResidualRerankCount;
	pgturbohybrid_last_graph_residual_rerank_bytes = so->graphResidualRerankBytes;
	pgturbohybrid_last_graph_residual_rerank_us = so->graphResidualRerankUs;
	pgturbohybrid_last_graph_residual_rerank_mode = so->graphResidualRerankMode;
	pgturbohybrid_last_graph_residual_rerank_weight_effective =
		so->graphResidualRerankWeightEffective;
	pgturbohybrid_last_graph_residual_rerank_band = so->graphResidualRerankBand;
	pgturbohybrid_last_graph_residual_rerank_band_multiplier =
		so->graphResidualRerankBandMultiplier;
	pgturbohybrid_last_graph_residual_rerank_max_adjustment =
		so->graphResidualRerankMaxAdjustment;
	pgturbohybrid_last_graph_residual_rerank_reordered_count =
		so->graphResidualRerankReorderedCount;
	pgturbohybrid_last_graph_residual_rerank_topk_changed =
		so->graphResidualRerankTopKChanged;
	pgturbohybrid_last_graph_heap_rescore_count = so->graphHeapRescoreCount;
	pgturbohybrid_last_graph_heap_fetch_us = so->graphHeapFetchUs;
	pgturbohybrid_last_graph_heap_rescore_us = so->graphHeapRescoreUs;
	pgturbohybrid_last_graph_heap_rescore_mode = so->graphHeapRescoreMode;
	pgturbohybrid_last_graph_heap_rescore_auto_enabled =
		so->graphHeapRescoreAutoEnabled;
	pgturbohybrid_last_graph_heap_rescore_reason = so->graphHeapRescoreReason;
	pgturbohybrid_last_graph_prepare_us = so->graphPrepareUs;
	pgturbohybrid_last_graph_traverse_us = so->graphTraverseUs;
	pgturbohybrid_last_graph_entry_us = so->graphEntryUs;
	pgturbohybrid_last_graph_base_us = so->graphBaseUs;
	pgturbohybrid_last_graph_batch_us = so->graphBatchUs;
	pgturbohybrid_last_graph_heap_us = so->graphHeapUs;
	pgturbohybrid_last_graph_fill_us = so->graphFillUs;
	pgturbohybrid_last_graph_fill_candidate_band_calls =
		so->graphFillCandidateBandCalls;
	pgturbohybrid_last_graph_fill_candidate_band_reason =
		so->graphFillCandidateBandReason;
	pgturbohybrid_last_graph_fill_candidate_band_visited =
		so->graphFillCandidateBandVisited;
	pgturbohybrid_last_graph_fill_candidate_band_scored =
		so->graphFillCandidateBandScored;
	pgturbohybrid_last_graph_fill_candidate_band_selected_before =
		so->graphFillCandidateBandSelectedBefore;
	pgturbohybrid_last_graph_fill_candidate_band_selected_after =
		so->graphFillCandidateBandSelectedAfter;
	pgturbohybrid_last_graph_fill_candidate_band_target =
		so->graphFillCandidateBandTarget;
	pgturbohybrid_last_graph_fill_candidate_band_used_payload_refs =
		so->graphFillCandidateBandUsedPayloadRefs;
	pgturbohybrid_last_graph_fill_candidate_band_payload_ref_count =
		so->graphFillCandidateBandPayloadRefCount;
	pgturbohybrid_last_graph_rescore_us = so->graphRescoreUs;
	pgturbohybrid_last_graph_sort_us = so->graphSortUs;
	pgturbohybrid_last_graph_total_us = so->graphTotalUs;
	pgturbohybrid_last_graph_dense_requested_k = so->graphDenseRequestedK;
	pgturbohybrid_last_graph_effective_result_target = so->graphEffectiveResultTarget;
	pgturbohybrid_last_graph_effective_search_ef = so->graphEffectiveSearchEf;
	pgturbohybrid_last_graph_effective_rescore_band = so->graphEffectiveRescoreBand;
	pgturbohybrid_last_graph_highdim_widening_multiplier = so->graphHighdimWideningMultiplier;
	pgturbohybrid_last_graph_widening_reason = so->graphWideningReason;
	pgturbohybrid_last_dense_filter_unmapped = so->graphDenseFilterUnmapped;
	pgturbohybrid_last_dense_linear_fallback_warning =
		so->graphDenseLinearFallbackWarning;
	pgturbohybrid_last_dense_linear_fallback_ratio =
		so->graphDenseLinearFallbackRatio;
	pgturbohybrid_last_graph_adaptive_widening_mode = so->graphAdaptiveWideningMode;
	pgturbohybrid_last_graph_adaptive_triggered = so->graphAdaptiveTriggered;
	pgturbohybrid_last_graph_adaptive_trigger_reason = so->graphAdaptiveTriggerReason;
	pgturbohybrid_last_graph_adaptive_initial_result_target = so->graphAdaptiveInitialResultTarget;
	pgturbohybrid_last_graph_adaptive_final_result_target = so->graphAdaptiveFinalResultTarget;
	pgturbohybrid_last_graph_adaptive_initial_search_ef = so->graphAdaptiveInitialSearchEf;
	pgturbohybrid_last_graph_adaptive_final_search_ef = so->graphAdaptiveFinalSearchEf;
	pgturbohybrid_last_graph_adaptive_gap_top10 = so->graphAdaptiveGapTop10;
	pgturbohybrid_last_graph_adaptive_gap_boundary = so->graphAdaptiveGapBoundary;
	pgturbohybrid_last_graph_uncertainty_retry_mode = so->graphUncertaintyRetryMode;
	pgturbohybrid_last_graph_uncertainty_retry_triggered =
		so->graphUncertaintyRetryTriggered;
	pgturbohybrid_last_graph_uncertainty_retry_reason =
		so->graphUncertaintyRetryReason;
	pgturbohybrid_last_graph_uncertainty_retry_passes =
		so->graphUncertaintyRetryPasses;
	pgturbohybrid_last_graph_uncertainty_initial_result_target =
		so->graphUncertaintyInitialResultTarget;
	pgturbohybrid_last_graph_uncertainty_final_result_target =
		so->graphUncertaintyFinalResultTarget;
	pgturbohybrid_last_graph_uncertainty_initial_search_ef =
		so->graphUncertaintyInitialSearchEf;
	pgturbohybrid_last_graph_uncertainty_final_search_ef =
		so->graphUncertaintyFinalSearchEf;
	pgturbohybrid_last_graph_uncertainty_gap_top10 =
		so->graphUncertaintyGapTop10;
	pgturbohybrid_last_graph_uncertainty_gap_boundary =
		so->graphUncertaintyGapBoundary;
	pgturbohybrid_last_graph_local_expansion_mode = so->graphLocalExpansionMode;
	pgturbohybrid_last_graph_local_expansion_triggered = so->graphLocalExpansionTriggered;
	pgturbohybrid_last_graph_local_expansion_seed_count = so->graphLocalExpansionSeedCount;
	pgturbohybrid_last_graph_local_expansion_neighbors_scored = so->graphLocalExpansionNeighborsScored;
	pgturbohybrid_last_graph_local_expansion_candidates_added = so->graphLocalExpansionCandidatesAdded;
	pgturbohybrid_last_graph_local_expansion_us = so->graphLocalExpansionUs;
	pgturbohybrid_last_graph_dense_budget_policy = so->graphDenseBudgetPolicy;
	pgturbohybrid_last_graph_rescore_band_policy = so->graphRescoreBandPolicy;
	pgturbohybrid_last_graph_exact_rescore_source = so->graphExactRescoreSource;
	pgturbohybrid_last_graph_scoring_kernel = so->tq.enabled ? so->tq.scoringKernel : PGTURBOHYBRID_SCORING_SCALAR;
	memcpy(pgturbohybrid_last_graph_score_kernel_nodes, so->graphScoreKernelNodes,
		   sizeof(pgturbohybrid_last_graph_score_kernel_nodes));
	memcpy(pgturbohybrid_last_graph_score_kernel_calls, so->graphScoreKernelCalls,
		   sizeof(pgturbohybrid_last_graph_score_kernel_calls));
	pgturbohybrid_last_graph_batch_calls = so->graphBatchCalls;
	pgturbohybrid_last_graph_batch_nodes = so->graphBatchNodes;
	pgturbohybrid_last_graph_base_frontier_pushes = so->graphBaseFrontierPushes;
	pgturbohybrid_last_graph_base_frontier_pops = so->graphBaseFrontierPops;
	pgturbohybrid_last_graph_base_nearest_offers = so->graphBaseNearestOffers;
	pgturbohybrid_last_graph_base_visited_checks = so->graphBaseVisitedChecks;
	pgturbohybrid_last_graph_base_duplicate_skips = so->graphBaseDuplicateSkips;
	pgturbohybrid_last_graph_base_batch_calls = so->graphBaseBatchCalls;
	pgturbohybrid_last_graph_base_batch_nodes = so->graphBaseBatchNodes;
	pgturbohybrid_last_graph_base_max_frontier = so->graphBaseMaxFrontier;
	pgturbohybrid_last_graph_score_mode = so->tq.enabled ? so->tq.scoreMode : PGTURBOHYBRID_SCORE_L2;
	pgturbohybrid_last_graph_simd_force = pgturbohybrid_dense_simd_force;
	if (!pgturbohybrid_last_exact_vector_kernel_recorded &&
		(so->graphRescoreCount > 0 || so->graphHeapRescoreCount > 0))
		pgturbohybrid_last_exact_vector_kernel = TqExpectedExactKernel();
	pgturbohybrid_last_graph_storage_kind = so->graphStorageKind;
	pgturbohybrid_last_graph_quantization_bits = so->tq.enabled ? so->tq.bits : 0;
	pgturbohybrid_last_graph_dimensions = so->tq.enabled ? so->tq.dimensions : 0;
	pgturbohybrid_last_graph_returned_rows = so->returnedRows;
	pgturbohybrid_last_graph_m = so->graphM;
	pgturbohybrid_last_graph_ef_construction = so->graphEfConstruction;
	pgturbohybrid_last_graph_ef_search = so->efSearch;
	pgturbohybrid_last_graph_oversampling = so->graphOversampling;
	pgturbohybrid_last_graph_exact_cache = so->graphExactCache;
	pgturbohybrid_last_graph_exact_storage = so->graphExactStorage;
	pgturbohybrid_last_graph_exact_storage_known = so->tq.enabled;
	pgturbohybrid_last_graph_build_exact_distances = so->graphBuildExactDistances;
	pgturbohybrid_last_graph_build_distance_mode = so->graphBuildDistanceMode;
	pgturbohybrid_last_graph_build_fast_edges = so->graphBuildFastEdges;
	pgturbohybrid_last_graph_build_neighbor_select_reason =
		so->graphBuildNeighborSelectReason;
	if (so->hasTupleTargetRows && so->tupleTargetRows > 0)
	{
		int			limit = (int) Min(so->tupleTargetRows, (int64) INT_MAX);

		pgturbohybrid_last_final_k_requested = 0;
		pgturbohybrid_last_final_k_effective = limit;
		pgturbohybrid_last_sql_limit = limit;
		pgturbohybrid_last_final_k_inferred = true;
	}
	else
	{
		pgturbohybrid_last_final_k_requested = 0;
		pgturbohybrid_last_final_k_effective = PGTURBOHYBRID_DEFAULT_FINAL_K;
		pgturbohybrid_last_sql_limit = 0;
		pgturbohybrid_last_final_k_inferred = false;
	}
#if defined(__aarch64__) || defined(_M_ARM64) || \
	defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
	pgturbohybrid_last_graph_query_split_active = so->tq.enabled && so->tq.signedSplit.enabled;
	pgturbohybrid_last_graph_u8_split_used = so->tq.enabled && so->tq.u8.enabled;
#else
	pgturbohybrid_last_graph_query_split_active = false;
	pgturbohybrid_last_graph_u8_split_used = false;
#endif
	/* U8 batch path and whole-code prefetch state for this scan. */
	pgturbohybrid_last_dense_u8_batch_x4_enabled = pgturbohybrid_dense_u8_batch_x4;
	pgturbohybrid_last_graph_u8_batch_mode = so->graphU8BatchMode;
	/* Exact u8 kernels resolved at query prep for this scan. */
	pgturbohybrid_last_graph_u8_kernel_single = so->tq.enabled ? so->tq.u8.kernelSingle : PGTURBOHYBRID_U8_KERNEL_NONE;
	pgturbohybrid_last_graph_u8_kernel_batch = so->tq.enabled ? so->tq.u8.kernelBatch : PGTURBOHYBRID_U8_KERNEL_NONE;
	pgturbohybrid_last_graph_large_code_arena = so->graphLargeCodeArena;
	pgturbohybrid_last_graph_whole_code_prefetch_active =
		pgturbohybrid_dense_graph_prefetch && so->graphLargeCodeArena;
	pgturbohybrid_last_graph_code_bytes = so->tq.enabled ? (int64) so->tq.codeBytes : 0;
	pgturbohybrid_last_graph_code_arena_estimated_bytes = so->graphCodeArenaEstimatedBytes;
	pgturbohybrid_last_graph_native_cache_mode = so->graphNativeCacheMode;
	pgturbohybrid_last_graph_native_cache_policy = so->graphNativeCachePolicy;
	pgturbohybrid_last_graph_native_cache_reason = so->graphNativeCacheReason;
	pgturbohybrid_last_graph_native_cache_used = so->graphNativeCacheUsed;
	pgturbohybrid_last_graph_native_cache_reused = so->graphNativeCacheReused;
	pgturbohybrid_last_graph_native_cache_built_this_scan = so->graphNativeCacheBuiltThisScan;
	pgturbohybrid_last_graph_native_cache_attach_us = so->graphNativeCacheAttachUs;
	pgturbohybrid_last_graph_native_cache_build_us = so->graphNativeCacheBuildUs;
	pgturbohybrid_last_graph_native_cache_wait_us = so->graphNativeCacheWaitUs;
	pgturbohybrid_last_graph_native_cache_refcount = so->graphNativeCacheRefcount;
	pgturbohybrid_last_graph_native_cache_bytes = so->graphNativeCacheBytes;
	pgturbohybrid_last_graph_native_cache_code_bytes = so->graphNativeCacheCodeBytes;
	pgturbohybrid_last_graph_native_cache_adj_bytes = so->graphNativeCacheAdjBytes;
	pgturbohybrid_last_graph_native_cache_exact_bytes = so->graphNativeCacheExactBytes;
	pgturbohybrid_last_graph_native_cache_warning = so->graphNativeCacheWarning;
	pgturbohybrid_last_graph_native_cache_warning_reason =
		so->graphNativeCacheWarningReason != NULL ?
		so->graphNativeCacheWarningReason : "none";
	pgturbohybrid_last_graph_scan_lock_wait_us = so->graphScanLockWaitUs;
	pgturbohybrid_last_graph_code_buffer_lock_wait_us = so->graphCodeBufferLockWaitUs;
	pgturbohybrid_last_graph_adj_buffer_lock_wait_us = so->graphAdjBufferLockWaitUs;
	/*
	 * Whether the integer query-split scorer will actually run for this query
	 * (full gate incl. dim >= 1024 and runtime SIMD availability), as opposed
	 * to signedSplit.enabled which only reflects the per-query prep.  Used to
	 * report the exact approximate scorer (query split vs LUT gather).
	 */
	pgturbohybrid_last_graph_querysplit_used = PgturbohybridGraphTqQuerySplitActive(&so->tq);
	pgturbohybrid_last_exact_vector_kernel_recorded = false;
}

void
PgturbohybridGraphRecordNonGraphScanStats(void)
{
	pgturbohybrid_last_scan_orchestration = PGTURBOHYBRID_SCAN_ORCHESTRATION_NONE;
	pgturbohybrid_last_graph_visited_nodes = 0;
	pgturbohybrid_last_graph_scored_codes = 0;
	pgturbohybrid_last_graph_batch_scored_codes = 0;
	pgturbohybrid_last_graph_scalar_scored_codes = 0;
	pgturbohybrid_last_graph_batch_kernel = PGTURBOHYBRID_SCORING_SCALAR;
	pgturbohybrid_last_weighted_code_code_kernel = PGTURBOHYBRID_SCORING_SCALAR;
	memset(pgturbohybrid_last_graph_score_kernel_nodes, 0,
		   sizeof(pgturbohybrid_last_graph_score_kernel_nodes));
	memset(pgturbohybrid_last_graph_score_kernel_calls, 0,
		   sizeof(pgturbohybrid_last_graph_score_kernel_calls));
	pgturbohybrid_last_graph_batch_calls = 0;
	pgturbohybrid_last_graph_batch_nodes = 0;
	pgturbohybrid_last_graph_base_frontier_pushes = 0;
	pgturbohybrid_last_graph_base_frontier_pops = 0;
	pgturbohybrid_last_graph_base_nearest_offers = 0;
	pgturbohybrid_last_graph_base_visited_checks = 0;
	pgturbohybrid_last_graph_base_duplicate_skips = 0;
	pgturbohybrid_last_graph_base_batch_calls = 0;
	pgturbohybrid_last_graph_base_batch_nodes = 0;
	pgturbohybrid_last_graph_base_max_frontier = 0;
	pgturbohybrid_last_graph_score_mode = PGTURBOHYBRID_SCORE_L2;
	pgturbohybrid_last_graph_simd_force = pgturbohybrid_dense_simd_force;
	pgturbohybrid_last_graph_candidate_count = 0;
	pgturbohybrid_last_graph_rescore_band = PGTURBOHYBRID_GRAPH_RESCORE_BAND_AUTO;
	pgturbohybrid_last_graph_rescore_count = 0;
	pgturbohybrid_last_graph_rescore_pages = 0;
	pgturbohybrid_last_graph_code_pages_read = 0;
	pgturbohybrid_last_graph_adj_pages_read = 0;
	pgturbohybrid_last_graph_segment_count = 0;
	pgturbohybrid_last_graph_segments_searched = 0;
	pgturbohybrid_last_graph_per_segment_budget_mode =
		PGTURBOHYBRID_NATIVE_SEGMENT_BUDGET_OFF;
	pgturbohybrid_last_graph_search_ef_before_segment_scaling = 0;
	pgturbohybrid_last_graph_search_ef_after_segment_scaling = 0;
	pgturbohybrid_last_graph_code_page_attempts = 0;
	pgturbohybrid_last_graph_code_page_hits = 0;
	pgturbohybrid_last_graph_code_page_misses = 0;
	pgturbohybrid_last_graph_code_tuples_copied = 0;
	pgturbohybrid_last_graph_code_arena_allocated_bytes = 0;
	pgturbohybrid_last_graph_code_arena_used_bytes = 0;
	pgturbohybrid_last_graph_entry_point_count = 0;
	pgturbohybrid_last_graph_entry_sample_configured = 0;
	pgturbohybrid_last_graph_entry_sample_effective = 0;
	pgturbohybrid_last_graph_entry_sample_scored = 0;
	pgturbohybrid_last_graph_entry_sidecar_count = 0;
	pgturbohybrid_last_graph_entry_sidecar_scored = 0;
	pgturbohybrid_last_graph_entry_sidecar_selected = 0;
	pgturbohybrid_last_graph_entry_sidecar_representatives_configured = 0;
	pgturbohybrid_last_graph_entry_sidecar_strategy =
		PGTURBOHYBRID_DEFAULT_ENTRY_SIDECAR_STRATEGY;
	pgturbohybrid_last_graph_entry_sidecar_us = 0;
	pgturbohybrid_last_payload_entry_seeding_mode =
		pgturbohybrid_payload_entry_seeding;
	pgturbohybrid_last_payload_entry_seeding_hit = false;
	pgturbohybrid_last_payload_entry_seed_count = 0;
	pgturbohybrid_last_payload_entry_seed_payload_slot = -1;
	pgturbohybrid_last_payload_entry_seed_range_count = 0;
	pgturbohybrid_last_payload_entry_seed_us = 0;
		pgturbohybrid_last_graph_residual_rerank_count = 0;
		pgturbohybrid_last_graph_residual_rerank_bytes = 0;
		pgturbohybrid_last_graph_residual_rerank_us = 0;
		pgturbohybrid_last_graph_residual_rerank_mode =
			pgturbohybrid_dense_residual_rerank_mode;
		pgturbohybrid_last_graph_residual_rerank_weight_effective = 0.0;
		pgturbohybrid_last_graph_residual_rerank_band = 0;
		pgturbohybrid_last_graph_residual_rerank_band_multiplier = 0;
		pgturbohybrid_last_graph_residual_rerank_max_adjustment = 0.0;
		pgturbohybrid_last_graph_residual_rerank_reordered_count = 0;
		pgturbohybrid_last_graph_residual_rerank_topk_changed = false;
		pgturbohybrid_last_graph_heap_rescore_count = 0;
		pgturbohybrid_last_graph_heap_fetch_us = 0;
		pgturbohybrid_last_graph_heap_rescore_us = 0;
		pgturbohybrid_last_graph_heap_rescore_mode = PGTURBOHYBRID_DENSE_HEAP_RESCORE_OFF;
		pgturbohybrid_last_graph_heap_rescore_auto_enabled = false;
		pgturbohybrid_last_graph_heap_rescore_reason = PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_UNKNOWN;
		pgturbohybrid_last_graph_prepare_us = 0;
	pgturbohybrid_last_graph_traverse_us = 0;
	pgturbohybrid_last_graph_entry_us = 0;
	pgturbohybrid_last_graph_base_us = 0;
	pgturbohybrid_last_graph_batch_us = 0;
	pgturbohybrid_last_graph_heap_us = 0;
	pgturbohybrid_last_graph_fill_us = 0;
	pgturbohybrid_last_graph_fill_candidate_band_calls = 0;
	pgturbohybrid_last_graph_fill_candidate_band_reason =
		PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_NONE;
	pgturbohybrid_last_graph_fill_candidate_band_visited = 0;
	pgturbohybrid_last_graph_fill_candidate_band_scored = 0;
	pgturbohybrid_last_graph_fill_candidate_band_selected_before = 0;
	pgturbohybrid_last_graph_fill_candidate_band_selected_after = 0;
	pgturbohybrid_last_graph_fill_candidate_band_target = 0;
	pgturbohybrid_last_graph_fill_candidate_band_used_payload_refs = false;
	pgturbohybrid_last_graph_fill_candidate_band_payload_ref_count = 0;
	pgturbohybrid_last_graph_rescore_us = 0;
	pgturbohybrid_last_graph_sort_us = 0;
	pgturbohybrid_last_graph_total_us = 0;
	pgturbohybrid_last_graph_dense_requested_k = 0;
	pgturbohybrid_last_graph_effective_result_target = 0;
	pgturbohybrid_last_graph_effective_search_ef = 0;
	pgturbohybrid_last_graph_effective_rescore_band = 0;
	pgturbohybrid_last_graph_highdim_widening_multiplier = 1.0;
	pgturbohybrid_last_graph_widening_reason = PGTURBOHYBRID_DENSE_WIDENING_NONE;
	pgturbohybrid_last_dense_filter_unmapped = false;
	pgturbohybrid_last_dense_linear_fallback_warning = false;
	pgturbohybrid_last_dense_linear_fallback_ratio = 0.0;
	pgturbohybrid_last_graph_adaptive_widening_mode = PGTURBOHYBRID_DENSE_ADAPTIVE_WIDENING_OFF;
	pgturbohybrid_last_graph_adaptive_triggered = false;
	pgturbohybrid_last_graph_adaptive_trigger_reason = PGTURBOHYBRID_DENSE_ADAPTIVE_REASON_NONE;
	pgturbohybrid_last_graph_adaptive_initial_result_target = 0;
	pgturbohybrid_last_graph_adaptive_final_result_target = 0;
	pgturbohybrid_last_graph_adaptive_initial_search_ef = 0;
	pgturbohybrid_last_graph_adaptive_final_search_ef = 0;
	pgturbohybrid_last_graph_adaptive_gap_top10 = 0.0;
	pgturbohybrid_last_graph_adaptive_gap_boundary = 0.0;
	pgturbohybrid_last_graph_uncertainty_retry_mode =
		pgturbohybrid_dense_uncertainty_retry;
	pgturbohybrid_last_graph_uncertainty_retry_triggered = false;
	pgturbohybrid_last_graph_uncertainty_retry_reason =
		PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_NONE;
	pgturbohybrid_last_graph_uncertainty_retry_passes = 0;
	pgturbohybrid_last_graph_uncertainty_initial_result_target = 0;
	pgturbohybrid_last_graph_uncertainty_final_result_target = 0;
	pgturbohybrid_last_graph_uncertainty_initial_search_ef = 0;
	pgturbohybrid_last_graph_uncertainty_final_search_ef = 0;
	pgturbohybrid_last_graph_uncertainty_gap_top10 = 0.0;
	pgturbohybrid_last_graph_uncertainty_gap_boundary = 0.0;
	pgturbohybrid_last_graph_local_expansion_mode = PGTURBOHYBRID_DENSE_LOCAL_EXPANSION_OFF;
	pgturbohybrid_last_graph_local_expansion_triggered = false;
	pgturbohybrid_last_graph_local_expansion_seed_count = 0;
	pgturbohybrid_last_graph_local_expansion_neighbors_scored = 0;
	pgturbohybrid_last_graph_local_expansion_candidates_added = 0;
	pgturbohybrid_last_graph_local_expansion_us = 0;
		pgturbohybrid_last_graph_dense_budget_policy = PGTURBOHYBRID_DENSE_BUDGET_AUTO;
		pgturbohybrid_last_graph_rescore_band_policy = PGTURBOHYBRID_RESCORE_BAND_POLICY_AUTO;
		pgturbohybrid_last_graph_exact_rescore_source = PGTURBOHYBRID_EXACT_RESCORE_SOURCE_NONE;
		pgturbohybrid_last_graph_scoring_kernel = PGTURBOHYBRID_SCORING_SCALAR;
	pgturbohybrid_last_exact_vector_kernel = PGTURBOHYBRID_EXACT_KERNEL_SCALAR;
	pgturbohybrid_last_exact_vector_kernel_recorded = false;
	pgturbohybrid_last_graph_storage_kind = PGTURBOHYBRID_GRAPH_STORAGE_GRAPH;
	pgturbohybrid_last_graph_quantization_bits = 0;
	pgturbohybrid_last_graph_exact_storage = false;
	pgturbohybrid_last_graph_exact_storage_known = false;
	pgturbohybrid_last_graph_build_exact_distances = false;
	pgturbohybrid_last_graph_build_distance_mode = PGTURBOHYBRID_DENSE_BUILD_DISTANCE_CODE;
	pgturbohybrid_last_graph_build_fast_edges = false;
	pgturbohybrid_last_graph_build_neighbor_select_reason =
		PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_UNKNOWN;
	pgturbohybrid_last_graph_query_split_active = false;
	pgturbohybrid_last_graph_querysplit_used = false;
	pgturbohybrid_last_graph_u8_split_used = false;
	pgturbohybrid_last_dense_u8_batch_x4_enabled = pgturbohybrid_dense_u8_batch_x4;
	pgturbohybrid_last_graph_u8_batch_mode = PGTURBOHYBRID_U8_BATCH_NONE;
	pgturbohybrid_last_graph_u8_kernel_single = PGTURBOHYBRID_U8_KERNEL_NONE;
	pgturbohybrid_last_graph_u8_kernel_batch = PGTURBOHYBRID_U8_KERNEL_NONE;
	pgturbohybrid_last_graph_large_code_arena = false;
	pgturbohybrid_last_graph_whole_code_prefetch_active = false;
	pgturbohybrid_last_graph_code_bytes = 0;
	pgturbohybrid_last_graph_code_arena_estimated_bytes = 0;
	pgturbohybrid_last_graph_native_cache_mode = PGTURBOHYBRID_GRAPH_NATIVE_CACHE_NONE;
	pgturbohybrid_last_graph_native_cache_policy = pgturbohybrid_native_cache_policy;
	pgturbohybrid_last_graph_native_cache_reason = PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_NONE;
	pgturbohybrid_last_graph_native_cache_used = false;
	pgturbohybrid_last_graph_native_cache_reused = false;
	pgturbohybrid_last_graph_native_cache_built_this_scan = false;
	pgturbohybrid_last_graph_native_cache_attach_us = 0;
	pgturbohybrid_last_graph_native_cache_build_us = 0;
	pgturbohybrid_last_graph_native_cache_wait_us = 0;
	pgturbohybrid_last_graph_native_cache_refcount = -1;
	pgturbohybrid_last_graph_native_cache_bytes = 0;
	pgturbohybrid_last_graph_native_cache_code_bytes = 0;
	pgturbohybrid_last_graph_native_cache_adj_bytes = 0;
	pgturbohybrid_last_graph_native_cache_exact_bytes = 0;
	pgturbohybrid_last_graph_native_cache_warning = false;
	pgturbohybrid_last_graph_native_cache_warning_reason = "none";
	pgturbohybrid_last_graph_scan_lock_wait_us = 0;
	pgturbohybrid_last_graph_code_buffer_lock_wait_us = 0;
	pgturbohybrid_last_graph_adj_buffer_lock_wait_us = 0;
	pgturbohybrid_last_graph_dimensions = 0;
	pgturbohybrid_last_graph_returned_rows = 0;
	pgturbohybrid_last_graph_m = 0;
	pgturbohybrid_last_graph_ef_construction = 0;
	pgturbohybrid_last_graph_ef_search = 0;
	pgturbohybrid_last_graph_oversampling = 0;
	pgturbohybrid_last_graph_exact_cache = PGTURBOHYBRID_GRAPH_EXACT_CACHE_AUTO;
}

void
PgturbohybridGraphRecordFlatScanStats(void)
{
	PgturbohybridGraphRecordNonGraphScanStats();
	pgturbohybrid_last_scan_orchestration = PGTURBOHYBRID_SCAN_ORCHESTRATION_FLAT;
	pgturbohybrid_last_graph_storage_kind = PGTURBOHYBRID_GRAPH_STORAGE_QUANT_FLAT;
}

static const char *
PgturbohybridGraphNativeCacheModeName(int mode)
{
	switch ((PgturbohybridGraphNativeCacheMode) mode)
	{
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_SHARED:
			return "shared";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_PER_BACKEND:
			return "per_backend";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_UNCACHED:
			return "uncached";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_NONE:
		default:
			return "none";
	}
}

static const char *
PgturbohybridGraphNativeCacheScopeName(int mode)
{
	switch ((PgturbohybridGraphNativeCacheMode) mode)
	{
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_SHARED:
			return "shared";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_PER_BACKEND:
			return "per_backend";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_UNCACHED:
			return "per_scan";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_NONE:
		default:
			return "none";
	}
}

static const char *
PgturbohybridNativeCachePolicyName(int policy)
{
	switch ((PgturbohybridNativeCachePolicy) policy)
	{
		case PGTURBOHYBRID_NATIVE_CACHE_POLICY_SHARED:
			return "shared";
		case PGTURBOHYBRID_NATIVE_CACHE_POLICY_PER_BACKEND:
			return "per_backend";
		case PGTURBOHYBRID_NATIVE_CACHE_POLICY_OFF:
			return "off";
		case PGTURBOHYBRID_NATIVE_CACHE_POLICY_AUTO:
		default:
			return "auto";
	}
}

static const char *
PgturbohybridGraphNativeCacheReasonName(int reason)
{
	switch ((PgturbohybridGraphNativeCacheReason) reason)
	{
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_SHARED_FITS_MAX_MB:
			return "shared_fits_max_mb";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_AUTO_FITS_MAX_MB:
			return "auto_fits_max_mb";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_PER_BACKEND_FITS_MAX_MB:
			return "per_backend_fits_max_mb";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_POLICY_OFF:
			return "policy_off";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_EXCEEDS_MAX_MB:
			return "exceeds_max_mb";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_SHARED_BUILD_TIMEOUT:
			return "shared_build_timeout";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_SHARED_ATTACH_FAILED:
			return "shared_attach_failed";
		case PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_NONE:
		default:
			return "none";
	}
}

static const char *
PgturbohybridGraphFillCandidateBandReasonName(int reason)
{
	switch ((PgturbohybridGraphFillCandidateBandReason) reason)
	{
		case PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_UNDERFILLED_FULL_TARGET:
			return "underfilled_full_target";
		case PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_ESTIMATED_SELECTIVITY:
			return "estimated_selectivity";
		case PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_ADAPTIVE_WIDENING:
			return "adaptive_widening";
		case PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_TIGHT_L2_EXACT_POLICY:
			return "tight_l2_exact_policy";
		case PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_PAYLOAD_EXACT_BAND_MISS:
			return "payload_exact_band_miss";
		case PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_UNKNOWN:
			return "unknown";
		case PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_NONE:
		default:
			return "none";
	}
}

static const char *
PgturbohybridGraphU8BatchModeName(int mode)
{
	switch (mode)
	{
		case PGTURBOHYBRID_U8_BATCH_X4:
			return "x4";
		case PGTURBOHYBRID_U8_BATCH_SINGLE:
			return "single";
		case PGTURBOHYBRID_U8_BATCH_NONE:
		default:
			return "none";
	}
}

/* Name of the exact u8 kernel resolved at query prep (TqU8Kernel). */
static const char *
PgturbohybridGraphU8KernelName(int kernel)
{
	switch (kernel)
	{
		case PGTURBOHYBRID_U8_KERNEL_AVX2_SINGLE:
			return "avx2_single";
		case PGTURBOHYBRID_U8_KERNEL_AVX2_X4:
			return "avx2_x4";
		case PGTURBOHYBRID_U8_KERNEL_AVX512VNNI_SINGLE:
			return "avx512vnni_single";
		case PGTURBOHYBRID_U8_KERNEL_AVX512VNNI_X4:
			return "avx512vnni_x4";
		case PGTURBOHYBRID_U8_KERNEL_NONE:
		default:
			return "none";
	}
}

static const char *
TqScanOrchestrationName(void)
{
	switch (pgturbohybrid_last_scan_orchestration)
	{
		case PGTURBOHYBRID_SCAN_ORCHESTRATION_GRAPH:
			return pgturbohybrid_last_graph_storage_kind == PGTURBOHYBRID_GRAPH_STORAGE_QUANT_GRAPH_NATIVE ? "graph_native" : "graph_hnsw";
		case PGTURBOHYBRID_SCAN_ORCHESTRATION_FLAT:
			return "flat";
		case PGTURBOHYBRID_SCAN_ORCHESTRATION_NONE:
		default:
			return "none";
	}
}

static const char *
PgturbohybridFinalKSourceName(void)
{
	if (pgturbohybrid_last_final_k_requested > 0)
		return "explicit";
	if (pgturbohybrid_last_final_k_inferred)
		return "limit";
	return "default";
}

static const char *
TqExactKernelName(int kernel)
{
	switch ((TqExactKernel) kernel)
	{
		case PGTURBOHYBRID_EXACT_KERNEL_SCALAR:
			return "scalar";
		case PGTURBOHYBRID_EXACT_KERNEL_AUTOVEC_FMA:
			return "autovec_fma";
		case PGTURBOHYBRID_EXACT_KERNEL_NEON:
			return "neon";
		case PGTURBOHYBRID_EXACT_KERNEL_AVX2:
			return "avx2";
		default:
			return "unknown";
	}
}

static const char *
PgturbohybridGraphAvx512WeightedModeName(int mode)
{
	switch ((PgturbohybridGraphAvx512WeightedMode) mode)
	{
		case PGTURBOHYBRID_GRAPH_AVX512_WEIGHTED_ON:
			return "on";
		case PGTURBOHYBRID_GRAPH_AVX512_WEIGHTED_AUTO:
			return "auto";
		case PGTURBOHYBRID_GRAPH_AVX512_WEIGHTED_OFF:
		default:
			return "off";
	}
}

#ifdef PGTURBOHYBRID_DEV_DIAGNOSTICS
static Datum pgturbohybrid_last_simd_stats(PG_FUNCTION_ARGS) pg_attribute_unused();
#endif

/*
 * Name of the approximate dense scorer (the query-split SIMD kernel selected
 * for batch and single-node code scoring).  Uses the diagnostic taxonomy:
 * "query_split_avx2" disambiguates the int8 query-split AVX2 path from the
 * "avx2_lut_gather" scalar-fallback path reported separately below.
 */
static const char *
PgturbohybridDenseScorerName(int scoringKernel)
{
	if (pgturbohybrid_last_graph_quantization_bits == 8)
		return "scalar_8bit";

	switch ((TqScoringKernel) scoringKernel)
	{
		case PGTURBOHYBRID_SCORING_AVX512VNNI:
			return "avx512vnni";
		case PGTURBOHYBRID_SCORING_AVXVNNI:
			return "avxvnni";
		case PGTURBOHYBRID_SCORING_AVX512BW_DQ:
			return "avx512bw_dq";
		case PGTURBOHYBRID_SCORING_AVX2:
			return "query_split_avx2";
		case PGTURBOHYBRID_SCORING_ARM_I8MM:
			return "arm_i8mm";
		case PGTURBOHYBRID_SCORING_NEON:
			return "neon";
		case PGTURBOHYBRID_SCORING_SCALAR:
		default:
			return "scalar";
	}
}

/*
 * Name of the fallback scorer used for codes that miss the query-split path
 * (single-node / sub-batch-of-4 remainders).  On x86 this is the per-dimension
 * AVX2 LUT-gather kernel (TqCodeDistanceAvx2); otherwise the scalar kernel.
 * A nonzero graph_scalar_scored_codes means this slow path actually ran.
 */
static const char *
PgturbohybridDenseScalarFallbackName(void)
{
	if (pgturbohybrid_last_graph_quantization_bits == 8)
		return "scalar_8bit";

	if (pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_SCALAR)
		return "scalar";
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && (defined(__x86_64__) || defined(_M_X64))
	return "avx2_lut_gather";
#else
	return "scalar";
#endif
}

/*
 * The exact approximate-distance scorer that the dense scan actually used.
 * Distinguishes the scalar/LUT and AVX2 LUT-gather paths from the integer
 * signed query-split kernels (and from a pure exact-function rescore).  The
 * qdrant_u8_* taxonomy values are reserved for a future unsigned-codebook
 * kernel and are intentionally never emitted by the current implementation.
 */
static const char *
PgturbohybridDenseScorerUsedName(bool querySplitUsed, int scoringKernel,
								 int64 approxCodes, int64 rescoreCount)
{
	if (approxCodes == 0 && rescoreCount > 0)
		return "exact_function_call";

	if (pgturbohybrid_last_graph_quantization_bits == 8)
		return "scalar_8bit";

	if (querySplitUsed)
	{
		switch ((TqScoringKernel) scoringKernel)
		{
			case PGTURBOHYBRID_SCORING_AVX512VNNI:
				return "signed_split_avx512vnni";
			case PGTURBOHYBRID_SCORING_AVXVNNI:
				return "signed_split_avxvnni";
			case PGTURBOHYBRID_SCORING_NEON:
			case PGTURBOHYBRID_SCORING_ARM_I8MM:
				return "signed_split_neon";
			case PGTURBOHYBRID_SCORING_AVX2:
			default:
				return "signed_split_avx2";
		}
	}

	/* Approximate scoring fell back to the per-dimension LUT path. */
	if ((TqScoringKernel) scoringKernel == PGTURBOHYBRID_SCORING_SCALAR)
		return "scalar_lut";
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && (defined(__x86_64__) || defined(_M_X64))
	return "avx2_lut_gather";
#else
	return "scalar_lut";
#endif
}

static const char *
PgturbohybridGraphExactCacheName(int mode)
{
	switch ((PgturbohybridGraphExactCache) mode)
	{
		case PGTURBOHYBRID_GRAPH_EXACT_CACHE_ON:
			return "on";
		case PGTURBOHYBRID_GRAPH_EXACT_CACHE_OFF:
			return "off";
		case PGTURBOHYBRID_GRAPH_EXACT_CACHE_AUTO:
		default:
			return "auto";
	}
}

/*
 * Consolidated last-scan stats view.
 *
 * turbohybrid_last_scan_stats() historically emitted one flat bag of ~100 keys
 * read straight from the pgturbohybrid_last_graph_* globals.  As hot-path flags
 * accumulated (the u8 x4 batch mode, whole-code prefetch, the once-resolved u8
 * kernels) it grew easy to add a scan-opaque counter and forget to surface it.
 *
 * TqLastScanStats gives every reported value a single, sectioned home: a dense
 * summary with kernels / cache / traversal / timing_us sub-sections, plus bm25,
 * fusion and query.  PgturbohybridCollectLastScanStats() is the one place that
 * gathers the values (from the globals, the scan-stats snapshot and the GUCs);
 * PgturbohybridEmitNestedScanStats() renders them under the "dense", "bm25",
 * "fusion" and "query" keys.  The legacy flat keys are still emitted verbatim
 * above for backwards compatibility -- the nested sections are an additional
 * grouped view built from the same numbers, so a new dense.* / bm25.* / query.*
 * field has an obvious place to live and the pgturbohybrid_x4_safety regression
 * asserts the hot-path flags appear (and agree with their flat counterparts).
 */
typedef struct TqLastScanKernels
{
	int			scoringKernel;
	int			batchKernel;
	int			exactVectorKernel;
	bool		u8SplitUsed;
	bool		u8BatchX4Enabled;
	int			u8BatchMode;
	int			u8KernelSingle;
	int			u8KernelBatch;
	bool		querySplitUsed;		/* inputs to the derived dense_scorer name */
	int64		scoredCodes;
	int64		rescoreCount;
	int64		batchScoredCodes;
	int64		scalarScoredCodes;
	int64		batchCalls;
	int64		batchNodes;
	int64		kernelNodes[PGTURBOHYBRID_SCORE_KERNEL_BUCKET_COUNT];
	int64		kernelCalls[PGTURBOHYBRID_SCORE_KERNEL_BUCKET_COUNT];
} TqLastScanKernels;

typedef struct TqLastScanCache
{
	int64		loadAttempts;
	int64		cacheHits;
	int64		cacheMisses;
	int64		codePagesRead;
	int64		adjPagesRead;
	int64		codeTuplesCopied;
	int64		arenaUsedBytes;
	int64		arenaAllocatedBytes;
	int64		scoredCodes;		/* denominator for pages_read_per_scored_code */
	int64		codeBytes;
	int64		codeArenaEstimatedBytes;
	bool		largeCodeArena;
	bool		wholeCodePrefetchActive;
	/* Per-backend native scan-cache provenance (see PgturbohybridGraphNativeCacheMode). */
	int			nativeCacheMode;
	int			nativeCachePolicy;
	int			nativeCacheReason;
	bool		nativeCacheUsed;
	bool		nativeCacheReused;
	bool		nativeCacheBuiltThisScan;
	int64		nativeCacheAttachUs;
	int64		nativeCacheBuildUs;
	int64		nativeCacheWaitUs;
	int64		nativeCacheRefcount;
	int64		nativeCacheBytes;
	int64		nativeCacheCodeBytes;
	int64		nativeCacheAdjBytes;
	int64		nativeCacheExactBytes;
	bool		nativeCacheWarning;
	const char *nativeCacheWarningReason;
	int64		codeBufferLockWaitUs;
	int64		adjBufferLockWaitUs;
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
} TqLastScanCache;

typedef struct TqLastScanTraversal
{
	int64		visitedNodes;
	int64		scoredCodes;
	int64		candidateCount;
	int64		fillCandidateBandCalls;
	int			fillCandidateBandReason;
	int64		fillCandidateBandVisited;
	int64		fillCandidateBandScored;
	int64		fillCandidateBandSelectedBefore;
	int64		fillCandidateBandSelectedAfter;
	int64		fillCandidateBandTarget;
	bool		fillCandidateBandUsedPayloadRefs;
	int64		fillCandidateBandPayloadRefCount;
	int64		baseFrontierPushes;
	int64		baseFrontierPops;
	int64		baseNearestOffers;
	int64		baseVisitedChecks;
	int64		baseDuplicateSkips;
	int64		baseBatchCalls;
	int64		baseBatchNodes;
	int64		baseMaxFrontier;
	int64		segmentCount;
	int64		segmentsSearched;
	int64		entryPointCount;
	int64		entrySidecarCount;
	int64		entrySidecarScored;
	int64		entrySidecarSelected;
	int			payloadEntrySeedingMode;
	bool		payloadEntrySeedingHit;
	int64		payloadEntrySeedCount;
	int			payloadEntrySeedPayloadSlot;
	int64		payloadEntrySeedRangeCount;
	int64		residualRerankCount;
	int64		residualRerankBytes;
	int			residualRerankMode;
	double		residualRerankWeightEffective;
	int64		residualRerankBand;
	int			residualRerankBandMultiplier;
	double		residualRerankMaxAdjustment;
	int64		residualRerankReorderedCount;
	bool		residualRerankTopKChanged;
	int64		heapRescoreCount;
	int64		rescoreCount;
	int64		rescorePages;
} TqLastScanTraversal;

typedef struct TqLastScanTiming
{
	int64		prepareUs;
	int64		traverseUs;
	int64		entryUs;
	int64		baseUs;
	int64		batchUs;
	int64		heapUs;
	int64		fillUs;
	int64		rescoreUs;
	int64		sortUs;
	int64		totalUs;
	int64		entrySidecarUs;
	int64		payloadEntrySeedUs;
	int64		residualRerankUs;
	int64		heapFetchUs;
	int64		heapRescoreUs;
	int64		localExpansionUs;
	int64		scanLockWaitUs;
} TqLastScanTiming;

typedef struct TqLastScanUncertaintyRetry
{
	int			mode;
	bool		triggered;
	int			reason;
	int64		passes;
	int64		initialResultTarget;
	int64		finalResultTarget;
	int64		initialSearchEf;
	int64		finalSearchEf;
	double		gapTop10;
	double		gapBoundary;
} TqLastScanUncertaintyRetry;

typedef struct TqLastScanDense
{
	bool		branchUsed;
	int			storageKind;
	int			scoreMode;
	int			simdForce;
	int			exactCache;
	int			rescoreBand;
	int64		effectiveRescoreBand;
	int64		oversampling;
	uint64		elapsedUs;
	bool		filterUnmapped;
	bool		linearFallbackWarning;
	double		linearFallbackRatio;
	TqLastScanKernels kernels;
	TqLastScanCache cache;
	TqLastScanTraversal traversal;
	TqLastScanTiming timing;
	TqLastScanUncertaintyRetry uncertaintyRetry;
} TqLastScanDense;

typedef struct TqLastScanBm25
{
	bool		branchAvailable;
	bool		branchUsed;
	int			strategy;
	int			impactOrMode;
	int64		hotPostingsCacheMb;
	int			hybridBound;
	int			accumulatorMode;
	bool		exactRescoreBm25Only;
	int64		candidatesEffective;
	bool		kDefaulted;
	bool		cacheHit;
	uint64		cacheBuildUs;
	bool		docstatsLoadedThisQuery;
	bool		livenessLoadedThisQuery;
	uint64		docstatsBytes;
	uint64		livenessBytes;
	bool		coldCacheONWork;
	double		postingsDecodeRatio;
	bool		commonTermFallback;
	uint64		wandPruned;
	uint64		hotPostingsCacheHits;
	uint64		hotPostingsCacheMisses;
	uint64		fusedScoreBoundBlocksPruned;
	uint64		fusedScoreBoundCandidatesPruned;
	char		heapTsvectorRerankMode[16];
	uint32		heapTsvectorRerankCount;
	uint64		heapTsvectorRerankFetchUs;
	uint64		heapTsvectorRerankScoreUs;
	bool		heapTsvectorRerankTopKChanged;
	uint64		elapsedUs;
} TqLastScanBm25;

typedef struct TqLastScanFusion
{
	uint64		elapsedUs;
	bool		autoBudget;
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
	int64		hybridDenseKChosen;
	int64		hybridBm25KChosen;
	char		hybridBudgetReason[96];
	char		strategy[24];
	int64		candidatesSeen;
	uint64		duplicates;
	uint64		heapReplacements;
	bool		generationArrayReused;
	bool		generationArrayReset;
	char		finalDiversityMode[24];
	int64		finalDiversityPayloadSlot;
	int64		finalDiversityPoolSize;
	int64		finalDiversitySelected;
	uint64		finalDiversityDuplicateGroupsSuppressed;
	uint64		finalDiversityUs;
} TqLastScanFusion;

typedef struct TqLastScanQuery
{
	char		indexShape[16];
	int64		dimensions;
	int64		quantizationBits;
	bool		exactStorageKnown;
	bool		exactStorage;
	bool		querySplitEnabled;
	int64		denseRequestedK;
	int64		effectiveResultTarget;
	int64		effectiveSearchEf;
	int64		denseCandidatesEffective;
	bool		denseKDefaulted;
	int64		finalKRequested;
	int64		finalKEffective;
	int64		detectedSqlLimit;
	bool		finalKInferred;
} TqLastScanQuery;

typedef struct TqLastScanStats
{
	TqLastScanDense dense;
	TqLastScanBm25 bm25;
	TqLastScanFusion fusion;
	TqLastScanQuery query;
} TqLastScanStats;

/*
 * The single gather point: copy every nested-view value out of the file-static
 * globals, the scan-stats snapshot and the BM25 GUCs into one struct.  The flat
 * emit below keeps reading the globals directly; collecting here means the
 * grouped view cannot silently diverge in value from a flat key.
 */
static void
PgturbohybridCollectLastScanStats(TqLastScanStats *s,
								  const PgturbohybridScanStatsSnapshot *scanStats,
								  uint64 denseElapsedUs, uint64 bm25ElapsedUs,
								  uint64 fusionElapsedUs)
{
	TqLastScanDense *d = &s->dense;

	memset(s, 0, sizeof(*s));

	strlcpy(s->query.indexShape, scanStats->indexShape,
			sizeof(s->query.indexShape));
	d->branchUsed = scanStats->denseBranchUsed;
	d->storageKind = pgturbohybrid_last_graph_storage_kind;
	d->scoreMode = pgturbohybrid_last_graph_score_mode;
	d->simdForce = pgturbohybrid_last_graph_simd_force;
	d->exactCache = pgturbohybrid_last_graph_exact_cache;
	d->rescoreBand = pgturbohybrid_last_graph_rescore_band;
	d->effectiveRescoreBand = pgturbohybrid_last_graph_effective_rescore_band;
	d->oversampling = pgturbohybrid_last_graph_oversampling;
	d->elapsedUs = denseElapsedUs;
	d->filterUnmapped = pgturbohybrid_last_dense_filter_unmapped;
	d->linearFallbackWarning = pgturbohybrid_last_dense_linear_fallback_warning;
	d->linearFallbackRatio = pgturbohybrid_last_dense_linear_fallback_ratio;

	d->kernels.scoringKernel = pgturbohybrid_last_graph_scoring_kernel;
	d->kernels.batchKernel = pgturbohybrid_last_graph_batch_kernel;
	d->kernels.exactVectorKernel = pgturbohybrid_last_exact_vector_kernel;
	d->kernels.u8SplitUsed = pgturbohybrid_last_graph_u8_split_used;
	d->kernels.u8BatchX4Enabled = pgturbohybrid_last_dense_u8_batch_x4_enabled;
	d->kernels.u8BatchMode = pgturbohybrid_last_graph_u8_batch_mode;
	d->kernels.u8KernelSingle = pgturbohybrid_last_graph_u8_kernel_single;
	d->kernels.u8KernelBatch = pgturbohybrid_last_graph_u8_kernel_batch;
	d->kernels.querySplitUsed = pgturbohybrid_last_graph_querysplit_used;
	d->kernels.scoredCodes = pgturbohybrid_last_graph_scored_codes;
	d->kernels.rescoreCount = pgturbohybrid_last_graph_rescore_count;
	d->kernels.batchScoredCodes = pgturbohybrid_last_graph_batch_scored_codes;
	d->kernels.scalarScoredCodes = pgturbohybrid_last_graph_scalar_scored_codes;
	d->kernels.batchCalls = pgturbohybrid_last_graph_batch_calls;
	d->kernels.batchNodes = pgturbohybrid_last_graph_batch_nodes;
	for (int b = 0; b < PGTURBOHYBRID_SCORE_KERNEL_BUCKET_COUNT; b++)
	{
		d->kernels.kernelNodes[b] = pgturbohybrid_last_graph_score_kernel_nodes[b];
		d->kernels.kernelCalls[b] = pgturbohybrid_last_graph_score_kernel_calls[b];
	}

	d->cache.loadAttempts = pgturbohybrid_last_graph_code_page_attempts;
	d->cache.cacheHits = pgturbohybrid_last_graph_code_page_hits;
	d->cache.cacheMisses = pgturbohybrid_last_graph_code_page_misses;
	d->cache.codePagesRead = pgturbohybrid_last_graph_code_pages_read;
	d->cache.adjPagesRead = pgturbohybrid_last_graph_adj_pages_read;
	d->cache.codeTuplesCopied = pgturbohybrid_last_graph_code_tuples_copied;
	d->cache.arenaUsedBytes = pgturbohybrid_last_graph_code_arena_used_bytes;
	d->cache.arenaAllocatedBytes = pgturbohybrid_last_graph_code_arena_allocated_bytes;
	d->cache.scoredCodes = pgturbohybrid_last_graph_scored_codes;
	d->cache.codeBytes = pgturbohybrid_last_graph_code_bytes;
	d->cache.codeArenaEstimatedBytes = pgturbohybrid_last_graph_code_arena_estimated_bytes;
	d->cache.largeCodeArena = pgturbohybrid_last_graph_large_code_arena;
	d->cache.wholeCodePrefetchActive = pgturbohybrid_last_graph_whole_code_prefetch_active;
	d->cache.nativeCacheMode = pgturbohybrid_last_graph_native_cache_mode;
	d->cache.nativeCachePolicy = pgturbohybrid_last_graph_native_cache_policy;
	d->cache.nativeCacheReason = pgturbohybrid_last_graph_native_cache_reason;
	d->cache.nativeCacheUsed = pgturbohybrid_last_graph_native_cache_used;
	d->cache.nativeCacheReused = pgturbohybrid_last_graph_native_cache_reused;
	d->cache.nativeCacheBuiltThisScan = pgturbohybrid_last_graph_native_cache_built_this_scan;
	d->cache.nativeCacheAttachUs = pgturbohybrid_last_graph_native_cache_attach_us;
	d->cache.nativeCacheBuildUs = pgturbohybrid_last_graph_native_cache_build_us;
	d->cache.nativeCacheWaitUs = pgturbohybrid_last_graph_native_cache_wait_us;
	d->cache.nativeCacheRefcount = pgturbohybrid_last_graph_native_cache_refcount;
	d->cache.nativeCacheBytes = pgturbohybrid_last_graph_native_cache_bytes;
	d->cache.nativeCacheCodeBytes = pgturbohybrid_last_graph_native_cache_code_bytes;
	d->cache.nativeCacheAdjBytes = pgturbohybrid_last_graph_native_cache_adj_bytes;
	d->cache.nativeCacheExactBytes = pgturbohybrid_last_graph_native_cache_exact_bytes;
	d->cache.nativeCacheWarning = pgturbohybrid_last_graph_native_cache_warning;
	d->cache.nativeCacheWarningReason = pgturbohybrid_last_graph_native_cache_warning_reason;
	d->cache.codeBufferLockWaitUs = pgturbohybrid_last_graph_code_buffer_lock_wait_us;
	d->cache.adjBufferLockWaitUs = pgturbohybrid_last_graph_adj_buffer_lock_wait_us;
	strlcpy(d->cache.multivectorDocSidecarCacheMode,
			scanStats->multivectorDocSidecarCacheMode,
			sizeof(d->cache.multivectorDocSidecarCacheMode));
	d->cache.multivectorDocSidecarPagesRead =
		scanStats->multivectorDocSidecarPagesRead;
	d->cache.multivectorDocSidecarCacheHits =
		scanStats->multivectorDocSidecarCacheHits;
	d->cache.multivectorDocSidecarCacheMisses =
		scanStats->multivectorDocSidecarCacheMisses;
	d->cache.multivectorDocSidecarBytesTouched =
		scanStats->multivectorDocSidecarBytesTouched;
	d->cache.multivectorDocSidecarVectorsLoaded =
		scanStats->multivectorDocSidecarVectorsLoaded;
	d->cache.multivectorDocSidecarDocMapPagesRead =
		scanStats->multivectorDocSidecarDocMapPagesRead;
	d->cache.multivectorDocSidecarDocMapBytesTouched =
		scanStats->multivectorDocSidecarDocMapBytesTouched;
	d->cache.multivectorDocSidecarResidentVectorsLoaded =
		scanStats->multivectorDocSidecarResidentVectorsLoaded;
	d->cache.multivectorDocSidecarResidentBytesLoaded =
		scanStats->multivectorDocSidecarResidentBytesLoaded;
	d->cache.multivectorDocSidecarVectorChunkRefBytesTouched =
		scanStats->multivectorDocSidecarVectorChunkRefBytesTouched;
	d->cache.multivectorDocSidecarPagedVectorPagesRead =
		scanStats->multivectorDocSidecarPagedVectorPagesRead;
	d->cache.multivectorDocSidecarPagedVectorBytesTouched =
		scanStats->multivectorDocSidecarPagedVectorBytesTouched;
	d->cache.multivectorSidecarPageReadUs =
		scanStats->multivectorSidecarPageReadUs;
	d->cache.multivectorSidecarVectorReconstructUs =
		scanStats->multivectorSidecarVectorReconstructUs;
	d->cache.multivectorTokensOriginal =
		scanStats->multivectorTokensOriginal;
	d->cache.multivectorTokensPooled =
		scanStats->multivectorTokensPooled;

	d->traversal.visitedNodes = pgturbohybrid_last_graph_visited_nodes;
	d->traversal.scoredCodes = pgturbohybrid_last_graph_scored_codes;
	d->traversal.candidateCount = pgturbohybrid_last_graph_candidate_count;
	d->traversal.fillCandidateBandCalls =
		pgturbohybrid_last_graph_fill_candidate_band_calls;
	d->traversal.fillCandidateBandReason =
		pgturbohybrid_last_graph_fill_candidate_band_reason;
	d->traversal.fillCandidateBandVisited =
		pgturbohybrid_last_graph_fill_candidate_band_visited;
	d->traversal.fillCandidateBandScored =
		pgturbohybrid_last_graph_fill_candidate_band_scored;
	d->traversal.fillCandidateBandSelectedBefore =
		pgturbohybrid_last_graph_fill_candidate_band_selected_before;
	d->traversal.fillCandidateBandSelectedAfter =
		pgturbohybrid_last_graph_fill_candidate_band_selected_after;
	d->traversal.fillCandidateBandTarget =
		pgturbohybrid_last_graph_fill_candidate_band_target;
	d->traversal.fillCandidateBandUsedPayloadRefs =
		pgturbohybrid_last_graph_fill_candidate_band_used_payload_refs;
	d->traversal.fillCandidateBandPayloadRefCount =
		pgturbohybrid_last_graph_fill_candidate_band_payload_ref_count;
	d->traversal.baseFrontierPushes = pgturbohybrid_last_graph_base_frontier_pushes;
	d->traversal.baseFrontierPops = pgturbohybrid_last_graph_base_frontier_pops;
	d->traversal.baseNearestOffers = pgturbohybrid_last_graph_base_nearest_offers;
	d->traversal.baseVisitedChecks = pgturbohybrid_last_graph_base_visited_checks;
	d->traversal.baseDuplicateSkips = pgturbohybrid_last_graph_base_duplicate_skips;
	d->traversal.baseBatchCalls = pgturbohybrid_last_graph_base_batch_calls;
	d->traversal.baseBatchNodes = pgturbohybrid_last_graph_base_batch_nodes;
	d->traversal.baseMaxFrontier = pgturbohybrid_last_graph_base_max_frontier;
	d->traversal.segmentCount = pgturbohybrid_last_graph_segment_count;
	d->traversal.segmentsSearched = pgturbohybrid_last_graph_segments_searched;
	d->traversal.entryPointCount = pgturbohybrid_last_graph_entry_point_count;
	d->traversal.entrySidecarCount = pgturbohybrid_last_graph_entry_sidecar_count;
	d->traversal.entrySidecarScored = pgturbohybrid_last_graph_entry_sidecar_scored;
	d->traversal.entrySidecarSelected = pgturbohybrid_last_graph_entry_sidecar_selected;
	d->traversal.payloadEntrySeedingMode =
		pgturbohybrid_last_payload_entry_seeding_mode;
	d->traversal.payloadEntrySeedingHit =
		pgturbohybrid_last_payload_entry_seeding_hit;
	d->traversal.payloadEntrySeedCount =
		pgturbohybrid_last_payload_entry_seed_count;
	d->traversal.payloadEntrySeedPayloadSlot =
		pgturbohybrid_last_payload_entry_seed_payload_slot;
	d->traversal.payloadEntrySeedRangeCount =
		pgturbohybrid_last_payload_entry_seed_range_count;
	d->traversal.residualRerankCount = pgturbohybrid_last_graph_residual_rerank_count;
	d->traversal.residualRerankBytes = pgturbohybrid_last_graph_residual_rerank_bytes;
	d->traversal.residualRerankMode = pgturbohybrid_last_graph_residual_rerank_mode;
	d->traversal.residualRerankWeightEffective =
		pgturbohybrid_last_graph_residual_rerank_weight_effective;
	d->traversal.residualRerankBand = pgturbohybrid_last_graph_residual_rerank_band;
	d->traversal.residualRerankBandMultiplier =
		pgturbohybrid_last_graph_residual_rerank_band_multiplier;
	d->traversal.residualRerankMaxAdjustment =
		pgturbohybrid_last_graph_residual_rerank_max_adjustment;
	d->traversal.residualRerankReorderedCount =
		pgturbohybrid_last_graph_residual_rerank_reordered_count;
	d->traversal.residualRerankTopKChanged =
		pgturbohybrid_last_graph_residual_rerank_topk_changed;
	d->traversal.heapRescoreCount = pgturbohybrid_last_graph_heap_rescore_count;
	d->traversal.rescoreCount = pgturbohybrid_last_graph_rescore_count;
	d->traversal.rescorePages = pgturbohybrid_last_graph_rescore_pages;

	d->timing.prepareUs = pgturbohybrid_last_graph_prepare_us;
	d->timing.traverseUs = pgturbohybrid_last_graph_traverse_us;
	d->timing.entryUs = pgturbohybrid_last_graph_entry_us;
	d->timing.baseUs = pgturbohybrid_last_graph_base_us;
	d->timing.batchUs = pgturbohybrid_last_graph_batch_us;
	d->timing.heapUs = pgturbohybrid_last_graph_heap_us;
	d->timing.fillUs = pgturbohybrid_last_graph_fill_us;
	d->timing.rescoreUs = pgturbohybrid_last_graph_rescore_us;
	d->timing.sortUs = pgturbohybrid_last_graph_sort_us;
	d->timing.totalUs = pgturbohybrid_last_graph_total_us;
	d->timing.entrySidecarUs = pgturbohybrid_last_graph_entry_sidecar_us;
	d->timing.payloadEntrySeedUs = pgturbohybrid_last_payload_entry_seed_us;
	d->timing.residualRerankUs = pgturbohybrid_last_graph_residual_rerank_us;
	d->timing.heapFetchUs = pgturbohybrid_last_graph_heap_fetch_us;
	d->timing.heapRescoreUs = pgturbohybrid_last_graph_heap_rescore_us;
	d->timing.localExpansionUs = pgturbohybrid_last_graph_local_expansion_us;
	d->timing.scanLockWaitUs = pgturbohybrid_last_graph_scan_lock_wait_us;
	d->uncertaintyRetry.mode =
		pgturbohybrid_last_graph_uncertainty_retry_mode;
	d->uncertaintyRetry.triggered =
		pgturbohybrid_last_graph_uncertainty_retry_triggered;
	d->uncertaintyRetry.reason =
		pgturbohybrid_last_graph_uncertainty_retry_reason;
	d->uncertaintyRetry.passes =
		pgturbohybrid_last_graph_uncertainty_retry_passes;
	d->uncertaintyRetry.initialResultTarget =
		pgturbohybrid_last_graph_uncertainty_initial_result_target;
	d->uncertaintyRetry.finalResultTarget =
		pgturbohybrid_last_graph_uncertainty_final_result_target;
	d->uncertaintyRetry.initialSearchEf =
		pgturbohybrid_last_graph_uncertainty_initial_search_ef;
	d->uncertaintyRetry.finalSearchEf =
		pgturbohybrid_last_graph_uncertainty_final_search_ef;
	d->uncertaintyRetry.gapTop10 =
		pgturbohybrid_last_graph_uncertainty_gap_top10;
	d->uncertaintyRetry.gapBoundary =
		pgturbohybrid_last_graph_uncertainty_gap_boundary;

	s->bm25.strategy = pgturbohybrid_bm25_strategy;
	s->bm25.branchAvailable = scanStats->bm25BranchAvailable;
	s->bm25.branchUsed = scanStats->bm25BranchUsed;
	s->bm25.impactOrMode = pgturbohybrid_bm25_impact_or_mode;
	s->bm25.hotPostingsCacheMb = pgturbohybrid_bm25_hot_postings_cache_mb;
	s->bm25.hybridBound = pgturbohybrid_bm25_hybrid_bound;
	s->bm25.accumulatorMode = pgturbohybrid_bm25_accumulator_mode;
	s->bm25.exactRescoreBm25Only = pgturbohybrid_enable_exact_rescore_for_bm25_only;
	s->bm25.candidatesEffective = scanStats->bm25CandidatesEffective;
	s->bm25.kDefaulted = scanStats->bm25KDefaulted;
	s->bm25.cacheHit = scanStats->bm25CacheHit;
	s->bm25.cacheBuildUs = scanStats->bm25CacheBuildUs;
	s->bm25.docstatsLoadedThisQuery =
		scanStats->bm25DocstatsLoadedThisQuery;
	s->bm25.livenessLoadedThisQuery =
		scanStats->bm25LivenessLoadedThisQuery;
	s->bm25.docstatsBytes = scanStats->bm25DocstatsBytes;
	s->bm25.livenessBytes = scanStats->bm25LivenessBytes;
	s->bm25.coldCacheONWork = scanStats->bm25ColdCacheONWork;
	s->bm25.postingsDecodeRatio = scanStats->bm25PostingsDecodeRatio;
	s->bm25.commonTermFallback = scanStats->bm25CommonTermFallback;
	s->bm25.wandPruned = scanStats->bm25WandPruned;
	s->bm25.hotPostingsCacheHits = scanStats->bm25HotPostingsCacheHits;
	s->bm25.hotPostingsCacheMisses = scanStats->bm25HotPostingsCacheMisses;
	s->bm25.fusedScoreBoundBlocksPruned =
		scanStats->bm25FusedScoreBoundBlocksPruned;
	s->bm25.fusedScoreBoundCandidatesPruned =
		scanStats->bm25FusedScoreBoundCandidatesPruned;
	strlcpy(s->bm25.heapTsvectorRerankMode,
			scanStats->bm25HeapTSVectorRerankMode,
			sizeof(s->bm25.heapTsvectorRerankMode));
	s->bm25.heapTsvectorRerankCount =
		scanStats->bm25HeapTSVectorRerankCount;
	s->bm25.heapTsvectorRerankFetchUs =
		scanStats->bm25HeapTSVectorRerankFetchUs;
	s->bm25.heapTsvectorRerankScoreUs =
		scanStats->bm25HeapTSVectorRerankScoreUs;
	s->bm25.heapTsvectorRerankTopKChanged =
		scanStats->bm25HeapTSVectorRerankTopKChanged;
	s->bm25.elapsedUs = bm25ElapsedUs;

	s->fusion.elapsedUs = fusionElapsedUs;
	s->fusion.autoBudget = pgturbohybrid_auto_budget;
	s->fusion.fastWeightedEnabled = scanStats->fastWeightedEnabled;
	s->fusion.fastWeightedAlpha = scanStats->fastWeightedAlpha;
	s->fusion.calibratedFusionEnabled =
		scanStats->calibratedFusionEnabled;
	strlcpy(s->fusion.calibratedFusionQueryShape,
			scanStats->calibratedFusionQueryShape,
			sizeof(s->fusion.calibratedFusionQueryShape));
	s->fusion.calibratedFusionAlphaEffective =
		scanStats->calibratedFusionAlphaEffective;
	s->fusion.calibratedFusionBothMatchBonus =
		scanStats->calibratedFusionBothMatchBonus;
	strlcpy(s->fusion.calibratedFusionDenseNormMode,
			scanStats->calibratedFusionDenseNormMode,
			sizeof(s->fusion.calibratedFusionDenseNormMode));
	strlcpy(s->fusion.calibratedFusionBm25NormMode,
			scanStats->calibratedFusionBm25NormMode,
			sizeof(s->fusion.calibratedFusionBm25NormMode));
	s->fusion.dbsfEnabled = scanStats->dbsfEnabled;
	memcpy(s->fusion.dbsfBranchMean, scanStats->dbsfBranchMean,
		   sizeof(s->fusion.dbsfBranchMean));
	memcpy(s->fusion.dbsfBranchStddev, scanStats->dbsfBranchStddev,
		   sizeof(s->fusion.dbsfBranchStddev));
	memcpy(s->fusion.dbsfBranchMin, scanStats->dbsfBranchMin,
		   sizeof(s->fusion.dbsfBranchMin));
	memcpy(s->fusion.dbsfBranchMax, scanStats->dbsfBranchMax,
		   sizeof(s->fusion.dbsfBranchMax));
	s->fusion.dbsfDegenerateBranches =
		scanStats->dbsfDegenerateBranches;
	strlcpy(s->fusion.bm25NormMode, scanStats->bm25NormMode,
			sizeof(s->fusion.bm25NormMode));
	strlcpy(s->fusion.denseNormMode, scanStats->denseNormMode,
			sizeof(s->fusion.denseNormMode));
	strlcpy(s->fusion.hybridBudgetPolicy, scanStats->hybridBudgetPolicy,
			sizeof(s->fusion.hybridBudgetPolicy));
	strlcpy(s->fusion.hybridQueryShape, scanStats->hybridQueryShape,
			sizeof(s->fusion.hybridQueryShape));
	s->fusion.hybridDenseKChosen = scanStats->hybridDenseKChosen;
	s->fusion.hybridBm25KChosen = scanStats->hybridBm25KChosen;
	strlcpy(s->fusion.hybridBudgetReason, scanStats->hybridBudgetReason,
			sizeof(s->fusion.hybridBudgetReason));
	strlcpy(s->fusion.strategy, scanStats->fusionStrategy,
			sizeof(s->fusion.strategy));
	s->fusion.candidatesSeen = scanStats->fusionCandidatesSeen;
	s->fusion.duplicates = scanStats->fusionDuplicates;
	s->fusion.heapReplacements = scanStats->fusionHeapReplacements;
	s->fusion.generationArrayReused =
		scanStats->fusionGenerationArrayReused;
	s->fusion.generationArrayReset =
		scanStats->fusionGenerationArrayReset;
	strlcpy(s->fusion.finalDiversityMode, scanStats->finalDiversityMode,
			sizeof(s->fusion.finalDiversityMode));
	s->fusion.finalDiversityPayloadSlot =
		scanStats->finalDiversityPayloadSlot;
	s->fusion.finalDiversityPoolSize =
		scanStats->finalDiversityPoolSize;
	s->fusion.finalDiversitySelected =
		scanStats->finalDiversitySelected;
	s->fusion.finalDiversityDuplicateGroupsSuppressed =
		scanStats->finalDiversityDuplicateGroupsSuppressed;
	s->fusion.finalDiversityUs =
		scanStats->finalDiversityUs;

	s->query.dimensions = pgturbohybrid_last_graph_dimensions;
	s->query.quantizationBits = pgturbohybrid_last_graph_quantization_bits;
	s->query.exactStorageKnown = pgturbohybrid_last_graph_exact_storage_known;
	s->query.exactStorage = pgturbohybrid_last_graph_exact_storage;
	s->query.querySplitEnabled = pgturbohybrid_last_graph_query_split_active;
	s->query.denseRequestedK = pgturbohybrid_last_graph_dense_requested_k;
	s->query.effectiveResultTarget = pgturbohybrid_last_graph_effective_result_target;
	s->query.effectiveSearchEf = pgturbohybrid_last_graph_effective_search_ef;
	s->query.denseCandidatesEffective = scanStats->denseCandidatesEffective;
	s->query.denseKDefaulted = scanStats->denseKDefaulted;
	s->query.finalKRequested = pgturbohybrid_last_final_k_requested;
	s->query.finalKEffective = pgturbohybrid_last_final_k_effective;
	s->query.detectedSqlLimit = pgturbohybrid_last_sql_limit;
	s->query.finalKInferred = pgturbohybrid_last_final_k_inferred;
}

/* Render the consolidated struct as the nested "dense"/"bm25"/"fusion"/"query"
 * sections.  Derived display names reuse the same helpers as the flat emit. */
static void
PgturbohybridEmitNestedScanStats(PgturbohybridJsonbState *state,
								 const TqLastScanStats *s)
{
	const TqLastScanDense *d = &s->dense;
	const TqLastScanKernels *k = &d->kernels;
	const TqLastScanCache *c = &d->cache;
	const TqLastScanTraversal *t = &d->traversal;
	const TqLastScanTiming *tm = &d->timing;

	/* dense ----------------------------------------------------------- */
	PgturbohybridJsonbAddKey(state, "dense");
	PgturbohybridJsonbBeginObject(state);
	PgturbohybridJsonbAddBool(state, "branch_used", d->branchUsed);
	PgturbohybridJsonbAddString(state, "storage_kind",
								PgturbohybridGraphStorageKindName(d->storageKind));
	PgturbohybridJsonbAddString(state, "score_mode",
								PgturbohybridGraphTqScoreModeName(d->scoreMode));
	PgturbohybridJsonbAddString(state, "simd_force",
								PgturbohybridGraphTqSimdForceName(d->simdForce));
	PgturbohybridJsonbAddString(state, "dense_scorer",
								k->u8SplitUsed ?
								PgturbohybridGraphU8SplitKernelName() :
								PgturbohybridDenseScorerUsedName(k->querySplitUsed,
																 k->scoringKernel,
																 k->scoredCodes,
																 k->rescoreCount));
	PgturbohybridJsonbAddString(state, "exact_cache",
								PgturbohybridGraphExactCacheName(d->exactCache));
	PgturbohybridJsonbAddBool(state, "exact_cache_active",
							  d->exactCache != PGTURBOHYBRID_GRAPH_EXACT_CACHE_OFF);
	PgturbohybridJsonbAddString(state, "rescore_band",
								PgturbohybridGraphRescoreBandName(d->rescoreBand));
	PgturbohybridJsonbAddBool(state, "rescore_band_active",
							  d->effectiveRescoreBand > 0);
	PgturbohybridJsonbAddInt64(state, "effective_rescore_band",
							   d->effectiveRescoreBand);
	PgturbohybridJsonbAddInt64(state, "oversampling", d->oversampling);
	PgturbohybridJsonbAddUint64(state, "elapsed_us", d->elapsedUs);
	PgturbohybridJsonbAddBool(state, "filter_unmapped", d->filterUnmapped);
	PgturbohybridJsonbAddBool(state, "linear_fallback_warning",
							  d->linearFallbackWarning);
	PgturbohybridJsonbAddFloat8(state, "linear_fallback_ratio",
								d->linearFallbackRatio);

	/* dense.kernels --------------------------------------------------- */
	PgturbohybridJsonbAddKey(state, "kernels");
	PgturbohybridJsonbBeginObject(state);
	PgturbohybridJsonbAddString(state, "scoring_kernel",
								PgturbohybridDenseScorerName(k->scoringKernel));
	PgturbohybridJsonbAddString(state, "batch_kernel",
								PgturbohybridDenseScorerName(k->batchKernel));
	PgturbohybridJsonbAddString(state, "scalar_fallback_kernel",
								PgturbohybridDenseScalarFallbackName());
	PgturbohybridJsonbAddString(state, "exact_kernel",
								TqExactKernelName(k->exactVectorKernel));
	PgturbohybridJsonbAddBool(state, "u8_split_enabled", k->u8SplitUsed);
	PgturbohybridJsonbAddBool(state, "u8_batch_x4_enabled", k->u8BatchX4Enabled);
	PgturbohybridJsonbAddString(state, "u8_batch_mode",
								PgturbohybridGraphU8BatchModeName(k->u8BatchMode));
	PgturbohybridJsonbAddString(state, "u8_kernel_single",
								PgturbohybridGraphU8KernelName(k->u8KernelSingle));
	PgturbohybridJsonbAddString(state, "u8_kernel_batch",
								PgturbohybridGraphU8KernelName(k->u8KernelBatch));
	PgturbohybridJsonbAddInt64(state, "batch_scored_codes", k->batchScoredCodes);
	PgturbohybridJsonbAddInt64(state, "simd_scored_codes",
							   Max(0, k->scoredCodes - k->scalarScoredCodes));
	PgturbohybridJsonbAddInt64(state, "scalar_scored_codes", k->scalarScoredCodes);
	PgturbohybridJsonbAddInt64(state, "batch_calls", k->batchCalls);
	PgturbohybridJsonbAddInt64(state, "batch_nodes", k->batchNodes);
	PgturbohybridJsonbAddFloat8(state, "avg_batch_size",
								k->batchCalls > 0 ?
								(double) k->batchNodes / (double) k->batchCalls : 0.0);
	PgturbohybridJsonbAddKey(state, "buckets");
	PgturbohybridJsonbBeginObject(state);
	for (int b = 0; b < PGTURBOHYBRID_SCORE_KERNEL_BUCKET_COUNT; b++)
	{
		PgturbohybridJsonbAddKey(state, PgturbohybridGraphScoreKernelBucketName(b));
		PgturbohybridJsonbBeginObject(state);
		PgturbohybridJsonbAddInt64(state, "nodes", k->kernelNodes[b]);
		PgturbohybridJsonbAddInt64(state, "calls", k->kernelCalls[b]);
		PgturbohybridJsonbCloseObject(state);
	}
	PgturbohybridJsonbCloseObject(state);	/* buckets */
	PgturbohybridJsonbCloseObject(state);	/* kernels */

	/* dense.cache ----------------------------------------------------- */
	PgturbohybridJsonbAddKey(state, "cache");
	PgturbohybridJsonbBeginObject(state);
	PgturbohybridJsonbAddInt64(state, "load_attempts", c->loadAttempts);
	PgturbohybridJsonbAddInt64(state, "cache_hits", c->cacheHits);
	PgturbohybridJsonbAddInt64(state, "cache_misses", c->cacheMisses);
	PgturbohybridJsonbAddInt64(state, "code_pages_read", c->codePagesRead);
	PgturbohybridJsonbAddInt64(state, "adj_pages_read", c->adjPagesRead);
	PgturbohybridJsonbAddInt64(state, "code_tuples_copied", c->codeTuplesCopied);
	PgturbohybridJsonbAddInt64(state, "arena_used_bytes", c->arenaUsedBytes);
	PgturbohybridJsonbAddInt64(state, "arena_allocated_bytes", c->arenaAllocatedBytes);
	PgturbohybridJsonbAddInt64(state, "code_bytes", c->codeBytes);
	PgturbohybridJsonbAddInt64(state, "code_arena_estimated_bytes",
							   c->codeArenaEstimatedBytes);
	PgturbohybridJsonbAddBool(state, "large_code_arena", c->largeCodeArena);
	PgturbohybridJsonbAddBool(state, "whole_code_prefetch_active",
							  c->wholeCodePrefetchActive);
	PgturbohybridJsonbAddFloat8(state, "hit_rate",
								c->loadAttempts > 0 ?
								(double) c->cacheHits / (double) c->loadAttempts : 0.0);
	PgturbohybridJsonbAddFloat8(state, "pages_read_per_scored_code",
								c->scoredCodes > 0 ?
								(double) c->codePagesRead / (double) c->scoredCodes : 0.0);
	/* Per-backend native cache: mode, the one-time cold-build cost, and the
	 * resident footprint duplicated across concurrent clients. */
	PgturbohybridJsonbAddString(state, "native_cache_mode",
								PgturbohybridGraphNativeCacheModeName(c->nativeCacheMode));
	PgturbohybridJsonbAddString(state, "native_cache_policy",
								PgturbohybridNativeCachePolicyName(c->nativeCachePolicy));
	PgturbohybridJsonbAddString(state, "native_cache_scope",
								PgturbohybridGraphNativeCacheScopeName(c->nativeCacheMode));
	PgturbohybridJsonbAddBool(state, "native_cache_used",
							  c->nativeCacheUsed);
	PgturbohybridJsonbAddString(state, "native_cache_reason",
								PgturbohybridGraphNativeCacheReasonName(c->nativeCacheReason));
	PgturbohybridJsonbAddBool(state, "native_cache_reused",
							  c->nativeCacheReused);
	PgturbohybridJsonbAddBool(state, "native_cache_built_this_scan",
							  c->nativeCacheBuiltThisScan);
	PgturbohybridJsonbAddInt64(state, "native_cache_attach_us", c->nativeCacheAttachUs);
	PgturbohybridJsonbAddInt64(state, "native_cache_build_us", c->nativeCacheBuildUs);
	PgturbohybridJsonbAddInt64(state, "native_cache_wait_us", c->nativeCacheWaitUs);
	PgturbohybridJsonbAddInt64(state, "native_cache_refcount", c->nativeCacheRefcount);
	PgturbohybridJsonbAddInt64(state, "native_cache_bytes", c->nativeCacheBytes);
	PgturbohybridJsonbAddInt64(state, "native_cache_code_bytes", c->nativeCacheCodeBytes);
	PgturbohybridJsonbAddInt64(state, "native_cache_adj_bytes", c->nativeCacheAdjBytes);
	PgturbohybridJsonbAddInt64(state, "native_cache_exact_bytes", c->nativeCacheExactBytes);
	PgturbohybridJsonbAddBool(state, "native_cache_warning", c->nativeCacheWarning);
	PgturbohybridJsonbAddString(state, "native_cache_warning_reason",
								c->nativeCacheWarningReason != NULL ?
								c->nativeCacheWarningReason : "none");
	PgturbohybridJsonbAddInt64(state, "code_buffer_lock_wait_us",
							   c->codeBufferLockWaitUs);
	PgturbohybridJsonbAddInt64(state, "adj_buffer_lock_wait_us",
							   c->adjBufferLockWaitUs);
	PgturbohybridJsonbAddString(state, "multivector_doc_sidecar_cache_mode",
								c->multivectorDocSidecarCacheMode[0] != '\0' ?
								c->multivectorDocSidecarCacheMode : "none");
	PgturbohybridJsonbAddUint64(state, "multivector_doc_sidecar_pages_read",
								c->multivectorDocSidecarPagesRead);
	PgturbohybridJsonbAddUint64(state, "multivector_doc_sidecar_cache_hits",
								c->multivectorDocSidecarCacheHits);
	PgturbohybridJsonbAddUint64(state, "multivector_doc_sidecar_cache_misses",
								c->multivectorDocSidecarCacheMisses);
	PgturbohybridJsonbAddUint64(state, "multivector_doc_sidecar_bytes_touched",
								c->multivectorDocSidecarBytesTouched);
	PgturbohybridJsonbAddUint64(state, "multivector_doc_sidecar_vectors_loaded",
								c->multivectorDocSidecarVectorsLoaded);
	PgturbohybridJsonbAddUint64(state,
								"multivector_doc_sidecar_docmap_pages_read",
								c->multivectorDocSidecarDocMapPagesRead);
	PgturbohybridJsonbAddUint64(state,
								"multivector_doc_sidecar_docmap_bytes_touched",
								c->multivectorDocSidecarDocMapBytesTouched);
	PgturbohybridJsonbAddUint64(state,
								"multivector_doc_sidecar_resident_vectors_loaded",
								c->multivectorDocSidecarResidentVectorsLoaded);
	PgturbohybridJsonbAddUint64(state,
								"multivector_doc_sidecar_resident_bytes_loaded",
								c->multivectorDocSidecarResidentBytesLoaded);
	PgturbohybridJsonbAddUint64(state,
								"multivector_doc_sidecar_vector_chunk_ref_bytes_touched",
								c->multivectorDocSidecarVectorChunkRefBytesTouched);
	PgturbohybridJsonbAddUint64(state,
								"multivector_doc_sidecar_paged_vector_pages_read",
								c->multivectorDocSidecarPagedVectorPagesRead);
	PgturbohybridJsonbAddUint64(state,
								"multivector_doc_sidecar_paged_vector_bytes_touched",
								c->multivectorDocSidecarPagedVectorBytesTouched);
	PgturbohybridJsonbAddUint64(state, "multivector_sidecar_page_read_time_us",
								c->multivectorSidecarPageReadUs);
	PgturbohybridJsonbAddUint64(state,
								"multivector_sidecar_vector_reconstruct_time_us",
								c->multivectorSidecarVectorReconstructUs);
	PgturbohybridJsonbAddUint64(state, "multivector_tokens_original",
								c->multivectorTokensOriginal);
	PgturbohybridJsonbAddUint64(state, "multivector_tokens_pooled",
								c->multivectorTokensPooled);
	PgturbohybridJsonbAddFloat8(state, "multivector_token_pooling_ratio",
								c->multivectorTokensOriginal > 0 ?
								(double) c->multivectorTokensPooled /
								(double) c->multivectorTokensOriginal : 0.0);
	PgturbohybridJsonbCloseObject(state);	/* cache */

	/* dense.traversal ------------------------------------------------- */
	PgturbohybridJsonbAddKey(state, "traversal");
	PgturbohybridJsonbBeginObject(state);
	PgturbohybridJsonbAddInt64(state, "visited_nodes", t->visitedNodes);
	PgturbohybridJsonbAddInt64(state, "scored_codes", t->scoredCodes);
	PgturbohybridJsonbAddInt64(state, "candidate_count", t->candidateCount);
	PgturbohybridJsonbAddInt64(state, "fill_candidate_band_calls",
							   t->fillCandidateBandCalls);
	PgturbohybridJsonbAddString(state, "fill_candidate_band_reason",
								PgturbohybridGraphFillCandidateBandReasonName(t->fillCandidateBandReason));
	PgturbohybridJsonbAddInt64(state, "fill_candidate_band_visited",
							   t->fillCandidateBandVisited);
	PgturbohybridJsonbAddInt64(state, "fill_candidate_band_scored",
							   t->fillCandidateBandScored);
	PgturbohybridJsonbAddInt64(state, "fill_candidate_band_selected_before",
							   t->fillCandidateBandSelectedBefore);
	PgturbohybridJsonbAddInt64(state, "fill_candidate_band_selected_after",
							   t->fillCandidateBandSelectedAfter);
	PgturbohybridJsonbAddInt64(state, "fill_candidate_band_target",
							   t->fillCandidateBandTarget);
	PgturbohybridJsonbAddBool(state, "fill_candidate_band_used_payload_refs",
							  t->fillCandidateBandUsedPayloadRefs);
	PgturbohybridJsonbAddInt64(state, "fill_candidate_band_payload_ref_count",
							   t->fillCandidateBandPayloadRefCount);
	PgturbohybridJsonbAddInt64(state, "rescore_count", t->rescoreCount);
	PgturbohybridJsonbAddInt64(state, "rescore_pages", t->rescorePages);
	PgturbohybridJsonbAddInt64(state, "segment_count", t->segmentCount);
	PgturbohybridJsonbAddInt64(state, "segments_searched", t->segmentsSearched);
	PgturbohybridJsonbAddInt64(state, "entry_point_count", t->entryPointCount);
	PgturbohybridJsonbAddInt64(state, "entry_sidecar_count", t->entrySidecarCount);
	PgturbohybridJsonbAddInt64(state, "entry_sidecar_scored", t->entrySidecarScored);
	PgturbohybridJsonbAddInt64(state, "entry_sidecar_selected", t->entrySidecarSelected);
	PgturbohybridJsonbAddString(state, "payload_entry_seeding_mode",
								PgturbohybridPayloadEntrySeedingName(
									t->payloadEntrySeedingMode));
	PgturbohybridJsonbAddBool(state, "payload_entry_seeding_hit",
							  t->payloadEntrySeedingHit);
	PgturbohybridJsonbAddInt64(state, "payload_entry_seed_count",
							   t->payloadEntrySeedCount);
	PgturbohybridJsonbAddInt64(state, "payload_entry_seed_payload_slot",
							   t->payloadEntrySeedPayloadSlot);
	PgturbohybridJsonbAddInt64(state, "payload_entry_seed_range_count",
							   t->payloadEntrySeedRangeCount);
	PgturbohybridJsonbAddInt64(state, "residual_rerank_count", t->residualRerankCount);
	PgturbohybridJsonbAddInt64(state, "residual_rerank_bytes", t->residualRerankBytes);
	PgturbohybridJsonbAddString(state, "residual_rerank_mode",
								PgturbohybridGraphDenseResidualRerankModeName(
									t->residualRerankMode));
	PgturbohybridJsonbAddFloat8(state, "residual_rerank_weight_effective",
								t->residualRerankWeightEffective);
	PgturbohybridJsonbAddInt64(state, "residual_rerank_band", t->residualRerankBand);
	PgturbohybridJsonbAddInt64(state, "residual_rerank_band_multiplier",
							   t->residualRerankBandMultiplier);
	PgturbohybridJsonbAddFloat8(state, "residual_rerank_max_adjustment",
								t->residualRerankMaxAdjustment);
	PgturbohybridJsonbAddInt64(state, "residual_rerank_reordered_count",
							   t->residualRerankReorderedCount);
	PgturbohybridJsonbAddBool(state, "residual_rerank_topk_changed",
							  t->residualRerankTopKChanged);
	PgturbohybridJsonbAddInt64(state, "heap_rescore_count", t->heapRescoreCount);
	PgturbohybridJsonbAddKey(state, "base_layer");
	PgturbohybridJsonbBeginObject(state);
	PgturbohybridJsonbAddInt64(state, "frontier_pushes", t->baseFrontierPushes);
	PgturbohybridJsonbAddInt64(state, "frontier_pops", t->baseFrontierPops);
	PgturbohybridJsonbAddInt64(state, "nearest_offers", t->baseNearestOffers);
	PgturbohybridJsonbAddInt64(state, "visited_checks", t->baseVisitedChecks);
	PgturbohybridJsonbAddInt64(state, "duplicate_skips", t->baseDuplicateSkips);
	PgturbohybridJsonbAddInt64(state, "batch_calls", t->baseBatchCalls);
	PgturbohybridJsonbAddInt64(state, "batch_nodes", t->baseBatchNodes);
	PgturbohybridJsonbAddInt64(state, "max_frontier_size", t->baseMaxFrontier);
	PgturbohybridJsonbCloseObject(state);	/* base_layer */
	PgturbohybridJsonbCloseObject(state);	/* traversal */

	/* dense.uncertainty_retry ---------------------------------------- */
	PgturbohybridJsonbAddKey(state, "uncertainty_retry");
	PgturbohybridJsonbBeginObject(state);
	PgturbohybridJsonbAddString(state, "mode",
								PgturbohybridGraphDenseUncertaintyRetryModeName(
									d->uncertaintyRetry.mode));
	PgturbohybridJsonbAddBool(state, "triggered",
							  d->uncertaintyRetry.triggered);
	PgturbohybridJsonbAddString(state, "reason",
								PgturbohybridGraphDenseUncertaintyRetryReasonName(
									d->uncertaintyRetry.reason));
	PgturbohybridJsonbAddInt64(state, "passes",
							   d->uncertaintyRetry.passes);
	PgturbohybridJsonbAddInt64(state, "initial_result_target",
							   d->uncertaintyRetry.initialResultTarget);
	PgturbohybridJsonbAddInt64(state, "final_result_target",
							   d->uncertaintyRetry.finalResultTarget);
	PgturbohybridJsonbAddInt64(state, "initial_search_ef",
							   d->uncertaintyRetry.initialSearchEf);
	PgturbohybridJsonbAddInt64(state, "final_search_ef",
							   d->uncertaintyRetry.finalSearchEf);
	PgturbohybridJsonbAddFloat8(state, "gap_top10",
								d->uncertaintyRetry.gapTop10);
	PgturbohybridJsonbAddFloat8(state, "gap_boundary",
								d->uncertaintyRetry.gapBoundary);
	PgturbohybridJsonbCloseObject(state);	/* uncertainty_retry */

	/* dense.timing_us ------------------------------------------------- */
	PgturbohybridJsonbAddKey(state, "timing_us");
	PgturbohybridJsonbBeginObject(state);
	PgturbohybridJsonbAddInt64(state, "prepare", tm->prepareUs);
	PgturbohybridJsonbAddInt64(state, "traverse", tm->traverseUs);
	PgturbohybridJsonbAddInt64(state, "entry", tm->entryUs);
	PgturbohybridJsonbAddInt64(state, "base", tm->baseUs);
	PgturbohybridJsonbAddInt64(state, "batch", tm->batchUs);
	PgturbohybridJsonbAddInt64(state, "heap", tm->heapUs);
	PgturbohybridJsonbAddInt64(state, "fill", tm->fillUs);
	PgturbohybridJsonbAddInt64(state, "rescore", tm->rescoreUs);
	PgturbohybridJsonbAddInt64(state, "sort", tm->sortUs);
	PgturbohybridJsonbAddInt64(state, "total", tm->totalUs);
	PgturbohybridJsonbAddInt64(state, "entry_sidecar", tm->entrySidecarUs);
	PgturbohybridJsonbAddInt64(state, "payload_entry_seed", tm->payloadEntrySeedUs);
	PgturbohybridJsonbAddInt64(state, "residual_rerank", tm->residualRerankUs);
	PgturbohybridJsonbAddInt64(state, "heap_fetch", tm->heapFetchUs);
	PgturbohybridJsonbAddInt64(state, "heap_rescore", tm->heapRescoreUs);
	PgturbohybridJsonbAddInt64(state, "local_expansion", tm->localExpansionUs);
	PgturbohybridJsonbAddInt64(state, "scan_lock_wait", tm->scanLockWaitUs);
	PgturbohybridJsonbCloseObject(state);	/* timing_us */

	PgturbohybridJsonbCloseObject(state);	/* dense */

	/* bm25 ------------------------------------------------------------ */
	PgturbohybridJsonbAddKey(state, "bm25");
	PgturbohybridJsonbBeginObject(state);
	PgturbohybridJsonbAddBool(state, "branch_available",
							  s->bm25.branchAvailable);
	PgturbohybridJsonbAddBool(state, "branch_used", s->bm25.branchUsed);
	PgturbohybridJsonbAddString(state, "strategy",
								PgturbohybridBm25StrategyName(s->bm25.strategy));
	PgturbohybridJsonbAddString(state, "impact_or_mode",
								PgturbohybridBm25ImpactOrModeName(s->bm25.impactOrMode));
	PgturbohybridJsonbAddString(state, "hybrid_bound",
								PgturbohybridBm25HybridBoundModeName(s->bm25.hybridBound));
	PgturbohybridJsonbAddString(state, "accumulator_mode",
								PgturbohybridBm25AccumulatorModeName(s->bm25.accumulatorMode));
	PgturbohybridJsonbAddInt64(state, "hot_postings_cache_mb",
							   s->bm25.hotPostingsCacheMb);
	PgturbohybridJsonbAddBool(state, "exact_rescore_for_bm25_only",
							  s->bm25.exactRescoreBm25Only);
	PgturbohybridJsonbAddInt64(state, "candidates_effective",
							   s->bm25.candidatesEffective);
	PgturbohybridJsonbAddInt64(state, "k_effective", s->bm25.candidatesEffective);
	PgturbohybridJsonbAddBool(state, "k_defaulted", s->bm25.kDefaulted);
	PgturbohybridJsonbAddBool(state, "cache_hit", s->bm25.cacheHit);
	PgturbohybridJsonbAddUint64(state, "cache_build_us", s->bm25.cacheBuildUs);
	PgturbohybridJsonbAddBool(state, "docstats_loaded_this_query",
							  s->bm25.docstatsLoadedThisQuery);
	PgturbohybridJsonbAddBool(state, "liveness_loaded_this_query",
							  s->bm25.livenessLoadedThisQuery);
	PgturbohybridJsonbAddUint64(state, "docstats_bytes",
								s->bm25.docstatsBytes);
	PgturbohybridJsonbAddUint64(state, "liveness_bytes",
								s->bm25.livenessBytes);
	PgturbohybridJsonbAddBool(state, "cold_cache_o_n_work",
							  s->bm25.coldCacheONWork);
	PgturbohybridJsonbAddFloat8(state, "postings_decode_ratio",
								s->bm25.postingsDecodeRatio);
	PgturbohybridJsonbAddBool(state, "common_term_fallback",
							  s->bm25.commonTermFallback);
	PgturbohybridJsonbAddUint64(state, "wand_pruned",
								s->bm25.wandPruned);
	PgturbohybridJsonbAddBool(state, "hot_postings_cache_hit",
							  s->bm25.hotPostingsCacheHits > 0);
	PgturbohybridJsonbAddUint64(state, "hot_postings_cache_hits",
								s->bm25.hotPostingsCacheHits);
	PgturbohybridJsonbAddUint64(state, "hot_postings_cache_misses",
								s->bm25.hotPostingsCacheMisses);
	PgturbohybridJsonbAddUint64(state, "blocks_pruned_by_fused_score_bound",
								s->bm25.fusedScoreBoundBlocksPruned);
	PgturbohybridJsonbAddUint64(state, "candidates_pruned_by_fused_score_bound",
								s->bm25.fusedScoreBoundCandidatesPruned);
	PgturbohybridJsonbAddString(state, "heap_tsvector_rerank_mode",
								s->bm25.heapTsvectorRerankMode[0] != '\0' ?
								s->bm25.heapTsvectorRerankMode : "off");
	PgturbohybridJsonbAddInt64(state, "heap_tsvector_rerank_count",
							   s->bm25.heapTsvectorRerankCount);
	PgturbohybridJsonbAddUint64(state, "heap_tsvector_rerank_fetch_us",
								s->bm25.heapTsvectorRerankFetchUs);
	PgturbohybridJsonbAddUint64(state, "heap_tsvector_rerank_score_us",
								s->bm25.heapTsvectorRerankScoreUs);
	PgturbohybridJsonbAddBool(state, "heap_tsvector_rerank_topk_changed",
							  s->bm25.heapTsvectorRerankTopKChanged);
	PgturbohybridJsonbAddUint64(state, "elapsed_us", s->bm25.elapsedUs);
	PgturbohybridJsonbCloseObject(state);	/* bm25 */

	/* fusion ---------------------------------------------------------- */
	PgturbohybridJsonbAddKey(state, "fusion");
	PgturbohybridJsonbBeginObject(state);
	PgturbohybridJsonbAddUint64(state, "elapsed_us", s->fusion.elapsedUs);
	PgturbohybridJsonbAddBool(state, "auto_budget", s->fusion.autoBudget);
	PgturbohybridJsonbAddBool(state, "fast_weighted_enabled",
							  s->fusion.fastWeightedEnabled);
	PgturbohybridJsonbAddFloat8(state, "fast_weighted_alpha",
								s->fusion.fastWeightedAlpha);
	PgturbohybridJsonbAddBool(state, "calibrated_fusion_enabled",
							  s->fusion.calibratedFusionEnabled);
	PgturbohybridJsonbAddString(state, "calibrated_fusion_query_shape",
								s->fusion.calibratedFusionQueryShape[0] != '\0' ?
								s->fusion.calibratedFusionQueryShape : "none");
	PgturbohybridJsonbAddFloat8(state, "calibrated_fusion_alpha_effective",
								s->fusion.calibratedFusionAlphaEffective);
	PgturbohybridJsonbAddFloat8(state, "calibrated_fusion_both_match_bonus",
								s->fusion.calibratedFusionBothMatchBonus);
	PgturbohybridJsonbAddString(state, "calibrated_fusion_dense_norm_mode",
								s->fusion.calibratedFusionDenseNormMode[0] != '\0' ?
								s->fusion.calibratedFusionDenseNormMode : "none");
	PgturbohybridJsonbAddString(state, "calibrated_fusion_bm25_norm_mode",
								s->fusion.calibratedFusionBm25NormMode[0] != '\0' ?
								s->fusion.calibratedFusionBm25NormMode : "none");
	PgturbohybridJsonbAddBool(state, "dbsf_enabled",
							  s->fusion.dbsfEnabled);
	PgturbohybridJsonbAddFloat8Array(state, "dbsf_branch_mean",
									 s->fusion.dbsfBranchMean, 2);
	PgturbohybridJsonbAddFloat8Array(state, "dbsf_branch_stddev",
									 s->fusion.dbsfBranchStddev, 2);
	PgturbohybridJsonbAddFloat8Array(state, "dbsf_branch_min",
									 s->fusion.dbsfBranchMin, 2);
	PgturbohybridJsonbAddFloat8Array(state, "dbsf_branch_max",
									 s->fusion.dbsfBranchMax, 2);
	PgturbohybridJsonbAddInt64(state, "dbsf_degenerate_branches",
							   s->fusion.dbsfDegenerateBranches);
	PgturbohybridJsonbAddString(state, "bm25_norm_mode",
								s->fusion.bm25NormMode[0] != '\0' ?
								s->fusion.bm25NormMode : "none");
	PgturbohybridJsonbAddString(state, "dense_norm_mode",
								s->fusion.denseNormMode[0] != '\0' ?
								s->fusion.denseNormMode : "none");
	PgturbohybridJsonbAddString(state, "hybrid_budget_policy",
								s->fusion.hybridBudgetPolicy[0] != '\0' ?
								s->fusion.hybridBudgetPolicy : "fixed");
	PgturbohybridJsonbAddString(state, "hybrid_query_shape",
								s->fusion.hybridQueryShape[0] != '\0' ?
								s->fusion.hybridQueryShape : "fixed");
	PgturbohybridJsonbAddInt64(state, "hybrid_dense_k_chosen",
							   s->fusion.hybridDenseKChosen);
	PgturbohybridJsonbAddInt64(state, "hybrid_bm25_k_chosen",
							   s->fusion.hybridBm25KChosen);
	PgturbohybridJsonbAddString(state, "hybrid_budget_reason",
								s->fusion.hybridBudgetReason[0] != '\0' ?
								s->fusion.hybridBudgetReason : "fixed_policy");
	PgturbohybridJsonbAddString(state, "strategy",
								s->fusion.strategy[0] != '\0' ?
								s->fusion.strategy : "none");
	PgturbohybridJsonbAddInt64(state, "candidates_seen",
							   s->fusion.candidatesSeen);
	PgturbohybridJsonbAddUint64(state, "duplicates",
								s->fusion.duplicates);
	PgturbohybridJsonbAddUint64(state, "heap_replacements",
								s->fusion.heapReplacements);
	PgturbohybridJsonbAddBool(state, "generation_array_reused",
							  s->fusion.generationArrayReused);
	PgturbohybridJsonbAddBool(state, "generation_array_reset",
							  s->fusion.generationArrayReset);
	PgturbohybridJsonbAddString(state, "final_diversity_mode",
								s->fusion.finalDiversityMode[0] != '\0' ?
								s->fusion.finalDiversityMode : "off");
	PgturbohybridJsonbAddInt64(state, "final_diversity_payload_slot",
							   s->fusion.finalDiversityPayloadSlot);
	PgturbohybridJsonbAddInt64(state, "final_diversity_pool_size",
							   s->fusion.finalDiversityPoolSize);
	PgturbohybridJsonbAddInt64(state, "final_diversity_selected",
							   s->fusion.finalDiversitySelected);
	PgturbohybridJsonbAddUint64(state, "final_diversity_duplicate_groups_suppressed",
								s->fusion.finalDiversityDuplicateGroupsSuppressed);
	PgturbohybridJsonbAddUint64(state, "final_diversity_us",
								s->fusion.finalDiversityUs);
	PgturbohybridJsonbCloseObject(state);	/* fusion */

	/* query ----------------------------------------------------------- */
	PgturbohybridJsonbAddKey(state, "query");
	PgturbohybridJsonbBeginObject(state);
	PgturbohybridJsonbAddString(state, "index_shape",
								s->query.indexShape[0] != '\0' ?
								s->query.indexShape : "unknown");
	PgturbohybridJsonbAddInt64(state, "dimensions", s->query.dimensions);
	PgturbohybridJsonbAddInt64(state, "quantization_bits", s->query.quantizationBits);
	if (s->query.exactStorageKnown)
	{
		PgturbohybridJsonbAddBool(state, "exact_storage", s->query.exactStorage);
		PgturbohybridJsonbAddBool(state, "exact_free", !s->query.exactStorage);
	}
	else
	{
		PgturbohybridJsonbAddNull(state, "exact_storage");
		PgturbohybridJsonbAddNull(state, "exact_free");
	}
	PgturbohybridJsonbAddBool(state, "query_split_enabled", s->query.querySplitEnabled);
	PgturbohybridJsonbAddInt64(state, "dense_requested_k", s->query.denseRequestedK);
	PgturbohybridJsonbAddInt64(state, "effective_result_target",
							   s->query.effectiveResultTarget);
	PgturbohybridJsonbAddInt64(state, "effective_search_ef",
							   s->query.effectiveSearchEf);
	PgturbohybridJsonbAddInt64(state, "dense_candidates_effective",
							   s->query.denseCandidatesEffective);
	PgturbohybridJsonbAddInt64(state, "dense_k_effective",
							   s->query.denseCandidatesEffective);
	PgturbohybridJsonbAddBool(state, "dense_k_defaulted", s->query.denseKDefaulted);
	PgturbohybridJsonbAddInt64(state, "final_k_requested", s->query.finalKRequested);
	PgturbohybridJsonbAddInt64(state, "final_k_effective", s->query.finalKEffective);
	PgturbohybridJsonbAddInt64(state, "detected_sql_limit", s->query.detectedSqlLimit);
	PgturbohybridJsonbAddBool(state, "final_k_inferred", s->query.finalKInferred);
	PgturbohybridJsonbAddString(state, "final_k_source",
								PgturbohybridFinalKSourceName());
	PgturbohybridJsonbCloseObject(state);	/* query */
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_last_build_stats);
FUNCTION_PREFIX Datum
pgturbohybrid_last_build_stats(PG_FUNCTION_ARGS)
{
	PgturbohybridJsonbState state;
	const PgturbohybridNativeBuildStatsSnapshot *s =
		&pgturbohybrid_last_native_build_stats;

	PgturbohybridJsonbStateInit(&state);
	PgturbohybridJsonbBeginObject(&state);
	PgturbohybridJsonbAddInt64(&state, "version", 1);
	PgturbohybridJsonbAddBool(&state, "available",
							  pgturbohybrid_last_native_build_stats_valid);

	if (!pgturbohybrid_last_native_build_stats_valid)
		PG_RETURN_JSONB_P(PgturbohybridJsonbEndObject(&state));

	PgturbohybridJsonbAddString(&state, "relation_name", s->relationName);
	PgturbohybridJsonbAddInt64(&state, "relid", (int64) s->relid);
	PgturbohybridJsonbAddString(&state, "index_shape", s->indexShape);
	PgturbohybridJsonbAddUint64(&state, "node_count", s->nodeCount);
	PgturbohybridJsonbAddInt64(&state, "dimensions", s->dimensions);
	PgturbohybridJsonbAddInt64(&state, "quantization_bits",
							   s->quantizationBits);
	PgturbohybridJsonbAddInt64(&state, "m", s->m);
	PgturbohybridJsonbAddInt64(&state, "ef_construction",
							   s->efConstruction);
	PgturbohybridJsonbAddBool(&state, "exact_storage", s->exactStorage);
	PgturbohybridJsonbAddBool(&state, "build_code_only", s->buildCodeOnly);
	PgturbohybridJsonbAddBool(&state, "build_fast_edges", s->buildFastEdges);
	PgturbohybridJsonbAddString(&state, "dense_build_distance_mode",
								PgturbohybridDenseBuildDistanceName(s->buildDistanceMode));
	PgturbohybridJsonbAddString(&state, "build_neighbor_select",
								s->buildFastEdges ? "fast" : "heuristic");
	PgturbohybridJsonbAddString(&state, "build_neighbor_select_reason",
								PgturbohybridBuildNeighborSelectReasonName(s->buildNeighborSelectReason));
	PgturbohybridJsonbAddUint64(&state, "build_distance_calls",
								s->buildDistanceCalls);
	PgturbohybridJsonbAddUint64(&state, "build_distance_query_split",
								s->buildDistanceQuerySplit);
	PgturbohybridJsonbAddUint64(&state, "build_distance_packed",
								s->buildDistancePacked);
	PgturbohybridJsonbAddUint64(&state, "build_distance_weighted",
								s->buildDistanceWeighted);
	PgturbohybridJsonbAddUint64(&state, "build_distance_code_code",
								s->buildDistanceCodeCode);
	PgturbohybridJsonbAddUint64(&state, "build_distance_exact",
								s->buildDistanceExact);
	PgturbohybridJsonbAddUint64(&state, "build_distance_fallback",
								s->buildDistanceFallback);
	PgturbohybridJsonbAddUint64(&state, "multivector_doc_exact_build_distance_calls",
								s->multivectorDocExactBuildDistanceCalls);
	PgturbohybridJsonbAddUint64(&state, "multivector_doc_exact_build_distance_us",
								s->multivectorDocExactBuildDistanceUs);
	PgturbohybridJsonbAddUint64(&state, "multivector_centroid_build_docs",
								s->multivectorCentroidBuildDocs);
	PgturbohybridJsonbAddUint64(&state, "multivector_centroid_build_us",
								s->multivectorCentroidBuildUs);
	PgturbohybridJsonbAddUint64(&state, "multivector_centroid_cluster_us",
								s->multivectorCentroidClusterUs);
	PgturbohybridJsonbAddUint64(&state, "multivector_centroid_residual_us",
								s->multivectorCentroidResidualUs);
	PgturbohybridJsonbAddUint64(&state, "multivector_centroid_build_vectors",
								s->multivectorCentroidBuildVectors);
	PgturbohybridJsonbAddUint64(&state, "multivector_centroid_posting_count",
								s->multivectorCentroidPostingCount);
	PgturbohybridJsonbAddUint64(&state, "multivector_centroid_posting_write_us",
								s->multivectorCentroidPostingWriteUs);
	PgturbohybridJsonbAddUint64(&state, "multivector_centroid_sidecar_write_us",
								s->multivectorCentroidSidecarWriteUs);
	PgturbohybridJsonbAddUint64(&state, "multivector_doc_sidecar_write_us",
								s->multivectorDocSidecarWriteUs);
	PgturbohybridJsonbAddUint64(&state, "multivector_proxy_build_us",
								s->multivectorProxyBuildUs);
	PgturbohybridJsonbAddUint64(&state, "build_distance_cache_hits",
								s->buildDistanceCacheHits);
	PgturbohybridJsonbAddUint64(&state, "build_distance_cache_misses",
								s->buildDistanceCacheMisses);
	PgturbohybridJsonbAddUint64(&state, "build_distance_cache_stores",
								s->buildDistanceCacheStores);
	PgturbohybridJsonbAddUint64(&state, "build_distance_cache_collisions",
								s->buildDistanceCacheCollisions);
	PgturbohybridJsonbAddUint64(&state, "build_edges_distance_calls",
								s->buildEdgeDistanceCalls);
	PgturbohybridJsonbAddUint64(&state, "build_edges_search_layer_us",
								s->buildEdgeSearchLayerUs);
	PgturbohybridJsonbAddUint64(&state, "build_edges_select_neighbor_us",
								s->buildEdgeSelectNeighborUs);
	PgturbohybridJsonbAddUint64(&state, "build_edges_add_neighbor_us",
								s->buildEdgeAddNeighborUs);
	PgturbohybridJsonbAddUint64(&state, "build_edges_prune_neighbor_us",
								s->buildEdgePruneNeighborUs);
	PgturbohybridJsonbAddUint64(&state, "build_edges_entry_update_us",
								s->buildEdgeEntryUpdateUs);
	PgturbohybridJsonbAddInt64(&state, "build_edges_max_frontier_size",
							   s->buildEdgeMaxFrontierSize);
	PgturbohybridJsonbAddFloat8(&state, "build_edges_average_nearest_count",
								s->buildEdgeNearestSamples > 0 ?
								(double) s->buildEdgeNearestTotal /
								(double) s->buildEdgeNearestSamples : 0.0);
	PgturbohybridJsonbAddUint64(&state, "fit_correction_scan_us",
								s->fitCorrectionScanUs);
	PgturbohybridJsonbAddUint64(&state, "scan_us", s->scanUs);
	PgturbohybridJsonbAddUint64(&state, "fit_correction_us",
								s->fitCorrectionUs);
	PgturbohybridJsonbAddUint64(&state, "encode_us", s->encodeUs);
	PgturbohybridJsonbAddUint64(&state, "build_edges_us",
								s->buildEdgesUs);
	PgturbohybridJsonbAddUint64(&state, "free_exact_vectors_us",
								s->freeExactVectorsUs);
	PgturbohybridJsonbAddUint64(&state, "reorder_nodes_us",
								s->reorderNodesUs);
	PgturbohybridJsonbAddUint64(&state, "connect_backbone_us",
								s->connectBackboneUs);
	PgturbohybridJsonbAddUint64(&state, "entry_sidecar_us",
								s->entrySidecarUs);
	PgturbohybridJsonbAddUint64(&state, "write_pages_us",
								s->writePagesUs);
	PgturbohybridJsonbAddUint64(&state, "wal_us", s->walUs);
	PgturbohybridJsonbAddUint64(&state, "total_us", s->totalUs);
	PgturbohybridJsonbAddInt64(&state, "worker_count", s->workerCount);
	PgturbohybridJsonbAddInt64(&state, "native_segments",
							   s->nativeSegmentCount);
	PgturbohybridJsonbAddInt64(&state, "native_segment_bytes",
							   s->nativeSegmentBytes);
	PgturbohybridJsonbAddBool(&state, "parallel_segment_build_enabled",
							  s->parallelSegmentBuildEnabled);
	PgturbohybridJsonbAddString(&state, "segment_build_mode",
								s->segmentBuildMode);
	PgturbohybridJsonbAddInt64(&state, "native_build_workers_requested",
							   s->nativeBuildWorkersRequested);
	PgturbohybridJsonbAddInt64(&state, "native_build_workers_launched",
							   s->nativeBuildWorkersLaunched);
	PgturbohybridJsonbAddBool(&state, "parallel_fit_enabled",
							  s->parallelFitEnabled);
	PgturbohybridJsonbAddBool(&state, "parallel_scan_enabled",
							  s->parallelScanEnabled);
	PgturbohybridJsonbAddBool(&state, "parallel_encode_enabled",
							  s->parallelEncodeEnabled);
	PgturbohybridJsonbAddBool(&state, "parallel_edge_build_enabled",
							  s->parallelEdgeBuildEnabled);
	PgturbohybridJsonbAddString(&state, "parallel_edge_build_disabled_reason",
								PgturbohybridParallelEdgeBuildDisabledReasonName(
									s->parallelEdgeBuildDisabledReason));
	PgturbohybridJsonbAddInt64(&state, "parallel_edge_segments",
							   s->parallelEdgeSegments);
	PgturbohybridJsonbAddInt64(&state, "parallel_edge_workers_launched",
							   s->parallelEdgeWorkersLaunched);
	PgturbohybridJsonbAddUint64(&state, "parallel_edge_repair_us",
								s->parallelEdgeRepairUs);
	PgturbohybridJsonbAddUint64(&state, "worker_merge_us",
								s->workerMergeUs);
	PgturbohybridJsonbAddUint64Array(&state, "worker_scan_us",
									 s->workerScanUs,
									 s->workerScanUsCount);

	PG_RETURN_JSONB_P(PgturbohybridJsonbEndObject(&state));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_graph_repair_dry_run);
FUNCTION_PREFIX Datum
pgturbohybrid_graph_repair_dry_run(PG_FUNCTION_ARGS)
{
	Oid			indexOid = PG_GETARG_OID(0);
	int32		sampleNodes = PG_GETARG_INT32(1);
	int32		searchEf = PG_GETARG_INT32(2);
	int32		candidateLimit = PG_GETARG_INT32(3);
	Relation	index;
	PgturbohybridGraphRepairDryRunStats stats;
	PgturbohybridJsonbState state;

	index = index_open(indexOid, AccessShareLock);
	PG_TRY();
	{
		if (RelationGetForm(index)->relkind != RELKIND_INDEX &&
			RelationGetForm(index)->relkind != RELKIND_PARTITIONED_INDEX)
			ereport(ERROR,
					(errcode(ERRCODE_WRONG_OBJECT_TYPE),
					 errmsg("\"%s\" is not an index",
							RelationGetRelationName(index))));

		PgturbohybridGraphRepairDryRun(index, sampleNodes, searchEf,
									   candidateLimit, &stats);
	}
	PG_CATCH();
	{
		index_close(index, AccessShareLock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	index_close(index, AccessShareLock);

	PgturbohybridJsonbStateInit(&state);
	PgturbohybridJsonbBeginObject(&state);
	PgturbohybridJsonbAddInt64(&state, "version", 1);
	PgturbohybridJsonbAddBool(&state, "dry_run", true);
	PgturbohybridJsonbAddBool(&state, "writes_index_pages", false);
	PgturbohybridJsonbAddBool(&state, "requires_access_exclusive_lock", false);
	PgturbohybridJsonbAddInt64(&state, "node_count", stats.nodeCount);
	PgturbohybridJsonbAddInt64(&state, "dimensions", stats.dimensions);
	PgturbohybridJsonbAddInt64(&state, "sampled_nodes", stats.sampledNodes);
	PgturbohybridJsonbAddFloat8(&state, "avg_overlap", stats.avgOverlap);
	PgturbohybridJsonbAddInt64(&state, "weak_nodes", stats.weakNodes);
	PgturbohybridJsonbAddUint64(&state, "missed_neighbor_count",
								stats.missedNeighbors);
	PgturbohybridJsonbAddUint64(&state, "weak_entry_navigation_cases",
								stats.weakEntryCases);
	PgturbohybridJsonbAddUint64(&state, "suggested_edges",
								stats.suggestedEdges);
	PgturbohybridJsonbAddUint64(&state, "elapsed_ms", stats.elapsedMs);
	PgturbohybridJsonbAddUint64(&state, "code_pages_read",
								stats.codePagesRead);
	PgturbohybridJsonbAddUint64(&state, "adj_pages_read",
								stats.adjPagesRead);
	PgturbohybridJsonbAddKey(&state, "parameters");
	PgturbohybridJsonbBeginObject(&state);
	PgturbohybridJsonbAddInt64(&state, "sample_nodes",
							   stats.sampleNodesRequested);
	PgturbohybridJsonbAddInt64(&state, "search_ef", stats.searchEf);
	PgturbohybridJsonbAddInt64(&state, "candidate_limit",
							   stats.candidateLimit);
	PgturbohybridJsonbCloseObject(&state);

	PG_RETURN_JSONB_P(PgturbohybridJsonbEndObject(&state));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_proxy_diagnostics);
FUNCTION_PREFIX Datum
pgturbohybrid_multivector_proxy_diagnostics(PG_FUNCTION_ARGS)
{
	Oid			indexOid = PG_GETARG_OID(0);
	int32		sampleDocsArg = PG_GETARG_INT32(1);
	int32		queryCountArg = PG_GETARG_INT32(2);
	Relation	index = NULL;
	PgturbohybridGraphMetaPageData meta;
	PgturbohybridGraphScanStorage storage;
	PgturbohybridMultiVectorDocSidecarAccessStats sidecarStats;
	PgturbohybridOptions *opts;
	MemoryContext workCtx = NULL;
	MemoryContext oldCtx = NULL;
	PgturbohybridMultiVectorProxyDiagnosticResult result;
	PgturbohybridJsonbState state;

	memset(&result, 0, sizeof(result));
	memset(&storage, 0, sizeof(storage));
	memset(&sidecarStats, 0, sizeof(sidecarStats));

	if (sampleDocsArg < 2 || sampleDocsArg > 10000)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("sample_docs must be between 2 and 10000"),
				 errhint("Use a bounded sample; this diagnostic is intentionally not a full-corpus benchmark.")));
	if (queryCountArg < 1 || queryCountArg > 1000)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("query_count must be between 1 and 1000"),
				 errhint("Use a bounded query sample; this diagnostic is intentionally not a full-corpus benchmark.")));

	index = index_open(indexOid, AccessShareLock);
	workCtx = AllocSetContextCreate(CurrentMemoryContext,
									"pgturbohybrid multivector proxy diagnostics",
									ALLOCSET_DEFAULT_SIZES);

	PG_TRY();
	{
		uint32		sampleDocs;
		uint32		queryCount;
		uint32	   *docIds;
		PgturbohybridMultiVector **docs;
		Vector	  **proxies;
		PgturbohybridMultiVectorProxyDiagnosticScore *exactScores;
		PgturbohybridMultiVectorProxyDiagnosticScore *proxyScores;
		int		   *exactRanks;
		int		   *proxyRanks;
		double		tokenSum = 0.0;
		double		recallSum = 0.0;
		double		correlationSum = 0.0;
		uint32		correlationCount = 0;
		uint32		recommendedK = 0;

		if (RelationGetForm(index)->relkind != RELKIND_INDEX &&
			RelationGetForm(index)->relkind != RELKIND_PARTITIONED_INDEX)
			ereport(ERROR,
					(errcode(ERRCODE_WRONG_OBJECT_TYPE),
					 errmsg("\"%s\" is not an index",
							RelationGetRelationName(index))));
		if (!PgturbohybridGraphReadMeta(index, &meta))
			ereport(ERROR,
					(errcode(ERRCODE_WRONG_OBJECT_TYPE),
					 errmsg("\"%s\" is not a valid pgturbohybrid index",
							RelationGetRelationName(index))));
		if (meta.tqMultivectorGraphMode !=
			PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES ||
			meta.tqMultivectorDocCount == 0 ||
			!BlockNumberIsValid(meta.tqMultivectorDocMapStartBlkno))
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("\"%s\" is not a document-node multivector index",
							RelationGetRelationName(index)),
					 errhint("Create the index with multivector_graph = document_nodes so document proxies and sidecar multivectors are available.")));

		opts = (PgturbohybridOptions *) index->rd_options;
		result.corpusDocs = meta.tqMultivectorDocCount;
		result.sampleDocsRequested = (uint32) sampleDocsArg;
		result.queryCountRequested = (uint32) queryCountArg;
		result.dimensions = meta.dimensions;
		result.proxyEncoder =
			opts != NULL ? opts->multivectorProxyEncoder :
			PGTURBOHYBRID_DEFAULT_MULTIVECTOR_PROXY_ENCODER;
		sampleDocs =
			Min((uint32) sampleDocsArg, meta.tqMultivectorDocCount);
		if (sampleDocs < 2)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("document-node multivector diagnostics require at least two documents")));
		queryCount = Min((uint32) queryCountArg, sampleDocs);

		oldCtx = MemoryContextSwitchTo(workCtx);
		docIds = palloc0(sizeof(uint32) * (Size) sampleDocs);
		docs = palloc0(sizeof(PgturbohybridMultiVector *) * (Size) sampleDocs);
		proxies = palloc0(sizeof(Vector *) * (Size) sampleDocs);
		exactScores =
			palloc0(sizeof(PgturbohybridMultiVectorProxyDiagnosticScore) *
					(Size) sampleDocs);
		proxyScores =
			palloc0(sizeof(PgturbohybridMultiVectorProxyDiagnosticScore) *
					(Size) sampleDocs);
		exactRanks = palloc0(sizeof(int) * (Size) sampleDocs);
		proxyRanks = palloc0(sizeof(int) * (Size) sampleDocs);
		PgturbohybridGraphInitScanStorage(index, &meta, &storage, NULL);
		storage.ctx = workCtx;
		if (!PgturbohybridGraphLoadMultiVectorDocMapWithStats(index, &meta,
															  &storage,
															  false,
															  &sidecarStats) ||
			!storage.multivectorDocMapLoaded ||
			!storage.multivectorDocVectorsLoaded)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("document-node multivector sidecar is not available"),
					 errhint("REINDEX with multivector_graph = document_nodes to rebuild the document sidecar.")));

		for (uint32 i = 0; i < sampleDocs; i++)
		{
			uint64		docId =
				((uint64) i * (uint64) meta.tqMultivectorDocCount) /
				(uint64) sampleDocs;

			if (docId >= meta.tqMultivectorDocCount)
				docId = meta.tqMultivectorDocCount - 1;
			if (i > 0 && docId <= docIds[i - 1] &&
				docIds[i - 1] + 1 < meta.tqMultivectorDocCount)
				docId = docIds[i - 1] + 1;
			docIds[i] = (uint32) docId;
			docs[i] =
				PgturbohybridGraphLoadMultiVectorDocVector(index, &meta,
														   &storage,
														   docIds[i],
														   workCtx,
														   &sidecarStats);
			proxies[i] =
				PgturbohybridMultiVectorBuildProxyVector(docs[i],
														 result.proxyEncoder,
														 workCtx);
			tokenSum += (double) docs[i]->count;
		}

		for (uint32 qi = 0; qi < queryCount; qi++)
		{
			uint32		querySampleIndex =
				(uint32) (((uint64) qi * (uint64) sampleDocs) /
						  (uint64) queryCount);
			uint32		topK = Min((uint32) 10, sampleDocs - 1);
			uint32		hits = 0;
			uint32		queryRequiredK = topK;
			uint32		rank;
			uint32		rankableCount = sampleDocs - 1;
			double		rankDiffSquaredSum = 0.0;

			if (querySampleIndex >= sampleDocs)
				querySampleIndex = sampleDocs - 1;
			memset(exactRanks, 0, sizeof(int) * (Size) sampleDocs);
			memset(proxyRanks, 0, sizeof(int) * (Size) sampleDocs);

			for (uint32 di = 0; di < sampleDocs; di++)
			{
				exactScores[di].sampleIndex = di;
				proxyScores[di].sampleIndex = di;
				if (di == querySampleIndex)
				{
					exactScores[di].score = -DBL_MAX;
					proxyScores[di].score = -DBL_MAX;
					continue;
				}
				result.exactPairsScored +=
					(uint64) docs[querySampleIndex]->count *
					(uint64) docs[di]->count;
				result.proxyPairsScored++;
				exactScores[di].score =
					TqMultiVectorMaxSim(docs[querySampleIndex], docs[di]);
				proxyScores[di].score =
					TqDotProductF32Scalar(proxies[querySampleIndex]->x,
										  proxies[di]->x,
										  proxies[querySampleIndex]->dim);
			}

			qsort(exactScores, sampleDocs, sizeof(*exactScores),
				  PgturbohybridMultiVectorProxyDiagnosticScoreCompare);
			qsort(proxyScores, sampleDocs, sizeof(*proxyScores),
				  PgturbohybridMultiVectorProxyDiagnosticScoreCompare);

			rank = 1;
			for (uint32 i = 0; i < sampleDocs; i++)
			{
				if (exactScores[i].sampleIndex == querySampleIndex)
					continue;
				exactRanks[exactScores[i].sampleIndex] = (int) rank++;
			}
			rank = 1;
			for (uint32 i = 0; i < sampleDocs; i++)
			{
				if (proxyScores[i].sampleIndex == querySampleIndex)
					continue;
				proxyRanks[proxyScores[i].sampleIndex] = (int) rank++;
			}

			for (uint32 i = 0; i < topK; i++)
			{
				uint32		sampleIndex = exactScores[i].sampleIndex;
				int			proxyRank = proxyRanks[sampleIndex];

				if (proxyRank > 0 && proxyRank <= (int) topK)
					hits++;
				if (proxyRank > (int) queryRequiredK)
					queryRequiredK = (uint32) proxyRank;
			}
			if (queryRequiredK > recommendedK)
				recommendedK = queryRequiredK;
			recallSum += (double) hits / (double) topK;

			if (rankableCount > 1)
			{
				for (uint32 di = 0; di < sampleDocs; di++)
				{
					double		diff;

					if (di == querySampleIndex)
						continue;
					diff = (double) exactRanks[di] - (double) proxyRanks[di];
					rankDiffSquaredSum += diff * diff;
				}
				correlationSum +=
					1.0 -
					(6.0 * rankDiffSquaredSum) /
					((double) rankableCount *
					 ((double) rankableCount * (double) rankableCount - 1.0));
				correlationCount++;
			}
		}
		MemoryContextSwitchTo(oldCtx);

		result.sampleDocs = sampleDocs;
		result.queryCount = queryCount;
		result.avgDocTokens = tokenSum / (double) sampleDocs;
		result.recallAt10ProxyToExact = recallSum / (double) queryCount;
		result.avgProxyExactRankCorrelation =
			correlationCount > 0 ? correlationSum / (double) correlationCount :
			0.0;
		result.recommendedDocCandidateK =
			Max(recommendedK, Min((uint32) 10, sampleDocs - 1));
	}
	PG_CATCH();
	{
		if (oldCtx != NULL)
			MemoryContextSwitchTo(oldCtx);
		if (index != NULL)
			index_close(index, AccessShareLock);
		if (workCtx != NULL)
			MemoryContextDelete(workCtx);
		PG_RE_THROW();
	}
	PG_END_TRY();

	index_close(index, AccessShareLock);
	if (workCtx != NULL)
		MemoryContextDelete(workCtx);

	PgturbohybridJsonbStateInit(&state);
	PgturbohybridJsonbBeginObject(&state);
	PgturbohybridJsonbAddInt64(&state, "version", 1);
	PgturbohybridJsonbAddBool(&state, "read_only", true);
	PgturbohybridJsonbAddBool(&state, "sample_limited", true);
	PgturbohybridJsonbAddBool(&state, "excludes_self_match", true);
	PgturbohybridJsonbAddInt64(&state, "corpus_docs", result.corpusDocs);
	PgturbohybridJsonbAddInt64(&state, "sample_docs_requested",
							   result.sampleDocsRequested);
	PgturbohybridJsonbAddInt64(&state, "query_count_requested",
							   result.queryCountRequested);
	PgturbohybridJsonbAddInt64(&state, "sample_docs", result.sampleDocs);
	PgturbohybridJsonbAddInt64(&state, "query_count", result.queryCount);
	PgturbohybridJsonbAddInt64(&state, "dimensions", result.dimensions);
	PgturbohybridJsonbAddFloat8(&state, "avg_doc_tokens",
								result.avgDocTokens);
	PgturbohybridJsonbAddString(&state, "proxy_encoder",
								PgturbohybridMultiVectorProxyEncoderName(
									result.proxyEncoder));
	PgturbohybridJsonbAddFloat8(&state, "recall_at_10_proxy_to_exact",
								result.recallAt10ProxyToExact);
	PgturbohybridJsonbAddFloat8(&state,
								"avg_proxy_exact_rank_correlation",
								result.avgProxyExactRankCorrelation);
	PgturbohybridJsonbAddInt64(&state, "recommended_doc_candidate_k",
							   result.recommendedDocCandidateK);
	PgturbohybridJsonbAddUint64(&state, "exact_pairs_scored",
								result.exactPairsScored);
	PgturbohybridJsonbAddUint64(&state, "proxy_pairs_scored",
								result.proxyPairsScored);

	PG_RETURN_JSONB_P(PgturbohybridJsonbEndObject(&state));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_last_scan_stats);
FUNCTION_PREFIX Datum
pgturbohybrid_last_scan_stats(PG_FUNCTION_ARGS)
{
	PgturbohybridJsonbState state;
	PgturbohybridScanStatsSnapshot scanStats;
	PgturbohybridValidationStats validationStats;
	uint64		denseElapsedUs;
	uint64		bm25ElapsedUs;
	uint64		fusionElapsedUs;
	uint64		elapsedUs;
	bool		indexUsed;

	PgturbohybridGetLastScanStatsSnapshot(&scanStats);
	PgturbohybridGetValidationStats(&validationStats);
	denseElapsedUs = scanStats.denseElapsedUs != 0 ? scanStats.denseElapsedUs :
		pgturbohybrid_last_graph_total_us;
	bm25ElapsedUs = scanStats.bm25ElapsedUs;
	fusionElapsedUs = scanStats.fusionElapsedUs;
	elapsedUs = scanStats.elapsedUs != 0 ? scanStats.elapsedUs :
		denseElapsedUs + bm25ElapsedUs + fusionElapsedUs;
	indexUsed = pgturbohybrid_last_scan_orchestration != PGTURBOHYBRID_SCAN_ORCHESTRATION_NONE ||
		scanStats.bm25Terms > 0 || bm25ElapsedUs > 0;

	PgturbohybridJsonbStateInit(&state);
	PgturbohybridJsonbBeginObject(&state);
	PgturbohybridJsonbAddInt64(&state, "version", 1);
	PgturbohybridJsonbAddString(&state, "profile",
								PgturbohybridProfileName(pgturbohybrid_profile));
	PgturbohybridJsonbAddBool(&state, "index_used", indexUsed);
	PgturbohybridJsonbAddString(&state, "scan_orchestration",
								TqScanOrchestrationName());
	PgturbohybridJsonbAddString(&state, "graph_storage_kind",
								PgturbohybridGraphStorageKindName(pgturbohybrid_last_graph_storage_kind));
	PgturbohybridJsonbAddString(&state, "index_shape",
								scanStats.indexShape[0] != '\0' ?
								scanStats.indexShape : "unknown");
	PgturbohybridJsonbAddBool(&state, "bm25_branch_available",
							  scanStats.bm25BranchAvailable);
	PgturbohybridJsonbAddBool(&state, "dense_branch_used",
							  scanStats.denseBranchUsed);
	PgturbohybridJsonbAddBool(&state, "bm25_branch_used",
							  scanStats.bm25BranchUsed);
	PgturbohybridJsonbAddBranchPlan(&state, &scanStats.branchPlan);
	PgturbohybridJsonbAddInt64(&state, "quantization_bits",
							   pgturbohybrid_last_graph_quantization_bits);
	if (pgturbohybrid_last_graph_exact_storage_known)
	{
		PgturbohybridJsonbAddBool(&state, "exact_storage",
								  pgturbohybrid_last_graph_exact_storage);
		/* exact_free = code-only index: no stored f32 vectors to rescore. */
		PgturbohybridJsonbAddBool(&state, "exact_free",
								  !pgturbohybrid_last_graph_exact_storage);
	}
	else
	{
		PgturbohybridJsonbAddNull(&state, "exact_storage");
		PgturbohybridJsonbAddNull(&state, "exact_free");
	}
	PgturbohybridJsonbAddBool(&state, "dense_build_exact_distances",
							  pgturbohybrid_last_graph_build_exact_distances);
	PgturbohybridJsonbAddString(&state, "dense_build_distance_mode",
								PgturbohybridDenseBuildDistanceName(pgturbohybrid_last_graph_build_distance_mode));
	PgturbohybridJsonbAddString(&state, "build_neighbor_select",
								pgturbohybrid_last_graph_build_fast_edges ?
								"fast" : "heuristic");
	PgturbohybridJsonbAddString(&state, "build_neighbor_select_reason",
								PgturbohybridBuildNeighborSelectReasonName(pgturbohybrid_last_graph_build_neighbor_select_reason));
	PgturbohybridJsonbAddBool(&state, "build_fast_edges",
							  pgturbohybrid_last_graph_build_fast_edges);

	/*
	 * Dense scan diagnostics: which scorer ran, the approximate-vs-exact and
	 * SIMD-vs-scalar code-scoring split, page/tuple overhead, candidate
	 * budgets, and per-phase timing.  Together these distinguish SIMD scoring
	 * cost (dense_scoring_kernel + graph_batch_us) from page/tuple/heap
	 * overhead (graph_*_pages_read + heap_tuples_returned + graph_heap_us).
	 */
	PgturbohybridJsonbAddInt64(&state, "dimensions",
							   pgturbohybrid_last_graph_dimensions);
	PgturbohybridJsonbAddBool(&state, "query_split_enabled",
							  pgturbohybrid_last_graph_query_split_active);
	/* Exact approximate scorer actually used (scalar_lut / avx2_lut_gather /
	 * signed_split_*); the authoritative answer to "did this scan use the LUT
	 * gather or the integer query split?". */
	PgturbohybridJsonbAddString(&state, "dense_scorer",
								pgturbohybrid_last_graph_u8_split_used ?
								PgturbohybridGraphU8SplitKernelName() :
								PgturbohybridDenseScorerUsedName(
									pgturbohybrid_last_graph_querysplit_used,
									pgturbohybrid_last_graph_scoring_kernel,
									pgturbohybrid_last_graph_scored_codes,
									pgturbohybrid_last_graph_rescore_count));
	PgturbohybridJsonbAddString(&state, "dense_scoring_kernel",
								PgturbohybridDenseScorerName(pgturbohybrid_last_graph_scoring_kernel));
	PgturbohybridJsonbAddString(&state, "dense_batch_kernel",
								PgturbohybridDenseScorerName(pgturbohybrid_last_graph_batch_kernel));
	PgturbohybridJsonbAddString(&state, "dense_scalar_fallback_kernel",
								PgturbohybridDenseScalarFallbackName());
	PgturbohybridJsonbAddString(&state, "dense_exact_kernel",
								TqExactKernelName(pgturbohybrid_last_exact_vector_kernel));
	PgturbohybridJsonbAddInt64(&state, "graph_batch_scored_codes",
							   pgturbohybrid_last_graph_batch_scored_codes);
	PgturbohybridJsonbAddInt64(&state, "graph_simd_scored_codes",
							   Max(0, pgturbohybrid_last_graph_scored_codes -
								   pgturbohybrid_last_graph_scalar_scored_codes));
	PgturbohybridJsonbAddInt64(&state, "graph_scalar_scored_codes",
							   pgturbohybrid_last_graph_scalar_scored_codes);
	PgturbohybridJsonbAddInt64(&state, "heap_tuples_returned",
							   pgturbohybrid_last_graph_returned_rows);
	PgturbohybridJsonbAddInt64(&state, "candidate_objects_allocated",
							   pgturbohybrid_last_graph_candidate_count);
	PgturbohybridJsonbAddInt64(&state, "graph_m",
							   pgturbohybrid_last_graph_m);
	PgturbohybridJsonbAddInt64(&state, "graph_ef_construction",
							   pgturbohybrid_last_graph_ef_construction);
	PgturbohybridJsonbAddInt64(&state, "graph_oversampling",
							   pgturbohybrid_last_graph_oversampling);
	PgturbohybridJsonbAddString(&state, "graph_exact_cache",
								PgturbohybridGraphExactCacheName(pgturbohybrid_last_graph_exact_cache));
	PgturbohybridJsonbAddBool(&state, "graph_exact_cache_active",
							  pgturbohybrid_last_graph_exact_cache != PGTURBOHYBRID_GRAPH_EXACT_CACHE_OFF);
	PgturbohybridJsonbAddBool(&state, "residual_rerank_active",
							  pgturbohybrid_last_graph_residual_rerank_count > 0);
	PgturbohybridJsonbAddString(&state, "dense_heap_rescore",
								PgturbohybridGraphDenseHeapRescoreName(pgturbohybrid_last_graph_heap_rescore_mode));
	PgturbohybridJsonbAddBool(&state, "heap_rescore_auto_enabled",
							  pgturbohybrid_last_graph_heap_rescore_auto_enabled);
	PgturbohybridJsonbAddString(&state, "heap_rescore_reason",
								PgturbohybridGraphDenseHeapRescoreReasonName(pgturbohybrid_last_graph_heap_rescore_reason));
	PgturbohybridJsonbAddBool(&state, "graph_rescore_band_active",
							  pgturbohybrid_last_graph_effective_rescore_band > 0);
	/* The index's configured rescore band (auto/none/exact). */
	PgturbohybridJsonbAddString(&state, "graph_rescore_band",
								PgturbohybridGraphRescoreBandName(pgturbohybrid_last_graph_rescore_band));
	PgturbohybridJsonbAddUint64(&state, "graph_prepare_us",
								(uint64) pgturbohybrid_last_graph_prepare_us);
	PgturbohybridJsonbAddUint64(&state, "graph_traverse_us",
								(uint64) pgturbohybrid_last_graph_traverse_us);
	PgturbohybridJsonbAddUint64(&state, "graph_batch_us",
								(uint64) pgturbohybrid_last_graph_batch_us);
	PgturbohybridJsonbAddUint64(&state, "graph_heap_us",
								(uint64) pgturbohybrid_last_graph_heap_us);
	PgturbohybridJsonbAddUint64(&state, "graph_rescore_us",
								(uint64) pgturbohybrid_last_graph_rescore_us);
	PgturbohybridJsonbAddUint64(&state, "graph_sort_us",
								(uint64) pgturbohybrid_last_graph_sort_us);
	PgturbohybridJsonbAddUint64(&state, "graph_total_us",
								(uint64) pgturbohybrid_last_graph_total_us);

	PgturbohybridJsonbAddInt64(&state, "graph_candidate_count",
							   pgturbohybrid_last_graph_candidate_count);
	PgturbohybridJsonbAddInt64(&state, "graph_visited_nodes",
							   pgturbohybrid_last_graph_visited_nodes);
	PgturbohybridJsonbAddInt64(&state, "graph_scored_codes",
							   pgturbohybrid_last_graph_scored_codes);
	PgturbohybridJsonbAddInt64(&state, "graph_fill_candidate_band_calls",
							   pgturbohybrid_last_graph_fill_candidate_band_calls);
	PgturbohybridJsonbAddString(&state, "graph_fill_candidate_band_reason",
								PgturbohybridGraphFillCandidateBandReasonName(
									pgturbohybrid_last_graph_fill_candidate_band_reason));
	PgturbohybridJsonbAddInt64(&state, "graph_fill_candidate_band_visited",
							   pgturbohybrid_last_graph_fill_candidate_band_visited);
	PgturbohybridJsonbAddInt64(&state, "graph_fill_candidate_band_scored",
							   pgturbohybrid_last_graph_fill_candidate_band_scored);
	PgturbohybridJsonbAddInt64(&state, "graph_fill_candidate_band_selected_before",
							   pgturbohybrid_last_graph_fill_candidate_band_selected_before);
	PgturbohybridJsonbAddInt64(&state, "graph_fill_candidate_band_selected_after",
							   pgturbohybrid_last_graph_fill_candidate_band_selected_after);
	PgturbohybridJsonbAddInt64(&state, "graph_fill_candidate_band_target",
							   pgturbohybrid_last_graph_fill_candidate_band_target);
	PgturbohybridJsonbAddBool(&state, "graph_fill_candidate_band_used_payload_refs",
							  pgturbohybrid_last_graph_fill_candidate_band_used_payload_refs);
	PgturbohybridJsonbAddInt64(&state, "graph_fill_candidate_band_payload_ref_count",
							   pgturbohybrid_last_graph_fill_candidate_band_payload_ref_count);
	PgturbohybridJsonbAddInt64(&state, "graph_code_pages_read",
							   pgturbohybrid_last_graph_code_pages_read);
	PgturbohybridJsonbAddInt64(&state, "graph_adj_pages_read",
							   pgturbohybrid_last_graph_adj_pages_read);
	PgturbohybridJsonbAddInt64(&state, "graph_segment_count",
							   pgturbohybrid_last_graph_segment_count);
	PgturbohybridJsonbAddInt64(&state, "graph_segments_searched",
							   pgturbohybrid_last_graph_segments_searched);
	PgturbohybridJsonbAddInt64(&state, "native_segments",
							   pgturbohybrid_last_graph_segment_count);
	PgturbohybridJsonbAddString(&state, "per_segment_budget_mode",
								PgturbohybridNativeSegmentBudgetName(pgturbohybrid_last_graph_per_segment_budget_mode));
	PgturbohybridJsonbAddInt64(&state, "effective_search_ef_before_segment_scaling",
							   pgturbohybrid_last_graph_search_ef_before_segment_scaling);
	PgturbohybridJsonbAddInt64(&state, "effective_search_ef_after_segment_scaling",
							   pgturbohybrid_last_graph_search_ef_after_segment_scaling);
	/*
	 * Native code-page cache effectiveness (PgturbohybridGraphLoadCodePage).
	 * A warm scan should be served almost entirely from already-loaded code
	 * pages / the in-memory codeArena: cache_hits dominate load_attempts,
	 * cache_misses ~ code_pages_read ~ 0, and pages_read_per_scored_code ~ 0.
	 * arena_used_bytes is the code copied into storage THIS scan (0 when fully
	 * served from the cross-scan native cache); arena_allocated_bytes is the
	 * contiguous code-arena size (0 when the index exceeds the native-cache
	 * cap and falls back to per-node code buffers).
	 */
	PgturbohybridJsonbAddKey(&state, "graph_code_pages");
	PgturbohybridJsonbBeginObject(&state);
	PgturbohybridJsonbAddInt64(&state, "load_attempts",
							   pgturbohybrid_last_graph_code_page_attempts);
	PgturbohybridJsonbAddInt64(&state, "cache_hits",
							   pgturbohybrid_last_graph_code_page_hits);
	PgturbohybridJsonbAddInt64(&state, "cache_misses",
							   pgturbohybrid_last_graph_code_page_misses);
	PgturbohybridJsonbAddInt64(&state, "code_pages_read",
							   pgturbohybrid_last_graph_code_pages_read);
	PgturbohybridJsonbAddInt64(&state, "code_tuples_copied",
							   pgturbohybrid_last_graph_code_tuples_copied);
	PgturbohybridJsonbAddInt64(&state, "arena_used_bytes",
							   pgturbohybrid_last_graph_code_arena_used_bytes);
	PgturbohybridJsonbAddInt64(&state, "arena_allocated_bytes",
							   pgturbohybrid_last_graph_code_arena_allocated_bytes);
	PgturbohybridJsonbAddFloat8(&state, "hit_rate",
								pgturbohybrid_last_graph_code_page_attempts > 0 ?
								(double) pgturbohybrid_last_graph_code_page_hits /
								(double) pgturbohybrid_last_graph_code_page_attempts : 0.0);
	PgturbohybridJsonbAddFloat8(&state, "pages_read_per_scored_code",
								pgturbohybrid_last_graph_scored_codes > 0 ?
								(double) pgturbohybrid_last_graph_code_pages_read /
								(double) pgturbohybrid_last_graph_scored_codes : 0.0);
	PgturbohybridJsonbCloseObject(&state);
	PgturbohybridJsonbAddInt64(&state, "graph_entry_point_count",
							   pgturbohybrid_last_graph_entry_point_count);
	PgturbohybridJsonbAddInt64(&state, "graph_entry_sample_configured",
							   pgturbohybrid_last_graph_entry_sample_configured);
	PgturbohybridJsonbAddInt64(&state, "graph_entry_sample_effective",
							   pgturbohybrid_last_graph_entry_sample_effective);
	PgturbohybridJsonbAddInt64(&state, "graph_entry_sample_scored",
							   pgturbohybrid_last_graph_entry_sample_scored);
	PgturbohybridJsonbAddInt64(&state, "graph_entry_sidecar_count",
							   pgturbohybrid_last_graph_entry_sidecar_count);
	PgturbohybridJsonbAddInt64(&state, "graph_entry_sidecar_scored",
							   pgturbohybrid_last_graph_entry_sidecar_scored);
	PgturbohybridJsonbAddInt64(&state, "graph_entry_sidecar_selected",
							   pgturbohybrid_last_graph_entry_sidecar_selected);
	PgturbohybridJsonbAddInt64(&state, "graph_entry_sidecar_representatives_configured",
							   pgturbohybrid_last_graph_entry_sidecar_representatives_configured);
	PgturbohybridJsonbAddString(&state, "graph_entry_sidecar_strategy",
								PgturbohybridEntrySidecarStrategyName(
									pgturbohybrid_last_graph_entry_sidecar_strategy));
	PgturbohybridJsonbAddInt64(&state, "graph_entry_sidecar_us",
							   pgturbohybrid_last_graph_entry_sidecar_us);
	PgturbohybridJsonbAddString(&state, "payload_entry_seeding_mode",
								PgturbohybridPayloadEntrySeedingName(
									pgturbohybrid_last_payload_entry_seeding_mode));
	PgturbohybridJsonbAddBool(&state, "payload_entry_seeding_hit",
							  pgturbohybrid_last_payload_entry_seeding_hit);
	PgturbohybridJsonbAddInt64(&state, "payload_entry_seed_count",
							   pgturbohybrid_last_payload_entry_seed_count);
	PgturbohybridJsonbAddInt64(&state, "payload_entry_seed_payload_slot",
							   pgturbohybrid_last_payload_entry_seed_payload_slot);
	PgturbohybridJsonbAddInt64(&state, "payload_entry_seed_range_count",
							   pgturbohybrid_last_payload_entry_seed_range_count);
	PgturbohybridJsonbAddInt64(&state, "payload_entry_seed_us",
							   pgturbohybrid_last_payload_entry_seed_us);
	PgturbohybridJsonbAddInt64(&state, "dense_residual_rerank_count",
							   pgturbohybrid_last_graph_residual_rerank_count);
	PgturbohybridJsonbAddInt64(&state, "dense_residual_rerank_bytes",
							   pgturbohybrid_last_graph_residual_rerank_bytes);
	PgturbohybridJsonbAddInt64(&state, "dense_residual_rerank_us",
							   pgturbohybrid_last_graph_residual_rerank_us);
	PgturbohybridJsonbAddString(&state, "residual_rerank_mode",
								PgturbohybridGraphDenseResidualRerankModeName(
									pgturbohybrid_last_graph_residual_rerank_mode));
	PgturbohybridJsonbAddFloat8(&state, "residual_rerank_weight_effective",
								pgturbohybrid_last_graph_residual_rerank_weight_effective);
	PgturbohybridJsonbAddInt64(&state, "residual_rerank_band",
							   pgturbohybrid_last_graph_residual_rerank_band);
	PgturbohybridJsonbAddInt64(&state, "residual_rerank_band_multiplier",
							   pgturbohybrid_last_graph_residual_rerank_band_multiplier);
	PgturbohybridJsonbAddFloat8(&state, "residual_rerank_max_adjustment",
								pgturbohybrid_last_graph_residual_rerank_max_adjustment);
	PgturbohybridJsonbAddInt64(&state, "residual_rerank_reordered_count",
							   pgturbohybrid_last_graph_residual_rerank_reordered_count);
	PgturbohybridJsonbAddBool(&state, "residual_rerank_topk_changed",
							  pgturbohybrid_last_graph_residual_rerank_topk_changed);
	PgturbohybridJsonbAddInt64(&state, "heap_rescore_count",
							   pgturbohybrid_last_graph_heap_rescore_count);
	PgturbohybridJsonbAddInt64(&state, "heap_rescore_us",
							   pgturbohybrid_last_graph_heap_rescore_us);
	PgturbohybridJsonbAddInt64(&state, "heap_fetch_us",
							   pgturbohybrid_last_graph_heap_fetch_us);
	PgturbohybridJsonbAddString(&state, "exact_rescore_source",
								PgturbohybridGraphExactRescoreSourceName(pgturbohybrid_last_graph_exact_rescore_source));
	PgturbohybridJsonbAddInt64(&state, "graph_rescore_count",
							   pgturbohybrid_last_graph_rescore_count);
	PgturbohybridJsonbAddInt64(&state, "graph_rescore_pages",
							   pgturbohybrid_last_graph_rescore_pages);
	/* Exact rescore count == graph_rescore_count; named explicitly so the
	 * scoring-kernel diagnostic is self-contained. */
	PgturbohybridJsonbAddInt64(&state, "exact_rescore_count",
							   pgturbohybrid_last_graph_rescore_count);
	/*
	 * Per-kernel attribution of the native dense scoring work: which exact
	 * kernel inside PgturbohybridGraphScoreNodeBatch()/PgturbohybridGraphScoreNode()
	 * scored how many nodes (and how many kernel calls).  This is the
	 * authoritative answer to "is the hot path u8-split, signed-split, or
	 * scalar/LUT?".  Nodes summed across buckets == graph_scored_codes.
	 */
	PgturbohybridJsonbAddKey(&state, "graph_score_kernels");
	PgturbohybridJsonbBeginObject(&state);
	for (int b = 0; b < PGTURBOHYBRID_SCORE_KERNEL_BUCKET_COUNT; b++)
	{
		PgturbohybridJsonbAddKey(&state, PgturbohybridGraphScoreKernelBucketName(b));
		PgturbohybridJsonbBeginObject(&state);
		PgturbohybridJsonbAddInt64(&state, "nodes",
								   pgturbohybrid_last_graph_score_kernel_nodes[b]);
		PgturbohybridJsonbAddInt64(&state, "calls",
								   pgturbohybrid_last_graph_score_kernel_calls[b]);
		PgturbohybridJsonbCloseObject(&state);
	}
	PgturbohybridJsonbCloseObject(&state);
	PgturbohybridJsonbAddInt64(&state, "graph_batch_calls",
							   pgturbohybrid_last_graph_batch_calls);
	PgturbohybridJsonbAddInt64(&state, "graph_batch_nodes",
							   pgturbohybrid_last_graph_batch_nodes);
	PgturbohybridJsonbAddFloat8(&state, "graph_avg_batch_size",
								pgturbohybrid_last_graph_batch_calls > 0 ?
								(double) pgturbohybrid_last_graph_batch_nodes /
								(double) pgturbohybrid_last_graph_batch_calls : 0.0);
	/*
	 * Base-layer (PgturbohybridGraphSearchBaseLayer) traversal volume: frontier
	 * heap push/pop, nearest offers, visited checks and how many were duplicate
	 * (already-visited) skips, the per-pop batch scoring calls/nodes, and the
	 * peak frontier size.  Diagnoses heap / visited-check / batch-formation
	 * overhead in the hot native traversal.
	 */
	PgturbohybridJsonbAddKey(&state, "graph_base_layer");
	PgturbohybridJsonbBeginObject(&state);
	PgturbohybridJsonbAddInt64(&state, "frontier_pushes",
							   pgturbohybrid_last_graph_base_frontier_pushes);
	PgturbohybridJsonbAddInt64(&state, "frontier_pops",
							   pgturbohybrid_last_graph_base_frontier_pops);
	PgturbohybridJsonbAddInt64(&state, "nearest_offers",
							   pgturbohybrid_last_graph_base_nearest_offers);
	PgturbohybridJsonbAddInt64(&state, "visited_checks",
							   pgturbohybrid_last_graph_base_visited_checks);
	PgturbohybridJsonbAddInt64(&state, "duplicate_skips",
							   pgturbohybrid_last_graph_base_duplicate_skips);
	PgturbohybridJsonbAddInt64(&state, "batch_calls",
							   pgturbohybrid_last_graph_base_batch_calls);
	PgturbohybridJsonbAddInt64(&state, "batch_nodes",
							   pgturbohybrid_last_graph_base_batch_nodes);
	PgturbohybridJsonbAddInt64(&state, "max_frontier_size",
							   pgturbohybrid_last_graph_base_max_frontier);
	PgturbohybridJsonbCloseObject(&state);
	PgturbohybridJsonbAddString(&state, "score_mode",
								PgturbohybridGraphTqScoreModeName(pgturbohybrid_last_graph_score_mode));
	PgturbohybridJsonbAddString(&state, "dense_simd_force",
								PgturbohybridGraphTqSimdForceName(pgturbohybrid_last_graph_simd_force));
	PgturbohybridJsonbAddBool(&state, "u8_split_enabled",
							  pgturbohybrid_last_graph_u8_split_used);
	/* Whether the x4 u8 batch kernel is enabled, and the path the u8 batch-of-4
	 * scorer actually took (x4 / single / none). */
	PgturbohybridJsonbAddBool(&state, "dense_u8_batch_x4_enabled",
							  pgturbohybrid_last_dense_u8_batch_x4_enabled);
	PgturbohybridJsonbAddString(&state, "graph_u8_batch_mode",
								PgturbohybridGraphU8BatchModeName(pgturbohybrid_last_graph_u8_batch_mode));
	/* Exact u8 kernels resolved once at query prep (no per-batch feature probe). */
	PgturbohybridJsonbAddString(&state, "u8_kernel_single",
								PgturbohybridGraphU8KernelName(pgturbohybrid_last_graph_u8_kernel_single));
	PgturbohybridJsonbAddString(&state, "u8_kernel_batch",
								PgturbohybridGraphU8KernelName(pgturbohybrid_last_graph_u8_kernel_batch));
	/* Whole-code prefetch fires only once the code working set exceeds CPU cache
	 * (graph_large_code_arena); below that the batch scorer prefetches just the
	 * first cache line. */
	PgturbohybridJsonbAddBool(&state, "graph_large_code_arena",
							  pgturbohybrid_last_graph_large_code_arena);
	PgturbohybridJsonbAddBool(&state, "graph_whole_code_prefetch_active",
							  pgturbohybrid_last_graph_whole_code_prefetch_active);
	PgturbohybridJsonbAddInt64(&state, "graph_code_bytes",
							   pgturbohybrid_last_graph_code_bytes);
	PgturbohybridJsonbAddInt64(&state, "graph_code_arena_estimated_bytes",
							   pgturbohybrid_last_graph_code_arena_estimated_bytes);
	/*
	 * Per-backend native scan-cache provenance.  native_cache_mode says whether
	 * the scan was served from the process-local per-backend cache, fell back to
	 * uncached per-scan page loading, or was a non-graph scan; built_this_scan +
	 * build_us isolate the one-time cold per-backend build (the prewarm A/B
	 * signal); native_cache_bytes (with the code/adj/exact breakdown) is the
	 * resident footprint each backend holds and therefore the memory duplicated
	 * across N concurrent clients.
	 */
	PgturbohybridJsonbAddString(&state, "native_cache_mode",
								PgturbohybridGraphNativeCacheModeName(pgturbohybrid_last_graph_native_cache_mode));
	PgturbohybridJsonbAddString(&state, "native_cache_policy",
								PgturbohybridNativeCachePolicyName(pgturbohybrid_last_graph_native_cache_policy));
	PgturbohybridJsonbAddString(&state, "native_cache_scope",
								PgturbohybridGraphNativeCacheScopeName(pgturbohybrid_last_graph_native_cache_mode));
	PgturbohybridJsonbAddBool(&state, "native_cache_used",
							  pgturbohybrid_last_graph_native_cache_used);
	PgturbohybridJsonbAddString(&state, "native_cache_reason",
								PgturbohybridGraphNativeCacheReasonName(pgturbohybrid_last_graph_native_cache_reason));
	PgturbohybridJsonbAddBool(&state, "native_cache_reused",
							  pgturbohybrid_last_graph_native_cache_reused);
	PgturbohybridJsonbAddBool(&state, "native_cache_built_this_scan",
							  pgturbohybrid_last_graph_native_cache_built_this_scan);
	PgturbohybridJsonbAddInt64(&state, "native_cache_attach_us",
							   pgturbohybrid_last_graph_native_cache_attach_us);
	PgturbohybridJsonbAddInt64(&state, "native_cache_build_us",
							   pgturbohybrid_last_graph_native_cache_build_us);
	PgturbohybridJsonbAddInt64(&state, "native_cache_wait_us",
							   pgturbohybrid_last_graph_native_cache_wait_us);
	PgturbohybridJsonbAddInt64(&state, "native_cache_refcount",
							   pgturbohybrid_last_graph_native_cache_refcount);
	PgturbohybridJsonbAddInt64(&state, "native_cache_bytes",
							   pgturbohybrid_last_graph_native_cache_bytes);
	PgturbohybridJsonbAddInt64(&state, "native_cache_code_bytes",
							   pgturbohybrid_last_graph_native_cache_code_bytes);
	PgturbohybridJsonbAddInt64(&state, "native_cache_adj_bytes",
							   pgturbohybrid_last_graph_native_cache_adj_bytes);
	PgturbohybridJsonbAddInt64(&state, "native_cache_exact_bytes",
							   pgturbohybrid_last_graph_native_cache_exact_bytes);
	PgturbohybridJsonbAddBool(&state, "native_cache_warning",
							  pgturbohybrid_last_graph_native_cache_warning);
	PgturbohybridJsonbAddString(&state, "native_cache_warning_reason",
								pgturbohybrid_last_graph_native_cache_warning_reason != NULL ?
								pgturbohybrid_last_graph_native_cache_warning_reason : "none");
	PgturbohybridJsonbAddInt64(&state, "graph_scan_lock_wait_us",
							   pgturbohybrid_last_graph_scan_lock_wait_us);
	PgturbohybridJsonbAddInt64(&state, "code_buffer_lock_wait_us",
							   pgturbohybrid_last_graph_code_buffer_lock_wait_us);
	PgturbohybridJsonbAddInt64(&state, "adj_buffer_lock_wait_us",
							   pgturbohybrid_last_graph_adj_buffer_lock_wait_us);
	PgturbohybridJsonbAddInt64(&state, "graph_dense_requested_k",
							   pgturbohybrid_last_graph_dense_requested_k);
	PgturbohybridJsonbAddInt64(&state, "graph_effective_result_target",
							   pgturbohybrid_last_graph_effective_result_target);
	PgturbohybridJsonbAddInt64(&state, "graph_ef_search",
							   pgturbohybrid_last_graph_ef_search);
	PgturbohybridJsonbAddInt64(&state, "graph_effective_search_ef",
							   pgturbohybrid_last_graph_effective_search_ef);
	PgturbohybridJsonbAddInt64(&state, "graph_effective_rescore_band",
							   pgturbohybrid_last_graph_effective_rescore_band);
	PgturbohybridJsonbAddInt64(&state, "final_k_requested",
							   pgturbohybrid_last_final_k_requested);
	PgturbohybridJsonbAddInt64(&state, "final_k_effective",
							   pgturbohybrid_last_final_k_effective);
	PgturbohybridJsonbAddInt64(&state, "detected_sql_limit",
							   pgturbohybrid_last_sql_limit);
	PgturbohybridJsonbAddBool(&state, "final_k_inferred",
							  pgturbohybrid_last_final_k_inferred);
	PgturbohybridJsonbAddString(&state, "final_k_source",
								PgturbohybridFinalKSourceName());
	PgturbohybridJsonbAddString(&state, "bm25_strategy",
								PgturbohybridBm25StrategyName(pgturbohybrid_bm25_strategy));
	PgturbohybridJsonbAddString(&state, "bm25_impact_or_mode",
								PgturbohybridBm25ImpactOrModeName(pgturbohybrid_bm25_impact_or_mode));
	PgturbohybridJsonbAddInt64(&state, "bm25_hot_postings_cache_mb",
							   pgturbohybrid_bm25_hot_postings_cache_mb);
	PgturbohybridJsonbAddString(&state, "bm25_hybrid_bound",
								PgturbohybridBm25HybridBoundModeName(pgturbohybrid_bm25_hybrid_bound));
	PgturbohybridJsonbAddString(&state, "bm25_accumulator_mode",
								PgturbohybridBm25AccumulatorModeName(pgturbohybrid_bm25_accumulator_mode));
	PgturbohybridJsonbAddBool(&state, "exact_rescore_for_bm25_only",
							  pgturbohybrid_enable_exact_rescore_for_bm25_only);
	PgturbohybridJsonbAddString(&state, "bm25_heap_tsvector_rerank",
								PgturbohybridBm25HeapTSVectorRerankModeName(
									pgturbohybrid_bm25_heap_tsvector_rerank));
	PgturbohybridJsonbAddInt64(&state, "bm25_heap_tsvector_rerank_multiplier",
							   pgturbohybrid_bm25_heap_tsvector_rerank_multiplier);
	PgturbohybridJsonbAddFloat8(&state, "bm25_heap_tsvector_rerank_weight",
								pgturbohybrid_bm25_heap_tsvector_rerank_weight);
	PgturbohybridJsonbAddString(&state, "final_diversity",
								PgturbohybridFinalDiversityName(
									pgturbohybrid_final_diversity));
	PgturbohybridJsonbAddFloat8(&state, "final_diversity_lambda",
								pgturbohybrid_final_diversity_lambda);
	PgturbohybridJsonbAddInt64(&state, "final_diversity_pool_multiplier",
							   pgturbohybrid_final_diversity_pool_multiplier);
	PgturbohybridJsonbAddBool(&state, "auto_budget", pgturbohybrid_auto_budget);
	PgturbohybridJsonbAddBool(&state, "fast_weighted_enabled",
							  scanStats.fastWeightedEnabled);
	PgturbohybridJsonbAddFloat8(&state, "fast_weighted_alpha",
								scanStats.fastWeightedAlpha);
	PgturbohybridJsonbAddBool(&state, "calibrated_fusion_enabled",
							  scanStats.calibratedFusionEnabled);
	PgturbohybridJsonbAddString(&state, "calibrated_fusion_query_shape",
								scanStats.calibratedFusionQueryShape[0] != '\0' ?
								scanStats.calibratedFusionQueryShape : "none");
	PgturbohybridJsonbAddFloat8(&state, "calibrated_fusion_alpha_effective",
								scanStats.calibratedFusionAlphaEffective);
	PgturbohybridJsonbAddFloat8(&state, "calibrated_fusion_both_match_bonus",
								scanStats.calibratedFusionBothMatchBonus);
	PgturbohybridJsonbAddString(&state, "calibrated_fusion_dense_norm_mode",
								scanStats.calibratedFusionDenseNormMode[0] != '\0' ?
								scanStats.calibratedFusionDenseNormMode : "none");
	PgturbohybridJsonbAddString(&state, "calibrated_fusion_bm25_norm_mode",
								scanStats.calibratedFusionBm25NormMode[0] != '\0' ?
								scanStats.calibratedFusionBm25NormMode : "none");
	PgturbohybridJsonbAddBool(&state, "dbsf_enabled",
							  scanStats.dbsfEnabled);
	PgturbohybridJsonbAddFloat8Array(&state, "dbsf_branch_mean",
									 scanStats.dbsfBranchMean, 2);
	PgturbohybridJsonbAddFloat8Array(&state, "dbsf_branch_stddev",
									 scanStats.dbsfBranchStddev, 2);
	PgturbohybridJsonbAddFloat8Array(&state, "dbsf_branch_min",
									 scanStats.dbsfBranchMin, 2);
	PgturbohybridJsonbAddFloat8Array(&state, "dbsf_branch_max",
									 scanStats.dbsfBranchMax, 2);
	PgturbohybridJsonbAddInt64(&state, "dbsf_degenerate_branches",
							   scanStats.dbsfDegenerateBranches);
	PgturbohybridJsonbAddString(&state, "bm25_norm_mode",
								scanStats.bm25NormMode[0] != '\0' ?
								scanStats.bm25NormMode : "none");
	PgturbohybridJsonbAddString(&state, "dense_norm_mode",
								scanStats.denseNormMode[0] != '\0' ?
								scanStats.denseNormMode : "none");
	PgturbohybridJsonbAddString(&state, "hybrid_budget_policy",
								scanStats.hybridBudgetPolicy[0] != '\0' ?
								scanStats.hybridBudgetPolicy : "fixed");
	PgturbohybridJsonbAddString(&state, "hybrid_query_shape",
								scanStats.hybridQueryShape[0] != '\0' ?
								scanStats.hybridQueryShape : "fixed");
	PgturbohybridJsonbAddInt64(&state, "hybrid_dense_k_chosen",
							   scanStats.hybridDenseKChosen);
	PgturbohybridJsonbAddInt64(&state, "hybrid_bm25_k_chosen",
							   scanStats.hybridBm25KChosen);
	PgturbohybridJsonbAddString(&state, "hybrid_budget_reason",
								scanStats.hybridBudgetReason[0] != '\0' ?
								scanStats.hybridBudgetReason : "fixed_policy");
	PgturbohybridJsonbAddString(&state, "fusion_strategy",
								scanStats.fusionStrategy[0] != '\0' ?
								scanStats.fusionStrategy : "none");
	PgturbohybridJsonbAddInt64(&state, "fusion_candidates_seen",
							   scanStats.fusionCandidatesSeen);
	PgturbohybridJsonbAddUint64(&state, "fusion_duplicates",
								scanStats.fusionDuplicates);
	PgturbohybridJsonbAddUint64(&state, "fusion_heap_replacements",
								scanStats.fusionHeapReplacements);
	PgturbohybridJsonbAddBool(&state, "fusion_generation_array_reused",
							  scanStats.fusionGenerationArrayReused);
	PgturbohybridJsonbAddBool(&state, "fusion_generation_array_reset",
							  scanStats.fusionGenerationArrayReset);
	PgturbohybridJsonbAddBool(&state, "multivector_enabled",
							  scanStats.multivectorEnabled);
	PgturbohybridJsonbAddInt64(&state, "multivector_query_vectors",
							   scanStats.multivectorQueryVectors);
	PgturbohybridJsonbAddInt64(&state, "multivector_doc_vectors_limit",
							   scanStats.multivectorDocVectorsLimit);
	PgturbohybridJsonbAddUint64(&state, "multivector_subvector_searches",
								scanStats.multivectorSubvectorSearches);
	PgturbohybridJsonbAddUint64(&state, "multivector_raw_subvector_hits",
								scanStats.multivectorRawSubvectorHits);
	PgturbohybridJsonbAddBool(&state, "multivector_adaptive_widening_triggered",
							  scanStats.multivectorAdaptiveWideningTriggered);
	PgturbohybridJsonbAddInt64(&state, "multivector_adaptive_initial_raw_target",
							   scanStats.multivectorAdaptiveInitialRawTarget);
	PgturbohybridJsonbAddInt64(&state, "multivector_adaptive_final_raw_target",
							   scanStats.multivectorAdaptiveFinalRawTarget);
		if (scanStats.multivectorCandidateSource[0] != '\0')
			PgturbohybridJsonbAddString(&state, "multivector_candidate_source",
										scanStats.multivectorCandidateSource);
		else
			PgturbohybridJsonbAddString(&state, "multivector_candidate_source",
										"graph");
		if (scanStats.multivectorCandidatePath[0] != '\0')
			PgturbohybridJsonbAddString(&state, "multivector_candidate_path",
										scanStats.multivectorCandidatePath);
		else
			PgturbohybridJsonbAddString(&state, "multivector_candidate_path",
										"graph");
		PgturbohybridJsonbAddUint64(&state, "multivector_proxy_graph_searches",
									scanStats.multivectorProxyGraphSearches);
		PgturbohybridJsonbAddInt64(&state, "multivector_proxy_candidate_target",
								   scanStats.multivectorProxyCandidateTarget);
		PgturbohybridJsonbAddInt64(&state, "multivector_proxy_candidates_returned",
								   scanStats.multivectorProxyCandidatesReturned);
			PgturbohybridJsonbAddInt64(&state, "multivector_exact_rerank_k_effective",
									   scanStats.multivectorExactRerankKEffective);
			PgturbohybridJsonbAddString(&state, "proxy_encoder_kind",
										scanStats.multivectorProxyEncoderKind[0] != '\0' ?
										scanStats.multivectorProxyEncoderKind : "none");
		PgturbohybridJsonbAddInt64(&state, "proxy_candidate_limit_effective",
								   scanStats.proxyCandidateLimitEffective);
		PgturbohybridJsonbAddString(&state, "proxy_candidate_limit_source",
									scanStats.proxyCandidateLimitSource[0] != '\0' ?
									scanStats.proxyCandidateLimitSource : "none");
		PgturbohybridJsonbAddInt64(&state, "proxy_candidates",
								   scanStats.proxyCandidates);
	PgturbohybridJsonbAddUint64(&state, "proxy_graph_nodes_visited",
								scanStats.proxyGraphNodesVisited);
	PgturbohybridJsonbAddUint64(&state, "proxy_graph_edges_visited",
								scanStats.proxyGraphEdgesVisited);
	PgturbohybridJsonbAddInt64(&state, "proxy_graph_candidates_seen",
							   scanStats.proxyGraphCandidatesSeen);
	PgturbohybridJsonbAddInt64(&state, "proxy_candidates_returned",
							   scanStats.proxyCandidatesReturned);
	PgturbohybridJsonbAddUint64(&state, "proxy_vector_scores_computed",
								scanStats.proxyVectorScoresComputed);
	PgturbohybridJsonbAddUint64(&state, "proxy_vector_score_time_us",
								scanStats.proxyVectorScoreUs);
	PgturbohybridJsonbAddBool(&state, "proxy_lazy_sidecar_vectors",
							  scanStats.proxyLazySidecarVectors);
	PgturbohybridJsonbAddString(&state, "multivector_doc_storage_cache_requested",
								scanStats.multivectorDocStorageCacheRequested[0] != '\0' ?
								scanStats.multivectorDocStorageCacheRequested : "auto");
	PgturbohybridJsonbAddString(&state, "multivector_doc_storage_cache_effective",
								scanStats.multivectorDocStorageCacheEffective[0] != '\0' ?
								scanStats.multivectorDocStorageCacheEffective : "auto");
	PgturbohybridJsonbAddBool(&state, "proxy_top1_admission",
							  scanStats.proxyTop1Admission);
	PgturbohybridJsonbAddInt64(&state, "proxy_exact_rerank_docs",
							   scanStats.proxyExactRerankDocs);
	PgturbohybridJsonbAddUint64(&state, "proxy_full_sidecar_vectors_loaded",
								scanStats.proxyFullSidecarVectorsLoaded);
	PgturbohybridJsonbAddUint64(&state, "proxy_full_sidecar_bytes_touched",
								scanStats.proxyFullSidecarBytesTouched);
	PgturbohybridJsonbAddUint64(&state, "proxy_full_sidecar_pages_read",
								scanStats.proxyFullSidecarPagesRead);
	PgturbohybridJsonbAddUint64(&state, "proxy_full_sidecar_load_time_us",
								scanStats.proxyFullSidecarLoadUs);
	PgturbohybridJsonbAddUint64(&state, "proxy_full_sidecar_reconstruct_time_us",
								scanStats.proxyFullSidecarReconstructUs);
	PgturbohybridJsonbAddUint64(&state, "proxy_exact_rerank_heap_fetches",
								scanStats.proxyExactRerankHeapFetches);
	PgturbohybridJsonbAddUint64(&state, "proxy_exact_rerank_sidecar_fetches",
								scanStats.proxyExactRerankSidecarFetches);
	PgturbohybridJsonbAddUint64(&state, "proxy_exact_rerank_bytes_touched",
								scanStats.proxyExactRerankBytesTouched);
	PgturbohybridJsonbAddUint64(&state, "proxy_exact_rerank_time_us",
								scanStats.proxyExactRerankUs);
	PgturbohybridJsonbAddBool(&state, "sidecar_cache_build_this_query",
							  scanStats.sidecarCacheBuildThisQuery);
		PgturbohybridJsonbAddUint64(&state, "sidecar_cache_build_bytes",
									scanStats.sidecarCacheBuildBytes);
		PgturbohybridJsonbAddUint64(&state, "sidecar_cache_build_pages_read",
									scanStats.sidecarCacheBuildPagesRead);
		PgturbohybridJsonbAddUint64(&state, "sidecar_cache_build_time_us",
									scanStats.sidecarCacheBuildUs);
	PgturbohybridJsonbAddUint64(&state, "sidecar_query_bytes_touched",
								scanStats.sidecarQueryBytesTouched);
	PgturbohybridJsonbAddUint64(&state, "sidecar_query_pages_read",
								scanStats.sidecarQueryPagesRead);
	PgturbohybridJsonbAddUint64(&state, "sidecar_query_vectors_loaded",
								scanStats.sidecarQueryVectorsLoaded);
		PgturbohybridJsonbAddUint64(&state, "sidecar_query_load_time_us",
									scanStats.sidecarQueryLoadUs);
		PgturbohybridJsonbAddUint64(&state, "sidecar_query_time_us",
									scanStats.sidecarQueryUs);
	PgturbohybridJsonbAddBool(&state, "proxy_vector_uses_full_sidecar_for_graph",
							  scanStats.proxyVectorUsesFullSidecarForGraph);
	PgturbohybridJsonbAddBool(&state, "proxy_vector_near_exhaustive_sidecar_touch",
							  scanStats.proxyVectorNearExhaustiveSidecarTouch);
	PgturbohybridJsonbAddString(&state, "proxy_vector_sidecar_touch_reason",
								scanStats.proxyVectorSidecarTouchReason[0] != '\0' ?
								scanStats.proxyVectorSidecarTouchReason : "none");
	PgturbohybridJsonbAddUint64(&state, "centroid_lists_visited",
								scanStats.centroidListsVisited);
	PgturbohybridJsonbAddUint64(&state, "centroid_docs_touched",
								scanStats.centroidDocsTouched);
	PgturbohybridJsonbAddUint64(&state, "centroid_pruned_docs",
								scanStats.centroidPrunedDocs);
	PgturbohybridJsonbAddUint64(&state, "centroid_postings_touched",
								scanStats.centroidPostingsTouched);
	PgturbohybridJsonbAddUint64(&state, "centroid_postings_skipped",
								scanStats.centroidPostingsSkipped);
	PgturbohybridJsonbAddInt64(&state, "centroid_posting_limit_per_token",
							   scanStats.centroidPostingLimitPerToken);
	PgturbohybridJsonbAddString(&state, "centroid_posting_cap_strategy",
								scanStats.centroidPostingCapStrategy[0] != '\0' ?
								scanStats.centroidPostingCapStrategy : "none");
	PgturbohybridJsonbAddInt64(&state, "centroid_candidates",
							   scanStats.centroidCandidates);
	PgturbohybridJsonbAddInt64(&state, "multivector_centroid_count",
							   scanStats.multivectorCentroidCount);
	PgturbohybridJsonbAddInt64(&state, "multivector_centroid_prerank_docs",
							   scanStats.multivectorCentroidPrerankDocs);
	PgturbohybridJsonbAddInt64(&state, "multivector_full_maxsim_rerank_docs",
							   scanStats.multivectorFullMaxsimRerankDocs);
	PgturbohybridJsonbAddUint64(&state, "quantized_inverted_lists_visited",
								scanStats.quantizedInvertedListsVisited);
	PgturbohybridJsonbAddUint64(&state, "quantized_inverted_postings_touched",
								scanStats.quantizedInvertedPostingsTouched);
	PgturbohybridJsonbAddUint64(&state, "quantized_inverted_docs_scored",
								scanStats.quantizedInvertedDocsScored);
	PgturbohybridJsonbAddInt64(&state, "quantized_inverted_candidates",
							   scanStats.quantizedInvertedCandidates);
	PgturbohybridJsonbAddInt64(&state, "quantized_inverted_exact_rerank_docs",
							   scanStats.quantizedInvertedExactRerankDocs);
	PgturbohybridJsonbAddInt64(&state, "quantized_inverted_codebook_size",
							   scanStats.quantizedInvertedCodebookSize);
	PgturbohybridJsonbAddUint64(&state, "quantized_inverted_list_offset_bytes",
								scanStats.quantizedInvertedListOffsetBytes);
	PgturbohybridJsonbAddUint64(&state, "quantized_inverted_posting_bytes",
								scanStats.quantizedInvertedPostingBytes);
	PgturbohybridJsonbAddUint64(&state, "quantized_inverted_sidecar_bytes",
								scanStats.quantizedInvertedSidecarBytes);
	if (scanStats.multivectorGraphMode[0] != '\0')
		PgturbohybridJsonbAddString(&state, "multivector_graph_mode",
									scanStats.multivectorGraphMode);
	else
		PgturbohybridJsonbAddString(&state, "multivector_graph_mode",
									PgturbohybridMultiVectorGraphModeName(
										PGTURBOHYBRID_DEFAULT_MULTIVECTOR_GRAPH_MODE));
	PgturbohybridJsonbAddBool(&state, "multivector_exact_token_scan_enabled",
							  scanStats.multivectorExactTokenScanEnabled);
	PgturbohybridJsonbAddUint64(&state, "multivector_exact_token_scan_nodes_scored",
								scanStats.multivectorExactTokenScanNodesScored);
	PgturbohybridJsonbAddBool(&state, "multivector_plain_fallback_used",
							  scanStats.multivectorPlainFallbackUsed);
	if (scanStats.multivectorPlainFallbackReason[0] != '\0')
		PgturbohybridJsonbAddString(&state, "multivector_plain_fallback_reason",
									scanStats.multivectorPlainFallbackReason);
	else
		PgturbohybridJsonbAddString(&state, "multivector_plain_fallback_reason",
									"not_applicable");
	PgturbohybridJsonbAddUint64(&state, "multivector_plain_fallback_docs_scored",
								scanStats.multivectorPlainFallbackDocsScored);
	PgturbohybridJsonbAddUint64(&state, "multivector_plain_fallback_pairs",
								scanStats.multivectorPlainFallbackPairs);
	PgturbohybridJsonbAddBool(&state, "multivector_doc_graph_prototype_enabled",
							  scanStats.multivectorDocGraphPrototypeEnabled);
	PgturbohybridJsonbAddUint64(&state, "multivector_doc_graph_nodes",
								scanStats.multivectorDocGraphNodes);
	PgturbohybridJsonbAddUint64(&state, "multivector_doc_graph_docs_scored",
								scanStats.multivectorDocGraphDocsScored);
	PgturbohybridJsonbAddUint64(&state, "multivector_doc_graph_edges_visited",
								scanStats.multivectorDocGraphEdgesVisited);
	PgturbohybridJsonbAddInt64(&state, "multivector_doc_graph_candidates",
							   scanStats.multivectorDocGraphCandidates);
	PgturbohybridJsonbAddInt64(&state, "multivector_doc_graph_search_ef",
							   scanStats.multivectorDocGraphSearchEf);
	PgturbohybridJsonbAddInt64(&state, "multivector_doc_graph_oversampling",
							   scanStats.multivectorDocGraphOversampling);
	PgturbohybridJsonbAddInt64(&state, "multivector_doc_graph_rescore_k",
							   scanStats.multivectorDocGraphRescoreK);
	PgturbohybridJsonbAddInt64(&state, "multivector_doc_graph_entry_sample_configured",
							   scanStats.multivectorDocGraphEntrySampleConfigured);
	PgturbohybridJsonbAddInt64(&state, "multivector_doc_graph_entry_sample_effective",
							   scanStats.multivectorDocGraphEntrySampleEffective);
	PgturbohybridJsonbAddInt64(&state, "multivector_doc_graph_entry_sample_scored",
							   scanStats.multivectorDocGraphEntrySampleScored);
	PgturbohybridJsonbAddUint64(&state, "multivector_doc_graph_quantized_scores",
								scanStats.multivectorDocGraphQuantizedScores);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_doc_graph_insert_full_maxsim_edges",
								pgturbohybrid_last_doc_insert_stats.fullMaxsimEdges);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_doc_graph_insert_representative_fallbacks",
								pgturbohybrid_last_doc_insert_stats.representativeFallbacks);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_doc_graph_insert_pairs_scored",
								pgturbohybrid_last_doc_insert_stats.pairsScored);
	if (scanStats.multivectorDocGraphStorageKind[0] != '\0')
		PgturbohybridJsonbAddString(&state,
									"multivector_doc_graph_storage_kind",
									scanStats.multivectorDocGraphStorageKind);
	else
		PgturbohybridJsonbAddString(&state,
									"multivector_doc_graph_storage_kind",
									"f32");
	if (scanStats.multivectorDocGraphRescoreSource[0] != '\0')
		PgturbohybridJsonbAddString(&state,
									"multivector_doc_graph_rescore_source",
									scanStats.multivectorDocGraphRescoreSource);
	else
		PgturbohybridJsonbAddString(&state,
									"multivector_doc_graph_rescore_source",
									"none");
	PgturbohybridJsonbAddInt64(&state, "multivector_doc_graph_exact_rerank_docs",
							   scanStats.multivectorDocGraphExactRerankDocs);
	PgturbohybridJsonbAddUint64(&state, "multivector_doc_graph_heap_fetches",
								scanStats.multivectorDocGraphHeapFetches);
	if (scanStats.multivectorDocGraphWarning[0] != '\0')
		PgturbohybridJsonbAddString(&state, "multivector_doc_graph_warning",
									scanStats.multivectorDocGraphWarning);
	else
		PgturbohybridJsonbAddString(&state, "multivector_doc_graph_warning",
									"not_applicable");
	PgturbohybridJsonbAddString(&state, "multivector_doc_sidecar_cache_mode",
								scanStats.multivectorDocSidecarCacheMode[0] != '\0' ?
								scanStats.multivectorDocSidecarCacheMode : "none");
	PgturbohybridJsonbAddUint64(&state, "multivector_doc_sidecar_pages_read",
								scanStats.multivectorDocSidecarPagesRead);
	PgturbohybridJsonbAddUint64(&state, "multivector_doc_sidecar_cache_hits",
								scanStats.multivectorDocSidecarCacheHits);
	PgturbohybridJsonbAddUint64(&state, "multivector_doc_sidecar_cache_misses",
								scanStats.multivectorDocSidecarCacheMisses);
	PgturbohybridJsonbAddUint64(&state, "multivector_doc_sidecar_bytes_touched",
								scanStats.multivectorDocSidecarBytesTouched);
	PgturbohybridJsonbAddUint64(&state, "multivector_doc_sidecar_vectors_loaded",
								scanStats.multivectorDocSidecarVectorsLoaded);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_doc_sidecar_docmap_pages_read",
								scanStats.multivectorDocSidecarDocMapPagesRead);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_doc_sidecar_docmap_bytes_touched",
								scanStats.multivectorDocSidecarDocMapBytesTouched);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_doc_sidecar_resident_vectors_loaded",
								scanStats.multivectorDocSidecarResidentVectorsLoaded);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_doc_sidecar_resident_bytes_loaded",
								scanStats.multivectorDocSidecarResidentBytesLoaded);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_doc_sidecar_vector_chunk_ref_bytes_touched",
								scanStats.multivectorDocSidecarVectorChunkRefBytesTouched);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_doc_sidecar_paged_vector_pages_read",
								scanStats.multivectorDocSidecarPagedVectorPagesRead);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_doc_sidecar_paged_vector_bytes_touched",
								scanStats.multivectorDocSidecarPagedVectorBytesTouched);
	PgturbohybridJsonbAddUint64(&state, "multivector_tokens_original",
								scanStats.multivectorTokensOriginal);
	PgturbohybridJsonbAddUint64(&state, "multivector_tokens_pooled",
								scanStats.multivectorTokensPooled);
	PgturbohybridJsonbAddFloat8(&state, "multivector_token_pooling_ratio",
								scanStats.multivectorTokensOriginal > 0 ?
								(double) scanStats.multivectorTokensPooled /
								(double) scanStats.multivectorTokensOriginal : 0.0);
	PgturbohybridJsonbAddBool(&state, "multivector_reservoirs_enabled",
							  scanStats.multivectorReservoirsEnabled);
	PgturbohybridJsonbAddInt64(&state, "multivector_reservoir_score_docs",
							   scanStats.multivectorReservoirScoreDocs);
	PgturbohybridJsonbAddInt64(&state, "multivector_reservoir_coverage_docs",
							   scanStats.multivectorReservoirCoverageDocs);
	PgturbohybridJsonbAddInt64(&state, "multivector_reservoir_mean_docs",
							   scanStats.multivectorReservoirMeanDocs);
	PgturbohybridJsonbAddInt64(&state, "multivector_reservoir_per_token_docs",
							   scanStats.multivectorReservoirPerTokenDocs);
	PgturbohybridJsonbAddInt64(&state, "multivector_reservoir_bm25_docs",
							   scanStats.multivectorReservoirBm25Docs);
	PgturbohybridJsonbAddInt64(&state, "multivector_reservoir_union_docs",
							   scanStats.multivectorReservoirUnionDocs);
	PgturbohybridJsonbAddInt64(&state, "multivector_reservoir_duplicates",
							   scanStats.multivectorReservoirDuplicates);
	PgturbohybridJsonbAddBool(&state, "multivector_bm25_injection_enabled",
							  scanStats.multivectorBm25InjectionEnabled);
	PgturbohybridJsonbAddInt64(&state, "multivector_bm25_injection_candidates",
							   scanStats.multivectorBm25InjectionCandidates);
	PgturbohybridJsonbAddInt64(&state, "multivector_bm25_injection_candidate_limit",
							   scanStats.multivectorBm25InjectionCandidateLimit);
	PgturbohybridJsonbAddInt64(&state, "multivector_bm25_injection_pool_size",
							   scanStats.multivectorBm25InjectionPoolSize);
	if (scanStats.multivectorBm25InjectionLimitReason[0] != '\0')
		PgturbohybridJsonbAddString(&state,
									"multivector_bm25_injection_limit_reason",
									scanStats.multivectorBm25InjectionLimitReason);
	else
		PgturbohybridJsonbAddString(&state,
									"multivector_bm25_injection_limit_reason",
									"not_applicable");
	PgturbohybridJsonbAddInt64(&state, "multivector_bm25_injection_retained",
							   scanStats.multivectorBm25InjectionRetained);
	PgturbohybridJsonbAddInt64(&state, "multivector_bm25_injection_exact_reranked",
							   scanStats.multivectorBm25InjectionExactReranked);
	PgturbohybridJsonbAddInt64(&state, "learned_sparse_candidates",
							   scanStats.learnedSparseCandidates);
	PgturbohybridJsonbAddInt64(&state, "learned_sparse_retained_for_maxsim",
							   scanStats.learnedSparseRetainedForMaxsim);
	PgturbohybridJsonbAddUint64(&state, "learned_sparse_branch_latency_us",
								scanStats.learnedSparseBranchLatencyUs);
	if (scanStats.multivectorDocMapSource[0] != '\0')
		PgturbohybridJsonbAddString(&state, "multivector_docmap_source",
									scanStats.multivectorDocMapSource);
	else
		PgturbohybridJsonbAddString(&state, "multivector_docmap_source",
									"none");
	PgturbohybridJsonbAddUint64(&state, "multivector_docmap_bytes",
								scanStats.multivectorDocMapBytes);
	PgturbohybridJsonbAddUint64(&state, "multivector_unique_docs",
								scanStats.multivectorUniqueDocs);
	PgturbohybridJsonbAddUint64(&state, "multivector_duplicate_doc_hits",
								scanStats.multivectorDuplicateDocHits);
	PgturbohybridJsonbAddUint64(&state, "multivector_maxsim_updates",
								scanStats.multivectorMaxsimUpdates);
	PgturbohybridJsonbAddInt64(&state, "multivector_doc_candidates",
							   scanStats.multivectorDocCandidates);
	PgturbohybridJsonbAddBool(&state, "multivector_exact_rerank_enabled",
							  scanStats.multivectorExactRerankEnabled);
	PgturbohybridJsonbAddInt64(&state, "multivector_exact_rerank_docs",
							   scanStats.multivectorExactRerankDocs);
	PgturbohybridJsonbAddUint64(&state, "multivector_exact_rerank_pairs",
								scanStats.multivectorExactRerankPairs);
	if (scanStats.multivectorExactRerankSource[0] != '\0')
		PgturbohybridJsonbAddString(&state,
									"multivector_exact_rerank_source",
									scanStats.multivectorExactRerankSource);
	else
		PgturbohybridJsonbAddString(&state,
									"multivector_exact_rerank_source",
									"off");
	PgturbohybridJsonbAddUint64(&state,
								"multivector_exact_rerank_heap_fetches",
								scanStats.multivectorExactRerankHeapFetches);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_exact_rerank_sidecar_reads",
								scanStats.multivectorExactRerankSidecarReads);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_exact_rerank_sidecar_bytes",
								scanStats.multivectorExactRerankSidecarBytes);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_candidate_source_us",
								scanStats.multivectorCandidateSourceUs);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_candidate_source_time_us",
								scanStats.multivectorCandidateSourceUs);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_doc_graph_traversal_us",
								scanStats.multivectorDocGraphTraversalUs);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_document_graph_traversal_time_us",
								scanStats.multivectorDocGraphTraversalUs);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_proxy_candidate_time_us",
								scanStats.multivectorProxyCandidateUs);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_proxy_graph_traversal_time_us",
								scanStats.multivectorProxyGraphTraversalUs);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_proxy_scoring_us",
								scanStats.multivectorProxyScoringUs);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_proxy_scoring_time_us",
								scanStats.multivectorProxyScoringUs);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_centroid_lite_posting_us",
								scanStats.multivectorCentroidLitePostingUs);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_centroid_lite_time_us",
								scanStats.multivectorCentroidLitePostingUs);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_quantized_inverted_posting_us",
								scanStats.multivectorQuantizedInvertedPostingUs);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_quantized_inverted_time_us",
								scanStats.multivectorQuantizedInvertedPostingUs);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_sidecar_load_us",
								scanStats.multivectorSidecarLoadUs);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_sidecar_load_time_us",
								scanStats.multivectorSidecarLoadUs);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_sidecar_page_read_time_us",
								scanStats.multivectorSidecarPageReadUs);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_sidecar_vector_reconstruct_time_us",
								scanStats.multivectorSidecarVectorReconstructUs);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_heap_visibility_us",
								scanStats.multivectorHeapVisibilityUs);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_exact_heap_fetch_us",
								scanStats.multivectorExactHeapFetchUs);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_exact_heap_fetch_time_us",
								scanStats.multivectorExactHeapFetchUs);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_exact_rerank_us",
								scanStats.multivectorExactRerankUs);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_exact_maxsim_rerank_time_us",
								scanStats.multivectorExactRerankUs);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_final_sort_us",
								scanStats.multivectorFinalSortUs);
	PgturbohybridJsonbAddUint64(&state,
								"multivector_final_sort_time_us",
								scanStats.multivectorFinalSortUs);
	PgturbohybridJsonbAddInt64(&state, "exact_rerank_candidates",
							   scanStats.exactRerankCandidates);
	PgturbohybridJsonbAddUint64(&state, "exact_rerank_tokens_evaluated",
								scanStats.exactRerankTokensEvaluated);
	PgturbohybridJsonbAddUint64(&state, "exact_rerank_tokens_skipped",
								scanStats.exactRerankTokensSkipped);
	PgturbohybridJsonbAddUint64(&state, "exact_rerank_pairs_saved",
								scanStats.exactRerankPairsSaved);
	PgturbohybridJsonbAddBool(&state, "adaptive_rerank_topk_changed_vs_full",
							  scanStats.adaptiveRerankTopKChangedVsFull);
	if (scanStats.multivectorExactKernel[0] != '\0')
		PgturbohybridJsonbAddString(&state, "multivector_exact_kernel",
									scanStats.multivectorExactKernel);
	else
		PgturbohybridJsonbAddNull(&state, "multivector_exact_kernel");
	if (scanStats.multivectorAccumulatorKind[0] != '\0')
		PgturbohybridJsonbAddString(&state, "multivector_accumulator_kind",
									scanStats.multivectorAccumulatorKind);
	else
		PgturbohybridJsonbAddNull(&state, "multivector_accumulator_kind");
	PgturbohybridJsonbAddUint64(&state, "multivector_memory_bytes_estimate",
								scanStats.multivectorMemoryBytesEstimate);
	PgturbohybridJsonbAddBool(&state, "multivector_admission_debug_enabled",
							  scanStats.multivectorAdmissionDebugEnabled);
	PgturbohybridJsonbAddInt64(&state, "multivector_admission_candidates_before_rerank",
							   scanStats.multivectorAdmissionCandidatesBeforeRerank);
	PgturbohybridJsonbAddInt64(&state, "multivector_admission_candidates_after_truncation",
							   scanStats.multivectorAdmissionCandidatesAfterTruncation);
	PgturbohybridJsonbAddInt64(&state, "multivector_admission_exact_rerank_docs",
							   scanStats.multivectorAdmissionExactRerankDocs);
	PgturbohybridJsonbAddBool(&state, "multivector_admission_truncated_by_doc_candidate_k",
							  scanStats.multivectorAdmissionTruncatedByDocCandidateK);
	PgturbohybridJsonbAddBool(&state, "multivector_admission_truncated_by_accumulator_memory",
							  scanStats.multivectorAdmissionTruncatedByAccumulatorMemory);
	PgturbohybridJsonbAddBool(&state, "multivector_admission_trace_available",
							  scanStats.multivectorAdmissionTraceAvailable);
	PgturbohybridJsonbAddMultiVectorAdmissionTrace(&state, &scanStats);
	PgturbohybridJsonbAddBool(&state, "multivector_query_token_stats_available",
							  scanStats.multivectorTokenStatsAvailable);
	PgturbohybridJsonbAddMultiVectorTokenStats(&state, &scanStats);
	PgturbohybridJsonbAddString(&state, "final_diversity_mode",
								scanStats.finalDiversityMode[0] != '\0' ?
								scanStats.finalDiversityMode : "off");
	PgturbohybridJsonbAddInt64(&state, "final_diversity_payload_slot",
							   scanStats.finalDiversityPayloadSlot);
	PgturbohybridJsonbAddInt64(&state, "final_diversity_pool_size",
							   scanStats.finalDiversityPoolSize);
	PgturbohybridJsonbAddInt64(&state, "final_diversity_selected",
							   scanStats.finalDiversitySelected);
	PgturbohybridJsonbAddUint64(&state, "final_diversity_duplicate_groups_suppressed",
								scanStats.finalDiversityDuplicateGroupsSuppressed);
	PgturbohybridJsonbAddUint64(&state, "final_diversity_us",
								scanStats.finalDiversityUs);
	PgturbohybridJsonbAddInt64(&state, "dense_candidates_effective",
							   scanStats.denseCandidatesEffective);
	PgturbohybridJsonbAddInt64(&state, "dense_k_effective",
							   scanStats.denseCandidatesEffective);
	PgturbohybridJsonbAddBool(&state, "dense_k_defaulted",
							  scanStats.denseKDefaulted);
	PgturbohybridJsonbAddInt64(&state, "bm25_candidates_effective",
							   scanStats.bm25CandidatesEffective);
	PgturbohybridJsonbAddInt64(&state, "bm25_k_effective",
							   scanStats.bm25CandidatesEffective);
	PgturbohybridJsonbAddBool(&state, "bm25_k_defaulted",
							  scanStats.bm25KDefaulted);
	PgturbohybridJsonbAddBool(&state, "bm25_cache_hit",
							  scanStats.bm25CacheHit);
	PgturbohybridJsonbAddUint64(&state, "bm25_cache_build_us",
								scanStats.bm25CacheBuildUs);
	PgturbohybridJsonbAddBool(&state, "bm25_docstats_loaded_this_query",
							  scanStats.bm25DocstatsLoadedThisQuery);
	PgturbohybridJsonbAddBool(&state, "bm25_liveness_loaded_this_query",
							  scanStats.bm25LivenessLoadedThisQuery);
	PgturbohybridJsonbAddUint64(&state, "bm25_docstats_bytes",
								scanStats.bm25DocstatsBytes);
	PgturbohybridJsonbAddUint64(&state, "bm25_liveness_bytes",
								scanStats.bm25LivenessBytes);
	PgturbohybridJsonbAddBool(&state, "bm25_cold_cache_o_n_work",
							  scanStats.bm25ColdCacheONWork);
	PgturbohybridJsonbAddFloat8(&state, "bm25_postings_decode_ratio",
								scanStats.bm25PostingsDecodeRatio);
	PgturbohybridJsonbAddBool(&state, "bm25_common_term_fallback",
							  scanStats.bm25CommonTermFallback);
	PgturbohybridJsonbAddUint64(&state, "bm25_wand_pruned",
								scanStats.bm25WandPruned);
	PgturbohybridJsonbAddBool(&state, "bm25_hot_postings_cache_hit",
							  scanStats.bm25HotPostingsCacheHits > 0);
	PgturbohybridJsonbAddUint64(&state, "bm25_hot_postings_cache_hits",
								scanStats.bm25HotPostingsCacheHits);
	PgturbohybridJsonbAddUint64(&state, "bm25_hot_postings_cache_misses",
								scanStats.bm25HotPostingsCacheMisses);
	PgturbohybridJsonbAddUint64(&state,
								"bm25_blocks_pruned_by_fused_score_bound",
								scanStats.bm25FusedScoreBoundBlocksPruned);
	PgturbohybridJsonbAddUint64(&state,
								"bm25_candidates_pruned_by_fused_score_bound",
								scanStats.bm25FusedScoreBoundCandidatesPruned);
	PgturbohybridJsonbAddString(&state, "bm25_heap_tsvector_rerank_mode",
								scanStats.bm25HeapTSVectorRerankMode[0] != '\0' ?
								scanStats.bm25HeapTSVectorRerankMode : "off");
	PgturbohybridJsonbAddInt64(&state, "bm25_heap_tsvector_rerank_count",
							   scanStats.bm25HeapTSVectorRerankCount);
	PgturbohybridJsonbAddUint64(&state, "bm25_heap_tsvector_rerank_fetch_us",
								scanStats.bm25HeapTSVectorRerankFetchUs);
	PgturbohybridJsonbAddUint64(&state, "bm25_heap_tsvector_rerank_score_us",
								scanStats.bm25HeapTSVectorRerankScoreUs);
	PgturbohybridJsonbAddBool(&state, "bm25_heap_tsvector_rerank_topk_changed",
							  scanStats.bm25HeapTSVectorRerankTopKChanged);
	PgturbohybridJsonbAddUint64(&state, "strict_vector_validations",
								validationStats.strictVectorValidations);
	PgturbohybridJsonbAddUint64(&state, "fast_vector_checks",
								validationStats.fastVectorChecks);
	PgturbohybridJsonbAddUint64(&state, "vector_type_cache_hits",
								validationStats.vectorTypeCacheHits);
	PgturbohybridJsonbAddUint64(&state, "vector_type_cache_misses",
								validationStats.vectorTypeCacheMisses);
	PgturbohybridJsonbAddUint64(&state, "dense_elapsed_us", denseElapsedUs);
	PgturbohybridJsonbAddUint64(&state, "bm25_elapsed_us", bm25ElapsedUs);
	PgturbohybridJsonbAddUint64(&state, "fusion_elapsed_us", fusionElapsedUs);
	PgturbohybridJsonbAddUint64(&state, "elapsed_us", elapsedUs);
	PgturbohybridJsonbAddFloat8(&state, "graph_highdim_widening_multiplier",
								pgturbohybrid_last_graph_highdim_widening_multiplier);
	PgturbohybridJsonbAddString(&state, "graph_widening_reason",
								PgturbohybridGraphDenseWideningReasonName(pgturbohybrid_last_graph_widening_reason));
	PgturbohybridJsonbAddBool(&state, "dense_filter_unmapped",
							  pgturbohybrid_last_dense_filter_unmapped);
	PgturbohybridJsonbAddBool(&state, "dense_linear_fallback_warning",
							  pgturbohybrid_last_dense_linear_fallback_warning);
	PgturbohybridJsonbAddFloat8(&state, "dense_linear_fallback_ratio",
								pgturbohybrid_last_dense_linear_fallback_ratio);
	PgturbohybridJsonbAddString(&state, "dense_adaptive_widening_mode",
								PgturbohybridGraphDenseAdaptiveWideningModeName(pgturbohybrid_last_graph_adaptive_widening_mode));
	PgturbohybridJsonbAddBool(&state, "dense_adaptive_triggered",
							  pgturbohybrid_last_graph_adaptive_triggered);
	PgturbohybridJsonbAddString(&state, "dense_adaptive_trigger_reason",
								PgturbohybridGraphDenseAdaptiveWideningReasonName(pgturbohybrid_last_graph_adaptive_trigger_reason));
	PgturbohybridJsonbAddInt64(&state, "dense_adaptive_initial_result_target",
							   pgturbohybrid_last_graph_adaptive_initial_result_target);
	PgturbohybridJsonbAddInt64(&state, "dense_adaptive_final_result_target",
							   pgturbohybrid_last_graph_adaptive_final_result_target);
	PgturbohybridJsonbAddInt64(&state, "dense_adaptive_initial_search_ef",
							   pgturbohybrid_last_graph_adaptive_initial_search_ef);
	PgturbohybridJsonbAddInt64(&state, "dense_adaptive_final_search_ef",
							   pgturbohybrid_last_graph_adaptive_final_search_ef);
	PgturbohybridJsonbAddFloat8(&state, "dense_adaptive_gap_top10",
								pgturbohybrid_last_graph_adaptive_gap_top10);
	PgturbohybridJsonbAddFloat8(&state, "dense_adaptive_gap_boundary",
								pgturbohybrid_last_graph_adaptive_gap_boundary);
	PgturbohybridJsonbAddBool(&state, "adaptive_widening_triggered",
							  pgturbohybrid_last_graph_adaptive_triggered);
	PgturbohybridJsonbAddString(&state, "adaptive_widening_reason",
								PgturbohybridGraphDenseAdaptiveWideningReasonName(pgturbohybrid_last_graph_adaptive_trigger_reason));
	PgturbohybridJsonbAddInt64(&state, "adaptive_initial_result_target",
							   pgturbohybrid_last_graph_adaptive_initial_result_target);
	PgturbohybridJsonbAddInt64(&state, "adaptive_final_result_target",
							   pgturbohybrid_last_graph_adaptive_final_result_target);
	PgturbohybridJsonbAddInt64(&state, "adaptive_initial_search_ef",
							   pgturbohybrid_last_graph_adaptive_initial_search_ef);
	PgturbohybridJsonbAddInt64(&state, "adaptive_final_search_ef",
							   pgturbohybrid_last_graph_adaptive_final_search_ef);
	PgturbohybridJsonbAddFloat8(&state, "adaptive_gap_top10",
								pgturbohybrid_last_graph_adaptive_gap_top10);
	PgturbohybridJsonbAddFloat8(&state, "adaptive_gap_boundary",
								pgturbohybrid_last_graph_adaptive_gap_boundary);
	PgturbohybridJsonbAddString(&state, "dense_uncertainty_retry_mode",
								PgturbohybridGraphDenseUncertaintyRetryModeName(
									pgturbohybrid_last_graph_uncertainty_retry_mode));
	PgturbohybridJsonbAddBool(&state, "dense_uncertainty_retry_triggered",
							  pgturbohybrid_last_graph_uncertainty_retry_triggered);
	PgturbohybridJsonbAddString(&state, "dense_uncertainty_retry_reason",
								PgturbohybridGraphDenseUncertaintyRetryReasonName(
									pgturbohybrid_last_graph_uncertainty_retry_reason));
	PgturbohybridJsonbAddInt64(&state, "dense_uncertainty_retry_passes",
							   pgturbohybrid_last_graph_uncertainty_retry_passes);
	PgturbohybridJsonbAddInt64(&state, "dense_uncertainty_initial_target",
							   pgturbohybrid_last_graph_uncertainty_initial_result_target);
	PgturbohybridJsonbAddInt64(&state, "dense_uncertainty_final_target",
							   pgturbohybrid_last_graph_uncertainty_final_result_target);
	PgturbohybridJsonbAddInt64(&state, "dense_uncertainty_initial_ef",
							   pgturbohybrid_last_graph_uncertainty_initial_search_ef);
	PgturbohybridJsonbAddInt64(&state, "dense_uncertainty_final_ef",
							   pgturbohybrid_last_graph_uncertainty_final_search_ef);
	PgturbohybridJsonbAddFloat8(&state, "dense_uncertainty_gap_top10",
								pgturbohybrid_last_graph_uncertainty_gap_top10);
	PgturbohybridJsonbAddFloat8(&state, "dense_uncertainty_gap_boundary",
								pgturbohybrid_last_graph_uncertainty_gap_boundary);
	PgturbohybridJsonbAddString(&state, "dense_local_expansion_mode",
								PgturbohybridGraphDenseLocalExpansionModeName(pgturbohybrid_last_graph_local_expansion_mode));
	PgturbohybridJsonbAddBool(&state, "dense_local_expansion_triggered",
							  pgturbohybrid_last_graph_local_expansion_triggered);
	PgturbohybridJsonbAddInt64(&state, "dense_local_expansion_seed_count",
							   pgturbohybrid_last_graph_local_expansion_seed_count);
	PgturbohybridJsonbAddInt64(&state, "dense_local_expansion_neighbors_scored",
							   pgturbohybrid_last_graph_local_expansion_neighbors_scored);
	PgturbohybridJsonbAddInt64(&state, "dense_local_expansion_candidates_added",
							   pgturbohybrid_last_graph_local_expansion_candidates_added);
	PgturbohybridJsonbAddInt64(&state, "dense_local_expansion_us",
							   pgturbohybrid_last_graph_local_expansion_us);
	PgturbohybridJsonbAddString(&state, "graph_dense_budget_policy",
								PgturbohybridGraphDenseBudgetPolicyNameExternal(pgturbohybrid_last_graph_dense_budget_policy));
	PgturbohybridJsonbAddString(&state, "graph_rescore_band_policy",
								PgturbohybridGraphRescoreBandPolicyNameExternal(pgturbohybrid_last_graph_rescore_band_policy));

	/*
	 * Additional grouped view: the same values organized into dense (with
	 * kernels/cache/traversal/timing_us sub-sections), bm25, fusion and query.
	 * Emitted from a single consolidated struct so a new hot-path flag has an
	 * obvious sectioned home and cannot be quietly dropped from diagnostics.
	 */
	{
		TqLastScanStats nested;

		PgturbohybridCollectLastScanStats(&nested, &scanStats, denseElapsedUs,
										  bm25ElapsedUs, fusionElapsedUs);
		PgturbohybridEmitNestedScanStats(&state, &nested);
	}

	PG_RETURN_JSONB_P(PgturbohybridJsonbEndObject(&state));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_simd_capabilities);
FUNCTION_PREFIX Datum
pgturbohybrid_simd_capabilities(PG_FUNCTION_ARGS)
{
	PgturbohybridJsonbState state;
	const char *architecture;
	bool		simdBuildDisabled;
	bool		compileAvx2;
	bool		compileAvx512Vnni;
	bool		compileAvx512Vpopcntdq;
	bool		compileAvx512Weighted;
	bool		compileAvxvnni;
	bool		compileArmDotprod;
	bool		compileArmI8mm;
	bool		runtimeAvx2 = false;
	bool		runtimeAvx512Vnni = false;
	bool		runtimeAvx512Vpopcntdq = false;
	bool		runtimeAvx512Weighted = false;
	bool		enabledAvx512Vnni = false;
	bool		enabledAvx512Vpopcntdq = false;
	bool		enabledAvx512Weighted = false;

#if defined(__aarch64__) || defined(_M_ARM64)
	architecture = "arm64";
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
	architecture = "amd64";
#else
	architecture = "unknown";
#endif
#if defined(PGTURBOHYBRID_DISABLE_SIMD)
	simdBuildDisabled = true;
#else
	simdBuildDisabled = false;
#endif
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && (defined(__AVX2__) || (defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))))
	compileAvx2 = true;
#else
	compileAvx2 = false;
#endif
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && (defined(__AVX512VNNI__) || (defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))))
	compileAvx512Vnni = true;
#else
	compileAvx512Vnni = false;
#endif
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && (defined(__AVX512VPOPCNTDQ__) || (defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))))
	compileAvx512Vpopcntdq = true;
#else
	compileAvx512Vpopcntdq = false;
#endif
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
	compileAvx512Weighted = true;
#else
	compileAvx512Weighted = false;
#endif
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && (defined(__AVXVNNI__) || (defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))))
	compileAvxvnni = true;
#else
	compileAvxvnni = false;
#endif
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && (defined(__aarch64__) || defined(_M_ARM64))
	compileArmDotprod = true;
	compileArmI8mm = true;
#else
	compileArmDotprod = false;
	compileArmI8mm = false;
#endif
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#if defined(__AVX2__)
	runtimeAvx2 = compileAvx2;
#else
	runtimeAvx2 = compileAvx2 && __builtin_cpu_supports("avx2");
#endif
#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512BW__)
	runtimeAvx512Vnni = compileAvx512Vnni;
#else
	runtimeAvx512Vnni = compileAvx512Vnni &&
		__builtin_cpu_supports("avx512vnni") &&
		__builtin_cpu_supports("avx512vl") &&
		__builtin_cpu_supports("avx512bw");
#endif
#if defined(__AVX512VPOPCNTDQ__) && defined(__AVX512BW__) && defined(__AVX512F__)
	runtimeAvx512Vpopcntdq = compileAvx512Vpopcntdq;
#else
	runtimeAvx512Vpopcntdq = compileAvx512Vpopcntdq &&
		__builtin_cpu_supports("avx512vpopcntdq") &&
		__builtin_cpu_supports("avx512bw") &&
		__builtin_cpu_supports("avx512f");
#endif
#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__) && defined(__AVX512VL__)
	runtimeAvx512Weighted = compileAvx512Weighted;
#else
	runtimeAvx512Weighted = compileAvx512Weighted &&
		__builtin_cpu_supports("avx512f") &&
		__builtin_cpu_supports("avx512bw") &&
		__builtin_cpu_supports("avx512dq") &&
		__builtin_cpu_supports("avx512vl");
#endif
#endif
	enabledAvx512Vnni = runtimeAvx512Vnni &&
		pgturbohybrid_dense_graph_avx512vnni &&
		(pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_AUTO ||
		 pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_AVX512VNNI);
	enabledAvx512Vpopcntdq = runtimeAvx512Vpopcntdq &&
		pgturbohybrid_dense_graph_avx512vpopcntdq &&
		(pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_AUTO ||
		 pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_AVX512VNNI);
	enabledAvx512Weighted = runtimeAvx512Weighted &&
		pgturbohybrid_dense_graph_avx512_weighted != PGTURBOHYBRID_GRAPH_AVX512_WEIGHTED_OFF &&
		(pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_AUTO ||
		 pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_AVX512VNNI);

	PgturbohybridJsonbStateInit(&state);
	PgturbohybridJsonbBeginObject(&state);
	PgturbohybridJsonbAddInt64(&state, "version", 1);
	PgturbohybridJsonbAddString(&state, "architecture", architecture);
	PgturbohybridJsonbAddString(&state, "simd_force",
								PgturbohybridGraphTqSimdForceName(pgturbohybrid_dense_simd_force));
	PgturbohybridJsonbAddString(&state, "avx512_weighted_policy",
								PgturbohybridGraphAvx512WeightedModeName(pgturbohybrid_dense_graph_avx512_weighted));
	PgturbohybridJsonbAddBool(&state, "simd_build_disabled", simdBuildDisabled);
	PgturbohybridJsonbAddBool(&state, "compile_avx2", compileAvx2);
	PgturbohybridJsonbAddBool(&state, "compile_avx512vnni", compileAvx512Vnni);
	PgturbohybridJsonbAddBool(&state, "compile_avx512vpopcntdq", compileAvx512Vpopcntdq);
	PgturbohybridJsonbAddBool(&state, "compile_avx512_weighted", compileAvx512Weighted);
	PgturbohybridJsonbAddBool(&state, "compile_avxvnni", compileAvxvnni);
	PgturbohybridJsonbAddBool(&state, "compile_arm_dotprod", compileArmDotprod);
	PgturbohybridJsonbAddBool(&state, "compile_arm_i8mm", compileArmI8mm);
	PgturbohybridJsonbAddBool(&state, "runtime_avx2", runtimeAvx2);
	PgturbohybridJsonbAddBool(&state, "runtime_avx512vnni", runtimeAvx512Vnni);
	PgturbohybridJsonbAddBool(&state, "runtime_avx512vpopcntdq", runtimeAvx512Vpopcntdq);
	PgturbohybridJsonbAddBool(&state, "runtime_avx512_weighted", runtimeAvx512Weighted);
	PgturbohybridJsonbAddBool(&state, "enabled_avx512vnni", enabledAvx512Vnni);
	PgturbohybridJsonbAddBool(&state, "enabled_avx512vpopcntdq", enabledAvx512Vpopcntdq);
	PgturbohybridJsonbAddBool(&state, "enabled_avx512_weighted", enabledAvx512Weighted);
	PG_RETURN_JSONB_P(PgturbohybridJsonbEndObject(&state));
}

#ifdef PGTURBOHYBRID_DEV_DIAGNOSTICS
static Datum
pgturbohybrid_last_simd_stats(PG_FUNCTION_ARGS)
{
	StringInfoData json;

	initStringInfo(&json);
	appendStringInfo(&json,
					 "{\"dense_graph_query_split_kernel\":\"%s\","
					 "\"dense_simd_kernel_counts\":{\"%s\":" INT64_FORMAT "},"
					 "\"graph_batch_scored_codes\":" INT64_FORMAT ","
					 "\"graph_scalar_scored_codes\":" INT64_FORMAT ","
					 "\"graph_batch_kernel\":\"%s\","
					 "\"exact_vector_kernel\":\"%s\","
					 "\"asymmetric_1bit_kernel\":\"scalar\","
					 "\"weighted_code_code_kernel\":\"%s\","
					 "\"avx512_weighted\":\"%s\","
					 "\"bm25_decode_kernel\":\"%s\","
					 "\"bm25_score_kernel\":\"%s\","
					 "\"bm25_simd_force\":\"%s\","
					 "\"bm25_simd_blocks\":" UINT64_FORMAT ","
					 "\"bm25_scalar_tail_postings\":" UINT64_FORMAT "}",
					 PgturbohybridGraphTqScoringKernelName(pgturbohybrid_last_graph_scoring_kernel),
					 PgturbohybridGraphTqScoringKernelName(pgturbohybrid_last_graph_scoring_kernel),
					 pgturbohybrid_last_graph_scored_codes,
					 pgturbohybrid_last_graph_batch_scored_codes,
					 pgturbohybrid_last_graph_scalar_scored_codes,
					 PgturbohybridGraphTqScoringKernelName(pgturbohybrid_last_graph_batch_kernel),
					 TqExactKernelName(pgturbohybrid_last_exact_vector_kernel),
					 PgturbohybridGraphTqScoringKernelName(pgturbohybrid_last_weighted_code_code_kernel),
					 PgturbohybridGraphAvx512WeightedModeName(pgturbohybrid_dense_graph_avx512_weighted),
					 PgturbohybridBm25KernelName(pgturbohybrid_last_bm25_decode_kernel),
					 PgturbohybridBm25KernelName(pgturbohybrid_last_bm25_score_kernel),
					 PgturbohybridBm25SimdForceName(pgturbohybrid_bm25_simd_force),
					 pgturbohybrid_last_bm25_simd_blocks,
					 pgturbohybrid_last_bm25_scalar_tail_postings);

	PG_RETURN_DATUM(DirectFunctionCall1(jsonb_in, CStringGetDatum(json.data)));
}
#endif
