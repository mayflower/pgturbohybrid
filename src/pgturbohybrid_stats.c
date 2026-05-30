#include "postgres.h"

#include <limits.h>
#include <string.h>

#include "fmgr.h"
#include "pgturbohybrid.h"
#include "lib/stringinfo.h"
#include "pgturbohybrid_quant.h"
#include "pgturbohybrid_am.h"
#include "pgturbohybrid_bm25.h"
#include "pgturbohybrid_vector_compat.h"
#include "pgturbohybrid_jsonb_compat.h"
#include "utils/fmgrprotos.h"
#include "utils/jsonb.h"
#include "utils/numeric.h"

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
static int64 pgturbohybrid_last_graph_rescore_count = 0;
static int64 pgturbohybrid_last_graph_rescore_pages = 0;
static int64 pgturbohybrid_last_graph_code_pages_read = 0;
static int64 pgturbohybrid_last_graph_adj_pages_read = 0;
static int64 pgturbohybrid_last_graph_entry_point_count = 0;
static int64 pgturbohybrid_last_graph_entry_sidecar_count = 0;
static int64 pgturbohybrid_last_graph_entry_sidecar_scored = 0;
static int64 pgturbohybrid_last_graph_entry_sidecar_selected = 0;
static int64 pgturbohybrid_last_graph_entry_sidecar_us = 0;
static int64 pgturbohybrid_last_graph_residual_rerank_count = 0;
static int64 pgturbohybrid_last_graph_residual_rerank_bytes = 0;
static int64 pgturbohybrid_last_graph_residual_rerank_us = 0;
static int64 pgturbohybrid_last_graph_prepare_us = 0;
static int64 pgturbohybrid_last_graph_traverse_us = 0;
static int64 pgturbohybrid_last_graph_entry_us = 0;
static int64 pgturbohybrid_last_graph_base_us = 0;
static int64 pgturbohybrid_last_graph_batch_us = 0;
static int64 pgturbohybrid_last_graph_heap_us = 0;
static int64 pgturbohybrid_last_graph_fill_us = 0;
static int64 pgturbohybrid_last_graph_rescore_us = 0;
static int64 pgturbohybrid_last_graph_sort_us = 0;
static int64 pgturbohybrid_last_graph_total_us = 0;
static int64 pgturbohybrid_last_graph_dense_requested_k = 0;
static int64 pgturbohybrid_last_graph_effective_result_target = 0;
static int64 pgturbohybrid_last_graph_effective_search_ef = 0;
static int64 pgturbohybrid_last_graph_effective_rescore_band = 0;
static double pgturbohybrid_last_graph_highdim_widening_multiplier = 1.0;
static int pgturbohybrid_last_graph_widening_reason = PGTURBOHYBRID_DENSE_WIDENING_NONE;
static int pgturbohybrid_last_graph_adaptive_widening_mode = PGTURBOHYBRID_DENSE_ADAPTIVE_WIDENING_OFF;
static bool pgturbohybrid_last_graph_adaptive_triggered = false;
static int pgturbohybrid_last_graph_adaptive_trigger_reason = PGTURBOHYBRID_DENSE_ADAPTIVE_REASON_NONE;
static int64 pgturbohybrid_last_graph_adaptive_initial_result_target = 0;
static int64 pgturbohybrid_last_graph_adaptive_final_result_target = 0;
static int64 pgturbohybrid_last_graph_adaptive_initial_search_ef = 0;
static int64 pgturbohybrid_last_graph_adaptive_final_search_ef = 0;
static double pgturbohybrid_last_graph_adaptive_gap_top10 = 0.0;
static double pgturbohybrid_last_graph_adaptive_gap_boundary = 0.0;
static int pgturbohybrid_last_graph_local_expansion_mode = PGTURBOHYBRID_DENSE_LOCAL_EXPANSION_OFF;
static bool pgturbohybrid_last_graph_local_expansion_triggered = false;
static int64 pgturbohybrid_last_graph_local_expansion_seed_count = 0;
static int64 pgturbohybrid_last_graph_local_expansion_neighbors_scored = 0;
static int64 pgturbohybrid_last_graph_local_expansion_candidates_added = 0;
static int64 pgturbohybrid_last_graph_local_expansion_us = 0;
static int pgturbohybrid_last_graph_dense_budget_policy = PGTURBOHYBRID_DENSE_BUDGET_AUTO;
static int pgturbohybrid_last_graph_rescore_band_policy = PGTURBOHYBRID_RESCORE_BAND_POLICY_AUTO;
static int pgturbohybrid_last_graph_scoring_kernel = PGTURBOHYBRID_SCORING_SCALAR;
static int pgturbohybrid_last_exact_vector_kernel = PGTURBOHYBRID_EXACT_KERNEL_SCALAR;
static bool pgturbohybrid_last_exact_vector_kernel_recorded = false;
static int pgturbohybrid_last_graph_storage_kind = PGTURBOHYBRID_GRAPH_STORAGE_GRAPH;
static int pgturbohybrid_last_scan_orchestration = PGTURBOHYBRID_SCAN_ORCHESTRATION_NONE;
static int pgturbohybrid_last_graph_quantization_bits = 0;
static bool pgturbohybrid_last_graph_exact_storage = false;
static bool pgturbohybrid_last_graph_exact_storage_known = false;
static bool pgturbohybrid_last_graph_query_split_active = false;
static int64 pgturbohybrid_last_graph_dimensions = 0;
static int64 pgturbohybrid_last_graph_returned_rows = 0;
static int64 pgturbohybrid_last_graph_oversampling = 0;
static int pgturbohybrid_last_graph_exact_cache = PGTURBOHYBRID_GRAPH_EXACT_CACHE_AUTO;

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
PgturbohybridJsonbAddFloat8(PgturbohybridJsonbState *state, const char *key, double val)
{
	JsonbValue	value;

	PgturbohybridJsonbAddKey(state, key);
	value.type = jbvNumeric;
	value.val.numeric = DatumGetNumeric(DirectFunctionCall1(float8_numeric,
															Float8GetDatum(val)));
	PgturbohybridJsonbPush(state, WJB_VALUE, &value);
}

static const char *TqScanOrchestrationName(void);
static const char *PgturbohybridFinalKSourceName(void);
static const char *TqExactKernelName(int kernel);
#ifdef PGTURBOHYBRID_DEV_DIAGNOSTICS
static const char *PgturbohybridGraphAvx512WeightedModeName(int mode);
#endif

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
	if (pgturbohybrid_dense_exact_simd_force != PGTURBOHYBRID_EXACT_SIMD_FORCE_AVX512F)
		return PGTURBOHYBRID_EXACT_KERNEL_NEON;
#endif
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && defined(__x86_64__)
	if (pgturbohybrid_dense_exact_simd_force == PGTURBOHYBRID_EXACT_SIMD_FORCE_AVX512F)
		return PGTURBOHYBRID_EXACT_KERNEL_AVX512F;
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
	pgturbohybrid_last_graph_rescore_count = so->graphRescoreCount;
	pgturbohybrid_last_graph_rescore_pages = so->graphRescorePages;
	pgturbohybrid_last_graph_code_pages_read = so->graphCodePagesRead;
	pgturbohybrid_last_graph_adj_pages_read = so->graphAdjPagesRead;
	pgturbohybrid_last_graph_entry_point_count = so->graphEntryPointCount;
	pgturbohybrid_last_graph_entry_sidecar_count = so->graphEntrySidecarCount;
	pgturbohybrid_last_graph_entry_sidecar_scored = so->graphEntrySidecarScored;
	pgturbohybrid_last_graph_entry_sidecar_selected = so->graphEntrySidecarSelected;
	pgturbohybrid_last_graph_entry_sidecar_us = so->graphEntrySidecarUs;
	pgturbohybrid_last_graph_residual_rerank_count = so->graphResidualRerankCount;
	pgturbohybrid_last_graph_residual_rerank_bytes = so->graphResidualRerankBytes;
	pgturbohybrid_last_graph_residual_rerank_us = so->graphResidualRerankUs;
	pgturbohybrid_last_graph_prepare_us = so->graphPrepareUs;
	pgturbohybrid_last_graph_traverse_us = so->graphTraverseUs;
	pgturbohybrid_last_graph_entry_us = so->graphEntryUs;
	pgturbohybrid_last_graph_base_us = so->graphBaseUs;
	pgturbohybrid_last_graph_batch_us = so->graphBatchUs;
	pgturbohybrid_last_graph_heap_us = so->graphHeapUs;
	pgturbohybrid_last_graph_fill_us = so->graphFillUs;
	pgturbohybrid_last_graph_rescore_us = so->graphRescoreUs;
	pgturbohybrid_last_graph_sort_us = so->graphSortUs;
	pgturbohybrid_last_graph_total_us = so->graphTotalUs;
	pgturbohybrid_last_graph_dense_requested_k = so->graphDenseRequestedK;
	pgturbohybrid_last_graph_effective_result_target = so->graphEffectiveResultTarget;
	pgturbohybrid_last_graph_effective_search_ef = so->graphEffectiveSearchEf;
	pgturbohybrid_last_graph_effective_rescore_band = so->graphEffectiveRescoreBand;
	pgturbohybrid_last_graph_highdim_widening_multiplier = so->graphHighdimWideningMultiplier;
	pgturbohybrid_last_graph_widening_reason = so->graphWideningReason;
	pgturbohybrid_last_graph_adaptive_widening_mode = so->graphAdaptiveWideningMode;
	pgturbohybrid_last_graph_adaptive_triggered = so->graphAdaptiveTriggered;
	pgturbohybrid_last_graph_adaptive_trigger_reason = so->graphAdaptiveTriggerReason;
	pgturbohybrid_last_graph_adaptive_initial_result_target = so->graphAdaptiveInitialResultTarget;
	pgturbohybrid_last_graph_adaptive_final_result_target = so->graphAdaptiveFinalResultTarget;
	pgturbohybrid_last_graph_adaptive_initial_search_ef = so->graphAdaptiveInitialSearchEf;
	pgturbohybrid_last_graph_adaptive_final_search_ef = so->graphAdaptiveFinalSearchEf;
	pgturbohybrid_last_graph_adaptive_gap_top10 = so->graphAdaptiveGapTop10;
	pgturbohybrid_last_graph_adaptive_gap_boundary = so->graphAdaptiveGapBoundary;
	pgturbohybrid_last_graph_local_expansion_mode = so->graphLocalExpansionMode;
	pgturbohybrid_last_graph_local_expansion_triggered = so->graphLocalExpansionTriggered;
	pgturbohybrid_last_graph_local_expansion_seed_count = so->graphLocalExpansionSeedCount;
	pgturbohybrid_last_graph_local_expansion_neighbors_scored = so->graphLocalExpansionNeighborsScored;
	pgturbohybrid_last_graph_local_expansion_candidates_added = so->graphLocalExpansionCandidatesAdded;
	pgturbohybrid_last_graph_local_expansion_us = so->graphLocalExpansionUs;
	pgturbohybrid_last_graph_dense_budget_policy = so->graphDenseBudgetPolicy;
	pgturbohybrid_last_graph_rescore_band_policy = so->graphRescoreBandPolicy;
	pgturbohybrid_last_graph_scoring_kernel = so->tq.enabled ? so->tq.scoringKernel : PGTURBOHYBRID_SCORING_SCALAR;
	if (!pgturbohybrid_last_exact_vector_kernel_recorded && so->graphRescoreCount > 0)
		pgturbohybrid_last_exact_vector_kernel = TqExpectedExactKernel();
	pgturbohybrid_last_graph_storage_kind = so->graphStorageKind;
	pgturbohybrid_last_graph_quantization_bits = so->tq.enabled ? so->tq.bits : 0;
	pgturbohybrid_last_graph_dimensions = so->tq.enabled ? so->tq.dimensions : 0;
	pgturbohybrid_last_graph_returned_rows = so->returnedRows;
	pgturbohybrid_last_graph_oversampling = so->graphOversampling;
	pgturbohybrid_last_graph_exact_cache = so->graphExactCache;
	pgturbohybrid_last_graph_exact_storage = so->graphExactStorage;
	pgturbohybrid_last_graph_exact_storage_known = so->tq.enabled;
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
	pgturbohybrid_last_graph_query_split_active = so->tq.enabled && so->tq.querySplitEnabled;
#else
	pgturbohybrid_last_graph_query_split_active = false;
#endif
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
	pgturbohybrid_last_graph_candidate_count = 0;
	pgturbohybrid_last_graph_rescore_count = 0;
	pgturbohybrid_last_graph_rescore_pages = 0;
	pgturbohybrid_last_graph_code_pages_read = 0;
	pgturbohybrid_last_graph_adj_pages_read = 0;
	pgturbohybrid_last_graph_entry_point_count = 0;
	pgturbohybrid_last_graph_entry_sidecar_count = 0;
	pgturbohybrid_last_graph_entry_sidecar_scored = 0;
	pgturbohybrid_last_graph_entry_sidecar_selected = 0;
	pgturbohybrid_last_graph_entry_sidecar_us = 0;
	pgturbohybrid_last_graph_residual_rerank_count = 0;
	pgturbohybrid_last_graph_residual_rerank_bytes = 0;
	pgturbohybrid_last_graph_residual_rerank_us = 0;
	pgturbohybrid_last_graph_prepare_us = 0;
	pgturbohybrid_last_graph_traverse_us = 0;
	pgturbohybrid_last_graph_entry_us = 0;
	pgturbohybrid_last_graph_base_us = 0;
	pgturbohybrid_last_graph_batch_us = 0;
	pgturbohybrid_last_graph_heap_us = 0;
	pgturbohybrid_last_graph_fill_us = 0;
	pgturbohybrid_last_graph_rescore_us = 0;
	pgturbohybrid_last_graph_sort_us = 0;
	pgturbohybrid_last_graph_total_us = 0;
	pgturbohybrid_last_graph_dense_requested_k = 0;
	pgturbohybrid_last_graph_effective_result_target = 0;
	pgturbohybrid_last_graph_effective_search_ef = 0;
	pgturbohybrid_last_graph_effective_rescore_band = 0;
	pgturbohybrid_last_graph_highdim_widening_multiplier = 1.0;
	pgturbohybrid_last_graph_widening_reason = PGTURBOHYBRID_DENSE_WIDENING_NONE;
	pgturbohybrid_last_graph_adaptive_widening_mode = PGTURBOHYBRID_DENSE_ADAPTIVE_WIDENING_OFF;
	pgturbohybrid_last_graph_adaptive_triggered = false;
	pgturbohybrid_last_graph_adaptive_trigger_reason = PGTURBOHYBRID_DENSE_ADAPTIVE_REASON_NONE;
	pgturbohybrid_last_graph_adaptive_initial_result_target = 0;
	pgturbohybrid_last_graph_adaptive_final_result_target = 0;
	pgturbohybrid_last_graph_adaptive_initial_search_ef = 0;
	pgturbohybrid_last_graph_adaptive_final_search_ef = 0;
	pgturbohybrid_last_graph_adaptive_gap_top10 = 0.0;
	pgturbohybrid_last_graph_adaptive_gap_boundary = 0.0;
	pgturbohybrid_last_graph_local_expansion_mode = PGTURBOHYBRID_DENSE_LOCAL_EXPANSION_OFF;
	pgturbohybrid_last_graph_local_expansion_triggered = false;
	pgturbohybrid_last_graph_local_expansion_seed_count = 0;
	pgturbohybrid_last_graph_local_expansion_neighbors_scored = 0;
	pgturbohybrid_last_graph_local_expansion_candidates_added = 0;
	pgturbohybrid_last_graph_local_expansion_us = 0;
	pgturbohybrid_last_graph_dense_budget_policy = PGTURBOHYBRID_DENSE_BUDGET_AUTO;
	pgturbohybrid_last_graph_rescore_band_policy = PGTURBOHYBRID_RESCORE_BAND_POLICY_AUTO;
	pgturbohybrid_last_graph_scoring_kernel = PGTURBOHYBRID_SCORING_SCALAR;
	pgturbohybrid_last_exact_vector_kernel = PGTURBOHYBRID_EXACT_KERNEL_SCALAR;
	pgturbohybrid_last_exact_vector_kernel_recorded = false;
	pgturbohybrid_last_graph_storage_kind = PGTURBOHYBRID_GRAPH_STORAGE_GRAPH;
	pgturbohybrid_last_graph_quantization_bits = 0;
	pgturbohybrid_last_graph_exact_storage = false;
	pgturbohybrid_last_graph_exact_storage_known = false;
	pgturbohybrid_last_graph_query_split_active = false;
	pgturbohybrid_last_graph_dimensions = 0;
	pgturbohybrid_last_graph_returned_rows = 0;
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
		case PGTURBOHYBRID_EXACT_KERNEL_AVX512F:
			return "avx512f";
		default:
			return "unknown";
	}
}

#ifdef PGTURBOHYBRID_DEV_DIAGNOSTICS
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
	if (pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_SCALAR)
		return "scalar";
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && (defined(__x86_64__) || defined(_M_X64))
	return "avx2_lut_gather";
#else
	return "scalar";
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
	PgturbohybridJsonbAddInt64(&state, "quantization_bits",
							   pgturbohybrid_last_graph_quantization_bits);
	if (pgturbohybrid_last_graph_exact_storage_known)
		PgturbohybridJsonbAddBool(&state, "exact_storage",
								  pgturbohybrid_last_graph_exact_storage);
	else
		PgturbohybridJsonbAddNull(&state, "exact_storage");

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
	PgturbohybridJsonbAddInt64(&state, "graph_oversampling",
							   pgturbohybrid_last_graph_oversampling);
	PgturbohybridJsonbAddString(&state, "graph_exact_cache",
								PgturbohybridGraphExactCacheName(pgturbohybrid_last_graph_exact_cache));
	PgturbohybridJsonbAddBool(&state, "graph_exact_cache_active",
							  pgturbohybrid_last_graph_exact_cache != PGTURBOHYBRID_GRAPH_EXACT_CACHE_OFF);
	PgturbohybridJsonbAddBool(&state, "residual_rerank_active",
							  pgturbohybrid_last_graph_residual_rerank_count > 0);
	PgturbohybridJsonbAddBool(&state, "graph_rescore_band_active",
							  pgturbohybrid_last_graph_effective_rescore_band > 0);
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
	PgturbohybridJsonbAddInt64(&state, "graph_code_pages_read",
							   pgturbohybrid_last_graph_code_pages_read);
	PgturbohybridJsonbAddInt64(&state, "graph_adj_pages_read",
							   pgturbohybrid_last_graph_adj_pages_read);
	PgturbohybridJsonbAddInt64(&state, "graph_entry_point_count",
							   pgturbohybrid_last_graph_entry_point_count);
	PgturbohybridJsonbAddInt64(&state, "graph_entry_sidecar_count",
							   pgturbohybrid_last_graph_entry_sidecar_count);
	PgturbohybridJsonbAddInt64(&state, "graph_entry_sidecar_scored",
							   pgturbohybrid_last_graph_entry_sidecar_scored);
	PgturbohybridJsonbAddInt64(&state, "graph_entry_sidecar_selected",
							   pgturbohybrid_last_graph_entry_sidecar_selected);
	PgturbohybridJsonbAddInt64(&state, "graph_entry_sidecar_us",
							   pgturbohybrid_last_graph_entry_sidecar_us);
	PgturbohybridJsonbAddInt64(&state, "dense_residual_rerank_count",
							   pgturbohybrid_last_graph_residual_rerank_count);
	PgturbohybridJsonbAddInt64(&state, "dense_residual_rerank_bytes",
							   pgturbohybrid_last_graph_residual_rerank_bytes);
	PgturbohybridJsonbAddInt64(&state, "dense_residual_rerank_us",
							   pgturbohybrid_last_graph_residual_rerank_us);
	PgturbohybridJsonbAddInt64(&state, "graph_rescore_count",
							   pgturbohybrid_last_graph_rescore_count);
	PgturbohybridJsonbAddInt64(&state, "graph_dense_requested_k",
							   pgturbohybrid_last_graph_dense_requested_k);
	PgturbohybridJsonbAddInt64(&state, "graph_effective_result_target",
							   pgturbohybrid_last_graph_effective_result_target);
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
	PgturbohybridJsonbAddBool(&state, "auto_budget", pgturbohybrid_auto_budget);
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
	PgturbohybridJsonbAddBool(&state, "bm25_hot_postings_cache_hit",
							  scanStats.bm25HotPostingsCacheHits > 0);
	PgturbohybridJsonbAddUint64(&state, "bm25_hot_postings_cache_hits",
								scanStats.bm25HotPostingsCacheHits);
	PgturbohybridJsonbAddUint64(&state, "bm25_hot_postings_cache_misses",
								scanStats.bm25HotPostingsCacheMisses);
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

	PgturbohybridJsonbStateInit(&state);
	PgturbohybridJsonbBeginObject(&state);
	PgturbohybridJsonbAddInt64(&state, "version", 1);
	PgturbohybridJsonbAddString(&state, "architecture", architecture);
	PgturbohybridJsonbAddBool(&state, "simd_build_disabled", simdBuildDisabled);
	PgturbohybridJsonbAddBool(&state, "compile_avx2", compileAvx2);
	PgturbohybridJsonbAddBool(&state, "compile_avx512vnni", compileAvx512Vnni);
	PgturbohybridJsonbAddBool(&state, "compile_avx512vpopcntdq", compileAvx512Vpopcntdq);
	PgturbohybridJsonbAddBool(&state, "compile_avx512_weighted", compileAvx512Weighted);
	PgturbohybridJsonbAddBool(&state, "compile_avxvnni", compileAvxvnni);
	PgturbohybridJsonbAddBool(&state, "compile_arm_dotprod", compileArmDotprod);
	PgturbohybridJsonbAddBool(&state, "compile_arm_i8mm", compileArmI8mm);
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
