#include "postgres.h"

#include "fmgr.h"
#include "pgturbohybrid.h"
#include "lib/stringinfo.h"
#include "pgturbohybrid_quant.h"
#include "pgturbohybrid_am.h"
#include "pgturbohybrid_bm25.h"
#include "utils/fmgrprotos.h"

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
static int pgturbohybrid_last_graph_dense_budget_policy = PGTURBOHYBRID_DENSE_BUDGET_AUTO;
static int pgturbohybrid_last_graph_rescore_band_policy = PGTURBOHYBRID_RESCORE_BAND_POLICY_AUTO;
static int pgturbohybrid_last_graph_scoring_kernel = PGTURBOHYBRID_SCORING_SCALAR;
static int pgturbohybrid_last_exact_vector_kernel = PGTURBOHYBRID_EXACT_KERNEL_SCALAR;
static bool pgturbohybrid_last_exact_vector_kernel_recorded = false;
static int pgturbohybrid_last_graph_storage_kind = PGTURBOHYBRID_GRAPH_STORAGE_GRAPH;
static int pgturbohybrid_last_scan_orchestration = PGTURBOHYBRID_SCAN_ORCHESTRATION_NONE;
static int pgturbohybrid_last_graph_quantization_bits = 0;
static bool pgturbohybrid_last_graph_query_split_active = false;

static const char *TqScanOrchestrationName(void);
#ifdef PGTURBOHYBRID_DEV_DIAGNOSTICS
static const char *TqExactKernelName(int kernel);
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
	pgturbohybrid_last_graph_dense_budget_policy = so->graphDenseBudgetPolicy;
	pgturbohybrid_last_graph_rescore_band_policy = so->graphRescoreBandPolicy;
	pgturbohybrid_last_graph_scoring_kernel = so->tq.enabled ? so->tq.scoringKernel : PGTURBOHYBRID_SCORING_SCALAR;
	if (!pgturbohybrid_last_exact_vector_kernel_recorded && so->graphRescoreCount > 0)
		pgturbohybrid_last_exact_vector_kernel = TqExpectedExactKernel();
	pgturbohybrid_last_graph_storage_kind = so->graphStorageKind;
	pgturbohybrid_last_graph_quantization_bits = so->tq.enabled ? so->tq.bits : 0;
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
	pgturbohybrid_last_graph_dense_budget_policy = PGTURBOHYBRID_DENSE_BUDGET_AUTO;
	pgturbohybrid_last_graph_rescore_band_policy = PGTURBOHYBRID_RESCORE_BAND_POLICY_AUTO;
	pgturbohybrid_last_graph_scoring_kernel = PGTURBOHYBRID_SCORING_SCALAR;
	pgturbohybrid_last_exact_vector_kernel = PGTURBOHYBRID_EXACT_KERNEL_SCALAR;
	pgturbohybrid_last_exact_vector_kernel_recorded = false;
	pgturbohybrid_last_graph_storage_kind = PGTURBOHYBRID_GRAPH_STORAGE_GRAPH;
	pgturbohybrid_last_graph_quantization_bits = 0;
	pgturbohybrid_last_graph_query_split_active = false;
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

#ifdef PGTURBOHYBRID_DEV_DIAGNOSTICS
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

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_last_scan_stats);
Datum
pgturbohybrid_last_scan_stats(PG_FUNCTION_ARGS)
{
	StringInfoData json;

	initStringInfo(&json);
	appendStringInfo(&json,
					 "{\"version\":1,"
					 "\"scan_orchestration\":\"%s\","
					 "\"graph_storage_kind\":\"%s\","
					 "\"quantization_bits\":%d,"
					 "\"graph_candidate_count\":" INT64_FORMAT ","
					 "\"graph_rescore_count\":" INT64_FORMAT ","
					 "\"graph_dense_requested_k\":" INT64_FORMAT ","
					 "\"graph_effective_result_target\":" INT64_FORMAT ","
					 "\"graph_effective_search_ef\":" INT64_FORMAT ","
					 "\"graph_effective_rescore_band\":" INT64_FORMAT ","
					 "\"graph_highdim_widening_multiplier\":%.3f,"
					 "\"graph_widening_reason\":\"%s\","
					 "\"graph_dense_budget_policy\":\"%s\","
					 "\"graph_rescore_band_policy\":\"%s\"}",
					 TqScanOrchestrationName(),
					 PgturbohybridGraphStorageKindName(pgturbohybrid_last_graph_storage_kind),
					 pgturbohybrid_last_graph_quantization_bits,
					 pgturbohybrid_last_graph_candidate_count,
					 pgturbohybrid_last_graph_rescore_count,
					 pgturbohybrid_last_graph_dense_requested_k,
					 pgturbohybrid_last_graph_effective_result_target,
					 pgturbohybrid_last_graph_effective_search_ef,
					 pgturbohybrid_last_graph_effective_rescore_band,
					 pgturbohybrid_last_graph_highdim_widening_multiplier,
					 PgturbohybridGraphDenseWideningReasonName(pgturbohybrid_last_graph_widening_reason),
					 PgturbohybridGraphDenseBudgetPolicyNameExternal(pgturbohybrid_last_graph_dense_budget_policy),
					 PgturbohybridGraphRescoreBandPolicyNameExternal(pgturbohybrid_last_graph_rescore_band_policy));

	PG_RETURN_DATUM(DirectFunctionCall1(jsonb_in, CStringGetDatum(json.data)));
}

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_simd_capabilities);
Datum
pgturbohybrid_simd_capabilities(PG_FUNCTION_ARGS)
{
	StringInfoData json;

	initStringInfo(&json);
	appendStringInfo(&json,
					 "{\"version\":1,"
					 "\"architecture\":\"%s\","
					 "\"simd_build_disabled\":%s,"
					 "\"compile_avx2\":%s,"
					 "\"compile_avx512vnni\":%s,"
					 "\"compile_avx512vpopcntdq\":%s,"
					 "\"compile_avx512_weighted\":%s,"
					 "\"compile_avxvnni\":%s,"
					 "\"compile_arm_dotprod\":%s,"
					 "\"compile_arm_i8mm\":%s}",
#if defined(__aarch64__) || defined(_M_ARM64)
					 "arm64",
#elif defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
					 "amd64",
#else
					 "unknown",
#endif
#if defined(PGTURBOHYBRID_DISABLE_SIMD)
					 "true",
#else
					 "false",
#endif
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && (defined(__AVX2__) || (defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))))
					 "true",
#else
					 "false",
#endif
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && (defined(__AVX512VNNI__) || (defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))))
					 "true",
#else
					 "false",
#endif
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && (defined(__AVX512VPOPCNTDQ__) || (defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))))
					 "true",
#else
					 "false",
#endif
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
					 "true",
#else
					 "false",
#endif
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && (defined(__AVXVNNI__) || (defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))))
					 "true",
#else
					 "false",
#endif
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && (defined(__aarch64__) || defined(_M_ARM64))
					 "true",
					 "true"
#else
					 "false",
					 "false"
#endif
		);

	PG_RETURN_DATUM(DirectFunctionCall1(jsonb_in, CStringGetDatum(json.data)));
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
