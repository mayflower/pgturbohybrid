#ifndef PGTURBOHYBRID_H
#define PGTURBOHYBRID_H

#include "postgres.h"

#include <math.h>

#include "access/genam.h"
#include "access/parallel.h"
#include "common/relpath.h"
#include "lib/pairingheap.h"
#include "nodes/execnodes.h"
#include "port.h"				/* for random() */
#include "storage/bufpage.h"
#include "storage/condition_variable.h"
#include "storage/lwlock.h"
#include "storage/s_lock.h"
#include "utils/relptr.h"
#include "utils/memutils.h"
#include "utils/sampling.h"
#include "pgturbohybrid_vector_compat.h"

#if PG_VERSION_NUM >= 190000
typedef Pointer Item;
#endif

/*
 * TurboHybrid stores quantized graph codes instead of full pgvector HNSW
 * tuples, so its dense dimension limit should track the vector type limit.
 */
#define PGTURBOHYBRID_GRAPH_MAX_DIM PGTURBOHYBRID_VECTOR_MAX_DIM
#define PGTURBOHYBRID_GRAPH_MAX_NNZ 1000

/* Support functions */
#define PGTURBOHYBRID_GRAPH_DISTANCE_PROC 1
#define PGTURBOHYBRID_GRAPH_NORM_PROC 2
#define PGTURBOHYBRID_GRAPH_TYPE_INFO_PROC 3

#define PGTURBOHYBRID_VERSION	1
#define PGTURBOHYBRID_MAGIC_NUMBER 0x54525944
#define PGTURBOHYBRID_PAGE_ID	0x5459
#define PGTURBOHYBRID_GRAPH_VERSION PGTURBOHYBRID_VERSION
#define PGTURBOHYBRID_GRAPH_MAGIC_NUMBER PGTURBOHYBRID_MAGIC_NUMBER
#define PGTURBOHYBRID_GRAPH_PAGE_ID PGTURBOHYBRID_PAGE_ID

/* Preserved page numbers */
#define PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO	0
#define PGTURBOHYBRID_GRAPH_HEAD_BLKNO		1	/* first element page */

/* Must correspond to page numbers since page lock is used */
#define PGTURBOHYBRID_GRAPH_UPDATE_LOCK 	0
#define PGTURBOHYBRID_GRAPH_SCAN_LOCK		1

/* Graph parameters */
#define PGTURBOHYBRID_GRAPH_DEFAULT_M	16
#define PGTURBOHYBRID_GRAPH_MIN_M	2
#define PGTURBOHYBRID_GRAPH_MAX_M		100
#define PGTURBOHYBRID_GRAPH_DEFAULT_EF_CONSTRUCTION	64
#define PGTURBOHYBRID_GRAPH_MIN_EF_CONSTRUCTION	4
#define PGTURBOHYBRID_GRAPH_MAX_EF_CONSTRUCTION		1000
#define PGTURBOHYBRID_GRAPH_DEFAULT_EF_SEARCH	40
#define PGTURBOHYBRID_GRAPH_MIN_EF_SEARCH		1
#define PGTURBOHYBRID_GRAPH_MAX_EF_SEARCH		1000

/* pgturbohybrid graph routing defaults */
#define PGTURBOHYBRID_DEFAULT_GRAPH_M 16
#define PGTURBOHYBRID_DEFAULT_GRAPH_EF_CONSTRUCTION 128
#define PGTURBOHYBRID_DEFAULT_GRAPH_EF_SEARCH 64
#define PGTURBOHYBRID_DEFAULT_GRAPH_OVERSAMPLING 4
#define PGTURBOHYBRID_GRAPH_EXACT_CACHE_AUTO_MAX_BYTES (16 * 1024 * 1024)
#define PGTURBOHYBRID_GRAPH_MAX_ROUTING_ENTRIES 15
#define PGTURBOHYBRID_GRAPH_MAX_ENTRY_SIDECAR_REPRESENTATIVES 256
#define PGTURBOHYBRID_GRAPH_MAX_NATIVE_SEGMENTS 16
#define PGTURBOHYBRID_NATIVE_BUILD_STATS_MAX_WORKERS 16
#define PGTURBOHYBRID_DEFAULT_ENTRY_SIDECAR false
#define PGTURBOHYBRID_DEFAULT_ENTRY_SIDECAR_REPRESENTATIVES 128
#define PGTURBOHYBRID_DEFAULT_ENTRY_SIDECAR_STRATEGY 0
#define PGTURBOHYBRID_DEFAULT_PAYLOAD_ENTRY_SEED_COUNT 8
#define PGTURBOHYBRID_MAX_PAYLOAD_ENTRY_SEED_COUNT 64
#define PGTURBOHYBRID_DEFAULT_GRAPH_BACKBONE false
#define PGTURBOHYBRID_GRAPH_MAX_RESIDUAL_RERANK_BYTES 64
#define PGTURBOHYBRID_DEFAULT_RESIDUAL_RERANK false
#define PGTURBOHYBRID_DEFAULT_RESIDUAL_RERANK_BYTES 32
#define PGTURBOHYBRID_DEFAULT_INDEX_BITS 4
#define PGTURBOHYBRID_DEFAULT_EXACT_STORAGE false
#define PGTURBOHYBRID_DEFAULT_BITS 4
#define PGTURBOHYBRID_BITS 4
#define PGTURBOHYBRID_LUT_WIDTH 16
#define PGTURBOHYBRID_CODE_SIZE_BITS(dim, bits) ((((dim) * (bits)) + 7) / 8)
#define PGTURBOHYBRID_CODE_SIZE(dim) (((dim) + 1) / 2)
#define PGTURBOHYBRID_CODE_SCALE_OFFSET(dim) PGTURBOHYBRID_CODE_SIZE(dim)
#define PGTURBOHYBRID_CODE_PAYLOAD_SIZE(dim) (PGTURBOHYBRID_CODE_SIZE(dim) + sizeof(float))
#define PGTURBOHYBRID_LUT_SIZE(dim) ((dim) * PGTURBOHYBRID_LUT_WIDTH * sizeof(float))
#ifndef PGTURBOHYBRID_GRAPH_ENABLE_SYMMETRIC_I8_DOT
#define PGTURBOHYBRID_GRAPH_ENABLE_SYMMETRIC_I8_DOT 0
#endif

/* Tuple types */
#define PGTURBOHYBRID_GRAPH_ELEMENT_TUPLE_TYPE  1
#define PGTURBOHYBRID_GRAPH_NEIGHBOR_TUPLE_TYPE 2

/* Page and storage identities */
#define PGTURBOHYBRID_GRAPH_STORAGE_GRAPH				0
#define PGTURBOHYBRID_GRAPH_STORAGE_QUANT_GRAPH	1
#define PGTURBOHYBRID_GRAPH_STORAGE_QUANT_FLAT	2
#define PGTURBOHYBRID_GRAPH_STORAGE_QUANT_IVF		3
#define PGTURBOHYBRID_GRAPH_STORAGE_QUANT_GRAPH_NATIVE	4

#define PGTURBOHYBRID_MULTIVECTOR_GRAPH_TOKEN_NODES		0
#define PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES	1
#define PGTURBOHYBRID_DEFAULT_MULTIVECTOR_GRAPH_MODE \
	PGTURBOHYBRID_MULTIVECTOR_GRAPH_TOKEN_NODES
#define PGTURBOHYBRID_GRAPH_PAGE_KIND_GRAPH			1
#define PGTURBOHYBRID_GRAPH_PAGE_KIND_META				2
#define PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_CODE			3
#define PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_ADJ			4
#define PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_EXACT			5
#define PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_CORRECTION	6
#define PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_META		7
#define PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_DOCSTATS	8
#define PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_LEXICON	9
#define PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_POSTINGS	10
#define PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_BLOCKMAX	11
#define PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_DELTA	12
#define PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_IMPACT	13
#define PGTURBOHYBRID_GRAPH_PAGE_KIND_BM25_DELTA_TERM 14
#define PGTURBOHYBRID_GRAPH_PAGE_KIND_MULTIVECTOR_DOCMAP 15
#define PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK				0x00ff
#define PGTURBOHYBRID_GRAPH_PAGE_GRAPH_OP_SHIFT		8

#define PGTURBOHYBRID_GRAPH_GRAPH_OP_NONE				0
#define PGTURBOHYBRID_GRAPH_GRAPH_OP_PAGE_INIT			1
#define PGTURBOHYBRID_GRAPH_GRAPH_OP_PAGE_LINK			2
#define PGTURBOHYBRID_GRAPH_GRAPH_OP_META_UPDATE		3
#define PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT	4
#define PGTURBOHYBRID_GRAPH_GRAPH_OP_NEIGHBOR_INSERT	5
#define PGTURBOHYBRID_GRAPH_GRAPH_OP_NEIGHBOR_UPDATE	6
#define PGTURBOHYBRID_GRAPH_GRAPH_OP_DUPLICATE_HEAPTID	7
#define PGTURBOHYBRID_GRAPH_GRAPH_OP_VACUUM_DELETE		8
#define PGTURBOHYBRID_GRAPH_GRAPH_OP_VACUUM_REPAIR		9

/* Make graph robust against non-HOT updates */
#define PGTURBOHYBRID_GRAPH_HEAPTIDS 10

#define PGTURBOHYBRID_GRAPH_UPDATE_ENTRY_GREATER 1
#define PGTURBOHYBRID_GRAPH_UPDATE_ENTRY_ALWAYS 2

/* Build phases */
/* PROGRESS_CREATEIDX_SUBPHASE_INITIALIZE is 1 */
#define PGTURBOHYBRID_PROGRESS_PHASE_LOAD		2

#define PGTURBOHYBRID_GRAPH_MAX_SIZE (BLCKSZ - MAXALIGN(SizeOfPageHeaderData) - MAXALIGN(sizeof(PgturbohybridGraphPageOpaqueData)) - sizeof(ItemIdData))
#define PGTURBOHYBRID_GRAPH_TUPLE_ALLOC_SIZE BLCKSZ

#define PGTURBOHYBRID_GRAPH_ELEMENT_TUPLE_SIZE(size)	MAXALIGN(offsetof(PgturbohybridGraphElementTupleData, data) + (size))
#define PGTURBOHYBRID_GRAPH_NEIGHBOR_TUPLE_SIZE(level, m)	MAXALIGN(offsetof(PgturbohybridGraphNeighborTupleData, indextids) + ((level) + 2) * (m) * sizeof(ItemPointerData))

#define PGTURBOHYBRID_GRAPH_NEIGHBOR_ARRAY_SIZE(lm)	(offsetof(PgturbohybridGraphNeighborArray, items) + sizeof(PgturbohybridGraphCandidate) * (lm))

#define PgturbohybridGraphPageGetOpaque(page)	((PgturbohybridGraphPageOpaque) PageGetSpecialPointer(page))

static inline Size
PgturbohybridCheckedArrayBytes(Size elemSize, Size count, const char *what)
{
	if (elemSize != 0 && count > MaxAllocSize / elemSize)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("%s is too large", what)));
	return elemSize * count;
}
#define PgturbohybridGraphPageGetMeta(page)	((PgturbohybridGraphMetaPageData *) PageGetContents(page))

#if PG_VERSION_NUM >= 150000
#define RandomDouble() pg_prng_double(&pg_global_prng_state)
#define SeedRandom(seed) pg_prng_seed(&pg_global_prng_state, seed)
#else
#define RandomDouble() (((double) random()) / MAX_RANDOM_VALUE)
#define SeedRandom(seed) srandom(seed)
#endif

#define PgturbohybridGraphIsElementTuple(tup) ((tup)->type == PGTURBOHYBRID_GRAPH_ELEMENT_TUPLE_TYPE)
#define PgturbohybridGraphIsNeighborTuple(tup) ((tup)->type == PGTURBOHYBRID_GRAPH_NEIGHBOR_TUPLE_TYPE)

/* 2 * M connections for ground layer */
#define PgturbohybridGraphGetLayerM(m, layer) (layer == 0 ? (m) * 2 : (m))

/* Optimal ML from paper */
#define PgturbohybridGraphGetMl(m) (1 / log(m))

/* Ensure fits on page and in uint8 */
#define PgturbohybridGraphGetMaxLevel(m) Min(((BLCKSZ - MAXALIGN(SizeOfPageHeaderData) - MAXALIGN(sizeof(PgturbohybridGraphPageOpaqueData)) - offsetof(PgturbohybridGraphNeighborTupleData, indextids) - sizeof(ItemIdData)) / (sizeof(ItemPointerData)) / (m)) - 2, 255)

#define PgturbohybridGraphGetSearchCandidate(membername, ptr) pairingheap_container(PgturbohybridGraphSearchCandidate, membername, ptr)
#define PgturbohybridGraphGetSearchCandidateConst(membername, ptr) pairingheap_const_container(PgturbohybridGraphSearchCandidate, membername, ptr)

#define PgturbohybridGraphGetValue(base, element) PointerGetDatum(PgturbohybridGraphPtrAccess(base, (element)->value))

#if PG_VERSION_NUM < 140005
#define relptr_offset(rp) ((rp).relptr_off - 1)
#endif

/* Pointer macros */
#define PgturbohybridGraphPtrAccess(base, hp) ((base) == NULL ? (hp).ptr : relptr_access(base, (hp).relptr))
#define PgturbohybridGraphPtrStore(base, hp, value) ((base) == NULL ? (void) ((hp).ptr = (value)) : (void) relptr_store(base, (hp).relptr, value))
#define PgturbohybridGraphPtrIsNull(base, hp) ((base) == NULL ? (hp).ptr == NULL : relptr_is_null((hp).relptr))
#define PgturbohybridGraphPtrEqual(base, hp1, hp2) ((base) == NULL ? (hp1).ptr == (hp2).ptr : relptr_offset((hp1).relptr) == relptr_offset((hp2).relptr))

/* For code paths dedicated to each type */
#define PgturbohybridGraphPtrPointer(hp) (hp).ptr
#define PgturbohybridGraphPtrOffset(hp) relptr_offset((hp).relptr)

/* Variables */
extern int	pgturbohybrid_ef_search;
extern int	pgturbohybrid_iterative_scan;
extern int	pgturbohybrid_max_scan_tuples;
extern double pgturbohybrid_scan_mem_multiplier;
extern bool pgturbohybrid_dense_graph_prefetch;
extern bool pgturbohybrid_dense_graph_stack_scratch;
extern bool pgturbohybrid_dense_graph_lowbit_popcnt;
extern bool pgturbohybrid_dense_graph_i8mm;
extern bool pgturbohybrid_dense_graph_avxvnni;
extern bool pgturbohybrid_dense_graph_avx512vnni;
extern bool pgturbohybrid_dense_graph_avx512vpopcntdq;
extern bool pgturbohybrid_dense_weighted;
extern bool pgturbohybrid_dense_renorm;
extern bool pgturbohybrid_dense_query_1bit_asymmetric;
extern int	pgturbohybrid_dense_query_1bit_asymmetric_bits;
extern bool pgturbohybrid_dense_build_exact_distances;
extern bool pgturbohybrid_dense_build_exact_distances_user_set;
extern int	pgturbohybrid_dense_build_distance;
extern int	pgturbohybrid_dense_build_neighbor_select;
extern bool pgturbohybrid_dense_hadamard_simd;
extern int	pgturbohybrid_dense_simd_force;
extern int	pgturbohybrid_dense_query_split_impl;
extern int	pgturbohybrid_dense_u8_split;
extern bool pgturbohybrid_dense_u8_batch_x4;
extern int	pgturbohybrid_native_cache_policy;
extern int	pgturbohybrid_native_cache_max_mb;
extern int	pgturbohybrid_native_cache_warn_mb;
extern char *pgturbohybrid_native_build_workers;
extern int	pgturbohybrid_native_parallel_edge_build;
extern int	pgturbohybrid_native_segment_budget;
extern int	pgturbohybrid_dense_exact_simd_force;
extern int	pgturbohybrid_dense_graph_batch_scoring;
extern int	pgturbohybrid_dense_graph_batch_size;
extern int	pgturbohybrid_dense_graph_avx512_weighted;
extern int	pgturbohybrid_dense_graph_lookahead_prefetch;
extern int	pgturbohybrid_dense_graph_lookahead_threshold_kb;
extern int	pgturbohybrid_dense_budget_policy;
extern int	pgturbohybrid_dense_max_candidate_multiplier;
extern double pgturbohybrid_dense_latency_multiplier;
extern int	pgturbohybrid_dense_max_rescore_multiplier;
extern int	pgturbohybrid_dense_rescore_band_policy;
extern int	pgturbohybrid_dense_heap_rescore;
extern bool pgturbohybrid_dense_heap_rescore_user_set;
extern int	pgturbohybrid_dense_residual_rerank_mode;
extern double pgturbohybrid_dense_residual_rerank_weight;
extern double pgturbohybrid_dense_residual_rerank_max_adjust_ratio;
extern int pgturbohybrid_dense_residual_rerank_band_multiplier;
extern int	pgturbohybrid_dense_adaptive_widening;
extern double pgturbohybrid_dense_adaptive_widening_multiplier;
extern double pgturbohybrid_dense_adaptive_widening_max_multiplier;
extern double pgturbohybrid_dense_adaptive_min_gap;
extern int	pgturbohybrid_dense_uncertainty_retry;
extern int	pgturbohybrid_dense_uncertainty_retry_max_passes;
extern double pgturbohybrid_dense_uncertainty_retry_multiplier;
extern double pgturbohybrid_dense_uncertainty_min_gap;
extern bool pgturbohybrid_warn_linear_fallback;
extern double pgturbohybrid_linear_fallback_notice_threshold_ratio;
extern int	pgturbohybrid_dense_local_expansion;
extern int	pgturbohybrid_dense_local_expansion_topn;
extern int	pgturbohybrid_dense_local_expansion_max_neighbors;
extern int	pgturbohybrid_payload_entry_seeding;
extern int	pgturbohybrid_payload_entry_seed_count;
extern int	pgturbohybrid_final_diversity;
extern int	pgturbohybrid_final_diversity_payload_slot;
extern double pgturbohybrid_final_diversity_lambda;
extern int	pgturbohybrid_final_diversity_pool_multiplier;
extern int	pgturbohybrid_graph_lock_tranche_id;

typedef enum PgturbohybridRoutingMode
{
	PGTURBOHYBRID_ROUTING_AUTO = 0,
	PGTURBOHYBRID_ROUTING_GRAPH = 1,
	PGTURBOHYBRID_ROUTING_IVF = 2,
	PGTURBOHYBRID_ROUTING_FLAT = 3,
	PGTURBOHYBRID_ROUTING_LEGACY_GRAPH = 4
}			PgturbohybridRoutingMode;

typedef enum PgturbohybridGraphRescoreBand
{
	PGTURBOHYBRID_GRAPH_RESCORE_BAND_AUTO,
	PGTURBOHYBRID_GRAPH_RESCORE_BAND_NONE,
	PGTURBOHYBRID_GRAPH_RESCORE_BAND_EXACT
}			PgturbohybridGraphRescoreBand;

typedef enum PgturbohybridDenseBuildNeighborSelect
{
	PGTURBOHYBRID_DENSE_BUILD_NEIGHBOR_SELECT_FAST,
	PGTURBOHYBRID_DENSE_BUILD_NEIGHBOR_SELECT_HEURISTIC,
	PGTURBOHYBRID_DENSE_BUILD_NEIGHBOR_SELECT_AUTO
}			PgturbohybridDenseBuildNeighborSelect;

typedef enum PgturbohybridDenseBuildDistance
{
	PGTURBOHYBRID_DENSE_BUILD_DISTANCE_AUTO,
	PGTURBOHYBRID_DENSE_BUILD_DISTANCE_CODE,
	PGTURBOHYBRID_DENSE_BUILD_DISTANCE_EXACT
}			PgturbohybridDenseBuildDistance;

typedef enum PgturbohybridEntrySidecarStrategy
{
	PGTURBOHYBRID_ENTRY_SIDECAR_HASH = 0,
	PGTURBOHYBRID_ENTRY_SIDECAR_FARTHEST_CODE,
	PGTURBOHYBRID_ENTRY_SIDECAR_LEVEL_COVERING,
	PGTURBOHYBRID_ENTRY_SIDECAR_HYBRID_LEVEL_COVERING
}			PgturbohybridEntrySidecarStrategy;

typedef enum PgturbohybridPayloadEntrySeedingMode
{
	PGTURBOHYBRID_PAYLOAD_ENTRY_SEEDING_OFF,
	PGTURBOHYBRID_PAYLOAD_ENTRY_SEEDING_AUTO,
	PGTURBOHYBRID_PAYLOAD_ENTRY_SEEDING_ON
}			PgturbohybridPayloadEntrySeedingMode;

#define PGTURBOHYBRID_FINAL_DIVERSITY_DEFAULT_LAMBDA 0.75
#define PGTURBOHYBRID_FINAL_DIVERSITY_DEFAULT_POOL_MULTIPLIER 3
#define PGTURBOHYBRID_FINAL_DIVERSITY_MAX_POOL_MULTIPLIER 64

typedef enum PgturbohybridFinalDiversityMode
{
	PGTURBOHYBRID_FINAL_DIVERSITY_OFF,
	PGTURBOHYBRID_FINAL_DIVERSITY_GROUP_PAYLOAD
}			PgturbohybridFinalDiversityMode;

typedef enum PgturbohybridNativeSegmentBudgetMode
{
	PGTURBOHYBRID_NATIVE_SEGMENT_BUDGET_AUTO,
	PGTURBOHYBRID_NATIVE_SEGMENT_BUDGET_OFF,
	PGTURBOHYBRID_NATIVE_SEGMENT_BUDGET_SQRT,
	PGTURBOHYBRID_NATIVE_SEGMENT_BUDGET_LINEAR
}			PgturbohybridNativeSegmentBudgetMode;

typedef enum PgturbohybridNativeParallelEdgeBuildMode
{
	PGTURBOHYBRID_NATIVE_PARALLEL_EDGE_BUILD_AUTO,
	PGTURBOHYBRID_NATIVE_PARALLEL_EDGE_BUILD_OFF,
	PGTURBOHYBRID_NATIVE_PARALLEL_EDGE_BUILD_ON
}			PgturbohybridNativeParallelEdgeBuildMode;

typedef enum PgturbohybridBuildNeighborSelectReason
{
	PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_UNKNOWN = 0,
	PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_EXPLICIT_FAST,
	PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_EXPLICIT_HEURISTIC,
	PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_AUTO_LOWDIM,
	PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_AUTO_BALANCED,
	PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_AUTO_QUALITY,
	PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_AUTO_MATCHED_RECALL,
	PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_AUTO_LATENCY_HIGHDIM
}			PgturbohybridBuildNeighborSelectReason;

typedef enum PgturbohybridDenseBudgetPolicy
{
	PGTURBOHYBRID_DENSE_BUDGET_QUALITY,
	PGTURBOHYBRID_DENSE_BUDGET_BALANCED,
	PGTURBOHYBRID_DENSE_BUDGET_LATENCY,
	PGTURBOHYBRID_DENSE_BUDGET_AUTO
}			PgturbohybridDenseBudgetPolicy;

typedef enum TqRescoreBandPolicy
{
	PGTURBOHYBRID_RESCORE_BAND_POLICY_EXACT,
	PGTURBOHYBRID_RESCORE_BAND_POLICY_LIMITED,
	PGTURBOHYBRID_RESCORE_BAND_POLICY_AUTO,
	PGTURBOHYBRID_RESCORE_BAND_POLICY_OFF
}			TqRescoreBandPolicy;

typedef enum TqDenseHeapRescoreMode
{
	PGTURBOHYBRID_DENSE_HEAP_RESCORE_OFF,
	PGTURBOHYBRID_DENSE_HEAP_RESCORE_TOPK,
	PGTURBOHYBRID_DENSE_HEAP_RESCORE_BAND,
	PGTURBOHYBRID_DENSE_HEAP_RESCORE_AUTO
}			TqDenseHeapRescoreMode;

typedef enum TqDenseHeapRescoreReason
{
	PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_UNKNOWN,
	PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_EXPLICIT_OFF,
	PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_EXPLICIT_TOPK,
	PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_EXPLICIT_BAND,
	PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_LATENCY,
	PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_BALANCED_LOWDIM,
	PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_BALANCED_HIGHDIM,
	PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_QUALITY_LOWDIM,
	PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_QUALITY_HIGHDIM,
	PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_MATCHED_RECALL_LOWDIM,
	PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_MATCHED_RECALL_HIGHDIM,
	PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_HIGH_RECALL,
	PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_EXACT_STORAGE
}			TqDenseHeapRescoreReason;

typedef enum TqExactRescoreSource
{
	PGTURBOHYBRID_EXACT_RESCORE_SOURCE_NONE,
	PGTURBOHYBRID_EXACT_RESCORE_SOURCE_INDEX_EXACT,
	PGTURBOHYBRID_EXACT_RESCORE_SOURCE_HEAP,
	PGTURBOHYBRID_EXACT_RESCORE_SOURCE_RESIDUAL
}			TqExactRescoreSource;

typedef enum TqDenseResidualRerankMode
{
	PGTURBOHYBRID_DENSE_RESIDUAL_RERANK_OFF,
	PGTURBOHYBRID_DENSE_RESIDUAL_RERANK_FIXED,
	PGTURBOHYBRID_DENSE_RESIDUAL_RERANK_CALIBRATED
}			TqDenseResidualRerankMode;

typedef enum TqDenseWideningReason
{
	PGTURBOHYBRID_DENSE_WIDENING_NONE,
	PGTURBOHYBRID_DENSE_WIDENING_DIMENSION,
	PGTURBOHYBRID_DENSE_WIDENING_FILTER,
	PGTURBOHYBRID_DENSE_WIDENING_EXACT_POLICY
}			TqDenseWideningReason;

typedef enum TqDenseAdaptiveWideningMode
{
	PGTURBOHYBRID_DENSE_ADAPTIVE_WIDENING_OFF,
	PGTURBOHYBRID_DENSE_ADAPTIVE_WIDENING_AUTO,
	PGTURBOHYBRID_DENSE_ADAPTIVE_WIDENING_ON
}			TqDenseAdaptiveWideningMode;

typedef enum TqDenseAdaptiveWideningReason
{
	PGTURBOHYBRID_DENSE_ADAPTIVE_REASON_NONE,
	PGTURBOHYBRID_DENSE_ADAPTIVE_REASON_FORCED,
	PGTURBOHYBRID_DENSE_ADAPTIVE_REASON_UNDERFILLED,
	PGTURBOHYBRID_DENSE_ADAPTIVE_REASON_FLAT_TOP10,
	PGTURBOHYBRID_DENSE_ADAPTIVE_REASON_FLAT_BOUNDARY
}			TqDenseAdaptiveWideningReason;

typedef enum TqDenseUncertaintyRetryMode
{
	PGTURBOHYBRID_DENSE_UNCERTAINTY_RETRY_OFF,
	PGTURBOHYBRID_DENSE_UNCERTAINTY_RETRY_AUTO,
	PGTURBOHYBRID_DENSE_UNCERTAINTY_RETRY_ON
}			TqDenseUncertaintyRetryMode;

typedef enum TqDenseUncertaintyRetryReason
{
	PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_NONE,
	PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_FORCED,
	PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_UNDERFILLED,
	PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_FLAT_TOP10,
	PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_FLAT_BOUNDARY,
	PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_SIDECAR_UNUSED,
	PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_PAYLOAD_UNDERFILLED,
	PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_RESIDUAL_REORDERED,
	PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_HEAP_REORDERED
}			TqDenseUncertaintyRetryReason;

typedef enum TqDenseLocalExpansionMode
{
	PGTURBOHYBRID_DENSE_LOCAL_EXPANSION_OFF,
	PGTURBOHYBRID_DENSE_LOCAL_EXPANSION_AUTO,
	PGTURBOHYBRID_DENSE_LOCAL_EXPANSION_ON
}			TqDenseLocalExpansionMode;

typedef enum PgturbohybridGraphExactCache
{
	PGTURBOHYBRID_GRAPH_EXACT_CACHE_AUTO,
	PGTURBOHYBRID_GRAPH_EXACT_CACHE_OFF,
	PGTURBOHYBRID_GRAPH_EXACT_CACHE_ON
}			PgturbohybridGraphExactCache;

typedef enum PgturbohybridNativeCachePolicy
{
	PGTURBOHYBRID_NATIVE_CACHE_POLICY_AUTO = 0,
	PGTURBOHYBRID_NATIVE_CACHE_POLICY_PER_BACKEND,
	PGTURBOHYBRID_NATIVE_CACHE_POLICY_OFF,
	PGTURBOHYBRID_NATIVE_CACHE_POLICY_SHARED
}			PgturbohybridNativeCachePolicy;

typedef enum PgturbohybridGraphReorder
{
	PGTURBOHYBRID_GRAPH_REORDER_AUTO,
	PGTURBOHYBRID_GRAPH_REORDER_OFF,
	PGTURBOHYBRID_GRAPH_REORDER_BFS
}			PgturbohybridGraphReorder;

typedef enum PgturbohybridGraphLookaheadPrefetch
{
	PGTURBOHYBRID_GRAPH_LOOKAHEAD_AUTO,	/* size-gated: on iff metadata working-set > threshold */
	PGTURBOHYBRID_GRAPH_LOOKAHEAD_OFF,
	PGTURBOHYBRID_GRAPH_LOOKAHEAD_ON
}			PgturbohybridGraphLookaheadPrefetch;

typedef enum TqScoreMode
{
	PGTURBOHYBRID_SCORE_L2,
	PGTURBOHYBRID_SCORE_IP,
	PGTURBOHYBRID_SCORE_COSINE,
	PGTURBOHYBRID_SCORE_L1
}			TqScoreMode;

typedef enum TqScoringKernel
{
	PGTURBOHYBRID_SCORING_SCALAR,
	PGTURBOHYBRID_SCORING_AVX2,
	PGTURBOHYBRID_SCORING_AVX512VNNI,
	PGTURBOHYBRID_SCORING_AVXVNNI,
	PGTURBOHYBRID_SCORING_AVX512BW_DQ,
	PGTURBOHYBRID_SCORING_ARM_I8MM,
	PGTURBOHYBRID_SCORING_NEON
}			TqScoringKernel;

/*
 * Exact scoring-kernel paths inside PgturbohybridGraphScoreNodeBatch() and
 * PgturbohybridGraphScoreNode(), counted per native scan so the scan stats can
 * prove which kernel actually did the dense scoring work (and, when the slow
 * scalar/LUT path runs, make that obvious).  Order matters: it indexes the
 * counter arrays on the scan opaque and the name table in graph_utils.c.
 */
typedef enum PgturbohybridGraphScoreKernelBucket
{
	PGTURBOHYBRID_SCORE_KERNEL_BATCH_U8_SPLIT_AVX2,
	PGTURBOHYBRID_SCORE_KERNEL_BATCH_U8_SPLIT_AVX512VNNI,
	PGTURBOHYBRID_SCORE_KERNEL_BATCH_SIGNED_SPLIT_AVX2,
	PGTURBOHYBRID_SCORE_KERNEL_BATCH_SIGNED_SPLIT_AVXVNNI,
	PGTURBOHYBRID_SCORE_KERNEL_BATCH_SIGNED_SPLIT_AVX512VNNI,
	PGTURBOHYBRID_SCORE_KERNEL_BATCH_1BIT_ASYM,
	PGTURBOHYBRID_SCORE_KERNEL_BATCH_SCALAR_OR_LUT,
	PGTURBOHYBRID_SCORE_KERNEL_SINGLE_U8_SPLIT,
	PGTURBOHYBRID_SCORE_KERNEL_SINGLE_SIGNED_SPLIT,
	PGTURBOHYBRID_SCORE_KERNEL_SINGLE_SCALAR_OR_LUT,
	PGTURBOHYBRID_SCORE_KERNEL_BUCKET_COUNT
}			PgturbohybridGraphScoreKernelBucket;

/*
 * Query-split representation for 4-bit dense scoring.  SIGNED is the original
 * signed-codebook split (HIGH_COEF 256); UNSIGNED is the x86 unsigned-codebook
 * split (maddubs/VPDPBUSD, HIGH_COEF 128); AUTO picks UNSIGNED on x86 when the
 * required SIMD is available, otherwise SIGNED.
 */
typedef enum TqQuerySplitImpl
{
	PGTURBOHYBRID_QUERY_SPLIT_IMPL_AUTO,
	PGTURBOHYBRID_QUERY_SPLIT_IMPL_SIGNED,
	PGTURBOHYBRID_QUERY_SPLIT_IMPL_UNSIGNED
}			TqQuerySplitImpl;

/*
 * Dedicated control for the unsigned-codebook (u8) 4-bit split scorer, for
 * controlled benchmarking.  AUTO defers to dense_query_split_impl; ON forces
 * the u8 split whenever its hard requirements are met (bits == 4, dim >= 1024,
 * mode != L1, AVX2+ available, SIMD not forced scalar); OFF disables it so the
 * signed split (or scalar/LUT) runs instead.
 */
typedef enum TqU8Split
{
	PGTURBOHYBRID_U8_SPLIT_AUTO,
	PGTURBOHYBRID_U8_SPLIT_ON,
	PGTURBOHYBRID_U8_SPLIT_OFF
}			TqU8Split;

/*
 * The exact unsigned-codebook (u8) scoring kernel resolved once per query in
 * TqPrepareQueryU8Split (via PgturbohybridGraphTqResolveU8Kernels), so the hot
 * scoring path switches on a precomputed value instead of probing CPU features
 * per batch.  Stored in PgturbohybridGraphTqQuery.u8.kernelSingle / u8.kernelBatch.
 */
typedef enum TqU8Kernel
{
	PGTURBOHYBRID_U8_KERNEL_NONE,
	PGTURBOHYBRID_U8_KERNEL_AVX2_SINGLE,
	PGTURBOHYBRID_U8_KERNEL_AVX2_X4,
	PGTURBOHYBRID_U8_KERNEL_AVX512VNNI_SINGLE,
	PGTURBOHYBRID_U8_KERNEL_AVX512VNNI_X4
}			TqU8Kernel;

typedef enum TqSimdForce
{
	PGTURBOHYBRID_SIMD_FORCE_AUTO,
	PGTURBOHYBRID_SIMD_FORCE_SCALAR,
	PGTURBOHYBRID_SIMD_FORCE_AVX2,
	PGTURBOHYBRID_SIMD_FORCE_AVXVNNI,
	PGTURBOHYBRID_SIMD_FORCE_AVX512VNNI,
	PGTURBOHYBRID_SIMD_FORCE_NEON,
	PGTURBOHYBRID_SIMD_FORCE_ARM_SDOT,
	PGTURBOHYBRID_SIMD_FORCE_ARM_I8MM
}			TqSimdForce;

typedef enum TqExactSimdForce
{
	PGTURBOHYBRID_EXACT_SIMD_FORCE_AUTO,
	PGTURBOHYBRID_EXACT_SIMD_FORCE_SCALAR,
	PGTURBOHYBRID_EXACT_SIMD_FORCE_NEON
}			TqExactSimdForce;

typedef enum TqExactKernel
{
	PGTURBOHYBRID_EXACT_KERNEL_SCALAR,
	PGTURBOHYBRID_EXACT_KERNEL_AUTOVEC_FMA,
	PGTURBOHYBRID_EXACT_KERNEL_NEON,
	PGTURBOHYBRID_EXACT_KERNEL_AVX2
}			TqExactKernel;

typedef enum PgturbohybridGraphBatchScoringMode
{
	PGTURBOHYBRID_GRAPH_BATCH_AUTO,
	PGTURBOHYBRID_GRAPH_BATCH_OFF,
	PGTURBOHYBRID_GRAPH_BATCH_ON
}			PgturbohybridGraphBatchScoringMode;

typedef enum PgturbohybridGraphAvx512WeightedMode
{
	PGTURBOHYBRID_GRAPH_AVX512_WEIGHTED_OFF,
	PGTURBOHYBRID_GRAPH_AVX512_WEIGHTED_ON,
	PGTURBOHYBRID_GRAPH_AVX512_WEIGHTED_AUTO
}			PgturbohybridGraphAvx512WeightedMode;

typedef enum PgturbohybridGraphIterativeScanMode
{
	PGTURBOHYBRID_GRAPH_ITERATIVE_SCAN_OFF,
	PGTURBOHYBRID_GRAPH_ITERATIVE_SCAN_RELAXED,
	PGTURBOHYBRID_GRAPH_ITERATIVE_SCAN_STRICT
}			PgturbohybridGraphIterativeScanMode;

typedef struct PgturbohybridGraphElementData PgturbohybridGraphElementData;
typedef struct PgturbohybridGraphNeighborArray PgturbohybridGraphNeighborArray;

#define PgturbohybridGraphPtrDeclare(type, relptrtype, ptrtype) \
	relptr_declare(type, relptrtype); \
	typedef union { type *ptr; relptrtype relptr; } ptrtype

/* Pointers that can be absolute or relative */
/* Use char for DatumPtr so works with Pointer */
PgturbohybridGraphPtrDeclare(PgturbohybridGraphElementData, PgturbohybridGraphElementRelptr, PgturbohybridGraphElementPtr);
PgturbohybridGraphPtrDeclare(PgturbohybridGraphNeighborArray, PgturbohybridGraphNeighborArrayRelptr, PgturbohybridGraphNeighborArrayPtr);
PgturbohybridGraphPtrDeclare(PgturbohybridGraphNeighborArrayPtr, PgturbohybridGraphNeighborsRelptr, PgturbohybridGraphNeighborsPtr);
PgturbohybridGraphPtrDeclare(char, DatumRelptr, DatumPtr);

struct PgturbohybridGraphElementData
{
	PgturbohybridGraphElementPtr next;
	ItemPointerData heaptids[PGTURBOHYBRID_GRAPH_HEAPTIDS];
	uint8		heaptidsLength;
	uint8		level;
	uint8		deleted;
	uint8		version;
	uint32		hash;
	PgturbohybridGraphNeighborsPtr neighbors;
	BlockNumber blkno;
	OffsetNumber offno;
	OffsetNumber neighborOffno;
	BlockNumber neighborPage;
	DatumPtr	value;
	LWLock		lock;
};

typedef PgturbohybridGraphElementData * PgturbohybridGraphElement;

typedef struct PgturbohybridGraphCandidate
{
	PgturbohybridGraphElementPtr element;
	float		distance;
	bool		closer;
}			PgturbohybridGraphCandidate;

struct PgturbohybridGraphNeighborArray
{
	int			length;
	bool		closerSet;
	PgturbohybridGraphCandidate items[FLEXIBLE_ARRAY_MEMBER];
};

typedef struct PgturbohybridGraphSearchCandidate
{
	pairingheap_node c_node;
	pairingheap_node w_node;
	PgturbohybridGraphElementPtr element;
	double		distance;
}			PgturbohybridGraphSearchCandidate;

/* HNSW index options */
typedef struct PgturbohybridGraphOptions
{
	int32		vl_len_;		/* varlena header (do not touch directly!) */
	int			m;				/* number of connections */
	int			efConstruction; /* size of dynamic candidate list */
}			PgturbohybridGraphOptions;

/* Keep the first fields layout-compatible with PgturbohybridGraphOptions */
typedef struct TqOptions
{
	int32		vl_len_;		/* varlena header (do not touch directly!) */
	int			m;				/* graph_m */
	int			efConstruction; /* graph_ef_construction */
	int			routing;
	int			graphEfSearch;
	int			graphOversampling;
	int			graphRescoreBand;
	int			graphExactCache;
	int			graphReorder;
	int			nativeSegments;
	int			tqBits;
	bool		tqWeighted;		/* internal weighted quantized scoring flag */
	bool		tqQuantileFit;	/* internal quantile-anchored correction fit flag */
	bool		tqRenorm;		/* internal quantized renormalization residual flag */
	bool		tqExactStorage; /* exact_storage: store full exact vectors for final rescoring, or omit them for exact-free quantized-only storage. */
	bool		entrySidecar;	/* entry_sidecar: store data-aware representative node IDs in metadata. */
	int			entrySidecarRepresentatives;
	int			entrySidecarStrategy;
	bool		graphBackbone;	/* graph_backbone: force adjacent level-0 graph edges at build time. */
	bool		residualRerank; /* residual_rerank: store tiny per-vector sketches for final-band reranking. */
	int			residualRerankBytes;
	int			multivectorGraphMode;	/* multivector_graph: token_nodes or document_nodes */
}			TqOptions;

typedef struct PgturbohybridGraphGraph
{
	/* Graph state */
	slock_t		lock;
	PgturbohybridGraphElementPtr head;
	double		indtuples;

	/* Entry state */
	LWLock		entryLock;
	LWLock		entryWaitLock;
	PgturbohybridGraphElementPtr entryPoint;

	/* Allocations state */
	LWLock		allocatorLock;
	Size		memoryUsed;
	Size		memoryTotal;

	/* Flushed state */
	LWLock		flushLock;
	bool		flushed;
}			PgturbohybridGraphGraph;

typedef struct PgturbohybridGraphShared
{
	/* Immutable state */
	Oid			heaprelid;
	Oid			indexrelid;
	bool		isconcurrent;

	/* Worker progress */
	ConditionVariable workersdonecv;

	/* Mutex for mutable state */
	slock_t		mutex;

	/* Mutable state */
	int			nparticipantsdone;
	double		reltuples;
	PgturbohybridGraphGraph	graphData;
}			PgturbohybridGraphShared;

#define ParallelTableScanFromPgturbohybridGraphShared(shared) \
	(ParallelTableScanDesc) ((char *) (shared) + BUFFERALIGN(sizeof(PgturbohybridGraphShared)))

typedef struct PgturbohybridGraphLeader
{
	ParallelContext *pcxt;
	int			nparticipanttuplesorts;
	PgturbohybridGraphShared *graphShared;
	Snapshot	snapshot;
	char	   *graphArea;
}			PgturbohybridGraphLeader;

typedef struct PgturbohybridGraphAllocator
{
	void	   *(*alloc) (Size size, void *state);
	void	   *state;
}			PgturbohybridGraphAllocator;

typedef struct PgturbohybridGraphTypeInfo
{
	int			maxDimensions;
	Datum		(*normalize) (PG_FUNCTION_ARGS);
	void		(*checkValue) (Pointer v);
}			PgturbohybridGraphTypeInfo;

typedef struct PgturbohybridGraphSupport
{
	FmgrInfo   *procinfo;
	FmgrInfo   *normprocinfo;
	Oid			collation;
}			PgturbohybridGraphSupport;

typedef struct PgturbohybridGraphQuery
{
	Datum		value;
}			PgturbohybridGraphQuery;

/*
 * Per-query scoring representations, grouped so it is obvious which fields a
 * given path owns.  These are (largely) mutually exclusive: a query builds at
 * most one of the SIMD splits (signedSplit OR u8), and the LUT table is filled
 * only when no integer split will run.  All pointer members are palloc'd in the
 * query's MemoryContext by the matching TqPrepareQuery* routine and freed with
 * that context (ownership noted per group); the fixed [16]-arrays are inline.
 */

/*
 * LUT scoring: per-dim 2^bits codebook-distance table.
 *   populated by: TqBuildQueryLut, only when the integer query split will NOT
 *                 run for this query (PgturbohybridGraphTqQuerySplitActive false).
 *   consumed by:  the scalar / avx2_lut_gather scorers (TqCodeDistanceScalar).
 *   memory:       table is palloc'd; NULL (table stays unset) when a split runs.
 */
typedef struct TqLutQuery
{
	float	   *table;
	int			width;			/* 1 << bits */
}			TqLutQuery;

/*
 * Signed-codebook query split (AVX2 / AVX-VNNI / AVX-512 VNNI / NEON SDOT).
 *   populated by: TqPrepareQuerySplit4.
 *   consumed by:  PgturbohybridGraphQuerySplitRaw{,2}{Avx2,AvxVnni,Avx512Vnni,NeonSdot}.
 *   memory:       low/high/lowU8/highU8 are palloc'd; the tail[16] arrays inline.
 * low/high are the signed 7-bit halves; lowU8/highU8 are the +128 (XOR 0x80)
 * variants the VNNI path feeds to maddubs/vpdpbusd.  Chunk geometry is shared
 * with the u8 split (querySplitChunks / querySplitTailDims on the parent).
 */
typedef struct TqSignedSplitQuery
{
	int8	   *low;
	int8	   *high;
	uint8	   *lowU8;
	uint8	   *highU8;
	int8		tailLow[16];
	int8		tailHigh[16];
	uint8		tailLowU8[16];
	uint8		tailHighU8[16];
	float		postprocessScale;
	bool		enabled;
}			TqSignedSplitQuery;

/*
 * Unsigned-codebook query split (x86 maddubs / VPDPBUSD).  Separate from the
 * signed split: query halves are stored signed (no +128 XOR) because the
 * unsigned u8 codebook is the unsigned operand, and the +128 codebook shift is
 * unwound by bias.  data holds [low0..15, high0..15] per 16-dim chunk (32
 * bytes/chunk) so one AVX2 load is a [low|high] pair and one ZMM load is two.
 *   populated by: TqPrepareQueryU8Split (which also resolves kernelSingle/Batch).
 *   consumed by:  PgturbohybridGraphQuerySplitU8Raw* via the resolved kernels.
 *   memory:       data is palloc'd; the tail[16] arrays inline.  Chunk geometry
 *                 is shared (querySplitChunks / querySplitTailDims on the parent).
 */
typedef struct TqU8SplitQuery
{
	int8	   *data;
	int8		tailLow[16];
	int8		tailHigh[16];
	int64		bias;			/* OFFSET * Sum(q_signed); subtract from raw dot */
	float		postprocessScale;
	bool		enabled;
	/* Exact u8 kernels resolved once at query prep (TqU8Kernel values); the hot
	 * scoring path switches on these instead of probing CPU features per batch. */
	int			kernelSingle;
	int			kernelBatch;
}			TqU8SplitQuery;

/*
 * Asymmetric 1-bit query encoding: bit-plane-decomposed 8-bit signed query
 * quantization in 128-dim blocks of 8*16 bytes (8 planes of 16 bytes), which
 * recovers query magnitude the symmetric querySignBits path discards.
 *   populated by: TqPrepareQueryAsymBit1, only when the asymmetric-query path is
 *                 enabled and quantization_bits = 1.
 *   consumed by:  PgturbohybridGraphAsymBit1{Scalar,Avx2,Avx512Vpopcntdq,Neon}*.
 *   memory:       planes is palloc'd.
 */
typedef struct TqBit1Query
{
	uint8	   *planes;
	int64		sumSigned;		/* Σ q_signed over all dims (full + tail) */
	float		scale;			/* c / q_scale — postprocess multiplier  */
	int			numFullBlocks;	/* full 128-dim blocks                    */
	int			tailBytes;		/* 0..15: bytes in trailing partial block */
	int			bits;			/* BITS captured at precompute (8/12/16) */
}			TqBit1Query;

typedef struct PgturbohybridGraphTqQuery
{
	uint8	   *code;
#if PGTURBOHYBRID_GRAPH_ENABLE_SYMMETRIC_I8_DOT
	int8	   *queryI8;
#endif
	TqLutQuery	lut;
	float	   *queryValues;
	float	   *rawQueryValues;
	float	   *ecShift;
	float	   *ecScale;
	uint8	   *querySignBits;
	TqBit1Query bit1;
#if defined(__aarch64__) || defined(_M_ARM64) || \
	defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
	TqSignedSplitQuery signedSplit;
	TqU8SplitQuery u8;
	/* Chunk geometry shared by both SIMD splits (16-dim chunks + tail dims). */
	int			querySplitChunks;
	int			querySplitTailDims;
#endif
#if PGTURBOHYBRID_GRAPH_ENABLE_SYMMETRIC_I8_DOT
	float		queryScale;
	float		queryCodeNorm;
#endif
	int			dimensions;
	int			bits;
	Size		codeBytes;
	int			scoreMode;
	int			scoringKernel;
	double		queryNorm;
	double		ecCorrection;
	/*
	 * Precomputed query-side postprocess reciprocals (set in
	 * TqPrepareQueryU8Split), so the hot scoring loop multiplies instead of
	 * recomputing sqrt(dim) / sqrt(queryNorm) per node.
	 *   invDimSqrt          = 1 / sqrt(dim)
	 *   invQueryNormDimSqrt = 1 / (sqrt(queryNorm) * sqrt(dim))   [0 if queryNorm == 0]
	 */
	double		invDimSqrt;
	double		invQueryNormDimSqrt;
	bool		enabled;
}			PgturbohybridGraphTqQuery;

typedef struct PgturbohybridGraphBuildState
{
	/* Info */
	Relation	heap;
	Relation	index;
	IndexInfo  *indexInfo;
	ForkNumber	forkNum;
	const		PgturbohybridGraphTypeInfo *typeInfo;

	/* Settings */
	int			dimensions;
	int			m;
	int			efConstruction;

	/* Statistics */
	double		indtuples;
	double		reltuples;

	/* Support functions */
	PgturbohybridGraphSupport support;

	/* Variables */
	PgturbohybridGraphGraph	graphData;
	PgturbohybridGraphGraph  *graph;
	double		ml;
	int			maxLevel;

	/* Memory */
	MemoryContext graphCtx;
	MemoryContext tmpCtx;
	PgturbohybridGraphAllocator allocator;

	/* Parallel builds */
	PgturbohybridGraphLeader *graphLeader;
	PgturbohybridGraphShared *graphShared;
	char	   *graphArea;
}			PgturbohybridGraphBuildState;

typedef struct PgturbohybridGraphSegmentMetaData
{
	uint32		startNodeId;
	uint32		nodeCount;
	uint32		entryNodeId;
	int16		entryLevel;
	uint16		reserved;
	BlockNumber codeStartBlkno;
	BlockNumber adjStartBlkno;
	BlockNumber exactStartBlkno;
	BlockNumber correctionStartBlkno;
}			PgturbohybridGraphSegmentMetaData;

typedef struct PgturbohybridGraphMetaPageData
{
	uint32		magicNumber;
	uint32		version;
	uint32		dimensions;
	uint16		m;
	uint16		efConstruction;
	uint16		storageKind;
	uint16		graphEfSearch;
	uint16		graphOversampling;
	uint16		graphRescoreBand;
	uint16		graphMaxLevel;
	uint16		graphFlags;
	BlockNumber entryBlkno;
	OffsetNumber entryOffno;
	int16		entryLevel;
	BlockNumber insertPage;
	uint32		tqNodeCount;
	uint32		tqEntryNodeId;
	uint16		tqCodeBytes;
	uint16		tqPayloadCount;
	uint16		tqPayloadBytes;
	uint16		tqFlags;
	uint16		tqBits;
	BlockNumber tqCodeStartBlkno;
	BlockNumber tqAdjStartBlkno;
	BlockNumber tqExactStartBlkno;
	BlockNumber tqCorrectionStartBlkno;
	BlockNumber tqBm25MetaStartBlkno;
	BlockNumber tqMultivectorDocMapStartBlkno;
	uint32		tqMultivectorDocMapPageCount;
	uint32		tqMultivectorDocCount;
	uint32		tqMultivectorDocMapBytes;
	uint16		tqMultivectorDocMapVersion;
	uint16		tqMultivectorDocMapFlags;
	uint16		tqEntrySidecarCount;
	uint16		tqEntrySidecarBytes;
	uint16		tqResidualRerankBytes;
	uint16		tqMultivectorGraphMode;
	uint32		tqEntrySidecarNodeIds[PGTURBOHYBRID_GRAPH_MAX_ENTRY_SIDECAR_REPRESENTATIVES];
	uint16		tqRoutingEntryCount;
	uint16		tqRoutingEntryBytes;
	uint32		tqRoutingEntryNodeIds[PGTURBOHYBRID_GRAPH_MAX_ROUTING_ENTRIES];
	uint16		tqSegmentCount;
	uint16		tqSegmentBytes;
	PgturbohybridGraphSegmentMetaData tqSegments[PGTURBOHYBRID_GRAPH_MAX_NATIVE_SEGMENTS];
	uint64		buildScanUs;
	uint64		buildCorrectionUs;
	uint64		buildEncodeUs;
	uint64		buildEdgeUs;
	uint64		buildWriteUs;
	uint32		buildWorkerCount;
	uint32		buildReserved;
}			PgturbohybridGraphMetaPageData;

typedef PgturbohybridGraphMetaPageData * PgturbohybridGraphMetaPage;

typedef struct PgturbohybridGraphPageOpaqueData
{
	BlockNumber nextblkno;
	uint16		pageKind;
	uint16		page_id;		/* for identification of HNSW indexes */
}			PgturbohybridGraphPageOpaqueData;

typedef PgturbohybridGraphPageOpaqueData * PgturbohybridGraphPageOpaque;

typedef struct PgturbohybridGraphElementTupleData
{
	uint8		type;
	uint8		level;
	uint8		deleted;
	uint8		version;
	ItemPointerData heaptids[PGTURBOHYBRID_GRAPH_HEAPTIDS];
	ItemPointerData neighbortid;
	uint16		unused;
	Vector		data;
}			PgturbohybridGraphElementTupleData;

typedef PgturbohybridGraphElementTupleData * PgturbohybridGraphElementTuple;

typedef struct PgturbohybridGraphNeighborTupleData
{
	uint8		type;
	uint8		version;
	uint16		count;
	ItemPointerData indextids[FLEXIBLE_ARRAY_MEMBER];
}			PgturbohybridGraphNeighborTupleData;

typedef PgturbohybridGraphNeighborTupleData * PgturbohybridGraphNeighborTuple;

typedef union
{
	struct pointerhash_hash *pointers;
	struct offsethash_hash *offsets;
	struct tidhash_hash *tids;
}			visited_hash;

typedef union
{
	PgturbohybridGraphElement element;
	ItemPointerData indextid;
}			PgturbohybridGraphUnvisited;

/*
 * Which path the unsigned-codebook batch-of-4 scorer
 * (PgturbohybridGraphScoreNodeBatchU8Split) took during a scan, surfaced in
 * turbohybrid_last_scan_stats() as graph_u8_batch_mode.  NONE means the u8 batch
 * scorer never scored a batch (e.g. scalar/LUT fallback, or a non-u8 build).
 */
typedef enum PgturbohybridGraphU8BatchMode
{
	PGTURBOHYBRID_U8_BATCH_NONE = 0,
	PGTURBOHYBRID_U8_BATCH_X4,
	PGTURBOHYBRID_U8_BATCH_SINGLE
}			PgturbohybridGraphU8BatchMode;

/*
 * How the native dense scan's per-backend code/adjacency cache was satisfied
 * for this scan, surfaced as native_cache_mode in turbohybrid_last_scan_stats().
 * The native cache is process-local (one copy per backend), so under N
 * concurrent clients the per_backend footprint is duplicated N times -- this
 * mode is the first thing to check when explaining concurrent-client collapse.
 *   none        non-graph scan (flat storage / no native quantized graph)
 *   uncached    index exceeds turbohybrid.native_cache_max_mb, so the scan
 *               storage is rebuilt and code pages reloaded every scan with no
 *               cross-scan reuse (cold page loading dominates each query)
 *   per_backend index fits the cap, so a process-local cache is built once and
 *               reused by every later scan in this backend (warm scans read 0
 *               code pages but each backend holds its own full copy)
 */
typedef enum PgturbohybridGraphNativeCacheMode
{
	PGTURBOHYBRID_GRAPH_NATIVE_CACHE_NONE = 0,
	PGTURBOHYBRID_GRAPH_NATIVE_CACHE_UNCACHED,
	PGTURBOHYBRID_GRAPH_NATIVE_CACHE_PER_BACKEND,
	PGTURBOHYBRID_GRAPH_NATIVE_CACHE_SHARED
}			PgturbohybridGraphNativeCacheMode;

typedef enum PgturbohybridGraphNativeCacheReason
{
	PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_NONE = 0,
	PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_AUTO_FITS_MAX_MB,
	PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_PER_BACKEND_FITS_MAX_MB,
	PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_SHARED_FITS_MAX_MB,
	PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_POLICY_OFF,
	PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_EXCEEDS_MAX_MB,
	PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_SHARED_BUILD_TIMEOUT,
	PGTURBOHYBRID_GRAPH_NATIVE_CACHE_REASON_SHARED_ATTACH_FAILED
}			PgturbohybridGraphNativeCacheReason;

typedef enum PgturbohybridGraphFillCandidateBandReason
{
	PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_NONE = 0,
	PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_UNDERFILLED_FULL_TARGET,
	PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_ESTIMATED_SELECTIVITY,
	PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_ADAPTIVE_WIDENING,
	PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_TIGHT_L2_EXACT_POLICY,
	PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_PAYLOAD_EXACT_BAND_MISS,
	PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_UNKNOWN
}			PgturbohybridGraphFillCandidateBandReason;

typedef struct PgturbohybridGraphScanOpaqueData
{
	const		PgturbohybridGraphTypeInfo *typeInfo;
	bool		first;
	List	   *w;
	visited_hash v;
	pairingheap *discarded;
	PgturbohybridGraphQuery	q;
	PgturbohybridGraphTqQuery tq;
	int			m;
	int			efSearch;
	int			graphM;
	int			graphEfConstruction;
	int			graphOversampling;
	int			graphRescoreBand;
	int			graphExactCache;
	int64		tuples;
	int64		returnedRows;
	int64		tupleTargetRows;
	int64		graphFinalK;
	double		estimatedFilterSelectivity;
	int			initialEffectiveEfSearch;
	bool		hasTupleTargetRows;
	bool		hasEstimatedFilterSelectivity;
	bool		hasInitialEffectiveEfSearch;
	double		previousDistance;
	Size		maxMemory;
	MemoryContext tmpCtx;
	int64		graphVisitedNodes;
	int64		graphScoredCodes;
	int64		graphBatchScoredCodes;
	int64		graphScalarScoredCodes;
	int			graphBatchKernel;
	/* Path taken by the u8 batch-of-4 scorer (x4 kernel vs four single passes). */
	PgturbohybridGraphU8BatchMode graphU8BatchMode;
	/* Per-kernel scoring attribution (indexed by PgturbohybridGraphScoreKernelBucket). */
	int64		graphScoreKernelNodes[PGTURBOHYBRID_SCORE_KERNEL_BUCKET_COUNT];
	int64		graphScoreKernelCalls[PGTURBOHYBRID_SCORE_KERNEL_BUCKET_COUNT];
	/* PgturbohybridGraphScoreNodeBatch() invocations and total nodes fed to them. */
	int64		graphBatchCalls;
	int64		graphBatchNodes;
	/* Base-layer traversal counters (PgturbohybridGraphSearchBaseLayer). */
	int64		graphBaseFrontierPushes;
	int64		graphBaseFrontierPops;
	int64		graphBaseNearestOffers;
	int64		graphBaseVisitedChecks;
	int64		graphBaseDuplicateSkips;
	int64		graphBaseBatchCalls;
	int64		graphBaseBatchNodes;
	int64		graphBaseMaxFrontier;
	/* True when the code arena exceeds CPU cache (scoring is RAM-bound) -> the
	 * batch scorer prefetches whole codes, not just their first cache line. */
	bool		graphLargeCodeArena;
	/* Code-page cache effectiveness (PgturbohybridGraphLoadCodePage). */
	int64		graphCodePageAttempts;
	int64		graphCodePageHits;
	int64		graphCodePageMisses;
	int64		graphCodeTuplesCopied;
	int64		graphCodeArenaAllocatedBytes;
	int64		graphCodeArenaUsedBytes;
	/* Estimated full code working set (tqNodeCount * tqCodeBytes) used to decide
	 * graphLargeCodeArena.  Per-code width is so->tq.codeBytes. */
	int64		graphCodeArenaEstimatedBytes;
	/*
	 * Per-backend native scan-cache provenance for this scan (diagnostics for
	 * concurrent-client scaling).  graphNativeCacheMode says which mode served
	 * the scan; graphNativeCacheBuiltThisScan/BuildUs isolate the one-time cold
	 * per-backend build cost (the prewarm A/B signal); the *Bytes fields are the
	 * resident footprint each backend holds (duplicated per concurrent client).
	 */
	PgturbohybridGraphNativeCacheMode graphNativeCacheMode;
	int			graphNativeCachePolicy;
	int			graphNativeCacheReason;
	bool		graphNativeCacheUsed;
	bool		graphNativeCacheReused;
	bool		graphNativeCacheBuiltThisScan;
	int64		graphNativeCacheAttachUs;
	int64		graphNativeCacheBuildUs;
	int64		graphNativeCacheWaitUs;
	int64		graphNativeCacheRefcount;
	int64		graphNativeCacheBytes;
	int64		graphNativeCacheCodeBytes;
	int64		graphNativeCacheAdjBytes;
	int64		graphNativeCacheExactBytes;
	bool		graphNativeCacheWarning;
	const char *graphNativeCacheWarningReason;
	int64		graphScanLockWaitUs;
	int64		graphCodeBufferLockWaitUs;
	int64		graphAdjBufferLockWaitUs;
	int64		graphCandidateCount;
	int64		graphRescoreCount;
	int64		graphRescorePages;
	int64		graphCodePagesRead;
	int64		graphAdjPagesRead;
	int64		graphSegmentCount;
	int64		graphSegmentsSearched;
	int			graphPerSegmentBudgetMode;
	int64		graphSearchEfBeforeSegmentScaling;
	int64		graphSearchEfAfterSegmentScaling;
	int64		graphEntryPointCount;
	int64		graphEntrySidecarCount;
	int64		graphEntrySidecarScored;
	int64		graphEntrySidecarSelected;
	int64		graphEntrySidecarRepresentativesConfigured;
	int			graphEntrySidecarStrategy;
	int64		graphEntrySidecarUs;
	int			graphPayloadEntrySeedingMode;
	bool		graphPayloadEntrySeedingHit;
	int64		graphPayloadEntrySeedCount;
	int			graphPayloadEntrySeedPayloadSlot;
	int64		graphPayloadEntrySeedRangeCount;
	int64		graphPayloadEntrySeedUs;
	int64		graphPrepareUs;
	int64		graphTraverseUs;
	int64		graphEntryUs;
	int64		graphBaseUs;
	int64		graphBatchUs;
	int64		graphHeapUs;
	int64		graphFillUs;
	/*
	 * Fill-candidate-band fallback counters.  This fallback can linearly
	 * inspect every native graph node (or every payload ref) when traversal
	 * underfills, so keep the O(N) work visible in last-scan stats.
	 */
	int64		graphFillCandidateBandCalls;
	int			graphFillCandidateBandReason;
	int64		graphFillCandidateBandVisited;
	int64		graphFillCandidateBandScored;
	int64		graphFillCandidateBandSelectedBefore;
	int64		graphFillCandidateBandSelectedAfter;
	int64		graphFillCandidateBandTarget;
	bool		graphFillCandidateBandUsedPayloadRefs;
	int64		graphFillCandidateBandPayloadRefCount;
	int64		graphRescoreUs;
	int64		graphSortUs;
	int64		graphTotalUs;
	int64		graphDenseRequestedK;
	int64		graphEffectiveResultTarget;
	int64		graphEffectiveSearchEf;
	int64		graphEffectiveRescoreBand;
	double		graphHighdimWideningMultiplier;
	int			graphWideningReason;
	bool		graphDenseFilterUnmapped;
	bool		graphDenseLinearFallbackWarning;
	double		graphDenseLinearFallbackRatio;
	int			graphAdaptiveWideningMode;
	bool		graphAdaptiveTriggered;
	int			graphAdaptiveTriggerReason;
	int64		graphAdaptiveInitialResultTarget;
	int64		graphAdaptiveFinalResultTarget;
	int64		graphAdaptiveInitialSearchEf;
	int64		graphAdaptiveFinalSearchEf;
	double		graphAdaptiveGapTop10;
	double		graphAdaptiveGapBoundary;
	int			graphUncertaintyRetryMode;
	bool		graphUncertaintyRetryTriggered;
	int			graphUncertaintyRetryReason;
	int64		graphUncertaintyRetryPasses;
	int64		graphUncertaintyInitialResultTarget;
	int64		graphUncertaintyFinalResultTarget;
	int64		graphUncertaintyInitialSearchEf;
	int64		graphUncertaintyFinalSearchEf;
	double		graphUncertaintyGapTop10;
	double		graphUncertaintyGapBoundary;
	int			graphLocalExpansionMode;
	bool		graphLocalExpansionTriggered;
	int64		graphLocalExpansionSeedCount;
	int64		graphLocalExpansionNeighborsScored;
	int64		graphLocalExpansionCandidatesAdded;
	int64		graphLocalExpansionUs;
	int64		graphResidualRerankCount;
	int64		graphResidualRerankBytes;
	int64		graphResidualRerankUs;
	int			graphResidualRerankMode;
	double		graphResidualRerankWeightEffective;
	int64		graphResidualRerankBand;
	int			graphResidualRerankBandMultiplier;
	double		graphResidualRerankMaxAdjustment;
	int64		graphResidualRerankReorderedCount;
	bool		graphResidualRerankTopKChanged;
	int64		graphHeapRescoreCount;
	int64		graphHeapFetchUs;
	int64		graphHeapRescoreUs;
	int			graphHeapRescoreMode;
	int			graphHeapRescoreReason;
	bool		graphHeapRescoreAutoEnabled;
	int			graphExactRescoreSource;
	int			graphDenseBudgetPolicy;
	int			graphRescoreBandPolicy;
	int			graphStorageKind;
	bool		graphExactStorage;
	bool		graphBuildExactDistances;
	int			graphBuildDistanceMode;
	bool		graphBuildFastEdges;
	int			graphBuildNeighborSelectReason;
	bool		pgturbohybridGraphScan;
	bool		pgturbohybridFlatScan;
	void	   *tqGraphResults;
	int			tqGraphResultCount;
	int			tqGraphResultIndex;
	void	   *tqHybridState;

	/* Support functions */
	PgturbohybridGraphSupport support;
}			PgturbohybridGraphScanOpaqueData;

typedef PgturbohybridGraphScanOpaqueData * PgturbohybridGraphScanOpaque;

typedef struct PgturbohybridGraphVacuumState
{
	/* Info */
	Relation	index;
	IndexBulkDeleteResult *stats;
	IndexBulkDeleteCallback callback;
	void	   *callback_state;

	/* Settings */
	int			m;
	int			efConstruction;

	/* Support functions */
	PgturbohybridGraphSupport support;

	/* Variables */
	struct tidhash_hash *deleted;
	BufferAccessStrategy bas;
	PgturbohybridGraphNeighborTuple ntup;
	PgturbohybridGraphElementData highestPoint;

	/* Memory */
	MemoryContext tmpCtx;
}			PgturbohybridGraphVacuumState;

/* Methods */
int			PgturbohybridGraphGetM(Relation index);
int			PgturbohybridGraphGetEfConstruction(Relation index);
int			PgturbohybridGraphGetEfSearch(Relation index);
int			PgturbohybridGraphGetGraphOversampling(Relation index);
int			PgturbohybridGraphGetGraphRescoreBand(Relation index);
int			PgturbohybridGraphGetGraphExactCache(Relation index);
int			PgturbohybridGraphGetGraphReorder(Relation index);
int			PgturbohybridGraphGetNativeSegments(Relation index);
const char *PgturbohybridDenseBuildNeighborSelectName(int mode);
const char *PgturbohybridDenseBuildDistanceName(int mode);
const char *PgturbohybridEntrySidecarStrategyName(int strategy);
const char *PgturbohybridPayloadEntrySeedingName(int mode);
const char *PgturbohybridFinalDiversityName(int mode);
const char *PgturbohybridNativeSegmentBudgetName(int mode);
const char *PgturbohybridBuildNeighborSelectReasonName(int reason);
const char *PgturbohybridGraphDenseHeapRescoreName(int mode);
const char *PgturbohybridGraphDenseHeapRescoreReasonName(int reason);
const char *PgturbohybridGraphDenseUncertaintyRetryModeName(int mode);
const char *PgturbohybridGraphDenseUncertaintyRetryReasonName(int reason);
const char *PgturbohybridGraphExactRescoreSourceName(int source);
bool		PgturbohybridGraphIspgturbohybridIndex(Relation index);
bool		PgturbohybridGraphUseTqGraph(Relation index);
bool		PgturbohybridGraphUseTqNativeGraph(Relation index);
bool		PgturbohybridGraphUseTqFlat(Relation index);
bool		PgturbohybridGraphUseTqCodes(Relation index);
void		PgturbohybridGraphSetForcepgturbohybridIndex(bool force);
Size		PgturbohybridGraphElementTupleSize(Relation index, Pointer value);
FmgrInfo   *PgturbohybridGraphOptionalProcInfo(Relation index, uint16 procnum);
void		PgturbohybridGraphInitSupport(PgturbohybridGraphSupport * support, Relation index);
Datum		PgturbohybridGraphNormValue(const PgturbohybridGraphTypeInfo * typeInfo, Oid collation, Datum value);
bool		PgturbohybridGraphCheckNorm(PgturbohybridGraphSupport * support, Datum value);
Buffer		PgturbohybridGraphNewBuffer(Relation index, ForkNumber forkNum);
void		PgturbohybridGraphInitPage(Buffer buf, Page page);
void		PgturbohybridGraphInitPageKind(Buffer buf, Page page, uint16 pageKind);
void		PgturbohybridGraphMarkPageGraphOp(Page page, uint16 graphOpKind);
void		PgturbohybridGraphInit(void);
void		PgturbohybridGraphControlInit(void);
void		PgturbohybridGraphLogGraphWalRecord(Relation index, ForkNumber forkNum, BlockNumber blkno, uint16 graphOpKind);
const char *PgturbohybridGraphGraphWalModeName(void);
void		PgturbohybridGraphRecordGraphScanStats(PgturbohybridGraphScanOpaque so);
void		PgturbohybridGraphRecordReturnedRows(int64 returnedRows);
void		PgturbohybridGraphRecordNonGraphScanStats(void);
void		PgturbohybridGraphRecordFlatScanStats(void);

typedef struct PgturbohybridNativeBuildStatsSnapshot
{
	Oid			relid;
	char		relationName[NAMEDATALEN];
	char		indexShape[16];
	uint64		nodeCount;
	uint32		dimensions;
	int			quantizationBits;
	int			m;
	int			efConstruction;
	bool		exactStorage;
	bool		buildCodeOnly;
	bool		buildFastEdges;
	int			buildDistanceMode;
	int			buildNeighborSelectReason;
	uint64		buildDistanceCalls;
	uint64		buildDistanceQuerySplit;
	uint64		buildDistancePacked;
	uint64		buildDistanceWeighted;
	uint64		buildDistanceCodeCode;
	uint64		buildDistanceExact;
	uint64		buildDistanceFallback;
	uint64		buildEdgeDistanceCalls;
	uint64		buildEdgeSearchLayerUs;
	uint64		buildEdgeSelectNeighborUs;
	uint64		buildEdgeAddNeighborUs;
	uint64		buildEdgePruneNeighborUs;
	uint64		buildEdgeEntryUpdateUs;
	uint64		buildEdgeNearestTotal;
	uint64		buildEdgeNearestSamples;
	uint32		buildEdgeMaxFrontierSize;
	uint64		fitCorrectionScanUs;
	uint64		scanUs;
	uint64		fitCorrectionUs;
	uint64		encodeUs;
	uint64		buildEdgesUs;
	uint64		freeExactVectorsUs;
	uint64		reorderNodesUs;
	uint64		connectBackboneUs;
	uint64		entrySidecarUs;
	uint64		writePagesUs;
	uint64		walUs;
	uint64		totalUs;
	uint32		workerCount;
	uint32		nativeSegmentCount;
	uint32		nativeSegmentBytes;
	bool		parallelSegmentBuildEnabled;
	char		segmentBuildMode[16];
	uint32		nativeBuildWorkersRequested;
	uint32		nativeBuildWorkersLaunched;
	bool		parallelFitEnabled;
	bool		parallelScanEnabled;
	bool		parallelEncodeEnabled;
	bool		parallelEdgeBuildEnabled;
	uint32		parallelEdgeSegments;
	uint32		parallelEdgeWorkersLaunched;
	uint64		parallelEdgeRepairUs;
	uint64		workerMergeUs;
	uint32		workerScanUsCount;
	uint64		workerScanUs[PGTURBOHYBRID_NATIVE_BUILD_STATS_MAX_WORKERS];
} PgturbohybridNativeBuildStatsSnapshot;

void		PgturbohybridGraphRecordNativeBuildStats(const PgturbohybridNativeBuildStatsSnapshot *stats);
const char *PgturbohybridGraphTqScoringKernelName(int scoringKernel);
const char *PgturbohybridGraphTqScoreModeName(int scoreMode);
const char *PgturbohybridGraphRescoreBandName(int band);
const char *PgturbohybridGraphScoreKernelBucketName(int bucket);
const char *PgturbohybridGraphTqSimdForceName(int force);
const char *PgturbohybridGraphTqExactSimdForceName(int force);
const char *PgturbohybridGraphStorageKindName(int storageKind);
const char *PgturbohybridMultiVectorGraphModeName(int mode);
const char *PgturbohybridGraphDenseResidualRerankModeName(int mode);
int			PgturbohybridGraphGetTqBits(Relation index);
bool		PgturbohybridGraphGetTqWeightedOption(Relation index);
bool		PgturbohybridGraphGetTqRenormOption(Relation index);
bool		PgturbohybridGraphGetTqQuantileFitOption(Relation index);
bool		PgturbohybridGraphGetTqExactStorageOption(Relation index);
bool		PgturbohybridGraphGetEntrySidecarOption(Relation index);
int			PgturbohybridGraphGetEntrySidecarRepresentatives(Relation index);
int			PgturbohybridGraphGetEntrySidecarStrategy(Relation index);
bool		PgturbohybridGraphGetBackboneOption(Relation index);
bool		PgturbohybridGraphGetResidualRerankOption(Relation index);
int			PgturbohybridGraphGetResidualRerankBytes(Relation index);
int			PgturbohybridGraphGetMultiVectorGraphModeOption(Relation index);
void		PgturbohybridGraphPrepareTqQuery(Relation index, PgturbohybridGraphSupport * support, Datum value, PgturbohybridGraphTqQuery * tq);
void		PgturbohybridGraphPrepareTqQueryWithBits(Relation index, PgturbohybridGraphSupport * support, Datum value, PgturbohybridGraphTqQuery * tq, int tqBits);
void		PgturbohybridGraphPrepareTqBuildQuery(Relation index, PgturbohybridGraphSupport * support, Datum value, PgturbohybridGraphTqQuery * tq);
void		PgturbohybridGraphPrepareTqBuildQueryWithCorrection(Relation index, PgturbohybridGraphSupport * support, Datum value, PgturbohybridGraphTqQuery * tq,
												  const float *ecShift, const float *ecScale);
float		TqPreprocessVector(Vector *vector, double *rotated);
Size		TqCodeSizeForBits(int dimensions, int bits);
int			TqGetCodeComponentBits(const uint8 *code, int i, int bits);
float		TqGetCodeCenterBits(int code, int bits);
float		TqEncodeVectorBits(Vector *vector, uint8 *code, int bits);
float		TqEncodeVectorWithCorrectionBits(Vector *vector, uint8 *code, int bits,
											const float *ecShift, const float *ecScale);
float		TqEncodeVectorWithCorrectionAndXmBits(Vector *vector, uint8 *code, int bits,
											   const float *ecShift, const float *ecScale,
											   float *xmOut);
float		TqEncodeVectorWithCorrectionXmRenormBits(Vector *vector, uint8 *code, int bits,
												  const float *ecShift, const float *ecScale,
												  float *xmOut, float *centroidNormOut);
float		TqEncodeVector(Vector *vector, uint8 *code);
float		TqEncodeVectorWithCorrection(Vector *vector, uint8 *code,
										const float *ecShift, const float *ecScale);
double		TqCodeDistance(const PgturbohybridGraphTqQuery *tq, const uint8 *valueCode, float valueScale);
bool		PgturbohybridGraphTqCodeQuerySplitDistance(const PgturbohybridGraphTqQuery *tq, const uint8 *valueCode, float valueScale, double *distance);
bool		PgturbohybridGraphTqQuerySplitActive(const PgturbohybridGraphTqQuery *tq);
bool		PgturbohybridGraphTqUseU8Split(const PgturbohybridGraphTqQuery *tq);
void		PgturbohybridGraphTqResolveU8Kernels(PgturbohybridGraphTqQuery *tq);
bool		PgturbohybridGraphTqCodeSignedSplitDistance(const PgturbohybridGraphTqQuery *tq, const uint8 *valueCode, float valueScale, double *distance);
bool		PgturbohybridGraphTqCodeU8SimdDistance(const PgturbohybridGraphTqQuery *tq, const uint8 *valueCode, float valueScale, double *distance);
bool		PgturbohybridGraphTqCodeU8Simdx4Distance(const PgturbohybridGraphTqQuery *tq, const uint8 *valueCode, float valueScale, double *distance);
bool		PgturbohybridGraphTqCodeU8Simdx4Batch(const PgturbohybridGraphTqQuery *tq, const uint8 *codes[4], const float scales[4], double dist[4]);
bool		PgturbohybridGraphTqCodeU8ScalarDistance(const PgturbohybridGraphTqQuery *tq, const uint8 *valueCode, float valueScale, double *distance);
const char *PgturbohybridGraphU8SplitKernelName(void);
int64		PgturbohybridGraphRescoreSearchCandidates(Relation index, PgturbohybridGraphSupport * support, PgturbohybridGraphQuery * q, List *items);
List	   *PgturbohybridGraphSearchLayer(char *base, PgturbohybridGraphQuery * q, List *ep, int ef, int lc, Relation index, PgturbohybridGraphSupport * support, int m, bool inserting, PgturbohybridGraphElement skipElement, visited_hash * v, pairingheap **discarded, bool initVisited, int64 *tuples, int64 tupleLimit, int64 *scoredCodes, PgturbohybridGraphTqQuery * tq);
PgturbohybridGraphElement PgturbohybridGraphGetEntryPoint(Relation index);
void		PgturbohybridGraphGetMetaPageInfo(Relation index, int *m, PgturbohybridGraphElement * entryPoint);
int			PgturbohybridGraphGetMetaPageStorageKind(Relation index);
void	   *PgturbohybridGraphAlloc(PgturbohybridGraphAllocator * allocator, Size size);
PgturbohybridGraphElement PgturbohybridGraphInitElement(char *base, ItemPointer tid, int m, double ml, int maxLevel, PgturbohybridGraphAllocator * alloc);
PgturbohybridGraphElement PgturbohybridGraphInitElementFromBlock(BlockNumber blkno, OffsetNumber offno);
void		PgturbohybridGraphFindElementNeighbors(char *base, PgturbohybridGraphElement element, PgturbohybridGraphElement entryPoint, Relation index, PgturbohybridGraphSupport * support, int m, int efConstruction, bool existing);
PgturbohybridGraphSearchCandidate *PgturbohybridGraphEntryCandidate(char *base, PgturbohybridGraphElement entryPoint, PgturbohybridGraphQuery * q, Relation index, PgturbohybridGraphSupport * support, bool loadVec);
void		PgturbohybridGraphUpdateMetaPage(Relation index, int updateEntry, PgturbohybridGraphElement entryPoint, BlockNumber insertPage, ForkNumber forkNum, bool building);
void		PgturbohybridGraphSetNeighborTuple(char *base, PgturbohybridGraphNeighborTuple ntup, PgturbohybridGraphElement e, int m);
void		PgturbohybridGraphAddHeapTid(PgturbohybridGraphElement element, ItemPointer heaptid);
PgturbohybridGraphNeighborArray *PgturbohybridGraphInitNeighborArray(int lm, PgturbohybridGraphAllocator * allocator);
void		PgturbohybridGraphInitNeighbors(char *base, PgturbohybridGraphElement element, int m, PgturbohybridGraphAllocator * alloc);
bool		PgturbohybridGraphInsertTupleOnDisk(Relation index, PgturbohybridGraphSupport * support, Datum value, ItemPointer heaptid, bool building);
void		PgturbohybridGraphUpdateNeighborsOnDisk(Relation index, PgturbohybridGraphSupport * support, PgturbohybridGraphElement e, int m, bool checkExisting, bool building);
void		PgturbohybridGraphLoadElementFromTuple(PgturbohybridGraphElement element, PgturbohybridGraphElementTuple etup, bool loadHeaptids, bool loadVec);
void		PgturbohybridGraphLoadElement(PgturbohybridGraphElement element, double *distance, PgturbohybridGraphQuery * q, Relation index, PgturbohybridGraphSupport * support, bool loadVec, double *maxDistance);
bool		PgturbohybridGraphFormIndexValue(Datum *out, Datum *values, bool *isnull, const PgturbohybridGraphTypeInfo * typeInfo, PgturbohybridGraphSupport * support);
void		PgturbohybridGraphSetElementTuple(Relation index, char *base, PgturbohybridGraphElementTuple etup, PgturbohybridGraphElement element);
void		PgturbohybridGraphUpdateConnection(char *base, PgturbohybridGraphNeighborArray * neighbors, PgturbohybridGraphElement newElement, float distance, int lm, int *updateIdx, Relation index, PgturbohybridGraphSupport * support);
bool		PgturbohybridGraphLoadNeighborTids(PgturbohybridGraphElement element, ItemPointerData *indextids, Relation index, int m, int lm, int lc);
void		PgturbohybridGraphInitLockTranche(void);
const		PgturbohybridGraphTypeInfo *PgturbohybridGraphGetTypeInfo(Relation index);
PGDLLEXPORT void PgturbohybridParallelBuildMain(dsm_segment *seg, shm_toc *toc);
PGDLLEXPORT void PgturbohybridNativeParallelBuildMain(dsm_segment *seg, shm_toc *toc);

/* Index access methods */
IndexBuildResult *pgturbohybrid_graph_build(Relation heap, Relation index, IndexInfo *indexInfo);
IndexBuildResult *pgturbohybridbuild(Relation heap, Relation index, IndexInfo *indexInfo);
void		pgturbohybrid_graph_build_empty(Relation index);
void		pgturbohybridbuildempty(Relation index);
bool		pgturbohybrid_graph_insert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid, Relation heap, IndexUniqueCheck checkUnique
#if PG_VERSION_NUM >= 140000
					   ,bool indexUnchanged
#endif
					   ,IndexInfo *indexInfo
);
bool		pgturbohybridinsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid, Relation heap, IndexUniqueCheck checkUnique
#if PG_VERSION_NUM >= 140000
					   ,bool indexUnchanged
#endif
					   ,IndexInfo *indexInfo
);
IndexBulkDeleteResult *pgturbohybrid_graph_bulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats, IndexBulkDeleteCallback callback, void *callback_state);
IndexBulkDeleteResult *pgturbohybrid_graph_vacuum_cleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats);
IndexScanDesc pgturbohybrid_graph_begin_scan(Relation index, int nkeys, int norderbys);
IndexScanDesc pgturbohybridbeginscan(Relation index, int nkeys, int norderbys);
void		pgturbohybrid_graph_rescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys);
void		pgturbohybridrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys);
bool		pgturbohybrid_graph_get_tuple(IndexScanDesc scan, ScanDirection dir);
bool		pgturbohybridgettuple(IndexScanDesc scan, ScanDirection dir);
bool		pgturbohybridamgettuple(IndexScanDesc scan, ScanDirection dir);
void		pgturbohybrid_graph_end_scan(IndexScanDesc scan);
void		pgturbohybridendscan(IndexScanDesc scan);

IndexBuildResult *tqgraphbuild(Relation heap, Relation index, IndexInfo *indexInfo);
void		tqgraphbuildempty(Relation index);
bool		tqgraphinsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid, Relation heap, IndexUniqueCheck checkUnique
#if PG_VERSION_NUM >= 140000
					   ,bool indexUnchanged
#endif
					   ,IndexInfo *indexInfo
);
IndexBulkDeleteResult *tqgraphbulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats, IndexBulkDeleteCallback callback, void *callback_state);
IndexBulkDeleteResult *tqgraphvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats);
IndexScanDesc tqgraphbeginscan(Relation index, int nkeys, int norderbys);
void		tqgraphrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys);
bool		tqgraphgettuple(IndexScanDesc scan, ScanDirection dir);
void		tqgraphendscan(IndexScanDesc scan);

FUNCTION_PREFIX Datum pgturbohybrid_last_scan_stats(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_last_build_stats(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_index_stats(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_estimate_memory(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_prewarm(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_simd_capabilities(PG_FUNCTION_ARGS);
void		PgturbohybridGraphRecordExactVectorKernel(int kernel);
void		PgturbohybridGraphRecordWeightedCodeCodeKernel(int kernel);

static inline PgturbohybridGraphNeighborArray *
PgturbohybridGraphGetNeighbors(char *base, PgturbohybridGraphElement element, int lc)
{
	PgturbohybridGraphNeighborArrayPtr *neighborList = PgturbohybridGraphPtrAccess(base, element->neighbors);

	Assert(element->level >= lc);

	return PgturbohybridGraphPtrAccess(base, neighborList[lc]);
}

/* Hash tables */
typedef struct TidHashEntry
{
	ItemPointerData tid;
	char		status;
}			TidHashEntry;

#define SH_PREFIX tidhash
#define SH_ELEMENT_TYPE TidHashEntry
#define SH_KEY_TYPE ItemPointerData
#define SH_SCOPE extern
#define SH_DECLARE
#include "lib/simplehash.h"

typedef struct PointerHashEntry
{
	uintptr_t	ptr;
	char		status;
}			PointerHashEntry;

#define SH_PREFIX pointerhash
#define SH_ELEMENT_TYPE PointerHashEntry
#define SH_KEY_TYPE uintptr_t
#define SH_SCOPE extern
#define SH_DECLARE
#include "lib/simplehash.h"

typedef struct OffsetHashEntry
{
	Size		offset;
	char		status;
}			OffsetHashEntry;

#define SH_PREFIX offsethash
#define SH_ELEMENT_TYPE OffsetHashEntry
#define SH_KEY_TYPE Size
#define SH_SCOPE extern
#define SH_DECLARE
#include "lib/simplehash.h"

#endif
