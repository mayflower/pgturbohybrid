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
#define PGTURBOHYBRID_DEFAULT_ENTRY_SIDECAR false
#define PGTURBOHYBRID_DEFAULT_ENTRY_SIDECAR_REPRESENTATIVES 128
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
extern bool pgturbohybrid_dense_hadamard_simd;
extern int	pgturbohybrid_dense_simd_force;
extern int	pgturbohybrid_dense_query_split_impl;
extern int	pgturbohybrid_dense_u8_split;
extern bool pgturbohybrid_dense_u8_batch_x4;
extern int	pgturbohybrid_native_cache_max_mb;
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
extern int	pgturbohybrid_dense_adaptive_widening;
extern double pgturbohybrid_dense_adaptive_widening_multiplier;
extern double pgturbohybrid_dense_adaptive_widening_max_multiplier;
extern double pgturbohybrid_dense_adaptive_min_gap;
extern int	pgturbohybrid_dense_local_expansion;
extern int	pgturbohybrid_dense_local_expansion_topn;
extern int	pgturbohybrid_dense_local_expansion_max_neighbors;
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
	int			tqBits;
	bool		tqWeighted;		/* internal weighted quantized scoring flag */
	bool		tqQuantileFit;	/* internal quantile-anchored correction fit flag */
	bool		tqRenorm;		/* internal quantized renormalization residual flag */
	bool		tqExactStorage; /* exact_storage: store full exact vectors for final rescoring, or omit them for exact-free quantized-only storage. */
	bool		entrySidecar;	/* entry_sidecar: store data-aware representative node IDs in metadata. */
	int			entrySidecarRepresentatives;
	bool		graphBackbone;	/* graph_backbone: force adjacent level-0 graph edges at build time. */
	bool		residualRerank; /* residual_rerank: store tiny per-vector sketches for final-band reranking. */
	int			residualRerankBytes;
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

typedef struct PgturbohybridGraphTqQuery
{
	uint8	   *code;
#if PGTURBOHYBRID_GRAPH_ENABLE_SYMMETRIC_I8_DOT
	int8	   *queryI8;
#endif
	float	   *lut;
	float	   *queryValues;
	float	   *rawQueryValues;
	float	   *ecShift;
	float	   *ecScale;
	uint8	   *querySignBits;
	/*
	 * Asymmetric 1-bit query encoding.  Bit-plane-decomposed
	 * 8-bit signed query quantization, laid out in 128-dim blocks of
	 * `8 * 16` bytes (8 planes of 16 bytes each).  Recovers query
	 * magnitude information that the symmetric `querySignBits` path
	 * discards.  Populated by TqPrepareQueryAsymBit1 only when
	 * the private asymmetric-query path is enabled and quantization_bits = 1.
	 */
	uint8	   *queryPlanes;
	int64		queryAsymSumSigned;		/* Σ q_signed over all dims (full + tail) */
	float		queryAsymScale;			/* c / q_scale — postprocess multiplier  */
	int			queryAsymNumFullBlocks;	/* full 128-dim blocks                    */
	int			queryAsymTailBytes;		/* 0..15: bytes in trailing partial block */
	int			queryAsymBits;			/* BITS captured at precompute (8/12/16) */
#if defined(__aarch64__) || defined(_M_ARM64) || \
	defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
	int8	   *querySplitLow;
	int8	   *querySplitHigh;
	uint8	   *querySplitLowU8;
	uint8	   *querySplitHighU8;
	int8		querySplitTailLow[16];
	int8		querySplitTailHigh[16];
	uint8		querySplitTailLowU8[16];
	uint8		querySplitTailHighU8[16];
	int			querySplitChunks;
	int			querySplitTailDims;
	float		querySplitPostprocessScale;
	bool		querySplitEnabled;
	/*
	 * Unsigned-codebook query-split representation (x86 maddubs / VPDPBUSD).
	 * Separate from the signed path above: query halves are stored signed
	 * (no +128 XOR) because the unsigned u8 codebook is the unsigned operand,
	 * and the +128 codebook shift is unwound by u8SplitBias.  u8SplitData
	 * holds [low0..15, high0..15] per 16-dim chunk (32 bytes/chunk) so one
	 * AVX2 load is a [low|high] pair and one ZMM load is two chunks.
	 */
	int8	   *u8SplitData;
	int8		u8SplitTailLow[16];
	int8		u8SplitTailHigh[16];
	int64		u8SplitBias;	/* OFFSET * Sum(q_signed); subtract from raw dot */
	float		u8SplitPostprocessScale;
	bool		u8SplitEnabled;
#endif
#if PGTURBOHYBRID_GRAPH_ENABLE_SYMMETRIC_I8_DOT
	float		queryScale;
	float		queryCodeNorm;
#endif
	int			dimensions;
	int			bits;
	int			lutWidth;
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
	uint16		tqEntrySidecarCount;
	uint16		tqEntrySidecarBytes;
	uint16		tqResidualRerankBytes;
	uint16		tqReserved;
	uint32		tqEntrySidecarNodeIds[PGTURBOHYBRID_GRAPH_MAX_ENTRY_SIDECAR_REPRESENTATIVES];
	uint16		tqRoutingEntryCount;
	uint16		tqRoutingEntryBytes;
	uint32		tqRoutingEntryNodeIds[PGTURBOHYBRID_GRAPH_MAX_ROUTING_ENTRIES];
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
	int			graphOversampling;
	int			graphRescoreBand;
	int			graphExactCache;
	int64		tuples;
	int64		returnedRows;
	int64		tupleTargetRows;
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
	int64		graphCandidateCount;
	int64		graphRescoreCount;
	int64		graphRescorePages;
	int64		graphCodePagesRead;
	int64		graphAdjPagesRead;
	int64		graphEntryPointCount;
	int64		graphEntrySidecarCount;
	int64		graphEntrySidecarScored;
	int64		graphEntrySidecarSelected;
	int64		graphEntrySidecarUs;
	int64		graphPrepareUs;
	int64		graphTraverseUs;
	int64		graphEntryUs;
	int64		graphBaseUs;
	int64		graphBatchUs;
	int64		graphHeapUs;
	int64		graphFillUs;
	int64		graphRescoreUs;
	int64		graphSortUs;
	int64		graphTotalUs;
	int64		graphDenseRequestedK;
	int64		graphEffectiveResultTarget;
	int64		graphEffectiveSearchEf;
	int64		graphEffectiveRescoreBand;
	double		graphHighdimWideningMultiplier;
	int			graphWideningReason;
	int			graphAdaptiveWideningMode;
	bool		graphAdaptiveTriggered;
	int			graphAdaptiveTriggerReason;
	int64		graphAdaptiveInitialResultTarget;
	int64		graphAdaptiveFinalResultTarget;
	int64		graphAdaptiveInitialSearchEf;
	int64		graphAdaptiveFinalSearchEf;
	double		graphAdaptiveGapTop10;
	double		graphAdaptiveGapBoundary;
	int			graphLocalExpansionMode;
	bool		graphLocalExpansionTriggered;
	int64		graphLocalExpansionSeedCount;
	int64		graphLocalExpansionNeighborsScored;
	int64		graphLocalExpansionCandidatesAdded;
	int64		graphLocalExpansionUs;
	int64		graphResidualRerankCount;
	int64		graphResidualRerankBytes;
	int64		graphResidualRerankUs;
	int			graphDenseBudgetPolicy;
	int			graphRescoreBandPolicy;
	int			graphStorageKind;
	bool		graphExactStorage;
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
const char *PgturbohybridGraphTqScoringKernelName(int scoringKernel);
const char *PgturbohybridGraphTqScoreModeName(int scoreMode);
const char *PgturbohybridGraphRescoreBandName(int band);
const char *PgturbohybridGraphScoreKernelBucketName(int bucket);
const char *PgturbohybridGraphTqSimdForceName(int force);
const char *PgturbohybridGraphTqExactSimdForceName(int force);
const char *PgturbohybridGraphStorageKindName(int storageKind);
int			PgturbohybridGraphGetTqBits(Relation index);
bool		PgturbohybridGraphGetTqWeightedOption(Relation index);
bool		PgturbohybridGraphGetTqRenormOption(Relation index);
bool		PgturbohybridGraphGetTqQuantileFitOption(Relation index);
bool		PgturbohybridGraphGetTqExactStorageOption(Relation index);
bool		PgturbohybridGraphGetEntrySidecarOption(Relation index);
int			PgturbohybridGraphGetEntrySidecarRepresentatives(Relation index);
bool		PgturbohybridGraphGetBackboneOption(Relation index);
bool		PgturbohybridGraphGetResidualRerankOption(Relation index);
int			PgturbohybridGraphGetResidualRerankBytes(Relation index);
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
FUNCTION_PREFIX Datum pgturbohybrid_index_stats(PG_FUNCTION_ARGS);
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
