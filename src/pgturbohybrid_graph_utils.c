#include "postgres.h"

#include <float.h>
#include <math.h>
#include <string.h>

#include "access/genam.h"
#include "access/generic_xlog.h"
#include "access/htup_details.h"
#include "catalog/pg_am.h"
#include "catalog/pg_class.h"
#include "catalog/pg_index.h"
#include "catalog/pg_opclass.h"
#include "catalog/pg_type_d.h"
#include "common/hashfn.h"
#include "fmgr.h"
#include "pgturbohybrid.h"
#include "lib/pairingheap.h"
#include "nodes/pg_list.h"
#include "port/atomics.h"
#include "storage/bufmgr.h"
#include "utils/datum.h"
#include "utils/lsyscache.h"
#include "utils/memdebug.h"
#include "utils/rel.h"
#include "utils/syscache.h"
#include "lib/stringinfo.h"
#include "utils/jsonb.h"
#include "utils/fmgrprotos.h"
#include "pgturbohybrid_quant.h"
#include "pgturbohybrid_vector_compat.h"

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#define PGTURBOHYBRID_X86 1
#else
#define PGTURBOHYBRID_X86 0
#endif

#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && \
	(defined(__AVX2__) || (PGTURBOHYBRID_X86 && (defined(__GNUC__) || defined(__clang__))))
#define PGTURBOHYBRID_COMPILE_AVX2 1
#else
#define PGTURBOHYBRID_COMPILE_AVX2 0
#endif

#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && \
	(defined(__AVX512VNNI__) || (PGTURBOHYBRID_X86 && (defined(__GNUC__) || defined(__clang__))))
#define PGTURBOHYBRID_COMPILE_AVX512VNNI 1
#else
#define PGTURBOHYBRID_COMPILE_AVX512VNNI 0
#endif

/*
 * AVX-VNNI runtime detection requires __builtin_cpu_supports("avxvnni"), which
 * GCC has since 11.x and Clang since 18. See tqgraph.c for the parallel guard.
 */
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && \
	(defined(__AVXVNNI__) || \
	(PGTURBOHYBRID_X86 && defined(__clang__) && __clang_major__ >= 18) || \
	(PGTURBOHYBRID_X86 && defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11))
#define PGTURBOHYBRID_COMPILE_AVXVNNI 1
#else
#define PGTURBOHYBRID_COMPILE_AVXVNNI 0
#endif

#if PGTURBOHYBRID_X86 && (defined(__GNUC__) || defined(__clang__))
#define PGTURBOHYBRID_RUNTIME_AVX2 1
#define PGTURBOHYBRID_RUNTIME_AVX512VNNI 1
#else
#define PGTURBOHYBRID_RUNTIME_AVX2 0
#define PGTURBOHYBRID_RUNTIME_AVX512VNNI 0
#endif

#if PGTURBOHYBRID_COMPILE_AVXVNNI
#define PGTURBOHYBRID_RUNTIME_AVXVNNI 1
#else
#define PGTURBOHYBRID_RUNTIME_AVXVNNI 0
#endif

#if PGTURBOHYBRID_COMPILE_AVX2 && !defined(__AVX2__) && (defined(__GNUC__) || defined(__clang__))
#define PGTURBOHYBRID_AVX2_TARGET __attribute__((target("avx2")))
#else
#define PGTURBOHYBRID_AVX2_TARGET
#endif

#if PGTURBOHYBRID_COMPILE_AVX2
#include <immintrin.h>
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#if defined(__APPLE__)
#include <sys/types.h>
#include <sys/sysctl.h>
#endif
#if defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif
#endif

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

#if defined(USE_TARGET_CLONES) && !defined(__FMA__)
#define PGTURBOHYBRID_TARGET_CLONES __attribute__((target_clones("default", "fma")))
#else
#define PGTURBOHYBRID_TARGET_CLONES
#endif

static const float TqCodeCenters[PGTURBOHYBRID_LUT_WIDTH] = {
	-2.733f, -2.069f, -1.618f, -1.256f,
	-0.9424f, -0.6568f, -0.3881f, -0.1284f,
	0.1284f, 0.3881f, 0.6568f, 0.9424f,
	1.256f, 1.618f, 2.069f, 2.733f
};

#if defined(__aarch64__) || defined(_M_ARM64) || PGTURBOHYBRID_COMPILE_AVX2
#define PGTURBOHYBRID_QUERY_SPLIT_HIGH_COEF 256
#define PGTURBOHYBRID_QUERY_SPLIT_ABS_MAX 32639.0f
#define PGTURBOHYBRID_CODEBOOK_ABS_MAX 2.733f
#define PGTURBOHYBRID_CODEBOOK_SCALE (127.0f / PGTURBOHYBRID_CODEBOOK_ABS_MAX)
/* x86 unsigned-codebook query split: 7-bit signed halves, HIGH_COEF 128,
 * codebook shifted by +128 (bias = 128 * Sum(q_signed)). */
#define PGTURBOHYBRID_U8_SPLIT_HIGH_COEF 128
#define PGTURBOHYBRID_U8_SPLIT_ABS_MAX 8127.0f
#define PGTURBOHYBRID_U8_CODEBOOK_OFFSET 128
#define PGTURBOHYBRID_CODEBOOK2_ABS_MAX 1.510f
#define PGTURBOHYBRID_CODEBOOK2_SCALE (127.0f / PGTURBOHYBRID_CODEBOOK2_ABS_MAX)
#endif

static const float TqCodeCenters1[2] = {
	-0.7978846f, 0.7978846f
};

static const float TqCodeCenters2[4] = {
	-1.510f, -0.4528f, 0.4528f, 1.510f
};

static const float TqCodeBoundaries[PGTURBOHYBRID_LUT_WIDTH - 1] = {
	-2.401f, -1.8435f, -1.437f, -1.0992f,
	-0.7996f, -0.52245f, -0.25825f, 0.0f,
	0.25825f, 0.52245f, 0.7996f, 1.0992f,
	1.437f, 1.8435f, 2.401f
};

static const float TqCodeBoundaries1[1] = {
	0.0f
};

static const float TqCodeBoundaries2[3] = {
	-0.9814f, 0.0f, 0.9814f
};

static const uint64 TqRotationSeeds[3] = {
	UINT64CONST(654605292835415893),
	UINT64CONST(8636605637963351413),
	UINT64CONST(1775280196666917949)
};

static bool pgturbohybrid_graph_force_pgturbohybrid_index = false;

const char *
PgturbohybridGraphTqScoringKernelName(int scoringKernel)
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
			return "avx2";
		case PGTURBOHYBRID_SCORING_ARM_I8MM:
			return "i8mm";
		case PGTURBOHYBRID_SCORING_NEON:
			return "neon";
		case PGTURBOHYBRID_SCORING_SCALAR:
		default:
			return "scalar";
	}
}

const char *
PgturbohybridGraphTqSimdForceName(int force)
{
	switch ((TqSimdForce) force)
	{
		case PGTURBOHYBRID_SIMD_FORCE_AUTO:
			return "auto";
		case PGTURBOHYBRID_SIMD_FORCE_SCALAR:
			return "scalar";
		case PGTURBOHYBRID_SIMD_FORCE_AVX2:
			return "avx2";
		case PGTURBOHYBRID_SIMD_FORCE_AVXVNNI:
			return "avxvnni";
		case PGTURBOHYBRID_SIMD_FORCE_AVX512VNNI:
			return "avx512vnni";
		case PGTURBOHYBRID_SIMD_FORCE_NEON:
			return "neon";
		case PGTURBOHYBRID_SIMD_FORCE_ARM_SDOT:
			return "arm_sdot";
		case PGTURBOHYBRID_SIMD_FORCE_ARM_I8MM:
			return "arm_i8mm";
		default:
			return "unknown";
	}
}

const char *
PgturbohybridGraphTqExactSimdForceName(int force)
{
	switch ((TqExactSimdForce) force)
	{
		case PGTURBOHYBRID_EXACT_SIMD_FORCE_AUTO:
			return "auto";
		case PGTURBOHYBRID_EXACT_SIMD_FORCE_SCALAR:
			return "scalar";
		case PGTURBOHYBRID_EXACT_SIMD_FORCE_NEON:
			return "neon";
		case PGTURBOHYBRID_EXACT_SIMD_FORCE_AVX512F:
			return "avx512f";
		default:
			return "unknown";
	}
}

const char *
PgturbohybridGraphStorageKindName(int storageKind)
{
	switch (storageKind)
	{
		case PGTURBOHYBRID_GRAPH_STORAGE_QUANT_GRAPH:
			return "pgturbohybrid_graph";
		case PGTURBOHYBRID_GRAPH_STORAGE_QUANT_FLAT:
			return "pgturbohybrid_flat";
		case PGTURBOHYBRID_GRAPH_STORAGE_QUANT_GRAPH_NATIVE:
			return "pgturbohybrid_graph_native";
		case PGTURBOHYBRID_GRAPH_STORAGE_GRAPH:
		default:
			return "pgturbohybrid_graph";
	}
}

static int
TqSelectScoringKernel(void)
{
	if (pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_SCALAR)
		return PGTURBOHYBRID_SCORING_SCALAR;

#if PGTURBOHYBRID_COMPILE_AVX512VNNI
#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512BW__)
	if (pgturbohybrid_dense_graph_avx512vnni &&
		(pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_AUTO ||
		 pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_AVX512VNNI))
		return PGTURBOHYBRID_SCORING_AVX512VNNI;
#elif PGTURBOHYBRID_RUNTIME_AVX512VNNI
	if (pgturbohybrid_dense_graph_avx512vnni &&
		(pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_AUTO ||
		 pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_AVX512VNNI) &&
		__builtin_cpu_supports("avx512vnni") &&
		__builtin_cpu_supports("avx512vl") &&
		__builtin_cpu_supports("avx512bw"))
		return PGTURBOHYBRID_SCORING_AVX512VNNI;
#endif
#endif

#if PGTURBOHYBRID_COMPILE_AVXVNNI
#if defined(__AVXVNNI__)
	if (pgturbohybrid_dense_graph_avxvnni &&
		(pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_AUTO ||
		 pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_AVXVNNI))
		return PGTURBOHYBRID_SCORING_AVXVNNI;
#elif PGTURBOHYBRID_RUNTIME_AVXVNNI
	if (pgturbohybrid_dense_graph_avxvnni &&
		(pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_AUTO ||
		 pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_AVXVNNI) &&
		__builtin_cpu_supports("avxvnni"))
		return PGTURBOHYBRID_SCORING_AVXVNNI;
#endif
#endif

#if PGTURBOHYBRID_COMPILE_AVX2
#if defined(__AVX2__)
	if (pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_AUTO ||
		pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_AVX2)
		return PGTURBOHYBRID_SCORING_AVX2;
#elif PGTURBOHYBRID_RUNTIME_AVX2
	if ((pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_AUTO ||
		 pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_AVX2) &&
		__builtin_cpu_supports("avx2"))
		return PGTURBOHYBRID_SCORING_AVX2;
#endif
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#if defined(__APPLE__)
	{
		int			value = 0;
		size_t		len = sizeof(value);

		if (pgturbohybrid_dense_graph_i8mm &&
			(pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_AUTO ||
			 pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_ARM_I8MM) &&
			sysctlbyname("hw.optional.arm.FEAT_I8MM", &value, &len,
						 NULL, 0) == 0 && value != 0)
			return PGTURBOHYBRID_SCORING_ARM_I8MM;
	}
#elif defined(__linux__) && defined(HWCAP2_I8MM)
	if (pgturbohybrid_dense_graph_i8mm &&
		(pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_AUTO ||
		 pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_ARM_I8MM) &&
		(getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0)
		return PGTURBOHYBRID_SCORING_ARM_I8MM;
#endif
	if (pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_AUTO ||
		pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_NEON ||
		pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_ARM_SDOT)
		return PGTURBOHYBRID_SCORING_NEON;
	return PGTURBOHYBRID_SCORING_SCALAR;
#else
	return PGTURBOHYBRID_SCORING_SCALAR;
#endif
}

#if PG_VERSION_NUM < 170000
static inline uint64
murmurhash64(uint64 data)
{
	uint64		h = data;

	h ^= h >> 33;
	h *= 0xff51afd7ed558ccd;
	h ^= h >> 33;
	h *= 0xc4ceb9fe1a85ec53;
	h ^= h >> 33;

	return h;
}
#endif

/* TID hash table */
static uint32
hash_tid(ItemPointerData tid)
{
	union
	{
		uint64		i;
		ItemPointerData tid;
	}			x;

	/* Initialize unused bytes */
	x.i = 0;
	x.tid = tid;

	return murmurhash64(x.i);
}

#define SH_PREFIX		tidhash
#define SH_ELEMENT_TYPE	TidHashEntry
#define SH_KEY_TYPE		ItemPointerData
#define	SH_KEY			tid
#define SH_HASH_KEY(tb, key)	hash_tid(key)
#define SH_EQUAL(tb, a, b)		ItemPointerEquals(&a, &b)
#define	SH_SCOPE		extern
#define SH_DEFINE
#include "lib/simplehash.h"

void
PgturbohybridGraphSetForcepgturbohybridIndex(bool force)
{
	pgturbohybrid_graph_force_pgturbohybrid_index = force;
}

/* Pointer hash table */
static uint32
hash_pointer(uintptr_t ptr)
{
#if SIZEOF_VOID_P == 8
	return murmurhash64((uint64) ptr);
#else
	return murmurhash32((uint32) ptr);
#endif
}

#define SH_PREFIX		pointerhash
#define SH_ELEMENT_TYPE	PointerHashEntry
#define SH_KEY_TYPE		uintptr_t
#define	SH_KEY			ptr
#define SH_HASH_KEY(tb, key)	hash_pointer(key)
#define SH_EQUAL(tb, a, b)		(a == b)
#define	SH_SCOPE		extern
#define SH_DEFINE
#include "lib/simplehash.h"

/* Offset hash table */
static uint32
hash_offset(Size offset)
{
#if SIZEOF_SIZE_T == 8
	return murmurhash64((uint64) offset);
#else
	return murmurhash32((uint32) offset);
#endif
}

#define SH_PREFIX		offsethash
#define SH_ELEMENT_TYPE	OffsetHashEntry
#define SH_KEY_TYPE		Size
#define	SH_KEY			offset
#define SH_HASH_KEY(tb, key)	hash_offset(key)
#define SH_EQUAL(tb, a, b)		(a == b)
#define	SH_SCOPE		extern
#define SH_DEFINE
#include "lib/simplehash.h"

static bool
PgturbohybridGraphAmOidUsesNativeTqGraph(Oid amoid)
{
	HeapTuple	amtuple;
	Form_pg_am	amform;
	bool		result;

	if (!OidIsValid(amoid))
		return false;

	amtuple = SearchSysCache1(AMOID, ObjectIdGetDatum(amoid));
	if (!HeapTupleIsValid(amtuple))
		return false;

	amform = (Form_pg_am) GETSTRUCT(amtuple);
	result = strcmp(NameStr(amform->amname), "turbohybrid") == 0;
	ReleaseSysCache(amtuple);

	return result;
}

/*
 * Get the max number of connections in an upper layer for each element in the index
 */
int
PgturbohybridGraphGetM(Relation index)
{
	PgturbohybridGraphOptions *opts = (PgturbohybridGraphOptions *) index->rd_options;

	if (opts)
		return opts->m;

	if (PgturbohybridGraphIspgturbohybridIndex(index))
		return PGTURBOHYBRID_DEFAULT_GRAPH_M;

	return PGTURBOHYBRID_GRAPH_DEFAULT_M;
}

/*
 * Get the size of the dynamic candidate list in the index
 */
int
PgturbohybridGraphGetEfConstruction(Relation index)
{
	PgturbohybridGraphOptions *opts = (PgturbohybridGraphOptions *) index->rd_options;

	if (opts)
		return opts->efConstruction;

	if (PgturbohybridGraphIspgturbohybridIndex(index))
		return PGTURBOHYBRID_DEFAULT_GRAPH_EF_CONSTRUCTION;

	return PGTURBOHYBRID_GRAPH_DEFAULT_EF_CONSTRUCTION;
}

bool
PgturbohybridGraphIspgturbohybridIndex(Relation index)
{
	HeapTuple	reltuple;
	HeapTuple	opclasstuple;
	Form_pg_class relform;
	Form_pg_opclass opclassform;
	Datum		indclassDatum;
	oidvector  *indclass;
	Oid			relam;
	bool		isnull;

	if (index == NULL || index->rd_rel == NULL)
		return false;

	if (pgturbohybrid_graph_force_pgturbohybrid_index)
		return true;

	reltuple = SearchSysCache1(RELOID, ObjectIdGetDatum(RelationGetRelid(index)));
	if (!HeapTupleIsValid(reltuple))
		return false;

	relform = (Form_pg_class) GETSTRUCT(reltuple);
	relam = relform->relam;
	ReleaseSysCache(reltuple);

	if (PgturbohybridGraphAmOidUsesNativeTqGraph(relam))
		return true;

	if (index->rd_index == NULL || index->rd_indextuple == NULL)
		return false;

	indclassDatum = SysCacheGetAttr(INDEXRELID, index->rd_indextuple,
									Anum_pg_index_indclass, &isnull);
	if (isnull)
		return false;

	indclass = (oidvector *) DatumGetPointer(indclassDatum);
	if (indclass->dim1 < 1)
		return false;

	opclasstuple = SearchSysCache1(CLAOID,
								   ObjectIdGetDatum(indclass->values[0]));
	if (!HeapTupleIsValid(opclasstuple))
		return false;

	opclassform = (Form_pg_opclass) GETSTRUCT(opclasstuple);
	relam = opclassform->opcmethod;
	ReleaseSysCache(opclasstuple);

	return PgturbohybridGraphAmOidUsesNativeTqGraph(relam);
}

static int
PgturbohybridGraphGetTqRouting(Relation index)
{
	TqOptions  *opts = (TqOptions *) index->rd_options;

	if (opts)
		return opts->routing;

	return PGTURBOHYBRID_ROUTING_AUTO;
}

int
PgturbohybridGraphGetTqBits(Relation index)
{
	TqOptions  *opts = (TqOptions *) index->rd_options;

	if (!PgturbohybridGraphIspgturbohybridIndex(index))
		return PGTURBOHYBRID_DEFAULT_BITS;

	if (opts != NULL && opts->tqBits > 0)
		return opts->tqBits;

	return PGTURBOHYBRID_DEFAULT_INDEX_BITS;
}

bool
PgturbohybridGraphGetTqWeightedOption(Relation index)
{
	TqOptions  *opts;

	if (!PgturbohybridGraphIspgturbohybridIndex(index))
		return false;

	opts = (TqOptions *) index->rd_options;
	return opts != NULL && opts->tqWeighted;
}

bool
PgturbohybridGraphGetTqRenormOption(Relation index)
{
	TqOptions  *opts;

	if (!PgturbohybridGraphIspgturbohybridIndex(index))
		return false;

	opts = (TqOptions *) index->rd_options;
	return opts != NULL && opts->tqRenorm;
}

bool
PgturbohybridGraphGetTqQuantileFitOption(Relation index)
{
	TqOptions  *opts;

	if (!PgturbohybridGraphIspgturbohybridIndex(index))
		return false;

	opts = (TqOptions *) index->rd_options;
	return opts != NULL && opts->tqQuantileFit;
}

bool
PgturbohybridGraphGetTqExactStorageOption(Relation index)
{
	TqOptions  *opts;

	if (!PgturbohybridGraphIspgturbohybridIndex(index))
		return true;

	opts = (TqOptions *) index->rd_options;
	return opts != NULL ? opts->tqExactStorage : PGTURBOHYBRID_DEFAULT_EXACT_STORAGE;
}

bool
PgturbohybridGraphGetEntrySidecarOption(Relation index)
{
	TqOptions  *opts;

	if (!PgturbohybridGraphIspgturbohybridIndex(index))
		return false;

	opts = (TqOptions *) index->rd_options;
	return opts != NULL ? opts->entrySidecar : PGTURBOHYBRID_DEFAULT_ENTRY_SIDECAR;
}

int
PgturbohybridGraphGetEntrySidecarRepresentatives(Relation index)
{
	TqOptions  *opts;

	if (!PgturbohybridGraphIspgturbohybridIndex(index))
		return 0;

	opts = (TqOptions *) index->rd_options;
	return opts != NULL ? opts->entrySidecarRepresentatives :
		PGTURBOHYBRID_DEFAULT_ENTRY_SIDECAR_REPRESENTATIVES;
}

bool
PgturbohybridGraphGetBackboneOption(Relation index)
{
	TqOptions  *opts;

	if (!PgturbohybridGraphIspgturbohybridIndex(index))
		return false;

	opts = (TqOptions *) index->rd_options;
	return opts != NULL ? opts->graphBackbone : PGTURBOHYBRID_DEFAULT_GRAPH_BACKBONE;
}

bool
PgturbohybridGraphGetResidualRerankOption(Relation index)
{
	TqOptions  *opts;

	if (!PgturbohybridGraphIspgturbohybridIndex(index))
		return false;

	opts = (TqOptions *) index->rd_options;
	return opts != NULL ? opts->residualRerank : PGTURBOHYBRID_DEFAULT_RESIDUAL_RERANK;
}

int
PgturbohybridGraphGetResidualRerankBytes(Relation index)
{
	TqOptions  *opts;

	if (!PgturbohybridGraphIspgturbohybridIndex(index))
		return 0;

	opts = (TqOptions *) index->rd_options;
	if (opts == NULL || !opts->residualRerank)
		return 0;
	return opts->residualRerankBytes;
}

static bool
PgturbohybridGraphTqSupportsPackedCodes(Relation index)
{
	FmgrInfo   *procinfo = PgturbohybridGraphOptionalProcInfo(index, PGTURBOHYBRID_GRAPH_TYPE_INFO_PROC);

	return procinfo == NULL || get_func_rettype(procinfo->fn_oid) != INTERNALOID;
}

bool
PgturbohybridGraphUseTqNativeGraph(Relation index)
{
	int			routing;

	if (!PgturbohybridGraphIspgturbohybridIndex(index))
		return false;

	routing = PgturbohybridGraphGetTqRouting(index);

	if (routing == PGTURBOHYBRID_ROUTING_GRAPH)
		return PgturbohybridGraphTqSupportsPackedCodes(index);
	if (routing == PGTURBOHYBRID_ROUTING_FLAT || routing == PGTURBOHYBRID_ROUTING_LEGACY_GRAPH)
		return false;

	/*
	 * Auto selects the native graph only for indexes that have the packed-code
	 * scorer. Build rejects unsupported opclasses instead of silently falling
	 * back to the pgvector-HNSW-compatible path.
	 */
	return PgturbohybridGraphTqSupportsPackedCodes(index);
}

bool
PgturbohybridGraphUseTqGraph(Relation index)
{
	return PgturbohybridGraphUseTqNativeGraph(index);
}

bool
PgturbohybridGraphUseTqFlat(Relation index)
{
	return PgturbohybridGraphIspgturbohybridIndex(index) &&
		PgturbohybridGraphGetTqRouting(index) == PGTURBOHYBRID_ROUTING_FLAT;
}

int
PgturbohybridGraphGetEfSearch(Relation index)
{
	if (PgturbohybridGraphUseTqNativeGraph(index))
	{
		TqOptions  *opts = (TqOptions *) index->rd_options;

		if (opts)
			return opts->graphEfSearch;

		return PGTURBOHYBRID_DEFAULT_GRAPH_EF_SEARCH;
	}

	return pgturbohybrid_ef_search;
}

int
PgturbohybridGraphGetGraphOversampling(Relation index)
{
	if (PgturbohybridGraphUseTqNativeGraph(index))
	{
		TqOptions  *opts = (TqOptions *) index->rd_options;

		if (opts)
			return opts->graphOversampling;

		return PGTURBOHYBRID_DEFAULT_GRAPH_OVERSAMPLING;
	}

	return 1;
}

int
PgturbohybridGraphGetGraphRescoreBand(Relation index)
{
	if (PgturbohybridGraphUseTqNativeGraph(index))
		return PGTURBOHYBRID_GRAPH_RESCORE_BAND_AUTO;

	return PGTURBOHYBRID_GRAPH_RESCORE_BAND_EXACT;
}

int
PgturbohybridGraphGetGraphExactCache(Relation index)
{
	if (PgturbohybridGraphUseTqNativeGraph(index))
		return PGTURBOHYBRID_GRAPH_EXACT_CACHE_AUTO;

	return PGTURBOHYBRID_GRAPH_EXACT_CACHE_AUTO;
}

int
PgturbohybridGraphGetGraphReorder(Relation index)
{
	if (PgturbohybridGraphUseTqNativeGraph(index))
		return PGTURBOHYBRID_GRAPH_REORDER_OFF;

	return PGTURBOHYBRID_GRAPH_REORDER_AUTO;
}

/*
 * The first native pgturbohybrid code path targets the plain vector opclasses.
 * Other data types keep using the exact HNSW tuple payload until they get a
 * type-specific packed representation.
 */
bool
PgturbohybridGraphUseTqCodes(Relation index)
{
	return PgturbohybridGraphUseTqNativeGraph(index) &&
		PgturbohybridGraphTqSupportsPackedCodes(index);
}

static inline uint8
TqEncodeComponentBits(float value, int bits)
{
	const float *boundaries;
	int			boundaryCount;

	if (isnan(value))
		value = 0;

	if (bits == 1)
	{
		boundaries = TqCodeBoundaries1;
		boundaryCount = lengthof(TqCodeBoundaries1);
	}
	else if (bits == 2)
	{
		boundaries = TqCodeBoundaries2;
		boundaryCount = lengthof(TqCodeBoundaries2);
	}
	else
	{
		boundaries = TqCodeBoundaries;
		boundaryCount = lengthof(TqCodeBoundaries);
	}

	for (int code = 0; code < boundaryCount; code++)
	{
		if (value <= boundaries[code])
			return (uint8) code;
	}

	return (uint8) boundaryCount;
}

static inline uint64
TqRotationLcgNext(uint64 state)
{
	return state * UINT64CONST(6364136223846793005) +
		UINT64CONST(1442695040888963407);
}

static inline uint64
TqRotationBoundedRand(uint64 value, uint64 bound)
{
	return (value >> 32) % bound;
}

static void
TqRotatePermute(double *values, int dim, uint64 seed)
{
	uint64		state = seed;

	for (int i = dim - 1; i >= 1; i--)
	{
		uint64		j;
		double		tmp;

		state = TqRotationLcgNext(state);
		j = TqRotationBoundedRand(state, (uint64) i + 1);
		tmp = values[i];
		values[i] = values[j];
		values[j] = tmp;
	}
}

/*
 * Hadamard butterfly inner loop: for a block of 2·h doubles starting
 * at `block`, compute pairwise (a+b, a-b) where a is the lower h
 * doubles and b is the upper h doubles.  Each (j, j+h) pair is
 * independent — the order of inner-loop iterations does not affect
 * the result.
 *
 * The scalar form is fully self-contained.  SIMD variants below
 * preserve bit-exactness by using the same f64 add / sub primitives
 * (no FMA, no reordering) on 4 (AVX2) or 2 (NEON) independent lanes
 * at a time.
 */
static inline void
TqHadamardBlockScalar(double *block, int h)
{
	for (int j = 0; j < h; j++)
	{
		double		a = block[j];
		double		b = block[j + h];

		block[j] = a + b;
		block[j + h] = a - b;
	}
}

#if PGTURBOHYBRID_COMPILE_AVX2
static inline void PGTURBOHYBRID_AVX2_TARGET
TqHadamardBlockAvx2(double *block, int h)
{
	int			j = 0;

	/* SIMD body: 4 independent f64 (a+b, a-b) per iter. */
	for (; j + 4 <= h; j += 4)
	{
		__m256d		a = _mm256_loadu_pd(block + j);
		__m256d		b = _mm256_loadu_pd(block + j + h);

		_mm256_storeu_pd(block + j, _mm256_add_pd(a, b));
		_mm256_storeu_pd(block + j + h, _mm256_sub_pd(a, b));
	}
	/* Tail: scalar remainder.  Bit-exact with full-scalar. */
	for (; j < h; j++)
	{
		double		a = block[j];
		double		b = block[j + h];

		block[j] = a + b;
		block[j + h] = a - b;
	}
}
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
static inline void
TqHadamardBlockNeon(double *block, int h)
{
	int			j = 0;

	for (; j + 2 <= h; j += 2)
	{
		float64x2_t a = vld1q_f64(block + j);
		float64x2_t b = vld1q_f64(block + j + h);

		vst1q_f64(block + j, vaddq_f64(a, b));
		vst1q_f64(block + j + h, vsubq_f64(a, b));
	}
	for (; j < h; j++)
	{
		double		a = block[j];
		double		b = block[j + h];

		block[j] = a + b;
		block[j + h] = a - b;
	}
}
#endif

static void
TqRotateHadamardChunks(double *values, int dim)
{
	int			offset = 0;
	int			remaining = dim;
	bool		simdGate = pgturbohybrid_dense_hadamard_simd;
#if PGTURBOHYBRID_RUNTIME_AVX2
	bool		useAvx2 = simdGate && __builtin_cpu_supports("avx2");
#endif

	while (remaining > 0)
	{
		int			chunk = 1;
		double		norm;

		while (chunk <= remaining / 2)
			chunk <<= 1;

		for (int h = 1; h < chunk; h <<= 1)
		{
			/*
			 * h=1 and h=2 levels can't use the wide SIMD butterfly
			 * cleanly (the load/store pattern interleaves a/b too
			 * tightly for a YMM/Q register), so they stay scalar.
			 * They only account for ~9 % of total butterfly ops at
			 * dim=1536 (chunk * 1 + chunk * 2 = 3·chunk out of
			 * chunk · log2(chunk) ≈ 11·chunk for chunk=1024).
			 */
			if (h >= 4)
			{
#if PGTURBOHYBRID_RUNTIME_AVX2
				if (useAvx2)
				{
					for (int i = offset; i < offset + chunk; i += h * 2)
						TqHadamardBlockAvx2(values + i, h);
					continue;
				}
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
				if (simdGate)
				{
					for (int i = offset; i < offset + chunk; i += h * 2)
						TqHadamardBlockNeon(values + i, h);
					continue;
				}
#endif
			}

			for (int i = offset; i < offset + chunk; i += h * 2)
				TqHadamardBlockScalar(values + i, h);
		}

		norm = 1.0 / sqrt((double) chunk);
		for (int i = offset; i < offset + chunk; i++)
			values[i] *= norm;

		offset += chunk;
		remaining -= chunk;
	}
}

static void
TqRotateVector(const float *input, int dim, double *rotated)
{
	for (int i = 0; i < dim; i++)
		rotated[i] = isnan(input[i]) ? 0.0 : (double) input[i];

	TqRotateHadamardChunks(rotated, dim);
	for (int i = 0; i < lengthof(TqRotationSeeds); i++)
	{
		TqRotatePermute(rotated, dim, TqRotationSeeds[i]);
		TqRotateHadamardChunks(rotated, dim);
	}
}

static void
TqRotateVectorFloat(const float *input, int dim, float *rotated)
{
	double	   *buffer = palloc(sizeof(double) * dim);

	TqRotateVector(input, dim, buffer);
	for (int i = 0; i < dim; i++)
		rotated[i] = (float) buffer[i];

	pfree(buffer);
}

static float
TqVectorLength(const double *values, int dim)
{
	double		norm = 0;

	for (int i = 0; i < dim; i++)
		norm += values[i] * values[i];

	return (float) sqrt(norm);
}

float
TqPreprocessVector(Vector *vector, double *rotated)
{
	int			dim = vector->dim;
	float		length;
	double		scale;

	TqRotateVector(vector->x, dim, rotated);
	length = TqVectorLength(rotated, dim);
	scale = length > 0 ? sqrt((double) dim) / (double) length : 1.0;

	for (int i = 0; i < dim; i++)
		rotated[i] *= scale;

	return length;
}

static float
TqEncodeVectorInternal(Vector *vector, uint8 *code, const float *ecShift,
					   const float *ecScale, int bits, float *xmOut,
					   float *centroidNormOut)
{
	int			dim = vector->dim;
	double	   *rotated = palloc(sizeof(double) * dim);
	float		length;
	double		xm = 0.0;
	double		cnSqSum = 0.0;
	bool		wantCn = centroidNormOut != NULL &&
		ecShift != NULL && ecScale != NULL;

	memset(code, 0, TqCodeSizeForBits(dim, bits));
	length = TqPreprocessVector(vector, rotated);

	/*
	 * TQ+ xm = ⟨rotated, -ecShift⟩, computed against the post-rescale
	 * rotated buffer (qdrant convention).  Captured here before the
	 * in-place transform on the next loop discards the rotated values.
	 */
	if (xmOut != NULL && ecShift != NULL)
	{
		for (int i = 0; i < dim; i++)
			xm -= rotated[i] * (double) ecShift[i];
	}

	for (int i = 0; i < dim; i++)
	{
		double		value = rotated[i];
		uint8		component;

		if (ecShift != NULL && ecScale != NULL)
			value = (value + ecShift[i]) * ecScale[i];

		component = TqEncodeComponentBits((float) value, bits);

		if (bits == 1)
			code[i / 8] |= component << (i & 7);
		else if (bits == 2)
			code[i / 4] |= component << ((i & 3) * 2);
		else if ((i & 1) == 0)
			code[i / 2] |= component;
		else
			code[i / 2] |= component << PGTURBOHYBRID_BITS;

		/*
		 * Accumulate the centroid_norm of the *decoded*
		 * vector in EC-reverted (rescaled-pre-EC) space.  Matches
		 * qdrant compute_centroid_norm: c_reverted = c / scale - shift.
		 * The deterministic ‖rescaled‖ is sqrt(d); cnSqSum drifts
		 * from d only due to quantization noise, which is exactly
		 * what the renormalization correction folds back in.
		 */
		if (wantCn)
		{
			double		c = (double) TqGetCodeCenterBits(component, bits);
			double		c_reverted = c / (double) ecScale[i] - (double) ecShift[i];

			cnSqSum += c_reverted * c_reverted;
		}
	}

	if (xmOut != NULL)
		*xmOut = (float) xm;

	if (centroidNormOut != NULL)
	{
		/*
		 * Cosine zero-input guard (mirrors qdrant's substitution at
		 * quantization.rs:245-262): if the input was identically zero,
		 * `length` is 0 and the rescaled rotated vector is all-zero;
		 * cnSqSum collapses to a small value driven by the EC-zero
		 * point's quantization and renorm would divide by ~0.  Substitute
		 * sqrt(d) so the renormalized factor sqrt(d) / cn equals 1.
		 */
		if (!wantCn || length == 0.0f || cnSqSum <= 0.0)
			*centroidNormOut = (float) sqrt((double) dim);
		else
			*centroidNormOut = (float) sqrt(cnSqSum);
	}

	pfree(rotated);
	return length;
}

Size
TqCodeSizeForBits(int dimensions, int bits)
{
	if (bits != 1 && bits != 2 && bits != PGTURBOHYBRID_DEFAULT_BITS)
		bits = PGTURBOHYBRID_DEFAULT_BITS;

	return PGTURBOHYBRID_CODE_SIZE_BITS(dimensions, bits);
}

int
TqGetCodeComponentBits(const uint8 *code, int i, int bits)
{
	if (bits == 1)
		return (code[i / 8] >> (i & 7)) & 0x01;
	if (bits == 2)
		return (code[i / 4] >> ((i & 3) * 2)) & 0x03;

	return ((i & 1) == 0 ? code[i / 2] : code[i / 2] >> PGTURBOHYBRID_BITS) & 0x0f;
}

float
TqGetCodeCenterBits(int code, int bits)
{
	if (bits == 1)
		return TqCodeCenters1[code & 0x01];
	if (bits == 2)
		return TqCodeCenters2[code & 0x03];

	return TqCodeCenters[code & 0x0f];
}

float
TqEncodeVectorBits(Vector *vector, uint8 *code, int bits)
{
	return TqEncodeVectorInternal(vector, code, NULL, NULL, bits, NULL, NULL);
}

float
TqEncodeVectorWithCorrectionBits(Vector *vector, uint8 *code, int bits,
								 const float *ecShift, const float *ecScale)
{
	return TqEncodeVectorInternal(vector, code, ecShift, ecScale, bits, NULL, NULL);
}

float
TqEncodeVectorWithCorrectionAndXmBits(Vector *vector, uint8 *code, int bits,
									   const float *ecShift, const float *ecScale,
									   float *xmOut)
{
	return TqEncodeVectorInternal(vector, code, ecShift, ecScale, bits, xmOut, NULL);
}

float
TqEncodeVectorWithCorrectionXmRenormBits(Vector *vector, uint8 *code, int bits,
										  const float *ecShift, const float *ecScale,
										  float *xmOut, float *centroidNormOut)
{
	return TqEncodeVectorInternal(vector, code, ecShift, ecScale, bits,
								  xmOut, centroidNormOut);
}

float
TqEncodeVector(Vector *vector, uint8 *code)
{
	return TqEncodeVectorBits(vector, code, PGTURBOHYBRID_DEFAULT_BITS);
}

float
TqEncodeVectorWithCorrection(Vector *vector, uint8 *code,
							 const float *ecShift, const float *ecScale)
{
	return TqEncodeVectorWithCorrectionBits(vector, code, PGTURBOHYBRID_DEFAULT_BITS,
										   ecShift, ecScale);
}

static inline double
TqDimSqrt(int dim)
{
	return sqrt((double) dim);
}

#if PGTURBOHYBRID_GRAPH_ENABLE_SYMMETRIC_I8_DOT
static float
TqEncodeQueryInt8(Vector *vector, int8 *code, float *codeNorm)
{
	int			dim = vector->dim;
	float		scale = 0;
	float		norm = 0;

	memset(code, 0, sizeof(int8) * dim);

	for (int i = 0; i < dim; i++)
	{
		float		value = fabsf(vector->x[i]);

		if (!isnan(value) && value > scale)
			scale = value;
	}

	if (scale <= 0)
		scale = 1;
	else
		scale /= 127.0f;

	for (int i = 0; i < dim; i++)
	{
		long		quantized;

		if (isnan(vector->x[i]))
			quantized = 0;
		else
			quantized = lrintf(vector->x[i] / scale);

		if (quantized < -127)
			quantized = -127;
		else if (quantized > 127)
			quantized = 127;

		code[i] = (int8) quantized;
		norm += (float) code[i] * (float) code[i];
	}

	*codeNorm = norm;
	return scale;
}
#endif

static void
TqBuildQueryLut(PgturbohybridGraphTqQuery *tq)
{
	int			dim = tq->dimensions;
	TqScoreMode mode = (TqScoreMode) tq->scoreMode;

	for (int i = 0; i < dim; i++)
	{
		float		qv = tq->queryValues[i];
		float	   *row = tq->lut + (i * PGTURBOHYBRID_LUT_WIDTH);

		for (int j = 0; j < tq->lutWidth; j++)
		{
			float		vv = TqGetCodeCenterBits(j, tq->bits);

			if (mode == PGTURBOHYBRID_SCORE_L1)
				row[j] = vv;
			else
				row[j] = qv * vv;
		}
	}
}

static void
TqPrepareQuerySignBits(PgturbohybridGraphTqQuery *tq)
{
	if (!tq->enabled || tq->bits != 1 || tq->dimensions <= 0)
		return;

	tq->querySignBits = palloc0(TqCodeSizeForBits(tq->dimensions, 1));
	for (int i = 0; i < tq->dimensions; i++)
	{
		if (tq->queryValues[i] > 0)
			tq->querySignBits[i / 8] |= (uint8) (1U << (i & 7));
	}
}

/*
 * Asymmetric 1-bit query encoding.
 *
 * Quantize the rotated, EC-rescaled query to 8-bit signed integers and
 * lay them out as bit-plane-decomposed 128-dim blocks (8 planes × 16
 * bytes each).  Per-block scoring is then
 *
 *   v_dot_q  = Σ_b w_b · popcount(v_block AND plane_b)
 *
 * with w_b = 2^b for b < 7 and -128 for the sign plane.  The asymmetric
 * dot is reconstructed via
 *
 *   signed_dot = 2 · v_dot_q − Σ q_signed
 *   score      = (c / q_scale) · signed_dot
 *
 * matching qdrant Query1bitSimd<8> from PR #8749.  The trailing partial
 * block (when dim is not a multiple of 128) shares the same SIMD kernel
 * via a zero-padded scratch buffer; only the data side needs padding —
 * the bit-plane bytes beyond `tail_bytes` are kept zero so `data AND
 * plane = 0` for the padding lanes.
 *
 * Mirrors TqPrepareQuerySignBits's signature: takes an already-rotated,
 * EC-rescaled query in tq->queryValues.  Skips if the GUC kill-switch
 * is off or bits != 1.
 */
#define PGTURBOHYBRID_QUERY_ASYM_BITS_DEFAULT	8
#define PGTURBOHYBRID_QUERY_ASYM_BLOCK_BYTES	16

/*
 * Snap an arbitrary GUC value to one of the kernel-supported
 * widths.  8, 12, 16 are the cleanly tested widths; 4/6 are below the
 * useful accuracy threshold for the data widths we care about, and
 * 10/14 are odd in a way that doesn't simplify the kernel.  Anything
 * else snaps to the nearest supported width.
 */
static int
TqQueryAsymSupportedBits(int bits)
{
	if (bits <= 8)
		return 8;
	if (bits <= 12)
		return 12;
	return 16;
}

static void
TqPrepareQueryAsymBit1(PgturbohybridGraphTqQuery *tq)
{
	int			dim;
	int			numFullBlocks;
	int			fullDims;
	int			tailDims;
	int			tailBytes;
	int			totalBlocks;
	int			BITS;
	int			qIntMax;
	float		qAbsMax = FLT_EPSILON;
	float		qScale;
	int64		sumSigned = 0;

	if (!pgturbohybrid_dense_query_1bit_asymmetric || !tq->enabled ||
		tq->bits != 1 || tq->dimensions <= 0)
		return;

	dim = tq->dimensions;
	BITS = TqQueryAsymSupportedBits(pgturbohybrid_dense_query_1bit_asymmetric_bits);
	qIntMax = (1 << (BITS - 1)) - 1;

	/*
	 * 1-bit packing rounds up: byte_len = ceil(dim / 8).  pgvector
	 * stores partial trailing bytes with zero-padding beyond `dim`,
	 * and PgturbohybridGraphBit1PopcntRawCodes masks them off.  The asymmetric
	 * scorer follows the same convention: a non-multiple-of-8 dim is
	 * encoded into ceil(dim/8) bytes of the trailing partial block,
	 * with bit positions beyond `dim` in the last partial byte left
	 * as zero so AND-popcount against them contributes 0.
	 */
	numFullBlocks = dim / (8 * PGTURBOHYBRID_QUERY_ASYM_BLOCK_BYTES);
	fullDims = numFullBlocks * 8 * PGTURBOHYBRID_QUERY_ASYM_BLOCK_BYTES;
	tailDims = dim - fullDims;
	Assert(tailDims < 8 * PGTURBOHYBRID_QUERY_ASYM_BLOCK_BYTES);
	tailBytes = (tailDims + 7) / 8;	/* ceil — handles non-multiple-of-8 dim */
	totalBlocks = numFullBlocks + (tailBytes > 0 ? 1 : 0);

	for (int i = 0; i < dim; i++)
	{
		float		v = fabsf(tq->queryValues[i]);

		if (v > qAbsMax)
			qAbsMax = v;
	}
	qScale = (float) qIntMax / qAbsMax;

	tq->queryPlanes = palloc0((Size) totalBlocks * BITS *
							  PGTURBOHYBRID_QUERY_ASYM_BLOCK_BYTES);
	tq->queryAsymNumFullBlocks = numFullBlocks;
	tq->queryAsymTailBytes = tailBytes;
	tq->queryAsymBits = BITS;

	for (int i = 0; i < dim; i++)
	{
		int			blockIdx = i / (8 * PGTURBOHYBRID_QUERY_ASYM_BLOCK_BYTES);
		int			inBlock = i - blockIdx * 8 * PGTURBOHYBRID_QUERY_ASYM_BLOCK_BYTES;
		int			byteInBlock = inBlock / 8;
		int			bitInByte = inBlock & 7;
		int			qInt;
		uint32		qBits;
		uint8	   *blockBase;
		float		scaled = tq->queryValues[i] * qScale;

		/*
		 * Round-half-away-from-zero is what qdrant's `(v *
		 * q_scale).round()` does on stable Rust; mirror it via
		 * roundf so SIMD parity is byte-exact across the scalar +
		 * vectorised paths.
		 */
		qInt = (int) roundf(scaled);
		if (qInt > qIntMax)
			qInt = qIntMax;
		else if (qInt < -qIntMax)
			qInt = -qIntMax;

		sumSigned += qInt;
		/* mask to BITS bits — two's-complement representation */
		qBits = (uint32) qInt & ((BITS >= 32) ? 0xFFFFFFFFu : ((1u << BITS) - 1));

		blockBase = tq->queryPlanes +
			(Size) blockIdx * BITS * PGTURBOHYBRID_QUERY_ASYM_BLOCK_BYTES;

		for (int b = 0; b < BITS; b++)
		{
			uint8		bit = (uint8) ((qBits >> b) & 1);

			blockBase[b * PGTURBOHYBRID_QUERY_ASYM_BLOCK_BYTES + byteInBlock] |=
				(uint8) (bit << bitInByte);
		}
	}

	tq->queryAsymSumSigned = sumSigned;
	tq->queryAsymScale = TqCodeCenters1[1] / qScale;	/* c / q_scale */
}

/*
 * Invariant for the AMD64 (x86) and aarch64 QuerySplit pipeline.
 *
 * Storage:
 *   codebook        signed i8 (PgturbohybridGraphCodebookI8 / PgturbohybridGraphCodebook2I8)
 *   query halves    signed i8 in tq->querySplitLow / querySplitHigh
 *   K               PGTURBOHYBRID_QUERY_SPLIT_HIGH_COEF = 256 on amd64 and aarch64
 *
 * Encoding:
 *   q_signed[i] in [-32639, 32639] (saturated by PGTURBOHYBRID_QUERY_SPLIT_ABS_MAX)
 *   low[i]      in [-127, 127]    (q_signed mod 256, mapped to symmetric range)
 *   high[i]     in [-127, 127]    ((q_signed - low) / 256)
 *   q_signed = K * high + low                                      (1)
 *
 * Scoring (per dim i, c is the signed codebook value):
 *   contribution = c * q_signed = c * (K * high + low)             (2)
 *
 * Postprocess:
 *   tq->querySplitPostprocessScale = 1 / (q_scale * CODEBOOK*_SCALE)
 *
 * SIMD implementations:
 *   AVX2 (no VNNI)   — `cvtepi8_epi16 + madd_epi16` → signed*signed dot
 *                      directly; no bias trick needed.  Two halves
 *                      computed separately then combined as (1).
 *   AVX-VNNI / AVX-512 VNNI
 *                    — `vpdpbusd` only takes (u8, i8).  We feed the query
 *                      half XOR'd with 0x80 (= q + 128 in u8) as the u8
 *                      operand and the codebook (i8) as the i8 operand,
 *                      then subtract 128 * Σc once at the end.  Σc is
 *                      accumulated in a third VNNI accumulator running
 *                      `dpbusd(acc, ones_u8, c_signed)` per chunk.
 *   NEON SDOT        — signed×signed natively; no bias trick.
 *
 * Postprocess scale and the (low, high, K=256) split are shared by all
 * implementations.  Any drift here changes scoring across CPU dispatches,
 * so the constants are pinned by test/sql/pgturbohybrid_simd_parity.sql.
 */
#if defined(__aarch64__) || defined(_M_ARM64) || PGTURBOHYBRID_COMPILE_AVX2
static void
TqPrepareQuerySplit4(PgturbohybridGraphTqQuery *tq)
{
	float		qAbsMax = 0;
	float		qScale;
	int			fullDims;

	if (!tq->enabled || (tq->bits != PGTURBOHYBRID_DEFAULT_BITS && tq->bits != 2) ||
		tq->scoreMode == PGTURBOHYBRID_SCORE_L1 || tq->dimensions <= 0)
		return;

	for (int i = 0; i < tq->dimensions; i++)
		qAbsMax = Max(qAbsMax, fabsf(tq->queryValues[i]));

	if (qAbsMax < FLT_EPSILON)
		qAbsMax = FLT_EPSILON;

	tq->querySplitChunks = tq->dimensions / 16;
	tq->querySplitTailDims = tq->dimensions - (tq->querySplitChunks * 16);
	tq->querySplitLow = palloc0(Max(tq->querySplitChunks, 1) * 16);
	tq->querySplitHigh = palloc0(Max(tq->querySplitChunks, 1) * 16);
	tq->querySplitLowU8 = palloc0(Max(tq->querySplitChunks, 1) * 16);
	tq->querySplitHighU8 = palloc0(Max(tq->querySplitChunks, 1) * 16);
	memset(tq->querySplitTailLowU8, 0x80, sizeof(tq->querySplitTailLowU8));
	memset(tq->querySplitTailHighU8, 0x80, sizeof(tq->querySplitTailHighU8));
	qScale = PGTURBOHYBRID_QUERY_SPLIT_ABS_MAX / qAbsMax;
	tq->querySplitPostprocessScale = 1.0f /
		(qScale * (tq->bits == 2 ? PGTURBOHYBRID_CODEBOOK2_SCALE : PGTURBOHYBRID_CODEBOOK_SCALE));
	fullDims = tq->querySplitChunks * 16;

	for (int i = 0; i < tq->dimensions; i++)
	{
		int			qSigned;
		int			lowMod;
		int8		low;
		int8		high;

		float		scaled = tq->queryValues[i] * qScale;

		if (scaled < -PGTURBOHYBRID_QUERY_SPLIT_ABS_MAX)
			scaled = -PGTURBOHYBRID_QUERY_SPLIT_ABS_MAX;
		else if (scaled > PGTURBOHYBRID_QUERY_SPLIT_ABS_MAX)
			scaled = PGTURBOHYBRID_QUERY_SPLIT_ABS_MAX;

		qSigned = (int) lrintf(scaled);
		lowMod = qSigned % PGTURBOHYBRID_QUERY_SPLIT_HIGH_COEF;
		if (lowMod < 0)
			lowMod += PGTURBOHYBRID_QUERY_SPLIT_HIGH_COEF;
		low = (int8) (lowMod >= PGTURBOHYBRID_QUERY_SPLIT_HIGH_COEF / 2 ?
					  lowMod - PGTURBOHYBRID_QUERY_SPLIT_HIGH_COEF : lowMod);
		high = (int8) ((qSigned - (int) low) / PGTURBOHYBRID_QUERY_SPLIT_HIGH_COEF);

		if (i < fullDims)
		{
			tq->querySplitLow[i] = low;
			tq->querySplitHigh[i] = high;
			tq->querySplitLowU8[i] = (uint8) low ^ 0x80;
			tq->querySplitHighU8[i] = (uint8) high ^ 0x80;
		}
		else
		{
			int			tail = i - fullDims;

			tq->querySplitTailLow[tail] = low;
			tq->querySplitTailHigh[tail] = high;
			tq->querySplitTailLowU8[tail] = (uint8) low ^ 0x80;
			tq->querySplitTailHighU8[tail] = (uint8) high ^ 0x80;
		}
	}

	tq->querySplitEnabled = true;
}

#if PGTURBOHYBRID_COMPILE_AVX2
/*
 * Prepare the x86 unsigned-codebook query split (4-bit only).  Mirrors
 * TqPrepareQuerySplit4 but stores SIGNED 7-bit query halves (no +128 XOR --
 * the codebook is the unsigned operand here), uses HIGH_COEF 128 and
 * QUERY_ABS_MAX 8127, and precomputes the +OFFSET codebook-shift bias
 * (128 * Sum(q_signed)).  Query data is laid out as [low0..15, high0..15] per
 * 16-dim chunk so one AVX2 load is a [low|high] pair.
 */
static void
TqPrepareQueryU8Split(PgturbohybridGraphTqQuery *tq)
{
	float		qAbsMax = 0;
	float		qScale;
	int			fullDims;
	int64		sumSigned = 0;

	if (!tq->enabled || tq->bits != PGTURBOHYBRID_DEFAULT_BITS ||
		tq->scoreMode == PGTURBOHYBRID_SCORE_L1 || tq->dimensions <= 0)
		return;

	for (int i = 0; i < tq->dimensions; i++)
		qAbsMax = Max(qAbsMax, fabsf(tq->queryValues[i]));
	if (qAbsMax < FLT_EPSILON)
		qAbsMax = FLT_EPSILON;

	tq->querySplitChunks = tq->dimensions / 16;
	tq->querySplitTailDims = tq->dimensions - (tq->querySplitChunks * 16);
	tq->u8SplitData = palloc0(Max(tq->querySplitChunks, 1) * 32);
	memset(tq->u8SplitTailLow, 0, sizeof(tq->u8SplitTailLow));
	memset(tq->u8SplitTailHigh, 0, sizeof(tq->u8SplitTailHigh));
	qScale = PGTURBOHYBRID_U8_SPLIT_ABS_MAX / qAbsMax;
	tq->u8SplitPostprocessScale = 1.0f / (qScale * PGTURBOHYBRID_CODEBOOK_SCALE);
	fullDims = tq->querySplitChunks * 16;

	for (int i = 0; i < tq->dimensions; i++)
	{
		int			qSigned;
		int			lowMod;
		int8		low;
		int8		high;
		float		scaled = tq->queryValues[i] * qScale;

		if (scaled < -PGTURBOHYBRID_U8_SPLIT_ABS_MAX)
			scaled = -PGTURBOHYBRID_U8_SPLIT_ABS_MAX;
		else if (scaled > PGTURBOHYBRID_U8_SPLIT_ABS_MAX)
			scaled = PGTURBOHYBRID_U8_SPLIT_ABS_MAX;

		qSigned = (int) lrintf(scaled);
		lowMod = qSigned % PGTURBOHYBRID_U8_SPLIT_HIGH_COEF;
		if (lowMod < 0)
			lowMod += PGTURBOHYBRID_U8_SPLIT_HIGH_COEF;
		low = (int8) (lowMod >= PGTURBOHYBRID_U8_SPLIT_HIGH_COEF / 2 ?
					  lowMod - PGTURBOHYBRID_U8_SPLIT_HIGH_COEF : lowMod);
		high = (int8) ((qSigned - (int) low) / PGTURBOHYBRID_U8_SPLIT_HIGH_COEF);
		sumSigned += qSigned;

		if (i < fullDims)
		{
			int			chunk = i / 16;
			int			lane = i % 16;

			tq->u8SplitData[chunk * 32 + lane] = low;
			tq->u8SplitData[chunk * 32 + 16 + lane] = high;
		}
		else
		{
			int			tail = i - fullDims;

			tq->u8SplitTailLow[tail] = low;
			tq->u8SplitTailHigh[tail] = high;
		}
	}

	tq->u8SplitBias = (int64) PGTURBOHYBRID_U8_CODEBOOK_OFFSET * sumSigned;
	tq->u8SplitEnabled = true;
}
#endif
#endif

Size
PgturbohybridGraphElementTupleSize(Relation index, Pointer value)
{
	Size		valueSize = VARSIZE_ANY(value);

	if (PgturbohybridGraphUseTqCodes(index))
		valueSize += PGTURBOHYBRID_CODE_PAYLOAD_SIZE(((Vector *) value)->dim);

	return PGTURBOHYBRID_GRAPH_ELEMENT_TUPLE_SIZE(valueSize);
}

/*
 * Get proc
 */
FmgrInfo *
PgturbohybridGraphOptionalProcInfo(Relation index, uint16 procnum)
{
	if (!OidIsValid(index_getprocid(index, 1, procnum)))
		return NULL;

	return index_getprocinfo(index, 1, procnum);
}

/*
 * Init support functions
 */
void
PgturbohybridGraphInitSupport(PgturbohybridGraphSupport * support, Relation index)
{
	support->procinfo = index_getprocinfo(index, 1, PGTURBOHYBRID_GRAPH_DISTANCE_PROC);
	support->collation = index->rd_indcollation[0];
	support->normprocinfo = PgturbohybridGraphOptionalProcInfo(index, PGTURBOHYBRID_GRAPH_NORM_PROC);
}

/*
 * Normalize value
 */
Datum
PgturbohybridGraphNormValue(const PgturbohybridGraphTypeInfo * typeInfo, Oid collation, Datum value)
{
	return DirectFunctionCall1Coll(typeInfo->normalize, collation, value);
}

/*
 * Check if non-zero norm
 */
bool
PgturbohybridGraphCheckNorm(PgturbohybridGraphSupport * support, Datum value)
{
	return DatumGetFloat8(FunctionCall1Coll(support->normprocinfo, support->collation, value)) > 0;
}

/*
 * New buffer
 */
Buffer
PgturbohybridGraphNewBuffer(Relation index, ForkNumber forkNum)
{
	Buffer		buf = ReadBufferExtended(index, forkNum, P_NEW, RBM_NORMAL, NULL);

	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	return buf;
}

/*
 * Init page
 */
void
PgturbohybridGraphInitPageKind(Buffer buf, Page page, uint16 pageKind)
{
	PageInit(page, BufferGetPageSize(buf), sizeof(PgturbohybridGraphPageOpaqueData));
	PgturbohybridGraphPageGetOpaque(page)->nextblkno = InvalidBlockNumber;
	PgturbohybridGraphPageGetOpaque(page)->pageKind = pageKind |
		(PGTURBOHYBRID_GRAPH_GRAPH_OP_PAGE_INIT << PGTURBOHYBRID_GRAPH_PAGE_GRAPH_OP_SHIFT);
	PgturbohybridGraphPageGetOpaque(page)->page_id = PGTURBOHYBRID_GRAPH_PAGE_ID;
}

void
PgturbohybridGraphInitPage(Buffer buf, Page page)
{
	PgturbohybridGraphInitPageKind(buf, page, PGTURBOHYBRID_GRAPH_PAGE_KIND_GRAPH);
}

void
PgturbohybridGraphMarkPageGraphOp(Page page, uint16 graphOpKind)
{
	PgturbohybridGraphPageOpaque opaque = PgturbohybridGraphPageGetOpaque(page);

	if (opaque->page_id != PGTURBOHYBRID_GRAPH_PAGE_ID)
		return;

	opaque->pageKind = (opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) |
		(graphOpKind << PGTURBOHYBRID_GRAPH_PAGE_GRAPH_OP_SHIFT);
}

/*
 * Allocate a neighbor array
 */
PgturbohybridGraphNeighborArray *
PgturbohybridGraphInitNeighborArray(int lm, PgturbohybridGraphAllocator * allocator)
{
	PgturbohybridGraphNeighborArray *a = PgturbohybridGraphAlloc(allocator, PGTURBOHYBRID_GRAPH_NEIGHBOR_ARRAY_SIZE(lm));

	a->length = 0;
	a->closerSet = false;
	return a;
}

/*
 * Allocate neighbors
 */
void
PgturbohybridGraphInitNeighbors(char *base, PgturbohybridGraphElement element, int m, PgturbohybridGraphAllocator * allocator)
{
	int			level = element->level;
	PgturbohybridGraphNeighborArrayPtr *neighborList = (PgturbohybridGraphNeighborArrayPtr *) PgturbohybridGraphAlloc(allocator, sizeof(PgturbohybridGraphNeighborArrayPtr) * (level + 1));

	PgturbohybridGraphPtrStore(base, element->neighbors, neighborList);

	for (int lc = 0; lc <= level; lc++)
		PgturbohybridGraphPtrStore(base, neighborList[lc], PgturbohybridGraphInitNeighborArray(PgturbohybridGraphGetLayerM(m, lc), allocator));
}

/*
 * Allocate memory from the allocator
 */
void *
PgturbohybridGraphAlloc(PgturbohybridGraphAllocator * allocator, Size size)
{
	if (allocator)
		return (*(allocator)->alloc) (size, (allocator)->state);

	return palloc(size);
}

/*
 * Allocate an element
 */
PgturbohybridGraphElement
PgturbohybridGraphInitElement(char *base, ItemPointer heaptid, int m, double ml, int maxLevel, PgturbohybridGraphAllocator * allocator)
{
	PgturbohybridGraphElement element = PgturbohybridGraphAlloc(allocator, sizeof(PgturbohybridGraphElementData));

	int			level = (int) (-log(RandomDouble()) * ml);

	/* Cap level */
	if (level > maxLevel)
		level = maxLevel;

	element->heaptidsLength = 0;
	PgturbohybridGraphAddHeapTid(element, heaptid);

	element->level = level;
	element->deleted = 0;
	/* Start at one to make it easier to find issues */
	element->version = 1;

	PgturbohybridGraphInitNeighbors(base, element, m, allocator);

	PgturbohybridGraphPtrStore(base, element->value, (char *) NULL);

	return element;
}

/*
 * Add a heap TID to an element
 */
void
PgturbohybridGraphAddHeapTid(PgturbohybridGraphElement element, ItemPointer heaptid)
{
	element->heaptids[element->heaptidsLength++] = *heaptid;
}

/*
 * Allocate an element from block and offset numbers
 */
PgturbohybridGraphElement
PgturbohybridGraphInitElementFromBlock(BlockNumber blkno, OffsetNumber offno)
{
	PgturbohybridGraphElement element = palloc(sizeof(PgturbohybridGraphElementData));
	char	   *base = NULL;

	element->blkno = blkno;
	element->offno = offno;
	PgturbohybridGraphPtrStore(base, element->neighbors, (PgturbohybridGraphNeighborArrayPtr *) NULL);
	PgturbohybridGraphPtrStore(base, element->value, (char *) NULL);
	return element;
}

/*
 * Get the metapage info
 */
void
PgturbohybridGraphGetMetaPageInfo(Relation index, int *m, PgturbohybridGraphElement * entryPoint)
{
	Buffer		buf;
	Page		page;
	PgturbohybridGraphMetaPage metap;

	buf = ReadBuffer(index, PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	metap = PgturbohybridGraphPageGetMeta(page);

	if (unlikely(metap->magicNumber != PGTURBOHYBRID_GRAPH_MAGIC_NUMBER))
		elog(ERROR, "pgturbohybrid index is not valid");

	if (m != NULL)
		*m = metap->m;

	if (entryPoint != NULL)
	{
		if (BlockNumberIsValid(metap->entryBlkno))
		{
			*entryPoint = PgturbohybridGraphInitElementFromBlock(metap->entryBlkno, metap->entryOffno);
			(*entryPoint)->level = metap->entryLevel;
		}
		else
			*entryPoint = NULL;
	}

	UnlockReleaseBuffer(buf);
}

int
PgturbohybridGraphGetMetaPageStorageKind(Relation index)
{
	Buffer		buf;
	Page		page;
	PgturbohybridGraphMetaPage metap;
	int			storageKind;

	buf = ReadBuffer(index, PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	metap = PgturbohybridGraphPageGetMeta(page);

	if (unlikely(metap->magicNumber != PGTURBOHYBRID_GRAPH_MAGIC_NUMBER))
		elog(ERROR, "pgturbohybrid index is not valid");

	storageKind = metap->storageKind;

	UnlockReleaseBuffer(buf);

	return storageKind;
}

/*
 * Get the entry point
 */
PgturbohybridGraphElement
PgturbohybridGraphGetEntryPoint(Relation index)
{
	PgturbohybridGraphElement entryPoint;

	PgturbohybridGraphGetMetaPageInfo(index, NULL, &entryPoint);

	return entryPoint;
}

/*
 * Update the metapage info
 */
static void
PgturbohybridGraphUpdateMetaPageInfo(Page page, int updateEntry, PgturbohybridGraphElement entryPoint, BlockNumber insertPage)
{
	PgturbohybridGraphMetaPage metap = PgturbohybridGraphPageGetMeta(page);

	if (updateEntry)
	{
		if (entryPoint == NULL)
		{
			metap->entryBlkno = InvalidBlockNumber;
			metap->entryOffno = InvalidOffsetNumber;
			metap->entryLevel = -1;
		}
		else if (entryPoint->level > metap->entryLevel || updateEntry == PGTURBOHYBRID_GRAPH_UPDATE_ENTRY_ALWAYS)
		{
			metap->entryBlkno = entryPoint->blkno;
			metap->entryOffno = entryPoint->offno;
			metap->entryLevel = entryPoint->level;
		}
	}

	if (BlockNumberIsValid(insertPage))
		metap->insertPage = insertPage;

	PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_META_UPDATE);
}

/*
 * Update the metapage
 */
void
PgturbohybridGraphUpdateMetaPage(Relation index, int updateEntry, PgturbohybridGraphElement entryPoint, BlockNumber insertPage, ForkNumber forkNum, bool building)
{
	Buffer		buf;
	Page		page;
	GenericXLogState *state;

	buf = ReadBufferExtended(index, forkNum, PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	if (building)
	{
		state = NULL;
		page = BufferGetPage(buf);
	}
	else
	{
		state = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(state, buf, 0);
	}

	PgturbohybridGraphUpdateMetaPageInfo(page, updateEntry, entryPoint, insertPage);

	if (building)
		MarkBufferDirty(buf);
	else
		GenericXLogFinish(state);
	UnlockReleaseBuffer(buf);

	if (!building)
		PgturbohybridGraphLogGraphWalRecord(index, forkNum, PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO, PGTURBOHYBRID_GRAPH_GRAPH_OP_META_UPDATE);
}

/*
 * Form index value
 */
bool
PgturbohybridGraphFormIndexValue(Datum *out, Datum *values, bool *isnull, const PgturbohybridGraphTypeInfo * typeInfo, PgturbohybridGraphSupport * support)
{
	/* Detoast once for all calls */
	Datum		value = PointerGetDatum(PG_DETOAST_DATUM(values[0]));

	/* Check value */
	if (typeInfo->checkValue != NULL)
		typeInfo->checkValue(DatumGetPointer(value));

	/* Normalize if needed */
	if (support->normprocinfo != NULL)
	{
		if (!PgturbohybridGraphCheckNorm(support, value))
			return false;

		value = PgturbohybridGraphNormValue(typeInfo, support->collation, value);
	}

	*out = value;

	return true;
}

/*
 * Set element tuple, except for neighbor info
 */
void
PgturbohybridGraphSetElementTuple(Relation index, char *base, PgturbohybridGraphElementTuple etup, PgturbohybridGraphElement element)
{
	Pointer		valuePtr = PgturbohybridGraphPtrAccess(base, element->value);
	Size		valueSize = VARSIZE_ANY(valuePtr);

	etup->type = PGTURBOHYBRID_GRAPH_ELEMENT_TUPLE_TYPE;
	etup->level = element->level;
	etup->deleted = 0;
	etup->version = element->version;
	for (int i = 0; i < PGTURBOHYBRID_GRAPH_HEAPTIDS; i++)
	{
		if (i < element->heaptidsLength)
			etup->heaptids[i] = element->heaptids[i];
		else
			ItemPointerSetInvalid(&etup->heaptids[i]);
	}
	memcpy(&etup->data, valuePtr, valueSize);

	if (PgturbohybridGraphUseTqCodes(index))
	{
		uint8	   *code = (uint8 *) ((char *) &etup->data + valueSize);
		float		scale = TqEncodeVector((Vector *) valuePtr, code);

		memcpy(code + PGTURBOHYBRID_CODE_SCALE_OFFSET(((Vector *) valuePtr)->dim), &scale, sizeof(float));
	}
}

/*
 * Set neighbor tuple
 */
void
PgturbohybridGraphSetNeighborTuple(char *base, PgturbohybridGraphNeighborTuple ntup, PgturbohybridGraphElement e, int m)
{
	int			idx = 0;

	ntup->type = PGTURBOHYBRID_GRAPH_NEIGHBOR_TUPLE_TYPE;

	for (int lc = e->level; lc >= 0; lc--)
	{
		PgturbohybridGraphNeighborArray *neighbors = PgturbohybridGraphGetNeighbors(base, e, lc);
		int			lm = PgturbohybridGraphGetLayerM(m, lc);

		for (int i = 0; i < lm; i++)
		{
			ItemPointer indextid = &ntup->indextids[idx++];

			if (i < neighbors->length)
			{
				PgturbohybridGraphCandidate *hc = &neighbors->items[i];
				PgturbohybridGraphElement hce = PgturbohybridGraphPtrAccess(base, hc->element);

				ItemPointerSet(indextid, hce->blkno, hce->offno);
			}
			else
				ItemPointerSetInvalid(indextid);
		}
	}

	ntup->count = idx;
	ntup->version = e->version;
}

/*
 * Load an element from a tuple
 */
void
PgturbohybridGraphLoadElementFromTuple(PgturbohybridGraphElement element, PgturbohybridGraphElementTuple etup, bool loadHeaptids, bool loadVec)
{
	element->level = etup->level;
	element->deleted = etup->deleted;
	element->version = etup->version;
	element->neighborPage = ItemPointerGetBlockNumber(&etup->neighbortid);
	element->neighborOffno = ItemPointerGetOffsetNumber(&etup->neighbortid);
	element->heaptidsLength = 0;

	if (loadHeaptids)
	{
		for (int i = 0; i < PGTURBOHYBRID_GRAPH_HEAPTIDS; i++)
		{
			/* Can stop at first invalid */
			if (!ItemPointerIsValid(&etup->heaptids[i]))
				break;

			PgturbohybridGraphAddHeapTid(element, &etup->heaptids[i]);
		}
	}

	if (loadVec)
	{
		char	   *base = NULL;
		Datum		value = datumCopy(PointerGetDatum(&etup->data), false, -1);

		PgturbohybridGraphPtrStore(base, element->value, (char *) DatumGetPointer(value));
	}
}

/*
 * Calculate the distance between values
 */
static inline double
PgturbohybridGraphGetDistance(Datum a, Datum b, PgturbohybridGraphSupport * support)
{
	return DatumGetFloat8(FunctionCall2Coll(support->procinfo, support->collation, a, b));
}

static bool
TqGetTupleCode(PgturbohybridGraphElementTuple etup, Size tupleSize, const uint8 **code, int *dimensions, float *scale)
{
	Vector	   *vector = &etup->data;
	Size		valueSize = VARSIZE_ANY(vector);
	int			dim = vector->dim;
	Size		requiredSize = PGTURBOHYBRID_GRAPH_ELEMENT_TUPLE_SIZE(valueSize + PGTURBOHYBRID_CODE_PAYLOAD_SIZE(dim));

	if (tupleSize < requiredSize)
		return false;

	*code = (const uint8 *) ((char *) vector + valueSize);
	*dimensions = dim;
	memcpy(scale, *code + PGTURBOHYBRID_CODE_SCALE_OFFSET(dim), sizeof(float));

	if (*scale <= 0 || isnan(*scale))
		*scale = 1;

	return true;
}

static TqScoreMode
TqGetScoreMode(PgturbohybridGraphSupport * support)
{
	char	   *procname = get_func_name(support->procinfo->fn_oid);
	TqScoreMode mode = PGTURBOHYBRID_SCORE_L2;

	if (procname == NULL)
		return mode;

	if ((strcmp(procname, "turbohybrid_negative_inner_product") == 0 ||
		 strcmp(procname, "turbohybrid_vector_negative_inner_product") == 0) &&
		support->normprocinfo != NULL)
		mode = PGTURBOHYBRID_SCORE_COSINE;
	else if (strcmp(procname, "turbohybrid_negative_inner_product") == 0 ||
			 strcmp(procname, "turbohybrid_vector_negative_inner_product") == 0)
		mode = PGTURBOHYBRID_SCORE_IP;
	else if (strcmp(procname, "l1_distance") == 0)
		mode = PGTURBOHYBRID_SCORE_L1;

	pfree(procname);

	return mode;
}

static void
PgturbohybridGraphPrepareTqQueryInternal(Relation index, PgturbohybridGraphSupport * support, Datum value,
						   PgturbohybridGraphTqQuery *tq, bool loadCorrection,
						   bool prepareSplit, const float *overrideEcShift,
						   const float *overrideEcScale, int tqBitsOverride)
{
	Vector	   *query;
	const float *ecShift = NULL;
	const float *ecScale = NULL;

	memset(tq, 0, sizeof(PgturbohybridGraphTqQuery));

	if (!PgturbohybridGraphTqSupportsPackedCodes(index) || DatumGetPointer(value) == NULL)
		return;

	query = (Vector *) DatumGetPointer(value);
	tq->dimensions = query->dim;
	tq->bits = tqBitsOverride > 0 ? tqBitsOverride : PgturbohybridGraphGetTqBits(index);
	tq->lutWidth = 1 << tq->bits;
	tq->codeBytes = TqCodeSizeForBits(query->dim, tq->bits);
	tq->scoreMode = TqGetScoreMode(support);
	tq->scoringKernel = TqSelectScoringKernel();
	tq->rawQueryValues = query->x;
	tq->queryValues = palloc(sizeof(float) * query->dim);
	TqRotateVectorFloat(query->x, query->dim, tq->queryValues);
	for (int i = 0; i < query->dim; i++)
		tq->queryNorm += (double) query->x[i] * query->x[i];

	if (overrideEcShift != NULL && overrideEcScale != NULL)
	{
		ecShift = overrideEcShift;
		ecScale = overrideEcScale;
		tq->ecShift = (float *) overrideEcShift;
		tq->ecScale = (float *) overrideEcScale;
	}
	else if (loadCorrection && PgturbohybridGraphUseTqNativeGraph(index) &&
			 PgturbohybridGraphLoadCorrection(index, query->dim, &tq->ecShift, &tq->ecScale))
	{
		ecShift = tq->ecShift;
		ecScale = tq->ecScale;
	}

	if (ecShift != NULL && ecScale != NULL)
	{
		for (int i = 0; i < query->dim; i++)
		{
			float		scale = ecScale[i];

			tq->ecCorrection += (double) tq->queryValues[i] * (double) -ecShift[i];
			if (fabsf(scale) > FLT_EPSILON)
				tq->queryValues[i] /= scale;
		}
	}
#if PGTURBOHYBRID_GRAPH_ENABLE_SYMMETRIC_I8_DOT
	tq->queryI8 = palloc(sizeof(int8) * query->dim);
	tq->queryScale = TqEncodeQueryInt8(query, tq->queryI8,
									   &tq->queryCodeNorm);
#endif
	tq->enabled = true;
	TqPrepareQuerySignBits(tq);
	TqPrepareQueryAsymBit1(tq);
#if defined(__aarch64__) || defined(_M_ARM64) || PGTURBOHYBRID_COMPILE_AVX2
	if (prepareSplit)
	{
#if PGTURBOHYBRID_COMPILE_AVX2
		/*
		 * Build exactly one query-split representation.  The unsigned-codebook
		 * (u8) split is the x86 default for 4-bit; otherwise (2-bit, forced
		 * signed, no AVX2) fall back to the signed split, which also serves as
		 * the LUT-skip fallback.
		 */
		if (PgturbohybridGraphTqUseU8Split(tq))
			TqPrepareQueryU8Split(tq);
		else
#endif
			TqPrepareQuerySplit4(tq);
	}
#endif

	/*
	 * Build the LUT-gather scoring table only when the integer query-split
	 * scorers will not handle this query.  When query split is active every
	 * scoring path (scan and build) tries QuerySplit before the LUT fallback,
	 * so tq->lut is never read -- skip allocating and filling it.  tq->lut
	 * stays NULL (zeroed by the memset above) in that case.
	 */
	if (!PgturbohybridGraphTqQuerySplitActive(tq))
	{
		tq->lut = palloc(PGTURBOHYBRID_LUT_SIZE(query->dim));
		TqBuildQueryLut(tq);
	}
}

void
PgturbohybridGraphPrepareTqQuery(Relation index, PgturbohybridGraphSupport * support, Datum value, PgturbohybridGraphTqQuery *tq)
{
	PgturbohybridGraphPrepareTqQueryInternal(index, support, value, tq, true, true, NULL, NULL, 0);
}

void
PgturbohybridGraphPrepareTqQueryWithBits(Relation index, PgturbohybridGraphSupport * support, Datum value,
						   PgturbohybridGraphTqQuery *tq, int tqBits)
{
	PgturbohybridGraphPrepareTqQueryInternal(index, support, value, tq, true, true, NULL, NULL,
							   tqBits);
}

void
PgturbohybridGraphPrepareTqBuildQuery(Relation index, PgturbohybridGraphSupport * support, Datum value, PgturbohybridGraphTqQuery *tq)
{
	PgturbohybridGraphPrepareTqQueryInternal(index, support, value, tq, false, true, NULL, NULL, 0);
}

void
PgturbohybridGraphPrepareTqBuildQueryWithCorrection(Relation index, PgturbohybridGraphSupport * support,
									  Datum value, PgturbohybridGraphTqQuery *tq,
									  const float *ecShift,
									  const float *ecScale)
{
	PgturbohybridGraphPrepareTqQueryInternal(index, support, value, tq, false, true,
							   ecShift, ecScale, 0);
}

static double
TqCodeDistanceScalar(const PgturbohybridGraphTqQuery *tq, const uint8 *valueCode, float valueScale)
{
	double		dot = 0;
	double		codeNorm = 0;
	double		distance;
	int			dim = tq->dimensions;
	double		dimSqrt = TqDimSqrt(dim);
	TqScoreMode mode = (TqScoreMode) tq->scoreMode;

	if (mode == PGTURBOHYBRID_SCORE_L1)
	{
		distance = 0;
		for (int i = 0; i < dim; i++)
		{
			int			vc = TqGetCodeComponentBits(valueCode, i, tq->bits);
			double		vv = valueScale * TqGetCodeCenterBits(vc, tq->bits) / dimSqrt;

			distance += fabs(tq->rawQueryValues[i] - vv);
		}

		return distance;
	}

	for (int i = 0; i < dim; i++)
	{
		int			vc = TqGetCodeComponentBits(valueCode, i, tq->bits);
		double		vv = TqGetCodeCenterBits(vc, tq->bits);

		dot += tq->lut[(i * PGTURBOHYBRID_LUT_WIDTH) + vc];
		codeNorm += vv * vv;
	}

	dot += tq->ecCorrection;

	if (mode == PGTURBOHYBRID_SCORE_IP)
		return -(valueScale * dot / dimSqrt);

	if (mode == PGTURBOHYBRID_SCORE_L2)
	{
		(void) codeNorm;
		distance = tq->queryNorm + ((double) valueScale * valueScale) -
			(2 * valueScale * dot / dimSqrt);

		return distance < 0 ? 0 : distance;
	}

	if (mode == PGTURBOHYBRID_SCORE_COSINE)
	{
		(void) codeNorm;
		if (tq->queryNorm == 0 || valueScale == 0)
			return 1;

		return 1 - (dot / (sqrt(tq->queryNorm) * dimSqrt));
	}

	return 1;
}

#if PGTURBOHYBRID_COMPILE_AVX2
static inline double PGTURBOHYBRID_AVX2_TARGET
TqHorizontalSumAvx(__m256 v)
{
	float		s[8];

	_mm256_storeu_ps(s, v);
	return (double) s[0] + s[1] + s[2] + s[3] + s[4] + s[5] + s[6] + s[7];
}

static double PGTURBOHYBRID_AVX2_TARGET
TqCodeDistanceAvx2(const PgturbohybridGraphTqQuery *tq, const uint8 *valueCode, float valueScale)
{
	__m256		acc = _mm256_setzero_ps();
	__m256		norm = _mm256_setzero_ps();
	int			i = 0;
	int			dim = tq->dimensions;
	double		dimSqrt = TqDimSqrt(dim);
	TqScoreMode mode = (TqScoreMode) tq->scoreMode;
	double		dot;
	double		codeNorm;
	int			idx[8];
	int			codes[8];

	if (mode == PGTURBOHYBRID_SCORE_L1)
		return TqCodeDistanceScalar(tq, valueCode, valueScale);

	for (; i + 8 <= dim; i += 8)
	{
		for (int j = 0; j < 8; j++)
		{
			int			code = TqGetCodeComponentBits(valueCode, i + j, tq->bits);

			codes[j] = code;
			idx[j] = ((i + j) * PGTURBOHYBRID_LUT_WIDTH) + code;
		}

		acc = _mm256_add_ps(acc,
							_mm256_i32gather_ps(tq->lut,
												_mm256_loadu_si256((const __m256i *) idx), 4));

		if (mode == PGTURBOHYBRID_SCORE_COSINE || mode == PGTURBOHYBRID_SCORE_L2)
		{
			__m256		centers;

			if (tq->bits == PGTURBOHYBRID_DEFAULT_BITS)
				centers = _mm256_i32gather_ps(TqCodeCenters,
											  _mm256_loadu_si256((const __m256i *) codes), 4);
			else
			{
				float		centerValues[8];

				for (int j = 0; j < 8; j++)
					centerValues[j] = TqGetCodeCenterBits(codes[j], tq->bits);
				centers = _mm256_loadu_ps(centerValues);
			}

			norm = _mm256_add_ps(norm, _mm256_mul_ps(centers, centers));
		}
	}

	dot = TqHorizontalSumAvx(acc);
	codeNorm = TqHorizontalSumAvx(norm);
	dot += tq->ecCorrection;

	if (mode == PGTURBOHYBRID_SCORE_COSINE)
	{
		for (; i < dim; i++)
		{
			int			code = TqGetCodeComponentBits(valueCode, i, tq->bits);

			dot += tq->lut[(i * PGTURBOHYBRID_LUT_WIDTH) + code];
		}

		if (tq->queryNorm == 0 || valueScale == 0)
			return 1;

		return 1 - (dot / (sqrt(tq->queryNorm) * dimSqrt));
	}

	for (; i < dim; i++)
	{
		int			code = TqGetCodeComponentBits(valueCode, i, tq->bits);
		double		vv = TqGetCodeCenterBits(code, tq->bits);

		dot += tq->lut[(i * PGTURBOHYBRID_LUT_WIDTH) + code];
		if (mode == PGTURBOHYBRID_SCORE_L2)
			codeNorm += vv * vv;
	}

	if (mode == PGTURBOHYBRID_SCORE_IP)
		return -(valueScale * dot / dimSqrt);

	(void) codeNorm;
	return Max(tq->queryNorm + ((double) valueScale * valueScale) -
			   (2 * valueScale * dot / dimSqrt), 0);
}
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
static inline double
TqHorizontalSumNeon(float32x4_t v)
{
#if defined(__aarch64__) || defined(_M_ARM64)
	return (double) vaddvq_f32(v);
#else
	float		s[4];

	vst1q_f32(s, v);
	return (double) s[0] + s[1] + s[2] + s[3];
#endif
}

static double
TqCodeDistanceNeon(const PgturbohybridGraphTqQuery *tq, const uint8 *valueCode, float valueScale)
{
	float32x4_t acc = vdupq_n_f32(0);
	float32x4_t norm = vdupq_n_f32(0);
	int			i = 0;
	int			dim = tq->dimensions;
	double		dimSqrt = TqDimSqrt(dim);
	TqScoreMode mode = (TqScoreMode) tq->scoreMode;
	double		dot;
	double		codeNorm;

	if (mode == PGTURBOHYBRID_SCORE_L1)
		return TqCodeDistanceScalar(tq, valueCode, valueScale);

	for (; i + 4 <= dim; i += 4)
	{
		float		scores[4];

		for (int j = 0; j < 4; j++)
		{
			int			code = TqGetCodeComponentBits(valueCode, i + j, tq->bits);

			scores[j] = tq->lut[((i + j) * PGTURBOHYBRID_LUT_WIDTH) + code];
		}

		acc = vaddq_f32(acc, vld1q_f32(scores));

		if (mode == PGTURBOHYBRID_SCORE_COSINE || mode == PGTURBOHYBRID_SCORE_L2)
		{
			float		centers[4];

			for (int j = 0; j < 4; j++)
				centers[j] = TqGetCodeCenterBits(TqGetCodeComponentBits(valueCode, i + j, tq->bits),
												 tq->bits);

			norm = vfmaq_f32(norm, vld1q_f32(centers), vld1q_f32(centers));
		}
	}

	dot = TqHorizontalSumNeon(acc);
	codeNorm = TqHorizontalSumNeon(norm);
	dot += tq->ecCorrection;

	if (mode == PGTURBOHYBRID_SCORE_COSINE)
	{
		for (; i < dim; i++)
		{
			int			code = TqGetCodeComponentBits(valueCode, i, tq->bits);

			dot += tq->lut[(i * PGTURBOHYBRID_LUT_WIDTH) + code];
		}

		if (tq->queryNorm == 0 || valueScale == 0)
			return 1;

		return 1 - (dot / (sqrt(tq->queryNorm) * dimSqrt));
	}

	for (; i < dim; i++)
	{
		int			code = TqGetCodeComponentBits(valueCode, i, tq->bits);
		double		vv = TqGetCodeCenterBits(code, tq->bits);

		dot += tq->lut[(i * PGTURBOHYBRID_LUT_WIDTH) + code];
		if (mode == PGTURBOHYBRID_SCORE_L2)
			codeNorm += vv * vv;
	}

	if (mode == PGTURBOHYBRID_SCORE_IP)
		return -(valueScale * dot / dimSqrt);

	(void) codeNorm;
	return Max(tq->queryNorm + ((double) valueScale * valueScale) -
			   (2 * valueScale * dot / dimSqrt), 0);
}
#endif

PGTURBOHYBRID_TARGET_CLONES double
TqCodeDistance(const PgturbohybridGraphTqQuery *tq, const uint8 *valueCode, float valueScale)
{
	double		querySplitDistance;

	/*
	 * Prefer the integer query-split SIMD scorers for 4-bit and 2-bit packed
	 * codes (dim >= 1024, not L1).  This keeps dense scan scoring off the
	 * AVX2 per-dimension LUT-gather path (TqCodeDistanceAvx2), which is far
	 * slower per code.  Falls through to the LUT/scalar kernels when query
	 * split is unavailable for this query.
	 */
	if (PgturbohybridGraphTqCodeQuerySplitDistance(tq, valueCode, valueScale,
												   &querySplitDistance))
		return querySplitDistance;

	switch ((TqScoringKernel) tq->scoringKernel)
	{
#if PGTURBOHYBRID_COMPILE_AVX2
		case PGTURBOHYBRID_SCORING_AVX2:
		case PGTURBOHYBRID_SCORING_AVX512VNNI:
		case PGTURBOHYBRID_SCORING_AVXVNNI:
			return TqCodeDistanceAvx2(tq, valueCode, valueScale);
#elif defined(__aarch64__) || defined(_M_ARM64)
		case PGTURBOHYBRID_SCORING_ARM_I8MM:
		case PGTURBOHYBRID_SCORING_NEON:
			return TqCodeDistanceNeon(tq, valueCode, valueScale);
#endif
		case PGTURBOHYBRID_SCORING_SCALAR:
		default:
			return TqCodeDistanceScalar(tq, valueCode, valueScale);
	}
}

static bool
TqGetApproximateDistance(PgturbohybridGraphElementTuple etup, Size tupleSize, PgturbohybridGraphQuery *q,
						 const PgturbohybridGraphTqQuery *tq, double *distance)
{
	const uint8 *valueCode;
	int			valueDimensions;
	float		valueScale;

	if (tq == NULL || !tq->enabled || DatumGetPointer(q->value) == NULL)
		return false;

	if (!TqGetTupleCode(etup, tupleSize, &valueCode, &valueDimensions, &valueScale) ||
		valueDimensions != tq->dimensions)
		return false;

	*distance = TqCodeDistance(tq, valueCode, valueScale);
	return true;
}

/*
 * Load an element and optionally get its distance from q
 */
static void
PgturbohybridGraphLoadElementImpl(BlockNumber blkno, OffsetNumber offno, double *distance, PgturbohybridGraphQuery * q, Relation index, PgturbohybridGraphSupport * support, bool loadVec, double *maxDistance, PgturbohybridGraphElement * element, bool useTqCodeScoring, const PgturbohybridGraphTqQuery *tq, int64 *scoredCodes)
{
	Buffer		buf;
	Page		page;
	ItemId		iid;
	PgturbohybridGraphElementTuple etup;
	Size		tupleSize;

	/* Read vector */
	buf = ReadBuffer(index, blkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);

	iid = PageGetItemId(page, offno);
	etup = (PgturbohybridGraphElementTuple) PageGetItem(page, iid);
	tupleSize = ItemIdGetLength(iid);

	Assert(PgturbohybridGraphIsElementTuple(etup));

	/* Calculate distance */
	if (distance != NULL)
	{
		if (DatumGetPointer(q->value) == NULL)
			*distance = 0;
		else if (useTqCodeScoring &&
				 TqGetApproximateDistance(etup, tupleSize, q, tq, distance))
		{
			if (scoredCodes != NULL)
				(*scoredCodes)++;
		}
		else
			*distance = PgturbohybridGraphGetDistance(q->value, PointerGetDatum(&etup->data), support);
	}

	/* Load element */
	if (distance == NULL || maxDistance == NULL || *distance < *maxDistance)
	{
		if (*element == NULL)
			*element = PgturbohybridGraphInitElementFromBlock(blkno, offno);

		PgturbohybridGraphLoadElementFromTuple(*element, etup, true, loadVec);
	}

	UnlockReleaseBuffer(buf);
}

/*
 * Load an element and optionally get its distance from q
 */
void
PgturbohybridGraphLoadElement(PgturbohybridGraphElement element, double *distance, PgturbohybridGraphQuery * q, Relation index, PgturbohybridGraphSupport * support, bool loadVec, double *maxDistance)
{
	PgturbohybridGraphLoadElementImpl(element->blkno, element->offno, distance, q, index, support, loadVec, maxDistance, &element, false, NULL, NULL);
}

static int
CompareSearchCandidateIndexTids(const ListCell *a, const ListCell *b)
{
	char	   *base = NULL;
	PgturbohybridGraphSearchCandidate *sca = lfirst(a);
	PgturbohybridGraphSearchCandidate *scb = lfirst(b);
	PgturbohybridGraphElement ea = PgturbohybridGraphPtrAccess(base, sca->element);
	PgturbohybridGraphElement eb = PgturbohybridGraphPtrAccess(base, scb->element);

	if (ea->blkno < eb->blkno)
		return -1;
	if (ea->blkno > eb->blkno)
		return 1;
	if (ea->offno < eb->offno)
		return -1;
	if (ea->offno > eb->offno)
		return 1;

	return 0;
}

int64
PgturbohybridGraphRescoreSearchCandidates(Relation index, PgturbohybridGraphSupport *support, PgturbohybridGraphQuery *q, List *items)
{
	char	   *base = NULL;
	List	   *sorted;
	ListCell   *lc;
	BlockNumber currentBlkno = InvalidBlockNumber;
	Buffer		buf = InvalidBuffer;
	Page		page = NULL;
	int64		pages = 0;

	if (items == NIL)
		return 0;

	sorted = list_copy(items);
	list_sort(sorted, CompareSearchCandidateIndexTids);

	foreach(lc, sorted)
	{
		PgturbohybridGraphSearchCandidate *sc = lfirst(lc);
		PgturbohybridGraphElement element = PgturbohybridGraphPtrAccess(base, sc->element);
		PgturbohybridGraphElementTuple etup;

		if (element->blkno != currentBlkno)
		{
			if (BufferIsValid(buf))
				UnlockReleaseBuffer(buf);

			buf = ReadBuffer(index, element->blkno);
			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);
			currentBlkno = element->blkno;
			pages++;
		}

		etup = (PgturbohybridGraphElementTuple) PageGetItem(page, PageGetItemId(page, element->offno));
		Assert(PgturbohybridGraphIsElementTuple(etup));

		if (DatumGetPointer(q->value) == NULL)
			sc->distance = 0;
		else
			sc->distance = PgturbohybridGraphGetDistance(q->value, PointerGetDatum(&etup->data), support);
	}

	if (BufferIsValid(buf))
		UnlockReleaseBuffer(buf);

	return pages;
}

/*
 * Get the distance for an element
 */
static double
GetElementDistance(char *base, PgturbohybridGraphElement element, PgturbohybridGraphQuery * q, PgturbohybridGraphSupport * support)
{
	Datum		value = PgturbohybridGraphGetValue(base, element);

	return PgturbohybridGraphGetDistance(q->value, value, support);
}

/*
 * Allocate a search candidate
 */
static PgturbohybridGraphSearchCandidate *
PgturbohybridGraphInitSearchCandidate(char *base, PgturbohybridGraphElement element, double distance)
{
	PgturbohybridGraphSearchCandidate *sc = palloc(sizeof(PgturbohybridGraphSearchCandidate));

	PgturbohybridGraphPtrStore(base, sc->element, element);
	sc->distance = distance;
	return sc;
}

/*
 * Create a candidate for the entry point
 */
PgturbohybridGraphSearchCandidate *
PgturbohybridGraphEntryCandidate(char *base, PgturbohybridGraphElement entryPoint, PgturbohybridGraphQuery * q, Relation index, PgturbohybridGraphSupport * support, bool loadVec)
{
	bool		inMemory = index == NULL;
	double		distance;

	if (inMemory)
		distance = GetElementDistance(base, entryPoint, q, support);
	else
		PgturbohybridGraphLoadElement(entryPoint, &distance, q, index, support, loadVec, NULL);

	return PgturbohybridGraphInitSearchCandidate(base, entryPoint, distance);
}

/*
 * Compare candidate distances
 */
static int
CompareNearestCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (PgturbohybridGraphGetSearchCandidateConst(c_node, a)->distance < PgturbohybridGraphGetSearchCandidateConst(c_node, b)->distance)
		return 1;

	if (PgturbohybridGraphGetSearchCandidateConst(c_node, a)->distance > PgturbohybridGraphGetSearchCandidateConst(c_node, b)->distance)
		return -1;

	return 0;
}

/*
 * Compare discarded candidate distances
 */
static int
CompareNearestDiscardedCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (PgturbohybridGraphGetSearchCandidateConst(w_node, a)->distance < PgturbohybridGraphGetSearchCandidateConst(w_node, b)->distance)
		return 1;

	if (PgturbohybridGraphGetSearchCandidateConst(w_node, a)->distance > PgturbohybridGraphGetSearchCandidateConst(w_node, b)->distance)
		return -1;

	return 0;
}

/*
 * Compare candidate distances
 */
static int
CompareFurthestCandidates(const pairingheap_node *a, const pairingheap_node *b, void *arg)
{
	if (PgturbohybridGraphGetSearchCandidateConst(w_node, a)->distance < PgturbohybridGraphGetSearchCandidateConst(w_node, b)->distance)
		return -1;

	if (PgturbohybridGraphGetSearchCandidateConst(w_node, a)->distance > PgturbohybridGraphGetSearchCandidateConst(w_node, b)->distance)
		return 1;

	return 0;
}

/*
 * Init visited
 */
static inline void
InitVisited(char *base, visited_hash * v, bool inMemory, int ef, int m)
{
	if (!inMemory)
		v->tids = tidhash_create(CurrentMemoryContext, ef * m * 2, NULL);
	else if (base != NULL)
		v->offsets = offsethash_create(CurrentMemoryContext, ef * m * 2, NULL);
	else
		v->pointers = pointerhash_create(CurrentMemoryContext, ef * m * 2, NULL);
}

/*
 * Add to visited
 */
static inline void
AddToVisited(char *base, visited_hash * v, PgturbohybridGraphElementPtr elementPtr, bool inMemory, bool *found)
{
	if (!inMemory)
	{
		PgturbohybridGraphElement element = PgturbohybridGraphPtrAccess(base, elementPtr);
		ItemPointerData indextid;

		ItemPointerSet(&indextid, element->blkno, element->offno);
		tidhash_insert(v->tids, indextid, found);
	}
	else if (base != NULL)
	{
		PgturbohybridGraphElement element = PgturbohybridGraphPtrAccess(base, elementPtr);

		offsethash_insert_hash(v->offsets, PgturbohybridGraphPtrOffset(elementPtr), element->hash, found);
	}
	else
	{
		PgturbohybridGraphElement element = PgturbohybridGraphPtrAccess(base, elementPtr);

		pointerhash_insert_hash(v->pointers, (uintptr_t) PgturbohybridGraphPtrPointer(elementPtr), element->hash, found);
	}
}

/*
 * Count element towards ef
 */
static inline bool
CountElement(PgturbohybridGraphElement skipElement, PgturbohybridGraphElement e)
{
	if (skipElement == NULL)
		return true;

	/* Ensure does not access heaptidsLength during in-memory build */
	pg_memory_barrier();

	/* Keep scan-build happy on Mac x86-64 */
	Assert(e);

	return e->heaptidsLength != 0;
}

/*
 * Load unvisited neighbors from memory
 */
static void
PgturbohybridGraphLoadUnvisitedFromMemory(char *base, PgturbohybridGraphElement element, PgturbohybridGraphUnvisited * unvisited, int *unvisitedLength, visited_hash * v, int lc, PgturbohybridGraphNeighborArray * localNeighborhood, Size neighborhoodSize)
{
	/* Get the neighborhood at layer lc */
	PgturbohybridGraphNeighborArray *neighborhood = PgturbohybridGraphGetNeighbors(base, element, lc);

	/* Copy neighborhood to local memory */
	LWLockAcquire(&element->lock, LW_SHARED);
	memcpy(localNeighborhood, neighborhood, neighborhoodSize);
	LWLockRelease(&element->lock);

	*unvisitedLength = 0;

	for (int i = 0; i < localNeighborhood->length; i++)
	{
		PgturbohybridGraphCandidate *hc = &localNeighborhood->items[i];
		bool		found;

		AddToVisited(base, v, hc->element, true, &found);

		if (!found)
			unvisited[(*unvisitedLength)++].element = PgturbohybridGraphPtrAccess(base, hc->element);
	}
}

/*
 * Load neighbor index TIDs
 */
bool
PgturbohybridGraphLoadNeighborTids(PgturbohybridGraphElement element, ItemPointerData *indextids, Relation index, int m, int lm, int lc)
{
	Buffer		buf;
	Page		page;
	PgturbohybridGraphNeighborTuple ntup;
	int			start;

	buf = ReadBuffer(index, element->neighborPage);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);

	ntup = (PgturbohybridGraphNeighborTuple) PageGetItem(page, PageGetItemId(page, element->neighborOffno));

	/*
	 * Ensure the neighbor tuple has not been deleted or replaced between
	 * index scan iterations
	 */
	if (ntup->version != element->version || ntup->count != (element->level + 2) * m)
	{
		UnlockReleaseBuffer(buf);
		return false;
	}

	/* Copy to minimize lock time */
	start = (element->level - lc) * m;
	memcpy(indextids, ntup->indextids + start, lm * sizeof(ItemPointerData));

	UnlockReleaseBuffer(buf);
	return true;
}

/*
 * Load unvisited neighbors from disk
 */
static void
PgturbohybridGraphLoadUnvisitedFromDisk(PgturbohybridGraphElement element, PgturbohybridGraphUnvisited * unvisited, int *unvisitedLength, visited_hash * v, Relation index, int m, int lm, int lc)
{
	ItemPointerData indextids[PGTURBOHYBRID_GRAPH_MAX_M * 2];

	*unvisitedLength = 0;

	if (!PgturbohybridGraphLoadNeighborTids(element, indextids, index, m, lm, lc))
		return;

	for (int i = 0; i < lm; i++)
	{
		ItemPointer indextid = &indextids[i];
		bool		found;

		if (!ItemPointerIsValid(indextid))
			break;

		tidhash_insert(v->tids, *indextid, &found);

		if (!found)
			unvisited[(*unvisitedLength)++].indextid = *indextid;
	}
}

/*
 * Algorithm 2 from paper
 */
List *
PgturbohybridGraphSearchLayer(char *base, PgturbohybridGraphQuery * q, List *ep, int ef, int lc, Relation index, PgturbohybridGraphSupport * support, int m, bool inserting, PgturbohybridGraphElement skipElement, visited_hash * v, pairingheap **discarded, bool initVisited, int64 *tuples, int64 tupleLimit, int64 *scoredCodes, PgturbohybridGraphTqQuery *tq)
{
	List	   *w = NIL;
	pairingheap *C = pairingheap_allocate(CompareNearestCandidates, NULL);
	pairingheap *W = pairingheap_allocate(CompareFurthestCandidates, NULL);
	int			wlen = 0;
	visited_hash vh;
	ListCell   *lc2;
	PgturbohybridGraphNeighborArray *localNeighborhood = NULL;
	Size		neighborhoodSize = 0;
	int			lm = PgturbohybridGraphGetLayerM(m, lc);
	PgturbohybridGraphUnvisited *unvisited = palloc(lm * sizeof(PgturbohybridGraphUnvisited));
	int			unvisitedLength;
	bool		inMemory = index == NULL;
	bool		tupleLimitReached = false;
	bool		useTqCodeScoring = !inserting && !inMemory && scoredCodes != NULL &&
		tq != NULL && tq->enabled;

	if (v == NULL)
	{
		v = &vh;
		initVisited = true;
	}

	if (initVisited)
	{
		InitVisited(base, v, inMemory, ef, m);

		if (discarded != NULL)
			*discarded = pairingheap_allocate(CompareNearestDiscardedCandidates, NULL);
	}

	/* Create local memory for neighborhood if needed */
	if (inMemory)
	{
		neighborhoodSize = PGTURBOHYBRID_GRAPH_NEIGHBOR_ARRAY_SIZE(lm);
		localNeighborhood = palloc(neighborhoodSize);
	}

	/* Add entry points to v, C, and W */
	foreach(lc2, ep)
	{
		PgturbohybridGraphSearchCandidate *sc = (PgturbohybridGraphSearchCandidate *) lfirst(lc2);
		bool		found;

		if (initVisited)
		{
			AddToVisited(base, v, sc->element, inMemory, &found);

			/* OK to count elements instead of tuples */
			if (tuples != NULL)
			{
				(*tuples)++;
				if (tupleLimit >= 0 && *tuples >= tupleLimit)
					tupleLimitReached = true;
			}
		}

		pairingheap_add(C, &sc->c_node);
		pairingheap_add(W, &sc->w_node);

		/*
		 * Do not count elements being deleted towards ef when vacuuming. It
		 * would be ideal to do this for inserts as well, but this could
		 * affect insert performance.
		 */
		if (CountElement(skipElement, PgturbohybridGraphPtrAccess(base, sc->element)))
			wlen++;
	}

	while (!pairingheap_is_empty(C))
	{
		PgturbohybridGraphSearchCandidate *c = PgturbohybridGraphGetSearchCandidate(c_node, pairingheap_remove_first(C));
		PgturbohybridGraphSearchCandidate *f = PgturbohybridGraphGetSearchCandidate(w_node, pairingheap_first(W));
		PgturbohybridGraphElement cElement;

		if (tupleLimitReached)
			break;

		if (c->distance > f->distance)
			break;

		cElement = PgturbohybridGraphPtrAccess(base, c->element);

		if (inMemory)
			PgturbohybridGraphLoadUnvisitedFromMemory(base, cElement, unvisited, &unvisitedLength, v, lc, localNeighborhood, neighborhoodSize);
		else
			PgturbohybridGraphLoadUnvisitedFromDisk(cElement, unvisited, &unvisitedLength, v, index, m, lm, lc);

		for (int i = 0; i < unvisitedLength; i++)
		{
			PgturbohybridGraphElement eElement;
			PgturbohybridGraphSearchCandidate *e;
			double		eDistance;
			bool		alwaysAdd = wlen < ef;

			if (tuples != NULL)
			{
				if (tupleLimit >= 0 && *tuples >= tupleLimit)
				{
					tupleLimitReached = true;
					break;
				}

				(*tuples)++;
			}

			f = PgturbohybridGraphGetSearchCandidate(w_node, pairingheap_first(W));

			if (inMemory)
			{
				eElement = unvisited[i].element;
				eDistance = GetElementDistance(base, eElement, q, support);
			}
			else
			{
				ItemPointer indextid = &unvisited[i].indextid;
				BlockNumber blkno = ItemPointerGetBlockNumber(indextid);
				OffsetNumber offno = ItemPointerGetOffsetNumber(indextid);

				/* Avoid any allocations if not adding */
				eElement = NULL;
				PgturbohybridGraphLoadElementImpl(blkno, offno, &eDistance, q, index, support, inserting, alwaysAdd || discarded != NULL ? NULL : &f->distance, &eElement, useTqCodeScoring, tq, scoredCodes);

				if (eElement == NULL)
					continue;
			}

			if (!(eDistance < f->distance || alwaysAdd))
			{
				if (discarded != NULL)
				{
					/* Create a new candidate */
					e = PgturbohybridGraphInitSearchCandidate(base, eElement, eDistance);
					pairingheap_add(*discarded, &e->w_node);
				}

				continue;
			}

			/* Make robust to issues */
			if (eElement->level < lc)
				continue;

			/* Create a new candidate */
			e = PgturbohybridGraphInitSearchCandidate(base, eElement, eDistance);
			pairingheap_add(C, &e->c_node);
			pairingheap_add(W, &e->w_node);

			/*
			 * Do not count elements being deleted towards ef when vacuuming.
			 * It would be ideal to do this for inserts as well, but this
			 * could affect insert performance.
			 */
			if (CountElement(skipElement, eElement))
			{
				wlen++;

				/* No need to decrement wlen */
				if (wlen > ef)
				{
					PgturbohybridGraphSearchCandidate *d = PgturbohybridGraphGetSearchCandidate(w_node, pairingheap_remove_first(W));

					if (discarded != NULL)
						pairingheap_add(*discarded, &d->w_node);
				}
			}
		}
	}

	/* Add each element of W to w */
	while (!pairingheap_is_empty(W))
	{
		PgturbohybridGraphSearchCandidate *sc = PgturbohybridGraphGetSearchCandidate(w_node, pairingheap_remove_first(W));

		w = lappend(w, sc);
	}

	return w;
}

/*
 * Compare candidate distances with pointer tie-breaker
 */
static int
CompareCandidateDistances(const ListCell *a, const ListCell *b)
{
	PgturbohybridGraphCandidate *hca = lfirst(a);
	PgturbohybridGraphCandidate *hcb = lfirst(b);

	if (hca->distance < hcb->distance)
		return 1;

	if (hca->distance > hcb->distance)
		return -1;

	if (PgturbohybridGraphPtrPointer(hca->element) < PgturbohybridGraphPtrPointer(hcb->element))
		return 1;

	if (PgturbohybridGraphPtrPointer(hca->element) > PgturbohybridGraphPtrPointer(hcb->element))
		return -1;

	return 0;
}

/*
 * Compare candidate distances with offset tie-breaker
 */
static int
CompareCandidateDistancesOffset(const ListCell *a, const ListCell *b)
{
	PgturbohybridGraphCandidate *hca = lfirst(a);
	PgturbohybridGraphCandidate *hcb = lfirst(b);

	if (hca->distance < hcb->distance)
		return 1;

	if (hca->distance > hcb->distance)
		return -1;

	if (PgturbohybridGraphPtrOffset(hca->element) < PgturbohybridGraphPtrOffset(hcb->element))
		return 1;

	if (PgturbohybridGraphPtrOffset(hca->element) > PgturbohybridGraphPtrOffset(hcb->element))
		return -1;

	return 0;
}

/*
 * Check if an element is closer to q than any element from R
 */
static bool
CheckElementCloser(char *base, PgturbohybridGraphCandidate * e, List *r, PgturbohybridGraphSupport * support)
{
	PgturbohybridGraphElement eElement = PgturbohybridGraphPtrAccess(base, e->element);
	Datum		eValue = PgturbohybridGraphGetValue(base, eElement);
	ListCell   *lc2;

	foreach(lc2, r)
	{
		PgturbohybridGraphCandidate *ri = lfirst(lc2);
		PgturbohybridGraphElement riElement = PgturbohybridGraphPtrAccess(base, ri->element);
		Datum		riValue = PgturbohybridGraphGetValue(base, riElement);
		float		distance = PgturbohybridGraphGetDistance(eValue, riValue, support);

		if (distance <= e->distance)
			return false;
	}

	return true;
}

/*
 * Algorithm 4 from paper
 */
static List *
SelectNeighbors(char *base, List *c, int lm, PgturbohybridGraphSupport * support, bool *closerSet, PgturbohybridGraphCandidate * newCandidate, PgturbohybridGraphCandidate * *pruned, bool sortCandidates)
{
	List	   *r = NIL;
	List	   *w = list_copy(c);
	PgturbohybridGraphCandidate **wd;
	int			wdlen = 0;
	int			wdoff = 0;
	bool		mustCalculate = !(*closerSet);
	List	   *added = NIL;
	bool		removedAny = false;

	if (list_length(w) <= lm)
		return w;

	wd = palloc(sizeof(PgturbohybridGraphCandidate *) * list_length(w));

	/* Ensure order of candidates is deterministic for closer caching */
	if (sortCandidates)
	{
		if (base == NULL)
			list_sort(w, CompareCandidateDistances);
		else
			list_sort(w, CompareCandidateDistancesOffset);
	}

	while (list_length(w) > 0 && list_length(r) < lm)
	{
		/* Assumes w is already ordered desc */
		PgturbohybridGraphCandidate *e = llast(w);

		w = list_delete_last(w);

		/* Use previous state of r and wd to skip work when possible */
		if (mustCalculate)
			e->closer = CheckElementCloser(base, e, r, support);
		else if (list_length(added) > 0)
		{
			/* Keep Valgrind happy for in-memory, parallel builds */
			if (base != NULL)
				VALGRIND_MAKE_MEM_DEFINED(&e->closer, 1);

			/*
			 * If the current candidate was closer, we only need to compare it
			 * with the other candidates that we have added.
			 */
			if (e->closer)
			{
				e->closer = CheckElementCloser(base, e, added, support);

				if (!e->closer)
					removedAny = true;
			}
			else
			{
				/*
				 * If we have removed any candidates from closer, a candidate
				 * that was not closer earlier might now be.
				 */
				if (removedAny)
				{
					e->closer = CheckElementCloser(base, e, r, support);
					if (e->closer)
						added = lappend(added, e);
				}
			}
		}
		else if (e == newCandidate)
		{
			e->closer = CheckElementCloser(base, e, r, support);
			if (e->closer)
				added = lappend(added, e);
		}

		/* Keep Valgrind happy for in-memory, parallel builds */
		if (base != NULL)
			VALGRIND_MAKE_MEM_DEFINED(&e->closer, 1);

		if (e->closer)
			r = lappend(r, e);
		else
			wd[wdlen++] = e;
	}

	/* Cached value can only be used in future if sorted deterministically */
	*closerSet = sortCandidates;

	/* Keep pruned connections */
	while (wdoff < wdlen && list_length(r) < lm)
		r = lappend(r, wd[wdoff++]);

	/* Return pruned for update connections */
	if (pruned != NULL)
	{
		if (wdoff < wdlen)
			*pruned = wd[wdoff];
		else
			*pruned = linitial(w);
	}

	return r;
}

/*
 * Add connections
 */
static void
AddConnections(char *base, PgturbohybridGraphElement element, List *neighbors, int lc)
{
	ListCell   *lc2;
	PgturbohybridGraphNeighborArray *a = PgturbohybridGraphGetNeighbors(base, element, lc);

	foreach(lc2, neighbors)
		a->items[a->length++] = *((PgturbohybridGraphCandidate *) lfirst(lc2));
}

/*
 * Update connections
 */
void
PgturbohybridGraphUpdateConnection(char *base, PgturbohybridGraphNeighborArray * neighbors, PgturbohybridGraphElement newElement, float distance, int lm, int *updateIdx, Relation index, PgturbohybridGraphSupport * support)
{
	PgturbohybridGraphCandidate newHc;

	PgturbohybridGraphPtrStore(base, newHc.element, newElement);
	newHc.distance = distance;

	if (neighbors->length < lm)
	{
		neighbors->items[neighbors->length++] = newHc;

		/* Track update */
		if (updateIdx != NULL)
			*updateIdx = -2;
	}
	else
	{
		/* Shrink connections */
		List	   *c = NIL;
		PgturbohybridGraphCandidate *pruned = NULL;

		/* Add candidates */
		for (int i = 0; i < neighbors->length; i++)
			c = lappend(c, &neighbors->items[i]);
		c = lappend(c, &newHc);

		SelectNeighbors(base, c, lm, support, &neighbors->closerSet, &newHc, &pruned, true);

		/* Should not happen */
		if (pruned == NULL)
			return;

		/* Find and replace the pruned element */
		for (int i = 0; i < neighbors->length; i++)
		{
			if (PgturbohybridGraphPtrEqual(base, neighbors->items[i].element, pruned->element))
			{
				neighbors->items[i] = newHc;

				/* Track update */
				if (updateIdx != NULL)
					*updateIdx = i;

				break;
			}
		}
	}
}

/*
 * Remove elements being deleted or skipped
 */
static List *
RemoveElements(char *base, List *w, PgturbohybridGraphElement skipElement)
{
	ListCell   *lc2;
	List	   *w2 = NIL;

	/* Ensure does not access heaptidsLength during in-memory build */
	pg_memory_barrier();

	foreach(lc2, w)
	{
		PgturbohybridGraphCandidate *hc = (PgturbohybridGraphCandidate *) lfirst(lc2);
		PgturbohybridGraphElement hce = PgturbohybridGraphPtrAccess(base, hc->element);

		/* Skip self for vacuuming update */
		if (skipElement != NULL && hce->blkno == skipElement->blkno && hce->offno == skipElement->offno)
			continue;

		if (hce->heaptidsLength != 0)
			w2 = lappend(w2, hc);
	}

	return w2;
}

/*
 * Precompute hash
 */
static void
PrecomputeHash(char *base, PgturbohybridGraphElement element)
{
	PgturbohybridGraphElementPtr ptr;

	PgturbohybridGraphPtrStore(base, ptr, element);

	if (base == NULL)
		element->hash = hash_pointer((uintptr_t) PgturbohybridGraphPtrPointer(ptr));
	else
		element->hash = hash_offset(PgturbohybridGraphPtrOffset(ptr));
}

/*
 * Algorithm 1 from paper
 */
void
PgturbohybridGraphFindElementNeighbors(char *base, PgturbohybridGraphElement element, PgturbohybridGraphElement entryPoint, Relation index, PgturbohybridGraphSupport * support, int m, int efConstruction, bool existing)
{
	List	   *ep;
	List	   *w;
	int			level = element->level;
	int			entryLevel;
	PgturbohybridGraphQuery	q;
	PgturbohybridGraphElement skipElement = existing ? element : NULL;
	bool		inMemory = index == NULL;

	q.value = PgturbohybridGraphGetValue(base, element);

	/* Precompute hash */
	if (inMemory)
		PrecomputeHash(base, element);

	/* No neighbors if no entry point */
	if (entryPoint == NULL)
		return;

	/* Get entry point and level */
	ep = list_make1(PgturbohybridGraphEntryCandidate(base, entryPoint, &q, index, support, true));
	entryLevel = entryPoint->level;

	/* 1st phase: greedy search to insert level */
	for (int lc = entryLevel; lc >= level + 1; lc--)
	{
		w = PgturbohybridGraphSearchLayer(base, &q, ep, 1, lc, index, support, m, true, skipElement, NULL, NULL, true, NULL, -1, NULL, NULL);
		ep = w;
	}

	if (level > entryLevel)
		level = entryLevel;

	/* Add one for existing element */
	if (existing)
		efConstruction++;

	/* 2nd phase */
	for (int lc = level; lc >= 0; lc--)
	{
		int			lm = PgturbohybridGraphGetLayerM(m, lc);
		List	   *neighbors;
		List	   *lw = NIL;
		ListCell   *lc2;

		w = PgturbohybridGraphSearchLayer(base, &q, ep, efConstruction, lc, index, support, m, true, skipElement, NULL, NULL, true, NULL, -1, NULL, NULL);

		/* Convert search candidates to candidates */
		foreach(lc2, w)
		{
			PgturbohybridGraphSearchCandidate *sc = lfirst(lc2);
			PgturbohybridGraphCandidate *hc = palloc(sizeof(PgturbohybridGraphCandidate));

			hc->element = sc->element;
			hc->distance = sc->distance;

			lw = lappend(lw, hc);
		}

		/* Elements being deleted or skipped can help with search */
		/* but should be removed before selecting neighbors */
		if (!inMemory)
			lw = RemoveElements(base, lw, skipElement);

		/*
		 * Candidates are sorted, but not deterministically. Could set
		 * sortCandidates to true for in-memory builds to enable closer
		 * caching, but there does not seem to be a difference in performance.
		 */
		neighbors = SelectNeighbors(base, lw, lm, support, &PgturbohybridGraphGetNeighbors(base, element, lc)->closerSet, NULL, NULL, false);

		AddConnections(base, element, neighbors, lc);

		ep = w;
	}
}

Datum		pgturbohybrid_l2_normalize(PG_FUNCTION_ARGS);

/*
 * Get type info
 */
const		PgturbohybridGraphTypeInfo *
PgturbohybridGraphGetTypeInfo(Relation index)
{
	FmgrInfo   *procinfo = PgturbohybridGraphOptionalProcInfo(index, PGTURBOHYBRID_GRAPH_TYPE_INFO_PROC);

	if (procinfo == NULL || get_func_rettype(procinfo->fn_oid) != INTERNALOID)
	{
		static const PgturbohybridGraphTypeInfo typeInfo = {
			.maxDimensions = PGTURBOHYBRID_GRAPH_MAX_DIM,
			.normalize = pgturbohybrid_l2_normalize,
			.checkValue = NULL
		};

		return (&typeInfo);
	}
	else
		return (const PgturbohybridGraphTypeInfo *) DatumGetPointer(FunctionCall0Coll(procinfo, InvalidOid));
}

/*
 * turbohybrid_scorer_distances(query vector, doc vector) -> jsonb
 *
 * Diagnostic that scores a single (query, doc) pair as a 4-bit packed code
 * under each dense approximate scorer.  Lets operators inspect quantization
 * error and lets regression tests prove the SIMD query-split path expands
 * nibbles through the non-uniform codebook (PgturbohybridGraphCodebookI8 =
 * {-127,-96,-75,-58,-44,-31,-18,-6,6,18,31,44,58,75,96,127}) rather than as
 * raw 0..15 values.
 *
 *   scalar_lut       : codebook via the scalar LUT reference (TqCodeDistanceScalar)
 *   signed_split     : codebook via the integer signed query-split SIMD kernel
 *   linear_reference : the same inner product but with each code nibble scored
 *                      as its raw 0..15 quantization level instead of the
 *                      codebook centre.  This is a reference value only -- it
 *                      quantifies how much the non-uniform codebook matters and
 *                      is NOT a selectable scan scorer (no kernel dispatch ever
 *                      references it).  A correct scorer must match scalar_lut,
 *                      not linear_reference.
 *
 * Inner-product scoring, no EC correction, 4-bit codes.  Use dim >= 1024 to
 * exercise the query-split path (otherwise signed_split falls back to scalar).
 */
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(turbohybrid_scorer_distances);
FUNCTION_PREFIX Datum
turbohybrid_scorer_distances(PG_FUNCTION_ARGS)
{
	Vector	   *query = PgturbohybridDatumGetVector(PG_GETARG_DATUM(0));
	Vector	   *doc = PgturbohybridDatumGetVector(PG_GETARG_DATUM(1));
	int			bits = PGTURBOHYBRID_DEFAULT_BITS;
	int			dim;
	PgturbohybridGraphTqQuery tq;
	uint8	   *code;
	float		scale;
	double		dimSqrt;
	double		dScalar;
	double		dSplit;
	double		dRaw;
	double		dotRaw = 0;
	bool		splitUsed;
	double		dU8Simd;
	double		dU8Scalar;
	bool		u8Used = false;
	const char *u8Kernel = "none";
	StringInfoData json;

	PgturbohybridCheckVector(query);
	PgturbohybridCheckVector(doc);
	dim = query->dim;
	if (doc->dim != dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("query and doc vectors must have equal dimension")));

	/* Prepare a minimal inner-product query (no index, no EC correction). */
	memset(&tq, 0, sizeof(tq));
	tq.dimensions = dim;
	tq.bits = bits;
	tq.lutWidth = 1 << bits;
	tq.codeBytes = TqCodeSizeForBits(dim, bits);
	tq.scoreMode = PGTURBOHYBRID_SCORE_IP;
	tq.scoringKernel = TqSelectScoringKernel();
	tq.rawQueryValues = query->x;
	tq.queryValues = palloc(sizeof(float) * dim);
	TqRotateVectorFloat(query->x, dim, tq.queryValues);
	for (int i = 0; i < dim; i++)
		tq.queryNorm += (double) query->x[i] * (double) query->x[i];
	tq.lut = palloc(PGTURBOHYBRID_LUT_SIZE(dim));
	TqBuildQueryLut(&tq);
	tq.enabled = true;
#if defined(__aarch64__) || defined(_M_ARM64) || PGTURBOHYBRID_COMPILE_AVX2
	TqPrepareQuerySplit4(&tq);
#endif

	code = palloc0(tq.codeBytes);
	scale = TqEncodeVectorBits(doc, code, bits);
	dimSqrt = sqrt((double) dim);

	/* (1) scalar/LUT codebook reference. */
	dScalar = TqCodeDistanceScalar(&tq, code, scale);

	/* (2) signed integer query-split SIMD kernel (codebook via shuffle). */
	splitUsed = PgturbohybridGraphTqCodeSignedSplitDistance(&tq, code, scale, &dSplit);
	if (!splitUsed)
		dSplit = dScalar;

	/* (2b) unsigned-codebook split: scalar reference + SIMD (maddubs/VPDPBUSD). */
	dU8Simd = dScalar;
	dU8Scalar = dScalar;
#if PGTURBOHYBRID_COMPILE_AVX2
	TqPrepareQueryU8Split(&tq);
	if (tq.u8SplitEnabled)
	{
		PgturbohybridGraphTqCodeU8ScalarDistance(&tq, code, scale, &dU8Scalar);
		if (PgturbohybridGraphTqCodeU8SimdDistance(&tq, code, scale, &dU8Simd))
		{
			u8Used = true;
			u8Kernel = PgturbohybridGraphU8SplitKernelName();
		}
		else
			dU8Simd = dU8Scalar;
	}
#endif

	/*
	 * (3) Linear/uniform reference: score each code nibble as its raw 0..15
	 * quantization level instead of the non-uniform codebook centre.  A correct
	 * scorer must match scalar_lut, not this; the gap shows the codebook effect.
	 */
	for (int i = 0; i < dim; i++)
	{
		int			vc = TqGetCodeComponentBits(code, i, bits);

		dotRaw += (double) tq.queryValues[i] * (double) vc;
	}
	dRaw = -((double) scale * dotRaw / dimSqrt);

	initStringInfo(&json);
	appendStringInfo(&json,
					 "{\"scalar_lut\":%.9g,\"signed_split\":%.9g,"
					 "\"unsigned_split_scalar\":%.9g,\"unsigned_split_simd\":%.9g,"
					 "\"linear_reference\":%.9g,\"split_kernel\":\"%s\","
					 "\"unsigned_split_kernel\":\"%s\","
					 "\"query_split_used\":%s,\"unsigned_split_used\":%s,"
					 "\"dimensions\":%d,\"bits\":%d}",
					 dScalar, dSplit, dU8Scalar, dU8Simd, dRaw,
					 PgturbohybridGraphTqScoringKernelName(tq.scoringKernel),
					 u8Kernel,
					 splitUsed ? "true" : "false", u8Used ? "true" : "false",
					 dim, bits);

	PG_RETURN_DATUM(DirectFunctionCall1(jsonb_in, CStringGetDatum(json.data)));
}
