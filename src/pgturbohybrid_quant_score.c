#include "postgres.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>

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

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#define PGTURBOHYBRID_GRAPH_X86 1
#else
#define PGTURBOHYBRID_GRAPH_X86 0
#endif


#define PGTURBOHYBRID_QUERY_SPLIT_HIGH_COEF 256
#define PGTURBOHYBRID_GRAPH_CODEBOOK_SCALE (127.0 / 2.733)
#define PGTURBOHYBRID_GRAPH_CODEBOOK2_SCALE (127.0 / 1.510)

#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && \
	(defined(__AVX2__) || (PGTURBOHYBRID_GRAPH_X86 && (defined(__GNUC__) || defined(__clang__))))
#define PGTURBOHYBRID_GRAPH_COMPILE_AVX2 1
#else
#define PGTURBOHYBRID_GRAPH_COMPILE_AVX2 0
#endif

#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && \
	(defined(__AVX512VNNI__) || (PGTURBOHYBRID_GRAPH_X86 && (defined(__GNUC__) || defined(__clang__))))
#define PGTURBOHYBRID_GRAPH_COMPILE_AVX512VNNI 1
#else
#define PGTURBOHYBRID_GRAPH_COMPILE_AVX512VNNI 0
#endif

/*
 * AVX-VNNI runtime detection requires __builtin_cpu_supports("avxvnni"), which
 * GCC has since 11.x and Clang since 18. Older Clang (e.g. 17, used for the
 * LLVM JIT bitcode build of this extension on Ubuntu 24.04) rejects that
 * feature string and breaks the build, so gate accordingly.
 */
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && \
	(defined(__AVXVNNI__) || \
	(PGTURBOHYBRID_GRAPH_X86 && defined(__clang__) && __clang_major__ >= 18) || \
	(PGTURBOHYBRID_GRAPH_X86 && defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11))
#define PGTURBOHYBRID_GRAPH_COMPILE_AVXVNNI 1
#else
#define PGTURBOHYBRID_GRAPH_COMPILE_AVXVNNI 0
#endif

#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2 && !defined(__AVX2__) && (defined(__GNUC__) || defined(__clang__))
#define PGTURBOHYBRID_GRAPH_AVX2_TARGET __attribute__((target("avx2")))
#else
#define PGTURBOHYBRID_GRAPH_AVX2_TARGET
#endif

#if PGTURBOHYBRID_GRAPH_COMPILE_AVX512VNNI && \
	!(defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512BW__)) && \
	(defined(__GNUC__) || defined(__clang__))
#define PGTURBOHYBRID_GRAPH_AVX512VNNI_TARGET __attribute__((target("avx512vnni,avx512vl,avx512bw,avx2")))
#else
#define PGTURBOHYBRID_GRAPH_AVX512VNNI_TARGET
#endif

/*
 * AVX-512 VPOPCNTDQ compile gates.  Available since
 * Ice Lake (client) / Sapphire Rapids (server).  Compile coverage
 * mirrors AVX-512 VNNI (broad target() attribute), runtime detection
 * is more specific because Skylake-X has VNNI but not VPOPCNTDQ.
 */
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && \
	(defined(__AVX512VPOPCNTDQ__) || (PGTURBOHYBRID_GRAPH_X86 && (defined(__GNUC__) || defined(__clang__))))
#define PGTURBOHYBRID_GRAPH_COMPILE_AVX512VPOPCNTDQ 1
#else
#define PGTURBOHYBRID_GRAPH_COMPILE_AVX512VPOPCNTDQ 0
#endif

#if PGTURBOHYBRID_GRAPH_COMPILE_AVX512VPOPCNTDQ && \
	!(defined(__AVX512VPOPCNTDQ__) && defined(__AVX512BW__)) && \
	(defined(__GNUC__) || defined(__clang__))
#define PGTURBOHYBRID_GRAPH_AVX512VPOPCNTDQ_TARGET __attribute__((target("avx512vpopcntdq,avx512bw,avx512f")))
#else
#define PGTURBOHYBRID_GRAPH_AVX512VPOPCNTDQ_TARGET
#endif

#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && \
	(defined(__AVX512F__) || (PGTURBOHYBRID_GRAPH_X86 && (defined(__GNUC__) || defined(__clang__))))
#define PGTURBOHYBRID_GRAPH_COMPILE_AVX512_WEIGHTED 1
#else
#define PGTURBOHYBRID_GRAPH_COMPILE_AVX512_WEIGHTED 0
#endif

#if PGTURBOHYBRID_GRAPH_COMPILE_AVX512_WEIGHTED && \
	!(defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__) && defined(__AVX512VL__)) && \
	(defined(__GNUC__) || defined(__clang__))
#define PGTURBOHYBRID_GRAPH_AVX512_WEIGHTED_TARGET __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx2")))
#else
#define PGTURBOHYBRID_GRAPH_AVX512_WEIGHTED_TARGET
#endif

#if PGTURBOHYBRID_GRAPH_COMPILE_AVXVNNI && !defined(__AVXVNNI__) && (defined(__GNUC__) || defined(__clang__))
#define PGTURBOHYBRID_GRAPH_AVXVNNI_TARGET __attribute__((target("avxvnni,avx2")))
#else
#define PGTURBOHYBRID_GRAPH_AVXVNNI_TARGET
#endif

#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2 || PGTURBOHYBRID_GRAPH_COMPILE_AVX512VNNI || \
	PGTURBOHYBRID_GRAPH_COMPILE_AVXVNNI || PGTURBOHYBRID_GRAPH_COMPILE_AVX512_WEIGHTED
#include <immintrin.h>
#endif

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

#include "utils/fmgrprotos.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"

#include "pgturbohybrid_quant_score.h"
#include "pgturbohybrid_quant_score_internal.h"

#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && (defined(__aarch64__) || defined(_M_ARM64))
#if defined(__clang__)
#define PGTURBOHYBRID_GRAPH_ARM_DOT_TARGET __attribute__((target("dotprod")))
#define PGTURBOHYBRID_GRAPH_COMPILE_ARM_DOT 1
#define PGTURBOHYBRID_GRAPH_ARM_I8MM_TARGET __attribute__((target("i8mm")))
#define PGTURBOHYBRID_GRAPH_COMPILE_ARM_I8MM 1
#elif defined(__GNUC__)
#define PGTURBOHYBRID_GRAPH_ARM_DOT_TARGET __attribute__((target("+dotprod")))
#define PGTURBOHYBRID_GRAPH_COMPILE_ARM_DOT 1
#define PGTURBOHYBRID_GRAPH_ARM_I8MM_TARGET __attribute__((target("+i8mm")))
#define PGTURBOHYBRID_GRAPH_COMPILE_ARM_I8MM 1
#else
#define PGTURBOHYBRID_GRAPH_ARM_DOT_TARGET
#define PGTURBOHYBRID_GRAPH_COMPILE_ARM_DOT 0
#define PGTURBOHYBRID_GRAPH_ARM_I8MM_TARGET
#define PGTURBOHYBRID_GRAPH_COMPILE_ARM_I8MM 0
#endif
#else
#define PGTURBOHYBRID_GRAPH_COMPILE_ARM_DOT 0
#define PGTURBOHYBRID_GRAPH_COMPILE_ARM_I8MM 0
#endif


#if PGTURBOHYBRID_GRAPH_COMPILE_ARM_DOT || PGTURBOHYBRID_GRAPH_COMPILE_AVX2 || \
	PGTURBOHYBRID_GRAPH_COMPILE_AVX512VNNI || PGTURBOHYBRID_GRAPH_COMPILE_AVXVNNI || \
	PGTURBOHYBRID_GRAPH_COMPILE_AVX512_WEIGHTED
#define PGTURBOHYBRID_GRAPH_COMPILE_QUERY_SPLIT 1
#else
#define PGTURBOHYBRID_GRAPH_COMPILE_QUERY_SPLIT 0
#endif



static inline float
PgturbohybridGraphCodeCenter(int code, int bits)
{
	return TqGetCodeCenterBits(code, bits);
}

double
PgturbohybridGraphExactDistance(PgturbohybridGraphSupport *support, Datum a, Datum b)
{
	return DatumGetFloat8(FunctionCall2Coll(support->procinfo, support->collation, a, b));
}

TqScoreMode
PgturbohybridGraphGetScoreMode(PgturbohybridGraphSupport *support)
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

float
PgturbohybridGraphVectorNorm(Vector *vector)
{
	double		norm = 0;

	for (int i = 0; i < vector->dim; i++)
		norm += (double) vector->x[i] * vector->x[i];

	return (float) sqrt(norm);
}

float
PgturbohybridGraphCodeNorm(const uint8 *code, int dimensions, int bits)
{
	float		norm = 0;

	for (int i = 0; i < dimensions; i++)
	{
		int			component = TqGetCodeComponentBits(code, i, bits);
		float		center = PgturbohybridGraphCodeCenter(component, bits);

		norm += center * center;
	}

	return norm;
}


/*
 * Scalar reference for the pgturbohybrid symmetric (code-code) raw weighted
 * inner product, mirroring qdrant's score_symmetric_ec:
 *
 *     raw = Σ PgturbohybridGraphCodeCenter(a[d]) · PgturbohybridGraphCodeCenter(b[d]) · D'²[d]
 *
 * where D'²[d] = 1 / ecScale[d]², matching qdrant's convention
 * (ec.scale[d] is the per-coord *multiplier* applied during encoding —
 * `(rotated + shift) * scale` — and D' = 1/scale undoes it during
 * scoring).  The point of the weighting is exactly to cancel the
 * scaling done at encode time, so the raw weighted dot reconstructs
 * ⟨rotated_a + shift, rotated_b + shift⟩ in the original space (modulo
 * quantization noise).
 *
 * The full TQ+ score is then: raw + xm_a + xm_b - mm_const, with
 *   xm = ⟨rotated, -ecShift⟩ (computed at encode time inside
 *        TqEncodeVectorInternal; persisted per vector)
 *   mm_const = Σ ecShift[d]² (PgturbohybridGraphMmConstScalar).
 *
 * Returns 0 when ecScale is NULL (legacy index, no weighted scoring).
 */
double
PgturbohybridGraphCodeCodeWeightedRawScalar(const uint8 *a, const uint8 *b,
								  int dimensions, int bits,
								  const float *ecScale)
{
	double		acc = 0.0;

	if (ecScale == NULL)
		return 0.0;

	for (int i = 0; i < dimensions; i++)
	{
		int			ca = TqGetCodeComponentBits(a, i, bits);
		int			cb = TqGetCodeComponentBits(b, i, bits);
		double		centerA = (double) PgturbohybridGraphCodeCenter(ca, bits);
		double		centerB = (double) PgturbohybridGraphCodeCenter(cb, bits);
		double		s = (double) ecScale[i];
		double		w;

		if (fabs(s) <= FLT_EPSILON)
			continue;

		w = 1.0 / (s * s);
		acc += centerA * centerB * w;
	}

	return acc;
}

/*
 * Σ ecShift[d]² — the TQ+ "mm_const" scalar.  Computed once per scan
 * setup (or per build distance call) and reused.  Returns 0 when
 * ecShift is NULL.
 */
double
PgturbohybridGraphMmConstScalar(const float *ecShift, int dimensions)
{
	double		acc = 0.0;

	if (ecShift == NULL)
		return 0.0;

	for (int i = 0; i < dimensions; i++)
		acc += (double) ecShift[i] * (double) ecShift[i];

	return acc;
}

float
PgturbohybridGraphEncodeVector(PgturbohybridQuantBuildState *state, Vector *vector, uint8 *code)
{
	if (state != NULL && state->ecShift != NULL && state->ecScale != NULL)
		return TqEncodeVectorWithCorrectionBits(vector, code, state->tqBits,
												state->ecShift, state->ecScale);

	return TqEncodeVectorBits(vector, code, state != NULL ? state->tqBits : PGTURBOHYBRID_DEFAULT_BITS);
}

/*
 * pgturbohybrid encode helper: emits both the code AND the qdrant-compatible
 * xm = ⟨rotated, -ecShift⟩ in one pass (no double rotation cost).  Used
 * by the build/insert paths when state->tqWeighted is set.
 *
 * Falls back to plain encoding (xm := 0) when ecShift/ecScale are not
 * available.
 */
float
PgturbohybridGraphEncodeVectorWithXm(PgturbohybridQuantBuildState *state, Vector *vector,
						   uint8 *code, float *xmOut)
{
	if (state != NULL && state->ecShift != NULL && state->ecScale != NULL)
		return TqEncodeVectorWithCorrectionAndXmBits(vector, code, state->tqBits,
													  state->ecShift, state->ecScale,
													  xmOut);

	if (xmOut != NULL)
		*xmOut = 0.0f;
	return TqEncodeVectorBits(vector, code, state != NULL ? state->tqBits : PGTURBOHYBRID_DEFAULT_BITS);
}

/*
 * Renormalizing encode helper: like PgturbohybridGraphEncodeVectorWithXm but also returns
 * the renormalized scaling factor.  When ecShift/ecScale are present, the
 * return value is `length · sqrt(d) / centroid_norm` so existing scorers
 * that compute `node->scale · dot / dimSqrt` yield qdrant-style
 * `length · dot / centroid_norm` — the per-vector renormalization fold.
 *
 * `centroid_norm` is measured in EC-reverted (rescaled-pre-EC) space and
 * tracks ‖decoded_quantized‖.  For perfect quantization centroid_norm
 * equals sqrt(d) and the renorm is a no-op; for lossy quantization the
 * drift between centroid_norm and sqrt(d) is what the correction folds
 * back into scoring.
 *
 * When ecShift/ecScale are absent (L2 builds, plain quantization), the
 * renorm is not applicable and the helper falls back to plain length.
 * This matches qdrant: L2 stores `scaling_factor = l2_length` directly,
 * Dot/Cosine store `l2_length / centroid_norm`.
 */
float
PgturbohybridGraphEncodeVectorWithXmRenorm(PgturbohybridQuantBuildState *state, Vector *vector,
								uint8 *code, float *xmOut)
{
	float		length;
	float		centroidNorm;
	double		dimSqrt;

	if (state == NULL || state->ecShift == NULL || state->ecScale == NULL)
	{
		if (xmOut != NULL)
			*xmOut = 0.0f;
		return TqEncodeVectorBits(vector, code,
								  state != NULL ? state->tqBits : PGTURBOHYBRID_DEFAULT_BITS);
	}

	length = TqEncodeVectorWithCorrectionXmRenormBits(vector, code, state->tqBits,
													  state->ecShift, state->ecScale,
													  xmOut, &centroidNorm);

	if (length == 0.0f || centroidNorm <= 0.0f)
		return length;

	dimSqrt = sqrt((double) vector->dim);
	return (float) ((double) length * dimSqrt / (double) centroidNorm);
}

#if (defined(__aarch64__) || defined(_M_ARM64)) && PGTURBOHYBRID_GRAPH_ENABLE_SYMMETRIC_I8_DOT
static inline int32 PgturbohybridGraphInt8Int8DotNeon(const int8 *query,
										   const int8 *components, int dim);
static inline void PgturbohybridGraphInt8Int8Dot4Neon(const int8 *query,
										   const uint8 **valueCodes, int dim,
										   int32 *dots);
#endif

#if PGTURBOHYBRID_GRAPH_ENABLE_SYMMETRIC_I8_DOT
static inline int32 PgturbohybridGraphInt8Int8DotScalar(const int8 *query,
											 const int8 *components, int dim);
#endif

static double
PgturbohybridGraphBuildCodeCodeRawScalarRange(const uint8 *a, const uint8 *b,
								   int startDim, int dimCount, int bits);

static double
PgturbohybridGraphPackedDistance(const PgturbohybridGraphTqQuery *tq, const uint8 *valueCode,
					  float valueScale, float valueCodeNorm, float valueNorm)
{
	(void) valueCodeNorm;
	(void) valueNorm;

	return TqCodeDistance(tq, valueCode, valueScale);
}

#if PGTURBOHYBRID_GRAPH_COMPILE_QUERY_SPLIT
static bool
PgturbohybridGraphPackedDistanceQuerySplit4(const PgturbohybridGraphTqQuery *tq, const uint8 *valueCode,
								 float valueScale, double *distance)
{
	double		dimSqrt;
	double		dot;
	int64		rawDot;
	TqScoreMode mode = (TqScoreMode) tq->scoreMode;

	if (!tq->signedSplit.enabled || tq->dimensions < 1024 ||
		tq->bits != PGTURBOHYBRID_DEFAULT_BITS || mode == PGTURBOHYBRID_SCORE_L1)
		return false;

#if PGTURBOHYBRID_GRAPH_COMPILE_ARM_DOT
	if (PgturbohybridGraphArmDotprodAvailable())
		rawDot = PgturbohybridGraphQuerySplitRawNeonSdot(tq, valueCode);
	else
#endif
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX512VNNI
	if (PgturbohybridGraphAvx512VnniAvailable())
		rawDot = PgturbohybridGraphQuerySplitRawAvx512Vnni(tq, valueCode);
	else
#endif
#if PGTURBOHYBRID_GRAPH_COMPILE_AVXVNNI
	if (PgturbohybridGraphAvxVnniAvailable())
		rawDot = PgturbohybridGraphQuerySplitRawAvxVnni(tq, valueCode);
	else
#endif
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
	if (PgturbohybridGraphAvx2Available())
		rawDot = PgturbohybridGraphQuerySplitRawAvx2(tq, valueCode);
	else
#endif
		return false;

	dimSqrt = sqrt((double) tq->dimensions);
	dot = (double) tq->signedSplit.postprocessScale *
		(double) rawDot;
	dot += tq->ecCorrection;

	if (mode == PGTURBOHYBRID_SCORE_IP)
		*distance = -(valueScale * dot / dimSqrt);
	else if (mode == PGTURBOHYBRID_SCORE_COSINE)
	{
		if (tq->queryNorm == 0 || valueScale == 0)
			*distance = 1;
		else
			*distance = 1 - (dot / (sqrt(tq->queryNorm) * dimSqrt));
	}
	else
	{
		*distance = tq->queryNorm + ((double) valueScale * valueScale) -
			(2 * valueScale * dot / dimSqrt);
		if (*distance < 0)
			*distance = 0;
	}

	return true;
}

#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
/*
 * Whether the unsigned-codebook (u8) split should be used for this query.
 * 4-bit only, dim >= 1024, not L1, not SIMD-forced-scalar, impl != signed, and
 * AVX2 (maddubs) available.  Pure function of GUCs + query shape; used by the
 * query prep (which representation to build), the scorer, and the LUT-skip
 * predicate.
 */
bool
PgturbohybridGraphTqUseU8Split(const PgturbohybridGraphTqQuery *tq)
{
	/*
	 * Hard requirements, independent of policy: the u8 split is a 4-bit-only
	 * maddubs/VPDPBUSD kernel and needs AVX2.  Forcing SIMD to scalar disables
	 * it (the scalar/LUT reference runs instead).
	 */
	if (tq == NULL ||
		pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_SCALAR ||
		tq->bits != PGTURBOHYBRID_DEFAULT_BITS ||
		tq->dimensions < 1024 ||
		(TqScoreMode) tq->scoreMode == PGTURBOHYBRID_SCORE_L1 ||
		!PgturbohybridGraphAvx2Available())
		return false;

	/*
	 * Dedicated dense_u8_split control takes precedence; auto defers to the
	 * broader split-impl policy (impl=signed keeps the signed-codebook split).
	 */
	switch ((TqU8Split) pgturbohybrid_dense_u8_split)
	{
		case PGTURBOHYBRID_U8_SPLIT_OFF:
			return false;
		case PGTURBOHYBRID_U8_SPLIT_ON:
			return true;
		case PGTURBOHYBRID_U8_SPLIT_AUTO:
		default:
			return pgturbohybrid_dense_query_split_impl != PGTURBOHYBRID_QUERY_SPLIT_IMPL_SIGNED;
	}
}

#endif

static bool
PgturbohybridGraphPackedDistanceQuerySplit2(const PgturbohybridGraphTqQuery *tq, const uint8 *valueCode,
								 float valueScale, double *distance)
{
	double		dimSqrt;
	double		dot;
	int64		rawDot;
	TqScoreMode mode = (TqScoreMode) tq->scoreMode;

	if (!tq->signedSplit.enabled || tq->dimensions < 1024 ||
		tq->bits != 2 || mode == PGTURBOHYBRID_SCORE_L1)
		return false;

#if PGTURBOHYBRID_GRAPH_COMPILE_ARM_DOT
	if (PgturbohybridGraphArmDotprodAvailable())
		rawDot = PgturbohybridGraphQuerySplit2RawNeonSdot(tq, valueCode);
	else
#endif
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX512VNNI
	if (PgturbohybridGraphAvx512VnniAvailable())
		rawDot = PgturbohybridGraphQuerySplit2RawAvx512Vnni(tq, valueCode);
	else
#endif
#if PGTURBOHYBRID_GRAPH_COMPILE_AVXVNNI
	if (PgturbohybridGraphAvxVnniAvailable())
		rawDot = PgturbohybridGraphQuerySplit2RawAvxVnni(tq, valueCode);
	else
#endif
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
	if (PgturbohybridGraphAvx2Available())
		rawDot = PgturbohybridGraphQuerySplit2RawAvx2(tq, valueCode);
	else
#endif
		return false;

	dimSqrt = sqrt((double) tq->dimensions);
	dot = (double) tq->signedSplit.postprocessScale *
		(double) rawDot;
	dot += tq->ecCorrection;

	if (mode == PGTURBOHYBRID_SCORE_IP)
		*distance = -(valueScale * dot / dimSqrt);
	else if (mode == PGTURBOHYBRID_SCORE_COSINE)
	{
		if (tq->queryNorm == 0 || valueScale == 0)
			*distance = 1;
		else
			*distance = 1 - (dot / (sqrt(tq->queryNorm) * dimSqrt));
	}
	else
	{
		*distance = tq->queryNorm + ((double) valueScale * valueScale) -
			(2 * valueScale * dot / dimSqrt);
		if (*distance < 0)
			*distance = 0;
	}

	return true;
}

static bool
PgturbohybridGraphBuildCodeCodeDistance4(PgturbohybridQuantBuildState *state, uint32 a, uint32 b,
							  double *distance)
{
	PgturbohybridGraphBuildNode *aNode;
	PgturbohybridGraphBuildNode *bNode;
	double		dot;
	double		scale;
	double		codebookScaleSq;
	int			simdDimensions;
	int			tailDims;
	int			sampleDims;
	int64		rawDot;
	TqScoreMode mode = (TqScoreMode) state->scoreMode;

	if ((mode != PGTURBOHYBRID_SCORE_L2 && mode != PGTURBOHYBRID_SCORE_COSINE && mode != PGTURBOHYBRID_SCORE_IP) ||
		state->tqBits != PGTURBOHYBRID_DEFAULT_BITS)
		return false;

	aNode = &state->nodes[a];
	bNode = &state->nodes[b];
	if (aNode->code == NULL || bNode->code == NULL)
		return false;

	tailDims = state->dimensions % 16;
	simdDimensions = state->dimensions - tailDims;

#if PGTURBOHYBRID_GRAPH_COMPILE_ARM_DOT
	if (PgturbohybridGraphArmDotprodAvailable())
		rawDot = PgturbohybridGraphCodeCodeRawNeonSdot(aNode->code, bNode->code,
											simdDimensions, &sampleDims);
	else
#endif
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX512VNNI
	if (PgturbohybridGraphAvx512VnniAvailable())
		rawDot = PgturbohybridGraphCodeCodeRawAvx512Vnni(aNode->code, bNode->code,
											  simdDimensions, &sampleDims);
	else
#endif
#if PGTURBOHYBRID_GRAPH_COMPILE_AVXVNNI
	if (PgturbohybridGraphAvxVnniAvailable())
		rawDot = PgturbohybridGraphCodeCodeRawAvxVnni(aNode->code, bNode->code,
										   simdDimensions, &sampleDims);
	else
#endif
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
	if (PgturbohybridGraphAvx2Available())
		rawDot = PgturbohybridGraphCodeCodeRawAvx2(aNode->code, bNode->code,
										simdDimensions, &sampleDims);
	else
#endif
		return false;

	codebookScaleSq = PGTURBOHYBRID_GRAPH_CODEBOOK_SCALE * PGTURBOHYBRID_GRAPH_CODEBOOK_SCALE;
	dot = (double) rawDot / codebookScaleSq;
	if (tailDims != 0 && sampleDims == simdDimensions)
	{
		dot += PgturbohybridGraphBuildCodeCodeRawScalarRange(aNode->code, bNode->code,
												  simdDimensions, tailDims,
												  state->tqBits);
		sampleDims += tailDims;
	}
	if (sampleDims <= 0)
		return false;

	scale = (double) sampleDims;
	if (mode == PGTURBOHYBRID_SCORE_IP)
		*distance = -((double) aNode->scale * (double) bNode->scale *
					  dot / scale);
	else if (mode == PGTURBOHYBRID_SCORE_COSINE)
		*distance = 1 - (dot / scale);
	else
	{
		*distance = ((double) aNode->scale * aNode->scale) +
			((double) bNode->scale * bNode->scale) -
			(2 * (double) aNode->scale * (double) bNode->scale * dot / scale);
		if (*distance < 0)
			*distance = 0;
	}

	return true;
}

static bool
PgturbohybridGraphBuildCodeCodeDistance2(PgturbohybridQuantBuildState *state, uint32 a, uint32 b,
							  double *distance)
{
	PgturbohybridGraphBuildNode *aNode;
	PgturbohybridGraphBuildNode *bNode;
	double		dot;
	double		scale;
	double		codebookScaleSq;
	int			simdDimensions;
	int			tailDims;
	int			sampleDims;
	int64		rawDot;
	TqScoreMode mode = (TqScoreMode) state->scoreMode;

	if ((mode != PGTURBOHYBRID_SCORE_L2 && mode != PGTURBOHYBRID_SCORE_COSINE && mode != PGTURBOHYBRID_SCORE_IP) ||
		state->tqBits != 2)
		return false;

	aNode = &state->nodes[a];
	bNode = &state->nodes[b];
	if (aNode->code == NULL || bNode->code == NULL)
		return false;

	tailDims = state->dimensions % 16;
	simdDimensions = state->dimensions - tailDims;

#if PGTURBOHYBRID_GRAPH_COMPILE_ARM_DOT
	if (PgturbohybridGraphArmDotprodAvailable())
		rawDot = PgturbohybridGraphCodeCode2RawNeonSdot(aNode->code, bNode->code,
											 simdDimensions, &sampleDims);
	else
#endif
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX512VNNI
	if (PgturbohybridGraphAvx512VnniAvailable())
		rawDot = PgturbohybridGraphCodeCode2RawAvx512Vnni(aNode->code, bNode->code,
											   simdDimensions, &sampleDims);
	else
#endif
#if PGTURBOHYBRID_GRAPH_COMPILE_AVXVNNI
	if (PgturbohybridGraphAvxVnniAvailable())
		rawDot = PgturbohybridGraphCodeCode2RawAvxVnni(aNode->code, bNode->code,
											simdDimensions, &sampleDims);
	else
#endif
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
	if (PgturbohybridGraphAvx2Available())
		rawDot = PgturbohybridGraphCodeCode2RawAvx2(aNode->code, bNode->code,
										 simdDimensions, &sampleDims);
	else
#endif
		return false;

	codebookScaleSq = PGTURBOHYBRID_GRAPH_CODEBOOK2_SCALE * PGTURBOHYBRID_GRAPH_CODEBOOK2_SCALE;
	dot = (double) rawDot / codebookScaleSq;
	if (tailDims != 0 && sampleDims == simdDimensions)
	{
		dot += PgturbohybridGraphBuildCodeCodeRawScalarRange(aNode->code, bNode->code,
												  simdDimensions, tailDims,
												  state->tqBits);
		sampleDims += tailDims;
	}
	if (sampleDims <= 0)
		return false;

	scale = (double) sampleDims;
	if (mode == PGTURBOHYBRID_SCORE_IP)
		*distance = -((double) aNode->scale * (double) bNode->scale *
					  dot / scale);
	else if (mode == PGTURBOHYBRID_SCORE_COSINE)
		*distance = 1 - (dot / scale);
	else
	{
		*distance = ((double) aNode->scale * aNode->scale) +
			((double) bNode->scale * bNode->scale) -
			(2 * (double) aNode->scale * (double) bNode->scale * dot / scale);
		if (*distance < 0)
			*distance = 0;
	}

	return true;
}
#endif

/*
 * Diagnostic/test entry points: score one code under a specific 4-bit scorer
 * for an already-prepared tq, to prove cross-kernel and cross-representation
 * parity from turbohybrid_scorer_distances.
 *
 * The signed-split entry is always defined so turbohybrid_scorer_distances --
 * which references it unconditionally -- links on every target.  The query-split
 * body is compiled only where the split helpers exist (QUERY_SPLIT: ARM dotprod
 * / AVX2 / AVX-VNNI / AVX-512); elsewhere (e.g. MSVC without /arch:AVX2, or a
 * scalar build) it reports "no split scorer ran" by returning false, mirroring
 * the U8 batch entry below.
 */
bool
PgturbohybridGraphTqCodeSignedSplitDistance(const PgturbohybridGraphTqQuery *tq,
											const uint8 *valueCode, float valueScale, double *distance)
{
#if PGTURBOHYBRID_GRAPH_COMPILE_QUERY_SPLIT
	return PgturbohybridGraphPackedDistanceQuerySplit4(tq, valueCode, valueScale, distance) ||
		PgturbohybridGraphPackedDistanceQuerySplit2(tq, valueCode, valueScale, distance);
#else
	(void) tq;
	(void) valueCode;
	(void) valueScale;
	(void) distance;
	return false;
#endif
}

/*
 * The unsigned-codebook (U8) split entry points wrap the x86-only U8 kernel
 * (PgturbohybridGraphPackedDistanceU8Split / ...U8DistanceFromRaw), so they stay
 * AVX2-gated; on non-x86 they are simply absent.
 */
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
bool
PgturbohybridGraphTqCodeU8SimdDistance(const PgturbohybridGraphTqQuery *tq,
									   const uint8 *valueCode, float valueScale, double *distance)
{
	return PgturbohybridGraphPackedDistanceU8Split(tq, valueCode, valueScale, distance);
}

/*
 * Score a single code through the x4 batch kernel (4 identical copies), for
 * parity testing that the x4 path equals the single-node path bit-for-bit.
 */
bool
PgturbohybridGraphTqCodeU8Simdx4Distance(const PgturbohybridGraphTqQuery *tq,
										 const uint8 *valueCode, float valueScale, double *distance)
{
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
	const uint8 *codes[4] = {valueCode, valueCode, valueCode, valueCode};
	float		scales[4] = {valueScale, valueScale, valueScale, valueScale};
	double		dist[4];

	if (PgturbohybridGraphPackedDistanceU8Splitx4(tq, codes, scales, dist))
	{
		*distance = dist[0];
		return true;
	}
#endif
	(void) tq;
	(void) valueCode;
	(void) valueScale;
	(void) distance;
	return false;
}

/*
 * Score four distinct codes through the x4 batch kernel in one call (shared
 * query loads).  Diagnostic/bench entry point that drives the real 4-candidate
 * batch the native scorer uses, rather than four single-node calls.
 */
bool
PgturbohybridGraphTqCodeU8Simdx4Batch(const PgturbohybridGraphTqQuery *tq,
									  const uint8 *codes[4], const float scales[4],
									  double dist[4])
{
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
	return PgturbohybridGraphPackedDistanceU8Splitx4(tq, codes, scales, dist);
#else
	(void) tq;
	(void) codes;
	(void) scales;
	(void) dist;
	return false;
#endif
}

bool
PgturbohybridGraphTqCodeU8ScalarDistance(const PgturbohybridGraphTqQuery *tq,
										 const uint8 *valueCode, float valueScale, double *distance)
{
	TqScoreMode mode = (TqScoreMode) tq->scoreMode;

	if (!tq->u8.enabled || tq->dimensions < 1024 ||
		tq->bits != PGTURBOHYBRID_DEFAULT_BITS || mode == PGTURBOHYBRID_SCORE_L1)
		return false;
	*distance = PgturbohybridGraphU8DistanceFromRaw(tq, valueScale,
													PgturbohybridGraphQuerySplitU8RawScalar(tq, valueCode));
	return true;
}

#endif

/* Name of the unsigned-codebook SIMD kernel that would run (for scan stats).
 * Always compiled so stats.c can name it; internal guards skip unavailable
 * checks on non-x86. */
const char *
PgturbohybridGraphU8SplitKernelName(void)
{
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX512VNNI
	if (PgturbohybridGraphAvx512VnniAvailable())
		return "unsigned_split_avx512vnni";
#endif
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
	if (PgturbohybridGraphAvx2Available())
		return "unsigned_split_avx2";
#endif
	return "unsigned_split_scalar";
}

/*
 * Public single-code query-split entry point.  Tries the integer query-split
 * SIMD scorers (4-bit then 2-bit) for one packed code; returns true and sets
 * *distance when a query-split kernel handled it, false to fall back to the
 * LUT-gather / scalar path.  Always compiled so callers in other translation
 * units (e.g. TqCodeDistance) need no SIMD-detection macros.
 */
bool
PgturbohybridGraphTqCodeQuerySplitDistance(const PgturbohybridGraphTqQuery *tq,
										   const uint8 *valueCode, float valueScale,
										   double *distance)
{
#if PGTURBOHYBRID_GRAPH_COMPILE_QUERY_SPLIT
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
	/* Prefer the unsigned-codebook split when it was prepared for this query. */
	if (tq->u8.enabled &&
		PgturbohybridGraphPackedDistanceU8Split(tq, valueCode, valueScale, distance))
		return true;
#endif
	return PgturbohybridGraphPackedDistanceQuerySplit4(tq, valueCode, valueScale, distance) ||
		PgturbohybridGraphPackedDistanceQuerySplit2(tq, valueCode, valueScale, distance);
#else
	(void) tq;
	(void) valueCode;
	(void) valueScale;
	(void) distance;
	return false;
#endif
}

/*
 * True iff query-split integer scoring will actually be used for this query
 * (so the LUT-gather table is unnecessary).  Mirrors the per-query gating and
 * runtime SIMD availability of the QuerySplit4/QuerySplit2 kernels.
 */
bool
PgturbohybridGraphTqQuerySplitActive(const PgturbohybridGraphTqQuery *tq)
{
#if PGTURBOHYBRID_GRAPH_COMPILE_QUERY_SPLIT
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
	/* Unsigned-codebook split prepared -> it will run; LUT not needed. */
	if (tq != NULL && tq->u8.enabled)
		return true;
#endif
	if (tq == NULL || !tq->signedSplit.enabled || tq->dimensions < 1024 ||
		(tq->bits != PGTURBOHYBRID_DEFAULT_BITS && tq->bits != 2) ||
		(TqScoreMode) tq->scoreMode == PGTURBOHYBRID_SCORE_L1)
		return false;
#if PGTURBOHYBRID_GRAPH_COMPILE_ARM_DOT
	if (PgturbohybridGraphArmDotprodAvailable())
		return true;
#endif
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX512VNNI
	if (PgturbohybridGraphAvx512VnniAvailable())
		return true;
#endif
#if PGTURBOHYBRID_GRAPH_COMPILE_AVXVNNI
	if (PgturbohybridGraphAvxVnniAvailable())
		return true;
#endif
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
	if (PgturbohybridGraphAvx2Available())
		return true;
#endif
	return false;
#else
	(void) tq;
	return false;
#endif
}

static double
PgturbohybridGraphBuildCodeCodeRawScalarRange(const uint8 *a, const uint8 *b,
								   int startDim, int dimCount, int bits)
{
	double		raw = 0.0;

	for (int dimIdx = startDim; dimIdx < startDim + dimCount; dimIdx++)
	{
		int			ca = TqGetCodeComponentBits(a, dimIdx, bits);
		int			cb = TqGetCodeComponentBits(b, dimIdx, bits);

		raw += (double) PgturbohybridGraphCodeCenter(ca, bits) *
			(double) PgturbohybridGraphCodeCenter(cb, bits);
	}

	return raw;
}

static double
PgturbohybridGraphBuildCodeCodeRawScalar(const uint8 *a, const uint8 *b, int dim,
							  int bits, int *sampleDims)
{
	double		raw = 0.0;
	int			chunkDims = 16;
	int			chunks = dim / chunkDims;
	int			tailDims = dim - chunks * chunkDims;
	int			scoredChunks = chunks;

	if (chunks > PGTURBOHYBRID_GRAPH_CODE_CODE_PRUNE_CHUNKS)
		scoredChunks = PGTURBOHYBRID_GRAPH_CODE_CODE_PRUNE_CHUNKS;
	*sampleDims = scoredChunks * chunkDims;

	for (int scored = 0; scored < scoredChunks; scored++)
	{
		int			chunk = scoredChunks == chunks ? scored :
			(int) (((int64) scored * chunks) / scoredChunks);
		int			start = chunk * chunkDims;

		raw += PgturbohybridGraphBuildCodeCodeRawScalarRange(a, b, start, chunkDims,
												  bits);
	}

	if (scoredChunks == chunks && tailDims != 0)
	{
		int			start = chunks * chunkDims;

		raw += PgturbohybridGraphBuildCodeCodeRawScalarRange(a, b, start, tailDims,
												  bits);
		*sampleDims += tailDims;
	}

	return raw;
}

static bool
PgturbohybridGraphBuildCodeCodeDistanceScalar(PgturbohybridQuantBuildState *state, uint32 a, uint32 b,
								   double *distance)
{
	PgturbohybridGraphBuildNode *aNode;
	PgturbohybridGraphBuildNode *bNode;
	double		dot;
	double		scale;
	int			sampleDims;
	TqScoreMode mode = (TqScoreMode) state->scoreMode;

	if ((mode != PGTURBOHYBRID_SCORE_L2 && mode != PGTURBOHYBRID_SCORE_COSINE && mode != PGTURBOHYBRID_SCORE_IP) ||
		(state->tqBits != PGTURBOHYBRID_DEFAULT_BITS && state->tqBits != 2 &&
		 state->tqBits != 8))
		return false;

	aNode = &state->nodes[a];
	bNode = &state->nodes[b];
	if (aNode->code == NULL || bNode->code == NULL)
		return false;

	dot = PgturbohybridGraphBuildCodeCodeRawScalar(aNode->code, bNode->code,
										state->dimensions, state->tqBits,
										&sampleDims);
	if (sampleDims <= 0)
		return false;

	scale = (double) sampleDims;
	if (mode == PGTURBOHYBRID_SCORE_IP)
		*distance = -((double) aNode->scale * (double) bNode->scale *
					  dot / scale);
	else if (mode == PGTURBOHYBRID_SCORE_COSINE)
		*distance = 1 - (dot / scale);
	else
	{
		*distance = ((double) aNode->scale * aNode->scale) +
			((double) bNode->scale * bNode->scale) -
			(2 * (double) aNode->scale * (double) bNode->scale * dot / scale);
		if (*distance < 0)
			*distance = 0;
	}

	return true;
}

/*
 * pgturbohybrid symmetric (code-code) build distance.
 *
 * The qdrant TQ+ formula reconstructs ⟨rotated_a, rotated_b⟩ from the
 * quantized representation:
 *
 *    sim = (Σ X+_a · X+_b · D'²) + xm_a + xm_b - mm_const
 *        = ⟨rotated_a, rotated_b⟩    (modulo quantization noise)
 *
 * TqPreprocessVector rescales each rotated vector to length √dim, so
 * ⟨rot_a, rot_b⟩ / dim = cos(angle).
 *
 * TurboHybrid cosine opclass uses turbohybrid_negative_inner_product as
 * FUNCTION 1, so its scoreMode is PGTURBOHYBRID_SCORE_IP — the "cosine-via-IP"
 * convention of the rest of the codebase.  We mirror the existing
 * unweighted scorer's distance shape so the value is comparable across
 * legacy and weighted paths within the same build:
 *    IP:     dist = -aScale · bScale · sim / dim
 *    COSINE: dist = 1 - sim / dim
 *
 * Activates only when state->tqWeighted is set, ecShift+ecScale are
 * present, the private weighted scorer policy is enabled, and the bit
 * width is 2 or 4.  Falls
 * through to the unweighted scorer otherwise.
 *
 * Build-time weighted scoring may use SIMD when the private weighted scorer
 * policy is enabled and the host supports the selected kernel.
 */
static bool
PgturbohybridGraphBuildCodeCodeWeighted(PgturbohybridQuantBuildState *state, uint32 a, uint32 b,
							  double *distance)
{
	PgturbohybridGraphBuildNode *aNode;
	PgturbohybridGraphBuildNode *bNode;
	double		raw;
	double		sim;
	TqScoreMode mode = (TqScoreMode) state->scoreMode;

	if (!state->tqWeighted ||
		(mode != PGTURBOHYBRID_SCORE_COSINE && mode != PGTURBOHYBRID_SCORE_IP) ||
		state->ecShift == NULL || state->ecScale == NULL ||
		!pgturbohybrid_dense_weighted)
		return false;

	if (state->tqBits != PGTURBOHYBRID_DEFAULT_BITS && state->tqBits != 2)
		return false;

	aNode = &state->nodes[a];
	bNode = &state->nodes[b];
	if (aNode->code == NULL || bNode->code == NULL)
		return false;

	raw = 0.0;
	if (state->dPrimeSqI16 != NULL && state->weightScale > FLT_EPSILON)
	{
		int64		rawI64 = 0;
		double		codebookScaleSq;
		bool		simdRan = false;

		if (state->tqBits == PGTURBOHYBRID_DEFAULT_BITS)
			codebookScaleSq = PGTURBOHYBRID_GRAPH_CODEBOOK_SCALE * PGTURBOHYBRID_GRAPH_CODEBOOK_SCALE;
		else
			codebookScaleSq = PGTURBOHYBRID_GRAPH_CODEBOOK2_SCALE * PGTURBOHYBRID_GRAPH_CODEBOOK2_SCALE;

#if PGTURBOHYBRID_GRAPH_COMPILE_AVX512_WEIGHTED
		if (!simdRan && PgturbohybridGraphAvx512WeightedAvailable())
		{
			if (state->tqBits == PGTURBOHYBRID_DEFAULT_BITS)
				rawI64 = PgturbohybridGraphCodeCodeWeightedRawAvx512(aNode->code, bNode->code,
														  state->dPrimeSqI16,
														  state->dimensions);
			else
				rawI64 = PgturbohybridGraphCodeCode2WeightedRawAvx512(aNode->code, bNode->code,
														   state->dPrimeSqI16,
														   state->dimensions);
			PgturbohybridGraphRecordWeightedCodeCodeKernel(PGTURBOHYBRID_SCORING_AVX512BW_DQ);
			simdRan = true;
		}
#endif
#if PGTURBOHYBRID_GRAPH_COMPILE_ARM_DOT
		if (!simdRan && PgturbohybridGraphArmDotprodAvailable())
		{
			if (state->tqBits == PGTURBOHYBRID_DEFAULT_BITS)
				rawI64 = PgturbohybridGraphCodeCodeWeightedRawNeonSdot(aNode->code, bNode->code,
															 state->dPrimeSqI16,
															 state->dimensions);
			else
				rawI64 = PgturbohybridGraphCodeCode2WeightedRawNeonSdot(aNode->code, bNode->code,
															  state->dPrimeSqI16,
															  state->dimensions);
			PgturbohybridGraphRecordWeightedCodeCodeKernel(PGTURBOHYBRID_SCORING_NEON);
			simdRan = true;
		}
#endif
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
		if (!simdRan && PgturbohybridGraphAvx2Available())
		{
			if (state->tqBits == PGTURBOHYBRID_DEFAULT_BITS)
				rawI64 = PgturbohybridGraphCodeCodeWeightedRawAvx2(aNode->code, bNode->code,
														 state->dPrimeSqI16,
														 state->dimensions);
			else
				rawI64 = PgturbohybridGraphCodeCode2WeightedRawAvx2(aNode->code, bNode->code,
														  state->dPrimeSqI16,
														  state->dimensions);
			PgturbohybridGraphRecordWeightedCodeCodeKernel(PGTURBOHYBRID_SCORING_AVX2);
			simdRan = true;
		}
#endif

		if (simdRan)
			raw = (double) rawI64 / ((double) state->weightScale * codebookScaleSq);
		else
		{
			PgturbohybridGraphRecordWeightedCodeCodeKernel(PGTURBOHYBRID_SCORING_SCALAR);
			raw = PgturbohybridGraphCodeCodeWeightedRawScalar(aNode->code, bNode->code,
												   state->dimensions, state->tqBits,
												   state->ecScale);
		}
	}
	else
	{
		PgturbohybridGraphRecordWeightedCodeCodeKernel(PGTURBOHYBRID_SCORING_SCALAR);
		raw = PgturbohybridGraphCodeCodeWeightedRawScalar(aNode->code, bNode->code,
											   state->dimensions, state->tqBits,
											   state->ecScale);
	}

	sim = raw + (double) aNode->ecCorrection + (double) bNode->ecCorrection -
		state->mmConst;

	if (mode == PGTURBOHYBRID_SCORE_IP)
		*distance = -((double) aNode->scale * (double) bNode->scale *
					  sim / (double) state->dimensions);
	else						/* PGTURBOHYBRID_SCORE_COSINE */
		*distance = 1.0 - sim / (double) state->dimensions;

	return true;
}

#if PGTURBOHYBRID_GRAPH_ENABLE_SYMMETRIC_I8_DOT
static inline int32
PgturbohybridGraphInt8Int8DotScalar(const int8 *query, const int8 *components, int dim)
{
	int32		dot = 0;

	for (int i = 0; i < dim; i++)
		dot += (int32) query[i] * (int32) components[i];

	return dot;
}
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#if PGTURBOHYBRID_GRAPH_ENABLE_SYMMETRIC_I8_DOT
static inline int32
PgturbohybridGraphInt8Int8DotNeon(const int8 *query, const int8 *components, int dim)
{
	int32		dot;
	int			i = 0;
#if defined(__ARM_FEATURE_DOTPROD)
	int32x4_t	acc = vdupq_n_s32(0);

	for (; i + 16 <= dim; i += 16)
	{
		int8x16_t	q = vld1q_s8(query + i);
		int8x16_t	c = vld1q_s8(components + i);

		acc = vdotq_s32(acc, q, c);
	}

	dot = vaddvq_s32(acc);
#else
	int32x4_t	acc0 = vdupq_n_s32(0);
	int32x4_t	acc1 = vdupq_n_s32(0);

	for (; i + 16 <= dim; i += 16)
	{
		int16x8_t	lo = vmull_s8(vld1_s8(query + i),
								   vld1_s8(components + i));
		int16x8_t	hi = vmull_s8(vld1_s8(query + i + 8),
								   vld1_s8(components + i + 8));

		acc0 = vpadalq_s16(acc0, lo);
		acc1 = vpadalq_s16(acc1, hi);
	}

	acc0 = vaddq_s32(acc0, acc1);
	dot = vaddvq_s32(acc0);
#endif
	for (; i < dim; i++)
		dot += (int32) query[i] * (int32) components[i];

	return dot;
}

static inline void
PgturbohybridGraphInt8Int8Dot4Neon(const int8 *query, const uint8 **valueCodes, int dim,
						int32 *dots)
{
	int			i = 0;
	const int8 *c0 = (const int8 *) valueCodes[0];
	const int8 *c1 = (const int8 *) valueCodes[1];
	const int8 *c2 = (const int8 *) valueCodes[2];
	const int8 *c3 = (const int8 *) valueCodes[3];
#if defined(__ARM_FEATURE_DOTPROD)
	int32x4_t	acc0 = vdupq_n_s32(0);
	int32x4_t	acc1 = vdupq_n_s32(0);
	int32x4_t	acc2 = vdupq_n_s32(0);
	int32x4_t	acc3 = vdupq_n_s32(0);

	for (; i + 16 <= dim; i += 16)
	{
		int8x16_t	q = vld1q_s8(query + i);

		acc0 = vdotq_s32(acc0, q, vld1q_s8(c0 + i));
		acc1 = vdotq_s32(acc1, q, vld1q_s8(c1 + i));
		acc2 = vdotq_s32(acc2, q, vld1q_s8(c2 + i));
		acc3 = vdotq_s32(acc3, q, vld1q_s8(c3 + i));
	}

	dots[0] = vaddvq_s32(acc0);
	dots[1] = vaddvq_s32(acc1);
	dots[2] = vaddvq_s32(acc2);
	dots[3] = vaddvq_s32(acc3);
#else
	dots[0] = dots[1] = dots[2] = dots[3] = 0;
	for (; i + 16 <= dim; i += 16)
	{
		int8x8_t	qlo = vld1_s8(query + i);
		int8x8_t	qhi = vld1_s8(query + i + 8);
		int16x8_t	m0lo = vmull_s8(qlo, vld1_s8(c0 + i));
		int16x8_t	m0hi = vmull_s8(qhi, vld1_s8(c0 + i + 8));
		int16x8_t	m1lo = vmull_s8(qlo, vld1_s8(c1 + i));
		int16x8_t	m1hi = vmull_s8(qhi, vld1_s8(c1 + i + 8));
		int16x8_t	m2lo = vmull_s8(qlo, vld1_s8(c2 + i));
		int16x8_t	m2hi = vmull_s8(qhi, vld1_s8(c2 + i + 8));
		int16x8_t	m3lo = vmull_s8(qlo, vld1_s8(c3 + i));
		int16x8_t	m3hi = vmull_s8(qhi, vld1_s8(c3 + i + 8));

		dots[0] += vaddvq_s16(m0lo) + vaddvq_s16(m0hi);
		dots[1] += vaddvq_s16(m1lo) + vaddvq_s16(m1hi);
		dots[2] += vaddvq_s16(m2lo) + vaddvq_s16(m2hi);
		dots[3] += vaddvq_s16(m3lo) + vaddvq_s16(m3hi);
	}
#endif
	for (; i < dim; i++)
	{
		int32		q = query[i];

		dots[0] += q * (int32) c0[i];
		dots[1] += q * (int32) c1[i];
		dots[2] += q * (int32) c2[i];
		dots[3] += q * (int32) c3[i];
	}
}
#endif

#endif


double
PgturbohybridGraphExactVectorDistance(PgturbohybridGraphScanOpaque so, Datum query, char *valuePtr)
{
	Vector	   *queryVector = (Vector *) DatumGetPointer(query);
	Vector	   *valueVector = (Vector *) valuePtr;
	double		distance = 0;
	double		dot = 0;
	double		valueNorm = 0;
	TqScoreMode mode;

	if (queryVector == NULL || valueVector == NULL ||
		queryVector->dim != valueVector->dim || !so->tq.enabled)
	{
		if (so->support.procinfo == NULL)
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("pgturbohybrid exact distance support function is missing")));
		PgturbohybridGraphRecordExactVectorKernel(pgturbohybrid_dense_exact_simd_force == PGTURBOHYBRID_EXACT_SIMD_FORCE_SCALAR ?
									PGTURBOHYBRID_EXACT_KERNEL_SCALAR :
									PGTURBOHYBRID_EXACT_KERNEL_AUTOVEC_FMA);
		return PgturbohybridGraphExactDistance(&so->support, query, PointerGetDatum(valuePtr));
	}

	mode = (TqScoreMode) so->tq.scoreMode;
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
	if (pgturbohybrid_dense_exact_simd_force != PGTURBOHYBRID_EXACT_SIMD_FORCE_SCALAR &&
		PgturbohybridGraphAvx2Available() &&
		PgturbohybridGraphExactVectorDistanceAvx2(so, queryVector, valueVector, &distance))
	{
		PgturbohybridGraphRecordExactVectorKernel(PGTURBOHYBRID_EXACT_KERNEL_AVX2);
		return distance;
	}
#endif
#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && (defined(__aarch64__) || defined(_M_ARM64))
	if (pgturbohybrid_dense_exact_simd_force == PGTURBOHYBRID_EXACT_SIMD_FORCE_SCALAR)
		goto scalar_exact_distance;

	if (mode == PGTURBOHYBRID_SCORE_L2)
	{
		float32x4_t acc0 = vdupq_n_f32(0);
		float32x4_t acc1 = vdupq_n_f32(0);
		float32x4_t acc2 = vdupq_n_f32(0);
		float32x4_t acc3 = vdupq_n_f32(0);
		int			i = 0;

		for (; i + 16 <= queryVector->dim; i += 16)
		{
			float32x4_t qv0 = vld1q_f32(&queryVector->x[i]);
			float32x4_t qv1 = vld1q_f32(&queryVector->x[i + 4]);
			float32x4_t qv2 = vld1q_f32(&queryVector->x[i + 8]);
			float32x4_t qv3 = vld1q_f32(&queryVector->x[i + 12]);
			float32x4_t vv0 = vld1q_f32(&valueVector->x[i]);
			float32x4_t vv1 = vld1q_f32(&valueVector->x[i + 4]);
			float32x4_t vv2 = vld1q_f32(&valueVector->x[i + 8]);
			float32x4_t vv3 = vld1q_f32(&valueVector->x[i + 12]);
			float32x4_t diff0 = vsubq_f32(qv0, vv0);
			float32x4_t diff1 = vsubq_f32(qv1, vv1);
			float32x4_t diff2 = vsubq_f32(qv2, vv2);
			float32x4_t diff3 = vsubq_f32(qv3, vv3);

			acc0 = vfmaq_f32(acc0, diff0, diff0);
			acc1 = vfmaq_f32(acc1, diff1, diff1);
			acc2 = vfmaq_f32(acc2, diff2, diff2);
			acc3 = vfmaq_f32(acc3, diff3, diff3);
		}

		acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
		for (; i + 4 <= queryVector->dim; i += 4)
		{
			float32x4_t qv = vld1q_f32(&queryVector->x[i]);
			float32x4_t vv = vld1q_f32(&valueVector->x[i]);
			float32x4_t diff = vsubq_f32(qv, vv);

			acc0 = vfmaq_f32(acc0, diff, diff);
		}

		distance = (double) vaddvq_f32(acc0);
		for (; i < queryVector->dim; i++)
		{
			double		diff = (double) queryVector->x[i] - valueVector->x[i];

			distance += diff * diff;
		}

		PgturbohybridGraphRecordExactVectorKernel(PGTURBOHYBRID_EXACT_KERNEL_NEON);
		return distance;
	}

	if (mode == PGTURBOHYBRID_SCORE_IP || mode == PGTURBOHYBRID_SCORE_COSINE)
	{
		float32x4_t dotAcc0 = vdupq_n_f32(0);
		float32x4_t dotAcc1 = vdupq_n_f32(0);
		float32x4_t dotAcc2 = vdupq_n_f32(0);
		float32x4_t dotAcc3 = vdupq_n_f32(0);
		float32x4_t normAcc0 = vdupq_n_f32(0);
		float32x4_t normAcc1 = vdupq_n_f32(0);
		float32x4_t normAcc2 = vdupq_n_f32(0);
		float32x4_t normAcc3 = vdupq_n_f32(0);
		int			i = 0;

		for (; i + 16 <= queryVector->dim; i += 16)
		{
			float32x4_t qv0 = vld1q_f32(&queryVector->x[i]);
			float32x4_t qv1 = vld1q_f32(&queryVector->x[i + 4]);
			float32x4_t qv2 = vld1q_f32(&queryVector->x[i + 8]);
			float32x4_t qv3 = vld1q_f32(&queryVector->x[i + 12]);
			float32x4_t vv0 = vld1q_f32(&valueVector->x[i]);
			float32x4_t vv1 = vld1q_f32(&valueVector->x[i + 4]);
			float32x4_t vv2 = vld1q_f32(&valueVector->x[i + 8]);
			float32x4_t vv3 = vld1q_f32(&valueVector->x[i + 12]);

			dotAcc0 = vfmaq_f32(dotAcc0, qv0, vv0);
			dotAcc1 = vfmaq_f32(dotAcc1, qv1, vv1);
			dotAcc2 = vfmaq_f32(dotAcc2, qv2, vv2);
			dotAcc3 = vfmaq_f32(dotAcc3, qv3, vv3);
			if (mode == PGTURBOHYBRID_SCORE_COSINE)
			{
				normAcc0 = vfmaq_f32(normAcc0, vv0, vv0);
				normAcc1 = vfmaq_f32(normAcc1, vv1, vv1);
				normAcc2 = vfmaq_f32(normAcc2, vv2, vv2);
				normAcc3 = vfmaq_f32(normAcc3, vv3, vv3);
			}
		}

		dotAcc0 = vaddq_f32(vaddq_f32(dotAcc0, dotAcc1),
							vaddq_f32(dotAcc2, dotAcc3));
		if (mode == PGTURBOHYBRID_SCORE_COSINE)
			normAcc0 = vaddq_f32(vaddq_f32(normAcc0, normAcc1),
								 vaddq_f32(normAcc2, normAcc3));

		for (; i + 4 <= queryVector->dim; i += 4)
		{
			float32x4_t qv = vld1q_f32(&queryVector->x[i]);
			float32x4_t vv = vld1q_f32(&valueVector->x[i]);

			dotAcc0 = vfmaq_f32(dotAcc0, qv, vv);
			if (mode == PGTURBOHYBRID_SCORE_COSINE)
				normAcc0 = vfmaq_f32(normAcc0, vv, vv);
		}

		dot = (double) vaddvq_f32(dotAcc0);
		if (mode == PGTURBOHYBRID_SCORE_COSINE)
			valueNorm = (double) vaddvq_f32(normAcc0);

		for (; i < queryVector->dim; i++)
		{
			double		qv = queryVector->x[i];
			double		vv = valueVector->x[i];

			dot += qv * vv;
			if (mode == PGTURBOHYBRID_SCORE_COSINE)
				valueNorm += vv * vv;
		}

		if (mode == PGTURBOHYBRID_SCORE_IP)
			return -dot;

		PgturbohybridGraphRecordExactVectorKernel(PGTURBOHYBRID_EXACT_KERNEL_NEON);
		if (so->tq.queryNorm == 0 || valueNorm == 0)
			return 1;

		return 1 - (dot / sqrt(so->tq.queryNorm * valueNorm));
	}
#endif

#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && (defined(__aarch64__) || defined(_M_ARM64))
scalar_exact_distance:
#endif
	for (int i = 0; i < queryVector->dim; i++)
	{
		double		qv = queryVector->x[i];
		double		vv = valueVector->x[i];

		if (mode == PGTURBOHYBRID_SCORE_L1)
			distance += fabs(qv - vv);
		else if (mode == PGTURBOHYBRID_SCORE_L2)
		{
			double		diff = qv - vv;

			distance += diff * diff;
		}
		else
		{
			dot += qv * vv;
			if (mode == PGTURBOHYBRID_SCORE_COSINE)
				valueNorm += vv * vv;
		}
	}

	if (mode == PGTURBOHYBRID_SCORE_L1 || mode == PGTURBOHYBRID_SCORE_L2)
	{
		PgturbohybridGraphRecordExactVectorKernel(PGTURBOHYBRID_EXACT_KERNEL_SCALAR);
		return distance;
	}

	if (mode == PGTURBOHYBRID_SCORE_IP)
	{
		PgturbohybridGraphRecordExactVectorKernel(PGTURBOHYBRID_EXACT_KERNEL_SCALAR);
		return -dot;
	}

	if (mode == PGTURBOHYBRID_SCORE_COSINE)
	{
		PgturbohybridGraphRecordExactVectorKernel(PGTURBOHYBRID_EXACT_KERNEL_SCALAR);
		if (so->tq.queryNorm == 0 || valueNorm == 0)
			return 1;

		return 1 - (dot / sqrt(so->tq.queryNorm * valueNorm));
	}

	PgturbohybridGraphRecordExactVectorKernel(pgturbohybrid_dense_exact_simd_force == PGTURBOHYBRID_EXACT_SIMD_FORCE_SCALAR ?
								PGTURBOHYBRID_EXACT_KERNEL_SCALAR :
								PGTURBOHYBRID_EXACT_KERNEL_AUTOVEC_FMA);
	return PgturbohybridGraphExactDistance(&so->support, query, PointerGetDatum(valuePtr));
}

static bool
PgturbohybridGraphUseExactLowBitRouting(PgturbohybridGraphScanOpaque so, Datum query)
{
	return DatumGetPointer(query) != NULL &&
		so->tq.enabled &&
		(TqScoreMode) so->tq.scoreMode == PGTURBOHYBRID_SCORE_L2 &&
		so->tq.bits < PGTURBOHYBRID_DEFAULT_BITS &&
		so->tq.dimensions >= 1024;
}

bool
PgturbohybridGraphCachedExactNodeDistance(PgturbohybridGraphScanOpaque so, Datum query,
							   PgturbohybridGraphScanNode *node, double *distance)
{
	if (!PgturbohybridGraphUseExactLowBitRouting(so, query) || node->exactVector == NULL)
		return false;

	*distance = PgturbohybridGraphExactVectorDistance(so, query, node->exactVector);
	return true;
}

static bool
PgturbohybridGraphUseExactHighdimL2Entry(PgturbohybridGraphScanOpaque so, Datum query)
{
	return DatumGetPointer(query) != NULL &&
		so->tq.enabled &&
		(TqScoreMode) so->tq.scoreMode == PGTURBOHYBRID_SCORE_L2 &&
		so->tq.bits == PGTURBOHYBRID_DEFAULT_BITS &&
		so->tq.dimensions >= 1024;
}

bool
PgturbohybridGraphExactHighdimEntryDistance(PgturbohybridGraphScanOpaque so, Datum query,
								 PgturbohybridGraphScanNode *node, double *distance)
{
	if (!PgturbohybridGraphUseExactHighdimL2Entry(so, query) || node->exactVector == NULL)
		return false;

	*distance = PgturbohybridGraphExactVectorDistance(so, query, node->exactVector);
	return true;
}

static inline int
PgturbohybridGraphPopcount8(uint8 value)
{
	value = value - ((value >> 1) & 0x55);
	value = (value & 0x33) + ((value >> 2) & 0x33);
	return (value + (value >> 4)) & 0x0F;
}

static int
PgturbohybridGraphBit1PopcntRawCodes(const uint8 *a, const uint8 *b, int dim)
{
	int			fullBytes = dim / 8;
	int			tailBits = dim & 7;
	int			same = 0;

	for (int i = 0; i < fullBytes; i++)
		same += PgturbohybridGraphPopcount8((uint8) ~(a[i] ^ b[i]));

	if (tailBits != 0)
	{
		uint8		mask = (uint8) ((1U << tailBits) - 1U);

		same += PgturbohybridGraphPopcount8((uint8) (~(a[fullBytes] ^ b[fullBytes]) & mask));
	}

	return (2 * same) - dim;
}

/*
 * Scalar asymmetric 1-bit query-vs-code scorer.
 *
 * Returns the centroid-space dot product computed from a bit-plane
 * decomposed 8-bit signed query (tq->bit1.planes, populated by
 * TqPrepareQueryAsymBit1) against a 1-bit packed code.  Reduction:
 *
 *   v_dot_q   = Σ_b w_b · popcount(code AND plane_b)
 *               with w_b = 2^b for b<7, -128 for b=7 (sign plane)
 *   signed_dot = 2 · v_dot_q − Σ q_signed
 *   score      = (c / q_scale) · signed_dot
 *
 * The trailing partial block (tail_bytes < 16) is processed via the
 * same scalar inner loop after copying the tail data bytes into a
 * zero-padded 16-byte scratch buffer.  Plane bytes beyond tail_bytes
 * are already zero from palloc0, so AND-popcount with the padding
 * lanes contributes 0.
 *
 * SIMD parity reference — the AVX2 / AVX-512 / NEON variants must match this byte-for-byte on equivalent inputs.
 */
#define PGTURBOHYBRID_QUERY_ASYM_BLOCK_BYTES_LOCAL 16

static float
PgturbohybridGraphAsymBit1ScalarRawScore(const PgturbohybridGraphTqQuery *tq, const uint8 *code)
{
	const uint8 *planes = tq->bit1.planes;
	int			numFullBlocks = tq->bit1.numFullBlocks;
	int			tailBytes = tq->bit1.tailBytes;
	int			BITS = tq->bit1.bits;
	int64		vDotQ = 0;

	for (int block = 0; block < numFullBlocks; block++)
	{
		const uint8 *dataBlock = code + (Size) block * PGTURBOHYBRID_QUERY_ASYM_BLOCK_BYTES_LOCAL;
		const uint8 *blockPlanes = planes +
			(Size) block * BITS * PGTURBOHYBRID_QUERY_ASYM_BLOCK_BYTES_LOCAL;

		for (int b = 0; b < BITS; b++)
		{
			const uint8 *plane = blockPlanes + b * PGTURBOHYBRID_QUERY_ASYM_BLOCK_BYTES_LOCAL;
			int			pop = 0;

			for (int i = 0; i < PGTURBOHYBRID_QUERY_ASYM_BLOCK_BYTES_LOCAL; i++)
				pop += PgturbohybridGraphPopcount8((uint8) (dataBlock[i] & plane[i]));

			if (b == BITS - 1)
				vDotQ -= (int64) (1 << (BITS - 1)) * pop;
			else
				vDotQ += (int64) (1 << b) * pop;
		}
	}

	if (tailBytes > 0)
	{
		uint8		dataScratch[PGTURBOHYBRID_QUERY_ASYM_BLOCK_BYTES_LOCAL] = {0};
		const uint8 *blockPlanes = planes +
			(Size) numFullBlocks * BITS * PGTURBOHYBRID_QUERY_ASYM_BLOCK_BYTES_LOCAL;
		const uint8 *tailSrc = code + (Size) numFullBlocks * PGTURBOHYBRID_QUERY_ASYM_BLOCK_BYTES_LOCAL;

		memcpy(dataScratch, tailSrc, tailBytes);

		for (int b = 0; b < BITS; b++)
		{
			const uint8 *plane = blockPlanes + b * PGTURBOHYBRID_QUERY_ASYM_BLOCK_BYTES_LOCAL;
			int			pop = 0;

			for (int i = 0; i < PGTURBOHYBRID_QUERY_ASYM_BLOCK_BYTES_LOCAL; i++)
				pop += PgturbohybridGraphPopcount8((uint8) (dataScratch[i] & plane[i]));

			if (b == BITS - 1)
				vDotQ -= (int64) (1 << (BITS - 1)) * pop;
			else
				vDotQ += (int64) (1 << b) * pop;
		}
	}

	{
		int64		signedDot = 2 * vDotQ - tq->bit1.sumSigned;

		return tq->bit1.scale * (float) signedDot;
	}
}

/*
 * SIMD asymmetric 1-bit scorer.
 *
 * AVX2 path uses the nibble-lookup popcount trick (pshufb): split each
 * byte into low/high nibbles, look up popcount per nibble from a
 * 16-entry table, add.  16-byte block processed per inner-loop chunk;
 * the per-block work is `BITS=8` AND + pshufb-popcount sequences.
 *
 * NEON path uses vcntq_u8 (per-byte popcount) directly, then horizontal
 * sum via vaddvq_u8.
 *
 * Both produce bit-identical scalar output (modulo the final `float`
 * cast of the i64 signed_dot).  Verified by the parity test.
 */

#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
static float PGTURBOHYBRID_GRAPH_AVX2_TARGET
PgturbohybridGraphAsymBit1Avx2RawScore(const PgturbohybridGraphTqQuery *tq, const uint8 *code)
{
	const uint8 *planes = tq->bit1.planes;
	int			numFullBlocks = tq->bit1.numFullBlocks;
	int			tailBytes = tq->bit1.tailBytes;
	int			BITS = tq->bit1.bits;
	int64		signWeight = (int64) 1 << (BITS - 1);
	int64		vDotQ = 0;
	const __m128i nibblePopLut = _mm_setr_epi8(
		0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
	const __m128i nibbleMask = _mm_set1_epi8(0x0F);

	for (int block = 0; block < numFullBlocks; block++)
	{
		const uint8 *dataBlock = code + (Size) block * 16;
		const uint8 *blockPlanes = planes + (Size) block * BITS * 16;
		__m128i		data = _mm_loadu_si128((const __m128i *) dataBlock);

		for (int b = 0; b < BITS; b++)
		{
			__m128i		plane = _mm_loadu_si128((const __m128i *) (blockPlanes + b * 16));
			__m128i		v = _mm_and_si128(data, plane);
			__m128i		lo = _mm_and_si128(v, nibbleMask);
			__m128i		hi = _mm_and_si128(_mm_srli_epi16(v, 4), nibbleMask);
			__m128i		popLo = _mm_shuffle_epi8(nibblePopLut, lo);
			__m128i		popHi = _mm_shuffle_epi8(nibblePopLut, hi);
			__m128i		popByte = _mm_add_epi8(popLo, popHi);
			__m128i		popSad = _mm_sad_epu8(popByte, _mm_setzero_si128());
			int			pop = (int) (_mm_cvtsi128_si32(popSad) +
									 _mm_cvtsi128_si32(_mm_srli_si128(popSad, 8)));

			if (b == BITS - 1)
				vDotQ -= signWeight * pop;
			else
				vDotQ += (int64) (1 << b) * pop;
		}
	}

	if (tailBytes > 0)
	{
		uint8		dataScratch[16] = {0};
		const uint8 *blockPlanes = planes + (Size) numFullBlocks * BITS * 16;
		const uint8 *tailSrc = code + (Size) numFullBlocks * 16;
		__m128i		data;

		memcpy(dataScratch, tailSrc, tailBytes);
		data = _mm_loadu_si128((const __m128i *) dataScratch);

		for (int b = 0; b < BITS; b++)
		{
			__m128i		plane = _mm_loadu_si128((const __m128i *) (blockPlanes + b * 16));
			__m128i		v = _mm_and_si128(data, plane);
			__m128i		lo = _mm_and_si128(v, nibbleMask);
			__m128i		hi = _mm_and_si128(_mm_srli_epi16(v, 4), nibbleMask);
			__m128i		popLo = _mm_shuffle_epi8(nibblePopLut, lo);
			__m128i		popHi = _mm_shuffle_epi8(nibblePopLut, hi);
			__m128i		popByte = _mm_add_epi8(popLo, popHi);
			__m128i		popSad = _mm_sad_epu8(popByte, _mm_setzero_si128());
			int			pop = (int) (_mm_cvtsi128_si32(popSad) +
									 _mm_cvtsi128_si32(_mm_srli_si128(popSad, 8)));

			if (b == BITS - 1)
				vDotQ -= signWeight * pop;
			else
				vDotQ += (int64) (1 << b) * pop;
		}
	}

	{
		int64		signedDot = 2 * vDotQ - tq->bit1.sumSigned;

		return tq->bit1.scale * (float) signedDot;
	}
}
#endif		/* PGTURBOHYBRID_GRAPH_COMPILE_AVX2 */

#if PGTURBOHYBRID_GRAPH_COMPILE_AVX512VPOPCNTDQ
/*
 * AVX-512 VPOPCNTDQ kernel for 1-bit asymmetric scoring.
 *
 * Layout: one block is 16 bytes of data and `BITS` × 16 bytes of plane.
 * For BITS = 8 the planes fit in one 128-byte chunk = two ZMM loads;
 * we process each plane as a single 16-byte load broadcast / AND with
 * data, popcount, accumulate.  This is the same per-plane work as the
 * AVX2 path, but VPOPCNTDQ replaces the pshufb nibble-lookup popcount
 * with per-qword popcount over the two meaningful qword lanes.
 */
static float PGTURBOHYBRID_GRAPH_AVX512VPOPCNTDQ_TARGET
PgturbohybridGraphAsymBit1Avx512VpopcntdqRawScore(const PgturbohybridGraphTqQuery *tq, const uint8 *code)
{
	const uint8 *planes = tq->bit1.planes;
	int			numFullBlocks = tq->bit1.numFullBlocks;
	int			tailBytes = tq->bit1.tailBytes;
	int			BITS = tq->bit1.bits;
	int64		signWeight = (int64) 1 << (BITS - 1);
	int64		vDotQ = 0;

	for (int block = 0; block < numFullBlocks; block++)
	{
		const uint8 *dataBlock = code + (Size) block * 16;
		const uint8 *blockPlanes = planes + (Size) block * BITS * 16;
		/* Broadcast 16-byte block into the low XMM lane of a ZMM */
		__m512i		data = _mm512_castsi128_si512(_mm_loadu_si128((const __m128i *) dataBlock));

		for (int b = 0; b < BITS; b++)
		{
			__m512i		plane = _mm512_castsi128_si512(
				_mm_loadu_si128((const __m128i *) (blockPlanes + b * 16)));
			__m512i		v = _mm512_and_si512(data, plane);
			/*
			 * VPOPCNTDQ gives per-qword popcount.  Our AND result has 16
			 * meaningful bytes (lanes 0..1 of the ZMM, lanes 2..7 are
			 * zero from the cast), so popcount(lane0) + popcount(lane1)
			 * equals the popcount of the 16-byte AND.  reduce_add_epi64
			 * sums all 8 lanes; the zero ones contribute 0.
			 */
			__m512i		popLanes = _mm512_popcnt_epi64(v);
			int			pop = (int) _mm512_reduce_add_epi64(popLanes);

			if (b == BITS - 1)
				vDotQ -= signWeight * pop;
			else
				vDotQ += (int64) (1 << b) * pop;
		}
	}

	if (tailBytes > 0)
	{
		uint8		dataScratch[16] = {0};
		const uint8 *blockPlanes = planes + (Size) numFullBlocks * BITS * 16;
		const uint8 *tailSrc = code + (Size) numFullBlocks * 16;
		__m512i		data;

		memcpy(dataScratch, tailSrc, tailBytes);
		data = _mm512_castsi128_si512(_mm_loadu_si128((const __m128i *) dataScratch));

		for (int b = 0; b < BITS; b++)
		{
			__m512i		plane = _mm512_castsi128_si512(
				_mm_loadu_si128((const __m128i *) (blockPlanes + b * 16)));
			__m512i		v = _mm512_and_si512(data, plane);
			/*
			 * VPOPCNTDQ gives per-qword popcount.  Our AND result has 16
			 * meaningful bytes (lanes 0..1 of the ZMM, lanes 2..7 are
			 * zero from the cast), so popcount(lane0) + popcount(lane1)
			 * equals the popcount of the 16-byte AND.  reduce_add_epi64
			 * sums all 8 lanes; the zero ones contribute 0.
			 */
			__m512i		popLanes = _mm512_popcnt_epi64(v);
			int			pop = (int) _mm512_reduce_add_epi64(popLanes);

			if (b == BITS - 1)
				vDotQ -= signWeight * pop;
			else
				vDotQ += (int64) (1 << b) * pop;
		}
	}

	{
		int64		signedDot = 2 * vDotQ - tq->bit1.sumSigned;

		return tq->bit1.scale * (float) signedDot;
	}
}

static bool
PgturbohybridGraphAvx512VpopcntdqAvailable(void)
{
	static int	available = -1;

	/*
	 * GUC kill-switch lets users force the dispatcher
	 * to fall through to the AVX2 kernel even on hosts that have
	 * VPOPCNTDQ.  Useful for parity testing and downclock measurement.
	 * Checked first so the runtime feature probe still memoizes only
	 * the hardware capability — flipping the GUC doesn't require
	 * resetting `available`.
	 */
	/* Valgrind cannot execute AVX-512 VPOPCNTDQ; fall back to AVX2/scalar. */
	if (PgturbohybridGraphRunningUnderValgrind())
		return false;

	if (!pgturbohybrid_dense_graph_avx512vpopcntdq)
		return false;
	if (pgturbohybrid_dense_simd_force != PGTURBOHYBRID_SIMD_FORCE_AUTO &&
		pgturbohybrid_dense_simd_force != PGTURBOHYBRID_SIMD_FORCE_AVX512VNNI)
		return false;

	if (available >= 0)
		return available != 0;

#if defined(__AVX512VPOPCNTDQ__) && defined(__AVX512BW__) && defined(__AVX512F__)
	available = 1;
#elif PGTURBOHYBRID_GRAPH_X86 && (defined(__GNUC__) || defined(__clang__))
	/*
	 * The runtime probe mirrors the target attribute so the dispatcher only
	 * enters code that the host can execute.
	 */
	available = __builtin_cpu_supports("avx512vpopcntdq") &&
		__builtin_cpu_supports("avx512bw") &&
		__builtin_cpu_supports("avx512f") ? 1 : 0;
#else
	available = 0;
#endif
	return available != 0;
}
#endif		/* PGTURBOHYBRID_GRAPH_COMPILE_AVX512VPOPCNTDQ */

#if defined(__aarch64__) || defined(_M_ARM64)
static float
PgturbohybridGraphAsymBit1NeonRawScore(const PgturbohybridGraphTqQuery *tq, const uint8 *code)
{
	const uint8 *planes = tq->bit1.planes;
	int			numFullBlocks = tq->bit1.numFullBlocks;
	int			tailBytes = tq->bit1.tailBytes;
	int			BITS = tq->bit1.bits;
	int64		signWeight = (int64) 1 << (BITS - 1);
	int64		vDotQ = 0;

	for (int block = 0; block < numFullBlocks; block++)
	{
		const uint8 *dataBlock = code + (Size) block * 16;
		const uint8 *blockPlanes = planes + (Size) block * BITS * 16;
		uint8x16_t	data = vld1q_u8(dataBlock);

		for (int b = 0; b < BITS; b++)
		{
			uint8x16_t	plane = vld1q_u8(blockPlanes + b * 16);
			uint8x16_t	v = vandq_u8(data, plane);
			uint8x16_t	popByte = vcntq_u8(v);
			int			pop = (int) vaddvq_u8(popByte);

			if (b == BITS - 1)
				vDotQ -= signWeight * pop;
			else
				vDotQ += (int64) (1 << b) * pop;
		}
	}

	if (tailBytes > 0)
	{
		uint8		dataScratch[16] = {0};
		const uint8 *blockPlanes = planes + (Size) numFullBlocks * BITS * 16;
		const uint8 *tailSrc = code + (Size) numFullBlocks * 16;
		uint8x16_t	data;

		memcpy(dataScratch, tailSrc, tailBytes);
		data = vld1q_u8(dataScratch);

		for (int b = 0; b < BITS; b++)
		{
			uint8x16_t	plane = vld1q_u8(blockPlanes + b * 16);
			uint8x16_t	v = vandq_u8(data, plane);
			uint8x16_t	popByte = vcntq_u8(v);
			int			pop = (int) vaddvq_u8(popByte);

			if (b == BITS - 1)
				vDotQ -= signWeight * pop;
			else
				vDotQ += (int64) (1 << b) * pop;
		}
	}

	{
		int64		signedDot = 2 * vDotQ - tq->bit1.sumSigned;

		return tq->bit1.scale * (float) signedDot;
	}
}
#endif		/* aarch64 */

/*
 * Asymmetric 1-bit dispatch: pick the fastest available kernel.  AVX2 on amd64
 * (runtime feature gated alongside the rest of the AMD64 SIMD knobs),
 * NEON on aarch64, scalar reference fallback elsewhere.
 *
 * The runtime AVX2 gate uses __builtin_cpu_supports("avx2") same as
 * the existing 1-bit + 2-bit fast paths.  When AVX2 is disabled
 * by the private VNNI / SDOT policy, we still take the AVX2 popcount
 * kernel because that policy does not gate basic SIMD popcount.
 */
static float
PgturbohybridGraphAsymBit1Score(const PgturbohybridGraphTqQuery *tq, const uint8 *code)
{
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX512VPOPCNTDQ
	if (PgturbohybridGraphAvx512VpopcntdqAvailable())
		return PgturbohybridGraphAsymBit1Avx512VpopcntdqRawScore(tq, code);
#endif
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
	if (PgturbohybridGraphAvx2Available())
		return PgturbohybridGraphAsymBit1Avx2RawScore(tq, code);
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
	return PgturbohybridGraphAsymBit1NeonRawScore(tq, code);
#endif
	return PgturbohybridGraphAsymBit1ScalarRawScore(tq, code);
}

static bool
PgturbohybridGraphBuildQueryBit1PopcntDistance(PgturbohybridQuantBuildState *state,
									PgturbohybridGraphBuildNode *node,
									double *distance)
{
	PgturbohybridGraphTqQuery *tq = &state->buildTq;
	double		dimSqrt;
	double		dot;
	double		center;
	TqScoreMode mode = (TqScoreMode) state->scoreMode;

	if (!pgturbohybrid_dense_graph_lowbit_popcnt || !state->buildTqValid ||
		tq->bits != 1 || tq->querySignBits == NULL || node->code == NULL ||
		tq->dimensions <= 0 || mode == PGTURBOHYBRID_SCORE_L1)
		return false;

	dimSqrt = sqrt((double) tq->dimensions);
	center = PgturbohybridGraphCodeCenter(1, 1);
	dot = tq->ecCorrection +
		(center * (double) PgturbohybridGraphBit1PopcntRawCodes(tq->querySignBits,
													 node->code,
													 tq->dimensions));

	if (mode == PGTURBOHYBRID_SCORE_IP)
		*distance = -(node->scale * dot / dimSqrt);
	else if (mode == PGTURBOHYBRID_SCORE_COSINE)
	{
		if (tq->queryNorm == 0 || node->scale == 0)
			*distance = 1;
		else
			*distance = 1 - (dot / (sqrt(tq->queryNorm) * dimSqrt));
	}
	else
	{
		*distance = tq->queryNorm + ((double) node->scale * node->scale) -
			(2 * node->scale * dot / dimSqrt);
		if (*distance < 0)
			*distance = 0;
	}

	return true;
}

static bool
PgturbohybridGraphBuildCodeCodeDistanceBits1Popcnt(PgturbohybridQuantBuildState *state,
										uint32 a, uint32 b,
										double *distance)
{
	PgturbohybridGraphBuildNode *aNode;
	PgturbohybridGraphBuildNode *bNode;
	double		dot;
	double		avg;
	double		center;
	TqScoreMode mode = (TqScoreMode) state->scoreMode;

	if (!pgturbohybrid_dense_graph_lowbit_popcnt || state->tqBits != 1 ||
		state->dimensions <= 0 || mode == PGTURBOHYBRID_SCORE_L1)
		return false;

	aNode = &state->nodes[a];
	bNode = &state->nodes[b];
	if (aNode->code == NULL || bNode->code == NULL)
		return false;

	center = PgturbohybridGraphCodeCenter(1, 1);
	dot = center * center *
		(double) PgturbohybridGraphBit1PopcntRawCodes(aNode->code, bNode->code,
										   state->dimensions);
	avg = dot / (double) state->dimensions;

	if (mode == PGTURBOHYBRID_SCORE_IP)
		*distance = -((double) aNode->scale * (double) bNode->scale * avg);
	else if (mode == PGTURBOHYBRID_SCORE_COSINE)
		*distance = 1 - (avg / (center * center));
	else
	{
		*distance = ((double) aNode->scale * aNode->scale) +
			((double) bNode->scale * bNode->scale) -
			(2 * (double) aNode->scale * (double) bNode->scale * avg);
		if (*distance < 0)
			*distance = 0;
	}

	return true;
}

static bool
PgturbohybridGraphTryBuildQueryDistance(PgturbohybridQuantBuildState *state, uint32 a, uint32 b,
							 double *distance)
{
	if (!state->buildTqValid || a != state->buildQueryNodeId ||
		state->nodes[b].code == NULL)
		return false;

	if (PgturbohybridGraphBuildQueryBit1PopcntDistance(state, &state->nodes[b], distance))
	{
		state->buildDistanceQuerySplit++;
		return true;
	}
#if PGTURBOHYBRID_GRAPH_COMPILE_QUERY_SPLIT
	if (PgturbohybridGraphPackedDistanceQuerySplit4(&state->buildTq, state->nodes[b].code,
									 state->nodes[b].scale, distance))
	{
		state->buildDistanceQuerySplit++;
		return true;
	}
	if (PgturbohybridGraphPackedDistanceQuerySplit2(&state->buildTq, state->nodes[b].code,
									 state->nodes[b].scale, distance))
	{
		state->buildDistanceQuerySplit++;
		return true;
	}
#endif

	state->buildDistancePacked++;
	*distance = PgturbohybridGraphPackedDistance(&state->buildTq, state->nodes[b].code,
									 state->nodes[b].scale,
									 state->nodes[b].correction,
									 state->nodes[b].norm);
	return true;
}

static bool
PgturbohybridGraphTryBuildCodeCodeDistance(PgturbohybridQuantBuildState *state, uint32 a, uint32 b,
								 double *distance)
{
	/*
	 * Weighted symmetric scoring takes precedence over the unweighted
	 * code-code path when weighted quantized scoring is enabled.
	 */
	if (PgturbohybridGraphBuildCodeCodeWeighted(state, a, b, distance))
	{
		state->buildDistanceWeighted++;
		return true;
	}

#if PGTURBOHYBRID_GRAPH_COMPILE_QUERY_SPLIT
	if (PgturbohybridGraphBuildCodeCodeDistance4(state, a, b, distance))
	{
		state->buildDistanceCodeCode++;
		return true;
	}
	if (PgturbohybridGraphBuildCodeCodeDistance2(state, a, b, distance))
	{
		state->buildDistanceCodeCode++;
		return true;
	}
#endif
	if (PgturbohybridGraphBuildCodeCodeDistanceBits1Popcnt(state, a, b, distance))
	{
		state->buildDistanceCodeCode++;
		return true;
	}
	if (PgturbohybridGraphBuildCodeCodeDistanceScalar(state, a, b, distance))
	{
		state->buildDistanceCodeCode++;
		return true;
	}

	return false;
}

bool
PgturbohybridGraphBuildCodeDistance(PgturbohybridQuantBuildState *state, uint32 a, uint32 b,
						 double *distance)
{
	if (state == NULL || distance == NULL ||
		a >= state->nodeCount || b >= state->nodeCount)
		return false;

	return PgturbohybridGraphTryBuildCodeCodeDistance(state, a, b, distance);
}

bool
PgturbohybridGraphCodeCodeDistance(PgturbohybridGraphScanOpaque so, PgturbohybridGraphMetaPageData *meta,
						PgturbohybridGraphScanNode *aNode, PgturbohybridGraphScanNode *bNode,
						double *distance)
{
	PgturbohybridQuantBuildState state;
	PgturbohybridGraphBuildNode nodes[2];

	if (so == NULL || !so->tq.enabled || meta == NULL ||
		aNode == NULL || bNode == NULL ||
		aNode->code == NULL || bNode->code == NULL)
		return false;

	memset(&state, 0, sizeof(state));
	memset(nodes, 0, sizeof(nodes));
	state.nodes = nodes;
	state.nodeCount = 2;
	state.dimensions = meta->dimensions;
	state.tqBits = meta->tqBits;
	state.tqWeighted = (meta->tqFlags & PGTURBOHYBRID_GRAPH_TQ_WEIGHTED) != 0;
	state.scoreMode = so->tq.scoreMode;
	state.ecShift = so->tq.ecShift;
	state.ecScale = so->tq.ecScale;
	if (state.ecShift != NULL)
		state.mmConst = PgturbohybridGraphMmConstScalar(state.ecShift, state.dimensions);

	nodes[0].code = aNode->code;
	nodes[0].scale = aNode->scale;
	nodes[0].norm = aNode->norm;
	nodes[0].correction = aNode->codeNorm;
	nodes[0].ecCorrection = aNode->ecCorrection;
	nodes[1].code = bNode->code;
	nodes[1].scale = bNode->scale;
	nodes[1].norm = bNode->norm;
	nodes[1].correction = bNode->codeNorm;
	nodes[1].ecCorrection = bNode->ecCorrection;

	return PgturbohybridGraphTryBuildCodeCodeDistance(&state, 0, 1, distance);
}

static double
PgturbohybridGraphBuildExactVectorDistance(PgturbohybridQuantBuildState *state, uint32 a, uint32 b)
{
	Vector	   *av = state->nodes[a].vector;
	Vector	   *bv = state->nodes[b].vector;
	TqScoreMode mode;
	double		distance = 0;
	double		dot = 0;
	double		aNorm = 0;
	double		bNorm = 0;

	if (av == NULL || bv == NULL || av->dim != bv->dim)
	{
		state->buildDistanceFallback++;
		return DBL_MAX;
	}

	state->buildDistanceExact++;

	mode = (TqScoreMode) state->scoreMode;
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
	if (PgturbohybridGraphAvx2Available() &&
		PgturbohybridGraphBuildExactDistanceAvx2(state, av, bv, &distance))
		return distance;
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
	if (mode == PGTURBOHYBRID_SCORE_L2)
	{
		float32x4_t acc0 = vdupq_n_f32(0);
		float32x4_t acc1 = vdupq_n_f32(0);
		float32x4_t acc2 = vdupq_n_f32(0);
		float32x4_t acc3 = vdupq_n_f32(0);
		int			i = 0;

		for (; i + 16 <= av->dim; i += 16)
		{
			float32x4_t av0 = vld1q_f32(&av->x[i]);
			float32x4_t av1 = vld1q_f32(&av->x[i + 4]);
			float32x4_t av2 = vld1q_f32(&av->x[i + 8]);
			float32x4_t av3 = vld1q_f32(&av->x[i + 12]);
			float32x4_t bv0 = vld1q_f32(&bv->x[i]);
			float32x4_t bv1 = vld1q_f32(&bv->x[i + 4]);
			float32x4_t bv2 = vld1q_f32(&bv->x[i + 8]);
			float32x4_t bv3 = vld1q_f32(&bv->x[i + 12]);
			float32x4_t diff0 = vsubq_f32(av0, bv0);
			float32x4_t diff1 = vsubq_f32(av1, bv1);
			float32x4_t diff2 = vsubq_f32(av2, bv2);
			float32x4_t diff3 = vsubq_f32(av3, bv3);

			acc0 = vfmaq_f32(acc0, diff0, diff0);
			acc1 = vfmaq_f32(acc1, diff1, diff1);
			acc2 = vfmaq_f32(acc2, diff2, diff2);
			acc3 = vfmaq_f32(acc3, diff3, diff3);
		}

		acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
		for (; i + 4 <= av->dim; i += 4)
		{
			float32x4_t avv = vld1q_f32(&av->x[i]);
			float32x4_t bvv = vld1q_f32(&bv->x[i]);
			float32x4_t diff = vsubq_f32(avv, bvv);

			acc0 = vfmaq_f32(acc0, diff, diff);
		}

		distance = (double) vaddvq_f32(acc0);
		for (; i < av->dim; i++)
		{
			double		diff = (double) av->x[i] - bv->x[i];

			distance += diff * diff;
		}

		return distance;
	}

	if (mode == PGTURBOHYBRID_SCORE_IP || mode == PGTURBOHYBRID_SCORE_COSINE)
	{
		float32x4_t dotAcc0 = vdupq_n_f32(0);
		float32x4_t dotAcc1 = vdupq_n_f32(0);
		float32x4_t dotAcc2 = vdupq_n_f32(0);
		float32x4_t dotAcc3 = vdupq_n_f32(0);
		float32x4_t aNormAcc0 = vdupq_n_f32(0);
		float32x4_t aNormAcc1 = vdupq_n_f32(0);
		float32x4_t aNormAcc2 = vdupq_n_f32(0);
		float32x4_t aNormAcc3 = vdupq_n_f32(0);
		float32x4_t bNormAcc0 = vdupq_n_f32(0);
		float32x4_t bNormAcc1 = vdupq_n_f32(0);
		float32x4_t bNormAcc2 = vdupq_n_f32(0);
		float32x4_t bNormAcc3 = vdupq_n_f32(0);
		int			i = 0;

		for (; i + 16 <= av->dim; i += 16)
		{
			float32x4_t av0 = vld1q_f32(&av->x[i]);
			float32x4_t av1 = vld1q_f32(&av->x[i + 4]);
			float32x4_t av2 = vld1q_f32(&av->x[i + 8]);
			float32x4_t av3 = vld1q_f32(&av->x[i + 12]);
			float32x4_t bv0 = vld1q_f32(&bv->x[i]);
			float32x4_t bv1 = vld1q_f32(&bv->x[i + 4]);
			float32x4_t bv2 = vld1q_f32(&bv->x[i + 8]);
			float32x4_t bv3 = vld1q_f32(&bv->x[i + 12]);

			dotAcc0 = vfmaq_f32(dotAcc0, av0, bv0);
			dotAcc1 = vfmaq_f32(dotAcc1, av1, bv1);
			dotAcc2 = vfmaq_f32(dotAcc2, av2, bv2);
			dotAcc3 = vfmaq_f32(dotAcc3, av3, bv3);
			if (mode == PGTURBOHYBRID_SCORE_COSINE)
			{
				aNormAcc0 = vfmaq_f32(aNormAcc0, av0, av0);
				aNormAcc1 = vfmaq_f32(aNormAcc1, av1, av1);
				aNormAcc2 = vfmaq_f32(aNormAcc2, av2, av2);
				aNormAcc3 = vfmaq_f32(aNormAcc3, av3, av3);
				bNormAcc0 = vfmaq_f32(bNormAcc0, bv0, bv0);
				bNormAcc1 = vfmaq_f32(bNormAcc1, bv1, bv1);
				bNormAcc2 = vfmaq_f32(bNormAcc2, bv2, bv2);
				bNormAcc3 = vfmaq_f32(bNormAcc3, bv3, bv3);
			}
		}

		dotAcc0 = vaddq_f32(vaddq_f32(dotAcc0, dotAcc1),
							vaddq_f32(dotAcc2, dotAcc3));
		if (mode == PGTURBOHYBRID_SCORE_COSINE)
		{
			aNormAcc0 = vaddq_f32(vaddq_f32(aNormAcc0, aNormAcc1),
								  vaddq_f32(aNormAcc2, aNormAcc3));
			bNormAcc0 = vaddq_f32(vaddq_f32(bNormAcc0, bNormAcc1),
								  vaddq_f32(bNormAcc2, bNormAcc3));
		}

		for (; i + 4 <= av->dim; i += 4)
		{
			float32x4_t avv = vld1q_f32(&av->x[i]);
			float32x4_t bvv = vld1q_f32(&bv->x[i]);

			dotAcc0 = vfmaq_f32(dotAcc0, avv, bvv);
			if (mode == PGTURBOHYBRID_SCORE_COSINE)
			{
				aNormAcc0 = vfmaq_f32(aNormAcc0, avv, avv);
				bNormAcc0 = vfmaq_f32(bNormAcc0, bvv, bvv);
			}
		}

		dot = (double) vaddvq_f32(dotAcc0);
		if (mode == PGTURBOHYBRID_SCORE_COSINE)
		{
			aNorm = (double) vaddvq_f32(aNormAcc0);
			bNorm = (double) vaddvq_f32(bNormAcc0);
		}

		for (; i < av->dim; i++)
		{
			double		aval = av->x[i];
			double		bval = bv->x[i];

			dot += aval * bval;
			if (mode == PGTURBOHYBRID_SCORE_COSINE)
			{
				aNorm += aval * aval;
				bNorm += bval * bval;
			}
		}

		if (mode == PGTURBOHYBRID_SCORE_IP)
			return -dot;
		if (aNorm == 0 || bNorm == 0)
			return 1;

		return 1 - (dot / sqrt(aNorm * bNorm));
	}
#endif

	for (int i = 0; i < av->dim; i++)
	{
		double		aval = av->x[i];
		double		bval = bv->x[i];

		if (mode == PGTURBOHYBRID_SCORE_L1)
			distance += fabs(aval - bval);
		else if (mode == PGTURBOHYBRID_SCORE_L2)
		{
			double		diff = aval - bval;

			distance += diff * diff;
		}
		else
		{
			dot += aval * bval;
			if (mode == PGTURBOHYBRID_SCORE_COSINE)
			{
				aNorm += aval * aval;
				bNorm += bval * bval;
			}
		}
	}

	if (mode == PGTURBOHYBRID_SCORE_L1 || mode == PGTURBOHYBRID_SCORE_L2)
		return distance;
	if (mode == PGTURBOHYBRID_SCORE_IP)
		return -dot;
	if (mode == PGTURBOHYBRID_SCORE_COSINE)
	{
		if (aNorm == 0 || bNorm == 0)
			return 1;

		return 1 - (dot / sqrt(aNorm * bNorm));
	}

	return PgturbohybridGraphExactDistance(&state->support,
							PointerGetDatum(av),
							PointerGetDatum(bv));
}

#define PGTURBOHYBRID_GRAPH_BUILD_DISTANCE_CACHE_MAX_ENTRIES (UINT32_C(1) << 24)

static uint64
PgturbohybridGraphBuildDistanceCacheHash(uint64 key)
{
	key ^= key >> 33;
	key *= UINT64CONST(0xff51afd7ed558ccd);
	key ^= key >> 33;
	key *= UINT64CONST(0xc4ceb9fe1a85ec53);
	key ^= key >> 33;
	return key;
}

static uint32
PgturbohybridGraphBuildDistanceCacheCapacity(uint32 nodeCount)
{
	uint64		wanted = Max((uint64) nodeCount * UINT64CONST(64),
							 UINT64CONST(1024));
	uint32		capacity = 1;

	wanted = Min(wanted,
				 (uint64) PGTURBOHYBRID_GRAPH_BUILD_DISTANCE_CACHE_MAX_ENTRIES);
	while ((uint64) capacity < wanted &&
		   capacity < PGTURBOHYBRID_GRAPH_BUILD_DISTANCE_CACHE_MAX_ENTRIES)
		capacity <<= 1;

	return Max(capacity, (uint32) 1024);
}

static void
PgturbohybridGraphInitBuildDistanceCache(PgturbohybridQuantBuildState *state)
{
	uint32		capacity;

	if (state->buildDistanceCache != NULL || state->nodeCount == 0)
		return;

	capacity = PgturbohybridGraphBuildDistanceCacheCapacity(state->nodeCount);
	state->buildDistanceCache =
		MemoryContextAllocZero(state->ctx,
							   mul_size(sizeof(PgturbohybridGraphBuildDistanceCacheEntry),
										(Size) capacity));
	state->buildDistanceCacheMask = capacity - 1;
}

static uint64
PgturbohybridGraphBuildDistanceElapsedUs(instr_time start)
{
	instr_time	duration;

	INSTR_TIME_SET_CURRENT(duration);
	INSTR_TIME_SUBTRACT(duration, start);
	return (uint64) INSTR_TIME_GET_MICROSEC(duration);
}

static bool
PgturbohybridGraphUseExactDocumentBuildDistance(PgturbohybridQuantBuildState *state)
{
	return state != NULL &&
		state->multivectorBuild &&
		state->multivectorGraphMode ==
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES &&
		state->multivectorDocBuildScorer ==
		PGTURBOHYBRID_MULTIVECTOR_DOC_BUILD_SCORER_EXACT_SYMMETRIC &&
		state->multivectorNodeMap != NULL &&
		state->multivectorDocVectors != NULL;
}

static double
PgturbohybridGraphExactDocumentBuildDistance(PgturbohybridQuantBuildState *state,
											 const PgturbohybridMultiVector *aDoc,
											 const PgturbohybridMultiVector *bDoc)
{
	instr_time	start;
	double		distance;

	INSTR_TIME_SET_CURRENT(start);
	distance = -TqMultiVectorSymmetricMaxSimAverageUnchecked(aDoc, bDoc);
	state->multivectorDocExactBuildDistanceCalls++;
	state->multivectorDocExactBuildDistanceUs +=
		PgturbohybridGraphBuildDistanceElapsedUs(start);

	return distance;
}

double
PgturbohybridGraphBuildDistance(PgturbohybridQuantBuildState *state, uint32 a, uint32 b)
{
	double		distance;

	state->buildDistanceCalls++;
	if (a >= state->nodeCount || b >= state->nodeCount)
		return DBL_MAX;

	if (PgturbohybridGraphUseExactDocumentBuildDistance(state))
	{
		TqDocId		aDocId = state->multivectorNodeMap[a].docId;
		TqDocId		bDocId = state->multivectorNodeMap[b].docId;
		PgturbohybridMultiVector *aDoc;
		PgturbohybridMultiVector *bDoc;
		uint64		cacheKey;

		if (aDocId >= state->multivectorDocCount ||
			bDocId >= state->multivectorDocCount)
			return DBL_MAX;
		aDoc = state->multivectorDocVectors[aDocId];
		bDoc = state->multivectorDocVectors[bDocId];
		if (aDoc == NULL || bDoc == NULL || aDoc->dim != bDoc->dim ||
			aDoc->count <= 0 || bDoc->count <= 0)
			return DBL_MAX;

		if (aDocId <= bDocId)
			cacheKey = ((uint64) aDocId << 32) | (uint64) bDocId;
		else
			cacheKey = ((uint64) bDocId << 32) | (uint64) aDocId;

		if (state->buildDistanceCache == NULL)
			PgturbohybridGraphInitBuildDistanceCache(state);
		if (state->buildDistanceCache != NULL)
		{
			PgturbohybridGraphBuildDistanceCacheEntry *entry;
			uint64		hash =
				PgturbohybridGraphBuildDistanceCacheHash(cacheKey);

			entry = &state->buildDistanceCache[hash & state->buildDistanceCacheMask];
			if (entry->valid && entry->key == cacheKey)
			{
				state->buildDistanceCacheHits++;
				return entry->distance;
			}
			if (entry->valid)
				state->buildDistanceCacheCollisions++;
			state->buildDistanceCacheMisses++;
			distance =
				PgturbohybridGraphExactDocumentBuildDistance(state, aDoc, bDoc);
			entry->valid = true;
			entry->key = cacheKey;
			entry->distance = distance;
			state->buildDistanceCacheStores++;
			return distance;
		}

		return PgturbohybridGraphExactDocumentBuildDistance(state, aDoc, bDoc);
	}

	/*
	 * When requested, short-circuit the quantized fast paths and route every
	 * build-time pruning call through exact f32 distance.  This locks in a
	 * graph topology built with perfect distances; scan-time scoring still
	 * uses the packed codes.
	 */
	if (state->buildExactDistances && state->nodes[a].vector != NULL &&
		state->nodes[b].vector != NULL &&
		state->nodes[a].vector->dim == state->nodes[b].vector->dim)
		return PgturbohybridGraphBuildExactVectorDistance(state, a, b);

	if (PgturbohybridGraphTryBuildQueryDistance(state, a, b, &distance))
		return distance;
	if (PgturbohybridGraphTryBuildCodeCodeDistance(state, a, b, &distance))
		return distance;

	return PgturbohybridGraphBuildExactVectorDistance(state, a, b);
}

void
PgturbohybridGraphPrepareBuildQuery(PgturbohybridQuantBuildState *state, uint32 nodeId)
{
	MemoryContext callerctx;
	TqScoreMode mode = (TqScoreMode) state->scoreMode;

	state->buildTqValid = false;
	if ((mode != PGTURBOHYBRID_SCORE_L2 && mode != PGTURBOHYBRID_SCORE_COSINE && mode != PGTURBOHYBRID_SCORE_IP) ||
		(state->tqBits != 1 && state->tqBits != 2 &&
		 state->tqBits != PGTURBOHYBRID_DEFAULT_BITS &&
		 state->tqBits != 8) ||
		nodeId >= state->nodeCount ||
		state->nodes[nodeId].vector == NULL || state->buildQueryCtx == NULL)
		return;

	callerctx = CurrentMemoryContext;
	if (callerctx == state->buildQueryCtx)
		callerctx = state->ctx;
	MemoryContextSwitchTo(state->ctx);
	MemoryContextReset(state->buildQueryCtx);
	MemoryContextSwitchTo(state->buildQueryCtx);
	if (state->ecShift != NULL && state->ecScale != NULL)
		PgturbohybridGraphPrepareTqBuildQueryWithCorrection(state->index, &state->support,
											  PointerGetDatum(state->nodes[nodeId].vector),
											  &state->buildTq,
											  state->ecShift,
											  state->ecScale);
	else
		PgturbohybridGraphPrepareTqBuildQuery(state->index, &state->support,
								PointerGetDatum(state->nodes[nodeId].vector),
								&state->buildTq);
	MemoryContextSwitchTo(callerctx);

	state->buildQueryNodeId = nodeId;
	state->buildTqValid = state->buildTq.enabled &&
		state->buildTq.scoreMode == state->scoreMode;
}

/*
 * Attribute `nodes` scored codes to one scoring-kernel bucket for the current
 * native scan (one call per kernel invocation: 4 for a batch-of-4 kernel, 1
 * for a single-node kernel).  Cheap enough for the hot path; lets
 * turbohybrid_last_scan_stats() report exactly which kernel did the work.
 */
static inline void
PgturbohybridGraphRecordScoreKernel(PgturbohybridGraphScanOpaque so, int bucket, int nodes)
{
	so->graphScoreKernelCalls[bucket]++;
	so->graphScoreKernelNodes[bucket] += nodes;
}

/* Map the selected signed query-split SIMD kernel to its batch bucket. */
static inline int
PgturbohybridGraphSignedSplitBatchBucket(int scoringKernel)
{
	switch ((TqScoringKernel) scoringKernel)
	{
		case PGTURBOHYBRID_SCORING_AVX512VNNI:
			return PGTURBOHYBRID_SCORE_KERNEL_BATCH_SIGNED_SPLIT_AVX512VNNI;
		case PGTURBOHYBRID_SCORING_AVXVNNI:
			return PGTURBOHYBRID_SCORE_KERNEL_BATCH_SIGNED_SPLIT_AVXVNNI;
		case PGTURBOHYBRID_SCORING_AVX2:
			return PGTURBOHYBRID_SCORE_KERNEL_BATCH_SIGNED_SPLIT_AVX2;
		default:
			/* NEON/i8mm/scalar (non-x86 or forced): not one of the named split
			 * buckets, so attribute to the catch-all rather than misreport. */
			return PGTURBOHYBRID_SCORE_KERNEL_BATCH_SCALAR_OR_LUT;
	}
}

/* The unsigned-codebook split runs the maddubs (AVX2) or VPDPBUSD
 * (AVX-512-VNNI) kernel; map by the selected SIMD kernel.  Gated to the AVX2
 * compile path -- its only caller is the U8-split batch scorer below, which is
 * AVX2-only -- so non-x86 builds don't carry it as an unused function. */
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
static inline int
PgturbohybridGraphU8SplitBatchBucket(int scoringKernel)
{
	return (TqScoringKernel) scoringKernel == PGTURBOHYBRID_SCORING_AVX512VNNI
		? PGTURBOHYBRID_SCORE_KERNEL_BATCH_U8_SPLIT_AVX512VNNI
		: PGTURBOHYBRID_SCORE_KERNEL_BATCH_U8_SPLIT_AVX2;
}
#endif

double
PgturbohybridGraphScoreNode(PgturbohybridGraphScanOpaque so, PgturbohybridGraphScanNode *node)
{
	if (!so->tq.enabled || node->code == NULL)
		return 0;

	so->graphScoredCodes++;

	/*
	 * Asymmetric 1-bit single-node fast path.  Mirrors the
	 * batch dispatch slot above so tail nodes (< 4 left over) get the
	 * same scoring math as the batch-of-4 path.  Falls through to
	 * PgturbohybridGraphPackedDistance when the GUC is off, bit1.planes is NULL,
	 * the bit-width isn't 1, the score mode is L1, or the node lacks a
	 * packed code.
	 */
	if (pgturbohybrid_dense_query_1bit_asymmetric && so->tq.bits == 1 &&
		so->tq.bit1.planes != NULL &&
		(TqScoreMode) so->tq.scoreMode != PGTURBOHYBRID_SCORE_L1)
	{
		TqScoreMode mode = (TqScoreMode) so->tq.scoreMode;
		double		dimSqrt = sqrt((double) so->tq.dimensions);
		double		dot;

		so->graphScalarScoredCodes++;
		/* Single-node 1-bit asym has no dedicated single bucket; it is not a
		 * split kernel, so it counts under single_scalar_or_lut. */
		PgturbohybridGraphRecordScoreKernel(so, PGTURBOHYBRID_SCORE_KERNEL_SINGLE_SCALAR_OR_LUT, 1);
		dot = so->tq.ecCorrection +
			(double) PgturbohybridGraphAsymBit1Score(&so->tq, node->code);

		if (mode == PGTURBOHYBRID_SCORE_IP)
			return -(node->scale * dot / dimSqrt);
		if (mode == PGTURBOHYBRID_SCORE_COSINE)
		{
			if (so->tq.queryNorm == 0 || node->scale == 0)
				return 1;
			return 1 - (dot / (sqrt(so->tq.queryNorm) * dimSqrt));
		}
		/* PGTURBOHYBRID_SCORE_L2 */
		{
			double		distance = so->tq.queryNorm +
				((double) node->scale * node->scale) -
				(2 * node->scale * dot / dimSqrt);

			return distance < 0 ? 0 : distance;
		}
	}

#if PGTURBOHYBRID_GRAPH_COMPILE_QUERY_SPLIT
	/*
	 * Single-node query-split fast path for 4-bit and 2-bit codes.  Mirrors the
	 * batch-of-4 dispatch so tail nodes (< 4 left over) and small frontier
	 * expansions use the same int8 SIMD kernel and scoring math as the batch
	 * path instead of falling back to the scalar LUT-gather float path.
	 */
	{
		double		querySplitDistance;

#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
		if (so->tq.u8.enabled &&
			PgturbohybridGraphPackedDistanceU8Split(&so->tq, node->code,
													node->scale, &querySplitDistance))
		{
			PgturbohybridGraphRecordScoreKernel(so, PGTURBOHYBRID_SCORE_KERNEL_SINGLE_U8_SPLIT, 1);
			return querySplitDistance;
		}
#endif
		if (PgturbohybridGraphPackedDistanceQuerySplit4(&so->tq, node->code,
												node->scale, &querySplitDistance) ||
			PgturbohybridGraphPackedDistanceQuerySplit2(&so->tq, node->code,
												node->scale, &querySplitDistance))
		{
			PgturbohybridGraphRecordScoreKernel(so, PGTURBOHYBRID_SCORE_KERNEL_SINGLE_SIGNED_SPLIT, 1);
			return querySplitDistance;
		}
	}
#endif

	so->graphScalarScoredCodes++;
	PgturbohybridGraphRecordScoreKernel(so, PGTURBOHYBRID_SCORE_KERNEL_SINGLE_SCALAR_OR_LUT, 1);
	return PgturbohybridGraphPackedDistance(&so->tq, node->code, node->scale,
								 node->codeNorm, node->norm);
}


#if PGTURBOHYBRID_GRAPH_COMPILE_QUERY_SPLIT
static bool
PgturbohybridGraphScoreNodeBatchQuerySplit4(PgturbohybridGraphScanOpaque so,
								 PgturbohybridGraphScanStorage *storage,
								 uint32 *nodeIds, double *distances)
{
	TqScoreMode mode = (TqScoreMode) so->tq.scoreMode;
	double		dimSqrt;
	double		queryNormSqrt = 0;

	if (!so->tq.signedSplit.enabled || so->tq.dimensions < 1024 ||
		so->tq.bits != PGTURBOHYBRID_DEFAULT_BITS || mode == PGTURBOHYBRID_SCORE_L1)
		return false;

	dimSqrt = sqrt((double) so->tq.dimensions);
	if (mode == PGTURBOHYBRID_SCORE_COSINE)
		queryNormSqrt = sqrt(so->tq.queryNorm);

	for (int j = 0; j < 4; j++)
	{
		PgturbohybridGraphScanNode *node = &storage->nodes[nodeIds[j]];
		double		dot;
		int64		rawDot;

		if (node->code == NULL)
			return false;

#if PGTURBOHYBRID_GRAPH_COMPILE_ARM_DOT
		if (PgturbohybridGraphArmDotprodAvailable())
			rawDot = PgturbohybridGraphQuerySplitRawNeonSdot(&so->tq, node->code);
		else
#endif
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX512VNNI
		if (PgturbohybridGraphAvx512VnniAvailable())
			rawDot = PgturbohybridGraphQuerySplitRawAvx512Vnni(&so->tq, node->code);
		else
#endif
#if PGTURBOHYBRID_GRAPH_COMPILE_AVXVNNI
		if (PgturbohybridGraphAvxVnniAvailable())
			rawDot = PgturbohybridGraphQuerySplitRawAvxVnni(&so->tq, node->code);
		else
#endif
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
		if (PgturbohybridGraphAvx2Available())
			rawDot = PgturbohybridGraphQuerySplitRawAvx2(&so->tq, node->code);
		else
#endif
			return false;

		dot = (double) so->tq.signedSplit.postprocessScale *
			(double) rawDot;
		dot += so->tq.ecCorrection;

		if (mode == PGTURBOHYBRID_SCORE_IP)
			distances[j] = -(node->scale * dot / dimSqrt);
		else if (mode == PGTURBOHYBRID_SCORE_COSINE)
		{
			if (so->tq.queryNorm == 0 || node->scale == 0)
				distances[j] = 1;
			else
				distances[j] = 1 - (dot / (queryNormSqrt * dimSqrt));
		}
		else
		{
			double		distance;

			distance = so->tq.queryNorm + ((double) node->scale * node->scale) -
				(2 * node->scale * dot / dimSqrt);
			distances[j] = distance < 0 ? 0 : distance;
		}
	}

	so->graphScoredCodes += 4;
	so->graphBatchScoredCodes += 4;
	so->graphBatchKernel = so->tq.scoringKernel;
	PgturbohybridGraphRecordScoreKernel(so,
		PgturbohybridGraphSignedSplitBatchBucket(so->tq.scoringKernel), 4);
	return true;
}

static bool
PgturbohybridGraphScoreNodeBatchQuerySplit2(PgturbohybridGraphScanOpaque so,
								 PgturbohybridGraphScanStorage *storage,
								 uint32 *nodeIds, double *distances)
{
	TqScoreMode mode = (TqScoreMode) so->tq.scoreMode;
	double		dimSqrt;
	double		queryNormSqrt = 0;

	if (!so->tq.signedSplit.enabled || so->tq.dimensions < 1024 ||
		so->tq.bits != 2 || mode == PGTURBOHYBRID_SCORE_L1)
		return false;

	dimSqrt = sqrt((double) so->tq.dimensions);
	if (mode == PGTURBOHYBRID_SCORE_COSINE)
		queryNormSqrt = sqrt(so->tq.queryNorm);

	for (int j = 0; j < 4; j++)
	{
		PgturbohybridGraphScanNode *node = &storage->nodes[nodeIds[j]];
		double		dot;
		int64		rawDot;

		if (node->code == NULL)
			return false;

#if PGTURBOHYBRID_GRAPH_COMPILE_ARM_DOT
		if (PgturbohybridGraphArmDotprodAvailable())
			rawDot = PgturbohybridGraphQuerySplit2RawNeonSdot(&so->tq, node->code);
		else
#endif
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX512VNNI
		if (PgturbohybridGraphAvx512VnniAvailable())
			rawDot = PgturbohybridGraphQuerySplit2RawAvx512Vnni(&so->tq, node->code);
		else
#endif
#if PGTURBOHYBRID_GRAPH_COMPILE_AVXVNNI
		if (PgturbohybridGraphAvxVnniAvailable())
			rawDot = PgturbohybridGraphQuerySplit2RawAvxVnni(&so->tq, node->code);
		else
#endif
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
		if (PgturbohybridGraphAvx2Available())
			rawDot = PgturbohybridGraphQuerySplit2RawAvx2(&so->tq, node->code);
		else
#endif
			return false;

		dot = (double) so->tq.signedSplit.postprocessScale *
			(double) rawDot;
		dot += so->tq.ecCorrection;

		if (mode == PGTURBOHYBRID_SCORE_IP)
			distances[j] = -(node->scale * dot / dimSqrt);
		else if (mode == PGTURBOHYBRID_SCORE_COSINE)
		{
			if (so->tq.queryNorm == 0 || node->scale == 0)
				distances[j] = 1;
			else
				distances[j] = 1 - (dot / (queryNormSqrt * dimSqrt));
		}
		else
		{
			double		distance;

			distance = so->tq.queryNorm + ((double) node->scale * node->scale) -
				(2 * node->scale * dot / dimSqrt);
			distances[j] = distance < 0 ? 0 : distance;
		}
	}

	so->graphScoredCodes += 4;
	so->graphBatchScoredCodes += 4;
	so->graphBatchKernel = so->tq.scoringKernel;
	PgturbohybridGraphRecordScoreKernel(so,
		PgturbohybridGraphSignedSplitBatchBucket(so->tq.scoringKernel), 4);
	return true;
}

#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
/*
 * Batch (groups of 4) unsigned-codebook scorer.  Used when the u8 split was
 * prepared for this query.  The raw maddubs/VPDPBUSD kernel and the
 * query-side reciprocal postprocess (precomputed in TqPrepareQueryU8Split) are
 * resolved once per node, so the loop carries no sqrt(), no gating re-check,
 * and no per-node kernel branch.
 *
 * By default (turbohybrid.dense_u8_batch_x4) the four codes are scored by the
 * true 4-candidate x4 kernel in a single pass: the query [low|high] data is
 * loaded once rather than four times, and the four code loads -- which in a
 * real scan point at four scattered graph neighbours -- are issued together so
 * their memory latency overlaps (memory-level parallelism) instead of being
 * serialized one full code at a time.  Kernel ns/code on amd64 (Ice Lake VNNI)
 * over a scattered access pattern (turbohybrid_scorer_bench): ~tied in the
 * compute-bound 10k regime (~1.0-1.05x, codes cache-resident) and ~1.4x faster
 * in the memory-bound 1M regime, where the win comes from.  At the 10k scan
 * level the two are within measurement noise, as expected when codes are
 * cache-resident and the kernel is a small slice of total scan latency.
 * Turning the knob off falls back to four single-node passes; that path is kept
 * for parity testing and as an escape hatch for microarchitectures where the
 * batch does not win.
 */
static bool
PgturbohybridGraphScoreNodeBatchU8Split(PgturbohybridGraphScanOpaque so,
										PgturbohybridGraphScanStorage *storage,
										uint32 *nodeIds, double *distances)
{
	const PgturbohybridGraphTqQuery *tq = &so->tq;
	const uint8 *codes[4];
	float		scales[4];

	if (!tq->u8.enabled)
		return false;

	/* All four must carry a packed code, else let the caller fall back. */
	for (int j = 0; j < 4; j++)
	{
		PgturbohybridGraphScanNode *node = &storage->nodes[nodeIds[j]];

		if (node->code == NULL)
			return false;
		codes[j] = node->code;
		scales[j] = node->scale;
	}

	if (pgturbohybrid_dense_u8_batch_x4)
	{
		/* Default (turbohybrid.dense_u8_batch_x4 = on): the true 4-candidate x4
		 * kernel in one pass, sharing the query [low|high] loads across the four
		 * scattered codes. */
		if (!PgturbohybridGraphPackedDistanceU8Splitx4(tq, codes, scales, distances))
			return false;
		so->graphU8BatchMode = PGTURBOHYBRID_U8_BATCH_X4;
	}
	else
	{
		/* Fallback / benchmark escape hatch (turbohybrid.dense_u8_batch_x4 = off):
		 * four single-node passes over the same four codes. */
		for (int j = 0; j < 4; j++)
		{
			if (!PgturbohybridGraphPackedDistanceU8Split(tq, codes[j], scales[j],
														 &distances[j]))
				return false;
		}
		so->graphU8BatchMode = PGTURBOHYBRID_U8_BATCH_SINGLE;
	}

	so->graphScoredCodes += 4;
	so->graphBatchScoredCodes += 4;
	so->graphBatchKernel = so->tq.scoringKernel;
	PgturbohybridGraphRecordScoreKernel(so,
		PgturbohybridGraphU8SplitBatchBucket(so->tq.scoringKernel), 4);
	return true;
}
#endif

#endif

static bool
PgturbohybridGraphScoreNodeBatchPacked4(PgturbohybridGraphScanOpaque so, PgturbohybridGraphScanStorage *storage,
							 uint32 *nodeIds, double *distances)
{
	TqScoreMode mode = (TqScoreMode) so->tq.scoreMode;
	const uint8 *codes[4];
	float		scales[4];
	double		dots[4];
	int			dim = so->tq.dimensions;
	double		dimSqrt = sqrt((double) dim);

	if (!so->tq.enabled || dim <= 0 || mode == PGTURBOHYBRID_SCORE_L1 ||
		so->tq.bits != PGTURBOHYBRID_DEFAULT_BITS)
		return false;

	for (int j = 0; j < 4; j++)
	{
		PgturbohybridGraphScanNode *node = &storage->nodes[nodeIds[j]];

		if (node->code == NULL)
			return false;

		codes[j] = node->code;
		scales[j] = node->scale;
		dots[j] = so->tq.ecCorrection;
	}

	for (int i = 0; i + 1 < dim; i += 2)
	{
		float	   *loRow = so->tq.lut.table + (i * PGTURBOHYBRID_LUT_WIDTH);
		float	   *hiRow = loRow + PGTURBOHYBRID_LUT_WIDTH;
		int			byteIndex = i / 2;

		for (int j = 0; j < 4; j++)
		{
			uint8		packed = codes[j][byteIndex];

			dots[j] += loRow[packed & 0x0f] + hiRow[packed >> PGTURBOHYBRID_BITS];
		}
	}

	if ((dim & 1) != 0)
	{
		float	   *row = so->tq.lut.table + ((dim - 1) * PGTURBOHYBRID_LUT_WIDTH);
		int			byteIndex = dim / 2;

		for (int j = 0; j < 4; j++)
			dots[j] += row[codes[j][byteIndex] & 0x0f];
	}

	for (int j = 0; j < 4; j++)
	{
		if (mode == PGTURBOHYBRID_SCORE_IP)
			distances[j] = -(scales[j] * dots[j] / dimSqrt);
		else if (mode == PGTURBOHYBRID_SCORE_COSINE)
		{
			if (so->tq.queryNorm == 0 || scales[j] == 0)
				distances[j] = 1;
			else
				distances[j] = 1 - (dots[j] / (sqrt(so->tq.queryNorm) * dimSqrt));
		}
		else
		{
			double		distance;

			distance = so->tq.queryNorm + ((double) scales[j] * scales[j]) -
				(2 * scales[j] * dots[j] / dimSqrt);
			distances[j] = distance < 0 ? 0 : distance;
		}
	}

	so->graphScoredCodes += 4;
	so->graphBatchScoredCodes += 4;
	so->graphBatchKernel = PGTURBOHYBRID_SCORING_SCALAR;
	PgturbohybridGraphRecordScoreKernel(so, PGTURBOHYBRID_SCORE_KERNEL_BATCH_SCALAR_OR_LUT, 4);
	return true;
}

static bool
PgturbohybridGraphScoreNodeBatchPackedLowBits(PgturbohybridGraphScanOpaque so,
								   PgturbohybridGraphScanStorage *storage,
								   uint32 *nodeIds, double *distances)
{
	TqScoreMode mode = (TqScoreMode) so->tq.scoreMode;
	const uint8 *codes[4];
	float		scales[4];
	double		dots[4];
	int			dim = so->tq.dimensions;
	int			bits = so->tq.bits;
	double		dimSqrt = sqrt((double) dim);
	double		queryNormSqrt = 0;

	if (!so->tq.enabled || dim <= 0 || mode == PGTURBOHYBRID_SCORE_L1 ||
		(bits != 1 && bits != 2))
		return false;

	if (mode == PGTURBOHYBRID_SCORE_COSINE)
		queryNormSqrt = sqrt(so->tq.queryNorm);

	for (int j = 0; j < 4; j++)
	{
		PgturbohybridGraphScanNode *node = &storage->nodes[nodeIds[j]];

		if (node->code == NULL)
			return false;

		codes[j] = node->code;
		scales[j] = node->scale;
		dots[j] = so->tq.ecCorrection;
	}

	if (bits == 2)
	{
		int			i = 0;

		for (; i + 4 <= dim; i += 4)
		{
			float	   *row0 = so->tq.lut.table + (i * PGTURBOHYBRID_LUT_WIDTH);
			float	   *row1 = row0 + PGTURBOHYBRID_LUT_WIDTH;
			float	   *row2 = row1 + PGTURBOHYBRID_LUT_WIDTH;
			float	   *row3 = row2 + PGTURBOHYBRID_LUT_WIDTH;
			int			byteIndex = i / 4;

			for (int j = 0; j < 4; j++)
			{
				uint8		packed = codes[j][byteIndex];

				dots[j] += row0[packed & 0x03] +
					row1[(packed >> 2) & 0x03] +
					row2[(packed >> 4) & 0x03] +
					row3[(packed >> 6) & 0x03];
			}
		}

		for (; i < dim; i++)
		{
			float	   *row = so->tq.lut.table + (i * PGTURBOHYBRID_LUT_WIDTH);
			int			byteIndex = i / 4;
			int			shift = (i & 3) * 2;

			for (int j = 0; j < 4; j++)
				dots[j] += row[(codes[j][byteIndex] >> shift) & 0x03];
		}
	}
	else
	{
		int			i = 0;

		for (; i + 8 <= dim; i += 8)
		{
			float	   *row0 = so->tq.lut.table + (i * PGTURBOHYBRID_LUT_WIDTH);
			float	   *row1 = row0 + PGTURBOHYBRID_LUT_WIDTH;
			float	   *row2 = row1 + PGTURBOHYBRID_LUT_WIDTH;
			float	   *row3 = row2 + PGTURBOHYBRID_LUT_WIDTH;
			float	   *row4 = row3 + PGTURBOHYBRID_LUT_WIDTH;
			float	   *row5 = row4 + PGTURBOHYBRID_LUT_WIDTH;
			float	   *row6 = row5 + PGTURBOHYBRID_LUT_WIDTH;
			float	   *row7 = row6 + PGTURBOHYBRID_LUT_WIDTH;
			int			byteIndex = i / 8;

			for (int j = 0; j < 4; j++)
			{
				uint8		packed = codes[j][byteIndex];

				dots[j] += row0[packed & 0x01] +
					row1[(packed >> 1) & 0x01] +
					row2[(packed >> 2) & 0x01] +
					row3[(packed >> 3) & 0x01] +
					row4[(packed >> 4) & 0x01] +
					row5[(packed >> 5) & 0x01] +
					row6[(packed >> 6) & 0x01] +
					row7[(packed >> 7) & 0x01];
			}
		}

		for (; i < dim; i++)
		{
			float	   *row = so->tq.lut.table + (i * PGTURBOHYBRID_LUT_WIDTH);
			int			byteIndex = i / 8;
			int			shift = i & 7;

			for (int j = 0; j < 4; j++)
				dots[j] += row[(codes[j][byteIndex] >> shift) & 0x01];
		}
	}

	for (int j = 0; j < 4; j++)
	{
		if (mode == PGTURBOHYBRID_SCORE_IP)
			distances[j] = -(scales[j] * dots[j] / dimSqrt);
		else if (mode == PGTURBOHYBRID_SCORE_COSINE)
		{
			if (so->tq.queryNorm == 0 || scales[j] == 0)
				distances[j] = 1;
			else
				distances[j] = 1 - (dots[j] / (queryNormSqrt * dimSqrt));
		}
		else
		{
			double		distance;

			distance = so->tq.queryNorm + ((double) scales[j] * scales[j]) -
				(2 * scales[j] * dots[j] / dimSqrt);
			distances[j] = distance < 0 ? 0 : distance;
		}
	}

	so->graphScoredCodes += 4;
	so->graphBatchScoredCodes += 4;
	so->graphBatchKernel = PGTURBOHYBRID_SCORING_SCALAR;
	PgturbohybridGraphRecordScoreKernel(so, PGTURBOHYBRID_SCORE_KERNEL_BATCH_SCALAR_OR_LUT, 4);
	return true;
}

static int
PgturbohybridGraphBit1PopcntRaw(PgturbohybridGraphScanOpaque so, const uint8 *code)
{
	return PgturbohybridGraphBit1PopcntRawCodes(so->tq.querySignBits, code,
									 so->tq.dimensions);
}

/*
 * Batch dispatch for the asymmetric 1-bit scoring path.
 *
 * Active when the private asymmetric 1-bit path is enabled and the query
 * precompute populated tq->bit1.planes (1-bit indexes only).  When
 * either condition is missing, returns false so the existing 1-bit
 * popcount or LUT path takes over — preserving baseline behaviour
 * for users who don't opt in.
 *
 * Per-node scoring uses PgturbohybridGraphAsymBit1Score (AVX2 / NEON / scalar
 * dispatched).  The IP / Cosine / L2 postprocess mirrors the
 * symmetric popcount path exactly so callers see the same shape of
 * output — only the dot value carries more magnitude information.
 */
static bool
PgturbohybridGraphScoreNodeBatchAsymBit1(PgturbohybridGraphScanOpaque so,
							  PgturbohybridGraphScanStorage *storage,
							  uint32 *nodeIds, double *distances)
{
	TqScoreMode mode = (TqScoreMode) so->tq.scoreMode;
	double		dimSqrt;
	double		queryNormSqrt = 0;

	if (!pgturbohybrid_dense_query_1bit_asymmetric || !so->tq.enabled ||
		so->tq.bits != 1 || so->tq.bit1.planes == NULL ||
		so->tq.dimensions <= 0 || mode == PGTURBOHYBRID_SCORE_L1)
		return false;

	dimSqrt = sqrt((double) so->tq.dimensions);
	if (mode == PGTURBOHYBRID_SCORE_COSINE)
		queryNormSqrt = sqrt(so->tq.queryNorm);

	for (int j = 0; j < 4; j++)
	{
		PgturbohybridGraphScanNode *node = &storage->nodes[nodeIds[j]];
		double		dot;

		if (node->code == NULL)
			return false;

		dot = so->tq.ecCorrection +
			(double) PgturbohybridGraphAsymBit1Score(&so->tq, node->code);

		if (mode == PGTURBOHYBRID_SCORE_IP)
			distances[j] = -(node->scale * dot / dimSqrt);
		else if (mode == PGTURBOHYBRID_SCORE_COSINE)
		{
			if (so->tq.queryNorm == 0 || node->scale == 0)
				distances[j] = 1;
			else
				distances[j] = 1 - (dot / (queryNormSqrt * dimSqrt));
		}
		else
		{
			double		distance;

			distance = so->tq.queryNorm + ((double) node->scale * node->scale) -
				(2 * node->scale * dot / dimSqrt);
			distances[j] = distance < 0 ? 0 : distance;
		}
	}

	so->graphScoredCodes += 4;
	so->graphBatchScoredCodes += 4;
	so->graphBatchKernel = so->tq.scoringKernel;
	PgturbohybridGraphRecordScoreKernel(so, PGTURBOHYBRID_SCORE_KERNEL_BATCH_1BIT_ASYM, 4);
	return true;
}

static bool
PgturbohybridGraphScoreNodeBatchPopcntLowBits(PgturbohybridGraphScanOpaque so,
								   PgturbohybridGraphScanStorage *storage,
								   uint32 *nodeIds, double *distances)
{
	TqScoreMode mode = (TqScoreMode) so->tq.scoreMode;
	double		dimSqrt;
	double		queryNormSqrt = 0;
	double		center;

	if (!pgturbohybrid_dense_graph_lowbit_popcnt || !so->tq.enabled ||
		so->tq.bits != 1 || so->tq.querySignBits == NULL ||
		so->tq.dimensions <= 0 || mode == PGTURBOHYBRID_SCORE_L1)
		return false;

	dimSqrt = sqrt((double) so->tq.dimensions);
	center = PgturbohybridGraphCodeCenter(1, 1);
	if (mode == PGTURBOHYBRID_SCORE_COSINE)
		queryNormSqrt = sqrt(so->tq.queryNorm);

	for (int j = 0; j < 4; j++)
	{
		PgturbohybridGraphScanNode *node = &storage->nodes[nodeIds[j]];
		double		dot;

		if (node->code == NULL)
			return false;

		dot = so->tq.ecCorrection +
			(center * (double) PgturbohybridGraphBit1PopcntRaw(so, node->code));

		if (mode == PGTURBOHYBRID_SCORE_IP)
			distances[j] = -(node->scale * dot / dimSqrt);
		else if (mode == PGTURBOHYBRID_SCORE_COSINE)
		{
			if (so->tq.queryNorm == 0 || node->scale == 0)
				distances[j] = 1;
			else
				distances[j] = 1 - (dot / (queryNormSqrt * dimSqrt));
		}
		else
		{
			double		distance;

			distance = so->tq.queryNorm + ((double) node->scale * node->scale) -
				(2 * node->scale * dot / dimSqrt);
			distances[j] = distance < 0 ? 0 : distance;
		}
	}

	so->graphScoredCodes += 4;
	so->graphBatchScoredCodes += 4;
	so->graphBatchKernel = PGTURBOHYBRID_SCORING_SCALAR;
	PgturbohybridGraphRecordScoreKernel(so, PGTURBOHYBRID_SCORE_KERNEL_BATCH_SCALAR_OR_LUT, 4);
	return true;
}

void
PgturbohybridGraphScoreNodeBatch(PgturbohybridGraphScanOpaque so, PgturbohybridGraphScanStorage *storage,
					  uint32 *nodeIds, int nodeCount, double *distances,
					  Datum query)
{
	/* Traversal-level batching metric: one call per ScoreNodeBatch invocation,
	 * nodeCount nodes fed to it (graph_avg_batch_size = nodes / calls). */
	so->graphBatchCalls++;
	so->graphBatchNodes += nodeCount;

	if (pgturbohybrid_dense_graph_batch_scoring == PGTURBOHYBRID_GRAPH_BATCH_OFF ||
		pgturbohybrid_dense_graph_batch_size < 4)
	{
		for (int i = 0; i < nodeCount; i++)
			distances[i] = PgturbohybridGraphScoreNode(so, &storage->nodes[nodeIds[i]]);
		return;
	}

	if (PgturbohybridGraphUseExactLowBitRouting(so, query))
	{
		bool		exactBatch = true;

		for (int j = 0; j < nodeCount; j++)
		{
			PgturbohybridGraphScanNode *node = &storage->nodes[nodeIds[j]];

			if (node->exactVector == NULL)
			{
				exactBatch = false;
				break;
			}
		}

		if (exactBatch)
		{
			for (int j = 0; j < nodeCount; j++)
				distances[j] = PgturbohybridGraphExactVectorDistance(so, query,
														  storage->nodes[nodeIds[j]].exactVector);
			return;
		}
	}

	for (int i = 0; i < nodeCount;)
	{
		for (int j = i; pgturbohybrid_dense_graph_prefetch && j < Min(i + 8, nodeCount); j++)
		{
			PgturbohybridGraphScanNode *node = &storage->nodes[nodeIds[j]];

			/*
			 * When the code arena exceeds CPU cache, each code is a scattered
			 * RAM read and the scoring kernel streams all codeBytes — prefetch
			 * every cache line to hide the full-code latency.  For cache-resident
			 * (small/medium) arenas the codes are already hot, so the extra
			 * prefetches are pure overhead: just touch the first line.
			 */
			if (node->code != NULL)
			{
				if (so->graphLargeCodeArena)
					for (Size off = 0; off < so->tq.codeBytes; off += 64)
						PGTURBOHYBRID_GRAPH_PREFETCH_READ(node->code + off);
				else
					PGTURBOHYBRID_GRAPH_PREFETCH_READ(node->code);
			}
		}

		if (i + 4 <= nodeCount)
		{
			bool		batchScored =
#if PGTURBOHYBRID_GRAPH_COMPILE_QUERY_SPLIT
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
				PgturbohybridGraphScoreNodeBatchU8Split(so, storage, nodeIds + i,
												distances + i) ||
#endif
				PgturbohybridGraphScoreNodeBatchQuerySplit4(so, storage, nodeIds + i,
												 distances + i) ||
				PgturbohybridGraphScoreNodeBatchQuerySplit2(so, storage, nodeIds + i,
												 distances + i) ||
#endif
				PgturbohybridGraphScoreNodeBatchPacked4(so, storage, nodeIds + i,
											 distances + i) ||
				PgturbohybridGraphScoreNodeBatchAsymBit1(so, storage, nodeIds + i,
											  distances + i) ||
				PgturbohybridGraphScoreNodeBatchPopcntLowBits(so, storage, nodeIds + i,
												   distances + i) ||
				PgturbohybridGraphScoreNodeBatchPackedLowBits(so, storage, nodeIds + i,
												   distances + i);

			if (batchScored)
			{
				i += 4;
				continue;
			}
		}

		distances[i] = PgturbohybridGraphScoreNode(so, &storage->nodes[nodeIds[i]]);
		i++;
	}
}

bool
PgturbohybridGraphCodeCodeWeightedRawSimdSelf(const uint8 *code, int dimensions, int bits,
								   const float *ecScale, double *raw)
{
	double		dPrimeSqMax = 0.0;
	double		weightScale;
	double		codebookScaleSq;
	int16	   *dPrimeSqI16;
	int64		rawI64 = 0;
	bool		simdRan = false;

	if (raw == NULL)
		return false;
	*raw = 0.0;

	if (code == NULL || ecScale == NULL)
		return false;

	for (int d = 0; d < dimensions; d++)
	{
		double		s = (double) ecScale[d];

		if (fabs(s) > FLT_EPSILON)
		{
			double		w = 1.0 / (s * s);

			if (w > dPrimeSqMax)
				dPrimeSqMax = w;
		}
	}

	if (dPrimeSqMax <= FLT_EPSILON)
		return false;

	weightScale = ((double) INT16_MAX - 1.0) / dPrimeSqMax;
	dPrimeSqI16 = palloc(sizeof(int16) * dimensions);

	for (int d = 0; d < dimensions; d++)
	{
		double		s = (double) ecScale[d];
		double		w = (fabs(s) > FLT_EPSILON) ? 1.0 / (s * s) : 0.0;
		double		q = round(w * weightScale);

		if (q < 0.0)
			q = 0.0;
		if (q > (double) (INT16_MAX - 1))
			q = (double) (INT16_MAX - 1);
		dPrimeSqI16[d] = (int16) q;
	}

	if (bits == PGTURBOHYBRID_DEFAULT_BITS)
		codebookScaleSq = PGTURBOHYBRID_GRAPH_CODEBOOK_SCALE * PGTURBOHYBRID_GRAPH_CODEBOOK_SCALE;
	else
		codebookScaleSq = PGTURBOHYBRID_GRAPH_CODEBOOK2_SCALE * PGTURBOHYBRID_GRAPH_CODEBOOK2_SCALE;

#if PGTURBOHYBRID_GRAPH_COMPILE_ARM_DOT
	if (bits == PGTURBOHYBRID_DEFAULT_BITS)
		rawI64 = PgturbohybridGraphCodeCodeWeightedRawNeonSdot(code, code,
													dPrimeSqI16,
													dimensions);
	else
		rawI64 = PgturbohybridGraphCodeCode2WeightedRawNeonSdot(code, code,
													 dPrimeSqI16,
													 dimensions);
	simdRan = true;
#endif
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
	if (!simdRan && PgturbohybridGraphAvx2Available())
	{
		if (bits == PGTURBOHYBRID_DEFAULT_BITS)
			rawI64 = PgturbohybridGraphCodeCodeWeightedRawAvx2(code, code,
													dPrimeSqI16,
													dimensions);
		else
			rawI64 = PgturbohybridGraphCodeCode2WeightedRawAvx2(code, code,
													 dPrimeSqI16,
													 dimensions);
		simdRan = true;
	}
#endif

	if (simdRan)
		*raw = (double) rawI64 / (weightScale * codebookScaleSq);

	pfree(dPrimeSqI16);
	return simdRan;
}
