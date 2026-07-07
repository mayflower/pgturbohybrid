/*
 * pgturbohybrid_quant_score_signed_x86.c
 *
 * x86 SIMD dense quantized-scoring kernels, extracted verbatim from
 * pgturbohybrid_quant_score.c (no behaviour change): the AVX2 / AVX-VNNI /
 * AVX-512 VNNI signed 4-bit/2-bit query-split scorers, the code-code and
 * weighted code-code raw scorers, their shared helpers, the CPU-feature probes,
 * and the tightly-coupled AVX2 exact-vector distance helpers (they share the
 * float horizontal-sum helper).  The generic scan-time / build-time dispatch
 * stays in pgturbohybrid_quant_score.c and calls the non-static entry points
 * declared in pgturbohybrid_quant_score_internal.h.
 *
 * On non-x86 (or SIMD_BUILD=none) PGTURBOHYBRID_GRAPH_COMPILE_AVX2 is 0, so the
 * whole body compiles to an empty object.  The compile/target guard macros below
 * are kept byte-for-byte in sync with the copies in pgturbohybrid_quant_score.c.
 */
#include "postgres.h"

#include <math.h>
#include <string.h>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#define PGTURBOHYBRID_GRAPH_X86 1
#else
#define PGTURBOHYBRID_GRAPH_X86 0
#endif

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

#include "pgturbohybrid_quant_score.h"
#include "pgturbohybrid_quant_score_internal.h"

/* Codebook tables + scale/coef anchors (shared single source of truth). */
#include "pgturbohybrid_quant_codebook.h"

#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2 && !PGTURBOHYBRID_GRAPH_COMPILE_ARM_DOT
static const int8 PgturbohybridGraphCodebook2PairEvenI8[16] = {
	-127, -38, 38, 127,
	-127, -38, 38, 127,
	-127, -38, 38, 127,
	-127, -38, 38, 127,
};
static const int8 PgturbohybridGraphCodebook2PairOddI8[16] = {
	-127, -127, -127, -127,
	-38, -38, -38, -38,
	38, 38, 38, 38,
	127, 127, 127, 127,
};
#endif

#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
bool
PgturbohybridGraphAvx2Available(void)
{
	static int	available = -1;

	if (pgturbohybrid_dense_simd_force != PGTURBOHYBRID_SIMD_FORCE_AUTO &&
		pgturbohybrid_dense_simd_force != PGTURBOHYBRID_SIMD_FORCE_AVX2)
		return false;

	if (available >= 0)
		return available != 0;

#if defined(__AVX2__)
	available = 1;
#elif PGTURBOHYBRID_GRAPH_X86 && (defined(__GNUC__) || defined(__clang__))
	available = __builtin_cpu_supports("avx2") ? 1 : 0;
#else
	available = 0;
#endif
	return available != 0;
}

static inline int64 PGTURBOHYBRID_GRAPH_AVX2_TARGET
PgturbohybridGraphHorizontalSumI32Avx2(__m256i v)
{
	int32		s[8];

	_mm256_storeu_si256((__m256i *) s, v);
	return (int64) s[0] + s[1] + s[2] + s[3] +
		s[4] + s[5] + s[6] + s[7];
}

static inline double PGTURBOHYBRID_GRAPH_AVX2_TARGET
PgturbohybridGraphHorizontalSumF32Avx2(__m256 v)
{
	float		s[8];

	_mm256_storeu_ps(s, v);
	return (double) s[0] + s[1] + s[2] + s[3] +
		s[4] + s[5] + s[6] + s[7];
}

static inline __m128i PGTURBOHYBRID_GRAPH_AVX2_TARGET
PgturbohybridGraphExpandPacked4Avx2(const uint8 *code)
{
	const __m128i mask = _mm_set1_epi8(0x0f);
	const __m128i codebook = _mm_loadu_si128((const __m128i *) PgturbohybridGraphCodebookI8);
	__m128i	packed = _mm_loadl_epi64((const __m128i *) code);
	__m128i	lo = _mm_and_si128(packed, mask);
	__m128i	hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
	__m128i	idx = _mm_unpacklo_epi8(lo, hi);

	return _mm_shuffle_epi8(codebook, idx);
}

static inline __m128i PGTURBOHYBRID_GRAPH_AVX2_TARGET
PgturbohybridGraphExpandPacked2Avx2(const uint8 *code)
{
	int32		codeI32;
	__m128i		data;
	__m128i		loMask = _mm_set1_epi8(0x0F);
	__m128i		loNibs;
	__m128i		hiNibs;
	__m128i		pairIdx;
	__m128i		tEven;
	__m128i		tOdd;
	__m128i		cEven;
	__m128i		cOdd;

	memcpy(&codeI32, code, sizeof(codeI32));
	data = _mm_cvtsi32_si128(codeI32);
	loNibs = _mm_and_si128(data, loMask);
	hiNibs = _mm_and_si128(_mm_srli_epi16(data, 4), loMask);
	pairIdx = _mm_unpacklo_epi8(loNibs, hiNibs);

	tEven = _mm_loadu_si128((const __m128i *) PgturbohybridGraphCodebook2PairEvenI8);
	tOdd = _mm_loadu_si128((const __m128i *) PgturbohybridGraphCodebook2PairOddI8);
	cEven = _mm_shuffle_epi8(tEven, pairIdx);
	cOdd = _mm_shuffle_epi8(tOdd, pairIdx);

	return _mm_unpacklo_epi8(cEven, cOdd);
}

static inline __m256i PGTURBOHYBRID_GRAPH_AVX2_TARGET
PgturbohybridGraphDotI8x16Avx2(__m128i a, __m128i b)
{
	__m256i	a16 = _mm256_cvtepi8_epi16(a);
	__m256i	b16 = _mm256_cvtepi8_epi16(b);

	return _mm256_madd_epi16(a16, b16);
}

#define TQ_CC_NAME PgturbohybridGraphQuerySplitRawAvx2
#define TQ_CC_EXPAND PgturbohybridGraphExpandPacked4Avx2
#define TQ_CC_STRIDE 8
#define TQ_CC_TAILADD 1
#define TQ_CC_TAILDIV 2
#include "pgturbohybrid_quant_score_x86_querysplit_avx2.inc"

#define TQ_CC_NAME PgturbohybridGraphQuerySplit2RawAvx2
#define TQ_CC_EXPAND PgturbohybridGraphExpandPacked2Avx2
#define TQ_CC_STRIDE 4
#define TQ_CC_TAILADD 3
#define TQ_CC_TAILDIV 4
#include "pgturbohybrid_quant_score_x86_querysplit_avx2.inc"

/*
 * AVX2 SIMD weighted symmetric (code-code) kernels for TQ+.
 *
 * For each chunk of 16 coords (4-bit unpack via shuffle, 2-bit unpack
 * via the pair-table), expand a/b to i8x16, widen to i16x16, then:
 *
 *    prod = c_a · c_b              (i16, fits since |c| ≤ 127)
 *    pw   = madd(prod, weights)    (Σ pairs of i16 → i32, 8 lanes)
 *    acc += widen i32 → i64
 *
 * Returns Σ c_a[d] · c_b[d] · D'²_i16[d] as i64.  Caller divides by
 * `weight_scale · CODEBOOK_SCALE²` to recover the f32 weighted dot.
 *
 * No pruning (unlike the unweighted kernel) — every coord must
 * contribute its weight, otherwise the formula degenerates.  At
 * dim=1536 this does more work than the unweighted kernel (32
 * chunks vs 96 chunks scanned, plus the widen-mul-madd overhead).
 * Keep benchmark claims about this path in benchmark artifacts, not
 * source comments.
 */
static inline int64 PGTURBOHYBRID_GRAPH_AVX2_TARGET
PgturbohybridGraphHorizontalSumI64Avx2(__m256i v)
{
	int64		s[4];

	_mm256_storeu_si256((__m256i *) s, v);
	return s[0] + s[1] + s[2] + s[3];
}

static inline __m256i PGTURBOHYBRID_GRAPH_AVX2_TARGET
PgturbohybridGraphWeightedDotI8x16Avx2(__m128i ca, __m128i cb,
							 const int16 *weightsAt)
{
	__m256i		caI16 = _mm256_cvtepi8_epi16(ca);
	__m256i		cbI16 = _mm256_cvtepi8_epi16(cb);
	__m256i		prod = _mm256_mullo_epi16(caI16, cbI16);
	__m256i		w = _mm256_loadu_si256((const __m256i *) weightsAt);
	__m256i		pw = _mm256_madd_epi16(prod, w);
	__m256i		pwLoI64 = _mm256_cvtepi32_epi64(_mm256_castsi256_si128(pw));
	__m256i		pwHiI64 = _mm256_cvtepi32_epi64(_mm256_extracti128_si256(pw, 1));

	return _mm256_add_epi64(pwLoI64, pwHiI64);
}

#define TQ_CC_NAME PgturbohybridGraphCodeCodeWeightedRawAvx2
#define TQ_CC_EXPAND PgturbohybridGraphExpandPacked4Avx2
#define TQ_CC_STRIDE 8
#define TQ_CC_TAILADD 1
#define TQ_CC_TAILDIV 2
#include "pgturbohybrid_quant_score_x86_ccweighted_avx2.inc"

#define TQ_CC_NAME PgturbohybridGraphCodeCode2WeightedRawAvx2
#define TQ_CC_EXPAND PgturbohybridGraphExpandPacked2Avx2
#define TQ_CC_STRIDE 4
#define TQ_CC_TAILADD 3
#define TQ_CC_TAILDIV 4
#include "pgturbohybrid_quant_score_x86_ccweighted_avx2.inc"

#if PGTURBOHYBRID_GRAPH_COMPILE_AVX512_WEIGHTED
bool
PgturbohybridGraphAvx512WeightedAvailable(void)
{
	static int	available = -1;

	/* Valgrind cannot execute AVX-512; fall back to AVX2/scalar. */
	if (PgturbohybridGraphRunningUnderValgrind())
		return false;

	if (pgturbohybrid_dense_graph_avx512_weighted == PGTURBOHYBRID_GRAPH_AVX512_WEIGHTED_OFF ||
		pgturbohybrid_dense_simd_force == PGTURBOHYBRID_SIMD_FORCE_SCALAR)
		return false;

	if (pgturbohybrid_dense_simd_force != PGTURBOHYBRID_SIMD_FORCE_AUTO &&
		pgturbohybrid_dense_simd_force != PGTURBOHYBRID_SIMD_FORCE_AVX512VNNI)
		return false;

	if (available >= 0)
		return available != 0;

#if defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512DQ__) && defined(__AVX512VL__)
	available = 1;
#elif PGTURBOHYBRID_GRAPH_X86 && (defined(__GNUC__) || defined(__clang__))
	available = __builtin_cpu_supports("avx512f") &&
		__builtin_cpu_supports("avx512bw") &&
		__builtin_cpu_supports("avx512dq") &&
		__builtin_cpu_supports("avx512vl") ? 1 : 0;
#else
	available = 0;
#endif
	return available != 0;
}

static inline int64 PGTURBOHYBRID_GRAPH_AVX512_WEIGHTED_TARGET
PgturbohybridGraphHorizontalSumI32x16Avx512(__m512i v)
{
	int32		s[16];
	int64		sum = 0;

	_mm512_storeu_si512((__m512i *) s, v);
	for (int i = 0; i < 16; i++)
		sum += s[i];

	return sum;
}

static inline __m256i PGTURBOHYBRID_GRAPH_AVX512_WEIGHTED_TARGET
PgturbohybridGraphExpandPacked4x32Avx512(const uint8 *code)
{
	__m128i		lo = PgturbohybridGraphExpandPacked4Avx2(code);
	__m128i		hi = PgturbohybridGraphExpandPacked4Avx2(code + 8);

	return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

static inline __m256i PGTURBOHYBRID_GRAPH_AVX512_WEIGHTED_TARGET
PgturbohybridGraphExpandPacked2x32Avx512(const uint8 *code)
{
	__m128i		lo = PgturbohybridGraphExpandPacked2Avx2(code);
	__m128i		hi = PgturbohybridGraphExpandPacked2Avx2(code + 4);

	return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

static inline int64 PGTURBOHYBRID_GRAPH_AVX512_WEIGHTED_TARGET
PgturbohybridGraphWeightedDotI8x32Avx512(__m256i ca, __m256i cb,
							  const int16 *weightsAt)
{
	__m512i		caI16 = _mm512_cvtepi8_epi16(ca);
	__m512i		cbI16 = _mm512_cvtepi8_epi16(cb);
	__m512i		prod = _mm512_mullo_epi16(caI16, cbI16);
	__m512i		w = _mm512_loadu_si512((const __m512i *) weightsAt);
	__m512i		pw = _mm512_madd_epi16(prod, w);

	return PgturbohybridGraphHorizontalSumI32x16Avx512(pw);
}

#define TQ_CC_NAME PgturbohybridGraphCodeCodeWeightedRawAvx512
#define TQ_CC_EXPAND PgturbohybridGraphExpandPacked4x32Avx512
#define TQ_CC_STRIDE 16
#define TQ_CC_TAILADD 1
#define TQ_CC_TAILDIV 2
#include "pgturbohybrid_quant_score_x86_ccweighted_avx512.inc"

#define TQ_CC_NAME PgturbohybridGraphCodeCode2WeightedRawAvx512
#define TQ_CC_EXPAND PgturbohybridGraphExpandPacked2x32Avx512
#define TQ_CC_STRIDE 8
#define TQ_CC_TAILADD 3
#define TQ_CC_TAILDIV 4
#include "pgturbohybrid_quant_score_x86_ccweighted_avx512.inc"
#endif

#define TQ_CC_NAME PgturbohybridGraphCodeCodeRawAvx2
#define TQ_CC_EXPAND PgturbohybridGraphExpandPacked4Avx2
#define TQ_CC_STRIDE 8
#define TQ_CC_TAILADD 1
#define TQ_CC_TAILDIV 2
#include "pgturbohybrid_quant_score_x86_codecode_avx2.inc"

#define TQ_CC_NAME PgturbohybridGraphCodeCode2RawAvx2
#define TQ_CC_EXPAND PgturbohybridGraphExpandPacked2Avx2
#define TQ_CC_STRIDE 4
#define TQ_CC_TAILADD 3
#define TQ_CC_TAILDIV 4
#include "pgturbohybrid_quant_score_x86_codecode_avx2.inc"

#if PGTURBOHYBRID_GRAPH_COMPILE_AVX512VNNI
bool
PgturbohybridGraphAvx512VnniAvailable(void)
{
	static int	available = -1;

	/* Valgrind cannot execute AVX-512 VNNI; fall back to AVX2/scalar. */
	if (PgturbohybridGraphRunningUnderValgrind())
		return false;

	/*
	 * The private AVX-512 VNNI policy is consulted on every call.  The
	 * CPU-feature probe is still memoised.
	 */
	if (!pgturbohybrid_dense_graph_avx512vnni)
		return false;
	if (pgturbohybrid_dense_simd_force != PGTURBOHYBRID_SIMD_FORCE_AUTO &&
		pgturbohybrid_dense_simd_force != PGTURBOHYBRID_SIMD_FORCE_AVX512VNNI)
		return false;

	if (available >= 0)
		return available != 0;

#if defined(__AVX512VNNI__) && defined(__AVX512VL__) && defined(__AVX512BW__)
	available = 1;
#elif PGTURBOHYBRID_GRAPH_X86 && (defined(__GNUC__) || defined(__clang__))
	available = __builtin_cpu_supports("avx512vnni") &&
		__builtin_cpu_supports("avx512vl") &&
		__builtin_cpu_supports("avx512bw") ? 1 : 0;
#else
	available = 0;
#endif
	return available != 0;
}

/*
 * AVX-512 VNNI ZMM scoring kernels.
 *
 * Process 64 dims per iteration via _mm512_dpbusd_epi32 on ZMM, with
 * 32-dim YMM and 16-dim XMM tails for residuals. Same XOR-0x80 trick as
 * the AVX-VNNI tier; the -128 * sum(c) correction is hoisted into a third
 * VNNI accumulator that runs in parallel.
 *
 * AVX-512 dispatch should be capped at hosts that don't downclock heavily
 * under wide vectors (Ice Lake server, Sapphire Rapids, Zen 4 server, and
 * newer). Skylake-X / Cascade Lake suffer measurable drops; the 256-bit
 * AVX-VNNI tier is preferred there.
 */

static inline __m512i PGTURBOHYBRID_GRAPH_AVX512VNNI_TARGET
PgturbohybridGraphExpandPacked4Avx512(const uint8 *code)
{
	__m128i		c0 = PgturbohybridGraphExpandPacked4Avx2(code + 0);
	__m128i		c1 = PgturbohybridGraphExpandPacked4Avx2(code + 8);
	__m128i		c2 = PgturbohybridGraphExpandPacked4Avx2(code + 16);
	__m128i		c3 = PgturbohybridGraphExpandPacked4Avx2(code + 24);
	__m256i		lo = _mm256_inserti128_si256(_mm256_castsi128_si256(c0), c1, 1);
	__m256i		hi = _mm256_inserti128_si256(_mm256_castsi128_si256(c2), c3, 1);

	return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

static inline __m512i PGTURBOHYBRID_GRAPH_AVX512VNNI_TARGET
PgturbohybridGraphExpandPacked2Avx512(const uint8 *code)
{
	__m128i		c0 = PgturbohybridGraphExpandPacked2Avx2(code + 0);
	__m128i		c1 = PgturbohybridGraphExpandPacked2Avx2(code + 4);
	__m128i		c2 = PgturbohybridGraphExpandPacked2Avx2(code + 8);
	__m128i		c3 = PgturbohybridGraphExpandPacked2Avx2(code + 12);
	__m256i		lo = _mm256_inserti128_si256(_mm256_castsi128_si256(c0), c1, 1);
	__m256i		hi = _mm256_inserti128_si256(_mm256_castsi128_si256(c2), c3, 1);

	return _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
}

static inline int64 PGTURBOHYBRID_GRAPH_AVX512VNNI_TARGET
PgturbohybridGraphHorizontalSumI32x4Avx512Vnni(__m128i v)
{
	int32		s[4];

	_mm_storeu_si128((__m128i *) s, v);
	return (int64) s[0] + s[1] + s[2] + s[3];
}

#define TQ_CC_NAME PgturbohybridGraphQuerySplitRawAvx512Vnni
#define TQ_CC_EXPAND512 PgturbohybridGraphExpandPacked4Avx512
#define TQ_CC_EXPAND PgturbohybridGraphExpandPacked4Avx2
#define TQ_CC_STRIDE 8
#define TQ_CC_TAILADD 1
#define TQ_CC_TAILDIV 2
#include "pgturbohybrid_quant_score_x86_querysplit_avx512vnni.inc"

#define TQ_CC_NAME PgturbohybridGraphQuerySplit2RawAvx512Vnni
#define TQ_CC_EXPAND512 PgturbohybridGraphExpandPacked2Avx512
#define TQ_CC_EXPAND PgturbohybridGraphExpandPacked2Avx2
#define TQ_CC_STRIDE 4
#define TQ_CC_TAILADD 3
#define TQ_CC_TAILDIV 4
#include "pgturbohybrid_quant_score_x86_querysplit_avx512vnni.inc"

#define TQ_CC_NAME PgturbohybridGraphCodeCodeRawAvx512Vnni
#define TQ_CC_EXPAND512 PgturbohybridGraphExpandPacked4Avx512
#define TQ_CC_EXPAND PgturbohybridGraphExpandPacked4Avx2
#define TQ_CC_STRIDE 8
#define TQ_CC_TAILADD 1
#define TQ_CC_TAILDIV 2
#include "pgturbohybrid_quant_score_x86_codecode_avx512vnni.inc"

#define TQ_CC_NAME PgturbohybridGraphCodeCode2RawAvx512Vnni
#define TQ_CC_EXPAND512 PgturbohybridGraphExpandPacked2Avx512
#define TQ_CC_EXPAND PgturbohybridGraphExpandPacked2Avx2
#define TQ_CC_STRIDE 4
#define TQ_CC_TAILADD 3
#define TQ_CC_TAILDIV 4
#include "pgturbohybrid_quant_score_x86_codecode_avx512vnni.inc"
#endif

#if PGTURBOHYBRID_GRAPH_COMPILE_AVXVNNI
bool
PgturbohybridGraphAvxVnniAvailable(void)
{
	static int	available = -1;

	/* Valgrind cannot execute AVX-VNNI; fall back to AVX2/scalar. */
	if (PgturbohybridGraphRunningUnderValgrind())
		return false;

	/* Private policy switch for AVX VNNI dispatch. */
	if (!pgturbohybrid_dense_graph_avxvnni)
		return false;
	if (pgturbohybrid_dense_simd_force != PGTURBOHYBRID_SIMD_FORCE_AUTO &&
		pgturbohybrid_dense_simd_force != PGTURBOHYBRID_SIMD_FORCE_AVXVNNI)
		return false;

	if (available >= 0)
		return available != 0;

#if defined(__AVXVNNI__)
	available = 1;
#elif PGTURBOHYBRID_GRAPH_X86 && \
	(defined(__GNUC__) && !defined(__clang__) && __GNUC__ >= 11) || \
	(defined(__clang__) && __clang_major__ >= 18)
	available = __builtin_cpu_supports("avxvnni") ? 1 : 0;
#else
	available = 0;
#endif
	return available != 0;
}

/*
 * AVX-VNNI YMM scoring kernels.
 *
 * Each iteration consumes 32 i8 dims via vpdpbusd on a 256-bit register
 * (one fused uop on Alder Lake / Zen 4 desktop, three or four on plain
 * AVX2 via vpmaddubsw + vpmaddwd + vpaddd). Signed-vs-signed dot product
 * is synthesised from VNNI's u8 * i8 form with the standard XOR-0x80
 * trick:
 *
 *   sum(a_signed * b_signed)
 *     = sum((a_signed XOR 0x80) * b_signed) - 128 * sum(b_signed)
 *
 * The right-hand correction is hoisted out of the inner loop with a third
 * VNNI accumulator running `dpbusd(accCSum, ones_u8, b_signed)`, which
 * produces sum(b_signed) once at the end and replaces the per-chunk
 * cvtepi8_epi16 / madd / hsum chain that the XMM version performed.
 */

static inline int64 PGTURBOHYBRID_GRAPH_AVXVNNI_TARGET
PgturbohybridGraphHorizontalSumI32x4AvxVnni(__m128i v)
{
	int32		s[4];

	_mm_storeu_si128((__m128i *) s, v);
	return (int64) s[0] + s[1] + s[2] + s[3];
}

#define TQ_CC_NAME PgturbohybridGraphQuerySplitRawAvxVnni
#define TQ_CC_EXPAND PgturbohybridGraphExpandPacked4Avx2
#define TQ_CC_STRIDE 8
#define TQ_CC_TAILADD 1
#define TQ_CC_TAILDIV 2
#include "pgturbohybrid_quant_score_x86_querysplit_avxvnni.inc"

#define TQ_CC_NAME PgturbohybridGraphQuerySplit2RawAvxVnni
#define TQ_CC_EXPAND PgturbohybridGraphExpandPacked2Avx2
#define TQ_CC_STRIDE 4
#define TQ_CC_TAILADD 3
#define TQ_CC_TAILDIV 4
#include "pgturbohybrid_quant_score_x86_querysplit_avxvnni.inc"

#define TQ_CC_NAME PgturbohybridGraphCodeCodeRawAvxVnni
#define TQ_CC_EXPAND PgturbohybridGraphExpandPacked4Avx2
#define TQ_CC_STRIDE 8
#define TQ_CC_TAILADD 1
#define TQ_CC_TAILDIV 2
#include "pgturbohybrid_quant_score_x86_codecode_avxvnni.inc"

#define TQ_CC_NAME PgturbohybridGraphCodeCode2RawAvxVnni
#define TQ_CC_EXPAND PgturbohybridGraphExpandPacked2Avx2
#define TQ_CC_STRIDE 4
#define TQ_CC_TAILADD 3
#define TQ_CC_TAILDIV 4
#include "pgturbohybrid_quant_score_x86_codecode_avxvnni.inc"
#endif

bool PGTURBOHYBRID_GRAPH_AVX2_TARGET
PgturbohybridGraphExactVectorDistanceAvx2(PgturbohybridGraphScanOpaque so, Vector *queryVector,
							   Vector *valueVector, double *result)
{
	TqScoreMode mode = (TqScoreMode) so->tq.scoreMode;
	int			i = 0;

	if (mode == PGTURBOHYBRID_SCORE_L2)
	{
		__m256		acc0 = _mm256_setzero_ps();
		__m256		acc1 = _mm256_setzero_ps();
		__m256		acc2 = _mm256_setzero_ps();
		__m256		acc3 = _mm256_setzero_ps();

		for (; i + 32 <= queryVector->dim; i += 32)
		{
			__m256	q0 = _mm256_loadu_ps(&queryVector->x[i]);
			__m256	q1 = _mm256_loadu_ps(&queryVector->x[i + 8]);
			__m256	q2 = _mm256_loadu_ps(&queryVector->x[i + 16]);
			__m256	q3 = _mm256_loadu_ps(&queryVector->x[i + 24]);
			__m256	v0 = _mm256_loadu_ps(&valueVector->x[i]);
			__m256	v1 = _mm256_loadu_ps(&valueVector->x[i + 8]);
			__m256	v2 = _mm256_loadu_ps(&valueVector->x[i + 16]);
			__m256	v3 = _mm256_loadu_ps(&valueVector->x[i + 24]);
			__m256	d0 = _mm256_sub_ps(q0, v0);
			__m256	d1 = _mm256_sub_ps(q1, v1);
			__m256	d2 = _mm256_sub_ps(q2, v2);
			__m256	d3 = _mm256_sub_ps(q3, v3);

			acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
			acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(d1, d1));
			acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(d2, d2));
			acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(d3, d3));
		}

		acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
							 _mm256_add_ps(acc2, acc3));
		*result = PgturbohybridGraphHorizontalSumF32Avx2(acc0);
		for (; i < queryVector->dim; i++)
		{
			double		diff = (double) queryVector->x[i] - valueVector->x[i];

			*result += diff * diff;
		}
		return true;
	}

	if (mode == PGTURBOHYBRID_SCORE_IP || mode == PGTURBOHYBRID_SCORE_COSINE)
	{
		__m256		dotAcc = _mm256_setzero_ps();
		__m256		normAcc = _mm256_setzero_ps();
		double		dot;
		double		valueNorm = 0;

		for (; i + 8 <= queryVector->dim; i += 8)
		{
			__m256	qv = _mm256_loadu_ps(&queryVector->x[i]);
			__m256	vv = _mm256_loadu_ps(&valueVector->x[i]);

			dotAcc = _mm256_add_ps(dotAcc, _mm256_mul_ps(qv, vv));
			if (mode == PGTURBOHYBRID_SCORE_COSINE)
				normAcc = _mm256_add_ps(normAcc, _mm256_mul_ps(vv, vv));
		}

		dot = PgturbohybridGraphHorizontalSumF32Avx2(dotAcc);
		if (mode == PGTURBOHYBRID_SCORE_COSINE)
			valueNorm = PgturbohybridGraphHorizontalSumF32Avx2(normAcc);

		for (; i < queryVector->dim; i++)
		{
			double		qv = queryVector->x[i];
			double		vv = valueVector->x[i];

			dot += qv * vv;
			if (mode == PGTURBOHYBRID_SCORE_COSINE)
				valueNorm += vv * vv;
		}

		if (mode == PGTURBOHYBRID_SCORE_IP)
			*result = -dot;
		else if (so->tq.queryNorm == 0 || valueNorm == 0)
			*result = 1;
		else
			*result = 1 - (dot / sqrt(so->tq.queryNorm * valueNorm));

		return true;
	}

	return false;
}

bool PGTURBOHYBRID_GRAPH_AVX2_TARGET
PgturbohybridGraphBuildExactDistanceAvx2(PgturbohybridQuantBuildState *state, Vector *av,
							  Vector *bv, double *result)
{
	TqScoreMode mode = (TqScoreMode) state->scoreMode;
	int			i = 0;

	if (mode == PGTURBOHYBRID_SCORE_L2)
	{
		__m256		acc0 = _mm256_setzero_ps();
		__m256		acc1 = _mm256_setzero_ps();
		__m256		acc2 = _mm256_setzero_ps();
		__m256		acc3 = _mm256_setzero_ps();

		for (; i + 32 <= av->dim; i += 32)
		{
			__m256	a0 = _mm256_loadu_ps(&av->x[i]);
			__m256	a1 = _mm256_loadu_ps(&av->x[i + 8]);
			__m256	a2 = _mm256_loadu_ps(&av->x[i + 16]);
			__m256	a3 = _mm256_loadu_ps(&av->x[i + 24]);
			__m256	b0 = _mm256_loadu_ps(&bv->x[i]);
			__m256	b1 = _mm256_loadu_ps(&bv->x[i + 8]);
			__m256	b2 = _mm256_loadu_ps(&bv->x[i + 16]);
			__m256	b3 = _mm256_loadu_ps(&bv->x[i + 24]);
			__m256	d0 = _mm256_sub_ps(a0, b0);
			__m256	d1 = _mm256_sub_ps(a1, b1);
			__m256	d2 = _mm256_sub_ps(a2, b2);
			__m256	d3 = _mm256_sub_ps(a3, b3);

			acc0 = _mm256_add_ps(acc0, _mm256_mul_ps(d0, d0));
			acc1 = _mm256_add_ps(acc1, _mm256_mul_ps(d1, d1));
			acc2 = _mm256_add_ps(acc2, _mm256_mul_ps(d2, d2));
			acc3 = _mm256_add_ps(acc3, _mm256_mul_ps(d3, d3));
		}

		acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
							 _mm256_add_ps(acc2, acc3));
		*result = PgturbohybridGraphHorizontalSumF32Avx2(acc0);
		for (; i < av->dim; i++)
		{
			double		diff = (double) av->x[i] - bv->x[i];

			*result += diff * diff;
		}
		return true;
	}

	if (mode == PGTURBOHYBRID_SCORE_IP || mode == PGTURBOHYBRID_SCORE_COSINE)
	{
		__m256		dotAcc = _mm256_setzero_ps();
		__m256		aNormAcc = _mm256_setzero_ps();
		__m256		bNormAcc = _mm256_setzero_ps();
		double		dot;
		double		aNorm = 0;
		double		bNorm = 0;

		for (; i + 8 <= av->dim; i += 8)
		{
			__m256	avv = _mm256_loadu_ps(&av->x[i]);
			__m256	bvv = _mm256_loadu_ps(&bv->x[i]);

			dotAcc = _mm256_add_ps(dotAcc, _mm256_mul_ps(avv, bvv));
			if (mode == PGTURBOHYBRID_SCORE_COSINE)
			{
				aNormAcc = _mm256_add_ps(aNormAcc, _mm256_mul_ps(avv, avv));
				bNormAcc = _mm256_add_ps(bNormAcc, _mm256_mul_ps(bvv, bvv));
			}
		}

		dot = PgturbohybridGraphHorizontalSumF32Avx2(dotAcc);
		if (mode == PGTURBOHYBRID_SCORE_COSINE)
		{
			aNorm = PgturbohybridGraphHorizontalSumF32Avx2(aNormAcc);
			bNorm = PgturbohybridGraphHorizontalSumF32Avx2(bNormAcc);
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
			*result = -dot;
		else if (aNorm == 0 || bNorm == 0)
			*result = 1;
		else
			*result = 1 - (dot / sqrt(aNorm * bNorm));

		return true;
	}

	return false;
}

#endif
