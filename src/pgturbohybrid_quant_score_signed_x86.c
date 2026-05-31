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

#include "pgturbohybrid_quant_score.h"
#include "pgturbohybrid_quant_score_internal.h"

#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2 && !PGTURBOHYBRID_GRAPH_COMPILE_ARM_DOT
static const int8 PgturbohybridGraphCodebookI8[PGTURBOHYBRID_LUT_WIDTH] = {
	-127, -96, -75, -58, -44, -31, -18, -6,
	6, 18, 31, 44, 58, 75, 96, 127
};
pg_attribute_unused()
static const int8 PgturbohybridGraphCodebook2I8[PGTURBOHYBRID_LUT_WIDTH] = {
	-127, -38, 38, 127, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0
};
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

int64 PGTURBOHYBRID_GRAPH_AVX2_TARGET
PgturbohybridGraphQuerySplitRawAvx2(const PgturbohybridGraphTqQuery *tq, const uint8 *code)
{
	__m256i	accLow = _mm256_setzero_si256();
	__m256i	accHigh = _mm256_setzero_si256();

	for (int chunk = 0; chunk < tq->querySplitChunks; chunk++)
	{
		__m128i	c = PgturbohybridGraphExpandPacked4Avx2(code + chunk * 8);
		__m128i	low = _mm_loadu_si128((const __m128i *) (tq->signedSplit.low + chunk * 16));
		__m128i	high = _mm_loadu_si128((const __m128i *) (tq->signedSplit.high + chunk * 16));

		accLow = _mm256_add_epi32(accLow, PgturbohybridGraphDotI8x16Avx2(low, c));
		accHigh = _mm256_add_epi32(accHigh, PgturbohybridGraphDotI8x16Avx2(high, c));
	}

	if (tq->querySplitTailDims != 0)
	{
		uint8		scratch[8] = {0};
		int			tailBytes = (tq->querySplitTailDims + 1) / 2;
		__m128i	c;
		__m128i	low;
		__m128i	high;

		memcpy(scratch, code + tq->querySplitChunks * 8, tailBytes);
		c = PgturbohybridGraphExpandPacked4Avx2(scratch);
		low = _mm_loadu_si128((const __m128i *) tq->signedSplit.tailLow);
		high = _mm_loadu_si128((const __m128i *) tq->signedSplit.tailHigh);
		accLow = _mm256_add_epi32(accLow, PgturbohybridGraphDotI8x16Avx2(low, c));
		accHigh = _mm256_add_epi32(accHigh, PgturbohybridGraphDotI8x16Avx2(high, c));
	}

	return PgturbohybridGraphHorizontalSumI32Avx2(accLow) +
		(int64) PGTURBOHYBRID_QUERY_SPLIT_HIGH_COEF * PgturbohybridGraphHorizontalSumI32Avx2(accHigh);
}


int64 PGTURBOHYBRID_GRAPH_AVX2_TARGET
PgturbohybridGraphQuerySplit2RawAvx2(const PgturbohybridGraphTqQuery *tq, const uint8 *code)
{
	__m256i	accLow = _mm256_setzero_si256();
	__m256i	accHigh = _mm256_setzero_si256();

	for (int chunk = 0; chunk < tq->querySplitChunks; chunk++)
	{
		__m128i	c = PgturbohybridGraphExpandPacked2Avx2(code + chunk * 4);
		__m128i	low = _mm_loadu_si128((const __m128i *) (tq->signedSplit.low + chunk * 16));
		__m128i	high = _mm_loadu_si128((const __m128i *) (tq->signedSplit.high + chunk * 16));

		accLow = _mm256_add_epi32(accLow, PgturbohybridGraphDotI8x16Avx2(low, c));
		accHigh = _mm256_add_epi32(accHigh, PgturbohybridGraphDotI8x16Avx2(high, c));
	}

	if (tq->querySplitTailDims != 0)
	{
		uint8		scratch[4] = {0};
		int			tailBytes = (tq->querySplitTailDims + 3) / 4;
		__m128i	c;
		__m128i	low;
		__m128i	high;

		memcpy(scratch, code + tq->querySplitChunks * 4, tailBytes);
		c = PgturbohybridGraphExpandPacked2Avx2(scratch);
		low = _mm_loadu_si128((const __m128i *) tq->signedSplit.tailLow);
		high = _mm_loadu_si128((const __m128i *) tq->signedSplit.tailHigh);
		accLow = _mm256_add_epi32(accLow, PgturbohybridGraphDotI8x16Avx2(low, c));
		accHigh = _mm256_add_epi32(accHigh, PgturbohybridGraphDotI8x16Avx2(high, c));
	}

	return PgturbohybridGraphHorizontalSumI32Avx2(accLow) +
		(int64) PGTURBOHYBRID_QUERY_SPLIT_HIGH_COEF * PgturbohybridGraphHorizontalSumI32Avx2(accHigh);
}

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

int64 PGTURBOHYBRID_GRAPH_AVX2_TARGET
PgturbohybridGraphCodeCodeWeightedRawAvx2(const uint8 *a, const uint8 *b,
								const int16 *weights, int dim)
{
	__m256i		acc = _mm256_setzero_si256();
	int			chunks = dim / 16;
	int			tailDims = dim - chunks * 16;

	for (int chunk = 0; chunk < chunks; chunk++)
	{
		__m128i		ca = PgturbohybridGraphExpandPacked4Avx2(a + chunk * 8);
		__m128i		cb = PgturbohybridGraphExpandPacked4Avx2(b + chunk * 8);

		acc = _mm256_add_epi64(acc,
								PgturbohybridGraphWeightedDotI8x16Avx2(ca, cb, weights + chunk * 16));
	}

	if (tailDims != 0)
	{
		uint8		scratchA[8] = {0};
		uint8		scratchB[8] = {0};
		int16		scratchW[16] = {0};
		int			tailBytes = (tailDims + 1) / 2;
		__m128i		ca;
		__m128i		cb;

		memcpy(scratchA, a + chunks * 8, tailBytes);
		memcpy(scratchB, b + chunks * 8, tailBytes);
		memcpy(scratchW, weights + chunks * 16, sizeof(int16) * tailDims);
		ca = PgturbohybridGraphExpandPacked4Avx2(scratchA);
		cb = PgturbohybridGraphExpandPacked4Avx2(scratchB);
		acc = _mm256_add_epi64(acc, PgturbohybridGraphWeightedDotI8x16Avx2(ca, cb, scratchW));
	}

	return PgturbohybridGraphHorizontalSumI64Avx2(acc);
}

int64 PGTURBOHYBRID_GRAPH_AVX2_TARGET
PgturbohybridGraphCodeCode2WeightedRawAvx2(const uint8 *a, const uint8 *b,
								 const int16 *weights, int dim)
{
	__m256i		acc = _mm256_setzero_si256();
	int			chunks = dim / 16;
	int			tailDims = dim - chunks * 16;

	for (int chunk = 0; chunk < chunks; chunk++)
	{
		__m128i		ca = PgturbohybridGraphExpandPacked2Avx2(a + chunk * 4);
		__m128i		cb = PgturbohybridGraphExpandPacked2Avx2(b + chunk * 4);

		acc = _mm256_add_epi64(acc,
								PgturbohybridGraphWeightedDotI8x16Avx2(ca, cb, weights + chunk * 16));
	}

	if (tailDims != 0)
	{
		uint8		scratchA[4] = {0};
		uint8		scratchB[4] = {0};
		int16		scratchW[16] = {0};
		int			tailBytes = (tailDims + 3) / 4;
		__m128i		ca;
		__m128i		cb;

		memcpy(scratchA, a + chunks * 4, tailBytes);
		memcpy(scratchB, b + chunks * 4, tailBytes);
		memcpy(scratchW, weights + chunks * 16, sizeof(int16) * tailDims);
		ca = PgturbohybridGraphExpandPacked2Avx2(scratchA);
		cb = PgturbohybridGraphExpandPacked2Avx2(scratchB);
		acc = _mm256_add_epi64(acc, PgturbohybridGraphWeightedDotI8x16Avx2(ca, cb, scratchW));
	}

	return PgturbohybridGraphHorizontalSumI64Avx2(acc);
}

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

int64 PGTURBOHYBRID_GRAPH_AVX512_WEIGHTED_TARGET
PgturbohybridGraphCodeCodeWeightedRawAvx512(const uint8 *a, const uint8 *b,
								 const int16 *weights, int dim)
{
	int64		acc = 0;
	int			chunks = dim / 32;
	int			tailDims = dim - chunks * 32;

	for (int chunk = 0; chunk < chunks; chunk++)
	{
		__m256i		ca = PgturbohybridGraphExpandPacked4x32Avx512(a + chunk * 16);
		__m256i		cb = PgturbohybridGraphExpandPacked4x32Avx512(b + chunk * 16);

		acc += PgturbohybridGraphWeightedDotI8x32Avx512(ca, cb, weights + chunk * 32);
	}

	if (tailDims != 0)
	{
		uint8		scratchA[16] = {0};
		uint8		scratchB[16] = {0};
		int16		scratchW[32] = {0};
		int			tailBytes = (tailDims + 1) / 2;
		__m256i		ca;
		__m256i		cb;

		memcpy(scratchA, a + chunks * 16, tailBytes);
		memcpy(scratchB, b + chunks * 16, tailBytes);
		memcpy(scratchW, weights + chunks * 32, sizeof(int16) * tailDims);
		ca = PgturbohybridGraphExpandPacked4x32Avx512(scratchA);
		cb = PgturbohybridGraphExpandPacked4x32Avx512(scratchB);
		acc += PgturbohybridGraphWeightedDotI8x32Avx512(ca, cb, scratchW);
	}

	return acc;
}

int64 PGTURBOHYBRID_GRAPH_AVX512_WEIGHTED_TARGET
PgturbohybridGraphCodeCode2WeightedRawAvx512(const uint8 *a, const uint8 *b,
								  const int16 *weights, int dim)
{
	int64		acc = 0;
	int			chunks = dim / 32;
	int			tailDims = dim - chunks * 32;

	for (int chunk = 0; chunk < chunks; chunk++)
	{
		__m256i		ca = PgturbohybridGraphExpandPacked2x32Avx512(a + chunk * 8);
		__m256i		cb = PgturbohybridGraphExpandPacked2x32Avx512(b + chunk * 8);

		acc += PgturbohybridGraphWeightedDotI8x32Avx512(ca, cb, weights + chunk * 32);
	}

	if (tailDims != 0)
	{
		uint8		scratchA[8] = {0};
		uint8		scratchB[8] = {0};
		int16		scratchW[32] = {0};
		int			tailBytes = (tailDims + 3) / 4;
		__m256i		ca;
		__m256i		cb;

		memcpy(scratchA, a + chunks * 8, tailBytes);
		memcpy(scratchB, b + chunks * 8, tailBytes);
		memcpy(scratchW, weights + chunks * 32, sizeof(int16) * tailDims);
		ca = PgturbohybridGraphExpandPacked2x32Avx512(scratchA);
		cb = PgturbohybridGraphExpandPacked2x32Avx512(scratchB);
		acc += PgturbohybridGraphWeightedDotI8x32Avx512(ca, cb, scratchW);
	}

	return acc;
}
#endif

int64 PGTURBOHYBRID_GRAPH_AVX2_TARGET
PgturbohybridGraphCodeCodeRawAvx2(const uint8 *a, const uint8 *b, int dim,
					   int *sampleDims)
{
	__m256i	acc = _mm256_setzero_si256();
	int		chunks = dim / 16;
	int		tailDims = dim - chunks * 16;
	int		scoredChunks = chunks;

	if (chunks > PGTURBOHYBRID_GRAPH_CODE_CODE_PRUNE_CHUNKS)
		scoredChunks = PGTURBOHYBRID_GRAPH_CODE_CODE_PRUNE_CHUNKS;
	*sampleDims = scoredChunks * 16;

	for (int scored = 0; scored < scoredChunks; scored++)
	{
		int			chunk = scoredChunks == chunks ? scored :
			(int) (((int64) scored * chunks) / scoredChunks);
		__m128i	ca = PgturbohybridGraphExpandPacked4Avx2(a + chunk * 8);
		__m128i	cb = PgturbohybridGraphExpandPacked4Avx2(b + chunk * 8);

		acc = _mm256_add_epi32(acc, PgturbohybridGraphDotI8x16Avx2(ca, cb));
	}

	if (scoredChunks == chunks && tailDims != 0)
	{
		uint8		scratchA[8] = {0};
		uint8		scratchB[8] = {0};
		int			tailBytes = (tailDims + 1) / 2;
		__m128i	ca;
		__m128i	cb;

		memcpy(scratchA, a + chunks * 8, tailBytes);
		memcpy(scratchB, b + chunks * 8, tailBytes);
		ca = PgturbohybridGraphExpandPacked4Avx2(scratchA);
		cb = PgturbohybridGraphExpandPacked4Avx2(scratchB);
		acc = _mm256_add_epi32(acc, PgturbohybridGraphDotI8x16Avx2(ca, cb));
		*sampleDims += tailDims;
	}

	return PgturbohybridGraphHorizontalSumI32Avx2(acc);
}

int64 PGTURBOHYBRID_GRAPH_AVX2_TARGET
PgturbohybridGraphCodeCode2RawAvx2(const uint8 *a, const uint8 *b, int dim,
						int *sampleDims)
{
	__m256i	acc = _mm256_setzero_si256();
	int		chunks = dim / 16;
	int		tailDims = dim - chunks * 16;
	int		scoredChunks = chunks;

	if (chunks > PGTURBOHYBRID_GRAPH_CODE_CODE_PRUNE_CHUNKS)
		scoredChunks = PGTURBOHYBRID_GRAPH_CODE_CODE_PRUNE_CHUNKS;
	*sampleDims = scoredChunks * 16;

	for (int scored = 0; scored < scoredChunks; scored++)
	{
		int			chunk = scoredChunks == chunks ? scored :
			(int) (((int64) scored * chunks) / scoredChunks);
		__m128i	ca = PgturbohybridGraphExpandPacked2Avx2(a + chunk * 4);
		__m128i	cb = PgturbohybridGraphExpandPacked2Avx2(b + chunk * 4);

		acc = _mm256_add_epi32(acc, PgturbohybridGraphDotI8x16Avx2(ca, cb));
	}

	if (scoredChunks == chunks && tailDims != 0)
	{
		uint8		scratchA[4] = {0};
		uint8		scratchB[4] = {0};
		int			tailBytes = (tailDims + 3) / 4;
		__m128i	ca;
		__m128i	cb;

		memcpy(scratchA, a + chunks * 4, tailBytes);
		memcpy(scratchB, b + chunks * 4, tailBytes);
		ca = PgturbohybridGraphExpandPacked2Avx2(scratchA);
		cb = PgturbohybridGraphExpandPacked2Avx2(scratchB);
		acc = _mm256_add_epi32(acc, PgturbohybridGraphDotI8x16Avx2(ca, cb));
		*sampleDims += tailDims;
	}

	return PgturbohybridGraphHorizontalSumI32Avx2(acc);
}

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

int64 PGTURBOHYBRID_GRAPH_AVX512VNNI_TARGET
PgturbohybridGraphQuerySplitRawAvx512Vnni(const PgturbohybridGraphTqQuery *tq, const uint8 *code)
{
	const __m512i ones512 = _mm512_set1_epi8(1);
	const __m256i ones256 = _mm256_set1_epi8(1);
	const __m128i ones128 = _mm_set1_epi8(1);
	__m512i		accLow512 = _mm512_setzero_si512();
	__m512i		accHigh512 = _mm512_setzero_si512();
	__m512i		accCSum512 = _mm512_setzero_si512();
	__m256i		accLow256 = _mm256_setzero_si256();
	__m256i		accHigh256 = _mm256_setzero_si256();
	__m256i		accCSum256 = _mm256_setzero_si256();
	__m128i		accLow128 = _mm_setzero_si128();
	__m128i		accHigh128 = _mm_setzero_si128();
	__m128i		accCSum128 = _mm_setzero_si128();
	int			chunk = 0;
	int64		dotLow;
	int64		dotHigh;
	int64		cSum;

	/* Quad-stepped main loop: 64 dims per iteration on ZMM. */
	for (; chunk + 4 <= tq->querySplitChunks; chunk += 4)
	{
		__m512i		c = PgturbohybridGraphExpandPacked4Avx512(code + chunk * 8);
		__m512i		low = _mm512_loadu_si512((const __m512i *) (tq->signedSplit.lowU8 + chunk * 16));
		__m512i		high = _mm512_loadu_si512((const __m512i *) (tq->signedSplit.highU8 + chunk * 16));

		accLow512 = _mm512_dpbusd_epi32(accLow512, low, c);
		accHigh512 = _mm512_dpbusd_epi32(accHigh512, high, c);
		accCSum512 = _mm512_dpbusd_epi32(accCSum512, ones512, c);
	}

	/* Pair-step trailing 32 dims on YMM. */
	if (chunk + 2 <= tq->querySplitChunks)
	{
		__m128i		c0 = PgturbohybridGraphExpandPacked4Avx2(code + chunk * 8);
		__m128i		c1 = PgturbohybridGraphExpandPacked4Avx2(code + (chunk + 1) * 8);
		__m256i		c = _mm256_inserti128_si256(_mm256_castsi128_si256(c0), c1, 1);
		__m256i		low = _mm256_loadu_si256((const __m256i *) (tq->signedSplit.lowU8 + chunk * 16));
		__m256i		high = _mm256_loadu_si256((const __m256i *) (tq->signedSplit.highU8 + chunk * 16));

		accLow256 = _mm256_dpbusd_epi32(accLow256, low, c);
		accHigh256 = _mm256_dpbusd_epi32(accHigh256, high, c);
		accCSum256 = _mm256_dpbusd_epi32(accCSum256, ones256, c);
		chunk += 2;
	}

	/* Trailing 16-dim chunk on XMM. */
	if (chunk < tq->querySplitChunks)
	{
		__m128i		c = PgturbohybridGraphExpandPacked4Avx2(code + chunk * 8);
		__m128i		low = _mm_loadu_si128((const __m128i *) (tq->signedSplit.lowU8 + chunk * 16));
		__m128i		high = _mm_loadu_si128((const __m128i *) (tq->signedSplit.highU8 + chunk * 16));

		accLow128 = _mm_dpbusd_epi32(accLow128, low, c);
		accHigh128 = _mm_dpbusd_epi32(accHigh128, high, c);
		accCSum128 = _mm_dpbusd_epi32(accCSum128, ones128, c);
	}

	/* Sub-chunk tail dims. */
	if (tq->querySplitTailDims != 0)
	{
		uint8		scratch[8] = {0};
		int			tailBytes = (tq->querySplitTailDims + 1) / 2;
		__m128i		c;
		__m128i		low;
		__m128i		high;

		memcpy(scratch, code + tq->querySplitChunks * 8, tailBytes);
		c = PgturbohybridGraphExpandPacked4Avx2(scratch);
		low = _mm_loadu_si128((const __m128i *) tq->signedSplit.tailLowU8);
		high = _mm_loadu_si128((const __m128i *) tq->signedSplit.tailHighU8);
		accLow128 = _mm_dpbusd_epi32(accLow128, low, c);
		accHigh128 = _mm_dpbusd_epi32(accHigh128, high, c);
		accCSum128 = _mm_dpbusd_epi32(accCSum128, ones128, c);
	}

	dotLow = (int64) _mm512_reduce_add_epi32(accLow512) +
		PgturbohybridGraphHorizontalSumI32Avx2(accLow256) +
		PgturbohybridGraphHorizontalSumI32x4Avx512Vnni(accLow128);
	dotHigh = (int64) _mm512_reduce_add_epi32(accHigh512) +
		PgturbohybridGraphHorizontalSumI32Avx2(accHigh256) +
		PgturbohybridGraphHorizontalSumI32x4Avx512Vnni(accHigh128);
	cSum = (int64) _mm512_reduce_add_epi32(accCSum512) +
		PgturbohybridGraphHorizontalSumI32Avx2(accCSum256) +
		PgturbohybridGraphHorizontalSumI32x4Avx512Vnni(accCSum128);

	return (dotLow - 128 * cSum) +
		(int64) PGTURBOHYBRID_QUERY_SPLIT_HIGH_COEF * (dotHigh - 128 * cSum);
}

int64 PGTURBOHYBRID_GRAPH_AVX512VNNI_TARGET
PgturbohybridGraphQuerySplit2RawAvx512Vnni(const PgturbohybridGraphTqQuery *tq, const uint8 *code)
{
	const __m512i ones512 = _mm512_set1_epi8(1);
	const __m256i ones256 = _mm256_set1_epi8(1);
	const __m128i ones128 = _mm_set1_epi8(1);
	__m512i		accLow512 = _mm512_setzero_si512();
	__m512i		accHigh512 = _mm512_setzero_si512();
	__m512i		accCSum512 = _mm512_setzero_si512();
	__m256i		accLow256 = _mm256_setzero_si256();
	__m256i		accHigh256 = _mm256_setzero_si256();
	__m256i		accCSum256 = _mm256_setzero_si256();
	__m128i		accLow128 = _mm_setzero_si128();
	__m128i		accHigh128 = _mm_setzero_si128();
	__m128i		accCSum128 = _mm_setzero_si128();
	int			chunk = 0;
	int64		dotLow;
	int64		dotHigh;
	int64		cSum;

	for (; chunk + 4 <= tq->querySplitChunks; chunk += 4)
	{
		__m512i		c = PgturbohybridGraphExpandPacked2Avx512(code + chunk * 4);
		__m512i		low = _mm512_loadu_si512((const __m512i *) (tq->signedSplit.lowU8 + chunk * 16));
		__m512i		high = _mm512_loadu_si512((const __m512i *) (tq->signedSplit.highU8 + chunk * 16));

		accLow512 = _mm512_dpbusd_epi32(accLow512, low, c);
		accHigh512 = _mm512_dpbusd_epi32(accHigh512, high, c);
		accCSum512 = _mm512_dpbusd_epi32(accCSum512, ones512, c);
	}

	if (chunk + 2 <= tq->querySplitChunks)
	{
		__m128i		c0 = PgturbohybridGraphExpandPacked2Avx2(code + chunk * 4);
		__m128i		c1 = PgturbohybridGraphExpandPacked2Avx2(code + (chunk + 1) * 4);
		__m256i		c = _mm256_inserti128_si256(_mm256_castsi128_si256(c0), c1, 1);
		__m256i		low = _mm256_loadu_si256((const __m256i *) (tq->signedSplit.lowU8 + chunk * 16));
		__m256i		high = _mm256_loadu_si256((const __m256i *) (tq->signedSplit.highU8 + chunk * 16));

		accLow256 = _mm256_dpbusd_epi32(accLow256, low, c);
		accHigh256 = _mm256_dpbusd_epi32(accHigh256, high, c);
		accCSum256 = _mm256_dpbusd_epi32(accCSum256, ones256, c);
		chunk += 2;
	}

	if (chunk < tq->querySplitChunks)
	{
		__m128i		c = PgturbohybridGraphExpandPacked2Avx2(code + chunk * 4);
		__m128i		low = _mm_loadu_si128((const __m128i *) (tq->signedSplit.lowU8 + chunk * 16));
		__m128i		high = _mm_loadu_si128((const __m128i *) (tq->signedSplit.highU8 + chunk * 16));

		accLow128 = _mm_dpbusd_epi32(accLow128, low, c);
		accHigh128 = _mm_dpbusd_epi32(accHigh128, high, c);
		accCSum128 = _mm_dpbusd_epi32(accCSum128, ones128, c);
	}

	if (tq->querySplitTailDims != 0)
	{
		uint8		scratch[4] = {0};
		int			tailBytes = (tq->querySplitTailDims + 3) / 4;
		__m128i		c;
		__m128i		low;
		__m128i		high;

		memcpy(scratch, code + tq->querySplitChunks * 4, tailBytes);
		c = PgturbohybridGraphExpandPacked2Avx2(scratch);
		low = _mm_loadu_si128((const __m128i *) tq->signedSplit.tailLowU8);
		high = _mm_loadu_si128((const __m128i *) tq->signedSplit.tailHighU8);
		accLow128 = _mm_dpbusd_epi32(accLow128, low, c);
		accHigh128 = _mm_dpbusd_epi32(accHigh128, high, c);
		accCSum128 = _mm_dpbusd_epi32(accCSum128, ones128, c);
	}

	dotLow = (int64) _mm512_reduce_add_epi32(accLow512) +
		PgturbohybridGraphHorizontalSumI32Avx2(accLow256) +
		PgturbohybridGraphHorizontalSumI32x4Avx512Vnni(accLow128);
	dotHigh = (int64) _mm512_reduce_add_epi32(accHigh512) +
		PgturbohybridGraphHorizontalSumI32Avx2(accHigh256) +
		PgturbohybridGraphHorizontalSumI32x4Avx512Vnni(accHigh128);
	cSum = (int64) _mm512_reduce_add_epi32(accCSum512) +
		PgturbohybridGraphHorizontalSumI32Avx2(accCSum256) +
		PgturbohybridGraphHorizontalSumI32x4Avx512Vnni(accCSum128);

	return (dotLow - 128 * cSum) +
		(int64) PGTURBOHYBRID_QUERY_SPLIT_HIGH_COEF * (dotHigh - 128 * cSum);
}

int64 PGTURBOHYBRID_GRAPH_AVX512VNNI_TARGET
PgturbohybridGraphCodeCodeRawAvx512Vnni(const uint8 *a, const uint8 *b, int dim,
							 int *sampleDims)
{
	const __m512i signFlip512 = _mm512_set1_epi8((char) 0x80);
	const __m512i ones512 = _mm512_set1_epi8(1);
	const __m256i signFlip256 = _mm256_set1_epi8((char) 0x80);
	const __m256i ones256 = _mm256_set1_epi8(1);
	const __m128i signFlip128 = _mm_set1_epi8((char) 0x80);
	const __m128i ones128 = _mm_set1_epi8(1);
	__m512i		acc512 = _mm512_setzero_si512();
	__m512i		accCSum512 = _mm512_setzero_si512();
	__m256i		acc256 = _mm256_setzero_si256();
	__m256i		accCSum256 = _mm256_setzero_si256();
	__m128i		acc128 = _mm_setzero_si128();
	__m128i		accCSum128 = _mm_setzero_si128();
	int			chunks = dim / 16;
	int			tailDims = dim - chunks * 16;
	int			scoredChunks = chunks;
	int			scored = 0;
	int64		dot;
	int64		cSum;

	if (chunks > PGTURBOHYBRID_GRAPH_CODE_CODE_PRUNE_CHUNKS)
		scoredChunks = PGTURBOHYBRID_GRAPH_CODE_CODE_PRUNE_CHUNKS;
	*sampleDims = scoredChunks * 16;

	if (scoredChunks == chunks)
	{
		/*
		 * Contiguous full traversal — drive ZMM/YMM strides directly across
		 * the input.
		 */
		for (; scored + 4 <= scoredChunks; scored += 4)
		{
			__m512i		ca = PgturbohybridGraphExpandPacked4Avx512(a + scored * 8);
			__m512i		cb = PgturbohybridGraphExpandPacked4Avx512(b + scored * 8);

			acc512 = _mm512_dpbusd_epi32(acc512,
										 _mm512_xor_si512(ca, signFlip512), cb);
			accCSum512 = _mm512_dpbusd_epi32(accCSum512, ones512, cb);
		}
		if (scored + 2 <= scoredChunks)
		{
			__m128i		ca0 = PgturbohybridGraphExpandPacked4Avx2(a + scored * 8);
			__m128i		cb0 = PgturbohybridGraphExpandPacked4Avx2(b + scored * 8);
			__m128i		ca1 = PgturbohybridGraphExpandPacked4Avx2(a + (scored + 1) * 8);
			__m128i		cb1 = PgturbohybridGraphExpandPacked4Avx2(b + (scored + 1) * 8);
			__m256i		ca = _mm256_inserti128_si256(_mm256_castsi128_si256(ca0), ca1, 1);
			__m256i		cb = _mm256_inserti128_si256(_mm256_castsi128_si256(cb0), cb1, 1);

			acc256 = _mm256_dpbusd_epi32(acc256,
										 _mm256_xor_si256(ca, signFlip256), cb);
			accCSum256 = _mm256_dpbusd_epi32(accCSum256, ones256, cb);
			scored += 2;
		}
	}
	else
	{
		/* Stride-sampled scoring; can still pair adjacent samples. */
		for (; scored + 2 <= scoredChunks; scored += 2)
		{
			int			chunkA = (int) (((int64) scored * chunks) / scoredChunks);
			int			chunkB = (int) (((int64) (scored + 1) * chunks) / scoredChunks);
			__m128i		ca0 = PgturbohybridGraphExpandPacked4Avx2(a + chunkA * 8);
			__m128i		cb0 = PgturbohybridGraphExpandPacked4Avx2(b + chunkA * 8);
			__m128i		ca1 = PgturbohybridGraphExpandPacked4Avx2(a + chunkB * 8);
			__m128i		cb1 = PgturbohybridGraphExpandPacked4Avx2(b + chunkB * 8);
			__m256i		ca = _mm256_inserti128_si256(_mm256_castsi128_si256(ca0), ca1, 1);
			__m256i		cb = _mm256_inserti128_si256(_mm256_castsi128_si256(cb0), cb1, 1);

			acc256 = _mm256_dpbusd_epi32(acc256,
										 _mm256_xor_si256(ca, signFlip256), cb);
			accCSum256 = _mm256_dpbusd_epi32(accCSum256, ones256, cb);
		}
	}

	if (scored < scoredChunks)
	{
		int			chunk = scoredChunks == chunks ? scored :
			(int) (((int64) scored * chunks) / scoredChunks);
		__m128i		ca = PgturbohybridGraphExpandPacked4Avx2(a + chunk * 8);
		__m128i		cb = PgturbohybridGraphExpandPacked4Avx2(b + chunk * 8);

		acc128 = _mm_dpbusd_epi32(acc128,
								  _mm_xor_si128(ca, signFlip128), cb);
		accCSum128 = _mm_dpbusd_epi32(accCSum128, ones128, cb);
	}

	if (scoredChunks == chunks && tailDims != 0)
	{
		uint8		scratchA[8] = {0};
		uint8		scratchB[8] = {0};
		int			tailBytes = (tailDims + 1) / 2;
		__m128i		ca;
		__m128i		cb;

		memcpy(scratchA, a + chunks * 8, tailBytes);
		memcpy(scratchB, b + chunks * 8, tailBytes);
		ca = PgturbohybridGraphExpandPacked4Avx2(scratchA);
		cb = PgturbohybridGraphExpandPacked4Avx2(scratchB);
		acc128 = _mm_dpbusd_epi32(acc128,
								  _mm_xor_si128(ca, signFlip128), cb);
		accCSum128 = _mm_dpbusd_epi32(accCSum128, ones128, cb);
		*sampleDims += tailDims;
	}

	dot = (int64) _mm512_reduce_add_epi32(acc512) +
		PgturbohybridGraphHorizontalSumI32Avx2(acc256) +
		PgturbohybridGraphHorizontalSumI32x4Avx512Vnni(acc128);
	cSum = (int64) _mm512_reduce_add_epi32(accCSum512) +
		PgturbohybridGraphHorizontalSumI32Avx2(accCSum256) +
		PgturbohybridGraphHorizontalSumI32x4Avx512Vnni(accCSum128);

	return dot - 128 * cSum;
}

int64 PGTURBOHYBRID_GRAPH_AVX512VNNI_TARGET
PgturbohybridGraphCodeCode2RawAvx512Vnni(const uint8 *a, const uint8 *b, int dim,
							  int *sampleDims)
{
	const __m512i signFlip512 = _mm512_set1_epi8((char) 0x80);
	const __m512i ones512 = _mm512_set1_epi8(1);
	const __m256i signFlip256 = _mm256_set1_epi8((char) 0x80);
	const __m256i ones256 = _mm256_set1_epi8(1);
	const __m128i signFlip128 = _mm_set1_epi8((char) 0x80);
	const __m128i ones128 = _mm_set1_epi8(1);
	__m512i		acc512 = _mm512_setzero_si512();
	__m512i		accCSum512 = _mm512_setzero_si512();
	__m256i		acc256 = _mm256_setzero_si256();
	__m256i		accCSum256 = _mm256_setzero_si256();
	__m128i		acc128 = _mm_setzero_si128();
	__m128i		accCSum128 = _mm_setzero_si128();
	int			chunks = dim / 16;
	int			tailDims = dim - chunks * 16;
	int			scoredChunks = chunks;
	int			scored = 0;
	int64		dot;
	int64		cSum;

	if (chunks > PGTURBOHYBRID_GRAPH_CODE_CODE_PRUNE_CHUNKS)
		scoredChunks = PGTURBOHYBRID_GRAPH_CODE_CODE_PRUNE_CHUNKS;
	*sampleDims = scoredChunks * 16;

	if (scoredChunks == chunks)
	{
		for (; scored + 4 <= scoredChunks; scored += 4)
		{
			__m512i		ca = PgturbohybridGraphExpandPacked2Avx512(a + scored * 4);
			__m512i		cb = PgturbohybridGraphExpandPacked2Avx512(b + scored * 4);

			acc512 = _mm512_dpbusd_epi32(acc512,
										 _mm512_xor_si512(ca, signFlip512), cb);
			accCSum512 = _mm512_dpbusd_epi32(accCSum512, ones512, cb);
		}
		if (scored + 2 <= scoredChunks)
		{
			__m128i		ca0 = PgturbohybridGraphExpandPacked2Avx2(a + scored * 4);
			__m128i		cb0 = PgturbohybridGraphExpandPacked2Avx2(b + scored * 4);
			__m128i		ca1 = PgturbohybridGraphExpandPacked2Avx2(a + (scored + 1) * 4);
			__m128i		cb1 = PgturbohybridGraphExpandPacked2Avx2(b + (scored + 1) * 4);
			__m256i		ca = _mm256_inserti128_si256(_mm256_castsi128_si256(ca0), ca1, 1);
			__m256i		cb = _mm256_inserti128_si256(_mm256_castsi128_si256(cb0), cb1, 1);

			acc256 = _mm256_dpbusd_epi32(acc256,
										 _mm256_xor_si256(ca, signFlip256), cb);
			accCSum256 = _mm256_dpbusd_epi32(accCSum256, ones256, cb);
			scored += 2;
		}
	}
	else
	{
		for (; scored + 2 <= scoredChunks; scored += 2)
		{
			int			chunkA = (int) (((int64) scored * chunks) / scoredChunks);
			int			chunkB = (int) (((int64) (scored + 1) * chunks) / scoredChunks);
			__m128i		ca0 = PgturbohybridGraphExpandPacked2Avx2(a + chunkA * 4);
			__m128i		cb0 = PgturbohybridGraphExpandPacked2Avx2(b + chunkA * 4);
			__m128i		ca1 = PgturbohybridGraphExpandPacked2Avx2(a + chunkB * 4);
			__m128i		cb1 = PgturbohybridGraphExpandPacked2Avx2(b + chunkB * 4);
			__m256i		ca = _mm256_inserti128_si256(_mm256_castsi128_si256(ca0), ca1, 1);
			__m256i		cb = _mm256_inserti128_si256(_mm256_castsi128_si256(cb0), cb1, 1);

			acc256 = _mm256_dpbusd_epi32(acc256,
										 _mm256_xor_si256(ca, signFlip256), cb);
			accCSum256 = _mm256_dpbusd_epi32(accCSum256, ones256, cb);
		}
	}

	if (scored < scoredChunks)
	{
		int			chunk = scoredChunks == chunks ? scored :
			(int) (((int64) scored * chunks) / scoredChunks);
		__m128i		ca = PgturbohybridGraphExpandPacked2Avx2(a + chunk * 4);
		__m128i		cb = PgturbohybridGraphExpandPacked2Avx2(b + chunk * 4);

		acc128 = _mm_dpbusd_epi32(acc128,
								  _mm_xor_si128(ca, signFlip128), cb);
		accCSum128 = _mm_dpbusd_epi32(accCSum128, ones128, cb);
	}

	if (scoredChunks == chunks && tailDims != 0)
	{
		uint8		scratchA[4] = {0};
		uint8		scratchB[4] = {0};
		int			tailBytes = (tailDims + 3) / 4;
		__m128i		ca;
		__m128i		cb;

		memcpy(scratchA, a + chunks * 4, tailBytes);
		memcpy(scratchB, b + chunks * 4, tailBytes);
		ca = PgturbohybridGraphExpandPacked2Avx2(scratchA);
		cb = PgturbohybridGraphExpandPacked2Avx2(scratchB);
		acc128 = _mm_dpbusd_epi32(acc128,
								  _mm_xor_si128(ca, signFlip128), cb);
		accCSum128 = _mm_dpbusd_epi32(accCSum128, ones128, cb);
		*sampleDims += tailDims;
	}

	dot = (int64) _mm512_reduce_add_epi32(acc512) +
		PgturbohybridGraphHorizontalSumI32Avx2(acc256) +
		PgturbohybridGraphHorizontalSumI32x4Avx512Vnni(acc128);
	cSum = (int64) _mm512_reduce_add_epi32(accCSum512) +
		PgturbohybridGraphHorizontalSumI32Avx2(accCSum256) +
		PgturbohybridGraphHorizontalSumI32x4Avx512Vnni(accCSum128);

	return dot - 128 * cSum;
}
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

int64 PGTURBOHYBRID_GRAPH_AVXVNNI_TARGET
PgturbohybridGraphQuerySplitRawAvxVnni(const PgturbohybridGraphTqQuery *tq, const uint8 *code)
{
	const __m256i ones256 = _mm256_set1_epi8(1);
	const __m128i ones128 = _mm_set1_epi8(1);
	__m256i		accLow = _mm256_setzero_si256();
	__m256i		accHigh = _mm256_setzero_si256();
	__m256i		accCSum = _mm256_setzero_si256();
	__m128i		accLowLo = _mm_setzero_si128();
	__m128i		accHighLo = _mm_setzero_si128();
	__m128i		accCSumLo = _mm_setzero_si128();
	int			chunk = 0;
	int64		dotLow;
	int64		dotHigh;
	int64		cSum;

	/* Pair-stepped main loop: 32 dims per iteration on YMM. */
	for (; chunk + 2 <= tq->querySplitChunks; chunk += 2)
	{
		__m128i		c0 = PgturbohybridGraphExpandPacked4Avx2(code + chunk * 8);
		__m128i		c1 = PgturbohybridGraphExpandPacked4Avx2(code + (chunk + 1) * 8);
		__m256i		c = _mm256_inserti128_si256(_mm256_castsi128_si256(c0), c1, 1);
		__m256i		low = _mm256_loadu_si256((const __m256i *) (tq->signedSplit.lowU8 + chunk * 16));
		__m256i		high = _mm256_loadu_si256((const __m256i *) (tq->signedSplit.highU8 + chunk * 16));

		accLow = _mm256_dpbusd_avx_epi32(accLow, low, c);
		accHigh = _mm256_dpbusd_avx_epi32(accHigh, high, c);
		accCSum = _mm256_dpbusd_avx_epi32(accCSum, ones256, c);
	}

	/* Trailing 16-dim chunk (if querySplitChunks is odd). */
	if (chunk < tq->querySplitChunks)
	{
		__m128i		c = PgturbohybridGraphExpandPacked4Avx2(code + chunk * 8);
		__m128i		low = _mm_loadu_si128((const __m128i *) (tq->signedSplit.lowU8 + chunk * 16));
		__m128i		high = _mm_loadu_si128((const __m128i *) (tq->signedSplit.highU8 + chunk * 16));

		accLowLo = _mm_dpbusd_avx_epi32(accLowLo, low, c);
		accHighLo = _mm_dpbusd_avx_epi32(accHighLo, high, c);
		accCSumLo = _mm_dpbusd_avx_epi32(accCSumLo, ones128, c);
	}

	/* Final sub-chunk tail dims. */
	if (tq->querySplitTailDims != 0)
	{
		uint8		scratch[8] = {0};
		int			tailBytes = (tq->querySplitTailDims + 1) / 2;
		__m128i		c;
		__m128i		low;
		__m128i		high;

		memcpy(scratch, code + tq->querySplitChunks * 8, tailBytes);
		c = PgturbohybridGraphExpandPacked4Avx2(scratch);
		low = _mm_loadu_si128((const __m128i *) tq->signedSplit.tailLowU8);
		high = _mm_loadu_si128((const __m128i *) tq->signedSplit.tailHighU8);
		accLowLo = _mm_dpbusd_avx_epi32(accLowLo, low, c);
		accHighLo = _mm_dpbusd_avx_epi32(accHighLo, high, c);
		accCSumLo = _mm_dpbusd_avx_epi32(accCSumLo, ones128, c);
	}

	dotLow = PgturbohybridGraphHorizontalSumI32Avx2(accLow) +
		PgturbohybridGraphHorizontalSumI32x4AvxVnni(accLowLo);
	dotHigh = PgturbohybridGraphHorizontalSumI32Avx2(accHigh) +
		PgturbohybridGraphHorizontalSumI32x4AvxVnni(accHighLo);
	cSum = PgturbohybridGraphHorizontalSumI32Avx2(accCSum) +
		PgturbohybridGraphHorizontalSumI32x4AvxVnni(accCSumLo);

	return (dotLow - 128 * cSum) +
		(int64) PGTURBOHYBRID_QUERY_SPLIT_HIGH_COEF * (dotHigh - 128 * cSum);
}

int64 PGTURBOHYBRID_GRAPH_AVXVNNI_TARGET
PgturbohybridGraphQuerySplit2RawAvxVnni(const PgturbohybridGraphTqQuery *tq, const uint8 *code)
{
	const __m256i ones256 = _mm256_set1_epi8(1);
	const __m128i ones128 = _mm_set1_epi8(1);
	__m256i		accLow = _mm256_setzero_si256();
	__m256i		accHigh = _mm256_setzero_si256();
	__m256i		accCSum = _mm256_setzero_si256();
	__m128i		accLowLo = _mm_setzero_si128();
	__m128i		accHighLo = _mm_setzero_si128();
	__m128i		accCSumLo = _mm_setzero_si128();
	int			chunk = 0;
	int64		dotLow;
	int64		dotHigh;
	int64		cSum;

	for (; chunk + 2 <= tq->querySplitChunks; chunk += 2)
	{
		__m128i		c0 = PgturbohybridGraphExpandPacked2Avx2(code + chunk * 4);
		__m128i		c1 = PgturbohybridGraphExpandPacked2Avx2(code + (chunk + 1) * 4);
		__m256i		c = _mm256_inserti128_si256(_mm256_castsi128_si256(c0), c1, 1);
		__m256i		low = _mm256_loadu_si256((const __m256i *) (tq->signedSplit.lowU8 + chunk * 16));
		__m256i		high = _mm256_loadu_si256((const __m256i *) (tq->signedSplit.highU8 + chunk * 16));

		accLow = _mm256_dpbusd_avx_epi32(accLow, low, c);
		accHigh = _mm256_dpbusd_avx_epi32(accHigh, high, c);
		accCSum = _mm256_dpbusd_avx_epi32(accCSum, ones256, c);
	}

	if (chunk < tq->querySplitChunks)
	{
		__m128i		c = PgturbohybridGraphExpandPacked2Avx2(code + chunk * 4);
		__m128i		low = _mm_loadu_si128((const __m128i *) (tq->signedSplit.lowU8 + chunk * 16));
		__m128i		high = _mm_loadu_si128((const __m128i *) (tq->signedSplit.highU8 + chunk * 16));

		accLowLo = _mm_dpbusd_avx_epi32(accLowLo, low, c);
		accHighLo = _mm_dpbusd_avx_epi32(accHighLo, high, c);
		accCSumLo = _mm_dpbusd_avx_epi32(accCSumLo, ones128, c);
	}

	if (tq->querySplitTailDims != 0)
	{
		uint8		scratch[4] = {0};
		int			tailBytes = (tq->querySplitTailDims + 3) / 4;
		__m128i		c;
		__m128i		low;
		__m128i		high;

		memcpy(scratch, code + tq->querySplitChunks * 4, tailBytes);
		c = PgturbohybridGraphExpandPacked2Avx2(scratch);
		low = _mm_loadu_si128((const __m128i *) tq->signedSplit.tailLowU8);
		high = _mm_loadu_si128((const __m128i *) tq->signedSplit.tailHighU8);
		accLowLo = _mm_dpbusd_avx_epi32(accLowLo, low, c);
		accHighLo = _mm_dpbusd_avx_epi32(accHighLo, high, c);
		accCSumLo = _mm_dpbusd_avx_epi32(accCSumLo, ones128, c);
	}

	dotLow = PgturbohybridGraphHorizontalSumI32Avx2(accLow) +
		PgturbohybridGraphHorizontalSumI32x4AvxVnni(accLowLo);
	dotHigh = PgturbohybridGraphHorizontalSumI32Avx2(accHigh) +
		PgturbohybridGraphHorizontalSumI32x4AvxVnni(accHighLo);
	cSum = PgturbohybridGraphHorizontalSumI32Avx2(accCSum) +
		PgturbohybridGraphHorizontalSumI32x4AvxVnni(accCSumLo);

	return (dotLow - 128 * cSum) +
		(int64) PGTURBOHYBRID_QUERY_SPLIT_HIGH_COEF * (dotHigh - 128 * cSum);
}

int64 PGTURBOHYBRID_GRAPH_AVXVNNI_TARGET
PgturbohybridGraphCodeCodeRawAvxVnni(const uint8 *a, const uint8 *b, int dim,
						  int *sampleDims)
{
	const __m256i signFlip256 = _mm256_set1_epi8((char) 0x80);
	const __m256i ones256 = _mm256_set1_epi8(1);
	const __m128i signFlip128 = _mm_set1_epi8((char) 0x80);
	const __m128i ones128 = _mm_set1_epi8(1);
	__m256i		acc256 = _mm256_setzero_si256();
	__m256i		accCSum256 = _mm256_setzero_si256();
	__m128i		accLo = _mm_setzero_si128();
	__m128i		accCSumLo = _mm_setzero_si128();
	int			chunks = dim / 16;
	int			tailDims = dim - chunks * 16;
	int			scoredChunks = chunks;
	int			scored = 0;
	int64		dot;
	int64		cSum;

	if (chunks > PGTURBOHYBRID_GRAPH_CODE_CODE_PRUNE_CHUNKS)
		scoredChunks = PGTURBOHYBRID_GRAPH_CODE_CODE_PRUNE_CHUNKS;
	*sampleDims = scoredChunks * 16;

	for (; scored + 2 <= scoredChunks; scored += 2)
	{
		int			chunkA = scoredChunks == chunks ? scored :
			(int) (((int64) scored * chunks) / scoredChunks);
		int			chunkB = scoredChunks == chunks ? scored + 1 :
			(int) (((int64) (scored + 1) * chunks) / scoredChunks);
		__m128i		ca0 = PgturbohybridGraphExpandPacked4Avx2(a + chunkA * 8);
		__m128i		cb0 = PgturbohybridGraphExpandPacked4Avx2(b + chunkA * 8);
		__m128i		ca1 = PgturbohybridGraphExpandPacked4Avx2(a + chunkB * 8);
		__m128i		cb1 = PgturbohybridGraphExpandPacked4Avx2(b + chunkB * 8);
		__m256i		ca = _mm256_inserti128_si256(_mm256_castsi128_si256(ca0), ca1, 1);
		__m256i		cb = _mm256_inserti128_si256(_mm256_castsi128_si256(cb0), cb1, 1);

		acc256 = _mm256_dpbusd_avx_epi32(acc256,
										 _mm256_xor_si256(ca, signFlip256), cb);
		accCSum256 = _mm256_dpbusd_avx_epi32(accCSum256, ones256, cb);
	}

	if (scored < scoredChunks)
	{
		int			chunk = scoredChunks == chunks ? scored :
			(int) (((int64) scored * chunks) / scoredChunks);
		__m128i		ca = PgturbohybridGraphExpandPacked4Avx2(a + chunk * 8);
		__m128i		cb = PgturbohybridGraphExpandPacked4Avx2(b + chunk * 8);

		accLo = _mm_dpbusd_avx_epi32(accLo,
									 _mm_xor_si128(ca, signFlip128), cb);
		accCSumLo = _mm_dpbusd_avx_epi32(accCSumLo, ones128, cb);
	}

	if (scoredChunks == chunks && tailDims != 0)
	{
		uint8		scratchA[8] = {0};
		uint8		scratchB[8] = {0};
		int			tailBytes = (tailDims + 1) / 2;
		__m128i		ca;
		__m128i		cb;

		memcpy(scratchA, a + chunks * 8, tailBytes);
		memcpy(scratchB, b + chunks * 8, tailBytes);
		ca = PgturbohybridGraphExpandPacked4Avx2(scratchA);
		cb = PgturbohybridGraphExpandPacked4Avx2(scratchB);
		accLo = _mm_dpbusd_avx_epi32(accLo,
									 _mm_xor_si128(ca, signFlip128), cb);
		accCSumLo = _mm_dpbusd_avx_epi32(accCSumLo, ones128, cb);
		*sampleDims += tailDims;
	}

	dot = PgturbohybridGraphHorizontalSumI32Avx2(acc256) +
		PgturbohybridGraphHorizontalSumI32x4AvxVnni(accLo);
	cSum = PgturbohybridGraphHorizontalSumI32Avx2(accCSum256) +
		PgturbohybridGraphHorizontalSumI32x4AvxVnni(accCSumLo);

	return dot - 128 * cSum;
}

int64 PGTURBOHYBRID_GRAPH_AVXVNNI_TARGET
PgturbohybridGraphCodeCode2RawAvxVnni(const uint8 *a, const uint8 *b, int dim,
						   int *sampleDims)
{
	const __m256i signFlip256 = _mm256_set1_epi8((char) 0x80);
	const __m256i ones256 = _mm256_set1_epi8(1);
	const __m128i signFlip128 = _mm_set1_epi8((char) 0x80);
	const __m128i ones128 = _mm_set1_epi8(1);
	__m256i		acc256 = _mm256_setzero_si256();
	__m256i		accCSum256 = _mm256_setzero_si256();
	__m128i		accLo = _mm_setzero_si128();
	__m128i		accCSumLo = _mm_setzero_si128();
	int			chunks = dim / 16;
	int			tailDims = dim - chunks * 16;
	int			scoredChunks = chunks;
	int			scored = 0;
	int64		dot;
	int64		cSum;

	if (chunks > PGTURBOHYBRID_GRAPH_CODE_CODE_PRUNE_CHUNKS)
		scoredChunks = PGTURBOHYBRID_GRAPH_CODE_CODE_PRUNE_CHUNKS;
	*sampleDims = scoredChunks * 16;

	for (; scored + 2 <= scoredChunks; scored += 2)
	{
		int			chunkA = scoredChunks == chunks ? scored :
			(int) (((int64) scored * chunks) / scoredChunks);
		int			chunkB = scoredChunks == chunks ? scored + 1 :
			(int) (((int64) (scored + 1) * chunks) / scoredChunks);
		__m128i		ca0 = PgturbohybridGraphExpandPacked2Avx2(a + chunkA * 4);
		__m128i		cb0 = PgturbohybridGraphExpandPacked2Avx2(b + chunkA * 4);
		__m128i		ca1 = PgturbohybridGraphExpandPacked2Avx2(a + chunkB * 4);
		__m128i		cb1 = PgturbohybridGraphExpandPacked2Avx2(b + chunkB * 4);
		__m256i		ca = _mm256_inserti128_si256(_mm256_castsi128_si256(ca0), ca1, 1);
		__m256i		cb = _mm256_inserti128_si256(_mm256_castsi128_si256(cb0), cb1, 1);

		acc256 = _mm256_dpbusd_avx_epi32(acc256,
										 _mm256_xor_si256(ca, signFlip256), cb);
		accCSum256 = _mm256_dpbusd_avx_epi32(accCSum256, ones256, cb);
	}

	if (scored < scoredChunks)
	{
		int			chunk = scoredChunks == chunks ? scored :
			(int) (((int64) scored * chunks) / scoredChunks);
		__m128i		ca = PgturbohybridGraphExpandPacked2Avx2(a + chunk * 4);
		__m128i		cb = PgturbohybridGraphExpandPacked2Avx2(b + chunk * 4);

		accLo = _mm_dpbusd_avx_epi32(accLo,
									 _mm_xor_si128(ca, signFlip128), cb);
		accCSumLo = _mm_dpbusd_avx_epi32(accCSumLo, ones128, cb);
	}

	if (scoredChunks == chunks && tailDims != 0)
	{
		uint8		scratchA[4] = {0};
		uint8		scratchB[4] = {0};
		int			tailBytes = (tailDims + 3) / 4;
		__m128i		ca;
		__m128i		cb;

		memcpy(scratchA, a + chunks * 4, tailBytes);
		memcpy(scratchB, b + chunks * 4, tailBytes);
		ca = PgturbohybridGraphExpandPacked2Avx2(scratchA);
		cb = PgturbohybridGraphExpandPacked2Avx2(scratchB);
		accLo = _mm_dpbusd_avx_epi32(accLo,
									 _mm_xor_si128(ca, signFlip128), cb);
		accCSumLo = _mm_dpbusd_avx_epi32(accCSumLo, ones128, cb);
		*sampleDims += tailDims;
	}

	dot = PgturbohybridGraphHorizontalSumI32Avx2(acc256) +
		PgturbohybridGraphHorizontalSumI32x4AvxVnni(accLo);
	cSum = PgturbohybridGraphHorizontalSumI32Avx2(accCSum256) +
		PgturbohybridGraphHorizontalSumI32x4AvxVnni(accCSumLo);

	return dot - 128 * cSum;
}
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
