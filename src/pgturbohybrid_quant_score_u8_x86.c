/*
 * pgturbohybrid_quant_score_u8_x86.c
 *
 * x86 unsigned-codebook (u8) 4-bit query-split scoring kernels, extracted
 * verbatim from pgturbohybrid_quant_score.c (no behaviour change).  Holds the
 * scalar / AVX2 / AVX-512 VNNI raw kernels, the shared u8 postprocess, and the
 * two PackedDistanceU8Split dispatchers; the generic dispatch in
 * pgturbohybrid_quant_score.c calls the non-static entry points declared in
 * pgturbohybrid_quant_score_internal.h.
 *
 * The compile/target guard macros below are kept byte-for-byte in sync with the
 * copies in pgturbohybrid_quant_score.c so the guards stay exactly equivalent.
 */
#include "postgres.h"

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

#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2 || PGTURBOHYBRID_GRAPH_COMPILE_AVX512VNNI
#include <immintrin.h>
#endif

#include "pgturbohybrid_quant_score.h"
#include "pgturbohybrid_quant_score_internal.h"

#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2

/*
 * x86 unsigned-codebook query-split representation.
 *
 * The codebook is the signed 4-bit codebook shifted by +OFFSET (=128) so the
 * un-shifted value equals the signed centre the stored codes were encoded with
 * (PgturbohybridGraphCodebookI8): codebook_u8[k] == PgturbohybridGraphCodebookI8[k] + 128.
 * This keeps scores numerically identical (codebook-wise) to the scalar/LUT
 * reference, while letting _mm256_maddubs_epi16 / _mm512_dpbusd_epi32 consume
 * it directly as their unsigned operand.  Query halves are signed 7-bit
 * (|low|,|high| <= 64) combined as q_signed = 128*high + low, keeping the
 * maddubs pair sum <= 2*255*64 = 32640 < 32767.  The +OFFSET shift contributes
 * a per-query bias OFFSET*Sum(q_signed) that the scorer subtracts once.
 */
/* U8_SPLIT_HIGH_COEF / U8_CODEBOOK_OFFSET come from the shared codebook header. */
#include "pgturbohybrid_quant_codebook.h"
#define PGTURBOHYBRID_U8_SPLIT_ABS_MAX 8127.0
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
static const uint8 PgturbohybridGraphCodebookU8[PGTURBOHYBRID_LUT_WIDTH] = {
	1, 32, 53, 70, 84, 97, 110, 122,
	134, 146, 159, 172, 186, 203, 224, 255
};
#endif

/*
 * Unsigned-codebook (u8) 4-bit query-split raw scorers.  Return
 *   D = Sum_i q_signed[i] * codebook_u8[code[i]]   (= dotLow + 128 * dotHigh)
 * with q_signed = 128*high + low.  The +128 codebook-shift bias is removed by
 * the caller (PgturbohybridGraphPackedDistanceU8Split) via u8.bias.
 *
 * Scalar reference -- the SIMD kernels below must match this exactly.
 */
int64
PgturbohybridGraphQuerySplitU8RawScalar(const PgturbohybridGraphTqQuery *tq, const uint8 *code)
{
	int64		dotLow = 0;
	int64		dotHigh = 0;
	int			chunks = tq->querySplitChunks;
	int			fullDims = chunks * 16;

	for (int i = 0; i < fullDims; i++)
	{
		int			nib = TqGetCodeComponentBits(code, i, PGTURBOHYBRID_DEFAULT_BITS);
		int			cu = PgturbohybridGraphCodebookU8[nib];
		int			chunk = i / 16;
		int			lane = i % 16;

		dotLow += (int64) tq->u8.data[chunk * 32 + lane] * cu;
		dotHigh += (int64) tq->u8.data[chunk * 32 + 16 + lane] * cu;
	}
	for (int j = 0; j < tq->querySplitTailDims; j++)
	{
		int			nib = TqGetCodeComponentBits(code, fullDims + j, PGTURBOHYBRID_DEFAULT_BITS);
		int			cu = PgturbohybridGraphCodebookU8[nib];

		dotLow += (int64) tq->u8.tailLow[j] * cu;
		dotHigh += (int64) tq->u8.tailHigh[j] * cu;
	}
	return dotLow + (int64) PGTURBOHYBRID_U8_SPLIT_HIGH_COEF * dotHigh;
}

/*
 * AVX2: codebook (u8) broadcast to both 128-bit lanes; the [low|high] query
 * pair fills the 256-bit register so one _mm256_maddubs_epi16 produces c*low in
 * the low lane and c*high in the high lane.
 */
static int64 PGTURBOHYBRID_GRAPH_AVX2_TARGET
PgturbohybridGraphQuerySplitU8RawAvx2(const PgturbohybridGraphTqQuery *tq, const uint8 *code)
{
	const __m128i mask = _mm_set1_epi8(0x0f);
	const __m128i cbU8 = _mm_loadu_si128((const __m128i *) PgturbohybridGraphCodebookU8);
	const __m256i ones = _mm256_set1_epi16(1);
	const __m128i ones128 = _mm_set1_epi16(1);
	__m256i		acc = _mm256_setzero_si256();
	__m128i		accLow128 = _mm_setzero_si128();
	__m128i		accHigh128 = _mm_setzero_si128();
	int32		s[8];
	int32		sl[4];
	int32		sh[4];
	int64		dotLow;
	int64		dotHigh;

	for (int chunk = 0; chunk < tq->querySplitChunks; chunk++)
	{
		__m128i		packed = _mm_loadl_epi64((const __m128i *) (code + chunk * 8));
		__m128i		lo = _mm_and_si128(packed, mask);
		__m128i		hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
		__m128i		idx = _mm_unpacklo_epi8(lo, hi);
		__m128i		cvals = _mm_shuffle_epi8(cbU8, idx);
		__m256i		c256 = _mm256_broadcastsi128_si256(cvals);
		__m256i		lh = _mm256_loadu_si256((const __m256i *) (tq->u8.data + chunk * 32));

		acc = _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(c256, lh), ones));
	}

	if (tq->querySplitTailDims != 0)
	{
		uint8		scratch[8] = {0};
		int			tailBytes = (tq->querySplitTailDims + 1) / 2;
		__m128i		packed;
		__m128i		lo;
		__m128i		hi;
		__m128i		idx;
		__m128i		cvals;
		__m128i		lowv;
		__m128i		highv;

		memcpy(scratch, code + tq->querySplitChunks * 8, tailBytes);
		packed = _mm_loadl_epi64((const __m128i *) scratch);
		lo = _mm_and_si128(packed, mask);
		hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
		idx = _mm_unpacklo_epi8(lo, hi);
		cvals = _mm_shuffle_epi8(cbU8, idx);
		lowv = _mm_loadu_si128((const __m128i *) tq->u8.tailLow);
		highv = _mm_loadu_si128((const __m128i *) tq->u8.tailHigh);
		accLow128 = _mm_add_epi32(accLow128, _mm_madd_epi16(_mm_maddubs_epi16(cvals, lowv), ones128));
		accHigh128 = _mm_add_epi32(accHigh128, _mm_madd_epi16(_mm_maddubs_epi16(cvals, highv), ones128));
	}

	_mm256_storeu_si256((__m256i *) s, acc);
	_mm_storeu_si128((__m128i *) sl, accLow128);
	_mm_storeu_si128((__m128i *) sh, accHigh128);
	dotLow = (int64) s[0] + s[1] + s[2] + s[3] + sl[0] + sl[1] + sl[2] + sl[3];
	dotHigh = (int64) s[4] + s[5] + s[6] + s[7] + sh[0] + sh[1] + sh[2] + sh[3];
	return dotLow + (int64) PGTURBOHYBRID_U8_SPLIT_HIGH_COEF * dotHigh;
}

/*
 * Inline helpers for the x4 batch kernel.  Decoding one 16-dim chunk and the
 * maddubs accumulate against the shared [low|high] query are factored out so the
 * batch loop body reads as four independent, named accumulator chains (see the
 * explicit-accumulator rationale on the x4 kernel below).
 */

/* Decode one 16-dim packed-4bit chunk to its u8 codebook values, broadcast to
 * both 128-bit lanes (so one maddubs scores it against [low|high] at once). */
static inline __m256i PGTURBOHYBRID_GRAPH_AVX2_TARGET
PgturbohybridGraphU8Decode16Avx2(const uint8 *codeChunk, __m128i mask, __m128i cbU8)
{
	__m128i		packed = _mm_loadl_epi64((const __m128i *) codeChunk);
	__m128i		lo = _mm_and_si128(packed, mask);
	__m128i		hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
	__m128i		idx = _mm_unpacklo_epi8(lo, hi);

	return _mm256_broadcastsi128_si256(_mm_shuffle_epi8(cbU8, idx));
}

/* Accumulate one decoded chunk (c256) against the shared [low|high] query pair. */
static inline __m256i PGTURBOHYBRID_GRAPH_AVX2_TARGET
PgturbohybridGraphU8Accum16Avx2(__m256i acc, __m256i c256, __m256i lh, __m256i ones)
{
	return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(c256, lh), ones));
}

/* Accumulate one code's non-multiple-of-16 tail (low/high 128-bit lanes). */
static inline void PGTURBOHYBRID_GRAPH_AVX2_TARGET
PgturbohybridGraphU8TailAccumAvx2(const uint8 *codeTail, int tailBytes, __m128i mask,
								  __m128i cbU8, __m128i lowv, __m128i highv, __m128i ones128,
								  __m128i *accLow, __m128i *accHigh)
{
	uint8		scratch[8] = {0};
	__m128i		packed;
	__m128i		lo;
	__m128i		hi;
	__m128i		idx;
	__m128i		cvals;

	memcpy(scratch, codeTail, tailBytes);
	packed = _mm_loadl_epi64((const __m128i *) scratch);
	lo = _mm_and_si128(packed, mask);
	hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
	idx = _mm_unpacklo_epi8(lo, hi);
	cvals = _mm_shuffle_epi8(cbU8, idx);
	*accLow = _mm_add_epi32(*accLow, _mm_madd_epi16(_mm_maddubs_epi16(cvals, lowv), ones128));
	*accHigh = _mm_add_epi32(*accHigh, _mm_madd_epi16(_mm_maddubs_epi16(cvals, highv), ones128));
}

/* Reduce one code's 256-bit full-chunk accumulator plus its 128-bit tail
 * low/high lanes to the raw integer dot dotLow + HIGH_COEF*dotHigh.  Same lane
 * order (low = first 4 i32, high = last 4) as PgturbohybridGraphQuerySplitU8RawAvx2. */
static inline int64 PGTURBOHYBRID_GRAPH_AVX2_TARGET
PgturbohybridGraphU8ReduceAvx2(__m256i acc, __m128i accLow, __m128i accHigh)
{
	int32		s[8];
	int32		sl[4];
	int32		sh[4];
	int64		dotLow;
	int64		dotHigh;

	_mm256_storeu_si256((__m256i *) s, acc);
	_mm_storeu_si128((__m128i *) sl, accLow);
	_mm_storeu_si128((__m128i *) sh, accHigh);
	dotLow = (int64) s[0] + s[1] + s[2] + s[3] + sl[0] + sl[1] + sl[2] + sl[3];
	dotHigh = (int64) s[4] + s[5] + s[6] + s[7] + sh[0] + sh[1] + sh[2] + sh[3];
	return dotLow + (int64) PGTURBOHYBRID_U8_SPLIT_HIGH_COEF * dotHigh;
}

/*
 * True 4-candidate AVX2 u8 batch: loads each query [low|high] chunk once and
 * decodes/accumulates four codes against it, instead of walking the whole query
 * four times.  Bit-identical to four single-node
 * PgturbohybridGraphQuerySplitU8RawAvx2() calls.
 *
 * Explicit per-candidate accumulators (acc0..acc3 and tail low0/high0..low3/high3)
 * rather than __m256i acc[4] + an inner c-loop: the four maddubs/madd chains are
 * independent, and naming each accumulator keeps it in its own vector register
 * and lets the scheduler interleave the four dependency chains.  The array form
 * relies on the compiler fully unrolling the inner loop AND proving the array
 * slots never alias through memory; with explicit scalars that is unconditional,
 * which avoids any address-taken / stack-materialised accumulator.
 */
static void PGTURBOHYBRID_GRAPH_AVX2_TARGET
PgturbohybridGraphQuerySplitU8RawAvx2x4(const PgturbohybridGraphTqQuery *tq,
										const uint8 *codes[4], int64 raw[4])
{
	const __m128i mask = _mm_set1_epi8(0x0f);
	const __m128i cbU8 = _mm_loadu_si128((const __m128i *) PgturbohybridGraphCodebookU8);
	const __m256i ones = _mm256_set1_epi16(1);
	const __m128i ones128 = _mm_set1_epi16(1);
	const uint8 *c0 = codes[0];
	const uint8 *c1 = codes[1];
	const uint8 *c2 = codes[2];
	const uint8 *c3 = codes[3];
	__m256i		acc0 = _mm256_setzero_si256();
	__m256i		acc1 = _mm256_setzero_si256();
	__m256i		acc2 = _mm256_setzero_si256();
	__m256i		acc3 = _mm256_setzero_si256();
	__m128i		low0 = _mm_setzero_si128();
	__m128i		high0 = _mm_setzero_si128();
	__m128i		low1 = _mm_setzero_si128();
	__m128i		high1 = _mm_setzero_si128();
	__m128i		low2 = _mm_setzero_si128();
	__m128i		high2 = _mm_setzero_si128();
	__m128i		low3 = _mm_setzero_si128();
	__m128i		high3 = _mm_setzero_si128();

	for (int chunk = 0; chunk < tq->querySplitChunks; chunk++)
	{
		__m256i		lh = _mm256_loadu_si256((const __m256i *) (tq->u8.data + chunk * 32));

		acc0 = PgturbohybridGraphU8Accum16Avx2(acc0,
				PgturbohybridGraphU8Decode16Avx2(c0 + chunk * 8, mask, cbU8), lh, ones);
		acc1 = PgturbohybridGraphU8Accum16Avx2(acc1,
				PgturbohybridGraphU8Decode16Avx2(c1 + chunk * 8, mask, cbU8), lh, ones);
		acc2 = PgturbohybridGraphU8Accum16Avx2(acc2,
				PgturbohybridGraphU8Decode16Avx2(c2 + chunk * 8, mask, cbU8), lh, ones);
		acc3 = PgturbohybridGraphU8Accum16Avx2(acc3,
				PgturbohybridGraphU8Decode16Avx2(c3 + chunk * 8, mask, cbU8), lh, ones);
	}

	if (tq->querySplitTailDims != 0)
	{
		int			tailBytes = (tq->querySplitTailDims + 1) / 2;
		int			tailOff = tq->querySplitChunks * 8;
		__m128i		lowv = _mm_loadu_si128((const __m128i *) tq->u8.tailLow);
		__m128i		highv = _mm_loadu_si128((const __m128i *) tq->u8.tailHigh);

		PgturbohybridGraphU8TailAccumAvx2(c0 + tailOff, tailBytes, mask, cbU8, lowv, highv, ones128, &low0, &high0);
		PgturbohybridGraphU8TailAccumAvx2(c1 + tailOff, tailBytes, mask, cbU8, lowv, highv, ones128, &low1, &high1);
		PgturbohybridGraphU8TailAccumAvx2(c2 + tailOff, tailBytes, mask, cbU8, lowv, highv, ones128, &low2, &high2);
		PgturbohybridGraphU8TailAccumAvx2(c3 + tailOff, tailBytes, mask, cbU8, lowv, highv, ones128, &low3, &high3);
	}

	raw[0] = PgturbohybridGraphU8ReduceAvx2(acc0, low0, high0);
	raw[1] = PgturbohybridGraphU8ReduceAvx2(acc1, low1, high1);
	raw[2] = PgturbohybridGraphU8ReduceAvx2(acc2, low2, high2);
	raw[3] = PgturbohybridGraphU8ReduceAvx2(acc3, low3, high3);
}

#if PGTURBOHYBRID_GRAPH_COMPILE_AVX512VNNI
/*
 * AVX-512 VNNI: two [low|high] chunks (64 bytes) per VPDPBUSD.  ZMM 128-lanes
 * carry [a.low, a.high, b.low, b.high]; the codebook is broadcast per chunk so
 * dpbusd's u8*i8 fused dot lands a.low/a.high in the low 256 and b.low/b.high
 * in the high 256.
 */
static int64 PGTURBOHYBRID_GRAPH_AVX512VNNI_TARGET
PgturbohybridGraphQuerySplitU8RawAvx512Vnni(const PgturbohybridGraphTqQuery *tq, const uint8 *code)
{
	const __m128i mask = _mm_set1_epi8(0x0f);
	const __m128i cbU8 = _mm_loadu_si128((const __m128i *) PgturbohybridGraphCodebookU8);
	const __m512i cb512 = _mm512_broadcast_i32x4(cbU8);
	const __m128i ones128 = _mm_set1_epi16(1);
	__m512i		acc = _mm512_setzero_si512();
	__m128i		sumLow;
	__m128i		sumHigh;
	int			chunks = tq->querySplitChunks;
	int			nPairs = chunks / 2;
	int32		sl[4];
	int32		sh[4];

	for (int i = 0; i < nPairs; i++)
	{
		__m512i		lh = _mm512_loadu_si512((const __m512i *) (tq->u8.data + i * 64));
		__m128i		packed = _mm_loadu_si128((const __m128i *) (code + i * 16));
		__m128i		lo = _mm_and_si128(packed, mask);
		__m128i		hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
		__m128i		va = _mm_unpacklo_epi8(lo, hi);
		__m128i		vb = _mm_unpackhi_epi8(lo, hi);
		__m256i		vaDup = _mm256_broadcastsi128_si256(va);
		__m256i		vbDup = _mm256_broadcastsi128_si256(vb);
		__m512i		v512 = _mm512_inserti64x4(_mm512_castsi256_si512(vaDup), vbDup, 1);
		__m512i		c512 = _mm512_shuffle_epi8(cb512, v512);

		acc = _mm512_dpbusd_epi32(acc, c512, lh);
	}

	{
		__m256i		lo256 = _mm512_castsi512_si256(acc);
		__m256i		hi256 = _mm512_extracti64x4_epi64(acc, 1);

		sumLow = _mm_add_epi32(_mm256_castsi256_si128(lo256),
							   _mm256_castsi256_si128(hi256));
		sumHigh = _mm_add_epi32(_mm256_extracti128_si256(lo256, 1),
								_mm256_extracti128_si256(hi256, 1));
	}

	if (chunks & 1)
	{
		int			chunk = nPairs * 2;
		__m128i		packed = _mm_loadl_epi64((const __m128i *) (code + chunk * 8));
		__m128i		lo = _mm_and_si128(packed, mask);
		__m128i		hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
		__m128i		idx = _mm_unpacklo_epi8(lo, hi);
		__m128i		cvals = _mm_shuffle_epi8(cbU8, idx);
		__m128i		lowv = _mm_loadu_si128((const __m128i *) (tq->u8.data + chunk * 32));
		__m128i		highv = _mm_loadu_si128((const __m128i *) (tq->u8.data + chunk * 32 + 16));

		sumLow = _mm_add_epi32(sumLow, _mm_madd_epi16(_mm_maddubs_epi16(cvals, lowv), ones128));
		sumHigh = _mm_add_epi32(sumHigh, _mm_madd_epi16(_mm_maddubs_epi16(cvals, highv), ones128));
	}

	if (tq->querySplitTailDims != 0)
	{
		uint8		scratch[8] = {0};
		int			tailBytes = (tq->querySplitTailDims + 1) / 2;
		__m128i		packed;
		__m128i		lo;
		__m128i		hi;
		__m128i		idx;
		__m128i		cvals;
		__m128i		lowv;
		__m128i		highv;

		memcpy(scratch, code + tq->querySplitChunks * 8, tailBytes);
		packed = _mm_loadl_epi64((const __m128i *) scratch);
		lo = _mm_and_si128(packed, mask);
		hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
		idx = _mm_unpacklo_epi8(lo, hi);
		cvals = _mm_shuffle_epi8(cbU8, idx);
		lowv = _mm_loadu_si128((const __m128i *) tq->u8.tailLow);
		highv = _mm_loadu_si128((const __m128i *) tq->u8.tailHigh);
		sumLow = _mm_add_epi32(sumLow, _mm_madd_epi16(_mm_maddubs_epi16(cvals, lowv), ones128));
		sumHigh = _mm_add_epi32(sumHigh, _mm_madd_epi16(_mm_maddubs_epi16(cvals, highv), ones128));
	}

	_mm_storeu_si128((__m128i *) sl, sumLow);
	_mm_storeu_si128((__m128i *) sh, sumHigh);
	return ((int64) sl[0] + sl[1] + sl[2] + sl[3]) +
		(int64) PGTURBOHYBRID_U8_SPLIT_HIGH_COEF * ((int64) sh[0] + sh[1] + sh[2] + sh[3]);
}

/* Decode one 32-dim packed chunk (two 16-dim halves a|b) to the VPDPBUSD code
 * operand: [a.low|a.high|b.low|b.high] codebook bytes across the four zmm lanes. */
static inline __m512i PGTURBOHYBRID_GRAPH_AVX512VNNI_TARGET
PgturbohybridGraphU8Decode32Avx512(const uint8 *codeChunk, __m128i mask, __m512i cb512)
{
	__m128i		packed = _mm_loadu_si128((const __m128i *) codeChunk);
	__m128i		lo = _mm_and_si128(packed, mask);
	__m128i		hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);
	__m128i		va = _mm_unpacklo_epi8(lo, hi);
	__m128i		vb = _mm_unpackhi_epi8(lo, hi);
	__m256i		vaDup = _mm256_broadcastsi128_si256(va);
	__m256i		vbDup = _mm256_broadcastsi128_si256(vb);
	__m512i		v512 = _mm512_inserti64x4(_mm512_castsi256_si512(vaDup), vbDup, 1);

	return _mm512_shuffle_epi8(cb512, v512);
}

/* Fold a code's 512-bit dpbusd accumulator into its 128-bit low/high lane sums
 * (low = lanes 0+2, high = lanes 1+3 of the ZMM). */
static inline void PGTURBOHYBRID_GRAPH_AVX512VNNI_TARGET
PgturbohybridGraphU8Fold512Avx512(__m512i acc, __m128i *sumLow, __m128i *sumHigh)
{
	__m256i		lo256 = _mm512_castsi512_si256(acc);
	__m256i		hi256 = _mm512_extracti64x4_epi64(acc, 1);

	*sumLow = _mm_add_epi32(_mm256_castsi256_si128(lo256), _mm256_castsi256_si128(hi256));
	*sumHigh = _mm_add_epi32(_mm256_extracti128_si256(lo256, 1), _mm256_extracti128_si256(hi256, 1));
}

/* 128-bit maddubs accumulate of one decoded 16-dim chunk (odd full chunk or the
 * non-multiple-of-16 tail) into a code's low/high lane sums. */
static inline void PGTURBOHYBRID_GRAPH_AVX512VNNI_TARGET
PgturbohybridGraphU8Madd128Avx512(__m128i cvals, __m128i lowv, __m128i highv, __m128i ones128,
								  __m128i *sumLow, __m128i *sumHigh)
{
	*sumLow = _mm_add_epi32(*sumLow, _mm_madd_epi16(_mm_maddubs_epi16(cvals, lowv), ones128));
	*sumHigh = _mm_add_epi32(*sumHigh, _mm_madd_epi16(_mm_maddubs_epi16(cvals, highv), ones128));
}

/* Decode one 16-dim packed chunk (already loaded into `packed`) to cvals. */
static inline __m128i PGTURBOHYBRID_GRAPH_AVX512VNNI_TARGET
PgturbohybridGraphU8Decode16CvalsAvx512(__m128i packed, __m128i mask, __m128i cbU8)
{
	__m128i		lo = _mm_and_si128(packed, mask);
	__m128i		hi = _mm_and_si128(_mm_srli_epi16(packed, 4), mask);

	return _mm_shuffle_epi8(cbU8, _mm_unpacklo_epi8(lo, hi));
}

/* Reduce one code's 128-bit low/high lane sums to the raw integer dot. */
static inline int64 PGTURBOHYBRID_GRAPH_AVX512VNNI_TARGET
PgturbohybridGraphU8Reduce512Avx512(__m128i sumLow, __m128i sumHigh)
{
	int32		sl[4];
	int32		sh[4];

	_mm_storeu_si128((__m128i *) sl, sumLow);
	_mm_storeu_si128((__m128i *) sh, sumHigh);
	return ((int64) sl[0] + sl[1] + sl[2] + sl[3]) +
		(int64) PGTURBOHYBRID_U8_SPLIT_HIGH_COEF * ((int64) sh[0] + sh[1] + sh[2] + sh[3]);
}

/*
 * True 4-candidate AVX-512 VNNI u8 batch: each 64-byte query pair is loaded once
 * and dpbusd'd against four decoded codes.  Bit-identical to four single-node
 * PgturbohybridGraphQuerySplitU8RawAvx512Vnni() calls.
 *
 * Explicit per-candidate accumulators (acc0..acc3 ZMM, sumLow0..sumHigh3 XMM)
 * rather than __m512i acc[4] + an inner c-loop: the four dpbusd chains are
 * independent and 4 of the 32 ZMM registers easily hold them, but the array form
 * forced the compiler to materialise the accumulators on the stack (measured: 9
 * vector spill stores in the inner loop).  Naming each accumulator keeps all four
 * dpbusd chains resident and lets the scheduler interleave them.
 */
static void PGTURBOHYBRID_GRAPH_AVX512VNNI_TARGET
PgturbohybridGraphQuerySplitU8RawAvx512Vnnix4(const PgturbohybridGraphTqQuery *tq,
											  const uint8 *codes[4], int64 raw[4])
{
	const __m128i mask = _mm_set1_epi8(0x0f);
	const __m128i cbU8 = _mm_loadu_si128((const __m128i *) PgturbohybridGraphCodebookU8);
	const __m512i cb512 = _mm512_broadcast_i32x4(cbU8);
	const __m128i ones128 = _mm_set1_epi16(1);
	const uint8 *c0 = codes[0];
	const uint8 *c1 = codes[1];
	const uint8 *c2 = codes[2];
	const uint8 *c3 = codes[3];
	int			chunks = tq->querySplitChunks;
	int			nPairs = chunks / 2;
	__m512i		acc0 = _mm512_setzero_si512();
	__m512i		acc1 = _mm512_setzero_si512();
	__m512i		acc2 = _mm512_setzero_si512();
	__m512i		acc3 = _mm512_setzero_si512();
	__m128i		sumLow0;
	__m128i		sumLow1;
	__m128i		sumLow2;
	__m128i		sumLow3;
	__m128i		sumHigh0;
	__m128i		sumHigh1;
	__m128i		sumHigh2;
	__m128i		sumHigh3;

	for (int i = 0; i < nPairs; i++)
	{
		__m512i		lh = _mm512_loadu_si512((const __m512i *) (tq->u8.data + i * 64));

		acc0 = _mm512_dpbusd_epi32(acc0, PgturbohybridGraphU8Decode32Avx512(c0 + i * 16, mask, cb512), lh);
		acc1 = _mm512_dpbusd_epi32(acc1, PgturbohybridGraphU8Decode32Avx512(c1 + i * 16, mask, cb512), lh);
		acc2 = _mm512_dpbusd_epi32(acc2, PgturbohybridGraphU8Decode32Avx512(c2 + i * 16, mask, cb512), lh);
		acc3 = _mm512_dpbusd_epi32(acc3, PgturbohybridGraphU8Decode32Avx512(c3 + i * 16, mask, cb512), lh);
	}

	PgturbohybridGraphU8Fold512Avx512(acc0, &sumLow0, &sumHigh0);
	PgturbohybridGraphU8Fold512Avx512(acc1, &sumLow1, &sumHigh1);
	PgturbohybridGraphU8Fold512Avx512(acc2, &sumLow2, &sumHigh2);
	PgturbohybridGraphU8Fold512Avx512(acc3, &sumLow3, &sumHigh3);

	if (chunks & 1)
	{
		int			chunk = nPairs * 2;
		__m128i		lowv = _mm_loadu_si128((const __m128i *) (tq->u8.data + chunk * 32));
		__m128i		highv = _mm_loadu_si128((const __m128i *) (tq->u8.data + chunk * 32 + 16));

		PgturbohybridGraphU8Madd128Avx512(PgturbohybridGraphU8Decode16CvalsAvx512(_mm_loadl_epi64((const __m128i *) (c0 + chunk * 8)), mask, cbU8), lowv, highv, ones128, &sumLow0, &sumHigh0);
		PgturbohybridGraphU8Madd128Avx512(PgturbohybridGraphU8Decode16CvalsAvx512(_mm_loadl_epi64((const __m128i *) (c1 + chunk * 8)), mask, cbU8), lowv, highv, ones128, &sumLow1, &sumHigh1);
		PgturbohybridGraphU8Madd128Avx512(PgturbohybridGraphU8Decode16CvalsAvx512(_mm_loadl_epi64((const __m128i *) (c2 + chunk * 8)), mask, cbU8), lowv, highv, ones128, &sumLow2, &sumHigh2);
		PgturbohybridGraphU8Madd128Avx512(PgturbohybridGraphU8Decode16CvalsAvx512(_mm_loadl_epi64((const __m128i *) (c3 + chunk * 8)), mask, cbU8), lowv, highv, ones128, &sumLow3, &sumHigh3);
	}

	if (tq->querySplitTailDims != 0)
	{
		int			tailBytes = (tq->querySplitTailDims + 1) / 2;
		int			tailOff = tq->querySplitChunks * 8;
		__m128i		lowv = _mm_loadu_si128((const __m128i *) tq->u8.tailLow);
		__m128i		highv = _mm_loadu_si128((const __m128i *) tq->u8.tailHigh);
		uint8		s0[8] = {0};
		uint8		s1[8] = {0};
		uint8		s2[8] = {0};
		uint8		s3[8] = {0};

		memcpy(s0, c0 + tailOff, tailBytes);
		memcpy(s1, c1 + tailOff, tailBytes);
		memcpy(s2, c2 + tailOff, tailBytes);
		memcpy(s3, c3 + tailOff, tailBytes);
		PgturbohybridGraphU8Madd128Avx512(PgturbohybridGraphU8Decode16CvalsAvx512(_mm_loadl_epi64((const __m128i *) s0), mask, cbU8), lowv, highv, ones128, &sumLow0, &sumHigh0);
		PgturbohybridGraphU8Madd128Avx512(PgturbohybridGraphU8Decode16CvalsAvx512(_mm_loadl_epi64((const __m128i *) s1), mask, cbU8), lowv, highv, ones128, &sumLow1, &sumHigh1);
		PgturbohybridGraphU8Madd128Avx512(PgturbohybridGraphU8Decode16CvalsAvx512(_mm_loadl_epi64((const __m128i *) s2), mask, cbU8), lowv, highv, ones128, &sumLow2, &sumHigh2);
		PgturbohybridGraphU8Madd128Avx512(PgturbohybridGraphU8Decode16CvalsAvx512(_mm_loadl_epi64((const __m128i *) s3), mask, cbU8), lowv, highv, ones128, &sumLow3, &sumHigh3);
	}

	raw[0] = PgturbohybridGraphU8Reduce512Avx512(sumLow0, sumHigh0);
	raw[1] = PgturbohybridGraphU8Reduce512Avx512(sumLow1, sumHigh1);
	raw[2] = PgturbohybridGraphU8Reduce512Avx512(sumLow2, sumHigh2);
	raw[3] = PgturbohybridGraphU8Reduce512Avx512(sumLow3, sumHigh3);
}
#endif

/*
 * Unsigned-codebook (maddubs / VPDPBUSD) 4-bit query-to-code scorer.  The raw
 * kernel returns Sum(q_signed * codebook_u8[code]); subtracting u8.bias
 * (= OFFSET * Sum(q_signed)) recovers Sum(q_signed * codebook_signed[code]),
 * after which the postprocess matches the signed path exactly.
 */
/* Shared u8 postprocess: bias removal, postprocess scale, IP/cosine/L2. */
double
PgturbohybridGraphU8DistanceFromRaw(const PgturbohybridGraphTqQuery *tq, float valueScale,
									int64 rawDot)
{
	double		dot;
	TqScoreMode mode = (TqScoreMode) tq->scoreMode;

	rawDot -= tq->u8.bias;
	dot = (double) tq->u8.postprocessScale * (double) rawDot;
	dot += tq->ecCorrection;

	/* Multiply by precomputed reciprocals (see TqPrepareQueryU8Split) instead
	 * of recomputing sqrt(dim) / sqrt(queryNorm) per node. */
	if (mode == PGTURBOHYBRID_SCORE_IP)
		return -(valueScale * dot * tq->invDimSqrt);
	if (mode == PGTURBOHYBRID_SCORE_COSINE)
	{
		if (tq->queryNorm == 0 || valueScale == 0)
			return 1;
		return 1 - (dot * tq->invQueryNormDimSqrt);
	}
	{
		double		distance = tq->queryNorm + ((double) valueScale * valueScale) -
			(2 * valueScale * dot * tq->invDimSqrt);

		return distance < 0 ? 0 : distance;
	}
}

bool
PgturbohybridGraphPackedDistanceU8Split(const PgturbohybridGraphTqQuery *tq, const uint8 *valueCode,
										float valueScale, double *distance)
{
	int64		rawDot;

	/*
	 * Single-code kernel was resolved once in TqPrepareQueryU8Split
	 * (PgturbohybridGraphTqResolveU8Kernels); no per-call feature probe.  The
	 * gate (4-bit, dim>=QUERY_SPLIT_MIN_DIM, mode!=L1, u8 enabled, SIMD available)
	 * is folded
	 * into u8.kernelSingle == NONE, so the default branch handles every fallback.
	 */
	switch (tq->u8.kernelSingle)
	{
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX512VNNI
		case PGTURBOHYBRID_U8_KERNEL_AVX512VNNI_SINGLE:
			rawDot = PgturbohybridGraphQuerySplitU8RawAvx512Vnni(tq, valueCode);
			break;
#endif
		case PGTURBOHYBRID_U8_KERNEL_AVX2_SINGLE:
			rawDot = PgturbohybridGraphQuerySplitU8RawAvx2(tq, valueCode);
			break;
		default:
			return false;
	}

	*distance = PgturbohybridGraphU8DistanceFromRaw(tq, valueScale, rawDot);
	return true;
}

/*
 * 4-candidate u8 split: one x4 raw kernel call (shared query loads) + per-code
 * postprocess.  Same gate and result as four PgturbohybridGraphPackedDistanceU8Split
 * calls; used by the batch-of-4 native scorer.
 */
bool
PgturbohybridGraphPackedDistanceU8Splitx4(const PgturbohybridGraphTqQuery *tq,
										  const uint8 *codes[4], const float scales[4],
										  double dist[4])
{
	int64		raw[4];

	/*
	 * Batch (x4) kernel resolved once in TqPrepareQueryU8Split; no per-batch
	 * feature probe.  u8.kernelBatch is NONE when the x4 batch is disabled
	 * (turbohybrid.dense_u8_batch_x4 = off) or no SIMD is available, so the
	 * default branch returns false and the caller falls back to single passes.
	 */
	switch (tq->u8.kernelBatch)
	{
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX512VNNI
		case PGTURBOHYBRID_U8_KERNEL_AVX512VNNI_X4:
			PgturbohybridGraphQuerySplitU8RawAvx512Vnnix4(tq, codes, raw);
			break;
#endif
		case PGTURBOHYBRID_U8_KERNEL_AVX2_X4:
			PgturbohybridGraphQuerySplitU8RawAvx2x4(tq, codes, raw);
			break;
		default:
			return false;
	}

	for (int c = 0; c < 4; c++)
		dist[c] = PgturbohybridGraphU8DistanceFromRaw(tq, scales[c], raw[c]);
	return true;
}

#endif							/* PGTURBOHYBRID_GRAPH_COMPILE_AVX2 */

/*
 * Resolve the exact u8 kernels for this query once, after TqPrepareQueryU8Split
 * has set up the query-split data.  Runs per scan (query prep is per scan), so
 * changing turbohybrid.dense_graph_avx512vnni / dense_u8_batch_x4 / simd between
 * scans re-selects the kernel.  The hot scoring path (PackedDistanceU8Split /
 * ...Splitx4) then switches on the result with no CPU-feature branch.
 *
 * u8.kernelSingle: best available single-code kernel.
 * u8.kernelBatch:  best available x4 kernel, but only when dense_u8_batch_x4 is on
 *                 (off -> NONE so the x4 path declines and the caller does four
 *                 single-code passes), matching the pre-resolution behaviour.
 */
void
PgturbohybridGraphTqResolveU8Kernels(PgturbohybridGraphTqQuery *tq)
{
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX2
	TqScoreMode mode = (TqScoreMode) tq->scoreMode;
	int			single = PGTURBOHYBRID_U8_KERNEL_NONE;
	int			x4 = PGTURBOHYBRID_U8_KERNEL_NONE;

	if (tq->u8.enabled && tq->dimensions >= PGTURBOHYBRID_QUERY_SPLIT_MIN_DIM &&
		tq->bits == PGTURBOHYBRID_DEFAULT_BITS && mode != PGTURBOHYBRID_SCORE_L1)
	{
#if PGTURBOHYBRID_GRAPH_COMPILE_AVX512VNNI
		if (PgturbohybridGraphAvx512VnniAvailable())
		{
			single = PGTURBOHYBRID_U8_KERNEL_AVX512VNNI_SINGLE;
			x4 = PGTURBOHYBRID_U8_KERNEL_AVX512VNNI_X4;
		}
		else
#endif
		if (PgturbohybridGraphAvx2Available())
		{
			single = PGTURBOHYBRID_U8_KERNEL_AVX2_SINGLE;
			x4 = PGTURBOHYBRID_U8_KERNEL_AVX2_X4;
		}
	}

	tq->u8.kernelSingle = single;
	tq->u8.kernelBatch = pgturbohybrid_dense_u8_batch_x4 ? x4 : PGTURBOHYBRID_U8_KERNEL_NONE;
#else
	tq->u8.kernelSingle = PGTURBOHYBRID_U8_KERNEL_NONE;
	tq->u8.kernelBatch = PGTURBOHYBRID_U8_KERNEL_NONE;
#endif
}
