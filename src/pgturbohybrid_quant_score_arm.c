/*
 * pgturbohybrid_quant_score_arm.c
 *
 * ARM NEON/SDOT/I8MM dense quantized-scoring kernels, extracted verbatim from
 * pgturbohybrid_quant_score.c (no behaviour change): the NEON dot-product
 * availability probes, the signed 4-bit/2-bit query-split scorers, and the
 * code-code / weighted code-code raw scorers.  The generic scan-time and
 * build-time dispatch stays in pgturbohybrid_quant_score.c and calls the
 * non-static entry points declared in pgturbohybrid_quant_score_internal.h.
 *
 * On non-ARM targets PGTURBOHYBRID_GRAPH_COMPILE_ARM_DOT is 0, so the whole body
 * compiles to an empty object.  The compile/target guard macros below are kept
 * byte-for-byte in sync with the copies in pgturbohybrid_quant_score.c.
 */
#include "postgres.h"

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

#define PGTURBOHYBRID_QUERY_SPLIT_HIGH_COEF 256


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
static const int8 PgturbohybridGraphCodebookI8[PGTURBOHYBRID_LUT_WIDTH] = {
	-127, -96, -75, -58, -44, -31, -18, -6,
	6, 18, 31, 44, 58, 75, 96, 127
};
static const int8 PgturbohybridGraphCodebook2I8[PGTURBOHYBRID_LUT_WIDTH] = {
	-127, -38, 38, 127, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0
};
#else
#define PGTURBOHYBRID_GRAPH_COMPILE_ARM_DOT 0
#define PGTURBOHYBRID_GRAPH_COMPILE_ARM_I8MM 0
#endif

#include "pgturbohybrid_quant_score.h"
#include "pgturbohybrid_quant_score_internal.h"

#if PGTURBOHYBRID_GRAPH_COMPILE_ARM_DOT
bool
PgturbohybridGraphArmDotprodAvailable(void)
{
	static int	available = -1;

	if (pgturbohybrid_dense_simd_force != PGTURBOHYBRID_SIMD_FORCE_AUTO &&
		pgturbohybrid_dense_simd_force != PGTURBOHYBRID_SIMD_FORCE_ARM_SDOT &&
		pgturbohybrid_dense_simd_force != PGTURBOHYBRID_SIMD_FORCE_NEON)
		return false;

	if (available >= 0)
		return available != 0;

#if defined(__APPLE__)
	{
		int			value = 0;
		size_t		len = sizeof(value);

		if (sysctlbyname("hw.optional.arm.FEAT_DotProd", &value, &len,
						 NULL, 0) == 0)
		{
			available = value != 0;
			return available != 0;
		}
	}
	available = 1;
#elif defined(__linux__) && defined(HWCAP_ASIMDDP)
	available = (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#else
	available = 0;
#endif
	return available != 0;
}

#if PGTURBOHYBRID_GRAPH_COMPILE_ARM_I8MM
static bool
PgturbohybridGraphArmI8mmAvailable(void)
{
	static int	available = -1;

	if (!pgturbohybrid_dense_graph_i8mm)
		return false;
	if (pgturbohybrid_dense_simd_force != PGTURBOHYBRID_SIMD_FORCE_AUTO &&
		pgturbohybrid_dense_simd_force != PGTURBOHYBRID_SIMD_FORCE_ARM_I8MM)
		return false;

	if (available >= 0)
		return available != 0;

#if defined(__APPLE__)
	{
		int			value = 0;
		size_t		len = sizeof(value);

		if (sysctlbyname("hw.optional.arm.FEAT_I8MM", &value, &len,
						 NULL, 0) == 0)
		{
			available = value != 0;
			return available != 0;
		}
	}
	available = 0;
#elif defined(__linux__) && defined(HWCAP2_I8MM)
	available = (getauxval(AT_HWCAP2) & HWCAP2_I8MM) != 0;
#else
	available = 0;
#endif
	return available != 0;
}

static int32x4_t PGTURBOHYBRID_GRAPH_ARM_I8MM_TARGET
PgturbohybridGraphDotI8x16ArmI8mm(int32x4_t acc, int8x16_t a, int8x16_t b)
{
	uint8x16_t	aUnsigned = vreinterpretq_u8_s8(veorq_s8(a, vdupq_n_s8((int8) 0x80)));
	int16x8_t	pairSums = vpaddlq_s8(b);
	int32x4_t	groupSums = vpaddlq_s16(pairSums);

	acc = vusdotq_s32(acc, aUnsigned, b);
	return vsubq_s32(acc, vmulq_n_s32(groupSums, 128));
}
#define PGTURBOHYBRID_GRAPH_ARM_DOT(acc, a, b) \
	(useI8mm ? PgturbohybridGraphDotI8x16ArmI8mm((acc), (a), (b)) : vdotq_s32((acc), (a), (b)))
#else
#define PGTURBOHYBRID_GRAPH_ARM_DOT(acc, a, b) vdotq_s32((acc), (a), (b))
#endif

int64 PGTURBOHYBRID_GRAPH_ARM_DOT_TARGET
PgturbohybridGraphQuerySplitRawNeonSdot(const PgturbohybridGraphTqQuery *tq, const uint8 *code)
{
	int32x4_t	accLow0 = vdupq_n_s32(0);
	int32x4_t	accLow1 = vdupq_n_s32(0);
	int32x4_t	accHigh0 = vdupq_n_s32(0);
	int32x4_t	accHigh1 = vdupq_n_s32(0);
	uint8x16_t	mask = vdupq_n_u8(0x0f);
	int8x16_t	codebook = vld1q_s8((const int8_t *) PgturbohybridGraphCodebookI8);
	int			pairs = tq->querySplitChunks / 2;
#if PGTURBOHYBRID_GRAPH_COMPILE_ARM_I8MM
	bool		useI8mm = PgturbohybridGraphArmI8mmAvailable();
#endif

	for (int pair = 0; pair < pairs; pair++)
	{
		uint8x16_t packed = vld1q_u8(code + pair * 16);
		uint8x16_t loNibbles = vandq_u8(packed, mask);
		uint8x16_t hiNibbles = vshrq_n_u8(packed, 4);
		uint8x16_t idx0 = vzip1q_u8(loNibbles, hiNibbles);
		uint8x16_t idx1 = vzip2q_u8(loNibbles, hiNibbles);
		int8x16_t	c0 = vqtbl1q_s8(codebook, idx0);
		int8x16_t	c1 = vqtbl1q_s8(codebook, idx1);
		const int8 *low = tq->signedSplit.low + pair * 32;
		const int8 *high = tq->signedSplit.high + pair * 32;

		accLow0 = PGTURBOHYBRID_GRAPH_ARM_DOT(accLow0, vld1q_s8((const int8_t *) low), c0);
		accLow1 = PGTURBOHYBRID_GRAPH_ARM_DOT(accLow1, vld1q_s8((const int8_t *) (low + 16)), c1);
		accHigh0 = PGTURBOHYBRID_GRAPH_ARM_DOT(accHigh0, vld1q_s8((const int8_t *) high), c0);
		accHigh1 = PGTURBOHYBRID_GRAPH_ARM_DOT(accHigh1, vld1q_s8((const int8_t *) (high + 16)), c1);
	}

	if ((tq->querySplitChunks & 1) != 0)
	{
		int			chunk = tq->querySplitChunks - 1;
		uint8x8_t	packed = vld1_u8(code + chunk * 8);
		uint8x8_t	loNibbles = vand_u8(packed, vdup_n_u8(0x0f));
		uint8x8_t	hiNibbles = vshr_n_u8(packed, 4);
		uint8x16_t	idx = vcombine_u8(vzip1_u8(loNibbles, hiNibbles),
									   vzip2_u8(loNibbles, hiNibbles));
		int8x16_t	c = vqtbl1q_s8(codebook, idx);
		const int8 *low = tq->signedSplit.low + chunk * 16;
		const int8 *high = tq->signedSplit.high + chunk * 16;

		accLow0 = PGTURBOHYBRID_GRAPH_ARM_DOT(accLow0, vld1q_s8((const int8_t *) low), c);
		accHigh0 = PGTURBOHYBRID_GRAPH_ARM_DOT(accHigh0, vld1q_s8((const int8_t *) high), c);
	}

	if (tq->querySplitTailDims != 0)
	{
		uint8		scratch[8] = {0};
		int			tailBytes = (tq->querySplitTailDims + 1) / 2;
		const uint8 *tail = code + tq->querySplitChunks * 8;
		uint8x8_t	packed;
		uint8x8_t	loNibbles;
		uint8x8_t	hiNibbles;
		uint8x16_t	idx;
		int8x16_t	c;

		memcpy(scratch, tail, tailBytes);
		packed = vld1_u8(scratch);
		loNibbles = vand_u8(packed, vdup_n_u8(0x0f));
		hiNibbles = vshr_n_u8(packed, 4);
		idx = vcombine_u8(vzip1_u8(loNibbles, hiNibbles),
						  vzip2_u8(loNibbles, hiNibbles));
		c = vqtbl1q_s8(codebook, idx);

		accLow0 = PGTURBOHYBRID_GRAPH_ARM_DOT(accLow0,
								   vld1q_s8((const int8_t *) tq->signedSplit.tailLow), c);
		accHigh0 = PGTURBOHYBRID_GRAPH_ARM_DOT(accHigh0,
									vld1q_s8((const int8_t *) tq->signedSplit.tailHigh), c);
	}

	accLow0 = vaddq_s32(accLow0, accLow1);
	accHigh0 = vaddq_s32(accHigh0, accHigh1);
	return (int64) vaddvq_s32(accLow0) +
		(int64) PGTURBOHYBRID_QUERY_SPLIT_HIGH_COEF * (int64) vaddvq_s32(accHigh0);
}

int64 PGTURBOHYBRID_GRAPH_ARM_DOT_TARGET
PgturbohybridGraphQuerySplit2RawNeonSdot(const PgturbohybridGraphTqQuery *tq, const uint8 *code)
{
	int32x4_t	accLow = vdupq_n_s32(0);
	int32x4_t	accHigh = vdupq_n_s32(0);
	int8x16_t	codebook = vld1q_s8((const int8_t *) PgturbohybridGraphCodebook2I8);
	uint8x16_t	mask = vdupq_n_u8(0x03);
	int8x16_t	shifts = {
		0, -2, -4, -6, 0, -2, -4, -6,
		0, -2, -4, -6, 0, -2, -4, -6
	};
#if PGTURBOHYBRID_GRAPH_COMPILE_ARM_I8MM
	bool		useI8mm = PgturbohybridGraphArmI8mmAvailable();
#endif

	for (int chunk = 0; chunk < tq->querySplitChunks; chunk++)
	{
		uint8		repeatedBytes[16];
		const uint8 *bytes = code + chunk * 4;
		uint8x16_t	repeated;
		uint8x16_t	idx;
		int8x16_t	c;
		const int8 *low = tq->signedSplit.low + chunk * 16;
		const int8 *high = tq->signedSplit.high + chunk * 16;

		for (int i = 0; i < 16; i++)
			repeatedBytes[i] = bytes[i / 4];

		repeated = vld1q_u8(repeatedBytes);
		idx = vandq_u8(vshlq_u8(repeated, shifts), mask);
		c = vqtbl1q_s8(codebook, idx);

		accLow = PGTURBOHYBRID_GRAPH_ARM_DOT(accLow, vld1q_s8((const int8_t *) low), c);
		accHigh = PGTURBOHYBRID_GRAPH_ARM_DOT(accHigh, vld1q_s8((const int8_t *) high), c);
	}

	if (tq->querySplitTailDims != 0)
	{
		uint8		repeatedBytes[16] = {0};
		const uint8 *bytes = code + tq->querySplitChunks * 4;
		uint8x16_t	repeated;
		uint8x16_t	idx;
		int8x16_t	c;

		for (int i = 0; i < tq->querySplitTailDims; i++)
			repeatedBytes[i] = bytes[i / 4];

		repeated = vld1q_u8(repeatedBytes);
		idx = vandq_u8(vshlq_u8(repeated, shifts), mask);
		c = vqtbl1q_s8(codebook, idx);

		accLow = PGTURBOHYBRID_GRAPH_ARM_DOT(accLow,
								  vld1q_s8((const int8_t *) tq->signedSplit.tailLow), c);
		accHigh = PGTURBOHYBRID_GRAPH_ARM_DOT(accHigh,
								   vld1q_s8((const int8_t *) tq->signedSplit.tailHigh), c);
	}

	return (int64) vaddvq_s32(accLow) +
		(int64) PGTURBOHYBRID_QUERY_SPLIT_HIGH_COEF * (int64) vaddvq_s32(accHigh);
}

int64 PGTURBOHYBRID_GRAPH_ARM_DOT_TARGET
PgturbohybridGraphCodeCodeRawNeonSdot(const uint8 *a, const uint8 *b, int dim,
						   int *sampleDims)
{
	int32x4_t	acc0 = vdupq_n_s32(0);
	int32x4_t	acc1 = vdupq_n_s32(0);
	int8x16_t	codebook = vld1q_s8((const int8_t *) PgturbohybridGraphCodebookI8);
	int			chunks = dim / 16;
	int			tailDims = dim - chunks * 16;
	int			scoredChunks = chunks;
#if PGTURBOHYBRID_GRAPH_COMPILE_ARM_I8MM
	bool		useI8mm = PgturbohybridGraphArmI8mmAvailable();
#endif

	if (chunks > PGTURBOHYBRID_GRAPH_CODE_CODE_PRUNE_CHUNKS)
		scoredChunks = PGTURBOHYBRID_GRAPH_CODE_CODE_PRUNE_CHUNKS;
	*sampleDims = scoredChunks * 16;

	for (int scored = 0; scored < scoredChunks; scored++)
	{
		int			chunk = scoredChunks == chunks ? scored :
			(int) (((int64) scored * chunks) / scoredChunks);
		uint8x8_t	pa = vld1_u8(a + chunk * 8);
		uint8x8_t	pb = vld1_u8(b + chunk * 8);
		uint8x8_t	loA = vand_u8(pa, vdup_n_u8(0x0f));
		uint8x8_t	hiA = vshr_n_u8(pa, 4);
		uint8x8_t	loB = vand_u8(pb, vdup_n_u8(0x0f));
		uint8x8_t	hiB = vshr_n_u8(pb, 4);
		uint8x16_t	idxA = vcombine_u8(vzip1_u8(loA, hiA),
										vzip2_u8(loA, hiA));
		uint8x16_t	idxB = vcombine_u8(vzip1_u8(loB, hiB),
										vzip2_u8(loB, hiB));
		int8x16_t	ca = vqtbl1q_s8(codebook, idxA);
		int8x16_t	cb = vqtbl1q_s8(codebook, idxB);

		acc0 = PGTURBOHYBRID_GRAPH_ARM_DOT(acc0, ca, cb);
	}

	if (scoredChunks == chunks && tailDims != 0)
	{
		uint8		scratchA[8] = {0};
		uint8		scratchB[8] = {0};
		int			tailBytes = (tailDims + 1) / 2;
		uint8x8_t	pa;
		uint8x8_t	pb;
		uint8x8_t	loA;
		uint8x8_t	hiA;
		uint8x8_t	loB;
		uint8x8_t	hiB;
		uint8x16_t	idxA;
		uint8x16_t	idxB;
		int8x16_t	ca;
		int8x16_t	cb;

		memcpy(scratchA, a + chunks * 8, tailBytes);
		memcpy(scratchB, b + chunks * 8, tailBytes);
		pa = vld1_u8(scratchA);
		pb = vld1_u8(scratchB);
		loA = vand_u8(pa, vdup_n_u8(0x0f));
		hiA = vshr_n_u8(pa, 4);
		loB = vand_u8(pb, vdup_n_u8(0x0f));
		hiB = vshr_n_u8(pb, 4);
		idxA = vcombine_u8(vzip1_u8(loA, hiA), vzip2_u8(loA, hiA));
		idxB = vcombine_u8(vzip1_u8(loB, hiB), vzip2_u8(loB, hiB));
		ca = vqtbl1q_s8(codebook, idxA);
		cb = vqtbl1q_s8(codebook, idxB);
		acc1 = PGTURBOHYBRID_GRAPH_ARM_DOT(acc1, ca, cb);
		*sampleDims += tailDims;
	}

	return (int64) vaddvq_s32(acc0) + (int64) vaddvq_s32(acc1);
}

int64 PGTURBOHYBRID_GRAPH_ARM_DOT_TARGET
PgturbohybridGraphCodeCode2RawNeonSdot(const uint8 *a, const uint8 *b, int dim,
							int *sampleDims)
{
	int32x4_t	acc0 = vdupq_n_s32(0);
	int32x4_t	acc1 = vdupq_n_s32(0);
	int8x16_t	codebook = vld1q_s8((const int8_t *) PgturbohybridGraphCodebook2I8);
	uint8x16_t	mask = vdupq_n_u8(0x03);
	int8x16_t	shifts = {
		0, -2, -4, -6, 0, -2, -4, -6,
		0, -2, -4, -6, 0, -2, -4, -6
	};
	int			chunks = dim / 16;
	int			tailDims = dim - chunks * 16;
	int			scoredChunks = chunks;
#if PGTURBOHYBRID_GRAPH_COMPILE_ARM_I8MM
	bool		useI8mm = PgturbohybridGraphArmI8mmAvailable();
#endif

	if (chunks > PGTURBOHYBRID_GRAPH_CODE_CODE_PRUNE_CHUNKS)
		scoredChunks = PGTURBOHYBRID_GRAPH_CODE_CODE_PRUNE_CHUNKS;
	*sampleDims = scoredChunks * 16;

	for (int scored = 0; scored < scoredChunks; scored++)
	{
		int			chunk = scoredChunks == chunks ? scored :
			(int) (((int64) scored * chunks) / scoredChunks);
		uint8		repeatedA[16];
		uint8		repeatedB[16];
		const uint8 *bytesA = a + chunk * 4;
		const uint8 *bytesB = b + chunk * 4;
		uint8x16_t	idxA;
		uint8x16_t	idxB;
		int8x16_t	ca;
		int8x16_t	cb;

		for (int i = 0; i < 16; i++)
		{
			repeatedA[i] = bytesA[i / 4];
			repeatedB[i] = bytesB[i / 4];
		}

		idxA = vandq_u8(vshlq_u8(vld1q_u8(repeatedA), shifts), mask);
		idxB = vandq_u8(vshlq_u8(vld1q_u8(repeatedB), shifts), mask);
		ca = vqtbl1q_s8(codebook, idxA);
		cb = vqtbl1q_s8(codebook, idxB);
		acc0 = PGTURBOHYBRID_GRAPH_ARM_DOT(acc0, ca, cb);
	}

	if (scoredChunks == chunks && tailDims != 0)
	{
		uint8		repeatedA[16] = {0};
		uint8		repeatedB[16] = {0};
		const uint8 *bytesA = a + chunks * 4;
		const uint8 *bytesB = b + chunks * 4;
		uint8x16_t	idxA;
		uint8x16_t	idxB;
		int8x16_t	ca;
		int8x16_t	cb;

		for (int i = 0; i < tailDims; i++)
		{
			repeatedA[i] = bytesA[i / 4];
			repeatedB[i] = bytesB[i / 4];
		}

		idxA = vandq_u8(vshlq_u8(vld1q_u8(repeatedA), shifts), mask);
		idxB = vandq_u8(vshlq_u8(vld1q_u8(repeatedB), shifts), mask);
		ca = vqtbl1q_s8(codebook, idxA);
		cb = vqtbl1q_s8(codebook, idxB);
		acc1 = PGTURBOHYBRID_GRAPH_ARM_DOT(acc1, ca, cb);
		*sampleDims += tailDims;
	}

	return (int64) vaddvq_s32(acc0) + (int64) vaddvq_s32(acc1);
}

/*
 * NEON weighted symmetric (code-code) kernels for TQ+.
 *
 * Mirrors PgturbohybridGraphCodeCodeWeightedRawAvx2 in math: per chunk of 16
 * coords (4-bit unpack via vqtbl1q_s8 codebook lookup), widen to
 * i16x16, multiply pairwise, multiply by per-coord i16 weight via
 * vmlal_s16 (i16×i16 → i32 multiply-accumulate), widen i32 → i64
 * via vpadalq_s32, accumulate.
 *
 * Returns Σ c_a · c_b · D'²_i16 as i64.  Caller divides by
 * `weight_scale · CODEBOOK_SCALE²` to recover the f32 weighted dot.
 *
 * Uses base NEON only (no SDOT or I8MM); compiles wherever the file
 * already targets aarch64.
 */
static inline int64x2_t PGTURBOHYBRID_GRAPH_ARM_DOT_TARGET
PgturbohybridGraphWeightedDotI8x16NeonSdot(int8x16_t ca, int8x16_t cb,
								const int16 *weightsAt)
{
	int16x8_t	caLo = vmovl_s8(vget_low_s8(ca));
	int16x8_t	caHi = vmovl_s8(vget_high_s8(ca));
	int16x8_t	cbLo = vmovl_s8(vget_low_s8(cb));
	int16x8_t	cbHi = vmovl_s8(vget_high_s8(cb));
	int16x8_t	prodLo = vmulq_s16(caLo, cbLo);
	int16x8_t	prodHi = vmulq_s16(caHi, cbHi);
	int16x8_t	wLo = vld1q_s16(weightsAt);
	int16x8_t	wHi = vld1q_s16(weightsAt + 8);
	int32x4_t	chunkAcc = vdupq_n_s32(0);

	chunkAcc = vmlal_s16(chunkAcc, vget_low_s16(prodLo), vget_low_s16(wLo));
	chunkAcc = vmlal_s16(chunkAcc, vget_high_s16(prodLo), vget_high_s16(wLo));
	chunkAcc = vmlal_s16(chunkAcc, vget_low_s16(prodHi), vget_low_s16(wHi));
	chunkAcc = vmlal_s16(chunkAcc, vget_high_s16(prodHi), vget_high_s16(wHi));

	return vpadalq_s32(vdupq_n_s64(0), chunkAcc);
}

int64 PGTURBOHYBRID_GRAPH_ARM_DOT_TARGET
PgturbohybridGraphCodeCodeWeightedRawNeonSdot(const uint8 *a, const uint8 *b,
								   const int16 *weights, int dim)
{
	int64x2_t	acc = vdupq_n_s64(0);
	int8x16_t	codebook = vld1q_s8((const int8_t *) PgturbohybridGraphCodebookI8);
	int			chunks = dim / 16;
	int			tailDims = dim - chunks * 16;

	for (int chunk = 0; chunk < chunks; chunk++)
	{
		uint8x8_t	pa = vld1_u8(a + chunk * 8);
		uint8x8_t	pb = vld1_u8(b + chunk * 8);
		uint8x8_t	loA = vand_u8(pa, vdup_n_u8(0x0f));
		uint8x8_t	hiA = vshr_n_u8(pa, 4);
		uint8x8_t	loB = vand_u8(pb, vdup_n_u8(0x0f));
		uint8x8_t	hiB = vshr_n_u8(pb, 4);
		uint8x16_t	idxA = vcombine_u8(vzip1_u8(loA, hiA),
										vzip2_u8(loA, hiA));
		uint8x16_t	idxB = vcombine_u8(vzip1_u8(loB, hiB),
										vzip2_u8(loB, hiB));
		int8x16_t	ca = vqtbl1q_s8(codebook, idxA);
		int8x16_t	cb = vqtbl1q_s8(codebook, idxB);

		acc = vaddq_s64(acc,
						 PgturbohybridGraphWeightedDotI8x16NeonSdot(ca, cb, weights + chunk * 16));
	}

	if (tailDims != 0)
	{
		uint8		scratchA[8] = {0};
		uint8		scratchB[8] = {0};
		int16		scratchW[16] = {0};
		int			tailBytes = (tailDims + 1) / 2;
		uint8x8_t	pa;
		uint8x8_t	pb;
		uint8x8_t	loA;
		uint8x8_t	hiA;
		uint8x8_t	loB;
		uint8x8_t	hiB;
		uint8x16_t	idxA;
		uint8x16_t	idxB;
		int8x16_t	ca;
		int8x16_t	cb;

		memcpy(scratchA, a + chunks * 8, tailBytes);
		memcpy(scratchB, b + chunks * 8, tailBytes);
		memcpy(scratchW, weights + chunks * 16, sizeof(int16) * tailDims);
		pa = vld1_u8(scratchA);
		pb = vld1_u8(scratchB);
		loA = vand_u8(pa, vdup_n_u8(0x0f));
		hiA = vshr_n_u8(pa, 4);
		loB = vand_u8(pb, vdup_n_u8(0x0f));
		hiB = vshr_n_u8(pb, 4);
		idxA = vcombine_u8(vzip1_u8(loA, hiA), vzip2_u8(loA, hiA));
		idxB = vcombine_u8(vzip1_u8(loB, hiB), vzip2_u8(loB, hiB));
		ca = vqtbl1q_s8(codebook, idxA);
		cb = vqtbl1q_s8(codebook, idxB);
		acc = vaddq_s64(acc, PgturbohybridGraphWeightedDotI8x16NeonSdot(ca, cb, scratchW));
	}

	return vaddvq_s64(acc);
}

int64 PGTURBOHYBRID_GRAPH_ARM_DOT_TARGET
PgturbohybridGraphCodeCode2WeightedRawNeonSdot(const uint8 *a, const uint8 *b,
									const int16 *weights, int dim)
{
	int64x2_t	acc = vdupq_n_s64(0);
	int8x16_t	codebook = vld1q_s8((const int8_t *) PgturbohybridGraphCodebook2I8);
	uint8x16_t	mask = vdupq_n_u8(0x03);
	int8x16_t	shifts = {
		0, -2, -4, -6, 0, -2, -4, -6,
		0, -2, -4, -6, 0, -2, -4, -6
	};
	int			chunks = dim / 16;
	int			tailDims = dim - chunks * 16;

	for (int chunk = 0; chunk < chunks; chunk++)
	{
		uint8		repeatedA[16];
		uint8		repeatedB[16];
		const uint8 *bytesA = a + chunk * 4;
		const uint8 *bytesB = b + chunk * 4;
		uint8x16_t	idxA;
		uint8x16_t	idxB;
		int8x16_t	ca;
		int8x16_t	cb;

		for (int i = 0; i < 16; i++)
		{
			repeatedA[i] = bytesA[i / 4];
			repeatedB[i] = bytesB[i / 4];
		}

		idxA = vandq_u8(vshlq_u8(vld1q_u8(repeatedA), shifts), mask);
		idxB = vandq_u8(vshlq_u8(vld1q_u8(repeatedB), shifts), mask);
		ca = vqtbl1q_s8(codebook, idxA);
		cb = vqtbl1q_s8(codebook, idxB);

		acc = vaddq_s64(acc,
						 PgturbohybridGraphWeightedDotI8x16NeonSdot(ca, cb, weights + chunk * 16));
	}

	if (tailDims != 0)
	{
		uint8		repeatedA[16] = {0};
		uint8		repeatedB[16] = {0};
		int16		scratchW[16] = {0};
		const uint8 *bytesA = a + chunks * 4;
		const uint8 *bytesB = b + chunks * 4;
		uint8x16_t	idxA;
		uint8x16_t	idxB;
		int8x16_t	ca;
		int8x16_t	cb;

		for (int i = 0; i < tailDims; i++)
		{
			repeatedA[i] = bytesA[i / 4];
			repeatedB[i] = bytesB[i / 4];
		}
		memcpy(scratchW, weights + chunks * 16, sizeof(int16) * tailDims);

		idxA = vandq_u8(vshlq_u8(vld1q_u8(repeatedA), shifts), mask);
		idxB = vandq_u8(vshlq_u8(vld1q_u8(repeatedB), shifts), mask);
		ca = vqtbl1q_s8(codebook, idxA);
		cb = vqtbl1q_s8(codebook, idxB);
		acc = vaddq_s64(acc, PgturbohybridGraphWeightedDotI8x16NeonSdot(ca, cb, scratchW));
	}

	return vaddvq_s64(acc);
}

#endif
