/*
 * pgturbohybrid_sparse_score.c
 *
 * SoA sparse postings scorer: scalar reference + runtime dispatch to the AVX2
 * (x86) / NEON (ARM) block kernels (prompt 8).  The kernels widen a block of
 * quantized weights, multiply by a broadcast term multiplier, and scatter-add
 * into scores[] with a scalar loop (no cheap general SIMD scatter); the scalar
 * kernel here is the correctness reference the SIMD kernels must match.
 */
#include "postgres.h"

#include "pgturbohybrid_sparse.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#define PGTURBOHYBRID_SPARSE_X86 1
#else
#define PGTURBOHYBRID_SPARSE_X86 0
#endif

#if defined(__aarch64__) || defined(_M_ARM64)
#define PGTURBOHYBRID_SPARSE_ARM 1
#else
#define PGTURBOHYBRID_SPARSE_ARM 0
#endif

/*
 * Scalar reference: dequantize+scatter a SoA postings block.  Handles f32
 * (bits=0), q8 (bits=8) and q16 (bits=16); every posting is counted as a
 * scalar-tail posting.
 */
static void
PgturbohybridSparseScoreSoaScalar(const void *weights, const uint16 *offsets,
								  uint32 count, uint32 base, int bits,
								  double termMul, double *scores, uint32 nodeCount,
								  uint64 *scalarTail)
{
	for (uint32 k = 0; k < count; k++)
	{
		uint32		node = base + offsets[k];

		if (node < nodeCount)
		{
			if (bits == 0)
				scores[node] += termMul * (double) ((const float4 *) weights)[k];
			else if (bits == 16)
				scores[node] += termMul * (double) ((const uint16 *) weights)[k];
			else
				scores[node] += termMul * (double) ((const uint8 *) weights)[k];
		}
	}
	*scalarTail += count;
}

int
PgturbohybridSparseResolveScoreKernel(int bits, bool simdEnabled)
{
	if (!simdEnabled || bits == 0)
		return PGTURBOHYBRID_SPARSE_SCORE_SCALAR;	/* f32 has no SIMD kernel */
#if PGTURBOHYBRID_SPARSE_X86 && !defined(PGTURBOHYBRID_DISABLE_SIMD)
	if (__builtin_cpu_supports("avx2"))
		return PGTURBOHYBRID_SPARSE_SCORE_AVX2;
#endif
#if PGTURBOHYBRID_SPARSE_ARM && !defined(PGTURBOHYBRID_DISABLE_SIMD)
	return PGTURBOHYBRID_SPARSE_SCORE_NEON;
#endif
	return PGTURBOHYBRID_SPARSE_SCORE_SCALAR;
}

const char *
PgturbohybridSparseScoreKernelName(int kernel, int bits)
{
	if (kernel == PGTURBOHYBRID_SPARSE_SCORE_AVX2)
		return bits == 16 ? "avx2_q16" : "avx2_q8";
	if (kernel == PGTURBOHYBRID_SPARSE_SCORE_NEON)
		return bits == 16 ? "neon_q16" : "neon_q8";
	return "scalar";
}

void
PgturbohybridSparseScoreSoa(int kernel, const void *weights, const uint16 *offsets,
							uint32 count, uint32 base, int bits, double termMul,
							double *scores, uint32 nodeCount, uint64 *simdBlocks,
							uint64 *scalarTail)
{
#if PGTURBOHYBRID_SPARSE_X86 && !defined(PGTURBOHYBRID_DISABLE_SIMD)
	if (kernel == PGTURBOHYBRID_SPARSE_SCORE_AVX2 && bits == 8)
	{
		PgturbohybridSparseScoreSoaAvx2Q8((const uint8 *) weights, offsets, count,
										  base, termMul, scores, nodeCount,
										  simdBlocks, scalarTail);
		return;
	}
	if (kernel == PGTURBOHYBRID_SPARSE_SCORE_AVX2 && bits == 16)
	{
		PgturbohybridSparseScoreSoaAvx2Q16((const uint16 *) weights, offsets, count,
										   base, termMul, scores, nodeCount,
										   simdBlocks, scalarTail);
		return;
	}
#endif
#if PGTURBOHYBRID_SPARSE_ARM && !defined(PGTURBOHYBRID_DISABLE_SIMD)
	if (kernel == PGTURBOHYBRID_SPARSE_SCORE_NEON && bits == 8)
	{
		PgturbohybridSparseScoreSoaNeonQ8((const uint8 *) weights, offsets, count,
										  base, termMul, scores, nodeCount,
										  simdBlocks, scalarTail);
		return;
	}
	if (kernel == PGTURBOHYBRID_SPARSE_SCORE_NEON && bits == 16)
	{
		PgturbohybridSparseScoreSoaNeonQ16((const uint16 *) weights, offsets, count,
										   base, termMul, scores, nodeCount,
										   simdBlocks, scalarTail);
		return;
	}
#endif
	PgturbohybridSparseScoreSoaScalar(weights, offsets, count, base, bits,
									  termMul, scores, nodeCount, scalarTail);
}
