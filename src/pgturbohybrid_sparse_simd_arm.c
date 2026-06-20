/*
 * pgturbohybrid_sparse_simd_arm.c
 *
 * ARM NEON SoA sparse postings scorers for q8/q16 (prompt 8).  Mirrors the AVX2
 * kernels: widen a block of 8 quantized weights to float, multiply by the
 * broadcast term multiplier, then scatter-add scores[base + offset] += contrib
 * with a scalar loop.  The scalar kernel in pgturbohybrid_sparse_score.c is the
 * correctness reference.  This file is empty on non-ARM builds.
 */
#include "postgres.h"

#include "pgturbohybrid_sparse.h"

#if (defined(__aarch64__) || defined(_M_ARM64)) && !defined(PGTURBOHYBRID_DISABLE_SIMD)
#include <arm_neon.h>

#define PGTURBOHYBRID_SPARSE_NEON_BLOCK 8

static inline void
PgturbohybridSparseScatterBlockNeon(const float *contrib, const uint16 *offsets,
									uint32 base, double *scores, uint32 nodeCount)
{
	for (int k = 0; k < PGTURBOHYBRID_SPARSE_NEON_BLOCK; k++)
	{
		uint32		node = base + offsets[k];

		if (node < nodeCount)
			scores[node] += (double) contrib[k];
	}
}

void
PgturbohybridSparseScoreSoaNeonQ8(const uint8 *weights, const uint16 *offsets,
								  uint32 count, uint32 base, double termMul,
								  double *scores, uint32 nodeCount,
								  uint64 *simdBlocks, uint64 *scalarTail)
{
	float		tm = (float) termMul;
	uint32		i = 0;

	for (; i + PGTURBOHYBRID_SPARSE_NEON_BLOCK <= count;
		 i += PGTURBOHYBRID_SPARSE_NEON_BLOCK)
	{
		uint16x8_t	w16 = vmovl_u8(vld1_u8(weights + i));
		float32x4_t flo = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16))), tm);
		float32x4_t fhi = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16))), tm);
		float		contrib[PGTURBOHYBRID_SPARSE_NEON_BLOCK];

		vst1q_f32(contrib, flo);
		vst1q_f32(contrib + 4, fhi);
		PgturbohybridSparseScatterBlockNeon(contrib, offsets + i, base, scores,
											nodeCount);
		(*simdBlocks)++;
	}
	for (; i < count; i++)
	{
		uint32		node = base + offsets[i];

		if (node < nodeCount)
			scores[node] += termMul * (double) weights[i];
		(*scalarTail)++;
	}
}

void
PgturbohybridSparseScoreSoaNeonQ16(const uint16 *weights, const uint16 *offsets,
								   uint32 count, uint32 base, double termMul,
								   double *scores, uint32 nodeCount,
								   uint64 *simdBlocks, uint64 *scalarTail)
{
	float		tm = (float) termMul;
	uint32		i = 0;

	for (; i + PGTURBOHYBRID_SPARSE_NEON_BLOCK <= count;
		 i += PGTURBOHYBRID_SPARSE_NEON_BLOCK)
	{
		uint16x8_t	w16 = vld1q_u16(weights + i);
		float32x4_t flo = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(w16))), tm);
		float32x4_t fhi = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(w16))), tm);
		float		contrib[PGTURBOHYBRID_SPARSE_NEON_BLOCK];

		vst1q_f32(contrib, flo);
		vst1q_f32(contrib + 4, fhi);
		PgturbohybridSparseScatterBlockNeon(contrib, offsets + i, base, scores,
											nodeCount);
		(*simdBlocks)++;
	}
	for (; i < count; i++)
	{
		uint32		node = base + offsets[i];

		if (node < nodeCount)
			scores[node] += termMul * (double) weights[i];
		(*scalarTail)++;
	}
}

#elif (defined(__aarch64__) || defined(_M_ARM64))

/* SIMD disabled at compile time on ARM: scalar fallbacks so symbols exist. */
void
PgturbohybridSparseScoreSoaNeonQ8(const uint8 *weights, const uint16 *offsets,
								  uint32 count, uint32 base, double termMul,
								  double *scores, uint32 nodeCount,
								  uint64 *simdBlocks, uint64 *scalarTail)
{
	(void) simdBlocks;
	for (uint32 i = 0; i < count; i++)
	{
		uint32		node = base + offsets[i];

		if (node < nodeCount)
			scores[node] += termMul * (double) weights[i];
		(*scalarTail)++;
	}
}

void
PgturbohybridSparseScoreSoaNeonQ16(const uint16 *weights, const uint16 *offsets,
								   uint32 count, uint32 base, double termMul,
								   double *scores, uint32 nodeCount,
								   uint64 *simdBlocks, uint64 *scalarTail)
{
	(void) simdBlocks;
	for (uint32 i = 0; i < count; i++)
	{
		uint32		node = base + offsets[i];

		if (node < nodeCount)
			scores[node] += termMul * (double) weights[i];
		(*scalarTail)++;
	}
}

#endif
