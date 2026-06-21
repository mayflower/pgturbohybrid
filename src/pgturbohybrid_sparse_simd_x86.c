/*
 * pgturbohybrid_sparse_simd_x86.c
 *
 * AVX2 SoA sparse postings scorers for q8/q16.  Each block widens 8
 * quantized weights to float, multiplies by the broadcast term multiplier, and
 * stores the contributions to a small stack array; a scalar loop then scatters
 * scores[base + offset] += contribution (AVX2 has no cheap general scatter).
 * The scalar kernel in pgturbohybrid_sparse_score.c is the correctness reference.
 */
#include "postgres.h"

#include "pgturbohybrid_sparse.h"

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
#define PGTURBOHYBRID_SPARSE_X86 1
#else
#define PGTURBOHYBRID_SPARSE_X86 0
#endif

#if PGTURBOHYBRID_SPARSE_X86 && !defined(PGTURBOHYBRID_DISABLE_SIMD) && \
	(defined(__GNUC__) || defined(__clang__))
#define PGTURBOHYBRID_SPARSE_COMPILE_AVX2 1
#else
#define PGTURBOHYBRID_SPARSE_COMPILE_AVX2 0
#endif

#if PGTURBOHYBRID_SPARSE_COMPILE_AVX2 && !defined(__AVX2__)
#define PGTURBOHYBRID_SPARSE_AVX2_TARGET __attribute__((target("avx2")))
#else
#define PGTURBOHYBRID_SPARSE_AVX2_TARGET
#endif

#if PGTURBOHYBRID_SPARSE_COMPILE_AVX2
#include <immintrin.h>

#define PGTURBOHYBRID_SPARSE_AVX2_BLOCK 8

/* Scatter-add a block of contributions: scores[base + offsets[k]] += contrib[k]. */
static inline void
PgturbohybridSparseScatterBlock(const float *contrib, const uint16 *offsets,
								uint32 base, double *scores, uint32 nodeCount)
{
	for (int k = 0; k < PGTURBOHYBRID_SPARSE_AVX2_BLOCK; k++)
	{
		uint32		node = base + offsets[k];

		if (node < nodeCount)
			scores[node] += (double) contrib[k];
	}
}

PGTURBOHYBRID_SPARSE_AVX2_TARGET void
PgturbohybridSparseScoreSoaAvx2Q8(const uint8 *weights, const uint16 *offsets,
								  uint32 count, uint32 base, double termMul,
								  double *scores, uint32 nodeCount,
								  uint64 *simdBlocks, uint64 *scalarTail)
{
	float		tm = (float) termMul;
	__m256		tmv = _mm256_set1_ps(tm);
	uint32		i = 0;

	for (; i + PGTURBOHYBRID_SPARSE_AVX2_BLOCK <= count;
		 i += PGTURBOHYBRID_SPARSE_AVX2_BLOCK)
	{
		__m128i		b = _mm_loadl_epi64((const __m128i *) (weights + i));
		__m256i		wi = _mm256_cvtepu8_epi32(b);
		__m256		cf = _mm256_mul_ps(_mm256_cvtepi32_ps(wi), tmv);
		float		contrib[PGTURBOHYBRID_SPARSE_AVX2_BLOCK];

		_mm256_storeu_ps(contrib, cf);
		PgturbohybridSparseScatterBlock(contrib, offsets + i, base, scores,
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

PGTURBOHYBRID_SPARSE_AVX2_TARGET void
PgturbohybridSparseScoreSoaAvx2Q16(const uint16 *weights, const uint16 *offsets,
								   uint32 count, uint32 base, double termMul,
								   double *scores, uint32 nodeCount,
								   uint64 *simdBlocks, uint64 *scalarTail)
{
	float		tm = (float) termMul;
	__m256		tmv = _mm256_set1_ps(tm);
	uint32		i = 0;

	for (; i + PGTURBOHYBRID_SPARSE_AVX2_BLOCK <= count;
		 i += PGTURBOHYBRID_SPARSE_AVX2_BLOCK)
	{
		__m128i		b = _mm_loadu_si128((const __m128i *) (weights + i));
		__m256i		wi = _mm256_cvtepu16_epi32(b);
		__m256		cf = _mm256_mul_ps(_mm256_cvtepi32_ps(wi), tmv);
		float		contrib[PGTURBOHYBRID_SPARSE_AVX2_BLOCK];

		_mm256_storeu_ps(contrib, cf);
		PgturbohybridSparseScatterBlock(contrib, offsets + i, base, scores,
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

#else							/* !PGTURBOHYBRID_SPARSE_COMPILE_AVX2 */

#if PGTURBOHYBRID_SPARSE_X86
/* SIMD disabled at compile time: provide scalar fallbacks so the symbols exist. */
void
PgturbohybridSparseScoreSoaAvx2Q8(const uint8 *weights, const uint16 *offsets,
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
PgturbohybridSparseScoreSoaAvx2Q16(const uint16 *weights, const uint16 *offsets,
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

#endif
