#ifndef PGTURBOHYBRID_QUANT_CODEBOOK_H
#define PGTURBOHYBRID_QUANT_CODEBOOK_H

/*
 * Single source of truth for the non-uniform (Lloyd-Max) quantization codebook
 * and the scale/coefficient anchors derived from it.
 *
 * These constants and the two int8 codebook tables were previously hand-copied
 * across pgturbohybrid_quant_score.c, *_signed_x86.c, *_u8_x86.c, *_arm.c and
 * pgturbohybrid_graph_utils.c with no compile-time tie -- a wrong edit to one
 * copy silently shifted recall on one code path or architecture.  Everything
 * derived from the codebook now lives here.
 *
 * Include AFTER postgres.h and pgturbohybrid.h (needs int8 and
 * PGTURBOHYBRID_LUT_WIDTH) and, in the SIMD scorer translation units, after
 * their PGTURBOHYBRID_GRAPH_COMPILE_* gate macros are set (the tables below are
 * emitted only where a consuming kernel is compiled).
 */

/*
 * Outermost codebook centroid magnitudes.  Two spellings on purpose: the float
 * form (…_ABS_MAX) feeds the f32 LUT / scalar path; the double form
 * (…_ABS_MAX_D) feeds the x86 asymmetric scorers, which historically used a
 * double literal.  They encode the same number -- keep both in sync if the
 * codebook is ever retrained.  (They cannot be StaticAssert-tied: (double)2.733f
 * != the double 2.733, and that difference is the intended per-path precision.)
 */
#define PGTURBOHYBRID_CODEBOOK_ABS_MAX		2.733f
#define PGTURBOHYBRID_CODEBOOK2_ABS_MAX		1.510f
#define PGTURBOHYBRID_CODEBOOK_ABS_MAX_D	2.733
#define PGTURBOHYBRID_CODEBOOK2_ABS_MAX_D	1.510

/* i8 dequant scale = 127 / outermost-centroid, in each precision. */
#define PGTURBOHYBRID_CODEBOOK_SCALE		(127.0f / PGTURBOHYBRID_CODEBOOK_ABS_MAX)
#define PGTURBOHYBRID_CODEBOOK2_SCALE		(127.0f / PGTURBOHYBRID_CODEBOOK2_ABS_MAX)
#define PGTURBOHYBRID_GRAPH_CODEBOOK_SCALE	(127.0 / PGTURBOHYBRID_CODEBOOK_ABS_MAX_D)
#define PGTURBOHYBRID_GRAPH_CODEBOOK2_SCALE (127.0 / PGTURBOHYBRID_CODEBOOK2_ABS_MAX_D)

/* Query-split / u8 combination coefficients (integer, identical everywhere). */
#define PGTURBOHYBRID_QUERY_SPLIT_HIGH_COEF 256
#define PGTURBOHYBRID_U8_SPLIT_HIGH_COEF	128
#define PGTURBOHYBRID_U8_CODEBOOK_OFFSET	128

/*
 * Signed 4-bit and 2-bit int8 codebooks for the SIMD query-split scorers,
 * emitted once here under the union of the arch gates that consume them (NEON
 * dotprod on aarch64, AVX2+ on x86).  pg_attribute_unused covers build configs
 * that reference only one of the two.
 */
#if PGTURBOHYBRID_GRAPH_COMPILE_ARM_DOT || PGTURBOHYBRID_GRAPH_COMPILE_AVX2
pg_attribute_unused()
static const int8 PgturbohybridGraphCodebookI8[PGTURBOHYBRID_LUT_WIDTH] = {
	-127, -96, -75, -58, -44, -31, -18, -6,
	6, 18, 31, 44, 58, 75, 96, 127
};
pg_attribute_unused()
static const int8 PgturbohybridGraphCodebook2I8[PGTURBOHYBRID_LUT_WIDTH] = {
	-127, -38, 38, 127, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0
};
#endif

#endif							/* PGTURBOHYBRID_QUANT_CODEBOOK_H */
