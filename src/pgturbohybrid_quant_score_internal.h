/*
 * pgturbohybrid_quant_score_internal.h
 *
 * Private declarations shared between the generic dispatch / high-level scoring
 * in pgturbohybrid_quant_score.c and the extracted x86 unsigned-codebook (u8)
 * query-split kernels in pgturbohybrid_quant_score_u8_x86.c.  Not a public API.
 *
 * The SIMD compile/target guard macros are intentionally NOT defined here: each
 * translation unit defines its own (byte-for-byte equivalent) copy so the guards
 * stay self-contained.  This header only declares the cross-TU entry points.
 */
#ifndef PGTURBOHYBRID_QUANT_SCORE_INTERNAL_H
#define PGTURBOHYBRID_QUANT_SCORE_INTERNAL_H

#include "pgturbohybrid_quant_score.h"

#include <stdio.h>
#include <string.h>

/*
 * Valgrind detection (relocated from pgturbohybrid_quant_score.c so the x86 SIMD
 * module's CPU probes can share it).  Valgrind cannot execute AVX-512 / AVX-VNNI
 * instructions, so the availability probes treat it as "not available".  static
 * inline: each translation unit that needs it gets its own copy; unused copies
 * are harmless.
 */
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)) && defined(__linux__)
/*
 * Detect whether the backend is running under Valgrind.  Valgrind passes the
 * host CPUID through, so the AVX-512 / AVX-VNNI feature probes below would
 * report the feature "available" -- but Valgrind's JIT cannot execute many
 * AVX-512 (EVEX) and AVX-VNNI instructions and raises SIGILL on them, which
 * crashes the backend (the CI valgrind job dies at the SIMD-forcing tests).
 * The AVX-512/AVX-VNNI availability probes treat Valgrind as "not available"
 * so dispatch falls back to AVX2/scalar, which Valgrind fully supports; AVX2
 * stays enabled.  Detected once via /proc/self/maps (Valgrind always maps its
 * vgpreload core); no libvalgrind build dependency.
 */
static inline bool pg_attribute_unused()
PgturbohybridGraphRunningUnderValgrind(void)
{
	static int	under = -1;
	FILE	   *f;
	char		line[256];

	if (under >= 0)
		return under != 0;

	under = 0;
	f = fopen("/proc/self/maps", "r");
	if (f != NULL)
	{
		while (fgets(line, sizeof(line), f) != NULL)
		{
			if (strstr(line, "/valgrind/") != NULL ||
				strstr(line, "vgpreload") != NULL)
			{
				under = 1;
				break;
			}
		}
		fclose(f);
	}
	return under != 0;
}
#else
#define PgturbohybridGraphRunningUnderValgrind() (false)
#endif


/*
 * CPU-feature probes, defined in pgturbohybrid_quant_score.c.  The u8 x86 module
 * dispatches on these.  Only ever referenced on x86 with AVX2 compiled in; on
 * other targets they are neither defined nor called.
 */
extern bool PgturbohybridGraphAvx2Available(void);
extern bool PgturbohybridGraphAvx512VnniAvailable(void);

/*
 * x86 unsigned-codebook (u8) 4-bit query-split scorers, defined in
 * pgturbohybrid_quant_score_u8_x86.c.  These are the cross-translation-unit
 * entry points the generic dispatch calls; the AVX2 / AVX-512 VNNI raw kernels
 * they wrap stay file-local to that module.  Declared unconditionally -- they
 * are only defined and only referenced under PGTURBOHYBRID_GRAPH_COMPILE_AVX2,
 * so on non-x86 builds the declarations are simply unused.
 */
extern int64 PgturbohybridGraphQuerySplitU8RawScalar(const PgturbohybridGraphTqQuery *tq,
													 const uint8 *code);
extern double PgturbohybridGraphU8DistanceFromRaw(const PgturbohybridGraphTqQuery *tq,
												  float valueScale, int64 rawDot);
extern bool PgturbohybridGraphPackedDistanceU8Split(const PgturbohybridGraphTqQuery *tq,
													const uint8 *valueCode, float valueScale,
													double *distance);
extern bool PgturbohybridGraphPackedDistanceU8Splitx4(const PgturbohybridGraphTqQuery *tq,
													  const uint8 *codes[4], const float scales[4],
													  double dist[4]);


/*
 * ARM NEON/SDOT dense scorers, defined in pgturbohybrid_quant_score_arm.c.
 * Declared unconditionally; only defined and only referenced under
 * PGTURBOHYBRID_GRAPH_COMPILE_ARM_DOT.
 */
extern bool PgturbohybridGraphArmDotprodAvailable(void);
extern int64 PgturbohybridGraphQuerySplitRawNeonSdot(const PgturbohybridGraphTqQuery *tq,
													 const uint8 *code);
extern int64 PgturbohybridGraphQuerySplit2RawNeonSdot(const PgturbohybridGraphTqQuery *tq,
													  const uint8 *code);
extern int64 PgturbohybridGraphCodeCodeRawNeonSdot(const uint8 *a, const uint8 *b, int dim,
												   int *sampleDims);
extern int64 PgturbohybridGraphCodeCode2RawNeonSdot(const uint8 *a, const uint8 *b, int dim,
													int *sampleDims);
extern int64 PgturbohybridGraphCodeCodeWeightedRawNeonSdot(const uint8 *a, const uint8 *b,
														   const int16 *weights, int dim);
extern int64 PgturbohybridGraphCodeCode2WeightedRawNeonSdot(const uint8 *a, const uint8 *b,
															const int16 *weights, int dim);


/*
 * x86 SIMD dense scorers + exact-vector helpers, defined in
 * pgturbohybrid_quant_score_signed_x86.c.  Declared unconditionally; only
 * defined and only referenced under PGTURBOHYBRID_GRAPH_COMPILE_AVX2 (and the
 * matching AVX-512 / AVX-VNNI / weighted compile gates).
 */
extern bool PgturbohybridGraphAvxVnniAvailable(void);
extern bool PgturbohybridGraphAvx512WeightedAvailable(void);
extern int64 PgturbohybridGraphQuerySplitRawAvx2(const PgturbohybridGraphTqQuery *tq, const uint8 *code);
extern int64 PgturbohybridGraphQuerySplit2RawAvx2(const PgturbohybridGraphTqQuery *tq, const uint8 *code);
extern int64 PgturbohybridGraphCodeCodeRawAvx2(const uint8 *a, const uint8 *b, int dim, int *sampleDims);
extern int64 PgturbohybridGraphCodeCode2RawAvx2(const uint8 *a, const uint8 *b, int dim, int *sampleDims);
extern int64 PgturbohybridGraphCodeCodeWeightedRawAvx2(const uint8 *a, const uint8 *b, const int16 *weights, int dim);
extern int64 PgturbohybridGraphCodeCode2WeightedRawAvx2(const uint8 *a, const uint8 *b, const int16 *weights, int dim);
extern int64 PgturbohybridGraphQuerySplitRawAvx512Vnni(const PgturbohybridGraphTqQuery *tq, const uint8 *code);
extern int64 PgturbohybridGraphQuerySplit2RawAvx512Vnni(const PgturbohybridGraphTqQuery *tq, const uint8 *code);
extern int64 PgturbohybridGraphCodeCodeRawAvx512Vnni(const uint8 *a, const uint8 *b, int dim, int *sampleDims);
extern int64 PgturbohybridGraphCodeCode2RawAvx512Vnni(const uint8 *a, const uint8 *b, int dim, int *sampleDims);
extern int64 PgturbohybridGraphCodeCodeWeightedRawAvx512(const uint8 *a, const uint8 *b, const int16 *weights, int dim);
extern int64 PgturbohybridGraphCodeCode2WeightedRawAvx512(const uint8 *a, const uint8 *b, const int16 *weights, int dim);
extern int64 PgturbohybridGraphQuerySplitRawAvxVnni(const PgturbohybridGraphTqQuery *tq, const uint8 *code);
extern int64 PgturbohybridGraphQuerySplit2RawAvxVnni(const PgturbohybridGraphTqQuery *tq, const uint8 *code);
extern int64 PgturbohybridGraphCodeCodeRawAvxVnni(const uint8 *a, const uint8 *b, int dim, int *sampleDims);
extern int64 PgturbohybridGraphCodeCode2RawAvxVnni(const uint8 *a, const uint8 *b, int dim, int *sampleDims);
extern bool PgturbohybridGraphExactVectorDistanceAvx2(PgturbohybridGraphScanOpaque so, Vector *queryVector,
													  Vector *valueVector, double *result);
extern bool PgturbohybridGraphBuildExactDistanceAvx2(PgturbohybridQuantBuildState *state, Vector *av,
													 Vector *bv, double *result);

#endif							/* PGTURBOHYBRID_QUANT_SCORE_INTERNAL_H */
