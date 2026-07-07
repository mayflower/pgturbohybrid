/*
 * Weighted code-code AVX2 raw kernel body, included twice from
 * pgturbohybrid_quant_score_signed_x86.c for the 4-bit / 2-bit codebooks
 * (formerly CodeCodeWeightedRawAvx2 / CodeCode2WeightedRawAvx2).  Parameters as
 * in pgturbohybrid_quant_score_x86_codecode_avx2.inc; objdump-verified identical.
 */
int64 PGTURBOHYBRID_GRAPH_AVX2_TARGET
TQ_CC_NAME(const uint8 *a, const uint8 *b,
								const int16 *weights, int dim)
{
	__m256i		acc = _mm256_setzero_si256();
	int			chunks = dim / 16;
	int			tailDims = dim - chunks * 16;

	for (int chunk = 0; chunk < chunks; chunk++)
	{
		__m128i		ca = TQ_CC_EXPAND(a + chunk * TQ_CC_STRIDE);
		__m128i		cb = TQ_CC_EXPAND(b + chunk * TQ_CC_STRIDE);

		acc = _mm256_add_epi64(acc,
								PgturbohybridGraphWeightedDotI8x16Avx2(ca, cb, weights + chunk * 16));
	}

	if (tailDims != 0)
	{
		uint8		scratchA[TQ_CC_STRIDE] = {0};
		uint8		scratchB[TQ_CC_STRIDE] = {0};
		int16		scratchW[16] = {0};
		int			tailBytes = (tailDims + TQ_CC_TAILADD) / TQ_CC_TAILDIV;
		__m128i		ca;
		__m128i		cb;

		memcpy(scratchA, a + chunks * TQ_CC_STRIDE, tailBytes);
		memcpy(scratchB, b + chunks * TQ_CC_STRIDE, tailBytes);
		memcpy(scratchW, weights + chunks * 16, sizeof(int16) * tailDims);
		ca = TQ_CC_EXPAND(scratchA);
		cb = TQ_CC_EXPAND(scratchB);
		acc = _mm256_add_epi64(acc, PgturbohybridGraphWeightedDotI8x16Avx2(ca, cb, scratchW));
	}

	return PgturbohybridGraphHorizontalSumI64Avx2(acc);
}
