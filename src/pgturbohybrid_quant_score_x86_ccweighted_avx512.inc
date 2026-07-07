/*
 * Weighted code-code AVX-512 raw kernel body, included twice from
 * pgturbohybrid_quant_score_signed_x86.c for the 4-bit / 2-bit codebooks
 * (formerly CodeCodeWeightedRawAvx512 / CodeCode2WeightedRawAvx512).  Parameters
 * as in pgturbohybrid_quant_score_x86_codecode_avx2.inc; objdump-verified.
 */
int64 PGTURBOHYBRID_GRAPH_AVX512_WEIGHTED_TARGET
TQ_CC_NAME(const uint8 *a, const uint8 *b,
								 const int16 *weights, int dim)
{
	int64		acc = 0;
	int			chunks = dim / 32;
	int			tailDims = dim - chunks * 32;

	for (int chunk = 0; chunk < chunks; chunk++)
	{
		__m256i		ca = TQ_CC_EXPAND(a + chunk * TQ_CC_STRIDE);
		__m256i		cb = TQ_CC_EXPAND(b + chunk * TQ_CC_STRIDE);

		acc += PgturbohybridGraphWeightedDotI8x32Avx512(ca, cb, weights + chunk * 32);
	}

	if (tailDims != 0)
	{
		uint8		scratchA[TQ_CC_STRIDE] = {0};
		uint8		scratchB[TQ_CC_STRIDE] = {0};
		int16		scratchW[32] = {0};
		int			tailBytes = (tailDims + TQ_CC_TAILADD) / TQ_CC_TAILDIV;
		__m256i		ca;
		__m256i		cb;

		memcpy(scratchA, a + chunks * TQ_CC_STRIDE, tailBytes);
		memcpy(scratchB, b + chunks * TQ_CC_STRIDE, tailBytes);
		memcpy(scratchW, weights + chunks * 32, sizeof(int16) * tailDims);
		ca = TQ_CC_EXPAND(scratchA);
		cb = TQ_CC_EXPAND(scratchB);
		acc += PgturbohybridGraphWeightedDotI8x32Avx512(ca, cb, scratchW);
	}

	return acc;
}

/* Undefine our parameters so the includer needs only the #defines. */
#undef TQ_CC_EXPAND
#undef TQ_CC_NAME
#undef TQ_CC_STRIDE
#undef TQ_CC_TAILADD
#undef TQ_CC_TAILDIV
