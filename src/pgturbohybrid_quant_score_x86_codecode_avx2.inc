/*
 * CodeCode AVX2 raw dot-product kernel body, included twice from
 * pgturbohybrid_quant_score_signed_x86.c to generate the 4-bit and 2-bit
 * codebook variants (formerly hand-duplicated as CodeCodeRawAvx2 /
 * CodeCode2RawAvx2, which differed only in the decode helper, packed stride and
 * tail rounding).  The includer #defines the parameters below and #undefs them
 * afterwards.  Codegen is byte-identical to the former twins (objdump-verified).
 *
 *   TQ_CC_NAME    - generated function name
 *   TQ_CC_EXPAND  - packed-nibble -> int8x16 expand helper
 *   TQ_CC_STRIDE  - packed bytes per 16-dim chunk (8 for 4-bit, 4 for 2-bit)
 *   TQ_CC_TAILADD / TQ_CC_TAILDIV - ceil-divide the tail dims into packed bytes
 */
int64 PGTURBOHYBRID_GRAPH_AVX2_TARGET
TQ_CC_NAME(const uint8 *a, const uint8 *b, int dim,
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
		__m128i	ca = TQ_CC_EXPAND(a + chunk * TQ_CC_STRIDE);
		__m128i	cb = TQ_CC_EXPAND(b + chunk * TQ_CC_STRIDE);

		acc = _mm256_add_epi32(acc, PgturbohybridGraphDotI8x16Avx2(ca, cb));
	}

	if (scoredChunks == chunks && tailDims != 0)
	{
		uint8		scratchA[TQ_CC_STRIDE] = {0};
		uint8		scratchB[TQ_CC_STRIDE] = {0};
		int			tailBytes = (tailDims + TQ_CC_TAILADD) / TQ_CC_TAILDIV;
		__m128i	ca;
		__m128i	cb;

		memcpy(scratchA, a + chunks * TQ_CC_STRIDE, tailBytes);
		memcpy(scratchB, b + chunks * TQ_CC_STRIDE, tailBytes);
		ca = TQ_CC_EXPAND(scratchA);
		cb = TQ_CC_EXPAND(scratchB);
		acc = _mm256_add_epi32(acc, PgturbohybridGraphDotI8x16Avx2(ca, cb));
		*sampleDims += tailDims;
	}

	return PgturbohybridGraphHorizontalSumI32Avx2(acc);
}
