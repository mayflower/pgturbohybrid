/*
 * CodeCode AVX-VNNI raw kernel body, included twice from
 * pgturbohybrid_quant_score_signed_x86.c for the 4-bit / 2-bit codebooks
 * (formerly CodeCodeRawAvxVnni / CodeCode2RawAvxVnni).  Parameters as in
 * pgturbohybrid_quant_score_x86_codecode_avx2.inc; objdump-verified identical.
 */
int64 PGTURBOHYBRID_GRAPH_AVXVNNI_TARGET
TQ_CC_NAME(const uint8 *a, const uint8 *b, int dim,
						  int *sampleDims)
{
	const __m256i signFlip256 = _mm256_set1_epi8((char) 0x80);
	const __m256i ones256 = _mm256_set1_epi8(1);
	const __m128i signFlip128 = _mm_set1_epi8((char) 0x80);
	const __m128i ones128 = _mm_set1_epi8(1);
	__m256i		acc256 = _mm256_setzero_si256();
	__m256i		accCSum256 = _mm256_setzero_si256();
	__m128i		accLo = _mm_setzero_si128();
	__m128i		accCSumLo = _mm_setzero_si128();
	int			chunks = dim / 16;
	int			tailDims = dim - chunks * 16;
	int			scoredChunks = chunks;
	int			scored = 0;
	int64		dot;
	int64		cSum;

	if (chunks > PGTURBOHYBRID_GRAPH_CODE_CODE_PRUNE_CHUNKS)
		scoredChunks = PGTURBOHYBRID_GRAPH_CODE_CODE_PRUNE_CHUNKS;
	*sampleDims = scoredChunks * 16;

	for (; scored + 2 <= scoredChunks; scored += 2)
	{
		int			chunkA = scoredChunks == chunks ? scored :
			(int) (((int64) scored * chunks) / scoredChunks);
		int			chunkB = scoredChunks == chunks ? scored + 1 :
			(int) (((int64) (scored + 1) * chunks) / scoredChunks);
		__m128i		ca0 = TQ_CC_EXPAND(a + chunkA * TQ_CC_STRIDE);
		__m128i		cb0 = TQ_CC_EXPAND(b + chunkA * TQ_CC_STRIDE);
		__m128i		ca1 = TQ_CC_EXPAND(a + chunkB * TQ_CC_STRIDE);
		__m128i		cb1 = TQ_CC_EXPAND(b + chunkB * TQ_CC_STRIDE);
		__m256i		ca = _mm256_inserti128_si256(_mm256_castsi128_si256(ca0), ca1, 1);
		__m256i		cb = _mm256_inserti128_si256(_mm256_castsi128_si256(cb0), cb1, 1);

		acc256 = _mm256_dpbusd_avx_epi32(acc256,
										 _mm256_xor_si256(ca, signFlip256), cb);
		accCSum256 = _mm256_dpbusd_avx_epi32(accCSum256, ones256, cb);
	}

	if (scored < scoredChunks)
	{
		int			chunk = scoredChunks == chunks ? scored :
			(int) (((int64) scored * chunks) / scoredChunks);
		__m128i		ca = TQ_CC_EXPAND(a + chunk * TQ_CC_STRIDE);
		__m128i		cb = TQ_CC_EXPAND(b + chunk * TQ_CC_STRIDE);

		accLo = _mm_dpbusd_avx_epi32(accLo,
									 _mm_xor_si128(ca, signFlip128), cb);
		accCSumLo = _mm_dpbusd_avx_epi32(accCSumLo, ones128, cb);
	}

	if (scoredChunks == chunks && tailDims != 0)
	{
		uint8		scratchA[TQ_CC_STRIDE] = {0};
		uint8		scratchB[TQ_CC_STRIDE] = {0};
		int			tailBytes = (tailDims + TQ_CC_TAILADD) / TQ_CC_TAILDIV;
		__m128i		ca;
		__m128i		cb;

		memcpy(scratchA, a + chunks * TQ_CC_STRIDE, tailBytes);
		memcpy(scratchB, b + chunks * TQ_CC_STRIDE, tailBytes);
		ca = TQ_CC_EXPAND(scratchA);
		cb = TQ_CC_EXPAND(scratchB);
		accLo = _mm_dpbusd_avx_epi32(accLo,
									 _mm_xor_si128(ca, signFlip128), cb);
		accCSumLo = _mm_dpbusd_avx_epi32(accCSumLo, ones128, cb);
		*sampleDims += tailDims;
	}

	dot = PgturbohybridGraphHorizontalSumI32Avx2(acc256) +
		PgturbohybridGraphHorizontalSumI32x4AvxVnni(accLo);
	cSum = PgturbohybridGraphHorizontalSumI32Avx2(accCSum256) +
		PgturbohybridGraphHorizontalSumI32x4AvxVnni(accCSumLo);

	return dot - 128 * cSum;
}
