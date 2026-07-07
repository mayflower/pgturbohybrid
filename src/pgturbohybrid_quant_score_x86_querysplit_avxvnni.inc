/*
 * Query-split AVX-VNNI raw kernel body, included twice from
 * pgturbohybrid_quant_score_signed_x86.c for the 4-bit / 2-bit codebooks
 * (formerly QuerySplitRawAvxVnni / QuerySplit2RawAvxVnni).  Parameters as in
 * pgturbohybrid_quant_score_x86_codecode_avx2.inc; objdump-verified identical.
 */
int64 PGTURBOHYBRID_GRAPH_AVXVNNI_TARGET
TQ_CC_NAME(const PgturbohybridGraphTqQuery *tq, const uint8 *code)
{
	const __m256i ones256 = _mm256_set1_epi8(1);
	const __m128i ones128 = _mm_set1_epi8(1);
	__m256i		accLow = _mm256_setzero_si256();
	__m256i		accHigh = _mm256_setzero_si256();
	__m256i		accCSum = _mm256_setzero_si256();
	__m128i		accLowLo = _mm_setzero_si128();
	__m128i		accHighLo = _mm_setzero_si128();
	__m128i		accCSumLo = _mm_setzero_si128();
	int			chunk = 0;
	int64		dotLow;
	int64		dotHigh;
	int64		cSum;

	/* Pair-stepped main loop: 32 dims per iteration on YMM. */
	for (; chunk + 2 <= tq->querySplitChunks; chunk += 2)
	{
		__m128i		c0 = TQ_CC_EXPAND(code + chunk * TQ_CC_STRIDE);
		__m128i		c1 = TQ_CC_EXPAND(code + (chunk + 1) * TQ_CC_STRIDE);
		__m256i		c = _mm256_inserti128_si256(_mm256_castsi128_si256(c0), c1, 1);
		__m256i		low = _mm256_loadu_si256((const __m256i *) (tq->signedSplit.lowU8 + chunk * 16));
		__m256i		high = _mm256_loadu_si256((const __m256i *) (tq->signedSplit.highU8 + chunk * 16));

		accLow = _mm256_dpbusd_avx_epi32(accLow, low, c);
		accHigh = _mm256_dpbusd_avx_epi32(accHigh, high, c);
		accCSum = _mm256_dpbusd_avx_epi32(accCSum, ones256, c);
	}

	/* Trailing 16-dim chunk (if querySplitChunks is odd). */
	if (chunk < tq->querySplitChunks)
	{
		__m128i		c = TQ_CC_EXPAND(code + chunk * TQ_CC_STRIDE);
		__m128i		low = _mm_loadu_si128((const __m128i *) (tq->signedSplit.lowU8 + chunk * 16));
		__m128i		high = _mm_loadu_si128((const __m128i *) (tq->signedSplit.highU8 + chunk * 16));

		accLowLo = _mm_dpbusd_avx_epi32(accLowLo, low, c);
		accHighLo = _mm_dpbusd_avx_epi32(accHighLo, high, c);
		accCSumLo = _mm_dpbusd_avx_epi32(accCSumLo, ones128, c);
	}

	/* Final sub-chunk tail dims. */
	if (tq->querySplitTailDims != 0)
	{
		uint8		scratch[TQ_CC_STRIDE] = {0};
		int			tailBytes = (tq->querySplitTailDims + TQ_CC_TAILADD) / TQ_CC_TAILDIV;
		__m128i		c;
		__m128i		low;
		__m128i		high;

		memcpy(scratch, code + tq->querySplitChunks * TQ_CC_STRIDE, tailBytes);
		c = TQ_CC_EXPAND(scratch);
		low = _mm_loadu_si128((const __m128i *) tq->signedSplit.tailLowU8);
		high = _mm_loadu_si128((const __m128i *) tq->signedSplit.tailHighU8);
		accLowLo = _mm_dpbusd_avx_epi32(accLowLo, low, c);
		accHighLo = _mm_dpbusd_avx_epi32(accHighLo, high, c);
		accCSumLo = _mm_dpbusd_avx_epi32(accCSumLo, ones128, c);
	}

	dotLow = PgturbohybridGraphHorizontalSumI32Avx2(accLow) +
		PgturbohybridGraphHorizontalSumI32x4AvxVnni(accLowLo);
	dotHigh = PgturbohybridGraphHorizontalSumI32Avx2(accHigh) +
		PgturbohybridGraphHorizontalSumI32x4AvxVnni(accHighLo);
	cSum = PgturbohybridGraphHorizontalSumI32Avx2(accCSum) +
		PgturbohybridGraphHorizontalSumI32x4AvxVnni(accCSumLo);

	return (dotLow - 128 * cSum) +
		(int64) PGTURBOHYBRID_QUERY_SPLIT_HIGH_COEF * (dotHigh - 128 * cSum);
}

/* Undefine our parameters so the includer needs only the #defines. */
#undef TQ_CC_EXPAND
#undef TQ_CC_NAME
#undef TQ_CC_STRIDE
#undef TQ_CC_TAILADD
#undef TQ_CC_TAILDIV
