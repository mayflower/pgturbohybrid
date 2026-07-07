/*
 * Query-split AVX2 raw kernel body, included twice from
 * pgturbohybrid_quant_score_signed_x86.c for the 4-bit / 2-bit codebooks
 * (formerly QuerySplitRawAvx2 / QuerySplit2RawAvx2).  Parameters as in
 * pgturbohybrid_quant_score_x86_codecode_avx2.inc; objdump-verified identical.
 */
int64 PGTURBOHYBRID_GRAPH_AVX2_TARGET
TQ_CC_NAME(const PgturbohybridGraphTqQuery *tq, const uint8 *code)
{
	__m256i	accLow = _mm256_setzero_si256();
	__m256i	accHigh = _mm256_setzero_si256();

	for (int chunk = 0; chunk < tq->querySplitChunks; chunk++)
	{
		__m128i	c = TQ_CC_EXPAND(code + chunk * TQ_CC_STRIDE);
		__m128i	low = _mm_loadu_si128((const __m128i *) (tq->signedSplit.low + chunk * 16));
		__m128i	high = _mm_loadu_si128((const __m128i *) (tq->signedSplit.high + chunk * 16));

		accLow = _mm256_add_epi32(accLow, PgturbohybridGraphDotI8x16Avx2(low, c));
		accHigh = _mm256_add_epi32(accHigh, PgturbohybridGraphDotI8x16Avx2(high, c));
	}

	if (tq->querySplitTailDims != 0)
	{
		uint8		scratch[TQ_CC_STRIDE] = {0};
		int			tailBytes = (tq->querySplitTailDims + TQ_CC_TAILADD) / TQ_CC_TAILDIV;
		__m128i	c;
		__m128i	low;
		__m128i	high;

		memcpy(scratch, code + tq->querySplitChunks * TQ_CC_STRIDE, tailBytes);
		c = TQ_CC_EXPAND(scratch);
		low = _mm_loadu_si128((const __m128i *) tq->signedSplit.tailLow);
		high = _mm_loadu_si128((const __m128i *) tq->signedSplit.tailHigh);
		accLow = _mm256_add_epi32(accLow, PgturbohybridGraphDotI8x16Avx2(low, c));
		accHigh = _mm256_add_epi32(accHigh, PgturbohybridGraphDotI8x16Avx2(high, c));
	}

	return PgturbohybridGraphHorizontalSumI32Avx2(accLow) +
		(int64) PGTURBOHYBRID_QUERY_SPLIT_HIGH_COEF * PgturbohybridGraphHorizontalSumI32Avx2(accHigh);
}

/* Undefine our parameters so the includer needs only the #defines. */
#undef TQ_CC_EXPAND
#undef TQ_CC_NAME
#undef TQ_CC_STRIDE
#undef TQ_CC_TAILADD
#undef TQ_CC_TAILDIV
