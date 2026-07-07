/*
 * Query-split AVX-512 VNNI raw kernel body, included twice from
 * pgturbohybrid_quant_score_signed_x86.c for the 4-bit / 2-bit codebooks
 * (formerly QuerySplitRawAvx512Vnni / QuerySplit2RawAvx512Vnni).  This shape
 * uses two expand helpers -- the ZMM (…Avx512) path and the YMM/XMM/tail
 * (…Avx2) path.  Parameters as in pgturbohybrid_quant_score_x86_codecode_avx2.inc
 * plus TQ_CC_EXPAND512; objdump-verified identical.
 */
int64 PGTURBOHYBRID_GRAPH_AVX512VNNI_TARGET
TQ_CC_NAME(const PgturbohybridGraphTqQuery *tq, const uint8 *code)
{
	const __m512i ones512 = _mm512_set1_epi8(1);
	const __m256i ones256 = _mm256_set1_epi8(1);
	const __m128i ones128 = _mm_set1_epi8(1);
	__m512i		accLow512 = _mm512_setzero_si512();
	__m512i		accHigh512 = _mm512_setzero_si512();
	__m512i		accCSum512 = _mm512_setzero_si512();
	__m256i		accLow256 = _mm256_setzero_si256();
	__m256i		accHigh256 = _mm256_setzero_si256();
	__m256i		accCSum256 = _mm256_setzero_si256();
	__m128i		accLow128 = _mm_setzero_si128();
	__m128i		accHigh128 = _mm_setzero_si128();
	__m128i		accCSum128 = _mm_setzero_si128();
	int			chunk = 0;
	int64		dotLow;
	int64		dotHigh;
	int64		cSum;

	/* Quad-stepped main loop: 64 dims per iteration on ZMM. */
	for (; chunk + 4 <= tq->querySplitChunks; chunk += 4)
	{
		__m512i		c = TQ_CC_EXPAND512(code + chunk * TQ_CC_STRIDE);
		__m512i		low = _mm512_loadu_si512((const __m512i *) (tq->signedSplit.lowU8 + chunk * 16));
		__m512i		high = _mm512_loadu_si512((const __m512i *) (tq->signedSplit.highU8 + chunk * 16));

		accLow512 = _mm512_dpbusd_epi32(accLow512, low, c);
		accHigh512 = _mm512_dpbusd_epi32(accHigh512, high, c);
		accCSum512 = _mm512_dpbusd_epi32(accCSum512, ones512, c);
	}

	/* Pair-step trailing 32 dims on YMM. */
	if (chunk + 2 <= tq->querySplitChunks)
	{
		__m128i		c0 = TQ_CC_EXPAND(code + chunk * TQ_CC_STRIDE);
		__m128i		c1 = TQ_CC_EXPAND(code + (chunk + 1) * TQ_CC_STRIDE);
		__m256i		c = _mm256_inserti128_si256(_mm256_castsi128_si256(c0), c1, 1);
		__m256i		low = _mm256_loadu_si256((const __m256i *) (tq->signedSplit.lowU8 + chunk * 16));
		__m256i		high = _mm256_loadu_si256((const __m256i *) (tq->signedSplit.highU8 + chunk * 16));

		accLow256 = _mm256_dpbusd_epi32(accLow256, low, c);
		accHigh256 = _mm256_dpbusd_epi32(accHigh256, high, c);
		accCSum256 = _mm256_dpbusd_epi32(accCSum256, ones256, c);
		chunk += 2;
	}

	/* Trailing 16-dim chunk on XMM. */
	if (chunk < tq->querySplitChunks)
	{
		__m128i		c = TQ_CC_EXPAND(code + chunk * TQ_CC_STRIDE);
		__m128i		low = _mm_loadu_si128((const __m128i *) (tq->signedSplit.lowU8 + chunk * 16));
		__m128i		high = _mm_loadu_si128((const __m128i *) (tq->signedSplit.highU8 + chunk * 16));

		accLow128 = _mm_dpbusd_epi32(accLow128, low, c);
		accHigh128 = _mm_dpbusd_epi32(accHigh128, high, c);
		accCSum128 = _mm_dpbusd_epi32(accCSum128, ones128, c);
	}

	/* Sub-chunk tail dims. */
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
		accLow128 = _mm_dpbusd_epi32(accLow128, low, c);
		accHigh128 = _mm_dpbusd_epi32(accHigh128, high, c);
		accCSum128 = _mm_dpbusd_epi32(accCSum128, ones128, c);
	}

	dotLow = (int64) _mm512_reduce_add_epi32(accLow512) +
		PgturbohybridGraphHorizontalSumI32Avx2(accLow256) +
		PgturbohybridGraphHorizontalSumI32x4Avx512Vnni(accLow128);
	dotHigh = (int64) _mm512_reduce_add_epi32(accHigh512) +
		PgturbohybridGraphHorizontalSumI32Avx2(accHigh256) +
		PgturbohybridGraphHorizontalSumI32x4Avx512Vnni(accHigh128);
	cSum = (int64) _mm512_reduce_add_epi32(accCSum512) +
		PgturbohybridGraphHorizontalSumI32Avx2(accCSum256) +
		PgturbohybridGraphHorizontalSumI32x4Avx512Vnni(accCSum128);

	return (dotLow - 128 * cSum) +
		(int64) PGTURBOHYBRID_QUERY_SPLIT_HIGH_COEF * (dotHigh - 128 * cSum);
}
