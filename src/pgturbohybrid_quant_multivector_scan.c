/*
 * pgturbohybrid_quant_multivector_scan.c
 *
 * Multivector / ColBERT scan + scoring subsystem, extracted verbatim from
 * pgturbohybrid_quant.c (no logic change): the document-node scan,
 * dense-candidate collection, MaxSim/reservoir scoring, and the quantized-
 * inverted codebook + codeword scoring helpers.  Hot inner-loop helpers shared
 * with the base scan stay static inline in pgturbohybrid_quant_internal.h so
 * both TUs still inline them.
 */
#include "postgres.h"

#include <ctype.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>

#include "access/generic_xlog.h"
#include "access/genam.h"
#include "access/parallel.h"
#include "access/relscan.h"
#include "access/table.h"
#include "access/tableam.h"
#include "access/xact.h"
#include "access/xloginsert.h"
#include "catalog/index.h"
#include "catalog/pg_type_d.h"
#include "commands/progress.h"
#include "executor/tuptable.h"
#include "miscadmin.h"
#include "nodes/execnodes.h"
#include "optimizer/optimizer.h"
#include "pgstat.h"
#include "port/pg_bitutils.h"
#include "portability/instr_time.h"
#include "storage/bufmgr.h"
#include "storage/fd.h"
#include "storage/barrier.h"
#include "storage/condition_variable.h"
#include "storage/lmgr.h"
#include "tcop/tcopprot.h"
#include "utils/array.h"
#include "utils/backend_status.h"
#include "utils/datum.h"
#include "utils/fmgrprotos.h"
#include "utils/hsearch.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/rel.h"
#include "utils/snapmgr.h"
#include "utils/wait_event.h"


#if PG_VERSION_NUM >= 140000
#include "utils/backend_progress.h"
#endif


#include "pgturbohybrid_quant.h"
#include "pgturbohybrid_am.h"
#include "pgturbohybrid_multivector.h"
#include "pgturbohybrid_quant_psquare.h"
#include "pgturbohybrid_quant_score.h"
#include "pgturbohybrid_vector_compat.h"
#include "pgturbohybrid_quant_internal.h"

#define PGTURBOHYBRID_QUANTIZED_INVERTED_COMPACT_BATCH 1024
#define PGTURBOHYBRID_MULTIVECTOR_COMPACT_DOT_BLOCK 8

typedef struct PgturbohybridMultiVectorPrecompactDoc
{
	uint32		docId;
	uint32		coverage;
	double		score;
} PgturbohybridMultiVectorPrecompactDoc;

static PgturbohybridQuantizedInvertedCodebook *pgturbohybrid_quantized_inverted_codebook_cache = NULL;

/* Forward declarations for intra-file references (moved with the code). */
static void PgturbohybridMultiVectorCandidateHeapOffer(TqDenseCandidate *heap,
											   int *count,
											   int limit,
											   const TqDenseCandidate *candidate);
static void PgturbohybridMultiVectorCandidateHeapSort(TqDenseCandidate *heap,
											 int count);
static bool PgturbohybridMultiVectorIndexUsesContextLevel(Relation index);
static double PgturbohybridMultiVectorIndexMaxSim(Relation index,
												  const PgturbohybridMultiVector *query,
												  const PgturbohybridMultiVector *doc,
												  const float4 *queryWeights,
												  const bool *queryMask);
static const char *PgturbohybridQuantizedInvertedCodebookSourceName(int source);
static uint32 PgturbohybridMultiVectorDeterministicCodeword(const PgturbohybridMultiVector *mv,
											 int32 token);
static double PgturbohybridMultiVectorCentroidPostingPayloadScore(uint16 payload);
static double PgturbohybridMultiVectorCentroidCodewordMaxSimScore(const PgturbohybridMultiVector *query,
																  const float4 *queryWeights,
																  const bool *queryMask,
																  const uint32 *docCodes,
																  uint32 docCodeCount,
																  const float *queryCodewordScores,
																  uint32 codebookSize,
																  uint64 *pairs);
static int	PgturbohybridMultiVectorPrecompactDocScoreCompare(const void *a,
															 const void *b);
static int	PgturbohybridMultiVectorPrecompactDocCoverageCompare(const void *a,
																const void *b);
static int	PgturbohybridUInt32AscCompare(const void *a, const void *b);
static void PgturbohybridMultiVectorPrecompactMarkDoc(bool *keep,
													  uint32 docId,
													  uint32 *unionDocs,
													  uint32 *duplicates);
static int16 PgturbohybridMultiVectorQuantizedInvertedCompactPayload(uint16 payload);
static int16 PgturbohybridMultiVectorQuantizedInvertedSignedCompactPayload(uint16 payload);

static int
PgturbohybridMultiVectorDocCandidateLimit(int targetK)
{
	int64		docLimit;

	docLimit = targetK > 0 ? targetK : pgturbohybrid_multivector_doc_candidate_k;
	docLimit = Min(docLimit, (int64) pgturbohybrid_multivector_doc_candidate_k);
	return (int) Max(docLimit, (int64) 1);
}

static int64
PgturbohybridMultiVectorDocumentProxyFinalK(PgturbohybridGraphScanOpaque so)
{
	if (so != NULL && so->graphFinalK > 0)
		return so->graphFinalK;
	if (so != NULL && so->hasTupleTargetRows && so->tupleTargetRows > 0)
		return so->tupleTargetRows;
	return PGTURBOHYBRID_DEFAULT_FINAL_K;
}

static int
PgturbohybridMultiVectorDocumentProxyCandidateTarget(PgturbohybridGraphScanOpaque so,
													int denseK,
													uint64 docCount,
													int *exactRerankKOut)
{
	int64		finalK;
	int64		oversampling;
	int64		finalBand;
	int64		target;
	int64		exactRerankK;

	if (exactRerankKOut != NULL)
		*exactRerankKOut = 0;
	if (docCount == 0)
		return 0;

	finalK = Max(PgturbohybridMultiVectorDocumentProxyFinalK(so), (int64) 1);
	oversampling =
		so != NULL ? Max((int64) so->graphOversampling, (int64) 1) :
		(int64) PGTURBOHYBRID_DEFAULT_GRAPH_OVERSAMPLING;
	if (finalK > PG_INT64_MAX / oversampling)
		finalBand = PG_INT64_MAX;
	else
		finalBand = finalK * oversampling;

	target = Max((int64) Max(denseK, 1), finalBand);
	target = Min(target, (int64) pgturbohybrid_multivector_doc_candidate_k);
	target = Min(target, (int64) docCount);
	target = Max(target, (int64) 1);

	if (pgturbohybrid_multivector_exact_rerank ==
		PGTURBOHYBRID_MULTIVECTOR_EXACT_RERANK_OFF)
		exactRerankK = 0;
	else
	{
		exactRerankK =
			Min(target, (int64) pgturbohybrid_multivector_exact_rerank_k);
		if (pgturbohybrid_multivector_doc_graph_rescore_k > 0)
			exactRerankK =
				Min(exactRerankK,
					(int64) pgturbohybrid_multivector_doc_graph_rescore_k);
		exactRerankK = Max(exactRerankK, (int64) 0);
	}

	if (exactRerankKOut != NULL)
		*exactRerankKOut = (int) Min(exactRerankK, (int64) PG_INT32_MAX);
	return (int) Min(target, (int64) PG_INT32_MAX);
}

typedef struct PgturbohybridMultiVectorExactRerankWorkStats
{
	uint32		candidates;
	uint64		tokensEvaluated;
	uint64		tokensSkipped;
	uint64		pairsSaved;
	int			source;
	uint64		heapFetches;
	uint64		sidecarReads;
	uint64		sidecarBytes;
	uint64		sidecarLoadUs;
	uint64		heapVisibilityUs;
	uint64		exactHeapFetchUs;
	uint64		exactMaxsimUs;
	uint64		totalUs;
	bool		adaptiveTopKChangedVsFull;
}			PgturbohybridMultiVectorExactRerankWorkStats;

static bool
PgturbohybridMultiVectorExactRerankCanUseSidecar(PgturbohybridGraphMetaPageData *meta,
												 PgturbohybridGraphScanStorage *storage)
{
	return meta != NULL &&
		storage != NULL &&
		storage->multivectorDocMapLoaded &&
		storage->multivectorDocVectorsLoaded &&
		storage->multivectorDocMap != NULL &&
		meta->tqMultivectorDocCount > 0;
}

static double
PgturbohybridMultiVectorAdaptiveDot(const float *a, const float *b, int32 dim)
{
	double		dot = 0.0;

	for (int32 i = 0; i < dim; i++)
		dot += (double) a[i] * (double) b[i];

	return dot;
}

static bool
PgturbohybridMultiVectorTokenNorm(const PgturbohybridMultiVector *mv,
								  int32 token,
								  double *norm)
{
	const float *values;
	double		sum = 0.0;

	if (mv == NULL || token < 0 || token >= mv->count || norm == NULL)
		return false;

	values = mv->values + ((Size) token * (Size) mv->dim);
	for (int32 i = 0; i < mv->dim; i++)
	{
		double		value = values[i];

		if (!isfinite(value))
			return false;
		sum += value * value;
		if (!isfinite(sum))
			return false;
	}

	*norm = sqrt(sum);
	return isfinite(*norm);
}

static bool
PgturbohybridMultiVectorBuildQueryNormOrder(const PgturbohybridMultiVector *query,
											const float4 *queryWeights,
											const bool *queryMask,
											int32 **orderOut,
											double **normsOut,
											double **suffixNormsOut)
{
	int32	   *order;
	double	   *norms;
	double	   *suffixNorms;

	if (query == NULL || query->count <= 0 ||
		orderOut == NULL || normsOut == NULL || suffixNormsOut == NULL)
		return false;

	order = palloc(sizeof(int32) * query->count);
	norms = palloc(sizeof(double) * query->count);
	suffixNorms = palloc(sizeof(double) * ((Size) query->count + 1));

	for (int32 qi = 0; qi < query->count; qi++)
	{
		double		weight = queryWeights != NULL ? (double) queryWeights[qi] : 1.0;

		order[qi] = qi;
		if (queryMask != NULL && queryMask[qi])
		{
			norms[qi] = 0.0;
			continue;
		}
		if (!isfinite(weight) || weight < 0.0)
		{
			pfree(order);
			pfree(norms);
			pfree(suffixNorms);
			return false;
		}
		if (!PgturbohybridMultiVectorTokenNorm(query, qi, &norms[qi]))
		{
			pfree(order);
			pfree(norms);
			pfree(suffixNorms);
			return false;
		}
		norms[qi] *= weight;
	}

	for (int32 i = 0; i < query->count - 1; i++)
	{
		int32		best = i;

		for (int32 j = i + 1; j < query->count; j++)
		{
			int32		bestQi = order[best];
			int32		jQi = order[j];

			if (norms[jQi] > norms[bestQi] ||
				(norms[jQi] == norms[bestQi] && jQi < bestQi))
				best = j;
		}
		if (best != i)
		{
			int32		tmp = order[i];

			order[i] = order[best];
			order[best] = tmp;
		}
	}

	suffixNorms[query->count] = 0.0;
	for (int32 pos = query->count - 1; pos >= 0; pos--)
	{
		suffixNorms[pos] = suffixNorms[pos + 1] + norms[order[pos]];
		if (!isfinite(suffixNorms[pos]))
		{
			pfree(order);
			pfree(norms);
			pfree(suffixNorms);
			return false;
		}
	}

	*orderOut = order;
	*normsOut = norms;
	*suffixNormsOut = suffixNorms;
	return true;
}

static bool
PgturbohybridMultiVectorDocMaxTokenNorm(const PgturbohybridMultiVector *doc,
										double *maxNorm)
{
	double		maxValue = 0.0;

	if (doc == NULL || doc->count <= 0 || maxNorm == NULL)
		return false;

	for (int32 di = 0; di < doc->count; di++)
	{
		double		norm;

		if (!PgturbohybridMultiVectorTokenNorm(doc, di, &norm))
			return false;
		if (norm > maxValue)
			maxValue = norm;
	}

	*maxNorm = maxValue;
	return isfinite(*maxNorm);
}

static bool
PgturbohybridMultiVectorMaxSimAdaptiveBounded(const PgturbohybridMultiVector *query,
											  const PgturbohybridMultiVector *doc,
											  const float4 *queryWeights,
											  const bool *queryMask,
											  const int32 *queryOrder,
											  const double *querySuffixNorms,
											  double threshold,
											  bool thresholdValid,
											  double *score,
											  bool *pruned,
											  uint64 *tokensEvaluated,
											  uint64 *tokensSkipped,
											  uint64 *pairsEvaluated,
											  uint64 *pairsSaved)
{
	double		docMaxNorm;
	double		partial = 0.0;

	if (score == NULL || pruned == NULL || tokensEvaluated == NULL ||
		tokensSkipped == NULL || pairsEvaluated == NULL ||
		pairsSaved == NULL)
		return false;
	*score = 0.0;
	*pruned = false;
	*tokensEvaluated = 0;
	*tokensSkipped = 0;
	*pairsEvaluated = 0;
	*pairsSaved = 0;

	if (query == NULL || doc == NULL || queryOrder == NULL ||
		querySuffixNorms == NULL || query->count <= 0 || doc->count <= 0)
		return false;

	PgturbohybridCheckSameMultiVectorDims(query, doc);
	if (!PgturbohybridMultiVectorDocMaxTokenNorm(doc, &docMaxNorm))
		return false;

	for (int32 pos = 0; pos < query->count; pos++)
	{
		int32		qi = queryOrder[pos];
		const float *qv =
			query->values + ((Size) qi * (Size) query->dim);
		double		weight = queryWeights != NULL ? (double) queryWeights[qi] : 1.0;
		double		best = -INFINITY;

		if ((queryMask != NULL && queryMask[qi]) || weight == 0.0)
		{
			(*tokensSkipped)++;
			*pairsSaved += (uint64) doc->count;
			continue;
		}
		if (!isfinite(weight) || weight < 0.0)
			return false;

		for (int32 di = 0; di < doc->count; di++)
		{
			const float *dv =
				doc->values + ((Size) di * (Size) doc->dim);
			double		dot =
				PgturbohybridMultiVectorAdaptiveDot(qv, dv, query->dim);

			if (!isfinite(dot))
				return false;
			if (dot > best)
				best = dot;
		}

		partial += weight * best;
		if (!isfinite(partial))
			return false;

		(*tokensEvaluated)++;
		*pairsEvaluated += (uint64) doc->count;

		if (thresholdValid && pos + 1 < query->count)
		{
			double		upperBound =
				partial + docMaxNorm * querySuffixNorms[pos + 1];

			if (!isfinite(upperBound))
				return false;
			if (upperBound < threshold)
			{
				uint64		skipped = (uint64) (query->count - pos - 1);

				*tokensSkipped += skipped;
				*pairsSaved += skipped * (uint64) doc->count;
				*score = upperBound;
				*pruned = true;
				return true;
			}
		}
	}

	*score = partial;
	return true;
}

static void
PgturbohybridMultiVectorAdaptiveTopKOffer(double score,
										  double *topScores,
										  int *topCount,
										  int topK,
										  double *threshold,
										  bool *thresholdValid)
{
	int			minIndex = 0;

	if (topScores == NULL || topCount == NULL || threshold == NULL ||
		thresholdValid == NULL || topK <= 0 || !isfinite(score))
		return;

	if (*topCount < topK)
	{
		topScores[*topCount] = score;
		(*topCount)++;
	}
	else if (score > *threshold)
	{
		for (int i = 1; i < topK; i++)
		{
			if (topScores[i] < topScores[minIndex])
				minIndex = i;
		}
		topScores[minIndex] = score;
	}
	else
		return;

	if (*topCount >= topK)
	{
		minIndex = 0;
		for (int i = 1; i < topK; i++)
		{
			if (topScores[i] < topScores[minIndex])
				minIndex = i;
		}
		*threshold = topScores[minIndex];
		*thresholdValid = true;
	}
}

static int
PgturbohybridMultiVectorAdaptiveTopK(PgturbohybridGraphScanOpaque so,
									 int limit)
{
	int64		topK;

	if (limit <= 0)
		return 0;

	if (so != NULL && so->graphFinalK > 0)
		topK = so->graphFinalK;
	else if (so != NULL && so->hasTupleTargetRows && so->tupleTargetRows > 0)
		topK = so->tupleTargetRows;
	else
		topK = limit;

	topK = Max((int64) 1, topK);
	return (int) Min((int64) limit, topK);
}

static int
PgturbohybridMultiVectorExactRerankLimit(int count)
{
	int64		limit;

	if (count <= 0 ||
		pgturbohybrid_multivector_exact_rerank ==
		PGTURBOHYBRID_MULTIVECTOR_EXACT_RERANK_OFF)
		return 0;

	limit = Min((int64) count,
				(int64) pgturbohybrid_multivector_doc_candidate_k);
	limit = Min(limit, (int64) pgturbohybrid_multivector_exact_rerank_k);
	return (int) Max((int64) 0, limit);
}

static int
PgturbohybridMultiVectorExactHeapRerank(IndexScanDesc scan,
										PgturbohybridGraphScanOpaque so,
										PgturbohybridGraphMetaPageData *sidecarMeta,
										PgturbohybridGraphScanStorage *sidecarStorage,
										PgturbohybridMultiVectorDocSidecarAccessStats *sidecarStats,
										const PgturbohybridMultiVector *query,
										const float4 *queryWeights,
										const bool *queryMask,
										double queryWeightSum,
										TqDenseCandidate *candidates,
										int count,
										int limitOverride,
										uint64 *exactPairs,
										PgturbohybridMultiVectorExactRerankWorkStats *rerankStats)
{
	TupleTableSlot *slot = NULL;
	TupleDesc	desc = NULL;
	AttrNumber	denseAttno = InvalidAttrNumber;
	int			limit;
	int			rescored = 0;
	bool		sidecarUsable;
	bool		adaptiveMode;
	bool		adaptiveReady = false;
	int			adaptiveTopK = 0;
	int			adaptiveTopCount = 0;
	int32	   *queryOrder = NULL;
	double	   *queryNorms = NULL;
	double	   *querySuffixNorms = NULL;
	double	   *adaptiveTopScores = NULL;
	double		adaptiveThreshold = -DBL_MAX;
	bool		adaptiveThresholdValid = false;
	bool		contextLevel = false;
	instr_time	start;

	if (limitOverride > 0 &&
		pgturbohybrid_multivector_exact_rerank !=
		PGTURBOHYBRID_MULTIVECTOR_EXACT_RERANK_OFF)
		limit = (int) Min((int64) count, (int64) limitOverride);
	else
		limit = PgturbohybridMultiVectorExactRerankLimit(count);
	if (limit <= 0 || query == NULL || candidates == NULL ||
		scan == NULL || scan->indexRelation == NULL ||
		scan->indexRelation->rd_index == NULL)
		return 0;

	sidecarUsable =
		sidecarStats != NULL &&
		PgturbohybridMultiVectorExactRerankCanUseSidecar(sidecarMeta,
														 sidecarStorage);
	adaptiveMode =
		pgturbohybrid_multivector_exact_rerank ==
		PGTURBOHYBRID_MULTIVECTOR_EXACT_RERANK_ADAPTIVE;
	contextLevel =
		PgturbohybridMultiVectorIndexUsesContextLevel(scan->indexRelation);
	if (adaptiveMode && !contextLevel)
	{
		adaptiveTopK = PgturbohybridMultiVectorAdaptiveTopK(so, limit);
		if (adaptiveTopK > 0 && adaptiveTopK < limit)
		{
			adaptiveReady =
				PgturbohybridMultiVectorBuildQueryNormOrder(query,
															queryWeights,
															queryMask,
															&queryOrder,
															&queryNorms,
															&querySuffixNorms);
			if (adaptiveReady)
				adaptiveTopScores = palloc(sizeof(double) * adaptiveTopK);
		}
	}

	INSTR_TIME_SET_CURRENT(start);
	for (int i = 0; i < limit; i++)
	{
		Datum		value = (Datum) 0;
		bool		isnull;
		bool		visible;
		char	   *valuePtr = NULL;
		PgturbohybridMultiVector *doc = NULL;
		double		exactMaxsim;
		uint64		docPairsEvaluated = 0;
		uint64		docTokensEvaluated = 0;
		uint64		docTokensSkipped = 0;
		uint64		docPairsSaved = 0;
		bool		adaptivePruned = false;
		bool		adaptiveScored = false;
		bool		docFromSidecar = false;
		bool		heapPathTimed = false;
		instr_time	fetchStart;
		instr_time	heapPathStart;
		instr_time	maxsimStart;

		CHECK_FOR_INTERRUPTS();
		if (candidates[i].exactScored)
			continue;

		if (sidecarUsable &&
			candidates[i].hasDocId &&
			candidates[i].docId < sidecarMeta->tqMultivectorDocCount)
		{
			if (rerankStats != NULL)
				INSTR_TIME_SET_CURRENT(fetchStart);
			doc = PgturbohybridGraphLoadMultiVectorDocVector(scan->indexRelation,
															 sidecarMeta,
															 sidecarStorage,
															 candidates[i].docId,
															 sidecarStorage->ctx != NULL ?
															 sidecarStorage->ctx :
															 CurrentMemoryContext,
															 sidecarStats);
			if (rerankStats != NULL)
				PgturbohybridGraphAddElapsedUint64(&rerankStats->sidecarLoadUs,
												   fetchStart);
			if (doc != NULL)
			{
				docFromSidecar = true;
				if (rerankStats != NULL)
				{
					rerankStats->sidecarReads++;
					rerankStats->sidecarBytes += VARSIZE_ANY(doc);
				}
			}
		}

		if (doc == NULL)
		{
			if (scan->heapRelation == NULL)
				continue;
			if (rerankStats != NULL)
			{
				INSTR_TIME_SET_CURRENT(heapPathStart);
				heapPathTimed = true;
			}
			if (denseAttno == InvalidAttrNumber)
			{
				denseAttno =
					scan->indexRelation->rd_index->indkey.values[PGTURBOHYBRID_DENSE_KEY_INDEX];
				desc = RelationGetDescr(scan->heapRelation);
				if (denseAttno <= 0 || denseAttno > desc->natts)
					break;
			}
			if (slot == NULL)
				slot = table_slot_create(scan->heapRelation, NULL);

			INSTR_TIME_SET_CURRENT(fetchStart);
			visible = table_tuple_fetch_row_version(scan->heapRelation,
													&candidates[i].heaptid,
													scan->xs_snapshot,
													slot);
			PgturbohybridGraphAddElapsedUs(&so->graphHeapFetchUs, fetchStart);
			if (rerankStats != NULL)
				PgturbohybridGraphAddElapsedUint64(&rerankStats->heapVisibilityUs,
												   fetchStart);
			if (rerankStats != NULL)
				rerankStats->heapFetches++;
			if (!visible)
			{
				if (rerankStats != NULL && heapPathTimed)
					PgturbohybridGraphAddElapsedUint64(&rerankStats->exactHeapFetchUs,
													   heapPathStart);
				ExecClearTuple(slot);
				continue;
			}

			value = slot_getattr(slot, denseAttno, &isnull);
			if (isnull)
			{
				if (rerankStats != NULL && heapPathTimed)
					PgturbohybridGraphAddElapsedUint64(&rerankStats->exactHeapFetchUs,
													   heapPathStart);
				ExecClearTuple(slot);
				continue;
			}

			valuePtr = DatumGetPointer(value);
			doc = PgturbohybridDatumGetMultiVector(value);
			if (rerankStats != NULL && heapPathTimed)
				PgturbohybridGraphAddElapsedUint64(&rerankStats->exactHeapFetchUs,
												   heapPathStart);
		}
		PgturbohybridCheckSameMultiVectorDims(query, doc);
		if (rerankStats != NULL)
			INSTR_TIME_SET_CURRENT(maxsimStart);
		if (adaptiveReady)
			adaptiveScored =
				PgturbohybridMultiVectorMaxSimAdaptiveBounded(query, doc,
															  queryWeights,
															  queryMask,
															  queryOrder,
															  querySuffixNorms,
															  adaptiveThreshold,
															  adaptiveThresholdValid,
															  &exactMaxsim,
															  &adaptivePruned,
															  &docTokensEvaluated,
															  &docTokensSkipped,
															  &docPairsEvaluated,
															  &docPairsSaved);
		if (!adaptiveScored)
		{
			docTokensEvaluated = (uint64) query->count;
			docTokensSkipped = 0;
			docPairsEvaluated =
				(uint64) query->count * (uint64) doc->count;
			docPairsSaved = 0;
			adaptivePruned = false;
			exactMaxsim =
				PgturbohybridMultiVectorIndexMaxSim(scan->indexRelation,
													query, doc,
													queryWeights,
													queryMask);
		}
		if (rerankStats != NULL)
			PgturbohybridGraphAddElapsedUint64(&rerankStats->exactMaxsimUs,
											   maxsimStart);
		if (exactPairs != NULL)
			*exactPairs += docPairsEvaluated;
		candidates[i].distance = -exactMaxsim;
		candidates[i].similarity =
			queryWeightSum > 0.0 ? exactMaxsim / queryWeightSum : 0.0;
		candidates[i].exactScored = true;
		if (rerankStats != NULL)
		{
			rerankStats->candidates++;
			rerankStats->tokensEvaluated += docTokensEvaluated;
			rerankStats->tokensSkipped += docTokensSkipped;
			rerankStats->pairsSaved += docPairsSaved;
		}
		if (adaptiveReady && !adaptivePruned)
			PgturbohybridMultiVectorAdaptiveTopKOffer(exactMaxsim,
													  adaptiveTopScores,
													  &adaptiveTopCount,
													  adaptiveTopK,
													  &adaptiveThreshold,
													  &adaptiveThresholdValid);
		rescored++;
		if (docFromSidecar)
		{
			if (sidecarStorage->multivectorDocVectorsPaged)
				pfree(doc);
		}
		else if ((char *) doc != valuePtr)
			pfree(doc);
		if (!docFromSidecar && slot != NULL)
			ExecClearTuple(slot);
	}
	if (slot != NULL)
		ExecDropSingleTupleTableSlot(slot);
	if (queryOrder != NULL)
		pfree(queryOrder);
	if (queryNorms != NULL)
		pfree(queryNorms);
	if (querySuffixNorms != NULL)
		pfree(querySuffixNorms);
	if (adaptiveTopScores != NULL)
		pfree(adaptiveTopScores);

	PgturbohybridGraphAddElapsedUs(&so->graphHeapRescoreUs, start);
	if (rerankStats != NULL)
		PgturbohybridGraphAddElapsedUint64(&rerankStats->totalUs, start);
	if (rescored > 0)
	{
		if (rerankStats != NULL)
		{
			if (rerankStats->heapFetches > 0)
				rerankStats->source =
					PGTURBOHYBRID_MULTIVECTOR_RERANK_SOURCE_HEAP;
			else if (rerankStats->sidecarReads > 0)
				rerankStats->source =
					PGTURBOHYBRID_MULTIVECTOR_RERANK_SOURCE_SIDECAR;
			else
				rerankStats->source =
					PGTURBOHYBRID_MULTIVECTOR_RERANK_SOURCE_OFF;
			so->graphHeapRescoreCount += rerankStats->heapFetches;
			if (rerankStats->heapFetches > 0)
				so->graphExactRescoreSource =
					PGTURBOHYBRID_EXACT_RESCORE_SOURCE_HEAP;
		}
		else
		{
			so->graphHeapRescoreCount += rescored;
			so->graphExactRescoreSource =
				PGTURBOHYBRID_EXACT_RESCORE_SOURCE_HEAP;
		}
		so->graphEffectiveRescoreBand = limit;
		return rescored;
	}
	return 0;
}

static uint64
PgturbohybridMultiVectorEstimatedDocs(IndexScanDesc scan,
									  const PgturbohybridGraphMetaPageData *meta)
{
	double		reltuples = -1.0;

	if (meta != NULL && meta->tqMultivectorDocCount > 0)
		return meta->tqMultivectorDocCount;
	if (scan != NULL && scan->heapRelation != NULL)
		reltuples = scan->heapRelation->rd_rel->reltuples;
	if (reltuples > 0.0)
	{
		if (reltuples >= (double) PG_UINT64_MAX)
			return PG_UINT64_MAX;
		return (uint64) ceil(reltuples);
	}
	if (meta != NULL && meta->tqNodeCount > 0)
		return meta->tqNodeCount;
	return 0;
}

static bool
PgturbohybridMultiVectorShouldUsePlainFallback(IndexScanDesc scan,
											   PgturbohybridGraphScanOpaque so,
											   const PgturbohybridGraphMetaPageData *meta,
											   int targetK,
											   char *reason,
											   Size reasonSize)
{
	uint64		estimatedDocs;
	double		candidateThreshold;
	int64		effectiveDocCandidateK;
	int64		effectiveExactRerankK;

	if (reason != NULL && reasonSize > 0)
		strlcpy(reason, "not_applicable", reasonSize);

	if (pgturbohybrid_multivector_plain_fallback ==
		PGTURBOHYBRID_MULTIVECTOR_PLAIN_FALLBACK_OFF)
	{
		if (reason != NULL && reasonSize > 0)
			strlcpy(reason, "off", reasonSize);
		return false;
	}
	if (pgturbohybrid_multivector_plain_fallback ==
		PGTURBOHYBRID_MULTIVECTOR_PLAIN_FALLBACK_FORCE)
	{
		if (reason != NULL && reasonSize > 0)
			strlcpy(reason, "force", reasonSize);
		return true;
	}

	estimatedDocs = PgturbohybridMultiVectorEstimatedDocs(scan, meta);
	if (estimatedDocs <= (uint64) pgturbohybrid_multivector_plain_fallback_max_docs)
	{
		if (reason != NULL && reasonSize > 0)
			strlcpy(reason, "small_estimated_docs", reasonSize);
		return true;
	}

	candidateThreshold =
		pgturbohybrid_multivector_plain_fallback_candidate_fraction *
		(double) estimatedDocs;
	effectiveDocCandidateK =
		targetK > 0 ? Min((int64) targetK,
						  (int64) pgturbohybrid_multivector_doc_candidate_k) :
		(int64) pgturbohybrid_multivector_doc_candidate_k;
	if ((double) effectiveDocCandidateK > candidateThreshold)
	{
		if (reason != NULL && reasonSize > 0)
			strlcpy(reason, "doc_candidate_fraction", reasonSize);
		return true;
	}

	if (meta != NULL &&
		meta->tqMultivectorGraphMode ==
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES &&
		pgturbohybrid_multivector_candidate_source !=
		PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_CENTROID_LITE &&
		pgturbohybrid_multivector_candidate_source !=
		PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_QUANTIZED_INVERTED_EXPERIMENTAL)
	{
		int64		docLimit;
		int64		rescoreLimit;
		int64		oversampling;
		int64		effectiveDocGraphCandidateK;

		if (pgturbohybrid_multivector_candidate_source ==
			PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_GRAPH ||
			pgturbohybrid_multivector_candidate_source ==
			PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_PROXY_VECTOR)
			effectiveDocGraphCandidateK =
				PgturbohybridMultiVectorDocumentProxyCandidateTarget(so,
																	targetK,
																	meta->tqMultivectorDocCount,
																	NULL);
		else
		{
			docLimit =
				targetK > 0 ? Min((int64) targetK,
								  (int64) pgturbohybrid_multivector_doc_candidate_k) :
				(int64) pgturbohybrid_multivector_doc_candidate_k;
			docLimit = Max(docLimit, (int64) 1);
			rescoreLimit =
				pgturbohybrid_multivector_doc_graph_rescore_k > 0 ?
				pgturbohybrid_multivector_doc_graph_rescore_k :
				pgturbohybrid_multivector_doc_candidate_k;
			rescoreLimit =
				Min(Max(rescoreLimit, docLimit), (int64) meta->tqMultivectorDocCount);
			oversampling =
				Max((int64) pgturbohybrid_multivector_doc_graph_oversampling,
					(int64) 1);
			if (rescoreLimit > PG_INT64_MAX / oversampling)
				effectiveDocGraphCandidateK = (int64) meta->tqMultivectorDocCount;
			else
				effectiveDocGraphCandidateK =
					Min((int64) meta->tqMultivectorDocCount,
						Max(docLimit, rescoreLimit * oversampling));
		}
		if ((double) effectiveDocGraphCandidateK > candidateThreshold)
		{
			if (reason != NULL && reasonSize > 0)
				strlcpy(reason, "document_node_candidate_fraction",
						reasonSize);
			return true;
		}
	}

	if (pgturbohybrid_multivector_exact_rerank !=
		PGTURBOHYBRID_MULTIVECTOR_EXACT_RERANK_OFF)
	{
		effectiveExactRerankK =
			Min((int64) pgturbohybrid_multivector_exact_rerank_k,
				(int64) pgturbohybrid_multivector_doc_candidate_k);
		if ((double) effectiveExactRerankK > candidateThreshold)
		{
			if (reason != NULL && reasonSize > 0)
				strlcpy(reason, "exact_rerank_fraction", reasonSize);
			return true;
		}
	}

	return false;
}

static int
PgturbohybridMultiVectorExactPlainFallback(IndexScanDesc scan,
										   PgturbohybridGraphScanOpaque so,
										   const PgturbohybridMultiVector *query,
										   const float4 *queryWeights,
										   const bool *queryMask,
										   double queryWeightSum,
										   int targetK,
										   TqDenseCandidate **outCandidates,
										   MemoryContext resultCtx,
										   TqDenseCandidateStats *stats,
										   const char *reason,
										   const char *candidateSource,
										   bool docGraphPrototype,
										   const char *docGraphWarning)
{
	TableScanDesc heapScan;
	TupleTableSlot *slot;
	TupleDesc	desc;
	AttrNumber	denseAttno;
	TqDenseCandidate *candidates;
	int			docLimit;
	int			docCount = 0;
	uint64		docsScored = 0;
	uint64		exactPairs = 0;
	MemoryContext oldCtx;
	instr_time	start;

	if (outCandidates == NULL)
		return 0;
	*outCandidates = NULL;
	if (scan == NULL || scan->heapRelation == NULL ||
		scan->indexRelation == NULL || scan->indexRelation->rd_index == NULL ||
		query == NULL)
		return 0;

	denseAttno =
		scan->indexRelation->rd_index->indkey.values[PGTURBOHYBRID_DENSE_KEY_INDEX];
	desc = RelationGetDescr(scan->heapRelation);
	if (denseAttno <= 0 || denseAttno > desc->natts)
		return 0;

	oldCtx = MemoryContextSwitchTo(resultCtx);
	docLimit = PgturbohybridMultiVectorDocCandidateLimit(targetK);
	candidates = palloc0(sizeof(TqDenseCandidate) * docLimit);
	slot = table_slot_create(scan->heapRelation, NULL);

	INSTR_TIME_SET_CURRENT(start);
#if PG_VERSION_NUM >= 190000
	heapScan = table_beginscan(scan->heapRelation, scan->xs_snapshot, 0, NULL,
							   SO_NONE);
#else
	heapScan = table_beginscan(scan->heapRelation, scan->xs_snapshot, 0, NULL);
#endif
	while (table_scan_getnextslot(heapScan, ForwardScanDirection, slot))
	{
		Datum		value;
		bool		isnull;
		char	   *valuePtr;
		PgturbohybridMultiVector *doc;
		double		exactMaxsim;
		TqDenseCandidate candidate;

		CHECK_FOR_INTERRUPTS();
		value = slot_getattr(slot, denseAttno, &isnull);
		if (isnull)
		{
			ExecClearTuple(slot);
			continue;
		}

		valuePtr = DatumGetPointer(value);
		doc = PgturbohybridDatumGetMultiVector(value);
		PgturbohybridCheckSameMultiVectorDims(query, doc);
		exactPairs += (uint64) query->count * (uint64) doc->count;
		exactMaxsim =
			PgturbohybridMultiVectorIndexMaxSim(scan->indexRelation, query,
												doc, queryWeights,
												queryMask);

		memset(&candidate, 0, sizeof(candidate));
		candidate.nodeId = UINT_MAX;
		candidate.heaptid = slot->tts_tid;
		candidate.distance = -exactMaxsim;
		candidate.similarity =
			queryWeightSum > 0.0 ? exactMaxsim / queryWeightSum : 0.0;
		candidate.rank = 0;
		candidate.exactScored = true;
		PgturbohybridMultiVectorCandidateHeapOffer(candidates, &docCount,
												  docLimit, &candidate);
		docsScored++;

		if ((char *) doc != valuePtr)
			pfree(doc);
		ExecClearTuple(slot);
	}
	table_endscan(heapScan);
	ExecDropSingleTupleTableSlot(slot);

	PgturbohybridMultiVectorCandidateHeapSort(candidates, docCount);
	for (int i = 0; i < docCount; i++)
		candidates[i].rank = i + 1;

	if (so != NULL)
	{
		so->graphHeapRescoreCount += docsScored;
		so->graphExactRescoreSource = PGTURBOHYBRID_EXACT_RESCORE_SOURCE_HEAP;
		so->graphEffectiveRescoreBand = (int) Min(docsScored, (uint64) PG_INT32_MAX);
		PgturbohybridGraphAddElapsedUs(&so->graphHeapRescoreUs, start);
	}

	if (stats != NULL)
	{
		stats->denseCandidatesRequested = targetK > 0 ? targetK : docLimit;
		stats->effectiveResultTarget = (uint32) docLimit;
		stats->effectiveSearchEf = 0;
		stats->effectiveRescoreBand =
			(uint32) Min(docsScored, (uint64) PG_UINT32_MAX);
		stats->denseCandidatesReturned = docCount;
		stats->heapRescoreCount = docsScored;
		stats->heapRescoreUs = so != NULL ? so->graphHeapRescoreUs : 0;
		stats->exactRescoreSource = PGTURBOHYBRID_EXACT_RESCORE_SOURCE_HEAP;
		stats->multivectorEnabled = true;
		stats->multivectorQueryVectors = (uint32) query->count;
		stats->multivectorDocVectorsLimit =
			(uint32) pgturbohybrid_multivector_max_doc_vectors;
		stats->multivectorSubvectorSearches = 0;
		stats->multivectorRawSubvectorHits = 0;
		stats->multivectorDocMapSource =
			PGTURBOHYBRID_MULTIVECTOR_DOCMAP_SOURCE_NONE;
		strlcpy(stats->multivectorCandidateSource,
				candidateSource != NULL ? candidateSource : "plain_fallback",
				sizeof(stats->multivectorCandidateSource));
		strlcpy(stats->multivectorGraphMode,
				PgturbohybridMultiVectorGraphModeName(PGTURBOHYBRID_DEFAULT_MULTIVECTOR_GRAPH_MODE),
				sizeof(stats->multivectorGraphMode));
		stats->multivectorExactTokenScanEnabled = false;
		stats->multivectorExactTokenScanNodesScored = 0;
		stats->multivectorPlainFallbackUsed = true;
		strlcpy(stats->multivectorPlainFallbackReason,
				reason != NULL ? reason : "not_applicable",
				sizeof(stats->multivectorPlainFallbackReason));
		stats->multivectorPlainFallbackDocsScored = docsScored;
		stats->multivectorPlainFallbackPairs = exactPairs;
		stats->multivectorDocGraphPrototypeEnabled = docGraphPrototype;
		stats->multivectorDocGraphNodes =
			docGraphPrototype ? docsScored : 0;
		stats->multivectorDocGraphDocsScored =
			docGraphPrototype ? docsScored : 0;
		stats->multivectorDocGraphEdgesVisited = 0;
		stats->multivectorDocGraphCandidates =
			docGraphPrototype ? (uint32) docCount : 0;
		stats->multivectorDocGraphSearchEf = 0;
		stats->multivectorDocGraphOversampling = 0;
		stats->multivectorDocGraphRescoreK = 0;
		stats->multivectorDocGraphQuantizedScores = 0;
		stats->multivectorDocGraphExactRerankDocs =
			docGraphPrototype ? (uint32) Min(docsScored, (uint64) PG_UINT32_MAX) : 0;
		stats->multivectorDocGraphHeapFetches =
			docGraphPrototype ? docsScored : 0;
		strlcpy(stats->multivectorDocGraphWarning,
				docGraphWarning != NULL ? docGraphWarning : "not_applicable",
				sizeof(stats->multivectorDocGraphWarning));
		stats->multivectorDocMapBytes = 0;
		stats->multivectorUniqueDocs = docsScored;
		stats->multivectorDuplicateDocHits = 0;
		stats->multivectorMaxsimUpdates = exactPairs;
		stats->multivectorDocCandidates = (uint32) docCount;
		stats->multivectorExactRerankEnabled = true;
		stats->multivectorExactRerankDocs =
			(uint32) Min(docsScored, (uint64) PG_UINT32_MAX);
		stats->multivectorExactRerankPairs = exactPairs;
		stats->multivectorExactRerankSource =
			docsScored > 0 ?
			PGTURBOHYBRID_MULTIVECTOR_RERANK_SOURCE_HEAP :
			PGTURBOHYBRID_MULTIVECTOR_RERANK_SOURCE_OFF;
		stats->multivectorExactRerankHeapFetches = docsScored;
		stats->multivectorExactRerankSidecarReads = 0;
		stats->multivectorExactRerankSidecarBytes = 0;
		stats->exactRerankCandidates =
			(uint32) Min(docsScored, (uint64) PG_UINT32_MAX);
		stats->exactRerankTokensEvaluated =
			docsScored * (uint64) query->count;
		stats->exactRerankTokensSkipped = 0;
		stats->exactRerankPairsSaved = 0;
		stats->adaptiveRerankTopKChangedVsFull = false;
		strlcpy(stats->multivectorExactKernel,
				docsScored > 0 ? TqMultiVectorMaxSimKernelName() : "",
				sizeof(stats->multivectorExactKernel));
		strlcpy(stats->multivectorAccumulatorKind, "plain_heap_scan",
				sizeof(stats->multivectorAccumulatorKind));
		stats->multivectorMemoryBytesEstimate =
			(uint64) sizeof(TqDenseCandidate) * (uint64) docLimit;
		stats->multivectorAdmissionDebugEnabled =
			pgturbohybrid_multivector_debug_admission !=
			PGTURBOHYBRID_MULTIVECTOR_DEBUG_ADMISSION_OFF;
		stats->multivectorAdmissionCandidatesBeforeRerank =
			(uint32) Min(docsScored, (uint64) PG_UINT32_MAX);
		stats->multivectorAdmissionCandidatesAfterTruncation =
			(uint32) docCount;
		stats->multivectorAdmissionExactRerankDocs =
			(uint32) Min(docsScored, (uint64) PG_UINT32_MAX);
		stats->multivectorAdmissionTruncatedByDocCandidateK =
			docsScored > (uint64) docCount;
		stats->multivectorAdmissionTruncatedByAccumulatorMemory = false;
		stats->multivectorAdmissionTraceAvailable = false;
		stats->multivectorAdmissionTraceCount = 0;
	}

	*outCandidates = candidates;
	MemoryContextSwitchTo(oldCtx);
	return docCount;
}

typedef struct PgturbohybridMultiVectorDocKey
{
	TqDocId		docId;
} PgturbohybridMultiVectorDocKey;

typedef struct PgturbohybridMultiVectorDocEntry
{
	PgturbohybridMultiVectorDocKey key;
	TqDocId		docId;
	ItemPointerData heaptid;
	uint32		bestNodeId;
	double		bestSimilarity;
	double		score;
	double	   *maxsim;
	bool	   *seen;
	int			matchedTokens;
	uint32		rawHitCount;
	uint32		duplicateHitCount;
} PgturbohybridMultiVectorDocEntry;

typedef struct PgturbohybridMultiVectorReservoirDedupeEntry
{
	TqDocId		docId;
} PgturbohybridMultiVectorReservoirDedupeEntry;

typedef struct PgturbohybridMultiVectorReservoirRankItem
{
	PgturbohybridMultiVectorDocEntry *doc;
	double		value;
	double		secondary;
	int			coverage;
} PgturbohybridMultiVectorReservoirRankItem;

typedef struct PgturbohybridMultiVectorTidKey
{
	BlockNumber block;
	OffsetNumber offset;
} PgturbohybridMultiVectorTidKey;

typedef struct PgturbohybridMultiVectorDocIdEntry
{
	PgturbohybridMultiVectorTidKey key;
	TqDocId		docId;
	ItemPointerData heaptid;
} PgturbohybridMultiVectorDocIdEntry;

typedef struct PgturbohybridMultiVectorTokenSeenEntry
{
	TqDocId		docId;
} PgturbohybridMultiVectorTokenSeenEntry;

typedef struct PgturbohybridMultiVectorAccumulatorArenaChunk
{
	struct PgturbohybridMultiVectorAccumulatorArenaChunk *next;
	uint32		capacity;
	uint32		used;
	double	   *maxsim;
	bool	   *seen;
} PgturbohybridMultiVectorAccumulatorArenaChunk;

typedef struct PgturbohybridMultiVectorAccumulatorArena
{
	MemoryContext ctx;
	int			queryCount;
	uint32		chunkDocCapacity;
	uint64		allocatedDocs;
	Size		allocatedBytes;
	PgturbohybridMultiVectorAccumulatorArenaChunk *chunks;
	PgturbohybridMultiVectorAccumulatorArenaChunk *current;
} PgturbohybridMultiVectorAccumulatorArena;

static Size
PgturbohybridMultiVectorAccumulatorArrayBytes(int queryCount)
{
	return add_size(PgturbohybridGraphArrayAllocSize(sizeof(double),
													 (Size) queryCount),
					PgturbohybridGraphArrayAllocSize(sizeof(bool),
													 (Size) queryCount));
}

static uint32
PgturbohybridMultiVectorAccumulatorArenaChunkCapacity(uint64 docCapacity)
{
	uint64		capacity = Max(docCapacity, (uint64) 16);

	capacity = Min(capacity, (uint64) 1024);
	return (uint32) capacity;
}

static Size
PgturbohybridMultiVectorAccumulatorArenaEstimatedBytes(uint64 docCapacity,
													   int queryCount)
{
	uint32		chunkDocCapacity;
	uint64		chunkCount;
	uint64		slabDocs;
	Size		arrayBytes;
	Size		chunkBytes;

	if (docCapacity == 0)
		return 0;
	chunkDocCapacity =
		PgturbohybridMultiVectorAccumulatorArenaChunkCapacity(docCapacity);
	chunkCount =
		(docCapacity + (uint64) chunkDocCapacity - 1) /
		(uint64) chunkDocCapacity;
	if (chunkCount > (uint64) (SIZE_MAX / sizeof(PgturbohybridMultiVectorAccumulatorArenaChunk)))
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("multivector accumulator memory estimate overflow")));
	chunkBytes =
		(Size) chunkCount *
		sizeof(PgturbohybridMultiVectorAccumulatorArenaChunk);
	if (chunkCount > PG_UINT64_MAX / (uint64) chunkDocCapacity)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("multivector accumulator memory estimate overflow")));
	slabDocs = chunkCount * (uint64) chunkDocCapacity;
	arrayBytes = PgturbohybridMultiVectorAccumulatorArrayBytes(queryCount);
	if (slabDocs > (uint64) (SIZE_MAX / arrayBytes))
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("multivector accumulator memory estimate overflow")));

	return add_size(chunkBytes, (Size) slabDocs * arrayBytes);
}

static uint64
PgturbohybridMultiVectorDocCapacity(int rawTarget, int queryCount)
{
	uint64		perToken;

	perToken = (uint64) Min(rawTarget,
							pgturbohybrid_multivector_unique_docs_per_token);
	if (queryCount > 0 &&
		perToken > PG_UINT64_MAX / (uint64) queryCount)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("multivector accumulator candidate count is too large")));

	return Max(perToken * (uint64) queryCount, (uint64) 1);
}

static Size
PgturbohybridMultiVectorAccumulatorBytesEstimate(uint64 docCapacity,
												 int queryCount)
{
	Size		arrayTotal;

	if (docCapacity > (uint64) LONG_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("multivector accumulator candidate count is too large")));

	arrayTotal =
		PgturbohybridMultiVectorAccumulatorArenaEstimatedBytes(docCapacity,
															  queryCount);

	return add_size(hash_estimate_size((long) docCapacity,
									   sizeof(PgturbohybridMultiVectorDocEntry)),
					add_size(hash_estimate_size((long) docCapacity,
												sizeof(PgturbohybridMultiVectorDocIdEntry)),
							 arrayTotal));
}

static void
PgturbohybridMultiVectorCheckAccumulatorMemory(uint64 docCapacity,
											   int queryCount)
{
	Size		estimated;
	uint64		limitBytes;

	estimated = PgturbohybridMultiVectorAccumulatorBytesEstimate(docCapacity,
																 queryCount);
	limitBytes = (uint64) pgturbohybrid_multivector_max_accumulator_mb *
		(uint64) 1024 * (uint64) 1024;
	if ((uint64) estimated > limitBytes)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("multivector accumulator memory estimate %zu bytes exceeds configured limit %d MB",
						estimated,
						pgturbohybrid_multivector_max_accumulator_mb),
				 errhint("Reduce dense_k, turbohybrid.multivector_subvector_k, turbohybrid.multivector_unique_docs_per_token, or turbohybrid.multivector_max_query_vectors.")));
}

static void
PgturbohybridMultiVectorAccumulatorArenaInit(PgturbohybridMultiVectorAccumulatorArena *arena,
											 MemoryContext ctx,
											 int queryCount,
											 uint64 docCapacity)
{
	memset(arena, 0, sizeof(*arena));
	arena->ctx = ctx;
	arena->queryCount = queryCount;
	arena->chunkDocCapacity =
		PgturbohybridMultiVectorAccumulatorArenaChunkCapacity(docCapacity);
}

static PgturbohybridMultiVectorAccumulatorArenaChunk *
PgturbohybridMultiVectorAccumulatorArenaAllocChunk(PgturbohybridMultiVectorAccumulatorArena *arena)
{
	PgturbohybridMultiVectorAccumulatorArenaChunk *chunk;
	Size		slotCount;
	Size		maxsimBytes;
	Size		seenBytes;

	if ((Size) arena->chunkDocCapacity > MaxAllocSize / (Size) arena->queryCount)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("multivector accumulator slab is too large")));
	slotCount = (Size) arena->chunkDocCapacity * (Size) arena->queryCount;
	maxsimBytes = PgturbohybridGraphArrayAllocSize(sizeof(double), slotCount);
	seenBytes = PgturbohybridGraphArrayAllocSize(sizeof(bool), slotCount);

	chunk = MemoryContextAllocZero(arena->ctx, sizeof(*chunk));
	chunk->capacity = arena->chunkDocCapacity;
	chunk->maxsim = MemoryContextAlloc(arena->ctx, maxsimBytes);
	chunk->seen = MemoryContextAllocZero(arena->ctx, seenBytes);
	chunk->next = arena->chunks;
	arena->chunks = chunk;
	arena->current = chunk;
	arena->allocatedBytes =
		add_size(arena->allocatedBytes,
				 add_size(sizeof(*chunk), add_size(maxsimBytes, seenBytes)));

	return chunk;
}

static void
PgturbohybridMultiVectorAccumulatorArenaAllocDoc(PgturbohybridMultiVectorAccumulatorArena *arena,
												 PgturbohybridMultiVectorDocEntry *entry)
{
	PgturbohybridMultiVectorAccumulatorArenaChunk *chunk;
	Size		offset;

	chunk = arena->current;
	if (chunk == NULL || chunk->used >= chunk->capacity)
		chunk = PgturbohybridMultiVectorAccumulatorArenaAllocChunk(arena);

	offset = (Size) chunk->used * (Size) arena->queryCount;
	entry->maxsim = chunk->maxsim + offset;
	entry->seen = chunk->seen + offset;
	memset(entry->seen, 0, sizeof(bool) * (Size) arena->queryCount);
	for (int i = 0; i < arena->queryCount; i++)
		entry->maxsim[i] = -INFINITY;
	chunk->used++;
	arena->allocatedDocs++;
}

static int
PgturbohybridMultiVectorDenseCandidateCompare(const void *a, const void *b)
{
	const TqDenseCandidate *ca = (const TqDenseCandidate *) a;
	const TqDenseCandidate *cb = (const TqDenseCandidate *) b;
	BlockNumber ablock;
	BlockNumber bblock;
	OffsetNumber aoff;
	OffsetNumber boff;

	if (ca->distance < cb->distance)
		return -1;
	if (ca->distance > cb->distance)
		return 1;
	ablock = ItemPointerGetBlockNumber(&ca->heaptid);
	bblock = ItemPointerGetBlockNumber(&cb->heaptid);
	if (ablock < bblock)
		return -1;
	if (ablock > bblock)
		return 1;
	aoff = ItemPointerGetOffsetNumber(&ca->heaptid);
	boff = ItemPointerGetOffsetNumber(&cb->heaptid);
	if (aoff < boff)
		return -1;
	if (aoff > boff)
		return 1;
	return (ca->nodeId > cb->nodeId) - (ca->nodeId < cb->nodeId);
}

static bool
PgturbohybridMultiVectorCandidateWorse(const TqDenseCandidate *a,
									   const TqDenseCandidate *b)
{
	return PgturbohybridMultiVectorDenseCandidateCompare(a, b) > 0;
}

static int
PgturbohybridFloat8DescCompare(const void *a, const void *b)
{
	double		da = *((const double *) a);
	double		db = *((const double *) b);

	if (da > db)
		return -1;
	if (da < db)
		return 1;
	return 0;
}

static double *
PgturbohybridMultiVectorBuildCodewordScoreBoundPrefix(const PgturbohybridMultiVector *query,
													  const float4 *queryWeights,
													  const bool *queryMask,
													  const float *queryCodewordScores,
													  uint32 codebookSize,
													  uint32 *activeTokenCount)
{
	double	   *bounds;
	double	   *prefix;
	uint32		boundCount = 0;

	*activeTokenCount = 0;
	if (query == NULL || queryCodewordScores == NULL || codebookSize == 0)
		return NULL;

	bounds = palloc0(sizeof(double) * (Size) Max(query->count, 1));
	for (int32 qi = 0; qi < query->count; qi++)
	{
		const float *tokenScores;
		double		weight =
			queryWeights != NULL ? (double) queryWeights[qi] : 1.0;
		float		tokenMax = -FLT_MAX;

		if (queryMask != NULL && queryMask[qi])
			continue;
		if (!(weight >= 0.0) || !isfinite(weight))
		{
			pfree(bounds);
			return NULL;
		}
		tokenScores =
			queryCodewordScores + (Size) qi * (Size) codebookSize;
		for (uint32 codeword = 0; codeword < codebookSize; codeword++)
		{
			if (tokenScores[codeword] > tokenMax)
				tokenMax = tokenScores[codeword];
		}
		if (!isfinite(tokenMax))
		{
			pfree(bounds);
			return NULL;
		}
		bounds[boundCount++] = weight * (double) tokenMax;
	}
	if (boundCount == 0)
	{
		pfree(bounds);
		return NULL;
	}
	qsort(bounds, boundCount, sizeof(double), PgturbohybridFloat8DescCompare);
	prefix = palloc0(sizeof(double) * (Size) (boundCount + 1U));
	for (uint32 i = 0; i < boundCount; i++)
		prefix[i + 1U] = prefix[i] + bounds[i];
	pfree(bounds);
	*activeTokenCount = boundCount;
	return prefix;
}

static bool
PgturbohybridMultiVectorPostingScoreCandidateBetter(
													const PgturbohybridMultiVectorPostingScoreCandidate *a,
													const PgturbohybridMultiVectorPostingScoreCandidate *b)
{
	if (a->score > b->score)
		return true;
	if (a->score < b->score)
		return false;
	if (a->docId < b->docId)
		return true;
	if (a->docId > b->docId)
		return false;
	return a->ordinal < b->ordinal;
}

static int
PgturbohybridMultiVectorPrecompactDocScoreCompare(const void *a, const void *b)
{
	const PgturbohybridMultiVectorPrecompactDoc *pa =
		(const PgturbohybridMultiVectorPrecompactDoc *) a;
	const PgturbohybridMultiVectorPrecompactDoc *pb =
		(const PgturbohybridMultiVectorPrecompactDoc *) b;

	if (pa->score > pb->score)
		return -1;
	if (pa->score < pb->score)
		return 1;
	if (pa->coverage > pb->coverage)
		return -1;
	if (pa->coverage < pb->coverage)
		return 1;
	if (pa->docId < pb->docId)
		return -1;
	if (pa->docId > pb->docId)
		return 1;
	return 0;
}

static int
PgturbohybridMultiVectorPrecompactDocCoverageCompare(const void *a,
													 const void *b)
{
	const PgturbohybridMultiVectorPrecompactDoc *pa =
		(const PgturbohybridMultiVectorPrecompactDoc *) a;
	const PgturbohybridMultiVectorPrecompactDoc *pb =
		(const PgturbohybridMultiVectorPrecompactDoc *) b;

	if (pa->coverage > pb->coverage)
		return -1;
	if (pa->coverage < pb->coverage)
		return 1;
	if (pa->score > pb->score)
		return -1;
	if (pa->score < pb->score)
		return 1;
	if (pa->docId < pb->docId)
		return -1;
	if (pa->docId > pb->docId)
		return 1;
	return 0;
}

static void
PgturbohybridMultiVectorPrecompactMarkDoc(bool *keep, uint32 docId,
										  uint32 *unionDocs,
										  uint32 *duplicates)
{
	if (keep[docId])
	{
		(*duplicates)++;
		return;
	}
	keep[docId] = true;
	(*unionDocs)++;
}

static int
PgturbohybridUInt32AscCompare(const void *a, const void *b)
{
	uint32		ia = *((const uint32 *) a);
	uint32		ib = *((const uint32 *) b);

	if (ia < ib)
		return -1;
	if (ia > ib)
		return 1;
	return 0;
}

int
PgturbohybridMultiVectorQuantizedPostingPayloadCompare(const void *a,
													   const void *b)
{
	const PgturbohybridGraphMultiVectorQuantizedPostingEntry *pa =
		(const PgturbohybridGraphMultiVectorQuantizedPostingEntry *) a;
	const PgturbohybridGraphMultiVectorQuantizedPostingEntry *pb =
		(const PgturbohybridGraphMultiVectorQuantizedPostingEntry *) b;

	if (pa->scorePayload > pb->scorePayload)
		return -1;
	if (pa->scorePayload < pb->scorePayload)
		return 1;
	if (pa->docId < pb->docId)
		return -1;
	if (pa->docId > pb->docId)
		return 1;
	if (pa->tokenOrdinal < pb->tokenOrdinal)
		return -1;
	if (pa->tokenOrdinal > pb->tokenOrdinal)
		return 1;
	return 0;
}

int
PgturbohybridMultiVectorQuantizedPostingSignedPayloadCompare(const void *a,
															 const void *b)
{
	const PgturbohybridGraphMultiVectorQuantizedPostingEntry *pa =
		(const PgturbohybridGraphMultiVectorQuantizedPostingEntry *) a;
	const PgturbohybridGraphMultiVectorQuantizedPostingEntry *pb =
		(const PgturbohybridGraphMultiVectorQuantizedPostingEntry *) b;
	int16		paScore =
		PgturbohybridMultiVectorQuantizedInvertedSignedCompactPayload(pa->scorePayload);
	int16		pbScore =
		PgturbohybridMultiVectorQuantizedInvertedSignedCompactPayload(pb->scorePayload);

	if (paScore > pbScore)
		return -1;
	if (paScore < pbScore)
		return 1;
	if (pa->docId < pb->docId)
		return -1;
	if (pa->docId > pb->docId)
		return 1;
	if (pa->tokenOrdinal < pb->tokenOrdinal)
		return -1;
	if (pa->tokenOrdinal > pb->tokenOrdinal)
		return 1;
	return 0;
}

int
PgturbohybridMultiVectorCentroidPostingPayloadCompare(const void *a,
													  const void *b)
{
	const PgturbohybridGraphMultiVectorCentroidPostingEntry *pa =
		(const PgturbohybridGraphMultiVectorCentroidPostingEntry *) a;
	const PgturbohybridGraphMultiVectorCentroidPostingEntry *pb =
		(const PgturbohybridGraphMultiVectorCentroidPostingEntry *) b;

	if (pa->scorePayload > pb->scorePayload)
		return -1;
	if (pa->scorePayload < pb->scorePayload)
		return 1;
	if (pa->docId < pb->docId)
		return -1;
	if (pa->docId > pb->docId)
		return 1;
	if (pa->centroidOrdinal < pb->centroidOrdinal)
		return -1;
	if (pa->centroidOrdinal > pb->centroidOrdinal)
		return 1;
	return 0;
}

static void
PgturbohybridMultiVectorPostingScoreCandidateOffer(
												   PgturbohybridMultiVectorPostingScoreCandidate *selected,
												   uint32 *selectedCount,
												   uint32 limit,
												   const PgturbohybridMultiVectorPostingScoreCandidate *candidate)
{
	uint32		worstIndex = 0;

	Assert(limit > 0);
	if (*selectedCount < limit)
	{
		selected[*selectedCount] = *candidate;
		(*selectedCount)++;
		return;
	}

	for (uint32 i = 1; i < *selectedCount; i++)
	{
		if (PgturbohybridMultiVectorPostingScoreCandidateBetter(
																&selected[worstIndex],
																&selected[i]))
			worstIndex = i;
	}
	if (PgturbohybridMultiVectorPostingScoreCandidateBetter(candidate,
														   &selected[worstIndex]))
		selected[worstIndex] = *candidate;
}

static void
PgturbohybridMultiVectorCandidateHeapSwap(TqDenseCandidate *a,
										  TqDenseCandidate *b)
{
	TqDenseCandidate tmp = *a;

	*a = *b;
	*b = tmp;
}

static void
PgturbohybridMultiVectorCandidateHeapSiftUp(TqDenseCandidate *heap, int index)
{
	while (index > 0)
	{
		int			parent = (index - 1) / 2;

		if (!PgturbohybridMultiVectorCandidateWorse(&heap[index],
													&heap[parent]))
			break;
		PgturbohybridMultiVectorCandidateHeapSwap(&heap[index], &heap[parent]);
		index = parent;
	}
}

static void
PgturbohybridMultiVectorCandidateHeapSiftDown(TqDenseCandidate *heap,
											  int count, int index)
{
	for (;;)
	{
		int			left = index * 2 + 1;
		int			right = left + 1;
		int			worst = index;

		if (left < count &&
			PgturbohybridMultiVectorCandidateWorse(&heap[left], &heap[worst]))
			worst = left;
		if (right < count &&
			PgturbohybridMultiVectorCandidateWorse(&heap[right], &heap[worst]))
			worst = right;
		if (worst == index)
			break;
		PgturbohybridMultiVectorCandidateHeapSwap(&heap[index], &heap[worst]);
		index = worst;
	}
}

static void
PgturbohybridMultiVectorCandidateHeapOffer(TqDenseCandidate *heap,
										   int *count,
										   int limit,
										   const TqDenseCandidate *candidate)
{
	if (*count < limit)
	{
		heap[*count] = *candidate;
		PgturbohybridMultiVectorCandidateHeapSiftUp(heap, *count);
		(*count)++;
	}
	else if (PgturbohybridMultiVectorDenseCandidateCompare(candidate,
														  &heap[0]) < 0)
	{
		heap[0] = *candidate;
		PgturbohybridMultiVectorCandidateHeapSiftDown(heap, *count, 0);
	}
}

static bool
PgturbohybridMultiVectorCentroidLitePruneBySafeUpperBound(TqDenseCandidate *heap,
														  int count,
														  int limit,
														  const TqDenseCandidate *candidate,
														  const PgturbohybridGraphScanStorage *storage,
														  TqDocId docId,
														  uint64 *unsafeFallbacks)
{
	float		residualMean;

	if (count < limit || limit <= 0)
		return false;
	if (storage == NULL ||
		storage->multivectorDocCentroidResiduals == NULL ||
		docId >= storage->multivectorDocCount)
	{
		(*unsafeFallbacks)++;
		return false;
	}

	/*
	 * The persisted centroid residual is a mean summary, not a radius. It is
	 * only a safe upper bound when it is exactly zero, because then the
	 * centroid interaction equals the original document-token interaction for
	 * this document. Otherwise keep the candidate and report the fallback.
	 */
	residualMean = storage->multivectorDocCentroidResiduals[docId];
	if (!isfinite(residualMean) || residualMean > 1.0e-12f)
	{
		(*unsafeFallbacks)++;
		return false;
	}

	return PgturbohybridMultiVectorDenseCandidateCompare(candidate,
														 &heap[0]) >= 0;
}

static void
PgturbohybridMultiVectorCandidateHeapSort(TqDenseCandidate *heap, int count)
{
	if (count > 1)
		qsort(heap, count, sizeof(TqDenseCandidate),
			  PgturbohybridMultiVectorDenseCandidateCompare);
}

static HTAB *
PgturbohybridMultiVectorTokenSeenCreate(MemoryContext ctx, long capacity)
{
	HASHCTL		hashCtl;

	memset(&hashCtl, 0, sizeof(hashCtl));
	hashCtl.keysize = sizeof(TqDocId);
	hashCtl.entrysize = sizeof(PgturbohybridMultiVectorTokenSeenEntry);
	hashCtl.hcxt = ctx;

	return hash_create("pgturbohybrid multivector token seen docs",
					   Max(capacity, 16L),
						   &hashCtl,
						   HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
}

static void
PgturbohybridMultiVectorTokenSeenReset(HTAB **seen,
									   MemoryContext tokenCtx,
									   long capacity)
{
	MemoryContextReset(tokenCtx);
	*seen = PgturbohybridMultiVectorTokenSeenCreate(tokenCtx, capacity);
}

static bool
PgturbohybridMultiVectorTokenSeenAdd(HTAB *seen, TqDocId docId)
{
	bool		found;

	hash_search(seen, &docId, HASH_ENTER, &found);
	return !found;
}

static TqDocId
PgturbohybridMultiVectorResolveDocId(HTAB *docIdHash,
									 const ItemPointerData *heaptid,
									 uint64 *nextDocOrdinal)
{
	PgturbohybridMultiVectorTidKey key;
	PgturbohybridMultiVectorDocIdEntry *entry;
	bool		found;

	memset(&key, 0, sizeof(key));
	key.block = ItemPointerGetBlockNumber(heaptid);
	key.offset = ItemPointerGetOffsetNumber(heaptid);
	entry = (PgturbohybridMultiVectorDocIdEntry *) hash_search(docIdHash, &key,
															  HASH_ENTER,
															  &found);
	if (!found)
	{
		entry->docId = PgturbohybridMultiVectorMakeDocId(*nextDocOrdinal);
		entry->heaptid = *heaptid;
		(*nextDocOrdinal)++;
	}
	return entry->docId;
}

static void
PgturbohybridMultiVectorAccumulateDoc(HTAB *docHash,
									  PgturbohybridMultiVectorAccumulatorArena *arena,
									  const PgturbohybridGraphResult *hit,
									  TqDocId docId,
									  int queryOrdinal, int queryCount,
									  double queryWeight,
									  bool duplicateDocHit,
									  uint64 *maxsimUpdates)
{
	PgturbohybridMultiVectorDocKey key;
	PgturbohybridMultiVectorDocEntry *entry;
	bool		found;
	double		similarity = -hit->distance;
	double		weightedSimilarity;

	if (!isfinite(queryWeight) || queryWeight < 0.0)
		queryWeight = 1.0;
	weightedSimilarity = similarity * queryWeight;

	memset(&key, 0, sizeof(key));
	key.docId = docId;
	entry = (PgturbohybridMultiVectorDocEntry *) hash_search(docHash, &key,
															 HASH_ENTER,
															 &found);
	if (!found)
	{
		entry->docId = docId;
		entry->heaptid = hit->heaptid;
		entry->bestNodeId = hit->nodeId;
		entry->bestSimilarity = -INFINITY;
		entry->score = 0.0;
		entry->matchedTokens = 0;
		entry->rawHitCount = 0;
		entry->duplicateHitCount = 0;
		PgturbohybridMultiVectorAccumulatorArenaAllocDoc(arena, entry);
	}
	entry->rawHitCount++;
	if (duplicateDocHit)
		entry->duplicateHitCount++;

	if (!entry->seen[queryOrdinal])
	{
		entry->seen[queryOrdinal] = true;
		entry->matchedTokens++;
		entry->maxsim[queryOrdinal] = similarity;
		entry->score += weightedSimilarity;
		if (similarity > entry->bestSimilarity)
		{
			entry->bestSimilarity = similarity;
			entry->bestNodeId = hit->nodeId;
		}
		if (maxsimUpdates != NULL)
			(*maxsimUpdates)++;
	}
	else if (similarity > entry->maxsim[queryOrdinal])
	{
		entry->score += (similarity - entry->maxsim[queryOrdinal]) *
			queryWeight;
		entry->maxsim[queryOrdinal] = similarity;
		if (similarity > entry->bestSimilarity)
		{
			entry->bestSimilarity = similarity;
			entry->bestNodeId = hit->nodeId;
		}
		if (maxsimUpdates != NULL)
			(*maxsimUpdates)++;
	}
}

static int
PgturbohybridMultiVectorCompareDocRank(const PgturbohybridMultiVectorDocEntry *a,
									   const PgturbohybridMultiVectorDocEntry *b)
{
	BlockNumber ablock;
	BlockNumber bblock;
	OffsetNumber aoff;
	OffsetNumber boff;

	if (a->score > b->score)
		return -1;
	if (a->score < b->score)
		return 1;
	ablock = ItemPointerGetBlockNumber(&a->heaptid);
	bblock = ItemPointerGetBlockNumber(&b->heaptid);
	if (ablock < bblock)
		return -1;
	if (ablock > bblock)
		return 1;
	aoff = ItemPointerGetOffsetNumber(&a->heaptid);
	boff = ItemPointerGetOffsetNumber(&b->heaptid);
	if (aoff < boff)
		return -1;
	if (aoff > boff)
		return 1;
	return (a->docId > b->docId) - (a->docId < b->docId);
}

static int
PgturbohybridMultiVectorCompareDocRankPtr(const void *a, const void *b)
{
	const PgturbohybridMultiVectorDocEntry *const *pa =
		(const PgturbohybridMultiVectorDocEntry *const *) a;
	const PgturbohybridMultiVectorDocEntry *const *pb =
		(const PgturbohybridMultiVectorDocEntry *const *) b;

	return PgturbohybridMultiVectorCompareDocRank(*pa, *pb);
}

static int
PgturbohybridMultiVectorCompareReservoirRankItem(const void *a, const void *b)
{
	const PgturbohybridMultiVectorReservoirRankItem *ia =
		(const PgturbohybridMultiVectorReservoirRankItem *) a;
	const PgturbohybridMultiVectorReservoirRankItem *ib =
		(const PgturbohybridMultiVectorReservoirRankItem *) b;

	if (ia->value > ib->value)
		return -1;
	if (ia->value < ib->value)
		return 1;
	if (ia->secondary > ib->secondary)
		return -1;
	if (ia->secondary < ib->secondary)
		return 1;
	if (ia->coverage > ib->coverage)
		return -1;
	if (ia->coverage < ib->coverage)
		return 1;
	return PgturbohybridMultiVectorCompareDocRank(ia->doc, ib->doc);
}

static void
PgturbohybridMultiVectorCandidateFromDoc(const PgturbohybridMultiVectorDocEntry *entry,
										 int queryCount,
										 double queryWeightSum,
										 TqDenseCandidate *candidate)
{
	double		divisor = queryWeightSum > 0.0 ? queryWeightSum :
		(double) queryCount;

	memset(candidate, 0, sizeof(*candidate));
	candidate->nodeId = entry->bestNodeId;
	candidate->docId = entry->docId;
	candidate->heaptid = entry->heaptid;
	candidate->distance = -entry->score;
	candidate->similarity = divisor > 0.0 ? entry->score / divisor : 0.0;
	candidate->rank = 0;
	candidate->hasDocId = true;
	candidate->exactScored = false;
}

static HTAB *
PgturbohybridMultiVectorReservoirDedupeCreate(MemoryContext ctx,
											  long capacity)
{
	HASHCTL		hashCtl;

	memset(&hashCtl, 0, sizeof(hashCtl));
	hashCtl.keysize = sizeof(TqDocId);
	hashCtl.entrysize = sizeof(PgturbohybridMultiVectorReservoirDedupeEntry);
	hashCtl.hcxt = ctx;
	return hash_create("pgturbohybrid multivector reservoir docs",
					   Max(capacity, 16L),
					   &hashCtl,
					   HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
}

static bool
PgturbohybridMultiVectorReservoirAddDoc(HTAB *selected,
										TqDenseCandidate *candidates,
										int *docCount,
										int docLimit,
										const PgturbohybridMultiVectorDocEntry *entry,
										int queryCount,
										double queryWeightSum,
										uint32 *added,
										uint32 *duplicates)
{
	PgturbohybridMultiVectorReservoirDedupeEntry *selectedEntry;
	TqDenseCandidate candidate;
	bool		found;

	selectedEntry =
		(PgturbohybridMultiVectorReservoirDedupeEntry *) hash_search(selected,
																	 &entry->docId,
																	 HASH_FIND,
																	 NULL);
	if (selectedEntry != NULL)
	{
		if (duplicates != NULL)
			(*duplicates)++;
		return false;
	}
	if (*docCount >= docLimit)
		return false;

	hash_search(selected, &entry->docId, HASH_ENTER, &found);
	PgturbohybridMultiVectorCandidateFromDoc(entry, queryCount, queryWeightSum,
											 &candidate);
	candidates[*docCount] = candidate;
	(*docCount)++;
	if (added != NULL)
		(*added)++;
	return true;
}

static void
PgturbohybridMultiVectorReservoirAddScoreDocs(PgturbohybridMultiVectorDocEntry **rankedDocs,
											  long rankedDocCount,
											  HTAB *selected,
											  TqDenseCandidate *candidates,
											  int *docCount,
											  int docLimit,
											  int queryCount,
											  double queryWeightSum,
											  int limit,
											  uint32 *added,
											  uint32 *duplicates)
{
	int			seen = 0;

	if (limit <= 0)
		return;
	for (long i = 0; i < rankedDocCount && seen < limit; i++)
	{
		CHECK_FOR_INTERRUPTS();
		if (PgturbohybridMultiVectorReservoirAddDoc(selected, candidates,
													docCount, docLimit,
													rankedDocs[i],
													queryCount,
													queryWeightSum, added,
													duplicates))
			seen++;
		if (*docCount >= docLimit)
			break;
	}
}

static void
PgturbohybridMultiVectorReservoirAddRankItems(PgturbohybridMultiVectorReservoirRankItem *items,
											  long itemCount,
											  HTAB *selected,
											  TqDenseCandidate *candidates,
											  int *docCount,
											  int docLimit,
											  int queryCount,
											  double queryWeightSum,
											  int limit,
											  uint32 *added,
											  uint32 *duplicates)
{
	int			seen = 0;

	if (limit <= 0 || itemCount <= 0)
		return;
	qsort(items, itemCount, sizeof(*items),
		  PgturbohybridMultiVectorCompareReservoirRankItem);
	for (long i = 0; i < itemCount && seen < limit; i++)
	{
		CHECK_FOR_INTERRUPTS();
		if (PgturbohybridMultiVectorReservoirAddDoc(selected, candidates,
													docCount, docLimit,
													items[i].doc,
													queryCount,
													queryWeightSum, added,
													duplicates))
			seen++;
		if (*docCount >= docLimit)
			break;
	}
}

static void
PgturbohybridMultiVectorReservoirAddPerTokenDocs(PgturbohybridMultiVectorDocEntry **rankedDocs,
												long rankedDocCount,
												HTAB *selected,
												TqDenseCandidate *candidates,
												int *docCount,
												int docLimit,
												int queryCount,
												double queryWeightSum,
												int perTokenLimit,
												PgturbohybridMultiVectorReservoirRankItem *items,
												uint32 *added,
												uint32 *duplicates)
{
	if (perTokenLimit <= 0 || rankedDocCount <= 0)
		return;

	for (int qi = 0; qi < queryCount && *docCount < docLimit; qi++)
	{
		long		itemCount = 0;
		int			considered = 0;

		for (long i = 0; i < rankedDocCount; i++)
		{
			if (!rankedDocs[i]->seen[qi])
				continue;
			items[itemCount].doc = rankedDocs[i];
			items[itemCount].value = rankedDocs[i]->maxsim[qi];
			items[itemCount].secondary = rankedDocs[i]->score;
			items[itemCount].coverage = rankedDocs[i]->matchedTokens;
			itemCount++;
		}
		if (itemCount <= 0)
			continue;
		qsort(items, itemCount, sizeof(*items),
			  PgturbohybridMultiVectorCompareReservoirRankItem);
		for (long i = 0; i < itemCount && considered < perTokenLimit; i++)
		{
			CHECK_FOR_INTERRUPTS();
			PgturbohybridMultiVectorReservoirAddDoc(selected, candidates,
													docCount, docLimit,
													items[i].doc,
													queryCount,
													queryWeightSum, added,
													duplicates);
			considered++;
			if (*docCount >= docLimit)
				break;
		}
	}
}

static void
PgturbohybridMultiVectorBuildReservoirCandidates(HTAB *docHash,
												 int queryCount,
												 double queryWeightSum,
												 int docLimit,
												 TqDenseCandidate *candidates,
												 int *docCount,
												 MemoryContext ctx,
												 uint32 *scoreDocs,
												 uint32 *coverageDocs,
												 uint32 *meanDocs,
												 uint32 *perTokenDocs,
												 uint32 *bm25Docs,
												 uint32 *unionDocs,
												 uint32 *duplicates)
{
	HASH_SEQ_STATUS seq;
	PgturbohybridMultiVectorDocEntry *entry;
	PgturbohybridMultiVectorDocEntry **rankedDocs;
	PgturbohybridMultiVectorReservoirRankItem *items;
	HTAB	   *selected;
	long		docEntryCount;
	long		rankedDocCount = 0;
	int			mode = pgturbohybrid_multivector_candidate_reservoirs;
	int			coverageLimit;
	int			meanLimit;
	int			perTokenLimit;
	int			reserved;
	int			scoreLimit;

	*docCount = 0;
	if (scoreDocs != NULL)
		*scoreDocs = 0;
	if (coverageDocs != NULL)
		*coverageDocs = 0;
	if (meanDocs != NULL)
		*meanDocs = 0;
	if (perTokenDocs != NULL)
		*perTokenDocs = 0;
	if (bm25Docs != NULL)
		*bm25Docs = 0;
	if (unionDocs != NULL)
		*unionDocs = 0;
	if (duplicates != NULL)
		*duplicates = 0;

	docEntryCount = hash_get_num_entries(docHash);
	if (docEntryCount <= 0 || docLimit <= 0)
		return;

	rankedDocs = MemoryContextAlloc(ctx, sizeof(*rankedDocs) * docEntryCount);
	items = MemoryContextAlloc(ctx, sizeof(*items) * docEntryCount);
	hash_seq_init(&seq, docHash);
	while ((entry = (PgturbohybridMultiVectorDocEntry *) hash_seq_search(&seq)) != NULL)
	{
		CHECK_FOR_INTERRUPTS();
		rankedDocs[rankedDocCount++] = entry;
	}
	qsort(rankedDocs, rankedDocCount, sizeof(*rankedDocs),
		  PgturbohybridMultiVectorCompareDocRankPtr);

	selected = PgturbohybridMultiVectorReservoirDedupeCreate(ctx, docLimit);
	coverageLimit = Min(pgturbohybrid_multivector_coverage_reservoir_k,
						docLimit);
	meanLimit = Min(pgturbohybrid_multivector_coverage_reservoir_k,
					docLimit);
	perTokenLimit = Min(pgturbohybrid_multivector_per_token_doc_reservoir_k,
						docLimit);
	if (mode == PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_RESERVOIRS_BALANCED)
		scoreLimit = Max(1, docLimit / 2);
	else
	{
		reserved = Min(docLimit / 2,
					   coverageLimit + meanLimit +
					   perTokenLimit * Max(queryCount, 1));
		scoreLimit = Max(1, docLimit - reserved);
	}

	PgturbohybridMultiVectorReservoirAddScoreDocs(rankedDocs, rankedDocCount,
												 selected, candidates,
												 docCount, docLimit,
												 queryCount,
												 queryWeightSum, scoreLimit,
												 scoreDocs, duplicates);

	if (*docCount < docLimit && perTokenLimit > 0)
		PgturbohybridMultiVectorReservoirAddPerTokenDocs(rankedDocs,
														 rankedDocCount,
														 selected, candidates,
														 docCount, docLimit,
														 queryCount,
														 queryWeightSum,
														 perTokenLimit,
														 items,
														 perTokenDocs,
														 duplicates);

	if (*docCount < docLimit && coverageLimit > 0)
	{
		for (long i = 0; i < rankedDocCount; i++)
		{
			items[i].doc = rankedDocs[i];
			items[i].value = (double) rankedDocs[i]->matchedTokens;
			items[i].secondary = rankedDocs[i]->score;
			items[i].coverage = rankedDocs[i]->matchedTokens;
		}
		PgturbohybridMultiVectorReservoirAddRankItems(items, rankedDocCount,
													 selected, candidates,
													 docCount, docLimit,
													 queryCount,
													 queryWeightSum,
													 coverageLimit,
													 coverageDocs,
													 duplicates);
	}

	if (*docCount < docLimit && meanLimit > 0)
	{
		for (long i = 0; i < rankedDocCount; i++)
		{
			items[i].doc = rankedDocs[i];
			items[i].value = rankedDocs[i]->matchedTokens > 0 ?
				rankedDocs[i]->score / (double) rankedDocs[i]->matchedTokens :
				-INFINITY;
			items[i].secondary = rankedDocs[i]->score;
			items[i].coverage = rankedDocs[i]->matchedTokens;
		}
		PgturbohybridMultiVectorReservoirAddRankItems(items, rankedDocCount,
													 selected, candidates,
													 docCount, docLimit,
													 queryCount,
													 queryWeightSum, meanLimit,
													 meanDocs, duplicates);
	}

	if (*docCount < docLimit)
		PgturbohybridMultiVectorReservoirAddScoreDocs(rankedDocs,
													 rankedDocCount,
													 selected, candidates,
													 docCount, docLimit,
													 queryCount,
													 queryWeightSum, docLimit,
													 scoreDocs, duplicates);
	if (unionDocs != NULL)
		*unionDocs = (uint32) *docCount;
}

static bool
PgturbohybridMultiVectorProxyReservoirCandidateSelected(const TqDenseCandidate *selected,
														int selectedCount,
														const TqDenseCandidate *candidate)
{
	for (int i = 0; i < selectedCount; i++)
	{
		if (selected[i].hasDocId && candidate->hasDocId &&
			selected[i].docId == candidate->docId)
			return true;
		if (ItemPointerIsValid(&selected[i].heaptid) &&
			ItemPointerIsValid(&candidate->heaptid) &&
			ItemPointerGetBlockNumber(&selected[i].heaptid) ==
			ItemPointerGetBlockNumber(&candidate->heaptid) &&
			ItemPointerGetOffsetNumber(&selected[i].heaptid) ==
			ItemPointerGetOffsetNumber(&candidate->heaptid))
			return true;
	}
	return false;
}

static bool
PgturbohybridMultiVectorProxyReservoirAddCandidate(const TqDenseCandidate *candidates,
												  int candidateIndex,
												  int candidateCount,
												  TqDenseCandidate *selected,
												  int *selectedCount,
												  int selectedLimit,
												  uint32 *added,
												  uint32 *duplicates)
{
	const TqDenseCandidate *candidate;

	if (candidateIndex < 0 || candidateIndex >= candidateCount ||
		*selectedCount >= selectedLimit)
		return false;
	candidate = &candidates[candidateIndex];
	if (!candidate->hasDocId && !ItemPointerIsValid(&candidate->heaptid))
		return false;
	if (PgturbohybridMultiVectorProxyReservoirCandidateSelected(selected,
															   *selectedCount,
															   candidate))
	{
		if (duplicates != NULL)
			(*duplicates)++;
		return false;
	}

	selected[*selectedCount] = *candidate;
	(*selectedCount)++;
	if (added != NULL)
		(*added)++;
	return true;
}

static int
PgturbohybridMultiVectorApplyProxyReservoirCandidates(TqDenseCandidate *candidates,
													 int candidateCount,
													 int rerankLimit,
													 MemoryContext ctx,
													 uint32 *scoreDocs,
													 uint32 *coverageDocs,
													 uint32 *meanDocs,
													 uint32 *perTokenDocs,
													 uint32 *unionDocs,
													 uint32 *duplicates)
{
	TqDenseCandidate *selected;
	bool	   *selectedByIndex;
	int			selectedCount = 0;
	int			scoreLimit;
	int			spreadLimit;
	int			mode = pgturbohybrid_multivector_candidate_reservoirs;

	if (scoreDocs != NULL)
		*scoreDocs = 0;
	if (coverageDocs != NULL)
		*coverageDocs = 0;
	if (meanDocs != NULL)
		*meanDocs = 0;
	if (perTokenDocs != NULL)
		*perTokenDocs = 0;
	if (unionDocs != NULL)
		*unionDocs = 0;
	if (duplicates != NULL)
		*duplicates = 0;
	if (candidates == NULL || candidateCount <= 0 || rerankLimit <= 0)
		return 0;

	rerankLimit = Min(rerankLimit, candidateCount);
	selected = MemoryContextAlloc(ctx, sizeof(*selected) * (Size) rerankLimit);
	selectedByIndex = MemoryContextAllocZero(ctx,
											sizeof(*selectedByIndex) *
											(Size) candidateCount);

	/*
	 * Document-node proxy candidates have one proxy score per document, not
	 * token-local coverage evidence.  Conservative mode keeps most of the
	 * proxy-ranked prefix and uses a small deterministic rank spread. Balanced
	 * mode keeps the best proxy-ranked half and fills the remaining
	 * exact-rerank band with the same spread.  This is an opt-in admission
	 * experiment; exact MaxSim remains the final rerank score.
	 */
	if (mode == PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_RESERVOIRS_BALANCED)
		scoreLimit = Max(1, rerankLimit / 2);
	else
		scoreLimit = Max(1, (rerankLimit * 3) / 4);
	spreadLimit = rerankLimit - scoreLimit;

	for (int i = 0; i < candidateCount && selectedCount < scoreLimit; i++)
	{
		if (PgturbohybridMultiVectorProxyReservoirAddCandidate(candidates,
															  i,
															  candidateCount,
															  selected,
															  &selectedCount,
															  rerankLimit,
															  scoreDocs,
															  duplicates))
			selectedByIndex[i] = true;
	}

	for (int i = 0; i < spreadLimit && selectedCount < rerankLimit; i++)
	{
		int			index;

		if (spreadLimit <= 1)
			index = candidateCount - 1;
		else
			index = (int) (((int64) i * (int64) (candidateCount - 1)) /
						   (int64) (spreadLimit - 1));
		if (index < scoreLimit && candidateCount > scoreLimit)
			index = scoreLimit +
				(int) (((int64) i * (int64) (candidateCount - scoreLimit - 1)) /
					   (int64) Max(spreadLimit - 1, 1));
		if (PgturbohybridMultiVectorProxyReservoirAddCandidate(candidates,
															  index,
															  candidateCount,
															  selected,
															  &selectedCount,
															  rerankLimit,
															  coverageDocs,
															  duplicates))
			selectedByIndex[index] = true;
	}

	for (int i = 0; i < candidateCount && selectedCount < rerankLimit; i++)
	{
		if (selectedByIndex[i])
			continue;
		if (PgturbohybridMultiVectorProxyReservoirAddCandidate(candidates,
															  i,
															  candidateCount,
															  selected,
															  &selectedCount,
															  rerankLimit,
															  scoreDocs,
															  duplicates))
			selectedByIndex[i] = true;
	}

	if (selectedCount > 0)
	{
		TqDenseCandidate *reordered;
		int			writeIndex = 0;

		reordered = MemoryContextAlloc(ctx,
									   sizeof(*reordered) *
									   (Size) candidateCount);
		for (int i = 0; i < selectedCount; i++)
			reordered[writeIndex++] = selected[i];
		for (int i = 0; i < candidateCount; i++)
		{
			if (selectedByIndex[i])
				continue;
			reordered[writeIndex++] = candidates[i];
		}
		memcpy(candidates, reordered, sizeof(*candidates) * (Size) candidateCount);
	}

	if (unionDocs != NULL)
		*unionDocs = (uint32) selectedCount;
	return selectedCount;
}

static bool
PgturbohybridMultiVectorSameHeapTid(const ItemPointerData *a,
									const ItemPointerData *b)
{
	return ItemPointerGetBlockNumber(a) == ItemPointerGetBlockNumber(b) &&
		ItemPointerGetOffsetNumber(a) == ItemPointerGetOffsetNumber(b);
}

static void
PgturbohybridMultiVectorFillAdmissionTraceEntry(const PgturbohybridMultiVectorDocEntry *doc,
												uint32 candidateRankBeforeTruncation,
												TqDenseCandidate *candidates,
												int docCount,
												int exactRerankLimit,
												PgturbohybridMultiVectorAdmissionTraceEntry *trace)
{
	memset(trace, 0, sizeof(*trace));
	trace->docId = doc->docId;
	trace->block = ItemPointerGetBlockNumber(&doc->heaptid);
	trace->offset = ItemPointerGetOffsetNumber(&doc->heaptid);
	trace->bestNodeId = doc->bestNodeId;
	trace->approximateScoreBeforeRerank = doc->score;
	trace->queryTokenCoverageCount = (uint32) Max(doc->matchedTokens, 0);
	trace->rawHitCount = doc->rawHitCount;
	trace->duplicateHitCount = doc->duplicateHitCount;
	trace->candidateRankBeforeTruncation =
		candidateRankBeforeTruncation;

	for (int i = 0; i < docCount; i++)
	{
		if (!PgturbohybridMultiVectorSameHeapTid(&candidates[i].heaptid,
												 &doc->heaptid))
			continue;

		trace->retainedForExactRerank = i < exactRerankLimit;
		if (candidates[i].exactScored)
		{
			trace->exactRerankScoreAvailable = true;
			trace->exactRerankScore = -candidates[i].distance;
		}
		break;
	}
}

static uint32
PgturbohybridMultiVectorBuildAdmissionTrace(HTAB *docHash,
											TqDenseCandidate *candidates,
											int docCount,
											int exactRerankLimit,
											PgturbohybridMultiVectorAdmissionTraceEntry *trace,
											uint32 traceLimit)
{
	HASH_SEQ_STATUS seq;
	PgturbohybridMultiVectorDocEntry *entry;
	PgturbohybridMultiVectorDocEntry **rankedDocs;
	long		docEntryCount;
	long		rankedDocCount = 0;
	uint32		traceCount = 0;

	if (docHash == NULL || trace == NULL || traceLimit == 0)
		return 0;
	docEntryCount = hash_get_num_entries(docHash);
	if (docEntryCount <= 0)
		return 0;

	rankedDocs = palloc(sizeof(*rankedDocs) * docEntryCount);

	hash_seq_init(&seq, docHash);
	while ((entry = (PgturbohybridMultiVectorDocEntry *) hash_seq_search(&seq)) != NULL)
	{
		CHECK_FOR_INTERRUPTS();
		rankedDocs[rankedDocCount++] = entry;
	}
	qsort(rankedDocs, rankedDocCount, sizeof(*rankedDocs),
		  PgturbohybridMultiVectorCompareDocRankPtr);

	for (long i = 0; i < rankedDocCount; i++)
	{
		CHECK_FOR_INTERRUPTS();
		if (traceCount >= traceLimit)
			break;
		PgturbohybridMultiVectorFillAdmissionTraceEntry(rankedDocs[i],
														(uint32) i + 1,
														candidates, docCount,
														exactRerankLimit,
														&trace[traceCount]);
		traceCount++;
	}
	pfree(rankedDocs);
	return traceCount;
}

static void
PgturbohybridMultiVectorParseSkipQueryTokens(const char *value,
											 int queryCount,
											 bool *skipQueryToken,
											 uint32 *skippedTokenCount)
{
	const char *p;

	if (skippedTokenCount != NULL)
		*skippedTokenCount = 0;
	if (value == NULL || skipQueryToken == NULL)
		return;

	p = value;
	while (*p != '\0')
	{
		uint64		ordinal = 0;

		while (isspace((unsigned char) *p))
			p++;
		if (*p == '\0')
			break;
		if (!isdigit((unsigned char) *p))
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("invalid multivector debug skip query token list"),
					 errdetail("Use comma-separated zero-based query token ordinals.")));
		while (isdigit((unsigned char) *p))
		{
			uint32		digit = (uint32) (*p - '0');

			if (ordinal > (UINT64_MAX - digit) / 10)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("multivector debug skip query token ordinal is too large")));
			ordinal = ordinal * 10 + digit;
			p++;
		}
		while (isspace((unsigned char) *p))
			p++;
		if (*p != '\0' && *p != ',')
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("invalid multivector debug skip query token list"),
					 errdetail("Use comma-separated zero-based query token ordinals.")));
		if (ordinal >= (uint64) queryCount)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("multivector debug skip query token ordinal %llu is out of range for query with %d vectors",
							(unsigned long long) ordinal, queryCount)));
		if (!skipQueryToken[ordinal])
		{
			skipQueryToken[ordinal] = true;
			if (skippedTokenCount != NULL)
				(*skippedTokenCount)++;
		}
		if (*p == ',')
		{
			p++;
			while (isspace((unsigned char) *p))
				p++;
			if (*p == '\0' || *p == ',')
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("invalid multivector debug skip query token list"),
						 errdetail("Empty entries are not allowed.")));
		}
	}
}

static void
PgturbohybridMultiVectorUpdateTokenCandidateStats(HTAB *docHash,
												 TqDenseCandidate *candidates,
												 int docCount,
												 int queryCount,
												 PgturbohybridMultiVectorTokenStatsEntry *tokenStats,
												 uint32 tokenStatsCount)
{
	HASH_SEQ_STATUS seq;
	PgturbohybridMultiVectorDocEntry *entry;

	if (docHash == NULL || candidates == NULL || docCount <= 0 ||
		tokenStats == NULL || tokenStatsCount == 0)
		return;

	hash_seq_init(&seq, docHash);
	while ((entry = (PgturbohybridMultiVectorDocEntry *) hash_seq_search(&seq)) != NULL)
	{
		bool		retained = false;

		CHECK_FOR_INTERRUPTS();
		for (int i = 0; i < docCount; i++)
		{
			if (PgturbohybridMultiVectorSameHeapTid(&entry->heaptid,
													&candidates[i].heaptid))
			{
				retained = true;
				break;
			}
		}
		if (!retained)
			continue;

		for (int qi = 0; qi < queryCount && qi < (int) tokenStatsCount; qi++)
		{
			if (!entry->seen[qi])
				continue;
			tokenStats[qi].candidateDocsRetainedFromToken++;
			tokenStats[qi].contributionToTopCandidates += entry->maxsim[qi];
		}
	}
}

static int
PgturbohybridMultiVectorExactTokenScan(Relation index,
									   PgturbohybridGraphScanOpaque so,
									   PgturbohybridGraphMetaPageData *meta,
									   PgturbohybridGraphScanStorage *storage,
									   Datum query,
									   int rawTarget,
									   PgturbohybridGraphResult *hits,
									   uint64 *nodesScored)
{
	uint32	   *batchNodeIds;
	double	   *batchDistances;
	int			batchCapacity;
	int			batchCount = 0;
	int			hitCount = 0;

	if (rawTarget <= 0)
		return 0;

	batchCapacity = Max(pgturbohybrid_dense_graph_batch_size, 32);
	batchCapacity = Min(batchCapacity, 1024);
	batchNodeIds =
		palloc(sizeof(uint32) * (Size) batchCapacity);
	batchDistances =
		palloc(sizeof(double) * (Size) batchCapacity);

	for (uint32 nodeId = 0; nodeId < meta->tqNodeCount; nodeId++)
	{
		PgturbohybridGraphScanNode *node;

		CHECK_FOR_INTERRUPTS();
		if (!PgturbohybridGraphLoadCodePage(index, so, meta, storage, nodeId))
			ereport(ERROR,
					(errcode(ERRCODE_INDEX_CORRUPTED),
					 errmsg("could not load multivector exact token scan node %u",
							nodeId)));
		node = &storage->nodes[nodeId];
		if ((node->flags & PGTURBOHYBRID_GRAPH_NODE_DEAD) != 0)
			continue;

		batchNodeIds[batchCount++] = nodeId;
		if (batchCount == batchCapacity)
		{
			PgturbohybridGraphScoreNodeBatchTimed(so, storage, batchNodeIds,
												 batchCount, batchDistances,
												 query);
			if (nodesScored != NULL)
				*nodesScored += (uint64) batchCount;
			for (int i = 0; i < batchCount; i++)
			{
				PgturbohybridGraphScanNode *scoredNode =
					&storage->nodes[batchNodeIds[i]];

				PgturbohybridGraphOfferCandidate(so, hits, rawTarget,
												 &hitCount, batchNodeIds[i],
												 &scoredNode->heaptid,
												 batchDistances[i], false);
			}
			batchCount = 0;
		}
	}

	if (batchCount > 0)
	{
		PgturbohybridGraphScoreNodeBatchTimed(so, storage, batchNodeIds,
											 batchCount, batchDistances,
											 query);
		if (nodesScored != NULL)
			*nodesScored += (uint64) batchCount;
		for (int i = 0; i < batchCount; i++)
		{
			PgturbohybridGraphScanNode *scoredNode =
				&storage->nodes[batchNodeIds[i]];

			PgturbohybridGraphOfferCandidate(so, hits, rawTarget,
											 &hitCount, batchNodeIds[i],
											 &scoredNode->heaptid,
											 batchDistances[i], false);
		}
	}

	if (hitCount > 1)
		qsort(hits, hitCount, sizeof(PgturbohybridGraphResult),
			  PgturbohybridGraphResultCompare);
	pfree(batchDistances);
	pfree(batchNodeIds);
	return hitCount;
}

const char *
PgturbohybridMultiVectorDocStorageKindName(int kind)
{
	switch (kind)
	{
		case PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F16:
			return "f16";
		case PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_SQ8:
			return "sq8";
		case PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CENTROID_ONLY:
			return "centroid_only";
		case PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_PROXY_ONLY:
			return "proxy_only";
		case PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32:
		default:
			return "f32";
	}
}

static const char *
PgturbohybridMultiVectorDocStorageCacheModeName(int mode)
{
	switch ((PgturbohybridMultiVectorDocStorageCacheMode) mode)
	{
		case PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_RESIDENT:
			return "resident";
		case PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_PAGED:
			return "paged";
		case PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_AUTO:
		default:
			return "auto";
	}
}

static bool
PgturbohybridMultiVectorIndexUsesContextLevel(Relation index)
{
	PgturbohybridOptions *opts;

	if (index == NULL)
		return false;
	opts = (PgturbohybridOptions *) index->rd_options;
	return opts != NULL &&
		opts->multivectorContextMode ==
		PGTURBOHYBRID_MULTIVECTOR_CONTEXT_MODE_CONTEXT_LEVEL;
}

static double
PgturbohybridMultiVectorIndexMaxSim(Relation index,
									const PgturbohybridMultiVector *query,
									const PgturbohybridMultiVector *doc,
									const float4 *queryWeights,
									const bool *queryMask)
{
	if (PgturbohybridMultiVectorIndexUsesContextLevel(index))
		return TqMultiVectorMaxSimContextLevelWeighted(query, doc,
													   queryWeights,
													   queryMask);
	return TqMultiVectorMaxSimWeighted(query, doc, queryWeights, queryMask);
}

static int
PgturbohybridMultiVectorChooseDocStorageCacheMode(PgturbohybridGraphMetaPageData *meta)
{
	Size		cacheMaxBytes =
		(Size) pgturbohybrid_native_cache_max_mb * 1024 * 1024;

	switch ((PgturbohybridMultiVectorDocStorageCacheMode)
			pgturbohybrid_multivector_doc_storage_cache)
	{
		case PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_RESIDENT:
		case PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_PAGED:
			return pgturbohybrid_multivector_doc_storage_cache;
		case PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_AUTO:
		default:
			if (pgturbohybrid_profile == PGTURBOHYBRID_PROFILE_LATENCY &&
				cacheMaxBytes > 0 &&
				(Size) meta->tqMultivectorDocMapBytes <= cacheMaxBytes)
				return PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_RESIDENT;
			return PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_PAGED;
	}
}

static void
PgturbohybridMultiVectorDocTokenTotals(PgturbohybridGraphScanStorage *storage,
									   uint64 *originalTokens,
									   uint64 *pooledTokens)
{
	uint64		original = 0;
	uint64		pooled = 0;

	if (storage == NULL || !storage->multivectorDocMapLoaded ||
		storage->multivectorDocMap == NULL)
	{
		*originalTokens = 0;
		*pooledTokens = 0;
		return;
	}

	for (uint32 docId = 0; docId < storage->multivectorDocCount; docId++)
	{
		TqMultiVectorDocMapEntry *entry = &storage->multivectorDocMap[docId];
		uint16		entryOriginal =
			entry->originalTokenCount != 0 ?
			entry->originalTokenCount : entry->tokenCount;
		uint16		entryPooled =
			entry->pooledTokenCount != 0 ?
			entry->pooledTokenCount : entry->tokenCount;

		original += entryOriginal;
		pooled += entryPooled;
	}
	*originalTokens = original;
	*pooledTokens = pooled;
}

typedef struct PgturbohybridMultiVectorDocCompactStorage
{
	int			kind;
	uint32		docCount;
	int32		dim;
	uint16	  **f16Values;
	int8	  **sq8Values;
	float	   *sq8Scales;
	uint64		bytes;
} PgturbohybridMultiVectorDocCompactStorage;

typedef struct PgturbohybridMultiVectorDocCompactScoreCache
{
	bool	   *valid;
	double	   *distances;
	uint32		docCount;
	uint64		hits;
	uint64		misses;
} PgturbohybridMultiVectorDocCompactScoreCache;

static Size
PgturbohybridMultiVectorDocCompactArraySize(Size elemSize, Size count)
{
	if (elemSize != 0 && count > MaxAllocSize / elemSize)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("multivector document compact sidecar allocation is too large")));
	return elemSize * count;
}

static PgturbohybridMultiVectorDocCompactStorage *
PgturbohybridMultiVectorInitDocCompactStorage(PgturbohybridGraphMetaPageData *meta,
											  int kind)
{
	PgturbohybridMultiVectorDocCompactStorage *compact;

	if (kind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32)
		return NULL;
	if (kind != PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F16 &&
		kind != PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_SQ8)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("unsupported multivector document storage mode")));

	compact = palloc0(sizeof(PgturbohybridMultiVectorDocCompactStorage));
	compact->kind = kind;
	compact->docCount = meta->tqMultivectorDocCount;
	compact->dim = meta->dimensions;
	compact->bytes = sizeof(PgturbohybridMultiVectorDocCompactStorage);
	return compact;
}

uint16
PgturbohybridMultiVectorFloatToHalf(float value)
{
	uint32		bits;
	uint32		sign;
	int32		exp;
	uint32		mant;

	memcpy(&bits, &value, sizeof(bits));
	sign = (bits >> 16) & 0x8000U;
	exp = (int32) ((bits >> 23) & 0xffU) - 127 + 15;
	mant = bits & 0x7fffffU;

	if (exp <= 0)
	{
		if (exp < -10)
			return (uint16) sign;
		mant = (mant | 0x800000U) >> (1 - exp);
		return (uint16) (sign | ((mant + 0x1000U) >> 13));
	}
	if (exp >= 31)
	{
		if (mant == 0)
			return (uint16) (sign | 0x7c00U);
		return (uint16) (sign | 0x7c00U | (mant >> 13) | 1U);
	}

	mant += 0x1000U;
	if (mant & 0x800000U)
	{
		mant = 0;
		exp++;
		if (exp >= 31)
			return (uint16) (sign | 0x7c00U);
	}
	return (uint16) (sign | ((uint32) exp << 10) | (mant >> 13));
}

static float
PgturbohybridMultiVectorHalfToFloat(uint16 half)
{
	uint32		sign = ((uint32) half & 0x8000U) << 16;
	uint32		exp = ((uint32) half >> 10) & 0x1fU;
	uint32		mant = (uint32) half & 0x03ffU;
	uint32		bits;
	float		value;

	if (exp == 0)
	{
		if (mant == 0)
			bits = sign;
		else
		{
			exp = 1;
			while ((mant & 0x0400U) == 0)
			{
				mant <<= 1;
				exp--;
			}
			mant &= 0x03ffU;
			bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
		}
	}
	else if (exp == 31)
		bits = sign | 0x7f800000U | (mant << 13);
	else
		bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);

	memcpy(&value, &bits, sizeof(value));
	return value;
}

float
PgturbohybridMultiVectorDocCompactSq8Scale(const PgturbohybridMultiVector *doc)
{
	Size		valueCount;
	float		maxAbs = 0.0f;

	valueCount = PgturbohybridMultiVectorFloatCount(doc->count, doc->dim);
	for (Size i = 0; i < valueCount; i++)
		maxAbs = Max(maxAbs, fabsf(doc->values[i]));
	return maxAbs > 0.0f ? maxAbs / 127.0f : 1.0f;
}

static PgturbohybridMultiVectorDocCompactStorage *
PgturbohybridMultiVectorBuildDocCompactStorage(PgturbohybridGraphMetaPageData *meta,
											   PgturbohybridGraphScanStorage *storage,
											   int kind)
{
	PgturbohybridMultiVectorDocCompactStorage *compact;
	Size		docPointerBytes;

	compact = PgturbohybridMultiVectorInitDocCompactStorage(meta, kind);
	if (compact == NULL)
		return NULL;

	docPointerBytes =
		PgturbohybridMultiVectorDocCompactArraySize(sizeof(void *),
													meta->tqMultivectorDocCount);
	if (kind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F16)
	{
		compact->f16Values = palloc0(docPointerBytes);
		compact->bytes += docPointerBytes;
	}
	else if (kind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_SQ8)
	{
		compact->sq8Values = palloc0(docPointerBytes);
		compact->sq8Scales = palloc0(
			PgturbohybridMultiVectorDocCompactArraySize(sizeof(float),
														meta->tqMultivectorDocCount));
		compact->bytes += docPointerBytes;
		compact->bytes +=
			(uint64) sizeof(float) * (uint64) meta->tqMultivectorDocCount;
	}

	for (uint32 docId = 0; docId < meta->tqMultivectorDocCount; docId++)
	{
		PgturbohybridMultiVector *doc = storage->multivectorDocVectors[docId];
		Size		valueCount;
		Size		valueBytes;

		CHECK_FOR_INTERRUPTS();
		if (doc == NULL)
			ereport(ERROR,
					(errcode(ERRCODE_INDEX_CORRUPTED),
					 errmsg("document-node multivector sidecar is missing a document vector"),
					 errhint("REINDEX the index to rebuild the document-node multivector sidecar.")));
		if (doc->dim != meta->dimensions)
			ereport(ERROR,
					(errcode(ERRCODE_INDEX_CORRUPTED),
					 errmsg("document-node multivector sidecar has inconsistent dimensions"),
					 errhint("REINDEX the index to rebuild the document-node multivector sidecar.")));

		valueCount = PgturbohybridMultiVectorFloatCount(doc->count, doc->dim);
		if (kind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F16)
		{
			valueBytes =
				PgturbohybridMultiVectorDocCompactArraySize(sizeof(uint16),
															valueCount);
			compact->f16Values[docId] = palloc(valueBytes);
			for (Size i = 0; i < valueCount; i++)
				compact->f16Values[docId][i] =
					PgturbohybridMultiVectorFloatToHalf(doc->values[i]);
			compact->bytes += valueBytes;
		}
		else
		{
			float		scale;

			valueBytes =
				PgturbohybridMultiVectorDocCompactArraySize(sizeof(int8),
															valueCount);
			compact->sq8Values[docId] = palloc(valueBytes);
			scale = PgturbohybridMultiVectorDocCompactSq8Scale(doc);
			compact->sq8Scales[docId] = scale;
			for (Size i = 0; i < valueCount; i++)
			{
				int			quantized = (int) lrintf(doc->values[i] / scale);

				quantized = Max(-127, Min(127, quantized));
				compact->sq8Values[docId][i] = (int8) quantized;
			}
			compact->bytes += valueBytes;
		}
	}

	return compact;
}

static double
PgturbohybridMultiVectorDocCompactMaxSimRange(const PgturbohybridMultiVector *query,
											  PgturbohybridMultiVectorDocCompactStorage *compact,
											  PgturbohybridMultiVector *doc,
											  TqDocId docId,
											  int32 startToken,
											  int32 endToken,
											  const float4 *queryWeights,
											  const bool *queryMask,
											  const uint16 *pagedF16Values,
											  const int8 *pagedSq8Values,
											  float pagedSq8Scale)
{
	double		score = 0.0;

	Assert(compact != NULL);
	if (doc == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INDEX_CORRUPTED),
				 errmsg("document-node multivector sidecar is missing a document vector"),
				 errhint("REINDEX the index to rebuild the document-node multivector sidecar.")));
	PgturbohybridCheckSameMultiVectorDims(query, doc);
	if (startToken < 0 || endToken > doc->count || startToken >= endToken)
		ereport(ERROR,
				(errcode(ERRCODE_INDEX_CORRUPTED),
				 errmsg("document-node multivector sidecar has invalid context metadata"),
				 errhint("REINDEX the index to rebuild the document-node multivector sidecar.")));

	for (int32 qi = 0; qi < query->count; qi++)
	{
		const float *qv =
			query->values + ((Size) qi * (Size) query->dim);
		double		weight = queryWeights != NULL ? (double) queryWeights[qi] : 1.0;
		double		best = -INFINITY;

		if ((queryMask != NULL && queryMask[qi]) || weight == 0.0)
			continue;
		if (!isfinite(weight) || weight < 0.0)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("query token weights must be finite non-negative values")));

		for (int32 di = startToken; di < endToken; di++)
		{
			double		dot = 0.0;
			Size		base = (Size) di * (Size) doc->dim;

			if (compact->kind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F16)
			{
				if (compact->f16Values != NULL &&
					compact->f16Values[docId] != NULL)
				{
					uint16	   *dv = compact->f16Values[docId] + base;

					for (int32 d = 0; d < doc->dim; d++)
						dot += (double) qv[d] *
							(double) PgturbohybridMultiVectorHalfToFloat(dv[d]);
				}
				else if (pagedF16Values != NULL)
				{
					const uint16 *dv = pagedF16Values + base;

					for (int32 d = 0; d < doc->dim; d++)
						dot += (double) qv[d] *
							(double) PgturbohybridMultiVectorHalfToFloat(dv[d]);
				}
				else
				{
					const float *dv = doc->values + base;

					for (int32 d = 0; d < doc->dim; d++)
					{
						uint16		half =
							PgturbohybridMultiVectorFloatToHalf(dv[d]);

						dot += (double) qv[d] *
							(double) PgturbohybridMultiVectorHalfToFloat(half);
					}
				}
			}
			else
			{
				float		scale = compact->sq8Scales != NULL ?
					compact->sq8Scales[docId] : pagedSq8Scale;

				if (compact->sq8Values != NULL &&
					compact->sq8Values[docId] != NULL)
				{
					int8	   *dv = compact->sq8Values[docId] + base;

					for (int32 d = 0; d < doc->dim; d++)
						dot += (double) qv[d] * (double) dv[d] *
							(double) scale;
				}
				else if (pagedSq8Values != NULL)
				{
					const int8 *dv = pagedSq8Values + base;

					for (int32 d = 0; d < doc->dim; d++)
						dot += (double) qv[d] * (double) dv[d] *
							(double) scale;
				}
				else
				{
					const float *dv = doc->values + base;

					for (int32 d = 0; d < doc->dim; d++)
					{
						int			quantized =
							(int) lrintf(dv[d] / scale);

						quantized = Max(-127, Min(127, quantized));
						dot += (double) qv[d] * (double) quantized *
							(double) scale;
					}
				}
			}
			if (dot > best)
				best = dot;
		}
		score += weight * best;
	}

	return score;
}

static double
PgturbohybridMultiVectorDocCompactMaxSim(const PgturbohybridMultiVector *query,
										 PgturbohybridMultiVectorDocCompactStorage *compact,
										 PgturbohybridMultiVector *doc,
										 TqDocId docId,
										 bool contextLevel,
										 const float4 *queryWeights,
										 const bool *queryMask,
										 uint64 *pairsScored)
{
	float		pagedSq8Scale = 1.0f;
	uint16	   *pagedF16Values = NULL;
	int8	   *pagedSq8Values = NULL;
	Size		valueCount;
	double		result;

	if (doc == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INDEX_CORRUPTED),
				 errmsg("document-node multivector sidecar is missing a document vector"),
				 errhint("REINDEX the index to rebuild the document-node multivector sidecar.")));
	if (pairsScored != NULL)
		*pairsScored += (uint64) query->count * (uint64) doc->count;

	valueCount = PgturbohybridMultiVectorFloatCount(doc->count, doc->dim);
	if (compact->kind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F16 &&
		(compact->f16Values == NULL || compact->f16Values[docId] == NULL))
	{
		Size		valueBytes =
			PgturbohybridMultiVectorDocCompactArraySize(sizeof(uint16),
														valueCount);

		pagedF16Values = palloc(valueBytes);
		for (Size i = 0; i < valueCount; i++)
			pagedF16Values[i] =
				PgturbohybridMultiVectorFloatToHalf(doc->values[i]);
	}
	if (compact->kind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_SQ8 &&
		(compact->sq8Values == NULL || compact->sq8Values[docId] == NULL))
	{
		Size		valueBytes =
			PgturbohybridMultiVectorDocCompactArraySize(sizeof(int8),
														valueCount);

		pagedSq8Scale = compact->sq8Scales != NULL ?
			compact->sq8Scales[docId] :
			PgturbohybridMultiVectorDocCompactSq8Scale(doc);
		pagedSq8Values = palloc(valueBytes);
		for (Size i = 0; i < valueCount; i++)
		{
			int			quantized =
				(int) lrintf(doc->values[i] / pagedSq8Scale);

			quantized = Max(-127, Min(127, quantized));
			pagedSq8Values[i] = (int8) quantized;
		}
	}
	if (!contextLevel || !PgturbohybridMultiVectorHasContexts(doc))
		result = PgturbohybridMultiVectorDocCompactMaxSimRange(query, compact,
															   doc, docId,
															   0, doc->count,
															   queryWeights,
															   queryMask,
															   pagedF16Values,
															   pagedSq8Values,
															   pagedSq8Scale);
	else
	{
		const int32 *offsets = PgturbohybridMultiVectorContextOffsets(doc);
		int32		contextCount = PgturbohybridMultiVectorContextCount(doc);
		double		best = -INFINITY;

		for (int32 ci = 0; ci < contextCount; ci++)
		{
			int32		start = offsets[ci];
			int32		end = (ci + 1 < contextCount) ? offsets[ci + 1] :
				doc->count;
			double		score =
				PgturbohybridMultiVectorDocCompactMaxSimRange(query, compact,
															  doc, docId,
															  start, end,
															  queryWeights,
															  queryMask,
															  pagedF16Values,
															  pagedSq8Values,
															  pagedSq8Scale);

			if (score > best)
				best = score;
		}
		result = best;
	}
	if (pagedF16Values != NULL)
		pfree(pagedF16Values);
	if (pagedSq8Values != NULL)
		pfree(pagedSq8Values);
	return result;
}

typedef struct PgturbohybridMultiVectorDocCompactPagedScoreState
{
	const PgturbohybridMultiVector *query;
	const float4 *queryWeights;
	const bool *queryMask;
	int			kind;
	int32		dim;
	uint32		tokenCount;
	bool		scaleOnly;
	bool		valid;
	bool		abandoned;
	bool		thresholdValid;
	float		maxAbs;
	float		maxTokenNorm;
	float		currentTokenNorm2;
	float		effectiveMaxTokenNorm;
	float		scale;
	float	   *scratch;
	double	   *best;
	float	   *queryNorms;
	double		scoreThreshold;
	uint64		boundChecks;
	uint32		tokensProcessed;
	uint64		pairsScored;
} PgturbohybridMultiVectorDocCompactPagedScoreState;

typedef struct PgturbohybridMultiVectorDocCompactPagedScratch
{
	float	   *scratch;
	double	   *best;
	float	   *queryNorms;
	int32		dimCapacity;
	int32		queryCapacity;
	int32		queryNormCapacity;
} PgturbohybridMultiVectorDocCompactPagedScratch;

static double
PgturbohybridMultiVectorDocCompactPagedUpperBound(PgturbohybridMultiVectorDocCompactPagedScoreState *state)
{
	double		upper = 0.0;

	for (int32 qi = 0; qi < state->query->count; qi++)
	{
		double		weight =
			state->queryWeights != NULL ? (double) state->queryWeights[qi] : 1.0;
		double		best = state->best[qi];
		double		possible;

		if ((state->queryMask != NULL && state->queryMask[qi]) ||
			weight == 0.0)
			continue;
		if (!isfinite(weight) || weight < 0.0)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("query token weights must be finite non-negative values")));
		possible =
			(double) state->queryNorms[qi] *
			(double) state->effectiveMaxTokenNorm;
		if (!isfinite(best) || possible > best)
			best = possible;
		upper += weight * best;
	}
	return upper;
}

static void
PgturbohybridMultiVectorDocCompactPagedScoreToken(PgturbohybridMultiVectorDocCompactPagedScoreState *state)
{
	for (int32 qb = 0; qb < state->query->count;
		 qb += PGTURBOHYBRID_MULTIVECTOR_COMPACT_DOT_BLOCK)
	{
		int32		blockCount =
			Min(state->query->count - qb,
				PGTURBOHYBRID_MULTIVECTOR_COMPACT_DOT_BLOCK);
		const float *queryValues =
			state->query->values + ((Size) qb * (Size) state->query->dim);
		double		dots[PGTURBOHYBRID_MULTIVECTOR_COMPACT_DOT_BLOCK];

		TqDotProductF32BlockAuto(queryValues, state->scratch, state->dim,
								 blockCount, dots);
		for (int32 bi = 0; bi < blockCount; bi++)
		{
			int32		qi = qb + bi;
			double		weight =
				state->queryWeights != NULL ? (double) state->queryWeights[qi] : 1.0;

			if ((state->queryMask != NULL && state->queryMask[qi]) ||
				weight == 0.0)
				continue;
			if (!isfinite(weight) || weight < 0.0)
				ereport(ERROR,
						(errcode(ERRCODE_DATA_EXCEPTION),
						 errmsg("query token weights must be finite non-negative values")));
			if (dots[bi] > state->best[qi])
				state->best[qi] = dots[bi];
		}
	}
	state->tokensProcessed++;
	if (state->thresholdValid)
	{
		double		upper;

		state->boundChecks++;
		upper = PgturbohybridMultiVectorDocCompactPagedUpperBound(state);
		if (upper <= state->scoreThreshold)
			state->abandoned = true;
	}
}

static bool
PgturbohybridMultiVectorDocCompactPagedScoreChunk(int storageKind,
												  const void *values,
												  float chunkScale,
												  uint16 count,
												  uint32 startFloat,
												  void *arg)
{
	PgturbohybridMultiVectorDocCompactPagedScoreState *state =
		(PgturbohybridMultiVectorDocCompactPagedScoreState *) arg;
	uint64		totalFloats =
		(uint64) state->tokenCount * (uint64) state->dim;

	if (!state->valid)
		return false;
	if (state->abandoned)
		return false;
	if ((uint64) startFloat + (uint64) count > totalFloats)
	{
		state->valid = false;
		return false;
	}

	for (uint16 i = 0; i < count; i++)
	{
		uint32		global = startFloat + (uint32) i;
		int32		dim = (int32) (global % (uint32) state->dim);
		float		value;

		if (storageKind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32)
			value = ((const float *) values)[i];
		else if (storageKind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F16)
			value =
				PgturbohybridMultiVectorHalfToFloat(((const uint16 *) values)[i]);
		else if (storageKind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_SQ8)
			value = (float) ((const int8 *) values)[i] * chunkScale;
		else
		{
			state->valid = false;
			return false;
		}

		if (state->scaleOnly)
		{
			state->maxAbs = Max(state->maxAbs, fabsf(value));
			state->currentTokenNorm2 += value * value;
			if (dim == state->dim - 1)
			{
				state->maxTokenNorm =
					Max(state->maxTokenNorm, sqrtf(state->currentTokenNorm2));
				state->currentTokenNorm2 = 0.0f;
			}
			continue;
		}

		if (state->kind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F16 &&
			storageKind != PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F16)
			value =
				PgturbohybridMultiVectorHalfToFloat(PgturbohybridMultiVectorFloatToHalf(value));
		else if (state->kind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_SQ8 &&
				 storageKind != PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_SQ8)
		{
			int			quantized = (int) lrintf(value / state->scale);

			quantized = Max(-127, Min(127, quantized));
			value = (float) quantized * state->scale;
		}
		state->scratch[dim] = value;
		if (dim == state->dim - 1)
		{
			PgturbohybridMultiVectorDocCompactPagedScoreToken(state);
			if (state->abandoned)
				return false;
		}
	}
	return true;
}

static bool
PgturbohybridMultiVectorDocCompactMaxSimPaged(Relation index,
											  PgturbohybridGraphMetaPageData *meta,
											  PgturbohybridGraphScanStorage *storage,
											  PgturbohybridMultiVectorDocCompactStorage *compact,
											  const PgturbohybridMultiVector *query,
											  TqDocId docId,
												  const float4 *queryWeights,
												  const bool *queryMask,
												  uint64 *pairsScored,
												  uint64 *boundChecks,
												  uint64 *docsPruned,
												  uint64 *tokensSkipped,
												  PgturbohybridMultiVectorDocSidecarAccessStats *sidecarStats,
												  PgturbohybridMultiVectorDocCompactPagedScratch *scratch,
												  bool scoreThresholdValid,
												  double scoreThreshold,
												  double *scoreOut)
{
	PgturbohybridMultiVectorDocCompactPagedScoreState state;
	TqMultiVectorDocMapEntry *entry;
	double		score = 0.0;
	bool		visited;
	bool		persistedSq8 = false;
	bool		ownScratch = false;
	bool		ownBest = false;
	bool		ownQueryNorms = false;

	if (scoreOut == NULL || compact == NULL || storage == NULL ||
		meta == NULL || !storage->multivectorDocVectorsPaged ||
		docId >= meta->tqMultivectorDocCount ||
		PgturbohybridMultiVectorIndexUsesContextLevel(index))
		return false;
	if (compact->kind != PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F16 &&
		compact->kind != PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_SQ8)
		return false;
	if (query->dim != meta->dimensions)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("query vector dimension does not match index dimension")));

	entry = &storage->multivectorDocMap[docId];
	if (entry->tokenCount == 0)
		return false;
	if (compact->kind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_SQ8 &&
		storage->multivectorDocVectorFirstChunk != NULL &&
		storage->multivectorDocVectorChunks != NULL &&
		docId < meta->tqMultivectorDocCount &&
		storage->multivectorDocVectorFirstChunk[docId] != PG_UINT32_MAX)
	{
		uint32		firstChunk = storage->multivectorDocVectorFirstChunk[docId];

		if (firstChunk < storage->multivectorDocVectorChunkCount &&
			storage->multivectorDocVectorChunks[firstChunk].storageKind ==
			PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_SQ8)
			persistedSq8 = true;
	}

	memset(&state, 0, sizeof(state));
	state.query = query;
	state.queryWeights = queryWeights;
	state.queryMask = queryMask;
	state.kind = compact->kind;
	state.dim = query->dim;
	state.tokenCount = entry->tokenCount;
	state.valid = true;
	state.scale = 1.0f;
	if (scratch != NULL &&
		scratch->scratch != NULL &&
		scratch->best != NULL &&
		scratch->queryNorms != NULL &&
		scratch->dimCapacity >= query->dim &&
		scratch->queryCapacity >= query->count &&
		scratch->queryNormCapacity >= query->count)
	{
		state.scratch = scratch->scratch;
		state.best = scratch->best;
		state.queryNorms = scratch->queryNorms;
	}
	else
	{
		state.scratch = palloc(sizeof(float) * (Size) query->dim);
		state.best = palloc(sizeof(double) * (Size) query->count);
		state.queryNorms = palloc(sizeof(float) * (Size) query->count);
		ownScratch = true;
		ownBest = true;
		ownQueryNorms = true;
	}
	for (int32 qi = 0; qi < query->count; qi++)
	{
		const float *queryValues =
			query->values + ((Size) qi * (Size) query->dim);
		double		norm2 = 0.0;

		state.best[qi] = -INFINITY;
		for (int32 dim = 0; dim < query->dim; dim++)
			norm2 += (double) queryValues[dim] * (double) queryValues[dim];
		state.queryNorms[qi] = (float) sqrt(norm2);
	}

	if (compact->kind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_SQ8 &&
		!persistedSq8)
	{
		state.scaleOnly = true;
		visited = PgturbohybridGraphVisitMultiVectorDocCompactChunks(index,
																	 meta,
																	 storage,
																	 docId,
																	 PgturbohybridMultiVectorDocCompactPagedScoreChunk,
																	 &state,
																	 sidecarStats);
		if (!visited || !state.valid)
		{
			if (ownQueryNorms)
				pfree(state.queryNorms);
			if (ownBest)
				pfree(state.best);
			if (ownScratch)
				pfree(state.scratch);
			return false;
		}
		state.scale = state.maxAbs > 0.0f ? state.maxAbs / 127.0f : 1.0f;
		state.effectiveMaxTokenNorm =
			state.maxTokenNorm + sqrtf((float) state.dim) * 0.5f * state.scale;
		state.thresholdValid = scoreThresholdValid;
		state.scoreThreshold = scoreThreshold;
		state.scaleOnly = false;
	}
	else
	{
		state.thresholdValid = false;
		state.effectiveMaxTokenNorm = 0.0f;
	}

	visited = PgturbohybridGraphVisitMultiVectorDocCompactChunks(index, meta,
																 storage, docId,
																 PgturbohybridMultiVectorDocCompactPagedScoreChunk,
																 &state,
																 sidecarStats);
	if (!visited || !state.valid)
	{
		if (ownQueryNorms)
			pfree(state.queryNorms);
		if (ownBest)
			pfree(state.best);
		if (ownScratch)
			pfree(state.scratch);
		return false;
	}
	if (state.abandoned)
	{
		if (boundChecks != NULL)
			*boundChecks += state.boundChecks;
		if (docsPruned != NULL)
			(*docsPruned)++;
		if (tokensSkipped != NULL && entry->tokenCount > state.tokensProcessed)
			*tokensSkipped +=
				(uint64) entry->tokenCount - (uint64) state.tokensProcessed;
		if (pairsScored != NULL)
			*pairsScored +=
				(uint64) query->count * (uint64) state.tokensProcessed;
		*scoreOut = -INFINITY;
		if (ownQueryNorms)
			pfree(state.queryNorms);
		if (ownBest)
			pfree(state.best);
		if (ownScratch)
			pfree(state.scratch);
		return true;
	}

	for (int32 qi = 0; qi < query->count; qi++)
	{
		double		weight =
			queryWeights != NULL ? (double) queryWeights[qi] : 1.0;

		if ((queryMask != NULL && queryMask[qi]) || weight == 0.0)
			continue;
		if (!isfinite(state.best[qi]))
		{
			if (ownQueryNorms)
				pfree(state.queryNorms);
			if (ownBest)
				pfree(state.best);
			if (ownScratch)
				pfree(state.scratch);
			return false;
		}
		score += weight * state.best[qi];
	}
	if (boundChecks != NULL)
		*boundChecks += state.boundChecks;
	if (pairsScored != NULL)
		*pairsScored += (uint64) query->count * (uint64) state.tokensProcessed;
	*scoreOut = score;
	if (ownQueryNorms)
		pfree(state.queryNorms);
	if (ownBest)
		pfree(state.best);
	if (ownScratch)
		pfree(state.scratch);
	return true;
}

static double
PgturbohybridMultiVectorDocumentGraphNodeDistance(Relation index,
												  PgturbohybridGraphScanOpaque so,
												  PgturbohybridGraphMetaPageData *meta,
												  PgturbohybridGraphScanStorage *storage,
												  PgturbohybridMultiVectorDocCompactStorage *compact,
												  const PgturbohybridMultiVector *query,
												  const float4 *queryWeights,
													  const bool *queryMask,
													  uint32 nodeId,
													  uint64 *pairsScored,
													  uint64 *boundChecks,
													  uint64 *docsPruned,
													  uint64 *tokensSkipped,
													  PgturbohybridMultiVectorDocCompactScoreCache *scoreCache,
													  PgturbohybridMultiVectorDocCompactPagedScratch *pagedScratch,
													  PgturbohybridMultiVectorDocSidecarAccessStats *sidecarStats,
													  bool scoreThresholdValid,
													  double scoreThreshold)
{
	PgturbohybridGraphScanNode *node;
	TqMultiVectorNodeMapEntry *nodeEntry;
	TqDocId		docId;
	PgturbohybridMultiVector *doc;
	double		maxsim;

	if (nodeId >= meta->tqNodeCount ||
		!PgturbohybridGraphLoadCodePage(index, so, meta, storage, nodeId))
		return DBL_MAX;

	node = &storage->nodes[nodeId];
	if ((node->flags & PGTURBOHYBRID_GRAPH_NODE_DEAD) != 0)
		return DBL_MAX;

	nodeEntry = &storage->multivectorNodeMap[nodeId];
	docId = nodeEntry->docId;
	if (docId >= meta->tqMultivectorDocCount)
		ereport(ERROR,
				(errcode(ERRCODE_INDEX_CORRUPTED),
					 errmsg("document-node multivector sidecar has invalid document id"),
					 errhint("REINDEX the index to rebuild the document-node multivector sidecar.")));

	if (compact != NULL &&
		scoreCache != NULL &&
		scoreCache->valid != NULL &&
		scoreCache->distances != NULL &&
		docId < scoreCache->docCount &&
		scoreCache->valid[docId])
	{
		scoreCache->hits++;
		return scoreCache->distances[docId];
	}
	if (compact != NULL && scoreCache != NULL)
		scoreCache->misses++;

	if (compact != NULL &&
		PgturbohybridMultiVectorDocCompactMaxSimPaged(index, meta, storage,
													  compact, query, docId,
														  queryWeights,
														  queryMask,
														  pairsScored,
														  boundChecks,
														  docsPruned,
														  tokensSkipped,
														  sidecarStats,
														  pagedScratch,
														  scoreThresholdValid,
														  scoreThreshold,
														  &maxsim))
	{
		if (compact != NULL &&
			scoreCache != NULL &&
			scoreCache->valid != NULL &&
			scoreCache->distances != NULL &&
			docId < scoreCache->docCount)
		{
			scoreCache->distances[docId] = -maxsim;
			scoreCache->valid[docId] = true;
		}
		return -maxsim;
	}

	doc = PgturbohybridGraphLoadMultiVectorDocVector(index, meta, storage,
													 docId, storage->ctx,
													 sidecarStats);

	PgturbohybridCheckSameMultiVectorDims(query, doc);
	if (compact != NULL)
		maxsim = PgturbohybridMultiVectorDocCompactMaxSim(query, compact, doc,
														  docId,
														  PgturbohybridMultiVectorIndexUsesContextLevel(index),
														  queryWeights,
														  queryMask,
														  pairsScored);
	else
	{
		if (pairsScored != NULL)
			*pairsScored += (uint64) query->count * (uint64) doc->count;
		maxsim = PgturbohybridMultiVectorIndexMaxSim(index, query, doc,
													 queryWeights,
													 queryMask);
	}
	if (storage->multivectorDocVectorsPaged)
		pfree(doc);
	if (compact != NULL &&
		scoreCache != NULL &&
		scoreCache->valid != NULL &&
		scoreCache->distances != NULL &&
		docId < scoreCache->docCount)
	{
		scoreCache->distances[docId] = -maxsim;
		scoreCache->valid[docId] = true;
	}
	return -maxsim;
}

static int
PgturbohybridMultiVectorProxyDocumentSidecarRescore(Relation index,
													PgturbohybridGraphScanOpaque so,
													PgturbohybridGraphMetaPageData *meta,
													PgturbohybridGraphScanStorage *storage,
													PgturbohybridMultiVectorDocCompactStorage *compact,
													const PgturbohybridMultiVector *query,
													const float4 *queryWeights,
													const bool *queryMask,
													double queryWeightSum,
													TqDenseCandidate *candidates,
													int candidateCount,
													uint64 *pairsScored,
													PgturbohybridMultiVectorDocSidecarAccessStats *sidecarStats)
{
	int			rescoredCount = 0;

	for (int i = 0; i < candidateCount; i++)
	{
		double		distance;

		CHECK_FOR_INTERRUPTS();
		distance = PgturbohybridMultiVectorDocumentGraphNodeDistance(index, so,
																	 meta,
																	 storage,
																	 compact,
																	 query,
																	 queryWeights,
																	 queryMask,
																			 candidates[i].nodeId,
																			 pairsScored,
																			 NULL,
																			 NULL,
																			 NULL,
																			 NULL,
																			 NULL,
																			 sidecarStats,
																			 false,
																			 0.0);
		if (distance == DBL_MAX)
			continue;
		if (rescoredCount != i)
			candidates[rescoredCount] = candidates[i];
		candidates[rescoredCount].distance = distance;
		candidates[rescoredCount].similarity =
			queryWeightSum > 0.0 ? (-distance) / queryWeightSum : 0.0;
		candidates[rescoredCount].exactScored = false;
		rescoredCount++;
	}
	PgturbohybridMultiVectorCandidateHeapSort(candidates, rescoredCount);
	return rescoredCount;
}

static int
PgturbohybridMultiVectorProxyCentroidPrerank(Relation index,
											 PgturbohybridGraphScanStorage *storage,
											 const PgturbohybridMultiVector *query,
											 const float4 *queryWeights,
											 const bool *queryMask,
											 double queryWeightSum,
											 TqDenseCandidate *candidates,
											 int candidateCount,
											 uint64 *pairsScored,
											 uint32 *centroidCountOut)
{
	int			prerankedCount = 0;
	uint32		maxCentroidCount = 0;

	for (int i = 0; i < candidateCount; i++)
	{
		TqDocId		docId;
		PgturbohybridMultiVector *centroids;
		double		maxsim;

		CHECK_FOR_INTERRUPTS();
		if (!candidates[i].hasDocId)
			continue;
		docId = candidates[i].docId;
		if (docId >= storage->multivectorDocCount)
			ereport(ERROR,
					(errcode(ERRCODE_INDEX_CORRUPTED),
					 errmsg("document-node centroid sidecar has invalid document id"),
					 errhint("REINDEX with multivector_graph = document_nodes and multivector_centroids = kmeans to rebuild persisted centroid sidecar tuples.")));
		centroids = storage->multivectorDocCentroids[docId];
		if (centroids == NULL)
			ereport(ERROR,
					(errcode(ERRCODE_INDEX_CORRUPTED),
					 errmsg("document-node centroid sidecar is missing a document centroid"),
					 errhint("REINDEX with multivector_graph = document_nodes and multivector_centroids = kmeans to rebuild persisted centroid sidecar tuples.")));
		PgturbohybridCheckSameMultiVectorDims(query, centroids);
		if (pairsScored != NULL)
			*pairsScored +=
				(uint64) query->count * (uint64) centroids->count;
		maxsim = PgturbohybridMultiVectorIndexMaxSim(index, query,
													 centroids,
													 queryWeights,
													 queryMask);
		candidates[i].distance = -maxsim;
		candidates[i].similarity =
			queryWeightSum > 0.0 ? maxsim / queryWeightSum : 0.0;
		candidates[i].exactScored = false;
		prerankedCount++;
		maxCentroidCount = Max(maxCentroidCount,
							   (uint32) centroids->count);
	}
	if (prerankedCount > 0)
		PgturbohybridMultiVectorCandidateHeapSort(candidates,
												  candidateCount);
	if (centroidCountOut != NULL)
		*centroidCountOut = maxCentroidCount;
	return prerankedCount;
}

static int
PgturbohybridMultiVectorDocumentGraphAddEntry(PgturbohybridGraphFrontierItem *entries,
											  int *entryCount,
											  int entryLimit,
											  uint32 nodeId,
											  double distance)
{
	PgturbohybridGraphFrontierItem entry;

	if (nodeId == UINT_MAX || distance == DBL_MAX)
		return *entryCount;

	for (int i = 0; i < *entryCount; i++)
	{
		if (entries[i].nodeId == nodeId)
			return *entryCount;
	}

	entry.nodeId = nodeId;
	entry.distance = distance;
	PgturbohybridGraphOfferDistanceEntry(entries, entryCount, entry);
	if (*entryCount > entryLimit)
		*entryCount = entryLimit;
	return *entryCount;
}

static PgturbohybridGraphFrontierItem
PgturbohybridMultiVectorDocumentGraphGreedySearch(Relation index,
												  PgturbohybridGraphScanOpaque so,
												  PgturbohybridGraphMetaPageData *meta,
												  PgturbohybridGraphScanStorage *storage,
												  PgturbohybridMultiVectorDocCompactStorage *compact,
												  const PgturbohybridMultiVector *query,
													  const float4 *queryWeights,
													  const bool *queryMask,
													  PgturbohybridGraphFrontierItem entry,
													  int level,
														  uint64 *pairsScored,
														  uint64 *edgesVisited,
														  PgturbohybridMultiVectorDocCompactScoreCache *scoreCache,
														  PgturbohybridMultiVectorDocCompactPagedScratch *pagedScratch,
														  PgturbohybridMultiVectorDocSidecarAccessStats *sidecarStats)
{
	PgturbohybridGraphFrontierItem current = entry;
	bool		changed = true;

	while (changed)
	{
		int			slot;

		CHECK_FOR_INTERRUPTS();
		changed = false;
		if (!PgturbohybridGraphLoadAdjPage(index, so, meta, storage,
										   current.nodeId, level))
			break;

		slot = PgturbohybridGraphScanAdjSlot(meta, current.nodeId, level);
		for (int i = 0; i < storage->neighborCounts[slot]; i++)
		{
			uint32		neighbor = storage->neighbors[slot][i];
			double		distance;

			CHECK_FOR_INTERRUPTS();
			if (edgesVisited != NULL)
				(*edgesVisited)++;
			if (neighbor >= meta->tqNodeCount ||
				!PgturbohybridGraphLoadCodePage(index, so, meta, storage,
												 neighbor) ||
				storage->nodes[neighbor].level < level)
				continue;

			distance = PgturbohybridMultiVectorDocumentGraphNodeDistance(index,
																		 so,
																		 meta,
																		 storage,
																		 compact,
																			 query,
																			 queryWeights,
																			 queryMask,
																				 neighbor,
																				 pairsScored,
																				 NULL,
																				 NULL,
																				 NULL,
																				 scoreCache,
																				 pagedScratch,
																				 sidecarStats,
																				 false,
																				 0.0);
			if (distance < current.distance)
			{
				current.nodeId = neighbor;
				current.distance = distance;
				changed = true;
			}
		}
	}

	return current;
}

static int
PgturbohybridMultiVectorDocumentGraphTraverse(Relation index,
											  PgturbohybridGraphScanOpaque so,
											  PgturbohybridGraphMetaPageData *meta,
											  PgturbohybridGraphScanStorage *storage,
											  PgturbohybridMultiVectorDocCompactStorage *compact,
											  const PgturbohybridMultiVector *query,
											  const float4 *queryWeights,
											  const bool *queryMask,
												  double queryWeightSum,
												  int resultTarget,
												  int searchEf,
												  TqDenseCandidate *candidates,
												  uint64 *docsScored,
												  uint64 *edgesVisited,
													  uint64 *pairsScored,
													  uint64 *compactScoreCacheHits,
													  uint64 *compactScoreCacheMisses,
													  uint64 *compactBoundChecks,
													  uint64 *compactDocsPruned,
													  uint64 *compactTokensSkipped,
													  PgturbohybridMultiVectorDocSidecarAccessStats *sidecarStats)
{
	PgturbohybridGraphFrontierItem entries[PGTURBOHYBRID_GRAPH_MAX_ENTRY_POINTS];
	PgturbohybridGraphFrontierItem *sampled = NULL;
	PgturbohybridMultiVectorDocCompactScoreCache scoreCache;
	PgturbohybridMultiVectorDocCompactScoreCache *scoreCachePtr = NULL;
	PgturbohybridMultiVectorDocCompactPagedScratch pagedScratch;
	PgturbohybridMultiVectorDocCompactPagedScratch *pagedScratchPtr = NULL;
	int			entryCount = 0;
	int			sampledCount = 0;
	int			sampleCount = 0;
	bool	   *visited;
	PgturbohybridGraphFrontierItem *frontier;
	PgturbohybridGraphFrontierItem *nearest;
	int			frontierCount = 0;
	int			frontierCapacity;
	int			nearestCount = 0;
	int			docCount = 0;
	int			maxNeighbors = PgturbohybridGraphLevelM(meta->m, 0);
	int			maxFrontierCapacity = (int) meta->tqNodeCount;
	uint32		entryNodeId =
		meta->tqEntryNodeId < meta->tqNodeCount ? meta->tqEntryNodeId : 0;
	PgturbohybridGraphFrontierItem entry;
	bool		useCompactScoreCache = false;

	if (meta->tqNodeCount == 0 || resultTarget <= 0 || searchEf <= 0)
		return 0;
	memset(&scoreCache, 0, sizeof(scoreCache));
	memset(&pagedScratch, 0, sizeof(pagedScratch));
	/*
	 * The docCount-sized score cache regressed paged compact scoring by
	 * increasing per-query memory and sidecar touches. Keep the stat plumbing
	 * stable, but leave the cache disabled until it has an explicit guard and
	 * evidence.
	 */
	if (useCompactScoreCache && compact != NULL &&
		meta->tqMultivectorDocCount > 0)
	{
		scoreCache.docCount = meta->tqMultivectorDocCount;
		scoreCache.valid =
			palloc0(sizeof(bool) * (Size) meta->tqMultivectorDocCount);
		scoreCache.distances =
			palloc(sizeof(double) * (Size) meta->tqMultivectorDocCount);
		scoreCachePtr = &scoreCache;
	}

	frontierCapacity =
		PgturbohybridGraphInitialFrontierCapacity(meta->tqNodeCount,
												  searchEf,
												  PGTURBOHYBRID_GRAPH_MAX_ENTRY_POINTS,
												  maxNeighbors);
	visited = palloc0(sizeof(bool) * meta->tqNodeCount);
	frontier = palloc(sizeof(PgturbohybridGraphFrontierItem) * frontierCapacity);
	nearest = palloc(sizeof(PgturbohybridGraphFrontierItem) * searchEf);
	if (compact != NULL && storage->multivectorDocVectorsPaged &&
		query->dim > 0 && query->count > 0)
	{
			pagedScratch.scratch = palloc(sizeof(float) * (Size) query->dim);
			pagedScratch.best = palloc(sizeof(double) * (Size) query->count);
			pagedScratch.queryNorms = palloc(sizeof(float) * (Size) query->count);
			pagedScratch.dimCapacity = query->dim;
			pagedScratch.queryCapacity = query->count;
			pagedScratch.queryNormCapacity = query->count;
			pagedScratchPtr = &pagedScratch;
		}

	entry.nodeId = entryNodeId;
	entry.distance =
		PgturbohybridMultiVectorDocumentGraphNodeDistance(index, so, meta,
														  storage, compact,
															  query,
															  queryWeights,
															  queryMask,
																	  entryNodeId,
																	  pairsScored,
																	  NULL,
																	  NULL,
																	  NULL,
																	  scoreCachePtr,
																	  pagedScratchPtr,
																	  sidecarStats,
																	  false,
																	  0.0);
	for (int level = meta->graphMaxLevel; level > 0; level--)
	{
		if (entry.distance != DBL_MAX &&
			storage->nodes[entry.nodeId].level >= level)
			entry = PgturbohybridMultiVectorDocumentGraphGreedySearch(index,
																	  so,
																	  meta,
																	  storage,
																	  compact,
																	  query,
																	  queryWeights,
																	  queryMask,
																		  entry,
																		  level,
																			  pairsScored,
																			  edgesVisited,
																			  scoreCachePtr,
																			  pagedScratchPtr,
																		  sidecarStats);
	}
	PgturbohybridMultiVectorDocumentGraphAddEntry(entries, &entryCount,
												 PGTURBOHYBRID_GRAPH_MAX_ENTRY_POINTS,
												 entry.nodeId, entry.distance);

	for (uint16 i = 0; i < meta->tqSegmentCount; i++)
	{
		PgturbohybridGraphSegmentMetaData *segment = &meta->tqSegments[i];
		double		distance;

		CHECK_FOR_INTERRUPTS();
		if (segment->entryNodeId >= meta->tqNodeCount)
			continue;
		distance = PgturbohybridMultiVectorDocumentGraphNodeDistance(index,
																	 so,
																	 meta,
																	 storage,
																	 compact,
																		 query,
																		 queryWeights,
																		 queryMask,
																				 segment->entryNodeId,
																				 pairsScored,
																				 NULL,
																				 NULL,
																				 NULL,
																				 scoreCachePtr,
																				 pagedScratchPtr,
																				 sidecarStats,
																				 false,
																				 0.0);
		PgturbohybridMultiVectorDocumentGraphAddEntry(entries, &entryCount,
													 PGTURBOHYBRID_GRAPH_MAX_ENTRY_POINTS,
													 segment->entryNodeId,
													 distance);
	}
	for (uint16 i = 0; i < meta->tqRoutingEntryCount; i++)
	{
		uint32		nodeId = meta->tqRoutingEntryNodeIds[i];
		double		distance;

		CHECK_FOR_INTERRUPTS();
		if (nodeId >= meta->tqNodeCount)
			continue;
		distance = PgturbohybridMultiVectorDocumentGraphNodeDistance(index,
																	 so,
																	 meta,
																	 storage,
																	 compact,
																		 query,
																		 queryWeights,
																		 queryMask,
																				 nodeId,
																				 pairsScored,
																				 NULL,
																				 NULL,
																				 NULL,
																				 scoreCachePtr,
																				 pagedScratchPtr,
																				 sidecarStats,
																				 false,
																				 0.0);
		PgturbohybridMultiVectorDocumentGraphAddEntry(entries, &entryCount,
													 PGTURBOHYBRID_GRAPH_MAX_ENTRY_POINTS,
													 nodeId, distance);
	}
	if (meta->tqNodeCount > 1)
	{
		int			configuredSamples =
			Max(pgturbohybrid_multivector_doc_graph_entry_sample_count, 0);

		if (configuredSamples > 0)
			sampleCount = Min((int) meta->tqNodeCount, configuredSamples);
		else
			sampleCount =
				Min((int) meta->tqNodeCount,
					Min(PGTURBOHYBRID_GRAPH_ENTRY_SAMPLE_COUNT,
						Max(searchEf, PGTURBOHYBRID_GRAPH_MAX_ENTRY_POINTS)));
		so->graphEntrySampleConfigured = configuredSamples;
		so->graphEntrySampleEffective = sampleCount;

		/*
		 * Document-node graphs need the same multi-entry robustness as the
		 * single-vector graph path.  A single global entry, plus a few segment
		 * entries, can start large ColBERT indexes in the wrong region and then
		 * spend a large rerank budget on irrelevant documents.  Score a bounded,
		 * deterministic spread of document proxy nodes and keep the best entry
		 * seeds before base-layer traversal.
		 */
		sampled = palloc(sizeof(PgturbohybridGraphFrontierItem) * sampleCount);
		for (int i = 0; i < sampleCount; i++)
		{
			uint32		nodeId = sampleCount == 1 ? 0 :
				(uint32) (((uint64) i * (meta->tqNodeCount - 1)) /
						  (sampleCount - 1));
			bool		seen =
				PgturbohybridGraphEntryAlreadySelected(entries, entryCount,
													   nodeId);

			CHECK_FOR_INTERRUPTS();
			for (int j = 0; j < sampledCount; j++)
			{
				if (sampled[j].nodeId == nodeId)
				{
					seen = true;
					break;
				}
			}
			if (seen || nodeId >= meta->tqNodeCount ||
				!PgturbohybridGraphLoadCodePage(index, so, meta, storage,
												 nodeId))
				continue;
			if (storage->nodes[nodeId].flags & PGTURBOHYBRID_GRAPH_NODE_DEAD)
				continue;

			sampled[sampledCount].nodeId = nodeId;
			sampled[sampledCount].distance =
				PgturbohybridMultiVectorDocumentGraphNodeDistance(index, so,
																  meta,
																  storage,
																  compact,
																	  query,
																	  queryWeights,
																	  queryMask,
																			  nodeId,
																			  pairsScored,
																			  NULL,
																			  NULL,
																			  NULL,
																			  scoreCachePtr,
																			  pagedScratchPtr,
																			  sidecarStats,
																			  false,
																			  0.0);
			if (sampled[sampledCount].distance != DBL_MAX)
				sampledCount++;
		}
		qsort(sampled, sampledCount, sizeof(PgturbohybridGraphFrontierItem),
			  PgturbohybridGraphFrontierCompare);
		for (int i = 0; i < sampledCount; i++)
			PgturbohybridMultiVectorDocumentGraphAddEntry(entries, &entryCount,
														 PGTURBOHYBRID_GRAPH_MAX_ENTRY_POINTS,
														 sampled[i].nodeId,
														 sampled[i].distance);
		so->graphEntrySampleScored = sampledCount;
	}

	if (entryCount == 0)
	{
		if (sampled != NULL)
			pfree(sampled);
		if (scoreCache.valid != NULL)
			pfree(scoreCache.valid);
		if (scoreCache.distances != NULL)
			pfree(scoreCache.distances);
		if (pagedScratch.scratch != NULL)
			pfree(pagedScratch.scratch);
		if (pagedScratch.best != NULL)
			pfree(pagedScratch.best);
		if (pagedScratch.queryNorms != NULL)
			pfree(pagedScratch.queryNorms);
		pfree(nearest);
		pfree(frontier);
		pfree(visited);
		return 0;
	}
	so->graphEntryPointCount = entryCount;

	for (int i = 0; i < entryCount; i++)
	{
		PgturbohybridGraphFrontierItem seed = entries[i];

		if (seed.nodeId >= meta->tqNodeCount || visited[seed.nodeId])
			continue;
		visited[seed.nodeId] = true;
		PgturbohybridGraphFrontierHeapPushGrowing(&frontier, &frontierCount,
												  &frontierCapacity,
												  maxFrontierCapacity,
												  seed, true);
		(void) PgturbohybridGraphOfferNearest(nearest, searchEf, &nearestCount,
											   seed.nodeId, seed.distance);
		if (docsScored != NULL)
			(*docsScored)++;
	}

	while (frontierCount > 0)
	{
		PgturbohybridGraphFrontierItem item =
			PgturbohybridGraphFrontierHeapPop(frontier, &frontierCount, true);
		int			slot;

		CHECK_FOR_INTERRUPTS();
		if (nearestCount >= searchEf &&
			PgturbohybridGraphFrontierGreater(item, nearest[0]))
			break;
		if (!PgturbohybridGraphLoadAdjPage(index, so, meta, storage,
										   item.nodeId, 0))
			continue;

		so->graphVisitedNodes++;
		slot = PgturbohybridGraphScanAdjSlot(meta, item.nodeId, 0);
		for (int i = 0; i < storage->neighborCounts[slot]; i++)
		{
			uint32		neighbor = storage->neighbors[slot][i];
			double		distance;
			bool		scoreThresholdValid = false;
			double		scoreThreshold = 0.0;

			CHECK_FOR_INTERRUPTS();
			if (edgesVisited != NULL)
				(*edgesVisited)++;
			if (neighbor >= meta->tqNodeCount || visited[neighbor])
				continue;
			visited[neighbor] = true;
			if (compact != NULL && nearestCount >= searchEf &&
				nearest[0].distance != DBL_MAX)
			{
				scoreThresholdValid = true;
				scoreThreshold = -nearest[0].distance;
			}
			distance = PgturbohybridMultiVectorDocumentGraphNodeDistance(index,
																		 so,
																		 meta,
																		 storage,
																		 compact,
																			 query,
																			 queryWeights,
																			 queryMask,
																			 neighbor,
																			 pairsScored,
																			 compactBoundChecks,
																			 compactDocsPruned,
																			 compactTokensSkipped,
																			 scoreCachePtr,
																			 pagedScratchPtr,
																			 sidecarStats,
																			 scoreThresholdValid,
																			 scoreThreshold);
			if (distance == DBL_MAX)
				continue;
			if (docsScored != NULL)
				(*docsScored)++;
			if (PgturbohybridGraphOfferNearest(nearest, searchEf,
											   &nearestCount, neighbor,
											   distance))
			{
				PgturbohybridGraphFrontierItem frontierItem;

				frontierItem.nodeId = neighbor;
				frontierItem.distance = distance;
				PgturbohybridGraphFrontierHeapPushGrowing(&frontier,
														  &frontierCount,
														  &frontierCapacity,
														  maxFrontierCapacity,
														  frontierItem,
														  true);
			}
		}
	}

	for (int i = 0; i < nearestCount; i++)
	{
		uint32		nodeId = nearest[i].nodeId;
		TqDocId		docId;
		TqMultiVectorDocMapEntry *docEntry;
		TqDenseCandidate candidate;

		if (nodeId >= meta->tqNodeCount ||
			storage->nodes[nodeId].flags & PGTURBOHYBRID_GRAPH_NODE_DEAD)
			continue;
		docId = storage->multivectorNodeMap[nodeId].docId;
		if (docId >= meta->tqMultivectorDocCount)
			continue;
		docEntry = &storage->multivectorDocMap[docId];

		memset(&candidate, 0, sizeof(candidate));
		candidate.nodeId = nodeId;
		candidate.docId = docId;
		candidate.heaptid = docEntry->heapTid;
		candidate.distance = nearest[i].distance;
		candidate.similarity =
			queryWeightSum > 0.0 ? (-nearest[i].distance) / queryWeightSum : 0.0;
		candidate.rank = 0;
		candidate.hasDocId = true;
		candidate.exactScored = false;
		PgturbohybridMultiVectorCandidateHeapOffer(candidates, &docCount,
												   resultTarget, &candidate);
	}

	if (sampled != NULL)
		pfree(sampled);
	if (compactScoreCacheHits != NULL)
		*compactScoreCacheHits = scoreCache.hits;
	if (compactScoreCacheMisses != NULL)
		*compactScoreCacheMisses = scoreCache.misses;
	if (scoreCache.valid != NULL)
		pfree(scoreCache.valid);
	if (scoreCache.distances != NULL)
		pfree(scoreCache.distances);
	if (pagedScratch.scratch != NULL)
		pfree(pagedScratch.scratch);
	if (pagedScratch.best != NULL)
		pfree(pagedScratch.best);
	if (pagedScratch.queryNorms != NULL)
		pfree(pagedScratch.queryNorms);
	pfree(nearest);
	pfree(frontier);
	pfree(visited);
	return docCount;
}

static uint32
PgturbohybridMultiVectorDeterministicCodeword(const PgturbohybridMultiVector *mv,
											  int32 token)
{
	const float *values;
	double		bestAbs = -1.0;
	int32		bestDim = 0;
	bool		negative = false;

	values = PgturbohybridMultiVectorValues(mv, token);
	for (int32 dim = 0; dim < mv->dim; dim++)
	{
		double		value = (double) values[dim];
		double		absValue = fabs(value);

		if (absValue > bestAbs)
		{
			bestAbs = absValue;
			bestDim = dim;
			negative = value < 0.0;
		}
	}

	return (uint32) bestDim * 2U + (negative ? 1U : 0U);
}

uint32
PgturbohybridMultiVectorDeterministicCodewordsWithScratch(const PgturbohybridMultiVector *mv,
														  int32 token,
														  uint32 limit,
														  uint32 *codewords,
														  PgturbohybridMultiVectorPostingScoreCandidate *selected)
{
	const float *values;
	uint32		codebookSize;
	uint32		selectedCount = 0;

	Assert(limit > 0);
	Assert(selected != NULL);
	values = PgturbohybridMultiVectorValues(mv, token);
	codebookSize = (uint32) mv->dim * 2U;
	limit = Min(limit, codebookSize);
	memset(selected, 0,
		   sizeof(PgturbohybridMultiVectorPostingScoreCandidate) *
		   (Size) limit);
	for (int32 dim = 0; dim < mv->dim; dim++)
	{
		PgturbohybridMultiVectorPostingScoreCandidate positive;
		PgturbohybridMultiVectorPostingScoreCandidate negative;
		double		value = (double) values[dim];

		positive.docId = (uint32) dim * 2U;
		positive.ordinal = 0;
		positive.score = value;
		PgturbohybridMultiVectorPostingScoreCandidateOffer(selected,
														   &selectedCount,
														   limit,
														   &positive);
		negative.docId = (uint32) dim * 2U + 1U;
		negative.ordinal = 0;
		negative.score = -value;
		PgturbohybridMultiVectorPostingScoreCandidateOffer(selected,
														   &selectedCount,
														   limit,
														   &negative);
	}
	for (uint32 i = 0; i < selectedCount; i++)
		codewords[i] = selected[i].docId;
	return selectedCount;
}

static uint32
PgturbohybridMultiVectorDeterministicCodewords(const PgturbohybridMultiVector *mv,
											   int32 token,
											   uint32 limit,
											   uint32 *codewords)
{
	PgturbohybridMultiVectorPostingScoreCandidate *selected;
	uint32		selectedCount;

	Assert(limit > 0);
	limit = Min(limit, (uint32) mv->dim * 2U);
	selected =
		palloc0(sizeof(PgturbohybridMultiVectorPostingScoreCandidate) *
				(Size) limit);
	selectedCount =
		PgturbohybridMultiVectorDeterministicCodewordsWithScratch(mv, token,
																  limit,
																  codewords,
																  selected);
	pfree(selected);
	return selectedCount;
}

double
PgturbohybridMultiVectorDeterministicCodewordScore(const PgturbohybridMultiVector *mv,
												   int32 token,
												   uint32 codeword)
{
	const float *values;
	uint32		dim = codeword / 2U;
	double		value;

	Assert(token >= 0 && token < mv->count);
	if (dim >= (uint32) mv->dim)
		return -DBL_MAX;
	values = PgturbohybridMultiVectorValues(mv, token);
	value = (double) values[dim];
	return (codeword & 1U) ? -value : value;
}

static double
PgturbohybridMultiVectorCentroidCodewordMaxSimScore(const PgturbohybridMultiVector *query,
													const float4 *queryWeights,
													const bool *queryMask,
													const uint32 *docCodes,
													uint32 docCodeCount,
													const float *queryCodewordScores,
													uint32 codebookSize,
													uint64 *pairs)
{
	double		score = 0.0;

	if (docCodes == NULL || docCodeCount == 0 ||
		queryCodewordScores == NULL || codebookSize == 0)
		return -DBL_MAX;

	if (queryWeights == NULL && queryMask == NULL)
	{
		for (int32 qi = 0; qi < query->count; qi++)
		{
			float		best = -FLT_MAX;
			const float *tokenScores =
				queryCodewordScores + (Size) qi * (Size) codebookSize;

			for (uint32 codeIndex = 0; codeIndex < docCodeCount; codeIndex++)
			{
				uint32		codeword = docCodes[codeIndex];
				float		codeScore;

				if (codeword >= codebookSize)
					continue;
				codeScore = tokenScores[codeword];
				if (codeScore > best)
					best = codeScore;
			}
			if (best > -FLT_MAX)
				score += (double) best;
		}
		if (pairs != NULL)
			*pairs += (uint64) query->count * (uint64) docCodeCount;
		return score;
	}

	for (int32 qi = 0; qi < query->count; qi++)
	{
		double		best = -DBL_MAX;
		double		weight =
			queryWeights != NULL ? (double) queryWeights[qi] : 1.0;
		const float *tokenScores =
			queryCodewordScores + (Size) qi * (Size) codebookSize;

		if (queryMask != NULL && queryMask[qi])
			continue;
		for (uint32 codeIndex = 0; codeIndex < docCodeCount; codeIndex++)
		{
			uint32		codeword = docCodes[codeIndex];
			double		codeScore;

			if (codeword >= codebookSize)
				continue;
			codeScore = (double) tokenScores[codeword];
			if (codeScore > best)
				best = codeScore;
		}
		if (pairs != NULL)
			*pairs += (uint64) docCodeCount;
		if (best > -DBL_MAX)
			score += weight * best;
	}
	return score;
}

static bool
PgturbohybridQuantizedInvertedStringIsEmpty(const char *value)
{
	return value == NULL || value[0] == '\0';
}

static const char *
PgturbohybridQuantizedInvertedCodebookSourceName(int source)
{
	switch (source)
	{
		case PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_EXTERNAL:
			return "external";
		case PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_DETERMINISTIC:
		default:
			return "deterministic";
	}
}

static void
PgturbohybridFreeQuantizedInvertedCodebookCache(void)
{
	if (pgturbohybrid_quantized_inverted_codebook_cache == NULL)
		return;
	pfree(pgturbohybrid_quantized_inverted_codebook_cache->path);
	pfree(pgturbohybrid_quantized_inverted_codebook_cache->checksum);
	pfree(pgturbohybrid_quantized_inverted_codebook_cache->values);
	pfree(pgturbohybrid_quantized_inverted_codebook_cache);
	pgturbohybrid_quantized_inverted_codebook_cache = NULL;
}

PgturbohybridQuantizedInvertedCodebook *
PgturbohybridLoadQuantizedInvertedCodebook(int32 dim)
{
	FILE	   *file;
	char		magic[64];
	char		checksum[128];
	int			headerDim;
	unsigned int codebookSize;
	Size		valueCount;
	Size		valueBytes;
	MemoryContext oldCtx;
	PgturbohybridQuantizedInvertedCodebook *loaded;

	if (PgturbohybridQuantizedInvertedStringIsEmpty(pgturbohybrid_multivector_quantized_inverted_codebook_path))
		ereport(ERROR,
				(errcode(ERRCODE_CONFIG_FILE_ERROR),
				 errmsg("external quantized_inverted_experimental codebook requires a configured codebook path"),
				 errhint("Set turbohybrid.multivector_quantized_inverted_codebook_path to a text file, or use turbohybrid.multivector_quantized_inverted_codebook = deterministic.")));
	if (pgturbohybrid_multivector_quantized_inverted_codebook_top_m < 1 ||
		pgturbohybrid_multivector_quantized_inverted_codebook_top_m > 16)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("quantized_inverted_experimental codebook top_m must be in [1, 16]")));
	if (pgturbohybrid_quantized_inverted_codebook_cache != NULL &&
		strcmp(pgturbohybrid_quantized_inverted_codebook_cache->path,
			   pgturbohybrid_multivector_quantized_inverted_codebook_path) == 0)
	{
		if (pgturbohybrid_quantized_inverted_codebook_cache->dim != dim)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("quantized_inverted_experimental codebook dimension mismatch"),
					 errdetail("Codebook dimension is %d but multivector dimension is %d.",
							   pgturbohybrid_quantized_inverted_codebook_cache->dim,
							   dim)));
		pgturbohybrid_quantized_inverted_codebook_cache->topM =
			(uint32) pgturbohybrid_multivector_quantized_inverted_codebook_top_m;
		return pgturbohybrid_quantized_inverted_codebook_cache;
	}

	file = AllocateFile(pgturbohybrid_multivector_quantized_inverted_codebook_path,
						"r");
	if (file == NULL)
		ereport(ERROR,
				(errcode_for_file_access(),
				 errmsg("could not open quantized_inverted_experimental codebook file \"%s\": %m",
						pgturbohybrid_multivector_quantized_inverted_codebook_path)));
	if (fscanf(file, "%63s %d %u %127s",
			   magic, &headerDim, &codebookSize, checksum) != 4)
	{
		FreeFile(file);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("invalid quantized_inverted_experimental codebook header"),
				 errdetail("Expected: pgturbohybrid_quantized_inverted_codebook_v1 <dim> <codebook_size> <checksum>.")));
	}
	if (strcmp(magic, "pgturbohybrid_quantized_inverted_codebook_v1") != 0)
	{
		FreeFile(file);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("invalid quantized_inverted_experimental codebook magic \"%s\"",
						magic)));
	}
	if (headerDim <= 0 || headerDim != dim || codebookSize == 0)
	{
		FreeFile(file);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("quantized_inverted_experimental codebook dimensions do not match multivector dimension"),
				 errdetail("Codebook declares dim=%d codebook_size=%u, but multivector dimension is %d.",
						   headerDim, codebookSize, dim)));
	}
	valueCount = (Size) headerDim * (Size) codebookSize;
	if (codebookSize > PG_UINT32_MAX ||
		valueCount > (Size) (MaxAllocSize / sizeof(float)))
	{
		FreeFile(file);
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("quantized_inverted_experimental codebook is too large")));
	}
	valueBytes = valueCount * sizeof(float);
	oldCtx = MemoryContextSwitchTo(TopMemoryContext);
	loaded = palloc0(sizeof(PgturbohybridQuantizedInvertedCodebook));
	loaded->path =
		pstrdup(pgturbohybrid_multivector_quantized_inverted_codebook_path);
	loaded->checksum = pstrdup(checksum);
	loaded->dim = headerDim;
	loaded->codebookSize = (uint32) codebookSize;
	loaded->topM =
		(uint32) pgturbohybrid_multivector_quantized_inverted_codebook_top_m;
	loaded->valueBytes = valueBytes;
	loaded->values = palloc0(valueBytes);
	MemoryContextSwitchTo(oldCtx);

	for (Size i = 0; i < valueCount; i++)
	{
		double		value;

		if (fscanf(file, "%lf", &value) != 1)
		{
			FreeFile(file);
			PgturbohybridFreeQuantizedInvertedCodebookCache();
			pfree(loaded->path);
			pfree(loaded->checksum);
			pfree(loaded->values);
			pfree(loaded);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("quantized_inverted_experimental codebook file ended before all values were read")));
		}
		loaded->values[i] = (float) value;
	}
	FreeFile(file);
	PgturbohybridFreeQuantizedInvertedCodebookCache();
	pgturbohybrid_quantized_inverted_codebook_cache = loaded;
	return loaded;
}

PgturbohybridMultiVectorQuantizedInvertedAssignment
PgturbohybridMultiVectorQuantizedInvertedBestCodewordAndScore(const PgturbohybridMultiVector *mv,
															 int32 token,
															 uint64 *assignmentUs)
{
	PgturbohybridMultiVectorQuantizedInvertedAssignment assignment;

	assignment.codeword = 0;
	assignment.score = 0.0;
	if (pgturbohybrid_multivector_quantized_inverted_codebook ==
		PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_EXTERNAL)
	{
		PgturbohybridQuantizedInvertedCodebook *codebook;
		TqDotProductF32BlockFunc blockDotProduct;
		instr_time	start;

		if (assignmentUs != NULL)
			INSTR_TIME_SET_CURRENT(start);
		codebook = PgturbohybridLoadQuantizedInvertedCodebook(mv->dim);
		blockDotProduct = TqResolveDotProductF32BlockKernel();
		assignment =
			PgturbohybridMultiVectorQuantizedInvertedBestExternalCodewordAndScore(mv,
																				  token,
																				  codebook,
																				  blockDotProduct);
		if (assignmentUs != NULL)
			PgturbohybridGraphAddElapsedUint64(assignmentUs, start);
		return assignment;
	}
	assignment.codeword =
		PgturbohybridMultiVectorDeterministicCodeword(mv, token);
	assignment.score =
		PgturbohybridMultiVectorDeterministicCodewordScore(mv, token,
														   assignment.codeword);
	return assignment;
}

PgturbohybridMultiVectorQuantizedInvertedAssignment
PgturbohybridMultiVectorQuantizedInvertedBestExternalCodewordAndScore(const PgturbohybridMultiVector *mv,
																	 int32 token,
																	 const PgturbohybridQuantizedInvertedCodebook *codebook,
																	 TqDotProductF32BlockFunc blockDotProduct)
{
	PgturbohybridMultiVectorQuantizedInvertedAssignment assignment;
	const float *values;
	double		best = -DBL_MAX;
	uint32		bestCodeword = 0;

	Assert(codebook != NULL);
	Assert(blockDotProduct != NULL);
	assignment.codeword = 0;
	assignment.score = 0.0;
	values = PgturbohybridMultiVectorValues(mv, token);
	for (uint32 codeword = 0; codeword < codebook->codebookSize;
		 codeword += PGTURBOHYBRID_MULTIVECTOR_COMPACT_DOT_BLOCK)
	{
		int32		blockCount =
			(int32) Min(codebook->codebookSize - codeword,
						(uint32) PGTURBOHYBRID_MULTIVECTOR_COMPACT_DOT_BLOCK);
		const float *centroids =
			&codebook->values[(Size) codeword * (Size) mv->dim];
		double		dots[PGTURBOHYBRID_MULTIVECTOR_COMPACT_DOT_BLOCK];

		blockDotProduct(centroids, values, mv->dim, blockCount, dots);
		for (int32 block = 0; block < blockCount; block++)
		{
			if (dots[block] > best)
			{
				best = dots[block];
				bestCodeword = codeword + (uint32) block;
			}
		}
	}
	assignment.codeword = bestCodeword;
	assignment.score = best;
	return assignment;
}

uint32
PgturbohybridMultiVectorQuantizedInvertedConfigurableCodewords(const PgturbohybridMultiVector *mv,
															  int32 token,
															  uint32 limit,
															  uint32 *codewords,
															  uint64 *assignmentUs)
{
	Assert(limit > 0);
	if (pgturbohybrid_multivector_quantized_inverted_codebook ==
		PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_EXTERNAL)
	{
		PgturbohybridQuantizedInvertedCodebook *codebook;
		PgturbohybridMultiVectorPostingScoreCandidate *selected;
		const float *values;
		uint32		selectedCount = 0;
		TqDotProductF32BlockFunc blockDotProduct;
		instr_time	start;

		if (assignmentUs != NULL)
			INSTR_TIME_SET_CURRENT(start);
		codebook = PgturbohybridLoadQuantizedInvertedCodebook(mv->dim);
		blockDotProduct = TqResolveDotProductF32BlockKernel();
		values = PgturbohybridMultiVectorValues(mv, token);
		limit = Min(limit, codebook->codebookSize);
		selected =
			palloc0(sizeof(PgturbohybridMultiVectorPostingScoreCandidate) *
					(Size) limit);
		for (uint32 codeword = 0; codeword < codebook->codebookSize;
			 codeword += PGTURBOHYBRID_MULTIVECTOR_COMPACT_DOT_BLOCK)
		{
			int32		blockCount =
				(int32) Min(codebook->codebookSize - codeword,
							(uint32) PGTURBOHYBRID_MULTIVECTOR_COMPACT_DOT_BLOCK);
			const float *centroids =
				&codebook->values[(Size) codeword * (Size) mv->dim];
			double		dots[PGTURBOHYBRID_MULTIVECTOR_COMPACT_DOT_BLOCK];

			blockDotProduct(centroids, values, mv->dim, blockCount, dots);
			for (int32 block = 0; block < blockCount; block++)
			{
				PgturbohybridMultiVectorPostingScoreCandidate scored;

				scored.docId = codeword + (uint32) block;
				scored.ordinal = 0;
				scored.score = dots[block];
				PgturbohybridMultiVectorPostingScoreCandidateOffer(selected,
																   &selectedCount,
																   limit,
																   &scored);
			}
		}
		for (uint32 i = 0; i < selectedCount; i++)
			codewords[i] = selected[i].docId;
		pfree(selected);
		if (assignmentUs != NULL)
			PgturbohybridGraphAddElapsedUint64(assignmentUs, start);
		return selectedCount;
	}
	return PgturbohybridMultiVectorDeterministicCodewords(mv, token, limit,
														  codewords);
}

double
PgturbohybridMultiVectorQuantizedInvertedCodewordScore(const PgturbohybridMultiVector *mv,
													   int32 token,
													   uint32 codeword)
{
	if (pgturbohybrid_multivector_quantized_inverted_codebook ==
		PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_EXTERNAL)
	{
		PgturbohybridQuantizedInvertedCodebook *codebook;
		const float *values;
		const float *centroid;
		double		dot = 0.0;

		codebook = PgturbohybridLoadQuantizedInvertedCodebook(mv->dim);
		if (codeword >= codebook->codebookSize)
			return -DBL_MAX;
		values = PgturbohybridMultiVectorValues(mv, token);
		centroid = &codebook->values[(Size) codeword * (Size) mv->dim];
		for (int32 dim = 0; dim < mv->dim; dim++)
			dot += (double) values[dim] * (double) centroid[dim];
		return dot;
	}
	return PgturbohybridMultiVectorDeterministicCodewordScore(mv, token,
															  codeword);
}

uint16
PgturbohybridMultiVectorCentroidPostingCodewordScorePayload(const PgturbohybridMultiVector *mv,
															int32 token,
															uint32 codeword)
{
	const float *values;
	uint32		dim;
	double		value;
	double		scaled;

	Assert(token >= 0 && token < mv->count);
	dim = codeword / 2U;
	if (dim >= (uint32) mv->dim)
		return 0;
	values = PgturbohybridMultiVectorValues(mv, token);
	value = (codeword & 1U) ? -(double) values[dim] : (double) values[dim];
	if (!isfinite(value) || value <= 0.0)
		return 0;
	scaled = Min(value, 1.0) * 65535.0;
	if (scaled >= 65535.0)
		return UINT16_MAX;
	return (uint16) rint(Max(0.0, scaled));
}

static double
PgturbohybridMultiVectorCentroidPostingPayloadScore(uint16 payload)
{
	return (double) payload / 65535.0;
}

uint16
PgturbohybridMultiVectorQuantizedInvertedScorePayload(const PgturbohybridMultiVector *mv,
													  int32 token)
{
	const float *values;
	double		norm2 = 0.0;
	double		scaled;

	values = PgturbohybridMultiVectorValues(mv, token);
	for (int32 dim = 0; dim < mv->dim; dim++)
		norm2 += (double) values[dim] * (double) values[dim];
	if (norm2 <= 0.0 || !isfinite(norm2))
		return 0;

	scaled = sqrt(norm2) * 65535.0;
	if (scaled >= 65535.0)
		return UINT16_MAX;
	return (uint16) rint(Max(0.0, scaled));
}

static int16
PgturbohybridMultiVectorQuantizedInvertedCompactPayload(uint16 payload)
{
	return (int16) Min((uint32) payload, (uint32) SHRT_MAX);
}

static int16
PgturbohybridMultiVectorQuantizedInvertedSignedCompactPayload(uint16 payload)
{
	return (int16) payload;
}

int16
PgturbohybridMultiVectorQuantizedInvertedCompactScorePayload(double score)
{
	double		clamped;

	if (!isfinite(score))
		return 0;
	clamped = Max(-1.0, Min(1.0, score));
	if (clamped >= 1.0)
		return SHRT_MAX;
	if (clamped <= -1.0)
		return -SHRT_MAX;
	return (int16) rint(clamped * (double) SHRT_MAX);
}

static inline double
PgturbohybridMultiVectorQuantizedInvertedCompactRawScore(int64 raw)
{
	return (double) raw / ((double) SHRT_MAX * (double) SHRT_MAX);
}

static double
PgturbohybridMultiVectorTokenDot(const PgturbohybridMultiVector *a,
								 int32 aToken,
								 const PgturbohybridMultiVector *b,
								 int32 bToken)
{
	const float *aValues;
	const float *bValues;
	double		dot = 0.0;

	PgturbohybridCheckSameMultiVectorDims(a, b);
	aValues = PgturbohybridMultiVectorValues(a, aToken);
	bValues = PgturbohybridMultiVectorValues(b, bToken);
	for (int32 dim = 0; dim < a->dim; dim++)
		dot += (double) aValues[dim] * (double) bValues[dim];

	return dot;
}

static void
PgturbohybridMultiVectorDocumentNodeCheckStorageCapabilities(bool proxyOnlyIndex,
															 bool centroidOnlyIndex,
															 bool proxyGraph,
															 bool centroidLite,
															 bool quantizedInvertedExperimental,
															 bool quantizedInvertedCompactScoring,
															 int centroidMode)
{
	if (proxyOnlyIndex && !proxyGraph)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("multivector_doc_storage = proxy_only only supports proxy_vector document-node graph admission"),
				 errhint("REINDEX with multivector_doc_storage = f32, f16, or sq8 to use full document multivector sidecar candidate sources.")));
	if (centroidOnlyIndex &&
		!(proxyGraph || centroidLite ||
		  (quantizedInvertedExperimental && quantizedInvertedCompactScoring)))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("multivector_doc_storage = centroid_only only supports proxy_vector, centroid_lite, and compact quantized_inverted_experimental document-node graph admission"),
				 errhint("REINDEX with multivector_doc_storage = f32, f16, or sq8 to use full document multivector sidecar candidate sources.")));
	if (centroidLite &&
		centroidMode != PGTURBOHYBRID_MULTIVECTOR_CENTROIDS_KMEANS)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("centroid_lite multivector candidate source requires multivector_centroids = kmeans"),
				 errhint("REINDEX with multivector_graph = document_nodes, multivector_centroids = kmeans, or use turbohybrid.multivector_candidate_source = document_nodes.")));
}

static const char *
PgturbohybridMultiVectorDocumentNodeGraphWarning(bool proxyGraph,
												bool quantizedInvertedExperimental,
												bool centroidLite,
												bool compactTraversal,
												int docStorageKind)
{
	if (proxyGraph)
		return "document_node_proxy_vector_graph_traversal";
	if (quantizedInvertedExperimental)
		return "quantized_inverted_experimental_persisted_postings";
	if (centroidLite)
		return "document_node_centroid_lite_prefilter";
	if (compactTraversal &&
		docStorageKind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F16)
		return "document_node_f16_sidecar_graph_traversal";
	if (compactTraversal &&
		docStorageKind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_SQ8)
		return "document_node_sq8_sidecar_graph_traversal";
	return "document_node_f32_sidecar_graph_traversal";
}

static int
PgturbohybridMultiVectorDocumentNodeScan(IndexScanDesc scan,
										 PgturbohybridGraphScanOpaque so,
										 PgturbohybridGraphMetaPageData *meta,
										 const PgturbohybridMultiVector *query,
										 const float4 *queryWeights,
										 const bool *queryMask,
										 double queryWeightSum,
										 int targetK,
										 TqDenseCandidate **outCandidates,
										 MemoryContext resultCtx,
										 TqDenseCandidateStats *stats)
{
	PgturbohybridGraphScanStorage storage;
	PgturbohybridGraphCacheInitInfo cacheInfo;
	TqDenseCandidate *candidates;
	MemoryContext oldCtx;
	int			docLimit;
	int			rescoreLimit;
	int			candidateLimit;
	int			docCount = 0;
	int			exactRerankCount = 0;
	int			searchEf;
	int			searchEfBase;
	int			docStorageKind;
	int			exactRerankLimitOverride;
	int			proxyCandidateTarget = 0;
	int			proxyExactRerankK = 0;
	int			exactRerankKEffective = 0;
	int			proxyEncoder =
		PgturbohybridGraphGetMultiVectorProxyEncoderOption(scan->indexRelation);
	int			centroidMode =
		PgturbohybridGraphGetMultiVectorCentroidsOption(scan->indexRelation);
	int			indexDocStorage =
		PgturbohybridGraphGetMultiVectorDocStorageOption(scan->indexRelation);
	bool		proxyOnlyIndex;
	bool		centroidOnlyIndex;
	bool		fullSidecarAvailable;
	bool		exhaustiveScan;
	bool		compactTraversal = false;
	bool		explicitProxyVector =
		pgturbohybrid_multivector_candidate_source ==
		PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_PROXY_VECTOR;
	bool		defaultGraphSource =
		pgturbohybrid_multivector_candidate_source ==
		PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_GRAPH;
	bool		proxyGraph =
		explicitProxyVector ||
		defaultGraphSource;
	bool		exactDocScan =
		pgturbohybrid_multivector_candidate_source ==
		PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_EXACT_DOC_SCAN;
	bool		qdrantLikeProxyVector =
		proxyGraph &&
		pgturbohybrid_multivector_branch_plan ==
		PGTURBOHYBRID_BRANCH_PLAN_QDRANT_LIKE;
	bool		centroidMeanProxy =
		proxyEncoder ==
		PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_CENTROID_MEAN;
	bool		proxyDocumentCompactRescore = false;
	bool		proxyLazySidecarVectors = false;
	bool		postingSidecarCache = false;
	bool		centroidLite =
		pgturbohybrid_multivector_candidate_source ==
		PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_CENTROID_LITE;
	bool		centroidLiteCompactScoring =
		centroidLite &&
		(pgturbohybrid_multivector_centroid_lite_posting_selection ==
		 PGTURBOHYBRID_MULTIVECTOR_POSTING_SELECTION_SCORE_TOPK ||
		 pgturbohybrid_multivector_centroid_lite_posting_selection ==
		 PGTURBOHYBRID_MULTIVECTOR_POSTING_SELECTION_UNION_SCORE);
	bool		centroidLiteUnionScore =
		centroidLite &&
		pgturbohybrid_multivector_centroid_lite_posting_selection ==
		PGTURBOHYBRID_MULTIVECTOR_POSTING_SELECTION_UNION_SCORE;
	bool		centroidLiteDocCentroidMaxsimScoring =
		centroidLiteCompactScoring &&
		pgturbohybrid_multivector_centroid_lite_candidate_scoring ==
		PGTURBOHYBRID_MULTIVECTOR_CENTROID_LITE_CANDIDATE_SCORING_DOC_CENTROID_MAXSIM;
	bool		centroidLiteCodewordMaxsimScoring =
		centroidLiteCompactScoring &&
		pgturbohybrid_multivector_centroid_lite_candidate_scoring ==
		PGTURBOHYBRID_MULTIVECTOR_CENTROID_LITE_CANDIDATE_SCORING_CODEWORD_MAXSIM;
	bool		centroidBitsetPrefilter =
		centroidLite &&
		pgturbohybrid_multivector_centroid_lite_bitset_prefilter ==
		PGTURBOHYBRID_MULTIVECTOR_CENTROID_LITE_BITSET_PREFILTER_EXPERIMENTAL;
	bool		centroidUpperBoundPruning =
		centroidLite &&
		pgturbohybrid_multivector_centroid_lite_pruning ==
		PGTURBOHYBRID_MULTIVECTOR_CENTROID_LITE_PRUNING_SAFE_UPPER_BOUND;
	bool		centroidScoreBoundPruning =
		centroidLite &&
		pgturbohybrid_multivector_centroid_lite_pruning ==
		PGTURBOHYBRID_MULTIVECTOR_CENTROID_LITE_PRUNING_SCORE_BOUND_EXPERIMENTAL;
	bool		quantizedInvertedExperimental =
		pgturbohybrid_multivector_candidate_source ==
		PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_QUANTIZED_INVERTED_EXPERIMENTAL;
	bool		quantizedInvertedCompactScoring =
		quantizedInvertedExperimental &&
		pgturbohybrid_multivector_quantized_inverted_compact_scoring ==
		PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_COMPACT_SCORING_EXPERIMENTAL;
	int			quantizedInvertedPrecompactMode =
		quantizedInvertedExperimental ?
		pgturbohybrid_multivector_quantized_inverted_precompact :
		PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_PRECOMPACT_OFF;
	bool		quantizedInvertedPrecompactEnabled =
		quantizedInvertedCompactScoring &&
		quantizedInvertedPrecompactMode !=
		PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_PRECOMPACT_OFF;
	bool		quantizedInvertedPrecompactReservoir =
		quantizedInvertedPrecompactEnabled &&
		quantizedInvertedPrecompactMode ==
		PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_PRECOMPACT_CENTROID_MAXSIM_RESERVOIR;
	bool		quantizedInvertedTokenCoverageLinear =
		quantizedInvertedExperimental &&
		pgturbohybrid_multivector_quantized_inverted_token_coverage ==
		PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_TOKEN_COVERAGE_LINEAR;
	bool		quantizedInvertedScoreBoundPruning =
		quantizedInvertedExperimental &&
		pgturbohybrid_multivector_quantized_inverted_pruning ==
		PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_PRUNING_SCORE_BOUND_EXPERIMENTAL;
	uint32		quantizedInvertedMinTokenMatches =
		quantizedInvertedExperimental ?
		(uint32) Max(pgturbohybrid_multivector_quantized_inverted_min_token_matches,
					 0) : 0;
	bool		quantizedInvertedNeedsTokenMatches =
		quantizedInvertedTokenCoverageLinear ||
		quantizedInvertedMinTokenMatches > 0 ||
		quantizedInvertedScoreBoundPruning ||
		quantizedInvertedPrecompactReservoir;
	bool		candidateSourceNeedsLoadedDocVectors;
	bool		exhaustiveSidecarScan;
	bool		documentNodesSource =
		pgturbohybrid_multivector_candidate_source ==
		PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_DOCUMENT_NODES;
	bool		proxyReservoirsEnabled =
		explicitProxyVector &&
		pgturbohybrid_multivector_candidate_reservoirs !=
		PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_RESERVOIRS_OFF;
	uint64		docsScored = 0;
	uint64		edgesVisited = 0;
	uint64		maxsimPairs = 0;
	uint64		exactPairs = 0;
	PgturbohybridMultiVectorExactRerankWorkStats exactStats;
	uint64		quantizedScores = 0;
	uint64		compactMaxsimScoreUs = 0;
	uint64		compactMaxsimCacheHits = 0;
	uint64		compactMaxsimCacheMisses = 0;
	uint64		compactMaxsimBoundChecks = 0;
	uint64		compactMaxsimDocsPruned = 0;
	uint64		compactMaxsimTokensSkipped = 0;
	uint64		compactBytes = 0;
	uint64		originalTokens = 0;
	uint64		pooledTokens = 0;
	uint64		centroidListsVisited = 0;
	uint64		centroidDocsTouched = 0;
	uint64		centroidPrunedDocs = 0;
	uint64		centroidPostingsSelected = 0;
	uint64		centroidPostingsSkipped = 0;
	uint32		centroidPostingLimitPerToken =
		(uint32) Max(pgturbohybrid_multivector_centroid_lite_max_postings_per_token,
					 0);
	uint32		centroidCodewordTopM =
		(uint32) Max(pgturbohybrid_multivector_centroid_lite_codeword_top_m,
					 1);
	bool	   *centroidLiteSelectedCodewords = NULL;
	uint32		centroidLitePostingCodebookSize = 0;
	uint32		centroidBitsetMinTokenMatches =
		(uint32) Max(pgturbohybrid_multivector_centroid_lite_bitset_min_token_matches,
					 1);
	double		centroidScoreThreshold =
		pgturbohybrid_multivector_centroid_lite_score_threshold;
	double		centroidScoreDropFromBest =
		pgturbohybrid_multivector_centroid_lite_score_drop_from_best;
	uint64		centroidListsSkippedByThreshold = 0;
	uint64		centroidProbeUs = 0;
	uint64		centroidPostingScanUs = 0;
	uint64		centroidAccumulateUs = 0;
	uint64		centroidCandidateHeapUs = 0;
	const char *centroidPostingCapStrategy = "none";
	const char *centroidCandidateScoring =
		centroidLiteDocCentroidMaxsimScoring ?
		"doc_centroid_maxsim" :
		centroidLiteCodewordMaxsimScoring ? "codeword_maxsim" :
		centroidLiteCompactScoring ? "posting_payload" : "exact_centroid_token";
	uint32		centroidCandidates = 0;
	uint32		centroidBitsetListsUsed = 0;
	uint32		centroidBitsetDocsSet = 0;
	uint32		centroidBitsetDocsAfterThreshold = 0;
	uint64		centroidBitsetPrefilterUs = 0;
	uint64		centroidBitsetMemoryBytes = 0;
	uint64		centroidUpperBoundDocsChecked = 0;
	uint64		centroidUpperBoundDocsPruned = 0;
	uint64		centroidUpperBoundPruneUs = 0;
	uint64		centroidUpperBoundUnsafeFallbacks = 0;
	uint32		centroidCandidatesBeforeBound = 0;
	uint32		centroidCandidatesAfterBound = 0;
	uint32		centroidCountEffective = 0;
	uint32		centroidPrerankDocs = 0;
	uint64		centroidDocCentroidMaxsimPairs = 0;
	uint64		quantizedInvertedListsVisited = 0;
	uint64		quantizedInvertedPostingsTouched = 0;
	uint64		quantizedInvertedPostingsSelected = 0;
	uint64		quantizedInvertedPostingsSkipped = 0;
	uint32		quantizedInvertedPostingLimitPerToken =
		(uint32) Max(pgturbohybrid_multivector_quantized_inverted_max_postings_per_token,
					 0);
	const char *quantizedInvertedPostingCapStrategy = "none";
	uint64		quantizedInvertedDocsScored = 0;
	uint32		quantizedInvertedCandidates = 0;
	TqCompactCodeScoreFunc quantizedInvertedCompactScorer = NULL;
	TqCompactCodeScoreBatchFunc quantizedInvertedCompactBatchScorer = NULL;
	const char *quantizedInvertedCompactKernel = "off";
	uint64		quantizedInvertedCompactScoreUs = 0;
	uint64		quantizedInvertedCompactDocsScored = 0;
	uint64		quantizedInvertedCompactPayloadBytes = 0;
	const char *quantizedInvertedCompactDocOrder = "original";
	uint64		quantizedInvertedCompactInnerAllocations = 0;
	uint32		quantizedInvertedCompactActiveQueryTokens = 0;
	uint64		quantizedInvertedCompactPairsEvaluated = 0;
	uint64		quantizedInvertedCompactPairsSkipped = 0;
	uint64		quantizedInvertedCompactPrefetches = 0;
	bool		quantizedInvertedCompactTopKChangedVsScalar = false;
	int16	   *quantizedInvertedCompactBatchCodes = NULL;
	int64	   *quantizedInvertedCompactBatchScores = NULL;
	uint32		quantizedInvertedActiveQueryTokens = 0;
	uint64		quantizedInvertedTokenMatchesTotal = 0;
	uint32		quantizedInvertedTokenMatchesMax = 0;
	uint64		quantizedInvertedTokenMatchFilteredDocs = 0;
	uint64		quantizedInvertedScoreBoundDocsChecked = 0;
	uint64		quantizedInvertedScoreBoundDocsPruned = 0;
	uint64		quantizedInvertedScoreBoundPruneUs = 0;
	uint64		quantizedInvertedScoreBoundUnsafeFallbacks = 0;
	uint32		quantizedInvertedCandidatesBeforeBound = 0;
	uint32		quantizedInvertedCandidatesAfterBound = 0;
	uint32		quantizedInvertedDocsTouchedBeforePrecompact = 0;
	uint32		quantizedInvertedPrecompactScoreK =
		(uint32) Max(pgturbohybrid_multivector_quantized_inverted_precompact_score_k,
					 0);
	uint32		quantizedInvertedPrecompactCoverageK =
		(uint32) Max(pgturbohybrid_multivector_quantized_inverted_precompact_coverage_k,
					 0);
	uint32		quantizedInvertedPrecompactPerTokenK =
		(uint32) Max(pgturbohybrid_multivector_quantized_inverted_precompact_per_token_k,
					 0);
	uint32		quantizedInvertedCompactMaxDocs =
		(uint32) Max(pgturbohybrid_multivector_quantized_inverted_compact_max_docs,
					 0);
	uint32		quantizedInvertedPrecompactScoreDocs = 0;
	uint32		quantizedInvertedPrecompactCoverageDocs = 0;
	uint32		quantizedInvertedPrecompactPerTokenDocs = 0;
	uint32		quantizedInvertedPrecompactUnionDocs = 0;
	uint32		quantizedInvertedPrecompactDuplicates = 0;
	uint32		quantizedInvertedPrecompactPrunedDocs = 0;
	uint64		quantizedInvertedPrecompactUs = 0;
	uint32		quantizedInvertedCompactDocsSkippedByPrecompact = 0;
	uint32		multivectorReservoirScoreDocs = 0;
	uint32		multivectorReservoirCoverageDocs = 0;
	uint32		multivectorReservoirMeanDocs = 0;
	uint32		multivectorReservoirPerTokenDocs = 0;
	uint32		multivectorReservoirUnionDocs = 0;
	uint32		multivectorReservoirDuplicates = 0;
	int			proxyGraphHitCount = 0;
	int			proxyCandidateLimitEffective = 0;
	const char *proxyCandidateLimitSource = "none";
	uint64		sidecarInitialBytesTouched = 0;
	uint64		sidecarInitialPagesRead = 0;
	uint64		sidecarInitialVectorsLoaded = 0;
	uint64		sidecarCacheBuildBytes = 0;
	uint64		sidecarCacheBuildPagesRead = 0;
	uint64		sidecarCacheBuildUs = 0;
	uint64		sidecarQueryBytesTouched = 0;
	uint64		sidecarQueryPagesRead = 0;
	uint64		sidecarQueryVectorsLoaded = 0;
	uint64		sidecarQueryLoadUs = 0;
	uint64		proxyFullSidecarVectorsLoaded = 0;
	uint64		proxyFullSidecarBytesTouched = 0;
	uint64		proxyFullSidecarPagesRead = 0;
	uint64		proxyFullSidecarLoadUs = 0;
	uint64		proxyFullSidecarReconstructUs = 0;
	bool		sidecarCacheBuildThisQuery = false;
	bool		proxyVectorUsesFullSidecarForGraph = false;
	bool		proxyVectorNearExhaustiveSidecarTouch = false;
	const char *proxyVectorSidecarTouchReason = "none";
	int			docStorageCacheMode;
	const char *docGraphWarning;
	const char *docStorageKindName;
	const char *docStorageCacheModeName;
	const char *docStorageCacheRequestedName;
	const char *docAccumulatorKind;
	instr_time	phaseStart;
	PgturbohybridMultiVectorDocCompactStorage *compact = NULL;
	PgturbohybridMultiVectorDocSidecarAccessStats sidecarStats;
	Vector	   *proxyQuery = NULL;
	ItemPointerData proxyTopTid;
	bool		proxyTopTidValid = false;
	bool		proxyTop1Admission = false;
	uint32		proxyDocumentRescoreDocs = 0;
	PgturbohybridGraphMetaPageData *exactRerankSidecarMeta = NULL;
	PgturbohybridGraphScanStorage *exactRerankSidecarStorage = NULL;
	PgturbohybridMultiVectorDocSidecarAccessStats *exactRerankSidecarStats = NULL;
	bool		collectPhaseStats = stats != NULL;
	uint64		candidateSourceUs = 0;
	uint64		docGraphTraversalUs = 0;
	uint64		proxyCandidateUs = 0;
	uint64		proxyGraphTraversalUs = 0;
	uint64		proxyScoringUs = 0;
	uint64		learnedProjectionQueryEncodeUs = 0;
	uint64		centroidLitePostingUs = 0;
	uint64		quantizedInvertedPostingUs = 0;
	uint64		quantizedInvertedAssignmentUs = 0;
	uint64		quantizedInvertedQueryCodewordScoreUs = 0;
	const char *quantizedInvertedQueryCodewordKernel = "off";
	uint64		quantizedInvertedQueryCodewordScoresComputed = 0;
	uint64		quantizedInvertedQueryCodewordBlocks = 0;
	uint64		quantizedInvertedQueryCodewordTopkUs = 0;
	bool		quantizedInvertedQueryCodewordFullMatrixMaterialized = false;
	uint32		quantizedInvertedQueryCodewordActiveQueryTokens = 0;
	uint32		quantizedInvertedQueryCodewordSkippedQueryTokens = 0;
	uint64		sidecarLoadUs = 0;
	uint64		finalSortUs = 0;
	instr_time	candidateSourceStart;
	instr_time	subphaseStart;
	instr_time	sortStart;

	if (outCandidates == NULL)
		return 0;
	proxyOnlyIndex =
		indexDocStorage == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_PROXY_ONLY ||
		((meta->tqMultivectorDocMapFlags &
		  PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_PROXY_ONLY) != 0);
	centroidOnlyIndex =
		indexDocStorage ==
		PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CENTROID_ONLY ||
		(((meta->tqMultivectorDocMapFlags &
		   PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_CENTROIDS) != 0) &&
		 ((meta->tqMultivectorDocMapFlags &
		   PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_DOC_VECTORS) == 0) &&
		 !proxyOnlyIndex);
	fullSidecarAvailable =
		(meta->tqMultivectorDocMapFlags &
		 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_DOC_VECTORS) != 0;
	if (proxyOnlyIndex)
		qdrantLikeProxyVector = false;
	if (centroidOnlyIndex)
		qdrantLikeProxyVector = false;
	PgturbohybridMultiVectorDocumentNodeCheckStorageCapabilities(proxyOnlyIndex,
																 centroidOnlyIndex,
																 proxyGraph,
																 centroidLite,
																 quantizedInvertedExperimental,
																 quantizedInvertedCompactScoring,
																 centroidMode);

	oldCtx = MemoryContextSwitchTo(resultCtx);
	*outCandidates = NULL;
	if (proxyGraph)
	{
		candidateLimit =
			PgturbohybridMultiVectorDocumentProxyCandidateTarget(so,
																targetK,
																meta->tqMultivectorDocCount,
																&proxyExactRerankK);
		docLimit = candidateLimit;
		rescoreLimit = proxyExactRerankK;
		proxyCandidateTarget = candidateLimit;
	}
	else
	{
		docLimit = PgturbohybridMultiVectorDocCandidateLimit(targetK);
		rescoreLimit =
			pgturbohybrid_multivector_doc_graph_rescore_k > 0 ?
			pgturbohybrid_multivector_doc_graph_rescore_k :
			pgturbohybrid_multivector_doc_candidate_k;
		rescoreLimit =
			(int) Min((int64) Max(rescoreLimit, docLimit),
					  (int64) meta->tqMultivectorDocCount);
		candidateLimit =
			(int) Min((int64) meta->tqMultivectorDocCount,
					  (int64) Max(docLimit,
								  rescoreLimit *
								  pgturbohybrid_multivector_doc_graph_oversampling));
	}
	candidates = palloc0(sizeof(TqDenseCandidate) * candidateLimit);

	so->graphM = meta->m;
	so->graphEfConstruction = meta->efConstruction;
	so->graphExactStorage =
		((meta->tqFlags & PGTURBOHYBRID_GRAPH_EXACT_FREE) == 0 &&
		 BlockNumberIsValid(meta->tqExactStartBlkno));
	so->graphBuildExactDistances =
		(meta->tqFlags & PGTURBOHYBRID_GRAPH_TQ_EXACT_BUILD) != 0;
	so->graphBuildDistanceMode = so->graphBuildExactDistances ?
		PGTURBOHYBRID_DENSE_BUILD_DISTANCE_EXACT :
		PGTURBOHYBRID_DENSE_BUILD_DISTANCE_CODE;
	so->graphBuildFastEdges =
		(meta->tqFlags & PGTURBOHYBRID_GRAPH_TQ_FAST_BUILD_EDGES) != 0;
	so->graphBuildNeighborSelectReason =
		PGTURBOHYBRID_GRAPH_TQ_BUILD_NEIGHBOR_REASON(meta->tqFlags);
	so->graphDenseRequestedK = targetK;
	so->graphEffectiveResultTarget = candidateLimit;
	searchEfBase =
		pgturbohybrid_multivector_doc_graph_search_ef > 0 ?
		pgturbohybrid_multivector_doc_graph_search_ef : so->efSearch;
	searchEf = Min(Max(searchEfBase, 1), (int) meta->tqMultivectorDocCount);
	exhaustiveScan = searchEf >= (int) meta->tqMultivectorDocCount;
	if (proxyGraph)
	{
		proxyCandidateLimitEffective = candidateLimit;
		proxyCandidateLimitSource = "candidate_target";
		if (proxyCandidateLimitEffective > searchEf)
		{
			proxyCandidateLimitEffective = searchEf;
			proxyCandidateLimitSource = "search_ef";
		}
		if (proxyCandidateLimitEffective > (int) meta->tqMultivectorDocCount)
		{
			proxyCandidateLimitEffective = (int) meta->tqMultivectorDocCount;
			proxyCandidateLimitSource = "doc_count";
		}
		if (proxyCandidateLimitEffective < 0)
			proxyCandidateLimitEffective = 0;
	}
	docStorageKind = proxyOnlyIndex ?
		PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_PROXY_ONLY :
		centroidOnlyIndex ?
		PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CENTROID_ONLY :
		indexDocStorage;
	if (!proxyOnlyIndex &&
		docStorageKind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_PROXY_ONLY)
		docStorageKind = PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32;
	if (docStorageKind != PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32 &&
		docStorageKind != PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F16 &&
		docStorageKind != PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_SQ8 &&
		docStorageKind != PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CENTROID_ONLY &&
		docStorageKind != PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_PROXY_ONLY)
		docStorageKind = PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32;
	proxyLazySidecarVectors =
		!proxyOnlyIndex &&
		!centroidOnlyIndex &&
		fullSidecarAvailable &&
		proxyGraph &&
		!qdrantLikeProxyVector &&
		!PgturbohybridMultiVectorIndexUsesContextLevel(scan->indexRelation);
	postingSidecarCache =
		(centroidLite || quantizedInvertedExperimental) &&
		!PgturbohybridMultiVectorIndexUsesContextLevel(scan->indexRelation);
	docStorageCacheRequestedName =
		PgturbohybridMultiVectorDocStorageCacheModeName(pgturbohybrid_multivector_doc_storage_cache);
	docStorageCacheMode =
		PgturbohybridMultiVectorChooseDocStorageCacheMode(meta);
	if (exactDocScan)
		docStorageCacheMode =
			PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_RESIDENT;
	if (centroidLiteCompactScoring &&
		!centroidUpperBoundPruning &&
		!centroidLiteDocCentroidMaxsimScoring &&
		!centroidOnlyIndex &&
		pgturbohybrid_multivector_doc_storage_cache !=
		PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_RESIDENT)
		docStorageCacheMode =
			PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_PAGED;
	if (proxyLazySidecarVectors &&
		pgturbohybrid_multivector_doc_storage_cache !=
		PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_RESIDENT)
		docStorageCacheMode =
			PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_PAGED;
	if (docStorageCacheMode ==
		PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_PAGED)
	{
		if ((centroidLite || quantizedInvertedExperimental) &&
			!postingSidecarCache)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("paged document-node sidecar cache does not support this multivector candidate source"),
					 errhint("Use turbohybrid.multivector_candidate_source = document_nodes or proxy_vector, or set turbohybrid.multivector_doc_storage_cache = resident.")));
		if (proxyGraph)
			docStorageKind = PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32;
	}
	compactTraversal =
		!exhaustiveScan &&
		!proxyGraph &&
		!centroidLite &&
		!quantizedInvertedExperimental &&
		!proxyOnlyIndex &&
		!centroidOnlyIndex &&
		docStorageKind != PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32;
	docStorageKindName =
		PgturbohybridMultiVectorDocStorageKindName(docStorageKind);
	proxyDocumentCompactRescore =
		qdrantLikeProxyVector &&
		!proxyOnlyIndex &&
		!centroidOnlyIndex &&
		docStorageKind != PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32;
	exhaustiveSidecarScan =
		exhaustiveScan &&
		!proxyOnlyIndex &&
		pgturbohybrid_multivector_candidate_source !=
		PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_CENTROID_LITE &&
		pgturbohybrid_multivector_candidate_source !=
		PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_QUANTIZED_INVERTED_EXPERIMENTAL;
	candidateSourceNeedsLoadedDocVectors =
		exactDocScan || exhaustiveSidecarScan || documentNodesSource ||
		proxyDocumentCompactRescore ||
		(pgturbohybrid_multivector_candidate_source ==
		 PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_QUANTIZED_INVERTED_EXPERIMENTAL &&
		 !quantizedInvertedCompactScoring);
	docStorageCacheModeName =
		PgturbohybridMultiVectorDocStorageCacheModeName(docStorageCacheMode);
	memset(&sidecarStats, 0, sizeof(sidecarStats));
	memset(&exactStats, 0, sizeof(exactStats));
	strlcpy(sidecarStats.cacheMode, docStorageCacheModeName,
			sizeof(sidecarStats.cacheMode));
	if (proxyGraph)
		docAccumulatorKind = "doc_proxy_graph";
	else if (centroidLite)
		docAccumulatorKind = "centroid_lite";
	else if (quantizedInvertedExperimental)
		docAccumulatorKind = "quantized_inverted_experimental";
	else if (compactTraversal)
	{
		if (docStorageKind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F16)
			docAccumulatorKind = "doc_graph_f16";
		else
			docAccumulatorKind = "doc_graph_sq8";
	}
	else
		docAccumulatorKind = "doc_graph_f32";
	so->graphEffectiveSearchEf = searchEf;
	so->graphEffectiveRescoreBand = 0;

	INSTR_TIME_SET_CURRENT(phaseStart);
	PgturbohybridGraphInitScanStorage(scan->indexRelation, meta, &storage,
									  &cacheInfo);
	so->graphNativeCacheMode = cacheInfo.mode;
	so->graphNativeCachePolicy = cacheInfo.policy;
	so->graphNativeCacheReason = cacheInfo.reason;
	so->graphNativeCacheUsed = cacheInfo.used;
	so->graphNativeCacheReused = cacheInfo.reused;
	so->graphNativeCacheBuiltThisScan = cacheInfo.builtThisScan;
	so->graphNativeCacheAttachUs = cacheInfo.attachUs;
	so->graphNativeCacheBuildUs = cacheInfo.buildUs;
	so->graphNativeCacheWaitUs = cacheInfo.waitUs;
	so->graphNativeCacheRefcount = cacheInfo.refcount;
	so->graphNativeCacheBytes = cacheInfo.totalBytes;
	so->graphNativeCacheCodeBytes = cacheInfo.codeBytes;
	so->graphNativeCacheAdjBytes = cacheInfo.adjBytes;
	so->graphNativeCacheExactBytes = cacheInfo.exactBytes;
	so->graphNativeCacheWarning = cacheInfo.warning;
	so->graphNativeCacheWarningReason = cacheInfo.warningReason;
	so->graphCodeBufferLockWaitUs += cacheInfo.codeBufferLockWaitUs;
	so->graphAdjBufferLockWaitUs += cacheInfo.adjBufferLockWaitUs;
	PgturbohybridGraphAddElapsedUs(&so->graphPrepareUs, phaseStart);

	if ((proxyLazySidecarVectors || compactTraversal) &&
		docStorageCacheMode ==
		PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_PAGED &&
		!storage.multivectorDocMapLoaded)
		(void) PgturbohybridGraphAttachMultiVectorDocSidecarCache(scan->indexRelation,
																  meta,
																  &storage,
																  true,
																  proxyLazySidecarVectors,
																  &cacheInfo);
	if (docStorageCacheMode ==
		PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_PAGED)
	{
		if (storage.multivectorDocMapLoaded &&
			!storage.multivectorDocVectorsPaged)
			ereport(ERROR,
					(errcode(ERRCODE_INTERNAL_ERROR),
					 errmsg("paged document-node sidecar cache resolved to resident storage"),
					 errhint("Disable or invalidate the native graph cache and retry with turbohybrid.multivector_doc_storage_cache = paged.")));
		storage.ctx = resultCtx;
		storage.multivectorDocVectorsPaged = true;
		storage.multivectorDocContextsSkipped =
			proxyLazySidecarVectors || postingSidecarCache;
	}
	else if ((proxyLazySidecarVectors || postingSidecarCache) &&
			 !storage.multivectorDocMapLoaded)
	{
		/*
		 * Normal proxy-vector admission uses only the document proxy graph.
		 * Centroid/quantized posting admission uses the persisted posting
		 * metadata.  Keep full document multivectors lazy so only bounded exact
		 * rerank candidates reconstruct heap/sidecar vectors.
		 */
		if (docStorageCacheMode ==
			PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_RESIDENT)
			(void) PgturbohybridGraphAttachMultiVectorDocSidecarCache(scan->indexRelation,
																	  meta,
																	  &storage,
																	  false,
																	  proxyLazySidecarVectors ||
																	  postingSidecarCache,
																	  &cacheInfo);
		if (!storage.multivectorDocMapLoaded)
		{
			storage.ctx = resultCtx;
			storage.multivectorDocVectorsPaged = true;
			storage.multivectorDocContextsSkipped = true;
		}
	}
	if (centroidLiteCompactScoring &&
		!centroidUpperBoundPruning &&
		!centroidLiteDocCentroidMaxsimScoring)
		storage.multivectorDocCentroidVectorsSkipped = true;
	if (centroidLiteCompactScoring && !centroidUpperBoundPruning)
	{
		centroidLitePostingCodebookSize = (uint32) query->dim * 2U;
		centroidLiteSelectedCodewords =
			palloc0(PgturbohybridCheckedArrayBytes(sizeof(bool),
												   centroidLitePostingCodebookSize,
												   "pgturbohybrid centroid_lite selected codewords"));
			for (int32 qi = 0; qi < query->count; qi++)
			{
				uint32	   *probeCodewords;
				double	   *probeScores;
				uint32		probeLimit =
					(uint32) Max(pgturbohybrid_multivector_centroid_lite_probe_centroids_per_token,
								 1);
				uint32		probeCount;
				double		bestProbeScore = 0.0;
				bool		haveBestProbeScore = false;

				if (queryMask != NULL && queryMask[qi])
					continue;
				probeCodewords =
					palloc0(sizeof(uint32) * (Size) probeLimit);
				probeScores =
					palloc0(sizeof(double) * (Size) probeLimit);
				probeCount =
					PgturbohybridMultiVectorDeterministicCodewords(query, qi,
																   probeLimit,
																   probeCodewords);
				for (uint32 probeIndex = 0; probeIndex < probeCount; probeIndex++)
				{
					uint32		queryCodeword = probeCodewords[probeIndex];
					double		queryCodewordScore;

					if (queryCodeword >= centroidLitePostingCodebookSize)
						ereport(ERROR,
								(errcode(ERRCODE_INDEX_CORRUPTED),
								 errmsg("document-node centroid_lite posting sidecar references an out-of-range centroid"),
								 errhint("REINDEX with multivector_graph = document_nodes and multivector_centroids = kmeans to rebuild persisted centroid posting tuples.")));
					queryCodewordScore =
						PgturbohybridMultiVectorDeterministicCodewordScore(query,
																		   qi,
																		   queryCodeword);
					probeScores[probeIndex] = queryCodewordScore;
					if (!haveBestProbeScore ||
						queryCodewordScore > bestProbeScore)
					{
						bestProbeScore = queryCodewordScore;
						haveBestProbeScore = true;
					}
				}
				for (uint32 probeIndex = 0; probeIndex < probeCount; probeIndex++)
				{
					uint32		queryCodeword = probeCodewords[probeIndex];
					double		queryCodewordScore = probeScores[probeIndex];

					if (queryCodewordScore >= centroidScoreThreshold &&
						(centroidScoreDropFromBest < 0.0 ||
						 !haveBestProbeScore ||
						 queryCodewordScore >= bestProbeScore - centroidScoreDropFromBest))
						centroidLiteSelectedCodewords[queryCodeword] = true;
				}
				pfree(probeScores);
				pfree(probeCodewords);
			}
		}
	if (storage.multivectorDocMapLoaded &&
		((candidateSourceNeedsLoadedDocVectors &&
		  !storage.multivectorDocVectorsLoaded) ||
		 (centroidLite &&
		  (!storage.multivectorCentroidPostingsLoaded ||
		   storage.multivectorCentroidPostings == NULL ||
		   storage.multivectorCentroidPostingListOffsets == NULL ||
		   storage.multivectorCentroidPostingCodebookSize == 0 ||
		   (centroidLiteCodewordMaxsimScoring &&
			(!storage.multivectorCentroidDocCodesLoaded ||
			 storage.multivectorCentroidDocCodeOffsets == NULL ||
			 storage.multivectorCentroidDocCodes == NULL)) ||
		   ((!centroidLiteCompactScoring ||
			 centroidLiteDocCentroidMaxsimScoring) &&
			!storage.multivectorDocCentroidsLoaded))) ||
		 (quantizedInvertedExperimental &&
		  (!storage.multivectorQuantizedInvertedPostingsLoaded ||
		   storage.multivectorQuantizedInvertedPostings == NULL ||
		   storage.multivectorQuantizedInvertedListOffsets == NULL ||
		   storage.multivectorQuantizedInvertedCodebookSize == 0 ||
		   storage.multivectorQuantizedInvertedCodebookDim == 0))))
	{
		/*
		 * Native graph cache views may carry the document map without the
		 * posting sidecars used by centroid_lite/quantized_inverted admission.
		 * Reload the docmap sidecar into scan-local storage so posting-based
		 * candidate generation never runs against a partial cached view.
		 */
		storage.multivectorNodeMap = NULL;
		storage.multivectorDocMap = NULL;
		storage.multivectorDocVectors = NULL;
		storage.multivectorDocVectorChunks = NULL;
		storage.multivectorDocVectorFirstChunk = NULL;
		storage.multivectorDocVectorChunkCounts = NULL;
		storage.multivectorDocVectorChunkCount = 0;
		storage.multivectorDocVectorChunkCapacity = 0;
		storage.multivectorDocCentroids = NULL;
		storage.multivectorDocCentroidResiduals = NULL;
		storage.multivectorCentroidPostings = NULL;
		storage.multivectorCentroidPostingListOffsets = NULL;
		storage.multivectorCentroidDocCodeOffsets = NULL;
		storage.multivectorCentroidDocCodes = NULL;
		storage.multivectorCentroidPostingCodebookSize = 0;
		storage.multivectorCentroidPostingCount = 0;
		storage.multivectorCentroidDocCodeCount = 0;
		storage.multivectorQuantizedInvertedPostings = NULL;
		storage.multivectorQuantizedInvertedListOffsets = NULL;
		storage.multivectorQuantizedInvertedDocCodeOffsets = NULL;
		storage.multivectorQuantizedInvertedDocCodes = NULL;
		storage.multivectorQuantizedInvertedCodebookSize = 0;
		storage.multivectorQuantizedInvertedCodebookDim = 0;
		storage.multivectorQuantizedInvertedCodebookTopM = 0;
		storage.multivectorQuantizedInvertedCodebookSource =
			PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_DETERMINISTIC;
		storage.multivectorQuantizedInvertedCodebookChecksum[0] = '\0';
		storage.multivectorQuantizedInvertedPostingCount = 0;
		storage.multivectorQuantizedInvertedDocCodeCount = 0;
		storage.multivectorDocCount = 0;
		storage.multivectorDocMapBytes = 0;
		storage.multivectorDocVectorsLoaded = false;
		storage.multivectorDocVectorsPaged = false;
		storage.multivectorDocContextsSkipped = false;
		storage.multivectorDocCentroidVectorsSkipped = false;
		storage.multivectorDocCentroidsLoaded = false;
		storage.multivectorCentroidPostingsLoaded = false;
		storage.multivectorCentroidDocCodesLoaded = false;
		storage.multivectorQuantizedInvertedPostingsLoaded = false;
		storage.multivectorQuantizedInvertedDocCodesLoaded = false;
		storage.multivectorDocMapLoaded = false;
	}
	if ((proxyLazySidecarVectors || compactTraversal) &&
		docStorageCacheMode ==
		PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_PAGED &&
		!storage.multivectorDocMapLoaded)
		(void) PgturbohybridGraphAttachMultiVectorDocSidecarCache(scan->indexRelation,
																  meta,
																  &storage,
																  true,
																  proxyLazySidecarVectors,
																  &cacheInfo);
	if (collectPhaseStats)
		INSTR_TIME_SET_CURRENT(subphaseStart);
	sidecarCacheBuildThisQuery = cacheInfo.docSidecarCacheBuiltThisScan;
	if (sidecarCacheBuildThisQuery)
	{
		sidecarCacheBuildBytes = cacheInfo.docSidecarCacheBytesTouched;
		sidecarCacheBuildPagesRead = cacheInfo.docSidecarCachePagesRead;
		sidecarCacheBuildUs = (uint64) Max(cacheInfo.docSidecarCacheBuildUs, 0);
	}
		if (centroidLiteCompactScoring &&
			!centroidUpperBoundPruning &&
			!centroidLiteDocCentroidMaxsimScoring &&
			!storage.multivectorDocMapLoaded &&
			centroidLiteSelectedCodewords != NULL)
			(void) PgturbohybridGraphLoadCentroidLiteCompactDocMapWithStats(scan->indexRelation,
																			meta,
																			&storage,
																			centroidLiteSelectedCodewords,
																			centroidLitePostingCodebookSize,
																			centroidLiteCodewordMaxsimScoring,
																			&sidecarStats);
	else
		(void) PgturbohybridGraphLoadMultiVectorDocMapWithStats(scan->indexRelation,
																meta,
																&storage,
																true,
																&sidecarStats);
	if (collectPhaseStats)
		PgturbohybridGraphAddElapsedUint64(&sidecarLoadUs, subphaseStart);
	sidecarInitialBytesTouched = sidecarStats.bytesTouched;
	sidecarInitialPagesRead = sidecarStats.pagesRead;
	sidecarInitialVectorsLoaded =
		sidecarStats.vectorsLoaded + sidecarStats.residentVectorsLoaded;
	if (sidecarCacheBuildThisQuery)
	{
		sidecarCacheBuildUs += sidecarLoadUs;
	}
	else
	{
		sidecarQueryBytesTouched = sidecarInitialBytesTouched;
		sidecarQueryPagesRead = sidecarInitialPagesRead;
		sidecarQueryVectorsLoaded = sidecarInitialVectorsLoaded;
		sidecarQueryLoadUs = sidecarLoadUs;
	}
		if (!storage.multivectorDocVectorsLoaded)
		{
			if (candidateSourceNeedsLoadedDocVectors)
					ereport(ERROR,
							(errcode(ERRCODE_INDEX_CORRUPTED),
							 errmsg("document-node multivector sidecar is not loaded"),
							 errhint("REINDEX the index to rebuild the document-node multivector sidecar.")));
		}
	if (centroidLite &&
		((meta->tqMultivectorDocMapFlags &
		  PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_CENTROIDS) == 0 ||
		 (meta->tqMultivectorDocMapFlags &
		  PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_CENTROID_POSTINGS) == 0 ||
		 !storage.multivectorCentroidPostingsLoaded ||
		 (centroidLiteCodewordMaxsimScoring &&
		  (!storage.multivectorCentroidDocCodesLoaded ||
		   storage.multivectorCentroidDocCodeOffsets == NULL ||
		   storage.multivectorCentroidDocCodes == NULL)) ||
		 ((!centroidLiteCompactScoring ||
		   centroidLiteDocCentroidMaxsimScoring) &&
		  !storage.multivectorDocCentroidsLoaded)))
		ereport(ERROR,
				(errcode(ERRCODE_INDEX_CORRUPTED),
				 errmsg("document-node centroid_lite sidecar is not loaded"),
				 errhint("REINDEX with multivector_graph = document_nodes and multivector_centroids = kmeans to rebuild persisted centroid sidecar and posting tuples.")));
	if (centroidMeanProxy &&
		((meta->tqMultivectorDocMapFlags &
		  PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_CENTROIDS) == 0 ||
		 !storage.multivectorDocCentroidsLoaded))
		ereport(ERROR,
				(errcode(ERRCODE_INDEX_CORRUPTED),
				 errmsg("document-node centroid sidecar is not loaded"),
				 errhint("REINDEX with multivector_graph = document_nodes, multivector_proxy_encoder = centroid_mean, and multivector_centroids = kmeans to rebuild persisted centroid sidecar tuples.")));
	if (quantizedInvertedExperimental &&
		((meta->tqMultivectorDocMapFlags &
		  PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_QUANTIZED_POSTINGS) == 0 ||
		 !storage.multivectorQuantizedInvertedPostingsLoaded))
		ereport(ERROR,
				(errcode(ERRCODE_INDEX_CORRUPTED),
				 errmsg("quantized_inverted_experimental posting sidecar is not loaded"),
				 errhint("REINDEX with multivector_graph = document_nodes to rebuild experimental quantized inverted posting tuples.")));
	if (quantizedInvertedExperimental)
	{
		int			requestedSource =
			pgturbohybrid_multivector_quantized_inverted_codebook;

		if (storage.multivectorQuantizedInvertedCodebookSource !=
			requestedSource)
			ereport(ERROR,
					(errcode(ERRCODE_INDEX_CORRUPTED),
					 errmsg("quantized_inverted_experimental codebook metadata mismatch"),
					 errdetail("Index was built with codebook source \"%s\", but the active setting is \"%s\".",
							   PgturbohybridQuantizedInvertedCodebookSourceName(storage.multivectorQuantizedInvertedCodebookSource),
							   PgturbohybridQuantizedInvertedCodebookSourceName(requestedSource)),
					 errhint("REINDEX with matching turbohybrid.multivector_quantized_inverted_codebook settings, or reset the GUCs to match the existing experimental sidecar.")));
		if (requestedSource ==
			PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_EXTERNAL)
		{
			PgturbohybridQuantizedInvertedCodebook *codebook =
				PgturbohybridLoadQuantizedInvertedCodebook(query->dim);

			if (storage.multivectorQuantizedInvertedCodebookDim !=
				(uint32) query->dim ||
				storage.multivectorQuantizedInvertedCodebookSize !=
				codebook->codebookSize ||
				storage.multivectorQuantizedInvertedCodebookTopM !=
				codebook->topM ||
				strcmp(storage.multivectorQuantizedInvertedCodebookChecksum,
					   codebook->checksum) != 0)
				ereport(ERROR,
						(errcode(ERRCODE_INDEX_CORRUPTED),
						 errmsg("quantized_inverted_experimental codebook metadata mismatch"),
						 errdetail("Index metadata is dim=%u size=%u top_m=%u checksum=\"%s\", but the active external codebook is dim=%d size=%u top_m=%u checksum=\"%s\".",
								   storage.multivectorQuantizedInvertedCodebookDim,
								   storage.multivectorQuantizedInvertedCodebookSize,
								   storage.multivectorQuantizedInvertedCodebookTopM,
								   storage.multivectorQuantizedInvertedCodebookChecksum,
								   codebook->dim,
								   codebook->codebookSize,
								   codebook->topM,
								   codebook->checksum),
						 errhint("REINDEX with the current external codebook, or restore the codebook file used to build this experimental sidecar.")));
		}
		else if (storage.multivectorQuantizedInvertedCodebookDim !=
				 (uint32) query->dim ||
				 storage.multivectorQuantizedInvertedCodebookSize !=
				 (uint32) query->dim * 2U ||
				 storage.multivectorQuantizedInvertedCodebookTopM != 1)
			ereport(ERROR,
					(errcode(ERRCODE_INDEX_CORRUPTED),
					 errmsg("quantized_inverted_experimental deterministic codebook metadata mismatch"),
					 errhint("REINDEX with multivector_graph = document_nodes to rebuild experimental quantized inverted posting tuples.")));
	}
	PgturbohybridMultiVectorDocTokenTotals(&storage, &originalTokens,
										   &pooledTokens);
	if (proxyGraph)
	{
		if (collectPhaseStats)
			INSTR_TIME_SET_CURRENT(subphaseStart);
		proxyQuery =
			PgturbohybridMultiVectorBuildQueryProxyVector(query,
														  proxyEncoder,
														  resultCtx);
		if (collectPhaseStats &&
			proxyEncoder ==
			PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_LEARNED_PROJECTION_V1)
			learnedProjectionQueryEncodeUs =
				PgturbohybridGraphElapsedUs(subphaseStart);
		PgturbohybridGraphPrepareTqQueryWithBits(scan->indexRelation,
												 &so->support,
												 PointerGetDatum(proxyQuery),
												 &so->tq,
												 meta->tqBits != 0 ?
												 meta->tqBits :
												 PGTURBOHYBRID_DEFAULT_BITS);
		if (proxyDocumentCompactRescore)
		{
			INSTR_TIME_SET_CURRENT(phaseStart);
			compact =
				PgturbohybridMultiVectorBuildDocCompactStorage(meta, &storage,
															   docStorageKind);
			compactBytes = compact != NULL ? compact->bytes : 0;
			PgturbohybridGraphAddElapsedUs(&so->graphPrepareUs, phaseStart);
		}
	}
	else if (compactTraversal)
	{
		INSTR_TIME_SET_CURRENT(phaseStart);
		if (docStorageCacheMode ==
			PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_PAGED)
			compact =
				PgturbohybridMultiVectorInitDocCompactStorage(meta,
															  docStorageKind);
		else
			compact =
				PgturbohybridMultiVectorBuildDocCompactStorage(meta, &storage,
															   docStorageKind);
		compactBytes = compact != NULL ? compact->bytes : 0;
		PgturbohybridGraphAddElapsedUs(&so->graphPrepareUs, phaseStart);
	}

	if (collectPhaseStats)
		INSTR_TIME_SET_CURRENT(candidateSourceStart);
	if (exhaustiveSidecarScan)
	{
		INSTR_TIME_SET_CURRENT(phaseStart);
		for (uint32 docId = 0; docId < meta->tqMultivectorDocCount; docId++)
		{
			TqMultiVectorDocMapEntry *docEntry;
			PgturbohybridMultiVector *doc;
			TqDenseCandidate candidate;
			double		maxsim;

			CHECK_FOR_INTERRUPTS();
			docEntry = &storage.multivectorDocMap[docId];
			doc = PgturbohybridGraphLoadMultiVectorDocVector(scan->indexRelation,
															 meta,
															 &storage,
															 docId,
															 storage.ctx,
															 &sidecarStats);

			PgturbohybridCheckSameMultiVectorDims(query, doc);
			maxsimPairs += (uint64) query->count * (uint64) doc->count;
			maxsim =
				PgturbohybridMultiVectorIndexMaxSim(scan->indexRelation,
													query, doc,
													queryWeights,
													queryMask);

			memset(&candidate, 0, sizeof(candidate));
			candidate.nodeId = docEntry->firstNodeId;
			candidate.docId = docId;
			candidate.heaptid = docEntry->heapTid;
			candidate.distance = -maxsim;
			candidate.similarity =
				queryWeightSum > 0.0 ? maxsim / queryWeightSum : 0.0;
			candidate.rank = 0;
			candidate.hasDocId = true;
			candidate.exactScored = false;
			PgturbohybridMultiVectorCandidateHeapOffer(candidates, &docCount,
													  candidateLimit,
													  &candidate);
			if (storage.multivectorDocVectorsPaged)
				pfree(doc);
			docsScored++;
		}
		PgturbohybridGraphAddElapsedUs(&so->graphTraverseUs, phaseStart);
		if (collectPhaseStats)
			PgturbohybridGraphAddElapsedUint64(&docGraphTraversalUs, phaseStart);
		docGraphWarning = "document_node_f32_sidecar_exact_scan";
	}
	else
	{
		INSTR_TIME_SET_CURRENT(phaseStart);
		if (proxyGraph)
		{
			PgturbohybridGraphResult *hits;
			int			hitCount;
			int64		batchUsBefore = so->graphBatchUs;
			int64		baseUsBefore = so->graphBaseUs;

			hits = palloc0(sizeof(PgturbohybridGraphResult) * candidateLimit);
			hitCount = PgturbohybridGraphRunTraversalPass(scan, so, meta,
														  &storage,
														  hits,
														  candidateLimit,
														  searchEf,
														  PointerGetDatum(proxyQuery),
														  -1, 0, false,
														  false, 1.0,
														  PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_NONE);
			proxyGraphHitCount = hitCount;
			if (collectPhaseStats)
				proxyScoringUs +=
					(uint64) Max((int64) 0,
								 (so->graphBatchUs - batchUsBefore) +
								 (so->graphBaseUs - baseUsBefore));
			for (int i = 0; i < hitCount; i++)
			{
				TqDocId		docId;
				TqMultiVectorDocMapEntry *docEntry;
				TqDenseCandidate candidate;

				CHECK_FOR_INTERRUPTS();
				if (hits[i].nodeId >= meta->tqNodeCount)
					elog(ERROR, "pgturbohybrid proxy-vector node id is out of range");
				docId = storage.multivectorNodeMap[hits[i].nodeId].docId;
				if (docId >= storage.multivectorDocCount)
					elog(ERROR, "pgturbohybrid proxy-vector doc id is out of range");
				docEntry = &storage.multivectorDocMap[docId];
				memset(&candidate, 0, sizeof(candidate));
				candidate.nodeId = hits[i].nodeId;
				candidate.docId = docId;
				candidate.heaptid = docEntry->heapTid;
				candidate.distance = hits[i].distance;
				candidate.similarity = -hits[i].distance;
				candidate.rank = 0;
				candidate.hasDocId = true;
				candidate.exactScored = false;
				PgturbohybridMultiVectorCandidateHeapOffer(candidates,
														  &docCount,
														  candidateLimit,
														  &candidate);
				docsScored++;
			}
			edgesVisited = so->graphVisitedNodes;
			pfree(hits);
		}
		else if (quantizedInvertedExperimental)
		{
			PgturbohybridGraphMultiVectorQuantizedPostingEntry *postings;
			uint32	   *listOffsets;
			double	   *docScores;
			double	   *docBest;
			uint32	   *docBestGeneration;
			uint16	   *docTokenMatches = NULL;
			bool	   *docMatched;
			uint32	   *matchedDocIds;
			uint32	   *touchedDocIds;
			float	   *queryCodewordScores = NULL;
			uint32	   *queryProbeCodewords = NULL;
			uint32	   *queryProbeCounts = NULL;
			double	   *queryScoreBoundPrefix = NULL;
			bool	   *precompactKeep = NULL;
			PgturbohybridMultiVectorPostingScoreCandidate *precompactPerTokenSelected = NULL;
			uint32	   *precompactPerTokenCounts = NULL;
			uint32		queryScoreBoundTokenCount = 0;
			uint32		queryProbeLimit = 0;
			uint32		codebookSize;
			uint32		matchedDocCount = 0;

			if (collectPhaseStats)
				INSTR_TIME_SET_CURRENT(subphaseStart);
			codebookSize = storage.multivectorQuantizedInvertedCodebookSize;
			postings = storage.multivectorQuantizedInvertedPostings;
			listOffsets = storage.multivectorQuantizedInvertedListOffsets;
			if (codebookSize == 0 ||
				storage.multivectorQuantizedInvertedCodebookDim !=
				(uint32) query->dim ||
				postings == NULL ||
				listOffsets == NULL ||
				(quantizedInvertedCompactScoring &&
				 (!storage.multivectorQuantizedInvertedDocCodesLoaded ||
				  storage.multivectorQuantizedInvertedDocCodeOffsets == NULL ||
				  storage.multivectorQuantizedInvertedDocCodes == NULL)))
				ereport(ERROR,
						(errcode(ERRCODE_INDEX_CORRUPTED),
						 errmsg("quantized_inverted_experimental posting sidecar is invalid"),
						 errdetail("codebook_size=%u codebook_dim=%u query_dim=%d postings_loaded=%s postings_ptr=%s list_offsets_ptr=%s doc_codes_loaded=%s doc_code_offsets_ptr=%s doc_codes_ptr=%s posting_count=%u",
								   codebookSize,
								   storage.multivectorQuantizedInvertedCodebookDim,
								   query->dim,
								   storage.multivectorQuantizedInvertedPostingsLoaded ? "true" : "false",
								   postings != NULL ? "present" : "null",
								   listOffsets != NULL ? "present" : "null",
								   storage.multivectorQuantizedInvertedDocCodesLoaded ? "true" : "false",
								   storage.multivectorQuantizedInvertedDocCodeOffsets != NULL ? "present" : "null",
								   storage.multivectorQuantizedInvertedDocCodes != NULL ? "present" : "null",
								   storage.multivectorQuantizedInvertedPostingCount),
						 errhint("REINDEX with multivector_graph = document_nodes to rebuild experimental quantized inverted posting tuples.")));

			docScores =
				palloc0(sizeof(double) *
						(Size) Max(meta->tqMultivectorDocCount, 1U));
			docBest =
				palloc0(sizeof(double) *
						(Size) Max(meta->tqMultivectorDocCount, 1U));
			docBestGeneration =
				palloc0(sizeof(uint32) *
						(Size) Max(meta->tqMultivectorDocCount, 1U));
			if (quantizedInvertedNeedsTokenMatches)
				docTokenMatches =
					palloc0(sizeof(uint16) *
							(Size) Max(meta->tqMultivectorDocCount, 1U));
			docMatched =
				palloc0(sizeof(bool) *
						(Size) Max(meta->tqMultivectorDocCount, 1U));
			matchedDocIds =
				palloc0(sizeof(uint32) *
						(Size) Max(meta->tqMultivectorDocCount, 1U));
			touchedDocIds =
				palloc0(sizeof(uint32) *
						(Size) Max(meta->tqMultivectorDocCount, 1U));
			if (quantizedInvertedPrecompactReservoir &&
				quantizedInvertedPrecompactPerTokenK > 0 &&
				query->count > 0)
			{
				Size		perTokenSlots =
					(Size) query->count *
					(Size) quantizedInvertedPrecompactPerTokenK;

				precompactPerTokenSelected =
					palloc0(PgturbohybridGraphArrayAllocSize(
						sizeof(PgturbohybridMultiVectorPostingScoreCandidate),
						perTokenSlots));
				precompactPerTokenCounts =
					palloc0(PgturbohybridGraphArrayAllocSize(sizeof(uint32),
															 (Size) query->count));
			}
			if (quantizedInvertedCompactScoring)
			{
				const char *kernelName;
				Size		scoreCount =
					(Size) query->count * (Size) codebookSize;
				PgturbohybridQuantizedInvertedCodebook *externalCodebook = NULL;
				TqDotProductF32BlockFunc blockDotProduct = NULL;
				instr_time	queryCodewordStart;
				bool		useBlockedCodewordKernel =
					pgturbohybrid_multivector_quantized_inverted_query_codeword_kernel !=
					PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_QUERY_CODEWORD_KERNEL_SCALAR;

				quantizedInvertedCompactScorer =
					TqResolveCompactCodeScoreKernel("auto");
				quantizedInvertedCompactBatchScorer =
					TqResolveCompactCodeScoreBatchKernel("auto");
				kernelName =
					TqCompactCodeScoreBatchKernelName(quantizedInvertedCompactBatchScorer);
				quantizedInvertedCompactKernel =
					strncmp(kernelName, "compact_", 8) == 0 ?
					kernelName + 8 : kernelName;
				quantizedInvertedCompactBatchCodes =
					palloc(sizeof(int16) *
						   PGTURBOHYBRID_QUANTIZED_INVERTED_COMPACT_BATCH);
				quantizedInvertedCompactBatchScores =
					palloc(sizeof(int64) *
						   PGTURBOHYBRID_QUANTIZED_INVERTED_COMPACT_BATCH);
				if (storage.multivectorQuantizedInvertedCodebookSource ==
					PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_EXTERNAL)
				{
					externalCodebook =
						PgturbohybridLoadQuantizedInvertedCodebook(query->dim);
					if (useBlockedCodewordKernel)
						blockDotProduct = TqResolveDotProductF32BlockKernel();
				}
				quantizedInvertedQueryCodewordKernel =
					externalCodebook != NULL && useBlockedCodewordKernel ?
					"blocked" : "scalar";
				queryCodewordScores =
					palloc0(sizeof(float) * Max(scoreCount, (Size) 1));
				quantizedInvertedQueryCodewordFullMatrixMaterialized = true;
				queryProbeLimit =
					(uint32) Max(pgturbohybrid_multivector_quantized_inverted_probe_codewords_per_token,
								 1);
				queryProbeLimit = Min(queryProbeLimit, codebookSize);
				queryProbeCounts =
					palloc0(sizeof(uint32) * (Size) query->count);
				queryProbeCodewords =
					palloc0(sizeof(uint32) * (Size) query->count *
							(Size) Max(queryProbeLimit, 1U));
				if (collectPhaseStats)
					INSTR_TIME_SET_CURRENT(queryCodewordStart);
				for (int32 qi = 0; qi < query->count; qi++)
				{
					float	   *tokenScores =
						queryCodewordScores + (Size) qi * (Size) codebookSize;
					PgturbohybridMultiVectorPostingScoreCandidate *selected =
						palloc0(sizeof(PgturbohybridMultiVectorPostingScoreCandidate) *
								(Size) Max(queryProbeLimit, 1U));
					uint32		selectedCount = 0;

					if (queryMask != NULL && queryMask[qi])
					{
						quantizedInvertedQueryCodewordSkippedQueryTokens++;
						pfree(selected);
						continue;
					}
					quantizedInvertedQueryCodewordActiveQueryTokens++;

					if (externalCodebook != NULL && useBlockedCodewordKernel)
					{
						const float *values =
							PgturbohybridMultiVectorValues(query, qi);
						double		dotBlock[PGTURBOHYBRID_MULTIVECTOR_COMPACT_DOT_BLOCK];

						for (uint32 codeword = 0; codeword < codebookSize;
							 codeword += PGTURBOHYBRID_MULTIVECTOR_COMPACT_DOT_BLOCK)
						{
							int32		blockCount =
								(int32) Min(codebookSize - codeword,
											(uint32) PGTURBOHYBRID_MULTIVECTOR_COMPACT_DOT_BLOCK);
							const float *centroids =
								&externalCodebook->values[(Size) codeword *
														  (Size) query->dim];

							blockDotProduct(centroids, values, query->dim,
											blockCount,
											dotBlock);
							quantizedInvertedQueryCodewordBlocks++;
							quantizedInvertedQueryCodewordScoresComputed +=
								(uint64) blockCount;
							if (collectPhaseStats)
								INSTR_TIME_SET_CURRENT(subphaseStart);
							for (int32 blockIndex = 0; blockIndex < blockCount;
								 blockIndex++)
							{
								PgturbohybridMultiVectorPostingScoreCandidate scored;
								uint32		scoredCodeword =
									codeword + (uint32) blockIndex;

								tokenScores[codeword + (uint32) blockIndex] =
									(float) dotBlock[blockIndex];
								scored.docId = scoredCodeword;
								scored.ordinal = 0;
								scored.score = dotBlock[blockIndex];
								PgturbohybridMultiVectorPostingScoreCandidateOffer(selected,
																				   &selectedCount,
																				   queryProbeLimit,
																				   &scored);
							}
							if (collectPhaseStats)
								PgturbohybridGraphAddElapsedUint64(
									&quantizedInvertedQueryCodewordTopkUs,
									subphaseStart);
						}
					}
					else
					{
						for (uint32 codeword = 0; codeword < codebookSize; codeword++)
						{
							double		score =
								PgturbohybridMultiVectorQuantizedInvertedCodewordScore(query,
																					   qi,
																					   codeword);
							PgturbohybridMultiVectorPostingScoreCandidate scored;

							tokenScores[codeword] =
								(float) score;
							scored.docId = codeword;
							scored.ordinal = 0;
							scored.score = score;
							quantizedInvertedQueryCodewordScoresComputed++;
							if (collectPhaseStats)
								INSTR_TIME_SET_CURRENT(subphaseStart);
							PgturbohybridMultiVectorPostingScoreCandidateOffer(selected,
																			   &selectedCount,
																			   queryProbeLimit,
																			   &scored);
							if (collectPhaseStats)
								PgturbohybridGraphAddElapsedUint64(
									&quantizedInvertedQueryCodewordTopkUs,
									subphaseStart);
						}
					}
					queryProbeCounts[qi] = selectedCount;
					for (uint32 selectedIndex = 0; selectedIndex < selectedCount;
						 selectedIndex++)
						queryProbeCodewords[(Size) qi * (Size) queryProbeLimit +
											selectedIndex] =
							selected[selectedIndex].docId;
					pfree(selected);
				}
				if (collectPhaseStats)
					PgturbohybridGraphAddElapsedUint64(
						&quantizedInvertedQueryCodewordScoreUs,
						queryCodewordStart);
				if (quantizedInvertedScoreBoundPruning)
				{
					queryScoreBoundPrefix =
						PgturbohybridMultiVectorBuildCodewordScoreBoundPrefix(query,
																			  queryWeights,
																			  queryMask,
																			  queryCodewordScores,
																			  codebookSize,
																			  &queryScoreBoundTokenCount);
					if (queryScoreBoundPrefix == NULL)
						quantizedInvertedScoreBoundUnsafeFallbacks++;
				}
			}
			for (int32 qi = 0; qi < query->count; qi++)
			{
				uint32	   *probeCodewords;
				uint32		probeLimit =
					(uint32) Max(pgturbohybrid_multivector_quantized_inverted_probe_codewords_per_token,
								 1);
				uint32		probeCount;
				uint32		touchedCount = 0;
				uint32		generation = (uint32) qi + 1U;
				float	   *tokenCodewordScores = NULL;
				double		weight =
					queryWeights != NULL ? (double) queryWeights[qi] : 1.0;

				if (queryMask != NULL && queryMask[qi])
					continue;
				quantizedInvertedActiveQueryTokens++;
				probeCodewords =
					palloc0(sizeof(uint32) * (Size) probeLimit);
				if (quantizedInvertedCompactScoring &&
					queryCodewordScores != NULL)
				{
					PgturbohybridMultiVectorPostingScoreCandidate *selected;
					uint32		selectedCount = 0;
					instr_time	assignmentStart;

					tokenCodewordScores =
						queryCodewordScores + (Size) qi * (Size) codebookSize;
					if (queryProbeCodewords != NULL &&
						queryProbeCounts != NULL &&
						queryProbeLimit > 0)
					{
						uint32		cachedCount = queryProbeCounts[qi];
						uint32	   *cachedCodewords =
							queryProbeCodewords + (Size) qi * (Size) queryProbeLimit;

						probeCount = Min(cachedCount, probeLimit);
						for (uint32 selectedIndex = 0; selectedIndex < probeCount;
							 selectedIndex++)
							probeCodewords[selectedIndex] =
								cachedCodewords[selectedIndex];
					}
					else
					{
						if (collectPhaseStats)
							INSTR_TIME_SET_CURRENT(assignmentStart);
						probeLimit = Min(probeLimit, codebookSize);
						selected =
							palloc0(sizeof(PgturbohybridMultiVectorPostingScoreCandidate) *
									(Size) probeLimit);
						for (uint32 codeword = 0; codeword < codebookSize; codeword++)
						{
							PgturbohybridMultiVectorPostingScoreCandidate scored;

							scored.docId = codeword;
							scored.ordinal = 0;
							scored.score = tokenCodewordScores[codeword];
							PgturbohybridMultiVectorPostingScoreCandidateOffer(selected,
																			   &selectedCount,
																			   probeLimit,
																			   &scored);
						}
						for (uint32 i = 0; i < selectedCount; i++)
							probeCodewords[i] = selected[i].docId;
						probeCount = selectedCount;
						pfree(selected);
						if (collectPhaseStats)
							PgturbohybridGraphAddElapsedUint64(
								&quantizedInvertedAssignmentUs,
								assignmentStart);
					}
				}
				else
				{
					probeCount =
						PgturbohybridMultiVectorQuantizedInvertedConfigurableCodewords(
							query, qi, probeLimit, probeCodewords,
							&quantizedInvertedAssignmentUs);
				}
				for (uint32 probeIndex = 0; probeIndex < probeCount;
					 probeIndex++)
				{
					uint32		queryCodeword = probeCodewords[probeIndex];
					int16		queryCompactCode = 0;
					uint32		start;
					uint32		end;

					if (queryCodeword >= codebookSize)
						ereport(ERROR,
								(errcode(ERRCODE_INDEX_CORRUPTED),
								 errmsg("quantized_inverted_experimental posting sidecar references an out-of-range codeword"),
								 errhint("REINDEX with multivector_graph = document_nodes to rebuild experimental quantized inverted posting tuples.")));
					if (quantizedInvertedCompactScoring)
						queryCompactCode =
							PgturbohybridMultiVectorQuantizedInvertedCompactScorePayload(
								tokenCodewordScores != NULL ?
								tokenCodewordScores[queryCodeword] :
								PgturbohybridMultiVectorQuantizedInvertedCodewordScore(query,
																					   qi,
																					   queryCodeword));
					start = listOffsets[queryCodeword];
					end = listOffsets[queryCodeword + 1];
					quantizedInvertedListsVisited++;
						if (quantizedInvertedPostingLimitPerToken > 0 &&
							end > start &&
							end - start > quantizedInvertedPostingLimitPerToken &&
							pgturbohybrid_multivector_quantized_inverted_posting_selection ==
							PGTURBOHYBRID_MULTIVECTOR_POSTING_SELECTION_SCORE_TOPK)
						{
							uint32		listLen = end - start;
							uint32		selectedCount =
								Min(listLen, quantizedInvertedPostingLimitPerToken);
							PgturbohybridMultiVectorPostingScoreCandidate *selected = NULL;
							bool		queryAwareSignedSelection =
								quantizedInvertedCompactScoring &&
								storage.multivectorQuantizedInvertedCodebookSource ==
								PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_EXTERNAL;

							if (queryAwareSignedSelection)
							{
								uint32		keptCount = 0;

								selected =
									palloc0(sizeof(PgturbohybridMultiVectorPostingScoreCandidate) *
											(Size) selectedCount);
								for (uint32 postingIndex = 0;
									 postingIndex < listLen;)
								{
									uint32		batchCount =
										Min(listLen - postingIndex,
											(uint32) PGTURBOHYBRID_QUANTIZED_INVERTED_COMPACT_BATCH);
									instr_time	compactStart;

									CHECK_FOR_INTERRUPTS();
									for (uint32 batchIndex = 0;
										 batchIndex < batchCount;
										 batchIndex++)
									{
										PgturbohybridGraphMultiVectorQuantizedPostingEntry *posting =
											&postings[start + postingIndex + batchIndex];

										quantizedInvertedCompactBatchCodes[batchIndex] =
											PgturbohybridMultiVectorQuantizedInvertedSignedCompactPayload(
												posting->scorePayload);
									}
									if (collectPhaseStats)
										INSTR_TIME_SET_CURRENT(compactStart);
									quantizedInvertedCompactBatchScorer(queryCompactCode,
																		quantizedInvertedCompactBatchCodes,
																		(int32) batchCount,
																		quantizedInvertedCompactBatchScores);
									if (collectPhaseStats)
										PgturbohybridGraphAddElapsedUint64(
											&quantizedInvertedCompactScoreUs,
											compactStart);
									for (uint32 batchIndex = 0;
										 batchIndex < batchCount;
										 batchIndex++)
									{
										PgturbohybridGraphMultiVectorQuantizedPostingEntry *posting =
											&postings[start + postingIndex + batchIndex];
										PgturbohybridMultiVectorPostingScoreCandidate scored;

										scored.docId = posting->docId;
										scored.ordinal = start + postingIndex + batchIndex;
										scored.score =
											PgturbohybridMultiVectorQuantizedInvertedCompactRawScore(
												quantizedInvertedCompactBatchScores[batchIndex]);
										PgturbohybridMultiVectorPostingScoreCandidateOffer(selected,
																						   &keptCount,
																						   selectedCount,
																						   &scored);
									}
									postingIndex += batchCount;
								}
								selectedCount = keptCount;
							}

							for (uint32 selectedIndex = 0;
								 selectedIndex < selectedCount;
								 selectedIndex++)
							{
							PgturbohybridGraphMultiVectorQuantizedPostingEntry *posting;
							PgturbohybridMultiVector *doc;
							uint32		postingOffset;
							uint32		docId;
								double		dot;

								CHECK_FOR_INTERRUPTS();
								postingOffset = selected != NULL ?
									selected[selectedIndex].ordinal : start + selectedIndex;
								posting = &postings[postingOffset];
								docId = posting->docId;
								if (docId >= storage.multivectorDocCount)
									ereport(ERROR,
										(errcode(ERRCODE_INDEX_CORRUPTED),
										 errmsg("quantized_inverted_experimental posting sidecar references an out-of-range document"),
										 errhint("REINDEX with multivector_graph = document_nodes to rebuild experimental quantized inverted posting tuples.")));
								if (quantizedInvertedCompactScoring)
								{
									int16		qCode = queryCompactCode;
									int16		dCode =
										storage.multivectorQuantizedInvertedCodebookSource ==
									PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_EXTERNAL ?
									PgturbohybridMultiVectorQuantizedInvertedSignedCompactPayload(
										posting->scorePayload) :
									PgturbohybridMultiVectorQuantizedInvertedCompactPayload(
										posting->scorePayload);
									int64		raw;
									instr_time	compactStart;

									if (selected != NULL)
										dot = selected[selectedIndex].score;
									else
									{
										if (collectPhaseStats)
											INSTR_TIME_SET_CURRENT(compactStart);
										raw =
											quantizedInvertedCompactScorer(&qCode, &dCode, 1);
										if (collectPhaseStats)
											PgturbohybridGraphAddElapsedUint64(
												&quantizedInvertedCompactScoreUs,
												compactStart);
										dot = (double) raw /
											((double) SHRT_MAX * (double) SHRT_MAX);
									}
									quantizedInvertedCompactPayloadBytes +=
										sizeof(posting->scorePayload);
								}
							else
							{
								if (storage.multivectorDocVectorsPaged)
									doc = PgturbohybridGraphLoadMultiVectorDocVector(scan->indexRelation,
																					meta,
																					&storage,
																					docId,
																					resultCtx,
																					&sidecarStats);
								else
								{
									doc = storage.multivectorDocVectors[docId];
								}
								if (doc == NULL ||
									posting->tokenOrdinal >= (uint32) doc->count)
									ereport(ERROR,
											(errcode(ERRCODE_INDEX_CORRUPTED),
											 errmsg("quantized_inverted_experimental posting sidecar is invalid"),
											 errdetail("posting doc_id=%u token_ordinal=%u document_token_count=%d",
													   docId, posting->tokenOrdinal,
													   doc != NULL ? doc->count : -1),
											 errhint("REINDEX with multivector_graph = document_nodes to rebuild experimental quantized inverted posting tuples.")));
								PgturbohybridCheckSameMultiVectorDims(query,
																	  doc);
								dot = PgturbohybridMultiVectorTokenDot(query, qi,
																	   doc,
																	   posting->tokenOrdinal);
							}
							if (docBestGeneration[docId] != generation)
							{
								docBestGeneration[docId] = generation;
								docBest[docId] = -DBL_MAX;
								touchedDocIds[touchedCount++] = docId;
							}
							if (dot > docBest[docId])
								docBest[docId] = dot;
							quantizedInvertedPostingsTouched++;
							}
							quantizedInvertedPostingsSelected += selectedCount;
							quantizedInvertedPostingsSkipped +=
								(uint64) (listLen - selectedCount);
							quantizedInvertedPostingCapStrategy =
								queryAwareSignedSelection ?
								"score_topk_query_aware_signed" :
								"score_topk_payload_sorted";
							if (selected != NULL)
								pfree(selected);
						}
					else
					{
						uint32		listLen = end - start;
						uint32		effectiveEnd = end;

						if (quantizedInvertedPostingLimitPerToken > 0 &&
							listLen > quantizedInvertedPostingLimitPerToken)
						{
							effectiveEnd = start + quantizedInvertedPostingLimitPerToken;
							quantizedInvertedPostingsSkipped +=
								(uint64) (listLen - quantizedInvertedPostingLimitPerToken);
							quantizedInvertedPostingCapStrategy = "uniform_stride";
						}
						else if (listLen > 0)
							quantizedInvertedPostingCapStrategy = "uncapped_full_list";
						if (quantizedInvertedCompactScoring &&
							!(quantizedInvertedPostingLimitPerToken > 0 &&
							  listLen > quantizedInvertedPostingLimitPerToken))
						{
							for (uint32 postingIndex = 0;
								 postingIndex < effectiveEnd - start;)
							{
								uint32		batchCount =
									Min((effectiveEnd - start) - postingIndex,
										(uint32) PGTURBOHYBRID_QUANTIZED_INVERTED_COMPACT_BATCH);
								instr_time	compactStart;

								CHECK_FOR_INTERRUPTS();
								for (uint32 batchIndex = 0;
									 batchIndex < batchCount;
									 batchIndex++)
								{
									PgturbohybridGraphMultiVectorQuantizedPostingEntry *posting =
										&postings[start + postingIndex + batchIndex];
									uint32		docId = posting->docId;

									if (docId >= storage.multivectorDocCount)
										ereport(ERROR,
												(errcode(ERRCODE_INDEX_CORRUPTED),
												 errmsg("quantized_inverted_experimental posting sidecar references an out-of-range document"),
												 errhint("REINDEX with multivector_graph = document_nodes to rebuild experimental quantized inverted posting tuples.")));
									if (docBestGeneration[docId] != generation)
									{
										docBestGeneration[docId] = generation;
										docBest[docId] = -DBL_MAX;
										touchedDocIds[touchedCount++] = docId;
									}
									quantizedInvertedCompactBatchCodes[batchIndex] =
										storage.multivectorQuantizedInvertedCodebookSource ==
										PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_EXTERNAL ?
										PgturbohybridMultiVectorQuantizedInvertedSignedCompactPayload(
											posting->scorePayload) :
										PgturbohybridMultiVectorQuantizedInvertedCompactPayload(
											posting->scorePayload);
								}
								if (collectPhaseStats)
									INSTR_TIME_SET_CURRENT(compactStart);
								quantizedInvertedCompactBatchScorer(queryCompactCode,
																	quantizedInvertedCompactBatchCodes,
																	(int32) batchCount,
																	quantizedInvertedCompactBatchScores);
								if (collectPhaseStats)
									PgturbohybridGraphAddElapsedUint64(
										&quantizedInvertedCompactScoreUs,
										compactStart);
								for (uint32 batchIndex = 0;
									 batchIndex < batchCount;
									 batchIndex++)
								{
									PgturbohybridGraphMultiVectorQuantizedPostingEntry *posting =
										&postings[start + postingIndex + batchIndex];
									uint32		docId = posting->docId;
									double		dot =
										PgturbohybridMultiVectorQuantizedInvertedCompactRawScore(
											quantizedInvertedCompactBatchScores[batchIndex]);

									if (dot > docBest[docId])
										docBest[docId] = dot;
									quantizedInvertedPostingsTouched++;
									quantizedInvertedPostingsSelected++;
									quantizedInvertedCompactPayloadBytes +=
										sizeof(posting->scorePayload);
								}
								postingIndex += batchCount;
							}
						}
						else
						{
							for (uint32 postingIndex = 0;
								 postingIndex < effectiveEnd - start;
								 postingIndex++)
							{
								PgturbohybridGraphMultiVectorQuantizedPostingEntry *posting;
								PgturbohybridMultiVector *doc;
								uint32		postingOffset;
								uint32		docId;
								double		dot;

								CHECK_FOR_INTERRUPTS();
								if (quantizedInvertedPostingLimitPerToken > 0 &&
									listLen > quantizedInvertedPostingLimitPerToken)
								{
									uint64		limit =
										(uint64) quantizedInvertedPostingLimitPerToken;
									uint64		offset =
										((uint64) postingIndex * 2U + 1U) *
										(uint64) listLen / (limit * 2U);

									postingOffset = start + (uint32) offset;
								}
								else
									postingOffset = start + postingIndex;
								posting = &postings[postingOffset];
								docId = posting->docId;
								if (docId >= storage.multivectorDocCount)
									ereport(ERROR,
											(errcode(ERRCODE_INDEX_CORRUPTED),
											 errmsg("quantized_inverted_experimental posting sidecar references an out-of-range document"),
											 errhint("REINDEX with multivector_graph = document_nodes to rebuild experimental quantized inverted posting tuples.")));
								if (docBestGeneration[docId] != generation)
								{
									docBestGeneration[docId] = generation;
									docBest[docId] = -DBL_MAX;
									touchedDocIds[touchedCount++] = docId;
								}
								if (quantizedInvertedCompactScoring)
								{
									int16		qCode = queryCompactCode;
									int16		dCode =
										storage.multivectorQuantizedInvertedCodebookSource ==
										PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_EXTERNAL ?
										PgturbohybridMultiVectorQuantizedInvertedSignedCompactPayload(
											posting->scorePayload) :
										PgturbohybridMultiVectorQuantizedInvertedCompactPayload(
											posting->scorePayload);
									int64		raw;
									instr_time	compactStart;

									if (collectPhaseStats)
										INSTR_TIME_SET_CURRENT(compactStart);
									raw =
										quantizedInvertedCompactScorer(&qCode, &dCode, 1);
									if (collectPhaseStats)
										PgturbohybridGraphAddElapsedUint64(
											&quantizedInvertedCompactScoreUs,
											compactStart);
									dot =
										PgturbohybridMultiVectorQuantizedInvertedCompactRawScore(raw);
									quantizedInvertedCompactPayloadBytes +=
										sizeof(posting->scorePayload);
								}
								else
								{
									if (storage.multivectorDocVectorsPaged)
										doc = PgturbohybridGraphLoadMultiVectorDocVector(scan->indexRelation,
																						meta,
																						&storage,
																						docId,
																						resultCtx,
																						&sidecarStats);
									else
									{
										doc = storage.multivectorDocVectors[docId];
									}
									if (doc == NULL ||
										posting->tokenOrdinal >= (uint32) doc->count)
										ereport(ERROR,
												(errcode(ERRCODE_INDEX_CORRUPTED),
												 errmsg("quantized_inverted_experimental posting sidecar is invalid"),
												 errdetail("posting doc_id=%u token_ordinal=%u document_token_count=%d",
														   docId, posting->tokenOrdinal,
														   doc != NULL ? doc->count : -1),
												 errhint("REINDEX with multivector_graph = document_nodes to rebuild experimental quantized inverted posting tuples.")));
									dot = PgturbohybridMultiVectorTokenDot(query, qi,
																		   doc,
																		   posting->tokenOrdinal);
								}
								if (dot > docBest[docId])
									docBest[docId] = dot;
								quantizedInvertedPostingsTouched++;
								quantizedInvertedPostingsSelected++;
							}
						}
					}
				}
				pfree(probeCodewords);
				for (uint32 touchedIndex = 0; touchedIndex < touchedCount;
					 touchedIndex++)
				{
					uint32		docId = touchedDocIds[touchedIndex];

					if (docBest[docId] <= -DBL_MAX)
						continue;
					if (!docMatched[docId])
					{
						docMatched[docId] = true;
						matchedDocIds[matchedDocCount++] = docId;
					}
					docScores[docId] += weight * docBest[docId];
					if (precompactPerTokenSelected != NULL &&
						precompactPerTokenCounts != NULL &&
						quantizedInvertedPrecompactPerTokenK > 0)
					{
						PgturbohybridMultiVectorPostingScoreCandidate scored;
						PgturbohybridMultiVectorPostingScoreCandidate *selected =
							precompactPerTokenSelected +
							(Size) qi *
							(Size) quantizedInvertedPrecompactPerTokenK;

						scored.docId = docId;
						scored.ordinal = (uint32) qi;
						scored.score = weight * docBest[docId];
						PgturbohybridMultiVectorPostingScoreCandidateOffer(
							selected,
							&precompactPerTokenCounts[qi],
							quantizedInvertedPrecompactPerTokenK,
							&scored);
					}
					if (docTokenMatches != NULL &&
						docTokenMatches[docId] < UINT16_MAX)
						docTokenMatches[docId]++;
				}
			}

			quantizedInvertedDocsTouchedBeforePrecompact = matchedDocCount;
			if (quantizedInvertedPrecompactEnabled && matchedDocCount > 0)
			{
				PgturbohybridMultiVectorPrecompactDoc *rankedDocs;
				instr_time	precompactStart;

				if (collectPhaseStats)
					INSTR_TIME_SET_CURRENT(precompactStart);
				precompactKeep =
					palloc0(PgturbohybridGraphArrayAllocSize(sizeof(bool),
															 (Size) Max(meta->tqMultivectorDocCount, 1U)));
				rankedDocs =
					palloc0(PgturbohybridGraphArrayAllocSize(sizeof(PgturbohybridMultiVectorPrecompactDoc),
															 (Size) matchedDocCount));
				for (uint32 matchedIndex = 0; matchedIndex < matchedDocCount;
					 matchedIndex++)
				{
					uint32		docId = matchedDocIds[matchedIndex];

					rankedDocs[matchedIndex].docId = docId;
					rankedDocs[matchedIndex].score = docScores[docId];
					rankedDocs[matchedIndex].coverage =
						docTokenMatches != NULL ? docTokenMatches[docId] : 0;
				}

				if (quantizedInvertedPrecompactMode ==
					PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_PRECOMPACT_CENTROID_MAXSIM_TOPK)
				{
					if (quantizedInvertedPrecompactScoreK == 0 ||
						quantizedInvertedPrecompactScoreK >= matchedDocCount)
					{
						for (uint32 matchedIndex = 0;
							 matchedIndex < matchedDocCount;
							 matchedIndex++)
							PgturbohybridMultiVectorPrecompactMarkDoc(
								precompactKeep,
								rankedDocs[matchedIndex].docId,
								&quantizedInvertedPrecompactUnionDocs,
								&quantizedInvertedPrecompactDuplicates);
					}
					else
					{
						uint32		limit =
							Min(quantizedInvertedPrecompactScoreK,
								matchedDocCount);

						qsort(rankedDocs, matchedDocCount,
							  sizeof(PgturbohybridMultiVectorPrecompactDoc),
							  PgturbohybridMultiVectorPrecompactDocScoreCompare);
						for (uint32 i = 0; i < limit; i++)
							PgturbohybridMultiVectorPrecompactMarkDoc(
								precompactKeep,
								rankedDocs[i].docId,
								&quantizedInvertedPrecompactUnionDocs,
								&quantizedInvertedPrecompactDuplicates);
						quantizedInvertedPrecompactScoreDocs = limit;
					}
					if (quantizedInvertedPrecompactScoreDocs == 0)
						quantizedInvertedPrecompactScoreDocs =
							quantizedInvertedPrecompactUnionDocs;
				}
				else if (quantizedInvertedPrecompactReservoir)
				{
					if (quantizedInvertedPrecompactScoreK > 0)
					{
						uint32		limit =
							Min(quantizedInvertedPrecompactScoreK,
								matchedDocCount);

						qsort(rankedDocs, matchedDocCount,
							  sizeof(PgturbohybridMultiVectorPrecompactDoc),
							  PgturbohybridMultiVectorPrecompactDocScoreCompare);
						for (uint32 i = 0; i < limit; i++)
							PgturbohybridMultiVectorPrecompactMarkDoc(
								precompactKeep,
								rankedDocs[i].docId,
								&quantizedInvertedPrecompactUnionDocs,
								&quantizedInvertedPrecompactDuplicates);
						quantizedInvertedPrecompactScoreDocs = limit;
					}
					if (quantizedInvertedPrecompactCoverageK > 0 &&
						docTokenMatches != NULL)
					{
						uint32		limit =
							Min(quantizedInvertedPrecompactCoverageK,
								matchedDocCount);

						qsort(rankedDocs, matchedDocCount,
							  sizeof(PgturbohybridMultiVectorPrecompactDoc),
							  PgturbohybridMultiVectorPrecompactDocCoverageCompare);
						for (uint32 i = 0; i < limit; i++)
							PgturbohybridMultiVectorPrecompactMarkDoc(
								precompactKeep,
								rankedDocs[i].docId,
								&quantizedInvertedPrecompactUnionDocs,
								&quantizedInvertedPrecompactDuplicates);
						quantizedInvertedPrecompactCoverageDocs = limit;
					}
					if (precompactPerTokenSelected != NULL &&
						precompactPerTokenCounts != NULL)
					{
						for (int32 qi = 0; qi < query->count; qi++)
						{
							PgturbohybridMultiVectorPostingScoreCandidate *selected =
								precompactPerTokenSelected +
								(Size) qi *
								(Size) quantizedInvertedPrecompactPerTokenK;
							uint32		selectedCount =
								precompactPerTokenCounts[qi];

							quantizedInvertedPrecompactPerTokenDocs +=
								selectedCount;
							for (uint32 i = 0; i < selectedCount; i++)
								PgturbohybridMultiVectorPrecompactMarkDoc(
									precompactKeep,
									selected[i].docId,
									&quantizedInvertedPrecompactUnionDocs,
									&quantizedInvertedPrecompactDuplicates);
						}
					}
					if (quantizedInvertedCompactMaxDocs > 0 &&
						quantizedInvertedPrecompactUnionDocs >
						quantizedInvertedCompactMaxDocs)
					{
						uint32		keptCount = 0;
						uint32		limit =
							Min(quantizedInvertedCompactMaxDocs,
								quantizedInvertedPrecompactUnionDocs);

						for (uint32 matchedIndex = 0;
							 matchedIndex < matchedDocCount;
							 matchedIndex++)
						{
							uint32		docId = matchedDocIds[matchedIndex];

							if (!precompactKeep[docId])
								continue;
							rankedDocs[keptCount].docId = docId;
							rankedDocs[keptCount].score = docScores[docId];
							rankedDocs[keptCount].coverage =
								docTokenMatches != NULL ? docTokenMatches[docId] : 0;
							keptCount++;
						}
						qsort(rankedDocs, keptCount,
							  sizeof(PgturbohybridMultiVectorPrecompactDoc),
							  PgturbohybridMultiVectorPrecompactDocScoreCompare);
						memset(precompactKeep, 0,
							   PgturbohybridGraphArrayAllocSize(sizeof(bool),
																(Size) Max(meta->tqMultivectorDocCount, 1U)));
						quantizedInvertedPrecompactUnionDocs = 0;
						for (uint32 i = 0; i < limit; i++)
							PgturbohybridMultiVectorPrecompactMarkDoc(
								precompactKeep,
								rankedDocs[i].docId,
								&quantizedInvertedPrecompactUnionDocs,
								&quantizedInvertedPrecompactDuplicates);
					}
				}
				if (quantizedInvertedPrecompactUnionDocs < matchedDocCount)
					quantizedInvertedPrecompactPrunedDocs =
						matchedDocCount - quantizedInvertedPrecompactUnionDocs;
				else
					quantizedInvertedPrecompactPrunedDocs = 0;
				if (collectPhaseStats)
					PgturbohybridGraphAddElapsedUint64(&quantizedInvertedPrecompactUs,
													   precompactStart);
				pfree(rankedDocs);
			}

			{
				uint32	   *compactDocIds = NULL;
				uint32		compactDocCount = 0;
				bool		compactDocIdOrder =
					quantizedInvertedCompactScoring &&
					queryCodewordScores != NULL &&
					pgturbohybrid_multivector_quantized_inverted_compact_doc_order ==
					PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_COMPACT_DOC_ORDER_DOCID;

				if (quantizedInvertedCompactScoring && queryCodewordScores != NULL)
				{
					quantizedInvertedCompactActiveQueryTokens =
						quantizedInvertedActiveQueryTokens > 0 ?
						quantizedInvertedActiveQueryTokens :
						quantizedInvertedQueryCodewordActiveQueryTokens;
					if (quantizedInvertedCompactActiveQueryTokens == 0)
					{
						for (int32 qi = 0; qi < query->count; qi++)
						{
							if (queryMask != NULL && queryMask[qi])
								continue;
							quantizedInvertedCompactActiveQueryTokens++;
						}
					}
				}
				if (compactDocIdOrder && matchedDocCount > 0)
				{
					compactDocIds =
						palloc(PgturbohybridGraphArrayAllocSize(sizeof(uint32),
																(Size) matchedDocCount));
					for (uint32 matchedIndex = 0;
						 matchedIndex < matchedDocCount;
						 matchedIndex++)
					{
						uint32		docId = matchedDocIds[matchedIndex];

						if (quantizedInvertedPrecompactEnabled &&
							precompactKeep != NULL &&
							!precompactKeep[docId])
							continue;
						compactDocIds[compactDocCount++] = docId;
					}
					if (compactDocCount > 1)
						qsort(compactDocIds, compactDocCount, sizeof(uint32),
							  PgturbohybridUInt32AscCompare);
					quantizedInvertedCompactDocOrder = "docid";
				}
				else
					compactDocCount = matchedDocCount;

				for (uint32 matchedIndex = 0; matchedIndex < compactDocCount;
					 matchedIndex++)
				{
					uint32		docId = compactDocIds != NULL ?
						compactDocIds[matchedIndex] : matchedDocIds[matchedIndex];
					TqMultiVectorDocMapEntry *docEntry;
					TqDenseCandidate candidate;
					double		candidateScore = docScores[docId];

					CHECK_FOR_INTERRUPTS();
					if (quantizedInvertedPrecompactEnabled &&
						precompactKeep != NULL &&
						!precompactKeep[docId])
						continue;
					docEntry = &storage.multivectorDocMap[docId];
					if (quantizedInvertedCompactScoring &&
						queryCodewordScores != NULL)
					{
						uint32		start =
							storage.multivectorQuantizedInvertedDocCodeOffsets[docId];
						uint32		end =
							storage.multivectorQuantizedInvertedDocCodeOffsets[docId + 1];
						uint64		pairs = 0;
						instr_time	compactStart;
						bool		prunedByScoreBound = false;

						if (end < start ||
							end > storage.multivectorQuantizedInvertedDocCodeCount)
							ereport(ERROR,
									(errcode(ERRCODE_INDEX_CORRUPTED),
									 errmsg("quantized_inverted_experimental doc-code map is invalid"),
									 errhint("REINDEX with multivector_graph = document_nodes to rebuild experimental quantized inverted posting tuples.")));
						if (quantizedInvertedScoreBoundPruning)
						{
							instr_time	pruneStart;

							quantizedInvertedScoreBoundDocsChecked++;
							if (collectPhaseStats)
								INSTR_TIME_SET_CURRENT(pruneStart);
							if (docCount >= candidateLimit &&
								candidateLimit > 0 &&
								queryScoreBoundPrefix != NULL &&
								docTokenMatches != NULL)
							{
								uint16		tokenMatches = docTokenMatches[docId];
								uint32		missingTokens =
									queryScoreBoundTokenCount > tokenMatches ?
									queryScoreBoundTokenCount - tokenMatches : 0;
								double		scoreBound =
									candidateScore + queryScoreBoundPrefix[missingTokens];
								double		worstScore = -candidates[0].distance;

								if (scoreBound <= worstScore)
									prunedByScoreBound = true;
							}
							else if (queryScoreBoundPrefix == NULL ||
									 docTokenMatches == NULL)
								quantizedInvertedScoreBoundUnsafeFallbacks++;
							if (collectPhaseStats)
								PgturbohybridGraphAddElapsedUint64(
									&quantizedInvertedScoreBoundPruneUs,
									pruneStart);
							if (prunedByScoreBound)
							{
								quantizedInvertedScoreBoundDocsPruned++;
								continue;
							}
						}
						if (collectPhaseStats)
							INSTR_TIME_SET_CURRENT(compactStart);
						candidateScore =
							PgturbohybridMultiVectorCentroidCodewordMaxSimScore(query,
																				queryWeights,
																				queryMask,
																				storage.multivectorQuantizedInvertedDocCodes + start,
																				end - start,
																				queryCodewordScores,
																				codebookSize,
																				&pairs);
						if (collectPhaseStats)
							PgturbohybridGraphAddElapsedUint64(&quantizedInvertedCompactScoreUs,
															   compactStart);
						quantizedInvertedCompactDocsScored++;
						quantizedInvertedCompactPairsEvaluated += pairs;
						if (query->count > (int32) quantizedInvertedCompactActiveQueryTokens)
							quantizedInvertedCompactPairsSkipped +=
								(uint64) (query->count -
										  (int32) quantizedInvertedCompactActiveQueryTokens) *
								(uint64) (end - start);
						quantizedInvertedCompactPayloadBytes +=
							(uint64) (end - start) * (uint64) sizeof(uint32);
					}
					if (docTokenMatches != NULL)
					{
						uint16		tokenMatches = docTokenMatches[docId];

						quantizedInvertedTokenMatchesTotal += tokenMatches;
						if (tokenMatches > quantizedInvertedTokenMatchesMax)
							quantizedInvertedTokenMatchesMax = tokenMatches;
						if (quantizedInvertedMinTokenMatches > 0 &&
							tokenMatches < quantizedInvertedMinTokenMatches)
						{
							quantizedInvertedTokenMatchFilteredDocs++;
							continue;
						}
						if (quantizedInvertedTokenCoverageLinear &&
							quantizedInvertedActiveQueryTokens > 0)
							candidateScore *=
								(double) tokenMatches /
								(double) quantizedInvertedActiveQueryTokens;
					}
					memset(&candidate, 0, sizeof(candidate));
					candidate.nodeId = docEntry->firstNodeId;
					candidate.docId = docId;
					candidate.heaptid = docEntry->heapTid;
					candidate.distance = -candidateScore;
					candidate.similarity =
						queryWeightSum > 0.0 ? candidateScore / queryWeightSum : 0.0;
					candidate.rank = 0;
					candidate.hasDocId = true;
					candidate.exactScored = false;
					PgturbohybridMultiVectorCandidateHeapOffer(candidates,
															  &docCount,
															  candidateLimit,
															  &candidate);
				}
				if (compactDocIds != NULL)
					pfree(compactDocIds);
			}
			quantizedInvertedCandidatesBeforeBound = matchedDocCount;
			quantizedInvertedCandidatesAfterBound = (uint32) docCount;
			if (quantizedInvertedCompactScoring)
			{
				quantizedInvertedDocsScored = quantizedInvertedCompactDocsScored;
				if (quantizedInvertedPrecompactEnabled &&
					matchedDocCount > quantizedInvertedCompactDocsScored)
					quantizedInvertedCompactDocsSkippedByPrecompact =
						matchedDocCount - (uint32) quantizedInvertedCompactDocsScored;
			}
			else
				quantizedInvertedDocsScored =
					matchedDocCount > quantizedInvertedScoreBoundDocsPruned ?
					matchedDocCount - quantizedInvertedScoreBoundDocsPruned : 0;
			if (collectPhaseStats)
				PgturbohybridGraphAddElapsedUint64(&quantizedInvertedPostingUs,
												   subphaseStart);
			if (queryCodewordScores != NULL)
				pfree(queryCodewordScores);
			if (queryProbeCodewords != NULL)
				pfree(queryProbeCodewords);
			if (queryProbeCounts != NULL)
				pfree(queryProbeCounts);
			if (precompactKeep != NULL)
				pfree(precompactKeep);
			if (precompactPerTokenSelected != NULL)
				pfree(precompactPerTokenSelected);
			if (precompactPerTokenCounts != NULL)
				pfree(precompactPerTokenCounts);
			if (queryScoreBoundPrefix != NULL)
				pfree(queryScoreBoundPrefix);
			if (quantizedInvertedCompactBatchCodes != NULL)
			{
				pfree(quantizedInvertedCompactBatchCodes);
				quantizedInvertedCompactBatchCodes = NULL;
			}
			if (quantizedInvertedCompactBatchScores != NULL)
			{
				pfree(quantizedInvertedCompactBatchScores);
				quantizedInvertedCompactBatchScores = NULL;
			}
		}
		else if (centroidLite)
		{
			PgturbohybridGraphMultiVectorCentroidPostingEntry *postings;
			uint32	   *listOffsets;
			double	   *docScores;
			double	   *docBest;
			uint32	   *docBestGeneration;
			bool	   *docMatched;
			uint32	   *matchedDocIds;
			uint32	   *touchedDocIds;
			uint8	   *centroidBitset = NULL;
			uint16	   *centroidBitsetTokenMatches = NULL;
			uint16	   *centroidScoreBoundTokenMatches = NULL;
			Size		centroidBitsetBytes = 0;
			float	   *queryCodewordScores = NULL;
			double	   *queryScoreBoundPrefix = NULL;
			uint32		queryScoreBoundTokenCount = 0;
			uint32		codebookSize;
			uint32		matchedDocCount = 0;
			uint64		centroidPostingsTouched = 0;
			uint32		denseDocCount;

			if (collectPhaseStats)
				INSTR_TIME_SET_CURRENT(subphaseStart);
			codebookSize = storage.multivectorCentroidPostingCodebookSize;
			postings = storage.multivectorCentroidPostings;
			listOffsets = storage.multivectorCentroidPostingListOffsets;
			if (codebookSize != (uint32) query->dim * 2U ||
				postings == NULL ||
				listOffsets == NULL)
				ereport(ERROR,
						(errcode(ERRCODE_INDEX_CORRUPTED),
						 errmsg("document-node centroid_lite posting sidecar is invalid"),
						 errhint("REINDEX with multivector_graph = document_nodes and multivector_centroids = kmeans to rebuild persisted centroid posting tuples.")));
			if (centroidLiteCodewordMaxsimScoring)
			{
				Size		scoreCount =
					(Size) query->count * (Size) codebookSize;

				queryCodewordScores =
					palloc(sizeof(float) * Max(scoreCount, (Size) 1));
				for (int32 qi = 0; qi < query->count; qi++)
				{
					float	   *tokenScores =
						queryCodewordScores + (Size) qi * (Size) codebookSize;

					for (uint32 codeword = 0; codeword < codebookSize;
						 codeword++)
						tokenScores[codeword] =
							(float) PgturbohybridMultiVectorDeterministicCodewordScore(query,
																					   qi,
																					   codeword);
				}
				if (centroidScoreBoundPruning)
				{
					queryScoreBoundPrefix =
						PgturbohybridMultiVectorBuildCodewordScoreBoundPrefix(query,
																			  queryWeights,
																			  queryMask,
																			  queryCodewordScores,
																			  codebookSize,
																			  &queryScoreBoundTokenCount);
					if (queryScoreBoundPrefix == NULL)
						centroidUpperBoundUnsafeFallbacks++;
				}
			}

			denseDocCount =
				Max(storage.multivectorDocCount, meta->tqMultivectorDocCount);
			docScores =
				palloc0(sizeof(double) * (Size) Max(denseDocCount, 1U));
			docBest =
				palloc0(sizeof(double) * (Size) Max(denseDocCount, 1U));
			docBestGeneration =
				palloc0(sizeof(uint32) * (Size) Max(denseDocCount, 1U));
			docMatched =
				palloc0(sizeof(bool) * (Size) Max(denseDocCount, 1U));
			matchedDocIds =
				palloc0(sizeof(uint32) * (Size) Max(denseDocCount, 1U));
			touchedDocIds =
				palloc0(sizeof(uint32) * (Size) Max(denseDocCount, 1U));
			if (centroidScoreBoundPruning)
				centroidScoreBoundTokenMatches =
					palloc0(sizeof(uint16) *
							(Size) Max(denseDocCount, 1U));
			if (centroidBitsetPrefilter)
			{
				instr_time	bitsetStart;
				uint64		bitsetLimitBytes;

				if (collectPhaseStats)
					INSTR_TIME_SET_CURRENT(bitsetStart);
				centroidBitsetBytes =
					((Size) meta->tqMultivectorDocCount + 7U) / 8U;
				bitsetLimitBytes =
					(uint64) Max(pgturbohybrid_multivector_max_accumulator_mb,
								 1) * 1024ULL * 1024ULL;
				if ((uint64) centroidBitsetBytes > bitsetLimitBytes)
					ereport(ERROR,
							(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
							 errmsg("centroid_lite bitset prefilter memory estimate %zu bytes exceeds configured limit %d MB",
									centroidBitsetBytes,
									pgturbohybrid_multivector_max_accumulator_mb),
							 errhint("Increase turbohybrid.multivector_max_accumulator_mb or set turbohybrid.multivector_centroid_lite_bitset_prefilter = off.")));
				centroidBitsetMemoryBytes = (uint64) centroidBitsetBytes;
				centroidBitset =
					palloc0(centroidBitsetBytes > 0 ? centroidBitsetBytes : 1);
				centroidBitsetTokenMatches =
					palloc0(sizeof(uint16) *
							(Size) Max(meta->tqMultivectorDocCount, 1U));
				centroidBitsetMemoryBytes +=
					(uint64) sizeof(uint16) *
					(uint64) Max(meta->tqMultivectorDocCount, 1U);
				if (collectPhaseStats)
					PgturbohybridGraphAddElapsedUint64(&centroidBitsetPrefilterUs,
													   bitsetStart);
			}
			for (int32 qi = 0; qi < query->count; qi++)
			{
				uint32	   *probeCodewords;
				double	   *probeScores;
				uint32		probeLimit =
					(uint32) Max(pgturbohybrid_multivector_centroid_lite_probe_centroids_per_token,
								 1);
				uint32		probeCount;
				uint32		touchedCount = 0;
				uint32		generation = (uint32) qi + 1U;
				double		weight =
					queryWeights != NULL ? (double) queryWeights[qi] : 1.0;
				double		bestProbeScore = 0.0;
				bool		haveBestProbeScore = false;
				instr_time	probeStart;
				instr_time	accumulateStart;

				if (queryMask != NULL && queryMask[qi])
					continue;
				if (collectPhaseStats)
					INSTR_TIME_SET_CURRENT(probeStart);
				probeCodewords =
					palloc0(sizeof(uint32) * (Size) probeLimit);
				probeScores =
					palloc0(sizeof(double) * (Size) probeLimit);
				probeCount =
					PgturbohybridMultiVectorDeterministicCodewords(query, qi,
																   probeLimit,
																   probeCodewords);
				for (uint32 probeIndex = 0; probeIndex < probeCount;
					 probeIndex++)
				{
					uint32		queryCodeword = probeCodewords[probeIndex];
					double		queryCodewordScore;

					if (queryCodeword >= codebookSize)
						ereport(ERROR,
								(errcode(ERRCODE_INDEX_CORRUPTED),
								 errmsg("document-node centroid_lite posting sidecar references an out-of-range centroid"),
								 errhint("REINDEX with multivector_graph = document_nodes and multivector_centroids = kmeans to rebuild persisted centroid posting tuples.")));
					queryCodewordScore =
						queryCodewordScores != NULL ?
						queryCodewordScores[(Size) qi * (Size) codebookSize +
										   queryCodeword] :
						PgturbohybridMultiVectorDeterministicCodewordScore(query,
																		   qi,
																		   queryCodeword);
					probeScores[probeIndex] = queryCodewordScore;
					if (!haveBestProbeScore ||
						queryCodewordScore > bestProbeScore)
					{
						bestProbeScore = queryCodewordScore;
						haveBestProbeScore = true;
					}
				}
				if (collectPhaseStats)
					PgturbohybridGraphAddElapsedUint64(&centroidProbeUs,
													   probeStart);
				for (uint32 probeIndex = 0; probeIndex < probeCount;
					 probeIndex++)
				{
					uint32		queryCodeword = probeCodewords[probeIndex];
					double		queryCodewordScore = probeScores[probeIndex];
					uint32		start;
					uint32		fullEnd;
					uint32		end;
					uint32		listLen;
					uint32		postingLimitForList = centroidPostingLimitPerToken;
					bool		usePerListScoreTopK = false;
					bool		useUniformPostingCap = false;
					instr_time	postingScanStart;

					if (queryCodeword >= codebookSize)
						ereport(ERROR,
								(errcode(ERRCODE_INDEX_CORRUPTED),
								 errmsg("document-node centroid_lite posting sidecar references an out-of-range centroid"),
								 errhint("REINDEX with multivector_graph = document_nodes and multivector_centroids = kmeans to rebuild persisted centroid posting tuples.")));
					start = listOffsets[queryCodeword];
					fullEnd = listOffsets[queryCodeword + 1];
					end = fullEnd;
					if (centroidLiteCodewordMaxsimScoring &&
						pgturbohybrid_multivector_centroid_lite_posting_selection ==
						PGTURBOHYBRID_MULTIVECTOR_POSTING_SELECTION_SCORE_TOPK &&
						centroidCodewordTopM > 1 &&
						centroidPostingLimitPerToken > 0)
					{
						uint64		expandedLimit =
							(uint64) centroidPostingLimitPerToken *
							(uint64) centroidCodewordTopM;

						postingLimitForList =
							expandedLimit > (uint64) PG_UINT32_MAX ?
							PG_UINT32_MAX : (uint32) expandedLimit;
					}
					if (queryCodewordScore < centroidScoreThreshold ||
						(centroidScoreDropFromBest >= 0.0 &&
						 haveBestProbeScore &&
						 queryCodewordScore <
						 bestProbeScore - centroidScoreDropFromBest))
					{
						if (fullEnd > start)
						{
							centroidListsSkippedByThreshold++;
							centroidPostingsSkipped +=
								(uint64) (fullEnd - start);
							centroidPostingCapStrategy =
								queryCodewordScore < centroidScoreThreshold ?
								"score_threshold" : "score_drop_from_best";
						}
						continue;
					}
					if (postingLimitForList > 0 &&
						fullEnd > start &&
						fullEnd - start > postingLimitForList)
					{
						listLen = fullEnd - start;

						if (pgturbohybrid_multivector_centroid_lite_posting_selection ==
							PGTURBOHYBRID_MULTIVECTOR_POSTING_SELECTION_SCORE_TOPK)
						{
							/*
							 * New centroid posting tuples are sorted by a compact
							 * per-posting payload for the document centroid's
							 * dominant component.  That makes bounded per-list
							 * reads deterministic and score-biased.  Older
							 * indexes have zero payloads, remain readable, and
							 * should be REINDEXed before using this path for
							 * quality evidence.
							 */
							centroidPostingCapStrategy =
								postingLimitForList != centroidPostingLimitPerToken ?
								"score_topk_payload_sorted_topm_expanded" :
								"score_topk_payload_sorted";
							usePerListScoreTopK = true;
						}
						else if (centroidLiteUnionScore)
						{
							/*
							 * NextPLAID-style diagnostic path: keep the posting-list
							 * union broad, then rank the touched documents by the
							 * configured compact document-level scorer below.
							 */
							centroidPostingCapStrategy = "union_full_list_then_score";
						}
						else
						{
							end = start + postingLimitForList;
							centroidPostingsSkipped +=
								(uint64) (listLen - postingLimitForList);
							centroidPostingCapStrategy = "uniform_stride";
							useUniformPostingCap = true;
						}
					}
					else if (fullEnd > start)
						centroidPostingCapStrategy =
							centroidLiteUnionScore ? "union_full_list_then_score" : "uncapped_full_list";
					centroidListsVisited++;
					if (centroidBitsetPrefilter && fullEnd > start)
						centroidBitsetListsUsed++;
					if (collectPhaseStats)
						INSTR_TIME_SET_CURRENT(postingScanStart);
					if (usePerListScoreTopK)
					{
						uint32		sampleCount =
							Min(listLen, postingLimitForList);
						PgturbohybridMultiVectorPostingScoreCandidate *selected;
						uint32		selectedCount = 0;

						selected =
							palloc0(sizeof(PgturbohybridMultiVectorPostingScoreCandidate) *
									(Size) sampleCount);
						for (uint32 sampleIndex = 0; sampleIndex < sampleCount;
							 sampleIndex++)
						{
							PgturbohybridGraphMultiVectorCentroidPostingEntry *posting;
							uint32		postingOffset;
							double		dot;
							PgturbohybridMultiVectorPostingScoreCandidate scored;

							CHECK_FOR_INTERRUPTS();
							postingOffset = start + sampleIndex;
							posting = &postings[postingOffset];
							if (posting->docId >= storage.multivectorDocCount)
								ereport(ERROR,
										(errcode(ERRCODE_INDEX_CORRUPTED),
										 errmsg("document-node centroid_lite posting sidecar references an out-of-range document"),
										 errhint("REINDEX with multivector_graph = document_nodes and multivector_centroids = kmeans to rebuild persisted centroid posting tuples.")));
							dot = queryCodewordScore *
								PgturbohybridMultiVectorCentroidPostingPayloadScore(posting->scorePayload);
							scored.docId = posting->docId;
							scored.ordinal = posting->centroidOrdinal;
							scored.score = dot;
							PgturbohybridMultiVectorPostingScoreCandidateOffer(selected,
																			   &selectedCount,
																			   sampleCount,
																			   &scored);
						}
						for (uint32 selectedIndex = 0;
							 selectedIndex < selectedCount;
							 selectedIndex++)
						{
							PgturbohybridMultiVectorPostingScoreCandidate *scored =
								&selected[selectedIndex];
							uint32		docId = scored->docId;

							if (!docMatched[docId])
							{
								docMatched[docId] = true;
								matchedDocIds[matchedDocCount++] = docId;
							}
							if (docBestGeneration[docId] != generation)
							{
								docBestGeneration[docId] = generation;
								docBest[docId] = -DBL_MAX;
								touchedDocIds[touchedCount++] = docId;
							}
							if (scored->score > docBest[docId])
								docBest[docId] = scored->score;
							centroidPostingsTouched++;
						}
						centroidPostingsSelected += selectedCount;
						centroidPostingsSkipped += (uint64) (listLen - selectedCount);
						centroidPostingCapStrategy =
							postingLimitForList != centroidPostingLimitPerToken ?
							"score_topk_payload_sorted_topm_expanded" :
							"score_topk_payload_sorted";
						pfree(selected);
					}
					else
					{
						for (uint32 postingIndex = 0; postingIndex < end - start;
							 postingIndex++)
						{
							PgturbohybridGraphMultiVectorCentroidPostingEntry *posting;
							PgturbohybridMultiVector *centroids;
							uint32		postingOffset;
							uint32		docId;
							double		dot;

							CHECK_FOR_INTERRUPTS();
							if (useUniformPostingCap)
							{
								uint64		listLen = (uint64) (fullEnd - start);
								uint64		limit = (uint64) centroidPostingLimitPerToken;
								uint64		offset =
									((uint64) postingIndex * 2U + 1U) * listLen /
									(limit * 2U);

								postingOffset = start + (uint32) offset;
							}
							else
								postingOffset = start + postingIndex;
							posting = &postings[postingOffset];
							docId = posting->docId;
							if (docId >= storage.multivectorDocCount)
								ereport(ERROR,
										(errcode(ERRCODE_INDEX_CORRUPTED),
										 errmsg("document-node centroid_lite posting sidecar references an out-of-range document"),
										 errhint("REINDEX with multivector_graph = document_nodes and multivector_centroids = kmeans to rebuild persisted centroid posting tuples.")));
							if (centroidLiteCodewordMaxsimScoring)
							{
								dot = queryCodewordScore *
									PgturbohybridMultiVectorCentroidPostingPayloadScore(posting->scorePayload);
							}
							else if (centroidLiteCompactScoring)
								dot = queryCodewordScore *
									PgturbohybridMultiVectorCentroidPostingPayloadScore(posting->scorePayload);
							else
							{
								centroids = storage.multivectorDocCentroids[docId];
								if (centroids == NULL ||
									posting->centroidOrdinal >=
									(uint32) centroids->count)
									ereport(ERROR,
											(errcode(ERRCODE_INDEX_CORRUPTED),
											 errmsg("document-node centroid_lite posting sidecar is invalid"),
											 errhint("REINDEX with multivector_graph = document_nodes and multivector_centroids = kmeans to rebuild persisted centroid posting tuples.")));
								PgturbohybridCheckSameMultiVectorDims(query,
																	  centroids);
								dot = PgturbohybridMultiVectorTokenDot(query, qi,
																	   centroids,
																	   posting->centroidOrdinal);
							}
							{
								if (!docMatched[docId])
								{
									docMatched[docId] = true;
									matchedDocIds[matchedDocCount++] = docId;
								}
								if (docBestGeneration[docId] != generation)
								{
									docBestGeneration[docId] = generation;
									docBest[docId] = -DBL_MAX;
									touchedDocIds[touchedCount++] = docId;
								}
								if (dot > docBest[docId])
									docBest[docId] = dot;
							}
							centroidPostingsTouched++;
							centroidPostingsSelected++;
						}
						}
						if (collectPhaseStats)
							PgturbohybridGraphAddElapsedUint64(&centroidPostingScanUs,
															   postingScanStart);
						}
						pfree(probeScores);
						pfree(probeCodewords);
						if (collectPhaseStats)
							INSTR_TIME_SET_CURRENT(accumulateStart);
						for (uint32 touchedIndex = 0; touchedIndex < touchedCount;
							 touchedIndex++)
						{
							uint32		docId = touchedDocIds[touchedIndex];

							if (docBestGeneration[docId] != generation ||
								docBest[docId] <= -DBL_MAX)
								continue;
							if (centroidBitset != NULL)
							{
							Size		byteIndex = (Size) docId >> 3;
							uint8		mask = (uint8) (1U << (docId & 7U));
							instr_time	bitsetStart;

							if (collectPhaseStats)
								INSTR_TIME_SET_CURRENT(bitsetStart);
							if ((centroidBitset[byteIndex] & mask) == 0)
							{
								centroidBitset[byteIndex] |= mask;
								centroidBitsetDocsSet++;
							}
							if (collectPhaseStats)
								PgturbohybridGraphAddElapsedUint64(&centroidBitsetPrefilterUs,
																   bitsetStart);
						}
							if (centroidBitsetTokenMatches != NULL &&
								centroidBitsetTokenMatches[docId] < UINT16_MAX)
								centroidBitsetTokenMatches[docId]++;
							if (centroidScoreBoundTokenMatches != NULL &&
								centroidScoreBoundTokenMatches[docId] < UINT16_MAX)
								centroidScoreBoundTokenMatches[docId]++;
							docScores[docId] += weight * docBest[docId];
						}
						if (collectPhaseStats)
							PgturbohybridGraphAddElapsedUint64(&centroidAccumulateUs,
															   accumulateStart);
					}

			if (centroidBitset != NULL)
			{
				instr_time	bitsetStart;

				if (collectPhaseStats)
					INSTR_TIME_SET_CURRENT(bitsetStart);
				centroidBitsetDocsSet =
					(uint32) pg_popcount((const char *) centroidBitset,
										 (int) centroidBitsetBytes);
				if (collectPhaseStats)
					PgturbohybridGraphAddElapsedUint64(&centroidBitsetPrefilterUs,
													   bitsetStart);
			}

			for (uint32 matchedIndex = 0; matchedIndex < matchedDocCount;
				 matchedIndex++)
				{
					uint32		docId = matchedDocIds[matchedIndex];
					TqMultiVectorDocMapEntry *docEntry;
					TqDenseCandidate candidate;
					bool		prunedByUpperBound = false;
					double		candidateScore = docScores[docId];
					instr_time	candidateHeapStart;

					CHECK_FOR_INTERRUPTS();
					if (centroidBitsetTokenMatches != NULL &&
						centroidBitsetTokenMatches[docId] < centroidBitsetMinTokenMatches)
						continue;
				if (centroidBitsetPrefilter)
					centroidBitsetDocsAfterThreshold++;
				if (centroidLiteDocCentroidMaxsimScoring)
				{
					PgturbohybridMultiVector *centroids;
					double		maxsim;

					centroids = storage.multivectorDocCentroids[docId];
					if (centroids == NULL)
						ereport(ERROR,
								(errcode(ERRCODE_INDEX_CORRUPTED),
								 errmsg("document-node centroid sidecar is missing a document centroid"),
								 errhint("REINDEX with multivector_graph = document_nodes and multivector_centroids = kmeans to rebuild persisted centroid sidecar tuples.")));
					PgturbohybridCheckSameMultiVectorDims(query, centroids);
					centroidDocCentroidMaxsimPairs +=
						(uint64) query->count * (uint64) centroids->count;
					maxsim = PgturbohybridMultiVectorIndexMaxSim(scan->indexRelation,
																 query,
																 centroids,
																 queryWeights,
																 queryMask);
					candidateScore = maxsim;
					centroidPrerankDocs++;
					centroidCountEffective =
						Max(centroidCountEffective, (uint32) centroids->count);
				}
				else if (centroidLiteCodewordMaxsimScoring)
				{
					uint32		start;
					uint32		end;
					bool		prunedByScoreBound = false;

					if (!storage.multivectorCentroidDocCodesLoaded ||
						storage.multivectorCentroidDocCodeOffsets == NULL ||
						storage.multivectorCentroidDocCodes == NULL)
						ereport(ERROR,
								(errcode(ERRCODE_INDEX_CORRUPTED),
								 errmsg("document-node centroid_lite codeword map is not loaded"),
								 errhint("REINDEX with multivector_graph = document_nodes and multivector_centroids = kmeans to rebuild persisted centroid posting tuples.")));
					start = storage.multivectorCentroidDocCodeOffsets[docId];
					end = storage.multivectorCentroidDocCodeOffsets[docId + 1];
					if (end < start ||
						end > storage.multivectorCentroidDocCodeCount)
						ereport(ERROR,
							(errcode(ERRCODE_INDEX_CORRUPTED),
							 errmsg("document-node centroid_lite codeword map is invalid"),
							 errhint("REINDEX with multivector_graph = document_nodes and multivector_centroids = kmeans to rebuild persisted centroid posting tuples.")));
					if (centroidScoreBoundPruning)
					{
						instr_time	pruneStart;

						centroidUpperBoundDocsChecked++;
						if (collectPhaseStats)
							INSTR_TIME_SET_CURRENT(pruneStart);
						if (docCount >= candidateLimit &&
							candidateLimit > 0 &&
							queryScoreBoundPrefix != NULL &&
							centroidScoreBoundTokenMatches != NULL)
						{
							uint16		tokenMatches =
								centroidScoreBoundTokenMatches[docId];
							uint32		missingTokens =
								queryScoreBoundTokenCount > tokenMatches ?
								queryScoreBoundTokenCount - tokenMatches : 0;
							double		scoreBound =
								candidateScore + queryScoreBoundPrefix[missingTokens];
							double		worstScore = -candidates[0].distance;

							if (scoreBound <= worstScore)
								prunedByScoreBound = true;
						}
						else if (queryScoreBoundPrefix == NULL ||
								 centroidScoreBoundTokenMatches == NULL)
							centroidUpperBoundUnsafeFallbacks++;
						if (collectPhaseStats)
							PgturbohybridGraphAddElapsedUint64(&centroidUpperBoundPruneUs,
															   pruneStart);
						if (prunedByScoreBound)
						{
							centroidUpperBoundDocsPruned++;
							continue;
						}
					}
					candidateScore =
						PgturbohybridMultiVectorCentroidCodewordMaxSimScore(query,
																			queryWeights,
																			queryMask,
																			storage.multivectorCentroidDocCodes + start,
																			end - start,
																			queryCodewordScores,
																			codebookSize,
																			&centroidDocCentroidMaxsimPairs);
					centroidPrerankDocs++;
					centroidCountEffective =
						Max(centroidCountEffective, end - start);
				}
				docEntry = &storage.multivectorDocMap[docId];
				memset(&candidate, 0, sizeof(candidate));
				candidate.nodeId = docEntry->firstNodeId;
				candidate.docId = docId;
				candidate.heaptid = docEntry->heapTid;
				candidate.distance = -candidateScore;
				candidate.similarity =
					queryWeightSum > 0.0 ? candidateScore / queryWeightSum : 0.0;
				candidate.rank = 0;
				candidate.hasDocId = true;
				candidate.exactScored = false;
					if (centroidUpperBoundPruning)
					{
						instr_time	pruneStart;

					centroidUpperBoundDocsChecked++;
					if (collectPhaseStats)
						INSTR_TIME_SET_CURRENT(pruneStart);
					prunedByUpperBound =
						PgturbohybridMultiVectorCentroidLitePruneBySafeUpperBound(candidates,
																				  docCount,
																				  candidateLimit,
																				  &candidate,
																				  &storage,
																				  docId,
																				  &centroidUpperBoundUnsafeFallbacks);
					if (collectPhaseStats)
						PgturbohybridGraphAddElapsedUint64(&centroidUpperBoundPruneUs,
														   pruneStart);
					if (prunedByUpperBound)
					{
						centroidUpperBoundDocsPruned++;
							continue;
						}
					}
					if (collectPhaseStats)
						INSTR_TIME_SET_CURRENT(candidateHeapStart);
					PgturbohybridMultiVectorCandidateHeapOffer(candidates,
															  &docCount,
															  candidateLimit,
															  &candidate);
					if (collectPhaseStats)
						PgturbohybridGraphAddElapsedUint64(&centroidCandidateHeapUs,
														   candidateHeapStart);
				}
			docsScored = matchedDocCount;
			centroidDocsTouched = matchedDocCount;
			if (!centroidBitsetPrefilter)
				centroidBitsetDocsAfterThreshold = matchedDocCount;
			centroidCandidatesBeforeBound = centroidBitsetPrefilter ?
				centroidBitsetDocsAfterThreshold : matchedDocCount;
			centroidCandidatesAfterBound = (uint32) docCount;
			edgesVisited = centroidPostingsTouched;
			maxsimPairs =
				(centroidLiteDocCentroidMaxsimScoring ||
				 centroidLiteCodewordMaxsimScoring) ?
				centroidDocCentroidMaxsimPairs : centroidPostingsTouched;
			if (collectPhaseStats)
				PgturbohybridGraphAddElapsedUint64(&centroidLitePostingUs,
												   subphaseStart);
			if (queryScoreBoundPrefix != NULL)
				pfree(queryScoreBoundPrefix);
		}
		else
			docCount =
				PgturbohybridMultiVectorDocumentGraphTraverse(scan->indexRelation,
															  so, meta, &storage,
															  compact,
															  query,
															  queryWeights,
															  queryMask,
															  queryWeightSum,
															  candidateLimit,
															  searchEf,
															  candidates,
																  &docsScored,
																  &edgesVisited,
																  &maxsimPairs,
																  &compactMaxsimCacheHits,
																  &compactMaxsimCacheMisses,
																  &compactMaxsimBoundChecks,
																  &compactMaxsimDocsPruned,
																  &compactMaxsimTokensSkipped,
																  &sidecarStats);
		PgturbohybridGraphAddElapsedUs(&so->graphTraverseUs, phaseStart);
		if (collectPhaseStats)
			PgturbohybridGraphAddElapsedUint64(&docGraphTraversalUs, phaseStart);
		if (proxyGraph)
			proxyGraphTraversalUs = docGraphTraversalUs;
		if (compactTraversal)
			compactMaxsimScoreUs = docGraphTraversalUs;
		if (centroidLiteCompactScoring)
		{
			compactMaxsimScoreUs = centroidLitePostingUs;
			compactBytes =
				(uint64) storage.multivectorCentroidPostingCount *
				(uint64) sizeof(PgturbohybridGraphMultiVectorCentroidPostingEntry);
		}
		quantizedScores =
			compactTraversal ? docsScored :
			centroidLiteCompactScoring ? centroidPostingsSelected : 0;
		if (quantizedInvertedExperimental)
		{
			docsScored = quantizedInvertedDocsScored;
			edgesVisited = quantizedInvertedPostingsTouched;
			maxsimPairs = quantizedInvertedPostingsTouched;
			quantizedScores = quantizedInvertedPostingsTouched;
		}
		docGraphWarning =
			PgturbohybridMultiVectorDocumentNodeGraphWarning(proxyGraph,
															quantizedInvertedExperimental,
															centroidLite,
															compactTraversal,
															docStorageKind);
	}
	if (collectPhaseStats)
		PgturbohybridGraphAddElapsedUint64(&candidateSourceUs,
										   candidateSourceStart);
	if (proxyGraph)
		proxyCandidateUs = candidateSourceUs;

	if (collectPhaseStats)
		INSTR_TIME_SET_CURRENT(sortStart);
	PgturbohybridMultiVectorCandidateHeapSort(candidates, docCount);
	if (collectPhaseStats)
		PgturbohybridGraphAddElapsedUint64(&finalSortUs, sortStart);
	if (proxyGraph && docCount > 0)
	{
		proxyTopTid = candidates[0].heaptid;
		proxyTopTidValid = true;
	}
	if (qdrantLikeProxyVector && docCount > 0)
	{
		uint64		beforeBytes = sidecarStats.bytesTouched;
		uint64		beforePages = sidecarStats.pagesRead;
		uint64		beforeVectors =
			sidecarStats.vectorsLoaded + sidecarStats.residentVectorsLoaded;
		uint64		beforeReconstructUs = sidecarStats.vectorReconstructUs;
		instr_time	proxyFullSidecarStart;

		if (collectPhaseStats)
			INSTR_TIME_SET_CURRENT(proxyFullSidecarStart);
		docCount =
			PgturbohybridMultiVectorProxyDocumentSidecarRescore(scan->indexRelation,
																so, meta,
																&storage,
																compact,
																query,
																queryWeights,
																queryMask,
																queryWeightSum,
																candidates,
																docCount,
																&maxsimPairs,
																&sidecarStats);
		if (collectPhaseStats)
			proxyFullSidecarLoadUs =
				(uint64) PgturbohybridGraphElapsedUs(proxyFullSidecarStart);
		proxyFullSidecarBytesTouched =
			sidecarStats.bytesTouched > beforeBytes ?
			sidecarStats.bytesTouched - beforeBytes : 0;
		proxyFullSidecarPagesRead =
			sidecarStats.pagesRead > beforePages ?
			sidecarStats.pagesRead - beforePages : 0;
		proxyFullSidecarVectorsLoaded =
			(sidecarStats.vectorsLoaded + sidecarStats.residentVectorsLoaded) >
			beforeVectors ?
			(sidecarStats.vectorsLoaded + sidecarStats.residentVectorsLoaded) -
			beforeVectors : 0;
		proxyFullSidecarReconstructUs =
			sidecarStats.vectorReconstructUs > beforeReconstructUs ?
			sidecarStats.vectorReconstructUs - beforeReconstructUs : 0;
		proxyDocumentRescoreDocs = (uint32) docCount;
	}
	else if (centroidMeanProxy && proxyGraph && docCount > 0)
	{
		if (collectPhaseStats)
			INSTR_TIME_SET_CURRENT(subphaseStart);
		centroidPrerankDocs =
			(uint32) PgturbohybridMultiVectorProxyCentroidPrerank(scan->indexRelation,
																  &storage,
																  query,
																  queryWeights,
																  queryMask,
																  queryWeightSum,
																  candidates,
																  docCount,
																  &maxsimPairs,
																  &centroidCountEffective);
		if (collectPhaseStats)
			PgturbohybridGraphAddElapsedUint64(&proxyScoringUs,
											   subphaseStart);
		centroidDocsTouched += centroidPrerankDocs;
	}
	exactRerankKEffective = 0;
	if (pgturbohybrid_multivector_exact_rerank !=
		PGTURBOHYBRID_MULTIVECTOR_EXACT_RERANK_OFF)
	{
		exactRerankKEffective = Min(docCount, rescoreLimit);
		exactRerankKEffective =
			Min(exactRerankKEffective,
				pgturbohybrid_multivector_exact_rerank_k);
		exactRerankKEffective = Max(exactRerankKEffective, 0);
	}
	if (proxyReservoirsEnabled && exactRerankKEffective > 0 && docCount > 0)
	{
		(void) PgturbohybridMultiVectorApplyProxyReservoirCandidates(candidates,
																	 docCount,
																	 exactRerankKEffective,
																	 resultCtx,
																	 &multivectorReservoirScoreDocs,
																	 &multivectorReservoirCoverageDocs,
																	 &multivectorReservoirMeanDocs,
																	 &multivectorReservoirPerTokenDocs,
																	 &multivectorReservoirUnionDocs,
																	 &multivectorReservoirDuplicates);
	}
	exactRerankLimitOverride = exactRerankKEffective;
	if (!centroidLite && !quantizedInvertedExperimental &&
		fullSidecarAvailable && storage.multivectorDocVectorsLoaded &&
		!proxyOnlyIndex &&
		!(compactTraversal &&
		  docStorageCacheMode ==
		  PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_PAGED))
	{
		exactRerankSidecarMeta = meta;
		exactRerankSidecarStorage = &storage;
		exactRerankSidecarStats = &sidecarStats;
	}
	exactRerankCount =
		PgturbohybridMultiVectorExactHeapRerank(scan, so,
												exactRerankSidecarMeta,
												exactRerankSidecarStorage,
												exactRerankSidecarStats,
												query,
												queryWeights,
												queryMask,
												queryWeightSum,
												candidates,
												docCount,
												exactRerankLimitOverride,
												&exactPairs,
												&exactStats);
	if (exactRerankCount > 0 && docCount > 1)
	{
		if (collectPhaseStats)
			INSTR_TIME_SET_CURRENT(sortStart);
		PgturbohybridMultiVectorCandidateHeapSort(candidates, docCount);
		if (collectPhaseStats)
			PgturbohybridGraphAddElapsedUint64(&finalSortUs, sortStart);
	}
	if (proxyGraph && proxyTopTidValid && docCount > 0)
		proxyTop1Admission =
			ItemPointerEquals(&proxyTopTid, &candidates[0].heaptid);
	if (proxyGraph &&
		meta->tqMultivectorDocCount >= 1000 &&
		candidateLimit > 0 &&
		(uint64) candidateLimit * 4U < (uint64) meta->tqMultivectorDocCount)
	{
		uint64		loadedVectors =
			sidecarStats.vectorsLoaded + sidecarStats.residentVectorsLoaded;

		if (loadedVectors * 10U >= (uint64) meta->tqMultivectorDocCount * 9U)
		{
			proxyVectorNearExhaustiveSidecarTouch = true;
			proxyVectorSidecarTouchReason =
				sidecarStats.residentVectorsLoaded > 0 ?
				(sidecarCacheBuildThisQuery ?
				 "resident_cache_build_all_docs" :
				 "resident_cache_query_materialize_all_docs") :
				"paged_vector_load_many_docs";
		}
		else if (sidecarInitialBytesTouched >= (64ULL * 1024ULL * 1024ULL) &&
				 storage.multivectorDocMapBytes > 0 &&
				 sidecarInitialBytesTouched * 10U >=
				 (uint64) storage.multivectorDocMapBytes * 9U)
		{
			proxyVectorNearExhaustiveSidecarTouch = true;
			proxyVectorSidecarTouchReason = "docmap_sidecar_scan_large";
		}
	}
	for (int i = 0; i < docCount; i++)
		candidates[i].rank = i + 1;

	if (stats != NULL)
	{
		uint64		sidecarFinalVectorsLoaded =
			sidecarStats.vectorsLoaded + sidecarStats.residentVectorsLoaded;

		if (sidecarStats.bytesTouched > sidecarInitialBytesTouched)
			sidecarQueryBytesTouched +=
				sidecarStats.bytesTouched - sidecarInitialBytesTouched;
		if (sidecarStats.pagesRead > sidecarInitialPagesRead)
			sidecarQueryPagesRead +=
				sidecarStats.pagesRead - sidecarInitialPagesRead;
		if (sidecarFinalVectorsLoaded > sidecarInitialVectorsLoaded)
			sidecarQueryVectorsLoaded +=
				sidecarFinalVectorsLoaded - sidecarInitialVectorsLoaded;
		sidecarQueryLoadUs += proxyFullSidecarLoadUs + exactStats.sidecarLoadUs;

		stats->visitedGraphNodes = docsScored;
		stats->scoredCodes = 0;
		stats->denseCandidatesRequested = targetK > 0 ? targetK : docLimit;
		stats->effectiveResultTarget = (uint32) candidateLimit;
		stats->effectiveSearchEf = (uint32) searchEf;
		stats->effectiveRescoreBand =
			(uint32) Max(so->graphEffectiveRescoreBand, 0);
		stats->denseCandidatesReturned = docCount;
		stats->heapRescoreCount = so->graphHeapRescoreCount;
		stats->codePagesRead = so->graphCodePagesRead;
		stats->adjPagesRead = so->graphAdjPagesRead;
		stats->prepareUs = so->graphPrepareUs;
		stats->traverseUs = so->graphTraverseUs;
		stats->entryUs = so->graphEntryUs;
		stats->baseUs = so->graphBaseUs;
		stats->batchUs = so->graphBatchUs;
		stats->heapFetchUs = so->graphHeapFetchUs;
		stats->heapRescoreUs = so->graphHeapRescoreUs;
		stats->sortUs = so->graphSortUs;
		stats->exactRescoreSource = so->graphExactRescoreSource;
		stats->multivectorEnabled = true;
		stats->multivectorQueryVectors = (uint32) query->count;
		stats->multivectorDocVectorsLimit =
			(uint32) pgturbohybrid_multivector_max_doc_vectors;
		stats->multivectorSubvectorSearches = 0;
		stats->multivectorRawSubvectorHits = docsScored;
		stats->multivectorDocMapSource =
			PGTURBOHYBRID_MULTIVECTOR_DOCMAP_SOURCE_SIDECAR;
		strlcpy(stats->multivectorCandidateSource,
				explicitProxyVector ? "proxy_vector" :
				centroidLite ? "centroid_lite" :
				quantizedInvertedExperimental ? "quantized_inverted_experimental" :
				documentNodesSource ? "document_nodes" : "graph",
				sizeof(stats->multivectorCandidateSource));
		strlcpy(stats->multivectorCandidatePath,
				proxyGraph && proxyOnlyIndex ? "proxy_graph" :
				exhaustiveScan ? "exact_doc_scan" :
				proxyGraph ? "proxy_graph" :
				centroidLite ? "centroid_lite" :
				quantizedInvertedExperimental ? "quantized_inverted_experimental" :
				"document_graph",
				sizeof(stats->multivectorCandidatePath));
		strlcpy(stats->multivectorProxyEncoderKind,
				proxyGraph ?
				PgturbohybridMultiVectorProxyEncoderName(proxyEncoder) :
				"none",
				sizeof(stats->multivectorProxyEncoderKind));
		{
			bool		projectionLoaded = false;
			int32		projectionDim = 0;
			uint64		projectionWeightBytes = 0;
			const char *projectionModel = "";
			const char *projectionChecksum = "";

			PgturbohybridMultiVectorLearnedProjectionInfo(&projectionLoaded,
														  &projectionDim,
														  &projectionWeightBytes,
														  &projectionModel,
														  &projectionChecksum);
			stats->learnedProjection.loaded = projectionLoaded;
			stats->learnedProjection.dim = (uint32) Max(projectionDim, 0);
			stats->learnedProjection.weightBytes = projectionWeightBytes;
			strlcpy(stats->learnedProjection.model,
					projectionModel,
					sizeof(stats->learnedProjection.model));
			strlcpy(stats->learnedProjection.checksum,
					projectionChecksum,
					sizeof(stats->learnedProjection.checksum));
			stats->learnedProjection.queryEncodeUs =
				learnedProjectionQueryEncodeUs;
		}
		strlcpy(stats->multivectorGraphMode,
				PgturbohybridMultiVectorGraphModeName(meta->tqMultivectorGraphMode),
				sizeof(stats->multivectorGraphMode));
		stats->multivectorProxyGraphSearches =
			(proxyGraph && !exhaustiveScan) ? 1 : 0;
		stats->multivectorExactTokenScanEnabled = false;
		stats->multivectorExactTokenScanNodesScored = 0;
		stats->multivectorPlainFallbackUsed = false;
		strlcpy(stats->multivectorPlainFallbackReason, "not_applicable",
				sizeof(stats->multivectorPlainFallbackReason));
		stats->multivectorPlainFallbackDocsScored = 0;
		stats->multivectorPlainFallbackPairs = 0;
		stats->multivectorDocGraphPrototypeEnabled = false;
		stats->multivectorDocGraphNodes =
			(uint64) meta->tqMultivectorDocCount;
		stats->multivectorDocGraphDocsScored = docsScored;
		stats->multivectorDocGraphEdgesVisited =
			exhaustiveScan ? docsScored : edgesVisited;
		stats->multivectorDocGraphCandidates = (uint32) docCount;
		stats->multivectorDocGraphSearchEf = (uint32) searchEf;
		stats->multivectorDocGraphOversampling =
			(uint32) pgturbohybrid_multivector_doc_graph_oversampling;
		stats->multivectorDocGraphRescoreK =
			(uint32) exactRerankKEffective;
		stats->multivectorDocGraphEntrySampleConfigured =
			(uint32) Max(so->graphEntrySampleConfigured, 0);
		stats->multivectorDocGraphEntrySampleEffective =
			(uint32) Max(so->graphEntrySampleEffective, 0);
		stats->multivectorDocGraphEntrySampleScored =
			(uint32) Max(so->graphEntrySampleScored, 0);
		stats->multivectorDocGraphQuantizedScores =
			proxyDocumentCompactRescore ? proxyDocumentRescoreDocs :
			quantizedScores;
		stats->compactMaxsimScoreUs =
			(compactTraversal || centroidLiteCompactScoring) ?
			compactMaxsimScoreUs : 0;
		stats->compactMaxsimPairs =
			(compactTraversal || centroidLiteCompactScoring) ?
			maxsimPairs : 0;
		stats->compactMaxsimCacheHits =
			compactTraversal ? compactMaxsimCacheHits : 0;
		stats->compactMaxsimCacheMisses =
			compactTraversal ? compactMaxsimCacheMisses : 0;
		stats->compactMaxsimBoundChecks =
			compactTraversal ? compactMaxsimBoundChecks : 0;
		stats->compactMaxsimDocsPruned =
			compactTraversal ? compactMaxsimDocsPruned : 0;
		stats->compactMaxsimTokensSkipped =
			compactTraversal ? compactMaxsimTokensSkipped : 0;
		strlcpy(stats->multivectorDocGraphStorageKind,
				docStorageKindName,
				sizeof(stats->multivectorDocGraphStorageKind));
		stats->proxy.onlyIndex = proxyOnlyIndex;
		stats->centroid.onlyIndex = centroidOnlyIndex;
		stats->fullMultivectorSidecarAvailable = fullSidecarAvailable;
			stats->centroid.sidecarAvailable =
				(meta->tqMultivectorDocMapFlags &
				 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_CENTROIDS) != 0;
			stats->centroid.docCodesAvailable =
				(meta->tqMultivectorDocMapFlags &
				 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_CENTROID_DOC_CODES) != 0;
			stats->quantizedInverted.sidecarAvailable =
				(meta->tqMultivectorDocMapFlags &
				 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_QUANTIZED_POSTINGS) != 0 &&
			(meta->tqMultivectorDocMapFlags &
			 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_QUANTIZED_CODEBOOK) != 0;
		strlcpy(stats->multivectorDocGraphRescoreSource,
				PgturbohybridMultiVectorRerankSourceName(exactStats.source),
				sizeof(stats->multivectorDocGraphRescoreSource));
		stats->multivectorDocGraphExactRerankDocs =
			(uint32) exactRerankCount;
		stats->multivectorDocGraphHeapFetches = exactStats.heapFetches;
		strlcpy(stats->multivectorDocGraphWarning, docGraphWarning,
				sizeof(stats->multivectorDocGraphWarning));
		stats->multivectorProxyCandidateTarget =
			proxyGraph ? (uint32) proxyCandidateTarget : 0;
		stats->multivectorProxyCandidatesReturned =
			proxyGraph ? (uint32) docCount : 0;
		stats->multivectorExactRerankKEffective =
			(uint32) exactRerankKEffective;
		stats->proxy.candidateLimitEffective =
			proxyGraph ? (uint32) proxyCandidateLimitEffective : 0;
		strlcpy(stats->proxy.candidateLimitSource,
				proxyGraph ? proxyCandidateLimitSource : "none",
				sizeof(stats->proxy.candidateLimitSource));
		stats->proxy.graphNodesVisited =
			proxyGraph ? so->graphVisitedNodes : 0;
		stats->proxy.graphEdgesVisited =
			proxyGraph ? so->graphBaseVisitedChecks : 0;
		stats->proxy.graphCandidatesSeen =
			proxyGraph ? (uint32) Max(proxyGraphHitCount, 0) : 0;
		stats->proxy.candidatesReturned =
			proxyGraph ? (uint32) docCount : 0;
		stats->proxy.vectorScoresComputed =
			proxyGraph ? so->graphBaseBatchNodes : 0;
		stats->proxy.vectorScoreUs =
			proxyGraph ? proxyScoringUs : 0;
		stats->proxy.candidates = proxyGraph ? (uint32) docCount : 0;
		stats->proxy.lazySidecarVectors =
			proxyGraph && proxyLazySidecarVectors;
		strlcpy(stats->multivectorDocStorageCacheRequested,
				docStorageCacheRequestedName,
				sizeof(stats->multivectorDocStorageCacheRequested));
		strlcpy(stats->multivectorDocStorageCacheEffective,
				docStorageCacheModeName,
				sizeof(stats->multivectorDocStorageCacheEffective));
		stats->proxy.top1Admission = proxyGraph && proxyTop1Admission;
		stats->proxy.exactRerankDocs =
			proxyGraph ? (uint32) exactRerankCount : 0;
		stats->proxy.fullSidecarVectorsLoaded =
			proxyGraph ? proxyFullSidecarVectorsLoaded : 0;
		stats->proxy.fullSidecarBytesTouched =
			proxyGraph ? proxyFullSidecarBytesTouched : 0;
		stats->proxy.fullSidecarPagesRead =
			proxyGraph ? proxyFullSidecarPagesRead : 0;
		stats->proxy.fullSidecarLoadUs =
			proxyGraph ? proxyFullSidecarLoadUs : 0;
		stats->proxy.fullSidecarReconstructUs =
			proxyGraph ? proxyFullSidecarReconstructUs : 0;
		stats->proxy.exactRerankHeapFetches =
			proxyGraph ? exactStats.heapFetches : 0;
		stats->proxy.exactRerankSidecarFetches =
			proxyGraph ? exactStats.sidecarReads : 0;
		stats->proxy.exactRerankBytesTouched =
			proxyGraph ? exactStats.sidecarBytes : 0;
		stats->proxy.exactRerankUs =
			proxyGraph ? exactStats.exactMaxsimUs : 0;
		stats->sidecar.cacheBuildThisQuery =
			(proxyGraph || centroidLite || quantizedInvertedExperimental) &&
			sidecarCacheBuildThisQuery;
		stats->sidecar.cacheBuildBytes =
			(proxyGraph || centroidLite || quantizedInvertedExperimental) ?
			sidecarCacheBuildBytes : 0;
		stats->sidecar.cacheBuildPagesRead =
			(proxyGraph || centroidLite || quantizedInvertedExperimental) ?
			sidecarCacheBuildPagesRead : 0;
		stats->sidecar.cacheBuildUs =
			(proxyGraph || centroidLite || quantizedInvertedExperimental) ?
			sidecarCacheBuildUs : 0;
		stats->sidecar.queryBytesTouched =
			(proxyGraph || centroidLite || quantizedInvertedExperimental) ?
			sidecarQueryBytesTouched : 0;
		stats->sidecar.queryPagesRead =
			(proxyGraph || centroidLite || quantizedInvertedExperimental) ?
			sidecarQueryPagesRead : 0;
		stats->sidecar.queryVectorsLoaded =
			(proxyGraph || centroidLite || quantizedInvertedExperimental) ?
			sidecarQueryVectorsLoaded : 0;
		stats->sidecar.queryLoadUs =
			(proxyGraph || centroidLite || quantizedInvertedExperimental) ?
			sidecarQueryLoadUs : 0;
		stats->sidecar.queryUs = stats->sidecar.queryLoadUs;
		stats->proxy.vectorUsesFullSidecarForGraph =
			proxyGraph && proxyVectorUsesFullSidecarForGraph;
		stats->proxy.vectorNearExhaustiveSidecarTouch =
			proxyGraph && proxyVectorNearExhaustiveSidecarTouch;
		strlcpy(stats->proxy.vectorSidecarTouchReason,
				proxyGraph ? proxyVectorSidecarTouchReason : "none",
				sizeof(stats->proxy.vectorSidecarTouchReason));
		centroidCandidates = centroidLite ? (uint32) docCount : 0;
		centroidPrunedDocs =
			centroidDocsTouched > (uint64) docCount ?
			centroidDocsTouched - (uint64) docCount : 0;
		stats->centroid.listsVisited = centroidListsVisited;
		stats->centroid.docsTouched = centroidDocsTouched;
		stats->centroid.prunedDocs = centroidPrunedDocs;
		stats->centroid.postingsTouched =
			centroidLite ? edgesVisited : 0;
		stats->centroid.postingsSelected =
			centroidLite ? centroidPostingsSelected : 0;
		stats->centroid.postingsSkipped =
			centroidLite ? centroidPostingsSkipped : 0;
		stats->centroid.probeUs =
			centroidLite ? centroidProbeUs : 0;
		stats->centroid.postingScanUs =
			centroidLite ? centroidPostingScanUs : 0;
		stats->centroid.accumulateUs =
			centroidLite ? centroidAccumulateUs : 0;
		stats->centroid.candidateHeapUs =
			centroidLite ? centroidCandidateHeapUs : 0;
		stats->centroid.postingLimitPerToken =
			centroidLite ? centroidPostingLimitPerToken : 0;
		stats->centroid.probeCentroidsPerToken =
			centroidLite ?
			(uint32) Max(pgturbohybrid_multivector_centroid_lite_probe_centroids_per_token,
						 1) : 0;
		stats->centroid.codewordTopM =
			centroidLite ?
			(uint32) Max(pgturbohybrid_multivector_centroid_lite_codeword_top_m,
						 1) : 0;
		stats->centroid.scoreThreshold =
			centroidLite ? centroidScoreThreshold : -1.0;
		stats->centroid.scoreDropFromBest =
			centroidLite ? centroidScoreDropFromBest : -1.0;
		stats->centroid.listsSkippedByThreshold =
			centroidLite ? centroidListsSkippedByThreshold : 0;
		if (centroidLite &&
			stats->centroid.postingLimitPerToken == 0 &&
			stats->centroid.postingsTouched > 0)
			centroidPostingCapStrategy = "uncapped_full_list";
		strlcpy(stats->centroid.postingCapStrategy,
				centroidLite ? centroidPostingCapStrategy : "none",
				sizeof(stats->centroid.postingCapStrategy));
		strlcpy(stats->centroid.candidateScoring,
				centroidLite ? centroidCandidateScoring : "none",
				sizeof(stats->centroid.candidateScoring));
		stats->centroid.candidates = centroidCandidates;
		stats->centroid.bitsetPrefilterEnabled = centroidBitsetPrefilter;
		stats->centroid.bitsetMinTokenMatches =
			centroidBitsetPrefilter ? centroidBitsetMinTokenMatches : 0;
		stats->centroid.bitsetListsUsed =
				centroidBitsetPrefilter ? centroidBitsetListsUsed : 0;
			stats->centroid.bitsetDocsSet =
				centroidBitsetPrefilter ? centroidBitsetDocsSet : 0;
			stats->centroid.bitsetDocsAfterThreshold =
				centroidBitsetPrefilter ? centroidBitsetDocsAfterThreshold : 0;
			stats->centroid.bitsetPrefilterUs =
				centroidBitsetPrefilter ? centroidBitsetPrefilterUs : 0;
			stats->centroid.bitsetMemoryBytes =
				centroidBitsetPrefilter ? centroidBitsetMemoryBytes : 0;
			stats->centroid.upperBoundEnabled =
				centroidUpperBoundPruning || centroidScoreBoundPruning;
			stats->centroid.upperBoundDocsChecked =
				(centroidUpperBoundPruning || centroidScoreBoundPruning) ?
				centroidUpperBoundDocsChecked : 0;
			stats->centroid.upperBoundDocsPruned =
				(centroidUpperBoundPruning || centroidScoreBoundPruning) ?
				centroidUpperBoundDocsPruned : 0;
			stats->centroid.upperBoundPruneUs =
				(centroidUpperBoundPruning || centroidScoreBoundPruning) ?
				centroidUpperBoundPruneUs : 0;
			stats->centroid.upperBoundUnsafeFallbacks =
				(centroidUpperBoundPruning || centroidScoreBoundPruning) ?
				centroidUpperBoundUnsafeFallbacks : 0;
			stats->centroid.candidatesBeforeBound =
				(centroidUpperBoundPruning || centroidScoreBoundPruning) ?
				centroidCandidatesBeforeBound : 0;
			stats->centroid.candidatesAfterBound =
				(centroidUpperBoundPruning || centroidScoreBoundPruning) ?
				centroidCandidatesAfterBound : 0;
		stats->multivectorCentroidCount = centroidCountEffective;
		stats->multivectorCentroidPrerankDocs = centroidPrerankDocs;
		stats->multivectorFullMaxsimRerankDocs =
			(uint32) exactRerankCount;
		quantizedInvertedCandidates =
			quantizedInvertedExperimental ? (uint32) docCount : 0;
		stats->quantizedInverted.listsVisited =
			quantizedInvertedListsVisited;
		stats->quantizedInverted.postingsTouched =
			quantizedInvertedPostingsTouched;
		stats->quantizedInverted.postingsSelected =
			quantizedInvertedExperimental ? quantizedInvertedPostingsSelected : 0;
		stats->quantizedInverted.postingsSkipped =
			quantizedInvertedExperimental ? quantizedInvertedPostingsSkipped : 0;
		stats->quantizedInverted.postingLimitPerToken =
			quantizedInvertedExperimental ? quantizedInvertedPostingLimitPerToken : 0;
		stats->quantizedInverted.probeCodewordsPerToken =
			quantizedInvertedExperimental ?
			(uint32) Max(pgturbohybrid_multivector_quantized_inverted_probe_codewords_per_token,
						 1) : 0;
		strlcpy(stats->quantizedInverted.postingCapStrategy,
				quantizedInvertedExperimental ? quantizedInvertedPostingCapStrategy : "none",
				sizeof(stats->quantizedInverted.postingCapStrategy));
		stats->quantizedInverted.docsScored =
			quantizedInvertedDocsScored;
		stats->quantizedInverted.candidates =
			quantizedInvertedCandidates;
		stats->quantizedInverted.exactRerankDocs =
			quantizedInvertedExperimental ? (uint32) exactRerankCount : 0;
		strlcpy(stats->quantizedInverted.codebookSource,
				quantizedInvertedExperimental ?
				PgturbohybridQuantizedInvertedCodebookSourceName(storage.multivectorQuantizedInvertedCodebookSource) :
				"",
				sizeof(stats->quantizedInverted.codebookSource));
		stats->quantizedInverted.codebookSize =
			quantizedInvertedExperimental ?
			storage.multivectorQuantizedInvertedCodebookSize : 0;
		stats->quantizedInverted.codebookDim =
			quantizedInvertedExperimental ?
			storage.multivectorQuantizedInvertedCodebookDim : 0;
		strlcpy(stats->quantizedInverted.codebookChecksum,
				quantizedInvertedExperimental ?
				storage.multivectorQuantizedInvertedCodebookChecksum : "",
				sizeof(stats->quantizedInverted.codebookChecksum));
		stats->quantizedInverted.codebookTopM =
			quantizedInvertedExperimental ?
			storage.multivectorQuantizedInvertedCodebookTopM : 0;
		stats->quantizedInverted.assignmentUs =
			quantizedInvertedExperimental ?
			quantizedInvertedAssignmentUs : 0;
		stats->quantizedInverted.queryCodewordScoreUs =
			quantizedInvertedExperimental ?
			quantizedInvertedQueryCodewordScoreUs : 0;
		strlcpy(stats->quantizedInverted.queryCodewordKernel,
				quantizedInvertedExperimental ?
				quantizedInvertedQueryCodewordKernel : "off",
				sizeof(stats->quantizedInverted.queryCodewordKernel));
		stats->quantizedInverted.queryCodewordScoresComputed =
			quantizedInvertedExperimental ?
			quantizedInvertedQueryCodewordScoresComputed : 0;
		stats->quantizedInverted.queryCodewordBlocks =
			quantizedInvertedExperimental ?
			quantizedInvertedQueryCodewordBlocks : 0;
		stats->quantizedInverted.queryCodewordTopkUs =
			quantizedInvertedExperimental ?
			quantizedInvertedQueryCodewordTopkUs : 0;
		stats->quantizedInverted.queryCodewordFullMatrixMaterialized =
			quantizedInvertedExperimental ?
			quantizedInvertedQueryCodewordFullMatrixMaterialized : false;
		stats->quantizedInverted.queryCodewordActiveQueryTokens =
			quantizedInvertedExperimental ?
			quantizedInvertedQueryCodewordActiveQueryTokens : 0;
		stats->quantizedInverted.queryCodewordSkippedQueryTokens =
			quantizedInvertedExperimental ?
			quantizedInvertedQueryCodewordSkippedQueryTokens : 0;
		stats->quantizedInverted.listOffsetBytes =
			quantizedInvertedExperimental ?
			((uint64) storage.multivectorQuantizedInvertedCodebookSize + 1) *
			(uint64) sizeof(uint32) : 0;
		stats->quantizedInverted.postingBytes =
			quantizedInvertedExperimental ?
			(uint64) storage.multivectorQuantizedInvertedPostingCount *
			(uint64) sizeof(PgturbohybridGraphMultiVectorQuantizedPostingEntry) : 0;
		stats->quantizedInverted.sidecarBytes =
			stats->quantizedInverted.listOffsetBytes +
			stats->quantizedInverted.postingBytes;
		strlcpy(stats->quantizedInverted.compactKernel,
				quantizedInvertedExperimental ?
				quantizedInvertedCompactKernel : "off",
				sizeof(stats->quantizedInverted.compactKernel));
		strlcpy(stats->quantizedInverted.compactScoreSource,
				quantizedInvertedExperimental && quantizedInvertedCompactScoring ?
				"full_doc_codeword_maxsim" : "none",
				sizeof(stats->quantizedInverted.compactScoreSource));
		stats->quantizedInverted.compactScoreUs =
			quantizedInvertedExperimental ? quantizedInvertedCompactScoreUs : 0;
		stats->quantizedInverted.compactDocsScored =
			quantizedInvertedExperimental ?
			quantizedInvertedCompactDocsScored : 0;
		stats->quantizedInverted.compactPayloadBytes =
				quantizedInvertedExperimental ?
				quantizedInvertedCompactPayloadBytes : 0;
			strlcpy(stats->quantizedInverted.compactDocOrder,
					quantizedInvertedExperimental &&
					quantizedInvertedCompactScoring ?
					quantizedInvertedCompactDocOrder : "original",
					sizeof(stats->quantizedInverted.compactDocOrder));
			stats->quantizedInverted.compactInnerAllocations =
				quantizedInvertedExperimental ?
				quantizedInvertedCompactInnerAllocations : 0;
			stats->quantizedInverted.compactActiveQueryTokens =
				quantizedInvertedExperimental ?
				quantizedInvertedCompactActiveQueryTokens : 0;
			stats->quantizedInverted.compactPairsEvaluated =
				quantizedInvertedExperimental ?
				quantizedInvertedCompactPairsEvaluated : 0;
			stats->quantizedInverted.compactPairsSkipped =
				quantizedInvertedExperimental ?
				quantizedInvertedCompactPairsSkipped : 0;
			stats->quantizedInverted.compactPrefetches =
				quantizedInvertedExperimental ?
				quantizedInvertedCompactPrefetches : 0;
			stats->quantizedInverted.compactAvgDocTokens =
				quantizedInvertedExperimental &&
				quantizedInvertedCompactDocsScored > 0 ?
				(double) (quantizedInvertedCompactPayloadBytes / sizeof(uint32)) /
				(double) quantizedInvertedCompactDocsScored : 0.0;
			stats->quantizedInverted.compactUsPerDoc =
				quantizedInvertedExperimental &&
				quantizedInvertedCompactDocsScored > 0 ?
				(double) quantizedInvertedCompactScoreUs /
				(double) quantizedInvertedCompactDocsScored : 0.0;
			stats->quantizedInverted.compactPayloadBytesPerDoc =
				quantizedInvertedExperimental &&
				quantizedInvertedCompactDocsScored > 0 ?
				(double) quantizedInvertedCompactPayloadBytes /
				(double) quantizedInvertedCompactDocsScored : 0.0;
			stats->quantizedInverted.compactTopKChangedVsScalar =
				quantizedInvertedExperimental ?
				quantizedInvertedCompactTopKChangedVsScalar : false;
			stats->quantizedInverted.precompactEnabled =
				quantizedInvertedExperimental &&
				quantizedInvertedPrecompactEnabled;
			strlcpy(stats->quantizedInverted.precompactMode,
					quantizedInvertedExperimental &&
					quantizedInvertedPrecompactEnabled ?
					quantizedInvertedPrecompactMode ==
					PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_PRECOMPACT_CENTROID_MAXSIM_TOPK ?
					"centroid_maxsim_topk" : "centroid_maxsim_reservoir" :
					"off",
					sizeof(stats->quantizedInverted.precompactMode));
			stats->quantizedInverted.docsTouchedBeforePrecompact =
				quantizedInvertedExperimental &&
				quantizedInvertedPrecompactEnabled ?
				quantizedInvertedDocsTouchedBeforePrecompact : 0;
			stats->quantizedInverted.precompactScoreK =
				quantizedInvertedExperimental &&
				quantizedInvertedPrecompactEnabled ?
				quantizedInvertedPrecompactScoreK : 0;
			stats->quantizedInverted.precompactCoverageK =
				quantizedInvertedExperimental &&
				quantizedInvertedPrecompactEnabled ?
				quantizedInvertedPrecompactCoverageK : 0;
			stats->quantizedInverted.precompactPerTokenK =
				quantizedInvertedExperimental &&
				quantizedInvertedPrecompactEnabled ?
				quantizedInvertedPrecompactPerTokenK : 0;
			stats->quantizedInverted.compactMaxDocs =
				quantizedInvertedExperimental &&
				quantizedInvertedPrecompactEnabled ?
				quantizedInvertedCompactMaxDocs : 0;
			stats->quantizedInverted.precompactScoreDocs =
				quantizedInvertedExperimental &&
				quantizedInvertedPrecompactEnabled ?
				quantizedInvertedPrecompactScoreDocs : 0;
			stats->quantizedInverted.precompactCoverageDocs =
				quantizedInvertedExperimental &&
				quantizedInvertedPrecompactEnabled ?
				quantizedInvertedPrecompactCoverageDocs : 0;
			stats->quantizedInverted.precompactPerTokenDocs =
				quantizedInvertedExperimental &&
				quantizedInvertedPrecompactEnabled ?
				quantizedInvertedPrecompactPerTokenDocs : 0;
			stats->quantizedInverted.precompactUnionDocs =
				quantizedInvertedExperimental &&
				quantizedInvertedPrecompactEnabled ?
				quantizedInvertedPrecompactUnionDocs : 0;
			stats->quantizedInverted.precompactDuplicates =
				quantizedInvertedExperimental &&
				quantizedInvertedPrecompactEnabled ?
				quantizedInvertedPrecompactDuplicates : 0;
			stats->quantizedInverted.precompactPrunedDocs =
				quantizedInvertedExperimental &&
				quantizedInvertedPrecompactEnabled ?
				quantizedInvertedPrecompactPrunedDocs : 0;
			stats->quantizedInverted.precompactUs =
				quantizedInvertedExperimental &&
				quantizedInvertedPrecompactEnabled ?
				quantizedInvertedPrecompactUs : 0;
			stats->quantizedInverted.compactDocsSkippedByPrecompact =
				quantizedInvertedExperimental &&
				quantizedInvertedPrecompactEnabled ?
				quantizedInvertedCompactDocsSkippedByPrecompact : 0;
			strlcpy(stats->quantizedInverted.tokenCoverageMode,
					quantizedInvertedExperimental && quantizedInvertedTokenCoverageLinear ?
					"linear" : "off",
					sizeof(stats->quantizedInverted.tokenCoverageMode));
			stats->quantizedInverted.activeQueryTokens =
				quantizedInvertedExperimental ?
				quantizedInvertedActiveQueryTokens : 0;
			stats->quantizedInverted.tokenMatchesTotal =
				quantizedInvertedExperimental ?
				quantizedInvertedTokenMatchesTotal : 0;
			stats->quantizedInverted.tokenMatchesMax =
				quantizedInvertedExperimental ?
				quantizedInvertedTokenMatchesMax : 0;
			stats->quantizedInverted.minTokenMatches =
				quantizedInvertedExperimental ?
				quantizedInvertedMinTokenMatches : 0;
			stats->quantizedInverted.tokenMatchFilteredDocs =
				quantizedInvertedExperimental ?
				quantizedInvertedTokenMatchFilteredDocs : 0;
			stats->quantizedInverted.scoreBoundPruningEnabled =
				quantizedInvertedScoreBoundPruning;
			stats->quantizedInverted.scoreBoundDocsChecked =
				quantizedInvertedScoreBoundPruning ?
				quantizedInvertedScoreBoundDocsChecked : 0;
			stats->quantizedInverted.scoreBoundDocsPruned =
				quantizedInvertedScoreBoundPruning ?
				quantizedInvertedScoreBoundDocsPruned : 0;
			stats->quantizedInverted.scoreBoundPruneUs =
				quantizedInvertedScoreBoundPruning ?
				quantizedInvertedScoreBoundPruneUs : 0;
			stats->quantizedInverted.scoreBoundUnsafeFallbacks =
				quantizedInvertedScoreBoundPruning ?
				quantizedInvertedScoreBoundUnsafeFallbacks : 0;
			stats->quantizedInverted.candidatesBeforeBound =
				quantizedInvertedScoreBoundPruning ?
				quantizedInvertedCandidatesBeforeBound : 0;
			stats->quantizedInverted.candidatesAfterBound =
				quantizedInvertedScoreBoundPruning ?
				quantizedInvertedCandidatesAfterBound : 0;
			strlcpy(stats->multivectorDocSidecarCacheMode,
					sidecarStats.cacheMode,
				sizeof(stats->multivectorDocSidecarCacheMode));
		stats->multivectorDocSidecarPagesRead = sidecarStats.pagesRead;
		stats->multivectorDocSidecarCacheHits = sidecarStats.cacheHits;
		stats->multivectorDocSidecarCacheMisses = sidecarStats.cacheMisses;
		stats->multivectorDocSidecarBytesTouched = sidecarStats.bytesTouched;
		stats->multivectorDocSidecarVectorsLoaded =
			sidecarStats.vectorsLoaded;
		stats->multivectorDocSidecarDocMapPagesRead =
			sidecarStats.docMapPagesRead;
		stats->multivectorDocSidecarDocMapBytesTouched =
			sidecarStats.docMapBytesTouched;
		stats->multivectorDocSidecarResidentVectorsLoaded =
			sidecarStats.residentVectorsLoaded;
		stats->multivectorDocSidecarResidentBytesLoaded =
			sidecarStats.residentVectorBytesLoaded;
		stats->multivectorDocSidecarVectorChunkRefBytesTouched =
			sidecarStats.vectorChunkRefBytesTouched;
		stats->multivectorDocSidecarPagedVectorPagesRead =
			sidecarStats.pagedVectorPagesRead;
		stats->multivectorDocSidecarPagedVectorBytesTouched =
			sidecarStats.pagedVectorBytesTouched;
		stats->multivectorSidecarPageReadUs = sidecarStats.pageReadUs;
		stats->multivectorSidecarVectorReconstructUs =
			sidecarStats.vectorReconstructUs;
		stats->multivectorTokensOriginal = originalTokens;
		stats->multivectorTokensPooled = pooledTokens;
		stats->multivectorReservoirsEnabled =
			proxyReservoirsEnabled && multivectorReservoirUnionDocs > 0;
		stats->multivectorReservoirScoreDocs =
			multivectorReservoirScoreDocs;
		stats->multivectorReservoirCoverageDocs =
			multivectorReservoirCoverageDocs;
		stats->multivectorReservoirMeanDocs =
			multivectorReservoirMeanDocs;
		stats->multivectorReservoirPerTokenDocs =
			multivectorReservoirPerTokenDocs;
		stats->multivectorReservoirBm25Docs = 0;
		stats->multivectorReservoirUnionDocs =
			multivectorReservoirUnionDocs;
		stats->multivectorReservoirDuplicates =
			multivectorReservoirDuplicates;
		stats->multivectorDocMapBytes = storage.multivectorDocMapBytes;
		stats->multivectorUniqueDocs = docsScored;
		stats->multivectorDuplicateDocHits = 0;
		stats->multivectorMaxsimUpdates = maxsimPairs;
		stats->multivectorDocCandidates = (uint32) docCount;
		stats->multivectorExactRerankEnabled =
			pgturbohybrid_multivector_exact_rerank !=
			PGTURBOHYBRID_MULTIVECTOR_EXACT_RERANK_OFF;
		stats->multivectorExactRerankDocs = (uint32) exactRerankCount;
		stats->multivectorExactRerankPairs = exactPairs;
		stats->multivectorExactRerankSource = exactStats.source;
		stats->multivectorExactRerankHeapFetches = exactStats.heapFetches;
		stats->multivectorExactRerankSidecarReads = exactStats.sidecarReads;
		stats->multivectorExactRerankSidecarBytes = exactStats.sidecarBytes;
		stats->multivectorCandidateSourceUs = candidateSourceUs;
		stats->multivectorDocGraphTraversalUs = docGraphTraversalUs;
		stats->multivectorProxyCandidateUs = proxyCandidateUs;
		stats->multivectorProxyGraphTraversalUs = proxyGraphTraversalUs;
		stats->multivectorProxyScoringUs = proxyScoringUs;
		stats->multivectorCentroidLitePostingUs = centroidLitePostingUs;
		stats->multivectorQuantizedInvertedPostingUs =
			quantizedInvertedPostingUs;
		stats->multivectorSidecarLoadUs =
			sidecarLoadUs + exactStats.sidecarLoadUs;
		stats->multivectorHeapVisibilityUs = exactStats.heapVisibilityUs;
		stats->multivectorExactHeapFetchUs = exactStats.exactHeapFetchUs;
		stats->multivectorExactRerankUs = exactStats.exactMaxsimUs;
		stats->multivectorFinalSortUs = finalSortUs;
		stats->exactRerankCandidates = exactStats.candidates;
		stats->exactRerankTokensEvaluated = exactStats.tokensEvaluated;
		stats->exactRerankTokensSkipped = exactStats.tokensSkipped;
		stats->exactRerankPairsSaved = exactStats.pairsSaved;
		stats->adaptiveRerankTopKChangedVsFull =
			exactStats.adaptiveTopKChangedVsFull;
		strlcpy(stats->multivectorExactKernel,
				(maxsimPairs > 0 || exactRerankCount > 0) ?
				TqMultiVectorMaxSimKernelName() : "",
				sizeof(stats->multivectorExactKernel));
		strlcpy(stats->multivectorAccumulatorKind, docAccumulatorKind,
				sizeof(stats->multivectorAccumulatorKind));
		stats->multivectorMemoryBytesEstimate =
			(uint64) sizeof(TqDenseCandidate) * (uint64) candidateLimit +
			compactBytes +
			(docStorageCacheMode ==
			 PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_RESIDENT ?
			 (uint64) storage.multivectorDocMapBytes : 0);
		stats->multivectorAdmissionDebugEnabled =
			pgturbohybrid_multivector_debug_admission !=
			PGTURBOHYBRID_MULTIVECTOR_DEBUG_ADMISSION_OFF;
		stats->multivectorAdmissionCandidatesBeforeRerank =
			(uint32) Min(docsScored, (uint64) PG_UINT32_MAX);
		stats->multivectorAdmissionCandidatesAfterTruncation =
			(uint32) docCount;
		stats->multivectorAdmissionExactRerankDocs =
			(uint32) exactRerankCount;
		stats->multivectorAdmissionTruncatedByDocCandidateK =
			docsScored > (uint64) docCount;
		stats->multivectorAdmissionTruncatedByAccumulatorMemory = false;
		stats->multivectorAdmissionTraceAvailable = false;
		stats->multivectorAdmissionTraceCount = 0;
		stats->multivectorTokenStatsAvailable = false;
		stats->multivectorTokenStatsCount = 0;
	}

	so->tqGraphResults = NULL;
	so->tqGraphResultCount = docCount;
	so->tqGraphResultIndex = 0;
	so->graphCandidateCount = docCount;
	PgturbohybridGraphRecordGraphScanStats(so);
	*outCandidates = candidates;
	MemoryContextSwitchTo(oldCtx);
	return docCount;
}

int
PgturbohybridGraphCollectMultiVectorDenseCandidates(IndexScanDesc scan,
													PgturbohybridQueryHeader *query,
													int targetK,
													TqDenseCandidate **outCandidates,
													MemoryContext resultCtx,
													TqDenseCandidateStats *stats)
{
	PgturbohybridGraphScanOpaque so = (PgturbohybridGraphScanOpaque) scan->opaque;
	PgturbohybridGraphMetaPageData meta;
	PgturbohybridGraphScanStorage storage;
	PgturbohybridGraphCacheInitInfo cacheInfo;
	PgturbohybridMultiVector *mv;
	HTAB	   *volatile docHash = NULL;
	HTAB	   *volatile docIdHash = NULL;
	HASHCTL		hashCtl;
	HASHCTL		docIdHashCtl;
	HASH_SEQ_STATUS seq;
	PgturbohybridMultiVectorDocEntry *entry;
	TqDenseCandidate *candidates;
	PgturbohybridMultiVectorAccumulatorArena accumulatorArena;
	Vector	   *token;
	MemoryContext oldCtx;
	MemoryContext tokenCtx;
	instr_time	lockStart;
	instr_time	phaseStart;
	volatile int rawTarget;
	int			searchEf;
	int			initialRawTarget;
	int			maxRawTarget;
	int			maxSearchEf = 0;
	int			docLimit;
	int			docCount = 0;
	int			exactRerankCount = 0;
	uint64		multivectorDocCapacity = 0;
	volatile Size multivectorMemoryEstimate = 0;
	volatile uint64 multivectorRawSubvectorHits = 0;
	volatile uint64 multivectorUniqueDocs = 0;
	volatile uint64 multivectorDuplicateDocHits = 0;
	volatile uint64 multivectorMaxsimUpdates = 0;
	uint64		multivectorExactPairs = 0;
	PgturbohybridMultiVectorExactRerankWorkStats exactStats;
	uint64		multivectorExactTokenScanNodesScored = 0;
	uint32		multivectorReservoirScoreDocs = 0;
	uint32		multivectorReservoirCoverageDocs = 0;
	uint32		multivectorReservoirMeanDocs = 0;
	volatile uint32 multivectorReservoirPerTokenDocs = 0;
	uint32		multivectorReservoirBm25Docs = 0;
	volatile uint32 multivectorReservoirUnionDocs = 0;
	uint32		multivectorReservoirDuplicates = 0;
	uint64		nextDocOrdinal = 0;
	uint32		admissionCandidatesBeforeRerank = 0;
	volatile uint32 admissionTraceCount = 0;
	volatile uint32 skippedQueryTokens = 0;
	volatile uint32 tokenStatsCount = 0;
	int			exactRerankLimit = 0;
	bool		adaptiveWidening =
		pgturbohybrid_multivector_adaptive_widening !=
		PGTURBOHYBRID_MULTIVECTOR_ADAPTIVE_WIDENING_OFF;
	bool		admissionDebugEnabled =
		pgturbohybrid_multivector_debug_admission !=
		PGTURBOHYBRID_MULTIVECTOR_DEBUG_ADMISSION_OFF;
	bool		admissionTraceEnabled =
		pgturbohybrid_multivector_debug_admission ==
		PGTURBOHYBRID_MULTIVECTOR_DEBUG_ADMISSION_TRACE;
	volatile bool exactTokenScan =
		pgturbohybrid_multivector_candidate_source ==
		PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_EXACT_TOKEN_SCAN;
	bool		exactDocScan =
		pgturbohybrid_multivector_candidate_source ==
		PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_EXACT_DOC_SCAN;
	bool		docGraphPrototype =
		pgturbohybrid_multivector_candidate_source ==
		PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_DOC_GRAPH_PROTOTYPE;
	bool		documentNodesSource =
		pgturbohybrid_multivector_candidate_source ==
		PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_DOCUMENT_NODES;
	bool		proxyVector =
		pgturbohybrid_multivector_candidate_source ==
		PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_PROXY_VECTOR;
	bool		centroidLite =
		pgturbohybrid_multivector_candidate_source ==
		PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_CENTROID_LITE;
	bool		quantizedInvertedExperimental =
		pgturbohybrid_multivector_candidate_source ==
		PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_QUANTIZED_INVERTED_EXPERIMENTAL;
	bool		quantizedInvertedCompactScoring =
		quantizedInvertedExperimental &&
		pgturbohybrid_multivector_quantized_inverted_compact_scoring ==
		PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_COMPACT_SCORING_EXPERIMENTAL;
	int			centroidMode =
		PgturbohybridGraphGetMultiVectorCentroidsOption(scan->indexRelation);
	int			indexDocStorage =
		PgturbohybridGraphGetMultiVectorDocStorageOption(scan->indexRelation);
	bool		centroidOnlyIndex = false;
	bool		reservoirsEnabled =
		pgturbohybrid_multivector_candidate_reservoirs !=
		PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_RESERVOIRS_OFF;
	bool		adaptiveWideningTriggered = false;
	uint32		adaptiveFinalRawTarget = 0;
	bool		useDocMapSidecar = false;
	int			docMapSource =
		PGTURBOHYBRID_MULTIVECTOR_DOCMAP_SOURCE_NONE;
	uint64		docMapBytes = 0;
	bool		emptyIndex = false;
	bool		plainFallback = false;
	bool		documentNodeGraph = false;
	bool		proxyOnlyIndex = false;
	char		plainFallbackReason[48] = "not_applicable";
	bool	   *skipQueryToken = NULL;
	const float4 *queryWeights = NULL;
	const bool *queryMask = NULL;
	double		queryWeightSum = 0.0;
	PgturbohybridMultiVectorTokenStatsEntry *tokenStats = NULL;

	exactTokenScan = exactTokenScan || centroidLite;

	if (so == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("pgturbohybrid multivector candidate collection requires an active scan")));
	mv = PgturbohybridQueryGetMultiVector(query);
	if (mv == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("multivector_query is required for multivector dense scan")));
	PgturbohybridMultiVectorCheckDim((uint32) mv->dim,
									 (uint32) pgturbohybrid_multivector_max_dim);
	PgturbohybridMultiVectorCheckTokenCount((uint32) mv->count,
											(uint32) pgturbohybrid_multivector_max_query_vectors);
	queryWeights = PgturbohybridQueryGetTokenWeights(query);
	queryMask = PgturbohybridQueryGetTokenMask(query);
	queryWeightSum = PgturbohybridQueryMultiVectorWeightSum(query);

	oldCtx = MemoryContextSwitchTo(resultCtx);
	*outCandidates = NULL;
	memset(&exactStats, 0, sizeof(exactStats));
	if (stats != NULL)
		memset(stats, 0, sizeof(*stats));
	skipQueryToken = palloc0(sizeof(bool) * (Size) mv->count);
	{
		uint32		skippedQueryTokensLocal = 0;

		PgturbohybridMultiVectorParseSkipQueryTokens(
			pgturbohybrid_multivector_debug_skip_query_tokens,
			mv->count,
			skipQueryToken,
			&skippedQueryTokensLocal);
		skippedQueryTokens = skippedQueryTokensLocal;
	}
	if (queryMask != NULL)
	{
		for (int qi = 0; qi < mv->count; qi++)
		{
			if (queryMask[qi] && !skipQueryToken[qi])
			{
				skipQueryToken[qi] = true;
				skippedQueryTokens++;
			}
		}
	}
	if (admissionDebugEnabled || skippedQueryTokens > 0)
	{
		tokenStatsCount =
			Min((uint32) mv->count,
				(uint32) PGTURBOHYBRID_MULTIVECTOR_TOKEN_STATS_LIMIT_MAX);
		tokenStats =
			palloc0(sizeof(PgturbohybridMultiVectorTokenStatsEntry) *
					(Size) tokenStatsCount);
		for (uint32 i = 0; i < tokenStatsCount; i++)
		{
			tokenStats[i].queryTokenOrdinal = i;
			tokenStats[i].topHitSimilarity = 0.0;
			tokenStats[i].skipped = skipQueryToken[i];
		}
	}

	INSTR_TIME_SET_CURRENT(lockStart);
	LockPage(scan->indexRelation, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ShareLock);
	so->graphScanLockWaitUs += PgturbohybridGraphElapsedUs(lockStart);
	PG_TRY();
	{
		if (!PgturbohybridGraphReadMeta(scan->indexRelation, &meta) ||
			meta.tqNodeCount == 0 ||
			!BlockNumberIsValid(meta.tqCodeStartBlkno) ||
			!BlockNumberIsValid(meta.tqAdjStartBlkno))
		{
			emptyIndex = true;
		}
		else if (meta.dimensions != mv->dim)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("multivector query dimensions %d do not match index dimensions %d",
							mv->dim, meta.dimensions)));
		plainFallback =
			PgturbohybridMultiVectorShouldUsePlainFallback(scan,
														   so,
														   emptyIndex ? NULL : &meta,
														   targetK,
														   plainFallbackReason,
														   sizeof(plainFallbackReason));
		proxyOnlyIndex =
			!emptyIndex &&
			(indexDocStorage == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_PROXY_ONLY ||
			 (meta.tqMultivectorDocMapFlags &
			  PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_PROXY_ONLY) != 0);
		centroidOnlyIndex =
			!emptyIndex &&
			(indexDocStorage ==
			 PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CENTROID_ONLY ||
			 (((meta.tqMultivectorDocMapFlags &
				PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_CENTROIDS) != 0) &&
			  ((meta.tqMultivectorDocMapFlags &
				PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_DOC_VECTORS) == 0) &&
			  !proxyOnlyIndex));
		if (proxyOnlyIndex &&
			(exactDocScan || documentNodesSource || centroidLite ||
			 quantizedInvertedExperimental || docGraphPrototype))
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("multivector_doc_storage = proxy_only only supports proxy_vector document-node graph admission"),
					 errhint("REINDEX with multivector_doc_storage = f32, f16, or sq8 to use full document multivector sidecar candidate sources.")));
		if (centroidOnlyIndex &&
			(exactDocScan || documentNodesSource ||
			 (quantizedInvertedExperimental && !quantizedInvertedCompactScoring) ||
			 docGraphPrototype))
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("multivector_doc_storage = centroid_only only supports proxy_vector, centroid_lite, and compact quantized_inverted_experimental document-node graph admission"),
					 errhint("REINDEX with multivector_doc_storage = f32, f16, or sq8 to use full document multivector sidecar candidate sources.")));
		if (exactDocScan)
		{
			plainFallback = true;
			strlcpy(plainFallbackReason, "exact_doc_scan",
					sizeof(plainFallbackReason));
		}
		else if (docGraphPrototype)
		{
			plainFallback = true;
			strlcpy(plainFallbackReason, "doc_graph_prototype_heap_scan",
					sizeof(plainFallbackReason));
		}
		else if (quantizedInvertedExperimental)
		{
			plainFallback = false;
			strlcpy(plainFallbackReason, "not_applicable",
					sizeof(plainFallbackReason));
		}
		else if (centroidLite)
		{
			plainFallback = false;
			strlcpy(plainFallbackReason, "not_applicable",
					sizeof(plainFallbackReason));
		}
		documentNodeGraph =
			!emptyIndex &&
			!plainFallback &&
			meta.tqMultivectorGraphMode ==
			PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES;
		if (quantizedInvertedExperimental && !emptyIndex &&
			meta.tqMultivectorGraphMode !=
			PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("quantized_inverted_experimental multivector candidate source requires multivector_graph = document_nodes"),
					 errhint("REINDEX with multivector_graph = document_nodes; the research branch needs document sidecar vectors for exact MaxSim rerank.")));
		if (centroidLite && !emptyIndex &&
			centroidMode != PGTURBOHYBRID_MULTIVECTOR_CENTROIDS_KMEANS)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("centroid_lite multivector candidate source requires multivector_centroids = kmeans"),
					 errhint("REINDEX with multivector_centroids = kmeans, or use another turbohybrid.multivector_candidate_source.")));
		if (proxyVector && !emptyIndex &&
			meta.tqMultivectorGraphMode !=
			PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("proxy_vector multivector candidate source requires multivector_graph = document_nodes"),
					 errhint("REINDEX with multivector_graph = document_nodes, or use turbohybrid.multivector_candidate_source = graph for token-node indexes.")));
		if (documentNodesSource && !emptyIndex && !plainFallback &&
			meta.tqMultivectorGraphMode !=
			PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("document_nodes multivector candidate source requires multivector_graph = document_nodes"),
					 errhint("REINDEX with multivector_graph = document_nodes, or use turbohybrid.multivector_candidate_source = graph for token-node indexes.")));
		if (!emptyIndex && !plainFallback && !documentNodeGraph)
		{
			so->graphM = meta.m;
			so->graphEfConstruction = meta.efConstruction;
			so->graphExactStorage =
				((meta.tqFlags & PGTURBOHYBRID_GRAPH_EXACT_FREE) == 0 &&
				 BlockNumberIsValid(meta.tqExactStartBlkno));
			so->graphBuildExactDistances =
				(meta.tqFlags & PGTURBOHYBRID_GRAPH_TQ_EXACT_BUILD) != 0;
			so->graphBuildDistanceMode = so->graphBuildExactDistances ?
				PGTURBOHYBRID_DENSE_BUILD_DISTANCE_EXACT :
				PGTURBOHYBRID_DENSE_BUILD_DISTANCE_CODE;
			so->graphBuildFastEdges =
				(meta.tqFlags & PGTURBOHYBRID_GRAPH_TQ_FAST_BUILD_EDGES) != 0;
			so->graphBuildNeighborSelectReason =
				PGTURBOHYBRID_GRAPH_TQ_BUILD_NEIGHBOR_REASON(meta.tqFlags);

			if (adaptiveWidening)
			{
				initialRawTarget = pgturbohybrid_multivector_subvector_k;
				maxRawTarget =
					Max(initialRawTarget,
						pgturbohybrid_multivector_unique_docs_per_token);
				maxRawTarget = Max(maxRawTarget, targetK > 0 ? targetK : 1);
			}
			else
			{
				initialRawTarget =
					Max(pgturbohybrid_multivector_subvector_k,
						pgturbohybrid_multivector_unique_docs_per_token);
				initialRawTarget =
					Max(initialRawTarget, targetK > 0 ? targetK : 1);
				maxRawTarget = initialRawTarget;
			}
			initialRawTarget =
				Min(initialRawTarget,
					pgturbohybrid_multivector_max_raw_hits_per_token);
			initialRawTarget = Min(initialRawTarget, (int) meta.tqNodeCount);
			initialRawTarget = Max(initialRawTarget, 1);
			maxRawTarget =
				Min(maxRawTarget,
					pgturbohybrid_multivector_max_raw_hits_per_token);
			maxRawTarget = Min(maxRawTarget, (int) meta.tqNodeCount);
			maxRawTarget = Max(maxRawTarget, initialRawTarget);
			rawTarget = initialRawTarget;
			searchEf = Min(Max(so->efSearch, rawTarget), (int) meta.tqNodeCount);
			searchEf = PgturbohybridGraphScaleSearchEfForSegments(so, &meta,
																  searchEf);
			maxSearchEf = searchEf;
			so->graphDenseRequestedK = targetK;
			so->graphEffectiveResultTarget = rawTarget;
			so->graphEffectiveSearchEf = searchEf;
			so->graphEffectiveRescoreBand = 0;

			multivectorDocCapacity =
				PgturbohybridMultiVectorDocCapacity(maxRawTarget, mv->count);
			multivectorMemoryEstimate =
				PgturbohybridMultiVectorAccumulatorBytesEstimate(multivectorDocCapacity,
																 mv->count);
			PgturbohybridMultiVectorCheckAccumulatorMemory(multivectorDocCapacity,
														   mv->count);
			memset(&hashCtl, 0, sizeof(hashCtl));
			hashCtl.keysize = sizeof(PgturbohybridMultiVectorDocKey);
			hashCtl.entrysize = sizeof(PgturbohybridMultiVectorDocEntry);
			hashCtl.hcxt = resultCtx;
			docHash = hash_create("pgturbohybrid multivector doc accumulator",
								  Max((long) multivectorDocCapacity, 16L),
								  &hashCtl,
								  HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
			PgturbohybridMultiVectorAccumulatorArenaInit(&accumulatorArena,
														 resultCtx, mv->count,
														 multivectorDocCapacity);

			INSTR_TIME_SET_CURRENT(phaseStart);
			PgturbohybridGraphInitScanStorage(scan->indexRelation, &meta,
											  &storage, &cacheInfo);
			so->graphNativeCacheMode = cacheInfo.mode;
			so->graphNativeCachePolicy = cacheInfo.policy;
			so->graphNativeCacheReason = cacheInfo.reason;
			so->graphNativeCacheUsed = cacheInfo.used;
			so->graphNativeCacheReused = cacheInfo.reused;
			so->graphNativeCacheBuiltThisScan = cacheInfo.builtThisScan;
			so->graphNativeCacheAttachUs = cacheInfo.attachUs;
			so->graphNativeCacheBuildUs = cacheInfo.buildUs;
			so->graphNativeCacheWaitUs = cacheInfo.waitUs;
			so->graphNativeCacheRefcount = cacheInfo.refcount;
			so->graphNativeCacheBytes = cacheInfo.totalBytes;
			so->graphNativeCacheCodeBytes = cacheInfo.codeBytes;
			so->graphNativeCacheAdjBytes = cacheInfo.adjBytes;
			so->graphNativeCacheExactBytes = cacheInfo.exactBytes;
			so->graphNativeCacheWarning = cacheInfo.warning;
			so->graphNativeCacheWarningReason = cacheInfo.warningReason;
			so->graphCodeBufferLockWaitUs += cacheInfo.codeBufferLockWaitUs;
			so->graphAdjBufferLockWaitUs += cacheInfo.adjBufferLockWaitUs;
			PgturbohybridGraphAddElapsedUs(&so->graphPrepareUs, phaseStart);
			if (pgturbohybrid_multivector_docmap !=
				PGTURBOHYBRID_MULTIVECTOR_DOCMAP_OFF)
				useDocMapSidecar =
					PgturbohybridGraphLoadMultiVectorDocMap(scan->indexRelation,
															&meta,
															&storage,
															pgturbohybrid_multivector_docmap ==
															PGTURBOHYBRID_MULTIVECTOR_DOCMAP_REQUIRE);
			if (useDocMapSidecar)
			{
				docMapSource =
					PGTURBOHYBRID_MULTIVECTOR_DOCMAP_SOURCE_SIDECAR;
				docMapBytes = storage.multivectorDocMapBytes;
			}
			else
			{
				docMapSource =
					PGTURBOHYBRID_MULTIVECTOR_DOCMAP_SOURCE_HEAP_TID_HASH;
				memset(&docIdHashCtl, 0, sizeof(docIdHashCtl));
				docIdHashCtl.keysize = sizeof(PgturbohybridMultiVectorTidKey);
				docIdHashCtl.entrysize =
					sizeof(PgturbohybridMultiVectorDocIdEntry);
				docIdHashCtl.hcxt = resultCtx;
				docIdHash =
					hash_create("pgturbohybrid multivector scan doc ids",
								Max((long) multivectorDocCapacity, 16L),
								&docIdHashCtl,
								HASH_ELEM | HASH_BLOBS | HASH_CONTEXT);
			}

			tokenCtx = AllocSetContextCreate(resultCtx,
											 "pgturbohybrid multivector token scan",
											 ALLOCSET_DEFAULT_SIZES);
			token = MemoryContextAlloc(resultCtx,
									   PgturbohybridMultiVectorSubvectorSize(mv));
			for (int qi = 0; qi < mv->count; qi++)
			{
				PgturbohybridGraphResult *hits;
				HTAB	   *countSeen;
				HTAB	   *tokenSeen;
				int			hitCount;
				int			tokenRawTarget = initialRawTarget;
				int			tokenUniqueDocs = 0;
				PgturbohybridMultiVectorTokenStatsEntry *tokenStat =
					(qi < (int) tokenStatsCount) ? &tokenStats[qi] : NULL;

				CHECK_FOR_INTERRUPTS();
				if (skipQueryToken[qi])
				{
					if (tokenStat != NULL)
						tokenStat->skipped = true;
					continue;
				}
				if (exactTokenScan)
				{
					tokenRawTarget = maxRawTarget;
					MemoryContextSwitchTo(tokenCtx);
					PgturbohybridMultiVectorCopySubvectorToVector(mv, qi,
																  token);
					PgturbohybridGraphPrepareTqQueryWithBits(scan->indexRelation,
															 &so->support,
															 PointerGetDatum(token),
															 &so->tq,
															 meta.tqBits != 0 ?
															 meta.tqBits :
															 PGTURBOHYBRID_DEFAULT_BITS);
					hits = palloc(sizeof(PgturbohybridGraphResult) *
								  tokenRawTarget);
					hitCount =
						PgturbohybridMultiVectorExactTokenScan(scan->indexRelation,
															  so, &meta,
															  &storage,
															  PointerGetDatum(token),
															  tokenRawTarget,
															  hits,
															  &multivectorExactTokenScanNodesScored);
					MemoryContextSwitchTo(resultCtx);
					maxSearchEf = Max(maxSearchEf, (int) meta.tqNodeCount);
				}
				else
				{
					for (;;)
					{
						int			currentSearchEf;

						tokenUniqueDocs = 0;
						PgturbohybridMultiVectorTokenSeenReset(&countSeen,
															   tokenCtx,
															   tokenRawTarget);
						MemoryContextSwitchTo(tokenCtx);
						PgturbohybridMultiVectorCopySubvectorToVector(mv, qi,
																	  token);
						PgturbohybridGraphPrepareTqQueryWithBits(scan->indexRelation,
																 &so->support,
																 PointerGetDatum(token),
																 &so->tq,
																 meta.tqBits != 0 ?
																 meta.tqBits :
																 PGTURBOHYBRID_DEFAULT_BITS);
						currentSearchEf =
							Min(Max(so->efSearch, tokenRawTarget),
								(int) meta.tqNodeCount);
						currentSearchEf =
							PgturbohybridGraphScaleSearchEfForSegments(so, &meta,
																	  currentSearchEf);
						maxSearchEf = Max(maxSearchEf, currentSearchEf);
						hits = palloc(sizeof(PgturbohybridGraphResult) *
									  tokenRawTarget);
						hitCount = PgturbohybridGraphRunTraversalPass(scan, so,
																	  &meta,
																	  &storage,
																	  hits,
																	  tokenRawTarget,
																	  currentSearchEf,
																	  PointerGetDatum(token),
																	  -1, 0, false,
																	  false, 1.0,
																	  PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_NONE);
						MemoryContextSwitchTo(resultCtx);
						for (int i = 0; i < hitCount; i++)
						{
							TqDocId		docId;

							if (useDocMapSidecar)
							{
								if (hits[i].nodeId >= meta.tqNodeCount)
									elog(ERROR, "pgturbohybrid multivector hit node id is out of range");
								docId =
									storage.multivectorNodeMap[hits[i].nodeId].docId;
							}
							else
								docId =
									PgturbohybridMultiVectorResolveDocId(docIdHash,
																		 &hits[i].heaptid,
																		 &nextDocOrdinal);
							if (PgturbohybridMultiVectorTokenSeenAdd(countSeen,
																	 docId))
								tokenUniqueDocs++;
						}
						if (!adaptiveWidening ||
							tokenUniqueDocs >=
							pgturbohybrid_multivector_unique_docs_per_token ||
							tokenRawTarget >= maxRawTarget)
							break;

						adaptiveWideningTriggered = true;
						tokenRawTarget =
							Min(maxRawTarget,
								Max(tokenRawTarget + 1, tokenRawTarget * 2));
					}
				}
				adaptiveFinalRawTarget =
					Max(adaptiveFinalRawTarget, (uint32) tokenRawTarget);
				if (tokenStat != NULL)
				{
					tokenStat->rawHits = (uint32) Max(hitCount, 0);
					for (int i = 0; i < hitCount; i++)
					{
						double		similarity = -hits[i].distance;

						if (!tokenStat->topHitSimilarityAvailable ||
							similarity > tokenStat->topHitSimilarity)
						{
							tokenStat->topHitSimilarityAvailable = true;
							tokenStat->topHitSimilarity = similarity;
						}
					}
				}
				tokenSeen =
					PgturbohybridMultiVectorTokenSeenCreate(tokenCtx,
															tokenRawTarget);
				tokenUniqueDocs = 0;
				for (int i = 0; i < hitCount; i++)
				{
					TqDocId		docId;
					bool		tokenUniqueDoc;
					PgturbohybridGraphResult docHit = hits[i];
					uint64		maxsimUpdates = multivectorMaxsimUpdates;

					multivectorRawSubvectorHits++;
					if (useDocMapSidecar)
					{
						TqMultiVectorDocMapEntry *docEntry;

						if (hits[i].nodeId >= meta.tqNodeCount)
							elog(ERROR, "pgturbohybrid multivector hit node id is out of range");
						docId =
							storage.multivectorNodeMap[hits[i].nodeId].docId;
						if (docId >= storage.multivectorDocCount)
							elog(ERROR, "pgturbohybrid multivector doc id is out of range");
						docEntry = &storage.multivectorDocMap[docId];
						docHit.heaptid = docEntry->heapTid;
					}
					else
						docId =
							PgturbohybridMultiVectorResolveDocId(docIdHash,
																 &hits[i].heaptid,
																 &nextDocOrdinal);
					tokenUniqueDoc =
						PgturbohybridMultiVectorTokenSeenAdd(tokenSeen, docId);
					if (!tokenUniqueDoc)
					{
						multivectorDuplicateDocHits++;
						if (tokenStat != NULL)
							tokenStat->duplicateDocHits++;
					}
					PgturbohybridMultiVectorAccumulateDoc(docHash,
														  &accumulatorArena,
														  &docHit, docId,
														  qi, mv->count,
														  queryWeights != NULL ?
														  (double) queryWeights[qi] : 1.0,
														  !tokenUniqueDoc,
														  &maxsimUpdates);
					multivectorMaxsimUpdates = maxsimUpdates;
					if (tokenUniqueDoc)
					{
						tokenUniqueDocs++;
						multivectorUniqueDocs++;
						if (tokenStat != NULL)
							tokenStat->uniqueDocs++;
						if (tokenUniqueDocs >=
							pgturbohybrid_multivector_unique_docs_per_token)
							break;
					}
				}
			}
			so->graphEffectiveResultTarget = (int) adaptiveFinalRawTarget;
			so->graphEffectiveSearchEf = maxSearchEf;
			MemoryContextDelete(tokenCtx);
		}
	}
	PG_CATCH();
	{
		UnlockPage(scan->indexRelation, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ShareLock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	UnlockPage(scan->indexRelation, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ShareLock);

	if (plainFallback)
	{
		int			fallbackCount;

		fallbackCount =
			PgturbohybridMultiVectorExactPlainFallback(scan, so, mv,
													   queryWeights,
													   queryMask,
													   queryWeightSum,
													   targetK,
													   outCandidates, resultCtx,
													   stats,
													   plainFallbackReason,
													   exactDocScan ? "exact_doc_scan" :
													   docGraphPrototype ? "doc_graph_prototype" :
													   NULL,
													   docGraphPrototype,
													   docGraphPrototype ?
													   "prototype_heap_scan_no_index_resident_doc_graph" :
													   NULL);
		so->tqGraphResults = NULL;
		so->tqGraphResultCount = fallbackCount;
		so->tqGraphResultIndex = 0;
		so->graphCandidateCount = fallbackCount;
		PgturbohybridGraphRecordGraphScanStats(so);
		MemoryContextSwitchTo(oldCtx);
		return fallbackCount;
	}

	if (emptyIndex)
	{
		*outCandidates = palloc0(sizeof(TqDenseCandidate));
		MemoryContextSwitchTo(oldCtx);
		return 0;
	}

	if (documentNodeGraph)
	{
		int			documentNodeCount;

		documentNodeCount =
			PgturbohybridMultiVectorDocumentNodeScan(scan, so, &meta, mv,
													 queryWeights,
													 queryMask,
													 queryWeightSum,
													 targetK, outCandidates,
													 resultCtx, stats);
		MemoryContextSwitchTo(oldCtx);
		return documentNodeCount;
	}

	admissionCandidatesBeforeRerank = (uint32) hash_get_num_entries(docHash);
	docLimit = PgturbohybridMultiVectorDocCandidateLimit(targetK);
	candidates = palloc0(sizeof(TqDenseCandidate) * docLimit);
	if (reservoirsEnabled)
	{
		uint32		reservoirPerTokenDocs = 0;
		uint32		reservoirUnionDocs = 0;

		PgturbohybridMultiVectorBuildReservoirCandidates(docHash, mv->count,
														 queryWeightSum,
														 docLimit,
														 candidates,
														 &docCount,
														 resultCtx,
														 &multivectorReservoirScoreDocs,
														 &multivectorReservoirCoverageDocs,
														 &multivectorReservoirMeanDocs,
														 &reservoirPerTokenDocs,
														 &multivectorReservoirBm25Docs,
														 &reservoirUnionDocs,
														 &multivectorReservoirDuplicates);
		multivectorReservoirPerTokenDocs = reservoirPerTokenDocs;
		multivectorReservoirUnionDocs = reservoirUnionDocs;
	}
	else
	{
		hash_seq_init(&seq, docHash);
		while ((entry = (PgturbohybridMultiVectorDocEntry *) hash_seq_search(&seq)) != NULL)
		{
			TqDenseCandidate candidate;

			CHECK_FOR_INTERRUPTS();
			PgturbohybridMultiVectorCandidateFromDoc(entry, mv->count,
													 queryWeightSum,
													 &candidate);

			PgturbohybridMultiVectorCandidateHeapOffer(candidates, &docCount,
													  docLimit, &candidate);
		}
	}
	PgturbohybridMultiVectorCandidateHeapSort(candidates, docCount);
	if (tokenStats != NULL)
		PgturbohybridMultiVectorUpdateTokenCandidateStats(docHash,
														  candidates,
														  docCount,
														  mv->count,
														  tokenStats,
														  tokenStatsCount);
	exactRerankLimit = PgturbohybridMultiVectorExactRerankLimit(docCount);
	exactRerankCount =
		PgturbohybridMultiVectorExactHeapRerank(scan, so,
												!centroidLite &&
												useDocMapSidecar &&
												storage.multivectorDocVectorsLoaded ?
												&meta : NULL,
												!centroidLite &&
												useDocMapSidecar &&
												storage.multivectorDocVectorsLoaded ?
												&storage : NULL,
												NULL,
												mv,
												queryWeights,
												queryMask,
												queryWeightSum,
												candidates,
												docCount,
												-1,
												&multivectorExactPairs,
												&exactStats);
	if (admissionTraceEnabled)
		admissionTraceCount =
			PgturbohybridMultiVectorBuildAdmissionTrace(docHash,
														candidates,
														docCount,
														exactRerankLimit,
														stats != NULL ?
														stats->multivectorAdmissionTrace :
														NULL,
														(uint32)
														pgturbohybrid_multivector_debug_trace_limit);
	if (exactRerankCount > 0 && docCount > 1)
		PgturbohybridMultiVectorCandidateHeapSort(candidates, docCount);
	for (int i = 0; i < docCount; i++)
		candidates[i].rank = i + 1;

	if (stats != NULL)
	{
		stats->visitedGraphNodes = so->graphVisitedNodes;
		stats->scoredCodes = so->graphScoredCodes;
		stats->denseCandidatesRequested = targetK > 0 ? targetK : docLimit;
		stats->effectiveResultTarget = (uint32) Max(so->graphEffectiveResultTarget, 0);
		stats->effectiveSearchEf = (uint32) Max(so->graphEffectiveSearchEf, 0);
		stats->effectiveRescoreBand = (uint32) Max(so->graphEffectiveRescoreBand, 0);
		stats->denseCandidatesReturned = docCount;
		stats->heapRescoreCount = so->graphHeapRescoreCount;
		stats->codePagesRead = so->graphCodePagesRead;
		stats->adjPagesRead = so->graphAdjPagesRead;
		stats->prepareUs = so->graphPrepareUs;
		stats->traverseUs = so->graphTraverseUs;
		stats->entryUs = so->graphEntryUs;
		stats->baseUs = so->graphBaseUs;
		stats->batchUs = so->graphBatchUs;
		stats->heapFetchUs = so->graphHeapFetchUs;
		stats->heapRescoreUs = so->graphHeapRescoreUs;
		stats->sortUs = so->graphSortUs;
		stats->exactRescoreSource = so->graphExactRescoreSource;
		stats->multivectorEnabled = true;
		stats->multivectorQueryVectors = (uint32) mv->count;
		stats->multivectorDocVectorsLimit =
			(uint32) pgturbohybrid_multivector_max_doc_vectors;
		stats->multivectorSubvectorSearches =
			(uint64) ((uint32) mv->count - skippedQueryTokens);
		stats->multivectorRawSubvectorHits = multivectorRawSubvectorHits;
		stats->multivectorAdaptiveWideningTriggered =
			adaptiveWideningTriggered;
		stats->multivectorAdaptiveInitialRawTarget =
			(uint32) initialRawTarget;
		stats->multivectorAdaptiveFinalRawTarget = adaptiveFinalRawTarget;
		stats->multivectorDocMapSource = docMapSource;
		strlcpy(stats->multivectorCandidateSource,
				centroidLite ? "centroid_lite" :
				exactTokenScan ? "exact_token_scan" : "graph",
				sizeof(stats->multivectorCandidateSource));
		strlcpy(stats->multivectorGraphMode,
				PgturbohybridMultiVectorGraphModeName(meta.tqMultivectorGraphMode),
				sizeof(stats->multivectorGraphMode));
		stats->multivectorExactTokenScanEnabled = exactTokenScan;
		stats->multivectorExactTokenScanNodesScored =
			multivectorExactTokenScanNodesScored;
		stats->multivectorPlainFallbackUsed = false;
		strlcpy(stats->multivectorPlainFallbackReason, "not_applicable",
				sizeof(stats->multivectorPlainFallbackReason));
		stats->multivectorPlainFallbackDocsScored = 0;
		stats->multivectorPlainFallbackPairs = 0;
		stats->multivectorDocGraphPrototypeEnabled = false;
		stats->multivectorDocGraphDocsScored = 0;
		stats->multivectorDocGraphEdgesVisited = 0;
		stats->multivectorDocGraphCandidates = 0;
		stats->multivectorDocGraphNodes = 0;
		stats->multivectorDocGraphSearchEf = 0;
		stats->multivectorDocGraphOversampling = 0;
		stats->multivectorDocGraphRescoreK = 0;
		stats->multivectorDocGraphQuantizedScores = 0;
		stats->multivectorDocGraphExactRerankDocs = 0;
		stats->multivectorDocGraphHeapFetches = 0;
		strlcpy(stats->multivectorDocGraphWarning,
				centroidLite ?
				"token_node_centroid_lite_exact_token_prefilter" :
				"not_applicable",
				sizeof(stats->multivectorDocGraphWarning));
		if (centroidLite)
		{
			stats->centroid.listsVisited =
				(uint64) ((uint32) mv->count - skippedQueryTokens);
			stats->centroid.docsTouched = admissionCandidatesBeforeRerank;
			stats->centroid.candidates = (uint32) docCount;
			strlcpy(stats->centroid.candidateScoring, "token_node_exact",
					sizeof(stats->centroid.candidateScoring));
			stats->centroid.prunedDocs =
				admissionCandidatesBeforeRerank > (uint32) docCount ?
				(uint64) (admissionCandidatesBeforeRerank - (uint32) docCount) :
				0;
		}
		stats->multivectorReservoirsEnabled = reservoirsEnabled;
		stats->multivectorReservoirScoreDocs =
			multivectorReservoirScoreDocs;
		stats->multivectorReservoirCoverageDocs =
			multivectorReservoirCoverageDocs;
		stats->multivectorReservoirMeanDocs = multivectorReservoirMeanDocs;
		stats->multivectorReservoirPerTokenDocs =
			multivectorReservoirPerTokenDocs;
		stats->multivectorReservoirBm25Docs = multivectorReservoirBm25Docs;
		stats->multivectorReservoirUnionDocs = multivectorReservoirUnionDocs;
		stats->multivectorReservoirDuplicates =
			multivectorReservoirDuplicates;
		stats->multivectorDocMapBytes = docMapBytes;
		stats->multivectorUniqueDocs = multivectorUniqueDocs;
		stats->multivectorDuplicateDocHits = multivectorDuplicateDocHits;
		stats->multivectorMaxsimUpdates = multivectorMaxsimUpdates;
		stats->multivectorDocCandidates = (uint32) docCount;
		stats->multivectorExactRerankEnabled =
			pgturbohybrid_multivector_exact_rerank !=
			PGTURBOHYBRID_MULTIVECTOR_EXACT_RERANK_OFF;
		stats->multivectorExactRerankDocs = (uint32) exactRerankCount;
		stats->multivectorExactRerankPairs = multivectorExactPairs;
		stats->multivectorExactRerankSource = exactStats.source;
		stats->multivectorExactRerankHeapFetches = exactStats.heapFetches;
		stats->multivectorExactRerankSidecarReads = exactStats.sidecarReads;
		stats->multivectorExactRerankSidecarBytes = exactStats.sidecarBytes;
		stats->exactRerankCandidates = exactStats.candidates;
		stats->exactRerankTokensEvaluated = exactStats.tokensEvaluated;
		stats->exactRerankTokensSkipped = exactStats.tokensSkipped;
		stats->exactRerankPairsSaved = exactStats.pairsSaved;
		stats->adaptiveRerankTopKChangedVsFull =
			exactStats.adaptiveTopKChangedVsFull;
		strlcpy(stats->multivectorExactKernel,
				exactRerankCount > 0 ? TqMultiVectorMaxSimKernelName() : "",
				sizeof(stats->multivectorExactKernel));
		strlcpy(stats->multivectorAccumulatorKind,
				centroidLite ? "centroid_lite_token_scan" : "docid_hash_slab",
				sizeof(stats->multivectorAccumulatorKind));
		stats->multivectorMemoryBytesEstimate =
			(uint64) multivectorMemoryEstimate;
		stats->multivectorAdmissionDebugEnabled = admissionDebugEnabled;
		stats->multivectorAdmissionCandidatesBeforeRerank =
			admissionCandidatesBeforeRerank;
		stats->multivectorAdmissionCandidatesAfterTruncation =
			(uint32) docCount;
		stats->multivectorAdmissionExactRerankDocs =
			(uint32) exactRerankCount;
		stats->multivectorAdmissionTruncatedByDocCandidateK =
			admissionCandidatesBeforeRerank > (uint32) docCount;
		stats->multivectorAdmissionTruncatedByAccumulatorMemory = false;
		stats->multivectorAdmissionTraceAvailable =
			admissionTraceEnabled && admissionTraceCount > 0;
		stats->multivectorAdmissionTraceCount = admissionTraceCount;
		stats->multivectorTokenStatsAvailable = tokenStats != NULL;
		stats->multivectorTokenStatsCount =
			tokenStats != NULL ? tokenStatsCount : 0;
		if (stats->multivectorTokenStatsCount > 0)
			memcpy(stats->multivectorTokenStats,
				   tokenStats,
				   sizeof(PgturbohybridMultiVectorTokenStatsEntry) *
				   stats->multivectorTokenStatsCount);
	}
	so->tqGraphResults = NULL;
	so->tqGraphResultCount = docCount;
	so->tqGraphResultIndex = 0;
	so->graphCandidateCount = docCount;
	PgturbohybridGraphRecordGraphScanStats(so);
	*outCandidates = candidates;
	MemoryContextSwitchTo(oldCtx);
	return docCount;
}
