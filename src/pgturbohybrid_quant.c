#include "postgres.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <string.h>

#include "access/generic_xlog.h"
#include "access/genam.h"
#include "access/tableam.h"
#include "access/xloginsert.h"
#include "catalog/index.h"
#include "catalog/pg_type_d.h"
#include "commands/progress.h"
#include "miscadmin.h"
#include "pgstat.h"
#include "port/pg_bitutils.h"
#include "portability/instr_time.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/array.h"
#include "utils/datum.h"
#include "utils/fmgrprotos.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/rel.h"


#if PG_VERSION_NUM >= 140000
#include "utils/backend_progress.h"
#endif


#include "pgturbohybrid_quant.h"
#include "pgturbohybrid_quant_psquare.h"
#include "pgturbohybrid_quant_score.h"

static int	PgturbohybridGraphResultCompare(const void *a, const void *b);
static bool PgturbohybridGraphEntryAlreadySelected(PgturbohybridGraphFrontierItem *entries, int entryCount,
										uint32 nodeId);


static bool
PgturbohybridGraphBuildVectorsEqual(PgturbohybridQuantBuildState *state, uint32 left, uint32 right)
{
	Vector	   *leftVector;
	Vector	   *rightVector;
	Size		vectorSize;

	if (state->nodes[left].vectorHash != state->nodes[right].vectorHash)
		return false;

	leftVector = state->nodes[left].vector;
	rightVector = state->nodes[right].vector;
	if (leftVector == rightVector)
		return true;
	if (leftVector == NULL || rightVector == NULL ||
		leftVector->dim != rightVector->dim)
		return false;

	vectorSize = PGTURBOHYBRID_VECTOR_SIZE(leftVector->dim);
	return memcmp(leftVector, rightVector, vectorSize) == 0;
}

static uint64
PgturbohybridGraphBuildVectorHash(Vector *vector)
{
	const unsigned char *bytes = (const unsigned char *) vector;
	Size		vectorSize = PGTURBOHYBRID_VECTOR_SIZE(vector->dim);
	uint64		hash = UINT64CONST(1469598103934665603);

	for (Size i = 0; i < vectorSize; i++)
	{
		hash ^= bytes[i];
		hash *= UINT64CONST(1099511628211);
	}
	return hash;
}


static int
PgturbohybridGraphIndexPayloadCount(Relation index)
{
	int			totalAttrs = IndexRelationGetNumberOfAttributes(index);
	int			keyAttrs = IndexRelationGetNumberOfKeyAttributes(index);
	int			payloadCount = totalAttrs - keyAttrs;

	if (payloadCount < 0)
		payloadCount = 0;
	if (payloadCount > PGTURBOHYBRID_GRAPH_MAX_PAYLOADS)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("pgturbohybrid native graph supports at most %d included payload columns",
						PGTURBOHYBRID_GRAPH_MAX_PAYLOADS)));

	for (int i = 0; i < payloadCount; i++)
	{
		Form_pg_attribute attr =
			TupleDescAttr(RelationGetDescr(index), keyAttrs + i);

		if (attr->atttypid != INT4OID)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("pgturbohybrid native graph INCLUDE payload columns must be integer")));
	}

	return payloadCount;
}

void
PgturbohybridGraphCopyPayloadValues(PgturbohybridQuantBuildState *state, int32 *payloads,
						 uint16 *payloadMask, Datum *values, bool *isnull)
{
	int			keyAttrs;

	*payloadMask = 0;
	if (state->payloadCount <= 0 || values == NULL || isnull == NULL)
		return;

	keyAttrs = state->indexInfo != NULL ? state->indexInfo->ii_NumIndexKeyAttrs :
		IndexRelationGetNumberOfKeyAttributes(state->index);

	for (int i = 0; i < state->payloadCount; i++)
	{
		int			attrIndex = keyAttrs + i;

		if (isnull[attrIndex])
			continue;

		payloads[i] = DatumGetInt32(values[attrIndex]);
		*payloadMask |= (uint16) (1U << i);
	}
}

static int
PgturbohybridGraphPayloadSlotForHeapAttr(Relation index, AttrNumber heapAttno)
{
	int			totalAttrs = IndexRelationGetNumberOfAttributes(index);
	int			keyAttrs = IndexRelationGetNumberOfKeyAttributes(index);

	for (int i = keyAttrs; i < totalAttrs; i++)
	{
		if (index->rd_index->indkey.values[i] == heapAttno)
			return i - keyAttrs;
	}

	return -1;
}

static bool
PgturbohybridGraphNodeMatchesPayload(PgturbohybridGraphScanNode *node, int payloadSlot, int32 payloadValue)
{
	if (payloadSlot < 0)
		return true;
	if (payloadSlot >= PGTURBOHYBRID_GRAPH_MAX_PAYLOADS || node->payloads == NULL)
		return false;
	if ((node->payloadMask & (uint16) (1U << payloadSlot)) == 0)
		return false;

	return node->payloads[payloadSlot] == payloadValue;
}


static uint64
PgturbohybridGraphMix64(uint64 x)
{
	x += UINT64CONST(0x9e3779b97f4a7c15);
	x = (x ^ (x >> 30)) * UINT64CONST(0xbf58476d1ce4e5b9);
	x = (x ^ (x >> 27)) * UINT64CONST(0x94d049bb133111eb);
	return x ^ (x >> 31);
}

int
PgturbohybridGraphPickLevel(uint32 nodeId, int m)
{
	uint64		mixed = PgturbohybridGraphMix64(nodeId);
	double		u = ((double) ((mixed >> 11) + 1)) * (1.0 / 9007199254740992.0);
	int			level = (int) floor(-log(u) * PgturbohybridGraphGetMl(Max(m, 2)));

	return Min(level, Min(PgturbohybridGraphGetMaxLevel(m), PGTURBOHYBRID_GRAPH_MAX_STORED_LEVEL));
}


static Size
PgturbohybridGraphCorrectionTupleSize(int count)
{
	return MAXALIGN(offsetof(PgturbohybridGraphCorrectionTupleData, values) +
					(sizeof(float) * count));
}

static int
PgturbohybridGraphCorrectionTupleMaxCount(void)
{
	Size		usable = BLCKSZ - MAXALIGN(SizeOfPageHeaderData) -
		MAXALIGN(sizeof(PgturbohybridGraphPageOpaqueData)) - sizeof(ItemIdData);
	int			count = (usable - offsetof(PgturbohybridGraphCorrectionTupleData, values)) /
		sizeof(float);

	while (count > 1 && PgturbohybridGraphCorrectionTupleSize(count) > usable)
		count--;

	return Max(1, count);
}

static bool
PgturbohybridGraphBuildNodeHasLevel(PgturbohybridQuantBuildState *state, uint32 nodeId, int level)
{
	return nodeId < state->nodeCount && level >= 0 && level <= state->nodes[nodeId].level;
}

static int
PgturbohybridGraphFrontierCompare(const void *a, const void *b)
{
	const PgturbohybridGraphFrontierItem *ia = (const PgturbohybridGraphFrontierItem *) a;
	const PgturbohybridGraphFrontierItem *ib = (const PgturbohybridGraphFrontierItem *) b;

	if (ia->distance < ib->distance)
		return -1;
	if (ia->distance > ib->distance)
		return 1;
	return (ia->nodeId > ib->nodeId) - (ia->nodeId < ib->nodeId);
}

static int
PgturbohybridGraphBuildOrderCompare(const void *a, const void *b)
{
	const PgturbohybridGraphBuildOrderItem *ia = (const PgturbohybridGraphBuildOrderItem *) a;
	const PgturbohybridGraphBuildOrderItem *ib = (const PgturbohybridGraphBuildOrderItem *) b;

	if (ia->key < ib->key)
		return -1;
	if (ia->key > ib->key)
		return 1;
	return (ia->nodeId > ib->nodeId) - (ia->nodeId < ib->nodeId);
}


static bool
PgturbohybridGraphFrontierLess(PgturbohybridGraphFrontierItem a, PgturbohybridGraphFrontierItem b)
{
	if (a.distance != b.distance)
		return a.distance < b.distance;
	return a.nodeId < b.nodeId;
}

static bool
PgturbohybridGraphFrontierGreater(PgturbohybridGraphFrontierItem a, PgturbohybridGraphFrontierItem b)
{
	if (a.distance != b.distance)
		return a.distance > b.distance;
	return a.nodeId > b.nodeId;
}

static void
PgturbohybridGraphFrontierSwap(PgturbohybridGraphFrontierItem *a, PgturbohybridGraphFrontierItem *b)
{
	PgturbohybridGraphFrontierItem tmp = *a;

	*a = *b;
	*b = tmp;
}

static void
PgturbohybridGraphFrontierHeapSiftUp(PgturbohybridGraphFrontierItem *heap, int idx, bool minHeap)
{
	while (idx > 0)
	{
		int			parent = (idx - 1) / 2;
		bool		before = minHeap ?
			PgturbohybridGraphFrontierLess(heap[idx], heap[parent]) :
			PgturbohybridGraphFrontierGreater(heap[idx], heap[parent]);

		if (!before)
			break;

		PgturbohybridGraphFrontierSwap(&heap[idx], &heap[parent]);
		idx = parent;
	}
}

static void
PgturbohybridGraphFrontierHeapSiftDown(PgturbohybridGraphFrontierItem *heap, int count, int idx,
							bool minHeap)
{
	for (;;)
	{
		int			left = idx * 2 + 1;
		int			right = left + 1;
		int			best = idx;

		if (left < count)
		{
			bool		before = minHeap ?
				PgturbohybridGraphFrontierLess(heap[left], heap[best]) :
				PgturbohybridGraphFrontierGreater(heap[left], heap[best]);

			if (before)
				best = left;
		}

		if (right < count)
		{
			bool		before = minHeap ?
				PgturbohybridGraphFrontierLess(heap[right], heap[best]) :
				PgturbohybridGraphFrontierGreater(heap[right], heap[best]);

			if (before)
				best = right;
		}

		if (best == idx)
			break;

		PgturbohybridGraphFrontierSwap(&heap[idx], &heap[best]);
		idx = best;
	}
}

static void
PgturbohybridGraphFrontierHeapPush(PgturbohybridGraphFrontierItem *heap, int *count,
						PgturbohybridGraphFrontierItem item, bool minHeap)
{
	heap[*count] = item;
	(*count)++;
	PgturbohybridGraphFrontierHeapSiftUp(heap, *count - 1, minHeap);
}

static int
PgturbohybridGraphInitialFrontierCapacity(uint32 nodeCount, int searchEf, int entryCount,
							   int maxNeighbors)
{
	int			capacity;

	if (nodeCount == 0)
		return 0;

	capacity = Max(8, searchEf + entryCount + maxNeighbors);
	capacity = Max(capacity, (searchEf * 2) + entryCount);
	if ((uint32) capacity > nodeCount)
		capacity = (int) nodeCount;

	return capacity;
}

static void
PgturbohybridGraphFrontierHeapPushGrowing(PgturbohybridGraphFrontierItem **heap, int *count,
							   int *capacity, int maxCapacity,
							   PgturbohybridGraphFrontierItem item, bool minHeap)
{
	if (*count >= *capacity)
	{
		int			newCapacity;

		if (*capacity >= maxCapacity)
			elog(ERROR, "pgturbohybrid graph frontier capacity exceeded");

		newCapacity = Max(8, *capacity * 2);
		if (newCapacity < *capacity || newCapacity > maxCapacity)
			newCapacity = maxCapacity;

		*heap = repalloc(*heap, sizeof(PgturbohybridGraphFrontierItem) * newCapacity);
		*capacity = newCapacity;
	}

	PgturbohybridGraphFrontierHeapPush(*heap, count, item, minHeap);
}

static PgturbohybridGraphFrontierItem
PgturbohybridGraphFrontierHeapPop(PgturbohybridGraphFrontierItem *heap, int *count, bool minHeap)
{
	PgturbohybridGraphFrontierItem item = heap[0];

	(*count)--;
	if (*count > 0)
	{
		heap[0] = heap[*count];
		PgturbohybridGraphFrontierHeapSiftDown(heap, *count, 0, minHeap);
	}

	return item;
}

static void
PgturbohybridGraphFrontierHeapReplaceRoot(PgturbohybridGraphFrontierItem *heap, int count,
							   PgturbohybridGraphFrontierItem item, bool minHeap)
{
	heap[0] = item;
	PgturbohybridGraphFrontierHeapSiftDown(heap, count, 0, minHeap);
}

static bool
PgturbohybridGraphOfferNearest(PgturbohybridGraphFrontierItem *heap, int capacity, int *count,
					uint32 nodeId, double distance)
{
	PgturbohybridGraphFrontierItem item;

	if (capacity <= 0)
		return false;

	item.nodeId = nodeId;
	item.distance = distance;

	if (*count < capacity)
	{
		PgturbohybridGraphFrontierHeapPush(heap, count, item, false);
		return true;
	}

	if (PgturbohybridGraphFrontierLess(item, heap[0]))
	{
		PgturbohybridGraphFrontierHeapReplaceRoot(heap, *count, item, false);
		return true;
	}

	return false;
}

static double
PgturbohybridGraphResultDistance(PgturbohybridGraphScanOpaque so, Datum query, PgturbohybridGraphScanNode *node,
					  double packedDistance, bool *exactScored)
{
	double		exactDistance;

	if (PgturbohybridGraphCachedExactNodeDistance(so, query, node, &exactDistance))
	{
		*exactScored = true;
		return exactDistance;
	}

	*exactScored = false;
	return packedDistance;
}

static double
PgturbohybridGraphEntryDistance(PgturbohybridGraphScanOpaque so, Datum query, PgturbohybridGraphScanNode *node)
{
	double		exactDistance;

	if (PgturbohybridGraphCachedExactNodeDistance(so, query, node, &exactDistance))
		return exactDistance;

	return PgturbohybridGraphScoreNode(so, node);
}

static void
PgturbohybridGraphEnsureNodeCapacity(PgturbohybridQuantBuildState *state)
{
	if (state->nodeCount < state->nodeCapacity)
		return;

	state->nodeCapacity = state->nodeCapacity == 0 ? 1024 : state->nodeCapacity * 2;
	state->nodes = repalloc(state->nodes, sizeof(PgturbohybridGraphBuildNode) * state->nodeCapacity);
}

static void
PgturbohybridGraphAppendBuildNode(PgturbohybridQuantBuildState *state, ItemPointer tid, Datum value,
					   Datum *values, bool *isnull)
{
	Vector	   *vector = (Vector *) DatumGetPointer(value);
	PgturbohybridGraphBuildNode *node;
	PgturbohybridGraphBuildNode *prev = NULL;
	Size		vectorSize;
	uint64		vectorHash;
	uint32		nodeId = state->nodeCount;
	int			level;
	bool		duplicatePrevious = false;

	if (state->dimensions == 0)
		state->dimensions = vector->dim;
	else if (state->dimensions != vector->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("different vector dimensions are not supported in the same pgturbohybrid graph")));

	PgturbohybridGraphEnsureNodeCapacity(state);
	vectorSize = PGTURBOHYBRID_VECTOR_SIZE(vector->dim);
	vectorHash = PgturbohybridGraphBuildVectorHash(vector);
	if (state->nodeCount > 0)
	{
		prev = &state->nodes[state->nodeCount - 1];
		duplicatePrevious =
			prev->vectorHash == vectorHash &&
			prev->vector != NULL &&
			prev->vector->dim == vector->dim &&
			memcmp(prev->vector, vector, vectorSize) == 0;
	}
	node = &state->nodes[state->nodeCount++];
	level = PgturbohybridGraphPickLevel(nodeId, state->m);

	node->vectorHash = vectorHash;
	if (duplicatePrevious)
	{
		node->vector = prev->vector;
		node->code = prev->code;
	}
	else
	{
		node->vector = palloc(vectorSize);
		memcpy(node->vector, vector, vectorSize);
		node->code = palloc(PgturbohybridGraphCodeBytesForBits(vector->dim, state->tqBits));
	}
	if (state->payloadCount > 0)
	{
		node->payloads = palloc0(state->payloadBytes);
		PgturbohybridGraphCopyPayloadValues(state, node->payloads, &node->payloadMask,
								 values, isnull);
	}
	node->level = level;
	node->norm = PgturbohybridGraphVectorNorm(vector);
	node->flags = 0;
	node->heaptid = *tid;
	node->neighbors = palloc0(sizeof(uint32 *) * (level + 1));
	node->neighborCounts = palloc0(sizeof(int) * (level + 1));
	for (int i = 0; i <= level; i++)
		node->neighbors[i] = palloc0(sizeof(uint32) * (PgturbohybridGraphLevelM(state->m, i) + 1));
	state->maxLevel = Max(state->maxLevel, level);
}

static void
PgturbohybridGraphBuildCallback(Relation index, ItemPointer tid, Datum *values,
					 bool *isnull, bool tupleIsAlive, void *opaque)
{
	PgturbohybridQuantBuildState *state = (PgturbohybridQuantBuildState *) opaque;
	MemoryContext oldCtx;
	Datum		value;

	(void) index;
	(void) tupleIsAlive;

	CHECK_FOR_INTERRUPTS();

	if (isnull[0])
		return;

	oldCtx = MemoryContextSwitchTo(state->ctx);
	if (PgturbohybridGraphFormIndexValue(&value, values, isnull, state->typeInfo, &state->support))
	{
		PgturbohybridGraphAppendBuildNode(state, tid, value, values, isnull);
		pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_DONE, state->nodeCount);
	}
	MemoryContextSwitchTo(oldCtx);
}

/*
 * Extended P-square one-quantile estimator (Jain & Chlamtac
 * 1985, N=5 markers).  Streaming, fixed memory per estimator,
 * independent of input size.  Used to fit the
 * quantile-anchored ecShift / ecScale per coord.
 *
 * Reference: <https://www.cse.wustl.edu/~jain/papers/ftp/psqr.pdf>.
 *
 * State per estimator: 5 marker heights + 5 marker positions + a
 * count + the target quantile.  Once `count >= 5` the estimator is
 * fully initialized and `Estimate` returns the running quantile;
 * before that it falls back to the median of observed values.
 *
 * Only base C math — no SIMD or threads.  At dim=1536 with 2
 * estimators per coord (q_lo, q_hi) the per-push cost is ~50 ns × 2
 * × 1536 ≈ 154 µs per vector, dominated by the FMA in
 * update_desired_positions and the parabolic adjust in adjust_step.
 * For a 57k-vector FIQA build that's ~9 s of fit-time pre-pass —
 * one-time cost, no runtime impact.
 */

/*
 * c_outer for the bit-width's Lloyd-Max codebook — the outermost
 * centroid magnitude.  Already encoded in the file as the denominator
 * of the per-bit CODEBOOK_SCALE, but we need the raw float here for
 * the quantile fit math.
 */
static double
PgturbohybridGraphCodebookOuter(int bits)
{
	if (bits == 2)
		return 1.510;
	if (bits == 1)
		return 1.000;
	return 2.733;					/* 4-bit default */
}

/*
 * Phi(x) — standard normal CDF.  Used to map the codebook outermost
 * centroid magnitude to the symmetric quantile probability that the
 * empirical fit anchors on.  For 4-bit, c_outer = 2.733 →
 * p_outer = 0.9968...; for 2-bit, c_outer = 1.510 → 0.9345.
 */
static double
PgturbohybridGraphPhi(double x)
{
	return 0.5 * (1.0 + erf(x / 1.41421356237309504880));
}

/*
 * Quantile-anchored ecShift / ecScale fit.
 *
 * Streams every build vector through TqPreprocessVector, then pushes
 * each per-coord rotated value into a pair of P-square estimators
 * (q_lo, q_hi).  After all observations:
 *
 *    shift[d] = -(q_lo[d] + q_hi[d]) / 2
 *    scale[d] = (2 · c_outer) / (q_hi[d] - q_lo[d])     (width-floor)
 *
 * The shift/scale arrays are written into state->ecShift /
 * state->ecScale — same downstream consumers as the Welford path.
 */
static void
PgturbohybridGraphFitCorrectionQuantile(PgturbohybridQuantBuildState *state)
{
	const double MIN_QUANTILE_WIDTH = 1e-3;
	double		c_outer;
	double		p_outer;
	double		q_lo_target;
	double		q_hi_target;
	TqPSquareState *qLo;
	TqPSquareState *qHi;
	double	   *buffer;

	c_outer = PgturbohybridGraphCodebookOuter(state->tqBits);
	p_outer = PgturbohybridGraphPhi(c_outer);
	q_lo_target = 1.0 - p_outer;
	q_hi_target = p_outer;

	qLo = palloc0(sizeof(TqPSquareState) * state->dimensions);
	qHi = palloc0(sizeof(TqPSquareState) * state->dimensions);
	for (int dim = 0; dim < state->dimensions; dim++)
	{
		TqPSquareInit(&qLo[dim], q_lo_target);
		TqPSquareInit(&qHi[dim], q_hi_target);
	}

	buffer = palloc(sizeof(double) * state->dimensions);
	for (uint32 row = 0; row < state->nodeCount; row++)
	{
		CHECK_FOR_INTERRUPTS();
		TqPreprocessVector(state->nodes[row].vector, buffer);
		for (int dim = 0; dim < state->dimensions; dim++)
		{
			TqPSquarePush(&qLo[dim], buffer[dim]);
			TqPSquarePush(&qHi[dim], buffer[dim]);
		}
	}
	pfree(buffer);

	state->ecShift = MemoryContextAlloc(state->ctx, sizeof(float) * state->dimensions);
	state->ecScale = MemoryContextAlloc(state->ctx, sizeof(float) * state->dimensions);
	for (int dim = 0; dim < state->dimensions; dim++)
	{
		double		q_lo = TqPSquareEstimate(&qLo[dim]);
		double		q_hi = TqPSquareEstimate(&qHi[dim]);
		double		denom = q_hi - q_lo;

		state->ecShift[dim] = (float) (-0.5 * (q_lo + q_hi));
		if (denom > MIN_QUANTILE_WIDTH)
			state->ecScale[dim] = (float) ((2.0 * c_outer) / denom);
		else
			state->ecScale[dim] = 1.0f;
	}

	pfree(qLo);
	pfree(qHi);
}

static void
PgturbohybridGraphFitCorrection(PgturbohybridQuantBuildState *state)
{
	double	   *mean;
	double	   *m2;
	double	   *buffer;

	if (state->nodeCount == 0 || state->dimensions <= 0 ||
		(state->scoreMode != PGTURBOHYBRID_SCORE_COSINE && state->scoreMode != PGTURBOHYBRID_SCORE_IP))
		return;

	if (state->tqQuantileFit)
	{
		PgturbohybridGraphFitCorrectionQuantile(state);
		goto post_fit;
	}

	mean = palloc0(sizeof(double) * state->dimensions);
	m2 = palloc0(sizeof(double) * state->dimensions);
	buffer = palloc(sizeof(double) * state->dimensions);

	for (uint32 row = 0; row < state->nodeCount;)
	{
		uint32		runEnd = row + 1;
		double		priorN = (double) row;
		double		runN;
		double		newN;

		CHECK_FOR_INTERRUPTS();
		while (runEnd < state->nodeCount &&
			   PgturbohybridGraphBuildVectorsEqual(state, row, runEnd))
		{
			if ((runEnd & 0x3FF) == 0)
				CHECK_FOR_INTERRUPTS();
			runEnd++;
		}

		TqPreprocessVector(state->nodes[row].vector, buffer);
		runN = (double) (runEnd - row);
		newN = priorN + runN;
		for (int dim = 0; dim < state->dimensions; dim++)
		{
			double		value = buffer[dim];
			double		delta = value - mean[dim];

			mean[dim] += delta * runN / newN;
			m2[dim] += delta * delta * priorN * runN / newN;
		}
		row = runEnd;
	}

	state->ecShift = MemoryContextAlloc(state->ctx, sizeof(float) * state->dimensions);
	state->ecScale = MemoryContextAlloc(state->ctx, sizeof(float) * state->dimensions);
	for (int dim = 0; dim < state->dimensions; dim++)
	{
		double		variance = state->nodeCount > 1 ? m2[dim] / ((double) state->nodeCount - 1.0) : 0.0;
		double		stddev = variance > 0 ? sqrt(variance) : 0.0;

		state->ecShift[dim] = (float) -mean[dim];
		state->ecScale[dim] = stddev > FLT_EPSILON ? (float) (1.0 / stddev) : 1.0f;
	}

	pfree(buffer);
	pfree(m2);
	pfree(mean);

post_fit:

	/*
	 * cache mm_const = Σ ecShift² so build-time TQ+ scoring
	 * doesn't recompute it per neighbor-distance call.
	 */
	state->mmConst = PgturbohybridGraphMmConstScalar(state->ecShift, state->dimensions);

	if (state->tqWeighted)
	{
		double		dPrimeSqMax = 0.0;
		double		weightScale;

		for (int dim = 0; dim < state->dimensions; dim++)
		{
			double		s = (double) state->ecScale[dim];

			if (fabs(s) > FLT_EPSILON)
			{
				double		w = 1.0 / (s * s);

				if (w > dPrimeSqMax)
					dPrimeSqMax = w;
			}
		}

		/*
		 * Quantize per-coord D'² to i16 so the AVX2 SIMD
		 * weighted-dot kernel can use _mm256_madd_epi16 directly.
		 * weight_scale = (INT16_MAX - 1) / max(D'²) keeps the largest
		 * weight at INT16_MAX-1; relative quantization error on the
		 * smallest non-zero weight is (min/max) · 1/32766 — well below
		 * the 4-bit code precision floor.
		 */
		weightScale = dPrimeSqMax > FLT_EPSILON
			? ((double) INT16_MAX - 1.0) / dPrimeSqMax
			: 1.0;
		state->weightScale = (float) weightScale;
		state->dPrimeSqI16 = MemoryContextAlloc(state->ctx,
												 sizeof(int16) * state->dimensions);
		for (int dim = 0; dim < state->dimensions; dim++)
		{
			double		s = (double) state->ecScale[dim];
			double		w = (fabs(s) > FLT_EPSILON) ? 1.0 / (s * s) : 0.0;
			double		q = round(w * weightScale);

			if (q < 0.0)
				q = 0.0;
			if (q > (double) (INT16_MAX - 1))
				q = (double) (INT16_MAX - 1);
			state->dPrimeSqI16[dim] = (int16) q;
		}

		elog(DEBUG2, "pgturbohybrid TQ+ fit: dim=%d mm_const=%g max_dprime_sq=%g weight_scale=%g (fit=%s)",
			 state->dimensions, state->mmConst, dPrimeSqMax, weightScale,
			 state->tqQuantileFit ? "quantile" : "welford");
	}
}

static void
PgturbohybridGraphEncodeBuildNodes(PgturbohybridQuantBuildState *state)
{
	Size		codeBytes = PgturbohybridGraphCodeBytesForBits(state->dimensions,
													state->tqBits);

	for (uint32 row = 0; row < state->nodeCount; row++)
	{
		PgturbohybridGraphBuildNode *node = &state->nodes[row];

		CHECK_FOR_INTERRUPTS();
		if (row > 0 && PgturbohybridGraphBuildVectorsEqual(state, row - 1, row))
		{
			PgturbohybridGraphBuildNode *prev = &state->nodes[row - 1];

			if (node->code != prev->code)
				memcpy(node->code, prev->code, codeBytes);
			node->scale = prev->scale;
			node->ecCorrection = prev->ecCorrection;
			node->correction = prev->correction;
			continue;
		}

		if (state->tqWeighted)
		{
			float		xm = 0.0f;

			if (state->tqRenorm)
				node->scale = PgturbohybridGraphEncodeVectorWithXmRenorm(state, node->vector,
															  node->code, &xm);
			else
				node->scale = PgturbohybridGraphEncodeVectorWithXm(state, node->vector,
														 node->code, &xm);
			node->ecCorrection = xm;
		}
		else
		{
			node->scale = PgturbohybridGraphEncodeVector(state, node->vector, node->code);
			node->ecCorrection = 0.0f;
		}

		node->correction = PgturbohybridGraphCodeNorm(node->code, state->dimensions, state->tqBits);
	}
}

static bool
PgturbohybridGraphHasNeighbor(PgturbohybridQuantBuildState *state, uint32 src, uint32 dst, int level)
{
	PgturbohybridGraphBuildNode *node = &state->nodes[src];

	if (!PgturbohybridGraphBuildNodeHasLevel(state, src, level))
		return false;

	for (int i = 0; i < node->neighborCounts[level]; i++)
	{
		if (node->neighbors[level][i] == dst)
			return true;
	}

	return false;
}

static int
PgturbohybridGraphSelectNeighbors(PgturbohybridQuantBuildState *state, uint32 src,
					   PgturbohybridGraphFrontierItem *candidates, int candidateCount,
					   int level, uint32 *selected)
{
	int			selectedCount = 0;
	int			maxNeighbors = PgturbohybridGraphLevelM(state->m, level);

	qsort(candidates, candidateCount, sizeof(PgturbohybridGraphFrontierItem), PgturbohybridGraphFrontierCompare);

	for (int i = 0; i < candidateCount && selectedCount < maxNeighbors; i++)
	{
		uint32		candidate = candidates[i].nodeId;
		bool		good = true;

		if (candidate == src)
			continue;

		for (int j = 0; j < selectedCount; j++)
		{
			double		selectedDistance = PgturbohybridGraphBuildDistance(state, candidate, selected[j]);

			if (selectedDistance < candidates[i].distance)
			{
				good = false;
				break;
			}
		}

		if (good)
			selected[selectedCount++] = candidate;
	}

	for (int i = 0; i < candidateCount && selectedCount < maxNeighbors; i++)
	{
		bool		seen = false;

		for (int j = 0; j < selectedCount; j++)
		{
			if (selected[j] == candidates[i].nodeId)
			{
				seen = true;
				break;
			}
		}

		if (!seen && candidates[i].nodeId != src)
			selected[selectedCount++] = candidates[i].nodeId;
	}

	return selectedCount;
}

static void
PgturbohybridGraphPruneNeighbors(PgturbohybridQuantBuildState *state, uint32 src, int level)
{
	PgturbohybridGraphBuildNode *node = &state->nodes[src];
	int			count;
	PgturbohybridGraphFrontierItem *candidates;
	uint32	   *selected;
	int			selectedCount;

	if (!PgturbohybridGraphBuildNodeHasLevel(state, src, level))
		return;

	count = node->neighborCounts[level];
	if (count <= PgturbohybridGraphLevelM(state->m, level))
		return;

	candidates = palloc(sizeof(PgturbohybridGraphFrontierItem) * count);
	selected = palloc(sizeof(uint32) * PgturbohybridGraphLevelM(state->m, level));

	for (int i = 0; i < count; i++)
	{
		candidates[i].nodeId = node->neighbors[level][i];
		candidates[i].distance = PgturbohybridGraphBuildDistance(state, src, candidates[i].nodeId);
	}

	selectedCount = PgturbohybridGraphSelectNeighbors(state, src, candidates, count, level, selected);
	memcpy(node->neighbors[level], selected, sizeof(uint32) * selectedCount);
	node->neighborCounts[level] = selectedCount;

	pfree(candidates);
	pfree(selected);
}

static void
PgturbohybridGraphAddNeighbor(PgturbohybridQuantBuildState *state, uint32 src, uint32 dst, int level)
{
	PgturbohybridGraphBuildNode *node = &state->nodes[src];
	int			maxNeighbors = PgturbohybridGraphLevelM(state->m, level);

	if (src == dst || !PgturbohybridGraphBuildNodeHasLevel(state, src, level) ||
		!PgturbohybridGraphBuildNodeHasLevel(state, dst, level) ||
		PgturbohybridGraphHasNeighbor(state, src, dst, level))
		return;

	if (node->neighborCounts[level] < maxNeighbors)
	{
		node->neighbors[level][node->neighborCounts[level]++] = dst;
		return;
	}

	node->neighbors[level][node->neighborCounts[level]++] = dst;
	PgturbohybridGraphPruneNeighbors(state, src, level);
}

static void
PgturbohybridGraphAddNeighborIfRoom(PgturbohybridQuantBuildState *state, uint32 src, uint32 dst,
						 int level)
{
	PgturbohybridGraphBuildNode *node = &state->nodes[src];
	int			maxNeighbors = PgturbohybridGraphLevelM(state->m, level);

	if (src == dst || !PgturbohybridGraphBuildNodeHasLevel(state, src, level) ||
		!PgturbohybridGraphBuildNodeHasLevel(state, dst, level) ||
		PgturbohybridGraphHasNeighbor(state, src, dst, level) ||
		node->neighborCounts[level] >= maxNeighbors)
		return;

	node->neighbors[level][node->neighborCounts[level]++] = dst;
}

static uint32
PgturbohybridGraphAdjacentDuplicateCount(PgturbohybridQuantBuildState *state)
{
	uint32		duplicates = 0;

	for (uint32 i = 1; i < state->nodeCount; i++)
	{
		if (PgturbohybridGraphBuildVectorsEqual(state, i - 1, i))
			duplicates++;
	}

	return duplicates;
}

static void
PgturbohybridGraphLinkAdjacentBuildNode(PgturbohybridQuantBuildState *state, uint32 nodeId,
							 uint32 prevId)
{
	PgturbohybridGraphBuildNode *node = &state->nodes[nodeId];

	if (!PgturbohybridGraphBuildNodeHasLevel(state, nodeId, 0) ||
		!PgturbohybridGraphBuildNodeHasLevel(state, prevId, 0))
		return;

	node->neighbors[0][0] = prevId;
	node->neighborCounts[0] = 1;
	PgturbohybridGraphAddNeighborIfRoom(state, prevId, nodeId, 0);
}

static PgturbohybridGraphFrontierItem
PgturbohybridGraphBuildGreedySearch(PgturbohybridQuantBuildState *state, uint32 queryNodeId,
						 uint32 entryNodeId, int level, bool *inserted)
{
	PgturbohybridGraphFrontierItem current;
	bool		changed = true;

	current.nodeId = entryNodeId;
	current.distance = PgturbohybridGraphBuildDistance(state, queryNodeId, entryNodeId);

	while (changed)
	{
		PgturbohybridGraphBuildNode *node = &state->nodes[current.nodeId];

		changed = false;
		if (!PgturbohybridGraphBuildNodeHasLevel(state, current.nodeId, level))
			break;

		for (int i = 0; i < node->neighborCounts[level]; i++)
		{
			uint32		neighbor = node->neighbors[level][i];
			double		distance;

			if (!inserted[neighbor] ||
				!PgturbohybridGraphBuildNodeHasLevel(state, neighbor, level))
				continue;

			distance = PgturbohybridGraphBuildDistance(state, queryNodeId, neighbor);
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
PgturbohybridGraphBuildSearchLayer(PgturbohybridQuantBuildState *state, uint32 queryNodeId,
						PgturbohybridGraphFrontierItem entry, int level, int ef,
						PgturbohybridGraphFrontierItem *nearest, bool *inserted)
{
	uint32		visitGeneration;
	int			maxNeighbors = PgturbohybridGraphLevelM(state->m, level);
	int			frontierCapacity = PgturbohybridGraphInitialFrontierCapacity(state->nodeCount, ef, 1,
																 maxNeighbors);
	int			maxFrontierCapacity = (int) state->nodeCount;
	PgturbohybridGraphFrontierItem *frontier = palloc(sizeof(PgturbohybridGraphFrontierItem) * frontierCapacity);
	int			frontierCount = 0;
	int			nearestCount = 0;

	visitGeneration = ++state->buildVisitGeneration;
	if (visitGeneration == 0)
	{
		memset(state->buildVisitedGeneration, 0,
			   sizeof(uint32) * state->nodeCount);
		visitGeneration = ++state->buildVisitGeneration;
	}

	state->buildVisitedGeneration[entry.nodeId] = visitGeneration;
	(void) PgturbohybridGraphOfferNearest(nearest, ef, &nearestCount, entry.nodeId, entry.distance);
	PgturbohybridGraphFrontierHeapPushGrowing(&frontier, &frontierCount, &frontierCapacity,
								   maxFrontierCapacity, entry, true);

	while (frontierCount > 0)
	{
		PgturbohybridGraphFrontierItem item = PgturbohybridGraphFrontierHeapPop(frontier, &frontierCount, true);
		PgturbohybridGraphBuildNode *node = &state->nodes[item.nodeId];

		if (nearestCount >= ef && PgturbohybridGraphFrontierGreater(item, nearest[0]))
			break;

		if (!PgturbohybridGraphBuildNodeHasLevel(state, item.nodeId, level))
			continue;

		for (int i = 0; i < node->neighborCounts[level]; i++)
		{
			uint32		neighbor = node->neighbors[level][i];
			double		distance;

			if (!inserted[neighbor] ||
				state->buildVisitedGeneration[neighbor] == visitGeneration ||
				!PgturbohybridGraphBuildNodeHasLevel(state, neighbor, level))
				continue;

			state->buildVisitedGeneration[neighbor] = visitGeneration;
			distance = PgturbohybridGraphBuildDistance(state, queryNodeId, neighbor);
			if (PgturbohybridGraphOfferNearest(nearest, ef, &nearestCount, neighbor, distance))
			{
				PgturbohybridGraphFrontierItem frontierItem;

				frontierItem.nodeId = neighbor;
				frontierItem.distance = distance;
				PgturbohybridGraphFrontierHeapPushGrowing(&frontier, &frontierCount,
											   &frontierCapacity,
											   maxFrontierCapacity, frontierItem,
											   true);
			}
		}
	}

	pfree(frontier);
	return nearestCount;
}

static void
PgturbohybridGraphBuildEdges(PgturbohybridQuantBuildState *state)
{
	int			ef = Max(state->efConstruction, PgturbohybridGraphLevelM(state->m, 0));
	uint32		entryNodeId;
	int			entryLevel;
	PgturbohybridGraphFrontierItem *nearest;
	uint32	   *selected;
	PgturbohybridGraphBuildOrderItem *order;
	bool	   *inserted;
	uint32		adjacentDuplicates;
	bool		preserveScanOrder;

	if (state->nodeCount == 0)
		return;

	nearest = palloc(sizeof(PgturbohybridGraphFrontierItem) * ef);
	selected = palloc(sizeof(uint32) * PgturbohybridGraphLevelM(state->m, 0));
	order = palloc(sizeof(PgturbohybridGraphBuildOrderItem) * state->nodeCount);
	inserted = palloc0(sizeof(bool) * state->nodeCount);
	state->buildVisitedGeneration = palloc0(sizeof(uint32) * state->nodeCount);
	state->buildVisitGeneration = 0;
	state->buildQueryCtx = AllocSetContextCreate(state->ctx,
												 "pgturbohybrid graph build query context",
												 ALLOCSET_DEFAULT_SIZES);

	for (uint32 i = 0; i < state->nodeCount; i++)
	{
		order[i].nodeId = i;
		order[i].key = PgturbohybridGraphMix64(i);
	}
	adjacentDuplicates = PgturbohybridGraphAdjacentDuplicateCount(state);
	preserveScanOrder = adjacentDuplicates > state->nodeCount / 2;
	elog(DEBUG1, "pgturbohybrid native graph duplicate-run build: nodes=%u adjacent_duplicates=%u preserve_scan_order=%s",
		 state->nodeCount, adjacentDuplicates, preserveScanOrder ? "on" : "off");
	if (!preserveScanOrder)
		qsort(order, state->nodeCount, sizeof(PgturbohybridGraphBuildOrderItem),
			  PgturbohybridGraphBuildOrderCompare);

	entryNodeId = order[0].nodeId;
	entryLevel = state->nodes[entryNodeId].level;
	inserted[entryNodeId] = true;

	for (uint32 orderIdx = 1; orderIdx < state->nodeCount; orderIdx++)
	{
		uint32		i = order[orderIdx].nodeId;
		PgturbohybridGraphFrontierItem levelEntry;
		int			nodeLevel = state->nodes[i].level;
		int			linkingLevel = Min(nodeLevel, entryLevel);

			CHECK_FOR_INTERRUPTS();
			if (preserveScanOrder && i > 0 && inserted[i - 1])
			{
				PgturbohybridGraphLinkAdjacentBuildNode(state, i, i - 1);
				if (nodeLevel > entryLevel)
				{
					entryNodeId = i;
					entryLevel = nodeLevel;
				}
				inserted[i] = true;
				continue;
			}
			PgturbohybridGraphPrepareBuildQuery(state, i);

			levelEntry.nodeId = entryNodeId;
			levelEntry.distance = PgturbohybridGraphBuildDistance(state, i, entryNodeId);

		for (int level = entryLevel; level > nodeLevel; level--)
		{
			if (PgturbohybridGraphBuildNodeHasLevel(state, levelEntry.nodeId, level))
				levelEntry = PgturbohybridGraphBuildGreedySearch(state, i,
													  levelEntry.nodeId,
													  level, inserted);
		}

		for (int level = linkingLevel; level >= 0; level--)
		{
			int			nearestCount;
			int			selectedCount;

			if (!PgturbohybridGraphBuildNodeHasLevel(state, levelEntry.nodeId, level))
				continue;

			nearestCount = PgturbohybridGraphBuildSearchLayer(state, i, levelEntry, level,
												   ef, nearest, inserted);
			selectedCount = PgturbohybridGraphSelectNeighbors(state, i, nearest, nearestCount,
												  level, selected);

			memcpy(state->nodes[i].neighbors[level], selected,
				   sizeof(uint32) * selectedCount);
			state->nodes[i].neighborCounts[level] = selectedCount;
			for (int j = 0; j < selectedCount; j++)
				PgturbohybridGraphAddNeighbor(state, selected[j], i, level);

			if (nearestCount > 0)
			{
				qsort(nearest, nearestCount, sizeof(PgturbohybridGraphFrontierItem), PgturbohybridGraphFrontierCompare);
				levelEntry = nearest[0];
			}
		}

		if (nodeLevel > entryLevel)
		{
			entryNodeId = i;
			entryLevel = nodeLevel;
		}
		inserted[i] = true;
	}

	state->entryNodeId = entryNodeId;
	state->maxLevel = entryLevel;
	pfree(nearest);
	pfree(selected);
	pfree(order);
	pfree(inserted);
	pfree(state->buildVisitedGeneration);
	state->buildVisitedGeneration = NULL;
	MemoryContextDelete(state->buildQueryCtx);
	state->buildQueryCtx = NULL;
	state->buildTqValid = false;
}

static void
PgturbohybridGraphReorderBuildNodesForLocality(PgturbohybridQuantBuildState *state)
{
	PgturbohybridGraphBuildNode *reordered;
	uint32	   *oldToNew;
	uint32	   *newToOld;
	uint32	   *queue;
	uint32		head = 0;
	uint32		tail = 0;
	uint32		orderCount = 0;
	bool		identity = true;

	if (PgturbohybridGraphGetGraphReorder(state->index) == PGTURBOHYBRID_GRAPH_REORDER_OFF)
		return;
	if (state->nodeCount < 2 || state->entryNodeId >= state->nodeCount)
		return;

	oldToNew = palloc(sizeof(uint32) * state->nodeCount);
	newToOld = palloc(sizeof(uint32) * state->nodeCount);
	queue = palloc(sizeof(uint32) * state->nodeCount);
	for (uint32 i = 0; i < state->nodeCount; i++)
		oldToNew[i] = UINT_MAX;

#define PGTURBOHYBRID_GRAPH_ENQUEUE_OLD_NODE(oldid) \
	do { \
		uint32 _oldid = (oldid); \
		if (_oldid < state->nodeCount && oldToNew[_oldid] == UINT_MAX) \
		{ \
			oldToNew[_oldid] = orderCount; \
			newToOld[orderCount++] = _oldid; \
			queue[tail++] = _oldid; \
		} \
	} while (0)

	PGTURBOHYBRID_GRAPH_ENQUEUE_OLD_NODE(state->entryNodeId);
	while (head < tail)
	{
		uint32		oldId = queue[head++];
		PgturbohybridGraphBuildNode *node = &state->nodes[oldId];

		if (node->level < 0 || node->neighborCounts == NULL ||
			node->neighbors == NULL)
			continue;

		for (int i = 0; i < node->neighborCounts[0]; i++)
			PGTURBOHYBRID_GRAPH_ENQUEUE_OLD_NODE(node->neighbors[0][i]);
	}

	for (uint32 i = 0; i < state->nodeCount; i++)
		PGTURBOHYBRID_GRAPH_ENQUEUE_OLD_NODE(i);

#undef PGTURBOHYBRID_GRAPH_ENQUEUE_OLD_NODE

	for (uint32 i = 0; i < state->nodeCount; i++)
	{
		if (newToOld[i] != i)
		{
			identity = false;
			break;
		}
	}

	if (identity)
	{
		pfree(oldToNew);
		pfree(newToOld);
		pfree(queue);
		return;
	}

	reordered = MemoryContextAllocZero(state->ctx,
									   sizeof(PgturbohybridGraphBuildNode) * state->nodeCount);
	for (uint32 newId = 0; newId < state->nodeCount; newId++)
		reordered[newId] = state->nodes[newToOld[newId]];

	for (uint32 newId = 0; newId < state->nodeCount; newId++)
	{
		PgturbohybridGraphBuildNode *node = &reordered[newId];

		for (int level = 0; level <= node->level; level++)
		{
			for (int i = 0; i < node->neighborCounts[level]; i++)
			{
				uint32		oldNeighbor = node->neighbors[level][i];

				if (oldNeighbor < state->nodeCount &&
					oldToNew[oldNeighbor] != UINT_MAX)
					node->neighbors[level][i] = oldToNew[oldNeighbor];
			}
		}
	}

	state->entryNodeId = oldToNew[state->entryNodeId];
	state->nodes = reordered;

	pfree(oldToNew);
	pfree(newToOld);
	pfree(queue);
}




static void
PgturbohybridGraphCreateMetaPage(Relation index, ForkNumber forkNum)
{
	Buffer		buf;
	Page		page;
	PgturbohybridGraphMetaPage metap;

	buf = PgturbohybridGraphNewBuffer(index, forkNum);
	page = BufferGetPage(buf);
	PgturbohybridGraphInitPageKind(buf, page, PGTURBOHYBRID_GRAPH_PAGE_KIND_META);
	metap = PgturbohybridGraphPageGetMeta(page);

	memset(metap, 0, sizeof(PgturbohybridGraphMetaPageData));
	metap->magicNumber = PGTURBOHYBRID_GRAPH_MAGIC_NUMBER;
	metap->version = PGTURBOHYBRID_GRAPH_VERSION;
	metap->storageKind = PGTURBOHYBRID_GRAPH_STORAGE_QUANT_GRAPH_NATIVE;
	metap->m = PgturbohybridGraphGetM(index);
	metap->efConstruction = PgturbohybridGraphGetEfConstruction(index);
	metap->graphEfSearch = PgturbohybridGraphGetEfSearch(index);
	metap->graphOversampling = PgturbohybridGraphGetGraphOversampling(index);
	metap->graphRescoreBand = PgturbohybridGraphGetGraphRescoreBand(index);
	metap->tqBits = PgturbohybridGraphGetTqBits(index);
	metap->graphMaxLevel = 0;
	metap->entryBlkno = InvalidBlockNumber;
	metap->entryOffno = InvalidOffsetNumber;
	metap->entryLevel = -1;
	metap->insertPage = InvalidBlockNumber;
	metap->tqEntryNodeId = UINT_MAX;
	metap->tqCodeStartBlkno = InvalidBlockNumber;
	metap->tqAdjStartBlkno = InvalidBlockNumber;
	metap->tqExactStartBlkno = InvalidBlockNumber;
	metap->tqCorrectionStartBlkno = InvalidBlockNumber;
	metap->tqBm25MetaStartBlkno = InvalidBlockNumber;
	((PageHeader) page)->pd_lower =
		((char *) metap + sizeof(PgturbohybridGraphMetaPageData)) - (char *) page;

	PgturbohybridGraphFinishPage(buf);
}

void
PgturbohybridQuantUpdateMetaPage(Relation index, PgturbohybridQuantBuildState *state,
					  BlockNumber codeStart, BlockNumber adjStart,
					  BlockNumber exactStart, BlockNumber correctionStart)
{
	Buffer		buf;
	Page		page;
	PgturbohybridGraphMetaPage metap;
	GenericXLogState *xlogState = NULL;

	buf = ReadBufferExtended(index, state->forkNum, PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

	if (!state->building && state->forkNum == MAIN_FORKNUM && RelationNeedsWAL(index))
	{
		xlogState = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(xlogState, buf, 0);
	}
	else
		page = BufferGetPage(buf);

	metap = PgturbohybridGraphPageGetMeta(page);

	metap->dimensions = state->dimensions;
	metap->m = state->m;
	metap->efConstruction = state->efConstruction;
	metap->storageKind = PGTURBOHYBRID_GRAPH_STORAGE_QUANT_GRAPH_NATIVE;
	metap->graphEfSearch = PgturbohybridGraphGetEfSearch(index);
	metap->graphOversampling = PgturbohybridGraphGetGraphOversampling(index);
	metap->graphRescoreBand = PgturbohybridGraphGetGraphRescoreBand(index);
	metap->graphMaxLevel = state->maxLevel;
	metap->graphFlags = metap->graphFlags == 0 ? 1 : metap->graphFlags + 1;
	metap->entryBlkno = codeStart;
	metap->entryOffno = state->nodeCount > 0 ? FirstOffsetNumber : InvalidOffsetNumber;
	metap->entryLevel = state->nodeCount > 0 ? state->nodes[state->entryNodeId].level : -1;
	metap->tqNodeCount = state->nodeCount;
	metap->tqEntryNodeId = state->nodeCount > 0 ? state->entryNodeId : UINT_MAX;
	metap->tqCodeBytes = state->dimensions > 0 ? PgturbohybridGraphCodeBytesForBits(state->dimensions, state->tqBits) : 0;
	metap->tqPayloadCount = state->payloadCount;
	metap->tqPayloadBytes = state->payloadBytes;
	metap->tqFlags = state->ecShift != NULL && state->ecScale != NULL ? PGTURBOHYBRID_GRAPH_TQ_PLUS : 0;
	if (state->tqWeighted)
		metap->tqFlags |= PGTURBOHYBRID_GRAPH_TQ_WEIGHTED;
	if (state->tqRenorm)
		metap->tqFlags |= PGTURBOHYBRID_GRAPH_TQ_RENORM;
	if (!state->tqExactStorage)
		metap->tqFlags |= PGTURBOHYBRID_GRAPH_EXACT_FREE;
	metap->tqBits = state->tqBits != 0 ? state->tqBits : PGTURBOHYBRID_DEFAULT_BITS;
	metap->tqCodeStartBlkno = codeStart;
	metap->tqAdjStartBlkno = adjStart;
	metap->tqExactStartBlkno = exactStart;
	metap->tqCorrectionStartBlkno = correctionStart;
	PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_META_UPDATE);

	if (xlogState != NULL)
		GenericXLogFinish(xlogState);
	else
		MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);

	PgturbohybridGraphLogGraphWalRecord(index, state->forkNum, PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO, PGTURBOHYBRID_GRAPH_GRAPH_OP_META_UPDATE);
}

static void
PgturbohybridGraphBumpMetaGeneration(Relation index)
{
	Buffer		buf;
	Page		page;
	PgturbohybridGraphMetaPage metap;
	GenericXLogState *xlogState = NULL;

	buf = ReadBuffer(index, PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

	if (RelationNeedsWAL(index))
	{
		xlogState = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(xlogState, buf, 0);
	}
	else
		page = BufferGetPage(buf);

	metap = PgturbohybridGraphPageGetMeta(page);
	metap->graphFlags = metap->graphFlags == 0 ? 1 : metap->graphFlags + 1;
	PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_META_UPDATE);

	if (xlogState != NULL)
		GenericXLogFinish(xlogState);
	else
		MarkBufferDirty(buf);

	UnlockReleaseBuffer(buf);
	PgturbohybridGraphLogGraphWalRecord(index, MAIN_FORKNUM, PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO, PGTURBOHYBRID_GRAPH_GRAPH_OP_META_UPDATE);
}

static BlockNumber
PgturbohybridGraphWriteCodePages(PgturbohybridQuantBuildState *state)
{
	Buffer		buf = InvalidBuffer;
	Page		page = NULL;
	BlockNumber start = InvalidBlockNumber;
	Size		tupleSize = PgturbohybridGraphCodeTupleSize(state->dimensions, state->payloadCount,
												 state->tqBits, state->tqWeighted);
	PgturbohybridGraphCodeTuple tuple = palloc0(tupleSize);

	for (uint32 i = 0; i < state->nodeCount; i++)
	{
		PgturbohybridGraphBuildNode *node = &state->nodes[i];

		CHECK_FOR_INTERRUPTS();

		if (!BufferIsValid(buf) || PageGetFreeSpace(page) < tupleSize)
		{
			PgturbohybridGraphAppendPage(state->index, state->forkNum, &buf, &page, PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_CODE);
			if (!BlockNumberIsValid(start))
				start = BufferGetBlockNumber(buf);
		}

		memset(tuple, 0, tupleSize);
		tuple->type = PGTURBOHYBRID_GRAPH_CODE_TUPLE_TYPE;
		tuple->level = node->level;
		tuple->flags = node->flags;
		tuple->nodeId = i;
		tuple->heaptid = node->heaptid;
		tuple->exactBlkno = node->exactBlkno;
		tuple->exactOffno = node->exactOffno;
		tuple->payloadMask = node->payloadMask;
		tuple->scale = node->scale;
		tuple->norm = node->norm;
		tuple->correction = node->correction;
		PgturbohybridGraphTupleSetEcCorrection(tuple, state->tqWeighted, node->ecCorrection);
		if (state->payloadCount > 0 && node->payloads != NULL)
			memcpy(PgturbohybridGraphTuplePayloads(tuple, state->tqWeighted), node->payloads, state->payloadBytes);
		memcpy(PgturbohybridGraphTupleCode(tuple, state->payloadBytes, state->tqWeighted), node->code,
			   PgturbohybridGraphCodeBytesForBits(state->dimensions, state->tqBits));

		if (PageAddItem(page, (Item) tuple, tupleSize, InvalidOffsetNumber, false, false) == InvalidOffsetNumber)
			elog(ERROR, "failed to add pgturbohybrid graph code item to \"%s\"", RelationGetRelationName(state->index));
		PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT);
	}

	if (BufferIsValid(buf))
		PgturbohybridGraphFinishPage(buf);

	pfree(tuple);
	return start;
}

static BlockNumber
PgturbohybridGraphWriteAdjPages(PgturbohybridQuantBuildState *state)
{
	Buffer		buf = InvalidBuffer;
	Page		page = NULL;
	BlockNumber start = InvalidBlockNumber;
	Size		maxTupleSize = PgturbohybridGraphAdjTupleSize(PgturbohybridGraphLevelM(state->m, 0));
	PgturbohybridGraphAdjTuple tuple = palloc0(maxTupleSize);
	int			levelCapacity = PgturbohybridGraphLevelCapacity(state->m);

	for (uint32 i = 0; i < state->nodeCount; i++)
	{
		PgturbohybridGraphBuildNode *node = &state->nodes[i];
		int			maxLevel = Min(node->level, levelCapacity - 1);

		CHECK_FOR_INTERRUPTS();

		for (int level = 0; level <= maxLevel; level++)
		{
			int			count = node->neighborCounts[level];
			Size		tupleSize = PgturbohybridGraphAdjTupleSize(PgturbohybridGraphLevelM(state->m, level));

			if (!BufferIsValid(buf) || PageGetFreeSpace(page) < tupleSize)
			{
				PgturbohybridGraphAppendPage(state->index, state->forkNum, &buf, &page, PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_ADJ);
				if (!BlockNumberIsValid(start))
					start = BufferGetBlockNumber(buf);
			}

			memset(tuple, 0, maxTupleSize);
			tuple->type = PGTURBOHYBRID_GRAPH_ADJ_TUPLE_TYPE;
			tuple->level = level;
			tuple->count = count;
			tuple->nodeId = i;
			for (int j = 0; j < count; j++)
				tuple->neighbors[j] = node->neighbors[level][j];

			if (PageAddItem(page, (Item) tuple, tupleSize, InvalidOffsetNumber, false, false) == InvalidOffsetNumber)
				elog(ERROR, "failed to add pgturbohybrid graph adjacency item to \"%s\"", RelationGetRelationName(state->index));
			PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_NEIGHBOR_INSERT);
		}
	}

	if (BufferIsValid(buf))
		PgturbohybridGraphFinishPage(buf);

	pfree(tuple);
	return start;
}


static BlockNumber
PgturbohybridGraphWriteCorrectionPages(PgturbohybridQuantBuildState *state)
{
	Buffer		buf = InvalidBuffer;
	Page		page = NULL;
	BlockNumber start = InvalidBlockNumber;
	Size		maxTupleSize;
	PgturbohybridGraphCorrectionTuple tuple;
	int			maxValues;

	if (state->nodeCount == 0 || state->dimensions <= 0 ||
		state->ecShift == NULL || state->ecScale == NULL)
		return InvalidBlockNumber;

	maxValues = PgturbohybridGraphCorrectionTupleMaxCount();
	maxTupleSize = PgturbohybridGraphCorrectionTupleSize(maxValues);
	tuple = palloc0(maxTupleSize);

	for (int field = 0; field < 2; field++)
	{
		const float *values = field == 0 ? state->ecShift : state->ecScale;

		for (int startDim = 0; startDim < state->dimensions; startDim += maxValues)
		{
			int			count = Min(maxValues, state->dimensions - startDim);
			Size		tupleSize = PgturbohybridGraphCorrectionTupleSize(count);

			if (!BufferIsValid(buf) || PageGetFreeSpace(page) < tupleSize)
			{
				PgturbohybridGraphAppendPage(state->index, state->forkNum, &buf, &page,
								  PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_CORRECTION);
				if (!BlockNumberIsValid(start))
					start = BufferGetBlockNumber(buf);
			}

			memset(tuple, 0, maxTupleSize);
			tuple->type = PGTURBOHYBRID_GRAPH_CORRECTION_TUPLE_TYPE;
			tuple->field = field;
			tuple->count = count;
			tuple->startDim = startDim;
			memcpy(tuple->values, values + startDim, sizeof(float) * count);

			if (PageAddItem(page, (Item) tuple, tupleSize, InvalidOffsetNumber, false, false) == InvalidOffsetNumber)
				elog(ERROR, "failed to add pgturbohybrid graph correction item to \"%s\"", RelationGetRelationName(state->index));
			PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT);
		}
	}

	if (BufferIsValid(buf))
		PgturbohybridGraphFinishPage(buf);

	pfree(tuple);
	return start;
}

static void
PgturbohybridGraphWriteGraphDataPages(PgturbohybridQuantBuildState *state, BlockNumber *codeStart,
						   BlockNumber *adjStart, BlockNumber *exactStart,
						   BlockNumber *correctionStart)
{
	if (state->nodeCount == 0)
	{
		*codeStart = InvalidBlockNumber;
		*adjStart = InvalidBlockNumber;
		*exactStart = InvalidBlockNumber;
		*correctionStart = InvalidBlockNumber;
		return;
	}

	if (state->tqExactStorage)
		*exactStart = PgturbohybridGraphWriteExactPages(state);
	else
	{
		*exactStart = InvalidBlockNumber;
		for (uint32 i = 0; i < state->nodeCount; i++)
		{
			CHECK_FOR_INTERRUPTS();
			state->nodes[i].exactBlkno = InvalidBlockNumber;
			state->nodes[i].exactOffno = InvalidOffsetNumber;
		}
	}
	*correctionStart = PgturbohybridGraphWriteCorrectionPages(state);
	*codeStart = PgturbohybridGraphWriteCodePages(state);
	*adjStart = PgturbohybridGraphWriteAdjPages(state);
}

static void
PgturbohybridGraphWriteGraphPages(PgturbohybridQuantBuildState *state)
{
	BlockNumber codeStart;
	BlockNumber adjStart;
	BlockNumber exactStart;
	BlockNumber correctionStart;

	PgturbohybridGraphWriteGraphDataPages(state, &codeStart, &adjStart, &exactStart,
							   &correctionStart);
	PgturbohybridQuantUpdateMetaPage(state->index, state, codeStart, adjStart, exactStart,
						  correctionStart);
}

static void
PgturbohybridGraphWriteIndex(PgturbohybridQuantBuildState *state)
{
	PgturbohybridGraphCreateMetaPage(state->index, state->forkNum);
	PgturbohybridGraphWriteGraphPages(state);
}

static int64
PgturbohybridGraphElapsedUs(instr_time start)
{
	instr_time	duration;

	INSTR_TIME_SET_CURRENT(duration);
	INSTR_TIME_SUBTRACT(duration, start);
	return (int64) INSTR_TIME_GET_MICROSEC(duration);
}

static void
PgturbohybridGraphDebugBuildPhaseStart(PgturbohybridQuantBuildState *state, const char *phase)
{
	elog(DEBUG1, "pgturbohybrid native graph build phase start: relation=%s phase=%s nodes=%u dimensions=%d m=%d ef_construction=%d score_mode=%d",
		 RelationGetRelationName(state->index), phase, state->nodeCount,
		 state->dimensions, state->m, state->efConstruction, state->scoreMode);
}

static void
PgturbohybridGraphDebugBuildPhaseDone(PgturbohybridQuantBuildState *state, const char *phase,
						   instr_time phaseStart)
{
	elog(DEBUG1, "pgturbohybrid native graph build phase done: relation=%s phase=%s elapsed_ms=%.3f nodes=%u",
		 RelationGetRelationName(state->index), phase,
		 (double) PgturbohybridGraphElapsedUs(phaseStart) / 1000.0,
		 state->nodeCount);
}

IndexBuildResult *
tqgraphbuild(Relation heap, Relation index, IndexInfo *indexInfo)
{
	PgturbohybridQuantBuildState state;
	IndexBuildResult *result;
	instr_time	totalStart;
	instr_time	phaseStart;
	int64		scanUs = 0;
	int64		correctionUs;
	int64		encodeUs;
	int64		edgesUs;
	int64		writeUs;
	int64		walUs = 0;

	memset(&state, 0, sizeof(state));
	INSTR_TIME_SET_CURRENT(totalStart);
	state.heap = heap;
	state.index = index;
	state.indexInfo = indexInfo;
	state.forkNum = MAIN_FORKNUM;
	state.building = true;
	state.typeInfo = PgturbohybridGraphGetTypeInfo(index);
	state.m = PgturbohybridGraphGetM(index);
	state.efConstruction = PgturbohybridGraphGetEfConstruction(index);
	state.tqBits = PgturbohybridGraphGetTqBits(index);
	state.tqWeighted = PgturbohybridGraphGetTqWeightedOption(index);
	state.tqQuantileFit = PgturbohybridGraphGetTqQuantileFitOption(index);
	state.tqRenorm = PgturbohybridGraphGetTqRenormOption(index);
	state.tqExactStorage = PgturbohybridGraphGetTqExactStorageOption(index);
	state.buildExactDistances = pgturbohybrid_dense_build_exact_distances;
	if (state.tqRenorm && state.tqBits >= PGTURBOHYBRID_DEFAULT_BITS)
		ereport(NOTICE,
				(errmsg("quantized renormalization has no measurable effect at quantization_bits = %d",
						state.tqBits),
				 errdetail("At 4-bit and above the Lloyd-Max codebook is fine-grained enough that the correction is within noise and only the encoder pays extra cost.")));
	else if (state.tqRenorm && state.tqBits == 1)
		ereport(NOTICE,
				(errmsg("quantized renormalization is not recommended at quantization_bits = 1"),
				 errdetail("At 1-bit the decoded vector carries only sign information; the correction can inject per-vector noise instead of correcting bias.")));
	state.payloadCount = PgturbohybridGraphIndexPayloadCount(index);
	state.payloadBytes = PgturbohybridGraphPayloadBytes(state.payloadCount);
	PgturbohybridGraphInitSupport(&state.support, index);
	state.scoreMode = PgturbohybridGraphGetScoreMode(&state.support);
	state.ctx = AllocSetContextCreate(CurrentMemoryContext,
									  "pgturbohybrid native graph build context",
									  ALLOCSET_DEFAULT_SIZES);
	state.nodes = MemoryContextAllocZero(state.ctx, sizeof(PgturbohybridGraphBuildNode) * 1024);
	state.nodeCapacity = 1024;

	if (heap != NULL)
	{
		PgturbohybridGraphDebugBuildPhaseStart(&state, "scan");
		INSTR_TIME_SET_CURRENT(phaseStart);
		state.reltuples = table_index_build_scan(heap, index, indexInfo,
												 true, true, PgturbohybridGraphBuildCallback, &state, NULL);
		scanUs = PgturbohybridGraphElapsedUs(phaseStart);
		PgturbohybridGraphDebugBuildPhaseDone(&state, "scan", phaseStart);
	}

	PgturbohybridGraphDebugBuildPhaseStart(&state, "fit_correction");
	INSTR_TIME_SET_CURRENT(phaseStart);
	PgturbohybridGraphFitCorrection(&state);
	correctionUs = PgturbohybridGraphElapsedUs(phaseStart);
	PgturbohybridGraphDebugBuildPhaseDone(&state, "fit_correction", phaseStart);

	PgturbohybridGraphDebugBuildPhaseStart(&state, "encode");
	INSTR_TIME_SET_CURRENT(phaseStart);
	PgturbohybridGraphEncodeBuildNodes(&state);
	encodeUs = PgturbohybridGraphElapsedUs(phaseStart);
	PgturbohybridGraphDebugBuildPhaseDone(&state, "encode", phaseStart);

	PgturbohybridGraphDebugBuildPhaseStart(&state, "build_edges");
	INSTR_TIME_SET_CURRENT(phaseStart);
	PgturbohybridGraphBuildEdges(&state);
	edgesUs = PgturbohybridGraphElapsedUs(phaseStart);
	PgturbohybridGraphDebugBuildPhaseDone(&state, "build_edges", phaseStart);

	PgturbohybridGraphDebugBuildPhaseStart(&state, "reorder_nodes");
	INSTR_TIME_SET_CURRENT(phaseStart);
	PgturbohybridGraphReorderBuildNodesForLocality(&state);
	PgturbohybridGraphDebugBuildPhaseDone(&state, "reorder_nodes", phaseStart);

	PgturbohybridGraphDebugBuildPhaseStart(&state, "write_pages");
	INSTR_TIME_SET_CURRENT(phaseStart);
	PgturbohybridGraphWriteIndex(&state);
	writeUs = PgturbohybridGraphElapsedUs(phaseStart);
	PgturbohybridGraphDebugBuildPhaseDone(&state, "write_pages", phaseStart);

	if (RelationNeedsWAL(index))
	{
		PgturbohybridGraphDebugBuildPhaseStart(&state, "wal_newpages");
		INSTR_TIME_SET_CURRENT(phaseStart);
		log_newpage_range(index, MAIN_FORKNUM, 0, RelationGetNumberOfBlocks(index), true);
		walUs = PgturbohybridGraphElapsedUs(phaseStart);
		PgturbohybridGraphDebugBuildPhaseDone(&state, "wal_newpages", phaseStart);
	}

	elog(DEBUG1, "pgturbohybrid native graph build timings: relation=%s nodes=%u dimensions=%d scan_ms=%.3f fit_correction_ms=%.3f encode_ms=%.3f build_edges_ms=%.3f write_pages_ms=%.3f wal_ms=%.3f total_ms=%.3f",
		 RelationGetRelationName(index), state.nodeCount, state.dimensions,
		 (double) scanUs / 1000.0,
		 (double) correctionUs / 1000.0,
		 (double) encodeUs / 1000.0,
		 (double) edgesUs / 1000.0,
		 (double) writeUs / 1000.0,
		 (double) walUs / 1000.0,
		 (double) PgturbohybridGraphElapsedUs(totalStart) / 1000.0);
	elog(DEBUG1, "pgturbohybrid native graph build distance paths: relation=%s calls=%llu query_split=%llu packed=%llu weighted=%llu code_code=%llu exact=%llu fallback=%llu",
		 RelationGetRelationName(index),
		 (unsigned long long) state.buildDistanceCalls,
		 (unsigned long long) state.buildDistanceQuerySplit,
		 (unsigned long long) state.buildDistancePacked,
		 (unsigned long long) state.buildDistanceWeighted,
		 (unsigned long long) state.buildDistanceCodeCode,
		 (unsigned long long) state.buildDistanceExact,
		 (unsigned long long) state.buildDistanceFallback);

	result = palloc0(sizeof(IndexBuildResult));
	result->heap_tuples = state.reltuples;
	result->index_tuples = state.nodeCount;

	MemoryContextDelete(state.ctx);

	return result;
}

void
tqgraphbuildempty(Relation index)
{
	PgturbohybridQuantBuildState state;

	memset(&state, 0, sizeof(state));
	state.index = index;
	state.forkNum = INIT_FORKNUM;
	state.building = true;
	state.m = PgturbohybridGraphGetM(index);
	state.efConstruction = PgturbohybridGraphGetEfConstruction(index);
	state.tqBits = PgturbohybridGraphGetTqBits(index);
	state.tqWeighted = PgturbohybridGraphGetTqWeightedOption(index);
	state.tqQuantileFit = PgturbohybridGraphGetTqQuantileFitOption(index);
	state.tqRenorm = PgturbohybridGraphGetTqRenormOption(index);
	state.tqExactStorage = PgturbohybridGraphGetTqExactStorageOption(index);
	state.buildExactDistances = pgturbohybrid_dense_build_exact_distances;
	if (state.tqRenorm && state.tqBits >= PGTURBOHYBRID_DEFAULT_BITS)
		ereport(NOTICE,
				(errmsg("quantized renormalization has no measurable effect at quantization_bits = %d",
						state.tqBits),
				 errdetail("At 4-bit and above the Lloyd-Max codebook is fine-grained enough that the correction is within noise and only the encoder pays extra cost.")));
	else if (state.tqRenorm && state.tqBits == 1)
		ereport(NOTICE,
				(errmsg("quantized renormalization is not recommended at quantization_bits = 1"),
				 errdetail("At 1-bit the decoded vector carries only sign information; the correction can inject per-vector noise instead of correcting bias.")));
	state.payloadCount = PgturbohybridGraphIndexPayloadCount(index);
	state.payloadBytes = PgturbohybridGraphPayloadBytes(state.payloadCount);
	PgturbohybridGraphWriteIndex(&state);
	log_newpage_range(index, INIT_FORKNUM, 0, RelationGetNumberOfBlocksInFork(index, INIT_FORKNUM), true);
}

static void
PgturbohybridGraphAddElapsedUs(int64 *target, instr_time start)
{
	instr_time	duration;

	INSTR_TIME_SET_CURRENT(duration);
	INSTR_TIME_SUBTRACT(duration, start);
	*target += (int64) INSTR_TIME_GET_MICROSEC(duration);
}

static void
PgturbohybridGraphScoreNodeBatchTimed(PgturbohybridGraphScanOpaque so, PgturbohybridGraphScanStorage *storage,
						   uint32 *nodeIds, int count, double *distances,
						   Datum query)
{
	instr_time	start;

	INSTR_TIME_SET_CURRENT(start);
	PgturbohybridGraphScoreNodeBatch(so, storage, nodeIds, count, distances, query);
	PgturbohybridGraphAddElapsedUs(&so->graphBatchUs, start);
}

static void
PgturbohybridGraphResetScan(PgturbohybridGraphScanOpaque so)
{
	so->first = true;
	so->returnedRows = 0;
	so->graphVisitedNodes = 0;
	so->graphScoredCodes = 0;
	so->graphCandidateCount = 0;
	so->graphRescoreCount = 0;
	so->graphRescorePages = 0;
	so->graphCodePagesRead = 0;
	so->graphAdjPagesRead = 0;
	so->graphEntryPointCount = 0;
	so->graphPrepareUs = 0;
	so->graphTraverseUs = 0;
	so->graphEntryUs = 0;
	so->graphBaseUs = 0;
	so->graphBatchUs = 0;
	so->graphHeapUs = 0;
	so->graphFillUs = 0;
	so->graphRescoreUs = 0;
	so->graphSortUs = 0;
	so->graphTotalUs = 0;
	so->graphDenseRequestedK = 0;
	so->graphEffectiveResultTarget = 0;
	so->graphEffectiveSearchEf = 0;
	so->graphEffectiveRescoreBand = 0;
	so->graphHighdimWideningMultiplier = 1.0;
	so->graphWideningReason = PGTURBOHYBRID_DENSE_WIDENING_NONE;
	so->graphDenseBudgetPolicy = pgturbohybrid_dense_budget_policy;
	so->graphRescoreBandPolicy = pgturbohybrid_dense_rescore_band_policy;
	so->tqGraphResults = NULL;
	so->tqGraphResultCount = 0;
	so->tqGraphResultIndex = 0;
	so->hasTupleTargetRows = false;
	so->hasEstimatedFilterSelectivity = false;
	so->hasInitialEffectiveEfSearch = false;
	so->returnedRows = 0;
	so->tupleTargetRows = -1;
	so->estimatedFilterSelectivity = -1.0;
	memset(&so->tq, 0, sizeof(PgturbohybridGraphTqQuery));
	MemoryContextReset(so->tmpCtx);
}

static bool
PgturbohybridGraphHasFilter(bool hasPayloadFilter, double estimatedSelectivity)
{
	return hasPayloadFilter ||
		(estimatedSelectivity > 0 && estimatedSelectivity < 1);
}

static bool
PgturbohybridGraphUseLatencyDenseBudget(PgturbohybridGraphMetaPageData *meta, bool hasPayloadFilter,
							 double estimatedSelectivity)
{
	switch ((TqDenseBudgetPolicy) pgturbohybrid_dense_budget_policy)
	{
		case PGTURBOHYBRID_DENSE_BUDGET_QUALITY:
			return false;
		case PGTURBOHYBRID_DENSE_BUDGET_LATENCY:
		case PGTURBOHYBRID_DENSE_BUDGET_BALANCED:
			return !PgturbohybridGraphHasFilter(hasPayloadFilter, estimatedSelectivity);
		case PGTURBOHYBRID_DENSE_BUDGET_AUTO:
		default:
			return meta->dimensions >= 1024 &&
				!PgturbohybridGraphHasFilter(hasPayloadFilter, estimatedSelectivity);
	}
}

static double
PgturbohybridGraphDenseBudgetMultiplier(void)
{
	switch ((TqDenseBudgetPolicy) pgturbohybrid_dense_budget_policy)
	{
		case PGTURBOHYBRID_DENSE_BUDGET_BALANCED:
			return Max(2.0, pgturbohybrid_dense_latency_multiplier);
		case PGTURBOHYBRID_DENSE_BUDGET_LATENCY:
		case PGTURBOHYBRID_DENSE_BUDGET_AUTO:
			return pgturbohybrid_dense_latency_multiplier;
		case PGTURBOHYBRID_DENSE_BUDGET_QUALITY:
		default:
			return 0.0;
	}
}

static const char *
PgturbohybridGraphDenseBudgetPolicyName(int policy)
{
	switch ((TqDenseBudgetPolicy) policy)
	{
		case PGTURBOHYBRID_DENSE_BUDGET_QUALITY:
			return "quality";
		case PGTURBOHYBRID_DENSE_BUDGET_BALANCED:
			return "balanced";
		case PGTURBOHYBRID_DENSE_BUDGET_LATENCY:
			return "latency";
		case PGTURBOHYBRID_DENSE_BUDGET_AUTO:
		default:
			return "auto";
	}
}

static const char *
PgturbohybridGraphRescoreBandPolicyName(int policy)
{
	switch ((TqRescoreBandPolicy) policy)
	{
		case PGTURBOHYBRID_RESCORE_BAND_POLICY_EXACT:
			return "exact";
		case PGTURBOHYBRID_RESCORE_BAND_POLICY_LIMITED:
			return "limited";
		case PGTURBOHYBRID_RESCORE_BAND_POLICY_AUTO:
			return "auto";
		case PGTURBOHYBRID_RESCORE_BAND_POLICY_OFF:
		default:
			return "off";
	}
}

const char *
PgturbohybridGraphDenseWideningReasonName(int reason)
{
	switch ((TqDenseWideningReason) reason)
	{
		case PGTURBOHYBRID_DENSE_WIDENING_DIMENSION:
			return "dimension";
		case PGTURBOHYBRID_DENSE_WIDENING_FILTER:
			return "filter";
		case PGTURBOHYBRID_DENSE_WIDENING_EXACT_POLICY:
			return "exact_policy";
		case PGTURBOHYBRID_DENSE_WIDENING_NONE:
		default:
			return "none";
	}
}

const char *
PgturbohybridGraphDenseBudgetPolicyNameExternal(int policy)
{
	return PgturbohybridGraphDenseBudgetPolicyName(policy);
}

const char *
PgturbohybridGraphRescoreBandPolicyNameExternal(int policy)
{
	return PgturbohybridGraphRescoreBandPolicyName(policy);
}

IndexScanDesc
tqgraphbeginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	PgturbohybridGraphScanOpaque so;

	scan = RelationGetIndexScan(index, nkeys, norderbys);
	so = palloc0(sizeof(PgturbohybridGraphScanOpaqueData));
	so->typeInfo = PgturbohybridGraphGetTypeInfo(index);
	PgturbohybridGraphInitSupport(&so->support, index);
	so->tmpCtx = AllocSetContextCreate(CurrentMemoryContext,
									   "pgturbohybrid native graph scan context",
									   0, 8 * 1024, 256 * 1024);
	so->efSearch = PgturbohybridGraphGetEfSearch(index);
	so->graphOversampling = PgturbohybridGraphGetGraphOversampling(index);
	so->graphRescoreBand = PgturbohybridGraphGetGraphRescoreBand(index);
	so->graphStorageKind = PGTURBOHYBRID_GRAPH_STORAGE_QUANT_GRAPH_NATIVE;
	so->pgturbohybridGraphScan = true;
	PgturbohybridGraphResetScan(so);
	scan->opaque = so;

	return scan;
}

void
tqgraphrescan(IndexScanDesc scan, ScanKey keys, int nkeys, ScanKey orderbys, int norderbys)
{
	PgturbohybridGraphScanOpaque so = (PgturbohybridGraphScanOpaque) scan->opaque;

	if (keys && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys, scan->numberOfKeys * sizeof(ScanKeyData));

	if (orderbys && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys, scan->numberOfOrderBys * sizeof(ScanKeyData));

	so->efSearch = PgturbohybridGraphGetEfSearch(scan->indexRelation);
	so->graphOversampling = PgturbohybridGraphGetGraphOversampling(scan->indexRelation);
	so->graphRescoreBand = PgturbohybridGraphGetGraphRescoreBand(scan->indexRelation);
	so->graphStorageKind = PGTURBOHYBRID_GRAPH_STORAGE_QUANT_GRAPH_NATIVE;
	so->pgturbohybridGraphScan = true;
	PgturbohybridGraphResetScan(so);
}

static Datum
PgturbohybridGraphGetScanValue(IndexScanDesc scan, PgturbohybridGraphScanOpaque so)
{
	Datum		value;

	if (scan->orderByData == NULL || scan->numberOfOrderBys < 1)
		elog(ERROR, "cannot scan pgturbohybrid graph index without order");

	value = scan->orderByData[0].sk_argument;
	if (DatumGetPointer(value) == NULL)
		return value;

	value = PointerGetDatum(PG_DETOAST_DATUM(value));

	if (so->support.normprocinfo != NULL)
	{
		if (!PgturbohybridGraphCheckNorm(&so->support, value))
			value = PointerGetDatum(NULL);
		else
			value = PgturbohybridGraphNormValue(so->typeInfo, so->support.collation, value);
	}

	return value;
}

static int
PgturbohybridGraphResultCompare(const void *a, const void *b)
{
	const PgturbohybridGraphResult *ra = (const PgturbohybridGraphResult *) a;
	const PgturbohybridGraphResult *rb = (const PgturbohybridGraphResult *) b;

	if (ra->distance < rb->distance)
		return -1;
	if (ra->distance > rb->distance)
		return 1;
	return (ra->nodeId > rb->nodeId) - (ra->nodeId < rb->nodeId);
}

static int
PgturbohybridGraphRescoreRefCompare(const void *a, const void *b)
{
	const PgturbohybridGraphRescoreRef *ra = (const PgturbohybridGraphRescoreRef *) a;
	const PgturbohybridGraphRescoreRef *rb = (const PgturbohybridGraphRescoreRef *) b;

	if (ra->blkno < rb->blkno)
		return -1;
	if (ra->blkno > rb->blkno)
		return 1;
	if (ra->offno < rb->offno)
		return -1;
	if (ra->offno > rb->offno)
		return 1;
	return (ra->nodeId > rb->nodeId) - (ra->nodeId < rb->nodeId);
}

static bool
PgturbohybridGraphResultLess(PgturbohybridGraphResult a, PgturbohybridGraphResult b)
{
	if (a.distance != b.distance)
		return a.distance < b.distance;
	return a.nodeId < b.nodeId;
}

static bool
PgturbohybridGraphResultGreater(PgturbohybridGraphResult a, PgturbohybridGraphResult b)
{
	if (a.distance != b.distance)
		return a.distance > b.distance;
	return a.nodeId > b.nodeId;
}

static void
PgturbohybridGraphResultSwap(PgturbohybridGraphResult *a, PgturbohybridGraphResult *b)
{
	PgturbohybridGraphResult tmp = *a;

	*a = *b;
	*b = tmp;
}

static void
PgturbohybridGraphResultHeapSiftUp(PgturbohybridGraphResult *heap, int idx)
{
	while (idx > 0)
	{
		int			parent = (idx - 1) / 2;

		if (!PgturbohybridGraphResultGreater(heap[idx], heap[parent]))
			break;

		PgturbohybridGraphResultSwap(&heap[idx], &heap[parent]);
		idx = parent;
	}
}

static void
PgturbohybridGraphResultHeapSiftDown(PgturbohybridGraphResult *heap, int count, int idx)
{
	for (;;)
	{
		int			left = idx * 2 + 1;
		int			right = left + 1;
		int			best = idx;

		if (left < count && PgturbohybridGraphResultGreater(heap[left], heap[best]))
			best = left;
		if (right < count && PgturbohybridGraphResultGreater(heap[right], heap[best]))
			best = right;
		if (best == idx)
			break;

		PgturbohybridGraphResultSwap(&heap[idx], &heap[best]);
		idx = best;
	}
}

static void
PgturbohybridGraphResultHeapPush(PgturbohybridGraphResult *heap, int *count, PgturbohybridGraphResult item)
{
	heap[*count] = item;
	(*count)++;
	PgturbohybridGraphResultHeapSiftUp(heap, *count - 1);
}

static void
PgturbohybridGraphResultHeapReplaceRoot(PgturbohybridGraphResult *heap, int count,
							 PgturbohybridGraphResult item)
{
	heap[0] = item;
	PgturbohybridGraphResultHeapSiftDown(heap, count, 0);
}

static void
PgturbohybridGraphOfferCandidate(PgturbohybridGraphScanOpaque so, PgturbohybridGraphResult *results, int target,
					  int *count, uint32 nodeId, ItemPointer heaptid,
					  double distance, bool exactScored)
{
	PgturbohybridGraphResult item;

	if (target <= 0)
		return;

	item.nodeId = nodeId;
	item.heaptid = *heaptid;
	item.distance = distance;
	item.exactScored = exactScored;

	if (*count < target)
	{
		PgturbohybridGraphResultHeapPush(results, count, item);
		if (exactScored)
			so->graphRescoreCount++;
		return;
	}

	if (PgturbohybridGraphResultLess(item, results[0]))
	{
		if (results[0].exactScored)
			so->graphRescoreCount--;
		PgturbohybridGraphResultHeapReplaceRoot(results, *count, item);
		if (exactScored)
			so->graphRescoreCount++;
	}
}

bool
PgturbohybridGraphReadMeta(Relation index, PgturbohybridGraphMetaPageData *meta)
{
	Buffer		buf;
	Page		page;
	PgturbohybridGraphMetaPage metap;

	buf = ReadBuffer(index, PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	metap = PgturbohybridGraphPageGetMeta(page);

	if (metap->magicNumber != PGTURBOHYBRID_GRAPH_MAGIC_NUMBER ||
		metap->storageKind != PGTURBOHYBRID_GRAPH_STORAGE_QUANT_GRAPH_NATIVE)
	{
		UnlockReleaseBuffer(buf);
		return false;
	}

	memcpy(meta, metap, sizeof(PgturbohybridGraphMetaPageData));
	if (meta->tqBits != 1 && meta->tqBits != 2 && meta->tqBits != PGTURBOHYBRID_DEFAULT_BITS)
		meta->tqBits = PGTURBOHYBRID_DEFAULT_BITS;
	if (meta->tqBm25MetaStartBlkno <= PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO)
		meta->tqBm25MetaStartBlkno = InvalidBlockNumber;
	UnlockReleaseBuffer(buf);
	return true;
}




static int
PgturbohybridGraphScanAdjSlot(PgturbohybridGraphMetaPageData *meta, uint32 nodeId, int level)
{
	return PgturbohybridGraphAdjSlot(meta, nodeId, level);
}

static bool
PgturbohybridGraphEntryAlreadySelected(PgturbohybridGraphFrontierItem *entries, int entryCount,
							uint32 nodeId)
{
	for (int i = 0; i < entryCount; i++)
	{
		if (entries[i].nodeId == nodeId)
			return true;
	}

	return false;
}

static void
PgturbohybridGraphOfferLevelEntry(PgturbohybridGraphFrontierItem *entries, int *entryCount,
					   int *entryLevels, uint32 nodeId, int level)
{
	int			worst = 0;

	if (PgturbohybridGraphEntryAlreadySelected(entries, *entryCount, nodeId))
		return;

	if (*entryCount < PGTURBOHYBRID_GRAPH_MAX_ENTRY_POINTS)
	{
		entries[*entryCount].nodeId = nodeId;
		entries[*entryCount].distance = DBL_MAX;
		entryLevels[*entryCount] = level;
		(*entryCount)++;
		return;
	}

	for (int i = 1; i < *entryCount; i++)
	{
		if (entryLevels[i] < entryLevels[worst] ||
			(entryLevels[i] == entryLevels[worst] &&
			 entries[i].nodeId > entries[worst].nodeId))
			worst = i;
	}

	if (level > entryLevels[worst] ||
		(level == entryLevels[worst] && nodeId < entries[worst].nodeId))
	{
		entries[worst].nodeId = nodeId;
		entries[worst].distance = DBL_MAX;
		entryLevels[worst] = level;
	}
}

static void
PgturbohybridGraphOfferDistanceEntry(PgturbohybridGraphFrontierItem *entries, int *entryCount,
						  PgturbohybridGraphFrontierItem entry)
{
	int			worst = 0;

	for (int i = 0; i < *entryCount; i++)
	{
		if (entries[i].nodeId == entry.nodeId)
		{
			entries[i].distance = Min(entries[i].distance, entry.distance);
			return;
		}
	}

	if (*entryCount < PGTURBOHYBRID_GRAPH_MAX_ENTRY_POINTS)
	{
		entries[*entryCount] = entry;
		(*entryCount)++;
		return;
	}

	for (int i = 1; i < *entryCount; i++)
	{
		if (entries[i].distance > entries[worst].distance)
			worst = i;
	}

	if (entry.distance < entries[worst].distance)
		entries[worst] = entry;
}

static PgturbohybridGraphFrontierItem
PgturbohybridGraphScanGreedySearch(Relation index, PgturbohybridGraphScanOpaque so, Datum query,
						PgturbohybridGraphMetaPageData *meta,
						PgturbohybridGraphScanStorage *storage, PgturbohybridGraphFrontierItem entry,
						int level)
{
	PgturbohybridGraphFrontierItem current = entry;
	bool		changed = true;
	int			maxNeighbors = PgturbohybridGraphLevelM(meta->m, level);
	uint32		batchNodeIds[PGTURBOHYBRID_GRAPH_MAX_NEIGHBORS];
	double		batchDistances[PGTURBOHYBRID_GRAPH_MAX_NEIGHBORS];
	bool		lookaheadPrefetch;

	if (maxNeighbors > PGTURBOHYBRID_GRAPH_MAX_NEIGHBORS)
		elog(ERROR, "pgturbohybrid graph neighbor batch exceeds fixed capacity");

	/* Size-gated look-ahead prefetch (see PgturbohybridGraphSearchBaseLayer). */
	if (pgturbohybrid_dense_graph_lookahead_prefetch == PGTURBOHYBRID_GRAPH_LOOKAHEAD_OFF)
		lookaheadPrefetch = false;
	else if (pgturbohybrid_dense_graph_lookahead_prefetch == PGTURBOHYBRID_GRAPH_LOOKAHEAD_ON)
		lookaheadPrefetch = true;
	else
	{
		Size		workingSetBytes = (Size) meta->tqNodeCount *
			(sizeof(PgturbohybridGraphScanNode) + sizeof(uint32));

		lookaheadPrefetch = workingSetBytes >
			(Size) pgturbohybrid_dense_graph_lookahead_threshold_kb * 1024;
	}

	while (changed)
	{
		int			slot;
		int			batchCount = 0;

		changed = false;
		if (!PgturbohybridGraphLoadAdjPage(index, so, meta, storage, current.nodeId, level))
			break;

		slot = PgturbohybridGraphScanAdjSlot(meta, current.nodeId, level);
		for (int i = 0; i < storage->neighborCounts[slot]; i++)
		{
			uint32		neighbor = storage->neighbors[slot][i];

			if (lookaheadPrefetch && i + 1 < storage->neighborCounts[slot])
			{
				uint32		la = storage->neighbors[slot][i + 1];

				if (la < meta->tqNodeCount)
					PGTURBOHYBRID_GRAPH_PREFETCH_READ(&storage->nodes[la]);
			}

			if (neighbor >= meta->tqNodeCount ||
				!PgturbohybridGraphLoadCodePage(index, so, meta, storage, neighbor) ||
				storage->nodes[neighbor].level < level)
				continue;

			batchNodeIds[batchCount++] = neighbor;
		}

		if (DatumGetPointer(query) != NULL)
		{
			for (int i = 0; i < batchCount; i++)
				batchDistances[i] =
					PgturbohybridGraphEntryDistance(so, query,
										 &storage->nodes[batchNodeIds[i]]);
		}
		else
			PgturbohybridGraphScoreNodeBatchTimed(so, storage, batchNodeIds, batchCount,
									   batchDistances, query);

		for (int i = 0; i < batchCount; i++)
		{
			if (batchDistances[i] < current.distance)
			{
				current.nodeId = batchNodeIds[i];
				current.distance = batchDistances[i];
				changed = true;
			}
		}
	}

	return current;
}

static int
PgturbohybridGraphSearchBaseLayer(Relation index, PgturbohybridGraphScanOpaque so, Datum query,
					   PgturbohybridGraphMetaPageData *meta,
					   PgturbohybridGraphScanStorage *storage,
					   PgturbohybridGraphFrontierItem *entries, int entryCount,
					   PgturbohybridGraphResult *results, int resultTarget, int searchEf,
					   int payloadSlot, int32 payloadValue)
{
	bool	   *visited = NULL;
	uint32		visitGeneration = 0;
	bool		useVisitGeneration = storage->visitedGeneration != NULL &&
		storage->visitGeneration != NULL;
	int			frontierCount = 0;
	int			nearestCount = 0;
	int			resultCount = 0;
	int			maxNeighbors = PgturbohybridGraphLevelM(meta->m, 0);
	int			frontierCapacity = PgturbohybridGraphInitialFrontierCapacity(meta->tqNodeCount,
																 searchEf,
																 entryCount,
																 maxNeighbors);
	int			maxFrontierCapacity = (int) meta->tqNodeCount;
	PgturbohybridGraphFrontierItem stackFrontier[PGTURBOHYBRID_GRAPH_STACK_FRONTIER_CAPACITY];
	PgturbohybridGraphFrontierItem stackNearest[PGTURBOHYBRID_GRAPH_STACK_FRONTIER_CAPACITY];
	PgturbohybridGraphFrontierItem *frontier =
		pgturbohybrid_dense_graph_stack_scratch &&
		maxFrontierCapacity <= PGTURBOHYBRID_GRAPH_STACK_FRONTIER_CAPACITY ?
		stackFrontier : palloc(sizeof(PgturbohybridGraphFrontierItem) * frontierCapacity);
	PgturbohybridGraphFrontierItem *nearest =
		pgturbohybrid_dense_graph_stack_scratch &&
		searchEf <= PGTURBOHYBRID_GRAPH_STACK_FRONTIER_CAPACITY ?
		stackNearest : palloc(sizeof(PgturbohybridGraphFrontierItem) * searchEf);
	bool		frontierAllocated = frontier != stackFrontier;
	bool		nearestAllocated = nearest != stackNearest;
	uint32		batchNodeIds[PGTURBOHYBRID_GRAPH_MAX_NEIGHBORS];
	double		batchDistances[PGTURBOHYBRID_GRAPH_MAX_NEIGHBORS];
	bool		lookaheadPrefetch;

	if (maxNeighbors > PGTURBOHYBRID_GRAPH_MAX_NEIGHBORS)
		elog(ERROR, "pgturbohybrid graph neighbor batch exceeds fixed capacity");

	/*
	 * Size-gated adjacency-list look-ahead prefetch.  The
	 * original FAISS-style prefetch was reverted on FIQA-57k (commit
	 * 67f38bd) because the metadata working set fits in cache and
	 * explicit __builtin_prefetch was paid-for-nothing uops,
	 * regressing p50.  Auto mode here gates on a corpus-size
	 * threshold: when (storage->nodes + visitedGeneration) bytes
	 * exceeds the private look-ahead threshold the hint kicks in.
	 * On FIQA-scale (57k × ~64 B ≈ 3.6 MB << 24 MB default) auto
	 * stays off (no regression).  On 1M+ corpora (>> 64 MB) auto
	 * turns on.  off / on bypass the gate.
	 */
	if (pgturbohybrid_dense_graph_lookahead_prefetch == PGTURBOHYBRID_GRAPH_LOOKAHEAD_OFF)
		lookaheadPrefetch = false;
	else if (pgturbohybrid_dense_graph_lookahead_prefetch == PGTURBOHYBRID_GRAPH_LOOKAHEAD_ON)
		lookaheadPrefetch = true;
	else
	{
		Size		workingSetBytes = (Size) meta->tqNodeCount *
			(sizeof(PgturbohybridGraphScanNode) + sizeof(uint32));

		lookaheadPrefetch = workingSetBytes >
			(Size) pgturbohybrid_dense_graph_lookahead_threshold_kb * 1024;
	}

	if (useVisitGeneration)
	{
		visitGeneration = ++(*storage->visitGeneration);
		if (visitGeneration == 0)
		{
			memset(storage->visitedGeneration, 0, sizeof(uint32) * meta->tqNodeCount);
			visitGeneration = ++(*storage->visitGeneration);
		}
	}
	else
		visited = palloc0(sizeof(bool) * meta->tqNodeCount);

	for (int i = 0; i < entryCount; i++)
	{
		PgturbohybridGraphFrontierItem entry = entries[i];
		instr_time	heapStart;

		if (entry.nodeId >= meta->tqNodeCount ||
			(useVisitGeneration ?
			 storage->visitedGeneration[entry.nodeId] == visitGeneration :
			 visited[entry.nodeId]))
			continue;

		if (useVisitGeneration)
			storage->visitedGeneration[entry.nodeId] = visitGeneration;
		else
			visited[entry.nodeId] = true;

		INSTR_TIME_SET_CURRENT(heapStart);
		PgturbohybridGraphFrontierHeapPushGrowing(&frontier, &frontierCount, &frontierCapacity,
									   maxFrontierCapacity, entry, true);
		(void) PgturbohybridGraphOfferNearest(nearest, searchEf, &nearestCount, entry.nodeId, entry.distance);
		PgturbohybridGraphAddElapsedUs(&so->graphHeapUs, heapStart);
	}

	while (frontierCount > 0)
	{
		PgturbohybridGraphFrontierItem item = PgturbohybridGraphFrontierHeapPop(frontier, &frontierCount, true);
		uint32		nodeId = item.nodeId;
		int			slot;

		if (!PgturbohybridGraphLoadCodePage(index, so, meta, storage, nodeId))
			continue;

		if (nearestCount >= searchEf && PgturbohybridGraphFrontierGreater(item, nearest[0]))
			break;

		so->graphVisitedNodes++;

		if (!PgturbohybridGraphLoadAdjPage(index, so, meta, storage, nodeId, 0))
			continue;

		slot = PgturbohybridGraphScanAdjSlot(meta, nodeId, 0);
		{
			int			batchCount = 0;

			for (int i = 0; i < storage->neighborCounts[slot]; i++)
			{
				uint32		neighbor = storage->neighbors[slot][i];

				/*
				 * Look-ahead prefetch — issue a HW prefetch
				 * for the *next* iteration's storage->nodes[] and
				 * visitedGeneration[] entries while the current
				 * iteration's load + visit-test stalls.  Gated on
				 * lookaheadPrefetch (size-aware) so it stays off on
				 * small corpora where the HW prefetcher already
				 * covers the access.
				 */
				if (lookaheadPrefetch && i + 1 < storage->neighborCounts[slot])
				{
					uint32		la = storage->neighbors[slot][i + 1];

					if (la < meta->tqNodeCount)
					{
						PGTURBOHYBRID_GRAPH_PREFETCH_READ(&storage->nodes[la]);
						if (useVisitGeneration)
							PGTURBOHYBRID_GRAPH_PREFETCH_READ(&storage->visitedGeneration[la]);
					}
				}

				if (neighbor >= meta->tqNodeCount ||
					(useVisitGeneration ?
					 storage->visitedGeneration[neighbor] == visitGeneration :
					 visited[neighbor]))
					continue;

				if (!PgturbohybridGraphLoadCodePage(index, so, meta, storage, neighbor))
					continue;

				if (useVisitGeneration)
					storage->visitedGeneration[neighbor] = visitGeneration;
				else
					visited[neighbor] = true;
				batchNodeIds[batchCount++] = neighbor;
			}

			PgturbohybridGraphScoreNodeBatchTimed(so, storage, batchNodeIds, batchCount,
									   batchDistances, query);
			for (int i = 0; i < batchCount; i++)
			{
				uint32		neighbor = batchNodeIds[i];
				double		neighborDistance = batchDistances[i];
				bool		accepted;
				instr_time	heapStart;

				INSTR_TIME_SET_CURRENT(heapStart);
				accepted = PgturbohybridGraphOfferNearest(nearest, searchEf, &nearestCount,
											   neighbor, neighborDistance);
				if (accepted)
				{
					PgturbohybridGraphFrontierItem frontierItem;

					frontierItem.nodeId = neighbor;
					frontierItem.distance = neighborDistance;
					PgturbohybridGraphFrontierHeapPushGrowing(&frontier, &frontierCount,
												   &frontierCapacity,
												   maxFrontierCapacity,
												   frontierItem, true);
				}
				PgturbohybridGraphAddElapsedUs(&so->graphHeapUs, heapStart);
			}
		}
	}

	for (int i = 0; i < nearestCount; i++)
	{
		uint32		nodeId = nearest[i].nodeId;
		PgturbohybridGraphScanNode *node;
		bool		exactScored;
		double		resultDistance;

		if (nodeId >= meta->tqNodeCount ||
			!PgturbohybridGraphLoadCodePage(index, so, meta, storage, nodeId))
			continue;

		node = &storage->nodes[nodeId];
		if (node->flags & PGTURBOHYBRID_GRAPH_NODE_DEAD ||
			!PgturbohybridGraphNodeMatchesPayload(node, payloadSlot, payloadValue))
			continue;

		resultDistance = PgturbohybridGraphResultDistance(so, query, node,
											  nearest[i].distance,
											  &exactScored);
		{
			instr_time	heapStart;

			INSTR_TIME_SET_CURRENT(heapStart);
			PgturbohybridGraphOfferCandidate(so, results, resultTarget, &resultCount,
								  nodeId, &node->heaptid, resultDistance,
								  exactScored);
			PgturbohybridGraphAddElapsedUs(&so->graphHeapUs, heapStart);
		}
	}

	if (visited != NULL)
		pfree(visited);
	if (frontierAllocated)
		pfree(frontier);
	if (nearestAllocated)
		pfree(nearest);
	return resultCount;
}

int
PgturbohybridGraphTraverse(Relation index, PgturbohybridGraphScanOpaque so, PgturbohybridGraphMetaPageData *meta,
				PgturbohybridGraphScanStorage *storage, PgturbohybridGraphResult *results,
				int resultTarget, int searchEf, Datum query,
				int payloadSlot, int32 payloadValue)
{
	uint32		entryNodeId = meta->tqEntryNodeId < meta->tqNodeCount ? meta->tqEntryNodeId : 0;
	PgturbohybridGraphFrontierItem entry;
	PgturbohybridGraphFrontierItem entries[PGTURBOHYBRID_GRAPH_MAX_ENTRY_POINTS];
	int			entryLevels[PGTURBOHYBRID_GRAPH_MAX_ENTRY_POINTS];
	PgturbohybridGraphFrontierItem sampled[PGTURBOHYBRID_GRAPH_ENTRY_SAMPLE_COUNT];
	uint32		sampledNodeIds[PGTURBOHYBRID_GRAPH_ENTRY_SAMPLE_COUNT];
	double		sampledDistances[PGTURBOHYBRID_GRAPH_ENTRY_SAMPLE_COUNT];
	int			entryCount = 0;
	int			sampledCount = 0;
	instr_time	phaseStart;

	if (meta->tqNodeCount == 0 ||
		!PgturbohybridGraphLoadCodePage(index, so, meta, storage, entryNodeId))
		return 0;

	INSTR_TIME_SET_CURRENT(phaseStart);
	entry.nodeId = entryNodeId;
	entry.distance = PgturbohybridGraphEntryDistance(so, query, &storage->nodes[entryNodeId]);

	for (int level = meta->graphMaxLevel; level > 0; level--)
	{
		if (storage->nodes[entry.nodeId].level >= level)
			entry = PgturbohybridGraphScanGreedySearch(index, so, query, meta, storage, entry,
											level);
	}

	entries[entryCount] = entry;
	entryLevels[entryCount] = storage->nodes[entry.nodeId].level;
	entryCount++;

	/*
	 * Keep alternative high-level entry points instead of relying on one
	 * global entry. Build that shape from the compact native graph: score the
	 * best level-bearing nodes, then let deterministic sampled candidates
	 * replace weak high-level entries when the graph topology is not yet
	 * strong enough for pure hierarchical routing.
	 */
	for (uint32 nodeId = 0; nodeId < meta->tqNodeCount; nodeId++)
	{
		PgturbohybridGraphScanNode *node = &storage->nodes[nodeId];

		if (nodeId == entry.nodeId || node->flags & PGTURBOHYBRID_GRAPH_NODE_DEAD ||
			node->level <= 0)
			continue;

		PgturbohybridGraphOfferLevelEntry(entries, &entryCount, entryLevels, nodeId,
							   node->level);
	}

	if (entryCount > 1)
	{
		uint32		entryNodeIds[PGTURBOHYBRID_GRAPH_MAX_ENTRY_POINTS];
		double		entryDistances[PGTURBOHYBRID_GRAPH_MAX_ENTRY_POINTS];
		int			scoreCount = 0;

		for (int i = 1; i < entryCount; i++)
		{
			if (!PgturbohybridGraphLoadCodePage(index, so, meta, storage, entries[i].nodeId))
				continue;

			entryNodeIds[scoreCount++] = entries[i].nodeId;
		}

		PgturbohybridGraphScoreNodeBatchTimed(so, storage, entryNodeIds, scoreCount,
								   entryDistances, query);
		for (int i = 1; i < entryCount; i++)
		{
			for (int j = 0; j < scoreCount; j++)
			{
				if (entries[i].nodeId == entryNodeIds[j])
				{
					entries[i].distance = entryDistances[j];
					break;
				}
			}
		}
	}

	if (meta->tqNodeCount > 1)
	{
		int			sampleTarget = searchEf;
		int			sampleCount;

		if (so->tq.bits == PGTURBOHYBRID_DEFAULT_BITS &&
			(TqScoreMode) so->tq.scoreMode == PGTURBOHYBRID_SCORE_L2 &&
			so->tq.dimensions >= 1024)
			sampleTarget = Max(PGTURBOHYBRID_GRAPH_MAX_ENTRY_POINTS,
							   searchEf / PGTURBOHYBRID_GRAPH_HIGHDIM_ENTRY_SAMPLE_DIVISOR);

		sampleCount = Min((int) meta->tqNodeCount,
						  Min(PGTURBOHYBRID_GRAPH_ENTRY_SAMPLE_COUNT,
							  Max(sampleTarget,
								  PGTURBOHYBRID_GRAPH_MAX_ENTRY_POINTS)));

		for (int i = 0; i < sampleCount; i++)
		{
			uint32		nodeId = sampleCount == 1 ? 0 :
				(uint32) (((uint64) i * (meta->tqNodeCount - 1)) / (sampleCount - 1));
			bool		seen = PgturbohybridGraphEntryAlreadySelected(entries, entryCount,
															nodeId);

			for (int j = 0; j < sampledCount; j++)
			{
				if (sampled[j].nodeId == nodeId)
				{
					seen = true;
					break;
				}
			}

			if (seen || !PgturbohybridGraphLoadCodePage(index, so, meta, storage, nodeId))
				continue;

			sampled[sampledCount].nodeId = nodeId;
			sampledNodeIds[sampledCount] = nodeId;
			sampledCount++;
		}

		PgturbohybridGraphScoreNodeBatchTimed(so, storage, sampledNodeIds, sampledCount,
								   sampledDistances, query);
		for (int i = 0; i < sampledCount; i++)
		{
			double		exactDistance;

			if (PgturbohybridGraphExactHighdimEntryDistance(so, query,
												 &storage->nodes[sampledNodeIds[i]],
												 &exactDistance))
				sampledDistances[i] = exactDistance;
			sampled[i].distance = sampledDistances[i];
		}

		qsort(sampled, sampledCount, sizeof(PgturbohybridGraphFrontierItem),
			  PgturbohybridGraphFrontierCompare);
		for (int i = 0; i < sampledCount; i++)
			PgturbohybridGraphOfferDistanceEntry(entries, &entryCount, sampled[i]);
	}

	so->graphEntryPointCount = entryCount;
	PgturbohybridGraphAddElapsedUs(&so->graphEntryUs, phaseStart);

	INSTR_TIME_SET_CURRENT(phaseStart);
	entryCount = PgturbohybridGraphSearchBaseLayer(index, so, query, meta, storage, entries, entryCount,
										results, resultTarget, searchEf,
										payloadSlot, payloadValue);
	PgturbohybridGraphAddElapsedUs(&so->graphBaseUs, phaseStart);
	return entryCount;
}

static int
PgturbohybridGraphFillCandidateBand(Relation index, PgturbohybridGraphScanOpaque so,
						 PgturbohybridGraphMetaPageData *meta,
						 PgturbohybridGraphScanStorage *storage,
						 PgturbohybridGraphResult *results, int resultTarget, int count,
						 int payloadSlot, int32 payloadValue, Datum query)
{
	bool	   *selected;
	uint32		batchNodeIds[PGTURBOHYBRID_GRAPH_MAX_NEIGHBORS];
	double		batchDistances[PGTURBOHYBRID_GRAPH_MAX_NEIGHBORS];
	int			batchCount = 0;
	uint32		payloadFirst = 0;
	uint32		payloadCount = 0;
	bool		usePayloadRefs;

	if (count >= resultTarget || resultTarget <= 0)
		return count;

	selected = palloc0(sizeof(bool) * meta->tqNodeCount);
	for (int i = 0; i < count; i++)
	{
		if (results[i].nodeId < meta->tqNodeCount)
			selected[results[i].nodeId] = true;
	}

	usePayloadRefs = PgturbohybridGraphPayloadRefRange(storage, payloadSlot, payloadValue,
											&payloadFirst, &payloadCount);
	if (payloadSlot >= 0 && storage->payloadRefs != NULL && !usePayloadRefs)
	{
		pfree(selected);
		return count;
	}

	for (uint32 i = 0; i < (usePayloadRefs ? payloadCount : meta->tqNodeCount); i++)
	{
		uint32		nodeId = usePayloadRefs ?
			storage->payloadRefs[payloadFirst + i].nodeId : i;
		PgturbohybridGraphScanNode *node;

		if (selected[nodeId])
			continue;
		if (!PgturbohybridGraphLoadCodePage(index, so, meta, storage, nodeId))
			continue;

		node = &storage->nodes[nodeId];
		if (node->flags & PGTURBOHYBRID_GRAPH_NODE_DEAD)
			continue;
		if (!PgturbohybridGraphNodeMatchesPayload(node, payloadSlot, payloadValue))
			continue;

		batchNodeIds[batchCount++] = nodeId;
		if (batchCount == PGTURBOHYBRID_GRAPH_MAX_NEIGHBORS)
		{
			PgturbohybridGraphScoreNodeBatchTimed(so, storage, batchNodeIds, batchCount,
									   batchDistances, query);
			for (int i = 0; i < batchCount; i++)
			{
				node = &storage->nodes[batchNodeIds[i]];
				{
					instr_time	heapStart;

					INSTR_TIME_SET_CURRENT(heapStart);
					PgturbohybridGraphOfferCandidate(so, results, resultTarget, &count,
										  batchNodeIds[i], &node->heaptid,
										  batchDistances[i], false);
					PgturbohybridGraphAddElapsedUs(&so->graphHeapUs, heapStart);
				}
			}
			batchCount = 0;
		}
	}

	if (batchCount > 0)
	{
		PgturbohybridGraphScoreNodeBatchTimed(so, storage, batchNodeIds, batchCount,
								   batchDistances, query);
		for (int i = 0; i < batchCount; i++)
		{
			PgturbohybridGraphScanNode *node = &storage->nodes[batchNodeIds[i]];

			{
				instr_time	heapStart;

				INSTR_TIME_SET_CURRENT(heapStart);
				PgturbohybridGraphOfferCandidate(so, results, resultTarget, &count,
									  batchNodeIds[i], &node->heaptid,
									  batchDistances[i], false);
				PgturbohybridGraphAddElapsedUs(&so->graphHeapUs, heapStart);
			}
		}
	}

	pfree(selected);
	return count;
}

static bool
PgturbohybridGraphCollectPayloadExactBand(PgturbohybridGraphScanOpaque so, PgturbohybridGraphMetaPageData *meta,
							   PgturbohybridGraphScanStorage *storage, Datum query,
							   int payloadSlot, int32 payloadValue,
							   PgturbohybridGraphResult *results, int resultTarget,
							   int *count)
{
	uint32		payloadFirst;
	uint32		payloadCount;

	if (payloadSlot < 0 || resultTarget <= 0 ||
		!PgturbohybridGraphPayloadRefRange(storage, payloadSlot, payloadValue,
								&payloadFirst, &payloadCount))
		return false;

	if (payloadCount > PGTURBOHYBRID_GRAPH_PAYLOAD_EXACT_MAX)
		return false;

	*count = 0;
	for (uint32 i = 0; i < payloadCount; i++)
	{
		uint32		nodeId = storage->payloadRefs[payloadFirst + i].nodeId;
		PgturbohybridGraphScanNode *node;
		double		distance;
		bool		exactScored = false;

		if (nodeId >= meta->tqNodeCount)
			continue;

		node = &storage->nodes[nodeId];
		if (node->flags & PGTURBOHYBRID_GRAPH_NODE_DEAD ||
			!PgturbohybridGraphNodeMatchesPayload(node, payloadSlot, payloadValue))
			continue;

		if (node->exactVector != NULL)
		{
			distance = PgturbohybridGraphExactVectorDistance(so, query, node->exactVector);
			exactScored = true;
		}
		else
			distance = PgturbohybridGraphScoreNode(so, node);

		PgturbohybridGraphOfferCandidate(so, results, resultTarget, count, nodeId,
							  &node->heaptid, distance, exactScored);
	}

	so->graphEntryPointCount = 0;
	return true;
}

static int
PgturbohybridGraphFinalRescoreCount(PgturbohybridGraphScanOpaque so, PgturbohybridGraphResult *results, int count,
						 int effectiveEf)
{
	int64		limitRows;
	int64		denseK;
	int64		limitCap;
	int64		denseCap;
	int64		cap;

	(void) results;
	(void) effectiveEf;

	if (count <= 0 || so->graphRescoreBand == PGTURBOHYBRID_GRAPH_RESCORE_BAND_NONE)
		return 0;

	if (so->graphRescoreBand == PGTURBOHYBRID_GRAPH_RESCORE_BAND_EXACT ||
		pgturbohybrid_dense_rescore_band_policy == PGTURBOHYBRID_RESCORE_BAND_POLICY_EXACT)
		return count;

	if (pgturbohybrid_dense_rescore_band_policy == PGTURBOHYBRID_RESCORE_BAND_POLICY_OFF)
		return 0;

	if (pgturbohybrid_dense_rescore_band_policy == PGTURBOHYBRID_RESCORE_BAND_POLICY_AUTO &&
		(pgturbohybrid_dense_budget_policy == PGTURBOHYBRID_DENSE_BUDGET_QUALITY ||
		 (pgturbohybrid_dense_budget_policy == PGTURBOHYBRID_DENSE_BUDGET_AUTO &&
		  so->graphWideningReason != PGTURBOHYBRID_DENSE_WIDENING_DIMENSION)))
		return count;

	limitRows = so->hasTupleTargetRows ?
		Max((int64) 1, so->tupleTargetRows) : Max((int64) 1, so->graphDenseRequestedK);
	denseK = Max((int64) 1, so->graphDenseRequestedK);
	limitCap = limitRows * 4;
	denseCap = denseK * Max(pgturbohybrid_dense_max_rescore_multiplier, 1);
	cap = Max(limitCap, denseCap);

	if (cap <= 0)
		return count;
	return (int) Min((int64) count, cap);
}

void
PgturbohybridGraphExactRescore(Relation index, PgturbohybridGraphScanOpaque so, Datum query,
					PgturbohybridGraphMetaPageData *meta, PgturbohybridGraphScanNode *nodes,
					PgturbohybridGraphResult *results, int count)
{
	PgturbohybridGraphRescoreRef *refs;
	PgturbohybridGraphRescoreRef stackRefs[PGTURBOHYBRID_GRAPH_STACK_RESCORE_CAPACITY];
	bool		refsAllocated;
	int			refCount = 0;
	Size		vectorSize = PGTURBOHYBRID_VECTOR_SIZE(meta->dimensions);
	char	   *vectorScratch;

	if (DatumGetPointer(query) == NULL || count == 0 ||
		so->graphRescoreBand == PGTURBOHYBRID_GRAPH_RESCORE_BAND_NONE)
		return;

	vectorScratch = palloc(vectorSize);

	if (pgturbohybrid_dense_graph_stack_scratch && count <= PGTURBOHYBRID_GRAPH_STACK_RESCORE_CAPACITY)
	{
		refs = stackRefs;
		refsAllocated = false;
	}
	else
	{
		refs = palloc(sizeof(PgturbohybridGraphRescoreRef) * count);
		refsAllocated = true;
	}
	for (int i = 0; i < count; i++)
	{
		PgturbohybridGraphScanNode *node;

		if (results[i].nodeId >= meta->tqNodeCount)
			continue;
		if (results[i].exactScored)
			continue;

		node = &nodes[results[i].nodeId];
		if (node->exactVector != NULL)
		{
			results[i].distance =
				PgturbohybridGraphExactVectorDistance(so, query, node->exactVector);
			so->graphRescoreCount++;
			results[i].exactScored = true;
			continue;
		}

		if (!BlockNumberIsValid(node->exactBlkno) ||
			!PgturbohybridGraphExactByteOffsetIsValid(node->exactOffno))
			continue;

		refs[refCount].resultIndex = i;
		refs[refCount].nodeId = results[i].nodeId;
		refs[refCount].blkno = node->exactBlkno;
		refs[refCount].offno = node->exactOffno;
		refCount++;
	}

	qsort(refs, refCount, sizeof(PgturbohybridGraphRescoreRef), PgturbohybridGraphRescoreRefCompare);

	for (int i = 0; i < refCount; i++)
	{
		PgturbohybridGraphScanNode *node;

		CHECK_FOR_INTERRUPTS();

		if (refs[i].nodeId >= meta->tqNodeCount)
			continue;

		node = &nodes[refs[i].nodeId];
		if (PgturbohybridGraphReadExactVectorInto(index, node, meta->dimensions,
									   vectorScratch, so))
		{
			results[refs[i].resultIndex].distance =
				PgturbohybridGraphExactVectorDistance(so, query, vectorScratch);
			so->graphRescoreCount++;
			results[refs[i].resultIndex].exactScored = true;
		}
	}

	pfree(vectorScratch);
	if (refsAllocated)
		pfree(refs);
}



static void
PgturbohybridGraphCollectResults(IndexScanDesc scan, PgturbohybridGraphScanOpaque so,
					  int minResultTarget)
{
	PgturbohybridGraphMetaPageData meta;
	instr_time	totalStart;
	instr_time	phaseStart;
	Datum		query = PgturbohybridGraphGetScanValue(scan, so);
	int			resultTarget;
	int			searchEf;
	int			effectiveEf;
	int			count = 0;
	PgturbohybridGraphResult *results;
	PgturbohybridGraphScanStorage storage;
	int64		activeTarget;
	double		estimatedSelectivity;
	int			rescoreCount;
	int			finalCount;
	AttrNumber	payloadHeapAttno = InvalidAttrNumber;
	int32		payloadValue = 0;
	int			payloadSlot = -1;
	int			candidateOversampling;
	bool		hasPayloadFilter = false;
	bool		exactFree;
	bool		highDimL2Widened = false;
	bool		latencyBudgetActive = false;
	int64		requestedBaseTarget;

	INSTR_TIME_SET_CURRENT(totalStart);
	INSTR_TIME_SET_CURRENT(phaseStart);

	if (!PgturbohybridGraphReadMeta(scan->indexRelation, &meta) ||
		meta.tqNodeCount == 0 ||
		!BlockNumberIsValid(meta.tqCodeStartBlkno) ||
		!BlockNumberIsValid(meta.tqAdjStartBlkno))
	{
		so->tqGraphResults = NULL;
		so->tqGraphResultCount = 0;
		return;
	}

	exactFree = (meta.tqFlags & PGTURBOHYBRID_GRAPH_EXACT_FREE) != 0 ||
		!BlockNumberIsValid(meta.tqExactStartBlkno);
	PgturbohybridGraphPrepareTqQueryWithBits(scan->indexRelation, &so->support, query,
							   &so->tq,
							   meta.tqBits != 0 ? meta.tqBits : PGTURBOHYBRID_DEFAULT_BITS);
	activeTarget = PgturbohybridGraphGetActiveLimitTupleTarget();
	estimatedSelectivity = PgturbohybridGraphGetActiveEstimatedFilterSelectivity();
	if (PgturbohybridGraphGetActivePayloadInt4Filter(&payloadHeapAttno, &payloadValue))
	{
		payloadSlot = PgturbohybridGraphPayloadSlotForHeapAttr(scan->indexRelation,
												   payloadHeapAttno);
		hasPayloadFilter = payloadSlot >= 0 &&
			payloadSlot < meta.tqPayloadCount;
		if (!hasPayloadFilter)
			payloadSlot = -1;
	}
	if (!so->hasTupleTargetRows && activeTarget >= 0)
	{
		so->hasTupleTargetRows = true;
		so->tupleTargetRows = activeTarget;
	}
	PgturbohybridGraphSeedScanContext(so, activeTarget, estimatedSelectivity);
	effectiveEf = so->hasInitialEffectiveEfSearch ?
		so->initialEffectiveEfSearch : so->efSearch;
	so->graphDenseRequestedK = minResultTarget;
	so->graphDenseBudgetPolicy = pgturbohybrid_dense_budget_policy;
	so->graphRescoreBandPolicy = pgturbohybrid_dense_rescore_band_policy;

	candidateOversampling = Max(so->graphOversampling, 1);
	if (hasPayloadFilter && (TqScoreMode) so->tq.scoreMode != PGTURBOHYBRID_SCORE_L2)
		candidateOversampling = Min(candidateOversampling, 2);

	if (so->hasTupleTargetRows)
	{
			resultTarget = (int) Min((int64) INT_MAX,
									 Max(Max((int64) 1, so->tupleTargetRows) *
										 candidateOversampling,
										 (int64) effectiveEf));
		if (estimatedSelectivity > 0 && estimatedSelectivity < 1)
		{
			int64		filteredTarget =
				(int64) ceil((double) Max((int64) 1, so->tupleTargetRows) /
							 estimatedSelectivity) *
				candidateOversampling;

			resultTarget = (int) Min((int64) INT_MAX,
									 Max((int64) resultTarget, filteredTarget));
		}
	}
	else
		resultTarget = effectiveEf;

	if (minResultTarget > 0)
		resultTarget = (int) Min((int64) INT_MAX,
								 Max((int64) resultTarget,
									 (int64) minResultTarget));

	if (so->graphRescoreBand == PGTURBOHYBRID_GRAPH_RESCORE_BAND_AUTO &&
		(TqScoreMode) so->tq.scoreMode == PGTURBOHYBRID_SCORE_L2)
	{
			int64		l2Target = (int64) effectiveEf *
				Max(so->graphOversampling, 1);

		if (meta.dimensions >= 1024)
		{
				if (so->tq.bits < PGTURBOHYBRID_DEFAULT_BITS)
					l2Target *= PGTURBOHYBRID_GRAPH_LOWBIT_HIGHDIM_L2_TARGET_MULT;
				else
					l2Target = (int64) effectiveEf *
						PGTURBOHYBRID_GRAPH_HIGHDIM_L2_TARGET_EF_MULT;
				highDimL2Widened = true;
		}

		resultTarget = (int) Min((int64) INT_MAX,
								 Max((int64) resultTarget, l2Target));
	}

	if (!hasPayloadFilter && estimatedSelectivity > 0 && estimatedSelectivity < 1)
	{
		int64		conservativeTarget =
			Min((int64) meta.tqNodeCount, (int64) pgturbohybrid_max_scan_tuples);

		/*
		 * Native graph scans currently receive planner selectivity but not
		 * the actual heap predicate. When clustered data makes the nearest
		 * global neighborhood mostly miss the filter, a k/selectivity band
		 * can return too few post-filter rows. Widen to the configured scan
		 * budget only when the predicate cannot be mapped to graph-owned
		 * payload columns.
		 */
		resultTarget = (int) Min((int64) INT_MAX,
								 Max((int64) resultTarget, conservativeTarget));
		so->graphWideningReason = PGTURBOHYBRID_DENSE_WIDENING_FILTER;
	}

	requestedBaseTarget = Max((int64) 1, (int64) minResultTarget);
	if (so->hasTupleTargetRows)
		requestedBaseTarget = Max(requestedBaseTarget,
								  Max((int64) 1, so->tupleTargetRows));
	latencyBudgetActive = PgturbohybridGraphUseLatencyDenseBudget(&meta, hasPayloadFilter,
													   estimatedSelectivity);
	if (latencyBudgetActive)
	{
		double		multiplier = PgturbohybridGraphDenseBudgetMultiplier();
		int64		adaptiveCap =
			(int64) ceil((double) requestedBaseTarget * multiplier);
		int64		maxCap = requestedBaseTarget *
			Max(pgturbohybrid_dense_max_candidate_multiplier, 1);
		int64		budgetCap = Max((int64) effectiveEf,
								   Min(adaptiveCap, maxCap));

		if (budgetCap > 0 && resultTarget > budgetCap)
			resultTarget = (int) Min((int64) INT_MAX, budgetCap);
	}
	if (so->graphWideningReason == PGTURBOHYBRID_DENSE_WIDENING_NONE && highDimL2Widened)
		so->graphWideningReason = PGTURBOHYBRID_DENSE_WIDENING_DIMENSION;
	so->graphHighdimWideningMultiplier =
		requestedBaseTarget > 0 ?
		((double) resultTarget / (double) requestedBaseTarget) : 1.0;
	resultTarget = Max(resultTarget, 1);
	resultTarget = Min(resultTarget, (int) meta.tqNodeCount);
	searchEf = Min(Max(effectiveEf, resultTarget), (int) meta.tqNodeCount);
	so->graphEffectiveResultTarget = resultTarget;
	so->graphEffectiveSearchEf = searchEf;
	results = palloc(sizeof(PgturbohybridGraphResult) * resultTarget);
	PgturbohybridGraphInitScanStorage(scan->indexRelation, &meta, &storage);
	PgturbohybridGraphAddElapsedUs(&so->graphPrepareUs, phaseStart);

	if (hasPayloadFilter &&
		PgturbohybridGraphCollectPayloadExactBand(so, &meta, &storage, query,
									   payloadSlot, payloadValue, results,
									   resultTarget, &count))
	{
		INSTR_TIME_SET_CURRENT(phaseStart);
		qsort(results, count, sizeof(PgturbohybridGraphResult), PgturbohybridGraphResultCompare);
		PgturbohybridGraphAddElapsedUs(&so->graphSortUs, phaseStart);
		so->graphCandidateCount = count;
		so->graphEffectiveRescoreBand = so->graphRescoreCount;
		so->tqGraphResults = results;
		so->tqGraphResultCount = count;
		so->tqGraphResultIndex = 0;
		PgturbohybridGraphAddElapsedUs(&so->graphTotalUs, totalStart);
		PgturbohybridGraphRecordGraphScanStats(so);
		return;
	}

	INSTR_TIME_SET_CURRENT(phaseStart);
	count = PgturbohybridGraphTraverse(scan->indexRelation, so, &meta, &storage, results,
							resultTarget, searchEf, query, payloadSlot,
							payloadValue);
	PgturbohybridGraphAddElapsedUs(&so->graphTraverseUs, phaseStart);
	if (estimatedSelectivity > 0 && estimatedSelectivity < 1 && count < resultTarget)
	{
		INSTR_TIME_SET_CURRENT(phaseStart);
		count = PgturbohybridGraphFillCandidateBand(scan->indexRelation, so, &meta,
										 &storage, results, resultTarget, count,
										 payloadSlot, payloadValue, query);
		PgturbohybridGraphAddElapsedUs(&so->graphFillUs, phaseStart);
	}
	INSTR_TIME_SET_CURRENT(phaseStart);
	qsort(results, count, sizeof(PgturbohybridGraphResult), PgturbohybridGraphResultCompare);
	PgturbohybridGraphAddElapsedUs(&so->graphSortUs, phaseStart);

	if (!latencyBudgetActive &&
		so->graphRescoreBand == PGTURBOHYBRID_GRAPH_RESCORE_BAND_AUTO &&
		(TqScoreMode) so->tq.scoreMode == PGTURBOHYBRID_SCORE_L2 &&
		count > 0 && count < resultTarget && results[0].distance < 1.0)
	{
		int			wideTarget = (int) Min((int64) meta.tqNodeCount,
											   Max((int64) resultTarget,
												   (int64) effectiveEf *
												   Max(so->graphOversampling, 1) *
												   PGTURBOHYBRID_GRAPH_TIGHT_L2_FILL_MULT));

		if (wideTarget > resultTarget)
		{
			pfree(results);
			resultTarget = wideTarget;
				searchEf = Min(Max(effectiveEf, resultTarget),
							   (int) meta.tqNodeCount);
			so->graphEffectiveResultTarget = resultTarget;
			so->graphEffectiveSearchEf = searchEf;
			so->graphHighdimWideningMultiplier =
				requestedBaseTarget > 0 ?
				((double) resultTarget / (double) requestedBaseTarget) : 1.0;
			so->graphWideningReason = PGTURBOHYBRID_DENSE_WIDENING_EXACT_POLICY;
			results = palloc(sizeof(PgturbohybridGraphResult) * resultTarget);
			INSTR_TIME_SET_CURRENT(phaseStart);
			count = PgturbohybridGraphTraverse(scan->indexRelation, so, &meta, &storage,
									results, resultTarget, searchEf, query,
									payloadSlot, payloadValue);
			PgturbohybridGraphAddElapsedUs(&so->graphTraverseUs, phaseStart);
			if (count < resultTarget)
			{
				INSTR_TIME_SET_CURRENT(phaseStart);
				count = PgturbohybridGraphFillCandidateBand(scan->indexRelation, so, &meta,
												 &storage, results,
												 resultTarget, count,
												 payloadSlot, payloadValue,
												 query);
				PgturbohybridGraphAddElapsedUs(&so->graphFillUs, phaseStart);
			}
			INSTR_TIME_SET_CURRENT(phaseStart);
			qsort(results, count, sizeof(PgturbohybridGraphResult), PgturbohybridGraphResultCompare);
			PgturbohybridGraphAddElapsedUs(&so->graphSortUs, phaseStart);
		}
	}

	so->graphCandidateCount = count;
	if (exactFree)
	{
		so->graphEffectiveRescoreBand = 0;
		so->tqGraphResults = results;
		so->tqGraphResultCount = count;
		so->tqGraphResultIndex = 0;
		PgturbohybridGraphAddElapsedUs(&so->graphTotalUs, totalStart);
		PgturbohybridGraphRecordGraphScanStats(so);
		return;
	}
	rescoreCount = PgturbohybridGraphFinalRescoreCount(so, results, count, effectiveEf);
	so->graphEffectiveRescoreBand = rescoreCount;
	finalCount = so->graphRescoreBand == PGTURBOHYBRID_GRAPH_RESCORE_BAND_AUTO &&
		rescoreCount > 0 ? rescoreCount : count;
	INSTR_TIME_SET_CURRENT(phaseStart);
	PgturbohybridGraphExactRescore(scan->indexRelation, so, query, &meta, storage.nodes,
						results, rescoreCount);
	PgturbohybridGraphAddElapsedUs(&so->graphRescoreUs, phaseStart);
	INSTR_TIME_SET_CURRENT(phaseStart);
	qsort(results, finalCount, sizeof(PgturbohybridGraphResult), PgturbohybridGraphResultCompare);
	PgturbohybridGraphAddElapsedUs(&so->graphSortUs, phaseStart);
	so->tqGraphResults = results;
	so->tqGraphResultCount = finalCount;
	so->tqGraphResultIndex = 0;
	PgturbohybridGraphAddElapsedUs(&so->graphTotalUs, totalStart);
	PgturbohybridGraphRecordGraphScanStats(so);
}

int
PgturbohybridGraphCollectDenseCandidates(IndexScanDesc scan, int targetK,
							  TqDenseCandidate **outCandidates,
							  MemoryContext resultCtx,
							  TqDenseCandidateStats *stats)
{
	PgturbohybridGraphScanOpaque so = (PgturbohybridGraphScanOpaque) scan->opaque;
	PgturbohybridGraphResult *results;
	TqDenseCandidate *candidates;
	int			count;
	int			limit;
	MemoryContext oldCtx;

	if (so == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("pgturbohybrid dense candidate collection requires an active scan")));

	if (so->tqGraphResults == NULL)
	{
		LockPage(scan->indexRelation, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ShareLock);
		PG_TRY();
		{
			PgturbohybridGraphCollectResults(scan, so, targetK);
		}
		PG_CATCH();
		{
			UnlockPage(scan->indexRelation, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ShareLock);
			PG_RE_THROW();
		}
		PG_END_TRY();
		UnlockPage(scan->indexRelation, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ShareLock);
	}

	results = (PgturbohybridGraphResult *) so->tqGraphResults;
	count = so->tqGraphResultCount;
	limit = targetK > 0 ? Min(targetK, count) : count;

	oldCtx = MemoryContextSwitchTo(resultCtx);
	candidates = palloc0(sizeof(TqDenseCandidate) * Max(limit, 1));
	for (int i = 0; i < limit; i++)
	{
		candidates[i].nodeId = results[i].nodeId;
		candidates[i].heaptid = results[i].heaptid;
		candidates[i].distance = results[i].distance;
		candidates[i].similarity = -results[i].distance;
		candidates[i].rank = i + 1;
		candidates[i].exactScored = results[i].exactScored;
	}
	MemoryContextSwitchTo(oldCtx);

	if (stats != NULL)
	{
		memset(stats, 0, sizeof(*stats));
		stats->visitedGraphNodes = so->graphVisitedNodes;
		stats->scoredCodes = so->graphScoredCodes;
		stats->denseCandidatesRequested = targetK > 0 ? targetK : limit;
		stats->effectiveResultTarget = (uint32) Max(so->graphEffectiveResultTarget, 0);
		stats->effectiveSearchEf = (uint32) Max(so->graphEffectiveSearchEf, 0);
		stats->effectiveRescoreBand = (uint32) Max(so->graphEffectiveRescoreBand, 0);
		stats->highdimWideningMultiplier = so->graphHighdimWideningMultiplier;
		stats->wideningReason = so->graphWideningReason;
		stats->denseBudgetPolicy = so->graphDenseBudgetPolicy;
		stats->rescoreBandPolicy = so->graphRescoreBandPolicy;
		stats->denseCandidatesReturned = limit;
		stats->exactRescoreCount = so->graphRescoreCount;
		stats->codePagesRead = so->graphCodePagesRead;
		stats->adjPagesRead = so->graphAdjPagesRead;
		stats->prepareUs = so->graphPrepareUs;
		stats->traverseUs = so->graphTraverseUs;
		stats->entryUs = so->graphEntryUs;
		stats->baseUs = so->graphBaseUs;
		stats->batchUs = so->graphBatchUs;
		stats->heapUs = so->graphHeapUs;
		stats->fillUs = so->graphFillUs;
		stats->rescoreUs = so->graphRescoreUs;
		stats->sortUs = so->graphSortUs;
	}

	*outCandidates = candidates;
	return limit;
}

bool
tqgraphgettuple(IndexScanDesc scan, ScanDirection dir)
{
	PgturbohybridGraphScanOpaque so = (PgturbohybridGraphScanOpaque) scan->opaque;
	MemoryContext oldCtx = MemoryContextSwitchTo(so->tmpCtx);
	PgturbohybridGraphResult *results;

	Assert(ScanDirectionIsForward(dir));

	if (so->first)
	{
		pgstat_count_index_scan(scan->indexRelation);
#if PG_VERSION_NUM >= 180000
		if (scan->instrument)
			scan->instrument->nsearches++;
#endif
		if (!IsMVCCSnapshot(scan->xs_snapshot))
			elog(ERROR, "non-MVCC snapshots are not supported with pgturbohybrid graph");

		LockPage(scan->indexRelation, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ShareLock);
		PgturbohybridGraphCollectResults(scan, so, 0);
		UnlockPage(scan->indexRelation, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ShareLock);
		so->first = false;
	}

	if (so->tqGraphResultIndex >= so->tqGraphResultCount)
	{
		MemoryContextSwitchTo(oldCtx);
		return false;
	}

	results = (PgturbohybridGraphResult *) so->tqGraphResults;
	scan->xs_heaptid = results[so->tqGraphResultIndex++].heaptid;
	scan->xs_recheck = false;
	scan->xs_recheckorderby = false;
	so->returnedRows++;

	MemoryContextSwitchTo(oldCtx);
	return true;
}

void
tqgraphendscan(IndexScanDesc scan)
{
	PgturbohybridGraphScanOpaque so = (PgturbohybridGraphScanOpaque) scan->opaque;

	MemoryContextDelete(so->tmpCtx);
	pfree(so);
	scan->opaque = NULL;
}

bool
tqgraphinsert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid,
			  Relation heap, IndexUniqueCheck checkUnique
#if PG_VERSION_NUM >= 140000
			  ,bool indexUnchanged
#endif
			  ,IndexInfo *indexInfo
)
{
	Datum		value;
	const PgturbohybridGraphTypeInfo *typeInfo = PgturbohybridGraphGetTypeInfo(index);
	PgturbohybridGraphSupport support;

	(void) heap;
	(void) checkUnique;
#if PG_VERSION_NUM >= 140000
	(void) indexUnchanged;
#endif
	if (isnull[0])
		return false;

	PgturbohybridGraphInitSupport(&support, index);
	if (!PgturbohybridGraphFormIndexValue(&value, values, isnull, typeInfo, &support))
		return false;

	LockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, ExclusiveLock);
	PG_TRY();
	{
		PgturbohybridGraphInsertValueInPlace(index, indexInfo, heap_tid, value,
								  values, isnull);
	}
	PG_CATCH();
	{
		UnlockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, ExclusiveLock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	UnlockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, ExclusiveLock);

	return true;
}

static bool
PgturbohybridGraphRepairAdjacencyForDeadNodes(Relation index, PgturbohybridGraphMetaPageData *meta,
								   bool *deadNodes)
{
	int			levelCapacity;
	bool		changedAny = false;
	BlockNumber blkno;
	BlockNumber nblocks;

	if (!BlockNumberIsValid(meta->tqAdjStartBlkno) || deadNodes == NULL)
		return false;

	levelCapacity = PgturbohybridGraphLevelCapacity(meta->m);
	blkno = meta->tqAdjStartBlkno;
	nblocks = RelationGetNumberOfBlocks(index);

	while (BlockNumberIsValid(blkno) && blkno < nblocks)
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		BlockNumber nextblkno;
		OffsetNumber maxoff;
		bool		changed = false;
		GenericXLogState *xlogState = NULL;

		CHECK_FOR_INTERRUPTS();

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

		if (RelationNeedsWAL(index))
		{
			xlogState = GenericXLogStart(index);
			page = GenericXLogRegisterBuffer(xlogState, buf, 0);
		}
		else
			page = BufferGetPage(buf);

		opaque = PgturbohybridGraphPageGetOpaque(page);
		if ((opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) != PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_ADJ)
		{
			if (xlogState != NULL)
				GenericXLogAbort(xlogState);
			UnlockReleaseBuffer(buf);
			break;
		}
		nextblkno = opaque->nextblkno;

		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
		{
			ItemId		iid = PageGetItemId(page, offno);
			PgturbohybridGraphAdjTuple tuple;
			uint16		maxNeighbors;
			uint16		oldCount;
			uint16		scanCount;
			uint16		newCount = 0;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (PgturbohybridGraphAdjTuple) PageGetItem(page, iid);
			if (tuple->type != PGTURBOHYBRID_GRAPH_ADJ_TUPLE_TYPE ||
				tuple->nodeId >= meta->tqNodeCount ||
				tuple->level >= levelCapacity)
				continue;

			maxNeighbors = PgturbohybridGraphLevelM(meta->m, tuple->level);
			oldCount = tuple->count;
			scanCount = Min(oldCount, maxNeighbors);

			if (!deadNodes[tuple->nodeId])
			{
				for (int i = 0; i < scanCount; i++)
				{
					uint32		neighbor = tuple->neighbors[i];

					if (neighbor < meta->tqNodeCount && !deadNodes[neighbor])
						tuple->neighbors[newCount++] = neighbor;
				}
			}

			if (newCount != oldCount)
			{
				if (newCount < scanCount)
					memset(&tuple->neighbors[newCount], 0,
						   sizeof(uint32) * (scanCount - newCount));
				tuple->count = newCount;
				changed = true;
			}
		}

		if (changed)
			PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_VACUUM_REPAIR);

		if (xlogState != NULL)
		{
			if (changed)
				GenericXLogFinish(xlogState);
			else
				GenericXLogAbort(xlogState);
		}
		else if (changed)
			MarkBufferDirty(buf);

		UnlockReleaseBuffer(buf);

		if (changed)
		{
			changedAny = true;
			PgturbohybridGraphLogGraphWalRecord(index, MAIN_FORKNUM, blkno, PGTURBOHYBRID_GRAPH_GRAPH_OP_VACUUM_REPAIR);
		}

		if (nextblkno == blkno)
			break;
		blkno = nextblkno;
	}

	return changedAny;
}

void
PgturbohybridGraphCollectVacuumStats(Relation index, PgturbohybridGraphMetaPageData *meta,
						  int64 *liveNodes, int64 *deadNodes,
						  int64 *adjacencyRefs, int64 *deadNeighborRefs)
{
	int			codeTuplesPerPage;
	int			codePageCount;
	int			tqBits = meta->tqBits != 0 ? meta->tqBits : PGTURBOHYBRID_DEFAULT_BITS;
	int			levelCapacity;
	bool	   *deadBitmap;
	BlockNumber nblocks;
	BlockNumber blkno;

	*liveNodes = 0;
	*deadNodes = 0;
	*adjacencyRefs = 0;
	*deadNeighborRefs = 0;

	if (meta->tqNodeCount == 0 ||
		!BlockNumberIsValid(meta->tqCodeStartBlkno))
		return;

	deadBitmap = palloc0(sizeof(bool) * meta->tqNodeCount);
	nblocks = RelationGetNumberOfBlocks(index);
	codeTuplesPerPage =
		PgturbohybridGraphTuplesPerPage(PgturbohybridGraphCodeTupleSize(meta->dimensions,
												  meta->tqPayloadCount,
												  tqBits,
												  (meta->tqFlags & PGTURBOHYBRID_GRAPH_TQ_WEIGHTED) != 0));
	codePageCount = PgturbohybridGraphPageCount(meta->tqNodeCount, codeTuplesPerPage);
	blkno = meta->tqCodeStartBlkno;

	for (int pageNo = 0;
		 pageNo < codePageCount && BlockNumberIsValid(blkno) && blkno < nblocks;
		 pageNo++)
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		OffsetNumber maxoff;

		CHECK_FOR_INTERRUPTS();

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = PgturbohybridGraphPageGetOpaque(page);
		if ((opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) != PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_CODE)
		{
			UnlockReleaseBuffer(buf);
			break;
		}

		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
		{
			ItemId		iid = PageGetItemId(page, offno);
			PgturbohybridGraphCodeTuple tuple;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (PgturbohybridGraphCodeTuple) PageGetItem(page, iid);
			if (tuple->type != PGTURBOHYBRID_GRAPH_CODE_TUPLE_TYPE ||
				tuple->nodeId >= meta->tqNodeCount)
				continue;

			if (tuple->flags & PGTURBOHYBRID_GRAPH_NODE_DEAD)
			{
				deadBitmap[tuple->nodeId] = true;
				(*deadNodes)++;
			}
			else
				(*liveNodes)++;
		}

		blkno = opaque->nextblkno;
		UnlockReleaseBuffer(buf);
	}

	if (!BlockNumberIsValid(meta->tqAdjStartBlkno))
	{
		pfree(deadBitmap);
		return;
	}

	levelCapacity = PgturbohybridGraphLevelCapacity(meta->m);
	blkno = meta->tqAdjStartBlkno;

	while (BlockNumberIsValid(blkno) && blkno < nblocks)
	{
		Buffer		buf;
		Page		page;
		PgturbohybridGraphPageOpaque opaque;
		OffsetNumber maxoff;
		BlockNumber nextblkno;

		CHECK_FOR_INTERRUPTS();

		buf = ReadBuffer(index, blkno);
		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = PgturbohybridGraphPageGetOpaque(page);
		if ((opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) != PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_ADJ)
		{
			UnlockReleaseBuffer(buf);
			break;
		}
		nextblkno = opaque->nextblkno;

		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
		{
			ItemId		iid = PageGetItemId(page, offno);
			PgturbohybridGraphAdjTuple tuple;
			uint16		maxNeighbors;
			bool		deadSource;

			if (!ItemIdIsUsed(iid))
				continue;

			tuple = (PgturbohybridGraphAdjTuple) PageGetItem(page, iid);
			if (tuple->type != PGTURBOHYBRID_GRAPH_ADJ_TUPLE_TYPE ||
				tuple->nodeId >= meta->tqNodeCount ||
				tuple->level >= levelCapacity)
				continue;

			maxNeighbors = PgturbohybridGraphLevelM(meta->m, tuple->level);
			deadSource = deadBitmap[tuple->nodeId];

			for (int i = 0; i < Min(tuple->count, maxNeighbors); i++)
			{
				uint32		neighbor = tuple->neighbors[i];

				(*adjacencyRefs)++;
				if (deadSource || neighbor >= meta->tqNodeCount ||
					deadBitmap[neighbor])
					(*deadNeighborRefs)++;
			}
		}

		UnlockReleaseBuffer(buf);
		if (nextblkno == blkno)
			break;
		blkno = nextblkno;
	}

	pfree(deadBitmap);
}

typedef struct PgturbohybridGraphBulkDeleteState
{
	double		liveTuples;
	bool		changedAny;
	bool		repairAny;
	bool		hasDeadNodes;
} PgturbohybridGraphBulkDeleteState;

IndexBulkDeleteResult *
tqgraphbulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats,
				  IndexBulkDeleteCallback callback, void *callback_state)
{
	Relation	index = info->index;
	IndexBulkDeleteResult *volatile result = stats;
	PgturbohybridGraphMetaPageData meta;
	PgturbohybridGraphBulkDeleteState *deleteState;
	int			codeTuplesPerPage;
	int			codePageCount;
	int			tqBits;
	bool	   *deadNodes = NULL;

	if (result == NULL)
		result = palloc0(sizeof(IndexBulkDeleteResult));

	if (callback == NULL || !PgturbohybridGraphReadMeta(index, &meta) ||
		meta.tqNodeCount == 0 ||
		!BlockNumberIsValid(meta.tqCodeStartBlkno))
		return (IndexBulkDeleteResult *) result;

	tqBits = meta.tqBits != 0 ? meta.tqBits : PGTURBOHYBRID_DEFAULT_BITS;
	codeTuplesPerPage =
		PgturbohybridGraphTuplesPerPage(PgturbohybridGraphCodeTupleSize(meta.dimensions,
													  meta.tqPayloadCount,
													  tqBits,
													  (meta.tqFlags & PGTURBOHYBRID_GRAPH_TQ_WEIGHTED) != 0));
	codePageCount = PgturbohybridGraphPageCount(meta.tqNodeCount, codeTuplesPerPage);
	deleteState = palloc0(sizeof(PgturbohybridGraphBulkDeleteState));
	deadNodes = palloc0(sizeof(bool) * meta.tqNodeCount);

	LockPage(index, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ExclusiveLock);
	PG_TRY();
	{
		BlockNumber blkno = meta.tqCodeStartBlkno;
		BlockNumber nblocks = RelationGetNumberOfBlocks(index);

		for (int pageNo = 0;
			 pageNo < codePageCount && BlockNumberIsValid(blkno) && blkno < nblocks;
			 pageNo++)
		{
			Buffer		buf;
			Page		page;
			PgturbohybridGraphPageOpaque opaque;
			BlockNumber nextblkno;
			OffsetNumber maxoff;
			bool		changed = false;
			GenericXLogState *xlogState = NULL;

			CHECK_FOR_INTERRUPTS();

			buf = ReadBuffer(index, blkno);
			LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

			if (RelationNeedsWAL(index))
			{
				xlogState = GenericXLogStart(index);
				page = GenericXLogRegisterBuffer(xlogState, buf, 0);
			}
			else
				page = BufferGetPage(buf);

			opaque = PgturbohybridGraphPageGetOpaque(page);
			if ((opaque->pageKind & PGTURBOHYBRID_GRAPH_PAGE_KIND_MASK) != PGTURBOHYBRID_GRAPH_PAGE_KIND_QUANT_CODE)
			{
				if (xlogState != NULL)
					GenericXLogAbort(xlogState);
				UnlockReleaseBuffer(buf);
				break;
			}
			nextblkno = opaque->nextblkno;

			maxoff = PageGetMaxOffsetNumber(page);
			for (OffsetNumber offno = FirstOffsetNumber; offno <= maxoff; offno = OffsetNumberNext(offno))
			{
				ItemId		iid = PageGetItemId(page, offno);
				PgturbohybridGraphCodeTuple tuple;

				if (!ItemIdIsUsed(iid))
					continue;

				tuple = (PgturbohybridGraphCodeTuple) PageGetItem(page, iid);
				if (tuple->type != PGTURBOHYBRID_GRAPH_CODE_TUPLE_TYPE ||
					tuple->nodeId >= meta.tqNodeCount)
					continue;

				if (tuple->flags & PGTURBOHYBRID_GRAPH_NODE_DEAD)
				{
					deadNodes[tuple->nodeId] = true;
					deleteState->hasDeadNodes = true;
					continue;
				}

				if (callback(&tuple->heaptid, callback_state))
				{
					tuple->flags |= PGTURBOHYBRID_GRAPH_NODE_DEAD;
					deadNodes[tuple->nodeId] = true;
					deleteState->hasDeadNodes = true;
					result->tuples_removed += 1;
					changed = true;
				}
				else
					deleteState->liveTuples += 1;
			}

			if (changed)
				PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_VACUUM_DELETE);

			if (xlogState != NULL)
				GenericXLogFinish(xlogState);
			else if (changed)
				MarkBufferDirty(buf);

			UnlockReleaseBuffer(buf);

			if (changed)
			{
				deleteState->changedAny = true;
				PgturbohybridGraphLogGraphWalRecord(index, MAIN_FORKNUM, blkno, PGTURBOHYBRID_GRAPH_GRAPH_OP_VACUUM_DELETE);
			}

			blkno = nextblkno;
		}

		if (deleteState->hasDeadNodes)
			deleteState->repairAny = PgturbohybridGraphRepairAdjacencyForDeadNodes(index, &meta, deadNodes);

		if (deleteState->changedAny || deleteState->repairAny)
			PgturbohybridGraphBumpMetaGeneration(index);
	}
	PG_CATCH();
	{
		UnlockPage(index, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ExclusiveLock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	UnlockPage(index, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ExclusiveLock);
	pfree(deadNodes);
	result->num_index_tuples = deleteState->liveTuples;
	pfree(deleteState);

	result->estimated_count = false;

	return (IndexBulkDeleteResult *) result;
}

IndexBulkDeleteResult *
tqgraphvacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats)
{
	(void) info;

	if (stats == NULL)
		stats = palloc0(sizeof(IndexBulkDeleteResult));

	return stats;
}
