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

typedef struct PgturbohybridNativeParallelShared PgturbohybridNativeParallelShared;



static int	PgturbohybridGraphScanAdjSlot(PgturbohybridGraphMetaPageData *meta,
										  uint32 nodeId, int level);
static bool PgturbohybridGraphEntryAlreadySelected(PgturbohybridGraphFrontierItem *entries, int entryCount,
										uint32 nodeId);
static void PgturbohybridGraphEncodeBuildNode(PgturbohybridQuantBuildState *state,
								 PgturbohybridGraphBuildNode *node,
								 Vector *vector);
static bool PgturbohybridGraphIndexIsMultiVector(Relation index);
static void PgturbohybridGraphStreamFitVector(PgturbohybridQuantBuildState *state,
								Vector *vector);
static void PgturbohybridGraphFinishStreamingFit(PgturbohybridQuantBuildState *state);
static bool PgturbohybridGraphUseExactBuildDistances(PgturbohybridQuantBuildState *state);
static bool PgturbohybridGraphUseFastEdgesForMultiVectorDocumentProxy(PgturbohybridQuantBuildState *state);
static bool PgturbohybridGraphUseFastBuildEdges(PgturbohybridQuantBuildState *state);
static void PgturbohybridGraphSetParallelEdgeBuildDisabledReason(PgturbohybridQuantBuildState *state,
																 int reason);
static bool PgturbohybridGraphCanUseParallelBuildWorkers(PgturbohybridQuantBuildState *state);
static void PgturbohybridGraphCheckExactSymmetricBuildAllowed(PgturbohybridQuantBuildState *state);
static void PgturbohybridNativeParallelEdgeWorker(Relation indexRel,
												  PgturbohybridNativeParallelShared *shared,
												  bool leader);
static int64 PgturbohybridGraphElapsedUs(instr_time start);
static bool PgturbohybridGraphMultiVectorDocMapHasContexts(PgturbohybridQuantBuildState *state);
static uint16 PgturbohybridGraphMultiVectorDocMapVectorTupleMaxCount(void);
static uint16 PgturbohybridGraphMultiVectorDocMapVectorF16TupleMaxCount(void);
static uint16 PgturbohybridGraphMultiVectorDocMapVectorSq8TupleMaxCount(void);
static uint16 PgturbohybridGraphMultiVectorDocMapCentroidTupleMaxCount(void);
static uint16 PgturbohybridGraphMultiVectorDocMapCentroidF16TupleMaxCount(void);
static uint16 PgturbohybridGraphMultiVectorDocMapCentroidDocCodeTupleMaxCount(void);
static bool PgturbohybridGraphMultiVectorDocMapStoresDocVectors(PgturbohybridQuantBuildState *state);
static bool PgturbohybridGraphMultiVectorDocMapStoresCentroids(PgturbohybridQuantBuildState *state);
static bool PgturbohybridGraphMultiVectorDocMapStoresQuantizedInvertedPostings(PgturbohybridQuantBuildState *state);
static void PgturbohybridGraphAddMultiVectorDocMapBytesEstimate(PgturbohybridQuantBuildState *state,
															   uint64 bytes,
															   const char *component);
static uint64 PgturbohybridGraphBuildMultiplyEstimate(uint64 left, uint64 right);
static uint64 PgturbohybridGraphBuildAddEstimate(uint64 left, uint64 right);
static void PgturbohybridGraphRefreshBuildMemoryEstimates(PgturbohybridQuantBuildState *state);
static void PgturbohybridGraphRecordBuildMemorySnapshot(PgturbohybridQuantBuildState *state,
													   const char *phase);
static uint64 PgturbohybridGraphMultiVectorDocMapChunkedFloatTupleBytes(Size floatCount,
																		 uint16 maxCount,
																		 Size (*tupleSizer) (uint16));
static uint64 PgturbohybridGraphMultiVectorDocMapChunkedVectorTupleBytes(Size floatCount,
																		 int storageKind);
static void PgturbohybridGraphAppendMultiVectorQuantizedInvertedBuildPostings(PgturbohybridQuantBuildState *state,
																			 const PgturbohybridMultiVector *mv,
																			 uint32 docId);

#define PARALLEL_KEY_PGTURBOHYBRID_NATIVE_SHARED	UINT64CONST(0xA000000000000011)
#define PARALLEL_KEY_PGTURBOHYBRID_NATIVE_QUERY		UINT64CONST(0xA000000000000012)
#define PGTURBOHYBRID_GRAPH_AUTO_HEURISTIC_MAX_DIMENSIONS 256
#define PGTURBOHYBRID_GRAPH_AUTO_EXACT_BUILD_MAX_DIMENSIONS 256
#define PGTURBOHYBRID_GRAPH_AUTO_HEAP_RESCORE_MAX_DIMENSIONS 256
/*
 * Aggressive low-bit codes (1- and 2-bit) are too lossy to rank on directly, so
 * the auto profiles default to a heap rescore for them at any dimension, the way
 * Qdrant's tq_bits_default_rescoring() rescores 1/1.5/2-bit but leaves 4-bit
 * off.  The default 4-bit index is therefore unaffected.
 */
#define PGTURBOHYBRID_GRAPH_LOWBIT_HEAP_RESCORE_MAX_BITS 2
#define PGTURBOHYBRID_GRAPH_PARALLEL_EDGE_WARMUP 256
#define PGTURBOHYBRID_GRAPH_PARALLEL_EDGE_BATCH_PER_PARTICIPANT 32
#define PGTURBOHYBRID_GRAPH_PARALLEL_EDGE_BATCH_MAX 256

typedef enum PgturbohybridNativeParallelPhase
{
	PGTURBOHYBRID_NATIVE_PARALLEL_FIT = 1,
	PGTURBOHYBRID_NATIVE_PARALLEL_ENCODE = 2,
	PGTURBOHYBRID_NATIVE_PARALLEL_EDGES = 3
} PgturbohybridNativeParallelPhase;

typedef struct PgturbohybridNativeParallelRecord
{
	ItemPointerData heaptid;
	uint64		vectorHash;
	float		norm;
	float		scale;
	float		correction;
	float		ecCorrection;
	uint16		payloadMask;
	uint16		reserved;
	int32		payloads[PGTURBOHYBRID_GRAPH_MAX_PAYLOADS];
} PgturbohybridNativeParallelRecord;

typedef struct PgturbohybridNativeParallelEdgeNode
{
	ItemPointerData heaptid;
	uint64		vectorHash;
	float		norm;
	float		scale;
	float		correction;
	float		ecCorrection;
	uint16		payloadMask;
	uint16		flags;
	int			level;
	int32		payloads[PGTURBOHYBRID_GRAPH_MAX_PAYLOADS];
} PgturbohybridNativeParallelEdgeNode;



typedef struct PgturbohybridNativeParallelShared
{
	Oid			heaprelid;
	Oid			indexrelid;
	bool		isconcurrent;
	int			phase;
	ConditionVariable workersdonecv;
	slock_t		mutex;
	int			nparticipantsdone;
	int			nparticipants;
	int			participantCapacity;
	int			nextParticipant;
	double		reltuples;
	uint32		dimensions;
	int			m;
	int			efConstruction;
	int			tqBits;
	int			scoreMode;
	bool		tqWeighted;
	bool		tqRenorm;
	bool		buildCodeOnly;
	bool		buildFastEdges;
	int			payloadCount;
	Size		payloadBytes;
	int			residualRerankBytes;
	uint32		recordCount;
	uint32		recordCapacity;
	Size		recordBytes;
	Size		sharedBytes;
	Size		recordsBytes;
	Size		fitCountOffset;
	Size		fitMeanOffset;
	Size		fitM2Offset;
	Size		encodeUsOffset;
	Size		scanUsOffset;
	Size		ecShiftOffset;
	Size		ecScaleOffset;
	Size		recordsOffset;
	uint32		edgeNodeCount;
	uint16		edgeSegmentCount;
	uint16		edgeFinalSegmentCount;
	uint32		edgeNextSegment;
	uint32		edgeLevelCapacity;
	uint32		edgeMaxNeighbors;
	Size		edgeCodeBytes;
	Size		edgeNodeOffset;
	Size		edgeCodeOffset;
	Size		edgeResidualOffset;
	Size		edgeNeighborCountOffset;
	Size		edgeNeighborOffset;
	Size		edgeNeighborDistanceOffset;
	Size		edgeSegmentOffset;
	Size		edgeWorkerUsOffset;
	Size		edgeOrderOffset;
	Size		edgeInsertedOffset;
	Barrier		edgeBarrier;
	bool		edgeBarrierReady;
	bool		edgeDone;
	uint32		edgeNextOrder;
	uint32		edgeBatchStartOrder;
	uint32		edgeBatchEndOrder;
	uint32		edgeWarmupCount;
	uint32		edgeBatchSize;
	uint32		edgeInsertedCount;
	uint32		edgeEntryNodeId;
	int			edgeEntryLevel;
	uint64		edgeDistanceCalls;
	uint64		edgeEntrySearchUs;
	uint64		edgeNeighborSearchUs;
	uint64		edgeSearchLayerUs;
	uint64		edgeSelectNeighborUs;
	uint64		edgeAddNeighborUs;
	uint64		edgePruneNeighborUs;
	uint64		edgeEntryUpdateUs;
	uint64		edgeNearestTotal;
	uint64		edgeNearestSamples;
	uint32		edgeMaxFrontierSize;
	bool		overflowed;
} PgturbohybridNativeParallelShared;

#define ParallelTableScanFromNativeShared(shared) \
	(ParallelTableScanDesc) ((char *) (shared) + BUFFERALIGN(sizeof(PgturbohybridNativeParallelShared)))


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
	if (leftVector == rightVector && leftVector != NULL)
		return true;
	if (leftVector == NULL || rightVector == NULL)
	{
		Size		codeBytes;

		if (state->dimensions <= 0 ||
			state->nodes[left].code == NULL ||
			state->nodes[right].code == NULL)
			return false;

		codeBytes = PgturbohybridGraphCodeBytesForBits(state->dimensions,
											 state->tqBits);
		return state->nodes[left].scale == state->nodes[right].scale &&
			state->nodes[left].norm == state->nodes[right].norm &&
			state->nodes[left].correction == state->nodes[right].correction &&
			state->nodes[left].ecCorrection == state->nodes[right].ecCorrection &&
			memcmp(state->nodes[left].code, state->nodes[right].code,
				   codeBytes) == 0;
	}
	if (leftVector->dim != rightVector->dim)
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

static bool
PgturbohybridGraphCanBuildCodeOnly(PgturbohybridQuantBuildState *state)
{
	TqScoreMode mode = (TqScoreMode) state->scoreMode;

	if (state->tqExactStorage || state->buildExactDistances)
		return false;
	if (state->tqBits != 1 && state->tqBits != 2 &&
		state->tqBits != PGTURBOHYBRID_DEFAULT_BITS &&
		state->tqBits != 8)
		return false;
	return mode == PGTURBOHYBRID_SCORE_L2 ||
		mode == PGTURBOHYBRID_SCORE_COSINE ||
		mode == PGTURBOHYBRID_SCORE_IP;
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

bool
PgturbohybridGraphLoadPayloadValue(Relation index, PgturbohybridGraphScanOpaque so,
								   PgturbohybridGraphMetaPageData *meta,
								   PgturbohybridGraphScanStorage *storage,
								   uint32 nodeId, int payloadSlot,
								   int32 *payloadValue)
{
	PgturbohybridGraphScanNode *node;

	if (payloadValue == NULL || meta == NULL || storage == NULL)
		return false;
	if (payloadSlot < 0 || payloadSlot >= PGTURBOHYBRID_GRAPH_MAX_PAYLOADS)
		return false;
	if ((uint16) payloadSlot >= meta->tqPayloadCount)
		return false;
	if (nodeId >= meta->tqNodeCount)
		return false;
	if (!PgturbohybridGraphLoadCodePage(index, so, meta, storage, nodeId))
		return false;

	node = &storage->nodes[nodeId];
	if (node->payloads == NULL)
		return false;
	if ((node->payloadMask & (uint16) (1U << payloadSlot)) == 0)
		return false;

	*payloadValue = node->payloads[payloadSlot];
	return true;
}

typedef struct PgturbohybridGraphRepairCandidate
{
	uint32		nodeId;
	double		distance;
	bool		direct;
} PgturbohybridGraphRepairCandidate;

static int
PgturbohybridGraphRepairCandidateCompare(const void *a, const void *b)
{
	const PgturbohybridGraphRepairCandidate *ca =
		(const PgturbohybridGraphRepairCandidate *) a;
	const PgturbohybridGraphRepairCandidate *cb =
		(const PgturbohybridGraphRepairCandidate *) b;

	if (ca->distance < cb->distance)
		return -1;
	if (ca->distance > cb->distance)
		return 1;
	return (ca->nodeId > cb->nodeId) - (ca->nodeId < cb->nodeId);
}

static bool
PgturbohybridGraphRepairHasDirectNeighbor(PgturbohybridGraphScanStorage *storage,
										  PgturbohybridGraphMetaPageData *meta,
										  uint32 sampleNodeId,
										  uint32 nodeId)
{
	int			slot = PgturbohybridGraphScanAdjSlot(meta, sampleNodeId, 0);

	for (int i = 0; i < storage->neighborCounts[slot]; i++)
	{
		if (storage->neighbors[slot][i] == nodeId)
			return true;
	}

	return false;
}

static double
PgturbohybridGraphRepairCodeDistance(PgturbohybridGraphScanStorage *storage,
									 PgturbohybridGraphMetaPageData *meta,
									 uint32 a, uint32 b)
{
	uint8	   *acode;
	uint8	   *bcode;
	double		distance = 0.0;

	if (a >= meta->tqNodeCount || b >= meta->tqNodeCount ||
		storage->nodes[a].code == NULL ||
		storage->nodes[b].code == NULL ||
		meta->tqCodeBytes == 0)
		return DBL_MAX;

	acode = storage->nodes[a].code;
	bcode = storage->nodes[b].code;
	for (uint16 i = 0; i < meta->tqCodeBytes; i++)
	{
		double		delta = (double) acode[i] - (double) bcode[i];

		distance += delta * delta;
	}

	return distance;
}

static bool
PgturbohybridGraphRepairCandidateSeen(PgturbohybridGraphRepairCandidate *candidates,
									  int count, uint32 nodeId)
{
	for (int i = 0; i < count; i++)
	{
		if (candidates[i].nodeId == nodeId)
			return true;
	}

	return false;
}

static void
PgturbohybridGraphRepairOfferCandidate(Relation index,
									   PgturbohybridGraphScanOpaque so,
									   PgturbohybridGraphMetaPageData *meta,
									   PgturbohybridGraphScanStorage *storage,
									   PgturbohybridGraphRepairCandidate *candidates,
									   int *candidateCount,
									   int candidateLimit,
									   uint32 sampleNodeId,
									   uint32 nodeId)
{
	double		distance;
	bool		direct;

	if (candidateLimit <= 0 ||
		nodeId >= meta->tqNodeCount ||
		nodeId == sampleNodeId ||
		PgturbohybridGraphRepairCandidateSeen(candidates, *candidateCount,
											  nodeId))
		return;
	if (!PgturbohybridGraphLoadCodePage(index, so, meta, storage, nodeId))
		return;
	if (storage->nodes[nodeId].flags & PGTURBOHYBRID_GRAPH_NODE_DEAD)
		return;

	distance = PgturbohybridGraphRepairCodeDistance(storage, meta, sampleNodeId,
													nodeId);
	if (distance == DBL_MAX || isnan(distance) || isinf(distance))
		return;
	direct = PgturbohybridGraphRepairHasDirectNeighbor(storage, meta,
													   sampleNodeId, nodeId);

	if (*candidateCount < candidateLimit)
	{
		candidates[*candidateCount].nodeId = nodeId;
		candidates[*candidateCount].distance = distance;
		candidates[*candidateCount].direct = direct;
		(*candidateCount)++;
		return;
	}

	for (int i = 0; i < candidateLimit; i++)
	{
		if (distance < candidates[i].distance ||
			(distance == candidates[i].distance && nodeId < candidates[i].nodeId))
		{
			candidates[i].nodeId = nodeId;
			candidates[i].distance = distance;
			candidates[i].direct = direct;
			return;
		}
	}
}

static int
PgturbohybridGraphRepairCollectCandidates(Relation index,
										  PgturbohybridGraphScanOpaque so,
										  PgturbohybridGraphMetaPageData *meta,
										  PgturbohybridGraphScanStorage *storage,
										  uint32 sampleNodeId,
										  PgturbohybridGraphRepairCandidate *candidates,
										  int candidateLimit)
{
	int			candidateCount = 0;
	int			slot;

	if (candidateLimit <= 0 ||
		sampleNodeId >= meta->tqNodeCount ||
		!PgturbohybridGraphLoadCodePage(index, so, meta, storage, sampleNodeId) ||
		!PgturbohybridGraphLoadAdjPage(index, so, meta, storage, sampleNodeId, 0))
		return 0;

	slot = PgturbohybridGraphScanAdjSlot(meta, sampleNodeId, 0);
	for (int i = 0; i < storage->neighborCounts[slot]; i++)
		PgturbohybridGraphRepairOfferCandidate(index, so, meta, storage,
											   candidates, &candidateCount,
											   candidateLimit, sampleNodeId,
											   storage->neighbors[slot][i]);

	for (int i = 0; i < storage->neighborCounts[slot]; i++)
	{
		uint32		neighbor = storage->neighbors[slot][i];
		int			neighborSlot;

		CHECK_FOR_INTERRUPTS();
		if (neighbor >= meta->tqNodeCount ||
			!PgturbohybridGraphLoadAdjPage(index, so, meta, storage, neighbor, 0))
			continue;
		neighborSlot = PgturbohybridGraphScanAdjSlot(meta, neighbor, 0);
		for (int j = 0; j < storage->neighborCounts[neighborSlot]; j++)
			PgturbohybridGraphRepairOfferCandidate(index, so, meta, storage,
												   candidates, &candidateCount,
												   candidateLimit, sampleNodeId,
												   storage->neighbors[neighborSlot][j]);
	}

	if (meta->tqEntryNodeId < meta->tqNodeCount)
		PgturbohybridGraphRepairOfferCandidate(index, so, meta, storage,
											   candidates, &candidateCount,
											   candidateLimit, sampleNodeId,
											   meta->tqEntryNodeId);
	for (uint16 i = 0; i < meta->tqRoutingEntryCount; i++)
		PgturbohybridGraphRepairOfferCandidate(index, so, meta, storage,
											   candidates, &candidateCount,
											   candidateLimit, sampleNodeId,
											   meta->tqRoutingEntryNodeIds[i]);
	for (uint16 i = 0; i < meta->tqEntrySidecarCount; i++)
		PgturbohybridGraphRepairOfferCandidate(index, so, meta, storage,
											   candidates, &candidateCount,
											   candidateLimit, sampleNodeId,
											   meta->tqEntrySidecarNodeIds[i]);
	for (uint16 i = 0; i < meta->tqSegmentCount; i++)
		PgturbohybridGraphRepairOfferCandidate(index, so, meta, storage,
											   candidates, &candidateCount,
											   candidateLimit, sampleNodeId,
											   meta->tqSegments[i].entryNodeId);

	if (candidateCount > 1)
		qsort(candidates, candidateCount,
			  sizeof(PgturbohybridGraphRepairCandidate),
			  PgturbohybridGraphRepairCandidateCompare);

	return candidateCount;
}

void
PgturbohybridGraphRepairDryRun(Relation index, int sampleNodes, int searchEf,
							   int candidateLimit,
							   PgturbohybridGraphRepairDryRunStats *stats)
{
	PgturbohybridGraphMetaPageData meta;
	PgturbohybridGraphScanStorage storage;
	PgturbohybridGraphScanOpaqueData so;
	PgturbohybridGraphRepairCandidate *candidates;
	int			targetSamples;
	int			effectiveCandidateLimit;
	uint32		step;
	uint32		start;
	volatile double overlapSum = 0.0;
	instr_time	startTime;

	memset(stats, 0, sizeof(*stats));
	stats->sampleNodesRequested = sampleNodes;
	stats->searchEf = searchEf;
	stats->candidateLimit = candidateLimit;

	if (sampleNodes < 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("sample_nodes must be non-negative")));
	if (searchEf <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("search_ef must be positive")));
	if (candidateLimit <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("candidate_limit must be positive")));

	INSTR_TIME_SET_CURRENT(startTime);
	if (!PgturbohybridGraphReadMeta(index, &meta))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("turbohybrid_graph_repair_dry_run only supports native graph indexes")));

	stats->nodeCount = meta.tqNodeCount;
	stats->dimensions = meta.dimensions;
	if (meta.tqNodeCount == 0)
	{
		stats->elapsedMs = PgturbohybridGraphElapsedUs(startTime) / 1000;
		return;
	}

	targetSamples = Min(sampleNodes, (int) meta.tqNodeCount);
	if (targetSamples == 0)
	{
		stats->elapsedMs = PgturbohybridGraphElapsedUs(startTime) / 1000;
		return;
	}

	effectiveCandidateLimit = Min(candidateLimit, searchEf);
	effectiveCandidateLimit = Min(effectiveCandidateLimit, (int) meta.tqNodeCount);
	stats->candidateLimit = effectiveCandidateLimit;

	memset(&so, 0, sizeof(so));
	PgturbohybridGraphInitScanStorage(index, &meta, &storage, NULL);
	candidates = palloc0(sizeof(*candidates) * effectiveCandidateLimit);
	step = Max(1U, meta.tqNodeCount / (uint32) targetSamples);
	start = step > 1 ? step / 2 : 0;
	Assert(meta.tqNodeCount > 0);

	LockPage(index, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ShareLock);
	PG_TRY();
	{
		for (int sample = 0; sample < targetSamples; sample++)
		{
			uint32		nodeId = start + (uint32) sample * step;
			int			slot;
			int			directCount;
			int			candidateCount;
			int			compareCount;
			int			overlap = 0;
			int			missed = 0;

			CHECK_FOR_INTERRUPTS();
			if (nodeId >= meta.tqNodeCount)
				nodeId = (uint32) sample % meta.tqNodeCount;
			if (!PgturbohybridGraphLoadCodePage(index, &so, &meta, &storage, nodeId) ||
				(storage.nodes[nodeId].flags & PGTURBOHYBRID_GRAPH_NODE_DEAD) ||
				!PgturbohybridGraphLoadAdjPage(index, &so, &meta, &storage, nodeId, 0))
				continue;

			memset(candidates, 0, sizeof(*candidates) * effectiveCandidateLimit);
			slot = PgturbohybridGraphScanAdjSlot(&meta, nodeId, 0);
			directCount = storage.neighborCounts[slot];
			candidateCount = PgturbohybridGraphRepairCollectCandidates(index, &so,
																	   &meta,
																	   &storage,
																	   nodeId,
																	   candidates,
																	   effectiveCandidateLimit);
			compareCount = Min(directCount, candidateCount);
			for (int i = 0; i < compareCount; i++)
			{
				if (candidates[i].direct)
					overlap++;
				else
					missed++;
			}

			stats->sampledNodes++;
			stats->missedNeighbors += missed;
			stats->suggestedEdges += missed;
			if (directCount == 0 || candidateCount == 0 ||
				(compareCount > 0 &&
				 (double) overlap / (double) compareCount < 0.5))
			{
				stats->weakNodes++;
				if (directCount == 0 || candidateCount == 0)
					stats->weakEntryCases++;
			}
			if (compareCount > 0)
				overlapSum += (double) overlap / (double) compareCount;
		}
	}
	PG_CATCH();
	{
		UnlockPage(index, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ShareLock);
		PG_RE_THROW();
	}
	PG_END_TRY();
	UnlockPage(index, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ShareLock);

	stats->codePagesRead = so.graphCodePagesRead;
	stats->adjPagesRead = so.graphAdjPagesRead;
	stats->avgOverlap = stats->sampledNodes > 0 ?
		overlapSum / (double) stats->sampledNodes : 0.0;
	stats->elapsedMs = PgturbohybridGraphElapsedUs(startTime) / 1000;
}


static uint64
PgturbohybridGraphMix64(uint64 x)
{
	x += UINT64CONST(0x9e3779b97f4a7c15);
	x = (x ^ (x >> 30)) * UINT64CONST(0xbf58476d1ce4e5b9);
	x = (x ^ (x >> 27)) * UINT64CONST(0x94d049bb133111eb);
	return x ^ (x >> 31);
}

static uint64
PgturbohybridGraphBuildCodeHash(PgturbohybridQuantBuildState *state, uint32 nodeId)
{
	PgturbohybridGraphBuildNode *node = &state->nodes[nodeId];
	Size		codeBytes = state->dimensions > 0 ?
		PgturbohybridGraphCodeBytesForBits(state->dimensions, state->tqBits) : 0;
	uint64		hash = UINT64CONST(1469598103934665603);
	const unsigned char *bytes;

#define PGTURBOHYBRID_GRAPH_FNV_BYTES(ptr, len) \
	do { \
		bytes = (const unsigned char *) (ptr); \
		for (Size _i = 0; _i < (Size) (len); _i++) \
		{ \
			hash ^= bytes[_i]; \
			hash *= UINT64CONST(1099511628211); \
		} \
	} while (0)

	if (node->code != NULL && codeBytes > 0)
		PGTURBOHYBRID_GRAPH_FNV_BYTES(node->code, codeBytes);
	PGTURBOHYBRID_GRAPH_FNV_BYTES(&node->scale, sizeof(node->scale));
	PGTURBOHYBRID_GRAPH_FNV_BYTES(&node->norm, sizeof(node->norm));
	PGTURBOHYBRID_GRAPH_FNV_BYTES(&node->correction, sizeof(node->correction));
	if (state->tqWeighted)
		PGTURBOHYBRID_GRAPH_FNV_BYTES(&node->ecCorrection, sizeof(node->ecCorrection));

#undef PGTURBOHYBRID_GRAPH_FNV_BYTES

	return hash;
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
	uint32		oldCapacity = state->nodeCapacity;

	if (state->nodeCount < state->nodeCapacity)
		return;

	if (state->nodeCapacity == 0)
		state->nodeCapacity = 1024;
	else
	{
		if (state->nodeCapacity > PG_UINT32_MAX / 2)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("pgturbohybrid graph build node capacity is too large")));
		state->nodeCapacity += state->nodeCapacity;
	}

	/*
	 * Zero the newly grown slots.  Build nodes are written verbatim into index
	 * code pages (PgturbohybridGraphWriteCodePages), and not every field is set
	 * for every node -- e.g. payloadMask is only assigned when the row carries
	 * payloads -- so a plain repalloc would leak uninitialized bytes onto disk
	 * (caught by valgrind's PageAddItem check).  The initial block is
	 * MemoryContextAllocZero'd; keep that invariant as the array doubles.
	 *
	 * repalloc0() would do this in one call but only exists on PostgreSQL 16+;
	 * repalloc + memset of the grown range works on every supported version.
	 */
	{
		Size		oldBytes = PgturbohybridGraphArrayAllocSize(sizeof(PgturbohybridGraphBuildNode),
																oldCapacity);
		Size		newBytes = PgturbohybridGraphArrayAllocSize(sizeof(PgturbohybridGraphBuildNode),
																state->nodeCapacity);

		state->nodes = repalloc(state->nodes, newBytes);
		memset((char *) state->nodes + oldBytes, 0, newBytes - oldBytes);
		if (state->multivectorBuild && state->multivectorNodeMap != NULL)
		{
			oldBytes = PgturbohybridGraphArrayAllocSize(sizeof(TqMultiVectorNodeMapEntry),
														oldCapacity);
			newBytes = PgturbohybridGraphArrayAllocSize(sizeof(TqMultiVectorNodeMapEntry),
														state->nodeCapacity);
			state->multivectorNodeMap = repalloc(state->multivectorNodeMap,
												 newBytes);
			memset((char *) state->multivectorNodeMap + oldBytes, 0,
				   newBytes - oldBytes);
		}
	}
}

static void
PgturbohybridGraphEnsureMultiVectorNodeMapCapacity(PgturbohybridQuantBuildState *state)
{
	Size		mapBytes;

	if (!state->multivectorBuild)
		return;

	mapBytes = PgturbohybridGraphArrayAllocSize(sizeof(TqMultiVectorNodeMapEntry),
												state->nodeCapacity);
	if (state->multivectorNodeMap == NULL)
		state->multivectorNodeMap = MemoryContextAllocZero(state->ctx, mapBytes);
}

static void
PgturbohybridGraphEnsureMultiVectorDocCapacity(PgturbohybridQuantBuildState *state)
{
	uint32		oldCapacity = state->multivectorDocCapacity;

	if (!state->multivectorBuild ||
		state->multivectorDocCount < state->multivectorDocCapacity)
		return;

	if (state->multivectorDocCapacity == 0)
		state->multivectorDocCapacity = 128;
	else
	{
		if (state->multivectorDocCapacity > PG_UINT32_MAX / 2)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("pgturbohybrid multivector document map is too large")));
		state->multivectorDocCapacity += state->multivectorDocCapacity;
	}

	{
		Size		oldBytes = PgturbohybridGraphArrayAllocSize(sizeof(TqMultiVectorDocMapEntry),
																oldCapacity);
		Size		newBytes = PgturbohybridGraphArrayAllocSize(sizeof(TqMultiVectorDocMapEntry),
																state->multivectorDocCapacity);
		Size		oldVectorBytes = PgturbohybridGraphArrayAllocSize(sizeof(PgturbohybridMultiVector *),
																	 oldCapacity);
		Size		newVectorBytes = PgturbohybridGraphArrayAllocSize(sizeof(PgturbohybridMultiVector *),
																	 state->multivectorDocCapacity);
		Size		oldFloatBytes = PgturbohybridGraphArrayAllocSize(sizeof(float),
																	oldCapacity);
		Size		newFloatBytes = PgturbohybridGraphArrayAllocSize(sizeof(float),
																	state->multivectorDocCapacity);

		if (state->multivectorDocMap == NULL)
			state->multivectorDocMap = MemoryContextAllocZero(state->ctx, newBytes);
		else
		{
			state->multivectorDocMap = repalloc(state->multivectorDocMap, newBytes);
			memset((char *) state->multivectorDocMap + oldBytes, 0, newBytes - oldBytes);
		}
		if (state->multivectorDocVectors == NULL)
			state->multivectorDocVectors =
				MemoryContextAllocZero(state->ctx, newVectorBytes);
		else
		{
			state->multivectorDocVectors =
				repalloc(state->multivectorDocVectors, newVectorBytes);
			memset((char *) state->multivectorDocVectors + oldVectorBytes, 0,
				   newVectorBytes - oldVectorBytes);
		}
		if (state->multivectorDocCentroids == NULL)
			state->multivectorDocCentroids =
				MemoryContextAllocZero(state->ctx, newVectorBytes);
		else
		{
			state->multivectorDocCentroids =
				repalloc(state->multivectorDocCentroids, newVectorBytes);
			memset((char *) state->multivectorDocCentroids + oldVectorBytes, 0,
				   newVectorBytes - oldVectorBytes);
		}
		if (state->multivectorDocCentroidResiduals == NULL)
			state->multivectorDocCentroidResiduals =
				MemoryContextAllocZero(state->ctx, newFloatBytes);
		else
		{
			state->multivectorDocCentroidResiduals =
				repalloc(state->multivectorDocCentroidResiduals, newFloatBytes);
			memset((char *) state->multivectorDocCentroidResiduals + oldFloatBytes,
				   0, newFloatBytes - oldFloatBytes);
		}
	}
}

static uint8 *
PgturbohybridGraphAllocBuildCode(PgturbohybridQuantBuildState *state, int dimensions)
{
	return palloc0(PgturbohybridGraphCodeBytesForBits(dimensions,
													  state->tqBits));
}

static void
PgturbohybridGraphAllocateBuildNeighbors(PgturbohybridQuantBuildState *state,
							   PgturbohybridGraphBuildNode *node)
{
	int			levelCount = node->level + 1;
	Size		pointerBytes = MAXALIGN(sizeof(uint32 *) * levelCount);
	Size		distancePointerBytes = MAXALIGN(sizeof(double *) * levelCount);
	Size		countBytes = MAXALIGN(sizeof(int) * levelCount);
	Size		slotBytes = 0;
	Size		distanceSlotBytes = 0;
	char	   *storage;
	char	   *cursor;
	char	   *distanceCursor;

	for (int i = 0; i <= node->level; i++)
	{
		slotBytes += sizeof(uint32) * (PgturbohybridGraphLevelM(state->m, i) + 1);
		distanceSlotBytes += sizeof(double) * (PgturbohybridGraphLevelM(state->m, i) + 1);
	}
	slotBytes = MAXALIGN(slotBytes);
	distanceSlotBytes = MAXALIGN(distanceSlotBytes);

	storage = palloc0(pointerBytes + distancePointerBytes + countBytes +
					  slotBytes + distanceSlotBytes);
	node->neighbors = (uint32 **) storage;
	node->neighborDistances = (double **) (storage + pointerBytes);
	node->neighborCounts = (int *) (storage + pointerBytes + distancePointerBytes);
	cursor = storage + pointerBytes + distancePointerBytes + countBytes;
	distanceCursor = cursor + slotBytes;
	for (int i = 0; i <= node->level; i++)
	{
		int			levelSlots = PgturbohybridGraphLevelM(state->m, i) + 1;

		node->neighbors[i] = (uint32 *) cursor;
		node->neighborDistances[i] = (double *) distanceCursor;
		cursor += sizeof(uint32) * levelSlots;
		distanceCursor += sizeof(double) * levelSlots;
	}
}

static uint32
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
	PgturbohybridGraphEnsureMultiVectorNodeMapCapacity(state);
	vectorSize = PGTURBOHYBRID_VECTOR_SIZE(vector->dim);
	vectorHash = PgturbohybridGraphBuildVectorHash(vector);
	if (state->nodeCount > 0)
	{
		prev = &state->nodes[state->nodeCount - 1];
		duplicatePrevious =
			!state->buildCodeOnly &&
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
		node->residualSketch = prev->residualSketch;
	}
	else
	{
		if (state->buildCodeOnly)
			node->vector = NULL;
		else
		{
			node->vector = palloc(vectorSize);
			memcpy(node->vector, vector, vectorSize);
		}
		node->code = PgturbohybridGraphAllocBuildCode(state, vector->dim);
		if (state->residualRerankBytes > 0)
		{
			node->residualSketch = palloc0(state->residualRerankBytes);
			PgturbohybridGraphBuildResidualSketch(vector->x, vector->dim,
												  node->residualSketch,
												  state->residualRerankBytes);
		}
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
	PgturbohybridGraphAllocateBuildNeighbors(state, node);
	state->maxLevel = Max(state->maxLevel, level);

	if (state->buildEncodeOnAppend)
		PgturbohybridGraphEncodeBuildNode(state, node, vector);

	return nodeId;
}

static bool
PgturbohybridGraphIndexIsMultiVector(Relation index)
{
	TupleDesc	desc = RelationGetDescr(index);

	return PgturbohybridTypeIsMultiVector(
		TupleDescAttr(desc, PGTURBOHYBRID_DENSE_KEY_INDEX)->atttypid);
}

static void
PgturbohybridGraphStreamFitMultiVector(PgturbohybridQuantBuildState *state,
									   const PgturbohybridMultiVector *mv)
{
	Vector	   *vector;

	vector = MemoryContextAlloc(state->buildTupleCtx,
								PgturbohybridMultiVectorSubvectorSize(mv));
	for (int32 i = 0; i < mv->count; i++)
	{
		PgturbohybridMultiVectorCopySubvectorToVector(mv, i, vector);
		if (state->scoreMode == PGTURBOHYBRID_SCORE_COSINE ||
			state->scoreMode == PGTURBOHYBRID_SCORE_IP)
			PgturbohybridGraphStreamFitVector(state, vector);
		else
			state->fitCount++;
	}
}

static void
PgturbohybridGraphAppendBuildMultiVector(PgturbohybridQuantBuildState *state,
										 ItemPointer tid,
										 const PgturbohybridMultiVector *mv,
										 Datum *values, bool *isnull)
{
	TqDocId		docId;
	TqMultiVectorDocMapEntry *docEntry;
	Vector	   *vector;

	PgturbohybridMultiVectorCheckDim((uint32) mv->dim,
									 (uint32) pgturbohybrid_multivector_max_dim);
	PgturbohybridMultiVectorCheckTokenCount((uint32) mv->count,
											(uint32) pgturbohybrid_multivector_max_doc_vectors);
	docId = PgturbohybridMultiVectorMakeDocId(state->multivectorDocCount);
	PgturbohybridGraphEnsureMultiVectorDocCapacity(state);
	docEntry = &state->multivectorDocMap[state->multivectorDocCount++];
	docEntry->heapTid = *tid;
	docEntry->firstNodeId = state->nodeCount;
	docEntry->tokenCount = (uint16) mv->count;
	docEntry->originalTokenCount = (uint16) mv->count;
	docEntry->pooledTokenCount = (uint16) mv->count;

	vector = MemoryContextAlloc(state->buildTupleCtx,
								PgturbohybridMultiVectorSubvectorSize(mv));
	for (int32 i = 0; i < mv->count; i++)
	{
		uint32		nodeId;

		PgturbohybridMultiVectorCopySubvectorToVector(mv, i, vector);
		nodeId = PgturbohybridGraphAppendBuildNode(state, tid,
												   PointerGetDatum(vector),
												   values, isnull);
		state->multivectorNodeMap[nodeId].docId = docId;
		state->multivectorNodeMap[nodeId].tokenOrdinal =
			PgturbohybridMultiVectorMakeSubvectorOrdinal((uint32) i);
	}
}

static void
PgturbohybridGraphStreamFitMultiVectorDocument(PgturbohybridQuantBuildState *state,
											   const PgturbohybridMultiVector *mv)
{
	PgturbohybridMultiVector *indexedMv;
	const PgturbohybridMultiVector *proxySource;
	Vector	   *vector;

	indexedMv =
		PgturbohybridMultiVectorPoolDocumentTokens(mv,
												   state->multivectorTokenPooling,
												   state->multivectorTokenPoolingTargetRatio,
												   state->multivectorTokenPoolingMinTokens,
												   state->buildTupleCtx);
	proxySource =
		state->multivectorProxyEncoder ==
		PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_CENTROID_MEAN ?
		mv : indexedMv;
	vector =
		PgturbohybridMultiVectorBuildProxyVectorWithCentroids(proxySource,
															  NULL,
															  state->multivectorProxyEncoder,
															  state->multivectorCentroidCount,
															  state->buildTupleCtx);
	if (state->scoreMode == PGTURBOHYBRID_SCORE_COSINE ||
		state->scoreMode == PGTURBOHYBRID_SCORE_IP)
		PgturbohybridGraphStreamFitVector(state, vector);
	else
		state->fitCount++;
}

static void
PgturbohybridGraphStoreBuildMultiVectorCentroids(PgturbohybridQuantBuildState *state,
												 uint32 docOrdinal,
												 const PgturbohybridMultiVector *doc)
{
	PgturbohybridMultiVector *centroids;
	PgturbohybridMultiVector *stored;
	Size		centroidSize;
	int			centroidCount;
	double		targetRatio;
	instr_time	start;
	instr_time	clusterStart;
	instr_time	residualStart;

	if (state->multivectorCentroids !=
		PGTURBOHYBRID_MULTIVECTOR_CENTROIDS_KMEANS)
		return;
	if (state->multivectorGraphMode !=
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES)
		return;
	if (doc == NULL || docOrdinal >= state->multivectorDocCapacity)
		elog(ERROR, "invalid multivector centroid sidecar document");

	centroidCount =
		PgturbohybridMultiVectorCentroidCountForDoc(doc,
													state->multivectorCentroidCount);
	if (centroidCount <= 0)
		elog(ERROR, "invalid multivector centroid count");
	INSTR_TIME_SET_CURRENT(start);
	targetRatio = (double) centroidCount / (double) doc->count;
	INSTR_TIME_SET_CURRENT(clusterStart);
	centroids =
		PgturbohybridMultiVectorPoolDocumentTokens(doc,
												   PGTURBOHYBRID_MULTIVECTOR_TOKEN_POOLING_KMEANS,
												   targetRatio,
												   1,
												   state->buildTupleCtx);
	state->multivectorCentroidClusterUs +=
		PgturbohybridGraphElapsedUs(clusterStart);
	centroidSize = VARSIZE_ANY(centroids);
	stored = MemoryContextAlloc(state->ctx, centroidSize);
	memcpy(stored, centroids, centroidSize);
	state->multivectorDocCentroids[docOrdinal] = stored;
	state->multivectorCentroidVectorBytes =
		PgturbohybridGraphBuildAddEstimate(state->multivectorCentroidVectorBytes,
										   (uint64) centroidSize);
	INSTR_TIME_SET_CURRENT(residualStart);
	state->multivectorDocCentroidResiduals[docOrdinal] =
		PgturbohybridMultiVectorCentroidResidualMean(doc, stored);
	state->multivectorCentroidResidualUs +=
		PgturbohybridGraphElapsedUs(residualStart);
	state->multivectorCentroidBuildUs += PgturbohybridGraphElapsedUs(start);
	state->multivectorCentroidBuildDocs++;
	state->multivectorCentroidBuildVectors += (uint64) stored->count;
}

static void
PgturbohybridGraphAppendBuildMultiVectorDocument(PgturbohybridQuantBuildState *state,
												 ItemPointer tid,
												 const PgturbohybridMultiVector *mv,
												 Datum *values, bool *isnull)
{
	TqDocId		docId;
	TqMultiVectorDocMapEntry *docEntry;
	PgturbohybridMultiVector *indexedMv;
	Vector	   *vector;
	PgturbohybridMultiVector *stored;
	PgturbohybridMultiVector *centroids;
	Size		mvSize;
	uint32		nodeId;
	uint32		docOrdinal;
	instr_time	proxyStart;

	PgturbohybridMultiVectorCheckDim((uint32) mv->dim,
									 (uint32) pgturbohybrid_multivector_max_dim);
	PgturbohybridMultiVectorCheckTokenCount((uint32) mv->count,
											(uint32) pgturbohybrid_multivector_max_doc_vectors);
	indexedMv =
		PgturbohybridMultiVectorPoolDocumentTokens(mv,
												   state->multivectorTokenPooling,
												   state->multivectorTokenPoolingTargetRatio,
												   state->multivectorTokenPoolingMinTokens,
												   state->buildTupleCtx);
	docId = PgturbohybridMultiVectorMakeDocId(state->multivectorDocCount);
	PgturbohybridGraphEnsureMultiVectorDocCapacity(state);
	docOrdinal = state->multivectorDocCount;
	docEntry = &state->multivectorDocMap[docOrdinal];
	docEntry->heapTid = *tid;
	docEntry->firstNodeId = state->nodeCount;
	docEntry->tokenCount = (uint16) mv->count;
	docEntry->originalTokenCount = (uint16) mv->count;
	docEntry->pooledTokenCount = (uint16) indexedMv->count;

	PgturbohybridGraphAddMultiVectorDocMapBytesEstimate(state,
														(uint64) PgturbohybridGraphMultiVectorDocMapNodeTupleSize(1) +
														(uint64) PgturbohybridGraphMultiVectorDocMapDocTupleSize(1),
														"document-node map entries");
	if (state->multivectorDocStorage ==
		PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_PROXY_ONLY)
	{
		stored = NULL;
		centroids = NULL;
	}
	else
	{
		Size		totalFloats = 0;

		if (PgturbohybridGraphMultiVectorDocMapStoresDocVectors(state))
		{
				totalFloats =
					PgturbohybridMultiVectorFloatCount(mv->count, mv->dim);
				PgturbohybridGraphAddMultiVectorDocMapBytesEstimate(state,
																	PgturbohybridGraphMultiVectorDocMapChunkedVectorTupleBytes(totalFloats,
																															   state->multivectorDocStorage),
																	"document multivector sidecar");
			if (PgturbohybridMultiVectorHasContexts(mv))
			{
				int32		contextCount =
					PgturbohybridMultiVectorContextCount(mv);
				const int32 *fields =
					PgturbohybridMultiVectorContextFields(mv);

				PgturbohybridGraphAddMultiVectorDocMapBytesEstimate(state,
																	(uint64) PgturbohybridGraphMultiVectorDocMapContextTupleSize((uint16) contextCount,
																																 fields != NULL),
																	"document context sidecar");
			}
		}
		if (state->multivectorCentroids ==
			PGTURBOHYBRID_MULTIVECTOR_CENTROIDS_KMEANS)
		{
			int			centroidCount =
				PgturbohybridMultiVectorCentroidCountForDoc(mv,
															state->multivectorCentroidCount);
			Size		centroidFloats;
			uint64		postingBytes;

			if (centroidCount <= 0)
				elog(ERROR, "invalid multivector centroid count");
			centroidFloats =
				PgturbohybridMultiVectorFloatCount(centroidCount, mv->dim);
			PgturbohybridGraphAddMultiVectorDocMapBytesEstimate(state,
																state->multivectorDocStorage ==
																PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CENTROID_ONLY ?
																PgturbohybridGraphMultiVectorDocMapChunkedFloatTupleBytes(centroidFloats,
																														  PgturbohybridGraphMultiVectorDocMapCentroidF16TupleMaxCount(),
																														  PgturbohybridGraphMultiVectorDocMapCentroidF16TupleSize) :
																PgturbohybridGraphMultiVectorDocMapChunkedFloatTupleBytes(centroidFloats,
																														  PgturbohybridGraphMultiVectorDocMapCentroidTupleMaxCount(),
																														  PgturbohybridGraphMultiVectorDocMapCentroidTupleSize),
																"document centroid sidecar");
			postingBytes =
				(uint64) centroidCount *
				(uint64) PgturbohybridGraphMultiVectorDocMapCentroidPostingTupleSize(1);
			PgturbohybridGraphAddMultiVectorDocMapBytesEstimate(state,
																postingBytes,
																"centroid posting sidecar");
		}
		if (PgturbohybridGraphMultiVectorDocMapStoresQuantizedInvertedPostings(state))
		{
			uint64		postingBytes =
				(uint64) mv->count *
				(uint64) PgturbohybridGraphMultiVectorDocMapQuantizedPostingTupleSize(1);

			PgturbohybridGraphAddMultiVectorDocMapBytesEstimate(state,
																postingBytes,
																"quantized inverted posting sidecar");
		}
		if (PgturbohybridGraphMultiVectorDocMapStoresDocVectors(state))
		{
			mvSize = VARSIZE_ANY(mv);
			stored = MemoryContextAlloc(state->ctx, mvSize);
			memcpy(stored, mv, mvSize);
			state->multivectorDocVectors[docOrdinal] = stored;
			state->multivectorDocVectorPayloadBytesEstimate =
				PgturbohybridGraphBuildAddEstimate(state->multivectorDocVectorPayloadBytesEstimate,
												   (uint64) mvSize);
		}
		else
			stored = NULL;
		PgturbohybridGraphStoreBuildMultiVectorCentroids(state, docOrdinal,
														stored != NULL ? stored : (PgturbohybridMultiVector *) mv);
		centroids = state->multivectorDocCentroids[docOrdinal];
		PgturbohybridGraphAppendMultiVectorQuantizedInvertedBuildPostings(state,
																		  mv,
																		  docOrdinal);
	}
	state->multivectorDocCount++;

	INSTR_TIME_SET_CURRENT(proxyStart);
	vector =
		PgturbohybridMultiVectorBuildProxyVectorWithCentroids(indexedMv,
															  centroids,
															  state->multivectorProxyEncoder,
															  state->multivectorCentroidCount,
															  state->buildTupleCtx);
	{
		uint64		proxyEncodeUs = PgturbohybridGraphElapsedUs(proxyStart);

		state->multivectorProxyBuildUs += proxyEncodeUs;
		if (state->multivectorProxyEncoder ==
			PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_LEARNED_PROJECTION_V1)
			state->learnedProjectionDocEncodeBuildUs += proxyEncodeUs;
	}
	nodeId = PgturbohybridGraphAppendBuildNode(state, tid,
											   PointerGetDatum(vector),
											   values, isnull);
	state->multivectorNodeMap[nodeId].docId = docId;
	state->multivectorNodeMap[nodeId].tokenOrdinal = 0;
}

static void
PgturbohybridGraphValidateMultiVectorBuildOptions(PgturbohybridQuantBuildState *state)
{
	bool		quantizedPostingOnly;

	if (!state->multivectorBuild)
		return;
	quantizedPostingOnly =
		state->multivectorDocStorage ==
		PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CENTROID_ONLY &&
		state->multivectorCentroids !=
		PGTURBOHYBRID_MULTIVECTOR_CENTROIDS_KMEANS &&
		pgturbohybrid_multivector_quantized_inverted_codebook ==
		PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_EXTERNAL;
	if (state->multivectorDocStorage ==
		PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_PROXY_ONLY &&
		state->multivectorGraphMode !=
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("multivector_doc_storage = proxy_only requires multivector_graph = document_nodes"),
				 errhint("Use multivector_doc_storage = f32, f16, or sq8 for token-node indexes.")));
	if (state->multivectorDocStorage ==
		PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CENTROID_ONLY &&
		state->multivectorGraphMode !=
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("multivector_doc_storage = centroid_only requires multivector_graph = document_nodes"),
				 errhint("Use multivector_doc_storage = f32, f16, or sq8 for token-node indexes.")));
	if (state->multivectorGraphMode !=
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES)
		return;
	if (state->multivectorDocStorage ==
		PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CENTROID_ONLY &&
		state->multivectorCentroids != PGTURBOHYBRID_MULTIVECTOR_CENTROIDS_KMEANS &&
		!quantizedPostingOnly)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("multivector_doc_storage = centroid_only requires multivector_centroids = kmeans"),
				 errhint("Use multivector_doc_storage = proxy_only for proxy-only graph admission, REINDEX with multivector_centroids = kmeans for centroid_lite, or enable the experimental external compact quantized inverted posting path.")));
	if (state->multivectorDocStorage ==
		PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_PROXY_ONLY &&
		state->multivectorCentroids == PGTURBOHYBRID_MULTIVECTOR_CENTROIDS_KMEANS)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("multivector_doc_storage = proxy_only does not store centroid sidecar tuples"),
				 errhint("Use multivector_doc_storage = f32, f16, or sq8, or REINDEX with multivector_centroids = off for proxy-only graph admission.")));
	if (state->multivectorDocStorage ==
		PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_PROXY_ONLY &&
		state->multivectorProxyEncoder ==
		PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_CENTROID_MEAN)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("centroid_mean multivector proxy encoder requires a centroid sidecar and is not compatible with multivector_doc_storage = proxy_only"),
				 errhint("Use multivector_proxy_encoder = normalized_mean, max_pool, or first_token, or REINDEX with multivector_doc_storage = f32, f16, or sq8.")));
	if (state->multivectorProxyEncoder !=
		PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_CENTROID_MEAN)
		return;
	if (state->multivectorCentroids !=
		PGTURBOHYBRID_MULTIVECTOR_CENTROIDS_KMEANS)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("centroid_mean multivector proxy encoder requires multivector_centroids = kmeans"),
				 errhint("Use multivector_proxy_encoder = normalized_mean, or rebuild with multivector_centroids = kmeans to persist centroid sidecar tuples.")));
}

static void
PgturbohybridGraphBuildCallback(Relation index, ItemPointer tid, Datum *values,
					 bool *isnull, bool tupleIsAlive, void *opaque)
{
	PgturbohybridQuantBuildState *state = (PgturbohybridQuantBuildState *) opaque;
	MemoryContext oldCtx;
	Datum		value;
	bool		formed;

	(void) index;
	(void) tupleIsAlive;

	CHECK_FOR_INTERRUPTS();

	if (isnull[0])
		return;

	oldCtx = CurrentMemoryContext;
	if (state->buildTupleCtx == NULL)
		state->buildTupleCtx = AllocSetContextCreate(state->ctx,
													 "pgturbohybrid graph build tuple context",
													 ALLOCSET_DEFAULT_SIZES);
	MemoryContextReset(state->buildTupleCtx);
	MemoryContextSwitchTo(state->buildTupleCtx);
	if (state->multivectorBuild)
	{
		PgturbohybridMultiVector *mv =
			(PgturbohybridMultiVector *) PG_DETOAST_DATUM(values[0]);

		PgturbohybridCheckMultiVector(mv);
		PgturbohybridMultiVectorCheckDim((uint32) mv->dim,
										 (uint32) pgturbohybrid_multivector_max_dim);
		PgturbohybridMultiVectorCheckTokenCount((uint32) mv->count,
												(uint32) pgturbohybrid_multivector_max_doc_vectors);
		if (state->buildFitPass)
		{
			if (state->multivectorGraphMode ==
				PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES)
				PgturbohybridGraphStreamFitMultiVectorDocument(state, mv);
			else
				PgturbohybridGraphStreamFitMultiVector(state, mv);
			pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_DONE, state->fitCount);
		}
		else
		{
			instr_time	nodeStart;

			MemoryContextSwitchTo(state->ctx);
			INSTR_TIME_SET_CURRENT(nodeStart);
			if (state->multivectorGraphMode ==
				PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES)
				PgturbohybridGraphAppendBuildMultiVectorDocument(state, tid, mv,
																 values, isnull);
			else
				PgturbohybridGraphAppendBuildMultiVector(state, tid, mv, values, isnull);
			state->buildGraphNodeAssignmentUs +=
				PgturbohybridGraphElapsedUs(nodeStart);
			pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_DONE, state->nodeCount);
		}
		MemoryContextSwitchTo(oldCtx);
		return;
	}

	formed = PgturbohybridGraphFormIndexValue(&value, values, isnull,
											   state->typeInfo, &state->support);
	if (formed)
	{
		Vector	   *vector = (Vector *) DatumGetPointer(value);

		MemoryContextSwitchTo(state->ctx);
		if (state->buildFitPass)
		{
			PgturbohybridGraphStreamFitVector(state, vector);
			pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_DONE, state->fitCount);
		}
		else
		{
			instr_time	nodeStart;

			INSTR_TIME_SET_CURRENT(nodeStart);
			PgturbohybridGraphAppendBuildNode(state, tid, value, values, isnull);
			state->buildGraphNodeAssignmentUs +=
				PgturbohybridGraphElapsedUs(nodeStart);
			pgstat_progress_update_param(PROGRESS_CREATEIDX_TUPLES_DONE, state->nodeCount);
		}
	}
	MemoryContextSwitchTo(oldCtx);
}

static char *
PgturbohybridNativeParallelBase(PgturbohybridNativeParallelShared *shared)
{
	return (char *) shared;
}

static uint64 *
PgturbohybridNativeParallelFitCounts(PgturbohybridNativeParallelShared *shared)
{
	return (uint64 *) (PgturbohybridNativeParallelBase(shared) + shared->fitCountOffset);
}

static double *
PgturbohybridNativeParallelFitMean(PgturbohybridNativeParallelShared *shared)
{
	return (double *) (PgturbohybridNativeParallelBase(shared) + shared->fitMeanOffset);
}

static double *
PgturbohybridNativeParallelFitM2(PgturbohybridNativeParallelShared *shared)
{
	return (double *) (PgturbohybridNativeParallelBase(shared) + shared->fitM2Offset);
}

static uint64 *
PgturbohybridNativeParallelEncodeUs(PgturbohybridNativeParallelShared *shared)
{
	return (uint64 *) (PgturbohybridNativeParallelBase(shared) + shared->encodeUsOffset);
}

static uint64 *
PgturbohybridNativeParallelScanUs(PgturbohybridNativeParallelShared *shared)
{
	return (uint64 *) (PgturbohybridNativeParallelBase(shared) + shared->scanUsOffset);
}

static float *
PgturbohybridNativeParallelEcShift(PgturbohybridNativeParallelShared *shared)
{
	return shared->ecShiftOffset != 0 ?
		(float *) (PgturbohybridNativeParallelBase(shared) + shared->ecShiftOffset) : NULL;
}

static float *
PgturbohybridNativeParallelEcScale(PgturbohybridNativeParallelShared *shared)
{
	return shared->ecScaleOffset != 0 ?
		(float *) (PgturbohybridNativeParallelBase(shared) + shared->ecScaleOffset) : NULL;
}

static char *
PgturbohybridNativeParallelRecords(PgturbohybridNativeParallelShared *shared)
{
	return (char *) shared + shared->recordsOffset;
}

static PgturbohybridNativeParallelEdgeNode *
PgturbohybridNativeParallelEdgeNodes(PgturbohybridNativeParallelShared *shared)
{
	return (PgturbohybridNativeParallelEdgeNode *)
		(PgturbohybridNativeParallelBase(shared) + shared->edgeNodeOffset);
}

static uint8 *
PgturbohybridNativeParallelEdgeCodes(PgturbohybridNativeParallelShared *shared)
{
	return (uint8 *) (PgturbohybridNativeParallelBase(shared) + shared->edgeCodeOffset);
}

static uint8 *
PgturbohybridNativeParallelEdgeResiduals(PgturbohybridNativeParallelShared *shared)
{
	return shared->edgeResidualOffset != 0 ?
		(uint8 *) (PgturbohybridNativeParallelBase(shared) + shared->edgeResidualOffset) : NULL;
}

static int *
PgturbohybridNativeParallelEdgeNeighborCounts(PgturbohybridNativeParallelShared *shared)
{
	return (int *) (PgturbohybridNativeParallelBase(shared) +
					   shared->edgeNeighborCountOffset);
}

static uint32 *
PgturbohybridNativeParallelEdgeNeighbors(PgturbohybridNativeParallelShared *shared)
{
	return (uint32 *) (PgturbohybridNativeParallelBase(shared) +
					   shared->edgeNeighborOffset);
}

static double *
PgturbohybridNativeParallelEdgeNeighborDistances(PgturbohybridNativeParallelShared *shared)
{
	return (double *) (PgturbohybridNativeParallelBase(shared) +
					   shared->edgeNeighborDistanceOffset);
}

static PgturbohybridGraphSegmentMetaData *
PgturbohybridNativeParallelEdgeSegments(PgturbohybridNativeParallelShared *shared)
{
	return (PgturbohybridGraphSegmentMetaData *)
		(PgturbohybridNativeParallelBase(shared) + shared->edgeSegmentOffset);
}

static uint64 *
PgturbohybridNativeParallelEdgeWorkerUs(PgturbohybridNativeParallelShared *shared)
{
	return (uint64 *) (PgturbohybridNativeParallelBase(shared) +
					   shared->edgeWorkerUsOffset);
}

static uint32 *
PgturbohybridNativeParallelEdgeOrder(PgturbohybridNativeParallelShared *shared)
{
	return (uint32 *) (PgturbohybridNativeParallelBase(shared) +
					   shared->edgeOrderOffset);
}

static bool *
PgturbohybridNativeParallelEdgeInserted(PgturbohybridNativeParallelShared *shared)
{
	return (bool *) (PgturbohybridNativeParallelBase(shared) +
					 shared->edgeInsertedOffset);
}

static inline Size
PgturbohybridNativeParallelEdgeNodeLevelSlot(PgturbohybridNativeParallelShared *shared,
											 uint32 nodeId, int level)
{
	return add_size(mul_size((Size) nodeId, (Size) shared->edgeLevelCapacity),
					(Size) level);
}

static inline Size
PgturbohybridNativeParallelEdgeNeighborSlot(PgturbohybridNativeParallelShared *shared,
											uint32 nodeId, int level)
{
	return mul_size(PgturbohybridNativeParallelEdgeNodeLevelSlot(shared, nodeId,
																level),
					(Size) shared->edgeMaxNeighbors);
}

static PgturbohybridNativeParallelRecord *
PgturbohybridNativeParallelRecordAt(PgturbohybridNativeParallelShared *shared,
									uint32 row)
{
	Assert(row < shared->recordCapacity);
	Assert(shared->recordBytes == MAXALIGN(shared->recordBytes));

	return (PgturbohybridNativeParallelRecord *)
		(PgturbohybridNativeParallelRecords(shared) +
		 mul_size((Size) row, shared->recordBytes));
}

static uint8 *
PgturbohybridNativeParallelRecordCode(PgturbohybridNativeParallelShared *shared,
									  PgturbohybridNativeParallelRecord *record)
{
	return (uint8 *) record + MAXALIGN(sizeof(PgturbohybridNativeParallelRecord));
}

static uint8 *
PgturbohybridNativeParallelRecordResidual(PgturbohybridNativeParallelShared *shared,
										  PgturbohybridNativeParallelRecord *record)
{
	return PgturbohybridNativeParallelRecordCode(shared, record) +
		PgturbohybridGraphCodeBytesForBits(shared->dimensions, shared->tqBits);
}

static int
PgturbohybridNativeParallelRecordCompare(const void *a, const void *b)
{
	const PgturbohybridNativeParallelRecord *left =
		(const PgturbohybridNativeParallelRecord *) a;
	const PgturbohybridNativeParallelRecord *right =
		(const PgturbohybridNativeParallelRecord *) b;
	BlockNumber leftBlock = ItemPointerGetBlockNumber(&left->heaptid);
	BlockNumber rightBlock = ItemPointerGetBlockNumber(&right->heaptid);
	OffsetNumber leftOff = ItemPointerGetOffsetNumber(&left->heaptid);
	OffsetNumber rightOff = ItemPointerGetOffsetNumber(&right->heaptid);

	if (leftBlock < rightBlock)
		return -1;
	if (leftBlock > rightBlock)
		return 1;
	if (leftOff < rightOff)
		return -1;
	if (leftOff > rightOff)
		return 1;
	return 0;
}

static int
PgturbohybridNativeParallelClaimParticipant(PgturbohybridNativeParallelShared *shared)
{
	int			slot;

	SpinLockAcquire(&shared->mutex);
	slot = shared->nextParticipant++;
	SpinLockRelease(&shared->mutex);

	if (slot >= shared->participantCapacity)
		elog(ERROR, "pgturbohybrid native parallel build participant overflow");

	return slot;
}

static void
PgturbohybridNativeInitWorkerState(PgturbohybridQuantBuildState *state,
								   Relation heap, Relation index,
								   IndexInfo *indexInfo,
								   PgturbohybridNativeParallelShared *shared)
{
	memset(state, 0, sizeof(*state));
	state->heap = heap;
	state->index = index;
	state->indexInfo = indexInfo;
	state->forkNum = MAIN_FORKNUM;
	state->building = true;
	state->typeInfo = PgturbohybridGraphGetTypeInfo(index);
	state->m = shared->m;
	state->efConstruction = shared->efConstruction;
	state->tqBits = shared->tqBits;
	state->tqWeighted = shared->tqWeighted;
	state->tqRenorm = shared->tqRenorm;
	state->buildCodeOnly = shared->buildCodeOnly;
	state->buildFastEdges = shared->buildFastEdges;
	state->scoreMode = shared->scoreMode;
	state->payloadCount = shared->payloadCount;
	state->payloadBytes = shared->payloadBytes;
	state->residualRerankBytes = shared->residualRerankBytes;
	state->dimensions = shared->dimensions;
	state->ctx = AllocSetContextCreate(CurrentMemoryContext,
									   "pgturbohybrid native parallel worker",
									   ALLOCSET_DEFAULT_SIZES);
	PgturbohybridGraphInitSupport(&state->support, index);
	if (shared->phase == PGTURBOHYBRID_NATIVE_PARALLEL_ENCODE)
	{
		state->ecShift = PgturbohybridNativeParallelEcShift(shared);
		state->ecScale = PgturbohybridNativeParallelEcScale(shared);
	}
}

static void
PgturbohybridNativeParallelBuildCallback(Relation index, ItemPointer tid,
										 Datum *values, bool *isnull,
										 bool tupleIsAlive, void *opaque)
{
	PgturbohybridQuantBuildState *state = (PgturbohybridQuantBuildState *) opaque;
	PgturbohybridNativeParallelShared *shared =
		(PgturbohybridNativeParallelShared *) state->parallelShared;
	MemoryContext oldCtx;
	Datum		value;
	bool		formed;

	(void) index;
	(void) tupleIsAlive;

	CHECK_FOR_INTERRUPTS();
	if (isnull[0])
		return;

	oldCtx = CurrentMemoryContext;
	if (state->buildTupleCtx == NULL)
		state->buildTupleCtx = AllocSetContextCreate(state->ctx,
													 "pgturbohybrid native parallel tuple context",
													 ALLOCSET_DEFAULT_SIZES);
	MemoryContextReset(state->buildTupleCtx);
	MemoryContextSwitchTo(state->buildTupleCtx);
	formed = PgturbohybridGraphFormIndexValue(&value, values, isnull,
											   state->typeInfo, &state->support);
	if (formed)
	{
		Vector	   *vector = (Vector *) DatumGetPointer(value);

		if (vector->dim != state->dimensions)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("different vector dimensions are not supported in the same pgturbohybrid graph")));

		if (shared->phase == PGTURBOHYBRID_NATIVE_PARALLEL_FIT)
		{
			MemoryContextSwitchTo(state->ctx);
			if (state->scoreMode == PGTURBOHYBRID_SCORE_COSINE ||
				state->scoreMode == PGTURBOHYBRID_SCORE_IP)
				PgturbohybridGraphStreamFitVector(state, vector);
			else
				state->fitCount++;
		}
		else
		{
			uint32		row;
			PgturbohybridNativeParallelRecord *record;
			uint8	   *code;
			PgturbohybridGraphBuildNode node;
			instr_time encodeStart;
			instr_time encodeEnd;

			SpinLockAcquire(&shared->mutex);
			if (shared->recordCount >= shared->recordCapacity)
			{
				shared->overflowed = true;
				SpinLockRelease(&shared->mutex);
				MemoryContextSwitchTo(oldCtx);
				return;
			}
			row = shared->recordCount++;
			SpinLockRelease(&shared->mutex);

			record = PgturbohybridNativeParallelRecordAt(shared, row);
			memset(record, 0, shared->recordBytes);
			record->heaptid = *tid;
			record->vectorHash = PgturbohybridGraphBuildVectorHash(vector);
			record->norm = PgturbohybridGraphVectorNorm(vector);
			PgturbohybridGraphCopyPayloadValues(state, record->payloads,
												&record->payloadMask,
												values, isnull);
			if (shared->residualRerankBytes > 0)
				PgturbohybridGraphBuildResidualSketch(vector->x, vector->dim,
													  PgturbohybridNativeParallelRecordResidual(shared, record),
													  shared->residualRerankBytes);

			memset(&node, 0, sizeof(node));
			code = PgturbohybridNativeParallelRecordCode(shared, record);
			node.code = code;
			INSTR_TIME_SET_CURRENT(encodeStart);
			PgturbohybridGraphEncodeBuildNode(state, &node, vector);
			INSTR_TIME_SET_CURRENT(encodeEnd);
			INSTR_TIME_SUBTRACT(encodeEnd, encodeStart);
			state->parallelEncodeUs += INSTR_TIME_GET_MICROSEC(encodeEnd);
			record->scale = node.scale;
			record->correction = node.correction;
			record->ecCorrection = node.ecCorrection;
		}
	}
	MemoryContextSwitchTo(oldCtx);
}

static void
PgturbohybridNativeParallelFinishParticipant(PgturbohybridNativeParallelShared *shared,
											 PgturbohybridQuantBuildState *state,
											 int slot, double reltuples,
											 uint64 scanUs)
{
	uint64	   *fitCounts = PgturbohybridNativeParallelFitCounts(shared);
	uint64	   *encodeUs = PgturbohybridNativeParallelEncodeUs(shared);
	uint64	   *scanUsBySlot = PgturbohybridNativeParallelScanUs(shared);

	if (shared->phase == PGTURBOHYBRID_NATIVE_PARALLEL_FIT)
	{
		fitCounts[slot] = state->fitCount;
		if ((state->scoreMode == PGTURBOHYBRID_SCORE_COSINE ||
			 state->scoreMode == PGTURBOHYBRID_SCORE_IP) &&
			state->fitMean != NULL && state->fitM2 != NULL)
		{
			double	   *means = PgturbohybridNativeParallelFitMean(shared);
			double	   *m2s = PgturbohybridNativeParallelFitM2(shared);
			Size		offset = mul_size((Size) slot,
										  (Size) shared->dimensions);
			Size		fitBytes = mul_size(sizeof(double),
											(Size) shared->dimensions);

			memcpy(means + offset, state->fitMean, fitBytes);
			memcpy(m2s + offset, state->fitM2, fitBytes);
		}
	}
	encodeUs[slot] = state->parallelEncodeUs;
	scanUsBySlot[slot] = scanUs;

	SpinLockAcquire(&shared->mutex);
	shared->nparticipantsdone++;
	shared->reltuples += reltuples;
	SpinLockRelease(&shared->mutex);
	ConditionVariableSignal(&shared->workersdonecv);
}

static void
PgturbohybridNativeParallelScan(Relation heapRel, Relation indexRel,
								PgturbohybridNativeParallelShared *shared,
								bool progress)
{
	PgturbohybridQuantBuildState state;
	TableScanDesc scan;
	double		reltuples;
	IndexInfo  *indexInfo;
	int			slot;
	instr_time	scanStart;
	uint64		scanUs;

	indexInfo = BuildIndexInfo(indexRel);
	indexInfo->ii_Concurrent = shared->isconcurrent;
	PgturbohybridNativeInitWorkerState(&state, heapRel, indexRel, indexInfo, shared);
	state.parallelShared = shared;
	slot = PgturbohybridNativeParallelClaimParticipant(shared);

	scan = table_beginscan_parallel(heapRel,
									ParallelTableScanFromNativeShared(shared)
#if PG_VERSION_NUM >= 190000
									,SO_NONE
#endif
		);
	INSTR_TIME_SET_CURRENT(scanStart);
	reltuples = table_index_build_scan(heapRel, indexRel, indexInfo,
									   true, progress,
									   PgturbohybridNativeParallelBuildCallback,
									   (void *) &state, scan);
	scanUs = (uint64) PgturbohybridGraphElapsedUs(scanStart);

	PgturbohybridNativeParallelFinishParticipant(shared, &state, slot, reltuples,
												 scanUs);
	MemoryContextDelete(state.ctx);
}

void
PgturbohybridNativeParallelBuildMain(dsm_segment *seg, shm_toc *toc)
{
	char	   *sharedquery;
	PgturbohybridNativeParallelShared *shared;
	Relation	heapRel;
	Relation	indexRel;
	LOCKMODE	heapLockmode;
	LOCKMODE	indexLockmode;

	(void) seg;

	sharedquery = shm_toc_lookup(toc, PARALLEL_KEY_PGTURBOHYBRID_NATIVE_QUERY, true);
	debug_query_string = sharedquery;
	pgstat_report_activity(STATE_RUNNING, debug_query_string);

	shared = shm_toc_lookup(toc, PARALLEL_KEY_PGTURBOHYBRID_NATIVE_SHARED, false);
	if (!shared->isconcurrent)
	{
		heapLockmode = ShareLock;
		indexLockmode = AccessExclusiveLock;
	}
	else
	{
		heapLockmode = ShareUpdateExclusiveLock;
		indexLockmode = RowExclusiveLock;
	}

	indexRel = index_open(shared->indexrelid, indexLockmode);
	if (shared->phase == PGTURBOHYBRID_NATIVE_PARALLEL_EDGES)
	{
		PgturbohybridNativeParallelEdgeWorker(indexRel, shared, false);
		index_close(indexRel, indexLockmode);
		return;
	}

	heapRel = table_open(shared->heaprelid, heapLockmode);
	PgturbohybridNativeParallelScan(heapRel, indexRel, shared, false);
	index_close(indexRel, indexLockmode);
	table_close(heapRel, heapLockmode);
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

static void
PgturbohybridGraphEnsureStreamingFit(PgturbohybridQuantBuildState *state, Vector *vector)
{
	if (state->dimensions == 0)
		state->dimensions = vector->dim;
	else if (state->dimensions != vector->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("different vector dimensions are not supported in the same pgturbohybrid graph")));

	if (state->fitBuffer != NULL)
		return;

	state->fitBuffer = MemoryContextAlloc(state->ctx, sizeof(double) * state->dimensions);
	if (state->tqQuantileFit)
	{
		double		pOuter;
		double		qLoTarget;
		double		qHiTarget;

		state->fitCOuter = PgturbohybridGraphCodebookOuter(state->tqBits);
		state->fitMinQuantileWidth = 1e-3;
		pOuter = PgturbohybridGraphPhi(state->fitCOuter);
		qLoTarget = 1.0 - pOuter;
		qHiTarget = pOuter;
		state->fitQLo = MemoryContextAllocZero(state->ctx,
											   sizeof(TqPSquareState) * state->dimensions);
		state->fitQHi = MemoryContextAllocZero(state->ctx,
											   sizeof(TqPSquareState) * state->dimensions);
		for (int dim = 0; dim < state->dimensions; dim++)
		{
			TqPSquareInit(&state->fitQLo[dim], qLoTarget);
			TqPSquareInit(&state->fitQHi[dim], qHiTarget);
		}
	}
	else
	{
		state->fitMean = MemoryContextAllocZero(state->ctx,
											   sizeof(double) * state->dimensions);
		state->fitM2 = MemoryContextAllocZero(state->ctx,
											 sizeof(double) * state->dimensions);
	}
}

static void
PgturbohybridGraphStreamFitVector(PgturbohybridQuantBuildState *state, Vector *vector)
{
	double		priorN;
	double		newN;

	if (vector == NULL ||
		(state->scoreMode != PGTURBOHYBRID_SCORE_COSINE &&
		 state->scoreMode != PGTURBOHYBRID_SCORE_IP))
		return;

	PgturbohybridGraphEnsureStreamingFit(state, vector);
	TqPreprocessVector(vector, state->fitBuffer);

	if (state->tqQuantileFit)
	{
		for (int dim = 0; dim < state->dimensions; dim++)
		{
			TqPSquarePush(&state->fitQLo[dim], state->fitBuffer[dim]);
			TqPSquarePush(&state->fitQHi[dim], state->fitBuffer[dim]);
		}
		state->fitCount++;
		return;
	}

	priorN = (double) state->fitCount;
	newN = priorN + 1.0;
	for (int dim = 0; dim < state->dimensions; dim++)
	{
		double		value = state->fitBuffer[dim];
		double		delta = value - state->fitMean[dim];

		state->fitMean[dim] += delta / newN;
		state->fitM2[dim] += delta * (value - state->fitMean[dim]);
	}
	state->fitCount++;
}

static void
PgturbohybridGraphFinishCorrectionFit(PgturbohybridQuantBuildState *state)
{
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
PgturbohybridGraphFinishStreamingFit(PgturbohybridQuantBuildState *state)
{
	if (state->fitCount == 0 || state->dimensions <= 0 ||
		(state->scoreMode != PGTURBOHYBRID_SCORE_COSINE &&
		 state->scoreMode != PGTURBOHYBRID_SCORE_IP))
		return;

	state->ecShift = MemoryContextAlloc(state->ctx, sizeof(float) * state->dimensions);
	state->ecScale = MemoryContextAlloc(state->ctx, sizeof(float) * state->dimensions);
	if (state->tqQuantileFit)
	{
		for (int dim = 0; dim < state->dimensions; dim++)
		{
			double		qLo = TqPSquareEstimate(&state->fitQLo[dim]);
			double		qHi = TqPSquareEstimate(&state->fitQHi[dim]);
			double		denom = qHi - qLo;

			state->ecShift[dim] = (float) (-0.5 * (qLo + qHi));
			state->ecScale[dim] = denom > state->fitMinQuantileWidth ?
				(float) ((2.0 * state->fitCOuter) / denom) : 1.0f;
		}
	}
	else
	{
		for (int dim = 0; dim < state->dimensions; dim++)
		{
			double		variance = state->fitCount > 1 ?
				state->fitM2[dim] / ((double) state->fitCount - 1.0) : 0.0;
			double		stddev = variance > 0 ? sqrt(variance) : 0.0;

			state->ecShift[dim] = (float) -state->fitMean[dim];
			state->ecScale[dim] = stddev > FLT_EPSILON ?
				(float) (1.0 / stddev) : 1.0f;
		}
	}

	PgturbohybridGraphFinishCorrectionFit(state);
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

	PgturbohybridGraphFinishCorrectionFit(state);
}

static void
PgturbohybridGraphEncodeBuildNode(PgturbohybridQuantBuildState *state,
							PgturbohybridGraphBuildNode *node,
							Vector *vector)
{
	if (state->tqWeighted)
	{
		float		xm = 0.0f;

		if (state->tqRenorm)
			node->scale = PgturbohybridGraphEncodeVectorWithXmRenorm(state, vector,
														  node->code, &xm);
		else
			node->scale = PgturbohybridGraphEncodeVectorWithXm(state, vector,
												 node->code, &xm);
		node->ecCorrection = xm;
	}
	else
	{
		node->scale = PgturbohybridGraphEncodeVector(state, vector, node->code);
		node->ecCorrection = 0.0f;
	}

	node->correction = PgturbohybridGraphCodeNorm(node->code, state->dimensions,
												 state->tqBits);
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
			if (state->residualRerankBytes > 0 && node->residualSketch != NULL &&
				prev->residualSketch != NULL && node->residualSketch != prev->residualSketch)
				memcpy(node->residualSketch, prev->residualSketch,
					   state->residualRerankBytes);
			node->scale = prev->scale;
			node->ecCorrection = prev->ecCorrection;
			node->correction = prev->correction;
			continue;
		}

		PgturbohybridGraphEncodeBuildNode(state, node, node->vector);
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
PgturbohybridGraphSelectNeighborsSimple(PgturbohybridQuantBuildState *state, uint32 src,
							 PgturbohybridGraphFrontierItem *candidates,
							 int candidateCount, int level, uint32 *selected)
{
	int			selectedCount = 0;
	int			maxNeighbors = PgturbohybridGraphLevelM(state->m, level);

	qsort(candidates, candidateCount, sizeof(PgturbohybridGraphFrontierItem),
		  PgturbohybridGraphFrontierCompare);

	for (int i = 0; i < candidateCount && selectedCount < maxNeighbors; i++)
	{
		uint32		candidate = candidates[i].nodeId;
		bool		seen = false;

		CHECK_FOR_INTERRUPTS();
		if (candidate == src)
			continue;

		for (int j = 0; j < selectedCount; j++)
		{
			if (selected[j] == candidate)
			{
				seen = true;
				break;
			}
		}

		if (!seen)
			selected[selectedCount++] = candidate;
	}

	return selectedCount;
}

static int
PgturbohybridGraphSelectNeighbors(PgturbohybridQuantBuildState *state, uint32 src,
					   PgturbohybridGraphFrontierItem *candidates, int candidateCount,
					   int level, uint32 *selected)
{
	int			selectedCount = 0;
	int			maxNeighbors = PgturbohybridGraphLevelM(state->m, level);
	instr_time	start;

	INSTR_TIME_SET_CURRENT(start);

	if (state->buildFastEdges)
	{
		selectedCount = PgturbohybridGraphSelectNeighborsSimple(state, src,
																candidates,
																candidateCount,
																level,
																selected);
		state->buildEdgeSelectNeighborUs += PgturbohybridGraphElapsedUs(start);
		return selectedCount;
	}

	qsort(candidates, candidateCount, sizeof(PgturbohybridGraphFrontierItem), PgturbohybridGraphFrontierCompare);

	for (int i = 0; i < candidateCount && selectedCount < maxNeighbors; i++)
	{
		uint32		candidate = candidates[i].nodeId;
		bool		good = true;

		CHECK_FOR_INTERRUPTS();
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

		CHECK_FOR_INTERRUPTS();
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

	state->buildEdgeSelectNeighborUs += PgturbohybridGraphElapsedUs(start);
	return selectedCount;
}

static bool
PgturbohybridGraphUseExactBuildDistances(PgturbohybridQuantBuildState *state)
{
	int			dimensions;
	bool		useExact;

	if (pgturbohybrid_dense_build_exact_distances_user_set)
	{
		useExact = pgturbohybrid_dense_build_exact_distances;
		state->buildDistanceMode = useExact ?
			PGTURBOHYBRID_DENSE_BUILD_DISTANCE_EXACT :
			PGTURBOHYBRID_DENSE_BUILD_DISTANCE_CODE;
		return useExact;
	}

	switch ((PgturbohybridDenseBuildDistance) pgturbohybrid_dense_build_distance)
	{
		case PGTURBOHYBRID_DENSE_BUILD_DISTANCE_CODE:
			state->buildDistanceMode = PGTURBOHYBRID_DENSE_BUILD_DISTANCE_CODE;
			return false;
		case PGTURBOHYBRID_DENSE_BUILD_DISTANCE_EXACT:
			state->buildDistanceMode = PGTURBOHYBRID_DENSE_BUILD_DISTANCE_EXACT;
			return true;
		case PGTURBOHYBRID_DENSE_BUILD_DISTANCE_AUTO:
		default:
			dimensions = TupleDescAttr(state->index->rd_att, 0)->atttypmod;
			useExact = dimensions > 0 &&
				dimensions <= PGTURBOHYBRID_GRAPH_AUTO_EXACT_BUILD_MAX_DIMENSIONS &&
				(pgturbohybrid_profile == PGTURBOHYBRID_PROFILE_BALANCED ||
				 pgturbohybrid_profile == PGTURBOHYBRID_PROFILE_QUALITY ||
				 pgturbohybrid_profile == PGTURBOHYBRID_PROFILE_MATCHED_RECALL);
			state->buildDistanceMode = useExact ?
				PGTURBOHYBRID_DENSE_BUILD_DISTANCE_EXACT :
				PGTURBOHYBRID_DENSE_BUILD_DISTANCE_CODE;
			return useExact;
	}
}

static bool
PgturbohybridGraphUseFastEdgesForMultiVectorDocumentProxy(PgturbohybridQuantBuildState *state)
{
	return state != NULL &&
		state->multivectorBuild &&
		state->multivectorGraphMode ==
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES &&
		state->multivectorDocBuildScorer ==
		PGTURBOHYBRID_MULTIVECTOR_DOC_BUILD_SCORER_PROXY;
}

static void
PgturbohybridGraphSetParallelEdgeBuildDisabledReason(PgturbohybridQuantBuildState *state,
													 int reason)
{
	if (state == NULL)
		return;
	if (state->parallelEdgeBuildDisabledReason ==
		PGTURBOHYBRID_PARALLEL_EDGE_BUILD_DISABLED_REASON_NONE)
		state->parallelEdgeBuildDisabledReason = reason;
}

static bool
PgturbohybridGraphCanUseParallelBuildWorkers(PgturbohybridQuantBuildState *state)
{
	if (state == NULL)
		return false;

	if (!state->multivectorBuild)
		return true;

	if (state->multivectorGraphMode ==
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_TOKEN_NODES)
	{
		PgturbohybridGraphSetParallelEdgeBuildDisabledReason(state,
															 PGTURBOHYBRID_PARALLEL_EDGE_BUILD_DISABLED_REASON_TOKEN_NODES_UNSAFE);
		return false;
	}

	if (state->multivectorGraphMode !=
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES)
	{
		PgturbohybridGraphSetParallelEdgeBuildDisabledReason(state,
															 PGTURBOHYBRID_PARALLEL_EDGE_BUILD_DISABLED_REASON_REQUIRES_FULL_MULTIVECTOR_SIDECAR);
		return false;
	}

	if (state->multivectorDocBuildScorer ==
		PGTURBOHYBRID_MULTIVECTOR_DOC_BUILD_SCORER_EXACT_SYMMETRIC)
	{
		PgturbohybridGraphSetParallelEdgeBuildDisabledReason(state,
															 PGTURBOHYBRID_PARALLEL_EDGE_BUILD_DISABLED_REASON_EXACT_DOCUMENT_MAXSIM_BUILD);
		return false;
	}

	if (state->multivectorDocBuildScorer !=
		PGTURBOHYBRID_MULTIVECTOR_DOC_BUILD_SCORER_PROXY)
	{
		PgturbohybridGraphSetParallelEdgeBuildDisabledReason(state,
															 PGTURBOHYBRID_PARALLEL_EDGE_BUILD_DISABLED_REASON_REQUIRES_FULL_MULTIVECTOR_SIDECAR);
		return false;
	}

	if (!state->buildCodeOnly)
	{
		PgturbohybridGraphSetParallelEdgeBuildDisabledReason(state,
															 PGTURBOHYBRID_PARALLEL_EDGE_BUILD_DISABLED_REASON_NOT_CODE_ONLY);
		return false;
	}

	return true;
}

static void
PgturbohybridGraphCheckExactSymmetricBuildAllowed(PgturbohybridQuantBuildState *state)
{
	uint32		docCount;
	int			maxDocs;
	int			maxTokens;

	if (state == NULL ||
		!state->multivectorBuild ||
		state->multivectorGraphMode !=
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES ||
		state->multivectorDocBuildScorer !=
		PGTURBOHYBRID_MULTIVECTOR_DOC_BUILD_SCORER_EXACT_SYMMETRIC ||
		pgturbohybrid_multivector_allow_exact_symmetric_build)
		return;

	docCount = state->multivectorDocCount;
	maxDocs = pgturbohybrid_multivector_exact_symmetric_build_max_docs;
	maxTokens = pgturbohybrid_multivector_exact_symmetric_build_max_tokens;

	/*
	 * Per-document token cap.  Exact symmetric MaxSim build cost is
	 * O(tokens_a * tokens_b * dim) per document pair, so a single token-heavy
	 * document is expensive regardless of the document count.  Mirrors Qdrant's
	 * StrictModeMultivectorConfig.max_vectors.  0 = unlimited (the default), so
	 * historical behavior is unchanged unless an operator opts in.
	 */
	if (maxTokens > 0 && state->multivectorDocVectors != NULL)
	{
		for (uint32 i = 0; i < docCount; i++)
		{
			PgturbohybridMultiVector *doc = state->multivectorDocVectors[i];

			if (doc != NULL && doc->count > maxTokens)
				ereport(ERROR,
						(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
						 errmsg("exact symmetric multivector document graph build is not allowed at this token count"),
						 errdetail("Document %u has %d tokens, above turbohybrid.multivector_exact_symmetric_build_max_tokens = %d.",
								   i, doc->count, maxTokens),
						 errhint("Use multivector_doc_build_scorer = proxy for production builds, or raise/disable turbohybrid.multivector_exact_symmetric_build_max_tokens for diagnostic experiments.")));
		}
	}

	if (maxDocs < 0 || docCount <= (uint32) maxDocs)
		return;

	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("exact symmetric multivector document graph build is not allowed at this scale"),
			 errdetail("Observed %u multivector documents, above turbohybrid.multivector_exact_symmetric_build_max_docs = %d.",
					   docCount, maxDocs),
			 errhint("Use multivector_doc_build_scorer = proxy for production builds, or set turbohybrid.multivector_allow_exact_symmetric_build = on for diagnostic experiments.")));
}

static bool
PgturbohybridGraphUseFastBuildEdges(PgturbohybridQuantBuildState *state)
{
	bool		useFast = false;
	int			reason = PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_UNKNOWN;

	switch ((PgturbohybridDenseBuildNeighborSelect) pgturbohybrid_dense_build_neighbor_select)
	{
		case PGTURBOHYBRID_DENSE_BUILD_NEIGHBOR_SELECT_FAST:
			reason = PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_EXPLICIT_FAST;
			useFast = true;
			break;
		case PGTURBOHYBRID_DENSE_BUILD_NEIGHBOR_SELECT_HEURISTIC:
			reason = PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_EXPLICIT_HEURISTIC;
			useFast = false;
			break;
		case PGTURBOHYBRID_DENSE_BUILD_NEIGHBOR_SELECT_AUTO:
		default:
			if (PgturbohybridGraphUseFastEdgesForMultiVectorDocumentProxy(state))
			{
				reason = PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_AUTO_MULTIVECTOR_DOCUMENT_PROXY;
				useFast = true;
			}
			else if (pgturbohybrid_profile == PGTURBOHYBRID_PROFILE_BALANCED)
			{
				reason = PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_AUTO_BALANCED;
				useFast = false;
			}
			else if (pgturbohybrid_profile == PGTURBOHYBRID_PROFILE_QUALITY ||
					 pgturbohybrid_profile == PGTURBOHYBRID_PROFILE_DEBUG)
			{
				reason = PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_AUTO_QUALITY;
				useFast = false;
			}
			else if (pgturbohybrid_profile == PGTURBOHYBRID_PROFILE_MATCHED_RECALL)
			{
				reason = PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_AUTO_MATCHED_RECALL;
				useFast = false;
			}
			else if (state->dimensions > 0 &&
					 state->dimensions <= PGTURBOHYBRID_GRAPH_AUTO_HEURISTIC_MAX_DIMENSIONS)
			{
				reason = PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_AUTO_LOWDIM;
				useFast = false;
			}
			else
			{
				reason = PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_AUTO_LATENCY_HIGHDIM;
				useFast = true;
			}
			break;
	}

	state->buildNeighborSelectReason = reason;

	if (!state->buildCodeOnly)
		return false;

	return useFast;
}

static double
PgturbohybridGraphCandidateDistance(PgturbohybridGraphFrontierItem *candidates,
						 int candidateCount, uint32 nodeId)
{
	for (int i = 0; i < candidateCount; i++)
	{
		if (candidates[i].nodeId == nodeId)
			return candidates[i].distance;
	}

	return DBL_MAX;
}

static void
PgturbohybridGraphPruneNeighbors(PgturbohybridQuantBuildState *state, uint32 src, int level)
{
	PgturbohybridGraphBuildNode *node = &state->nodes[src];
	int			count;
	PgturbohybridGraphFrontierItem *candidates;
	uint32	   *selected;
	int			selectedCount;
	instr_time	start;

	if (!PgturbohybridGraphBuildNodeHasLevel(state, src, level))
		return;

	count = node->neighborCounts[level];
	if (count <= PgturbohybridGraphLevelM(state->m, level))
		return;

	INSTR_TIME_SET_CURRENT(start);
	candidates = palloc(sizeof(PgturbohybridGraphFrontierItem) * count);
	selected = palloc(sizeof(uint32) * PgturbohybridGraphLevelM(state->m, level));

	for (int i = 0; i < count; i++)
	{
		CHECK_FOR_INTERRUPTS();
		candidates[i].nodeId = node->neighbors[level][i];
		candidates[i].distance = node->neighborDistances != NULL ?
			node->neighborDistances[level][i] :
			PgturbohybridGraphBuildDistance(state, src, candidates[i].nodeId);
	}

	selectedCount = PgturbohybridGraphSelectNeighbors(state, src, candidates, count, level, selected);
	for (int i = 0; i < selectedCount; i++)
	{
		node->neighbors[level][i] = selected[i];
		if (node->neighborDistances != NULL)
			node->neighborDistances[level][i] =
				PgturbohybridGraphCandidateDistance(candidates, count, selected[i]);
	}
	node->neighborCounts[level] = selectedCount;

	pfree(candidates);
	pfree(selected);
	state->buildEdgePruneNeighborUs += PgturbohybridGraphElapsedUs(start);
}

static void
PgturbohybridGraphAddNeighbor(PgturbohybridQuantBuildState *state, uint32 src, uint32 dst,
					int level, double distance)
{
	PgturbohybridGraphBuildNode *node = &state->nodes[src];
	int			maxNeighbors = PgturbohybridGraphLevelM(state->m, level);

	if (src == dst || !PgturbohybridGraphBuildNodeHasLevel(state, src, level) ||
		!PgturbohybridGraphBuildNodeHasLevel(state, dst, level) ||
		PgturbohybridGraphHasNeighbor(state, src, dst, level))
		return;

	if (node->neighborCounts[level] < maxNeighbors)
	{
		int			slot = node->neighborCounts[level]++;

		node->neighbors[level][slot] = dst;
		if (node->neighborDistances != NULL)
			node->neighborDistances[level][slot] = distance;
		return;
	}

	if (state->buildFastEdges && node->neighborDistances != NULL)
	{
		int			worst = 0;

		for (int i = 1; i < node->neighborCounts[level]; i++)
		{
			if (node->neighborDistances[level][i] > node->neighborDistances[level][worst])
				worst = i;
		}

		if (distance < node->neighborDistances[level][worst])
		{
			node->neighbors[level][worst] = dst;
			node->neighborDistances[level][worst] = distance;
		}
		return;
	}

	node->neighbors[level][node->neighborCounts[level]++] = dst;
	if (node->neighborDistances != NULL)
		node->neighborDistances[level][node->neighborCounts[level] - 1] = distance;
	PgturbohybridGraphPruneNeighbors(state, src, level);
}

static void
PgturbohybridGraphAddNeighborIfRoom(PgturbohybridQuantBuildState *state, uint32 src, uint32 dst,
						 int level, double distance)
{
	PgturbohybridGraphBuildNode *node = &state->nodes[src];
	int			maxNeighbors = PgturbohybridGraphLevelM(state->m, level);

	if (src == dst || !PgturbohybridGraphBuildNodeHasLevel(state, src, level) ||
		!PgturbohybridGraphBuildNodeHasLevel(state, dst, level) ||
		PgturbohybridGraphHasNeighbor(state, src, dst, level) ||
		node->neighborCounts[level] >= maxNeighbors)
		return;

	node->neighbors[level][node->neighborCounts[level]] = dst;
	if (node->neighborDistances != NULL)
		node->neighborDistances[level][node->neighborCounts[level]] = distance;
	node->neighborCounts[level]++;
}

static bool
PgturbohybridGraphIsBackboneNeighbor(PgturbohybridQuantBuildState *state, uint32 src,
						 uint32 neighbor)
{
	(void) state;

	return (src > 0 && neighbor == src - 1) ||
		(src + 1 < state->nodeCount && neighbor == src + 1);
}

static void
PgturbohybridGraphForceNeighbor(PgturbohybridQuantBuildState *state, uint32 src, uint32 dst,
					 int level)
{
	PgturbohybridGraphBuildNode *node = &state->nodes[src];
	int			maxNeighbors = PgturbohybridGraphLevelM(state->m, level);

	if (src == dst || !PgturbohybridGraphBuildNodeHasLevel(state, src, level) ||
		!PgturbohybridGraphBuildNodeHasLevel(state, dst, level) ||
		PgturbohybridGraphHasNeighbor(state, src, dst, level))
		return;

	if (node->neighborCounts[level] < maxNeighbors)
	{
		int			slot = node->neighborCounts[level]++;

		node->neighbors[level][slot] = dst;
		if (node->neighborDistances != NULL)
			node->neighborDistances[level][slot] = DBL_MAX;
		return;
	}

	{
		int			worst = -1;
		double		worstDistance = -DBL_MAX;

		for (int i = 0; i < node->neighborCounts[level]; i++)
		{
			uint32		neighbor = node->neighbors[level][i];
			double		distance;

			if (PgturbohybridGraphIsBackboneNeighbor(state, src, neighbor))
				continue;

			if (node->neighborDistances != NULL)
				distance = node->neighborDistances[level][i];
			else
				distance = PgturbohybridGraphBuildDistance(state, src, neighbor);

			if (worst < 0 || distance > worstDistance)
			{
				worst = i;
				worstDistance = distance;
			}
		}

		if (worst < 0)
			worst = 0;

		node->neighbors[level][worst] = dst;
		if (node->neighborDistances != NULL)
			node->neighborDistances[level][worst] = DBL_MAX;
	}
}

static void
PgturbohybridGraphEnsureLevel0Backbone(PgturbohybridQuantBuildState *state)
{
	if (state->nodeCount < 2)
		return;

	for (uint32 nodeId = 1; nodeId < state->nodeCount; nodeId++)
	{
		uint32		prevId = nodeId - 1;

		CHECK_FOR_INTERRUPTS();
		PgturbohybridGraphForceNeighbor(state, prevId, nodeId, 0);
		PgturbohybridGraphForceNeighbor(state, nodeId, prevId, 0);
	}
}

static uint32
PgturbohybridGraphAdjacentDuplicateCountRange(PgturbohybridQuantBuildState *state,
											 uint32 startNodeId, uint32 endNodeId)
{
	uint32		duplicates = 0;

	for (uint32 i = startNodeId + 1; i < endNodeId; i++)
	{
		CHECK_FOR_INTERRUPTS();
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
	if (node->neighborDistances != NULL)
		node->neighborDistances[0][0] =
			PgturbohybridGraphBuildDistance(state, nodeId, prevId);
	node->neighborCounts[0] = 1;
	PgturbohybridGraphAddNeighborIfRoom(state, prevId, nodeId, 0,
							 PgturbohybridGraphBuildDistance(state, prevId, nodeId));
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

		CHECK_FOR_INTERRUPTS();
		changed = false;
		if (!PgturbohybridGraphBuildNodeHasLevel(state, current.nodeId, level))
			break;

		for (int i = 0; i < node->neighborCounts[level]; i++)
		{
			uint32		neighbor = node->neighbors[level][i];
			double		distance;

			CHECK_FOR_INTERRUPTS();
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
	if ((uint32) frontierCount > state->buildEdgeMaxFrontierSize)
		state->buildEdgeMaxFrontierSize = frontierCount;

	while (frontierCount > 0)
	{
		PgturbohybridGraphFrontierItem item = PgturbohybridGraphFrontierHeapPop(frontier, &frontierCount, true);
		PgturbohybridGraphBuildNode *node = &state->nodes[item.nodeId];

		CHECK_FOR_INTERRUPTS();
		if (nearestCount >= ef && PgturbohybridGraphFrontierGreater(item, nearest[0]))
			break;

		if (!PgturbohybridGraphBuildNodeHasLevel(state, item.nodeId, level))
			continue;

		for (int i = 0; i < node->neighborCounts[level]; i++)
		{
			uint32		neighbor = node->neighbors[level][i];
			double		distance;

			CHECK_FOR_INTERRUPTS();
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
				if ((uint32) frontierCount > state->buildEdgeMaxFrontierSize)
					state->buildEdgeMaxFrontierSize = frontierCount;
			}
		}
	}

	pfree(frontier);
	return nearestCount;
}

static void
PgturbohybridGraphBuildEdgesRange(PgturbohybridQuantBuildState *state,
								  uint32 startNodeId, uint32 endNodeId,
								  PgturbohybridGraphSegmentMetaData *segment)
{
	int			ef = Max(state->efConstruction, PgturbohybridGraphLevelM(state->m, 0));
	uint32		entryNodeId;
	int			entryLevel;
	PgturbohybridGraphFrontierItem *nearest;
	uint32	   *selected;
	PgturbohybridGraphBuildOrderItem *order;
	bool	   *inserted;
	uint32		adjacentDuplicates;
	uint32		rangeNodeCount = endNodeId - startNodeId;
	bool		preserveScanOrder;

	if (rangeNodeCount == 0)
		return;

	nearest = palloc(sizeof(PgturbohybridGraphFrontierItem) * ef);
	selected = palloc(sizeof(uint32) * PgturbohybridGraphLevelM(state->m, 0));
	order = palloc(sizeof(PgturbohybridGraphBuildOrderItem) * rangeNodeCount);
	inserted = palloc0(sizeof(bool) * state->nodeCount);
	state->buildVisitedGeneration = palloc0(sizeof(uint32) * state->nodeCount);
	state->buildVisitGeneration = 0;
	state->buildQueryCtx = AllocSetContextCreate(state->ctx,
												 "pgturbohybrid graph build query context",
												 ALLOCSET_DEFAULT_SIZES);

	for (uint32 i = 0; i < rangeNodeCount; i++)
	{
		uint32		nodeId = startNodeId + i;

		order[i].nodeId = nodeId;
		order[i].key = PgturbohybridGraphMix64(nodeId);
	}
	adjacentDuplicates = PgturbohybridGraphAdjacentDuplicateCountRange(state,
																	   startNodeId,
																	   endNodeId);
	preserveScanOrder = adjacentDuplicates > rangeNodeCount / 2;
	elog(DEBUG1, "pgturbohybrid native graph segment duplicate-run build: start_node=%u nodes=%u adjacent_duplicates=%u preserve_scan_order=%s",
		 startNodeId, rangeNodeCount, adjacentDuplicates, preserveScanOrder ? "on" : "off");
	if (!preserveScanOrder)
		qsort(order, rangeNodeCount, sizeof(PgturbohybridGraphBuildOrderItem),
			  PgturbohybridGraphBuildOrderCompare);

	entryNodeId = order[0].nodeId;
	entryLevel = state->nodes[entryNodeId].level;
	inserted[entryNodeId] = true;

	for (uint32 orderIdx = 1; orderIdx < rangeNodeCount; orderIdx++)
	{
		uint32		i = order[orderIdx].nodeId;
		PgturbohybridGraphFrontierItem levelEntry;
		int			nodeLevel = state->nodes[i].level;
		int			linkingLevel = Min(nodeLevel, entryLevel);

		CHECK_FOR_INTERRUPTS();
		if (preserveScanOrder && i > startNodeId && inserted[i - 1])
		{
			instr_time	addStart;

			INSTR_TIME_SET_CURRENT(addStart);
			PgturbohybridGraphLinkAdjacentBuildNode(state, i, i - 1);
			state->buildEdgeAddNeighborUs += PgturbohybridGraphElapsedUs(addStart);
			if (nodeLevel > entryLevel)
			{
				instr_time	entryStart;

				INSTR_TIME_SET_CURRENT(entryStart);
				entryNodeId = i;
				entryLevel = nodeLevel;
				state->buildEdgeEntryUpdateUs += PgturbohybridGraphElapsedUs(entryStart);
			}
			inserted[i] = true;
			continue;
		}
		PgturbohybridGraphPrepareBuildQuery(state, i);

		levelEntry.nodeId = entryNodeId;
		levelEntry.distance = PgturbohybridGraphBuildDistance(state, i, entryNodeId);

		for (int level = entryLevel; level > nodeLevel; level--)
		{
			instr_time	searchStart;

			CHECK_FOR_INTERRUPTS();
			if (PgturbohybridGraphBuildNodeHasLevel(state, levelEntry.nodeId, level))
			{
				INSTR_TIME_SET_CURRENT(searchStart);
				levelEntry = PgturbohybridGraphBuildGreedySearch(state, i,
													  levelEntry.nodeId,
													  level, inserted);
				state->buildEdgeEntrySearchUs += PgturbohybridGraphElapsedUs(searchStart);
			}
		}

		for (int level = linkingLevel; level >= 0; level--)
		{
			int			nearestCount;
			int			selectedCount;
			instr_time	searchStart;

			CHECK_FOR_INTERRUPTS();
			if (!PgturbohybridGraphBuildNodeHasLevel(state, levelEntry.nodeId, level))
				continue;

			INSTR_TIME_SET_CURRENT(searchStart);
			nearestCount = PgturbohybridGraphBuildSearchLayer(state, i, levelEntry, level,
												   ef, nearest, inserted);
			state->buildEdgeNeighborSearchUs += PgturbohybridGraphElapsedUs(searchStart);
			state->buildEdgeNearestTotal += nearestCount;
			state->buildEdgeNearestSamples++;
			selectedCount = PgturbohybridGraphSelectNeighbors(state, i, nearest, nearestCount,
												  level, selected);

			for (int j = 0; j < selectedCount; j++)
			{
				state->nodes[i].neighbors[level][j] = selected[j];
				if (state->nodes[i].neighborDistances != NULL)
					state->nodes[i].neighborDistances[level][j] =
						PgturbohybridGraphCandidateDistance(nearest, nearestCount,
															selected[j]);
			}
			state->nodes[i].neighborCounts[level] = selectedCount;
			for (int j = 0; j < selectedCount; j++)
			{
				instr_time	addStart;

				INSTR_TIME_SET_CURRENT(addStart);
				PgturbohybridGraphAddNeighbor(state, selected[j], i, level,
											  PgturbohybridGraphCandidateDistance(nearest,
																				  nearestCount,
																				  selected[j]));
				state->buildEdgeAddNeighborUs += PgturbohybridGraphElapsedUs(addStart);
			}

			if (nearestCount > 0)
			{
				qsort(nearest, nearestCount, sizeof(PgturbohybridGraphFrontierItem), PgturbohybridGraphFrontierCompare);
				levelEntry = nearest[0];
			}
		}

		if (nodeLevel > entryLevel)
		{
			instr_time	entryStart;

			INSTR_TIME_SET_CURRENT(entryStart);
			entryNodeId = i;
			entryLevel = nodeLevel;
			state->buildEdgeEntryUpdateUs += PgturbohybridGraphElapsedUs(entryStart);
		}
		inserted[i] = true;
	}

	if (segment != NULL)
	{
		segment->startNodeId = startNodeId;
		segment->nodeCount = rangeNodeCount;
		segment->entryNodeId = entryNodeId;
		segment->entryLevel = entryLevel;
	}
	if (state->segmentCount <= 1 || entryLevel > state->maxLevel)
	{
		state->entryNodeId = entryNodeId;
		state->maxLevel = entryLevel;
	}
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
PgturbohybridGraphBuildEdges(PgturbohybridQuantBuildState *state)
{
	int			requestedSegments;
	uint16		segmentCount;
	uint32		baseSegmentSize;
	uint32		remainder;
	uint32		startNodeId = 0;

	if (state->nodeCount == 0)
		return;

	requestedSegments = PgturbohybridGraphGetNativeSegments(state->index);
	segmentCount = (uint16) Min((uint32) requestedSegments, state->nodeCount);
	segmentCount = Max(segmentCount, 1);
	state->segmentCount = segmentCount;
	state->entryNodeId = 0;
	state->maxLevel = -1;
	memset(state->segments, 0, sizeof(state->segments));

	baseSegmentSize = state->nodeCount / segmentCount;
	remainder = state->nodeCount % segmentCount;
	for (uint16 segmentIdx = 0; segmentIdx < segmentCount; segmentIdx++)
	{
		uint32		segmentSize = baseSegmentSize + (segmentIdx < remainder ? 1 : 0);
		uint32		endNodeId = startNodeId + segmentSize;

		CHECK_FOR_INTERRUPTS();
		PgturbohybridGraphBuildEdgesRange(state, startNodeId, endNodeId,
										  &state->segments[segmentIdx]);
		startNodeId = endNodeId;
	}
}

static uint16
PgturbohybridNativeParallelEdgeSegmentCount(PgturbohybridQuantBuildState *state,
											int workerCount)
{
	int			requestedSegments = PgturbohybridGraphGetNativeSegments(state->index);
	uint32		segmentCount;

	if (state->nodeCount == 0)
		return 0;

	if (requestedSegments > 1)
		segmentCount = (uint32) requestedSegments;
	else
		segmentCount = (uint32) Max(workerCount + 1, 1);

	segmentCount = Min(segmentCount, state->nodeCount);
	segmentCount = Min(segmentCount, (uint32) PGTURBOHYBRID_GRAPH_MAX_NATIVE_SEGMENTS);
	return (uint16) Max(segmentCount, 1);
}

static bool
PgturbohybridNativeParallelEdgeEligible(PgturbohybridQuantBuildState *state,
										bool force)
{
	if (pgturbohybrid_native_parallel_edge_build ==
		PGTURBOHYBRID_NATIVE_PARALLEL_EDGE_BUILD_OFF)
	{
		PgturbohybridGraphSetParallelEdgeBuildDisabledReason(state,
															 PGTURBOHYBRID_PARALLEL_EDGE_BUILD_DISABLED_REASON_PARALLEL_EDGE_GUC_OFF);
		return false;
	}

	if (state->nodeCount < 2 || state->dimensions <= 0 || state->m <= 0 ||
		!state->buildCodeOnly || state->buildExactDistances ||
		state->tqExactStorage)
	{
		if (force)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("turbohybrid.native_parallel_edge_build=on requires a code-only native graph build"),
					 errdetail("Exact-storage and exact-distance build modes keep edge construction serial.")));
		PgturbohybridGraphSetParallelEdgeBuildDisabledReason(state,
															 !state->buildCodeOnly ||
															 state->buildExactDistances ||
															 state->tqExactStorage ?
															 PGTURBOHYBRID_PARALLEL_EDGE_BUILD_DISABLED_REASON_NOT_CODE_ONLY :
															 PGTURBOHYBRID_PARALLEL_EDGE_BUILD_DISABLED_REASON_INVALID_GRAPH_STATE);
		return false;
	}
	if (PgturbohybridGraphGetNativeSegments(state->index) > 1)
	{
		if (force)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("turbohybrid.native_parallel_edge_build=on requires native_segments = 1"),
					 errdetail("Explicit multi-segment native graph builds keep the segment-local serial edge builder.")));
		PgturbohybridGraphSetParallelEdgeBuildDisabledReason(state,
															 PGTURBOHYBRID_PARALLEL_EDGE_BUILD_DISABLED_REASON_SEGMENTED_GRAPH);
		return false;
	}

	state->parallelEdgeBuildDisabledReason =
		PGTURBOHYBRID_PARALLEL_EDGE_BUILD_DISABLED_REASON_NONE;
	return true;
}

static void
PgturbohybridNativeParallelEdgeCopyNodes(PgturbohybridQuantBuildState *state,
										 PgturbohybridNativeParallelShared *shared)
{
	PgturbohybridNativeParallelEdgeNode *edgeNodes =
		PgturbohybridNativeParallelEdgeNodes(shared);
	uint8	   *codes = PgturbohybridNativeParallelEdgeCodes(shared);
	uint8	   *residuals = PgturbohybridNativeParallelEdgeResiduals(shared);

	for (uint32 nodeId = 0; nodeId < state->nodeCount; nodeId++)
	{
		PgturbohybridGraphBuildNode *node = &state->nodes[nodeId];
		PgturbohybridNativeParallelEdgeNode *edgeNode = &edgeNodes[nodeId];

		CHECK_FOR_INTERRUPTS();
		edgeNode->heaptid = node->heaptid;
		edgeNode->vectorHash = node->vectorHash;
		edgeNode->norm = node->norm;
		edgeNode->scale = node->scale;
		edgeNode->correction = node->correction;
		edgeNode->ecCorrection = node->ecCorrection;
		edgeNode->payloadMask = node->payloadMask;
		edgeNode->flags = node->flags;
		edgeNode->level = node->level;
		if (node->payloads != NULL && state->payloadBytes > 0)
			memcpy(edgeNode->payloads, node->payloads, state->payloadBytes);
		if (node->code != NULL && shared->edgeCodeBytes > 0)
			memcpy(codes + mul_size((Size) nodeId, shared->edgeCodeBytes),
				   node->code, shared->edgeCodeBytes);
		if (residuals != NULL && node->residualSketch != NULL &&
			state->residualRerankBytes > 0)
			memcpy(residuals + mul_size((Size) nodeId,
										(Size) state->residualRerankBytes),
				   node->residualSketch, state->residualRerankBytes);
	}
}

static void
PgturbohybridNativeParallelEdgeLoadSharedGraph(PgturbohybridQuantBuildState *state,
											   PgturbohybridNativeParallelShared *shared)
{
	PgturbohybridNativeParallelEdgeNode *edgeNodes =
		PgturbohybridNativeParallelEdgeNodes(shared);
	uint8	   *codes = PgturbohybridNativeParallelEdgeCodes(shared);
	uint8	   *residuals = PgturbohybridNativeParallelEdgeResiduals(shared);
	int		   *counts = PgturbohybridNativeParallelEdgeNeighborCounts(shared);
	uint32	   *neighbors = PgturbohybridNativeParallelEdgeNeighbors(shared);
	double	   *distances = PgturbohybridNativeParallelEdgeNeighborDistances(shared);

	state->nodeCount = shared->edgeNodeCount;
	state->nodeCapacity = shared->edgeNodeCount;
	state->nodes = MemoryContextAllocZero(state->ctx,
										  mul_size(sizeof(PgturbohybridGraphBuildNode),
												   (Size) state->nodeCount));
	state->entryNodeId = shared->edgeEntryNodeId;
	state->maxLevel = shared->edgeEntryLevel;
	state->segmentCount = 1;
	for (uint32 nodeId = 0; nodeId < state->nodeCount; nodeId++)
	{
		PgturbohybridNativeParallelEdgeNode *edgeNode = &edgeNodes[nodeId];
		PgturbohybridGraphBuildNode *node = &state->nodes[nodeId];
		int			levelCount = edgeNode->level + 1;

		node->heaptid = edgeNode->heaptid;
		node->vectorHash = edgeNode->vectorHash;
		node->norm = edgeNode->norm;
		node->scale = edgeNode->scale;
		node->correction = edgeNode->correction;
		node->ecCorrection = edgeNode->ecCorrection;
		node->payloadMask = edgeNode->payloadMask;
		node->flags = edgeNode->flags;
		node->level = edgeNode->level;
		node->code = codes + mul_size((Size) nodeId, shared->edgeCodeBytes);
		if (residuals != NULL && shared->residualRerankBytes > 0)
			node->residualSketch =
				residuals + mul_size((Size) nodeId,
									 (Size) shared->residualRerankBytes);
		if (shared->payloadBytes > 0)
			node->payloads = edgeNode->payloads;
		node->neighbors = MemoryContextAllocZero(state->ctx,
												 mul_size(sizeof(uint32 *),
														  (Size) levelCount));
		node->neighborCounts =
			counts + PgturbohybridNativeParallelEdgeNodeLevelSlot(shared,
																  nodeId, 0);
		node->neighborDistances = MemoryContextAllocZero(state->ctx,
														 mul_size(sizeof(double *),
																  (Size) levelCount));
		for (int level = 0; level <= node->level; level++)
		{
			node->neighbors[level] =
				neighbors + PgturbohybridNativeParallelEdgeNeighborSlot(shared,
																		nodeId,
																		level);
			node->neighborDistances[level] =
				distances + PgturbohybridNativeParallelEdgeNeighborSlot(shared,
																		nodeId,
																		level);
		}
		state->maxLevel = Max(state->maxLevel, node->level);
	}
}

static void
PgturbohybridNativeParallelEdgeBuildNodeLinks(PgturbohybridQuantBuildState *state,
											  uint32 nodeId, uint32 entryNodeId,
											  int entryLevel, bool *inserted,
											  PgturbohybridGraphFrontierItem *nearest,
											  uint32 *selected)
{
	int			ef = Max(state->efConstruction, PgturbohybridGraphLevelM(state->m, 0));
	PgturbohybridGraphBuildNode *node = &state->nodes[nodeId];
	PgturbohybridGraphFrontierItem levelEntry;
	int			nodeLevel = node->level;
	int			linkingLevel = Min(nodeLevel, entryLevel);

	for (int level = 0; level <= nodeLevel; level++)
		node->neighborCounts[level] = 0;

	if (entryNodeId >= state->nodeCount || entryLevel < 0)
		return;

	PgturbohybridGraphPrepareBuildQuery(state, nodeId);
	levelEntry.nodeId = entryNodeId;
	levelEntry.distance = PgturbohybridGraphBuildDistance(state, nodeId,
														  entryNodeId);

	for (int level = entryLevel; level > nodeLevel; level--)
	{
		instr_time	searchStart;

		CHECK_FOR_INTERRUPTS();
		if (PgturbohybridGraphBuildNodeHasLevel(state, levelEntry.nodeId, level))
		{
			INSTR_TIME_SET_CURRENT(searchStart);
			levelEntry = PgturbohybridGraphBuildGreedySearch(state, nodeId,
															 levelEntry.nodeId,
															 level, inserted);
			state->buildEdgeEntrySearchUs += PgturbohybridGraphElapsedUs(searchStart);
		}
	}

	for (int level = linkingLevel; level >= 0; level--)
	{
		int			nearestCount;
		int			selectedCount;
		instr_time	searchStart;

		CHECK_FOR_INTERRUPTS();
		if (!PgturbohybridGraphBuildNodeHasLevel(state, levelEntry.nodeId, level))
			continue;

		INSTR_TIME_SET_CURRENT(searchStart);
		nearestCount = PgturbohybridGraphBuildSearchLayer(state, nodeId,
														  levelEntry, level,
														  ef, nearest,
														  inserted);
		state->buildEdgeNeighborSearchUs += PgturbohybridGraphElapsedUs(searchStart);
		state->buildEdgeNearestTotal += nearestCount;
		state->buildEdgeNearestSamples++;
		selectedCount = PgturbohybridGraphSelectNeighbors(state, nodeId,
														  nearest,
														  nearestCount,
														  level, selected);

		for (int j = 0; j < selectedCount; j++)
		{
			node->neighbors[level][j] = selected[j];
			node->neighborDistances[level][j] =
				PgturbohybridGraphCandidateDistance(nearest, nearestCount,
													selected[j]);
		}
		node->neighborCounts[level] = selectedCount;

		if (nearestCount > 0)
		{
			qsort(nearest, nearestCount, sizeof(PgturbohybridGraphFrontierItem),
				  PgturbohybridGraphFrontierCompare);
			levelEntry = nearest[0];
		}
	}
}

static void
PgturbohybridNativeParallelEdgeApplyNodeBacklinks(PgturbohybridQuantBuildState *state,
												  uint32 nodeId, bool *inserted,
												  int applyShard,
												  int applyShardCount)
{
	PgturbohybridGraphBuildNode *node = &state->nodes[nodeId];

	for (int level = 0; level <= node->level; level++)
	{
		int			count = node->neighborCounts[level];

		for (int j = 0; j < count; j++)
		{
			uint32		neighbor = node->neighbors[level][j];
			double		distance = node->neighborDistances[level][j];
			instr_time	addStart;

			CHECK_FOR_INTERRUPTS();
			if (neighbor >= state->nodeCount || !inserted[neighbor])
				continue;
			if (applyShardCount > 1 &&
				(int) (neighbor % (uint32) applyShardCount) != applyShard)
				continue;
			INSTR_TIME_SET_CURRENT(addStart);
			PgturbohybridGraphAddNeighbor(state, neighbor, nodeId, level,
										  distance);
			state->buildEdgeAddNeighborUs += PgturbohybridGraphElapsedUs(addStart);
		}
	}
}

static void
PgturbohybridNativeParallelEdgePublishNode(PgturbohybridQuantBuildState *state,
										   PgturbohybridNativeParallelShared *shared,
										   uint32 nodeId, bool *inserted)
{
	PgturbohybridGraphBuildNode *node = &state->nodes[nodeId];

	inserted[nodeId] = true;
	shared->edgeInsertedCount++;
	if (node->level > shared->edgeEntryLevel)
	{
		instr_time	entryStart;

		INSTR_TIME_SET_CURRENT(entryStart);
		shared->edgeEntryNodeId = nodeId;
		shared->edgeEntryLevel = node->level;
		state->entryNodeId = nodeId;
		state->maxLevel = node->level;
		state->buildEdgeEntryUpdateUs += PgturbohybridGraphElapsedUs(entryStart);
	}
}

static void
PgturbohybridNativeParallelEdgeBuildWarmup(PgturbohybridQuantBuildState *state,
										   PgturbohybridNativeParallelShared *shared,
										   PgturbohybridGraphFrontierItem *nearest,
										   uint32 *selected)
{
	uint32	   *order = PgturbohybridNativeParallelEdgeOrder(shared);
	bool	   *inserted = PgturbohybridNativeParallelEdgeInserted(shared);
	uint32		warmup = Min(shared->edgeWarmupCount, shared->edgeNodeCount);

	if (warmup == 0)
		return;

	shared->edgeEntryNodeId = order[0];
	shared->edgeEntryLevel = state->nodes[order[0]].level;
	state->entryNodeId = shared->edgeEntryNodeId;
	state->maxLevel = shared->edgeEntryLevel;
	inserted[order[0]] = true;
	shared->edgeInsertedCount = 1;

	for (uint32 orderIdx = 1; orderIdx < warmup; orderIdx++)
	{
		uint32		nodeId = order[orderIdx];

		CHECK_FOR_INTERRUPTS();
		PgturbohybridNativeParallelEdgeBuildNodeLinks(state, nodeId,
													  shared->edgeEntryNodeId,
													  shared->edgeEntryLevel,
													  inserted, nearest,
													  selected);
		PgturbohybridNativeParallelEdgeApplyNodeBacklinks(state, nodeId,
														  inserted, 0, 1);
		PgturbohybridNativeParallelEdgePublishNode(state, shared, nodeId,
												   inserted);
	}
	shared->edgeBatchStartOrder = warmup;
}

static void
PgturbohybridNativeParallelEdgeWorker(Relation indexRel,
									  PgturbohybridNativeParallelShared *shared,
									  bool leader)
{
	PgturbohybridQuantBuildState state;
	IndexInfo  *indexInfo = BuildIndexInfo(indexRel);
	int			slot;
	instr_time	start;
	uint64		elapsedUs;
	PgturbohybridGraphFrontierItem *nearest;
	uint32	   *selected;
	bool	   *inserted;
	uint32	   *order;

	PgturbohybridNativeInitWorkerState(&state, NULL, indexRel, indexInfo, shared);
	state.parallelShared = shared;
	state.ecShift = PgturbohybridNativeParallelEcShift(shared);
	state.ecScale = PgturbohybridNativeParallelEcScale(shared);
	if (state.ecShift != NULL && state.ecScale != NULL)
		PgturbohybridGraphFinishCorrectionFit(&state);
	slot = PgturbohybridNativeParallelClaimParticipant(shared);
	PgturbohybridNativeParallelEdgeLoadSharedGraph(&state, shared);
	state.buildVisitedGeneration = palloc0(sizeof(uint32) * state.nodeCount);
	state.buildVisitGeneration = 0;
	state.buildQueryCtx = AllocSetContextCreate(state.ctx,
												"pgturbohybrid graph build query context",
												ALLOCSET_DEFAULT_SIZES);
	nearest = palloc(sizeof(PgturbohybridGraphFrontierItem) *
					 Max(state.efConstruction,
						 PgturbohybridGraphLevelM(state.m, 0)));
	selected = palloc(sizeof(uint32) * PgturbohybridGraphLevelM(state.m, 0));
	inserted = PgturbohybridNativeParallelEdgeInserted(shared);
	order = PgturbohybridNativeParallelEdgeOrder(shared);

	if (!leader)
	{
		for (;;)
		{
			bool		ready;

			ConditionVariablePrepareToSleep(&shared->workersdonecv);
			SpinLockAcquire(&shared->mutex);
			ready = shared->edgeBarrierReady;
			SpinLockRelease(&shared->mutex);
			if (ready)
				break;
			ConditionVariableSleep(&shared->workersdonecv,
								   WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
		}
		ConditionVariableCancelSleep();
	}

	INSTR_TIME_SET_CURRENT(start);
	if (leader)
		PgturbohybridNativeParallelEdgeBuildWarmup(&state, shared, nearest,
												   selected);
	BarrierArriveAndWait(&shared->edgeBarrier,
						  WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

	for (;;)
	{
		uint32		batchStart;
		uint32		batchEnd;

		CHECK_FOR_INTERRUPTS();
		if (leader)
		{
			batchStart = shared->edgeBatchStartOrder;
			if (batchStart >= shared->edgeNodeCount)
			{
				shared->edgeDone = true;
				shared->edgeBatchEndOrder = batchStart;
			}
			else
			{
				batchEnd = Min(batchStart + shared->edgeBatchSize,
							   shared->edgeNodeCount);
				shared->edgeBatchEndOrder = batchEnd;
				shared->edgeNextOrder = batchStart;
			}
		}
		BarrierArriveAndWait(&shared->edgeBarrier,
							  WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
		if (shared->edgeDone)
			break;

		for (;;)
		{
			uint32		orderIdx;

			CHECK_FOR_INTERRUPTS();
			SpinLockAcquire(&shared->mutex);
			orderIdx = shared->edgeNextOrder++;
			SpinLockRelease(&shared->mutex);
			if (orderIdx >= shared->edgeBatchEndOrder)
				break;
			PgturbohybridNativeParallelEdgeBuildNodeLinks(&state,
														  order[orderIdx],
														  shared->edgeEntryNodeId,
														  shared->edgeEntryLevel,
														  inserted, nearest,
														  selected);
		}

		BarrierArriveAndWait(&shared->edgeBarrier,
							  WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

		/*
		 * Apply backlinks by destination node ownership.  Every participant
		 * scans the whole batch, but only mutates neighbor lists for destination
		 * nodes in its shard.  Sharding the source queue would drop backlinks
		 * for destinations owned by other participants.
		 */
		for (uint32 orderIdx = shared->edgeBatchStartOrder;
			 orderIdx < shared->edgeBatchEndOrder; orderIdx++)
		{
			CHECK_FOR_INTERRUPTS();
			PgturbohybridNativeParallelEdgeApplyNodeBacklinks(&state,
															  order[orderIdx],
															  inserted, slot,
															  shared->nparticipants);
		}

		BarrierArriveAndWait(&shared->edgeBarrier,
							  WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);

		if (leader)
		{
			for (uint32 orderIdx = shared->edgeBatchStartOrder;
				 orderIdx < shared->edgeBatchEndOrder; orderIdx++)
				PgturbohybridNativeParallelEdgePublishNode(&state, shared,
														   order[orderIdx],
														   inserted);
			shared->edgeBatchStartOrder = shared->edgeBatchEndOrder;
		}

		BarrierArriveAndWait(&shared->edgeBarrier,
							  WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
	}
	elapsedUs = (uint64) PgturbohybridGraphElapsedUs(start);
	PgturbohybridNativeParallelEdgeWorkerUs(shared)[slot] = elapsedUs;

	SpinLockAcquire(&shared->mutex);
	shared->edgeDistanceCalls += state.buildDistanceCalls;
	shared->edgeEntrySearchUs += state.buildEdgeEntrySearchUs;
	shared->edgeNeighborSearchUs += state.buildEdgeNeighborSearchUs;
	shared->edgeSearchLayerUs +=
		state.buildEdgeEntrySearchUs + state.buildEdgeNeighborSearchUs;
	shared->edgeSelectNeighborUs += state.buildEdgeSelectNeighborUs;
	shared->edgeAddNeighborUs += state.buildEdgeAddNeighborUs;
	shared->edgePruneNeighborUs += state.buildEdgePruneNeighborUs;
	shared->edgeEntryUpdateUs += state.buildEdgeEntryUpdateUs;
	shared->edgeNearestTotal += state.buildEdgeNearestTotal;
	shared->edgeNearestSamples += state.buildEdgeNearestSamples;
	if (state.buildEdgeMaxFrontierSize > shared->edgeMaxFrontierSize)
		shared->edgeMaxFrontierSize = state.buildEdgeMaxFrontierSize;
	shared->nparticipantsdone++;
	SpinLockRelease(&shared->mutex);
	ConditionVariableSignal(&shared->workersdonecv);
	MemoryContextDelete(state.ctx);
}

static void
PgturbohybridNativeParallelApplyEdges(PgturbohybridQuantBuildState *state,
									  PgturbohybridNativeParallelShared *shared,
									  uint64 *repairUs)
{
	int		   *counts = PgturbohybridNativeParallelEdgeNeighborCounts(shared);
	uint32	   *neighbors = PgturbohybridNativeParallelEdgeNeighbors(shared);
	double	   *distances = PgturbohybridNativeParallelEdgeNeighborDistances(shared);
	PgturbohybridGraphSegmentMetaData *segments =
		PgturbohybridNativeParallelEdgeSegments(shared);

	state->segmentCount = 1;
	state->entryNodeId = shared->edgeEntryNodeId;
	state->maxLevel = shared->edgeEntryLevel;
	memset(state->segments, 0, sizeof(state->segments));
	for (uint32 nodeId = 0; nodeId < state->nodeCount; nodeId++)
	{
		PgturbohybridGraphBuildNode *node = &state->nodes[nodeId];

		if (node->neighbors == NULL)
			PgturbohybridGraphAllocateBuildNeighbors(state, node);
		for (int level = 0; level <= node->level; level++)
		{
			Size		levelSlot =
				PgturbohybridNativeParallelEdgeNodeLevelSlot(shared, nodeId, level);
			Size		neighborSlot =
				PgturbohybridNativeParallelEdgeNeighborSlot(shared, nodeId, level);
			int			count = Min((int) counts[levelSlot],
									PgturbohybridGraphLevelM(state->m, level));

			node->neighborCounts[level] = count;
			if (count > 0)
			{
				memcpy(node->neighbors[level], neighbors + neighborSlot,
					   mul_size(sizeof(uint32), (Size) count));
				memcpy(node->neighborDistances[level], distances + neighborSlot,
					   mul_size(sizeof(double), (Size) count));
			}
		}
		state->maxLevel = Max(state->maxLevel, node->level);
	}

	*repairUs = 0;
	memset(segments, 0, sizeof(PgturbohybridGraphSegmentMetaData) *
		   shared->edgeSegmentCount);
	state->segments[0].startNodeId = 0;
	state->segments[0].nodeCount = state->nodeCount;
	state->segments[0].entryNodeId = state->entryNodeId;
	state->segments[0].entryLevel = state->entryNodeId < state->nodeCount ?
		state->nodes[state->entryNodeId].level : -1;
}

static void
PgturbohybridGraphFreeExactBuildVectors(PgturbohybridQuantBuildState *state)
{
	Vector	   *lastVector = NULL;

	if (!state->buildExactDistances || state->tqExactStorage)
		return;

	for (uint32 nodeId = 0; nodeId < state->nodeCount; nodeId++)
	{
		Vector	   *vector = state->nodes[nodeId].vector;

		if (vector != NULL && vector != lastVector)
			pfree(vector);
		lastVector = vector;
		state->nodes[nodeId].vector = NULL;
	}
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
	if (state->segmentCount > 1)
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

		CHECK_FOR_INTERRUPTS();
		if (node->level < 0 || node->neighborCounts == NULL ||
			node->neighbors == NULL)
			continue;

		for (int i = 0; i < node->neighborCounts[0]; i++)
			PGTURBOHYBRID_GRAPH_ENQUEUE_OLD_NODE(node->neighbors[0][i]);
	}

	for (uint32 i = 0; i < state->nodeCount; i++)
	{
		CHECK_FOR_INTERRUPTS();
		PGTURBOHYBRID_GRAPH_ENQUEUE_OLD_NODE(i);
	}

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
	{
		CHECK_FOR_INTERRUPTS();
		reordered[newId] = state->nodes[newToOld[newId]];
	}

	for (uint32 newId = 0; newId < state->nodeCount; newId++)
	{
		PgturbohybridGraphBuildNode *node = &reordered[newId];

		CHECK_FOR_INTERRUPTS();
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

typedef struct PgturbohybridGraphEntrySidecarCandidate
{
	uint32		nodeId;
	uint32		bucket;
	int			level;
	uint64		key;
} PgturbohybridGraphEntrySidecarCandidate;

static int
PgturbohybridGraphEntrySidecarCandidateCompare(const void *a, const void *b)
{
	const PgturbohybridGraphEntrySidecarCandidate *ca =
		(const PgturbohybridGraphEntrySidecarCandidate *) a;
	const PgturbohybridGraphEntrySidecarCandidate *cb =
		(const PgturbohybridGraphEntrySidecarCandidate *) b;

	if (ca->level != cb->level)
		return ca->level > cb->level ? -1 : 1;
	if (ca->bucket != cb->bucket)
		return ca->bucket < cb->bucket ? -1 : 1;
	if (ca->key != cb->key)
		return ca->key < cb->key ? -1 : 1;
	if (ca->nodeId != cb->nodeId)
		return ca->nodeId < cb->nodeId ? -1 : 1;
	return 0;
}

static bool
PgturbohybridGraphEntrySidecarUsable(PgturbohybridQuantBuildState *state,
									 uint32 nodeId)
{
	PgturbohybridGraphBuildNode *node;

	if (nodeId >= state->nodeCount)
		return false;

	node = &state->nodes[nodeId];
	return (node->flags & PGTURBOHYBRID_GRAPH_NODE_DEAD) == 0 &&
		node->code != NULL;
}

static bool
PgturbohybridGraphEntrySidecarSelected(PgturbohybridQuantBuildState *state,
									   uint32 nodeId)
{
	for (uint32 i = 0; i < state->entrySidecarCount; i++)
	{
		if (state->entrySidecarNodeIds[i] == nodeId)
			return true;
	}

	return false;
}

static bool
PgturbohybridGraphEntrySidecarAdd(PgturbohybridQuantBuildState *state,
								  uint32 nodeId, uint32 target)
{
	if (state->entrySidecarCount >= target ||
		!PgturbohybridGraphEntrySidecarUsable(state, nodeId) ||
		PgturbohybridGraphEntrySidecarSelected(state, nodeId))
		return false;

	state->entrySidecarNodeIds[state->entrySidecarCount++] = nodeId;
	return true;
}

static double
PgturbohybridGraphEntrySidecarCodeDistance(PgturbohybridQuantBuildState *state,
										   uint32 a, uint32 b)
{
	double		distance;
	uint64		ha;
	uint64		hb;

	if (PgturbohybridGraphBuildCodeDistance(state, a, b, &distance))
		return distance;

	ha = PgturbohybridGraphBuildCodeHash(state, a);
	hb = PgturbohybridGraphBuildCodeHash(state, b);
	return ha == hb ? 0.0 : 1.0;
}

static void
PgturbohybridGraphBuildEntrySidecarHash(PgturbohybridQuantBuildState *state,
										uint32 target)
{
	uint32	   *bucketNodeIds;
	uint64	   *bucketKeys;

	Assert(target > 0);
	bucketNodeIds = palloc(sizeof(uint32) * target);
	bucketKeys = palloc(sizeof(uint64) * target);
	for (uint32 i = 0; i < target; i++)
	{
		bucketNodeIds[i] = UINT_MAX;
		bucketKeys[i] = UINT64_MAX;
	}

	for (uint32 nodeId = 0; nodeId < state->nodeCount; nodeId++)
	{
		uint64		hash;
		uint64		key;
		uint32		bucket;

		CHECK_FOR_INTERRUPTS();
		if (!PgturbohybridGraphEntrySidecarUsable(state, nodeId))
			continue;

		hash = PgturbohybridGraphBuildCodeHash(state, nodeId);
		bucket = (uint32) (hash % target);
		key = PgturbohybridGraphMix64(hash ^ ((uint64) nodeId << 32) ^ nodeId);
		if (bucketNodeIds[bucket] == UINT_MAX ||
			key < bucketKeys[bucket] ||
			(key == bucketKeys[bucket] && nodeId < bucketNodeIds[bucket]))
		{
			bucketNodeIds[bucket] = nodeId;
			bucketKeys[bucket] = key;
		}
	}

	for (uint32 i = 0; i < target; i++)
	{
		if (bucketNodeIds[i] != UINT_MAX)
			PgturbohybridGraphEntrySidecarAdd(state, bucketNodeIds[i], target);
	}

	pfree(bucketNodeIds);
	pfree(bucketKeys);
}

static void
PgturbohybridGraphBuildEntrySidecarFarthest(PgturbohybridQuantBuildState *state,
											uint32 target)
{
	double	   *minDistances;
	uint32		scoredSelected = 0;

	if (state->entrySidecarCount == 0)
	{
		if (!PgturbohybridGraphEntrySidecarAdd(state, state->entryNodeId, target))
		{
			for (uint32 nodeId = 0; nodeId < state->nodeCount; nodeId++)
			{
				CHECK_FOR_INTERRUPTS();
				if (PgturbohybridGraphEntrySidecarAdd(state, nodeId, target))
					break;
			}
		}
	}

	minDistances = palloc(sizeof(double) * state->nodeCount);
	for (uint32 nodeId = 0; nodeId < state->nodeCount; nodeId++)
		minDistances[nodeId] = DBL_MAX;

	while (state->entrySidecarCount < target)
	{
		uint32		bestNodeId = UINT_MAX;
		double		bestMinDistance = -DBL_MAX;

		while (scoredSelected < state->entrySidecarCount)
		{
			uint32		selectedNodeId = state->entrySidecarNodeIds[scoredSelected++];

			for (uint32 nodeId = 0; nodeId < state->nodeCount; nodeId++)
			{
				double		distance;

				CHECK_FOR_INTERRUPTS();
				if (!PgturbohybridGraphEntrySidecarUsable(state, nodeId) ||
					PgturbohybridGraphEntrySidecarSelected(state, nodeId))
					continue;

				distance = PgturbohybridGraphEntrySidecarCodeDistance(state, nodeId,
																	  selectedNodeId);
				if (distance < minDistances[nodeId])
					minDistances[nodeId] = distance;
			}
		}

		for (uint32 nodeId = 0; nodeId < state->nodeCount; nodeId++)
		{
			CHECK_FOR_INTERRUPTS();
			if (!PgturbohybridGraphEntrySidecarUsable(state, nodeId) ||
				PgturbohybridGraphEntrySidecarSelected(state, nodeId))
				continue;

			if (bestNodeId == UINT_MAX ||
				minDistances[nodeId] > bestMinDistance ||
				(minDistances[nodeId] == bestMinDistance && nodeId < bestNodeId))
			{
				bestNodeId = nodeId;
				bestMinDistance = minDistances[nodeId];
			}
		}

		if (bestNodeId == UINT_MAX)
			break;
		PgturbohybridGraphEntrySidecarAdd(state, bestNodeId, target);
	}

	pfree(minDistances);
}

static void
PgturbohybridGraphBuildEntrySidecarLevelCovering(PgturbohybridQuantBuildState *state,
												 uint32 target)
{
	PgturbohybridGraphEntrySidecarCandidate *buckets;
	PgturbohybridGraphEntrySidecarCandidate *candidates;
	uint32		candidateCount = 0;
	int			maxLevel = 0;

	Assert(target > 0);
	buckets = palloc(sizeof(PgturbohybridGraphEntrySidecarCandidate) * target);
	candidates = palloc(sizeof(PgturbohybridGraphEntrySidecarCandidate) * target);
	for (uint32 i = 0; i < target; i++)
	{
		buckets[i].nodeId = UINT_MAX;
		buckets[i].bucket = i;
		buckets[i].level = INT_MIN;
		buckets[i].key = UINT64_MAX;
	}

	for (uint32 nodeId = 0; nodeId < state->nodeCount; nodeId++)
	{
		PgturbohybridGraphBuildNode *node = &state->nodes[nodeId];
		uint64		hash;
		uint64		key;
		uint32		bucket;

		CHECK_FOR_INTERRUPTS();
		if (!PgturbohybridGraphEntrySidecarUsable(state, nodeId))
			continue;

		hash = PgturbohybridGraphBuildCodeHash(state, nodeId);
		bucket = (uint32) (hash % target);
		key = PgturbohybridGraphMix64(hash ^ ((uint64) nodeId << 32) ^ nodeId);
		if (node->level > maxLevel)
			maxLevel = node->level;
		if (buckets[bucket].nodeId == UINT_MAX ||
			node->level > buckets[bucket].level ||
			(node->level == buckets[bucket].level &&
			 (key < buckets[bucket].key ||
			  (key == buckets[bucket].key && nodeId < buckets[bucket].nodeId))))
		{
			buckets[bucket].nodeId = nodeId;
			buckets[bucket].level = node->level;
			buckets[bucket].key = key;
		}
	}

	for (uint32 i = 0; i < target; i++)
	{
		if (buckets[i].nodeId != UINT_MAX)
			candidates[candidateCount++] = buckets[i];
	}
	qsort(candidates, candidateCount,
		  sizeof(PgturbohybridGraphEntrySidecarCandidate),
		  PgturbohybridGraphEntrySidecarCandidateCompare);

	for (uint32 i = 0; i < candidateCount; i++)
		PgturbohybridGraphEntrySidecarAdd(state, candidates[i].nodeId, target);

	for (int level = maxLevel; level >= 0 && state->entrySidecarCount < target; level--)
	{
		for (uint32 nodeId = 0; nodeId < state->nodeCount &&
			 state->entrySidecarCount < target; nodeId++)
		{
			CHECK_FOR_INTERRUPTS();
			if (PgturbohybridGraphEntrySidecarUsable(state, nodeId) &&
				state->nodes[nodeId].level == level)
				PgturbohybridGraphEntrySidecarAdd(state, nodeId, target);
		}
	}

	pfree(candidates);
	pfree(buckets);
}

static void
PgturbohybridGraphBuildEntrySidecar(PgturbohybridQuantBuildState *state)
{
	uint32		target;

	state->entrySidecarCount = 0;
	state->entrySidecarBytes = 0;
	memset(state->entrySidecarNodeIds, 0, sizeof(state->entrySidecarNodeIds));

	if (!state->entrySidecar || state->nodeCount == 0 ||
		state->entrySidecarRepresentatives <= 0)
		return;

	target = Min((uint32) state->entrySidecarRepresentatives,
				 Min(state->nodeCount,
					 (uint32) PGTURBOHYBRID_GRAPH_MAX_ENTRY_SIDECAR_REPRESENTATIVES));
	if (target == 0)
		return;

	switch ((PgturbohybridEntrySidecarStrategy) state->entrySidecarStrategy)
	{
		case PGTURBOHYBRID_ENTRY_SIDECAR_FARTHEST_CODE:
			PgturbohybridGraphBuildEntrySidecarFarthest(state, target);
			break;
		case PGTURBOHYBRID_ENTRY_SIDECAR_LEVEL_COVERING:
			PgturbohybridGraphBuildEntrySidecarLevelCovering(state, target);
			break;
		case PGTURBOHYBRID_ENTRY_SIDECAR_HYBRID_LEVEL_COVERING:
			PgturbohybridGraphBuildEntrySidecarLevelCovering(state,
															 Max((uint32) 1, target / 4));
			PgturbohybridGraphBuildEntrySidecarFarthest(state, target);
			break;
		case PGTURBOHYBRID_ENTRY_SIDECAR_HASH:
		default:
			PgturbohybridGraphBuildEntrySidecarHash(state, target);
			break;
	}
	state->entrySidecarBytes = state->entrySidecarCount * sizeof(uint32);
}

static void
PgturbohybridGraphOfferRoutingEntry(PgturbohybridQuantBuildState *state,
						 int *routingLevels, uint32 nodeId, int level)
{
	int			worst = 0;

	if (state->routingEntryCount < PGTURBOHYBRID_GRAPH_MAX_ROUTING_ENTRIES)
	{
		int			slot = state->routingEntryCount++;

		state->routingEntryNodeIds[slot] = nodeId;
		routingLevels[slot] = level;
		return;
	}

	for (uint32 i = 1; i < state->routingEntryCount; i++)
	{
		if (routingLevels[i] < routingLevels[worst] ||
			(routingLevels[i] == routingLevels[worst] &&
			 state->routingEntryNodeIds[i] > state->routingEntryNodeIds[worst]))
			worst = i;
	}

	if (level > routingLevels[worst] ||
		(level == routingLevels[worst] && nodeId < state->routingEntryNodeIds[worst]))
	{
		state->routingEntryNodeIds[worst] = nodeId;
		routingLevels[worst] = level;
	}
}

static void
PgturbohybridGraphBuildRoutingEntries(PgturbohybridQuantBuildState *state)
{
	int			routingLevels[PGTURBOHYBRID_GRAPH_MAX_ROUTING_ENTRIES];

	state->routingEntryCount = 0;
	state->routingEntryBytes = 0;
	memset(state->routingEntryNodeIds, 0, sizeof(state->routingEntryNodeIds));
	memset(routingLevels, 0, sizeof(routingLevels));

	if (state->nodeCount == 0 || state->entryNodeId >= state->nodeCount)
		return;

	for (uint32 nodeId = 0; nodeId < state->nodeCount; nodeId++)
	{
		PgturbohybridGraphBuildNode *node = &state->nodes[nodeId];

		CHECK_FOR_INTERRUPTS();
		if (nodeId == state->entryNodeId ||
			(node->flags & PGTURBOHYBRID_GRAPH_NODE_DEAD) != 0 ||
			node->level <= 0)
			continue;

		PgturbohybridGraphOfferRoutingEntry(state, routingLevels, nodeId,
											node->level);
	}

	state->routingEntryBytes = state->routingEntryCount * sizeof(uint32);
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
	metap->version = PGTURBOHYBRID_GRAPH_NATIVE_VERSION;
	metap->graphGeneration = 1;
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
	metap->tqMultivectorDocMapStartBlkno = InvalidBlockNumber;
	metap->tqMultivectorDocMapPageCount = 0;
	metap->tqMultivectorDocCount = 0;
	metap->tqMultivectorDocMapBytes = 0;
	metap->tqMultivectorDocMapVersion = 0;
	metap->tqMultivectorDocMapFlags = 0;
	metap->tqMultivectorGraphMode =
		PgturbohybridGraphIndexIsMultiVector(index) ?
		(uint16) PgturbohybridGraphGetMultiVectorGraphModeOption(index) :
		PGTURBOHYBRID_DEFAULT_MULTIVECTOR_GRAPH_MODE;
	metap->tqEntrySidecarCount = 0;
	metap->tqEntrySidecarBytes = 0;
	metap->tqRoutingEntryCount = 0;
	metap->tqRoutingEntryBytes = 0;
	metap->tqSegmentCount = 0;
	metap->tqSegmentBytes = 0;
	((PageHeader) page)->pd_lower =
		((char *) metap + sizeof(PgturbohybridGraphMetaPageData)) - (char *) page;

	PgturbohybridGraphFinishPage(buf);
}

static void
PgturbohybridQuantInitMetaUpdateFromBuild(PgturbohybridQuantBuildState *state,
							   PgturbohybridQuantMetaUpdate *update,
							   BlockNumber codeStart, BlockNumber adjStart,
							   BlockNumber exactStart,
							   BlockNumber correctionStart,
							   BlockNumber docMapStart,
							   uint32 docMapPageCount,
							   uint32 docMapBytes)
{
	memset(update, 0, sizeof(*update));
	update->forkNum = state->forkNum;
	update->building = state->building;
	update->dimensions = state->dimensions;
	update->m = state->m;
	update->efConstruction = state->efConstruction;
	update->graphMaxLevel = state->maxLevel;
	update->nodeCount = state->nodeCount;
	update->entryNodeId = state->entryNodeId;
	update->entryLevel = state->nodeCount > 0 ?
		state->nodes[state->entryNodeId].level : -1;
	update->tqBits = state->tqBits;
	update->tqPayloadCount = state->payloadCount;
	update->tqPayloadBytes = state->payloadBytes;
	update->tqFlags = state->ecShift != NULL && state->ecScale != NULL ?
		PGTURBOHYBRID_GRAPH_TQ_PLUS : 0;
	if (state->tqWeighted)
		update->tqFlags |= PGTURBOHYBRID_GRAPH_TQ_WEIGHTED;
	if (state->tqRenorm)
		update->tqFlags |= PGTURBOHYBRID_GRAPH_TQ_RENORM;
	if (!state->tqExactStorage)
		update->tqFlags |= PGTURBOHYBRID_GRAPH_EXACT_FREE;
	if (state->residualRerankBytes > 0)
		update->tqFlags |= PGTURBOHYBRID_GRAPH_TQ_RESIDUAL_RERANK;
	if (state->buildExactDistances)
		update->tqFlags |= PGTURBOHYBRID_GRAPH_TQ_EXACT_BUILD;
	if (state->graphBackbone)
		update->tqFlags |= PGTURBOHYBRID_GRAPH_TQ_BACKBONE;
	if (state->buildFastEdges)
		update->tqFlags |= PGTURBOHYBRID_GRAPH_TQ_FAST_BUILD_EDGES;
	update->tqFlags |=
		PGTURBOHYBRID_GRAPH_TQ_BUILD_NEIGHBOR_REASON_BITS(state->buildNeighborSelectReason);
	update->tqEntrySidecarCount = state->entrySidecarCount;
	update->tqEntrySidecarBytes = state->entrySidecarBytes;
	update->tqResidualRerankBytes = state->residualRerankBytes;
	update->tqMultivectorDocMapStartBlkno = docMapStart;
	update->tqMultivectorDocMapPageCount = docMapPageCount;
	update->tqMultivectorDocCount =
		BlockNumberIsValid(docMapStart) ? state->multivectorDocCount : 0;
	update->tqMultivectorDocMapBytes = docMapBytes;
	update->tqMultivectorDocMapVersion =
		BlockNumberIsValid(docMapStart) ?
		PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION : 0;
	update->tqMultivectorDocMapFlags = 0;
	if (state->multivectorBuild &&
		state->multivectorGraphMode ==
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES &&
		BlockNumberIsValid(docMapStart))
	{
		if (state->multivectorDocStorage ==
			PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_PROXY_ONLY)
			update->tqMultivectorDocMapFlags |=
				PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_PROXY_ONLY;
		else if (PgturbohybridGraphMultiVectorDocMapStoresDocVectors(state))
			update->tqMultivectorDocMapFlags |=
				PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_DOC_VECTORS;
	}
	if (BlockNumberIsValid(docMapStart) &&
		PgturbohybridGraphMultiVectorDocMapStoresDocVectors(state) &&
		PgturbohybridGraphMultiVectorDocMapHasContexts(state))
		update->tqMultivectorDocMapFlags |=
			PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_CONTEXTS;
	if (BlockNumberIsValid(docMapStart) &&
		PgturbohybridGraphMultiVectorDocMapStoresQuantizedInvertedPostings(state) &&
		state->multivectorGraphMode ==
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES)
	{
		update->tqMultivectorDocMapFlags |=
			PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_QUANTIZED_POSTINGS;
		update->tqMultivectorDocMapFlags |=
			PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_QUANTIZED_CODEBOOK;
	}
	if (BlockNumberIsValid(docMapStart) &&
		PgturbohybridGraphMultiVectorDocMapStoresCentroids(state) &&
		state->multivectorGraphMode ==
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES)
	{
		update->tqMultivectorDocMapFlags |=
			PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_CENTROIDS;
		update->tqMultivectorDocMapFlags |=
			PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_CENTROID_POSTINGS;
		update->tqMultivectorDocMapFlags |=
			PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_CENTROID_DOC_CODES;
		if (state->multivectorDocStorage ==
			PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CENTROID_ONLY)
			update->tqMultivectorDocMapFlags |=
				PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_CENTROIDS_F16;
	}
	update->tqMultivectorGraphMode =
		state->multivectorBuild ? (uint16) state->multivectorGraphMode :
		PGTURBOHYBRID_DEFAULT_MULTIVECTOR_GRAPH_MODE;
	if (state->entrySidecarCount > 0)
		memcpy(update->tqEntrySidecarNodeIds, state->entrySidecarNodeIds,
			   sizeof(uint32) * state->entrySidecarCount);
	update->tqRoutingEntryCount = state->routingEntryCount;
	update->tqRoutingEntryBytes = state->routingEntryBytes;
	if (state->routingEntryCount > 0)
		memcpy(update->tqRoutingEntryNodeIds, state->routingEntryNodeIds,
			   sizeof(uint32) * state->routingEntryCount);
	update->tqSegmentCount = state->segmentCount;
	for (uint16 i = 0; i < state->segmentCount; i++)
	{
		update->tqSegments[i] = state->segments[i];
		update->tqSegments[i].codeStartBlkno = codeStart;
		update->tqSegments[i].adjStartBlkno = adjStart;
		update->tqSegments[i].exactStartBlkno = exactStart;
		update->tqSegments[i].correctionStartBlkno = correctionStart;
	}
	update->buildScanUs = state->buildScanUs;
	update->buildCorrectionUs = state->buildCorrectionUs;
	update->buildEncodeUs = state->buildEncodeUs;
	update->buildEdgeUs = state->buildEdgeUs;
	update->buildWriteUs = state->buildWriteUs;
	update->buildWorkerCount = state->buildWorkerCount;
}

void
PgturbohybridQuantUpdateMetaPageFromUpdate(Relation index,
							 const PgturbohybridQuantMetaUpdate *update,
							 BlockNumber codeStart, BlockNumber adjStart,
							 BlockNumber exactStart,
							 BlockNumber correctionStart)
{
	Buffer		buf;
	Page		page;
	PgturbohybridGraphMetaPage metap;
	GenericXLogState *xlogState = NULL;

	buf = ReadBufferExtended(index, update->forkNum, PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);

	if (!update->building && update->forkNum == MAIN_FORKNUM && RelationNeedsWAL(index))
	{
		xlogState = GenericXLogStart(index);
		page = GenericXLogRegisterBuffer(xlogState, buf, 0);
	}
	else
		page = BufferGetPage(buf);

	metap = PgturbohybridGraphPageGetMeta(page);

	metap->version = PGTURBOHYBRID_GRAPH_NATIVE_VERSION;
	metap->dimensions = update->dimensions;
	metap->m = update->m;
	metap->efConstruction = update->efConstruction;
	metap->storageKind = PGTURBOHYBRID_GRAPH_STORAGE_QUANT_GRAPH_NATIVE;
	metap->graphEfSearch = PgturbohybridGraphGetEfSearch(index);
	metap->graphOversampling = PgturbohybridGraphGetGraphOversampling(index);
	metap->graphRescoreBand = PgturbohybridGraphGetGraphRescoreBand(index);
	metap->graphMaxLevel = update->graphMaxLevel;
	if (metap->graphGeneration == UINT64_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pgturbohybrid graph generation exhausted"),
				 errhint("REINDEX the index to rebuild it.")));
	metap->graphGeneration = metap->graphGeneration == 0 ? 1 : metap->graphGeneration + 1;
	metap->entryBlkno = codeStart;
	metap->entryOffno = update->nodeCount > 0 ? FirstOffsetNumber : InvalidOffsetNumber;
	metap->entryLevel = update->nodeCount > 0 ? update->entryLevel : -1;
	metap->tqNodeCount = update->nodeCount;
	metap->tqEntryNodeId = update->nodeCount > 0 ? update->entryNodeId : UINT_MAX;
	metap->tqCodeBytes = update->dimensions > 0 ? PgturbohybridGraphCodeBytesForBits(update->dimensions, update->tqBits) : 0;
	metap->tqPayloadCount = update->tqPayloadCount;
	metap->tqPayloadBytes = update->tqPayloadBytes;
	metap->tqFlags = update->tqFlags;
	metap->tqBits = update->tqBits != 0 ? update->tqBits : PGTURBOHYBRID_DEFAULT_BITS;
	metap->tqCodeStartBlkno = codeStart;
	metap->tqAdjStartBlkno = adjStart;
	metap->tqExactStartBlkno = exactStart;
	metap->tqCorrectionStartBlkno = correctionStart;
	metap->tqMultivectorDocMapStartBlkno =
		update->tqMultivectorDocMapStartBlkno;
	metap->tqMultivectorDocMapPageCount =
		update->tqMultivectorDocMapPageCount;
	metap->tqMultivectorDocCount = update->tqMultivectorDocCount;
	metap->tqMultivectorDocMapBytes =
		update->tqMultivectorDocMapBytes;
	metap->tqMultivectorDocMapVersion =
		update->tqMultivectorDocMapVersion;
	metap->tqMultivectorDocMapFlags =
		update->tqMultivectorDocMapFlags;
	metap->tqMultivectorGraphMode =
		update->tqMultivectorGraphMode;
	metap->tqEntrySidecarCount = update->tqEntrySidecarCount;
	metap->tqEntrySidecarBytes = update->tqEntrySidecarBytes;
	metap->tqResidualRerankBytes = update->tqResidualRerankBytes;
	memset(metap->tqEntrySidecarNodeIds, 0, sizeof(metap->tqEntrySidecarNodeIds));
	if (update->tqEntrySidecarCount > 0)
		memcpy(metap->tqEntrySidecarNodeIds, update->tqEntrySidecarNodeIds,
			   sizeof(uint32) * update->tqEntrySidecarCount);
	metap->tqRoutingEntryCount = update->tqRoutingEntryCount;
	metap->tqRoutingEntryBytes = update->tqRoutingEntryBytes;
	memset(metap->tqRoutingEntryNodeIds, 0, sizeof(metap->tqRoutingEntryNodeIds));
	if (update->tqRoutingEntryCount > 0)
		memcpy(metap->tqRoutingEntryNodeIds, update->tqRoutingEntryNodeIds,
			   sizeof(uint32) * update->tqRoutingEntryCount);
	metap->tqSegmentCount = update->tqSegmentCount;
	metap->tqSegmentBytes = update->tqSegmentCount * sizeof(PgturbohybridGraphSegmentMetaData);
	memset(metap->tqSegments, 0, sizeof(metap->tqSegments));
	if (update->tqSegmentCount > 0)
		memcpy(metap->tqSegments, update->tqSegments,
			   sizeof(PgturbohybridGraphSegmentMetaData) * update->tqSegmentCount);
	metap->buildScanUs = update->buildScanUs;
	metap->buildCorrectionUs = update->buildCorrectionUs;
	metap->buildEncodeUs = update->buildEncodeUs;
	metap->buildEdgeUs = update->buildEdgeUs;
	metap->buildWriteUs = update->buildWriteUs;
	metap->buildWorkerCount = update->buildWorkerCount;
	PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_META_UPDATE);

	if (xlogState != NULL)
		GenericXLogFinish(xlogState);
	else
		MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);

	PgturbohybridGraphLogGraphWalRecord(index, update->forkNum, PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO, PGTURBOHYBRID_GRAPH_GRAPH_OP_META_UPDATE);
}

void
PgturbohybridQuantUpdateMetaPage(Relation index, PgturbohybridQuantBuildState *state,
					  BlockNumber codeStart, BlockNumber adjStart,
					  BlockNumber exactStart, BlockNumber correctionStart,
					  BlockNumber docMapStart, uint32 docMapPageCount,
					  uint32 docMapBytes)
{
	PgturbohybridQuantMetaUpdate update;

	PgturbohybridQuantInitMetaUpdateFromBuild(state, &update, codeStart, adjStart,
											  exactStart, correctionStart,
											  docMapStart, docMapPageCount,
											  docMapBytes);
	PgturbohybridQuantUpdateMetaPageFromUpdate(index, &update, codeStart, adjStart,
											   exactStart, correctionStart);
}

void
PgturbohybridGraphUpdateMultiVectorDocMapMeta(Relation index,
									 BlockNumber startBlkno,
									 uint32 pageCount,
									 uint32 docCount,
									 uint32 docMapBytes,
									 uint16 docMapFlags)
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
	if (metap->graphGeneration == UINT64_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pgturbohybrid graph generation exhausted"),
				 errhint("REINDEX the index to rebuild it.")));
	metap->graphGeneration = metap->graphGeneration == 0 ? 1 : metap->graphGeneration + 1;
	metap->tqMultivectorDocMapStartBlkno = startBlkno;
	metap->tqMultivectorDocMapPageCount = pageCount;
	metap->tqMultivectorDocCount = docCount;
	metap->tqMultivectorDocMapBytes = docMapBytes;
	metap->tqMultivectorDocMapVersion =
		BlockNumberIsValid(startBlkno) ?
		PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION : 0;
	metap->tqMultivectorDocMapFlags = docMapFlags;
	PgturbohybridGraphMarkPageGraphOp(page,
									  PGTURBOHYBRID_GRAPH_GRAPH_OP_META_UPDATE);

	if (xlogState != NULL)
		GenericXLogFinish(xlogState);
	else
		MarkBufferDirty(buf);

	UnlockReleaseBuffer(buf);
	PgturbohybridGraphLogGraphWalRecord(index, MAIN_FORKNUM,
										PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO,
										PGTURBOHYBRID_GRAPH_GRAPH_OP_META_UPDATE);
}

/* Non-static: also called from the extracted vacuum module. */
void
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
	if (metap->graphGeneration == UINT64_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pgturbohybrid graph generation exhausted"),
				 errhint("REINDEX the index to rebuild it.")));
	metap->graphGeneration = metap->graphGeneration == 0 ? 1 : metap->graphGeneration + 1;
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
												 state->tqBits, state->tqWeighted,
												 state->residualRerankBytes);
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
		if (state->residualRerankBytes > 0 && node->residualSketch != NULL)
			memcpy(PgturbohybridGraphTupleResidual(tuple, state->payloadBytes,
												   state->residualRerankBytes,
												   state->tqWeighted),
				   node->residualSketch, state->residualRerankBytes);
		memcpy(PgturbohybridGraphTupleCode(tuple, state->payloadBytes,
										   state->residualRerankBytes,
										   state->tqWeighted), node->code,
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

static uint16
PgturbohybridGraphMultiVectorDocMapNodeTupleMaxCount(void)
{
	Size		header = MAXALIGN(offsetof(PgturbohybridGraphMultiVectorDocMapNodeTupleData,
										   entries));
	Size		usable = PgturbohybridGraphDocMapMaxItemSize();
	Size		maxCount;

	if (header >= usable)
		return 1;
	maxCount = (usable - header) / sizeof(TqMultiVectorNodeMapEntry);
	return (uint16) Max(1, Min((Size) UINT16_MAX, maxCount));
}

static uint16
PgturbohybridGraphMultiVectorDocMapDocTupleMaxCount(void)
{
	Size		header = MAXALIGN(offsetof(PgturbohybridGraphMultiVectorDocMapDocTupleData,
										   entries));
	Size		usable = PgturbohybridGraphDocMapMaxItemSize();
	Size		maxCount;

	if (header >= usable)
		return 1;
	maxCount = (usable - header) / sizeof(TqMultiVectorDocMapEntry);
	return (uint16) Max(1, Min((Size) UINT16_MAX, maxCount));
}

static uint16
PgturbohybridGraphMultiVectorDocMapVectorTupleMaxCount(void)
{
	Size		header = MAXALIGN(offsetof(PgturbohybridGraphMultiVectorDocMapVectorTupleData,
										   values));
	Size		usable = PgturbohybridGraphDocMapMaxItemSize();
	Size		maxCount;

	if (header >= usable)
		return 1;
	maxCount = (usable - header) / sizeof(float);
	return (uint16) Max(1, Min((Size) UINT16_MAX, maxCount));
}

static uint16
PgturbohybridGraphMultiVectorDocMapVectorF16TupleMaxCount(void)
{
	Size		header = MAXALIGN(offsetof(PgturbohybridGraphMultiVectorDocMapVectorF16TupleData,
										   values));
	Size		usable = PgturbohybridGraphDocMapMaxItemSize();
	Size		maxCount;

	if (header >= usable)
		return 1;
	maxCount = (usable - header) / sizeof(uint16);
	return (uint16) Max(1, Min((Size) UINT16_MAX, maxCount));
}

static uint16
PgturbohybridGraphMultiVectorDocMapVectorSq8TupleMaxCount(void)
{
	Size		header = MAXALIGN(offsetof(PgturbohybridGraphMultiVectorDocMapVectorSq8TupleData,
										   values));
	Size		usable = PgturbohybridGraphDocMapMaxItemSize();
	Size		maxCount;

	if (header >= usable)
		return 1;
	maxCount = (usable - header) / sizeof(int8);
	return (uint16) Max(1, Min((Size) UINT16_MAX, maxCount));
}

static uint16
PgturbohybridGraphMultiVectorDocMapCentroidTupleMaxCount(void)
{
	Size		header = MAXALIGN(offsetof(PgturbohybridGraphMultiVectorDocMapCentroidTupleData,
										   values));
	Size		usable = PgturbohybridGraphDocMapMaxItemSize();
	Size		maxCount;

	if (header >= usable)
		return 1;
	maxCount = (usable - header) / sizeof(float);
	return (uint16) Max(1, Min((Size) UINT16_MAX, maxCount));
}

static uint16
PgturbohybridGraphMultiVectorDocMapCentroidF16TupleMaxCount(void)
{
	Size		header = MAXALIGN(offsetof(PgturbohybridGraphMultiVectorDocMapCentroidF16TupleData,
										   values));
	Size		usable = PgturbohybridGraphDocMapMaxItemSize();
	Size		maxCount;

	if (header >= usable)
		return 1;
	maxCount = (usable - header) / sizeof(uint16);
	return (uint16) Max(1, Min((Size) UINT16_MAX, maxCount));
}

static uint16
PgturbohybridGraphMultiVectorDocMapCentroidDocCodeTupleMaxCount(void)
{
	Size		header = MAXALIGN(offsetof(PgturbohybridGraphMultiVectorDocMapCentroidDocCodeTupleData,
										   codes));
	Size		usable = PgturbohybridGraphDocMapMaxItemSize();
	Size		maxCount;

	if (header >= usable)
		return 1;
	maxCount = (usable - header) / sizeof(uint32);
	return (uint16) Max(1, Min((Size) UINT16_MAX, maxCount));
}

static bool
PgturbohybridGraphMultiVectorDocMapStoresDocVectors(PgturbohybridQuantBuildState *state)
{
	return state != NULL &&
		state->multivectorGraphMode ==
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES &&
		state->multivectorDocStorage !=
		PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_PROXY_ONLY &&
		state->multivectorDocStorage !=
		PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CENTROID_ONLY;
}

static bool
PgturbohybridGraphMultiVectorDocMapStoresCentroids(PgturbohybridQuantBuildState *state)
{
	return state != NULL &&
		state->multivectorGraphMode ==
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES &&
		state->multivectorDocStorage !=
		PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_PROXY_ONLY &&
		state->multivectorCentroids ==
		PGTURBOHYBRID_MULTIVECTOR_CENTROIDS_KMEANS;
}

static bool
PgturbohybridGraphMultiVectorDocMapStoresQuantizedInvertedPostings(PgturbohybridQuantBuildState *state)
{
	if (state == NULL ||
		state->multivectorGraphMode !=
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES ||
		state->multivectorDocStorage ==
		PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_PROXY_ONLY)
		return false;
	if (PgturbohybridGraphMultiVectorDocMapStoresDocVectors(state))
		return true;
	return state->multivectorDocStorage ==
		PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CENTROID_ONLY &&
		pgturbohybrid_multivector_quantized_inverted_codebook ==
		PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_EXTERNAL;
}

static void
PgturbohybridGraphAppendMultiVectorQuantizedInvertedBuildPostings(PgturbohybridQuantBuildState *state,
																  const PgturbohybridMultiVector *mv,
																  uint32 docId)
{
	uint32		codebookSize;
	uint32		codebookTopM = 1;
	PgturbohybridQuantizedInvertedCodebook *externalCodebook = NULL;
	bool		useExternalCodebook;
	uint32	   *assignedCodewords = NULL;

	if (!PgturbohybridGraphMultiVectorDocMapStoresQuantizedInvertedPostings(state) ||
		PgturbohybridGraphMultiVectorDocMapStoresDocVectors(state) ||
		mv == NULL || mv->count <= 0)
		return;
	useExternalCodebook =
		pgturbohybrid_multivector_quantized_inverted_codebook ==
		PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_EXTERNAL;
	if (useExternalCodebook)
	{
		externalCodebook = PgturbohybridLoadQuantizedInvertedCodebook(mv->dim);
		codebookSize = externalCodebook->codebookSize;
		codebookTopM = externalCodebook->topM;
	}
	else
		codebookSize = (uint32) mv->dim * 2U;
	if (state->multivectorQuantizedBuildCodebookSize == 0)
		state->multivectorQuantizedBuildCodebookSize = codebookSize;
	else if (state->multivectorQuantizedBuildCodebookSize != codebookSize)
		ereport(ERROR,
				(errcode(ERRCODE_INDEX_CORRUPTED),
				 errmsg("quantized_inverted_experimental build codebook size changed during index build")));
	if ((uint64) state->multivectorQuantizedBuildPostingCount +
		(uint64) mv->count * (uint64) codebookTopM > (uint64) PG_UINT32_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pgturbohybrid quantized inverted sidecar is too large")));
	if (state->multivectorQuantizedBuildPostingLists == NULL)
	{
		if ((Size) codebookSize >
			MaxAllocSize /
			sizeof(PgturbohybridGraphMultiVectorQuantizedBuildPostingList))
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("pgturbohybrid quantized inverted sidecar is too large")));
		state->multivectorQuantizedBuildPostingLists =
			MemoryContextAllocZero(state->ctx,
								   sizeof(PgturbohybridGraphMultiVectorQuantizedBuildPostingList) *
								   (Size) codebookSize);
	}
	assignedCodewords = palloc(sizeof(uint32) * (Size) codebookTopM);
	for (int32 token = 0; token < mv->count; token++)
	{
		uint32		assignedCount;

		if (useExternalCodebook)
		{
			Assert(externalCodebook != NULL);
			assignedCount =
				PgturbohybridMultiVectorQuantizedInvertedConfigurableCodewords(mv,
																			  token,
																			  codebookTopM,
																			  assignedCodewords,
																			  NULL);
		}
		else
		{
			PgturbohybridMultiVectorQuantizedInvertedAssignment assignment =
				PgturbohybridMultiVectorQuantizedInvertedBestCodewordAndScore(mv,
																			  token,
																			  NULL);
			assignedCodewords[0] = assignment.codeword;
			assignedCount = 1;
		}
		if (assignedCount == 0)
			ereport(ERROR,
					(errcode(ERRCODE_INDEX_CORRUPTED),
					 errmsg("quantized_inverted_experimental build assigned no codeword")));
		for (uint32 assignedIndex = 0; assignedIndex < assignedCount;
			 assignedIndex++)
		{
			uint32		codeword = assignedCodewords[assignedIndex];
			PgturbohybridGraphMultiVectorQuantizedBuildPostingList *list;
			PgturbohybridGraphMultiVectorQuantizedPostingEntry *entry;
			double		score;

			if (codeword >= codebookSize)
				ereport(ERROR,
						(errcode(ERRCODE_INDEX_CORRUPTED),
						 errmsg("quantized_inverted_experimental build assigned an out-of-range codeword")));
			list = &state->multivectorQuantizedBuildPostingLists[codeword];
			if (list->count >= list->capacity)
			{
				uint32		newCapacity =
					list->capacity == 0 ? 1024U : list->capacity;
				Size		newBytes;

				while (newCapacity <= list->count)
				{
					if (newCapacity > PG_UINT32_MAX / 2U)
					{
						newCapacity = list->count + 1;
						break;
					}
					newCapacity *= 2U;
				}
				if ((Size) newCapacity >
					MaxAllocSize /
					sizeof(PgturbohybridGraphMultiVectorQuantizedPostingEntry))
					ereport(ERROR,
							(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
							 errmsg("pgturbohybrid quantized inverted posting list is too large"),
							 errhint("Use a smaller external codebook, token pooling, or a more selective quantized inverted profile.")));
				newBytes =
					sizeof(PgturbohybridGraphMultiVectorQuantizedPostingEntry) *
					(Size) newCapacity;
				if (list->postings == NULL)
					list->postings = MemoryContextAlloc(state->ctx, newBytes);
				else
					list->postings = repalloc(list->postings, newBytes);
				list->capacity = newCapacity;
			}
			entry = &list->postings[list->count++];
			state->multivectorQuantizedBuildPostingCount++;
			entry->docId = docId;
			entry->tokenOrdinal = (uint16) token;
			score =
				useExternalCodebook ?
				PgturbohybridMultiVectorQuantizedInvertedCodewordScore(mv,
																		token,
																		codeword) :
				PgturbohybridMultiVectorDeterministicCodewordScore(mv, token,
																   codeword);
			entry->scorePayload =
				useExternalCodebook ?
				(uint16) PgturbohybridMultiVectorQuantizedInvertedCompactScorePayload(score) :
				PgturbohybridMultiVectorQuantizedInvertedScorePayload(mv, token);
		}
	}
	pfree(assignedCodewords);
	state->multivectorQuantizedPostingBytes =
		PgturbohybridGraphBuildMultiplyEstimate(sizeof(PgturbohybridGraphMultiVectorQuantizedPostingEntry),
												(uint64) state->multivectorQuantizedBuildPostingCount);
}

static uint64
PgturbohybridGraphBuildMultiplyEstimate(uint64 left, uint64 right)
{
	if (left != 0 && right > PG_UINT64_MAX / left)
		return PG_UINT64_MAX;
	return left * right;
}

static uint64
PgturbohybridGraphBuildAddEstimate(uint64 left, uint64 right)
{
	if (right > PG_UINT64_MAX - left)
		return PG_UINT64_MAX;
	return left + right;
}

static uint64
PgturbohybridGraphBuildNeighborBytesEstimate(PgturbohybridQuantBuildState *state)
{
	uint64		perNode;
	uint64		levelSlots;

	if (state == NULL || state->nodeCapacity == 0)
		return 0;

	/*
	 * Build-time neighbor storage is variable by random graph level.  Keep this
	 * estimate cheap by accounting for level-0 slots, which dominate the dense
	 * document-node builds that trigger the 1M memory peak.
	 */
	levelSlots = (uint64) PgturbohybridGraphLevelM(state->m, 0) + 1;
	perNode = MAXALIGN(sizeof(uint32 *));
	perNode = PgturbohybridGraphBuildAddEstimate(perNode,
												 MAXALIGN(sizeof(double *)));
	perNode = PgturbohybridGraphBuildAddEstimate(perNode,
												 MAXALIGN(sizeof(int)));
	perNode = PgturbohybridGraphBuildAddEstimate(perNode,
												 MAXALIGN(PgturbohybridGraphBuildMultiplyEstimate(sizeof(uint32),
																								  levelSlots)));
	perNode = PgturbohybridGraphBuildAddEstimate(perNode,
												 MAXALIGN(PgturbohybridGraphBuildMultiplyEstimate(sizeof(double),
																								  levelSlots)));
	return PgturbohybridGraphBuildMultiplyEstimate(perNode,
												   (uint64) state->nodeCapacity);
}

static void
PgturbohybridGraphRefreshBuildMemoryEstimates(PgturbohybridQuantBuildState *state)
{
	uint64		graphNodeBytes;

	if (state == NULL)
		return;

	state->multivectorDocVectorsPointerBytes =
		state->multivectorDocVectors != NULL ?
		PgturbohybridGraphBuildMultiplyEstimate(sizeof(PgturbohybridMultiVector *),
												(uint64) state->multivectorDocCapacity) : 0;
	state->multivectorDocVectorChunkRefBytes =
		state->multivectorDocVectorChunks != NULL ?
		PgturbohybridGraphBuildMultiplyEstimate(sizeof(PgturbohybridGraphMultiVectorDocVectorChunkRef),
												(uint64) state->multivectorDocVectorChunkCapacity) : 0;
	if (state->multivectorDocVectorFirstChunk != NULL)
		state->multivectorDocVectorChunkRefBytes =
			PgturbohybridGraphBuildAddEstimate(state->multivectorDocVectorChunkRefBytes,
											   PgturbohybridGraphBuildMultiplyEstimate(sizeof(uint32),
																					  (uint64) state->multivectorDocCapacity));
	if (state->multivectorDocVectorChunkCounts != NULL)
		state->multivectorDocVectorChunkRefBytes =
			PgturbohybridGraphBuildAddEstimate(state->multivectorDocVectorChunkRefBytes,
											   PgturbohybridGraphBuildMultiplyEstimate(sizeof(uint32),
																					  (uint64) state->multivectorDocCapacity));
	state->multivectorCentroidResidualBytes =
		state->multivectorDocCentroidResiduals != NULL ?
		PgturbohybridGraphBuildMultiplyEstimate(sizeof(float),
												(uint64) state->multivectorDocCapacity) : 0;
	graphNodeBytes =
		PgturbohybridGraphBuildMultiplyEstimate(sizeof(PgturbohybridGraphBuildNode),
												(uint64) state->nodeCapacity);
	if (state->multivectorNodeMap != NULL)
		graphNodeBytes =
			PgturbohybridGraphBuildAddEstimate(graphNodeBytes,
											   PgturbohybridGraphBuildMultiplyEstimate(sizeof(TqMultiVectorNodeMapEntry),
																					  (uint64) state->nodeCapacity));
	state->graphNodeBytesEstimate = graphNodeBytes;
	state->graphNeighborBytesEstimate =
		PgturbohybridGraphBuildNeighborBytesEstimate(state);
}

static void
PgturbohybridGraphRecordBuildMemorySnapshot(PgturbohybridQuantBuildState *state,
											const char *phase)
{
	uint64		bytes;

	if (state == NULL || state->ctx == NULL || phase == NULL)
		return;

	PgturbohybridGraphRefreshBuildMemoryEstimates(state);
	bytes = (uint64) MemoryContextMemAllocated(state->ctx, true);
	if (strcmp(phase, "heap_scan_decode") == 0)
		state->buildMemoryHeapScanDecodeBytes = bytes;
	else if (strcmp(phase, "token_pooling") == 0)
		state->buildMemoryTokenPoolingBytes = bytes;
	else if (strcmp(phase, "proxy_encoding") == 0)
		state->buildMemoryProxyEncodingBytes = bytes;
	else if (strcmp(phase, "centroid_clustering") == 0)
		state->buildMemoryCentroidClusteringBytes = bytes;
	else if (strcmp(phase, "centroid_residual_computation") == 0)
		state->buildMemoryCentroidResidualBytes = bytes;
	else if (strcmp(phase, "sidecar_tuple_construction") == 0)
		state->buildMemorySidecarTupleBytes = bytes;
	else if (strcmp(phase, "posting_tuple_construction") == 0)
		state->buildMemoryPostingTupleBytes = bytes;
	else if (strcmp(phase, "graph_edge_build") == 0)
		state->buildMemoryGraphEdgeBytes = bytes;
	else if (strcmp(phase, "page_wal_write") == 0)
		state->buildMemoryPageWalBytes = bytes;

	if (bytes > state->buildPeakMemoryContextBytes)
	{
		state->buildPeakMemoryContextBytes = bytes;
		strlcpy(state->buildPeakMemoryPhase, phase,
				sizeof(state->buildPeakMemoryPhase));
	}
}

static uint16
PgturbohybridGraphMultiVectorDocMapCentroidPostingTupleMaxCount(void)
{
	Size		header = MAXALIGN(offsetof(PgturbohybridGraphMultiVectorDocMapCentroidPostingTupleData,
										   entries));
	Size		usable = PgturbohybridGraphDocMapMaxItemSize();
	Size		maxCount;

	if (header >= usable)
		return 1;
	maxCount = (usable - header) /
		sizeof(PgturbohybridGraphMultiVectorCentroidPostingEntry);
	return (uint16) Max(1, Min((Size) UINT16_MAX, maxCount));
}

static uint16
PgturbohybridGraphMultiVectorDocMapQuantizedPostingTupleMaxCount(void)
{
	Size		header = MAXALIGN(offsetof(PgturbohybridGraphMultiVectorDocMapQuantizedPostingTupleData,
										   entries));
	Size		usable = PgturbohybridGraphDocMapMaxItemSize();
	Size		maxCount;

	if (header >= usable)
		return 1;
	maxCount = (usable - header) /
		sizeof(PgturbohybridGraphMultiVectorQuantizedPostingEntry);
	return (uint16) Max(1, Min((Size) UINT16_MAX, maxCount));
}

static void
PgturbohybridGraphAddMultiVectorDocMapBytesEstimate(PgturbohybridQuantBuildState *state,
													uint64 bytes,
													const char *component)
{
	if (state == NULL || bytes == 0)
		return;
	if (state->multivectorDocMapBytesEstimate >
		(uint64) UINT32_MAX - bytes)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pgturbohybrid multivector docmap sidecar is too large"),
				 errdetail("Estimated document-node docmap sidecar bytes exceed the current 32-bit persisted format limit while adding %s.",
						   component != NULL ? component : "component"),
				 errhint("Use multivector_doc_storage = proxy_only for proxy_vector-only admission, multivector_doc_storage = centroid_only for centroid_lite heap-rerank admission, reduce the corpus or token count, or REINDEX after adding a versioned large-sidecar format.")));
	state->multivectorDocMapBytesEstimate += bytes;
}

static uint64
PgturbohybridGraphMultiVectorDocMapChunkedFloatTupleBytes(Size floatCount,
														  uint16 maxCount,
														  Size (*tupleSizer) (uint16))
{
	uint64		bytes = 0;

	for (Size startFloat = 0; startFloat < floatCount;)
	{
		uint16		count =
			(uint16) Min((Size) maxCount, floatCount - startFloat);
		Size		tupleSize = tupleSizer(count);

		if (bytes > PG_UINT64_MAX - (uint64) tupleSize)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("pgturbohybrid multivector docmap sidecar is too large")));
		bytes += (uint64) tupleSize;
		startFloat += count;
	}
	return bytes;
}

static uint64
PgturbohybridGraphMultiVectorDocMapChunkedVectorTupleBytes(Size floatCount,
														  int storageKind)
{
	if (storageKind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F16)
		return PgturbohybridGraphMultiVectorDocMapChunkedFloatTupleBytes(floatCount,
																		 PgturbohybridGraphMultiVectorDocMapVectorF16TupleMaxCount(),
																		 PgturbohybridGraphMultiVectorDocMapVectorF16TupleSize);
	if (storageKind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_SQ8)
		return PgturbohybridGraphMultiVectorDocMapChunkedFloatTupleBytes(floatCount,
																		 PgturbohybridGraphMultiVectorDocMapVectorSq8TupleMaxCount(),
																		 PgturbohybridGraphMultiVectorDocMapVectorSq8TupleSize);
	return PgturbohybridGraphMultiVectorDocMapChunkedFloatTupleBytes(floatCount,
																	 PgturbohybridGraphMultiVectorDocMapVectorTupleMaxCount(),
																	 PgturbohybridGraphMultiVectorDocMapVectorTupleSize);
}

static void
PgturbohybridGraphWriteMultiVectorDocMapItem(PgturbohybridQuantBuildState *state,
											 Buffer *buf, Page *page,
											 BlockNumber *start,
											 uint32 *pageCount,
											 Item tuple, Size tupleSize)
{
	if (!BufferIsValid(*buf) || PageGetFreeSpace(*page) < tupleSize)
	{
		PgturbohybridGraphAppendPage(state->index, state->forkNum, buf, page,
									 PGTURBOHYBRID_GRAPH_PAGE_KIND_MULTIVECTOR_DOCMAP);
		if (!BlockNumberIsValid(*start))
			*start = BufferGetBlockNumber(*buf);
		(*pageCount)++;
	}

	if (PageAddItem(*page, tuple, tupleSize, InvalidOffsetNumber,
					false, false) == InvalidOffsetNumber)
		ereport(ERROR,
				(errcode(ERRCODE_INDEX_CORRUPTED),
				 errmsg("failed to add pgturbohybrid multivector docmap item to \"%s\"",
						RelationGetRelationName(state->index)),
				 errdetail("tuple size %lu bytes, page free space %lu bytes, configured docmap max item size %lu bytes.",
						   (unsigned long) tupleSize,
						   (unsigned long) PageGetFreeSpace(*page),
						   (unsigned long) PgturbohybridGraphDocMapMaxItemSize())));
	PgturbohybridGraphMarkPageGraphOp(*page,
									  PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT);
}

static void
PgturbohybridGraphWriteMultiVectorCentroidPostingTuples(PgturbohybridQuantBuildState *state,
														Buffer *buf,
														Page *page,
														BlockNumber *start,
														uint32 *pageCount,
														uint32 *docMapBytes)
{
	uint32		codebookSize;
	uint32	   *listCounts;
	uint32	   *listOffsets;
	uint32	   *listWrite;
	PgturbohybridGraphMultiVectorCentroidPostingEntry *postings;
	uint32		totalPostings = 0;
	uint32		codebookTopM;
	uint32	   *centroidCodewords;
	PgturbohybridMultiVectorPostingScoreCandidate *codewordScratch;
	uint16		maxPostingCount;
	uint16		maxDocCodeCount;
	instr_time	startTime;

	if (state->multivectorCentroids !=
		PGTURBOHYBRID_MULTIVECTOR_CENTROIDS_KMEANS ||
		state->multivectorDocCentroids == NULL ||
		state->dimensions <= 0)
		return;

	INSTR_TIME_SET_CURRENT(startTime);
	codebookSize = (uint32) state->dimensions * 2U;
	codebookTopM =
		Min((uint32) Max(pgturbohybrid_multivector_centroid_lite_codeword_top_m,
						 1),
			codebookSize);
	listCounts = palloc0(sizeof(uint32) * (Size) codebookSize);
	listOffsets = palloc0(sizeof(uint32) * ((Size) codebookSize + 1));
	listWrite = palloc0(sizeof(uint32) * (Size) codebookSize);
	centroidCodewords = palloc(sizeof(uint32) * (Size) codebookTopM);
	codewordScratch =
		palloc0(sizeof(PgturbohybridMultiVectorPostingScoreCandidate) *
				(Size) codebookTopM);

	for (uint32 docId = 0; docId < state->multivectorDocCount; docId++)
	{
		PgturbohybridMultiVector *centroids =
			state->multivectorDocCentroids[docId];

		if (centroids == NULL)
			ereport(ERROR,
					(errcode(ERRCODE_INDEX_CORRUPTED),
					 errmsg("document-node centroid posting sidecar is missing document centroids")));
		if ((uint64) totalPostings +
			(uint64) centroids->count * (uint64) codebookTopM >
			(uint64) PG_UINT32_MAX)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("pgturbohybrid centroid posting sidecar is too large")));
		for (int32 centroid = 0; centroid < centroids->count; centroid++)
		{
			uint32		codewordCount =
				PgturbohybridMultiVectorDeterministicCodewordsWithScratch(centroids,
																		  centroid,
																		  codebookTopM,
																		  centroidCodewords,
																		  codewordScratch);

			for (uint32 codewordIndex = 0; codewordIndex < codewordCount;
				 codewordIndex++)
			{
				uint32		codeword = centroidCodewords[codewordIndex];

				if (codeword >= codebookSize)
					ereport(ERROR,
							(errcode(ERRCODE_INDEX_CORRUPTED),
							 errmsg("pgturbohybrid centroid posting codeword is out of range")));
				listCounts[codeword]++;
				totalPostings++;
			}
		}
	}
	if ((uint64) totalPostings >
		(uint64) (MaxAllocSize /
				  sizeof(PgturbohybridGraphMultiVectorCentroidPostingEntry)))
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pgturbohybrid centroid posting sidecar is too large")));
	state->multivectorCentroidPostingBytes =
		PgturbohybridGraphBuildMultiplyEstimate(sizeof(PgturbohybridGraphMultiVectorCentroidPostingEntry),
												(uint64) totalPostings);

	listOffsets[0] = 0;
	for (uint32 codeword = 0; codeword < codebookSize; codeword++)
	{
		listOffsets[codeword + 1] = listOffsets[codeword] + listCounts[codeword];
		listWrite[codeword] = listOffsets[codeword];
	}
	postings =
		palloc0(sizeof(PgturbohybridGraphMultiVectorCentroidPostingEntry) *
				(Size) Max(totalPostings, 1U));
	PgturbohybridGraphRecordBuildMemorySnapshot(state,
												"posting_tuple_construction");
	for (uint32 docId = 0; docId < state->multivectorDocCount; docId++)
	{
		PgturbohybridMultiVector *centroids =
			state->multivectorDocCentroids[docId];

		for (int32 centroid = 0; centroid < centroids->count; centroid++)
		{
			uint32		codewordCount =
				PgturbohybridMultiVectorDeterministicCodewordsWithScratch(centroids,
																		  centroid,
																		  codebookTopM,
																		  centroidCodewords,
																		  codewordScratch);

			for (uint32 codewordIndex = 0; codewordIndex < codewordCount;
				 codewordIndex++)
			{
				uint32		codeword = centroidCodewords[codewordIndex];
				uint32		offset = listWrite[codeword]++;

				postings[offset].docId = docId;
				postings[offset].centroidOrdinal = (uint16) centroid;
				postings[offset].scorePayload =
					PgturbohybridMultiVectorCentroidPostingCodewordScorePayload(centroids,
																				centroid,
																				codeword);
			}
		}
	}
	for (uint32 codeword = 0; codeword < codebookSize; codeword++)
	{
		uint32		startPosting = listOffsets[codeword];
		uint32		endPosting = listOffsets[codeword + 1];

		if (endPosting > startPosting + 1)
			qsort(&postings[startPosting],
				  endPosting - startPosting,
				  sizeof(PgturbohybridGraphMultiVectorCentroidPostingEntry),
				  PgturbohybridMultiVectorCentroidPostingPayloadCompare);
	}

	/*
	 * Persist a per-document centroid code sequence separately from the
	 * inverted lists.  Query-time centroid_lite uses the posting lists only to
	 * form a bounded candidate set, then computes approximate MaxSim over the
	 * candidate's full compact code sequence, matching the next-plaid/PLAID
	 * shape while leaving final SQL ordering to exact heap MaxSim.
	 */
	maxDocCodeCount =
		PgturbohybridGraphMultiVectorDocMapCentroidDocCodeTupleMaxCount();
	for (uint32 docId = 0; docId < state->multivectorDocCount; docId++)
	{
		PgturbohybridMultiVector *centroids =
			state->multivectorDocCentroids[docId];

		for (uint32 startCode = 0;
			 startCode < (uint32) centroids->count;)
		{
			uint16		count =
				(uint16) Min((uint32) maxDocCodeCount,
							 (uint32) centroids->count - startCode);
			Size		tupleSize =
				PgturbohybridGraphMultiVectorDocMapCentroidDocCodeTupleSize(count);
			PgturbohybridGraphMultiVectorDocMapCentroidDocCodeTuple tuple =
				palloc0(tupleSize);

			CHECK_FOR_INTERRUPTS();
			tuple->type =
				PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_CENTROID_DOC_CODE_TUPLE_TYPE;
			tuple->version = PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION;
			tuple->count = count;
			tuple->magic = PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC;
			tuple->docId = docId;
			tuple->codebookSize = codebookSize;
			tuple->startCode = startCode;
			for (uint16 i = 0; i < count; i++)
			{
				uint32		codewordCount =
					PgturbohybridMultiVectorDeterministicCodewordsWithScratch(centroids,
																			  (int32) startCode + i,
																			  1,
																			  centroidCodewords,
																			  codewordScratch);

				if (codewordCount == 0 ||
					centroidCodewords[0] >= codebookSize)
					ereport(ERROR,
							(errcode(ERRCODE_INDEX_CORRUPTED),
							 errmsg("pgturbohybrid centroid doc-code sidecar codeword is out of range")));
				tuple->codes[i] = centroidCodewords[0];
			}
			PgturbohybridGraphWriteMultiVectorDocMapItem(state, buf, page,
														 start, pageCount,
														 (Item) tuple,
														 tupleSize);
			if (tupleSize > UINT32_MAX - *docMapBytes)
				ereport(ERROR,
						(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
						 errmsg("pgturbohybrid multivector docmap sidecar is too large")));
			*docMapBytes += (uint32) tupleSize;
			pfree(tuple);
			startCode += count;
		}
	}

	maxPostingCount =
		PgturbohybridGraphMultiVectorDocMapCentroidPostingTupleMaxCount();
	for (uint32 codeword = 0; codeword < codebookSize; codeword++)
	{
		uint32		startPosting = listOffsets[codeword];
		uint32		endPosting = listOffsets[codeword + 1];

		for (uint32 postingOffset = startPosting; postingOffset < endPosting;)
		{
			uint16		count =
				(uint16) Min((uint32) maxPostingCount,
							 endPosting - postingOffset);
			Size		tupleSize =
				PgturbohybridGraphMultiVectorDocMapCentroidPostingTupleSize(count);
			PgturbohybridGraphMultiVectorDocMapCentroidPostingTuple tuple =
				palloc0(tupleSize);

			CHECK_FOR_INTERRUPTS();
			tuple->type =
				PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_CENTROID_POSTING_TUPLE_TYPE;
			tuple->version = PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION;
			tuple->count = count;
			tuple->magic = PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC;
			tuple->codeword = codeword;
			tuple->startOffset = postingOffset - startPosting;
			memcpy(tuple->entries, &postings[postingOffset],
				   sizeof(PgturbohybridGraphMultiVectorCentroidPostingEntry) *
				   count);
			PgturbohybridGraphWriteMultiVectorDocMapItem(state, buf, page,
														 start, pageCount,
														 (Item) tuple,
														 tupleSize);
			if (tupleSize > UINT32_MAX - *docMapBytes)
				ereport(ERROR,
						(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
						 errmsg("pgturbohybrid multivector docmap sidecar is too large")));
			*docMapBytes += (uint32) tupleSize;
			pfree(tuple);
			postingOffset += count;
		}
	}

	pfree(postings);
	pfree(codewordScratch);
	pfree(centroidCodewords);
	pfree(listWrite);
	pfree(listOffsets);
	pfree(listCounts);
	state->multivectorCentroidPostingWriteUs +=
		PgturbohybridGraphElapsedUs(startTime);
	state->multivectorCentroidPostingCount += (uint64) totalPostings;
}

static void
PgturbohybridGraphWriteMultiVectorQuantizedInvertedPostingTuples(PgturbohybridQuantBuildState *state,
																 Buffer *buf,
																 Page *page,
																 BlockNumber *start,
																 uint32 *pageCount,
																 uint32 *docMapBytes)
{
	uint32		codebookSize;
	uint32		codebookTopM = 1;
	int			codebookSource =
		pgturbohybrid_multivector_quantized_inverted_codebook;
	const char *codebookChecksum = "deterministic";
	uint32	   *listCounts;
	uint32	   *listOffsets;
	uint32	   *listWrite;
	PgturbohybridGraphMultiVectorQuantizedPostingEntry *postings;
	PgturbohybridGraphMultiVectorQuantizedBuildPostingList *buildPostingLists;
	bool		useBuildPostings;
	uint32		totalPostings = 0;
	uint16		maxPostingCount;
	Size		codebookTupleSize;
	PgturbohybridGraphMultiVectorDocMapQuantizedCodebookTuple codebookTuple;

	if (state->multivectorGraphMode !=
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES ||
		state->dimensions <= 0)
		return;
	useBuildPostings =
		state->multivectorQuantizedBuildPostingLists != NULL &&
		state->multivectorQuantizedBuildPostingCount > 0;
	if (!useBuildPostings && state->multivectorDocVectors == NULL)
		return;

	if (codebookSource ==
		PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_EXTERNAL)
	{
		PgturbohybridQuantizedInvertedCodebook *codebook =
			PgturbohybridLoadQuantizedInvertedCodebook(state->dimensions);

		codebookSize = codebook->codebookSize;
		codebookTopM = codebook->topM;
		codebookChecksum = codebook->checksum;
	}
	else
	{
		if (pgturbohybrid_multivector_quantized_inverted_codebook_top_m != 1)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("quantized_inverted_experimental deterministic codebook top_m values greater than 1 are not supported yet"),
					 errhint("Set turbohybrid.multivector_quantized_inverted_codebook_top_m = 1 until a versioned multi-posting sidecar format is available.")));
		codebookSize = (uint32) state->dimensions * 2U;
	}
	listCounts = palloc0(sizeof(uint32) * (Size) codebookSize);
	listOffsets = palloc0(sizeof(uint32) * ((Size) codebookSize + 1));
	listWrite = useBuildPostings ? NULL :
		palloc0(sizeof(uint32) * (Size) codebookSize);

	if (useBuildPostings)
	{
		buildPostingLists = state->multivectorQuantizedBuildPostingLists;
		totalPostings = state->multivectorQuantizedBuildPostingCount;
		for (uint32 codeword = 0; codeword < codebookSize; codeword++)
		{
			PgturbohybridGraphMultiVectorQuantizedBuildPostingList *list =
				&buildPostingLists[codeword];

			if (list->count > 0 && list->postings == NULL)
				ereport(ERROR,
						(errcode(ERRCODE_INDEX_CORRUPTED),
						 errmsg("quantized_inverted_experimental posting build list is invalid")));
			listCounts[codeword] = list->count;
			if (list->count > 1)
				qsort(list->postings,
					  list->count,
					  sizeof(PgturbohybridGraphMultiVectorQuantizedPostingEntry),
					  codebookSource ==
					  PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_EXTERNAL ?
					  PgturbohybridMultiVectorQuantizedPostingSignedPayloadCompare :
					  PgturbohybridMultiVectorQuantizedPostingPayloadCompare);
		}
	}
	else
	{
		uint32	   *assignedCodewords =
			palloc(sizeof(uint32) * (Size) Max(codebookTopM, 1U));

		buildPostingLists = NULL;
		for (uint32 docId = 0; docId < state->multivectorDocCount; docId++)
		{
			PgturbohybridMultiVector *doc =
				state->multivectorDocVectors[docId];

			if (doc == NULL)
				ereport(ERROR,
						(errcode(ERRCODE_INDEX_CORRUPTED),
						 errmsg("document-node quantized inverted sidecar is missing document vectors")));
			if ((uint64) totalPostings +
				(uint64) doc->count * (uint64) codebookTopM >
				(uint64) PG_UINT32_MAX)
				ereport(ERROR,
						(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
						 errmsg("pgturbohybrid quantized inverted sidecar is too large")));
			for (int32 token = 0; token < doc->count; token++)
			{
				uint32		codewordCount =
					PgturbohybridMultiVectorQuantizedInvertedConfigurableCodewords(doc,
																				   token,
																				   codebookTopM,
																				   assignedCodewords,
																				   NULL);

				if (codewordCount == 0)
					ereport(ERROR,
							(errcode(ERRCODE_INDEX_CORRUPTED),
							 errmsg("quantized_inverted_experimental build assigned no codeword")));
				for (uint32 codewordIndex = 0; codewordIndex < codewordCount;
					 codewordIndex++)
				{
					uint32		codeword = assignedCodewords[codewordIndex];

					if (codeword >= codebookSize)
						ereport(ERROR,
								(errcode(ERRCODE_INDEX_CORRUPTED),
								 errmsg("quantized_inverted_experimental build assigned an out-of-range codeword")));
					listCounts[codeword]++;
					totalPostings++;
				}
			}
		}
		pfree(assignedCodewords);
	}
	if ((uint64) totalPostings >
		(uint64) (MaxAllocSize /
				  sizeof(PgturbohybridGraphMultiVectorQuantizedPostingEntry)))
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pgturbohybrid quantized inverted sidecar is too large")));

	listOffsets[0] = 0;
	for (uint32 codeword = 0; codeword < codebookSize; codeword++)
	{
		listOffsets[codeword + 1] = listOffsets[codeword] + listCounts[codeword];
		if (listWrite != NULL)
			listWrite[codeword] = listOffsets[codeword];
	}
	if (useBuildPostings)
		postings = NULL;
	else
	{
		uint32	   *assignedCodewords =
			palloc(sizeof(uint32) * (Size) Max(codebookTopM, 1U));

		postings =
			palloc0(sizeof(PgturbohybridGraphMultiVectorQuantizedPostingEntry) *
					(Size) Max(totalPostings, 1U));
		for (uint32 docId = 0; docId < state->multivectorDocCount; docId++)
		{
			PgturbohybridMultiVector *doc =
				state->multivectorDocVectors[docId];

			for (int32 token = 0; token < doc->count; token++)
			{
				uint32		codewordCount =
					PgturbohybridMultiVectorQuantizedInvertedConfigurableCodewords(doc,
																				   token,
																				   codebookTopM,
																				   assignedCodewords,
																				   NULL);

				if (codewordCount == 0)
					ereport(ERROR,
							(errcode(ERRCODE_INDEX_CORRUPTED),
							 errmsg("quantized_inverted_experimental build assigned no codeword")));
				for (uint32 codewordIndex = 0; codewordIndex < codewordCount;
					 codewordIndex++)
				{
					uint32		codeword = assignedCodewords[codewordIndex];
					uint32		offset;
					double		score;

					if (codeword >= codebookSize)
						ereport(ERROR,
								(errcode(ERRCODE_INDEX_CORRUPTED),
								 errmsg("quantized_inverted_experimental build assigned an out-of-range codeword")));
					Assert(listWrite != NULL);
					offset = listWrite[codeword]++;
					postings[offset].docId = docId;
					postings[offset].tokenOrdinal = (uint16) token;
					score =
						codebookSource ==
						PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_EXTERNAL ?
						PgturbohybridMultiVectorQuantizedInvertedCodewordScore(doc,
																			   token,
																			   codeword) :
						PgturbohybridMultiVectorDeterministicCodewordScore(doc,
																		   token,
																		   codeword);
					postings[offset].scorePayload =
						codebookSource ==
						PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_EXTERNAL ?
						(uint16) PgturbohybridMultiVectorQuantizedInvertedCompactScorePayload(score) :
						PgturbohybridMultiVectorQuantizedInvertedScorePayload(doc,
																			  token);
				}
			}
		}
		pfree(assignedCodewords);
		for (uint32 codeword = 0; codeword < codebookSize; codeword++)
		{
			uint32		startPosting = listOffsets[codeword];
			uint32		endPosting = listOffsets[codeword + 1];

			if (endPosting > startPosting + 1)
				qsort(&postings[startPosting],
					  endPosting - startPosting,
					  sizeof(PgturbohybridGraphMultiVectorQuantizedPostingEntry),
					  codebookSource ==
					  PGTURBOHYBRID_MULTIVECTOR_QUANTIZED_INVERTED_CODEBOOK_EXTERNAL ?
					  PgturbohybridMultiVectorQuantizedPostingSignedPayloadCompare :
					  PgturbohybridMultiVectorQuantizedPostingPayloadCompare);
		}
	}

	codebookTupleSize =
		MAXALIGN(sizeof(PgturbohybridGraphMultiVectorDocMapQuantizedCodebookTupleData));
	codebookTuple = palloc0(codebookTupleSize);
	codebookTuple->type =
		PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_QUANTIZED_CODEBOOK_TUPLE_TYPE;
	codebookTuple->version = PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION;
	codebookTuple->source = (uint16) codebookSource;
	codebookTuple->magic = PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC;
	codebookTuple->dim = (uint32) state->dimensions;
	codebookTuple->codebookSize = codebookSize;
	codebookTuple->topM = codebookTopM;
	strlcpy(codebookTuple->checksum, codebookChecksum,
			sizeof(codebookTuple->checksum));
	PgturbohybridGraphWriteMultiVectorDocMapItem(state, buf, page,
												 start, pageCount,
												 (Item) codebookTuple,
												 codebookTupleSize);
	if (codebookTupleSize > UINT32_MAX - *docMapBytes)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pgturbohybrid multivector docmap sidecar is too large")));
	*docMapBytes += (uint32) codebookTupleSize;
	pfree(codebookTuple);

	maxPostingCount =
		PgturbohybridGraphMultiVectorDocMapQuantizedPostingTupleMaxCount();
	for (uint32 codeword = 0; codeword < codebookSize; codeword++)
	{
		uint32		startPosting = listOffsets[codeword];
		uint32		endPosting = listOffsets[codeword + 1];

		for (uint32 postingOffset = startPosting; postingOffset < endPosting;)
		{
			uint16		count =
				(uint16) Min((uint32) maxPostingCount,
							 endPosting - postingOffset);
			Size		tupleSize =
				PgturbohybridGraphMultiVectorDocMapQuantizedPostingTupleSize(count);
			PgturbohybridGraphMultiVectorDocMapQuantizedPostingTuple tuple =
				palloc0(tupleSize);

			CHECK_FOR_INTERRUPTS();
			tuple->type =
				PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_QUANTIZED_POSTING_TUPLE_TYPE;
			tuple->version = PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION;
			tuple->count = count;
			tuple->magic = PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC;
			tuple->codeword = codeword;
			tuple->startOffset = postingOffset - startPosting;
			if (useBuildPostings)
			{
				PgturbohybridGraphMultiVectorQuantizedBuildPostingList *list =
					&buildPostingLists[codeword];
				uint32		listOffset = postingOffset - startPosting;

				for (uint16 i = 0; i < count; i++)
					tuple->entries[i] = list->postings[listOffset + i];
			}
			else
				memcpy(tuple->entries, &postings[postingOffset],
					   sizeof(PgturbohybridGraphMultiVectorQuantizedPostingEntry) *
					   count);
			PgturbohybridGraphWriteMultiVectorDocMapItem(state, buf, page,
														 start, pageCount,
														 (Item) tuple,
														 tupleSize);
			if (tupleSize > UINT32_MAX - *docMapBytes)
				ereport(ERROR,
						(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
						 errmsg("pgturbohybrid multivector docmap sidecar is too large")));
			*docMapBytes += (uint32) tupleSize;
			pfree(tuple);
			postingOffset += count;
		}
	}

	if (postings != NULL)
		pfree(postings);
	if (listWrite != NULL)
		pfree(listWrite);
	pfree(listOffsets);
	pfree(listCounts);
}

static BlockNumber
PgturbohybridGraphWriteMultiVectorDocMapPages(PgturbohybridQuantBuildState *state,
											  uint32 *pageCount,
											  uint32 *docMapBytes)
{
	Buffer		buf = InvalidBuffer;
	Page		page = NULL;
	BlockNumber start = InvalidBlockNumber;
	uint16		maxNodeCount;
	uint16		maxDocCount;
	instr_time	sidecarStart;

	*pageCount = 0;
	*docMapBytes = 0;
	if (!state->multivectorBuild ||
		state->multivectorNodeMap == NULL ||
		state->multivectorDocMap == NULL ||
		state->nodeCount == 0 ||
		state->multivectorDocCount == 0)
		return InvalidBlockNumber;

	INSTR_TIME_SET_CURRENT(sidecarStart);
	maxNodeCount = PgturbohybridGraphMultiVectorDocMapNodeTupleMaxCount();
	maxDocCount = PgturbohybridGraphMultiVectorDocMapDocTupleMaxCount();

	for (uint32 firstNodeId = 0; firstNodeId < state->nodeCount;)
	{
		uint16		count =
			(uint16) Min((uint32) maxNodeCount,
						 state->nodeCount - firstNodeId);
		Size		tupleSize =
			PgturbohybridGraphMultiVectorDocMapNodeTupleSize(count);
		PgturbohybridGraphMultiVectorDocMapNodeTuple tuple =
			palloc0(tupleSize);

		CHECK_FOR_INTERRUPTS();
		tuple->type =
			PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_NODE_TUPLE_TYPE;
		tuple->version = PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION;
		tuple->count = count;
		tuple->magic = PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC;
		tuple->firstNodeId = firstNodeId;
		memcpy(tuple->entries, &state->multivectorNodeMap[firstNodeId],
			   sizeof(TqMultiVectorNodeMapEntry) * count);
		PgturbohybridGraphWriteMultiVectorDocMapItem(state, &buf, &page,
													 &start, pageCount,
													 (Item) tuple, tupleSize);
		if (tupleSize > UINT32_MAX - *docMapBytes)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("pgturbohybrid multivector docmap sidecar is too large")));
		*docMapBytes += (uint32) tupleSize;
		pfree(tuple);
		firstNodeId += count;
	}

	for (uint32 firstDocId = 0; firstDocId < state->multivectorDocCount;)
	{
		uint16		count =
			(uint16) Min((uint32) maxDocCount,
						 state->multivectorDocCount - firstDocId);
		Size		tupleSize =
			PgturbohybridGraphMultiVectorDocMapDocTupleSize(count);
		PgturbohybridGraphMultiVectorDocMapDocTuple tuple =
			palloc0(tupleSize);

		CHECK_FOR_INTERRUPTS();
		tuple->type =
			PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_DOC_TUPLE_TYPE;
		tuple->version = PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION;
		tuple->count = count;
		tuple->magic = PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC;
		tuple->firstDocId = firstDocId;
		memcpy(tuple->entries, &state->multivectorDocMap[firstDocId],
			   sizeof(TqMultiVectorDocMapEntry) * count);
		PgturbohybridGraphWriteMultiVectorDocMapItem(state, &buf, &page,
													 &start, pageCount,
													 (Item) tuple, tupleSize);
		if (tupleSize > UINT32_MAX - *docMapBytes)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("pgturbohybrid multivector docmap sidecar is too large")));
		*docMapBytes += (uint32) tupleSize;
		pfree(tuple);
		firstDocId += count;
	}

		if (state->multivectorGraphMode ==
			PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES &&
			state->multivectorDocStorage !=
			PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_PROXY_ONLY)
		{
			uint16		maxVectorCount =
				PgturbohybridGraphMultiVectorDocMapVectorTupleMaxCount();
			uint16		maxVectorF16Count =
				PgturbohybridGraphMultiVectorDocMapVectorF16TupleMaxCount();
			uint16		maxVectorSq8Count =
				PgturbohybridGraphMultiVectorDocMapVectorSq8TupleMaxCount();
			uint16		maxCentroidCount =
				PgturbohybridGraphMultiVectorDocMapCentroidTupleMaxCount();
			uint16		maxCentroidF16Count =
				PgturbohybridGraphMultiVectorDocMapCentroidF16TupleMaxCount();
			bool		writeDocVectors =
				PgturbohybridGraphMultiVectorDocMapStoresDocVectors(state);
			bool		writeCentroidF16 =
				state->multivectorDocStorage ==
				PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CENTROID_ONLY;

			for (uint32 docId = 0; docId < state->multivectorDocCount; docId++)
			{
				PgturbohybridMultiVector *mv = state->multivectorDocVectors != NULL ?
					state->multivectorDocVectors[docId] : NULL;
				Size		totalFloats;
				float		docSq8Scale = 1.0f;

				if (writeDocVectors)
				{
					if (mv == NULL)
						ereport(ERROR,
								(errcode(ERRCODE_INDEX_CORRUPTED),
								 errmsg("document-node multivector sidecar is missing document vectors")));
					totalFloats =
						PgturbohybridMultiVectorFloatCount(mv->count, mv->dim);
					if (state->multivectorDocStorage ==
						PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_SQ8)
						docSq8Scale =
							PgturbohybridMultiVectorDocCompactSq8Scale(mv);

					for (uint32 startFloat = 0; startFloat < totalFloats;)
					{
						uint16		count =
							(uint16) Min((Size)
										 (state->multivectorDocStorage ==
										  PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F16 ?
										  maxVectorF16Count :
										  state->multivectorDocStorage ==
										  PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_SQ8 ?
										  maxVectorSq8Count : maxVectorCount),
										 totalFloats - (Size) startFloat);
						Size		tupleSize;
						Item		tuple;

						CHECK_FOR_INTERRUPTS();
						if (state->multivectorDocStorage ==
							PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F16)
						{
							PgturbohybridGraphMultiVectorDocMapVectorF16Tuple f16Tuple;

							tupleSize =
								PgturbohybridGraphMultiVectorDocMapVectorF16TupleSize(count);
							f16Tuple = palloc0(tupleSize);
							f16Tuple->type =
								PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VECTOR_F16_TUPLE_TYPE;
							f16Tuple->version =
								PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION;
							f16Tuple->count = count;
							f16Tuple->magic =
								PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC;
							f16Tuple->docId = docId;
							f16Tuple->startFloat = startFloat;
							for (uint16 i = 0; i < count; i++)
								f16Tuple->values[i] =
									PgturbohybridMultiVectorFloatToHalf(mv->values[startFloat + i]);
							tuple = (Item) f16Tuple;
						}
						else if (state->multivectorDocStorage ==
								 PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_SQ8)
						{
							PgturbohybridGraphMultiVectorDocMapVectorSq8Tuple sq8Tuple;

							tupleSize =
								PgturbohybridGraphMultiVectorDocMapVectorSq8TupleSize(count);
							sq8Tuple = palloc0(tupleSize);
							sq8Tuple->type =
								PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VECTOR_SQ8_TUPLE_TYPE;
							sq8Tuple->version =
								PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION;
							sq8Tuple->count = count;
							sq8Tuple->magic =
								PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC;
							sq8Tuple->docId = docId;
							sq8Tuple->startFloat = startFloat;
							sq8Tuple->scale = docSq8Scale;
							for (uint16 i = 0; i < count; i++)
							{
								int			quantized =
									(int) lrintf(mv->values[startFloat + i] / docSq8Scale);

								quantized = Max(-127, Min(127, quantized));
								sq8Tuple->values[i] = (int8) quantized;
							}
							tuple = (Item) sq8Tuple;
						}
						else
						{
							PgturbohybridGraphMultiVectorDocMapVectorTuple f32Tuple;

							tupleSize =
								PgturbohybridGraphMultiVectorDocMapVectorTupleSize(count);
							f32Tuple = palloc0(tupleSize);
							f32Tuple->type =
								PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VECTOR_TUPLE_TYPE;
							f32Tuple->version =
								PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION;
							f32Tuple->count = count;
							f32Tuple->magic =
								PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC;
							f32Tuple->docId = docId;
							f32Tuple->startFloat = startFloat;
							memcpy(f32Tuple->values, mv->values + startFloat,
								   sizeof(float) * count);
							tuple = (Item) f32Tuple;
						}
						PgturbohybridGraphWriteMultiVectorDocMapItem(state, &buf,
																	 &page, &start,
																	 pageCount,
																	 tuple,
																	 tupleSize);
						if (tupleSize > UINT32_MAX - *docMapBytes)
							ereport(ERROR,
									(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
									 errmsg("pgturbohybrid multivector docmap sidecar is too large")));
						*docMapBytes += (uint32) tupleSize;
						pfree(tuple);
						startFloat += count;
					}
				if (PgturbohybridMultiVectorHasContexts(mv))
				{
					int32		contextCount =
						PgturbohybridMultiVectorContextCount(mv);
					const int32 *offsets =
						PgturbohybridMultiVectorContextOffsets(mv);
					const int32 *fields =
						PgturbohybridMultiVectorContextFields(mv);
					bool		hasFields = fields != NULL;
					Size		tupleSize =
						PgturbohybridGraphMultiVectorDocMapContextTupleSize((uint16) contextCount,
																			hasFields);
					PgturbohybridGraphMultiVectorDocMapContextTuple tuple =
						palloc0(tupleSize);

					tuple->type =
						PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_CONTEXT_TUPLE_TYPE;
					tuple->version = PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION;
					tuple->contextCount = (uint16) contextCount;
					tuple->magic = PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC;
					tuple->docId = docId;
					tuple->flags = mv->flags;
					memcpy(tuple->values, offsets,
						   sizeof(int32) * (Size) contextCount);
					if (hasFields)
						memcpy(tuple->values + contextCount, fields,
							   sizeof(int32) * (Size) contextCount);
					PgturbohybridGraphWriteMultiVectorDocMapItem(state, &buf,
																 &page, &start,
																 pageCount,
																 (Item) tuple,
																 tupleSize);
					if (tupleSize > UINT32_MAX - *docMapBytes)
						ereport(ERROR,
								(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
								 errmsg("pgturbohybrid multivector docmap sidecar is too large")));
					*docMapBytes += (uint32) tupleSize;
					pfree(tuple);
				}
			}
			if (state->multivectorCentroids ==
				PGTURBOHYBRID_MULTIVECTOR_CENTROIDS_KMEANS)
			{
				PgturbohybridMultiVector *centroids =
					state->multivectorDocCentroids != NULL ?
					state->multivectorDocCentroids[docId] : NULL;
				Size		centroidFloats;
				instr_time	centroidWriteStart;

				if (centroids == NULL)
					ereport(ERROR,
							(errcode(ERRCODE_INDEX_CORRUPTED),
							 errmsg("document-node centroid sidecar is missing document centroids")));
				if (centroids->dim != state->dimensions)
					ereport(ERROR,
							(errcode(ERRCODE_INDEX_CORRUPTED),
							 errmsg("document-node centroid sidecar has inconsistent dimensions")));
				if (mv != NULL)
					PgturbohybridCheckSameMultiVectorDims(mv, centroids);
				centroidFloats =
					PgturbohybridMultiVectorFloatCount(centroids->count,
													   centroids->dim);
				INSTR_TIME_SET_CURRENT(centroidWriteStart);
				for (uint32 startFloat = 0; startFloat < centroidFloats;)
				{
					uint16		count =
						(uint16) Min((Size) (writeCentroidF16 ?
											 maxCentroidF16Count :
											 maxCentroidCount),
									 centroidFloats - (Size) startFloat);
					Size		tupleSize;

					CHECK_FOR_INTERRUPTS();
					if (writeCentroidF16)
					{
						PgturbohybridGraphMultiVectorDocMapCentroidF16Tuple tuple;

						tupleSize =
							PgturbohybridGraphMultiVectorDocMapCentroidF16TupleSize(count);
						tuple = palloc0(tupleSize);
						tuple->type =
							PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_CENTROID_F16_TUPLE_TYPE;
						tuple->version =
							PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION;
						tuple->count = count;
						tuple->magic = PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC;
						tuple->docId = docId;
						tuple->centroidCount = (uint16) centroids->count;
						tuple->flags = 0;
						tuple->startFloat = startFloat;
						tuple->residualMean =
							state->multivectorDocCentroidResiduals != NULL ?
							state->multivectorDocCentroidResiduals[docId] :
							0.0f;
						for (uint16 i = 0; i < count; i++)
							tuple->values[i] =
								PgturbohybridMultiVectorFloatToHalf(centroids->values[startFloat + i]);
						PgturbohybridGraphWriteMultiVectorDocMapItem(state,
																	 &buf,
																	 &page,
																	 &start,
																	 pageCount,
																	 (Item) tuple,
																	 tupleSize);
						pfree(tuple);
					}
					else
					{
						PgturbohybridGraphMultiVectorDocMapCentroidTuple tuple;

						tupleSize =
							PgturbohybridGraphMultiVectorDocMapCentroidTupleSize(count);
						tuple = palloc0(tupleSize);
						tuple->type =
							PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_CENTROID_TUPLE_TYPE;
						tuple->version =
							PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION;
						tuple->count = count;
						tuple->magic = PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC;
						tuple->docId = docId;
						tuple->centroidCount = (uint16) centroids->count;
						tuple->flags = 0;
						tuple->startFloat = startFloat;
						tuple->residualMean =
							state->multivectorDocCentroidResiduals != NULL ?
							state->multivectorDocCentroidResiduals[docId] :
							0.0f;
						memcpy(tuple->values, centroids->values + startFloat,
							   sizeof(float) * count);
						PgturbohybridGraphWriteMultiVectorDocMapItem(state,
																	 &buf,
																	 &page,
																	 &start,
																	 pageCount,
																	 (Item) tuple,
																	 tupleSize);
						pfree(tuple);
					}
					if (tupleSize > UINT32_MAX - *docMapBytes)
						ereport(ERROR,
								(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
								 errmsg("pgturbohybrid multivector docmap sidecar is too large")));
					*docMapBytes += (uint32) tupleSize;
					startFloat += count;
				}
				state->multivectorCentroidSidecarWriteUs +=
					PgturbohybridGraphElapsedUs(centroidWriteStart);
			}
		}
		if (state->multivectorCentroids ==
			PGTURBOHYBRID_MULTIVECTOR_CENTROIDS_KMEANS)
			PgturbohybridGraphWriteMultiVectorCentroidPostingTuples(state,
																	&buf,
																	&page,
																	&start,
																	pageCount,
																	docMapBytes);
		if (PgturbohybridGraphMultiVectorDocMapStoresQuantizedInvertedPostings(state))
			PgturbohybridGraphWriteMultiVectorQuantizedInvertedPostingTuples(state,
																			 &buf,
																			 &page,
																			 &start,
																			 pageCount,
																			 docMapBytes);
	}

	if (BufferIsValid(buf))
		PgturbohybridGraphFinishPage(buf);

	state->multivectorDocSidecarWriteUs +=
		PgturbohybridGraphElapsedUs(sidecarStart);
	return start;
}

static bool
PgturbohybridGraphMultiVectorDocMapHasContexts(PgturbohybridQuantBuildState *state)
{
	if (state == NULL || state->multivectorDocVectors == NULL)
		return false;
	for (uint32 docId = 0; docId < state->multivectorDocCount; docId++)
	{
		PgturbohybridMultiVector *mv = state->multivectorDocVectors[docId];

		if (mv != NULL && PgturbohybridMultiVectorHasContexts(mv))
			return true;
	}
	return false;
}

static void
PgturbohybridGraphWriteGraphDataPages(PgturbohybridQuantBuildState *state, BlockNumber *codeStart,
						   BlockNumber *adjStart, BlockNumber *exactStart,
						   BlockNumber *correctionStart,
						   BlockNumber *docMapStart,
						   uint32 *docMapPageCount,
						   uint32 *docMapBytes)
{
	if (state->nodeCount == 0)
	{
		*codeStart = InvalidBlockNumber;
		*adjStart = InvalidBlockNumber;
		*exactStart = InvalidBlockNumber;
		*correctionStart = InvalidBlockNumber;
		*docMapStart = InvalidBlockNumber;
		*docMapPageCount = 0;
		*docMapBytes = 0;
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
	*docMapStart =
		PgturbohybridGraphWriteMultiVectorDocMapPages(state,
													  docMapPageCount,
													  docMapBytes);
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
	BlockNumber docMapStart;
	uint32		docMapPageCount;
	uint32		docMapBytes;
	instr_time	writeStart;

	INSTR_TIME_SET_CURRENT(writeStart);
	PgturbohybridGraphWriteGraphDataPages(state, &codeStart, &adjStart, &exactStart,
							   &correctionStart, &docMapStart,
							   &docMapPageCount, &docMapBytes);
	state->buildWriteUs = PgturbohybridGraphElapsedUs(writeStart);
	PgturbohybridQuantUpdateMetaPage(state->index, state, codeStart, adjStart, exactStart,
						  correctionStart, docMapStart, docMapPageCount,
						  docMapBytes);
}

static void
PgturbohybridGraphWriteIndex(PgturbohybridQuantBuildState *state)
{
	PgturbohybridGraphCreateMetaPage(state->index, state->forkNum);
	PgturbohybridGraphWriteGraphPages(state);
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

static int
PgturbohybridNativeBuildWorkerRequest(Relation heap, Relation index)
{
	char	   *setting = pgturbohybrid_native_build_workers;
	char	   *endptr;
	long		value;
	int			parallelWorkers;

	if (heap == NULL || setting == NULL)
		return 0;
	if (pg_strcasecmp(setting, "auto") == 0)
	{
		parallelWorkers = plan_create_index_workers(RelationGetRelid(heap),
													RelationGetRelid(index));
		if (parallelWorkers <= 0)
			return 0;
		return Min(parallelWorkers, max_parallel_maintenance_workers);
	}

	errno = 0;
	value = strtol(setting, &endptr, 10);
	if (errno != 0 || endptr == setting || *endptr != '\0' || value < 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid value for turbohybrid.native_build_workers"),
				 errdetail("Use \"auto\", 0, 1, 2, 4, or 8.")));
	if (!(value == 0 || value == 1 || value == 2 || value == 4 || value == 8))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid value for turbohybrid.native_build_workers"),
				 errdetail("Use \"auto\", 0, 1, 2, 4, or 8.")));
	if (value == 0)
		return 0;

	return Min((int) Min(value, (long) INT_MAX), max_parallel_maintenance_workers);
}

static bool
PgturbohybridNativeParallelEligible(PgturbohybridQuantBuildState *state,
									bool needsCorrectionFit)
{
	int			dimensions;

	if (state->heap == NULL || !state->buildCodeOnly)
		return false;
	if (state->tqQuantileFit && needsCorrectionFit)
		return false;
	dimensions = TupleDescAttr(state->index->rd_att, 0)->atttypmod;
	if (dimensions < 0 || dimensions > state->typeInfo->maxDimensions)
		return false;
	state->dimensions = dimensions;
	return true;
}

static Size
PgturbohybridNativeParallelRecordBytes(PgturbohybridQuantBuildState *state)
{
	Size		bytes = MAXALIGN(sizeof(PgturbohybridNativeParallelRecord));

	bytes = add_size(bytes,
					 PgturbohybridGraphCodeBytesForBits(state->dimensions,
														state->tqBits));
	bytes = MAXALIGN(bytes);
	bytes = add_size(bytes, MAXALIGN((Size) state->residualRerankBytes));
	return MAXALIGN(bytes);
}

static Size
PgturbohybridNativeParallelEstimateShared(PgturbohybridQuantBuildState *state,
										  Snapshot snapshot, int participantCapacity,
										  int phase, uint32 recordCapacity,
										  Size recordBytes)
{
	Size		bytes;
	Size		participants = (Size) participantCapacity;
	Size		dimensions = (Size) state->dimensions;
	Size		dimSlots = mul_size(participants, dimensions);

	if (phase == PGTURBOHYBRID_NATIVE_PARALLEL_EDGES)
	{
		uint16		edgeSegmentCount = (uint16) recordCapacity;
		Size		levelCapacity = (Size) PgturbohybridGraphLevelCapacity(state->m);
		Size		edgeMaxNeighbors = (Size) PgturbohybridGraphLevelM(state->m, 0) + 1;
		Size		nodeCount = (Size) state->nodeCount;
		Size		nodeLevelSlots = mul_size(nodeCount, levelCapacity);
		Size		codeBytes =
			PgturbohybridGraphCodeBytesForBits(state->dimensions, state->tqBits);

		bytes = BUFFERALIGN(sizeof(PgturbohybridNativeParallelShared));
		bytes = add_size(bytes, MAXALIGN(mul_size(sizeof(uint64), participants)));
		if (state->ecShift != NULL && state->ecScale != NULL)
		{
			bytes = add_size(bytes, MAXALIGN(mul_size(sizeof(float), dimensions)));
			bytes = add_size(bytes, MAXALIGN(mul_size(sizeof(float), dimensions)));
		}
		bytes = add_size(bytes, MAXALIGN(mul_size(sizeof(PgturbohybridNativeParallelEdgeNode),
												  nodeCount)));
		bytes = add_size(bytes, MAXALIGN(mul_size(codeBytes, nodeCount)));
		if (state->residualRerankBytes > 0)
			bytes = add_size(bytes,
							 MAXALIGN(mul_size((Size) state->residualRerankBytes,
											   nodeCount)));
		bytes = add_size(bytes, MAXALIGN(mul_size(sizeof(int),
												  nodeLevelSlots)));
		bytes = add_size(bytes, MAXALIGN(mul_size(sizeof(uint32),
												  mul_size(nodeLevelSlots,
														   edgeMaxNeighbors))));
		bytes = add_size(bytes, MAXALIGN(mul_size(sizeof(double),
												  mul_size(nodeLevelSlots,
														   edgeMaxNeighbors))));
		bytes = add_size(bytes,
						 MAXALIGN(mul_size(sizeof(PgturbohybridGraphSegmentMetaData),
										   (Size) edgeSegmentCount)));
		bytes = add_size(bytes, MAXALIGN(mul_size(sizeof(uint32), nodeCount)));
		bytes = add_size(bytes, MAXALIGN(mul_size(sizeof(bool), nodeCount)));
		return bytes;
	}

	bytes = add_size(BUFFERALIGN(sizeof(PgturbohybridNativeParallelShared)),
					 table_parallelscan_estimate(state->heap, snapshot));
	bytes = MAXALIGN(bytes);
	bytes = add_size(bytes, MAXALIGN(mul_size(sizeof(uint64), participants)));
	bytes = add_size(bytes, MAXALIGN(mul_size(sizeof(uint64), participants)));
	bytes = add_size(bytes, MAXALIGN(mul_size(sizeof(uint64), participants)));
	if (state->scoreMode == PGTURBOHYBRID_SCORE_COSINE ||
		state->scoreMode == PGTURBOHYBRID_SCORE_IP)
	{
		bytes = add_size(bytes, MAXALIGN(mul_size(sizeof(double), dimSlots)));
		bytes = add_size(bytes, MAXALIGN(mul_size(sizeof(double), dimSlots)));
	}
	if (phase == PGTURBOHYBRID_NATIVE_PARALLEL_ENCODE)
	{
		if (state->ecShift != NULL && state->ecScale != NULL)
		{
			bytes = add_size(bytes, MAXALIGN(mul_size(sizeof(float), dimensions)));
			bytes = add_size(bytes, MAXALIGN(mul_size(sizeof(float), dimensions)));
		}
		bytes = add_size(bytes, mul_size(recordBytes, (Size) recordCapacity));
	}
	return bytes;
}

static void
PgturbohybridNativeParallelInitShared(PgturbohybridQuantBuildState *state,
									  PgturbohybridNativeParallelShared *shared,
									  Snapshot snapshot, int phase,
									  int participantCapacity,
									  uint32 recordCapacity, Size recordBytes,
									  Size sharedBytes)
{
	Size		cursor;
	Size		participants = (Size) participantCapacity;
	Size		dimensions = (Size) state->dimensions;
	Size		participantUsBytes = MAXALIGN(mul_size(sizeof(uint64),
													 participants));
	Size		headerBytes = BUFFERALIGN(sizeof(PgturbohybridNativeParallelShared));

	memset(shared, 0, sharedBytes);
	shared->sharedBytes = sharedBytes;
	shared->heaprelid = state->heap != NULL ? RelationGetRelid(state->heap) :
		InvalidOid;
	shared->indexrelid = RelationGetRelid(state->index);
	shared->isconcurrent = state->indexInfo != NULL &&
		state->indexInfo->ii_Concurrent;
	shared->phase = phase;
	shared->participantCapacity = participantCapacity;
	shared->dimensions = state->dimensions;
	shared->m = state->m;
	shared->efConstruction = state->efConstruction;
	shared->tqBits = state->tqBits;
	shared->scoreMode = state->scoreMode;
	shared->tqWeighted = state->tqWeighted;
	shared->tqRenorm = state->tqRenorm;
	shared->buildCodeOnly = state->buildCodeOnly;
	shared->buildFastEdges = state->buildFastEdges;
	shared->payloadCount = state->payloadCount;
	shared->payloadBytes = state->payloadBytes;
	shared->residualRerankBytes = state->residualRerankBytes;
	shared->recordCapacity = recordCapacity;
	shared->recordBytes = recordBytes;
	shared->recordsBytes = phase == PGTURBOHYBRID_NATIVE_PARALLEL_ENCODE ?
		mul_size(recordBytes, (Size) recordCapacity) : 0;
	ConditionVariableInit(&shared->workersdonecv);
	SpinLockInit(&shared->mutex);

	if (phase == PGTURBOHYBRID_NATIVE_PARALLEL_EDGES)
	{
		Size		levelSlots;
		Size		neighborSlots;
		Size		correctionBytes = mul_size(sizeof(float), dimensions);
		PgturbohybridGraphBuildOrderItem *orderItems;
		uint32	   *order;

		cursor = headerBytes;
		shared->edgeNodeCount = state->nodeCount;
		shared->edgeSegmentCount = (uint16) recordCapacity;
		shared->edgeFinalSegmentCount = 1;
		shared->edgeLevelCapacity = PgturbohybridGraphLevelCapacity(state->m);
		shared->edgeMaxNeighbors = PgturbohybridGraphLevelM(state->m, 0) + 1;
		shared->edgeCodeBytes =
			PgturbohybridGraphCodeBytesForBits(state->dimensions,
											   state->tqBits);
		shared->edgeEntryNodeId = UINT_MAX;
		shared->edgeEntryLevel = -1;
		shared->edgeWarmupCount =
			Min((uint32) PGTURBOHYBRID_GRAPH_PARALLEL_EDGE_WARMUP,
				state->nodeCount);
		shared->edgeBatchSize =
			Min((uint32) PGTURBOHYBRID_GRAPH_PARALLEL_EDGE_BATCH_MAX,
				Max((uint32) participantCapacity *
					(uint32) PGTURBOHYBRID_GRAPH_PARALLEL_EDGE_BATCH_PER_PARTICIPANT,
					(uint32) participantCapacity));
		shared->edgeWorkerUsOffset = cursor;
		cursor = add_size(cursor, participantUsBytes);
		if (state->ecShift != NULL && state->ecScale != NULL)
		{
			shared->ecShiftOffset = cursor;
			memcpy(PgturbohybridNativeParallelEcShift(shared), state->ecShift,
				   correctionBytes);
			cursor = add_size(cursor, MAXALIGN(correctionBytes));
			shared->ecScaleOffset = cursor;
			memcpy(PgturbohybridNativeParallelEcScale(shared), state->ecScale,
				   correctionBytes);
			cursor = add_size(cursor, MAXALIGN(correctionBytes));
		}
		shared->edgeNodeOffset = cursor;
		cursor = add_size(cursor,
						  MAXALIGN(mul_size(sizeof(PgturbohybridNativeParallelEdgeNode),
											(Size) state->nodeCount)));
		shared->edgeCodeOffset = cursor;
		cursor = add_size(cursor,
						  MAXALIGN(mul_size(shared->edgeCodeBytes,
											(Size) state->nodeCount)));
		if (state->residualRerankBytes > 0)
		{
			shared->edgeResidualOffset = cursor;
			cursor = add_size(cursor,
							  MAXALIGN(mul_size((Size) state->residualRerankBytes,
												(Size) state->nodeCount)));
		}
		levelSlots = mul_size((Size) state->nodeCount,
							  (Size) shared->edgeLevelCapacity);
		shared->edgeNeighborCountOffset = cursor;
		cursor = add_size(cursor, MAXALIGN(mul_size(sizeof(int), levelSlots)));
		shared->edgeNeighborOffset = cursor;
		neighborSlots = mul_size(levelSlots, (Size) shared->edgeMaxNeighbors);
		cursor = add_size(cursor, MAXALIGN(mul_size(sizeof(uint32),
													neighborSlots)));
		shared->edgeNeighborDistanceOffset = cursor;
		cursor = add_size(cursor, MAXALIGN(mul_size(sizeof(double),
													neighborSlots)));
		shared->edgeSegmentOffset = cursor;
		cursor = add_size(cursor,
						  MAXALIGN(mul_size(sizeof(PgturbohybridGraphSegmentMetaData),
											(Size) shared->edgeSegmentCount)));
		shared->edgeOrderOffset = cursor;
		cursor = add_size(cursor,
						  MAXALIGN(mul_size(sizeof(uint32),
											(Size) state->nodeCount)));
		shared->edgeInsertedOffset = cursor;
		cursor = add_size(cursor,
						  MAXALIGN(mul_size(sizeof(bool),
											(Size) state->nodeCount)));
		PgturbohybridNativeParallelEdgeCopyNodes(state, shared);
		orderItems = palloc(sizeof(PgturbohybridGraphBuildOrderItem) *
							state->nodeCount);
		for (uint32 nodeId = 0; nodeId < state->nodeCount; nodeId++)
		{
			orderItems[nodeId].nodeId = nodeId;
			orderItems[nodeId].key = PgturbohybridGraphMix64(nodeId);
		}
		qsort(orderItems, state->nodeCount,
			  sizeof(PgturbohybridGraphBuildOrderItem),
			  PgturbohybridGraphBuildOrderCompare);
		order = PgturbohybridNativeParallelEdgeOrder(shared);
		for (uint32 i = 0; i < state->nodeCount; i++)
			order[i] = orderItems[i].nodeId;
		pfree(orderItems);
		goto check_estimate;
	}

	table_parallelscan_initialize(state->heap,
								  ParallelTableScanFromNativeShared(shared),
								  snapshot);

	cursor = add_size(headerBytes,
					  table_parallelscan_estimate(state->heap, snapshot));
	cursor = MAXALIGN(cursor);
	shared->fitCountOffset = cursor;
	cursor = add_size(cursor, participantUsBytes);
	shared->encodeUsOffset = cursor;
	cursor = add_size(cursor, participantUsBytes);
	shared->scanUsOffset = cursor;
	cursor = add_size(cursor, participantUsBytes);
	if (state->scoreMode == PGTURBOHYBRID_SCORE_COSINE ||
		state->scoreMode == PGTURBOHYBRID_SCORE_IP)
	{
		Size		dimSlots = mul_size(participants, dimensions);

		shared->fitMeanOffset = cursor;
		cursor = add_size(cursor,
						  MAXALIGN(mul_size(sizeof(double), dimSlots)));
		shared->fitM2Offset = cursor;
		cursor = add_size(cursor,
						  MAXALIGN(mul_size(sizeof(double), dimSlots)));
	}
	if (phase == PGTURBOHYBRID_NATIVE_PARALLEL_ENCODE)
	{
		if (state->ecShift != NULL && state->ecScale != NULL)
		{
			Size		correctionBytes = mul_size(sizeof(float), dimensions);

			shared->ecShiftOffset = cursor;
			memcpy(PgturbohybridNativeParallelEcShift(shared), state->ecShift,
				   correctionBytes);
			cursor = add_size(cursor, MAXALIGN(correctionBytes));
			shared->ecScaleOffset = cursor;
			memcpy(PgturbohybridNativeParallelEcScale(shared), state->ecScale,
				   correctionBytes);
			cursor = add_size(cursor, MAXALIGN(correctionBytes));
		}
		shared->recordsOffset = cursor;
		cursor = add_size(cursor, shared->recordsBytes);
	}
check_estimate:
	if (cursor > sharedBytes)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pgturbohybrid native parallel build shared memory estimate was too small"),
				 errdetail("required_bytes=%llu allocated_bytes=%llu record_capacity=%u record_bytes=%llu records_bytes=%llu",
						   (unsigned long long) cursor,
						   (unsigned long long) sharedBytes,
						   recordCapacity,
						   (unsigned long long) recordBytes,
						   (unsigned long long) shared->recordsBytes)));
}

static void
PgturbohybridNativeParallelWait(PgturbohybridNativeParallelShared *shared)
{
	for (;;)
	{
		SpinLockAcquire(&shared->mutex);
		if (shared->nparticipantsdone >= shared->nparticipants)
		{
			SpinLockRelease(&shared->mutex);
			break;
		}
		SpinLockRelease(&shared->mutex);
		ConditionVariableSleep(&shared->workersdonecv,
							   WAIT_EVENT_PARALLEL_CREATE_INDEX_SCAN);
	}
	ConditionVariableCancelSleep();
}

static void
PgturbohybridNativeParallelCombineFit(PgturbohybridQuantBuildState *state,
									  PgturbohybridNativeParallelShared *shared)
{
	uint64	   *counts = PgturbohybridNativeParallelFitCounts(shared);
	double	   *means = PgturbohybridNativeParallelFitMean(shared);
	double	   *m2s = PgturbohybridNativeParallelFitM2(shared);
	uint64		totalCount = 0;

	if (state->scoreMode != PGTURBOHYBRID_SCORE_COSINE &&
		state->scoreMode != PGTURBOHYBRID_SCORE_IP)
	{
		for (int slot = 0; slot < shared->participantCapacity; slot++)
			totalCount += counts[slot];
		state->fitCount = totalCount;
		return;
	}

	state->fitMean = MemoryContextAllocZero(state->ctx,
											mul_size(sizeof(double),
													 (Size) state->dimensions));
	state->fitM2 = MemoryContextAllocZero(state->ctx,
										  mul_size(sizeof(double),
												   (Size) state->dimensions));
	for (int slot = 0; slot < shared->participantCapacity; slot++)
	{
		uint64		count = counts[slot];
		double	   *slotMean;
		double	   *slotM2;
		double		total;

		if (count == 0)
			continue;
		slotMean = means + mul_size((Size) slot, (Size) state->dimensions);
		slotM2 = m2s + mul_size((Size) slot, (Size) state->dimensions);
		total = (double) totalCount + (double) count;
		for (int dim = 0; dim < state->dimensions; dim++)
		{
			double		delta = slotMean[dim] - state->fitMean[dim];

			state->fitMean[dim] += delta * ((double) count / total);
			state->fitM2[dim] += slotM2[dim] +
				delta * delta * ((double) totalCount * (double) count / total);
		}
		totalCount += count;
	}
	state->fitCount = totalCount;
	PgturbohybridGraphFinishStreamingFit(state);
}

static void
PgturbohybridNativeParallelMergeRecords(PgturbohybridQuantBuildState *state,
										PgturbohybridNativeParallelShared *shared)
{
	char	   *records = PgturbohybridNativeParallelRecords(shared);
	Size		codeBytes = PgturbohybridGraphCodeBytesForBits(state->dimensions,
															  state->tqBits);

	if (shared->overflowed || shared->recordCount > shared->recordCapacity)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pgturbohybrid native parallel build record area overflowed"),
				 errdetail("record_count=%u record_capacity=%u record_bytes=%llu records_bytes=%llu shared_bytes=%llu",
						   shared->recordCount,
						   shared->recordCapacity,
						   (unsigned long long) shared->recordBytes,
						   (unsigned long long) shared->recordsBytes,
						   (unsigned long long) shared->sharedBytes)));

	qsort(records, shared->recordCount, shared->recordBytes,
		  PgturbohybridNativeParallelRecordCompare);

	for (uint32 row = 0; row < shared->recordCount; row++)
	{
		PgturbohybridNativeParallelRecord *record =
			PgturbohybridNativeParallelRecordAt(shared, row);
		PgturbohybridGraphBuildNode *node;
		uint32		nodeId = state->nodeCount;

		PgturbohybridGraphEnsureNodeCapacity(state);
		node = &state->nodes[state->nodeCount++];
		node->heaptid = record->heaptid;
		node->vectorHash = record->vectorHash;
		node->norm = record->norm;
		node->scale = record->scale;
		node->correction = record->correction;
		node->ecCorrection = record->ecCorrection;
		node->level = PgturbohybridGraphPickLevel(nodeId, state->m);
		node->flags = 0;
		node->exactBlkno = InvalidBlockNumber;
		node->exactOffno = InvalidOffsetNumber;
		node->code = palloc(codeBytes);
		memcpy(node->code, PgturbohybridNativeParallelRecordCode(shared, record),
			   codeBytes);
		if (state->residualRerankBytes > 0)
		{
			node->residualSketch = palloc(state->residualRerankBytes);
			memcpy(node->residualSketch,
				   PgturbohybridNativeParallelRecordResidual(shared, record),
				   state->residualRerankBytes);
		}
		if (state->payloadCount > 0)
		{
			node->payloads = palloc0(state->payloadBytes);
			memcpy(node->payloads, record->payloads, state->payloadBytes);
			node->payloadMask = record->payloadMask;
		}
		PgturbohybridGraphAllocateBuildNeighbors(state, node);
		state->maxLevel = Max(state->maxLevel, node->level);
	}
}

static bool
PgturbohybridNativeRunParallelPhase(PgturbohybridQuantBuildState *state,
									int phase, int workerRequest,
									uint32 recordCapacity, Size recordBytes,
									int64 *elapsedUs,
									PgturbohybridNativeParallelShared **sharedOut,
									ParallelContext **pcxtOut,
									Snapshot *snapshotOut)
{
	ParallelContext *pcxt;
	Snapshot	snapshot;
	Size		sharedBytes;
	PgturbohybridNativeParallelShared *shared;
	int			querylen = 0;
	instr_time	start;

	Assert(workerRequest > 0);
	EnterParallelMode();
	pcxt = CreateParallelContext("pgturbohybrid",
								 "PgturbohybridNativeParallelBuildMain",
								 workerRequest);
	snapshot = state->indexInfo->ii_Concurrent ?
		RegisterSnapshot(GetTransactionSnapshot()) : SnapshotAny;
	sharedBytes = PgturbohybridNativeParallelEstimateShared(state, snapshot,
															workerRequest + 1,
															phase,
															recordCapacity,
															recordBytes);
	shm_toc_estimate_chunk(&pcxt->estimator, sharedBytes);
	shm_toc_estimate_keys(&pcxt->estimator, 1);
	if (debug_query_string)
	{
		querylen = strlen(debug_query_string);
		shm_toc_estimate_chunk(&pcxt->estimator, querylen + 1);
		shm_toc_estimate_keys(&pcxt->estimator, 1);
	}
	InitializeParallelDSM(pcxt);
	if (pcxt->seg == NULL)
	{
		if (IsMVCCSnapshot(snapshot))
			UnregisterSnapshot(snapshot);
		DestroyParallelContext(pcxt);
		ExitParallelMode();
		return false;
	}

	shared = (PgturbohybridNativeParallelShared *) shm_toc_allocate(pcxt->toc,
																	sharedBytes);
	PgturbohybridNativeParallelInitShared(state, shared, snapshot, phase,
										  workerRequest + 1, recordCapacity,
										  recordBytes, sharedBytes);
	shm_toc_insert(pcxt->toc, PARALLEL_KEY_PGTURBOHYBRID_NATIVE_SHARED, shared);
	if (debug_query_string)
	{
		char	   *sharedquery = (char *) shm_toc_allocate(pcxt->toc,
															querylen + 1);

		memcpy(sharedquery, debug_query_string, querylen + 1);
		shm_toc_insert(pcxt->toc, PARALLEL_KEY_PGTURBOHYBRID_NATIVE_QUERY,
					   sharedquery);
	}

	LaunchParallelWorkers(pcxt);
	if (pcxt->nworkers_launched == 0)
	{
		WaitForParallelWorkersToFinish(pcxt);
		if (IsMVCCSnapshot(snapshot))
			UnregisterSnapshot(snapshot);
		DestroyParallelContext(pcxt);
		ExitParallelMode();
		return false;
	}

	shared->nparticipants = pcxt->nworkers_launched + 1;
	elog(DEBUG1, "pgturbohybrid native graph parallel build using %d workers for phase %d",
		 pcxt->nworkers_launched, phase);

	INSTR_TIME_SET_CURRENT(start);
	PgturbohybridNativeParallelScan(state->heap, state->index, shared, true);
	WaitForParallelWorkersToAttach(pcxt);
	PgturbohybridNativeParallelWait(shared);
	if (elapsedUs != NULL)
		*elapsedUs = PgturbohybridGraphElapsedUs(start);

	*sharedOut = shared;
	*pcxtOut = pcxt;
	*snapshotOut = snapshot;
	return true;
}

static bool
PgturbohybridNativeRunParallelEdgePhase(PgturbohybridQuantBuildState *state,
										int workerRequest,
										uint16 edgeSegmentCount,
										int64 *elapsedUs,
										PgturbohybridNativeParallelShared **sharedOut,
										ParallelContext **pcxtOut)
{
	ParallelContext *pcxt;
	Size		sharedBytes;
	PgturbohybridNativeParallelShared *shared;
	int			querylen = 0;
	instr_time	start;

	Assert(workerRequest > 0);
	Assert(edgeSegmentCount > 1);
	EnterParallelMode();
	pcxt = CreateParallelContext("pgturbohybrid",
								 "PgturbohybridNativeParallelBuildMain",
								 workerRequest);
	sharedBytes = PgturbohybridNativeParallelEstimateShared(state, InvalidSnapshot,
															workerRequest + 1,
															PGTURBOHYBRID_NATIVE_PARALLEL_EDGES,
															edgeSegmentCount, 0);
	shm_toc_estimate_chunk(&pcxt->estimator, sharedBytes);
	shm_toc_estimate_keys(&pcxt->estimator, 1);
	if (debug_query_string)
	{
		querylen = strlen(debug_query_string);
		shm_toc_estimate_chunk(&pcxt->estimator, querylen + 1);
		shm_toc_estimate_keys(&pcxt->estimator, 1);
	}
	InitializeParallelDSM(pcxt);
	if (pcxt->seg == NULL)
	{
		DestroyParallelContext(pcxt);
		ExitParallelMode();
		return false;
	}

	shared = (PgturbohybridNativeParallelShared *) shm_toc_allocate(pcxt->toc,
																	sharedBytes);
	PgturbohybridNativeParallelInitShared(state, shared, InvalidSnapshot,
										  PGTURBOHYBRID_NATIVE_PARALLEL_EDGES,
										  workerRequest + 1,
										  edgeSegmentCount, 0, sharedBytes);
	shm_toc_insert(pcxt->toc, PARALLEL_KEY_PGTURBOHYBRID_NATIVE_SHARED, shared);
	if (debug_query_string)
	{
		char	   *sharedquery = (char *) shm_toc_allocate(pcxt->toc,
															querylen + 1);

		memcpy(sharedquery, debug_query_string, querylen + 1);
		shm_toc_insert(pcxt->toc, PARALLEL_KEY_PGTURBOHYBRID_NATIVE_QUERY,
					   sharedquery);
	}

	LaunchParallelWorkers(pcxt);
	if (pcxt->nworkers_launched == 0)
	{
		WaitForParallelWorkersToFinish(pcxt);
		DestroyParallelContext(pcxt);
		ExitParallelMode();
		return false;
	}

	/*
	 * Confirm every launched worker actually attached BEFORE the leader enters
	 * the static-party edge barrier below.  If a counted worker fails to attach
	 * (e.g. postmaster resource pressure) it never arrives at the barrier;
	 * detecting that here lets the build error out cleanly instead of the leader
	 * blocking on BarrierArriveAndWait forever -- the attach check cannot run
	 * once the leader is parked in the barrier.  Mirrors the fit/encode phase.
	 */
	WaitForParallelWorkersToAttach(pcxt);

	shared->nparticipants = pcxt->nworkers_launched + 1;
	BarrierInit(&shared->edgeBarrier, shared->nparticipants);
	SpinLockAcquire(&shared->mutex);
	shared->edgeBarrierReady = true;
	SpinLockRelease(&shared->mutex);
	ConditionVariableBroadcast(&shared->workersdonecv);
	elog(DEBUG1, "pgturbohybrid native graph parallel build using %d workers for batch edge construction",
		 pcxt->nworkers_launched);

	INSTR_TIME_SET_CURRENT(start);
	PgturbohybridNativeParallelEdgeWorker(state->index, shared, true);
	PgturbohybridNativeParallelWait(shared);
	if (elapsedUs != NULL)
		*elapsedUs = PgturbohybridGraphElapsedUs(start);

	*sharedOut = shared;
	*pcxtOut = pcxt;
	return true;
}

static void
PgturbohybridNativeFinishParallelPhase(ParallelContext *pcxt, Snapshot snapshot)
{
	WaitForParallelWorkersToFinish(pcxt);
	if (snapshot != InvalidSnapshot && IsMVCCSnapshot(snapshot))
		UnregisterSnapshot(snapshot);
	DestroyParallelContext(pcxt);
	ExitParallelMode();
}


IndexBuildResult *
tqgraphbuild(Relation heap, Relation index, IndexInfo *indexInfo)
{
	PgturbohybridQuantBuildState state;
	IndexBuildResult *result;
	instr_time	totalStart;
	instr_time	phaseStart;
	instr_time	mergeStart;
	int64		fitCorrectionScanUs = 0;
	int64		scanUs = 0;
	int64		fitCorrectionUs = 0;
	int64		encodeUs = 0;
	int64		edgesUs = 0;
	int64		freeExactVectorsUs = 0;
	int64		reorderNodesUs = 0;
	int64		connectBackboneUs = 0;
	int64		entrySidecarUs = 0;
	int64		writeUs = 0;
	int64		walUs = 0;
	int64		totalUs;
	int64		workerMergeUs = 0;
	int			workerCount = 0;
	int			workerRequest = 0;
	bool		parallelFitEnabled = false;
	bool		parallelScanEnabled = false;
	bool		parallelEncodeEnabled = false;
	bool		parallelEdgeBuildEnabled = false;
	uint16		parallelEdgeSegments = 0;
	uint32		parallelEdgeWorkersLaunched = 0;
	uint64		parallelEdgeRepairUs = 0;
	uint32		workerScanUsCount = 0;
	uint64		workerScanUs[PGTURBOHYBRID_NATIVE_BUILD_STATS_MAX_WORKERS];
	uint64		edgeDistanceStart;
	bool		needsCorrectionFit;
	bool		parallelBuilt = false;

	memset(&state, 0, sizeof(state));
	memset(workerScanUs, 0, sizeof(workerScanUs));
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
	state.entrySidecar = PgturbohybridGraphGetEntrySidecarOption(index);
	state.entrySidecarRepresentatives = PgturbohybridGraphGetEntrySidecarRepresentatives(index);
	state.entrySidecarStrategy = PgturbohybridGraphGetEntrySidecarStrategy(index);
	state.graphBackbone = PgturbohybridGraphGetBackboneOption(index);
	state.residualRerank = PgturbohybridGraphGetResidualRerankOption(index);
	state.residualRerankBytes = PgturbohybridGraphGetResidualRerankBytes(index);
	state.multivectorBuild = PgturbohybridGraphIndexIsMultiVector(index);
	state.multivectorGraphMode = state.multivectorBuild ?
		PgturbohybridGraphGetMultiVectorGraphModeOption(index) :
		PGTURBOHYBRID_DEFAULT_MULTIVECTOR_GRAPH_MODE;
	state.multivectorDocBuildScorer = state.multivectorBuild ?
		PgturbohybridGraphGetMultiVectorDocBuildScorerOption(index) :
		PGTURBOHYBRID_DEFAULT_MULTIVECTOR_DOC_BUILD_SCORER;
	state.multivectorDocStorage = state.multivectorBuild ?
		PgturbohybridGraphGetMultiVectorDocStorageOption(index) :
		PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32;
	state.multivectorTokenPooling = state.multivectorBuild ?
		PgturbohybridGraphGetMultiVectorTokenPoolingOption(index) :
		PGTURBOHYBRID_MULTIVECTOR_TOKEN_POOLING_OFF;
	state.multivectorTokenPoolingTargetRatio =
		PgturbohybridGraphGetMultiVectorTokenPoolingTargetRatio(index);
	state.multivectorTokenPoolingMinTokens =
		PgturbohybridGraphGetMultiVectorTokenPoolingMinTokens(index);
	state.multivectorProxyEncoder =
		PgturbohybridGraphGetMultiVectorProxyEncoderOption(index);
	state.multivectorCentroids = state.multivectorBuild ?
		PgturbohybridGraphGetMultiVectorCentroidsOption(index) :
		PGTURBOHYBRID_MULTIVECTOR_CENTROIDS_OFF;
	state.multivectorCentroidCount =
		PgturbohybridGraphGetMultiVectorCentroidCountOption(index);
	PgturbohybridGraphValidateMultiVectorBuildOptions(&state);
	state.buildExactDistances = PgturbohybridGraphUseExactBuildDistances(&state);
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
	state.buildCodeOnly = PgturbohybridGraphCanBuildCodeOnly(&state);
	state.buildFastEdges = false;
	state.buildNeighborSelectReason = PGTURBOHYBRID_BUILD_NEIGHBOR_SELECT_REASON_UNKNOWN;
	needsCorrectionFit = state.scoreMode == PGTURBOHYBRID_SCORE_COSINE ||
		state.scoreMode == PGTURBOHYBRID_SCORE_IP;
	state.ctx = AllocSetContextCreate(CurrentMemoryContext,
									  "pgturbohybrid native graph build context",
									  ALLOCSET_DEFAULT_SIZES);
	state.nodes = MemoryContextAllocZero(state.ctx, sizeof(PgturbohybridGraphBuildNode) * 1024);
	state.nodeCapacity = 1024;

	if (heap != NULL)
		workerRequest = PgturbohybridNativeBuildWorkerRequest(heap, index);
	if (!PgturbohybridGraphCanUseParallelBuildWorkers(&state))
		workerRequest = 0;

	if (heap != NULL && workerRequest > 0 &&
		!state.multivectorBuild &&
		PgturbohybridNativeParallelEligible(&state, needsCorrectionFit))
	{
			PgturbohybridNativeParallelShared *fitShared = NULL;
			PgturbohybridNativeParallelShared *encodeShared = NULL;
			ParallelContext *fitPcxt = NULL;
			ParallelContext *encodePcxt = NULL;
			Snapshot	fitSnapshot = InvalidSnapshot;
			Snapshot	encodeSnapshot = InvalidSnapshot;
			int64		fitUs = 0;
			int64		encodeWallUs = 0;

			PgturbohybridGraphDebugBuildPhaseStart(&state, "parallel_fit_count_scan");
			INSTR_TIME_SET_CURRENT(phaseStart);
			if (PgturbohybridNativeRunParallelPhase(&state,
													PGTURBOHYBRID_NATIVE_PARALLEL_FIT,
													workerRequest, 0, 0,
													&fitUs, &fitShared,
													&fitPcxt, &fitSnapshot))
			{
				uint32		recordCapacity;
				Size		recordBytes;

				PgturbohybridNativeParallelCombineFit(&state, fitShared);
				state.reltuples = fitShared->reltuples;
				fitCorrectionScanUs = fitUs;
				parallelFitEnabled = true;
				PgturbohybridNativeFinishParallelPhase(fitPcxt, fitSnapshot);
				PgturbohybridGraphDebugBuildPhaseDone(&state,
													  "parallel_fit_count_scan",
													  phaseStart);

				if (state.fitCount > PG_UINT32_MAX)
					ereport(ERROR,
							(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
							 errmsg("pgturbohybrid native graph has too many rows for parallel build")));
				recordCapacity = (uint32) state.fitCount;
				recordBytes = PgturbohybridNativeParallelRecordBytes(&state);
				PgturbohybridGraphDebugBuildPhaseStart(&state,
													  "parallel_scan_encode");
				INSTR_TIME_SET_CURRENT(phaseStart);
				if (PgturbohybridNativeRunParallelPhase(&state,
														PGTURBOHYBRID_NATIVE_PARALLEL_ENCODE,
														workerRequest,
														recordCapacity,
														recordBytes,
														&encodeWallUs,
														&encodeShared,
														&encodePcxt,
														&encodeSnapshot))
				{
					uint64	   *encodeUsBySlot =
						PgturbohybridNativeParallelEncodeUs(encodeShared);

					scanUs = encodeWallUs;
					encodeUs = 0;
					for (int slot = 0; slot < encodeShared->participantCapacity; slot++)
						encodeUs += (int64) encodeUsBySlot[slot];
					state.reltuples = encodeShared->reltuples;
					PgturbohybridGraphDebugBuildPhaseStart(&state,
														  "parallel_worker_merge");
					INSTR_TIME_SET_CURRENT(mergeStart);
					PgturbohybridNativeParallelMergeRecords(&state, encodeShared);
					workerMergeUs = PgturbohybridGraphElapsedUs(mergeStart);
					PgturbohybridGraphDebugBuildPhaseDone(&state,
														  "parallel_worker_merge",
														  mergeStart);
					workerCount = Max(encodeShared->nparticipants - 1, 0);
					parallelScanEnabled = true;
					parallelEncodeEnabled = true;
					workerScanUsCount = Min(encodeShared->nparticipants,
											PGTURBOHYBRID_NATIVE_BUILD_STATS_MAX_WORKERS);
					memcpy(workerScanUs,
						   PgturbohybridNativeParallelScanUs(encodeShared),
						   sizeof(uint64) * workerScanUsCount);
					PgturbohybridNativeFinishParallelPhase(encodePcxt,
														   encodeSnapshot);
					PgturbohybridGraphDebugBuildPhaseDone(&state,
														  "parallel_scan_encode",
														  phaseStart);
					parallelBuilt = true;
				}
				else
				{
					state.fitCount = 0;
					state.fitMean = NULL;
					state.fitM2 = NULL;
					state.ecShift = NULL;
					state.ecScale = NULL;
					state.dPrimeSqI16 = NULL;
					state.weightScale = 0.0f;
					state.mmConst = 0.0;
					fitCorrectionScanUs = 0;
					workerCount = 0;
				}
			}
	}

	if (!parallelBuilt && heap != NULL && state.buildCodeOnly && needsCorrectionFit)
	{
		PgturbohybridGraphDebugBuildPhaseStart(&state, "fit_correction_scan");
		INSTR_TIME_SET_CURRENT(phaseStart);
		state.buildFitPass = true;
		(void) table_index_build_scan(heap, index, indexInfo,
									  true, true, PgturbohybridGraphBuildCallback, &state, NULL);
		PgturbohybridGraphFinishStreamingFit(&state);
		state.buildFitPass = false;
		fitCorrectionScanUs = PgturbohybridGraphElapsedUs(phaseStart);
		PgturbohybridGraphDebugBuildPhaseDone(&state, "fit_correction_scan", phaseStart);
	}

	if (!parallelBuilt && heap != NULL)
	{
		PgturbohybridGraphDebugBuildPhaseStart(&state, "scan");
		INSTR_TIME_SET_CURRENT(phaseStart);
		state.buildEncodeOnAppend = state.buildCodeOnly;
		state.reltuples = table_index_build_scan(heap, index, indexInfo,
												 true, true, PgturbohybridGraphBuildCallback, &state, NULL);
		state.buildEncodeOnAppend = false;
		scanUs = PgturbohybridGraphElapsedUs(phaseStart);
		PgturbohybridGraphDebugBuildPhaseDone(&state, "scan", phaseStart);
		PgturbohybridGraphRecordBuildMemorySnapshot(&state,
													"heap_scan_decode");
		PgturbohybridGraphRecordBuildMemorySnapshot(&state, "token_pooling");
		PgturbohybridGraphRecordBuildMemorySnapshot(&state, "proxy_encoding");
		PgturbohybridGraphRecordBuildMemorySnapshot(&state,
													"centroid_clustering");
		PgturbohybridGraphRecordBuildMemorySnapshot(&state,
													"centroid_residual_computation");
	}

	if (!parallelBuilt && !state.buildCodeOnly)
	{
		PgturbohybridGraphDebugBuildPhaseStart(&state, "fit_correction");
		INSTR_TIME_SET_CURRENT(phaseStart);
		PgturbohybridGraphFitCorrection(&state);
		fitCorrectionUs = PgturbohybridGraphElapsedUs(phaseStart);
		PgturbohybridGraphDebugBuildPhaseDone(&state, "fit_correction", phaseStart);

		PgturbohybridGraphDebugBuildPhaseStart(&state, "encode");
		INSTR_TIME_SET_CURRENT(phaseStart);
		PgturbohybridGraphEncodeBuildNodes(&state);
		encodeUs = PgturbohybridGraphElapsedUs(phaseStart);
		PgturbohybridGraphDebugBuildPhaseDone(&state, "encode", phaseStart);
	}

	PgturbohybridGraphCheckExactSymmetricBuildAllowed(&state);
	if (state.buildMemoryHeapScanDecodeBytes == 0)
	{
		PgturbohybridGraphRecordBuildMemorySnapshot(&state,
													"heap_scan_decode");
		PgturbohybridGraphRecordBuildMemorySnapshot(&state, "token_pooling");
		PgturbohybridGraphRecordBuildMemorySnapshot(&state, "proxy_encoding");
		PgturbohybridGraphRecordBuildMemorySnapshot(&state,
													"centroid_clustering");
		PgturbohybridGraphRecordBuildMemorySnapshot(&state,
													"centroid_residual_computation");
	}
	state.buildFastEdges = PgturbohybridGraphUseFastBuildEdges(&state);
	PgturbohybridGraphDebugBuildPhaseStart(&state, "build_edges");
	INSTR_TIME_SET_CURRENT(phaseStart);
	edgeDistanceStart = state.buildDistanceCalls;
	if (workerRequest <= 0)
		PgturbohybridGraphSetParallelEdgeBuildDisabledReason(&state,
															 PGTURBOHYBRID_PARALLEL_EDGE_BUILD_DISABLED_REASON_NO_WORKERS_REQUESTED);
	if (workerRequest > 0 &&
		PgturbohybridNativeParallelEdgeEligible(&state,
												pgturbohybrid_native_parallel_edge_build ==
												PGTURBOHYBRID_NATIVE_PARALLEL_EDGE_BUILD_ON))
	{
		PgturbohybridNativeParallelShared *edgeShared = NULL;
		ParallelContext *edgePcxt = NULL;
		int64		edgeWallUs = 0;
		uint16		edgeSegmentCount =
			PgturbohybridNativeParallelEdgeSegmentCount(&state, workerRequest);

		if (edgeSegmentCount > 1 &&
			PgturbohybridNativeRunParallelEdgePhase(&state, workerRequest,
													edgeSegmentCount,
													&edgeWallUs,
													&edgeShared,
													&edgePcxt))
		{
			PgturbohybridNativeParallelApplyEdges(&state, edgeShared,
												  &parallelEdgeRepairUs);
			state.buildDistanceCalls += edgeShared->edgeDistanceCalls;
			state.buildEdgeEntrySearchUs += edgeShared->edgeEntrySearchUs;
			state.buildEdgeNeighborSearchUs += edgeShared->edgeNeighborSearchUs;
			state.buildEdgeSearchLayerUs +=
				edgeShared->edgeEntrySearchUs + edgeShared->edgeNeighborSearchUs;
			state.buildEdgeSelectNeighborUs += edgeShared->edgeSelectNeighborUs;
			state.buildEdgeAddNeighborUs += edgeShared->edgeAddNeighborUs;
			state.buildEdgePruneNeighborUs += edgeShared->edgePruneNeighborUs;
			state.buildEdgeEntryUpdateUs += edgeShared->edgeEntryUpdateUs;
			state.buildEdgeNearestTotal += edgeShared->edgeNearestTotal;
			state.buildEdgeNearestSamples += edgeShared->edgeNearestSamples;
			state.buildEdgeMaxFrontierSize =
				Max(state.buildEdgeMaxFrontierSize,
					edgeShared->edgeMaxFrontierSize);
			parallelEdgeBuildEnabled = true;
			parallelEdgeSegments = edgeShared->edgeSegmentCount;
			parallelEdgeWorkersLaunched = Max(edgeShared->nparticipants - 1, 0);
			workerCount = Max(workerCount, (int) parallelEdgeWorkersLaunched);
			state.parallelEdgeBuildDisabledReason =
				PGTURBOHYBRID_PARALLEL_EDGE_BUILD_DISABLED_REASON_NONE;
			edgesUs = edgeWallUs;
			PgturbohybridNativeFinishParallelPhase(edgePcxt, InvalidSnapshot);
		}
		else
		{
			PgturbohybridGraphSetParallelEdgeBuildDisabledReason(&state,
																 edgeSegmentCount <= 1 ?
																 PGTURBOHYBRID_PARALLEL_EDGE_BUILD_DISABLED_REASON_TOO_FEW_SEGMENTS :
																 PGTURBOHYBRID_PARALLEL_EDGE_BUILD_DISABLED_REASON_WORKER_LAUNCH_FAILED);
			PgturbohybridGraphBuildEdges(&state);
		}
	}
	else
		PgturbohybridGraphBuildEdges(&state);
	state.buildEdgeDistanceCalls = state.buildDistanceCalls - edgeDistanceStart;
	if (!parallelEdgeBuildEnabled)
		edgesUs = PgturbohybridGraphElapsedUs(phaseStart);
	PgturbohybridGraphDebugBuildPhaseDone(&state, "build_edges", phaseStart);
	PgturbohybridGraphRecordBuildMemorySnapshot(&state, "graph_edge_build");

	PgturbohybridGraphDebugBuildPhaseStart(&state, "free_exact_build_vectors");
	INSTR_TIME_SET_CURRENT(phaseStart);
	PgturbohybridGraphFreeExactBuildVectors(&state);
	freeExactVectorsUs = PgturbohybridGraphElapsedUs(phaseStart);
	PgturbohybridGraphDebugBuildPhaseDone(&state, "free_exact_build_vectors", phaseStart);

	PgturbohybridGraphDebugBuildPhaseStart(&state, "reorder_nodes");
	INSTR_TIME_SET_CURRENT(phaseStart);
	PgturbohybridGraphReorderBuildNodesForLocality(&state);
	reorderNodesUs = PgturbohybridGraphElapsedUs(phaseStart);
	PgturbohybridGraphDebugBuildPhaseDone(&state, "reorder_nodes", phaseStart);

	if (state.graphBackbone && state.segmentCount <= 1)
	{
		PgturbohybridGraphDebugBuildPhaseStart(&state, "connect_backbone");
		INSTR_TIME_SET_CURRENT(phaseStart);
		PgturbohybridGraphEnsureLevel0Backbone(&state);
		connectBackboneUs = PgturbohybridGraphElapsedUs(phaseStart);
		PgturbohybridGraphDebugBuildPhaseDone(&state, "connect_backbone", phaseStart);
	}

	PgturbohybridGraphDebugBuildPhaseStart(&state, "entry_sidecar");
	INSTR_TIME_SET_CURRENT(phaseStart);
	PgturbohybridGraphBuildRoutingEntries(&state);
	PgturbohybridGraphBuildEntrySidecar(&state);
	entrySidecarUs = PgturbohybridGraphElapsedUs(phaseStart);
	PgturbohybridGraphDebugBuildPhaseDone(&state, "entry_sidecar", phaseStart);

	state.buildScanUs = scanUs;
	state.buildCorrectionUs = fitCorrectionScanUs + fitCorrectionUs;
	state.buildEncodeUs = encodeUs;
	state.buildEdgeUs = edgesUs;
	state.buildWorkerCount = workerCount;

	PgturbohybridGraphDebugBuildPhaseStart(&state, "write_pages");
	INSTR_TIME_SET_CURRENT(phaseStart);
	PgturbohybridGraphWriteIndex(&state);
	writeUs = PgturbohybridGraphElapsedUs(phaseStart);
	PgturbohybridGraphDebugBuildPhaseDone(&state, "write_pages", phaseStart);
	PgturbohybridGraphRecordBuildMemorySnapshot(&state,
												"sidecar_tuple_construction");
	PgturbohybridGraphRecordBuildMemorySnapshot(&state, "page_wal_write");

	if (RelationNeedsWAL(index))
	{
		PgturbohybridGraphDebugBuildPhaseStart(&state, "wal_newpages");
		INSTR_TIME_SET_CURRENT(phaseStart);
		log_newpage_range(index, MAIN_FORKNUM, 0, RelationGetNumberOfBlocks(index), true);
		walUs = PgturbohybridGraphElapsedUs(phaseStart);
		PgturbohybridGraphDebugBuildPhaseDone(&state, "wal_newpages", phaseStart);
		PgturbohybridGraphRecordBuildMemorySnapshot(&state, "page_wal_write");
	}

	totalUs = PgturbohybridGraphElapsedUs(totalStart);

	elog(DEBUG1, "pgturbohybrid native graph build timings: relation=%s nodes=%u dimensions=%d workers=%d scan_ms=%.3f fit_correction_ms=%.3f encode_ms=%.3f build_edges_ms=%.3f write_pages_ms=%.3f wal_ms=%.3f total_ms=%.3f",
		 RelationGetRelationName(index), state.nodeCount, state.dimensions,
		 workerCount,
		 (double) scanUs / 1000.0,
		 (double) (fitCorrectionScanUs + fitCorrectionUs) / 1000.0,
		 (double) encodeUs / 1000.0,
		 (double) edgesUs / 1000.0,
		 (double) writeUs / 1000.0,
		 (double) walUs / 1000.0,
		 (double) totalUs / 1000.0);
	elog(DEBUG1, "pgturbohybrid native graph build distance paths: relation=%s calls=%llu query_split=%llu packed=%llu weighted=%llu code_code=%llu exact=%llu fallback=%llu",
		 RelationGetRelationName(index),
		 (unsigned long long) state.buildDistanceCalls,
		 (unsigned long long) state.buildDistanceQuerySplit,
		 (unsigned long long) state.buildDistancePacked,
		 (unsigned long long) state.buildDistanceWeighted,
		 (unsigned long long) state.buildDistanceCodeCode,
		 (unsigned long long) state.buildDistanceExact,
		 (unsigned long long) state.buildDistanceFallback);

	{
		PgturbohybridNativeBuildStatsSnapshot buildStats;

		memset(&buildStats, 0, sizeof(buildStats));
		buildStats.relid = RelationGetRelid(index);
		strlcpy(buildStats.relationName, RelationGetRelationName(index),
				sizeof(buildStats.relationName));
		strlcpy(buildStats.indexShape,
				PgturbohybridIndexHasLexical(index) ? "hybrid" : "dense_only",
				sizeof(buildStats.indexShape));
		buildStats.nodeCount = state.nodeCount;
		buildStats.dimensions = state.dimensions;
		buildStats.quantizationBits = state.tqBits;
		buildStats.m = state.m;
		buildStats.efConstruction = state.efConstruction;
		buildStats.exactStorage = state.tqExactStorage;
		buildStats.buildCodeOnly = state.buildCodeOnly;
		buildStats.buildFastEdges = state.buildFastEdges;
		buildStats.buildDistanceMode = state.buildDistanceMode;
		buildStats.buildNeighborSelectReason = state.buildNeighborSelectReason;
		buildStats.buildDistanceCalls = state.buildDistanceCalls;
		buildStats.buildDistanceQuerySplit = state.buildDistanceQuerySplit;
		buildStats.buildDistancePacked = state.buildDistancePacked;
		buildStats.buildDistanceWeighted = state.buildDistanceWeighted;
		buildStats.buildDistanceCodeCode = state.buildDistanceCodeCode;
		buildStats.buildDistanceExact = state.buildDistanceExact;
		buildStats.buildDistanceFallback = state.buildDistanceFallback;
		buildStats.multivectorDocExactBuildDistanceCalls =
			state.multivectorDocExactBuildDistanceCalls;
		buildStats.multivectorDocExactBuildDistanceUs =
			state.multivectorDocExactBuildDistanceUs;
		buildStats.multivectorGraphNodeAssignmentUs =
			state.buildGraphNodeAssignmentUs;
		buildStats.multivectorGraphEntrySearchUs =
			state.buildEdgeEntrySearchUs;
		buildStats.multivectorGraphNeighborSearchUs =
			state.buildEdgeNeighborSearchUs;
		buildStats.multivectorGraphNeighborSelectUs =
			state.buildEdgeSelectNeighborUs;
		buildStats.multivectorGraphLinkInsertUs =
			state.buildEdgeAddNeighborUs;
		buildStats.multivectorGraphReciprocalPruneUs =
			state.buildEdgePruneNeighborUs;
		buildStats.multivectorGraphSegmentWriteUs = writeUs;
		buildStats.multivectorGraphWalUs = walUs;
		buildStats.multivectorGraphBuildDistanceProxyCalls =
			state.buildDistanceQuerySplit + state.buildDistancePacked +
			state.buildDistanceWeighted + state.buildDistanceCodeCode;
		buildStats.multivectorGraphBuildDistanceExactCalls =
			state.buildDistanceExact + state.multivectorDocExactBuildDistanceCalls;
		buildStats.multivectorGraphBuildDistanceCacheHits =
			state.buildDistanceCacheHits;
		buildStats.multivectorGraphBuildDistanceCacheMisses =
			state.buildDistanceCacheMisses;
		buildStats.multivectorCentroidBuildUs =
			state.multivectorCentroidBuildUs;
		buildStats.multivectorCentroidClusterUs =
			state.multivectorCentroidClusterUs;
		buildStats.multivectorCentroidResidualUs =
			state.multivectorCentroidResidualUs;
		buildStats.multivectorCentroidBuildDocs =
			state.multivectorCentroidBuildDocs;
		buildStats.multivectorCentroidBuildVectors =
			state.multivectorCentroidBuildVectors;
		buildStats.multivectorProxyBuildUs = state.multivectorProxyBuildUs;
		buildStats.learnedProjectionDocEncodeBuildUs =
			state.learnedProjectionDocEncodeBuildUs;
		buildStats.multivectorDocSidecarWriteUs =
			state.multivectorDocSidecarWriteUs;
		buildStats.multivectorCentroidSidecarWriteUs =
			state.multivectorCentroidSidecarWriteUs;
		buildStats.multivectorCentroidPostingWriteUs =
			state.multivectorCentroidPostingWriteUs;
		buildStats.multivectorCentroidPostingCount =
			state.multivectorCentroidPostingCount;
		PgturbohybridGraphRefreshBuildMemoryEstimates(&state);
		buildStats.multivectorDocVectorsPointerBytes =
			state.multivectorDocVectorsPointerBytes;
		buildStats.multivectorDocVectorChunkRefBytes =
			state.multivectorDocVectorChunkRefBytes;
		buildStats.multivectorDocMapBytesEstimate =
			state.multivectorDocMapBytesEstimate;
		buildStats.multivectorCentroidVectorBytes =
			state.multivectorCentroidVectorBytes;
		buildStats.multivectorCentroidResidualBytes =
			state.multivectorCentroidResidualBytes;
		buildStats.multivectorCentroidPostingBytes =
			state.multivectorCentroidPostingBytes;
		buildStats.graphNodeBytesEstimate = state.graphNodeBytesEstimate;
		buildStats.graphNeighborBytesEstimate =
			state.graphNeighborBytesEstimate;
		buildStats.buildPeakMemoryContextBytes =
			state.buildPeakMemoryContextBytes;
		strlcpy(buildStats.buildPeakMemoryPhase,
				state.buildPeakMemoryPhase[0] != '\0' ?
				state.buildPeakMemoryPhase : "unavailable",
				sizeof(buildStats.buildPeakMemoryPhase));
		buildStats.buildMemoryHeapScanDecodeBytes =
			state.buildMemoryHeapScanDecodeBytes;
		buildStats.buildMemoryTokenPoolingBytes =
			state.buildMemoryTokenPoolingBytes;
		buildStats.buildMemoryProxyEncodingBytes =
			state.buildMemoryProxyEncodingBytes;
		buildStats.buildMemoryCentroidClusteringBytes =
			state.buildMemoryCentroidClusteringBytes;
		buildStats.buildMemoryCentroidResidualBytes =
			state.buildMemoryCentroidResidualBytes;
		buildStats.buildMemorySidecarTupleBytes =
			state.buildMemorySidecarTupleBytes;
		buildStats.buildMemoryPostingTupleBytes =
			state.buildMemoryPostingTupleBytes;
		buildStats.buildMemoryGraphEdgeBytes =
			state.buildMemoryGraphEdgeBytes;
		buildStats.buildMemoryPageWalBytes =
			state.buildMemoryPageWalBytes;
		buildStats.buildDistanceCacheHits = state.buildDistanceCacheHits;
		buildStats.buildDistanceCacheMisses = state.buildDistanceCacheMisses;
		buildStats.buildDistanceCacheStores = state.buildDistanceCacheStores;
		buildStats.buildDistanceCacheCollisions =
			state.buildDistanceCacheCollisions;
		buildStats.buildEdgeDistanceCalls = state.buildEdgeDistanceCalls;
		buildStats.buildEdgeSearchLayerUs =
			state.buildEdgeEntrySearchUs + state.buildEdgeNeighborSearchUs;
		buildStats.buildEdgeSelectNeighborUs = state.buildEdgeSelectNeighborUs;
		buildStats.buildEdgeAddNeighborUs = state.buildEdgeAddNeighborUs;
		buildStats.buildEdgePruneNeighborUs = state.buildEdgePruneNeighborUs;
		buildStats.buildEdgeEntryUpdateUs = state.buildEdgeEntryUpdateUs;
		buildStats.buildEdgeNearestTotal = state.buildEdgeNearestTotal;
		buildStats.buildEdgeNearestSamples = state.buildEdgeNearestSamples;
		buildStats.buildEdgeMaxFrontierSize = state.buildEdgeMaxFrontierSize;
		buildStats.fitCorrectionScanUs = fitCorrectionScanUs;
		buildStats.scanUs = scanUs;
		buildStats.fitCorrectionUs = fitCorrectionUs;
		buildStats.encodeUs = encodeUs;
		buildStats.buildEdgesUs = edgesUs;
		buildStats.freeExactVectorsUs = freeExactVectorsUs;
		buildStats.reorderNodesUs = reorderNodesUs;
		buildStats.connectBackboneUs = connectBackboneUs;
		buildStats.entrySidecarUs = entrySidecarUs;
		buildStats.writePagesUs = writeUs;
		buildStats.walUs = walUs;
		buildStats.totalUs = totalUs;
		buildStats.workerCount = workerCount;
		buildStats.nativeSegmentCount = state.segmentCount > 0 ? state.segmentCount : 1;
		buildStats.nativeSegmentBytes =
			buildStats.nativeSegmentCount * sizeof(PgturbohybridGraphSegmentMetaData);
		buildStats.parallelSegmentBuildEnabled = false;
		strlcpy(buildStats.segmentBuildMode,
				parallelEdgeBuildEnabled ?
				"parallel_batch" :
				(buildStats.nativeSegmentCount > 1 ? "serial" : "single"),
				sizeof(buildStats.segmentBuildMode));
		buildStats.nativeBuildWorkersRequested = workerRequest;
		buildStats.nativeBuildWorkersLaunched = workerCount;
		buildStats.parallelFitEnabled = parallelFitEnabled;
		buildStats.parallelScanEnabled = parallelScanEnabled;
		buildStats.parallelEncodeEnabled = parallelEncodeEnabled;
		buildStats.parallelSegmentBuildEnabled = parallelEdgeBuildEnabled;
		buildStats.parallelEdgeBuildEnabled = parallelEdgeBuildEnabled;
		buildStats.parallelEdgeBuildDisabledReason =
			state.parallelEdgeBuildDisabledReason;
		buildStats.parallelEdgeSegments = parallelEdgeSegments;
		buildStats.parallelEdgeWorkersLaunched = parallelEdgeWorkersLaunched;
		buildStats.parallelEdgeRepairUs = parallelEdgeRepairUs;
		buildStats.workerMergeUs = workerMergeUs;
		buildStats.workerScanUsCount = workerScanUsCount;
		memcpy(buildStats.workerScanUs, workerScanUs,
			   sizeof(uint64) * workerScanUsCount);
		PgturbohybridGraphRecordNativeBuildStats(&buildStats);
	}

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
	state.typeInfo = PgturbohybridGraphGetTypeInfo(index);
	state.m = PgturbohybridGraphGetM(index);
	state.efConstruction = PgturbohybridGraphGetEfConstruction(index);
	state.tqBits = PgturbohybridGraphGetTqBits(index);
	state.tqWeighted = PgturbohybridGraphGetTqWeightedOption(index);
	state.tqQuantileFit = PgturbohybridGraphGetTqQuantileFitOption(index);
	state.tqRenorm = PgturbohybridGraphGetTqRenormOption(index);
	state.tqExactStorage = PgturbohybridGraphGetTqExactStorageOption(index);
	state.entrySidecar = PgturbohybridGraphGetEntrySidecarOption(index);
	state.entrySidecarRepresentatives = PgturbohybridGraphGetEntrySidecarRepresentatives(index);
	state.entrySidecarStrategy = PgturbohybridGraphGetEntrySidecarStrategy(index);
	state.graphBackbone = PgturbohybridGraphGetBackboneOption(index);
	state.residualRerank = PgturbohybridGraphGetResidualRerankOption(index);
	state.residualRerankBytes = PgturbohybridGraphGetResidualRerankBytes(index);
	state.multivectorBuild = PgturbohybridGraphIndexIsMultiVector(index);
	state.multivectorGraphMode = state.multivectorBuild ?
		PgturbohybridGraphGetMultiVectorGraphModeOption(index) :
		PGTURBOHYBRID_DEFAULT_MULTIVECTOR_GRAPH_MODE;
	state.multivectorDocBuildScorer = state.multivectorBuild ?
		PgturbohybridGraphGetMultiVectorDocBuildScorerOption(index) :
		PGTURBOHYBRID_DEFAULT_MULTIVECTOR_DOC_BUILD_SCORER;
	state.multivectorDocStorage = state.multivectorBuild ?
		PgturbohybridGraphGetMultiVectorDocStorageOption(index) :
		PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32;
	state.multivectorTokenPooling = state.multivectorBuild ?
		PgturbohybridGraphGetMultiVectorTokenPoolingOption(index) :
		PGTURBOHYBRID_MULTIVECTOR_TOKEN_POOLING_OFF;
	state.multivectorTokenPoolingTargetRatio =
		PgturbohybridGraphGetMultiVectorTokenPoolingTargetRatio(index);
	state.multivectorTokenPoolingMinTokens =
		PgturbohybridGraphGetMultiVectorTokenPoolingMinTokens(index);
	state.multivectorProxyEncoder =
		PgturbohybridGraphGetMultiVectorProxyEncoderOption(index);
	state.multivectorCentroids = state.multivectorBuild ?
		PgturbohybridGraphGetMultiVectorCentroidsOption(index) :
		PGTURBOHYBRID_MULTIVECTOR_CENTROIDS_OFF;
	state.multivectorCentroidCount =
		PgturbohybridGraphGetMultiVectorCentroidCountOption(index);
	PgturbohybridGraphValidateMultiVectorBuildOptions(&state);
	state.buildExactDistances = PgturbohybridGraphUseExactBuildDistances(&state);
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
	state.buildCodeOnly = PgturbohybridGraphCanBuildCodeOnly(&state);
	state.buildFastEdges = PgturbohybridGraphUseFastBuildEdges(&state);
	PgturbohybridGraphWriteIndex(&state);
	log_newpage_range(index, INIT_FORKNUM, 0, RelationGetNumberOfBlocksInFork(index, INIT_FORKNUM), true);
}



void
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
	so->graphBatchScoredCodes = 0;
	so->graphScalarScoredCodes = 0;
	so->graphBatchKernel = PGTURBOHYBRID_SCORING_SCALAR;
	memset(so->graphScoreKernelNodes, 0, sizeof(so->graphScoreKernelNodes));
	memset(so->graphScoreKernelCalls, 0, sizeof(so->graphScoreKernelCalls));
	so->graphBatchCalls = 0;
	so->graphBatchNodes = 0;
	so->graphBaseFrontierPushes = 0;
	so->graphBaseFrontierPops = 0;
	so->graphBaseNearestOffers = 0;
	so->graphBaseVisitedChecks = 0;
	so->graphBaseDuplicateSkips = 0;
	so->graphBaseBatchCalls = 0;
	so->graphBaseBatchNodes = 0;
	so->graphBaseMaxFrontier = 0;
	so->graphCodePageAttempts = 0;
	so->graphCodePageHits = 0;
	so->graphCodePageMisses = 0;
	so->graphCodeTuplesCopied = 0;
	so->graphCodeArenaAllocatedBytes = 0;
	so->graphCodeArenaUsedBytes = 0;
	so->graphCandidateCount = 0;
	so->graphRescoreCount = 0;
	so->graphRescorePages = 0;
	so->graphCodePagesRead = 0;
	so->graphAdjPagesRead = 0;
	so->graphEntryPointCount = 0;
	so->graphEntrySampleConfigured = 0;
	so->graphEntrySampleEffective = 0;
	so->graphEntrySampleScored = 0;
	so->graphEntrySidecarCount = 0;
	so->graphEntrySidecarScored = 0;
	so->graphEntrySidecarSelected = 0;
	so->graphEntrySidecarRepresentativesConfigured = 0;
	so->graphEntrySidecarStrategy = PGTURBOHYBRID_DEFAULT_ENTRY_SIDECAR_STRATEGY;
	so->graphEntrySidecarUs = 0;
	so->graphPayloadEntrySeedingMode = pgturbohybrid_payload_entry_seeding;
	so->graphPayloadEntrySeedingHit = false;
	so->graphPayloadEntrySeedCount = 0;
	so->graphPayloadEntrySeedPayloadSlot = -1;
	so->graphPayloadEntrySeedRangeCount = 0;
	so->graphPayloadEntrySeedUs = 0;
	so->graphResidualRerankCount = 0;
	so->graphResidualRerankBytes = 0;
	so->graphResidualRerankUs = 0;
	so->graphResidualRerankMode = pgturbohybrid_dense_residual_rerank_mode;
	so->graphResidualRerankWeightEffective = 0.0;
	so->graphResidualRerankBand = 0;
	so->graphResidualRerankMaxAdjustment = 0.0;
	so->graphResidualRerankReorderedCount = 0;
	so->graphResidualRerankTopKChanged = false;
	so->graphHeapRescoreCount = 0;
	so->graphHeapFetchUs = 0;
	so->graphHeapRescoreUs = 0;
	so->graphHeapRescoreMode = PGTURBOHYBRID_DENSE_HEAP_RESCORE_OFF;
	so->graphHeapRescoreReason =
		PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_UNKNOWN;
	so->graphHeapRescoreAutoEnabled = false;
	so->graphExactRescoreSource = PGTURBOHYBRID_EXACT_RESCORE_SOURCE_NONE;
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
	so->graphAdaptiveWideningMode = pgturbohybrid_dense_adaptive_widening;
	so->graphAdaptiveTriggered = false;
	so->graphAdaptiveTriggerReason = PGTURBOHYBRID_DENSE_ADAPTIVE_REASON_NONE;
	so->graphAdaptiveInitialResultTarget = 0;
	so->graphAdaptiveFinalResultTarget = 0;
	so->graphAdaptiveInitialSearchEf = 0;
	so->graphAdaptiveFinalSearchEf = 0;
	so->graphAdaptiveGapTop10 = 0.0;
	so->graphAdaptiveGapBoundary = 0.0;
	so->graphUncertaintyRetryMode = pgturbohybrid_dense_uncertainty_retry;
	so->graphUncertaintyRetryTriggered = false;
	so->graphUncertaintyRetryReason = PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_NONE;
	so->graphUncertaintyRetryPasses = 0;
	so->graphUncertaintyInitialResultTarget = 0;
	so->graphUncertaintyFinalResultTarget = 0;
	so->graphUncertaintyInitialSearchEf = 0;
	so->graphUncertaintyFinalSearchEf = 0;
	so->graphUncertaintyGapTop10 = 0.0;
	so->graphUncertaintyGapBoundary = 0.0;
	so->graphLocalExpansionMode = pgturbohybrid_dense_local_expansion;
	so->graphLocalExpansionTriggered = false;
	so->graphLocalExpansionSeedCount = 0;
	so->graphLocalExpansionNeighborsScored = 0;
	so->graphLocalExpansionCandidatesAdded = 0;
	so->graphLocalExpansionUs = 0;
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
	so->graphFinalK = 0;
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
	switch ((PgturbohybridDenseBudgetPolicy) pgturbohybrid_dense_budget_policy)
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
	switch ((PgturbohybridDenseBudgetPolicy) pgturbohybrid_dense_budget_policy)
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
	switch ((PgturbohybridDenseBudgetPolicy) policy)
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
PgturbohybridGraphDenseAdaptiveWideningModeName(int mode)
{
	switch ((TqDenseAdaptiveWideningMode) mode)
	{
		case PGTURBOHYBRID_DENSE_ADAPTIVE_WIDENING_AUTO:
			return "auto";
		case PGTURBOHYBRID_DENSE_ADAPTIVE_WIDENING_ON:
			return "on";
		case PGTURBOHYBRID_DENSE_ADAPTIVE_WIDENING_OFF:
		default:
			return "off";
	}
}

const char *
PgturbohybridGraphDenseAdaptiveWideningReasonName(int reason)
{
	switch ((TqDenseAdaptiveWideningReason) reason)
	{
		case PGTURBOHYBRID_DENSE_ADAPTIVE_REASON_FORCED:
			return "forced";
		case PGTURBOHYBRID_DENSE_ADAPTIVE_REASON_UNDERFILLED:
			return "underfilled";
		case PGTURBOHYBRID_DENSE_ADAPTIVE_REASON_FLAT_TOP10:
			return "flat_top10";
		case PGTURBOHYBRID_DENSE_ADAPTIVE_REASON_FLAT_BOUNDARY:
			return "flat_boundary";
		case PGTURBOHYBRID_DENSE_ADAPTIVE_REASON_NONE:
		default:
			return "none";
	}
}

const char *
PgturbohybridGraphDenseLocalExpansionModeName(int mode)
{
	switch ((TqDenseLocalExpansionMode) mode)
	{
		case PGTURBOHYBRID_DENSE_LOCAL_EXPANSION_AUTO:
			return "auto";
		case PGTURBOHYBRID_DENSE_LOCAL_EXPANSION_ON:
			return "on";
		case PGTURBOHYBRID_DENSE_LOCAL_EXPANSION_OFF:
		default:
			return "off";
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
	so->graphExactCache = PgturbohybridGraphGetGraphExactCache(index);
	so->graphStorageKind = PGTURBOHYBRID_GRAPH_STORAGE_QUANT_GRAPH_NATIVE;
	so->pgturbohybridGraphScan = true;
	PgturbohybridGraphResetScan(so);
	so->graphEntrySidecarRepresentativesConfigured =
		PgturbohybridGraphGetEntrySidecarRepresentatives(index);
	so->graphEntrySidecarStrategy = PgturbohybridGraphGetEntrySidecarStrategy(index);
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
	so->graphEntrySidecarRepresentativesConfigured =
		PgturbohybridGraphGetEntrySidecarRepresentatives(scan->indexRelation);
	so->graphEntrySidecarStrategy =
		PgturbohybridGraphGetEntrySidecarStrategy(scan->indexRelation);
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
		if (so->typeInfo->normalize == pgturbohybrid_l2_normalize)
		{
			Vector	   *vector = (Vector *) DatumGetPointer(value);

			if (PgturbohybridVectorNorm(vector) <= 0.0)
				value = PointerGetDatum(NULL);
			else
				value = PointerGetDatum(PgturbohybridL2NormalizeFast(vector));
		}
		else if (!PgturbohybridGraphCheckNorm(&so->support, value))
			value = PointerGetDatum(NULL);
		else
			value = PgturbohybridGraphNormValue(so->typeInfo, so->support.collation, value);
	}

	return value;
}

int
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









bool
PgturbohybridGraphReadMeta(Relation index, PgturbohybridGraphMetaPageData *meta)
{
	Buffer		buf;
	Page		page;
	PageHeader	pageHeader;
	PgturbohybridGraphMetaPage metap;
	Size		metaBytes;
	Size		metaStart;

	buf = ReadBuffer(index, PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	pageHeader = (PageHeader) page;
	metap = PgturbohybridGraphPageGetMeta(page);

	if (metap->magicNumber == PGTURBOHYBRID_GRAPH_MAGIC_NUMBER &&
		metap->storageKind != PGTURBOHYBRID_GRAPH_STORAGE_QUANT_GRAPH_NATIVE)
	{
		uint32		foundVersion = metap->version;
		uint32		foundStorageKind = metap->storageKind;

		UnlockReleaseBuffer(buf);
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("pgturbohybrid index \"%s\" uses removed legacy storage format %u (version %u)",
						RelationGetRelationName(index), foundStorageKind,
						foundVersion),
				 errhint("REINDEX the index to rebuild it with the native TurboQuant graph.")));
	}
	if (metap->magicNumber != PGTURBOHYBRID_GRAPH_MAGIC_NUMBER)
	{
		UnlockReleaseBuffer(buf);
		return false;
	}

	/*
	 * The magic + storageKind matched, so this metapage was written by this
	 * extension's native graph writer.  That writer always stamps the current
	 * native format version (PGTURBOHYBRID_GRAPH_NATIVE_VERSION), so any other
	 * value here is
	 * either disk corruption or an index built by an incompatible (future)
	 * format -- both of which would otherwise be misread silently.  Reject it
	 * with a clear error and a REINDEX hint rather than proceeding to clamp
	 * fields under an unknown layout.  This cannot fire for a validly-built
	 * current-version index.
	 */
	if (metap->version != PGTURBOHYBRID_GRAPH_NATIVE_VERSION)
	{
		uint32		foundVersion = metap->version;

		UnlockReleaseBuffer(buf);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("pgturbohybrid index \"%s\" uses unsupported on-disk metapage format version %u (expected %u)",
						RelationGetRelationName(index), foundVersion,
						(uint32) PGTURBOHYBRID_GRAPH_NATIVE_VERSION),
				 errhint("REINDEX the index to rebuild it in the current format.")));
	}

	memset(meta, 0, sizeof(PgturbohybridGraphMetaPageData));
	metaStart = (Size) ((char *) metap - (char *) page);
	metaBytes = pageHeader->pd_lower > metaStart ? pageHeader->pd_lower - metaStart : 0;
	if (metaBytes > sizeof(PgturbohybridGraphMetaPageData))
		metaBytes = sizeof(PgturbohybridGraphMetaPageData);
	memcpy(meta, metap, metaBytes);
	if (meta->tqBits != 1 && meta->tqBits != 2 &&
		meta->tqBits != PGTURBOHYBRID_DEFAULT_BITS &&
		meta->tqBits != 8)
		meta->tqBits = PGTURBOHYBRID_DEFAULT_BITS;
	if (meta->tqMultivectorGraphMode >
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("invalid multivector graph mode %u in index \"%s\"",
						meta->tqMultivectorGraphMode,
						RelationGetRelationName(index))));
	if (meta->tqBm25MetaStartBlkno <= PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO)
		meta->tqBm25MetaStartBlkno = InvalidBlockNumber;
	if (meta->tqMultivectorDocMapStartBlkno <=
		PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO ||
		meta->tqMultivectorDocMapPageCount == 0 ||
		meta->tqMultivectorDocCount == 0)
	{
		meta->tqMultivectorDocMapStartBlkno = InvalidBlockNumber;
		meta->tqMultivectorDocMapPageCount = 0;
		meta->tqMultivectorDocCount = 0;
		meta->tqMultivectorDocMapBytes = 0;
		meta->tqMultivectorDocMapVersion = 0;
		meta->tqMultivectorDocMapFlags = 0;
	}
	else if (meta->tqMultivectorDocMapVersion !=
			 PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION)
	{
		/* Preserve the start block so require/auto loaders can report REINDEX guidance. */
		meta->tqMultivectorDocMapPageCount = 0;
		meta->tqMultivectorDocMapBytes = 0;
	}
	if (meta->tqEntrySidecarCount > PGTURBOHYBRID_GRAPH_MAX_ENTRY_SIDECAR_REPRESENTATIVES)
		meta->tqEntrySidecarCount = 0;
	if (meta->tqEntrySidecarBytes != meta->tqEntrySidecarCount * sizeof(uint32))
	{
		meta->tqEntrySidecarCount = 0;
		meta->tqEntrySidecarBytes = 0;
	}
	if (meta->tqRoutingEntryCount > PGTURBOHYBRID_GRAPH_MAX_ROUTING_ENTRIES)
		meta->tqRoutingEntryCount = 0;
	if (meta->tqRoutingEntryBytes != meta->tqRoutingEntryCount * sizeof(uint32))
	{
		meta->tqRoutingEntryCount = 0;
		meta->tqRoutingEntryBytes = 0;
	}
	for (uint16 i = 0; i < meta->tqRoutingEntryCount; i++)
	{
		if (meta->tqRoutingEntryNodeIds[i] >= meta->tqNodeCount)
		{
			meta->tqRoutingEntryCount = 0;
			meta->tqRoutingEntryBytes = 0;
			break;
		}
	}
	if (meta->tqSegmentCount == 0 && meta->tqNodeCount > 0)
	{
		meta->tqSegmentCount = 1;
		meta->tqSegmentBytes = sizeof(PgturbohybridGraphSegmentMetaData);
		memset(meta->tqSegments, 0, sizeof(meta->tqSegments));
		meta->tqSegments[0].startNodeId = 0;
		meta->tqSegments[0].nodeCount = meta->tqNodeCount;
		meta->tqSegments[0].entryNodeId = meta->tqEntryNodeId;
		meta->tqSegments[0].entryLevel = meta->entryLevel;
		meta->tqSegments[0].codeStartBlkno = meta->tqCodeStartBlkno;
		meta->tqSegments[0].adjStartBlkno = meta->tqAdjStartBlkno;
		meta->tqSegments[0].exactStartBlkno = meta->tqExactStartBlkno;
		meta->tqSegments[0].correctionStartBlkno = meta->tqCorrectionStartBlkno;
	}
	if (meta->tqSegmentCount > PGTURBOHYBRID_GRAPH_MAX_NATIVE_SEGMENTS ||
		meta->tqSegmentBytes != meta->tqSegmentCount * sizeof(PgturbohybridGraphSegmentMetaData))
	{
		meta->tqSegmentCount = 0;
		meta->tqSegmentBytes = 0;
	}
	for (uint16 i = 0; i < meta->tqSegmentCount; i++)
	{
		PgturbohybridGraphSegmentMetaData *segment = &meta->tqSegments[i];
		uint64		endNodeId = (uint64) segment->startNodeId + segment->nodeCount;

		if (segment->nodeCount == 0 || endNodeId > meta->tqNodeCount ||
			(segment->entryNodeId == UINT_MAX ? segment->entryLevel != -1 :
			 (segment->entryNodeId >= meta->tqNodeCount ||
			  segment->entryNodeId < segment->startNodeId ||
			  segment->entryNodeId >= endNodeId)))
		{
			meta->tqSegmentCount = 0;
			meta->tqSegmentBytes = 0;
			break;
		}
	}
	if ((meta->tqFlags & PGTURBOHYBRID_GRAPH_TQ_RESIDUAL_RERANK) == 0)
		meta->tqResidualRerankBytes = 0;
	if (meta->tqResidualRerankBytes > PGTURBOHYBRID_GRAPH_MAX_RESIDUAL_RERANK_BYTES)
		meta->tqResidualRerankBytes = 0;
	UnlockReleaseBuffer(buf);
	return true;
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


static bool
PgturbohybridGraphPayloadSeedAlreadyQueued(uint32 *nodeIds, int count, uint32 nodeId)
{
	for (int i = 0; i < count; i++)
	{
		if (nodeIds[i] == nodeId)
			return true;
	}

	return false;
}

static void
PgturbohybridGraphAddPayloadEntrySeeds(Relation index, PgturbohybridGraphScanOpaque so,
						PgturbohybridGraphMetaPageData *meta,
						PgturbohybridGraphScanStorage *storage,
						PgturbohybridGraphFrontierItem *entries, int *entryCount,
						Datum query, int payloadSlot, int32 payloadValue)
{
	uint32		payloadFirst = 0;
	uint32		payloadCount = 0;
	uint32		seedNodeIds[PGTURBOHYBRID_MAX_PAYLOAD_ENTRY_SEED_COUNT];
	double		seedDistances[PGTURBOHYBRID_MAX_PAYLOAD_ENTRY_SEED_COUNT];
	int			seedCount = 0;
	int			seedTarget;
	instr_time	seedStart;

	so->graphPayloadEntrySeedingMode = pgturbohybrid_payload_entry_seeding;
	so->graphPayloadEntrySeedingHit = false;
	so->graphPayloadEntrySeedCount = 0;
	so->graphPayloadEntrySeedPayloadSlot = payloadSlot;
	so->graphPayloadEntrySeedRangeCount = 0;

	if (pgturbohybrid_payload_entry_seeding == PGTURBOHYBRID_PAYLOAD_ENTRY_SEEDING_OFF ||
		payloadSlot < 0 || meta->tqNodeCount == 0)
		return;

	INSTR_TIME_SET_CURRENT(seedStart);
	if (!PgturbohybridGraphPayloadRefRange(storage, payloadSlot, payloadValue,
										   &payloadFirst, &payloadCount))
	{
		PgturbohybridGraphAddElapsedUs(&so->graphPayloadEntrySeedUs, seedStart);
		return;
	}

	so->graphPayloadEntrySeedingHit = true;
	so->graphPayloadEntrySeedRangeCount = payloadCount;
	seedTarget = Min(pgturbohybrid_payload_entry_seed_count,
					 PGTURBOHYBRID_MAX_PAYLOAD_ENTRY_SEED_COUNT);
	if (payloadCount < (uint32) seedTarget)
		seedTarget = (int) payloadCount;

	for (int i = 0; i < seedTarget; i++)
	{
		uint32		rangeOffset;
		uint32		refIndex;
		uint32		nodeId;

		CHECK_FOR_INTERRUPTS();
		rangeOffset = seedTarget == 1 ? 0 :
			(uint32) (((uint64) i * (payloadCount - 1)) / (seedTarget - 1));
		refIndex = payloadFirst + rangeOffset;
		nodeId = storage->payloadRefs[refIndex].nodeId;

		if (nodeId >= meta->tqNodeCount ||
			PgturbohybridGraphEntryAlreadySelected(entries, *entryCount, nodeId) ||
			PgturbohybridGraphPayloadSeedAlreadyQueued(seedNodeIds, seedCount, nodeId) ||
			!PgturbohybridGraphLoadCodePage(index, so, meta, storage, nodeId))
			continue;

		seedNodeIds[seedCount++] = nodeId;
	}

	if (seedCount == 0)
	{
		PgturbohybridGraphAddElapsedUs(&so->graphPayloadEntrySeedUs, seedStart);
		return;
	}

	PgturbohybridGraphScoreNodeBatchTimed(so, storage, seedNodeIds, seedCount,
										  seedDistances, query);
	for (int i = 0; i < seedCount; i++)
	{
		PgturbohybridGraphFrontierItem payloadEntry;

		payloadEntry.nodeId = seedNodeIds[i];
		payloadEntry.distance = seedDistances[i];
		PgturbohybridGraphOfferDistanceEntry(entries, entryCount, payloadEntry);
	}

	so->graphPayloadEntrySeedCount = seedCount;
	PgturbohybridGraphAddElapsedUs(&so->graphPayloadEntrySeedUs, seedStart);
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

		CHECK_FOR_INTERRUPTS();
		changed = false;
		if (!PgturbohybridGraphLoadAdjPage(index, so, meta, storage, current.nodeId, level))
			break;

		slot = PgturbohybridGraphScanAdjSlot(meta, current.nodeId, level);
		for (int i = 0; i < storage->neighborCounts[slot]; i++)
		{
			uint32		neighbor = storage->neighbors[slot][i];

			CHECK_FOR_INTERRUPTS();
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
	/*
	 * Traversal counters accumulated in locals (kept in registers) and flushed
	 * to so-> once at the end -- a per-neighbor so->field RMW in this hot loop
	 * (thousands of visited checks/query) measurably hurts p50.
	 */
	int64		cFrontierPushes = 0;
	int64		cFrontierPops = 0;
	int64		cNearestOffers = 0;
	int64		cVisitedChecks = 0;
	int64		cDuplicateSkips = 0;
	int64		cBatchCalls = 0;
	int64		cBatchNodes = 0;
	int64		cMaxFrontier = 0;
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

		cVisitedChecks++;
		if (entry.nodeId >= meta->tqNodeCount)
			continue;
		if (useVisitGeneration ?
			storage->visitedGeneration[entry.nodeId] == visitGeneration :
			visited[entry.nodeId])
		{
			cDuplicateSkips++;
			continue;
		}

		if (useVisitGeneration)
			storage->visitedGeneration[entry.nodeId] = visitGeneration;
		else
			visited[entry.nodeId] = true;

		PgturbohybridGraphFrontierHeapPushGrowing(&frontier, &frontierCount, &frontierCapacity,
									   maxFrontierCapacity, entry, true);
		cFrontierPushes++;
		(void) PgturbohybridGraphOfferNearest(nearest, searchEf, &nearestCount, entry.nodeId, entry.distance);
		cNearestOffers++;
	}

	while (frontierCount > 0)
	{
		PgturbohybridGraphFrontierItem item;
		uint32		nodeId;
		int			slot;

		if (frontierCount > cMaxFrontier)
			cMaxFrontier = frontierCount;

		item = PgturbohybridGraphFrontierHeapPop(frontier, &frontierCount, true);
		nodeId = item.nodeId;
		cFrontierPops++;

		CHECK_FOR_INTERRUPTS();
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

				CHECK_FOR_INTERRUPTS();
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

				cVisitedChecks++;
				if (neighbor >= meta->tqNodeCount)
					continue;
				if (useVisitGeneration ?
					storage->visitedGeneration[neighbor] == visitGeneration :
					visited[neighbor])
				{
					cDuplicateSkips++;
					continue;
				}

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
			cBatchCalls++;
			cBatchNodes += batchCount;
			for (int i = 0; i < batchCount; i++)
			{
				uint32		neighbor = batchNodeIds[i];
				double		neighborDistance = batchDistances[i];
				bool		accepted;

				accepted = PgturbohybridGraphOfferNearest(nearest, searchEf, &nearestCount,
											   neighbor, neighborDistance);
				cNearestOffers++;
				if (accepted)
				{
					PgturbohybridGraphFrontierItem frontierItem;

					frontierItem.nodeId = neighbor;
					frontierItem.distance = neighborDistance;
					PgturbohybridGraphFrontierHeapPushGrowing(&frontier, &frontierCount,
												   &frontierCapacity,
												   maxFrontierCapacity,
												   frontierItem, true);
					cFrontierPushes++;
				}
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
		PgturbohybridGraphOfferCandidate(so, results, resultTarget, &resultCount,
							  nodeId, &node->heaptid, resultDistance,
							  exactScored);
	}

	if (visited != NULL)
		pfree(visited);
	if (frontierAllocated)
		pfree(frontier);
	if (nearestAllocated)
		pfree(nearest);

	/* Flush traversal counters once (accumulates across re-traversals). */
	so->graphBaseFrontierPushes += cFrontierPushes;
	so->graphBaseFrontierPops += cFrontierPops;
	so->graphBaseNearestOffers += cNearestOffers;
	so->graphBaseVisitedChecks += cVisitedChecks;
	so->graphBaseDuplicateSkips += cDuplicateSkips;
	so->graphBaseBatchCalls += cBatchCalls;
	so->graphBaseBatchNodes += cBatchNodes;
	if (cMaxFrontier > so->graphBaseMaxFrontier)
		so->graphBaseMaxFrontier = cMaxFrontier;
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
	PgturbohybridGraphFrontierItem *sampled = NULL;
	uint32	   *sampledNodeIds = NULL;
	double	   *sampledDistances = NULL;
	int			entryCount = 0;
	int			sampledCount = 0;
	int64		segmentsSearched = 0;
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
	so->graphSegmentCount = meta->tqSegmentCount > 0 ? meta->tqSegmentCount : 1;

	for (uint16 i = 0; i < meta->tqSegmentCount; i++)
	{
		PgturbohybridGraphSegmentMetaData *segment = &meta->tqSegments[i];
		uint32		nodeId = segment->entryNodeId;

		CHECK_FOR_INTERRUPTS();
		if (nodeId == entry.nodeId || nodeId >= meta->tqNodeCount ||
			!PgturbohybridGraphLoadCodePage(index, so, meta, storage, nodeId))
			continue;
		PgturbohybridGraphOfferLevelEntry(entries, &entryCount, entryLevels,
										  nodeId, storage->nodes[nodeId].level);
		segmentsSearched++;
	}
	so->graphSegmentsSearched = meta->tqSegmentCount > 1 ?
		Min((int64) meta->tqSegmentCount, segmentsSearched + 1) : 1;

	/*
	 * Keep alternative high-level entry points instead of relying on one
	 * global entry. The best level-bearing node IDs are stored at build time
	 * so the scan path stays bounded on large corpora.
	 */
	for (uint16 i = 0; i < meta->tqRoutingEntryCount; i++)
	{
		uint32		nodeId = meta->tqRoutingEntryNodeIds[i];
		PgturbohybridGraphScanNode *node;

		CHECK_FOR_INTERRUPTS();
		if (nodeId == entry.nodeId || nodeId >= meta->tqNodeCount ||
			!PgturbohybridGraphLoadCodePage(index, so, meta, storage, nodeId))
			continue;
		node = &storage->nodes[nodeId];
		if (node->flags & PGTURBOHYBRID_GRAPH_NODE_DEAD || node->level <= 0)
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
		int			configuredSamples =
			Max(pgturbohybrid_multivector_doc_graph_entry_sample_count, 0);

		if (so->tq.bits == PGTURBOHYBRID_DEFAULT_BITS &&
			(TqScoreMode) so->tq.scoreMode == PGTURBOHYBRID_SCORE_L2 &&
			so->tq.dimensions >= 1024)
			sampleTarget = Max(PGTURBOHYBRID_GRAPH_MAX_ENTRY_POINTS,
							   searchEf / PGTURBOHYBRID_GRAPH_HIGHDIM_ENTRY_SAMPLE_DIVISOR);

		if (configuredSamples > 0)
			sampleCount = Min((int) meta->tqNodeCount, configuredSamples);
		else
			sampleCount = Min((int) meta->tqNodeCount,
							  Min(PGTURBOHYBRID_GRAPH_ENTRY_SAMPLE_COUNT,
								  Max(sampleTarget,
									  PGTURBOHYBRID_GRAPH_MAX_ENTRY_POINTS)));
		so->graphEntrySampleConfigured = configuredSamples;
		so->graphEntrySampleEffective = sampleCount;
		sampled = palloc(sizeof(PgturbohybridGraphFrontierItem) * sampleCount);
		sampledNodeIds = palloc(sizeof(uint32) * sampleCount);
		sampledDistances = palloc(sizeof(double) * sampleCount);

		for (int i = 0; i < sampleCount; i++)
		{
			uint32		nodeId = sampleCount == 1 ? 0 :
				(uint32) (((uint64) i * (meta->tqNodeCount - 1)) / (sampleCount - 1));
			bool		seen = PgturbohybridGraphEntryAlreadySelected(entries, entryCount,
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
		so->graphEntrySampleScored = sampledCount;
		pfree(sampledDistances);
		pfree(sampledNodeIds);
		pfree(sampled);
	}

	PgturbohybridGraphAddPayloadEntrySeeds(index, so, meta, storage, entries,
										   &entryCount, query, payloadSlot,
										   payloadValue);

	if (meta->tqEntrySidecarCount > 0)
	{
		uint32		sidecarNodeIds[PGTURBOHYBRID_GRAPH_MAX_ENTRY_SIDECAR_REPRESENTATIVES];
		double		sidecarDistances[PGTURBOHYBRID_GRAPH_MAX_ENTRY_SIDECAR_REPRESENTATIVES];
		int			sidecarScored = 0;
		int			sidecarSelected = 0;
		instr_time	sidecarStart;

		INSTR_TIME_SET_CURRENT(sidecarStart);
		so->graphEntrySidecarCount = meta->tqEntrySidecarCount;
		for (int i = 0; i < meta->tqEntrySidecarCount; i++)
		{
			uint32		nodeId = meta->tqEntrySidecarNodeIds[i];
			bool		seen = PgturbohybridGraphEntryAlreadySelected(entries, entryCount,
														 nodeId);

			CHECK_FOR_INTERRUPTS();
			for (int j = 0; j < sidecarScored; j++)
			{
				if (sidecarNodeIds[j] == nodeId)
				{
					seen = true;
					break;
				}
			}

			if (seen || nodeId >= meta->tqNodeCount ||
				!PgturbohybridGraphLoadCodePage(index, so, meta, storage, nodeId))
				continue;
			sidecarNodeIds[sidecarScored++] = nodeId;
		}

		PgturbohybridGraphScoreNodeBatchTimed(so, storage, sidecarNodeIds,
								   sidecarScored, sidecarDistances, query);
		for (int i = 0; i < sidecarScored; i++)
		{
			PgturbohybridGraphFrontierItem sidecarEntry;

			sidecarEntry.nodeId = sidecarNodeIds[i];
			sidecarEntry.distance = sidecarDistances[i];
			PgturbohybridGraphOfferDistanceEntry(entries, &entryCount, sidecarEntry);
		}
		for (int i = 0; i < sidecarScored; i++)
		{
			if (PgturbohybridGraphEntryAlreadySelected(entries, entryCount,
											sidecarNodeIds[i]))
				sidecarSelected++;
		}
		so->graphEntrySidecarScored = sidecarScored;
		so->graphEntrySidecarSelected = sidecarSelected;
		PgturbohybridGraphAddElapsedUs(&so->graphEntrySidecarUs, sidecarStart);
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
						 int payloadSlot, int32 payloadValue, Datum query,
						 PgturbohybridGraphFillCandidateBandReason reason)
{
	bool	   *selected;
	uint32		batchNodeIds[PGTURBOHYBRID_GRAPH_MAX_NEIGHBORS];
	double		batchDistances[PGTURBOHYBRID_GRAPH_MAX_NEIGHBORS];
	int			batchCount = 0;
	uint32		payloadFirst = 0;
	uint32		payloadCount = 0;
	bool		usePayloadRefs;
	bool		firstFillCall = so->graphFillCandidateBandCalls == 0;

	so->graphFillCandidateBandCalls++;
	so->graphFillCandidateBandReason =
		reason >= PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_NONE &&
		reason <= PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_UNKNOWN ?
		reason : PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_UNKNOWN;
	if (firstFillCall)
		so->graphFillCandidateBandSelectedBefore = count;
	so->graphFillCandidateBandSelectedAfter = count;
	so->graphFillCandidateBandTarget = resultTarget;
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
	so->graphFillCandidateBandUsedPayloadRefs =
		so->graphFillCandidateBandUsedPayloadRefs || usePayloadRefs;
	so->graphFillCandidateBandPayloadRefCount = usePayloadRefs ? payloadCount : 0;
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

		CHECK_FOR_INTERRUPTS();
		so->graphFillCandidateBandVisited++;
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
			so->graphFillCandidateBandScored += batchCount;
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
		so->graphFillCandidateBandScored += batchCount;
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
	so->graphFillCandidateBandSelectedAfter = count;
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

		CHECK_FOR_INTERRUPTS();
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

static double
PgturbohybridGraphNormalizedGap(double best, double boundary)
{
	double		gap = boundary - best;
	double		scale = Max(Max(fabs(best), fabs(boundary)), 1.0);

	if (gap < 0)
		gap = 0;
	return gap / scale;
}

static bool
PgturbohybridGraphAutoAdaptiveWideningEligible(PgturbohybridGraphMetaPageData *meta,
									bool exactFree, int64 finalTarget)
{
	if (meta == NULL || !exactFree)
		return false;
	if (meta->dimensions <= 0 || meta->dimensions > 256)
		return false;
	if (meta->tqBits > 4)
		return false;
	if (finalTarget <= 0 || finalTarget > 20)
		return false;

	switch ((PgturbohybridProfile) pgturbohybrid_profile)
	{
		case PGTURBOHYBRID_PROFILE_BALANCED:
		case PGTURBOHYBRID_PROFILE_QUALITY:
		case PGTURBOHYBRID_PROFILE_MATCHED_RECALL:
		case PGTURBOHYBRID_PROFILE_DEBUG:
			return true;
		case PGTURBOHYBRID_PROFILE_LATENCY:
		default:
			return false;
	}
}

static int
PgturbohybridGraphEffectiveAdaptiveWideningMode(PgturbohybridGraphMetaPageData *meta,
									 bool exactFree, int64 finalTarget)
{
	switch ((TqDenseAdaptiveWideningMode) pgturbohybrid_dense_adaptive_widening)
	{
		case PGTURBOHYBRID_DENSE_ADAPTIVE_WIDENING_ON:
			return PGTURBOHYBRID_DENSE_ADAPTIVE_WIDENING_ON;
		case PGTURBOHYBRID_DENSE_ADAPTIVE_WIDENING_AUTO:
			return PgturbohybridGraphAutoAdaptiveWideningEligible(meta, exactFree,
																  finalTarget) ?
				PGTURBOHYBRID_DENSE_ADAPTIVE_WIDENING_AUTO :
				PGTURBOHYBRID_DENSE_ADAPTIVE_WIDENING_OFF;
		case PGTURBOHYBRID_DENSE_ADAPTIVE_WIDENING_OFF:
		default:
			return PGTURBOHYBRID_DENSE_ADAPTIVE_WIDENING_OFF;
	}
}

static double
PgturbohybridGraphEffectiveAdaptiveWideningMultiplier(void)
{
	double		profileCap = pgturbohybrid_dense_adaptive_widening_max_multiplier;

	switch ((PgturbohybridProfile) pgturbohybrid_profile)
	{
		case PGTURBOHYBRID_PROFILE_BALANCED:
		case PGTURBOHYBRID_PROFILE_MATCHED_RECALL:
			profileCap = Min(profileCap, 1.5);
			break;
		case PGTURBOHYBRID_PROFILE_QUALITY:
		case PGTURBOHYBRID_PROFILE_DEBUG:
			profileCap = Min(profileCap, 2.0);
			break;
		case PGTURBOHYBRID_PROFILE_LATENCY:
		default:
			break;
	}

	return Max(1.0, Min(pgturbohybrid_dense_adaptive_widening_multiplier,
						profileCap));
}

static double
PgturbohybridGraphEffectiveAdaptiveWideningMaxMultiplier(void)
{
	switch ((PgturbohybridProfile) pgturbohybrid_profile)
	{
		case PGTURBOHYBRID_PROFILE_BALANCED:
		case PGTURBOHYBRID_PROFILE_MATCHED_RECALL:
			return Max(1.0, Min(pgturbohybrid_dense_adaptive_widening_max_multiplier,
								1.5));
		case PGTURBOHYBRID_PROFILE_QUALITY:
		case PGTURBOHYBRID_PROFILE_DEBUG:
			return Max(1.0, Min(pgturbohybrid_dense_adaptive_widening_max_multiplier,
								2.0));
		case PGTURBOHYBRID_PROFILE_LATENCY:
		default:
			return Max(1.0, pgturbohybrid_dense_adaptive_widening_max_multiplier);
	}
}

static int64
PgturbohybridGraphAdaptiveFinalTarget(PgturbohybridGraphScanOpaque so,
						   int64 requestedBaseTarget)
{
	if (so != NULL && so->graphFinalK > 0)
		return so->graphFinalK;
	if (so != NULL && so->hasTupleTargetRows && so->tupleTargetRows > 0)
		return so->tupleTargetRows;
	return Min(requestedBaseTarget, (int64) PGTURBOHYBRID_DEFAULT_FINAL_K);
}

static int
PgturbohybridGraphEffectiveSegmentBudgetMode(PgturbohybridGraphMetaPageData *meta)
{
	uint16		segmentCount = meta->tqSegmentCount > 0 ? meta->tqSegmentCount : 1;

	if (segmentCount <= 1)
		return PGTURBOHYBRID_NATIVE_SEGMENT_BUDGET_OFF;

	switch ((PgturbohybridNativeSegmentBudgetMode) pgturbohybrid_native_segment_budget)
	{
		case PGTURBOHYBRID_NATIVE_SEGMENT_BUDGET_OFF:
		case PGTURBOHYBRID_NATIVE_SEGMENT_BUDGET_SQRT:
		case PGTURBOHYBRID_NATIVE_SEGMENT_BUDGET_LINEAR:
			return pgturbohybrid_native_segment_budget;
		case PGTURBOHYBRID_NATIVE_SEGMENT_BUDGET_AUTO:
		default:
			if (pgturbohybrid_profile == PGTURBOHYBRID_PROFILE_QUALITY ||
				pgturbohybrid_profile == PGTURBOHYBRID_PROFILE_MATCHED_RECALL ||
				(meta->tqFlags & PGTURBOHYBRID_GRAPH_TQ_EXACT_BUILD) != 0)
				return PGTURBOHYBRID_NATIVE_SEGMENT_BUDGET_LINEAR;
			return PGTURBOHYBRID_NATIVE_SEGMENT_BUDGET_SQRT;
	}
}

int
PgturbohybridGraphScaleSearchEfForSegments(PgturbohybridGraphScanOpaque so,
								PgturbohybridGraphMetaPageData *meta,
								int searchEf)
{
	uint16		segmentCount = meta->tqSegmentCount > 0 ? meta->tqSegmentCount : 1;
	int			mode = PgturbohybridGraphEffectiveSegmentBudgetMode(meta);
	double		multiplier = 1.0;
	int64		scaledSearchEf;

	so->graphPerSegmentBudgetMode = mode;
	so->graphSearchEfBeforeSegmentScaling = searchEf;

	if (segmentCount <= 1 || mode == PGTURBOHYBRID_NATIVE_SEGMENT_BUDGET_OFF)
	{
		so->graphSearchEfAfterSegmentScaling = searchEf;
		return searchEf;
	}

	if (mode == PGTURBOHYBRID_NATIVE_SEGMENT_BUDGET_LINEAR)
		multiplier = (double) segmentCount;
	else if (mode == PGTURBOHYBRID_NATIVE_SEGMENT_BUDGET_SQRT)
		multiplier = sqrt((double) segmentCount);

	scaledSearchEf = (int64) ceil((double) Max(searchEf, 1) * multiplier);
	scaledSearchEf = Min(scaledSearchEf, (int64) meta->tqNodeCount);
	scaledSearchEf = Min(scaledSearchEf, (int64) INT_MAX);
	so->graphSearchEfAfterSegmentScaling = scaledSearchEf;
	return (int) Max(scaledSearchEf, (int64) searchEf);
}

static bool
PgturbohybridGraphShouldAdaptiveWiden(PgturbohybridGraphScanOpaque so,
						   PgturbohybridGraphResult *results, int count,
						   int resultTarget, int64 requestedBaseTarget,
						   int adaptiveMode)
{
	double		threshold;
	int			top10Index;
	int			boundaryIndex;
	int64		finalTarget;

	so->graphAdaptiveWideningMode = adaptiveMode;
	so->graphAdaptiveInitialResultTarget = resultTarget;
	so->graphAdaptiveFinalResultTarget = resultTarget;
	so->graphAdaptiveTriggerReason = PGTURBOHYBRID_DENSE_ADAPTIVE_REASON_NONE;
	so->graphAdaptiveGapTop10 = 0.0;
	so->graphAdaptiveGapBoundary = 0.0;

	if (adaptiveMode == PGTURBOHYBRID_DENSE_ADAPTIVE_WIDENING_OFF ||
		resultTarget <= 0 || count <= 0)
		return false;

	finalTarget = PgturbohybridGraphAdaptiveFinalTarget(so, requestedBaseTarget);
	top10Index = Min(count, 10) - 1;
	so->graphAdaptiveGapTop10 =
		PgturbohybridGraphNormalizedGap(results[0].distance,
										 results[top10Index].distance);
	boundaryIndex = (int) Min((int64) count - 1,
							  Max((int64) 0, finalTarget - 1));
	if (boundaryIndex + 1 < count)
		so->graphAdaptiveGapBoundary =
			PgturbohybridGraphNormalizedGap(results[boundaryIndex].distance,
											 results[boundaryIndex + 1].distance);
	else
		so->graphAdaptiveGapBoundary = 0.0;

	if (adaptiveMode == PGTURBOHYBRID_DENSE_ADAPTIVE_WIDENING_ON)
	{
		so->graphAdaptiveTriggerReason = PGTURBOHYBRID_DENSE_ADAPTIVE_REASON_FORCED;
		return true;
	}

	if (count < resultTarget)
	{
		so->graphAdaptiveTriggerReason = PGTURBOHYBRID_DENSE_ADAPTIVE_REASON_UNDERFILLED;
		return true;
	}

	threshold = pgturbohybrid_dense_adaptive_min_gap > 0.0 ?
		pgturbohybrid_dense_adaptive_min_gap : 0.05;

	if (top10Index >= 1 && so->graphAdaptiveGapTop10 <= threshold)
	{
		so->graphAdaptiveTriggerReason = PGTURBOHYBRID_DENSE_ADAPTIVE_REASON_FLAT_TOP10;
		return true;
	}

	if (boundaryIndex + 1 < count &&
		so->graphAdaptiveGapBoundary <= threshold * 4.0)
	{
		so->graphAdaptiveTriggerReason = PGTURBOHYBRID_DENSE_ADAPTIVE_REASON_FLAT_BOUNDARY;
		return true;
	}

	return false;
}

static bool
PgturbohybridGraphLocalExpansionShouldTrigger(PgturbohybridGraphResult *results, int count,
								   int resultTarget)
{
	double		threshold;
	double		gapTop10;
	double		gapBoundary;
	int			top10Index;

	if (pgturbohybrid_dense_local_expansion == PGTURBOHYBRID_DENSE_LOCAL_EXPANSION_OFF ||
		resultTarget <= 0 || count <= 0)
		return false;
	if (pgturbohybrid_dense_local_expansion == PGTURBOHYBRID_DENSE_LOCAL_EXPANSION_ON)
		return true;

	top10Index = Min(count, 10) - 1;
	gapTop10 = PgturbohybridGraphNormalizedGap(results[0].distance,
											   results[top10Index].distance);
	gapBoundary = PgturbohybridGraphNormalizedGap(results[0].distance,
												  results[count - 1].distance);
	threshold = pgturbohybrid_dense_adaptive_min_gap > 0.0 ?
		pgturbohybrid_dense_adaptive_min_gap : 0.05;

	if (count < resultTarget)
		return true;
	if (top10Index >= 1 && gapTop10 <= threshold)
		return true;
	if (count >= resultTarget && gapBoundary <= threshold * 4.0)
		return true;
	return false;
}

static bool
PgturbohybridGraphResultContainsNode(PgturbohybridGraphResult *results, int count,
						  uint32 nodeId)
{
	for (int i = 0; i < count; i++)
	{
		if (results[i].nodeId == nodeId)
			return true;
	}
	return false;
}

static bool
PgturbohybridGraphNodeIdArrayContains(uint32 *nodeIds, int count, uint32 nodeId)
{
	for (int i = 0; i < count; i++)
	{
		if (nodeIds[i] == nodeId)
			return true;
	}
	return false;
}

static bool
PgturbohybridGraphOfferCandidateSorted(PgturbohybridGraphResult *results, int target,
							int *count, uint32 nodeId, ItemPointer heaptid,
							double distance)
{
	PgturbohybridGraphResult item;

	if (target <= 0)
		return false;
	item.nodeId = nodeId;
	item.heaptid = *heaptid;
	item.distance = distance;
	item.exactScored = false;

	if (*count < target)
	{
		results[*count] = item;
		(*count)++;
		return true;
	}
	if (*count > 0 && PgturbohybridGraphResultLess(item, results[*count - 1]))
	{
		results[*count - 1] = item;
		return true;
	}
	return false;
}

static int
PgturbohybridGraphApplyLocalExpansion(Relation index, PgturbohybridGraphScanOpaque so,
						   PgturbohybridGraphMetaPageData *meta,
						   PgturbohybridGraphScanStorage *storage,
						   PgturbohybridGraphResult *results, int resultTarget,
						   int count, Datum query, int payloadSlot,
						   int32 payloadValue)
{
	uint32	   *candidateNodeIds;
	uint32	   *batchNodeIds;
	double	   *batchDistances;
	int			maxNeighbors;
	int			seedCount;
	int			candidateCount = 0;
	int			batchCount = 0;
	int			acceptedCount = 0;
	instr_time	start;

	so->graphLocalExpansionMode = pgturbohybrid_dense_local_expansion;
	if (!PgturbohybridGraphLocalExpansionShouldTrigger(results, count, resultTarget))
		return count;

	maxNeighbors = Max(1, pgturbohybrid_dense_local_expansion_max_neighbors);
	seedCount = Min(count, Max(1, pgturbohybrid_dense_local_expansion_topn));
	candidateNodeIds = palloc(sizeof(uint32) * maxNeighbors);
	batchNodeIds = palloc(sizeof(uint32) * Min(maxNeighbors, PGTURBOHYBRID_GRAPH_MAX_NEIGHBORS));
	batchDistances = palloc(sizeof(double) * Min(maxNeighbors, PGTURBOHYBRID_GRAPH_MAX_NEIGHBORS));

	INSTR_TIME_SET_CURRENT(start);
	for (int seedIndex = 0; seedIndex < seedCount && candidateCount < maxNeighbors; seedIndex++)
	{
		uint32		seedNodeId = results[seedIndex].nodeId;
		int			slot;

		CHECK_FOR_INTERRUPTS();
		if (seedNodeId >= meta->tqNodeCount ||
			!PgturbohybridGraphLoadAdjPage(index, so, meta, storage, seedNodeId, 0))
			continue;
		slot = PgturbohybridGraphScanAdjSlot(meta, seedNodeId, 0);
		for (int i = 0; i < storage->neighborCounts[slot] && candidateCount < maxNeighbors; i++)
		{
			uint32		neighbor = storage->neighbors[slot][i];
			PgturbohybridGraphScanNode *node;

			CHECK_FOR_INTERRUPTS();
			if (neighbor >= meta->tqNodeCount ||
				PgturbohybridGraphResultContainsNode(results, count, neighbor) ||
				PgturbohybridGraphNodeIdArrayContains(candidateNodeIds, candidateCount, neighbor) ||
				!PgturbohybridGraphLoadCodePage(index, so, meta, storage, neighbor))
				continue;

			node = &storage->nodes[neighbor];
			if (node->flags & PGTURBOHYBRID_GRAPH_NODE_DEAD ||
				!PgturbohybridGraphNodeMatchesPayload(node, payloadSlot, payloadValue))
				continue;
			candidateNodeIds[candidateCount++] = neighbor;
		}
	}

	for (int i = 0; i < candidateCount; i++)
	{
		CHECK_FOR_INTERRUPTS();
		batchNodeIds[batchCount++] = candidateNodeIds[i];
		if (batchCount == PGTURBOHYBRID_GRAPH_MAX_NEIGHBORS)
		{
			PgturbohybridGraphScoreNodeBatchTimed(so, storage, batchNodeIds,
										   batchCount, batchDistances, query);
			for (int j = 0; j < batchCount; j++)
			{
				PgturbohybridGraphScanNode *node = &storage->nodes[batchNodeIds[j]];

				if (PgturbohybridGraphOfferCandidateSorted(results, resultTarget,
												&count, batchNodeIds[j],
												&node->heaptid, batchDistances[j]))
					acceptedCount++;
			}
			batchCount = 0;
		}
	}
	if (batchCount > 0)
	{
		PgturbohybridGraphScoreNodeBatchTimed(so, storage, batchNodeIds,
									   batchCount, batchDistances, query);
		for (int j = 0; j < batchCount; j++)
		{
			PgturbohybridGraphScanNode *node = &storage->nodes[batchNodeIds[j]];

			if (PgturbohybridGraphOfferCandidateSorted(results, resultTarget,
											&count, batchNodeIds[j],
											&node->heaptid, batchDistances[j]))
				acceptedCount++;
		}
	}

	so->graphLocalExpansionTriggered = true;
	so->graphLocalExpansionSeedCount = seedCount;
	so->graphLocalExpansionNeighborsScored = candidateCount;
	so->graphLocalExpansionCandidatesAdded = acceptedCount;
	PgturbohybridGraphAddElapsedUs(&so->graphLocalExpansionUs, start);
	pfree(candidateNodeIds);
	pfree(batchNodeIds);
	pfree(batchDistances);
	return count;
}

int
PgturbohybridGraphRunTraversalPass(IndexScanDesc scan,
						PgturbohybridGraphScanOpaque so,
						PgturbohybridGraphMetaPageData *meta,
						PgturbohybridGraphScanStorage *storage,
						PgturbohybridGraphResult *results,
						int resultTarget, int searchEf, Datum query,
						int payloadSlot, int32 payloadValue,
						bool hasPayloadFilter, bool payloadExactBandMissed,
						double estimatedSelectivity, int fillReason)
{
	instr_time	phaseStart;
	int			count;

	INSTR_TIME_SET_CURRENT(phaseStart);
	count = PgturbohybridGraphTraverse(scan->indexRelation, so, meta, storage, results,
							resultTarget, searchEf, query, payloadSlot,
							payloadValue);
	PgturbohybridGraphAddElapsedUs(&so->graphTraverseUs, phaseStart);
	if (!hasPayloadFilter && count < resultTarget &&
		resultTarget >= (int) meta->tqNodeCount)
	{
		INSTR_TIME_SET_CURRENT(phaseStart);
		count = PgturbohybridGraphFillCandidateBand(scan->indexRelation, so, meta,
										 storage, results, resultTarget, count,
										 payloadSlot, payloadValue, query,
										 PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_UNDERFILLED_FULL_TARGET);
		PgturbohybridGraphAddElapsedUs(&so->graphFillUs, phaseStart);
	}
	if (estimatedSelectivity > 0 && estimatedSelectivity < 1 && count < resultTarget)
	{
		INSTR_TIME_SET_CURRENT(phaseStart);
		count = PgturbohybridGraphFillCandidateBand(scan->indexRelation, so, meta,
										 storage, results, resultTarget, count,
										 payloadSlot, payloadValue, query,
										 payloadExactBandMissed ?
										 PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_PAYLOAD_EXACT_BAND_MISS :
										 fillReason);
		PgturbohybridGraphAddElapsedUs(&so->graphFillUs, phaseStart);
	}
	INSTR_TIME_SET_CURRENT(phaseStart);
	qsort(results, count, sizeof(PgturbohybridGraphResult), PgturbohybridGraphResultCompare);
	PgturbohybridGraphAddElapsedUs(&so->graphSortUs, phaseStart);

	count = PgturbohybridGraphApplyLocalExpansion(scan->indexRelation, so, meta,
									   storage, results, resultTarget, count,
									   query, payloadSlot, payloadValue);
	if (so->graphLocalExpansionTriggered)
	{
		INSTR_TIME_SET_CURRENT(phaseStart);
		qsort(results, count, sizeof(PgturbohybridGraphResult), PgturbohybridGraphResultCompare);
		PgturbohybridGraphAddElapsedUs(&so->graphSortUs, phaseStart);
	}
	return count;
}

static int
PgturbohybridGraphCaptureTopNodeIds(PgturbohybridGraphResult *results, int count,
						 int requested, uint32 **nodeIds)
{
	int			limit = Min(count, requested);

	*nodeIds = NULL;
	if (limit <= 0)
		return 0;
	*nodeIds = palloc(sizeof(uint32) * limit);
	for (int i = 0; i < limit; i++)
		(*nodeIds)[i] = results[i].nodeId;
	return limit;
}

static bool
PgturbohybridGraphTopNodeIdsChanged(PgturbohybridGraphResult *results, int count,
						 uint32 *nodeIds, int nodeIdCount)
{
	if (nodeIds == NULL || nodeIdCount <= 0 || count < nodeIdCount)
		return false;
	for (int i = 0; i < nodeIdCount; i++)
	{
		if (results[i].nodeId != nodeIds[i])
			return true;
	}
	return false;
}

static bool
PgturbohybridGraphShouldUncertaintyRetry(PgturbohybridGraphScanOpaque so,
							  PgturbohybridGraphResult *results, int count,
							  int resultTarget, int searchEf,
							  int64 requestedBaseTarget, bool hasPayloadFilter,
							  bool residualReordered, bool heapReordered,
							  int64 nodeCount, int *reason)
{
	int			mode = pgturbohybrid_dense_uncertainty_retry;
	int			top10Index;
	int			boundaryIndex;
	int64		finalTarget;
	double		threshold = pgturbohybrid_dense_uncertainty_min_gap;

	so->graphUncertaintyRetryMode = mode;
	so->graphUncertaintyInitialResultTarget = resultTarget;
	so->graphUncertaintyFinalResultTarget = resultTarget;
	so->graphUncertaintyInitialSearchEf = searchEf;
	so->graphUncertaintyFinalSearchEf = searchEf;
	so->graphUncertaintyRetryReason = PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_NONE;
	so->graphUncertaintyGapTop10 = 0.0;
	so->graphUncertaintyGapBoundary = 0.0;
	*reason = PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_NONE;

	if (mode == PGTURBOHYBRID_DENSE_UNCERTAINTY_RETRY_OFF ||
		pgturbohybrid_dense_uncertainty_retry_max_passes < 1 ||
		resultTarget <= 0 || searchEf <= 0 || nodeCount <= 0)
		return false;

	if (count > 0)
	{
		finalTarget = PgturbohybridGraphAdaptiveFinalTarget(so, requestedBaseTarget);
		top10Index = Min(count, 10) - 1;
		so->graphUncertaintyGapTop10 =
			PgturbohybridGraphNormalizedGap(results[0].distance,
											 results[top10Index].distance);
		boundaryIndex = (int) Min((int64) count - 1,
								  Max((int64) 0, finalTarget - 1));
		if (boundaryIndex + 1 < count)
			so->graphUncertaintyGapBoundary =
				PgturbohybridGraphNormalizedGap(results[boundaryIndex].distance,
												 results[boundaryIndex + 1].distance);
	}

	if (mode == PGTURBOHYBRID_DENSE_UNCERTAINTY_RETRY_ON)
	{
		*reason = PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_FORCED;
		return true;
	}

	if (hasPayloadFilter && count < resultTarget)
		*reason = PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_PAYLOAD_UNDERFILLED;
	else if (count < resultTarget)
		*reason = PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_UNDERFILLED;
	else if (count > 0 && Min(count, 10) > 1 &&
			 so->graphUncertaintyGapTop10 <= threshold)
		*reason = PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_FLAT_TOP10;
	else if (count > 0 && so->graphUncertaintyGapBoundary > 0.0 &&
			 so->graphUncertaintyGapBoundary <= threshold * 4.0)
		*reason = PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_FLAT_BOUNDARY;
	else if (so->graphEntrySidecarScored > 0 &&
			 so->graphEntrySidecarSelected == 0)
		*reason = PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_SIDECAR_UNUSED;
	else if (residualReordered)
		*reason = PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_RESIDUAL_REORDERED;
	else if (heapReordered)
		*reason = PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_HEAP_REORDERED;

	return *reason != PGTURBOHYBRID_DENSE_UNCERTAINTY_REASON_NONE;
}

static bool
PgturbohybridGraphApplyUncertaintyRetry(IndexScanDesc scan,
							 PgturbohybridGraphScanOpaque so,
							 PgturbohybridGraphMetaPageData *meta,
							 PgturbohybridGraphScanStorage *storage,
							 PgturbohybridGraphResult **results,
							 int *resultTarget, int *searchEf, int *count,
							 Datum query, int payloadSlot, int32 payloadValue,
							 bool hasPayloadFilter, bool payloadExactBandMissed,
							 double estimatedSelectivity, int64 requestedBaseTarget,
							 bool residualReordered, bool heapReordered)
{
	int			reason;
	double		multiplier = pgturbohybrid_dense_uncertainty_retry_multiplier;
	int64		scanCap = pgturbohybrid_max_scan_tuples > 0 ?
		(int64) pgturbohybrid_max_scan_tuples : (int64) meta->tqNodeCount;
	int64		targetCap = Min((int64) meta->tqNodeCount, scanCap);
	int64		wideTarget;
	int64		wideEf;

	so->graphUncertaintyRetryPasses = 1;
	if (!PgturbohybridGraphShouldUncertaintyRetry(so, *results, *count,
												  *resultTarget, *searchEf,
												  requestedBaseTarget,
												  hasPayloadFilter,
												  residualReordered,
												  heapReordered,
												  meta->tqNodeCount,
												  &reason))
		return false;

	targetCap = Max(targetCap, (int64) 1);
	wideTarget = (int64) ceil((double) *resultTarget * Max(multiplier, 1.0));
	wideTarget = Min(wideTarget, targetCap);
	wideTarget = Min(wideTarget, (int64) INT_MAX);
	wideTarget = Max(wideTarget, (int64) *resultTarget + 1);
	wideTarget = Min(wideTarget, targetCap);

	wideEf = (int64) ceil((double) *searchEf * Max(multiplier, 1.0));
	wideEf = Max(wideEf, wideTarget);
	wideEf = Min(wideEf, targetCap);
	wideEf = Min(wideEf, (int64) INT_MAX);

	if (wideTarget <= *resultTarget && wideEf <= *searchEf)
		return false;

	pfree(*results);
	*resultTarget = (int) Max(wideTarget, (int64) 1);
	*searchEf = (int) Max(wideEf, (int64) *resultTarget);
	so->graphUncertaintyRetryTriggered = true;
	so->graphUncertaintyRetryReason = reason;
	so->graphUncertaintyRetryPasses = 2;
	so->graphUncertaintyFinalResultTarget = *resultTarget;
	so->graphUncertaintyFinalSearchEf = *searchEf;
	so->graphEffectiveResultTarget = *resultTarget;
	so->graphEffectiveSearchEf = *searchEf;
	so->graphHighdimWideningMultiplier =
		requestedBaseTarget > 0 ?
		((double) *resultTarget / (double) requestedBaseTarget) : 1.0;

	*results = palloc(sizeof(PgturbohybridGraphResult) * *resultTarget);
	*count = PgturbohybridGraphRunTraversalPass(scan, so, meta, storage,
												 *results, *resultTarget,
												 *searchEf, query,
												 payloadSlot, payloadValue,
												 hasPayloadFilter,
												 payloadExactBandMissed,
												 estimatedSelectivity,
												 PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_ADAPTIVE_WIDENING);
	return true;
}

static double
PgturbohybridGraphResidualSketchSimilarity(const uint8 *querySketch,
								const uint8 *nodeSketch, int bytes)
{
	double		dot = 0.0;
	double		queryNorm = 0.0;
	double		nodeNorm = 0.0;

	for (int i = 0; i < bytes; i++)
	{
		double		q = (double) ((int) querySketch[i] - 128);
		double		n = (double) ((int) nodeSketch[i] - 128);

		dot += q * n;
		queryNorm += q * q;
		nodeNorm += n * n;
	}

	if (queryNorm <= 0.0 || nodeNorm <= 0.0)
		return 0.0;
	return dot / sqrt(queryNorm * nodeNorm);
}

static double
PgturbohybridGraphResidualBandSpread(PgturbohybridGraphResult *results,
						  int band, int boundaryIndex)
{
	double		best;
	double		boundary;
	double		full;
	double		q1;
	double		q3;
	double		robust;
	double		spread;

	if (band <= 1)
		return 0.0;

	best = results[0].distance;
	boundary = results[Min(Max(boundaryIndex, 1), band - 1)].distance;
	full = results[band - 1].distance - best;
	q1 = results[band / 4].distance;
	q3 = results[(band * 3) / 4].distance;
	robust = q3 - q1;
	spread = Max(boundary - best, robust);
	spread = Max(spread, full);

	if (!isfinite(spread) || spread <= 0.0)
		return 0.0;
	return spread;
}

static int
PgturbohybridGraphApplyResidualRerank(PgturbohybridGraphScanOpaque so,
						   PgturbohybridGraphMetaPageData *meta,
						   PgturbohybridGraphScanStorage *storage,
						   PgturbohybridGraphResult *results, int count)
{
	uint8		querySketch[PGTURBOHYBRID_GRAPH_MAX_RESIDUAL_RERANK_BYTES] = {0};
	int			bytes = meta->tqResidualRerankBytes;
	int			band;
	int			topK;
	int			modeSetting = pgturbohybrid_dense_residual_rerank_mode;
	int64		baseTarget;
	uint32	   *beforeBandNodeIds = NULL;
	uint32	   *beforeTopKNodeIds = NULL;
	int			beforeBandNodeIdCount = 0;
	int			beforeTopKNodeIdCount = 0;
	double		spread = 0.0;
	double		effectiveWeight = 0.0;
	double		maxAdjustmentAllowed = 0.0;
	double		maxAdjustmentApplied = 0.0;
	instr_time	start;
	TqScoreMode mode = (TqScoreMode) so->tq.scoreMode;

	so->graphResidualRerankMode = modeSetting;
	so->graphResidualRerankCount = 0;
	so->graphResidualRerankBytes = 0;
	so->graphResidualRerankWeightEffective = 0.0;
	so->graphResidualRerankBand = 0;
	so->graphResidualRerankBandMultiplier = 0;
	so->graphResidualRerankMaxAdjustment = 0.0;
	so->graphResidualRerankReorderedCount = 0;
	so->graphResidualRerankTopKChanged = false;
	if (bytes <= 0 || bytes > PGTURBOHYBRID_GRAPH_MAX_RESIDUAL_RERANK_BYTES ||
		count <= 1 || so->tq.rawQueryValues == NULL ||
		modeSetting == PGTURBOHYBRID_DENSE_RESIDUAL_RERANK_OFF)
		return count;
	if (mode != PGTURBOHYBRID_SCORE_COSINE && mode != PGTURBOHYBRID_SCORE_IP)
		return count;

	baseTarget = so->hasTupleTargetRows ?
		Max((int64) 1, so->tupleTargetRows) :
		Max((int64) 1, so->graphDenseRequestedK);
	band = (int) Min((int64) count,
					 Max(baseTarget *
						 (int64) pgturbohybrid_dense_residual_rerank_band_multiplier,
						 (int64) 20));
	if (band <= 1)
		return count;
	topK = (int) Min((int64) band, baseTarget);

	if (modeSetting == PGTURBOHYBRID_DENSE_RESIDUAL_RERANK_FIXED)
		effectiveWeight = 0.03;
	else
	{
		spread = PgturbohybridGraphResidualBandSpread(results, band,
													  topK - 1);
		effectiveWeight =
			pgturbohybrid_dense_residual_rerank_weight >= 0.0 ?
			pgturbohybrid_dense_residual_rerank_weight : 1.0;
		if (spread <= 0.0 || effectiveWeight <= 0.0 ||
			pgturbohybrid_dense_residual_rerank_max_adjust_ratio <= 0.0)
			effectiveWeight = 0.0;
		maxAdjustmentAllowed =
			spread * pgturbohybrid_dense_residual_rerank_max_adjust_ratio;
	}

	INSTR_TIME_SET_CURRENT(start);
	beforeBandNodeIdCount = PgturbohybridGraphCaptureTopNodeIds(results, count,
																band,
																&beforeBandNodeIds);
	beforeTopKNodeIdCount = PgturbohybridGraphCaptureTopNodeIds(results, count,
																topK,
																&beforeTopKNodeIds);
	PgturbohybridGraphBuildResidualSketch(so->tq.rawQueryValues, meta->dimensions,
										  querySketch, bytes);
	for (int i = 0; i < band; i++)
	{
		PgturbohybridGraphScanNode *node;
		double		similarity;
		double		adjustment;

		CHECK_FOR_INTERRUPTS();
		if (results[i].nodeId >= meta->tqNodeCount)
			continue;
		node = &storage->nodes[results[i].nodeId];
		if (node->residualSketch == NULL)
			continue;

		similarity = PgturbohybridGraphResidualSketchSimilarity(querySketch,
														 node->residualSketch,
														 bytes);
		if (modeSetting == PGTURBOHYBRID_DENSE_RESIDUAL_RERANK_FIXED)
			adjustment = 0.03 * similarity;
		else
		{
			adjustment = similarity * effectiveWeight * spread;
			if (maxAdjustmentAllowed > 0.0)
			{
				if (adjustment > maxAdjustmentAllowed)
					adjustment = maxAdjustmentAllowed;
				else if (adjustment < -maxAdjustmentAllowed)
					adjustment = -maxAdjustmentAllowed;
			}
			else
				adjustment = 0.0;
		}
		results[i].distance -= adjustment;
		maxAdjustmentApplied = Max(maxAdjustmentApplied, fabs(adjustment));
		results[i].exactScored = false;
	}
	qsort(results, band, sizeof(PgturbohybridGraphResult), PgturbohybridGraphResultCompare);
	if (beforeBandNodeIds != NULL)
	{
		for (int i = 0; i < beforeBandNodeIdCount; i++)
		{
			if (results[i].nodeId != beforeBandNodeIds[i])
				so->graphResidualRerankReorderedCount++;
		}
		pfree(beforeBandNodeIds);
	}
	so->graphResidualRerankTopKChanged =
		PgturbohybridGraphTopNodeIdsChanged(results, count, beforeTopKNodeIds,
											beforeTopKNodeIdCount);
	if (beforeTopKNodeIds != NULL)
		pfree(beforeTopKNodeIds);
	so->graphResidualRerankCount = band;
	so->graphResidualRerankBytes = bytes;
	so->graphResidualRerankWeightEffective = effectiveWeight;
	so->graphResidualRerankBand = band;
	so->graphResidualRerankBandMultiplier =
		pgturbohybrid_dense_residual_rerank_band_multiplier;
	so->graphResidualRerankMaxAdjustment = maxAdjustmentApplied;
	so->graphExactRescoreSource = PGTURBOHYBRID_EXACT_RESCORE_SOURCE_RESIDUAL;
	PgturbohybridGraphAddElapsedUs(&so->graphResidualRerankUs, start);
	return count;
}

static int
PgturbohybridGraphEffectiveHeapRescoreMode(PgturbohybridGraphScanOpaque so,
							   PgturbohybridGraphMetaPageData *meta,
							   bool exactFree)
{
	bool		lowDim = meta != NULL &&
		meta->dimensions > 0 &&
		meta->dimensions <= PGTURBOHYBRID_GRAPH_AUTO_HEAP_RESCORE_MAX_DIMENSIONS;
	/*
	 * Low-bit (1/2-bit) codes are too lossy to rank on directly, so the auto
	 * profiles rescore them at any dimension (Qdrant's bit-width-keyed default).
	 * 4-bit -- the default -- stays code-only unless lowDim or an explicit GUC.
	 */
	bool		lowBit = meta != NULL &&
		meta->tqBits > 0 &&
		meta->tqBits <= PGTURBOHYBRID_GRAPH_LOWBIT_HEAP_RESCORE_MAX_BITS;
	bool		rescoreAuto = lowDim || lowBit;

	if (!exactFree)
	{
		so->graphHeapRescoreReason =
			PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_EXACT_STORAGE;
		so->graphHeapRescoreAutoEnabled = false;
		return PGTURBOHYBRID_DENSE_HEAP_RESCORE_OFF;
	}

	if (pgturbohybrid_dense_heap_rescore_user_set)
	{
		so->graphHeapRescoreAutoEnabled = false;
		switch ((TqDenseHeapRescoreMode) pgturbohybrid_dense_heap_rescore)
		{
			case PGTURBOHYBRID_DENSE_HEAP_RESCORE_TOPK:
				so->graphHeapRescoreReason =
					PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_EXPLICIT_TOPK;
				break;
			case PGTURBOHYBRID_DENSE_HEAP_RESCORE_BAND:
				so->graphHeapRescoreReason =
					PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_EXPLICIT_BAND;
				break;
			case PGTURBOHYBRID_DENSE_HEAP_RESCORE_OFF:
			default:
				so->graphHeapRescoreReason =
					PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_EXPLICIT_OFF;
				break;
		}
		return pgturbohybrid_dense_heap_rescore;
	}

	switch ((PgturbohybridProfile) pgturbohybrid_profile)
	{
		case PGTURBOHYBRID_PROFILE_BALANCED:
			so->graphHeapRescoreReason = lowDim ?
				PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_BALANCED_LOWDIM :
				(lowBit ? PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_LOWBIT :
				 PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_BALANCED_HIGHDIM);
			so->graphHeapRescoreAutoEnabled = rescoreAuto;
			return rescoreAuto ? PGTURBOHYBRID_DENSE_HEAP_RESCORE_TOPK :
				PGTURBOHYBRID_DENSE_HEAP_RESCORE_OFF;
		case PGTURBOHYBRID_PROFILE_MATCHED_RECALL:
			so->graphHeapRescoreReason = lowDim ?
				PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_MATCHED_RECALL_LOWDIM :
				(lowBit ? PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_LOWBIT :
				 PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_MATCHED_RECALL_HIGHDIM);
			so->graphHeapRescoreAutoEnabled = rescoreAuto;
			return rescoreAuto ? PGTURBOHYBRID_DENSE_HEAP_RESCORE_TOPK :
				PGTURBOHYBRID_DENSE_HEAP_RESCORE_OFF;
		case PGTURBOHYBRID_PROFILE_HIGH_RECALL:
			/*
			 * high_recall is the exact-free, spend-the-latency-headroom
			 * profile: default to full band heap rescore at every dimension so
			 * 4-bit code candidates are re-ranked against exact heap vectors.
			 * An explicit turbohybrid.dense_heap_rescore still overrides this
			 * (handled above via the user_set branch).
			 */
			so->graphHeapRescoreReason =
				PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_HIGH_RECALL;
			so->graphHeapRescoreAutoEnabled = true;
			return PGTURBOHYBRID_DENSE_HEAP_RESCORE_BAND;
		case PGTURBOHYBRID_PROFILE_QUALITY:
		case PGTURBOHYBRID_PROFILE_DEBUG:
			so->graphHeapRescoreReason = lowDim ?
				PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_QUALITY_LOWDIM :
				(lowBit ? PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_LOWBIT :
				 PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_QUALITY_HIGHDIM);
			so->graphHeapRescoreAutoEnabled = rescoreAuto;
			return rescoreAuto ? PGTURBOHYBRID_DENSE_HEAP_RESCORE_TOPK :
				PGTURBOHYBRID_DENSE_HEAP_RESCORE_OFF;
		case PGTURBOHYBRID_PROFILE_LATENCY:
		default:
			so->graphHeapRescoreReason =
				PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_LATENCY;
			so->graphHeapRescoreAutoEnabled = false;
			return PGTURBOHYBRID_DENSE_HEAP_RESCORE_OFF;
	}
}

static int
PgturbohybridGraphHeapRescoreLimit(PgturbohybridGraphScanOpaque so, int count)
{
	int64		baseTarget;

	if (count <= 0 ||
		so->graphHeapRescoreMode == PGTURBOHYBRID_DENSE_HEAP_RESCORE_OFF)
		return 0;

	if (so->graphHeapRescoreMode == PGTURBOHYBRID_DENSE_HEAP_RESCORE_BAND)
		return count;

	baseTarget = so->graphFinalK > 0 ?
		so->graphFinalK : (so->hasTupleTargetRows ?
		Max((int64) 1, so->tupleTargetRows) :
		(int64) PGTURBOHYBRID_DEFAULT_FINAL_K);
	if (baseTarget <= 0)
		baseTarget = PGTURBOHYBRID_DEFAULT_FINAL_K;

	return (int) Min((int64) count, baseTarget);
}

static int
PgturbohybridGraphHeapRescore(IndexScanDesc scan, PgturbohybridGraphScanOpaque so,
				   Datum query, PgturbohybridGraphResult *results, int count)
{
	TupleTableSlot *slot;
	TupleDesc	desc;
	AttrNumber	denseAttno;
	int			limit;
	int			rescored = 0;
	instr_time	start;

	limit = PgturbohybridGraphHeapRescoreLimit(so, count);
	if (limit <= 0 || DatumGetPointer(query) == NULL ||
		scan == NULL || scan->heapRelation == NULL ||
		scan->indexRelation == NULL || scan->indexRelation->rd_index == NULL)
		return 0;

	denseAttno =
		scan->indexRelation->rd_index->indkey.values[PGTURBOHYBRID_DENSE_KEY_INDEX];
	desc = RelationGetDescr(scan->heapRelation);
	if (denseAttno <= 0 || denseAttno > desc->natts)
		return 0;

	INSTR_TIME_SET_CURRENT(start);
	slot = table_slot_create(scan->heapRelation, NULL);
	for (int i = 0; i < limit; i++)
	{
		Datum		value;
		bool		isnull;
		bool		visible;
		char	   *valuePtr;
		char	   *vectorPtr;
		instr_time	fetchStart;

		CHECK_FOR_INTERRUPTS();
		if (results[i].exactScored)
			continue;

		INSTR_TIME_SET_CURRENT(fetchStart);
		visible = table_tuple_fetch_row_version(scan->heapRelation,
												&results[i].heaptid,
												scan->xs_snapshot,
												slot);
		PgturbohybridGraphAddElapsedUs(&so->graphHeapFetchUs, fetchStart);
		if (!visible)
		{
			ExecClearTuple(slot);
			continue;
		}

		value = slot_getattr(slot, denseAttno, &isnull);
		if (isnull)
		{
			ExecClearTuple(slot);
			continue;
		}

		valuePtr = DatumGetPointer(value);
		vectorPtr = (char *) PG_DETOAST_DATUM(value);
		results[i].distance =
			PgturbohybridGraphExactVectorDistance(so, query, vectorPtr);
		results[i].exactScored = true;
		rescored++;
		if (vectorPtr != valuePtr)
			pfree(vectorPtr);
		ExecClearTuple(slot);
	}
	ExecDropSingleTupleTableSlot(slot);

	if (rescored > 0)
	{
		so->graphHeapRescoreCount += rescored;
		so->graphExactRescoreSource = PGTURBOHYBRID_EXACT_RESCORE_SOURCE_HEAP;
		PgturbohybridGraphAddElapsedUs(&so->graphHeapRescoreUs, start);
		return limit;
	}
	PgturbohybridGraphAddElapsedUs(&so->graphHeapRescoreUs, start);
	return 0;
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
	PgturbohybridGraphCacheInitInfo cacheInfo;
	int64		activeTarget;
	double		estimatedSelectivity;
	int			rescoreCount;
	int			finalCount;
	AttrNumber	payloadHeapAttno = InvalidAttrNumber;
	int32		payloadValue = 0;
	int			payloadSlot = -1;
	int			candidateOversampling;
	bool		hasPayloadFilter = false;
	bool		payloadExactBandMissed = false;
	bool		exactFree;
	bool		highDimL2Widened = false;
	bool		latencyBudgetActive = false;
	int64		requestedBaseTarget;
	int			adaptiveMode;

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
	if (meta.tqMultivectorGraphMode ==
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("document-node multivector indexes require multivector search"),
				 errdetail("This index declares multivector_graph = document_nodes, but the current scan is using the single-vector graph path."),
				 errhint("Use turbohybrid_query(multivector_query => ...), or REINDEX with multivector_graph = token_nodes for token-node compatibility.")));

	exactFree = (meta.tqFlags & PGTURBOHYBRID_GRAPH_EXACT_FREE) != 0 ||
		!BlockNumberIsValid(meta.tqExactStartBlkno);
	so->graphM = meta.m;
	so->graphEfConstruction = meta.efConstruction;
	so->graphExactStorage = !exactFree;
	so->graphBuildExactDistances =
		(meta.tqFlags & PGTURBOHYBRID_GRAPH_TQ_EXACT_BUILD) != 0;
	so->graphBuildDistanceMode = so->graphBuildExactDistances ?
		PGTURBOHYBRID_DENSE_BUILD_DISTANCE_EXACT :
		PGTURBOHYBRID_DENSE_BUILD_DISTANCE_CODE;
	so->graphBuildFastEdges =
		(meta.tqFlags & PGTURBOHYBRID_GRAPH_TQ_FAST_BUILD_EDGES) != 0;
	so->graphBuildNeighborSelectReason =
		PGTURBOHYBRID_GRAPH_TQ_BUILD_NEIGHBOR_REASON(meta.tqFlags);
	so->graphHeapRescoreMode =
		PgturbohybridGraphEffectiveHeapRescoreMode(so, &meta, exactFree);
	PgturbohybridGraphPrepareTqQueryWithBits(scan->indexRelation, &so->support, query,
							   &so->tq,
							   meta.tqBits != 0 ? meta.tqBits : PGTURBOHYBRID_DEFAULT_BITS);
	activeTarget = PgturbohybridGraphGetActiveLimitTupleTarget();
	if (activeTarget < 0)
	{
		int			plannedLimit = PgturbohybridCurrentLimit();

		if (plannedLimit > 0)
			activeTarget = plannedLimit;
	}
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
		resultTarget = minResultTarget > 0 ?
			minResultTarget : PGTURBOHYBRID_DEFAULT_FINAL_K;

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
		so->graphDenseFilterUnmapped = true;
		so->graphDenseLinearFallbackRatio =
			meta.tqNodeCount > 0 ?
			(double) resultTarget / (double) meta.tqNodeCount : 0.0;
		so->graphDenseLinearFallbackWarning =
			so->graphDenseLinearFallbackRatio >=
			pgturbohybrid_linear_fallback_notice_threshold_ratio;
		if (pgturbohybrid_warn_linear_fallback &&
			so->graphDenseLinearFallbackWarning)
			ereport(DEBUG1,
					(errmsg("turbohybrid dense graph scan widened for unmapped heap filter"),
					 errdetail("node_count=%u result_target=%d estimated_selectivity=%.6g max_scan_tuples=%d ratio=%.6g threshold_ratio=%.6g",
							   meta.tqNodeCount, resultTarget,
							   estimatedSelectivity,
							   pgturbohybrid_max_scan_tuples,
							   so->graphDenseLinearFallbackRatio,
							   pgturbohybrid_linear_fallback_notice_threshold_ratio),
					 errhint("Add the filter column as an INCLUDE int4 payload where possible, or lower candidate budgets/max_scan_tuples.")));
	}

	requestedBaseTarget = Max((int64) 1,
							  minResultTarget > 0 ?
							  (int64) minResultTarget :
							  (int64) PGTURBOHYBRID_DEFAULT_FINAL_K);
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
	searchEf = PgturbohybridGraphScaleSearchEfForSegments(so, &meta, searchEf);
	so->graphEffectiveResultTarget = resultTarget;
	so->graphEffectiveSearchEf = searchEf;
	results = palloc(sizeof(PgturbohybridGraphResult) * resultTarget);
	PgturbohybridGraphInitScanStorage(scan->indexRelation, &meta, &storage, &cacheInfo);
	/*
	 * Per-backend native scan-cache provenance for this scan: which cache mode
	 * served it, whether the per-backend cache was built during this scan (the
	 * cold-build cost that the prewarm A/B comparison isolates), and the
	 * resident footprint each backend duplicates.
	 */
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
	/*
	 * Whole-code prefetch in the batch scorer pays off only once the code arena
	 * is too big for CPU cache (codes become scattered RAM reads); below that
	 * the extra prefetches are wasted work.  64MB is comfortably above typical
	 * L3 and below large indexes (1M x 3072-dim 4-bit ~= 1.5GB).
	 */
	so->graphCodeArenaEstimatedBytes =
		(int64) ((Size) meta.tqNodeCount * (Size) meta.tqCodeBytes);
	so->graphLargeCodeArena =
		(Size) so->graphCodeArenaEstimatedBytes > ((Size) 64 * 1024 * 1024);
	PgturbohybridGraphAddElapsedUs(&so->graphPrepareUs, phaseStart);

	if (hasPayloadFilter)
	{
		if (PgturbohybridGraphCollectPayloadExactBand(so, &meta, &storage, query,
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
		payloadExactBandMissed = true;
	}

	count = PgturbohybridGraphRunTraversalPass(scan, so, &meta, &storage,
											   results, resultTarget, searchEf,
											   query, payloadSlot, payloadValue,
											   hasPayloadFilter,
											   payloadExactBandMissed,
											   estimatedSelectivity,
											   PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_ESTIMATED_SELECTIVITY);

	so->graphAdaptiveInitialSearchEf = searchEf;
	so->graphAdaptiveFinalSearchEf = searchEf;
	so->graphAdaptiveInitialResultTarget = resultTarget;
	so->graphAdaptiveFinalResultTarget = resultTarget;
	adaptiveMode =
		PgturbohybridGraphEffectiveAdaptiveWideningMode(&meta, exactFree,
														PgturbohybridGraphAdaptiveFinalTarget(so,
																 requestedBaseTarget));
	if (PgturbohybridGraphShouldAdaptiveWiden(so, results, count, resultTarget,
								   requestedBaseTarget, adaptiveMode))
	{
		int64		adaptiveTarget =
			(int64) ceil((double) resultTarget *
						 PgturbohybridGraphEffectiveAdaptiveWideningMultiplier());
		int64		adaptiveMax =
			(int64) ceil((double) resultTarget *
						 PgturbohybridGraphEffectiveAdaptiveWideningMaxMultiplier());
		int64		scanCap = pgturbohybrid_max_scan_tuples > 0 ?
			(int64) pgturbohybrid_max_scan_tuples : (int64) meta.tqNodeCount;
		int			wideTarget;

		adaptiveMax = Max(adaptiveMax, (int64) resultTarget + 1);
		adaptiveTarget = Min(adaptiveTarget, adaptiveMax);
		adaptiveTarget = Min(adaptiveTarget, scanCap);
		adaptiveTarget = Min(adaptiveTarget, (int64) meta.tqNodeCount);
		wideTarget = (int) Min((int64) INT_MAX,
							   Max(adaptiveTarget, (int64) resultTarget + 1));
		if (wideTarget > resultTarget)
		{
			pfree(results);
			resultTarget = wideTarget;
			searchEf = Min(Max(searchEf + 1, resultTarget), (int) meta.tqNodeCount);
			so->graphAdaptiveTriggered = true;
			so->graphAdaptiveFinalResultTarget = resultTarget;
			so->graphAdaptiveFinalSearchEf = searchEf;
			so->graphEffectiveResultTarget = resultTarget;
			so->graphEffectiveSearchEf = searchEf;
			so->graphHighdimWideningMultiplier =
				requestedBaseTarget > 0 ?
				((double) resultTarget / (double) requestedBaseTarget) : 1.0;
			results = palloc(sizeof(PgturbohybridGraphResult) * resultTarget);
			INSTR_TIME_SET_CURRENT(phaseStart);
			count = PgturbohybridGraphTraverse(scan->indexRelation, so, &meta, &storage,
									results, resultTarget, searchEf, query,
									payloadSlot, payloadValue);
			PgturbohybridGraphAddElapsedUs(&so->graphTraverseUs, phaseStart);
			if (estimatedSelectivity > 0 && estimatedSelectivity < 1 && count < resultTarget)
			{
				INSTR_TIME_SET_CURRENT(phaseStart);
				count = PgturbohybridGraphFillCandidateBand(scan->indexRelation, so, &meta,
											 &storage, results, resultTarget, count,
											 payloadSlot, payloadValue, query,
											 PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_ADAPTIVE_WIDENING);
				PgturbohybridGraphAddElapsedUs(&so->graphFillUs, phaseStart);
			}
			INSTR_TIME_SET_CURRENT(phaseStart);
			qsort(results, count, sizeof(PgturbohybridGraphResult), PgturbohybridGraphResultCompare);
			PgturbohybridGraphAddElapsedUs(&so->graphSortUs, phaseStart);
		}
	}

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
												 query,
												 PGTURBOHYBRID_GRAPH_FILL_CANDIDATE_BAND_REASON_TIGHT_L2_EXACT_POLICY);
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
		int			heapSortCount;
		bool		residualReordered = false;
		bool		heapReordered = false;
		int			orderLimit = (int) Min((int64) INT_MAX,
										   PgturbohybridGraphAdaptiveFinalTarget(so,
																 requestedBaseTarget));
		uint32	   *beforeNodeIds = NULL;
		int			beforeNodeIdCount = 0;

		beforeNodeIdCount = PgturbohybridGraphCaptureTopNodeIds(results, count,
																orderLimit,
																&beforeNodeIds);
		count = PgturbohybridGraphApplyResidualRerank(so, &meta, &storage,
										   results, count);
		residualReordered = PgturbohybridGraphTopNodeIdsChanged(results, count,
																beforeNodeIds,
																beforeNodeIdCount);
		if (beforeNodeIds != NULL)
			pfree(beforeNodeIds);
		beforeNodeIdCount = PgturbohybridGraphCaptureTopNodeIds(results, count,
																orderLimit,
																&beforeNodeIds);
		heapSortCount = PgturbohybridGraphHeapRescore(scan, so, query, results, count);
		if (heapSortCount > 1)
		{
			INSTR_TIME_SET_CURRENT(phaseStart);
			qsort(results, heapSortCount, sizeof(PgturbohybridGraphResult),
				  PgturbohybridGraphResultCompare);
			PgturbohybridGraphAddElapsedUs(&so->graphSortUs, phaseStart);
		}
		heapReordered = PgturbohybridGraphTopNodeIdsChanged(results, count,
															beforeNodeIds,
															beforeNodeIdCount);
		if (beforeNodeIds != NULL)
			pfree(beforeNodeIds);
		if (PgturbohybridGraphApplyUncertaintyRetry(scan, so, &meta, &storage,
													&results, &resultTarget,
													&searchEf, &count, query,
													payloadSlot, payloadValue,
													hasPayloadFilter,
													payloadExactBandMissed,
													estimatedSelectivity,
													requestedBaseTarget,
													residualReordered,
													heapReordered))
		{
			count = PgturbohybridGraphApplyResidualRerank(so, &meta, &storage,
											   results, count);
			heapSortCount = PgturbohybridGraphHeapRescore(scan, so, query,
														  results, count);
			if (heapSortCount > 1)
			{
				INSTR_TIME_SET_CURRENT(phaseStart);
				qsort(results, heapSortCount, sizeof(PgturbohybridGraphResult),
					  PgturbohybridGraphResultCompare);
				PgturbohybridGraphAddElapsedUs(&so->graphSortUs, phaseStart);
			}
			so->graphCandidateCount = count;
		}
		so->graphEffectiveRescoreBand = heapSortCount > 0 ?
			heapSortCount : so->graphResidualRerankCount;
		so->tqGraphResults = results;
		so->tqGraphResultCount = count;
		so->tqGraphResultIndex = 0;
		PgturbohybridGraphAddElapsedUs(&so->graphTotalUs, totalStart);
		PgturbohybridGraphRecordGraphScanStats(so);
		return;
	}
	(void) PgturbohybridGraphApplyUncertaintyRetry(scan, so, &meta, &storage,
												   &results, &resultTarget,
												   &searchEf, &count, query,
												   payloadSlot, payloadValue,
												   hasPayloadFilter,
												   payloadExactBandMissed,
												   estimatedSelectivity,
												   requestedBaseTarget,
												   false, false);
	so->graphCandidateCount = count;
	rescoreCount = PgturbohybridGraphFinalRescoreCount(so, results, count, effectiveEf);
	so->graphEffectiveRescoreBand = rescoreCount;
	finalCount = so->graphRescoreBand == PGTURBOHYBRID_GRAPH_RESCORE_BAND_AUTO &&
		rescoreCount > 0 ? rescoreCount : count;
	INSTR_TIME_SET_CURRENT(phaseStart);
	PgturbohybridGraphExactRescore(scan->indexRelation, so, query, &meta, storage.nodes,
						results, rescoreCount);
	if (so->graphRescoreCount > 0)
		so->graphExactRescoreSource = PGTURBOHYBRID_EXACT_RESCORE_SOURCE_INDEX_EXACT;
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
		instr_time	lockStart;

		INSTR_TIME_SET_CURRENT(lockStart);
		LockPage(scan->indexRelation, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ShareLock);
		so->graphScanLockWaitUs += PgturbohybridGraphElapsedUs(lockStart);
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
		stats->dense.candidatesRequested = targetK > 0 ? targetK : limit;
		stats->effectiveResultTarget = (uint32) Max(so->graphEffectiveResultTarget, 0);
		stats->effectiveSearchEf = (uint32) Max(so->graphEffectiveSearchEf, 0);
		stats->effectiveRescoreBand = (uint32) Max(so->graphEffectiveRescoreBand, 0);
		stats->highdimWideningMultiplier = so->graphHighdimWideningMultiplier;
		stats->wideningReason = so->graphWideningReason;
		stats->dense.budgetPolicy = so->graphDenseBudgetPolicy;
			stats->rescoreBandPolicy = so->graphRescoreBandPolicy;
			stats->dense.candidatesReturned = limit;
			stats->exactRescoreCount = so->graphRescoreCount;
			stats->heapRescoreCount = so->graphHeapRescoreCount;
			stats->heapRescoreAutoEnabled =
				so->graphHeapRescoreAutoEnabled;
			stats->heapRescoreReason = so->graphHeapRescoreReason;
			stats->codePagesRead = so->graphCodePagesRead;
			stats->adjPagesRead = so->graphAdjPagesRead;
			stats->prepareUs = so->graphPrepareUs;
		stats->traverseUs = so->graphTraverseUs;
		stats->entryUs = so->graphEntryUs;
			stats->baseUs = so->graphBaseUs;
			stats->batchUs = so->graphBatchUs;
			stats->heapUs = so->graphHeapUs;
			stats->heapFetchUs = so->graphHeapFetchUs;
			stats->heapRescoreUs = so->graphHeapRescoreUs;
			stats->fillUs = so->graphFillUs;
			stats->fillCandidateBandCalls = so->graphFillCandidateBandCalls;
			stats->fillCandidateBandReason = so->graphFillCandidateBandReason;
			stats->fillCandidateBandVisited = so->graphFillCandidateBandVisited;
			stats->fillCandidateBandScored = so->graphFillCandidateBandScored;
			stats->fillCandidateBandSelectedBefore =
				so->graphFillCandidateBandSelectedBefore;
			stats->fillCandidateBandSelectedAfter =
				so->graphFillCandidateBandSelectedAfter;
			stats->fillCandidateBandTarget = so->graphFillCandidateBandTarget;
			stats->fillCandidateBandUsedPayloadRefs =
				so->graphFillCandidateBandUsedPayloadRefs;
			stats->fillCandidateBandPayloadRefCount =
				so->graphFillCandidateBandPayloadRefCount;
			stats->rescoreUs = so->graphRescoreUs;
			stats->sortUs = so->graphSortUs;
			stats->exactRescoreSource = so->graphExactRescoreSource;
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

		{
			instr_time	lockStart;

			INSTR_TIME_SET_CURRENT(lockStart);
			LockPage(scan->indexRelation, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ShareLock);
			so->graphScanLockWaitUs += PgturbohybridGraphElapsedUs(lockStart);
		}
		PG_TRY();
		{
			PgturbohybridGraphCollectResults(scan, so, 0);
		}
		PG_CATCH();
		{
			UnlockPage(scan->indexRelation, PGTURBOHYBRID_GRAPH_SCAN_LOCK, ShareLock);
			PG_RE_THROW();
		}
		PG_END_TRY();
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

	if (so->pgturbohybridGraphScan)
		PgturbohybridGraphRecordReturnedRows(so->returnedRows);

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
	const PgturbohybridGraphTypeInfo *typeInfo;
	PgturbohybridGraphSupport support;

	(void) heap;
	(void) checkUnique;
#if PG_VERSION_NUM >= 140000
	(void) indexUnchanged;
#endif
	if (isnull[0])
		return false;

	if (PgturbohybridGraphIndexIsMultiVector(index))
	{
		LockPage(index, PGTURBOHYBRID_GRAPH_UPDATE_LOCK, ExclusiveLock);
		PG_TRY();
		{
			PgturbohybridGraphInsertMultiVectorInPlace(index, indexInfo, heap_tid,
													   values[0], values, isnull,
													   NULL);
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

	typeInfo = PgturbohybridGraphGetTypeInfo(index);
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
