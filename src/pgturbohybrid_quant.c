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

typedef struct PgturbohybridNativeParallelShared PgturbohybridNativeParallelShared;

static int	PgturbohybridGraphResultCompare(const void *a, const void *b);
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
static bool PgturbohybridGraphUseFastBuildEdges(PgturbohybridQuantBuildState *state);
static void PgturbohybridNativeParallelEdgeWorker(Relation indexRel,
												  PgturbohybridNativeParallelShared *shared,
												  bool leader);
static int64 PgturbohybridGraphElapsedUs(instr_time start);
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
static bool PgturbohybridGraphMultiVectorDocMapHasContexts(PgturbohybridQuantBuildState *state);
static uint32 PgturbohybridMultiVectorQuantizedInvertedCodeword(const PgturbohybridMultiVector *mv,
												  int32 token);
static uint16 PgturbohybridMultiVectorQuantizedInvertedScorePayload(const PgturbohybridMultiVector *mv,
												  int32 token);

#define PARALLEL_KEY_PGTURBOHYBRID_NATIVE_SHARED	UINT64CONST(0xA000000000000011)
#define PARALLEL_KEY_PGTURBOHYBRID_NATIVE_QUERY		UINT64CONST(0xA000000000000012)
#define PGTURBOHYBRID_GRAPH_AUTO_HEURISTIC_MAX_DIMENSIONS 256
#define PGTURBOHYBRID_GRAPH_AUTO_EXACT_BUILD_MAX_DIMENSIONS 256
#define PGTURBOHYBRID_GRAPH_AUTO_HEAP_RESCORE_MAX_DIMENSIONS 256
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
	uint32		step;
	uint32		start;
	double		overlapSum = 0.0;
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

	candidateLimit = Min(candidateLimit, searchEf);
	candidateLimit = Min(candidateLimit, (int) meta.tqNodeCount);
	stats->candidateLimit = candidateLimit;

	memset(&so, 0, sizeof(so));
	PgturbohybridGraphInitScanStorage(index, &meta, &storage, NULL);
	candidates = palloc0(sizeof(*candidates) * candidateLimit);
	step = Max(1U, meta.tqNodeCount / (uint32) targetSamples);
	start = step > 1 ? step / 2 : 0;

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

			memset(candidates, 0, sizeof(*candidates) * candidateLimit);
			slot = PgturbohybridGraphScanAdjSlot(&meta, nodeId, 0);
			directCount = storage.neighborCounts[slot];
			candidateCount = PgturbohybridGraphRepairCollectCandidates(index, &so,
																	   &meta,
																	   &storage,
																	   nodeId,
																	   candidates,
																	   candidateLimit);
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

static Size
PgturbohybridGraphArrayAllocSize(Size elemSize, Size count)
{
	if (elemSize != 0 && count > MaxAllocSize / elemSize)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pgturbohybrid graph array is too large")));
	return elemSize * count;
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

		if (*capacity > PG_INT32_MAX / 2)
			newCapacity = maxCapacity;
		else
			newCapacity = Max(8, *capacity + *capacity);
		if (newCapacity < *capacity || newCapacity > maxCapacity)
			newCapacity = maxCapacity;

		*heap = repalloc(*heap,
						 PgturbohybridGraphArrayAllocSize(sizeof(PgturbohybridGraphFrontierItem),
														  newCapacity));
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
	Oid			multivectorOid = PgturbohybridMultiVectorTypeOid();
	TupleDesc	desc = RelationGetDescr(index);

	return OidIsValid(multivectorOid) &&
		TupleDescAttr(desc, PGTURBOHYBRID_DENSE_KEY_INDEX)->atttypid == multivectorOid;
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
	Vector	   *vector;

	indexedMv =
		PgturbohybridMultiVectorPoolDocumentTokens(mv,
												   state->multivectorTokenPooling,
												   state->multivectorTokenPoolingTargetRatio,
												   state->multivectorTokenPoolingMinTokens,
												   state->buildTupleCtx);
	vector =
		PgturbohybridMultiVectorBuildProxyVector(indexedMv,
												 state->multivectorProxyEncoder,
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
	targetRatio = (double) centroidCount / (double) doc->count;
	centroids =
		PgturbohybridMultiVectorPoolDocumentTokens(doc,
												   PGTURBOHYBRID_MULTIVECTOR_TOKEN_POOLING_KMEANS,
												   targetRatio,
												   1,
												   state->buildTupleCtx);
	centroidSize = VARSIZE_ANY(centroids);
	stored = MemoryContextAlloc(state->ctx, centroidSize);
	memcpy(stored, centroids, centroidSize);
	state->multivectorDocCentroids[docOrdinal] = stored;
	state->multivectorDocCentroidResiduals[docOrdinal] =
		PgturbohybridMultiVectorCentroidResidualMean(doc, stored);
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
	Size		mvSize;
	uint32		nodeId;
	uint32		docOrdinal;

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
	docEntry->tokenCount = (uint16) indexedMv->count;
	docEntry->originalTokenCount = (uint16) mv->count;
	docEntry->pooledTokenCount = (uint16) indexedMv->count;

	mvSize = VARSIZE_ANY(indexedMv);
	stored = MemoryContextAlloc(state->ctx, mvSize);
	memcpy(stored, indexedMv, mvSize);
	state->multivectorDocVectors[docOrdinal] = stored;
	PgturbohybridGraphStoreBuildMultiVectorCentroids(state, docOrdinal, stored);
	state->multivectorDocCount++;

	vector =
		PgturbohybridMultiVectorBuildProxyVector(indexedMv,
												 state->multivectorProxyEncoder,
												 state->buildTupleCtx);
	nodeId = PgturbohybridGraphAppendBuildNode(state, tid,
											   PointerGetDatum(vector),
											   values, isnull);
	state->multivectorNodeMap[nodeId].docId = docId;
	state->multivectorNodeMap[nodeId].tokenOrdinal = 0;
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
			MemoryContextSwitchTo(state->ctx);
			if (state->multivectorGraphMode ==
				PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES)
				PgturbohybridGraphAppendBuildMultiVectorDocument(state, tid, mv,
																 values, isnull);
			else
				PgturbohybridGraphAppendBuildMultiVector(state, tid, mv, values, isnull);
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
			PgturbohybridGraphAppendBuildNode(state, tid, value, values, isnull);
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
			if (pgturbohybrid_profile == PGTURBOHYBRID_PROFILE_BALANCED)
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
				state->buildEdgeSearchLayerUs += PgturbohybridGraphElapsedUs(searchStart);
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
			state->buildEdgeSearchLayerUs += PgturbohybridGraphElapsedUs(searchStart);
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
		return false;

	if (state->nodeCount < 2 || state->dimensions <= 0 || state->m <= 0 ||
		!state->buildCodeOnly || state->buildExactDistances ||
		state->tqExactStorage)
	{
		if (force)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("turbohybrid.native_parallel_edge_build=on requires a code-only native graph build"),
					 errdetail("Exact-storage and exact-distance build modes keep edge construction serial.")));
		return false;
	}
	if (PgturbohybridGraphGetNativeSegments(state->index) > 1)
	{
		if (force)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("turbohybrid.native_parallel_edge_build=on requires native_segments = 1"),
					 errdetail("Explicit multi-segment native graph builds keep the segment-local serial edge builder.")));
		return false;
	}

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
			state->buildEdgeSearchLayerUs += PgturbohybridGraphElapsedUs(searchStart);
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
		state->buildEdgeSearchLayerUs += PgturbohybridGraphElapsedUs(searchStart);
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
	shared->edgeSearchLayerUs += state.buildEdgeSearchLayerUs;
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
	update->tqMultivectorDocMapFlags =
		state->multivectorBuild &&
		state->multivectorGraphMode ==
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES &&
		BlockNumberIsValid(docMapStart) ?
		PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_DOC_VECTORS : 0;
	if (BlockNumberIsValid(docMapStart) &&
		PgturbohybridGraphMultiVectorDocMapHasContexts(state))
		update->tqMultivectorDocMapFlags |=
			PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_CONTEXTS;
	if (BlockNumberIsValid(docMapStart) &&
		state->multivectorGraphMode ==
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES)
		update->tqMultivectorDocMapFlags |=
			PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_QUANTIZED_POSTINGS;
	if (BlockNumberIsValid(docMapStart) &&
		state->multivectorGraphMode ==
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES &&
		state->multivectorCentroids ==
		PGTURBOHYBRID_MULTIVECTOR_CENTROIDS_KMEANS)
	{
		update->tqMultivectorDocMapFlags |=
			PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_CENTROIDS;
		update->tqMultivectorDocMapFlags |=
			PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_CENTROID_POSTINGS;
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

	metap->dimensions = update->dimensions;
	metap->m = update->m;
	metap->efConstruction = update->efConstruction;
	metap->storageKind = PGTURBOHYBRID_GRAPH_STORAGE_QUANT_GRAPH_NATIVE;
	metap->graphEfSearch = PgturbohybridGraphGetEfSearch(index);
	metap->graphOversampling = PgturbohybridGraphGetGraphOversampling(index);
	metap->graphRescoreBand = PgturbohybridGraphGetGraphRescoreBand(index);
	metap->graphMaxLevel = update->graphMaxLevel;
	metap->graphFlags = metap->graphFlags == 0 ? 1 : metap->graphFlags + 1;
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
	metap->graphFlags = metap->graphFlags == 0 ? 1 : metap->graphFlags + 1;
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
	uint16		maxPostingCount;

	if (state->multivectorCentroids !=
		PGTURBOHYBRID_MULTIVECTOR_CENTROIDS_KMEANS ||
		state->multivectorDocCentroids == NULL ||
		state->dimensions <= 0)
		return;

	codebookSize = (uint32) state->dimensions * 2U;
	listCounts = palloc0(sizeof(uint32) * (Size) codebookSize);
	listOffsets = palloc0(sizeof(uint32) * ((Size) codebookSize + 1));
	listWrite = palloc0(sizeof(uint32) * (Size) codebookSize);

	for (uint32 docId = 0; docId < state->multivectorDocCount; docId++)
	{
		PgturbohybridMultiVector *centroids =
			state->multivectorDocCentroids[docId];

		if (centroids == NULL)
			ereport(ERROR,
					(errcode(ERRCODE_INDEX_CORRUPTED),
					 errmsg("document-node centroid posting sidecar is missing document centroids")));
		if ((uint64) totalPostings + (uint64) centroids->count >
			(uint64) PG_UINT32_MAX)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("pgturbohybrid centroid posting sidecar is too large")));
		for (int32 centroid = 0; centroid < centroids->count; centroid++)
		{
			uint32		codeword =
				PgturbohybridMultiVectorQuantizedInvertedCodeword(centroids,
																  centroid);

			listCounts[codeword]++;
			totalPostings++;
		}
	}
	if ((uint64) totalPostings >
		(uint64) (MaxAllocSize /
				  sizeof(PgturbohybridGraphMultiVectorCentroidPostingEntry)))
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pgturbohybrid centroid posting sidecar is too large")));

	listOffsets[0] = 0;
	for (uint32 codeword = 0; codeword < codebookSize; codeword++)
	{
		listOffsets[codeword + 1] = listOffsets[codeword] + listCounts[codeword];
		listWrite[codeword] = listOffsets[codeword];
	}
	postings =
		palloc0(sizeof(PgturbohybridGraphMultiVectorCentroidPostingEntry) *
				(Size) Max(totalPostings, 1U));
	for (uint32 docId = 0; docId < state->multivectorDocCount; docId++)
	{
		PgturbohybridMultiVector *centroids =
			state->multivectorDocCentroids[docId];

		for (int32 centroid = 0; centroid < centroids->count; centroid++)
		{
			uint32		codeword =
				PgturbohybridMultiVectorQuantizedInvertedCodeword(centroids,
																  centroid);
			uint32		offset = listWrite[codeword]++;

			postings[offset].docId = docId;
			postings[offset].centroidOrdinal = (uint16) centroid;
			postings[offset].unused = 0;
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
	pfree(listWrite);
	pfree(listOffsets);
	pfree(listCounts);
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
	uint32	   *listCounts;
	uint32	   *listOffsets;
	uint32	   *listWrite;
	PgturbohybridGraphMultiVectorQuantizedPostingEntry *postings;
	uint32		totalPostings = 0;
	uint16		maxPostingCount;

	if (state->multivectorGraphMode !=
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES ||
		state->multivectorDocVectors == NULL ||
		state->dimensions <= 0)
		return;

	codebookSize = (uint32) state->dimensions * 2U;
	listCounts = palloc0(sizeof(uint32) * (Size) codebookSize);
	listOffsets = palloc0(sizeof(uint32) * ((Size) codebookSize + 1));
	listWrite = palloc0(sizeof(uint32) * (Size) codebookSize);

	for (uint32 docId = 0; docId < state->multivectorDocCount; docId++)
	{
		PgturbohybridMultiVector *doc =
			state->multivectorDocVectors[docId];

		if (doc == NULL)
			ereport(ERROR,
					(errcode(ERRCODE_INDEX_CORRUPTED),
					 errmsg("document-node quantized inverted sidecar is missing document vectors")));
		if ((uint64) totalPostings + (uint64) doc->count >
			(uint64) PG_UINT32_MAX)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("pgturbohybrid quantized inverted sidecar is too large")));
		for (int32 token = 0; token < doc->count; token++)
		{
			uint32		codeword =
				PgturbohybridMultiVectorQuantizedInvertedCodeword(doc, token);

			listCounts[codeword]++;
			totalPostings++;
		}
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
		listWrite[codeword] = listOffsets[codeword];
	}
	postings =
		palloc0(sizeof(PgturbohybridGraphMultiVectorQuantizedPostingEntry) *
				(Size) Max(totalPostings, 1U));
	for (uint32 docId = 0; docId < state->multivectorDocCount; docId++)
	{
		PgturbohybridMultiVector *doc =
			state->multivectorDocVectors[docId];

		for (int32 token = 0; token < doc->count; token++)
		{
			uint32		codeword =
				PgturbohybridMultiVectorQuantizedInvertedCodeword(doc, token);
			uint32		offset = listWrite[codeword]++;

			postings[offset].docId = docId;
			postings[offset].tokenOrdinal = (uint16) token;
			postings[offset].scorePayload =
				PgturbohybridMultiVectorQuantizedInvertedScorePayload(doc,
																	  token);
		}
	}

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

	pfree(postings);
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

	*pageCount = 0;
	*docMapBytes = 0;
	if (!state->multivectorBuild ||
		state->multivectorNodeMap == NULL ||
		state->multivectorDocMap == NULL ||
		state->nodeCount == 0 ||
		state->multivectorDocCount == 0)
		return InvalidBlockNumber;

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
		PGTURBOHYBRID_MULTIVECTOR_GRAPH_DOCUMENT_NODES)
	{
		uint16		maxVectorCount =
			PgturbohybridGraphMultiVectorDocMapVectorTupleMaxCount();
		uint16		maxCentroidCount =
			PgturbohybridGraphMultiVectorDocMapCentroidTupleMaxCount();

		for (uint32 docId = 0; docId < state->multivectorDocCount; docId++)
		{
			PgturbohybridMultiVector *mv = state->multivectorDocVectors != NULL ?
				state->multivectorDocVectors[docId] : NULL;
			Size		totalFloats;

			if (mv == NULL)
				ereport(ERROR,
						(errcode(ERRCODE_INDEX_CORRUPTED),
						 errmsg("document-node multivector sidecar is missing document vectors")));
			totalFloats = PgturbohybridMultiVectorFloatCount(mv->count, mv->dim);
			for (uint32 startFloat = 0; startFloat < totalFloats;)
			{
				uint16		count =
					(uint16) Min((Size) maxVectorCount,
								 totalFloats - (Size) startFloat);
				Size		tupleSize =
					PgturbohybridGraphMultiVectorDocMapVectorTupleSize(count);
				PgturbohybridGraphMultiVectorDocMapVectorTuple tuple =
					palloc0(tupleSize);

				CHECK_FOR_INTERRUPTS();
				tuple->type =
					PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VECTOR_TUPLE_TYPE;
				tuple->version = PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION;
				tuple->count = count;
				tuple->magic = PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC;
				tuple->docId = docId;
				tuple->startFloat = startFloat;
				memcpy(tuple->values, mv->values + startFloat,
					   sizeof(float) * count);
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
			if (state->multivectorCentroids ==
				PGTURBOHYBRID_MULTIVECTOR_CENTROIDS_KMEANS)
			{
				PgturbohybridMultiVector *centroids =
					state->multivectorDocCentroids != NULL ?
					state->multivectorDocCentroids[docId] : NULL;
				Size		centroidFloats;

				if (centroids == NULL)
					ereport(ERROR,
							(errcode(ERRCODE_INDEX_CORRUPTED),
							 errmsg("document-node centroid sidecar is missing document centroids")));
				PgturbohybridCheckSameMultiVectorDims(mv, centroids);
				centroidFloats =
					PgturbohybridMultiVectorFloatCount(centroids->count,
													   centroids->dim);
				for (uint32 startFloat = 0; startFloat < centroidFloats;)
				{
					uint16		count =
						(uint16) Min((Size) maxCentroidCount,
									 centroidFloats - (Size) startFloat);
					Size		tupleSize =
						PgturbohybridGraphMultiVectorDocMapCentroidTupleSize(count);
					PgturbohybridGraphMultiVectorDocMapCentroidTuple tuple =
						palloc0(tupleSize);

					CHECK_FOR_INTERRUPTS();
					tuple->type =
						PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_CENTROID_TUPLE_TYPE;
					tuple->version = PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_VERSION;
					tuple->count = count;
					tuple->magic = PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_MAGIC;
					tuple->docId = docId;
					tuple->centroidCount = (uint16) centroids->count;
					tuple->flags = 0;
					tuple->startFloat = startFloat;
					tuple->residualMean =
						state->multivectorDocCentroidResiduals != NULL ?
						state->multivectorDocCentroidResiduals[docId] : 0.0f;
					memcpy(tuple->values, centroids->values + startFloat,
						   sizeof(float) * count);
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
					startFloat += count;
				}
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
		PgturbohybridGraphWriteMultiVectorQuantizedInvertedPostingTuples(state,
																		 &buf,
																		 &page,
																		 &start,
																		 pageCount,
																		 docMapBytes);
	}

	if (BufferIsValid(buf))
		PgturbohybridGraphFinishPage(buf);

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
	WaitForParallelWorkersToAttach(pcxt);
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
	if (state.multivectorBuild)
		workerRequest = 0;

	if (heap != NULL && workerRequest > 0 &&
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
				workerCount = Max(fitShared->nparticipants - 1, 0);
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

	state.buildFastEdges = PgturbohybridGraphUseFastBuildEdges(&state);
	PgturbohybridGraphDebugBuildPhaseStart(&state, "build_edges");
	INSTR_TIME_SET_CURRENT(phaseStart);
	edgeDistanceStart = state.buildDistanceCalls;
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
			state.buildEdgeSearchLayerUs += edgeShared->edgeSearchLayerUs;
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
			edgesUs = edgeWallUs;
			PgturbohybridNativeFinishParallelPhase(edgePcxt, InvalidSnapshot);
		}
		else
			PgturbohybridGraphBuildEdges(&state);
	}
	else
		PgturbohybridGraphBuildEdges(&state);
	state.buildEdgeDistanceCalls = state.buildDistanceCalls - edgeDistanceStart;
	if (!parallelEdgeBuildEnabled)
		edgesUs = PgturbohybridGraphElapsedUs(phaseStart);
	PgturbohybridGraphDebugBuildPhaseDone(&state, "build_edges", phaseStart);

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

	if (RelationNeedsWAL(index))
	{
		PgturbohybridGraphDebugBuildPhaseStart(&state, "wal_newpages");
		INSTR_TIME_SET_CURRENT(phaseStart);
		log_newpage_range(index, MAIN_FORKNUM, 0, RelationGetNumberOfBlocks(index), true);
		walUs = PgturbohybridGraphElapsedUs(phaseStart);
		PgturbohybridGraphDebugBuildPhaseDone(&state, "wal_newpages", phaseStart);
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
		buildStats.buildDistanceCacheHits = state.buildDistanceCacheHits;
		buildStats.buildDistanceCacheMisses = state.buildDistanceCacheMisses;
		buildStats.buildDistanceCacheStores = state.buildDistanceCacheStores;
		buildStats.buildDistanceCacheCollisions =
			state.buildDistanceCacheCollisions;
		buildStats.buildEdgeDistanceCalls = state.buildEdgeDistanceCalls;
		buildStats.buildEdgeSearchLayerUs = state.buildEdgeSearchLayerUs;
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
	PageHeader	pageHeader;
	PgturbohybridGraphMetaPage metap;
	Size		metaBytes;
	Size		metaStart;

	buf = ReadBuffer(index, PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	pageHeader = (PageHeader) page;
	metap = PgturbohybridGraphPageGetMeta(page);

	if (metap->magicNumber != PGTURBOHYBRID_GRAPH_MAGIC_NUMBER ||
		metap->storageKind != PGTURBOHYBRID_GRAPH_STORAGE_QUANT_GRAPH_NATIVE)
	{
		UnlockReleaseBuffer(buf);
		return false;
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
			segment->entryNodeId >= meta->tqNodeCount ||
			segment->entryNodeId < segment->startNodeId ||
			segment->entryNodeId >= endNodeId)
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
	PgturbohybridGraphFrontierItem sampled[PGTURBOHYBRID_GRAPH_ENTRY_SAMPLE_COUNT];
	uint32		sampledNodeIds[PGTURBOHYBRID_GRAPH_ENTRY_SAMPLE_COUNT];
	double		sampledDistances[PGTURBOHYBRID_GRAPH_ENTRY_SAMPLE_COUNT];
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

static int
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

static int
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
	uint8		querySketch[PGTURBOHYBRID_GRAPH_MAX_RESIDUAL_RERANK_BYTES];
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
				PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_BALANCED_HIGHDIM;
			so->graphHeapRescoreAutoEnabled = lowDim;
			return lowDim ? PGTURBOHYBRID_DENSE_HEAP_RESCORE_TOPK :
				PGTURBOHYBRID_DENSE_HEAP_RESCORE_OFF;
		case PGTURBOHYBRID_PROFILE_MATCHED_RECALL:
			so->graphHeapRescoreReason = lowDim ?
				PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_MATCHED_RECALL_LOWDIM :
				PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_MATCHED_RECALL_HIGHDIM;
			so->graphHeapRescoreAutoEnabled = lowDim;
			return lowDim ? PGTURBOHYBRID_DENSE_HEAP_RESCORE_TOPK :
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
				PGTURBOHYBRID_DENSE_HEAP_RESCORE_REASON_PROFILE_QUALITY_HIGHDIM;
			so->graphHeapRescoreAutoEnabled = lowDim;
			return lowDim ? PGTURBOHYBRID_DENSE_HEAP_RESCORE_TOPK :
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

static int
PgturbohybridMultiVectorDocCandidateLimit(int targetK)
{
	int64		docLimit;

	docLimit = targetK > 0 ? targetK : pgturbohybrid_multivector_doc_candidate_k;
	docLimit = Min(docLimit, (int64) pgturbohybrid_multivector_doc_candidate_k);
	return (int) Max(docLimit, (int64) 1);
}

typedef struct PgturbohybridMultiVectorExactRerankWorkStats
{
	uint32		candidates;
	uint64		tokensEvaluated;
	uint64		tokensSkipped;
	uint64		pairsSaved;
	bool		adaptiveTopKChangedVsFull;
}			PgturbohybridMultiVectorExactRerankWorkStats;

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
	TupleTableSlot *slot;
	TupleDesc	desc;
	AttrNumber	denseAttno;
	int			limit;
	int			rescored = 0;
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
		scan == NULL || scan->heapRelation == NULL ||
		scan->indexRelation == NULL || scan->indexRelation->rd_index == NULL)
		return 0;

	denseAttno =
		scan->indexRelation->rd_index->indkey.values[PGTURBOHYBRID_DENSE_KEY_INDEX];
	desc = RelationGetDescr(scan->heapRelation);
	if (denseAttno <= 0 || denseAttno > desc->natts)
		return 0;

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
	slot = table_slot_create(scan->heapRelation, NULL);
	for (int i = 0; i < limit; i++)
	{
		Datum		value;
		bool		isnull;
		bool		visible;
		char	   *valuePtr;
		PgturbohybridMultiVector *doc;
		double		exactMaxsim;
		uint64		docPairsEvaluated = 0;
		uint64		docTokensEvaluated = 0;
		uint64		docTokensSkipped = 0;
		uint64		docPairsSaved = 0;
		bool		adaptivePruned = false;
		bool		adaptiveScored = false;
		instr_time	fetchStart;

		CHECK_FOR_INTERRUPTS();
		if (candidates[i].exactScored)
			continue;

		INSTR_TIME_SET_CURRENT(fetchStart);
		visible = table_tuple_fetch_row_version(scan->heapRelation,
												&candidates[i].heaptid,
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
		doc = PgturbohybridDatumGetMultiVector(value);
		PgturbohybridCheckSameMultiVectorDims(query, doc);
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
		if ((char *) doc != valuePtr)
			pfree(doc);
		ExecClearTuple(slot);
	}
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
	if (rescored > 0)
	{
		so->graphHeapRescoreCount += rescored;
		so->graphExactRescoreSource = PGTURBOHYBRID_EXACT_RESCORE_SOURCE_HEAP;
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
		PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_PROXY_VECTOR &&
		pgturbohybrid_multivector_candidate_source !=
		PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_CENTROID_LITE &&
		pgturbohybrid_multivector_candidate_source !=
		PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_QUANTIZED_INVERTED_EXPERIMENTAL)
	{
		int64		docLimit;
		int64		rescoreLimit;
		int64		oversampling;
		int64		effectiveDocGraphCandidateK;

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
	heapScan = table_beginscan(scan->heapRelation, scan->xs_snapshot, 0, NULL);
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
	candidate->heaptid = entry->heaptid;
	candidate->distance = -entry->score;
	candidate->similarity = divisor > 0.0 ? entry->score / divisor : 0.0;
	candidate->rank = 0;
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

static const char *
PgturbohybridMultiVectorDocStorageKindName(int kind)
{
	switch (kind)
	{
		case PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F16:
			return "f16";
		case PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_SQ8:
			return "sq8";
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

static Size
PgturbohybridMultiVectorDocCompactArraySize(Size elemSize, Size count)
{
	if (elemSize != 0 && count > MaxAllocSize / elemSize)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("multivector document compact sidecar allocation is too large")));
	return elemSize * count;
}

static uint16
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

static PgturbohybridMultiVectorDocCompactStorage *
PgturbohybridMultiVectorBuildDocCompactStorage(PgturbohybridGraphMetaPageData *meta,
											   PgturbohybridGraphScanStorage *storage,
											   int kind)
{
	PgturbohybridMultiVectorDocCompactStorage *compact;
	Size		docPointerBytes;

	if (kind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32)
		return NULL;

	compact = palloc0(sizeof(PgturbohybridMultiVectorDocCompactStorage));
	compact->kind = kind;
	compact->docCount = meta->tqMultivectorDocCount;
	compact->dim = meta->dimensions;
	compact->bytes = sizeof(PgturbohybridMultiVectorDocCompactStorage);

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
	else
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("unsupported multivector document storage mode")));

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
			float		maxAbs = 0.0f;
			float		scale;

			valueBytes =
				PgturbohybridMultiVectorDocCompactArraySize(sizeof(int8),
															valueCount);
			compact->sq8Values[docId] = palloc(valueBytes);
			for (Size i = 0; i < valueCount; i++)
				maxAbs = Max(maxAbs, fabsf(doc->values[i]));
			scale = maxAbs > 0.0f ? maxAbs / 127.0f : 1.0f;
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
											  PgturbohybridGraphScanStorage *storage,
											  PgturbohybridMultiVectorDocCompactStorage *compact,
											  TqDocId docId,
											  int32 startToken,
											  int32 endToken,
											  const float4 *queryWeights,
											  const bool *queryMask)
{
	PgturbohybridMultiVector *doc;
	double		score = 0.0;

	Assert(compact != NULL);
	doc = storage->multivectorDocVectors[docId];
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
				uint16	   *dv = compact->f16Values[docId] + base;

				for (int32 d = 0; d < doc->dim; d++)
					dot += (double) qv[d] *
						(double) PgturbohybridMultiVectorHalfToFloat(dv[d]);
			}
			else
			{
				int8	   *dv = compact->sq8Values[docId] + base;
				float		scale = compact->sq8Scales[docId];

				for (int32 d = 0; d < doc->dim; d++)
					dot += (double) qv[d] * (double) dv[d] *
						(double) scale;
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
										 PgturbohybridGraphScanStorage *storage,
										 PgturbohybridMultiVectorDocCompactStorage *compact,
										 TqDocId docId,
										 bool contextLevel,
										 const float4 *queryWeights,
										 const bool *queryMask,
										 uint64 *pairsScored)
{
	PgturbohybridMultiVector *doc = storage->multivectorDocVectors[docId];

	if (doc == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INDEX_CORRUPTED),
				 errmsg("document-node multivector sidecar is missing a document vector"),
				 errhint("REINDEX the index to rebuild the document-node multivector sidecar.")));
	if (pairsScored != NULL)
		*pairsScored += (uint64) query->count * (uint64) doc->count;
	if (!contextLevel || !PgturbohybridMultiVectorHasContexts(doc))
		return PgturbohybridMultiVectorDocCompactMaxSimRange(query, storage,
															 compact, docId,
															 0, doc->count,
															 queryWeights,
															 queryMask);
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
				PgturbohybridMultiVectorDocCompactMaxSimRange(query, storage,
															  compact, docId,
															  start, end,
															  queryWeights,
															  queryMask);

			if (score > best)
				best = score;
		}
		return best;
	}
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
												  PgturbohybridMultiVectorDocSidecarAccessStats *sidecarStats)
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

	doc = PgturbohybridGraphLoadMultiVectorDocVector(index, meta, storage,
													 docId, storage->ctx,
													 sidecarStats);

	PgturbohybridCheckSameMultiVectorDims(query, doc);
	if (compact != NULL)
		maxsim = PgturbohybridMultiVectorDocCompactMaxSim(query, storage,
														  compact, docId,
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
																	 sidecarStats);
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
																		 sidecarStats);
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
											  PgturbohybridMultiVectorDocSidecarAccessStats *sidecarStats)
{
	PgturbohybridGraphFrontierItem entries[PGTURBOHYBRID_GRAPH_MAX_ENTRY_POINTS];
	PgturbohybridGraphFrontierItem sampled[PGTURBOHYBRID_GRAPH_ENTRY_SAMPLE_COUNT];
	int			entryCount = 0;
	int			sampledCount = 0;
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

	if (meta->tqNodeCount == 0 || resultTarget <= 0 || searchEf <= 0)
		return 0;

	frontierCapacity =
		PgturbohybridGraphInitialFrontierCapacity(meta->tqNodeCount,
												  searchEf,
												  PGTURBOHYBRID_GRAPH_MAX_ENTRY_POINTS,
												  maxNeighbors);
	visited = palloc0(sizeof(bool) * meta->tqNodeCount);
	frontier = palloc(sizeof(PgturbohybridGraphFrontierItem) * frontierCapacity);
	nearest = palloc(sizeof(PgturbohybridGraphFrontierItem) * searchEf);

	entry.nodeId = entryNodeId;
	entry.distance =
		PgturbohybridMultiVectorDocumentGraphNodeDistance(index, so, meta,
														  storage, compact,
														  query,
														  queryWeights,
														  queryMask,
														  entryNodeId,
														  pairsScored,
														  sidecarStats);
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
																	 sidecarStats);
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
																	 sidecarStats);
		PgturbohybridMultiVectorDocumentGraphAddEntry(entries, &entryCount,
													 PGTURBOHYBRID_GRAPH_MAX_ENTRY_POINTS,
													 nodeId, distance);
	}
	if (meta->tqNodeCount > 1)
	{
		int			sampleCount =
			Min((int) meta->tqNodeCount,
				Min(PGTURBOHYBRID_GRAPH_ENTRY_SAMPLE_COUNT,
					Max(searchEf, PGTURBOHYBRID_GRAPH_MAX_ENTRY_POINTS)));

		/*
		 * Document-node graphs need the same multi-entry robustness as the
		 * single-vector graph path.  A single global entry, plus a few segment
		 * entries, can start large ColBERT indexes in the wrong region and then
		 * spend a large MaxSim budget on irrelevant documents.  Score a bounded,
		 * deterministic spread of document nodes with the real document MaxSim
		 * distance and keep the best entry seeds before base-layer traversal.
		 */
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
																  sidecarStats);
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
	}

	if (entryCount == 0)
	{
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

			CHECK_FOR_INTERRUPTS();
			if (edgesVisited != NULL)
				(*edgesVisited)++;
			if (neighbor >= meta->tqNodeCount || visited[neighbor])
				continue;
			visited[neighbor] = true;
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
																		 sidecarStats);
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
		candidate.heaptid = docEntry->heapTid;
		candidate.distance = nearest[i].distance;
		candidate.similarity =
			queryWeightSum > 0.0 ? (-nearest[i].distance) / queryWeightSum : 0.0;
		candidate.rank = 0;
		candidate.exactScored = false;
		PgturbohybridMultiVectorCandidateHeapOffer(candidates, &docCount,
												   resultTarget, &candidate);
	}

	pfree(nearest);
	pfree(frontier);
	pfree(visited);
	return docCount;
}

static uint32
PgturbohybridMultiVectorQuantizedInvertedCodeword(const PgturbohybridMultiVector *mv,
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

static uint16
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
	int			proxyEncoder =
		PgturbohybridGraphGetMultiVectorProxyEncoderOption(scan->indexRelation);
	int			centroidMode =
		PgturbohybridGraphGetMultiVectorCentroidsOption(scan->indexRelation);
	bool		exhaustiveScan;
	bool		compactTraversal = false;
	bool		proxyVector =
		pgturbohybrid_multivector_candidate_source ==
		PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_PROXY_VECTOR;
	bool		qdrantLikeProxyVector =
		proxyVector &&
		pgturbohybrid_multivector_branch_plan ==
		PGTURBOHYBRID_BRANCH_PLAN_QDRANT_LIKE;
	bool		proxyDocumentCompactRescore = false;
	bool		centroidLite =
		pgturbohybrid_multivector_candidate_source ==
		PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_CENTROID_LITE;
	bool		quantizedInvertedExperimental =
		pgturbohybrid_multivector_candidate_source ==
		PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_QUANTIZED_INVERTED_EXPERIMENTAL;
	bool		documentNodesSource =
		pgturbohybrid_multivector_candidate_source ==
		PGTURBOHYBRID_MULTIVECTOR_CANDIDATE_SOURCE_DOCUMENT_NODES;
	uint64		docsScored = 0;
	uint64		edgesVisited = 0;
	uint64		maxsimPairs = 0;
	uint64		exactPairs = 0;
	PgturbohybridMultiVectorExactRerankWorkStats exactStats;
	uint64		quantizedScores = 0;
	uint64		compactBytes = 0;
	uint64		originalTokens = 0;
	uint64		pooledTokens = 0;
	uint64		centroidListsVisited = 0;
	uint64		centroidDocsTouched = 0;
	uint64		centroidPrunedDocs = 0;
	uint32		centroidCandidates = 0;
	uint64		quantizedInvertedListsVisited = 0;
	uint64		quantizedInvertedPostingsTouched = 0;
	uint64		quantizedInvertedDocsScored = 0;
	uint32		quantizedInvertedCandidates = 0;
	int			docStorageCacheMode;
	const char *docGraphWarning;
	const char *docStorageKindName;
	const char *docStorageCacheModeName;
	const char *docAccumulatorKind;
	instr_time	phaseStart;
	PgturbohybridMultiVectorDocCompactStorage *compact = NULL;
	PgturbohybridMultiVectorDocSidecarAccessStats sidecarStats;
	Vector	   *proxyQuery = NULL;
	ItemPointerData proxyTopTid;
	bool		proxyTopTidValid = false;
	bool		proxyTop1Admission = false;
	uint32		proxyDocumentRescoreDocs = 0;

	if (outCandidates == NULL)
		return 0;
	if (centroidLite &&
		centroidMode != PGTURBOHYBRID_MULTIVECTOR_CENTROIDS_KMEANS)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("centroid_lite multivector candidate source requires multivector_centroids = kmeans"),
				 errhint("REINDEX with multivector_graph = document_nodes, multivector_centroids = kmeans, or use turbohybrid.multivector_candidate_source = document_nodes.")));

	oldCtx = MemoryContextSwitchTo(resultCtx);
	*outCandidates = NULL;
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
	docStorageKind = pgturbohybrid_multivector_doc_storage;
	if (docStorageKind != PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32 &&
		docStorageKind != PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F16 &&
		docStorageKind != PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_SQ8)
		docStorageKind = PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32;
	docStorageCacheMode =
		PgturbohybridMultiVectorChooseDocStorageCacheMode(meta);
	if (docStorageCacheMode ==
		PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_CACHE_PAGED)
	{
		if (centroidLite || quantizedInvertedExperimental)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("paged document-node sidecar cache does not support this multivector candidate source"),
					 errhint("Use turbohybrid.multivector_candidate_source = document_nodes or proxy_vector, or set turbohybrid.multivector_doc_storage_cache = resident.")));
		docStorageKind = PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32;
	}
	compactTraversal =
		!exhaustiveScan &&
		!proxyVector &&
		!centroidLite &&
		!quantizedInvertedExperimental &&
		docStorageKind != PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32;
	docStorageKindName =
		PgturbohybridMultiVectorDocStorageKindName(docStorageKind);
	proxyDocumentCompactRescore =
		qdrantLikeProxyVector &&
		docStorageKind != PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F32;
	docStorageCacheModeName =
		PgturbohybridMultiVectorDocStorageCacheModeName(docStorageCacheMode);
	memset(&sidecarStats, 0, sizeof(sidecarStats));
	memset(&exactStats, 0, sizeof(exactStats));
	strlcpy(sidecarStats.cacheMode, docStorageCacheModeName,
			sizeof(sidecarStats.cacheMode));
	if (proxyVector)
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
	}
	(void) PgturbohybridGraphLoadMultiVectorDocMapWithStats(scan->indexRelation,
															meta,
															&storage,
															true,
															&sidecarStats);
	if (!storage.multivectorDocVectorsLoaded)
		ereport(ERROR,
				(errcode(ERRCODE_INDEX_CORRUPTED),
				 errmsg("document-node multivector sidecar is not loaded"),
				 errhint("REINDEX the index to rebuild the document-node multivector sidecar.")));
	if (centroidLite &&
		((meta->tqMultivectorDocMapFlags &
		  PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_CENTROIDS) == 0 ||
		 (meta->tqMultivectorDocMapFlags &
		  PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_CENTROID_POSTINGS) == 0 ||
		 !storage.multivectorDocCentroidsLoaded ||
		 !storage.multivectorCentroidPostingsLoaded))
		ereport(ERROR,
				(errcode(ERRCODE_INDEX_CORRUPTED),
				 errmsg("document-node centroid_lite sidecar is not loaded"),
				 errhint("REINDEX with multivector_graph = document_nodes and multivector_centroids = kmeans to rebuild persisted centroid sidecar and posting tuples.")));
	if (quantizedInvertedExperimental &&
		((meta->tqMultivectorDocMapFlags &
		  PGTURBOHYBRID_GRAPH_MULTIVECTOR_DOCMAP_FLAG_QUANTIZED_POSTINGS) == 0 ||
		 !storage.multivectorQuantizedInvertedPostingsLoaded))
		ereport(ERROR,
				(errcode(ERRCODE_INDEX_CORRUPTED),
				 errmsg("quantized_inverted_experimental posting sidecar is not loaded"),
				 errhint("REINDEX with multivector_graph = document_nodes to rebuild experimental quantized inverted posting tuples.")));
	PgturbohybridMultiVectorDocTokenTotals(&storage, &originalTokens,
										   &pooledTokens);
	if (proxyVector)
	{
		proxyQuery =
			PgturbohybridMultiVectorBuildProxyVector(query, proxyEncoder,
													 resultCtx);
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
		compact =
			PgturbohybridMultiVectorBuildDocCompactStorage(meta, &storage,
														   docStorageKind);
		compactBytes = compact != NULL ? compact->bytes : 0;
		PgturbohybridGraphAddElapsedUs(&so->graphPrepareUs, phaseStart);
	}

	if (exhaustiveScan && !centroidLite && !quantizedInvertedExperimental)
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
		candidate.heaptid = docEntry->heapTid;
		candidate.distance = -maxsim;
		candidate.similarity =
			queryWeightSum > 0.0 ? maxsim / queryWeightSum : 0.0;
			candidate.rank = 0;
			candidate.exactScored = false;
			PgturbohybridMultiVectorCandidateHeapOffer(candidates, &docCount,
													  candidateLimit,
													  &candidate);
			if (storage.multivectorDocVectorsPaged)
				pfree(doc);
			docsScored++;
		}
		PgturbohybridGraphAddElapsedUs(&so->graphTraverseUs, phaseStart);
		docGraphWarning = "document_node_f32_sidecar_exact_scan";
	}
	else
	{
		INSTR_TIME_SET_CURRENT(phaseStart);
		if (proxyVector)
		{
			PgturbohybridGraphResult *hits;
			int			hitCount;

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
				candidate.heaptid = docEntry->heapTid;
				candidate.distance = hits[i].distance;
				candidate.similarity = -hits[i].distance;
				candidate.rank = 0;
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
			bool	   *docMatched;
			uint32	   *matchedDocIds;
			uint32	   *touchedDocIds;
			uint32		codebookSize;
			uint32		matchedDocCount = 0;

			codebookSize = storage.multivectorQuantizedInvertedCodebookSize;
			postings = storage.multivectorQuantizedInvertedPostings;
			listOffsets = storage.multivectorQuantizedInvertedListOffsets;
			if (codebookSize != (uint32) query->dim * 2U ||
				postings == NULL ||
				listOffsets == NULL)
				ereport(ERROR,
						(errcode(ERRCODE_INDEX_CORRUPTED),
						 errmsg("quantized_inverted_experimental posting sidecar is invalid"),
						 errhint("REINDEX with multivector_graph = document_nodes to rebuild experimental quantized inverted posting tuples.")));

			for (uint32 docId = 0; docId < meta->tqMultivectorDocCount; docId++)
			{
				PgturbohybridMultiVector *doc;

				CHECK_FOR_INTERRUPTS();
				doc = storage.multivectorDocVectors[docId];
				if (doc == NULL)
					ereport(ERROR,
							(errcode(ERRCODE_INDEX_CORRUPTED),
							 errmsg("document-node multivector sidecar is missing a document vector"),
							 errhint("REINDEX the index to rebuild the document-node multivector sidecar.")));

				PgturbohybridCheckSameMultiVectorDims(query, doc);
			}

			docScores =
				palloc0(sizeof(double) *
						(Size) Max(meta->tqMultivectorDocCount, 1U));
			docBest =
				palloc0(sizeof(double) *
						(Size) Max(meta->tqMultivectorDocCount, 1U));
			docBestGeneration =
				palloc0(sizeof(uint32) *
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
			for (int32 qi = 0; qi < query->count; qi++)
			{
				uint32		queryCodeword;
				uint32		start;
				uint32		end;
				uint32		touchedCount = 0;
				uint32		generation = (uint32) qi + 1U;
				double		weight =
					queryWeights != NULL ? (double) queryWeights[qi] : 1.0;

				if (queryMask != NULL && queryMask[qi])
					continue;
				queryCodeword =
					PgturbohybridMultiVectorQuantizedInvertedCodeword(query,
																	  qi);
				start = listOffsets[queryCodeword];
				end = listOffsets[queryCodeword + 1];
				quantizedInvertedListsVisited++;
				for (uint32 postingOffset = start; postingOffset < end;
					 postingOffset++)
				{
					PgturbohybridGraphMultiVectorQuantizedPostingEntry *posting;
					PgturbohybridMultiVector *doc;
					uint32		docId;
					double		dot;

					CHECK_FOR_INTERRUPTS();
					posting = &postings[postingOffset];
					docId = posting->docId;
					doc = storage.multivectorDocVectors[docId];
					if (doc == NULL ||
						posting->tokenOrdinal >= (uint32) doc->count)
						ereport(ERROR,
								(errcode(ERRCODE_INDEX_CORRUPTED),
								 errmsg("quantized_inverted_experimental posting sidecar is invalid"),
								 errhint("REINDEX with multivector_graph = document_nodes to rebuild experimental quantized inverted posting tuples.")));
					if (docBestGeneration[docId] != generation)
					{
						docBestGeneration[docId] = generation;
						docBest[docId] = -DBL_MAX;
						touchedDocIds[touchedCount++] = docId;
					}
					dot = PgturbohybridMultiVectorTokenDot(query, qi,
														   doc,
														   posting->tokenOrdinal);
					if (dot > docBest[docId])
						docBest[docId] = dot;
					quantizedInvertedPostingsTouched++;
				}
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
				}
			}

			for (uint32 matchedIndex = 0; matchedIndex < matchedDocCount;
				 matchedIndex++)
			{
				uint32		docId = matchedDocIds[matchedIndex];
				TqMultiVectorDocMapEntry *docEntry;
				TqDenseCandidate candidate;

				CHECK_FOR_INTERRUPTS();
				docEntry = &storage.multivectorDocMap[docId];
				memset(&candidate, 0, sizeof(candidate));
				candidate.nodeId = docEntry->firstNodeId;
				candidate.heaptid = docEntry->heapTid;
				candidate.distance = -docScores[docId];
				candidate.similarity =
					queryWeightSum > 0.0 ? docScores[docId] / queryWeightSum : 0.0;
				candidate.rank = 0;
				candidate.exactScored = false;
				PgturbohybridMultiVectorCandidateHeapOffer(candidates,
														  &docCount,
														  candidateLimit,
														  &candidate);
			}
			quantizedInvertedDocsScored = matchedDocCount;
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
			uint32		codebookSize;
			uint32		matchedDocCount = 0;
			uint64		centroidPostingsTouched = 0;

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

			for (uint32 docId = 0; docId < meta->tqMultivectorDocCount; docId++)
			{
				PgturbohybridMultiVector *centroids;

				CHECK_FOR_INTERRUPTS();
				centroids = storage.multivectorDocCentroids[docId];
				if (storage.multivectorDocVectors[docId] == NULL)
					ereport(ERROR,
							(errcode(ERRCODE_INDEX_CORRUPTED),
							 errmsg("document-node multivector sidecar is missing a document vector"),
							 errhint("REINDEX the index to rebuild the document-node multivector sidecar.")));
				if (centroids == NULL)
					ereport(ERROR,
							(errcode(ERRCODE_INDEX_CORRUPTED),
							 errmsg("document-node centroid_lite sidecar is missing a document centroid"),
							 errhint("REINDEX with multivector_graph = document_nodes and multivector_centroids = kmeans to rebuild persisted centroid sidecar tuples.")));

				PgturbohybridCheckSameMultiVectorDims(query, centroids);
			}

			docScores =
				palloc0(sizeof(double) *
						(Size) Max(meta->tqMultivectorDocCount, 1U));
			docBest =
				palloc0(sizeof(double) *
						(Size) Max(meta->tqMultivectorDocCount, 1U));
			docBestGeneration =
				palloc0(sizeof(uint32) *
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
			for (int32 qi = 0; qi < query->count; qi++)
			{
				uint32		queryCodeword;
				uint32		start;
				uint32		end;
				uint32		touchedCount = 0;
				uint32		generation = (uint32) qi + 1U;
				double		weight =
					queryWeights != NULL ? (double) queryWeights[qi] : 1.0;

				if (queryMask != NULL && queryMask[qi])
					continue;
				queryCodeword =
					PgturbohybridMultiVectorQuantizedInvertedCodeword(query,
																	  qi);
				start = listOffsets[queryCodeword];
				end = listOffsets[queryCodeword + 1];
				centroidListsVisited++;
				for (uint32 postingOffset = start; postingOffset < end;
					 postingOffset++)
				{
					PgturbohybridGraphMultiVectorCentroidPostingEntry *posting;
					PgturbohybridMultiVector *centroids;
					uint32		docId;
					double		dot;

					CHECK_FOR_INTERRUPTS();
					posting = &postings[postingOffset];
					docId = posting->docId;
					centroids = storage.multivectorDocCentroids[docId];
					if (centroids == NULL ||
						posting->centroidOrdinal >= (uint32) centroids->count)
						ereport(ERROR,
								(errcode(ERRCODE_INDEX_CORRUPTED),
								 errmsg("document-node centroid_lite posting sidecar is invalid"),
								 errhint("REINDEX with multivector_graph = document_nodes and multivector_centroids = kmeans to rebuild persisted centroid posting tuples.")));
					if (docBestGeneration[docId] != generation)
					{
						docBestGeneration[docId] = generation;
						docBest[docId] = -DBL_MAX;
						touchedDocIds[touchedCount++] = docId;
					}
					dot = PgturbohybridMultiVectorTokenDot(query, qi,
														   centroids,
														   posting->centroidOrdinal);
					if (dot > docBest[docId])
						docBest[docId] = dot;
					centroidPostingsTouched++;
				}
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
				}
			}

			for (uint32 matchedIndex = 0; matchedIndex < matchedDocCount;
				 matchedIndex++)
			{
				uint32		docId = matchedDocIds[matchedIndex];
				TqMultiVectorDocMapEntry *docEntry;
				TqDenseCandidate candidate;

				CHECK_FOR_INTERRUPTS();
				docEntry = &storage.multivectorDocMap[docId];
				memset(&candidate, 0, sizeof(candidate));
				candidate.nodeId = docEntry->firstNodeId;
				candidate.heaptid = docEntry->heapTid;
				candidate.distance = -docScores[docId];
				candidate.similarity =
					queryWeightSum > 0.0 ? docScores[docId] / queryWeightSum : 0.0;
				candidate.rank = 0;
				candidate.exactScored = false;
				PgturbohybridMultiVectorCandidateHeapOffer(candidates,
														  &docCount,
														  candidateLimit,
														  &candidate);
			}
			docsScored = matchedDocCount;
			centroidDocsTouched = matchedDocCount;
			edgesVisited = centroidPostingsTouched;
			maxsimPairs = centroidPostingsTouched;
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
															  &sidecarStats);
		PgturbohybridGraphAddElapsedUs(&so->graphTraverseUs, phaseStart);
		quantizedScores = compactTraversal ? docsScored : 0;
		if (proxyVector)
			docGraphWarning = "document_node_proxy_vector_graph_traversal";
		else if (quantizedInvertedExperimental)
		{
			docsScored = quantizedInvertedDocsScored;
			edgesVisited = quantizedInvertedPostingsTouched;
			maxsimPairs = quantizedInvertedPostingsTouched;
			quantizedScores = quantizedInvertedPostingsTouched;
			docGraphWarning =
				"quantized_inverted_experimental_persisted_postings";
		}
		else if (centroidLite)
			docGraphWarning = "document_node_centroid_lite_prefilter";
		else if (compactTraversal &&
			docStorageKind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_F16)
			docGraphWarning = "document_node_f16_sidecar_graph_traversal";
		else if (compactTraversal &&
				 docStorageKind == PGTURBOHYBRID_MULTIVECTOR_DOC_STORAGE_SQ8)
			docGraphWarning = "document_node_sq8_sidecar_graph_traversal";
		else
			docGraphWarning = "document_node_f32_sidecar_graph_traversal";
	}

	PgturbohybridMultiVectorCandidateHeapSort(candidates, docCount);
	if (proxyVector && docCount > 0)
	{
		proxyTopTid = candidates[0].heaptid;
		proxyTopTidValid = true;
	}
	if (qdrantLikeProxyVector && docCount > 0)
	{
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
		proxyDocumentRescoreDocs = (uint32) docCount;
	}
	exactRerankLimitOverride = qdrantLikeProxyVector ?
		Min(docCount, rescoreLimit) : (proxyVector ? docCount : rescoreLimit);
	exactRerankCount =
		PgturbohybridMultiVectorExactHeapRerank(scan, so, query,
												queryWeights,
												queryMask,
												queryWeightSum,
												candidates,
												docCount,
												exactRerankLimitOverride,
												&exactPairs,
												&exactStats);
	if (exactRerankCount > 0 && docCount > 1)
		PgturbohybridMultiVectorCandidateHeapSort(candidates, docCount);
	if (proxyVector && proxyTopTidValid && docCount > 0)
		proxyTop1Admission =
			ItemPointerEquals(&proxyTopTid, &candidates[0].heaptid);
	for (int i = 0; i < docCount; i++)
		candidates[i].rank = i + 1;

	if (stats != NULL)
	{
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
				proxyVector ? "proxy_vector" :
				centroidLite ? "centroid_lite" :
				quantizedInvertedExperimental ? "quantized_inverted_experimental" :
				documentNodesSource ? "document_nodes" : "graph",
				sizeof(stats->multivectorCandidateSource));
		strlcpy(stats->multivectorProxyEncoderKind,
				proxyVector ?
				PgturbohybridMultiVectorProxyEncoderName(proxyEncoder) :
				"none",
				sizeof(stats->multivectorProxyEncoderKind));
		strlcpy(stats->multivectorGraphMode,
				PgturbohybridMultiVectorGraphModeName(meta->tqMultivectorGraphMode),
				sizeof(stats->multivectorGraphMode));
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
				(uint32) exactRerankLimitOverride;
			stats->multivectorDocGraphQuantizedScores =
				proxyDocumentCompactRescore ? proxyDocumentRescoreDocs :
				quantizedScores;
		strlcpy(stats->multivectorDocGraphStorageKind,
				docStorageKindName,
				sizeof(stats->multivectorDocGraphStorageKind));
		strlcpy(stats->multivectorDocGraphRescoreSource,
				exactRerankCount > 0 ? "heap" : "none",
				sizeof(stats->multivectorDocGraphRescoreSource));
		stats->multivectorDocGraphExactRerankDocs =
			(uint32) exactRerankCount;
		stats->multivectorDocGraphHeapFetches = so->graphHeapRescoreCount;
		strlcpy(stats->multivectorDocGraphWarning, docGraphWarning,
				sizeof(stats->multivectorDocGraphWarning));
		stats->proxyCandidates = proxyVector ? (uint32) docCount : 0;
		stats->proxyTop1Admission = proxyVector && proxyTop1Admission;
		stats->proxyExactRerankDocs =
			proxyVector ? (uint32) exactRerankCount : 0;
		centroidCandidates = centroidLite ? (uint32) docCount : 0;
		centroidPrunedDocs =
			centroidDocsTouched > (uint64) docCount ?
			centroidDocsTouched - (uint64) docCount : 0;
		stats->centroidListsVisited = centroidListsVisited;
		stats->centroidDocsTouched = centroidDocsTouched;
		stats->centroidPrunedDocs = centroidPrunedDocs;
		stats->centroidCandidates = centroidCandidates;
		quantizedInvertedCandidates =
			quantizedInvertedExperimental ? (uint32) docCount : 0;
		stats->quantizedInvertedListsVisited =
			quantizedInvertedListsVisited;
		stats->quantizedInvertedPostingsTouched =
			quantizedInvertedPostingsTouched;
		stats->quantizedInvertedDocsScored =
			quantizedInvertedDocsScored;
		stats->quantizedInvertedCandidates =
			quantizedInvertedCandidates;
		stats->quantizedInvertedExactRerankDocs =
			quantizedInvertedExperimental ? (uint32) exactRerankCount : 0;
		stats->quantizedInvertedCodebookSize =
			quantizedInvertedExperimental ?
			storage.multivectorQuantizedInvertedCodebookSize : 0;
			strlcpy(stats->multivectorDocSidecarCacheMode,
					sidecarStats.cacheMode,
					sizeof(stats->multivectorDocSidecarCacheMode));
			stats->multivectorDocSidecarPagesRead = sidecarStats.pagesRead;
			stats->multivectorDocSidecarCacheHits = sidecarStats.cacheHits;
			stats->multivectorDocSidecarCacheMisses = sidecarStats.cacheMisses;
			stats->multivectorDocSidecarBytesTouched = sidecarStats.bytesTouched;
			stats->multivectorDocSidecarVectorsLoaded =
				sidecarStats.vectorsLoaded;
			stats->multivectorTokensOriginal = originalTokens;
			stats->multivectorTokensPooled = pooledTokens;
		stats->multivectorReservoirsEnabled = false;
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
	HTAB	   *docHash;
	HTAB	   *docIdHash = NULL;
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
	int			rawTarget;
	int			searchEf;
	int			initialRawTarget;
	int			maxRawTarget;
	int			maxSearchEf = 0;
	int			docLimit;
	int			docCount = 0;
	int			exactRerankCount = 0;
	uint64		multivectorDocCapacity = 0;
	Size		multivectorMemoryEstimate = 0;
	uint64		multivectorRawSubvectorHits = 0;
	uint64		multivectorUniqueDocs = 0;
	uint64		multivectorDuplicateDocHits = 0;
	uint64		multivectorMaxsimUpdates = 0;
	uint64		multivectorExactPairs = 0;
	PgturbohybridMultiVectorExactRerankWorkStats exactStats;
	uint64		multivectorExactTokenScanNodesScored = 0;
	uint32		multivectorReservoirScoreDocs = 0;
	uint32		multivectorReservoirCoverageDocs = 0;
	uint32		multivectorReservoirMeanDocs = 0;
	uint32		multivectorReservoirPerTokenDocs = 0;
	uint32		multivectorReservoirBm25Docs = 0;
	uint32		multivectorReservoirUnionDocs = 0;
	uint32		multivectorReservoirDuplicates = 0;
	uint64		nextDocOrdinal = 0;
	uint32		admissionCandidatesBeforeRerank = 0;
	uint32		admissionTraceCount = 0;
	uint32		skippedQueryTokens = 0;
	uint32		tokenStatsCount = 0;
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
	bool		exactTokenScan =
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
	int			centroidMode =
		PgturbohybridGraphGetMultiVectorCentroidsOption(scan->indexRelation);
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
	PgturbohybridMultiVectorParseSkipQueryTokens(
		pgturbohybrid_multivector_debug_skip_query_tokens,
		mv->count,
		skipQueryToken,
		&skippedQueryTokens);
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
														   emptyIndex ? NULL : &meta,
														   targetK,
														   plainFallbackReason,
														   sizeof(plainFallbackReason));
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
		if (proxyVector && !emptyIndex && !plainFallback &&
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
														  &multivectorMaxsimUpdates);
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
		PgturbohybridMultiVectorBuildReservoirCandidates(docHash, mv->count,
														 queryWeightSum,
														 docLimit,
														 candidates,
														 &docCount,
														 resultCtx,
														 &multivectorReservoirScoreDocs,
														 &multivectorReservoirCoverageDocs,
														 &multivectorReservoirMeanDocs,
														 &multivectorReservoirPerTokenDocs,
														 &multivectorReservoirBm25Docs,
														 &multivectorReservoirUnionDocs,
														 &multivectorReservoirDuplicates);
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
		PgturbohybridMultiVectorExactHeapRerank(scan, so, mv,
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
			stats->centroidListsVisited =
				(uint64) ((uint32) mv->count - skippedQueryTokens);
			stats->centroidDocsTouched = admissionCandidatesBeforeRerank;
			stats->centroidCandidates = (uint32) docCount;
			stats->centroidPrunedDocs =
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
												  (meta->tqFlags & PGTURBOHYBRID_GRAPH_TQ_WEIGHTED) != 0,
												  meta->tqResidualRerankBytes));
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
														  (meta.tqFlags & PGTURBOHYBRID_GRAPH_TQ_WEIGHTED) != 0,
														  meta.tqResidualRerankBytes));
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
