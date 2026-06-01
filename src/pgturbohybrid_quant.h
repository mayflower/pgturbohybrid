#ifndef PGTURBOHYBRID_QUANT_H
#define PGTURBOHYBRID_QUANT_H

#include "postgres.h"

#include "pgturbohybrid.h"
#include "pgturbohybrid_quant_psquare.h"

#define PGTURBOHYBRID_GRAPH_CODE_TUPLE_TYPE		0x51
#define PGTURBOHYBRID_GRAPH_ADJ_TUPLE_TYPE			0x52
#define PGTURBOHYBRID_GRAPH_EXACT_TUPLE_TYPE		0x53
#define PGTURBOHYBRID_GRAPH_CORRECTION_TUPLE_TYPE	0x54
#define PGTURBOHYBRID_GRAPH_EXACT_SLAB_MAGIC		0x54514553U
#define PGTURBOHYBRID_GRAPH_NODE_DEAD				0x0001
#define PGTURBOHYBRID_GRAPH_TQ_PLUS				0x0001	/* metapage: ecShift/ecScale correction tuples present */
#define PGTURBOHYBRID_GRAPH_TQ_WEIGHTED			0x0002	/* metapage: code tuples carry per-vector ec_correction */
#define PGTURBOHYBRID_GRAPH_TQ_RENORM				0x0004	/* metapage: per-vector scale field stores renormalized l2 / centroid_norm instead of plain l2 */
#define PGTURBOHYBRID_GRAPH_EXACT_FREE				0x0008	/* metapage: exact vector slabs are omitted */
#define PGTURBOHYBRID_GRAPH_TQ_RESIDUAL_RERANK	0x0010	/* metapage: code tuples carry residual rerank sketch bytes */
#define PGTURBOHYBRID_GRAPH_TQ_EXACT_BUILD		0x0020	/* metapage: graph edges were built using exact distances */
#define PGTURBOHYBRID_GRAPH_TQ_BACKBONE			0x0040	/* metapage: level-0 adjacent backbone edges were forced at build */
#define PGTURBOHYBRID_GRAPH_TQ_FAST_BUILD_EDGES	0x0080	/* metapage: build used simple nearest-neighbor edge selection */
#define PGTURBOHYBRID_GRAPH_TQ_BUILD_NEIGHBOR_REASON_SHIFT 8
#define PGTURBOHYBRID_GRAPH_TQ_BUILD_NEIGHBOR_REASON_MASK 0x0700
#define PGTURBOHYBRID_GRAPH_TQ_BUILD_NEIGHBOR_REASON(flags) (((flags) & PGTURBOHYBRID_GRAPH_TQ_BUILD_NEIGHBOR_REASON_MASK) >> PGTURBOHYBRID_GRAPH_TQ_BUILD_NEIGHBOR_REASON_SHIFT)
#define PGTURBOHYBRID_GRAPH_TQ_BUILD_NEIGHBOR_REASON_BITS(reason) ((((uint16) (reason)) << PGTURBOHYBRID_GRAPH_TQ_BUILD_NEIGHBOR_REASON_SHIFT) & PGTURBOHYBRID_GRAPH_TQ_BUILD_NEIGHBOR_REASON_MASK)
#define PGTURBOHYBRID_GRAPH_MAX_ENTRY_POINTS		16
#define PGTURBOHYBRID_GRAPH_ENTRY_SAMPLE_COUNT		608
#define PGTURBOHYBRID_GRAPH_MAX_NEIGHBORS			(PGTURBOHYBRID_GRAPH_MAX_M * 2)
#define PGTURBOHYBRID_GRAPH_MAX_PAYLOADS			16
#define PGTURBOHYBRID_GRAPH_MAX_STORED_LEVEL		8
#define PGTURBOHYBRID_GRAPH_LOWBIT_L2_RESCORE_EF_MULT 6
#define PGTURBOHYBRID_GRAPH_LOWBIT_HIGHDIM_L2_TARGET_MULT 2
#define PGTURBOHYBRID_GRAPH_HIGHDIM_L2_RESCORE_EF_MULT 4
#define PGTURBOHYBRID_GRAPH_HIGHDIM_L2_TARGET_EF_MULT 8
#define PGTURBOHYBRID_GRAPH_HIGHDIM_ENTRY_SAMPLE_DIVISOR 4
#define PGTURBOHYBRID_GRAPH_TIGHT_L2_FILL_MULT		8
#define PGTURBOHYBRID_GRAPH_PAYLOAD_EXACT_MAX		1024
#define PGTURBOHYBRID_GRAPH_CODE_CODE_PRUNE_CHUNKS	32
#define PGTURBOHYBRID_GRAPH_STACK_FRONTIER_CAPACITY	256
#define PGTURBOHYBRID_GRAPH_STACK_RESCORE_CAPACITY	256

#if defined(__GNUC__) || defined(__clang__)
#define PGTURBOHYBRID_GRAPH_PREFETCH_READ(addr) __builtin_prefetch((addr), 0, 1)
#else
#define PGTURBOHYBRID_GRAPH_PREFETCH_READ(addr) ((void) 0)
#endif


/*
 * On-disk code tuple.
 *
 * The "data" flexible array holds the per-vector payloads followed by the
 * encoded code bytes.  When weighted quantized scoring is enabled and
 * the metapage carries PGTURBOHYBRID_GRAPH_TQ_WEIGHTED, a single additional float
 * (ec_correction = ⟨X+, M⟩, the per-vector renormalization scalar that
 * qdrant's TQ+ formula needs) is laid down BEFORE data[] — so data[]
 * starts 4 bytes later than in the legacy format.  Existing indexes without
 * the internal weighted flag are byte-for-byte unchanged.
 *
 * Use PgturbohybridGraphTupleEcCorrection / PgturbohybridGraphTupleSetEcCorrection /
 * PgturbohybridGraphTuplePayloads / PgturbohybridGraphTupleCode helpers — never index data[]
 * directly, since the offset depends on the internal weighted flag.
 */
typedef struct PgturbohybridGraphCodeTupleData
{
	uint8		type;
	uint8		level;
	uint16		flags;
	uint32		nodeId;
	ItemPointerData heaptid;
	BlockNumber exactBlkno;
	OffsetNumber exactOffno;
	uint16		payloadMask;
	float		scale;
	float		norm;
	float		correction;	/* ||X+||² (cached at build/insert time) */
	uint8		data[FLEXIBLE_ARRAY_MEMBER];
} PgturbohybridGraphCodeTupleData;

typedef PgturbohybridGraphCodeTupleData *PgturbohybridGraphCodeTuple;

typedef struct PgturbohybridGraphAdjTupleData
{
	uint8		type;
	uint8		level;
	uint16		count;
	uint32		nodeId;
	uint32		neighbors[FLEXIBLE_ARRAY_MEMBER];
} PgturbohybridGraphAdjTupleData;

typedef PgturbohybridGraphAdjTupleData *PgturbohybridGraphAdjTuple;

typedef struct PgturbohybridGraphExactTupleData
{
	uint8		type;
	uint8		flags;
	uint16		unused;
	uint32		nodeId;
	char		vector[FLEXIBLE_ARRAY_MEMBER];
} PgturbohybridGraphExactTupleData;

typedef PgturbohybridGraphExactTupleData *PgturbohybridGraphExactTuple;

typedef struct PgturbohybridGraphExactSlabPageHeaderData
{
	uint32		magic;
	uint16		used;
	uint16		unused;
	uint32		capacity;
	char		data[FLEXIBLE_ARRAY_MEMBER];
} PgturbohybridGraphExactSlabPageHeaderData;

typedef PgturbohybridGraphExactSlabPageHeaderData *PgturbohybridGraphExactSlabPageHeader;

typedef struct PgturbohybridGraphCorrectionTupleData
{
	uint8		type;
	uint8		field;
	uint16		count;
	uint32		startDim;
	float		values[FLEXIBLE_ARRAY_MEMBER];
} PgturbohybridGraphCorrectionTupleData;

typedef PgturbohybridGraphCorrectionTupleData *PgturbohybridGraphCorrectionTuple;

typedef struct PgturbohybridGraphBuildNode
{
	ItemPointerData heaptid;
	Vector	   *vector;
	uint8	   *code;
	uint8	   *residualSketch;
	int32	   *payloads;
	uint16		payloadMask;
	int			level;
	float		scale;
	float		norm;
	uint64		vectorHash;		/* build-time duplicate-run fingerprint */
	float		correction;		/* ||X+||² */
	float		ecCorrection;	/* ⟨X+, M⟩ — only meaningful when state->tqWeighted */
	BlockNumber exactBlkno;
	OffsetNumber exactOffno;
	uint16		flags;
	uint32	  **neighbors;
	double	  **neighborDistances;
	int		   *neighborCounts;
} PgturbohybridGraphBuildNode;

typedef struct PgturbohybridQuantBuildState
{
	Relation	heap;
	Relation	index;
	IndexInfo  *indexInfo;
	ForkNumber	forkNum;
	const PgturbohybridGraphTypeInfo *typeInfo;
	PgturbohybridGraphSupport support;
	MemoryContext ctx;
	MemoryContext buildTupleCtx;
	PgturbohybridGraphBuildNode *nodes;
	uint32		nodeCount;
	uint32		nodeCapacity;
	int			dimensions;
	int			m;
	int			efConstruction;
	int			tqBits;
	bool		tqWeighted;
	bool		tqQuantileFit;
	bool		tqRenorm;
	bool		tqExactStorage;
	bool		entrySidecar;
	int			entrySidecarRepresentatives;
	bool		graphBackbone;
	bool		residualRerank;
	int			residualRerankBytes;
	bool		buildExactDistances;	/* short-circuit quantized fast paths during build */
	int			buildDistanceMode; /* effective code/exact build distance mode */
	bool		buildCodeOnly;	/* avoid retaining raw vectors during exact-free builds */
	bool		buildFitPass;	/* table scan is only collecting correction statistics */
	bool		buildEncodeOnAppend;	/* encode node immediately during collection scan */
	bool		buildFastEdges; /* use bounded simple edge selection for code-only builds */
	int			buildNeighborSelectReason; /* why the final build edge selector was chosen */
	int			scoreMode;
	int			maxLevel;
	uint32		entryNodeId;
	uint16		segmentCount;
	PgturbohybridGraphSegmentMetaData segments[PGTURBOHYBRID_GRAPH_MAX_NATIVE_SEGMENTS];
	uint32		routingEntryCount;
	uint16		routingEntryBytes;
	uint32		routingEntryNodeIds[PGTURBOHYBRID_GRAPH_MAX_ROUTING_ENTRIES];
	uint32		entrySidecarCount;
	uint16		entrySidecarBytes;
	uint32		entrySidecarNodeIds[PGTURBOHYBRID_GRAPH_MAX_ENTRY_SIDECAR_REPRESENTATIVES];
	int			payloadCount;
	Size		payloadBytes;
	double		reltuples;
	bool		building;
	float	   *ecShift;
	float	   *ecScale;
	int16	   *dPrimeSqI16;	/* TQ+ per-coord weights, quantized to i16 (SIMD path) */
	float		weightScale;	/* TQ+ quantization scale: D'² / max(D'²) · (INT16_MAX-1) */
	double		mmConst;		/* TQ+ Σ ecShift[d]², cached at fit time */
	TqPSquareState *fitQLo;
	TqPSquareState *fitQHi;
	double	   *fitMean;
	double	   *fitM2;
	double	   *fitBuffer;
	uint64		fitCount;
	double		fitCOuter;
	double		fitMinQuantileWidth;
	uint64		buildDistanceWeighted;
	uint32	   *buildVisitedGeneration;
	uint32		buildVisitGeneration;
	MemoryContext buildQueryCtx;
	PgturbohybridGraphTqQuery buildTq;
	uint32		buildQueryNodeId;
	bool		buildTqValid;
	uint64		buildDistanceCalls;
	uint64		buildDistanceQuerySplit;
	uint64		buildDistancePacked;
	uint64		buildDistanceCodeCode;
	uint64		buildDistanceExact;
	uint64		buildDistanceFallback;
	uint64		buildEdgeDistanceCalls;
	uint64		buildEdgeSearchLayerUs;
	uint64		buildEdgeSelectNeighborUs;
	uint64		buildEdgeAddNeighborUs;
	uint64		buildEdgePruneNeighborUs;
	uint64		buildEdgeEntryUpdateUs;
	uint64		buildEdgeNearestTotal;
	uint64		buildEdgeNearestSamples;
	uint32		buildEdgeMaxFrontierSize;
	uint64		parallelEncodeUs;
	void	   *parallelShared;
	uint64		buildScanUs;
	uint64		buildCorrectionUs;
	uint64		buildEncodeUs;
	uint64		buildEdgeUs;
	uint64		buildWriteUs;
	uint32		buildWorkerCount;
} PgturbohybridQuantBuildState;

typedef struct PgturbohybridGraphResult
{
	ItemPointerData heaptid;
	uint32		nodeId;
	double		distance;
	bool		exactScored;
} PgturbohybridGraphResult;

typedef struct TqDenseCandidate
{
	uint32		nodeId;
	ItemPointerData heaptid;
	double		distance;
	double		similarity;
	int32		rank;
	bool		exactScored;
} TqDenseCandidate;

typedef struct TqDenseCandidateStats
{
	uint32		denseCandidatesRequested;
	uint32		effectiveResultTarget;
	uint32		effectiveSearchEf;
	uint32		effectiveRescoreBand;
	double		highdimWideningMultiplier;
	int			wideningReason;
	int			denseBudgetPolicy;
	int			rescoreBandPolicy;
	uint64		visitedGraphNodes;
	uint64		scoredCodes;
	uint32		denseCandidatesReturned;
	uint64		exactRescoreCount;
	uint64		heapRescoreCount;
	bool		heapRescoreAutoEnabled;
	int			heapRescoreReason;
	uint64		codePagesRead;
	uint64		adjPagesRead;
	uint64		prepareUs;
	uint64		traverseUs;
	uint64		entryUs;
	uint64		baseUs;
	uint64		batchUs;
	uint64		heapUs;
	uint64		heapFetchUs;
	uint64		heapRescoreUs;
	uint64		fillUs;
	uint64		rescoreUs;
	uint64		sortUs;
	int			exactRescoreSource;
} TqDenseCandidateStats;

typedef struct PgturbohybridGraphScanNode
{
	ItemPointerData heaptid;
	uint8	   *code;
	uint8	   *residualSketch;
	int32	   *payloads;
	uint16		payloadMask;
	char	   *exactVector;
	int			level;
	BlockNumber exactBlkno;
	OffsetNumber exactOffno;
	float		scale;
	float		norm;
	float		codeNorm;
	float		ecCorrection;	/* ⟨X+, M⟩, only meaningful when meta->tqFlags & PGTURBOHYBRID_GRAPH_TQ_WEIGHTED */
	uint16		flags;
	bool		loaded;
} PgturbohybridGraphScanNode;

typedef struct PgturbohybridGraphPayloadRef
{
	int16		payloadSlot;
	int32		payloadValue;
	uint32		nodeId;
} PgturbohybridGraphPayloadRef;

typedef struct PgturbohybridGraphScanStorage
{
	PgturbohybridGraphScanNode *nodes;
	uint8	   *codeArena;
	uint8	   *residualArena;
	uint8	   *payloadArena;
	char	   *exactArena;
	Size		exactBytes;
	uint32	  **neighbors;
	uint16	   *neighborCounts;
	uint32	   *visitedGeneration;
	uint32	   *visitGeneration;
	bool	   *codePagesLoaded;
	bool	   *adjPagesLoaded;
	BlockNumber *codeBlknos;
	BlockNumber *adjBlknos;
	OffsetNumber *adjOffnos;
	PgturbohybridGraphPayloadRef *payloadRefs;
	uint32		payloadRefCount;
	MemoryContext ctx;
	int			codeTuplesPerPage;
	int			codePageCount;
	int			adjPageCount;
	int			levelCount;
	bool		cached;
} PgturbohybridGraphScanStorage;

typedef struct PgturbohybridGraphNativeCache
{
	Oid			relid;
	Oid			relfilenumber;
	uint32		dimensions;
	uint16		m;
	uint16		graphMaxLevel;
	uint16		graphFlags;
	uint32		tqNodeCount;
	uint32		tqEntryNodeId;
	uint16		tqSegmentCount;
	uint16		tqCodeBytes;
	uint16		tqBits;
	uint16		tqPayloadCount;
	uint16		tqPayloadBytes;
	uint16		tqResidualRerankBytes;
	BlockNumber tqCodeStartBlkno;
	BlockNumber tqAdjStartBlkno;
	BlockNumber tqExactStartBlkno;
	BlockNumber tqCorrectionStartBlkno;
	PgturbohybridGraphSegmentMetaData tqSegments[PGTURBOHYBRID_GRAPH_MAX_NATIVE_SEGMENTS];
	PgturbohybridGraphScanStorage storage;
	MemoryContext ctx;
	/*
	 * Diagnostics captured when this per-backend cache was built: how long the
	 * one-time build took, and the resident footprint (code / adjacency / exact
	 * arenas plus node + metadata) that this backend now holds.  Surfaced via
	 * turbohybrid_last_scan_stats() so concurrent-client scaling can be reasoned
	 * about (each backend duplicates residentTotalBytes).
	 */
	int64		buildUs;
	Size		residentCodeBytes;
	Size		residentAdjBytes;
	Size		residentExactBytes;
	Size		residentTotalBytes;
	int64		buildCodeBufferLockWaitUs;
	int64		buildAdjBufferLockWaitUs;
	struct PgturbohybridGraphNativeCache *next;
} PgturbohybridGraphNativeCache;

/*
 * Out-param for PgturbohybridGraphInitScanStorage: how the scan's storage was
 * satisfied this scan (per-backend cache vs uncached per-scan loading), whether
 * the per-backend cache had to be built during this very scan (the cold-build
 * cost), how long that build took, and the resident byte breakdown.  Pass NULL
 * to ignore (e.g. the insert path).
 */
typedef struct PgturbohybridGraphCacheInitInfo
{
	PgturbohybridGraphNativeCacheMode mode;
	int			policy;
	int			reason;
	bool		used;
	bool		reused;
	bool		builtThisScan;
	int64		attachUs;
	int64		buildUs;
	int64		waitUs;
	int64		refcount;
	int64		totalBytes;
	int64		codeBytes;
	int64		adjBytes;
	int64		exactBytes;
	int64		codeBufferLockWaitUs;
	int64		adjBufferLockWaitUs;
} PgturbohybridGraphCacheInitInfo;

typedef struct PgturbohybridGraphCorrectionCache
{
	Oid			relid;
	Oid			relfilenumber;
	uint32		dimensions;
	uint16		tqFlags;
	BlockNumber tqCorrectionStartBlkno;
	float	   *ecShift;
	float	   *ecScale;
	MemoryContext ctx;
	struct PgturbohybridGraphCorrectionCache *next;
} PgturbohybridGraphCorrectionCache;

typedef struct PgturbohybridGraphFrontierItem
{
	uint32		nodeId;
	double		distance;
} PgturbohybridGraphFrontierItem;

typedef struct PgturbohybridGraphBuildOrderItem
{
	uint32		nodeId;
	uint64		key;
} PgturbohybridGraphBuildOrderItem;

typedef struct PgturbohybridGraphRescoreRef
{
	int			resultIndex;
	uint32		nodeId;
	BlockNumber blkno;
	OffsetNumber offno;
} PgturbohybridGraphRescoreRef;

static inline Size
PgturbohybridGraphCodeBytesForBits(int dimensions, int bits)
{
	return MAXALIGN(TqCodeSizeForBits(dimensions, bits));
}

static inline Size
PgturbohybridGraphPayloadBytes(int payloadCount)
{
	return MAXALIGN(sizeof(int32) * payloadCount);
}

static inline Size
PgturbohybridGraphTupleExtraHeaderBytes(bool tqWeighted)
{
	return tqWeighted ? sizeof(float) : 0;
}

static inline float
PgturbohybridGraphTupleEcCorrection(PgturbohybridGraphCodeTuple tuple, bool tqWeighted)
{
	float		value;

	if (!tqWeighted)
		return 0.0f;

	memcpy(&value, tuple->data, sizeof(float));
	return value;
}

static inline void
PgturbohybridGraphTupleSetEcCorrection(PgturbohybridGraphCodeTuple tuple, bool tqWeighted, float value)
{
	if (!tqWeighted)
		return;

	memcpy(tuple->data, &value, sizeof(float));
}

static inline int32 *
PgturbohybridGraphTuplePayloads(PgturbohybridGraphCodeTuple tuple, bool tqWeighted)
{
	return (int32 *) (tuple->data + PgturbohybridGraphTupleExtraHeaderBytes(tqWeighted));
}

static inline uint8 *
PgturbohybridGraphTupleResidual(PgturbohybridGraphCodeTuple tuple, Size payloadBytes,
								Size residualBytes, bool tqWeighted)
{
	return tuple->data + PgturbohybridGraphTupleExtraHeaderBytes(tqWeighted) + payloadBytes;
}

static inline Size
PgturbohybridGraphTupleResidualBytes(Size residualBytes)
{
	return MAXALIGN(residualBytes);
}

static inline uint8 *
PgturbohybridGraphTupleCode(PgturbohybridGraphCodeTuple tuple, Size payloadBytes,
							Size residualBytes, bool tqWeighted)
{
	return PgturbohybridGraphTupleResidual(tuple, payloadBytes, residualBytes,
										   tqWeighted) +
		PgturbohybridGraphTupleResidualBytes(residualBytes);
}

static inline Size
PgturbohybridGraphCodeTupleSize(int dimensions, int payloadCount, int bits,
								bool tqWeighted, Size residualBytes)
{
	return MAXALIGN(offsetof(PgturbohybridGraphCodeTupleData, data) +
					PgturbohybridGraphTupleExtraHeaderBytes(tqWeighted) +
					PgturbohybridGraphPayloadBytes(payloadCount) +
					PgturbohybridGraphTupleResidualBytes(residualBytes) +
					PgturbohybridGraphCodeBytesForBits(dimensions, bits));
}

static inline uint32
PgturbohybridGraphResidualMix32(uint32 x)
{
	x ^= x >> 16;
	x *= UINT32_C(0x7feb352d);
	x ^= x >> 15;
	x *= UINT32_C(0x846ca68b);
	x ^= x >> 16;
	return x;
}

static inline void
PgturbohybridGraphBuildResidualSketch(const float *values, int dimensions,
									  uint8 *out, int bytes)
{
	if (values == NULL || out == NULL || dimensions <= 0 || bytes <= 0)
		return;

	for (int byte = 0; byte < bytes; byte++)
	{
		double		acc = 0.0;

		for (int lane = 0; lane < 16; lane++)
		{
			uint32		hash = PgturbohybridGraphResidualMix32((uint32) byte * 0x9e3779b1U ^
															  (uint32) lane * 0x85ebca6bU ^
															  0x51ed270bU);
			int			dim = (int) (hash % (uint32) dimensions);
			double		sign = (hash & 0x80000000U) ? 1.0 : -1.0;

			acc += sign * (double) values[dim];
		}

		acc *= 24.0;
		if (acc > 127.0)
			acc = 127.0;
		else if (acc < -127.0)
			acc = -127.0;
		out[byte] = (uint8) ((int) lrint(acc) + 128);
	}
}

static inline Size
PgturbohybridGraphAdjTupleSize(int count)
{
	return MAXALIGN(offsetof(PgturbohybridGraphAdjTupleData, neighbors) + (sizeof(uint32) * count));
}

static inline int
PgturbohybridGraphLevelM(int m, int level)
{
	return PgturbohybridGraphGetLayerM(m, level);
}

static inline int
PgturbohybridGraphLevelCapacity(int m)
{
	return Min(PgturbohybridGraphGetMaxLevel(m), PGTURBOHYBRID_GRAPH_MAX_STORED_LEVEL) + 1;
}

static inline int
PgturbohybridGraphAdjRecordCount(PgturbohybridGraphMetaPageData *meta)
{
	int			levelCapacity = PgturbohybridGraphLevelCapacity(meta->m);

	if (levelCapacity > 0 && meta->tqNodeCount > (uint32) (PG_INT32_MAX / levelCapacity))
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pgturbohybrid graph adjacency metadata is too large")));
	return meta->tqNodeCount * levelCapacity;
}

static inline int
PgturbohybridGraphAdjSlot(PgturbohybridGraphMetaPageData *meta, uint32 nodeId, int level)
{
	return nodeId * PgturbohybridGraphLevelCapacity(meta->m) + level;
}

static inline int
PgturbohybridGraphTuplesPerPage(Size tupleSize)
{
	Size		usable = BLCKSZ - MAXALIGN(SizeOfPageHeaderData) -
		MAXALIGN(sizeof(PgturbohybridGraphPageOpaqueData));

	return Max(1, (int) (usable / (tupleSize + sizeof(ItemIdData))));
}

static inline int
PgturbohybridGraphPageCount(uint32 nodeCount, int tuplesPerPage)
{
	if (nodeCount == 0)
		return 0;

	return (nodeCount + tuplesPerPage - 1) / tuplesPerPage;
}

Oid			PgturbohybridGraphRelFileNumber(Relation index);
void		PgturbohybridGraphInitBlockMap(BlockNumber *blknos, int count);
bool		PgturbohybridGraphEnsureBlockMap(Relation index, BlockNumber startBlkno, int pageCount,
							  uint16 pageKind, BlockNumber *blknos);
BlockNumber PgturbohybridGraphGetChainBlockNumber(Relation index, BlockNumber startBlkno,
									 int pageNo, int pageCount, uint16 pageKind);
BlockNumber PgturbohybridGraphGetMappedBlockNumber(BlockNumber startBlkno, int pageNo,
									 BlockNumber *blknos);
bool		PgturbohybridGraphResolveChainBlockNumber(Relation index, BlockNumber startBlkno,
										   int pageNo, int pageCount, uint16 pageKind,
										   BlockNumber *blknos, BlockNumber *blkno);
bool		PgturbohybridGraphReadMeta(Relation index, PgturbohybridGraphMetaPageData *meta);
void		PgturbohybridGraphFinishPage(Buffer buf);
void		PgturbohybridGraphAppendPage(Relation index, ForkNumber forkNum, Buffer *buf,
							Page *page, uint16 pageKind);
OffsetNumber PgturbohybridGraphAppendTuple(Relation index, ForkNumber forkNum,
								BlockNumber *startBlkno, uint16 pageKind,
								Item tuple, Size tupleSize, uint16 graphOpKind,
								BlockNumber *insertBlkno);

bool		PgturbohybridGraphLoadCorrection(Relation index, int dimensions,
							  float **ecShift, float **ecScale);
void		PgturbohybridGraphCopyPayloadValues(PgturbohybridQuantBuildState *state, int32 *payloads,
									 uint16 *payloadMask, Datum *values,
									 bool *isnull);
void		PgturbohybridGraphExactRescore(Relation index, PgturbohybridGraphScanOpaque so, Datum query,
								 PgturbohybridGraphMetaPageData *meta,
								 PgturbohybridGraphScanNode *nodes,
								 PgturbohybridGraphResult *results, int count);
uint32		PgturbohybridGraphInsertValueInPlace(Relation index, IndexInfo *indexInfo,
									  ItemPointer heap_tid, Datum value,
									  Datum *values, bool *isnull);
int			PgturbohybridGraphPickLevel(uint32 nodeId, int m);
int			PgturbohybridGraphTraverse(Relation index, PgturbohybridGraphScanOpaque so,
						 PgturbohybridGraphMetaPageData *meta,
						 PgturbohybridGraphScanStorage *storage,
						 PgturbohybridGraphResult *results, int resultTarget,
						 int searchEf, Datum query, int payloadSlot,
						 int32 payloadValue);
void		PgturbohybridQuantUpdateMetaPage(Relation index, PgturbohybridQuantBuildState *state,
								  BlockNumber codeStart, BlockNumber adjStart,
								  BlockNumber exactStart,
								  BlockNumber correctionStart);
bool		PgturbohybridGraphLoadCodePage(Relation index, PgturbohybridGraphScanOpaque so,
								PgturbohybridGraphMetaPageData *meta,
								PgturbohybridGraphScanStorage *storage,
								uint32 nodeId);
bool		PgturbohybridGraphLoadAdjPage(Relation index, PgturbohybridGraphScanOpaque so,
							   PgturbohybridGraphMetaPageData *meta,
							   PgturbohybridGraphScanStorage *storage,
							   uint32 nodeId, int level);
void		PgturbohybridGraphCollectVacuumStats(Relation index, PgturbohybridGraphMetaPageData *meta,
									  int64 *liveNodes, int64 *deadNodes,
									  int64 *adjacencyRefs,
									  int64 *deadNeighborRefs);
bool		PgturbohybridGraphPayloadRefRange(PgturbohybridGraphScanStorage *storage, int payloadSlot,
								   int32 payloadValue, uint32 *firstIndex,
								   uint32 *refCount);
void		PgturbohybridGraphAppendInsertedExact(Relation index, BlockNumber *exactStart,
									   uint32 nodeId, Vector *vector,
									   int dimensions, BlockNumber *exactBlkno,
									   OffsetNumber *exactOffno);
bool		PgturbohybridGraphExactByteOffsetIsValid(OffsetNumber offno);
bool		PgturbohybridGraphReadExactVectorInto(Relation index, PgturbohybridGraphScanNode *node,
									   int dimensions, char *dest,
									   PgturbohybridGraphScanOpaque so);
Vector	   *PgturbohybridGraphReadExactVector(Relation index, PgturbohybridGraphScanNode *node,
								   int dimensions);
BlockNumber PgturbohybridGraphWriteExactPages(PgturbohybridQuantBuildState *state);
void		PgturbohybridGraphInvalidateCaches(Relation index);
void		PgturbohybridGraphInitScanStorage(Relation index, PgturbohybridGraphMetaPageData *meta,
							   PgturbohybridGraphScanStorage *storage,
							   PgturbohybridGraphCacheInitInfo *info);
PgturbohybridGraphNativeCache *PgturbohybridGraphInitInsertStorage(Relation index, PgturbohybridGraphMetaPageData *meta,
										  PgturbohybridGraphScanStorage *storage);
void		PgturbohybridGraphAppendInsertCacheNode(PgturbohybridGraphNativeCache *cache,
										 PgturbohybridGraphMetaPageData *meta,
										 uint32 nodeId, ItemPointer heapTid,
										 int nodeLevel, Vector *vector,
										 uint8 *code, uint8 *residualSketch,
										 float scale, float norm, float codeNorm,
										 float ecCorrection,
										 int32 *payloads, uint16 payloadMask,
										 BlockNumber exactBlkno,
										 OffsetNumber exactOffno,
										 uint32 **selected,
										 int *selectedCounts,
										 BlockNumber *adjBlknos,
										 OffsetNumber *adjOffnos,
										 BlockNumber codeStart,
										 BlockNumber adjStart,
										 BlockNumber exactStart,
										 uint32 entryNodeId,
										 uint16 graphMaxLevel);
int64		PgturbohybridGraphGetActiveLimitTupleTarget(void);
double		PgturbohybridGraphGetActiveEstimatedFilterSelectivity(void);
bool		PgturbohybridGraphGetActivePayloadInt4Filter(AttrNumber *heap_attno, int32 *value);
void		PgturbohybridGraphSeedScanContext(PgturbohybridGraphScanOpaque so, int64 tuple_target,
							   double estimated_filter_selectivity);
int			PgturbohybridGraphCollectDenseCandidates(IndexScanDesc scan, int targetK,
										  TqDenseCandidate **outCandidates,
										  MemoryContext resultCtx,
										  TqDenseCandidateStats *stats);
const char *PgturbohybridGraphDenseWideningReasonName(int reason);
const char *PgturbohybridGraphDenseAdaptiveWideningModeName(int mode);
const char *PgturbohybridGraphDenseAdaptiveWideningReasonName(int reason);
const char *PgturbohybridGraphDenseLocalExpansionModeName(int mode);
const char *PgturbohybridGraphDenseBudgetPolicyNameExternal(int policy);
const char *PgturbohybridGraphRescoreBandPolicyNameExternal(int policy);

#endif
