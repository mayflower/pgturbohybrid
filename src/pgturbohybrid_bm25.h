#ifndef PGTURBOHYBRID_BM25_H
#define PGTURBOHYBRID_BM25_H

#include "postgres.h"

#include "nodes/execnodes.h"
#include "tsearch/ts_type.h"
#include "utils/rel.h"

#include "pgturbohybrid.h"

typedef struct PgturbohybridBm25BuildDoc
{
	uint32		nodeId;
	ItemPointerData heaptid;
	uint32		docLen;
} PgturbohybridBm25BuildDoc;

typedef struct PgturbohybridBm25TermTuple
{
	uint64		termHash;
	uint32		nodeId;
	uint32		termOffset;
	uint16		tf;
	uint16		termLen;
} PgturbohybridBm25TermTuple;

typedef struct PgturbohybridBm25Result
{
	uint32		nodeId;
	ItemPointerData heaptid;
	float8		bm25Score;
	int32		rank;
} PgturbohybridBm25Result;

typedef struct PgturbohybridBm25FusedScoreBoundDenseEntry
{
	uint32		nodeId;
	double		contribution;
} PgturbohybridBm25FusedScoreBoundDenseEntry;

typedef struct PgturbohybridBm25FusedScoreBoundContext
{
	bool		enabled;
	double		alpha;
	double		kthScore;
	double		maxDenseContribution;
	const PgturbohybridBm25FusedScoreBoundDenseEntry *dense;
	uint32		denseCount;
} PgturbohybridBm25FusedScoreBoundContext;

static inline double
PgturbohybridBm25NormalizeSaturating(double score)
{
	if (score <= 0.0)
		return 0.0;
	return score / (score + 1.0);
}

typedef struct PgturbohybridBm25QueryStats
{
	uint32		queryTerms;
	uint64		postingsDecoded;
	uint64		blocksVisited;
	uint64		blocksSkipped;
	uint64		fusedScoreBoundBlocksPruned;
	uint64		fusedScoreBoundCandidatesPruned;
	uint32		candidatesScored;
	uint32		accumulatorEntries;
	uint64		cacheBytes;
	uint32		cacheLexiconEntries;
	bool		cacheHit;
	uint64		cacheBuildUs;
	bool		cacheDocstatsLoaded;
	bool		cacheLivenessLoaded;
	bool		docstatsLoadedThisQuery;
	bool		livenessLoadedThisQuery;
	uint64		docstatsBytes;
	uint64		livenessBytes;
	bool		coldCacheONWork;
	double		postingsDecodeRatio;
	bool		commonTermFallback;
	uint64		wandPruned;
	uint64		hotPostingsCacheHits;
	uint64		hotPostingsCacheMisses;
	uint64		hotPostingsCacheBytes;
	uint64		hotPostingsCacheEvictions;
	int			deltaLookupMode;
	uint64		deltaPagesScanned;
	uint64		deltaTermPagesRead;
	uint64		deltaBlocksVisited;
	uint64		deltaPostingsDecoded;
	uint64		deltaCacheBytes;
	uint32		deltaCacheTerms;
	bool		deltaCacheHit;
	uint64		wandIterations;
	uint64		wandThresholdUpdates;
	uint64		wandActiveSorts;
	uint64		wandHeapUpdates;
	uint64		wandFullReorders;
	uint64		wandBoundTighteningHits;
	int			wandBoundType;
	uint64		wandHeapReplacements;
	int			strategy;
	uint32		andDriverDf;
	uint32		andVerifiedCandidates;
	uint32		andRejectedCandidates;
	uint32		impactTerms;
	uint32		impactTiersRead;
	uint64		impactPostingsRead;
	double		impactRemainingUpperBound;
	bool		impactEarlyStop;
	bool		impactExactSafe;
	bool		impactFullPostingsAvoided;
	bool		impactLoadedFromStorage;
	bool		impactBuiltLazily;
	uint64		impactLazyPostingsScanned;
	int			accumulatorMode;
	uint64		accumulatorHashLookups;
	uint64		accumulatorDenseUpdates;
	uint64		finalHeapReplacements;
	uint32		finalSortedCount;
	bool		fullSortAvoided;
	int			queryShape;
	int			booleanEvalMode;
	uint64		booleanEvalCalls;
	int			decodeKernel;
	int			scoreKernel;
	uint64		simdBlocks;
	uint64		scalarTailPostings;
	uint64		prefetches;
	int			heapTsvectorRerankMode;
	uint32		heapTsvectorRerankCount;
	uint64		heapTsvectorRerankFetchUs;
	uint64		heapTsvectorRerankScoreUs;
	bool		heapTsvectorRerankTopKChanged;
	bool		usedWand;
} PgturbohybridBm25QueryStats;

typedef struct PgturbohybridBm25PlanningStats
{
	uint32		docCount;
	uint32		termCount;
	uint32		termTupleCount;
	uint32		deltaDocCount;
	uint32		deltaTermCount;
	uint32		postingsPages;
	uint32		blockMaxPages;
	uint32		deltaPages;
	uint32		deltaTermPages;
	bool		hasBm25;
} PgturbohybridBm25PlanningStats;

typedef struct PgturbohybridBm25MemoryEstimate
{
	bool		available;
	uint32		docCount;
	uint32		deltaDocCount;
	uint32		termCount;
	uint32		termTupleCount;
	uint32		postingsPages;
	uint32		blockMaxPages;
	uint32		deltaPages;
	uint32		deltaTermPages;
	uint64		docLensBytes;
	uint64		heapTidsBytes;
	uint64		liveNodesBytes;
	uint64		lexiconBytes;
	uint64		estimatedBaseCacheBytes;
} PgturbohybridBm25MemoryEstimate;

typedef struct PgturbohybridBm25QuerySignals
{
	bool		valid;
	uint32		queryTerms;
	uint32		resolvedTerms;
	uint32		docCount;
	double		maxIdf;
	double		meanIdf;
	uint32		minPostings;
	bool		hasIdentifierToken;
	int			queryShape;
} PgturbohybridBm25QuerySignals;

#define PGTURBOHYBRID_BM25_VERSION 1
#define PGTURBOHYBRID_BM25_META_TUPLE_TYPE		0x61
#define PGTURBOHYBRID_BM25_DOCSTATS_TUPLE_TYPE	0x62
#define PGTURBOHYBRID_BM25_LEXICON_TUPLE_TYPE	0x63
#define PGTURBOHYBRID_BM25_POSTINGS_TUPLE_TYPE	0x64
#define PGTURBOHYBRID_BM25_BLOCKMAX_TUPLE_TYPE	0x65
#define PGTURBOHYBRID_BM25_DELTA_TUPLE_TYPE		0x66
#define PGTURBOHYBRID_BM25_IMPACT_TUPLE_TYPE		0x67
#define PGTURBOHYBRID_BM25_DELTA_TERM_TUPLE_TYPE	0x68
#define PGTURBOHYBRID_BM25_DELTA_DIRECTORY_TUPLE_TYPE 0x69
#define PGTURBOHYBRID_BM25_DELTA_TERM_BUCKETS	64
#define PGTURBOHYBRID_BM25_META_FLAG_TFNORM_Q16 0x0001
#define PGTURBOHYBRID_BM25_META_FLAG_IMPACT_HEAD 0x0002
#define PGTURBOHYBRID_BM25_POSTINGS_ENCODING_MASK 0x00ff
#define PGTURBOHYBRID_BM25_POSTINGS_ENCODING_TFNORM_Q16 0x8000
#define PGTURBOHYBRID_BM25_POSTINGS_ENCODING_DELTA_VARINT 1
#define PGTURBOHYBRID_BM25_POSTINGS_ENCODING_DELTA16 2
#define PGTURBOHYBRID_BM25_POSTINGS_ENCODING_OFFSET16 3
#define PGTURBOHYBRID_BM25_POSTINGS_ENCODING_OFFSET16_SOA 4

typedef struct PgturbohybridBm25MetaTupleData
{
	uint8		type;
	uint8		reserved1;
	uint16		reserved2;
	uint32		bm25Version;
	uint32		docCount;
	uint64		totalDocLen;
	float4		avgDocLen;
	float4		k1;
	float4		b;
	uint32		termCount;
	uint32		termTupleCount;
	uint32		maxDocLen;
	BlockNumber docStatsStartBlkno;
	BlockNumber lexiconStartBlkno;
	BlockNumber postingsStartBlkno;
	BlockNumber blockMaxStartBlkno;
	BlockNumber impactStartBlkno;
	BlockNumber deltaStartBlkno;
	BlockNumber deltaTermDirectoryBlkno;
	uint64		deltaGeneration;
	uint32		deltaDocCount;
	uint64		deltaTotalDocLen;
	uint32		deltaTermCount;
	uint32		postingsPages;
	uint32		blockMaxPages;
	uint32		impactPages;
	uint32		deltaPages;
	uint32		deltaTermPages;
	uint64		lastCompactionGeneration;
	uint32		compactionCount;
} PgturbohybridBm25MetaTupleData;

typedef PgturbohybridBm25MetaTupleData *PgturbohybridBm25MetaTuple;

typedef struct TqBm25DocStat
{
	uint32		docLen;
	uint16		flags;
	uint16		reserved;
} TqBm25DocStat;

typedef struct PgturbohybridBm25DocStatsTupleData
{
	uint8		type;
	uint8		reserved1;
	uint16		count;
	uint32		startNodeId;
	TqBm25DocStat docs[FLEXIBLE_ARRAY_MEMBER];
} PgturbohybridBm25DocStatsTupleData;

typedef PgturbohybridBm25DocStatsTupleData *PgturbohybridBm25DocStatsTuple;

typedef struct PgturbohybridBm25Posting
{
	uint32		nodeId;
	uint16		tf;
	uint16		reserved;
} PgturbohybridBm25Posting;

typedef struct PgturbohybridBm25PostingsTupleData
{
	uint8		type;
	uint8		reserved1;
	uint16		count;
	uint32		termId;
	uint32		chunkNo;
	uint32		firstNodeId;
	uint32		lastNodeId;
	BlockNumber nextBlkno;
	OffsetNumber nextOffno;
	uint16		maxTf;
	uint16		encoding;
	uint16		maxTfNormQ16;
	uint16		payloadBytes;
	float4		maxScoreFactor;
	char		payload[FLEXIBLE_ARRAY_MEMBER];
} PgturbohybridBm25PostingsTupleData;

typedef PgturbohybridBm25PostingsTupleData *PgturbohybridBm25PostingsTuple;

typedef struct PgturbohybridBm25BlockMaxTupleData
{
	uint8		type;
	uint8		reserved1;
	uint16		count;
	uint32		termId;
	uint32		firstNodeId;
	uint32		lastNodeId;
	uint16		maxTf;
	uint16		reserved2;
	float4		maxScoreUpperBound;
} PgturbohybridBm25BlockMaxTupleData;

typedef PgturbohybridBm25BlockMaxTupleData *PgturbohybridBm25BlockMaxTuple;

typedef struct PgturbohybridBm25ImpactTupleEntry
{
	uint32		nodeId;
	uint16		tfNormQ16;
	uint16		reserved;
	float4		score;
} PgturbohybridBm25ImpactTupleEntry;

typedef struct PgturbohybridBm25ImpactTupleData
{
	uint8		type;
	uint8		reserved1;
	uint16		count;
	uint32		termId;
	BlockNumber nextBlkno;
	OffsetNumber nextOffno;
	uint16		reserved2;
	PgturbohybridBm25ImpactTupleEntry entries[FLEXIBLE_ARRAY_MEMBER];
} PgturbohybridBm25ImpactTupleData;

typedef PgturbohybridBm25ImpactTupleData *PgturbohybridBm25ImpactTuple;

typedef struct PgturbohybridBm25DeltaTerm
{
	uint64		termHash;
	uint32		termOffset;
	uint16		tf;
	uint16		termLen;
} PgturbohybridBm25DeltaTerm;

typedef struct PgturbohybridBm25DeltaTupleData
{
	uint8		type;
	uint8		reserved1;
	uint16		termCount;
	uint32		nodeId;
	ItemPointerData heaptid;
	uint32		docLen;
	uint32		termBytesLen;
	PgturbohybridBm25DeltaTerm terms[FLEXIBLE_ARRAY_MEMBER];
	/* followed by term bytes */
} PgturbohybridBm25DeltaTupleData;

typedef PgturbohybridBm25DeltaTupleData *PgturbohybridBm25DeltaTuple;

typedef struct PgturbohybridBm25DeltaTermPosting
{
	uint32		nodeId;
	ItemPointerData heaptid;
	uint32		docLen;
	uint16		tf;
	uint16		reserved;
} PgturbohybridBm25DeltaTermPosting;

typedef struct PgturbohybridBm25DeltaTermTupleData
{
	uint8		type;
	uint8		reserved1;
	uint16		postingCount;
	uint64		termHash;
	uint32		termBytesLen;
	uint16		termLen;
	uint16		reserved2;
	PgturbohybridBm25DeltaTermPosting postings[FLEXIBLE_ARRAY_MEMBER];
	/* followed by term bytes */
} PgturbohybridBm25DeltaTermTupleData;

typedef PgturbohybridBm25DeltaTermTupleData *PgturbohybridBm25DeltaTermTuple;

typedef struct PgturbohybridBm25DeltaTermBucket
{
	BlockNumber startBlkno;
	BlockNumber tailBlkno;
	uint32		pages;
} PgturbohybridBm25DeltaTermBucket;

typedef struct PgturbohybridBm25DeltaDirectoryTupleData
{
	uint8		type;
	uint8		reserved1;
	uint16		bucketCount;
	uint32		reserved2;
	PgturbohybridBm25DeltaTermBucket buckets[PGTURBOHYBRID_BM25_DELTA_TERM_BUCKETS];
} PgturbohybridBm25DeltaDirectoryTupleData;

typedef PgturbohybridBm25DeltaDirectoryTupleData *PgturbohybridBm25DeltaDirectoryTuple;

typedef enum PgturbohybridBm25DeltaLookupMode
{
	PGTURBOHYBRID_BM25_DELTA_LOOKUP_NONE = 0,
	PGTURBOHYBRID_BM25_DELTA_LOOKUP_DOC_SCAN,
	PGTURBOHYBRID_BM25_DELTA_LOOKUP_TERM_SEGMENT
} PgturbohybridBm25DeltaLookupMode;

typedef struct PgturbohybridBm25LexiconEntryData
{
	uint8		type;
	uint8		reserved1;
	uint16		termLen;
	uint64		termHash;
	uint32		termId;
	uint32		df;
	uint32		cf;
	BlockNumber postingsBlkno;
	OffsetNumber postingsOffno;
	uint16		reserved2;
	uint32		postingsChunkCount;
	uint32		postingsBytes;
	BlockNumber blockMaxBlkno;
	OffsetNumber blockMaxOffno;
	uint16		reserved3;
	BlockNumber impactBlkno;
	OffsetNumber impactOffno;
	uint16		impactCount;
	char		termBytes[FLEXIBLE_ARRAY_MEMBER];
} PgturbohybridBm25LexiconEntryData;

typedef PgturbohybridBm25LexiconEntryData *PgturbohybridBm25LexiconEntry;

void		PgturbohybridBm25BuildEmpty(Relation index);
void		PgturbohybridBm25BuildCollect(Relation heap, Relation index, IndexInfo *indexInfo);
void		PgturbohybridBm25AppendDelta(Relation index, uint32 nodeId,
									ItemPointer heapTid, Datum tsvectorDatum);
bool		PgturbohybridBm25MaybeCompact(Relation index);
void		PgturbohybridBm25InvalidateCache(Relation index);
bool		PgturbohybridBm25GetPlanningStats(Relation index,
										 PgturbohybridBm25PlanningStats *stats);
bool		PgturbohybridBm25EstimateMemory(Relation index,
									  const PgturbohybridGraphMetaPageData *graphMeta,
									  PgturbohybridBm25MemoryEstimate *estimate);
bool		PgturbohybridBm25AnalyzeQuerySignals(Relation index, TSQuery query,
									  MemoryContext memoryContext,
									  PgturbohybridBm25QuerySignals *signals);
bool		PgturbohybridBm25QueryHasPhrase(TSQuery query);
int			PgturbohybridBm25TopK(Relation index, TSQuery query, int32 k,
							  bool useWand, MemoryContext memoryContext,
							  PgturbohybridBm25Result **results,
							  PgturbohybridBm25QueryStats *stats,
							  const PgturbohybridBm25FusedScoreBoundContext *fusedBound);

typedef enum PgturbohybridBm25Kernel
{
	PGTURBOHYBRID_BM25_KERNEL_SCALAR,
	PGTURBOHYBRID_BM25_KERNEL_NEON,
	PGTURBOHYBRID_BM25_KERNEL_AVX2,
	PGTURBOHYBRID_BM25_KERNEL_NEON_SOA,
	PGTURBOHYBRID_BM25_KERNEL_AVX2_SOA
}			PgturbohybridBm25Kernel;

extern int	pgturbohybrid_last_bm25_decode_kernel;
extern int	pgturbohybrid_last_bm25_score_kernel;
extern uint64 pgturbohybrid_last_bm25_simd_blocks;
extern uint64 pgturbohybrid_last_bm25_scalar_tail_postings;

const char *PgturbohybridBm25KernelName(int kernel);
const char *PgturbohybridBm25QueryShapeName(int shape);
const char *PgturbohybridBm25BooleanEvalModeName(int mode);
const char *PgturbohybridBm25WandBoundTypeName(int boundType);
const char *PgturbohybridBm25DeltaLookupModeName(int mode);

#endif
