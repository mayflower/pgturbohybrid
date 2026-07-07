/*
 * pgturbohybrid_scan_stats.h
 *
 * Shared sub-struct types grouping the flat scan-stats fields that were
 * duplicated verbatim across the scan-stats god-structs
 * (PgturbohybridScanStatsSnapshot / PgturbohybridLastScanStats /
 * TqDenseCandidateStats).  Each group is defined once here and embedded as
 * a named member, so per-group marshaling collapses to struct assignment.
 */
#ifndef PGTURBOHYBRID_SCAN_STATS_H
#define PGTURBOHYBRID_SCAN_STATS_H

#include "pgturbohybrid_multivector.h"


typedef struct PgturbohybridQuantizedInvertedStats
{
	bool		sidecarAvailable;
	uint64		listsVisited;
	uint64		postingsTouched;
	uint64		postingsSelected;
	uint64		postingsSkipped;
	uint32		postingLimitPerToken;
	uint32		probeCodewordsPerToken;
	char		postingCapStrategy[32];
	uint64		docsScored;
	uint32		candidates;
	uint32		exactRerankDocs;
	char		codebookSource[16];
	uint32		codebookSize;
	uint32		codebookDim;
	char		codebookChecksum[128];
	uint32		codebookTopM;
	uint64		assignmentUs;
	uint64		queryCodewordScoreUs;
	char		queryCodewordKernel[16];
	uint64		queryCodewordScoresComputed;
	uint64		queryCodewordBlocks;
	uint64		queryCodewordTopkUs;
	bool		queryCodewordFullMatrixMaterialized;
	uint32		queryCodewordActiveQueryTokens;
	uint32		queryCodewordSkippedQueryTokens;
	uint64		listOffsetBytes;
	uint64		postingBytes;
	uint64		sidecarBytes;
	char		compactKernel[24];
	char		compactScoreSource[32];
	uint64		compactScoreUs;
	uint64		compactDocsScored;
	uint64		compactPayloadBytes;
	char		compactDocOrder[16];
	uint64		compactInnerAllocations;
	uint32		compactActiveQueryTokens;
	uint64		compactPairsEvaluated;
	uint64		compactPairsSkipped;
	uint64		compactPrefetches;
	double		compactAvgDocTokens;
	double		compactUsPerDoc;
	double		compactPayloadBytesPerDoc;
	bool		compactTopKChangedVsScalar;
	bool		precompactEnabled;
	char		precompactMode[32];
	uint32		docsTouchedBeforePrecompact;
	uint32		precompactScoreK;
	uint32		precompactCoverageK;
	uint32		precompactPerTokenK;
	uint32		compactMaxDocs;
	uint32		precompactScoreDocs;
	uint32		precompactCoverageDocs;
	uint32		precompactPerTokenDocs;
	uint32		precompactUnionDocs;
	uint32		precompactDuplicates;
	uint32		precompactPrunedDocs;
	uint64		precompactUs;
	uint32		compactDocsSkippedByPrecompact;
	char		tokenCoverageMode[24];
	uint32		activeQueryTokens;
	uint64		tokenMatchesTotal;
	uint32		tokenMatchesMax;
	uint32		minTokenMatches;
	uint64		tokenMatchFilteredDocs;
	bool		scoreBoundPruningEnabled;
	uint64		scoreBoundDocsChecked;
	uint64		scoreBoundDocsPruned;
	uint64		scoreBoundPruneUs;
	uint64		scoreBoundUnsafeFallbacks;
	uint32		candidatesBeforeBound;
	uint32		candidatesAfterBound;
}			PgturbohybridQuantizedInvertedStats;

typedef struct PgturbohybridCentroidStats
{
	bool		onlyIndex;
	bool		sidecarAvailable;
	bool		docCodesAvailable;
	uint64		listsVisited;
	uint64		docsTouched;
	uint64		prunedDocs;
	uint64		postingsTouched;
	uint64		postingsSelected;
	uint64		postingsSkipped;
	uint64		probeUs;
	uint64		postingScanUs;
	uint64		accumulateUs;
	uint64		candidateHeapUs;
	uint32		postingLimitPerToken;
	uint32		probeCentroidsPerToken;
	uint32		codewordTopM;
	double		scoreThreshold;
	double		scoreDropFromBest;
	uint64		listsSkippedByThreshold;
	char		postingCapStrategy[32];
	char		candidateScoring[32];
	uint32		candidates;
	bool		bitsetPrefilterEnabled;
	uint32		bitsetMinTokenMatches;
	uint32		bitsetListsUsed;
	uint32		bitsetDocsSet;
	uint32		bitsetDocsAfterThreshold;
	uint64		bitsetPrefilterUs;
	uint64		bitsetMemoryBytes;
	bool		upperBoundEnabled;
	uint64		upperBoundDocsChecked;
	uint64		upperBoundDocsPruned;
	uint64		upperBoundPruneUs;
	uint64		upperBoundUnsafeFallbacks;
	uint32		candidatesBeforeBound;
	uint32		candidatesAfterBound;
}			PgturbohybridCentroidStats;

typedef struct PgturbohybridProxyStats
{
	bool		onlyIndex;
	uint32		candidateLimitEffective;
	char		candidateLimitSource[32];
	uint64		graphNodesVisited;
	uint64		graphEdgesVisited;
	uint32		graphCandidatesSeen;
	uint32		candidatesReturned;
	uint64		vectorScoresComputed;
	uint64		vectorScoreUs;
	uint32		candidates;
	bool		lazySidecarVectors;
	bool		top1Admission;
	uint32		exactRerankDocs;
	uint64		fullSidecarVectorsLoaded;
	uint64		fullSidecarBytesTouched;
	uint64		fullSidecarPagesRead;
	uint64		fullSidecarLoadUs;
	uint64		fullSidecarReconstructUs;
	uint64		exactRerankHeapFetches;
	uint64		exactRerankSidecarFetches;
	uint64		exactRerankBytesTouched;
	uint64		exactRerankUs;
	bool		vectorUsesFullSidecarForGraph;
	bool		vectorNearExhaustiveSidecarTouch;
	char		vectorSidecarTouchReason[64];
}			PgturbohybridProxyStats;

typedef struct PgturbohybridSidecarStats
{
	bool		cacheBuildThisQuery;
	uint64		cacheBuildBytes;
	uint64		cacheBuildPagesRead;
	uint64		cacheBuildUs;
	uint64		queryBytesTouched;
	uint64		queryPagesRead;
	uint64		queryVectorsLoaded;
	uint64		queryLoadUs;
	uint64		queryUs;
}			PgturbohybridSidecarStats;

typedef struct PgturbohybridSparseSnapshotStats
{
	bool		branchAvailable;
	bool		branchUsed;
	uint32		terms;
	uint32		resolvedTerms;
	uint64		postingsTouched;
	uint64		candidatesScored;
	uint64		elapsedUs;
	uint32		candidatesRequested;
	uint32		candidatesEffective;
	bool		kDefaulted;
	uint32		candidates;
	int			quantBits;
	int			quantMode;
	int			encoding;
	uint64		scalarTailPostings;
	int			rerankMode;
	uint64		exactRerankCount;
	uint64		exactRerankFetchUs;
	uint64		exactRerankScoreUs;
	bool		exactRerankTopkChanged;
	int			scoreKernel;
	uint64		simdBlocks;
	bool		usedWand;
	uint64		blocksVisited;
	uint64		blocksSkipped;
	uint64		wandPruned;
	uint64		wandIterations;
	uint64		wandThresholdUpdates;
	uint64		wandHeapUpdates;
	bool		cacheHit;
	uint64		cacheBuildUs;
	uint64		cacheBytes;
	uint64		hotCacheHits;
	uint64		hotCacheMisses;
	uint64		hotCacheBytes;
	uint64		hotCacheEvictions;
	uint32		deltaPages;
	uint32		deltaTerms;
	uint64		deltaPostingsDecoded;
	bool		deltaCacheHit;
	uint32		deltaGeneration;
}			PgturbohybridSparseSnapshotStats;

typedef struct PgturbohybridLearnedProjectionStats
{
	bool		loaded;
	uint32		dim;
	uint64		weightBytes;
	char		model[128];
	char		checksum[128];
	uint64		queryEncodeUs;
}			PgturbohybridLearnedProjectionStats;

typedef struct PgturbohybridLearnedSparseStats
{
	uint32		candidates;
	uint32		retainedForMaxsim;
	uint64		branchLatencyUs;
}			PgturbohybridLearnedSparseStats;

typedef struct PgturbohybridCalibratedFusionStats
{
	bool		enabled;
	char		queryShape[32];
	double		alphaEffective;
	double		bothMatchBonus;
	char		denseNormMode[16];
	char		bm25NormMode[16];
}			PgturbohybridCalibratedFusionStats;

typedef struct PgturbohybridDbsfStats
{
	bool		enabled;
	double		branchMean[2];
	double		branchStddev[2];
	double		branchMin[2];
	double		branchMax[2];
	uint32		degenerateBranches;
}			PgturbohybridDbsfStats;

typedef struct PgturbohybridHybridBudgetStats
{
	char		budgetPolicy[16];
	char		queryShape[32];
	uint32		denseKChosen;
	uint32		bm25KChosen;
	char		budgetReason[96];
}			PgturbohybridHybridBudgetStats;

typedef struct PgturbohybridFastWeightedStats
{
	bool		enabled;
	double		alpha;
}			PgturbohybridFastWeightedStats;

#endif							/* PGTURBOHYBRID_SCAN_STATS_H */
