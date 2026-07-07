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

typedef struct PgturbohybridMultivectorStats
{
	bool		branchUsed;
	uint32		candidatesRequested;
	uint32		candidatesEffective;
	uint32		candidates;
	bool		enabled;
	uint32		queryVectors;
	uint32		docVectorsLimit;
	uint64		subvectorSearches;
	uint64		rawSubvectorHits;
	bool		adaptiveWideningTriggered;
	uint32		adaptiveInitialRawTarget;
	uint32		adaptiveFinalRawTarget;
	char		docMapSource[16];
	char		candidateSource[48];
	char		candidatePath[48];
	char		proxyEncoderKind[32];
	char		graphMode[24];
	uint64		proxyGraphSearches;
	bool		exactTokenScanEnabled;
	uint64		exactTokenScanNodesScored;
	bool		plainFallbackUsed;
	char		plainFallbackReason[48];
	uint64		plainFallbackDocsScored;
	uint64		plainFallbackPairs;
	bool		docGraphPrototypeEnabled;
	uint64		docGraphNodes;
	uint64		docGraphDocsScored;
	uint64		docGraphEdgesVisited;
	uint32		docGraphCandidates;
	uint32		docGraphSearchEf;
	uint32		docGraphOversampling;
	uint32		docGraphRescoreK;
	uint32		docGraphEntrySampleConfigured;
	uint32		docGraphEntrySampleEffective;
	uint32		docGraphEntrySampleScored;
	uint64		docGraphQuantizedScores;
	char		docGraphStorageKind[16];
	char		docGraphRescoreSource[16];
	uint32		docGraphExactRerankDocs;
	uint64		docGraphHeapFetches;
	char		docGraphWarning[96];
	uint32		proxyCandidateTarget;
	uint32		proxyCandidatesReturned;
	uint32		exactRerankKEffective;
	char		docStorageCacheRequested[16];
	char		docStorageCacheEffective[16];
	uint32		centroidCount;
	uint32		centroidPrerankDocs;
	uint32		fullMaxsimRerankDocs;
	char		docSidecarCacheMode[16];
	uint64		docSidecarPagesRead;
	uint64		docSidecarCacheHits;
	uint64		docSidecarCacheMisses;
	uint64		docSidecarBytesTouched;
	uint64		docSidecarVectorsLoaded;
	uint64		docSidecarDocMapPagesRead;
	uint64		docSidecarDocMapBytesTouched;
	uint64		docSidecarResidentVectorsLoaded;
	uint64		docSidecarResidentBytesLoaded;
	uint64		docSidecarVectorChunkRefBytesTouched;
	uint64		docSidecarPagedVectorPagesRead;
	uint64		docSidecarPagedVectorBytesTouched;
	uint64		sidecarPageReadUs;
	uint64		sidecarVectorReconstructUs;
	uint64		tokensOriginal;
	uint64		tokensPooled;
	bool		reservoirsEnabled;
	uint32		reservoirScoreDocs;
	uint32		reservoirCoverageDocs;
	uint32		reservoirMeanDocs;
	uint32		reservoirPerTokenDocs;
	uint32		reservoirBm25Docs;
	uint32		reservoirUnionDocs;
	uint32		reservoirDuplicates;
	bool		bm25InjectionEnabled;
	uint32		bm25InjectionCandidates;
	uint32		bm25InjectionCandidateLimit;
	uint32		bm25InjectionPoolSize;
	char		bm25InjectionLimitReason[32];
	uint32		bm25InjectionRetained;
	uint32		bm25InjectionExactReranked;
	uint64		docMapBytes;
	/* Token-local unique document hits summed across query tokens. */
	uint64		uniqueDocs;
	/* Raw hits whose document was already seen for the same query token. */
	uint64		duplicateDocHits;
	uint64		maxsimUpdates;
	uint32		docCandidates;
	bool		exactRerankEnabled;
	uint32		exactRerankDocs;
	uint64		exactRerankPairs;
	char		exactRerankSource[16];
	uint64		exactRerankHeapFetches;
	uint64		exactRerankSidecarReads;
	uint64		exactRerankSidecarBytes;
	uint64		candidateSourceUs;
	uint64		docGraphTraversalUs;
	uint64		proxyCandidateUs;
	uint64		proxyGraphTraversalUs;
	uint64		proxyScoringUs;
	uint64		centroidLitePostingUs;
	uint64		quantizedInvertedPostingUs;
	uint64		sidecarLoadUs;
	uint64		heapVisibilityUs;
	uint64		exactHeapFetchUs;
	uint64		exactRerankUs;
	uint64		finalSortUs;
	char		exactKernel[16];
	char		accumulatorKind[48];
	uint64		memoryBytesEstimate;
	bool		admissionDebugEnabled;
	uint32		admissionCandidatesBeforeRerank;
	uint32		admissionCandidatesAfterTruncation;
	uint32		admissionExactRerankDocs;
	bool		admissionTruncatedByDocCandidateK;
	bool		admissionTruncatedByAccumulatorMemory;
	bool		admissionTraceAvailable;
	uint32		admissionTraceCount;
	PgturbohybridMultiVectorAdmissionTraceEntry admissionTrace[PGTURBOHYBRID_MULTIVECTOR_DEBUG_TRACE_LIMIT_MAX];
	bool		tokenStatsAvailable;
	uint32		tokenStatsCount;
	PgturbohybridMultiVectorTokenStatsEntry tokenStats[PGTURBOHYBRID_MULTIVECTOR_TOKEN_STATS_LIMIT_MAX];
}			PgturbohybridMultivectorStats;

typedef struct PgturbohybridMultivectorScanStats
{
	bool		enabled;
	uint32		queryVectors;
	uint32		docVectorsLimit;
	uint64		subvectorSearches;
	uint64		rawSubvectorHits;
	bool		adaptiveWideningTriggered;
	uint32		adaptiveInitialRawTarget;
	uint32		adaptiveFinalRawTarget;
	int			docMapSource;
	char		candidateSource[48];
	char		candidatePath[48];
	char		proxyEncoderKind[32];
	char		graphMode[24];
	uint64		proxyGraphSearches;
	bool		exactTokenScanEnabled;
	uint64		exactTokenScanNodesScored;
	bool		plainFallbackUsed;
	char		plainFallbackReason[48];
	uint64		plainFallbackDocsScored;
	uint64		plainFallbackPairs;
	bool		docGraphPrototypeEnabled;
	uint64		docGraphNodes;
	uint64		docGraphDocsScored;
	uint64		docGraphEdgesVisited;
	uint32		docGraphCandidates;
	uint32		docGraphSearchEf;
	uint32		docGraphOversampling;
	uint32		docGraphRescoreK;
	uint32		docGraphEntrySampleConfigured;
	uint32		docGraphEntrySampleEffective;
	uint32		docGraphEntrySampleScored;
	uint64		docGraphQuantizedScores;
	char		docGraphStorageKind[16];
	char		docGraphRescoreSource[16];
	uint32		docGraphExactRerankDocs;
	uint64		docGraphHeapFetches;
	char		docGraphWarning[96];
	uint32		proxyCandidateTarget;
	uint32		proxyCandidatesReturned;
	uint32		exactRerankKEffective;
	char		docStorageCacheRequested[16];
	char		docStorageCacheEffective[16];
	uint32		centroidCount;
	uint32		centroidPrerankDocs;
	uint32		fullMaxsimRerankDocs;
	char		docSidecarCacheMode[16];
	uint64		docSidecarPagesRead;
	uint64		docSidecarCacheHits;
	uint64		docSidecarCacheMisses;
	uint64		docSidecarBytesTouched;
	uint64		docSidecarVectorsLoaded;
	uint64		docSidecarDocMapPagesRead;
	uint64		docSidecarDocMapBytesTouched;
	uint64		docSidecarResidentVectorsLoaded;
	uint64		docSidecarResidentBytesLoaded;
	uint64		docSidecarVectorChunkRefBytesTouched;
	uint64		docSidecarPagedVectorPagesRead;
	uint64		docSidecarPagedVectorBytesTouched;
	uint64		sidecarPageReadUs;
	uint64		sidecarVectorReconstructUs;
	uint64		tokensOriginal;
	uint64		tokensPooled;
	bool		reservoirsEnabled;
	uint32		reservoirScoreDocs;
	uint32		reservoirCoverageDocs;
	uint32		reservoirMeanDocs;
	uint32		reservoirPerTokenDocs;
	uint32		reservoirBm25Docs;
	uint32		reservoirUnionDocs;
	uint32		reservoirDuplicates;
	bool		bm25InjectionEnabled;
	uint32		bm25InjectionCandidates;
	uint32		bm25InjectionCandidateLimit;
	uint32		bm25InjectionPoolSize;
	char		bm25InjectionLimitReason[32];
	uint32		bm25InjectionRetained;
	uint32		bm25InjectionExactReranked;
	uint64		docMapBytes;
	/* Token-local unique document hits summed across query tokens. */
	uint64		uniqueDocs;
	/* Raw hits whose document was already seen for the same query token. */
	uint64		duplicateDocHits;
	uint64		maxsimUpdates;
	uint32		docCandidates;
	bool		exactRerankEnabled;
	uint32		exactRerankDocs;
	uint64		exactRerankPairs;
	int			exactRerankSource;
	uint64		exactRerankHeapFetches;
	uint64		exactRerankSidecarReads;
	uint64		exactRerankSidecarBytes;
	uint64		candidateSourceUs;
	uint64		docGraphTraversalUs;
	uint64		proxyCandidateUs;
	uint64		proxyGraphTraversalUs;
	uint64		proxyScoringUs;
	uint64		centroidLitePostingUs;
	uint64		quantizedInvertedPostingUs;
	uint64		sidecarLoadUs;
	uint64		heapVisibilityUs;
	uint64		exactHeapFetchUs;
	uint64		exactRerankUs;
	uint64		finalSortUs;
	char		exactKernel[16];
	char		accumulatorKind[48];
	uint64		memoryBytesEstimate;
	bool		admissionDebugEnabled;
	uint32		admissionCandidatesBeforeRerank;
	uint32		admissionCandidatesAfterTruncation;
	uint32		admissionExactRerankDocs;
	bool		admissionTruncatedByDocCandidateK;
	bool		admissionTruncatedByAccumulatorMemory;
	bool		admissionTraceAvailable;
	uint32		admissionTraceCount;
	PgturbohybridMultiVectorAdmissionTraceEntry admissionTrace[PGTURBOHYBRID_MULTIVECTOR_DEBUG_TRACE_LIMIT_MAX];
	bool		tokenStatsAvailable;
	uint32		tokenStatsCount;
	PgturbohybridMultiVectorTokenStatsEntry tokenStats[PGTURBOHYBRID_MULTIVECTOR_TOKEN_STATS_LIMIT_MAX];
}			PgturbohybridMultivectorScanStats;

typedef struct PgturbohybridBm25SnapshotStats
{
	bool		branchAvailable;
	bool		branchUsed;
	uint32		candidatesEffective;
	bool		kDefaulted;
	bool		cacheHit;
	uint64		cacheBuildUs;
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
	uint32		terms;
	uint64		fusedScoreBoundBlocksPruned;
	uint64		fusedScoreBoundCandidatesPruned;
	char		heapTSVectorRerankMode[16];
	uint32		heapTSVectorRerankCount;
	uint64		heapTSVectorRerankFetchUs;
	uint64		heapTSVectorRerankScoreUs;
	bool		heapTSVectorRerankTopKChanged;
	char		normMode[16];
	uint64		elapsedUs;
}			PgturbohybridBm25SnapshotStats;

typedef struct PgturbohybridBm25LastScanStats
{
	bool		branchAvailable;
	bool		branchUsed;
	uint32		candidatesRequested;
	uint32		candidatesEffective;
	bool		kDefaulted;
	uint32		candidates;
	char		budgetReason[48];
	double		denseConfidence;
	int			hybridBoundMode;
	uint32		hybridBoundStopRank;
	uint32		hybridBoundSkippedEstimated;
	double		hybridBoundThreshold;
	bool		hybridBoundSafe;
	uint32		only;
	uint32		terms;
	uint64		postingsDecoded;
	uint64		blocksVisited;
	uint64		blocksSkipped;
	uint64		fusedScoreBoundBlocksPruned;
	uint64		fusedScoreBoundCandidatesPruned;
	uint32		candidatesScored;
	int			heapTSVectorRerankMode;
	uint32		heapTSVectorRerankCount;
	uint64		heapTSVectorRerankFetchUs;
	uint64		heapTSVectorRerankScoreUs;
	bool		heapTSVectorRerankTopKChanged;
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
	char		normMode[16];
	uint64		elapsedUs;
}			PgturbohybridBm25LastScanStats;

typedef struct PgturbohybridDenseSnapshotStats
{
	bool		branchUsed;
	uint32		candidatesEffective;
	bool		kDefaulted;
	char		normMode[16];
	uint64		elapsedUs;
}			PgturbohybridDenseSnapshotStats;

typedef struct PgturbohybridDenseLastScanStats
{
	bool		branchUsed;
	uint32		candidatesRequested;
	uint32		candidatesEffective;
	bool		kDefaulted;
	uint32		candidates;
	uint32		effectiveResultTarget;
	uint32		effectiveSearchEf;
	uint32		effectiveRescoreBand;
	double		highdimWideningMultiplier;
	int			wideningReason;
	int			budgetPolicy;
	int			rescoreBandPolicy;
	uint32		only;
	char		normMode[16];
	uint64		elapsedUs;
}			PgturbohybridDenseLastScanStats;

typedef struct PgturbohybridDenseScanStats
{
	uint32		candidatesRequested;
	int			budgetPolicy;
	uint32		candidatesReturned;
}			PgturbohybridDenseScanStats;

typedef struct PgturbohybridFusionSnapshotStats
{
	char		strategy[24];
	uint32		candidatesSeen;
	uint64		duplicates;
	uint64		heapReplacements;
	bool		generationArrayReused;
	bool		generationArrayReset;
	uint64		elapsedUs;
}			PgturbohybridFusionSnapshotStats;

typedef struct PgturbohybridFusionLastScanStats
{
	char		strategy[24];
	uint32		candidatesSeen;
	uint32		heapSize;
	uint64		duplicates;
	uint64		heapReplacements;
	bool		generationArrayReused;
	bool		generationArrayReset;
	uint64		elapsedUs;
}			PgturbohybridFusionLastScanStats;

typedef struct PgturbohybridFinalDiversitySnapshotStats
{
	char		mode[24];
	int32		payloadSlot;
	uint32		poolSize;
	uint32		selected;
	uint64		duplicateGroupsSuppressed;
	uint64		us;
}			PgturbohybridFinalDiversitySnapshotStats;

typedef struct PgturbohybridFinalDiversityLastScanStats
{
	int			mode;
	int32		payloadSlot;
	uint32		poolSize;
	uint32		selected;
	uint64		duplicateGroupsSuppressed;
	uint64		us;
}			PgturbohybridFinalDiversityLastScanStats;

#endif							/* PGTURBOHYBRID_SCAN_STATS_H */
