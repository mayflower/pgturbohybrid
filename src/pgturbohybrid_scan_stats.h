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

#endif							/* PGTURBOHYBRID_SCAN_STATS_H */
