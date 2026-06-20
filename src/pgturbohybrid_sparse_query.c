/*
 * pgturbohybrid_sparse_query.c
 *
 * Scan path for the native sparse-vector inverted index (prompt 4).  Resolves
 * the query's sparse_query terms against the lexicon, exact-OR-accumulates
 * score[node] += query_weight * doc_weight over the per-term postings, applies
 * MVCC node liveness, and returns the top-k candidates by descending score.
 * Exact float32; no quantization / SIMD / WAND / cache.
 */
#include "postgres.h"

#include <string.h>

#include "miscadmin.h"
#include "portability/instr_time.h"
#include "storage/bufmgr.h"
#include "utils/memutils.h"
#include "utils/rel.h"

#include "pgturbohybrid.h"
#include "pgturbohybrid_quant.h"
#include "pgturbohybrid_am.h"
#include "pgturbohybrid_query.h"
#include "pgturbohybrid_sparse.h"

typedef struct PgturbohybridSparseLexEntry
{
	int32		termId;
	uint32		df;
	BlockNumber postingsBlkno;
	OffsetNumber postingsOffno;
} PgturbohybridSparseLexEntry;

typedef struct PgturbohybridSparseScored
{
	uint32		nodeId;
	double		score;
} PgturbohybridSparseScored;

static int
PgturbohybridSparseLexCompare(const void *a, const void *b)
{
	const PgturbohybridSparseLexEntry *la = (const PgturbohybridSparseLexEntry *) a;
	const PgturbohybridSparseLexEntry *lb = (const PgturbohybridSparseLexEntry *) b;

	if (la->termId != lb->termId)
		return la->termId < lb->termId ? -1 : 1;
	return 0;
}

static int
PgturbohybridSparseScoredCompare(const void *a, const void *b)
{
	const PgturbohybridSparseScored *sa = (const PgturbohybridSparseScored *) a;
	const PgturbohybridSparseScored *sb = (const PgturbohybridSparseScored *) b;

	if (sa->score != sb->score)
		return sa->score > sb->score ? -1 : 1;	/* descending */
	if (sa->nodeId != sb->nodeId)
		return sa->nodeId < sb->nodeId ? -1 : 1;	/* deterministic tiebreak */
	return 0;
}

/* Read the sparse meta tuple into *out; return false if no sparse data. */
static bool
PgturbohybridSparseReadMeta(Relation index, BlockNumber metaBlkno,
							PgturbohybridSparseMetaTupleData *out)
{
	Buffer		buf;
	Page		page;
	PgturbohybridSparseMetaTuple tuple;

	if (metaBlkno == InvalidBlockNumber || metaBlkno == 0)
		return false;
	buf = ReadBuffer(index, metaBlkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	if (PageGetMaxOffsetNumber(page) < FirstOffsetNumber)
	{
		UnlockReleaseBuffer(buf);
		return false;
	}
	tuple = (PgturbohybridSparseMetaTuple)
		PageGetItem(page, PageGetItemId(page, FirstOffsetNumber));
	if (tuple->type != PGTURBOHYBRID_SPARSE_META_TUPLE_TYPE ||
		tuple->sparseVersion != PGTURBOHYBRID_SPARSE_VERSION)
	{
		UnlockReleaseBuffer(buf);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("pgturbohybrid sparse meta tuple is malformed or has an unsupported version")));
	}
	memcpy(out, tuple, sizeof(*out));
	UnlockReleaseBuffer(buf);
	return true;
}

/* Load all lexicon entries into a palloc'd, term-id-sorted array. */
static PgturbohybridSparseLexEntry *
PgturbohybridSparseLoadLexicon(Relation index, BlockNumber start, uint32 termCount,
							   uint32 *outCount)
{
	PgturbohybridSparseLexEntry *entries;
	uint32		count = 0;
	BlockNumber blkno = start;

	entries = (PgturbohybridSparseLexEntry *)
		palloc(sizeof(PgturbohybridSparseLexEntry) * Max(termCount, 1u));

	while (blkno != InvalidBlockNumber)
	{
		Buffer		buf = ReadBuffer(index, blkno);
		Page		page;
		OffsetNumber maxoff;

		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		maxoff = PageGetMaxOffsetNumber(page);
		for (OffsetNumber off = FirstOffsetNumber; off <= maxoff; off++)
		{
			PgturbohybridSparseLexiconTuple lt = (PgturbohybridSparseLexiconTuple)
				PageGetItem(page, PageGetItemId(page, off));

			if (lt->type != PGTURBOHYBRID_SPARSE_LEXICON_TUPLE_TYPE)
				continue;
			for (uint16 e = 0; e < lt->count && count < termCount; e++)
			{
				entries[count].termId = lt->entries[e].termId;
				entries[count].df = lt->entries[e].df;
				entries[count].postingsBlkno = lt->entries[e].postingsBlkno;
				entries[count].postingsOffno = lt->entries[e].postingsOffno;
				count++;
			}
		}
		blkno = PgturbohybridGraphPageGetOpaque(page)->nextblkno;
		UnlockReleaseBuffer(buf);
	}

	if (count > 1)
		qsort(entries, count, sizeof(PgturbohybridSparseLexEntry),
			  PgturbohybridSparseLexCompare);
	*outCount = count;
	return entries;
}

/*
 * Accumulate one term's postings into scores[]: walk ceil(df/PER_CHUNK)
 * consecutive chunks from (blkno, offno), adding qWeight * doc_weight per node.
 */
static uint64
PgturbohybridSparseAccumulateTerm(Relation index, BlockNumber blkno,
								  OffsetNumber offno, uint32 df, double qWeight,
								  double *scores, uint32 nodeCount)
{
	uint32		remaining = df;
	uint64		touched = 0;

	while (remaining > 0 && blkno != InvalidBlockNumber)
	{
		Buffer		buf = ReadBuffer(index, blkno);
		Page		page;
		OffsetNumber maxoff;

		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		maxoff = PageGetMaxOffsetNumber(page);

		while (remaining > 0 && offno <= maxoff)
		{
			PgturbohybridSparsePostingsTuple ct = (PgturbohybridSparsePostingsTuple)
				PageGetItem(page, PageGetItemId(page, offno));

			if (ct->type == PGTURBOHYBRID_SPARSE_POSTINGS_TUPLE_TYPE)
			{
				for (uint16 p = 0; p < ct->count; p++)
				{
					uint32		nodeId = ct->postings[p].nodeId;

					if (nodeId < nodeCount)
						scores[nodeId] += qWeight * (double) ct->postings[p].weight;
				}
				touched += ct->count;
				remaining -= Min(remaining, (uint32) ct->count);
			}
			offno++;
		}

		blkno = PgturbohybridGraphPageGetOpaque(page)->nextblkno;
		offno = FirstOffsetNumber;
		UnlockReleaseBuffer(buf);
	}
	return touched;
}

int
PgturbohybridSparseCollectCandidates(Relation index, PgturbohybridQueryHeader *query,
									 int k, PgturbohybridSparseCandidate **out,
									 MemoryContext ctx, PgturbohybridSparseScanStats *stats)
{
	PgturbohybridGraphMetaPageData graphMeta;
	PgturbohybridSparseMetaTupleData meta;
	PgturbohybridSparseLexEntry *lexicon;
	uint32		lexCount = 0;
	struct varlena *querySparse;
	const PgturbohybridSparseVectorEntry *qEntries;
	uint32		qCount = 0;
	uint32		resolved = 0;
	uint64		postingsTouched = 0;
	PgturbohybridNodeState *states;
	uint32		nodeCount = 0;
	double	   *scores;
	PgturbohybridSparseScored *scored;
	uint64		scoredCount = 0;
	MemoryContext oldCtx;
	int			resultCount;
	PgturbohybridSparseCandidate *result;
	instr_time	t0,
				t1;

	if (out != NULL)
		*out = NULL;
	if (stats != NULL)
		memset(stats, 0, sizeof(*stats));
	if (k <= 0 || query == NULL || !PgturbohybridQueryHasSparse(query))
		return 0;

	INSTR_TIME_SET_CURRENT(t0);

	if (!PgturbohybridGraphReadMeta(index, &graphMeta))
		return 0;
	if (!PgturbohybridSparseReadMeta(index, graphMeta.tqSparseMetaStartBlkno, &meta))
		return 0;
	if (stats != NULL)
		stats->branchAvailable = true;

	querySparse = PgturbohybridQueryGetSparseVector(query);
	if (querySparse == NULL)
		return 0;
	qEntries = PgturbohybridSparseVectorData(querySparse, &qCount);
	if (qCount == 0)
		return 0;

	oldCtx = MemoryContextSwitchTo(ctx);

	lexicon = PgturbohybridSparseLoadLexicon(index, meta.lexiconStartBlkno,
											 meta.termCount, &lexCount);
	states = PgturbohybridReadNodeStates(index, &graphMeta, &nodeCount);
	scores = (double *) palloc0(sizeof(double) * Max(nodeCount, 1u));

	for (uint32 i = 0; i < qCount; i++)
	{
		PgturbohybridSparseLexEntry key;
		PgturbohybridSparseLexEntry *found;

		if (qEntries[i].weight == 0.0f)
			continue;
		key.termId = qEntries[i].termId;
		found = (PgturbohybridSparseLexEntry *)
			bsearch(&key, lexicon, lexCount, sizeof(PgturbohybridSparseLexEntry),
					PgturbohybridSparseLexCompare);
		if (found == NULL)
			continue;
		resolved++;
		postingsTouched += PgturbohybridSparseAccumulateTerm(index,
															 found->postingsBlkno,
															 found->postingsOffno,
															 found->df,
															 (double) qEntries[i].weight,
															 scores, nodeCount);
	}

	/* Collect live, non-zero-scoring nodes. */
	scored = (PgturbohybridSparseScored *)
		palloc(sizeof(PgturbohybridSparseScored) * Max(nodeCount, 1u));
	for (uint32 n = 0; n < nodeCount; n++)
	{
		if (scores[n] != 0.0 && states[n].live)
		{
			scored[scoredCount].nodeId = n;
			scored[scoredCount].score = scores[n];
			scoredCount++;
		}
	}

	if (scoredCount > 1)
		qsort(scored, scoredCount, sizeof(PgturbohybridSparseScored),
			  PgturbohybridSparseScoredCompare);

	resultCount = (int) Min((uint64) k, scoredCount);
	result = resultCount > 0 ? (PgturbohybridSparseCandidate *)
		palloc(sizeof(PgturbohybridSparseCandidate) * resultCount) : NULL;
	for (int r = 0; r < resultCount; r++)
	{
		result[r].nodeId = scored[r].nodeId;
		result[r].heaptid = states[scored[r].nodeId].tid;
		result[r].score = scored[r].score;
	}

	MemoryContextSwitchTo(oldCtx);

	INSTR_TIME_SET_CURRENT(t1);
	INSTR_TIME_SUBTRACT(t1, t0);

	if (stats != NULL)
	{
		stats->branchUsed = true;
		stats->terms = qCount;
		stats->resolvedTerms = resolved;
		stats->postingsTouched = postingsTouched;
		stats->candidatesScored = scoredCount;
		stats->elapsedUs = (uint64) INSTR_TIME_GET_MICROSEC(t1);
	}

	if (out != NULL)
		*out = result;
	return resultCount;
}
