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
#include "utils/float.h"
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
	float4		scale;
	float4		maxWeight;
	BlockNumber postingsBlkno;
	OffsetNumber postingsOffno;
	BlockNumber blockMaxBlkno;
	OffsetNumber blockMaxOffno;
	uint32		blockCount;
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
				entries[count].scale = lt->entries[e].scale;
				entries[count].maxWeight = lt->entries[e].maxWeight;
				entries[count].postingsBlkno = lt->entries[e].postingsBlkno;
				entries[count].postingsOffno = lt->entries[e].postingsOffno;
				entries[count].blockMaxBlkno = lt->entries[e].blockMaxBlkno;
				entries[count].blockMaxOffno = lt->entries[e].blockMaxOffno;
				entries[count].blockCount = lt->entries[e].blockCount;
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

/* Dequantize+accumulate one decoded posting into scores[]. */
/*
 * Decode + accumulate a single postings chunk.  effMul is the effective
 * per-posting multiplier already folding in the dequant scale (effMul = qWeight
 * for f32, qWeight*scale for q8/q16).  SoA chunks route through the SIMD-capable
 * kernel dispatch; varint chunks decode node deltas with a scalar walk.
 */
static void
PgturbohybridSparseScoreChunk(PgturbohybridSparsePostingsTuple ct, int bits,
							  double effMul, int scoreKernel, double *scores,
							  uint32 nodeCount, uint64 *simdBlocks,
							  uint64 *scalarTail)
{
	uint32		n = ct->count;
	uint32		base = ct->baseNodeId;
	uint32		wwidth = PgturbohybridSparseWeightWidth(bits);
	const char *payload = ct->payload;

	if (ct->encoding == PGTURBOHYBRID_SPARSE_ENCODING_VARINT)
	{
		uint16		deltaBytes;
		Size		weightsStart;
		const uint8 *deltas = (const uint8 *) (payload + sizeof(uint16));
		int			pos = 0;
		uint32		node = base;

		memcpy(&deltaBytes, payload, sizeof(uint16));
		weightsStart = PgturbohybridSparseAlignUp(sizeof(uint16) + (Size) deltaBytes,
												  wwidth);
		for (uint32 k = 0; k < n; k++)
		{
			int			consumed;
			const char *wptr = payload + weightsStart + (Size) k * wwidth;

			node += PgturbohybridSparseVarintDecode(deltas + pos, &consumed);
			pos += consumed;
			if (node < nodeCount)
			{
				if (bits == 0)
				{
					float4		w;

					memcpy(&w, wptr, sizeof(float4));
					scores[node] += effMul * (double) w;
				}
				else if (bits == 16)
				{
					uint16		q;

					memcpy(&q, wptr, sizeof(uint16));
					scores[node] += effMul * (double) q;
				}
				else
					scores[node] += effMul * (double) (*(const uint8 *) wptr);
			}
		}
		*scalarTail += n;		/* varint node decode is inherently scalar */
	}
	else						/* SoA */
	{
		Size		offsetsStart = PgturbohybridSparseAlignUp((Size) n * wwidth, 2);
		const uint16 *offsets = (const uint16 *) (payload + offsetsStart);

		PgturbohybridSparseScoreSoa(scoreKernel, payload, offsets, n, base, bits,
									effMul, scores, nodeCount, simdBlocks,
									scalarTail);
	}
}

/*
 * Accumulate one term's postings into scores[]: physically walk the term's
 * chunks from (blkno, offno) until df postings are consumed, dequantizing per
 * the index's bit width.  effMul folds the per-term dequant scale into the
 * query weight (= qWeight for f32, qWeight*scale otherwise).
 */
static uint64
PgturbohybridSparseAccumulateTerm(Relation index, BlockNumber blkno,
								  OffsetNumber offno, uint32 df, double qWeight,
								  double scale, int bits, int scoreKernel,
								  double *scores, uint32 nodeCount,
								  uint64 *simdBlocks, uint64 *scalarTail)
{
	uint32		remaining = df;
	uint64		touched = 0;
	double		effMul = bits == 0 ? qWeight : qWeight * scale;

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
				PgturbohybridSparseScoreChunk(ct, bits, effMul, scoreKernel,
											  scores, nodeCount, simdBlocks,
											  scalarTail);
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

/* ---- Block-max WAND (prompt 9) ----------------------------------------- */

#define PGTURBOHYBRID_WAND_INF PG_UINT32_MAX

typedef struct PgturbohybridWandTerm
{
	double		effMul;			/* qWeight (f32) or qWeight*scale (quantized) */
	double		termMaxUB;		/* qWeight * lexicon maxWeight */
	int			bits;
	PgturbohybridSparseBlockMax *blocks;
	uint32		nBlocks;
	uint32		curBlock;
	uint32	   *nodes;			/* decoded node_ids of the loaded block */
	double	   *contribs;		/* decoded contributions of the loaded block */
	uint32		chunkCount;
	uint32		chunkPos;
	bool		loaded;
	uint32		curNode;		/* PGTURBOHYBRID_WAND_INF when exhausted */
} PgturbohybridWandTerm;

typedef struct PgturbohybridWandHeapEntry
{
	double		score;
	uint32		nodeId;
	ItemPointerData heaptid;
} PgturbohybridWandHeapEntry;

static inline double
PgturbohybridWandWeight(const char *wptr, int bits)
{
	if (bits == 0)
	{
		float4		w;

		memcpy(&w, wptr, sizeof(float4));
		return (double) w;
	}
	if (bits == 16)
	{
		uint16		q;

		memcpy(&q, wptr, sizeof(uint16));
		return (double) q;
	}
	return (double) (*(const uint8 *) wptr);
}

/* Decode a postings chunk at (blk,off) into nodes[]/contribs[] (contrib = effMul*weight). */
static void
PgturbohybridWandDecodeChunk(Relation index, BlockNumber blk, OffsetNumber off,
							 int bits, double effMul, uint32 *nodes,
							 double *contribs, uint32 *outCount)
{
	Buffer		buf = ReadBuffer(index, blk);
	Page		page;
	PgturbohybridSparsePostingsTuple ct;
	uint32		n;
	uint32		base;
	uint32		wwidth = PgturbohybridSparseWeightWidth(bits);
	const char *payload;

	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	ct = (PgturbohybridSparsePostingsTuple) PageGetItem(page, PageGetItemId(page, off));
	n = ct->count;
	base = ct->baseNodeId;
	payload = ct->payload;

	if (ct->encoding == PGTURBOHYBRID_SPARSE_ENCODING_VARINT)
	{
		uint16		deltaBytes;
		Size		weightsStart;
		const uint8 *deltas = (const uint8 *) (payload + sizeof(uint16));
		int			pos = 0;
		uint32		node = base;

		memcpy(&deltaBytes, payload, sizeof(uint16));
		weightsStart = PgturbohybridSparseAlignUp(sizeof(uint16) + (Size) deltaBytes,
												  wwidth);
		for (uint32 k = 0; k < n; k++)
		{
			int			consumed;

			node += PgturbohybridSparseVarintDecode(deltas + pos, &consumed);
			pos += consumed;
			nodes[k] = node;
			contribs[k] = effMul *
				PgturbohybridWandWeight(payload + weightsStart + (Size) k * wwidth, bits);
		}
	}
	else						/* SoA */
	{
		Size		offsetsStart = PgturbohybridSparseAlignUp((Size) n * wwidth, 2);

		for (uint32 k = 0; k < n; k++)
		{
			uint16		o;

			memcpy(&o, payload + offsetsStart + (Size) k * sizeof(uint16), sizeof(uint16));
			nodes[k] = base + o;
			contribs[k] = effMul *
				PgturbohybridWandWeight(payload + (Size) k * wwidth, bits);
		}
	}
	*outCount = n;
	UnlockReleaseBuffer(buf);
}

/* Load a term's block-max directory (blockCount entries) by physical walk. */
static PgturbohybridSparseBlockMax *
PgturbohybridWandLoadBlockMax(Relation index, BlockNumber blk, OffsetNumber off,
							  uint32 count)
{
	PgturbohybridSparseBlockMax *out = (PgturbohybridSparseBlockMax *)
		palloc(sizeof(PgturbohybridSparseBlockMax) * Max(count, 1u));
	uint32		got = 0;

	while (got < count && blk != InvalidBlockNumber)
	{
		Buffer		buf = ReadBuffer(index, blk);
		Page		page;
		OffsetNumber maxoff;

		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		maxoff = PageGetMaxOffsetNumber(page);
		while (got < count && off <= maxoff)
		{
			PgturbohybridSparseBlockMaxTuple bt = (PgturbohybridSparseBlockMaxTuple)
				PageGetItem(page, PageGetItemId(page, off));

			if (bt->type == PGTURBOHYBRID_SPARSE_BLOCKMAX_TUPLE_TYPE)
			{
				uint32		take = (uint32) Min((uint64) (count - got), (uint64) bt->count);

				memcpy(out + got, bt->entries,
					   sizeof(PgturbohybridSparseBlockMax) * take);
				got += take;
			}
			off++;
		}
		blk = PgturbohybridGraphPageGetOpaque(page)->nextblkno;
		off = FirstOffsetNumber;
		UnlockReleaseBuffer(buf);
	}
	return out;
}

/*
 * Advance a term iterator to the first node_id >= target, skipping whole blocks
 * via the block-max directory (without reading their postings).  *visited and
 * *skipped accumulate chunk reads / block skips.
 */
static void
PgturbohybridWandAdvance(Relation index, PgturbohybridWandTerm *t, uint32 target,
						 int bits, uint64 *visited, uint64 *skipped)
{
	while (t->curBlock < t->nBlocks &&
		   t->blocks[t->curBlock].lastNodeId < target)
	{
		t->curBlock++;
		if (t->loaded)
			t->loaded = false;
		(*skipped)++;
	}
	if (t->curBlock >= t->nBlocks)
	{
		t->curNode = PGTURBOHYBRID_WAND_INF;
		return;
	}
	if (!t->loaded)
	{
		PgturbohybridWandDecodeChunk(index, t->blocks[t->curBlock].postingsBlkno,
									 t->blocks[t->curBlock].postingsOffno, bits,
									 t->effMul, t->nodes, t->contribs, &t->chunkCount);
		t->chunkPos = 0;
		t->loaded = true;
		(*visited)++;
	}
	while (t->chunkPos < t->chunkCount && t->nodes[t->chunkPos] < target)
		t->chunkPos++;
	if (t->chunkPos >= t->chunkCount)
	{
		/* Past this chunk: move to the next block and retry. */
		t->curBlock++;
		t->loaded = false;
		PgturbohybridWandAdvance(index, t, target, bits, visited, skipped);
		return;
	}
	t->curNode = t->nodes[t->chunkPos];
}

static void
PgturbohybridWandHeapSiftUp(PgturbohybridWandHeapEntry *heap, int i)
{
	while (i > 0)
	{
		int			parent = (i - 1) / 2;

		if (heap[parent].score <= heap[i].score)
			break;
		{
			PgturbohybridWandHeapEntry tmp = heap[parent];

			heap[parent] = heap[i];
			heap[i] = tmp;
		}
		i = parent;
	}
}

static void
PgturbohybridWandHeapSiftDown(PgturbohybridWandHeapEntry *heap, int size, int i)
{
	for (;;)
	{
		int			l = 2 * i + 1;
		int			r = 2 * i + 2;
		int			small = i;

		if (l < size && heap[l].score < heap[small].score)
			small = l;
		if (r < size && heap[r].score < heap[small].score)
			small = r;
		if (small == i)
			break;
		{
			PgturbohybridWandHeapEntry tmp = heap[small];

			heap[small] = heap[i];
			heap[i] = tmp;
		}
		i = small;
	}
}

/*
 * Block-max WAND top-k collection.  Returns the count; fills *out (palloc'd in
 * the current context) sorted by descending score.  Exact: never drops a true
 * top-k candidate (pruning only skips documents whose summed block upper bound
 * cannot exceed the current k-th score).
 */
static int
PgturbohybridSparseCollectWand(Relation index, PgturbohybridSparseMetaTuple meta,
							   PgturbohybridSparseLexEntry *lexicon, uint32 lexCount,
							   const PgturbohybridSparseVectorEntry *qEntries,
							   uint32 qCount, PgturbohybridNodeState *states,
							   uint32 nodeCount, int k,
							   PgturbohybridSparseCandidate **out,
							   PgturbohybridSparseScanStats *stats)
{
	PgturbohybridWandTerm *terms;
	PgturbohybridWandTerm **order;
	int			numTerms = 0;
	int			bits = (int) meta->quantBits;
	uint32		blockSize = meta->blockSize > 0 ? meta->blockSize :
		PGTURBOHYBRID_SPARSE_DEFAULT_BLOCK_SIZE;
	PgturbohybridWandHeapEntry *heap;
	int			heapSize = 0;
	double		theta = -get_float8_infinity();
	uint64		visited = 0,
				skipped = 0,
				pruned = 0,
				iterations = 0,
				thresholdUpdates = 0,
				heapUpdates = 0,
				docsScored = 0;
	uint32		resolved = 0;
	PgturbohybridSparseCandidate *result;
	int			resultCount;

	terms = (PgturbohybridWandTerm *)
		palloc0(sizeof(PgturbohybridWandTerm) * Max(qCount, 1u));
	for (uint32 i = 0; i < qCount; i++)
	{
		PgturbohybridSparseLexEntry key;
		PgturbohybridSparseLexEntry *found;
		PgturbohybridWandTerm *t;

		if (qEntries[i].weight == 0.0f)
			continue;
		key.termId = qEntries[i].termId;
		found = (PgturbohybridSparseLexEntry *)
			bsearch(&key, lexicon, lexCount, sizeof(PgturbohybridSparseLexEntry),
					PgturbohybridSparseLexCompare);
		if (found == NULL || found->blockCount == 0 ||
			found->blockMaxBlkno == InvalidBlockNumber)
			continue;
		resolved++;
		t = &terms[numTerms++];
		t->effMul = bits == 0 ? (double) qEntries[i].weight :
			(double) qEntries[i].weight * (double) found->scale;
		/*
		 * Term upper bound = effMul * max quantized code, which equals the exact
		 * maximum contribution any posting of this term can make (round() is
		 * monotonic, so the max-weight posting has the max code).  Using the raw
		 * maxWeight would be unsafe for quantized scores (rounding can exceed it).
		 */
		t->termMaxUB = t->effMul * (bits == 0 ? (double) found->maxWeight :
									(double) PgturbohybridSparseQuantize(found->maxWeight,
																		 found->scale, bits));
		t->bits = bits;
		t->nBlocks = found->blockCount;
		t->blocks = PgturbohybridWandLoadBlockMax(index, found->blockMaxBlkno,
												  found->blockMaxOffno,
												  found->blockCount);
		t->curBlock = 0;
		t->loaded = false;
		t->nodes = (uint32 *) palloc(sizeof(uint32) * blockSize);
		t->contribs = (double *) palloc(sizeof(double) * blockSize);
		PgturbohybridWandAdvance(index, t, 0, bits, &visited, &skipped);
	}

	heap = (PgturbohybridWandHeapEntry *)
		palloc(sizeof(PgturbohybridWandHeapEntry) * Max(k, 1));
	order = (PgturbohybridWandTerm **)
		palloc(sizeof(PgturbohybridWandTerm *) * Max(numTerms, 1));
	for (int i = 0; i < numTerms; i++)
		order[i] = &terms[i];

	while (numTerms > 0)
	{
		int			pivotIdx = -1;
		uint32		pivotNode;
		double		ub = 0.0;

		iterations++;
		CHECK_FOR_INTERRUPTS();

		/* Sort iterators by current node_id (exhausted = INF, sort last). */
		for (int a = 1; a < numTerms; a++)
		{
			PgturbohybridWandTerm *key = order[a];
			int			b = a - 1;

			while (b >= 0 && order[b]->curNode > key->curNode)
			{
				order[b + 1] = order[b];
				b--;
			}
			order[b + 1] = key;
		}

		for (int i = 0; i < numTerms; i++)
		{
			if (order[i]->curNode == PGTURBOHYBRID_WAND_INF)
				break;
			ub += order[i]->termMaxUB;
			if (ub > theta)
			{
				pivotIdx = i;
				break;
			}
		}
		if (pivotIdx < 0)
			break;				/* no remaining document can exceed theta */
		pivotNode = order[pivotIdx]->curNode;

		if (order[0]->curNode == pivotNode)
		{
			double		score = 0.0;

			/*
			 * Sum every term positioned at pivotNode -- ties at the pivot can
			 * sort beyond pivotIdx, so iterate all terms (not just [0,pivotIdx]).
			 */
			for (int i = 0; i < numTerms; i++)
				if (order[i]->curNode == pivotNode)
					score += order[i]->contribs[order[i]->chunkPos];
			for (int i = 0; i < numTerms; i++)
				if (order[i]->curNode == pivotNode)
					PgturbohybridWandAdvance(index, order[i], pivotNode + 1, bits,
											 &visited, &skipped);
			docsScored++;

			if (pivotNode < nodeCount && states[pivotNode].live)
			{
				if (heapSize < k)
				{
					heap[heapSize].score = score;
					heap[heapSize].nodeId = pivotNode;
					heap[heapSize].heaptid = states[pivotNode].tid;
					PgturbohybridWandHeapSiftUp(heap, heapSize);
					heapSize++;
					heapUpdates++;
					if (heapSize == k)
					{
						theta = heap[0].score;
						thresholdUpdates++;
					}
				}
				else if (score > heap[0].score)
				{
					heap[0].score = score;
					heap[0].nodeId = pivotNode;
					heap[0].heaptid = states[pivotNode].tid;
					PgturbohybridWandHeapSiftDown(heap, heapSize, 0);
					heapUpdates++;
					if (heap[0].score > theta)
					{
						theta = heap[0].score;
						thresholdUpdates++;
					}
				}
			}
		}
		else
		{
			/* Advance a lagging term before the pivot to >= pivotNode. */
			for (int i = 0; i < pivotIdx; i++)
				if (order[i]->curNode < pivotNode)
				{
					PgturbohybridWandAdvance(index, order[i], pivotNode, bits,
											 &visited, &skipped);
					break;
				}
			pruned++;
		}
	}

	/* Emit the heap as candidates sorted by descending score. */
	resultCount = heapSize;
	result = resultCount > 0 ? (PgturbohybridSparseCandidate *)
		palloc(sizeof(PgturbohybridSparseCandidate) * resultCount) : NULL;
	for (int i = 0; i < heapSize; i++)
	{
		result[i].nodeId = heap[i].nodeId;
		result[i].heaptid = heap[i].heaptid;
		result[i].score = heap[i].score;
	}
	if (resultCount > 1)
	{
		/* Reuse the scored comparator (descending score, node_id tiebreak). */
		PgturbohybridSparseScored *tmp = (PgturbohybridSparseScored *)
			palloc(sizeof(PgturbohybridSparseScored) * resultCount);

		for (int i = 0; i < resultCount; i++)
		{
			tmp[i].nodeId = result[i].nodeId;
			tmp[i].score = result[i].score;
		}
		qsort(tmp, resultCount, sizeof(PgturbohybridSparseScored),
			  PgturbohybridSparseScoredCompare);
		for (int i = 0; i < resultCount; i++)
		{
			result[i].nodeId = tmp[i].nodeId;
			result[i].score = tmp[i].score;
			result[i].heaptid = states[tmp[i].nodeId].tid;
		}
		pfree(tmp);
	}

	if (stats != NULL)
	{
		stats->branchUsed = true;
		stats->usedWand = true;
		stats->terms = qCount;
		stats->resolvedTerms = resolved;
		stats->candidatesScored = docsScored;	/* documents fully evaluated */
		stats->blocksVisited = visited;
		stats->blocksSkipped = skipped;
		stats->wandPruned = pruned;
		stats->wandIterations = iterations;
		stats->wandThresholdUpdates = thresholdUpdates;
		stats->wandHeapUpdates = heapUpdates;
		stats->quantBits = bits;
		stats->quantMode = (int) meta->quantMode;
		stats->encoding = (int) meta->postingsEncoding;
		stats->scoreKernel = PGTURBOHYBRID_SPARSE_SCORE_SCALAR;
	}

	if (out != NULL)
		*out = result;
	return resultCount;
}

int
PgturbohybridSparseCollectCandidates(Relation index, PgturbohybridQueryHeader *query,
									 int k, bool simdEnabled, bool wandEnabled,
									 PgturbohybridSparseCandidate **out,
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
	uint64		simdBlocks = 0;
	uint64		scalarTail = 0;
	int			scoreKernel;
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
	scoreKernel = PgturbohybridSparseResolveScoreKernel((int) meta.quantBits,
														simdEnabled);

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

	/*
	 * Block-max WAND path (prompt 9): exact top-k with safe pruning, when the
	 * index has a block-max directory and WAND is enabled.  Otherwise fall back
	 * to the exact OR-accumulation (also the WAND correctness reference).
	 */
	if (wandEnabled && meta.hasBlockMax)
	{
		resultCount = PgturbohybridSparseCollectWand(index, &meta, lexicon, lexCount,
													 qEntries, qCount, states,
													 nodeCount, k, &result, stats);
		MemoryContextSwitchTo(oldCtx);
		INSTR_TIME_SET_CURRENT(t1);
		INSTR_TIME_SUBTRACT(t1, t0);
		if (stats != NULL)
			stats->elapsedUs = (uint64) INSTR_TIME_GET_MICROSEC(t1);
		if (out != NULL)
			*out = result;
		return resultCount;
	}

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
															 (double) found->scale,
															 (int) meta.quantBits,
															 scoreKernel,
															 scores, nodeCount,
															 &simdBlocks, &scalarTail);
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
		stats->quantBits = (int) meta.quantBits;
		stats->quantMode = (int) meta.quantMode;
		stats->encoding = (int) meta.postingsEncoding;
		stats->scoreKernel = scoreKernel;
		stats->simdBlocks = simdBlocks;
		stats->scalarTailPostings = scalarTail;
	}

	if (out != NULL)
		*out = result;
	return resultCount;
}
