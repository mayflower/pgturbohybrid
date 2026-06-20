/*
 * pgturbohybrid_sparse_build.c
 *
 * Build path for the native sparse-vector inverted index (prompt 4).  After the
 * dense graph build has assigned node_ids, scan the heap, collect (term_id,
 * node_id, weight) postings from the sparse index key, sort by (term_id,
 * node_id), and write the lexicon + postings + meta tuples (WAL'd via
 * PgturbohybridGraphAppendTuple), anchored by graphMeta.tqSparseMetaStartBlkno.
 * Exact float32 only.
 */
#include "postgres.h"

#include <string.h>

#include "access/generic_xlog.h"
#include "access/genam.h"
#include "access/tableam.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "storage/lmgr.h"
#include "utils/memutils.h"
#include "utils/rel.h"

#include "pgturbohybrid.h"
#include "pgturbohybrid_quant.h"
#include "pgturbohybrid_am.h"
#include "pgturbohybrid_query.h"
#include "pgturbohybrid_sparse.h"

typedef struct PgturbohybridSparseTriple
{
	int32		termId;
	uint32		nodeId;
	float4		weight;
} PgturbohybridSparseTriple;

typedef struct PgturbohybridSparseCollector
{
	Relation	index;
	int			sparseKey;		/* index-key position of the sparse column */
	PgturbohybridTidNode *tidNodes;
	uint32		tidNodeCount;
	PgturbohybridSparseTriple *triples;
	uint64		count;
	uint64		capacity;
	uint64		docCount;		/* docs contributing >= 1 posting */
} PgturbohybridSparseCollector;

static int
PgturbohybridSparseTidCompare(const void *a, const void *b)
{
	const PgturbohybridTidNode *na = (const PgturbohybridTidNode *) a;
	const PgturbohybridTidNode *nb = (const PgturbohybridTidNode *) b;

	return ItemPointerCompare((ItemPointer) &na->tid, (ItemPointer) &nb->tid);
}

static int
PgturbohybridSparseTripleCompare(const void *a, const void *b)
{
	const PgturbohybridSparseTriple *ta = (const PgturbohybridSparseTriple *) a;
	const PgturbohybridSparseTriple *tb = (const PgturbohybridSparseTriple *) b;

	if (ta->termId != tb->termId)
		return ta->termId < tb->termId ? -1 : 1;
	if (ta->nodeId != tb->nodeId)
		return ta->nodeId < tb->nodeId ? -1 : 1;
	return 0;
}

static bool
PgturbohybridSparseLookupNodeId(PgturbohybridSparseCollector *collector,
								ItemPointer tid, uint32 *nodeId)
{
	PgturbohybridTidNode key;
	PgturbohybridTidNode *found;

	if (collector->tidNodeCount == 0)
		return false;
	key.tid = *tid;
	key.nodeId = 0;
	found = bsearch(&key, collector->tidNodes, collector->tidNodeCount,
					sizeof(PgturbohybridTidNode), PgturbohybridSparseTidCompare);
	if (found == NULL)
		return false;
	*nodeId = found->nodeId;
	return true;
}

static void
PgturbohybridSparseBuildCallback(Relation index, ItemPointer tid, Datum *values,
								 bool *isnull, bool tupleIsAlive, void *opaque)
{
	PgturbohybridSparseCollector *collector = (PgturbohybridSparseCollector *) opaque;
	struct varlena *detoasted;
	const PgturbohybridSparseVectorEntry *entries;
	uint32		entryCount;
	uint32		nodeId;
	bool		contributed = false;

	(void) tupleIsAlive;

	if (isnull[collector->sparseKey])
		return;
	if (!PgturbohybridSparseLookupNodeId(collector, tid, &nodeId))
		return;

	detoasted = (struct varlena *) PG_DETOAST_DATUM(values[collector->sparseKey]);
	entries = PgturbohybridSparseVectorData(detoasted, &entryCount);

	for (uint32 i = 0; i < entryCount; i++)
	{
		if (entries[i].weight == 0.0f)
			continue;			/* a zero weight contributes nothing to the IP */

		if (collector->count == collector->capacity)
		{
			collector->capacity = collector->capacity == 0 ? 1024
				: collector->capacity * 2;
			/* repalloc() requires a valid pointer; palloc the first block. */
			if (collector->triples == NULL)
				collector->triples = (PgturbohybridSparseTriple *)
					palloc(sizeof(PgturbohybridSparseTriple) * collector->capacity);
			else
				collector->triples = (PgturbohybridSparseTriple *)
					repalloc(collector->triples,
							 sizeof(PgturbohybridSparseTriple) * collector->capacity);
		}
		collector->triples[collector->count].termId = entries[i].termId;
		collector->triples[collector->count].nodeId = nodeId;
		collector->triples[collector->count].weight = entries[i].weight;
		collector->count++;
		contributed = true;
	}
	if (contributed)
		collector->docCount++;

	if ((void *) detoasted != (void *) DatumGetPointer(values[collector->sparseKey]))
		pfree(detoasted);
}

/* Anchor the sparse meta block in the graph metapage (mirrors BM25). */
static void
PgturbohybridSparseSetMetaBlock(Relation index, BlockNumber metaBlkno)
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
	if (metap->magicNumber != PGTURBOHYBRID_GRAPH_MAGIC_NUMBER ||
		metap->storageKind != PGTURBOHYBRID_GRAPH_STORAGE_QUANT_GRAPH_NATIVE)
	{
		if (xlogState != NULL)
			GenericXLogAbort(xlogState);
		UnlockReleaseBuffer(buf);
		ereport(ERROR,
				(errcode(ERRCODE_DATA_CORRUPTED),
				 errmsg("pgturbohybrid native graph metadata is missing")));
	}

	metap->tqSparseMetaStartBlkno = metaBlkno;
	PgturbohybridGraphMarkPageGraphOp(page, PGTURBOHYBRID_GRAPH_GRAPH_OP_META_UPDATE);

	if (xlogState != NULL)
		GenericXLogFinish(xlogState);
	else
		MarkBufferDirty(buf);
	UnlockReleaseBuffer(buf);
	PgturbohybridGraphLogGraphWalRecord(index, MAIN_FORKNUM,
										PGTURBOHYBRID_GRAPH_METAPAGE_BLKNO,
										PGTURBOHYBRID_GRAPH_GRAPH_OP_META_UPDATE);
}

/*
 * Write the sorted triples as per-term postings chunks + a lexicon, then the
 * meta tuple, and anchor it.  Returns nothing; sets the graph metapage anchor.
 */
static void
PgturbohybridSparseWriteStorage(PgturbohybridSparseCollector *collector)
{
	Relation	index = collector->index;
	BlockNumber postingsStart = InvalidBlockNumber;
	BlockNumber lexiconStart = InvalidBlockNumber;
	BlockNumber metaStart = InvalidBlockNumber;
	BlockNumber insertBlkno;
	uint32		postingsPages = 0;
	uint32		lexiconPages = 0;
	uint32		termCount = 0;
	PgturbohybridSparseLexiconEntry *lexEntries;
	uint64		lexCapacity;
	uint64		lexCount = 0;
	Size		chunkMax;
	Size		postingsTupleMax;
	PgturbohybridSparsePostingsTupleData *chunk;
	PgturbohybridSparseMetaTupleData meta;
	uint64		i = 0;

	chunkMax = PgturbohybridSparsePostingsTupleSize(PGTURBOHYBRID_SPARSE_POSTINGS_PER_CHUNK);
	postingsTupleMax = chunkMax;
	chunk = (PgturbohybridSparsePostingsTupleData *) palloc0(postingsTupleMax);

	lexCapacity = 1024;
	lexEntries = (PgturbohybridSparseLexiconEntry *)
		palloc(sizeof(PgturbohybridSparseLexiconEntry) * lexCapacity);

	/* One pass over the (term,node)-sorted triples: emit per-term chunks. */
	while (i < collector->count)
	{
		int32		termId = collector->triples[i].termId;
		uint64		termStart = i;
		uint32		df;
		BlockNumber firstChunkBlkno = InvalidBlockNumber;
		OffsetNumber firstChunkOffno = InvalidOffsetNumber;
		float4		maxWeight = 0.0f;
		bool		firstChunkRecorded = false;

		while (i < collector->count && collector->triples[i].termId == termId)
		{
			if (collector->triples[i].weight > maxWeight)
				maxWeight = collector->triples[i].weight;
			i++;
		}
		df = (uint32) (i - termStart);

		/* Emit ceil(df / PER_CHUNK) chunks, each (except the last) full. */
		for (uint64 p = termStart; p < i; p += PGTURBOHYBRID_SPARSE_POSTINGS_PER_CHUNK)
		{
			uint32		n = (uint32) Min((uint64) PGTURBOHYBRID_SPARSE_POSTINGS_PER_CHUNK,
										 i - p);
			OffsetNumber off;

			chunk->type = PGTURBOHYBRID_SPARSE_POSTINGS_TUPLE_TYPE;
			chunk->reserved1 = 0;
			chunk->count = (uint16) n;
			chunk->termId = termId;
			for (uint32 k = 0; k < n; k++)
			{
				chunk->postings[k].nodeId = collector->triples[p + k].nodeId;
				chunk->postings[k].weight = collector->triples[p + k].weight;
			}
			off = PgturbohybridGraphAppendTuple(index, MAIN_FORKNUM, &postingsStart,
												PGTURBOHYBRID_GRAPH_PAGE_KIND_SPARSE_POSTINGS,
												(Item) chunk,
												PgturbohybridSparsePostingsTupleSize(n),
												PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT,
												&insertBlkno);
			if (!firstChunkRecorded)
			{
				firstChunkBlkno = insertBlkno;
				firstChunkOffno = off;
				firstChunkRecorded = true;
			}
		}

		if (lexCount == lexCapacity)
		{
			lexCapacity *= 2;
			lexEntries = (PgturbohybridSparseLexiconEntry *)
				repalloc(lexEntries,
						 sizeof(PgturbohybridSparseLexiconEntry) * lexCapacity);
		}
		lexEntries[lexCount].termId = termId;
		lexEntries[lexCount].df = df;
		lexEntries[lexCount].maxWeight = maxWeight;
		lexEntries[lexCount].postingsBlkno = firstChunkBlkno;
		lexEntries[lexCount].postingsOffno = firstChunkOffno;
		lexEntries[lexCount].reserved = 0;
		lexCount++;
		termCount++;
	}
	postingsPages = postingsStart == InvalidBlockNumber ? 0 : 1;	/* count approx */

	/* Write the lexicon: pack entries into tuples of up to LEXICON_PER_TUPLE. */
	{
		Size		lexTupleMax =
			PgturbohybridSparseLexiconTupleSize(PGTURBOHYBRID_SPARSE_LEXICON_PER_TUPLE);
		PgturbohybridSparseLexiconTupleData *lexTuple =
			(PgturbohybridSparseLexiconTupleData *) palloc0(lexTupleMax);
		uint64		j = 0;

		while (j < lexCount)
		{
			uint32		n = (uint32) Min((uint64) PGTURBOHYBRID_SPARSE_LEXICON_PER_TUPLE,
										 lexCount - j);

			lexTuple->type = PGTURBOHYBRID_SPARSE_LEXICON_TUPLE_TYPE;
			lexTuple->reserved1 = 0;
			lexTuple->count = (uint16) n;
			memcpy(lexTuple->entries, &lexEntries[j],
				   sizeof(PgturbohybridSparseLexiconEntry) * n);
			(void) PgturbohybridGraphAppendTuple(index, MAIN_FORKNUM, &lexiconStart,
												 PGTURBOHYBRID_GRAPH_PAGE_KIND_SPARSE_LEXICON,
												 (Item) lexTuple,
												 PgturbohybridSparseLexiconTupleSize(n),
												 PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT,
												 &insertBlkno);
			j += n;
			lexiconPages = 1;	/* approx (page count not load-bearing here) */
		}
		pfree(lexTuple);
	}

	/* Write the meta tuple + anchor it. */
	memset(&meta, 0, sizeof(meta));
	meta.type = PGTURBOHYBRID_SPARSE_META_TUPLE_TYPE;
	meta.sparseVersion = PGTURBOHYBRID_SPARSE_VERSION;
	meta.termCount = termCount;
	meta.docCount = (uint32) collector->docCount;
	meta.postingCount = collector->count;
	meta.lexiconStartBlkno = lexiconStart;
	meta.lexiconPages = lexiconPages;
	meta.postingsPages = postingsPages;
	(void) PgturbohybridGraphAppendTuple(index, MAIN_FORKNUM, &metaStart,
										 PGTURBOHYBRID_GRAPH_PAGE_KIND_SPARSE_META,
										 (Item) &meta,
										 MAXALIGN(sizeof(meta)),
										 PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT,
										 &insertBlkno);
	PgturbohybridSparseSetMetaBlock(index, metaStart);

	pfree(chunk);
	pfree(lexEntries);
}

bool
PgturbohybridSparseIndexAvailable(Relation index)
{
	PgturbohybridGraphMetaPageData meta;

	if (index == NULL)
		return false;
	if (!PgturbohybridGraphReadMeta(index, &meta))
		return false;
	return meta.tqSparseMetaStartBlkno != InvalidBlockNumber &&
		meta.tqSparseMetaStartBlkno != 0;
}

void
PgturbohybridSparseBuildCollect(Relation heap, Relation index, IndexInfo *indexInfo)
{
	MemoryContext ctx;
	MemoryContext oldCtx;
	PgturbohybridSparseCollector collector;
	PgturbohybridIndexKeyMap map;
	PgturbohybridGraphMetaPageData graphMeta;

	if (index == NULL || heap == NULL)
		return;

	PgturbohybridBuildIndexKeyMap(index, indexInfo, &map);
	if (!map.hasSparse)
		return;
	/* Dense-present sparse only (prompt 4): the dense graph owns node identity. */
	if (map.graphKey < 0)
		return;

	ctx = AllocSetContextCreate(CurrentMemoryContext,
								"pgturbohybrid sparse build collector",
								ALLOCSET_DEFAULT_SIZES);
	oldCtx = MemoryContextSwitchTo(ctx);

	memset(&collector, 0, sizeof(collector));
	collector.index = index;
	collector.sparseKey = map.sparseKey;

	if (!PgturbohybridGraphReadMeta(index, &graphMeta))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("pgturbohybrid sparse collection requires native pgturbohybrid graph storage")));

	if (graphMeta.tqNodeCount > 0)
	{
		collector.tidNodes = PgturbohybridReadNodeMap(index, &collector.tidNodeCount);
		if (collector.tidNodeCount == 0)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_CORRUPTED),
					 errmsg("pgturbohybrid native graph storage is missing during sparse collection")));
	}

	(void) table_index_build_scan(heap, index, indexInfo, true, true,
								  PgturbohybridSparseBuildCallback, &collector, NULL);

	if (collector.count > 1)
		qsort(collector.triples, collector.count,
			  sizeof(PgturbohybridSparseTriple), PgturbohybridSparseTripleCompare);

	PgturbohybridSparseWriteStorage(&collector);

	MemoryContextSwitchTo(oldCtx);
	MemoryContextDelete(ctx);
}
