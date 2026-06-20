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
	int			quantBits;		/* 0 (f32), 8, or 16 */
	int			quantMode;		/* PGTURBOHYBRID_SPARSE_QUANT_* */
	int			encoding;		/* resolved PGTURBOHYBRID_SPARSE_ENCODING_* */
	int			blockSize;		/* postings per chunk */
	bool		blockMax;		/* write a block-max directory (WAND) */
} PgturbohybridSparseCollector;

/* Resolve effective sparse build options (index->rd_options or defaults). */
static void
PgturbohybridSparseResolveOptions(Relation index, int *bits, int *mode,
								  int *encoding, int *blockSize, bool *blockMax)
{
	PgturbohybridOptions *opts = (PgturbohybridOptions *) index->rd_options;
	int			optBits = opts != NULL ? opts->sparseQuantBits :
		PGTURBOHYBRID_SPARSE_DEFAULT_QUANT_BITS;
	int			optMode = opts != NULL ? opts->sparseQuantMode :
		PGTURBOHYBRID_SPARSE_QUANT_PER_TERM_LINEAR;
	int			optEnc = opts != NULL ? opts->sparsePostingsEncoding : 0;
	int			optBlock = opts != NULL ? opts->sparseBlockSize :
		PGTURBOHYBRID_SPARSE_DEFAULT_BLOCK_SIZE;

	*blockMax = opts != NULL ? opts->sparseBlockMax : true;

	/* "f32" mode (or 0 bits) means exact f32 storage regardless of bit width. */
	if (optMode == PGTURBOHYBRID_SPARSE_QUANT_F32)
		optBits = 0;
	*mode = optBits == 0 ? PGTURBOHYBRID_SPARSE_QUANT_F32 :
		PGTURBOHYBRID_SPARSE_QUANT_PER_TERM_LINEAR;
	*bits = optBits;

	/* Reloption encoding: auto(0)->SoA, offset16_soa(1)->SoA, varint(2)->varint. */
	*encoding = (optEnc == 2) ? PGTURBOHYBRID_SPARSE_ENCODING_VARINT :
		PGTURBOHYBRID_SPARSE_ENCODING_SOA;

	if (optBlock < 1)
		optBlock = PGTURBOHYBRID_SPARSE_DEFAULT_BLOCK_SIZE;
	if (optBlock > PGTURBOHYBRID_SPARSE_MAX_BLOCK_SIZE)
		optBlock = PGTURBOHYBRID_SPARSE_MAX_BLOCK_SIZE;
	*blockSize = optBlock;
}

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

/* Worst-case encoded bytes per posting for a (bits, encoding) combination. */
static uint32
PgturbohybridSparsePerPostingBytes(int encoding, int bits)
{
	uint32		w = PgturbohybridSparseWeightWidth(bits);

	if (encoding == PGTURBOHYBRID_SPARSE_ENCODING_VARINT)
		return w + 5;			/* up to 5 varint bytes + weight */
	return w + 2;				/* uint16 offset + weight (SoA) */
}

/* Max postings per chunk so the encoded tuple fits on a fresh page. */
static uint32
PgturbohybridSparseMaxChunkPostings(int encoding, int bits)
{
	Size		budget = BLCKSZ - 256;	/* conservative: page header + special + slack */
	uint32		per = PgturbohybridSparsePerPostingBytes(encoding, bits);
	Size		usable = budget - PGTURBOHYBRID_SPARSE_POSTINGS_HEADER - 8;

	return (uint32) Max(usable / per, (Size) 1);
}

/* Write one quantized/f32 weight at dst for the given bit width. */
static inline void
PgturbohybridSparseWriteWeight(char *dst, float4 weight, float4 scale, int bits)
{
	if (bits == 0)
		memcpy(dst, &weight, sizeof(float4));
	else if (bits == 16)
	{
		uint16		q = (uint16) PgturbohybridSparseQuantize(weight, scale, 16);

		memcpy(dst, &q, sizeof(uint16));
	}
	else
	{
		uint8		q = (uint8) PgturbohybridSparseQuantize(weight, scale, 8);

		*(uint8 *) dst = q;
	}
}

/*
 * Encode triples[start, start+n) into the chunk buffer for the given encoding
 * and bit width (node_ids relative to triples[start].nodeId).  Returns the
 * MAXALIGN'd tuple size.
 */
static Size
PgturbohybridSparseEncodeChunk(PgturbohybridSparsePostingsTupleData *chunk,
							   int encoding, int bits, float4 scale,
							   const PgturbohybridSparseTriple *triples,
							   uint64 start, uint32 n)
{
	uint32		base = triples[start].nodeId;
	uint32		wwidth = PgturbohybridSparseWeightWidth(bits);
	char	   *payload = chunk->payload;
	Size		payloadBytes;

	chunk->type = PGTURBOHYBRID_SPARSE_POSTINGS_TUPLE_TYPE;
	chunk->encoding = (uint8) encoding;
	chunk->count = (uint16) n;
	chunk->termId = triples[start].termId;
	chunk->baseNodeId = base;

	if (encoding == PGTURBOHYBRID_SPARSE_ENCODING_VARINT)
	{
		uint8	   *deltas = (uint8 *) (payload + sizeof(uint16));
		int			pos = 0;
		uint32		prev = base;
		Size		weightsStart;

		for (uint32 k = 0; k < n; k++)
		{
			uint32		node = triples[start + k].nodeId;

			pos += PgturbohybridSparseVarintEncode(node - prev, deltas + pos);
			prev = node;
		}
		{
			uint16		deltaBytes = (uint16) pos;

			memcpy(payload, &deltaBytes, sizeof(uint16));
		}
		weightsStart = PgturbohybridSparseAlignUp(sizeof(uint16) + (Size) pos, wwidth);
		for (uint32 k = 0; k < n; k++)
			PgturbohybridSparseWriteWeight(payload + weightsStart + (Size) k * wwidth,
										   triples[start + k].weight, scale, bits);
		payloadBytes = weightsStart + (Size) n * wwidth;
	}
	else						/* SoA: weights[] then uint16 offsets[] */
	{
		Size		offsetsStart = PgturbohybridSparseAlignUp((Size) n * wwidth, 2);

		for (uint32 k = 0; k < n; k++)
			PgturbohybridSparseWriteWeight(payload + (Size) k * wwidth,
										   triples[start + k].weight, scale, bits);
		for (uint32 k = 0; k < n; k++)
		{
			uint16		off = (uint16) (triples[start + k].nodeId - base);

			memcpy(payload + offsetsStart + (Size) k * sizeof(uint16), &off,
				   sizeof(uint16));
		}
		payloadBytes = offsetsStart + (Size) n * sizeof(uint16);
	}

	return MAXALIGN(PGTURBOHYBRID_SPARSE_POSTINGS_HEADER + payloadBytes);
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
	BlockNumber blockMaxStart = InvalidBlockNumber;
	BlockNumber metaStart = InvalidBlockNumber;
	BlockNumber insertBlkno;
	uint32		postingsPages = 0;
	uint32		lexiconPages = 0;
	uint32		termCount = 0;
	PgturbohybridSparseLexiconEntry *lexEntries;
	uint64		lexCapacity;
	uint64		lexCount = 0;
	PgturbohybridSparsePostingsTupleData *chunk;
	PgturbohybridSparseBlockMax *bmEntries;
	uint64		bmCapacity;
	PgturbohybridSparseBlockMaxTupleData *bmTuple = NULL;
	PgturbohybridSparseMetaTupleData meta;
	uint64		i = 0;
	int			bits = collector->quantBits;
	int			encoding = collector->encoding;
	uint32		maxq = PgturbohybridSparseQuantMax(bits);
	uint32		chunkCap;

	/* Cap postings/chunk by both the configured block size and page capacity. */
	chunkCap = (uint32) Min((uint64) collector->blockSize,
							(uint64) PgturbohybridSparseMaxChunkPostings(encoding, bits));
	chunk = (PgturbohybridSparsePostingsTupleData *) palloc0(BLCKSZ);

	bmCapacity = 1024;
	bmEntries = (PgturbohybridSparseBlockMax *)
		palloc(sizeof(PgturbohybridSparseBlockMax) * bmCapacity);
	if (collector->blockMax)
		bmTuple = (PgturbohybridSparseBlockMaxTupleData *) palloc0(BLCKSZ);

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
		float4		scale;
		bool		firstChunkRecorded = false;
		uint64		p;
		uint64		nBlocks;

		while (i < collector->count && collector->triples[i].termId == termId)
		{
			if (collector->triples[i].weight > maxWeight)
				maxWeight = collector->triples[i].weight;
			i++;
		}
		df = (uint32) (i - termStart);
		scale = bits == 0 ? 0.0f : (maxWeight / (float4) maxq);

		/*
		 * Emit chunks: at most chunkCap postings, and for the SoA encoding at
		 * most a 2^16 node_id span (so offsets fit in uint16).  Collect one
		 * block-max directory entry per chunk for WAND pruning.
		 */
		p = termStart;
		nBlocks = 0;
		while (p < i)
		{
			uint32		base = collector->triples[p].nodeId;
			uint32		n = 0;
			float4		chunkMax = 0.0f;
			Size		size;
			OffsetNumber off;

			while (p + n < i && n < chunkCap)
			{
				if (encoding == PGTURBOHYBRID_SPARSE_ENCODING_SOA &&
					collector->triples[p + n].nodeId - base >= PGTURBOHYBRID_SPARSE_SOA_RANGE)
					break;
				if (collector->triples[p + n].weight > chunkMax)
					chunkMax = collector->triples[p + n].weight;
				n++;
			}

			size = PgturbohybridSparseEncodeChunk(chunk, encoding, bits, scale,
												  collector->triples, p, n);
			off = PgturbohybridGraphAppendTuple(index, MAIN_FORKNUM, &postingsStart,
												PGTURBOHYBRID_GRAPH_PAGE_KIND_SPARSE_POSTINGS,
												(Item) chunk, size,
												PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT,
												&insertBlkno);
			if (!firstChunkRecorded)
			{
				firstChunkBlkno = insertBlkno;
				firstChunkOffno = off;
				firstChunkRecorded = true;
			}
			if (collector->blockMax)
			{
				if (nBlocks == bmCapacity)
				{
					bmCapacity *= 2;
					bmEntries = (PgturbohybridSparseBlockMax *)
						repalloc(bmEntries,
								 sizeof(PgturbohybridSparseBlockMax) * bmCapacity);
				}
				bmEntries[nBlocks].firstNodeId = base;
				bmEntries[nBlocks].lastNodeId = collector->triples[p + n - 1].nodeId;
				bmEntries[nBlocks].maxWeight = chunkMax;
				bmEntries[nBlocks].postingsBlkno = insertBlkno;
				bmEntries[nBlocks].postingsOffno = off;
				bmEntries[nBlocks].reserved = 0;
				nBlocks++;
			}
			p += n;
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
		lexEntries[lexCount].scale = scale;
		lexEntries[lexCount].postingsBlkno = firstChunkBlkno;
		lexEntries[lexCount].postingsOffno = firstChunkOffno;
		lexEntries[lexCount].reserved = 0;
		lexEntries[lexCount].blockMaxBlkno = InvalidBlockNumber;
		lexEntries[lexCount].blockMaxOffno = InvalidOffsetNumber;
		lexEntries[lexCount].reserved2 = 0;
		lexEntries[lexCount].blockCount = 0;

		/* Write the term's block-max directory entries (split across tuples). */
		if (collector->blockMax && nBlocks > 0)
		{
			uint64		b = 0;
			bool		firstBmRecorded = false;

			while (b < nBlocks)
			{
				uint32		bn = (uint32) Min((uint64) PGTURBOHYBRID_SPARSE_BLOCKMAX_PER_TUPLE,
											  nBlocks - b);
				OffsetNumber bmoff;

				bmTuple->type = PGTURBOHYBRID_SPARSE_BLOCKMAX_TUPLE_TYPE;
				bmTuple->reserved1 = 0;
				bmTuple->count = (uint16) bn;
				bmTuple->termId = termId;
				memcpy(bmTuple->entries, &bmEntries[b],
					   sizeof(PgturbohybridSparseBlockMax) * bn);
				bmoff = PgturbohybridGraphAppendTuple(index, MAIN_FORKNUM,
													  &blockMaxStart,
													  PGTURBOHYBRID_GRAPH_PAGE_KIND_SPARSE_BLOCKMAX,
													  (Item) bmTuple,
													  PgturbohybridSparseBlockMaxTupleSize(bn),
													  PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT,
													  &insertBlkno);
				if (!firstBmRecorded)
				{
					lexEntries[lexCount].blockMaxBlkno = insertBlkno;
					lexEntries[lexCount].blockMaxOffno = bmoff;
					firstBmRecorded = true;
				}
				b += bn;
			}
			lexEntries[lexCount].blockCount = nBlocks;
		}

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
	meta.quantBits = (uint8) collector->quantBits;
	meta.quantMode = (uint8) collector->quantMode;
	meta.postingsEncoding = (uint8) collector->encoding;
	meta.sparseVersion = PGTURBOHYBRID_SPARSE_VERSION;
	meta.termCount = termCount;
	meta.docCount = (uint32) collector->docCount;
	meta.postingCount = collector->count;
	meta.lexiconStartBlkno = lexiconStart;
	meta.lexiconPages = lexiconPages;
	meta.postingsPages = postingsPages;
	meta.blockSize = (uint32) collector->blockSize;
	meta.blockMaxStartBlkno = blockMaxStart;
	meta.hasBlockMax = (collector->blockMax && blockMaxStart != InvalidBlockNumber) ? 1 : 0;
	(void) PgturbohybridGraphAppendTuple(index, MAIN_FORKNUM, &metaStart,
										 PGTURBOHYBRID_GRAPH_PAGE_KIND_SPARSE_META,
										 (Item) &meta,
										 MAXALIGN(sizeof(meta)),
										 PGTURBOHYBRID_GRAPH_GRAPH_OP_ELEMENT_INSERT,
										 &insertBlkno);
	PgturbohybridSparseSetMetaBlock(index, metaStart);

	pfree(chunk);
	pfree(lexEntries);
	pfree(bmEntries);
	if (bmTuple != NULL)
		pfree(bmTuple);
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
	PgturbohybridSparseResolveOptions(index, &collector.quantBits,
									  &collector.quantMode, &collector.encoding,
									  &collector.blockSize, &collector.blockMax);

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
