/*
 * pgturbohybrid_sparse.h
 *
 * Native sparse-vector (SPLADE) retrieval branch: on-disk inverted index over
 * a turbohybrid_sparse_vector index key, mirroring the BM25 branch but with
 * exact float32 weights and OR-accumulation scoring.  The sparse postings key
 * on the dense graph's node_id (see PgturbohybridReadNodeMap), so a sparse key
 * currently requires a dense/multivector graph key in the same index.
 *
 * Storage (chained index pages, WAL'd via PgturbohybridGraphAppendTuple):
 *   - one meta tuple, anchored by graphMeta.tqSparseMetaStartBlkno;
 *   - a lexicon: tuples packing PgturbohybridSparseLexiconEntry records
 *     (term_id -> df, max weight, first postings chunk);
 *   - postings: per-term chunks of {node_id, float4 weight}, sorted by
 *     (term_id, node_id), chained via nextBlkno/nextOffno.
 *
 * Exact f32 only here (prompt 4): no quantization, SIMD, WAND, or cache.
 */
#ifndef PGTURBOHYBRID_SPARSE_H
#define PGTURBOHYBRID_SPARSE_H

#include "postgres.h"

#include "access/itup.h"
#include "nodes/execnodes.h"
#include "storage/block.h"
#include "storage/itemptr.h"
#include "utils/rel.h"

#include "pgturbohybrid_query.h"

/*
 * Format v2 (prompt 6): postings carry per-term linearly-quantized weights
 * (q8/q16) or exact f32, in one of two physical encodings.  v1 was the
 * exact-f32 AoS layout from prompt 4; the on-disk format is unreleased, so the
 * build always writes v2 and the scan reads the bits/encoding from the meta
 * tuple and the per-term scale from the lexicon.
 */
#define PGTURBOHYBRID_SPARSE_VERSION 2

#define PGTURBOHYBRID_SPARSE_META_TUPLE_TYPE 0x71
#define PGTURBOHYBRID_SPARSE_LEXICON_TUPLE_TYPE 0x72
#define PGTURBOHYBRID_SPARSE_POSTINGS_TUPLE_TYPE 0x73

/* Quantization mode (per_term_linear = scalar scale per term; f32 = no quant). */
#define PGTURBOHYBRID_SPARSE_QUANT_F32 0
#define PGTURBOHYBRID_SPARSE_QUANT_PER_TERM_LINEAR 1

/* Physical postings encoding. */
#define PGTURBOHYBRID_SPARSE_ENCODING_SOA 0		/* base + uint16 offsets, SoA */
#define PGTURBOHYBRID_SPARSE_ENCODING_VARINT 1	/* LEB128 node deltas + weights */

#define PGTURBOHYBRID_SPARSE_DEFAULT_QUANT_BITS 8
#define PGTURBOHYBRID_SPARSE_DEFAULT_BLOCK_SIZE 512
#define PGTURBOHYBRID_SPARSE_MAX_BLOCK_SIZE 4096
/* SoA offsets are uint16, so a SoA chunk spans at most this many node_ids. */
#define PGTURBOHYBRID_SPARSE_SOA_RANGE 65536

/* Round x up to a multiple of a (a in {1,2,4}). */
static inline Size
PgturbohybridSparseAlignUp(Size x, uint32 a)
{
	return ((x + a - 1) / a) * a;
}

/* Effective weight byte width for a quantization bit width (0 = f32). */
static inline uint32
PgturbohybridSparseWeightWidth(int bits)
{
	if (bits == 16)
		return 2;
	if (bits == 8)
		return 1;
	return (uint32) sizeof(float4);
}

static inline uint32
PgturbohybridSparseQuantMax(int bits)
{
	return bits == 16 ? 65535u : 255u;
}

/* Round a weight to its quantized code in [0, 2^bits-1] (scale<=0 -> 0). */
static inline uint32
PgturbohybridSparseQuantize(float4 weight, float4 scale, int bits)
{
	double		q;
	uint32		maxq = PgturbohybridSparseQuantMax(bits);

	if (scale <= 0.0f || weight <= 0.0f)
		return 0;
	q = (double) weight / (double) scale + 0.5;
	if (q <= 0.0)
		return 0;
	if (q >= (double) maxq)
		return maxq;
	return (uint32) q;
}

/* Unsigned LEB128 varint helpers (used by the VARINT postings encoding). */
static inline int
PgturbohybridSparseVarintEncode(uint32 value, uint8 *buf)
{
	int			n = 0;

	while (value >= 0x80)
	{
		buf[n++] = (uint8) (value | 0x80);
		value >>= 7;
	}
	buf[n++] = (uint8) value;
	return n;
}

static inline uint32
PgturbohybridSparseVarintDecode(const uint8 *buf, int *consumed)
{
	uint32		value = 0;
	int			shift = 0;
	int			n = 0;
	uint8		b;

	do
	{
		b = buf[n++];
		value |= (uint32) (b & 0x7f) << shift;
		shift += 7;
	} while ((b & 0x80) != 0 && shift < 35);
	*consumed = n;
	return value;
}

/* Meta tuple: written once per build, anchored by tqSparseMetaStartBlkno. */
typedef struct PgturbohybridSparseMetaTupleData
{
	uint8		type;			/* PGTURBOHYBRID_SPARSE_META_TUPLE_TYPE */
	uint8		quantBits;		/* 0 (f32), 8, or 16 */
	uint8		quantMode;		/* PGTURBOHYBRID_SPARSE_QUANT_* */
	uint8		postingsEncoding;	/* PGTURBOHYBRID_SPARSE_ENCODING_* (resolved) */
	uint32		sparseVersion;	/* = PGTURBOHYBRID_SPARSE_VERSION */
	uint32		termCount;		/* distinct terms */
	uint32		docCount;		/* docs with >= 1 retained sparse term */
	uint64		postingCount;	/* total (term, node) postings */
	BlockNumber lexiconStartBlkno;	/* head of lexicon page chain */
	uint32		lexiconPages;
	uint32		postingsPages;
	uint32		blockSize;		/* postings per chunk (build-time block) */
} PgturbohybridSparseMetaTupleData;

typedef PgturbohybridSparseMetaTupleData *PgturbohybridSparseMetaTuple;

/* One lexicon record per distinct term; packed into lexicon tuples. */
typedef struct PgturbohybridSparseLexiconEntry
{
	int32		termId;
	uint32		df;				/* # postings (docs) for this term */
	float4		maxWeight;		/* max doc weight (also used for WAND later) */
	float4		scale;			/* dequant scale = maxWeight/(2^bits-1); 0 if f32 */
	BlockNumber postingsBlkno;	/* first postings chunk for this term */
	OffsetNumber postingsOffno;
	uint16		reserved;
} PgturbohybridSparseLexiconEntry;

typedef struct PgturbohybridSparseLexiconTupleData
{
	uint8		type;			/* PGTURBOHYBRID_SPARSE_LEXICON_TUPLE_TYPE */
	uint8		reserved1;
	uint16		count;			/* # entries in this tuple */
	PgturbohybridSparseLexiconEntry entries[FLEXIBLE_ARRAY_MEMBER];
} PgturbohybridSparseLexiconTupleData;

typedef PgturbohybridSparseLexiconTupleData *PgturbohybridSparseLexiconTuple;

/*
 * One postings chunk for a single term.  Chunks for a term are written
 * consecutively (sorted by node_id) and read by physical walk from the
 * lexicon's first-chunk pointer until the term's df postings are consumed.
 * The payload after the header is encoding/bits specific (see
 * pgturbohybrid_sparse_build.c / _query.c); node_ids are stored relative to
 * baseNodeId (the chunk's first node_id).  A SoA chunk spans < 2^16 node_ids
 * (build splits on that range), so chunks are not necessarily blockSize-full.
 */
typedef struct PgturbohybridSparsePostingsTupleData
{
	uint8		type;			/* PGTURBOHYBRID_SPARSE_POSTINGS_TUPLE_TYPE */
	uint8		encoding;		/* PGTURBOHYBRID_SPARSE_ENCODING_* */
	uint16		count;			/* # postings in this chunk */
	int32		termId;
	uint32		baseNodeId;		/* node_id of the first posting in the chunk */
	char		payload[FLEXIBLE_ARRAY_MEMBER];
} PgturbohybridSparsePostingsTupleData;

#define PGTURBOHYBRID_SPARSE_POSTINGS_PER_CHUNK PGTURBOHYBRID_SPARSE_DEFAULT_BLOCK_SIZE
#define PGTURBOHYBRID_SPARSE_LEXICON_PER_TUPLE 200

typedef PgturbohybridSparsePostingsTupleData *PgturbohybridSparsePostingsTuple;

#define PGTURBOHYBRID_SPARSE_POSTINGS_HEADER \
	(offsetof(PgturbohybridSparsePostingsTupleData, payload))

/* Worst-case chunk payload: f32 weights + max-size node addressing per posting. */
#define PgturbohybridSparseChunkMaxSize(n) \
	(MAXALIGN(PGTURBOHYBRID_SPARSE_POSTINGS_HEADER + 4 + \
			  (Size) (n) * (sizeof(float4) + 5)))

#define PgturbohybridSparseLexiconTupleSize(n) \
	(MAXALIGN(offsetof(PgturbohybridSparseLexiconTupleData, entries) + \
			  (Size) (n) * sizeof(PgturbohybridSparseLexiconEntry)))

/* A scored sparse candidate returned by the scan branch (higher score = better). */
typedef struct PgturbohybridSparseCandidate
{
	uint32		nodeId;
	ItemPointerData heaptid;
	double		score;			/* exact sparse inner product */
} PgturbohybridSparseCandidate;

/* Per-scan stats for the sparse branch (mirrors the BM25 branch stats). */
typedef struct PgturbohybridSparseScanStats
{
	bool		branchAvailable;	/* index has a sparse key + sparse meta */
	bool		branchUsed;
	uint32		terms;				/* query sparse terms */
	uint32		resolvedTerms;		/* query terms found in the lexicon */
	uint64		postingsTouched;
	uint64		candidatesScored;
	uint64		elapsedUs;
	int			quantBits;			/* 0 (f32), 8, or 16 */
	int			quantMode;			/* PGTURBOHYBRID_SPARSE_QUANT_* */
	int			encoding;			/* PGTURBOHYBRID_SPARSE_ENCODING_* */
	uint64		scalarTailPostings; /* postings scored by the scalar path */
	int			rerankMode;			/* PgturbohybridSparseRerankMode */
	uint64		exactRerankCount;	/* candidates exact-reranked from the heap */
	uint64		exactRerankFetchUs;
	uint64		exactRerankScoreUs;
	bool		exactRerankTopkChanged;
} PgturbohybridSparseScanStats;

/*
 * Build the sparse inverted index over the index's sparse key, after the dense
 * graph build has assigned node_ids.  No-op if the index has no sparse key.
 */
void		PgturbohybridSparseBuildCollect(Relation heap, Relation index,
											IndexInfo *indexInfo);

/*
 * Resolve the query's sparse_query against the index, exact-OR-accumulate
 * scores over live nodes, and return up to k candidates sorted by descending
 * score (palloc'd in ctx).  Returns the count; 0 if the index has no sparse
 * data or the query has no sparse_query.  Fills stats when non-NULL.
 */
int			PgturbohybridSparseCollectCandidates(Relation index,
												 PgturbohybridQueryHeader *query,
												 int k,
												 PgturbohybridSparseCandidate **out,
												 MemoryContext ctx,
												 PgturbohybridSparseScanStats *stats);

/* True iff the index has a sparse key with built sparse meta. */
bool		PgturbohybridSparseIndexAvailable(Relation index);

#endif							/* PGTURBOHYBRID_SPARSE_H */
