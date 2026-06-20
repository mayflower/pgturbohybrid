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

struct PgturbohybridGraphMetaPageData;

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
#define PGTURBOHYBRID_SPARSE_BLOCKMAX_TUPLE_TYPE 0x74
#define PGTURBOHYBRID_SPARSE_DELTA_TUPLE_TYPE 0x75
#define PGTURBOHYBRID_SPARSE_NODEMAP_TUPLE_TYPE 0x76

/*
 * Sparse-primary node-map tuple (prompt 12): a run of heap TIDs, node_id =
 * firstNodeId + index, so a sparse-only/sparse+BM25 index owns node identity
 * without a dense graph.  Liveness is delegated to heap-tuple MVCC visibility
 * (the executor filters dead TIDs), so all mapped nodes are treated as live.
 */
typedef struct PgturbohybridSparseNodeMapTupleData
{
	uint8		type;			/* PGTURBOHYBRID_SPARSE_NODEMAP_TUPLE_TYPE */
	uint8		reserved1;
	uint16		count;
	uint32		firstNodeId;
	ItemPointerData tids[FLEXIBLE_ARRAY_MEMBER];
} PgturbohybridSparseNodeMapTupleData;

typedef PgturbohybridSparseNodeMapTupleData *PgturbohybridSparseNodeMapTuple;

#define PgturbohybridSparseNodeMapTupleSize(n) \
	(MAXALIGN(offsetof(PgturbohybridSparseNodeMapTupleData, tids) + \
			  (Size) (n) * sizeof(ItemPointerData)))

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
	BlockNumber blockMaxStartBlkno; /* head of block-max directory chain (or Invalid) */
	uint8		hasBlockMax;	/* 1 if a block-max directory was written */
	uint8		reservedB1;
	uint16		reservedB2;
	BlockNumber deltaStartBlkno;	/* head of the delta tuple chain (or Invalid) */
	uint32		deltaGeneration;	/* bumped on every delta append and compaction */
	uint32		deltaDocCount;		/* docs appended since the last compaction */
} PgturbohybridSparseMetaTupleData;

typedef PgturbohybridSparseMetaTupleData *PgturbohybridSparseMetaTuple;

/* One lexicon record per distinct term; packed into lexicon tuples. */
typedef struct PgturbohybridSparseLexiconEntry
{
	int32		termId;
	uint32		df;				/* # postings (docs) for this term */
	float4		maxWeight;		/* max doc weight (WAND term upper bound basis) */
	float4		scale;			/* dequant scale = maxWeight/(2^bits-1); 0 if f32 */
	BlockNumber postingsBlkno;	/* first postings chunk for this term */
	OffsetNumber postingsOffno;
	uint16		reserved;
	BlockNumber blockMaxBlkno;	/* first block-max directory entry (or Invalid) */
	OffsetNumber blockMaxOffno;
	uint16		reserved2;
	uint32		blockCount;		/* # block-max entries (postings chunks) for this term */
} PgturbohybridSparseLexiconEntry;

/* One block-max directory entry per postings chunk (block-max WAND, prompt 9). */
typedef struct PgturbohybridSparseBlockMax
{
	uint32		firstNodeId;	/* first node_id in the chunk (= chunk baseNodeId) */
	uint32		lastNodeId;		/* last node_id in the chunk */
	float4		maxWeight;		/* max doc weight in the chunk (block upper bound) */
	BlockNumber postingsBlkno;	/* the chunk's location (for lazy loads) */
	OffsetNumber postingsOffno;
	uint16		reserved;
} PgturbohybridSparseBlockMax;

typedef struct PgturbohybridSparseBlockMaxTupleData
{
	uint8		type;			/* PGTURBOHYBRID_SPARSE_BLOCKMAX_TUPLE_TYPE */
	uint8		reserved1;
	uint16		count;			/* # entries in this tuple */
	int32		termId;
	PgturbohybridSparseBlockMax entries[FLEXIBLE_ARRAY_MEMBER];
} PgturbohybridSparseBlockMaxTupleData;

typedef PgturbohybridSparseBlockMaxTupleData *PgturbohybridSparseBlockMaxTuple;

#define PgturbohybridSparseBlockMaxTupleSize(n) \
	(MAXALIGN(offsetof(PgturbohybridSparseBlockMaxTupleData, entries) + \
			  (Size) (n) * sizeof(PgturbohybridSparseBlockMax)))
#define PGTURBOHYBRID_SPARSE_BLOCKMAX_PER_TUPLE 240

/* One (term_id, exact f32 weight) entry of an inserted row's sparse vector. */
typedef struct PgturbohybridSparseDeltaEntry
{
	int32		termId;
	float4		weight;
} PgturbohybridSparseDeltaEntry;

/* Delta tuple: one inserted/updated row's sparse vector, keyed on its node_id. */
typedef struct PgturbohybridSparseDeltaTupleData
{
	uint8		type;			/* PGTURBOHYBRID_SPARSE_DELTA_TUPLE_TYPE */
	uint8		reserved1;
	uint16		termCount;
	uint32		nodeId;
	ItemPointerData heaptid;
	PgturbohybridSparseDeltaEntry entries[FLEXIBLE_ARRAY_MEMBER];
} PgturbohybridSparseDeltaTupleData;

typedef PgturbohybridSparseDeltaTupleData *PgturbohybridSparseDeltaTuple;

#define PgturbohybridSparseDeltaTupleSize(n) \
	(MAXALIGN(offsetof(PgturbohybridSparseDeltaTupleData, entries) + \
			  (Size) (n) * sizeof(PgturbohybridSparseDeltaEntry)))

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
#define PGTURBOHYBRID_SPARSE_LEXICON_PER_TUPLE 150

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
	int			scoreKernel;		/* PGTURBOHYBRID_SPARSE_SCORE_* */
	uint64		simdBlocks;			/* SoA blocks scored via SIMD */
	bool		usedWand;			/* block-max WAND ran (vs exact accumulation) */
	uint64		blocksVisited;		/* postings chunks read by WAND */
	uint64		blocksSkipped;		/* postings chunks skipped via block-max */
	uint64		wandPruned;			/* documents pruned (pivot upper bound <= theta) */
	uint64		wandIterations;		/* WAND main-loop iterations */
	uint64		wandThresholdUpdates;	/* top-k threshold (theta) raises */
	uint64		wandHeapUpdates;	/* heap insert/replace operations */
	bool		cacheHit;			/* reader cache (lexicon+states) already built */
	uint64		cacheBuildUs;		/* time to build the reader cache this scan */
	uint64		cacheBytes;			/* reader cache bytes (lexicon + node states) */
	uint64		hotCacheHits;		/* hot-postings chunk cache hits */
	uint64		hotCacheMisses;		/* hot-postings chunk cache misses */
	uint64		hotCacheBytes;		/* hot-postings cache resident bytes */
	uint64		hotCacheEvictions;	/* hot-postings cache evictions */
	uint32		deltaPages;			/* delta tuples merged this scan */
	uint32		deltaTerms;			/* distinct delta terms merged */
	uint64		deltaPostingsDecoded;	/* delta (term, node) postings decoded */
	bool		deltaCacheHit;		/* delta postings served from the cache */
	uint32		deltaGeneration;	/* delta generation at scan time */
} PgturbohybridSparseScanStats;

/* SoA score-kernel ISA family (combined with bit width for the stat name). */
#define PGTURBOHYBRID_SPARSE_SCORE_SCALAR 0
#define PGTURBOHYBRID_SPARSE_SCORE_AVX2 1
#define PGTURBOHYBRID_SPARSE_SCORE_NEON 2

/*
 * SoA postings scorer (prompt 8): scores[base + offsets[k]] += termMul *
 * dequant(weights[k]) over a chunk, where weights is uint8 (q8) / uint16 (q16)
 * / float4 (f32).  SIMD kernels widen+multiply a block then scatter-add with a
 * scalar loop; the scalar kernel is the correctness reference.
 */
int			PgturbohybridSparseResolveScoreKernel(int bits, bool simdEnabled);
const char *PgturbohybridSparseScoreKernelName(int kernel, int bits);
void		PgturbohybridSparseScoreSoa(int kernel, const void *weights,
										const uint16 *offsets, uint32 count,
										uint32 base, int bits, double termMul,
										double *scores, uint32 nodeCount,
										uint64 *simdBlocks, uint64 *scalarTail);

/* ISA-specific SoA block kernels (defined in the per-arch SIMD files). */
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386) || defined(_M_IX86)
void		PgturbohybridSparseScoreSoaAvx2Q8(const uint8 *weights,
											  const uint16 *offsets, uint32 count,
											  uint32 base, double termMul,
											  double *scores, uint32 nodeCount,
											  uint64 *simdBlocks, uint64 *scalarTail);
void		PgturbohybridSparseScoreSoaAvx2Q16(const uint16 *weights,
											   const uint16 *offsets, uint32 count,
											   uint32 base, double termMul,
											   double *scores, uint32 nodeCount,
											   uint64 *simdBlocks, uint64 *scalarTail);
#endif
#if defined(__aarch64__) || defined(_M_ARM64)
void		PgturbohybridSparseScoreSoaNeonQ8(const uint8 *weights,
											  const uint16 *offsets, uint32 count,
											  uint32 base, double termMul,
											  double *scores, uint32 nodeCount,
											  uint64 *simdBlocks, uint64 *scalarTail);
void		PgturbohybridSparseScoreSoaNeonQ16(const uint16 *weights,
											   const uint16 *offsets, uint32 count,
											   uint32 base, double termMul,
											   double *scores, uint32 nodeCount,
											   uint64 *simdBlocks, uint64 *scalarTail);
#endif

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
												 int k, bool simdEnabled,
												 bool wandEnabled,
												 PgturbohybridSparseCandidate **out,
												 MemoryContext ctx,
												 PgturbohybridSparseScanStats *stats);

/* True iff the index has a sparse key with built sparse meta. */
bool		PgturbohybridSparseIndexAvailable(Relation index);

/* Read the sparse meta tuple at metaBlkno into *out; false if absent/malformed. */
bool		PgturbohybridSparseReadMeta(Relation index, BlockNumber metaBlkno,
										PgturbohybridSparseMetaTupleData *out);

/* Per-backend memory estimate for the sparse branch (prompt 10). */
typedef struct PgturbohybridSparseMemoryEstimate
{
	bool		available;
	uint32		termCount;
	uint32		docCount;
	uint32		nodeCount;
	int			quantBits;
	uint64		lexiconBytes;
	uint64		heapTidsBytes;
	uint64		livenessBytes;
	uint64		hotPostingsCacheMaxBytes;
	uint64		totalBytesPerBackend;
} PgturbohybridSparseMemoryEstimate;

bool		PgturbohybridSparseEstimateMemory(Relation index,
											  struct PgturbohybridGraphMetaPageData *graphMeta,
											  PgturbohybridSparseMemoryEstimate *out);

/* Drop the backend-local sparse cache for a relation (relfilenumber change). */
void		PgturbohybridSparseCacheInvalidate(Oid relid);

/*
 * Append an inserted row's sparse vector to the delta chain (prompt 11), keyed
 * on its dense graph node_id.  No-op if the index has no sparse data or the
 * datum is NULL/empty.  Returns true if a delta tuple was written.
 */
bool		PgturbohybridSparseAppendDelta(Relation index, uint32 nodeId,
										   ItemPointer heaptid, Datum sparseDatum);

/*
 * Compact the delta chain into the quantized base when the delta document count
 * reaches the configured threshold (or force=true): rebuild postings/lexicon/
 * block-max over the live base+delta postings and clear the delta chain.
 */
void		PgturbohybridSparseMaybeCompact(Relation index, bool force);

/* ---- Sparse-primary node space (prompt 12) ---------------------------- */

/* True iff the index's primary key is sparse/bm25 (no dense/multivector graph). */
bool		PgturbohybridSparseIsPrimary(Relation index);

/* Build a sparse-primary index: create the node-space metapage + node map over
 * the heap, then collect the sparse (and BM25) branches. */
IndexBuildResult *PgturbohybridSparsePrimaryBuild(Relation heap, Relation index,
												  IndexInfo *indexInfo);
void		PgturbohybridSparsePrimaryBuildEmpty(Relation index);

/* Allocate a node_id for an inserted row in the sparse-primary node space and
 * append it to the node map; returns the new node_id. */
uint32		PgturbohybridSparsePrimaryInsert(Relation index, ItemPointer heaptid);

#endif							/* PGTURBOHYBRID_SPARSE_H */
