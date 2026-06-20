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

#define PGTURBOHYBRID_SPARSE_VERSION 1

#define PGTURBOHYBRID_SPARSE_META_TUPLE_TYPE 0x71
#define PGTURBOHYBRID_SPARSE_LEXICON_TUPLE_TYPE 0x72
#define PGTURBOHYBRID_SPARSE_POSTINGS_TUPLE_TYPE 0x73

/* Meta tuple: written once per build, anchored by tqSparseMetaStartBlkno. */
typedef struct PgturbohybridSparseMetaTupleData
{
	uint8		type;			/* PGTURBOHYBRID_SPARSE_META_TUPLE_TYPE */
	uint8		reserved1;
	uint16		reserved2;
	uint32		sparseVersion;	/* = PGTURBOHYBRID_SPARSE_VERSION */
	uint32		termCount;		/* distinct terms */
	uint32		docCount;		/* docs with >= 1 retained sparse term */
	uint64		postingCount;	/* total (term, node) postings */
	BlockNumber lexiconStartBlkno;	/* head of lexicon page chain */
	uint32		lexiconPages;
	uint32		postingsPages;
} PgturbohybridSparseMetaTupleData;

typedef PgturbohybridSparseMetaTupleData *PgturbohybridSparseMetaTuple;

/* One lexicon record per distinct term; packed into lexicon tuples. */
typedef struct PgturbohybridSparseLexiconEntry
{
	int32		termId;
	uint32		df;				/* # postings (docs) for this term */
	float4		maxWeight;		/* max doc weight (reserved for WAND) */
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

/* On-disk posting: dense graph node_id + exact float32 weight. */
typedef struct PgturbohybridSparsePosting
{
	uint32		nodeId;
	float4		weight;
} PgturbohybridSparsePosting;

/*
 * One postings chunk for a single term.  Chunks for a term are written
 * consecutively (sorted by node_id) into the postings page chain, so the scan
 * reads a term's ceil(df / PGTURBOHYBRID_SPARSE_POSTINGS_PER_CHUNK) chunks by
 * physical walk from the lexicon's first-chunk pointer -- no per-chunk links.
 * Every chunk except a term's last is full (PER_CHUNK postings).
 */
typedef struct PgturbohybridSparsePostingsTupleData
{
	uint8		type;			/* PGTURBOHYBRID_SPARSE_POSTINGS_TUPLE_TYPE */
	uint8		reserved1;
	uint16		count;			/* # postings in this chunk */
	int32		termId;
	PgturbohybridSparsePosting postings[FLEXIBLE_ARRAY_MEMBER];
} PgturbohybridSparsePostingsTupleData;

#define PGTURBOHYBRID_SPARSE_POSTINGS_PER_CHUNK 512
#define PGTURBOHYBRID_SPARSE_LEXICON_PER_TUPLE 200

typedef PgturbohybridSparsePostingsTupleData *PgturbohybridSparsePostingsTuple;

#define PgturbohybridSparseLexiconTupleSize(n) \
	(MAXALIGN(offsetof(PgturbohybridSparseLexiconTupleData, entries) + \
			  (Size) (n) * sizeof(PgturbohybridSparseLexiconEntry)))
#define PgturbohybridSparsePostingsTupleSize(n) \
	(MAXALIGN(offsetof(PgturbohybridSparsePostingsTupleData, postings) + \
			  (Size) (n) * sizeof(PgturbohybridSparsePosting)))

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
