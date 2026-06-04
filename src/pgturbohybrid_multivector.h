#ifndef PGTURBOHYBRID_MULTIVECTOR_H
#define PGTURBOHYBRID_MULTIVECTOR_H

#include "postgres.h"

#include "fmgr.h"
#include "storage/itemptr.h"

#include "pgturbohybrid_vector_compat.h"

#define PGTURBOHYBRID_MULTIVECTOR_MAX_DIM PGTURBOHYBRID_VECTOR_MAX_DIM
#define PGTURBOHYBRID_MULTIVECTOR_MAX_COUNT 4096
#define PGTURBOHYBRID_MULTIVECTOR_DEFAULT_MAX_DOC_VECTORS 256
#define PGTURBOHYBRID_MULTIVECTOR_DEFAULT_MAX_QUERY_VECTORS 64
#define PGTURBOHYBRID_MULTIVECTOR_SIZE(_count, _dim) \
	(offsetof(PgturbohybridMultiVector, values) + \
	 sizeof(float) * (Size) (_count) * (Size) (_dim))

#define TQ_INVALID_DOC_ID UINT32_MAX
#define TQ_INVALID_SUBVECTOR_ORDINAL UINT16_MAX

/*
 * Multivector indexes expand one heap tuple into multiple graph nodes.
 *
 * Single-vector path: nodeId can still be treated as the row/result identity.
 * Multivector path: nodeId identifies one document token/subvector only.
 * Result ranking and deduplication must use docId, then docId resolves back
 * to one heap TID.
 */
typedef uint32 TqDocId;
typedef uint16 TqSubvectorOrdinal;

typedef struct TqMultiVectorNodeMapEntry
{
	TqDocId		docId;
	TqSubvectorOrdinal tokenOrdinal;
} TqMultiVectorNodeMapEntry;

typedef struct TqMultiVectorDocMapEntry
{
	ItemPointerData heapTid;
	uint32		firstNodeId;
	uint16		tokenCount;
} TqMultiVectorDocMapEntry;

typedef struct PgturbohybridMultiVector
{
	int32		vl_len_;
	int32		dim;
	int32		count;
	uint32		flags;
	float		values[FLEXIBLE_ARRAY_MEMBER];
} PgturbohybridMultiVector;

#define DatumGetPgturbohybridMultiVector(x) \
	PgturbohybridDatumGetMultiVector(x)
#define PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(x) \
	DatumGetPgturbohybridMultiVector(PG_GETARG_DATUM(x))
#define PG_RETURN_PGTURBOHYBRID_MULTIVECTOR_P(x) PG_RETURN_POINTER(x)

PgturbohybridMultiVector *PgturbohybridDatumGetMultiVector(Datum value);
Oid			PgturbohybridMultiVectorTypeOid(void);
void		PgturbohybridCheckMultiVector(const PgturbohybridMultiVector *mv);
void		PgturbohybridCheckSameMultiVectorDims(const PgturbohybridMultiVector *a,
												   const PgturbohybridMultiVector *b);
TqDocId		PgturbohybridMultiVectorMakeDocId(uint64 docOrdinal);
TqSubvectorOrdinal PgturbohybridMultiVectorMakeSubvectorOrdinal(uint32 tokenOrdinal);
void		PgturbohybridMultiVectorCheckTokenCount(uint32 tokenCount,
													 uint32 maxTokenCount);
void		PgturbohybridMultiVectorCheckDim(uint32 dim, uint32 maxDim);
Size		PgturbohybridMultiVectorFloatCount(int32 count, int32 dim);
Size		PgturbohybridMultiVectorSize(int32 count, int32 dim);
const float *PgturbohybridMultiVectorValues(const PgturbohybridMultiVector *mv,
											int32 ordinal);
double		TqDotProductF32Scalar(const float *a, const float *b, int32 dim);
double		TqMultiVectorMaxSimScalar(const PgturbohybridMultiVector *query,
									   const PgturbohybridMultiVector *doc);
double		TqMultiVectorMaxSim(const PgturbohybridMultiVector *query,
								 const PgturbohybridMultiVector *doc);

FUNCTION_PREFIX Datum pgturbohybrid_multivector_in(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_out(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_constructor(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_dims(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_count(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_maxsim(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_maxsim_distance(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_query_distance(PG_FUNCTION_ARGS);

#endif
