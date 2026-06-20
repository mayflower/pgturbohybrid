#ifndef PGTURBOHYBRID_QUERY_H
#define PGTURBOHYBRID_QUERY_H

#include "postgres.h"

#include "tsearch/ts_type.h"
#include "pgturbohybrid_multivector.h"
#include "pgturbohybrid_vector_compat.h"

#define PGTURBOHYBRID_QUERY_VERSION 4

#define PGTURBOHYBRID_QUERY_FLAG_HAS_VECTOR			0x0001
#define PGTURBOHYBRID_QUERY_FLAG_HAS_TSQUERY			0x0002
#define PGTURBOHYBRID_QUERY_FLAG_ALPHA_IS_SET			0x0004
#define PGTURBOHYBRID_QUERY_FLAG_FINAL_K_IS_SET		0x0008
#define PGTURBOHYBRID_QUERY_FLAG_REQUIRE_BM25_MATCH	0x0010
#define PGTURBOHYBRID_QUERY_FLAG_DENSE_K_DEFAULTED		0x0020
#define PGTURBOHYBRID_QUERY_FLAG_BM25_K_DEFAULTED		0x0040
#define PGTURBOHYBRID_QUERY_FLAG_RRF_K_DEFAULTED		0x0080
#define PGTURBOHYBRID_QUERY_FLAG_HAS_DENSE			0x0100
#define PGTURBOHYBRID_QUERY_FLAG_HAS_MULTIVECTOR		0x0200
#define PGTURBOHYBRID_QUERY_FLAG_HAS_TOKEN_WEIGHTS		0x0400
#define PGTURBOHYBRID_QUERY_FLAG_HAS_TOKEN_MASK		0x0800
#define PGTURBOHYBRID_QUERY_FLAG_HAS_SPARSE			0x1000
#define PGTURBOHYBRID_QUERY_FLAG_REQUIRE_SPARSE_MATCH	0x2000
#define PGTURBOHYBRID_QUERY_FLAG_SPARSE_K_DEFAULTED	0x4000

typedef enum PgturbohybridDenseQueryKind
{
	PGTURBOHYBRID_DENSE_QUERY_NONE = 0,
	PGTURBOHYBRID_DENSE_QUERY_VECTOR = 1,
	PGTURBOHYBRID_DENSE_QUERY_MULTIVECTOR = 2
} PgturbohybridDenseQueryKind;

typedef enum PgturbohybridFusionMode
{
	PGTURBOHYBRID_FUSION_RRF = 1,
	PGTURBOHYBRID_FUSION_WEIGHTED = 2,
	PGTURBOHYBRID_FUSION_FAST_WEIGHTED = 3,
	PGTURBOHYBRID_FUSION_CALIBRATED = 4,
	PGTURBOHYBRID_FUSION_DBSF = 5
} PgturbohybridFusionMode;

typedef struct PgturbohybridQueryHeader
{
	int32		vl_len_;
	uint16		version;
	uint16		flags;
	uint16		fusion;
	uint16		reserved;
	float8		denseWeight;
	float8		bm25Weight;
	float8		multivectorWeight;
	float8		alpha;
	int32		rrfK;
	int32		denseK;
	int32		multivectorK;
	int32		bm25K;
	int32		finalK;
	uint16		denseKind;
	uint16		reserved2;
	int32		vectorBytes;
	int32		multivectorBytes;
	int32		tsqueryBytes;
	int32		multivectorDim;
	int32		multivectorCount;
	float8		sparseWeight;
	int32		sparseBytes;
	int32		sparseK;
	/* payload starts at MAXALIGN(sizeof(PgturbohybridQueryHeader)) */
} PgturbohybridQueryHeader;

#define DatumGetPgturbohybridQuery(x) ((PgturbohybridQueryHeader *) PG_DETOAST_DATUM(x))
#define PG_GETARG_PGTURBOHYBRID_QUERY_P(x) DatumGetPgturbohybridQuery(PG_GETARG_DATUM(x))
#define PG_RETURN_PGTURBOHYBRID_QUERY_P(x) PG_RETURN_POINTER(x)

static inline PgturbohybridDenseQueryKind
PgturbohybridQueryDenseKind(const PgturbohybridQueryHeader *query)
{
	return (PgturbohybridDenseQueryKind) query->denseKind;
}

static inline bool
PgturbohybridQueryHasVector(const PgturbohybridQueryHeader *query)
{
	return (query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_DENSE) != 0 &&
		(query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_VECTOR) != 0;
}

static inline bool
PgturbohybridQueryHasMultiVector(const PgturbohybridQueryHeader *query)
{
	return (query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_DENSE) != 0 &&
		(query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_MULTIVECTOR) != 0;
}

static inline bool
PgturbohybridQueryHasDense(const PgturbohybridQueryHeader *query)
{
	return PgturbohybridQueryHasVector(query) ||
		PgturbohybridQueryHasMultiVector(query);
}

static inline bool
PgturbohybridQueryHasText(const PgturbohybridQueryHeader *query)
{
	return (query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_TSQUERY) != 0;
}

static inline bool
PgturbohybridQueryHasSparse(const PgturbohybridQueryHeader *query)
{
	return (query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_SPARSE) != 0;
}

static inline bool
PgturbohybridQueryRequireSparseMatch(const PgturbohybridQueryHeader *query)
{
	return (query->flags & PGTURBOHYBRID_QUERY_FLAG_REQUIRE_SPARSE_MATCH) != 0;
}

Vector	   *PgturbohybridQueryGetVector(PgturbohybridQueryHeader *query);
PgturbohybridMultiVector *PgturbohybridQueryGetMultiVector(PgturbohybridQueryHeader *query);
const float4 *PgturbohybridQueryGetTokenWeights(PgturbohybridQueryHeader *query);
const bool *PgturbohybridQueryGetTokenMask(PgturbohybridQueryHeader *query);
bool		PgturbohybridQueryHasTokenWeights(const PgturbohybridQueryHeader *query);
bool		PgturbohybridQueryHasTokenMask(const PgturbohybridQueryHeader *query);
double		PgturbohybridQueryMultiVectorWeightSum(PgturbohybridQueryHeader *query);
TSQuery		PgturbohybridQueryGetTsQuery(PgturbohybridQueryHeader *query);
struct varlena *PgturbohybridQueryGetSparseVector(PgturbohybridQueryHeader *query);
void		PgturbohybridQueryValidate(PgturbohybridQueryHeader *query);
void		PgturbohybridQueryValidateFast(PgturbohybridQueryHeader *query);
const char *PgturbohybridQueryFusionName(uint16 fusion);
bool		PgturbohybridQueryTextIndexOrderByContext(FunctionCallInfo fcinfo);

#endif
