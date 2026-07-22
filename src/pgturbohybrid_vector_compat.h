#ifndef PGTURBOHYBRID_VECTOR_COMPAT_H
#define PGTURBOHYBRID_VECTOR_COMPAT_H

#include "postgres.h"

#include "fmgr.h"

#ifdef PGTURBOHYBRID_USE_PGVECTOR_HEADER
#include "vector.h"
#define PGTURBOHYBRID_VECTOR_MAX_DIM VECTOR_MAX_DIM
#define PGTURBOHYBRID_VECTOR_SIZE(_dim) ((Size) VECTOR_SIZE(_dim))
/* pgvector's header defines this for its own exports. */
#undef FUNCTION_PREFIX
#else
/* Explicit developer escape hatch; release, CI, and package builds require the
 * upstream header and never compile this private ABI mirror. */
#define PGTURBOHYBRID_VECTOR_MAX_DIM 16000
#define PGTURBOHYBRID_VECTOR_SIZE(_dim) \
	(offsetof(Vector, x) + sizeof(float) * (Size) (_dim))
typedef struct Vector
{
	int32		vl_len_;
	int16		dim;
	int16		unused;
	float		x[FLEXIBLE_ARRAY_MEMBER];
}			Vector;
#endif

typedef struct PgturbohybridValidationStats
{
	uint64		strictVectorValidations;
	uint64		fastVectorChecks;
	uint64		vectorTypeCacheHits;
	uint64		vectorTypeCacheMisses;
}			PgturbohybridValidationStats;

StaticAssertDecl(offsetof(Vector, dim) == sizeof(int32),
				 "pgvector vector dim offset changed");
StaticAssertDecl(offsetof(Vector, unused) == sizeof(int32) + sizeof(int16),
				 "pgvector vector unused offset changed");
StaticAssertDecl(offsetof(Vector, x) == sizeof(int32) + sizeof(int16) + sizeof(int16),
				 "pgvector vector data offset changed");

/* SQL-callable functions must stay visible even with -fvisibility=hidden. */
#define FUNCTION_PREFIX PGDLLEXPORT

#define PG_GETARG_PGTURBOHYBRID_VECTOR_P(_n) \
	PgturbohybridDatumGetVector(PG_GETARG_DATUM(_n))
#define PG_RETURN_PGTURBOHYBRID_VECTOR_P(_x) PG_RETURN_POINTER(_x)

Vector	   *PgturbohybridDatumGetVector(Datum value);
Oid			PgturbohybridVectorTypeOid(void);
int			PgturbohybridVectorDims(const Vector *vector);
void		PgturbohybridCheckVector(const Vector *vector);
void		PgturbohybridCheckSameDims(const Vector *a, const Vector *b);
int			PgturbohybridVectorDimsFast(const Vector *vector);
void		PgturbohybridCheckVectorFast(const Vector *vector);
void		PgturbohybridCheckSameDimsFast(const Vector *a, const Vector *b);
double		PgturbohybridL2SquaredDistance(const Vector *a, const Vector *b);
double		PgturbohybridL2Distance(const Vector *a, const Vector *b);
double		PgturbohybridNegativeInnerProduct(const Vector *a, const Vector *b);
double		PgturbohybridCosineDistance(const Vector *a, const Vector *b);
double		PgturbohybridVectorNorm(const Vector *vector);
Vector	   *PgturbohybridL2NormalizeFast(const Vector *vector);
void		PgturbohybridGetValidationStats(PgturbohybridValidationStats *stats);
FUNCTION_PREFIX Datum pgturbohybrid_l2_normalize(PG_FUNCTION_ARGS);

#endif
