#ifndef PGTURBOHYBRID_VECTOR_COMPAT_H
#define PGTURBOHYBRID_VECTOR_COMPAT_H

#include "postgres.h"

#include "fmgr.h"

/*
 * Private pgvector vector layout compatibility.
 *
 * pgturbohybrid depends on the SQL type provided by the vector extension, but
 * does not link against pgvector C symbols. The layout below matches pgvector's
 * vector varlena representation used by pgvector 0.5.x through 0.8.x. Recheck
 * this file before declaring support for a pgvector release with a different
 * on-disk/vector ABI.
 */
#define PGTURBOHYBRID_VECTOR_MAX_DIM 16000
#define PGTURBOHYBRID_VECTOR_SIZE(_dim) \
	(offsetof(Vector, x) + sizeof(float) * (Size) (_dim))

typedef struct Vector
{
	int32		vl_len_;		/* varlena header */
	int16		dim;			/* number of dimensions */
	int16		unused;			/* reserved by pgvector, expected to be zero */
	float		x[FLEXIBLE_ARRAY_MEMBER];
}			Vector;

StaticAssertDecl(offsetof(Vector, dim) == sizeof(int32),
				 "pgvector vector dim offset changed");
StaticAssertDecl(offsetof(Vector, unused) == sizeof(int32) + sizeof(int16),
				 "pgvector vector unused offset changed");
StaticAssertDecl(offsetof(Vector, x) == sizeof(int32) + sizeof(int16) + sizeof(int16),
				 "pgvector vector data offset changed");

#define PG_GETARG_PGTURBOHYBRID_VECTOR_P(_n) \
	PgturbohybridDatumGetVector(PG_GETARG_DATUM(_n))
#define PG_RETURN_PGTURBOHYBRID_VECTOR_P(_x) PG_RETURN_POINTER(_x)

Vector	   *PgturbohybridDatumGetVector(Datum value);
Oid			PgturbohybridVectorTypeOid(void);
int			PgturbohybridVectorDims(const Vector *vector);
void		PgturbohybridCheckVector(const Vector *vector);
void		PgturbohybridCheckSameDims(const Vector *a, const Vector *b);
double		PgturbohybridL2SquaredDistance(const Vector *a, const Vector *b);
double		PgturbohybridL2Distance(const Vector *a, const Vector *b);
double		PgturbohybridNegativeInnerProduct(const Vector *a, const Vector *b);
double		PgturbohybridCosineDistance(const Vector *a, const Vector *b);
double		PgturbohybridVectorNorm(const Vector *vector);
Datum		pgturbohybrid_l2_normalize(PG_FUNCTION_ARGS);

/* TODO Move to better place */
#if PG_VERSION_NUM >= 160000
#define FUNCTION_PREFIX
#else
#define FUNCTION_PREFIX PGDLLEXPORT
#endif

#endif
