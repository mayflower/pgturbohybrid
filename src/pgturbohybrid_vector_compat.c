#include "postgres.h"

#include <math.h>

#include "access/htup_details.h"
#include "catalog/pg_extension.h"
#include "catalog/pg_type.h"
#include "commands/extension.h"
#include "utils/syscache.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

#include "pgturbohybrid_vector_compat.h"

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_vector_l2_squared_distance);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_vector_l2_distance);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_vector_negative_inner_product);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_vector_cosine_distance);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_vector_norm);

static Oid	pgturbohybrid_vector_type_oid = InvalidOid;
static uint64 pgturbohybrid_strict_vector_validations = 0;
static uint64 pgturbohybrid_fast_vector_checks = 0;
static uint64 pgturbohybrid_vector_type_cache_hits = 0;
static uint64 pgturbohybrid_vector_type_cache_misses = 0;

static void PgturbohybridEnsureVectorType(void);
static Oid PgturbohybridExtensionSchema(Oid extensionOid);
static int PgturbohybridCheckVectorInternal(const Vector *vector, bool strict);

static void
PgturbohybridEnsureVectorType(void)
{
	(void) PgturbohybridVectorTypeOid();
}

static Oid
PgturbohybridExtensionSchema(Oid extensionOid)
{
	Form_pg_extension extensionForm;
	HeapTuple	tuple;
	Oid			schemaOid;

	tuple = SearchSysCache1(EXTENSIONOID, ObjectIdGetDatum(extensionOid));
	if (!HeapTupleIsValid(tuple))
		return InvalidOid;

	extensionForm = (Form_pg_extension) GETSTRUCT(tuple);
	schemaOid = extensionForm->extnamespace;
	ReleaseSysCache(tuple);

	return schemaOid;
}

Oid
PgturbohybridVectorTypeOid(void)
{
	Oid			extensionOid;
	Oid			schemaOid;
	Oid			typeOid;

	if (OidIsValid(pgturbohybrid_vector_type_oid))
	{
		pgturbohybrid_vector_type_cache_hits++;
		return pgturbohybrid_vector_type_oid;
	}

	pgturbohybrid_vector_type_cache_misses++;
	extensionOid = get_extension_oid("vector", true);
	if (!OidIsValid(extensionOid))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("vector extension is required by pgturbohybrid"),
				 errhint("Run CREATE EXTENSION vector before using pgturbohybrid.")));

	schemaOid = PgturbohybridExtensionSchema(extensionOid);
	if (!OidIsValid(schemaOid))
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_SCHEMA),
				 errmsg("could not find schema for vector extension")));

	typeOid = GetSysCacheOid2(TYPENAMENSP, Anum_pg_type_oid,
							  CStringGetDatum("vector"),
							  ObjectIdGetDatum(schemaOid));
	if (!OidIsValid(typeOid))
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("could not find vector type installed by vector extension")));

	pgturbohybrid_vector_type_oid = typeOid;
	return typeOid;
}

Vector *
PgturbohybridDatumGetVector(Datum value)
{
	Vector	   *vector;

	vector = (Vector *) PG_DETOAST_DATUM(value);
	PgturbohybridCheckVector(vector);
	return vector;
}

int
PgturbohybridVectorDims(const Vector *vector)
{
	PgturbohybridCheckVector(vector);
	return vector->dim;
}

static int
PgturbohybridCheckVectorInternal(const Vector *vector, bool strict)
{
	Size		size;
	Size		expected;
	int			dim;

	if (strict)
	{
		pgturbohybrid_strict_vector_validations++;
		PgturbohybridEnsureVectorType();
	}
	else
		pgturbohybrid_fast_vector_checks++;

	if (vector == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("vector value cannot be null")));

	size = VARSIZE_ANY(vector);
	if (size < offsetof(Vector, x))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("malformed vector value"),
				 errdetail("Vector varlena size is too small.")));

	dim = vector->dim;
	if (dim < 1 || dim > PGTURBOHYBRID_VECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("invalid vector dimensions %d", dim)));

	if (vector->unused != 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("malformed vector value"),
				 errdetail("Reserved vector header field is not zero.")));

	expected = PGTURBOHYBRID_VECTOR_SIZE(dim);
	if (size != expected)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("malformed vector value"),
				 errdetail("Vector payload size is %zu bytes but %zu bytes were expected.",
						   size, expected)));

	if (strict)
	{
		for (int i = 0; i < dim; i++)
		{
			if (!isfinite(vector->x[i]))
				ereport(ERROR,
						(errcode(ERRCODE_DATA_EXCEPTION),
						 errmsg("vector cannot contain NaN or infinite values")));
		}
	}

	return dim;
}

void
PgturbohybridCheckVector(const Vector *vector)
{
	(void) PgturbohybridCheckVectorInternal(vector, true);
}

int
PgturbohybridVectorDimsFast(const Vector *vector)
{
	return PgturbohybridCheckVectorInternal(vector, false);
}

void
PgturbohybridCheckVectorFast(const Vector *vector)
{
	(void) PgturbohybridCheckVectorInternal(vector, false);
}

void
PgturbohybridCheckSameDims(const Vector *a, const Vector *b)
{
	int			adim;
	int			bdim;

	adim = PgturbohybridVectorDims(a);
	bdim = PgturbohybridVectorDims(b);

	if (adim != bdim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("different vector dimensions %d and %d", adim, bdim)));
}

void
PgturbohybridCheckSameDimsFast(const Vector *a, const Vector *b)
{
	int			adim;
	int			bdim;

	adim = PgturbohybridVectorDimsFast(a);
	bdim = PgturbohybridVectorDimsFast(b);

	if (adim != bdim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("different vector dimensions %d and %d", adim, bdim)));
}

double
PgturbohybridL2SquaredDistance(const Vector *a, const Vector *b)
{
	double		distance = 0.0;
	int			dim;

	PgturbohybridCheckSameDimsFast(a, b);
	dim = a->dim;

	for (int i = 0; i < dim; i++)
	{
		double		diff = (double) a->x[i] - (double) b->x[i];

		distance += diff * diff;
	}

	return distance;
}

double
PgturbohybridL2Distance(const Vector *a, const Vector *b)
{
	return sqrt(PgturbohybridL2SquaredDistance(a, b));
}

double
PgturbohybridNegativeInnerProduct(const Vector *a, const Vector *b)
{
	double		distance = 0.0;
	int			dim;

	PgturbohybridCheckSameDimsFast(a, b);
	dim = a->dim;

	for (int i = 0; i < dim; i++)
		distance -= (double) a->x[i] * (double) b->x[i];

	return distance;
}

double
PgturbohybridCosineDistance(const Vector *a, const Vector *b)
{
	double		similarity = 0.0;
	double		norma = 0.0;
	double		normb = 0.0;
	int			dim;

	PgturbohybridCheckSameDimsFast(a, b);
	dim = a->dim;

	for (int i = 0; i < dim; i++)
	{
		similarity += (double) a->x[i] * (double) b->x[i];
		norma += (double) a->x[i] * (double) a->x[i];
		normb += (double) b->x[i] * (double) b->x[i];
	}

	if (norma == 0.0 || normb == 0.0)
		return 1.0;

	similarity = similarity / sqrt(norma * normb);
	if (similarity > 1.0)
		similarity = 1.0;
	else if (similarity < -1.0)
		similarity = -1.0;

	return 1.0 - similarity;
}

double
PgturbohybridVectorNorm(const Vector *vector)
{
	double		norm = 0.0;
	int			dim;

	dim = PgturbohybridVectorDimsFast(vector);

	for (int i = 0; i < dim; i++)
		norm += (double) vector->x[i] * (double) vector->x[i];

	return sqrt(norm);
}

Vector *
PgturbohybridL2NormalizeFast(const Vector *vector)
{
	Vector	   *result;
	double		norm = 0.0;
	int			dim;
	Size		size;

	dim = PgturbohybridVectorDimsFast(vector);

	for (int i = 0; i < dim; i++)
		norm += (double) vector->x[i] * (double) vector->x[i];

	norm = sqrt(norm);
	size = PGTURBOHYBRID_VECTOR_SIZE(dim);

	result = palloc0(size);
	SET_VARSIZE(result, size);
	result->dim = dim;

	if (norm > 0.0)
	{
		for (int i = 0; i < dim; i++)
			result->x[i] = (float) ((double) vector->x[i] / norm);
	}

	return result;
}

void
PgturbohybridGetValidationStats(PgturbohybridValidationStats *stats)
{
	stats->strictVectorValidations = pgturbohybrid_strict_vector_validations;
	stats->fastVectorChecks = pgturbohybrid_fast_vector_checks;
	stats->vectorTypeCacheHits = pgturbohybrid_vector_type_cache_hits;
	stats->vectorTypeCacheMisses = pgturbohybrid_vector_type_cache_misses;
}

Datum
pgturbohybrid_vector_l2_squared_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_PGTURBOHYBRID_VECTOR_P(0);
	Vector	   *b = PG_GETARG_PGTURBOHYBRID_VECTOR_P(1);

	PG_RETURN_FLOAT8(PgturbohybridL2SquaredDistance(a, b));
}

Datum
pgturbohybrid_vector_l2_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_PGTURBOHYBRID_VECTOR_P(0);
	Vector	   *b = PG_GETARG_PGTURBOHYBRID_VECTOR_P(1);

	PG_RETURN_FLOAT8(PgturbohybridL2Distance(a, b));
}

Datum
pgturbohybrid_vector_negative_inner_product(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_PGTURBOHYBRID_VECTOR_P(0);
	Vector	   *b = PG_GETARG_PGTURBOHYBRID_VECTOR_P(1);

	PG_RETURN_FLOAT8(PgturbohybridNegativeInnerProduct(a, b));
}

Datum
pgturbohybrid_vector_cosine_distance(PG_FUNCTION_ARGS)
{
	Vector	   *a = PG_GETARG_PGTURBOHYBRID_VECTOR_P(0);
	Vector	   *b = PG_GETARG_PGTURBOHYBRID_VECTOR_P(1);

	PG_RETURN_FLOAT8(PgturbohybridCosineDistance(a, b));
}

Datum
pgturbohybrid_vector_norm(PG_FUNCTION_ARGS)
{
	Vector	   *vector = PG_GETARG_PGTURBOHYBRID_VECTOR_P(0);

	PG_RETURN_FLOAT8(PgturbohybridVectorNorm(vector));
}

FUNCTION_PREFIX Datum
pgturbohybrid_l2_normalize(PG_FUNCTION_ARGS)
{
	Vector	   *vector = PG_GETARG_PGTURBOHYBRID_VECTOR_P(0);

	PG_RETURN_PGTURBOHYBRID_VECTOR_P(PgturbohybridL2NormalizeFast(vector));
}
