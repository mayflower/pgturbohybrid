#include "postgres.h"

#include <math.h>

#include "access/htup_details.h"
#include "catalog/pg_extension.h"
#include "catalog/pg_type.h"
#include "commands/extension.h"
#include "fmgr.h"
#include "lib/stringinfo.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/syscache.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

#include "pgturbohybrid_multivector.h"
#include "pgturbohybrid_query.h"

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_in);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_out);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_constructor);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_dims);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_count);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_maxsim);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_maxsim_distance);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_query_distance);

static Oid	pgturbohybrid_multivector_type_oid = InvalidOid;

static Oid PgturbohybridExtensionSchema(Oid extensionOid);
static void PgturbohybridCheckMultiVectorHeader(int32 count, int32 dim);

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
PgturbohybridMultiVectorTypeOid(void)
{
	Oid			extensionOid;
	Oid			schemaOid;

	if (OidIsValid(pgturbohybrid_multivector_type_oid))
		return pgturbohybrid_multivector_type_oid;

	extensionOid = get_extension_oid("pgturbohybrid", true);
	if (!OidIsValid(extensionOid))
		return InvalidOid;

	schemaOid = PgturbohybridExtensionSchema(extensionOid);
	if (!OidIsValid(schemaOid))
		return InvalidOid;

	pgturbohybrid_multivector_type_oid =
		GetSysCacheOid2(TYPENAMENSP, Anum_pg_type_oid,
						CStringGetDatum("turbohybrid_multivector"),
						ObjectIdGetDatum(schemaOid));
	return pgturbohybrid_multivector_type_oid;
}

static void
PgturbohybridCheckMultiVectorHeader(int32 count, int32 dim)
{
	if (dim <= 0 || dim > PGTURBOHYBRID_MULTIVECTOR_MAX_DIM)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("invalid multivector dimensions %d", dim)));

	if (count <= 0 || count > PGTURBOHYBRID_MULTIVECTOR_MAX_COUNT)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("invalid multivector count %d", count)));
}

Size
PgturbohybridMultiVectorFloatCount(int32 count, int32 dim)
{
	Size		countSize;
	Size		dimSize;

	PgturbohybridCheckMultiVectorHeader(count, dim);
	countSize = (Size) count;
	dimSize = (Size) dim;

	if (countSize > MaxAllocSize / dimSize)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("multivector value is too large")));

	return countSize * dimSize;
}

Size
PgturbohybridMultiVectorSize(int32 count, int32 dim)
{
	Size		floatCount = PgturbohybridMultiVectorFloatCount(count, dim);

	if (floatCount > (MaxAllocSize - offsetof(PgturbohybridMultiVector, values)) / sizeof(float))
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("multivector value is too large")));

	return offsetof(PgturbohybridMultiVector, values) + floatCount * sizeof(float);
}

void
PgturbohybridCheckMultiVector(const PgturbohybridMultiVector *mv)
{
	Size		actual;
	Size		expected;
	Size		floatCount;

	if (mv == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("multivector value cannot be null")));

	actual = VARSIZE_ANY(mv);
	if (actual < offsetof(PgturbohybridMultiVector, values))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("malformed multivector value"),
				 errdetail("Multivector varlena size is too small.")));

	expected = PgturbohybridMultiVectorSize(mv->count, mv->dim);
	if (actual != expected)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("malformed multivector value"),
				 errdetail("Multivector payload size is %zu bytes but %zu bytes were expected.",
						   actual, expected)));

	floatCount = PgturbohybridMultiVectorFloatCount(mv->count, mv->dim);
	for (Size i = 0; i < floatCount; i++)
	{
		if (!isfinite(mv->values[i]))
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("multivector cannot contain NaN or infinite values")));
	}
}

PgturbohybridMultiVector *
PgturbohybridDatumGetMultiVector(Datum value)
{
	PgturbohybridMultiVector *mv;

	mv = (PgturbohybridMultiVector *) PG_DETOAST_DATUM(value);
	PgturbohybridCheckMultiVector(mv);
	return mv;
}

void
PgturbohybridCheckSameMultiVectorDims(const PgturbohybridMultiVector *a,
									  const PgturbohybridMultiVector *b)
{
	PgturbohybridCheckMultiVector(a);
	PgturbohybridCheckMultiVector(b);

	if (a->dim != b->dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("different multivector dimensions %d and %d", a->dim, b->dim)));
}

TqDocId
PgturbohybridMultiVectorMakeDocId(uint64 docOrdinal)
{
	if (docOrdinal >= (uint64) TQ_INVALID_DOC_ID)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("too many multivector documents")));

	return (TqDocId) docOrdinal;
}

TqSubvectorOrdinal
PgturbohybridMultiVectorMakeSubvectorOrdinal(uint32 tokenOrdinal)
{
	if (tokenOrdinal >= (uint32) TQ_INVALID_SUBVECTOR_ORDINAL)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("too many subvectors in one multivector document")));

	return (TqSubvectorOrdinal) tokenOrdinal;
}

void
PgturbohybridMultiVectorCheckTokenCount(uint32 tokenCount, uint32 maxTokenCount)
{
	if (tokenCount == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("multivector token count must be greater than zero")));

	if (maxTokenCount == 0 || tokenCount > maxTokenCount)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("multivector token count %u exceeds configured limit %u",
						tokenCount, maxTokenCount)));
}

void
PgturbohybridMultiVectorCheckDim(uint32 dim, uint32 maxDim)
{
	if (dim == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("multivector dimensions must be greater than zero")));

	if (maxDim == 0 || dim > maxDim)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("multivector dimensions %u exceed configured limit %u",
						dim, maxDim)));
}

const float *
PgturbohybridMultiVectorValues(const PgturbohybridMultiVector *mv, int32 ordinal)
{
	PgturbohybridCheckMultiVector(mv);

	if (ordinal < 0 || ordinal >= mv->count)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("multivector ordinal %d is out of range", ordinal)));

	return mv->values + ((Size) ordinal * (Size) mv->dim);
}

double
TqDotProductF32Scalar(const float *a, const float *b, int32 dim)
{
	double		result = 0.0;

	for (int32 i = 0; i < dim; i++)
		result += (double) a[i] * (double) b[i];

	return result;
}

double
TqMultiVectorMaxSimScalar(const PgturbohybridMultiVector *query,
						  const PgturbohybridMultiVector *doc)
{
	double		score = 0.0;

	PgturbohybridCheckSameMultiVectorDims(query, doc);

	for (int32 qi = 0; qi < query->count; qi++)
	{
		const float *qv = PgturbohybridMultiVectorValues(query, qi);
		double		best = -INFINITY;

		for (int32 di = 0; di < doc->count; di++)
		{
			const float *dv = PgturbohybridMultiVectorValues(doc, di);
			double		dot = TqDotProductF32Scalar(qv, dv, query->dim);

			if (dot > best)
				best = dot;
		}

		score += best;
	}

	return score;
}

Datum
pgturbohybrid_multivector_in(PG_FUNCTION_ARGS)
{
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("turbohybrid_multivector text input is not supported"),
			 errhint("Use turbohybrid_multivector(vector[]) to construct a multivector value.")));

	PG_RETURN_POINTER(NULL);
}

Datum
pgturbohybrid_multivector_out(PG_FUNCTION_ARGS)
{
	PgturbohybridMultiVector *mv = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(0);
	StringInfoData buf;

	initStringInfo(&buf);
	appendStringInfo(&buf, "turbohybrid_multivector(dim=%d,count=%d,values=[", mv->dim, mv->count);

	for (int32 i = 0; i < mv->count; i++)
	{
		const float *values = PgturbohybridMultiVectorValues(mv, i);

		if (i > 0)
			appendStringInfoChar(&buf, ',');
		appendStringInfoChar(&buf, '[');
		for (int32 j = 0; j < mv->dim; j++)
		{
			if (j > 0)
				appendStringInfoChar(&buf, ',');
			appendStringInfo(&buf, "%.9g", values[j]);
		}
		appendStringInfoChar(&buf, ']');
	}

	appendStringInfoString(&buf, "])");
	PG_RETURN_CSTRING(buf.data);
}

Datum
pgturbohybrid_multivector_constructor(PG_FUNCTION_ARGS)
{
	ArrayType  *array;
	Datum	   *elements;
	bool	   *nulls;
	int			nelems;
	Oid			vectorOid = PgturbohybridVectorTypeOid();
	int16		typlen;
	bool		typbyval;
	char		typalign;
	int32		dim = 0;
	Size		resultSize;
	PgturbohybridMultiVector *result;
	float	   *dest;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("vector array cannot be null")));

	array = PG_GETARG_ARRAYTYPE_P(0);
	if (ARR_NDIM(array) == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector array cannot be empty")));

	if (ARR_ELEMTYPE(array) != vectorOid)
		ereport(ERROR,
				(errcode(ERRCODE_DATATYPE_MISMATCH),
				 errmsg("expected vector[] input")));

	get_typlenbyvalalign(vectorOid, &typlen, &typbyval, &typalign);
	deconstruct_array(array, vectorOid, typlen, typbyval, typalign,
					  &elements, &nulls, &nelems);

	if (nelems <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("vector array cannot be empty")));

	for (int i = 0; i < nelems; i++)
	{
		Vector	   *vector;
		int			vectorDim;

		if (nulls[i])
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("vector array cannot contain null elements")));

		vector = PgturbohybridDatumGetVector(elements[i]);
		vectorDim = PgturbohybridVectorDims(vector);
		if (i == 0)
			dim = vectorDim;
		else if (vectorDim != dim)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("all vectors in a multivector must have the same dimensions")));
	}

	resultSize = PgturbohybridMultiVectorSize(nelems, dim);
	result = (PgturbohybridMultiVector *) palloc0(resultSize);
	SET_VARSIZE(result, resultSize);
	result->dim = dim;
	result->count = nelems;
	result->flags = 0;
	dest = result->values;

	for (int i = 0; i < nelems; i++)
	{
		Vector	   *vector = PgturbohybridDatumGetVector(elements[i]);

		memcpy(dest, vector->x, sizeof(float) * (Size) dim);
		dest += dim;
	}

	PG_RETURN_PGTURBOHYBRID_MULTIVECTOR_P(result);
}

Datum
pgturbohybrid_multivector_dims(PG_FUNCTION_ARGS)
{
	PgturbohybridMultiVector *mv = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(0);

	PG_RETURN_INT32(mv->dim);
}

Datum
pgturbohybrid_multivector_count(PG_FUNCTION_ARGS)
{
	PgturbohybridMultiVector *mv = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(0);

	PG_RETURN_INT32(mv->count);
}

Datum
pgturbohybrid_multivector_maxsim(PG_FUNCTION_ARGS)
{
	PgturbohybridMultiVector *query = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(0);
	PgturbohybridMultiVector *doc = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(1);

	PG_RETURN_FLOAT8(TqMultiVectorMaxSimScalar(query, doc));
}

Datum
pgturbohybrid_multivector_maxsim_distance(PG_FUNCTION_ARGS)
{
	PgturbohybridMultiVector *query = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(0);
	PgturbohybridMultiVector *doc = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(1);

	PG_RETURN_FLOAT8(-TqMultiVectorMaxSimScalar(query, doc));
}

Datum
pgturbohybrid_multivector_query_distance(PG_FUNCTION_ARGS)
{
	PgturbohybridMultiVector *doc = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(0);
	PgturbohybridQueryHeader *query = PG_GETARG_PGTURBOHYBRID_QUERY_P(1);
	PgturbohybridMultiVector *queryMv;

	queryMv = PgturbohybridQueryGetMultiVector(query);
	if (queryMv == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("turbohybrid_query requires multivector_query for multivector distance")));

	PG_RETURN_FLOAT8(-TqMultiVectorMaxSimScalar(queryMv, doc));
}
