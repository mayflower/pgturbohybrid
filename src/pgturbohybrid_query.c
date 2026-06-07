#include "postgres.h"

#include <math.h>

#include "catalog/pg_type.h"
#include "fmgr.h"
#include "lib/stringinfo.h"
#include "nodes/execnodes.h"
#include "nodes/nodes.h"
#include "nodes/plannodes.h"
#include "nodes/primnodes.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/float.h"
#include "utils/memutils.h"

#include "pgturbohybrid_query.h"
#include "pgturbohybrid_am.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_query_in);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_query_out);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_sparse_vector_in);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_sparse_vector_out);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_sparse_vector_from_arrays);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_sparse_vector_terms);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_sparse_vector_query_terms);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_query_constructor);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_distance);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_l2_distance);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_negative_inner_product);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_cosine_distance);

#define PGTURBOHYBRID_SPARSE_VECTOR_VERSION 1
#define PGTURBOHYBRID_SPARSE_VECTOR_FIELD_NONE (-1)

typedef struct PgturbohybridSparseVectorEntry
{
	int32		termId;
	float4		weight;
	int16		fieldId;
	uint16		reserved;
} PgturbohybridSparseVectorEntry;

typedef struct PgturbohybridSparseVector
{
	int32		vl_len_;
	uint16		version;
	uint16		flags;
	uint32		count;
	/* entries follow */
} PgturbohybridSparseVector;

static PgturbohybridSparseVectorEntry *PgturbohybridSparseVectorEntries(PgturbohybridSparseVector *sparse);
static void PgturbohybridSparseVectorValidate(PgturbohybridSparseVector *sparse);
static void PgturbohybridSparseVectorAppendTerm(StringInfo buf,
												int32 termId, int16 fieldId);
static Size PgturbohybridQueryVectorOffset(void);
static Size PgturbohybridQueryMultiVectorOffset(PgturbohybridQueryHeader *query);
static Size PgturbohybridQueryTsQueryOffset(PgturbohybridQueryHeader *query);
static Size PgturbohybridQueryTokenWeightsOffset(PgturbohybridQueryHeader *query);
static Size PgturbohybridQueryTokenMaskOffset(PgturbohybridQueryHeader *query);
static Size PgturbohybridQueryAlignedSize(Size value);
static Size PgturbohybridQuerySizeAdd(Size a, Size b);
static Size PgturbohybridQueryTokenWeightsBytes(PgturbohybridQueryHeader *query);
static Size PgturbohybridQueryTokenMaskBytes(PgturbohybridQueryHeader *query);
static uint16 PgturbohybridQueryParseFusion(text *fusion);
static void PgturbohybridQueryCheckPositiveInt(const char *name, int32 value);
static void PgturbohybridQueryCheckNonNegativeInt(const char *name, int32 value);
static void PgturbohybridQueryRejectTextFallback(void);
static void PgturbohybridQueryValidateInternal(PgturbohybridQueryHeader *query,
											   bool strict);
static PgturbohybridQueryHeader *PgturbohybridQueryConstructorCached(FunctionCallInfo fcinfo);
static void PgturbohybridQueryConstructorStoreCache(FunctionCallInfo fcinfo,
												   PgturbohybridQueryHeader *query);

static PgturbohybridSparseVectorEntry *
PgturbohybridSparseVectorEntries(PgturbohybridSparseVector *sparse)
{
	return (PgturbohybridSparseVectorEntry *)
		((char *) sparse + MAXALIGN(sizeof(PgturbohybridSparseVector)));
}

static void
PgturbohybridSparseVectorValidate(PgturbohybridSparseVector *sparse)
{
	if (sparse == NULL ||
		VARSIZE_ANY(sparse) < MAXALIGN(sizeof(PgturbohybridSparseVector)) ||
		sparse->version != PGTURBOHYBRID_SPARSE_VECTOR_VERSION ||
		VARSIZE_ANY(sparse) !=
		MAXALIGN(sizeof(PgturbohybridSparseVector)) +
		(Size) sparse->count * sizeof(PgturbohybridSparseVectorEntry))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("malformed turbohybrid_sparse_vector payload")));
}

static void
PgturbohybridSparseVectorAppendTerm(StringInfo buf, int32 termId, int16 fieldId)
{
	if (fieldId == PGTURBOHYBRID_SPARSE_VECTOR_FIELD_NONE)
		appendStringInfo(buf, "ths%d", termId);
	else
		appendStringInfo(buf, "thsf%d_%d", fieldId, termId);
}

static Size
PgturbohybridQueryVectorOffset(void)
{
	return MAXALIGN(sizeof(PgturbohybridQueryHeader));
}

static Size
PgturbohybridQueryTsQueryOffset(PgturbohybridQueryHeader *query)
{
	return PgturbohybridQuerySizeAdd(PgturbohybridQueryMultiVectorOffset(query),
									 PgturbohybridQueryAlignedSize(query->multivectorBytes));
}

static Size
PgturbohybridQueryTokenWeightsBytes(PgturbohybridQueryHeader *query)
{
	if ((query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_TOKEN_WEIGHTS) == 0)
		return 0;
	return (Size) query->multivectorCount * sizeof(float4);
}

static Size
PgturbohybridQueryTokenMaskBytes(PgturbohybridQueryHeader *query)
{
	if ((query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_TOKEN_MASK) == 0)
		return 0;
	return (Size) query->multivectorCount * sizeof(bool);
}

static Size
PgturbohybridQueryTokenWeightsOffset(PgturbohybridQueryHeader *query)
{
	return PgturbohybridQuerySizeAdd(PgturbohybridQueryTsQueryOffset(query),
									 PgturbohybridQueryAlignedSize(query->tsqueryBytes));
}

static Size
PgturbohybridQueryTokenMaskOffset(PgturbohybridQueryHeader *query)
{
	return PgturbohybridQuerySizeAdd(PgturbohybridQueryTokenWeightsOffset(query),
									 PgturbohybridQueryAlignedSize(PgturbohybridQueryTokenWeightsBytes(query)));
}

static Size
PgturbohybridQueryMultiVectorOffset(PgturbohybridQueryHeader *query)
{
	return PgturbohybridQuerySizeAdd(PgturbohybridQueryVectorOffset(),
									 PgturbohybridQueryAlignedSize(query->vectorBytes));
}

static Size
PgturbohybridQueryAlignedSize(Size value)
{
	if (value > MaxAllocSize - (MAXIMUM_ALIGNOF - 1))
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("turbohybrid_query payload is too large")));

	return MAXALIGN(value);
}

static Size
PgturbohybridQuerySizeAdd(Size a, Size b)
{
	if (a > MaxAllocSize - b)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("turbohybrid_query payload is too large")));

	return a + b;
}

Vector *
PgturbohybridQueryGetVector(PgturbohybridQueryHeader *query)
{
	PgturbohybridQueryValidateFast(query);

	if (!PgturbohybridQueryHasVector(query))
		return NULL;

	return (Vector *) ((char *) query + PgturbohybridQueryVectorOffset());
}

PgturbohybridMultiVector *
PgturbohybridQueryGetMultiVector(PgturbohybridQueryHeader *query)
{
	PgturbohybridQueryValidateFast(query);

	if (!PgturbohybridQueryHasMultiVector(query))
		return NULL;

	return (PgturbohybridMultiVector *) ((char *) query +
										 PgturbohybridQueryMultiVectorOffset(query));
}

bool
PgturbohybridQueryHasTokenWeights(const PgturbohybridQueryHeader *query)
{
	return query != NULL &&
		(query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_TOKEN_WEIGHTS) != 0;
}

bool
PgturbohybridQueryHasTokenMask(const PgturbohybridQueryHeader *query)
{
	return query != NULL &&
		(query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_TOKEN_MASK) != 0;
}

const float4 *
PgturbohybridQueryGetTokenWeights(PgturbohybridQueryHeader *query)
{
	PgturbohybridQueryValidateFast(query);

	if (!PgturbohybridQueryHasTokenWeights(query))
		return NULL;

	return (const float4 *) ((char *) query +
							 PgturbohybridQueryTokenWeightsOffset(query));
}

const bool *
PgturbohybridQueryGetTokenMask(PgturbohybridQueryHeader *query)
{
	PgturbohybridQueryValidateFast(query);

	if (!PgturbohybridQueryHasTokenMask(query))
		return NULL;

	return (const bool *) ((char *) query +
						   PgturbohybridQueryTokenMaskOffset(query));
}

double
PgturbohybridQueryMultiVectorWeightSum(PgturbohybridQueryHeader *query)
{
	const float4 *weights;
	const bool *mask;
	double		total = 0.0;

	PgturbohybridQueryValidateFast(query);
	if (!PgturbohybridQueryHasMultiVector(query) || query->multivectorCount <= 0)
		return 0.0;

	weights = PgturbohybridQueryGetTokenWeights(query);
	mask = PgturbohybridQueryGetTokenMask(query);
	for (int32 i = 0; i < query->multivectorCount; i++)
	{
		if (mask != NULL && mask[i])
			continue;
		total += weights != NULL ? (double) weights[i] : 1.0;
	}
	return total;
}

TSQuery
PgturbohybridQueryGetTsQuery(PgturbohybridQueryHeader *query)
{
	PgturbohybridQueryValidateFast(query);

	if (!PgturbohybridQueryHasText(query))
		return NULL;

	return (TSQuery) ((char *) query + PgturbohybridQueryTsQueryOffset(query));
}

const char *
PgturbohybridQueryFusionName(uint16 fusion)
{
	switch ((PgturbohybridFusionMode) fusion)
	{
		case PGTURBOHYBRID_FUSION_RRF:
			return "rrf";
		case PGTURBOHYBRID_FUSION_WEIGHTED:
			return "weighted";
		case PGTURBOHYBRID_FUSION_FAST_WEIGHTED:
			return "fast_weighted";
		case PGTURBOHYBRID_FUSION_CALIBRATED:
			return "calibrated";
		case PGTURBOHYBRID_FUSION_DBSF:
			return "dbsf";
	}

	return "unknown";
}

void
PgturbohybridQueryValidate(PgturbohybridQueryHeader *query)
{
	PgturbohybridQueryValidateInternal(query, true);
}

void
PgturbohybridQueryValidateFast(PgturbohybridQueryHeader *query)
{
	PgturbohybridQueryValidateInternal(query, false);
}

static void
PgturbohybridQueryValidateInternal(PgturbohybridQueryHeader *query, bool strict)
{
	Size		actual;
	Size		vectorOffset;
	Size		multivectorOffset;
	Size		tsqueryOffset;
	Size		tokenWeightsOffset;
	Size		tokenMaskOffset;
	Size		expected;

	if (query == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("turbohybrid_query cannot be null")));

	actual = VARSIZE_ANY(query);
	vectorOffset = PgturbohybridQueryVectorOffset();
	if (actual < vectorOffset)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("truncated turbohybrid_query payload")));

	if (query->version != PGTURBOHYBRID_QUERY_VERSION)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("unsupported turbohybrid_query version %u", query->version)));

	if (query->fusion != PGTURBOHYBRID_FUSION_RRF &&
		query->fusion != PGTURBOHYBRID_FUSION_WEIGHTED &&
		query->fusion != PGTURBOHYBRID_FUSION_FAST_WEIGHTED &&
		query->fusion != PGTURBOHYBRID_FUSION_CALIBRATED &&
		query->fusion != PGTURBOHYBRID_FUSION_DBSF)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("invalid turbohybrid_query fusion mode %u", query->fusion)));

	if (query->denseKind != PGTURBOHYBRID_DENSE_QUERY_NONE &&
		query->denseKind != PGTURBOHYBRID_DENSE_QUERY_VECTOR &&
		query->denseKind != PGTURBOHYBRID_DENSE_QUERY_MULTIVECTOR)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("invalid turbohybrid_query dense kind %u", query->denseKind)));

	if (query->vectorBytes < 0 || query->multivectorBytes < 0 ||
		query->tsqueryBytes < 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("invalid turbohybrid_query payload size")));

	if (((query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_VECTOR) != 0) &&
		((query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_MULTIVECTOR) != 0))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("turbohybrid_query cannot contain both vector and multivector payloads")));

	if ((query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_DENSE) == 0 &&
		query->denseKind != PGTURBOHYBRID_DENSE_QUERY_NONE)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("turbohybrid_query dense payload is inconsistent")));

	if ((query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_DENSE) != 0 &&
		query->denseKind == PGTURBOHYBRID_DENSE_QUERY_NONE)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("turbohybrid_query dense payload is inconsistent")));

	if ((query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_VECTOR) == 0 &&
		query->vectorBytes != 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("turbohybrid_query vector payload is inconsistent")));

	if ((query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_VECTOR) != 0)
	{
		Vector	   *vector = (Vector *) ((char *) query + PgturbohybridQueryVectorOffset());
		Size		vectorBytes;

		if (query->denseKind != PGTURBOHYBRID_DENSE_QUERY_VECTOR ||
			(query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_DENSE) == 0)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("turbohybrid_query vector payload is inconsistent")));

		if (PgturbohybridQuerySizeAdd(vectorOffset, query->vectorBytes) > actual)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("truncated turbohybrid_query payload")));

		if (strict)
			PgturbohybridCheckVector(vector);
		else
			PgturbohybridCheckVectorFast(vector);
		vectorBytes = PGTURBOHYBRID_VECTOR_SIZE(vector->dim);
		if (query->vectorBytes != vectorBytes)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("turbohybrid_query vector payload is inconsistent")));
	}

	if ((query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_MULTIVECTOR) == 0 &&
		query->multivectorBytes != 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("turbohybrid_query multivector payload is inconsistent")));

	multivectorOffset = PgturbohybridQueryMultiVectorOffset(query);
	if ((query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_MULTIVECTOR) != 0)
	{
		PgturbohybridMultiVector *mv =
			(PgturbohybridMultiVector *) ((char *) query + multivectorOffset);
		Size		multivectorBytes;

		if (query->denseKind != PGTURBOHYBRID_DENSE_QUERY_MULTIVECTOR ||
			(query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_DENSE) == 0)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("turbohybrid_query multivector payload is inconsistent")));

		if (PgturbohybridQuerySizeAdd(multivectorOffset, query->multivectorBytes) > actual)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("truncated turbohybrid_query payload")));

		PgturbohybridCheckMultiVector(mv);
		multivectorBytes = VARSIZE_ANY(mv);
		if (query->multivectorBytes != multivectorBytes ||
			query->multivectorDim != mv->dim ||
			query->multivectorCount != mv->count)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("turbohybrid_query multivector payload is inconsistent")));
	}
	else if (query->multivectorDim != 0 || query->multivectorCount != 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("turbohybrid_query multivector payload is inconsistent")));

	if (((query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_TOKEN_WEIGHTS) != 0 ||
		 (query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_TOKEN_MASK) != 0) &&
		!PgturbohybridQueryHasMultiVector(query))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("turbohybrid_query token metadata requires multivector_query")));

	if ((query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_TSQUERY) == 0 &&
		query->tsqueryBytes != 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("turbohybrid_query tsquery payload is inconsistent")));

	tsqueryOffset = PgturbohybridQueryTsQueryOffset(query);
	if (PgturbohybridQuerySizeAdd(tsqueryOffset, query->tsqueryBytes) > actual)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("truncated turbohybrid_query payload")));
	tokenWeightsOffset = PgturbohybridQueryTokenWeightsOffset(query);
	if (PgturbohybridQuerySizeAdd(tokenWeightsOffset,
								  PgturbohybridQueryTokenWeightsBytes(query)) > actual)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("truncated turbohybrid_query token weights payload")));
	if ((query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_TOKEN_WEIGHTS) != 0)
	{
		const float4 *weights = (const float4 *) ((char *) query + tokenWeightsOffset);

		for (int32 i = 0; i < query->multivectorCount; i++)
		{
			if (!isfinite((double) weights[i]) || weights[i] < 0.0f)
				ereport(ERROR,
						(errcode(ERRCODE_DATA_EXCEPTION),
						 errmsg("turbohybrid_query token weights must be finite non-negative values")));
		}
	}
	tokenMaskOffset = PgturbohybridQueryTokenMaskOffset(query);
	if (PgturbohybridQuerySizeAdd(tokenMaskOffset,
								  PgturbohybridQueryTokenMaskBytes(query)) > actual)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("truncated turbohybrid_query token mask payload")));
	expected = PgturbohybridQuerySizeAdd(tokenMaskOffset,
										 PgturbohybridQueryAlignedSize(PgturbohybridQueryTokenMaskBytes(query)));
	if (actual != expected)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("malformed turbohybrid_query payload")));
}

Datum
pgturbohybrid_query_in(PG_FUNCTION_ARGS)
{
	char	   *input = PG_GETARG_CSTRING(0);

	(void) input;

	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("text input is not supported for turbohybrid_query"),
			 errhint("Use the turbohybrid_query(...) constructor.")));

	PG_RETURN_NULL();
}

Datum
pgturbohybrid_query_out(PG_FUNCTION_ARGS)
{
	PgturbohybridQueryHeader *query = PG_GETARG_PGTURBOHYBRID_QUERY_P(0);
	StringInfoData buf;

	PgturbohybridQueryValidate(query);

	initStringInfo(&buf);
	appendStringInfo(&buf,
					 "turbohybrid_query(fusion=%s,vector=%s",
					 PgturbohybridQueryFusionName(query->fusion),
					 PgturbohybridQueryHasVector(query) ? "true" : "false");
	if (PgturbohybridQueryHasMultiVector(query))
		appendStringInfoString(&buf, ",multivector=true");
	if (PgturbohybridQueryHasTokenWeights(query))
		appendStringInfoString(&buf, ",query_token_weights=true");
	if (PgturbohybridQueryHasTokenMask(query))
		appendStringInfoString(&buf, ",query_token_mask=true");

	appendStringInfo(&buf,
					 ",tsquery=%s,dense_weight=%g,bm25_weight=%g,alpha=",
					 PgturbohybridQueryHasText(query) ? "true" : "false",
					 query->denseWeight,
					 query->bm25Weight);
	if (query->flags & PGTURBOHYBRID_QUERY_FLAG_ALPHA_IS_SET)
		appendStringInfo(&buf, "%g", query->alpha);
	else
		appendStringInfoString(&buf, "null");

	appendStringInfo(&buf,
					 ",rrf_k=%d,dense_k=%d,bm25_k=%d,final_k=",
					 query->rrfK,
					 query->denseK,
					 query->bm25K);
	if (query->flags & PGTURBOHYBRID_QUERY_FLAG_FINAL_K_IS_SET)
		appendStringInfo(&buf, "%d", query->finalK);
	else
		appendStringInfoString(&buf, "null");

	appendStringInfo(&buf,
					 ",require_bm25_match=%s)",
					 (query->flags & PGTURBOHYBRID_QUERY_FLAG_REQUIRE_BM25_MATCH) ? "true" : "false");

	PG_RETURN_CSTRING(buf.data);
}

Datum
pgturbohybrid_sparse_vector_in(PG_FUNCTION_ARGS)
{
	char	   *input = PG_GETARG_CSTRING(0);

	(void) input;

	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("text input is not supported for turbohybrid_sparse_vector"),
			 errhint("Use turbohybrid_sparse_vector_from_arrays(term_ids, weights).")));

	PG_RETURN_NULL();
}

Datum
pgturbohybrid_sparse_vector_out(PG_FUNCTION_ARGS)
{
	PgturbohybridSparseVector *sparse =
		(PgturbohybridSparseVector *) PG_DETOAST_DATUM(PG_GETARG_DATUM(0));
	StringInfoData buf;

	PgturbohybridSparseVectorValidate(sparse);

	initStringInfo(&buf);
	appendStringInfo(&buf, "turbohybrid_sparse_vector(count=%u)",
					 sparse->count);
	PG_RETURN_CSTRING(buf.data);
}

Datum
pgturbohybrid_sparse_vector_from_arrays(PG_FUNCTION_ARGS)
{
	ArrayType  *termArray = PG_GETARG_ARRAYTYPE_P(0);
	ArrayType  *weightArray = PG_GETARG_ARRAYTYPE_P(1);
	Datum	   *termDatums;
	Datum	   *weightDatums;
	bool	   *termNulls;
	bool	   *weightNulls;
	int			termCount;
	int			weightCount;
	Size		headerSize = MAXALIGN(sizeof(PgturbohybridSparseVector));
	Size		payloadSize;
	Size		totalSize;
	PgturbohybridSparseVector *result;
	PgturbohybridSparseVectorEntry *entries;

	if (ARR_NDIM(termArray) > 1 || ARR_NDIM(weightArray) > 1)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("sparse vector arrays must be one-dimensional")));

	deconstruct_array(termArray, INT4OID, sizeof(int32), true, TYPALIGN_INT,
					  &termDatums, &termNulls, &termCount);
	deconstruct_array(weightArray, FLOAT4OID, sizeof(float4), true, TYPALIGN_INT,
					  &weightDatums, &weightNulls, &weightCount);

	if (termCount != weightCount)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("term_ids and weights must have the same length")));

	if ((Size) termCount >
		(MaxAllocSize - headerSize) / sizeof(PgturbohybridSparseVectorEntry))
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("turbohybrid_sparse_vector payload is too large")));

	payloadSize = (Size) termCount * sizeof(PgturbohybridSparseVectorEntry);
	totalSize = headerSize + payloadSize;
	result = palloc0(totalSize);
	SET_VARSIZE(result, totalSize);
	result->version = PGTURBOHYBRID_SPARSE_VECTOR_VERSION;
	result->count = (uint32) termCount;
	entries = (PgturbohybridSparseVectorEntry *) ((char *) result + headerSize);

	for (int i = 0; i < termCount; i++)
	{
		int32		termId;
		float4		weight;

		if (termNulls[i] || weightNulls[i])
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("sparse vector arrays cannot contain nulls")));

		termId = DatumGetInt32(termDatums[i]);
		weight = DatumGetFloat4(weightDatums[i]);
		if (termId < 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("sparse vector term ids must be non-negative")));
		if (!isfinite((double) weight))
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("sparse vector weights must be finite")));

		entries[i].termId = termId;
		entries[i].weight = weight;
		entries[i].fieldId = PGTURBOHYBRID_SPARSE_VECTOR_FIELD_NONE;
	}

	PG_RETURN_POINTER(result);
}

Datum
pgturbohybrid_sparse_vector_terms(PG_FUNCTION_ARGS)
{
	PgturbohybridSparseVector *sparse =
		(PgturbohybridSparseVector *) PG_DETOAST_DATUM(PG_GETARG_DATUM(0));
	PgturbohybridSparseVectorEntry *entries;
	StringInfoData buf;
	bool		first = true;

	PgturbohybridSparseVectorValidate(sparse);
	entries = PgturbohybridSparseVectorEntries(sparse);

	initStringInfo(&buf);
	for (uint32 i = 0; i < sparse->count; i++)
	{
		int			repeats;

		if (entries[i].weight <= 0.0f)
			continue;
		repeats = (int) ceilf(entries[i].weight);
		repeats = Max(repeats, 1);
		repeats = Min(repeats, 255);
		for (int j = 0; j < repeats; j++)
		{
			if (!first)
				appendStringInfoChar(&buf, ' ');
			PgturbohybridSparseVectorAppendTerm(&buf, entries[i].termId,
												entries[i].fieldId);
			first = false;
		}
	}

	PG_RETURN_TEXT_P(cstring_to_text(buf.data));
}

Datum
pgturbohybrid_sparse_vector_query_terms(PG_FUNCTION_ARGS)
{
	PgturbohybridSparseVector *sparse =
		(PgturbohybridSparseVector *) PG_DETOAST_DATUM(PG_GETARG_DATUM(0));
	PgturbohybridSparseVectorEntry *entries;
	StringInfoData buf;
	bool		first = true;

	PgturbohybridSparseVectorValidate(sparse);
	entries = PgturbohybridSparseVectorEntries(sparse);

	initStringInfo(&buf);
	for (uint32 i = 0; i < sparse->count; i++)
	{
		if (entries[i].weight <= 0.0f)
			continue;
		if (!first)
			appendStringInfoString(&buf, " | ");
		PgturbohybridSparseVectorAppendTerm(&buf, entries[i].termId,
											entries[i].fieldId);
		first = false;
	}

	if (first)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("sparse vector query must contain at least one positive-weight term")));

	PG_RETURN_TEXT_P(cstring_to_text(buf.data));
}

static uint16
PgturbohybridQueryParseFusion(text *fusion)
{
	char	   *name;
	uint16		result;

	if (fusion == NULL)
		return PGTURBOHYBRID_FUSION_RRF;

	name = text_to_cstring(fusion);
	if (pg_strcasecmp(name, "rrf") == 0)
		result = PGTURBOHYBRID_FUSION_RRF;
	else if (pg_strcasecmp(name, "weighted") == 0)
		result = PGTURBOHYBRID_FUSION_WEIGHTED;
	else if (pg_strcasecmp(name, "fast_weighted") == 0)
		result = PGTURBOHYBRID_FUSION_FAST_WEIGHTED;
	else if (pg_strcasecmp(name, "calibrated") == 0)
		result = PGTURBOHYBRID_FUSION_CALIBRATED;
	else if (pg_strcasecmp(name, "dbsf") == 0)
		result = PGTURBOHYBRID_FUSION_DBSF;
	else
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid hybrid fusion mode \"%s\"", name),
				 errdetail("Valid values are \"rrf\", \"weighted\", \"fast_weighted\", \"calibrated\", and \"dbsf\".")));

	pfree(name);
	return result;
}

static void
PgturbohybridQueryCheckPositiveInt(const char *name, int32 value)
{
	if (value <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("%s must be greater than zero", name)));
}

static void
PgturbohybridQueryCheckNonNegativeInt(const char *name, int32 value)
{
	if (value < 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("%s must be greater than or equal to zero", name)));
}

typedef struct PgturbohybridQueryPlanCheck
{
	Node	   *expr;
	Oid			fnOid;
	bool		hasUserVisibleExpr;
	bool		hasIndexOrderByResjunkExpr;
} PgturbohybridQueryPlanCheck;

typedef struct PgturbohybridQueryPlanCheckCache
{
	PlannedStmt *plannedstmt;
	Node	   *expr;
	Oid			fnOid;
	bool		result;
} PgturbohybridQueryPlanCheckCache;

typedef struct PgturbohybridQueryConstructorCache
{
	Node	   *expr;
	uint64		gucGeneration;
	PgturbohybridQueryHeader *query;
} PgturbohybridQueryConstructorCache;

static List *
PgturbohybridQueryDistanceCallArgs(Node *expr, Oid *fnOid)
{
	if (expr == NULL)
		return NIL;

	if (IsA(expr, OpExpr))
	{
		OpExpr	   *op = (OpExpr *) expr;

		*fnOid = op->opfuncid;
		return op->args;
	}

	if (IsA(expr, FuncExpr))
	{
		FuncExpr   *func = (FuncExpr *) expr;

		*fnOid = func->funcid;
		return func->args;
	}

	return NIL;
}

static bool
PgturbohybridQueryDistanceCallMatches(Node *candidate, PgturbohybridQueryPlanCheck *check)
{
	Oid			candidateFnOid = InvalidOid;
	Oid			exprFnOid = InvalidOid;
	List	   *candidateArgs;
	List	   *exprArgs;

	if (candidate == NULL || check->expr == NULL)
		return false;

	if (equal(candidate, check->expr))
		return true;

	candidateArgs = PgturbohybridQueryDistanceCallArgs(candidate, &candidateFnOid);
	exprArgs = PgturbohybridQueryDistanceCallArgs(check->expr, &exprFnOid);

	return OidIsValid(candidateFnOid) &&
		candidateFnOid == check->fnOid &&
		OidIsValid(exprFnOid) &&
		exprFnOid == check->fnOid &&
		equal(candidateArgs, exprArgs);
}

static bool
PgturbohybridQueryPlanHasIndexOrderBy(Plan *plan)
{
	if (plan == NULL)
		return false;

	if (IsA(plan, IndexScan) &&
		((IndexScan *) plan)->indexorderbyorig != NIL)
		return true;

	return PgturbohybridQueryPlanHasIndexOrderBy(plan->lefttree) ||
		PgturbohybridQueryPlanHasIndexOrderBy(plan->righttree);
}

static void
PgturbohybridQueryInspectPlan(Plan *plan, PgturbohybridQueryPlanCheck *check,
							  bool topLevel)
{
	ListCell   *lc;

	if (plan == NULL)
		return;

	foreach(lc, plan->targetlist)
	{
		TargetEntry *tle = (TargetEntry *) lfirst(lc);

		if (tle == NULL || !IsA(tle, TargetEntry) ||
			!PgturbohybridQueryDistanceCallMatches((Node *) tle->expr, check))
			continue;

		if (!tle->resjunk && topLevel)
		{
			check->hasUserVisibleExpr = true;
			continue;
		}

		if (PgturbohybridQueryPlanHasIndexOrderBy(plan))
			check->hasIndexOrderByResjunkExpr = true;
	}

	PgturbohybridQueryInspectPlan(plan->lefttree, check, false);
	PgturbohybridQueryInspectPlan(plan->righttree, check, false);
}

bool
PgturbohybridQueryTextIndexOrderByContext(FunctionCallInfo fcinfo)
{
	PlannedStmt *plannedstmt;
	PgturbohybridQueryPlanCheckCache *cache;
	PgturbohybridQueryPlanCheck check;

	if (fcinfo->flinfo == NULL || fcinfo->flinfo->fn_expr == NULL)
		return false;

	plannedstmt = PgturbohybridCurrentPlannedStmt();
	if (plannedstmt == NULL || plannedstmt->planTree == NULL)
		return false;

	cache = (PgturbohybridQueryPlanCheckCache *) fcinfo->flinfo->fn_extra;
	if (cache != NULL &&
		cache->plannedstmt == plannedstmt &&
		cache->expr == fcinfo->flinfo->fn_expr &&
		cache->fnOid == fcinfo->flinfo->fn_oid)
		return cache->result;

	check.expr = fcinfo->flinfo->fn_expr;
	check.fnOid = fcinfo->flinfo->fn_oid;
	check.hasUserVisibleExpr = false;
	check.hasIndexOrderByResjunkExpr = false;

	PgturbohybridQueryInspectPlan(plannedstmt->planTree, &check, true);
	if (!check.hasIndexOrderByResjunkExpr)
		check.hasIndexOrderByResjunkExpr =
			PgturbohybridQueryPlanHasIndexOrderBy(plannedstmt->planTree);

	if (cache == NULL)
	{
		MemoryContext cacheCtx = fcinfo->flinfo->fn_mcxt != NULL ?
			fcinfo->flinfo->fn_mcxt : TopMemoryContext;
		MemoryContext oldCtx = MemoryContextSwitchTo(cacheCtx);

		cache = palloc0(sizeof(PgturbohybridQueryPlanCheckCache));
		fcinfo->flinfo->fn_extra = cache;
		MemoryContextSwitchTo(oldCtx);
	}

	cache->plannedstmt = plannedstmt;
	cache->expr = fcinfo->flinfo->fn_expr;
	cache->fnOid = fcinfo->flinfo->fn_oid;
	cache->result = check.hasIndexOrderByResjunkExpr && !check.hasUserVisibleExpr;

	return cache->result;
}

static void
PgturbohybridQueryRejectTextFallback(void)
{
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("hybrid text queries require a turbohybrid index scan"),
			 errdetail("The scalar hybrid distance function can only evaluate the vector payload.")));
}

static bool
PgturbohybridQueryConstructorConstArg(Node *node)
{
	if (node == NULL)
		return false;

	if (IsA(node, NamedArgExpr))
		node = (Node *) ((NamedArgExpr *) node)->arg;

	return IsA(node, Const);
}

static bool
PgturbohybridQueryConstructorCacheable(FunctionCallInfo fcinfo)
{
	FuncExpr   *func;
	ListCell   *lc;

	if (fcinfo->flinfo == NULL || fcinfo->flinfo->fn_expr == NULL ||
		!IsA(fcinfo->flinfo->fn_expr, FuncExpr))
		return false;

	func = (FuncExpr *) fcinfo->flinfo->fn_expr;
	foreach(lc, func->args)
	{
		if (!PgturbohybridQueryConstructorConstArg((Node *) lfirst(lc)))
			return false;
	}

	return true;
}

static PgturbohybridQueryHeader *
PgturbohybridQueryConstructorCached(FunctionCallInfo fcinfo)
{
	PgturbohybridQueryConstructorCache *cache;

	if (!PgturbohybridQueryConstructorCacheable(fcinfo))
		return NULL;

	cache = (PgturbohybridQueryConstructorCache *) fcinfo->flinfo->fn_extra;
	if (cache == NULL ||
		cache->expr != fcinfo->flinfo->fn_expr ||
		cache->gucGeneration != pgturbohybrid_guc_generation ||
		cache->query == NULL)
		return NULL;

	return cache->query;
}

static void
PgturbohybridQueryConstructorStoreCache(FunctionCallInfo fcinfo,
										PgturbohybridQueryHeader *query)
{
	PgturbohybridQueryConstructorCache *cache;
	MemoryContext cacheCtx;
	MemoryContext oldCtx;
	Size		size;

	if (!PgturbohybridQueryConstructorCacheable(fcinfo))
		return;

	cacheCtx = fcinfo->flinfo->fn_mcxt != NULL ?
		fcinfo->flinfo->fn_mcxt : TopMemoryContext;
	oldCtx = MemoryContextSwitchTo(cacheCtx);

	cache = palloc0(sizeof(PgturbohybridQueryConstructorCache));
	size = VARSIZE_ANY(query);
	cache->query = palloc(size);
	memcpy(cache->query, query, size);
	cache->expr = fcinfo->flinfo->fn_expr;
	cache->gucGeneration = pgturbohybrid_guc_generation;
	fcinfo->flinfo->fn_extra = cache;

	MemoryContextSwitchTo(oldCtx);
}

Datum
pgturbohybrid_query_constructor(PG_FUNCTION_ARGS)
{
	PgturbohybridQueryHeader *cached;
	struct varlena *vectorDatum = NULL;
	struct varlena *multivectorDatum = NULL;
	struct varlena *tsqueryDatum = NULL;
	float4	   *tokenWeights = NULL;
	bool	   *tokenMask = NULL;
	int32		vectorBytes = 0;
	int32		multivectorBytes = 0;
	int32		tsqueryBytes = 0;
	int32		multivectorDim = 0;
	int32		multivectorCount = 0;
	Size		vectorSize = 0;
	Size		multivectorSize = 0;
	Size		tsquerySize = 0;
	Size		tokenWeightsBytes = 0;
	Size		tokenMaskBytes = 0;
	uint16		flags = 0;
	uint16		denseKind = PGTURBOHYBRID_DENSE_QUERY_NONE;
	uint16		fusion;
	float8		denseWeight;
	float8		bm25Weight;
	float8		alpha = 0.0;
	int32		rrfK;
	int32		denseK;
	int32		bm25K;
	int32		finalK = 0;
	bool		requireBm25Match;
	Size		totalSize;
	PgturbohybridQueryHeader *result;

	cached = PgturbohybridQueryConstructorCached(fcinfo);
	if (cached != NULL)
		PG_RETURN_PGTURBOHYBRID_QUERY_P(cached);

	if (!PG_ARGISNULL(0))
	{
		vectorDatum = PG_DETOAST_DATUM_COPY(PG_GETARG_DATUM(0));
		vectorSize = VARSIZE_ANY(vectorDatum);
		if (vectorSize > PG_INT32_MAX)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("turbohybrid_query vector payload is too large")));
		vectorBytes = (int32) vectorSize;
		flags |= PGTURBOHYBRID_QUERY_FLAG_HAS_VECTOR | PGTURBOHYBRID_QUERY_FLAG_HAS_DENSE;
		denseKind = PGTURBOHYBRID_DENSE_QUERY_VECTOR;
	}

	if (!PG_ARGISNULL(1))
	{
		tsqueryDatum = PG_DETOAST_DATUM_COPY(PG_GETARG_DATUM(1));
		tsquerySize = VARSIZE_ANY(tsqueryDatum);
		if (tsquerySize > PG_INT32_MAX)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("turbohybrid_query tsquery payload is too large")));
		tsqueryBytes = (int32) tsquerySize;
		flags |= PGTURBOHYBRID_QUERY_FLAG_HAS_TSQUERY;
	}

	/*
	 * multivector_query is the optional 12th argument (index 11).  Guard the
	 * read with PG_NARGS() so an extension still installed at the older 11-arg
	 * signature (no multivector_query) does not make us read past the argument
	 * array -- otherwise the out-of-bounds PG_ARGISNULL(11) reads garbage and
	 * spuriously raises the "both vector and multivector" error for a plain
	 * vector_query.  Keeps a new library backward compatible with a not-yet
	 * re-created extension.
	 */
	if (PG_NARGS() > 11 && !PG_ARGISNULL(11))
	{
		PgturbohybridMultiVector *mv;

		if ((flags & PGTURBOHYBRID_QUERY_FLAG_HAS_VECTOR) != 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("turbohybrid_query cannot contain both vector_query and multivector_query")));

		multivectorDatum = PG_DETOAST_DATUM_COPY(PG_GETARG_DATUM(11));
		mv = (PgturbohybridMultiVector *) multivectorDatum;
		PgturbohybridCheckMultiVector(mv);
		multivectorSize = VARSIZE_ANY(multivectorDatum);
		if (multivectorSize > PG_INT32_MAX)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("turbohybrid_query multivector payload is too large")));
		multivectorBytes = (int32) multivectorSize;
		multivectorDim = mv->dim;
		multivectorCount = mv->count;
		flags |= PGTURBOHYBRID_QUERY_FLAG_HAS_MULTIVECTOR | PGTURBOHYBRID_QUERY_FLAG_HAS_DENSE;
		denseKind = PGTURBOHYBRID_DENSE_QUERY_MULTIVECTOR;
	}

	if (PG_NARGS() > 12 && !PG_ARGISNULL(12))
	{
		ArrayType  *weightArray = PG_GETARG_ARRAYTYPE_P(12);
		Datum	   *datums;
		bool	   *nulls;
		int			count;

		if (multivectorCount <= 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("query_token_weights requires multivector_query")));
		if (ARR_NDIM(weightArray) > 1)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("query_token_weights must be a one-dimensional real array")));
		deconstruct_array(weightArray, FLOAT4OID, sizeof(float4), true,
						  TYPALIGN_INT, &datums, &nulls, &count);
		if (count != multivectorCount)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("query_token_weights length %d does not match multivector query token count %d",
							count, multivectorCount)));
		tokenWeightsBytes = (Size) count * sizeof(float4);
		tokenWeights = palloc(tokenWeightsBytes);
		for (int i = 0; i < count; i++)
		{
			float4		weight;

			if (nulls[i])
				ereport(ERROR,
						(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
						 errmsg("query_token_weights cannot contain nulls")));
			weight = DatumGetFloat4(datums[i]);
			if (!isfinite((double) weight) || weight < 0.0f)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("query_token_weights must contain finite non-negative values")));
			tokenWeights[i] = weight;
		}
		flags |= PGTURBOHYBRID_QUERY_FLAG_HAS_TOKEN_WEIGHTS;
	}

	if (PG_NARGS() > 13 && !PG_ARGISNULL(13))
	{
		ArrayType  *maskArray = PG_GETARG_ARRAYTYPE_P(13);
		Datum	   *datums;
		bool	   *nulls;
		int			count;

		if (multivectorCount <= 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("query_token_mask requires multivector_query")));
		if (ARR_NDIM(maskArray) > 1)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("query_token_mask must be a one-dimensional boolean array")));
		deconstruct_array(maskArray, BOOLOID, sizeof(bool), true,
						  TYPALIGN_CHAR, &datums, &nulls, &count);
		if (count != multivectorCount)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("query_token_mask length %d does not match multivector query token count %d",
							count, multivectorCount)));
		tokenMaskBytes = (Size) count * sizeof(bool);
		tokenMask = palloc(tokenMaskBytes);
		for (int i = 0; i < count; i++)
		{
			if (nulls[i])
				ereport(ERROR,
						(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
						 errmsg("query_token_mask cannot contain nulls")));
			tokenMask[i] = DatumGetBool(datums[i]);
		}
		flags |= PGTURBOHYBRID_QUERY_FLAG_HAS_TOKEN_MASK;
	}

	if ((flags & (PGTURBOHYBRID_QUERY_FLAG_HAS_DENSE | PGTURBOHYBRID_QUERY_FLAG_HAS_TSQUERY)) == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("turbohybrid_query requires a vector_query, multivector_query, or text_query")));

	fusion = PgturbohybridQueryParseFusion(PG_ARGISNULL(2) ? NULL : PG_GETARG_TEXT_PP(2));

	if (PG_ARGISNULL(3))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("dense_weight cannot be null")));
	denseWeight = PG_GETARG_FLOAT8(3);
	if (denseWeight < 0 || isnan(denseWeight) || isinf(denseWeight))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("dense_weight must be a finite non-negative value")));

	if (PG_ARGISNULL(4))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("bm25_weight cannot be null")));
	bm25Weight = PG_GETARG_FLOAT8(4);
	if (bm25Weight < 0 || isnan(bm25Weight) || isinf(bm25Weight))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("bm25_weight must be a finite non-negative value")));

	if (!PG_ARGISNULL(5))
	{
		alpha = PG_GETARG_FLOAT8(5);
		if (alpha < 0.0 || alpha > 1.0 || isnan(alpha) || isinf(alpha))
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("alpha must be between 0 and 1")));
		flags |= PGTURBOHYBRID_QUERY_FLAG_ALPHA_IS_SET;
	}

	if (PG_ARGISNULL(6))
	{
		rrfK = pgturbohybrid_default_rrf_k;
		flags |= PGTURBOHYBRID_QUERY_FLAG_RRF_K_DEFAULTED;
	}
	else
		rrfK = PG_GETARG_INT32(6);
	PgturbohybridQueryCheckPositiveInt("rrf_k", rrfK);

	if (PG_ARGISNULL(7))
	{
		denseK = pgturbohybrid_default_dense_k;
		flags |= PGTURBOHYBRID_QUERY_FLAG_DENSE_K_DEFAULTED;
	}
	else
		denseK = PG_GETARG_INT32(7);
	PgturbohybridQueryCheckNonNegativeInt("dense_k", denseK);

	if (PG_ARGISNULL(8))
	{
		bm25K = pgturbohybrid_default_bm25_k;
		flags |= PGTURBOHYBRID_QUERY_FLAG_BM25_K_DEFAULTED;
	}
	else
		bm25K = PG_GETARG_INT32(8);
	PgturbohybridQueryCheckNonNegativeInt("bm25_k", bm25K);

	if (!PG_ARGISNULL(9))
	{
		finalK = PG_GETARG_INT32(9);
		PgturbohybridQueryCheckPositiveInt("final_k", finalK);
		flags |= PGTURBOHYBRID_QUERY_FLAG_FINAL_K_IS_SET;
	}

	if (PG_ARGISNULL(10))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("require_bm25_match cannot be null")));
	requireBm25Match = PG_GETARG_BOOL(10);
	if (requireBm25Match)
		flags |= PGTURBOHYBRID_QUERY_FLAG_REQUIRE_BM25_MATCH;

	totalSize = PgturbohybridQuerySizeAdd(PgturbohybridQueryVectorOffset(),
										  PgturbohybridQueryAlignedSize(vectorBytes));
	totalSize = PgturbohybridQuerySizeAdd(totalSize,
										  PgturbohybridQueryAlignedSize(multivectorBytes));
	totalSize = PgturbohybridQuerySizeAdd(totalSize,
										  PgturbohybridQueryAlignedSize(tsqueryBytes));
	totalSize = PgturbohybridQuerySizeAdd(totalSize,
										  PgturbohybridQueryAlignedSize(tokenWeightsBytes));
	totalSize = PgturbohybridQuerySizeAdd(totalSize,
										  PgturbohybridQueryAlignedSize(tokenMaskBytes));
	result = palloc0(totalSize);
	SET_VARSIZE(result, totalSize);
	result->version = PGTURBOHYBRID_QUERY_VERSION;
	result->flags = flags;
	result->fusion = fusion;
	result->denseWeight = denseWeight;
	result->bm25Weight = bm25Weight;
	result->alpha = alpha;
	result->rrfK = rrfK;
	result->denseK = denseK;
	result->bm25K = bm25K;
	result->finalK = finalK;
	result->denseKind = denseKind;
	result->vectorBytes = vectorBytes;
	result->multivectorBytes = multivectorBytes;
	result->tsqueryBytes = tsqueryBytes;
	result->multivectorDim = multivectorDim;
	result->multivectorCount = multivectorCount;

	if (vectorDatum != NULL)
	{
		memcpy((char *) result + PgturbohybridQueryVectorOffset(), vectorDatum, vectorBytes);
		pfree(vectorDatum);
	}
	if (multivectorDatum != NULL)
	{
		memcpy((char *) result + PgturbohybridQueryMultiVectorOffset(result),
			   multivectorDatum, multivectorBytes);
		pfree(multivectorDatum);
	}
	if (tsqueryDatum != NULL)
	{
		memcpy((char *) result + PgturbohybridQueryTsQueryOffset(result), tsqueryDatum, tsqueryBytes);
		pfree(tsqueryDatum);
	}
	if (tokenWeights != NULL)
	{
		memcpy((char *) result + PgturbohybridQueryTokenWeightsOffset(result),
			   tokenWeights, tokenWeightsBytes);
		pfree(tokenWeights);
	}
	if (tokenMask != NULL)
	{
		memcpy((char *) result + PgturbohybridQueryTokenMaskOffset(result),
			   tokenMask, tokenMaskBytes);
		pfree(tokenMask);
	}

	PgturbohybridQueryValidate(result);
	PgturbohybridQueryConstructorStoreCache(fcinfo, result);

	PG_RETURN_PGTURBOHYBRID_QUERY_P(result);
}

Datum
pgturbohybrid_l2_distance(PG_FUNCTION_ARGS)
{
	PgturbohybridQueryHeader *query = PG_GETARG_PGTURBOHYBRID_QUERY_P(1);
	Vector	   *vectorQuery;
	Vector	   *value;

	if (PgturbohybridQueryTextIndexOrderByContext(fcinfo))
	{
		PgturbohybridQueryValidateFast(query);
		if (PgturbohybridQueryGetTsQuery(query) != NULL)
			PG_RETURN_FLOAT8(0.0);

		vectorQuery = PgturbohybridQueryGetVector(query);
		if (vectorQuery == NULL)
			PG_RETURN_FLOAT8(0.0);

		value = (Vector *) PG_DETOAST_DATUM(PG_GETARG_DATUM(0));
		PgturbohybridCheckVectorFast(value);
		PG_RETURN_FLOAT8(PgturbohybridL2Distance(value, vectorQuery));
	}

	value = PG_GETARG_PGTURBOHYBRID_VECTOR_P(0);
	PgturbohybridQueryValidate(query);
	if (PgturbohybridQueryGetTsQuery(query) != NULL)
	{
		if (!PgturbohybridQueryTextIndexOrderByContext(fcinfo))
			PgturbohybridQueryRejectTextFallback();
		PG_RETURN_FLOAT8(0.0);
	}
	vectorQuery = PgturbohybridQueryGetVector(query);

	if (vectorQuery == NULL)
		PG_RETURN_FLOAT8(0.0);

	PG_RETURN_FLOAT8(PgturbohybridL2Distance(value, vectorQuery));
}

Datum
pgturbohybrid_negative_inner_product(PG_FUNCTION_ARGS)
{
	PgturbohybridQueryHeader *query = PG_GETARG_PGTURBOHYBRID_QUERY_P(1);
	Vector	   *vectorQuery;
	Vector	   *value;

	if (PgturbohybridQueryTextIndexOrderByContext(fcinfo))
	{
		PgturbohybridQueryValidateFast(query);
		if (PgturbohybridQueryGetTsQuery(query) != NULL)
			PG_RETURN_FLOAT8(0.0);

		vectorQuery = PgturbohybridQueryGetVector(query);
		if (vectorQuery == NULL)
			PG_RETURN_FLOAT8(0.0);

		value = (Vector *) PG_DETOAST_DATUM(PG_GETARG_DATUM(0));
		PgturbohybridCheckVectorFast(value);
		PG_RETURN_FLOAT8(PgturbohybridNegativeInnerProduct(value, vectorQuery));
	}

	value = PG_GETARG_PGTURBOHYBRID_VECTOR_P(0);
	PgturbohybridQueryValidate(query);
	if (PgturbohybridQueryGetTsQuery(query) != NULL)
	{
		if (!PgturbohybridQueryTextIndexOrderByContext(fcinfo))
			PgturbohybridQueryRejectTextFallback();
		PG_RETURN_FLOAT8(0.0);
	}
	vectorQuery = PgturbohybridQueryGetVector(query);

	if (vectorQuery == NULL)
		PG_RETURN_FLOAT8(0.0);

	PG_RETURN_FLOAT8(PgturbohybridNegativeInnerProduct(value, vectorQuery));
}

Datum
pgturbohybrid_cosine_distance(PG_FUNCTION_ARGS)
{
	PgturbohybridQueryHeader *query = PG_GETARG_PGTURBOHYBRID_QUERY_P(1);
	Vector	   *vectorQuery;
	Vector	   *value;

	if (PgturbohybridQueryTextIndexOrderByContext(fcinfo))
	{
		PgturbohybridQueryValidateFast(query);
		if (PgturbohybridQueryGetTsQuery(query) != NULL)
			PG_RETURN_FLOAT8(0.0);

		vectorQuery = PgturbohybridQueryGetVector(query);
		if (vectorQuery == NULL)
			PG_RETURN_FLOAT8(0.0);

		value = (Vector *) PG_DETOAST_DATUM(PG_GETARG_DATUM(0));
		PgturbohybridCheckVectorFast(value);
		PG_RETURN_FLOAT8(PgturbohybridCosineDistance(value, vectorQuery));
	}

	value = PG_GETARG_PGTURBOHYBRID_VECTOR_P(0);
	PgturbohybridQueryValidate(query);
	if (PgturbohybridQueryGetTsQuery(query) != NULL)
	{
		if (!PgturbohybridQueryTextIndexOrderByContext(fcinfo))
			PgturbohybridQueryRejectTextFallback();
		PG_RETURN_FLOAT8(0.0);
	}
	vectorQuery = PgturbohybridQueryGetVector(query);

	if (vectorQuery == NULL)
		PG_RETURN_FLOAT8(0.0);

	PG_RETURN_FLOAT8(PgturbohybridCosineDistance(value, vectorQuery));
}

Datum
pgturbohybrid_distance(PG_FUNCTION_ARGS)
{
	return pgturbohybrid_cosine_distance(fcinfo);
}
