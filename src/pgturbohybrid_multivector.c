#include "postgres.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>

#include "access/htup_details.h"
#include "catalog/pg_extension.h"
#include "catalog/pg_type.h"
#include "commands/extension.h"
#include "fmgr.h"
#include "lib/stringinfo.h"
#include "libpq/pqformat.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/syscache.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

#include "pgturbohybrid.h"
#include "pgturbohybrid_multivector.h"
#include "pgturbohybrid_query.h"

#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && \
	(defined(__AVX2__) || (defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))))
#define PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX2 1
#include <immintrin.h>
#else
#define PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX2 0
#endif

#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX2 && !defined(__AVX2__) && \
	(defined(__GNUC__) || defined(__clang__))
#define PGTURBOHYBRID_MULTIVECTOR_AVX2_TARGET __attribute__((target("avx2")))
#else
#define PGTURBOHYBRID_MULTIVECTOR_AVX2_TARGET
#endif

#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && \
	(defined(__AVX512F__) || (defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))))
#define PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX512F 1
#else
#define PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX512F 0
#endif

#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX512F && !defined(__AVX512F__) && \
	(defined(__GNUC__) || defined(__clang__))
#define PGTURBOHYBRID_MULTIVECTOR_AVX512F_TARGET __attribute__((target("avx512f")))
#else
#define PGTURBOHYBRID_MULTIVECTOR_AVX512F_TARGET
#endif

#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && (defined(__aarch64__) || defined(_M_ARM64))
#define PGTURBOHYBRID_MULTIVECTOR_COMPILE_NEON 1
#include <arm_neon.h>
#else
#define PGTURBOHYBRID_MULTIVECTOR_COMPILE_NEON 0
#endif

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_in);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_out);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_recv);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_send);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_constructor);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_from_float4);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_dims);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_count);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_subvector);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_to_vector_array);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_maxsim);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_maxsim_scalar);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_maxsim_blocked_scalar);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_maxsim_distance);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_query_distance);

static Oid	pgturbohybrid_multivector_type_oid = InvalidOid;

#define PGTURBOHYBRID_MULTIVECTOR_BINARY_VERSION 1
#define PGTURBOHYBRID_MULTIVECTOR_BLOCKED_SCALAR_Q 8

static Oid PgturbohybridExtensionSchema(Oid extensionOid);
static void PgturbohybridCheckMultiVectorHeader(int32 count, int32 dim);
static void TqMvSkipSpaces(const char **cursor);
static bool TqMvConsumeLiteral(const char **cursor, const char *literal);
static int32 TqMvParseInt32(const char **cursor, const char *fieldName);
static float4 TqMvParseFloat4(const char **cursor);
static void TqMvExpectChar(const char **cursor, char expected);
static void PgturbohybridMultiVectorRejectTextFallback(void);
static Vector *PgturbohybridMultiVectorSubvectorCopy(const PgturbohybridMultiVector *mv,
													 int32 ordinal);

typedef double (*TqDotProductF32Func) (const float *a, const float *b, int32 dim);
typedef double (*TqMultiVectorMaxSimFunc) (const PgturbohybridMultiVector *query,
										   const PgturbohybridMultiVector *doc);

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

Size
PgturbohybridMultiVectorSubvectorSize(const PgturbohybridMultiVector *mv)
{
	return PGTURBOHYBRID_VECTOR_SIZE(mv->dim);
}

void
PgturbohybridMultiVectorCopySubvectorToVector(const PgturbohybridMultiVector *mv,
											  int32 ordinal, Vector *dst)
{
	Size		vectorSize;

	vectorSize = PgturbohybridMultiVectorSubvectorSize(mv);
	SET_VARSIZE(dst, vectorSize);
	dst->dim = (int16) mv->dim;
	dst->unused = 0;
	memcpy(dst->x, PgturbohybridMultiVectorValues(mv, ordinal),
		   sizeof(float) * (Size) mv->dim);
}

static Vector *
PgturbohybridMultiVectorSubvectorCopy(const PgturbohybridMultiVector *mv,
									  int32 ordinal)
{
	Vector	   *result;
	Size		vectorSize;

	PgturbohybridCheckMultiVector(mv);
	if (ordinal < 0 || ordinal >= mv->count)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("multivector subvector ordinal %d is out of range",
						ordinal + 1)));

	vectorSize = PGTURBOHYBRID_VECTOR_SIZE(mv->dim);
	result = (Vector *) palloc0(vectorSize);
	PgturbohybridMultiVectorCopySubvectorToVector(mv, ordinal, result);
	PgturbohybridCheckVector(result);

	return result;
}

double
TqDotProductF32Scalar(const float *a, const float *b, int32 dim)
{
	double		result = 0.0;

	for (int32 i = 0; i < dim; i++)
		result += (double) a[i] * (double) b[i];

	return result;
}

#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX2
static bool
TqMultiVectorAvx2Available(void)
{
#if defined(__AVX2__)
	return true;
#elif defined(__GNUC__) || defined(__clang__)
	return __builtin_cpu_supports("avx2");
#else
	return false;
#endif
}

static double PGTURBOHYBRID_MULTIVECTOR_AVX2_TARGET
TqDotProductF32Avx2(const float *a, const float *b, int32 dim)
{
	__m256		acc0 = _mm256_setzero_ps();
	__m256		acc1 = _mm256_setzero_ps();
	__m256		acc2 = _mm256_setzero_ps();
	__m256		acc3 = _mm256_setzero_ps();
	float		tmp[8];
	float		result;
	int32		i = 0;

	for (; i + 32 <= dim; i += 32)
	{
		acc0 = _mm256_add_ps(acc0,
							 _mm256_mul_ps(_mm256_loadu_ps(a + i),
										   _mm256_loadu_ps(b + i)));
		acc1 = _mm256_add_ps(acc1,
							 _mm256_mul_ps(_mm256_loadu_ps(a + i + 8),
										   _mm256_loadu_ps(b + i + 8)));
		acc2 = _mm256_add_ps(acc2,
							 _mm256_mul_ps(_mm256_loadu_ps(a + i + 16),
										   _mm256_loadu_ps(b + i + 16)));
		acc3 = _mm256_add_ps(acc3,
							 _mm256_mul_ps(_mm256_loadu_ps(a + i + 24),
										   _mm256_loadu_ps(b + i + 24)));
	}

	acc0 = _mm256_add_ps(_mm256_add_ps(acc0, acc1),
						 _mm256_add_ps(acc2, acc3));
	_mm256_storeu_ps(tmp, acc0);
	result = 0.0f;
	for (int lane = 0; lane < 8; lane++)
		result += tmp[lane];

	for (; i < dim; i++)
		result += a[i] * b[i];

	return (double) result;
}

static void PGTURBOHYBRID_MULTIVECTOR_AVX2_TARGET
TqDotProductF32BlockAvx2(const float *queryValues, const float *docValues,
						 int32 dim, int32 blockCount, double *dots)
{
	__m256		acc[PGTURBOHYBRID_MULTIVECTOR_BLOCKED_SCALAR_Q];
	float		tmp[8];
	int32		i = 0;

	for (int32 qi = 0; qi < blockCount; qi++)
		acc[qi] = _mm256_setzero_ps();

	for (; i + 8 <= dim; i += 8)
	{
		__m256		dv = _mm256_loadu_ps(docValues + i);

		for (int32 qi = 0; qi < blockCount; qi++)
		{
			const float *qv = queryValues + ((Size) qi * (Size) dim);

			acc[qi] = _mm256_add_ps(acc[qi],
									_mm256_mul_ps(_mm256_loadu_ps(qv + i), dv));
		}
	}

	for (int32 qi = 0; qi < blockCount; qi++)
	{
		const float *qv = queryValues + ((Size) qi * (Size) dim);
		float		result = 0.0f;

		_mm256_storeu_ps(tmp, acc[qi]);
		for (int lane = 0; lane < 8; lane++)
			result += tmp[lane];
		for (int32 tail = i; tail < dim; tail++)
			result += qv[tail] * docValues[tail];
		dots[qi] = (double) result;
	}
}
#endif

#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX512F
static bool
TqMultiVectorAvx512fAvailable(void)
{
#if defined(__AVX512F__)
	return true;
#elif defined(__GNUC__) || defined(__clang__)
	return __builtin_cpu_supports("avx512f");
#else
	return false;
#endif
}

static double PGTURBOHYBRID_MULTIVECTOR_AVX512F_TARGET
TqDotProductF32Avx512f(const float *a, const float *b, int32 dim)
{
	__m512		acc0 = _mm512_setzero_ps();
	__m512		acc1 = _mm512_setzero_ps();
	__m512		acc2 = _mm512_setzero_ps();
	__m512		acc3 = _mm512_setzero_ps();
	float		tmp[16];
	float		result;
	int32		i = 0;

	for (; i + 64 <= dim; i += 64)
	{
		acc0 = _mm512_add_ps(acc0,
							 _mm512_mul_ps(_mm512_loadu_ps(a + i),
										   _mm512_loadu_ps(b + i)));
		acc1 = _mm512_add_ps(acc1,
							 _mm512_mul_ps(_mm512_loadu_ps(a + i + 16),
										   _mm512_loadu_ps(b + i + 16)));
		acc2 = _mm512_add_ps(acc2,
							 _mm512_mul_ps(_mm512_loadu_ps(a + i + 32),
										   _mm512_loadu_ps(b + i + 32)));
		acc3 = _mm512_add_ps(acc3,
							 _mm512_mul_ps(_mm512_loadu_ps(a + i + 48),
										   _mm512_loadu_ps(b + i + 48)));
	}

	acc0 = _mm512_add_ps(_mm512_add_ps(acc0, acc1),
						 _mm512_add_ps(acc2, acc3));
	_mm512_storeu_ps(tmp, acc0);
	result = 0.0f;
	for (int lane = 0; lane < 16; lane++)
		result += tmp[lane];

	for (; i < dim; i++)
		result += a[i] * b[i];

	return (double) result;
}

static void PGTURBOHYBRID_MULTIVECTOR_AVX512F_TARGET
TqDotProductF32BlockAvx512f(const float *queryValues, const float *docValues,
							int32 dim, int32 blockCount, double *dots)
{
	__m512		acc[PGTURBOHYBRID_MULTIVECTOR_BLOCKED_SCALAR_Q];
	float		tmp[16];
	int32		i = 0;

	for (int32 qi = 0; qi < blockCount; qi++)
		acc[qi] = _mm512_setzero_ps();

	for (; i + 16 <= dim; i += 16)
	{
		__m512		dv = _mm512_loadu_ps(docValues + i);

		for (int32 qi = 0; qi < blockCount; qi++)
		{
			const float *qv = queryValues + ((Size) qi * (Size) dim);

			acc[qi] = _mm512_add_ps(acc[qi],
									_mm512_mul_ps(_mm512_loadu_ps(qv + i), dv));
		}
	}

	for (int32 qi = 0; qi < blockCount; qi++)
	{
		const float *qv = queryValues + ((Size) qi * (Size) dim);
		float		result = 0.0f;

		_mm512_storeu_ps(tmp, acc[qi]);
		for (int lane = 0; lane < 16; lane++)
			result += tmp[lane];
		for (int32 tail = i; tail < dim; tail++)
			result += qv[tail] * docValues[tail];
		dots[qi] = (double) result;
	}
}
#endif

#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_NEON
static double
TqDotProductF32Neon(const float *a, const float *b, int32 dim)
{
	float32x4_t acc0 = vdupq_n_f32(0);
	float32x4_t acc1 = vdupq_n_f32(0);
	float32x4_t acc2 = vdupq_n_f32(0);
	float32x4_t acc3 = vdupq_n_f32(0);
	float		result;
	int32		i = 0;

	for (; i + 16 <= dim; i += 16)
	{
		acc0 = vaddq_f32(acc0, vmulq_f32(vld1q_f32(a + i),
										  vld1q_f32(b + i)));
		acc1 = vaddq_f32(acc1, vmulq_f32(vld1q_f32(a + i + 4),
										  vld1q_f32(b + i + 4)));
		acc2 = vaddq_f32(acc2, vmulq_f32(vld1q_f32(a + i + 8),
										  vld1q_f32(b + i + 8)));
		acc3 = vaddq_f32(acc3, vmulq_f32(vld1q_f32(a + i + 12),
										  vld1q_f32(b + i + 12)));
	}

	acc0 = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
	result = vaddvq_f32(acc0);

	for (; i < dim; i++)
		result += a[i] * b[i];

	return (double) result;
}

static void
TqDotProductF32BlockNeon(const float *queryValues, const float *docValues,
						 int32 dim, int32 blockCount, double *dots)
{
	float32x4_t acc[PGTURBOHYBRID_MULTIVECTOR_BLOCKED_SCALAR_Q];
	int32		i = 0;

	for (int32 qi = 0; qi < blockCount; qi++)
		acc[qi] = vdupq_n_f32(0);

	for (; i + 4 <= dim; i += 4)
	{
		float32x4_t dv = vld1q_f32(docValues + i);

		for (int32 qi = 0; qi < blockCount; qi++)
		{
			const float *qv = queryValues + ((Size) qi * (Size) dim);

			acc[qi] = vaddq_f32(acc[qi], vmulq_f32(vld1q_f32(qv + i), dv));
		}
	}

	for (int32 qi = 0; qi < blockCount; qi++)
	{
		const float *qv = queryValues + ((Size) qi * (Size) dim);
		float		result = vaddvq_f32(acc[qi]);

		for (int32 tail = i; tail < dim; tail++)
			result += qv[tail] * docValues[tail];
		dots[qi] = (double) result;
	}
}
#endif

static double
TqMultiVectorMaxSimWithDot(const PgturbohybridMultiVector *query,
						   const PgturbohybridMultiVector *doc,
						   TqDotProductF32Func dotProduct)
{
	double		score = 0.0;

	PgturbohybridCheckSameMultiVectorDims(query, doc);

	for (int32 qi = 0; qi < query->count; qi++)
	{
		const float *qv = query->values + ((Size) qi * (Size) query->dim);
		double		best = -INFINITY;

		for (int32 di = 0; di < doc->count; di++)
		{
			const float *dv = doc->values + ((Size) di * (Size) doc->dim);
			double		dot = dotProduct(qv, dv, query->dim);

			if (dot > best)
				best = dot;
		}

		score += best;
	}

	return score;
}

static double
TqMultiVectorMaxSimBlockedWithDot(const PgturbohybridMultiVector *query,
								  const PgturbohybridMultiVector *doc,
								  TqDotProductF32Func dotProduct)
{
	double		score = 0.0;

	PgturbohybridCheckSameMultiVectorDims(query, doc);

	for (int32 qb = 0; qb < query->count;
		 qb += PGTURBOHYBRID_MULTIVECTOR_BLOCKED_SCALAR_Q)
	{
		int32		blockCount =
			Min(query->count - qb, PGTURBOHYBRID_MULTIVECTOR_BLOCKED_SCALAR_Q);
		double		best[PGTURBOHYBRID_MULTIVECTOR_BLOCKED_SCALAR_Q];

		for (int32 qi = 0; qi < blockCount; qi++)
			best[qi] = -INFINITY;

		for (int32 di = 0; di < doc->count; di++)
		{
			const float *dv = doc->values + ((Size) di * (Size) doc->dim);

			for (int32 qi = 0; qi < blockCount; qi++)
			{
				const float *qv =
					query->values + ((Size) (qb + qi) * (Size) query->dim);
				double		dot = dotProduct(qv, dv, query->dim);

				if (dot > best[qi])
					best[qi] = dot;
			}
		}

		for (int32 qi = 0; qi < blockCount; qi++)
			score += best[qi];
	}

	return score;
}

double
TqMultiVectorMaxSimScalar(const PgturbohybridMultiVector *query,
						  const PgturbohybridMultiVector *doc)
{
	return TqMultiVectorMaxSimWithDot(query, doc, TqDotProductF32Scalar);
}

double
TqMultiVectorMaxSimBlockedScalar(const PgturbohybridMultiVector *query,
								 const PgturbohybridMultiVector *doc)
{
	return TqMultiVectorMaxSimBlockedWithDot(query, doc,
											 TqDotProductF32Scalar);
}

#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX2
static double PGTURBOHYBRID_MULTIVECTOR_AVX2_TARGET
TqMultiVectorMaxSimBlockedAvx2(const PgturbohybridMultiVector *query,
							   const PgturbohybridMultiVector *doc)
{
	double		score = 0.0;

	PgturbohybridCheckSameMultiVectorDims(query, doc);

	for (int32 qb = 0; qb < query->count;
		 qb += PGTURBOHYBRID_MULTIVECTOR_BLOCKED_SCALAR_Q)
	{
		int32		blockCount =
			Min(query->count - qb, PGTURBOHYBRID_MULTIVECTOR_BLOCKED_SCALAR_Q);
		const float *queryValues =
			query->values + ((Size) qb * (Size) query->dim);
		double		best[PGTURBOHYBRID_MULTIVECTOR_BLOCKED_SCALAR_Q];
		double		dots[PGTURBOHYBRID_MULTIVECTOR_BLOCKED_SCALAR_Q];

		for (int32 qi = 0; qi < blockCount; qi++)
			best[qi] = -INFINITY;

		for (int32 di = 0; di < doc->count; di++)
		{
			const float *dv = doc->values + ((Size) di * (Size) doc->dim);

			if (blockCount == 1)
				dots[0] = TqDotProductF32Avx2(queryValues, dv, query->dim);
			else
				TqDotProductF32BlockAvx2(queryValues, dv, query->dim,
										 blockCount, dots);

			for (int32 qi = 0; qi < blockCount; qi++)
			{
				if (dots[qi] > best[qi])
					best[qi] = dots[qi];
			}
		}

		for (int32 qi = 0; qi < blockCount; qi++)
			score += best[qi];
	}

	return score;
}
#endif

#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_NEON
static double
TqMultiVectorMaxSimBlockedNeon(const PgturbohybridMultiVector *query,
							   const PgturbohybridMultiVector *doc)
{
	double		score = 0.0;

	PgturbohybridCheckSameMultiVectorDims(query, doc);

	for (int32 qb = 0; qb < query->count;
		 qb += PGTURBOHYBRID_MULTIVECTOR_BLOCKED_SCALAR_Q)
	{
		int32		blockCount =
			Min(query->count - qb, PGTURBOHYBRID_MULTIVECTOR_BLOCKED_SCALAR_Q);
		const float *queryValues =
			query->values + ((Size) qb * (Size) query->dim);
		double		best[PGTURBOHYBRID_MULTIVECTOR_BLOCKED_SCALAR_Q];
		double		dots[PGTURBOHYBRID_MULTIVECTOR_BLOCKED_SCALAR_Q];

		for (int32 qi = 0; qi < blockCount; qi++)
			best[qi] = -INFINITY;

		for (int32 di = 0; di < doc->count; di++)
		{
			const float *dv = doc->values + ((Size) di * (Size) doc->dim);

			if (blockCount == 1)
				dots[0] = TqDotProductF32Neon(queryValues, dv, query->dim);
			else
				TqDotProductF32BlockNeon(queryValues, dv, query->dim,
										 blockCount, dots);

			for (int32 qi = 0; qi < blockCount; qi++)
			{
				if (dots[qi] > best[qi])
					best[qi] = dots[qi];
			}
		}

		for (int32 qi = 0; qi < blockCount; qi++)
			score += best[qi];
	}

	return score;
}
#endif

#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX512F
static double PGTURBOHYBRID_MULTIVECTOR_AVX512F_TARGET
TqMultiVectorMaxSimBlockedAvx512f(const PgturbohybridMultiVector *query,
								  const PgturbohybridMultiVector *doc)
{
	double		score = 0.0;

	PgturbohybridCheckSameMultiVectorDims(query, doc);

	for (int32 qb = 0; qb < query->count;
		 qb += PGTURBOHYBRID_MULTIVECTOR_BLOCKED_SCALAR_Q)
	{
		int32		blockCount =
			Min(query->count - qb, PGTURBOHYBRID_MULTIVECTOR_BLOCKED_SCALAR_Q);
		const float *queryValues =
			query->values + ((Size) qb * (Size) query->dim);
		double		best[PGTURBOHYBRID_MULTIVECTOR_BLOCKED_SCALAR_Q];
		double		dots[PGTURBOHYBRID_MULTIVECTOR_BLOCKED_SCALAR_Q];

		for (int32 qi = 0; qi < blockCount; qi++)
			best[qi] = -INFINITY;

		for (int32 di = 0; di < doc->count; di++)
		{
			const float *dv = doc->values + ((Size) di * (Size) doc->dim);

			if (blockCount == 1)
				dots[0] = TqDotProductF32Avx512f(queryValues, dv, query->dim);
			else
				TqDotProductF32BlockAvx512f(queryValues, dv, query->dim,
											blockCount, dots);

			for (int32 qi = 0; qi < blockCount; qi++)
			{
				if (dots[qi] > best[qi])
					best[qi] = dots[qi];
			}
		}

		for (int32 qi = 0; qi < blockCount; qi++)
			score += best[qi];
	}

	return score;
}
#endif

static TqMultiVectorMaxSimFunc
TqResolveMultiVectorMaxSimKernel(void)
{
	if (pgturbohybrid_dense_exact_simd_force == PGTURBOHYBRID_EXACT_SIMD_FORCE_SCALAR)
		return TqMultiVectorMaxSimBlockedScalar;

#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX512F
	if (TqMultiVectorAvx512fAvailable())
		return TqMultiVectorMaxSimBlockedAvx512f;
#endif
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX2
	if (TqMultiVectorAvx2Available())
		return TqMultiVectorMaxSimBlockedAvx2;
#endif
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_NEON
	return TqMultiVectorMaxSimBlockedNeon;
#endif
	return TqMultiVectorMaxSimBlockedScalar;
}

const char *
TqMultiVectorMaxSimKernelName(void)
{
	TqMultiVectorMaxSimFunc func = TqResolveMultiVectorMaxSimKernel();

	if (func == TqMultiVectorMaxSimBlockedScalar)
		return "blocked_scalar";
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX512F
	if (func == TqMultiVectorMaxSimBlockedAvx512f)
		return "blocked_avx512";
#endif
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX2
	if (func == TqMultiVectorMaxSimBlockedAvx2)
		return "blocked_avx2";
#endif
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_NEON
	if (func == TqMultiVectorMaxSimBlockedNeon)
		return "blocked_neon";
#endif
	return "unknown";
}

double
TqMultiVectorMaxSim(const PgturbohybridMultiVector *query,
					const PgturbohybridMultiVector *doc)
{
	return TqResolveMultiVectorMaxSimKernel()(query, doc);
}

static void
TqMvSkipSpaces(const char **cursor)
{
	while (**cursor != '\0' && isspace((unsigned char) **cursor))
		(*cursor)++;
}

static bool
TqMvConsumeLiteral(const char **cursor, const char *literal)
{
	Size		len;

	TqMvSkipSpaces(cursor);
	len = strlen(literal);
	if (strncmp(*cursor, literal, len) != 0)
		return false;

	*cursor += len;
	return true;
}

static void
TqMvInputError(const char *message)
{
	ereport(ERROR,
			(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
			 errmsg("invalid input syntax for turbohybrid_multivector"),
			 errdetail("%s", message)));
}

static int32
TqMvParseInt32(const char **cursor, const char *fieldName)
{
	char	   *endptr;
	long		value;

	TqMvSkipSpaces(cursor);
	errno = 0;
	value = strtol(*cursor, &endptr, 10);
	if (endptr == *cursor)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("invalid input syntax for turbohybrid_multivector"),
				 errdetail("Expected integer value for %s.", fieldName)));
	if (errno == ERANGE || value < PG_INT32_MIN || value > PG_INT32_MAX)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("turbohybrid_multivector %s is out of range", fieldName)));

	*cursor = endptr;
	return (int32) value;
}

static float4
TqMvParseFloat4(const char **cursor)
{
	char	   *endptr;
	double		value;
	float4		result;

	TqMvSkipSpaces(cursor);
	errno = 0;
	value = strtod(*cursor, &endptr);
	if (endptr == *cursor)
		TqMvInputError("Expected floating-point value.");
	if (errno == ERANGE || !isfinite(value))
		TqMvInputError("Multivector values must be finite.");

	result = (float4) value;
	if (!isfinite(result))
		TqMvInputError("Multivector values must be finite.");

	*cursor = endptr;
	return result;
}

static void
TqMvExpectChar(const char **cursor, char expected)
{
	TqMvSkipSpaces(cursor);
	if (**cursor != expected)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("invalid input syntax for turbohybrid_multivector"),
				 errdetail("Expected '%c'.", expected)));
	(*cursor)++;
}

static void
PgturbohybridMultiVectorRejectTextFallback(void)
{
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("hybrid text queries require a turbohybrid index scan"),
			 errdetail("Scalar multivector MaxSim can only evaluate the multivector payload.")));
}

Datum
pgturbohybrid_multivector_in(PG_FUNCTION_ARGS)
{
	const char *cursor = PG_GETARG_CSTRING(0);
	int32		dim;
	int32		count;
	Size		resultSize;
	PgturbohybridMultiVector *result;
	float	   *dest;

	if (!TqMvConsumeLiteral(&cursor, "turbohybrid_multivector"))
		TqMvInputError("Expected turbohybrid_multivector prefix.");
	TqMvExpectChar(&cursor, '(');
	if (!TqMvConsumeLiteral(&cursor, "dim"))
		TqMvInputError("Expected dim field.");
	TqMvExpectChar(&cursor, '=');
	dim = TqMvParseInt32(&cursor, "dim");
	TqMvExpectChar(&cursor, ',');
	if (!TqMvConsumeLiteral(&cursor, "count"))
		TqMvInputError("Expected count field.");
	TqMvExpectChar(&cursor, '=');
	count = TqMvParseInt32(&cursor, "count");

	resultSize = PgturbohybridMultiVectorSize(count, dim);
	result = (PgturbohybridMultiVector *) palloc0(resultSize);
	SET_VARSIZE(result, resultSize);
	result->dim = dim;
	result->count = count;
	result->flags = 0;
	dest = result->values;

	TqMvExpectChar(&cursor, ',');
	if (!TqMvConsumeLiteral(&cursor, "values"))
		TqMvInputError("Expected values field.");
	TqMvExpectChar(&cursor, '=');
	TqMvExpectChar(&cursor, '[');
	for (int32 i = 0; i < count; i++)
	{
		TqMvExpectChar(&cursor, '[');
		for (int32 j = 0; j < dim; j++)
		{
			*dest++ = TqMvParseFloat4(&cursor);
			if (j + 1 < dim)
				TqMvExpectChar(&cursor, ',');
		}
		TqMvExpectChar(&cursor, ']');
		if (i + 1 < count)
			TqMvExpectChar(&cursor, ',');
	}
	TqMvExpectChar(&cursor, ']');
	TqMvExpectChar(&cursor, ')');
	TqMvSkipSpaces(&cursor);
	if (*cursor != '\0')
		TqMvInputError("Trailing junk after multivector literal.");

	PgturbohybridCheckMultiVector(result);
	PG_RETURN_PGTURBOHYBRID_MULTIVECTOR_P(result);
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
pgturbohybrid_multivector_recv(PG_FUNCTION_ARGS)
{
	StringInfo	buf = (StringInfo) PG_GETARG_POINTER(0);
	int32		formatVersion;
	int32		dim;
	int32		count;
	uint32		flags;
	Size		floatCount;
	Size		resultSize;
	PgturbohybridMultiVector *result;

	formatVersion = (int32) pq_getmsgint(buf, 4);
	if (formatVersion != PGTURBOHYBRID_MULTIVECTOR_BINARY_VERSION)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("unsupported turbohybrid_multivector binary format version %d",
						formatVersion)));

	dim = (int32) pq_getmsgint(buf, 4);
	count = (int32) pq_getmsgint(buf, 4);
	flags = (uint32) pq_getmsgint(buf, 4);
	floatCount = PgturbohybridMultiVectorFloatCount(count, dim);
	resultSize = PgturbohybridMultiVectorSize(count, dim);

	result = (PgturbohybridMultiVector *) palloc0(resultSize);
	SET_VARSIZE(result, resultSize);
	result->dim = dim;
	result->count = count;
	result->flags = flags;

	for (Size i = 0; i < floatCount; i++)
		result->values[i] = pq_getmsgfloat4(buf);
	pq_getmsgend(buf);

	PgturbohybridCheckMultiVector(result);
	PG_RETURN_PGTURBOHYBRID_MULTIVECTOR_P(result);
}

Datum
pgturbohybrid_multivector_send(PG_FUNCTION_ARGS)
{
	PgturbohybridMultiVector *mv = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(0);
	Size		floatCount = PgturbohybridMultiVectorFloatCount(mv->count, mv->dim);
	StringInfoData buf;

	pq_begintypsend(&buf);
	pq_sendint32(&buf, PGTURBOHYBRID_MULTIVECTOR_BINARY_VERSION);
	pq_sendint32(&buf, (uint32) mv->dim);
	pq_sendint32(&buf, (uint32) mv->count);
	pq_sendint32(&buf, mv->flags);
	for (Size i = 0; i < floatCount; i++)
		pq_sendfloat4(&buf, mv->values[i]);

	PG_RETURN_BYTEA_P(pq_endtypsend(&buf));
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
pgturbohybrid_multivector_from_float4(PG_FUNCTION_ARGS)
{
	ArrayType  *array;
	Datum	   *elements;
	bool	   *nulls;
	int			nelems;
	int32		dim = PG_GETARG_INT32(1);
	int32		count;
	Size		resultSize;
	PgturbohybridMultiVector *result;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("multivector values array cannot be null")));

	if (dim <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("multivector dimensions must be greater than zero")));

	array = PG_GETARG_ARRAYTYPE_P(0);
	if (ARR_NDIM(array) == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("multivector values array cannot be empty")));

	if (ARR_ELEMTYPE(array) != FLOAT4OID)
		ereport(ERROR,
				(errcode(ERRCODE_DATATYPE_MISMATCH),
				 errmsg("expected real[] input")));

	deconstruct_array(array, FLOAT4OID, sizeof(float4), true, TYPALIGN_INT,
					  &elements, &nulls, &nelems);

	if (nelems <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("multivector values array cannot be empty")));

	if (nelems % dim != 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("multivector values array length must be divisible by dimensions")));

	count = nelems / dim;
	resultSize = PgturbohybridMultiVectorSize(count, dim);
	result = (PgturbohybridMultiVector *) palloc0(resultSize);
	SET_VARSIZE(result, resultSize);
	result->dim = dim;
	result->count = count;
	result->flags = 0;

	for (int i = 0; i < nelems; i++)
	{
		float4		value;

		if (nulls[i])
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("multivector values array cannot contain null elements")));

		value = DatumGetFloat4(elements[i]);
		if (!isfinite(value))
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("multivector cannot contain NaN or infinite values")));
		result->values[i] = value;
	}

	PgturbohybridCheckMultiVector(result);
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
pgturbohybrid_multivector_subvector(PG_FUNCTION_ARGS)
{
	PgturbohybridMultiVector *mv = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(0);
	int32		ordinal = PG_GETARG_INT32(1);

	if (ordinal < 1 || ordinal > mv->count)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("multivector subvector ordinal %d is out of range",
						ordinal)));

	PG_RETURN_PGTURBOHYBRID_VECTOR_P(
		PgturbohybridMultiVectorSubvectorCopy(mv, ordinal - 1));
}

Datum
pgturbohybrid_multivector_to_vector_array(PG_FUNCTION_ARGS)
{
	PgturbohybridMultiVector *mv = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(0);
	Oid			vectorOid = PgturbohybridVectorTypeOid();
	Datum	   *elements;
	int16		typlen;
	bool		typbyval;
	char		typalign;
	ArrayType  *array;

	get_typlenbyvalalign(vectorOid, &typlen, &typbyval, &typalign);
	elements = palloc(sizeof(Datum) * (Size) mv->count);
	for (int32 i = 0; i < mv->count; i++)
		elements[i] =
			PointerGetDatum(PgturbohybridMultiVectorSubvectorCopy(mv, i));

	array = construct_array(elements, mv->count, vectorOid, typlen, typbyval,
							typalign);
	PG_RETURN_ARRAYTYPE_P(array);
}

Datum
pgturbohybrid_multivector_maxsim(PG_FUNCTION_ARGS)
{
	PgturbohybridMultiVector *query = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(0);
	PgturbohybridMultiVector *doc = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(1);

	PG_RETURN_FLOAT8(TqMultiVectorMaxSim(query, doc));
}

Datum
pgturbohybrid_multivector_maxsim_scalar(PG_FUNCTION_ARGS)
{
	PgturbohybridMultiVector *query = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(0);
	PgturbohybridMultiVector *doc = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(1);

	PG_RETURN_FLOAT8(TqMultiVectorMaxSimScalar(query, doc));
}

Datum
pgturbohybrid_multivector_maxsim_blocked_scalar(PG_FUNCTION_ARGS)
{
	PgturbohybridMultiVector *query = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(0);
	PgturbohybridMultiVector *doc = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(1);

	PG_RETURN_FLOAT8(TqMultiVectorMaxSimBlockedScalar(query, doc));
}

Datum
pgturbohybrid_multivector_maxsim_distance(PG_FUNCTION_ARGS)
{
	PgturbohybridMultiVector *query = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(0);
	PgturbohybridMultiVector *doc = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(1);

	PG_RETURN_FLOAT8(-TqMultiVectorMaxSim(query, doc));
}

Datum
pgturbohybrid_multivector_query_distance(PG_FUNCTION_ARGS)
{
	PgturbohybridMultiVector *doc = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(0);
	PgturbohybridQueryHeader *query = PG_GETARG_PGTURBOHYBRID_QUERY_P(1);
	PgturbohybridMultiVector *queryMv;
	bool		sqlExprContext = fcinfo->flinfo != NULL &&
		fcinfo->flinfo->fn_expr != NULL;

	if (PgturbohybridQueryTextIndexOrderByContext(fcinfo))
	{
		PgturbohybridQueryValidateFast(query);
		if (PgturbohybridQueryHasText(query))
			PG_RETURN_FLOAT8(0.0);
	}

	PgturbohybridQueryValidate(query);
	queryMv = PgturbohybridQueryGetMultiVector(query);
	if (PgturbohybridQueryHasText(query) && sqlExprContext)
		PgturbohybridMultiVectorRejectTextFallback();
	if (queryMv == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("turbohybrid_query requires multivector_query for multivector distance")));

	PG_RETURN_FLOAT8(-TqMultiVectorMaxSim(queryMv, doc));
}
