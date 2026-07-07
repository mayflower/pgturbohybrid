#include "postgres.h"

#include <ctype.h>
#include <errno.h>
#include <float.h>
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
#include "storage/fd.h"
#include "portability/instr_time.h"
#include "utils/fmgrprotos.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/jsonb.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/numeric.h"
#include "utils/syscache.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

#include "pgturbohybrid.h"
#include "pgturbohybrid_jsonb_compat.h"
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

#if !defined(PGTURBOHYBRID_DISABLE_SIMD) && \
	(defined(__AVX512BW__) || (defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))))
#define PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX512BW 1
#else
#define PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX512BW 0
#endif

#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX512BW && \
	(!defined(__AVX512BW__) || !defined(__AVX512F__)) && \
	(defined(__GNUC__) || defined(__clang__))
#define PGTURBOHYBRID_MULTIVECTOR_AVX512BW_TARGET __attribute__((target("avx512bw,avx512f")))
#else
#define PGTURBOHYBRID_MULTIVECTOR_AVX512BW_TARGET
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
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_from_contexts);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_from_contexts_and_fields);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_dims);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_count);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_context_count);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_context_offsets);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_field_ids);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_subvector);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_to_vector_array);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_maxsim);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_context_maxsim);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_field_weighted_maxsim);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_maxsim_scalar);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_maxsim_blocked_scalar);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_maxsim_distance);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_query_distance);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_multivector_model_info);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_experimental_compact_code_score);

static Oid	pgturbohybrid_multivector_type_oid = InvalidOid;
static Oid	pgturbohybrid_sparse_vector_type_oid = InvalidOid;
extern char *pgturbohybrid_multivector_model_name;
extern int	pgturbohybrid_multivector_max_doc_vectors;
extern int	pgturbohybrid_multivector_max_query_vectors;
extern char *pgturbohybrid_multivector_learned_projection_path;
extern char *pgturbohybrid_multivector_learned_projection_model;
extern char *pgturbohybrid_multivector_learned_projection_checksum;

#define PGTURBOHYBRID_MULTIVECTOR_BINARY_VERSION 1
#define PGTURBOHYBRID_MULTIVECTOR_BINARY_VERSION_CONTEXTS 2
#define PGTURBOHYBRID_MULTIVECTOR_BLOCKED_SCALAR_Q 8

static Oid PgturbohybridExtensionSchema(Oid extensionOid);
static void PgturbohybridCheckMultiVectorHeader(int32 count, int32 dim);
static void TqMvSkipSpaces(const char **cursor);
static bool TqMvConsumeLiteral(const char **cursor, const char *literal);
static int32 TqMvParseInt32(const char **cursor, const char *fieldName);
static float4 TqMvParseFloat4(const char **cursor);
static void TqMvExpectChar(const char **cursor, char expected);
static void PgturbohybridMultiVectorRejectTextFallback(void);
static int32 *PgturbohybridMultiVectorReadInt4Array(ArrayType *array,
													const char *name,
													int *nelems);
static int16 *PgturbohybridMultiVectorReadInt2Array(ArrayType *array,
													const char *name,
													int *nelems);
static float4 *PgturbohybridMultiVectorReadFloat4Array(ArrayType *array,
													   const char *name,
													   int *nelems);
static PgturbohybridMultiVector *PgturbohybridMultiVectorBuildFromFlatArray(ArrayType *array,
																			 int32 dim,
																			 int32 contextCount,
																			 const int32 *contextOffsets,
																			 const int32 *fieldIds);
static void PgturbohybridMultiVectorValidateContexts(const PgturbohybridMultiVector *mv,
													 int32 contextCount,
													 const int32 *contextOffsets,
													 const int32 *fieldIds);
static const PgturbohybridMultiVectorModelInfo *PgturbohybridMultiVectorConfiguredModel(void);
static void PgturbohybridMultiVectorWarnSuspiciousTokenCount(uint32 tokenCount,
															 uint32 maxTokenCount);
static Vector *PgturbohybridMultiVectorSubvectorCopy(const PgturbohybridMultiVector *mv,
													 int32 ordinal);
static void PgturbohybridMultiVectorNormalizeToken(float *values, int32 dim);
static uint32 PgturbohybridMultiVectorProxyHash(uint32 a, uint32 b, uint32 salt);
static void PgturbohybridMultiVectorProxyMean(const PgturbohybridMultiVector *mv,
											  Vector *vector);
static void PgturbohybridMultiVectorProxyNormalizedMean(const PgturbohybridMultiVector *mv,
														Vector *vector);
static void PgturbohybridMultiVectorProxyLearnedProjectionV1(const PgturbohybridMultiVector *mv,
															 Vector *vector);

typedef struct PgturbohybridLearnedProjectionWeights
{
	char	   *path;
	char	   *model;
	char	   *checksum;
	int32		inputDim;
	int32		outputDim;
	Size		weightBytes;
	float	   *weights;
} PgturbohybridLearnedProjectionWeights;

static PgturbohybridLearnedProjectionWeights *pgturbohybrid_learned_projection_cache = NULL;

typedef double (*TqDotProductF32Func) (const float *a, const float *b, int32 dim);
typedef double (*TqMultiVectorMaxSimFunc) (const PgturbohybridMultiVector *query,
										   const PgturbohybridMultiVector *doc);
static double TqMultiVectorSymmetricMaxSimAverageWithDotUnchecked(const PgturbohybridMultiVector *a,
																  const PgturbohybridMultiVector *b,
																  TqDotProductF32Func dotProduct,
																  TqDotProductF32BlockFunc blockDotProduct);

static const PgturbohybridMultiVectorModelInfo pgturbohybrid_multivector_model_registry[] = {
	{
		"colbert-ir/colbertv2.0", 128, 32, 180, "dot_product", true, "f16",
		"special_and_punctuation", "flat_text", "stable",
		"Classic ColBERTv2 late-interaction text model."
	},
	{
		"answerdotai/answerai-colbert-small-v1", 96, 32, 512, "dot_product",
		true, "f16", "special_and_punctuation", "flat_text", "stable",
		"Small multilingual ColBERT model supported by FastEmbed."
	},
	{
		"jinaai/jina-colbert-v2", 128, 32, 8192, "dot_product", true, "f16",
		"model_skiplist", "long_context_text", "stable",
		"Jina-ColBERT-v2 default 128-dimensional Matryoshka variant."
	},
	{
		"jinaai/jina-colbert-v2-96", 96, 32, 8192, "dot_product", true, "f16",
		"model_skiplist", "long_context_text", "stable",
		"Jina-ColBERT-v2 96-dimensional Matryoshka variant."
	},
	{
		"jinaai/jina-colbert-v2-64", 64, 32, 8192, "dot_product", true, "f16",
		"model_skiplist", "long_context_text", "stable",
		"Jina-ColBERT-v2 64-dimensional Matryoshka variant."
	},
	{
		"lightonai/GTE-ModernColBERT-v1", 128, 48, 300, "maxsim", true, "f16",
		"pylate_skiplist", "long_context_text", "profile",
		"PyLate ModernColBERT profile; document length can be raised for long-context runs."
	},
	{
		"chadboyda/Reason-ModernColBERT", 128, 48, 300, "maxsim", true, "f16",
		"pylate_skiplist", "reasoning_text", "profile",
		"Reasoning-tuned ModernColBERT profile; verify tokenizer policy with the exported model."
	},
	{
		"VAGOsolutions/SauerkrautLM-Multi-ColBERT-15m", 128, 32, 256,
		"maxsim", true, "f16", "special_and_punctuation", "flat_text",
		"validation", "Small validation model used by the pgturbohybrid benchmark suite."
	},
	{
		"johannhartmann/SauerkrautLM-Multi-ColBERT-15m-GGUF", 128, 32, 256,
		"maxsim", true, "f16", "special_and_punctuation", "flat_text",
		"validation", "GGUF companion for live pg_colbert_llama validation."
	},
	{
		"vidore/colpali-v1.2", 128, 64, 0, "maxsim", true, "f16",
		"processor_policy", "visual_patch_multivector", "placeholder",
		"ColPali-style visual multivectors use image patch counts that vary by processor settings."
	},
	{
		"colpali-like-visual", 128, 64, 0, "maxsim", true, "f16",
		"processor_policy", "visual_patch_multivector", "placeholder",
		"Generic visual-document late-interaction profile for ColPali-compatible exports."
	}
};

static void
PgturbohybridMultiVectorModelJsonbAddKey(PgturbohybridJsonbState *state,
										 const char *key)
{
	JsonbValue	value;

	value.type = jbvString;
	value.val.string.val = (char *) key;
	value.val.string.len = strlen(key);
	PgturbohybridJsonbPush(state, WJB_KEY, &value);
}

static void
PgturbohybridMultiVectorModelJsonbAddString(PgturbohybridJsonbState *state,
											const char *key, const char *val)
{
	JsonbValue	value;

	PgturbohybridMultiVectorModelJsonbAddKey(state, key);
	value.type = jbvString;
	value.val.string.val = (char *) val;
	value.val.string.len = strlen(val);
	PgturbohybridJsonbPush(state, WJB_VALUE, &value);
}

static void
PgturbohybridMultiVectorModelJsonbAddBool(PgturbohybridJsonbState *state,
										  const char *key, bool val)
{
	JsonbValue	value;

	PgturbohybridMultiVectorModelJsonbAddKey(state, key);
	value.type = jbvBool;
	value.val.boolean = val;
	PgturbohybridJsonbPush(state, WJB_VALUE, &value);
}

static void
PgturbohybridMultiVectorModelJsonbAddInt32(PgturbohybridJsonbState *state,
										   const char *key, int32 val)
{
	JsonbValue	value;

	PgturbohybridMultiVectorModelJsonbAddKey(state, key);
	if (val <= 0)
	{
		value.type = jbvNull;
		PgturbohybridJsonbPush(state, WJB_VALUE, &value);
		return;
	}
	value.type = jbvNumeric;
	value.val.numeric = DatumGetNumeric(DirectFunctionCall1(int8_numeric,
															Int64GetDatum((int64) val)));
	PgturbohybridJsonbPush(state, WJB_VALUE, &value);
}

static void
PgturbohybridMultiVectorModelJsonbAddInt64(PgturbohybridJsonbState *state,
										   const char *key, int64 val)
{
	JsonbValue	value;

	PgturbohybridMultiVectorModelJsonbAddKey(state, key);
	value.type = jbvNumeric;
	value.val.numeric = DatumGetNumeric(DirectFunctionCall1(int8_numeric,
															Int64GetDatum(val)));
	PgturbohybridJsonbPush(state, WJB_VALUE, &value);
}

static void
PgturbohybridMultiVectorModelJsonbAddFloat8(PgturbohybridJsonbState *state,
											const char *key, double val)
{
	JsonbValue	value;

	PgturbohybridMultiVectorModelJsonbAddKey(state, key);
	value.type = jbvNumeric;
	value.val.numeric = DatumGetNumeric(DirectFunctionCall1(float8_numeric,
															Float8GetDatum(val)));
	PgturbohybridJsonbPush(state, WJB_VALUE, &value);
}

static Jsonb *
PgturbohybridMultiVectorModelInfoJsonb(const PgturbohybridMultiVectorModelInfo *info,
									   const char *requestedName)
{
	PgturbohybridJsonbState state;

	PgturbohybridJsonbStateInit(&state);
	PgturbohybridJsonbBeginObject(&state);
	PgturbohybridMultiVectorModelJsonbAddString(&state, "model_name",
												info != NULL ? info->modelName : requestedName);
	PgturbohybridMultiVectorModelJsonbAddBool(&state, "known", info != NULL);
	if (info == NULL)
	{
		PgturbohybridMultiVectorModelJsonbAddString(&state, "status", "unknown");
		PgturbohybridMultiVectorModelJsonbAddInt32(&state, "dim", 0);
		PgturbohybridMultiVectorModelJsonbAddInt32(&state, "default_query_max_tokens", 0);
		PgturbohybridMultiVectorModelJsonbAddInt32(&state, "default_doc_max_tokens", 0);
		PgturbohybridMultiVectorModelJsonbAddString(&state, "distance_mode", "unknown");
		PgturbohybridMultiVectorModelJsonbAddBool(&state, "normalized_tokens", false);
		PgturbohybridMultiVectorModelJsonbAddString(&state, "recommended_storage_kind", "unknown");
		PgturbohybridMultiVectorModelJsonbAddString(&state, "token_mask_policy", "unknown");
		PgturbohybridMultiVectorModelJsonbAddString(&state, "field_context_policy", "unknown");
		PgturbohybridMultiVectorModelJsonbAddString(&state, "notes",
													"Pass an explicit expected dimension for unregistered models.");
		return PgturbohybridJsonbEndObject(&state);
	}

	PgturbohybridMultiVectorModelJsonbAddString(&state, "status", info->status);
	PgturbohybridMultiVectorModelJsonbAddInt32(&state, "dim", info->dim);
	PgturbohybridMultiVectorModelJsonbAddInt32(&state, "default_query_max_tokens",
											   info->defaultQueryMaxTokens);
	PgturbohybridMultiVectorModelJsonbAddInt32(&state, "default_doc_max_tokens",
											   info->defaultDocMaxTokens);
	PgturbohybridMultiVectorModelJsonbAddString(&state, "distance_mode",
												info->distanceMode);
	PgturbohybridMultiVectorModelJsonbAddBool(&state, "normalized_tokens",
											  info->normalizedTokens);
	PgturbohybridMultiVectorModelJsonbAddString(&state, "recommended_storage_kind",
												info->recommendedStorageKind);
	PgturbohybridMultiVectorModelJsonbAddString(&state, "token_mask_policy",
												info->tokenMaskPolicy);
	PgturbohybridMultiVectorModelJsonbAddString(&state, "field_context_policy",
												info->fieldContextPolicy);
	PgturbohybridMultiVectorModelJsonbAddString(&state, "notes", info->notes);
	return PgturbohybridJsonbEndObject(&state);
}

const PgturbohybridMultiVectorModelInfo *
PgturbohybridMultiVectorLookupModel(const char *modelName)
{
	if (modelName == NULL || modelName[0] == '\0')
		return NULL;

	for (int i = 0; i < lengthof(pgturbohybrid_multivector_model_registry); i++)
	{
		const PgturbohybridMultiVectorModelInfo *info =
			&pgturbohybrid_multivector_model_registry[i];

		if (pg_strcasecmp(modelName, info->modelName) == 0)
			return info;
	}
	return NULL;
}

static const PgturbohybridMultiVectorModelInfo *
PgturbohybridMultiVectorConfiguredModel(void)
{
	return PgturbohybridMultiVectorLookupModel(pgturbohybrid_multivector_model_name);
}

FUNCTION_PREFIX Datum
pgturbohybrid_multivector_model_info(PG_FUNCTION_ARGS)
{
	char	   *modelName = text_to_cstring(PG_GETARG_TEXT_PP(0));
	const PgturbohybridMultiVectorModelInfo *info =
		PgturbohybridMultiVectorLookupModel(modelName);

	PG_RETURN_JSONB_P(PgturbohybridMultiVectorModelInfoJsonb(info, modelName));
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

Oid
PgturbohybridSparseVectorTypeOid(void)
{
	Oid			extensionOid;
	Oid			schemaOid;

	if (OidIsValid(pgturbohybrid_sparse_vector_type_oid))
		return pgturbohybrid_sparse_vector_type_oid;

	extensionOid = get_extension_oid("pgturbohybrid", true);
	if (!OidIsValid(extensionOid))
		return InvalidOid;

	schemaOid = PgturbohybridExtensionSchema(extensionOid);
	if (!OidIsValid(schemaOid))
		return InvalidOid;

	pgturbohybrid_sparse_vector_type_oid =
		GetSysCacheOid2(TYPENAMENSP, Anum_pg_type_oid,
						CStringGetDatum("turbohybrid_sparse_vector"),
						ObjectIdGetDatum(schemaOid));
	return pgturbohybrid_sparse_vector_type_oid;
}

bool
PgturbohybridTypeIsSparseVector(Oid typeOid)
{
	Oid			sparseOid = PgturbohybridSparseVectorTypeOid();

	return OidIsValid(sparseOid) && typeOid == sparseOid;
}

bool
PgturbohybridTypeIsMultiVector(Oid typeOid)
{
	Oid			multivectorOid = PgturbohybridMultiVectorTypeOid();

	if (!OidIsValid(multivectorOid) || !OidIsValid(typeOid))
		return false;

	return getBaseType(typeOid) == multivectorOid;
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

Size
PgturbohybridMultiVectorExtendedSize(int32 count, int32 dim,
									 int32 contextCount, bool hasFields)
{
	Size		size = PgturbohybridMultiVectorSize(count, dim);

	if (contextCount <= 0 || contextCount > count)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid multivector context count %d", contextCount)));
	if ((Size) contextCount > (MaxAllocSize - size - sizeof(int32)) / sizeof(int32))
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("multivector context metadata is too large")));

	size += sizeof(int32);
	size += sizeof(int32) * (Size) contextCount;
	if (hasFields)
	{
		if ((Size) contextCount > (MaxAllocSize - size) / sizeof(int32))
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("multivector field metadata is too large")));
		size += sizeof(int32) * (Size) contextCount;
	}
	return size;
}

static int32 *
PgturbohybridMultiVectorContextCountPtr(const PgturbohybridMultiVector *mv)
{
	return (int32 *) ((char *) mv + PgturbohybridMultiVectorSize(mv->count,
																 mv->dim));
}

static int32 *
PgturbohybridMultiVectorMutableContextOffsets(PgturbohybridMultiVector *mv)
{
	return PgturbohybridMultiVectorContextCountPtr(mv) + 1;
}

static int32 *
PgturbohybridMultiVectorMutableContextFields(PgturbohybridMultiVector *mv)
{
	int32	   *contextCount = PgturbohybridMultiVectorContextCountPtr(mv);

	return PgturbohybridMultiVectorMutableContextOffsets(mv) + *contextCount;
}

bool
PgturbohybridMultiVectorHasContexts(const PgturbohybridMultiVector *mv)
{
	PgturbohybridCheckMultiVector(mv);
	return (mv->flags & PGTURBOHYBRID_MULTIVECTOR_FLAG_CONTEXTS) != 0;
}

int32
PgturbohybridMultiVectorContextCount(const PgturbohybridMultiVector *mv)
{
	PgturbohybridCheckMultiVector(mv);
	if ((mv->flags & PGTURBOHYBRID_MULTIVECTOR_FLAG_CONTEXTS) == 0)
		return 1;
	return *PgturbohybridMultiVectorContextCountPtr(mv);
}

const int32 *
PgturbohybridMultiVectorContextOffsets(const PgturbohybridMultiVector *mv)
{
	PgturbohybridCheckMultiVector(mv);
	if ((mv->flags & PGTURBOHYBRID_MULTIVECTOR_FLAG_CONTEXTS) == 0)
		return NULL;
	return PgturbohybridMultiVectorContextCountPtr(mv) + 1;
}

const int32 *
PgturbohybridMultiVectorContextFields(const PgturbohybridMultiVector *mv)
{
	int32		contextCount;

	PgturbohybridCheckMultiVector(mv);
	if ((mv->flags & PGTURBOHYBRID_MULTIVECTOR_FLAG_FIELDS) == 0)
		return NULL;
	contextCount = *PgturbohybridMultiVectorContextCountPtr(mv);
	return PgturbohybridMultiVectorContextCountPtr(mv) + 1 + contextCount;
}

void
PgturbohybridCheckMultiVector(const PgturbohybridMultiVector *mv)
{
	Size		actual;
	Size		expected;
	Size		floatCount;
	int32		contextCount = 0;
	const int32 *contextOffsets = NULL;
	const int32 *fieldIds = NULL;

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

	if ((mv->flags & ~PGTURBOHYBRID_MULTIVECTOR_KNOWN_FLAGS) != 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("malformed multivector value"),
				 errdetail("Multivector flags contain unsupported bits 0x%x.",
						   mv->flags & ~PGTURBOHYBRID_MULTIVECTOR_KNOWN_FLAGS)));
	if ((mv->flags & PGTURBOHYBRID_MULTIVECTOR_FLAG_FIELDS) != 0 &&
		(mv->flags & PGTURBOHYBRID_MULTIVECTOR_FLAG_CONTEXTS) == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("malformed multivector value"),
				 errdetail("Field metadata requires context metadata.")));

	expected = PgturbohybridMultiVectorSize(mv->count, mv->dim);
	if ((mv->flags & PGTURBOHYBRID_MULTIVECTOR_FLAG_CONTEXTS) != 0)
	{
		if (actual < expected + sizeof(int32))
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("malformed multivector value"),
					 errdetail("Context metadata is truncated.")));
		contextCount = *PgturbohybridMultiVectorContextCountPtr(mv);
		expected =
			PgturbohybridMultiVectorExtendedSize(mv->count, mv->dim,
												 contextCount,
												 (mv->flags &
												  PGTURBOHYBRID_MULTIVECTOR_FLAG_FIELDS) != 0);
		contextOffsets = PgturbohybridMultiVectorContextCountPtr(mv) + 1;
		if ((mv->flags & PGTURBOHYBRID_MULTIVECTOR_FLAG_FIELDS) != 0)
			fieldIds = contextOffsets + contextCount;
	}
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

	if (contextCount > 0)
		PgturbohybridMultiVectorValidateContexts(mv, contextCount,
												 contextOffsets, fieldIds);
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

	PgturbohybridMultiVectorWarnSuspiciousTokenCount(tokenCount, maxTokenCount);
}

void
PgturbohybridMultiVectorCheckDim(uint32 dim, uint32 maxDim)
{
	const PgturbohybridMultiVectorModelInfo *modelInfo;

	if (dim == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("multivector dimensions must be greater than zero")));

	modelInfo = PgturbohybridMultiVectorConfiguredModel();
	if (modelInfo != NULL && modelInfo->dim > 0 && dim != (uint32) modelInfo->dim)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("multivector dimensions %u do not match configured model \"%s\" dimensions %d",
						dim, modelInfo->modelName, modelInfo->dim),
				 errhint("Check turbohybrid.multivector_model_name, pg_colbert_llama.expected_dim, the embedding export, or rebuild the index with matching multivectors.")));

	if (maxDim == 0 || dim > maxDim)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("multivector dimensions %u exceed configured limit %u",
						dim, maxDim)));
}

static void
PgturbohybridMultiVectorWarnSuspiciousTokenCount(uint32 tokenCount, uint32 maxTokenCount)
{
	static bool warnedQuery = false;
	static bool warnedDoc = false;
	const PgturbohybridMultiVectorModelInfo *modelInfo =
		PgturbohybridMultiVectorConfiguredModel();
	int32		modelLimit = 0;
	const char *role = NULL;
	bool	   *warned = NULL;

	if (modelInfo == NULL)
		return;

	if (maxTokenCount == (uint32) pgturbohybrid_multivector_max_query_vectors)
	{
		modelLimit = modelInfo->defaultQueryMaxTokens;
		role = "query";
		warned = &warnedQuery;
	}
	else if (maxTokenCount == (uint32) pgturbohybrid_multivector_max_doc_vectors)
	{
		modelLimit = modelInfo->defaultDocMaxTokens;
		role = "document";
		warned = &warnedDoc;
	}

	if (role == NULL || warned == NULL || *warned || modelLimit <= 0 ||
		tokenCount <= (uint32) modelLimit)
		return;

	*warned = true;
	ereport(WARNING,
			(errmsg("multivector %s token count %u exceeds %s profile default %d",
					role, tokenCount, modelInfo->modelName, modelLimit),
			 errhint("This is allowed by the current GUC limit, but benchmark reports should record the model profile and token limit override.")));
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

static void
PgturbohybridMultiVectorValidateContexts(const PgturbohybridMultiVector *mv,
										 int32 contextCount,
										 const int32 *contextOffsets,
										 const int32 *fieldIds)
{
	if (contextOffsets == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("multivector context offsets cannot be null")));
	if (contextCount <= 0 || contextCount > mv->count)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid multivector context count %d", contextCount)));
	if (contextOffsets[0] != 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("multivector context offsets must start at zero")));
	for (int32 i = 0; i < contextCount; i++)
	{
		if (contextOffsets[i] < 0 || contextOffsets[i] >= mv->count)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("multivector context offset %d is out of range",
							contextOffsets[i])));
		if (i > 0 && contextOffsets[i] <= contextOffsets[i - 1])
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("multivector context offsets must be strictly increasing")));
		if (fieldIds != NULL && fieldIds[i] < 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("multivector field ids must be non-negative")));
	}
}

static int32 *
PgturbohybridMultiVectorReadInt4Array(ArrayType *array, const char *name,
									  int *nelems)
{
	Datum	   *elements;
	bool	   *nulls;
	int32	   *values;

	if (array == NULL || ARR_NDIM(array) == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("%s array cannot be empty", name)));
	if (ARR_ELEMTYPE(array) != INT4OID)
		ereport(ERROR,
				(errcode(ERRCODE_DATATYPE_MISMATCH),
				 errmsg("expected integer[] input for %s", name)));

	deconstruct_array(array, INT4OID, sizeof(int32), true, TYPALIGN_INT,
					  &elements, &nulls, nelems);
	if (*nelems <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("%s array cannot be empty", name)));

	values = palloc(sizeof(int32) * (Size) *nelems);
	for (int i = 0; i < *nelems; i++)
	{
		if (nulls[i])
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("%s array cannot contain null elements", name)));
		values[i] = DatumGetInt32(elements[i]);
	}
	return values;
}

static int16 *
PgturbohybridMultiVectorReadInt2Array(ArrayType *array, const char *name,
									  int *nelems)
{
	Datum	   *elements;
	bool	   *nulls;
	int16	   *values;

	if (array == NULL || ARR_NDIM(array) == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("%s array cannot be empty", name)));
	if (ARR_ELEMTYPE(array) != INT2OID)
		ereport(ERROR,
				(errcode(ERRCODE_DATATYPE_MISMATCH),
				 errmsg("expected smallint[] input for %s", name)));

	deconstruct_array(array, INT2OID, sizeof(int16), true, TYPALIGN_SHORT,
					  &elements, &nulls, nelems);
	if (*nelems <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("%s array cannot be empty", name)));

	values = palloc(sizeof(int16) * (Size) *nelems);
	for (int i = 0; i < *nelems; i++)
	{
		if (nulls[i])
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("%s array cannot contain null elements", name)));
		values[i] = DatumGetInt16(elements[i]);
	}
	return values;
}

static float4 *
PgturbohybridMultiVectorReadFloat4Array(ArrayType *array, const char *name,
										int *nelems)
{
	Datum	   *elements;
	bool	   *nulls;
	float4	   *values;

	if (array == NULL || ARR_NDIM(array) == 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("%s array cannot be empty", name)));
	if (ARR_ELEMTYPE(array) != FLOAT4OID)
		ereport(ERROR,
				(errcode(ERRCODE_DATATYPE_MISMATCH),
				 errmsg("expected real[] input for %s", name)));

	deconstruct_array(array, FLOAT4OID, sizeof(float4), true, TYPALIGN_INT,
					  &elements, &nulls, nelems);
	if (*nelems <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("%s array cannot be empty", name)));

	values = palloc(sizeof(float4) * (Size) *nelems);
	for (int i = 0; i < *nelems; i++)
	{
		if (nulls[i])
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("%s array must contain finite non-null values", name)));
		values[i] = DatumGetFloat4(elements[i]);
		if (!isfinite(values[i]))
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("%s array must contain finite non-null values", name)));
	}
	return values;
}

static PgturbohybridMultiVector *
PgturbohybridMultiVectorBuildFromFlatArray(ArrayType *array, int32 dim,
										   int32 contextCount,
										   const int32 *contextOffsets,
										   const int32 *fieldIds)
{
	Datum	   *elements;
	bool	   *nulls;
	int			nelems;
	int32		count;
	Size		resultSize;
	PgturbohybridMultiVector *result;
	bool		hasContexts = contextCount > 0;
	bool		hasFields = fieldIds != NULL;

	if (array == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("multivector values array cannot be null")));
	if (dim <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("multivector dimensions must be greater than zero")));
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
	if (hasContexts)
	{
		PgturbohybridMultiVector stackMv;

		memset(&stackMv, 0, sizeof(stackMv));
		stackMv.dim = dim;
		stackMv.count = count;
		PgturbohybridMultiVectorValidateContexts(&stackMv, contextCount,
												 contextOffsets, fieldIds);
		resultSize =
			PgturbohybridMultiVectorExtendedSize(count, dim, contextCount,
												 hasFields);
	}
	else
		resultSize = PgturbohybridMultiVectorSize(count, dim);

	result = (PgturbohybridMultiVector *) palloc0(resultSize);
	SET_VARSIZE(result, resultSize);
	result->dim = dim;
	result->count = count;
	result->flags = 0;
	if (hasContexts)
		result->flags |= PGTURBOHYBRID_MULTIVECTOR_FLAG_CONTEXTS;
	if (hasFields)
		result->flags |= PGTURBOHYBRID_MULTIVECTOR_FLAG_FIELDS;

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

	if (hasContexts)
	{
		int32	   *storedContextCount =
			PgturbohybridMultiVectorContextCountPtr(result);
		int32	   *storedOffsets =
			PgturbohybridMultiVectorMutableContextOffsets(result);

		*storedContextCount = contextCount;
		memcpy(storedOffsets, contextOffsets,
			   sizeof(int32) * (Size) contextCount);
		if (hasFields)
			memcpy(PgturbohybridMultiVectorMutableContextFields(result),
				   fieldIds, sizeof(int32) * (Size) contextCount);
	}

	PgturbohybridCheckMultiVector(result);
	return result;
}

static double
PgturbohybridMultiVectorTokenNorm(const PgturbohybridMultiVector *mv, int32 token)
{
	const float *values = PgturbohybridMultiVectorValues(mv, token);
	double		norm = 0.0;

	for (int32 dim = 0; dim < mv->dim; dim++)
	{
		double		x = values[dim];

		norm += x * x;
	}
	return norm > 0.0 ? sqrt(norm) : 0.0;
}

static double
PgturbohybridMultiVectorTokenCosineWithNorms(const PgturbohybridMultiVector *mv,
											 int32 a, int32 b,
											 const double *tokenNorms)
{
	const float *av = PgturbohybridMultiVectorValues(mv, a);
	const float *bv = PgturbohybridMultiVectorValues(mv, b);
	double		dot = 0.0;
	double		denom;

	denom = tokenNorms[a] * tokenNorms[b];
	if (denom <= 0.0)
		return 0.0;
	for (int32 dim = 0; dim < mv->dim; dim++)
		dot += (double) av[dim] * (double) bv[dim];
	return dot / denom;
}

static void
PgturbohybridMultiVectorNormalizeToken(float *values, int32 dim)
{
	double		norm = 0.0;

	for (int32 i = 0; i < dim; i++)
		norm += (double) values[i] * (double) values[i];
	if (norm <= 0.0)
		return;
	norm = sqrt(norm);
	for (int32 i = 0; i < dim; i++)
		values[i] = (float) ((double) values[i] / norm);
}

PgturbohybridMultiVector *
PgturbohybridMultiVectorPoolDocumentTokens(const PgturbohybridMultiVector *mv,
										   int mode, double targetRatio,
										   int minTokens, MemoryContext ctx)
{
	MemoryContext oldCtx;
	PgturbohybridMultiVector *pooled;
	int32		targetCount;
	int		   *selected;
	bool	   *isSelected;
	int		   *clusterCounts;
	double	   *tokenNorms;
	double	   *nearestSelectedSimilarity;
	float	   *centroids;
	float	   *sums;
	Size		centroidBytes;
	Size		resultSize;

	PgturbohybridCheckMultiVector(mv);
	if (mode == PGTURBOHYBRID_MULTIVECTOR_TOKEN_POOLING_OFF ||
		mv->count <= 1 ||
		minTokens <= 0 ||
		mv->count < minTokens ||
		targetRatio >= 1.0)
		return (PgturbohybridMultiVector *) mv;
	if (targetRatio <= 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("multivector token pooling target ratio must be greater than zero")));

	targetCount = (int32) ceil((double) mv->count * targetRatio);
	targetCount = Max(1, Min(targetCount, mv->count));
	if (targetCount >= mv->count)
		return (PgturbohybridMultiVector *) mv;
	if ((mv->flags & PGTURBOHYBRID_MULTIVECTOR_FLAG_CONTEXTS) != 0)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("multivector token pooling does not support context-aware multivectors"),
				 errhint("Use multivector_token_pooling = off for context-aware multivectors.")));

	oldCtx = MemoryContextSwitchTo(ctx);
	resultSize = PgturbohybridMultiVectorSize(targetCount, mv->dim);
	pooled = palloc0(resultSize);
	SET_VARSIZE(pooled, resultSize);
	pooled->dim = mv->dim;
	pooled->count = targetCount;
	pooled->flags = mv->flags;

	selected = palloc0(sizeof(int) * targetCount);
	isSelected = palloc0(sizeof(bool) * mv->count);
	clusterCounts = palloc0(sizeof(int) * targetCount);
	tokenNorms = palloc0(sizeof(double) * mv->count);
	nearestSelectedSimilarity = palloc0(sizeof(double) * mv->count);
	centroidBytes = sizeof(float) * (Size) targetCount * (Size) mv->dim;
	centroids = palloc0(centroidBytes);
	sums = palloc0(centroidBytes);

	for (int32 token = 0; token < mv->count; token++)
		tokenNorms[token] = PgturbohybridMultiVectorTokenNorm(mv, token);
	selected[0] = 0;
	isSelected[0] = true;
	for (int32 token = 0; token < mv->count; token++)
	{
		if (token == selected[0])
			nearestSelectedSimilarity[token] = DBL_MAX;
		else
			nearestSelectedSimilarity[token] =
				PgturbohybridMultiVectorTokenCosineWithNorms(mv, token,
															 selected[0],
															 tokenNorms);
	}
	for (int32 cluster = 1; cluster < targetCount; cluster++)
	{
		double		bestWorstSimilarity = DBL_MAX;
		int			bestToken = -1;

		for (int32 token = 0; token < mv->count; token++)
		{
			if (isSelected[token])
				continue;
			if (bestToken < 0 ||
				nearestSelectedSimilarity[token] < bestWorstSimilarity)
			{
				bestWorstSimilarity = nearestSelectedSimilarity[token];
				bestToken = token;
			}
		}
		if (bestToken < 0)
			bestToken = cluster;
		selected[cluster] = bestToken;
		isSelected[bestToken] = true;
		nearestSelectedSimilarity[bestToken] = DBL_MAX;
		for (int32 token = 0; token < mv->count; token++)
		{
			double		similarity;

			if (isSelected[token])
				continue;
			similarity =
				PgturbohybridMultiVectorTokenCosineWithNorms(mv, token,
															 bestToken,
															 tokenNorms);
			if (similarity > nearestSelectedSimilarity[token])
				nearestSelectedSimilarity[token] = similarity;
		}
	}

	for (int32 cluster = 0; cluster < targetCount; cluster++)
		memcpy(centroids + (Size) cluster * mv->dim,
			   PgturbohybridMultiVectorValues(mv, selected[cluster]),
			   sizeof(float) * mv->dim);

	for (int iteration = 0;
		 iteration < (mode == PGTURBOHYBRID_MULTIVECTOR_TOKEN_POOLING_KMEANS ? 4 : 1);
		 iteration++)
	{
		memset(sums, 0, centroidBytes);
		memset(clusterCounts, 0, sizeof(int) * targetCount);
		for (int32 token = 0; token < mv->count; token++)
		{
			const float *values = PgturbohybridMultiVectorValues(mv, token);
			double		bestSimilarity = -DBL_MAX;
			int			bestCluster = 0;

			for (int32 cluster = 0; cluster < targetCount; cluster++)
			{
				float	   *centroid = centroids + (Size) cluster * mv->dim;
				double		dot = 0.0;

				for (int32 dim = 0; dim < mv->dim; dim++)
					dot += (double) values[dim] * (double) centroid[dim];
				if (cluster == 0 || dot > bestSimilarity)
				{
					bestSimilarity = dot;
					bestCluster = cluster;
				}
			}
			clusterCounts[bestCluster]++;
			for (int32 dim = 0; dim < mv->dim; dim++)
				sums[(Size) bestCluster * mv->dim + dim] += values[dim];
		}

		for (int32 cluster = 0; cluster < targetCount; cluster++)
		{
			float	   *centroid = centroids + (Size) cluster * mv->dim;

			if (clusterCounts[cluster] == 0)
			{
				memcpy(centroid,
					   PgturbohybridMultiVectorValues(mv, selected[cluster]),
					   sizeof(float) * mv->dim);
				continue;
			}
			for (int32 dim = 0; dim < mv->dim; dim++)
				centroid[dim] =
					sums[(Size) cluster * mv->dim + dim] /
					(float) clusterCounts[cluster];
			PgturbohybridMultiVectorNormalizeToken(centroid, mv->dim);
		}
	}

	memcpy(pooled->values, centroids, centroidBytes);
	pfree(sums);
	pfree(centroids);
	pfree(nearestSelectedSimilarity);
	pfree(tokenNorms);
	pfree(clusterCounts);
	pfree(isSelected);
	pfree(selected);
	MemoryContextSwitchTo(oldCtx);

	return pooled;
}

int
PgturbohybridMultiVectorCentroidCountForDoc(const PgturbohybridMultiVector *doc,
											int requested)
{
	int			autoCount;

	if (doc == NULL || doc->count <= 0)
		return 0;
	if (requested > 0)
		return Min(requested, doc->count);
	if (doc->count <= 4)
		return doc->count;

	autoCount = (int) ceil(sqrt((double) doc->count));
	autoCount = Max(4, autoCount);
	autoCount = Min(autoCount, 64);
	return Min(autoCount, doc->count);
}

float
PgturbohybridMultiVectorCentroidResidualMean(const PgturbohybridMultiVector *doc,
											 const PgturbohybridMultiVector *centroids)
{
	double		sum = 0.0;

	PgturbohybridCheckSameMultiVectorDims(doc, centroids);
	if (doc->count <= 0 || centroids->count <= 0)
		return 0.0f;

	for (int32 token = 0; token < doc->count; token++)
	{
		const float *values = PgturbohybridMultiVectorValues(doc, token);
		const float *bestCentroid = NULL;
		double		bestSimilarity = -DBL_MAX;

		for (int32 centroid = 0; centroid < centroids->count; centroid++)
		{
			const float *centroidValues =
				PgturbohybridMultiVectorValues(centroids, centroid);
			double		dot = 0.0;

			for (int32 dim = 0; dim < doc->dim; dim++)
				dot += (double) values[dim] * (double) centroidValues[dim];
			if (centroid == 0 || dot > bestSimilarity)
			{
				bestSimilarity = dot;
				bestCentroid = centroidValues;
			}
		}

		if (bestCentroid != NULL)
		{
			double		residual = 0.0;

			for (int32 dim = 0; dim < doc->dim; dim++)
			{
				double		delta =
					(double) values[dim] - (double) bestCentroid[dim];

				residual += delta * delta;
			}
			sum += residual;
		}
	}

	return (float) (sum / (double) doc->count);
}

static uint32
PgturbohybridMultiVectorProxyHash(uint32 a, uint32 b, uint32 salt)
{
	uint32		x = (uint32) 0x9e3779b9U;

	x ^= a + (uint32) 0x85ebca6bU + (x << 6) + (x >> 2);
	x ^= b + (uint32) 0xc2b2ae35U + (x << 6) + (x >> 2);
	x ^= salt + (uint32) 0x27d4eb2dU + (x << 6) + (x >> 2);
	x ^= x >> 16;
	x *= (uint32) 0x7feb352dU;
	x ^= x >> 15;
	x *= (uint32) 0x846ca68bU;
	x ^= x >> 16;
	return x;
}

static bool
PgturbohybridStringIsEmpty(const char *value)
{
	return value == NULL || value[0] == '\0';
}

static void
PgturbohybridLearnedProjectionFreeCache(void)
{
	if (pgturbohybrid_learned_projection_cache == NULL)
		return;

	pfree(pgturbohybrid_learned_projection_cache->path);
	pfree(pgturbohybrid_learned_projection_cache->model);
	pfree(pgturbohybrid_learned_projection_cache->checksum);
	pfree(pgturbohybrid_learned_projection_cache->weights);
	pfree(pgturbohybrid_learned_projection_cache);
	pgturbohybrid_learned_projection_cache = NULL;
}

static void
PgturbohybridValidateLearnedProjectionExpectedMetadata(const PgturbohybridLearnedProjectionWeights *projection)
{
	if (!PgturbohybridStringIsEmpty(pgturbohybrid_multivector_learned_projection_model) &&
		strcmp(projection->model,
			   pgturbohybrid_multivector_learned_projection_model) != 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("learned_projection_v1 model mismatch"),
				 errdetail("Projection file declares model \"%s\", but turbohybrid.multivector_learned_projection_model is \"%s\".",
						   projection->model,
						   pgturbohybrid_multivector_learned_projection_model)));
	if (!PgturbohybridStringIsEmpty(pgturbohybrid_multivector_learned_projection_checksum) &&
		strcmp(projection->checksum,
			   pgturbohybrid_multivector_learned_projection_checksum) != 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("learned_projection_v1 checksum mismatch"),
				 errdetail("Projection file declares checksum \"%s\", but turbohybrid.multivector_learned_projection_checksum is \"%s\".",
						   projection->checksum,
						   pgturbohybrid_multivector_learned_projection_checksum)));
}

static PgturbohybridLearnedProjectionWeights *
PgturbohybridLoadLearnedProjectionWeights(int32 dim)
{
	FILE	   *file;
	char		magic[64];
	char		model[128];
	char		checksum[128];
	int			inputDim;
	int			outputDim;
	Size		weightCount;
	Size		weightBytes;
	MemoryContext oldCtx;
	PgturbohybridLearnedProjectionWeights *loaded;

	if (PgturbohybridStringIsEmpty(pgturbohybrid_multivector_learned_projection_path))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("learned_projection_v1 multivector proxy encoder requires configured projection weights"),
				 errhint("Set turbohybrid.multivector_learned_projection_path to an administrator-provided projection weight file, or use multivector_proxy_encoder = normalized_mean.")));

	if (pgturbohybrid_learned_projection_cache != NULL &&
		strcmp(pgturbohybrid_learned_projection_cache->path,
			   pgturbohybrid_multivector_learned_projection_path) == 0)
	{
		PgturbohybridValidateLearnedProjectionExpectedMetadata(pgturbohybrid_learned_projection_cache);
		if (pgturbohybrid_learned_projection_cache->inputDim != dim ||
			pgturbohybrid_learned_projection_cache->outputDim != dim)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("learned_projection_v1 dimensions do not match multivector dimension"),
					 errdetail("Projection file has input_dim=%d and output_dim=%d, but the multivector dimension is %d.",
							   pgturbohybrid_learned_projection_cache->inputDim,
							   pgturbohybrid_learned_projection_cache->outputDim,
							   dim)));
		return pgturbohybrid_learned_projection_cache;
	}

	file = AllocateFile(pgturbohybrid_multivector_learned_projection_path, "r");
	if (file == NULL)
		ereport(ERROR,
				(errcode_for_file_access(),
				 errmsg("could not open learned_projection_v1 weights file \"%s\": %m",
						pgturbohybrid_multivector_learned_projection_path)));

	if (fscanf(file, "%63s %127s %d %d %127s",
			   magic, model, &inputDim, &outputDim, checksum) != 5)
	{
		FreeFile(file);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("invalid learned_projection_v1 weights header"),
				 errdetail("Expected: pgturbohybrid_learned_projection_v1 <model> <input_dim> <output_dim> <checksum>.")));
	}
	if (strcmp(magic, "pgturbohybrid_learned_projection_v1") != 0)
	{
		FreeFile(file);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
				 errmsg("invalid learned_projection_v1 weights magic \"%s\"",
						magic)));
	}
	if (!PgturbohybridStringIsEmpty(pgturbohybrid_multivector_learned_projection_model) &&
		strcmp(model, pgturbohybrid_multivector_learned_projection_model) != 0)
	{
		FreeFile(file);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("learned_projection_v1 model mismatch"),
				 errdetail("Projection file declares model \"%s\", but turbohybrid.multivector_learned_projection_model is \"%s\".",
						   model,
						   pgturbohybrid_multivector_learned_projection_model)));
	}
	if (!PgturbohybridStringIsEmpty(pgturbohybrid_multivector_learned_projection_checksum) &&
		strcmp(checksum, pgturbohybrid_multivector_learned_projection_checksum) != 0)
	{
		FreeFile(file);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("learned_projection_v1 checksum mismatch"),
				 errdetail("Projection file declares checksum \"%s\", but turbohybrid.multivector_learned_projection_checksum is \"%s\".",
						   checksum,
						   pgturbohybrid_multivector_learned_projection_checksum)));
	}
	if (inputDim <= 0 || outputDim <= 0 ||
		inputDim > PGTURBOHYBRID_MULTIVECTOR_MAX_DIM ||
		outputDim > PGTURBOHYBRID_MULTIVECTOR_MAX_DIM)
	{
		FreeFile(file);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("learned_projection_v1 dimensions are out of range")));
	}
	if (inputDim != dim || outputDim != dim)
	{
		FreeFile(file);
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("learned_projection_v1 dimensions do not match multivector dimension"),
				 errdetail("Projection file has input_dim=%d and output_dim=%d, but the multivector dimension is %d.",
						   inputDim, outputDim, dim),
				 errhint("This first safe slice keeps proxy vectors in the index multivector dimension; use a matching projection or rebuild with a compatible model profile.")));
	}

	weightCount = (Size) inputDim * (Size) outputDim;
	weightBytes = sizeof(float) * weightCount;
	oldCtx = MemoryContextSwitchTo(TopMemoryContext);
	loaded = palloc0(sizeof(PgturbohybridLearnedProjectionWeights));
	loaded->path = pstrdup(pgturbohybrid_multivector_learned_projection_path);
	loaded->model = pstrdup(model);
	loaded->checksum = pstrdup(checksum);
	loaded->inputDim = inputDim;
	loaded->outputDim = outputDim;
	loaded->weightBytes = weightBytes;
	loaded->weights = palloc0(weightBytes);
	MemoryContextSwitchTo(oldCtx);

	for (Size i = 0; i < weightCount; i++)
	{
		double		value;

		if (fscanf(file, "%lf", &value) != 1)
		{
			FreeFile(file);
			PgturbohybridLearnedProjectionFreeCache();
			pfree(loaded->path);
			pfree(loaded->model);
			pfree(loaded->checksum);
			pfree(loaded->weights);
			pfree(loaded);
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_TEXT_REPRESENTATION),
					 errmsg("learned_projection_v1 weights file ended before all weights were read"),
					 errdetail("Expected %zu float weights after the header.",
							   weightCount)));
		}
		loaded->weights[i] = (float) value;
	}
	FreeFile(file);

	PgturbohybridLearnedProjectionFreeCache();
	pgturbohybrid_learned_projection_cache = loaded;
	return loaded;
}

bool
PgturbohybridMultiVectorLearnedProjectionInfo(bool *loaded,
											  int32 *dim,
											  uint64 *weightBytes,
											  const char **model,
											  const char **checksum)
{
	if (loaded != NULL)
		*loaded = pgturbohybrid_learned_projection_cache != NULL;
	if (dim != NULL)
		*dim = pgturbohybrid_learned_projection_cache != NULL ?
			pgturbohybrid_learned_projection_cache->outputDim : 0;
	if (weightBytes != NULL)
		*weightBytes = pgturbohybrid_learned_projection_cache != NULL ?
			(uint64) pgturbohybrid_learned_projection_cache->weightBytes : 0;
	if (model != NULL)
		*model = pgturbohybrid_learned_projection_cache != NULL ?
			pgturbohybrid_learned_projection_cache->model : "";
	if (checksum != NULL)
		*checksum = pgturbohybrid_learned_projection_cache != NULL ?
			pgturbohybrid_learned_projection_cache->checksum : "";
	return pgturbohybrid_learned_projection_cache != NULL;
}

const char *
PgturbohybridMultiVectorProxyEncoderName(int encoder)
{
	switch ((PgturbohybridMultiVectorProxyEncoder) encoder)
	{
		case PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_NORMALIZED_MEAN:
			return "normalized_mean";
		case PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_FIRST_TOKEN:
			return "first_token";
		case PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_MAX_ABS_MEAN:
			return "max_abs_mean";
		case PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_CENTROID_MEAN:
			return "centroid_mean";
		case PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_MAX_POOL:
			return "max_pool";
		case PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_RANDOM_PROJECTION_FDE:
			return "random_projection_fde";
		case PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_LEARNED_PROJECTION_PLACEHOLDER:
			return "learned_projection_placeholder";
		case PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_LEARNED_PROJECTION_V1:
			return "learned_projection_v1";
		case PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_MEAN:
		default:
			return "mean";
	}
}

static void
PgturbohybridMultiVectorProxyMean(const PgturbohybridMultiVector *mv,
								  Vector *vector)
{
	for (int32 token = 0; token < mv->count; token++)
	{
		const float *values = PgturbohybridMultiVectorValues(mv, token);

		for (int32 dim = 0; dim < mv->dim; dim++)
			vector->x[dim] += values[dim];
	}
	for (int32 dim = 0; dim < mv->dim; dim++)
		vector->x[dim] /= (float) mv->count;
}

static void
PgturbohybridMultiVectorProxyNormalizedMean(const PgturbohybridMultiVector *mv,
											Vector *vector)
{
	PgturbohybridMultiVectorProxyMean(mv, vector);
	PgturbohybridMultiVectorNormalizeToken(vector->x, vector->dim);
}

static void
PgturbohybridMultiVectorProxyFirstToken(const PgturbohybridMultiVector *mv,
										Vector *vector)
{
	const float *values = PgturbohybridMultiVectorValues(mv, 0);

	memcpy(vector->x, values, sizeof(float) * (Size) mv->dim);
}

static void
PgturbohybridMultiVectorProxyMaxAbsMean(const PgturbohybridMultiVector *mv,
										Vector *vector)
{
	for (int32 token = 0; token < mv->count; token++)
	{
		const float *values = PgturbohybridMultiVectorValues(mv, token);

		for (int32 dim = 0; dim < mv->dim; dim++)
		{
			if (token == 0 || fabsf(values[dim]) > fabsf(vector->x[dim]))
				vector->x[dim] = values[dim];
		}
	}
}

static void
PgturbohybridMultiVectorProxyLearnedProjectionV1(const PgturbohybridMultiVector *mv,
												 Vector *vector)
{
	PgturbohybridLearnedProjectionWeights *projection;
	float	   *source;

	projection = PgturbohybridLoadLearnedProjectionWeights(mv->dim);
	source = palloc0(sizeof(float) * (Size) mv->dim);

	for (int32 token = 0; token < mv->count; token++)
	{
		const float *values = PgturbohybridMultiVectorValues(mv, token);

		for (int32 dim = 0; dim < mv->dim; dim++)
			source[dim] += values[dim];
	}
	for (int32 dim = 0; dim < mv->dim; dim++)
		source[dim] /= (float) mv->count;
	PgturbohybridMultiVectorNormalizeToken(source, mv->dim);

	for (int32 outDim = 0; outDim < projection->outputDim; outDim++)
	{
		double		sum = 0.0;
		const float *weights =
			projection->weights + (Size) outDim * (Size) projection->inputDim;

		for (int32 inDim = 0; inDim < projection->inputDim; inDim++)
			sum += (double) weights[inDim] * (double) source[inDim];
		vector->x[outDim] = (float) sum;
	}
	PgturbohybridMultiVectorNormalizeToken(vector->x, vector->dim);
	pfree(source);
}

Vector *
PgturbohybridMultiVectorBuildProxyVectorWithCentroids(const PgturbohybridMultiVector *mv,
													  const PgturbohybridMultiVector *centroids,
													  int encoder,
													  int centroidCount,
													  MemoryContext ctx)
{
	Vector	   *vector;
	PgturbohybridMultiVector *localCentroids = NULL;

	PgturbohybridCheckMultiVector(mv);
	if (encoder ==
		PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_LEARNED_PROJECTION_PLACEHOLDER)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("learned multivector proxy projection is not configured"),
				 errhint("Use multivector_proxy_encoder = learned_projection_v1 with turbohybrid.multivector_learned_projection_path, or use normalized_mean, mean, first_token, max_abs_mean, centroid_mean, max_pool, or random_projection_fde.")));

	vector = MemoryContextAllocZero(ctx, PGTURBOHYBRID_VECTOR_SIZE(mv->dim));
	SET_VARSIZE(vector, PGTURBOHYBRID_VECTOR_SIZE(mv->dim));
	vector->dim = mv->dim;

	switch ((PgturbohybridMultiVectorProxyEncoder) encoder)
	{
		case PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_NORMALIZED_MEAN:
			PgturbohybridMultiVectorProxyNormalizedMean(mv, vector);
			break;
		case PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_FIRST_TOKEN:
			PgturbohybridMultiVectorProxyFirstToken(mv, vector);
			break;
		case PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_MAX_ABS_MEAN:
			PgturbohybridMultiVectorProxyMaxAbsMean(mv, vector);
			break;
		case PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_CENTROID_MEAN:
			if (centroids == NULL)
			{
				int			effectiveCount;
				double		targetRatio;

				effectiveCount =
					PgturbohybridMultiVectorCentroidCountForDoc(mv,
																centroidCount);
				if (effectiveCount <= 0)
					elog(ERROR, "invalid multivector centroid count");
				targetRatio = (double) effectiveCount / (double) mv->count;
				localCentroids =
					PgturbohybridMultiVectorPoolDocumentTokens(mv,
															   PGTURBOHYBRID_MULTIVECTOR_TOKEN_POOLING_KMEANS,
															   targetRatio,
															   1,
															   ctx);
				centroids = localCentroids;
			}
			PgturbohybridCheckSameMultiVectorDims(mv, centroids);
			PgturbohybridMultiVectorProxyNormalizedMean(centroids, vector);
			break;
		case PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_MAX_POOL:
			for (int32 dim = 0; dim < mv->dim; dim++)
				vector->x[dim] = -FLT_MAX;
			for (int32 token = 0; token < mv->count; token++)
			{
				const float *values = PgturbohybridMultiVectorValues(mv, token);

				for (int32 dim = 0; dim < mv->dim; dim++)
					vector->x[dim] = Max(vector->x[dim], values[dim]);
			}
			break;
		case PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_RANDOM_PROJECTION_FDE:
			{
				double		scale = 1.0 / sqrt((double) mv->count);

				for (int32 token = 0; token < mv->count; token++)
				{
					const float *values = PgturbohybridMultiVectorValues(mv, token);

					for (int32 dim = 0; dim < mv->dim; dim++)
					{
						uint32		hash =
							PgturbohybridMultiVectorProxyHash((uint32) token,
															  (uint32) dim, 0);
						int32		bucket = (int32) (hash % (uint32) mv->dim);
						float		sign =
							(hash & (uint32) 0x80000000U) ? -1.0f : 1.0f;

						vector->x[bucket] += (float) ((double) sign *
													  (double) values[dim] * scale);
					}
				}
			}
			break;
		case PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_LEARNED_PROJECTION_V1:
			PgturbohybridMultiVectorProxyLearnedProjectionV1(mv, vector);
			break;
		case PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_MEAN:
		default:
			PgturbohybridMultiVectorProxyMean(mv, vector);
			break;
	}

	return vector;
}

Vector *
PgturbohybridMultiVectorBuildProxyVector(const PgturbohybridMultiVector *mv,
										 int encoder,
										 MemoryContext ctx)
{
	return PgturbohybridMultiVectorBuildProxyVectorWithCentroids(mv, NULL,
																 encoder,
																 PGTURBOHYBRID_MULTIVECTOR_CENTROID_COUNT_AUTO,
																 ctx);
}

Vector *
PgturbohybridMultiVectorBuildQueryProxyVector(const PgturbohybridMultiVector *query,
											  int encoder,
											  MemoryContext ctx)
{
	return PgturbohybridMultiVectorBuildProxyVector(query, encoder, ctx);
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

static void
TqDotProductF32BlockScalar(const float *queryValues, const float *docValues,
						   int32 dim, int32 blockCount, double *dots)
{
	for (int32 qi = 0; qi < blockCount; qi++)
	{
		const float *qv = queryValues + ((Size) qi * (Size) dim);

		dots[qi] = TqDotProductF32Scalar(qv, docValues, dim);
	}
}

TqDotProductF32BlockFunc
TqResolveDotProductF32BlockKernel(void)
{
	/*
	 * CPU feature detection is process-stable, so resolve the SIMD kernel once
	 * and cache it. The previous code re-ran the __builtin_cpu_supports ladder
	 * on every call, which is pure overhead in the per-token-block exact-MaxSim
	 * hot loop reached via TqDotProductF32BlockAuto. The force-scalar GUC is
	 * still honored on every call (checked ahead of the cache) so runtime
	 * overrides keep working; the cache only memoizes the hardware probe.
	 */
	static TqDotProductF32BlockFunc cached = NULL;

	if (pgturbohybrid_dense_exact_simd_force == PGTURBOHYBRID_EXACT_SIMD_FORCE_SCALAR)
		return TqDotProductF32BlockScalar;

	if (cached != NULL)
		return cached;

#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX512F
	if (TqMultiVectorAvx512fAvailable())
		return (cached = TqDotProductF32BlockAvx512f);
#endif
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX2
	if (TqMultiVectorAvx2Available())
		return (cached = TqDotProductF32BlockAvx2);
#endif
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_NEON
	return (cached = TqDotProductF32BlockNeon);
#endif
	return (cached = TqDotProductF32BlockScalar);
}

void
TqDotProductF32BlockAuto(const float *queryValues, const float *docValues,
						 int32 dim, int32 blockCount, double *dots)
{
	Assert(blockCount > 0 &&
		   blockCount <= PGTURBOHYBRID_MULTIVECTOR_BLOCKED_SCALAR_Q);
	TqResolveDotProductF32BlockKernel()(queryValues, docValues, dim, blockCount,
										dots);
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
TqResolveMultiVectorMaxSimBlockedKernel(void)
{
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

double
TqMultiVectorMaxSimBlocked(const PgturbohybridMultiVector *query,
						   const PgturbohybridMultiVector *doc)
{
	return TqResolveMultiVectorMaxSimBlockedKernel()(query, doc);
}

static TqMultiVectorMaxSimFunc
TqResolveMultiVectorMaxSimKernel(void)
{
	if (pgturbohybrid_dense_exact_simd_force == PGTURBOHYBRID_EXACT_SIMD_FORCE_SCALAR)
		return TqMultiVectorMaxSimScalar;

	return TqResolveMultiVectorMaxSimBlockedKernel();
}

const char *
TqMultiVectorMaxSimKernelName(void)
{
	TqMultiVectorMaxSimFunc func = TqResolveMultiVectorMaxSimKernel();

	if (func == TqMultiVectorMaxSimScalar)
		return "scalar";
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

int64
TqCompactCodeScoreScalar(const int16 *queryCodes, const int16 *docCodes,
						 int32 count)
{
	int64		score = 0;

	for (int32 i = 0; i < count; i++)
		score += (int64) queryCodes[i] * (int64) docCodes[i];
	return score;
}

void
TqCompactCodeScoreBatchScalar(int16 queryCode, const int16 *docCodes,
							  int32 count, int64 *scores)
{
	for (int32 i = 0; i < count; i++)
		scores[i] = (int64) queryCode * (int64) docCodes[i];
}

#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX2
static inline int64 PGTURBOHYBRID_MULTIVECTOR_AVX2_TARGET
TqCompactCodeHsum256Epi32(__m256i values)
{
	__m128i		low = _mm256_castsi256_si128(values);
	__m128i		high = _mm256_extracti128_si256(values, 1);
	__m128i		sum = _mm_add_epi32(low, high);

	sum = _mm_hadd_epi32(sum, sum);
	sum = _mm_hadd_epi32(sum, sum);
	return (int64) _mm_cvtsi128_si32(sum);
}

static int64 PGTURBOHYBRID_MULTIVECTOR_AVX2_TARGET
TqCompactCodeScoreAvx2(const int16 *queryCodes, const int16 *docCodes,
					   int32 count)
{
	int32		i = 0;
	int64		score = 0;

	for (; i + 16 <= count; i += 16)
	{
		__m256i		q = _mm256_loadu_si256((const __m256i *) (queryCodes + i));
		__m256i		d = _mm256_loadu_si256((const __m256i *) (docCodes + i));
		__m256i		prod = _mm256_madd_epi16(q, d);

		score += TqCompactCodeHsum256Epi32(prod);
	}

	for (; i < count; i++)
		score += (int64) queryCodes[i] * (int64) docCodes[i];
	return score;
}

static void PGTURBOHYBRID_MULTIVECTOR_AVX2_TARGET
TqCompactCodeScoreBatchAvx2(int16 queryCode, const int16 *docCodes,
							int32 count, int64 *scores)
{
	__m256i		q = _mm256_set1_epi32((int32) queryCode);
	int32		i = 0;
	int32		tmp[8];

	for (; i + 8 <= count; i += 8)
	{
		__m128i		packed = _mm_loadu_si128((const __m128i *) (docCodes + i));
		__m256i		d = _mm256_cvtepi16_epi32(packed);
		__m256i		prod = _mm256_mullo_epi32(q, d);

		_mm256_storeu_si256((__m256i *) tmp, prod);
		for (int lane = 0; lane < 8; lane++)
			scores[i + lane] = (int64) tmp[lane];
	}

	for (; i < count; i++)
		scores[i] = (int64) queryCode * (int64) docCodes[i];
}
#endif

#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX512BW
static bool
TqMultiVectorAvx512bwAvailable(void)
{
#if defined(__AVX512BW__) && defined(__AVX512F__)
	return true;
#elif defined(__GNUC__) || defined(__clang__)
	return __builtin_cpu_supports("avx512f") &&
		__builtin_cpu_supports("avx512bw");
#else
	return false;
#endif
}

static int64 PGTURBOHYBRID_MULTIVECTOR_AVX512BW_TARGET
TqCompactCodeScoreAvx512(const int16 *queryCodes, const int16 *docCodes,
						 int32 count)
{
	int32		i = 0;
	int64		score = 0;

	for (; i + 32 <= count; i += 32)
	{
		__m512i		q = _mm512_loadu_si512((const void *) (queryCodes + i));
		__m512i		d = _mm512_loadu_si512((const void *) (docCodes + i));
		__m512i		prod = _mm512_madd_epi16(q, d);

		score += (int64) _mm512_reduce_add_epi32(prod);
	}

	for (; i < count; i++)
		score += (int64) queryCodes[i] * (int64) docCodes[i];
	return score;
}

static void PGTURBOHYBRID_MULTIVECTOR_AVX512BW_TARGET
TqCompactCodeScoreBatchAvx512(int16 queryCode, const int16 *docCodes,
							  int32 count, int64 *scores)
{
	__m512i		q = _mm512_set1_epi32((int32) queryCode);
	int32		i = 0;
	int32		tmp[16];

	for (; i + 16 <= count; i += 16)
	{
		__m256i		packed = _mm256_loadu_si256((const __m256i *) (docCodes + i));
		__m512i		d = _mm512_cvtepi16_epi32(packed);
		__m512i		prod = _mm512_mullo_epi32(q, d);

		_mm512_storeu_si512((void *) tmp, prod);
		for (int lane = 0; lane < 16; lane++)
			scores[i + lane] = (int64) tmp[lane];
	}

	for (; i < count; i++)
		scores[i] = (int64) queryCode * (int64) docCodes[i];
}
#endif

#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_NEON
static int64
TqCompactCodeScoreNeon(const int16 *queryCodes, const int16 *docCodes,
					   int32 count)
{
	int64		score = 0;
	int32		i = 0;

	for (; i + 8 <= count; i += 8)
	{
		int16x8_t	q = vld1q_s16(queryCodes + i);
		int16x8_t	d = vld1q_s16(docCodes + i);
		int32x4_t	prod0 = vmull_s16(vget_low_s16(q), vget_low_s16(d));
		int32x4_t	prod1 = vmull_s16(vget_high_s16(q), vget_high_s16(d));

		score += (int64) vaddvq_s32(prod0);
		score += (int64) vaddvq_s32(prod1);
	}

	for (; i < count; i++)
		score += (int64) queryCodes[i] * (int64) docCodes[i];
	return score;
}

static void
TqCompactCodeScoreBatchNeon(int16 queryCode, const int16 *docCodes,
							int32 count, int64 *scores)
{
	int32		tmp[8];
	int32		i = 0;
	int32x4_t	q = vdupq_n_s32((int32) queryCode);

	for (; i + 8 <= count; i += 8)
	{
		int16x8_t	packed = vld1q_s16(docCodes + i);
		int32x4_t	low = vmovl_s16(vget_low_s16(packed));
		int32x4_t	high = vmovl_s16(vget_high_s16(packed));
		int32x4_t	prod0 = vmulq_s32(q, low);
		int32x4_t	prod1 = vmulq_s32(q, high);

		vst1q_s32(tmp, prod0);
		vst1q_s32(tmp + 4, prod1);
		for (int lane = 0; lane < 8; lane++)
			scores[i + lane] = (int64) tmp[lane];
	}

	for (; i < count; i++)
		scores[i] = (int64) queryCode * (int64) docCodes[i];
}
#endif

TqCompactCodeScoreFunc
TqResolveCompactCodeScoreKernel(const char *forceKernel)
{
	if (forceKernel == NULL || pg_strcasecmp(forceKernel, "auto") == 0)
	{
		if (pgturbohybrid_dense_exact_simd_force ==
			PGTURBOHYBRID_EXACT_SIMD_FORCE_SCALAR)
			return TqCompactCodeScoreScalar;
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX512BW
		if (TqMultiVectorAvx512bwAvailable())
			return TqCompactCodeScoreAvx512;
#endif
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX2
		if (TqMultiVectorAvx2Available())
			return TqCompactCodeScoreAvx2;
#endif
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_NEON
		return TqCompactCodeScoreNeon;
#endif
		return TqCompactCodeScoreScalar;
	}

	if (pg_strcasecmp(forceKernel, "scalar") == 0 ||
		pg_strcasecmp(forceKernel, "compact_scalar") == 0)
		return TqCompactCodeScoreScalar;
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX512BW
	if (pg_strcasecmp(forceKernel, "avx512") == 0 ||
		pg_strcasecmp(forceKernel, "compact_avx512") == 0)
	{
		if (TqMultiVectorAvx512bwAvailable())
			return TqCompactCodeScoreAvx512;
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("compact code scoring kernel \"compact_avx512\" is not available")));
	}
#else
	if (pg_strcasecmp(forceKernel, "avx512") == 0 ||
		pg_strcasecmp(forceKernel, "compact_avx512") == 0)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("compact code scoring kernel \"compact_avx512\" is not available")));
#endif
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX2
	if (pg_strcasecmp(forceKernel, "avx2") == 0 ||
		pg_strcasecmp(forceKernel, "compact_avx2") == 0)
	{
		if (TqMultiVectorAvx2Available())
			return TqCompactCodeScoreAvx2;
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("compact code scoring kernel \"compact_avx2\" is not available")));
	}
#else
	if (pg_strcasecmp(forceKernel, "avx2") == 0 ||
		pg_strcasecmp(forceKernel, "compact_avx2") == 0)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("compact code scoring kernel \"compact_avx2\" is not available")));
#endif
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_NEON
	if (pg_strcasecmp(forceKernel, "neon") == 0 ||
		pg_strcasecmp(forceKernel, "compact_neon") == 0)
		return TqCompactCodeScoreNeon;
#else
	if (pg_strcasecmp(forceKernel, "neon") == 0 ||
		pg_strcasecmp(forceKernel, "compact_neon") == 0)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("compact code scoring kernel \"compact_neon\" is not available")));
#endif

	ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("unknown compact code scoring kernel \"%s\"", forceKernel),
			 errhint("Use auto, scalar, avx2, avx512, or neon.")));
	return TqCompactCodeScoreScalar;
}

const char *
TqCompactCodeScoreKernelName(TqCompactCodeScoreFunc func)
{
	if (func == TqCompactCodeScoreScalar)
		return "compact_scalar";
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX512BW
	if (func == TqCompactCodeScoreAvx512)
		return "compact_avx512";
#endif
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX2
	if (func == TqCompactCodeScoreAvx2)
		return "compact_avx2";
#endif
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_NEON
	if (func == TqCompactCodeScoreNeon)
		return "compact_neon";
#endif
	return "unknown";
}

TqCompactCodeScoreBatchFunc
TqResolveCompactCodeScoreBatchKernel(const char *forceKernel)
{
	if (forceKernel == NULL || pg_strcasecmp(forceKernel, "auto") == 0)
	{
		if (pgturbohybrid_dense_exact_simd_force ==
			PGTURBOHYBRID_EXACT_SIMD_FORCE_SCALAR)
			return TqCompactCodeScoreBatchScalar;
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX512BW
		if (TqMultiVectorAvx512bwAvailable())
			return TqCompactCodeScoreBatchAvx512;
#endif
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX2
		if (TqMultiVectorAvx2Available())
			return TqCompactCodeScoreBatchAvx2;
#endif
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_NEON
		return TqCompactCodeScoreBatchNeon;
#endif
		return TqCompactCodeScoreBatchScalar;
	}

	if (pg_strcasecmp(forceKernel, "scalar") == 0 ||
		pg_strcasecmp(forceKernel, "compact_scalar") == 0)
		return TqCompactCodeScoreBatchScalar;
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX512BW
	if (pg_strcasecmp(forceKernel, "avx512") == 0 ||
		pg_strcasecmp(forceKernel, "compact_avx512") == 0)
	{
		if (TqMultiVectorAvx512bwAvailable())
			return TqCompactCodeScoreBatchAvx512;
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("compact code scoring kernel \"compact_avx512\" is not available")));
	}
#else
	if (pg_strcasecmp(forceKernel, "avx512") == 0 ||
		pg_strcasecmp(forceKernel, "compact_avx512") == 0)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("compact code scoring kernel \"compact_avx512\" is not available")));
#endif
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX2
	if (pg_strcasecmp(forceKernel, "avx2") == 0 ||
		pg_strcasecmp(forceKernel, "compact_avx2") == 0)
	{
		if (TqMultiVectorAvx2Available())
			return TqCompactCodeScoreBatchAvx2;
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("compact code scoring kernel \"compact_avx2\" is not available")));
	}
#else
	if (pg_strcasecmp(forceKernel, "avx2") == 0 ||
		pg_strcasecmp(forceKernel, "compact_avx2") == 0)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("compact code scoring kernel \"compact_avx2\" is not available")));
#endif
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_NEON
	if (pg_strcasecmp(forceKernel, "neon") == 0 ||
		pg_strcasecmp(forceKernel, "compact_neon") == 0)
		return TqCompactCodeScoreBatchNeon;
#else
	if (pg_strcasecmp(forceKernel, "neon") == 0 ||
		pg_strcasecmp(forceKernel, "compact_neon") == 0)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("compact code scoring kernel \"compact_neon\" is not available")));
#endif

	ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("unknown compact code scoring kernel \"%s\"", forceKernel),
			 errhint("Use auto, scalar, avx2, avx512, or neon.")));
	return TqCompactCodeScoreBatchScalar;
}

const char *
TqCompactCodeScoreBatchKernelName(TqCompactCodeScoreBatchFunc func)
{
	if (func == TqCompactCodeScoreBatchScalar)
		return "compact_scalar";
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX512BW
	if (func == TqCompactCodeScoreBatchAvx512)
		return "compact_avx512";
#endif
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX2
	if (func == TqCompactCodeScoreBatchAvx2)
		return "compact_avx2";
#endif
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_NEON
	if (func == TqCompactCodeScoreBatchNeon)
		return "compact_neon";
#endif
	return "unknown";
}

double
TqMultiVectorMaxSim(const PgturbohybridMultiVector *query,
					const PgturbohybridMultiVector *doc)
{
	return TqResolveMultiVectorMaxSimKernel()(query, doc);
}

static double
TqMultiVectorSymmetricMaxSimAverageWithDot(const PgturbohybridMultiVector *a,
										   const PgturbohybridMultiVector *b,
										   TqDotProductF32Func dotProduct,
										   TqDotProductF32BlockFunc blockDotProduct)
{
	PgturbohybridCheckSameMultiVectorDims(a, b);

	return TqMultiVectorSymmetricMaxSimAverageWithDotUnchecked(a, b,
															  dotProduct,
															  blockDotProduct);
}

static double
TqMultiVectorSymmetricMaxSimAverageWithDotUnchecked(const PgturbohybridMultiVector *a,
													const PgturbohybridMultiVector *b,
													TqDotProductF32Func dotProduct,
													TqDotProductF32BlockFunc blockDotProduct)
{
	double		bestA[PGTURBOHYBRID_MULTIVECTOR_MAX_COUNT];
	double		bestB[PGTURBOHYBRID_MULTIVECTOR_MAX_COUNT];
	double		dots[PGTURBOHYBRID_MULTIVECTOR_BLOCKED_SCALAR_Q];
	double		sumA = 0.0;
	double		sumB = 0.0;

	/*
	 * "Unchecked" contract: callers pass multivectors already validated at
	 * ingest (the count bound in TqParseMultiVector), so count fits the fixed
	 * bestA[]/bestB[] stack arrays.  Assert it so a future caller that skips
	 * validation trips in assert-enabled builds instead of smashing the stack.
	 */
	Assert(a->count <= PGTURBOHYBRID_MULTIVECTOR_MAX_COUNT &&
		   b->count <= PGTURBOHYBRID_MULTIVECTOR_MAX_COUNT);

	for (int32 ai = 0; ai < a->count; ai++)
		bestA[ai] = -INFINITY;
	for (int32 bi = 0; bi < b->count; bi++)
		bestB[bi] = -INFINITY;

	for (int32 ab = 0; ab < a->count;
		 ab += PGTURBOHYBRID_MULTIVECTOR_BLOCKED_SCALAR_Q)
	{
		int32		blockCount =
			Min(a->count - ab, PGTURBOHYBRID_MULTIVECTOR_BLOCKED_SCALAR_Q);
		const float *aValues = a->values + ((Size) ab * (Size) a->dim);

		for (int32 bi = 0; bi < b->count; bi++)
		{
			const float *bv = b->values + ((Size) bi * (Size) b->dim);

			if (blockCount == 1)
				dots[0] = dotProduct(aValues, bv, a->dim);
			else
				blockDotProduct(aValues, bv, a->dim, blockCount, dots);

			for (int32 ai = 0; ai < blockCount; ai++)
			{
				double		dot = dots[ai];

				if (dot > bestA[ab + ai])
					bestA[ab + ai] = dot;
				if (dot > bestB[bi])
					bestB[bi] = dot;
			}
		}
	}

	for (int32 ai = 0; ai < a->count; ai++)
		sumA += bestA[ai];
	for (int32 bi = 0; bi < b->count; bi++)
		sumB += bestB[bi];

	return 0.5 * ((sumA / (double) a->count) + (sumB / (double) b->count));
}

double
TqMultiVectorSymmetricMaxSimAverage(const PgturbohybridMultiVector *a,
									const PgturbohybridMultiVector *b)
{
	if (pgturbohybrid_dense_exact_simd_force == PGTURBOHYBRID_EXACT_SIMD_FORCE_SCALAR)
		return TqMultiVectorSymmetricMaxSimAverageWithDot(a, b,
														  TqDotProductF32Scalar,
														  TqDotProductF32BlockScalar);

#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX512F
	if (TqMultiVectorAvx512fAvailable())
		return TqMultiVectorSymmetricMaxSimAverageWithDot(a, b,
														  TqDotProductF32Avx512f,
														  TqDotProductF32BlockAvx512f);
#endif
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX2
	if (TqMultiVectorAvx2Available())
		return TqMultiVectorSymmetricMaxSimAverageWithDot(a, b,
														  TqDotProductF32Avx2,
														  TqDotProductF32BlockAvx2);
#endif
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_NEON
	return TqMultiVectorSymmetricMaxSimAverageWithDot(a, b,
													  TqDotProductF32Neon,
													  TqDotProductF32BlockNeon);
#endif
	return TqMultiVectorSymmetricMaxSimAverageWithDot(a, b,
													  TqDotProductF32Scalar,
													  TqDotProductF32BlockScalar);
}

double
TqMultiVectorSymmetricMaxSimAverageUnchecked(const PgturbohybridMultiVector *a,
											 const PgturbohybridMultiVector *b)
{
	if (pgturbohybrid_dense_exact_simd_force == PGTURBOHYBRID_EXACT_SIMD_FORCE_SCALAR)
		return TqMultiVectorSymmetricMaxSimAverageWithDotUnchecked(a, b,
																   TqDotProductF32Scalar,
																   TqDotProductF32BlockScalar);

#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX512F
	if (TqMultiVectorAvx512fAvailable())
		return TqMultiVectorSymmetricMaxSimAverageWithDotUnchecked(a, b,
																   TqDotProductF32Avx512f,
																   TqDotProductF32BlockAvx512f);
#endif
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_AVX2
	if (TqMultiVectorAvx2Available())
		return TqMultiVectorSymmetricMaxSimAverageWithDotUnchecked(a, b,
																   TqDotProductF32Avx2,
																   TqDotProductF32BlockAvx2);
#endif
#if PGTURBOHYBRID_MULTIVECTOR_COMPILE_NEON
	return TqMultiVectorSymmetricMaxSimAverageWithDotUnchecked(a, b,
															   TqDotProductF32Neon,
															   TqDotProductF32BlockNeon);
#endif
	return TqMultiVectorSymmetricMaxSimAverageWithDotUnchecked(a, b,
															   TqDotProductF32Scalar,
															   TqDotProductF32BlockScalar);
}

static double
TqMultiVectorMaxSimTokenRangeWeighted(const PgturbohybridMultiVector *query,
									  const PgturbohybridMultiVector *doc,
									  int32 startToken, int32 endToken,
									  const float4 *queryWeights,
									  const bool *queryMask)
{
	double		score = 0.0;

	PgturbohybridCheckSameMultiVectorDims(query, doc);
	if (startToken < 0 || endToken > doc->count || startToken >= endToken)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("invalid multivector context token range")));

	for (int32 qi = 0; qi < query->count; qi++)
	{
		const float *qv = query->values + ((Size) qi * (Size) query->dim);
		double		weight = queryWeights != NULL ? (double) queryWeights[qi] : 1.0;
		double		best = -INFINITY;

		if (queryMask != NULL && queryMask[qi])
			continue;
		if (!isfinite(weight) || weight < 0.0)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("query token weights must be finite non-negative values")));
		if (weight == 0.0)
			continue;

		for (int32 di = startToken; di < endToken; di++)
		{
			const float *dv = doc->values + ((Size) di * (Size) doc->dim);
			double		dot = TqDotProductF32Scalar(qv, dv, query->dim);

			if (dot > best)
				best = dot;
		}
		score += weight * best;
	}
	return score;
}

static double
TqMultiVectorMaxSimTokenRange(const PgturbohybridMultiVector *query,
							  const PgturbohybridMultiVector *doc,
							  int32 startToken, int32 endToken)
{
	return TqMultiVectorMaxSimTokenRangeWeighted(query, doc, startToken,
												 endToken, NULL, NULL);
}

double
TqMultiVectorMaxSimWeighted(const PgturbohybridMultiVector *query,
							const PgturbohybridMultiVector *doc,
							const float4 *queryWeights,
							const bool *queryMask)
{
	if (queryWeights == NULL && queryMask == NULL)
		return TqMultiVectorMaxSim(query, doc);
	return TqMultiVectorMaxSimTokenRangeWeighted(query, doc, 0, doc->count,
												 queryWeights, queryMask);
}

double
TqMultiVectorMaxSimContextLevel(const PgturbohybridMultiVector *query,
								const PgturbohybridMultiVector *doc)
{
	const int32 *offsets;
	int32		contextCount;
	double		best = -INFINITY;

	PgturbohybridCheckSameMultiVectorDims(query, doc);
	if (!PgturbohybridMultiVectorHasContexts(doc))
		return TqMultiVectorMaxSim(query, doc);

	offsets = PgturbohybridMultiVectorContextOffsets(doc);
	contextCount = PgturbohybridMultiVectorContextCount(doc);
	for (int32 ci = 0; ci < contextCount; ci++)
	{
		int32		start = offsets[ci];
		int32		end = (ci + 1 < contextCount) ? offsets[ci + 1] : doc->count;
		double		score =
			TqMultiVectorMaxSimTokenRange(query, doc, start, end);

		if (score > best)
			best = score;
	}
	return best;
}

double
TqMultiVectorMaxSimContextLevelWeighted(const PgturbohybridMultiVector *query,
										const PgturbohybridMultiVector *doc,
										const float4 *queryWeights,
										const bool *queryMask)
{
	const int32 *offsets;
	int32		contextCount;
	double		best = -INFINITY;

	if (queryWeights == NULL && queryMask == NULL)
		return TqMultiVectorMaxSimContextLevel(query, doc);

	PgturbohybridCheckSameMultiVectorDims(query, doc);
	if (!PgturbohybridMultiVectorHasContexts(doc))
		return TqMultiVectorMaxSimWeighted(query, doc, queryWeights, queryMask);

	offsets = PgturbohybridMultiVectorContextOffsets(doc);
	contextCount = PgturbohybridMultiVectorContextCount(doc);
	for (int32 ci = 0; ci < contextCount; ci++)
	{
		int32		start = offsets[ci];
		int32		end = (ci + 1 < contextCount) ? offsets[ci + 1] : doc->count;
		double		score =
			TqMultiVectorMaxSimTokenRangeWeighted(query, doc, start, end,
												  queryWeights, queryMask);

		if (score > best)
			best = score;
	}
	return best;
}

double
TqMultiVectorMaxSimFieldWeighted(const PgturbohybridMultiVector *query,
								 const PgturbohybridMultiVector *doc,
								 const int32 *fieldIds,
								 const float4 *weights,
								 int fieldCount)
{
	const int32 *offsets;
	const int32 *docFields;
	int32		contextCount;
	double		total = 0.0;

	PgturbohybridCheckSameMultiVectorDims(query, doc);
	if (fieldIds == NULL || weights == NULL || fieldCount <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("field weighted MaxSim requires at least one field weight")));

	if (!PgturbohybridMultiVectorHasContexts(doc))
	{
		for (int i = 0; i < fieldCount; i++)
		{
			if (fieldIds[i] == 0)
				return (double) weights[i] * TqMultiVectorMaxSim(query, doc);
		}
		return 0.0;
	}

	offsets = PgturbohybridMultiVectorContextOffsets(doc);
	docFields = PgturbohybridMultiVectorContextFields(doc);
	contextCount = PgturbohybridMultiVectorContextCount(doc);
	for (int wi = 0; wi < fieldCount; wi++)
	{
		double		fieldBest = -INFINITY;

		if (!isfinite(weights[wi]))
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("field weights must be finite")));
		for (int32 ci = 0; ci < contextCount; ci++)
		{
			int32		docField = docFields != NULL ? docFields[ci] : 0;
			int32		start;
			int32		end;
			double		score;

			if (docField != fieldIds[wi])
				continue;
			start = offsets[ci];
			end = (ci + 1 < contextCount) ? offsets[ci + 1] : doc->count;
			score = TqMultiVectorMaxSimTokenRange(query, doc, start, end);
			if (score > fieldBest)
				fieldBest = score;
		}
		if (fieldBest > -INFINITY)
			total += (double) weights[wi] * fieldBest;
	}
	return total;
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
	TqMvSkipSpaces(&cursor);
	if (*cursor == ',')
	{
		int32	   *contextOffsets = NULL;
		int32	   *fieldIds = NULL;
		int32		contextCount = 0;
		bool		hasFields = false;

		cursor++;
		if (!TqMvConsumeLiteral(&cursor, "contexts"))
			TqMvInputError("Expected contexts field.");
		TqMvExpectChar(&cursor, '=');
		TqMvExpectChar(&cursor, '[');
		for (;;)
		{
			int32		offset = TqMvParseInt32(&cursor, "contexts");

			contextOffsets = contextOffsets == NULL ?
				palloc(sizeof(int32) * (Size) (contextCount + 1)) :
				repalloc(contextOffsets,
						 sizeof(int32) * (Size) (contextCount + 1));
			contextOffsets[contextCount++] = offset;
			TqMvSkipSpaces(&cursor);
			if (*cursor != ',')
				break;
			cursor++;
		}
		TqMvExpectChar(&cursor, ']');
		TqMvSkipSpaces(&cursor);
		if (*cursor == ',')
		{
			int32		fieldCount = 0;

			cursor++;
			if (!TqMvConsumeLiteral(&cursor, "fields"))
				TqMvInputError("Expected fields field.");
			TqMvExpectChar(&cursor, '=');
			TqMvExpectChar(&cursor, '[');
			for (;;)
			{
				int32		fieldId = TqMvParseInt32(&cursor, "fields");

				fieldIds = fieldIds == NULL ?
					palloc(sizeof(int32) * (Size) (fieldCount + 1)) :
					repalloc(fieldIds,
							 sizeof(int32) * (Size) (fieldCount + 1));
				fieldIds[fieldCount++] = fieldId;
				TqMvSkipSpaces(&cursor);
				if (*cursor != ',')
					break;
				cursor++;
			}
			TqMvExpectChar(&cursor, ']');
			if (fieldCount != contextCount)
				TqMvInputError("fields count must match contexts count.");
			hasFields = true;
		}

		PgturbohybridMultiVectorValidateContexts(result, contextCount,
												 contextOffsets, fieldIds);
		resultSize =
			PgturbohybridMultiVectorExtendedSize(count, dim, contextCount,
												 hasFields);
		result = repalloc(result, resultSize);
		SET_VARSIZE(result, resultSize);
		result->flags = PGTURBOHYBRID_MULTIVECTOR_FLAG_CONTEXTS;
		if (hasFields)
			result->flags |= PGTURBOHYBRID_MULTIVECTOR_FLAG_FIELDS;
		*PgturbohybridMultiVectorContextCountPtr(result) = contextCount;
		memcpy(PgturbohybridMultiVectorMutableContextOffsets(result),
			   contextOffsets, sizeof(int32) * (Size) contextCount);
		if (hasFields)
			memcpy(PgturbohybridMultiVectorMutableContextFields(result),
				   fieldIds, sizeof(int32) * (Size) contextCount);
	}
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
	const int32 *contextOffsets = PgturbohybridMultiVectorContextOffsets(mv);
	const int32 *fieldIds = PgturbohybridMultiVectorContextFields(mv);
	int32		contextCount = PgturbohybridMultiVectorContextCount(mv);

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

	appendStringInfoChar(&buf, ']');
	if (contextOffsets != NULL)
	{
		appendStringInfoString(&buf, ",contexts=[");
		for (int32 i = 0; i < contextCount; i++)
		{
			if (i > 0)
				appendStringInfoChar(&buf, ',');
			appendStringInfo(&buf, "%d", contextOffsets[i]);
		}
		appendStringInfoChar(&buf, ']');
		if (fieldIds != NULL)
		{
			appendStringInfoString(&buf, ",fields=[");
			for (int32 i = 0; i < contextCount; i++)
			{
				if (i > 0)
					appendStringInfoChar(&buf, ',');
				appendStringInfo(&buf, "%d", fieldIds[i]);
			}
			appendStringInfoChar(&buf, ']');
		}
	}
	appendStringInfoChar(&buf, ')');
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
	if (formatVersion != PGTURBOHYBRID_MULTIVECTOR_BINARY_VERSION &&
		formatVersion != PGTURBOHYBRID_MULTIVECTOR_BINARY_VERSION_CONTEXTS)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("unsupported turbohybrid_multivector binary format version %d",
						formatVersion)));

	dim = (int32) pq_getmsgint(buf, 4);
	count = (int32) pq_getmsgint(buf, 4);
	flags = (uint32) pq_getmsgint(buf, 4);
	floatCount = PgturbohybridMultiVectorFloatCount(count, dim);
	if ((flags & PGTURBOHYBRID_MULTIVECTOR_FLAG_CONTEXTS) != 0)
	{
		int32		contextCount = (int32) pq_getmsgint(buf, 4);
		bool		hasFields =
			(flags & PGTURBOHYBRID_MULTIVECTOR_FLAG_FIELDS) != 0;

		if (formatVersion < PGTURBOHYBRID_MULTIVECTOR_BINARY_VERSION_CONTEXTS)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("context-aware multivector requires binary format version 2")));
		resultSize =
			PgturbohybridMultiVectorExtendedSize(count, dim, contextCount,
												 hasFields);
		result = (PgturbohybridMultiVector *) palloc0(resultSize);
		SET_VARSIZE(result, resultSize);
		result->dim = dim;
		result->count = count;
		result->flags = flags;
		*PgturbohybridMultiVectorContextCountPtr(result) = contextCount;
		for (int32 i = 0; i < contextCount; i++)
			PgturbohybridMultiVectorMutableContextOffsets(result)[i] =
				(int32) pq_getmsgint(buf, 4);
		if (hasFields)
		{
			for (int32 i = 0; i < contextCount; i++)
				PgturbohybridMultiVectorMutableContextFields(result)[i] =
					(int32) pq_getmsgint(buf, 4);
		}
	}
	else
	{
		resultSize = PgturbohybridMultiVectorSize(count, dim);
		result = (PgturbohybridMultiVector *) palloc0(resultSize);
		SET_VARSIZE(result, resultSize);
		result->dim = dim;
		result->count = count;
		result->flags = flags;
	}

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
	bool		hasContexts =
		(mv->flags & PGTURBOHYBRID_MULTIVECTOR_FLAG_CONTEXTS) != 0;

	pq_begintypsend(&buf);
	pq_sendint32(&buf, hasContexts ?
				 PGTURBOHYBRID_MULTIVECTOR_BINARY_VERSION_CONTEXTS :
				 PGTURBOHYBRID_MULTIVECTOR_BINARY_VERSION);
	pq_sendint32(&buf, (uint32) mv->dim);
	pq_sendint32(&buf, (uint32) mv->count);
	pq_sendint32(&buf, mv->flags);
	if (hasContexts)
	{
		int32		contextCount = PgturbohybridMultiVectorContextCount(mv);
		const int32 *contextOffsets =
			PgturbohybridMultiVectorContextOffsets(mv);
		const int32 *fieldIds = PgturbohybridMultiVectorContextFields(mv);

		pq_sendint32(&buf, contextCount);
		for (int32 i = 0; i < contextCount; i++)
			pq_sendint32(&buf, contextOffsets[i]);
		if (fieldIds != NULL)
		{
			for (int32 i = 0; i < contextCount; i++)
				pq_sendint32(&buf, fieldIds[i]);
		}
	}
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
	int32		dim = PG_GETARG_INT32(1);
	PgturbohybridMultiVector *result;

	if (PG_ARGISNULL(0))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("multivector values array cannot be null")));

	array = PG_GETARG_ARRAYTYPE_P(0);
	result = PgturbohybridMultiVectorBuildFromFlatArray(array, dim, 0, NULL,
														NULL);
	PG_RETURN_PGTURBOHYBRID_MULTIVECTOR_P(result);
}

Datum
pgturbohybrid_multivector_from_contexts(PG_FUNCTION_ARGS)
{
	ArrayType  *valuesArray;
	ArrayType  *contextsArray;
	int32		dim = PG_GETARG_INT32(1);
	int		   contextCount;
	int32	   *contextOffsets;
	PgturbohybridMultiVector *result;

	if (PG_ARGISNULL(0) || PG_ARGISNULL(2))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("multivector values and context offsets cannot be null")));

	valuesArray = PG_GETARG_ARRAYTYPE_P(0);
	contextsArray = PG_GETARG_ARRAYTYPE_P(2);
	contextOffsets =
		PgturbohybridMultiVectorReadInt4Array(contextsArray,
											  "context offsets",
											  &contextCount);
	result = PgturbohybridMultiVectorBuildFromFlatArray(valuesArray, dim,
														contextCount,
														contextOffsets,
														NULL);
	PG_RETURN_PGTURBOHYBRID_MULTIVECTOR_P(result);
}

Datum
pgturbohybrid_multivector_from_contexts_and_fields(PG_FUNCTION_ARGS)
{
	ArrayType  *valuesArray;
	ArrayType  *contextsArray;
	ArrayType  *fieldsArray;
	int32		dim = PG_GETARG_INT32(1);
	int			contextCount;
	int			fieldCount;
	int32	   *contextOffsets;
	int32	   *fieldIds;
	PgturbohybridMultiVector *result;

	if (PG_ARGISNULL(0) || PG_ARGISNULL(2) || PG_ARGISNULL(3))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("multivector values, context offsets, and field ids cannot be null")));

	valuesArray = PG_GETARG_ARRAYTYPE_P(0);
	contextsArray = PG_GETARG_ARRAYTYPE_P(2);
	fieldsArray = PG_GETARG_ARRAYTYPE_P(3);
	contextOffsets =
		PgturbohybridMultiVectorReadInt4Array(contextsArray,
											  "context offsets",
											  &contextCount);
	fieldIds =
		PgturbohybridMultiVectorReadInt4Array(fieldsArray, "field ids",
											  &fieldCount);
	if (fieldCount != contextCount)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("field id count must match context offset count")));

	result = PgturbohybridMultiVectorBuildFromFlatArray(valuesArray, dim,
														contextCount,
														contextOffsets,
														fieldIds);
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
pgturbohybrid_multivector_context_count(PG_FUNCTION_ARGS)
{
	PgturbohybridMultiVector *mv = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(0);

	PG_RETURN_INT32(PgturbohybridMultiVectorContextCount(mv));
}

Datum
pgturbohybrid_multivector_context_offsets(PG_FUNCTION_ARGS)
{
	PgturbohybridMultiVector *mv = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(0);
	const int32 *offsets = PgturbohybridMultiVectorContextOffsets(mv);
	int32		contextCount = PgturbohybridMultiVectorContextCount(mv);
	Datum	   *elements;
	ArrayType  *array;

	elements = palloc(sizeof(Datum) * (Size) contextCount);
	if (offsets == NULL)
		elements[0] = Int32GetDatum(0);
	else
	{
		for (int32 i = 0; i < contextCount; i++)
			elements[i] = Int32GetDatum(offsets[i]);
	}
	array = construct_array(elements, contextCount, INT4OID, sizeof(int32),
							true, TYPALIGN_INT);
	PG_RETURN_ARRAYTYPE_P(array);
}

Datum
pgturbohybrid_multivector_field_ids(PG_FUNCTION_ARGS)
{
	PgturbohybridMultiVector *mv = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(0);
	const int32 *fields = PgturbohybridMultiVectorContextFields(mv);
	int32		contextCount = PgturbohybridMultiVectorContextCount(mv);
	Datum	   *elements;
	ArrayType  *array;

	elements = palloc(sizeof(Datum) * (Size) contextCount);
	for (int32 i = 0; i < contextCount; i++)
		elements[i] = Int32GetDatum(fields != NULL ? fields[i] : 0);
	array = construct_array(elements, contextCount, INT4OID, sizeof(int32),
							true, TYPALIGN_INT);
	PG_RETURN_ARRAYTYPE_P(array);
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
pgturbohybrid_multivector_context_maxsim(PG_FUNCTION_ARGS)
{
	PgturbohybridMultiVector *query = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(0);
	PgturbohybridMultiVector *doc = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(1);

	PG_RETURN_FLOAT8(TqMultiVectorMaxSimContextLevel(query, doc));
}

Datum
pgturbohybrid_multivector_field_weighted_maxsim(PG_FUNCTION_ARGS)
{
	PgturbohybridMultiVector *query = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(0);
	PgturbohybridMultiVector *doc = PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(1);
	ArrayType  *fieldArray;
	ArrayType  *weightArray;
	int			fieldCount;
	int			weightCount;
	int32	   *fieldIds;
	float4	   *weights;

	if (PG_ARGISNULL(2) || PG_ARGISNULL(3))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("field ids and weights cannot be null")));

	fieldArray = PG_GETARG_ARRAYTYPE_P(2);
	weightArray = PG_GETARG_ARRAYTYPE_P(3);
	fieldIds =
		PgturbohybridMultiVectorReadInt4Array(fieldArray, "field ids",
											  &fieldCount);
	weights =
		PgturbohybridMultiVectorReadFloat4Array(weightArray, "field weights",
												&weightCount);
	if (fieldCount != weightCount)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("field id count must match field weight count")));

	PG_RETURN_FLOAT8(TqMultiVectorMaxSimFieldWeighted(query, doc, fieldIds,
													  weights, fieldCount));
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
pgturbohybrid_experimental_compact_code_score(PG_FUNCTION_ARGS)
{
	ArrayType  *queryArray;
	ArrayType  *docArray;
	bool		experimental;
	char	   *forceKernel = "auto";
	int			queryCount;
	int			docCount;
	int16	   *queryCodes;
	int16	   *docCodes;
	TqCompactCodeScoreFunc scorer;
	const char *kernelName;
	instr_time	start;
	instr_time	elapsed;
	int64		score;
	int64		scoreUs;
	PgturbohybridJsonbState state;

	if (PG_ARGISNULL(0) || PG_ARGISNULL(1))
		ereport(ERROR,
				(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
				 errmsg("compact code arrays cannot be null")));
	experimental = !PG_ARGISNULL(2) && PG_GETARG_BOOL(2);
	if (!experimental)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("experimental compact-code scoring is disabled"),
				 errhint("Pass experimental => true to call this diagnostic prototype. It is not used for final SQL ordering.")));
	if (!PG_ARGISNULL(3))
		forceKernel = text_to_cstring(PG_GETARG_TEXT_PP(3));

	queryArray = PG_GETARG_ARRAYTYPE_P(0);
	docArray = PG_GETARG_ARRAYTYPE_P(1);
	queryCodes =
		PgturbohybridMultiVectorReadInt2Array(queryArray, "query compact codes",
											  &queryCount);
	docCodes =
		PgturbohybridMultiVectorReadInt2Array(docArray, "document compact codes",
											  &docCount);
	if (queryCount != docCount)
		ereport(ERROR,
				(errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
				 errmsg("query and document compact code arrays must have the same length")));

	scorer = TqResolveCompactCodeScoreKernel(forceKernel);
	kernelName = TqCompactCodeScoreKernelName(scorer);
	INSTR_TIME_SET_CURRENT(start);
	score = scorer(queryCodes, docCodes, (int32) queryCount);
	INSTR_TIME_SET_CURRENT(elapsed);
	INSTR_TIME_SUBTRACT(elapsed, start);
	scoreUs = (int64) INSTR_TIME_GET_MICROSEC(elapsed);

	PgturbohybridJsonbStateInit(&state);
	PgturbohybridJsonbBeginObject(&state);
	PgturbohybridMultiVectorModelJsonbAddBool(&state, "experimental", true);
	PgturbohybridMultiVectorModelJsonbAddString(&state,
												"approximate_scoring_kernel",
												kernelName);
	PgturbohybridMultiVectorModelJsonbAddString(&state, "requested_kernel",
												forceKernel);
	PgturbohybridMultiVectorModelJsonbAddInt64(&state, "code_count",
											   queryCount);
	PgturbohybridMultiVectorModelJsonbAddFloat8(&state, "score",
												(double) score);
	PgturbohybridMultiVectorModelJsonbAddInt64(&state,
											   "approximate_scoring_us",
											   scoreUs);
	PgturbohybridMultiVectorModelJsonbAddString(&state,
												"final_sql_ordering",
												"exact_heap_maxsim");
	PG_RETURN_JSONB_P(PgturbohybridJsonbEndObject(&state));
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

	PG_RETURN_FLOAT8(-TqMultiVectorMaxSimWeighted(queryMv, doc,
												  PgturbohybridQueryGetTokenWeights(query),
												  PgturbohybridQueryGetTokenMask(query)));
}
