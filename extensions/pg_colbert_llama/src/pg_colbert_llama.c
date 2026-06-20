#include "postgres.h"

#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <sys/stat.h>

#include "access/htup_details.h"
#include "catalog/pg_extension.h"
#include "catalog/pg_type.h"
#include "catalog/pg_type_d.h"
#include "commands/extension.h"
#include "fmgr.h"
#include "lib/stringinfo.h"
#include "nodes/makefuncs.h"
#include "parser/parse_func.h"
#include "utils/array.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/jsonb.h"
#include "utils/lsyscache.h"
#include "utils/memutils.h"
#include "utils/numeric.h"
#include "utils/syscache.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

#include "colbert_engine.h"

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(pg_colbert_llama_colbert);
PG_FUNCTION_INFO_V1(pg_colbert_llama_colbert_vectors);
PG_FUNCTION_INFO_V1(pg_colbert_llama_colbert_float4);
PG_FUNCTION_INFO_V1(pg_colbert_llama_colbert_dim);
PG_FUNCTION_INFO_V1(pg_colbert_llama_colbert_mv);
PG_FUNCTION_INFO_V1(pg_colbert_llama_colbert_mv_batch);
PG_FUNCTION_INFO_V1(pg_colbert_llama_colbert_debug);
PG_FUNCTION_INFO_V1(pg_colbert_llama_colbert_model_info);
PG_FUNCTION_INFO_V1(pg_colbert_llama_llama_embed);
PG_FUNCTION_INFO_V1(pg_colbert_llama_llama_embed_vector);
PG_FUNCTION_INFO_V1(pg_colbert_llama_llama_embed_vector_batch);
PG_FUNCTION_INFO_V1(pg_colbert_llama_llama_embed_tokens);
PG_FUNCTION_INFO_V1(pg_colbert_llama_llama_embed_mv);
PG_FUNCTION_INFO_V1(pg_colbert_llama_llama_embed_mv_batch);
PG_FUNCTION_INFO_V1(pg_colbert_llama_llama_embed_sparse);
PG_FUNCTION_INFO_V1(pg_colbert_llama_llama_embed_sparse_batch);
PG_FUNCTION_INFO_V1(pg_colbert_llama_llama_embed_model_info);

#define PG_COLBERT_LLAMA_VECTOR_SIZE(_dim) \
	(offsetof(PgColbertVector, x) + sizeof(float4) * (Size) (_dim))

typedef struct PgColbertVector
{
	int32		vl_len_;
	int16		dim;
	int16		unused;
	float4		x[FLEXIBLE_ARRAY_MEMBER];
} PgColbertVector;

static char *pg_colbert_llama_model_dir = NULL;
static int	pg_colbert_llama_threads = 4;
static int	pg_colbert_llama_n_ctx = 512;
static int	pg_colbert_llama_n_batch = 512;
static int	pg_colbert_llama_batch_sequences = 8;
static int	pg_colbert_llama_n_gpu_layers = 0;
static int	pg_colbert_llama_cache_size = 2;
static char *pg_colbert_llama_query_prefix = NULL;
static char *pg_colbert_llama_document_prefix = NULL;
static int	pg_colbert_llama_max_query_vectors = 32;
static int	pg_colbert_llama_max_doc_vectors = 256;
static int	pg_colbert_llama_query_length = 32;
static bool pg_colbert_llama_strict_profile = false;
static bool pg_colbert_llama_require_normalized = true;
static bool pg_colbert_llama_log_timing = false;
static int	pg_colbert_llama_expected_dim = 128;
static char *pg_colbert_llama_allowed_models = NULL;
static Oid	pg_colbert_llama_vector_type_oid = InvalidOid;
static Oid	pg_colbert_llama_sparse_type_oid = InvalidOid;

void		_PG_init(void);

static Oid PgColbertExtensionSchema(Oid extensionOid);
static Oid PgColbertVectorTypeOid(void);
static Oid PgColbertSparseVectorTypeOid(void);
static Datum PgLlamaEmbedSparseBuild(const PgColbertEngineOutput *output,
									 Jsonb *options);
static Oid PgColbertLookupPgturbohybridFunction(const char *funcname,
												int nargs, Oid *argtypes,
												bool missing_ok);
static void PgColbertParseModel(text *modelText, PgColbertModelSpec *spec);
static void PgLlamaEmbedParseModel(text *modelText, Jsonb *options,
								   PgLlamaEmbedOutputMode defaultMode,
								   PgColbertModelSpec *spec);
static void PgColbertCheckAllowedModel(const char *alias);
static void PgColbertLoadRuntimeProfile(PgColbertModelSpec *spec);
static void PgColbertBuildGucFallbackProfile(PgColbertModelSpec *spec);
static void PgColbertValidateOutput(const PgColbertModelSpec *spec,
									const PgColbertEngineOutput *output);
static void PgColbertEncodeOrError(text *modelText, text *inputText,
								   MemoryContext ctx,
								   PgColbertModelSpec *spec,
								   PgColbertEngineOutput *output,
								   bool debugTokens);
static void PgLlamaEmbedEncodeOrError(text *modelText, text *inputText,
									  Jsonb *options,
									  PgLlamaEmbedOutputMode defaultMode,
									  MemoryContext ctx,
									  PgColbertModelSpec *spec,
									  PgColbertEngineOutput *output);
static int32 PgColbertEncodeBatchOrError(text *modelText, ArrayType *inputsArray,
										 MemoryContext ctx,
										 PgColbertModelSpec *spec,
										 PgColbertEngineOutput **outputs,
										 bool debugTokens);
static int32 PgLlamaEmbedEncodeBatchOrError(text *modelText,
											ArrayType *inputsArray,
											Jsonb *options,
											PgLlamaEmbedOutputMode defaultMode,
											MemoryContext ctx,
											PgColbertModelSpec *spec,
											PgColbertEngineOutput **outputs);
static PgColbertVector *PgColbertMakeVector(const float4 *values, int32 dim);
static ArrayType *PgColbertBuildVectorArray(const PgColbertEngineOutput *output);
static ArrayType *PgColbertBuildFloat4Array(const PgColbertEngineOutput *output);
static Datum PgColbertBuildMultiVector(const PgColbertEngineOutput *output);
static ArrayType *PgColbertBuildMultiVectorArray(PgColbertEngineOutput *outputs,
												 int32 outputCount,
												 Oid multivectorOid);
static void PgColbertAppendJsonString(StringInfo buf, const char *value);
static const char *PgLlamaEmbedOutputModeName(PgLlamaEmbedOutputMode mode);
static const char *PgLlamaEmbedPoolingName(PgLlamaEmbedPooling pooling);
static void PgColbertAppendOutputJson(StringInfo buf,
									  const PgColbertModelSpec *spec,
									  const PgColbertEngineOutput *output,
									  bool includeVectors);
static void PgColbertAppendDebugJson(StringInfo buf,
									 const PgColbertModelSpec *spec,
									 const PgColbertEngineOutput *output);
static void PgColbertAppendModelInfoJson(StringInfo buf,
										 const PgColbertModelSpec *spec,
										 const PgColbertEngineModelInfo *info);
static Datum PgColbertJsonbFromCString(const char *json);

void
_PG_init(void)
{
	DefineCustomStringVariable("pg_colbert_llama.model_dir",
							   "Directory containing admin-installed ColBERT GGUF model files.",
							   NULL,
							   &pg_colbert_llama_model_dir,
							   "/var/lib/postgresql/colbert-models",
							   PGC_SUSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("pg_colbert_llama.threads",
							"Number of llama.cpp worker threads.",
							NULL,
							&pg_colbert_llama_threads,
							4, 1, 1024,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("pg_colbert_llama.n_ctx",
							"llama.cpp context size for embedding.",
							NULL,
							&pg_colbert_llama_n_ctx,
							512, 1, INT_MAX,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("pg_colbert_llama.n_batch",
							"llama.cpp batch size for embedding.",
							NULL,
							&pg_colbert_llama_n_batch,
							512, 1, INT_MAX,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("pg_colbert_llama.batch_sequences",
							"Maximum independent input sequences per llama.cpp embedding batch.",
							NULL,
							&pg_colbert_llama_batch_sequences,
							8, 1, 1024,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("pg_colbert_llama.n_gpu_layers",
							"Number of model layers to offload to GPU.",
							NULL,
							&pg_colbert_llama_n_gpu_layers,
							0, 0, INT_MAX,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("pg_colbert_llama.cache_size",
							"Maximum backend-local llama model cache entries.",
							NULL,
							&pg_colbert_llama_cache_size,
							2, 0, 128,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomStringVariable("pg_colbert_llama.query_prefix",
							   "Prefix prepended before query text.",
							   NULL,
							   &pg_colbert_llama_query_prefix,
							   "[Q] ",
							   PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomStringVariable("pg_colbert_llama.document_prefix",
							   "Prefix prepended before document text.",
							   NULL,
							   &pg_colbert_llama_document_prefix,
							   "[D] ",
							   PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("pg_colbert_llama.max_query_vectors",
							"Maximum retained query token vectors.",
							NULL,
							&pg_colbert_llama_max_query_vectors,
							32, 1, 4096,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("pg_colbert_llama.max_doc_vectors",
							"Maximum retained document token vectors.",
							NULL,
							&pg_colbert_llama_max_doc_vectors,
							256, 1, 4096,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("pg_colbert_llama.query_length",
							"PyLate-style query expansion length.",
							"Short query inputs are padded with mask tokens up to this length before embedding.",
							&pg_colbert_llama_query_length,
							32, 1, 4096,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("pg_colbert_llama.strict_profile",
							 "Reject profile requests that pg_colbert_llama cannot execute exactly.",
							 "When off, model_info and colbert_debug report approximated profile capabilities instead of raising errors.",
							 &pg_colbert_llama_strict_profile,
							 false,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("pg_colbert_llama.require_normalized",
							 "Require each ColBERT token vector to be L2-normalized.",
							 NULL,
							 &pg_colbert_llama_require_normalized,
							 true,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("pg_colbert_llama.log_timing",
							 "Log ColBERT llama embedding phase timings.",
							 NULL,
							 &pg_colbert_llama_log_timing,
							 false,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("pg_colbert_llama.expected_dim",
							"Expected ColBERT embedding dimension.",
							NULL,
							&pg_colbert_llama_expected_dim,
							128, 1, 16000,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomStringVariable("pg_colbert_llama.allowed_models",
							   "Comma-separated allowlist of model aliases. Empty allows all valid aliases.",
							   NULL,
							   &pg_colbert_llama_allowed_models,
							   "",
							   PGC_SUSET, 0, NULL, NULL, NULL);
}

static Oid
PgColbertExtensionSchema(Oid extensionOid)
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

static Oid
PgColbertVectorTypeOid(void)
{
	Oid			extensionOid;
	Oid			schemaOid;

	if (OidIsValid(pg_colbert_llama_vector_type_oid))
		return pg_colbert_llama_vector_type_oid;

	extensionOid = get_extension_oid("vector", true);
	if (!OidIsValid(extensionOid))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("vector extension is required by pg_colbert_llama"),
				 errhint("Run CREATE EXTENSION vector before using pg_colbert_llama.")));

	schemaOid = PgColbertExtensionSchema(extensionOid);
	if (!OidIsValid(schemaOid))
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_SCHEMA),
				 errmsg("could not find schema for vector extension")));

	pg_colbert_llama_vector_type_oid =
		GetSysCacheOid2(TYPENAMENSP, Anum_pg_type_oid,
						CStringGetDatum("vector"),
						ObjectIdGetDatum(schemaOid));
	if (!OidIsValid(pg_colbert_llama_vector_type_oid))
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("could not find vector type installed by vector extension")));

	return pg_colbert_llama_vector_type_oid;
}

static Oid
PgColbertLookupPgturbohybridFunction(const char *funcname, int nargs,
									 Oid *argtypes, bool missing_ok)
{
	Oid			extensionOid;
	Oid			schemaOid;
	char	   *schemaName;
	Oid			funcOid;

	extensionOid = get_extension_oid("pgturbohybrid", true);
	if (!OidIsValid(extensionOid))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("pgturbohybrid extension is required by pg_colbert_llama"),
				 errhint("Run CREATE EXTENSION pgturbohybrid before using pg_colbert_llama.")));

	schemaOid = PgColbertExtensionSchema(extensionOid);
	if (!OidIsValid(schemaOid))
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_SCHEMA),
				 errmsg("could not find schema for pgturbohybrid extension")));

	schemaName = get_namespace_name(schemaOid);
	if (schemaName == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_SCHEMA),
				 errmsg("could not find schema name for pgturbohybrid extension")));

	funcOid = LookupFuncName(list_make2(makeString(schemaName),
										makeString(pstrdup(funcname))),
							 nargs, argtypes, missing_ok);
	pfree(schemaName);
	return funcOid;
}

/* Resolve (and cache) the pgturbohybrid turbohybrid_sparse_vector type Oid. */
static Oid
PgColbertSparseVectorTypeOid(void)
{
	Oid			extensionOid;
	Oid			schemaOid;

	if (OidIsValid(pg_colbert_llama_sparse_type_oid))
		return pg_colbert_llama_sparse_type_oid;

	extensionOid = get_extension_oid("pgturbohybrid", true);
	if (!OidIsValid(extensionOid))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("pgturbohybrid extension is required by pg_colbert_llama"),
				 errhint("Run CREATE EXTENSION pgturbohybrid before using pg_colbert_llama.")));

	schemaOid = PgColbertExtensionSchema(extensionOid);
	if (!OidIsValid(schemaOid))
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_SCHEMA),
				 errmsg("could not find schema for pgturbohybrid extension")));

	pg_colbert_llama_sparse_type_oid =
		GetSysCacheOid2(TYPENAMENSP, Anum_pg_type_oid,
						CStringGetDatum("turbohybrid_sparse_vector"),
						ObjectIdGetDatum(schemaOid));
	if (!OidIsValid(pg_colbert_llama_sparse_type_oid))
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("could not find turbohybrid_sparse_vector type installed by pgturbohybrid")));

	return pg_colbert_llama_sparse_type_oid;
}

static void
PgColbertCheckAllowedModel(const char *alias)
{
	char	   *copy;
	char	   *token;

	if (pg_colbert_llama_allowed_models == NULL ||
		pg_colbert_llama_allowed_models[0] == '\0')
		return;

	copy = pstrdup(pg_colbert_llama_allowed_models);
	for (token = strtok(copy, ","); token != NULL; token = strtok(NULL, ","))
	{
		char	   *start = token;
		char	   *end;

		while (isspace((unsigned char) *start))
			start++;
		end = start + strlen(start);
		while (end > start && isspace((unsigned char) end[-1]))
			*--end = '\0';
		if (strcmp(start, alias) == 0)
			return;
	}

	ereport(ERROR,
			(errcode(ERRCODE_INSUFFICIENT_PRIVILEGE),
			 errmsg("model alias \"%s\" is not allowed", alias)));
}

static void
PgColbertCopyProfileString(char *dest, Size destSize, const char *src)
{
	if (destSize == 0)
		return;
	if (src == NULL)
		src = "";
	snprintf(dest, destSize, "%s", src);
}

static const char *
PgColbertSkipJsonWhitespace(const char *p, const char *end)
{
	while (p < end && isspace((unsigned char) *p))
		p++;
	return p;
}

static bool
PgColbertJsonKeyEquals(const char *p, const char *end, const char *key,
					   const char **after)
{
	const char *k = key;

	if (p >= end || *p != '"')
		return false;
	p++;
	while (p < end && *p != '"')
	{
		if (*p == '\\')
			return false;
		if (*k == '\0' || *p != *k)
			return false;
		p++;
		k++;
	}
	if (p >= end || *p != '"' || *k != '\0')
		return false;
	*after = p + 1;
	return true;
}

static bool
PgColbertJsonFindKey(const char *json, const char *end, const char *key,
					 const char **value)
{
	int			objectDepth = 0;
	int			arrayDepth = 0;
	bool		inString = false;
	bool		escape = false;

	for (const char *p = json; p < end; p++)
	{
		const char *after;

		if (inString)
		{
			if (escape)
				escape = false;
			else if (*p == '\\')
				escape = true;
			else if (*p == '"')
				inString = false;
			continue;
		}

		if (*p == '{')
		{
			objectDepth++;
			continue;
		}
		if (*p == '}')
		{
			objectDepth--;
			continue;
		}
		if (*p == '[')
		{
			arrayDepth++;
			continue;
		}
		if (*p == ']')
		{
			arrayDepth--;
			continue;
		}

		if (*p != '"')
			continue;

		if (objectDepth != 1 || arrayDepth != 0 ||
			!PgColbertJsonKeyEquals(p, end, key, &after))
		{
			inString = true;
			continue;
		}
		after = PgColbertSkipJsonWhitespace(after, end);
		if (after >= end || *after != ':')
		{
			inString = true;
			continue;
		}
		after = PgColbertSkipJsonWhitespace(after + 1, end);
		*value = after;
		return true;
	}
	return false;
}

static bool
PgColbertJsonSliceBalanced(const char *start, const char *end,
						   char openChar, char closeChar,
						   const char **sliceEnd)
{
	int			depth = 0;
	bool		inString = false;
	bool		escape = false;

	if (start >= end || *start != openChar)
		return false;

	for (const char *p = start; p < end; p++)
	{
		if (inString)
		{
			if (escape)
				escape = false;
			else if (*p == '\\')
				escape = true;
			else if (*p == '"')
				inString = false;
			continue;
		}
		if (*p == '"')
			inString = true;
		else if (*p == openChar)
			depth++;
		else if (*p == closeChar)
		{
			depth--;
			if (depth == 0)
			{
				*sliceEnd = p + 1;
				return true;
			}
		}
	}
	return false;
}

static bool
PgColbertJsonObjectForKey(const char *json, const char *end, const char *key,
						  const char **objectStart, const char **objectEnd)
{
	const char *value;

	if (!PgColbertJsonFindKey(json, end, key, &value))
		return false;
	value = PgColbertSkipJsonWhitespace(value, end);
	if (!PgColbertJsonSliceBalanced(value, end, '{', '}', objectEnd))
		return false;
	*objectStart = value;
	return true;
}

static bool
PgColbertJsonArrayForKey(const char *json, const char *end, const char *key,
						 const char **arrayStart, const char **arrayEnd)
{
	const char *value;

	if (!PgColbertJsonFindKey(json, end, key, &value))
		return false;
	value = PgColbertSkipJsonWhitespace(value, end);
	if (!PgColbertJsonSliceBalanced(value, end, '[', ']', arrayEnd))
		return false;
	*arrayStart = value;
	return true;
}

static bool
PgColbertJsonReadStringValue(const char *value, const char *end,
							 char *dest, Size destSize)
{
	char	   *out = dest;
	char	   *outEnd = dest + destSize - 1;

	if (destSize == 0)
		return false;
	value = PgColbertSkipJsonWhitespace(value, end);
	if (value >= end || *value != '"')
		return false;
	value++;
	while (value < end && *value != '"')
	{
		char		c = *value++;

		if (c == '\\')
		{
			if (value >= end)
				return false;
			c = *value++;
			switch (c)
			{
				case '"':
				case '\\':
				case '/':
					break;
				case 'b':
					c = '\b';
					break;
				case 'f':
					c = '\f';
					break;
				case 'n':
					c = '\n';
					break;
				case 'r':
					c = '\r';
					break;
				case 't':
					c = '\t';
					break;
				default:
					return false;
			}
		}
		if (out < outEnd)
			*out++ = c;
	}
	if (value >= end || *value != '"')
		return false;
	*out = '\0';
	return true;
}

static bool
PgColbertJsonStringField(const char *json, const char *end, const char *key,
						 char *dest, Size destSize)
{
	const char *value;

	if (!PgColbertJsonFindKey(json, end, key, &value))
		return false;
	return PgColbertJsonReadStringValue(value, end, dest, destSize);
}

static bool
PgColbertJsonIntField(const char *json, const char *end, const char *key,
					  int32 *dest)
{
	const char *value;
	char	   *parseEnd;
	long		parsed;

	if (!PgColbertJsonFindKey(json, end, key, &value))
		return false;
	value = PgColbertSkipJsonWhitespace(value, end);
	if (value + 4 <= end && strncmp(value, "null", 4) == 0)
	{
		*dest = -1;
		return true;
	}
	errno = 0;
	parsed = strtol(value, &parseEnd, 10);
	if (errno != 0 || parseEnd == value || parsed < INT_MIN || parsed > INT_MAX)
		return false;
	*dest = (int32) parsed;
	return true;
}

static bool
PgColbertJsonBoolField(const char *json, const char *end, const char *key,
					   bool *dest)
{
	const char *value;

	if (!PgColbertJsonFindKey(json, end, key, &value))
		return false;
	value = PgColbertSkipJsonWhitespace(value, end);
	if (value + 4 <= end && strncmp(value, "true", 4) == 0)
	{
		*dest = true;
		return true;
	}
	if (value + 5 <= end && strncmp(value, "false", 5) == 0)
	{
		*dest = false;
		return true;
	}
	return false;
}

static int32
PgColbertJsonIntArrayField(const char *json, const char *end, const char *key,
						   int32 *dest, int32 maxValues)
{
	const char *arrayStart;
	const char *arrayEnd;
	const char *p;
	int32		count = 0;

	if (!PgColbertJsonArrayForKey(json, end, key, &arrayStart, &arrayEnd))
		return 0;
	p = arrayStart + 1;
	while (p < arrayEnd - 1)
	{
		char	   *parseEnd;
		long		parsed;

		p = PgColbertSkipJsonWhitespace(p, arrayEnd - 1);
		if (p >= arrayEnd - 1)
			break;
		errno = 0;
		parsed = strtol(p, &parseEnd, 10);
		if (errno != 0 || parseEnd == p || parsed < 0 || parsed > INT_MAX)
			break;
		if (count < maxValues)
			dest[count++] = (int32) parsed;
		p = PgColbertSkipJsonWhitespace(parseEnd, arrayEnd - 1);
		if (p < arrayEnd - 1 && *p == ',')
			p++;
	}
	return count;
}

static char *
PgColbertReadTextFile(const char *path)
{
	FILE	   *file;
	struct stat st;
	char	   *data;
	size_t		readBytes;

	if (stat(path, &st) != 0 || st.st_size <= 0 || st.st_size > 1024 * 1024)
		return NULL;
	file = fopen(path, "rb");
	if (file == NULL)
		return NULL;
	data = (char *) palloc((Size) st.st_size + 1);
	readBytes = fread(data, 1, (size_t) st.st_size, file);
	fclose(file);
	if (readBytes != (size_t) st.st_size)
		return NULL;
	data[st.st_size] = '\0';
	return data;
}

static bool
PgColbertReadExact(FILE *file, void *dest, size_t size)
{
	return fread(dest, 1, size, file) == size;
}

static bool
PgColbertSkipBytes(FILE *file, uint64 bytes)
{
	char		buffer[4096];

	while (bytes > 0)
	{
		size_t		chunk = bytes < sizeof(buffer) ? (size_t) bytes : sizeof(buffer);

		if (fread(buffer, 1, chunk, file) != chunk)
			return false;
		bytes -= chunk;
	}
	return true;
}

static char *
PgColbertReadGgufString(FILE *file)
{
	uint64		len;
	char	   *value;

	if (!PgColbertReadExact(file, &len, sizeof(len)))
		return NULL;
	if (len > 1024 * 1024)
		return NULL;
	value = (char *) palloc((Size) len + 1);
	if (!PgColbertReadExact(file, value, (size_t) len))
		return NULL;
	value[len] = '\0';
	return value;
}

static bool
PgColbertSkipGgufValue(FILE *file, uint32 type)
{
	uint32		arrayType;
	uint64		arrayLen;

	switch (type)
	{
		case 0:
		case 1:
		case 7:
			return PgColbertSkipBytes(file, 1);
		case 2:
		case 3:
			return PgColbertSkipBytes(file, 2);
		case 4:
		case 5:
		case 6:
			return PgColbertSkipBytes(file, 4);
		case 8:
		{
			char	   *ignored = PgColbertReadGgufString(file);

			return ignored != NULL;
		}
		case 10:
		case 11:
		case 12:
			return PgColbertSkipBytes(file, 8);
		case 9:
			if (!PgColbertReadExact(file, &arrayType, sizeof(arrayType)) ||
				!PgColbertReadExact(file, &arrayLen, sizeof(arrayLen)))
				return false;
			for (uint64 i = 0; i < arrayLen; i++)
			{
				if (!PgColbertSkipGgufValue(file, arrayType))
					return false;
			}
			return true;
		default:
			return false;
	}
}

static char *
PgColbertReadGgufProfileJson(const char *path)
{
	FILE	   *file;
	char		magic[4];
	uint32		version;
	uint64		tensorCount;
	uint64		kvCount;
	char	   *profile = NULL;

	file = fopen(path, "rb");
	if (file == NULL)
		return NULL;
	if (!PgColbertReadExact(file, magic, sizeof(magic)) ||
		memcmp(magic, "GGUF", sizeof(magic)) != 0 ||
		!PgColbertReadExact(file, &version, sizeof(version)) ||
		!PgColbertReadExact(file, &tensorCount, sizeof(tensorCount)) ||
		!PgColbertReadExact(file, &kvCount, sizeof(kvCount)))
		goto cleanup;

	for (uint64 i = 0; i < kvCount; i++)
	{
		char	   *key;
		uint32		type;

		key = PgColbertReadGgufString(file);
		if (key == NULL || !PgColbertReadExact(file, &type, sizeof(type)))
			goto cleanup;
		if (strcmp(key, "pg_colbert.profile_json") == 0 && type == 8)
		{
			profile = PgColbertReadGgufString(file);
			break;
		}
		if (!PgColbertSkipGgufValue(file, type))
			goto cleanup;
	}

cleanup:
	fclose(file);
	return profile;
}

static bool
PgColbertParseProfileJson(PgColbertRuntimeProfile *profile,
						  const char *json,
						  PgColbertProfileSource source,
						  const char *sourceName,
						  char **errorMessage)
{
	const char *end = json + strlen(json);
	const char *objectStart;
	const char *objectEnd;

	memset(profile, 0, sizeof(*profile));
	profile->loaded = true;
	profile->source = source;
	PgColbertCopyProfileString(profile->sourceName, sizeof(profile->sourceName),
							   sourceName);
	profile->specialTokens.clsTokenId = -1;
	profile->specialTokens.sepTokenId = -1;
	profile->specialTokens.padTokenId = -1;
	profile->specialTokens.maskTokenId = -1;
	profile->specialTokens.qTokenId = -1;
	profile->specialTokens.dTokenId = -1;
	profile->queryPadTo = -1;
	profile->queryPadTokenId = -1;
	profile->queryAttendToExpansionTokens = true;

	if (!PgColbertJsonStringField(json, end, "schema", profile->schema,
								  sizeof(profile->schema)) ||
		strcmp(profile->schema, "pg_colbert_profile_v1") != 0)
	{
		*errorMessage = pstrdup("ColBERT profile schema must be pg_colbert_profile_v1");
		return false;
	}
	if (!PgColbertJsonIntField(json, end, "output_dim", &profile->outputDim) ||
		profile->outputDim <= 0)
	{
		*errorMessage = pstrdup("ColBERT profile output_dim must be positive");
		return false;
	}
	(void) PgColbertJsonBoolField(json, end, "normalize", &profile->normalize);
	(void) PgColbertJsonStringField(json, end, "backbone_family",
									profile->backboneFamily,
									sizeof(profile->backboneFamily));
	(void) PgColbertJsonStringField(json, end, "colbert_family",
									profile->colbertFamily,
									sizeof(profile->colbertFamily));

	if (PgColbertJsonObjectForKey(json, end, "tokenizer", &objectStart, &objectEnd))
	{
		const char *specialStart;
		const char *specialEnd;

		(void) PgColbertJsonStringField(objectStart, objectEnd, "source",
										profile->tokenizerSource,
										sizeof(profile->tokenizerSource));
		PgColbertCopyProfileString(profile->tokenizerStatus,
								   sizeof(profile->tokenizerStatus), "ok");
		if (PgColbertJsonObjectForKey(objectStart, objectEnd, "special_tokens",
									  &specialStart, &specialEnd))
		{
			(void) PgColbertJsonIntField(specialStart, specialEnd,
										 "cls_token_id",
										 &profile->specialTokens.clsTokenId);
			(void) PgColbertJsonIntField(specialStart, specialEnd,
										 "sep_token_id",
										 &profile->specialTokens.sepTokenId);
			(void) PgColbertJsonIntField(specialStart, specialEnd,
										 "pad_token_id",
										 &profile->specialTokens.padTokenId);
			(void) PgColbertJsonIntField(specialStart, specialEnd,
										 "mask_token_id",
										 &profile->specialTokens.maskTokenId);
			(void) PgColbertJsonIntField(specialStart, specialEnd,
										 "q_token_id",
										 &profile->specialTokens.qTokenId);
			(void) PgColbertJsonIntField(specialStart, specialEnd,
										 "d_token_id",
										 &profile->specialTokens.dTokenId);
		}
	}

	if (!PgColbertJsonObjectForKey(json, end, "query", &objectStart, &objectEnd))
	{
		*errorMessage = pstrdup("ColBERT profile query object is required");
		return false;
	}
	(void) PgColbertJsonStringField(objectStart, objectEnd, "prefix",
									profile->queryPrefix,
									sizeof(profile->queryPrefix));
	if (!PgColbertJsonIntField(objectStart, objectEnd, "max_length",
							   &profile->queryMaxLength) ||
		profile->queryMaxLength <= 0)
	{
		*errorMessage = pstrdup("ColBERT profile query.max_length must be positive");
		return false;
	}
	(void) PgColbertJsonIntField(objectStart, objectEnd, "pad_to",
								 &profile->queryPadTo);
	(void) PgColbertJsonIntField(objectStart, objectEnd, "pad_token_id",
								 &profile->queryPadTokenId);
	(void) PgColbertJsonIntField(objectStart, objectEnd, "token_type_id",
								 &profile->queryTokenTypeId);
	(void) PgColbertJsonBoolField(objectStart, objectEnd,
								  "attend_to_expansion_tokens",
								  &profile->queryAttendToExpansionTokens);
	(void) PgColbertJsonStringField(objectStart, objectEnd,
									"attention_mask_policy",
									profile->queryAttentionMaskPolicy,
									sizeof(profile->queryAttentionMaskPolicy));
	(void) PgColbertJsonStringField(objectStart, objectEnd, "retain_policy",
									profile->queryRetainPolicy,
									sizeof(profile->queryRetainPolicy));
	(void) PgColbertJsonStringField(objectStart, objectEnd, "output_policy",
									profile->queryOutputPolicy,
									sizeof(profile->queryOutputPolicy));

	if (!PgColbertJsonObjectForKey(json, end, "document", &objectStart, &objectEnd))
	{
		*errorMessage = pstrdup("ColBERT profile document object is required");
		return false;
	}
	(void) PgColbertJsonStringField(objectStart, objectEnd, "prefix",
									profile->documentPrefix,
									sizeof(profile->documentPrefix));
	if (!PgColbertJsonIntField(objectStart, objectEnd, "max_length",
							   &profile->documentMaxLength) ||
		profile->documentMaxLength <= 0)
	{
		*errorMessage = pstrdup("ColBERT profile document.max_length must be positive");
		return false;
	}
	(void) PgColbertJsonIntField(objectStart, objectEnd, "token_type_id",
								 &profile->documentTokenTypeId);
	(void) PgColbertJsonStringField(objectStart, objectEnd, "retain_policy",
									profile->documentRetainPolicy,
									sizeof(profile->documentRetainPolicy));
	(void) PgColbertJsonStringField(objectStart, objectEnd,
									"attention_mask_policy",
									profile->documentAttentionMaskPolicy,
									sizeof(profile->documentAttentionMaskPolicy));
	profile->skiplistTokenCount =
		PgColbertJsonIntArrayField(objectStart, objectEnd, "skiplist_token_ids",
								   profile->skiplistTokenIds,
								   PG_COLBERT_LLAMA_MAX_SKIPLIST_TOKENS);

	if (PgColbertJsonObjectForKey(json, end, "projection", &objectStart, &objectEnd))
	{
		const char *modulesStart;
		const char *modulesEnd;
		const char *moduleStart;
		const char *moduleEnd;

		(void) PgColbertJsonStringField(objectStart, objectEnd, "kind",
										profile->projectionKind,
										sizeof(profile->projectionKind));
		(void) PgColbertJsonIntField(objectStart, objectEnd, "input_dim",
									 &profile->projectionInputDim);
		if (!PgColbertJsonIntField(objectStart, objectEnd, "output_dim",
								   &profile->projectionOutputDim))
			profile->projectionOutputDim = profile->outputDim;
		if (profile->projectionOutputDim != profile->outputDim &&
			strcmp(profile->projectionKind, "identity") != 0)
		{
			*errorMessage = pstrdup("ColBERT profile projection output_dim must match output_dim");
			return false;
		}
		(void) PgColbertJsonBoolField(objectStart, objectEnd, "normalize_after",
									  &profile->projectionNormalizeAfter);
		if (PgColbertJsonArrayForKey(objectStart, objectEnd, "modules",
									 &modulesStart, &modulesEnd))
		{
			const char *p = modulesStart + 1;

			while (p < modulesEnd - 1 &&
				   profile->projectionModuleCount < PG_COLBERT_LLAMA_MAX_PROJECTION_MODULES)
			{
				p = PgColbertSkipJsonWhitespace(p, modulesEnd - 1);
				if (p >= modulesEnd - 1 || *p != '{')
					break;
				if (!PgColbertJsonSliceBalanced(p, modulesEnd - 1, '{', '}',
												&moduleEnd))
					break;
				moduleStart = p;
				(void) PgColbertJsonStringField(moduleStart, moduleEnd, "type",
												profile->projectionModules[profile->projectionModuleCount].type,
												sizeof(profile->projectionModules[0].type));
				(void) PgColbertJsonStringField(moduleStart, moduleEnd,
												"activation",
												profile->projectionModules[profile->projectionModuleCount].activation,
												sizeof(profile->projectionModules[0].activation));
				(void) PgColbertJsonStringField(moduleStart, moduleEnd,
												"activation_function",
												profile->projectionModules[profile->projectionModuleCount].activation,
												sizeof(profile->projectionModules[0].activation));
				(void) PgColbertJsonStringField(moduleStart, moduleEnd,
												"weight",
												profile->projectionModules[profile->projectionModuleCount].weightTensor,
												sizeof(profile->projectionModules[0].weightTensor));
				(void) PgColbertJsonStringField(moduleStart, moduleEnd,
												"weight_tensor",
												profile->projectionModules[profile->projectionModuleCount].weightTensor,
												sizeof(profile->projectionModules[0].weightTensor));
				(void) PgColbertJsonStringField(moduleStart, moduleEnd,
												"weight_tensor_name",
												profile->projectionModules[profile->projectionModuleCount].weightTensor,
												sizeof(profile->projectionModules[0].weightTensor));
				(void) PgColbertJsonStringField(moduleStart, moduleEnd,
												"bias_tensor",
												profile->projectionModules[profile->projectionModuleCount].biasTensor,
												sizeof(profile->projectionModules[0].biasTensor));
				(void) PgColbertJsonStringField(moduleStart, moduleEnd,
												"bias_tensor_name",
												profile->projectionModules[profile->projectionModuleCount].biasTensor,
												sizeof(profile->projectionModules[0].biasTensor));
				(void) PgColbertJsonStringField(moduleStart, moduleEnd,
												"bias",
												profile->projectionModules[profile->projectionModuleCount].biasTensor,
												sizeof(profile->projectionModules[0].biasTensor));
				(void) PgColbertJsonIntField(moduleStart, moduleEnd, "input_dim",
											 &profile->projectionModules[profile->projectionModuleCount].inputDim);
				(void) PgColbertJsonIntField(moduleStart, moduleEnd, "in_features",
											 &profile->projectionModules[profile->projectionModuleCount].inputDim);
				(void) PgColbertJsonIntField(moduleStart, moduleEnd, "output_dim",
											 &profile->projectionModules[profile->projectionModuleCount].outputDim);
				(void) PgColbertJsonIntField(moduleStart, moduleEnd, "out_features",
											 &profile->projectionModules[profile->projectionModuleCount].outputDim);
				(void) PgColbertJsonBoolField(moduleStart, moduleEnd, "bias",
											  &profile->projectionModules[profile->projectionModuleCount].hasBias);
				if (profile->projectionModules[profile->projectionModuleCount].biasTensor[0] != '\0')
					profile->projectionModules[profile->projectionModuleCount].hasBias = true;
				(void) PgColbertJsonBoolField(moduleStart, moduleEnd, "normalize_after",
											  &profile->projectionModules[profile->projectionModuleCount].normalizeAfter);
				(void) PgColbertJsonIntField(moduleStart, moduleEnd, "p",
											 &profile->projectionModules[profile->projectionModuleCount].p);
				profile->projectionModuleCount++;
				p = PgColbertSkipJsonWhitespace(moduleEnd, modulesEnd - 1);
				if (p < modulesEnd - 1 && *p == ',')
					p++;
			}
		}
	}

	if (PgColbertJsonObjectForKey(json, end, "compatibility",
								  &objectStart, &objectEnd))
	{
		(void) PgColbertJsonBoolField(objectStart, objectEnd,
									  "llama_cpp_loadable",
									  &profile->llamaCppLoadable);
		(void) PgColbertJsonBoolField(objectStart, objectEnd,
									  "requires_profile",
									  &profile->requiresProfile);
		(void) PgColbertJsonBoolField(objectStart, objectEnd,
									  "strict_pylate_profile",
									  &profile->strictPylateProfile);
	}

	if (profile->queryPadTo <= 0)
		profile->queryPadTo = profile->queryMaxLength;
	if (profile->queryPadTokenId < 0)
		profile->queryPadTokenId = profile->specialTokens.maskTokenId;
	if (profile->projectionOutputDim <= 0)
		profile->projectionOutputDim = profile->outputDim;
	if (profile->projectionKind[0] == '\0')
		PgColbertCopyProfileString(profile->projectionKind,
								   sizeof(profile->projectionKind), "dense");
	if (profile->tokenizerSource[0] == '\0')
		PgColbertCopyProfileString(profile->tokenizerSource,
								   sizeof(profile->tokenizerSource), "unknown");
	if (profile->tokenizerStatus[0] == '\0')
		PgColbertCopyProfileString(profile->tokenizerStatus,
								   sizeof(profile->tokenizerStatus), "unknown");
	if (profile->queryRetainPolicy[0] == '\0')
		PgColbertCopyProfileString(profile->queryRetainPolicy,
								   sizeof(profile->queryRetainPolicy), "all");
	if (profile->queryOutputPolicy[0] == '\0')
		PgColbertCopyProfileString(profile->queryOutputPolicy,
								   sizeof(profile->queryOutputPolicy), "all");
	if (profile->documentRetainPolicy[0] == '\0')
		PgColbertCopyProfileString(profile->documentRetainPolicy,
								   sizeof(profile->documentRetainPolicy),
								   "non_padding_skiplist");

	if (profile->queryAttentionMaskPolicy[0] == '\0')
		PgColbertCopyProfileString(profile->queryAttentionMaskPolicy,
								   sizeof(profile->queryAttentionMaskPolicy),
								   profile->queryAttendToExpansionTokens ?
								   "llama_default_noncausal" :
								   "pylate_query_expansion_requested");
	if (profile->documentAttentionMaskPolicy[0] == '\0')
		PgColbertCopyProfileString(profile->documentAttentionMaskPolicy,
								   sizeof(profile->documentAttentionMaskPolicy),
								   "llama_default_noncausal");
	PgColbertCopyProfileString(profile->compatibilityLevel,
							   sizeof(profile->compatibilityLevel),
							   "profile_loaded");
	if (!profile->queryAttendToExpansionTokens)
		PgColbertCopyProfileString(profile->knownLimitations,
								   sizeof(profile->knownLimitations),
								   "query expansion attention uses llama.cpp default non-causal attention; PyLate expansion-token attention is approximated");

	return true;
}

static void
PgColbertValidateProfileProjection(const PgColbertRuntimeProfile *profile)
{
	for (int32 i = 0; i < profile->projectionModuleCount; i++)
	{
		const PgColbertProjectionModuleProfile *module =
			&profile->projectionModules[i];
		bool		isLinear =
			strcmp(module->type, "linear") == 0 ||
			strcmp(module->type, "dense") == 0;
		bool		isNormalize = strcmp(module->type, "normalize") == 0;
		bool		isTruncate = strcmp(module->type, "truncate") == 0;

		if (isLinear)
		{
			if (module->activation[0] != '\0' &&
				strcmp(module->activation, "identity") != 0)
				ereport(ERROR,
						(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
						 errmsg("ColBERT projection profile uses unsupported activation \"%s\"",
								module->activation)));
			continue;
		}
		if (isNormalize)
		{
			if (module->p > 0 && module->p != 2)
				ereport(ERROR,
						(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
						 errmsg("ColBERT normalize projection module only supports p=2")));
			continue;
		}
		if (isTruncate)
		{
			if (module->outputDim > 0 &&
				module->outputDim != profile->outputDim)
				ereport(ERROR,
						(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
						 errmsg("ColBERT truncate projection module output_dim %d does not match profile output_dim %d",
								module->outputDim, profile->outputDim)));
			continue;
		}
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("ColBERT projection profile uses unsupported module type \"%s\"",
						module->type)));
	}
}

static void
PgColbertBuildGucFallbackProfile(PgColbertModelSpec *spec)
{
	PgColbertRuntimeProfile *profile = &spec->profile;

	memset(profile, 0, sizeof(*profile));
	profile->loaded = false;
	profile->source = PG_COLBERT_PROFILE_SOURCE_GUC_FALLBACK;
	PgColbertCopyProfileString(profile->sourceName, sizeof(profile->sourceName),
							   "guc_fallback");
	PgColbertCopyProfileString(profile->schema, sizeof(profile->schema),
							   "guc_fallback");
	PgColbertCopyProfileString(profile->compatibilityLevel,
							   sizeof(profile->compatibilityLevel),
							   "ranking_smoke");
	PgColbertCopyProfileString(profile->tokenizerSource,
							   sizeof(profile->tokenizerSource),
							   "guc_fallback");
	PgColbertCopyProfileString(profile->tokenizerStatus,
							   sizeof(profile->tokenizerStatus), "unknown");
	PgColbertCopyProfileString(profile->projectionKind,
							   sizeof(profile->projectionKind), "dense");
	PgColbertCopyProfileString(profile->queryPrefix,
							   sizeof(profile->queryPrefix),
							   pg_colbert_llama_query_prefix);
	PgColbertCopyProfileString(profile->documentPrefix,
							   sizeof(profile->documentPrefix),
							   pg_colbert_llama_document_prefix);
	PgColbertCopyProfileString(profile->queryRetainPolicy,
							   sizeof(profile->queryRetainPolicy), "all");
	PgColbertCopyProfileString(profile->queryOutputPolicy,
							   sizeof(profile->queryOutputPolicy), "all");
	PgColbertCopyProfileString(profile->documentRetainPolicy,
							   sizeof(profile->documentRetainPolicy),
							   "non_padding_ascii_punctuation");
	PgColbertCopyProfileString(profile->queryAttentionMaskPolicy,
							   sizeof(profile->queryAttentionMaskPolicy),
							   "llama_default_noncausal");
	PgColbertCopyProfileString(profile->documentAttentionMaskPolicy,
							   sizeof(profile->documentAttentionMaskPolicy),
							   "llama_default_noncausal");
	profile->outputDim = pg_colbert_llama_expected_dim;
	profile->queryMaxLength = pg_colbert_llama_query_length;
	profile->queryPadTo = pg_colbert_llama_query_length;
	profile->queryPadTokenId = -1;
	profile->queryTokenTypeId = 0;
	profile->queryAttendToExpansionTokens = true;
	profile->documentMaxLength = pg_colbert_llama_max_doc_vectors;
	profile->documentTokenTypeId = 0;
	profile->projectionOutputDim = pg_colbert_llama_expected_dim;
}

static void
PgColbertLoadRuntimeProfile(PgColbertModelSpec *spec)
{
	char		modelPath[MAXPGPATH];
	char		sidecarPath[MAXPGPATH];
	char	   *json = NULL;
	char	   *errorMessage = NULL;

	snprintf(modelPath, sizeof(modelPath), "%s/%s.gguf", spec->modelDir,
			 spec->alias);
	json = PgColbertReadGgufProfileJson(modelPath);
	if (json != NULL &&
		PgColbertParseProfileJson(&spec->profile, json,
								  PG_COLBERT_PROFILE_SOURCE_GGUF, "gguf",
								  &errorMessage))
		goto loaded;
	if (errorMessage != NULL)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("%s", errorMessage)));

	snprintf(sidecarPath, sizeof(sidecarPath), "%s.colbert_profile.json",
			 modelPath);
	json = PgColbertReadTextFile(sidecarPath);
	if (json != NULL)
	{
		if (!PgColbertParseProfileJson(&spec->profile, json,
									   PG_COLBERT_PROFILE_SOURCE_SIDECAR,
									   "sidecar", &errorMessage))
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("%s", errorMessage)));
		goto loaded;
	}

	PgColbertBuildGucFallbackProfile(spec);

loaded:
	if (spec->profile.loaded && spec->profile.outputDim != spec->expectedDim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("ColBERT profile output_dim %d does not match expected dimension %d",
						spec->profile.outputDim, spec->expectedDim)));
	if (spec->profile.loaded)
	{
		PgColbertValidateProfileProjection(&spec->profile);
		spec->prefix = spec->role == PG_COLBERT_ROLE_QUERY ?
			spec->profile.queryPrefix : spec->profile.documentPrefix;
		if (spec->role == PG_COLBERT_ROLE_QUERY)
			spec->queryLength = spec->profile.queryPadTo;
	}
	if (spec->strictProfile &&
		spec->role == PG_COLBERT_ROLE_QUERY &&
		spec->profile.loaded &&
		spec->profile.strictPylateProfile &&
		!spec->profile.queryAttendToExpansionTokens)
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("ColBERT profile requests PyLate query expansion attention that llama.cpp cannot execute exactly"),
				 errhint("Disable pg_colbert_llama.strict_profile to allow the documented approximation.")));
}

static void
PgColbertParseModel(text *modelText, PgColbertModelSpec *spec)
{
	char	   *model = text_to_cstring(modelText);
	char	   *colon = strrchr(model, ':');
	Size		aliasLen;

	memset(spec, 0, sizeof(*spec));
	if (colon == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("model must use alias:query or alias:doc syntax")));

	aliasLen = (Size) (colon - model);
	if (aliasLen == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("model alias cannot be empty")));
	if (aliasLen > PG_COLBERT_LLAMA_MAX_ALIAS)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("model alias is too long")));
	if (model[0] == '.')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("model alias cannot start with a dot")));

	for (Size i = 0; i < aliasLen; i++)
	{
		unsigned char c = (unsigned char) model[i];

		if (!(isalnum(c) || c == '_' || c == '-' || c == '.'))
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("model alias may contain only letters, digits, underscore, dot, and dash")));
		if (model[i] == '.' && i + 1 < aliasLen && model[i + 1] == '.')
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("model alias cannot contain \"..\"")));
	}

	memcpy(spec->alias, model, aliasLen);
	spec->alias[aliasLen] = '\0';
	spec->outputMode = PG_LLAMA_EMBED_OUTPUT_TOKENS;
	spec->pooling = PG_LLAMA_EMBED_POOLING_NONE;
	spec->checkExpectedDim = true;
	spec->normalize = pg_colbert_llama_require_normalized;

	if (strcmp(colon + 1, "query") == 0)
	{
		spec->role = PG_COLBERT_ROLE_QUERY;
		spec->roleName = "query";
		spec->prefix = pg_colbert_llama_query_prefix;
		spec->maxVectors = pg_colbert_llama_max_query_vectors;
	}
	else if (strcmp(colon + 1, "doc") == 0)
	{
		spec->role = PG_COLBERT_ROLE_DOC;
		spec->roleName = "doc";
		spec->prefix = pg_colbert_llama_document_prefix;
		spec->maxVectors = pg_colbert_llama_max_doc_vectors;
	}
	else
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("unknown model role \"%s\"", colon + 1),
				 errhint("Use alias:query or alias:doc.")));

	spec->modelDir = pg_colbert_llama_model_dir;
	spec->expectedDim = pg_colbert_llama_expected_dim;
	spec->threads = pg_colbert_llama_threads;
	spec->nCtx = pg_colbert_llama_n_ctx;
	spec->nBatch = pg_colbert_llama_n_batch;
	spec->batchSequences = pg_colbert_llama_batch_sequences;
	spec->nGpuLayers = pg_colbert_llama_n_gpu_layers;
	spec->cacheSize = pg_colbert_llama_cache_size;
	spec->queryLength = pg_colbert_llama_query_length;
	spec->strictProfile = pg_colbert_llama_strict_profile;
	spec->logTiming = pg_colbert_llama_log_timing;
	PgColbertCheckAllowedModel(spec->alias);
	PgColbertLoadRuntimeProfile(spec);
}

static bool
PgLlamaEmbedJsonbGet(Jsonb *options, const char *key, JsonbValue *value)
{
	JsonbIterator *it;
	JsonbIteratorToken token;
	JsonbValue	current;

	if (options == NULL || !JB_ROOT_IS_OBJECT(options))
		return false;

	it = JsonbIteratorInit(&options->root);
	while ((token = JsonbIteratorNext(&it, &current, false)) != WJB_DONE)
	{
		if (token == WJB_KEY &&
			current.type == jbvString &&
			current.val.string.len == (int) strlen(key) &&
			strncmp(current.val.string.val, key, current.val.string.len) == 0)
		{
			token = JsonbIteratorNext(&it, value, false);
			return token == WJB_VALUE;
		}
	}
	return false;
}

static char *
PgLlamaEmbedJsonbGetString(Jsonb *options, const char *key)
{
	JsonbValue	value;

	if (!PgLlamaEmbedJsonbGet(options, key, &value))
		return NULL;
	if (value.type != jbvString)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("llama_embed option \"%s\" must be a string", key)));
	return pnstrdup(value.val.string.val, value.val.string.len);
}

static bool
PgLlamaEmbedJsonbGetBool(Jsonb *options, const char *key, bool defaultValue)
{
	JsonbValue	value;

	if (!PgLlamaEmbedJsonbGet(options, key, &value))
		return defaultValue;
	if (value.type != jbvBool)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("llama_embed option \"%s\" must be a boolean", key)));
	return value.val.boolean;
}

static bool
PgLlamaEmbedJsonbGetInt(Jsonb *options, const char *key, int *result)
{
	JsonbValue	value;

	if (!PgLlamaEmbedJsonbGet(options, key, &value))
		return false;
	if (value.type != jbvNumeric)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("llama_embed option \"%s\" must be an integer", key)));
	*result = DatumGetInt32(DirectFunctionCall1(numeric_int4,
											   NumericGetDatum(value.val.numeric)));
	return true;
}

static PgLlamaEmbedOutputMode
PgLlamaEmbedParseMode(const char *mode)
{
	if (mode == NULL || strcmp(mode, "tokens") == 0)
		return PG_LLAMA_EMBED_OUTPUT_TOKENS;
	if (strcmp(mode, "dense") == 0)
		return PG_LLAMA_EMBED_OUTPUT_DENSE;
	if (strcmp(mode, "sparse") == 0)
		return PG_LLAMA_EMBED_OUTPUT_SPARSE;
	ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("unknown llama_embed mode \"%s\"", mode),
			 errhint("Use \"dense\", \"tokens\", or \"sparse\".")));
	return PG_LLAMA_EMBED_OUTPUT_TOKENS;
}

static PgLlamaEmbedPooling
PgLlamaEmbedParsePooling(const char *pooling,
						 PgLlamaEmbedOutputMode outputMode)
{
	if (pooling == NULL || strcmp(pooling, "model") == 0)
		return outputMode == PG_LLAMA_EMBED_OUTPUT_DENSE ?
			PG_LLAMA_EMBED_POOLING_MEAN : PG_LLAMA_EMBED_POOLING_NONE;
	/* Sparse output is a per-vocabulary bag, not pooled token vectors. */
	if (strcmp(pooling, "none") == 0)
		return PG_LLAMA_EMBED_POOLING_NONE;
	if (strcmp(pooling, "mean") == 0)
		return PG_LLAMA_EMBED_POOLING_MEAN;
	if (strcmp(pooling, "cls") == 0)
		return PG_LLAMA_EMBED_POOLING_CLS;
	if (strcmp(pooling, "last") == 0)
		return PG_LLAMA_EMBED_POOLING_LAST;
	if (strcmp(pooling, "rank") == 0)
		return PG_LLAMA_EMBED_POOLING_RANK;
	ereport(ERROR,
			(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
			 errmsg("unknown llama_embed pooling \"%s\"", pooling),
			 errhint("Use \"model\", \"none\", \"mean\", \"cls\", \"last\", or \"rank\".")));
	return PG_LLAMA_EMBED_POOLING_NONE;
}

static void
PgLlamaEmbedValidateAlias(const char *model, Size aliasLen)
{
	if (aliasLen == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("model alias cannot be empty")));
	if (aliasLen > PG_COLBERT_LLAMA_MAX_ALIAS)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("model alias is too long")));
	if (model[0] == '.')
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("model alias cannot start with a dot")));

	for (Size i = 0; i < aliasLen; i++)
	{
		unsigned char c = (unsigned char) model[i];

		if (!(isalnum(c) || c == '_' || c == '-' || c == '.'))
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("model alias may contain only letters, digits, underscore, dot, and dash")));
		if (model[i] == '.' && i + 1 < aliasLen && model[i + 1] == '.')
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("model alias cannot contain \"..\"")));
	}
}

static void
PgLlamaEmbedParseModel(text *modelText, Jsonb *options,
					   PgLlamaEmbedOutputMode defaultMode,
					   PgColbertModelSpec *spec)
{
	char	   *model = text_to_cstring(modelText);
	char	   *colon = strrchr(model, ':');
	char	   *modeText;
	char	   *poolingText;
	char	   *prefixText;
	Size		aliasLen;
	int			intValue;

	memset(spec, 0, sizeof(*spec));
	modeText = PgLlamaEmbedJsonbGetString(options, "mode");
	spec->outputMode = modeText != NULL ?
		PgLlamaEmbedParseMode(modeText) : defaultMode;
	poolingText = PgLlamaEmbedJsonbGetString(options, "pooling");
	spec->pooling = PgLlamaEmbedParsePooling(poolingText, spec->outputMode);
	if (spec->outputMode == PG_LLAMA_EMBED_OUTPUT_DENSE &&
		spec->pooling == PG_LLAMA_EMBED_POOLING_NONE)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("dense llama_embed output requires pooled embeddings"),
				 errhint("Use pooling \"mean\", \"cls\", \"last\", \"rank\", or \"model\".")));
	if (spec->outputMode == PG_LLAMA_EMBED_OUTPUT_TOKENS &&
		spec->pooling != PG_LLAMA_EMBED_POOLING_NONE)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("token llama_embed output requires pooling \"none\"")));
	if (spec->outputMode == PG_LLAMA_EMBED_OUTPUT_SPARSE &&
		spec->pooling != PG_LLAMA_EMBED_POOLING_NONE)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("sparse llama_embed output requires pooling \"none\"")));

	aliasLen = colon == NULL ? strlen(model) : (Size) (colon - model);
	PgLlamaEmbedValidateAlias(model, aliasLen);
	memcpy(spec->alias, model, aliasLen);
	spec->alias[aliasLen] = '\0';

	spec->role = PG_COLBERT_ROLE_NONE;
	spec->roleName = "generic";
	spec->prefix = "";
	spec->maxVectors = spec->outputMode == PG_LLAMA_EMBED_OUTPUT_DENSE ?
		1 : pg_colbert_llama_max_doc_vectors;
	if (colon != NULL)
	{
		if (strcmp(colon + 1, "query") == 0)
		{
			spec->role = PG_COLBERT_ROLE_QUERY;
			spec->roleName = "query";
			spec->prefix = pg_colbert_llama_query_prefix;
			spec->maxVectors = pg_colbert_llama_max_query_vectors;
		}
		else if (strcmp(colon + 1, "doc") == 0)
		{
			spec->role = PG_COLBERT_ROLE_DOC;
			spec->roleName = "doc";
			spec->prefix = pg_colbert_llama_document_prefix;
			spec->maxVectors = pg_colbert_llama_max_doc_vectors;
		}
		else
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("unknown model role \"%s\"", colon + 1),
					 errhint("Use a plain alias or alias:query/alias:doc.")));
	}

	prefixText = PgLlamaEmbedJsonbGetString(options, "prefix");
	if (prefixText != NULL)
		spec->prefix = prefixText;
	if (PgLlamaEmbedJsonbGetInt(options, "max_vectors", &intValue))
	{
		if (intValue < 1 || intValue > 4096)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("llama_embed option \"max_vectors\" is out of range")));
		spec->maxVectors = intValue;
	}
	spec->expectedDim = pg_colbert_llama_expected_dim;
	spec->checkExpectedDim = false;
	if (PgLlamaEmbedJsonbGetInt(options, "expected_dim", &intValue))
	{
		if (intValue < 1 || intValue > 16000)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("llama_embed option \"expected_dim\" is out of range")));
		spec->expectedDim = intValue;
		spec->checkExpectedDim = true;
	}
	/*
	 * For dense/token output "normalize" is a boolean (whether to L2-normalize
	 * the emitted vectors).  For sparse output "normalize" is instead a string
	 * mode (none/l2/...) consumed by turbohybrid_sparse_vector_build, so leave
	 * spec->normalize untouched and let the builder interpret the key.
	 */
	spec->normalize = spec->outputMode == PG_LLAMA_EMBED_OUTPUT_SPARSE ? false :
		PgLlamaEmbedJsonbGetBool(options, "normalize", true);
	spec->modelDir = pg_colbert_llama_model_dir;
	spec->threads = pg_colbert_llama_threads;
	spec->nCtx = pg_colbert_llama_n_ctx;
	spec->nBatch = pg_colbert_llama_n_batch;
	spec->batchSequences = pg_colbert_llama_batch_sequences;
	spec->nGpuLayers = pg_colbert_llama_n_gpu_layers;
	spec->cacheSize = pg_colbert_llama_cache_size;
	spec->queryLength = pg_colbert_llama_query_length;
	spec->strictProfile = false;
	spec->logTiming = pg_colbert_llama_log_timing;
	spec->profile.loaded = false;
	spec->profile.source = PG_COLBERT_PROFILE_SOURCE_GUC_FALLBACK;
	strlcpy(spec->profile.sourceName, "generic",
			sizeof(spec->profile.sourceName));
	strlcpy(spec->profile.schema, "llama_embed_options",
			sizeof(spec->profile.schema));
	strlcpy(spec->profile.projectionKind, "model",
			sizeof(spec->profile.projectionKind));
	strlcpy(spec->profile.tokenizerStatus, "model",
			sizeof(spec->profile.tokenizerStatus));
	PgColbertCheckAllowedModel(spec->alias);
}

static void
PgColbertValidateOutput(const PgColbertModelSpec *spec,
						const PgColbertEngineOutput *output)
{
	const char *label;

	/*
	 * Sparse output is a term/weight bag (no fixed dim or vector count), so it
	 * has its own shape checks: non-negative term count, in-range term ids, and
	 * finite weights.  Term de-duplication / filtering / normalization happen
	 * later when the turbohybrid_sparse_vector is built.
	 */
	if (spec->outputMode == PG_LLAMA_EMBED_OUTPUT_SPARSE)
	{
		if (output->sparseCount < 0)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("llama_embed sparse engine returned a negative term count")));
		if (output->sparseCount > 0 &&
			(output->sparseTermIds == NULL || output->sparseWeights == NULL))
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("llama_embed sparse engine returned a malformed term bag")));
		for (int32 i = 0; i < output->sparseCount; i++)
		{
			if (output->sparseTermIds[i] < 0 ||
				(output->sparseVocabSize > 0 &&
				 output->sparseTermIds[i] >= output->sparseVocabSize))
				ereport(ERROR,
						(errcode(ERRCODE_DATA_EXCEPTION),
						 errmsg("llama_embed sparse engine returned out-of-range term id %d",
								output->sparseTermIds[i])));
			if (!isfinite(output->sparseWeights[i]))
				ereport(ERROR,
						(errcode(ERRCODE_DATA_EXCEPTION),
						 errmsg("llama_embed sparse engine returned a non-finite weight")));
		}
		return;
	}

	label = spec->outputMode == PG_LLAMA_EMBED_OUTPUT_DENSE ?
		"llama_embed" : "ColBERT";

	if (output->count <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("%s engine returned no vectors", label)));
	if (output->count > spec->maxVectors)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("%s engine returned %d vectors but limit is %d",
						label, output->count, spec->maxVectors)));
	if (spec->checkExpectedDim && output->dim != spec->expectedDim)
	{
		if (spec->outputMode == PG_LLAMA_EMBED_OUTPUT_DENSE)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("llama_embed embedding dimension %d does not match expected dimension %d",
							output->dim, spec->expectedDim)));
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("%s embedding dimension %d does not match expected dimension %d",
						label,
						output->dim, spec->expectedDim),
				 errhint("The GGUF must include the ColBERT dense projection. Reconvert with llama.cpp convert_hf_to_gguf.py --sentence-transformers-dense-modules or provide a GGUF whose embedding output is 128-dimensional.")));
	}

	for (int32 i = 0; i < output->count * output->dim; i++)
	{
		if (!isfinite(output->values[i]))
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("ColBERT engine returned a non-finite value")));
	}

	if (spec->normalize)
	{
		for (int32 i = 0; i < output->count; i++)
		{
			double		norm = 0.0;
			const float4 *values =
				output->values + ((Size) i * (Size) output->dim);

			for (int32 j = 0; j < output->dim; j++)
				norm += (double) values[j] * (double) values[j];
			if (fabs(sqrt(norm) - 1.0) > 1e-3)
				ereport(ERROR,
						(errcode(ERRCODE_DATA_EXCEPTION),
						 errmsg("ColBERT engine returned an unnormalized token vector")));
		}
	}
}

static void
PgColbertEncodeOrError(text *modelText, text *inputText,
					   MemoryContext ctx,
					   PgColbertModelSpec *spec,
					   PgColbertEngineOutput *output,
					   bool debugTokens)
{
	char	   *input = text_to_cstring(inputText);
	char	   *errorMessage = NULL;

	PgColbertParseModel(modelText, spec);
	spec->debugTokens = debugTokens;
	if (!PgColbertEngineEncode(spec, input, ctx, output, &errorMessage))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("%s", errorMessage != NULL ? errorMessage : "ColBERT engine failed")));
	PgColbertValidateOutput(spec, output);
}

static void
PgLlamaEmbedEncodeOrError(text *modelText, text *inputText, Jsonb *options,
						  PgLlamaEmbedOutputMode defaultMode,
						  MemoryContext ctx,
						  PgColbertModelSpec *spec,
						  PgColbertEngineOutput *output)
{
	char	   *input = text_to_cstring(inputText);
	char	   *errorMessage = NULL;

	PgLlamaEmbedParseModel(modelText, options, defaultMode, spec);
	if (!PgColbertEngineEncode(spec, input, ctx, output, &errorMessage))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("%s", errorMessage != NULL ? errorMessage : "llama_embed engine failed")));
	PgColbertValidateOutput(spec, output);
}

static int32
PgColbertEncodeBatchOrError(text *modelText, ArrayType *inputsArray,
							MemoryContext ctx,
							PgColbertModelSpec *spec,
							PgColbertEngineOutput **outputs,
							bool debugTokens)
{
	Datum	   *inputDatums;
	bool	   *inputNulls;
	int			inputCount;
	const char **inputs;
	char	   *errorMessage = NULL;

	if (ARR_ELEMTYPE(inputsArray) != TEXTOID)
		ereport(ERROR,
				(errcode(ERRCODE_DATATYPE_MISMATCH),
				 errmsg("colbert_mv_batch inputs must be text[]")));

	deconstruct_array(inputsArray, TEXTOID, -1, false, TYPALIGN_INT,
					  &inputDatums, &inputNulls, &inputCount);

	PgColbertParseModel(modelText, spec);
	spec->debugTokens = debugTokens;
	*outputs =
		(PgColbertEngineOutput *) palloc0(sizeof(PgColbertEngineOutput) *
										  (Size) inputCount);
	if (inputCount == 0)
		return 0;

	inputs = (const char **) palloc0(sizeof(char *) * (Size) inputCount);
	for (int i = 0; i < inputCount; i++)
	{
		if (inputNulls[i])
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("colbert_mv_batch inputs cannot contain null elements")));
		inputs[i] = TextDatumGetCString(inputDatums[i]);
	}

	if (!PgColbertEngineEncodeBatch(spec, inputs, inputCount, ctx, *outputs,
									&errorMessage))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("%s", errorMessage != NULL ? errorMessage : "ColBERT engine failed")));
	for (int i = 0; i < inputCount; i++)
		PgColbertValidateOutput(spec, &(*outputs)[i]);

	return (int32) inputCount;
}

static int32
PgLlamaEmbedEncodeBatchOrError(text *modelText, ArrayType *inputsArray,
							   Jsonb *options,
							   PgLlamaEmbedOutputMode defaultMode,
							   MemoryContext ctx,
							   PgColbertModelSpec *spec,
							   PgColbertEngineOutput **outputs)
{
	Datum	   *inputDatums;
	bool	   *inputNulls;
	int			inputCount;
	const char **inputs;
	char	   *errorMessage = NULL;

	if (ARR_ELEMTYPE(inputsArray) != TEXTOID)
		ereport(ERROR,
				(errcode(ERRCODE_DATATYPE_MISMATCH),
				 errmsg("llama_embed batch inputs must be text[]")));

	deconstruct_array(inputsArray, TEXTOID, -1, false, TYPALIGN_INT,
					  &inputDatums, &inputNulls, &inputCount);

	PgLlamaEmbedParseModel(modelText, options, defaultMode, spec);
	*outputs =
		(PgColbertEngineOutput *) palloc0(sizeof(PgColbertEngineOutput) *
										  (Size) inputCount);
	if (inputCount == 0)
		return 0;

	inputs = (const char **) palloc0(sizeof(char *) * (Size) inputCount);
	for (int i = 0; i < inputCount; i++)
	{
		if (inputNulls[i])
			ereport(ERROR,
					(errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
					 errmsg("llama_embed batch inputs cannot contain null elements")));
		inputs[i] = TextDatumGetCString(inputDatums[i]);
	}

	if (!PgColbertEngineEncodeBatch(spec, inputs, inputCount, ctx, *outputs,
									&errorMessage))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("%s", errorMessage != NULL ? errorMessage : "llama_embed engine failed")));
	for (int i = 0; i < inputCount; i++)
		PgColbertValidateOutput(spec, &(*outputs)[i]);

	return (int32) inputCount;
}

static PgColbertVector *
PgColbertMakeVector(const float4 *values, int32 dim)
{
	Size		size;
	PgColbertVector *vector;

	size = PG_COLBERT_LLAMA_VECTOR_SIZE(dim);
	vector = (PgColbertVector *) palloc0(size);
	SET_VARSIZE(vector, size);
	vector->dim = (int16) dim;
	vector->unused = 0;
	memcpy(vector->x, values, sizeof(float4) * (Size) dim);

	return vector;
}

static ArrayType *
PgColbertBuildVectorArray(const PgColbertEngineOutput *output)
{
	Datum	   *datums;
	Oid			vectorOid;
	int16		typlen;
	bool		typbyval;
	char		typalign;

	vectorOid = PgColbertVectorTypeOid();
	get_typlenbyvalalign(vectorOid, &typlen, &typbyval, &typalign);
	datums = (Datum *) palloc0(sizeof(Datum) * (Size) output->count);
	for (int32 i = 0; i < output->count; i++)
	{
		const float4 *values =
			output->values + ((Size) i * (Size) output->dim);

		datums[i] = PointerGetDatum(PgColbertMakeVector(values, output->dim));
	}

	return construct_array(datums, output->count, vectorOid, typlen, typbyval,
						   typalign);
}

static Datum
PgColbertBuildDenseVector(const PgColbertEngineOutput *output)
{
	if (output->count != 1)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("llama_embed dense output expected one vector but engine returned %d",
						output->count)));
	return PointerGetDatum(PgColbertMakeVector(output->values, output->dim));
}

static ArrayType *
PgColbertBuildDenseVectorArray(PgColbertEngineOutput *outputs,
							   int32 outputCount)
{
	Datum	   *datums;
	Oid			vectorOid;
	int16		typlen;
	bool		typbyval;
	char		typalign;

	vectorOid = PgColbertVectorTypeOid();
	if (outputCount == 0)
		return construct_empty_array(vectorOid);

	get_typlenbyvalalign(vectorOid, &typlen, &typbyval, &typalign);
	datums = (Datum *) palloc0(sizeof(Datum) * (Size) outputCount);
	for (int32 i = 0; i < outputCount; i++)
		datums[i] = PgColbertBuildDenseVector(&outputs[i]);

	return construct_array(datums, outputCount, vectorOid, typlen, typbyval,
						   typalign);
}

static ArrayType *
PgColbertBuildFloat4Array(const PgColbertEngineOutput *output)
{
	Datum	   *datums;
	int32		nelems = output->count * output->dim;

	datums = (Datum *) palloc0(sizeof(Datum) * (Size) nelems);
	for (int32 i = 0; i < nelems; i++)
		datums[i] = Float4GetDatum(output->values[i]);

	return construct_array(datums, nelems, FLOAT4OID, sizeof(float4), true,
						   TYPALIGN_INT);
}

static Datum
PgColbertBuildMultiVector(const PgColbertEngineOutput *output)
{
	Oid			argtypes[2];
	Oid			constructorOid;
	ArrayType  *array;

	argtypes[0] = get_array_type(FLOAT4OID);
	argtypes[1] = INT4OID;
	constructorOid =
		PgColbertLookupPgturbohybridFunction("turbohybrid_multivector_from_float4",
											 2, argtypes, true);
	if (OidIsValid(constructorOid))
	{
		array = PgColbertBuildFloat4Array(output);
		return OidFunctionCall2(constructorOid,
								PointerGetDatum(array),
								Int32GetDatum(output->dim));
	}

	argtypes[0] = get_array_type(PgColbertVectorTypeOid());
	constructorOid =
		PgColbertLookupPgturbohybridFunction("turbohybrid_multivector",
											 1, argtypes, true);
	if (!OidIsValid(constructorOid))
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_FUNCTION),
				 errmsg("could not find pgturbohybrid multivector constructor")));

	array = PgColbertBuildVectorArray(output);
	return OidFunctionCall1(constructorOid, PointerGetDatum(array));
}

static ArrayType *
PgColbertBuildMultiVectorArray(PgColbertEngineOutput *outputs,
							   int32 outputCount,
							   Oid multivectorOid)
{
	Datum	   *datums;
	int16		typlen;
	bool		typbyval;
	char		typalign;

	if (outputCount == 0)
		return construct_empty_array(multivectorOid);

	get_typlenbyvalalign(multivectorOid, &typlen, &typbyval, &typalign);
	datums = (Datum *) palloc0(sizeof(Datum) * (Size) outputCount);
	for (int32 i = 0; i < outputCount; i++)
		datums[i] = PgColbertBuildMultiVector(&outputs[i]);

	return construct_array(datums, outputCount, multivectorOid, typlen,
						   typbyval, typalign);
}

static void
PgColbertAppendJsonString(StringInfo buf, const char *value)
{
	appendStringInfoChar(buf, '"');
	for (const unsigned char *p = (const unsigned char *) value; *p != '\0'; p++)
	{
		switch (*p)
		{
			case '\\':
			case '"':
				appendStringInfo(buf, "\\%c", *p);
				break;
			case '\b':
				appendStringInfoString(buf, "\\b");
				break;
			case '\f':
				appendStringInfoString(buf, "\\f");
				break;
			case '\n':
				appendStringInfoString(buf, "\\n");
				break;
			case '\r':
				appendStringInfoString(buf, "\\r");
				break;
			case '\t':
				appendStringInfoString(buf, "\\t");
				break;
			default:
				if (*p < 0x20)
					appendStringInfo(buf, "\\u%04x", *p);
				else
					appendStringInfoChar(buf, (char) *p);
				break;
		}
	}
	appendStringInfoChar(buf, '"');
}

static const char *
PgLlamaEmbedOutputModeName(PgLlamaEmbedOutputMode mode)
{
	switch (mode)
	{
		case PG_LLAMA_EMBED_OUTPUT_DENSE:
			return "dense";
		case PG_LLAMA_EMBED_OUTPUT_TOKENS:
			return "tokens";
		case PG_LLAMA_EMBED_OUTPUT_SPARSE:
			return "sparse";
	}
	return "tokens";
}

static const char *
PgLlamaEmbedPoolingName(PgLlamaEmbedPooling pooling)
{
	switch (pooling)
	{
		case PG_LLAMA_EMBED_POOLING_NONE:
			return "none";
		case PG_LLAMA_EMBED_POOLING_MEAN:
			return "mean";
		case PG_LLAMA_EMBED_POOLING_CLS:
			return "cls";
		case PG_LLAMA_EMBED_POOLING_LAST:
			return "last";
		case PG_LLAMA_EMBED_POOLING_RANK:
			return "rank";
	}
	return "none";
}

static void
PgColbertAppendOutputJson(StringInfo buf,
						  const PgColbertModelSpec *spec,
						  const PgColbertEngineOutput *output,
						  bool includeVectors)
{
	appendStringInfoChar(buf, '{');
	appendStringInfoString(buf, "\"engine\":");
	PgColbertAppendJsonString(buf, output->engine);
	appendStringInfoString(buf, ",\"alias\":");
	PgColbertAppendJsonString(buf, spec->alias);
	appendStringInfoString(buf, ",\"role\":");
	PgColbertAppendJsonString(buf, spec->roleName);
	appendStringInfoString(buf, ",\"mode\":");
	PgColbertAppendJsonString(buf, PgLlamaEmbedOutputModeName(spec->outputMode));
	appendStringInfoString(buf, ",\"pooling\":");
	PgColbertAppendJsonString(buf, PgLlamaEmbedPoolingName(spec->pooling));
	appendStringInfoString(buf, ",\"profile_source\":");
	PgColbertAppendJsonString(buf, spec->profile.sourceName);
	appendStringInfo(buf, ",\"dim\":%d,\"count\":%d,\"normalized\":%s",
					 output->dim, output->count,
					 output->normalized ? "true" : "false");
	appendStringInfoString(buf, ",\"token_ids\":[");
	for (int32 i = 0; i < output->count; i++)
	{
		if (i > 0)
			appendStringInfoChar(buf, ',');
		appendStringInfo(buf, "%d", output->tokenIds[i]);
	}
	appendStringInfoChar(buf, ']');

	if (includeVectors)
	{
		appendStringInfoString(buf, ",\"vectors\":[");
		for (int32 i = 0; i < output->count; i++)
		{
			if (i > 0)
				appendStringInfoChar(buf, ',');
			appendStringInfoChar(buf, '[');
			for (int32 j = 0; j < output->dim; j++)
			{
				if (j > 0)
					appendStringInfoChar(buf, ',');
				appendStringInfo(buf, "%.9g",
								 output->values[(Size) i * (Size) output->dim + (Size) j]);
			}
			appendStringInfoChar(buf, ']');
		}
		appendStringInfoChar(buf, ']');
	}
	appendStringInfoChar(buf, '}');
}

/*
 * Build a turbohybrid_sparse_vector from the engine's raw (term_id, weight) bag.
 * De-duplication, drop_non_positive, min_weight, top_k, sort and normalization
 * are delegated to pgturbohybrid's turbohybrid_sparse_vector_build(int4[],
 * float4[], jsonb), which reads those keys (with the documented defaults) out of
 * the llama_embed options object passed straight through.
 */
static Datum
PgLlamaEmbedSparseBuild(const PgColbertEngineOutput *output, Jsonb *options)
{
	Oid			argtypes[3];
	Oid			buildOid;
	Datum	   *idDatums;
	Datum	   *weightDatums;
	ArrayType  *idArray;
	ArrayType  *weightArray;
	int32		n = output->sparseCount;

	argtypes[0] = INT4ARRAYOID;
	argtypes[1] = FLOAT4ARRAYOID;
	argtypes[2] = JSONBOID;
	buildOid = PgColbertLookupPgturbohybridFunction("turbohybrid_sparse_vector_build",
													3, argtypes, false);

	idDatums = (Datum *) palloc(sizeof(Datum) * (Size) Max(n, 1));
	weightDatums = (Datum *) palloc(sizeof(Datum) * (Size) Max(n, 1));
	for (int32 i = 0; i < n; i++)
	{
		idDatums[i] = Int32GetDatum(output->sparseTermIds[i]);
		weightDatums[i] = Float4GetDatum(output->sparseWeights[i]);
	}
	idArray = construct_array(idDatums, n, INT4OID, sizeof(int32), true,
							  TYPALIGN_INT);
	weightArray = construct_array(weightDatums, n, FLOAT4OID, sizeof(float4),
								  true, TYPALIGN_INT);

	return OidFunctionCall3(buildOid,
							PointerGetDatum(idArray),
							PointerGetDatum(weightArray),
							PointerGetDatum(options));
}

/*
 * Emit the post-processed sparse output (the same term_ids/weights that
 * llama_embed_sparse would store) as a JSON object for the llama_embed debug
 * view.  Reads them back from the built turbohybrid_sparse_vector so the JSON
 * and typed paths are always consistent.
 */
static void
PgLlamaEmbedAppendSparseJson(StringInfo buf, const PgColbertModelSpec *spec,
							 const PgColbertEngineOutput *output, Jsonb *options)
{
	Datum		sparseDatum = PgLlamaEmbedSparseBuild(output, options);
	Oid			sparseType = PgColbertSparseVectorTypeOid();
	Oid			argtypes[1];
	Oid			termIdsOid;
	Oid			weightsOid;
	ArrayType  *idArr;
	ArrayType  *wtArr;
	Datum	   *idDatums;
	Datum	   *wtDatums;
	bool	   *idNulls;
	bool	   *wtNulls;
	int			idCount;
	int			wtCount;

	argtypes[0] = sparseType;
	termIdsOid =
		PgColbertLookupPgturbohybridFunction("turbohybrid_sparse_vector_term_ids",
											 1, argtypes, false);
	weightsOid =
		PgColbertLookupPgturbohybridFunction("turbohybrid_sparse_vector_weights",
											 1, argtypes, false);
	idArr = DatumGetArrayTypeP(OidFunctionCall1(termIdsOid, sparseDatum));
	wtArr = DatumGetArrayTypeP(OidFunctionCall1(weightsOid, sparseDatum));
	deconstruct_array(idArr, INT4OID, sizeof(int32), true, TYPALIGN_INT,
					  &idDatums, &idNulls, &idCount);
	deconstruct_array(wtArr, FLOAT4OID, sizeof(float4), true, TYPALIGN_INT,
					  &wtDatums, &wtNulls, &wtCount);

	appendStringInfoChar(buf, '{');
	appendStringInfoString(buf, "\"engine\":");
	PgColbertAppendJsonString(buf, output->engine);
	appendStringInfoString(buf, ",\"alias\":");
	PgColbertAppendJsonString(buf, spec->alias);
	appendStringInfoString(buf, ",\"role\":");
	PgColbertAppendJsonString(buf, spec->roleName);
	appendStringInfoString(buf, ",\"mode\":");
	PgColbertAppendJsonString(buf, PgLlamaEmbedOutputModeName(spec->outputMode));
	appendStringInfoString(buf, ",\"profile_source\":");
	PgColbertAppendJsonString(buf, spec->profile.sourceName);
	appendStringInfo(buf, ",\"vocab_size\":%d,\"count\":%d",
					 output->sparseVocabSize, idCount);
	appendStringInfoString(buf, ",\"term_ids\":[");
	for (int i = 0; i < idCount; i++)
	{
		if (i > 0)
			appendStringInfoChar(buf, ',');
		appendStringInfo(buf, "%d", DatumGetInt32(idDatums[i]));
	}
	appendStringInfoString(buf, "],\"weights\":[");
	for (int i = 0; i < wtCount; i++)
	{
		if (i > 0)
			appendStringInfoChar(buf, ',');
		appendStringInfo(buf, "%.9g", (double) DatumGetFloat4(wtDatums[i]));
	}
	appendStringInfoString(buf, "]}");
}

static void
PgColbertAppendNullableInt(StringInfo buf, int32 value)
{
	if (value < 0)
		appendStringInfoString(buf, "null");
	else
		appendStringInfo(buf, "%d", value);
}

static void
PgColbertAppendProjectionModulesJson(StringInfo buf,
									 const PgColbertRuntimeProfile *profile)
{
	appendStringInfoChar(buf, '[');
	for (int32 i = 0; i < profile->projectionModuleCount; i++)
	{
		const PgColbertProjectionModuleProfile *module =
			&profile->projectionModules[i];

		if (i > 0)
			appendStringInfoChar(buf, ',');
		appendStringInfoChar(buf, '{');
		appendStringInfoString(buf, "\"type\":");
		PgColbertAppendJsonString(buf, module->type);
		appendStringInfoString(buf, ",\"activation\":");
		PgColbertAppendJsonString(buf, module->activation[0] != '\0' ?
								  module->activation : "identity");
		appendStringInfoString(buf, ",\"weight_tensor\":");
		if (module->weightTensor[0] != '\0')
			PgColbertAppendJsonString(buf, module->weightTensor);
		else
			appendStringInfoString(buf, "null");
		appendStringInfoString(buf, ",\"bias_tensor\":");
		if (module->biasTensor[0] != '\0')
			PgColbertAppendJsonString(buf, module->biasTensor);
		else
			appendStringInfoString(buf, "null");
		appendStringInfo(buf,
						 ",\"input_dim\":%d,\"output_dim\":%d,\"bias\":%s",
						 module->inputDim, module->outputDim,
						 module->hasBias ? "true" : "false");
		if (module->p > 0)
			appendStringInfo(buf, ",\"p\":%d", module->p);
		appendStringInfoChar(buf, '}');
	}
	appendStringInfoChar(buf, ']');
}

static void
PgColbertAppendProfileJson(StringInfo buf,
						   const PgColbertModelSpec *spec)
{
	const PgColbertRuntimeProfile *profile = &spec->profile;

	appendStringInfo(buf, ",\"profile_loaded\":%s",
					 profile->loaded ? "true" : "false");
	appendStringInfoString(buf, ",\"profile_source\":");
	PgColbertAppendJsonString(buf, profile->sourceName);
	appendStringInfoString(buf, ",\"profile_schema\":");
	PgColbertAppendJsonString(buf, profile->schema);
	appendStringInfoString(buf, ",\"compatibility_level\":");
	PgColbertAppendJsonString(buf, profile->compatibilityLevel);
	appendStringInfoString(buf, ",\"tokenizer_source\":");
	PgColbertAppendJsonString(buf, profile->tokenizerSource);
	appendStringInfoString(buf, ",\"tokenizer_status\":");
	PgColbertAppendJsonString(buf, profile->tokenizerStatus);
	appendStringInfoString(buf, ",\"tokenizer_known_limitations\":");
	PgColbertAppendJsonString(buf, profile->tokenizerKnownLimitations);
	appendStringInfoString(buf, ",\"known_limitations\":");
	PgColbertAppendJsonString(buf, profile->knownLimitations);
	appendStringInfo(buf,
					 ",\"query_length_source\":%s,\"document_length_source\":%s",
					 profile->loaded ? "\"profile\"" : "\"guc\"",
					 profile->loaded ? "\"profile\"" : "\"guc\"");
	appendStringInfo(buf, ",\"query_max_length\":%d,\"query_pad_to\":",
					 profile->queryMaxLength);
	PgColbertAppendNullableInt(buf, profile->queryPadTo);
	appendStringInfo(buf, ",\"document_max_length\":%d",
					 profile->documentMaxLength);
	appendStringInfo(buf, ",\"skiplist_token_count\":%d",
					 profile->skiplistTokenCount);
	appendStringInfoString(buf, ",\"attention_mask_status\":");
	if (spec->role == PG_COLBERT_ROLE_QUERY &&
		profile->loaded &&
		profile->strictPylateProfile &&
		!profile->queryAttendToExpansionTokens)
		PgColbertAppendJsonString(buf, "approximated");
	else
		PgColbertAppendJsonString(buf, "ok");
	appendStringInfo(buf, ",\"strict_profile\":%s",
					 spec->strictProfile ? "true" : "false");
	appendStringInfoString(buf, ",\"projection_kind\":");
	PgColbertAppendJsonString(buf, profile->projectionKind);
	appendStringInfoString(buf, ",\"projection_modules\":");
	PgColbertAppendProjectionModulesJson(buf, profile);
}

static void
PgColbertAppendDebugJson(StringInfo buf,
						 const PgColbertModelSpec *spec,
						 const PgColbertEngineOutput *output)
{
	appendStringInfoChar(buf, '{');
	appendStringInfoString(buf, "\"engine\":");
	PgColbertAppendJsonString(buf, output->engine);
	appendStringInfoString(buf, ",\"alias\":");
	PgColbertAppendJsonString(buf, spec->alias);
	appendStringInfoString(buf, ",\"role\":");
	PgColbertAppendJsonString(buf, spec->roleName);
	appendStringInfoString(buf, ",\"profile_source\":");
	PgColbertAppendJsonString(buf, spec->profile.sourceName);
	appendStringInfo(buf, ",\"dim\":%d,\"vector_count\":%d,\"normalized\":%s",
					 output->dim, output->count,
					 output->normalized ? "true" : "false");
	appendStringInfoString(buf, ",\"input\":");
	PgColbertAppendJsonString(buf, output->input != NULL ? output->input : "");
	appendStringInfoString(buf, ",\"prefix\":");
	PgColbertAppendJsonString(buf, output->prefix != NULL ? output->prefix : spec->prefix);
	appendStringInfoString(buf, ",\"attention_mask_status\":");
	PgColbertAppendJsonString(buf, output->attentionMaskStatus != NULL ?
							  output->attentionMaskStatus : "ok");
	appendStringInfoString(buf, ",\"known_limitations\":");
	PgColbertAppendJsonString(buf, output->knownLimitations != NULL ?
							  output->knownLimitations : spec->profile.knownLimitations);
	appendStringInfo(buf,
					 ",\"timing_us\":{\"total\":%lld,\"tokenization\":%lld,\"llama\":%lld,\"output\":%lld,\"debug\":%lld,\"projection\":%lld,\"inputs\":%lld,\"tokens\":%lld,\"output_vectors\":%lld}",
					 (long long) output->timing.totalUs,
					 (long long) output->timing.tokenizationUs,
					 (long long) output->timing.llamaUs,
					 (long long) output->timing.outputUs,
					 (long long) output->timing.debugUs,
					 (long long) output->timing.projectionUs,
					 (long long) output->timing.inputs,
					 (long long) output->timing.tokens,
					 (long long) output->timing.outputVectors);
	appendStringInfoString(buf, ",\"token_plan\":{\"tokens\":[");
	for (int32 i = 0;
		 output->tokenDebug != NULL && i < output->planTokenCount;
		 i++)
	{
		const PgColbertTokenDebug *token = &output->tokenDebug[i];

		if (i > 0)
			appendStringInfoChar(buf, ',');
		appendStringInfoChar(buf, '{');
		appendStringInfo(buf, "\"index\":%d,\"id\":%d,\"piece\":",
						 token->index, token->id);
		if (token->piece != NULL)
			PgColbertAppendJsonString(buf, token->piece);
		else
			appendStringInfoString(buf, "null");
		appendStringInfo(buf, ",\"position_id\":%d,\"token_type_id\":",
						 token->positionId);
		PgColbertAppendNullableInt(buf, token->tokenTypeId);
		appendStringInfoString(buf, ",\"attention_mask\":");
		PgColbertAppendNullableInt(buf, token->attentionMask);
		appendStringInfo(buf,
						 ",\"output_enabled\":%s,\"retained\":%s,\"retain_reason\":",
						 token->outputEnabled ? "true" : "false",
						 token->retained ? "true" : "false");
		PgColbertAppendJsonString(buf, token->retainReason != NULL ?
								  token->retainReason : "");
		appendStringInfoChar(buf, '}');
	}
	appendStringInfoString(buf, "]}}");
}

static void
PgColbertAppendModelInfoJson(StringInfo buf,
							 const PgColbertModelSpec *spec,
							 const PgColbertEngineModelInfo *info)
{
	const char *projectionStatus =
		info->projectionStatus != NULL ? info->projectionStatus :
		(!spec->checkExpectedDim || info->nEmbdOut == spec->expectedDim ? "ok" :
		 "missing_or_unexpected_dim");

	appendStringInfoChar(buf, '{');
	appendStringInfoString(buf, "\"engine\":");
	PgColbertAppendJsonString(buf, info->engine);
	appendStringInfoString(buf, ",\"engine_implemented\":");
	appendStringInfoString(buf, info->implemented ? "true" : "false");
	appendStringInfoString(buf, ",\"alias\":");
	PgColbertAppendJsonString(buf, spec->alias);
	appendStringInfoString(buf, ",\"role\":");
	PgColbertAppendJsonString(buf, spec->roleName);
	appendStringInfoString(buf, ",\"mode\":");
	PgColbertAppendJsonString(buf, PgLlamaEmbedOutputModeName(spec->outputMode));
	appendStringInfoString(buf, ",\"pooling\":");
	PgColbertAppendJsonString(buf, PgLlamaEmbedPoolingName(spec->pooling));
	appendStringInfoString(buf, ",\"path\":");
	PgColbertAppendJsonString(buf, info->path);
	appendStringInfo(buf,
					 ",\"max_vectors\":%d,\"query_length\":%d,\"expected_dim\":%d,\"n_embd_out\":%d",
					 spec->maxVectors, spec->queryLength, spec->expectedDim,
					 info->nEmbdOut);
	appendStringInfoString(buf, ",\"projection_status\":");
	PgColbertAppendJsonString(buf, projectionStatus);
	appendStringInfo(buf, ",\"projection_module_count\":%d",
					 info->projectionModuleCount);
	appendStringInfo(buf,
					 ",\"n_ctx\":%d,\"n_batch\":%d,\"batch_sequences\":%d,\"threads\":%d,\"n_gpu_layers\":%d,\"cache_size\":%d,\"require_normalized\":%s",
					 spec->nCtx, spec->nBatch, spec->batchSequences,
					 spec->threads, spec->nGpuLayers, spec->cacheSize,
					 pg_colbert_llama_require_normalized ? "true" : "false");
	appendStringInfo(buf,
					 ",\"n_ctx_train\":%d,\"n_layer\":%d,\"n_head\":%d,\"loaded_from_cache\":%s",
					 info->nCtxTrain, info->nLayer, info->nHead,
					 info->loadedFromCache ? "true" : "false");
	PgColbertAppendProfileJson(buf, spec);
	appendStringInfoChar(buf, '}');
}

static Datum
PgColbertJsonbFromCString(const char *json)
{
	return DirectFunctionCall1(jsonb_in, CStringGetDatum(json));
}

Datum
pg_colbert_llama_colbert(PG_FUNCTION_ARGS)
{
	PgColbertModelSpec spec;
	PgColbertEngineOutput output;
	StringInfoData buf;

	PgColbertEncodeOrError(PG_GETARG_TEXT_PP(0), PG_GETARG_TEXT_PP(1),
						   CurrentMemoryContext, &spec, &output, false);
	initStringInfo(&buf);
	PgColbertAppendOutputJson(&buf, &spec, &output, true);
	PG_RETURN_DATUM(PgColbertJsonbFromCString(buf.data));
}

Datum
pg_colbert_llama_colbert_vectors(PG_FUNCTION_ARGS)
{
	PgColbertModelSpec spec;
	PgColbertEngineOutput output;
	ArrayType  *array;

	PgColbertEncodeOrError(PG_GETARG_TEXT_PP(0), PG_GETARG_TEXT_PP(1),
						   CurrentMemoryContext, &spec, &output, false);

	array = PgColbertBuildVectorArray(&output);
	PG_RETURN_ARRAYTYPE_P(array);
}

Datum
pg_colbert_llama_colbert_float4(PG_FUNCTION_ARGS)
{
	PgColbertModelSpec spec;
	PgColbertEngineOutput output;
	ArrayType  *array;

	PgColbertEncodeOrError(PG_GETARG_TEXT_PP(0), PG_GETARG_TEXT_PP(1),
						   CurrentMemoryContext, &spec, &output, false);

	array = PgColbertBuildFloat4Array(&output);
	PG_RETURN_ARRAYTYPE_P(array);
}

Datum
pg_colbert_llama_colbert_dim(PG_FUNCTION_ARGS)
{
	PgColbertModelSpec spec;
	PgColbertEngineOutput output;

	PgColbertEncodeOrError(PG_GETARG_TEXT_PP(0), PG_GETARG_TEXT_PP(1),
						   CurrentMemoryContext, &spec, &output, false);
	PG_RETURN_INT32(output.dim);
}

Datum
pg_colbert_llama_colbert_mv(PG_FUNCTION_ARGS)
{
	PgColbertModelSpec spec;
	PgColbertEngineOutput output;

	PgColbertEncodeOrError(PG_GETARG_TEXT_PP(0), PG_GETARG_TEXT_PP(1),
						   CurrentMemoryContext, &spec, &output, false);
	PG_RETURN_DATUM(PgColbertBuildMultiVector(&output));
}

Datum
pg_colbert_llama_colbert_mv_batch(PG_FUNCTION_ARGS)
{
	PgColbertModelSpec spec;
	PgColbertEngineOutput *outputs;
	ArrayType  *array;
	Oid			resultType;
	Oid			multivectorOid;
	int32		outputCount;

	outputCount =
		PgColbertEncodeBatchOrError(PG_GETARG_TEXT_PP(0),
									PG_GETARG_ARRAYTYPE_P(1),
									CurrentMemoryContext,
									&spec,
									&outputs,
									false);
	resultType = get_fn_expr_rettype(fcinfo->flinfo);
	multivectorOid = OidIsValid(resultType) ? get_element_type(resultType) :
		InvalidOid;
	if (!OidIsValid(multivectorOid))
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("could not resolve colbert_mv_batch result element type")));

	array = PgColbertBuildMultiVectorArray(outputs, outputCount, multivectorOid);
	PG_RETURN_ARRAYTYPE_P(array);
}

Datum
pg_colbert_llama_colbert_debug(PG_FUNCTION_ARGS)
{
	PgColbertModelSpec spec;
	PgColbertEngineOutput output;
	StringInfoData buf;

	PgColbertEncodeOrError(PG_GETARG_TEXT_PP(0), PG_GETARG_TEXT_PP(1),
						   CurrentMemoryContext, &spec, &output, true);
	initStringInfo(&buf);
	PgColbertAppendDebugJson(&buf, &spec, &output);
	PG_RETURN_DATUM(PgColbertJsonbFromCString(buf.data));
}

Datum
pg_colbert_llama_colbert_model_info(PG_FUNCTION_ARGS)
{
	PgColbertModelSpec spec;
	PgColbertEngineModelInfo info;
	StringInfoData buf;
	char	   *errorMessage = NULL;

	PgColbertParseModel(PG_GETARG_TEXT_PP(0), &spec);
	if (!PgColbertEngineGetModelInfo(&spec, CurrentMemoryContext, &info,
									 &errorMessage))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("%s", errorMessage != NULL ? errorMessage : "ColBERT engine failed")));

	initStringInfo(&buf);
	PgColbertAppendModelInfoJson(&buf, &spec, &info);

	PG_RETURN_DATUM(PgColbertJsonbFromCString(buf.data));
}

Datum
pg_colbert_llama_llama_embed(PG_FUNCTION_ARGS)
{
	PgColbertModelSpec spec;
	PgColbertEngineOutput output;
	StringInfoData buf;
	Jsonb	   *options = PG_GETARG_JSONB_P(2);

	PgLlamaEmbedEncodeOrError(PG_GETARG_TEXT_PP(0), PG_GETARG_TEXT_PP(1),
							  options,
							  PG_LLAMA_EMBED_OUTPUT_TOKENS,
							  CurrentMemoryContext, &spec, &output);
	initStringInfo(&buf);
	if (spec.outputMode == PG_LLAMA_EMBED_OUTPUT_SPARSE)
		PgLlamaEmbedAppendSparseJson(&buf, &spec, &output, options);
	else
		PgColbertAppendOutputJson(&buf, &spec, &output, true);
	PG_RETURN_DATUM(PgColbertJsonbFromCString(buf.data));
}

Datum
pg_colbert_llama_llama_embed_sparse(PG_FUNCTION_ARGS)
{
	PgColbertModelSpec spec;
	PgColbertEngineOutput output;
	Jsonb	   *options = PG_GETARG_JSONB_P(2);

	PgLlamaEmbedEncodeOrError(PG_GETARG_TEXT_PP(0), PG_GETARG_TEXT_PP(1),
							  options,
							  PG_LLAMA_EMBED_OUTPUT_SPARSE,
							  CurrentMemoryContext, &spec, &output);
	PG_RETURN_DATUM(PgLlamaEmbedSparseBuild(&output, options));
}

Datum
pg_colbert_llama_llama_embed_sparse_batch(PG_FUNCTION_ARGS)
{
	PgColbertModelSpec spec;
	PgColbertEngineOutput *outputs;
	Jsonb	   *options = PG_GETARG_JSONB_P(2);
	Datum	   *datums;
	ArrayType  *array;
	Oid			sparseType;
	int16		typlen;
	bool		typbyval;
	char		typalign;
	int32		outputCount;

	outputCount =
		PgLlamaEmbedEncodeBatchOrError(PG_GETARG_TEXT_PP(0),
									   PG_GETARG_ARRAYTYPE_P(1),
									   options,
									   PG_LLAMA_EMBED_OUTPUT_SPARSE,
									   CurrentMemoryContext,
									   &spec,
									   &outputs);
	sparseType = PgColbertSparseVectorTypeOid();
	if (outputCount == 0)
		PG_RETURN_ARRAYTYPE_P(construct_empty_array(sparseType));

	get_typlenbyvalalign(sparseType, &typlen, &typbyval, &typalign);
	datums = (Datum *) palloc0(sizeof(Datum) * (Size) outputCount);
	for (int32 i = 0; i < outputCount; i++)
		datums[i] = PgLlamaEmbedSparseBuild(&outputs[i], options);

	array = construct_array(datums, outputCount, sparseType, typlen, typbyval,
							typalign);
	PG_RETURN_ARRAYTYPE_P(array);
}

Datum
pg_colbert_llama_llama_embed_vector(PG_FUNCTION_ARGS)
{
	PgColbertModelSpec spec;
	PgColbertEngineOutput output;

	PgLlamaEmbedEncodeOrError(PG_GETARG_TEXT_PP(0), PG_GETARG_TEXT_PP(1),
							  PG_GETARG_JSONB_P(2),
							  PG_LLAMA_EMBED_OUTPUT_DENSE,
							  CurrentMemoryContext, &spec, &output);
	PG_RETURN_DATUM(PgColbertBuildDenseVector(&output));
}

Datum
pg_colbert_llama_llama_embed_vector_batch(PG_FUNCTION_ARGS)
{
	PgColbertModelSpec spec;
	PgColbertEngineOutput *outputs;
	ArrayType  *array;
	int32		outputCount;

	outputCount =
		PgLlamaEmbedEncodeBatchOrError(PG_GETARG_TEXT_PP(0),
									   PG_GETARG_ARRAYTYPE_P(1),
									   PG_GETARG_JSONB_P(2),
									   PG_LLAMA_EMBED_OUTPUT_DENSE,
									   CurrentMemoryContext,
									   &spec,
									   &outputs);
	array = PgColbertBuildDenseVectorArray(outputs, outputCount);
	PG_RETURN_ARRAYTYPE_P(array);
}

Datum
pg_colbert_llama_llama_embed_tokens(PG_FUNCTION_ARGS)
{
	PgColbertModelSpec spec;
	PgColbertEngineOutput output;
	ArrayType  *array;

	PgLlamaEmbedEncodeOrError(PG_GETARG_TEXT_PP(0), PG_GETARG_TEXT_PP(1),
							  PG_GETARG_JSONB_P(2),
							  PG_LLAMA_EMBED_OUTPUT_TOKENS,
							  CurrentMemoryContext, &spec, &output);
	array = PgColbertBuildVectorArray(&output);
	PG_RETURN_ARRAYTYPE_P(array);
}

Datum
pg_colbert_llama_llama_embed_mv(PG_FUNCTION_ARGS)
{
	PgColbertModelSpec spec;
	PgColbertEngineOutput output;

	PgLlamaEmbedEncodeOrError(PG_GETARG_TEXT_PP(0), PG_GETARG_TEXT_PP(1),
							  PG_GETARG_JSONB_P(2),
							  PG_LLAMA_EMBED_OUTPUT_TOKENS,
							  CurrentMemoryContext, &spec, &output);
	PG_RETURN_DATUM(PgColbertBuildMultiVector(&output));
}

Datum
pg_colbert_llama_llama_embed_mv_batch(PG_FUNCTION_ARGS)
{
	PgColbertModelSpec spec;
	PgColbertEngineOutput *outputs;
	ArrayType  *array;
	Oid			resultType;
	Oid			multivectorOid;
	int32		outputCount;

	outputCount =
		PgLlamaEmbedEncodeBatchOrError(PG_GETARG_TEXT_PP(0),
									   PG_GETARG_ARRAYTYPE_P(1),
									   PG_GETARG_JSONB_P(2),
									   PG_LLAMA_EMBED_OUTPUT_TOKENS,
									   CurrentMemoryContext,
									   &spec,
									   &outputs);
	resultType = get_fn_expr_rettype(fcinfo->flinfo);
	multivectorOid = OidIsValid(resultType) ? get_element_type(resultType) :
		InvalidOid;
	if (!OidIsValid(multivectorOid))
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("could not resolve llama_embed_mv_batch result element type")));

	array = PgColbertBuildMultiVectorArray(outputs, outputCount, multivectorOid);
	PG_RETURN_ARRAYTYPE_P(array);
}

Datum
pg_colbert_llama_llama_embed_model_info(PG_FUNCTION_ARGS)
{
	PgColbertModelSpec spec;
	PgColbertEngineModelInfo info;
	StringInfoData buf;
	char	   *errorMessage = NULL;

	PgLlamaEmbedParseModel(PG_GETARG_TEXT_PP(0),
						   PG_NARGS() > 1 ? PG_GETARG_JSONB_P(1) : NULL,
						   PG_LLAMA_EMBED_OUTPUT_DENSE,
						   &spec);
	if (!PgColbertEngineGetModelInfo(&spec, CurrentMemoryContext, &info,
									 &errorMessage))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("%s", errorMessage != NULL ? errorMessage : "llama_embed engine failed")));

	initStringInfo(&buf);
	PgColbertAppendModelInfoJson(&buf, &spec, &info);

	PG_RETURN_DATUM(PgColbertJsonbFromCString(buf.data));
}
