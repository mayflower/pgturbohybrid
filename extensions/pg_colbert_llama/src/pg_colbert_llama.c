#include "postgres.h"

#include <ctype.h>
#include <limits.h>
#include <math.h>

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
PG_FUNCTION_INFO_V1(pg_colbert_llama_colbert_model_info);

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
static int	pg_colbert_llama_n_gpu_layers = 0;
static int	pg_colbert_llama_cache_size = 2;
static char *pg_colbert_llama_query_prefix = NULL;
static char *pg_colbert_llama_document_prefix = NULL;
static int	pg_colbert_llama_max_query_vectors = 64;
static int	pg_colbert_llama_max_doc_vectors = 256;
static bool pg_colbert_llama_require_normalized = true;
static int	pg_colbert_llama_expected_dim = 128;
static char *pg_colbert_llama_allowed_models = NULL;
static Oid	pg_colbert_llama_vector_type_oid = InvalidOid;

void		_PG_init(void);

static Oid PgColbertExtensionSchema(Oid extensionOid);
static Oid PgColbertVectorTypeOid(void);
static Oid PgColbertLookupPgturbohybridFunction(const char *funcname,
												int nargs, Oid *argtypes,
												bool missing_ok);
static void PgColbertParseModel(text *modelText, PgColbertModelSpec *spec);
static void PgColbertCheckAllowedModel(const char *alias);
static void PgColbertValidateOutput(const PgColbertModelSpec *spec,
									const PgColbertEngineOutput *output);
static void PgColbertEncodeOrError(text *modelText, text *inputText,
								   MemoryContext ctx,
								   PgColbertModelSpec *spec,
								   PgColbertEngineOutput *output);
static PgColbertVector *PgColbertMakeVector(const float4 *values, int32 dim);
static ArrayType *PgColbertBuildVectorArray(const PgColbertEngineOutput *output);
static ArrayType *PgColbertBuildFloat4Array(const PgColbertEngineOutput *output);
static Datum PgColbertBuildMultiVector(const PgColbertEngineOutput *output);
static void PgColbertAppendJsonString(StringInfo buf, const char *value);
static void PgColbertAppendOutputJson(StringInfo buf,
									  const PgColbertModelSpec *spec,
									  const PgColbertEngineOutput *output,
									  bool includeVectors);
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
							64, 1, 4096,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("pg_colbert_llama.max_doc_vectors",
							"Maximum retained document token vectors.",
							NULL,
							&pg_colbert_llama_max_doc_vectors,
							256, 1, 4096,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("pg_colbert_llama.require_normalized",
							 "Require each ColBERT token vector to be L2-normalized.",
							 NULL,
							 &pg_colbert_llama_require_normalized,
							 true,
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
	spec->nGpuLayers = pg_colbert_llama_n_gpu_layers;
	spec->cacheSize = pg_colbert_llama_cache_size;
	PgColbertCheckAllowedModel(spec->alias);
}

static void
PgColbertValidateOutput(const PgColbertModelSpec *spec,
						const PgColbertEngineOutput *output)
{
	if (output->count <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("ColBERT engine returned no token vectors")));
	if (output->count > spec->maxVectors)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("ColBERT engine returned %d vectors but limit is %d",
						output->count, spec->maxVectors)));
	if (output->dim != spec->expectedDim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("ColBERT embedding dimension %d does not match expected dimension %d",
						output->dim, spec->expectedDim),
				 errhint("The GGUF must include the ColBERT dense projection. Reconvert with llama.cpp convert_hf_to_gguf.py --sentence-transformers-dense-modules or provide a GGUF whose embedding output is 128-dimensional.")));

	for (int32 i = 0; i < output->count * output->dim; i++)
	{
		if (!isfinite(output->values[i]))
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("ColBERT engine returned a non-finite value")));
	}

	if (pg_colbert_llama_require_normalized)
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
					   PgColbertEngineOutput *output)
{
	char	   *input = text_to_cstring(inputText);
	char	   *errorMessage = NULL;

	PgColbertParseModel(modelText, spec);
	if (!PgColbertEngineEncode(spec, input, ctx, output, &errorMessage))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("%s", errorMessage != NULL ? errorMessage : "ColBERT engine failed")));
	PgColbertValidateOutput(spec, output);
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
						   CurrentMemoryContext, &spec, &output);
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
						   CurrentMemoryContext, &spec, &output);

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
						   CurrentMemoryContext, &spec, &output);

	array = PgColbertBuildFloat4Array(&output);
	PG_RETURN_ARRAYTYPE_P(array);
}

Datum
pg_colbert_llama_colbert_dim(PG_FUNCTION_ARGS)
{
	PgColbertModelSpec spec;
	PgColbertEngineOutput output;

	PgColbertEncodeOrError(PG_GETARG_TEXT_PP(0), PG_GETARG_TEXT_PP(1),
						   CurrentMemoryContext, &spec, &output);
	PG_RETURN_INT32(output.dim);
}

Datum
pg_colbert_llama_colbert_mv(PG_FUNCTION_ARGS)
{
	PgColbertModelSpec spec;
	PgColbertEngineOutput output;

	PgColbertEncodeOrError(PG_GETARG_TEXT_PP(0), PG_GETARG_TEXT_PP(1),
						   CurrentMemoryContext, &spec, &output);
	PG_RETURN_DATUM(PgColbertBuildMultiVector(&output));
}

Datum
pg_colbert_llama_colbert_model_info(PG_FUNCTION_ARGS)
{
	PgColbertModelSpec spec;
	PgColbertEngineModelInfo info;
	StringInfoData buf;
	char	   *errorMessage = NULL;
	const char *projectionStatus;

	PgColbertParseModel(PG_GETARG_TEXT_PP(0), &spec);
	if (!PgColbertEngineGetModelInfo(&spec, CurrentMemoryContext, &info,
									 &errorMessage))
		ereport(ERROR,
				(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
				 errmsg("%s", errorMessage != NULL ? errorMessage : "ColBERT engine failed")));
	projectionStatus =
		info.nEmbdOut == spec.expectedDim ? "ok" : "missing_or_unexpected_dim";

	initStringInfo(&buf);
	appendStringInfoChar(&buf, '{');
	appendStringInfoString(&buf, "\"engine\":");
	PgColbertAppendJsonString(&buf, info.engine);
	appendStringInfoString(&buf, ",\"engine_implemented\":");
	appendStringInfoString(&buf, info.implemented ? "true" : "false");
	appendStringInfoString(&buf, ",\"alias\":");
	PgColbertAppendJsonString(&buf, spec.alias);
	appendStringInfoString(&buf, ",\"role\":");
	PgColbertAppendJsonString(&buf, spec.roleName);
	appendStringInfoString(&buf, ",\"path\":");
	PgColbertAppendJsonString(&buf, info.path);
	appendStringInfo(&buf,
					 ",\"max_vectors\":%d,\"expected_dim\":%d,\"n_embd_out\":%d",
					 spec.maxVectors, spec.expectedDim, info.nEmbdOut);
	appendStringInfoString(&buf, ",\"projection_status\":");
	PgColbertAppendJsonString(&buf, projectionStatus);
	appendStringInfo(&buf,
					 ",\"n_ctx\":%d,\"n_batch\":%d,\"threads\":%d,\"n_gpu_layers\":%d,\"cache_size\":%d,\"require_normalized\":%s",
					 spec.nCtx, spec.nBatch, spec.threads, spec.nGpuLayers,
					 spec.cacheSize,
					 pg_colbert_llama_require_normalized ? "true" : "false");
	appendStringInfo(&buf,
					 ",\"n_ctx_train\":%d,\"n_layer\":%d,\"n_head\":%d,\"loaded_from_cache\":%s",
					 info.nCtxTrain, info.nLayer, info.nHead,
					 info.loadedFromCache ? "true" : "false");
	appendStringInfoChar(&buf, '}');

	PG_RETURN_DATUM(PgColbertJsonbFromCString(buf.data));
}
