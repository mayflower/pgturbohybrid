extern "C" {
#include "postgres.h"

#include "lib/stringinfo.h"
#include "utils/memutils.h"

#include "colbert_engine.h"
}

#include "ggml-backend.h"
#include "llama.h"

#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <unistd.h>

typedef struct PgColbertLegacyBertMetadata
{
	bool		hasHiddenSize;
	bool		hasIntermediateSize;
	bool		hasNumHiddenLayers;
	bool		hasNumAttentionHeads;
	bool		hasMaxPositionEmbeddings;
	bool		hasLayerNormEps;
	bool		hasTypeVocabSize;
	bool		hasContextLength;
	bool		hasEmbeddingLength;
	bool		hasFeedForwardLength;
	bool		hasBlockCount;
	bool		hasAttentionHeadCount;
	bool		hasAttentionLayerNormEps;
	bool		hasTokenizerTokenTypeCount;
	bool		hasTokenizerModel;
	bool		hasTokenizerList;
	bool		hasHuggingFaceTokenizer;
	bool		hasPgColbertSchema;
	int64_t		hiddenSize;
	int64_t		intermediateSize;
	int64_t		numHiddenLayers;
	int64_t		numAttentionHeads;
	int64_t		maxPositionEmbeddings;
	int64_t		typeVocabSize;
	double		layerNormEps;
} PgColbertLegacyBertMetadata;

typedef struct PgColbertCachedModel
{
	char	   *alias;
	char	   *path;
	int			nCtx;
	int			nBatch;
	int			nGpuLayers;
	struct llama_model *model;
	struct llama_context *ctx;
	int32		nCtxTrain;
	int32		nEmbdModel;
	int32		nEmbdOut;
	int32		nLayer;
	int32		nHead;
	bool		hasProjection;
	bool		projectionTransposed;
	int32		projectionIn;
	int32		projectionOut;
	float	   *projection;
	uint64		lastUsed;
	struct PgColbertCachedModel *next;
} PgColbertCachedModel;

static bool pg_colbert_llama_backend_initialized = false;
static bool pg_colbert_llama_backend_gpu_loaded = false;
static PgColbertCachedModel *pg_colbert_llama_cache = NULL;
static int	pg_colbert_llama_cache_count = 0;
static uint64 pg_colbert_llama_cache_clock = 0;

static bool PgColbertSetError(MemoryContext ctx, char **errorMessage,
							  const char *fmt,...);
static bool PgColbertResolvePath(const PgColbertModelSpec *spec,
								 MemoryContext ctx, char **path,
								 char **errorMessage);
static bool PgColbertPathUnderDir(const char *path, const char *dir);
static PgColbertCachedModel *PgColbertFindCachedModel(const PgColbertModelSpec *spec,
													  const char *path);
static void PgColbertFreeCachedModel(PgColbertCachedModel *entry);
static void PgColbertEvictCache(int maxEntries, PgColbertCachedModel *protect);
static bool PgColbertLoadModel(const PgColbertModelSpec *spec,
							   const char *path,
							   MemoryContext ctx,
							   PgColbertCachedModel **entry,
							   bool *loadedFromCache,
							   bool *cacheOwned,
							   char **errorMessage);
static int	PgColbertBuildLegacyBertOverrides(const PgColbertModelSpec *spec,
											   const char *path,
											   struct llama_model_kv_override *overrides,
											   int maxOverrides);
static bool PgColbertLoadProjection(const char *path,
									PgColbertCachedModel *entry,
									MemoryContext ctx,
									char **errorMessage);
static bool PgColbertLoadProjectionSidecar(const char *path,
										   PgColbertCachedModel *entry,
										   MemoryContext ctx,
										   char **errorMessage);
static bool PgColbertUnsupportedGgufMetadata(const char *path,
											 const char **reason);
static bool PgColbertIsSpecialToken(const struct llama_vocab *vocab,
									llama_token token);
static bool PgColbertEncodeBody(const PgColbertModelSpec *spec,
								const char *input,
								MemoryContext ctx,
								PgColbertEngineOutput *output,
								char **errorMessage);
static bool PgColbertModelInfoBody(const PgColbertModelSpec *spec,
								   MemoryContext ctx,
								   PgColbertEngineModelInfo *info,
								   char **errorMessage);

#ifdef PG_COLBERT_LLAMA_BACKEND_DIR
static void PgColbertLoadCpuBackendsFromPath(const char *dirPath);
#endif

extern "C" const char *
PgColbertEngineName(void)
{
	return "llama";
}

extern "C" bool
PgColbertEngineImplemented(void)
{
	return true;
}

static bool
PgColbertSetError(MemoryContext ctx, char **errorMessage, const char *fmt,...)
{
	MemoryContext oldCtx;
	va_list		args;
	char		buffer[2048];

	if (errorMessage == NULL)
		return false;

	va_start(args, fmt);
	vsnprintf(buffer, sizeof(buffer), fmt, args);
	va_end(args);

	oldCtx = MemoryContextSwitchTo(ctx);
	*errorMessage = pstrdup(buffer);
	MemoryContextSwitchTo(oldCtx);
	return false;
}

static bool
PgColbertPathUnderDir(const char *path, const char *dir)
{
	size_t		dirLen = strlen(dir);

	if (dirLen == 1 && dir[0] == '/')
		return path[0] == '/';
	if (strncmp(path, dir, dirLen) != 0)
		return false;
	return path[dirLen] == '/';
}

static bool
PgColbertResolvePath(const PgColbertModelSpec *spec, MemoryContext ctx,
					 char **path, char **errorMessage)
{
	MemoryContext oldCtx;
	char		resolvedDir[PATH_MAX];
	char		candidate[PATH_MAX];
	char		resolvedPath[PATH_MAX];
	int			written;

	if (realpath(spec->modelDir, resolvedDir) == NULL)
		return PgColbertSetError(ctx, errorMessage,
								 "could not resolve pg_colbert_llama.model_dir \"%s\": %s",
								 spec->modelDir, strerror(errno));

	written = snprintf(candidate, sizeof(candidate), "%s/%s.gguf",
					   resolvedDir, spec->alias);
	if (written < 0 || written >= (int) sizeof(candidate))
		return PgColbertSetError(ctx, errorMessage,
								 "resolved model path is too long");

	if (realpath(candidate, resolvedPath) == NULL)
		return PgColbertSetError(ctx, errorMessage,
								 "could not resolve model file \"%s\": %s",
								 candidate, strerror(errno));

	if (!PgColbertPathUnderDir(resolvedPath, resolvedDir))
		return PgColbertSetError(ctx, errorMessage,
								 "resolved model file is outside pg_colbert_llama.model_dir");

	oldCtx = MemoryContextSwitchTo(ctx);
	*path = pstrdup(resolvedPath);
	MemoryContextSwitchTo(oldCtx);
	return true;
}

static PgColbertCachedModel *
PgColbertFindCachedModel(const PgColbertModelSpec *spec, const char *path)
{
	PgColbertCachedModel *entry;

	for (entry = pg_colbert_llama_cache; entry != NULL; entry = entry->next)
	{
		if (strcmp(entry->alias, spec->alias) == 0 &&
			strcmp(entry->path, path) == 0 &&
			entry->nCtx == spec->nCtx &&
			entry->nBatch == spec->nBatch &&
			entry->nGpuLayers == spec->nGpuLayers)
		{
			entry->lastUsed = ++pg_colbert_llama_cache_clock;
			return entry;
		}
	}
	return NULL;
}

static void
PgColbertFreeCachedModel(PgColbertCachedModel *entry)
{
	if (entry == NULL)
		return;
	if (entry->ctx != NULL)
		llama_free(entry->ctx);
	if (entry->model != NULL)
		llama_model_free(entry->model);
	free(entry->alias);
	free(entry->path);
	free(entry->projection);
	free(entry);
}

static void
PgColbertEvictCache(int maxEntries, PgColbertCachedModel *protect)
{
	while (pg_colbert_llama_cache_count > maxEntries)
	{
		PgColbertCachedModel *entry = pg_colbert_llama_cache;
		PgColbertCachedModel *prev = NULL;
		PgColbertCachedModel *oldest = NULL;
		PgColbertCachedModel *oldestPrev = NULL;

		while (entry != NULL)
		{
			if (entry != protect &&
				(oldest == NULL || entry->lastUsed < oldest->lastUsed))
			{
				oldest = entry;
				oldestPrev = prev;
			}
			prev = entry;
			entry = entry->next;
		}

		if (oldest == NULL)
			return;

		if (oldestPrev == NULL)
			pg_colbert_llama_cache = oldest->next;
		else
			oldestPrev->next = oldest->next;
		pg_colbert_llama_cache_count--;
		oldest->next = NULL;
		PgColbertFreeCachedModel(oldest);
	}
}

static bool
PgColbertReadExact(FILE *file, void *ptr, size_t size)
{
	return fread(ptr, 1, size, file) == size;
}

static bool
PgColbertSkipBytes(FILE *file, uint64_t bytes)
{
	while (bytes > 0)
	{
		long		step = bytes > (uint64_t) LONG_MAX ? LONG_MAX : (long) bytes;

		if (fseek(file, step, SEEK_CUR) != 0)
			return false;
		bytes -= (uint64_t) step;
	}
	return true;
}

static bool
PgColbertReadGgufString(FILE *file, char *buffer, size_t bufferSize)
{
	uint64_t	len;

	if (bufferSize == 0 || !PgColbertReadExact(file, &len, sizeof(len)))
		return false;

	if (len >= bufferSize)
	{
		if (!PgColbertSkipBytes(file, len))
			return false;
		buffer[0] = '\0';
		return true;
	}

	if (!PgColbertReadExact(file, buffer, (size_t) len))
		return false;
	buffer[len] = '\0';
	return true;
}

static bool
PgColbertSkipGgufValue(FILE *file, uint32_t type)
{
	char		scratch[1];
	uint32_t	arrayType;
	uint64_t	arrayLen;

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
			return PgColbertReadGgufString(file, scratch, sizeof(scratch));
		case 10:
		case 11:
		case 12:
			return PgColbertSkipBytes(file, 8);
		case 9:
			if (!PgColbertReadExact(file, &arrayType, sizeof(arrayType)) ||
				!PgColbertReadExact(file, &arrayLen, sizeof(arrayLen)))
				return false;
			for (uint64_t i = 0; i < arrayLen; i++)
			{
				if (!PgColbertSkipGgufValue(file, arrayType))
					return false;
			}
			return true;
		default:
			return false;
	}
}

static void
PgColbertMarkLegacyBertKey(PgColbertLegacyBertMetadata *metadata,
						   const char *key)
{
	if (strcmp(key, "tokenizer.ggml.model") == 0)
		metadata->hasTokenizerModel = true;
	else if (strcmp(key, "tokenizer.ggml.tokens") == 0)
		metadata->hasTokenizerList = true;
	else if (strcmp(key, "tokenizer.huggingface.json") == 0)
		metadata->hasHuggingFaceTokenizer = true;
	else if (strcmp(key, "pg_colbert.gguf_schema") == 0)
		metadata->hasPgColbertSchema = true;
}

static void
PgColbertSetLegacyBertInt(PgColbertLegacyBertMetadata *metadata,
						  const char *key, int64_t value)
{
	if (strcmp(key, "bert.hidden_size") == 0)
	{
		metadata->hasHiddenSize = true;
		metadata->hiddenSize = value;
	}
	else if (strcmp(key, "bert.intermediate_size") == 0)
	{
		metadata->hasIntermediateSize = true;
		metadata->intermediateSize = value;
	}
	else if (strcmp(key, "bert.num_hidden_layers") == 0)
	{
		metadata->hasNumHiddenLayers = true;
		metadata->numHiddenLayers = value;
	}
	else if (strcmp(key, "bert.num_attention_heads") == 0)
	{
		metadata->hasNumAttentionHeads = true;
		metadata->numAttentionHeads = value;
	}
	else if (strcmp(key, "bert.max_position_embeddings") == 0)
	{
		metadata->hasMaxPositionEmbeddings = true;
		metadata->maxPositionEmbeddings = value;
	}
	else if (strcmp(key, "bert.type_vocab_size") == 0)
	{
		metadata->hasTypeVocabSize = true;
		metadata->typeVocabSize = value;
	}
	else if (strcmp(key, "bert.context_length") == 0)
		metadata->hasContextLength = true;
	else if (strcmp(key, "bert.embedding_length") == 0)
		metadata->hasEmbeddingLength = true;
	else if (strcmp(key, "bert.feed_forward_length") == 0)
		metadata->hasFeedForwardLength = true;
	else if (strcmp(key, "bert.block_count") == 0)
		metadata->hasBlockCount = true;
	else if (strcmp(key, "bert.attention.head_count") == 0)
		metadata->hasAttentionHeadCount = true;
	else if (strcmp(key, "tokenizer.ggml.token_type_count") == 0)
		metadata->hasTokenizerTokenTypeCount = true;
}

static void
PgColbertSetLegacyBertFloat(PgColbertLegacyBertMetadata *metadata,
							const char *key, double value)
{
	if (strcmp(key, "bert.layer_norm_eps") == 0)
	{
		metadata->hasLayerNormEps = true;
		metadata->layerNormEps = value;
	}
	else if (strcmp(key, "bert.attention.layer_norm_epsilon") == 0)
		metadata->hasAttentionLayerNormEps = true;
}

static bool
PgColbertReadLegacyBertMetadata(const char *path,
								PgColbertLegacyBertMetadata *metadata)
{
	FILE	   *file;
	char		magic[4];
	uint32_t	version;
	uint64_t	tensorCount;
	uint64_t	kvCount;
	bool		ok = false;

	memset(metadata, 0, sizeof(*metadata));
	file = fopen(path, "rb");
	if (file == NULL)
		return false;

	if (!PgColbertReadExact(file, magic, sizeof(magic)) ||
		memcmp(magic, "GGUF", sizeof(magic)) != 0 ||
		!PgColbertReadExact(file, &version, sizeof(version)) ||
		!PgColbertReadExact(file, &tensorCount, sizeof(tensorCount)) ||
		!PgColbertReadExact(file, &kvCount, sizeof(kvCount)))
		goto cleanup;

	for (uint64_t i = 0; i < kvCount; i++)
	{
		char		key[128];
		uint32_t	type;

		if (!PgColbertReadGgufString(file, key, sizeof(key)) ||
			!PgColbertReadExact(file, &type, sizeof(type)))
			goto cleanup;

		PgColbertMarkLegacyBertKey(metadata, key);

		switch (type)
		{
			case 4:
			{
				uint32_t	value;

				if (!PgColbertReadExact(file, &value, sizeof(value)))
					goto cleanup;
				PgColbertSetLegacyBertInt(metadata, key, value);
				break;
			}
			case 5:
			{
				int32_t		value;

				if (!PgColbertReadExact(file, &value, sizeof(value)))
					goto cleanup;
				PgColbertSetLegacyBertInt(metadata, key, value);
				break;
			}
			case 10:
			{
				uint64_t	value;

				if (!PgColbertReadExact(file, &value, sizeof(value)))
					goto cleanup;
				if (value <= (uint64_t) INT64_MAX)
					PgColbertSetLegacyBertInt(metadata, key, (int64_t) value);
				break;
			}
			case 11:
			{
				int64_t		value;

				if (!PgColbertReadExact(file, &value, sizeof(value)))
					goto cleanup;
				PgColbertSetLegacyBertInt(metadata, key, value);
				break;
			}
			case 6:
			{
				float		value;

				if (!PgColbertReadExact(file, &value, sizeof(value)))
					goto cleanup;
				PgColbertSetLegacyBertFloat(metadata, key, value);
				break;
			}
			case 12:
			{
				double		value;

				if (!PgColbertReadExact(file, &value, sizeof(value)))
					goto cleanup;
				PgColbertSetLegacyBertFloat(metadata, key, value);
				break;
			}
			default:
				if (!PgColbertSkipGgufValue(file, type))
					goto cleanup;
				break;
		}
	}

	ok = true;

cleanup:
	fclose(file);
	return ok;
}

static void
PgColbertAddIntOverride(struct llama_model_kv_override *overrides,
						int *count, int maxOverrides, const char *key,
						int64_t value)
{
	if (*count >= maxOverrides - 1)
		return;
	overrides[*count].tag = LLAMA_KV_OVERRIDE_TYPE_INT;
	snprintf(overrides[*count].key, sizeof(overrides[*count].key), "%s", key);
	overrides[*count].val_i64 = value;
	(*count)++;
}

static void
PgColbertAddFloatOverride(struct llama_model_kv_override *overrides,
						  int *count, int maxOverrides, const char *key,
						  double value)
{
	if (*count >= maxOverrides - 1)
		return;
	overrides[*count].tag = LLAMA_KV_OVERRIDE_TYPE_FLOAT;
	snprintf(overrides[*count].key, sizeof(overrides[*count].key), "%s", key);
	overrides[*count].val_f64 = value;
	(*count)++;
}

static int
PgColbertBuildLegacyBertOverrides(const PgColbertModelSpec *spec,
								  const char *path,
								  struct llama_model_kv_override *overrides,
								  int maxOverrides)
{
	PgColbertLegacyBertMetadata metadata;
	int			count = 0;

	memset(overrides, 0,
		   sizeof(struct llama_model_kv_override) * (size_t) maxOverrides);
	if (!PgColbertReadLegacyBertMetadata(path, &metadata))
		return 0;

	if (!metadata.hasContextLength && metadata.hasMaxPositionEmbeddings)
		PgColbertAddIntOverride(overrides, &count, maxOverrides,
								"bert.context_length",
								metadata.maxPositionEmbeddings < spec->nCtx ?
								metadata.maxPositionEmbeddings : spec->nCtx);
	if (!metadata.hasEmbeddingLength && metadata.hasHiddenSize)
		PgColbertAddIntOverride(overrides, &count, maxOverrides,
								"bert.embedding_length",
								metadata.hiddenSize);
	if (!metadata.hasFeedForwardLength && metadata.hasIntermediateSize)
		PgColbertAddIntOverride(overrides, &count, maxOverrides,
								"bert.feed_forward_length",
								metadata.intermediateSize);
	if (!metadata.hasBlockCount && metadata.hasNumHiddenLayers)
		PgColbertAddIntOverride(overrides, &count, maxOverrides,
								"bert.block_count",
								metadata.numHiddenLayers);
	if (!metadata.hasAttentionHeadCount && metadata.hasNumAttentionHeads)
		PgColbertAddIntOverride(overrides, &count, maxOverrides,
								"bert.attention.head_count",
								metadata.numAttentionHeads);
	if (!metadata.hasAttentionLayerNormEps && metadata.hasLayerNormEps)
		PgColbertAddFloatOverride(overrides, &count, maxOverrides,
								  "bert.attention.layer_norm_epsilon",
								  metadata.layerNormEps);
	if (!metadata.hasTokenizerTokenTypeCount && metadata.hasTypeVocabSize)
		PgColbertAddIntOverride(overrides, &count, maxOverrides,
								"tokenizer.ggml.token_type_count",
								metadata.typeVocabSize);

	return count;
}

#ifdef PG_COLBERT_LLAMA_BACKEND_DIR
static void
PgColbertLoadCpuBackendsFromPath(const char *dirPath)
{
	DIR		   *dir;
	struct dirent *entry;

	dir = opendir(dirPath);
	if (dir == NULL)
		return;

	while ((entry = readdir(dir)) != NULL)
	{
		char		backendPath[PATH_MAX];
		int			written;

		if (strncmp(entry->d_name, "libggml-cpu", strlen("libggml-cpu")) != 0)
			continue;

		written = snprintf(backendPath, sizeof(backendPath), "%s/%s",
						   dirPath, entry->d_name);
		if (written < 0 || written >= (int) sizeof(backendPath))
			continue;

		ggml_backend_load(backendPath);
	}

	closedir(dir);
}
#endif

static uint64_t
PgColbertAlign(uint64_t value, uint64_t alignment)
{
	return ((value + alignment - 1) / alignment) * alignment;
}

static int
PgColbertGgmlTypeSize(uint32_t type)
{
	switch (type)
	{
		case 0:
			return 4;
		case 1:
			return 2;
		default:
			return 0;
	}
}

static float
PgColbertHalfToFloat(uint16_t value)
{
	uint32_t	sign = ((uint32_t) value & 0x8000U) << 16;
	uint32_t	exp = ((uint32_t) value >> 10) & 0x1fU;
	uint32_t	mant = (uint32_t) value & 0x03ffU;
	uint32_t	out;
	float		result;

	if (exp == 0)
	{
		if (mant == 0)
			out = sign;
		else
		{
			exp = 1;
			while ((mant & 0x0400U) == 0)
			{
				mant <<= 1;
				exp--;
			}
			mant &= 0x03ffU;
			out = sign | ((exp + 127 - 15) << 23) | (mant << 13);
		}
	}
	else if (exp == 31)
		out = sign | 0x7f800000U | (mant << 13);
	else
		out = sign | ((exp + 127 - 15) << 23) | (mant << 13);

	memcpy(&result, &out, sizeof(result));
	return result;
}

static bool
PgColbertReadProjectionData(FILE *file, uint32_t projType, uint64_t dim0,
							uint64_t dim1, PgColbertCachedModel *entry,
							MemoryContext ctx, char **errorMessage)
{
	bool		transposed = false;
	uint64_t	projElements;
	int			typeSize;
	float	   *projection;

	if (dim0 == (uint64_t) entry->nEmbdModel)
	{
		entry->projectionIn = (int32) dim0;
		entry->projectionOut = (int32) dim1;
		transposed = false;
	}
	else if (dim1 == (uint64_t) entry->nEmbdModel)
	{
		entry->projectionIn = (int32) dim1;
		entry->projectionOut = (int32) dim0;
		transposed = true;
	}
	else
		return PgColbertSetError(ctx, errorMessage,
								 "GGUF ColBERT projection shape is %llu x %llu, but model embedding output is %d",
								 (unsigned long long) dim0,
								 (unsigned long long) dim1,
								 entry->nEmbdModel);

	typeSize = PgColbertGgmlTypeSize(projType);
	if (typeSize == 0)
		return PgColbertSetError(ctx, errorMessage,
								 "GGUF ColBERT projection uses unsupported tensor type %u",
								 projType);

	projElements = dim0 * dim1;
	projection = (float *) malloc(sizeof(float) * (size_t) projElements);
	if (projection == NULL)
		return PgColbertSetError(ctx, errorMessage,
								 "out of memory while loading ColBERT projection");

	if (projType == 0)
	{
		if (!PgColbertReadExact(file, projection,
								sizeof(float) * (size_t) projElements))
		{
			free(projection);
			return PgColbertSetError(ctx, errorMessage,
									 "could not read ColBERT projection data");
		}
	}
	else
	{
		for (uint64_t i = 0; i < projElements; i++)
		{
			uint16_t	value;

			if (!PgColbertReadExact(file, &value, sizeof(value)))
			{
				free(projection);
				return PgColbertSetError(ctx, errorMessage,
										 "could not read ColBERT projection data");
			}
			projection[i] = PgColbertHalfToFloat(value);
		}
	}

	entry->projection = projection;
	entry->hasProjection = true;
	entry->projectionTransposed = transposed;
	entry->nEmbdOut = entry->projectionOut;
	return true;
}

static bool
PgColbertLoadProjectionSidecar(const char *path, PgColbertCachedModel *entry,
							   MemoryContext ctx, char **errorMessage)
{
	FILE	   *file;
	char	   *sidecarPath;
	size_t		pathLen = strlen(path);
	char		magic[8];
	uint32_t	projType;
	uint64_t	dim0;
	uint64_t	dim1;
	bool		ok;

	sidecarPath = (char *) malloc(pathLen + strlen(".colbert_proj") + 1);
	if (sidecarPath == NULL)
		return PgColbertSetError(ctx, errorMessage,
								 "out of memory while resolving ColBERT projection sidecar");
	snprintf(sidecarPath, pathLen + strlen(".colbert_proj") + 1,
			 "%s.colbert_proj", path);

	file = fopen(sidecarPath, "rb");
	if (file == NULL)
	{
		free(sidecarPath);
		return true;
	}

	if (!PgColbertReadExact(file, magic, sizeof(magic)) ||
		memcmp(magic, "PGCPROJ1", sizeof(magic)) != 0 ||
		!PgColbertReadExact(file, &projType, sizeof(projType)) ||
		!PgColbertReadExact(file, &dim0, sizeof(dim0)) ||
		!PgColbertReadExact(file, &dim1, sizeof(dim1)))
	{
		fclose(file);
		ok = PgColbertSetError(ctx, errorMessage,
							   "could not parse ColBERT projection sidecar \"%s\"",
							   sidecarPath);
		free(sidecarPath);
		return ok;
	}

	ok = PgColbertReadProjectionData(file, projType, dim0, dim1, entry, ctx,
									 errorMessage);
	fclose(file);
	free(sidecarPath);
	return ok;
}

static bool
PgColbertLoadProjection(const char *path, PgColbertCachedModel *entry,
						MemoryContext ctx, char **errorMessage)
{
	FILE	   *file;
	char		magic[4];
	uint32_t	version;
	uint64_t	tensorCount;
	uint64_t	kvCount;
	uint64_t	alignment = 32;
	bool		found = false;
	uint64_t	dim0 = 0;
	uint64_t	dim1 = 0;
	uint32_t	projType = 0;
	uint64_t	projOffset = 0;
	uint64_t	dataStart;
	bool		ok = false;

	file = fopen(path, "rb");
	if (file == NULL)
		return PgColbertSetError(ctx, errorMessage,
								 "could not read GGUF projection from \"%s\": %s",
								 path, strerror(errno));

	if (!PgColbertReadExact(file, magic, sizeof(magic)) ||
		memcmp(magic, "GGUF", sizeof(magic)) != 0 ||
		!PgColbertReadExact(file, &version, sizeof(version)) ||
		!PgColbertReadExact(file, &tensorCount, sizeof(tensorCount)) ||
		!PgColbertReadExact(file, &kvCount, sizeof(kvCount)))
		goto malformed;

	for (uint64_t i = 0; i < kvCount; i++)
	{
		char		key[128];
		uint32_t	type;

		if (!PgColbertReadGgufString(file, key, sizeof(key)) ||
			!PgColbertReadExact(file, &type, sizeof(type)))
			goto malformed;

		if (strcmp(key, "general.alignment") == 0 && type == 4)
		{
			uint32_t	value;

			if (!PgColbertReadExact(file, &value, sizeof(value)))
				goto malformed;
			if (value > 0)
				alignment = value;
		}
		else if (strcmp(key, "general.alignment") == 0 && type == 10)
		{
			uint64_t	value;

			if (!PgColbertReadExact(file, &value, sizeof(value)))
				goto malformed;
			if (value > 0)
				alignment = value;
		}
		else if (!PgColbertSkipGgufValue(file, type))
			goto malformed;
	}

	for (uint64_t i = 0; i < tensorCount; i++)
	{
		char		name[128];
		uint32_t	nDims;
		uint64_t	dims[4] = {0, 0, 0, 0};
		uint32_t	type;
		uint64_t	offset;

		if (!PgColbertReadGgufString(file, name, sizeof(name)) ||
			!PgColbertReadExact(file, &nDims, sizeof(nDims)))
			goto malformed;
		if (nDims == 0 || nDims > lengthof(dims))
			goto malformed;
		for (uint32_t j = 0; j < nDims; j++)
		{
			if (!PgColbertReadExact(file, &dims[j], sizeof(dims[j])))
				goto malformed;
		}
		if (!PgColbertReadExact(file, &type, sizeof(type)) ||
			!PgColbertReadExact(file, &offset, sizeof(offset)))
			goto malformed;

		if (strcmp(name, "colbert.proj.weight") == 0)
		{
			if (nDims != 2)
				return PgColbertSetError(ctx, errorMessage,
										 "GGUF ColBERT projection tensor must be 2-dimensional");
			found = true;
			dim0 = dims[0];
			dim1 = dims[1];
			projType = type;
			projOffset = offset;
		}
	}

	if (!found)
	{
		fclose(file);
		return PgColbertLoadProjectionSidecar(path, entry, ctx, errorMessage);
	}

	{
		long		pos = ftell(file);
		uint64_t	target;

		if (pos < 0)
			goto malformed;
		dataStart = PgColbertAlign((uint64_t) pos, alignment);
		target = dataStart + projOffset;
		if (fseek(file, 0, SEEK_SET) != 0 ||
			!PgColbertSkipBytes(file, target))
			goto malformed;
	}

	ok = PgColbertReadProjectionData(file, projType, dim0, dim1, entry, ctx,
									 errorMessage);
	goto cleanup;

malformed:
	PgColbertSetError(ctx, errorMessage,
					  "could not parse GGUF ColBERT projection metadata from \"%s\"",
					  path);

cleanup:
	fclose(file);
	return ok;
}

static bool
PgColbertUnsupportedGgufMetadata(const char *path, const char **reason)
{
	PgColbertLegacyBertMetadata metadata;

	*reason = NULL;
	if (!PgColbertReadLegacyBertMetadata(path, &metadata))
		return false;

	if (metadata.hasHuggingFaceTokenizer &&
		(!metadata.hasTokenizerModel || !metadata.hasTokenizerList))
	{
		if (metadata.hasPgColbertSchema)
			*reason = "the GGUF uses the pg_colbert schema with embedded Hugging Face tokenizer JSON, but this llama.cpp engine currently requires canonical tokenizer.ggml metadata and tensors";
		else
			*reason = "the GGUF embeds Hugging Face tokenizer JSON but does not provide canonical tokenizer.ggml metadata required by this llama.cpp engine";
		return true;
	}

	return false;
}

static bool
PgColbertLoadModel(const PgColbertModelSpec *spec, const char *path,
				   MemoryContext ctx, PgColbertCachedModel **entry,
				   bool *loadedFromCache, bool *cacheOwned,
				   char **errorMessage)
{
	struct llama_model_params modelParams;
	struct llama_context_params contextParams;
	struct llama_model_kv_override kvOverrides[16];
	PgColbertCachedModel *loaded;
	int			kvOverrideCount;
	const char *unsupportedReason;

	*entry = NULL;
	*loadedFromCache = false;
	*cacheOwned = false;

	if (!pg_colbert_llama_backend_initialized)
	{
		if (spec->nGpuLayers > 0)
		{
#ifdef PG_COLBERT_LLAMA_BACKEND_DIR
			ggml_backend_load_all_from_path(PG_COLBERT_LLAMA_BACKEND_DIR);
#endif
			ggml_backend_load_all();
			pg_colbert_llama_backend_gpu_loaded = true;
		}
		else
		{
#ifdef PG_COLBERT_LLAMA_BACKEND_DIR
			PgColbertLoadCpuBackendsFromPath(PG_COLBERT_LLAMA_BACKEND_DIR);
#else
			ggml_backend_load_all();
#endif
		}
		llama_backend_init();
		pg_colbert_llama_backend_initialized = true;
	}
	else if (spec->nGpuLayers > 0 && !pg_colbert_llama_backend_gpu_loaded)
	{
#ifdef PG_COLBERT_LLAMA_BACKEND_DIR
		ggml_backend_load_all_from_path(PG_COLBERT_LLAMA_BACKEND_DIR);
#endif
		ggml_backend_load_all();
		pg_colbert_llama_backend_gpu_loaded = true;
	}

	if (spec->cacheSize > 0)
	{
		loaded = PgColbertFindCachedModel(spec, path);
		if (loaded != NULL)
		{
			*entry = loaded;
			*loadedFromCache = true;
			*cacheOwned = true;
			return true;
		}
	}

	if (PgColbertUnsupportedGgufMetadata(path, &unsupportedReason))
		return PgColbertSetError(ctx, errorMessage,
								 "unsupported GGUF model \"%s\": %s",
								 path, unsupportedReason);

	loaded = (PgColbertCachedModel *) calloc(1, sizeof(PgColbertCachedModel));
	if (loaded == NULL)
		return PgColbertSetError(ctx, errorMessage,
								 "out of memory while allocating model cache entry");
	loaded->alias = strdup(spec->alias);
	loaded->path = strdup(path);
	if (loaded->alias == NULL || loaded->path == NULL)
	{
		PgColbertFreeCachedModel(loaded);
		return PgColbertSetError(ctx, errorMessage,
								 "out of memory while allocating model cache key");
	}

	modelParams = llama_model_default_params();
	modelParams.n_gpu_layers = spec->nGpuLayers;
	modelParams.use_mmap = llama_supports_mmap();
	kvOverrideCount = PgColbertBuildLegacyBertOverrides(spec, path,
														kvOverrides,
														lengthof(kvOverrides));
	if (kvOverrideCount > 0)
		modelParams.kv_overrides = kvOverrides;

	loaded->model = llama_model_load_from_file(path, modelParams);
	if (loaded->model == NULL)
	{
		PgColbertFreeCachedModel(loaded);
		return PgColbertSetError(ctx, errorMessage,
								 "could not load GGUF model \"%s\"", path);
	}

	loaded->nCtxTrain = llama_model_n_ctx_train(loaded->model);
	loaded->nEmbdModel = llama_model_n_embd_out(loaded->model);
	loaded->nEmbdOut = loaded->nEmbdModel;
	loaded->nLayer = llama_model_n_layer(loaded->model);
	loaded->nHead = llama_model_n_head(loaded->model);
	loaded->nCtx = spec->nCtx;
	loaded->nBatch = spec->nBatch;
	loaded->nGpuLayers = spec->nGpuLayers;

	if (!PgColbertLoadProjection(path, loaded, ctx, errorMessage))
	{
		PgColbertFreeCachedModel(loaded);
		return false;
	}

	contextParams = llama_context_default_params();
	contextParams.embeddings = true;
	contextParams.pooling_type = LLAMA_POOLING_TYPE_NONE;
	contextParams.attention_type = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
	contextParams.n_ctx = (uint32_t) spec->nCtx;
	contextParams.n_batch = (uint32_t) spec->nBatch;
	contextParams.n_ubatch = (uint32_t) spec->nBatch;
	contextParams.n_seq_max = 1;
	contextParams.n_threads = spec->threads;
	contextParams.n_threads_batch = spec->threads;

	loaded->ctx = llama_init_from_model(loaded->model, contextParams);
	if (loaded->ctx == NULL)
	{
		PgColbertFreeCachedModel(loaded);
		return PgColbertSetError(ctx, errorMessage,
								 "could not create llama context for \"%s\"", path);
	}
	llama_set_n_threads(loaded->ctx, spec->threads, spec->threads);

	loaded->lastUsed = ++pg_colbert_llama_cache_clock;
	if (spec->cacheSize > 0)
	{
		loaded->next = pg_colbert_llama_cache;
		pg_colbert_llama_cache = loaded;
		pg_colbert_llama_cache_count++;
		PgColbertEvictCache(spec->cacheSize, loaded);
		*cacheOwned = true;
	}

	*entry = loaded;
	return true;
}

static bool
PgColbertIsSpecialToken(const struct llama_vocab *vocab, llama_token token)
{
	llama_token special;

	if (token == LLAMA_TOKEN_NULL)
		return true;

	special = llama_vocab_bos(vocab);
	if (special != LLAMA_TOKEN_NULL && token == special)
		return true;
	special = llama_vocab_eos(vocab);
	if (special != LLAMA_TOKEN_NULL && token == special)
		return true;
	special = llama_vocab_sep(vocab);
	if (special != LLAMA_TOKEN_NULL && token == special)
		return true;
	special = llama_vocab_pad(vocab);
	if (special != LLAMA_TOKEN_NULL && token == special)
		return true;

	return false;
}

static bool
PgColbertEncodeBody(const PgColbertModelSpec *spec, const char *input,
					MemoryContext ctx, PgColbertEngineOutput *output,
					char **errorMessage)
{
	MemoryContext oldCtx;
	char	   *path = NULL;
	char	   *fullText = NULL;
	llama_token *tokens = NULL;
	float	   *projected = NULL;
	struct llama_batch batch;
	const struct llama_vocab *vocab;
	PgColbertCachedModel *entry = NULL;
	bool		loadedFromCache = false;
	bool		cacheOwned = false;
	int32		textLen;
	int32		requiredTokens;
	int32		nTokens;
	int32		retained;
	bool		ok = false;
	bool		batchInitialized = false;
	int32		rc;

	memset(&batch, 0, sizeof(batch));
	memset(output, 0, sizeof(*output));

	if (!PgColbertResolvePath(spec, ctx, &path, errorMessage))
		return false;
	if (!PgColbertLoadModel(spec, path, ctx, &entry, &loadedFromCache,
							&cacheOwned, errorMessage))
		return false;

	if (entry->nEmbdOut != spec->expectedDim)
	{
		if (!cacheOwned)
			PgColbertFreeCachedModel(entry);
		return PgColbertSetError(ctx, errorMessage,
								 "GGUF output dimension is %d, expected %d. The GGUF must include colbert.proj.weight or provide a %d-dimensional ColBERT GGUF.",
								 entry->nEmbdOut, spec->expectedDim,
								 spec->expectedDim);
	}

	textLen = strlen(spec->prefix) + strlen(input);
	fullText = (char *) malloc((size_t) textLen + 1);
	if (fullText == NULL)
		goto oom;
	memcpy(fullText, spec->prefix, strlen(spec->prefix));
	memcpy(fullText + strlen(spec->prefix), input, strlen(input) + 1);

	vocab = llama_model_get_vocab(entry->model);
	requiredTokens = llama_tokenize(vocab, fullText, textLen, NULL, 0, true,
									true);
	if (requiredTokens == INT32_MIN)
		goto tokenize_error;
	if (requiredTokens < 0)
		requiredTokens = -requiredTokens;
	if (requiredTokens <= 0)
	{
		PgColbertSetError(ctx, errorMessage,
						  "llama tokenization returned no tokens");
		goto cleanup;
	}

	tokens = (llama_token *) malloc(sizeof(llama_token) * (size_t) requiredTokens);
	if (tokens == NULL)
		goto oom;
	nTokens = llama_tokenize(vocab, fullText, textLen, tokens, requiredTokens,
							 true, true);
	if (nTokens < 0)
		goto tokenize_error;
	if (nTokens <= 0)
	{
		PgColbertSetError(ctx, errorMessage,
						  "llama tokenization returned no tokens");
		goto cleanup;
	}
	if (nTokens > spec->nCtx || nTokens > spec->nBatch)
	{
		PgColbertSetError(ctx, errorMessage,
						  "tokenized input has %d tokens, exceeding n_ctx=%d or n_batch=%d",
						  nTokens, spec->nCtx, spec->nBatch);
		goto cleanup;
	}

	retained = 0;
	for (int32 i = 0; i < nTokens; i++)
	{
		if (!PgColbertIsSpecialToken(vocab, tokens[i]))
			retained++;
	}
	if (retained > spec->maxVectors)
		retained = spec->maxVectors;
	if (retained <= 0)
	{
		PgColbertSetError(ctx, errorMessage,
						  "llama tokenization retained no non-special token vectors");
		goto cleanup;
	}

	batch = llama_batch_init(nTokens, 0, 1);
	batchInitialized = true;
	if (batch.token == NULL || batch.pos == NULL || batch.n_seq_id == NULL ||
		batch.seq_id == NULL || batch.logits == NULL)
		goto oom;

	batch.n_tokens = nTokens;
	for (int32 i = 0; i < nTokens; i++)
	{
		batch.token[i] = tokens[i];
		batch.pos[i] = i;
		batch.n_seq_id[i] = 1;
		batch.seq_id[i][0] = 0;
		batch.logits[i] = 1;
	}

	llama_memory_clear(llama_get_memory(entry->ctx), true);
	if (llama_model_has_encoder(entry->model) &&
		!llama_model_has_decoder(entry->model))
		rc = llama_encode(entry->ctx, batch);
	else
		rc = llama_decode(entry->ctx, batch);
	if (rc != 0)
	{
		PgColbertSetError(ctx, errorMessage,
						  "llama model evaluation failed with status %d", rc);
		goto cleanup;
	}

	oldCtx = MemoryContextSwitchTo(ctx);
	output->engine = PgColbertEngineName();
	output->path = pstrdup(entry->path);
	output->dim = entry->nEmbdOut;
	output->count = retained;
	output->normalized = true;
	output->loadedFromCache = loadedFromCache;
	output->tokenIds = (int32 *) palloc0(sizeof(int32) * (Size) retained);
	output->values = (float4 *) palloc0(sizeof(float4) * (Size) retained *
										(Size) entry->nEmbdOut);
	MemoryContextSwitchTo(oldCtx);

	if (entry->hasProjection)
	{
		projected = (float *) malloc(sizeof(float) * (size_t) entry->projectionOut);
		if (projected == NULL)
			goto oom;
	}

	retained = 0;
	for (int32 i = 0; i < nTokens && retained < output->count; i++)
	{
		const float *embedding;
		const float *vector;
		double		norm = 0.0;

		if (PgColbertIsSpecialToken(vocab, tokens[i]))
			continue;

		embedding = llama_get_embeddings_ith(entry->ctx, i);
		if (embedding == NULL)
		{
			PgColbertSetError(ctx, errorMessage,
							  "llama did not return token embedding %d", i);
			goto cleanup;
		}

		vector = embedding;
		if (entry->hasProjection)
		{
			for (int32 j = 0; j < entry->projectionOut; j++)
			{
				double		value = 0.0;

				for (int32 k = 0; k < entry->projectionIn; k++)
				{
					size_t		idx;

					if (entry->projectionTransposed)
						idx = (size_t) k * (size_t) entry->projectionOut +
							(size_t) j;
					else
						idx = (size_t) j * (size_t) entry->projectionIn +
							(size_t) k;
					value += (double) embedding[k] *
						(double) entry->projection[idx];
				}
				projected[j] = (float) value;
			}
			vector = projected;
		}

		for (int32 j = 0; j < output->dim; j++)
			norm += (double) vector[j] * (double) vector[j];
		if (norm <= 0.0 || !std::isfinite(norm))
		{
			PgColbertSetError(ctx, errorMessage,
							  "llama returned a zero or non-finite token embedding");
			goto cleanup;
		}

		norm = sqrt(norm);
		output->tokenIds[retained] = tokens[i];
		for (int32 j = 0; j < output->dim; j++)
			output->values[(Size) retained * (Size) output->dim + (Size) j] =
				(float4) (vector[j] / norm);
		retained++;
	}

	ok = true;
	goto cleanup;

oom:
	PgColbertSetError(ctx, errorMessage,
					  "out of memory while running llama embedding");
	goto cleanup;

tokenize_error:
	PgColbertSetError(ctx, errorMessage,
					  "llama tokenization failed");
	goto cleanup;

cleanup:
	if (batchInitialized)
		llama_batch_free(batch);
	free(projected);
	free(tokens);
	free(fullText);
	if (!cacheOwned)
		PgColbertFreeCachedModel(entry);
	return ok;
}

static bool
PgColbertModelInfoBody(const PgColbertModelSpec *spec, MemoryContext ctx,
					   PgColbertEngineModelInfo *info,
					   char **errorMessage)
{
	MemoryContext oldCtx;
	char	   *path = NULL;
	PgColbertCachedModel *entry = NULL;
	bool		loadedFromCache = false;
	bool		cacheOwned = false;

	memset(info, 0, sizeof(*info));
	if (!PgColbertResolvePath(spec, ctx, &path, errorMessage))
		return false;
	if (!PgColbertLoadModel(spec, path, ctx, &entry, &loadedFromCache,
							&cacheOwned, errorMessage))
		return false;

	oldCtx = MemoryContextSwitchTo(ctx);
	info->engine = PgColbertEngineName();
	info->path = pstrdup(entry->path);
	info->implemented = true;
	info->loadedFromCache = loadedFromCache;
	info->nCtxTrain = entry->nCtxTrain;
	info->nEmbdOut = entry->nEmbdOut;
	info->nLayer = entry->nLayer;
	info->nHead = entry->nHead;
	MemoryContextSwitchTo(oldCtx);

	if (!cacheOwned)
		PgColbertFreeCachedModel(entry);
	return true;
}

extern "C" bool
PgColbertEngineEncode(const PgColbertModelSpec *spec,
					  const char *input,
					  MemoryContext ctx,
					  PgColbertEngineOutput *output,
					  char **errorMessage)
{
	try
	{
		return PgColbertEncodeBody(spec, input, ctx, output, errorMessage);
	}
	catch (...)
	{
		return PgColbertSetError(ctx, errorMessage,
								 "unexpected C++ exception in llama engine");
	}
}

extern "C" bool
PgColbertEngineGetModelInfo(const PgColbertModelSpec *spec,
							MemoryContext ctx,
							PgColbertEngineModelInfo *info,
							char **errorMessage)
{
	try
	{
		return PgColbertModelInfoBody(spec, ctx, info, errorMessage);
	}
	catch (...)
	{
		return PgColbertSetError(ctx, errorMessage,
								 "unexpected C++ exception in llama engine");
	}
}
