extern "C" {
#include "postgres.h"

#include "lib/stringinfo.h"
#include "portability/instr_time.h"
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

typedef enum PgColbertProjectionModuleKind
{
	PG_COLBERT_PROJECTION_DENSE = 1,
	PG_COLBERT_PROJECTION_NORMALIZE = 2,
	PG_COLBERT_PROJECTION_TRUNCATE = 3
} PgColbertProjectionModuleKind;

typedef struct PgColbertProjectionModule
{
	PgColbertProjectionModuleKind kind;
	int32		inputDim;
	int32		outputDim;
	bool		transposed;
	float	   *weight;
	float	   *bias;
	int32		p;
} PgColbertProjectionModule;

typedef struct PgColbertCachedModel
{
	char	   *alias;
	char	   *path;
	int			nCtx;
	int			nBatch;
	int			nSeqMax;
	int			nGpuLayers;
	struct llama_model *model;
	struct llama_context *ctx;
	int32		nCtxTrain;
	int32		nEmbdModel;
	int32		nEmbdOut;
	int32		nLayer;
	int32		nHead;
	bool		hasProjection;
	PgColbertProjectionModule projectionModules[PG_COLBERT_LLAMA_MAX_PROJECTION_MODULES];
	int32		projectionModuleCount;
	int32		maxProjectionDim;
	uint64		lastUsed;
	struct PgColbertCachedModel *next;
} PgColbertCachedModel;

typedef struct PgColbertTensorInfo
{
	bool		found;
	uint32_t	type;
	uint32_t	nDims;
	uint64_t	dims[4];
	uint64_t	offset;
} PgColbertTensorInfo;

typedef struct PgColbertEncodePlan
{
	llama_token *tokens;
	int32	   *positionIds;
	int32	   *tokenTypeIds;
	int32	   *attentionMask;
	bool	   *retainMask;
	bool	   *outputMask;
	const char **retainReasons;
	int32		nTokens;
	int32		outputCount;
} PgColbertEncodePlan;

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
													  const char *path,
													  int nSeqMax);
static void PgColbertFreeCachedModel(PgColbertCachedModel *entry);
static void PgColbertEvictCache(int maxEntries, PgColbertCachedModel *protect);
static bool PgColbertLoadModel(const PgColbertModelSpec *spec,
							   const char *path,
							   int nSeqMax,
							   MemoryContext ctx,
							   PgColbertCachedModel **entry,
							   bool *loadedFromCache,
							   bool *cacheOwned,
							   char **errorMessage);
static int	PgColbertBuildLegacyBertOverrides(const PgColbertModelSpec *spec,
											   const char *path,
											   struct llama_model_kv_override *overrides,
											   int maxOverrides);
static bool PgColbertLoadProjection(const PgColbertModelSpec *spec,
									const char *path,
									PgColbertCachedModel *entry,
									MemoryContext ctx,
									char **errorMessage);
static bool PgColbertLoadProjectionSidecar(const PgColbertModelSpec *spec,
										   const char *path,
										   PgColbertCachedModel *entry,
										   MemoryContext ctx,
										   char **errorMessage);
static bool PgColbertValidateProjectionProfile(const PgColbertModelSpec *spec,
											  MemoryContext ctx,
											  char **errorMessage);
static bool PgColbertCheckTokenizerCapability(const PgColbertModelSpec *spec,
											  const char *path,
											  const char **reason);
static bool PgColbertCheckCanonicalBertMetadata(const char *path,
												MemoryContext ctx,
												char **errorMessage);
static bool PgColbertIsPunctuationToken(const struct llama_vocab *vocab,
										llama_token token);
static bool PgColbertProfileHasSkiplistToken(const PgColbertModelSpec *spec,
											 llama_token token);
static bool PgColbertShouldRetainToken(const struct llama_vocab *vocab,
									   const PgColbertModelSpec *spec,
									   llama_token token,
									   const char **reason);
static bool PgColbertEncodeBody(const PgColbertModelSpec *spec,
								const char *input,
								MemoryContext ctx,
								PgColbertEngineOutput *output,
								char **errorMessage);
static bool PgColbertEncodeBatchBody(const PgColbertModelSpec *spec,
									 const char *const *inputs,
									 int32 inputCount,
									 MemoryContext ctx,
									 PgColbertEngineOutput *outputs,
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
PgColbertFindCachedModel(const PgColbertModelSpec *spec, const char *path,
						 int nSeqMax)
{
	PgColbertCachedModel *entry;

	for (entry = pg_colbert_llama_cache; entry != NULL; entry = entry->next)
	{
		if (strcmp(entry->alias, spec->alias) == 0 &&
			strcmp(entry->path, path) == 0 &&
			entry->nCtx == spec->nCtx &&
			entry->nBatch == spec->nBatch &&
			entry->nSeqMax == nSeqMax &&
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
	for (int32 i = 0; i < entry->projectionModuleCount; i++)
	{
		free(entry->projectionModules[i].weight);
		free(entry->projectionModules[i].bias);
	}
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
								metadata.maxPositionEmbeddings);
	if (!metadata.hasEmbeddingLength && metadata.hasHiddenSize)
		PgColbertAddIntOverride(overrides, &count, maxOverrides,
								"bert.embedding_length",
								metadata.hiddenSize);
	if (metadata.hasIntermediateSize)
		PgColbertAddIntOverride(overrides, &count, maxOverrides,
								"bert.feed_forward_length",
								metadata.intermediateSize);
	if (!metadata.hasBlockCount && metadata.hasNumHiddenLayers)
		PgColbertAddIntOverride(overrides, &count, maxOverrides,
								"bert.block_count",
								metadata.numHiddenLayers);
	if (metadata.hasNumAttentionHeads)
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

static void
PgColbertAppendMissingKey(char *buffer, size_t bufferSize, const char *key)
{
	if (buffer[0] != '\0')
		strlcat(buffer, ", ", bufferSize);
	strlcat(buffer, key, bufferSize);
}

static bool
PgColbertCheckCanonicalBertMetadata(const char *path, MemoryContext ctx,
									char **errorMessage)
{
	PgColbertLegacyBertMetadata metadata;
	bool		legacyBert;
	char		missing[512];

	if (!PgColbertReadLegacyBertMetadata(path, &metadata))
		return true;

	legacyBert = metadata.hasHiddenSize || metadata.hasIntermediateSize ||
		metadata.hasNumHiddenLayers || metadata.hasNumAttentionHeads ||
		metadata.hasMaxPositionEmbeddings || metadata.hasLayerNormEps ||
		metadata.hasTypeVocabSize;
	if (!legacyBert)
		return true;

	missing[0] = '\0';
	if (!metadata.hasContextLength)
		PgColbertAppendMissingKey(missing, sizeof(missing),
								  "bert.context_length");
	if (!metadata.hasEmbeddingLength)
		PgColbertAppendMissingKey(missing, sizeof(missing),
								  "bert.embedding_length");
	if (!metadata.hasFeedForwardLength)
		PgColbertAppendMissingKey(missing, sizeof(missing),
								  "bert.feed_forward_length");
	if (!metadata.hasBlockCount)
		PgColbertAppendMissingKey(missing, sizeof(missing),
								  "bert.block_count");
	if (!metadata.hasAttentionHeadCount)
		PgColbertAppendMissingKey(missing, sizeof(missing),
								  "bert.attention.head_count");
	if (!metadata.hasAttentionLayerNormEps)
		PgColbertAppendMissingKey(missing, sizeof(missing),
								  "bert.attention.layer_norm_epsilon");
	if (!metadata.hasTokenizerTokenTypeCount)
		PgColbertAppendMissingKey(missing, sizeof(missing),
								  "tokenizer.ggml.token_type_count");
	if (missing[0] == '\0')
		return true;

	return PgColbertSetError(ctx, errorMessage,
							 "unsupported GGUF model \"%s\": missing llama.cpp BERT metadata keys: %s; re-export with canonical llama.cpp BERT hparams",
							 path, missing);
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
PgColbertReadFloatTensor(FILE *file, uint64_t dataStart,
						 const PgColbertTensorInfo *tensor,
						 float **values, MemoryContext ctx,
						 char **errorMessage)
{
	uint64_t	elements = 1;
	int			typeSize;
	uint64_t	target;
	float	   *out;

	typeSize = PgColbertGgmlTypeSize(tensor->type);
	if (typeSize == 0)
		return PgColbertSetError(ctx, errorMessage,
								 "GGUF ColBERT projection tensor uses unsupported type %u",
								 tensor->type);
	for (uint32_t i = 0; i < tensor->nDims; i++)
		elements *= tensor->dims[i];

	out = (float *) malloc(sizeof(float) * (size_t) elements);
	if (out == NULL)
		return PgColbertSetError(ctx, errorMessage,
								 "out of memory while loading ColBERT projection tensor");

	target = dataStart + tensor->offset;
	if (fseek(file, 0, SEEK_SET) != 0 || !PgColbertSkipBytes(file, target))
	{
		free(out);
		return PgColbertSetError(ctx, errorMessage,
								 "could not seek to ColBERT projection tensor data");
	}

	if (tensor->type == 0)
	{
		if (!PgColbertReadExact(file, out, sizeof(float) * (size_t) elements))
		{
			free(out);
			return PgColbertSetError(ctx, errorMessage,
									 "could not read ColBERT projection tensor data");
		}
	}
	else
	{
		for (uint64_t i = 0; i < elements; i++)
		{
			uint16_t	value;

			if (!PgColbertReadExact(file, &value, sizeof(value)))
			{
				free(out);
				return PgColbertSetError(ctx, errorMessage,
										 "could not read ColBERT projection tensor data");
			}
			out[i] = PgColbertHalfToFloat(value);
		}
	}

	*values = out;
	return true;
}

static bool
PgColbertConfigureDenseProjection(PgColbertProjectionModule *module,
								  const PgColbertTensorInfo *weight,
								  int32 currentDim,
								  MemoryContext ctx,
								  char **errorMessage)
{
	uint64_t	dim0;
	uint64_t	dim1;

	if (weight->nDims != 2)
		return PgColbertSetError(ctx, errorMessage,
								 "GGUF ColBERT dense projection tensor must be 2-dimensional");
	dim0 = weight->dims[0];
	dim1 = weight->dims[1];

	if (dim0 == (uint64_t) currentDim)
	{
		module->inputDim = currentDim;
		module->outputDim = (int32) dim1;
		module->transposed = false;
	}
	else if (dim1 == (uint64_t) currentDim)
	{
		module->inputDim = currentDim;
		module->outputDim = (int32) dim0;
		module->transposed = true;
	}
	else
		return PgColbertSetError(ctx, errorMessage,
								 "GGUF ColBERT dense projection shape is %llu x %llu, but current projection dimension is %d",
								 (unsigned long long) dim0,
								 (unsigned long long) dim1,
								 currentDim);

	return true;
}

static bool
PgColbertValidateBiasTensor(const PgColbertTensorInfo *bias, int32 outputDim,
							MemoryContext ctx, char **errorMessage)
{
	if (bias->nDims != 1)
		return PgColbertSetError(ctx, errorMessage,
								 "GGUF ColBERT dense bias tensor must be 1-dimensional");
	if (bias->dims[0] != (uint64_t) outputDim)
		return PgColbertSetError(ctx, errorMessage,
								 "GGUF ColBERT dense bias has dimension %llu, expected %d",
								 (unsigned long long) bias->dims[0],
								 outputDim);
	return true;
}

static void
PgColbertProjectionDefaultWeightName(char *name, size_t size, int denseOrdinal)
{
	if (denseOrdinal == 0)
		snprintf(name, size, "%s", "colbert.proj.weight");
	else
		snprintf(name, size, "%s.%d", "colbert.proj.weight", denseOrdinal);
}

static void
PgColbertProjectionDefaultBiasName(char *name, size_t size, int denseOrdinal)
{
	if (denseOrdinal == 0)
		snprintf(name, size, "%s", "colbert.proj.bias");
	else
		snprintf(name, size, "%s.%d", "colbert.proj.bias", denseOrdinal);
}

static bool
PgColbertLoadProjectionSidecar(const PgColbertModelSpec *spec,
							   const char *path, PgColbertCachedModel *entry,
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
	PgColbertTensorInfo weight;
	PgColbertProjectionModule module;

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

	memset(&weight, 0, sizeof(weight));
	weight.found = true;
	weight.type = projType;
	weight.nDims = 2;
	weight.dims[0] = dim0;
	weight.dims[1] = dim1;
	weight.offset = 0;
	memset(&module, 0, sizeof(module));
	module.kind = PG_COLBERT_PROJECTION_DENSE;
	if (PgColbertGgmlTypeSize(projType) == 0)
		ok = PgColbertSetError(ctx, errorMessage,
							   "ColBERT projection sidecar uses unsupported tensor type %u",
							   projType);
	else if (spec->profile.loaded &&
			 spec->profile.projectionModuleCount > 0 &&
			 spec->profile.projectionModules[0].hasBias)
		ok = PgColbertSetError(ctx, errorMessage,
							   "ColBERT projection sidecar cannot satisfy dense bias tensor \"%s\"; store the bias in GGUF metadata",
							   spec->profile.projectionModules[0].biasTensor[0] != '\0' ?
							   spec->profile.projectionModules[0].biasTensor :
							   "colbert.proj.bias");
	else
		ok = PgColbertConfigureDenseProjection(&module, &weight,
											   entry->nEmbdOut, ctx,
											   errorMessage);
	if (ok)
	{
		uint64_t	elements = dim0 * dim1;

		module.weight = (float *) malloc(sizeof(float) * (size_t) elements);
		if (module.weight == NULL)
			ok = PgColbertSetError(ctx, errorMessage,
								   "out of memory while loading ColBERT projection sidecar");
		else if (projType == 0)
		{
			if (!PgColbertReadExact(file, module.weight,
									sizeof(float) * (size_t) elements))
				ok = PgColbertSetError(ctx, errorMessage,
									   "could not read ColBERT projection sidecar data");
		}
		else
		{
			for (uint64_t i = 0; ok && i < elements; i++)
			{
				uint16_t	value;

				if (!PgColbertReadExact(file, &value, sizeof(value)))
					ok = PgColbertSetError(ctx, errorMessage,
										   "could not read ColBERT projection sidecar data");
				else
					module.weight[i] = PgColbertHalfToFloat(value);
			}
		}
	}
	if (ok)
	{
		entry->projectionModules[entry->projectionModuleCount++] = module;
		entry->hasProjection = true;
		entry->nEmbdOut = module.outputDim;
		entry->maxProjectionDim = Max(entry->maxProjectionDim, module.outputDim);
		if (spec->profile.loaded && spec->profile.projectionModuleCount > 1)
		{
			for (int32 i = 1; i < spec->profile.projectionModuleCount; i++)
			{
				const PgColbertProjectionModuleProfile *profileModule =
					&spec->profile.projectionModules[i];
				PgColbertProjectionModule extra;
				bool		isNormalize =
					strcmp(profileModule->type, "normalize") == 0;
				bool		isTruncate =
					strcmp(profileModule->type, "truncate") == 0;

				memset(&extra, 0, sizeof(extra));
				if (isNormalize)
				{
					extra.kind = PG_COLBERT_PROJECTION_NORMALIZE;
					extra.inputDim = entry->nEmbdOut;
					extra.outputDim = entry->nEmbdOut;
					extra.p = profileModule->p > 0 ? profileModule->p : 2;
				}
				else if (isTruncate)
				{
					extra.kind = PG_COLBERT_PROJECTION_TRUNCATE;
					extra.inputDim = entry->nEmbdOut;
					extra.outputDim = profileModule->outputDim;
					if (extra.outputDim <= 0 || extra.outputDim > extra.inputDim)
					{
						ok = PgColbertSetError(ctx, errorMessage,
											   "ColBERT truncate projection output_dim %d is invalid for input_dim %d",
											   extra.outputDim, extra.inputDim);
						break;
					}
				}
				else
				{
					ok = PgColbertSetError(ctx, errorMessage,
										   "ColBERT projection sidecar can only satisfy the first dense module; module %d requires GGUF tensors",
										   i);
					break;
				}
				entry->projectionModules[entry->projectionModuleCount++] = extra;
				entry->nEmbdOut = extra.outputDim;
				entry->maxProjectionDim = Max(entry->maxProjectionDim,
											  extra.outputDim);
			}
		}
	}
	if (!ok)
		free(module.weight);
	fclose(file);
	free(sidecarPath);
	return ok;
}

static bool
PgColbertLoadProjection(const PgColbertModelSpec *spec, const char *path,
						PgColbertCachedModel *entry,
						MemoryContext ctx, char **errorMessage)
{
	FILE	   *file;
	char		magic[4];
	uint32_t	version;
	uint64_t	tensorCount;
	uint64_t	kvCount;
	uint64_t	alignment = 32;
	uint64_t	dataStart;
	bool		ok = false;
	PgColbertTensorInfo weightInfos[PG_COLBERT_LLAMA_MAX_PROJECTION_MODULES];
	PgColbertTensorInfo biasInfos[PG_COLBERT_LLAMA_MAX_PROJECTION_MODULES];
	char		weightNames[PG_COLBERT_LLAMA_MAX_PROJECTION_MODULES][PG_COLBERT_LLAMA_MAX_PROFILE_STRING];
	char		biasNames[PG_COLBERT_LLAMA_MAX_PROJECTION_MODULES][PG_COLBERT_LLAMA_MAX_PROFILE_STRING];
	int32		moduleCount = 0;
	int32		denseOrdinal = 0;
	bool		requiresProjection = false;

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

	memset(weightInfos, 0, sizeof(weightInfos));
	memset(biasInfos, 0, sizeof(biasInfos));
	memset(weightNames, 0, sizeof(weightNames));
	memset(biasNames, 0, sizeof(biasNames));

	if (spec->profile.loaded && spec->profile.projectionModuleCount > 0)
		moduleCount = spec->profile.projectionModuleCount;
	else
		moduleCount = 1;

	for (int32 i = 0; i < moduleCount; i++)
	{
		const PgColbertProjectionModuleProfile *profileModule =
			(spec->profile.loaded && spec->profile.projectionModuleCount > 0) ?
			&spec->profile.projectionModules[i] : NULL;
		const char *type = profileModule != NULL ? profileModule->type : "dense";
		bool		isDense =
			strcmp(type, "dense") == 0 || strcmp(type, "linear") == 0;

		if (!isDense)
			continue;
		if (profileModule != NULL && profileModule->weightTensor[0] != '\0')
			snprintf(weightNames[i], sizeof(weightNames[i]), "%s",
					 profileModule->weightTensor);
		else
			PgColbertProjectionDefaultWeightName(weightNames[i],
												sizeof(weightNames[i]),
												denseOrdinal);
		if (profileModule != NULL && profileModule->biasTensor[0] != '\0')
			snprintf(biasNames[i], sizeof(biasNames[i]), "%s",
					 profileModule->biasTensor);
		else if (profileModule != NULL && profileModule->hasBias)
			PgColbertProjectionDefaultBiasName(biasNames[i],
											  sizeof(biasNames[i]),
											  denseOrdinal);
		requiresProjection = true;
		denseOrdinal++;
	}

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

		for (int32 j = 0; j < moduleCount; j++)
		{
			if (weightNames[j][0] != '\0' && strcmp(name, weightNames[j]) == 0)
			{
				weightInfos[j].found = true;
				weightInfos[j].type = type;
				weightInfos[j].nDims = nDims;
				memcpy(weightInfos[j].dims, dims, sizeof(dims));
				weightInfos[j].offset = offset;
			}
			if (biasNames[j][0] != '\0' && strcmp(name, biasNames[j]) == 0)
			{
				biasInfos[j].found = true;
				biasInfos[j].type = type;
				biasInfos[j].nDims = nDims;
				memcpy(biasInfos[j].dims, dims, sizeof(dims));
				biasInfos[j].offset = offset;
			}
		}
	}

	if (ftell(file) < 0)
		goto malformed;
	dataStart = PgColbertAlign((uint64_t) ftell(file), alignment);

	if (requiresProjection)
	{
		bool		missingWeight = false;

		for (int32 i = 0; i < moduleCount; i++)
		{
			if (weightNames[i][0] != '\0' && !weightInfos[i].found)
				missingWeight = true;
			if (biasNames[i][0] != '\0' && !biasInfos[i].found)
				return PgColbertSetError(ctx, errorMessage,
										 "GGUF ColBERT projection bias tensor \"%s\" was not found",
										 biasNames[i]);
		}
		if (missingWeight)
		{
			fclose(file);
			return PgColbertLoadProjectionSidecar(spec, path, entry, ctx,
												 errorMessage);
		}
	}
	else if (!spec->profile.loaded || spec->profile.projectionModuleCount == 0)
		goto no_projection;

	entry->maxProjectionDim = entry->nEmbdOut;
	for (int32 i = 0; i < moduleCount; i++)
	{
		const PgColbertProjectionModuleProfile *profileModule =
			(spec->profile.loaded && spec->profile.projectionModuleCount > 0) ?
			&spec->profile.projectionModules[i] : NULL;
		const char *type = profileModule != NULL ? profileModule->type : "dense";
		bool		isDense =
			strcmp(type, "dense") == 0 || strcmp(type, "linear") == 0;
		bool		isNormalize = strcmp(type, "normalize") == 0;
		bool		isTruncate = strcmp(type, "truncate") == 0;
		PgColbertProjectionModule module;

		memset(&module, 0, sizeof(module));
		if (isDense)
		{
			module.kind = PG_COLBERT_PROJECTION_DENSE;
			if (!PgColbertConfigureDenseProjection(&module, &weightInfos[i],
												   entry->nEmbdOut, ctx,
												   errorMessage))
				goto cleanup;
			if (!PgColbertReadFloatTensor(file, dataStart, &weightInfos[i],
										  &module.weight, ctx, errorMessage))
			{
				free(module.weight);
				free(module.bias);
				goto cleanup;
			}
			if (biasNames[i][0] != '\0')
			{
				if (!PgColbertValidateBiasTensor(&biasInfos[i],
												module.outputDim, ctx,
												errorMessage) ||
					!PgColbertReadFloatTensor(file, dataStart, &biasInfos[i],
											  &module.bias, ctx,
											  errorMessage))
				{
					free(module.weight);
					free(module.bias);
					goto cleanup;
				}
			}
		}
		else if (isNormalize)
		{
			module.kind = PG_COLBERT_PROJECTION_NORMALIZE;
			module.inputDim = entry->nEmbdOut;
			module.outputDim = entry->nEmbdOut;
			module.p = profileModule != NULL && profileModule->p > 0 ?
				profileModule->p : 2;
		}
		else if (isTruncate)
		{
			module.kind = PG_COLBERT_PROJECTION_TRUNCATE;
			module.inputDim = entry->nEmbdOut;
			module.outputDim = profileModule != NULL ?
				profileModule->outputDim : 0;
			if (module.outputDim <= 0 || module.outputDim > module.inputDim)
			{
				PgColbertSetError(ctx, errorMessage,
								  "ColBERT truncate projection output_dim %d is invalid for input_dim %d",
								  module.outputDim, module.inputDim);
				goto cleanup;
			}
		}
		else
		{
			PgColbertSetError(ctx, errorMessage,
							  "ColBERT projection profile uses unsupported module type \"%s\"",
							  type);
			goto cleanup;
		}

		entry->projectionModules[entry->projectionModuleCount++] = module;
		entry->hasProjection = true;
		entry->nEmbdOut = module.outputDim;
		entry->maxProjectionDim = Max(entry->maxProjectionDim,
									  module.outputDim);
	}

	if (entry->projectionModuleCount == 0)
		goto no_projection;
	ok = true;
	goto cleanup;

no_projection:
	ok = true;
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
PgColbertValidateProjectionProfile(const PgColbertModelSpec *spec,
								   MemoryContext ctx,
								   char **errorMessage)
{
	if (!spec->profile.loaded || spec->profile.projectionModuleCount == 0)
		return true;

	for (int32 i = 0; i < spec->profile.projectionModuleCount; i++)
	{
		const PgColbertProjectionModuleProfile *module =
			&spec->profile.projectionModules[i];
		bool		isLinear =
			strcmp(module->type, "linear") == 0 ||
			strcmp(module->type, "dense") == 0;
		bool		isNormalize = strcmp(module->type, "normalize") == 0;
		bool		isTruncate = strcmp(module->type, "truncate") == 0;

		if (isLinear)
		{
			if (module->activation[0] != '\0' &&
				strcmp(module->activation, "identity") != 0)
				return PgColbertSetError(ctx, errorMessage,
										 "ColBERT projection profile uses unsupported activation \"%s\"",
										 module->activation);
			continue;
		}
		if (isNormalize)
			continue;
		if (isTruncate)
		{
			if (module->outputDim > 0 && module->outputDim != spec->expectedDim)
				return PgColbertSetError(ctx, errorMessage,
										 "ColBERT truncate module output_dim %d does not match expected dimension %d",
										 module->outputDim, spec->expectedDim);
			continue;
		}
		return PgColbertSetError(ctx, errorMessage,
								 "ColBERT projection profile uses unsupported module type \"%s\"",
								 module->type);
	}
	return true;
}

static bool
PgColbertCheckTokenizerCapability(const PgColbertModelSpec *spec,
								  const char *path, const char **reason)
{
	PgColbertLegacyBertMetadata metadata;

	*reason = NULL;
	if (!PgColbertReadLegacyBertMetadata(path, &metadata))
		return false;

	if (metadata.hasHuggingFaceTokenizer &&
		(!metadata.hasTokenizerModel || !metadata.hasTokenizerList))
	{
		if (spec->profile.loaded &&
			strcmp(spec->profile.tokenizerSource, "hf_json") == 0)
			*reason = "unsupported_tokenizer: prepare this GGUF with tokenizer.ggml metadata using colbert-gguf-converter --target-runtime llama_cpp";
		else if (metadata.hasPgColbertSchema)
			*reason = "unsupported_tokenizer: the GGUF uses the pg_colbert schema with embedded Hugging Face tokenizer JSON, but this llama.cpp engine currently requires canonical tokenizer.ggml metadata and tensors";
		else
			*reason = "unsupported_tokenizer: the GGUF embeds Hugging Face tokenizer JSON but does not provide canonical tokenizer.ggml metadata required by this llama.cpp engine";
		return true;
	}

	return false;
}

static bool
PgColbertNormalizeVector(float *values, int32 dim)
{
	double		norm = 0.0;

	for (int32 i = 0; i < dim; i++)
		norm += (double) values[i] * (double) values[i];
	if (norm <= 0.0 || !std::isfinite(norm))
		return false;
	norm = sqrt(norm);
	for (int32 i = 0; i < dim; i++)
		values[i] = (float) ((double) values[i] / norm);
	return true;
}

static int64
PgColbertElapsedUs(instr_time start)
{
	instr_time	end;

	INSTR_TIME_SET_CURRENT(end);
	INSTR_TIME_SUBTRACT(end, start);
	return (int64) INSTR_TIME_GET_MICROSEC(end);
}

static void
PgColbertSetTotalTiming(PgColbertEngineTiming *timing,
						instr_time totalStart)
{
	timing->totalUs = PgColbertElapsedUs(totalStart);
}

static void
PgColbertLogTiming(const PgColbertModelSpec *spec,
				   const PgColbertEngineTiming *timing,
				   const char *mode)
{
	if (!spec->logTiming)
		return;

	ereport(LOG,
			(errmsg("pg_colbert_llama %s timing: inputs=%lld tokens=%lld output_vectors=%lld total_us=%lld tokenization_us=%lld llama_us=%lld output_us=%lld debug_us=%lld projection_us=%lld",
					mode,
					(long long) timing->inputs,
					(long long) timing->tokens,
					(long long) timing->outputVectors,
					(long long) timing->totalUs,
					(long long) timing->tokenizationUs,
					(long long) timing->llamaUs,
					(long long) timing->outputUs,
					(long long) timing->debugUs,
					(long long) timing->projectionUs)));
}

static bool
PgColbertApplyProjectionChain(const PgColbertCachedModel *entry,
							  const float *embedding,
							  float *workspaceA,
							  float *workspaceB,
							  const float **vector,
							  int32 *dim,
							  MemoryContext ctx,
							  char **errorMessage)
{
	const float *current = embedding;
	float	   *currentOwned = NULL;
	int32		currentDim = entry->nEmbdModel;
	bool		useA = true;

	for (int32 i = 0; i < entry->projectionModuleCount; i++)
	{
		const PgColbertProjectionModule *module = &entry->projectionModules[i];
		float	   *out = useA ? workspaceA : workspaceB;

		if (module->inputDim != currentDim)
			return PgColbertSetError(ctx, errorMessage,
									 "ColBERT projection module %d expected input_dim %d but received %d",
									 i, module->inputDim, currentDim);

		if (module->kind == PG_COLBERT_PROJECTION_DENSE)
		{
			for (int32 j = 0; j < module->outputDim; j++)
			{
				double		value = module->bias != NULL ?
					(double) module->bias[j] : 0.0;

				for (int32 k = 0; k < module->inputDim; k++)
				{
					size_t		idx;

					if (module->transposed)
						idx = (size_t) k * (size_t) module->outputDim +
							(size_t) j;
					else
						idx = (size_t) j * (size_t) module->inputDim +
							(size_t) k;
					value += (double) current[k] *
						(double) module->weight[idx];
				}
				out[j] = (float) value;
			}
			current = out;
			currentOwned = out;
			currentDim = module->outputDim;
			useA = !useA;
			continue;
		}

		if (module->kind == PG_COLBERT_PROJECTION_NORMALIZE)
		{
			if (currentOwned == NULL)
			{
				memcpy(out, current, sizeof(float) * (size_t) currentDim);
				current = out;
				currentOwned = out;
				useA = !useA;
			}
			if (!PgColbertNormalizeVector(currentOwned, currentDim))
				return PgColbertSetError(ctx, errorMessage,
										 "ColBERT projection normalization saw a zero or non-finite vector");
			continue;
		}

		if (module->kind == PG_COLBERT_PROJECTION_TRUNCATE)
		{
			if (module->outputDim > currentDim)
				return PgColbertSetError(ctx, errorMessage,
										 "ColBERT truncate projection output_dim %d exceeds input_dim %d",
										 module->outputDim, currentDim);
			if (currentOwned == NULL)
			{
				memcpy(out, current, sizeof(float) * (size_t) module->outputDim);
				current = out;
				currentOwned = out;
				useA = !useA;
			}
			currentDim = module->outputDim;
			continue;
		}

		return PgColbertSetError(ctx, errorMessage,
								 "ColBERT projection module %d has unknown kind %d",
								 i, (int) module->kind);
	}

	*vector = current;
	*dim = currentDim;
	return true;
}

static bool
PgColbertLoadModel(const PgColbertModelSpec *spec, const char *path,
				   int nSeqMax,
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
		loaded = PgColbertFindCachedModel(spec, path, nSeqMax);
		if (loaded != NULL)
		{
			*entry = loaded;
			*loadedFromCache = true;
			*cacheOwned = true;
			return true;
		}
	}

	if (PgColbertCheckTokenizerCapability(spec, path, &unsupportedReason))
		return PgColbertSetError(ctx, errorMessage,
								 "unsupported GGUF model \"%s\": %s",
								 path, unsupportedReason);
	if (!PgColbertCheckCanonicalBertMetadata(path, ctx, errorMessage))
		return false;
	if (!PgColbertValidateProjectionProfile(spec, ctx, errorMessage))
		return false;

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
	loaded->nSeqMax = nSeqMax;
	loaded->nGpuLayers = spec->nGpuLayers;

	if (!PgColbertLoadProjection(spec, path, loaded, ctx, errorMessage))
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
	contextParams.n_seq_max = (uint32_t) nSeqMax;
	contextParams.n_threads = spec->threads;
	contextParams.n_threads_batch = spec->threads;
	if (nSeqMax > 1)
	{
		contextParams.swa_full = true;
		contextParams.kv_unified = false;
	}

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
PgColbertIsPunctuationToken(const struct llama_vocab *vocab, llama_token token)
{
	const char *text = llama_vocab_get_text(vocab, token);
	const char *piece = text;

	if (piece == NULL || piece[0] == '\0')
		return false;

	/* llama.cpp's BERT WPM tokenizer prefixes word-start tokens with U+2581. */
	if ((unsigned char) piece[0] == 0xe2 &&
		(unsigned char) piece[1] == 0x96 &&
		(unsigned char) piece[2] == 0x81)
		piece += 3;

	return piece[0] != '\0' &&
		piece[1] == '\0' &&
			strchr("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~", piece[0]) != NULL;
}

static bool
PgColbertProfileHasSkiplistToken(const PgColbertModelSpec *spec,
								 llama_token token)
{
	for (int32 i = 0; i < spec->profile.skiplistTokenCount; i++)
	{
		if (spec->profile.skiplistTokenIds[i] == (int32) token)
			return true;
	}
	return false;
}

static bool
PgColbertShouldRetainToken(const struct llama_vocab *vocab,
						   const PgColbertModelSpec *spec,
						   llama_token token,
						   const char **reason)
{
	llama_token special;

	if (token == LLAMA_TOKEN_NULL)
	{
		*reason = "dropped_null";
		return false;
	}

	/*
	 * PyLate keeps [CLS], the ColBERT prefix token, [SEP], and query
	 * expansion [MASK] vectors for query scoring.  Documents keep attended
	 * non-padding tokens; punctuation skiplist handling is a scorer concern.
	 */
	if (spec->role == PG_COLBERT_ROLE_QUERY)
	{
		*reason = "retained_query";
		return true;
	}

	special = llama_vocab_pad(vocab);
	if (special != LLAMA_TOKEN_NULL && token == special)
	{
		*reason = "dropped_pad";
		return false;
	}
	if (spec->profile.specialTokens.padTokenId >= 0 &&
		token == (llama_token) spec->profile.specialTokens.padTokenId)
	{
		*reason = "dropped_pad";
		return false;
	}
	if (PgColbertProfileHasSkiplistToken(spec, token))
	{
		*reason = "dropped_skiplist_token";
		return false;
	}
	if (PgColbertIsPunctuationToken(vocab, token))
	{
		*reason = "dropped_punctuation";
		return false;
	}
	*reason = "retained_document";
	return true;
}

static void
PgColbertFreeEncodePlan(PgColbertEncodePlan *plan)
{
	free(plan->tokens);
	free(plan->positionIds);
	free(plan->tokenTypeIds);
	free(plan->attentionMask);
	free(plan->retainMask);
	free(plan->outputMask);
	free(plan->retainReasons);
	memset(plan, 0, sizeof(*plan));
}

static bool
PgColbertBuildEncodePlan(const struct llama_vocab *vocab,
						 const PgColbertModelSpec *spec,
						 const char *fullText,
						 int32 textLen,
						 PgColbertEncodePlan *plan,
						 MemoryContext ctx,
						 char **errorMessage)
{
	int32		requiredTokens;
	int32		tokenCapacity;
	int32		maxPlanTokens;
	int32		queryPadTo;
	int32		outputCount = 0;

	memset(plan, 0, sizeof(*plan));
	requiredTokens = llama_tokenize(vocab, fullText, textLen, NULL, 0, true,
									true);
	if (requiredTokens == INT32_MIN)
		return PgColbertSetError(ctx, errorMessage,
								 "llama tokenization failed");
	if (requiredTokens < 0)
		requiredTokens = -requiredTokens;
	if (requiredTokens <= 0)
		return PgColbertSetError(ctx, errorMessage,
								 "llama tokenization returned no tokens");

	maxPlanTokens = spec->role == PG_COLBERT_ROLE_QUERY ?
		spec->profile.queryMaxLength : spec->profile.documentMaxLength;
	if (maxPlanTokens <= 0)
		maxPlanTokens = spec->role == PG_COLBERT_ROLE_QUERY ?
			spec->queryLength : spec->maxVectors;
	queryPadTo = spec->role == PG_COLBERT_ROLE_QUERY ?
		spec->profile.queryPadTo : -1;
	if (queryPadTo <= 0)
		queryPadTo = spec->queryLength;

	tokenCapacity = requiredTokens;
	if (spec->role == PG_COLBERT_ROLE_QUERY && queryPadTo > tokenCapacity)
		tokenCapacity = queryPadTo;

	plan->tokens = (llama_token *) malloc(sizeof(llama_token) *
										  (size_t) tokenCapacity);
	if (plan->tokens == NULL)
		goto oom;
	plan->nTokens = llama_tokenize(vocab, fullText, textLen, plan->tokens,
								   tokenCapacity, true, true);
	if (plan->nTokens < 0)
	{
		PgColbertFreeEncodePlan(plan);
		return PgColbertSetError(ctx, errorMessage,
								 "llama tokenization failed");
	}
	if (plan->nTokens <= 0)
	{
		PgColbertFreeEncodePlan(plan);
		return PgColbertSetError(ctx, errorMessage,
								 "llama tokenization returned no tokens");
	}
	if (maxPlanTokens > 0 && plan->nTokens > maxPlanTokens)
		plan->nTokens = maxPlanTokens;
	if (spec->role == PG_COLBERT_ROLE_QUERY && plan->nTokens < queryPadTo)
	{
		llama_token mask = spec->profile.queryPadTokenId >= 0 ?
			(llama_token) spec->profile.queryPadTokenId : llama_vocab_mask(vocab);

		if (mask == LLAMA_TOKEN_NULL)
		{
			PgColbertFreeEncodePlan(plan);
			return PgColbertSetError(ctx, errorMessage,
									 "llama vocabulary does not define a mask token required for query expansion");
		}
		while (plan->nTokens < queryPadTo)
			plan->tokens[plan->nTokens++] = mask;
	}

	plan->positionIds = (int32 *) malloc(sizeof(int32) * (size_t) plan->nTokens);
	plan->tokenTypeIds = (int32 *) malloc(sizeof(int32) * (size_t) plan->nTokens);
	plan->attentionMask = (int32 *) malloc(sizeof(int32) * (size_t) plan->nTokens);
	plan->retainMask = (bool *) malloc(sizeof(bool) * (size_t) plan->nTokens);
	plan->outputMask = (bool *) malloc(sizeof(bool) * (size_t) plan->nTokens);
	plan->retainReasons =
		(const char **) malloc(sizeof(const char *) * (size_t) plan->nTokens);
	if (plan->positionIds == NULL || plan->tokenTypeIds == NULL ||
		plan->attentionMask == NULL || plan->retainMask == NULL ||
		plan->outputMask == NULL || plan->retainReasons == NULL)
		goto oom;

	for (int32 i = 0; i < plan->nTokens; i++)
	{
		const char *reason = NULL;
		bool		retain =
			PgColbertShouldRetainToken(vocab, spec, plan->tokens[i], &reason);

		plan->positionIds[i] = i;
		plan->tokenTypeIds[i] = spec->role == PG_COLBERT_ROLE_QUERY ?
			spec->profile.queryTokenTypeId : spec->profile.documentTokenTypeId;
		plan->attentionMask[i] = 1;
		plan->retainMask[i] = retain;
		plan->outputMask[i] = retain && outputCount < spec->maxVectors;
		plan->retainReasons[i] = reason != NULL ? reason : "";
		if (plan->outputMask[i])
			outputCount++;
	}
	plan->outputCount = outputCount;
	if (plan->outputCount <= 0)
	{
		PgColbertFreeEncodePlan(plan);
		return PgColbertSetError(ctx, errorMessage,
								 "llama tokenization retained no non-special token vectors");
	}
	return true;

oom:
	PgColbertFreeEncodePlan(plan);
	return PgColbertSetError(ctx, errorMessage,
							 "out of memory while building ColBERT encode plan");
}

static bool
PgColbertEncodeBody(const PgColbertModelSpec *spec, const char *input,
					MemoryContext ctx, PgColbertEngineOutput *output,
					char **errorMessage)
{
	MemoryContext oldCtx;
	char	   *path = NULL;
	char	   *fullText = NULL;
	PgColbertEncodePlan plan;
	float	   *projectionWorkspaceA = NULL;
	float	   *projectionWorkspaceB = NULL;
	const float *embeddings = NULL;
	struct llama_batch batch;
	const struct llama_vocab *vocab;
	PgColbertCachedModel *entry = NULL;
	bool		loadedFromCache = false;
	bool		cacheOwned = false;
	int32		textLen;
	int32		retained;
	bool		ok = false;
	bool		batchInitialized = false;
	int32		rc;
	const char *attentionMaskStatus = "ok";
	instr_time	totalStart;
	instr_time	phaseStart;
	PgColbertEngineTiming timing;

	memset(&batch, 0, sizeof(batch));
	memset(&plan, 0, sizeof(plan));
	memset(output, 0, sizeof(*output));
	memset(&timing, 0, sizeof(timing));
	INSTR_TIME_SET_CURRENT(totalStart);

	if (!PgColbertResolvePath(spec, ctx, &path, errorMessage))
		return false;
	if (!PgColbertLoadModel(spec, path, Max(spec->batchSequences, 1),
							ctx, &entry, &loadedFromCache,
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
	INSTR_TIME_SET_CURRENT(phaseStart);
	fullText = (char *) malloc((size_t) textLen + 1);
	if (fullText == NULL)
		goto oom;
	memcpy(fullText, spec->prefix, strlen(spec->prefix));
	memcpy(fullText + strlen(spec->prefix), input, strlen(input) + 1);

	vocab = llama_model_get_vocab(entry->model);
	if (!PgColbertBuildEncodePlan(vocab, spec, fullText, textLen, &plan,
								  ctx, errorMessage))
		goto cleanup;
	timing.tokenizationUs += PgColbertElapsedUs(phaseStart);
	if (spec->role == PG_COLBERT_ROLE_QUERY &&
		spec->profile.loaded &&
		spec->profile.strictPylateProfile &&
		!spec->profile.queryAttendToExpansionTokens)
		attentionMaskStatus = "approximated";

	if (plan.nTokens > spec->nCtx || plan.nTokens > spec->nBatch)
	{
		PgColbertSetError(ctx, errorMessage,
						  "tokenized input has %d tokens, exceeding n_ctx=%d or n_batch=%d",
						  plan.nTokens, spec->nCtx, spec->nBatch);
		goto cleanup;
	}

	retained = plan.outputCount;

	batch = llama_batch_init(plan.nTokens, 0, 1);
	batchInitialized = true;
	if (batch.token == NULL || batch.pos == NULL || batch.n_seq_id == NULL ||
		batch.seq_id == NULL || batch.logits == NULL)
		goto oom;

	batch.n_tokens = plan.nTokens;
	for (int32 i = 0; i < plan.nTokens; i++)
	{
		batch.token[i] = plan.tokens[i];
		batch.pos[i] = plan.positionIds[i];
		batch.n_seq_id[i] = 1;
		batch.seq_id[i][0] = 0;
		batch.logits[i] = 1;
	}

	if (llama_model_has_encoder(entry->model) &&
		!llama_model_has_decoder(entry->model))
	{
		INSTR_TIME_SET_CURRENT(phaseStart);
		rc = llama_encode(entry->ctx, batch);
		timing.llamaUs += PgColbertElapsedUs(phaseStart);
	}
	else
	{
		llama_memory_clear(llama_get_memory(entry->ctx), true);
		INSTR_TIME_SET_CURRENT(phaseStart);
		rc = llama_decode(entry->ctx, batch);
		timing.llamaUs += PgColbertElapsedUs(phaseStart);
	}
	if (rc != 0)
	{
		PgColbertSetError(ctx, errorMessage,
						  "llama model evaluation failed with status %d", rc);
		goto cleanup;
	}

	INSTR_TIME_SET_CURRENT(phaseStart);
	oldCtx = MemoryContextSwitchTo(ctx);
	output->engine = PgColbertEngineName();
	output->path = pstrdup(entry->path);
	output->input = pstrdup(input);
	output->prefix = pstrdup(spec->prefix);
	output->profileSource = pstrdup(spec->profile.sourceName);
	output->attentionMaskStatus = pstrdup(attentionMaskStatus);
	output->knownLimitations = pstrdup(spec->profile.knownLimitations);
	output->dim = entry->nEmbdOut;
	output->count = retained;
	output->planTokenCount = plan.nTokens;
	output->normalized = true;
	output->loadedFromCache = loadedFromCache;
	output->tokenIds = (int32 *) palloc0(sizeof(int32) * (Size) retained);
	output->values = (float4 *) palloc0(sizeof(float4) * (Size) retained *
										(Size) entry->nEmbdOut);
	MemoryContextSwitchTo(oldCtx);
	timing.outputUs += PgColbertElapsedUs(phaseStart);

	if (spec->debugTokens)
	{
		INSTR_TIME_SET_CURRENT(phaseStart);
		oldCtx = MemoryContextSwitchTo(ctx);
		output->tokenDebug =
			(PgColbertTokenDebug *) palloc0(sizeof(PgColbertTokenDebug) *
											(Size) plan.nTokens);
		for (int32 i = 0; i < plan.nTokens; i++)
		{
			const char *piece = llama_vocab_get_text(vocab, plan.tokens[i]);

			output->tokenDebug[i].index = i;
			output->tokenDebug[i].id = (int32) plan.tokens[i];
			output->tokenDebug[i].piece = piece != NULL ? pstrdup(piece) : NULL;
			output->tokenDebug[i].positionId = plan.positionIds[i];
			output->tokenDebug[i].tokenTypeId = plan.tokenTypeIds[i];
			output->tokenDebug[i].attentionMask = plan.attentionMask[i];
			output->tokenDebug[i].outputEnabled = plan.outputMask[i];
			output->tokenDebug[i].retained = plan.retainMask[i];
			output->tokenDebug[i].retainReason =
				pstrdup(plan.retainReasons[i]);
		}
		MemoryContextSwitchTo(oldCtx);
		timing.debugUs += PgColbertElapsedUs(phaseStart);
	}

	if (entry->hasProjection)
	{
		int32		workspaceDim = Max(entry->maxProjectionDim, entry->nEmbdModel);

		projectionWorkspaceA =
			(float *) malloc(sizeof(float) * (size_t) workspaceDim);
		projectionWorkspaceB =
			(float *) malloc(sizeof(float) * (size_t) workspaceDim);
		if (projectionWorkspaceA == NULL || projectionWorkspaceB == NULL)
			goto oom;
	}

	embeddings = llama_get_embeddings(entry->ctx);
	if (embeddings == NULL)
	{
		PgColbertSetError(ctx, errorMessage,
						  "llama did not return token embeddings");
		goto cleanup;
	}

	retained = 0;
	INSTR_TIME_SET_CURRENT(phaseStart);
	for (int32 i = 0; i < plan.nTokens && retained < output->count; i++)
	{
		const float *vector;
		int32		vectorDim;
		double		norm = 0.0;

		if (!plan.outputMask[i])
			continue;

		vector = embeddings + ((Size) i * (Size) entry->nEmbdModel);
		vectorDim = entry->nEmbdModel;
		if (entry->hasProjection)
		{
			if (!PgColbertApplyProjectionChain(entry, vector,
											   projectionWorkspaceA,
											   projectionWorkspaceB,
											   &vector, &vectorDim,
											   ctx, errorMessage))
				goto cleanup;
			if (vectorDim != output->dim)
			{
				PgColbertSetError(ctx, errorMessage,
								  "ColBERT projection output dimension is %d, expected %d",
								  vectorDim, output->dim);
				goto cleanup;
			}
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
		output->tokenIds[retained] = plan.tokens[i];
		for (int32 j = 0; j < output->dim; j++)
			output->values[(Size) retained * (Size) output->dim + (Size) j] =
				(float4) (vector[j] / norm);
		retained++;
	}
	timing.projectionUs += PgColbertElapsedUs(phaseStart);
	timing.inputs = 1;
	timing.tokens = plan.nTokens;
	timing.outputVectors = output->count;
	PgColbertSetTotalTiming(&timing, totalStart);
	output->timing = timing;
	PgColbertLogTiming(spec, &timing, "single");

	ok = true;
	goto cleanup;

oom:
	PgColbertSetError(ctx, errorMessage,
					  "out of memory while running llama embedding");
	goto cleanup;

cleanup:
	if (batchInitialized)
		llama_batch_free(batch);
	free(projectionWorkspaceA);
	free(projectionWorkspaceB);
	PgColbertFreeEncodePlan(&plan);
	free(fullText);
	if (!cacheOwned)
		PgColbertFreeCachedModel(entry);
	return ok;
}

static bool
PgColbertEncodeBatchBody(const PgColbertModelSpec *spec,
						 const char *const *inputs,
						 int32 inputCount,
						 MemoryContext ctx,
						 PgColbertEngineOutput *outputs,
						 char **errorMessage)
{
	MemoryContext oldCtx;
	char	   *path = NULL;
	char	  **fullTexts = NULL;
	PgColbertEncodePlan *plans = NULL;
	float	   *projectionWorkspaceA = NULL;
	float	   *projectionWorkspaceB = NULL;
	const float *embeddings = NULL;
	struct llama_batch batch;
	const struct llama_vocab *vocab;
	PgColbertCachedModel *entry = NULL;
	bool		loadedFromCache = false;
	bool		cacheOwned = false;
	bool		ok = false;
	bool		batchInitialized = false;
	int			batchSequences = Max(spec->batchSequences, 1);
	int			batchSequenceCapacity;
	int32		maxBatchTokens = 0;
	size_t		prefixLen = strlen(spec->prefix);
	const char *attentionMaskStatus = "ok";
	instr_time	totalStart;
	instr_time	phaseStart;
	PgColbertEngineTiming timing;

	memset(&batch, 0, sizeof(batch));
	memset(&timing, 0, sizeof(timing));
	timing.inputs = inputCount;
	INSTR_TIME_SET_CURRENT(totalStart);

	if (inputCount <= 0)
		return true;

	if (!PgColbertResolvePath(spec, ctx, &path, errorMessage))
		return false;
	if (!PgColbertLoadModel(spec, path, batchSequences, ctx, &entry,
							&loadedFromCache, &cacheOwned, errorMessage))
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

	fullTexts = (char **) calloc((size_t) inputCount, sizeof(char *));
	plans = (PgColbertEncodePlan *) calloc((size_t) inputCount,
										   sizeof(PgColbertEncodePlan));
	if (fullTexts == NULL || plans == NULL)
		goto oom;

	vocab = llama_model_get_vocab(entry->model);
	INSTR_TIME_SET_CURRENT(phaseStart);
	for (int32 i = 0; i < inputCount; i++)
	{
		int32		textLen = prefixLen + strlen(inputs[i]);

		fullTexts[i] = (char *) malloc((size_t) textLen + 1);
		if (fullTexts[i] == NULL)
			goto oom;
		memcpy(fullTexts[i], spec->prefix, prefixLen);
		memcpy(fullTexts[i] + prefixLen, inputs[i],
			   strlen(inputs[i]) + 1);

		if (!PgColbertBuildEncodePlan(vocab, spec, fullTexts[i], textLen,
									  &plans[i], ctx, errorMessage))
			goto cleanup;
		timing.tokens += plans[i].nTokens;
		timing.outputVectors += plans[i].outputCount;
		if (plans[i].nTokens > spec->nCtx || plans[i].nTokens > spec->nBatch)
		{
			PgColbertSetError(ctx, errorMessage,
							  "tokenized input %d has %d tokens, exceeding n_ctx=%d or n_batch=%d",
							  i + 1, plans[i].nTokens, spec->nCtx,
							  spec->nBatch);
			goto cleanup;
		}
	}
	timing.tokenizationUs += PgColbertElapsedUs(phaseStart);

	for (int32 start = 0; start < inputCount;)
	{
		int32		end = start;
		int32		totalTokens = 0;

		while (end < inputCount && end - start < batchSequences)
		{
			if (totalTokens > 0 &&
				totalTokens + plans[end].nTokens > spec->nBatch)
				break;
			totalTokens += plans[end].nTokens;
			end++;
		}
		if (end == start)
		{
			PgColbertSetError(ctx, errorMessage,
							  "tokenized input %d has %d tokens, exceeding n_batch=%d",
							  start + 1, plans[start].nTokens, spec->nBatch);
			goto cleanup;
		}
		maxBatchTokens = Max(maxBatchTokens, totalTokens);
		start = end;
	}

	batchSequenceCapacity = Min(batchSequences, inputCount);
	batch = llama_batch_init(maxBatchTokens, 0, batchSequenceCapacity);
	batchInitialized = true;
	if (batch.token == NULL || batch.pos == NULL ||
		batch.n_seq_id == NULL || batch.seq_id == NULL ||
		batch.logits == NULL)
		goto oom;

	if (spec->role == PG_COLBERT_ROLE_QUERY &&
		spec->profile.loaded &&
		spec->profile.strictPylateProfile &&
		!spec->profile.queryAttendToExpansionTokens)
		attentionMaskStatus = "approximated";

	if (entry->hasProjection)
	{
		int32		workspaceDim = Max(entry->maxProjectionDim, entry->nEmbdModel);

		projectionWorkspaceA =
			(float *) malloc(sizeof(float) * (size_t) workspaceDim);
		projectionWorkspaceB =
			(float *) malloc(sizeof(float) * (size_t) workspaceDim);
		if (projectionWorkspaceA == NULL || projectionWorkspaceB == NULL)
			goto oom;
	}

	for (int32 start = 0; start < inputCount;)
	{
		int32		end = start;
		int32		totalTokens = 0;
		int32		totalOutputs = 0;
		int32		globalToken = 0;
		int32		rc;
		int32		outputOrdinal = 0;
		int32		readToken = 0;

		while (end < inputCount && end - start < batchSequences)
		{
			if (totalTokens > 0 &&
				totalTokens + plans[end].nTokens > spec->nBatch)
				break;
			totalTokens += plans[end].nTokens;
			totalOutputs += plans[end].outputCount;
			end++;
		}
		if (end == start)
		{
			PgColbertSetError(ctx, errorMessage,
							  "tokenized input %d has %d tokens, exceeding n_batch=%d",
							  start + 1, plans[start].nTokens, spec->nBatch);
			goto cleanup;
		}

		batch.n_tokens = totalTokens;
		for (int32 inputIndex = start; inputIndex < end; inputIndex++)
		{
			PgColbertEncodePlan *plan = &plans[inputIndex];
			llama_seq_id seqId = inputIndex - start;

			for (int32 tokenIndex = 0; tokenIndex < plan->nTokens; tokenIndex++)
			{
				batch.token[globalToken] = plan->tokens[tokenIndex];
				batch.pos[globalToken] = plan->positionIds[tokenIndex];
				batch.n_seq_id[globalToken] = 1;
				batch.seq_id[globalToken][0] = seqId;
				batch.logits[globalToken] = 1;
				globalToken++;
			}
		}

		if (llama_model_has_encoder(entry->model) &&
			!llama_model_has_decoder(entry->model))
		{
			INSTR_TIME_SET_CURRENT(phaseStart);
			rc = llama_encode(entry->ctx, batch);
			timing.llamaUs += PgColbertElapsedUs(phaseStart);
		}
		else
		{
			llama_memory_clear(llama_get_memory(entry->ctx), true);
			INSTR_TIME_SET_CURRENT(phaseStart);
			rc = llama_decode(entry->ctx, batch);
			timing.llamaUs += PgColbertElapsedUs(phaseStart);
		}
		if (rc != 0)
		{
			PgColbertSetError(ctx, errorMessage,
							  "llama model evaluation failed with status %d", rc);
			goto cleanup;
		}

		embeddings = llama_get_embeddings(entry->ctx);
		if (embeddings == NULL)
		{
			PgColbertSetError(ctx, errorMessage,
							  "llama did not return token embeddings");
			goto cleanup;
		}

		for (int32 inputIndex = start; inputIndex < end; inputIndex++)
		{
			PgColbertEncodePlan *plan = &plans[inputIndex];
			PgColbertEngineOutput *output = &outputs[inputIndex];
			int32		retained = 0;

			INSTR_TIME_SET_CURRENT(phaseStart);
			oldCtx = MemoryContextSwitchTo(ctx);
			output->engine = PgColbertEngineName();
			output->path = pstrdup(entry->path);
			output->input = pstrdup(inputs[inputIndex]);
			output->prefix = pstrdup(spec->prefix);
			output->profileSource = pstrdup(spec->profile.sourceName);
			output->attentionMaskStatus = pstrdup(attentionMaskStatus);
			output->knownLimitations = pstrdup(spec->profile.knownLimitations);
			output->dim = entry->nEmbdOut;
			output->count = plan->outputCount;
			output->planTokenCount = plan->nTokens;
			output->normalized = true;
			output->loadedFromCache = loadedFromCache;
			output->tokenIds =
				(int32 *) palloc0(sizeof(int32) * (Size) plan->outputCount);
			output->values =
				(float4 *) palloc0(sizeof(float4) * (Size) plan->outputCount *
								   (Size) entry->nEmbdOut);
			MemoryContextSwitchTo(oldCtx);
			timing.outputUs += PgColbertElapsedUs(phaseStart);

			if (spec->debugTokens)
			{
				INSTR_TIME_SET_CURRENT(phaseStart);
				oldCtx = MemoryContextSwitchTo(ctx);
				output->tokenDebug =
					(PgColbertTokenDebug *) palloc0(sizeof(PgColbertTokenDebug) *
													(Size) plan->nTokens);
				for (int32 tokenIndex = 0;
					 tokenIndex < plan->nTokens;
					 tokenIndex++)
				{
					const char *piece = llama_vocab_get_text(vocab,
															 plan->tokens[tokenIndex]);

					output->tokenDebug[tokenIndex].index = tokenIndex;
					output->tokenDebug[tokenIndex].id =
						(int32) plan->tokens[tokenIndex];
					output->tokenDebug[tokenIndex].piece =
						piece != NULL ? pstrdup(piece) : NULL;
					output->tokenDebug[tokenIndex].positionId =
						plan->positionIds[tokenIndex];
					output->tokenDebug[tokenIndex].tokenTypeId =
						plan->tokenTypeIds[tokenIndex];
					output->tokenDebug[tokenIndex].attentionMask =
						plan->attentionMask[tokenIndex];
					output->tokenDebug[tokenIndex].outputEnabled =
						plan->outputMask[tokenIndex];
					output->tokenDebug[tokenIndex].retained =
						plan->retainMask[tokenIndex];
					output->tokenDebug[tokenIndex].retainReason =
						pstrdup(plan->retainReasons[tokenIndex]);
				}
				MemoryContextSwitchTo(oldCtx);
				timing.debugUs += PgColbertElapsedUs(phaseStart);
			}

			INSTR_TIME_SET_CURRENT(phaseStart);
			for (int32 tokenIndex = 0; tokenIndex < plan->nTokens; tokenIndex++)
			{
				const float *vector;
				int32		vectorDim;
				double		norm = 0.0;

				if (!plan->outputMask[tokenIndex])
				{
					readToken++;
					continue;
				}

				outputOrdinal++;
				vector = embeddings + ((Size) readToken *
									   (Size) entry->nEmbdModel);
				readToken++;
				vectorDim = entry->nEmbdModel;
				if (entry->hasProjection)
				{
					if (!PgColbertApplyProjectionChain(entry, vector,
													   projectionWorkspaceA,
													   projectionWorkspaceB,
													   &vector, &vectorDim,
													   ctx, errorMessage))
						goto cleanup;
					if (vectorDim != output->dim)
					{
						PgColbertSetError(ctx, errorMessage,
										  "ColBERT projection output dimension is %d, expected %d",
										  vectorDim, output->dim);
						goto cleanup;
					}
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
				output->tokenIds[retained] = plan->tokens[tokenIndex];
				for (int32 j = 0; j < output->dim; j++)
					output->values[(Size) retained * (Size) output->dim +
								   (Size) j] = (float4) (vector[j] / norm);
				retained++;
			}
			timing.projectionUs += PgColbertElapsedUs(phaseStart);
		}

		if (outputOrdinal != totalOutputs || readToken != totalTokens)
		{
			PgColbertSetError(ctx, errorMessage,
							  "llama returned %d embeddings for %d requested outputs across %d tokens",
							  outputOrdinal, totalOutputs, totalTokens);
			goto cleanup;
		}
		start = end;
	}

	PgColbertSetTotalTiming(&timing, totalStart);
	for (int32 i = 0; i < inputCount; i++)
		outputs[i].timing = timing;
	PgColbertLogTiming(spec, &timing, "batch");
	ok = true;
	goto cleanup;

oom:
	PgColbertSetError(ctx, errorMessage,
					  "out of memory while running batched llama embedding");
	goto cleanup;

cleanup:
	if (batchInitialized)
		llama_batch_free(batch);
	free(projectionWorkspaceA);
	free(projectionWorkspaceB);
	if (plans != NULL)
	{
		for (int32 i = 0; i < inputCount; i++)
			PgColbertFreeEncodePlan(&plans[i]);
	}
	if (fullTexts != NULL)
	{
		for (int32 i = 0; i < inputCount; i++)
			free(fullTexts[i]);
	}
	free(plans);
	free(fullTexts);
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
	if (!PgColbertLoadModel(spec, path, Max(spec->batchSequences, 1),
							ctx, &entry, &loadedFromCache,
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
	info->tokenizerStatus = pstrdup(spec->profile.tokenizerStatus);
	info->projectionKind = pstrdup(spec->profile.projectionKind);
	info->projectionStatus = pstrdup(entry->nEmbdOut == spec->expectedDim ?
									 "ok" : "missing_or_unexpected_dim");
	info->projectionModuleCount = entry->projectionModuleCount;
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
PgColbertEngineEncodeBatch(const PgColbertModelSpec *spec,
						   const char *const *inputs,
						   int32 inputCount,
						   MemoryContext ctx,
						   PgColbertEngineOutput *outputs,
						   char **errorMessage)
{
	try
	{
		return PgColbertEncodeBatchBody(spec, inputs, inputCount, ctx, outputs,
										errorMessage);
	}
	catch (...)
	{
		return PgColbertSetError(ctx, errorMessage,
								 "unexpected C++ exception in batched llama engine");
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
