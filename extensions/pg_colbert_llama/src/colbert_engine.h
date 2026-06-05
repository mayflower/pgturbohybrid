#ifndef PG_COLBERT_LLAMA_ENGINE_H
#define PG_COLBERT_LLAMA_ENGINE_H

#include "postgres.h"

#include "utils/memutils.h"

#define PG_COLBERT_LLAMA_MAX_ALIAS 128
#define PG_COLBERT_LLAMA_MAX_PROFILE_STRING 256
#define PG_COLBERT_LLAMA_MAX_SKIPLIST_TOKENS 512
#define PG_COLBERT_LLAMA_MAX_PROJECTION_MODULES 8

typedef enum PgColbertRole
{
	PG_COLBERT_ROLE_QUERY = 1,
	PG_COLBERT_ROLE_DOC = 2
} PgColbertRole;

typedef enum PgColbertProfileSource
{
	PG_COLBERT_PROFILE_SOURCE_GUC_FALLBACK = 0,
	PG_COLBERT_PROFILE_SOURCE_GGUF = 1,
	PG_COLBERT_PROFILE_SOURCE_SIDECAR = 2
} PgColbertProfileSource;

typedef struct PgColbertSpecialTokensProfile
{
	int32		clsTokenId;
	int32		sepTokenId;
	int32		padTokenId;
	int32		maskTokenId;
	int32		qTokenId;
	int32		dTokenId;
} PgColbertSpecialTokensProfile;

typedef struct PgColbertProjectionModuleProfile
{
	char		type[32];
	char		activation[64];
	char		weightTensor[PG_COLBERT_LLAMA_MAX_PROFILE_STRING];
	char		biasTensor[PG_COLBERT_LLAMA_MAX_PROFILE_STRING];
	int32		inputDim;
	int32		outputDim;
	bool		hasBias;
	bool		normalizeAfter;
	int32		p;
} PgColbertProjectionModuleProfile;

typedef struct PgColbertRuntimeProfile
{
	bool		loaded;
	PgColbertProfileSource source;
	char		sourceName[32];
	char		schema[64];
	char		compatibilityLevel[64];
	char		knownLimitations[PG_COLBERT_LLAMA_MAX_PROFILE_STRING];
	char		tokenizerSource[64];
	char		tokenizerStatus[64];
	char		tokenizerKnownLimitations[PG_COLBERT_LLAMA_MAX_PROFILE_STRING];
	char		backboneFamily[64];
	char		colbertFamily[64];
	char		queryPrefix[PG_COLBERT_LLAMA_MAX_PROFILE_STRING];
	char		documentPrefix[PG_COLBERT_LLAMA_MAX_PROFILE_STRING];
	char		queryRetainPolicy[64];
	char		queryOutputPolicy[64];
	char		documentRetainPolicy[64];
	char		queryAttentionMaskPolicy[64];
	char		documentAttentionMaskPolicy[64];
	char		projectionKind[64];
	int32		outputDim;
	bool		normalize;
	int32		queryMaxLength;
	int32		queryPadTo;
	int32		queryPadTokenId;
	int32		queryTokenTypeId;
	bool		queryAttendToExpansionTokens;
	int32		documentMaxLength;
	int32		documentTokenTypeId;
	PgColbertSpecialTokensProfile specialTokens;
	int32		skiplistTokenIds[PG_COLBERT_LLAMA_MAX_SKIPLIST_TOKENS];
	int32		skiplistTokenCount;
	int32		projectionInputDim;
	int32		projectionOutputDim;
	bool		projectionNormalizeAfter;
	PgColbertProjectionModuleProfile projectionModules[PG_COLBERT_LLAMA_MAX_PROJECTION_MODULES];
	int32		projectionModuleCount;
	bool		requiresProfile;
	bool		strictPylateProfile;
	bool		llamaCppLoadable;
} PgColbertRuntimeProfile;

typedef struct PgColbertModelSpec
{
	char		alias[PG_COLBERT_LLAMA_MAX_ALIAS + 1];
	PgColbertRole role;
	const char *roleName;
	const char *prefix;
	const char *modelDir;
	int			maxVectors;
	int			expectedDim;
	int			threads;
	int			nCtx;
	int			nBatch;
	int			batchSequences;
	int			nGpuLayers;
	int			cacheSize;
	int			queryLength;
	bool		strictProfile;
	PgColbertRuntimeProfile profile;
} PgColbertModelSpec;

typedef struct PgColbertTokenDebug
{
	int32		index;
	int32		id;
	const char *piece;
	int32		positionId;
	int32		tokenTypeId;
	int32		attentionMask;
	bool		outputEnabled;
	bool		retained;
	const char *retainReason;
} PgColbertTokenDebug;

typedef struct PgColbertEngineOutput
{
	const char *engine;
	const char *path;
	const char *input;
	const char *prefix;
	const char *profileSource;
	const char *attentionMaskStatus;
	const char *knownLimitations;
	int32		dim;
	int32		count;
	int32		planTokenCount;
	bool		normalized;
	bool		loadedFromCache;
	PgColbertTokenDebug *tokenDebug;
	int32	   *tokenIds;
	float4	   *values;
} PgColbertEngineOutput;

typedef struct PgColbertEngineModelInfo
{
	const char *engine;
	const char *path;
	bool		implemented;
	bool		loadedFromCache;
	int32		nCtxTrain;
	int32		nEmbdOut;
	int32		nLayer;
	int32		nHead;
	const char *tokenizerStatus;
	const char *projectionKind;
	const char *projectionStatus;
	int32		projectionModuleCount;
} PgColbertEngineModelInfo;

extern bool PgColbertEngineEncode(const PgColbertModelSpec *spec,
								  const char *input,
								  MemoryContext ctx,
								  PgColbertEngineOutput *output,
								  char **errorMessage);
extern bool PgColbertEngineEncodeBatch(const PgColbertModelSpec *spec,
									   const char *const *inputs,
									   int32 inputCount,
									   MemoryContext ctx,
									   PgColbertEngineOutput *outputs,
									   char **errorMessage);
extern bool PgColbertEngineGetModelInfo(const PgColbertModelSpec *spec,
										MemoryContext ctx,
										PgColbertEngineModelInfo *info,
										char **errorMessage);
extern const char *PgColbertEngineName(void);
extern bool PgColbertEngineImplemented(void);

#endif
