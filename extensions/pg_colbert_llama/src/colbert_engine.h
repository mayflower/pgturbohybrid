#ifndef PG_COLBERT_LLAMA_ENGINE_H
#define PG_COLBERT_LLAMA_ENGINE_H

#include "postgres.h"

#include "utils/memutils.h"

#define PG_COLBERT_LLAMA_MAX_ALIAS 128

typedef enum PgColbertRole
{
	PG_COLBERT_ROLE_QUERY = 1,
	PG_COLBERT_ROLE_DOC = 2
} PgColbertRole;

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
	int			nGpuLayers;
	int			cacheSize;
	int			queryLength;
} PgColbertModelSpec;

typedef struct PgColbertEngineOutput
{
	const char *engine;
	const char *path;
	int32		dim;
	int32		count;
	bool		normalized;
	bool		loadedFromCache;
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
} PgColbertEngineModelInfo;

extern bool PgColbertEngineEncode(const PgColbertModelSpec *spec,
								  const char *input,
								  MemoryContext ctx,
								  PgColbertEngineOutput *output,
								  char **errorMessage);
extern bool PgColbertEngineGetModelInfo(const PgColbertModelSpec *spec,
										MemoryContext ctx,
										PgColbertEngineModelInfo *info,
										char **errorMessage);
extern const char *PgColbertEngineName(void);
extern bool PgColbertEngineImplemented(void);

#endif
