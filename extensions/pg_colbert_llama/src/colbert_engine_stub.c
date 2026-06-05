#include "postgres.h"

#include "lib/stringinfo.h"

#include "colbert_engine.h"

const char *
PgColbertEngineName(void)
{
	return "stub";
}

bool
PgColbertEngineImplemented(void)
{
	return true;
}

bool
PgColbertEngineEncode(const PgColbertModelSpec *spec,
					  const char *input,
					  MemoryContext ctx,
					  PgColbertEngineOutput *output,
					  char **errorMessage)
{
	MemoryContext oldCtx;
	int32		count;
	int32		dim = 4;

	(void) input;
	(void) errorMessage;

	count = spec->role == PG_COLBERT_ROLE_QUERY ? 2 : 3;
	if (count > spec->maxVectors)
		count = spec->maxVectors;

	oldCtx = MemoryContextSwitchTo(ctx);
	memset(output, 0, sizeof(*output));
	output->engine = PgColbertEngineName();
	output->dim = dim;
	output->count = count;
	output->normalized = true;
	output->loadedFromCache = false;
	output->tokenIds = (int32 *) palloc0(sizeof(int32) * (Size) count);
	output->values = (float4 *) palloc0(sizeof(float4) * (Size) count * (Size) dim);

	for (int32 i = 0; i < count; i++)
	{
		output->tokenIds[i] = i + 1;
		output->values[(Size) i * (Size) dim + (i % dim)] = 1.0f;
	}

	MemoryContextSwitchTo(oldCtx);
	return true;
}

bool
PgColbertEngineGetModelInfo(const PgColbertModelSpec *spec,
							MemoryContext ctx,
							PgColbertEngineModelInfo *info,
							char **errorMessage)
{
	MemoryContext oldCtx;
	StringInfoData path;

	(void) errorMessage;

	oldCtx = MemoryContextSwitchTo(ctx);
	initStringInfo(&path);
	appendStringInfo(&path, "%s/%s.gguf", spec->modelDir, spec->alias);
	memset(info, 0, sizeof(*info));
	info->engine = PgColbertEngineName();
	info->path = path.data;
	info->implemented = true;
	info->loadedFromCache = false;
	info->nCtxTrain = 0;
	info->nEmbdOut = 4;
	info->nLayer = 0;
	info->nHead = 0;
	MemoryContextSwitchTo(oldCtx);
	return true;
}
