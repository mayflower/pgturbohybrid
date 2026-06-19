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

	count = spec->outputMode == PG_LLAMA_EMBED_OUTPUT_DENSE ? 1 :
		(spec->role == PG_COLBERT_ROLE_QUERY ? 2 : 3);
	if (count > spec->maxVectors)
		count = spec->maxVectors;

	oldCtx = MemoryContextSwitchTo(ctx);
	memset(output, 0, sizeof(*output));
	output->engine = PgColbertEngineName();
	output->input = pstrdup(input);
	output->prefix = pstrdup(spec->prefix);
	output->profileSource = pstrdup(spec->profile.sourceName);
	output->attentionMaskStatus = pstrdup("ok");
	output->knownLimitations = pstrdup(spec->profile.knownLimitations);
	output->dim = dim;
	output->count = count;
	output->planTokenCount = count;
	output->normalized = spec->normalize;
	output->loadedFromCache = false;
	output->tokenIds = (int32 *) palloc0(sizeof(int32) * (Size) count);
	output->values = (float4 *) palloc0(sizeof(float4) * (Size) count * (Size) dim);
	output->timing.inputs = 1;
	output->timing.tokens = count;
	output->timing.outputVectors = count;

	if (spec->debugTokens)
		output->tokenDebug =
			(PgColbertTokenDebug *) palloc0(sizeof(PgColbertTokenDebug) *
											(Size) count);

	for (int32 i = 0; i < count; i++)
	{
		output->tokenIds[i] = i + 1;
		if (output->tokenDebug != NULL)
		{
			output->tokenDebug[i].index = i;
			output->tokenDebug[i].id = i + 1;
			output->tokenDebug[i].piece = pstrdup(i == 0 ? "[CLS]" : "stub");
			output->tokenDebug[i].positionId = i;
			output->tokenDebug[i].tokenTypeId =
				spec->role == PG_COLBERT_ROLE_QUERY ?
				spec->profile.queryTokenTypeId :
				(spec->role == PG_COLBERT_ROLE_DOC ?
				 spec->profile.documentTokenTypeId : -1);
			output->tokenDebug[i].attentionMask = 1;
			output->tokenDebug[i].outputEnabled = true;
			output->tokenDebug[i].retained = true;
			output->tokenDebug[i].retainReason =
				pstrdup(spec->outputMode == PG_LLAMA_EMBED_OUTPUT_DENSE ?
					   "retained_dense" :
					   (spec->role == PG_COLBERT_ROLE_QUERY ?
						"retained_query" : "retained_document"));
		}
		if (spec->normalize)
			output->values[(Size) i * (Size) dim + (i % dim)] = 1.0f;
		else
			for (int32 j = 0; j < dim; j++)
				output->values[(Size) i * (Size) dim + (Size) j] =
					(float4) (i + j + 1);
	}

	MemoryContextSwitchTo(oldCtx);
	return true;
}

bool
PgColbertEngineEncodeBatch(const PgColbertModelSpec *spec,
						   const char *const *inputs,
						   int32 inputCount,
						   MemoryContext ctx,
						   PgColbertEngineOutput *outputs,
						   char **errorMessage)
{
	for (int32 i = 0; i < inputCount; i++)
	{
		if (!PgColbertEngineEncode(spec, inputs[i], ctx, &outputs[i],
								   errorMessage))
			return false;
	}
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
	info->tokenizerStatus = spec->profile.tokenizerStatus;
	info->projectionKind = spec->profile.projectionKind;
	info->projectionStatus = "ok";
	info->projectionModuleCount = spec->profile.projectionModuleCount;
	MemoryContextSwitchTo(oldCtx);
	return true;
}
