#include "postgres.h"

#include <ctype.h>

#include "lib/stringinfo.h"

#include "colbert_engine.h"

/* Deterministic stub vocabulary size (mirrors a BERT-family WordPiece vocab). */
#define PG_COLBERT_STUB_SPARSE_VOCAB 30522

const char *
PgColbertEngineName(void)
{
	return "stub";
}

/*
 * Deterministically derive a sparse (term_id, weight) bag from the input text:
 * one entry per whitespace/punctuation-delimited token, term_id = FNV-1a hash of
 * the lowercased token modulo the vocab size, weight = token length.  Repeated
 * tokens emit repeated entries so the SQL layer's deduplicate option is
 * exercised.  No model download is required.
 */
static void
PgColbertStubSparseEncode(const PgColbertModelSpec *spec, const char *input,
						  MemoryContext ctx, PgColbertEngineOutput *output)
{
	MemoryContext oldCtx = MemoryContextSwitchTo(ctx);
	Size		len = strlen(input);
	int32		capacity = 16;
	int32		count = 0;
	int32	   *termIds = (int32 *) palloc(sizeof(int32) * (Size) capacity);
	float4	   *weights = (float4 *) palloc(sizeof(float4) * (Size) capacity);
	Size		i = 0;

	memset(output, 0, sizeof(*output));
	output->engine = PgColbertEngineName();
	output->input = pstrdup(input);
	output->prefix = pstrdup(spec->prefix);
	output->profileSource = pstrdup(spec->profile.sourceName);
	output->attentionMaskStatus = pstrdup("ok");
	output->knownLimitations = pstrdup(spec->profile.knownLimitations);
	output->sparseVocabSize = PG_COLBERT_STUB_SPARSE_VOCAB;

	while (i < len)
	{
		Size		start;
		uint32		hash = 2166136261u;	/* FNV-1a offset basis */
		int32		tokenLen;

		if (!isalnum((unsigned char) input[i]))
		{
			i++;
			continue;
		}
		start = i;
		while (i < len && isalnum((unsigned char) input[i]))
		{
			unsigned char c = (unsigned char) tolower((unsigned char) input[i]);

			hash ^= c;
			hash *= 16777619u;		/* FNV-1a prime */
			i++;
		}
		tokenLen = (int32) (i - start);

		if (count == capacity)
		{
			capacity *= 2;
			termIds = (int32 *) repalloc(termIds, sizeof(int32) * (Size) capacity);
			weights = (float4 *) repalloc(weights, sizeof(float4) * (Size) capacity);
		}
		termIds[count] = (int32) (hash % (uint32) PG_COLBERT_STUB_SPARSE_VOCAB);
		weights[count] = (float4) tokenLen;
		count++;
	}

	output->sparseCount = count;
	output->sparseTermIds = termIds;
	output->sparseWeights = weights;
	output->timing.inputs = 1;
	output->timing.tokens = count;
	output->timing.outputVectors = count;

	MemoryContextSwitchTo(oldCtx);
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

	(void) errorMessage;

	if (spec->outputMode == PG_LLAMA_EMBED_OUTPUT_SPARSE)
	{
		PgColbertStubSparseEncode(spec, input, ctx, output);
		return true;
	}

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
