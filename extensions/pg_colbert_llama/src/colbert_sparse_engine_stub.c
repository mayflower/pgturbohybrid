/*
 * colbert_sparse_engine_stub.c
 *
 * Deterministic stub implementation of the sparse (SPLADE-style) model backend
 * seam declared in colbert_engine.h.  It requires no model download and emits
 * reproducible (term_id, weight) bags derived from the input text, so the sparse
 * llama_embed API is exercisable in CI without ONNX Runtime or a trained model.
 *
 * It reports implemented=false: the output is a hashing stub, not a real sparse
 * model.  A real backend would replace this object (compile-time selected via
 * PG_COLBERT_LLAMA_SPARSE_ENGINE) and report implemented=true.
 */
#include "postgres.h"

#include <ctype.h>
#include <string.h>

#include "lib/stringinfo.h"

#include "colbert_engine.h"

/* Deterministic stub vocabulary size (mirrors a BERT-family WordPiece vocab). */
#define PG_COLBERT_SPARSE_STUB_VOCAB 30522

const char *
PgSparseEngineName(void)
{
	return "stub";
}

bool
PgSparseEngineImplemented(void)
{
	/* The stub is deterministic but is not a trained sparse model. */
	return false;
}

/*
 * Derive a sparse (term_id, weight) bag from the input text: one entry per
 * alphanumeric token, term_id = FNV-1a hash of the lowercased token modulo the
 * vocab size, weight = token length.  Repeated tokens emit repeated entries so
 * the SQL layer's deduplicate option is exercised.
 */
bool
PgSparseEngineEncode(const PgColbertModelSpec *spec, const char *input,
					 MemoryContext ctx, PgColbertEngineOutput *output,
					 char **errorMessage)
{
	MemoryContext oldCtx;
	Size		len;
	int32		capacity = 16;
	int32		count = 0;
	int32	   *termIds;
	float4	   *weights;
	Size		i = 0;

	(void) errorMessage;

	oldCtx = MemoryContextSwitchTo(ctx);
	len = strlen(input);
	termIds = (int32 *) palloc(sizeof(int32) * (Size) capacity);
	weights = (float4 *) palloc(sizeof(float4) * (Size) capacity);

	memset(output, 0, sizeof(*output));
	output->engine = PgSparseEngineName();
	output->input = pstrdup(input);
	output->prefix = pstrdup(spec->prefix);
	output->profileSource = pstrdup(spec->profile.sourceName);
	output->attentionMaskStatus = pstrdup("ok");
	output->knownLimitations = pstrdup(spec->profile.knownLimitations);
	output->sparseVocabSize = PG_COLBERT_SPARSE_STUB_VOCAB;

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
		termIds[count] = (int32) (hash % (uint32) PG_COLBERT_SPARSE_STUB_VOCAB);
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
	return true;
}

bool
PgSparseEngineEncodeBatch(const PgColbertModelSpec *spec,
						  const char *const *inputs, int32 inputCount,
						  MemoryContext ctx, PgColbertEngineOutput *outputs,
						  char **errorMessage)
{
	for (int32 i = 0; i < inputCount; i++)
	{
		if (!PgSparseEngineEncode(spec, inputs[i], ctx, &outputs[i],
								  errorMessage))
			return false;
	}
	return true;
}

bool
PgSparseEngineGetModelInfo(const PgColbertModelSpec *spec, MemoryContext ctx,
						   PgSparseEngineModelInfo *info, char **errorMessage)
{
	MemoryContext oldCtx;
	StringInfoData path;

	(void) errorMessage;

	oldCtx = MemoryContextSwitchTo(ctx);
	initStringInfo(&path);
	appendStringInfo(&path, "%s/%s", spec->modelDir, spec->alias);
	memset(info, 0, sizeof(*info));
	info->engine = PgSparseEngineName();
	info->implemented = PgSparseEngineImplemented();
	info->supportsSparse = true;
	info->vocabSize = PG_COLBERT_SPARSE_STUB_VOCAB;
	info->path = path.data;
	info->limitations =
		"deterministic hashing stub; not a trained SPLADE model. "
		"Build with a real PG_COLBERT_LLAMA_SPARSE_ENGINE for production sparse embeddings.";
	MemoryContextSwitchTo(oldCtx);
	return true;
}
