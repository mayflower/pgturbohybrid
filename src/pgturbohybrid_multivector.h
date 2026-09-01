#ifndef PGTURBOHYBRID_MULTIVECTOR_H
#define PGTURBOHYBRID_MULTIVECTOR_H

#include "postgres.h"

#include "fmgr.h"
#include "storage/itemptr.h"
#include "utils/memutils.h"

#include "pgturbohybrid_vector_compat.h"

#define PGTURBOHYBRID_MULTIVECTOR_MAX_DIM PGTURBOHYBRID_VECTOR_MAX_DIM
#define PGTURBOHYBRID_MULTIVECTOR_MAX_COUNT 4096
#define PGTURBOHYBRID_MULTIVECTOR_DEFAULT_MAX_DOC_VECTORS 256
#define PGTURBOHYBRID_MULTIVECTOR_DEFAULT_MAX_QUERY_VECTORS 64
#define PGTURBOHYBRID_MULTIVECTOR_DEBUG_TRACE_LIMIT_MAX 1000
#define PGTURBOHYBRID_MULTIVECTOR_TOKEN_STATS_LIMIT_MAX 4096
#define PGTURBOHYBRID_MULTIVECTOR_TOKEN_POOLING_OFF 0
#define PGTURBOHYBRID_MULTIVECTOR_TOKEN_POOLING_KMEANS 1
#define PGTURBOHYBRID_MULTIVECTOR_TOKEN_POOLING_GREEDY_COSINE 2
#define PGTURBOHYBRID_MULTIVECTOR_TOKEN_POOLING_DEFAULT_TARGET_RATIO 0.5
#define PGTURBOHYBRID_MULTIVECTOR_TOKEN_POOLING_DEFAULT_MIN_TOKENS 16
#define PGTURBOHYBRID_MULTIVECTOR_CENTROIDS_OFF 0
#define PGTURBOHYBRID_MULTIVECTOR_CENTROIDS_KMEANS 1
#define PGTURBOHYBRID_MULTIVECTOR_CENTROID_COUNT_AUTO 0
#define PGTURBOHYBRID_MULTIVECTOR_CENTROID_COUNT_MAX 1024
typedef enum PgturbohybridMultiVectorProxyEncoder
{
	PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_MEAN = 0,
	PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_MAX_POOL = 1,
	PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_RANDOM_PROJECTION_FDE = 2,
	/* value 3 is retired; do not reuse it for persisted reloptions */
	PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_NORMALIZED_MEAN = 4,
	PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_FIRST_TOKEN = 5,
	PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_MAX_ABS_MEAN = 6,
	PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_CENTROID_MEAN = 7,
	PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_LEARNED_PROJECTION_V1 = 8
} PgturbohybridMultiVectorProxyEncoder;

#define PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_MEAN_POOL \
	PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_MEAN
#define PGTURBOHYBRID_DEFAULT_MULTIVECTOR_PROXY_ENCODER \
	PGTURBOHYBRID_MULTIVECTOR_PROXY_ENCODER_NORMALIZED_MEAN
#define PGTURBOHYBRID_MULTIVECTOR_FLAG_CONTEXTS 0x00000001U
#define PGTURBOHYBRID_MULTIVECTOR_FLAG_FIELDS 0x00000002U
#define PGTURBOHYBRID_MULTIVECTOR_KNOWN_FLAGS \
	(PGTURBOHYBRID_MULTIVECTOR_FLAG_CONTEXTS | PGTURBOHYBRID_MULTIVECTOR_FLAG_FIELDS)
#define PGTURBOHYBRID_MULTIVECTOR_CONTEXT_MODE_FLAT 0
#define PGTURBOHYBRID_MULTIVECTOR_CONTEXT_MODE_CONTEXT_LEVEL 1
#define PGTURBOHYBRID_MULTIVECTOR_FIELD_MODE_OFF 0
#define PGTURBOHYBRID_MULTIVECTOR_FIELD_MODE_WEIGHTED 1
#define PGTURBOHYBRID_MULTIVECTOR_SIZE(_count, _dim) \
	(offsetof(PgturbohybridMultiVector, values) + \
	 sizeof(float) * (Size) (_count) * (Size) (_dim))

#define TQ_INVALID_DOC_ID UINT32_MAX
#define TQ_INVALID_SUBVECTOR_ORDINAL UINT16_MAX

/*
 * Multivector indexes expand one heap tuple into multiple graph nodes.
 *
 * Single-vector path: nodeId can still be treated as the row/result identity.
 * Multivector path: nodeId identifies one document token/subvector only.
 * Result ranking and deduplication must use docId, then docId resolves back
 * to one heap TID.
 */
typedef uint32 TqDocId;
typedef uint16 TqSubvectorOrdinal;

typedef struct TqMultiVectorNodeMapEntry
{
	TqDocId		docId;
	TqSubvectorOrdinal tokenOrdinal;
} TqMultiVectorNodeMapEntry;

typedef struct TqMultiVectorDocMapEntry
{
	ItemPointerData heapTid;
	uint32		firstNodeId;
	uint16		tokenCount;
	uint16		originalTokenCount;
	uint16		pooledTokenCount;
} TqMultiVectorDocMapEntry;

typedef struct PgturbohybridMultiVectorAdmissionTraceEntry
{
	TqDocId		docId;
	BlockNumber block;
	OffsetNumber offset;
	uint32		bestNodeId;
	double		approximateScoreBeforeRerank;
	uint32		queryTokenCoverageCount;
	uint32		rawHitCount;
	uint32		duplicateHitCount;
	uint32		candidateRankBeforeTruncation;
	bool		retainedForExactRerank;
	bool		exactRerankScoreAvailable;
	double		exactRerankScore;
}			PgturbohybridMultiVectorAdmissionTraceEntry;

typedef struct PgturbohybridMultiVectorTokenStatsEntry
{
	uint32		queryTokenOrdinal;
	uint32		rawHits;
	uint32		uniqueDocs;
	uint32		duplicateDocHits;
	bool		topHitSimilarityAvailable;
	double		topHitSimilarity;
	double		contributionToTopCandidates;
	uint32		candidateDocsRetainedFromToken;
	bool		skipped;
}			PgturbohybridMultiVectorTokenStatsEntry;

typedef struct PgturbohybridMultiVector
{
	int32		vl_len_;
	int32		dim;
	int32		count;
	uint32		flags;
	float		values[FLEXIBLE_ARRAY_MEMBER];
} PgturbohybridMultiVector;

#define DatumGetPgturbohybridMultiVector(x) \
	PgturbohybridDatumGetMultiVector(x)
#define PG_GETARG_PGTURBOHYBRID_MULTIVECTOR_P(x) \
	DatumGetPgturbohybridMultiVector(PG_GETARG_DATUM(x))
#define PG_RETURN_PGTURBOHYBRID_MULTIVECTOR_P(x) PG_RETURN_POINTER(x)

typedef struct PgturbohybridMultiVectorModelInfo
{
	const char *modelName;
	int32		dim;
	int32		defaultQueryMaxTokens;
	int32		defaultDocMaxTokens;
	const char *distanceMode;
	bool		normalizedTokens;
	const char *recommendedStorageKind;
	const char *tokenMaskPolicy;
	const char *fieldContextPolicy;
	const char *status;
	const char *notes;
} PgturbohybridMultiVectorModelInfo;

PgturbohybridMultiVector *PgturbohybridDatumGetMultiVector(Datum value);
Oid			PgturbohybridMultiVectorTypeOid(void);
bool		PgturbohybridTypeIsMultiVector(Oid typeOid);
void		PgturbohybridCheckMultiVector(const PgturbohybridMultiVector *mv);
void		PgturbohybridCheckSameMultiVectorDims(const PgturbohybridMultiVector *a,
												   const PgturbohybridMultiVector *b);
const PgturbohybridMultiVectorModelInfo *PgturbohybridMultiVectorLookupModel(const char *modelName);
TqDocId		PgturbohybridMultiVectorMakeDocId(uint64 docOrdinal);
TqSubvectorOrdinal PgturbohybridMultiVectorMakeSubvectorOrdinal(uint32 tokenOrdinal);
void		PgturbohybridMultiVectorCheckTokenCount(uint32 tokenCount,
													 uint32 maxTokenCount);
void		PgturbohybridMultiVectorCheckDim(uint32 dim, uint32 maxDim);
Size		PgturbohybridMultiVectorFloatCount(int32 count, int32 dim);
Size		PgturbohybridMultiVectorSize(int32 count, int32 dim);
Size		PgturbohybridMultiVectorExtendedSize(int32 count, int32 dim,
												  int32 contextCount,
												  bool hasFields);
const float *PgturbohybridMultiVectorValues(const PgturbohybridMultiVector *mv,
											int32 ordinal);
bool		PgturbohybridMultiVectorHasContexts(const PgturbohybridMultiVector *mv);
int32		PgturbohybridMultiVectorContextCount(const PgturbohybridMultiVector *mv);
const int32 *PgturbohybridMultiVectorContextOffsets(const PgturbohybridMultiVector *mv);
const int32 *PgturbohybridMultiVectorContextFields(const PgturbohybridMultiVector *mv);
Size		PgturbohybridMultiVectorSubvectorSize(const PgturbohybridMultiVector *mv);
void		PgturbohybridMultiVectorCopySubvectorToVector(const PgturbohybridMultiVector *mv,
														   int32 ordinal,
														   Vector *dst);
PgturbohybridMultiVector *PgturbohybridMultiVectorPoolDocumentTokens(const PgturbohybridMultiVector *mv,
																	 int mode,
																	 double targetRatio,
																	 int minTokens,
																	 MemoryContext ctx);
int			PgturbohybridMultiVectorCentroidCountForDoc(const PgturbohybridMultiVector *doc,
														 int requested);
float		PgturbohybridMultiVectorCentroidResidualMean(const PgturbohybridMultiVector *doc,
														 const PgturbohybridMultiVector *centroids);
const char *PgturbohybridMultiVectorProxyEncoderName(int encoder);
bool		PgturbohybridMultiVectorLearnedProjectionInfo(bool *loaded,
														  int32 *dim,
														  uint64 *weightBytes,
														  const char **model,
														  const char **checksum);
Vector	   *PgturbohybridMultiVectorBuildProxyVector(const PgturbohybridMultiVector *mv,
													 int encoder,
													 MemoryContext ctx);
Vector	   *PgturbohybridMultiVectorBuildProxyVectorWithCentroids(const PgturbohybridMultiVector *mv,
																  const PgturbohybridMultiVector *centroids,
																  int encoder,
																  int centroidCount,
																  MemoryContext ctx);
Vector	   *PgturbohybridMultiVectorBuildQueryProxyVector(const PgturbohybridMultiVector *query,
														  int encoder,
														  MemoryContext ctx);
double		TqDotProductF32Scalar(const float *a, const float *b, int32 dim);
typedef void (*TqDotProductF32BlockFunc) (const float *queryValues,
										  const float *docValues,
										  int32 dim,
										  int32 blockCount,
										  double *dots);
TqDotProductF32BlockFunc TqResolveDotProductF32BlockKernel(void);
void		TqDotProductF32BlockAuto(const float *queryValues,
									   const float *docValues,
									   int32 dim,
									   int32 blockCount,
									   double *dots);
typedef int64 (*TqCompactCodeScoreFunc) (const int16 *queryCodes,
										 const int16 *docCodes,
										 int32 count);
typedef void (*TqCompactCodeScoreBatchFunc) (int16 queryCode,
											 const int16 *docCodes,
											 int32 count,
											 int64 *scores);
int64		TqCompactCodeScoreScalar(const int16 *queryCodes,
									 const int16 *docCodes,
									 int32 count);
void		TqCompactCodeScoreBatchScalar(int16 queryCode,
										   const int16 *docCodes,
										   int32 count,
										   int64 *scores);
TqCompactCodeScoreFunc TqResolveCompactCodeScoreKernel(const char *forceKernel);
const char *TqCompactCodeScoreKernelName(TqCompactCodeScoreFunc func);
TqCompactCodeScoreBatchFunc TqResolveCompactCodeScoreBatchKernel(const char *forceKernel);
const char *TqCompactCodeScoreBatchKernelName(TqCompactCodeScoreBatchFunc func);
double		TqMultiVectorMaxSimScalar(const PgturbohybridMultiVector *query,
									   const PgturbohybridMultiVector *doc);
double		TqMultiVectorMaxSimBlockedScalar(const PgturbohybridMultiVector *query,
											  const PgturbohybridMultiVector *doc);
double		TqMultiVectorMaxSimBlocked(const PgturbohybridMultiVector *query,
										const PgturbohybridMultiVector *doc);
double		TqMultiVectorMaxSim(const PgturbohybridMultiVector *query,
								 const PgturbohybridMultiVector *doc);
double		TqMultiVectorSymmetricMaxSimAverage(const PgturbohybridMultiVector *a,
												const PgturbohybridMultiVector *b);
double		TqMultiVectorSymmetricMaxSimAverageUnchecked(const PgturbohybridMultiVector *a,
														 const PgturbohybridMultiVector *b);
double		TqMultiVectorMaxSimWeighted(const PgturbohybridMultiVector *query,
										 const PgturbohybridMultiVector *doc,
										 const float4 *queryWeights,
										 const bool *queryMask);
double		TqMultiVectorMaxSimContextLevel(const PgturbohybridMultiVector *query,
											 const PgturbohybridMultiVector *doc);
double		TqMultiVectorMaxSimContextLevelWeighted(const PgturbohybridMultiVector *query,
													 const PgturbohybridMultiVector *doc,
													 const float4 *queryWeights,
													 const bool *queryMask);
double		TqMultiVectorMaxSimFieldWeighted(const PgturbohybridMultiVector *query,
											  const PgturbohybridMultiVector *doc,
											  const int32 *fieldIds,
											  const float4 *weights,
											  int fieldCount);
const char *TqMultiVectorMaxSimKernelName(void);

FUNCTION_PREFIX Datum pgturbohybrid_multivector_in(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_out(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_recv(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_send(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_constructor(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_from_float4(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_from_contexts(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_from_contexts_and_fields(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_dims(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_count(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_context_count(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_context_offsets(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_field_ids(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_subvector(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_to_vector_array(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_maxsim(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_context_maxsim(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_field_weighted_maxsim(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_maxsim_scalar(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_maxsim_blocked_scalar(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_maxsim_distance(PG_FUNCTION_ARGS);
FUNCTION_PREFIX Datum pgturbohybrid_multivector_query_distance(PG_FUNCTION_ARGS);

#endif
