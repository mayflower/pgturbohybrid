#ifndef TQGRAPH_SCORE_H
#define TQGRAPH_SCORE_H

#include "pgturbohybrid_quant.h"

double		PgturbohybridGraphBuildDistance(PgturbohybridQuantBuildState *state, uint32 a, uint32 b);
bool		PgturbohybridGraphCachedExactNodeDistance(PgturbohybridGraphScanOpaque so, Datum query,
										   PgturbohybridGraphScanNode *node,
										   double *distance);
bool		PgturbohybridGraphCodeCodeWeightedRawSimdSelf(const uint8 *code,
											  int dimensions, int bits,
											  const float *ecScale,
											  double *raw);
bool		PgturbohybridGraphCodeCodeDistance(PgturbohybridGraphScanOpaque so, PgturbohybridGraphMetaPageData *meta,
									PgturbohybridGraphScanNode *aNode,
									PgturbohybridGraphScanNode *bNode,
									double *distance);
float		PgturbohybridGraphCodeNorm(const uint8 *code, int dimensions, int bits);
double		PgturbohybridGraphCodeCodeWeightedRawScalar(const uint8 *a, const uint8 *b,
											  int dimensions, int bits,
											  const float *ecScale);
float		PgturbohybridGraphEncodeVector(PgturbohybridQuantBuildState *state, Vector *vector,
								uint8 *code);
float		PgturbohybridGraphEncodeVectorWithXm(PgturbohybridQuantBuildState *state, Vector *vector,
									  uint8 *code, float *xmOut);
float		PgturbohybridGraphEncodeVectorWithXmRenorm(PgturbohybridQuantBuildState *state,
											Vector *vector, uint8 *code,
											float *xmOut);
bool		PgturbohybridGraphExactHighdimEntryDistance(PgturbohybridGraphScanOpaque so, Datum query,
											 PgturbohybridGraphScanNode *node,
											 double *distance);
double		PgturbohybridGraphExactDistance(PgturbohybridGraphSupport *support, Datum a, Datum b);
double		PgturbohybridGraphExactVectorDistance(PgturbohybridGraphScanOpaque so, Datum query,
										char *valuePtr);
TqScoreMode PgturbohybridGraphGetScoreMode(PgturbohybridGraphSupport *support);
double		PgturbohybridGraphMmConstScalar(const float *ecShift, int dimensions);
void		PgturbohybridGraphPrepareBuildQuery(PgturbohybridQuantBuildState *state, uint32 nodeId);
double		PgturbohybridGraphScoreNode(PgturbohybridGraphScanOpaque so, PgturbohybridGraphScanNode *node);
void		PgturbohybridGraphScoreNodeBatch(PgturbohybridGraphScanOpaque so, PgturbohybridGraphScanStorage *storage,
								  uint32 *nodeIds, int nodeCount,
								  double *distances, Datum query);
float		PgturbohybridGraphVectorNorm(Vector *vector);

#endif
