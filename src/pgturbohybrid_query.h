#ifndef PGTURBOHYBRID_QUERY_H
#define PGTURBOHYBRID_QUERY_H

#include "postgres.h"

#include "tsearch/ts_type.h"
#include "pgturbohybrid_vector_compat.h"

#define PGTURBOHYBRID_QUERY_VERSION 1

#define PGTURBOHYBRID_QUERY_FLAG_HAS_VECTOR			0x0001
#define PGTURBOHYBRID_QUERY_FLAG_HAS_TSQUERY			0x0002
#define PGTURBOHYBRID_QUERY_FLAG_ALPHA_IS_SET			0x0004
#define PGTURBOHYBRID_QUERY_FLAG_FINAL_K_IS_SET		0x0008
#define PGTURBOHYBRID_QUERY_FLAG_REQUIRE_BM25_MATCH	0x0010
#define PGTURBOHYBRID_QUERY_FLAG_DENSE_K_DEFAULTED		0x0020
#define PGTURBOHYBRID_QUERY_FLAG_BM25_K_DEFAULTED		0x0040
#define PGTURBOHYBRID_QUERY_FLAG_RRF_K_DEFAULTED		0x0080

typedef enum PgturbohybridFusionMode
{
	PGTURBOHYBRID_FUSION_RRF = 1,
	PGTURBOHYBRID_FUSION_WEIGHTED = 2
} PgturbohybridFusionMode;

typedef struct PgturbohybridQueryHeader
{
	int32		vl_len_;
	uint16		version;
	uint16		flags;
	uint16		fusion;
	uint16		reserved;
	float8		denseWeight;
	float8		bm25Weight;
	float8		alpha;
	int32		rrfK;
	int32		denseK;
	int32		bm25K;
	int32		finalK;
	int32		vectorBytes;
	int32		tsqueryBytes;
	/* payload starts at MAXALIGN(sizeof(PgturbohybridQueryHeader)) */
} PgturbohybridQueryHeader;

#define DatumGetPgturbohybridQuery(x) ((PgturbohybridQueryHeader *) PG_DETOAST_DATUM(x))
#define PG_GETARG_PGTURBOHYBRID_QUERY_P(x) DatumGetPgturbohybridQuery(PG_GETARG_DATUM(x))
#define PG_RETURN_PGTURBOHYBRID_QUERY_P(x) PG_RETURN_POINTER(x)

Vector	   *PgturbohybridQueryGetVector(PgturbohybridQueryHeader *query);
TSQuery		PgturbohybridQueryGetTsQuery(PgturbohybridQueryHeader *query);
void		PgturbohybridQueryValidate(PgturbohybridQueryHeader *query);
void		PgturbohybridQueryValidateFast(PgturbohybridQueryHeader *query);
const char *PgturbohybridQueryFusionName(uint16 fusion);

#endif
