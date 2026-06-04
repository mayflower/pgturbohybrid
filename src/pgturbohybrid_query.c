#include "postgres.h"

#include <math.h>

#include "fmgr.h"
#include "lib/stringinfo.h"
#include "nodes/execnodes.h"
#include "nodes/nodes.h"
#include "nodes/plannodes.h"
#include "nodes/primnodes.h"
#include "utils/builtins.h"
#include "utils/float.h"
#include "utils/memutils.h"

#include "pgturbohybrid_query.h"
#include "pgturbohybrid_am.h"

#if PG_VERSION_NUM >= 160000
#include "varatt.h"
#endif

FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_query_in);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_query_out);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_query_constructor);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_distance);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_l2_distance);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_negative_inner_product);
FUNCTION_PREFIX PG_FUNCTION_INFO_V1(pgturbohybrid_cosine_distance);

static Size PgturbohybridQueryVectorOffset(void);
static Size PgturbohybridQueryMultiVectorOffset(PgturbohybridQueryHeader *query);
static Size PgturbohybridQueryTsQueryOffset(PgturbohybridQueryHeader *query);
static Size PgturbohybridQueryAlignedSize(Size value);
static Size PgturbohybridQuerySizeAdd(Size a, Size b);
static uint16 PgturbohybridQueryParseFusion(text *fusion);
static void PgturbohybridQueryCheckPositiveInt(const char *name, int32 value);
static void PgturbohybridQueryCheckNonNegativeInt(const char *name, int32 value);
static void PgturbohybridQueryRejectTextFallback(void);
static bool PgturbohybridQueryTextIndexOrderByContext(FunctionCallInfo fcinfo);
static void PgturbohybridQueryValidateInternal(PgturbohybridQueryHeader *query,
											   bool strict);
static PgturbohybridQueryHeader *PgturbohybridQueryConstructorCached(FunctionCallInfo fcinfo);
static void PgturbohybridQueryConstructorStoreCache(FunctionCallInfo fcinfo,
												   PgturbohybridQueryHeader *query);

static Size
PgturbohybridQueryVectorOffset(void)
{
	return MAXALIGN(sizeof(PgturbohybridQueryHeader));
}

static Size
PgturbohybridQueryTsQueryOffset(PgturbohybridQueryHeader *query)
{
	return PgturbohybridQuerySizeAdd(PgturbohybridQueryMultiVectorOffset(query),
									 PgturbohybridQueryAlignedSize(query->multivectorBytes));
}

static Size
PgturbohybridQueryMultiVectorOffset(PgturbohybridQueryHeader *query)
{
	return PgturbohybridQuerySizeAdd(PgturbohybridQueryVectorOffset(),
									 PgturbohybridQueryAlignedSize(query->vectorBytes));
}

static Size
PgturbohybridQueryAlignedSize(Size value)
{
	if (value > MaxAllocSize - (MAXIMUM_ALIGNOF - 1))
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("turbohybrid_query payload is too large")));

	return MAXALIGN(value);
}

static Size
PgturbohybridQuerySizeAdd(Size a, Size b)
{
	if (a > MaxAllocSize - b)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("turbohybrid_query payload is too large")));

	return a + b;
}

Vector *
PgturbohybridQueryGetVector(PgturbohybridQueryHeader *query)
{
	PgturbohybridQueryValidateFast(query);

	if ((query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_VECTOR) == 0)
		return NULL;

	return (Vector *) ((char *) query + PgturbohybridQueryVectorOffset());
}

PgturbohybridMultiVector *
PgturbohybridQueryGetMultiVector(PgturbohybridQueryHeader *query)
{
	PgturbohybridQueryValidateFast(query);

	if ((query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_MULTIVECTOR) == 0)
		return NULL;

	return (PgturbohybridMultiVector *) ((char *) query +
										 PgturbohybridQueryMultiVectorOffset(query));
}

TSQuery
PgturbohybridQueryGetTsQuery(PgturbohybridQueryHeader *query)
{
	PgturbohybridQueryValidateFast(query);

	if ((query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_TSQUERY) == 0)
		return NULL;

	return (TSQuery) ((char *) query + PgturbohybridQueryTsQueryOffset(query));
}

const char *
PgturbohybridQueryFusionName(uint16 fusion)
{
	switch ((PgturbohybridFusionMode) fusion)
	{
		case PGTURBOHYBRID_FUSION_RRF:
			return "rrf";
		case PGTURBOHYBRID_FUSION_WEIGHTED:
			return "weighted";
		case PGTURBOHYBRID_FUSION_FAST_WEIGHTED:
			return "fast_weighted";
		case PGTURBOHYBRID_FUSION_CALIBRATED:
			return "calibrated";
	}

	return "unknown";
}

void
PgturbohybridQueryValidate(PgturbohybridQueryHeader *query)
{
	PgturbohybridQueryValidateInternal(query, true);
}

void
PgturbohybridQueryValidateFast(PgturbohybridQueryHeader *query)
{
	PgturbohybridQueryValidateInternal(query, false);
}

static void
PgturbohybridQueryValidateInternal(PgturbohybridQueryHeader *query, bool strict)
{
	Size		actual;
	Size		vectorOffset;
	Size		multivectorOffset;
	Size		tsqueryOffset;
	Size		expected;

	if (query == NULL)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("turbohybrid_query cannot be null")));

	actual = VARSIZE_ANY(query);
	vectorOffset = PgturbohybridQueryVectorOffset();
	if (actual < vectorOffset)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("truncated turbohybrid_query payload")));

	if (query->version != PGTURBOHYBRID_QUERY_VERSION)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("unsupported turbohybrid_query version %u", query->version)));

	if (query->fusion != PGTURBOHYBRID_FUSION_RRF &&
		query->fusion != PGTURBOHYBRID_FUSION_WEIGHTED &&
		query->fusion != PGTURBOHYBRID_FUSION_FAST_WEIGHTED &&
		query->fusion != PGTURBOHYBRID_FUSION_CALIBRATED)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("invalid turbohybrid_query fusion mode %u", query->fusion)));

	if (query->denseKind != PGTURBOHYBRID_DENSE_QUERY_NONE &&
		query->denseKind != PGTURBOHYBRID_DENSE_QUERY_VECTOR &&
		query->denseKind != PGTURBOHYBRID_DENSE_QUERY_MULTIVECTOR)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("invalid turbohybrid_query dense kind %u", query->denseKind)));

	if (query->vectorBytes < 0 || query->multivectorBytes < 0 ||
		query->tsqueryBytes < 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("invalid turbohybrid_query payload size")));

	if (((query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_VECTOR) != 0) &&
		((query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_MULTIVECTOR) != 0))
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("turbohybrid_query cannot contain both vector and multivector payloads")));

	if ((query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_DENSE) == 0 &&
		query->denseKind != PGTURBOHYBRID_DENSE_QUERY_NONE)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("turbohybrid_query dense payload is inconsistent")));

	if ((query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_DENSE) != 0 &&
		query->denseKind == PGTURBOHYBRID_DENSE_QUERY_NONE)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("turbohybrid_query dense payload is inconsistent")));

	if ((query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_VECTOR) == 0 &&
		query->vectorBytes != 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("turbohybrid_query vector payload is inconsistent")));

	if ((query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_VECTOR) != 0)
	{
		Vector	   *vector = (Vector *) ((char *) query + PgturbohybridQueryVectorOffset());
		Size		vectorBytes;

		if (query->denseKind != PGTURBOHYBRID_DENSE_QUERY_VECTOR ||
			(query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_DENSE) == 0)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("turbohybrid_query vector payload is inconsistent")));

		if (PgturbohybridQuerySizeAdd(vectorOffset, query->vectorBytes) > actual)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("truncated turbohybrid_query payload")));

		if (strict)
			PgturbohybridCheckVector(vector);
		else
			PgturbohybridCheckVectorFast(vector);
		vectorBytes = PGTURBOHYBRID_VECTOR_SIZE(vector->dim);
		if (query->vectorBytes != vectorBytes)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("turbohybrid_query vector payload is inconsistent")));
	}

	if ((query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_MULTIVECTOR) == 0 &&
		query->multivectorBytes != 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("turbohybrid_query multivector payload is inconsistent")));

	multivectorOffset = PgturbohybridQueryMultiVectorOffset(query);
	if ((query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_MULTIVECTOR) != 0)
	{
		PgturbohybridMultiVector *mv =
			(PgturbohybridMultiVector *) ((char *) query + multivectorOffset);
		Size		multivectorBytes;

		if (query->denseKind != PGTURBOHYBRID_DENSE_QUERY_MULTIVECTOR ||
			(query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_DENSE) == 0)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("turbohybrid_query multivector payload is inconsistent")));

		if (PgturbohybridQuerySizeAdd(multivectorOffset, query->multivectorBytes) > actual)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("truncated turbohybrid_query payload")));

		PgturbohybridCheckMultiVector(mv);
		multivectorBytes = PgturbohybridMultiVectorSize(mv->count, mv->dim);
		if (query->multivectorBytes != multivectorBytes ||
			query->multivectorDim != mv->dim ||
			query->multivectorCount != mv->count)
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("turbohybrid_query multivector payload is inconsistent")));
	}
	else if (query->multivectorDim != 0 || query->multivectorCount != 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("turbohybrid_query multivector payload is inconsistent")));

	if ((query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_TSQUERY) == 0 &&
		query->tsqueryBytes != 0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("turbohybrid_query tsquery payload is inconsistent")));

	tsqueryOffset = PgturbohybridQueryTsQueryOffset(query);
	if (PgturbohybridQuerySizeAdd(tsqueryOffset, query->tsqueryBytes) > actual)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("truncated turbohybrid_query payload")));
	expected = PgturbohybridQuerySizeAdd(tsqueryOffset,
										 PgturbohybridQueryAlignedSize(query->tsqueryBytes));
	if (actual != expected)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("malformed turbohybrid_query payload")));
}

Datum
pgturbohybrid_query_in(PG_FUNCTION_ARGS)
{
	char	   *input = PG_GETARG_CSTRING(0);

	(void) input;

	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("text input is not supported for turbohybrid_query"),
			 errhint("Use the turbohybrid_query(...) constructor.")));

	PG_RETURN_NULL();
}

Datum
pgturbohybrid_query_out(PG_FUNCTION_ARGS)
{
	PgturbohybridQueryHeader *query = PG_GETARG_PGTURBOHYBRID_QUERY_P(0);
	StringInfoData buf;

	PgturbohybridQueryValidate(query);

	initStringInfo(&buf);
	appendStringInfo(&buf,
					 "turbohybrid_query(fusion=%s,vector=%s",
					 PgturbohybridQueryFusionName(query->fusion),
					 (query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_VECTOR) ? "true" : "false");
	if (query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_MULTIVECTOR)
		appendStringInfoString(&buf, ",multivector=true");

	appendStringInfo(&buf,
					 ",tsquery=%s,dense_weight=%g,bm25_weight=%g,alpha=",
					 (query->flags & PGTURBOHYBRID_QUERY_FLAG_HAS_TSQUERY) ? "true" : "false",
					 query->denseWeight,
					 query->bm25Weight);
	if (query->flags & PGTURBOHYBRID_QUERY_FLAG_ALPHA_IS_SET)
		appendStringInfo(&buf, "%g", query->alpha);
	else
		appendStringInfoString(&buf, "null");

	appendStringInfo(&buf,
					 ",rrf_k=%d,dense_k=%d,bm25_k=%d,final_k=",
					 query->rrfK,
					 query->denseK,
					 query->bm25K);
	if (query->flags & PGTURBOHYBRID_QUERY_FLAG_FINAL_K_IS_SET)
		appendStringInfo(&buf, "%d", query->finalK);
	else
		appendStringInfoString(&buf, "null");

	appendStringInfo(&buf,
					 ",require_bm25_match=%s)",
					 (query->flags & PGTURBOHYBRID_QUERY_FLAG_REQUIRE_BM25_MATCH) ? "true" : "false");

	PG_RETURN_CSTRING(buf.data);
}

static uint16
PgturbohybridQueryParseFusion(text *fusion)
{
	char	   *name;
	uint16		result;

	if (fusion == NULL)
		return PGTURBOHYBRID_FUSION_RRF;

	name = text_to_cstring(fusion);
	if (pg_strcasecmp(name, "rrf") == 0)
		result = PGTURBOHYBRID_FUSION_RRF;
	else if (pg_strcasecmp(name, "weighted") == 0)
		result = PGTURBOHYBRID_FUSION_WEIGHTED;
	else if (pg_strcasecmp(name, "fast_weighted") == 0)
		result = PGTURBOHYBRID_FUSION_FAST_WEIGHTED;
	else if (pg_strcasecmp(name, "calibrated") == 0)
		result = PGTURBOHYBRID_FUSION_CALIBRATED;
	else
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid hybrid fusion mode \"%s\"", name),
				 errdetail("Valid values are \"rrf\", \"weighted\", \"fast_weighted\", and \"calibrated\".")));

	pfree(name);
	return result;
}

static void
PgturbohybridQueryCheckPositiveInt(const char *name, int32 value)
{
	if (value <= 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("%s must be greater than zero", name)));
}

static void
PgturbohybridQueryCheckNonNegativeInt(const char *name, int32 value)
{
	if (value < 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("%s must be greater than or equal to zero", name)));
}

typedef struct PgturbohybridQueryPlanCheck
{
	Node	   *expr;
	Oid			fnOid;
	bool		hasUserVisibleExpr;
	bool		hasIndexOrderByResjunkExpr;
} PgturbohybridQueryPlanCheck;

typedef struct PgturbohybridQueryPlanCheckCache
{
	PlannedStmt *plannedstmt;
	Node	   *expr;
	Oid			fnOid;
	bool		result;
} PgturbohybridQueryPlanCheckCache;

typedef struct PgturbohybridQueryConstructorCache
{
	Node	   *expr;
	uint64		gucGeneration;
	PgturbohybridQueryHeader *query;
} PgturbohybridQueryConstructorCache;

static List *
PgturbohybridQueryDistanceCallArgs(Node *expr, Oid *fnOid)
{
	if (expr == NULL)
		return NIL;

	if (IsA(expr, OpExpr))
	{
		OpExpr	   *op = (OpExpr *) expr;

		*fnOid = op->opfuncid;
		return op->args;
	}

	if (IsA(expr, FuncExpr))
	{
		FuncExpr   *func = (FuncExpr *) expr;

		*fnOid = func->funcid;
		return func->args;
	}

	return NIL;
}

static bool
PgturbohybridQueryDistanceCallMatches(Node *candidate, PgturbohybridQueryPlanCheck *check)
{
	Oid			candidateFnOid = InvalidOid;
	Oid			exprFnOid = InvalidOid;
	List	   *candidateArgs;
	List	   *exprArgs;

	if (candidate == NULL || check->expr == NULL)
		return false;

	if (equal(candidate, check->expr))
		return true;

	candidateArgs = PgturbohybridQueryDistanceCallArgs(candidate, &candidateFnOid);
	exprArgs = PgturbohybridQueryDistanceCallArgs(check->expr, &exprFnOid);

	return OidIsValid(candidateFnOid) &&
		candidateFnOid == check->fnOid &&
		OidIsValid(exprFnOid) &&
		exprFnOid == check->fnOid &&
		equal(candidateArgs, exprArgs);
}

static bool
PgturbohybridQueryExprListContains(List *exprs, PgturbohybridQueryPlanCheck *check)
{
	ListCell   *lc;

	foreach(lc, exprs)
	{
		Node	   *candidate = (Node *) lfirst(lc);

		if (PgturbohybridQueryDistanceCallMatches(candidate, check))
			return true;
	}

	return false;
}

static void
PgturbohybridQueryInspectPlan(Plan *plan, PgturbohybridQueryPlanCheck *check)
{
	ListCell   *lc;

	if (plan == NULL)
		return;

	foreach(lc, plan->targetlist)
	{
		TargetEntry *tle = (TargetEntry *) lfirst(lc);

		if (tle == NULL || !IsA(tle, TargetEntry) ||
			!PgturbohybridQueryDistanceCallMatches((Node *) tle->expr, check))
			continue;

		if (!tle->resjunk)
		{
			check->hasUserVisibleExpr = true;
			continue;
		}

		if (IsA(plan, IndexScan) &&
			PgturbohybridQueryExprListContains(((IndexScan *) plan)->indexorderbyorig,
										check))
			check->hasIndexOrderByResjunkExpr = true;
	}

	PgturbohybridQueryInspectPlan(plan->lefttree, check);
	PgturbohybridQueryInspectPlan(plan->righttree, check);
}

static bool
PgturbohybridQueryTextIndexOrderByContext(FunctionCallInfo fcinfo)
{
	PlannedStmt *plannedstmt;
	PgturbohybridQueryPlanCheckCache *cache;
	PgturbohybridQueryPlanCheck check;

	if (fcinfo->flinfo == NULL || fcinfo->flinfo->fn_expr == NULL)
		return false;

	plannedstmt = PgturbohybridCurrentPlannedStmt();
	if (plannedstmt == NULL || plannedstmt->planTree == NULL)
		return false;

	cache = (PgturbohybridQueryPlanCheckCache *) fcinfo->flinfo->fn_extra;
	if (cache != NULL &&
		cache->plannedstmt == plannedstmt &&
		cache->expr == fcinfo->flinfo->fn_expr &&
		cache->fnOid == fcinfo->flinfo->fn_oid)
		return cache->result;

	check.expr = fcinfo->flinfo->fn_expr;
	check.fnOid = fcinfo->flinfo->fn_oid;
	check.hasUserVisibleExpr = false;
	check.hasIndexOrderByResjunkExpr = false;

	PgturbohybridQueryInspectPlan(plannedstmt->planTree, &check);

	if (cache == NULL)
	{
		MemoryContext cacheCtx = fcinfo->flinfo->fn_mcxt != NULL ?
			fcinfo->flinfo->fn_mcxt : TopMemoryContext;
		MemoryContext oldCtx = MemoryContextSwitchTo(cacheCtx);

		cache = palloc0(sizeof(PgturbohybridQueryPlanCheckCache));
		fcinfo->flinfo->fn_extra = cache;
		MemoryContextSwitchTo(oldCtx);
	}

	cache->plannedstmt = plannedstmt;
	cache->expr = fcinfo->flinfo->fn_expr;
	cache->fnOid = fcinfo->flinfo->fn_oid;
	cache->result = check.hasIndexOrderByResjunkExpr && !check.hasUserVisibleExpr;

	return cache->result;
}

static void
PgturbohybridQueryRejectTextFallback(void)
{
	ereport(ERROR,
			(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
			 errmsg("hybrid text queries require a turbohybrid index scan"),
			 errdetail("The scalar hybrid distance function can only evaluate the vector payload.")));
}

static bool
PgturbohybridQueryConstructorConstArg(Node *node)
{
	if (node == NULL)
		return false;

	if (IsA(node, NamedArgExpr))
		node = (Node *) ((NamedArgExpr *) node)->arg;

	return IsA(node, Const);
}

static bool
PgturbohybridQueryConstructorCacheable(FunctionCallInfo fcinfo)
{
	FuncExpr   *func;
	ListCell   *lc;

	if (fcinfo->flinfo == NULL || fcinfo->flinfo->fn_expr == NULL ||
		!IsA(fcinfo->flinfo->fn_expr, FuncExpr))
		return false;

	func = (FuncExpr *) fcinfo->flinfo->fn_expr;
	foreach(lc, func->args)
	{
		if (!PgturbohybridQueryConstructorConstArg((Node *) lfirst(lc)))
			return false;
	}

	return true;
}

static PgturbohybridQueryHeader *
PgturbohybridQueryConstructorCached(FunctionCallInfo fcinfo)
{
	PgturbohybridQueryConstructorCache *cache;

	if (!PgturbohybridQueryConstructorCacheable(fcinfo))
		return NULL;

	cache = (PgturbohybridQueryConstructorCache *) fcinfo->flinfo->fn_extra;
	if (cache == NULL ||
		cache->expr != fcinfo->flinfo->fn_expr ||
		cache->gucGeneration != pgturbohybrid_guc_generation ||
		cache->query == NULL)
		return NULL;

	return cache->query;
}

static void
PgturbohybridQueryConstructorStoreCache(FunctionCallInfo fcinfo,
										PgturbohybridQueryHeader *query)
{
	PgturbohybridQueryConstructorCache *cache;
	MemoryContext cacheCtx;
	MemoryContext oldCtx;
	Size		size;

	if (!PgturbohybridQueryConstructorCacheable(fcinfo))
		return;

	cacheCtx = fcinfo->flinfo->fn_mcxt != NULL ?
		fcinfo->flinfo->fn_mcxt : TopMemoryContext;
	oldCtx = MemoryContextSwitchTo(cacheCtx);

	cache = palloc0(sizeof(PgturbohybridQueryConstructorCache));
	size = VARSIZE_ANY(query);
	cache->query = palloc(size);
	memcpy(cache->query, query, size);
	cache->expr = fcinfo->flinfo->fn_expr;
	cache->gucGeneration = pgturbohybrid_guc_generation;
	fcinfo->flinfo->fn_extra = cache;

	MemoryContextSwitchTo(oldCtx);
}

Datum
pgturbohybrid_query_constructor(PG_FUNCTION_ARGS)
{
	PgturbohybridQueryHeader *cached;
	struct varlena *vectorDatum = NULL;
	struct varlena *multivectorDatum = NULL;
	struct varlena *tsqueryDatum = NULL;
	int32		vectorBytes = 0;
	int32		multivectorBytes = 0;
	int32		tsqueryBytes = 0;
	int32		multivectorDim = 0;
	int32		multivectorCount = 0;
	Size		vectorSize = 0;
	Size		multivectorSize = 0;
	Size		tsquerySize = 0;
	uint16		flags = 0;
	uint16		denseKind = PGTURBOHYBRID_DENSE_QUERY_NONE;
	uint16		fusion;
	float8		denseWeight;
	float8		bm25Weight;
	float8		alpha = 0.0;
	int32		rrfK;
	int32		denseK;
	int32		bm25K;
	int32		finalK = 0;
	bool		requireBm25Match;
	Size		totalSize;
	PgturbohybridQueryHeader *result;

	cached = PgturbohybridQueryConstructorCached(fcinfo);
	if (cached != NULL)
		PG_RETURN_PGTURBOHYBRID_QUERY_P(cached);

	if (!PG_ARGISNULL(0))
	{
		vectorDatum = PG_DETOAST_DATUM_COPY(PG_GETARG_DATUM(0));
		vectorSize = VARSIZE_ANY(vectorDatum);
		if (vectorSize > PG_INT32_MAX)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("turbohybrid_query vector payload is too large")));
		vectorBytes = (int32) vectorSize;
		flags |= PGTURBOHYBRID_QUERY_FLAG_HAS_VECTOR | PGTURBOHYBRID_QUERY_FLAG_HAS_DENSE;
		denseKind = PGTURBOHYBRID_DENSE_QUERY_VECTOR;
	}

	if (!PG_ARGISNULL(1))
	{
		tsqueryDatum = PG_DETOAST_DATUM_COPY(PG_GETARG_DATUM(1));
		tsquerySize = VARSIZE_ANY(tsqueryDatum);
		if (tsquerySize > PG_INT32_MAX)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("turbohybrid_query tsquery payload is too large")));
		tsqueryBytes = (int32) tsquerySize;
		flags |= PGTURBOHYBRID_QUERY_FLAG_HAS_TSQUERY;
	}

	if (!PG_ARGISNULL(11))
	{
		PgturbohybridMultiVector *mv;

		if ((flags & PGTURBOHYBRID_QUERY_FLAG_HAS_VECTOR) != 0)
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("turbohybrid_query cannot contain both vector_query and multivector_query")));

		multivectorDatum = PG_DETOAST_DATUM_COPY(PG_GETARG_DATUM(11));
		mv = (PgturbohybridMultiVector *) multivectorDatum;
		PgturbohybridCheckMultiVector(mv);
		multivectorSize = VARSIZE_ANY(multivectorDatum);
		if (multivectorSize > PG_INT32_MAX)
			ereport(ERROR,
					(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
					 errmsg("turbohybrid_query multivector payload is too large")));
		multivectorBytes = (int32) multivectorSize;
		multivectorDim = mv->dim;
		multivectorCount = mv->count;
		flags |= PGTURBOHYBRID_QUERY_FLAG_HAS_MULTIVECTOR | PGTURBOHYBRID_QUERY_FLAG_HAS_DENSE;
		denseKind = PGTURBOHYBRID_DENSE_QUERY_MULTIVECTOR;
	}

	if ((flags & (PGTURBOHYBRID_QUERY_FLAG_HAS_DENSE | PGTURBOHYBRID_QUERY_FLAG_HAS_TSQUERY)) == 0)
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("turbohybrid_query requires a vector_query, multivector_query, or text_query")));

	fusion = PgturbohybridQueryParseFusion(PG_ARGISNULL(2) ? NULL : PG_GETARG_TEXT_PP(2));

	if (PG_ARGISNULL(3))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("dense_weight cannot be null")));
	denseWeight = PG_GETARG_FLOAT8(3);
	if (denseWeight < 0 || isnan(denseWeight) || isinf(denseWeight))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("dense_weight must be a finite non-negative value")));

	if (PG_ARGISNULL(4))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("bm25_weight cannot be null")));
	bm25Weight = PG_GETARG_FLOAT8(4);
	if (bm25Weight < 0 || isnan(bm25Weight) || isinf(bm25Weight))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("bm25_weight must be a finite non-negative value")));

	if (!PG_ARGISNULL(5))
	{
		alpha = PG_GETARG_FLOAT8(5);
		if (alpha < 0.0 || alpha > 1.0 || isnan(alpha) || isinf(alpha))
			ereport(ERROR,
					(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
					 errmsg("alpha must be between 0 and 1")));
		flags |= PGTURBOHYBRID_QUERY_FLAG_ALPHA_IS_SET;
	}

	if (PG_ARGISNULL(6))
	{
		rrfK = pgturbohybrid_default_rrf_k;
		flags |= PGTURBOHYBRID_QUERY_FLAG_RRF_K_DEFAULTED;
	}
	else
		rrfK = PG_GETARG_INT32(6);
	PgturbohybridQueryCheckPositiveInt("rrf_k", rrfK);

	if (PG_ARGISNULL(7))
	{
		denseK = pgturbohybrid_default_dense_k;
		flags |= PGTURBOHYBRID_QUERY_FLAG_DENSE_K_DEFAULTED;
	}
	else
		denseK = PG_GETARG_INT32(7);
	PgturbohybridQueryCheckNonNegativeInt("dense_k", denseK);

	if (PG_ARGISNULL(8))
	{
		bm25K = pgturbohybrid_default_bm25_k;
		flags |= PGTURBOHYBRID_QUERY_FLAG_BM25_K_DEFAULTED;
	}
	else
		bm25K = PG_GETARG_INT32(8);
	PgturbohybridQueryCheckNonNegativeInt("bm25_k", bm25K);

	if (!PG_ARGISNULL(9))
	{
		finalK = PG_GETARG_INT32(9);
		PgturbohybridQueryCheckPositiveInt("final_k", finalK);
		flags |= PGTURBOHYBRID_QUERY_FLAG_FINAL_K_IS_SET;
	}

	if (PG_ARGISNULL(10))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("require_bm25_match cannot be null")));
	requireBm25Match = PG_GETARG_BOOL(10);
	if (requireBm25Match)
		flags |= PGTURBOHYBRID_QUERY_FLAG_REQUIRE_BM25_MATCH;

	totalSize = PgturbohybridQuerySizeAdd(PgturbohybridQueryVectorOffset(),
										  PgturbohybridQueryAlignedSize(vectorBytes));
	totalSize = PgturbohybridQuerySizeAdd(totalSize,
										  PgturbohybridQueryAlignedSize(multivectorBytes));
	totalSize = PgturbohybridQuerySizeAdd(totalSize,
										  PgturbohybridQueryAlignedSize(tsqueryBytes));
	result = palloc0(totalSize);
	SET_VARSIZE(result, totalSize);
	result->version = PGTURBOHYBRID_QUERY_VERSION;
	result->flags = flags;
	result->fusion = fusion;
	result->denseWeight = denseWeight;
	result->bm25Weight = bm25Weight;
	result->alpha = alpha;
	result->rrfK = rrfK;
	result->denseK = denseK;
	result->bm25K = bm25K;
	result->finalK = finalK;
	result->denseKind = denseKind;
	result->vectorBytes = vectorBytes;
	result->multivectorBytes = multivectorBytes;
	result->tsqueryBytes = tsqueryBytes;
	result->multivectorDim = multivectorDim;
	result->multivectorCount = multivectorCount;

	if (vectorDatum != NULL)
	{
		memcpy((char *) result + PgturbohybridQueryVectorOffset(), vectorDatum, vectorBytes);
		pfree(vectorDatum);
	}
	if (multivectorDatum != NULL)
	{
		memcpy((char *) result + PgturbohybridQueryMultiVectorOffset(result),
			   multivectorDatum, multivectorBytes);
		pfree(multivectorDatum);
	}
	if (tsqueryDatum != NULL)
	{
		memcpy((char *) result + PgturbohybridQueryTsQueryOffset(result), tsqueryDatum, tsqueryBytes);
		pfree(tsqueryDatum);
	}

	PgturbohybridQueryValidate(result);
	PgturbohybridQueryConstructorStoreCache(fcinfo, result);

	PG_RETURN_PGTURBOHYBRID_QUERY_P(result);
}

Datum
pgturbohybrid_l2_distance(PG_FUNCTION_ARGS)
{
	PgturbohybridQueryHeader *query = PG_GETARG_PGTURBOHYBRID_QUERY_P(1);
	Vector	   *vectorQuery;
	Vector	   *value;

	if (PgturbohybridQueryTextIndexOrderByContext(fcinfo))
	{
		PgturbohybridQueryValidateFast(query);
		if (PgturbohybridQueryGetTsQuery(query) != NULL)
			PG_RETURN_FLOAT8(0.0);

		vectorQuery = PgturbohybridQueryGetVector(query);
		if (vectorQuery == NULL)
			PG_RETURN_FLOAT8(0.0);

		value = (Vector *) PG_DETOAST_DATUM(PG_GETARG_DATUM(0));
		PgturbohybridCheckVectorFast(value);
		PG_RETURN_FLOAT8(PgturbohybridL2Distance(value, vectorQuery));
	}

	value = PG_GETARG_PGTURBOHYBRID_VECTOR_P(0);
	PgturbohybridQueryValidate(query);
	if (PgturbohybridQueryGetTsQuery(query) != NULL)
	{
		if (!PgturbohybridQueryTextIndexOrderByContext(fcinfo))
			PgturbohybridQueryRejectTextFallback();
		PG_RETURN_FLOAT8(0.0);
	}
	vectorQuery = PgturbohybridQueryGetVector(query);

	if (vectorQuery == NULL)
		PG_RETURN_FLOAT8(0.0);

	PG_RETURN_FLOAT8(PgturbohybridL2Distance(value, vectorQuery));
}

Datum
pgturbohybrid_negative_inner_product(PG_FUNCTION_ARGS)
{
	PgturbohybridQueryHeader *query = PG_GETARG_PGTURBOHYBRID_QUERY_P(1);
	Vector	   *vectorQuery;
	Vector	   *value;

	if (PgturbohybridQueryTextIndexOrderByContext(fcinfo))
	{
		PgturbohybridQueryValidateFast(query);
		if (PgturbohybridQueryGetTsQuery(query) != NULL)
			PG_RETURN_FLOAT8(0.0);

		vectorQuery = PgturbohybridQueryGetVector(query);
		if (vectorQuery == NULL)
			PG_RETURN_FLOAT8(0.0);

		value = (Vector *) PG_DETOAST_DATUM(PG_GETARG_DATUM(0));
		PgturbohybridCheckVectorFast(value);
		PG_RETURN_FLOAT8(PgturbohybridNegativeInnerProduct(value, vectorQuery));
	}

	value = PG_GETARG_PGTURBOHYBRID_VECTOR_P(0);
	PgturbohybridQueryValidate(query);
	if (PgturbohybridQueryGetTsQuery(query) != NULL)
	{
		if (!PgturbohybridQueryTextIndexOrderByContext(fcinfo))
			PgturbohybridQueryRejectTextFallback();
		PG_RETURN_FLOAT8(0.0);
	}
	vectorQuery = PgturbohybridQueryGetVector(query);

	if (vectorQuery == NULL)
		PG_RETURN_FLOAT8(0.0);

	PG_RETURN_FLOAT8(PgturbohybridNegativeInnerProduct(value, vectorQuery));
}

Datum
pgturbohybrid_cosine_distance(PG_FUNCTION_ARGS)
{
	PgturbohybridQueryHeader *query = PG_GETARG_PGTURBOHYBRID_QUERY_P(1);
	Vector	   *vectorQuery;
	Vector	   *value;

	if (PgturbohybridQueryTextIndexOrderByContext(fcinfo))
	{
		PgturbohybridQueryValidateFast(query);
		if (PgturbohybridQueryGetTsQuery(query) != NULL)
			PG_RETURN_FLOAT8(0.0);

		vectorQuery = PgturbohybridQueryGetVector(query);
		if (vectorQuery == NULL)
			PG_RETURN_FLOAT8(0.0);

		value = (Vector *) PG_DETOAST_DATUM(PG_GETARG_DATUM(0));
		PgturbohybridCheckVectorFast(value);
		PG_RETURN_FLOAT8(PgturbohybridCosineDistance(value, vectorQuery));
	}

	value = PG_GETARG_PGTURBOHYBRID_VECTOR_P(0);
	PgturbohybridQueryValidate(query);
	if (PgturbohybridQueryGetTsQuery(query) != NULL)
	{
		if (!PgturbohybridQueryTextIndexOrderByContext(fcinfo))
			PgturbohybridQueryRejectTextFallback();
		PG_RETURN_FLOAT8(0.0);
	}
	vectorQuery = PgturbohybridQueryGetVector(query);

	if (vectorQuery == NULL)
		PG_RETURN_FLOAT8(0.0);

	PG_RETURN_FLOAT8(PgturbohybridCosineDistance(value, vectorQuery));
}

Datum
pgturbohybrid_distance(PG_FUNCTION_ARGS)
{
	return pgturbohybrid_cosine_distance(fcinfo);
}
