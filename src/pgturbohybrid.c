#include "postgres.h"

#include <string.h>

#include "access/amapi.h"
#include "access/relation.h"
#include "access/relscan.h"
#include "catalog/pg_type_d.h"
#include "commands/explain.h"
#include "executor/executor.h"
#include "fmgr.h"
#include "pgturbohybrid.h"
#include "pgturbohybrid_am.h"
#include "lib/stringinfo.h"
#include "miscadmin.h"
#include "nodes/execnodes.h"
#include "nodes/nodeFuncs.h"
#include "nodes/plannodes.h"
#include "nodes/primnodes.h"
#include "pgstat.h"
#include "storage/lmgr.h"
#include "utils/fmgroids.h"
#include "tcop/tcopprot.h"
#include "pgturbohybrid_quant.h"
#include "utils/json.h"
#include "utils/lsyscache.h"
#include "utils/rel.h"

#if PG_VERSION_NUM >= 180000
PG_MODULE_MAGIC_EXT(.name = "pgturbohybrid",.version = "0.1.0");
#else
PG_MODULE_MAGIC;
#endif

typedef struct PgturbohybridGraphExecWrapperState
{
	PlanState   *planstate;
	ExecProcNodeMtd original_exec_proc_node;
	LimitState  *limitstate;
}			PgturbohybridGraphExecWrapperState;

static ExecutorStart_hook_type prev_ExecutorStart_hook = NULL;
static ExecutorRun_hook_type prev_ExecutorRun_hook = NULL;
static ExecutorEnd_hook_type prev_ExecutorEnd_hook = NULL;
static List *tqgraph_exec_wrapper_states = NIL;
static int64 tqgraph_active_limit_tuple_target = -1;
static double tqgraph_active_estimated_filter_selectivity = -1.0;
static bool tqgraph_active_payload_filter_valid = false;
static AttrNumber tqgraph_active_payload_filter_attno = InvalidAttrNumber;
static int32 tqgraph_active_payload_filter_value = 0;

static void PgturbohybridExecutorHooksInit(void);
static void PgturbohybridExecutorStartHook(QueryDesc *queryDesc, int eflags);
#if PG_VERSION_NUM >= 180000
static void PgturbohybridExecutorRunHook(QueryDesc *queryDesc,
										 ScanDirection direction,
										 uint64 count);
#else
static void PgturbohybridExecutorRunHook(QueryDesc *queryDesc,
										 ScanDirection direction,
										 uint64 count,
										 bool execute_once);
#endif
static void PgturbohybridExecutorEndHook(QueryDesc *queryDesc);
static void PgturbohybridGraphExecutorStart(QueryDesc *queryDesc, int eflags);
static void PgturbohybridGraphExecutorEnd(QueryDesc *queryDesc);
static TupleTableSlot *PgturbohybridGraphExecIndexScanWithController(PlanState *planstate);
static void PgturbohybridGraphWrapControlledIndexScans(PlanState *planstate, LimitState *limitstate);
static PgturbohybridGraphExecWrapperState *PgturbohybridGraphFindWrapperState(PlanState *planstate);
static bool PgturbohybridGraphIsIndexScanState(PlanState *planstate);
static bool PgturbohybridGraphIndexScanUsespgturbohybrid(IndexScanState *indexstate);
static int64 PgturbohybridGraphGetLimitTupleTarget(LimitState *limitstate);
static double PgturbohybridGraphEstimateFilterSelectivity(IndexScanState *indexstate);
static bool PgturbohybridGraphExtractPayloadInt4Filter(IndexScanState *indexstate,
										 AttrNumber *heap_attno,
										 int32 *value);
static bool PgturbohybridGraphExtractPayloadInt4FilterExpr(Node *node,
											 AttrNumber *heap_attno,
											 int32 *value);

PGDLLEXPORT void _PG_init(void);
void
_PG_init(void)
{
	PgturbohybridGraphInit();
	PgturbohybridInit();
	PgturbohybridExecutorHooksInit();
}

int64
PgturbohybridGraphGetActiveLimitTupleTarget(void)
{
	return tqgraph_active_limit_tuple_target;
}

double
PgturbohybridGraphGetActiveEstimatedFilterSelectivity(void)
{
	return tqgraph_active_estimated_filter_selectivity;
}

bool
PgturbohybridGraphGetActivePayloadInt4Filter(AttrNumber *heap_attno, int32 *value)
{
	if (!tqgraph_active_payload_filter_valid)
		return false;

	if (heap_attno != NULL)
		*heap_attno = tqgraph_active_payload_filter_attno;
	if (value != NULL)
		*value = tqgraph_active_payload_filter_value;

	return true;
}

void
PgturbohybridGraphControlInit(void)
{
	/*
	 * Executor hooks are installed once by PgturbohybridExecutorHooksInit()
	 * after graph and AM initialization. This entry point remains because the
	 * graph initializer owns other graph setup and older code paths call it.
	 */
}

static void
PgturbohybridExecutorHooksInit(void)
{
	prev_ExecutorStart_hook = ExecutorStart_hook;
	ExecutorStart_hook = PgturbohybridExecutorStartHook;

	prev_ExecutorRun_hook = ExecutorRun_hook;
	ExecutorRun_hook = PgturbohybridExecutorRunHook;

	prev_ExecutorEnd_hook = ExecutorEnd_hook;
	ExecutorEnd_hook = PgturbohybridExecutorEndHook;
}

static void
PgturbohybridExecutorStartHook(QueryDesc *queryDesc, int eflags)
{
	bool		am_started = false;

	PG_TRY();
	{
		PgturbohybridAmExecutorStart(queryDesc, eflags);
		am_started = true;

		if (prev_ExecutorStart_hook)
			prev_ExecutorStart_hook(queryDesc, eflags);
		else
			standard_ExecutorStart(queryDesc, eflags);

		PgturbohybridGraphExecutorStart(queryDesc, eflags);
	}
	PG_CATCH();
	{
		tqgraph_exec_wrapper_states = NIL;
		if (am_started)
			PgturbohybridAmExecutorEnd(queryDesc);
		else
			PgturbohybridAmExecutorAbort();
		PG_RE_THROW();
	}
	PG_END_TRY();
}

static void
PgturbohybridExecutorRunHook(QueryDesc *queryDesc, ScanDirection direction,
#if PG_VERSION_NUM >= 180000
							 uint64 count)
#else
							 uint64 count, bool execute_once)
#endif
{
	PG_TRY();
	{
		if (prev_ExecutorRun_hook)
#if PG_VERSION_NUM >= 180000
			prev_ExecutorRun_hook(queryDesc, direction, count);
#else
			prev_ExecutorRun_hook(queryDesc, direction, count, execute_once);
#endif
		else
#if PG_VERSION_NUM >= 180000
			standard_ExecutorRun(queryDesc, direction, count);
#else
			standard_ExecutorRun(queryDesc, direction, count, execute_once);
#endif
	}
	PG_CATCH();
	{
		tqgraph_exec_wrapper_states = NIL;
		PgturbohybridAmExecutorAbort();
		PG_RE_THROW();
	}
	PG_END_TRY();
}

static void
PgturbohybridGraphExecutorStart(QueryDesc *queryDesc, int eflags)
{
	(void) eflags;

	tqgraph_exec_wrapper_states = NIL;
	PgturbohybridGraphWrapControlledIndexScans(queryDesc->planstate, NULL);
}

static void
PgturbohybridExecutorEndHook(QueryDesc *queryDesc)
{
	PgturbohybridGraphExecutorEnd(queryDesc);
	PgturbohybridAmExecutorEnd(queryDesc);

	if (prev_ExecutorEnd_hook)
		prev_ExecutorEnd_hook(queryDesc);
	else
		standard_ExecutorEnd(queryDesc);
}

static void
PgturbohybridGraphExecutorEnd(QueryDesc *queryDesc)
{
	(void) queryDesc;

	tqgraph_exec_wrapper_states = NIL;
}

static TupleTableSlot *
PgturbohybridGraphExecIndexScanWithController(PlanState *planstate)
{
	PgturbohybridGraphExecWrapperState *wrapper_state = PgturbohybridGraphFindWrapperState(planstate);
	IndexScanState *indexstate = castNode(IndexScanState, planstate);
	TupleTableSlot *slot;
	IndexScanDesc scan;
	PgturbohybridGraphScanOpaque so;
	int64		prev_limit_tuple_target = tqgraph_active_limit_tuple_target;
	double		prev_estimated_filter_selectivity = tqgraph_active_estimated_filter_selectivity;
	bool		prev_payload_filter_valid = tqgraph_active_payload_filter_valid;
	AttrNumber	prev_payload_filter_attno = tqgraph_active_payload_filter_attno;
	int32		prev_payload_filter_value = tqgraph_active_payload_filter_value;
	int64		tuple_target = PgturbohybridGraphGetLimitTupleTarget(wrapper_state->limitstate);
	double		estimated_filter_selectivity = PgturbohybridGraphEstimateFilterSelectivity(indexstate);
	AttrNumber	payload_filter_attno = InvalidAttrNumber;
	int32		payload_filter_value = 0;
	bool		payload_filter_valid =
		PgturbohybridGraphExtractPayloadInt4Filter(indexstate, &payload_filter_attno,
									 &payload_filter_value);

	tqgraph_active_limit_tuple_target = tuple_target;
	tqgraph_active_estimated_filter_selectivity = estimated_filter_selectivity;
	tqgraph_active_payload_filter_valid = payload_filter_valid;
	tqgraph_active_payload_filter_attno = payload_filter_attno;
	tqgraph_active_payload_filter_value = payload_filter_value;

	PG_TRY();
	{
		slot = wrapper_state->original_exec_proc_node(planstate);
	}
	PG_CATCH();
	{
		tqgraph_active_limit_tuple_target = prev_limit_tuple_target;
		tqgraph_active_estimated_filter_selectivity = prev_estimated_filter_selectivity;
		tqgraph_active_payload_filter_valid = prev_payload_filter_valid;
		tqgraph_active_payload_filter_attno = prev_payload_filter_attno;
		tqgraph_active_payload_filter_value = prev_payload_filter_value;
		PG_RE_THROW();
	}
	PG_END_TRY();
	tqgraph_active_limit_tuple_target = prev_limit_tuple_target;
	tqgraph_active_estimated_filter_selectivity = prev_estimated_filter_selectivity;
	tqgraph_active_payload_filter_valid = prev_payload_filter_valid;
	tqgraph_active_payload_filter_attno = prev_payload_filter_attno;
	tqgraph_active_payload_filter_value = prev_payload_filter_value;

	scan = indexstate->iss_ScanDesc;
	if (scan == NULL || scan->opaque == NULL)
		return slot;

	so = (PgturbohybridGraphScanOpaque) scan->opaque;
	PgturbohybridGraphSeedScanContext(so, tuple_target, estimated_filter_selectivity);

	if (!TupIsNull(slot))
		so->returnedRows++;

	return slot;
}

static void
PgturbohybridGraphWrapControlledIndexScans(PlanState *planstate, LimitState *limitstate)
{
	if (planstate == NULL)
		return;

	if (IsA(planstate, LimitState))
	{
		PgturbohybridGraphWrapControlledIndexScans(outerPlanState(planstate),
									 castNode(LimitState, planstate));
		return;
	}

	if (PgturbohybridGraphIsIndexScanState(planstate) &&
		PgturbohybridGraphIndexScanUsespgturbohybrid(castNode(IndexScanState, planstate)))
	{
		PgturbohybridGraphExecWrapperState *wrapper_state =
			palloc(sizeof(PgturbohybridGraphExecWrapperState));

		wrapper_state->planstate = planstate;
		wrapper_state->original_exec_proc_node = planstate->ExecProcNodeReal;
		wrapper_state->limitstate = limitstate;
		tqgraph_exec_wrapper_states = lappend(tqgraph_exec_wrapper_states, wrapper_state);
		ExecSetExecProcNode(planstate, PgturbohybridGraphExecIndexScanWithController);
	}

	if (IsA(planstate, AppendState))
	{
		AppendState *appendstate = castNode(AppendState, planstate);

		for (int i = 0; i < appendstate->as_nplans; i++)
			PgturbohybridGraphWrapControlledIndexScans(appendstate->appendplans[i], limitstate);
		return;
	}

	if (IsA(planstate, MergeAppendState))
	{
		MergeAppendState *mergeappendstate = castNode(MergeAppendState, planstate);

		for (int i = 0; i < mergeappendstate->ms_nplans; i++)
			PgturbohybridGraphWrapControlledIndexScans(mergeappendstate->mergeplans[i], limitstate);
		return;
	}

	if (IsA(planstate, ResultState))
	{
		ResultState *resultstate = castNode(ResultState, planstate);

		PgturbohybridGraphWrapControlledIndexScans(outerPlanState(planstate),
									 resultstate->ps.qual == NULL ? limitstate : NULL);
		return;
	}

	if (IsA(planstate, SubqueryScanState))
	{
		SubqueryScanState *subquerystate = castNode(SubqueryScanState, planstate);

		PgturbohybridGraphWrapControlledIndexScans(subquerystate->subplan,
									 subquerystate->ss.ps.qual == NULL ? limitstate : NULL);
		return;
	}

	if (IsA(planstate, GatherState) || IsA(planstate, GatherMergeState))
	{
		PgturbohybridGraphWrapControlledIndexScans(outerPlanState(planstate), limitstate);
		return;
	}

	PgturbohybridGraphWrapControlledIndexScans(outerPlanState(planstate), NULL);
	PgturbohybridGraphWrapControlledIndexScans(innerPlanState(planstate), NULL);
}

static PgturbohybridGraphExecWrapperState *
PgturbohybridGraphFindWrapperState(PlanState *planstate)
{
	ListCell   *lc;

	foreach(lc, tqgraph_exec_wrapper_states)
	{
		PgturbohybridGraphExecWrapperState *wrapper_state = lfirst(lc);

		if (wrapper_state->planstate == planstate)
			return wrapper_state;
	}

	elog(ERROR, "missing pgturbohybrid graph scan wrapper state");
	return NULL;
}

static bool
PgturbohybridGraphIsIndexScanState(PlanState *planstate)
{
	return planstate != NULL && IsA(planstate, IndexScanState);
}

static bool
PgturbohybridGraphIndexScanUsespgturbohybrid(IndexScanState *indexstate)
{
	return indexstate->iss_RelationDesc != NULL &&
		indexstate->iss_RelationDesc->rd_indam != NULL &&
		indexstate->iss_RelationDesc->rd_indam->amgettuple == pgturbohybridamgettuple;
}

static pg_noinline int64
PgturbohybridGraphGetLimitTupleTarget(LimitState *limitstate)
{
	int64		tuple_target;

	if (limitstate == NULL || limitstate->noCount || limitstate->count < 0)
		return -1;

	tuple_target = limitstate->count;

	if (limitstate->offset > 0)
		tuple_target += limitstate->offset;

	return tuple_target;
}

static bool
PgturbohybridGraphExtractPayloadInt4FilterExpr(Node *node, AttrNumber *heap_attno, int32 *value)
{
	OpExpr	   *op;
	Node	   *left;
	Node	   *right;
	Var		   *var = NULL;
	Const	   *constant = NULL;
	Oid			opfuncid;

	if (node == NULL || !IsA(node, OpExpr))
		return false;

	op = castNode(OpExpr, node);
	if (list_length(op->args) != 2)
		return false;

	opfuncid = op->opfuncid;
	if (!OidIsValid(opfuncid))
		opfuncid = get_opcode(op->opno);
	if (opfuncid != F_INT4EQ)
		return false;

	left = linitial(op->args);
	right = lsecond(op->args);

	if (IsA(left, Var) && IsA(right, Const))
	{
		var = castNode(Var, left);
		constant = castNode(Const, right);
	}
	else if (IsA(left, Const) && IsA(right, Var))
	{
		var = castNode(Var, right);
		constant = castNode(Const, left);
	}
	else
		return false;

	if (var->varattno <= 0 || var->vartype != INT4OID ||
		constant->consttype != INT4OID || constant->constisnull)
		return false;

	*heap_attno = var->varattno;
	*value = DatumGetInt32(constant->constvalue);
	return true;
}

static bool
PgturbohybridGraphExtractPayloadInt4Filter(IndexScanState *indexstate, AttrNumber *heap_attno,
							 int32 *value)
{
	List	   *quals;
	ListCell   *lc;

	if (indexstate == NULL || indexstate->ss.ps.plan == NULL)
		return false;

	quals = indexstate->ss.ps.plan->qual;
	foreach(lc, quals)
	{
		if (PgturbohybridGraphExtractPayloadInt4FilterExpr((Node *) lfirst(lc),
											 heap_attno, value))
			return true;
	}

	return false;
}

static double
PgturbohybridGraphEstimateFilterSelectivity(IndexScanState *indexstate)
{
	Relation	heap_relation;
	double		reltuples;
	double		estimated_rows;
	Index		scanrelid;
	bool		close_heap_relation = false;
	RangeTblEntry *rte;

	if (indexstate == NULL)
		return -1.0;

	heap_relation = indexstate->ss.ss_currentRelation;
	if (heap_relation == NULL &&
		indexstate->ss.ps.state != NULL &&
		indexstate->ss.ps.plan != NULL)
	{
		scanrelid = ((Scan *) indexstate->ss.ps.plan)->scanrelid;
		if (scanrelid > 0)
		{
			rte = exec_rt_fetch(scanrelid, indexstate->ss.ps.state);
			if (rte->relid != InvalidOid)
			{
				heap_relation = relation_open(rte->relid, AccessShareLock);
				close_heap_relation = true;
			}
		}
	}

	if (heap_relation == NULL || heap_relation->rd_rel == NULL)
		return -1.0;

	reltuples = heap_relation->rd_rel->reltuples;
	if (!(reltuples > 0))
		return -1.0;

	estimated_rows = indexstate->ss.ps.plan->plan_rows;
	if (estimated_rows < 0)
	{
		if (close_heap_relation)
			relation_close(heap_relation, AccessShareLock);
		return -1.0;
	}

	if (close_heap_relation)
		relation_close(heap_relation, AccessShareLock);

	return Max(Min(estimated_rows / reltuples, 1.0), 0.0);
}

static double
PgturbohybridGraphClampEstimatedFilterSelectivity(double estimated_selectivity)
{
	double		min_selectivity;

	if (!(estimated_selectivity > 0))
		return 1.0;

	min_selectivity = 1.0 / (double) Max((int64) pgturbohybrid_max_scan_tuples, (int64) 1);

	return Max(Min(estimated_selectivity, 1.0), min_selectivity);
}

void
PgturbohybridGraphSeedScanContext(PgturbohybridGraphScanOpaque so, int64 tuple_target,
					double estimated_filter_selectivity)
{
	if (!so->hasTupleTargetRows && tuple_target >= 0)
	{
		so->hasTupleTargetRows = true;
		so->tupleTargetRows = tuple_target;
	}

	if (!so->hasEstimatedFilterSelectivity && estimated_filter_selectivity >= 0)
	{
		so->estimatedFilterSelectivity =
			PgturbohybridGraphClampEstimatedFilterSelectivity(estimated_filter_selectivity);
		so->hasEstimatedFilterSelectivity = true;
	}

	if (!so->hasInitialEffectiveEfSearch)
	{
		so->initialEffectiveEfSearch = so->efSearch;
		so->hasInitialEffectiveEfSearch = true;
	}
}
