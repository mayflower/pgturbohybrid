#ifndef PGTURBOHYBRID_QUANT_INTERNAL_H
#define PGTURBOHYBRID_QUANT_INTERNAL_H

/*
 * pgturbohybrid_quant_internal.h
 *
 * Small, hot inner-loop helpers (frontier + result min/max heaps, elapsed-time
 * accumulators, candidate/offer helpers) shared between the base graph scan in
 * pgturbohybrid_quant.c and the extracted multivector scan.  Kept as
 * static inline in a header so both translation units inline them -- extracting
 * the multivector code must not turn these into cross-TU extern calls on the
 * hot query path.  All types/macros come from pgturbohybrid.h / _quant.h; these
 * carry no file-scope state.
 */
#include "postgres.h"

#include "portability/instr_time.h"

#include "pgturbohybrid.h"
#include "pgturbohybrid_quant.h"

static inline int
PgturbohybridGraphFrontierCompare(const void *a, const void *b)
{
	const PgturbohybridGraphFrontierItem *ia = (const PgturbohybridGraphFrontierItem *) a;
	const PgturbohybridGraphFrontierItem *ib = (const PgturbohybridGraphFrontierItem *) b;

	if (ia->distance < ib->distance)
		return -1;
	if (ia->distance > ib->distance)
		return 1;
	return (ia->nodeId > ib->nodeId) - (ia->nodeId < ib->nodeId);
}

static inline bool
PgturbohybridGraphFrontierLess(PgturbohybridGraphFrontierItem a, PgturbohybridGraphFrontierItem b)
{
	if (a.distance != b.distance)
		return a.distance < b.distance;
	return a.nodeId < b.nodeId;
}

static inline bool
PgturbohybridGraphFrontierGreater(PgturbohybridGraphFrontierItem a, PgturbohybridGraphFrontierItem b)
{
	if (a.distance != b.distance)
		return a.distance > b.distance;
	return a.nodeId > b.nodeId;
}

static inline void
PgturbohybridGraphFrontierSwap(PgturbohybridGraphFrontierItem *a, PgturbohybridGraphFrontierItem *b)
{
	PgturbohybridGraphFrontierItem tmp = *a;

	*a = *b;
	*b = tmp;
}

static inline void
PgturbohybridGraphFrontierHeapSiftUp(PgturbohybridGraphFrontierItem *heap, int idx, bool minHeap)
{
	while (idx > 0)
	{
		int			parent = (idx - 1) / 2;
		bool		before = minHeap ?
			PgturbohybridGraphFrontierLess(heap[idx], heap[parent]) :
			PgturbohybridGraphFrontierGreater(heap[idx], heap[parent]);

		if (!before)
			break;

		PgturbohybridGraphFrontierSwap(&heap[idx], &heap[parent]);
		idx = parent;
	}
}

static inline void
PgturbohybridGraphFrontierHeapSiftDown(PgturbohybridGraphFrontierItem *heap, int count, int idx,
							bool minHeap)
{
	for (;;)
	{
		int			left = idx * 2 + 1;
		int			right = left + 1;
		int			best = idx;

		if (left < count)
		{
			bool		before = minHeap ?
				PgturbohybridGraphFrontierLess(heap[left], heap[best]) :
				PgturbohybridGraphFrontierGreater(heap[left], heap[best]);

			if (before)
				best = left;
		}

		if (right < count)
		{
			bool		before = minHeap ?
				PgturbohybridGraphFrontierLess(heap[right], heap[best]) :
				PgturbohybridGraphFrontierGreater(heap[right], heap[best]);

			if (before)
				best = right;
		}

		if (best == idx)
			break;

		PgturbohybridGraphFrontierSwap(&heap[idx], &heap[best]);
		idx = best;
	}
}

static inline void
PgturbohybridGraphFrontierHeapPush(PgturbohybridGraphFrontierItem *heap, int *count,
						PgturbohybridGraphFrontierItem item, bool minHeap)
{
	heap[*count] = item;
	(*count)++;
	PgturbohybridGraphFrontierHeapSiftUp(heap, *count - 1, minHeap);
}

static inline int
PgturbohybridGraphInitialFrontierCapacity(uint32 nodeCount, int searchEf, int entryCount,
							   int maxNeighbors)
{
	int			capacity;

	if (nodeCount == 0)
		return 0;

	capacity = Max(8, searchEf + entryCount + maxNeighbors);
	capacity = Max(capacity, (searchEf * 2) + entryCount);
	if ((uint32) capacity > nodeCount)
		capacity = (int) nodeCount;

	return capacity;
}

static inline Size
PgturbohybridGraphArrayAllocSize(Size elemSize, Size count)
{
	if (elemSize != 0 && count > MaxAllocSize / elemSize)
		ereport(ERROR,
				(errcode(ERRCODE_PROGRAM_LIMIT_EXCEEDED),
				 errmsg("pgturbohybrid graph array is too large")));
	return elemSize * count;
}

static inline void
PgturbohybridGraphFrontierHeapPushGrowing(PgturbohybridGraphFrontierItem **heap, int *count,
							   int *capacity, int maxCapacity,
							   PgturbohybridGraphFrontierItem item, bool minHeap)
{
	if (*count >= *capacity)
	{
		int			newCapacity;

		if (*capacity >= maxCapacity)
			elog(ERROR, "pgturbohybrid graph frontier capacity exceeded");

		if (*capacity > PG_INT32_MAX / 2)
			newCapacity = maxCapacity;
		else
			newCapacity = Max(8, *capacity + *capacity);
		if (newCapacity < *capacity || newCapacity > maxCapacity)
			newCapacity = maxCapacity;

		*heap = repalloc(*heap,
						 PgturbohybridGraphArrayAllocSize(sizeof(PgturbohybridGraphFrontierItem),
														  newCapacity));
		*capacity = newCapacity;
	}

	PgturbohybridGraphFrontierHeapPush(*heap, count, item, minHeap);
}

static inline PgturbohybridGraphFrontierItem
PgturbohybridGraphFrontierHeapPop(PgturbohybridGraphFrontierItem *heap, int *count, bool minHeap)
{
	PgturbohybridGraphFrontierItem item = heap[0];

	(*count)--;
	if (*count > 0)
	{
		heap[0] = heap[*count];
		PgturbohybridGraphFrontierHeapSiftDown(heap, *count, 0, minHeap);
	}

	return item;
}

static inline void
PgturbohybridGraphFrontierHeapReplaceRoot(PgturbohybridGraphFrontierItem *heap, int count,
							   PgturbohybridGraphFrontierItem item, bool minHeap)
{
	heap[0] = item;
	PgturbohybridGraphFrontierHeapSiftDown(heap, count, 0, minHeap);
}

static inline bool
PgturbohybridGraphOfferNearest(PgturbohybridGraphFrontierItem *heap, int capacity, int *count,
					uint32 nodeId, double distance)
{
	PgturbohybridGraphFrontierItem item;

	if (capacity <= 0)
		return false;

	item.nodeId = nodeId;
	item.distance = distance;

	if (*count < capacity)
	{
		PgturbohybridGraphFrontierHeapPush(heap, count, item, false);
		return true;
	}

	if (PgturbohybridGraphFrontierLess(item, heap[0]))
	{
		PgturbohybridGraphFrontierHeapReplaceRoot(heap, *count, item, false);
		return true;
	}

	return false;
}

static inline int64
PgturbohybridGraphElapsedUs(instr_time start)
{
	instr_time	duration;

	INSTR_TIME_SET_CURRENT(duration);
	INSTR_TIME_SUBTRACT(duration, start);
	return (int64) INSTR_TIME_GET_MICROSEC(duration);
}

static inline void
PgturbohybridGraphAddElapsedUs(int64 *target, instr_time start)
{
	instr_time	duration;

	INSTR_TIME_SET_CURRENT(duration);
	INSTR_TIME_SUBTRACT(duration, start);
	*target += (int64) INSTR_TIME_GET_MICROSEC(duration);
}

static inline void
PgturbohybridGraphAddElapsedUint64(uint64 *target, instr_time start)
{
	instr_time	duration;

	INSTR_TIME_SET_CURRENT(duration);
	INSTR_TIME_SUBTRACT(duration, start);
	*target += (uint64) INSTR_TIME_GET_MICROSEC(duration);
}

static inline bool
PgturbohybridGraphResultLess(PgturbohybridGraphResult a, PgturbohybridGraphResult b)
{
	if (a.distance != b.distance)
		return a.distance < b.distance;
	return a.nodeId < b.nodeId;
}

static inline bool
PgturbohybridGraphResultGreater(PgturbohybridGraphResult a, PgturbohybridGraphResult b)
{
	if (a.distance != b.distance)
		return a.distance > b.distance;
	return a.nodeId > b.nodeId;
}

static inline void
PgturbohybridGraphResultSwap(PgturbohybridGraphResult *a, PgturbohybridGraphResult *b)
{
	PgturbohybridGraphResult tmp = *a;

	*a = *b;
	*b = tmp;
}

static inline void
PgturbohybridGraphResultHeapSiftUp(PgturbohybridGraphResult *heap, int idx)
{
	while (idx > 0)
	{
		int			parent = (idx - 1) / 2;

		if (!PgturbohybridGraphResultGreater(heap[idx], heap[parent]))
			break;

		PgturbohybridGraphResultSwap(&heap[idx], &heap[parent]);
		idx = parent;
	}
}

static inline void
PgturbohybridGraphResultHeapSiftDown(PgturbohybridGraphResult *heap, int count, int idx)
{
	for (;;)
	{
		int			left = idx * 2 + 1;
		int			right = left + 1;
		int			best = idx;

		if (left < count && PgturbohybridGraphResultGreater(heap[left], heap[best]))
			best = left;
		if (right < count && PgturbohybridGraphResultGreater(heap[right], heap[best]))
			best = right;
		if (best == idx)
			break;

		PgturbohybridGraphResultSwap(&heap[idx], &heap[best]);
		idx = best;
	}
}

static inline void
PgturbohybridGraphResultHeapPush(PgturbohybridGraphResult *heap, int *count, PgturbohybridGraphResult item)
{
	heap[*count] = item;
	(*count)++;
	PgturbohybridGraphResultHeapSiftUp(heap, *count - 1);
}

static inline void
PgturbohybridGraphResultHeapReplaceRoot(PgturbohybridGraphResult *heap, int count,
							 PgturbohybridGraphResult item)
{
	heap[0] = item;
	PgturbohybridGraphResultHeapSiftDown(heap, count, 0);
}

static inline void
PgturbohybridGraphOfferCandidate(PgturbohybridGraphScanOpaque so, PgturbohybridGraphResult *results, int target,
					  int *count, uint32 nodeId, ItemPointer heaptid,
					  double distance, bool exactScored)
{
	PgturbohybridGraphResult item;

	if (target <= 0)
		return;

	item.nodeId = nodeId;
	item.heaptid = *heaptid;
	item.distance = distance;
	item.exactScored = exactScored;

	if (*count < target)
	{
		PgturbohybridGraphResultHeapPush(results, count, item);
		if (exactScored)
			so->graphRescoreCount++;
		return;
	}

	if (PgturbohybridGraphResultLess(item, results[0]))
	{
		if (results[0].exactScored)
			so->graphRescoreCount--;
		PgturbohybridGraphResultHeapReplaceRoot(results, *count, item);
		if (exactScored)
			so->graphRescoreCount++;
	}
}

static inline int
PgturbohybridGraphScanAdjSlot(PgturbohybridGraphMetaPageData *meta, uint32 nodeId, int level)
{
	return PgturbohybridGraphAdjSlot(meta, nodeId, level);
}

static inline bool
PgturbohybridGraphEntryAlreadySelected(PgturbohybridGraphFrontierItem *entries, int entryCount,
							uint32 nodeId)
{
	for (int i = 0; i < entryCount; i++)
	{
		if (entries[i].nodeId == nodeId)
			return true;
	}

	return false;
}

static inline void
PgturbohybridGraphOfferDistanceEntry(PgturbohybridGraphFrontierItem *entries, int *entryCount,
						  PgturbohybridGraphFrontierItem entry)
{
	int			worst = 0;

	for (int i = 0; i < *entryCount; i++)
	{
		if (entries[i].nodeId == entry.nodeId)
		{
			entries[i].distance = Min(entries[i].distance, entry.distance);
			return;
		}
	}

	if (*entryCount < PGTURBOHYBRID_GRAPH_MAX_ENTRY_POINTS)
	{
		entries[*entryCount] = entry;
		(*entryCount)++;
		return;
	}

	for (int i = 1; i < *entryCount; i++)
	{
		if (entries[i].distance > entries[worst].distance)
			worst = i;
	}

	if (entry.distance < entries[worst].distance)
		entries[worst] = entry;
}

#endif							/* PGTURBOHYBRID_QUANT_INTERNAL_H */
