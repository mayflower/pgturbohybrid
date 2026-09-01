#ifndef PGTURBOHYBRID_GUC_H
#define PGTURBOHYBRID_GUC_H

#include "postgres.h"

typedef enum PgturbohybridBm25HybridBoundMode
{
	PGTURBOHYBRID_BM25_HYBRID_BOUND_OFF,
	PGTURBOHYBRID_BM25_HYBRID_BOUND_SAFE,
	PGTURBOHYBRID_BM25_HYBRID_BOUND_APPROX
} PgturbohybridBm25HybridBoundMode;

typedef enum PgturbohybridBm25HeapTSVectorRerankMode
{
	PGTURBOHYBRID_BM25_HEAP_TSVECTOR_RERANK_OFF,
	PGTURBOHYBRID_BM25_HEAP_TSVECTOR_RERANK_TOPK,
	PGTURBOHYBRID_BM25_HEAP_TSVECTOR_RERANK_BAND,
	PGTURBOHYBRID_BM25_HEAP_TSVECTOR_RERANK_AUTO
} PgturbohybridBm25HeapTSVectorRerankMode;

typedef enum PgturbohybridDbsfRobustMode
{
	PGTURBOHYBRID_DBSF_ROBUST_OFF,
	PGTURBOHYBRID_DBSF_ROBUST_MAD
}			PgturbohybridDbsfRobustMode;

/*
 * Registers all turbohybrid.* custom GUCs (and applies profile-driven dynamic
 * defaults). Called from PgturbohybridInit() after reloption registration and
 * the parallel-worker / already-defined guards.
 */
extern void PgturbohybridRegisterGUCs(void);

#endif							/* PGTURBOHYBRID_GUC_H */
