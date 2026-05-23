#ifndef TQGRAPH_PSQUARE_H
#define TQGRAPH_PSQUARE_H

#include "postgres.h"

#define PGTURBOHYBRID_PSQUARE_N 5

typedef struct TqPSquareState
{
	int64		count;
	double		target_q;
	double		heights[PGTURBOHYBRID_PSQUARE_N];
	double		n_positions[PGTURBOHYBRID_PSQUARE_N];
}			TqPSquareState;

void		TqPSquareInit(TqPSquareState *p, double quantile);
void		TqPSquarePush(TqPSquareState *p, double x);
double		TqPSquareEstimate(const TqPSquareState *p);

#endif
