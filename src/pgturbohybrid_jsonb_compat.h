#ifndef PGTURBOHYBRID_JSONB_COMPAT_H
#define PGTURBOHYBRID_JSONB_COMPAT_H

#include "postgres.h"

#include <string.h>

#include "utils/jsonb.h"

#if PG_VERSION_NUM >= 190000
typedef JsonbInState PgturbohybridJsonbState;
#else
typedef JsonbParseState *PgturbohybridJsonbState;
#endif

static inline void
PgturbohybridJsonbStateInit(PgturbohybridJsonbState *state)
{
#if PG_VERSION_NUM >= 190000
	memset(state, 0, sizeof(*state));
#else
	*state = NULL;
#endif
}

static inline void
PgturbohybridJsonbPush(PgturbohybridJsonbState *state,
					   JsonbIteratorToken token, JsonbValue *value)
{
#if PG_VERSION_NUM >= 190000
	pushJsonbValue(state, token, value);
#else
	(void) pushJsonbValue(state, token, value);
#endif
}

static inline void
PgturbohybridJsonbBeginObject(PgturbohybridJsonbState *state)
{
	PgturbohybridJsonbPush(state, WJB_BEGIN_OBJECT, NULL);
}

/*
 * Close a nested object.  Unlike PgturbohybridJsonbEndObject this does NOT
 * materialise a Jsonb: on PG19 pushJsonbValue() only populates state->result
 * when the *top-level* container closes, so reading it after a nested close
 * yields NULL and JsonbValueToJsonb(NULL) segfaults.  Use this for every
 * non-top-level object; use PgturbohybridJsonbEndObject only for the outermost.
 */
static inline void
PgturbohybridJsonbCloseObject(PgturbohybridJsonbState *state)
{
	PgturbohybridJsonbPush(state, WJB_END_OBJECT, NULL);
}

static inline Jsonb *
PgturbohybridJsonbEndObject(PgturbohybridJsonbState *state)
{
#if PG_VERSION_NUM >= 190000
	PgturbohybridJsonbPush(state, WJB_END_OBJECT, NULL);
	return JsonbValueToJsonb(state->result);
#else
	return JsonbValueToJsonb(pushJsonbValue(state, WJB_END_OBJECT, NULL));
#endif
}

#endif
