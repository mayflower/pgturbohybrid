EXTENSION = pg_colbert_llama llama_embed
EXTVERSION = 0.1.0

MODULE_big = pg_colbert_llama
DATA = sql/pg_colbert_llama--0.1.0.sql sql/llama_embed--0.1.0.sql

PG_CONFIG ?= pg_config
PG_COLBERT_LLAMA_ENGINE ?= stub
LLAMA_CPP_BACKEND_DIR ?=

OBJS = \
	src/pg_colbert_llama.o

ifeq ($(PG_COLBERT_LLAMA_ENGINE),stub)
	OBJS += src/colbert_engine_stub.o
	PG_CPPFLAGS += -DPG_COLBERT_LLAMA_ENGINE_STUB=1
else ifeq ($(PG_COLBERT_LLAMA_ENGINE),llama)
	OBJS += src/colbert_engine_llama.o
	PG_CPPFLAGS += -DPG_COLBERT_LLAMA_ENGINE_LLAMA=1
	ifneq ($(LLAMA_CPP_INCLUDE),)
		PG_CPPFLAGS += -I$(LLAMA_CPP_INCLUDE)
	endif
	ifneq ($(LLAMA_CPP_LIB),)
		SHLIB_LINK += -L$(LLAMA_CPP_LIB)
	endif
	ifneq ($(LLAMA_CPP_BACKEND_DIR),)
		PG_CPPFLAGS += -DPG_COLBERT_LLAMA_BACKEND_DIR=\"$(LLAMA_CPP_BACKEND_DIR)\"
	endif
	SHLIB_LINK += -lllama -lggml -lggml-base $(LLAMA_CPP_LDFLAGS) -lstdc++
else
$(error unsupported PG_COLBERT_LLAMA_ENGINE=$(PG_COLBERT_LLAMA_ENGINE); expected stub or llama)
endif

$(OBJS): src/*.h

REGRESS = llama_embed pg_colbert_llama
REGRESS_OPTS = --inputdir=test

PGVECTOR_SERVER_INCLUDE := $(shell $(PG_CONFIG) --includedir-server)
PGVECTOR_INSTALLED_INCLUDE := $(PGVECTOR_SERVER_INCLUDE)/extension/vector
PGVECTOR_INSTALLED_HEADER := $(PGVECTOR_INSTALLED_INCLUDE)/vector.h

ifneq ($(wildcard $(PGVECTOR_INSTALLED_HEADER)),)
	PG_CPPFLAGS += -I$(PGVECTOR_INSTALLED_INCLUDE)
else ifneq ($(VECTOR_INCLUDE),)
	ifeq ($(wildcard $(VECTOR_INCLUDE)/vector.h),)
$(error VECTOR_INCLUDE must point to a directory containing vector.h)
	endif
	PG_CPPFLAGS += -I$(VECTOR_INCLUDE)
endif

PG_CFLAGS += -ftree-vectorize

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

ifeq ($(PROVE),)
	PROVE = prove
endif

PG_TEST_PORT ?= $(if $(PGPORT),$(PGPORT),6$(DEF_PGPORT))
PROVE_FLAGS += -I ./test/perl

prove_stub_installcheck:
	cd $(srcdir) && TESTDIR='$(CURDIR)' PATH="$(bindir):$$PATH" PGPORT='$(PG_TEST_PORT)' PG_REGRESS='$(top_builddir)/src/test/regress/pg_regress' $(PROVE) $(PG_PROVE_FLAGS) $(PROVE_FLAGS) test/t/001_profile_stub.pl

prove_live_installcheck:
	if [ -n "$$PG_COLBERT_LLAMA_TEST_MODEL" ]; then \
		rm -rf $(srcdir)/tmp_check/t_002_live_pgturbohybrid_multivector_pg_colbert_llama_live_data; \
		cd $(srcdir) && TESTDIR='$(CURDIR)' PATH="$(bindir):$$PATH" PGPORT='$(PG_TEST_PORT)' PG_REGRESS='$(top_builddir)/src/test/regress/pg_regress' $(PROVE) $(PG_PROVE_FLAGS) $(PROVE_FLAGS) test/t/002_live_pgturbohybrid_multivector.pl; \
	else \
		echo "PG_COLBERT_LLAMA_TEST_MODEL is not set; skipping live pg_colbert_llama TAP tests"; \
	fi

prove_installcheck: prove_stub_installcheck prove_live_installcheck
