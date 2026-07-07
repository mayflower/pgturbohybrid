EXTENSION = pgturbohybrid
EXTVERSION = 0.1.1

.DEFAULT_GOAL := all

MODULE_big = pgturbohybrid
DATA = sql/pgturbohybrid--0.1.0.sql sql/pgturbohybrid--0.1.1.sql sql/pgturbohybrid--0.1.0--0.1.1.sql

PG_CONFIG ?= pg_config

OBJS = \
	src/pgturbohybrid.o \
	src/pgturbohybrid_am.o \
	src/pgturbohybrid_bm25_build.o \
	src/pgturbohybrid_bm25_query.o \
	src/pgturbohybrid_build.o \
	src/pgturbohybrid_graph.o \
	src/pgturbohybrid_graph_utils.o \
	src/pgturbohybrid_guc.o \
	src/pgturbohybrid_insert.o \
	src/pgturbohybrid_multivector.o \
	src/pgturbohybrid_quant.o \
	src/pgturbohybrid_quant_vacuum.o \
	src/pgturbohybrid_quant_cache.o \
	src/pgturbohybrid_quant_exact.o \
	src/pgturbohybrid_quant_insert.o \
	src/pgturbohybrid_quant_psquare.o \
	src/pgturbohybrid_quant_scan_cache.o \
	src/pgturbohybrid_quant_score.o \
	src/pgturbohybrid_quant_score_u8_x86.o \
	src/pgturbohybrid_quant_score_arm.o \
	src/pgturbohybrid_quant_score_signed_x86.o \
	src/pgturbohybrid_quant_storage.o \
	src/pgturbohybrid_query.o \
	src/pgturbohybrid_scan.o \
	src/pgturbohybrid_sparse_build.o \
	src/pgturbohybrid_sparse_primary.o \
	src/pgturbohybrid_sparse_query.o \
	src/pgturbohybrid_sparse_score.o \
	src/pgturbohybrid_sparse_simd_x86.o \
	src/pgturbohybrid_sparse_simd_arm.o \
	src/pgturbohybrid_diagnostics.o \
	src/pgturbohybrid_vacuum.o \
	src/pgturbohybrid_vector_compat.o

HEADERS =

$(OBJS): $(wildcard src/*.h src/*.inc)

REGRESS = extension pgturbohybrid pgturbohybrid_comments pgturbohybrid_gucs pgturbohybrid_guc_defaults pgturbohybrid_diagnostics pgturbohybrid_api_ledger pgturbohybrid_query pgturbohybrid_sparse pgturbohybrid_sparse_query pgturbohybrid_sparse_scan pgturbohybrid_sparse_fusion pgturbohybrid_sparse_quant pgturbohybrid_sparse_rerank pgturbohybrid_sparse_simd_parity pgturbohybrid_sparse_wand pgturbohybrid_sparse_cache pgturbohybrid_sparse_delta pgturbohybrid_sparse_primary pgturbohybrid_sparse_hardening pgturbohybrid_sparse_bitpacked pgturbohybrid_keymap pgturbohybrid_querysplit pgturbohybrid_multivector pgturbohybrid_multivector_many_moderate pgturbohybrid_codebook pgturbohybrid_u8split pgturbohybrid_nibble_guard pgturbohybrid_x4_safety pgturbohybrid_simd_parity pgturbohybrid_rescore pgturbohybrid_wrappers pgturbohybrid_fuzz security
REGRESS_OPTS = --inputdir=test

SIMD_BUILD ?= portable
MATH_MODE ?= strict
PGTURBOHYBRID_REQUIRE_VECTOR_HEADER ?= 0

PGVECTOR_SERVER_INCLUDE := $(shell $(PG_CONFIG) --includedir-server)
PGVECTOR_INSTALLED_INCLUDE := $(PGVECTOR_SERVER_INCLUDE)/extension/vector
PGVECTOR_INSTALLED_HEADER := $(PGVECTOR_INSTALLED_INCLUDE)/vector.h

ifneq ($(wildcard $(PGVECTOR_INSTALLED_HEADER)),)
	PG_CPPFLAGS += -I$(PGVECTOR_INSTALLED_INCLUDE)
	PGTURBOHYBRID_VECTOR_HEADER := $(PGVECTOR_INSTALLED_HEADER)
else ifneq ($(VECTOR_INCLUDE),)
	ifeq ($(wildcard $(VECTOR_INCLUDE)/vector.h),)
$(error VECTOR_INCLUDE must point to a directory containing vector.h)
	endif
	PG_CPPFLAGS += -I$(VECTOR_INCLUDE)
	PGTURBOHYBRID_VECTOR_HEADER := $(VECTOR_INCLUDE)/vector.h
endif

ifeq ($(PGTURBOHYBRID_REQUIRE_VECTOR_HEADER),1)
	ifeq ($(PGTURBOHYBRID_VECTOR_HEADER),)
$(error pgvector header vector.h not found; install pgvector headers or set VECTOR_INCLUDE=/path/to/pgvector/include/extension/vector)
	endif
	PG_CPPFLAGS += -DPGTURBOHYBRID_USE_PGVECTOR_HEADER=1
endif

ifeq ($(SIMD_BUILD),native)
	OPTFLAGS = -march=native
else ifeq ($(SIMD_BUILD),none)
	OPTFLAGS =
	PG_CFLAGS += -DPGTURBOHYBRID_DISABLE_SIMD=1
else
	OPTFLAGS =
endif

ifneq ($(filter-out portable native none,$(SIMD_BUILD)),)
$(error unsupported SIMD_BUILD=$(SIMD_BUILD); expected portable, native, or none)
endif

ifeq ($(MATH_MODE),fast)
	MATHFLAGS = -fassociative-math -fno-signed-zeros -fno-trapping-math
else ifeq ($(MATH_MODE),strict)
	MATHFLAGS =
else
$(error unsupported MATH_MODE=$(MATH_MODE); expected strict or fast)
endif

PG_CFLAGS += $(OPTFLAGS) -ftree-vectorize $(MATHFLAGS)

PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

ifeq ($(PROVE),)
	PROVE = prove
endif

PROVE_FLAGS += -I ./test/perl

prove_installcheck:
	rm -rf $(CURDIR)/tmp_check
	cd $(srcdir) && TESTDIR='$(CURDIR)' PATH="$(bindir):$$PATH" PGPORT='6$(DEF_PGPORT)' PG_REGRESS='$(top_builddir)/src/test/regress/pg_regress' $(PROVE) $(PG_PROVE_FLAGS) $(PROVE_FLAGS) $(if $(PROVE_TESTS),$(PROVE_TESTS),test/t/*.pl)

.PHONY: dist

dist:
	@test -z "$$(git status --porcelain --untracked-files=all)" || (echo "make dist requires a clean working tree" >&2; git status --short --untracked-files=all >&2; exit 1)
	@tracked_artifacts="$$(git ls-files | grep -E '(^|/)regression\.(diffs|out)$$|(^|/)\.DS_Store$$|(^|/)(benchmarks/(results|output)|results)/|(^|/)perf-smoke-results\.json$$|(^|/).*\.(o|so|bc|dll|dylib|obj|lib|exp|pyc)$$|(^|/)__pycache__/|^benchmarks/.*\.(csv|md|json)$$' | grep -v '^benchmarks/README\.md$$' | grep -v '^benchmarks/dev/README\.md$$' | grep -v '^benchmarks/dbpedia_openai3_large\.md$$' | grep -v '^benchmarks/config/.*\.json$$' || true)"; \
	if test -n "$$tracked_artifacts"; then \
		echo "make dist refuses to package generated artifacts:" >&2; \
		printf '%s\n' "$$tracked_artifacts" >&2; \
		exit 1; \
	fi
	rm -rf dist/$(EXTENSION)-$(EXTVERSION).zip dist/$(EXTENSION)-$(EXTVERSION).tar.gz
	mkdir -p dist
	git archive --mtime='1970-01-01T00:00:00Z' --format=zip --prefix=$(EXTENSION)-$(EXTVERSION)/ -o dist/$(EXTENSION)-$(EXTVERSION).zip HEAD^{tree}
	git archive --mtime='1970-01-01T00:00:00Z' --format=tar --prefix=$(EXTENSION)-$(EXTVERSION)/ HEAD^{tree} | gzip -n > dist/$(EXTENSION)-$(EXTVERSION).tar.gz
