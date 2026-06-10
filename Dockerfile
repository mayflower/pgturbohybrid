# pgturbohybrid: PostgreSQL + pgvector + pgturbohybrid in one image.
#
# Build:
#   docker build --build-arg PG_MAJOR=17 --build-arg PGVECTOR_REF=v0.8.2 -t pgturbohybrid .
#
# The extension is built with SIMD_BUILD=portable (runtime CPU dispatch), so a
# single amd64 image runs across CPU generations; arm64 uses the NEON kernels.
ARG PG_MAJOR=17

FROM postgres:${PG_MAJOR} AS build
ARG PG_MAJOR
ARG PGVECTOR_REF=v0.8.2
ENV PG_CONFIG=/usr/lib/postgresql/${PG_MAJOR}/bin/pg_config

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates git \
        "postgresql-server-dev-${PG_MAJOR}" \
    && rm -rf /var/lib/apt/lists/*

# pgvector (the hard runtime dependency). OPTFLAGS="" keeps it portable.
RUN git clone --depth 1 --branch "${PGVECTOR_REF}" https://github.com/pgvector/pgvector.git /tmp/pgvector \
    && make -C /tmp/pgvector OPTFLAGS="" \
    && make -C /tmp/pgvector install DESTDIR=/out

# pgturbohybrid, portable SIMD build.
COPY . /tmp/pgturbohybrid
RUN make -C /tmp/pgturbohybrid clean \
    && make -C /tmp/pgturbohybrid SIMD_BUILD=portable \
    && make -C /tmp/pgturbohybrid install DESTDIR=/out

FROM postgres:${PG_MAJOR}
ARG PG_MAJOR
LABEL org.opencontainers.image.source="https://github.com/agentxagi/pgturbohybrid" \
      org.opencontainers.image.description="PostgreSQL with pgvector and pgturbohybrid preinstalled" \
      org.opencontainers.image.licenses="PostgreSQL"

# Stage the built extensions into the runtime image at their install paths.
COPY --from=build /out/ /

# Create the extensions on first cluster initialization.
COPY docker/initdb/10-extensions.sql /docker-entrypoint-initdb.d/10-extensions.sql
