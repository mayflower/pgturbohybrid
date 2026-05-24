# Compatibility Matrix

`pgturbohybrid` depends on the pgvector SQL extension and its `vector` type.
It does not link against private pgvector C symbols.

## Supported Range

| Dependency | Supported versions | Notes |
| --- | --- | --- |
| PostgreSQL | 14, 15, 16, 17, 18 | CI builds and runs regression tests. PostgreSQL 19 is added when available in the setup action. |
| pgvector | 0.8.2 and newer compatible 0.8.x releases | 0.8.2 is the minimum runtime version. |
| pgvector master | tested | CI installs upstream `pgvector/pgvector` master as a forward-compatibility signal. |

## Build-Time Model

The build can use pgvector's installed `vector.h` only through
`src/pgturbohybrid_vector_compat.h` when `PGTURBOHYBRID_REQUIRE_VECTOR_HEADER=1`
is set. By default, the extension uses its private compatibility struct and
static layout assertions for the pgvector `vector` varlena layout.

Use:

```sh
make PGTURBOHYBRID_REQUIRE_VECTOR_HEADER=1
```

to force the build to fail if pgvector headers are not installed.

## Runtime Checks

At install time, `sql/pgturbohybrid--0.1.0.sql` checks:

- PostgreSQL server version is 14 or newer.
- `vector` is installed.
- pgvector version is 0.8.2 or newer.

## Extension Dependency Search Path

The SQL install script intentionally uses the unqualified `vector` type in
function signatures and operator class declarations. During `CREATE EXTENSION`,
PostgreSQL makes prerequisite extension schemas available to the script for
extensions listed in `pgturbohybrid.control` with `requires = 'vector'`.

The regression suite covers pgvector installed outside `public` and
`pgturbohybrid` installed in a separate schema. This keeps the SQL portable
without hard-coding either extension schema.

At runtime, vector access goes through the compatibility layer, which validates
dimensions, varlena size, reserved header fields, and finite float payloads.
