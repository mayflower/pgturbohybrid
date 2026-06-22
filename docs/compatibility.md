# Compatibility Matrix

This file helps you check whether your PostgreSQL and pgvector versions are in
the current public compatibility range for `pgturbohybrid`.

`pgturbohybrid` depends on the pgvector SQL extension and its `vector` type. It
does not link against private pgvector C symbols.

## Supported Range

| Dependency | Supported versions | Notes |
| --- | --- | --- |
| PostgreSQL | 14, 15, 16, 17, 18, 19 | CI builds and runs regression tests on every version. PostgreSQL 19 is tested against pgvector `master` (it has no tagged pgvector 0.8.2 build). |
| pgvector | 0.8.2 and newer compatible 0.8.x releases | 0.8.2 is the minimum runtime version. |
| pgvector master | tested | CI installs upstream `pgvector/pgvector` master as a forward-compatibility signal. |

## Build-Time Model

The build can use pgvector's installed `vector.h` only through
`src/pgturbohybrid_vector_compat.h` when `PGTURBOHYBRID_REQUIRE_VECTOR_HEADER=1`
is set. By default, the extension uses its private compatibility struct and
static layout assertions for the pgvector `vector` varlena layout.

Most users do not need this option. It is mainly for developers who want the
build to prove that pgvector headers are available.

Use:

```sh
make PGTURBOHYBRID_REQUIRE_VECTOR_HEADER=1
```

to force the build to fail if pgvector headers are not installed.

## Platform Build Profiles

On Linux and macOS (GCC/clang) the full `Makefile` compiles the hand-vectorized
SIMD kernels and runs the complete regression suite.

The Windows build (`Makefile.win`, MSVC/`nmake`) is a **reduced-coverage
profile**, and this is intentional, not drift:

- **Sources.** MSVC cannot compile the architecture-specific SIMD kernels
  (`pgturbohybrid_quant_score_u8_x86`, `pgturbohybrid_quant_score_signed_x86`,
  `pgturbohybrid_quant_score_arm`, `pgturbohybrid_sparse_simd_x86`,
  `pgturbohybrid_sparse_simd_arm`) because they rely on GCC/clang
  `__attribute__((target(...)))` and `__builtin_cpu_supports`. The Windows build
  omits those objects and uses the portable scalar fallbacks in
  `pgturbohybrid_quant_score` / `pgturbohybrid_sparse_score`. Results are
  identical; only the kernel that computes them differs.
- **Tests.** Windows runs the SIMD-independent, output-deterministic regression
  tests (extension install, catalog comments, GUC defaults, diagnostic key
  types, query-wrapper equivalence, input fuzzing, security). Score- and
  ranking-sensitive suites (sparse, multivector, SIMD parity, rescore, and the
  dense kernel tests) stay Unix-only because the MSVC scalar build can differ in
  the last ULP of scored output, which would produce spurious regression diffs
  rather than real failures.

`scripts/check-build-file-parity.py` (run by `scripts/release-check.sh`)
enforces that the only source-object difference between the two Makefiles is the
documented SIMD allowlist above, so accidental drift is caught when a new `.c`
file is added.

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
