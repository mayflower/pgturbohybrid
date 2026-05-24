# Security Notes

`pgturbohybrid` is a PostgreSQL C extension. Install it only from source you
trust and only into PostgreSQL installations where loading C extensions is an
accepted operational risk.

## Reporting Vulnerabilities

Report security issues privately to the maintainers before opening a public
issue. If this code is published under the Mayflower organization, use the
repository's GitHub security advisory workflow or email
security@mayflower.de.

## Extension SQL

- The install script creates only extension objects.
- It does not create roles, databases, tablespaces, policies, or security
  labels.
- It does not use `SECURITY DEFINER`.
- It does not mark functions `LEAKPROOF`.
- It uses `MODULE_PATHNAME` for C functions.
- It references pgvector's `vector` type through the extension dependency
  search path.
- PostgreSQL built-in types and catalogs are schema-qualified where ambiguity
  would matter.

## Runtime Checks

The extension validates that:

- PostgreSQL is 14 or newer.
- The `vector` extension is installed.
- pgvector is version 0.8.2 or newer.
- vector values have a valid pgvector-compatible varlena layout.
- vector dimensions are valid and match where required.
- NaN and infinite vector values are rejected.

Index options are registered through PostgreSQL reloptions and use PostgreSQL's
normal validation paths.
