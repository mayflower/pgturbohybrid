# Maintainers

`pgturbohybrid` is maintained by Mayflower.

## Current Maintainers

- Mayflower maintainers

## Responsibilities

Maintainers are responsible for:

- reviewing pull requests for correctness, tests, compatibility, and release
  impact
- preserving pgvector attribution and the standalone extension boundary
- keeping release notes, compatibility docs, and benchmark claims accurate
- deciding whether a change belongs in the current alpha, a patch release, a
  later minor release, or a future design discussion
- responding to security reports according to [SECURITY.md](SECURITY.md)
- enforcing [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)

## Maintainer Decisions

For the alpha period, maintainership decisions are made by Mayflower. Changes to
maintainer status, release authority, project scope, or governance should be
documented in this file or in a future `GOVERNANCE.md`.

Routine technical decisions may be made in pull request review. Larger decisions
should consider:

- correctness and crash-safety risk
- PostgreSQL and pgvector compatibility
- on-disk format and `REINDEX` impact
- benchmark evidence and reproducibility
- release timing and user-facing documentation

## Release Criteria

Public releases should have:

- a clear `CHANGELOG.md` entry
- release notes under `docs/release-notes/`
- passing SQL regression and TAP coverage in CI
- compatibility notes for supported PostgreSQL and pgvector versions
- benchmark artifacts attached outside the git tree for performance claims
