# Security Code Audit Signoff

This file records the final branch evidence for the `v0.1.0-alpha.2`
release-hardening pass.

## Branch State

- Branch: `security-hardening-alpha2`
- Latest pushed code evidence commit:
  `3003d0b41c3540390d98fc0d00394c524de60684`
- Final tag commit: verify with `git rev-parse HEAD` immediately before
  tagging. This tracked signoff file cannot contain its own commit hash without
  changing that hash.
- Date: 2026-05-26

## Fixed Release Blockers

- Parallel-safety markings for mutable scan diagnostics and hybrid distance
  functions were restricted.
- USERSET GUCs now have bounded candidate, RRF, union, and cache limits.
- `turbohybrid_query(...)` cache entries are invalidated when relevant GUCs
  change.
- Query varlena validation rejects trailing bytes and overflowed sizes.
- Graph scan page locks now release locally on errors.
- Executor hooks are installed through one hook manager; planned-statement and
  graph-wrapper state are cleaned up on executor end, start failure, and abort.
- BM25 query/cache allocation paths now use overflow and metadata caps.
- BM25 queries with more than 64 terms fail explicitly.
- Fast math is opt-in; strict math is the default build mode.
- Insert path now uses the actual graph node ID returned by in-place insertion.

## Evidence

- Local release gate: `scripts/release-check.sh` passed.
- Local strict no-SIMD build: `SIMD_BUILD=none MATH_MODE=strict` passed with
  assertion and warning-as-error flags.
- Local SQL regression: `installcheck` passed 4 tests.
- Local TAP: `prove_installcheck` reported `NOTESTS` because PostgreSQL TAP
  Perl modules were unavailable in the local PGXS installation.
- GitHub build workflow: passed on the pushed branch.
- GitHub build run: https://github.com/mayflower/pgturbohybrid/actions/runs/26472545673
- CI coverage included Linux PostgreSQL 14-19, pgvector v0.8.2 and master,
  Linux i386, macOS, Windows, perf smoke, and valgrind.

## Source Archives

`make dist` passed from a clean branch tree. The archives are generated from
`git archive`, so ignored generated benchmark outputs and local build products
are not included.

- `cf2a3d3a078c671769140002004e8a32ea885c483dbfde8ced3531f61e04121c  dist/pgturbohybrid-0.1.0.tar.gz`
- `707a07c306c05afbc4775c46157ce931451bc6ce1b461bc14ae1460dd58af8d2  dist/pgturbohybrid-0.1.0.zip`

Recompute these after the final signoff commit with:

```sh
sha256sum dist/pgturbohybrid-0.1.0.tar.gz
sha256sum dist/pgturbohybrid-0.1.0.zip
```

## Remaining Alpha Risks

- Full corruption fuzzing for every graph and BM25 page type remains follow-up
  work. This pass hardened the release-facing allocation and metadata paths.
- The new `hardening` workflow could not be manually dispatched from this branch
  because GitHub only exposes workflow dispatch for workflows already present on
  the default branch. Run it after the workflow file lands on the default branch.

## Decision

This branch is suitable to merge for the public alpha release-preparation work.
Before tagging, run the hardening workflow once it is available on the default
branch, then rerun the release checklist from the final release commit.
