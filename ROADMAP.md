# Roadmap

`pgturbohybrid` is **alpha**. This roadmap states the gates between alpha, a
**narrow beta** (dense + BM25 + RRF + diagnostics as the stable contract), and
production/1.0. It is a checklist of blockers, not a schedule. See the
[beta scope](docs/beta-scope.md) for exactly what the beta contract does and
does not cover, and the [feature & maturity matrix](docs/feature-matrix.md) for
per-feature status.

## Alpha.2 (completed hardening baseline)

- [x] Standalone alpha packaging and CI matrix (PG 14–19 × pgvector v0.8.2/master, i386, Windows, macOS, valgrind)
- [x] Feature & maturity matrix; COMMENT ON maturity labels on the SQL surface
- [x] Fuzz / negative-input and on-disk metadata-corruption test baseline
- [x] Reusable release workflow + version-consistency check
- [x] Operations guide; README reduced to a landing page

## Beta.1 — narrow beta

Beta freezes only the **stable public core**; everything else stays visibly
experimental (see [docs/beta-scope.md](docs/beta-scope.md)).

- [x] Declared beta scope: dense-only, dense+BM25, RRF, support diagnostics — stable
- [x] Sparse / multivector / ColBERT / `quantized_inverted_experimental` / `pg_colbert_llama` isolated in `pgturbohybrid_experimental`
- [x] Alpha upgrade policy exercised transactionally with explicit migration guidance
- [x] Release artifacts carry SHA256 checksums (source archives in release.yml; .deb/.rpm/Windows zip in the package workflows); Docker images build with `provenance: mode=max` + SBOM
- [x] Cryptographic signing and verification of release archives and SPDX SBOM
- [x] Storage-format compatibility document ([docs/storage-format.md](docs/storage-format.md))
- [x] Long-running concurrency/recovery tests run in scheduled CI
- [ ] Multivector beta blocker resolved or fenced (see below)

## Production / 1.0

- [ ] At least one non-alpha upgrade exercised in CI; on-disk compatibility policy proven, not just documented
- [ ] Release artifacts signed and reproducibly verifiable; SBOM/provenance for Docker and packages
- [ ] Corruption/fuzz coverage beyond metadata-anchor pages; binary `RECEIVE` paths fuzzed
- [ ] Multi-backend concurrency soak tests routine; documented support/security policy with response expectations
- [ ] Experimental features cannot be mistaken for stable production paths; ≥1 external workload report with full provenance

## Tracked design / blocker notes

- **Multivector document-node build scaling** —
  [docs/dev/multivector-document-node-build-scaling.md](docs/dev/multivector-document-node-build-scaling.md).
  Exact symmetric MaxSim topology does not scale to DBpedia-size ColBERT
  document-node builds. This is a **beta blocker for the multivector feature
  only**, not for the dense+BM25 core. Mitigation already in tree: the
  `multivector_doc_build_scorer` reloption defaults to the scalable `proxy`
  scorer, and `exact_symmetric` topology is gated behind
  `turbohybrid.multivector_allow_exact_symmetric_build` (default off) with a
  `multivector_exact_symmetric_build_max_docs` cap.
- **Parallel edge construction** —
  [docs/roadmap/parallel-edge-construction.md](docs/roadmap/parallel-edge-construction.md).
