#!/usr/bin/env python3
"""Regenerate docs/api-ledger.json from the pgturbohybrid_api_ledger regression
snapshot (test/expected/pgturbohybrid_api_ledger.out) plus curated GUC/reloption
/diagnostic-key entries. Run after the snapshot changes; scripts/check-api-ledger.py
verifies the two stay consistent. No database required."""
import json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
extension_objs = {
    ext: {"type": {}, "function": {}, "operator": {}, "opclass": {}}
    for ext in ("pgturbohybrid", "pgturbohybrid_experimental")
}
for line in (ROOT / "test/expected/pgturbohybrid_api_ledger.out").read_text().splitlines():
    m = re.match(r"\s*(pgturbohybrid(?:_experimental)?)\|(type|function|operator|opclass)\|(.+)\|([a-z][a-z -]*)\s*$", line)
    if not m:
        continue
    ext, kind, name, mat = m.groups()
    extension_objs[ext][kind].setdefault(name, set()).add(mat)
def fmt(d):
    return {n: (sorted(v) if len(v) > 1 else next(iter(v))) for n, v in sorted(d.items())}
keys = sorted(re.findall(r'#define PGTURBOHYBRID_DIAG_KEY_\w+\s+"([a-z0-9_]+)"',
                         (ROOT / "src/pgturbohybrid_diagnostics.h").read_text()))
reloptions = {
  "graph_m": "stable public", "graph_ef_construction": "stable public",
  "graph_ef_search": "stable public", "graph_oversampling": "stable public",
  "native_segments": "experimental public", "quantization_bits": "stable public",
  "exact_storage": "stable public", "routing": "stable public",
  "graph_backbone": "experimental public", "entry_sidecar": "experimental public",
  "entry_sidecar_representatives": "experimental public", "entry_sidecar_strategy": "experimental public",
  "residual_rerank": "experimental public", "residual_rerank_bytes": "experimental public",
  "multivector_doc_storage": "experimental public", "multivector_doc_build_scorer": "experimental public",
}
gucs_stable = ["turbohybrid.profile", "turbohybrid.default_dense_k", "turbohybrid.default_bm25_k",
  "turbohybrid.default_sparse_k", "turbohybrid.default_rrf_k", "turbohybrid.max_union_candidates",
  "turbohybrid.enable_wand", "turbohybrid.simd", "turbohybrid.native_cache_scope",
  "turbohybrid.native_cache_max_mb", "turbohybrid.native_cache_warn_mb"]
ledger = {
 "_comment": "Machine-readable public-API ledger for pgturbohybrid. The type/function/operator/opclass sets are the source of truth checked against the pgturbohybrid_api_ledger regression snapshot by scripts/check-api-ledger.py; maturity labels mirror the COMMENT ON tags and docs/feature-matrix.md. Alpha: even 'stable public' does not promise on-disk compatibility across pre-1.0 tags.",
 "sql_version": "0.2.0",
 "extensions": {
   ext: {
     "types": fmt(objs["type"]), "operators": fmt(objs["operator"]),
     "opclasses": fmt(objs["opclass"]), "functions": fmt(objs["function"]),
   }
   for ext, objs in extension_objs.items()
 },
 "reloptions": reloptions, "gucs_stable_public": gucs_stable,
 "gucs_note": "139 turbohybrid.* GUCs exist; only the stable-public tuning surface is enumerated here. pg_settings (name LIKE 'turbohybrid.%') is authoritative and docs/architecture.md classifies the rest as experimental public or developer/benchmark.",
 "diagnostics_stable_keys": keys,
 "upgrade_impact": {
   "types": "Adding/removing/renaming requires an extension SQL upgrade script.",
   "functions": "Adding/removing/changing a signature requires an extension SQL upgrade script.",
   "operators": "As for functions.",
   "opclasses": "As for functions; opclass changes typically also require REINDEX.",
   "reloptions": "Build-time; changing a value for an existing index requires REINDEX.",
   "gucs": "Query-time GUCs need no REINDEX; build-time GUCs require REINDEX (see docs/operations.md).",
   "diagnostics_stable_keys": "Removing or retyping a stable key is a breaking diagnostics-contract change.",
 },
}
(ROOT / "docs/api-ledger.json").write_text(json.dumps(ledger, indent=2) + "\n")
print("regenerated docs/api-ledger.json")
