#ifndef PGTURBOHYBRID_DIAGNOSTICS_H
#define PGTURBOHYBRID_DIAGNOSTICS_H

/*
 * pgturbohybrid_diagnostics.h
 *
 * Canonical key names for the *stable* subset of the turbohybrid_last_scan_stats()
 * diagnostics JSON. turbohybrid_last_scan_stats() emits hundreds of flat keys;
 * most are experimental/diagnostic-only internal counters that may change
 * between alpha releases. The keys below are the ones documented as STABLE in
 * docs/diagnostics-schema.md -- tools may depend on them, so they are defined
 * here once and referenced at their emit sites in pgturbohybrid_diagnostics.c
 * (formerly pgturbohybrid_stats.c) so a rename is caught at compile time.
 *
 * If you change a string here you are changing the public diagnostics contract:
 * update docs/diagnostics-schema.md and the pgturbohybrid_diagnostics regression
 * test as well.
 */

/* query / scan shape */
#define PGTURBOHYBRID_DIAG_KEY_SCAN_ORCHESTRATION	"scan_orchestration"
#define PGTURBOHYBRID_DIAG_KEY_SCORE_MODE			"score_mode"
#define PGTURBOHYBRID_DIAG_KEY_DIMENSIONS			"dimensions"
#define PGTURBOHYBRID_DIAG_KEY_QUANTIZATION_BITS	"quantization_bits"
#define PGTURBOHYBRID_DIAG_KEY_FINAL_K_EFFECTIVE	"final_k_effective"

/* dense branch */
#define PGTURBOHYBRID_DIAG_KEY_DENSE_SCORER			"dense_scorer"
#define PGTURBOHYBRID_DIAG_KEY_GRAPH_STORAGE_KIND	"graph_storage_kind"
#define PGTURBOHYBRID_DIAG_KEY_DENSE_K_EFFECTIVE	"dense_k_effective"
#define PGTURBOHYBRID_DIAG_KEY_GRAPH_VISITED_NODES	"graph_visited_nodes"
#define PGTURBOHYBRID_DIAG_KEY_GRAPH_SCORED_CODES	"graph_scored_codes"

#endif							/* PGTURBOHYBRID_DIAGNOSTICS_H */
