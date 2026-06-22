#ifndef PGTURBOHYBRID_DIAGNOSTICS_H
#define PGTURBOHYBRID_DIAGNOSTICS_H

/*
 * pgturbohybrid_diagnostics.h
 *
 * Canonical key names for the STABLE subset of the turbohybrid_last_scan_stats()
 * diagnostics JSON. turbohybrid_last_scan_stats() emits hundreds of flat keys;
 * most are experimental/diagnostic-only internal counters that may change
 * between alpha releases. The keys below are the complete set documented as
 * STABLE in docs/diagnostics-schema.md: tools may depend on them, so each is
 * defined here once and referenced at its emit site in
 * pgturbohybrid_diagnostics.c so a rename is caught at compile time.
 *
 * Contract: this list, the "stable" rows of docs/diagnostics-schema.md, and the
 * pgturbohybrid_diagnostics regression test must stay in 1:1 correspondence.
 * Changing a string here changes the public diagnostics contract -- update the
 * doc and the test together.
 */

/* query / scan shape */
#define PGTURBOHYBRID_DIAG_KEY_SCAN_ORCHESTRATION	"scan_orchestration"
#define PGTURBOHYBRID_DIAG_KEY_SCORE_MODE			"score_mode"
#define PGTURBOHYBRID_DIAG_KEY_DIMENSIONS			"dimensions"
#define PGTURBOHYBRID_DIAG_KEY_QUANTIZATION_BITS	"quantization_bits"
#define PGTURBOHYBRID_DIAG_KEY_FINAL_K_EFFECTIVE	"final_k_effective"

/* dense branch */
#define PGTURBOHYBRID_DIAG_KEY_DENSE_BRANCH_USED	"dense_branch_used"
#define PGTURBOHYBRID_DIAG_KEY_DENSE_SCORER			"dense_scorer"
#define PGTURBOHYBRID_DIAG_KEY_GRAPH_STORAGE_KIND	"graph_storage_kind"
#define PGTURBOHYBRID_DIAG_KEY_DENSE_K_EFFECTIVE	"dense_k_effective"
#define PGTURBOHYBRID_DIAG_KEY_DENSE_K_DEFAULTED	"dense_k_defaulted"
#define PGTURBOHYBRID_DIAG_KEY_GRAPH_VISITED_NODES	"graph_visited_nodes"
#define PGTURBOHYBRID_DIAG_KEY_GRAPH_SCORED_CODES	"graph_scored_codes"

/* bm25 branch */
#define PGTURBOHYBRID_DIAG_KEY_BM25_BRANCH_AVAILABLE	"bm25_branch_available"
#define PGTURBOHYBRID_DIAG_KEY_BM25_BRANCH_USED			"bm25_branch_used"
#define PGTURBOHYBRID_DIAG_KEY_BM25_K_EFFECTIVE			"bm25_k_effective"
#define PGTURBOHYBRID_DIAG_KEY_BM25_K_DEFAULTED			"bm25_k_defaulted"

/* sparse branch */
#define PGTURBOHYBRID_DIAG_KEY_SPARSE_BRANCH_AVAILABLE	"sparse_branch_available"
#define PGTURBOHYBRID_DIAG_KEY_SPARSE_BRANCH_USED		"sparse_branch_used"

/* multivector branch */
#define PGTURBOHYBRID_DIAG_KEY_MULTIVECTOR_BRANCH_USED	"multivector_branch_used"

/* fusion */
#define PGTURBOHYBRID_DIAG_KEY_BRANCH_COUNT			"branch_count"
#define PGTURBOHYBRID_DIAG_KEY_BRANCH_FUSION_MODE	"branch_fusion_mode"

/* native cache */
#define PGTURBOHYBRID_DIAG_KEY_NATIVE_CACHE_POLICY	"native_cache_policy"
#define PGTURBOHYBRID_DIAG_KEY_NATIVE_CACHE_SCOPE	"native_cache_scope"
#define PGTURBOHYBRID_DIAG_KEY_NATIVE_CACHE_USED	"native_cache_used"

#endif							/* PGTURBOHYBRID_DIAGNOSTICS_H */
