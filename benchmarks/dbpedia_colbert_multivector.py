#!/usr/bin/env python3
"""Run DBpedia ColBERT multivector generation, insert, retrieval, and quality benchmarks."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.26",
#   "psycopg[binary]>=3.2",
#   "pyarrow>=16",
# ]
# ///

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shlex
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

try:
    import psycopg
except ImportError:  # pragma: no cover - exercised by --help without bench deps
    psycopg = None  # type: ignore[assignment]


DEFAULT_METHODS = "pgturbohybrid_colbert_multivector_query_only"
QUERY_ONLY_METHOD = "pgturbohybrid_colbert_multivector_query_only"
RRF_METHOD = "pgturbohybrid_colbert_multivector_rrf"
EXACT_SCAN_METHOD = "pgturbohybrid_colbert_multivector_exact_scan"
DEFAULT_MODEL_PATH = ".nix-dev/models/colbert-15m/sauerkraut-modern.gguf"
DEFAULT_COLBERT_MODEL_NAME = "VAGOsolutions/SauerkrautLM-Multi-ColBERT-15m"
COLBERT_MODEL_DIMENSIONS = {
    "colbert-ir/colbertv2.0": 128,
    "answerdotai/answerai-colbert-small-v1": 96,
    "jinaai/jina-colbert-v2": 128,
    "jinaai/jina-colbert-v2-96": 96,
    "jinaai/jina-colbert-v2-64": 64,
    "lightonai/gte-moderncolbert-v1": 128,
    "chadboyda/reason-moderncolbert": 128,
    "vagosolutions/sauerkrautlm-multi-colbert-15m": 128,
    "johannhartmann/sauerkrautlm-multi-colbert-15m-gguf": 128,
    "vidore/colpali-v1.2": 128,
    "colpali-like-visual": 128,
}
DOCUMENT_NODE_STORAGE_CHOICES = ("f32", "f16", "sq8")
DOCUMENT_NODE_STORAGE_CACHE_CHOICES = ("auto", "resident", "paged")
DOCUMENT_NODE_TOKEN_POOLING_CHOICES = ("off", "kmeans", "greedy_cosine")
DOCUMENT_NODE_CENTROIDS_CHOICES = ("off", "kmeans")
MULTIVECTOR_PROXY_ENCODER_CHOICES = (
    "normalized_mean",
    "first_token",
    "centroid_mean",
    "mean_pool",
    "max_pool",
    "random_projection_fde",
    "learned_projection_placeholder",
    "learned_projection_v1",
)
MULTIVECTOR_SPARSE_CANDIDATE_SOURCE_CHOICES = ("off", "bm25", "learned_sparse")
MULTIVECTOR_DOC_BUILD_SCORER_CHOICES = ("proxy", "exact_symmetric")
DOCUMENT_NODE_BASELINE_MODES = (
    "token_nodes",
    "exact_token_scan",
    "plain_fallback",
    "exact_doc_scan",
    "document_nodes",
    "proxy_vector",
    "centroid_lite",
    "quantized_inverted_experimental",
)
HYBRID_EVALUATION_MODES = (
    "exact_scan",
    "document_nodes",
    "document_nodes_bm25_admission",
    "document_nodes_bm25_rrf",
    "document_nodes_bm25_dbsf",
    "proxy_vector_document_nodes",
    "learned_sparse_exact_maxsim",
)
OPTIONAL_HYBRID_EVALUATION_MODES = (
    "quantized_inverted_experimental",
)
SUPPORTED_HYBRID_EVALUATION_MODES = (
    *HYBRID_EVALUATION_MODES,
    *OPTIONAL_HYBRID_EVALUATION_MODES,
)
DEFAULT_PARALLEL_SAFETY_MAX_SERIAL_P95_MS = 5000.0
DEFAULT_PARALLEL_SAFETY_MAX_DOCS_SCORED = 5000
DEFAULT_PARALLEL_SAFETY_MAX_EXACT_RERANK_DOCS = 1000
DEFAULT_PARALLEL_SAFETY_MAX_EXACT_PAIRS = 5_000_000
DEFAULT_PARALLEL_SAFETY_MAX_SIDECAR_BYTES = 512 * 1024 * 1024
DEFAULT_ADMISSION_BUDGET_SWEEP = "100,200,400,800,1600,3200,6400,10000"
DOCUMENT_NODE_SERVING_GRID_BUDGETS = (50, 100, 200, 400, 800)
DOCUMENT_NODE_SERVING_GRID_BUDGET_SWEEP = ",".join(
    str(budget) for budget in DOCUMENT_NODE_SERVING_GRID_BUDGETS
)
DOCUMENT_NODE_SERVING_GRID_EF = (50, 100, 200)
DOCUMENT_NODE_SERVING_GRID_OVERSAMPLING = (1, 2)
DOCUMENT_NODE_SERVING_GRID_SMOKE_BUDGETS = (200, 800)
DOCUMENT_NODE_SERVING_GRID_SMOKE_EF = (50, 100)
DOCUMENT_NODE_SERVING_GRID_SMOKE_OVERSAMPLING = (1,)
DOCUMENT_NODE_SERVING_GRID_CENTROID_LITE_CAPS = (16, 32, 64)
DOCUMENT_NODE_SERVING_GRID_CENTROID_LITE_CAP_SWEEP = ",".join(
    str(cap) for cap in DOCUMENT_NODE_SERVING_GRID_CENTROID_LITE_CAPS
)
DOCUMENT_NODE_SERVING_GRID_ENTRY_SAMPLE_COUNTS = (32, 128)
DOCUMENT_NODE_SERVING_GRID_ENTRY_SAMPLE_SWEEP = ",".join(
    str(count) for count in DOCUMENT_NODE_SERVING_GRID_ENTRY_SAMPLE_COUNTS
)
DOCUMENT_NODE_SERVING_GRID_ENTRY_SIDECAR_REPRESENTATIVES = (128, 256)
DOCUMENT_NODE_SERVING_GRID_ENTRY_SIDECAR_SWEEP = ",".join(
    str(count) for count in DOCUMENT_NODE_SERVING_GRID_ENTRY_SIDECAR_REPRESENTATIVES
)
DOCUMENT_NODE_SERVING_GRID_PROXY_ADMISSION_FOCUS_PROFILES = (
    "proxy_normalized_mean_f16",
    "proxy_max_pool_f16",
    "centroid_mean_f16",
    "proxy_normalized_mean_f16_entry_sample_032",
    "proxy_normalized_mean_f16_entry_sample_128",
    "proxy_max_pool_f16_entry_sample_032",
    "proxy_max_pool_f16_entry_sample_128",
    "centroid_mean_f16_entry_sample_032",
    "centroid_mean_f16_entry_sample_128",
    "proxy_normalized_mean_f16_entry_sidecar_128",
    "proxy_normalized_mean_f16_entry_sidecar_256",
    "proxy_max_pool_f16_entry_sidecar_128",
    "proxy_max_pool_f16_entry_sidecar_256",
    "centroid_mean_f16_entry_sidecar_128",
    "centroid_mean_f16_entry_sidecar_256",
)
DOCUMENT_NODE_SERVING_GRID_PROXY_ADMISSION_FOCUS_EF = (100, 200, 400)
DOCUMENT_NODE_SERVING_GRID_PROXY_ADMISSION_FOCUS_OVERSAMPLING = (1, 2)
DOCUMENT_NODE_SERVING_GRID_CENTROID_LITE_FOCUS_PROFILES = (
	"centroid_lite_f16",
	"centroid_lite_f16_cap_016",
	"centroid_lite_f16_cap_016_prune_safe_upper_bound",
	"centroid_lite_f16_cap_032",
	"centroid_lite_f16_cap_032_prune_safe_upper_bound",
	"centroid_lite_f16_cap_064",
	"centroid_lite_f16_cap_064_prune_safe_upper_bound",
	"centroid_lite_f16_prune_safe_upper_bound",
	"centroid_lite_f16_pool_050",
	"centroid_mean_f16",
)
DOCUMENT_NODE_SERVING_GRID_TOKEN_POOLING_FOCUS_PROFILES = (
    "proxy_normalized_mean_f16",
    "proxy_normalized_mean_f16_pool_075",
    "proxy_normalized_mean_f16_pool_050",
    "proxy_normalized_mean_f16_pool_033",
    "centroid_mean_f16",
    "centroid_mean_f16_pool_075",
    "centroid_mean_f16_pool_050",
    "centroid_mean_f16_pool_033",
)
DOCUMENT_NODE_SERVING_GRID_SMOKE_PROFILES = (
    "proxy_normalized_mean_f16",
    "centroid_mean_f16",
    "centroid_lite_f16",
)
DOCUMENT_NODE_SERVING_GRID_SMOKE_MAX_QUERIES = 25
DOCUMENT_NODE_SERVING_LATENCY_DEFAULT_PROFILE = "proxy_normalized_mean_f16"
DOCUMENT_NODE_SERVING_LATENCY_DEFAULT_CANDIDATE_K = 800
SERVING_STATS_FIELD_GROUPS: dict[str, tuple[str, ...]] = {
    "core": (
        "multivector_candidate_source",
        "multivector_branch_plan",
        "multivector_plain_fallback_used",
        "multivector_plain_fallback_reason",
        "multivector_plain_fallback_docs_scored",
        "multivector_doc_graph_warning",
        "multivector_doc_graph_nodes",
        "multivector_doc_graph_docs_scored",
        "multivector_doc_graph_edges_visited",
        "multivector_doc_graph_candidates",
        "multivector_doc_candidates",
        "multivector_unique_docs",
        "multivector_doc_graph_exact_rerank_docs",
        "multivector_doc_graph_search_ef",
        "multivector_doc_graph_oversampling",
        "multivector_proxy_candidate_target",
        "multivector_proxy_candidates_returned",
        "multivector_exact_rerank_k_effective",
        "effective_result_target",
        "graph_effective_result_target",
        "final_k_effective",
        "multivector_doc_graph_entry_sample_configured",
        "multivector_doc_graph_entry_sample_effective",
        "multivector_doc_graph_entry_sample_scored",
        "multivector_exact_rerank_docs",
        "multivector_exact_rerank_pairs",
        "multivector_exact_kernel",
        "graph_entry_sample_configured",
        "graph_entry_sample_effective",
        "graph_entry_sample_scored",
        "graph_entry_sidecar_count",
        "graph_entry_sidecar_scored",
        "graph_entry_sidecar_selected",
        "graph_entry_sidecar_representatives_configured",
        "graph_entry_sidecar_strategy",
        "graph_entry_sidecar_us",
    ),
    "proxy": (
        "proxy_encoder_kind",
        "proxy_graph_nodes_visited",
        "proxy_graph_edges_visited",
        "proxy_graph_candidates_seen",
        "proxy_candidates_returned",
        "proxy_candidate_limit_effective",
        "proxy_candidate_limit_source",
        "proxy_vector_scores_computed",
        "proxy_vector_score_time_us",
        "proxy_candidates",
        "proxy_lazy_sidecar_vectors",
        "proxy_top1_admission",
        "proxy_exact_rerank_docs",
        "proxy_full_sidecar_vectors_loaded",
        "proxy_full_sidecar_bytes_touched",
        "proxy_full_sidecar_pages_read",
        "proxy_full_sidecar_load_time_us",
        "proxy_full_sidecar_reconstruct_time_us",
        "proxy_exact_rerank_heap_fetches",
        "proxy_exact_rerank_sidecar_fetches",
        "proxy_exact_rerank_bytes_touched",
        "proxy_exact_rerank_time_us",
        "sidecar_cache_build_this_query",
        "sidecar_cache_build_bytes",
        "sidecar_cache_build_pages_read",
        "sidecar_cache_build_time_us",
        "sidecar_query_bytes_touched",
        "sidecar_query_pages_read",
        "sidecar_query_vectors_loaded",
        "sidecar_query_load_time_us",
        "sidecar_query_time_us",
        "proxy_vector_uses_full_sidecar_for_graph",
        "proxy_vector_near_exhaustive_sidecar_touch",
        "proxy_vector_sidecar_touch_reason",
        "multivector_centroid_count",
        "multivector_centroid_prerank_docs",
        "multivector_full_maxsim_rerank_docs",
    ),
    "centroid_lite": (
        "centroid_lists_visited",
        "centroid_docs_touched",
        "centroid_pruned_docs",
        "centroid_postings_touched",
        "centroid_postings_skipped",
        "centroid_posting_limit_per_token",
        "centroid_posting_cap_strategy",
        "centroid_candidates",
        "centroid_bitset_prefilter_enabled",
        "centroid_bitset_lists_used",
        "centroid_bitset_docs_set",
        "centroid_bitset_docs_after_threshold",
        "centroid_bitset_prefilter_time_us",
        "centroid_bitset_memory_bytes",
        "centroid_upper_bound_enabled",
        "centroid_upper_bound_docs_checked",
        "centroid_upper_bound_docs_pruned",
        "centroid_upper_bound_prune_time_us",
        "centroid_upper_bound_unsafe_fallbacks",
        "centroid_candidates_before_bound",
        "centroid_candidates_after_bound",
        "learned_projection_loaded",
        "learned_projection_dim",
        "learned_projection_weight_bytes",
        "learned_projection_model",
        "learned_projection_checksum",
        "learned_projection_query_encode_us",
    ),
    "storage_cache": (
        "multivector_doc_sidecar_cache_mode",
        "multivector_doc_storage_cache_requested",
        "multivector_doc_storage_cache_effective",
        "multivector_doc_sidecar_pages_read",
        "multivector_doc_sidecar_cache_hits",
        "multivector_doc_sidecar_cache_misses",
        "multivector_doc_sidecar_bytes_touched",
        "multivector_doc_sidecar_vectors_loaded",
        "multivector_doc_sidecar_docmap_pages_read",
        "multivector_doc_sidecar_docmap_bytes_touched",
        "multivector_doc_sidecar_resident_vectors_loaded",
        "multivector_doc_sidecar_resident_bytes_loaded",
        "multivector_doc_sidecar_vector_chunk_ref_bytes_touched",
        "multivector_doc_sidecar_paged_vector_pages_read",
        "multivector_doc_sidecar_paged_vector_bytes_touched",
        "native_cache_used",
        "native_cache_built_this_scan",
        "native_cache_reused",
        "native_cache_bytes",
        "native_cache_exact_bytes",
        "native_cache_mode",
        "native_cache_scope",
    ),
    "phase_timing": (
        "multivector_candidate_source_time_us",
        "multivector_document_graph_traversal_time_us",
        "multivector_proxy_candidate_time_us",
        "multivector_proxy_graph_traversal_time_us",
        "multivector_proxy_scoring_time_us",
        "multivector_centroid_lite_time_us",
        "multivector_quantized_inverted_time_us",
        "multivector_sidecar_load_time_us",
        "multivector_sidecar_page_read_time_us",
        "multivector_sidecar_vector_reconstruct_time_us",
        "multivector_exact_heap_fetch_time_us",
        "multivector_exact_maxsim_rerank_time_us",
        "multivector_final_sort_time_us",
        "multivector_candidate_source_us",
        "multivector_doc_graph_traversal_us",
        "multivector_proxy_scoring_us",
        "multivector_centroid_lite_posting_us",
        "multivector_quantized_inverted_posting_us",
        "multivector_sidecar_load_us",
        "multivector_heap_visibility_us",
        "multivector_exact_heap_fetch_us",
        "multivector_exact_rerank_us",
        "multivector_final_sort_us",
        "phase_total_time_us",
        "phase_prepare_time_us",
        "document_graph_traversal_time_us",
        "document_graph_entry_time_us",
        "document_graph_base_time_us",
        "document_graph_batch_time_us",
        "heap_time_us",
        "fill_time_us",
        "rescore_time_us",
        "final_sort_time_us",
        "entry_sidecar_time_us",
        "payload_entry_seed_time_us",
        "residual_rerank_time_us",
        "exact_heap_fetch_time_us",
        "exact_maxsim_rerank_time_us",
        "local_expansion_time_us",
        "scan_lock_wait_time_us",
        "sidecar_load_time_us",
        "sidecar_page_read_time_us",
        "sidecar_vector_reconstruct_time_us",
        "proxy_candidate_time_us",
        "proxy_graph_traversal_time_us",
        "proxy_scoring_time_us",
        "centroid_lite_time_us",
        "quantized_inverted_time_us",
        "phase_timing_source",
        "phase_timing_missing",
        "phase_timing_known_total_us",
        "phase_timing_unattributed_sql_ms",
    ),
    "pooling": (
        "multivector_tokens_original",
        "multivector_tokens_pooled",
        "multivector_token_pooling_ratio",
    ),
    "sparse_bm25_rescue": (
        "multivector_bm25_injection_enabled",
        "multivector_bm25_injection_candidates",
        "multivector_bm25_injection_candidate_limit",
        "multivector_bm25_injection_pool_size",
        "multivector_bm25_injection_limit_reason",
        "multivector_bm25_injection_retained",
        "multivector_bm25_injection_exact_reranked",
        "learned_sparse_candidates",
        "learned_sparse_retained_for_maxsim",
        "learned_sparse_branch_latency_us",
    ),
    "reservoirs": (
        "multivector_reservoirs_enabled",
        "multivector_reservoir_score_docs",
        "multivector_reservoir_coverage_docs",
        "multivector_reservoir_mean_docs",
        "multivector_reservoir_per_token_docs",
        "multivector_reservoir_bm25_docs",
        "multivector_reservoir_union_docs",
        "multivector_reservoir_duplicates",
    ),
}


@dataclass(frozen=True)
class QueryItem:
    query_id: str
    query_text: str


@dataclass
class WorkerResult:
    client_id: int
    warm_durations_ms: list[float] = field(default_factory=list)
    timed_durations_ms: list[float] = field(default_factory=list)
    timed_wall_ms: float = 0.0
    first_timed_stats: dict[str, Any] | None = None
    last_timed_stats: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class PersistWorkerResult:
    worker_id: int
    doc_latencies_ms: list[float] = field(default_factory=list)
    batch_latencies_ms: list[float] = field(default_factory=list)
    batch_sizes: list[int] = field(default_factory=list)
    doc_vector_counts: list[int] = field(default_factory=list)
    batches: int = 0
    error: str | None = None


@dataclass(frozen=True)
class DocumentNodeServingProfile:
    name: str
    candidate_source: str
    branch_plan: str = "dense_only"
    bm25_candidate_injection: str = "off"
    sparse_candidate_source: str = "off"
    proxy_encoder: str = "normalized_mean"
    centroids: str = "off"
    centroid_count: str = "auto"
    storage_kind: str = "f16"
    cache_mode: str = "auto"
    token_pooling: str = "off"
    token_pooling_target_ratio: float = 1.0
    plain_fallback: str = "off"
    candidate_reservoirs: str = "off"
    per_token_doc_reservoir_k: int = 1
    coverage_reservoir_k: int = 10
    centroid_lite_max_postings_per_token: int = 0
    centroid_lite_pruning: str = "off"
    entry_sample_count: int = 0
    entry_sidecar: bool = False
    entry_sidecar_representatives: int = 128
    entry_sidecar_strategy: str = "hybrid_level_covering"


@dataclass(frozen=True)
class DocumentNodeServingConfig:
    profile: DocumentNodeServingProfile
    ef: int
    oversampling: int


def require_pyarrow() -> Any:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise SystemExit(
            "pyarrow is required to read DBpedia parquet shards. "
            "Use the Nix bench shell or run through th-bench-dbpedia-colbert."
        ) from exc
    return pq


def portable_path(path: Path) -> str:
    try:
        return str(path.absolute().relative_to(Path.cwd().absolute()))
    except ValueError:
        pass
    try:
        return str(path.resolve().relative_to(Path.cwd().resolve()))
    except ValueError:
        return str(path)


def portable_argv(argv: list[str]) -> list[str]:
    result: list[str] = []
    for item in argv:
        if item.startswith(("/", "./", "../")):
            result.append(portable_path(Path(item)))
        else:
            result.append(item)
    return result


def command_metadata() -> dict[str, Any]:
    executable = portable_path(Path(sys.executable))
    argv = portable_argv(sys.argv)
    full_argv = [executable, *argv]
    return {
        "executable": executable,
        "argv": argv,
        "full_argv": full_argv,
        "shell": shlex.join(full_argv),
        "cwd": portable_path(Path.cwd()),
        "env": {
            "PGDATABASE": os.environ.get("PGDATABASE", ""),
            "PGHOST": os.environ.get("PGHOST", ""),
            "PGPORT": os.environ.get("PGPORT", ""),
            "PGUSER": os.environ.get("PGUSER", ""),
            "DBPEDIA_DATASET": os.environ.get("DBPEDIA_DATASET", ""),
            "BEIR_DBPEDIA_DATASET": os.environ.get("BEIR_DBPEDIA_DATASET", ""),
            "BEIR_DBPEDIA_QRELS": os.environ.get("BEIR_DBPEDIA_QRELS", ""),
            "PG_COLBERT_LLAMA_TEST_MODEL": os.environ.get("PG_COLBERT_LLAMA_TEST_MODEL", ""),
        },
    }


def git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    sha = result.stdout.strip()
    return sha or None


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    pos = (len(ordered) - 1) * pct / 100.0
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (pos - lower)


def summarize_ms(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": round(statistics.mean(values), 3) if values else 0.0,
        "p50_ms": round(percentile(values, 50), 3),
        "p95_ms": round(percentile(values, 95), 3),
        "p99_ms": round(percentile(values, 99), 3),
        "qps": round(1000.0 / statistics.mean(values), 3) if values else 0.0,
        "runs": len(values),
    }


def summarize_ints(values: list[int]) -> dict[str, float]:
    as_float = [float(value) for value in values]
    return {
        "mean": round(statistics.mean(as_float), 3) if values else 0.0,
        "min": min(values) if values else 0,
        "p50": round(percentile(as_float, 50), 3),
        "p95": round(percentile(as_float, 95), 3),
        "max": max(values) if values else 0,
        "count": len(values),
    }


def summarize_floats(values: list[float]) -> dict[str, float]:
    finite: list[float] = []
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed):
            finite.append(parsed)
    return {
        "mean": round(statistics.mean(finite), 6) if finite else 0.0,
        "min": round(min(finite), 6) if finite else 0.0,
        "p50": round(percentile(finite, 50), 6),
        "p95": round(percentile(finite, 95), 6),
        "max": round(max(finite), 6) if finite else 0.0,
        "count": len(finite),
    }


def summarize_strings(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if not value:
            continue
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def elapsed_ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 3)


def dcg(rels: list[int]) -> float:
    return sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(rels))


def metrics_for_run(run: dict[str, list[str]], qrels: dict[str, dict[str, int]], k: int) -> dict[str, float]:
    ndcg_total = 0.0
    recall_total = 0.0
    mrr_total = 0.0
    map_total = 0.0
    count = 0
    for qid, relevant in qrels.items():
        docs = run.get(qid, [])[:k]
        rels = [relevant.get(doc_id, 0) for doc_id in docs]
        ideal = sorted(relevant.values(), reverse=True)[:k]
        ideal_dcg = dcg(ideal)
        ndcg_total += dcg(rels) / ideal_dcg if ideal_dcg > 0 else 0.0
        hits = sum(1 for rel in rels if rel > 0)
        recall_total += hits / min(len(relevant), k)

        rr = 0.0
        precision_sum = 0.0
        seen_hits = 0
        for idx, rel in enumerate(rels, start=1):
            if rel > 0:
                if rr == 0.0:
                    rr = 1.0 / idx
                seen_hits += 1
                precision_sum += seen_hits / idx
        mrr_total += rr
        map_total += precision_sum / min(len(relevant), k)
        count += 1

    if count == 0:
        return {f"ndcg@{k}": 0.0, f"recall@{k}": 0.0, f"mrr@{k}": 0.0, f"map@{k}": 0.0}
    return {
        f"ndcg@{k}": round(ndcg_total / count, 6),
        f"recall@{k}": round(recall_total / count, 6),
        f"mrr@{k}": round(mrr_total / count, 6),
        f"map@{k}": round(map_total / count, 6),
    }


def find_parquet_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(root.glob(pattern)))
    return sorted({path.resolve() for path in files if path.is_file()})


def find_corpus_parquet(dataset: Path) -> list[Path]:
    files = find_parquet_files(dataset, ("data/*.parquet", "train/*.parquet", "*.parquet", "default/train/*.parquet"))
    if not files:
        raise FileNotFoundError(f"no DBpedia corpus parquet shards found under {dataset}")
    return files


def find_query_parquet(beir_dataset: Path) -> list[Path]:
    files = find_parquet_files(beir_dataset, ("queries/*.parquet", "*queries*.parquet", "*.parquet"))
    if not files:
        raise FileNotFoundError(f"no BEIR DBpedia query parquet file found under {beir_dataset}")
    return files


def resolve_qrels_path(path_arg: str | None, beir_dataset: Path) -> Path:
    candidates: list[Path] = []
    if path_arg:
        candidates.append(Path(path_arg))
    env_path = os.environ.get("BEIR_DBPEDIA_QRELS")
    if env_path:
        candidates.append(Path(env_path))
    candidates.extend([
        beir_dataset / "qrels" / "test.tsv",
        beir_dataset / "test.tsv",
        beir_dataset.parent / "beir-dbpedia-entity-qrels" / "test.tsv",
    ])
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("could not find BEIR DBpedia qrels; pass --qrels or set BEIR_DBPEDIA_QRELS")


def load_queries(beir_dataset: Path) -> dict[str, str]:
    pq = require_pyarrow()
    queries: dict[str, str] = {}
    for path in find_query_parquet(beir_dataset):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(columns=["_id", "title", "text"]):
            for row in batch.to_pylist():
                query_id = str(row["_id"])
                parts = [row.get("title") or "", row.get("text") or ""]
                queries[query_id] = " ".join(part for part in parts if part).strip()
    if not queries:
        raise ValueError(f"no BEIR queries loaded from {beir_dataset}")
    return queries


def read_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        has_header = "query-id" in header or "query_id" in header
        if not has_header:
            f.seek(0)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                raise ValueError(f"unexpected qrels row in {path}: {line!r}")
            qid, docid, score = parts
            qrels.setdefault(qid, {})[docid] = int(score)
    if not qrels:
        raise ValueError(f"no qrels loaded from {path}")
    return qrels


def choose_query_ids(qrels: dict[str, dict[str, int]], queries: dict[str, str], max_queries: int) -> list[str]:
    qids = [qid for qid in sorted(qrels) if qid in queries]
    if max_queries > 0:
        qids = qids[:max_queries]
    if not qids:
        raise ValueError("no benchmark queries have both text and qrels")
    return qids


def iter_corpus_rows(dataset: Path) -> Iterable[tuple[str, str, str]]:
    pq = require_pyarrow()
    for path in find_corpus_parquet(dataset):
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=1024, columns=["_id", "title", "text"]):
            for row in batch.to_pylist():
                yield str(row["_id"]), row.get("title") or "", row.get("text") or ""


def corpus_rows(
    dataset: Path,
    max_docs: int,
    priority_doc_ids: set[str] | None = None,
) -> Iterable[tuple[str, str, str]]:
    if not priority_doc_ids or max_docs == 0:
        emitted = 0
        for row in iter_corpus_rows(dataset):
            if max_docs > 0 and emitted >= max_docs:
                return
            emitted += 1
            yield row
        return

    emitted: set[str] = set()
    for doc_id, title, text in iter_corpus_rows(dataset):
        if doc_id not in priority_doc_ids or doc_id in emitted:
            continue
        emitted.add(doc_id)
        yield doc_id, title, text
        if len(emitted) >= max_docs:
            return

    for doc_id, title, text in iter_corpus_rows(dataset):
        if doc_id in emitted:
            continue
        emitted.add(doc_id)
        yield doc_id, title, text
        if len(emitted) >= max_docs:
            return


def query_rows(qids: list[str], queries: dict[str, str]) -> Iterable[tuple[str, str]]:
    for qid in qids:
        yield qid, queries[qid]


def qrel_rows(qids: list[str], qrels: dict[str, dict[str, int]]) -> Iterable[tuple[str, str, int]]:
    for qid in qids:
        for doc_id, score in sorted(qrels.get(qid, {}).items()):
            yield qid, doc_id, score


def qrel_doc_ids(qids: list[str], qrels: dict[str, dict[str, int]]) -> set[str]:
    return {doc_id for qid in qids for doc_id in qrels.get(qid, {})}


def filter_qrels_to_loaded_docs(
    qids: list[str],
    qrels: dict[str, dict[str, int]],
    loaded_doc_ids: set[str],
) -> tuple[list[str], dict[str, dict[str, int]]]:
    filtered: dict[str, dict[str, int]] = {}
    filtered_qids: list[str] = []
    for qid in qids:
        qid_qrels = {
            doc_id: score
            for doc_id, score in qrels.get(qid, {}).items()
            if doc_id in loaded_doc_ids
        }
        if qid_qrels:
            filtered[qid] = qid_qrels
            filtered_qids.append(qid)
    return filtered_qids, filtered


def connect(args: argparse.Namespace) -> psycopg.Connection[Any]:
    if psycopg is None:
        raise SystemExit(
            "psycopg is required to run the DBpedia ColBERT benchmark. "
            "Use the Nix benchmark helper or install psycopg[binary]."
        )
    return psycopg.connect(
        dbname=args.database,
        autocommit=True,
        application_name="dbpedia_colbert_multivector",
    )


def exec_sql(conn: psycopg.Connection[Any], sql: str, params: tuple[Any, ...] | None = None) -> None:
    with conn.cursor() as cur:
        cur.execute(sql, params)


def fetch_one(conn: psycopg.Connection[Any], sql: str, params: tuple[Any, ...] | None = None) -> tuple[Any, ...] | None:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def fetch_all(conn: psycopg.Connection[Any], sql: str, params: tuple[Any, ...] | None = None) -> list[tuple[Any, ...]]:
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return list(cur.fetchall())


def loaded_document_count(conn: psycopg.Connection[Any]) -> int:
    row = fetch_one(conn, "SELECT count(*) FROM dbpedia_colbert_docs WHERE colbert IS NOT NULL")
    return int(row[0]) if row else 0


def learned_sparse_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(float(numerator) / float(denominator), 6)


def learned_sparse_coverage_warnings(
    coverage: dict[str, Any],
    *,
    threshold: float = 0.95,
) -> list[str]:
    warnings: list[str] = []
    doc_ratio = coverage.get("doc_coverage_ratio")
    query_ratio = coverage.get("query_coverage_ratio")
    if doc_ratio is not None and float(doc_ratio) < threshold:
        warnings.append(
            "learned_sparse_doc_coverage_below_{:.0f}pct".format(threshold * 100.0)
        )
    if query_ratio is not None and float(query_ratio) < threshold:
        warnings.append(
            "learned_sparse_query_coverage_below_{:.0f}pct".format(threshold * 100.0)
        )
    return warnings


def learned_sparse_coverage_summary(conn: psycopg.Connection[Any]) -> dict[str, Any]:
    row = fetch_one(
        conn,
        """
        SELECT
          coalesce((SELECT count(*) FROM dbpedia_colbert_docs WHERE colbert IS NOT NULL), 0),
          coalesce((
            SELECT count(*)
            FROM dbpedia_colbert_docs
            WHERE colbert IS NOT NULL AND learned_sparse IS NOT NULL
          ), 0),
          coalesce((SELECT count(*) FROM dbpedia_colbert_queries WHERE colbert IS NOT NULL), 0),
          coalesce((
            SELECT count(*)
            FROM dbpedia_colbert_queries
            WHERE colbert IS NOT NULL AND learned_sparse IS NOT NULL
          ), 0)
        """,
    )
    loaded_docs = int(row[0] or 0) if row else 0
    sparse_docs = int(row[1] or 0) if row else 0
    loaded_queries = int(row[2] or 0) if row else 0
    sparse_queries = int(row[3] or 0) if row else 0
    coverage = {
        "loaded_documents": loaded_docs,
        "learned_sparse_documents": sparse_docs,
        "doc_coverage_ratio": learned_sparse_ratio(sparse_docs, loaded_docs),
        "loaded_queries": loaded_queries,
        "learned_sparse_queries": sparse_queries,
        "query_coverage_ratio": learned_sparse_ratio(sparse_queries, loaded_queries),
    }
    warnings = learned_sparse_coverage_warnings(coverage)
    coverage["partial_coverage"] = bool(warnings)
    coverage["warnings"] = warnings
    return coverage


def annotate_learned_sparse_evidence(
    row: dict[str, Any],
    coverage: dict[str, Any],
) -> None:
    if row.get("sparse_candidate_source") != "learned_sparse":
        return
    warnings = coverage.get("warnings", [])
    if not isinstance(warnings, list):
        warnings = []
    row["learned_sparse_coverage"] = coverage
    row["learned_sparse_partial_coverage"] = bool(coverage.get("partial_coverage", False))
    if warnings:
        evidence_warnings = row.get("evidence_warnings", [])
        if not isinstance(evidence_warnings, list):
            evidence_warnings = []
        row["evidence_warnings"] = sorted({*map(str, evidence_warnings), *map(str, warnings)})


def annotate_serving_grid_learned_sparse_evidence(
    serving_grid: dict[str, Any],
    learned_sparse_phase: dict[str, Any],
) -> None:
    if not isinstance(serving_grid, dict) or not isinstance(learned_sparse_phase, dict):
        return
    if bool(learned_sparse_phase.get("skipped", False)):
        return
    coverage = learned_sparse_phase.get("coverage", {})
    if not isinstance(coverage, dict):
        return
    serving_grid["learned_sparse_coverage"] = coverage
    serving_grid["learned_sparse_partial_coverage"] = bool(
        coverage.get("partial_coverage", False)
    )
    serving_grid["learned_sparse_evidence_warnings"] = list(
        coverage.get("warnings", [])
    )
    for key in (
        "results",
        "summary_rows",
        "stage1_results",
        "stage2_results",
        "profile_summaries",
    ):
        rows = serving_grid.get(key, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict):
                annotate_learned_sparse_evidence(row, coverage)


def multivector_dataset_stats(conn: psycopg.Connection[Any]) -> dict[str, Any]:
    row = fetch_one(
        conn,
        """
        WITH doc_stats AS (
          SELECT
            turbohybrid_multivector_count(colbert)::double precision AS token_count,
            turbohybrid_multivector_dims(colbert)::int AS dim
          FROM dbpedia_colbert_docs
          WHERE colbert IS NOT NULL
        )
        SELECT
          count(*)::bigint,
          avg(token_count)::double precision,
          percentile_cont(0.5) WITHIN GROUP (ORDER BY token_count)::double precision,
          percentile_cont(0.9) WITHIN GROUP (ORDER BY token_count)::double precision,
          min(token_count)::double precision,
          max(token_count)::double precision,
          min(dim)::int,
          max(dim)::int
        FROM doc_stats
        """,
    )
    if not row or int(row[0] or 0) == 0:
        return {
            "docs": 0,
            "avg_tokens": 0.0,
            "p50_tokens": 0.0,
            "p90_tokens": 0.0,
            "min_tokens": 0,
            "max_tokens": 0,
            "dim": 0,
            "dim_min": 0,
            "dim_max": 0,
        }

    dim_min = int(row[6] or 0)
    dim_max = int(row[7] or 0)
    return {
        "docs": int(row[0]),
        "avg_tokens": round(float(row[1] or 0.0), 3),
        "p50_tokens": round(float(row[2] or 0.0), 3),
        "p90_tokens": round(float(row[3] or 0.0), 3),
        "min_tokens": int(float(row[4] or 0.0)),
        "max_tokens": int(float(row[5] or 0.0)),
        "dim": dim_min if dim_min == dim_max else None,
        "dim_min": dim_min,
        "dim_max": dim_max,
    }


def validate_document_node_proxy_build(
    *,
    args: argparse.Namespace,
    index_stats: dict[str, Any],
    build_stats: dict[str, Any],
    row_count: int,
) -> dict[str, Any]:
    graph_mode = index_stats.get("multivector_graph_mode")
    requested_scorer = getattr(args, "multivector_doc_build_scorer", "proxy")
    observed_scorer = index_stats.get("multivector_doc_build_scorer")
    exact_calls = int(build_stats.get("multivector_doc_exact_build_distance_calls", 0) or 0)
    node_count = int(index_stats.get("node_count", 0) or 0)
    build_fast_edges = bool(index_stats.get("build_fast_edges", False))

    checks = {
        "graph_mode": graph_mode,
        "requested_doc_build_scorer": requested_scorer,
        "observed_doc_build_scorer": observed_scorer,
        "doc_exact_build_distance_calls": exact_calls,
        "node_count": node_count,
        "row_count": row_count,
        "build_fast_edges": build_fast_edges,
    }
    if graph_mode != "document_nodes":
        return {**checks, "enforced": False, "reason": "not_document_nodes"}
    if requested_scorer == "exact_symmetric":
        return {**checks, "enforced": False, "reason": "exact_symmetric_explicitly_allowed"}

    if observed_scorer != "proxy":
        raise SystemExit(
            "refusing benchmark: document_nodes index used "
            f"multivector_doc_build_scorer={observed_scorer!r}, expected 'proxy'"
        )
    if exact_calls != 0:
        raise SystemExit(
            "refusing benchmark: document_nodes proxy build made "
            f"{exact_calls} exact document-document MaxSim build-distance calls"
        )
    if not build_fast_edges:
        raise SystemExit(
            "refusing benchmark: document_nodes proxy index did not use "
            "fast edge construction"
        )
    if node_count != row_count:
        raise SystemExit(
            "refusing benchmark: document_nodes proxy index node_count "
            f"{node_count} does not match loaded document count {row_count}"
        )
    return {**checks, "enforced": True, "reason": "ok"}


def build_index_reloptions(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    pooling_mode = getattr(args, "multivector_token_pooling", "off")
    pooling_ratio = float(getattr(args, "multivector_token_pooling_target_ratio", 0.5))
    pooling_min_tokens = int(getattr(args, "multivector_token_pooling_min_tokens", 16))
    centroids = getattr(args, "multivector_centroids", "off")
    centroid_count = getattr(args, "multivector_centroid_count", "auto")
    if isinstance(centroid_count, str) and centroid_count.lower() == "auto":
        centroid_count_sql = 0
    else:
        centroid_count_sql = int(centroid_count)
    proxy_encoder = getattr(args, "multivector_proxy_encoder", "normalized_mean")
    reloptions = [
        "quantization_bits = 4",
        "exact_storage = off",
        f"multivector_graph = {args.multivector_graph}",
        f"multivector_doc_build_scorer = {args.multivector_doc_build_scorer}",
        f"multivector_token_pooling = {pooling_mode}",
        f"multivector_token_pooling_target_ratio = {pooling_ratio}",
        f"multivector_token_pooling_min_tokens = {pooling_min_tokens}",
        f"multivector_centroids = {centroids}",
        f"multivector_centroid_count = {centroid_count_sql}",
        f"multivector_proxy_encoder = {proxy_encoder}",
    ]
    index_graph_m = int(getattr(args, "index_graph_m", 0))
    index_graph_ef_construction = int(getattr(args, "index_graph_ef_construction", 0))
    index_graph_ef_search = int(getattr(args, "index_graph_ef_search", 0))
    index_native_segments = int(getattr(args, "index_native_segments", 0))
    if index_graph_m > 0:
        reloptions.append(f"graph_m = {index_graph_m}")
    if index_graph_ef_construction > 0:
        reloptions.append(f"graph_ef_construction = {index_graph_ef_construction}")
    if index_graph_ef_search > 0:
        reloptions.append(f"graph_ef_search = {index_graph_ef_search}")
    if index_native_segments > 0:
        reloptions.append(f"native_segments = {index_native_segments}")
    entry_sidecar = bool(getattr(args, "entry_sidecar", False))
    if entry_sidecar:
        representatives = int(getattr(args, "entry_sidecar_representatives", 128))
        strategy = str(getattr(args, "entry_sidecar_strategy", "hybrid_level_covering"))
        reloptions.extend([
            "entry_sidecar = on",
            f"entry_sidecar_representatives = {representatives}",
            f"entry_sidecar_strategy = {strategy}",
        ])
    return reloptions, {
        "index_graph_m": index_graph_m,
        "index_graph_ef_construction": index_graph_ef_construction,
        "index_graph_ef_search": index_graph_ef_search,
        "index_native_segments": index_native_segments,
        "entry_sidecar": entry_sidecar,
        "entry_sidecar_representatives": int(
            getattr(args, "entry_sidecar_representatives", 128)
        ),
        "entry_sidecar_strategy": str(
            getattr(args, "entry_sidecar_strategy", "hybrid_level_covering")
        ),
    }


def build_index_needs_lexical_key(args: argparse.Namespace) -> bool:
    return build_index_lexical_column(args) is not None


def build_index_lexical_column(args: argparse.Namespace) -> str | None:
    methods = set(getattr(args, "methods", []))
    bm25_injection = getattr(args, "multivector_bm25_candidate_injection", "off")
    sparse_source = getattr(args, "multivector_sparse_candidate_source", "off")
    learned_sparse_needed = sparse_source == "learned_sparse"
    text_bm25_needed = (
        RRF_METHOD in methods
        or bool(getattr(args, "hybrid_evaluation_harness", False))
        or bm25_injection != "off"
        or sparse_source == "bm25"
    )
    if learned_sparse_needed and text_bm25_needed:
        raise SystemExit(
            "refusing benchmark index build: learned_sparse and BM25/body text "
            "candidate sources need different lexical index columns"
        )
    if learned_sparse_needed:
        return "learned_sparse_tsv"
    if text_bm25_needed:
        return "body_tsv"
    if sparse_source not in {"off", "bm25", "learned_sparse"}:
        raise SystemExit(f"unknown multivector sparse candidate source: {sparse_source!r}")
    return None


def serving_profile_index_signature(
    args: argparse.Namespace,
    profile: DocumentNodeServingProfile,
) -> tuple[tuple[str, Any], ...]:
    index_args = document_node_serving_profile_args(
        args,
        profile,
        ef=0,
        oversampling=1,
    )
    reloptions, index_option_values = build_index_reloptions(index_args)
    return (
        ("reloptions", tuple(reloptions)),
        ("index_option_values", tuple(sorted(index_option_values.items()))),
        ("lexical_column", build_index_lexical_column(index_args)),
        ("lexical_key_required", build_index_needs_lexical_key(index_args)),
    )


def load_pgturbohybrid_library(conn: psycopg.Connection[Any]) -> None:
    # CREATE EXTENSION defines the SQL objects, but the C library and its GUCs
    # are loaded lazily when a C function is first called in this session.
    fetch_one(conn, "SELECT turbohybrid_last_scan_stats()")


def jsonb_value(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"value": parsed}
        except json.JSONDecodeError:
            return {"value": value}
    return {"value": value}


def copy_rows(conn: psycopg.Connection[Any], sql: str, rows: Iterable[tuple[Any, ...]]) -> int:
    count = 0
    with conn.cursor() as cur:
        with cur.copy(sql) as copy:
            for row in rows:
                copy.write_row(row)
                count += 1
    return count


def parse_sparse_jsonl(path: Path, preferred_id_key: str) -> list[tuple[str, list[int], list[float]]]:
    rows: list[tuple[str, list[int], list[float]]] = []
    id_keys = (preferred_id_key, "id")
    term_keys = ("term_ids", "terms", "indices")
    weight_keys = ("weights", "scores", "values")
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSONL row") from exc
            if not isinstance(item, dict):
                raise SystemExit(f"{path}:{line_no}: expected a JSON object")

            sparse_id = next((item[key] for key in id_keys if key in item), None)
            terms = next((item[key] for key in term_keys if key in item), None)
            weights = next((item[key] for key in weight_keys if key in item), None)
            if sparse_id is None or terms is None or weights is None:
                raise SystemExit(
                    f"{path}:{line_no}: expected {preferred_id_key!r}, term_ids, and weights fields"
                )
            if not isinstance(terms, list) or not isinstance(weights, list) or len(terms) != len(weights):
                raise SystemExit(f"{path}:{line_no}: term_ids and weights must be equal-length arrays")

            term_ids: list[int] = []
            sparse_weights: list[float] = []
            for term, weight in zip(terms, weights):
                term_id = int(term)
                sparse_weight = float(weight)
                if term_id < 0 or not math.isfinite(sparse_weight):
                    raise SystemExit(f"{path}:{line_no}: sparse terms must be non-negative and weights finite")
                term_ids.append(term_id)
                sparse_weights.append(sparse_weight)
            rows.append((str(sparse_id), term_ids, sparse_weights))
    return rows


def require_complete_learned_sparse_args(args: argparse.Namespace) -> None:
    if (args.learned_sparse_doc_jsonl is None) != (args.learned_sparse_query_jsonl is None):
        raise SystemExit(
            "--learned-sparse-doc-jsonl and --learned-sparse-query-jsonl must be supplied together"
        )


def learned_sparse_text_query_sql() -> str:
    return """
    CASE
      WHEN q.learned_sparse IS NOT NULL THEN turbohybrid_sparse_vector_to_tsquery(q.learned_sparse)
      ELSE websearch_to_tsquery('simple', q.query_text)
    END
    """


def query_only_needs_text_query(args: argparse.Namespace) -> bool:
    return (
        getattr(args, "multivector_bm25_candidate_injection", "off") != "off"
        or getattr(args, "multivector_sparse_candidate_source", "off") != "off"
    )


def query_only_text_query_sql(args: argparse.Namespace) -> str:
    if getattr(args, "multivector_sparse_candidate_source", "off") == "learned_sparse":
        return learned_sparse_text_query_sql()
    return "websearch_to_tsquery('simple', q.query_text)"


def load_learned_sparse_vectors(conn: psycopg.Connection[Any], args: argparse.Namespace) -> dict[str, Any]:
    require_complete_learned_sparse_args(args)
    if args.learned_sparse_doc_jsonl is None:
        return {"skipped": True, "reason": "no learned sparse JSONL files supplied"}

    started = time.perf_counter()
    doc_rows = parse_sparse_jsonl(args.learned_sparse_doc_jsonl, "doc_id")
    query_rows_sparse = parse_sparse_jsonl(args.learned_sparse_query_jsonl, "query_id")
    doc_updates = 0
    query_updates = 0
    with conn.cursor() as cur:
        for doc_id, term_ids, weights in doc_rows:
            cur.execute(
                """
                UPDATE dbpedia_colbert_docs
                SET learned_sparse = turbohybrid_sparse_vector_from_arrays(%s::int[], %s::real[])
                WHERE doc_id = %s
                """,
                (term_ids, weights, doc_id),
            )
            doc_updates += cur.rowcount
        for query_id, term_ids, weights in query_rows_sparse:
            cur.execute(
                """
                UPDATE dbpedia_colbert_queries
                SET learned_sparse = turbohybrid_sparse_vector_from_arrays(%s::int[], %s::real[])
                WHERE query_id = %s
                """,
                (term_ids, weights, query_id),
            )
            query_updates += cur.rowcount

    if doc_rows and doc_updates == 0:
        raise RuntimeError(f"no learned sparse document rows matched loaded docs from {args.learned_sparse_doc_jsonl}")
    if query_rows_sparse and query_updates == 0:
        raise RuntimeError(f"no learned sparse query rows matched loaded queries from {args.learned_sparse_query_jsonl}")

    coverage = learned_sparse_coverage_summary(conn)
    return {
        "skipped": False,
        "doc_jsonl": portable_path(args.learned_sparse_doc_jsonl),
        "query_jsonl": portable_path(args.learned_sparse_query_jsonl),
        "doc_rows_read": len(doc_rows),
        "doc_rows_updated": doc_updates,
        "query_rows_read": len(query_rows_sparse),
        "query_rows_updated": query_updates,
        "coverage": coverage,
        "partial_coverage": bool(coverage.get("partial_coverage", False)),
        "warnings": list(coverage.get("warnings", [])),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def set_colbert_gucs(conn: psycopg.Connection[Any], args: argparse.Namespace) -> None:
    settings = {
        "jit": "off",
        "max_parallel_workers_per_gather": "0",
        "pg_colbert_llama.model_dir": str(args.model_path.parent),
        "pg_colbert_llama.threads": str(args.generation_threads),
        "pg_colbert_llama.n_batch": str(args.generation_n_batch),
        "pg_colbert_llama.batch_sequences": str(args.generation_batch_sequences),
        "pg_colbert_llama.n_gpu_layers": str(args.generation_n_gpu_layers),
        "pg_colbert_llama.expected_dim": str(args.expected_dim),
        "pg_colbert_llama.max_doc_vectors": str(args.max_doc_vectors),
        "pg_colbert_llama.max_query_vectors": str(args.max_query_vectors),
        "pg_colbert_llama.query_length": str(args.query_length),
        "turbohybrid.multivector_model_name": args.colbert_model_name,
        "turbohybrid.multivector_subvector_k": str(args.multivector_subvector_k),
        "turbohybrid.multivector_unique_docs_per_token": str(args.multivector_unique_docs_per_token),
        "turbohybrid.multivector_max_raw_hits_per_token": str(args.multivector_max_raw_hits_per_token),
        "turbohybrid.multivector_adaptive_widening": args.multivector_adaptive_widening,
        "turbohybrid.multivector_doc_candidate_k": str(args.multivector_doc_candidate_k),
        "turbohybrid.multivector_exact_rerank": args.multivector_exact_rerank,
        "turbohybrid.multivector_exact_rerank_k": str(args.multivector_exact_rerank_k),
        "turbohybrid.multivector_doc_graph_search_ef": str(args.multivector_doc_graph_search_ef),
        "turbohybrid.multivector_doc_graph_oversampling": str(args.multivector_doc_graph_oversampling),
        "turbohybrid.multivector_doc_graph_rescore_k": str(args.multivector_doc_graph_rescore_k),
        "turbohybrid.multivector_doc_graph_entry_sample_count": str(
            args.multivector_doc_graph_entry_sample_count
        ),
        "turbohybrid.multivector_doc_storage": args.multivector_doc_storage,
        "turbohybrid.multivector_doc_storage_cache": args.multivector_doc_storage_cache,
        "turbohybrid.multivector_proxy_encoder": args.multivector_proxy_encoder,
        "turbohybrid.multivector_learned_projection_path": str(
            getattr(args, "multivector_learned_projection_path", "") or ""
        ),
        "turbohybrid.multivector_learned_projection_model": str(
            getattr(args, "multivector_learned_projection_model", "") or ""
        ),
        "turbohybrid.multivector_learned_projection_checksum": str(
            getattr(args, "multivector_learned_projection_checksum", "") or ""
        ),
        "turbohybrid.multivector_centroid_lite_max_postings_per_token": str(
            args.multivector_centroid_lite_max_postings_per_token
        ),
        "turbohybrid.multivector_centroid_lite_pruning": str(
            args.multivector_centroid_lite_pruning
        ),
        "turbohybrid.multivector_candidate_source": args.multivector_candidate_source,
        "turbohybrid.multivector_branch_plan": getattr(args, "multivector_branch_plan", "auto"),
        "turbohybrid.multivector_plain_fallback": args.multivector_plain_fallback,
        "turbohybrid.multivector_plain_fallback_max_docs": str(args.multivector_plain_fallback_max_docs),
        "turbohybrid.multivector_plain_fallback_candidate_fraction": str(args.multivector_plain_fallback_candidate_fraction),
        "turbohybrid.multivector_candidate_reservoirs": args.multivector_candidate_reservoirs,
        "turbohybrid.multivector_per_token_doc_reservoir_k": str(args.multivector_per_token_doc_reservoir_k),
        "turbohybrid.multivector_coverage_reservoir_k": str(args.multivector_coverage_reservoir_k),
        "turbohybrid.multivector_bm25_candidate_injection": args.multivector_bm25_candidate_injection,
        "turbohybrid.multivector_sparse_candidate_source": args.multivector_sparse_candidate_source,
        "turbohybrid.multivector_debug_skip_query_tokens": "",
    }
    for key, value in settings.items():
        exec_sql(conn, "SELECT set_config(%s, %s, false)", (key, value))


def set_retrieval_gucs(conn: psycopg.Connection[Any], args: argparse.Namespace, app_name: str) -> None:
    settings = {
        "application_name": app_name,
        "enable_seqscan": "off",
        "jit": "off",
        "max_parallel_workers_per_gather": "0",
        "pg_colbert_llama.model_dir": str(args.model_path.parent),
        "pg_colbert_llama.expected_dim": str(args.expected_dim),
        "pg_colbert_llama.max_doc_vectors": str(args.max_doc_vectors),
        "pg_colbert_llama.max_query_vectors": str(args.max_query_vectors),
        "pg_colbert_llama.query_length": str(args.query_length),
        "turbohybrid.multivector_model_name": args.colbert_model_name,
        "turbohybrid.multivector_subvector_k": str(args.multivector_subvector_k),
        "turbohybrid.multivector_unique_docs_per_token": str(args.multivector_unique_docs_per_token),
        "turbohybrid.multivector_max_raw_hits_per_token": str(args.multivector_max_raw_hits_per_token),
        "turbohybrid.multivector_adaptive_widening": args.multivector_adaptive_widening,
        "turbohybrid.multivector_doc_candidate_k": str(args.multivector_doc_candidate_k),
        "turbohybrid.multivector_exact_rerank": args.multivector_exact_rerank,
        "turbohybrid.multivector_exact_rerank_k": str(args.multivector_exact_rerank_k),
        "turbohybrid.multivector_doc_graph_search_ef": str(args.multivector_doc_graph_search_ef),
        "turbohybrid.multivector_doc_graph_oversampling": str(args.multivector_doc_graph_oversampling),
        "turbohybrid.multivector_doc_graph_rescore_k": str(args.multivector_doc_graph_rescore_k),
        "turbohybrid.multivector_doc_graph_entry_sample_count": str(
            args.multivector_doc_graph_entry_sample_count
        ),
        "turbohybrid.multivector_doc_storage": args.multivector_doc_storage,
        "turbohybrid.multivector_doc_storage_cache": args.multivector_doc_storage_cache,
        "turbohybrid.multivector_proxy_encoder": args.multivector_proxy_encoder,
        "turbohybrid.multivector_learned_projection_path": str(
            getattr(args, "multivector_learned_projection_path", "") or ""
        ),
        "turbohybrid.multivector_learned_projection_model": str(
            getattr(args, "multivector_learned_projection_model", "") or ""
        ),
        "turbohybrid.multivector_learned_projection_checksum": str(
            getattr(args, "multivector_learned_projection_checksum", "") or ""
        ),
        "turbohybrid.multivector_centroid_lite_max_postings_per_token": str(
            args.multivector_centroid_lite_max_postings_per_token
        ),
        "turbohybrid.multivector_centroid_lite_pruning": str(
            args.multivector_centroid_lite_pruning
        ),
        "turbohybrid.multivector_candidate_source": args.multivector_candidate_source,
        "turbohybrid.multivector_branch_plan": getattr(args, "multivector_branch_plan", "auto"),
        "turbohybrid.multivector_plain_fallback": args.multivector_plain_fallback,
        "turbohybrid.multivector_plain_fallback_max_docs": str(args.multivector_plain_fallback_max_docs),
        "turbohybrid.multivector_plain_fallback_candidate_fraction": str(args.multivector_plain_fallback_candidate_fraction),
        "turbohybrid.multivector_candidate_reservoirs": args.multivector_candidate_reservoirs,
        "turbohybrid.multivector_per_token_doc_reservoir_k": str(args.multivector_per_token_doc_reservoir_k),
        "turbohybrid.multivector_coverage_reservoir_k": str(args.multivector_coverage_reservoir_k),
        "turbohybrid.multivector_bm25_candidate_injection": args.multivector_bm25_candidate_injection,
        "turbohybrid.multivector_sparse_candidate_source": args.multivector_sparse_candidate_source,
        "turbohybrid.multivector_debug_skip_query_tokens": "",
    }
    for key, value in settings.items():
        exec_sql(conn, "SELECT set_config(%s, %s, false)", (key, value))


def setup_schema(conn: psycopg.Connection[Any], include_colbert_llama: bool = True) -> None:
    exec_sql(conn, "CREATE EXTENSION IF NOT EXISTS vector")
    exec_sql(conn, "CREATE EXTENSION IF NOT EXISTS pgturbohybrid")
    load_pgturbohybrid_library(conn)
    if include_colbert_llama:
        exec_sql(conn, "CREATE EXTENSION IF NOT EXISTS pg_colbert_llama")
    exec_sql(
        conn,
        """
        CREATE TABLE IF NOT EXISTS dbpedia_colbert_docs (
            doc_id text PRIMARY KEY,
            title text NOT NULL,
            body text NOT NULL,
            body_tsv tsvector GENERATED ALWAYS AS (
                to_tsvector('simple', coalesce(title, '') || ' ' || coalesce(body, ''))
            ) STORED,
            learned_sparse turbohybrid_sparse_vector,
            learned_sparse_tsv tsvector GENERATED ALWAYS AS (
                CASE
                  WHEN learned_sparse IS NULL THEN NULL
                  ELSE turbohybrid_sparse_vector_to_tsvector(learned_sparse)
                END
            ) STORED,
            colbert turbohybrid_multivector
        )
        """,
    )
    exec_sql(conn, "ALTER TABLE dbpedia_colbert_docs ADD COLUMN IF NOT EXISTS learned_sparse turbohybrid_sparse_vector")
    exec_sql(
        conn,
        """
        ALTER TABLE dbpedia_colbert_docs
        ADD COLUMN IF NOT EXISTS learned_sparse_tsv tsvector GENERATED ALWAYS AS (
            CASE
              WHEN learned_sparse IS NULL THEN NULL
              ELSE turbohybrid_sparse_vector_to_tsvector(learned_sparse)
            END
        ) STORED
        """,
    )
    exec_sql(
        conn,
        """
        CREATE TABLE IF NOT EXISTS dbpedia_colbert_queries (
            query_id text PRIMARY KEY,
            query_text text NOT NULL,
            learned_sparse turbohybrid_sparse_vector,
            colbert turbohybrid_multivector
        )
        """,
    )
    exec_sql(conn, "ALTER TABLE dbpedia_colbert_queries ADD COLUMN IF NOT EXISTS learned_sparse turbohybrid_sparse_vector")
    exec_sql(
        conn,
        """
        CREATE TABLE IF NOT EXISTS dbpedia_colbert_qrels (
            query_id text NOT NULL,
            doc_id text NOT NULL,
            relevance int NOT NULL,
            PRIMARY KEY (query_id, doc_id)
        )
        """,
    )


def validate_embedding_health(conn: psycopg.Connection[Any], args: argparse.Namespace) -> dict[str, Any]:
    set_colbert_gucs(conn, args)
    query = jsonb_value(fetch_one(conn, "SELECT colbert(%s, 'red planet')", (f"{args.model_alias}:query",))[0])
    doc = jsonb_value(fetch_one(conn, "SELECT colbert(%s, 'red planet')", (f"{args.model_alias}:doc",))[0])
    punct_doc = jsonb_value(
        fetch_one(conn, "SELECT colbert(%s, 'Mars is often called the red planet.')", (f"{args.model_alias}:doc",))[0]
    )

    query_tokens = [int(token) for token in query.get("token_ids", [])]
    doc_tokens = [int(token) for token in doc.get("token_ids", [])]
    punct_doc_tokens = [int(token) for token in punct_doc.get("token_ids", [])]
    expected_query_prefix = [101, 30522, 2417, 4774, 102]
    expected_doc_tokens = [101, 30523, 2417, 4774, 102]
    failures: list[str] = []

    if int(query.get("dim", 0)) != args.expected_dim or int(doc.get("dim", 0)) != args.expected_dim:
        failures.append("probe embeddings have the wrong dimension")
    if query_tokens[: len(expected_query_prefix)] != expected_query_prefix:
        failures.append(f"query tokenization probe mismatch: got {query_tokens[:len(expected_query_prefix)]}")
    if args.query_length <= args.max_query_vectors and len(query_tokens) != args.query_length:
        failures.append(f"query expansion length mismatch: got {len(query_tokens)}, expected {args.query_length}")
    if 100 in query_tokens[: len(expected_query_prefix)]:
        failures.append("query tokenization produced [UNK] for normal probe words")
    if doc_tokens != expected_doc_tokens:
        failures.append(f"document tokenization probe mismatch: got {doc_tokens}")
    if 1012 in punct_doc_tokens:
        failures.append("document tokenization retained '.' punctuation skiplist token")

    result = {
        "validated": not failures,
        "query_probe_token_ids": query_tokens,
        "document_probe_token_ids": doc_tokens,
        "punctuation_probe_token_ids": punct_doc_tokens,
        "expected_query_prefix": expected_query_prefix,
        "expected_document_tokens": expected_doc_tokens,
        "query_length": args.query_length,
        "failures": failures,
    }
    if failures and not args.allow_unvalidated_embeddings:
        raise RuntimeError("invalid ColBERT embedding preflight: " + "; ".join(failures))
    return result


def resolve_expected_dim(value: str, model_name: str) -> tuple[int, str]:
    requested = value.strip().lower()
    if requested == "auto":
        dim = COLBERT_MODEL_DIMENSIONS.get(model_name.lower())
        if dim is None:
            raise SystemExit(
                f"--expected-dim auto does not know model {model_name!r}; "
                "pass --expected-dim explicitly or add the model to the registry."
            )
        return dim, "model_registry"

    try:
        dim = int(value)
    except ValueError as exc:
        raise SystemExit("--expected-dim must be a positive integer or 'auto'") from exc
    if dim <= 0:
        raise SystemExit("--expected-dim must be positive")
    return dim, "explicit"


def load_data(
    conn: psycopg.Connection[Any],
    args: argparse.Namespace,
    qids: list[str],
    queries: dict[str, str],
    qrels: dict[str, dict[str, int]],
) -> dict[str, Any]:
    if args.reuse_data and not args.force_reload:
        row = fetch_one(
            conn,
            """
            SELECT
              coalesce((SELECT count(*) FROM dbpedia_colbert_docs), 0),
              coalesce((SELECT count(*) FROM dbpedia_colbert_queries), 0),
              coalesce((SELECT count(*) FROM dbpedia_colbert_qrels), 0),
              coalesce((
                SELECT count(*)
                FROM dbpedia_colbert_qrels q
                JOIN dbpedia_colbert_docs d ON d.doc_id = q.doc_id
              ), 0)
            """,
        )
        if row and int(row[0]) > 0 and int(row[1]) > 0 and int(row[2]) > 0 and int(row[2]) == int(row[3]):
            return {
                "reused": True,
                "documents": int(row[0]),
                "queries": int(row[1]),
                "qrels": int(row[2]),
                "qrels_in_loaded_docs": int(row[3]),
            }

    started = time.perf_counter()
    exec_sql(conn, "DROP INDEX IF EXISTS dbpedia_colbert_docs_colbert_idx")
    exec_sql(conn, "TRUNCATE dbpedia_colbert_qrels, dbpedia_colbert_queries, dbpedia_colbert_docs")
    requested_qrel_doc_ids = qrel_doc_ids(qids, qrels)
    doc_count = copy_rows(
        conn,
        "COPY dbpedia_colbert_docs (doc_id, title, body) FROM STDIN",
        corpus_rows(args.dataset, args.max_docs, requested_qrel_doc_ids if args.prioritize_qrels else None),
    )
    loaded_doc_ids = set(selected_doc_ids(conn, 0))
    loaded_qids, loaded_qrels = filter_qrels_to_loaded_docs(qids, qrels, loaded_doc_ids)
    if not loaded_qids:
        raise RuntimeError(
            "none of the selected qrels are present in the loaded corpus; "
            "increase --max-docs or verify the corpus/qrels id space"
        )
    query_count = copy_rows(
        conn,
        "COPY dbpedia_colbert_queries (query_id, query_text) FROM STDIN",
        query_rows(loaded_qids, queries),
    )
    qrel_count = copy_rows(
        conn,
        "COPY dbpedia_colbert_qrels (query_id, doc_id, relevance) FROM STDIN",
        qrel_rows(loaded_qids, loaded_qrels),
    )
    return {
        "reused": False,
        "documents": doc_count,
        "queries": query_count,
        "qrels": qrel_count,
        "input_qrels": sum(len(qrels.get(qid, {})) for qid in qids),
        "requested_qrel_doc_ids": len(requested_qrel_doc_ids),
        "qrels_in_loaded_docs": qrel_count,
        "queries_with_loaded_qrels": len(loaded_qids),
        "prioritized_qrels": bool(args.prioritize_qrels),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
    }


def selected_doc_ids(conn: psycopg.Connection[Any], limit: int) -> list[str]:
    if limit > 0:
        rows = fetch_all(conn, "SELECT doc_id FROM dbpedia_colbert_docs ORDER BY doc_id LIMIT %s", (limit,))
    else:
        rows = fetch_all(conn, "SELECT doc_id FROM dbpedia_colbert_docs ORDER BY doc_id")
    return [str(row[0]) for row in rows]


def selected_query_ids(conn: psycopg.Connection[Any]) -> list[str]:
    rows = fetch_all(conn, "SELECT query_id FROM dbpedia_colbert_queries ORDER BY query_id")
    return [str(row[0]) for row in rows]


def loaded_qrels(conn: psycopg.Connection[Any]) -> dict[str, dict[str, int]]:
    rows = fetch_all(
        conn,
        """
        SELECT query_id, doc_id, relevance
        FROM dbpedia_colbert_qrels
        ORDER BY query_id, doc_id
        """,
    )
    qrels: dict[str, dict[str, int]] = {}
    for query_id, doc_id, relevance in rows:
        qrels.setdefault(str(query_id), {})[str(doc_id)] = int(relevance)
    return qrels


def measure_generation_sample(
    conn: psycopg.Connection[Any],
    args: argparse.Namespace,
    doc_ids: list[str],
    query_ids: list[str],
) -> dict[str, Any]:
    set_colbert_gucs(conn, args)
    doc_latencies: list[float] = []
    doc_counts: list[int] = []
    for doc_id in doc_ids[: args.generation_sample_docs]:
        started = time.perf_counter()
        row = fetch_one(
            conn,
            """
            WITH encoded AS MATERIALIZED (
              SELECT colbert_mv(%s, concat_ws(' ', nullif(title, ''), body)) AS mv
              FROM dbpedia_colbert_docs
              WHERE doc_id = %s
            )
            SELECT turbohybrid_multivector_dims(mv), turbohybrid_multivector_count(mv)
            FROM encoded
            """,
            (f"{args.model_alias}:doc", doc_id),
        )
        doc_latencies.append((time.perf_counter() - started) * 1000.0)
        if row:
            doc_counts.append(int(row[1]))

    query_latencies: list[float] = []
    query_counts: list[int] = []
    for query_id in query_ids:
        started = time.perf_counter()
        row = fetch_one(
            conn,
            """
            WITH encoded AS MATERIALIZED (
              SELECT colbert_mv(%s, query_text) AS mv
              FROM dbpedia_colbert_queries
              WHERE query_id = %s
            )
            SELECT turbohybrid_multivector_dims(mv), turbohybrid_multivector_count(mv)
            FROM encoded
            """,
            (f"{args.model_alias}:query", query_id),
        )
        query_latencies.append((time.perf_counter() - started) * 1000.0)
        if row:
            query_counts.append(int(row[1]))

    return {
        "documents": summarize_ms(doc_latencies),
        "document_token_vectors": summarize_ints(doc_counts),
        "queries": summarize_ms(query_latencies),
        "query_token_vectors": summarize_ints(query_counts),
        "note": "generation-only sample; persisted insert/update phase below includes generation plus storage",
    }


def count_existing_embeddings(conn: psycopg.Connection[Any]) -> tuple[int, int]:
    row = fetch_one(
        conn,
        """
        SELECT
          count(*) FILTER (WHERE colbert IS NOT NULL),
          count(*)
        FROM dbpedia_colbert_docs
        """,
    )
    return (int(row[0]), int(row[1])) if row else (0, 0)


def ids_missing_embeddings(conn: psycopg.Connection[Any], table_name: str, id_column: str) -> list[str]:
    rows = fetch_all(
        conn,
        f"""
        SELECT {id_column}
        FROM {table_name}
        WHERE colbert IS NULL
        ORDER BY {id_column}
        """,
    )
    return [str(row[0]) for row in rows]


def run_persist_worker(
    *,
    args: argparse.Namespace,
    batches: list[list[str]],
    next_batch: list[int],
    batch_lock: threading.Lock,
    ready_barrier: threading.Barrier,
    start_event: threading.Event,
    progress: dict[str, int],
    progress_lock: threading.Lock,
    total_docs: int,
    result: PersistWorkerResult,
) -> None:
    conn = connect(args)
    try:
        set_colbert_gucs(conn, args)
        if args.generation_warmup:
            fetch_one(conn, "SELECT colbert_mv_batch(%s, %s::text[])", (f"{args.model_alias}:doc", ["warmup document"]))
        ready_barrier.wait()
        start_event.wait()
        while True:
            with batch_lock:
                if next_batch[0] >= len(batches):
                    break
                batch = batches[next_batch[0]]
                next_batch[0] += 1

            started = time.perf_counter()
            rows = fetch_all(
                conn,
                """
                WITH requested AS MATERIALIZED (
                  SELECT doc_id, ord
                  FROM unnest(%s::text[]) WITH ORDINALITY AS u(doc_id, ord)
                ),
                encoded AS MATERIALIZED (
                  SELECT
                    array_agg(d.doc_id ORDER BY r.ord) AS doc_ids,
                    colbert_mv_batch(
                      %s,
                      array_agg(concat_ws(' ', nullif(d.title, ''), d.body) ORDER BY r.ord)
                    ) AS multivectors
                  FROM requested r
                  JOIN dbpedia_colbert_docs d ON d.doc_id = r.doc_id
                )
                UPDATE dbpedia_colbert_docs d
                SET colbert = encoded.multivectors[i]
                FROM encoded,
                     generate_subscripts(encoded.doc_ids, 1) AS s(i)
                WHERE d.doc_id = encoded.doc_ids[i]
                RETURNING turbohybrid_multivector_count(d.colbert)
                """,
                (batch, f"{args.model_alias}:doc"),
            )
            elapsed = (time.perf_counter() - started) * 1000.0
            per_doc = elapsed / max(len(rows), 1)
            result.doc_latencies_ms.extend([per_doc] * len(rows))
            result.batch_latencies_ms.append(elapsed)
            result.batch_sizes.append(len(rows))
            result.doc_vector_counts.extend(int(row[0]) for row in rows)
            result.batches += 1

            with progress_lock:
                progress["documents"] += len(rows)
                completed = progress["documents"]
                next_report = progress["next_report"]
                if completed >= next_report:
                    print(f"persisted {completed}/{total_docs} document multivectors", file=sys.stderr)
                    step = max(args.progress_every, 1)
                    progress["next_report"] = ((completed // step) + 1) * step
    except Exception as exc:  # pragma: no cover - benchmark error path
        result.error = str(exc)
        try:
            ready_barrier.abort()
        except Exception:
            pass
    finally:
        conn.close()


def persist_document_multivectors(
    args: argparse.Namespace,
    doc_ids: list[str],
) -> tuple[list[float], list[int], dict[str, Any]]:
    if not doc_ids:
        return [], [], {
            "workers": 0,
            "batches": 0,
            "elapsed_ms": 0.0,
        }

    batches = [doc_ids[offset : offset + args.insert_batch_size] for offset in range(0, len(doc_ids), args.insert_batch_size)]
    worker_count = max(1, min(args.generation_clients, len(batches)))
    next_batch = [0]
    batch_lock = threading.Lock()
    ready_barrier = threading.Barrier(worker_count + 1)
    start_event = threading.Event()
    progress_lock = threading.Lock()
    progress = {
        "documents": 0,
        "next_report": max(args.progress_every, 1),
    }
    results = [PersistWorkerResult(worker_id=i) for i in range(worker_count)]
    threads = [
        threading.Thread(
            target=run_persist_worker,
            kwargs={
                "args": args,
                "batches": batches,
                "next_batch": next_batch,
                "batch_lock": batch_lock,
                "ready_barrier": ready_barrier,
                "start_event": start_event,
                "progress": progress,
                "progress_lock": progress_lock,
                "total_docs": len(doc_ids),
                "result": result,
            },
        )
        for result in results
    ]
    for thread in threads:
        thread.start()
    try:
        ready_barrier.wait()
    except threading.BrokenBarrierError:
        pass
    errors = [result.error for result in results if result.error]
    if errors:
        for thread in threads:
            thread.join()
        raise RuntimeError("parallel document multivector persistence failed: " + "; ".join(errors))
    started = time.perf_counter()
    start_event.set()
    for thread in threads:
        thread.join()

    errors = [result.error for result in results if result.error]
    if errors:
        raise RuntimeError("parallel document multivector persistence failed: " + "; ".join(errors))

    doc_latencies: list[float] = []
    batch_latencies: list[float] = []
    batch_sizes: list[int] = []
    doc_counts: list[int] = []
    for result in results:
        doc_latencies.extend(result.doc_latencies_ms)
        batch_latencies.extend(result.batch_latencies_ms)
        batch_sizes.extend(result.batch_sizes)
        doc_counts.extend(result.doc_vector_counts)

    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 3)
    return doc_latencies, doc_counts, {
        "workers": worker_count,
        "threads_per_worker": args.generation_threads,
        "n_gpu_layers": args.generation_n_gpu_layers,
        "warmup_excluded": bool(args.generation_warmup),
        "batches": sum(result.batches for result in results),
        "elapsed_ms": elapsed_ms,
        "documents_per_second": round(len(doc_counts) / (elapsed_ms / 1000.0), 3) if elapsed_ms > 0 else 0.0,
        "batch_latency": summarize_ms(batch_latencies),
        "batch_size": summarize_ints(batch_sizes),
    }


def persist_multivectors(
    conn: psycopg.Connection[Any],
    args: argparse.Namespace,
    doc_ids: list[str],
    query_ids: list[str],
) -> dict[str, Any]:
    if args.reuse_embeddings and not args.force_reload:
        row = fetch_one(
            conn,
            """
            SELECT
              count(*) FILTER (WHERE colbert IS NOT NULL),
              (SELECT count(*) FROM dbpedia_colbert_docs),
              (SELECT count(*) FILTER (WHERE colbert IS NOT NULL) FROM dbpedia_colbert_queries),
              (SELECT count(*) FROM dbpedia_colbert_queries)
            FROM dbpedia_colbert_docs
            """,
        )
        if row and int(row[0]) == int(row[1]) and int(row[2]) == int(row[3]) and int(row[1]) > 0:
            return {"reused": True, "documents": int(row[1]), "queries": int(row[3])}

    set_colbert_gucs(conn, args)
    existing_doc_embeddings, total_docs = count_existing_embeddings(conn)
    docs_to_encode = (
        ids_missing_embeddings(conn, "dbpedia_colbert_docs", "doc_id")
        if args.reuse_embeddings and not args.force_reload
        else doc_ids
    )
    doc_latencies, doc_counts, parallel_stats = persist_document_multivectors(args, docs_to_encode)

    query_latencies: list[float] = []
    query_counts: list[int] = []
    query_ids_to_encode = (
        ids_missing_embeddings(conn, "dbpedia_colbert_queries", "query_id")
        if args.reuse_embeddings and not args.force_reload
        else query_ids
    )
    for query_id in query_ids_to_encode:
        started = time.perf_counter()
        row = fetch_one(
            conn,
            """
            UPDATE dbpedia_colbert_queries
            SET colbert = colbert_mv(%s, query_text)
            WHERE query_id = %s
            RETURNING turbohybrid_multivector_count(colbert)
            """,
            (f"{args.model_alias}:query", query_id),
        )
        query_latencies.append((time.perf_counter() - started) * 1000.0)
        if row:
            query_counts.append(int(row[0]))

    return {
        "reused": False,
        "documents": summarize_ms(doc_latencies),
        "document_token_vectors": summarize_ints(doc_counts),
        "queries": summarize_ms(query_latencies),
        "query_token_vectors": summarize_ints(query_counts),
        "parallel_document_workers": parallel_stats["workers"],
        "parallel_document_threads_per_worker": parallel_stats["threads_per_worker"],
        "parallel_document_warmup_excluded": parallel_stats["warmup_excluded"],
        "parallel_document_batches": parallel_stats["batches"],
        "parallel_document_elapsed_ms": parallel_stats["elapsed_ms"],
        "parallel_document_docs_per_second": parallel_stats["documents_per_second"],
        "document_batch_latency": parallel_stats["batch_latency"],
        "document_batch_size": parallel_stats["batch_size"],
        "existing_document_embeddings": existing_doc_embeddings,
        "total_documents": total_docs,
        "generated_document_embeddings": len(doc_counts),
        "remaining_document_embeddings": max(total_docs - existing_doc_embeddings - len(doc_counts), 0),
        "generated_query_embeddings": len(query_counts),
        "note": "persisted phase measures PostgreSQL generation plus row storage/WAL",
    }


def load_precomputed_multivectors(conn: psycopg.Connection[Any], args: argparse.Namespace) -> dict[str, Any]:
    from dbpedia_colbert_hf_dataset import import_precomputed_dataset_to_postgres

    if args.reuse_data and not args.force_reload:
        row = fetch_one(
            conn,
            """
            SELECT
              coalesce((SELECT count(*) FROM dbpedia_colbert_docs), 0),
              coalesce((SELECT count(*) FILTER (WHERE colbert IS NOT NULL) FROM dbpedia_colbert_docs), 0),
              coalesce((SELECT count(*) FROM dbpedia_colbert_queries), 0),
              coalesce((SELECT count(*) FILTER (WHERE colbert IS NOT NULL) FROM dbpedia_colbert_queries), 0),
              coalesce((SELECT count(*) FROM dbpedia_colbert_qrels), 0),
              coalesce((
                SELECT count(*)
                FROM dbpedia_colbert_qrels q
                JOIN dbpedia_colbert_docs d ON d.doc_id = q.doc_id
                JOIN dbpedia_colbert_queries bq ON bq.query_id = q.query_id
              ), 0)
            """,
        )
        if (
            row
            and int(row[0]) > 0
            and int(row[0]) == int(row[1])
            and int(row[2]) > 0
            and int(row[2]) == int(row[3])
            and int(row[4]) > 0
            and int(row[4]) == int(row[5])
        ):
            return {
                "reused": True,
                "source": args.precomputed_dataset,
                "documents": int(row[0]),
                "queries": int(row[2]),
                "qrels": int(row[4]),
                "qrels_in_loaded_docs": int(row[5]),
            }

    return import_precomputed_dataset_to_postgres(
        conn=conn,
        source=args.precomputed_dataset,
        batch_size=args.precomputed_batch_size,
        max_docs=args.max_docs,
        max_queries=args.max_queries,
        force_reload=True,
        prioritize_qrels=args.prioritize_qrels,
    )


def build_index(conn: psycopg.Connection[Any], args: argparse.Namespace) -> dict[str, Any]:
    reloptions, index_option_values = build_index_reloptions(args)
    if args.reuse_index:
        row = fetch_one(conn, "SELECT to_regclass('dbpedia_colbert_docs_colbert_idx') IS NOT NULL")
        if row and bool(row[0]):
            stats = fetch_one(conn, "SELECT turbohybrid_index_stats('dbpedia_colbert_docs_colbert_idx'::regclass)")
            build_stats = fetch_one(conn, "SELECT turbohybrid_last_build_stats()")
            size = fetch_one(conn, "SELECT pg_relation_size('dbpedia_colbert_docs_colbert_idx'::regclass)")
            index_stats = jsonb_value(stats[0]) if stats else {}
            build_stats_value = jsonb_value(build_stats[0]) if build_stats else {}
            row_count = loaded_document_count(conn)
            return {
                "reused": True,
                "reloptions": reloptions,
                **index_option_values,
                "index_bytes": int(size[0]) if size else 0,
                "index_stats": index_stats,
                "build_stats": build_stats_value,
                "safety_checks": validate_document_node_proxy_build(
                    args=args,
                    index_stats=index_stats,
                    build_stats={},
                    row_count=row_count,
                ),
            }

    exec_sql(conn, "DROP INDEX IF EXISTS dbpedia_colbert_docs_colbert_idx")
    started = time.perf_counter()
    lexical_column = build_index_lexical_column(args)
    reloptions_sql = ",\n              ".join(reloptions)
    if lexical_column is None:
        exec_sql(
            conn,
            f"""
            CREATE INDEX dbpedia_colbert_docs_colbert_idx
            ON dbpedia_colbert_docs USING turbohybrid (
              colbert multivector_maxsim_ip_turbohybrid_ops
            )
            WITH ({reloptions_sql})
            """,
        )
    else:
        exec_sql(
            conn,
            f"""
            CREATE INDEX dbpedia_colbert_docs_colbert_idx
            ON dbpedia_colbert_docs USING turbohybrid (
              colbert multivector_maxsim_ip_turbohybrid_ops,
              {lexical_column} bm25_tsvector_turbohybrid_ops
            )
            WITH ({reloptions_sql})
            """,
        )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    build_stats = fetch_one(conn, "SELECT turbohybrid_last_build_stats()")
    stats = fetch_one(conn, "SELECT turbohybrid_index_stats('dbpedia_colbert_docs_colbert_idx'::regclass)")
    size = fetch_one(conn, "SELECT pg_relation_size('dbpedia_colbert_docs_colbert_idx'::regclass)")
    index_stats = jsonb_value(stats[0]) if stats else {}
    build_stats_value = jsonb_value(build_stats[0]) if build_stats else {}
    row_count = loaded_document_count(conn)
    return {
        "reused": False,
        "elapsed_ms": round(elapsed_ms, 3),
        "lexical_indexed": lexical_column is not None,
        "lexical_column": lexical_column,
        "reloptions": reloptions,
        **index_option_values,
        "index_bytes": int(size[0]) if size else 0,
        "index_stats": index_stats,
        "build_stats": build_stats_value,
        "safety_checks": validate_document_node_proxy_build(
            args=args,
            index_stats=index_stats,
            build_stats=build_stats_value,
            row_count=row_count,
        ),
    }


def load_encoded_queries(conn: psycopg.Connection[Any]) -> list[QueryItem]:
    rows = fetch_all(
        conn,
        """
        SELECT query_id, query_text
        FROM dbpedia_colbert_queries
        WHERE colbert IS NOT NULL
        ORDER BY query_id
        """,
    )
    queries = [QueryItem(str(row[0]), str(row[1])) for row in rows]
    if not queries:
        raise RuntimeError("no encoded query multivectors are available")
    return queries


def run_retrieval_query(
    conn: psycopg.Connection[Any],
    method: str,
    query: QueryItem,
    args: argparse.Namespace,
    final_k: int,
    dense_k: int | None = None,
) -> list[str]:
    if method == QUERY_ONLY_METHOD:
        if query_only_needs_text_query(args):
            text_query_expr = query_only_text_query_sql(args)
            rows = fetch_all(
                conn,
                f"""
                SELECT d.doc_id
                FROM dbpedia_colbert_docs d
                WHERE d.colbert IS NOT NULL
                ORDER BY d.colbert <~> (
                  SELECT turbohybrid_query(
                    multivector_query => q.colbert,
                    text_query => {text_query_expr},
                    dense_k => %s,
                    bm25_k => %s,
                    final_k => %s
                  )
                  FROM dbpedia_colbert_queries q
                  WHERE q.query_id = %s
                )
                LIMIT %s
                """,
                (
                    dense_k or args.dense_k,
                    args.bm25_k,
                    final_k,
                    query.query_id,
                    final_k,
                ),
            )
        else:
            rows = fetch_all(
                conn,
                """
                SELECT d.doc_id
                FROM dbpedia_colbert_docs d
                WHERE d.colbert IS NOT NULL
                ORDER BY d.colbert <~> (
                  SELECT turbohybrid_query(
                    multivector_query => q.colbert,
                    dense_k => %s,
                    final_k => %s
                  )
                  FROM dbpedia_colbert_queries q
                  WHERE q.query_id = %s
                )
                LIMIT %s
                """,
                (dense_k or args.dense_k, final_k, query.query_id, final_k),
            )
    elif method == EXACT_SCAN_METHOD:
        rows = fetch_all(
            conn,
            """
            SELECT d.doc_id
            FROM dbpedia_colbert_docs d
            JOIN dbpedia_colbert_queries q ON q.query_id = %s
            WHERE d.colbert IS NOT NULL
            ORDER BY turbohybrid_multivector_maxsim_distance(
                       q.colbert,
                       d.colbert
                     ),
                     d.doc_id
            LIMIT %s
            """,
            (query.query_id, final_k),
        )
    elif method == RRF_METHOD:
        rows = fetch_all(
            conn,
            """
            SELECT d.doc_id
            FROM dbpedia_colbert_docs d
            WHERE d.colbert IS NOT NULL
            ORDER BY d.colbert <~> (
              SELECT turbohybrid_query(
                multivector_query => q.colbert,
                text_query => websearch_to_tsquery('simple', q.query_text),
                fusion => 'rrf',
                dense_k => %s,
                bm25_k => %s,
                rrf_k => %s,
                final_k => %s
              )
              FROM dbpedia_colbert_queries q
              WHERE q.query_id = %s
            )
            LIMIT %s
            """,
            (dense_k or args.dense_k, args.bm25_k, args.rrf_k, final_k, query.query_id, final_k),
        )
    else:
        raise ValueError(f"unknown method: {method}")
    return [str(row[0]) for row in rows]


def uses_turbohybrid_index(method: str) -> bool:
    return method in {QUERY_ONLY_METHOD, RRF_METHOD}


def last_scan_stats(conn: psycopg.Connection[Any]) -> dict[str, Any]:
    row = fetch_one(conn, "SELECT turbohybrid_last_scan_stats()")
    return jsonb_value(row[0] if row else None)


def tid_parts(tid_text: str) -> tuple[int, int]:
    block, offset = tid_text.strip("()").split(",", 1)
    return int(block), int(offset)


def exact_admission_top(
    conn: psycopg.Connection[Any],
    query: QueryItem,
    admission_k: int,
) -> list[dict[str, Any]]:
    rows = fetch_all(
        conn,
        """
        SELECT d.doc_id,
               d.ctid::text,
               turbohybrid_multivector_maxsim_distance(q.colbert, d.colbert) AS distance
        FROM dbpedia_colbert_docs d
        JOIN dbpedia_colbert_queries q ON q.query_id = %s
        WHERE d.colbert IS NOT NULL
        ORDER BY turbohybrid_multivector_maxsim_distance(q.colbert, d.colbert),
                 d.doc_id
        LIMIT %s
        """,
        (query.query_id, admission_k),
    )
    result: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        block, offset = tid_parts(str(row[1]))
        result.append({
            "rank": rank,
            "doc_id": str(row[0]),
            "heap_block": block,
            "heap_offset": offset,
            "distance": float(row[2]),
            "maxsim": -float(row[2]),
        })
    return result


ExactTopProvider = Callable[
    [Any, QueryItem, int],
    list[dict[str, Any]],
]


def exact_top_cache_key(query_id: str, admission_k: int) -> str:
    return f"{admission_k}:{query_id}"


def validate_exact_top_cache_document_count(expected: int | None, current: int | None) -> None:
    if expected is not None and current is not None and expected != current:
        raise RuntimeError(
            "exact admission top cache document count mismatch: "
            f"expected {expected}, current {current}; reload or rebuild the serving grid run"
        )


def cached_exact_admission_top(
    conn: psycopg.Connection[Any],
    query: QueryItem,
    admission_k: int,
    exact_top_cache: dict[str, list[dict[str, Any]]] | None,
    exact_top_provider: ExactTopProvider = exact_admission_top,
) -> tuple[list[dict[str, Any]], float, bool]:
    key = exact_top_cache_key(query.query_id, admission_k)
    if exact_top_cache is not None and key in exact_top_cache:
        return exact_top_cache[key], 0.0, True
    exact_started = time.perf_counter()
    exact_top = exact_top_provider(conn, query, admission_k)
    exact_elapsed_ms = elapsed_ms_since(exact_started)
    if exact_top_cache is not None:
        exact_top_cache[key] = exact_top
    return exact_top, exact_elapsed_ms, False


def trace_key(entry: dict[str, Any]) -> tuple[int, int] | None:
    try:
        return int(entry["heap_block"]), int(entry["heap_offset"])
    except (KeyError, TypeError, ValueError):
        return None


def can_infer_admission_from_result_docs(
    candidate_source: Any,
    *,
    fallback_used: bool = False,
    plain_fallback: Any = None,
    graph_mode: Any = None,
) -> bool:
    """Return whether final result membership proves exact-rerank admission.

    Some document-level paths exact-rerank candidate documents but do not emit a
    bounded per-document admission trace. In those cases, a returned exact-top
    document proves it was admitted and retained for exact rerank, but not its
    pre-rerank candidate rank.
    """
    if fallback_used or plain_fallback == "force":
        return True
    if graph_mode == "document_nodes":
        return True
    return candidate_source in {
        "exact_doc_scan",
        "doc_graph_prototype",
        "plain_fallback",
        "document_nodes",
        "proxy_vector",
        "quantized_inverted_experimental",
    }


def scan_stat_int(stats: dict[str, Any], key: str) -> int:
    value = stats.get(key, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def scan_stat_float(stats: dict[str, Any], key: str) -> float:
    value = stats.get(key, 0.0)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    return result if math.isfinite(result) else 0.0


def scan_stat_bool(stats: dict[str, Any], key: str) -> bool:
    return stats.get(key) is True or stats.get(key) == "true"


def scan_stat_str(stats: dict[str, Any], key: str) -> str | None:
    value = stats.get(key)
    return value if isinstance(value, str) else None


def _serving_timing_number(value: Any, *, allow_zero: bool = True) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError, OverflowError):
        return None
    if number < 0:
        return None
    if not allow_zero and number == 0:
        return None
    return number


def _serving_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _nested_dense_timing_value(stats: dict[str, Any], key: str) -> int | None:
    dense = stats.get("dense", {})
    if not isinstance(dense, dict):
        return None
    timing = dense.get("timing_us", {})
    if not isinstance(timing, dict):
        return None
    return _serving_timing_number(timing.get(key))


def _flat_graph_timing_value(stats: dict[str, Any], key: str) -> int | None:
    return _serving_timing_number(stats.get(key))


def normalize_serving_phase_timings(stats: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy scan timing keys into serving phase timing names."""
    if not isinstance(stats, dict):
        stats = {}

    mapping = {
        "phase_total_time_us": ("total", "graph_total_us"),
        "phase_prepare_time_us": ("prepare", "graph_prepare_us"),
        "document_graph_traversal_time_us": ("traverse", "graph_traverse_us"),
        "document_graph_entry_time_us": ("entry", "graph_entry_us"),
        "document_graph_base_time_us": ("base", "graph_base_us"),
        "document_graph_batch_time_us": ("batch", "graph_batch_us"),
        "heap_time_us": ("heap", "graph_heap_us"),
        "fill_time_us": ("fill", "graph_fill_us"),
        "rescore_time_us": ("rescore", "graph_rescore_us"),
        "final_sort_time_us": ("sort", "graph_sort_us"),
        "entry_sidecar_time_us": ("entry_sidecar", "graph_entry_sidecar_us"),
        "payload_entry_seed_time_us": ("payload_entry_seed", "graph_payload_entry_seed_us"),
        "residual_rerank_time_us": ("residual_rerank", "graph_residual_rerank_us"),
        "exact_heap_fetch_time_us": ("heap_fetch", "graph_heap_fetch_us"),
        "local_expansion_time_us": ("local_expansion", "graph_local_expansion_us"),
        "scan_lock_wait_time_us": ("scan_lock_wait", "graph_scan_lock_wait_us"),
    }
    values: dict[str, int | None] = {}
    used_direct = False
    used_nested = False
    used_flat = False
    for canonical, (nested_key, flat_key) in mapping.items():
        direct_key = {
            "document_graph_traversal_time_us": (
                "multivector_document_graph_traversal_time_us"
            ),
            "exact_heap_fetch_time_us": "multivector_exact_heap_fetch_time_us",
            "final_sort_time_us": "multivector_final_sort_time_us",
        }.get(canonical)
        direct_value = (
            _serving_timing_number(stats.get(direct_key))
            if direct_key is not None
            else None
        )
        if direct_value is not None:
            values[canonical] = direct_value
            used_direct = True
            continue
        nested_value = _nested_dense_timing_value(stats, nested_key)
        if nested_value is not None:
            values[canonical] = nested_value
            used_nested = True
            continue
        flat_value = _flat_graph_timing_value(stats, flat_key)
        if flat_value is not None:
            values[canonical] = flat_value
            used_flat = True
        else:
            values[canonical] = None

    heap_rescore = _serving_timing_number(
        stats.get("multivector_exact_maxsim_rerank_time_us")
    )
    if heap_rescore is not None:
        used_direct = True
    if heap_rescore is None:
        heap_rescore = _nested_dense_timing_value(stats, "heap_rescore")
    if heap_rescore is None:
        heap_rescore = _flat_graph_timing_value(stats, "graph_heap_rescore_us")
    rescore = values.get("rescore_time_us")
    values["exact_maxsim_rerank_time_us"] = (
        heap_rescore if heap_rescore is not None else rescore
    )
    if heap_rescore is not None:
        used_nested = used_nested or _nested_dense_timing_value(stats, "heap_rescore") is not None
        used_flat = used_flat or _flat_graph_timing_value(stats, "graph_heap_rescore_us") is not None

    true_timing_fields = {
        "sidecar_load_time_us": (
            "multivector_sidecar_load_time_us",
            "multivector_sidecar_load_us",
            "sidecar_load_us",
        ),
        "sidecar_page_read_time_us": (
            "multivector_sidecar_page_read_time_us",
            "sidecar_page_read_time_us",
        ),
        "sidecar_vector_reconstruct_time_us": (
            "multivector_sidecar_vector_reconstruct_time_us",
            "sidecar_vector_reconstruct_time_us",
        ),
        "proxy_candidate_time_us": (
            "multivector_proxy_candidate_time_us",
            "proxy_candidate_time_us",
        ),
        "proxy_graph_traversal_time_us": (
            "multivector_proxy_graph_traversal_time_us",
            "proxy_graph_traversal_time_us",
        ),
        "proxy_scoring_time_us": (
            "multivector_proxy_scoring_time_us",
            "multivector_proxy_scoring_us",
            "proxy_scoring_us",
        ),
        "centroid_lite_time_us": (
            "multivector_centroid_lite_time_us",
            "multivector_centroid_lite_posting_us",
            "centroid_lite_time_us",
        ),
        "quantized_inverted_time_us": (
            "multivector_quantized_inverted_time_us",
            "multivector_quantized_inverted_posting_us",
            "quantized_inverted_time_us",
        ),
    }
    for canonical, source_keys in true_timing_fields.items():
        value = None
        for key in source_keys:
            value = _serving_timing_number(stats.get(key))
            if value is not None:
                used_direct = True
                break
        values[canonical] = value

    total_value = values.get("phase_total_time_us")
    if total_value in (None, 0):
        dense = stats.get("dense", {})
        dense_elapsed = (
            _serving_timing_number(dense.get("elapsed_us"), allow_zero=False)
            if isinstance(dense, dict)
            else None
        )
        total_value = (
            dense_elapsed
            or _serving_timing_number(stats.get("dense_elapsed_us"), allow_zero=False)
            or _serving_timing_number(stats.get("elapsed_us"), allow_zero=False)
            or total_value
        )
        values["phase_total_time_us"] = total_value

    source = (
        "direct_multivector_timing"
        if used_direct
        else "nested_dense_timing_us"
        if used_nested
        else "flat_graph_timing_us"
        if used_flat
        else "none"
    )
    non_overlapping = [
        "phase_prepare_time_us",
        "document_graph_traversal_time_us",
        "document_graph_entry_time_us",
        "document_graph_base_time_us",
        "document_graph_batch_time_us",
        "heap_time_us",
        "fill_time_us",
        "final_sort_time_us",
        "entry_sidecar_time_us",
        "payload_entry_seed_time_us",
        "residual_rerank_time_us",
        "exact_heap_fetch_time_us",
        "exact_maxsim_rerank_time_us",
        "local_expansion_time_us",
        "scan_lock_wait_time_us",
        "sidecar_load_time_us",
        "proxy_scoring_time_us",
        "centroid_lite_time_us",
        "quantized_inverted_time_us",
    ]
    if heap_rescore is not None:
        non_overlapping.append("rescore_time_us")
    known_total = sum(
        int(values[key])
        for key in non_overlapping
        if isinstance(values.get(key), int)
    )
    observed_latency_ms = _serving_float(stats.get("observed_latency_ms"))
    if observed_latency_ms is None:
        observed_latency_ms = _serving_float(stats.get("latency_ms"))
    unattributed_ms = None
    if observed_latency_ms is not None and isinstance(total_value, int):
        unattributed_ms = round(observed_latency_ms - (total_value / 1000.0), 3)

    canonical: dict[str, Any] = {
        **values,
        "phase_timing_source": source,
        "phase_timing_missing": [
            key
            for key, value in values.items()
            if value is None
        ],
        "phase_timing_known_total_us": known_total,
        "phase_timing_unattributed_sql_ms": unattributed_ms,
    }
    return canonical


def memory_estimate_from_stats(stats: dict[str, Any]) -> int:
    return scan_stat_int(stats, "multivector_memory_bytes_estimate")


def scan_docs_scored(stats: dict[str, Any]) -> int:
    return max(
        scan_stat_int(stats, "multivector_doc_graph_docs_scored"),
        scan_stat_int(stats, "multivector_plain_fallback_docs_scored"),
        scan_stat_int(stats, "multivector_doc_candidates"),
        scan_stat_int(stats, "multivector_unique_docs"),
    )


def scan_graph_edges_visited(stats: dict[str, Any]) -> int:
    return scan_stat_int(stats, "multivector_doc_graph_edges_visited")


def scan_exact_rerank_docs(stats: dict[str, Any]) -> int:
    return max(
        scan_stat_int(stats, "multivector_exact_rerank_docs"),
        scan_stat_int(stats, "multivector_doc_graph_exact_rerank_docs"),
    )


def serving_simd_expected() -> bool:
    forced_values = {
        os.environ.get("SIMD_BUILD", ""),
        os.environ.get("PGTURBOHYBRID_SIMD_BUILD", ""),
        os.environ.get("PGTURBOHYBRID_DENSE_EXACT_SIMD_FORCE", ""),
        os.environ.get("PGTURBOHYBRID_DENSE_SIMD_FORCE", ""),
    }
    return not any(str(value).lower() in {"none", "off", "scalar"} for value in forced_values)


def validate_serving_scan_path(
    stats: dict[str, Any],
    args: argparse.Namespace,
    profile: DocumentNodeServingProfile,
) -> list[str]:
    warnings: list[str] = []
    if not isinstance(stats, dict):
        return ["missing_scan_stats"]

    expected_source = profile.candidate_source
    observed_source = scan_stat_str(stats, "multivector_candidate_source")
    if observed_source is not None and observed_source != expected_source:
        warnings.append(
            "candidate_source_mismatch:"
            f"expected={expected_source},observed={observed_source}"
        )

    if scan_stat_bool(stats, "multivector_plain_fallback_used"):
        warnings.append("plain_fallback_used")
    fallback_reason = scan_stat_str(stats, "multivector_plain_fallback_reason")
    if fallback_reason not in (None, "", "not_applicable"):
        warnings.append(f"plain_fallback_reason:{fallback_reason}")

    doc_warning = scan_stat_str(stats, "multivector_doc_graph_warning")
    if (
        doc_warning is not None
        and "document_node_f32_sidecar_exact_scan" in doc_warning
        and expected_source != "exact_doc_scan"
    ):
        warnings.append(f"doc_graph_warning:{doc_warning}")

    exact_rerank_docs = scan_exact_rerank_docs(stats)
    configured_rerank_k = int(
        getattr(
            args,
            "multivector_exact_rerank_k",
            getattr(args, "serving_exact_rerank_k", 0),
        )
        or 0
    )
    rerank_tolerance = max(2, int(math.ceil(max(configured_rerank_k, 1) * 0.05)))
    if configured_rerank_k > 0 and exact_rerank_docs > configured_rerank_k + rerank_tolerance:
        warnings.append(
            "exact_rerank_docs_exceeds_serving_budget:"
            f"observed={exact_rerank_docs},limit={configured_rerank_k},tolerance={rerank_tolerance}"
        )

    loaded_document_count = int(getattr(args, "serving_loaded_document_count", 0) or 0)
    docs_scored = scan_docs_scored(stats)
    doc_graph_docs_scored = scan_stat_int(stats, "multivector_doc_graph_docs_scored")
    if loaded_document_count > 0:
        near_threshold = max(1, int(math.floor(loaded_document_count * 0.90)))
        if docs_scored >= near_threshold:
            warnings.append(
                "docs_scored_near_table_size:"
                f"observed={docs_scored},table={loaded_document_count}"
            )
        if doc_graph_docs_scored >= near_threshold:
            warnings.append(
                "doc_graph_docs_scored_near_table_size:"
                f"observed={doc_graph_docs_scored},table={loaded_document_count}"
            )
        if expected_source == "proxy_vector":
            candidate_k = max(
                scan_stat_int(stats, "multivector_proxy_candidate_target"),
                scan_stat_int(stats, "multivector_doc_graph_candidates"),
                scan_stat_int(stats, "proxy_candidates"),
                int(getattr(args, "serving_candidate_k", 0) or 0),
                int(getattr(args, "multivector_doc_candidate_k", 0) or 0),
            )
            sidecar_vectors_loaded = scan_stat_int(
                stats, "multivector_doc_sidecar_vectors_loaded"
            )
            resident_vectors_loaded = scan_stat_int(
                stats, "multivector_doc_sidecar_resident_vectors_loaded"
            )
            sidecar_bytes_touched = scan_stat_int(
                stats, "multivector_doc_sidecar_bytes_touched"
            )
            sidecar_docmap_bytes = scan_stat_int(stats, "multivector_docmap_bytes")
            docmap_bytes_touched = scan_stat_int(
                stats, "multivector_doc_sidecar_docmap_bytes_touched"
            )
            bounded_proxy_candidate_band = candidate_k > 0 and candidate_k < near_threshold
            if bounded_proxy_candidate_band and (
                sidecar_vectors_loaded >= near_threshold
                or resident_vectors_loaded >= near_threshold
                or doc_graph_docs_scored >= near_threshold
            ):
                warnings.append(
                    "proxy_vector_near_exhaustive_sidecar_touch:"
                    f"candidate_k={candidate_k},table={loaded_document_count},"
                    f"sidecar_vectors_loaded={sidecar_vectors_loaded},"
                    f"resident_vectors_loaded={resident_vectors_loaded},"
                    f"docs_scored={doc_graph_docs_scored}"
                )
            if (
                bounded_proxy_candidate_band
                and sidecar_docmap_bytes > 0
                and max(sidecar_bytes_touched, docmap_bytes_touched)
                >= int(math.floor(sidecar_docmap_bytes * 0.90))
            ):
                warnings.append(
                    "proxy_vector_near_exhaustive_sidecar_touch:"
                    f"candidate_k={candidate_k},table={loaded_document_count},"
                    f"sidecar_bytes_touched={sidecar_bytes_touched},"
                    f"docmap_bytes_touched={docmap_bytes_touched},"
                    f"docmap_bytes={sidecar_docmap_bytes}"
                )

    observed_cache_mode = scan_stat_str(stats, "multivector_doc_sidecar_cache_mode")
    pages_read = scan_stat_int(stats, "multivector_doc_sidecar_pages_read")
    if (
        profile.cache_mode in {"auto", "resident"}
        and observed_cache_mode == "paged"
        and pages_read > max(128, int(max(loaded_document_count, 0) * 0.01))
    ):
        warnings.append(
            "paged_sidecar_high_pages_read:"
            f"expected={profile.cache_mode},observed=paged,pages_read={pages_read}"
        )

    exact_kernel = scan_stat_str(stats, "multivector_exact_kernel")
    if exact_kernel in {"scalar", "blocked_scalar"} and serving_simd_expected():
        warnings.append(f"exact_kernel_scalar:{exact_kernel}")

    if scan_stat_bool(stats, "native_cache_built_this_scan"):
        warnings.append("native_cache_built_this_scan")
    return warnings


def unique_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def fail_on_serving_slow_path_if_requested(
    warnings: list[str],
    args: argparse.Namespace,
) -> None:
    if warnings and bool(getattr(args, "serving_fail_on_slow_path", False)):
        raise RuntimeError("serving slow path detected: " + "; ".join(warnings))


def exact_rerank_work_from_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "exact_rerank_candidates": scan_stat_int(stats, "exact_rerank_candidates"),
        "exact_rerank_pairs": scan_stat_int(stats, "multivector_exact_rerank_pairs"),
        "exact_rerank_tokens_evaluated": scan_stat_int(stats, "exact_rerank_tokens_evaluated"),
        "exact_rerank_tokens_skipped": scan_stat_int(stats, "exact_rerank_tokens_skipped"),
        "exact_rerank_pairs_saved": scan_stat_int(stats, "exact_rerank_pairs_saved"),
        "adaptive_rerank_topk_changed_vs_full": bool(
            stats.get("adaptive_rerank_topk_changed_vs_full", False)
        ),
    }


def token_pooling_work_from_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "multivector_tokens_original": scan_stat_int(stats, "multivector_tokens_original"),
        "multivector_tokens_pooled": scan_stat_int(stats, "multivector_tokens_pooled"),
        "multivector_token_pooling_ratio": scan_stat_float(
            stats,
            "multivector_token_pooling_ratio",
        ),
    }


def proxy_work_from_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "proxy_encoder_kind": scan_stat_str(stats, "proxy_encoder_kind"),
        "proxy_candidates": scan_stat_int(stats, "proxy_candidates"),
        "proxy_top1_admission": scan_stat_bool(stats, "proxy_top1_admission"),
        "proxy_exact_rerank_docs": scan_stat_int(stats, "proxy_exact_rerank_docs"),
    }


def centroid_work_from_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "centroid_lists_visited": scan_stat_int(stats, "centroid_lists_visited"),
        "centroid_docs_touched": scan_stat_int(stats, "centroid_docs_touched"),
        "centroid_pruned_docs": scan_stat_int(stats, "centroid_pruned_docs"),
        "centroid_candidates": scan_stat_int(stats, "centroid_candidates"),
    }


def quantized_inverted_work_from_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "quantized_inverted_lists_visited": scan_stat_int(
            stats, "quantized_inverted_lists_visited"
        ),
        "quantized_inverted_postings_touched": scan_stat_int(
            stats, "quantized_inverted_postings_touched"
        ),
        "quantized_inverted_docs_scored": scan_stat_int(
            stats, "quantized_inverted_docs_scored"
        ),
        "quantized_inverted_candidates": scan_stat_int(
            stats, "quantized_inverted_candidates"
        ),
        "quantized_inverted_exact_rerank_docs": scan_stat_int(
            stats, "quantized_inverted_exact_rerank_docs"
        ),
        "quantized_inverted_codebook_source": scan_stat_str(
            stats, "quantized_inverted_codebook_source"
        ),
        "quantized_inverted_codebook_size": scan_stat_int(
            stats, "quantized_inverted_codebook_size"
        ),
        "quantized_inverted_codebook_dim": scan_stat_int(
            stats, "quantized_inverted_codebook_dim"
        ),
        "quantized_inverted_codebook_checksum": scan_stat_str(
            stats, "quantized_inverted_codebook_checksum"
        ),
        "quantized_inverted_codebook_top_m": scan_stat_int(
            stats, "quantized_inverted_codebook_top_m"
        ),
        "quantized_inverted_assignment_us": scan_stat_int(
            stats, "quantized_inverted_assignment_us"
        ),
        "quantized_inverted_list_offset_bytes": scan_stat_int(
            stats, "quantized_inverted_list_offset_bytes"
        ),
        "quantized_inverted_posting_bytes": scan_stat_int(
            stats, "quantized_inverted_posting_bytes"
        ),
        "quantized_inverted_sidecar_bytes": scan_stat_int(
            stats, "quantized_inverted_sidecar_bytes"
        ),
        "quantized_inverted_compact_kernel": scan_stat_str(
            stats, "quantized_inverted_compact_kernel"
        ),
        "quantized_inverted_compact_score_us": scan_stat_int(
            stats, "quantized_inverted_compact_score_us"
        ),
        "quantized_inverted_compact_docs_scored": scan_stat_int(
            stats, "quantized_inverted_compact_docs_scored"
        ),
        "quantized_inverted_compact_payload_bytes": scan_stat_int(
            stats, "quantized_inverted_compact_payload_bytes"
        ),
        "quantized_inverted_compact_topk_changed_vs_scalar": scan_stat_bool(
            stats, "quantized_inverted_compact_topk_changed_vs_scalar"
        ),
    }


def learned_sparse_work_from_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "learned_sparse_candidates": scan_stat_int(stats, "learned_sparse_candidates"),
        "learned_sparse_retained_for_maxsim": scan_stat_int(
            stats, "learned_sparse_retained_for_maxsim"
        ),
        "learned_sparse_branch_latency_us": scan_stat_int(
            stats, "learned_sparse_branch_latency_us"
        ),
    }


def bm25_rescue_work_from_stats(stats: dict[str, Any]) -> dict[str, Any]:
    return {
        "bm25_injection_enabled": scan_stat_bool(
            stats, "multivector_bm25_injection_enabled"
        ),
        "bm25_injection_candidates": scan_stat_int(
            stats, "multivector_bm25_injection_candidates"
        ),
        "bm25_injection_candidate_limit": scan_stat_int(
            stats, "multivector_bm25_injection_candidate_limit"
        ),
        "bm25_injection_pool_size": scan_stat_int(
            stats, "multivector_bm25_injection_pool_size"
        ),
        "bm25_injection_limit_reason": scan_stat_str(
            stats, "multivector_bm25_injection_limit_reason"
        ),
        "bm25_injection_retained": scan_stat_int(
            stats, "multivector_bm25_injection_retained"
        ),
        "bm25_injection_exact_reranked": scan_stat_int(
            stats, "multivector_bm25_injection_exact_reranked"
        ),
    }


def is_experimental_quantized_stat_key(key: str) -> bool:
    lowered = key.lower()
    return (
        lowered.startswith("quantized_")
        or lowered.startswith("quantized_inverted_")
        or "codeword" in lowered
        or "posting" in lowered
    )


def _nested_dense_cache(stats: dict[str, Any]) -> dict[str, Any]:
    dense = stats.get("dense", {})
    if not isinstance(dense, dict):
        return {}
    cache = dense.get("cache", {})
    return cache if isinstance(cache, dict) else {}


def _set_if_absent(target: dict[str, Any], key: str, value: Any) -> None:
    if key not in target and value is not None:
        target[key] = value


def _cache_bool(cache: dict[str, Any], key: str) -> bool | None:
    if key not in cache:
        return None
    value = cache.get(key)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "on", "yes", "1"}:
            return True
        if lowered in {"false", "off", "no", "0"}:
            return False
    return None


def _cache_int(cache: dict[str, Any], key: str) -> int | None:
    return _serving_timing_number(cache.get(key))


def _derive_serving_stats_from_nested_cache(
    stats: dict[str, Any],
    extracted: dict[str, Any],
) -> None:
    """Backfill flat serving fields from legacy nested dense.cache stats."""
    cache = _nested_dense_cache(stats)
    if not cache:
        return

    for key in SERVING_STATS_FIELD_GROUPS["storage_cache"]:
        _set_if_absent(extracted, key, cache.get(key))

    native_cache_built = _cache_bool(cache, "native_cache_built_this_scan")
    native_cache_reused = _cache_bool(cache, "native_cache_reused")
    sidecar_bytes = _cache_int(cache, "multivector_doc_sidecar_bytes_touched") or 0
    sidecar_pages = _cache_int(cache, "multivector_doc_sidecar_pages_read") or 0
    vectors_loaded = (
        (_cache_int(cache, "multivector_doc_sidecar_vectors_loaded") or 0)
        + (_cache_int(cache, "multivector_doc_sidecar_resident_vectors_loaded") or 0)
    )
    page_read_us = _serving_timing_number(
        stats.get("multivector_sidecar_page_read_time_us")
    ) or _serving_timing_number(cache.get("multivector_sidecar_page_read_time_us")) or 0
    reconstruct_us = _serving_timing_number(
        stats.get("multivector_sidecar_vector_reconstruct_time_us")
    ) or _serving_timing_number(cache.get("multivector_sidecar_vector_reconstruct_time_us")) or 0
    sidecar_time_us = page_read_us + reconstruct_us

    if native_cache_built is not None:
        _set_if_absent(extracted, "sidecar_cache_build_this_query", native_cache_built)
    if native_cache_built:
        _set_if_absent(extracted, "sidecar_cache_build_bytes", sidecar_bytes)
        _set_if_absent(extracted, "sidecar_cache_build_pages_read", sidecar_pages)
        _set_if_absent(extracted, "sidecar_cache_build_time_us", sidecar_time_us)
        _set_if_absent(extracted, "sidecar_query_bytes_touched", 0)
        _set_if_absent(extracted, "sidecar_query_pages_read", 0)
        _set_if_absent(extracted, "sidecar_query_vectors_loaded", 0)
        _set_if_absent(extracted, "sidecar_query_load_time_us", 0)
        _set_if_absent(extracted, "sidecar_query_time_us", 0)
    elif native_cache_built is False or native_cache_reused is True:
        _set_if_absent(extracted, "sidecar_cache_build_bytes", 0)
        _set_if_absent(extracted, "sidecar_cache_build_pages_read", 0)
        _set_if_absent(extracted, "sidecar_cache_build_time_us", 0)
        _set_if_absent(extracted, "sidecar_query_bytes_touched", sidecar_bytes)
        _set_if_absent(extracted, "sidecar_query_pages_read", sidecar_pages)
        _set_if_absent(extracted, "sidecar_query_vectors_loaded", vectors_loaded)
        _set_if_absent(extracted, "sidecar_query_load_time_us", sidecar_time_us)
        _set_if_absent(extracted, "sidecar_query_time_us", sidecar_time_us)

    if (
        stats.get("multivector_candidate_source") == "proxy_vector"
        and stats.get("multivector_candidate_path") == "proxy_graph"
    ):
        _set_if_absent(extracted, "proxy_vector_uses_full_sidecar_for_graph", False)
        _set_if_absent(extracted, "proxy_full_sidecar_vectors_loaded", 0)
        _set_if_absent(extracted, "proxy_full_sidecar_bytes_touched", 0)
        _set_if_absent(extracted, "proxy_full_sidecar_pages_read", 0)
        _set_if_absent(extracted, "proxy_full_sidecar_load_time_us", 0)
        _set_if_absent(extracted, "proxy_full_sidecar_reconstruct_time_us", 0)

        observed_candidates = max(
            scan_stat_int(stats, "proxy_candidates"),
            scan_stat_int(stats, "multivector_doc_graph_candidates"),
        )
        search_ef = scan_stat_int(stats, "multivector_doc_graph_search_ef")
        branch_limits = scan_stat_int_list(stats, "branch_candidate_limits")
        target = branch_limits[0] if branch_limits else scan_stat_int(
            stats, "multivector_proxy_candidate_target"
        )
        if observed_candidates:
            _set_if_absent(
                extracted,
                "proxy_candidate_limit_effective",
                observed_candidates,
            )
            if search_ef and observed_candidates == search_ef and (
                target == 0 or target > observed_candidates
            ):
                source = "search_ef"
            elif target and observed_candidates == target:
                source = "candidate_target"
            else:
                source = "observed_candidates"
            _set_if_absent(extracted, "proxy_candidate_limit_source", source)

        doc_count = scan_stat_int(stats, "multivector_doc_graph_nodes")
        near_exhaustive = (
            bool(doc_count)
            and bool(vectors_loaded)
            and vectors_loaded * 10 >= doc_count * 9
            and bool(target)
            and target * 4 < doc_count
        )
        _set_if_absent(
            extracted,
            "proxy_vector_near_exhaustive_sidecar_touch",
            near_exhaustive,
        )
        if near_exhaustive:
            reason = (
                "resident_cache_build_all_docs"
                if native_cache_built
                else "resident_cache_query_materialize_all_docs"
            )
        else:
            reason = "none"
        _set_if_absent(extracted, "proxy_vector_sidecar_touch_reason", reason)


def extract_document_node_serving_stats(stats: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(stats, dict):
        return {}
    fields = {
        key
        for group_fields in SERVING_STATS_FIELD_GROUPS.values()
        for key in group_fields
    }
    extracted = {key: stats[key] for key in fields if key in stats}
    extracted.update(normalize_serving_phase_timings(stats))
    _derive_serving_stats_from_nested_cache(stats, extracted)
    for key, value in stats.items():
        if is_experimental_quantized_stat_key(str(key)):
            extracted[str(key)] = value
    return dict(sorted(extracted.items()))


def document_node_serving_stats_available(extracted: dict[str, Any]) -> dict[str, bool]:
    if not isinstance(extracted, dict):
        extracted = {}
    available = {
        group: any(key in extracted for key in fields)
        for group, fields in SERVING_STATS_FIELD_GROUPS.items()
    }
    available["phase_timing"] = extracted.get("phase_timing_source") not in (None, "", "none")
    available["experimental_quantized"] = any(
        is_experimental_quantized_stat_key(str(key))
        for key in extracted
    )
    return available


def merge_document_node_serving_stats_available(
    samples_by_budget: dict[str, dict[str, Any]]
) -> dict[str, bool]:
    merged = {
        group: False
        for group in (*SERVING_STATS_FIELD_GROUPS.keys(), "experimental_quantized")
    }
    for sample in samples_by_budget.values():
        if not isinstance(sample, dict):
            continue
        available = sample.get("stats_available", {})
        if not isinstance(available, dict):
            continue
        for group, value in available.items():
            merged[str(group)] = bool(merged.get(str(group), False) or value)
    return dict(sorted(merged.items()))


def scan_stat_int_list(stats: dict[str, Any], key: str) -> list[int]:
    value = stats.get(key)
    if not isinstance(value, list):
        return []
    result: list[int] = []
    for item in value:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            continue
    return result


def clone_args(args: argparse.Namespace, **overrides: Any) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(overrides)
    return argparse.Namespace(**values)


def sidecar_stats_from_scan(stats: dict[str, Any]) -> dict[str, Any]:
    dedicated_keys = (
        "multivector_doc_sidecar_cache_mode",
        "multivector_doc_sidecar_pages_read",
        "multivector_doc_sidecar_cache_hits",
        "multivector_doc_sidecar_cache_misses",
        "multivector_doc_sidecar_bytes_touched",
        "multivector_doc_sidecar_vectors_loaded",
        "multivector_doc_sidecar_docmap_pages_read",
        "multivector_doc_sidecar_docmap_bytes_touched",
        "multivector_doc_sidecar_resident_vectors_loaded",
        "multivector_doc_sidecar_resident_bytes_loaded",
        "multivector_doc_sidecar_vector_chunk_ref_bytes_touched",
        "multivector_doc_sidecar_paged_vector_pages_read",
        "multivector_doc_sidecar_paged_vector_bytes_touched",
    )
    return {
        "sidecar_stats_available": any(key in stats for key in dedicated_keys),
        "sidecar_cache_mode": str(stats.get("multivector_doc_sidecar_cache_mode", "none")),
        "sidecar_pages_read": scan_stat_int(stats, "multivector_doc_sidecar_pages_read"),
        "sidecar_bytes_touched": scan_stat_int(stats, "multivector_doc_sidecar_bytes_touched"),
        "sidecar_cache_hits": scan_stat_int(stats, "multivector_doc_sidecar_cache_hits"),
        "sidecar_cache_misses": scan_stat_int(stats, "multivector_doc_sidecar_cache_misses"),
        "sidecar_vectors_loaded": scan_stat_int(stats, "multivector_doc_sidecar_vectors_loaded"),
        "sidecar_docmap_pages_read": scan_stat_int(
            stats, "multivector_doc_sidecar_docmap_pages_read"
        ),
        "sidecar_docmap_bytes_touched": scan_stat_int(
            stats, "multivector_doc_sidecar_docmap_bytes_touched"
        ),
        "sidecar_resident_vectors_loaded": scan_stat_int(
            stats, "multivector_doc_sidecar_resident_vectors_loaded"
        ),
        "sidecar_resident_bytes_loaded": scan_stat_int(
            stats, "multivector_doc_sidecar_resident_bytes_loaded"
        ),
        "sidecar_vector_chunk_ref_bytes_touched": scan_stat_int(
            stats, "multivector_doc_sidecar_vector_chunk_ref_bytes_touched"
        ),
        "sidecar_paged_vector_pages_read": scan_stat_int(
            stats, "multivector_doc_sidecar_paged_vector_pages_read"
        ),
        "sidecar_paged_vector_bytes_touched": scan_stat_int(
            stats, "multivector_doc_sidecar_paged_vector_bytes_touched"
        ),
        "native_cache_used": scan_stat_bool(stats, "native_cache_used"),
        "native_cache_reused": scan_stat_bool(stats, "native_cache_reused"),
        "native_cache_built_this_scan": scan_stat_bool(stats, "native_cache_built_this_scan"),
        "native_cache_bytes": scan_stat_int(stats, "native_cache_bytes"),
        "native_cache_exact_bytes": scan_stat_int(stats, "native_cache_exact_bytes"),
        "native_cache_code_bytes": scan_stat_int(stats, "native_cache_code_bytes"),
        "native_cache_adj_bytes": scan_stat_int(stats, "native_cache_adj_bytes"),
        "native_cache_mode": scan_stat_str(stats, "native_cache_mode"),
        "native_cache_scope": scan_stat_str(stats, "native_cache_scope"),
        "memory_bytes_estimate": memory_estimate_from_stats(stats),
    }


def parse_int_grid(value: str, option_name: str) -> list[int]:
    try:
        items = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise SystemExit(f"{option_name} must contain comma-separated integers") from exc
    items = sorted({item for item in items if item > 0})
    if not items:
        raise SystemExit(f"{option_name} must contain at least one positive integer")
    return items


def effective_serving_grid_budget_sweep(args: argparse.Namespace) -> list[int]:
    if bool(getattr(args, "admission_budget_sweep_explicit", False)):
        source = str(getattr(args, "admission_budget_sweep", "") or "")
    elif bool(getattr(args, "document_node_serving_grid_proxy_admission_focus", False)) or bool(
        getattr(args, "document_node_serving_grid_centroid_lite_focus", False)
    ):
        source = str(DOCUMENT_NODE_SERVING_LATENCY_DEFAULT_CANDIDATE_K)
    elif bool(getattr(args, "document_node_serving_grid_smoke", False)):
        source = ",".join(str(budget) for budget in DOCUMENT_NODE_SERVING_GRID_SMOKE_BUDGETS)
    elif (
        bool(getattr(args, "document_node_serving_grid", False))
        or bool(getattr(args, "document_node_serving_build_only", False))
    ):
        source = DOCUMENT_NODE_SERVING_GRID_BUDGET_SWEEP
    else:
        source = DEFAULT_ADMISSION_BUDGET_SWEEP
    return parse_int_grid(source, "--admission-budget-sweep")


def effective_document_node_serving_centroid_lite_caps(args: argparse.Namespace) -> list[int]:
    source = str(
        getattr(
            args,
            "document_node_serving_grid_centroid_lite_posting_caps",
            DOCUMENT_NODE_SERVING_GRID_CENTROID_LITE_CAP_SWEEP,
        )
        or DOCUMENT_NODE_SERVING_GRID_CENTROID_LITE_CAP_SWEEP
    )
    return parse_int_grid(source, "--document-node-serving-grid-centroid-lite-posting-caps")


def effective_document_node_serving_entry_sample_counts(args: argparse.Namespace) -> list[int]:
    source = str(
        getattr(
            args,
            "document_node_serving_grid_entry_sample_counts",
            DOCUMENT_NODE_SERVING_GRID_ENTRY_SAMPLE_SWEEP,
        )
        or DOCUMENT_NODE_SERVING_GRID_ENTRY_SAMPLE_SWEEP
    )
    return parse_int_grid(source, "--document-node-serving-grid-entry-sample-counts")


def effective_document_node_serving_entry_sidecar_representatives(args: argparse.Namespace) -> list[int]:
    source = str(
        getattr(
            args,
            "document_node_serving_grid_entry_sidecar_representatives",
            DOCUMENT_NODE_SERVING_GRID_ENTRY_SIDECAR_SWEEP,
        )
        or DOCUMENT_NODE_SERVING_GRID_ENTRY_SIDECAR_SWEEP
    )
    return parse_int_grid(source, "--document-node-serving-grid-entry-sidecar-representatives")


def effective_serving_grid_executed_budgets(args: argparse.Namespace) -> list[int]:
    budget_sweep = effective_serving_grid_budget_sweep(args)
    mode = str(getattr(args, "document_node_serving_grid_budget_mode", "sweep") or "sweep")
    if mode == "largest_only":
        return [max(budget_sweep)]
    if mode == "sweep":
        return budget_sweep
    raise SystemExit("--document-node-serving-grid-budget-mode must be largest_only or sweep")


def effective_document_node_serving_stage_mode(args: argparse.Namespace) -> str:
    mode = getattr(args, "document_node_serving_grid_stage_mode", None)
    if mode is None:
        return "two_stage" if bool(getattr(args, "document_node_serving_grid_smoke", False)) else "single"
    mode = str(mode)
    if mode not in {"single", "two_stage"}:
        raise SystemExit("--document-node-serving-grid-stage-mode must be single or two_stage")
    return mode


def effective_admission_exact_rerank_k(args: argparse.Namespace, budget: int) -> int:
    mode = str(getattr(args, "serving_exact_rerank_mode", "admission_exhaustive") or "admission_exhaustive")
    if mode == "serving":
        return min(int(getattr(args, "serving_exact_rerank_k", 100)), int(budget))
    if mode == "admission_exhaustive":
        return max(int(getattr(args, "multivector_exact_rerank_k", 0)), int(budget))
    raise SystemExit("--serving-exact-rerank-mode must be serving or admission_exhaustive")


def effective_admission_debug_mode(args: argparse.Namespace) -> str:
    mode = getattr(args, "admission_debug_mode", None)
    if mode is not None:
        return str(mode)
    context = str(getattr(args, "admission_debug_context", "") or "")
    if context == "serving_grid":
        return "summary"
    return "trace"


def parse_choice_grid(value: str, choices: tuple[str, ...], option_name: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise SystemExit(f"{option_name} must contain at least one value")
    unknown = sorted(set(items) - set(choices))
    if unknown:
        raise SystemExit(f"{option_name} contains unsupported value(s): {', '.join(unknown)}")
    return list(dict.fromkeys(items))


def parse_document_node_pooling_grid(value: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    for raw in value.split(","):
        entry = raw.strip()
        if not entry:
            continue
        if ":" in entry:
            mode, ratio_text = entry.split(":", 1)
        else:
            mode, ratio_text = ("off" if entry == "1.0" else "greedy_cosine", entry)
        mode = mode.strip()
        ratio_text = ratio_text.strip()
        if mode not in DOCUMENT_NODE_TOKEN_POOLING_CHOICES:
            raise SystemExit(f"--document-node-pooling-grid contains unsupported mode: {mode}")
        try:
            ratio = float(ratio_text)
        except ValueError as exc:
            raise SystemExit("--document-node-pooling-grid ratios must be numeric") from exc
        if ratio <= 0.0 or ratio > 1.0:
            raise SystemExit("--document-node-pooling-grid ratios must be in (0, 1]")
        if mode == "off" and ratio != 1.0:
            raise SystemExit("--document-node-pooling-grid uses off only with ratio 1.0")
        key = (mode, ratio)
        if key in seen:
            continue
        seen.add(key)
        items.append({"mode": mode, "ratio": ratio})
    if not items:
        raise SystemExit("--document-node-pooling-grid must contain at least one entry")
    return items


def run_admission_budget(
    conn: psycopg.Connection[Any],
    query: QueryItem,
    args: argparse.Namespace,
    budget: int,
    exact_top: list[dict[str, Any]],
) -> dict[str, Any]:
    exact_rerank_k = effective_admission_exact_rerank_k(args, budget)
    admission_debug_mode = effective_admission_debug_mode(args)
    exec_sql(conn, "SELECT set_config('turbohybrid.multivector_doc_candidate_k', %s, false)", (str(budget),))
    exec_sql(
        conn,
        "SELECT set_config('turbohybrid.multivector_exact_rerank_k', %s, false)",
        (str(exact_rerank_k),),
    )
    started = time.perf_counter()
    docs = run_retrieval_query(
        conn,
        QUERY_ONLY_METHOD,
        query,
        args,
        max(args.final_k, args.admission_k),
        dense_k=budget,
    )
    latency_ms = elapsed_ms_since(started)
    stats = last_scan_stats(conn)
    if isinstance(stats, dict):
        stats = {**stats, "observed_latency_ms": latency_ms}
    trace = stats.get("multivector_admission_trace")
    trace_entries = trace if isinstance(trace, list) else []
    trace_by_heap = {
        key: entry
        for entry in trace_entries
        if isinstance(entry, dict) and (key := trace_key(entry)) is not None
    }
    candidate_source = stats.get("multivector_candidate_source")
    fallback_used = bool(stats.get("multivector_plain_fallback_used"))
    infer_admission_from_results = (
        not trace_by_heap
        and can_infer_admission_from_result_docs(candidate_source, fallback_used=fallback_used)
    )

    admitted_ranks: list[int] = []
    exact_top_with_admission: list[dict[str, Any]] = []
    for exact in exact_top:
        key = (int(exact["heap_block"]), int(exact["heap_offset"]))
        entry = trace_by_heap.get(key)
        result_rank = docs.index(exact["doc_id"]) + 1 if exact["doc_id"] in docs else None
        inferred_admitted = infer_admission_from_results and result_rank is not None
        admitted = entry is not None or inferred_admitted
        admission_evidence = (
            "trace"
            if entry is not None
            else "result_doc"
            if inferred_admitted
            else "unavailable"
        )
        if admitted:
            admitted_ranks.append(int(exact["rank"]))
        exact_top_with_admission.append({
            "rank": exact["rank"],
            "doc_id": exact["doc_id"],
            "admitted_before_rerank": admitted,
            "admission_evidence": admission_evidence,
            "candidate_rank_before_rerank": (
                int(entry["candidate_rank_before_truncation"])
                if isinstance(entry, dict) and entry.get("candidate_rank_before_truncation") is not None
                else None
            ),
            "retained_for_exact_rerank": (
                bool(entry.get("retained_for_exact_rerank"))
                if isinstance(entry, dict) and "retained_for_exact_rerank" in entry
                else True
                if inferred_admitted
                else None
            ),
            "exact_rerank_score": (
                float(entry["exact_rerank_score"])
                if isinstance(entry, dict) and entry.get("exact_rerank_score") is not None
                else float(exact["maxsim"])
                if inferred_admitted
                else None
            ),
            "exact_rerank_rank": result_rank,
        })

    top10 = exact_top[: min(10, len(exact_top))]
    top10_admitted = sum(1 for item in exact_top_with_admission[: len(top10)] if item["admitted_before_rerank"])
    top1_admission = bool(exact_top_with_admission and exact_top_with_admission[0]["admitted_before_rerank"])
    docs_scored = scan_docs_scored(stats)
    graph_edges_visited = scan_graph_edges_visited(stats)
    exact_rerank_docs = scan_exact_rerank_docs(stats)
    admission_source = (
        "trace"
        if trace_by_heap
        else "result_doc"
        if infer_admission_from_results
        else "unavailable"
    )
    admission_limitations = None
    if admission_source == "result_doc":
        admission_limitations = (
            "trace unavailable; admission is inferred only from final returned "
            "documents that survived the bounded exact rerank set"
        )
    candidate_diagnostics = derive_candidate_underfill_diagnostics({
        "candidate_budget": budget,
        "candidate_source": candidate_source,
        "final_k": args.final_k,
        "multivector_doc_candidate_k": int(
            getattr(args, "multivector_doc_candidate_k", 0) or 0
        ),
        "multivector_doc_graph_search_ef": scan_stat_int(
            stats,
            "multivector_doc_graph_search_ef",
        )
        or int(getattr(args, "multivector_doc_graph_search_ef", 0) or 0),
        "multivector_doc_graph_oversampling": scan_stat_int(
            stats,
            "multivector_doc_graph_oversampling",
        )
        or int(getattr(args, "multivector_doc_graph_oversampling", 1) or 1),
        "effective_exact_rerank_k": exact_rerank_k,
        "last_scan_stats_sample": stats,
    })
    return {
        "budget": budget,
        **candidate_diagnostics,
        "admission_debug_mode": admission_debug_mode,
        "trace_enabled": admission_debug_mode == "trace",
        "serving_exact_rerank_mode": getattr(args, "serving_exact_rerank_mode", "admission_exhaustive"),
        "requested_serving_exact_rerank_k": int(getattr(args, "serving_exact_rerank_k", 100)),
        "effective_exact_rerank_k": exact_rerank_k,
        "candidate_source": candidate_source,
        "exact_token_scan_nodes_scored": scan_stat_int(stats, "multivector_exact_token_scan_nodes_scored"),
        "plain_fallback_used": bool(stats.get("multivector_plain_fallback_used")),
        "plain_fallback_reason": stats.get("multivector_plain_fallback_reason"),
        "plain_fallback_docs_scored": scan_stat_int(stats, "multivector_plain_fallback_docs_scored"),
        "plain_fallback_pairs": scan_stat_int(stats, "multivector_plain_fallback_pairs"),
        "doc_graph_prototype_enabled": scan_stat_bool(stats, "multivector_doc_graph_prototype_enabled"),
        "doc_graph_docs_scored": scan_stat_int(stats, "multivector_doc_graph_docs_scored"),
        "doc_graph_edges_visited": scan_stat_int(stats, "multivector_doc_graph_edges_visited"),
        "doc_graph_candidates": scan_stat_int(stats, "multivector_doc_graph_candidates"),
        "doc_graph_heap_fetches": scan_stat_int(stats, "multivector_doc_graph_heap_fetches"),
        "doc_graph_warning": stats.get("multivector_doc_graph_warning"),
        "proxy_candidate_limit_effective": scan_stat_int(
            stats,
            "proxy_candidate_limit_effective",
        ),
        "proxy_candidate_limit_source": scan_stat_str(
            stats,
            "proxy_candidate_limit_source",
        ),
        "multivector_proxy_candidate_target": scan_stat_int(
            stats,
            "multivector_proxy_candidate_target",
        ),
        "multivector_proxy_candidates_returned": scan_stat_int(
            stats,
            "multivector_proxy_candidates_returned",
        ),
        "reservoirs_enabled": scan_stat_bool(stats, "multivector_reservoirs_enabled"),
        "reservoir_score_docs": scan_stat_int(stats, "multivector_reservoir_score_docs"),
        "reservoir_coverage_docs": scan_stat_int(stats, "multivector_reservoir_coverage_docs"),
        "reservoir_mean_docs": scan_stat_int(stats, "multivector_reservoir_mean_docs"),
        "reservoir_per_token_docs": scan_stat_int(stats, "multivector_reservoir_per_token_docs"),
        "reservoir_bm25_docs": scan_stat_int(stats, "multivector_reservoir_bm25_docs"),
        "reservoir_union_docs": scan_stat_int(stats, "multivector_reservoir_union_docs"),
        "reservoir_duplicates": scan_stat_int(stats, "multivector_reservoir_duplicates"),
        **bm25_rescue_work_from_stats(stats),
        **learned_sparse_work_from_stats(stats),
        "latency_ms": latency_ms,
        "latency": {"ms": latency_ms},
        "retrieval_elapsed_ms": latency_ms,
        "retrieval_query_count": 1,
        "result_doc_ids": docs,
        "exact_top1_admission": top1_admission,
        "exact_top1_admitted": top1_admission,
        "exact_top1_admitted_before_rerank": top1_admission,
        "exact_top10_admission_recall": round(top10_admitted / len(top10), 6) if top10 else 0.0,
        "exact_top1_candidate_rank_before_rerank": (
            exact_top_with_admission[0]["candidate_rank_before_rerank"] if exact_top_with_admission else None
        ),
        "exact_top1_rank": (
            exact_top_with_admission[0]["exact_rerank_rank"] if exact_top_with_admission else None
        ),
        "exact_top1_exact_rerank_rank": (
            exact_top_with_admission[0]["exact_rerank_rank"] if exact_top_with_admission else None
        ),
        "raw_subvector_hits": scan_stat_int(stats, "multivector_raw_subvector_hits"),
        "unique_docs": scan_stat_int(stats, "multivector_unique_docs"),
        "maxsim_updates": scan_stat_int(stats, "multivector_maxsim_updates"),
        "doc_candidates": scan_stat_int(stats, "multivector_doc_candidates"),
        "docs_scored": docs_scored,
        "graph_edges_visited": graph_edges_visited,
        "exact_rerank_docs": exact_rerank_docs,
        **exact_rerank_work_from_stats(stats),
        **token_pooling_work_from_stats(stats),
        **proxy_work_from_stats(stats),
        **centroid_work_from_stats(stats),
        **quantized_inverted_work_from_stats(stats),
        "sidecar_stats": sidecar_stats_from_scan(stats),
        "doc_graph_search_ef": scan_stat_int(stats, "multivector_doc_graph_search_ef"),
        "doc_graph_oversampling": scan_stat_int(stats, "multivector_doc_graph_oversampling"),
        "doc_graph_rescore_k": scan_stat_int(stats, "multivector_doc_graph_rescore_k"),
        "doc_graph_storage_kind": scan_stat_str(stats, "multivector_doc_graph_storage_kind"),
        "doc_graph_rescore_source": scan_stat_str(stats, "multivector_doc_graph_rescore_source"),
        "memory_bytes_estimate": memory_estimate_from_stats(stats),
        "trace_available": scan_stat_bool(stats, "multivector_admission_trace_available"),
        "trace_entries": len(trace_entries),
        "admission_inferred_from_result_docs": infer_admission_from_results,
        "admission_evidence_mode": admission_source,
        "pre_rerank_admission_source": admission_source,
        "pre_rerank_admission_limitations": admission_limitations,
        "exact_top": exact_top_with_admission,
        "scan_stats": stats,
    }


def run_admission_debug(
    conn: psycopg.Connection[Any],
    args: argparse.Namespace,
    queries: list[QueryItem],
    exact_top_cache: dict[str, list[dict[str, Any]]] | None = None,
    exact_top_cache_document_count: int | None = None,
    exact_top_provider: ExactTopProvider = exact_admission_top,
    executed_budgets: list[int] | None = None,
) -> dict[str, Any]:
    debug_started = time.perf_counter()
    exact_top_cache_current_doc_count = (
        loaded_document_count(conn) if exact_top_cache is not None else None
    )
    validate_exact_top_cache_document_count(
        exact_top_cache_document_count,
        exact_top_cache_current_doc_count,
    )
    budgets = [int(item.strip()) for item in args.admission_budget_sweep.split(",") if item.strip()]
    if not budgets:
        raise ValueError("--admission-budget-sweep must contain at least one integer budget")
    budgets = sorted({budget for budget in budgets if budget > 0})
    if not budgets:
        raise ValueError("--admission-budget-sweep budgets must be positive")
    requested_budgets = budgets
    if executed_budgets is None:
        executed_budget_values = requested_budgets
    else:
        executed_budget_values = sorted({int(budget) for budget in executed_budgets if int(budget) > 0})
    if not executed_budget_values:
        raise ValueError("executed admission budgets must contain at least one positive integer")
    missing_executed_budgets = sorted(set(executed_budget_values) - set(requested_budgets))
    if missing_executed_budgets:
        raise ValueError(
            "executed admission budgets must be included in --admission-budget-sweep: "
            + ",".join(str(budget) for budget in missing_executed_budgets)
        )

    admission_debug_mode = effective_admission_debug_mode(args)
    set_retrieval_gucs(conn, args, "dbpedia_colbert_admission_debug")
    exec_sql(
        conn,
        "SELECT set_config('turbohybrid.multivector_debug_admission', %s, false)",
        (admission_debug_mode,),
    )
    exec_sql(
        conn,
        "SELECT set_config('turbohybrid.multivector_debug_trace_limit', %s, false)",
        (str(min(args.admission_trace_limit, 1000) if admission_debug_mode == "trace" else 0),),
    )

    per_query: list[dict[str, Any]] = []
    latency_by_budget: dict[int, list[float]] = {budget: [] for budget in executed_budget_values}
    top1_by_budget: dict[int, list[bool]] = {budget: [] for budget in executed_budget_values}
    top10_recall_by_budget: dict[int, list[float]] = {budget: [] for budget in executed_budget_values}
    docs_scored_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    graph_edges_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    exact_rerank_docs_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    exact_rerank_pairs_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    exact_rerank_tokens_evaluated_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    exact_rerank_tokens_skipped_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    exact_rerank_pairs_saved_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    tokens_original_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    tokens_pooled_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    token_pooling_ratio_by_budget: dict[int, list[float]] = {budget: [] for budget in executed_budget_values}
    proxy_candidates_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    proxy_exact_rerank_docs_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    reservoirs_enabled_by_budget: dict[int, list[bool]] = {budget: [] for budget in executed_budget_values}
    reservoir_score_docs_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    reservoir_coverage_docs_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    reservoir_mean_docs_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    reservoir_per_token_docs_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    reservoir_bm25_docs_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    reservoir_union_docs_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    reservoir_duplicates_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    centroid_lists_visited_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    centroid_docs_touched_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    centroid_pruned_docs_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    centroid_postings_touched_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    centroid_postings_skipped_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    centroid_posting_limit_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    centroid_posting_cap_strategy_by_budget: dict[int, list[str]] = {
        budget: [] for budget in executed_budget_values
    }
    centroid_candidates_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    learned_sparse_candidates_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    learned_sparse_retained_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    learned_sparse_latency_us_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    bm25_injection_enabled_by_budget: dict[int, list[bool]] = {budget: [] for budget in executed_budget_values}
    bm25_injection_candidates_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    bm25_injection_candidate_limit_by_budget: dict[int, list[int]] = {
        budget: [] for budget in executed_budget_values
    }
    bm25_injection_pool_size_by_budget: dict[int, list[int]] = {
        budget: [] for budget in executed_budget_values
    }
    bm25_injection_limit_reason_by_budget: dict[int, list[str]] = {
        budget: [] for budget in executed_budget_values
    }
    bm25_injection_retained_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    bm25_injection_exact_reranked_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    quantized_inverted_candidates_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    quantized_inverted_postings_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    quantized_inverted_docs_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    exact_top1_rank_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    native_cache_bytes_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    native_cache_exact_bytes_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    sidecar_pages_read_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    sidecar_bytes_touched_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    sidecar_cache_hits_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    sidecar_vectors_loaded_by_budget: dict[int, list[int]] = {budget: [] for budget in executed_budget_values}
    top1_first_budget_values: list[int] = []
    top1_admitted_queries = 0
    top10_recall_values: list[float] = []
    run_at_largest_budget: dict[str, list[str]] = {}
    exact_baseline_elapsed_ms_total = 0.0
    retrieval_elapsed_ms_total = 0.0
    exact_baseline_query_count = 0
    retrieval_query_count = 0
    exact_top_cache_hits = 0
    exact_top_cache_misses = 0
    admission_evidence_modes: dict[str, int] = {}
    admission_evidence_modes_by_budget: dict[int, dict[str, int]] = {
        budget: {} for budget in executed_budget_values
    }
    trace_entries_by_budget: dict[int, int] = {budget: 0 for budget in executed_budget_values}
    trace_entry_count = 0

    for query in queries:
        exact_top, exact_elapsed_ms, exact_top_cache_hit = cached_exact_admission_top(
            conn,
            query,
            args.admission_k,
            exact_top_cache,
            exact_top_provider,
        )
        if exact_top_cache_hit:
            exact_top_cache_hits += 1
        else:
            exact_top_cache_misses += 1
            exact_baseline_elapsed_ms_total += exact_elapsed_ms
            exact_baseline_query_count += 1
        budget_results: list[dict[str, Any]] = []
        first_budget: int | None = None
        top1_candidate_rank: int | None = None
        top1_exact_rank: int | None = None
        for budget in executed_budget_values:
            result = run_admission_budget(conn, query, args, budget, exact_top)
            evidence_mode = str(result.get("admission_evidence_mode", "unavailable") or "unavailable")
            admission_evidence_modes[evidence_mode] = admission_evidence_modes.get(evidence_mode, 0) + 1
            budget_evidence_modes = admission_evidence_modes_by_budget[budget]
            budget_evidence_modes[evidence_mode] = budget_evidence_modes.get(evidence_mode, 0) + 1
            budget_trace_entries = int(result.get("trace_entries", 0) or 0)
            trace_entries_by_budget[budget] += budget_trace_entries
            trace_entry_count += budget_trace_entries
            retrieval_elapsed_ms_total += float(result.get("retrieval_elapsed_ms", 0.0) or 0.0)
            retrieval_query_count += int(result.get("retrieval_query_count", 0) or 0)
            budget_results.append(result)
            latency_by_budget[budget].append(float(result["latency_ms"]))
            top1_by_budget[budget].append(bool(result["exact_top1_admission"]))
            top10_recall_by_budget[budget].append(float(result["exact_top10_admission_recall"]))
            docs_scored_by_budget[budget].append(int(result["docs_scored"]))
            graph_edges_by_budget[budget].append(int(result["graph_edges_visited"]))
            exact_rerank_docs_by_budget[budget].append(int(result["exact_rerank_docs"]))
            exact_rerank_pairs_by_budget[budget].append(
                int(result.get("exact_rerank_pairs", 0))
            )
            exact_rerank_tokens_evaluated_by_budget[budget].append(
                int(result.get("exact_rerank_tokens_evaluated", 0))
            )
            exact_rerank_tokens_skipped_by_budget[budget].append(
                int(result.get("exact_rerank_tokens_skipped", 0))
            )
            exact_rerank_pairs_saved_by_budget[budget].append(
                int(result.get("exact_rerank_pairs_saved", 0))
            )
            tokens_original_by_budget[budget].append(
                int(result.get("multivector_tokens_original", 0))
            )
            tokens_pooled_by_budget[budget].append(
                int(result.get("multivector_tokens_pooled", 0))
            )
            token_pooling_ratio_by_budget[budget].append(
                float(result.get("multivector_token_pooling_ratio", 0.0) or 0.0)
            )
            proxy_candidates_by_budget[budget].append(int(result.get("proxy_candidates", 0)))
            proxy_exact_rerank_docs_by_budget[budget].append(
                int(result.get("proxy_exact_rerank_docs", 0))
            )
            reservoirs_enabled_by_budget[budget].append(
                bool(result.get("reservoirs_enabled", False))
            )
            reservoir_score_docs_by_budget[budget].append(
                int(result.get("reservoir_score_docs", 0))
            )
            reservoir_coverage_docs_by_budget[budget].append(
                int(result.get("reservoir_coverage_docs", 0))
            )
            reservoir_mean_docs_by_budget[budget].append(
                int(result.get("reservoir_mean_docs", 0))
            )
            reservoir_per_token_docs_by_budget[budget].append(
                int(result.get("reservoir_per_token_docs", 0))
            )
            reservoir_bm25_docs_by_budget[budget].append(
                int(result.get("reservoir_bm25_docs", 0))
            )
            reservoir_union_docs_by_budget[budget].append(
                int(result.get("reservoir_union_docs", 0))
            )
            reservoir_duplicates_by_budget[budget].append(
                int(result.get("reservoir_duplicates", 0))
            )
            centroid_lists_visited_by_budget[budget].append(
                int(result.get("centroid_lists_visited", 0))
            )
            centroid_docs_touched_by_budget[budget].append(
                int(result.get("centroid_docs_touched", 0))
            )
            centroid_pruned_docs_by_budget[budget].append(
                int(result.get("centroid_pruned_docs", 0))
            )
            centroid_postings_touched_by_budget[budget].append(
                int(result.get("centroid_postings_touched", 0))
            )
            centroid_postings_skipped_by_budget[budget].append(
                int(result.get("centroid_postings_skipped", 0))
            )
            centroid_posting_limit_by_budget[budget].append(
                int(result.get("centroid_posting_limit_per_token", 0))
            )
            centroid_posting_cap_strategy = str(
                result.get("centroid_posting_cap_strategy", "") or ""
            )
            if centroid_posting_cap_strategy:
                centroid_posting_cap_strategy_by_budget[budget].append(
                    centroid_posting_cap_strategy
                )
            centroid_candidates_by_budget[budget].append(
                int(result.get("centroid_candidates", 0))
            )
            learned_sparse_candidates_by_budget[budget].append(
                int(result.get("learned_sparse_candidates", 0))
            )
            learned_sparse_retained_by_budget[budget].append(
                int(result.get("learned_sparse_retained_for_maxsim", 0))
            )
            learned_sparse_latency_us_by_budget[budget].append(
                int(result.get("learned_sparse_branch_latency_us", 0))
            )
            bm25_injection_enabled_by_budget[budget].append(
                bool(result.get("bm25_injection_enabled", False))
            )
            bm25_injection_candidates_by_budget[budget].append(
                int(result.get("bm25_injection_candidates", 0))
            )
            bm25_injection_candidate_limit_by_budget[budget].append(
                int(result.get("bm25_injection_candidate_limit", 0))
            )
            bm25_injection_pool_size_by_budget[budget].append(
                int(result.get("bm25_injection_pool_size", 0))
            )
            bm25_limit_reason = str(result.get("bm25_injection_limit_reason", "") or "")
            if bm25_limit_reason:
                bm25_injection_limit_reason_by_budget[budget].append(bm25_limit_reason)
            bm25_injection_retained_by_budget[budget].append(
                int(result.get("bm25_injection_retained", 0))
            )
            bm25_injection_exact_reranked_by_budget[budget].append(
                int(result.get("bm25_injection_exact_reranked", 0))
            )
            quantized_inverted_candidates_by_budget[budget].append(
                int(result.get("quantized_inverted_candidates", 0))
            )
            quantized_inverted_postings_by_budget[budget].append(
                int(result.get("quantized_inverted_postings_touched", 0))
            )
            quantized_inverted_docs_by_budget[budget].append(
                int(result.get("quantized_inverted_docs_scored", 0))
            )
            if result["exact_top1_exact_rerank_rank"] is not None:
                exact_top1_rank_by_budget[budget].append(int(result["exact_top1_exact_rerank_rank"]))
            sidecar_stats = result.get("sidecar_stats", {})
            if isinstance(sidecar_stats, dict):
                native_cache_bytes_by_budget[budget].append(int(sidecar_stats.get("native_cache_bytes", 0)))
                native_cache_exact_bytes_by_budget[budget].append(int(sidecar_stats.get("native_cache_exact_bytes", 0)))
                sidecar_pages_read_by_budget[budget].append(int(sidecar_stats.get("sidecar_pages_read", 0)))
                sidecar_bytes_touched_by_budget[budget].append(int(sidecar_stats.get("sidecar_bytes_touched", 0)))
                sidecar_cache_hits_by_budget[budget].append(int(sidecar_stats.get("sidecar_cache_hits", 0)))
                sidecar_vectors_loaded_by_budget[budget].append(int(sidecar_stats.get("sidecar_vectors_loaded", 0)))
            if result["exact_top1_admitted_before_rerank"] and first_budget is None:
                first_budget = budget
                top1_candidate_rank = result["exact_top1_candidate_rank_before_rerank"]
                top1_exact_rank = result["exact_top1_exact_rerank_rank"]
            if budget == executed_budget_values[-1]:
                run_at_largest_budget[query.query_id] = list(result["result_doc_ids"])
        if first_budget is not None:
            top1_first_budget_values.append(first_budget)
            top1_admitted_queries += 1
        if budget_results:
            top10_recall_values.append(float(budget_results[-1]["exact_top10_admission_recall"]))

        per_query.append({
            "query_id": query.query_id,
            "query_text": query.query_text,
            "exact_top": exact_top,
            "exact_baseline_elapsed_ms": exact_elapsed_ms,
            "exact_baseline_query_count": 0 if exact_top_cache_hit else 1,
            "exact_top_cache_hit": exact_top_cache_hit,
            "exact_top_cache_key": exact_top_cache_key(query.query_id, args.admission_k),
            "exact_top1_admitted": first_budget is not None,
            "exact_top1_admitted_before_rerank": first_budget is not None,
            "exact_top10_admission_recall": budget_results[-1]["exact_top10_admission_recall"] if budget_results else 0.0,
            "exact_top1_first_budget_admitted": first_budget,
            "first_budget_admitting_exact_top1": first_budget,
            "exact_top1_rank": top1_exact_rank,
            "exact_top1_candidate_rank_before_rerank": top1_candidate_rank,
            "exact_top1_exact_rerank_rank": top1_exact_rank,
            "budgets": budget_results,
        })

    total_elapsed_ms = elapsed_ms_since(debug_started)
    aggregate = {
        "queries": len(per_query),
        "admission_k": args.admission_k,
        "admission_debug_mode": admission_debug_mode,
        "trace_enabled": admission_debug_mode == "trace",
        "admission_evidence_mode": (
            "trace"
            if admission_evidence_modes.get("trace", 0) > 0
            else "result_doc"
            if admission_evidence_modes.get("result_doc", 0) > 0
            else "unavailable"
        ),
        "admission_evidence_modes": dict(sorted(admission_evidence_modes.items())),
        "trace_entries": trace_entry_count,
        "budget_sweep": requested_budgets,
        "executed_budgets": executed_budget_values,
        "executed_budget_count": len(executed_budget_values),
        "serving_exact_rerank_mode": getattr(args, "serving_exact_rerank_mode", "admission_exhaustive"),
        "requested_serving_exact_rerank_k": int(getattr(args, "serving_exact_rerank_k", 100)),
        "effective_exact_rerank_by_budget": {
            str(budget): effective_admission_exact_rerank_k(args, budget)
            for budget in executed_budget_values
        },
        "total_elapsed_ms": total_elapsed_ms,
        "admission_debug_elapsed_ms_total": total_elapsed_ms,
        "exact_baseline_elapsed_ms_total": round(exact_baseline_elapsed_ms_total, 3),
        "retrieval_elapsed_ms_total": round(retrieval_elapsed_ms_total, 3),
        "exact_baseline_query_count": exact_baseline_query_count,
        "retrieval_query_count": retrieval_query_count,
        "exact_top_cache_enabled": exact_top_cache is not None,
        "exact_top_cache_hits": exact_top_cache_hits,
        "exact_top_cache_misses": exact_top_cache_misses,
        "exact_top_cache_entries": len(exact_top_cache) if exact_top_cache is not None else 0,
        "exact_top_cache_loaded_document_count": exact_top_cache_current_doc_count,
        "effective_budget_count": len(executed_budget_values),
        "effective_budgets": executed_budget_values,
        "exact_top1_admission": round(top1_admitted_queries / len(per_query), 6) if per_query else 0.0,
        "exact_top1_admitted": round(top1_admitted_queries / len(per_query), 6) if per_query else 0.0,
        "exact_top1_admission_rate": round(top1_admitted_queries / len(per_query), 6) if per_query else 0.0,
        "exact_top10_admission_recall": round(statistics.mean(top10_recall_values), 6) if top10_recall_values else 0.0,
        "exact_top1_first_budget_admitted": summarize_ints(top1_first_budget_values),
        "first_budget_admitting_exact_top1": summarize_ints(top1_first_budget_values),
        "admission_by_budget": {
            str(budget): {
                "serving_exact_rerank_mode": getattr(args, "serving_exact_rerank_mode", "admission_exhaustive"),
                "admission_debug_mode": admission_debug_mode,
                "trace_enabled": admission_debug_mode == "trace",
                "admission_evidence_mode": (
                    "trace"
                    if admission_evidence_modes_by_budget[budget].get("trace", 0) > 0
                    else "result_doc"
                    if admission_evidence_modes_by_budget[budget].get("result_doc", 0) > 0
                    else "unavailable"
                ),
                "admission_evidence_modes": dict(
                    sorted(admission_evidence_modes_by_budget[budget].items())
                ),
                "trace_entries": trace_entries_by_budget[budget],
                "requested_serving_exact_rerank_k": int(getattr(args, "serving_exact_rerank_k", 100)),
                "effective_exact_rerank_k": effective_admission_exact_rerank_k(args, budget),
                "exact_top1_admission": round(
                    sum(1 for value in top1_by_budget[budget] if value) / len(top1_by_budget[budget]),
                    6,
                ) if top1_by_budget[budget] else 0.0,
                "exact_top1_admitted": round(
                    sum(1 for value in top1_by_budget[budget] if value) / len(top1_by_budget[budget]),
                    6,
                ) if top1_by_budget[budget] else 0.0,
                "exact_top10_admission_recall": round(
                    statistics.mean(top10_recall_by_budget[budget]),
                    6,
                ) if top10_recall_by_budget[budget] else 0.0,
                "latency": summarize_ms(latency_by_budget[budget]),
                "docs_scored": summarize_ints(docs_scored_by_budget[budget]),
                "graph_edges_visited": summarize_ints(graph_edges_by_budget[budget]),
                "exact_rerank_docs": summarize_ints(exact_rerank_docs_by_budget[budget]),
                "exact_rerank_pairs": summarize_ints(exact_rerank_pairs_by_budget[budget]),
                "exact_rerank_tokens_evaluated": summarize_ints(
                    exact_rerank_tokens_evaluated_by_budget[budget]
                ),
                "exact_rerank_tokens_skipped": summarize_ints(
                    exact_rerank_tokens_skipped_by_budget[budget]
                ),
                "exact_rerank_pairs_saved": summarize_ints(
                    exact_rerank_pairs_saved_by_budget[budget]
                ),
                "multivector_tokens_original": summarize_ints(
                    tokens_original_by_budget[budget]
                ),
                "multivector_tokens_pooled": summarize_ints(
                    tokens_pooled_by_budget[budget]
                ),
                "multivector_token_pooling_ratio": summarize_floats(
                    token_pooling_ratio_by_budget[budget]
                ),
                "proxy_candidates": summarize_ints(proxy_candidates_by_budget[budget]),
                "proxy_exact_rerank_docs": summarize_ints(
                    proxy_exact_rerank_docs_by_budget[budget]
                ),
                "reservoirs_enabled_queries": sum(
                    1 for value in reservoirs_enabled_by_budget[budget] if value
                ),
                "reservoir_score_docs": summarize_ints(
                    reservoir_score_docs_by_budget[budget]
                ),
                "reservoir_coverage_docs": summarize_ints(
                    reservoir_coverage_docs_by_budget[budget]
                ),
                "reservoir_mean_docs": summarize_ints(
                    reservoir_mean_docs_by_budget[budget]
                ),
                "reservoir_per_token_docs": summarize_ints(
                    reservoir_per_token_docs_by_budget[budget]
                ),
                "reservoir_bm25_docs": summarize_ints(
                    reservoir_bm25_docs_by_budget[budget]
                ),
                "reservoir_union_docs": summarize_ints(
                    reservoir_union_docs_by_budget[budget]
                ),
                "reservoir_duplicates": summarize_ints(
                    reservoir_duplicates_by_budget[budget]
                ),
                "centroid_lists_visited": summarize_ints(
                    centroid_lists_visited_by_budget[budget]
                ),
                "centroid_docs_touched": summarize_ints(
                    centroid_docs_touched_by_budget[budget]
                ),
                "centroid_pruned_docs": summarize_ints(
                    centroid_pruned_docs_by_budget[budget]
                ),
                "centroid_postings_touched": summarize_ints(
                    centroid_postings_touched_by_budget[budget]
                ),
                "centroid_postings_skipped": summarize_ints(
                    centroid_postings_skipped_by_budget[budget]
                ),
                "centroid_posting_limit_per_token": summarize_ints(
                    centroid_posting_limit_by_budget[budget]
                ),
                "centroid_posting_cap_strategy": summarize_strings(
                    centroid_posting_cap_strategy_by_budget[budget]
                ),
                "centroid_candidates": summarize_ints(
                    centroid_candidates_by_budget[budget]
                ),
                "learned_sparse_candidates": summarize_ints(
                    learned_sparse_candidates_by_budget[budget]
                ),
                "learned_sparse_retained_for_maxsim": summarize_ints(
                    learned_sparse_retained_by_budget[budget]
                ),
                "learned_sparse_branch_latency_us": summarize_ints(
                    learned_sparse_latency_us_by_budget[budget]
                ),
                "bm25_injection_enabled_queries": sum(
                    1 for value in bm25_injection_enabled_by_budget[budget] if value
                ),
                "bm25_injection_candidates": summarize_ints(
                    bm25_injection_candidates_by_budget[budget]
                ),
                "bm25_injection_candidate_limit": summarize_ints(
                    bm25_injection_candidate_limit_by_budget[budget]
                ),
                "bm25_injection_pool_size": summarize_ints(
                    bm25_injection_pool_size_by_budget[budget]
                ),
                "bm25_injection_limit_reason": summarize_strings(
                    bm25_injection_limit_reason_by_budget[budget]
                ),
                "bm25_injection_retained": summarize_ints(
                    bm25_injection_retained_by_budget[budget]
                ),
                "bm25_injection_exact_reranked": summarize_ints(
                    bm25_injection_exact_reranked_by_budget[budget]
                ),
                "quantized_inverted_candidates": summarize_ints(
                    quantized_inverted_candidates_by_budget[budget]
                ),
                "quantized_inverted_postings_touched": summarize_ints(
                    quantized_inverted_postings_by_budget[budget]
                ),
                "quantized_inverted_docs_scored": summarize_ints(
                    quantized_inverted_docs_by_budget[budget]
                ),
                "exact_top1_rank": summarize_ints(exact_top1_rank_by_budget[budget]),
                "native_cache_bytes": summarize_ints(native_cache_bytes_by_budget[budget]),
                "native_cache_exact_bytes": summarize_ints(native_cache_exact_bytes_by_budget[budget]),
                "sidecar_pages_read": summarize_ints(sidecar_pages_read_by_budget[budget]),
                "sidecar_bytes_touched": summarize_ints(sidecar_bytes_touched_by_budget[budget]),
                "sidecar_cache_hits": summarize_ints(sidecar_cache_hits_by_budget[budget]),
                "sidecar_vectors_loaded": summarize_ints(sidecar_vectors_loaded_by_budget[budget]),
            }
            for budget in executed_budget_values
        },
        "latency_by_budget": {
            str(budget): summarize_ms(values)
            for budget, values in latency_by_budget.items()
        },
        "run_at_largest_budget": run_at_largest_budget,
    }

    set_retrieval_gucs(conn, args, "dbpedia_colbert_serial")
    return {
        "enabled": True,
        "candidate_source": args.multivector_candidate_source,
        "graph_mode": args.multivector_graph,
        "storage_kind": args.multivector_doc_storage,
        "token_pooling": args.multivector_token_pooling,
        "token_pooling_target_ratio": args.multivector_token_pooling_target_ratio,
        "token_pooling_min_tokens": args.multivector_token_pooling_min_tokens,
        "centroids": args.multivector_centroids,
        "centroid_count": args.multivector_centroid_count,
        "doc_graph_search_ef": args.multivector_doc_graph_search_ef,
        "doc_graph_oversampling": args.multivector_doc_graph_oversampling,
        "doc_graph_rescore_k": args.multivector_doc_graph_rescore_k,
        "doc_graph_entry_sample_count": args.multivector_doc_graph_entry_sample_count,
        "plain_fallback": args.multivector_plain_fallback,
        "candidate_reservoirs": args.multivector_candidate_reservoirs,
        "bm25_candidate_injection": args.multivector_bm25_candidate_injection,
        "sparse_candidate_source": args.multivector_sparse_candidate_source,
        "retrieval_method": QUERY_ONLY_METHOD,
        "aggregate": aggregate,
        "per_query": per_query,
    }


def grid_mode_args(args: argparse.Namespace, mode: str, graph_mode: str) -> argparse.Namespace:
    if mode == "token_nodes":
        return clone_args(
            args,
            multivector_graph=graph_mode,
            multivector_candidate_source="graph",
            multivector_plain_fallback="off",
            multivector_candidate_reservoirs="off",
            multivector_doc_storage="f32",
            multivector_doc_storage_cache="auto",
            multivector_token_pooling="off",
            multivector_token_pooling_target_ratio=1.0,
            multivector_centroids="off",
            multivector_centroid_count="auto",
            multivector_doc_graph_search_ef=0,
            multivector_doc_graph_oversampling=1,
            multivector_doc_graph_rescore_k=0,
        )
    if mode == "exact_token_scan":
        return clone_args(
            args,
            multivector_graph=graph_mode,
            multivector_candidate_source="exact_token_scan",
            multivector_plain_fallback="off",
            multivector_candidate_reservoirs="off",
            multivector_doc_storage="f32",
            multivector_doc_storage_cache="auto",
            multivector_token_pooling="off",
            multivector_token_pooling_target_ratio=1.0,
            multivector_centroids="off",
            multivector_centroid_count="auto",
            multivector_doc_graph_search_ef=0,
            multivector_doc_graph_oversampling=1,
            multivector_doc_graph_rescore_k=0,
        )
    if mode == "plain_fallback":
        return clone_args(
            args,
            multivector_graph=graph_mode,
            multivector_candidate_source="graph",
            multivector_plain_fallback="force",
            multivector_candidate_reservoirs="off",
            multivector_doc_storage="f32",
            multivector_doc_storage_cache="auto",
            multivector_token_pooling="off",
            multivector_token_pooling_target_ratio=1.0,
            multivector_centroids="off",
            multivector_centroid_count="auto",
            multivector_doc_graph_search_ef=0,
            multivector_doc_graph_oversampling=1,
            multivector_doc_graph_rescore_k=0,
        )
    if mode == "exact_doc_scan":
        return clone_args(
            args,
            multivector_graph=graph_mode,
            multivector_candidate_source="exact_doc_scan",
            multivector_plain_fallback="off",
            multivector_candidate_reservoirs="off",
            multivector_doc_storage="f32",
            multivector_doc_storage_cache="paged",
            multivector_token_pooling="off",
            multivector_token_pooling_target_ratio=1.0,
            multivector_centroids="off",
            multivector_centroid_count="auto",
            multivector_doc_graph_search_ef=0,
            multivector_doc_graph_oversampling=1,
            multivector_doc_graph_rescore_k=0,
        )
    if mode == "proxy_vector":
        return clone_args(
            args,
            multivector_graph=graph_mode,
            multivector_candidate_source="proxy_vector",
            multivector_plain_fallback="off",
            multivector_candidate_reservoirs="off",
            multivector_doc_storage="f32",
            multivector_doc_storage_cache="auto",
            multivector_token_pooling="off",
            multivector_token_pooling_target_ratio=1.0,
            multivector_centroids="off",
            multivector_centroid_count="auto",
            multivector_doc_graph_search_ef=0,
            multivector_doc_graph_oversampling=1,
            multivector_doc_graph_rescore_k=0,
        )
    if mode == "centroid_lite":
        return clone_args(
            args,
            multivector_graph=graph_mode,
            multivector_candidate_source="centroid_lite",
            multivector_plain_fallback="off",
            multivector_candidate_reservoirs="off",
            multivector_doc_storage="f32",
            multivector_doc_storage_cache="auto",
            multivector_token_pooling="off",
            multivector_token_pooling_target_ratio=1.0,
            multivector_centroids="kmeans",
            multivector_centroid_count=args.multivector_centroid_count,
            multivector_doc_graph_search_ef=0,
            multivector_doc_graph_oversampling=1,
            multivector_doc_graph_rescore_k=0,
        )
    if mode == "quantized_inverted_experimental":
        return clone_args(
            args,
            multivector_graph=graph_mode,
            multivector_candidate_source="quantized_inverted_experimental",
            multivector_plain_fallback="off",
            multivector_candidate_reservoirs="off",
            multivector_doc_storage="f32",
            multivector_doc_storage_cache="auto",
            multivector_token_pooling="off",
            multivector_token_pooling_target_ratio=1.0,
            multivector_centroids="off",
            multivector_centroid_count="auto",
            multivector_doc_graph_search_ef=0,
            multivector_doc_graph_oversampling=1,
            multivector_doc_graph_rescore_k=0,
        )
    raise ValueError(f"unsupported grid mode: {mode}")


def grid_admission_summary(
    *,
    mode: str,
    args: argparse.Namespace,
    index_phase: dict[str, Any],
    admission: dict[str, Any],
    qrels: dict[str, dict[str, int]],
) -> dict[str, Any]:
    aggregate = admission.get("aggregate", {})
    run = aggregate.get("run_at_largest_budget", {})
    if not isinstance(run, dict):
        run = {}
    metrics = method_metrics(run, qrels, args.final_k, args.quality_k) if qrels else {}
    budget_sweep = aggregate.get("budget_sweep", [])
    largest_budget = int(budget_sweep[-1]) if isinstance(budget_sweep, list) and budget_sweep else None
    return {
        "mode": mode,
        "candidate_source": args.multivector_candidate_source,
        "graph_mode": args.multivector_graph,
        "storage_kind": args.multivector_doc_storage,
        "storage_cache_mode": args.multivector_doc_storage_cache,
        "token_pooling": args.multivector_token_pooling,
        "token_pooling_target_ratio": args.multivector_token_pooling_target_ratio,
        "token_pooling_min_tokens": args.multivector_token_pooling_min_tokens,
        "doc_graph_search_ef": args.multivector_doc_graph_search_ef,
        "doc_graph_oversampling": args.multivector_doc_graph_oversampling,
        "doc_graph_rescore_k": args.multivector_doc_graph_rescore_k,
        "index_graph_m": index_phase.get("index_graph_m", 0),
        "index_graph_ef_construction": index_phase.get("index_graph_ef_construction", 0),
        "index_graph_ef_search": index_phase.get("index_graph_ef_search", 0),
        "plain_fallback": args.multivector_plain_fallback,
        "largest_budget": largest_budget,
        "exact_top1_admission_rate": aggregate.get("exact_top1_admission_rate", 0.0),
        "exact_top1_admitted": aggregate.get("exact_top1_admitted", aggregate.get("exact_top1_admission", 0.0)),
        "exact_top10_admission_recall": aggregate.get("exact_top10_admission_recall", 0.0),
        "metrics": metrics,
        "admission_by_budget": aggregate.get("admission_by_budget", {}),
        "latency_by_budget": aggregate.get("latency_by_budget", {}),
        "index_stats": index_phase.get("index_stats", {}),
        "index_bytes": index_phase.get("index_bytes", 0),
    }


def document_node_serving_profiles(
    *,
    include_experimental: bool,
    include_proxy_encoder_variants: bool,
    include_bm25_rescue: bool,
    include_learned_sparse_rescue: bool,
    include_reservoirs: bool,
    include_centroid_lite_caps: bool = False,
    centroid_lite_posting_caps: Sequence[int] = (),
    include_token_pooling_focus: bool = False,
    include_entry_samples: bool = False,
    entry_sample_counts: Sequence[int] = (),
    include_entry_sidecar: bool = False,
    entry_sidecar_representatives: Sequence[int] = (),
    explicit_entry_sidecar_profile_names: bool = False,
    centroid_count: str,
    include_learned_projection: bool = False,
) -> list[DocumentNodeServingProfile]:
    profiles = [
        DocumentNodeServingProfile(
            name="proxy_normalized_mean_f16",
            candidate_source="proxy_vector",
            proxy_encoder="normalized_mean",
            storage_kind="f16",
        ),
        DocumentNodeServingProfile(
            name="docnodes_normalized_mean_f16",
            candidate_source="document_nodes",
            proxy_encoder="normalized_mean",
            storage_kind="f16",
        ),
        DocumentNodeServingProfile(
            name="centroid_mean_f16",
            candidate_source="proxy_vector",
            proxy_encoder="centroid_mean",
            centroids="kmeans",
            centroid_count=centroid_count,
            storage_kind="f16",
        ),
        DocumentNodeServingProfile(
            name="centroid_lite_f16",
            candidate_source="centroid_lite",
            centroids="kmeans",
            centroid_count=centroid_count,
            storage_kind="f16",
        ),
        DocumentNodeServingProfile(
            name="centroid_lite_f16_pool_050",
            candidate_source="centroid_lite",
            centroids="kmeans",
            centroid_count=centroid_count,
            storage_kind="f16",
            token_pooling="greedy_cosine",
            token_pooling_target_ratio=0.5,
        ),
        DocumentNodeServingProfile(
            name="proxy_normalized_mean_sq8",
            candidate_source="proxy_vector",
            proxy_encoder="normalized_mean",
            storage_kind="sq8",
        ),
    ]
    if include_centroid_lite_caps:
        profiles.extend(
            DocumentNodeServingProfile(
                name=f"centroid_lite_f16_cap_{int(cap):03d}",
                candidate_source="centroid_lite",
                centroids="kmeans",
                centroid_count=centroid_count,
                storage_kind="f16",
                centroid_lite_max_postings_per_token=int(cap),
            )
            for cap in centroid_lite_posting_caps
        )
        profiles.append(
            DocumentNodeServingProfile(
                name="centroid_lite_f16_prune_safe_upper_bound",
                candidate_source="centroid_lite",
                centroids="kmeans",
                centroid_count=centroid_count,
                storage_kind="f16",
                centroid_lite_pruning="safe_upper_bound",
            )
        )
        for cap in centroid_lite_posting_caps:
            profiles.append(
                DocumentNodeServingProfile(
                    name=f"centroid_lite_f16_cap_{int(cap):03d}_prune_safe_upper_bound",
                    candidate_source="centroid_lite",
                    centroids="kmeans",
                    centroid_count=centroid_count,
                    storage_kind="f16",
                    centroid_lite_max_postings_per_token=int(cap),
                    centroid_lite_pruning="safe_upper_bound",
                )
            )
    if include_token_pooling_focus:
        for ratio_label, ratio in (("075", 0.75), ("050", 0.5), ("033", 0.33)):
            profiles.extend(
                [
                    DocumentNodeServingProfile(
                        name=f"proxy_normalized_mean_f16_pool_{ratio_label}",
                        candidate_source="proxy_vector",
                        proxy_encoder="normalized_mean",
                        storage_kind="f16",
                        token_pooling="greedy_cosine",
                        token_pooling_target_ratio=ratio,
                    ),
                    DocumentNodeServingProfile(
                        name=f"centroid_mean_f16_pool_{ratio_label}",
                        candidate_source="proxy_vector",
                        proxy_encoder="centroid_mean",
                        centroids="kmeans",
                        centroid_count=centroid_count,
                        storage_kind="f16",
                        token_pooling="greedy_cosine",
                        token_pooling_target_ratio=ratio,
                    ),
                ]
            )
    if include_entry_samples:
        for count in entry_sample_counts:
            entry_sample_count = int(count)
            profiles.extend(
                [
                    DocumentNodeServingProfile(
                        name=f"proxy_normalized_mean_f16_entry_sample_{entry_sample_count:03d}",
                        candidate_source="proxy_vector",
                        proxy_encoder="normalized_mean",
                        storage_kind="f16",
                        entry_sample_count=entry_sample_count,
                    ),
                    DocumentNodeServingProfile(
                        name=f"centroid_mean_f16_entry_sample_{entry_sample_count:03d}",
                        candidate_source="proxy_vector",
                        proxy_encoder="centroid_mean",
                        centroids="kmeans",
                        centroid_count=centroid_count,
                        storage_kind="f16",
                        entry_sample_count=entry_sample_count,
                    ),
                ]
            )
            if include_proxy_encoder_variants:
                profiles.append(
                    DocumentNodeServingProfile(
                        name=f"proxy_max_pool_f16_entry_sample_{entry_sample_count:03d}",
                        candidate_source="proxy_vector",
                        proxy_encoder="max_pool",
                        storage_kind="f16",
                        entry_sample_count=entry_sample_count,
                    )
                )
    if include_entry_sidecar:
        representative_counts = [
            int(count)
            for count in (entry_sidecar_representatives or (128,))
            if int(count) > 0
        ]
        if not representative_counts:
            representative_counts = [128]
        for representatives in representative_counts:
            suffix = f"_{representatives:03d}" if explicit_entry_sidecar_profile_names else ""
            profiles.extend(
                [
                    DocumentNodeServingProfile(
                        name=f"proxy_normalized_mean_f16_entry_sidecar{suffix}",
                        candidate_source="proxy_vector",
                        proxy_encoder="normalized_mean",
                        storage_kind="f16",
                        entry_sidecar=True,
                        entry_sidecar_representatives=representatives,
                    ),
                    DocumentNodeServingProfile(
                        name=f"centroid_mean_f16_entry_sidecar{suffix}",
                        candidate_source="proxy_vector",
                        proxy_encoder="centroid_mean",
                        centroids="kmeans",
                        centroid_count=centroid_count,
                        storage_kind="f16",
                        entry_sidecar=True,
                        entry_sidecar_representatives=representatives,
                    ),
                ]
            )
            if include_proxy_encoder_variants:
                profiles.append(
                    DocumentNodeServingProfile(
                        name=f"proxy_max_pool_f16_entry_sidecar{suffix}",
                        candidate_source="proxy_vector",
                        proxy_encoder="max_pool",
                        storage_kind="f16",
                        entry_sidecar=True,
                        entry_sidecar_representatives=representatives,
                    )
                )
    if include_proxy_encoder_variants:
        profiles.extend(
            [
                DocumentNodeServingProfile(
                    name="proxy_max_pool_f16",
                    candidate_source="proxy_vector",
                    proxy_encoder="max_pool",
                    storage_kind="f16",
                ),
                DocumentNodeServingProfile(
                    name="proxy_random_projection_fde_f16",
                    candidate_source="proxy_vector",
                    proxy_encoder="random_projection_fde",
                    storage_kind="f16",
                ),
            ]
        )
    if include_learned_projection:
        profiles.append(
            DocumentNodeServingProfile(
                name="proxy_learned_projection_v1_f16",
                candidate_source="proxy_vector",
                proxy_encoder="learned_projection_v1",
                storage_kind="f16",
            )
        )
    if include_bm25_rescue:
        profiles.extend(
            [
                DocumentNodeServingProfile(
                    name="proxy_normalized_mean_f16_bm25_rescue",
                    candidate_source="proxy_vector",
                    branch_plan="dense_only",
                    bm25_candidate_injection="dense_with_text",
                    proxy_encoder="normalized_mean",
                    storage_kind="f16",
                ),
                DocumentNodeServingProfile(
                    name="centroid_mean_f16_bm25_rescue",
                    candidate_source="proxy_vector",
                    branch_plan="dense_only",
                    bm25_candidate_injection="dense_with_text",
                    proxy_encoder="centroid_mean",
                    centroids="kmeans",
                    centroid_count=centroid_count,
                    storage_kind="f16",
                ),
            ]
        )
        if include_proxy_encoder_variants:
            profiles.append(
                DocumentNodeServingProfile(
                    name="proxy_max_pool_f16_bm25_rescue",
                    candidate_source="proxy_vector",
                    branch_plan="dense_only",
                    bm25_candidate_injection="dense_with_text",
                    proxy_encoder="max_pool",
                    storage_kind="f16",
                )
            )
    if include_learned_sparse_rescue:
        profiles.extend(
            [
                DocumentNodeServingProfile(
                    name="proxy_normalized_mean_f16_learned_sparse_rescue",
                    candidate_source="proxy_vector",
                    branch_plan="dense_only",
                    sparse_candidate_source="learned_sparse",
                    proxy_encoder="normalized_mean",
                    storage_kind="f16",
                ),
                DocumentNodeServingProfile(
                    name="centroid_mean_f16_learned_sparse_rescue",
                    candidate_source="proxy_vector",
                    branch_plan="dense_only",
                    sparse_candidate_source="learned_sparse",
                    proxy_encoder="centroid_mean",
                    centroids="kmeans",
                    centroid_count=centroid_count,
                    storage_kind="f16",
                ),
            ]
        )
        if include_proxy_encoder_variants:
            profiles.append(
                DocumentNodeServingProfile(
                    name="proxy_max_pool_f16_learned_sparse_rescue",
                    candidate_source="proxy_vector",
                    branch_plan="dense_only",
                    sparse_candidate_source="learned_sparse",
                    proxy_encoder="max_pool",
                    storage_kind="f16",
                )
            )
    if include_reservoirs:
        profiles.extend(
            [
                DocumentNodeServingProfile(
                    name="proxy_normalized_mean_f16_reservoir_balanced",
                    candidate_source="proxy_vector",
                    proxy_encoder="normalized_mean",
                    storage_kind="f16",
                    candidate_reservoirs="balanced",
                ),
                DocumentNodeServingProfile(
                    name="centroid_mean_f16_reservoir_balanced",
                    candidate_source="proxy_vector",
                    proxy_encoder="centroid_mean",
                    centroids="kmeans",
                    centroid_count=centroid_count,
                    storage_kind="f16",
                    candidate_reservoirs="balanced",
                ),
            ]
        )
        if include_proxy_encoder_variants:
            profiles.append(
                DocumentNodeServingProfile(
                    name="proxy_max_pool_f16_reservoir_balanced",
                    candidate_source="proxy_vector",
                    proxy_encoder="max_pool",
                    storage_kind="f16",
                    candidate_reservoirs="balanced",
                )
            )
    if include_experimental:
        profiles.append(
            DocumentNodeServingProfile(
                name="quantized_inverted_experimental_f32",
                candidate_source="quantized_inverted_experimental",
                storage_kind="f32",
                plain_fallback="off",
            )
        )
    return profiles


def validate_document_node_serving_profile_inputs(
    args: argparse.Namespace,
    profiles: Sequence[DocumentNodeServingProfile],
) -> None:
    if not any(profile.sparse_candidate_source == "learned_sparse" for profile in profiles):
        return
    require_complete_learned_sparse_args(args)
    if getattr(args, "learned_sparse_doc_jsonl", None) is None:
        raise SystemExit(
            "learned-sparse serving-grid profiles require "
            "--learned-sparse-doc-jsonl and --learned-sparse-query-jsonl"
        )


def effective_document_node_serving_profiles(
    args: argparse.Namespace,
) -> list[DocumentNodeServingProfile]:
    proxy_admission_focus = bool(
        getattr(args, "document_node_serving_grid_proxy_admission_focus", False)
    )
    centroid_lite_focus = bool(
        getattr(args, "document_node_serving_grid_centroid_lite_focus", False)
    )
    token_pooling_focus = bool(
        getattr(args, "document_node_serving_grid_token_pooling_focus", False)
    )
    profiles = document_node_serving_profiles(
        include_experimental=args.document_node_serving_grid_include_experimental,
        include_proxy_encoder_variants=(
            bool(getattr(args, "document_node_serving_grid_include_proxy_encoders", False))
            or proxy_admission_focus
        ),
        include_learned_projection=bool(
            getattr(args, "document_node_serving_grid_include_learned_projection", False)
        ),
        include_bm25_rescue=bool(
            getattr(args, "document_node_serving_grid_include_bm25_rescue", False)
        ),
        include_learned_sparse_rescue=bool(
            getattr(args, "document_node_serving_grid_include_learned_sparse_rescue", False)
        ),
        include_reservoirs=bool(
            getattr(args, "document_node_serving_grid_include_reservoirs", False)
        ),
        include_centroid_lite_caps=bool(
            getattr(args, "document_node_serving_grid_include_centroid_lite_caps", False)
        )
        or centroid_lite_focus,
        centroid_lite_posting_caps=effective_document_node_serving_centroid_lite_caps(args),
        include_token_pooling_focus=token_pooling_focus,
        include_entry_samples=bool(
            getattr(args, "document_node_serving_grid_include_entry_samples", False)
        )
        or proxy_admission_focus,
        entry_sample_counts=effective_document_node_serving_entry_sample_counts(args),
        include_entry_sidecar=bool(
            getattr(args, "document_node_serving_grid_include_entry_sidecar", False)
        )
        or proxy_admission_focus,
        entry_sidecar_representatives=(
            effective_document_node_serving_entry_sidecar_representatives(args)
            if proxy_admission_focus
            else ()
        ),
        explicit_entry_sidecar_profile_names=proxy_admission_focus,
        centroid_count=args.multivector_centroid_count,
    )
    if proxy_admission_focus:
        allowed = set(DOCUMENT_NODE_SERVING_GRID_PROXY_ADMISSION_FOCUS_PROFILES)
        order = {
            name: index
            for index, name in enumerate(
                DOCUMENT_NODE_SERVING_GRID_PROXY_ADMISSION_FOCUS_PROFILES
            )
        }
        profiles = sorted(
            [profile for profile in profiles if profile.name in allowed],
            key=lambda profile: order[profile.name],
        )
        if not profiles:
            raise SystemExit("--document-node-serving-grid-proxy-admission-focus produced no profiles")
    if centroid_lite_focus:
        allowed = set(DOCUMENT_NODE_SERVING_GRID_CENTROID_LITE_FOCUS_PROFILES)
        order = {
            name: index
            for index, name in enumerate(
                DOCUMENT_NODE_SERVING_GRID_CENTROID_LITE_FOCUS_PROFILES
            )
        }
        profiles = sorted(
            [profile for profile in profiles if profile.name in allowed],
            key=lambda profile: order[profile.name],
        )
        if not profiles:
            raise SystemExit("--document-node-serving-grid-centroid-lite-focus produced no profiles")
    if token_pooling_focus:
        allowed = set(DOCUMENT_NODE_SERVING_GRID_TOKEN_POOLING_FOCUS_PROFILES)
        order = {
            name: index
            for index, name in enumerate(
                DOCUMENT_NODE_SERVING_GRID_TOKEN_POOLING_FOCUS_PROFILES
            )
        }
        profiles = sorted(
            [profile for profile in profiles if profile.name in allowed],
            key=lambda profile: order[profile.name],
        )
        if not profiles:
            raise SystemExit("--document-node-serving-grid-token-pooling-focus produced no profiles")
    selected = str(getattr(args, "document_node_serving_grid_profiles", "") or "").strip()
    if selected:
        requested = [item.strip() for item in selected.split(",") if item.strip()]
        if not requested:
            raise SystemExit("--document-node-serving-grid-profiles must name at least one profile")
        by_name = {profile.name: profile for profile in profiles}
        unknown = [name for name in requested if name not in by_name]
        if unknown:
            known = ", ".join(sorted(by_name))
            raise SystemExit(
                "--document-node-serving-grid-profiles contains unknown profile(s): "
                f"{', '.join(unknown)}; expected one of: {known}"
            )
        profiles = [by_name[name] for name in dict.fromkeys(requested)]
    if bool(getattr(args, "document_node_serving_grid_smoke", False)):
        allowed = set(DOCUMENT_NODE_SERVING_GRID_SMOKE_PROFILES)
        profiles = [profile for profile in profiles if profile.name in allowed]
        if not profiles:
            raise SystemExit(
                "--document-node-serving-grid-smoke only supports profiles: "
                f"{', '.join(DOCUMENT_NODE_SERVING_GRID_SMOKE_PROFILES)}"
            )
    validate_document_node_serving_profile_inputs(args, profiles)
    return profiles


def document_node_serving_profile_by_name(
    args: argparse.Namespace,
    name: str,
) -> DocumentNodeServingProfile:
    proxy_admission_focus = bool(
        getattr(args, "document_node_serving_grid_proxy_admission_focus", False)
    )
    centroid_lite_focus = bool(
        getattr(args, "document_node_serving_grid_centroid_lite_focus", False)
    )
    token_pooling_focus = bool(
        getattr(args, "document_node_serving_grid_token_pooling_focus", False)
    )
    profiles = document_node_serving_profiles(
        include_experimental=args.document_node_serving_grid_include_experimental,
        include_proxy_encoder_variants=(
            bool(getattr(args, "document_node_serving_grid_include_proxy_encoders", False))
            or proxy_admission_focus
        ),
        include_learned_projection=bool(
            getattr(args, "document_node_serving_grid_include_learned_projection", False)
        ),
        include_bm25_rescue=bool(
            getattr(args, "document_node_serving_grid_include_bm25_rescue", False)
        ),
        include_learned_sparse_rescue=bool(
            getattr(args, "document_node_serving_grid_include_learned_sparse_rescue", False)
        ),
        include_reservoirs=bool(
            getattr(args, "document_node_serving_grid_include_reservoirs", False)
        ),
        include_centroid_lite_caps=bool(
            getattr(args, "document_node_serving_grid_include_centroid_lite_caps", False)
        )
        or centroid_lite_focus,
        centroid_lite_posting_caps=effective_document_node_serving_centroid_lite_caps(args),
        include_token_pooling_focus=token_pooling_focus,
        include_entry_samples=bool(
            getattr(args, "document_node_serving_grid_include_entry_samples", False)
        )
        or proxy_admission_focus,
        entry_sample_counts=effective_document_node_serving_entry_sample_counts(args),
        include_entry_sidecar=bool(
            getattr(args, "document_node_serving_grid_include_entry_sidecar", False)
        )
        or proxy_admission_focus,
        entry_sidecar_representatives=(
            effective_document_node_serving_entry_sidecar_representatives(args)
            if proxy_admission_focus
            else ()
        ),
        explicit_entry_sidecar_profile_names=proxy_admission_focus,
        centroid_count=args.multivector_centroid_count,
    )
    by_name = {profile.name: profile for profile in profiles}
    profile = by_name.get(name)
    if profile is not None:
        validate_document_node_serving_profile_inputs(args, [profile])
        return profile
    if name == "quantized_inverted_experimental_f32":
        raise SystemExit(
            "--serving-profile-name quantized_inverted_experimental_f32 requires "
            "--document-node-serving-grid-include-experimental"
        )
    known = ", ".join(sorted(by_name))
    raise SystemExit(f"unknown --serving-profile-name {name!r}; expected one of: {known}")


def document_node_serving_latency_profile(
    args: argparse.Namespace,
) -> DocumentNodeServingProfile:
    name = (
        str(args.serving_profile_name)
        if getattr(args, "serving_profile_name", None)
        else DOCUMENT_NODE_SERVING_LATENCY_DEFAULT_PROFILE
    )
    profile = document_node_serving_profile_by_name(args, name)
    storage = getattr(args, "serving_storage", None)
    cache = getattr(args, "serving_cache", None)
    if storage is not None:
        profile = replace(profile, storage_kind=str(storage))
    if cache is not None:
        profile = replace(profile, cache_mode=str(cache))
    return profile


def document_node_serving_latency_args(
    args: argparse.Namespace,
) -> tuple[argparse.Namespace, DocumentNodeServingProfile, int]:
    profile = document_node_serving_latency_profile(args)
    candidate_k = int(getattr(args, "serving_candidate_k", 0) or 0)
    if candidate_k < 1:
        raise SystemExit("--serving-candidate-k must be positive")
    requested_rerank_k = int(getattr(args, "serving_exact_rerank_k", 0) or 0)
    if requested_rerank_k < 1:
        raise SystemExit("--serving-exact-rerank-k must be positive")
    effective_rerank_k = min(requested_rerank_k, candidate_k)
    latency_args = document_node_serving_profile_args(
        args,
        profile,
        ef=int(args.serving_ef),
        oversampling=int(args.serving_oversampling),
        budget_sweep=[candidate_k],
    )
    latency_args = clone_args(
        latency_args,
        dense_k=candidate_k,
        multivector_doc_candidate_k=candidate_k,
        multivector_exact_rerank="topk",
        multivector_exact_rerank_k=effective_rerank_k,
        reuse_index=args.reuse_index,
        admission_debug_context="serving_latency_only",
    )
    return latency_args, profile, effective_rerank_k


def effective_document_node_serving_ef_grid(args: argparse.Namespace) -> list[int]:
    values = getattr(args, "document_node_serving_ef_grid_values", None)
    if values is not None:
        return list(values)
    if bool(getattr(args, "document_node_serving_grid_proxy_admission_focus", False)):
        return list(DOCUMENT_NODE_SERVING_GRID_PROXY_ADMISSION_FOCUS_EF)
    if bool(getattr(args, "document_node_serving_grid_smoke", False)):
        return list(DOCUMENT_NODE_SERVING_GRID_SMOKE_EF)
    return list(DOCUMENT_NODE_SERVING_GRID_EF)


def effective_document_node_serving_oversampling_grid(args: argparse.Namespace) -> list[int]:
    values = getattr(args, "document_node_serving_oversampling_grid_values", None)
    if values is not None:
        return list(values)
    if bool(getattr(args, "document_node_serving_grid_proxy_admission_focus", False)):
        return list(DOCUMENT_NODE_SERVING_GRID_PROXY_ADMISSION_FOCUS_OVERSAMPLING)
    if bool(getattr(args, "document_node_serving_grid_smoke", False)):
        return list(DOCUMENT_NODE_SERVING_GRID_SMOKE_OVERSAMPLING)
    return list(DOCUMENT_NODE_SERVING_GRID_OVERSAMPLING)


def document_node_serving_configs(
    profiles: list[DocumentNodeServingProfile],
    ef_grid: list[int],
    oversampling_grid: list[int],
) -> list[DocumentNodeServingConfig]:
    return [
        DocumentNodeServingConfig(profile=profile, ef=ef, oversampling=oversampling)
        for profile in profiles
        for ef in ef_grid
        for oversampling in oversampling_grid
    ]


def document_node_serving_config_id(config: DocumentNodeServingConfig) -> str:
    return f"{config.profile.name}_ef{int(config.ef)}_os{int(config.oversampling)}"


def serving_stage1_score_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered_by_latency = sorted(
        rows,
        key=lambda row: (
            serving_row_p95(row) if serving_row_p95(row) is not None else float("inf"),
            serving_row_id(row),
        ),
    )
    latency_rank = {
        serving_row_id(row): rank
        for rank, row in enumerate(ordered_by_latency, start=1)
    }
    scored: list[dict[str, Any]] = []
    for row in rows:
        admission = serving_row_metric(row, "exact_top10_admission_recall")
        admission_loss_penalty = round(max(0.0, 1.0 - (admission or 0.0)) * 100.0, 3)
        experimental_penalty = 1_000_000 if serving_row_experimental(row) else 0
        score = latency_rank.get(serving_row_id(row), len(rows) + 1) + admission_loss_penalty + experimental_penalty
        scored.append({
            **serving_recommendation_row(row),
            "stage1_score": round(score, 3),
            "stage1_score_components": {
                "p95_rank": latency_rank.get(serving_row_id(row), len(rows) + 1),
                "admission_loss_penalty": admission_loss_penalty,
                "experimental_penalty": experimental_penalty,
            },
        })
    return sorted(
        scored,
        key=lambda item: (
            float(item["stage1_score"]),
            serving_row_p95(item) if serving_row_p95(item) is not None else float("inf"),
            str(item.get("id", "")),
        ),
    )


def select_document_node_serving_stage2_configs(
    stage1_rows: list[dict[str, Any]],
    configs: list[DocumentNodeServingConfig],
    *,
    finalists: int,
) -> tuple[list[DocumentNodeServingConfig], list[dict[str, Any]], list[dict[str, Any]]]:
    config_by_id = {
        document_node_serving_config_id(config): config
        for config in configs
    }
    scored = serving_stage1_score_rows(stage1_rows)
    selected_ids: set[str] = set()
    selected_configs: list[DocumentNodeServingConfig] = []
    for item in scored:
        row_id = str(item.get("id", ""))
        if not row_id or row_id in selected_ids:
            continue
        if serving_row_experimental(item):
            continue
        config = config_by_id.get(row_id)
        if config is None:
            continue
        selected_ids.add(row_id)
        selected_configs.append(config)
        if len(selected_configs) >= finalists:
            break

    annotated_stage1: list[dict[str, Any]] = []
    score_by_id = {str(item.get("id", "")): item for item in scored}
    rank_by_id = {
        str(item.get("id", "")): rank
        for rank, item in enumerate(scored, start=1)
    }
    for row in stage1_rows:
        row_id = serving_row_id(row)
        score = score_by_id.get(row_id, {})
        annotated_stage1.append({
            **row,
            "stage1_score": score.get("stage1_score"),
            "stage1_score_components": score.get("stage1_score_components", {}),
            "stage1_rank": rank_by_id.get(row_id),
            "stage1_selected_finalist": row_id in selected_ids,
        })

    pruned: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in scored:
        row_id = str(item.get("id", ""))
        if not row_id or row_id in seen_ids or row_id in selected_ids:
            continue
        seen_ids.add(row_id)
        reason = (
            "experimental_not_eligible_for_finalist"
            if serving_row_experimental(item)
            else "not_in_top_finalists"
        )
        pruned.append({
            **item,
            "reason": reason,
        })
    return selected_configs, annotated_stage1, pruned


def _self_check_document_node_serving_profiles() -> None:
    base = document_node_serving_profiles(
        include_experimental=False,
        include_proxy_encoder_variants=False,
        include_bm25_rescue=False,
        include_learned_sparse_rescue=False,
        include_reservoirs=False,
        centroid_count="auto",
    )
    assert [profile.name for profile in base] == [
        "proxy_normalized_mean_f16",
        "docnodes_normalized_mean_f16",
        "centroid_mean_f16",
        "centroid_lite_f16",
        "centroid_lite_f16_pool_050",
        "proxy_normalized_mean_sq8",
    ]
    assert all(profile.candidate_source != "quantized_inverted_experimental" for profile in base)
    profiles_by_name = {profile.name: profile for profile in base}
    assert profiles_by_name["centroid_mean_f16"].proxy_encoder == "centroid_mean"
    assert profiles_by_name["centroid_mean_f16"].centroids == "kmeans"
    assert profiles_by_name["centroid_lite_f16_pool_050"].token_pooling == "greedy_cosine"
    assert profiles_by_name["centroid_lite_f16_pool_050"].token_pooling_target_ratio == 0.5
    assert profiles_by_name["proxy_normalized_mean_f16"].branch_plan == "dense_only"
    assert profiles_by_name["proxy_normalized_mean_sq8"].branch_plan == "dense_only"
    experimental = document_node_serving_profiles(
        include_experimental=True,
        include_proxy_encoder_variants=False,
        include_bm25_rescue=False,
        include_learned_sparse_rescue=False,
        include_reservoirs=False,
        centroid_count="auto",
    )
    assert experimental[-1].name == "quantized_inverted_experimental_f32"
    assert experimental[-1].candidate_source == "quantized_inverted_experimental"
    assert experimental[-1].storage_kind == "f32"
    assert experimental[-1].plain_fallback == "off"
    proxy_variants = document_node_serving_profiles(
        include_experimental=False,
        include_proxy_encoder_variants=True,
        include_bm25_rescue=False,
        include_learned_sparse_rescue=False,
        include_reservoirs=False,
        centroid_count="auto",
    )
    proxy_variants_by_name = {profile.name: profile for profile in proxy_variants}
    assert proxy_variants_by_name["proxy_max_pool_f16"].proxy_encoder == "max_pool"
    assert (
        proxy_variants_by_name["proxy_random_projection_fde_f16"].proxy_encoder
        == "random_projection_fde"
    )
    entry_sidecar_profiles = document_node_serving_profiles(
        include_experimental=False,
        include_proxy_encoder_variants=False,
        include_bm25_rescue=False,
        include_learned_sparse_rescue=False,
        include_reservoirs=False,
        include_entry_sidecar=True,
        centroid_count="auto",
    )
    entry_sidecar_by_name = {profile.name: profile for profile in entry_sidecar_profiles}
    assert "proxy_normalized_mean_f16_entry_sidecar" in entry_sidecar_by_name
    assert "centroid_mean_f16_entry_sidecar" in entry_sidecar_by_name
    assert entry_sidecar_by_name["proxy_normalized_mean_f16_entry_sidecar"].entry_sidecar is True
    assert (
        entry_sidecar_by_name["centroid_mean_f16_entry_sidecar"].proxy_encoder
        == "centroid_mean"
    )
    assert (
        entry_sidecar_by_name["centroid_mean_f16_entry_sidecar"].entry_sidecar_strategy
        == "hybrid_level_covering"
    )
    entry_sidecar_proxy_variants = document_node_serving_profiles(
        include_experimental=False,
        include_proxy_encoder_variants=True,
        include_bm25_rescue=False,
        include_learned_sparse_rescue=False,
        include_reservoirs=False,
        include_entry_sidecar=True,
        centroid_count="auto",
    )
    entry_sidecar_proxy_variants_by_name = {
        profile.name: profile for profile in entry_sidecar_proxy_variants
    }
    assert (
        entry_sidecar_proxy_variants_by_name[
            "proxy_max_pool_f16_entry_sidecar"
        ].proxy_encoder
        == "max_pool"
    )
    assert (
        entry_sidecar_proxy_variants_by_name[
            "proxy_max_pool_f16_entry_sidecar"
        ].entry_sidecar
        is True
    )
    bm25_rescue = document_node_serving_profiles(
        include_experimental=False,
        include_proxy_encoder_variants=False,
        include_bm25_rescue=True,
        include_learned_sparse_rescue=False,
        include_reservoirs=False,
        centroid_count="auto",
    )
    bm25_rescue_by_name = {profile.name: profile for profile in bm25_rescue}
    assert "proxy_normalized_mean_f16_bm25_rescue" in bm25_rescue_by_name
    assert "centroid_mean_f16_bm25_rescue" in bm25_rescue_by_name
    assert (
        bm25_rescue_by_name["proxy_normalized_mean_f16_bm25_rescue"].bm25_candidate_injection
        == "dense_with_text"
    )
    assert (
        bm25_rescue_by_name["proxy_normalized_mean_f16_bm25_rescue"].branch_plan
        == "dense_only"
    )
    assert (
        bm25_rescue_by_name["centroid_mean_f16_bm25_rescue"].proxy_encoder
        == "centroid_mean"
    )
    assert (
        bm25_rescue_by_name["centroid_mean_f16_bm25_rescue"].branch_plan
        == "dense_only"
    )
    assert "proxy_max_pool_f16_bm25_rescue" not in bm25_rescue_by_name
    bm25_rescue_with_proxy_variants = document_node_serving_profiles(
        include_experimental=False,
        include_proxy_encoder_variants=True,
        include_bm25_rescue=True,
        include_learned_sparse_rescue=False,
        include_reservoirs=False,
        centroid_count="auto",
    )
    bm25_rescue_with_proxy_variants_by_name = {
        profile.name: profile for profile in bm25_rescue_with_proxy_variants
    }
    assert "proxy_max_pool_f16_bm25_rescue" in bm25_rescue_with_proxy_variants_by_name
    assert (
        bm25_rescue_with_proxy_variants_by_name[
            "proxy_max_pool_f16_bm25_rescue"
        ].proxy_encoder
        == "max_pool"
    )
    assert (
        bm25_rescue_with_proxy_variants_by_name[
            "proxy_max_pool_f16_bm25_rescue"
        ].bm25_candidate_injection
        == "dense_with_text"
    )
    learned_sparse_rescue = document_node_serving_profiles(
        include_experimental=False,
        include_proxy_encoder_variants=False,
        include_bm25_rescue=False,
        include_learned_sparse_rescue=True,
        include_reservoirs=False,
        centroid_count="auto",
    )
    learned_sparse_rescue_by_name = {
        profile.name: profile for profile in learned_sparse_rescue
    }
    assert "proxy_normalized_mean_f16_learned_sparse_rescue" in learned_sparse_rescue_by_name
    assert "centroid_mean_f16_learned_sparse_rescue" in learned_sparse_rescue_by_name
    assert (
        learned_sparse_rescue_by_name[
            "proxy_normalized_mean_f16_learned_sparse_rescue"
        ].sparse_candidate_source
        == "learned_sparse"
    )
    assert (
        learned_sparse_rescue_by_name[
            "proxy_normalized_mean_f16_learned_sparse_rescue"
        ].branch_plan
        == "dense_only"
    )
    assert (
        learned_sparse_rescue_by_name[
            "centroid_mean_f16_learned_sparse_rescue"
        ].proxy_encoder
        == "centroid_mean"
    )
    assert (
        learned_sparse_rescue_by_name[
            "centroid_mean_f16_learned_sparse_rescue"
        ].centroids
        == "kmeans"
    )
    assert "proxy_max_pool_f16_learned_sparse_rescue" not in learned_sparse_rescue_by_name
    learned_sparse_rescue_with_proxy_variants = document_node_serving_profiles(
        include_experimental=False,
        include_proxy_encoder_variants=True,
        include_bm25_rescue=False,
        include_learned_sparse_rescue=True,
        include_reservoirs=False,
        centroid_count="auto",
    )
    learned_sparse_rescue_with_proxy_variants_by_name = {
        profile.name: profile for profile in learned_sparse_rescue_with_proxy_variants
    }
    assert (
        "proxy_max_pool_f16_learned_sparse_rescue"
        in learned_sparse_rescue_with_proxy_variants_by_name
    )
    assert (
        learned_sparse_rescue_with_proxy_variants_by_name[
            "proxy_max_pool_f16_learned_sparse_rescue"
        ].proxy_encoder
        == "max_pool"
    )
    assert (
        learned_sparse_rescue_with_proxy_variants_by_name[
            "proxy_max_pool_f16_learned_sparse_rescue"
        ].sparse_candidate_source
        == "learned_sparse"
    )
    reservoir_profiles = document_node_serving_profiles(
        include_experimental=False,
        include_proxy_encoder_variants=False,
        include_bm25_rescue=False,
        include_learned_sparse_rescue=False,
        include_reservoirs=True,
        centroid_count="auto",
    )
    reservoir_profiles_by_name = {
        profile.name: profile for profile in reservoir_profiles
    }
    assert "proxy_normalized_mean_f16_reservoir_balanced" in reservoir_profiles_by_name
    assert "centroid_mean_f16_reservoir_balanced" in reservoir_profiles_by_name
    assert (
        reservoir_profiles_by_name[
            "proxy_normalized_mean_f16_reservoir_balanced"
        ].candidate_reservoirs
        == "balanced"
    )
    assert (
        reservoir_profiles_by_name[
            "centroid_mean_f16_reservoir_balanced"
        ].proxy_encoder
        == "centroid_mean"
    )
    assert (
        reservoir_profiles_by_name[
            "centroid_mean_f16_reservoir_balanced"
        ].centroids
        == "kmeans"
    )
    assert "proxy_max_pool_f16_reservoir_balanced" not in reservoir_profiles_by_name
    reservoir_profiles_with_proxy_variants = document_node_serving_profiles(
        include_experimental=False,
        include_proxy_encoder_variants=True,
        include_bm25_rescue=False,
        include_learned_sparse_rescue=False,
        include_reservoirs=True,
        centroid_count="auto",
    )
    reservoir_profiles_with_proxy_variants_by_name = {
        profile.name: profile for profile in reservoir_profiles_with_proxy_variants
    }
    assert (
        "proxy_max_pool_f16_reservoir_balanced"
        in reservoir_profiles_with_proxy_variants_by_name
    )
    assert (
        reservoir_profiles_with_proxy_variants_by_name[
            "proxy_max_pool_f16_reservoir_balanced"
        ].proxy_encoder
        == "max_pool"
    )
    assert (
        reservoir_profiles_with_proxy_variants_by_name[
            "proxy_max_pool_f16_reservoir_balanced"
        ].candidate_reservoirs
        == "balanced"
    )
    capped_centroid_lite = document_node_serving_profiles(
        include_experimental=False,
        include_proxy_encoder_variants=False,
        include_bm25_rescue=False,
        include_learned_sparse_rescue=False,
        include_reservoirs=False,
        include_centroid_lite_caps=True,
        centroid_lite_posting_caps=[16, 32],
        centroid_count="auto",
    )
    capped_centroid_lite_by_name = {
        profile.name: profile for profile in capped_centroid_lite
    }
    assert "centroid_lite_f16_cap_016" in capped_centroid_lite_by_name
    assert "centroid_lite_f16_cap_032" in capped_centroid_lite_by_name
    assert "centroid_lite_f16_prune_safe_upper_bound" in capped_centroid_lite_by_name
    assert "centroid_lite_f16_cap_016_prune_safe_upper_bound" in capped_centroid_lite_by_name
    assert (
        capped_centroid_lite_by_name[
            "centroid_lite_f16_cap_016"
        ].centroid_lite_max_postings_per_token
        == 16
    )
    assert (
        capped_centroid_lite_by_name["centroid_lite_f16_cap_016"].candidate_source
        == "centroid_lite"
    )
    assert capped_centroid_lite_by_name["centroid_lite_f16_cap_016"].centroids == "kmeans"
    assert (
        capped_centroid_lite_by_name[
            "centroid_lite_f16_cap_016_prune_safe_upper_bound"
        ].centroid_lite_pruning
        == "safe_upper_bound"
    )
    entry_sample_profiles = document_node_serving_profiles(
        include_experimental=False,
        include_proxy_encoder_variants=False,
        include_bm25_rescue=False,
        include_learned_sparse_rescue=False,
        include_reservoirs=False,
        include_entry_samples=True,
        entry_sample_counts=[32],
        centroid_count="auto",
    )
    entry_sample_profiles_by_name = {
        profile.name: profile for profile in entry_sample_profiles
    }
    assert "proxy_normalized_mean_f16_entry_sample_032" in entry_sample_profiles_by_name
    assert "centroid_mean_f16_entry_sample_032" in entry_sample_profiles_by_name
    assert (
        entry_sample_profiles_by_name[
            "proxy_normalized_mean_f16_entry_sample_032"
        ].entry_sample_count
        == 32
    )
    assert (
        entry_sample_profiles_by_name[
            "centroid_mean_f16_entry_sample_032"
        ].proxy_encoder
        == "centroid_mean"
    )
    entry_sample_proxy_variants = document_node_serving_profiles(
        include_experimental=False,
        include_proxy_encoder_variants=True,
        include_bm25_rescue=False,
        include_learned_sparse_rescue=False,
        include_reservoirs=False,
        include_entry_samples=True,
        entry_sample_counts=[128],
        centroid_count="auto",
    )
    entry_sample_proxy_variants_by_name = {
        profile.name: profile for profile in entry_sample_proxy_variants
    }
    assert "proxy_max_pool_f16_entry_sample_128" in entry_sample_proxy_variants_by_name
    assert (
        entry_sample_proxy_variants_by_name[
            "proxy_max_pool_f16_entry_sample_128"
        ].entry_sample_count
        == 128
    )
    assert all(
        profile.bm25_candidate_injection == "off"
        for profile in base
    )
    dense_query_args = argparse.Namespace(
        multivector_bm25_candidate_injection="off",
        multivector_sparse_candidate_source="off",
    )
    bm25_query_args = argparse.Namespace(
        multivector_bm25_candidate_injection="dense_with_text",
        multivector_sparse_candidate_source="off",
    )
    learned_sparse_query_args = argparse.Namespace(
        multivector_bm25_candidate_injection="off",
        multivector_sparse_candidate_source="learned_sparse",
    )
    assert query_only_needs_text_query(dense_query_args) is False
    assert query_only_needs_text_query(bm25_query_args) is True
    assert query_only_needs_text_query(learned_sparse_query_args) is True
    assert query_only_text_query_sql(bm25_query_args) == "websearch_to_tsquery('simple', q.query_text)"
    assert "turbohybrid_sparse_vector_to_tsquery" in query_only_text_query_sql(
        learned_sparse_query_args
    )
    assert build_index_lexical_column(dense_query_args) is None
    assert build_index_lexical_column(bm25_query_args) == "body_tsv"
    assert build_index_lexical_column(learned_sparse_query_args) == "learned_sparse_tsv"
    partial_sparse_coverage = {
        "loaded_documents": 100,
        "learned_sparse_documents": 20,
        "doc_coverage_ratio": learned_sparse_ratio(20, 100),
        "loaded_queries": 10,
        "learned_sparse_queries": 10,
        "query_coverage_ratio": learned_sparse_ratio(10, 10),
    }
    assert partial_sparse_coverage["doc_coverage_ratio"] == 0.2
    assert learned_sparse_coverage_warnings(partial_sparse_coverage) == [
        "learned_sparse_doc_coverage_below_95pct"
    ]
    complete_sparse_coverage = {
        "loaded_documents": 100,
        "learned_sparse_documents": 100,
        "doc_coverage_ratio": learned_sparse_ratio(100, 100),
        "loaded_queries": 10,
        "learned_sparse_queries": 10,
        "query_coverage_ratio": learned_sparse_ratio(10, 10),
    }
    assert learned_sparse_coverage_warnings(complete_sparse_coverage) == []
    try:
        build_index_lexical_column(
            argparse.Namespace(
                methods=[],
                hybrid_evaluation_harness=False,
                multivector_bm25_candidate_injection="dense_with_text",
                multivector_sparse_candidate_source="learned_sparse",
            )
        )
    except SystemExit as exc:
        assert "different lexical index columns" in str(exc)
    else:
        raise AssertionError("mixed BM25 and learned-sparse lexical sources should fail")
    smoke_args = argparse.Namespace(
        admission_budget_sweep=None,
        admission_budget_sweep_explicit=False,
        document_node_serving_grid=True,
        document_node_serving_grid_smoke=True,
        document_node_serving_grid_include_experimental=True,
        document_node_serving_grid_include_proxy_encoders=False,
        document_node_serving_grid_include_learned_projection=False,
        document_node_serving_grid_include_bm25_rescue=False,
        document_node_serving_grid_include_learned_sparse_rescue=False,
        document_node_serving_grid_include_reservoirs=False,
        document_node_serving_grid_include_centroid_lite_caps=False,
        document_node_serving_grid_include_entry_samples=False,
        document_node_serving_grid_entry_sample_counts=(
            DOCUMENT_NODE_SERVING_GRID_ENTRY_SAMPLE_SWEEP
        ),
        document_node_serving_grid_include_entry_sidecar=False,
        document_node_serving_grid_entry_sidecar_representatives=(
            DOCUMENT_NODE_SERVING_GRID_ENTRY_SIDECAR_SWEEP
        ),
        document_node_serving_grid_proxy_admission_focus=False,
        document_node_serving_grid_centroid_lite_focus=False,
        document_node_serving_grid_token_pooling_focus=False,
        document_node_serving_grid_centroid_lite_posting_caps=(
            DOCUMENT_NODE_SERVING_GRID_CENTROID_LITE_CAP_SWEEP
        ),
        document_node_serving_grid_profiles="",
        document_node_serving_grid_budget_mode="largest_only",
        document_node_serving_ef_grid_values=None,
        document_node_serving_oversampling_grid_values=None,
        multivector_centroid_count="auto",
    )
    assert effective_serving_grid_budget_sweep(smoke_args) == [200, 800]
    assert effective_serving_grid_executed_budgets(smoke_args) == [800]
    serving_rerank_args = clone_args(
        smoke_args,
        serving_exact_rerank_mode="serving",
        serving_exact_rerank_k=100,
        multivector_exact_rerank_k=25,
    )
    assert effective_admission_exact_rerank_k(serving_rerank_args, 800) == 100
    assert effective_admission_exact_rerank_k(serving_rerank_args, 50) == 50
    exhaustive_rerank_args = clone_args(
        serving_rerank_args,
        serving_exact_rerank_mode="admission_exhaustive",
        multivector_exact_rerank_k=120,
    )
    assert effective_admission_exact_rerank_k(exhaustive_rerank_args, 800) == 800
    assert effective_admission_exact_rerank_k(exhaustive_rerank_args, 50) == 120
    assert [profile.name for profile in effective_document_node_serving_profiles(smoke_args)] == [
        "proxy_normalized_mean_f16",
        "centroid_mean_f16",
        "centroid_lite_f16",
    ]
    assert effective_document_node_serving_ef_grid(smoke_args) == [50, 100]
    assert effective_document_node_serving_oversampling_grid(smoke_args) == [1]
    smoke_override_args = clone_args(
        smoke_args,
        document_node_serving_ef_grid_values=[400, 800],
        document_node_serving_oversampling_grid_values=[1, 2],
    )
    assert effective_document_node_serving_ef_grid(smoke_override_args) == [400, 800]
    assert effective_document_node_serving_oversampling_grid(smoke_override_args) == [1, 2]
    assert effective_document_node_serving_stage_mode(smoke_args) == "two_stage"
    full_args = clone_args(
        smoke_args,
        document_node_serving_grid_smoke=False,
    )
    assert effective_serving_grid_budget_sweep(full_args) == [50, 100, 200, 400, 800]
    assert effective_serving_grid_executed_budgets(full_args) == [800]
    assert effective_document_node_serving_ef_grid(full_args) == [50, 100, 200]
    assert effective_document_node_serving_oversampling_grid(full_args) == [1, 2]
    assert effective_document_node_serving_stage_mode(full_args) == "single"
    assert effective_document_node_serving_stage_mode(
        clone_args(full_args, document_node_serving_grid_stage_mode="two_stage")
    ) == "two_stage"
    proxy_focus_args = clone_args(
        full_args,
        document_node_serving_grid_proxy_admission_focus=True,
    )
    assert effective_serving_grid_budget_sweep(proxy_focus_args) == [800]
    assert effective_serving_grid_executed_budgets(proxy_focus_args) == [800]
    assert effective_document_node_serving_ef_grid(proxy_focus_args) == [100, 200, 400]
    assert effective_document_node_serving_oversampling_grid(proxy_focus_args) == [1, 2]
    proxy_focus_profiles = effective_document_node_serving_profiles(proxy_focus_args)
    assert [profile.name for profile in proxy_focus_profiles] == list(
        DOCUMENT_NODE_SERVING_GRID_PROXY_ADMISSION_FOCUS_PROFILES
    )
    proxy_focus_by_name = {profile.name: profile for profile in proxy_focus_profiles}
    assert (
        proxy_focus_by_name[
            "proxy_max_pool_f16_entry_sample_032"
        ].entry_sample_count
        == 32
    )
    assert (
        proxy_focus_by_name[
            "proxy_normalized_mean_f16_entry_sidecar_256"
        ].entry_sidecar_representatives
        == 256
    )
    centroid_focus_args = clone_args(
        full_args,
        document_node_serving_grid_centroid_lite_focus=True,
    )
    assert effective_serving_grid_budget_sweep(centroid_focus_args) == [800]
    assert effective_serving_grid_executed_budgets(centroid_focus_args) == [800]
    centroid_focus_profiles = effective_document_node_serving_profiles(centroid_focus_args)
    assert [profile.name for profile in centroid_focus_profiles] == list(
        DOCUMENT_NODE_SERVING_GRID_CENTROID_LITE_FOCUS_PROFILES
    )
    centroid_focus_by_name = {profile.name: profile for profile in centroid_focus_profiles}
    assert (
        centroid_focus_by_name[
            "centroid_lite_f16_cap_016"
        ].centroid_lite_max_postings_per_token
        == 16
    )
    assert (
        centroid_focus_by_name[
            "centroid_lite_f16_cap_032"
        ].centroid_lite_max_postings_per_token
        == 32
    )
    assert (
        centroid_focus_by_name["centroid_lite_f16_pool_050"].token_pooling
        == "greedy_cosine"
    )
    assert centroid_focus_by_name["centroid_mean_f16"].candidate_source == "proxy_vector"
    centroid_warning_profile = DocumentNodeServingProfile(
        name="centroid_lite_f16_cap_016",
        candidate_source="centroid_lite",
        centroids="kmeans",
        centroid_lite_max_postings_per_token=16,
    )
    centroid_warnings = centroid_lite_work_warnings(
        profile=centroid_warning_profile,
        largest_work={
            "centroid_docs_touched": {"count": 3, "p95": 900.0},
            "centroid_postings_touched": {"count": 3, "p95": 10000.0},
            "centroid_postings_skipped": {"count": 3, "p95": 5000.0},
            "centroid_candidates": {"count": 3, "p95": 100.0},
        },
        loaded_document_count=1000,
        candidate_budget=800,
    )
    assert "centroid_lite_near_exhaustive_docs_touched" in centroid_warnings
    assert "centroid_lite_postings_near_exhaustive" in centroid_warnings
    assert "centroid_lite_cap_too_aggressive" in centroid_warnings
    token_pooling_focus_args = clone_args(
        full_args,
        document_node_serving_grid_token_pooling_focus=True,
    )
    token_pooling_profiles = effective_document_node_serving_profiles(
        token_pooling_focus_args
    )
    assert [profile.name for profile in token_pooling_profiles] == list(
        DOCUMENT_NODE_SERVING_GRID_TOKEN_POOLING_FOCUS_PROFILES
    )
    token_pooling_by_name = {
        profile.name: profile for profile in token_pooling_profiles
    }
    assert token_pooling_by_name["proxy_normalized_mean_f16"].token_pooling == "off"
    assert (
        token_pooling_by_name[
            "proxy_normalized_mean_f16_pool_075"
        ].token_pooling_target_ratio
        == 0.75
    )
    assert (
        token_pooling_by_name["centroid_mean_f16_pool_033"].proxy_encoder
        == "centroid_mean"
    )
    assert (
        token_pooling_by_name["centroid_mean_f16_pool_033"].centroids
        == "kmeans"
    )
    assert (
        serving_profile_index_signature(
            token_pooling_focus_args,
            token_pooling_by_name["proxy_normalized_mean_f16"],
        )
        != serving_profile_index_signature(
            token_pooling_focus_args,
            token_pooling_by_name["proxy_normalized_mean_f16_pool_050"],
        )
    )
    filtered_args = clone_args(
        full_args,
        document_node_serving_grid_profiles="centroid_mean_f16,proxy_normalized_mean_f16",
    )
    assert [profile.name for profile in effective_document_node_serving_profiles(filtered_args)] == [
        "centroid_mean_f16",
        "proxy_normalized_mean_f16",
    ]
    proxy_variant_filtered_args = clone_args(
        full_args,
        document_node_serving_grid_include_proxy_encoders=True,
        document_node_serving_grid_profiles="proxy_max_pool_f16,proxy_random_projection_fde_f16",
    )
    assert [profile.name for profile in effective_document_node_serving_profiles(proxy_variant_filtered_args)] == [
        "proxy_max_pool_f16",
        "proxy_random_projection_fde_f16",
    ]
    learned_projection_filtered_args = clone_args(
        full_args,
        document_node_serving_grid_include_learned_projection=True,
        document_node_serving_grid_profiles="proxy_learned_projection_v1_f16",
    )
    learned_projection_filtered = effective_document_node_serving_profiles(
        learned_projection_filtered_args
    )
    assert [profile.name for profile in learned_projection_filtered] == [
        "proxy_learned_projection_v1_f16",
    ]
    assert learned_projection_filtered[0].proxy_encoder == "learned_projection_v1"
    try:
        effective_document_node_serving_profiles(
            clone_args(
                full_args,
                document_node_serving_grid_profiles="proxy_learned_projection_v1_f16",
            )
        )
    except SystemExit as exc:
        assert "unknown profile" in str(exc)
    else:
        raise AssertionError("learned-projection profile should require opt-in flag")
    capped_filtered_args = clone_args(
        full_args,
        document_node_serving_grid_include_centroid_lite_caps=True,
        document_node_serving_grid_centroid_lite_posting_caps="16,32",
        document_node_serving_grid_profiles="centroid_lite_f16_cap_016",
    )
    capped_filtered = effective_document_node_serving_profiles(capped_filtered_args)
    assert [profile.name for profile in capped_filtered] == [
        "centroid_lite_f16_cap_016",
    ]
    assert capped_filtered[0].centroid_lite_max_postings_per_token == 16
    try:
        effective_document_node_serving_profiles(
            clone_args(
                full_args,
                document_node_serving_grid_profiles="centroid_lite_f16_cap_016",
            )
        )
    except SystemExit as exc:
        assert "unknown profile" in str(exc)
    else:
        raise AssertionError("capped centroid-lite profile should require opt-in flag")
    entry_sample_filtered_args = clone_args(
        full_args,
        document_node_serving_grid_include_entry_samples=True,
        document_node_serving_grid_entry_sample_counts="32,128",
        document_node_serving_grid_profiles="proxy_normalized_mean_f16_entry_sample_032",
    )
    entry_sample_filtered = effective_document_node_serving_profiles(
        entry_sample_filtered_args
    )
    assert [profile.name for profile in entry_sample_filtered] == [
        "proxy_normalized_mean_f16_entry_sample_032",
    ]
    assert entry_sample_filtered[0].entry_sample_count == 32
    entry_sample_profile_args = document_node_serving_profile_args(
        full_args,
        entry_sample_filtered[0],
        ef=100,
        oversampling=1,
    )
    assert entry_sample_profile_args.multivector_doc_graph_entry_sample_count == 32
    try:
        effective_document_node_serving_profiles(
            clone_args(
                full_args,
                document_node_serving_grid_profiles="proxy_normalized_mean_f16_entry_sample_032",
            )
        )
    except SystemExit as exc:
        assert "unknown profile" in str(exc)
    else:
        raise AssertionError("entry-sample profile should require opt-in flag")
    entry_sidecar_filtered_args = clone_args(
        full_args,
        document_node_serving_grid_include_entry_sidecar=True,
        document_node_serving_grid_profiles="proxy_normalized_mean_f16_entry_sidecar",
    )
    entry_sidecar_filtered = effective_document_node_serving_profiles(
        entry_sidecar_filtered_args
    )
    assert [profile.name for profile in entry_sidecar_filtered] == [
        "proxy_normalized_mean_f16_entry_sidecar",
    ]
    assert entry_sidecar_filtered[0].entry_sidecar is True
    try:
        effective_document_node_serving_profiles(
            clone_args(
                full_args,
                document_node_serving_grid_profiles="proxy_normalized_mean_f16_entry_sidecar",
            )
        )
    except SystemExit as exc:
        assert "unknown profile" in str(exc)
    else:
        raise AssertionError("entry-sidecar profile should require opt-in flag")
    bm25_rescue_filtered_args = clone_args(
        full_args,
        document_node_serving_grid_include_bm25_rescue=True,
        document_node_serving_grid_profiles="centroid_mean_f16_bm25_rescue",
    )
    bm25_rescue_filtered = effective_document_node_serving_profiles(
        bm25_rescue_filtered_args
    )
    assert [profile.name for profile in bm25_rescue_filtered] == [
        "centroid_mean_f16_bm25_rescue",
    ]
    assert bm25_rescue_filtered[0].bm25_candidate_injection == "dense_with_text"
    learned_sparse_rescue_filtered_args = clone_args(
        full_args,
        document_node_serving_grid_include_learned_sparse_rescue=True,
        document_node_serving_grid_profiles="proxy_normalized_mean_f16_learned_sparse_rescue",
        learned_sparse_doc_jsonl=Path("learned-docs.jsonl"),
        learned_sparse_query_jsonl=Path("learned-queries.jsonl"),
    )
    learned_sparse_rescue_filtered = effective_document_node_serving_profiles(
        learned_sparse_rescue_filtered_args
    )
    assert [profile.name for profile in learned_sparse_rescue_filtered] == [
        "proxy_normalized_mean_f16_learned_sparse_rescue",
    ]
    assert learned_sparse_rescue_filtered[0].sparse_candidate_source == "learned_sparse"
    try:
        effective_document_node_serving_profiles(
            clone_args(
                full_args,
                document_node_serving_grid_include_learned_sparse_rescue=True,
                document_node_serving_grid_profiles="proxy_normalized_mean_f16_learned_sparse_rescue",
                learned_sparse_doc_jsonl=None,
                learned_sparse_query_jsonl=None,
            )
        )
    except SystemExit as exc:
        assert "learned-sparse serving-grid profiles require" in str(exc)
    else:
        raise AssertionError("learned-sparse serving-grid profile should require JSONL inputs")
    reservoir_filtered_args = clone_args(
        full_args,
        document_node_serving_grid_include_reservoirs=True,
        document_node_serving_grid_profiles="proxy_normalized_mean_f16_reservoir_balanced",
    )
    reservoir_filtered = effective_document_node_serving_profiles(
        reservoir_filtered_args
    )
    assert [profile.name for profile in reservoir_filtered] == [
        "proxy_normalized_mean_f16_reservoir_balanced",
    ]
    assert reservoir_filtered[0].candidate_reservoirs == "balanced"
    try:
        effective_document_node_serving_profiles(
            clone_args(
                full_args,
                document_node_serving_grid_profiles="proxy_normalized_mean_f16_reservoir_balanced",
            )
        )
    except SystemExit as exc:
        assert "unknown profile" in str(exc)
    else:
        raise AssertionError("reservoir profile should require opt-in flag")
    smoke_filtered_args = clone_args(
        smoke_args,
        document_node_serving_grid_profiles="proxy_normalized_mean_f16,proxy_normalized_mean_sq8",
    )
    assert [profile.name for profile in effective_document_node_serving_profiles(smoke_filtered_args)] == [
        "proxy_normalized_mean_f16",
    ]
    try:
        effective_document_node_serving_profiles(
            clone_args(full_args, document_node_serving_grid_profiles="missing_profile")
        )
    except SystemExit as exc:
        assert "unknown profile" in str(exc)
    else:
        raise AssertionError("unknown serving-grid profile should fail")
    sweep_args = clone_args(
        full_args,
        document_node_serving_grid_budget_mode="sweep",
    )
    assert effective_serving_grid_executed_budgets(sweep_args) == [50, 100, 200, 400, 800]
    explicit_args = clone_args(
        smoke_args,
        admission_budget_sweep="25,50",
        admission_budget_sweep_explicit=True,
    )
    assert effective_serving_grid_budget_sweep(explicit_args) == [25, 50]
    assert effective_serving_grid_executed_budgets(explicit_args) == [50]
    non_serving_args = clone_args(
        smoke_args,
        admission_budget_sweep=None,
        document_node_serving_grid=False,
        document_node_serving_grid_smoke=False,
        document_node_serving_grid_budget_mode="sweep",
    )
    assert effective_serving_grid_budget_sweep(non_serving_args) == [
        100,
        200,
        400,
        800,
        1600,
        3200,
        6400,
        10000,
    ]
    debug_default_args = argparse.Namespace(admission_debug_mode=None)
    assert effective_admission_debug_mode(
        clone_args(debug_default_args, admission_debug_context="explicit")
    ) == "trace"
    assert effective_admission_debug_mode(
        clone_args(debug_default_args, admission_debug_context="admission_grid")
    ) == "trace"
    assert effective_admission_debug_mode(
        clone_args(debug_default_args, admission_debug_context="serving_grid")
    ) == "summary"
    assert effective_admission_debug_mode(
        clone_args(
            debug_default_args,
            admission_debug_mode="off",
            admission_debug_context="serving_grid",
        )
    ) == "off"
    signature_args = clone_args(
        smoke_args,
        multivector_doc_build_scorer="proxy",
        multivector_token_pooling_min_tokens=16,
        index_graph_m=0,
        index_graph_ef_construction=0,
        index_graph_ef_search=0,
        index_native_segments=0,
    )
    signature_by_name = {
        profile.name: serving_profile_index_signature(signature_args, profile)
        for profile in base
    }
    proxy_variant_signature_by_name = {
        profile.name: serving_profile_index_signature(signature_args, profile)
        for profile in proxy_variants
    }
    bm25_signature_by_name = {
        profile.name: serving_profile_index_signature(signature_args, profile)
        for profile in bm25_rescue
    }
    learned_sparse_signature_by_name = {
        profile.name: serving_profile_index_signature(signature_args, profile)
        for profile in learned_sparse_rescue
    }
    capped_signature_by_name = {
        profile.name: serving_profile_index_signature(signature_args, profile)
        for profile in capped_centroid_lite
    }
    entry_sample_signature_by_name = {
        profile.name: serving_profile_index_signature(signature_args, profile)
        for profile in entry_sample_profiles
    }
    entry_sidecar_signature_by_name = {
        profile.name: serving_profile_index_signature(signature_args, profile)
        for profile in entry_sidecar_profiles
    }
    entry_sidecar_proxy_variant_signature_by_name = {
        profile.name: serving_profile_index_signature(signature_args, profile)
        for profile in entry_sidecar_proxy_variants
    }
    proxy_focus_signature_by_name = {
        profile.name: serving_profile_index_signature(signature_args, profile)
        for profile in proxy_focus_profiles
    }
    assert signature_by_name["proxy_normalized_mean_f16"] == signature_by_name["docnodes_normalized_mean_f16"]
    assert signature_by_name["proxy_normalized_mean_f16"] == signature_by_name["proxy_normalized_mean_sq8"]
    assert signature_by_name["proxy_normalized_mean_f16"] != bm25_signature_by_name["proxy_normalized_mean_f16_bm25_rescue"]
    assert signature_by_name["centroid_mean_f16"] != bm25_signature_by_name["centroid_mean_f16_bm25_rescue"]
    assert signature_by_name["centroid_mean_f16"] != signature_by_name["centroid_lite_f16"]
    assert signature_by_name["centroid_lite_f16"] != signature_by_name["centroid_lite_f16_pool_050"]
    assert signature_by_name["centroid_lite_f16"] == capped_signature_by_name["centroid_lite_f16_cap_016"]
    assert signature_by_name["centroid_lite_f16"] == capped_signature_by_name["centroid_lite_f16_cap_032"]
    assert (
        signature_by_name["proxy_normalized_mean_f16"]
        == entry_sample_signature_by_name["proxy_normalized_mean_f16_entry_sample_032"]
    )
    assert (
        signature_by_name["proxy_normalized_mean_f16"]
        == proxy_focus_signature_by_name["proxy_normalized_mean_f16_entry_sample_128"]
    )
    assert (
        signature_by_name["centroid_mean_f16"]
        == entry_sample_signature_by_name["centroid_mean_f16_entry_sample_032"]
    )
    assert (
        proxy_variant_signature_by_name["proxy_max_pool_f16"]
        == proxy_focus_signature_by_name["proxy_max_pool_f16_entry_sample_032"]
    )
    assert (
        signature_by_name["proxy_normalized_mean_f16"]
        != entry_sidecar_signature_by_name["proxy_normalized_mean_f16_entry_sidecar"]
    )
    assert (
        proxy_focus_signature_by_name["proxy_normalized_mean_f16_entry_sidecar_128"]
        != proxy_focus_signature_by_name["proxy_normalized_mean_f16_entry_sidecar_256"]
    )
    assert (
        proxy_focus_signature_by_name["proxy_max_pool_f16_entry_sidecar_128"]
        != proxy_focus_signature_by_name["proxy_max_pool_f16_entry_sidecar_256"]
    )
    assert (
        signature_by_name["centroid_mean_f16"]
        != entry_sidecar_signature_by_name["centroid_mean_f16_entry_sidecar"]
    )
    assert (
        entry_sidecar_signature_by_name["proxy_normalized_mean_f16_entry_sidecar"]
        != entry_sidecar_signature_by_name["centroid_mean_f16_entry_sidecar"]
    )
    assert (
        entry_sidecar_signature_by_name["proxy_normalized_mean_f16_entry_sidecar"]
        != entry_sidecar_proxy_variant_signature_by_name[
            "proxy_max_pool_f16_entry_sidecar"
        ]
    )
    assert (
        proxy_variant_signature_by_name["proxy_max_pool_f16"]
        != entry_sidecar_proxy_variant_signature_by_name[
            "proxy_max_pool_f16_entry_sidecar"
        ]
    )
    proxy_reloptions = dict(signature_by_name["proxy_normalized_mean_f16"])["reloptions"]
    entry_sidecar_reloptions = dict(
        entry_sidecar_signature_by_name["proxy_normalized_mean_f16_entry_sidecar"]
    )["reloptions"]
    assert not any("multivector_doc_storage" in str(item) for item in proxy_reloptions)
    assert any("multivector_proxy_encoder = normalized_mean" in str(item) for item in proxy_reloptions)
    assert any("entry_sidecar = on" in str(item) for item in entry_sidecar_reloptions)
    assert any(
        "entry_sidecar_strategy = hybrid_level_covering" in str(item)
        for item in entry_sidecar_reloptions
    )
    assert dict(signature_by_name["proxy_normalized_mean_f16"])["lexical_key_required"] is False
    assert dict(bm25_signature_by_name["proxy_normalized_mean_f16_bm25_rescue"])["lexical_key_required"] is True
    assert dict(bm25_signature_by_name["proxy_normalized_mean_f16_bm25_rescue"])["lexical_column"] == "body_tsv"
    capped_profile_args = document_node_serving_profile_args(
        signature_args,
        capped_centroid_lite_by_name["centroid_lite_f16_cap_032"],
        ef=100,
        oversampling=1,
    )
    assert capped_profile_args.multivector_centroid_lite_max_postings_per_token == 32
    pruned_profile_args = document_node_serving_profile_args(
        signature_args,
        capped_centroid_lite_by_name["centroid_lite_f16_cap_032_prune_safe_upper_bound"],
        ef=100,
        oversampling=1,
    )
    assert pruned_profile_args.multivector_centroid_lite_max_postings_per_token == 32
    assert pruned_profile_args.multivector_centroid_lite_pruning == "safe_upper_bound"
    entry_sample_signature_profile_args = document_node_serving_profile_args(
        signature_args,
        entry_sample_profiles_by_name["proxy_normalized_mean_f16_entry_sample_032"],
        ef=100,
        oversampling=1,
    )
    assert (
        entry_sample_signature_profile_args.multivector_doc_graph_entry_sample_count
        == 32
    )
    assert dict(learned_sparse_signature_by_name[
        "proxy_normalized_mean_f16_learned_sparse_rescue"
    ])["lexical_key_required"] is True
    assert dict(learned_sparse_signature_by_name[
        "proxy_normalized_mean_f16_learned_sparse_rescue"
    ])["lexical_column"] == "learned_sparse_tsv"
    assert bm25_signature_by_name[
        "proxy_normalized_mean_f16_bm25_rescue"
    ] != learned_sparse_signature_by_name[
        "proxy_normalized_mean_f16_learned_sparse_rescue"
    ]
    validate_serving_profile_index_reuse(
        args=signature_args,
        profile=profiles_by_name["proxy_normalized_mean_f16"],
        index_phase={
            "index_signature_tuple": signature_by_name["proxy_normalized_mean_f16"],
            "index_stats": {
                "multivector_graph_mode": "document_nodes",
                "multivector_doc_build_scorer": "proxy",
                "multivector_proxy_encoder": "normalized_mean",
                "multivector_centroids": "off",
                "multivector_centroid_count": 0,
            },
        },
        expected_signature=signature_by_name["proxy_normalized_mean_f16"],
    )
    validate_serving_profile_index_reuse(
        args=signature_args,
        profile=entry_sidecar_by_name["proxy_normalized_mean_f16_entry_sidecar"],
        index_phase={
            "index_signature_tuple": entry_sidecar_signature_by_name[
                "proxy_normalized_mean_f16_entry_sidecar"
            ],
            "index_stats": {
                "multivector_graph_mode": "document_nodes",
                "multivector_doc_build_scorer": "proxy",
                "multivector_proxy_encoder": "normalized_mean",
                "multivector_centroids": "off",
                "multivector_centroid_count": 0,
                "entry_sidecar_count": 8,
                "entry_sidecar_representatives_configured": 128,
                "entry_sidecar_strategy": "hybrid_level_covering",
            },
        },
        expected_signature=entry_sidecar_signature_by_name[
            "proxy_normalized_mean_f16_entry_sidecar"
        ],
    )
    try:
        validate_serving_profile_index_reuse(
            args=signature_args,
            profile=entry_sidecar_by_name["proxy_normalized_mean_f16_entry_sidecar"],
            index_phase={
                "index_signature_tuple": entry_sidecar_signature_by_name[
                    "proxy_normalized_mean_f16_entry_sidecar"
                ],
                "index_stats": {
                    "multivector_graph_mode": "document_nodes",
                    "multivector_doc_build_scorer": "proxy",
                    "multivector_proxy_encoder": "normalized_mean",
                    "multivector_centroids": "off",
                    "multivector_centroid_count": 0,
                    "entry_sidecar_count": 0,
                    "entry_sidecar_representatives_configured": 128,
                    "entry_sidecar_strategy": "hybrid_level_covering",
                },
            },
            expected_signature=entry_sidecar_signature_by_name[
                "proxy_normalized_mean_f16_entry_sidecar"
            ],
        )
    except RuntimeError as exc:
        assert "entry_sidecar_count > 0" in str(exc)
    else:
        raise AssertionError("entry-sidecar profile should reject missing entry sidecar stats")
    try:
        validate_serving_profile_index_reuse(
            args=signature_args,
            profile=profiles_by_name["proxy_normalized_mean_f16"],
            index_phase={
                "index_signature_tuple": signature_by_name["proxy_normalized_mean_f16"],
                "index_stats": {
                    "multivector_graph_mode": "document_nodes",
                    "multivector_doc_build_scorer": "proxy",
                    "multivector_proxy_encoder": "first_token",
                    "multivector_centroids": "off",
                    "multivector_centroid_count": 0,
                },
            },
            expected_signature=signature_by_name["proxy_normalized_mean_f16"],
        )
    except RuntimeError as exc:
        assert "index reuse stats mismatch" in str(exc)
    else:
        raise AssertionError("serving-grid index reuse should reject mismatched stats")
    try:
        validate_serving_profile_index_reuse(
            args=signature_args,
            profile=profiles_by_name["proxy_normalized_mean_f16"],
            index_phase={
                "index_signature_tuple": signature_by_name["centroid_lite_f16"],
                "index_stats": {},
            },
            expected_signature=signature_by_name["proxy_normalized_mean_f16"],
        )
    except RuntimeError as exc:
        assert "signature mismatch" in str(exc)
    else:
        raise AssertionError("serving-grid index reuse should reject mismatched signatures")


def _self_check_exact_top_cache() -> None:
    calls: list[tuple[str, int]] = []

    def provider(
        _conn: psycopg.Connection[Any],
        query: QueryItem,
        admission_k: int,
    ) -> list[dict[str, Any]]:
        calls.append((query.query_id, admission_k))
        return [{"rank": 1, "doc_id": f"{query.query_id}-{admission_k}"}]

    cache: dict[str, list[dict[str, Any]]] = {}
    query = QueryItem("q1", "test query")
    first, first_elapsed, first_hit = cached_exact_admission_top(
        None,  # type: ignore[arg-type]
        query,
        10,
        cache,
        provider,
    )
    assert first == [{"rank": 1, "doc_id": "q1-10"}]
    assert first_elapsed >= 0.0
    assert first_hit is False
    assert calls == [("q1", 10)]
    second, second_elapsed, second_hit = cached_exact_admission_top(
        None,  # type: ignore[arg-type]
        query,
        10,
        cache,
        provider,
    )
    assert second == first
    assert second_elapsed == 0.0
    assert second_hit is True
    assert calls == [("q1", 10)]
    third, _, third_hit = cached_exact_admission_top(
        None,  # type: ignore[arg-type]
        query,
        5,
        cache,
        provider,
    )
    assert third == [{"rank": 1, "doc_id": "q1-5"}]
    assert third_hit is False
    assert calls == [("q1", 10), ("q1", 5)]
    assert sorted(cache) == ["10:q1", "5:q1"]
    validate_exact_top_cache_document_count(10, 10)
    try:
        validate_exact_top_cache_document_count(10, 11)
    except RuntimeError as exc:
        assert "document count mismatch" in str(exc)
    else:
        raise AssertionError("document count mismatch should reject exact top cache")


def _self_check_learned_sparse_evidence_annotation() -> None:
    coverage = {
        "loaded_documents": 100,
        "learned_sparse_documents": 20,
        "doc_coverage_ratio": learned_sparse_ratio(20, 100),
        "loaded_queries": 10,
        "learned_sparse_queries": 10,
        "query_coverage_ratio": learned_sparse_ratio(10, 10),
    }
    coverage["warnings"] = learned_sparse_coverage_warnings(coverage)
    coverage["partial_coverage"] = bool(coverage["warnings"])
    learned_sparse_row = {
        "profile": "proxy_normalized_mean_f16_learned_sparse_rescue",
        "candidate_source": "proxy_vector",
        "sparse_candidate_source": "learned_sparse",
        "proxy_encoder": "normalized_mean",
        "storage_kind": "f16",
        "ef": 50,
        "oversampling": 1,
        "largest_budget": 800,
        "p95_ms": 1.0,
        "exact_top10_admission_recall": 1.0,
        "ndcg@10": 1.0,
    }
    dense_row = {
        "profile": "proxy_normalized_mean_f16",
        "candidate_source": "proxy_vector",
        "sparse_candidate_source": "off",
        "proxy_encoder": "normalized_mean",
        "storage_kind": "f16",
        "ef": 50,
        "oversampling": 1,
        "largest_budget": 800,
        "p95_ms": 10.0,
        "exact_top10_admission_recall": 0.9,
        "ndcg@10": 0.9,
    }
    grid = {
        "results": [learned_sparse_row, dense_row],
        "summary_rows": [learned_sparse_row, dense_row],
    }
    annotate_serving_grid_learned_sparse_evidence(
        grid,
        {
            "skipped": False,
            "coverage": coverage,
            "warnings": coverage["warnings"],
        },
    )
    assert learned_sparse_row["learned_sparse_partial_coverage"] is True
    assert "learned_sparse_doc_coverage_below_95pct" in learned_sparse_row["evidence_warnings"]
    assert "learned_sparse_partial_coverage" in serving_threshold_failures(
        learned_sparse_row,
        min_top10_admission=0.8,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
        exact_baseline_ndcg=None,
    )[0]
    recommendation = compute_document_node_serving_recommendation(
        grid,
        exact_baseline={"available": False},
        min_top10_admission=0.8,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    assert recommendation["best_latency_safe"]["profile"] == "proxy_normalized_mean_f16"
    assert recommendation["best_balanced"]["profile"] == "proxy_normalized_mean_f16"


def _self_check_learned_sparse_jsonl_parser() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        sparse_path = tmp / "sparse.jsonl"
        sparse_path.write_text(
            "\n".join(
                [
                    json.dumps({"doc_id": "d1", "term_ids": [1, 2], "weights": [0.5, 1.25]}),
                    json.dumps({"id": "d2", "terms": [3], "scores": [2.0]}),
                    "",
                ]
            ),
            encoding="utf-8",
        )
        assert parse_sparse_jsonl(sparse_path, "doc_id") == [
            ("d1", [1, 2], [0.5, 1.25]),
            ("d2", [3], [2.0]),
        ]

        malformed_path = tmp / "malformed.jsonl"
        malformed_path.write_text(
            json.dumps({"query_id": "q1", "term_ids": [1, 2], "weights": [1.0]}),
            encoding="utf-8",
        )
        try:
            parse_sparse_jsonl(malformed_path, "query_id")
        except SystemExit as exc:
            assert "equal-length arrays" in str(exc)
        else:
            raise AssertionError("malformed learned-sparse JSONL should fail")

    try:
        require_complete_learned_sparse_args(
            argparse.Namespace(
                learned_sparse_doc_jsonl=Path("docs.jsonl"),
                learned_sparse_query_jsonl=None,
            )
        )
    except SystemExit as exc:
        assert "must be supplied together" in str(exc)
    else:
        raise AssertionError("incomplete learned-sparse JSONL arguments should fail")


def document_node_serving_profile_args(
    args: argparse.Namespace,
    profile: DocumentNodeServingProfile,
    *,
    ef: int,
    oversampling: int,
    budget_sweep: list[int] | None = None,
) -> argparse.Namespace:
    budget_sweep_text = ",".join(str(item) for item in (budget_sweep or effective_serving_grid_budget_sweep(args)))
    return clone_args(
        args,
        multivector_graph="document_nodes",
        multivector_doc_build_scorer="proxy",
        multivector_candidate_source=profile.candidate_source,
        multivector_branch_plan=profile.branch_plan,
        multivector_bm25_candidate_injection=profile.bm25_candidate_injection,
        multivector_sparse_candidate_source=profile.sparse_candidate_source,
        multivector_proxy_encoder=profile.proxy_encoder,
        multivector_centroids=profile.centroids,
        multivector_centroid_count=profile.centroid_count,
        multivector_doc_storage=profile.storage_kind,
        multivector_doc_storage_cache=profile.cache_mode,
        multivector_token_pooling=profile.token_pooling,
        multivector_token_pooling_target_ratio=profile.token_pooling_target_ratio,
        multivector_centroid_lite_max_postings_per_token=(
            profile.centroid_lite_max_postings_per_token
        ),
        multivector_centroid_lite_pruning=profile.centroid_lite_pruning,
        entry_sidecar=profile.entry_sidecar,
        entry_sidecar_representatives=profile.entry_sidecar_representatives,
        entry_sidecar_strategy=profile.entry_sidecar_strategy,
        multivector_plain_fallback=profile.plain_fallback,
        multivector_candidate_reservoirs=profile.candidate_reservoirs,
        multivector_per_token_doc_reservoir_k=profile.per_token_doc_reservoir_k,
        multivector_coverage_reservoir_k=profile.coverage_reservoir_k,
        multivector_doc_graph_search_ef=ef,
        multivector_doc_graph_oversampling=oversampling,
        multivector_doc_graph_rescore_k=0,
        multivector_doc_graph_entry_sample_count=(
            profile.entry_sample_count
            if profile.entry_sample_count > 0
            else int(getattr(args, "multivector_doc_graph_entry_sample_count", 0) or 0)
        ),
        admission_budget_sweep=budget_sweep_text,
        reuse_index=False,
    )


def serializable_index_signature(signature: tuple[tuple[str, Any], ...]) -> list[list[Any]]:
    return [[key, value] for key, value in signature]


def expected_centroid_count_value(value: Any) -> int:
    if isinstance(value, str) and value.lower() == "auto":
        return 0
    return int(value)


def validate_serving_profile_index_reuse(
    *,
    args: argparse.Namespace,
    profile: DocumentNodeServingProfile,
    index_phase: dict[str, Any],
    expected_signature: tuple[tuple[str, Any], ...],
) -> None:
    observed_signature = index_phase.get("index_signature_tuple")
    if observed_signature != expected_signature:
        raise RuntimeError(
            "serving-grid index reuse signature mismatch for "
            f"{profile.name}: expected {expected_signature!r}, observed {observed_signature!r}"
        )

    index_stats = index_phase.get("index_stats", {})
    if not isinstance(index_stats, dict):
        raise RuntimeError(f"serving-grid index stats are unavailable for {profile.name}")

    expected = {
        "multivector_graph_mode": "document_nodes",
        "multivector_doc_build_scorer": "proxy",
        "multivector_proxy_encoder": profile.proxy_encoder,
        "multivector_centroids": profile.centroids,
    }
    for key, expected_value in expected.items():
        observed_value = index_stats.get(key)
        if observed_value is not None and str(observed_value) != str(expected_value):
            raise RuntimeError(
                "serving-grid index reuse stats mismatch for "
                f"{profile.name}: {key} expected {expected_value!r}, observed {observed_value!r}"
            )

    observed_centroid_count = index_stats.get("multivector_centroid_count")
    if observed_centroid_count is not None:
        expected_centroid_count = expected_centroid_count_value(profile.centroid_count)
        try:
            observed_centroid_count_int = int(observed_centroid_count)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "serving-grid index reuse stats mismatch for "
                f"{profile.name}: multivector_centroid_count is not an integer: "
                f"{observed_centroid_count!r}"
            ) from exc
        if observed_centroid_count_int != expected_centroid_count:
            raise RuntimeError(
                "serving-grid index reuse stats mismatch for "
                f"{profile.name}: multivector_centroid_count expected "
                f"{expected_centroid_count}, observed {observed_centroid_count_int}"
            )

    observed_entry_sidecar_count = index_stats.get("entry_sidecar_count")
    if observed_entry_sidecar_count is not None:
        try:
            observed_entry_sidecar_count_int = int(observed_entry_sidecar_count)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "serving-grid index reuse stats mismatch for "
                f"{profile.name}: entry_sidecar_count is not an integer: "
                f"{observed_entry_sidecar_count!r}"
            ) from exc
        if profile.entry_sidecar and observed_entry_sidecar_count_int <= 0:
            raise RuntimeError(
                "serving-grid index reuse stats mismatch for "
                f"{profile.name}: expected entry_sidecar_count > 0"
            )
        if not profile.entry_sidecar and observed_entry_sidecar_count_int != 0:
            raise RuntimeError(
                "serving-grid index reuse stats mismatch for "
                f"{profile.name}: expected entry_sidecar_count = 0, "
                f"observed {observed_entry_sidecar_count_int}"
            )
    observed_entry_sidecar_reps = index_stats.get("entry_sidecar_representatives_configured")
    if profile.entry_sidecar and observed_entry_sidecar_reps is not None:
        try:
            observed_entry_sidecar_reps_int = int(observed_entry_sidecar_reps)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                "serving-grid index reuse stats mismatch for "
                f"{profile.name}: entry_sidecar_representatives_configured is not an integer: "
                f"{observed_entry_sidecar_reps!r}"
            ) from exc
        if observed_entry_sidecar_reps_int != profile.entry_sidecar_representatives:
            raise RuntimeError(
                "serving-grid index reuse stats mismatch for "
                f"{profile.name}: entry_sidecar_representatives_configured expected "
                f"{profile.entry_sidecar_representatives}, observed {observed_entry_sidecar_reps_int}"
            )
    observed_entry_sidecar_strategy = index_stats.get("entry_sidecar_strategy")
    if (
        profile.entry_sidecar
        and observed_entry_sidecar_strategy is not None
        and str(observed_entry_sidecar_strategy) != profile.entry_sidecar_strategy
    ):
        raise RuntimeError(
            "serving-grid index reuse stats mismatch for "
            f"{profile.name}: entry_sidecar_strategy expected "
            f"{profile.entry_sidecar_strategy!r}, observed {observed_entry_sidecar_strategy!r}"
        )


def serving_grid_scan_stats_by_budget(admission: dict[str, Any]) -> dict[str, dict[str, Any]]:
    per_query = admission.get("per_query", [])
    if not isinstance(per_query, list) or not per_query:
        return {}
    samples: dict[str, dict[str, Any]] = {}
    for query_result in per_query:
        if not isinstance(query_result, dict):
            continue
        budgets = query_result.get("budgets", [])
        if not isinstance(budgets, list):
            continue
        for item in budgets:
            if not isinstance(item, dict):
                continue
            budget = item.get("budget")
            if budget is None:
                continue
            budget_key = str(budget)
            if budget_key in samples:
                continue
            stats = item.get("scan_stats", {})
            if not isinstance(stats, dict):
                continue
            extracted = extract_document_node_serving_stats(stats)
            samples[budget_key] = {
                "scan_stats_sample": extracted,
                "stats_available": document_node_serving_stats_available(extracted),
            }
    return {
        key: samples[key]
        for key in sorted(
            samples,
            key=lambda value: (
                0,
                int(value),
            )
            if value.isdigit()
            else (1, value),
        )
    }


def serving_grid_scan_stats_sample(admission: dict[str, Any], largest_budget: int | None) -> dict[str, Any]:
    if largest_budget is None:
        return {}
    by_budget = serving_grid_scan_stats_by_budget(admission)
    selected = by_budget.get(str(largest_budget), {})
    if not isinstance(selected, dict):
        return {}
    sample = selected.get("scan_stats_sample", {})
    return sample if isinstance(sample, dict) else {}


def centroid_lite_work_warnings(
    *,
    profile: DocumentNodeServingProfile,
    largest_work: dict[str, Any],
    loaded_document_count: int,
    candidate_budget: int | None,
) -> list[str]:
    if profile.candidate_source != "centroid_lite":
        return []
    warnings: list[str] = []
    docs_touched = summary_stat_float(largest_work.get("centroid_docs_touched"), "p95")
    postings_touched = summary_stat_float(
        largest_work.get("centroid_postings_touched"),
        "p95",
    )
    postings_skipped = summary_stat_float(
        largest_work.get("centroid_postings_skipped"),
        "p95",
    )
    candidates = summary_stat_float(largest_work.get("centroid_candidates"), "p95")
    if loaded_document_count > 0 and docs_touched >= max(100.0, loaded_document_count * 0.8):
        warnings.append("centroid_lite_near_exhaustive_docs_touched")
    if postings_touched > 0.0 and (
        "centroid_lite_near_exhaustive_docs_touched" in warnings
        or (profile.centroid_lite_max_postings_per_token <= 0 and postings_skipped <= 0.0)
    ):
        warnings.append("centroid_lite_postings_near_exhaustive")
    if (
        profile.centroid_lite_max_postings_per_token > 0
        and candidate_budget is not None
        and candidate_budget > 0
        and candidates > 0.0
        and candidates < candidate_budget * 0.5
    ):
        warnings.append("centroid_lite_cap_too_aggressive")
    return sorted(dict.fromkeys(warnings))


def document_node_serving_summary_row(
    *,
    profile: DocumentNodeServingProfile,
    args: argparse.Namespace,
    ef: int,
    oversampling: int,
    index_phase: dict[str, Any],
    admission: dict[str, Any],
    qrels: dict[str, dict[str, int]],
) -> dict[str, Any]:
    aggregate = admission.get("aggregate", {})
    if not isinstance(aggregate, dict):
        aggregate = {}
    budget_sweep = aggregate.get("budget_sweep", [])
    executed_budgets = aggregate.get("executed_budgets", aggregate.get("effective_budgets", budget_sweep))
    largest_budget = (
        int(executed_budgets[-1])
        if isinstance(executed_budgets, list) and executed_budgets
        else None
    )
    latency_by_budget = aggregate.get("latency_by_budget", {})
    largest_latency = (
        latency_by_budget.get(str(largest_budget), {})
        if isinstance(latency_by_budget, dict) and largest_budget is not None
        else {}
    )
    if not isinstance(largest_latency, dict):
        largest_latency = {}
    admission_by_budget = aggregate.get("admission_by_budget", {})
    largest_work = (
        admission_by_budget.get(str(largest_budget), {})
        if isinstance(admission_by_budget, dict) and largest_budget is not None
        else {}
    )
    if not isinstance(largest_work, dict):
        largest_work = {}
    exact_rerank_docs_work = largest_work.get("exact_rerank_docs", {})
    if not isinstance(exact_rerank_docs_work, dict):
        exact_rerank_docs_work = {}
    run = aggregate.get("run_at_largest_budget", {})
    if not isinstance(run, dict):
        run = {}
    metrics = method_metrics(run, qrels, args.final_k, args.quality_k) if qrels else {}
    stats_by_budget = serving_grid_scan_stats_by_budget(admission)
    stats_sample = serving_grid_scan_stats_sample(admission, largest_budget)
    index_stats = index_phase.get("index_stats", {})
    index_build_stats = index_phase.get("build_stats", {})
    if not isinstance(index_build_stats, dict):
        index_build_stats = {}
    loaded_document_count = (
        scan_stat_int(index_stats, "node_count")
        if isinstance(index_stats, dict)
        else 0
    )

    def first_sample_value(*keys: str) -> Any:
        for key in keys:
            value = stats_sample.get(key)
            if value is not None:
                return value
        return None

    slow_path_args = clone_args(
        args,
        serving_loaded_document_count=loaded_document_count,
    )
    serving_slow_path_warnings = (
        validate_serving_scan_path(stats_sample, slow_path_args, profile)
        if stats_sample
        else []
    )
    evidence_warnings: list[str] = []
    if profile.candidate_reservoirs != "off":
        reservoirs_enabled_queries = int(
            largest_work.get("reservoirs_enabled_queries") or 0
        )
        sample_reservoirs_enabled = (
            bool(stats_sample.get("multivector_reservoirs_enabled"))
            if isinstance(stats_sample, dict)
            else False
        )
        if reservoirs_enabled_queries <= 0 and not sample_reservoirs_enabled:
            evidence_warnings.append("candidate_reservoirs_not_executed")
    candidate_diagnostics = derive_candidate_underfill_diagnostics({
        "candidate_budget": largest_budget,
        "candidate_source": profile.candidate_source,
        "final_k": args.final_k,
        "multivector_doc_candidate_k": int(
            getattr(args, "multivector_doc_candidate_k", 0) or 0
        ),
        "multivector_doc_graph_search_ef": ef,
        "multivector_doc_graph_oversampling": oversampling,
        "effective_exact_rerank_k": largest_work.get("effective_exact_rerank_k"),
        "proxy_candidates": largest_work.get("proxy_candidates"),
        "centroid_candidates": largest_work.get("centroid_candidates"),
        "quantized_inverted_candidates": largest_work.get("quantized_inverted_candidates"),
        "last_scan_stats_sample": stats_sample,
    })
    centroid_lite_warnings = centroid_lite_work_warnings(
        profile=profile,
        largest_work=largest_work,
        loaded_document_count=loaded_document_count,
        candidate_budget=largest_budget,
    )
    return {
        "profile": profile.name,
        "candidate_source": profile.candidate_source,
        "branch_plan": profile.branch_plan,
        "bm25_candidate_injection": profile.bm25_candidate_injection,
        "bm25_text_query_available": (
            profile.bm25_candidate_injection == "off"
            or query_only_needs_text_query(args)
        ),
        "sparse_candidate_source": profile.sparse_candidate_source,
        "graph_mode": "document_nodes",
        "proxy_encoder": profile.proxy_encoder,
        "centroids": profile.centroids,
        "centroid_count": profile.centroid_count,
        "storage_kind": profile.storage_kind,
        "cache_mode": profile.cache_mode,
        "storage_cache_mode": profile.cache_mode,
        "token_pooling": profile.token_pooling,
        "token_pooling_target_ratio": profile.token_pooling_target_ratio,
        "token_pooling_min_tokens": args.multivector_token_pooling_min_tokens,
        "candidate_reservoirs": profile.candidate_reservoirs,
        "per_token_doc_reservoir_k": profile.per_token_doc_reservoir_k,
        "coverage_reservoir_k": profile.coverage_reservoir_k,
        "entry_sample_count": profile.entry_sample_count,
        "effective_entry_sample_count": int(
            getattr(args, "multivector_doc_graph_entry_sample_count", 0) or 0
        ),
        "entry_sidecar": profile.entry_sidecar,
        "entry_sidecar_representatives": profile.entry_sidecar_representatives,
        "entry_sidecar_strategy": profile.entry_sidecar_strategy,
        "centroid_lite_max_postings_per_token": (
            profile.centroid_lite_max_postings_per_token
        ),
        "centroid_lite_pruning": profile.centroid_lite_pruning,
        "centroid_lite_warnings": centroid_lite_warnings,
        "ef": ef,
        "oversampling": oversampling,
        "run_elapsed_ms": aggregate.get("total_elapsed_ms", 0.0),
        "exact_baseline_elapsed_ms": aggregate.get("exact_baseline_elapsed_ms_total", 0.0),
        "retrieval_elapsed_ms": aggregate.get("retrieval_elapsed_ms_total", 0.0),
        "exact_baseline_query_count": aggregate.get("exact_baseline_query_count", 0),
        "retrieval_query_count": aggregate.get("retrieval_query_count", 0),
        "budget_sweep": budget_sweep if isinstance(budget_sweep, list) else [],
        "executed_budget_count": aggregate.get(
            "executed_budget_count",
            len(executed_budgets) if isinstance(executed_budgets, list) else 0,
        ),
        "executed_budgets": executed_budgets if isinstance(executed_budgets, list) else [],
        "effective_budget_count": aggregate.get(
            "effective_budget_count",
            len(executed_budgets) if isinstance(executed_budgets, list) else 0,
        ),
        "effective_budgets": aggregate.get(
            "effective_budgets",
            executed_budgets if isinstance(executed_budgets, list) else [],
        ),
        "largest_budget": largest_budget,
        "candidate_budget": largest_budget,
        **candidate_diagnostics,
        "serving_exact_rerank_mode": aggregate.get(
            "serving_exact_rerank_mode",
            getattr(args, "serving_exact_rerank_mode", "admission_exhaustive"),
        ),
        "admission_debug_mode": aggregate.get(
            "admission_debug_mode",
            effective_admission_debug_mode(args),
        ),
        "trace_enabled": bool(aggregate.get("trace_enabled", False)),
        "admission_evidence_mode": aggregate.get("admission_evidence_mode", "unavailable"),
        "trace_entries": aggregate.get("trace_entries", 0),
        "requested_serving_exact_rerank_k": aggregate.get(
            "requested_serving_exact_rerank_k",
            int(getattr(args, "serving_exact_rerank_k", 100)),
        ),
        "effective_exact_rerank_k": largest_work.get("effective_exact_rerank_k"),
        "exact_rerank_docs": exact_rerank_docs_work,
        "exact_rerank_docs_p50": exact_rerank_docs_work.get("p50"),
        "exact_rerank_docs_p95": exact_rerank_docs_work.get("p95"),
        "exact_rerank_pairs": largest_work.get("exact_rerank_pairs"),
        "exact_rerank_pairs_p50": summary_stat_float(
            largest_work.get("exact_rerank_pairs"),
            "p50",
        ),
        "multivector_tokens_original": largest_work.get("multivector_tokens_original"),
        "multivector_tokens_pooled": largest_work.get("multivector_tokens_pooled"),
        "multivector_token_pooling_ratio": largest_work.get(
            "multivector_token_pooling_ratio"
        ),
        "tokens_original_p50": summary_stat_float(
            largest_work.get("multivector_tokens_original"),
            "p50",
        ),
        "tokens_pooled_p50": summary_stat_float(
            largest_work.get("multivector_tokens_pooled"),
            "p50",
        ),
        "token_pooling_ratio_p50": summary_stat_float(
            largest_work.get("multivector_token_pooling_ratio"),
            "p50",
        ),
        "p50_ms": largest_latency.get("p50_ms"),
        "p95_ms": largest_latency.get("p95_ms"),
        "p99_ms": largest_latency.get("p99_ms"),
        "p50_latency_ms_at_largest_budget": largest_latency.get("p50_ms"),
        "p95_latency_ms_at_largest_budget": largest_latency.get("p95_ms"),
        "p99_latency_ms_at_largest_budget": largest_latency.get("p99_ms"),
        "exact_top1_admission_rate": aggregate.get("exact_top1_admission_rate", 0.0),
        "exact_top10_admission_recall": aggregate.get("exact_top10_admission_recall", 0.0),
        "recall@10": metrics.get("recall@10") if metrics else None,
        "ndcg@10": metrics.get("ndcg@10") if metrics else None,
        "mrr@10": metrics.get("mrr@10") if metrics else None,
        "metrics": metrics,
        "index_bytes": index_phase.get("index_bytes", 0),
        "index_stats": index_phase.get("index_stats", {}),
        "index_build_stats": index_build_stats,
        "index_signature": index_phase.get("index_signature"),
        "index_build_reused_for_profiles": index_phase.get("index_build_reused_for_profiles", []),
        "largest_budget_work": largest_work,
        "bm25_injection_enabled_queries": largest_work.get("bm25_injection_enabled_queries"),
        "bm25_injection_candidates": largest_work.get("bm25_injection_candidates"),
        "bm25_injection_candidate_limit": largest_work.get("bm25_injection_candidate_limit"),
        "bm25_injection_pool_size": largest_work.get("bm25_injection_pool_size"),
        "bm25_injection_limit_reason": largest_work.get("bm25_injection_limit_reason"),
        "bm25_injection_retained": largest_work.get("bm25_injection_retained"),
        "bm25_injection_exact_reranked": largest_work.get("bm25_injection_exact_reranked"),
        "reservoirs_enabled_queries": largest_work.get("reservoirs_enabled_queries"),
        "reservoir_score_docs": largest_work.get("reservoir_score_docs"),
        "reservoir_coverage_docs": largest_work.get("reservoir_coverage_docs"),
        "reservoir_mean_docs": largest_work.get("reservoir_mean_docs"),
        "reservoir_per_token_docs": largest_work.get("reservoir_per_token_docs"),
        "reservoir_bm25_docs": largest_work.get("reservoir_bm25_docs"),
        "reservoir_union_docs": largest_work.get("reservoir_union_docs"),
        "reservoir_duplicates": largest_work.get("reservoir_duplicates"),
        "learned_sparse_candidates": largest_work.get("learned_sparse_candidates"),
        "learned_sparse_retained_for_maxsim": largest_work.get(
            "learned_sparse_retained_for_maxsim"
        ),
        "learned_sparse_branch_latency_us": largest_work.get(
            "learned_sparse_branch_latency_us"
        ),
        "centroid_lists_visited": largest_work.get("centroid_lists_visited"),
        "centroid_docs_touched": largest_work.get("centroid_docs_touched"),
        "centroid_pruned_docs": largest_work.get("centroid_pruned_docs"),
        "centroid_postings_touched": largest_work.get("centroid_postings_touched"),
        "centroid_postings_skipped": largest_work.get("centroid_postings_skipped"),
        "centroid_posting_limit_per_token": largest_work.get(
            "centroid_posting_limit_per_token"
        ),
        "centroid_posting_cap_strategy": largest_work.get(
            "centroid_posting_cap_strategy"
        ),
        "centroid_candidates": largest_work.get("centroid_candidates"),
        "proxy_candidate_limit_effective": stats_sample.get(
            "proxy_candidate_limit_effective"
        ),
        "proxy_candidate_limit_source": stats_sample.get("proxy_candidate_limit_source"),
        "multivector_proxy_candidate_target": stats_sample.get(
            "multivector_proxy_candidate_target"
        ),
        "multivector_proxy_candidates_returned": stats_sample.get(
            "multivector_proxy_candidates_returned"
        ),
        "proxy_candidates_returned": first_sample_value(
            "proxy_candidates_returned",
            "multivector_proxy_candidates_returned",
        ),
        "graph_entry_sample_scored": first_sample_value(
            "graph_entry_sample_scored",
            "multivector_doc_graph_entry_sample_scored",
        ),
        "graph_entry_sidecar_scored": first_sample_value("graph_entry_sidecar_scored"),
        "graph_entry_sidecar_selected": first_sample_value("graph_entry_sidecar_selected"),
        "graph_traversal_time_us": first_sample_value(
            "document_graph_traversal_time_us",
            "multivector_document_graph_traversal_time_us",
            "graph_traverse_us",
        ),
        "exact_rerank_time_us": first_sample_value(
            "exact_maxsim_rerank_time_us",
            "multivector_exact_maxsim_rerank_time_us",
            "graph_heap_rescore_us",
            "graph_rescore_us",
        ),
        "sidecar_query_bytes_touched": first_sample_value(
            "sidecar_query_bytes_touched",
            "multivector_doc_sidecar_bytes_touched",
        ),
        "sidecar_query_time_us": first_sample_value(
            "sidecar_query_time_us",
            "sidecar_query_load_time_us",
            "sidecar_load_time_us",
            "multivector_sidecar_load_time_us",
        ),
        "multivector_doc_graph_search_ef": stats_sample.get(
            "multivector_doc_graph_search_ef",
            ef,
        ),
        "multivector_doc_graph_oversampling": stats_sample.get(
            "multivector_doc_graph_oversampling",
            oversampling,
        ),
        "serving_stats_by_budget": stats_by_budget,
        "stats_available": merge_document_node_serving_stats_available(stats_by_budget),
        "serving_stats_sample": stats_sample,
        "last_scan_stats_sample": stats_sample,
        "serving_slow_path_warnings": serving_slow_path_warnings,
        "serving_slow_path_failed": bool(
            serving_slow_path_warnings
            and getattr(args, "serving_fail_on_slow_path", False)
        ),
        "evidence_warnings": evidence_warnings,
    }


def serving_row_id(row: dict[str, Any]) -> str:
    return "{profile}_ef{ef}_os{oversampling}".format(
        profile=row.get("profile", ""),
        ef=int(row.get("ef", 0) or 0),
        oversampling=int(row.get("oversampling", 0) or 0),
    )


def serving_row_p95(row: dict[str, Any]) -> float | None:
    value = row.get("p95_latency_ms_at_largest_budget")
    if value is None:
        value = row.get("p95_ms")
    if value is None:
        return None
    try:
        p95 = float(value)
    except (TypeError, ValueError):
        return None
    return p95 if math.isfinite(p95) else None


def serving_row_metric(row: dict[str, Any], key: str) -> float | None:
    value = row.get(key)
    if value is None:
        metrics = row.get("metrics", {})
        if isinstance(metrics, dict):
            value = metrics.get(key)
    if value is None:
        return None
    try:
        metric = float(value)
    except (TypeError, ValueError):
        return None
    return metric if math.isfinite(metric) else None


def serving_candidate_delta_baseline(profile_name: str) -> tuple[str, str] | None:
    marker = "_entry_sample_"
    if marker in profile_name:
        baseline, count_text = profile_name.rsplit(marker, 1)
        if baseline and count_text.isdigit():
            return baseline, "entry_sample"
    marker = "_entry_sidecar_"
    if marker in profile_name:
        baseline, count_text = profile_name.rsplit(marker, 1)
        if baseline and count_text.isdigit():
            return baseline, "entry_sidecar"
    suffix_baselines = (
        ("_reservoir_balanced", "candidate_reservoir"),
        ("_learned_sparse_rescue", "learned_sparse_rescue"),
        ("_bm25_rescue", "bm25_rescue"),
        ("_entry_sidecar", "entry_sidecar"),
    )
    for suffix, comparison in suffix_baselines:
        if profile_name.endswith(suffix):
            baseline = profile_name[: -len(suffix)]
            if baseline:
                return baseline, comparison
    if profile_name.startswith("centroid_lite_f16_cap_"):
        return "centroid_lite_f16", "centroid_lite_posting_cap"
    proxy_variants = {
        "docnodes_normalized_mean_f16": ("proxy_normalized_mean_f16", "candidate_source_variant"),
        "centroid_mean_f16": ("proxy_normalized_mean_f16", "proxy_encoder_variant"),
        "centroid_lite_f16": ("centroid_mean_f16", "candidate_source_variant"),
        "proxy_max_pool_f16": ("proxy_normalized_mean_f16", "proxy_encoder_variant"),
        "proxy_random_projection_fde_f16": (
            "proxy_normalized_mean_f16",
            "proxy_encoder_variant",
        ),
    }
    return proxy_variants.get(profile_name)


def serving_candidate_delta_key(row: dict[str, Any], profile_name: str) -> tuple[Any, ...]:
    return (
        profile_name,
        str(row.get("stage", "")),
        int(row.get("ef", 0) or 0),
        int(row.get("oversampling", 0) or 0),
        int(row.get("candidate_budget", row.get("largest_budget", 0)) or 0),
    )


def serving_candidate_delta_number(row: dict[str, Any], key: str) -> float | None:
    if key == "p95_ms":
        return serving_row_p95(row)
    if key == "p50_ms":
        value = row.get("p50_ms")
        if value is None:
            value = row.get("p50_latency_ms_at_largest_budget")
        if value is not None:
            try:
                metric = float(value)
            except (TypeError, ValueError):
                return None
            return metric if math.isfinite(metric) else None
    return serving_row_metric(row, key)


def serving_candidate_delta_value(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
    key: str,
) -> dict[str, Any]:
    candidate_value = serving_candidate_delta_number(candidate, key)
    baseline_value = serving_candidate_delta_number(baseline, key)
    if candidate_value is None or baseline_value is None:
        return {
            key: candidate_value,
            f"baseline_{key}": baseline_value,
            f"{key}_delta": None,
        }
    return {
        key: candidate_value,
        f"baseline_{key}": baseline_value,
        f"{key}_delta": candidate_value - baseline_value,
    }


def serving_candidate_delta_evidence(row: dict[str, Any]) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    sample = row.get("last_scan_stats_sample", row.get("serving_stats_sample", {}))
    if not isinstance(sample, dict):
        sample = {}

    def usable(value: Any) -> bool:
        return value is not None and value != "" and value != {}

    def add_row(key: str) -> None:
        value = row.get(key)
        if usable(value):
            evidence[key] = value

    def add_sample(out_key: str, *sample_keys: str) -> None:
        if out_key in evidence:
            return
        for sample_key in sample_keys:
            value = sample.get(sample_key)
            if usable(value):
                evidence[out_key] = value
                return

    for key in (
        "effective_entry_sample_count",
        "entry_sample_count",
        "entry_sidecar",
        "entry_sidecar_representatives",
        "entry_sidecar_strategy",
        "candidate_reservoirs",
        "reservoirs_enabled_queries",
        "reservoir_score_docs",
        "reservoir_coverage_docs",
        "reservoir_mean_docs",
        "reservoir_per_token_docs",
        "reservoir_bm25_docs",
        "reservoir_union_docs",
        "reservoir_duplicates",
        "bm25_injection_enabled_queries",
        "bm25_injection_candidates",
        "bm25_injection_candidate_limit",
        "bm25_injection_pool_size",
        "bm25_injection_limit_reason",
        "bm25_injection_retained",
        "bm25_injection_exact_reranked",
        "learned_sparse_candidates",
        "learned_sparse_retained_for_maxsim",
        "learned_sparse_branch_latency_us",
        "learned_sparse_partial_coverage",
        "centroid_lite_max_postings_per_token",
        "centroid_lite_pruning",
        "centroid_postings_touched",
        "centroid_postings_skipped",
        "centroid_posting_limit_per_token",
        "centroid_posting_cap_strategy",
    ):
        add_row(key)

    add_sample(
        "graph_entry_sample_configured",
        "multivector_doc_graph_entry_sample_configured",
        "graph_entry_sample_configured",
    )
    add_sample(
        "graph_entry_sample_effective",
        "multivector_doc_graph_entry_sample_effective",
        "graph_entry_sample_effective",
    )
    add_sample(
        "graph_entry_sample_scored",
        "multivector_doc_graph_entry_sample_scored",
        "graph_entry_sample_scored",
    )
    add_sample("graph_entry_sidecar_count", "graph_entry_sidecar_count")
    add_sample("graph_entry_sidecar_scored", "graph_entry_sidecar_scored")
    add_sample("graph_entry_sidecar_selected", "graph_entry_sidecar_selected")
    add_sample(
        "graph_entry_sidecar_strategy",
        "graph_entry_sidecar_strategy",
        "entry_sidecar_scan_strategy",
    )
    add_sample("multivector_reservoirs_enabled", "multivector_reservoirs_enabled")
    return evidence


def serving_candidate_delta_evidence_text(evidence: dict[str, Any]) -> str:
    if not isinstance(evidence, dict) or not evidence:
        return ""

    def summary_value(value: Any, key: str = "p95") -> str:
        if isinstance(value, dict):
            selected = value.get(key)
            if selected is None:
                selected = value.get("mean")
            if selected is None:
                selected = value.get("max")
            if selected is None:
                selected = value.get("count")
            return str(selected) if selected is not None else ""
        return str(value)

    parts: list[str] = []
    entry_sample = evidence.get("effective_entry_sample_count", evidence.get("entry_sample_count"))
    if entry_sample not in (None, "", 0):
        entry_part = f"entry_sample={entry_sample}"
        scored = evidence.get("graph_entry_sample_scored")
        if scored not in (None, ""):
            entry_part += f"/scored={scored}"
        parts.append(entry_part)
    if bool(evidence.get("entry_sidecar", False)) or evidence.get("graph_entry_sidecar_count") not in (None, ""):
        sidecar_part = "entry_sidecar"
        scored = evidence.get("graph_entry_sidecar_scored")
        selected = evidence.get("graph_entry_sidecar_selected")
        if scored not in (None, ""):
            sidecar_part += f"/scored={scored}"
        if selected not in (None, ""):
            sidecar_part += f"/selected={selected}"
        parts.append(sidecar_part)
    if evidence.get("candidate_reservoirs") not in (None, "", "off"):
        reservoir_part = f"reservoirs={evidence.get('candidate_reservoirs')}"
        enabled = evidence.get("reservoirs_enabled_queries")
        if enabled not in (None, ""):
            reservoir_part += f"/enabled={enabled}"
        union = summary_value(evidence.get("reservoir_union_docs"))
        if union:
            reservoir_part += f"/union_p95={union}"
        duplicates = summary_value(evidence.get("reservoir_duplicates"))
        if duplicates:
            reservoir_part += f"/dups_p95={duplicates}"
        parts.append(reservoir_part)
    bm25_enabled = evidence.get("bm25_injection_enabled_queries")
    if bm25_enabled not in (None, "", 0):
        bm25_part = f"bm25_queries={bm25_enabled}"
        retained = summary_value(evidence.get("bm25_injection_retained"))
        if retained:
            bm25_part += f"/retained_p95={retained}"
        parts.append(bm25_part)
    learned_sparse = summary_value(evidence.get("learned_sparse_candidates"))
    if learned_sparse:
        parts.append(f"learned_sparse_candidates_p95={learned_sparse}")
    cap = evidence.get(
        "centroid_posting_limit_per_token",
        evidence.get("centroid_lite_max_postings_per_token"),
    )
    if cap not in (None, "", 0):
        cap_part = f"centroid_cap={cap}"
        touched = summary_value(evidence.get("centroid_postings_touched"))
        skipped = summary_value(evidence.get("centroid_postings_skipped"))
        if touched:
            cap_part += f"/touched_p95={touched}"
        if skipped:
            cap_part += f"/skipped_p95={skipped}"
        parts.append(cap_part)
    return "; ".join(parts)


def compute_document_node_candidate_source_deltas(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    valid_rows = [row for row in rows if isinstance(row, dict)]
    rows_by_key = {
        serving_candidate_delta_key(row, str(row.get("profile", ""))): row
        for row in valid_rows
    }
    deltas: list[dict[str, Any]] = []
    missing_baselines: list[dict[str, Any]] = []
    for row in valid_rows:
        profile_name = str(row.get("profile", ""))
        baseline_info = serving_candidate_delta_baseline(profile_name)
        if baseline_info is None:
            continue
        baseline_profile, comparison = baseline_info
        baseline_key = serving_candidate_delta_key(row, baseline_profile)
        baseline = rows_by_key.get(baseline_key)
        if baseline is None:
            missing_baselines.append({
                "profile": profile_name,
                "baseline_profile": baseline_profile,
                "comparison": comparison,
                "stage": row.get("stage"),
                "ef": row.get("ef"),
                "oversampling": row.get("oversampling"),
                "candidate_budget": row.get("candidate_budget", row.get("largest_budget")),
            })
            continue
        delta_row: dict[str, Any] = {
            "comparison": comparison,
            "profile": profile_name,
            "baseline_profile": baseline_profile,
            "stage": row.get("stage"),
            "candidate_source": row.get("candidate_source"),
            "baseline_candidate_source": baseline.get("candidate_source"),
            "proxy_encoder": row.get("proxy_encoder"),
            "baseline_proxy_encoder": baseline.get("proxy_encoder"),
            "centroids": row.get("centroids"),
            "baseline_centroids": baseline.get("centroids"),
            "entry_sample_count": row.get("entry_sample_count"),
            "baseline_entry_sample_count": baseline.get("entry_sample_count"),
            "entry_sidecar": bool(row.get("entry_sidecar", False)),
            "baseline_entry_sidecar": bool(baseline.get("entry_sidecar", False)),
            "candidate_reservoirs": row.get("candidate_reservoirs"),
            "baseline_candidate_reservoirs": baseline.get("candidate_reservoirs"),
            "candidate_evidence": serving_candidate_delta_evidence(row),
            "baseline_evidence": serving_candidate_delta_evidence(baseline),
            "storage_kind": row.get("storage_kind"),
            "storage_cache_mode": row.get("storage_cache_mode", row.get("cache_mode")),
            "ef": int(row.get("ef", 0) or 0),
            "oversampling": int(row.get("oversampling", 0) or 0),
            "candidate_budget": int(
                row.get("candidate_budget", row.get("largest_budget", 0)) or 0
            ),
        }
        evidence_warnings: list[str] = []
        if serving_row_unsupported_reservoir_evidence(row):
            evidence_warnings.append("candidate_reservoirs_not_executed")
        if serving_row_unsupported_reservoir_evidence(baseline):
            evidence_warnings.append("baseline_candidate_reservoirs_not_executed")
        delta_row["evidence_usable"] = not evidence_warnings
        delta_row["evidence_warnings"] = evidence_warnings
        delta_row["unsupported_reservoir_evidence"] = (
            serving_row_unsupported_reservoir_evidence(row)
        )
        for metric_key in (
            "exact_top1_admission_rate",
            "exact_top10_admission_recall",
            "recall@10",
            "ndcg@10",
            "mrr@10",
            "p50_ms",
            "p95_ms",
        ):
            delta_row.update(serving_candidate_delta_value(row, baseline, metric_key))
        top10_delta = delta_row.get("exact_top10_admission_recall_delta")
        p95_delta = delta_row.get("p95_ms_delta")
        delta_row["admission_improved"] = (
            top10_delta is not None and float(top10_delta) > 0.0
        )
        delta_row["latency_improved"] = p95_delta is not None and float(p95_delta) < 0.0
        if comparison == "centroid_lite_posting_cap":
            for warning_name in row.get("centroid_lite_warnings", []) or []:
                evidence_warnings.append(str(warning_name))
            if top10_delta is not None and float(top10_delta) <= 0.0:
                evidence_warnings.append("centroid_lite_no_admission_gain")
            if top10_delta is not None and float(top10_delta) < -0.02:
                evidence_warnings.append("centroid_lite_cap_too_aggressive")
            if p95_delta is not None and float(p95_delta) > 0.0:
                evidence_warnings.append("centroid_lite_latency_regression")
            delta_row["evidence_warnings"] = sorted(dict.fromkeys(evidence_warnings))
        deltas.append(delta_row)
    deltas.sort(
        key=lambda item: (
            str(item.get("comparison", "")),
            str(item.get("profile", "")),
            int(item.get("ef", 0) or 0),
            int(item.get("oversampling", 0) or 0),
            int(item.get("candidate_budget", 0) or 0),
        )
    )
    missing_baselines.sort(
        key=lambda item: (
            str(item.get("comparison", "")),
            str(item.get("profile", "")),
            int(item.get("ef", 0) or 0),
            int(item.get("oversampling", 0) or 0),
            int(item.get("candidate_budget", 0) or 0),
        )
    )
    return {
        "enabled": True,
        "rows": deltas,
        "missing_baselines": missing_baselines,
        "comparison_count": len(deltas),
        "missing_baseline_count": len(missing_baselines),
    }


def document_node_candidate_source_delta_summary(
    candidate_source_deltas: dict[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(candidate_source_deltas, dict):
        return {
            "available": False,
            "reason": "candidate_source_deltas missing",
        }
    rows = [
        item
        for item in candidate_source_deltas.get("rows", [])
        if isinstance(item, dict)
    ]
    if not rows:
        return {
            "available": False,
            "reason": "no paired candidate-source delta rows",
            "missing_baseline_count": int(
                candidate_source_deltas.get("missing_baseline_count", 0) or 0
            ),
        }

    def metric(item: dict[str, Any], key: str) -> float | None:
        value = item.get(key)
        if value is None or isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None

    def compact(item: dict[str, Any] | None, *, basis: str) -> dict[str, Any] | None:
        if item is None:
            return None
        return {
            "basis": basis,
            "comparison": item.get("comparison"),
            "profile": item.get("profile"),
            "baseline_profile": item.get("baseline_profile"),
            "ef": item.get("ef"),
            "oversampling": item.get("oversampling"),
            "candidate_budget": item.get("candidate_budget"),
            "exact_top10_admission_recall_delta": item.get(
                "exact_top10_admission_recall_delta"
            ),
            "ndcg@10_delta": item.get("ndcg@10_delta"),
            "p95_ms_delta": item.get("p95_ms_delta"),
            "exact_top10_admission_recall": item.get("exact_top10_admission_recall"),
            "baseline_exact_top10_admission_recall": item.get(
                "baseline_exact_top10_admission_recall"
            ),
            "ndcg@10": item.get("ndcg@10"),
            "baseline_ndcg@10": item.get("baseline_ndcg@10"),
            "p95_ms": item.get("p95_ms"),
            "baseline_p95_ms": item.get("baseline_p95_ms"),
            "evidence_usable": item.get("evidence_usable", True),
            "evidence_warnings": item.get("evidence_warnings", []),
            "candidate_evidence_summary": serving_candidate_delta_evidence_text(
                item.get("candidate_evidence", {})
            ),
        }

    usable_rows = [
        item for item in rows
        if bool(item.get("evidence_usable", True))
    ]

    positive_admission = [
        item
        for item in usable_rows
        if (metric(item, "exact_top10_admission_recall_delta") or 0.0) > 0.0
    ]
    positive_quality = [
        item
        for item in usable_rows
        if (metric(item, "ndcg@10_delta") or 0.0) > 0.0
    ]
    latency_improved = [
        item
        for item in usable_rows
        if (metric(item, "p95_ms_delta") or 0.0) < 0.0
    ]

    best_admission = max(
        positive_admission,
        key=lambda item: (
            metric(item, "exact_top10_admission_recall_delta") or 0.0,
            metric(item, "ndcg@10_delta") or 0.0,
            -(metric(item, "p95_ms_delta") or 0.0),
            str(item.get("profile", "")),
        ),
        default=None,
    )
    best_quality = max(
        positive_quality,
        key=lambda item: (
            metric(item, "ndcg@10_delta") or 0.0,
            metric(item, "exact_top10_admission_recall_delta") or 0.0,
            -(metric(item, "p95_ms_delta") or 0.0),
            str(item.get("profile", "")),
        ),
        default=None,
    )
    best_latency = min(
        latency_improved,
        key=lambda item: (
            metric(item, "p95_ms_delta") or 0.0,
            -(metric(item, "exact_top10_admission_recall_delta") or 0.0),
            str(item.get("profile", "")),
        ),
        default=None,
    )

    by_comparison: dict[str, dict[str, Any]] = {}
    for comparison in sorted({str(item.get("comparison", "")) for item in rows}):
        comparison_rows = [
            item for item in positive_admission
            if str(item.get("comparison", "")) == comparison
        ]
        by_comparison[comparison] = compact(
            max(
                comparison_rows,
                key=lambda item: (
                    metric(item, "exact_top10_admission_recall_delta") or 0.0,
                    metric(item, "ndcg@10_delta") or 0.0,
                    -(metric(item, "p95_ms_delta") or 0.0),
                    str(item.get("profile", "")),
                ),
                default=None,
            ),
            basis="top10_admission_delta",
        )

    return {
        "available": True,
        "comparison_count": len(rows),
        "usable_comparison_count": len(usable_rows),
        "unusable_comparison_count": len(rows) - len(usable_rows),
        "positive_admission_delta_count": len(positive_admission),
        "positive_quality_delta_count": len(positive_quality),
        "latency_improvement_count": len(latency_improved),
        "missing_baseline_count": int(
            candidate_source_deltas.get("missing_baseline_count", 0) or 0
        ),
        "best_admission_delta": compact(
            best_admission,
            basis="top10_admission_delta",
        ),
        "best_quality_delta": compact(best_quality, basis="ndcg@10_delta"),
        "best_latency_delta": compact(best_latency, basis="p95_ms_delta"),
        "best_admission_delta_by_comparison": by_comparison,
    }


def serving_storage_penalty(row: dict[str, Any]) -> int:
    return {
        "sq8": 0,
        "f16": 1,
        "f32": 2,
    }.get(str(row.get("storage_kind", "")), 3)


def serving_row_experimental(row: dict[str, Any]) -> bool:
    profile = str(row.get("profile", ""))
    return (
        str(row.get("candidate_source", "")) == "quantized_inverted_experimental"
        or profile == "quantized_inverted_experimental"
        or profile.startswith("quantized_inverted_experimental_")
    )


def serving_row_reservoirs_requested(row: dict[str, Any]) -> bool:
    return str(row.get("candidate_reservoirs") or "off") != "off"


def serving_row_reservoirs_executed(row: dict[str, Any]) -> bool:
    enabled_queries = row.get("reservoirs_enabled_queries")
    try:
        if enabled_queries is not None and int(enabled_queries) > 0:
            return True
    except (TypeError, ValueError):
        pass

    sample = row.get("last_scan_stats_sample", row.get("serving_stats_sample", {}))
    if isinstance(sample, dict) and bool(sample.get("multivector_reservoirs_enabled")):
        return True
    return False


def serving_row_unsupported_reservoir_evidence(row: dict[str, Any]) -> bool:
    return serving_row_reservoirs_requested(row) and not serving_row_reservoirs_executed(row)


def serving_row_has_usable_metrics(row: dict[str, Any]) -> bool:
    return (
        serving_row_p95(row) is not None
        or serving_row_metric(row, "exact_top10_admission_recall") is not None
        or serving_row_metric(row, "ndcg@10") is not None
    )


def serving_recommendation_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": serving_row_id(row),
        "profile": row.get("profile"),
        "candidate_source": row.get("candidate_source"),
        "experimental": serving_row_experimental(row),
        "branch_plan": row.get("branch_plan"),
        "bm25_candidate_injection": row.get("bm25_candidate_injection"),
        "bm25_text_query_available": row.get("bm25_text_query_available"),
        "sparse_candidate_source": row.get("sparse_candidate_source"),
        "proxy_encoder": row.get("proxy_encoder"),
        "centroids": row.get("centroids"),
        "storage_kind": row.get("storage_kind"),
        "cache_mode": row.get("cache_mode"),
        "token_pooling": row.get("token_pooling"),
        "token_pooling_target_ratio": row.get("token_pooling_target_ratio"),
        "multivector_tokens_original": row.get("multivector_tokens_original"),
        "multivector_tokens_pooled": row.get("multivector_tokens_pooled"),
        "multivector_token_pooling_ratio": row.get("multivector_token_pooling_ratio"),
        "tokens_original_p50": row.get("tokens_original_p50"),
        "tokens_pooled_p50": row.get("tokens_pooled_p50"),
        "token_pooling_ratio_p50": row.get("token_pooling_ratio_p50"),
        "candidate_reservoirs": row.get("candidate_reservoirs"),
        "per_token_doc_reservoir_k": row.get("per_token_doc_reservoir_k"),
        "coverage_reservoir_k": row.get("coverage_reservoir_k"),
        "reservoirs_executed": serving_row_reservoirs_executed(row),
        "unsupported_reservoir_evidence": serving_row_unsupported_reservoir_evidence(row),
        "entry_sample_count": row.get("entry_sample_count"),
        "effective_entry_sample_count": row.get("effective_entry_sample_count"),
        "entry_sidecar": row.get("entry_sidecar"),
        "entry_sidecar_representatives": row.get("entry_sidecar_representatives"),
        "entry_sidecar_strategy": row.get("entry_sidecar_strategy"),
        "centroid_lite_max_postings_per_token": row.get(
            "centroid_lite_max_postings_per_token"
        ),
        "centroid_lite_pruning": row.get("centroid_lite_pruning", "off"),
        "centroid_lite_warnings": row.get("centroid_lite_warnings", []),
        "ef": row.get("ef"),
        "oversampling": row.get("oversampling"),
        "largest_budget": row.get("largest_budget"),
        "candidate_budget": row.get("candidate_budget", row.get("largest_budget")),
        "requested_candidate_k": row.get("requested_candidate_k"),
        "effective_candidate_k": row.get("effective_candidate_k"),
        "candidate_underfill": row.get("candidate_underfill"),
        "candidate_underfill_ratio": row.get("candidate_underfill_ratio"),
        "candidate_underfill_reason": row.get("candidate_underfill_reason"),
        "next_admission_hint": row.get("next_admission_hint"),
        "proxy_candidate_limit_effective": row.get("proxy_candidate_limit_effective"),
        "proxy_candidate_limit_source": row.get("proxy_candidate_limit_source"),
        "multivector_proxy_candidate_target": row.get("multivector_proxy_candidate_target"),
        "multivector_proxy_candidates_returned": row.get(
            "multivector_proxy_candidates_returned"
        ),
        "serving_exact_rerank_mode": row.get("serving_exact_rerank_mode"),
        "admission_debug_mode": row.get("admission_debug_mode"),
        "trace_enabled": row.get("trace_enabled"),
        "admission_evidence_mode": row.get("admission_evidence_mode"),
        "trace_entries": row.get("trace_entries"),
        "requested_serving_exact_rerank_k": row.get("requested_serving_exact_rerank_k"),
        "effective_exact_rerank_k": row.get("effective_exact_rerank_k"),
        "exact_rerank_docs_p50": row.get("exact_rerank_docs_p50"),
        "exact_rerank_docs_p95": row.get("exact_rerank_docs_p95"),
        "exact_rerank_pairs": row.get("exact_rerank_pairs"),
        "exact_rerank_pairs_p50": row.get("exact_rerank_pairs_p50"),
        "p50_ms": row.get("p50_ms"),
        "p95_ms": row.get("p95_ms"),
        "p50_latency_ms_at_largest_budget": row.get("p50_latency_ms_at_largest_budget"),
        "p95_latency_ms_at_largest_budget": row.get("p95_latency_ms_at_largest_budget"),
        "exact_top10_admission_recall": row.get("exact_top10_admission_recall"),
        "exact_top10_admission_recall_delta": row.get(
            "exact_top10_admission_recall_delta"
        ),
        "bm25_admission_delta": row.get("bm25_admission_delta"),
        "exact_top1_admission_rate": row.get("exact_top1_admission_rate"),
        "recall@10": row.get("recall@10"),
        "ndcg@10": row.get("ndcg@10"),
        "mrr@10": row.get("mrr@10"),
        "p95_ms_delta": row.get("p95_ms_delta"),
        "bm25_latency_delta_ms": row.get("bm25_latency_delta_ms"),
        "index_bytes": row.get("index_bytes"),
        "profile_elapsed_ms": row.get("profile_elapsed_ms"),
        "index_build_elapsed_ms": row.get("index_build_elapsed_ms"),
        "index_signature": row.get("index_signature"),
        "index_build_reused": row.get("index_build_reused"),
        "index_build_reused_for_profiles": row.get("index_build_reused_for_profiles", []),
        "bm25_injection_enabled_queries": row.get("bm25_injection_enabled_queries"),
        "bm25_injection_candidates": row.get("bm25_injection_candidates"),
        "bm25_injection_candidate_limit": row.get("bm25_injection_candidate_limit"),
        "bm25_injection_pool_size": row.get("bm25_injection_pool_size"),
        "bm25_injection_limit_reason": row.get("bm25_injection_limit_reason"),
        "bm25_injection_retained": row.get("bm25_injection_retained"),
        "bm25_injection_exact_reranked": row.get("bm25_injection_exact_reranked"),
        "reservoirs_enabled_queries": row.get("reservoirs_enabled_queries"),
        "reservoir_score_docs": row.get(
            "reservoir_score_docs", row.get("multivector_reservoir_score_docs")
        ),
        "reservoir_coverage_docs": row.get(
            "reservoir_coverage_docs", row.get("multivector_reservoir_coverage_docs")
        ),
        "reservoir_mean_docs": row.get(
            "reservoir_mean_docs", row.get("multivector_reservoir_mean_docs")
        ),
        "reservoir_per_token_docs": row.get(
            "reservoir_per_token_docs", row.get("multivector_reservoir_per_token_docs")
        ),
        "reservoir_bm25_docs": row.get(
            "reservoir_bm25_docs", row.get("multivector_reservoir_bm25_docs")
        ),
        "reservoir_union_docs": row.get(
            "reservoir_union_docs", row.get("multivector_reservoir_union_docs")
        ),
        "reservoir_duplicates": row.get(
            "reservoir_duplicates", row.get("multivector_reservoir_duplicates")
        ),
        "learned_sparse_candidates": row.get("learned_sparse_candidates"),
        "learned_sparse_retained_for_maxsim": row.get("learned_sparse_retained_for_maxsim"),
        "learned_sparse_branch_latency_us": row.get("learned_sparse_branch_latency_us"),
        "learned_sparse_partial_coverage": row.get("learned_sparse_partial_coverage"),
        "centroid_lists_visited": row.get("centroid_lists_visited"),
        "centroid_docs_touched": row.get("centroid_docs_touched"),
        "centroid_pruned_docs": row.get("centroid_pruned_docs"),
        "centroid_postings_touched": row.get("centroid_postings_touched"),
        "centroid_postings_skipped": row.get("centroid_postings_skipped"),
        "centroid_posting_limit_per_token": row.get("centroid_posting_limit_per_token"),
        "centroid_posting_cap_strategy": row.get("centroid_posting_cap_strategy"),
        "centroid_candidates": row.get("centroid_candidates"),
        "evidence_warnings": row.get("evidence_warnings", []),
        "stats_available": row.get("stats_available", {}),
        "last_scan_stats_sample": row.get("last_scan_stats_sample", {}),
    }


def serving_row_candidate_budget(row: dict[str, Any]) -> int | None:
    for key in ("candidate_budget", "largest_budget"):
        value = row.get(key)
        if value is None:
            continue
        try:
            budget = int(value)
        except (TypeError, ValueError):
            continue
        if budget > 0:
            return budget
    return None


def int_from_mapping(mapping: dict[str, Any], *keys: str) -> int:
    for key in keys:
        value = mapping.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        return parsed
    return 0


def summary_p95_int(value: Any) -> int:
    if isinstance(value, dict):
        return int_from_mapping(value, "p95", "max", "mean")
    if value is None or isinstance(value, bool):
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def serving_candidate_effective_count(row: dict[str, Any], sample: dict[str, Any]) -> int:
    direct = int_from_mapping(
        row,
        "effective_candidate_k",
        "proxy_candidates_returned",
        "multivector_proxy_candidates_returned",
        "proxy_candidates",
        "multivector_doc_graph_candidates",
        "centroid_candidates",
        "quantized_inverted_candidates",
    )
    if direct > 0:
        return direct
    for key in (
        "proxy_candidates",
        "multivector_proxy_candidates_returned",
        "proxy_candidates_returned",
        "multivector_doc_graph_candidates",
        "centroid_candidates",
        "quantized_inverted_candidates",
    ):
        summarized = summary_p95_int(row.get(key))
        if summarized > 0:
            return summarized
    return int_from_mapping(
        sample,
        "proxy_candidates_returned",
        "multivector_proxy_candidates_returned",
        "proxy_candidates",
        "multivector_doc_graph_candidates",
        "centroid_candidates",
        "quantized_inverted_candidates",
    )


def serving_candidate_underfill_reason(
    row: dict[str, Any],
    *,
    requested_candidate_k: int,
    effective_candidate_k: int,
    sample: dict[str, Any],
) -> str:
    if requested_candidate_k <= 0 or effective_candidate_k <= 0:
        return "unknown"
    if effective_candidate_k >= requested_candidate_k:
        exact_k = int_from_mapping(row, "effective_exact_rerank_k")
        if exact_k > 0 and exact_k < effective_candidate_k:
            return "exact_rerank_k_limits_rescue"
        return "none"

    limit_source = str(
        row.get("proxy_candidate_limit_source")
        or sample.get("proxy_candidate_limit_source")
        or ""
    )
    if limit_source == "search_ef":
        return "capped_by_search_ef"
    if limit_source == "doc_candidate_k":
        return "capped_by_doc_candidate_k"

    search_ef = int_from_mapping(
        row,
        "multivector_doc_graph_search_ef",
        "doc_graph_search_ef",
        "ef",
    ) or int_from_mapping(sample, "multivector_doc_graph_search_ef")
    if search_ef > 0 and search_ef <= effective_candidate_k:
        return "capped_by_search_ef"

    target = int_from_mapping(row, "multivector_proxy_candidate_target") or int_from_mapping(
        sample,
        "multivector_proxy_candidate_target",
    )
    final_k = int_from_mapping(row, "final_k", "final_k_effective") or int_from_mapping(
        sample,
        "final_k_effective",
        "effective_result_target",
        "graph_effective_result_target",
    )
    oversampling = int_from_mapping(
        row,
        "multivector_doc_graph_oversampling",
        "doc_graph_oversampling",
        "oversampling",
    ) or int_from_mapping(sample, "multivector_doc_graph_oversampling")
    if (
        target > 0
        and final_k > 0
        and oversampling > 0
        and target <= max(final_k * oversampling, final_k)
        and target <= effective_candidate_k
    ):
        return "capped_by_final_k_oversampling"

    doc_candidate_k = int_from_mapping(row, "multivector_doc_candidate_k") or int_from_mapping(
        sample,
        "multivector_doc_candidate_k",
    )
    if doc_candidate_k > 0 and doc_candidate_k <= effective_candidate_k:
        return "capped_by_doc_candidate_k"

    return "candidate_source_underfilled"


def next_admission_hint_for_underfill_reason(reason: str) -> str:
    return {
        "capped_by_search_ef": "try_higher_ef_or_entry_sample",
        "capped_by_final_k_oversampling": "increase_oversampling",
        "capped_by_doc_candidate_k": "increase_multivector_doc_candidate_k",
        "candidate_source_underfilled": "try_entry_sidecar_or_stronger_proxy",
        "exact_rerank_k_limits_rescue": "increase_serving_exact_rerank_k_or_candidate_reservoir",
        "unknown": "inspect_candidate_limit_stats",
        "none": "",
    }.get(reason, "inspect_candidate_limit_stats")


def derive_candidate_underfill_diagnostics(row: dict[str, Any]) -> dict[str, Any]:
    sample = row.get("last_scan_stats_sample", row.get("serving_stats_sample", {}))
    if not isinstance(sample, dict):
        sample = {}
    requested = serving_row_candidate_budget(row) or int_from_mapping(
        sample,
        "multivector_proxy_candidate_target",
    )
    effective = serving_candidate_effective_count(row, sample)
    reason = serving_candidate_underfill_reason(
        row,
        requested_candidate_k=requested or 0,
        effective_candidate_k=effective,
        sample=sample,
    )
    ratio = (
        round(min(max(effective / requested, 0.0), 1.0), 6)
        if requested and effective >= 0
        else None
    )
    underfill = bool(requested and effective > 0 and effective < requested)
    hint = next_admission_hint_for_underfill_reason(reason)
    return {
        "requested_candidate_k": requested,
        "effective_candidate_k": effective,
        "candidate_underfill": underfill,
        "candidate_underfill_ratio": ratio,
        "candidate_underfill_reason": reason,
        "next_admission_hint": hint,
    }


def serving_row_exhausted_admitted_band(row: dict[str, Any]) -> bool:
    budget = serving_row_candidate_budget(row)
    if budget is None:
        return False
    mode = str(row.get("serving_exact_rerank_mode") or "")
    effective_rerank = row.get("effective_exact_rerank_k")
    try:
        effective_rerank_int = int(effective_rerank)
    except (TypeError, ValueError):
        effective_rerank_int = 0
    rerank_docs_p95 = row.get("exact_rerank_docs_p95")
    try:
        rerank_docs_p95_float = float(rerank_docs_p95)
    except (TypeError, ValueError):
        rerank_docs_p95_float = 0.0
    if mode == "admission_exhaustive":
        return True
    return effective_rerank_int >= budget and rerank_docs_p95_float >= budget * 0.95


def serving_row_candidate_admission_bottleneck(
    row: dict[str, Any],
    *,
    min_top10_admission: float,
) -> bool:
    admission = serving_row_metric(row, "exact_top10_admission_recall")
    if admission is None or admission >= min_top10_admission:
        return False
    return serving_row_exhausted_admitted_band(row)


def summary_stat_float(value: Any, key: str, default: float = 0.0) -> float:
    if isinstance(value, dict):
        value = value.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def summary_reason_present(value: Any, reason: str) -> bool:
    if isinstance(value, dict):
        return summary_stat_float(value.get(reason), "count", 0.0) > 0.0
    return str(value or "") == reason


def summary_stat_float_any(
    row: dict[str, Any],
    keys: tuple[str, ...],
    stat: str,
    default: float = 0.0,
) -> float:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        result = summary_stat_float(value, stat, default)
        if result != default:
            return result
    return default


def serving_admission_improvement_hints(
    row: dict[str, Any],
    *,
    min_top10_admission: float,
) -> list[str]:
    if serving_row_unsupported_reservoir_evidence(row):
        return ["candidate_reservoirs_not_executed"]

    admission = serving_row_metric(row, "exact_top10_admission_recall")
    if admission is not None and admission >= min_top10_admission:
        return []

    hints: list[str] = []
    sample = row.get("last_scan_stats_sample", {})
    if not isinstance(sample, dict):
        sample = {}

    candidate_hint = str(row.get("next_admission_hint") or "")
    if candidate_hint:
        hints.append(candidate_hint)
    candidate_underfill_reason = str(row.get("candidate_underfill_reason") or "")
    if candidate_underfill_reason and candidate_underfill_reason not in {"none", "unknown"}:
        hints.append(candidate_underfill_reason)

    if serving_row_candidate_admission_bottleneck(
        row,
        min_top10_admission=min_top10_admission,
    ):
        hints.append("candidate_source_quality_bottleneck")
        hints.append("admitted_band_exhausted")
        proxy_encoder = str(row.get("proxy_encoder") or sample.get("proxy_encoder_kind") or "")
        candidate_source = str(row.get("candidate_source") or sample.get("multivector_candidate_source") or "")
        sparse_source_for_bottleneck = str(row.get("sparse_candidate_source") or "off")
        bm25_for_bottleneck = str(row.get("bm25_candidate_injection") or "off")
        reservoirs_for_bottleneck = str(row.get("candidate_reservoirs") or "off")
        if candidate_source == "proxy_vector":
            entry_sample_count = int(
                row.get(
                    "effective_entry_sample_count",
                    row.get("entry_sample_count", 0),
                )
                or 0
            )
            if entry_sample_count <= 0:
                hints.append("try_entry_sample_sweep")
            if proxy_encoder == "normalized_mean":
                hints.append("try_max_pool_or_centroid_mean_proxy")
            elif proxy_encoder == "max_pool":
                hints.append("try_centroid_mean_proxy")
            elif proxy_encoder == "centroid_mean":
                hints.append("try_sparse_rescue_or_centroid_lite")
        if reservoirs_for_bottleneck == "off":
            hints.append("try_balanced_candidate_reservoirs")
        if sparse_source_for_bottleneck == "off" and bm25_for_bottleneck == "off":
            hints.append("try_bm25_or_learned_sparse_rescue")

    if row.get("candidate_reservoirs") not in (None, "off"):
        if serving_row_unsupported_reservoir_evidence(row):
            hints.append("candidate_reservoirs_not_executed")
        reservoir_union_p95 = summary_stat_float_any(
            row,
            ("multivector_reservoir_union_docs", "reservoir_union_docs"),
            "p95",
        )
        reservoir_score_p95 = summary_stat_float_any(
            row,
            ("multivector_reservoir_score_docs", "reservoir_score_docs"),
            "p95",
        )
        reservoir_duplicates_p95 = summary_stat_float_any(
            row,
            ("multivector_reservoir_duplicates", "reservoir_duplicates"),
            "p95",
        )
        if reservoir_union_p95 <= 0.0:
            hints.append("candidate_reservoirs_no_union")
        elif reservoir_score_p95 > 0.0 and reservoir_union_p95 <= reservoir_score_p95:
            hints.append("candidate_reservoirs_no_extra_docs")
        if reservoir_duplicates_p95 > reservoir_union_p95 and reservoir_union_p95 > 0.0:
            hints.append("candidate_reservoirs_high_duplicates")

    proxy_limit_source = row.get("proxy_candidate_limit_source")
    if proxy_limit_source is None:
        proxy_limit_source = sample.get("proxy_candidate_limit_source")
    if proxy_limit_source == "search_ef":
        hints.append("proxy_candidates_capped_by_search_ef")
    elif proxy_limit_source == "doc_candidate_k":
        hints.append("proxy_candidates_capped_by_doc_candidate_k")

    bm25_injection = row.get("bm25_candidate_injection")
    bm25_enabled_queries = int(row.get("bm25_injection_enabled_queries") or 0)
    if bm25_injection and bm25_injection != "off":
        if row.get("bm25_text_query_available") is False:
            hints.append("bm25_rescue_missing_text_query")
        if bm25_enabled_queries <= 0:
            hints.append("bm25_rescue_not_triggering")
        reason_counts = row.get("bm25_injection_limit_reason")
        if summary_reason_present(reason_counts, "exact_rerank_k"):
            hints.append("bm25_rescue_limited_by_exact_rerank_k")
        if summary_reason_present(reason_counts, "doc_candidate_k"):
            hints.append("bm25_rescue_limited_by_doc_candidate_k")
        if summary_reason_present(reason_counts, "bm25_count"):
            hints.append("bm25_rescue_lexical_underfill")
            hints.append("bm25_rescue_underfilled")
        if summary_stat_float(row.get("bm25_injection_candidates"), "p95") <= 0.0:
            hints.append("bm25_rescue_underfilled")
        if (
            summary_stat_float(row.get("bm25_injection_candidates"), "p95") > 0.0
            and summary_stat_float(row.get("bm25_injection_retained"), "p95") <= 0.0
        ):
            hints.append("bm25_rescue_not_retained")
        for admission_delta_key in (
            "exact_top10_admission_recall_delta",
            "bm25_admission_delta",
        ):
            if admission_delta_key in row:
                try:
                    if float(row.get(admission_delta_key) or 0.0) <= 0.0:
                        hints.append("bm25_rescue_no_admission_gain")
                except (TypeError, ValueError):
                    pass
                break
        for latency_delta_key in ("p95_ms_delta", "bm25_latency_delta_ms"):
            if latency_delta_key in row:
                try:
                    if float(row.get(latency_delta_key) or 0.0) > 0.0:
                        hints.append("bm25_rescue_latency_regression")
                except (TypeError, ValueError):
                    pass
                break

    sparse_source = row.get("sparse_candidate_source")
    if sparse_source == "learned_sparse":
        if bool(row.get("learned_sparse_partial_coverage", False)):
            hints.append("learned_sparse_partial_coverage")
        if summary_stat_float(row.get("learned_sparse_candidates"), "p95") <= 0.0:
            hints.append("learned_sparse_no_candidates")
        elif summary_stat_float(row.get("learned_sparse_retained_for_maxsim"), "p95") <= 0.0:
            hints.append("learned_sparse_not_retained")

    if row.get("candidate_source") == "centroid_lite":
        for warning_name in row.get("centroid_lite_warnings", []) or []:
            hints.append(str(warning_name))
        if str(row.get("comparison", "")) == "centroid_lite_posting_cap":
            try:
                admission_delta = float(
                    row.get("exact_top10_admission_recall_delta") or 0.0
                )
            except (TypeError, ValueError):
                admission_delta = 0.0
            try:
                latency_delta = float(row.get("p95_ms_delta") or 0.0)
            except (TypeError, ValueError):
                latency_delta = 0.0
            if admission_delta <= 0.0:
                hints.append("centroid_lite_no_admission_gain")
            if admission_delta < -0.02:
                hints.append("centroid_lite_cap_too_aggressive")
            if latency_delta > 0.0:
                hints.append("centroid_lite_latency_regression")

    warning = str(sample.get("multivector_doc_graph_warning", "") or "")
    if "near_exhaustive" in warning:
        hints.append("candidate_source_near_exhaustive_scan")

    if not hints:
        hints.append("try_higher_quality_candidate_source_or_rescue")
    return sorted(dict.fromkeys(hints))


def serving_profile_explanation(row: dict[str, Any]) -> str:
    if not isinstance(row, dict) or not row:
        return "No profile was selected."
    sample = row.get("last_scan_stats_sample", {})
    if not isinstance(sample, dict):
        sample = {}
    parts = [
        f"source={row.get('candidate_source', '')}",
        f"p95={float(serving_row_p95(row) or 0.0):.3f}ms",
        f"top10_admission={float(row.get('exact_top10_admission_recall') or 0.0):.6f}",
    ]
    if row.get("ndcg@10") is not None:
        parts.append(f"ndcg@10={float(row.get('ndcg@10') or 0.0):.6f}")
    if row.get("entry_sidecar") is not None:
        parts.append(f"entry_sidecar={bool(row.get('entry_sidecar'))}")
    if row.get("entry_sidecar_representatives") is not None:
        parts.append(f"entry_sidecar_reps={row.get('entry_sidecar_representatives')}")
    if row.get("entry_sidecar_strategy") is not None:
        parts.append(f"entry_sidecar_strategy={row.get('entry_sidecar_strategy')}")
    if row.get("candidate_reservoirs") not in (None, "off"):
        parts.append(f"reservoirs={row.get('candidate_reservoirs')}")
    for key, label in (
        ("bm25_injection_enabled_queries", "bm25_enabled_queries"),
        ("bm25_injection_candidates", "bm25_candidates_summary"),
        ("bm25_injection_candidate_limit", "bm25_candidate_limit_summary"),
        ("bm25_injection_pool_size", "bm25_pool_size_summary"),
        ("bm25_injection_limit_reason", "bm25_limit_reason_summary"),
        ("bm25_injection_retained", "bm25_retained_summary"),
        ("bm25_injection_exact_reranked", "bm25_exact_reranked_summary"),
        ("learned_sparse_candidates", "learned_sparse_candidates_summary"),
        ("learned_sparse_retained_for_maxsim", "learned_sparse_retained_summary"),
        ("learned_sparse_branch_latency_us", "learned_sparse_latency_us_summary"),
        ("multivector_doc_graph_warning", "warning"),
        ("graph_entry_sidecar_count", "entry_sidecar_count"),
        ("graph_entry_sidecar_scored", "entry_sidecar_scored"),
        ("graph_entry_sidecar_selected", "entry_sidecar_selected"),
        ("graph_entry_sidecar_strategy", "entry_sidecar_scan_strategy"),
        ("proxy_encoder_kind", "proxy"),
        ("proxy_candidates", "proxy_candidates"),
        ("centroid_candidates", "centroid_candidates"),
        ("centroid_docs_touched", "centroid_docs_touched"),
        ("multivector_doc_graph_exact_rerank_docs", "doc_graph_rerank_docs"),
        ("multivector_exact_rerank_docs", "exact_rerank_docs"),
        ("multivector_exact_rerank_pairs", "exact_pairs"),
        ("multivector_exact_kernel", "kernel"),
        ("multivector_doc_sidecar_cache_mode", "sidecar_cache"),
        ("multivector_doc_sidecar_bytes_touched", "sidecar_bytes"),
        ("multivector_doc_sidecar_docmap_bytes_touched", "sidecar_docmap_bytes"),
        ("multivector_doc_sidecar_resident_vectors_loaded", "sidecar_resident_vectors"),
        ("multivector_doc_sidecar_resident_bytes_loaded", "sidecar_resident_bytes"),
        ("multivector_doc_sidecar_vector_chunk_ref_bytes_touched", "sidecar_chunk_ref_bytes"),
        ("multivector_doc_sidecar_paged_vector_bytes_touched", "sidecar_paged_vector_bytes"),
        ("multivector_tokens_original", "tokens_original"),
        ("multivector_tokens_pooled", "tokens_pooled"),
        ("multivector_bm25_injection_enabled", "bm25_injection"),
        ("multivector_bm25_injection_candidates", "bm25_candidates"),
        ("multivector_bm25_injection_candidate_limit", "bm25_candidate_limit"),
        ("multivector_bm25_injection_pool_size", "bm25_pool_size"),
        ("multivector_bm25_injection_limit_reason", "bm25_limit_reason"),
        ("multivector_bm25_injection_retained", "bm25_retained"),
        ("multivector_bm25_injection_exact_reranked", "bm25_exact_reranked"),
        ("learned_sparse_candidates", "learned_sparse_candidates"),
        ("learned_sparse_retained_for_maxsim", "learned_sparse_retained"),
        ("reservoirs_enabled_queries", "reservoirs_enabled_queries"),
        ("reservoir_score_docs", "reservoir_score_docs_summary"),
        ("reservoir_coverage_docs", "reservoir_coverage_docs_summary"),
        ("reservoir_mean_docs", "reservoir_mean_docs_summary"),
        ("reservoir_per_token_docs", "reservoir_per_token_docs_summary"),
        ("reservoir_union_docs", "reservoir_union_docs_summary"),
        ("reservoir_duplicates", "reservoir_duplicates_summary"),
        ("multivector_reservoirs_enabled", "reservoirs_enabled"),
        ("multivector_reservoir_score_docs", "reservoir_score_docs"),
        ("multivector_reservoir_coverage_docs", "reservoir_coverage_docs"),
        ("multivector_reservoir_mean_docs", "reservoir_mean_docs"),
        ("multivector_reservoir_per_token_docs", "reservoir_per_token_docs"),
        ("multivector_reservoir_union_docs", "reservoir_union_docs"),
        ("multivector_reservoir_duplicates", "reservoir_duplicates"),
        ("quantized_inverted_postings_touched", "quantized_postings"),
    ):
        if key in row and row.get(key) is not None:
            parts.append(f"{label}={row[key]}")
        elif key in sample:
            parts.append(f"{label}={sample[key]}")
    return "; ".join(parts)


def serving_quality_available(rows: list[dict[str, Any]]) -> bool:
    return any(serving_row_metric(row, "ndcg@10") is not None for row in rows)


def serving_exact_baseline_from_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    for item in results:
        if not isinstance(item, dict):
            continue
        if item.get("method") == EXACT_SCAN_METHOD or item.get("retrieval_mode") == "multivector_exact_scan":
            metrics = item.get("metrics", {})
            serial_latency = item.get("serial_latency", {})
            return {
                "available": True,
                "method": item.get("method"),
                "metrics": metrics if isinstance(metrics, dict) else {},
                "serial_latency": serial_latency if isinstance(serial_latency, dict) else {},
            }
    return {
        "available": False,
        "reason": "exact scan method was not included in this benchmark run",
    }


def serving_threshold_failures(
    row: dict[str, Any],
    *,
    min_top10_admission: float,
    min_ndcg_ratio_vs_exact: float,
    max_p95_ms: float,
    exact_baseline_ndcg: float | None,
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    unavailable: list[str] = []
    p95 = serving_row_p95(row)
    if p95 is None:
        failures.append("missing_p95_latency")
    elif max_p95_ms > 0.0 and p95 > max_p95_ms:
        failures.append("p95_latency_above_cap")

    admission = serving_row_metric(row, "exact_top10_admission_recall")
    if admission is None:
        failures.append("missing_exact_top10_admission_recall")
    elif admission < min_top10_admission:
        failures.append("top10_admission_below_threshold")

    ndcg = serving_row_metric(row, "ndcg@10")
    if exact_baseline_ndcg is None:
        unavailable.append("ndcg_ratio_vs_exact")
    elif ndcg is None:
        failures.append("missing_ndcg@10")
    elif ndcg < exact_baseline_ndcg * min_ndcg_ratio_vs_exact:
        failures.append("ndcg_ratio_below_threshold")

    if bool(row.get("learned_sparse_partial_coverage", False)):
        failures.append("learned_sparse_partial_coverage")
    if serving_row_unsupported_reservoir_evidence(row):
        failures.append("candidate_reservoirs_not_executed")

    return failures, unavailable


def pareto_frontier_latency_quality(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    use_ndcg = serving_quality_available(rows)
    frontier: list[dict[str, Any]] = []
    for row in rows:
        p95 = serving_row_p95(row)
        if p95 is None:
            continue
        admission = serving_row_metric(row, "exact_top10_admission_recall") or 0.0
        ndcg = serving_row_metric(row, "ndcg@10") or 0.0
        dominated = False
        for other in rows:
            if other is row:
                continue
            other_p95 = serving_row_p95(other)
            other_comparable_p95 = other_p95 if other_p95 is not None else float("inf")
            other_admission = serving_row_metric(other, "exact_top10_admission_recall") or 0.0
            other_ndcg = serving_row_metric(other, "ndcg@10") or 0.0
            better_or_equal = (
                other_comparable_p95 <= p95
                and other_admission >= admission
                and (not use_ndcg or other_ndcg >= ndcg)
            )
            strictly_better = (
                other_comparable_p95 < p95
                or other_admission > admission
                or (use_ndcg and other_ndcg > ndcg)
            )
            if better_or_equal and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(serving_recommendation_row(row))
    return sorted(
        frontier,
        key=lambda item: (
            serving_row_p95(item) if serving_row_p95(item) is not None else float("inf"),
            str(item.get("id", "")),
        ),
    )


def compute_document_node_serving_recommendation(
    serving_grid: dict[str, Any],
    *,
    exact_baseline: dict[str, Any],
    min_top10_admission: float,
    min_ndcg_ratio_vs_exact: float,
    max_p95_ms: float,
) -> dict[str, Any]:
    raw_rows = serving_grid.get("results", serving_grid.get("summary_rows", []))
    rows = [
        item
        for item in raw_rows
        if isinstance(item, dict)
    ]
    exact_metrics = exact_baseline.get("metrics", {}) if isinstance(exact_baseline, dict) else {}
    exact_baseline_ndcg = (
        serving_row_metric({"ndcg@10": exact_metrics.get("ndcg@10")}, "ndcg@10")
        if isinstance(exact_metrics, dict)
        else None
    )
    threshold_rows: list[dict[str, Any]] = []
    rejected_profiles: list[dict[str, Any]] = []
    for row in rows:
        failures, unavailable = serving_threshold_failures(
            row,
            min_top10_admission=min_top10_admission,
            min_ndcg_ratio_vs_exact=min_ndcg_ratio_vs_exact,
            max_p95_ms=max_p95_ms,
            exact_baseline_ndcg=exact_baseline_ndcg,
        )
        summary = serving_recommendation_row(row)
        candidate_bottleneck = serving_row_candidate_admission_bottleneck(
            row,
            min_top10_admission=min_top10_admission,
        )
        summary["threshold_pass"] = not failures
        summary["failure_reasons"] = failures
        if (
            summary["failure_reasons"]
            and bool(summary.get("candidate_underfill"))
            and "candidate_underfill" not in summary["failure_reasons"]
        ):
            summary["failure_reasons"] = [
                *summary["failure_reasons"],
                "candidate_underfill",
            ]
        if candidate_bottleneck and "candidate_admission_bottleneck" not in summary["failure_reasons"]:
            summary["failure_reasons"] = [
                *summary["failure_reasons"],
                "candidate_admission_bottleneck",
            ]
        summary["unavailable_criteria"] = unavailable
        summary["candidate_admission_bottleneck"] = candidate_bottleneck
        if candidate_bottleneck:
            summary["candidate_admission_bottleneck_reason"] = (
                "exact rerank exhausted the admitted candidate band, but "
                "exact_top10_admission_recall remained below threshold"
            )
        summary["admission_improvement_hints"] = serving_admission_improvement_hints(
            summary,
            min_top10_admission=min_top10_admission,
        )
        threshold_rows.append(summary)
        if summary["failure_reasons"]:
            rejected_profiles.append(summary)

    eligible = [
        item for item in threshold_rows
        if (
            item["threshold_pass"]
            and not serving_row_experimental(item)
            and serving_row_p95(item) is not None
        )
    ]
    best_latency_safe = min(
        eligible,
        key=lambda item: (
            serving_row_p95(item) if serving_row_p95(item) is not None else float("inf"),
            str(item.get("id", "")),
        ),
        default=None,
    )
    if best_latency_safe is not None:
        best_latency_safe = {
            **best_latency_safe,
            "criteria_unavailable": sorted({
                criterion
                for item in threshold_rows
                for criterion in item.get("unavailable_criteria", [])
            }),
        }

    use_ndcg = serving_quality_available(rows)
    quality_rows = [
        row for row in rows
        if not serving_row_unsupported_reservoir_evidence(row)
    ] or rows
    best_quality_row = max(
        quality_rows,
        key=lambda item: (
            (serving_row_metric(item, "ndcg@10") or 0.0)
            if use_ndcg
            else (serving_row_metric(item, "exact_top10_admission_recall") or 0.0),
            serving_row_metric(item, "exact_top10_admission_recall") or 0.0,
            0 if not serving_row_unsupported_reservoir_evidence(item) else -1,
            0 if not serving_row_experimental(item) else -1,
            -(serving_row_p95(item) if serving_row_p95(item) is not None else float("inf")),
            str(item.get("profile", "")),
        ),
        default=None,
    )
    best_quality = serving_recommendation_row(best_quality_row) if best_quality_row else None
    if best_quality is not None:
        best_quality["selection_basis"] = "ndcg@10" if use_ndcg else "exact_top10_admission_recall"

    ordered_by_latency = sorted(
        rows,
        key=lambda item: (
            serving_row_p95(item) if serving_row_p95(item) is not None else float("inf"),
            str(item.get("profile", "")),
            int(item.get("ef", 0) or 0),
            int(item.get("oversampling", 0) or 0),
        ),
    )
    latency_rank = {
        serving_row_id(item): rank
        for rank, item in enumerate(ordered_by_latency, start=1)
    }
    best_admission = max(
        (serving_row_metric(item, "exact_top10_admission_recall") or 0.0 for item in rows),
        default=0.0,
    )
    best_ndcg = max(
        (serving_row_metric(item, "ndcg@10") or 0.0 for item in rows),
        default=0.0,
    )
    non_experimental_has_usable_metrics = any(
        not serving_row_experimental(item) and serving_row_has_usable_metrics(item)
        for item in rows
    )
    balanced_candidates: list[dict[str, Any]] = []
    for row in rows:
        admission = serving_row_metric(row, "exact_top10_admission_recall") or 0.0
        ndcg = serving_row_metric(row, "ndcg@10") or 0.0
        admission_loss_penalty = round(max(0.0, best_admission - admission) * 100.0, 3)
        quality_loss_penalty = round(max(0.0, best_ndcg - ndcg) * 100.0, 3) if use_ndcg else 0.0
        storage_penalty = serving_storage_penalty(row)
        experimental_penalty = (
            1_000_000
            if serving_row_experimental(row) and non_experimental_has_usable_metrics
            else 0
        )
        partial_evidence_penalty = (
            500_000
            if bool(row.get("learned_sparse_partial_coverage", False))
            and non_experimental_has_usable_metrics
            else 0
        )
        unsupported_reservoir_penalty = (
            500_000
            if serving_row_unsupported_reservoir_evidence(row)
            and non_experimental_has_usable_metrics
            else 0
        )
        score = (
            latency_rank.get(serving_row_id(row), len(rows) + 1)
            + admission_loss_penalty
            + quality_loss_penalty
            + storage_penalty
            + experimental_penalty
            + partial_evidence_penalty
            + unsupported_reservoir_penalty
        )
        balanced_candidates.append({
            **serving_recommendation_row(row),
            "score": round(score, 3),
            "score_components": {
                "latency_rank": latency_rank.get(serving_row_id(row), len(rows) + 1),
                "admission_loss_penalty": admission_loss_penalty,
                "quality_loss_penalty": quality_loss_penalty,
                "storage_penalty": storage_penalty,
                "experimental_penalty": experimental_penalty,
                "partial_evidence_penalty": partial_evidence_penalty,
                "unsupported_reservoir_penalty": unsupported_reservoir_penalty,
            },
        })
    best_balanced = min(
        balanced_candidates,
        key=lambda item: (
            float(item["score"]),
            serving_row_p95(item) if serving_row_p95(item) is not None else float("inf"),
            str(item.get("id", "")),
        ),
        default=None,
    )

    return {
        "thresholds": {
            "serving_min_top10_admission": min_top10_admission,
            "serving_min_ndcg_ratio_vs_exact": min_ndcg_ratio_vs_exact,
            "serving_max_p95_ms": max_p95_ms,
        },
        "exact_baseline": exact_baseline,
        "best_latency_safe": best_latency_safe,
        "best_quality": best_quality,
        "best_balanced": best_balanced,
        "balanced_formula": (
            "score = latency_rank + max(0,best_admission-admission)*100 "
            "+ max(0,best_ndcg-ndcg)*100 when qrels exist + storage_penalty; "
            "storage_penalty is sq8=0, f16=1, f32=2, other=3; "
            "experimental_penalty is 1000000 when any non-experimental profile has usable metrics; "
            "learned-sparse partial-coverage and unsupported-reservoir penalties are 500000 under the same condition"
        ),
        "pareto_frontier_latency_quality": pareto_frontier_latency_quality(rows),
        "pareto_frontier": pareto_frontier_latency_quality(rows),
        "profile_thresholds": sorted(
            threshold_rows,
            key=lambda item: (
                serving_row_p95(item) if serving_row_p95(item) is not None else float("inf"),
                str(item.get("id", "")),
            ),
        ),
        "candidate_admission_bottlenecks": sorted(
            [
                item for item in threshold_rows
                if item.get("candidate_admission_bottleneck")
            ],
            key=lambda item: (
                float(item.get("exact_top10_admission_recall") or 0.0),
                serving_row_p95(item) if serving_row_p95(item) is not None else float("inf"),
                str(item.get("id", "")),
            ),
        ),
        "candidate_source_delta_summary": document_node_candidate_source_delta_summary(
            serving_grid.get("candidate_source_deltas")
        ),
        "rejected_profiles": sorted(
            rejected_profiles,
            key=lambda item: (
                serving_row_p95(item) if serving_row_p95(item) is not None else float("inf"),
                str(item.get("id", "")),
            ),
        ),
    }


def compute_document_node_token_pooling_recommendation(
    serving_grid: dict[str, Any],
    *,
    exact_baseline: dict[str, Any],
    min_top10_admission: float,
    min_ndcg_ratio_vs_exact: float,
    max_p95_ms: float,
) -> dict[str, Any]:
    raw_rows = serving_grid.get("results", serving_grid.get("summary_rows", []))
    rows = [
        item
        for item in raw_rows
        if isinstance(item, dict)
        and str(item.get("profile", "")) in DOCUMENT_NODE_SERVING_GRID_TOKEN_POOLING_FOCUS_PROFILES
    ]
    exact_metrics = exact_baseline.get("metrics", {}) if isinstance(exact_baseline, dict) else {}
    exact_baseline_ndcg = (
        serving_row_metric({"ndcg@10": exact_metrics.get("ndcg@10")}, "ndcg@10")
        if isinstance(exact_metrics, dict)
        else None
    )
    threshold_rows: list[dict[str, Any]] = []
    rejected_profiles: list[dict[str, Any]] = []
    unavailable_criteria: set[str] = set()
    for row in rows:
        failures, unavailable = serving_threshold_failures(
            row,
            min_top10_admission=min_top10_admission,
            min_ndcg_ratio_vs_exact=min_ndcg_ratio_vs_exact,
            max_p95_ms=max_p95_ms,
            exact_baseline_ndcg=exact_baseline_ndcg,
        )
        summary = serving_recommendation_row(row)
        summary["threshold_pass"] = not failures
        summary["failure_reasons"] = failures
        summary["unavailable_criteria"] = unavailable
        summary["selection_metrics"] = {
            "p95_ms": serving_row_p95(row),
            "exact_top10_admission_recall": serving_row_metric(
                row,
                "exact_top10_admission_recall",
            ),
            "ndcg@10": serving_row_metric(row, "ndcg@10"),
            "tokens_pooled_p50": row.get("tokens_pooled_p50"),
            "token_pooling_ratio_p50": row.get("token_pooling_ratio_p50"),
            "exact_rerank_pairs_p50": row.get("exact_rerank_pairs_p50"),
            "index_build_elapsed_ms": row.get("index_build_elapsed_ms"),
        }
        unavailable_criteria.update(str(item) for item in unavailable)
        threshold_rows.append(summary)
        if failures:
            rejected_profiles.append(summary)

    eligible = [
        item for item in threshold_rows
        if (
            item["threshold_pass"]
            and not serving_row_experimental(item)
            and serving_row_p95(item) is not None
        )
    ]
    best_pooling_latency_safe = min(
        eligible,
        key=lambda item: (
            serving_row_p95(item) if serving_row_p95(item) is not None else float("inf"),
            str(item.get("id", "")),
        ),
        default=None,
    )
    use_ndcg = serving_quality_available(rows)
    best_pooling_quality_safe = max(
        eligible,
        key=lambda item: (
            (serving_row_metric(item, "ndcg@10") or 0.0)
            if use_ndcg
            else (serving_row_metric(item, "exact_top10_admission_recall") or 0.0),
            serving_row_metric(item, "exact_top10_admission_recall") or 0.0,
            -(serving_row_p95(item) if serving_row_p95(item) is not None else float("inf")),
            str(item.get("id", "")),
        ),
        default=None,
    )
    if best_pooling_latency_safe is not None:
        best_pooling_latency_safe = {
            **best_pooling_latency_safe,
            "criteria_unavailable": sorted(unavailable_criteria),
            "selection_basis": "lowest_p95_passing_thresholds",
        }
    if best_pooling_quality_safe is not None:
        best_pooling_quality_safe = {
            **best_pooling_quality_safe,
            "criteria_unavailable": sorted(unavailable_criteria),
            "selection_basis": "ndcg@10" if use_ndcg else "exact_top10_admission_recall",
        }
    return {
        "available": bool(rows),
        "profile_count": len(rows),
        "thresholds": {
            "serving_min_top10_admission": min_top10_admission,
            "serving_min_ndcg_ratio_vs_exact": min_ndcg_ratio_vs_exact,
            "serving_max_p95_ms": max_p95_ms,
        },
        "exact_baseline": exact_baseline,
        "selection_basis": (
            "qrels_ndcg_and_admission_latency"
            if use_ndcg
            else "admission_latency_without_qrels"
        ),
        "best_pooling_latency_safe": best_pooling_latency_safe,
        "best_pooling_quality_safe": best_pooling_quality_safe,
        "profile_thresholds": sorted(
            threshold_rows,
            key=lambda item: (
                serving_row_p95(item) if serving_row_p95(item) is not None else float("inf"),
                str(item.get("id", "")),
            ),
        ),
        "rejected_pooling_profiles": sorted(
            rejected_profiles,
            key=lambda item: (
                serving_row_p95(item) if serving_row_p95(item) is not None else float("inf"),
                str(item.get("id", "")),
            ),
        ),
    }


def _synthetic_serving_row(
    profile: str,
    *,
    p95: float | None,
    admission: float,
    ndcg: float | None,
    storage: str = "f16",
    candidate_source: str = "proxy_vector",
    proxy_encoder: str = "normalized_mean",
    centroids: str = "off",
    token_pooling: str = "off",
    token_pooling_target_ratio: float = 1.0,
    ef: int = 100,
    oversampling: int = 1,
    stats_sample: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if stats_sample is None:
        stats_sample = {}
    return {
        "profile": profile,
        "candidate_source": candidate_source,
        "proxy_encoder": proxy_encoder,
        "centroids": centroids,
        "storage_kind": storage,
        "cache_mode": "auto",
        "token_pooling": token_pooling,
        "token_pooling_target_ratio": token_pooling_target_ratio,
        "ef": ef,
        "oversampling": oversampling,
        "largest_budget": 800,
        "p50_latency_ms_at_largest_budget": None if p95 is None else round(p95 * 0.7, 3),
        "p95_latency_ms_at_largest_budget": p95,
        "exact_top1_admission_rate": admission,
        "exact_top10_admission_recall": admission,
        "recall@10": ndcg,
        "ndcg@10": ndcg,
        "mrr@10": ndcg,
        "index_bytes": 1000,
        "stats_available": document_node_serving_stats_available(stats_sample),
        "last_scan_stats_sample": stats_sample,
    }


def _self_check_document_node_serving_recommendation() -> None:
    rows = [
        _synthetic_serving_row(
            "latency_winner_fails_admission",
            p95=8.0,
            admission=0.55,
            ndcg=0.70,
            storage="sq8",
            stats_sample={"proxy_candidate_limit_source": "search_ef"},
        ),
        _synthetic_serving_row("latency_safe_winner", p95=12.0, admission=0.86, ndcg=0.86, storage="f16"),
        _synthetic_serving_row(
            "balanced_winner",
            p95=20.0,
            admission=0.94,
            ndcg=0.88,
            storage="sq8",
            stats_sample={
                "multivector_candidate_source": "proxy_vector",
                "proxy_encoder_kind": "normalized_mean",
                "multivector_doc_graph_exact_rerank_docs": 400,
                "multivector_exact_rerank_pairs": 8192,
                "multivector_exact_kernel": "blocked_avx2",
                "multivector_doc_sidecar_cache_mode": "resident",
            },
        ),
        _synthetic_serving_row("quality_winner_slow", p95=80.0, admission=0.91, ndcg=0.90, storage="f32"),
        _synthetic_serving_row("missing_latency", p95=None, admission=0.88, ndcg=0.86, storage="f16"),
    ]
    bm25_limited = _synthetic_serving_row(
        "bm25_limited_low_admission",
        p95=30.0,
        admission=0.60,
        ndcg=0.70,
        storage="f16",
    )
    bm25_limited.update({
        "bm25_candidate_injection": "dense_with_text",
        "bm25_text_query_available": False,
        "bm25_injection_enabled_queries": 4,
        "bm25_injection_candidates": {"count": 5, "p95": 12.0},
        "bm25_injection_retained": {"count": 5, "p95": 0.0},
        "bm25_injection_limit_reason": {"exact_rerank_k": 3, "bm25_count": 2},
        "exact_top10_admission_recall_delta": 0.0,
        "p95_ms_delta": 8.0,
    })
    learned_sparse_partial = _synthetic_serving_row(
        "learned_sparse_partial_low_admission",
        p95=35.0,
        admission=0.62,
        ndcg=0.72,
        storage="f16",
    )
    learned_sparse_partial.update({
        "sparse_candidate_source": "learned_sparse",
        "learned_sparse_partial_coverage": True,
        "learned_sparse_candidates": {"count": 5, "p95": 0.0},
        "learned_sparse_retained_for_maxsim": {"count": 5, "p95": 0.0},
    })
    rows.extend([bm25_limited, learned_sparse_partial])
    grid = {"results": rows}
    exact = {"available": True, "metrics": {"ndcg@10": 0.90}}
    rec = compute_document_node_serving_recommendation(
        grid,
        exact_baseline=exact,
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    assert rec["best_latency_safe"]["profile"] == "latency_safe_winner"
    assert rec["best_quality"]["profile"] == "quality_winner_slow"
    assert rec["best_balanced"]["profile"] == "balanced_winner"
    assert rec["best_balanced"]["profile"] != rec["best_latency_safe"]["profile"]
    assert rec["best_balanced"]["profile"] != rec["best_quality"]["profile"]
    rejected = {item["profile"]: item["failure_reasons"] for item in rec["rejected_profiles"]}
    rejected_hints = {
        item["profile"]: item["admission_improvement_hints"]
        for item in rec["rejected_profiles"]
    }
    assert "top10_admission_below_threshold" in rejected["latency_winner_fails_admission"]
    assert "proxy_candidates_capped_by_search_ef" in rejected_hints["latency_winner_fails_admission"]
    assert "bm25_rescue_limited_by_exact_rerank_k" in rejected_hints["bm25_limited_low_admission"]
    assert "bm25_rescue_missing_text_query" in rejected_hints["bm25_limited_low_admission"]
    assert "bm25_rescue_underfilled" in rejected_hints["bm25_limited_low_admission"]
    assert "bm25_rescue_lexical_underfill" in rejected_hints["bm25_limited_low_admission"]
    assert "bm25_rescue_not_retained" in rejected_hints["bm25_limited_low_admission"]
    assert "bm25_rescue_no_admission_gain" in rejected_hints["bm25_limited_low_admission"]
    assert "bm25_rescue_latency_regression" in rejected_hints["bm25_limited_low_admission"]
    assert "learned_sparse_partial_coverage" in rejected_hints["learned_sparse_partial_low_admission"]
    assert "learned_sparse_no_candidates" in rejected_hints["learned_sparse_partial_low_admission"]
    assert "missing_p95_latency" in rejected["missing_latency"]
    rec_again = compute_document_node_serving_recommendation(
        grid,
        exact_baseline=exact,
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    assert rec == rec_again

    no_qrels_rows = [
        _synthetic_serving_row("fast_no_qrels", p95=10.0, admission=0.82, ndcg=None, storage="sq8"),
        _synthetic_serving_row("quality_no_qrels", p95=50.0, admission=0.95, ndcg=None, storage="f16"),
    ]
    no_qrels = compute_document_node_serving_recommendation(
        {"summary_rows": no_qrels_rows},
        exact_baseline={"available": False},
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    assert no_qrels["best_latency_safe"]["profile"] == "fast_no_qrels"
    assert "ndcg_ratio_vs_exact" in no_qrels["best_latency_safe"]["criteria_unavailable"]
    assert no_qrels["best_quality"]["profile"] == "quality_no_qrels"
    assert no_qrels["best_quality"]["selection_basis"] == "exact_top10_admission_recall"

    exact_baseline_missing_rows = [
        _synthetic_serving_row("baseline_missing_fast", p95=7.0, admission=0.82, ndcg=0.50, storage="sq8"),
        _synthetic_serving_row("baseline_missing_quality", p95=40.0, admission=0.95, ndcg=0.90, storage="f16"),
    ]
    exact_baseline_missing = compute_document_node_serving_recommendation(
        {"results": exact_baseline_missing_rows},
        exact_baseline={"available": False, "reason": "synthetic exact baseline missing"},
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    assert exact_baseline_missing["best_latency_safe"]["profile"] == "baseline_missing_fast"
    assert (
        "ndcg_ratio_vs_exact"
        in exact_baseline_missing["best_latency_safe"]["criteria_unavailable"]
    )
    assert exact_baseline_missing["best_quality"]["profile"] == "baseline_missing_quality"
    assert exact_baseline_missing["best_quality"]["selection_basis"] == "ndcg@10"

    none_pass = compute_document_node_serving_recommendation(
        {
            "results": [
                _synthetic_serving_row("too_low_admission", p95=9.0, admission=0.40, ndcg=0.60, storage="sq8"),
                _synthetic_serving_row("too_slow", p95=200.0, admission=0.90, ndcg=0.88, storage="f16"),
            ]
        },
        exact_baseline=exact,
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=100.0,
    )
    assert none_pass["best_latency_safe"] is None
    none_pass_rejected = {item["profile"]: item["failure_reasons"] for item in none_pass["rejected_profiles"]}
    assert "top10_admission_below_threshold" in none_pass_rejected["too_low_admission"]
    assert "p95_latency_above_cap" in none_pass_rejected["too_slow"]

    bottleneck_row = _synthetic_serving_row(
        "centroid_mean_exhaustive_low_admission",
        p95=113.0,
        admission=0.388,
        ndcg=0.369,
        storage="f16",
        proxy_encoder="centroid_mean",
        centroids="kmeans",
    )
    bottleneck_row.update({
        "serving_exact_rerank_mode": "admission_exhaustive",
        "candidate_budget": 800,
        "effective_exact_rerank_k": 800,
        "exact_rerank_docs_p95": 800.0,
    })
    bottleneck_rec = compute_document_node_serving_recommendation(
        {"results": [bottleneck_row]},
        exact_baseline=exact,
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    bottlenecks = bottleneck_rec["candidate_admission_bottlenecks"]
    assert len(bottlenecks) == 1
    assert bottlenecks[0]["profile"] == "centroid_mean_exhaustive_low_admission"
    assert bottlenecks[0]["candidate_admission_bottleneck"] is True
    bottleneck_rejected = {
        item["profile"]: item["failure_reasons"]
        for item in bottleneck_rec["rejected_profiles"]
    }
    assert (
        "candidate_admission_bottleneck"
        in bottleneck_rejected["centroid_mean_exhaustive_low_admission"]
    )
    bottleneck_hints = {
        item["profile"]: item["admission_improvement_hints"]
        for item in bottleneck_rec["rejected_profiles"]
    }
    assert "admitted_band_exhausted" in bottleneck_hints["centroid_mean_exhaustive_low_admission"]
    assert (
        "try_sparse_rescue_or_centroid_lite"
        in bottleneck_hints["centroid_mean_exhaustive_low_admission"]
    )
    assert (
        "try_balanced_candidate_reservoirs"
        in bottleneck_hints["centroid_mean_exhaustive_low_admission"]
    )
    assert (
        "try_bm25_or_learned_sparse_rescue"
        in bottleneck_hints["centroid_mean_exhaustive_low_admission"]
    )
    bottleneck_markdown = markdown_benchmark_summary({
        "document_node_serving_recommendation": bottleneck_rec,
    })
    assert "#### Candidate admission bottlenecks" in bottleneck_markdown
    assert "try_sparse_rescue_or_centroid_lite" in bottleneck_markdown

    reservoir_bottleneck = _synthetic_serving_row(
        "centroid_mean_reservoir_low_admission",
        p95=118.0,
        admission=0.45,
        ndcg=0.40,
        storage="f16",
        proxy_encoder="centroid_mean",
        centroids="kmeans",
    )
    reservoir_bottleneck.update({
        "candidate_reservoirs": "balanced",
        "serving_exact_rerank_mode": "admission_exhaustive",
        "candidate_budget": 800,
        "effective_exact_rerank_k": 800,
        "exact_rerank_docs_p95": 800.0,
        "reservoirs_enabled_queries": 5,
        "multivector_reservoir_score_docs": {"count": 5, "p95": 800.0},
        "multivector_reservoir_union_docs": {"count": 5, "p95": 800.0},
        "multivector_reservoir_duplicates": {"count": 5, "p95": 900.0},
    })
    reservoir_rec = compute_document_node_serving_recommendation(
        {"results": [reservoir_bottleneck]},
        exact_baseline=exact,
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    reservoir_hints = {
        item["profile"]: item["admission_improvement_hints"]
        for item in reservoir_rec["rejected_profiles"]
    }
    assert (
        "try_balanced_candidate_reservoirs"
        not in reservoir_hints["centroid_mean_reservoir_low_admission"]
    )
    assert (
        "candidate_reservoirs_no_extra_docs"
        in reservoir_hints["centroid_mean_reservoir_low_admission"]
    )
    assert (
        "candidate_reservoirs_high_duplicates"
        in reservoir_hints["centroid_mean_reservoir_low_admission"]
    )

    reservoir_noop = _synthetic_serving_row(
        "centroid_mean_reservoir_noop",
        p95=60.0,
        admission=0.90,
        ndcg=0.90,
        storage="f16",
        proxy_encoder="centroid_mean",
        centroids="kmeans",
    )
    reservoir_noop.update({
        "candidate_reservoirs": "balanced",
        "reservoirs_enabled_queries": 0,
        "last_scan_stats_sample": {
            "multivector_reservoirs_enabled": False,
        },
    })
    reservoir_supported = _synthetic_serving_row(
        "centroid_mean_reservoir_supported",
        p95=70.0,
        admission=0.88,
        ndcg=0.88,
        storage="f16",
        proxy_encoder="centroid_mean",
        centroids="kmeans",
    )
    reservoir_supported.update({
        "candidate_reservoirs": "balanced",
        "reservoirs_enabled_queries": 10,
        "last_scan_stats_sample": {
            "multivector_reservoirs_enabled": True,
        },
    })
    reservoir_noop_rec = compute_document_node_serving_recommendation(
        {"results": [reservoir_noop, reservoir_supported]},
        exact_baseline=exact,
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.80,
        max_p95_ms=0.0,
    )
    assert (
        reservoir_noop_rec["best_latency_safe"]["profile"]
        == "centroid_mean_reservoir_supported"
    )
    assert (
        reservoir_noop_rec["best_quality"]["profile"]
        == "centroid_mean_reservoir_supported"
    )
    reservoir_noop_rejections = {
        item["profile"]: item
        for item in reservoir_noop_rec["rejected_profiles"]
    }
    assert "candidate_reservoirs_not_executed" in reservoir_noop_rejections[
        "centroid_mean_reservoir_noop"
    ]["failure_reasons"]
    assert "candidate_reservoirs_not_executed" in reservoir_noop_rejections[
        "centroid_mean_reservoir_noop"
    ]["admission_improvement_hints"]
    assert reservoir_noop_rejections["centroid_mean_reservoir_noop"][
        "unsupported_reservoir_evidence"
    ] is True

    normalized_mean_bottleneck = _synthetic_serving_row(
        "normalized_mean_exhaustive_low_admission",
        p95=80.0,
        admission=0.30,
        ndcg=0.25,
        storage="f16",
        proxy_encoder="normalized_mean",
    )
    normalized_mean_bottleneck.update({
        "serving_exact_rerank_mode": "admission_exhaustive",
        "candidate_budget": 800,
        "effective_exact_rerank_k": 800,
        "exact_rerank_docs_p95": 800.0,
    })
    max_pool_bottleneck = _synthetic_serving_row(
        "max_pool_exhaustive_low_admission",
        p95=82.0,
        admission=0.35,
        ndcg=0.28,
        storage="f16",
        proxy_encoder="max_pool",
    )
    max_pool_bottleneck.update({
        "serving_exact_rerank_mode": "admission_exhaustive",
        "candidate_budget": 800,
        "effective_exact_rerank_k": 800,
        "exact_rerank_docs_p95": 800.0,
    })
    proxy_hint_rec = compute_document_node_serving_recommendation(
        {"results": [normalized_mean_bottleneck, max_pool_bottleneck]},
        exact_baseline=exact,
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    proxy_hint_rejected = {
        item["profile"]: item["admission_improvement_hints"]
        for item in proxy_hint_rec["rejected_profiles"]
    }
    assert (
        "try_max_pool_or_centroid_mean_proxy"
        in proxy_hint_rejected["normalized_mean_exhaustive_low_admission"]
    )
    assert "try_centroid_mean_proxy" in proxy_hint_rejected["max_pool_exhaustive_low_admission"]

    experimental_rows = [
        _synthetic_serving_row(
            "quantized_inverted_experimental_f32",
            p95=5.0,
            admission=0.99,
            ndcg=0.90,
            storage="f32",
            candidate_source="quantized_inverted_experimental",
        ),
        _synthetic_serving_row("valid_non_experimental", p95=30.0, admission=0.85, ndcg=0.87, storage="f16"),
    ]
    experimental_rec = compute_document_node_serving_recommendation(
        {"results": experimental_rows},
        exact_baseline=exact,
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    assert experimental_rec["best_latency_safe"]["profile"] == "valid_non_experimental"
    assert experimental_rec["best_balanced"]["profile"] == "valid_non_experimental"
    experimental_only = compute_document_node_serving_recommendation(
        {"results": [experimental_rows[0]]},
        exact_baseline=exact,
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    assert experimental_only["best_latency_safe"] is None
    assert experimental_only["best_balanced"]["profile"] == "quantized_inverted_experimental_f32"

    tie_rows = [
        _synthetic_serving_row("tie_a", p95=10.0, admission=0.90, ndcg=0.90, storage="f16"),
        _synthetic_serving_row("tie_b", p95=10.0, admission=0.90, ndcg=0.90, storage="f16"),
    ]
    tie = compute_document_node_serving_recommendation(
        {"results": tie_rows},
        exact_baseline=exact,
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    assert tie["best_latency_safe"]["profile"] == "tie_a"
    assert tie["best_balanced"]["profile"] == "tie_a"
    tie_again = compute_document_node_serving_recommendation(
        {"results": tie_rows},
        exact_baseline=exact,
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    assert tie == tie_again


def _self_check_document_node_token_pooling_recommendation() -> None:
    rows = [
        _synthetic_serving_row(
            "proxy_normalized_mean_f16",
            p95=12.0,
            admission=0.84,
            ndcg=0.84,
            token_pooling="off",
            token_pooling_target_ratio=1.0,
        ),
        _synthetic_serving_row(
            "proxy_normalized_mean_f16_pool_075",
            p95=9.0,
            admission=0.79,
            ndcg=0.80,
            token_pooling="greedy_cosine",
            token_pooling_target_ratio=0.75,
        ),
        _synthetic_serving_row(
            "proxy_normalized_mean_f16_pool_050",
            p95=8.0,
            admission=0.83,
            ndcg=0.82,
            token_pooling="greedy_cosine",
            token_pooling_target_ratio=0.5,
        ),
        _synthetic_serving_row(
            "centroid_mean_f16_pool_033",
            p95=20.0,
            admission=0.92,
            ndcg=0.90,
            proxy_encoder="centroid_mean",
            centroids="kmeans",
            token_pooling="greedy_cosine",
            token_pooling_target_ratio=0.33,
        ),
        _synthetic_serving_row(
            "unrelated_profile",
            p95=1.0,
            admission=1.0,
            ndcg=1.0,
        ),
    ]
    rows[0].update({
        "tokens_original_p50": 64.0,
        "tokens_pooled_p50": 64.0,
        "token_pooling_ratio_p50": 1.0,
        "exact_rerank_pairs_p50": 6400.0,
    })
    rows[1].update({
        "tokens_original_p50": 64.0,
        "tokens_pooled_p50": 48.0,
        "token_pooling_ratio_p50": 0.75,
        "exact_rerank_pairs_p50": 4800.0,
    })
    rows[2].update({
        "tokens_original_p50": 64.0,
        "tokens_pooled_p50": 32.0,
        "token_pooling_ratio_p50": 0.5,
        "exact_rerank_pairs_p50": 3200.0,
    })
    rows[3].update({
        "tokens_original_p50": 64.0,
        "tokens_pooled_p50": 21.0,
        "token_pooling_ratio_p50": 0.33,
        "exact_rerank_pairs_p50": 2100.0,
    })
    rec = compute_document_node_token_pooling_recommendation(
        {"results": rows},
        exact_baseline={"available": True, "metrics": {"ndcg@10": 0.90}},
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.90,
        max_p95_ms=0.0,
    )
    assert rec["available"] is True
    assert rec["profile_count"] == 4
    assert rec["best_pooling_latency_safe"]["profile"] == "proxy_normalized_mean_f16_pool_050"
    assert rec["best_pooling_quality_safe"]["profile"] == "centroid_mean_f16_pool_033"
    rejected = {
        item["profile"]: item["failure_reasons"]
        for item in rec["rejected_pooling_profiles"]
    }
    assert "top10_admission_below_threshold" in rejected["proxy_normalized_mean_f16_pool_075"]
    assert rec["best_pooling_latency_safe"]["selection_metrics"]["tokens_pooled_p50"] == 32.0
    no_qrels_rows = [
        {
            **row,
            "ndcg@10": None,
            "recall@10": None,
            "mrr@10": None,
        }
        for row in rows
    ]
    no_qrels = compute_document_node_token_pooling_recommendation(
        {"results": no_qrels_rows},
        exact_baseline={"available": False},
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    assert no_qrels["best_pooling_latency_safe"]["profile"] == "proxy_normalized_mean_f16_pool_050"
    assert no_qrels["best_pooling_quality_safe"]["profile"] == "centroid_mean_f16_pool_033"
    assert (
        no_qrels["best_pooling_quality_safe"]["selection_basis"]
        == "exact_top10_admission_recall"
    )


def _self_check_document_node_serving_recommendation_schema_fixture() -> None:
    fixture_rows = [
        _synthetic_serving_row(
            "proxy_normalized_mean_f16",
            p95=8.0,
            admission=0.70,
            ndcg=0.91,
            storage="f16",
        ),
        _synthetic_serving_row(
            "centroid_mean_f16",
            p95=20.0,
            admission=0.86,
            ndcg=0.91,
            storage="f16",
            proxy_encoder="centroid_mean",
            centroids="kmeans",
        ),
        _synthetic_serving_row(
            "centroid_lite_f16",
            p95=55.0,
            admission=0.93,
            ndcg=0.94,
            storage="f16",
            candidate_source="centroid_lite",
            centroids="kmeans",
        ),
        _synthetic_serving_row(
            "quantized_inverted_experimental_f32",
            p95=6.0,
            admission=0.98,
            ndcg=0.93,
            storage="f32",
            candidate_source="quantized_inverted_experimental",
        ),
    ]
    fixture_rows[1].update({
        "bm25_injection_enabled_queries": 8,
        "bm25_injection_candidates": {"count": 10, "mean": 4.5, "p95": 12.0},
        "bm25_injection_candidate_limit": {"count": 10, "mean": 100.0, "p95": 100.0},
        "bm25_injection_pool_size": {"count": 10, "mean": 840.0, "p95": 900.0},
        "bm25_injection_limit_reason": {"exact_rerank_k": 7, "doc_candidate_k": 3},
        "bm25_injection_retained": {"count": 10, "mean": 3.0, "p95": 9.0},
        "bm25_injection_exact_reranked": {"count": 10, "mean": 4.0, "p95": 12.0},
        "learned_sparse_candidates": {"count": 10, "mean": 0.0, "p95": 0.0},
        "learned_sparse_retained_for_maxsim": {"count": 10, "mean": 0.0, "p95": 0.0},
        "learned_sparse_branch_latency_us": {"count": 10, "mean": 0.0, "p95": 0.0},
    })
    report = {
        "document_node_serving_grid": {
            "enabled": True,
            "results": fixture_rows,
        }
    }
    report["document_node_serving_recommendation"] = compute_document_node_serving_recommendation(
        report["document_node_serving_grid"],
        exact_baseline={"available": True, "metrics": {"ndcg@10": 0.95}},
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    rec = report["document_node_serving_recommendation"]
    assert rec
    assert rec["best_latency_safe"]["profile"] == "centroid_mean_f16"
    assert rec["best_latency_safe"]["bm25_injection_enabled_queries"] == 8
    assert rec["best_latency_safe"]["bm25_injection_limit_reason"] == {
        "exact_rerank_k": 7,
        "doc_candidate_k": 3,
    }
    assert rec["best_quality"]["profile"] == "centroid_lite_f16"
    balanced_profile = rec["best_balanced"]["profile"]
    assert balanced_profile == "centroid_lite_f16"
    repeat = compute_document_node_serving_recommendation(
        report["document_node_serving_grid"],
        exact_baseline={"available": True, "metrics": {"ndcg@10": 0.95}},
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    assert repeat["best_balanced"]["profile"] == balanced_profile
    thresholds = {item["profile"]: item for item in rec["profile_thresholds"]}
    assert thresholds["quantized_inverted_experimental_f32"]["experimental"] is True
    assert thresholds["centroid_mean_f16"]["bm25_injection_pool_size"]["p95"] == 900.0
    assert rec["best_latency_safe"]["profile"] != "quantized_inverted_experimental_f32"
    rejected = {item["profile"]: item["failure_reasons"] for item in rec["rejected_profiles"]}
    assert rejected["proxy_normalized_mean_f16"] == ["top10_admission_below_threshold"]
    markdown = markdown_benchmark_summary(report)
    assert "### Document-node serving recommendation" in markdown
    assert "centroid_mean_f16" in markdown
    assert "quantized_inverted_experimental_f32" in markdown
    assert "bm25_limit_reason_summary" in markdown
    assert "exact_rerank_k" in markdown


def _self_check_document_node_serving_stage_selection() -> None:
    profiles = document_node_serving_profiles(
        include_experimental=True,
        include_proxy_encoder_variants=False,
        include_bm25_rescue=False,
        include_learned_sparse_rescue=False,
        include_reservoirs=False,
        centroid_count="auto",
    )
    profile_by_name = {profile.name: profile for profile in profiles}
    configs = [
        DocumentNodeServingConfig(profile_by_name["proxy_normalized_mean_f16"], 50, 1),
        DocumentNodeServingConfig(profile_by_name["centroid_mean_f16"], 50, 1),
        DocumentNodeServingConfig(profile_by_name["centroid_lite_f16"], 100, 1),
        DocumentNodeServingConfig(profile_by_name["quantized_inverted_experimental_f32"], 50, 1),
    ]
    rows = [
        _synthetic_serving_row(
            "proxy_normalized_mean_f16",
            p95=5.0,
            admission=0.20,
            ndcg=None,
            ef=50,
            oversampling=1,
        ),
        _synthetic_serving_row(
            "centroid_mean_f16",
            p95=12.0,
            admission=0.90,
            ndcg=None,
            ef=50,
            oversampling=1,
            proxy_encoder="centroid_mean",
            centroids="kmeans",
        ),
        _synthetic_serving_row(
            "centroid_lite_f16",
            p95=30.0,
            admission=0.98,
            ndcg=None,
            ef=100,
            oversampling=1,
            candidate_source="centroid_lite",
            centroids="kmeans",
        ),
        _synthetic_serving_row(
            "quantized_inverted_experimental_f32",
            p95=6.0,
            admission=0.99,
            ndcg=None,
            ef=50,
            oversampling=1,
            storage="f32",
            candidate_source="quantized_inverted_experimental",
        ),
    ]
    selected, annotated, pruned = select_document_node_serving_stage2_configs(
        rows,
        configs,
        finalists=2,
    )
    selected_ids = [document_node_serving_config_id(config) for config in selected]
    assert selected_ids == [
        "centroid_lite_f16_ef100_os1",
        "centroid_mean_f16_ef50_os1",
    ]
    assert any(item["stage1_selected_finalist"] for item in annotated)
    pruned_by_id = {item["id"]: item["reason"] for item in pruned}
    assert (
        pruned_by_id["quantized_inverted_experimental_f32_ef50_os1"]
        == "experimental_not_eligible_for_finalist"
    )
    assert pruned_by_id["proxy_normalized_mean_f16_ef50_os1"] == "not_in_top_finalists"
    repeat = select_document_node_serving_stage2_configs(rows, configs, finalists=2)
    assert [document_node_serving_config_id(config) for config in repeat[0]] == selected_ids
    recommendation = compute_document_node_serving_recommendation(
        {"results": [row for row in annotated if row["stage1_selected_finalist"]]},
        exact_baseline={"available": False},
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    assert recommendation["best_latency_safe"]["profile"] == "centroid_mean_f16"


def _self_check_document_node_serving_stats_extraction() -> None:
    def assert_close(actual: Any, expected: float, *, tolerance: float = 0.0001) -> None:
        assert actual is not None
        assert abs(float(actual) - expected) <= tolerance

    nested = normalize_serving_phase_timings({
        "observed_latency_ms": 15.5,
        "dense": {
            "timing_us": {
                "total": 10000,
                "prepare": 1000,
                "traverse": 2000,
                "entry": 10,
                "base": 20,
                "batch": 30,
                "heap": 40,
                "fill": 50,
                "rescore": 300,
                "sort": 60,
                "entry_sidecar": 70,
                "payload_entry_seed": 80,
                "residual_rerank": 90,
                "heap_fetch": 100,
                "heap_rescore": 400,
                "local_expansion": 110,
                "scan_lock_wait": 120,
            }
        },
    })
    assert nested["phase_timing_source"] == "nested_dense_timing_us"
    assert nested["phase_total_time_us"] == 10000
    assert nested["phase_prepare_time_us"] == 1000
    assert nested["document_graph_traversal_time_us"] == 2000
    assert nested["document_graph_batch_time_us"] == 30
    assert nested["exact_heap_fetch_time_us"] == 100
    assert nested["exact_maxsim_rerank_time_us"] == 400
    assert nested["final_sort_time_us"] == 60
    assert nested["sidecar_load_time_us"] is None
    assert "sidecar_load_time_us" in nested["phase_timing_missing"]
    assert_close(nested["phase_timing_unattributed_sql_ms"], 5.5)

    flat = normalize_serving_phase_timings({
        "latency_ms": 8.25,
        "graph_total_us": 6000,
        "graph_prepare_us": 100,
        "graph_traverse_us": 200,
        "graph_entry_us": 20,
        "graph_base_us": 30,
        "graph_batch_us": 40,
        "graph_heap_us": 50,
        "graph_fill_us": 60,
        "graph_rescore_us": 700,
        "graph_sort_us": 80,
        "graph_heap_fetch_us": 90,
    })
    assert flat["phase_timing_source"] == "flat_graph_timing_us"
    assert flat["phase_total_time_us"] == 6000
    assert flat["document_graph_traversal_time_us"] == 200
    assert flat["exact_heap_fetch_time_us"] == 90
    assert flat["exact_maxsim_rerank_time_us"] == 700
    assert_close(flat["phase_timing_unattributed_sql_ms"], 2.25)

    direct = normalize_serving_phase_timings({
        "dense": {"timing_us": {"total": 12000, "heap_rescore": 100}},
        "multivector_document_graph_traversal_time_us": 345,
        "multivector_proxy_candidate_time_us": 234,
        "multivector_proxy_graph_traversal_time_us": 235,
        "multivector_exact_heap_fetch_time_us": 456,
        "multivector_exact_maxsim_rerank_time_us": 567,
        "multivector_final_sort_time_us": 678,
        "multivector_sidecar_load_time_us": 789,
        "multivector_sidecar_page_read_time_us": 790,
        "multivector_sidecar_vector_reconstruct_time_us": 791,
        "multivector_proxy_scoring_time_us": 890,
        "multivector_centroid_lite_time_us": 901,
        "multivector_quantized_inverted_time_us": 123,
    })
    assert direct["phase_timing_source"] == "direct_multivector_timing"
    assert direct["document_graph_traversal_time_us"] == 345
    assert direct["proxy_candidate_time_us"] == 234
    assert direct["proxy_graph_traversal_time_us"] == 235
    assert direct["exact_heap_fetch_time_us"] == 456
    assert direct["exact_maxsim_rerank_time_us"] == 567
    assert direct["final_sort_time_us"] == 678
    assert direct["sidecar_load_time_us"] == 789
    assert direct["sidecar_page_read_time_us"] == 790
    assert direct["sidecar_vector_reconstruct_time_us"] == 791
    assert direct["proxy_scoring_time_us"] == 890
    assert direct["centroid_lite_time_us"] == 901
    assert direct["quantized_inverted_time_us"] == 123

    missing = normalize_serving_phase_timings({})
    assert missing["phase_timing_source"] == "none"
    assert missing["phase_total_time_us"] is None
    assert missing["phase_timing_unattributed_sql_ms"] is None
    assert "proxy_scoring_time_us" in missing["phase_timing_missing"]

    extracted_timing = extract_document_node_serving_stats({
        "observed_latency_ms": 12.0,
        "dense": {"timing_us": {"total": 9000, "heap_rescore": 1234}},
        "multivector_doc_sidecar_pages_read": 42,
        "multivector_sidecar_page_read_time_us": 55,
        "multivector_sidecar_vector_reconstruct_time_us": 66,
    })
    assert extracted_timing["phase_total_time_us"] == 9000
    assert extracted_timing["exact_maxsim_rerank_time_us"] == 1234
    assert extracted_timing["sidecar_load_time_us"] is None
    assert extracted_timing["sidecar_page_read_time_us"] == 55
    assert extracted_timing["sidecar_vector_reconstruct_time_us"] == 66
    assert extracted_timing["multivector_doc_sidecar_pages_read"] == 42

    legacy_cache_stats = {
        "multivector_candidate_source": "proxy_vector",
        "multivector_candidate_path": "proxy_graph",
        "multivector_doc_graph_warning": "document_node_proxy_vector_graph_traversal",
        "multivector_doc_graph_nodes": 10000,
        "multivector_doc_graph_candidates": 100,
        "multivector_doc_graph_search_ef": 100,
        "proxy_candidates": 100,
        "branch_candidate_limits": [800],
        "multivector_sidecar_page_read_time_us": 51000,
        "multivector_sidecar_vector_reconstruct_time_us": 2000,
        "dense": {
            "cache": {
                "multivector_doc_sidecar_cache_mode": "resident",
                "multivector_doc_sidecar_pages_read": 34072,
                "multivector_doc_sidecar_bytes_touched": 257899824,
                "multivector_doc_sidecar_docmap_pages_read": 34072,
                "multivector_doc_sidecar_docmap_bytes_touched": 257899824,
                "multivector_doc_sidecar_resident_vectors_loaded": 10000,
                "multivector_doc_sidecar_resident_bytes_loaded": 253653888,
                "multivector_doc_sidecar_vectors_loaded": 0,
                "native_cache_used": True,
                "native_cache_reused": True,
                "native_cache_built_this_scan": False,
                "native_cache_bytes": 314572800,
                "native_cache_exact_bytes": 253653888,
            },
        },
    }
    legacy_extracted = extract_document_node_serving_stats(legacy_cache_stats)
    assert legacy_extracted["native_cache_reused"] is True
    assert legacy_extracted["sidecar_cache_build_this_query"] is False
    assert legacy_extracted["sidecar_cache_build_bytes"] == 0
    assert legacy_extracted["sidecar_query_bytes_touched"] == 257899824
    assert legacy_extracted["sidecar_query_pages_read"] == 34072
    assert legacy_extracted["sidecar_query_vectors_loaded"] == 10000
    assert legacy_extracted["sidecar_query_time_us"] == 53000
    assert legacy_extracted["proxy_vector_uses_full_sidecar_for_graph"] is False
    assert legacy_extracted["proxy_full_sidecar_vectors_loaded"] == 0
    assert legacy_extracted["proxy_candidate_limit_effective"] == 100
    assert legacy_extracted["proxy_candidate_limit_source"] == "search_ef"
    assert legacy_extracted["proxy_vector_near_exhaustive_sidecar_touch"] is True
    assert (
        legacy_extracted["proxy_vector_sidecar_touch_reason"]
        == "resident_cache_query_materialize_all_docs"
    )

    partial_stats = {
        "multivector_candidate_source": "proxy_vector",
        "multivector_doc_graph_warning": "document_node_proxy_vector_graph_traversal",
        "multivector_doc_graph_nodes": 1000,
        "multivector_doc_graph_exact_rerank_docs": 400,
        "multivector_exact_rerank_docs": 400,
        "multivector_exact_rerank_pairs": 123456,
        "multivector_exact_kernel": "blocked_neon",
        "proxy_encoder_kind": "normalized_mean",
        "proxy_graph_nodes_visited": 90,
        "proxy_graph_edges_visited": 91,
        "proxy_graph_candidates_seen": 92,
        "proxy_candidates_returned": 93,
        "proxy_candidate_limit_effective": 100,
        "proxy_candidate_limit_source": "search_ef",
        "proxy_vector_scores_computed": 94,
        "proxy_vector_score_time_us": 95,
        "proxy_candidates": 800,
        "proxy_top1_admission": True,
        "proxy_exact_rerank_docs": 100,
        "proxy_full_sidecar_vectors_loaded": 0,
        "proxy_full_sidecar_bytes_touched": 0,
        "proxy_full_sidecar_pages_read": 0,
        "proxy_full_sidecar_load_time_us": 0,
        "proxy_full_sidecar_reconstruct_time_us": 0,
        "proxy_exact_rerank_heap_fetches": 0,
        "proxy_exact_rerank_sidecar_fetches": 100,
        "proxy_exact_rerank_bytes_touched": 65536,
        "proxy_exact_rerank_time_us": 1234,
        "sidecar_cache_build_this_query": True,
        "sidecar_cache_build_bytes": 4096,
        "sidecar_cache_build_pages_read": 1,
        "sidecar_cache_build_time_us": 77,
        "sidecar_query_bytes_touched": 2048,
        "sidecar_query_pages_read": 3,
        "sidecar_query_vectors_loaded": 2,
        "sidecar_query_load_time_us": 88,
        "sidecar_query_time_us": 88,
        "proxy_vector_uses_full_sidecar_for_graph": False,
        "proxy_vector_near_exhaustive_sidecar_touch": False,
        "proxy_vector_sidecar_touch_reason": "none",
        "multivector_doc_sidecar_cache_mode": "resident",
        "multivector_doc_sidecar_cache_hits": 20,
        "multivector_doc_sidecar_docmap_bytes_touched": 4096,
        "multivector_doc_sidecar_resident_vectors_loaded": 8,
        "multivector_doc_sidecar_resident_bytes_loaded": 2048,
        "centroid_lists_visited": 16,
        "centroid_docs_touched": 700,
        "centroid_pruned_docs": 600,
        "centroid_postings_touched": 1200,
        "centroid_postings_skipped": 300,
        "centroid_posting_limit_per_token": 75,
        "centroid_posting_cap_strategy": "uniform_stride",
        "centroid_candidates": 100,
        "centroid_bitset_prefilter_enabled": True,
        "centroid_bitset_lists_used": 8,
        "centroid_bitset_docs_set": 101,
        "centroid_bitset_docs_after_threshold": 99,
        "centroid_bitset_prefilter_time_us": 44,
        "centroid_bitset_memory_bytes": 128,
        "multivector_tokens_original": 6400,
        "multivector_tokens_pooled": 3200,
        "multivector_bm25_injection_enabled": False,
        "multivector_bm25_injection_candidates": 140,
        "multivector_bm25_injection_candidate_limit": 100,
        "multivector_bm25_injection_pool_size": 900,
        "multivector_bm25_injection_limit_reason": "exact_rerank_k",
        "multivector_bm25_injection_retained": 32,
        "multivector_bm25_injection_exact_reranked": 140,
        "learned_sparse_candidates": 0,
        "quantized_inverted_postings_touched": 0,
        "quantized_inverted_posting_bytes": 2048,
        "quantized_inverted_sidecar_bytes": 2304,
        "quantized_inverted_compact_kernel": "neon",
        "quantized_inverted_compact_score_us": 321,
        "quantized_inverted_compact_docs_scored": 12,
        "quantized_inverted_compact_payload_bytes": 64,
        "quantized_inverted_compact_topk_changed_vs_scalar": False,
        "quantized_codeword_debug_counter": 7,
        "unrelated_large_field": "ignored",
    }
    extracted = extract_document_node_serving_stats(partial_stats)
    assert extracted["multivector_candidate_source"] == "proxy_vector"
    assert extracted["proxy_graph_nodes_visited"] == 90
    assert extracted["proxy_graph_edges_visited"] == 91
    assert extracted["proxy_graph_candidates_seen"] == 92
    assert extracted["proxy_candidates_returned"] == 93
    assert extracted["proxy_candidate_limit_effective"] == 100
    assert extracted["proxy_candidate_limit_source"] == "search_ef"
    assert extracted["proxy_vector_scores_computed"] == 94
    assert extracted["proxy_vector_score_time_us"] == 95
    assert extracted["proxy_candidates"] == 800
    assert extracted["proxy_exact_rerank_sidecar_fetches"] == 100
    assert extracted["proxy_exact_rerank_bytes_touched"] == 65536
    assert extracted["sidecar_cache_build_this_query"] is True
    assert extracted["sidecar_cache_build_pages_read"] == 1
    assert extracted["multivector_bm25_injection_candidate_limit"] == 100
    assert extracted["multivector_bm25_injection_pool_size"] == 900
    assert extracted["multivector_bm25_injection_limit_reason"] == "exact_rerank_k"
    assert extracted["multivector_bm25_injection_exact_reranked"] == 140
    assert extracted["sidecar_query_vectors_loaded"] == 2
    assert extracted["sidecar_query_time_us"] == 88
    assert extracted["proxy_vector_uses_full_sidecar_for_graph"] is False
    assert extracted["proxy_vector_near_exhaustive_sidecar_touch"] is False
    assert extracted["multivector_doc_sidecar_cache_hits"] == 20
    assert extracted["multivector_doc_sidecar_docmap_bytes_touched"] == 4096
    assert extracted["multivector_doc_sidecar_resident_vectors_loaded"] == 8
    assert extracted["centroid_postings_touched"] == 1200
    assert extracted["centroid_postings_skipped"] == 300
    assert extracted["centroid_posting_limit_per_token"] == 75
    assert extracted["centroid_posting_cap_strategy"] == "uniform_stride"
    assert extracted["centroid_bitset_prefilter_enabled"] is True
    assert extracted["centroid_bitset_lists_used"] == 8
    assert extracted["centroid_bitset_docs_set"] == 101
    assert extracted["centroid_bitset_docs_after_threshold"] == 99
    assert extracted["centroid_bitset_prefilter_time_us"] == 44
    assert extracted["centroid_bitset_memory_bytes"] == 128
    assert extracted["quantized_inverted_posting_bytes"] == 2048
    assert extracted["quantized_inverted_sidecar_bytes"] == 2304
    assert extracted["quantized_inverted_compact_kernel"] == "neon"
    assert extracted["quantized_inverted_compact_score_us"] == 321
    assert extracted["quantized_inverted_compact_docs_scored"] == 12
    assert extracted["quantized_inverted_compact_payload_bytes"] == 64
    assert extracted["quantized_inverted_compact_topk_changed_vs_scalar"] is False
    assert extracted["quantized_codeword_debug_counter"] == 7
    assert "unrelated_large_field" not in extracted
    available = document_node_serving_stats_available(extracted)
    assert available["core"] is True
    assert available["proxy"] is True
    assert available["centroid_lite"] is True
    assert available["storage_cache"] is True
    assert available["pooling"] is True
    assert available["sparse_bm25_rescue"] is True
    assert available["experimental_quantized"] is True
    assert document_node_serving_stats_available({})["core"] is False

    admission = {
        "per_query": [
            {
                "query_id": "q1",
                "budgets": [
                    {"budget": 50, "scan_stats": {"multivector_candidate_source": "proxy_vector"}},
                    {"budget": 800, "scan_stats": partial_stats},
                ],
            }
        ]
    }
    by_budget = serving_grid_scan_stats_by_budget(admission)
    assert by_budget["50"]["scan_stats_sample"]["multivector_candidate_source"] == "proxy_vector"
    assert by_budget["800"]["scan_stats_sample"]["multivector_exact_kernel"] == "blocked_neon"
    assert serving_grid_scan_stats_sample(admission, 800)["proxy_encoder_kind"] == "normalized_mean"
    assert serving_grid_scan_stats_sample({"per_query": []}, 800) == {}

    row = _synthetic_serving_row(
        "stats_roundtrip",
        p95=12.0,
        admission=0.91,
        ndcg=0.87,
        stats_sample=by_budget["800"]["scan_stats_sample"],
    )
    round_trip = json.loads(json.dumps(row))
    assert round_trip["last_scan_stats_sample"]["multivector_exact_kernel"] == "blocked_neon"
    rec = compute_document_node_serving_recommendation(
        {"summary_rows": [row]},
        exact_baseline={"available": True, "metrics": {"ndcg@10": 0.90}},
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    markdown = markdown_benchmark_summary({
        "document_node_serving_recommendation": rec,
    })
    assert "Why latency-safe won" in markdown
    assert "blocked_neon" in markdown
    assert "document_node_proxy_vector_graph_traversal" in markdown


def _self_check_document_node_serving_grid_serialization() -> None:
    profile = DocumentNodeServingProfile(
        name="proxy_normalized_mean_f16",
        candidate_source="proxy_vector",
        proxy_encoder="normalized_mean",
        storage_kind="f16",
    )
    args = argparse.Namespace(
        final_k=10,
        quality_k=10,
        multivector_token_pooling_min_tokens=16,
        admission_budget_sweep="50,100,200,400,800",
        admission_budget_sweep_explicit=True,
        document_node_serving_grid=True,
        document_node_serving_grid_smoke=False,
        document_node_serving_grid_budget_mode="largest_only",
        serving_exact_rerank_mode="serving",
        serving_exact_rerank_k=100,
        admission_debug_mode=None,
    )
    admission = {
        "aggregate": {
            "budget_sweep": [50, 100, 200, 400, 800],
            "executed_budgets": [800],
            "executed_budget_count": 1,
            "serving_exact_rerank_mode": "serving",
            "admission_debug_mode": "summary",
            "trace_enabled": False,
            "admission_evidence_mode": "result_doc",
            "trace_entries": 0,
            "requested_serving_exact_rerank_k": 100,
            "effective_exact_rerank_by_budget": {"800": 100},
            "total_elapsed_ms": 123.0,
            "exact_baseline_elapsed_ms_total": 23.0,
            "retrieval_elapsed_ms_total": 100.0,
            "exact_baseline_query_count": 1,
            "retrieval_query_count": 5,
            "effective_budget_count": 1,
            "effective_budgets": [800],
            "latency_by_budget": {
                "800": {"p50_ms": 12.5, "p95_ms": 21.0, "p99_ms": 25.0},
            },
            "admission_by_budget": {
                "800": {
                    "serving_exact_rerank_mode": "serving",
                    "admission_debug_mode": "summary",
                    "trace_enabled": False,
                    "admission_evidence_mode": "result_doc",
                    "trace_entries": 0,
                    "requested_serving_exact_rerank_k": 100,
                    "effective_exact_rerank_k": 100,
                    "exact_rerank_docs": {"p50": 100, "p95": 100},
                    "reservoirs_enabled_queries": 0,
                    "reservoir_union_docs": {"count": 1, "mean": 0.0, "p95": 0.0},
                },
            },
            "exact_top1_admission_rate": 0.75,
            "exact_top10_admission_recall": 0.92,
            "run_at_largest_budget": {},
        },
        "per_query": [
            {
                "query_id": "q1",
                "budgets": [
                    {
                        "budget": 800,
                        "scan_stats": {
                            "multivector_candidate_source": "proxy_vector",
                            "proxy_encoder_kind": "normalized_mean",
                            "multivector_exact_kernel": "blocked_scalar",
                            "multivector_proxy_candidates_returned": 777,
                            "multivector_doc_graph_entry_sample_scored": 32,
                            "graph_entry_sidecar_scored": 128,
                            "graph_entry_sidecar_selected": 8,
                            "multivector_document_graph_traversal_time_us": 1234,
                            "multivector_exact_maxsim_rerank_time_us": 5678,
                            "sidecar_query_bytes_touched": 65536,
                            "sidecar_query_time_us": 9876,
                        },
                    },
                ],
            }
        ],
    }
    row = document_node_serving_summary_row(
        profile=profile,
        args=args,
        ef=100,
        oversampling=2,
        index_phase={
            "index_bytes": 4096,
            "index_stats": {"multivector_proxy_encoder": "normalized_mean"},
        },
        admission=admission,
        qrels={},
    )
    assert row["graph_mode"] == "document_nodes"
    assert row["storage_cache_mode"] == "auto"
    assert row["centroid_lite_max_postings_per_token"] == 0
    assert row["p50_ms"] == 12.5
    assert row["p95_ms"] == 21.0
    assert row["p99_ms"] == 25.0
    assert row["run_elapsed_ms"] == 123.0
    assert row["exact_baseline_elapsed_ms"] == 23.0
    assert row["retrieval_elapsed_ms"] == 100.0
    assert row["retrieval_query_count"] == 5
    assert row["exact_baseline_query_count"] == 1
    assert row["budget_sweep"] == [50, 100, 200, 400, 800]
    assert row["executed_budget_count"] == 1
    assert row["executed_budgets"] == [800]
    assert row["effective_budget_count"] == 1
    assert row["effective_budgets"] == [800]
    assert row["largest_budget"] == 800
    assert row["candidate_budget"] == 800
    assert row["serving_exact_rerank_mode"] == "serving"
    assert row["admission_debug_mode"] == "summary"
    assert row["trace_enabled"] is False
    assert row["admission_evidence_mode"] == "result_doc"
    assert row["trace_entries"] == 0
    assert row["requested_serving_exact_rerank_k"] == 100
    assert row["effective_exact_rerank_k"] == 100
    assert row["exact_rerank_docs_p50"] == 100
    assert row["exact_rerank_docs_p95"] == 100
    assert row["candidate_reservoirs"] == "off"
    assert row["reservoirs_enabled_queries"] == 0
    assert row["reservoir_union_docs"]["p95"] == 0.0
    assert row["proxy_candidates_returned"] == 777
    assert row["graph_entry_sample_scored"] == 32
    assert row["graph_entry_sidecar_scored"] == 128
    assert row["graph_entry_sidecar_selected"] == 8
    assert row["graph_traversal_time_us"] == 1234
    assert row["exact_rerank_time_us"] == 5678
    assert row["sidecar_query_bytes_touched"] == 65536
    assert row["sidecar_query_time_us"] == 9876
    assert row["last_scan_stats_sample"]["proxy_encoder_kind"] == "normalized_mean"
    rec = compute_document_node_serving_recommendation(
        {"summary_rows": [row]},
        exact_baseline={"available": False},
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    assert rec["best_latency_safe"]["profile"] == "proxy_normalized_mean_f16"

    grid = {
        "document_node_serving_grid": {
            "enabled": True,
            "mode": "smoke",
            "stage_mode": "two_stage",
            "probe_queries": 10,
            "finalists": 6,
            "smoke_only": True,
            "serving_evidence": False,
            "budget_mode": "largest_only",
            "serving_exact_rerank_mode": "serving",
            "requested_serving_exact_rerank_k": 100,
            "admission_debug_mode": "summary",
            "trace_enabled": False,
            "include_centroid_lite_caps": False,
            "effective_centroid_lite_posting_caps": [16, 32, 64],
            "total_elapsed_ms": 130.0,
            "index_build_elapsed_ms_total": 7.0,
            "exact_baseline_elapsed_ms_total": 23.0,
            "retrieval_elapsed_ms_total": 100.0,
            "recommendation_elapsed_ms": 0.5,
            "profiles_run": 1,
            "index_builds": 1,
            "index_build_group_count": 1,
            "index_reuse_count": 0,
            "index_build_reused_for_profiles": [
                {
                    "index_signature": [["reloptions", ["multivector_graph = document_nodes"]]],
                    "profiles": ["proxy_normalized_mean_f16"],
                }
            ],
            "ef_values_run": [100],
            "oversampling_values_run": [2],
            "budgets_run": [800],
            "queries_run": 1,
            "requested_budget_sweep": [50, 100, 200, 400, 800],
            "budget_sweep": [50, 100, 200, 400, 800],
            "effective_budget_sweep": [50, 100, 200, 400, 800],
            "executed_budgets": [800],
            "executed_budget_count": 1,
            "effective_profiles": ["proxy_normalized_mean_f16"],
            "effective_ef_grid": [100],
            "effective_oversampling_grid": [2],
            "query_subset_used": True,
            "queries_available": 42,
            "query_subset_limit": 25,
            "retrieval_query_count": 5,
            "exact_baseline_query_count": 1,
            "exact_top_cache_enabled": True,
            "exact_top_cache_hits": 5,
            "exact_top_cache_misses": 1,
            "exact_top_cache_entries": 1,
            "exact_top_cache_loaded_document_count": 1000,
            "exact_top_cache_metadata": {
                "loaded_document_count": 1000,
                "key_fields": ["admission_k", "query_id"],
                "scope": "document_node_serving_grid_run",
            },
            "exact_top_cache_verified": False,
            "stage1_results": [
                {
                    **row,
                    "stage": "stage1_probe",
                    "stage1_score": 99.0,
                    "stage1_rank": 1,
                    "stage1_selected_finalist": True,
                }
            ],
            "stage2_results": [{**row, "stage": "stage2_finalists"}],
            "pruned_configs": [
                {
                    "id": "proxy_normalized_mean_sq8_ef50_os1",
                    "profile": "proxy_normalized_mean_sq8",
                    "reason": "not_in_top_finalists",
                }
            ],
            "profiles": [profile.__dict__],
            "profile_summaries": [
                {
                    "profile": "proxy_normalized_mean_f16",
                    "profile_elapsed_ms": 130.0,
                    "index_build_elapsed_ms": 7.0,
                    "admission_debug_elapsed_ms_total": 123.0,
                    "exact_baseline_elapsed_ms_total": 23.0,
                    "retrieval_elapsed_ms_total": 100.0,
                    "retrieval_query_count": 5,
                    "exact_baseline_query_count": 1,
                    "serving_exact_rerank_mode": "serving",
                    "requested_serving_exact_rerank_k": 100,
                    "admission_debug_mode": "summary",
                    "index_build_reused": False,
                    "index_build_reused_for_profiles": ["proxy_normalized_mean_f16"],
                    "runs": 1,
                }
            ],
            "results": [row],
        }
    }
    serialized = json.loads(json.dumps(grid))
    assert serialized["document_node_serving_grid"]["results"][0]["profile"] == "proxy_normalized_mean_f16"
    assert serialized["document_node_serving_grid"]["stage_mode"] == "two_stage"
    assert serialized["document_node_serving_grid"]["stage2_results"][0]["profile"] == "proxy_normalized_mean_f16"
    markdown = markdown_benchmark_summary(serialized)
    assert "### Document-node serving grid" in markdown
    assert "### Serving grid cost breakdown" in markdown
    assert "proxy_normalized_mean_f16" in markdown
    assert "21.000" in markdown
    assert "130.000 ms" in markdown
    assert "smoke-only" in markdown
    assert "Exact top cache" in markdown
    assert "Index build groups" in markdown
    assert "Budget mode" in markdown
    assert "Requested budget sweep" in markdown
    assert "Executed budgets" in markdown
    assert "Exact rerank mode" in markdown
    assert "Admission debug mode" in markdown
    assert "Stage mode" in markdown
    assert "Two-stage mode probes" in markdown
    assert "candidate budget" in markdown
    assert "exact rerank k" in markdown

    base_args = argparse.Namespace(
        admission_budget_sweep="25,50",
        admission_budget_sweep_explicit=True,
        document_node_serving_grid=True,
        document_node_serving_grid_smoke=False,
        document_node_serving_grid_budget_mode="largest_only",
        serving_exact_rerank_mode="serving",
        serving_exact_rerank_k=100,
    )
    profile_args = document_node_serving_profile_args(
        base_args,
        profile,
        ef=50,
        oversampling=1,
    )
    assert profile_args.admission_budget_sweep == "25,50"
    reservoir_profile = DocumentNodeServingProfile(
        name="proxy_normalized_mean_f16_reservoir_balanced",
        candidate_source="proxy_vector",
        proxy_encoder="normalized_mean",
        storage_kind="f16",
        candidate_reservoirs="balanced",
        per_token_doc_reservoir_k=2,
        coverage_reservoir_k=20,
    )
    reservoir_profile_args = document_node_serving_profile_args(
        base_args,
        reservoir_profile,
        ef=50,
        oversampling=1,
    )
    assert reservoir_profile_args.multivector_candidate_reservoirs == "balanced"
    assert reservoir_profile_args.multivector_per_token_doc_reservoir_k == 2
    assert reservoir_profile_args.multivector_coverage_reservoir_k == 20


def _self_check_candidate_underfill_diagnostics() -> None:
    cases = [
        (
            {
                "candidate_budget": 800,
                "ef": 100,
                "last_scan_stats_sample": {
                    "proxy_candidates_returned": 100,
                    "proxy_candidate_limit_source": "search_ef",
                },
            },
            "capped_by_search_ef",
            "try_higher_ef_or_entry_sample",
        ),
        (
            {
                "candidate_budget": 800,
                "final_k": 10,
                "oversampling": 2,
                "last_scan_stats_sample": {
                    "proxy_candidates_returned": 20,
                    "multivector_proxy_candidate_target": 20,
                },
            },
            "capped_by_final_k_oversampling",
            "increase_oversampling",
        ),
        (
            {
                "candidate_budget": 800,
                "multivector_doc_candidate_k": 200,
                "last_scan_stats_sample": {"proxy_candidates_returned": 200},
            },
            "capped_by_doc_candidate_k",
            "increase_multivector_doc_candidate_k",
        ),
        (
            {
                "candidate_budget": 800,
                "ef": 1000,
                "multivector_doc_candidate_k": 800,
                "last_scan_stats_sample": {
                    "proxy_candidates_returned": 300,
                    "multivector_proxy_candidate_target": 800,
                },
            },
            "candidate_source_underfilled",
            "try_entry_sidecar_or_stronger_proxy",
        ),
        (
            {
                "candidate_budget": 800,
                "effective_exact_rerank_k": 100,
                "last_scan_stats_sample": {"proxy_candidates_returned": 800},
            },
            "exact_rerank_k_limits_rescue",
            "increase_serving_exact_rerank_k_or_candidate_reservoir",
        ),
        (
            {"candidate_budget": 800, "last_scan_stats_sample": {}},
            "unknown",
            "inspect_candidate_limit_stats",
        ),
    ]
    for row, reason, hint in cases:
        diagnostics = derive_candidate_underfill_diagnostics(row)
        assert diagnostics["candidate_underfill_reason"] == reason
        assert diagnostics["next_admission_hint"] == hint

    ratio = derive_candidate_underfill_diagnostics({
        "candidate_budget": 800,
        "last_scan_stats_sample": {"proxy_candidates_returned": 100},
    })["candidate_underfill_ratio"]
    assert ratio == 0.125

    underfill = derive_candidate_underfill_diagnostics({
        "candidate_budget": 800,
        "last_scan_stats_sample": {
            "proxy_candidates_returned": 100,
            "proxy_candidate_limit_source": "search_ef",
        },
    })
    rejected_row = {
        **_synthetic_serving_row(
            "underfilled",
            p95=10.0,
            admission=0.40,
            ndcg=0.40,
            stats_sample={
                "proxy_candidates_returned": 100,
                "proxy_candidate_limit_source": "search_ef",
            },
        ),
        **underfill,
    }
    rec = compute_document_node_serving_recommendation(
        {"summary_rows": [rejected_row]},
        exact_baseline={"available": False},
        min_top10_admission=0.80,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    rejected = rec["rejected_profiles"][0]
    assert "candidate_underfill" in rejected["failure_reasons"]
    assert "try_higher_ef_or_entry_sample" in rejected["admission_improvement_hints"]


def _self_check_document_node_candidate_source_deltas() -> None:
    def row(
        profile: str,
        *,
        top10: float,
        p95: float,
        ndcg: float,
        source: str = "proxy_vector",
        proxy: str = "normalized_mean",
    ) -> dict[str, Any]:
        return {
            "stage": "single",
            "profile": profile,
            "candidate_source": source,
            "proxy_encoder": proxy,
            "centroids": "off",
            "storage_kind": "f16",
            "storage_cache_mode": "auto",
            "ef": 50,
            "oversampling": 1,
            "largest_budget": 800,
            "candidate_budget": 800,
            "exact_top1_admission_rate": top10 / 2.0,
            "exact_top10_admission_recall": top10,
            "ndcg@10": ndcg,
            "p50_ms": p95 / 2.0,
            "p95_ms": p95,
        }

    rows = [
        row("proxy_normalized_mean_f16", top10=0.42, p95=20.0, ndcg=0.40),
        {
            **row(
                "proxy_normalized_mean_f16_entry_sample_032",
                top10=0.50,
                p95=21.0,
                ndcg=0.45,
            ),
            "entry_sample_count": 32,
            "effective_entry_sample_count": 32,
            "last_scan_stats_sample": {
                "multivector_doc_graph_entry_sample_effective": 32,
                "multivector_doc_graph_entry_sample_scored": 32,
            },
        },
        row(
            "proxy_max_pool_f16",
            top10=0.66,
            p95=18.0,
            ndcg=0.62,
            proxy="max_pool",
        ),
        row(
            "proxy_normalized_mean_f16_entry_sidecar",
            top10=0.42,
            p95=22.0,
            ndcg=0.40,
        ) | {
            "entry_sidecar": True,
            "last_scan_stats_sample": {
                "graph_entry_sidecar_count": 128,
                "graph_entry_sidecar_scored": 115,
                "graph_entry_sidecar_selected": 8,
            },
        },
        row(
            "proxy_max_pool_f16_bm25_rescue",
            top10=0.72,
            p95=24.0,
            ndcg=0.68,
            proxy="max_pool",
        ),
        row(
            "centroid_mean_f16",
            top10=0.54,
            p95=25.0,
            ndcg=0.55,
            proxy="centroid_mean",
        ),
        {
            **row(
                "centroid_lite_f16",
                top10=0.34,
                p95=26.0,
                ndcg=0.25,
                source="centroid_lite",
            ),
            "centroid_docs_touched": {"count": 1, "p95": 900.0},
            "centroid_postings_touched": {"count": 1, "p95": 5000.0},
            "centroid_postings_skipped": {"count": 1, "p95": 0.0},
            "centroid_candidates": {"count": 1, "p95": 800.0},
        },
        {
            **row(
                "centroid_mean_f16_reservoir_balanced",
                top10=0.90,
                p95=19.0,
                ndcg=0.88,
                proxy="centroid_mean",
            ),
            "candidate_reservoirs": "balanced",
            "reservoirs_enabled_queries": 0,
            "reservoir_union_docs": {"count": 5, "p95": 0.0},
            "reservoir_duplicates": {"count": 5, "p95": 0.0},
            "last_scan_stats_sample": {
                "multivector_reservoirs_enabled": False,
            },
        },
        row(
            "centroid_lite_f16_cap_032",
            top10=0.30,
            p95=28.0,
            ndcg=0.20,
            source="centroid_lite",
        )
        | {
            "centroid_lite_warnings": ["centroid_lite_cap_too_aggressive"],
            "centroid_lite_max_postings_per_token": 32,
            "centroid_docs_touched": {"count": 1, "p95": 180.0},
            "centroid_postings_touched": {"count": 1, "p95": 1200.0},
            "centroid_postings_skipped": {"count": 1, "p95": 3800.0},
            "centroid_posting_limit_per_token": {"count": 1, "p95": 32.0},
            "centroid_posting_cap_strategy": {"count": 1, "values": ["uniform_stride"]},
            "centroid_candidates": {"count": 1, "p95": 100.0},
        },
    ]
    deltas = compute_document_node_candidate_source_deltas(rows)
    assert deltas["comparison_count"] == 8
    assert deltas["missing_baseline_count"] == 0
    summary = document_node_candidate_source_delta_summary(deltas)
    assert summary["available"] is True
    assert summary["usable_comparison_count"] == 7
    assert summary["unusable_comparison_count"] == 1
    assert summary["best_admission_delta"]["profile"] == "proxy_max_pool_f16"
    assert summary["best_quality_delta"]["profile"] == "proxy_max_pool_f16"
    assert summary["best_latency_delta"]["profile"] == "proxy_max_pool_f16"
    assert (
        summary["best_admission_delta_by_comparison"]["bm25_rescue"]["profile"]
        == "proxy_max_pool_f16_bm25_rescue"
    )
    assert (
        summary["best_admission_delta_by_comparison"]["entry_sample"]["profile"]
        == "proxy_normalized_mean_f16_entry_sample_032"
    )
    by_profile = {
        item["profile"]: item
        for item in deltas["rows"]
    }
    def close(value: Any, expected: float) -> bool:
        return abs(float(value) - expected) < 1e-9

    assert by_profile["proxy_max_pool_f16"]["baseline_profile"] == "proxy_normalized_mean_f16"
    assert by_profile["proxy_max_pool_f16"]["comparison"] == "proxy_encoder_variant"
    assert close(by_profile["proxy_max_pool_f16"]["exact_top10_admission_recall_delta"], 0.24)
    assert close(by_profile["proxy_max_pool_f16"]["p95_ms_delta"], -2.0)
    assert by_profile["proxy_normalized_mean_f16_entry_sample_032"]["comparison"] == "entry_sample"
    assert by_profile["proxy_normalized_mean_f16_entry_sample_032"]["entry_sample_count"] == 32
    assert (
        by_profile["proxy_normalized_mean_f16_entry_sample_032"]["candidate_evidence"][
            "graph_entry_sample_scored"
        ]
        == 32
    )
    assert close(
        by_profile[
            "proxy_normalized_mean_f16_entry_sample_032"
        ]["exact_top10_admission_recall_delta"],
        0.08,
    )
    assert by_profile["proxy_normalized_mean_f16_entry_sidecar"]["comparison"] == "entry_sidecar"
    assert (
        by_profile["proxy_normalized_mean_f16_entry_sidecar"]["candidate_evidence"][
            "graph_entry_sidecar_selected"
        ]
        == 8
    )
    assert close(by_profile["proxy_normalized_mean_f16_entry_sidecar"]["p95_ms_delta"], 2.0)
    assert by_profile["proxy_max_pool_f16_bm25_rescue"]["baseline_profile"] == "proxy_max_pool_f16"
    assert by_profile["proxy_max_pool_f16_bm25_rescue"]["comparison"] == "bm25_rescue"
    assert by_profile["centroid_mean_f16"]["comparison"] == "proxy_encoder_variant"
    assert by_profile["centroid_mean_f16_reservoir_balanced"]["evidence_usable"] is False
    assert (
        by_profile["centroid_mean_f16_reservoir_balanced"]["candidate_evidence"][
            "reservoir_union_docs"
        ]["p95"]
        == 0.0
    )
    assert (
        "candidate_reservoirs_not_executed"
        in by_profile["centroid_mean_f16_reservoir_balanced"]["evidence_warnings"]
    )
    assert (
        summary["best_admission_delta_by_comparison"]["candidate_reservoir"]
        is None
    )
    centroid_cap_delta = by_profile["centroid_lite_f16_cap_032"]
    assert centroid_cap_delta["comparison"] == "centroid_lite_posting_cap"
    assert "centroid_lite_cap_too_aggressive" in centroid_cap_delta["evidence_warnings"]
    assert "centroid_lite_no_admission_gain" in centroid_cap_delta["evidence_warnings"]
    assert "centroid_lite_latency_regression" in centroid_cap_delta["evidence_warnings"]
    assert (
        centroid_cap_delta["candidate_evidence"]["centroid_posting_limit_per_token"][
            "p95"
        ]
        == 32.0
    )

    grid = {
        "document_node_serving_grid": {
            "enabled": True,
            "mode": "full",
            "stage_mode": "single",
            "probe_queries": 10,
            "finalists": 6,
            "smoke_only": False,
            "serving_evidence": True,
            "budget_mode": "largest_only",
            "serving_exact_rerank_mode": "serving",
            "requested_serving_exact_rerank_k": 100,
            "admission_debug_mode": "summary",
            "trace_enabled": False,
            "effective_profiles": [item["profile"] for item in rows],
            "effective_ef_grid": [50],
            "effective_oversampling_grid": [1],
            "effective_budget_sweep": [800],
            "executed_budgets": [800],
            "queries_run": 1,
            "queries_available": 1,
            "results": rows,
            "candidate_source_deltas": deltas,
        }
    }
    grid["document_node_serving_recommendation"] = (
        compute_document_node_serving_recommendation(
            grid["document_node_serving_grid"],
            exact_baseline={
                "available": True,
                "metrics": {"ndcg@10": 0.75},
                "method": EXACT_SCAN_METHOD,
            },
            min_top10_admission=0.8,
            min_ndcg_ratio_vs_exact=0.95,
            max_p95_ms=0.0,
        )
    )
    markdown = markdown_benchmark_summary(grid)
    assert "#### Candidate-source deltas" in markdown
    assert "proxy_max_pool_f16_bm25_rescue" in markdown
    assert "entry_sample=32/scored=32" in markdown
    assert "entry_sidecar/scored=115/selected=8" in markdown
    assert "reservoirs=balanced/enabled=0/union_p95=0.0" in markdown
    assert "candidate_reservoirs_not_executed" in markdown
    assert "#### Candidate-source delta summary" in markdown
    assert "0.240000" in markdown


def _self_check_document_node_serving_build_only_serialization() -> None:
    profile = document_node_serving_profiles(
        include_experimental=False,
        include_proxy_encoder_variants=False,
        include_bm25_rescue=False,
        include_learned_sparse_rescue=False,
        include_reservoirs=False,
        centroid_count="auto",
    )[2]
    signature = (
        ("multivector_centroids", "kmeans"),
        ("multivector_proxy_encoder", "centroid_mean"),
    )
    index_phase = {
        "elapsed_ms": 1234.5,
        "index_bytes": 987654,
        "index_stats": {
            "multivector_graph_mode": "document_nodes",
            "multivector_doc_build_scorer": "proxy",
            "multivector_proxy_encoder": "centroid_mean",
            "multivector_centroids": "kmeans",
        },
        "build_stats": {
            "multivector_centroid_build_us": 100000,
            "multivector_centroid_cluster_us": 70000,
            "multivector_centroid_residual_us": 20000,
            "multivector_centroid_build_docs": 10,
            "multivector_centroid_build_vectors": 80,
            "multivector_proxy_build_us": 20000,
            "multivector_doc_sidecar_write_us": 30000,
            "multivector_centroid_sidecar_write_us": 40000,
            "multivector_centroid_posting_write_us": 50000,
            "multivector_centroid_posting_count": 160,
            "multivector_graph_neighbor_search_us": 60000,
            "multivector_graph_build_distance_proxy_calls": 1234,
            "multivector_graph_build_distance_exact_calls": 0,
        },
    }
    row = document_node_serving_build_only_row(
        profile=profile,
        index_phase=index_phase,
        signature=signature,
        profile_reused_index=False,
        grouped_profiles=[profile],
    )
    report = {
        "dataset": {"documents": 10, "queries": 0, "qrels": 0},
        "settings": {},
        "results": [],
        "document_node_serving_build_only": {
            "enabled": True,
            "retrieval_skipped": True,
            "admission_skipped": True,
            "total_elapsed_ms": 1300.0,
            "index_build_elapsed_ms_total": 1234.5,
            "profiles_run": 1,
            "index_builds": 1,
            "index_build_group_count": 1,
            "index_reuse_count": 0,
            "effective_profiles": [profile.name],
            "results": [row],
        },
    }
    serialized = json.loads(json.dumps(report))
    assert serialized["document_node_serving_build_only"]["results"][0]["profile"] == "centroid_mean_f16"
    assert (
        serialized["document_node_serving_build_only"]["results"][0][
            "multivector_centroid_posting_count"
        ]
        == 160
    )
    assert (
        serialized["document_node_serving_build_only"]["results"][0][
            "dominant_build_phase"
        ]
        == "centroid_build"
    )
    assert (
        serialized["document_node_serving_build_only"]["results"][0][
            "dominant_centroid_build_subphase"
        ]
        == "centroid_cluster"
    )
    assert (
        serialized["document_node_serving_build_only"]["results"][0][
            "multivector_centroid_residual_us"
        ]
        == 20000
    )
    assert (
        serialized["document_node_serving_build_only"]["results"][0][
            "build_phase_known_ms"
        ]
        == 300.0
    )
    assert (
        serialized["document_node_serving_build_only"]["results"][0][
            "build_phase_times_ms"
        ]["graph_neighbor_search"]
        == 60.0
    )
    assert (
        serialized["document_node_serving_build_only"]["results"][0][
            "multivector_graph_build_distance_proxy_calls"
        ]
        == 1234
    )
    markdown = markdown_benchmark_summary(serialized)
    assert "### Document-node serving build-only" in markdown
    assert "centroid_mean_f16" in markdown
    assert "centroid postings" in markdown
    assert "dominant build phase" in markdown
    assert "unattributed ms" in markdown
    assert "#### Document-node topology build phases" in markdown
    assert "proxy distance calls" in markdown
    assert "1234" in markdown


def _self_check_document_node_serving_latency_only() -> None:
    args = argparse.Namespace(
        document_node_serving_grid_include_experimental=False,
        document_node_serving_grid_include_proxy_encoders=False,
        document_node_serving_grid_include_bm25_rescue=False,
        multivector_centroid_count="auto",
        serving_profile_name=None,
        serving_storage=None,
        serving_cache=None,
        serving_ef=100,
        serving_oversampling=1,
        serving_candidate_k=800,
        serving_exact_rerank_k=100,
        final_k=10,
        reuse_index=False,
        serving_fail_on_slow_path=False,
        serving_latency_warmup_queries=1,
        serving_loaded_document_count=1000,
        admission_budget_sweep=None,
        admission_budget_sweep_explicit=False,
        document_node_serving_grid=True,
        document_node_serving_grid_smoke=False,
        document_node_serving_grid_budget_mode="largest_only",
        multivector_token_pooling_min_tokens=16,
    )
    profile = document_node_serving_latency_profile(args)
    assert profile.name == DOCUMENT_NODE_SERVING_LATENCY_DEFAULT_PROFILE
    assert profile.storage_kind == "f16"

    override_profile = document_node_serving_latency_profile(
        clone_args(
            args,
            serving_profile_name="proxy_normalized_mean_f16",
            serving_storage="sq8",
            serving_cache="resident",
        )
    )
    assert override_profile.storage_kind == "sq8"
    assert override_profile.cache_mode == "resident"

    latency_args, selected_profile, effective_rerank_k = document_node_serving_latency_args(args)
    assert selected_profile.name == "proxy_normalized_mean_f16"
    assert latency_args.multivector_doc_candidate_k == 800
    assert latency_args.multivector_exact_rerank == "topk"
    assert latency_args.multivector_exact_rerank_k == 100
    assert latency_args.multivector_branch_plan == "dense_only"
    assert latency_args.dense_k == 800
    assert effective_rerank_k == 100

    capped_args, _, capped_rerank_k = document_node_serving_latency_args(
        clone_args(args, serving_candidate_k=50, serving_exact_rerank_k=100)
    )
    assert capped_args.multivector_doc_candidate_k == 50
    assert capped_args.multivector_exact_rerank_k == 50
    assert capped_rerank_k == 50

    calls: list[tuple[str, int | None]] = []

    def fake_retrieval(
        conn: psycopg.Connection[Any],
        method: str,
        query: QueryItem,
        loop_args: argparse.Namespace,
        final_k: int,
        dense_k: int | None = None,
    ) -> list[str]:
        assert method == QUERY_ONLY_METHOD
        assert final_k == 10
        calls.append((query.query_id, dense_k))
        return [f"{query.query_id}-doc-{i}" for i in range(final_k)]

    def fake_stats(conn: psycopg.Connection[Any]) -> dict[str, Any]:
        call_number = len(calls)
        return {
            "index_used": True,
            "multivector_enabled": True,
            "multivector_candidate_source": "proxy_vector",
            "proxy_encoder_kind": "normalized_mean",
            "multivector_doc_graph_docs_scored": 800,
            "multivector_doc_graph_edges_visited": 2400,
            "multivector_exact_rerank_docs": 100,
            "multivector_exact_rerank_pairs": 320000,
            "multivector_exact_kernel": "blocked_scalar",
            "sidecar_cache_build_this_query": call_number == 1,
            "sidecar_cache_build_bytes": 4096 if call_number == 1 else 0,
            "sidecar_cache_build_pages_read": 1 if call_number == 1 else 0,
            "sidecar_query_bytes_touched": 0,
            "sidecar_query_pages_read": 0,
            "sidecar_query_vectors_loaded": 0,
            "sidecar_query_time_us": 0,
            "native_cache_reused": call_number > 1,
        }

    loop = run_serving_latency_query_loop(
        None,  # type: ignore[arg-type]
        latency_args,
        [QueryItem("q1", "one"), QueryItem("q2", "two")],
        selected_profile,
        retrieval_provider=fake_retrieval,
        stats_provider=fake_stats,
    )
    assert loop["query_count"] == 2
    assert loop["warmup_query_count"] == 1
    assert loop["warmup_excluded"] is True
    assert loop["warmup_latency"]["runs"] == 1
    assert loop["cache_build_queries"] == 1
    assert loop["warm_queries"] == 2
    assert loop["cache_reused_on_warm_queries"] == 2
    assert loop["exact_admission_baseline_calls"] == 0
    assert loop["serving_slow_path_failed"] is False
    assert calls == [("q1", 800), ("q1", 800), ("q2", 800)]
    assert loop["latency"]["runs"] == 2
    scan = loop["scan_stats_summary"]
    assert scan["field_summary"]["multivector_exact_rerank_docs"]["p50"] == 100
    assert scan["last_selected"]["multivector_exact_kernel"] == "blocked_scalar"

    markdown = markdown_benchmark_summary({
        "document_node_serving_latency_only": {
            "enabled": True,
            "profile": "proxy_normalized_mean_f16",
            "settings": {
                "candidate_source": "proxy_vector",
                "proxy_encoder": "normalized_mean",
                "centroids": "off",
                "storage_kind": "f16",
                "storage_cache_mode": "auto",
                "ef": 100,
                "oversampling": 1,
                "candidate_k": 800,
                "effective_exact_rerank_k": 100,
                "warmup_queries": 1,
            },
            "warmup_latency": loop["warmup_latency"],
            "warmup_excluded": loop["warmup_excluded"],
            "latency": loop["latency"],
            "cache_build_queries": loop["cache_build_queries"],
            "warm_queries": loop["warm_queries"],
            "cache_reused_on_warm_queries": loop["cache_reused_on_warm_queries"],
            "query_count": loop["query_count"],
            "admission_metrics": {
                "available": False,
                "reason": "latency-only mode does not compute exact admission baselines",
            },
            "serving_slow_path_warnings": loop["serving_slow_path_warnings"],
            "serving_slow_path_failed": loop["serving_slow_path_failed"],
            "scan_stats_summary": scan,
        }
    })
    assert "### Document-node serving latency-only" in markdown
    assert "admission baselines" in markdown
    assert "blocked_scalar" in markdown
    assert "#### Slow path warnings" in markdown

    slow_profile = DocumentNodeServingProfile(
        name="proxy_normalized_mean_f16",
        candidate_source="proxy_vector",
        proxy_encoder="normalized_mean",
        storage_kind="f16",
        cache_mode="auto",
    )
    slow_args = clone_args(
        args,
        multivector_exact_rerank_k=100,
        serving_loaded_document_count=1000,
    )
    slow_stats = {
        "multivector_candidate_source": "exact_doc_scan",
        "multivector_plain_fallback_used": True,
        "multivector_plain_fallback_reason": "small_table",
        "multivector_doc_graph_warning": "document_node_f32_sidecar_exact_scan",
        "multivector_exact_rerank_docs": 140,
        "multivector_doc_graph_docs_scored": 950,
        "multivector_doc_sidecar_cache_mode": "paged",
        "multivector_doc_sidecar_pages_read": 200,
        "multivector_doc_sidecar_vectors_loaded": 950,
        "multivector_doc_sidecar_resident_vectors_loaded": 0,
        "multivector_doc_sidecar_bytes_touched": 10_000,
        "multivector_doc_sidecar_docmap_bytes_touched": 10_000,
        "multivector_docmap_bytes": 10_000,
        "multivector_exact_kernel": "scalar",
        "native_cache_built_this_scan": True,
    }
    simd_env_keys = (
        "SIMD_BUILD",
        "PGTURBOHYBRID_SIMD_BUILD",
        "PGTURBOHYBRID_DENSE_EXACT_SIMD_FORCE",
        "PGTURBOHYBRID_DENSE_SIMD_FORCE",
    )
    old_simd_env = {key: os.environ.pop(key, None) for key in simd_env_keys}
    try:
        warnings = validate_serving_scan_path(slow_stats, slow_args, slow_profile)
    finally:
        for key, value in old_simd_env.items():
            if value is not None:
                os.environ[key] = value
    assert any("candidate_source_mismatch" in warning for warning in warnings)
    assert "plain_fallback_used" in warnings
    assert any("plain_fallback_reason" in warning for warning in warnings)
    assert any("doc_graph_warning" in warning for warning in warnings)
    assert any("exact_rerank_docs_exceeds_serving_budget" in warning for warning in warnings)
    assert any("docs_scored_near_table_size" in warning for warning in warnings)
    assert any("doc_graph_docs_scored_near_table_size" in warning for warning in warnings)
    assert any("proxy_vector_near_exhaustive_sidecar_touch" in warning for warning in warnings)
    assert any("paged_sidecar_high_pages_read" in warning for warning in warnings)
    assert any("exact_kernel_scalar" in warning for warning in warnings)
    assert "native_cache_built_this_scan" in warnings

    try:
        fail_on_serving_slow_path_if_requested(
            warnings,
            clone_args(args, serving_fail_on_slow_path=True),
        )
    except RuntimeError as exc:
        assert "serving slow path detected" in str(exc)
    else:
        raise AssertionError("--serving-fail-on-slow-path should raise on warnings")

    warning_row = {
        "profile": "warned",
        "candidate_source": "proxy_vector",
        "proxy_encoder": "normalized_mean",
        "storage_kind": "f16",
        "token_pooling": "off",
        "token_pooling_target_ratio": 1.0,
        "ef": 100,
        "oversampling": 1,
        "largest_budget": 800,
        "p95_ms": 10.0,
        "exact_top10_admission_recall": 0.9,
        "ndcg@10": 0.8,
        "serving_slow_path_warnings": warnings,
        "serving_slow_path_failed": False,
    }
    rec = compute_document_node_serving_recommendation(
        {"summary_rows": [warning_row]},
        exact_baseline={"available": False},
        min_top10_admission=0.8,
        min_ndcg_ratio_vs_exact=0.95,
        max_p95_ms=0.0,
    )
    assert rec["best_latency_safe"]["profile"] == "warned"


def run_document_node_serving_grid(
    conn: psycopg.Connection[Any],
    args: argparse.Namespace,
    queries: list[QueryItem],
    qrels: dict[str, dict[str, int]],
) -> dict[str, Any]:
    grid_started = time.perf_counter()
    mode = "smoke" if args.document_node_serving_grid_smoke else "full"
    stage_mode = effective_document_node_serving_stage_mode(args)
    profiles = effective_document_node_serving_profiles(args)
    ef_grid = effective_document_node_serving_ef_grid(args)
    oversampling_grid = effective_document_node_serving_oversampling_grid(args)
    budget_sweep = effective_serving_grid_budget_sweep(args)
    all_configs = document_node_serving_configs(profiles, ef_grid, oversampling_grid)
    query_subset_used = False
    queries_available = len(queries)
    effective_queries = queries
    if (
        args.document_node_serving_grid_smoke
        and len(queries) > DOCUMENT_NODE_SERVING_GRID_SMOKE_MAX_QUERIES
    ):
        query_subset_used = True
        effective_queries = queries[:DOCUMENT_NODE_SERVING_GRID_SMOKE_MAX_QUERIES]
    exact_top_cache: dict[str, list[dict[str, Any]]] = {}
    exact_top_cache_loaded_document_count = loaded_document_count(conn)

    def run_stage(
        *,
        stage_name: str,
        stage_args: argparse.Namespace,
        configs: list[DocumentNodeServingConfig],
        stage_queries: list[QueryItem],
    ) -> dict[str, Any]:
        stage_started = time.perf_counter()
        stage_budget_sweep = effective_serving_grid_budget_sweep(stage_args)
        stage_executed_budgets = effective_serving_grid_executed_budgets(stage_args)
        summary_rows: list[dict[str, Any]] = []
        full_admission: list[dict[str, Any]] = []
        profile_summaries: list[dict[str, Any]] = []
        index_build_elapsed_ms_total = 0.0
        exact_baseline_elapsed_ms_total = 0.0
        retrieval_elapsed_ms_total = 0.0
        exact_baseline_query_count = 0
        retrieval_query_count = 0
        exact_top_cache_hits = 0
        exact_top_cache_misses = 0
        index_builds = 0

        profiles_by_name: dict[str, DocumentNodeServingProfile] = {}
        configs_by_profile: dict[str, list[DocumentNodeServingConfig]] = {}
        for config in configs:
            profiles_by_name.setdefault(config.profile.name, config.profile)
            configs_by_profile.setdefault(config.profile.name, []).append(config)

        profile_groups: list[tuple[tuple[tuple[str, Any], ...], list[DocumentNodeServingProfile]]] = []
        profile_groups_by_signature: dict[
            tuple[tuple[str, Any], ...],
            list[DocumentNodeServingProfile],
        ] = {}
        for profile in profiles_by_name.values():
            signature = serving_profile_index_signature(stage_args, profile)
            if signature not in profile_groups_by_signature:
                profile_groups_by_signature[signature] = []
                profile_groups.append((signature, profile_groups_by_signature[signature]))
            profile_groups_by_signature[signature].append(profile)
        index_build_reused_for_profiles = [
            {
                "index_signature": serializable_index_signature(signature),
                "profiles": [profile.name for profile in grouped_profiles],
            }
            for signature, grouped_profiles in profile_groups
        ]

        for signature, grouped_profiles in profile_groups:
            representative_profile = grouped_profiles[0]
            index_args = document_node_serving_profile_args(
                stage_args,
                representative_profile,
                ef=0,
                oversampling=1,
                budget_sweep=stage_budget_sweep,
            )
            index_phase = build_index(conn, index_args)
            index_phase["index_signature_tuple"] = signature
            index_phase["index_signature"] = serializable_index_signature(signature)
            index_phase["index_build_reused_for_profiles"] = [
                profile.name for profile in grouped_profiles
            ]
            index_builds += 1
            index_build_elapsed_ms = float(index_phase.get("elapsed_ms", 0.0) or 0.0)
            index_build_elapsed_ms_total += index_build_elapsed_ms
            for profile_index, profile in enumerate(grouped_profiles):
                profile_started = time.perf_counter()
                profile_rows: list[dict[str, Any]] = []
                profile_reused_index = profile_index > 0
                for config in configs_by_profile.get(profile.name, []):
                    mode_args = document_node_serving_profile_args(
                        stage_args,
                        profile,
                        ef=config.ef,
                        oversampling=config.oversampling,
                        budget_sweep=stage_budget_sweep,
                    )
                    validate_serving_profile_index_reuse(
                        args=mode_args,
                        profile=profile,
                        index_phase=index_phase,
                        expected_signature=signature,
                    )
                    mode_args = clone_args(mode_args, admission_debug_context="serving_grid")
                    admission = run_admission_debug(
                        conn,
                        mode_args,
                        stage_queries,
                        exact_top_cache=exact_top_cache,
                        exact_top_cache_document_count=exact_top_cache_loaded_document_count,
                        executed_budgets=stage_executed_budgets,
                    )
                    aggregate = admission.get("aggregate", {})
                    if not isinstance(aggregate, dict):
                        aggregate = {}
                    exact_baseline_elapsed_ms_total += float(
                        aggregate.get("exact_baseline_elapsed_ms_total", 0.0) or 0.0
                    )
                    retrieval_elapsed_ms_total += float(
                        aggregate.get("retrieval_elapsed_ms_total", 0.0) or 0.0
                    )
                    exact_baseline_query_count += int(
                        aggregate.get("exact_baseline_query_count", 0) or 0
                    )
                    retrieval_query_count += int(aggregate.get("retrieval_query_count", 0) or 0)
                    exact_top_cache_hits += int(aggregate.get("exact_top_cache_hits", 0) or 0)
                    exact_top_cache_misses += int(aggregate.get("exact_top_cache_misses", 0) or 0)
                    full_admission.append({
                        "stage": stage_name,
                        "profile": profile.name,
                        "ef": config.ef,
                        "oversampling": config.oversampling,
                        "index_signature": serializable_index_signature(signature),
                        "index_build_reused": profile_reused_index,
                        "admission_debug": admission,
                    })
                    row = document_node_serving_summary_row(
                        profile=profile,
                        args=mode_args,
                        ef=config.ef,
                        oversampling=config.oversampling,
                        index_phase=index_phase,
                        admission=admission,
                        qrels=qrels,
                    )
                    row["stage"] = stage_name
                    row["index_signature"] = serializable_index_signature(signature)
                    row["index_build_reused"] = profile_reused_index
                    row["index_build_reused_for_profiles"] = [
                        grouped_profile.name for grouped_profile in grouped_profiles
                    ]
                    summary_rows.append(row)
                    profile_rows.append(row)

                profile_exact_baseline_elapsed_ms = sum(
                    float(item.get("exact_baseline_elapsed_ms", 0.0) or 0.0)
                    for item in profile_rows
                )
                profile_retrieval_elapsed_ms = sum(
                    float(item.get("retrieval_elapsed_ms", 0.0) or 0.0)
                    for item in profile_rows
                )
                profile_admission_debug_elapsed_ms = sum(
                    float(item.get("run_elapsed_ms", 0.0) or 0.0)
                    for item in profile_rows
                )
                profile_build_stats = index_phase.get("build_stats", {})
                if not isinstance(profile_build_stats, dict):
                    profile_build_stats = {}
                profile_index_build_elapsed_ms = round(
                    0.0 if profile_reused_index else index_build_elapsed_ms,
                    3,
                )
                profile_build_phase_summary = document_node_build_phase_summary(
                    profile_build_stats,
                    index_build_elapsed_ms=profile_index_build_elapsed_ms,
                )
                profile_elapsed_ms = elapsed_ms_since(profile_started)
                for profile_row in profile_rows:
                    profile_row["profile_elapsed_ms"] = profile_elapsed_ms
                    profile_row["index_build_elapsed_ms"] = profile_index_build_elapsed_ms
                    profile_row["build_phase_summary"] = profile_build_phase_summary
                profile_summaries.append({
                    "stage": stage_name,
                    "profile": profile.name,
                    "candidate_source": profile.candidate_source,
                    "branch_plan": profile.branch_plan,
                    "bm25_candidate_injection": profile.bm25_candidate_injection,
                    "sparse_candidate_source": profile.sparse_candidate_source,
                    "proxy_encoder": profile.proxy_encoder,
                    "centroids": profile.centroids,
                    "storage_kind": profile.storage_kind,
                    "storage_cache_mode": profile.cache_mode,
                    "token_pooling": profile.token_pooling,
                    "token_pooling_target_ratio": profile.token_pooling_target_ratio,
                    "entry_sample_count": profile.entry_sample_count,
                    "effective_entry_sample_count": (
                        document_node_serving_profile_args(
                            stage_args,
                            profile,
                            ef=0,
                            oversampling=1,
                            budget_sweep=stage_budget_sweep,
                        ).multivector_doc_graph_entry_sample_count
                    ),
                    "entry_sidecar": profile.entry_sidecar,
                    "entry_sidecar_representatives": profile.entry_sidecar_representatives,
                    "entry_sidecar_strategy": profile.entry_sidecar_strategy,
                    "centroid_lite_max_postings_per_token": (
                        profile.centroid_lite_max_postings_per_token
                    ),
                    "centroid_lite_pruning": profile.centroid_lite_pruning,
                    "index_signature": serializable_index_signature(signature),
                    "index_build_reused": profile_reused_index,
                    "index_build_reused_for_profiles": [
                        grouped_profile.name for grouped_profile in grouped_profiles
                    ],
                    "serving_exact_rerank_mode": stage_args.serving_exact_rerank_mode,
                    "requested_serving_exact_rerank_k": stage_args.serving_exact_rerank_k,
                    "admission_debug_mode": effective_admission_debug_mode(
                        clone_args(stage_args, admission_debug_context="serving_grid")
                    ),
                    "profile_elapsed_ms": profile_elapsed_ms,
                    "index_build_elapsed_ms": profile_index_build_elapsed_ms,
                    "index_build_stats": profile_build_stats,
                    **profile_build_phase_summary,
                    "multivector_centroid_build_us": profile_build_stats.get(
                        "multivector_centroid_build_us"
                    ),
                    "multivector_centroid_cluster_us": profile_build_stats.get(
                        "multivector_centroid_cluster_us"
                    ),
                    "multivector_centroid_residual_us": profile_build_stats.get(
                        "multivector_centroid_residual_us"
                    ),
                    "multivector_proxy_build_us": profile_build_stats.get(
                        "multivector_proxy_build_us"
                    ),
                    "learned_projection_doc_encode_build_us": profile_build_stats.get(
                        "learned_projection_doc_encode_build_us"
                    ),
                    "multivector_doc_sidecar_write_us": profile_build_stats.get(
                        "multivector_doc_sidecar_write_us"
                    ),
                    "multivector_centroid_sidecar_write_us": profile_build_stats.get(
                        "multivector_centroid_sidecar_write_us"
                    ),
                    "multivector_centroid_posting_write_us": profile_build_stats.get(
                        "multivector_centroid_posting_write_us"
                    ),
                    "multivector_centroid_posting_count": profile_build_stats.get(
                        "multivector_centroid_posting_count"
                    ),
                    "admission_debug_elapsed_ms_total": round(profile_admission_debug_elapsed_ms, 3),
                    "exact_baseline_elapsed_ms_total": round(profile_exact_baseline_elapsed_ms, 3),
                    "retrieval_elapsed_ms_total": round(profile_retrieval_elapsed_ms, 3),
                    "retrieval_query_count": sum(
                        int(item.get("retrieval_query_count", 0) or 0)
                        for item in profile_rows
                    ),
                    "exact_baseline_query_count": sum(
                        int(item.get("exact_baseline_query_count", 0) or 0)
                        for item in profile_rows
                    ),
                    "runs": len(profile_rows),
                })

        return {
            "stage": stage_name,
            "elapsed_ms": elapsed_ms_since(stage_started),
            "configs_run": len(configs),
            "profiles_run": len(profiles_by_name),
            "index_builds": index_builds,
            "index_build_group_count": len(profile_groups),
            "index_reuse_count": max(len(profiles_by_name) - len(profile_groups), 0),
            "index_build_reused_for_profiles": index_build_reused_for_profiles,
            "index_build_elapsed_ms_total": round(index_build_elapsed_ms_total, 3),
            "exact_baseline_elapsed_ms_total": round(exact_baseline_elapsed_ms_total, 3),
            "retrieval_elapsed_ms_total": round(retrieval_elapsed_ms_total, 3),
            "exact_baseline_query_count": exact_baseline_query_count,
            "retrieval_query_count": retrieval_query_count,
            "exact_top_cache_hits": exact_top_cache_hits,
            "exact_top_cache_misses": exact_top_cache_misses,
            "ef_values_run": sorted({config.ef for config in configs}),
            "oversampling_values_run": sorted({config.oversampling for config in configs}),
            "budgets_run": stage_executed_budgets,
            "budget_sweep": stage_budget_sweep,
            "executed_budgets": stage_executed_budgets,
            "queries_run": len(stage_queries),
            "profile_summaries": profile_summaries,
            "summary_rows": summary_rows,
            "admission_debug_runs": full_admission,
        }

    if stage_mode == "two_stage":
        probe_query_count = min(max(int(args.serving_grid_probe_queries), 1), len(effective_queries))
        stage1_debug_mode = effective_admission_debug_mode(
            clone_args(args, admission_debug_context="serving_grid")
        )
        if stage1_debug_mode == "trace":
            stage1_debug_mode = "summary"
        stage1_args = clone_args(
            args,
            document_node_serving_grid_budget_mode="largest_only",
            serving_exact_rerank_mode="serving",
            admission_debug_mode=stage1_debug_mode,
            admission_debug_context="serving_grid",
        )
        stage1 = run_stage(
            stage_name="stage1_probe",
            stage_args=stage1_args,
            configs=all_configs,
            stage_queries=effective_queries[:probe_query_count],
        )
        selected_configs, annotated_stage1_rows, pruned_configs = select_document_node_serving_stage2_configs(
            stage1["summary_rows"],
            all_configs,
            finalists=max(int(args.serving_grid_finalists), 1),
        )
        stage1["summary_rows"] = annotated_stage1_rows
        stage2_args = clone_args(args, admission_debug_context="serving_grid")
        stage2 = run_stage(
            stage_name="stage2_finalists",
            stage_args=stage2_args,
            configs=selected_configs,
            stage_queries=effective_queries,
        )
        active_stage = stage2
        stage_results = stage2["summary_rows"]
        stages = [stage1, stage2]
        stage1_results = annotated_stage1_rows
        stage2_results = stage2["summary_rows"]
    else:
        single_args = clone_args(args, admission_debug_context="serving_grid")
        single = run_stage(
            stage_name="single",
            stage_args=single_args,
            configs=all_configs,
            stage_queries=effective_queries,
        )
        active_stage = single
        stage_results = single["summary_rows"]
        stages = [single]
        stage1_results = []
        stage2_results = []
        pruned_configs = []
        selected_configs = all_configs

    exact_top_cache_verified = False
    if args.document_node_serving_grid_verify_exact_cache and effective_queries:
        query = effective_queries[0]
        cache_key = exact_top_cache_key(query.query_id, args.admission_k)
        cached = exact_top_cache.get(cache_key)
        if cached is None:
            raise RuntimeError(f"exact admission top cache missing verification key {cache_key}")
        recomputed = exact_admission_top(conn, query, args.admission_k)
        cached_doc_ids = [str(item.get("doc_id", "")) for item in cached]
        recomputed_doc_ids = [str(item.get("doc_id", "")) for item in recomputed]
        if cached_doc_ids != recomputed_doc_ids:
            raise RuntimeError(
                "exact admission top cache verification failed for "
                f"{cache_key}: cached {cached_doc_ids}, recomputed {recomputed_doc_ids}"
            )
        exact_top_cache_verified = True

    index_builds = sum(int(stage.get("index_builds", 0) or 0) for stage in stages)
    index_build_group_count = sum(
        int(stage.get("index_build_group_count", 0) or 0)
        for stage in stages
    )
    index_reuse_count = sum(int(stage.get("index_reuse_count", 0) or 0) for stage in stages)
    index_build_elapsed_ms_total = sum(
        float(stage.get("index_build_elapsed_ms_total", 0.0) or 0.0)
        for stage in stages
    )
    exact_baseline_elapsed_ms_total = sum(
        float(stage.get("exact_baseline_elapsed_ms_total", 0.0) or 0.0)
        for stage in stages
    )
    retrieval_elapsed_ms_total = sum(
        float(stage.get("retrieval_elapsed_ms_total", 0.0) or 0.0)
        for stage in stages
    )
    exact_baseline_query_count = sum(
        int(stage.get("exact_baseline_query_count", 0) or 0)
        for stage in stages
    )
    retrieval_query_count = sum(
        int(stage.get("retrieval_query_count", 0) or 0)
        for stage in stages
    )
    exact_top_cache_hits = sum(int(stage.get("exact_top_cache_hits", 0) or 0) for stage in stages)
    exact_top_cache_misses = sum(int(stage.get("exact_top_cache_misses", 0) or 0) for stage in stages)
    profile_summaries = [
        item
        for stage in stages
        for item in stage.get("profile_summaries", [])
        if isinstance(item, dict)
    ]
    full_admission = [
        item
        for stage in stages
        for item in stage.get("admission_debug_runs", [])
        if isinstance(item, dict)
    ]
    index_build_reused_for_profiles = [
        item
        for stage in stages
        for item in stage.get("index_build_reused_for_profiles", [])
        if isinstance(item, dict)
    ]
    candidate_source_deltas = compute_document_node_candidate_source_deltas(stage_results)

    return {
        "enabled": True,
        "production_oriented": True,
        "mode": mode,
        "stage_mode": stage_mode,
        "probe_queries": int(args.serving_grid_probe_queries),
        "finalists": int(args.serving_grid_finalists),
        "stage1_results": stage1_results,
        "stage2_results": stage2_results,
        "pruned_configs": pruned_configs,
        "smoke_only": mode == "smoke",
        "serving_evidence": mode == "full",
        "budget_mode": args.document_node_serving_grid_budget_mode,
        "serving_exact_rerank_mode": args.serving_exact_rerank_mode,
        "requested_serving_exact_rerank_k": args.serving_exact_rerank_k,
        "admission_debug_mode": effective_admission_debug_mode(
            clone_args(args, admission_debug_context="serving_grid")
        ),
        "trace_enabled": effective_admission_debug_mode(
            clone_args(args, admission_debug_context="serving_grid")
        ) == "trace",
        "include_experimental": args.document_node_serving_grid_include_experimental,
        "include_proxy_encoder_variants": args.document_node_serving_grid_include_proxy_encoders,
        "include_learned_projection": args.document_node_serving_grid_include_learned_projection,
        "include_bm25_rescue": args.document_node_serving_grid_include_bm25_rescue,
        "include_learned_sparse_rescue": args.document_node_serving_grid_include_learned_sparse_rescue,
        "include_reservoirs": args.document_node_serving_grid_include_reservoirs,
        "include_centroid_lite_caps": args.document_node_serving_grid_include_centroid_lite_caps,
        "include_entry_samples": args.document_node_serving_grid_include_entry_samples,
        "include_entry_sidecar": args.document_node_serving_grid_include_entry_sidecar,
        "proxy_admission_focus": args.document_node_serving_grid_proxy_admission_focus,
        "centroid_lite_focus": args.document_node_serving_grid_centroid_lite_focus,
        "token_pooling_focus": args.document_node_serving_grid_token_pooling_focus,
        "effective_centroid_lite_posting_caps": effective_document_node_serving_centroid_lite_caps(args),
        "effective_entry_sample_counts": effective_document_node_serving_entry_sample_counts(args),
        "effective_entry_sidecar_representatives": (
            effective_document_node_serving_entry_sidecar_representatives(args)
        ),
        "total_elapsed_ms": elapsed_ms_since(grid_started),
        "index_build_elapsed_ms_total": round(index_build_elapsed_ms_total, 3),
        "exact_baseline_elapsed_ms_total": round(exact_baseline_elapsed_ms_total, 3),
        "retrieval_elapsed_ms_total": round(retrieval_elapsed_ms_total, 3),
        "recommendation_elapsed_ms": None,
        "profiles_run": len(profiles),
        "index_builds": index_builds,
        "index_build_group_count": index_build_group_count,
        "index_reuse_count": index_reuse_count,
        "index_build_reused_for_profiles": index_build_reused_for_profiles,
        "ef_values_run": active_stage.get("ef_values_run", []),
        "oversampling_values_run": active_stage.get("oversampling_values_run", []),
        "budgets_run": active_stage.get("budgets_run", []),
        "queries_run": int(active_stage.get("queries_run", len(effective_queries)) or 0),
        "retrieval_query_count": retrieval_query_count,
        "exact_baseline_query_count": exact_baseline_query_count,
        "exact_top_cache_enabled": True,
        "exact_top_cache_hits": exact_top_cache_hits,
        "exact_top_cache_misses": exact_top_cache_misses,
        "exact_top_cache_entries": len(exact_top_cache),
        "exact_top_cache_loaded_document_count": exact_top_cache_loaded_document_count,
        "exact_top_cache_metadata": {
            "loaded_document_count": exact_top_cache_loaded_document_count,
            "key_fields": ["admission_k", "query_id"],
            "scope": "document_node_serving_grid_run",
            "force_reload_safe": "not persisted across process runs or reload boundaries",
        },
        "exact_top_cache_verified": exact_top_cache_verified,
        "budget_sweep": budget_sweep,
        "requested_budget_sweep": budget_sweep,
        "executed_budgets": active_stage.get("executed_budgets", []),
        "executed_budget_count": len(active_stage.get("executed_budgets", [])),
        "effective_budget_sweep": budget_sweep,
        "effective_profiles": [profile.name for profile in profiles],
        "effective_ef_grid": ef_grid,
        "effective_oversampling_grid": oversampling_grid,
        "effective_stage2_configs": [
            document_node_serving_config_id(config)
            for config in selected_configs
        ],
        "query_subset_used": query_subset_used,
        "queries_available": queries_available,
        "query_subset_limit": (
            DOCUMENT_NODE_SERVING_GRID_SMOKE_MAX_QUERIES
            if args.document_node_serving_grid_smoke
            else None
        ),
        "ef_grid": ef_grid,
        "oversampling_grid": oversampling_grid,
        "cache_grid": ["auto"],
        "profiles": [profile.__dict__ for profile in profiles],
        "profile_summaries": profile_summaries,
        "candidate_source_deltas": candidate_source_deltas,
        "queries": int(active_stage.get("queries_run", len(effective_queries)) or 0),
        "results": stage_results,
        "summary_rows": stage_results,
        "admission_debug_runs": full_admission,
    }


def document_node_build_phase_summary(
    build_stats: dict[str, Any],
    *,
    index_build_elapsed_ms: float,
) -> dict[str, Any]:
    def phase_ms(key: str) -> float:
        value = build_stats.get(key)
        if isinstance(value, bool) or value is None:
            return 0.0
        try:
            return max(float(value), 0.0) / 1000.0
        except (TypeError, ValueError):
            return 0.0

    phase_times_ms = {
        "centroid_build": phase_ms("multivector_centroid_build_us"),
        "proxy_build": phase_ms("multivector_proxy_build_us"),
        "doc_sidecar_write": phase_ms("multivector_doc_sidecar_write_us"),
        "centroid_sidecar_write": phase_ms("multivector_centroid_sidecar_write_us"),
        "centroid_posting_write": phase_ms("multivector_centroid_posting_write_us"),
        "graph_node_assignment": phase_ms("multivector_graph_node_assignment_us"),
        "graph_entry_search": phase_ms("multivector_graph_entry_search_us"),
        "graph_neighbor_search": phase_ms("multivector_graph_neighbor_search_us"),
        "graph_neighbor_select": phase_ms("multivector_graph_neighbor_select_us"),
        "graph_link_insert": phase_ms("multivector_graph_link_insert_us"),
        "graph_reciprocal_prune": phase_ms("multivector_graph_reciprocal_prune_us"),
        "graph_segment_write": phase_ms("multivector_graph_segment_write_us"),
        "graph_wal": phase_ms("multivector_graph_wal_us"),
    }
    centroid_subphase_times_ms = {
        "centroid_cluster": phase_ms("multivector_centroid_cluster_us"),
        "centroid_residual": phase_ms("multivector_centroid_residual_us"),
    }
    known_ms = round(sum(phase_times_ms.values()), 3)
    elapsed_ms = max(float(index_build_elapsed_ms or 0.0), 0.0)
    unattributed_ms = round(max(elapsed_ms - known_ms, 0.0), 3)
    dominant_phase, dominant_ms = max(
        phase_times_ms.items(),
        key=lambda item: (item[1], item[0]),
    )
    if dominant_ms <= 0.0:
        dominant_phase = "unavailable"
    dominant_centroid_subphase, dominant_centroid_subphase_ms = max(
        centroid_subphase_times_ms.items(),
        key=lambda item: (item[1], item[0]),
    )
    if dominant_centroid_subphase_ms <= 0.0:
        dominant_centroid_subphase = "unavailable"
    return {
        "build_phase_times_ms": {key: round(value, 3) for key, value in phase_times_ms.items()},
        "build_centroid_subphase_times_ms": {
            key: round(value, 3) for key, value in centroid_subphase_times_ms.items()
        },
        "build_phase_known_ms": known_ms,
        "build_phase_unattributed_ms": unattributed_ms,
        "build_phase_known_ratio": (
            round(min(known_ms / elapsed_ms, 1.0), 6)
            if elapsed_ms > 0.0
            else None
        ),
        "dominant_build_phase": dominant_phase,
        "dominant_build_phase_ms": round(dominant_ms, 3),
        "dominant_centroid_build_subphase": dominant_centroid_subphase,
        "dominant_centroid_build_subphase_ms": round(dominant_centroid_subphase_ms, 3),
    }


def document_node_serving_build_only_row(
    *,
    profile: DocumentNodeServingProfile,
    index_phase: dict[str, Any],
    signature: tuple[tuple[str, Any], ...],
    profile_reused_index: bool,
    grouped_profiles: list[DocumentNodeServingProfile],
) -> dict[str, Any]:
    build_stats = index_phase.get("build_stats", {})
    if not isinstance(build_stats, dict):
        build_stats = {}
    index_stats = index_phase.get("index_stats", {})
    if not isinstance(index_stats, dict):
        index_stats = {}
    index_build_elapsed_ms = round(
        0.0 if profile_reused_index else float(index_phase.get("elapsed_ms", 0.0) or 0.0),
        3,
    )
    phase_summary = document_node_build_phase_summary(
        build_stats,
        index_build_elapsed_ms=index_build_elapsed_ms,
    )
    return {
        "profile": profile.name,
        "candidate_source": profile.candidate_source,
        "branch_plan": profile.branch_plan,
        "bm25_candidate_injection": profile.bm25_candidate_injection,
        "sparse_candidate_source": profile.sparse_candidate_source,
        "graph_mode": "document_nodes",
        "proxy_encoder": profile.proxy_encoder,
        "centroids": profile.centroids,
        "centroid_count": profile.centroid_count,
        "storage_kind": profile.storage_kind,
        "storage_cache_mode": profile.cache_mode,
        "token_pooling": profile.token_pooling,
        "token_pooling_target_ratio": profile.token_pooling_target_ratio,
        "entry_sidecar": profile.entry_sidecar,
        "entry_sidecar_representatives": profile.entry_sidecar_representatives,
        "entry_sidecar_strategy": profile.entry_sidecar_strategy,
        "centroid_lite_max_postings_per_token": (
            profile.centroid_lite_max_postings_per_token
        ),
        "centroid_lite_pruning": profile.centroid_lite_pruning,
        "plain_fallback": profile.plain_fallback,
        "index_signature": serializable_index_signature(signature),
        "index_build_reused": profile_reused_index,
        "index_build_reused_for_profiles": [
            grouped_profile.name for grouped_profile in grouped_profiles
        ],
        "index_build_elapsed_ms": index_build_elapsed_ms,
        "index_bytes": int(index_phase.get("index_bytes", 0) or 0),
        "index_stats": index_stats,
        "index_build_stats": build_stats,
        **phase_summary,
        "multivector_centroid_build_us": build_stats.get("multivector_centroid_build_us"),
        "multivector_centroid_cluster_us": build_stats.get(
            "multivector_centroid_cluster_us"
        ),
        "multivector_centroid_residual_us": build_stats.get(
            "multivector_centroid_residual_us"
        ),
        "multivector_centroid_build_docs": build_stats.get("multivector_centroid_build_docs"),
        "multivector_centroid_build_vectors": build_stats.get("multivector_centroid_build_vectors"),
        "multivector_proxy_build_us": build_stats.get("multivector_proxy_build_us"),
        "learned_projection_doc_encode_build_us": build_stats.get(
            "learned_projection_doc_encode_build_us"
        ),
        "multivector_doc_sidecar_write_us": build_stats.get("multivector_doc_sidecar_write_us"),
        "multivector_centroid_sidecar_write_us": build_stats.get(
            "multivector_centroid_sidecar_write_us"
        ),
        "multivector_centroid_posting_write_us": build_stats.get(
            "multivector_centroid_posting_write_us"
        ),
        "multivector_centroid_posting_count": build_stats.get(
            "multivector_centroid_posting_count"
        ),
        "multivector_graph_node_assignment_us": build_stats.get(
            "multivector_graph_node_assignment_us"
        ),
        "multivector_graph_entry_search_us": build_stats.get(
            "multivector_graph_entry_search_us"
        ),
        "multivector_graph_neighbor_search_us": build_stats.get(
            "multivector_graph_neighbor_search_us"
        ),
        "multivector_graph_neighbor_select_us": build_stats.get(
            "multivector_graph_neighbor_select_us"
        ),
        "multivector_graph_link_insert_us": build_stats.get(
            "multivector_graph_link_insert_us"
        ),
        "multivector_graph_reciprocal_prune_us": build_stats.get(
            "multivector_graph_reciprocal_prune_us"
        ),
        "multivector_graph_segment_write_us": build_stats.get(
            "multivector_graph_segment_write_us"
        ),
        "multivector_graph_wal_us": build_stats.get(
            "multivector_graph_wal_us"
        ),
        "multivector_graph_build_distance_proxy_calls": build_stats.get(
            "multivector_graph_build_distance_proxy_calls"
        ),
        "multivector_graph_build_distance_exact_calls": build_stats.get(
            "multivector_graph_build_distance_exact_calls"
        ),
        "multivector_graph_build_distance_cache_hits": build_stats.get(
            "multivector_graph_build_distance_cache_hits"
        ),
        "multivector_graph_build_distance_cache_misses": build_stats.get(
            "multivector_graph_build_distance_cache_misses"
        ),
    }


def run_document_node_serving_build_only(
    conn: psycopg.Connection[Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    started = time.perf_counter()
    profiles = effective_document_node_serving_profiles(args)
    budget_sweep = effective_serving_grid_budget_sweep(args)
    profile_groups: list[tuple[tuple[tuple[str, Any], ...], list[DocumentNodeServingProfile]]] = []
    profile_groups_by_signature: dict[
        tuple[tuple[str, Any], ...],
        list[DocumentNodeServingProfile],
    ] = {}
    for profile in profiles:
        signature = serving_profile_index_signature(args, profile)
        if signature not in profile_groups_by_signature:
            profile_groups_by_signature[signature] = []
            profile_groups.append((signature, profile_groups_by_signature[signature]))
        profile_groups_by_signature[signature].append(profile)

    results: list[dict[str, Any]] = []
    index_build_elapsed_ms_total = 0.0
    last_index_phase: dict[str, Any] | None = None
    for signature, grouped_profiles in profile_groups:
        representative_profile = grouped_profiles[0]
        index_args = document_node_serving_profile_args(
            args,
            representative_profile,
            ef=0,
            oversampling=1,
            budget_sweep=budget_sweep,
        )
        index_phase = build_index(conn, index_args)
        index_phase["index_signature_tuple"] = signature
        index_phase["index_signature"] = serializable_index_signature(signature)
        index_phase["index_build_reused_for_profiles"] = [
            profile.name for profile in grouped_profiles
        ]
        last_index_phase = index_phase
        index_build_elapsed_ms_total += float(index_phase.get("elapsed_ms", 0.0) or 0.0)
        for profile_index, profile in enumerate(grouped_profiles):
            validate_serving_profile_index_reuse(
                args=document_node_serving_profile_args(
                    args,
                    profile,
                    ef=0,
                    oversampling=1,
                    budget_sweep=budget_sweep,
                ),
                profile=profile,
                index_phase=index_phase,
                expected_signature=signature,
            )
            results.append(
                document_node_serving_build_only_row(
                    profile=profile,
                    index_phase=index_phase,
                    signature=signature,
                    profile_reused_index=profile_index > 0,
                    grouped_profiles=grouped_profiles,
                )
            )

    if last_index_phase is None:
        raise RuntimeError("document-node serving build-only selected no profiles")

    return {
        "enabled": True,
        "mode": "build_only",
        "retrieval_skipped": True,
        "admission_skipped": True,
        "reason": "measure CREATE INDEX and document-node sidecar construction cost only",
        "total_elapsed_ms": elapsed_ms_since(started),
        "index_build_elapsed_ms_total": round(index_build_elapsed_ms_total, 3),
        "profiles_run": len(profiles),
        "index_builds": len(profile_groups),
        "index_build_group_count": len(profile_groups),
        "index_reuse_count": max(len(profiles) - len(profile_groups), 0),
        "effective_profiles": [profile.name for profile in profiles],
        "effective_budget_sweep": budget_sweep,
        "index_build_reused_for_profiles": [
            {
                "index_signature": serializable_index_signature(signature),
                "profiles": [profile.name for profile in grouped_profiles],
            }
            for signature, grouped_profiles in profile_groups
        ],
        "results": results,
        "last_index_phase": last_index_phase,
    }


def run_document_node_admission_grid(
    conn: psycopg.Connection[Any],
    args: argparse.Namespace,
    queries: list[QueryItem],
    qrels: dict[str, dict[str, int]],
) -> dict[str, Any]:
    if not bool(getattr(args, "serving_exact_rerank_mode_explicit", False)):
        args = clone_args(args, serving_exact_rerank_mode="admission_exhaustive")
    args = clone_args(args, admission_debug_context="admission_grid")
    storage_grid = args.document_node_storage_grid_values
    cache_grid = args.document_node_cache_grid_values
    pooling_grid = args.document_node_pooling_grid_values
    proxy_encoder_grid = args.document_node_proxy_encoder_grid_values
    ef_grid = args.document_node_ef_grid_values
    oversampling_grid = args.document_node_oversampling_grid_values
    budget_grid = parse_int_grid(args.admission_budget_sweep, "--admission-budget-sweep")
    results: list[dict[str, Any]] = []
    full_admission: list[dict[str, Any]] = []
    default_document_modes_done = False
    default_document_modes_reused_grid_index = False

    def run_default_document_modes(index_phase: dict[str, Any], *, reused_grid_index: bool) -> None:
        nonlocal default_document_modes_done, default_document_modes_reused_grid_index
        for mode in ("proxy_vector", "exact_doc_scan", "quantized_inverted_experimental"):
            mode_args = grid_mode_args(args, mode, "document_nodes")
            admission = run_admission_debug(conn, mode_args, queries)
            mode_name = f"{mode}_document_index" if mode == "exact_doc_scan" else mode
            full_admission.append({"mode": mode_name, "admission_debug": admission})
            results.append(grid_admission_summary(
                mode=mode_name,
                args=mode_args,
                index_phase=index_phase,
                admission=admission,
                qrels=qrels,
            ))
        default_document_modes_done = True
        default_document_modes_reused_grid_index = reused_grid_index

    token_args = clone_args(args, multivector_graph="token_nodes", reuse_index=False)
    token_index_phase = build_index(conn, token_args)
    for mode in ("token_nodes", "exact_token_scan", "plain_fallback", "exact_doc_scan"):
        mode_args = grid_mode_args(args, mode, "token_nodes")
        admission = run_admission_debug(conn, mode_args, queries)
        full_admission.append({"mode": mode, "admission_debug": admission})
        results.append(grid_admission_summary(
            mode=mode,
            args=mode_args,
            index_phase=token_index_phase,
            admission=admission,
            qrels=qrels,
        ))

    token_centroid_args = clone_args(
        args,
        multivector_graph="token_nodes",
        reuse_index=False,
        multivector_centroids="kmeans",
        multivector_centroid_count=args.multivector_centroid_count,
    )
    token_centroid_index_phase = build_index(conn, token_centroid_args)
    mode_args = grid_mode_args(token_centroid_args, "centroid_lite", "token_nodes")
    admission = run_admission_debug(conn, mode_args, queries)
    full_admission.append({"mode": "centroid_lite_token_index", "admission_debug": admission})
    results.append(grid_admission_summary(
        mode="centroid_lite_token_index",
        args=mode_args,
        index_phase=token_centroid_index_phase,
        admission=admission,
        qrels=qrels,
    ))

    for proxy_encoder in proxy_encoder_grid:
        proxy_centroids = "kmeans" if proxy_encoder == "centroid_mean" else args.multivector_centroids
        for pooling in pooling_grid:
            pooling_mode = str(pooling["mode"])
            pooling_ratio = float(pooling["ratio"])
            document_args = clone_args(
                args,
                multivector_graph="document_nodes",
                reuse_index=False,
                multivector_token_pooling=pooling_mode,
                multivector_token_pooling_target_ratio=pooling_ratio,
                multivector_proxy_encoder=proxy_encoder,
                multivector_centroids=proxy_centroids,
                multivector_centroid_count=args.multivector_centroid_count,
            )
            document_index_phase = build_index(conn, document_args)
            for storage in storage_grid:
                for cache_mode in cache_grid:
                    for ef in ef_grid:
                        for oversampling in oversampling_grid:
                            mode_args = clone_args(
                                args,
                                multivector_graph="document_nodes",
                                multivector_candidate_source="document_nodes",
                                multivector_plain_fallback="off",
                                multivector_candidate_reservoirs="off",
                                multivector_doc_storage=storage,
                                multivector_doc_storage_cache=cache_mode,
                                multivector_token_pooling=pooling_mode,
                                multivector_token_pooling_target_ratio=pooling_ratio,
                                multivector_proxy_encoder=proxy_encoder,
                                multivector_centroids=proxy_centroids,
                                multivector_centroid_count=args.multivector_centroid_count,
                                multivector_doc_graph_search_ef=ef,
                                multivector_doc_graph_oversampling=oversampling,
                                multivector_doc_graph_rescore_k=0,
                            )
                            admission = run_admission_debug(conn, mode_args, queries)
                            ratio_label = str(pooling_ratio).replace(".", "_")
                            mode_name = (
                                f"document_nodes_{storage}_{cache_mode}_{proxy_encoder}_"
                                f"{pooling_mode}_r{ratio_label}_ef{ef}_os{oversampling}"
                            )
                            full_admission.append({"mode": mode_name, "admission_debug": admission})
                            results.append(grid_admission_summary(
                                mode=mode_name,
                                args=mode_args,
                                index_phase=document_index_phase,
                                admission=admission,
                                qrels=qrels,
                            ))
            if (
                not default_document_modes_done
                and proxy_encoder == args.multivector_proxy_encoder
                and pooling_mode == "off"
                and pooling_ratio == 1.0
            ):
                run_default_document_modes(document_index_phase, reused_grid_index=True)

    if not default_document_modes_done:
        default_document_args = clone_args(
            args,
            multivector_graph="document_nodes",
            reuse_index=False,
            multivector_token_pooling="off",
            multivector_token_pooling_target_ratio=1.0,
            multivector_proxy_encoder=args.multivector_proxy_encoder,
        )
        default_document_index_phase = build_index(conn, default_document_args)
        run_default_document_modes(default_document_index_phase, reused_grid_index=False)

    document_centroid_args = clone_args(
        args,
        multivector_graph="document_nodes",
        reuse_index=False,
        multivector_centroids="kmeans",
        multivector_centroid_count=args.multivector_centroid_count,
    )
    document_centroid_index_phase = build_index(conn, document_centroid_args)
    mode_args = grid_mode_args(document_centroid_args, "centroid_lite", "document_nodes")
    admission = run_admission_debug(conn, mode_args, queries)
    full_admission.append({"mode": "centroid_lite_document_index", "admission_debug": admission})
    results.append(grid_admission_summary(
        mode="centroid_lite_document_index",
        args=mode_args,
        index_phase=document_centroid_index_phase,
        admission=admission,
        qrels=qrels,
    ))

    return {
        "enabled": True,
        "rebuilds_index": True,
        "budget_sweep": budget_grid,
        "storage_grid": storage_grid,
        "cache_grid": cache_grid,
        "pooling_grid": pooling_grid,
        "proxy_encoder_grid": proxy_encoder_grid,
        "ef_grid": ef_grid,
        "oversampling_grid": oversampling_grid,
        "index_graph_m": args.index_graph_m,
        "index_graph_ef_construction": args.index_graph_ef_construction,
        "index_graph_ef_search": args.index_graph_ef_search,
        "index_native_segments": args.index_native_segments,
        "default_document_modes_reused_grid_index": default_document_modes_reused_grid_index,
        "modes": list(DOCUMENT_NODE_BASELINE_MODES),
        "queries": len(queries),
        "results": results,
        "admission_debug_runs": full_admission,
    }


def hybrid_evaluation_mode_args(args: argparse.Namespace, mode: str) -> argparse.Namespace:
    common = {
        "multivector_graph": "document_nodes",
        "multivector_plain_fallback": "off",
        "multivector_candidate_reservoirs": "off",
        "multivector_doc_storage_cache": args.multivector_doc_storage_cache,
        "multivector_doc_graph_search_ef": args.multivector_doc_graph_search_ef,
        "multivector_doc_graph_oversampling": args.multivector_doc_graph_oversampling,
        "multivector_doc_graph_rescore_k": args.multivector_doc_graph_rescore_k,
        "multivector_exact_rerank": args.multivector_exact_rerank,
        "multivector_exact_rerank_k": args.multivector_exact_rerank_k,
    }
    if mode == "exact_scan":
        return clone_args(args, **common, multivector_candidate_source="exact_doc_scan")
    if mode == "document_nodes":
        return clone_args(
            args,
            **common,
            multivector_candidate_source="document_nodes",
            multivector_bm25_candidate_injection="off",
            multivector_sparse_candidate_source="off",
        )
    if mode == "document_nodes_bm25_admission":
        return clone_args(
            args,
            **common,
            multivector_candidate_source="document_nodes",
            multivector_bm25_candidate_injection="dense_with_text",
            multivector_sparse_candidate_source="bm25",
        )
    if mode in {"document_nodes_bm25_rrf", "document_nodes_bm25_dbsf"}:
        return clone_args(
            args,
            **common,
            multivector_candidate_source="document_nodes",
            multivector_bm25_candidate_injection="hybrid_only",
            multivector_sparse_candidate_source="bm25",
        )
    if mode == "proxy_vector_document_nodes":
        return clone_args(
            args,
            **common,
            multivector_candidate_source="proxy_vector",
            multivector_bm25_candidate_injection="off",
            multivector_sparse_candidate_source="off",
        )
    if mode == "learned_sparse_exact_maxsim":
        return clone_args(
            args,
            **common,
            multivector_candidate_source="document_nodes",
            multivector_bm25_candidate_injection="off",
            multivector_sparse_candidate_source="learned_sparse",
        )
    if mode == "quantized_inverted_experimental":
        return clone_args(
            args,
            **common,
            multivector_candidate_source="quantized_inverted_experimental",
            multivector_bm25_candidate_injection="off",
            multivector_sparse_candidate_source="off",
        )
    raise ValueError(f"unsupported hybrid evaluation mode: {mode}")


def set_hybrid_mode_gucs(
    conn: psycopg.Connection[Any],
    mode: str,
    mode_args: argparse.Namespace,
) -> None:
    set_retrieval_gucs(conn, mode_args, f"dbpedia_colbert_hybrid_{mode}")
    branch_plan = "qdrant_like" if mode in {
        "document_nodes_bm25_rrf",
        "document_nodes_bm25_dbsf",
        "proxy_vector_document_nodes",
    } else "dense_only"
    exec_sql(conn, "SELECT set_config('turbohybrid.multivector_branch_plan', %s, false)", (branch_plan,))
    exec_sql(conn, "SELECT set_config('turbohybrid.multivector_debug_admission', 'trace', false)")
    exec_sql(
        conn,
        "SELECT set_config('turbohybrid.multivector_debug_trace_limit', %s, false)",
        (str(min(mode_args.admission_trace_limit, 1000)),),
    )
    if mode == "document_nodes_bm25_dbsf":
        exec_sql(
            conn,
            "SELECT set_config('turbohybrid.dbsf_min_branch_candidates', %s, false)",
            (str(max(1, getattr(mode_args, "hybrid_evaluation_dbsf_min_branch_candidates", 1))),),
        )


def run_hybrid_mode_query(
    conn: psycopg.Connection[Any],
    mode: str,
    query: QueryItem,
    args: argparse.Namespace,
    final_k: int,
) -> list[str]:
    if mode == "exact_scan":
        return run_retrieval_query(conn, EXACT_SCAN_METHOD, query, args, final_k)

    if mode in {"document_nodes", "proxy_vector_document_nodes", "quantized_inverted_experimental"}:
        return run_retrieval_query(conn, QUERY_ONLY_METHOD, query, args, final_k)

    if mode in {"document_nodes_bm25_admission", "learned_sparse_exact_maxsim"}:
        text_query_expr = (
            learned_sparse_text_query_sql()
            if mode == "learned_sparse_exact_maxsim"
            else "websearch_to_tsquery('simple', q.query_text)"
        )
        rows = fetch_all(
            conn,
            f"""
            SELECT d.doc_id
            FROM dbpedia_colbert_docs d
            WHERE d.colbert IS NOT NULL
            ORDER BY d.colbert <~> (
              SELECT turbohybrid_query(
                multivector_query => q.colbert,
                text_query => {text_query_expr},
                dense_k => %s,
                bm25_k => %s,
                final_k => %s,
                bm25_weight => 0
              )
              FROM dbpedia_colbert_queries q
              WHERE q.query_id = %s
            )
            LIMIT %s
            """,
            (args.dense_k, args.bm25_k, final_k, query.query_id, final_k),
        )
        return [str(row[0]) for row in rows]

    fusion = "rrf" if mode == "document_nodes_bm25_rrf" else "dbsf"
    rows = fetch_all(
        conn,
        """
        SELECT d.doc_id
        FROM dbpedia_colbert_docs d
        WHERE d.colbert IS NOT NULL
        ORDER BY d.colbert <~> (
          SELECT turbohybrid_query(
            multivector_query => q.colbert,
            text_query => websearch_to_tsquery('simple', q.query_text),
            fusion => %s,
            dense_k => %s,
            bm25_k => %s,
            rrf_k => %s,
            final_k => %s
          )
          FROM dbpedia_colbert_queries q
          WHERE q.query_id = %s
        )
        LIMIT %s
        """,
        (fusion, args.dense_k, args.bm25_k, args.rrf_k, final_k, query.query_id, final_k),
    )
    return [str(row[0]) for row in rows]


def admission_from_trace_or_results(
    exact_top: list[dict[str, Any]],
    result_docs: list[str],
    stats: dict[str, Any],
    *,
    candidate_source: Any = None,
    graph_mode: Any = None,
    plain_fallback: Any = None,
    exact_scan: bool = False,
) -> dict[str, Any]:
    if exact_scan:
        return {
            "exact_top1_admission": True,
            "exact_top10_admission_recall": 1.0 if exact_top else 0.0,
            "candidate_failures": [],
        }

    trace = stats.get("multivector_admission_trace")
    trace_entries = trace if isinstance(trace, list) else []
    trace_by_heap = {
        key: entry
        for entry in trace_entries
        if isinstance(entry, dict) and (key := trace_key(entry)) is not None
    }
    infer_from_results = (
        not trace_by_heap
        and can_infer_admission_from_result_docs(
            candidate_source or stats.get("multivector_candidate_source"),
            fallback_used=bool(stats.get("multivector_plain_fallback_used")),
            plain_fallback=plain_fallback,
            graph_mode=graph_mode or stats.get("multivector_graph_mode"),
        )
    )
    candidate_failures: list[str] = []
    admitted = 0
    top = exact_top[: min(10, len(exact_top))]
    for item in top:
        key = (int(item["heap_block"]), int(item["heap_offset"]))
        is_admitted = key in trace_by_heap or (infer_from_results and item["doc_id"] in result_docs)
        if is_admitted:
            admitted += 1
        else:
            candidate_failures.append(item["doc_id"])
    return {
        "exact_top1_admission": bool(top and not candidate_failures[:1]),
        "exact_top10_admission_recall": round(admitted / len(top), 6) if top else 0.0,
        "admission_inferred_from_result_docs": infer_from_results,
        "candidate_failures": candidate_failures,
    }


def hybrid_scan_record(stats: dict[str, Any]) -> dict[str, Any]:
    branch_latency = scan_stat_int_list(stats, "branch_latency_us")
    branch_candidates = scan_stat_int_list(stats, "branch_candidate_counts")
    return {
        "branch_kinds": stats.get("branch_kinds", []),
        "branch_candidate_counts": branch_candidates,
        "branch_latency_us": branch_latency,
        "branch_latency_us_total": sum(branch_latency),
        "branch_candidates_total": sum(branch_candidates),
        "branch_fusion_mode": scan_stat_str(stats, "branch_fusion_mode"),
        "fusion_strategy": scan_stat_str(stats, "fusion_strategy"),
        "docs_scored": scan_docs_scored(stats),
        "exact_maxsim_pairs": scan_stat_int(stats, "multivector_exact_rerank_pairs"),
        "exact_rerank_docs": scan_exact_rerank_docs(stats),
        "sidecar_bytes": sidecar_stats_from_scan(stats).get("sidecar_bytes_touched", 0),
        "native_exact_bytes": sidecar_stats_from_scan(stats).get("native_cache_exact_bytes", 0),
        **exact_rerank_work_from_stats(stats),
        **proxy_work_from_stats(stats),
        **centroid_work_from_stats(stats),
        **quantized_inverted_work_from_stats(stats),
        **learned_sparse_work_from_stats(stats),
    }


def summarize_hybrid_scan_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "branch_latency_us_total": summarize_ints([int(item.get("branch_latency_us_total", 0)) for item in records]),
        "branch_candidates_total": summarize_ints([int(item.get("branch_candidates_total", 0)) for item in records]),
        "docs_scored": summarize_ints([int(item.get("docs_scored", 0)) for item in records]),
        "exact_maxsim_pairs": summarize_ints([int(item.get("exact_maxsim_pairs", 0)) for item in records]),
        "exact_rerank_docs": summarize_ints([int(item.get("exact_rerank_docs", 0)) for item in records]),
        "exact_rerank_pairs_saved": summarize_ints([int(item.get("exact_rerank_pairs_saved", 0)) for item in records]),
        "sidecar_bytes": summarize_ints([int(item.get("sidecar_bytes", 0)) for item in records]),
        "native_exact_bytes": summarize_ints([int(item.get("native_exact_bytes", 0)) for item in records]),
        "learned_sparse_candidates": summarize_ints([int(item.get("learned_sparse_candidates", 0)) for item in records]),
        "proxy_candidates": summarize_ints([int(item.get("proxy_candidates", 0)) for item in records]),
        "centroid_candidates": summarize_ints([int(item.get("centroid_candidates", 0)) for item in records]),
        "centroid_docs_touched": summarize_ints([int(item.get("centroid_docs_touched", 0)) for item in records]),
        "quantized_inverted_candidates": summarize_ints([int(item.get("quantized_inverted_candidates", 0)) for item in records]),
        "quantized_inverted_postings_touched": summarize_ints([int(item.get("quantized_inverted_postings_touched", 0)) for item in records]),
        "quantized_inverted_docs_scored": summarize_ints([int(item.get("quantized_inverted_docs_scored", 0)) for item in records]),
    }


def hybrid_mode_recommendation(mode: str | None, reason: str) -> dict[str, Any]:
    candidate_source = "document_nodes"
    branch_plan = "dense_only"
    sparse_source = "off"
    bm25_injection = "off"
    fusion = None
    if mode == "exact_scan":
        candidate_source = "exact_doc_scan"
    elif mode == "proxy_vector_document_nodes":
        candidate_source = "proxy_vector"
        branch_plan = "qdrant_like"
    elif mode == "document_nodes_bm25_admission":
        sparse_source = "bm25"
        bm25_injection = "dense_with_text"
    elif mode == "document_nodes_bm25_rrf":
        sparse_source = "bm25"
        branch_plan = "qdrant_like"
        fusion = "rrf"
    elif mode == "document_nodes_bm25_dbsf":
        sparse_source = "bm25"
        branch_plan = "qdrant_like"
        fusion = "dbsf"
    elif mode == "learned_sparse_exact_maxsim":
        sparse_source = "learned_sparse"
    elif mode == "quantized_inverted_experimental":
        candidate_source = "quantized_inverted_experimental"

    gucs: dict[str, str] = {
        "turbohybrid.multivector_candidate_source": candidate_source,
        "turbohybrid.multivector_branch_plan": branch_plan,
        "turbohybrid.multivector_sparse_candidate_source": sparse_source,
        "turbohybrid.multivector_bm25_candidate_injection": bm25_injection,
    }
    if fusion is not None:
        gucs["fusion"] = fusion
    return {
        "mode": mode,
        "reason": reason,
        "index_options": {
            "multivector_graph": "document_nodes",
        },
        "gucs": gucs,
    }


def hybrid_profile_recommendations(
    mode_results: list[dict[str, Any]],
    best_quality: dict[str, Any] | None,
    best_latency_under_floor: dict[str, Any] | None,
) -> dict[str, Any]:
    latency = min(
        mode_results,
        key=lambda item: float(item.get("latency", {}).get("p50_ms", float("inf"))),
        default=None,
    )
    high_recall = max(
        mode_results,
        key=lambda item: (
            float(item.get("exact_top10_admission_recall", 0.0)),
            float(item.get("exact_top1_admission_rate", 0.0)),
            float(item.get("metrics", {}).get("ndcg@10", 0.0)),
            -float(item.get("latency", {}).get("p50_ms", float("inf"))),
        ),
        default=None,
    )
    return {
        "latency": hybrid_mode_recommendation(
            latency.get("mode") if latency else None,
            "lowest p50 latency across evaluated modes",
        ),
        "balanced": hybrid_mode_recommendation(
            best_latency_under_floor.get("mode") if best_latency_under_floor else None,
            "lowest p50 latency among modes within the configured relative NDCG floor",
        ),
        "quality": hybrid_mode_recommendation(
            best_quality.get("mode") if best_quality else None,
            "highest NDCG@10 with recall and admission recall as tie breakers",
        ),
        "high_recall": hybrid_mode_recommendation(
            high_recall.get("mode") if high_recall else None,
            "highest exact top-10 admission recall with top-1 admission and NDCG tie breakers",
        ),
    }


def run_hybrid_evaluation_harness(
    conn: psycopg.Connection[Any],
    args: argparse.Namespace,
    queries: list[QueryItem],
    qrels: dict[str, dict[str, int]],
) -> dict[str, Any]:
    modes = [
        mode.strip()
        for mode in args.hybrid_evaluation_modes.split(",")
        if mode.strip()
    ]
    unknown = sorted(set(modes) - set(SUPPORTED_HYBRID_EVALUATION_MODES))
    if unknown:
        raise ValueError(f"unknown hybrid evaluation mode(s): {', '.join(unknown)}")

    document_index_args = clone_args(args, multivector_graph="document_nodes", reuse_index=False)
    index_phase = build_index(conn, document_index_args)
    exact_top_by_query = {
        query.query_id: exact_admission_top(conn, query, args.admission_k)
        for query in queries
    }

    mode_results: list[dict[str, Any]] = []
    for mode in modes:
        mode_args = hybrid_evaluation_mode_args(args, mode)
        set_hybrid_mode_gucs(conn, mode, mode_args)
        run: dict[str, list[str]] = {}
        latencies: list[float] = []
        scan_records: list[dict[str, Any]] = []
        top1_admissions: list[bool] = []
        top10_admissions: list[float] = []
        inferred_admissions = 0
        candidate_failures: dict[str, list[str]] = {}

        for query in queries:
            started = time.perf_counter()
            docs = run_hybrid_mode_query(conn, mode, query, mode_args, args.final_k)
            latency_ms = (time.perf_counter() - started) * 1000.0
            stats = last_scan_stats(conn) if mode != "exact_scan" else {}
            admission = admission_from_trace_or_results(
                exact_top_by_query.get(query.query_id, []),
                docs,
                stats,
                candidate_source=mode_args.multivector_candidate_source,
                graph_mode=mode_args.multivector_graph,
                plain_fallback=mode_args.multivector_plain_fallback,
                exact_scan=(mode == "exact_scan"),
            )
            run[query.query_id] = docs
            latencies.append(latency_ms)
            scan_records.append(hybrid_scan_record(stats) if stats else {})
            top1_admissions.append(bool(admission["exact_top1_admission"]))
            top10_admissions.append(float(admission["exact_top10_admission_recall"]))
            if admission.get("admission_inferred_from_result_docs"):
                inferred_admissions += 1
            failures = admission.get("candidate_failures", [])
            if isinstance(failures, list) and failures:
                candidate_failures[query.query_id] = [str(item) for item in failures]

        metrics = method_metrics(run, qrels, args.final_k, args.quality_k) if qrels else {}
        mode_results.append({
            "mode": mode,
            "metrics": metrics,
            "latency": summarize_ms(latencies),
            "exact_top1_admission_rate": round(sum(1 for item in top1_admissions if item) / len(top1_admissions), 6)
            if top1_admissions else 0.0,
            "exact_top10_admission_recall": round(statistics.mean(top10_admissions), 6)
            if top10_admissions else 0.0,
            "scan_work": summarize_hybrid_scan_records(scan_records),
            "admission_inferred_from_result_docs_queries": inferred_admissions,
            "candidate_admission_failures": candidate_failures,
            "failure_query_count": len(candidate_failures),
            "top10_by_query": {qid: docs[:10] for qid, docs in run.items()},
            "first_scan": scan_records[0] if scan_records else {},
            "index_stats": index_phase.get("index_stats", {}),
        })

    best_quality = max(
        mode_results,
        key=lambda item: (
            float(item.get("metrics", {}).get("ndcg@10", 0.0)),
            float(item.get("metrics", {}).get("recall@10", 0.0)),
            float(item.get("exact_top10_admission_recall", 0.0)),
        ),
        default=None,
    )
    best_ndcg = float(best_quality.get("metrics", {}).get("ndcg@10", 0.0)) if best_quality else 0.0
    floor = best_ndcg * float(args.hybrid_evaluation_quality_floor)
    eligible = [
        item for item in mode_results
        if float(item.get("metrics", {}).get("ndcg@10", 0.0)) >= floor
    ] or mode_results
    best_latency = min(
        eligible,
        key=lambda item: float(item.get("latency", {}).get("p50_ms", float("inf"))),
        default=None,
    )
    profile_recommendations = hybrid_profile_recommendations(
        mode_results,
        best_quality,
        best_latency,
    )
    return {
        "enabled": True,
        "rebuilds_index": True,
        "modes": modes,
        "queries": len(queries),
        "quality_floor_relative_to_best_ndcg": args.hybrid_evaluation_quality_floor,
        "index_phase": index_phase,
        "results": mode_results,
        "best_quality_mode": best_quality.get("mode") if best_quality else None,
        "best_latency_under_quality_floor_mode": best_latency.get("mode") if best_latency else None,
        "recommended_default_profile": profile_recommendations["balanced"],
        "profile_recommendations": profile_recommendations,
    }


def run_token_ablation(
    conn: psycopg.Connection[Any],
    args: argparse.Namespace,
    queries: list[QueryItem],
    qrels: dict[str, dict[str, int]],
) -> dict[str, Any] | None:
    if not args.token_ablation_query_id:
        return None

    query = next((item for item in queries if item.query_id == args.token_ablation_query_id), None)
    if query is None:
        available = ", ".join(item.query_id for item in queries[:10])
        raise ValueError(
            f"--token-ablation-query-id {args.token_ablation_query_id!r} was not loaded; "
            f"first loaded query ids: {available}"
        )

    final_k = args.token_ablation_final_k or args.final_k
    dense_k = args.token_ablation_dense_k or args.dense_k
    relevant = qrels.get(query.query_id, {})

    set_retrieval_gucs(conn, args, "dbpedia_colbert_token_ablation")
    exec_sql(conn, "SELECT set_config('turbohybrid.multivector_debug_admission', 'summary', false)")

    def run_variant(label: str, skip_tokens: str) -> dict[str, Any]:
        exec_sql(
            conn,
            "SELECT set_config('turbohybrid.multivector_debug_skip_query_tokens', %s, false)",
            (skip_tokens,),
        )
        started = time.perf_counter()
        docs = run_retrieval_query(
            conn,
            QUERY_ONLY_METHOD,
            query,
            args,
            final_k,
            dense_k=dense_k,
        )
        latency_ms = (time.perf_counter() - started) * 1000.0
        stats = last_scan_stats(conn)
        hits = [doc_id for doc_id in docs[:final_k] if relevant.get(doc_id, 0) > 0]
        token_stats = stats.get("multivector_query_token_stats")
        return {
            "label": label,
            "skip_tokens": skip_tokens,
            "latency_ms": round(latency_ms, 3),
            "top_docs": docs,
            "relevant_hits": hits,
            "recall_at_k": round(len(hits) / len(relevant), 6) if relevant else None,
            "scan_stats": stats,
            "query_token_stats": token_stats if isinstance(token_stats, list) else [],
        }

    variants = [run_variant("baseline", "")]
    if args.token_ablation_skip_tokens:
        variants.append(run_variant("skip_tokens", args.token_ablation_skip_tokens))

    exec_sql(conn, "SELECT set_config('turbohybrid.multivector_debug_skip_query_tokens', '', false)")
    exec_sql(conn, "SELECT set_config('turbohybrid.multivector_debug_admission', 'off', false)")
    return {
        "enabled": True,
        "query_id": query.query_id,
        "query_text": query.query_text,
        "final_k": final_k,
        "dense_k": dense_k,
        "relevant_docs": len(relevant),
        "variants": variants,
    }


def run_serial_retrieval(
    conn: psycopg.Connection[Any],
    method: str,
    queries: list[QueryItem],
    args: argparse.Namespace,
    final_k: int,
) -> tuple[dict[str, list[str]], list[float], list[dict[str, Any]]]:
    run: dict[str, list[str]] = {}
    latencies: list[float] = []
    scan_stats: list[dict[str, Any]] = []
    for query in queries:
        started = time.perf_counter()
        run[query.query_id] = run_retrieval_query(conn, method, query, args, final_k)
        latencies.append((time.perf_counter() - started) * 1000.0)
        scan_stats.append(last_scan_stats(conn) if uses_turbohybrid_index(method) else {})
    return run, latencies, scan_stats


def run_worker(
    *,
    args: argparse.Namespace,
    method: str,
    queries: list[QueryItem],
    client_id: int,
    ready_barrier: threading.Barrier,
    timed_barrier: threading.Barrier,
    result: WorkerResult,
) -> None:
    try:
        conn = connect(args)
        try:
            set_retrieval_gucs(conn, args, f"dbpedia_colbert_{method}_{client_id}")
            ready_barrier.wait()
            for i in range(args.warm_queries):
                query = queries[(client_id + i * args.clients) % len(queries)]
                started = time.perf_counter()
                run_retrieval_query(conn, method, query, args, args.final_k)
                result.warm_durations_ms.append((time.perf_counter() - started) * 1000.0)

            timed_barrier.wait()
            timed_started = time.perf_counter()
            for i in range(args.timed_queries):
                query = queries[(client_id + i * args.clients) % len(queries)]
                started = time.perf_counter()
                run_retrieval_query(conn, method, query, args, args.final_k)
                result.timed_durations_ms.append((time.perf_counter() - started) * 1000.0)
                if i == 0 and uses_turbohybrid_index(method):
                    result.first_timed_stats = last_scan_stats(conn)
            result.timed_wall_ms = (time.perf_counter() - timed_started) * 1000.0
            if args.timed_queries > 0 and uses_turbohybrid_index(method):
                result.last_timed_stats = last_scan_stats(conn)
        finally:
            conn.close()
    except Exception as exc:  # pragma: no cover - benchmark error path
        result.error = str(exc)
        for barrier in (ready_barrier, timed_barrier):
            try:
                barrier.abort()
            except Exception:
                pass


def run_parallel_retrieval(args: argparse.Namespace, method: str, queries: list[QueryItem]) -> dict[str, Any]:
    ready_barrier = threading.Barrier(args.clients)
    timed_barrier = threading.Barrier(args.clients)
    results = [WorkerResult(client_id=i) for i in range(args.clients)]
    threads = [
        threading.Thread(
            target=run_worker,
            kwargs={
                "args": args,
                "method": method,
                "queries": queries,
                "client_id": i,
                "ready_barrier": ready_barrier,
                "timed_barrier": timed_barrier,
                "result": results[i],
            },
            daemon=True,
        )
        for i in range(args.clients)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    errors = [result.error for result in results if result.error]
    if errors:
        raise RuntimeError("; ".join(errors))
    timed = [value for result in results for value in result.timed_durations_ms]
    warm = [value for result in results for value in result.warm_durations_ms]
    wall_ms = max((result.timed_wall_ms for result in results), default=0.0)
    total_queries = len(timed)
    return {
        "clients": args.clients,
        "warm_queries_per_client": args.warm_queries,
        "timed_queries_per_client": args.timed_queries,
        "total_timed_queries": total_queries,
        "warm_latency": summarize_ms(warm),
        "timed_latency": summarize_ms(timed),
        "timed_wall_ms": round(wall_ms, 3),
        "throughput_qps": round((total_queries * 1000.0 / wall_ms), 3) if wall_ms > 0 else 0.0,
        "first_timed_scan_stats": results[0].first_timed_stats if results else None,
        "last_timed_scan_stats": results[0].last_timed_stats if results else None,
    }


def scan_stat_samples(scan_stats: list[dict[str, Any]], key: str) -> list[int]:
    return [scan_stat_int(stats, key) for stats in scan_stats if key in stats]


def max_scan_work_value(scan_stats: list[dict[str, Any]], keys: tuple[str, ...]) -> int:
    values: list[int] = []
    for stats in scan_stats:
        values.append(max(scan_stat_int(stats, key) for key in keys))
    return max(values, default=0)


def parallel_safety_preflight(
    args: argparse.Namespace,
    method: str,
    serial_latencies: list[float],
    serial_stats: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return a skip decision when the serial probe is already pathological."""
    decision: dict[str, Any] = {
        "enabled": args.parallel_safety_preflight != "off",
        "forced": bool(args.force_parallel_retrieval),
        "method": method,
        "skipped": False,
        "reasons": [],
        "thresholds": {
            "max_serial_p95_ms": args.parallel_safety_max_serial_p95_ms,
            "max_docs_scored": args.parallel_safety_max_docs_scored,
            "max_exact_rerank_docs": args.parallel_safety_max_exact_rerank_docs,
            "max_exact_pairs": args.parallel_safety_max_exact_pairs,
            "max_sidecar_bytes": args.parallel_safety_max_sidecar_bytes,
        },
        "observed": {
            "serial_latency": summarize_ms(serial_latencies),
            "docs_scored_max": max(scan_docs_scored(stats) for stats in serial_stats) if serial_stats else 0,
            "exact_rerank_docs_max": max(scan_exact_rerank_docs(stats) for stats in serial_stats) if serial_stats else 0,
            "exact_pairs_max": max_scan_work_value(serial_stats, ("multivector_exact_rerank_pairs",)),
            "sidecar_bytes_max": max_scan_work_value(serial_stats, ("multivector_doc_sidecar_bytes_touched",)),
        },
    }
    if args.parallel_safety_preflight == "off" or args.force_parallel_retrieval:
        return decision

    reasons: list[str] = []
    observed = decision["observed"]
    latency = observed.get("serial_latency", {})
    p95_ms = float(latency.get("p95_ms", 0.0)) if isinstance(latency, dict) else 0.0
    if p95_ms > args.parallel_safety_max_serial_p95_ms:
        reasons.append(
            "serial p95 {observed:.3f} ms exceeds {limit:.3f} ms".format(
                observed=p95_ms,
                limit=args.parallel_safety_max_serial_p95_ms,
            )
        )
    if int(observed["docs_scored_max"]) > args.parallel_safety_max_docs_scored:
        reasons.append(
            "docs scored {observed} exceeds {limit}".format(
                observed=observed["docs_scored_max"],
                limit=args.parallel_safety_max_docs_scored,
            )
        )
    if int(observed["exact_rerank_docs_max"]) > args.parallel_safety_max_exact_rerank_docs:
        reasons.append(
            "exact rerank docs {observed} exceeds {limit}".format(
                observed=observed["exact_rerank_docs_max"],
                limit=args.parallel_safety_max_exact_rerank_docs,
            )
        )
    if int(observed["exact_pairs_max"]) > args.parallel_safety_max_exact_pairs:
        reasons.append(
            "exact MaxSim pairs {observed} exceeds {limit}".format(
                observed=observed["exact_pairs_max"],
                limit=args.parallel_safety_max_exact_pairs,
            )
        )
    if int(observed["sidecar_bytes_max"]) > args.parallel_safety_max_sidecar_bytes:
        reasons.append(
            "sidecar bytes {observed} exceeds {limit}".format(
                observed=observed["sidecar_bytes_max"],
                limit=args.parallel_safety_max_sidecar_bytes,
            )
        )

    if reasons:
        decision["skipped"] = True
        decision["reasons"] = reasons
    return decision


def run_parallel_retrieval_report(
    args: argparse.Namespace,
    method: str,
    queries: list[QueryItem],
    serial_latencies: list[float],
    serial_stats: list[dict[str, Any]],
) -> dict[str, Any]:
    preflight = parallel_safety_preflight(args, method, serial_latencies, serial_stats)
    if args.skip_parallel_retrieval:
        return {
            "clients": args.clients,
            "warm_queries_per_client": args.warm_queries,
            "timed_queries_per_client": args.timed_queries,
            "total_timed_queries": 0,
            "failed": False,
            "skipped": True,
            "skip_reason": "disabled by --skip-parallel-retrieval",
            "safety_preflight": preflight,
        }
    if preflight["skipped"]:
        return {
            "clients": args.clients,
            "warm_queries_per_client": args.warm_queries,
            "timed_queries_per_client": args.timed_queries,
            "total_timed_queries": 0,
            "failed": False,
            "skipped": True,
            "skip_reason": "; ".join(preflight["reasons"]),
            "safety_preflight": preflight,
        }
    try:
        parallel = run_parallel_retrieval(args, method, queries)
    except Exception as exc:  # pragma: no cover - host benchmark failure path
        return {
            "clients": args.clients,
            "warm_queries_per_client": args.warm_queries,
            "timed_queries_per_client": args.timed_queries,
            "total_timed_queries": 0,
            "failed": True,
            "skipped": False,
            "error": str(exc),
            "safety_preflight": preflight,
        }

    parallel["failed"] = False
    parallel["skipped"] = False
    parallel["safety_preflight"] = preflight
    return parallel


def method_metrics(
    run: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    final_k: int,
    quality_k: int,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for k in sorted({10, final_k, quality_k}):
        if k <= 0 or k > final_k:
            continue
        metrics.update(metrics_for_run(run, qrels, k))
    return metrics


def scan_summary(scan_stats: list[dict[str, Any]]) -> dict[str, Any]:
    if not scan_stats:
        return {}
    first = scan_stats[0]
    last = scan_stats[-1]
    return {
        "first": first,
        "last": last,
        "runs": len(scan_stats),
        "index_used_runs": sum(1 for stats in scan_stats if stats.get("index_used") is True),
        "multivector_enabled_runs": sum(1 for stats in scan_stats if stats.get("multivector_enabled") is True),
    }


def summarize_document_node_latency_scan_stats(
    scan_stats: list[dict[str, Any]]
) -> dict[str, Any]:
    base = scan_summary(scan_stats)
    extracted_samples = [
        extract_document_node_serving_stats(stats)
        for stats in scan_stats
        if isinstance(stats, dict)
    ]
    field_summary: dict[str, Any] = {}
    for key in sorted({key for sample in extracted_samples for key in sample}):
        values = [sample[key] for sample in extracted_samples if key in sample]
        numeric_values: list[float] = []
        integer_values: list[int] = []
        all_numeric = bool(values)
        all_integer = bool(values)
        for value in values:
            if isinstance(value, bool):
                all_numeric = False
                all_integer = False
                break
            if isinstance(value, int):
                numeric_values.append(float(value))
                integer_values.append(value)
                continue
            if isinstance(value, float):
                numeric_values.append(value)
                all_integer = False
                continue
            all_numeric = False
            all_integer = False
            break
        if all_integer and integer_values:
            field_summary[key] = summarize_ints(integer_values)
        elif all_numeric and numeric_values:
            field_summary[key] = {
                "mean": round(statistics.mean(numeric_values), 3),
                "min": round(min(numeric_values), 3),
                "p50": round(percentile(numeric_values, 50), 3),
                "p95": round(percentile(numeric_values, 95), 3),
                "max": round(max(numeric_values), 3),
                "count": len(numeric_values),
            }
        elif all(isinstance(value, bool) for value in values):
            field_summary[key] = {
                "true": sum(1 for value in values if value is True),
                "false": sum(1 for value in values if value is False),
                "count": len(values),
            }
        else:
            distinct = sorted({str(value) for value in values})
            field_summary[key] = {
                "first": values[0] if values else None,
                "last": values[-1] if values else None,
                "distinct": distinct[:20],
                "distinct_count": len(distinct),
                "count": len(values),
            }

    available = {
        group: any(
            key in sample
            for sample in extracted_samples
            for key in fields
        )
        for group, fields in SERVING_STATS_FIELD_GROUPS.items()
    }
    available["phase_timing"] = any(
        sample.get("phase_timing_source") not in (None, "", "none")
        for sample in extracted_samples
    )
    available["experimental_quantized"] = any(
        is_experimental_quantized_stat_key(str(key))
        for sample in extracted_samples
        for key in sample
    )
    return {
        **base,
        "first_selected": extracted_samples[0] if extracted_samples else {},
        "last_selected": extracted_samples[-1] if extracted_samples else {},
        "stats_available": dict(sorted(available.items())),
        "field_summary": field_summary,
    }


def run_serving_latency_query_loop(
    conn: psycopg.Connection[Any],
    args: argparse.Namespace,
    queries: list[QueryItem],
    profile: DocumentNodeServingProfile,
    *,
    retrieval_provider: Callable[
        [psycopg.Connection[Any], str, QueryItem, argparse.Namespace, int, int | None],
        list[str],
    ] = run_retrieval_query,
    stats_provider: Callable[[psycopg.Connection[Any]], dict[str, Any]] = last_scan_stats,
) -> dict[str, Any]:
    started = time.perf_counter()
    warmup_latencies_ms: list[float] = []
    warmup_scan_stats: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    scan_stats: list[dict[str, Any]] = []
    top_docs: dict[str, list[str]] = {}
    slow_path_warnings: list[str] = []
    dense_k = int(args.multivector_doc_candidate_k)
    warmup_query_count = max(0, int(getattr(args, "serving_latency_warmup_queries", 0) or 0))
    warmup_queries = queries[: min(warmup_query_count, len(queries))]
    for query in warmup_queries:
        query_started = time.perf_counter()
        retrieval_provider(
            conn,
            QUERY_ONLY_METHOD,
            query,
            args,
            args.final_k,
            dense_k,
        )
        latency_ms = elapsed_ms_since(query_started)
        warmup_latencies_ms.append(latency_ms)
        stats = stats_provider(conn)
        if isinstance(stats, dict):
            stats = {
                **stats,
                "observed_latency_ms": latency_ms,
                "serving_latency_warmup_query": True,
            }
        warmup_scan_stats.append(stats)
    measured_started = time.perf_counter()
    for query in queries:
        query_started = time.perf_counter()
        docs = retrieval_provider(
            conn,
            QUERY_ONLY_METHOD,
            query,
            args,
            args.final_k,
            dense_k,
        )
        latency_ms = elapsed_ms_since(query_started)
        latencies_ms.append(latency_ms)
        top_docs[query.query_id] = docs[: args.final_k]
        stats = stats_provider(conn)
        if isinstance(stats, dict):
            stats = {**stats, "observed_latency_ms": latency_ms}
        scan_stats.append(stats)
        for warning in validate_serving_scan_path(stats, args, profile):
            slow_path_warnings.append(f"{query.query_id}:{warning}")
    native_cache_rebuilds_after_warmup = sum(
        1
        for stats in scan_stats
        if scan_stat_bool(stats, "native_cache_built_this_scan")
    )
    if native_cache_rebuilds_after_warmup > 0:
        slow_path_warnings.append(
            "native_cache_built_repeatedly_after_warmup:"
            f"count={native_cache_rebuilds_after_warmup}"
        )
    total_elapsed_ms = elapsed_ms_since(started)
    measured_elapsed_ms = elapsed_ms_since(measured_started)
    latency = summarize_ms(latencies_ms)
    warmup_latency = summarize_ms(warmup_latencies_ms)
    warm_cache_reuse_count = sum(
        1 for stats in scan_stats if scan_stat_bool(stats, "native_cache_reused")
    )
    cache_build_query_count = sum(
        1
        for stats in (*warmup_scan_stats, *scan_stats)
        if scan_stat_bool(stats, "sidecar_cache_build_this_query")
        or scan_stat_bool(stats, "native_cache_built_this_scan")
    )
    unique_warnings = unique_preserve_order(slow_path_warnings)
    return {
        "total_elapsed_ms": total_elapsed_ms,
        "measured_elapsed_ms": measured_elapsed_ms,
        "warmup_latencies_ms": warmup_latencies_ms,
        "latencies_ms": latencies_ms,
        "warmup_latency": {
            **warmup_latency,
            "total_elapsed_ms": round(sum(warmup_latencies_ms), 3),
        },
        "latency": {
            **latency,
            "total_elapsed_ms": measured_elapsed_ms,
            "qps_total": round((len(queries) * 1000.0) / measured_elapsed_ms, 3)
            if measured_elapsed_ms > 0
            else 0.0,
        },
        "warmup_excluded": bool(warmup_queries),
        "warmup_query_count": len(warmup_queries),
        "query_count": len(queries),
        "retrieval_query_count": len(queries),
        "cache_build_queries": cache_build_query_count,
        "warm_queries": len(queries),
        "cache_reused_on_warm_queries": warm_cache_reuse_count,
        "exact_admission_baseline_calls": 0,
        "serving_slow_path_warnings": unique_warnings,
        "serving_slow_path_failed": bool(
            unique_warnings and getattr(args, "serving_fail_on_slow_path", False)
        ),
        "top_docs_by_query": top_docs,
        "warmup_scan_stats": warmup_scan_stats,
        "scan_stats": scan_stats,
        "warmup_scan_stats_summary": summarize_document_node_latency_scan_stats(
            warmup_scan_stats
        ),
        "scan_stats_summary": summarize_document_node_latency_scan_stats(scan_stats),
    }


def run_document_node_serving_latency_only(
    conn: psycopg.Connection[Any],
    args: argparse.Namespace,
    queries: list[QueryItem],
) -> dict[str, Any]:
    latency_args, profile, effective_rerank_k = document_node_serving_latency_args(args)
    index_phase = build_index(conn, latency_args)
    latency_args = clone_args(
        latency_args,
        serving_loaded_document_count=loaded_document_count(conn),
    )
    set_retrieval_gucs(conn, latency_args, "dbpedia_colbert_serving_latency_only")
    exec_sql(conn, "SELECT set_config('turbohybrid.multivector_debug_admission', 'off', false)")
    loop = run_serving_latency_query_loop(conn, latency_args, queries, profile)
    fail_on_serving_slow_path_if_requested(
        loop["serving_slow_path_warnings"],
        latency_args,
    )
    return {
        "enabled": True,
        "profile": profile.name,
            "settings": {
                "profile": profile.name,
                "candidate_source": profile.candidate_source,
                "branch_plan": profile.branch_plan,
                "bm25_candidate_injection": profile.bm25_candidate_injection,
                "sparse_candidate_source": profile.sparse_candidate_source,
                "graph_mode": "document_nodes",
                "proxy_encoder": profile.proxy_encoder,
            "centroids": profile.centroids,
            "centroid_count": profile.centroid_count,
            "storage_kind": profile.storage_kind,
            "storage_cache_mode": profile.cache_mode,
            "token_pooling": profile.token_pooling,
            "token_pooling_target_ratio": profile.token_pooling_target_ratio,
            "entry_sidecar": profile.entry_sidecar,
            "entry_sidecar_representatives": profile.entry_sidecar_representatives,
            "entry_sidecar_strategy": profile.entry_sidecar_strategy,
            "centroid_lite_max_postings_per_token": (
                profile.centroid_lite_max_postings_per_token
            ),
            "centroid_lite_pruning": profile.centroid_lite_pruning,
            "ef": int(args.serving_ef),
            "oversampling": int(args.serving_oversampling),
            "entry_sample_count": int(args.multivector_doc_graph_entry_sample_count),
            "candidate_k": int(args.serving_candidate_k),
            "requested_exact_rerank_k": int(args.serving_exact_rerank_k),
            "effective_exact_rerank_k": effective_rerank_k,
            "final_k": int(args.final_k),
            "warmup_queries": int(getattr(args, "serving_latency_warmup_queries", 0) or 0),
        },
        "total_elapsed_ms": loop["total_elapsed_ms"],
        "measured_elapsed_ms": loop["measured_elapsed_ms"],
        "warmup_excluded": loop["warmup_excluded"],
        "warmup_latency": loop["warmup_latency"],
        "latency": loop["latency"],
        "qps": loop["latency"].get("qps_total", 0.0),
        "cache_build_queries": loop["cache_build_queries"],
        "warm_queries": loop["warm_queries"],
        "cache_reused_on_warm_queries": loop["cache_reused_on_warm_queries"],
        "query_count": loop["query_count"],
        "retrieval_query_count": loop["retrieval_query_count"],
        "admission_metrics": {
            "available": False,
            "reason": "latency-only mode does not compute exact admission baselines",
        },
        "exact_admission_baseline_calls": 0,
        "serving_slow_path_warnings": loop["serving_slow_path_warnings"],
        "serving_slow_path_failed": loop["serving_slow_path_failed"],
        "warmup_scan_stats_summary": loop["warmup_scan_stats_summary"],
        "scan_stats_summary": loop["scan_stats_summary"],
        "top10_by_query": {
            query_id: docs[:10]
            for query_id, docs in loop["top_docs_by_query"].items()
        },
        "index_phase": index_phase,
    }


SYNTHETIC_GATE_QUERY = """turbohybrid_multivector(ARRAY[
    '[1,0,0,0]'::vector,
    '[0,1,0,0]'::vector,
    '[0,0,1,0]'::vector,
    '[0,0,0,1]'::vector
])"""


def setup_multivector_recall_gate(conn: psycopg.Connection[Any]) -> None:
    exec_sql(conn, "CREATE EXTENSION IF NOT EXISTS vector")
    exec_sql(conn, "CREATE EXTENSION IF NOT EXISTS pgturbohybrid")
    load_pgturbohybrid_library(conn)
    exec_sql(conn, "DROP TABLE IF EXISTS mv_recall_gate_docs")
    exec_sql(
        conn,
        """
        CREATE TEMP TABLE mv_recall_gate_docs (
            id text PRIMARY KEY,
            colbert turbohybrid_multivector
        ) ON COMMIT PRESERVE ROWS
        """,
    )
    exec_sql(
        conn,
        """
        INSERT INTO mv_recall_gate_docs VALUES
          ('spike_1', turbohybrid_multivector(ARRAY['[1,0,0,0]'::vector])),
          ('spike_2', turbohybrid_multivector(ARRAY['[0,1,0,0]'::vector])),
          ('spike_3', turbohybrid_multivector(ARRAY['[0,0,1,0]'::vector])),
          ('spike_4', turbohybrid_multivector(ARRAY['[0,0,0,1]'::vector])),
          ('good', turbohybrid_multivector(ARRAY[
            '[0.8,0.6,0,0]'::vector,
            '[0.6,0.8,0,0]'::vector,
            '[0,0,0.8,0.6]'::vector,
            '[0,0,0.6,0.8]'::vector
          ]))
        """,
    )
    create_synthetic_gate_index(conn, "token_nodes")
    exec_sql(conn, "ANALYZE mv_recall_gate_docs")


def create_synthetic_gate_index(conn: psycopg.Connection[Any], graph_mode: str) -> None:
    if graph_mode not in {"token_nodes", "document_nodes"}:
        raise ValueError(f"unsupported synthetic gate graph mode: {graph_mode}")
    exec_sql(conn, "DROP INDEX IF EXISTS mv_recall_gate_docs_idx")
    exec_sql(
        conn,
        f"""
        CREATE INDEX mv_recall_gate_docs_idx ON mv_recall_gate_docs USING turbohybrid
          (colbert multivector_cosine_turbohybrid_ops)
          WITH (multivector_graph = {graph_mode},
                multivector_doc_build_scorer = proxy,
                exact_storage = off,
                graph_m = 4,
                graph_ef_construction = 8,
                graph_ef_search = 8)
        """,
    )


def synthetic_exact_top(conn: psycopg.Connection[Any], limit: int) -> list[dict[str, Any]]:
    rows = fetch_all(
        conn,
        f"""
        WITH q AS (SELECT {SYNTHETIC_GATE_QUERY} AS mv)
        SELECT d.id,
               d.ctid::text,
               turbohybrid_multivector_maxsim_distance(q.mv, d.colbert) AS distance
        FROM mv_recall_gate_docs d, q
        ORDER BY turbohybrid_multivector_maxsim_distance(q.mv, d.colbert), d.id
        LIMIT %s
        """,
        (limit,),
    )
    exact: list[dict[str, Any]] = []
    for rank, row in enumerate(rows, start=1):
        block, offset = tid_parts(str(row[1]))
        exact.append({
            "rank": rank,
            "doc_id": str(row[0]),
            "heap_block": block,
            "heap_offset": offset,
            "distance": float(row[2]),
            "maxsim": -float(row[2]),
        })
    return exact


def run_synthetic_exact_scan(conn: psycopg.Connection[Any], final_k: int) -> dict[str, Any]:
    started = time.perf_counter()
    rows = fetch_all(
        conn,
        f"""
        WITH q AS (SELECT {SYNTHETIC_GATE_QUERY} AS mv)
        SELECT d.id
        FROM mv_recall_gate_docs d, q
        ORDER BY turbohybrid_multivector_maxsim_distance(q.mv, d.colbert), d.id
        LIMIT %s
        """,
        (final_k,),
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    docs = [str(row[0]) for row in rows]
    return {
        "mode": "exact_scan",
        "candidate_source": "exact_scan",
        "result_doc_ids": docs,
        "top1": docs[0] if docs else None,
        "latency_ms": round(latency_ms, 3),
        "latency": summarize_ms([latency_ms]),
        "exact_top1_admitted_before_rerank": True,
        "exact_top10_admission_recall": 1.0,
        "doc_candidates": len(docs),
        "exact_rerank_docs": len(docs),
        "memory_bytes_estimate": 0,
        "scan_stats": {},
    }


def run_synthetic_gate_mode(
    conn: psycopg.Connection[Any],
    mode: dict[str, str],
    exact_top: list[dict[str, Any]],
    budget: int,
    final_k: int,
) -> dict[str, Any]:
    create_synthetic_gate_index(conn, mode.get("graph_mode", "token_nodes"))
    settings = {
        "enable_seqscan": "off",
        "jit": "off",
        "turbohybrid.multivector_subvector_k": "1",
        "turbohybrid.multivector_unique_docs_per_token": "1",
        "turbohybrid.multivector_max_raw_hits_per_token": "1",
        "turbohybrid.multivector_doc_candidate_k": str(budget),
        "turbohybrid.multivector_exact_rerank_k": str(max(budget, final_k, 10)),
        "turbohybrid.multivector_adaptive_widening": "off",
        "turbohybrid.multivector_candidate_source": mode["candidate_source"],
        "turbohybrid.multivector_plain_fallback": mode["plain_fallback"],
        "turbohybrid.multivector_candidate_reservoirs": mode["reservoirs"],
        "turbohybrid.multivector_debug_admission": "trace",
        "turbohybrid.multivector_debug_trace_limit": "1000",
    }
    for key, value in settings.items():
        exec_sql(conn, "SELECT set_config(%s, %s, false)", (key, value))

    started = time.perf_counter()
    rows = fetch_all(
        conn,
        f"""
        WITH q AS (SELECT {SYNTHETIC_GATE_QUERY} AS mv)
        SELECT d.id
        FROM mv_recall_gate_docs d, q
        ORDER BY d.colbert <~> turbohybrid_query(
            multivector_query => q.mv,
            dense_k => %s,
            final_k => %s
        )
        LIMIT %s
        """,
        (budget, final_k, final_k),
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    docs = [str(row[0]) for row in rows]
    stats = last_scan_stats(conn)
    trace = stats.get("multivector_admission_trace")
    trace_entries = trace if isinstance(trace, list) else []
    trace_by_heap = {
        key: entry
        for entry in trace_entries
        if isinstance(entry, dict) and (key := trace_key(entry)) is not None
    }
    inferred_exact_doc_admission = (
        not trace_by_heap
        and can_infer_admission_from_result_docs(
            mode["candidate_source"],
            plain_fallback=mode["plain_fallback"],
            graph_mode=mode.get("graph_mode"),
        )
    )
    exact_with_admission: list[dict[str, Any]] = []
    for exact in exact_top:
        key = (int(exact["heap_block"]), int(exact["heap_offset"]))
        entry = trace_by_heap.get(key)
        result_rank = docs.index(exact["doc_id"]) + 1 if exact["doc_id"] in docs else None
        admitted = entry is not None or (inferred_exact_doc_admission and result_rank is not None)
        admission_evidence = (
            "trace"
            if entry is not None
            else "result_doc"
            if inferred_exact_doc_admission and result_rank is not None
            else "missing"
        )
        exact_with_admission.append({
            "rank": exact["rank"],
            "doc_id": exact["doc_id"],
            "admitted_before_rerank": admitted,
            "admission_evidence": admission_evidence,
            "exact_rerank_rank": result_rank,
            "candidate_rank_before_rerank": (
                int(entry["candidate_rank_before_truncation"])
                if isinstance(entry, dict) and entry.get("candidate_rank_before_truncation") is not None
                else None
            ),
        })
    top = exact_with_admission[: min(10, len(exact_with_admission))]
    admitted_count = sum(1 for item in top if item["admitted_before_rerank"])
    return {
        "mode": mode["name"],
        "candidate_source": stats.get("multivector_candidate_source"),
        "configured_candidate_source": mode["candidate_source"],
        "graph_mode": stats.get("multivector_graph_mode"),
        "configured_graph_mode": mode.get("graph_mode", "token_nodes"),
        "plain_fallback": mode["plain_fallback"],
        "reservoirs": mode["reservoirs"],
        "result_doc_ids": docs,
        "top1": docs[0] if docs else None,
        "latency_ms": round(latency_ms, 3),
        "latency": summarize_ms([latency_ms]),
        "exact_top1_admitted_before_rerank": bool(exact_with_admission and exact_with_admission[0]["admitted_before_rerank"]),
        "exact_top10_admission_recall": round(admitted_count / len(top), 6) if top else 0.0,
        "doc_candidates": max(
            scan_stat_int(stats, "multivector_doc_candidates"),
            scan_stat_int(stats, "multivector_doc_graph_candidates"),
        ),
        "exact_rerank_docs": max(
            scan_stat_int(stats, "multivector_exact_rerank_docs"),
            scan_stat_int(stats, "multivector_doc_graph_exact_rerank_docs"),
        ),
        **exact_rerank_work_from_stats(stats),
        **proxy_work_from_stats(stats),
        "memory_bytes_estimate": memory_estimate_from_stats(stats),
        "raw_subvector_hits": scan_stat_int(stats, "multivector_raw_subvector_hits"),
        "unique_docs": scan_stat_int(stats, "multivector_unique_docs"),
        "trace_entries": len(trace_entries),
        "admission_inferred_from_result_docs": inferred_exact_doc_admission,
        "exact_top": exact_with_admission,
        "scan_stats": stats,
    }


def markdown_recall_gate_summary(report: dict[str, Any]) -> str:
    status = "PASS" if report["passed"] else "FAIL"
    lines = [
        "## Multivector Recall Gate",
        "",
        f"Status: **{status}**",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Exact top-1: `{report['exact_top1']}`",
        f"- Small candidate budget: `{report['candidate_budget']}`",
        f"- DBpedia data: `{report['dbpedia_status']}`",
        "",
        "| mode | top1 | top1 admitted | top10 admission recall | p50 ms | p95 ms | candidates | exact rerank docs | memory bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["results"]:
        latency = item.get("latency", {})
        lines.append(
            "| {mode} | {top1} | {admitted} | {recall:.3f} | {p50:.3f} | {p95:.3f} | {candidates} | {rerank} | {memory} |".format(
                mode=item["mode"],
                top1=item.get("top1"),
                admitted="yes" if item.get("exact_top1_admitted_before_rerank") else "no",
                recall=float(item.get("exact_top10_admission_recall", 0.0)),
                p50=float(latency.get("p50_ms", 0.0)),
                p95=float(latency.get("p95_ms", 0.0)),
                candidates=item.get("doc_candidates", 0),
                rerank=item.get("exact_rerank_docs", 0),
                memory=item.get("memory_bytes_estimate", 0),
            )
        )
    lines.extend([
        "",
        "Gate condition: `exact_scan`, `plain_fallback`, `exact_doc_scan`, "
        "`doc_graph_prototype`, `document_nodes`, and `proxy_vector` must return "
        "and admit the synthetic exact top-1 document at the configured small candidate "
        "budget. DBpedia admission reporting remains optional and is run with "
        "`--admission-debug` when local data is available.",
        "",
    ])
    return "\n".join(lines)


def markdown_benchmark_summary(report: dict[str, Any]) -> str:
    dataset = report.get("dataset", {})
    settings = report.get("settings", {})
    lines = [
        "## DBpedia ColBERT Multivector Benchmark",
        "",
        f"- Documents: `{dataset.get('documents', 0)}`",
        f"- Queries: `{dataset.get('queries', 0)}`",
        f"- Qrels: `{dataset.get('qrels', 0)}`",
        f"- Final K: `{settings.get('final_k', 0)}`",
        f"- Clients: `{settings.get('clients', 0)}`",
        f"- Multivector graph: `{settings.get('multivector_graph', '')}`",
        f"- Candidate source: `{settings.get('multivector_candidate_source', '')}`",
        f"- Sparse candidate source: `{settings.get('multivector_sparse_candidate_source', '')}`",
        f"- Token pooling: `{settings.get('multivector_token_pooling', '')}`",
        f"- Token pooling target ratio: `{settings.get('multivector_token_pooling_target_ratio', '')}`",
        f"- Centroids: `{settings.get('multivector_centroids', '')}`",
        f"- Centroid count: `{settings.get('multivector_centroid_count', '')}`",
        f"- Proxy encoder: `{settings.get('multivector_proxy_encoder', '')}`",
        f"- Doc graph search EF: `{settings.get('multivector_doc_graph_search_ef', 0)}`",
        f"- Doc graph oversampling: `{settings.get('multivector_doc_graph_oversampling', 1)}`",
        f"- Doc graph rescore K: `{settings.get('multivector_doc_graph_rescore_k', 0)}`",
        f"- Doc graph entry sample count: `{settings.get('multivector_doc_graph_entry_sample_count', 0)}`",
        f"- Plain fallback: `{settings.get('multivector_plain_fallback', '')}`",
        f"- Reservoirs: `{settings.get('multivector_candidate_reservoirs', '')}`",
        "",
        "| method | recall@10 | ndcg@10 | serial p50 ms | serial p95 ms | 8x qps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in report.get("results", []):
        metrics = item.get("metrics", {})
        serial = item.get("serial_latency", {})
        parallel = item.get("parallel_8x", {})
        lines.append(
            "| {method} | {recall:.6f} | {ndcg:.6f} | {p50:.3f} | {p95:.3f} | {qps:.3f} |".format(
                method=item.get("method", ""),
                recall=float(metrics.get("recall@10", 0.0)),
                ndcg=float(metrics.get("ndcg@10", 0.0)),
                p50=float(serial.get("p50_ms", 0.0)),
                p95=float(serial.get("p95_ms", 0.0)),
                qps=float(parallel.get("throughput_qps", 0.0)),
            )
        )
    skipped_parallel = [
        item
        for item in report.get("results", [])
        if isinstance(item, dict)
        and isinstance(item.get("parallel_8x"), dict)
        and item["parallel_8x"].get("skipped") is True
    ]
    failed_parallel = [
        item
        for item in report.get("results", [])
        if isinstance(item, dict)
        and isinstance(item.get("parallel_8x"), dict)
        and item["parallel_8x"].get("failed") is True
    ]
    if skipped_parallel or failed_parallel:
        lines.extend([
            "",
            "### Parallel Retrieval Notes",
            "",
        ])
        for item in skipped_parallel:
            parallel = item["parallel_8x"]
            lines.append(
                "- `{method}` skipped 8x retrieval: {reason}".format(
                    method=item.get("method", ""),
                    reason=parallel.get("skip_reason", ""),
                )
            )
        for item in failed_parallel:
            parallel = item["parallel_8x"]
            lines.append(
                "- `{method}` failed 8x retrieval: {error}".format(
                    method=item.get("method", ""),
                    error=parallel.get("error", ""),
                )
            )

    phases = report.get("phases", {})
    learned_sparse_phase = {}
    if isinstance(phases, dict):
        candidate_phase = phases.get("load_learned_sparse_vectors", {})
        if isinstance(candidate_phase, dict):
            learned_sparse_phase = candidate_phase
    if learned_sparse_phase and not bool(learned_sparse_phase.get("skipped", False)):
        coverage = learned_sparse_phase.get("coverage", {})
        if not isinstance(coverage, dict):
            coverage = {}
        warnings = learned_sparse_phase.get("warnings", coverage.get("warnings", []))
        if not isinstance(warnings, list):
            warnings = []
        doc_ratio = coverage.get("doc_coverage_ratio")
        query_ratio = coverage.get("query_coverage_ratio")
        lines.extend([
            "",
            "### Learned-sparse fixture coverage",
            "",
            f"- Documents covered: `{int(coverage.get('learned_sparse_documents', 0) or 0)}` / `{int(coverage.get('loaded_documents', 0) or 0)}`",
            f"- Document coverage ratio: `{float(doc_ratio) if doc_ratio is not None else 0.0:.6f}`",
            f"- Queries covered: `{int(coverage.get('learned_sparse_queries', 0) or 0)}` / `{int(coverage.get('loaded_queries', 0) or 0)}`",
            f"- Query coverage ratio: `{float(query_ratio) if query_ratio is not None else 0.0:.6f}`",
            f"- Partial coverage: `{bool(learned_sparse_phase.get('partial_coverage', coverage.get('partial_coverage', False)))}`",
        ])
        if warnings:
            lines.append("- Warnings: `" + ",".join(str(item) for item in warnings) + "`")
            lines.append(
                "- Learned-sparse rescue results with partial fixture coverage are "
                "candidate-source plumbing evidence, not production serving evidence."
            )
        else:
            lines.append("- Warnings: `none`")

    admission = report.get("admission_debug")
    if isinstance(admission, dict):
        aggregate = admission.get("aggregate", {})
        first_budget = aggregate.get("exact_top1_first_budget_admitted", {})
        lines.extend([
            "",
            "### Admission Debug",
            "",
            f"- Candidate source: `{admission.get('candidate_source', '')}`",
            f"- Sparse candidate source: `{admission.get('sparse_candidate_source', '')}`",
            f"- Plain fallback: `{admission.get('plain_fallback', '')}`",
            f"- Reservoirs: `{admission.get('candidate_reservoirs', '')}`",
            f"- Admission K: `{aggregate.get('admission_k', 0)}`",
            f"- Admission debug mode: `{aggregate.get('admission_debug_mode', 'trace')}`",
            f"- Trace enabled: `{bool(aggregate.get('trace_enabled', False))}`",
            f"- Admission evidence mode: `{aggregate.get('admission_evidence_mode', 'unavailable')}`",
            f"- Trace entries: `{int(aggregate.get('trace_entries', 0) or 0)}`",
            f"- Exact rerank mode: `{aggregate.get('serving_exact_rerank_mode', 'admission_exhaustive')}`",
            f"- Requested serving exact rerank K: `{int(aggregate.get('requested_serving_exact_rerank_k', 0) or 0)}`",
            f"- Exact top-1 admission rate: `{float(aggregate.get('exact_top1_admission_rate', 0.0)):.6f}`",
            f"- Exact top-10 admission recall: `{float(aggregate.get('exact_top10_admission_recall', 0.0)):.6f}`",
            f"- First admitted budget p50: `{first_budget.get('p50', 0)}`",
            "",
            "| candidate budget | exact rerank k | top1 admission | top10 admission recall | latency p50 ms | latency p95 ms | docs scored p50 | graph edges p50 | exact rerank docs p50 | pairs saved p50 | runs |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        admission_by_budget = aggregate.get("admission_by_budget", {})
        if isinstance(admission_by_budget, dict):
            for budget, budget_stats in sorted(admission_by_budget.items(), key=lambda item: int(item[0])):
                if not isinstance(budget_stats, dict):
                    continue
                latency = budget_stats.get("latency", {})
                docs_scored = budget_stats.get("docs_scored", {})
                graph_edges = budget_stats.get("graph_edges_visited", {})
                exact_rerank_docs = budget_stats.get("exact_rerank_docs", {})
                exact_rerank_pairs_saved = budget_stats.get("exact_rerank_pairs_saved", {})
                if not isinstance(latency, dict):
                    latency = {}
                if not isinstance(docs_scored, dict):
                    docs_scored = {}
                if not isinstance(graph_edges, dict):
                    graph_edges = {}
                if not isinstance(exact_rerank_docs, dict):
                    exact_rerank_docs = {}
                if not isinstance(exact_rerank_pairs_saved, dict):
                    exact_rerank_pairs_saved = {}
                lines.append(
                    "| {budget} | {exact_k} | {top1:.6f} | {top10:.6f} | {p50:.3f} | {p95:.3f} | {docs_p50:.3f} | {edges_p50:.3f} | {rerank_p50:.3f} | {pairs_saved_p50:.3f} | {runs} |".format(
                        budget=budget,
                        exact_k=int(budget_stats.get("effective_exact_rerank_k", 0) or 0),
                        top1=float(budget_stats.get("exact_top1_admission", 0.0)),
                        top10=float(budget_stats.get("exact_top10_admission_recall", 0.0)),
                        p50=float(latency.get("p50_ms", 0.0)),
                        p95=float(latency.get("p95_ms", 0.0)),
                        docs_p50=float(docs_scored.get("p50", 0.0)),
                        edges_p50=float(graph_edges.get("p50", 0.0)),
                        rerank_p50=float(exact_rerank_docs.get("p50", 0.0)),
                        pairs_saved_p50=float(exact_rerank_pairs_saved.get("p50", 0.0)),
                        runs=int(latency.get("runs", 0)),
                    )
                )
        lines.extend([
            "",
            "Admission debug compares exact document top-K with the candidate set "
            "before exact rerank. Improvements here mean better admission; unchanged "
            "admission with different final metrics should be treated as reranking or "
            "metric variance until proven otherwise.",
        ])

    admission_grid = report.get("document_node_admission_grid")
    if isinstance(admission_grid, dict):
        pooling_grid_text = ",".join(
            "{mode}:{ratio}".format(mode=item.get("mode"), ratio=item.get("ratio"))
            for item in admission_grid.get("pooling_grid", [])
            if isinstance(item, dict)
        )
        lines.extend([
            "",
            "### Document-Node Admission Grid",
            "",
            f"- Rebuilds index: `{admission_grid.get('rebuilds_index', False)}`",
            f"- Budget sweep: `{','.join(str(item) for item in admission_grid.get('budget_sweep', []))}`",
            f"- Storage grid: `{','.join(str(item) for item in admission_grid.get('storage_grid', []))}`",
            f"- Pooling grid: `{pooling_grid_text}`",
            f"- EF grid: `{','.join(str(item) for item in admission_grid.get('ef_grid', []))}`",
            f"- Oversampling grid: `{','.join(str(item) for item in admission_grid.get('oversampling_grid', []))}`",
            f"- Index graph build: `m={admission_grid.get('index_graph_m', 0)}, ef_construction={admission_grid.get('index_graph_ef_construction', 0)}, ef_search={admission_grid.get('index_graph_ef_search', 0)}, native_segments={admission_grid.get('index_native_segments', 0)}`",
            "",
            "| mode | graph | storage | pooling | ratio | ef | oversampling | budget | top1 admission | top10 admission recall | recall@10 | ndcg@10 | mrr@10 | p50 ms | p95 ms | docs scored p50 | edges p50 | exact rerank p50 | native exact bytes p50 |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        grid_results = admission_grid.get("results", [])
        if isinstance(grid_results, list):
            for item in grid_results:
                if not isinstance(item, dict):
                    continue
                budget = item.get("largest_budget")
                admission_by_budget = item.get("admission_by_budget", {})
                budget_stats = {}
                if isinstance(admission_by_budget, dict) and budget is not None:
                    budget_stats = admission_by_budget.get(str(budget), {})
                    if not isinstance(budget_stats, dict):
                        budget_stats = {}
                latency = budget_stats.get("latency", {}) if isinstance(budget_stats, dict) else {}
                docs_scored = budget_stats.get("docs_scored", {}) if isinstance(budget_stats, dict) else {}
                graph_edges = budget_stats.get("graph_edges_visited", {}) if isinstance(budget_stats, dict) else {}
                exact_rerank_docs = budget_stats.get("exact_rerank_docs", {}) if isinstance(budget_stats, dict) else {}
                native_cache_exact = budget_stats.get("native_cache_exact_bytes", {}) if isinstance(budget_stats, dict) else {}
                metrics = item.get("metrics", {})
                if not isinstance(latency, dict):
                    latency = {}
                if not isinstance(docs_scored, dict):
                    docs_scored = {}
                if not isinstance(graph_edges, dict):
                    graph_edges = {}
                if not isinstance(exact_rerank_docs, dict):
                    exact_rerank_docs = {}
                if not isinstance(native_cache_exact, dict):
                    native_cache_exact = {}
                if not isinstance(metrics, dict):
                    metrics = {}
                lines.append(
                    "| {mode} | {graph} | {storage} | {pooling} | {ratio:.2f} | {ef} | {oversampling} | {budget} | {top1:.6f} | {top10:.6f} | {recall:.6f} | {ndcg:.6f} | {mrr:.6f} | {p50:.3f} | {p95:.3f} | {docs:.3f} | {edges:.3f} | {rerank:.3f} | {native_exact:.3f} |".format(
                        mode=item.get("mode", ""),
                        graph=item.get("graph_mode", ""),
                        storage=item.get("storage_kind", ""),
                        pooling=item.get("token_pooling", ""),
                        ratio=float(item.get("token_pooling_target_ratio", 1.0) or 1.0),
                        ef=int(item.get("doc_graph_search_ef", 0) or 0),
                        oversampling=int(item.get("doc_graph_oversampling", 0) or 0),
                        budget=budget or "",
                        top1=float(item.get("exact_top1_admission_rate", 0.0)),
                        top10=float(item.get("exact_top10_admission_recall", 0.0)),
                        recall=float(metrics.get("recall@10", 0.0)),
                        ndcg=float(metrics.get("ndcg@10", 0.0)),
                        mrr=float(metrics.get("mrr@10", 0.0)),
                        p50=float(latency.get("p50_ms", 0.0)),
                        p95=float(latency.get("p95_ms", 0.0)),
                        docs=float(docs_scored.get("p50", 0.0)),
                        edges=float(graph_edges.get("p50", 0.0)),
                        rerank=float(exact_rerank_docs.get("p50", 0.0)),
                        native_exact=float(native_cache_exact.get("p50", 0.0)),
                    )
                )
        lines.extend([
            "",
            "The grid rebuilds token-node and document-node index layouts so the "
            "report can compare token baselines with document-node admission. "
            "Document-sidecar page/cache counters are emitted when the engine "
            "exposes them; until then the table includes native cache exact-byte "
            "counters as the available storage footprint signal.",
        ])

    build_only = report.get("document_node_serving_build_only")
    if isinstance(build_only, dict):
        rows = build_only.get("results", [])
        if isinstance(rows, list):
            sorted_rows = sorted(
                [item for item in rows if isinstance(item, dict)],
                key=lambda item: (
                    float(item.get("index_build_elapsed_ms", 0.0) or 0.0),
                    str(item.get("profile", "")),
                ),
                reverse=True,
            )
        else:
            sorted_rows = []
        lines.extend([
            "",
            "### Document-node serving build-only",
            "",
            f"- Retrieval skipped: `{bool(build_only.get('retrieval_skipped', False))}`",
            f"- Admission skipped: `{bool(build_only.get('admission_skipped', False))}`",
            f"- Total elapsed: `{float(build_only.get('total_elapsed_ms', 0.0) or 0.0):.3f} ms`",
            f"- Index build elapsed: `{float(build_only.get('index_build_elapsed_ms_total', 0.0) or 0.0):.3f} ms` across `{int(build_only.get('index_builds', 0) or 0)}` builds",
            f"- Index build groups: `{int(build_only.get('index_build_group_count', 0) or 0)}`; reuses: `{int(build_only.get('index_reuse_count', 0) or 0)}`",
            f"- Effective profiles: `{','.join(str(item) for item in build_only.get('effective_profiles', []))}`",
            "",
            "| profile | source | proxy | centroids | entry sidecar | storage | pooling | reused | dominant build phase | index build ms | known phase ms | unattributed ms | centroid build ms | centroid cluster ms | centroid residual ms | proxy build ms | doc sidecar write ms | centroid sidecar write ms | centroid posting write ms | centroid postings | index bytes |",
            "|---|---|---|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for item in sorted_rows:
            lines.append(
                "| {profile} | {source} | {proxy} | {centroids} | {entry_sidecar} | {storage} | {pooling}:{ratio:.2f} | {reused} | {dominant} | {index:.3f} | {known:.3f} | {unknown:.3f} | {centroid:.3f} | {centroid_cluster:.3f} | {centroid_residual:.3f} | {proxy_build:.3f} | {doc_sidecar:.3f} | {centroid_sidecar:.3f} | {posting:.3f} | {posting_count} | {index_bytes} |".format(
                    profile=item.get("profile", ""),
                    source=item.get("candidate_source", ""),
                    proxy=item.get("proxy_encoder", ""),
                    centroids=item.get("centroids", ""),
                    entry_sidecar=(
                        f"on:{item.get('entry_sidecar_representatives', '')}:{item.get('entry_sidecar_strategy', '')}"
                        if bool(item.get("entry_sidecar", False))
                        else "off"
                    ),
                    storage=item.get("storage_kind", ""),
                    pooling=item.get("token_pooling", ""),
                    ratio=float(item.get("token_pooling_target_ratio", 1.0) or 1.0),
                    reused=bool(item.get("index_build_reused", False)),
                    dominant=item.get("dominant_build_phase", ""),
                    index=float(item.get("index_build_elapsed_ms", 0.0) or 0.0),
                    known=float(item.get("build_phase_known_ms", 0.0) or 0.0),
                    unknown=float(item.get("build_phase_unattributed_ms", 0.0) or 0.0),
                    centroid=float(item.get("multivector_centroid_build_us", 0.0) or 0.0) / 1000.0,
                    centroid_cluster=float(item.get("multivector_centroid_cluster_us", 0.0) or 0.0) / 1000.0,
                    centroid_residual=float(item.get("multivector_centroid_residual_us", 0.0) or 0.0) / 1000.0,
                    proxy_build=float(item.get("multivector_proxy_build_us", 0.0) or 0.0) / 1000.0,
                    doc_sidecar=float(item.get("multivector_doc_sidecar_write_us", 0.0) or 0.0) / 1000.0,
                    centroid_sidecar=float(item.get("multivector_centroid_sidecar_write_us", 0.0) or 0.0) / 1000.0,
                    posting=float(item.get("multivector_centroid_posting_write_us", 0.0) or 0.0) / 1000.0,
                    posting_count=int(item.get("multivector_centroid_posting_count", 0) or 0),
                    index_bytes=int(item.get("index_bytes", 0) or 0),
                )
            )
        if sorted_rows:
            lines.extend([
                "",
                "#### Document-node topology build phases",
                "",
                "| profile | node assignment ms | entry search ms | neighbor search ms | neighbor select ms | link insert ms | reciprocal prune ms | segment write ms | wal ms | proxy distance calls | exact distance calls | distance cache hits | distance cache misses |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ])
            for item in sorted_rows:
                lines.append(
                    "| {profile} | {node_assignment:.3f} | {entry_search:.3f} | {neighbor_search:.3f} | {neighbor_select:.3f} | {link_insert:.3f} | {reciprocal_prune:.3f} | {segment_write:.3f} | {wal:.3f} | {proxy_calls} | {exact_calls} | {cache_hits} | {cache_misses} |".format(
                        profile=item.get("profile", ""),
                        node_assignment=float(item.get("multivector_graph_node_assignment_us", 0.0) or 0.0) / 1000.0,
                        entry_search=float(item.get("multivector_graph_entry_search_us", 0.0) or 0.0) / 1000.0,
                        neighbor_search=float(item.get("multivector_graph_neighbor_search_us", 0.0) or 0.0) / 1000.0,
                        neighbor_select=float(item.get("multivector_graph_neighbor_select_us", 0.0) or 0.0) / 1000.0,
                        link_insert=float(item.get("multivector_graph_link_insert_us", 0.0) or 0.0) / 1000.0,
                        reciprocal_prune=float(item.get("multivector_graph_reciprocal_prune_us", 0.0) or 0.0) / 1000.0,
                        segment_write=float(item.get("multivector_graph_segment_write_us", 0.0) or 0.0) / 1000.0,
                        wal=float(item.get("multivector_graph_wal_us", 0.0) or 0.0) / 1000.0,
                        proxy_calls=int(item.get("multivector_graph_build_distance_proxy_calls", 0) or 0),
                        exact_calls=int(item.get("multivector_graph_build_distance_exact_calls", 0) or 0),
                        cache_hits=int(item.get("multivector_graph_build_distance_cache_hits", 0) or 0),
                        cache_misses=int(item.get("multivector_graph_build_distance_cache_misses", 0) or 0),
                    )
                )

    latency_only = report.get("document_node_serving_latency_only")
    if isinstance(latency_only, dict):
        settings = latency_only.get("settings", {})
        latency = latency_only.get("latency", {})
        admission_metrics = latency_only.get("admission_metrics", {})
        scan_stats = latency_only.get("scan_stats_summary", {})
        if not isinstance(settings, dict):
            settings = {}
        if not isinstance(latency, dict):
            latency = {}
        if not isinstance(admission_metrics, dict):
            admission_metrics = {}
        if not isinstance(scan_stats, dict):
            scan_stats = {}
        warmup_latency = latency_only.get("warmup_latency", {})
        if not isinstance(warmup_latency, dict):
            warmup_latency = {}
        slow_path_warnings = latency_only.get("serving_slow_path_warnings", [])
        if not isinstance(slow_path_warnings, list):
            slow_path_warnings = []
        field_summary = scan_stats.get("field_summary", {})
        if not isinstance(field_summary, dict):
            field_summary = {}

        def scan_field_p50(key: str) -> float:
            value = field_summary.get(key, {})
            return float(value.get("p50", 0.0) or 0.0) if isinstance(value, dict) else 0.0

        def scan_field_p95(key: str) -> float:
            value = field_summary.get(key, {})
            return float(value.get("p95", 0.0) or 0.0) if isinstance(value, dict) else 0.0

        last_selected = scan_stats.get("last_selected", {})
        if not isinstance(last_selected, dict):
            last_selected = {}
        phase_source = last_selected.get("phase_timing_source", "")
        phase_missing = last_selected.get("phase_timing_missing", [])
        if not isinstance(phase_missing, list):
            phase_missing = []
        lines.extend([
            "",
            "### Document-node serving latency-only",
            "",
            f"- Profile: `{latency_only.get('profile', '')}`",
            f"- Candidate source: `{settings.get('candidate_source', '')}`",
            f"- Proxy encoder: `{settings.get('proxy_encoder', '')}`",
            f"- Centroids: `{settings.get('centroids', '')}`",
            f"- Storage/cache: `{settings.get('storage_kind', '')}/{settings.get('storage_cache_mode', '')}`",
            f"- EF / oversampling: `{settings.get('ef', 0)}` / `{settings.get('oversampling', 0)}`",
            f"- Entry sample count: `{settings.get('entry_sample_count', 0)}`",
            f"- Candidate K / exact rerank K: `{settings.get('candidate_k', 0)}` / `{settings.get('effective_exact_rerank_k', 0)}`",
            f"- Queries: `{int(latency_only.get('query_count', 0) or 0)}`",
            f"- Warmup queries: `{int(settings.get('warmup_queries', 0) or 0)}`; excluded from measured latency: `{bool(latency_only.get('warmup_excluded', False))}`",
            f"- Cache build queries: `{int(latency_only.get('cache_build_queries', 0) or 0)}`; cache reused on warm queries: `{int(latency_only.get('cache_reused_on_warm_queries', 0) or 0)}` / `{int(latency_only.get('warm_queries', 0) or 0)}`",
            f"- Admission metrics: `{bool(admission_metrics.get('available', False))}` ({admission_metrics.get('reason', '')})",
            "",
            "| phase | p50 ms | p95 ms | p99 ms | qps | graph docs p50 | graph edges p50 | exact rerank docs p50 | exact pairs p50 | kernel |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            "| warmup/cold | {p50:.3f} | {p95:.3f} | {p99:.3f} | {qps:.3f} |  |  |  |  |  |".format(
                p50=float(warmup_latency.get("p50_ms", 0.0) or 0.0),
                p95=float(warmup_latency.get("p95_ms", 0.0) or 0.0),
                p99=float(warmup_latency.get("p99_ms", 0.0) or 0.0),
                qps=float(warmup_latency.get("qps", 0.0) or 0.0),
            ),
            "| measured/warm | {p50:.3f} | {p95:.3f} | {p99:.3f} | {qps:.3f} | {docs:.3f} | {edges:.3f} | {rerank:.3f} | {pairs:.3f} | {kernel} |".format(
                p50=float(latency.get("p50_ms", 0.0) or 0.0),
                p95=float(latency.get("p95_ms", 0.0) or 0.0),
                p99=float(latency.get("p99_ms", 0.0) or 0.0),
                qps=float(latency.get("qps_total", latency.get("qps", 0.0)) or 0.0),
                docs=scan_field_p50("multivector_doc_graph_docs_scored"),
                edges=scan_field_p50("multivector_doc_graph_edges_visited"),
                rerank=scan_field_p50("multivector_exact_rerank_docs"),
                pairs=scan_field_p50("multivector_exact_rerank_pairs"),
                kernel=last_selected.get("multivector_exact_kernel", ""),
            ),
            "",
            "#### Serving phase timings",
            "",
            f"- Source: `{phase_source}`",
            f"- Missing phases: `{','.join(str(item) for item in phase_missing)}`",
            "",
            "| phase | p50 ms | p95 ms |",
            "|---|---:|---:|",
        ])
        for phase_key, label in (
            ("phase_total_time_us", "total"),
            ("phase_prepare_time_us", "prepare"),
            ("document_graph_traversal_time_us", "graph traversal"),
            ("proxy_candidate_time_us", "proxy candidates"),
            ("proxy_graph_traversal_time_us", "proxy graph traversal"),
            ("document_graph_batch_time_us", "graph batch scoring"),
            ("sidecar_load_time_us", "sidecar load"),
            ("sidecar_page_read_time_us", "sidecar page reads"),
            ("sidecar_vector_reconstruct_time_us", "sidecar vector reconstruct"),
            ("exact_heap_fetch_time_us", "exact heap fetch"),
            ("exact_maxsim_rerank_time_us", "exact MaxSim rerank"),
            ("final_sort_time_us", "final sort"),
            ("phase_timing_known_total_us", "known subtotal"),
            ("phase_timing_unattributed_sql_ms", "unattributed SQL"),
        ):
            if phase_key.endswith("_ms"):
                p50_value = scan_field_p50(phase_key)
                p95_value = scan_field_p95(phase_key)
            else:
                p50_value = scan_field_p50(phase_key) / 1000.0
                p95_value = scan_field_p95(phase_key) / 1000.0
            lines.append(f"| {label} | {p50_value:.3f} | {p95_value:.3f} |")
        lines.extend([
            "",
            "This mode runs normal retrieval only. It intentionally does not call "
            "exact admission baselines, so admission and qrel metrics are unavailable "
            "rather than zero.",
            "",
            "#### Slow path warnings",
            "",
            f"- Failed: `{bool(latency_only.get('serving_slow_path_failed', False))}`",
        ])
        if slow_path_warnings:
            for warning in slow_path_warnings:
                lines.append(f"- `{warning}`")
        else:
            lines.append("- None")

    serving_grid = report.get("document_node_serving_grid")
    if isinstance(serving_grid, dict):
        rows = serving_grid.get("results", serving_grid.get("summary_rows", []))
        if isinstance(rows, list):
            sorted_rows = sorted(
                [item for item in rows if isinstance(item, dict)],
                key=lambda item: (
                    serving_row_p95(item) if serving_row_p95(item) is not None else float("inf"),
                    str(item.get("profile", "")),
                    int(item.get("ef", 0) or 0),
                    int(item.get("oversampling", 0) or 0),
                ),
            )
        else:
            sorted_rows = []
        serving_grid_mode = str(serving_grid.get("mode", "full") or "full")
        serving_grid_stage_mode = str(serving_grid.get("stage_mode", "single") or "single")
        lines.extend([
            "",
            "### Document-node serving grid",
            "",
            f"- Mode: `{serving_grid_mode}`",
            f"- Stage mode: `{serving_grid_stage_mode}`",
            f"- Probe queries: `{int(serving_grid.get('probe_queries', 0) or 0)}`",
            f"- Finalists: `{int(serving_grid.get('finalists', 0) or 0)}`",
            f"- Smoke-only: `{bool(serving_grid.get('smoke_only', False))}`",
            f"- Serving evidence: `{bool(serving_grid.get('serving_evidence', True))}`",
            f"- Budget mode: `{serving_grid.get('budget_mode', 'sweep')}`",
            f"- Exact rerank mode: `{serving_grid.get('serving_exact_rerank_mode', 'admission_exhaustive')}`",
            f"- Requested serving exact rerank K: `{int(serving_grid.get('requested_serving_exact_rerank_k', 0) or 0)}`",
            f"- Admission debug mode: `{serving_grid.get('admission_debug_mode', 'summary')}`",
            f"- Trace enabled: `{bool(serving_grid.get('trace_enabled', False))}`",
            f"- Effective profiles: `{','.join(str(item) for item in serving_grid.get('effective_profiles', []))}`",
            f"- Opt-in reservoirs: `{bool(serving_grid.get('include_reservoirs', False))}`",
            f"- Opt-in centroid-lite caps: `{bool(serving_grid.get('include_centroid_lite_caps', False))}`",
            f"- Effective centroid-lite posting caps: `{','.join(str(item) for item in serving_grid.get('effective_centroid_lite_posting_caps', []))}`",
            f"- Opt-in entry samples: `{bool(serving_grid.get('include_entry_samples', False))}`",
            f"- Effective entry sample counts: `{','.join(str(item) for item in serving_grid.get('effective_entry_sample_counts', []))}`",
            f"- Proxy admission focus: `{bool(serving_grid.get('proxy_admission_focus', False))}`",
            f"- Centroid-lite focus: `{bool(serving_grid.get('centroid_lite_focus', False))}`",
            f"- Token-pooling focus: `{bool(serving_grid.get('token_pooling_focus', False))}`",
            f"- Effective entry-sidecar representatives: `{','.join(str(item) for item in serving_grid.get('effective_entry_sidecar_representatives', []))}`",
            f"- Effective EF grid: `{','.join(str(item) for item in serving_grid.get('effective_ef_grid', []))}`",
            f"- Effective oversampling grid: `{','.join(str(item) for item in serving_grid.get('effective_oversampling_grid', []))}`",
            f"- Requested budget sweep: `{','.join(str(item) for item in serving_grid.get('effective_budget_sweep', serving_grid.get('budget_sweep', [])))}`",
            f"- Executed budgets: `{','.join(str(item) for item in serving_grid.get('executed_budgets', serving_grid.get('budgets_run', [])))}`",
            f"- Query subset used: `{bool(serving_grid.get('query_subset_used', False))}`",
            f"- Queries: `{int(serving_grid.get('queries_run', serving_grid.get('queries', 0)) or 0)}` of `{int(serving_grid.get('queries_available', serving_grid.get('queries', 0)) or 0)}` loaded",
            "",
        ])
        if serving_grid_mode == "smoke":
            lines.extend([
                "Smoke mode is a smoke-only runtime check and must not be treated as serving evidence.",
                "",
            ])
        if serving_grid_stage_mode == "two_stage":
            lines.extend([
                "Two-stage mode probes all effective configs on the first probe queries, "
                "ranks non-experimental configs by p95 rank plus admission loss, and "
                "fully evaluates only the finalists. Recommendations use the final "
                "stage rows.",
                "",
                f"- Stage 1 rows: `{len(serving_grid.get('stage1_results', []) if isinstance(serving_grid.get('stage1_results', []), list) else [])}`",
                f"- Stage 2 rows: `{len(serving_grid.get('stage2_results', []) if isinstance(serving_grid.get('stage2_results', []), list) else [])}`",
                f"- Pruned configs: `{len(serving_grid.get('pruned_configs', []) if isinstance(serving_grid.get('pruned_configs', []), list) else [])}`",
                "",
            ])
        lines.extend([
            "| profile | source | graph | proxy | centroids | reservoirs | centroid cap | entry samples | entry sidecar | storage | cache | pooling | ef | oversampling | candidate budget | effective candidates | underfill reason | exact rerank k | rerank docs p50 | top1 admission | top10 admission | recall@10 | ndcg@10 | mrr@10 | p50 ms | p95 ms | p99 ms | index bytes |",
            "|---|---|---|---|---|---|---:|---:|---|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for item in sorted_rows:
            lines.append(
                "| {profile} | {source} | {graph} | {proxy} | {centroids} | {reservoirs} | {cap} | {entry_samples} | {entry_sidecar} | {storage} | {cache} | {pooling}:{ratio:.2f} | {ef} | {oversampling} | {candidate_budget} | {effective_candidates} | {underfill_reason} | {exact_k} | {rerank_docs_p50:.3f} | {top1:.6f} | {top10:.6f} | {recall:.6f} | {ndcg:.6f} | {mrr:.6f} | {p50:.3f} | {p95:.3f} | {p99:.3f} | {index_bytes} |".format(
                    profile=item.get("profile", ""),
                    source=item.get("candidate_source", ""),
                    graph=item.get("graph_mode", ""),
                    proxy=item.get("proxy_encoder", ""),
                    centroids=item.get("centroids", ""),
                    reservoirs=item.get("candidate_reservoirs", "off"),
                    cap=int(item.get("centroid_lite_max_postings_per_token", 0) or 0),
                    entry_samples=int(
                        item.get(
                            "effective_entry_sample_count",
                            item.get("entry_sample_count", 0),
                        )
                        or 0
                    ),
                    entry_sidecar=(
                        f"on:{item.get('entry_sidecar_representatives', '')}:{item.get('entry_sidecar_strategy', '')}"
                        if bool(item.get("entry_sidecar", False))
                        else "off"
                    ),
                    storage=item.get("storage_kind", ""),
                    cache=item.get("storage_cache_mode", item.get("cache_mode", "")),
                    pooling=item.get("token_pooling", ""),
                    ratio=float(item.get("token_pooling_target_ratio", 1.0) or 1.0),
                    ef=int(item.get("ef", 0) or 0),
                    oversampling=int(item.get("oversampling", 0) or 0),
                    candidate_budget=int(item.get("candidate_budget", item.get("largest_budget", 0)) or 0),
                    effective_candidates=int(item.get("effective_candidate_k", 0) or 0),
                    underfill_reason=item.get("candidate_underfill_reason", ""),
                    exact_k=int(item.get("effective_exact_rerank_k", 0) or 0),
                    rerank_docs_p50=float(item.get("exact_rerank_docs_p50", 0.0) or 0.0),
                    top1=float(item.get("exact_top1_admission_rate", 0.0) or 0.0),
                    top10=float(item.get("exact_top10_admission_recall", 0.0) or 0.0),
                    recall=float(item.get("recall@10", 0.0) or 0.0),
                    ndcg=float(item.get("ndcg@10", 0.0) or 0.0),
                    mrr=float(item.get("mrr@10", 0.0) or 0.0),
                    p50=float(item.get("p50_ms", item.get("p50_latency_ms_at_largest_budget", 0.0)) or 0.0),
                    p95=float(item.get("p95_ms", item.get("p95_latency_ms_at_largest_budget", 0.0)) or 0.0),
                    p99=float(item.get("p99_ms", item.get("p99_latency_ms_at_largest_budget", 0.0)) or 0.0),
                    index_bytes=int(item.get("index_bytes", 0) or 0),
                )
            )
        if sorted_rows:
            lines.extend([
                "",
                "#### Serving grid phase timings",
                "",
                "| profile | ef | oversampling | source | total ms | graph traversal ms | exact MaxSim ms | final sort ms | unattributed SQL ms |",
                "|---|---:|---:|---|---:|---:|---:|---:|---:|",
            ])
            for item in sorted_rows:
                sample = item.get("last_scan_stats_sample", item.get("serving_stats_sample", {}))
                if not isinstance(sample, dict):
                    sample = {}

                def sample_ms(key: str) -> float:
                    value = sample.get(key)
                    if isinstance(value, bool) or value is None:
                        return 0.0
                    try:
                        number = float(value)
                    except (TypeError, ValueError):
                        return 0.0
                    return number if key.endswith("_ms") else number / 1000.0

                lines.append(
                    "| {profile} | {ef} | {oversampling} | {source} | {total:.3f} | {traverse:.3f} | {maxsim:.3f} | {sort:.3f} | {unattributed:.3f} |".format(
                        profile=item.get("profile", ""),
                        ef=int(item.get("ef", 0) or 0),
                        oversampling=int(item.get("oversampling", 0) or 0),
                        source=sample.get("phase_timing_source", ""),
                        total=sample_ms("phase_total_time_us"),
                        traverse=sample_ms("document_graph_traversal_time_us"),
                        maxsim=sample_ms("exact_maxsim_rerank_time_us"),
                        sort=sample_ms("final_sort_time_us"),
                        unattributed=sample_ms("phase_timing_unattributed_sql_ms"),
                    )
                )
        candidate_source_deltas = serving_grid.get("candidate_source_deltas", {})
        if isinstance(candidate_source_deltas, dict):
            delta_rows = candidate_source_deltas.get("rows", [])
            if not isinstance(delta_rows, list):
                delta_rows = []
            sorted_delta_rows = sorted(
                [item for item in delta_rows if isinstance(item, dict)],
                key=lambda item: (
                    0 if bool(item.get("evidence_usable", True)) else 1,
                    -float(item.get("exact_top10_admission_recall_delta", 0.0) or 0.0),
                    float(item.get("p95_ms_delta", 0.0) or 0.0),
                    str(item.get("comparison", "")),
                    str(item.get("profile", "")),
                ),
            )
            lines.extend([
                "",
                "#### Candidate-source deltas",
                "",
            ])
            if sorted_delta_rows:
                lines.extend([
                    "Paired rows compare opt-in candidate-source variants against their "
                    "matching plain baseline at the same EF, oversampling, and executed "
                    "candidate budget.",
                    "",
                    "| comparison | profile | baseline | evidence | detail | ef | oversampling | budget | top10 delta | ndcg delta | p95 delta ms | top10 | baseline top10 | p95 ms | baseline p95 ms |",
                    "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
                ])
                for item in sorted_delta_rows:
                    def delta_float(key: str) -> float:
                        value = item.get(key)
                        if isinstance(value, bool) or value is None:
                            return 0.0
                        try:
                            return float(value)
                        except (TypeError, ValueError):
                            return 0.0

                    lines.append(
                        "| {comparison} | {profile} | {baseline} | {evidence} | {detail} | {ef} | {oversampling} | {budget} | {top10_delta:.6f} | {ndcg_delta:.6f} | {p95_delta:.3f} | {top10:.6f} | {baseline_top10:.6f} | {p95:.3f} | {baseline_p95:.3f} |".format(
                            comparison=item.get("comparison", ""),
                            profile=item.get("profile", ""),
                            baseline=item.get("baseline_profile", ""),
                            evidence=(
                                "usable"
                                if bool(item.get("evidence_usable", True))
                                else ",".join(
                                    str(warning)
                                    for warning in item.get("evidence_warnings", [])
                                )
                            ),
                            detail=serving_candidate_delta_evidence_text(
                                item.get("candidate_evidence", {})
                            ),
                            ef=int(item.get("ef", 0) or 0),
                            oversampling=int(item.get("oversampling", 0) or 0),
                            budget=int(item.get("candidate_budget", 0) or 0),
                            top10_delta=delta_float("exact_top10_admission_recall_delta"),
                            ndcg_delta=delta_float("ndcg@10_delta"),
                            p95_delta=delta_float("p95_ms_delta"),
                            top10=delta_float("exact_top10_admission_recall"),
                            baseline_top10=delta_float("baseline_exact_top10_admission_recall"),
                            p95=delta_float("p95_ms"),
                            baseline_p95=delta_float("baseline_p95_ms"),
                        )
                    )
            else:
                lines.append("- No paired candidate-source variants were available.")
            missing = candidate_source_deltas.get("missing_baselines", [])
            if isinstance(missing, list) and missing:
                lines.extend([
                    "",
                    f"- Missing paired baselines: `{len(missing)}`",
                ])
        lines.extend([
            "",
            "### Serving grid cost breakdown",
            "",
            f"- Total elapsed: `{float(serving_grid.get('total_elapsed_ms', 0.0) or 0.0):.3f} ms`",
            f"- Index build elapsed: `{float(serving_grid.get('index_build_elapsed_ms_total', 0.0) or 0.0):.3f} ms` across `{int(serving_grid.get('index_builds', 0) or 0)}` builds",
            f"- Index build groups: `{int(serving_grid.get('index_build_group_count', serving_grid.get('index_builds', 0)) or 0)}`; reuses: `{int(serving_grid.get('index_reuse_count', 0) or 0)}`",
            f"- Exact baseline elapsed: `{float(serving_grid.get('exact_baseline_elapsed_ms_total', 0.0) or 0.0):.3f} ms`",
            f"- Retrieval elapsed: `{float(serving_grid.get('retrieval_elapsed_ms_total', 0.0) or 0.0):.3f} ms`",
            f"- Recommendation elapsed: `{float(serving_grid.get('recommendation_elapsed_ms', 0.0) or 0.0):.3f} ms`",
            f"- Exact baseline scans: `{int(serving_grid.get('exact_baseline_query_count', 0) or 0)}`",
            f"- Retrieval queries: `{int(serving_grid.get('retrieval_query_count', 0) or 0)}`",
            f"- Exact top cache: `enabled={bool(serving_grid.get('exact_top_cache_enabled', False))}, hits={int(serving_grid.get('exact_top_cache_hits', 0) or 0)}, misses={int(serving_grid.get('exact_top_cache_misses', 0) or 0)}, entries={int(serving_grid.get('exact_top_cache_entries', 0) or 0)}`",
            "",
            "| profile | elapsed ms | index build ms | dominant build phase | known build ms | unattributed build ms | centroid build ms | centroid cluster ms | centroid residual ms | proxy build ms | doc sidecar write ms | centroid sidecar write ms | centroid posting write ms | exact baseline ms | retrieval ms | exact scans | retrieval queries | runs |",
            "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        profile_summaries = serving_grid.get("profile_summaries", [])
        if isinstance(profile_summaries, list):
            for item in profile_summaries:
                if not isinstance(item, dict):
                    continue
                lines.append(
                    "| {profile} | {elapsed:.3f} | {index:.3f} | {dominant} | {known:.3f} | {unknown:.3f} | {centroid:.3f} | {centroid_cluster:.3f} | {centroid_residual:.3f} | {proxy:.3f} | {doc_sidecar:.3f} | {centroid_sidecar:.3f} | {posting:.3f} | {exact:.3f} | {retrieval:.3f} | {exact_count} | {retrieval_count} | {runs} |".format(
                        profile=item.get("profile", ""),
                        elapsed=float(item.get("profile_elapsed_ms", 0.0) or 0.0),
                        index=float(item.get("index_build_elapsed_ms", 0.0) or 0.0),
                        dominant=item.get("dominant_build_phase", ""),
                        known=float(item.get("build_phase_known_ms", 0.0) or 0.0),
                        unknown=float(item.get("build_phase_unattributed_ms", 0.0) or 0.0),
                        centroid=float(item.get("multivector_centroid_build_us", 0.0) or 0.0) / 1000.0,
                        centroid_cluster=float(item.get("multivector_centroid_cluster_us", 0.0) or 0.0) / 1000.0,
                        centroid_residual=float(item.get("multivector_centroid_residual_us", 0.0) or 0.0) / 1000.0,
                        proxy=float(item.get("multivector_proxy_build_us", 0.0) or 0.0) / 1000.0,
                        doc_sidecar=float(item.get("multivector_doc_sidecar_write_us", 0.0) or 0.0) / 1000.0,
                        centroid_sidecar=float(item.get("multivector_centroid_sidecar_write_us", 0.0) or 0.0) / 1000.0,
                        posting=float(item.get("multivector_centroid_posting_write_us", 0.0) or 0.0) / 1000.0,
                        exact=float(item.get("exact_baseline_elapsed_ms_total", 0.0) or 0.0),
                        retrieval=float(item.get("retrieval_elapsed_ms_total", 0.0) or 0.0),
                        exact_count=int(item.get("exact_baseline_query_count", 0) or 0),
                        retrieval_count=int(item.get("retrieval_query_count", 0) or 0),
                        runs=int(item.get("runs", 0) or 0),
                    )
                )
        warned_rows = [
            item
            for item in sorted_rows
            if isinstance(item.get("serving_slow_path_warnings"), list)
            and item.get("serving_slow_path_warnings")
        ]
        lines.extend([
            "",
            "#### Slow path warnings",
            "",
        ])
        if warned_rows:
            lines.extend([
                "| profile | ef | oversampling | failed | warnings |",
                "|---|---:|---:|---|---|",
            ])
            for item in warned_rows:
                warnings = item.get("serving_slow_path_warnings", [])
                if not isinstance(warnings, list):
                    warnings = []
                lines.append(
                    "| {profile} | {ef} | {oversampling} | {failed} | {warnings} |".format(
                        profile=item.get("profile", ""),
                        ef=int(item.get("ef", 0) or 0),
                        oversampling=int(item.get("oversampling", 0) or 0),
                        failed=bool(item.get("serving_slow_path_failed", False)),
                        warnings=", ".join(f"`{warning}`" for warning in warnings),
                    )
                )
        else:
            lines.append("- None")

    serving_rec = report.get("document_node_serving_recommendation")
    if isinstance(serving_rec, dict):
        thresholds = serving_rec.get("thresholds", {})
        best_latency_safe = serving_rec.get("best_latency_safe") or {}
        best_quality = serving_rec.get("best_quality") or {}
        best_balanced = serving_rec.get("best_balanced") or {}
        if not isinstance(thresholds, dict):
            thresholds = {}
        if not isinstance(best_latency_safe, dict):
            best_latency_safe = {}
        if not isinstance(best_quality, dict):
            best_quality = {}
        if not isinstance(best_balanced, dict):
            best_balanced = {}
        lines.extend([
            "",
            "### Document-node serving recommendation",
            "",
            f"- Minimum top-10 admission: `{float(thresholds.get('serving_min_top10_admission', 0.0)):.3f}`",
            f"- Minimum NDCG ratio vs exact: `{float(thresholds.get('serving_min_ndcg_ratio_vs_exact', 0.0)):.3f}`",
            f"- Maximum p95 ms: `{float(thresholds.get('serving_max_p95_ms', 0.0)):.3f}` (`0` means no hard cap)",
            f"- Best latency-safe: `{best_latency_safe.get('id', '')}`",
            f"- Best quality: `{best_quality.get('id', '')}`",
            f"- Best balanced: `{best_balanced.get('id', '')}`",
            f"- Why latency-safe won: {serving_profile_explanation(best_latency_safe)}",
            f"- Why quality won: {serving_profile_explanation(best_quality)}",
            f"- Why balanced won: {serving_profile_explanation(best_balanced)}",
            "",
            "| pass | profile | source | proxy | centroid cap | entry samples | entry sidecar | storage | pooling | ef | oversampling | budget | effective candidates | underfill | top10 admission | ndcg@10 | p95 ms | reasons | hints |",
            "|---|---|---|---|---:|---:|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---|---|",
        ])
        rows = serving_rec.get("profile_thresholds", [])
        if isinstance(rows, list):
            for item in rows:
                if not isinstance(item, dict):
                    continue
                failures = item.get("failure_reasons", [])
                unavailable = item.get("unavailable_criteria", [])
                reasons = []
                if isinstance(failures, list):
                    reasons.extend(str(reason) for reason in failures)
                if isinstance(unavailable, list):
                    reasons.extend(f"unavailable:{criterion}" for criterion in unavailable)
                hints = item.get("admission_improvement_hints", [])
                if not isinstance(hints, list):
                    hints = []
                lines.append(
                    "| {passed} | {profile} | {source} | {proxy} | {cap} | {entry_samples} | {entry_sidecar} | {storage} | {pooling}:{ratio:.2f} | {ef} | {oversampling} | {budget} | {effective_candidates} | {underfill} | {admission:.6f} | {ndcg:.6f} | {p95:.3f} | {reasons} | {hints} |".format(
                        passed="pass" if item.get("threshold_pass") else "fail",
                        profile=item.get("profile", ""),
                        source=item.get("candidate_source", ""),
                        proxy=item.get("proxy_encoder", ""),
                        cap=int(item.get("centroid_lite_max_postings_per_token", 0) or 0),
                        entry_samples=int(
                            item.get(
                                "effective_entry_sample_count",
                                item.get("entry_sample_count", 0),
                            )
                            or 0
                        ),
                        entry_sidecar=(
                            f"on:{item.get('entry_sidecar_representatives', '')}:{item.get('entry_sidecar_strategy', '')}"
                            if bool(item.get("entry_sidecar", False))
                            else "off"
                        ),
                        storage=item.get("storage_kind", ""),
                        pooling=item.get("token_pooling", ""),
                        ratio=float(item.get("token_pooling_target_ratio", 1.0) or 1.0),
                        ef=int(item.get("ef", 0) or 0),
                        oversampling=int(item.get("oversampling", 0) or 0),
                        budget=int(item.get("largest_budget", 0) or 0),
                        effective_candidates=int(item.get("effective_candidate_k", 0) or 0),
                        underfill=item.get("candidate_underfill_reason", ""),
                        admission=float(item.get("exact_top10_admission_recall", 0.0) or 0.0),
                        ndcg=float(item.get("ndcg@10", 0.0) or 0.0),
                        p95=float(serving_row_p95(item) or 0.0),
                        reasons=", ".join(reasons) if reasons else "",
                        hints=", ".join(str(hint) for hint in hints),
                    )
                )
        bottlenecks = serving_rec.get("candidate_admission_bottlenecks", [])
        if isinstance(bottlenecks, list) and bottlenecks:
            lines.extend([
                "",
                "#### Candidate admission bottlenecks",
                "",
                "These rows exhausted the admitted candidate band during exact rerank "
                "but still missed the top-10 admission threshold. Increase rerank depth "
                "only after improving candidate generation.",
                "",
                "| profile | source | proxy | entry samples | entry sidecar | ef | oversampling | budget | top10 admission | p95 ms | reason | hints |",
                "|---|---|---|---:|---|---:|---:|---:|---:|---:|---|---|",
            ])
            for item in bottlenecks:
                if not isinstance(item, dict):
                    continue
                hints = item.get("admission_improvement_hints", [])
                if not isinstance(hints, list):
                    hints = []
                lines.append(
                    "| {profile} | {source} | {proxy} | {entry_samples} | {entry_sidecar} | {ef} | {oversampling} | {budget} | {admission:.6f} | {p95:.3f} | {reason} | {hints} |".format(
                        profile=item.get("profile", ""),
                        source=item.get("candidate_source", ""),
                        proxy=item.get("proxy_encoder", ""),
                        entry_samples=int(
                            item.get(
                                "effective_entry_sample_count",
                                item.get("entry_sample_count", 0),
                            )
                            or 0
                        ),
                        entry_sidecar=(
                            f"on:{item.get('entry_sidecar_representatives', '')}:{item.get('entry_sidecar_strategy', '')}"
                            if bool(item.get("entry_sidecar", False))
                            else "off"
                        ),
                        ef=int(item.get("ef", 0) or 0),
                        oversampling=int(item.get("oversampling", 0) or 0),
                        budget=int(item.get("largest_budget", 0) or 0),
                        admission=float(item.get("exact_top10_admission_recall", 0.0) or 0.0),
                        p95=float(serving_row_p95(item) or 0.0),
                        reason=item.get("candidate_admission_bottleneck_reason", ""),
                        hints=", ".join(str(hint) for hint in hints),
                    )
                )
        delta_summary = serving_rec.get("candidate_source_delta_summary", {})
        if isinstance(delta_summary, dict):
            lines.extend([
                "",
                "#### Candidate-source delta summary",
                "",
            ])
            if delta_summary.get("available"):
                def delta_item_line(label: str, item: Any) -> str:
                    if not isinstance(item, dict) or not item:
                        return f"- {label}: unavailable"
                    return (
                        "- {label}: `{profile}` vs `{baseline}` "
                        "({comparison}, ef={ef}, os={oversampling}, budget={budget}); "
                        "top10 delta `{top10:.6f}`, ndcg delta `{ndcg:.6f}`, "
                        "p95 delta `{p95:.3f} ms`"
                    ).format(
                        label=label,
                        profile=item.get("profile", ""),
                        baseline=item.get("baseline_profile", ""),
                        comparison=item.get("comparison", ""),
                        ef=int(item.get("ef", 0) or 0),
                        oversampling=int(item.get("oversampling", 0) or 0),
                        budget=int(item.get("candidate_budget", 0) or 0),
                        top10=float(
                            item.get("exact_top10_admission_recall_delta", 0.0)
                            or 0.0
                        ),
                        ndcg=float(item.get("ndcg@10_delta", 0.0) or 0.0),
                        p95=float(item.get("p95_ms_delta", 0.0) or 0.0),
                    )

                lines.extend([
                    f"- Paired comparisons: `{int(delta_summary.get('comparison_count', 0) or 0)}`",
                    f"- Usable paired comparisons: `{int(delta_summary.get('usable_comparison_count', 0) or 0)}`",
                    f"- Unusable paired comparisons: `{int(delta_summary.get('unusable_comparison_count', 0) or 0)}`",
                    f"- Positive admission deltas: `{int(delta_summary.get('positive_admission_delta_count', 0) or 0)}`",
                    f"- Positive quality deltas: `{int(delta_summary.get('positive_quality_delta_count', 0) or 0)}`",
                    f"- Latency-improving deltas: `{int(delta_summary.get('latency_improvement_count', 0) or 0)}`",
                    delta_item_line(
                        "Best admission delta",
                        delta_summary.get("best_admission_delta"),
                    ),
                    delta_item_line(
                        "Best quality delta",
                        delta_summary.get("best_quality_delta"),
                    ),
                    delta_item_line(
                        "Best latency delta",
                        delta_summary.get("best_latency_delta"),
                    ),
                ])
                by_comparison = delta_summary.get("best_admission_delta_by_comparison", {})
                if isinstance(by_comparison, dict) and by_comparison:
                    lines.extend([
                        "",
                        "| comparison | best admission profile | baseline | top10 delta | ndcg delta | p95 delta ms |",
                        "|---|---|---|---:|---:|---:|",
                    ])
                    for comparison in sorted(by_comparison):
                        item = by_comparison.get(comparison)
                        if not isinstance(item, dict) or not item:
                            continue
                        lines.append(
                            "| {comparison} | {profile} | {baseline} | {top10:.6f} | {ndcg:.6f} | {p95:.3f} |".format(
                                comparison=comparison,
                                profile=item.get("profile", ""),
                                baseline=item.get("baseline_profile", ""),
                                top10=float(
                                    item.get("exact_top10_admission_recall_delta", 0.0)
                                    or 0.0
                                ),
                                ndcg=float(item.get("ndcg@10_delta", 0.0) or 0.0),
                                p95=float(item.get("p95_ms_delta", 0.0) or 0.0),
                            )
                        )
            else:
                lines.append(
                    f"- Unavailable: `{delta_summary.get('reason', 'unknown')}`"
                )
        lines.extend([
            "",
            "The balanced score is stable and simple: latency rank plus admission "
            "loss, optional NDCG loss when qrels exist, and a small storage "
            "penalty (`sq8=0`, `f16=1`, `f32=2`). Experimental profiles remain "
            "opt-in and should not be baked into serving defaults; they receive "
            "a large balanced-score penalty whenever a non-experimental profile "
            "has usable metrics. Learned-sparse rows with partial fixture coverage "
            "and reservoir rows without execution evidence also receive a large "
            "penalty and fail the latency-safe threshold. "
            "Final ranking remains exact heap MaxSim unless the candidate source "
            "is explicitly experimental.",
        ])

    token_pooling_rec = report.get("document_node_token_pooling_recommendation")
    if not isinstance(token_pooling_rec, dict):
        serving_grid_for_token_pooling = report.get("document_node_serving_grid")
        if isinstance(serving_grid_for_token_pooling, dict):
            token_pooling_rec = serving_grid_for_token_pooling.get(
                "token_pooling_recommendation",
            )
    if isinstance(token_pooling_rec, dict):
        best_pooling_latency_safe = token_pooling_rec.get("best_pooling_latency_safe") or {}
        best_pooling_quality_safe = token_pooling_rec.get("best_pooling_quality_safe") or {}
        if not isinstance(best_pooling_latency_safe, dict):
            best_pooling_latency_safe = {}
        if not isinstance(best_pooling_quality_safe, dict):
            best_pooling_quality_safe = {}
        lines.extend([
            "",
            "### Document-node token-pooling recommendation",
            "",
            f"- Available: `{bool(token_pooling_rec.get('available', False))}`",
            f"- Profiles evaluated: `{int(token_pooling_rec.get('profile_count', 0) or 0)}`",
            f"- Selection basis: `{token_pooling_rec.get('selection_basis', '')}`",
            f"- Best pooling latency-safe: `{best_pooling_latency_safe.get('id', '')}`",
            f"- Best pooling quality-safe: `{best_pooling_quality_safe.get('id', '')}`",
            f"- Why latency-safe won: {serving_profile_explanation(best_pooling_latency_safe)}",
            f"- Why quality-safe won: {serving_profile_explanation(best_pooling_quality_safe)}",
            "",
            "| pass | profile | proxy | pooling | tokens pooled p50 | pooling ratio p50 | exact pairs p50 | index build ms | top10 admission | ndcg@10 | p95 ms | reasons |",
            "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ])
        token_rows = token_pooling_rec.get("profile_thresholds", [])
        if isinstance(token_rows, list):
            for item in token_rows:
                if not isinstance(item, dict):
                    continue
                failures = item.get("failure_reasons", [])
                unavailable = item.get("unavailable_criteria", [])
                reasons: list[str] = []
                if isinstance(failures, list):
                    reasons.extend(str(reason) for reason in failures)
                if isinstance(unavailable, list):
                    reasons.extend(f"unavailable:{criterion}" for criterion in unavailable)
                lines.append(
                    "| {passed} | {profile} | {proxy} | {pooling}:{ratio:.2f} | {tokens_pooled:.3f} | {pooling_ratio:.6f} | {pairs:.3f} | {index_build:.3f} | {admission:.6f} | {ndcg:.6f} | {p95:.3f} | {reasons} |".format(
                        passed="pass" if item.get("threshold_pass") else "fail",
                        profile=item.get("profile", ""),
                        proxy=item.get("proxy_encoder", ""),
                        pooling=item.get("token_pooling", ""),
                        ratio=float(item.get("token_pooling_target_ratio", 1.0) or 1.0),
                        tokens_pooled=float(item.get("tokens_pooled_p50", 0.0) or 0.0),
                        pooling_ratio=float(item.get("token_pooling_ratio_p50", 0.0) or 0.0),
                        pairs=float(item.get("exact_rerank_pairs_p50", 0.0) or 0.0),
                        index_build=float(item.get("index_build_elapsed_ms", 0.0) or 0.0),
                        admission=float(item.get("exact_top10_admission_recall", 0.0) or 0.0),
                        ndcg=float(item.get("ndcg@10", 0.0) or 0.0),
                        p95=float(serving_row_p95(item) or 0.0),
                        reasons=", ".join(reasons) if reasons else "",
                    )
                )

    hybrid_eval = report.get("hybrid_evaluation")
    if isinstance(hybrid_eval, dict):
        recommended = hybrid_eval.get("recommended_default_profile", {})
        if not isinstance(recommended, dict):
            recommended = {}
        profile_recommendations = hybrid_eval.get("profile_recommendations", {})
        if not isinstance(profile_recommendations, dict):
            profile_recommendations = {}
        lines.extend([
            "",
            "### Hybrid Evaluation Harness",
            "",
            f"- Rebuilds index: `{hybrid_eval.get('rebuilds_index', False)}`",
            f"- Modes: `{','.join(str(item) for item in hybrid_eval.get('modes', []))}`",
            f"- Best quality mode: `{hybrid_eval.get('best_quality_mode', '')}`",
            f"- Best latency under quality floor: `{hybrid_eval.get('best_latency_under_quality_floor_mode', '')}`",
            f"- Recommended default mode: `{recommended.get('mode', '')}`",
            "",
            "| mode | recall@10 | ndcg@10 | mrr@10 | top1 admission | top10 admission | p50 ms | p95 ms | branch us p50 | candidates p50 | exact pairs p50 | sidecar bytes p50 | failure queries |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        results = hybrid_eval.get("results", [])
        if isinstance(results, list):
            for item in results:
                if not isinstance(item, dict):
                    continue
                metrics = item.get("metrics", {})
                latency = item.get("latency", {})
                work = item.get("scan_work", {})
                if not isinstance(metrics, dict):
                    metrics = {}
                if not isinstance(latency, dict):
                    latency = {}
                if not isinstance(work, dict):
                    work = {}
                branch_latency = work.get("branch_latency_us_total", {})
                branch_candidates = work.get("branch_candidates_total", {})
                exact_pairs = work.get("exact_maxsim_pairs", {})
                sidecar_bytes = work.get("sidecar_bytes", {})
                if not isinstance(branch_latency, dict):
                    branch_latency = {}
                if not isinstance(branch_candidates, dict):
                    branch_candidates = {}
                if not isinstance(exact_pairs, dict):
                    exact_pairs = {}
                if not isinstance(sidecar_bytes, dict):
                    sidecar_bytes = {}
                lines.append(
                    "| {mode} | {recall:.6f} | {ndcg:.6f} | {mrr:.6f} | {top1:.6f} | {top10:.6f} | {p50:.3f} | {p95:.3f} | {branch_p50:.3f} | {candidates_p50:.3f} | {pairs_p50:.3f} | {sidecar_p50:.3f} | {failures} |".format(
                        mode=item.get("mode", ""),
                        recall=float(metrics.get("recall@10", 0.0)),
                        ndcg=float(metrics.get("ndcg@10", 0.0)),
                        mrr=float(metrics.get("mrr@10", 0.0)),
                        top1=float(item.get("exact_top1_admission_rate", 0.0)),
                        top10=float(item.get("exact_top10_admission_recall", 0.0)),
                        p50=float(latency.get("p50_ms", 0.0)),
                        p95=float(latency.get("p95_ms", 0.0)),
                        branch_p50=float(branch_latency.get("p50", 0.0)),
                        candidates_p50=float(branch_candidates.get("p50", 0.0)),
                        pairs_p50=float(exact_pairs.get("p50", 0.0)),
                        sidecar_p50=float(sidecar_bytes.get("p50", 0.0)),
                        failures=int(item.get("failure_query_count", 0)),
                    )
                )
        if profile_recommendations:
            lines.extend([
                "",
                "| profile | mode | reason |",
                "|---|---|---|",
            ])
            for profile in ("latency", "balanced", "quality", "high_recall"):
                item = profile_recommendations.get(profile, {})
                if not isinstance(item, dict):
                    item = {}
                lines.append(
                    "| {profile} | {mode} | {reason} |".format(
                        profile=profile,
                        mode=item.get("mode", ""),
                        reason=item.get("reason", ""),
                    )
                )
        lines.extend([
            "",
            "The harness compares exact scan, document-node retrieval, BM25 "
            "admission-only, BM25 RRF, BM25 DBSF, proxy-vector document-node "
            "prefetch, and learned-sparse admission. Candidate admission failures "
            "list exact top-K documents not seen before final rerank, so a zero or "
            "low admission row is a candidate-generation failure, not a final-ranker "
            "success.",
        ])

    lines.append("")
    return "\n".join(lines)


def run_self_checks() -> None:
    _self_check_document_node_serving_profiles()
    _self_check_exact_top_cache()
    _self_check_document_node_serving_recommendation()
    _self_check_document_node_token_pooling_recommendation()
    _self_check_document_node_serving_recommendation_schema_fixture()
    _self_check_document_node_serving_stage_selection()
    _self_check_document_node_serving_stats_extraction()
    _self_check_document_node_serving_grid_serialization()
    _self_check_candidate_underfill_diagnostics()
    _self_check_document_node_candidate_source_deltas()
    _self_check_document_node_serving_build_only_serialization()
    _self_check_document_node_serving_latency_only()
    _self_check_learned_sparse_jsonl_parser()
    _self_check_learned_sparse_evidence_annotation()


def run_multivector_recall_gate(conn: psycopg.Connection[Any], args: argparse.Namespace) -> dict[str, Any]:
    setup_multivector_recall_gate(conn)
    exact_top = synthetic_exact_top(conn, 10)
    if not exact_top or exact_top[0]["doc_id"] != "good":
        raise RuntimeError(f"synthetic exact MaxSim top-1 is not good: {exact_top[:1]}")

    modes = [
        {"name": "token_graph", "graph_mode": "token_nodes", "candidate_source": "graph", "plain_fallback": "off", "reservoirs": "off"},
        {"name": "exact_token_scan", "graph_mode": "token_nodes", "candidate_source": "exact_token_scan", "plain_fallback": "off", "reservoirs": "off"},
        {"name": "reservoir_balanced", "graph_mode": "token_nodes", "candidate_source": "graph", "plain_fallback": "off", "reservoirs": "balanced"},
        {"name": "plain_fallback", "graph_mode": "token_nodes", "candidate_source": "graph", "plain_fallback": "force", "reservoirs": "off"},
        {"name": "exact_doc_scan", "graph_mode": "token_nodes", "candidate_source": "exact_doc_scan", "plain_fallback": "off", "reservoirs": "off"},
        {"name": "doc_graph_prototype", "graph_mode": "token_nodes", "candidate_source": "doc_graph_prototype", "plain_fallback": "off", "reservoirs": "off"},
        {"name": "document_nodes", "graph_mode": "document_nodes", "candidate_source": "document_nodes", "plain_fallback": "off", "reservoirs": "off"},
        {"name": "proxy_vector", "graph_mode": "document_nodes", "candidate_source": "proxy_vector", "plain_fallback": "off", "reservoirs": "off"},
    ]
    results = [run_synthetic_exact_scan(conn, args.final_k)]
    for mode in modes:
        results.append(run_synthetic_gate_mode(conn, mode, exact_top, args.recall_gate_budget, args.final_k))

    required_modes = {"exact_scan", "plain_fallback", "exact_doc_scan", "doc_graph_prototype", "document_nodes", "proxy_vector"}
    passed = all(
        item["mode"] not in required_modes
        or (item.get("top1") == "good" and item.get("exact_top1_admitted_before_rerank") is True)
        for item in results
    )
    report = {
        "suite": "multivector_recall_gate",
        "layer": "ir_quality_and_systems",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "synthetic_many_moderate",
        "dbpedia_status": "not_required",
        "candidate_budget": args.recall_gate_budget,
        "final_k": args.final_k,
        "exact_top1": exact_top[0]["doc_id"],
        "exact_top": exact_top,
        "passed": passed,
        "results": results,
    }
    report["markdown_summary"] = markdown_recall_gate_summary(report)
    return report


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    env_dataset = os.environ.get("DBPEDIA_DATASET")
    env_beir = os.environ.get("BEIR_DBPEDIA_DATASET")
    env_model = os.environ.get("PG_COLBERT_LLAMA_TEST_MODEL")
    env_precomputed = os.environ.get("DBPEDIA_COLBERT_PRECOMPUTED_DATASET")
    if args.dataset is None and env_dataset:
        args.dataset = Path(env_dataset)
    if args.beir_dataset is None and env_beir:
        args.beir_dataset = Path(env_beir)
    if args.precomputed_dataset is None and env_precomputed:
        args.precomputed_dataset = env_precomputed
    if args.model_path is None:
        args.model_path = Path(env_model or DEFAULT_MODEL_PATH)
    require_complete_learned_sparse_args(args)
    if args.token_ablation_skip_tokens and not args.token_ablation_query_id:
        raise SystemExit("--token-ablation-skip-tokens requires --token-ablation-query-id")
    if args.token_ablation_final_k is not None and args.token_ablation_final_k <= 0:
        raise SystemExit("--token-ablation-final-k must be positive")
    if args.token_ablation_dense_k is not None and args.token_ablation_dense_k <= 0:
        raise SystemExit("--token-ablation-dense-k must be positive")

    if args.precomputed_dataset is None and not args.reuse_data and args.dataset is None and not args.multivector_recall_gate:
        raise SystemExit("pass --dataset or set DBPEDIA_DATASET")
    if args.dataset is not None:
        args.dataset = args.dataset.resolve()
        if not args.dataset.exists():
            raise SystemExit(f"dataset path does not exist: {args.dataset}")
    if args.precomputed_dataset is None and args.beir_dataset is None and not args.multivector_recall_gate:
        raise SystemExit("pass --beir-dataset or set BEIR_DBPEDIA_DATASET")
    if args.beir_dataset is not None:
        args.beir_dataset = args.beir_dataset.resolve()
    if args.beir_dataset is not None and not args.beir_dataset.exists():
        raise SystemExit(f"BEIR dataset path does not exist: {args.beir_dataset}")
    if args.learned_sparse_doc_jsonl is not None:
        args.learned_sparse_doc_jsonl = args.learned_sparse_doc_jsonl.resolve()
        if not args.learned_sparse_doc_jsonl.is_file():
            raise SystemExit(f"learned sparse document JSONL does not exist: {args.learned_sparse_doc_jsonl}")
    if args.learned_sparse_query_jsonl is not None:
        args.learned_sparse_query_jsonl = args.learned_sparse_query_jsonl.resolve()
        if not args.learned_sparse_query_jsonl.is_file():
            raise SystemExit(f"learned sparse query JSONL does not exist: {args.learned_sparse_query_jsonl}")
    if args.document_node_serving_grid or args.document_node_serving_build_only:
        effective_document_node_serving_profiles(args)
    if args.document_node_serving_latency_only:
        document_node_serving_latency_profile(args)

    args.model_path = args.model_path.resolve()
    if args.precomputed_dataset is None and not args.model_path.is_file() and not args.multivector_recall_gate:
        raise SystemExit(
            f"model file does not exist: {args.model_path}\n"
            "Use johannhartmann/SauerkrautLM-Multi-ColBERT-15m-GGUF and pass --model-path."
        )
    if args.model_alias is None:
        args.model_alias = args.model_path.stem
    args.expected_dim_requested = str(args.expected_dim)
    args.expected_dim, args.expected_dim_source = resolve_expected_dim(
        args.expected_dim_requested,
        args.colbert_model_name,
    )
    methods = [method.strip() for method in args.methods.split(",") if method.strip()]
    unknown = sorted(set(methods) - {QUERY_ONLY_METHOD, RRF_METHOD, EXACT_SCAN_METHOD})
    if unknown:
        raise SystemExit(f"unknown method(s): {', '.join(unknown)}")
    args.methods = methods
    if args.generation_clients is None:
        args.generation_clients = args.clients
    if args.generation_clients < 1:
        raise SystemExit("--generation-clients must be at least 1")
    if args.precomputed_batch_size < 1:
        raise SystemExit("--precomputed-batch-size must be at least 1")
    if args.generation_threads < 1:
        raise SystemExit("--generation-threads must be at least 1")
    if args.generation_n_gpu_layers < 0:
        raise SystemExit("--generation-n-gpu-layers must be at least 0")
    if args.generation_batch_sequences < 1:
        raise SystemExit("--generation-batch-sequences must be at least 1")
    if args.generation_n_batch is None:
        args.generation_n_batch = max(512, args.generation_batch_sequences * args.max_doc_vectors)
    if args.generation_n_batch < 1:
        raise SystemExit("--generation-n-batch must be at least 1")
    if args.output is None:
        output_stem = "multivector-recall-gate" if args.multivector_recall_gate else "dbpedia-colbert-multivector"
        args.output = Path("benchmarks/results/" + datetime.now(timezone.utc).strftime(f"{output_stem}-%Y%m%dT%H%M%SZ.json"))
    args.output = args.output.resolve()
    if args.markdown_output is not None:
        args.markdown_output = args.markdown_output.resolve()
    if args.recall_gate_markdown_output is not None:
        args.recall_gate_markdown_output = args.recall_gate_markdown_output.resolve()
    if args.quality_k > args.final_k:
        print(
            f"--quality-k {args.quality_k} is greater than --final-k {args.final_k}; "
            "quality metrics will be capped by final_k",
            file=sys.stderr,
        )
    if args.clients < 1:
        raise SystemExit("--clients must be at least 1")
    if args.warm_queries < 0:
        raise SystemExit("--warm-queries must be non-negative")
    if args.timed_queries < 0:
        raise SystemExit("--timed-queries must be non-negative")
    if args.parallel_safety_max_serial_p95_ms < 0:
        raise SystemExit("--parallel-safety-max-serial-p95-ms must be non-negative")
    if args.parallel_safety_max_docs_scored < 0:
        raise SystemExit("--parallel-safety-max-docs-scored must be non-negative")
    if args.parallel_safety_max_exact_rerank_docs < 0:
        raise SystemExit("--parallel-safety-max-exact-rerank-docs must be non-negative")
    if args.parallel_safety_max_exact_pairs < 0:
        raise SystemExit("--parallel-safety-max-exact-pairs must be non-negative")
    if args.parallel_safety_max_sidecar_bytes < 0:
        raise SystemExit("--parallel-safety-max-sidecar-bytes must be non-negative")
    if args.force_parallel_retrieval and args.skip_parallel_retrieval:
        raise SystemExit("--force-parallel-retrieval and --skip-parallel-retrieval are mutually exclusive")
    if (
        args.document_node_serving_grid_smoke
        or args.document_node_serving_grid_proxy_admission_focus
        or args.document_node_serving_grid_centroid_lite_focus
        or args.document_node_serving_grid_token_pooling_focus
    ):
        args.document_node_serving_grid = True
    if args.document_node_serving_grid_budget_mode is None:
        args.document_node_serving_grid_budget_mode = (
            "largest_only" if args.document_node_serving_grid else "sweep"
        )
    if args.document_node_serving_grid_stage_mode is None:
        args.document_node_serving_grid_stage_mode = (
            "two_stage" if args.document_node_serving_grid_smoke else "single"
        )
    if args.serving_exact_rerank_mode is None:
        args.serving_exact_rerank_mode = (
            "serving"
            if (
                args.document_node_serving_grid
                or args.document_node_serving_latency_only
                or args.document_node_serving_build_only
            )
            else "admission_exhaustive"
        )
    if args.admission_k < 1:
        raise SystemExit("--admission-k must be at least 1")
    if args.admission_trace_limit < 0:
        raise SystemExit("--admission-trace-limit must be non-negative")
    if args.multivector_plain_fallback_max_docs < 0:
        raise SystemExit("--multivector-plain-fallback-max-docs must be non-negative")
    if not 0.0 <= args.multivector_plain_fallback_candidate_fraction <= 1.0:
        raise SystemExit("--multivector-plain-fallback-candidate-fraction must be between 0 and 1")
    if args.multivector_doc_graph_search_ef < 0:
        raise SystemExit("--multivector-doc-graph-search-ef must be non-negative")
    if args.multivector_doc_graph_oversampling < 1:
        raise SystemExit("--multivector-doc-graph-oversampling must be at least 1")
    if args.multivector_doc_graph_rescore_k < 0:
        raise SystemExit("--multivector-doc-graph-rescore-k must be non-negative")
    if args.serving_exact_rerank_k < 1:
        raise SystemExit("--serving-exact-rerank-k must be positive")
    if args.serving_ef < 1:
        raise SystemExit("--serving-ef must be positive")
    if args.serving_oversampling < 1:
        raise SystemExit("--serving-oversampling must be at least 1")
    if args.serving_candidate_k < 1:
        raise SystemExit("--serving-candidate-k must be positive")
    if (
        args.document_node_serving_latency_only
        or args.document_node_serving_build_only
    ) and args.serving_profile_name is not None:
        document_node_serving_profile_by_name(args, str(args.serving_profile_name))
    if args.serving_grid_probe_queries < 1:
        raise SystemExit("--serving-grid-probe-queries must be positive")
    if args.serving_grid_finalists < 1:
        raise SystemExit("--serving-grid-finalists must be positive")
    if args.index_graph_m < 0:
        raise SystemExit("--index-graph-m must be non-negative")
    if args.index_graph_ef_construction < 0:
        raise SystemExit("--index-graph-ef-construction must be non-negative")
    if args.index_graph_ef_search < 0:
        raise SystemExit("--index-graph-ef-search must be non-negative")
    if args.index_native_segments < 0:
        raise SystemExit("--index-native-segments must be non-negative")
    if args.multivector_doc_build_scorer == "exact_symmetric":
        if not args.allow_exact_symmetric_build:
            raise SystemExit(
                "--multivector-doc-build-scorer exact_symmetric requires "
                "--allow-exact-symmetric-build"
            )
        print(
            "WARNING: exact_symmetric document-node graph builds compute exact "
            "document-document MaxSim inside topology construction; cost is "
            "O(distance_calls * La * Lb * d). Use proxy for DBpedia benchmark runs.",
            file=sys.stderr,
        )
    if not 0.0 < args.multivector_token_pooling_target_ratio <= 1.0:
        raise SystemExit("--multivector-token-pooling-target-ratio must be in (0, 1]")
    if args.multivector_token_pooling_min_tokens < 1:
        raise SystemExit("--multivector-token-pooling-min-tokens must be positive")
    if args.multivector_centroid_lite_max_postings_per_token < 0:
        raise SystemExit("--multivector-centroid-lite-max-postings-per-token must be non-negative")
    if args.multivector_per_token_doc_reservoir_k < 0:
        raise SystemExit("--multivector-per-token-doc-reservoir-k must be non-negative")
    if args.multivector_coverage_reservoir_k < 0:
        raise SystemExit("--multivector-coverage-reservoir-k must be non-negative")
    if args.admission_budget_sweep is None:
        args.admission_budget_sweep = ",".join(
            str(budget) for budget in effective_serving_grid_budget_sweep(args)
        )
    admission_budgets = effective_serving_grid_budget_sweep(args)
    if (
        args.admission_debug
        or args.document_node_admission_grid
        or args.document_node_serving_grid
        or args.hybrid_evaluation_harness
    ) and (
        not admission_budgets or any(budget < 1 for budget in admission_budgets)
    ):
        raise SystemExit("--admission-budget-sweep must contain positive integer budgets")
    if not 0.0 < args.hybrid_evaluation_quality_floor <= 1.0:
        raise SystemExit("--hybrid-evaluation-quality-floor must be in (0, 1]")
    if args.hybrid_evaluation_dbsf_min_branch_candidates < 1:
        raise SystemExit("--hybrid-evaluation-dbsf-min-branch-candidates must be positive")
    if not 0.0 <= args.serving_min_top10_admission <= 1.0:
        raise SystemExit("--serving-min-top10-admission must be in [0, 1]")
    if args.serving_min_ndcg_ratio_vs_exact < 0.0:
        raise SystemExit("--serving-min-ndcg-ratio-vs-exact must be non-negative")
    if args.serving_max_p95_ms < 0.0:
        raise SystemExit("--serving-max-p95-ms must be non-negative")
    hybrid_modes = [
        mode.strip()
        for mode in args.hybrid_evaluation_modes.split(",")
        if mode.strip()
    ]
    unknown_hybrid_modes = sorted(set(hybrid_modes) - set(SUPPORTED_HYBRID_EVALUATION_MODES))
    if unknown_hybrid_modes:
        raise SystemExit(
            "--hybrid-evaluation-modes contains unsupported value(s): "
            + ", ".join(unknown_hybrid_modes)
        )
    args.hybrid_evaluation_mode_values = hybrid_modes
    args.document_node_storage_grid_values = parse_choice_grid(
        args.document_node_storage_grid,
        DOCUMENT_NODE_STORAGE_CHOICES,
        "--document-node-storage-grid",
    )
    args.document_node_cache_grid_values = parse_choice_grid(
        args.document_node_cache_grid,
        DOCUMENT_NODE_STORAGE_CACHE_CHOICES,
        "--document-node-cache-grid",
    )
    args.document_node_ef_grid_values = parse_int_grid(
        args.document_node_ef_grid,
        "--document-node-ef-grid",
    )
    args.document_node_oversampling_grid_values = parse_int_grid(
        args.document_node_oversampling_grid,
        "--document-node-oversampling-grid",
    )
    args.document_node_serving_ef_grid_values = (
        parse_int_grid(
            args.document_node_serving_ef_grid,
            "--document-node-serving-ef-grid",
        )
        if args.document_node_serving_ef_grid
        else None
    )
    args.document_node_serving_oversampling_grid_values = (
        parse_int_grid(
            args.document_node_serving_oversampling_grid,
            "--document-node-serving-oversampling-grid",
        )
        if args.document_node_serving_oversampling_grid
        else None
    )
    args.document_node_serving_grid_centroid_lite_posting_cap_values = (
        effective_document_node_serving_centroid_lite_caps(args)
    )
    args.document_node_serving_grid_entry_sample_count_values = (
        effective_document_node_serving_entry_sample_counts(args)
    )
    args.document_node_serving_grid_entry_sidecar_representative_values = (
        effective_document_node_serving_entry_sidecar_representatives(args)
    )
    args.document_node_pooling_grid_values = parse_document_node_pooling_grid(
        args.document_node_pooling_grid,
    )
    args.document_node_proxy_encoder_grid_values = parse_choice_grid(
        args.document_node_proxy_encoder_grid,
        tuple(
            choice
            for choice in MULTIVECTOR_PROXY_ENCODER_CHOICES
            if choice not in {"learned_projection_placeholder", "learned_projection_v1"}
        ),
        "--document-node-proxy-encoder-grid",
    )
    if args.recall_gate_budget < 1:
        raise SystemExit("--recall-gate-budget must be positive")
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="run pure Python benchmark report construction self-checks and exit",
    )
    parser.add_argument("--database", default=os.environ.get("PGDATABASE", "pgturbohybrid_dbpedia_colbert"))
    parser.add_argument("--dataset", type=Path, default=None, help="Qdrant DBpedia parquet dataset root")
    parser.add_argument("--beir-dataset", type=Path, default=None, help="BEIR DBpedia dataset root containing queries")
    parser.add_argument("--qrels", default=None, help="BEIR DBpedia qrels TSV path")
    parser.add_argument(
        "--precomputed-dataset",
        default=None,
        help="local directory or Hugging Face dataset repo id with precomputed DBpedia ColBERT multivectors",
    )
    parser.add_argument(
        "--precomputed-batch-size",
        type=int,
        default=4096,
        help="Parquet/COPY batch size when loading --precomputed-dataset",
    )
    parser.add_argument("--model-path", type=Path, default=None, help="GGUF ColBERT model path")
    parser.add_argument("--model-alias", default=None, help="pg_colbert_llama model alias, defaults to model stem")
    parser.add_argument(
        "--colbert-model-name",
        default=DEFAULT_COLBERT_MODEL_NAME,
        help="late-interaction model profile name for dimension auto-detection and benchmark provenance",
    )
    parser.add_argument("--expected-dim", default="auto", help="expected embedding dimension or 'auto'")
    parser.add_argument("--max-doc-vectors", type=int, default=256)
    parser.add_argument("--max-query-vectors", type=int, default=32)
    parser.add_argument("--query-length", type=int, default=32)
    parser.add_argument("--max-docs", type=int, default=1000, help="documents to load; 0 means all available")
    parser.add_argument("--max-queries", type=int, default=32, help="queries to load from qrels; 0 means all")
    parser.add_argument(
        "--no-prioritize-qrels",
        dest="prioritize_qrels",
        action="store_false",
        help="load the first corpus rows instead of filling the sample with judged relevant documents first",
    )
    parser.add_argument("--generation-sample-docs", type=int, default=100)
    parser.add_argument("--insert-batch-size", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--methods", default=DEFAULT_METHODS)
    parser.add_argument(
        "--generation-clients",
        type=int,
        default=None,
        help="parallel PostgreSQL backends for document multivector generation; defaults to --clients",
    )
    parser.add_argument(
        "--generation-threads",
        type=int,
        default=1,
        help="llama.cpp worker threads per generation backend",
    )
    parser.add_argument(
        "--generation-n-gpu-layers",
        type=int,
        default=0,
        help="llama.cpp model layers to offload to GPU per generation backend",
    )
    parser.add_argument(
        "--generation-batch-sequences",
        type=int,
        default=8,
        help="independent documents per pg_colbert_llama C batch",
    )
    parser.add_argument(
        "--generation-n-batch",
        type=int,
        default=None,
        help="llama.cpp token batch budget for C batched generation; defaults to batch sequences times max doc vectors",
    )
    parser.add_argument(
        "--no-generation-warmup",
        dest="generation_warmup",
        action="store_false",
        help="include backend model/context warmup in persisted generation timing",
    )
    parser.add_argument("--dense-k", type=int, default=100)
    parser.add_argument("--bm25-k", type=int, default=100)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--multivector-subvector-k", type=int, default=100)
    parser.add_argument("--multivector-unique-docs-per-token", type=int, default=100)
    parser.add_argument("--multivector-max-raw-hits-per-token", type=int, default=400)
    parser.add_argument("--multivector-adaptive-widening", choices=("off", "auto", "on"), default="auto")
    parser.add_argument("--multivector-doc-candidate-k", type=int, default=100)
    parser.add_argument("--multivector-exact-rerank", choices=("off", "topk", "adaptive"), default="adaptive")
    parser.add_argument("--multivector-exact-rerank-k", type=int, default=100)
    parser.add_argument(
        "--serving-exact-rerank-k",
        type=int,
        default=100,
        help=(
            "bounded exact MaxSim rerank count for serving-grid serving mode "
            "and --document-node-serving-latency-only"
        ),
    )
    parser.add_argument(
        "--serving-exact-rerank-mode",
        choices=("serving", "admission_exhaustive"),
        default=None,
        help=(
            "exact rerank budget mode for admission diagnostics; default is "
            "serving for --document-node-serving-grid and admission_exhaustive otherwise"
        ),
    )
    parser.add_argument(
        "--multivector-graph",
        choices=("token_nodes", "document_nodes"),
        default="document_nodes",
        help="turbohybrid multivector graph layout used for the DBpedia ColBERT index",
    )
    parser.add_argument(
        "--multivector-doc-build-scorer",
        choices=MULTIVECTOR_DOC_BUILD_SCORER_CHOICES,
        default="proxy",
        help=(
            "document_nodes graph-build distance scorer; proxy is the production-safe "
            "default, exact_symmetric is a diagnostic-only O(distance_calls * La * Lb * d) path"
        ),
    )
    parser.add_argument(
        "--allow-exact-symmetric-build",
        action="store_true",
        help="allow diagnostic exact_symmetric document-document MaxSim graph topology builds",
    )
    parser.add_argument(
        "--multivector-doc-graph-search-ef",
        type=int,
        default=0,
        help="document-node graph traversal search_ef override; 0 uses the index graph_ef_search",
    )
    parser.add_argument(
        "--multivector-doc-graph-oversampling",
        type=int,
        default=1,
        help="candidate oversampling multiplier for document-node graph traversal",
    )
    parser.add_argument(
        "--multivector-doc-graph-rescore-k",
        type=int,
        default=0,
        help="document-node candidate rescore budget; 0 follows multivector_doc_candidate_k",
    )
    parser.add_argument(
        "--multivector-doc-graph-entry-sample-count",
        type=int,
        default=0,
        help=(
            "document-node graph entry samples scored before traversal; "
            "0 preserves the extension default"
        ),
    )
    parser.add_argument(
        "--index-graph-m",
        type=int,
        default=0,
        help="turbohybrid index graph_m reloption; 0 uses the extension default",
    )
    parser.add_argument(
        "--index-graph-ef-construction",
        type=int,
        default=0,
        help="turbohybrid index graph_ef_construction reloption; 0 uses the extension default",
    )
    parser.add_argument(
        "--index-graph-ef-search",
        type=int,
        default=0,
        help="turbohybrid index graph_ef_search reloption; 0 uses the extension default",
    )
    parser.add_argument(
        "--index-native-segments",
        type=int,
        default=0,
        help="turbohybrid index native_segments reloption; 0 uses the extension default",
    )
    parser.add_argument(
        "--multivector-doc-storage",
        choices=("f32", "f16", "sq8"),
        default="f32",
        help="document-node MaxSim sidecar scoring storage for graph traversal",
    )
    parser.add_argument(
        "--multivector-doc-storage-cache",
        choices=("auto", "resident", "paged"),
        default="auto",
        help="document-node sidecar cache mode; paged keeps graph adjacency resident and reports sidecar page reads per scan",
    )
    parser.add_argument(
        "--multivector-token-pooling",
        choices=DOCUMENT_NODE_TOKEN_POOLING_CHOICES,
        default="off",
        help="document-node index-time token pooling mode",
    )
    parser.add_argument(
        "--multivector-token-pooling-target-ratio",
        type=float,
        default=0.5,
        help="target pooled/original document-token ratio when token pooling is enabled",
    )
    parser.add_argument(
        "--multivector-token-pooling-min-tokens",
        type=int,
        default=16,
        help="minimum document token count before index-time token pooling is applied",
    )
    parser.add_argument(
        "--multivector-proxy-encoder",
        choices=MULTIVECTOR_PROXY_ENCODER_CHOICES,
        default="normalized_mean",
        help="document-node proxy_vector fixed-dimensional encoder used at index build and query time",
    )
    parser.add_argument(
        "--multivector-learned-projection-path",
        default="",
        help=(
            "administrator-provided learned_projection_v1 weight file used when "
            "--multivector-proxy-encoder learned_projection_v1 is selected"
        ),
    )
    parser.add_argument(
        "--multivector-learned-projection-model",
        default="",
        help="expected learned_projection_v1 model profile name; empty accepts the file header",
    )
    parser.add_argument(
        "--multivector-learned-projection-checksum",
        default="",
        help="expected learned_projection_v1 weight checksum/version string; empty accepts the file header",
    )
    parser.add_argument(
        "--multivector-centroids",
        choices=DOCUMENT_NODE_CENTROIDS_CHOICES,
        default="off",
        help="experimental PLAID-lite centroid mode persisted as an index reloption",
    )
    parser.add_argument(
        "--multivector-centroid-count",
        default="auto",
        help="per-document centroid count for centroid_lite; 'auto' maps to the extension default",
    )
    parser.add_argument(
        "--multivector-centroid-lite-max-postings-per-token",
        type=int,
        default=0,
        help=(
            "experimental centroid_lite posting-list cap per query token; "
            "0 leaves the persisted posting lists uncapped"
        ),
    )
    parser.add_argument(
        "--multivector-centroid-lite-pruning",
        choices=("off", "safe_upper_bound"),
        default="off",
        help=(
            "experimental centroid_lite pruning mode; safe_upper_bound keeps "
            "unsafe candidates and only prunes when a persisted centroid bound "
            "proves the document cannot enter the current candidate band"
        ),
    )
    parser.add_argument(
        "--multivector-candidate-source",
        choices=("graph", "document_nodes", "exact_token_scan", "exact_doc_scan", "doc_graph_prototype", "proxy_vector", "centroid_lite", "quantized_inverted_experimental"),
        default="graph",
        help="multivector candidate source for query-only/RRF retrieval; document_nodes/proxy_vector require multivector_graph=document_nodes; centroid_lite also supports token_nodes compatibility indexes built with multivector_centroids=kmeans; quantized_inverted_experimental is a guarded research-only document-node branch using persisted experimental codeword postings before exact MaxSim rerank",
    )
    parser.add_argument(
        "--multivector-plain-fallback",
        choices=("auto", "off", "force"),
        default="auto",
        help="exact heap MaxSim fallback for small or near-exhaustive multivector scans",
    )
    parser.add_argument("--multivector-plain-fallback-max-docs", type=int, default=1000)
    parser.add_argument("--multivector-plain-fallback-candidate-fraction", type=float, default=0.5)
    parser.add_argument(
        "--multivector-candidate-reservoirs",
        choices=("off", "conservative", "balanced"),
        default="conservative",
        help="bounded multivector candidate reservoir selection before exact rerank",
    )
    parser.add_argument("--multivector-per-token-doc-reservoir-k", type=int, default=1)
    parser.add_argument("--multivector-coverage-reservoir-k", type=int, default=10)
    parser.add_argument(
        "--multivector-bm25-candidate-injection",
        choices=("off", "hybrid_only", "dense_with_text"),
        default="off",
        help="inject BM25 candidates into the multivector exact MaxSim rerank candidate set",
    )
    parser.add_argument(
        "--multivector-sparse-candidate-source",
        choices=MULTIVECTOR_SPARSE_CANDIDATE_SOURCE_CHOICES,
        default="off",
        help=(
            "explicit sparse candidate source for multivector admission; "
            "learned_sparse reuses exported sparse postings and exact-MaxSim "
            "reranks admitted documents"
        ),
    )
    parser.add_argument(
        "--learned-sparse-doc-jsonl",
        type=Path,
        default=None,
        help="optional JSONL with doc_id, term_ids, and weights for learned-sparse document admission",
    )
    parser.add_argument(
        "--learned-sparse-query-jsonl",
        type=Path,
        default=None,
        help="optional JSONL with query_id, term_ids, and weights for learned-sparse query admission",
    )
    parser.add_argument(
        "--admission-debug",
        action="store_true",
        help="run exact-vs-candidate admission recall diagnostics after normal retrieval",
    )
    parser.add_argument(
        "--admission-debug-mode",
        choices=("off", "summary", "trace"),
        default=None,
        help=(
            "admission diagnostics mode; default is trace for --admission-debug "
            "and --document-node-admission-grid, summary for --document-node-serving-grid"
        ),
    )
    parser.add_argument(
        "--admission-k",
        type=int,
        default=10,
        help="exact top-K document set to compare against candidate admission",
    )
    parser.add_argument(
        "--admission-budget-sweep",
        default=None,
        help=(
            "comma-separated multivector document candidate budgets to sweep; "
            "--document-node-serving-grid-smoke defaults to 200,800, "
            f"--document-node-serving-grid defaults to {DOCUMENT_NODE_SERVING_GRID_BUDGET_SWEEP}, "
            f"other modes default to {DEFAULT_ADMISSION_BUDGET_SWEEP}"
        ),
    )
    parser.add_argument(
        "--admission-trace-limit",
        type=int,
        default=1000,
        help="bounded trace entries requested from turbohybrid_last_scan_stats()",
    )
    parser.add_argument(
        "--document-node-admission-grid",
        action="store_true",
        help="run DBpedia admission comparisons for token-node baselines and document-node storage/EF/oversampling grid",
    )
    parser.add_argument(
        "--document-node-serving-grid",
        action="store_true",
        help=(
            "run a compact production-oriented document-node serving grid over "
            "named f16/sq8 proxy and centroid profiles"
        ),
    )
    parser.add_argument(
        "--document-node-serving-grid-include-experimental",
        action="store_true",
        help=(
            "include the guarded quantized_inverted_experimental_f32 profile in "
            "--document-node-serving-grid"
        ),
    )
    parser.add_argument(
        "--document-node-serving-grid-include-proxy-encoders",
        action="store_true",
        help=(
            "include opt-in proxy encoder comparison profiles "
            "proxy_max_pool_f16 and proxy_random_projection_fde_f16 in the "
            "document-node serving grid/profile lookup"
        ),
    )
    parser.add_argument(
        "--document-node-serving-grid-include-learned-projection",
        action="store_true",
        help=(
            "include the opt-in proxy_learned_projection_v1_f16 serving-grid "
            "profile; requires --multivector-learned-projection-path at runtime"
        ),
    )
    parser.add_argument(
        "--document-node-serving-grid-include-bm25-rescue",
        action="store_true",
        help=(
            "include opt-in BM25 rescue profiles for candidate-admission "
            "experiments; this uses existing dense_with_text injection and is "
            "not part of the default serving grid; pair with "
            "--document-node-serving-grid-include-proxy-encoders to also add "
            "proxy_max_pool_f16_bm25_rescue"
        ),
    )
    parser.add_argument(
        "--document-node-serving-grid-include-learned-sparse-rescue",
        action="store_true",
        help=(
            "include opt-in learned-sparse rescue profiles for candidate-admission "
            "experiments; requires --learned-sparse-doc-jsonl and "
            "--learned-sparse-query-jsonl to run; pair with "
            "--document-node-serving-grid-include-proxy-encoders to also add "
            "proxy_max_pool_f16_learned_sparse_rescue"
        ),
    )
    parser.add_argument(
        "--document-node-serving-grid-include-reservoirs",
        action="store_true",
        help=(
            "include opt-in balanced candidate-reservoir profiles for "
            "document-node serving admission experiments; these use the existing "
            "bounded reservoir GUCs before exact MaxSim rerank and are not part "
            "of the default serving grid"
        ),
    )
    parser.add_argument(
        "--document-node-serving-grid-include-centroid-lite-caps",
        action="store_true",
        help=(
            "include opt-in capped centroid_lite profiles for posting-list pruning "
            "experiments; capped rows are not part of the default serving grid"
        ),
    )
    parser.add_argument(
        "--document-node-serving-grid-include-entry-samples",
        action="store_true",
        help=(
            "include opt-in proxy_vector entry-sample profiles for scan-time "
            "graph-entry admission experiments; these do not change the persisted "
            "index format and are not part of the default serving grid"
        ),
    )
    parser.add_argument(
        "--document-node-serving-grid-entry-sample-counts",
        default=DOCUMENT_NODE_SERVING_GRID_ENTRY_SAMPLE_SWEEP,
        help=(
            "comma-separated positive entry-sample counts used with "
            "--document-node-serving-grid-include-entry-samples; defaults to "
            f"{DOCUMENT_NODE_SERVING_GRID_ENTRY_SAMPLE_SWEEP}"
        ),
    )
    parser.add_argument(
        "--document-node-serving-grid-include-entry-sidecar",
        action="store_true",
        help=(
            "include opt-in proxy_vector entry-sidecar profiles for candidate-admission "
            "experiments; these persist graph entry representatives and are not part "
            "of the default serving grid"
        ),
    )
    parser.add_argument(
        "--document-node-serving-grid-entry-sidecar-representatives",
        default=DOCUMENT_NODE_SERVING_GRID_ENTRY_SIDECAR_SWEEP,
        help=(
            "comma-separated positive representative counts for "
            "--document-node-serving-grid-proxy-admission-focus entry-sidecar "
            f"variants; defaults to {DOCUMENT_NODE_SERVING_GRID_ENTRY_SIDECAR_SWEEP}"
        ),
    )
    parser.add_argument(
        "--document-node-serving-grid-proxy-admission-focus",
        action="store_true",
        help=(
            "run a focused document-node proxy/entry admission grid: "
            "normalized_mean, max_pool, centroid_mean, entry_sample_032/128, "
            "entry_sidecar_128/256, EF 100/200/400, oversampling 1/2, and "
            "largest-only candidate budget 800 unless overridden"
        ),
    )
    parser.add_argument(
        "--document-node-serving-grid-centroid-lite-focus",
        action="store_true",
        help=(
            "run a focused centroid_lite cap grid: uncapped, posting caps "
            "16/32/64, pooled centroid_lite, and centroid_mean baseline with "
            "largest-only candidate budget 800 unless overridden"
        ),
    )
    parser.add_argument(
        "--document-node-serving-grid-token-pooling-focus",
        action="store_true",
        help=(
            "run a focused token-pooling grid: normalized_mean and "
            "centroid_mean with no pooling and greedy_cosine target ratios "
            "0.75/0.50/0.33"
        ),
    )
    parser.add_argument(
        "--document-node-serving-grid-centroid-lite-posting-caps",
        default=DOCUMENT_NODE_SERVING_GRID_CENTROID_LITE_CAP_SWEEP,
        help=(
            "comma-separated positive per-query-token posting caps used with "
            "--document-node-serving-grid-include-centroid-lite-caps or "
            "--document-node-serving-grid-centroid-lite-focus; defaults to "
            f"{DOCUMENT_NODE_SERVING_GRID_CENTROID_LITE_CAP_SWEEP}"
        ),
    )
    parser.add_argument(
        "--document-node-serving-grid-profiles",
        default="",
        help=(
            "comma-separated subset of document-node serving-grid profile names; "
            "defaults to all serving profiles, or the smoke subset with "
            "--document-node-serving-grid-smoke"
        ),
    )
    parser.add_argument(
        "--document-node-serving-grid-smoke",
        action="store_true",
        help=(
            "run a smoke-only serving grid subset: proxy_normalized_mean_f16, "
            "centroid_mean_f16, centroid_lite_f16; EF 50,100; oversampling 1; "
            "budgets 200,800 unless --admission-budget-sweep is explicit"
        ),
    )
    parser.add_argument(
        "--document-node-serving-grid-budget-mode",
        choices=("largest_only", "sweep"),
        default=None,
        help=(
            "budget execution mode for --document-node-serving-grid; default is "
            "largest_only for serving-grid runs and sweep for other admission modes"
        ),
    )
    parser.add_argument(
        "--document-node-serving-grid-stage-mode",
        choices=("single", "two_stage"),
        default=None,
        help=(
            "serving-grid config execution mode; default is two_stage for "
            "--document-node-serving-grid-smoke and single for full serving-grid runs"
        ),
    )
    parser.add_argument(
        "--document-node-serving-ef-grid",
        default=None,
        help=(
            "comma-separated document-node graph search EF values for "
            "--document-node-serving-grid; defaults to 50,100,200, or 50,100 "
            "for --document-node-serving-grid-smoke"
        ),
    )
    parser.add_argument(
        "--document-node-serving-oversampling-grid",
        default=None,
        help=(
            "comma-separated document-node graph oversampling values for "
            "--document-node-serving-grid; defaults to 1,2, or 1 for "
            "--document-node-serving-grid-smoke"
        ),
    )
    parser.add_argument(
        "--serving-grid-probe-queries",
        type=int,
        default=10,
        help="number of first queries used by serving-grid two-stage probing",
    )
    parser.add_argument(
        "--serving-grid-finalists",
        type=int,
        default=6,
        help="non-experimental serving-grid configs fully evaluated after two-stage probing",
    )
    parser.add_argument(
        "--document-node-serving-grid-verify-exact-cache",
        action="store_true",
        help=(
            "debug-only check for --document-node-serving-grid: recompute one "
            "exact admission top-K result and compare cached doc_ids"
        ),
    )
    parser.add_argument(
        "--document-node-serving-latency-only",
        action="store_true",
        help=(
            "run one document-node serving profile as normal retrieval queries only; "
            "does not compute exact admission baselines"
        ),
    )
    parser.add_argument(
        "--document-node-serving-build-only",
        action="store_true",
        help=(
            "build the selected document-node serving profiles and report CREATE INDEX "
            "and sidecar construction costs without running retrieval or admission"
        ),
    )
    parser.add_argument(
        "--serving-profile-name",
        default=None,
        help=(
            "named profile for --document-node-serving-latency-only; defaults to "
            f"{DOCUMENT_NODE_SERVING_LATENCY_DEFAULT_PROFILE}"
        ),
    )
    parser.add_argument(
        "--serving-ef",
        type=int,
        default=100,
        help="document-node graph search EF for --document-node-serving-latency-only",
    )
    parser.add_argument(
        "--serving-oversampling",
        type=int,
        default=1,
        help="document-node graph oversampling for --document-node-serving-latency-only",
    )
    parser.add_argument(
        "--serving-candidate-k",
        type=int,
        default=DOCUMENT_NODE_SERVING_LATENCY_DEFAULT_CANDIDATE_K,
        help="document candidate budget for --document-node-serving-latency-only",
    )
    parser.add_argument(
        "--serving-storage",
        choices=DOCUMENT_NODE_STORAGE_CHOICES,
        default=None,
        help="override profile document-sidecar storage for --document-node-serving-latency-only",
    )
    parser.add_argument(
        "--serving-cache",
        choices=DOCUMENT_NODE_STORAGE_CACHE_CHOICES,
        default=None,
        help="override profile document-sidecar cache mode for --document-node-serving-latency-only",
    )
    parser.add_argument(
        "--serving-fail-on-slow-path",
        action="store_true",
        help="fail serving latency-only runs when scan stats indicate an accidental slow path",
    )
    parser.add_argument(
        "--serving-latency-warmup-queries",
        type=int,
        default=1,
        help=(
            "number of latency-only warmup queries to run before measured "
            "serving queries; warmup stats are reported separately"
        ),
    )
    parser.add_argument(
        "--serving-min-top10-admission",
        type=float,
        default=0.80,
        help="minimum exact top-10 admission recall for --document-node-serving-grid best_latency_safe",
    )
    parser.add_argument(
        "--serving-min-ndcg-ratio-vs-exact",
        type=float,
        default=0.95,
        help=(
            "minimum ndcg@10 ratio versus the exact-scan baseline for "
            "--document-node-serving-grid best_latency_safe; ignored with an "
            "unavailable-criteria note when qrels or exact baseline metrics are missing"
        ),
    )
    parser.add_argument(
        "--serving-max-p95-ms",
        type=float,
        default=0.0,
        help="hard p95 latency cap for --document-node-serving-grid; 0 disables the cap",
    )
    parser.add_argument(
        "--document-node-storage-grid",
        default="f32,f16,sq8",
        help="comma-separated document-node storage kinds to sweep when --document-node-admission-grid is enabled",
    )
    parser.add_argument(
        "--document-node-cache-grid",
        default="auto,paged",
        help="comma-separated document-node sidecar cache modes to sweep when --document-node-admission-grid is enabled",
    )
    parser.add_argument(
        "--document-node-ef-grid",
        default="50,100,200,400,800",
        help="comma-separated document-node graph search EF values to sweep",
    )
    parser.add_argument(
        "--document-node-oversampling-grid",
        default="1,2,4,8",
        help="comma-separated document-node graph oversampling multipliers to sweep",
    )
    parser.add_argument(
        "--document-node-pooling-grid",
        default="off:1.0,greedy_cosine:0.75,greedy_cosine:0.5,greedy_cosine:0.33",
        help="comma-separated pooling mode:ratio entries to rebuild document-node indexes for",
    )
    parser.add_argument(
        "--document-node-proxy-encoder-grid",
        default="normalized_mean,centroid_mean,max_pool,random_projection_fde",
        help="comma-separated proxy encoders to rebuild document-node indexes for",
    )
    parser.add_argument(
        "--hybrid-evaluation-harness",
        action="store_true",
        help="run Prompt 15 end-to-end hybrid comparison modes over a document-node index",
    )
    parser.add_argument(
        "--hybrid-evaluation-modes",
        default=",".join(HYBRID_EVALUATION_MODES),
        help=(
            "comma-separated Prompt 15 hybrid modes to run; "
            "quantized_inverted_experimental is also supported when named explicitly"
        ),
    )
    parser.add_argument(
        "--hybrid-evaluation-quality-floor",
        type=float,
        default=0.95,
        help="relative NDCG@10 floor used to pick the fastest recommended hybrid profile",
    )
    parser.add_argument(
        "--hybrid-evaluation-dbsf-min-branch-candidates",
        type=int,
        default=1,
        help="DBSF minimum branch candidates for the Prompt 15 harness",
    )
    parser.add_argument(
        "--multivector-recall-gate",
        action="store_true",
        help="run the deterministic synthetic many-moderate admission recall gate only",
    )
    parser.add_argument(
        "--recall-gate-budget",
        type=int,
        default=1,
        help="document candidate budget for the synthetic multivector recall gate",
    )
    parser.add_argument(
        "--recall-gate-markdown-output",
        type=Path,
        default=None,
        help="optional Markdown summary path for --multivector-recall-gate",
    )
    parser.add_argument(
        "--token-ablation-query-id",
        default=None,
        help="run one multivector query with per-token diagnostics for this query id",
    )
    parser.add_argument(
        "--token-ablation-skip-tokens",
        default="",
        help="comma-separated query token ordinals to skip in the token-ablation variant",
    )
    parser.add_argument(
        "--token-ablation-final-k",
        type=int,
        default=None,
        help="final_k for the optional token-ablation query; defaults to --final-k",
    )
    parser.add_argument(
        "--token-ablation-dense-k",
        type=int,
        default=None,
        help="dense_k for the optional token-ablation query; defaults to --dense-k",
    )
    parser.add_argument("--final-k", type=int, default=10)
    parser.add_argument("--quality-k", type=int, default=10)
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--warm-queries", type=int, default=2)
    parser.add_argument("--timed-queries", type=int, default=10)
    parser.add_argument(
        "--parallel-safety-preflight",
        choices=("serial_probe", "off"),
        default="serial_probe",
        help="skip the 8x phase when the serial probe already shows pathological multivector scan work",
    )
    parser.add_argument(
        "--parallel-safety-max-serial-p95-ms",
        type=float,
        default=DEFAULT_PARALLEL_SAFETY_MAX_SERIAL_P95_MS,
        help="maximum serial p95 latency allowed before the 8x phase is skipped",
    )
    parser.add_argument(
        "--parallel-safety-max-docs-scored",
        type=int,
        default=DEFAULT_PARALLEL_SAFETY_MAX_DOCS_SCORED,
        help="maximum per-query document candidates/docs scored allowed before the 8x phase is skipped",
    )
    parser.add_argument(
        "--parallel-safety-max-exact-rerank-docs",
        type=int,
        default=DEFAULT_PARALLEL_SAFETY_MAX_EXACT_RERANK_DOCS,
        help="maximum exact MaxSim-reranked documents allowed before the 8x phase is skipped",
    )
    parser.add_argument(
        "--parallel-safety-max-exact-pairs",
        type=int,
        default=DEFAULT_PARALLEL_SAFETY_MAX_EXACT_PAIRS,
        help="maximum exact MaxSim token-pair evaluations allowed before the 8x phase is skipped",
    )
    parser.add_argument(
        "--parallel-safety-max-sidecar-bytes",
        type=int,
        default=DEFAULT_PARALLEL_SAFETY_MAX_SIDECAR_BYTES,
        help="maximum per-query document-sidecar bytes touched allowed before the 8x phase is skipped",
    )
    parser.add_argument(
        "--force-parallel-retrieval",
        action="store_true",
        help="run the 8x retrieval phase even when the serial safety preflight exceeds configured thresholds",
    )
    parser.add_argument(
        "--skip-parallel-retrieval",
        action="store_true",
        help="record serial retrieval metrics only and skip the 8x throughput phase",
    )
    parser.add_argument("--reuse-data", action="store_true")
    parser.add_argument("--reuse-embeddings", action="store_true")
    parser.add_argument("--reuse-index", action="store_true")
    parser.add_argument("--force-reload", action="store_true")
    parser.add_argument(
        "--allow-unvalidated-embeddings",
        action="store_true",
        help="continue even if the Sauerkraut 15m tokenization preflight fails",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="optional Markdown summary path for benchmark results or --multivector-recall-gate",
    )
    args = parser.parse_args()
    args.admission_budget_sweep_explicit = args.admission_budget_sweep is not None
    args.serving_exact_rerank_mode_explicit = args.serving_exact_rerank_mode is not None
    args.admission_debug_mode_explicit = args.admission_debug_mode is not None
    if args.self_check:
        return args
    return validate_args(args)


def main() -> None:
    args = parse_args()
    if args.self_check:
        run_self_checks()
        print("self-check ok")
        return
    if args.multivector_recall_gate:
        conn = connect(args)
        try:
            report = run_multivector_recall_gate(conn, args)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            markdown_paths = {
                path
                for path in (args.markdown_output, args.recall_gate_markdown_output)
                if path is not None
            }
            for markdown_path in sorted(markdown_paths):
                markdown_path.parent.mkdir(parents=True, exist_ok=True)
                markdown_path.write_text(report["markdown_summary"], encoding="utf-8")
            print(args.output)
            for markdown_path in sorted(markdown_paths):
                print(markdown_path)
            if not report["passed"]:
                raise SystemExit(1)
        finally:
            conn.close()
        return

    if args.precomputed_dataset is None:
        queries = load_queries(args.beir_dataset)
        qrels_path = resolve_qrels_path(args.qrels, args.beir_dataset)
        all_qrels = read_qrels(qrels_path)
        qids = choose_query_ids(all_qrels, queries, args.max_queries)
        qrels = {qid: all_qrels[qid] for qid in qids}
    else:
        qrels_path = None
        qids = []
        queries = {}
        qrels = {}

    conn = connect(args)
    try:
        setup_schema(conn, include_colbert_llama=args.precomputed_dataset is None)
        if args.precomputed_dataset is None:
            embedding_health = validate_embedding_health(conn, args)
            load_phase = load_data(conn, args, qids, queries, qrels)
        else:
            embedding_health = {
                "validated": True,
                "skipped": True,
                "reason": "precomputed multivectors loaded from dataset",
            }
            load_phase = load_precomputed_multivectors(conn, args)
        doc_ids = selected_doc_ids(conn, args.max_docs)
        query_ids = selected_query_ids(conn)
        qrels = loaded_qrels(conn)
        if args.precomputed_dataset is None:
            generation_phase = measure_generation_sample(conn, args, doc_ids, query_ids)
            insert_phase = persist_multivectors(conn, args, doc_ids, query_ids)
        else:
            generation_phase = {
                "skipped": True,
                "reason": "precomputed multivectors loaded from dataset",
            }
            insert_phase = {
                "reused": True,
                "precomputed": True,
                "documents": len(doc_ids),
                "queries": len(query_ids),
                "source": args.precomputed_dataset,
                "note": "runtime generation skipped; rows were loaded from precomputed multivector dataset",
            }
        dataset_stats = multivector_dataset_stats(conn)
        learned_sparse_phase = load_learned_sparse_vectors(conn, args)
        encoded_queries = load_encoded_queries(conn)
        document_node_serving_latency_only = None
        document_node_serving_build_only = None

        if args.document_node_serving_build_only:
            document_node_serving_build_only = run_document_node_serving_build_only(
                conn,
                args,
            )
            index_phase = document_node_serving_build_only["last_index_phase"]
            result_methods = []
            admission_debug = None
            document_node_admission_grid = None
            document_node_serving_grid = None
            document_node_serving_recommendation = None
            document_node_token_pooling_recommendation = None
            hybrid_evaluation = None
            token_ablation = None
        elif args.document_node_serving_latency_only:
            document_node_serving_latency_only = run_document_node_serving_latency_only(
                conn,
                args,
                encoded_queries,
            )
            index_phase = document_node_serving_latency_only["index_phase"]
            result_methods: list[dict[str, Any]] = []
            admission_debug = None
            document_node_admission_grid = None
            document_node_serving_grid = None
            document_node_serving_recommendation = None
            document_node_token_pooling_recommendation = None
            hybrid_evaluation = None
            token_ablation = None
        else:
            index_phase = build_index(conn, args)
            set_retrieval_gucs(conn, args, "dbpedia_colbert_serial")

            result_methods = []
            for method in args.methods:
                run, serial_latencies, serial_stats = run_serial_retrieval(conn, method, encoded_queries, args, args.final_k)
                parallel = run_parallel_retrieval_report(
                    args,
                    method,
                    encoded_queries,
                    serial_latencies,
                    serial_stats,
                )
                result_methods.append({
                    "method": method,
                    "retrieval_mode": {
                        QUERY_ONLY_METHOD: "multivector_query_only",
                        RRF_METHOD: "multivector_plus_bm25_rrf",
                        EXACT_SCAN_METHOD: "multivector_exact_scan",
                    }[method],
                    "metrics": method_metrics(run, qrels, args.final_k, args.quality_k) if qrels else {},
                    "serial_latency": summarize_ms(serial_latencies),
                    "parallel_8x": parallel,
                    "top10_by_query": {qid: docs[:10] for qid, docs in run.items()},
                    "scan_summary": scan_summary(serial_stats),
                    "index_stats": index_phase.get("index_stats", {}),
                })

            admission_debug = None
            if args.admission_debug:
                admission_debug_args = (
                    args
                    if args.serving_exact_rerank_mode_explicit
                    else clone_args(args, serving_exact_rerank_mode="admission_exhaustive")
                )
                admission_debug_args = clone_args(
                    admission_debug_args,
                    admission_debug_context="explicit",
                )
                admission_debug = run_admission_debug(conn, admission_debug_args, encoded_queries)
            document_node_admission_grid = (
                run_document_node_admission_grid(conn, args, encoded_queries, qrels)
                if args.document_node_admission_grid
                else None
            )
            document_node_serving_grid = (
                run_document_node_serving_grid(conn, args, encoded_queries, qrels)
                if args.document_node_serving_grid
                else None
            )
            if document_node_serving_grid is not None:
                annotate_serving_grid_learned_sparse_evidence(
                    document_node_serving_grid,
                    learned_sparse_phase,
                )
            document_node_serving_recommendation = None
            if document_node_serving_grid is not None:
                recommendation_started = time.perf_counter()
                exact_baseline = serving_exact_baseline_from_results(result_methods)
                document_node_serving_recommendation = compute_document_node_serving_recommendation(
                    document_node_serving_grid,
                    exact_baseline=exact_baseline,
                    min_top10_admission=args.serving_min_top10_admission,
                    min_ndcg_ratio_vs_exact=args.serving_min_ndcg_ratio_vs_exact,
                    max_p95_ms=args.serving_max_p95_ms,
                )
                document_node_token_pooling_recommendation = None
                if bool(document_node_serving_grid.get("token_pooling_focus", False)):
                    document_node_token_pooling_recommendation = (
                        compute_document_node_token_pooling_recommendation(
                            document_node_serving_grid,
                            exact_baseline=exact_baseline,
                            min_top10_admission=args.serving_min_top10_admission,
                            min_ndcg_ratio_vs_exact=args.serving_min_ndcg_ratio_vs_exact,
                            max_p95_ms=args.serving_max_p95_ms,
                        )
                    )
                    document_node_serving_grid["token_pooling_recommendation"] = (
                        document_node_token_pooling_recommendation
                    )
                document_node_serving_grid["recommendation_elapsed_ms"] = elapsed_ms_since(
                    recommendation_started
                )
            else:
                document_node_token_pooling_recommendation = None
            hybrid_evaluation = (
                run_hybrid_evaluation_harness(conn, args, encoded_queries, qrels)
                if args.hybrid_evaluation_harness
                else None
            )
            token_ablation = run_token_ablation(conn, args, encoded_queries, qrels)

        output = {
            "suite": "dbpedia_colbert_multivector",
            "layer": "ir_quality_and_systems",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
            },
            "command": command_metadata(),
            "pgturbohybrid": {
                "git_sha": git_sha(),
            },
            "dataset": {
                "name": "dbpedia-colbert-multivector",
                "corpus_path": portable_path(args.dataset) if args.dataset else None,
                "beir_dataset_path": portable_path(args.beir_dataset) if args.beir_dataset else None,
                "qrels_path": portable_path(qrels_path) if qrels_path else None,
                "precomputed_dataset": args.precomputed_dataset,
                "stats": dataset_stats,
                "documents": load_phase["documents"],
                "queries": load_phase["queries"],
                "qrels": load_phase["qrels"],
                "max_docs": args.max_docs,
                "max_queries": args.max_queries,
                "prioritize_qrels": args.prioritize_qrels,
            },
            "model": {
                "gguf": "johannhartmann/SauerkrautLM-Multi-ColBERT-15m-GGUF",
                "source_model": "VAGOsolutions/SauerkrautLM-Multi-ColBERT-15m",
                "colbert_model_name": args.colbert_model_name,
                "model_path": portable_path(args.model_path),
                "model_alias": args.model_alias,
                "expected_dim": args.expected_dim,
                "expected_dim_requested": args.expected_dim_requested,
                "expected_dim_source": args.expected_dim_source,
                "max_doc_vectors": args.max_doc_vectors,
                "max_query_vectors": args.max_query_vectors,
                "query_length": args.query_length,
                "embedding_health": embedding_health,
            },
            "settings": {
                "dense_k": args.dense_k,
                "bm25_k": args.bm25_k,
                "rrf_k": args.rrf_k,
                "multivector_subvector_k": args.multivector_subvector_k,
                "multivector_unique_docs_per_token": args.multivector_unique_docs_per_token,
                "multivector_max_raw_hits_per_token": args.multivector_max_raw_hits_per_token,
                "multivector_adaptive_widening": args.multivector_adaptive_widening,
                "multivector_doc_candidate_k": args.multivector_doc_candidate_k,
                "multivector_exact_rerank": args.multivector_exact_rerank,
                "multivector_exact_rerank_k": args.multivector_exact_rerank_k,
                "serving_exact_rerank_k": args.serving_exact_rerank_k,
                "serving_exact_rerank_mode": args.serving_exact_rerank_mode,
                "admission_debug_mode": args.admission_debug_mode,
                "effective_admission_debug_mode": effective_admission_debug_mode(args),
                "serving_grid_admission_debug_mode": effective_admission_debug_mode(
                    clone_args(args, admission_debug_context="serving_grid")
                ),
                "multivector_graph": args.multivector_graph,
                "multivector_doc_build_scorer": args.multivector_doc_build_scorer,
                "allow_exact_symmetric_build": args.allow_exact_symmetric_build,
                "index_reloptions": index_phase.get("reloptions", []),
                "multivector_doc_graph_search_ef": args.multivector_doc_graph_search_ef,
                "multivector_doc_graph_oversampling": args.multivector_doc_graph_oversampling,
                "multivector_doc_graph_rescore_k": args.multivector_doc_graph_rescore_k,
                "multivector_doc_graph_entry_sample_count": args.multivector_doc_graph_entry_sample_count,
                "index_graph_m": args.index_graph_m,
                "index_graph_ef_construction": args.index_graph_ef_construction,
                "index_graph_ef_search": args.index_graph_ef_search,
                "index_native_segments": args.index_native_segments,
                "multivector_doc_storage": args.multivector_doc_storage,
                "multivector_doc_storage_cache": args.multivector_doc_storage_cache,
                "multivector_token_pooling": args.multivector_token_pooling,
                "multivector_token_pooling_target_ratio": args.multivector_token_pooling_target_ratio,
                "multivector_token_pooling_min_tokens": args.multivector_token_pooling_min_tokens,
                "multivector_centroids": args.multivector_centroids,
                "multivector_centroid_count": args.multivector_centroid_count,
                "multivector_proxy_encoder": args.multivector_proxy_encoder,
                "multivector_candidate_source": args.multivector_candidate_source,
                "multivector_plain_fallback": args.multivector_plain_fallback,
                "multivector_plain_fallback_max_docs": args.multivector_plain_fallback_max_docs,
                "multivector_plain_fallback_candidate_fraction": args.multivector_plain_fallback_candidate_fraction,
                "multivector_candidate_reservoirs": args.multivector_candidate_reservoirs,
                "multivector_per_token_doc_reservoir_k": args.multivector_per_token_doc_reservoir_k,
                "multivector_coverage_reservoir_k": args.multivector_coverage_reservoir_k,
                "multivector_bm25_candidate_injection": args.multivector_bm25_candidate_injection,
                "multivector_sparse_candidate_source": args.multivector_sparse_candidate_source,
                "learned_sparse_doc_jsonl": portable_path(args.learned_sparse_doc_jsonl) if args.learned_sparse_doc_jsonl else None,
                "learned_sparse_query_jsonl": portable_path(args.learned_sparse_query_jsonl) if args.learned_sparse_query_jsonl else None,
                "document_node_admission_grid": args.document_node_admission_grid,
                "document_node_serving_grid": args.document_node_serving_grid,
                "document_node_serving_grid_include_experimental": args.document_node_serving_grid_include_experimental,
                "document_node_serving_grid_include_proxy_encoders": args.document_node_serving_grid_include_proxy_encoders,
                "document_node_serving_grid_include_learned_projection": args.document_node_serving_grid_include_learned_projection,
                "document_node_serving_grid_include_bm25_rescue": args.document_node_serving_grid_include_bm25_rescue,
                "document_node_serving_grid_include_learned_sparse_rescue": args.document_node_serving_grid_include_learned_sparse_rescue,
                "document_node_serving_grid_include_reservoirs": args.document_node_serving_grid_include_reservoirs,
                "document_node_serving_grid_include_entry_sidecar": args.document_node_serving_grid_include_entry_sidecar,
                "document_node_serving_grid_include_entry_samples": args.document_node_serving_grid_include_entry_samples,
                "document_node_serving_grid_entry_sample_counts": args.document_node_serving_grid_entry_sample_count_values,
                "document_node_serving_grid_entry_sidecar_representatives": args.document_node_serving_grid_entry_sidecar_representative_values,
                "document_node_serving_grid_proxy_admission_focus": args.document_node_serving_grid_proxy_admission_focus,
                "document_node_serving_grid_centroid_lite_focus": args.document_node_serving_grid_centroid_lite_focus,
                "document_node_serving_grid_token_pooling_focus": args.document_node_serving_grid_token_pooling_focus,
                "document_node_serving_grid_stage_mode": args.document_node_serving_grid_stage_mode,
                "serving_grid_probe_queries": args.serving_grid_probe_queries,
                "serving_grid_finalists": args.serving_grid_finalists,
                "document_node_serving_latency_only": args.document_node_serving_latency_only,
                "document_node_serving_build_only": args.document_node_serving_build_only,
                "serving_profile_name": args.serving_profile_name,
                "serving_ef": args.serving_ef,
                "serving_oversampling": args.serving_oversampling,
                "serving_candidate_k": args.serving_candidate_k,
                "serving_storage": args.serving_storage,
                "serving_cache": args.serving_cache,
                "serving_fail_on_slow_path": args.serving_fail_on_slow_path,
                "serving_min_top10_admission": args.serving_min_top10_admission,
                "serving_min_ndcg_ratio_vs_exact": args.serving_min_ndcg_ratio_vs_exact,
                "serving_max_p95_ms": args.serving_max_p95_ms,
                "document_node_storage_grid": args.document_node_storage_grid_values,
                "document_node_cache_grid": args.document_node_cache_grid_values,
                "document_node_pooling_grid": args.document_node_pooling_grid_values,
                "document_node_proxy_encoder_grid": args.document_node_proxy_encoder_grid_values,
                "document_node_ef_grid": args.document_node_ef_grid_values,
                "document_node_oversampling_grid": args.document_node_oversampling_grid_values,
                "hybrid_evaluation_harness": args.hybrid_evaluation_harness,
                "hybrid_evaluation_modes": args.hybrid_evaluation_mode_values,
                "hybrid_evaluation_quality_floor": args.hybrid_evaluation_quality_floor,
                "hybrid_evaluation_dbsf_min_branch_candidates": args.hybrid_evaluation_dbsf_min_branch_candidates,
                "token_ablation_query_id": args.token_ablation_query_id,
                "token_ablation_skip_tokens": args.token_ablation_skip_tokens,
                "token_ablation_final_k": args.token_ablation_final_k,
                "token_ablation_dense_k": args.token_ablation_dense_k,
                "final_k": args.final_k,
                "quality_k": args.quality_k,
                "clients": args.clients,
                "parallel_safety_preflight": args.parallel_safety_preflight,
                "parallel_safety_max_serial_p95_ms": args.parallel_safety_max_serial_p95_ms,
                "parallel_safety_max_docs_scored": args.parallel_safety_max_docs_scored,
                "parallel_safety_max_exact_rerank_docs": args.parallel_safety_max_exact_rerank_docs,
                "parallel_safety_max_exact_pairs": args.parallel_safety_max_exact_pairs,
                "parallel_safety_max_sidecar_bytes": args.parallel_safety_max_sidecar_bytes,
                "force_parallel_retrieval": args.force_parallel_retrieval,
                "skip_parallel_retrieval": args.skip_parallel_retrieval,
                "generation_clients": args.generation_clients,
                "generation_threads": args.generation_threads,
                "generation_n_gpu_layers": args.generation_n_gpu_layers,
                "generation_batch_sequences": args.generation_batch_sequences,
                "generation_n_batch": args.generation_n_batch,
                "generation_warmup": args.generation_warmup,
                "warm_queries": args.warm_queries,
                "timed_queries": args.timed_queries,
            },
            "phases": {
                "load_text": load_phase,
                "generation_in_postgres": generation_phase,
                "insert_generated_multivectors": insert_phase,
                "load_learned_sparse_vectors": learned_sparse_phase,
                "build_index": index_phase,
            },
            "results": result_methods,
        }
        if admission_debug is not None:
            output["admission_debug"] = admission_debug
        if document_node_admission_grid is not None:
            output["document_node_admission_grid"] = document_node_admission_grid
        if document_node_serving_grid is not None:
            output["document_node_serving_grid"] = document_node_serving_grid
        if document_node_serving_recommendation is not None:
            output["document_node_serving_recommendation"] = document_node_serving_recommendation
        if document_node_token_pooling_recommendation is not None:
            output["document_node_token_pooling_recommendation"] = (
                document_node_token_pooling_recommendation
            )
        if document_node_serving_latency_only is not None:
            output["document_node_serving_latency_only"] = document_node_serving_latency_only
        if document_node_serving_build_only is not None:
            output["document_node_serving_build_only"] = document_node_serving_build_only
        if hybrid_evaluation is not None:
            output["hybrid_evaluation"] = hybrid_evaluation
        if token_ablation is not None:
            output["token_ablation"] = token_ablation
        output["markdown_summary"] = markdown_benchmark_summary(output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.markdown_output is not None:
            args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
            args.markdown_output.write_text(output["markdown_summary"], encoding="utf-8")
        print(args.output)
        if args.markdown_output is not None:
            print(args.markdown_output)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
