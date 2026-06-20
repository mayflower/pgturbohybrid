"""Pure reporting helpers for DBpedia ColBERT benchmark artifacts."""

from __future__ import annotations

import math
from typing import Any


def serving_timing_number(value: Any, *, allow_zero: bool = True) -> int | None:
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
    return serving_timing_number(timing.get(key))


def _flat_graph_timing_value(stats: dict[str, Any], key: str) -> int | None:
    return serving_timing_number(stats.get(key))


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
        "payload_entry_seed_time_us": (
            "payload_entry_seed",
            "graph_payload_entry_seed_us",
        ),
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
            serving_timing_number(stats.get(direct_key))
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

    heap_rescore = serving_timing_number(
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
        used_nested = (
            used_nested or _nested_dense_timing_value(stats, "heap_rescore") is not None
        )
        used_flat = (
            used_flat or _flat_graph_timing_value(stats, "graph_heap_rescore_us") is not None
        )

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
            value = serving_timing_number(stats.get(key))
            if value is not None:
                used_direct = True
                break
        values[canonical] = value

    total_value = values.get("phase_total_time_us")
    if total_value in (None, 0):
        dense = stats.get("dense", {})
        dense_elapsed = (
            serving_timing_number(dense.get("elapsed_us"), allow_zero=False)
            if isinstance(dense, dict)
            else None
        )
        total_value = (
            dense_elapsed
            or serving_timing_number(stats.get("dense_elapsed_us"), allow_zero=False)
            or serving_timing_number(stats.get("elapsed_us"), allow_zero=False)
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

    return {
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
