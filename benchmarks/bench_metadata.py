"""Shared benchmark provenance metadata (prompt 11).

A benchmark number is only trustworthy if a reader can tell *what* produced it.
This module collects the provenance every pgturbohybrid benchmark claim should
carry:

  - git commit, dirty-tree status, branch
  - host OS, architecture, CPU model, logical CPU count, Python version
  - PostgreSQL version, pgvector version/ref, pgturbohybrid extension version
  - the dataset (name/version), row count, vector dimensions, query count
  - warm-up passes, measured passes, cache state (cold vs warm)
  - the index reloptions and any non-default turbohybrid.* GUCs

Python drivers call ``collect(...)`` and embed the result. SQL/shell benchmarks
run this module as a script (``python3 benchmarks/bench_metadata.py``) and
capture the JSON it prints; the database fields are filled in when a psql
connection is reachable. Everything degrades gracefully: a missing git binary,
unreachable database, or absent /proc just leaves a field ``None`` rather than
failing the benchmark.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
from typing import Any, Callable, Optional

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# turbohybrid.* GUCs that are non-default get recorded; this query returns a
# JSON object of name -> setting and an empty object when everything is default.
_GUC_SQL = (
    "SELECT COALESCE(json_object_agg(name, setting), '{}'::json) "
    "FROM pg_settings WHERE name LIKE 'turbohybrid.%' AND source <> 'default';"
)


def _run(cmd: list[str]) -> Optional[str]:
    try:
        out = subprocess.run(cmd, cwd=_REPO, text=True, capture_output=True, timeout=30)
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except Exception:
        return None


def git_provenance() -> dict[str, Any]:
    commit = _run(["git", "rev-parse", "HEAD"])
    branch = _run(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    dirty: Optional[bool] = None
    if status is not None:
        dirty = bool(status.strip())
    return {"commit": commit, "dirty": dirty, "branch": branch}


def _cpu_model() -> Optional[str]:
    system = platform.system()
    if system == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    elif system == "Darwin":
        model = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if model:
            return model
    return platform.processor() or None


def host_provenance() -> dict[str, Any]:
    return {
        "os": platform.platform(),
        "arch": platform.machine(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }


def database_provenance(query: Callable[[str], Optional[str]]) -> dict[str, Any]:
    """Collect DB-side provenance.

    ``query`` is any callable that runs a single SQL statement and returns the
    first column of the first row as text (e.g. a wrapper around psql -tAc or a
    psycopg cursor). Failures are swallowed into None.
    """

    def scalar(sql: str) -> Optional[str]:
        try:
            value = query(sql)
        except Exception:
            return None
        return value.strip() if isinstance(value, str) else value

    gucs_raw = scalar(_GUC_SQL)
    gucs: Any = None
    if gucs_raw:
        try:
            gucs = json.loads(gucs_raw)
        except (ValueError, TypeError):
            gucs = gucs_raw

    return {
        "postgres_version": scalar("SHOW server_version;"),
        "pgvector_version": scalar("SELECT extversion FROM pg_extension WHERE extname = 'vector';"),
        "pgturbohybrid_version": scalar(
            "SELECT COALESCE((SELECT extversion FROM pg_extension WHERE extname = 'pgturbohybrid'), 'not-installed');"
        ),
        "pgvector_ref": os.environ.get("PGVECTOR_REF") or None,
        "turbohybrid_non_default_gucs": gucs,
    }


def collect(
    query: Optional[Callable[[str], Optional[str]]] = None,
    *,
    suite: Optional[str] = None,
    dataset: Optional[str] = None,
    dataset_version: Optional[str] = None,
    rows: Optional[int] = None,
    dimensions: Optional[int] = None,
    query_count: Optional[int] = None,
    warmup_passes: Optional[int] = None,
    measured_passes: Optional[int] = None,
    cache_state: Optional[str] = None,
    index_reloptions: Any = None,
    extra: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Return a complete provenance block for a benchmark result.

    Caller-supplied dataset/run fields are recorded as given (``None`` when the
    driver does not know them); git, host, and database fields are gathered
    automatically.
    """
    provenance: dict[str, Any] = {
        "suite": suite,
        "git": git_provenance(),
        "host": host_provenance(),
        "dataset": {
            "name": dataset,
            "version": dataset_version,
            "rows": rows,
            "dimensions": dimensions,
            "query_count": query_count,
        },
        "run": {
            "warmup_passes": warmup_passes,
            "measured_passes": measured_passes,
            "cache_state": cache_state,
        },
        "index_reloptions": index_reloptions,
    }
    if query is not None:
        provenance["database"] = database_provenance(query)
    if extra:
        provenance.update(extra)
    return provenance


def _psql_query(database: str, psql: str = "psql") -> Callable[[str], Optional[str]]:
    def run(sql: str) -> Optional[str]:
        return _run([psql, "-d", database, "-tAqc", sql])

    return run


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", default=os.environ.get("PGDATABASE", ""),
                        help="database to query for DB-side provenance (optional)")
    parser.add_argument("--psql", default=os.environ.get("PSQL", "psql"))
    parser.add_argument("--suite", default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--dimensions", type=int, default=None)
    parser.add_argument("--query-count", type=int, default=None)
    parser.add_argument("--warmup-passes", type=int, default=None)
    parser.add_argument("--measured-passes", type=int, default=None)
    parser.add_argument("--cache-state", default=None)
    args = parser.parse_args()

    query = _psql_query(args.database, args.psql) if args.database else None
    md = collect(
        query=query,
        suite=args.suite,
        dataset=args.dataset,
        rows=args.rows,
        dimensions=args.dimensions,
        query_count=args.query_count,
        warmup_passes=args.warmup_passes,
        measured_passes=args.measured_passes,
        cache_state=args.cache_state,
    )
    print(json.dumps(md, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
