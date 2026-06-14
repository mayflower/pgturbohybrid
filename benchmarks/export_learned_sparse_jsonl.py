#!/usr/bin/env python3
"""Export model-agnostic learned-sparse JSONL fixtures.

The script intentionally does not train or run a specific sparse model. It
provides two adapters:

hash_plumbing
    Deterministic stdlib-only toy features for pipeline tests. Exports are
    always marked ``plumbing_only=true`` and must not be used as serving
    evidence.

external_command
    Calls a user-supplied command once for docs and once for queries. The
    command receives JSONL on stdin with rows:

        {"kind": "doc", "id": "...", "text": "..."}
        {"kind": "query", "id": "...", "text": "..."}

    It must write JSONL on stdout with matching IDs and sparse arrays:

        {"id": "...", "term_ids": [1, 2], "weights": [0.5, 1.2]}

    ``doc_id``/``query_id`` are also accepted in output. The command is invoked
    without a shell using ``shlex.split``; pass a quoted command string if it
    contains arguments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

try:
    import psycopg
except ImportError:  # pragma: no cover - --help and jsonl mode do not need psycopg
    psycopg = None  # type: ignore[assignment]


TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
HASH_MODULUS = 1_000_000


@dataclass(frozen=True)
class TextRow:
    row_id: str
    text: str


@dataclass(frozen=True)
class SparseRow:
    row_id: str
    term_ids: list[int]
    weights: list[float]


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"expected true or false, got {value!r}")


def connect(database: str):
    if psycopg is None:
        raise SystemExit("psycopg is required for --input-kind dbpedia_postgres")
    if "=" in database or "://" in database:
        return psycopg.connect(database)
    return psycopg.connect(dbname=database)


def fetch_dbpedia_rows(
    database: str,
    *,
    max_docs: int,
    max_queries: int,
) -> tuple[list[TextRow], list[TextRow]]:
    with connect(database) as conn:
        with conn.cursor() as cur:
            doc_sql = """
                SELECT doc_id, concat_ws(E'\n', title, body) AS text
                FROM dbpedia_colbert_docs
                ORDER BY doc_id
            """
            doc_params: tuple[Any, ...] = ()
            if max_docs > 0:
                doc_sql += " LIMIT %s"
                doc_params = (max_docs,)
            cur.execute(doc_sql, doc_params)
            docs = [TextRow(str(row[0]), str(row[1] or "")) for row in cur.fetchall()]

            query_sql = """
                SELECT query_id, query_text
                FROM dbpedia_colbert_queries
                ORDER BY query_id
            """
            query_params: tuple[Any, ...] = ()
            if max_queries > 0:
                query_sql += " LIMIT %s"
                query_params = (max_queries,)
            cur.execute(query_sql, query_params)
            queries = [TextRow(str(row[0]), str(row[1] or "")) for row in cur.fetchall()]
    return docs, queries


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON") from exc
            if not isinstance(item, dict):
                raise SystemExit(f"{path}:{line_no}: expected JSON object")
            yield item


def jsonl_text_rows(path: Path, *, id_key: str, max_rows: int) -> list[TextRow]:
    rows: list[TextRow] = []
    for item in read_jsonl(path):
        if str(item.get("kind") or "").lower() == "metadata":
            continue
        row_id = item.get(id_key, item.get("id"))
        if row_id is None:
            raise SystemExit(f"{path}: expected {id_key!r} or 'id' field")
        text = item.get("text")
        if text is None and id_key == "doc_id":
            text = " ".join(
                str(item.get(key) or "")
                for key in ("title", "body")
                if item.get(key) is not None
            )
        if text is None and id_key == "query_id":
            text = item.get("query_text")
        if text is None:
            raise SystemExit(f"{path}: expected text field for {row_id!r}")
        rows.append(TextRow(str(row_id), str(text)))
        if max_rows > 0 and len(rows) >= max_rows:
            break
    return rows


def load_input_rows(args: argparse.Namespace) -> tuple[list[TextRow], list[TextRow]]:
    if args.input_kind == "dbpedia_postgres":
        if not args.database:
            raise SystemExit("--database is required with --input-kind dbpedia_postgres")
        return fetch_dbpedia_rows(
            args.database,
            max_docs=args.max_docs,
            max_queries=args.max_queries,
        )
    if args.doc_input is None or args.query_input is None:
        raise SystemExit("--input-kind jsonl requires --doc-input and --query-input")
    docs = jsonl_text_rows(args.doc_input, id_key="doc_id", max_rows=args.max_docs)
    queries = jsonl_text_rows(args.query_input, id_key="query_id", max_rows=args.max_queries)
    return docs, queries


def hash_term(token: str) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % HASH_MODULUS + 1


def hash_plumbing_features(text: str) -> tuple[list[int], list[float]]:
    counts: Counter[int] = Counter()
    for token in TOKEN_RE.findall(text.lower()):
        counts[hash_term(token)] += 1
    if not counts:
        return [], []
    term_ids = sorted(counts)
    weights = [round(1.0 + math.log(float(counts[term_id])), 6) for term_id in term_ids]
    return term_ids, weights


def export_hash_plumbing(rows: Sequence[TextRow]) -> list[SparseRow]:
    sparse_rows: list[SparseRow] = []
    for row in rows:
        term_ids, weights = hash_plumbing_features(row.text)
        sparse_rows.append(SparseRow(row.row_id, term_ids, weights))
    return sparse_rows


def parse_external_sparse_rows(
    output: str,
    *,
    preferred_id_key: str,
    expected_ids: set[str],
) -> list[SparseRow]:
    sparse_rows: list[SparseRow] = []
    seen: set[str] = set()
    for line_no, raw_line in enumerate(output.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SystemExit(f"external_command stdout:{line_no}: invalid JSON") from exc
        if not isinstance(item, dict):
            raise SystemExit(f"external_command stdout:{line_no}: expected JSON object")
        row_id = item.get(preferred_id_key, item.get("id"))
        term_ids = item.get("term_ids", item.get("terms", item.get("indices")))
        weights = item.get("weights", item.get("scores", item.get("values")))
        if row_id is None or term_ids is None or weights is None:
            raise SystemExit(
                f"external_command stdout:{line_no}: expected id, term_ids, and weights"
            )
        row_id = str(row_id)
        if row_id in seen:
            raise SystemExit(f"external_command stdout:{line_no}: duplicate id {row_id!r}")
        seen.add(row_id)
        if row_id not in expected_ids:
            raise SystemExit(f"external_command stdout:{line_no}: unexpected id {row_id!r}")
        if not isinstance(term_ids, list) or not isinstance(weights, list):
            raise SystemExit(f"external_command stdout:{line_no}: term_ids and weights must be arrays")
        if len(term_ids) != len(weights):
            raise SystemExit(f"external_command stdout:{line_no}: term_ids and weights length mismatch")
        parsed_terms: list[int] = []
        parsed_weights: list[float] = []
        for term, weight in zip(term_ids, weights):
            term_id = int(term)
            sparse_weight = float(weight)
            if term_id < 0 or not math.isfinite(sparse_weight):
                raise SystemExit(
                    f"external_command stdout:{line_no}: term_ids must be non-negative and weights finite"
                )
            parsed_terms.append(term_id)
            parsed_weights.append(sparse_weight)
        sparse_rows.append(SparseRow(row_id, parsed_terms, parsed_weights))
    missing = expected_ids - seen
    if missing:
        sample = ", ".join(sorted(missing)[:10])
        raise SystemExit(f"external_command did not emit features for {len(missing)} row(s): {sample}")
    return sparse_rows


def run_external_command(
    command: str,
    rows: Sequence[TextRow],
    *,
    kind: str,
    preferred_id_key: str,
) -> list[SparseRow]:
    if not command:
        raise SystemExit("--external-command is required with --adapter external_command")
    stdin = "\n".join(
        json.dumps({"kind": kind, "id": row.row_id, "text": row.text}, sort_keys=True)
        for row in rows
    )
    if stdin:
        stdin += "\n"
    env = os.environ.copy()
    env["LEARNED_SPARSE_KIND"] = kind
    proc = subprocess.run(
        shlex.split(command),
        input=stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"external_command failed for {kind} rows with exit {proc.returncode}: "
            f"{proc.stderr.strip()}"
        )
    return parse_external_sparse_rows(
        proc.stdout,
        preferred_id_key=preferred_id_key,
        expected_ids={row.row_id for row in rows},
    )


def export_sparse_rows(
    args: argparse.Namespace,
    docs: Sequence[TextRow],
    queries: Sequence[TextRow],
) -> tuple[list[SparseRow], list[SparseRow]]:
    if args.adapter == "hash_plumbing":
        return export_hash_plumbing(docs), export_hash_plumbing(queries)
    if args.adapter == "external_command":
        return (
            run_external_command(
                args.external_command,
                docs,
                kind="doc",
                preferred_id_key="doc_id",
            ),
            run_external_command(
                args.external_command,
                queries,
                kind="query",
                preferred_id_key="query_id",
            ),
        )
    raise SystemExit(f"unsupported adapter: {args.adapter}")


def sparse_row_json(row: SparseRow, *, id_key: str) -> dict[str, Any]:
    return {
        id_key: row.row_id,
        "term_ids": row.term_ids,
        "weights": row.weights,
    }


def write_sparse_jsonl(path: Path, rows: Sequence[SparseRow], *, id_key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(sparse_row_json(row, id_key=id_key), sort_keys=True) + "\n")


def external_command_has_safe_provenance(args: argparse.Namespace) -> bool:
    return bool(args.feature_source and args.feature_version)


def effective_plumbing_only(args: argparse.Namespace) -> bool:
    if args.adapter == "hash_plumbing":
        return True
    if args.plumbing_only:
        return True
    return not external_command_has_safe_provenance(args)


def manifest_for_output(
    args: argparse.Namespace,
    *,
    output_kind: str,
    doc_count: int,
    query_count: int,
) -> dict[str, Any]:
    plumbing_only = effective_plumbing_only(args)
    warnings: list[str] = []
    if args.adapter == "hash_plumbing":
        warnings.append("hash_plumbing_features_are_not_safe_serving_evidence")
    if args.adapter == "external_command" and plumbing_only and not args.plumbing_only:
        warnings.append("external_command_missing_feature_provenance")
    return {
        "kind": "metadata",
        "output_kind": output_kind,
        "feature_source": args.feature_source or (
            "hash_plumbing" if args.adapter == "hash_plumbing" else "external_command"
        ),
        "feature_version": args.feature_version or "",
        "model_name": args.model_name or "",
        "model_checksum": args.model_checksum or "",
        "adapter": args.adapter,
        "plumbing_only": plumbing_only,
        "safe_serving_evidence": not plumbing_only,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "doc_count": doc_count,
        "query_count": query_count,
        "coverage_target": {
            "input_kind": args.input_kind,
            "max_docs": args.max_docs,
            "max_queries": args.max_queries,
        },
        "warnings": warnings,
    }


def manifest_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.name + ".manifest.json")


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest_path(path).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def export(args: argparse.Namespace) -> dict[str, Any]:
    docs, queries = load_input_rows(args)
    doc_sparse, query_sparse = export_sparse_rows(args, docs, queries)
    write_sparse_jsonl(args.doc_output, doc_sparse, id_key="doc_id")
    write_sparse_jsonl(args.query_output, query_sparse, id_key="query_id")
    doc_manifest = manifest_for_output(
        args,
        output_kind="doc",
        doc_count=len(doc_sparse),
        query_count=len(query_sparse),
    )
    query_manifest = manifest_for_output(
        args,
        output_kind="query",
        doc_count=len(doc_sparse),
        query_count=len(query_sparse),
    )
    write_manifest(args.doc_output, doc_manifest)
    write_manifest(args.query_output, query_manifest)
    return {
        "doc_output": str(args.doc_output),
        "query_output": str(args.query_output),
        "doc_manifest": str(manifest_path(args.doc_output)),
        "query_manifest": str(manifest_path(args.query_output)),
        "doc_count": len(doc_sparse),
        "query_count": len(query_sparse),
        "adapter": args.adapter,
        "plumbing_only": effective_plumbing_only(args),
        "safe_serving_evidence": not effective_plumbing_only(args),
    }


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.adapter == "external_command" and not args.external_command:
        raise SystemExit("--external-command is required with --adapter external_command")
    if args.adapter == "hash_plumbing" and args.plumbing_only is False:
        args.plumbing_only = True
    if args.input_kind == "jsonl":
        if args.doc_input is None or args.query_input is None:
            raise SystemExit("--input-kind jsonl requires --doc-input and --query-input")
        args.doc_input = args.doc_input.resolve()
        args.query_input = args.query_input.resolve()
        if not args.doc_input.is_file():
            raise SystemExit(f"document input JSONL does not exist: {args.doc_input}")
        if not args.query_input.is_file():
            raise SystemExit(f"query input JSONL does not exist: {args.query_input}")
    elif args.input_kind == "dbpedia_postgres" and not args.database:
        raise SystemExit("--database is required with --input-kind dbpedia_postgres")
    args.doc_output = args.doc_output.resolve()
    args.query_output = args.query_output.resolve()
    if args.max_docs < 0:
        raise SystemExit("--max-docs must be non-negative")
    if args.max_queries < 0:
        raise SystemExit("--max-queries must be non-negative")
    return args


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true", help="run cheap stdlib self-checks")
    parser.add_argument(
        "--input-kind",
        choices=("dbpedia_postgres", "jsonl"),
        default="dbpedia_postgres",
        help="source of document/query text",
    )
    parser.add_argument("--database", default=os.environ.get("PGDATABASE", "pgturbohybrid_dbpedia_colbert"))
    parser.add_argument("--doc-input", type=Path, default=None, help="document text JSONL for --input-kind jsonl")
    parser.add_argument("--query-input", type=Path, default=None, help="query text JSONL for --input-kind jsonl")
    parser.add_argument("--doc-output", type=Path, default=Path("learned-sparse-docs.jsonl"))
    parser.add_argument("--query-output", type=Path, default=Path("learned-sparse-queries.jsonl"))
    parser.add_argument("--max-docs", type=int, default=0, help="maximum docs to export; 0 means all")
    parser.add_argument("--max-queries", type=int, default=0, help="maximum queries to export; 0 means all")
    parser.add_argument(
        "--adapter",
        choices=("hash_plumbing", "external_command"),
        default="hash_plumbing",
        help="feature exporter adapter",
    )
    parser.add_argument(
        "--external-command",
        default="",
        help="command implementing the documented JSONL stdin/stdout protocol",
    )
    parser.add_argument("--feature-source", default="", help="feature generator source label")
    parser.add_argument("--feature-version", default="", help="feature generator version")
    parser.add_argument("--model-name", default="", help="optional model name recorded in manifest")
    parser.add_argument("--model-checksum", default="", help="optional model checksum recorded in manifest")
    parser.add_argument(
        "--plumbing-only",
        type=parse_bool,
        default=False,
        metavar="true|false",
        help="mark exports as plumbing-only; hash_plumbing always forces true",
    )
    args = parser.parse_args(argv)
    if args.self_check:
        return args
    return validate_args(args)


def import_validator_helpers():
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    from dbpedia_colbert_multivector import (  # pylint: disable=import-outside-toplevel
        learned_sparse_jsonl_validation_report_from_expected,
    )

    return learned_sparse_jsonl_validation_report_from_expected


def self_check() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        docs_in = tmp / "docs.jsonl"
        queries_in = tmp / "queries.jsonl"
        docs_in.write_text(
            "\n".join(
                [
                    json.dumps({"doc_id": "d1", "title": "Alpha", "body": "beta beta"}),
                    json.dumps({"doc_id": "d2", "text": "Gamma delta"}),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        queries_in.write_text(
            json.dumps({"query_id": "q1", "query_text": "alpha delta"}) + "\n",
            encoding="utf-8",
        )
        docs_out = tmp / "hash-docs.jsonl"
        queries_out = tmp / "hash-queries.jsonl"
        args = validate_args(
            parse_args(
                [
                    "--input-kind",
                    "jsonl",
                    "--doc-input",
                    str(docs_in),
                    "--query-input",
                    str(queries_in),
                    "--doc-output",
                    str(docs_out),
                    "--query-output",
                    str(queries_out),
                    "--adapter",
                    "hash_plumbing",
                    "--plumbing-only",
                    "false",
                ]
            )
        )
        first = export(args)
        first_docs = docs_out.read_text(encoding="utf-8")
        second = export(args)
        assert first == second
        assert first_docs == docs_out.read_text(encoding="utf-8")
        doc_manifest = json.loads(manifest_path(docs_out).read_text(encoding="utf-8"))
        assert doc_manifest["adapter"] == "hash_plumbing"
        assert doc_manifest["plumbing_only"] is True
        assert doc_manifest["safe_serving_evidence"] is False

        validation_report_from_expected = import_validator_helpers()
        validation = validation_report_from_expected(
            argparse.Namespace(
                learned_sparse_doc_jsonl=docs_out,
                learned_sparse_query_jsonl=queries_out,
                learned_sparse_require_full_coverage=True,
                database="self_check",
            ),
            ["d1", "d2"],
            ["q1"],
        )["learned_sparse_jsonl_validation"]
        assert validation["doc_coverage"] == 1.0
        assert validation["query_coverage"] == 1.0
        assert validation["feature_source"] == "hash_plumbing"
        assert validation["plumbing_only"] is True
        assert validation["safe_serving_evidence"] is False

        fake = tmp / "fake_external.py"
        fake.write_text(
            "\n".join(
                [
                    "import json, sys",
                    "for line in sys.stdin:",
                    "    item = json.loads(line)",
                    "    print(json.dumps({'id': item['id'], 'term_ids': [7], 'weights': [1.5]}))",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        ext_docs = tmp / "external-docs.jsonl"
        ext_queries = tmp / "external-queries.jsonl"
        ext_args = validate_args(
            parse_args(
                [
                    "--input-kind",
                    "jsonl",
                    "--doc-input",
                    str(docs_in),
                    "--query-input",
                    str(queries_in),
                    "--doc-output",
                    str(ext_docs),
                    "--query-output",
                    str(ext_queries),
                    "--adapter",
                    "external_command",
                    "--external-command",
                    f"{shlex.quote(sys.executable)} {shlex.quote(str(fake))}",
                    "--feature-source",
                    "fixture_sparse",
                    "--feature-version",
                    "v1",
                    "--model-name",
                    "fixture-model",
                    "--model-checksum",
                    "sha256:fixture",
                    "--plumbing-only",
                    "false",
                ]
            )
        )
        ext_summary = export(ext_args)
        assert ext_summary["safe_serving_evidence"] is True
        ext_manifest = json.loads(manifest_path(ext_docs).read_text(encoding="utf-8"))
        assert ext_manifest["adapter"] == "external_command"
        assert ext_manifest["plumbing_only"] is False
        assert json.loads(ext_docs.read_text(encoding="utf-8").splitlines()[0])["term_ids"] == [7]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.self_check:
        self_check()
        print("self-check ok")
        return
    summary = export(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
