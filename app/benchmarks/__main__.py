"""Command-line entry point for the GEO benchmark runner.

Examples::

    python -m app.benchmarks catalog
    python -m app.benchmarks fixture --benchmark egeo --directory .\\fixtures
    python -m app.benchmarks evaluate --benchmark egeo --cases cases.json \
        --predictions outputs.jsonl --baseline CONTROL --output report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from .io import BenchmarkIOError
from .registry import get_benchmark_spec, list_benchmarks
from .runner import (
    BenchmarkRunner,
    BenchmarkValidationError,
    write_markdown_report,
    write_report,
    write_synthetic_fixture,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.benchmarks",
        description=(
            "Evaluate authorised GEO benchmark exports. The runner does not download datasets or call external engines."
        ),
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    catalog = subcommands.add_parser("catalog", help="List supported benchmark and framework profiles.")
    catalog.add_argument("--json", action="store_true", help="Emit full machine-readable registry metadata.")

    fixture = subcommands.add_parser("fixture", help="Write a tiny synthetic smoke-test fixture (not research data).")
    fixture.add_argument("--benchmark", required=True, help="Registry id, display name, or alias.")
    fixture.add_argument("--directory", required=True, type=Path, help="Directory in which to write fixture files.")
    fixture.add_argument("--overwrite", action="store_true", help="Allow replacement of existing fixture files.")

    validate = subcommands.add_parser("validate", help="Validate inputs without calculating results.")
    _input_arguments(validate, predictions_optional=True)

    evaluate = subcommands.add_parser("evaluate", help="Evaluate a benchmark and write a JSON report.")
    _input_arguments(evaluate, predictions_optional=False)
    evaluate.add_argument("--baseline", help="Control/baseline method label for paired comparisons.")
    evaluate.add_argument("--output", required=True, type=Path, help="Destination JSON report path.")
    evaluate.add_argument("--markdown", type=Path, help="Optional companion Markdown report path.")
    evaluate.add_argument("--overwrite", action="store_true", help="Allow replacement of existing report files.")
    evaluate.add_argument("--strict", action="store_true", help="Fail if input validation finds an error.")
    evaluate.add_argument(
        "--run-metadata-json",
        help="Optional JSON object stored verbatim in the report receipt (for model, provider, seed, locale, etc.).",
    )
    return parser


def _input_arguments(parser: argparse.ArgumentParser, *, predictions_optional: bool) -> None:
    parser.add_argument("--benchmark", required=True, help="Registry id, display name, or alias.")
    parser.add_argument("--cases", required=True, type=Path, help="Cases manifest / JSONL / CSV export.")
    parser.add_argument(
        "--predictions",
        required=not predictions_optional,
        type=Path,
        help="Prediction JSON / JSONL / CSV export.",
    )
    parser.add_argument("--targets", type=Path, help="Optional target/relevance-label export to merge with cases.")
    parser.add_argument("--traces", type=Path, help="Optional trace-event JSON / JSONL / CSV export.")


def _load_metadata(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    decoded = json.loads(value)
    if not isinstance(decoded, dict):
        raise ValueError("--run-metadata-json must be a JSON object.")
    return decoded


def _print_catalog(as_json: bool) -> int:
    specs = list_benchmarks()
    if as_json:
        print(json.dumps([spec.as_dict() for spec in specs], ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    print("Supported benchmark and framework profiles:\n")
    for spec in specs:
        print(f"- {spec.benchmark_id}: {spec.display_name} — {spec.primary_task}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "catalog":
            return _print_catalog(args.json)
        if args.command == "fixture":
            spec = get_benchmark_spec(args.benchmark)
            paths = write_synthetic_fixture(args.directory, benchmark=spec.benchmark_id, overwrite=args.overwrite)
            print(json.dumps({key: str(path) for key, path in paths.items()}, ensure_ascii=False, indent=2, sort_keys=True))
            return 0

        runner = BenchmarkRunner()
        if args.command == "validate":
            report = runner.validate(
                args.cases,
                benchmark=args.benchmark,
                predictions_path=args.predictions,
                targets_path=args.targets,
                traces_path=args.traces,
            )
            print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
            return 0 if report["valid"] else 2

        if args.command == "evaluate":
            report = runner.evaluate(
                args.cases,
                args.predictions,
                benchmark=args.benchmark,
                baseline_method=args.baseline,
                targets_path=args.targets,
                traces_path=args.traces,
                strict=args.strict,
                run_metadata=_load_metadata(args.run_metadata_json),
            )
            write_report(args.output, report, overwrite=args.overwrite)
            if args.markdown:
                write_markdown_report(args.markdown, report, overwrite=args.overwrite)
            print(json.dumps({"report": str(args.output), "markdown": str(args.markdown) if args.markdown else None}, ensure_ascii=False))
            return 0
    except (BenchmarkIOError, BenchmarkValidationError, FileExistsError, KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
