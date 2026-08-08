"""Reproducible file-based runner for GEO benchmark evaluation.

The runner accepts authorised JSON, JSONL, or CSV exports and writes only
derived reports.  It deliberately does *not* download benchmark datasets, call
commercial answer engines, scrape marketplaces, or generate optimisation /
manipulation prompts.  Those actions need separate permissions and versioned
collection protocols; this module evaluates the artefacts researchers supply.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .evaluators import BenchmarkEvaluator
from .io import (
    BenchmarkIOError,
    merge_target_selections,
    read_benchmark_dataset,
    read_predictions,
    read_target_selections,
    read_trace_events,
    write_benchmark_dataset,
    write_predictions,
    write_trace_events,
)
from .models import BenchmarkCase, BenchmarkDataset, Candidate, CitationSpan, Prediction, TraceEvent
from .registry import BenchmarkSpec, get_benchmark_spec, list_benchmarks


class BenchmarkValidationError(ValueError):
    """Raised when ``strict`` execution encounters malformed study inputs."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    return str(value)


def _method_label(prediction: Prediction) -> str:
    metadata = dict(prediction.metadata)
    return str(
        prediction.condition
        or metadata.get("method")
        or metadata.get("method_id")
        or metadata.get("system")
        or prediction.run_id
        or "candidate"
    )


def _event_payload(event: TraceEvent) -> dict[str, Any]:
    return event.to_dict()


def attach_trace_events(
    predictions: Iterable[Prediction], traces: Iterable[TraceEvent]
) -> list[Prediction]:
    """Attach protocol traces to matching immutable prediction records.

    The canonical prediction model keeps a ``trace_id`` rather than embedding
    every event.  Evaluators, however, need a single record view.  This helper
    preserves the original prediction and adds the supplied trace payload to
    its metadata.  It first matches ``(case_id, run_id)`` and then falls back
    to a case-level trace when an export does not carry run identifiers.
    """

    exact: dict[tuple[str, str], list[TraceEvent]] = defaultdict(list)
    by_case: dict[str, list[TraceEvent]] = defaultdict(list)
    for event in traces:
        exact[(event.case_id, event.run_id)].append(event)
        by_case[event.case_id].append(event)

    attached: list[Prediction] = []
    for prediction in predictions:
        events = exact.get((prediction.case_id, prediction.run_id))
        # A case-level fallback is safe only for exports that genuinely omit
        # run identifiers.  Falling back for a named run could accidentally
        # copy a treatment trajectory onto its control counterpart.
        if events is None and prediction.run_id == "default":
            events = [event for event in by_case.get(prediction.case_id, []) if event.run_id == "default"]
        if events is None:
            events = []
        if not events:
            attached.append(prediction)
            continue
        metadata = dict(prediction.metadata)
        metadata["trace"] = [_event_payload(event) for event in events]
        attached.append(replace(prediction, metadata=metadata))
    return attached


def validate_inputs(
    cases: Iterable[BenchmarkCase],
    predictions: Iterable[Prediction] = (),
    *,
    benchmark: str | None = None,
) -> dict[str, Any]:
    """Return transparent data-quality diagnostics without changing inputs."""

    case_rows = list(cases)
    prediction_rows = list(predictions)
    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    case_ids: set[str] = set()
    target_count = 0
    candidate_counts: list[int] = []

    for case in case_rows:
        if case.case_id in case_ids:
            errors.append({"case_id": case.case_id, "message": "Duplicate case_id."})
        case_ids.add(case.case_id)
        candidate_ids = set(case.candidate_ids)
        candidate_counts.append(len(candidate_ids))
        if not case.candidates:
            warnings.append({"case_id": case.case_id, "message": "No static candidate set supplied."})
        if not case.target_candidate_ids:
            warnings.append({"case_id": case.case_id, "message": "No target product/document label supplied."})
        for target in case.target_candidate_ids:
            target_count += 1
            if candidate_ids and target not in candidate_ids:
                warnings.append(
                    {
                        "case_id": case.case_id,
                        "message": (
                            f"Target '{target}' is absent from the static candidate set. "
                            "This is allowed for trajectory/fictitious-target protocols but must be documented."
                        ),
                    }
                )

    seen_predictions: set[tuple[str, str]] = set()
    methods: Counter[str] = Counter()
    for prediction in prediction_rows:
        label = _method_label(prediction)
        methods[label] += 1
        key = (label, prediction.case_id)
        if key in seen_predictions:
            warnings.append(
                {
                    "case_id": prediction.case_id,
                    "message": f"Multiple predictions found for method '{label}'; the evaluator retains the last one.",
                }
            )
        seen_predictions.add(key)
        if prediction.case_id not in case_ids:
            errors.append({"case_id": prediction.case_id, "message": "Prediction references an unknown case_id."})
        if not prediction.ranked_candidate_ids and not prediction.cited_candidate_ids and not prediction.answer:
            warnings.append({"case_id": prediction.case_id, "message": "Prediction has no ranking, citation, or answer payload."})

    benchmark_key = None
    if benchmark:
        try:
            benchmark_key = get_benchmark_spec(benchmark).benchmark_id
        except KeyError:
            benchmark_key = str(benchmark).strip().lower().replace("-", "_")

    if benchmark_key and any(case.benchmark not in {benchmark_key, "unspecified"} for case in case_rows):
        warnings.append(
            {
                "case_id": "*",
                "message": "Some case-level benchmark labels differ from the selected benchmark; source labels were retained.",
            }
        )

    # Protocol checks are warnings rather than silent coercions.  Emerging
    # benchmark releases may legitimately differ in a documented version, but
    # the researcher should notice before calling a result a replication.
    def protocol_warning(message: str) -> None:
        warnings.append({"case_id": "*", "message": message})

    if benchmark_key == "egeo":
        nonstandard = [case.case_id for case in case_rows if case.candidates and len(case.candidates) != 10]
        if nonstandard:
            protocol_warning(
                f"E-GEO's supplied preprint protocol fixes ten candidates per query; {len(nonstandard)} loaded cases differ "
                f"(examples: {', '.join(nonstandard[:3])}). Record the release/protocol reason."
            )
        multi_target = [case.case_id for case in case_rows if len(case.target_candidate_ids) > 1]
        if multi_target:
            protocol_warning(
                f"E-GEO reranking selects one target per candidate set; {len(multi_target)} cases have multiple target labels. "
                "Use a documented selection rule or a separate multi-relevance analysis."
            )
    elif benchmark_key == "autogeo_ecommerce":
        nonstandard = [case.case_id for case in case_rows if case.candidates and len(case.candidates) != 5]
        if nonstandard:
            protocol_warning(
                f"AutoGEO's paper protocol uses five retrieved candidates; {len(nonstandard)} loaded cases differ "
                f"(examples: {', '.join(nonstandard[:3])})."
            )
    elif benchmark_key == "ifgeo":
        missing_clusters = [
            case.case_id
            for case in case_rows
            if not (case.metadata.get("query_cluster_id") or case.metadata.get("query_cluster") or case.metadata.get("parent_id"))
        ]
        if missing_clusters:
            protocol_warning(
                f"IF-GEO stability requires related-query clusters; {len(missing_clusters)} cases lack a query_cluster_id "
                f"(examples: {', '.join(missing_clusters[:3])})."
            )
    elif benchmark_key == "opr_bench":
        traced = sum(
            1
            for prediction in prediction_rows
            if isinstance(prediction.metadata.get("trace"), Sequence) and prediction.metadata.get("trace")
        )
        if prediction_rows and not traced:
            protocol_warning(
                "OPR-Bench is trajectory-level: no matching trace events were loaded. Recommendation rate can be reported, "
                "but crawl/follow-up/internal-link outcomes will be empty."
            )
    elif benchmark_key == "aces":
        missing_expected = [
            case.case_id
            for case in case_rows
            if not (case.metadata.get("expected_choice_ids") or case.metadata.get("expected_ids") or case.target_candidate_ids)
        ]
        if missing_expected:
            protocol_warning(
                f"ACES rationality requires a task-grounded expected choice label; {len(missing_expected)} cases lack one "
                f"(examples: {', '.join(missing_expected[:3])})."
            )
    elif benchmark_key == "external_engine_monitor":
        undocumented = [
            prediction.case_id
            for prediction in prediction_rows
            if not (
                prediction.model_name
                or prediction.model_version
                or prediction.metadata.get("provider")
                or prediction.metadata.get("engine")
            )
        ]
        if undocumented:
            protocol_warning(
                f"External-engine monitoring needs provider/model provenance; {len(undocumented)} predictions lack it "
                f"(examples: {', '.join(undocumented[:3])})."
            )
    elif benchmark_key == "cultural_coverage":
        missing_locale = [case.case_id for case in case_rows if not (case.metadata.get("locale") or case.metadata.get("language"))]
        if missing_locale:
            protocol_warning(
                f"Cultural coverage needs locale/language labels; {len(missing_locale)} cases lack them "
                f"(examples: {', '.join(missing_locale[:3])})."
            )
    elif benchmark_key == "human_shopping_experiment":
        with_behaviour = sum(
            1
            for prediction in prediction_rows
            if prediction.metadata.get("trace") or prediction.metadata.get("survey_scores") or prediction.metadata.get("survey")
        )
        if prediction_rows and not with_behaviour:
            protocol_warning(
                "The human-shopping profile needs pseudonymous event and/or survey data; no behavioural payload was loaded."
            )

    return {
        "valid": not errors,
        "case_count": len(case_rows),
        "prediction_count": len(prediction_rows),
        "labelled_target_count": target_count,
        "mean_candidates_per_case": round(sum(candidate_counts) / len(candidate_counts), 8) if candidate_counts else None,
        "methods": dict(sorted(methods.items())),
        "errors": errors,
        "warnings": warnings,
    }


class BenchmarkRunner:
    """Load, validate, evaluate, and report an authorised GEO benchmark export."""

    def __init__(self, evaluator: BenchmarkEvaluator | None = None) -> None:
        self.evaluator = evaluator or BenchmarkEvaluator()

    def load_dataset(
        self,
        cases_path: str | Path,
        *,
        benchmark: str,
        targets_path: str | Path | None = None,
    ) -> BenchmarkDataset:
        spec = get_benchmark_spec(benchmark)
        dataset = read_benchmark_dataset(cases_path, benchmark=spec.benchmark_id)
        cases = dataset.cases
        if targets_path is not None:
            targets = read_target_selections(targets_path)
            cases = tuple(merge_target_selections(cases, targets))
        return replace(dataset, benchmark=spec.benchmark_id, cases=cases)

    def load_predictions(
        self,
        predictions_path: str | Path,
        *,
        traces_path: str | Path | None = None,
    ) -> list[Prediction]:
        predictions = read_predictions(predictions_path)
        if traces_path is None:
            return predictions
        return attach_trace_events(predictions, read_trace_events(traces_path))

    def validate(
        self,
        cases_path: str | Path,
        *,
        benchmark: str,
        predictions_path: str | Path | None = None,
        targets_path: str | Path | None = None,
        traces_path: str | Path | None = None,
    ) -> dict[str, Any]:
        dataset = self.load_dataset(cases_path, benchmark=benchmark, targets_path=targets_path)
        predictions = (
            self.load_predictions(predictions_path, traces_path=traces_path)
            if predictions_path is not None
            else []
        )
        return validate_inputs(dataset.cases, predictions, benchmark=dataset.benchmark)

    def evaluate(
        self,
        cases_path: str | Path,
        predictions_path: str | Path,
        *,
        benchmark: str,
        baseline_method: str | None = None,
        targets_path: str | Path | None = None,
        traces_path: str | Path | None = None,
        strict: bool = False,
        run_metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        spec = get_benchmark_spec(benchmark)
        dataset = self.load_dataset(cases_path, benchmark=spec.benchmark_id, targets_path=targets_path)
        predictions = self.load_predictions(predictions_path, traces_path=traces_path)
        validation = validate_inputs(dataset.cases, predictions, benchmark=spec.benchmark_id)
        if strict and not validation["valid"]:
            messages = "; ".join(item["message"] for item in validation["errors"][:5])
            raise BenchmarkValidationError(f"Input validation failed: {messages}")

        report = self.evaluator.evaluate(
            spec.benchmark_id,
            dataset.cases,
            predictions,
            baseline_method=baseline_method,
        )
        report["benchmark_spec"] = spec.as_dict()
        report["dataset"] = {
            "benchmark": dataset.benchmark,
            "source": dataset.source,
            "protocol": dataset.protocol,
            "version": dataset.version,
            "split": dataset.split,
            "metadata": _json_safe(dataset.metadata),
        }
        report["run"] = {
            "evaluated_at": _utc_now(),
            "baseline_method": baseline_method,
            "runner": "geo-benchmark-runner/v1",
            "metadata": _json_safe(run_metadata or {}),
        }
        report.setdefault("validation", {})["input"] = validation
        return report


def _format_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, (int, str, bool)):
        return str(value)
    if isinstance(value, Mapping):
        return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ", ".join(_format_value(item) for item in value)
    return str(value)


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    output = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    output.extend("| " + " | ".join(_format_value(value).replace("|", "\\|") for value in row) + " |" for row in rows)
    return output


def report_to_markdown(report: Mapping[str, Any]) -> str:
    """Create a compact human-readable companion to the machine JSON report."""

    lines = [
        f"# {report.get('benchmark_spec', {}).get('display_name', report.get('benchmark', 'GEO benchmark'))} evaluation",
        "",
        "This report evaluates supplied, versioned outputs. It is not a claim that an official benchmark was downloaded or reproduced by this application.",
        "",
        "## Run receipt",
        "",
    ]
    dataset = report.get("dataset", {}) if isinstance(report.get("dataset"), Mapping) else {}
    validation = report.get("validation", {}) if isinstance(report.get("validation"), Mapping) else {}
    input_validation = validation.get("input", {}) if isinstance(validation.get("input"), Mapping) else {}
    lines.extend(
        _markdown_table(
            ("Field", "Value"),
            (
                ("Benchmark", report.get("benchmark")),
                ("Profiles", ", ".join(report.get("profiles", []))),
                ("Dataset version", dataset.get("version")),
                ("Dataset split", dataset.get("split")),
                ("Cases loaded", report.get("cases_loaded")),
                ("Predictions loaded", validation.get("predictions_loaded")),
                ("Input valid", input_validation.get("valid")),
                ("Baseline method", report.get("baseline_method")),
            ),
        )
    )

    method_sections = {
        "ranking": ("## Ranking", ("evaluated_cases", "mean_rank", "median_rank", "mrr", "missing_rate")),
        "visibility": ("## Citation visibility and utility", ("evaluated_cases", "citation_rate", "word", "pos", "word_pos", "overall", "precision", "recall", "clarity", "insight", "kpr", "kpc")),
        "trajectory": ("## OPR trajectory", ("evaluated_cases", "target_recommendation_rate", "initial_target_result_crawl_rate", "target_follow_up_search_rate", "internal_link_crawl_rate", "mean_trajectory_events")),
        "agent_choice": ("## ACES choice audit", ("evaluated_choices", "rationality_accuracy", "market")),
        "citation_absorption": ("## Citation absorption", ("citation_selection_rate", "claim_absorption_coverage", "target_answer_share")),
        "discovery": ("## Discovery", ("direct_queries", "organic_queries", "direct_recognition_rate", "organic_discovery_rate", "discovery_gap")),
        "consumer_behavior": ("## Consumer shopping behaviour", ("sessions", "product_view_rate", "citation_open_rate", "comparison_rate", "add_to_cart_rate", "purchase_intent_event_rate", "purchase_event_rate", "mean_logged_events")),
    }
    for key, (title, fields) in method_sections.items():
        values = report.get(key)
        if not isinstance(values, Mapping) or not values:
            continue
        lines.extend(["", title, ""])
        lines.extend(_markdown_table(("Method", *fields), ((method, *(metrics.get(field) if isinstance(metrics, Mapping) else None for field in fields)) for method, metrics in sorted(values.items()))))

    stability = report.get("multi_query_stability")
    if isinstance(stability, Mapping) and stability:
        lines.extend(["", "## Multi-query stability", ""])
        fields = ("clusters", "query_variant_pairs", "mean", "variance", "wcp", "downside_risk", "win_tie_rate", "worst_cluster_mean")
        lines.extend(_markdown_table(("Method", *fields), ((method, *(metrics.get(field) if isinstance(metrics, Mapping) else None for field in fields)) for method, metrics in sorted(stability.items()))))

    stages = report.get("stages")
    if isinstance(stages, Mapping) and stages:
        lines.extend(["", "## Stage results", ""])
        stage_rows: list[tuple[Any, ...]] = []
        for method, stage_metrics in sorted(stages.items()):
            if not isinstance(stage_metrics, Mapping):
                continue
            for stage, metrics in sorted(stage_metrics.items()):
                stage_rows.append((method, stage, metrics.get("evaluated_cases"), metrics.get("mean_rank"), metrics.get("mrr")))
        lines.extend(_markdown_table(("Method", "Stage", "Cases", "Mean rank", "MRR"), stage_rows))

    external = report.get("external_monitor")
    if isinstance(external, Mapping):
        lines.extend(["", "## External-engine monitoring", ""])
        lines.extend(_markdown_table(("Metric", "Value"), ((key, value) for key, value in external.items() if key != "methods")))

    coverage = report.get("coverage")
    if isinstance(coverage, Mapping) and coverage:
        lines.extend(["", "## Cultural / brand coverage", ""])
        lines.extend(_markdown_table(("Method", "Queries", "Surface rate"), ((method, metrics.get("evaluated_queries"), metrics.get("overall_surface_rate")) for method, metrics in sorted(coverage.items()) if isinstance(metrics, Mapping))))

    paired = report.get("paired_comparisons")
    if isinstance(paired, Mapping) and paired:
        lines.extend(["", "## Paired treatment comparisons", ""])
        paired_rows: list[tuple[Any, ...]] = []
        for method, metrics in sorted(paired.items()):
            if not isinstance(metrics, Mapping):
                continue
            for metric, details in metrics.items():
                if isinstance(details, Mapping):
                    paired_rows.append((method, metric, details.get("matched_cases"), details.get("absolute_lift"), details.get("relative_lift"), details.get("win_tie_rate")))
        lines.extend(_markdown_table(("Method", "Metric", "Matched cases", "Absolute lift", "Relative lift", "Win/tie rate"), paired_rows))

    lines.extend([
        "",
        "## Interpretation guardrail",
        "",
        "Retrieval/ranking, citations, evidence absorption, agent selection, and consumer behaviour are separate outcomes. A citation or reranking gain is not evidence of broader product discovery or purchase impact without the corresponding protocol and data.",
    ])
    return "\n".join(lines) + "\n"


def write_report(
    destination: str | Path,
    report: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Write a deterministic JSON report, refusing accidental replacement by default."""

    path = Path(destination)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {path}. Pass overwrite=True to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def write_markdown_report(
    destination: str | Path,
    report: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    path = Path(destination)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing report: {path}. Pass overwrite=True to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_to_markdown(report), encoding="utf-8")
    return path


def build_synthetic_fixture(benchmark: str) -> tuple[BenchmarkDataset, list[Prediction], list[TraceEvent]]:
    """Return a tiny, labelled fixture for smoke tests and demonstrations.

    It is intentionally fictional and too small for inference.  Its sole role
    is to prove import/evaluation/report wiring before an authorised benchmark
    export is introduced.
    """

    spec = get_benchmark_spec(benchmark)
    common_candidates = (
        Candidate(candidate_id="P-1", title="Field Bottle", metadata={"brand": "Northwind", "position": 3, "price": 34, "rating": 4.7, "sponsored": False}),
        Candidate(candidate_id="P-2", title="Trail Flask", metadata={"brand": "Contoso", "position": 1, "price": 35, "rating": 4.6, "sponsored": True}),
        Candidate(candidate_id="P-3", title="Camp Cup", metadata={"brand": "Fabrikam", "position": 2, "price": 29, "rating": 4.5, "sponsored": False}),
    )
    cases = (
        BenchmarkCase(
            case_id="fixture-001",
            query="I need a durable, leak-resistant bottle for everyday commuting.",
            benchmark=spec.benchmark_id,
            candidates=common_candidates,
            target_candidate_ids=("P-1",),
            split="fixture",
            metadata={"query_cluster_id": "durability", "query_mode": "organic", "query_type": "recommendation", "locale": "en-IN"},
        ),
        BenchmarkCase(
            case_id="fixture-002",
            query="Is Field Bottle a good option for a commuter who wants a durable bottle?",
            benchmark=spec.benchmark_id,
            candidates=common_candidates,
            target_candidate_ids=("P-1",),
            split="fixture",
            metadata={"query_cluster_id": "durability", "query_mode": "direct", "query_type": "recognition", "locale": "en-IN"},
        ),
    )
    control = [
        Prediction(
            case_id=case.case_id,
            ranked_candidate_ids=("P-2", "P-3", "P-1"),
            cited_candidate_ids=("P-2",),
            answer="Trail Flask is a commonly selected option. Field Bottle is also available.",
            condition="CONTROL",
            run_id="control-fixture",
            metadata={
                "method": "CONTROL",
                "quality_metrics": {"precision": 0.5, "recall": 0.5, "clarity": 0.6, "insight": 0.5},
                "mentioned": False,
                "survey_scores": {"trust": 4, "usefulness": 4, "purchase_intent": 3},
                "selected_id": "P-2",
            },
        )
        for case in cases
    ]
    treatment = [
        Prediction(
            case_id=case.case_id,
            ranked_candidate_ids=("P-1", "P-2", "P-3"),
            cited_candidate_ids=("P-1",),
            citations=(CitationSpan(candidate_id="P-1", text="Field Bottle is listed with a sealed lid.", source_url="https://northwind.example/products/p-1", entails=True),),
            answer="Field Bottle is the best match because its listed sealed lid and steel body address daily commuting needs.",
            condition="GEO_OPTIMIZED",
            run_id="geo-fixture",
            metadata={
                "method": "GEO_OPTIMIZED",
                "quality_metrics": {"precision": 0.8, "recall": 0.8, "clarity": 0.8, "insight": 0.7},
                "mentioned": True,
                "selected_id": "P-1",
                "absorption": {"covered_claims": 2, "eligible_claims": 2, "answer_share": 0.55},
                "claims": [{"claim": "Field Bottle has a sealed lid", "supported": True, "cited": True}],
                "survey_scores": {"trust": 5, "usefulness": 5, "purchase_intent": 5},
                "citation_urls": ["https://northwind.example/products/p-1"],
                "source_types": ["brand"],
            },
        )
        for case in cases
    ]
    traces = [
        TraceEvent(case_id="fixture-001", run_id="geo-fixture", event_type="initial_result_crawl", step=1, output_candidate_ids=("P-1",)),
        TraceEvent(case_id="fixture-001", run_id="geo-fixture", event_type="target_follow_up_search", step=2, output_candidate_ids=("P-1",)),
        TraceEvent(case_id="fixture-001", run_id="geo-fixture", event_type="internal_link_crawl", step=3, output_candidate_ids=("P-1",)),
        TraceEvent(case_id="fixture-001", run_id="geo-fixture", event_type="product_view", step=4, output_candidate_ids=("P-1",)),
        TraceEvent(case_id="fixture-001", run_id="geo-fixture", event_type="citation_open", step=5, output_candidate_ids=("P-1",)),
        TraceEvent(case_id="fixture-001", run_id="geo-fixture", event_type="add_to_cart", step=6, output_candidate_ids=("P-1",)),
    ]
    return (
        BenchmarkDataset(
            benchmark=spec.benchmark_id,
            cases=cases,
            source="synthetic-fixture (not an official benchmark release)",
            protocol="smoke-test-only",
            version="fixture-v1",
            split="fixture",
            metadata={"synthetic": True, "warning": "Not for scientific inference or benchmark claims."},
        ),
        [*control, *treatment],
        traces,
    )


def write_synthetic_fixture(
    directory: str | Path,
    *,
    benchmark: str,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Write a fixture manifest, predictions, and traces for a CLI demo."""

    folder = Path(directory)
    paths = {
        "cases": folder / f"{get_benchmark_spec(benchmark).benchmark_id}-fixture-cases.json",
        "predictions": folder / f"{get_benchmark_spec(benchmark).benchmark_id}-fixture-predictions.jsonl",
        "traces": folder / f"{get_benchmark_spec(benchmark).benchmark_id}-fixture-traces.jsonl",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refusing to overwrite existing fixture files: {names}. Pass overwrite=True to replace them.")
    folder.mkdir(parents=True, exist_ok=True)
    dataset, predictions, traces = build_synthetic_fixture(benchmark)
    write_benchmark_dataset(paths["cases"], dataset)
    write_predictions(paths["predictions"], predictions, format="jsonl")
    write_trace_events(paths["traces"], traces, format="jsonl")
    return paths


__all__ = [
    "BenchmarkRunner",
    "BenchmarkValidationError",
    "attach_trace_events",
    "build_synthetic_fixture",
    "report_to_markdown",
    "validate_inputs",
    "write_markdown_report",
    "write_report",
    "write_synthetic_fixture",
]
