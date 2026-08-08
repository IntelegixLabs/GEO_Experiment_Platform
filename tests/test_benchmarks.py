"""Regression tests for the research-only GEO benchmark runner."""

from __future__ import annotations

import json
from pathlib import Path
import sys


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.benchmarks import (
    BenchmarkEvaluator,
    BenchmarkRunner,
    CitationSpan,
    Prediction,
    attach_trace_events,
    build_synthetic_fixture,
    citation_visibility,
    list_benchmarks,
    report_to_markdown,
    write_synthetic_fixture,
)


def _evaluate_fixture(benchmark: str) -> dict[str, object]:
    dataset, predictions, traces = build_synthetic_fixture(benchmark)
    return BenchmarkEvaluator().evaluate(
        benchmark,
        dataset.cases,
        attach_trace_events(predictions, traces),
        baseline_method="CONTROL",
    )


def test_every_registered_profile_evaluates_with_a_reproducible_fixture() -> None:
    for spec in list_benchmarks():
        report = _evaluate_fixture(spec.benchmark_id)
        assert report["benchmark"] == spec.benchmark_id
        assert report["cases_loaded"] == 2
        assert report["validation"]["orphan_predictions"] == []


def test_egeo_keeps_rank_lift_and_multiquery_stability_separate() -> None:
    report = _evaluate_fixture("E-GEO")
    ranking = report["ranking"]
    assert ranking["CONTROL"]["mean_rank"] == 3.0
    assert ranking["GEO_OPTIMIZED"]["mean_rank"] == 1.0
    assert report["paired_comparisons"]["GEO_OPTIMIZED"]["rank"]["absolute_lift"] == 2.0
    assert report["paired_comparisons"]["GEO_OPTIMIZED"]["rank"]["relative_lift"] == 0.66666667
    stability = report["multi_query_stability"]["GEO_OPTIMIZED"]
    assert stability["wcp"] == 2.0
    assert stability["downside_risk"] == 0.0


def test_autogeo_and_citation_profiles_attribute_candidate_not_citation_row_id() -> None:
    visibility = citation_visibility(
        "Field Bottle is a match for commuting.",
        [CitationSpan(candidate_id="P-1", citation_id="citation-row-99", text="Field Bottle is a match.")],
        "P-1",
    )
    assert visibility["cited"] is True
    assert visibility["citation_rate"] == 1
    preserved_span = CitationSpan.from_mapping(
        {"candidate_id": "P-1", "sentence_index": 0, "attributed_words": 7, "citation_id": "c-7"}
    )
    assert citation_visibility("one two three four five six seven.", [preserved_span], "P-1")["word"] == 1.0
    report = _evaluate_fixture("autogeo_ecommerce")
    assert report["visibility"]["GEO_OPTIMIZED"]["citation_rate"] == 1.0
    assert report["visibility"]["GEO_OPTIMIZED"]["precision"] == 0.8


def test_method_column_preserved_in_prediction_metadata_is_used_for_grouping() -> None:
    dataset, _, _ = build_synthetic_fixture("egeo")
    prediction = Prediction.from_mapping(
        {"case_id": "fixture-001", "method": "CONTROL", "ranking": ["P-2", "P-3", "P-1"]}
    )
    report = BenchmarkEvaluator().evaluate("egeo", dataset.cases, [prediction], baseline_method="CONTROL")
    assert report["methods"] == ["CONTROL"]


def test_opr_aces_and_human_profiles_report_protocol_specific_outcomes() -> None:
    opr = _evaluate_fixture("opr_bench")
    assert opr["trajectory"]["GEO_OPTIMIZED"]["target_recommendation_rate"] == 1.0
    assert opr["trajectory"]["GEO_OPTIMIZED"]["initial_target_result_crawl_rate"] == 0.5
    assert opr["trajectory"]["GEO_OPTIMIZED"]["target_follow_up_search_rate"] == 0.5
    assert opr["trajectory"]["GEO_OPTIMIZED"]["internal_link_crawl_rate"] == 0.5

    aces = _evaluate_fixture("aces")
    assert aces["agent_choice"]["GEO_OPTIMIZED"]["evaluated_choices"] == 2
    assert "market" in aces["agent_choice"]["GEO_OPTIMIZED"]

    behavior = _evaluate_fixture("human_shopping_experiment")
    assert behavior["consumer_behavior"]["GEO_OPTIMIZED"]["sessions"] == 2
    assert behavior["consumer_behavior"]["GEO_OPTIMIZED"]["product_view_rate"] == 0.5
    assert behavior["consumer_behavior"]["GEO_OPTIMIZED"]["add_to_cart_rate"] == 0.5


def test_defensive_manipulation_and_external_monitor_metrics_are_read_only() -> None:
    manipulation = _evaluate_fixture("geo_bench_manipulation")
    audit = manipulation["manipulation_audit"]["GEO_OPTIMIZED"]["paired_rank_audit"]
    assert audit["normalized_rank_gain"] == 1.0
    assert audit["success_at"]["0.3"] == 1.0

    monitor = _evaluate_fixture("external_engine_monitor")
    assert monitor["external_monitor"]["comparable_query_cases"] == 2
    assert "mean_citation_domain_jaccard" in monitor["external_monitor"]


def test_runner_writes_fixture_and_machine_plus_markdown_reports(tmp_path: Path) -> None:
    paths = write_synthetic_fixture(tmp_path, benchmark="egeo")
    report = BenchmarkRunner().evaluate(
        paths["cases"],
        paths["predictions"],
        benchmark="egeo",
        traces_path=paths["traces"],
        baseline_method="CONTROL",
        strict=True,
        run_metadata={"seed": 42, "model": "fixture"},
    )
    assert report["validation"]["input"]["valid"] is True
    assert report["dataset"]["metadata"]["synthetic"] is True
    assert any("fixes ten candidates" in item["message"] for item in report["validation"]["input"]["warnings"])
    markdown = report_to_markdown(report)
    assert "# E-GEO evaluation" in markdown
    assert "not a claim" in markdown

    output = tmp_path / "report.json"
    from app.benchmarks import write_report

    write_report(output, report)
    decoded = json.loads(output.read_text(encoding="utf-8"))
    assert decoded["run"]["metadata"]["seed"] == 42
