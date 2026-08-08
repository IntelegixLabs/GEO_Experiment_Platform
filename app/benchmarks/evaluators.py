"""Benchmark-profile evaluators for transparent GEO experiments.

The code evaluates logged outputs.  It does not call commercial models, scrape
answer engines, or create manipulation prompts.  This makes the results
reproducible and keeps external-engine replication as an explicit, separately
logged study mode.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from math import ceil
from statistics import mean, pvariance
from typing import Any
from urllib.parse import urlparse

from .metrics import (
    as_float,
    citation_visibility,
    claim_support_metrics,
    hit_at,
    market_concentration,
    mean_or_none,
    paired_effects,
    rank_of,
    reciprocal_rank,
    safe_divide,
    structural_features,
)
from .registry import normalize_benchmark_id


PROFILE_BY_BENCHMARK = {
    "egeo": ("ranking", "multi_query", "integrity"),
    "autogeo_ecommerce": ("ranking", "visibility", "utility", "integrity"),
    "geo_bench": ("visibility", "utility", "integrity"),
    "researchy_geo": ("visibility", "utility", "integrity"),
    "opr_bench": ("trajectory", "integrity"),
    "aces": ("agent_choice",),
    "sageo": ("stage", "visibility", "integrity"),
    "ifgeo": ("visibility", "multi_query", "integrity"),
    "citation_absorption": ("visibility", "absorption", "integrity"),
    "geo_bench_manipulation": ("ranking", "manipulation_audit", "integrity"),
    "study_framework": ("stage", "visibility", "multi_query", "discovery", "behavior", "integrity"),
    "external_engine_monitor": ("ranking", "visibility", "external_monitor", "integrity"),
    "discovery_gap": ("ranking", "discovery", "integrity"),
    "cultural_coverage": ("visibility", "coverage", "integrity"),
    "multimodal_geo": ("stage", "ranking", "visibility", "integrity"),
    "human_shopping_experiment": ("behavior", "visibility", "integrity"),
    "competitive_citation": ("visibility", "integrity"),
}


def _read(record: Any, *keys: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        for key in keys:
            if key in record and record[key] is not None:
                return record[key]
        return default
    for key in keys:
        value = getattr(record, key, None)
        if value is not None:
            return value
    return default


def _as_mapping(record: Any) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    method = getattr(record, "as_dict", None)
    if callable(method):
        result = method()
        return dict(result) if isinstance(result, Mapping) else {}
    raw = getattr(record, "__dict__", None)
    return dict(raw) if isinstance(raw, Mapping) else {}


def _case_id(case: Any) -> str:
    return str(_read(case, "case_id", "id", "query_id", "custom_id", default=""))


def _prediction_case_id(prediction: Any) -> str:
    return str(_read(prediction, "case_id", "id", "query_id", "custom_id", default=""))


def _method_id(prediction: Any) -> str:
    # ``Prediction`` deliberately uses ``condition`` (CONTROL, GEO_OPTIMIZED,
    # etc.) rather than assuming every benchmark calls an experimental arm a
    # "method".  Raw provider exports frequently keep their method label in
    # metadata, so resolve both forms before falling back to a run identifier.
    metadata = _metadata(prediction)
    value = _read(prediction, "method", "method_id", "system", "model", "label", "condition", default=None)
    if value in (None, ""):
        value = (
            metadata.get("method")
            or metadata.get("method_id")
            or metadata.get("system")
            or metadata.get("condition")
        )
    if value in (None, ""):
        value = _read(prediction, "run_id", default=None)
    return str(value or "candidate")


def _metadata(record: Any) -> dict[str, Any]:
    candidate = _read(record, "metadata", "meta", default={})
    return dict(candidate) if isinstance(candidate, Mapping) else {}


def _identifier(record: Any, fallback: int | None = None) -> str | None:
    # Prefer the entity identity over a row/citation id.  This matters for
    # citation exports in which ``id`` identifies a citation span while
    # ``product_id`` identifies the cited product.
    value = _read(record, "candidate_id", "product_id", "document_id", "source_id", "asin", "id", default=None)
    if value in (None, ""):
        return str(fallback) if fallback is not None else None
    return str(value)


def _candidate_records(case: Any) -> list[Any]:
    records = _read(case, "candidates", "products", "documents", "items", "listings", default=[])
    return list(records) if isinstance(records, Sequence) and not isinstance(records, (str, bytes)) else []


def _candidate_ids(case: Any) -> list[str]:
    output: list[str] = []
    for index, candidate in enumerate(_candidate_records(case), start=1):
        identifier = _identifier(candidate, fallback=index)
        if identifier is not None:
            output.append(identifier)
    return output


def _target_id(case: Any) -> str | None:
    direct = _read(
        case,
        "target_id",
        "target_product_id",
        "target_document_id",
        "selected_product_id",
        "target_candidate_ids",
        "target_ids",
        default=None,
    )
    if direct not in (None, ""):
        if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)):
            return str(direct[0]) if direct else None
        return str(direct)
    target = _read(case, "target", "target_product", "target_document", default=None)
    if target not in (None, ""):
        if isinstance(target, Mapping) or hasattr(target, "__dict__"):
            return _identifier(target)
        return str(target)
    return None


def _ids(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Mapping):
        maybe_ids = _read(value, "ids", "ranking", "ranked_ids", "items", default=None)
        if maybe_ids is not None:
            return _ids(maybe_ids)
        identifier = _identifier(value)
        return [identifier] if identifier else []
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        output: list[str] = []
        for index, item in enumerate(value, start=1):
            if isinstance(item, Mapping) or hasattr(item, "__dict__"):
                identifier = _identifier(item, fallback=index)
                if identifier is not None:
                    output.append(identifier)
            elif item not in (None, ""):
                output.append(str(item))
        return output
    return [str(value)]


def _ranked_ids(prediction: Any, stage: str | None = None) -> list[str]:
    metadata = _metadata(prediction)
    if stage:
        stage_key = stage.lower().replace("-", "_")
        stages = _read(prediction, "stages", "stage_rankings", default=metadata.get("stages"))
        if isinstance(stages, Mapping):
            nested = stages.get(stage_key) or stages.get(stage)
            if nested is not None:
                return _ids(nested)
        for key in (
            f"{stage_key}_ranked_ids",
            f"{stage_key}_ranking",
            f"{stage_key}_candidates",
        ):
            nested = _read(prediction, key, default=metadata.get(key))
            if nested is not None:
                return _ids(nested)
        return []
    value = _read(
        prediction,
        "ranked_ids",
        "ranked_product_ids",
        "ranked_document_ids",
        "ranking",
        "ranked_candidates",
        default=None,
    )
    if value is None:
        value = metadata.get("ranked_ids") or metadata.get("ranking")
    return _ids(value)


def _cited_ids(prediction: Any) -> list[str]:
    metadata = _metadata(prediction)
    value = _read(
        prediction,
        "cited_ids",
        "citation_ids",
        "cited_product_ids",
        "cited_document_ids",
        "recommended_ids",
        default=None,
    )
    if value is None:
        value = metadata.get("cited_ids") or metadata.get("recommended_ids")
    if value is not None:
        return _ids(value)
    citations = _read(prediction, "citations", "citation_spans", "attributions", default=[])
    values: list[str] = []
    for citation in citations if isinstance(citations, Sequence) else []:
        values.extend(_ids(citation))
    return values


def _citations(prediction: Any) -> list[Any]:
    value = _read(prediction, "citations", "citation_spans", "attributions", default=None)
    if value is None:
        value = _metadata(prediction).get("citations") or _metadata(prediction).get("citation_spans")
    return list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []


def _trace_events(prediction: Any) -> list[Any]:
    value = _read(prediction, "trace", "trajectory", "events", default=None)
    if value is None:
        metadata = _metadata(prediction)
        value = metadata.get("trace") or metadata.get("trajectory") or metadata.get("events")
    return list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []


def _claims(prediction: Any) -> list[Any]:
    value = _read(prediction, "claims", "claim_labels", default=None)
    if value is None:
        value = _metadata(prediction).get("claims") or _metadata(prediction).get("claim_labels")
    return list(value) if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) else []


def _content(prediction: Any) -> str:
    value = _read(prediction, "rewritten_content", "generated_content", "content", "document", default=None)
    if value is None:
        metadata = _metadata(prediction)
        value = metadata.get("rewritten_content") or metadata.get("generated_content") or ""
    return str(value or "")


def _answer(prediction: Any) -> str:
    return str(_read(prediction, "answer", "response", "response_text", default="") or "")


def _selected_id(prediction: Any) -> str | None:
    selected = _read(prediction, "selected_id", "selected_product_id", "choice_id", "chosen_product_id", default=None)
    if selected is None:
        selected = _metadata(prediction).get("selected_id") or _metadata(prediction).get("choice_id")
    return str(selected) if selected not in (None, "") else None


def _case_lookup(cases: Iterable[Any]) -> dict[str, Any]:
    return {identifier: case for case in cases if (identifier := _case_id(case))}


def _prediction_lookup(predictions: Iterable[Any]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = defaultdict(dict)
    for prediction in predictions:
        case_id = _prediction_case_id(prediction)
        if case_id:
            output[_method_id(prediction)][case_id] = prediction
    return dict(output)


def _rank_rows(cases: Mapping[str, Any], predictions: Mapping[str, Any], *, stage: str | None = None) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for case_id, prediction in predictions.items():
        case = cases.get(case_id)
        if case is None:
            continue
        target_id = _target_id(case)
        if target_id is None:
            continue
        candidate_ids = _candidate_ids(case)
        ranked = _ranked_ids(prediction, stage=stage)
        if not ranked:
            continue
        target_rank = rank_of(ranked, target_id, candidate_count=len(candidate_ids) or len(ranked))
        rows[case_id] = {
            "case_id": case_id,
            "target_id": target_id,
            "target_rank": target_rank,
            "candidate_count": len(candidate_ids) or len(ranked),
            "ranking_length": len(ranked),
            "missing_target": target_rank is not None and target_rank > (len(candidate_ids) or len(ranked)),
        }
    return rows


def _rank_summary(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    ranks = [as_float(row.get("target_rank")) for row in rows.values()]
    usable = [rank for rank in ranks if rank is not None]
    if not usable:
        return {"evaluated_cases": 0, "mean_rank": None, "mrr": None, "hit_at": {}, "missing_rate": None}
    output: dict[str, Any] = {
        "evaluated_cases": len(usable),
        "mean_rank": mean_or_none(usable),
        "median_rank": sorted(usable)[len(usable) // 2] if len(usable) % 2 else mean_or_none(sorted(usable)[len(usable) // 2 - 1 : len(usable) // 2 + 1]),
        "mrr": mean_or_none([reciprocal_rank(int(rank)) for rank in usable]),
        "hit_at": {str(cutoff): mean_or_none([hit_at(int(rank), cutoff) for rank in usable]) for cutoff in (1, 3, 5, 10, 20, 100)},
        "missing_rate": mean_or_none([int(bool(row.get("missing_target"))) for row in rows.values()]),
    }
    return output


def _visibility_rows(cases: Mapping[str, Any], predictions: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for case_id, prediction in predictions.items():
        case = cases.get(case_id)
        if case is None:
            continue
        target_id = _target_id(case)
        if target_id is None:
            continue
        visibility = citation_visibility(_answer(prediction), _citations(prediction), target_id)
        cited_ids = set(_cited_ids(prediction))
        if target_id in cited_ids:
            visibility["cited"] = True
            visibility["citation_rate"] = 1
        quality = _read(prediction, "quality_metrics", "subjective_metrics", default=None)
        if not isinstance(quality, Mapping):
            metadata = _metadata(prediction)
            quality = metadata.get("quality_metrics") or metadata.get("subjective_metrics") or {}
        claims = _claims(prediction)
        row: dict[str, Any] = {
            "case_id": case_id,
            "target_id": target_id,
            **visibility,
            "overall": as_float(quality.get("overall")) if isinstance(quality, Mapping) else None,
            "precision": as_float(quality.get("precision")) if isinstance(quality, Mapping) else None,
            "recall": as_float(quality.get("recall")) if isinstance(quality, Mapping) else None,
            "clarity": as_float(quality.get("clarity")) if isinstance(quality, Mapping) else None,
            "insight": as_float(quality.get("insight")) if isinstance(quality, Mapping) else None,
            "kpr": as_float(quality.get("kpr")) if isinstance(quality, Mapping) else None,
            "kpc": as_float(quality.get("kpc")) if isinstance(quality, Mapping) else None,
            "claim_metrics": claim_support_metrics(claims) if claims else None,
            "structural_features": structural_features(_content(prediction)) if _content(prediction) else None,
        }
        rows[case_id] = row
    return rows


def _visibility_summary(rows: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    scalar_names = (
        "citation_rate",
        "word",
        "pos",
        "word_pos",
        "overall",
        "precision",
        "recall",
        "clarity",
        "insight",
        "kpr",
        "kpc",
    )
    output: dict[str, Any] = {"evaluated_cases": len(rows)}
    for name in scalar_names:
        output[name] = mean_or_none([as_float(row.get(name)) for row in rows.values()])
    all_claims: list[Any] = []
    for row in rows.values():
        metrics = row.get("claim_metrics")
        if isinstance(metrics, Mapping):
            all_claims.append(metrics)
    if all_claims:
        output["claim_audit"] = {
            key: mean_or_none([as_float(metrics.get(key)) for metrics in all_claims])
            for key in ("citation_precision", "citation_recall", "unsupported_claim_rate")
        }
    return output


def _method_comparisons(
    rank_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    visibility_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    baseline_method: str | None,
) -> dict[str, Any]:
    if not baseline_method or baseline_method not in set(rank_rows) | set(visibility_rows):
        return {}
    comparisons: dict[str, Any] = {}
    baseline_ranks = {case_id: row.get("target_rank") for case_id, row in rank_rows.get(baseline_method, {}).items()}
    baseline_visibility = visibility_rows.get(baseline_method, {})
    for method in sorted((set(rank_rows) | set(visibility_rows)) - {baseline_method}):
        method_comparison: dict[str, Any] = {}
        current_ranks = {case_id: row.get("target_rank") for case_id, row in rank_rows.get(method, {}).items()}
        if baseline_ranks and current_ranks:
            method_comparison["rank"] = paired_effects(
                baseline_ranks,
                current_ranks,
                higher_is_better=False,
            )
        for metric in ("citation_rate", "word", "pos", "word_pos", "overall", "precision", "recall", "clarity", "insight", "kpr", "kpc"):
            before = {case_id: row.get(metric) for case_id, row in baseline_visibility.items()}
            after = {case_id: row.get(metric) for case_id, row in visibility_rows.get(method, {}).items()}
            if before and after:
                method_comparison[metric] = paired_effects(before, after, higher_is_better=metric != "kpc")
        comparisons[method] = method_comparison
    return comparisons


def _stage_evaluation(cases: Mapping[str, Any], by_method: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    stages = ("retrieval", "rerank")
    output: dict[str, Any] = {}
    for method, predictions in by_method.items():
        method_stages: dict[str, Any] = {}
        for stage in stages:
            rows = _rank_rows(cases, predictions, stage=stage)
            if rows:
                method_stages[stage] = _rank_summary(rows)
        output[method] = method_stages
    return output


def _trajectory_evaluation(cases: Mapping[str, Any], by_method: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for method, predictions in by_method.items():
        target_recommended: list[int] = []
        initial_crawl: list[int] = []
        target_followup: list[int] = []
        internal_crawl: list[int] = []
        lengths: list[int] = []
        for case_id, prediction in predictions.items():
            case = cases.get(case_id)
            target_id = _target_id(case) if case else None
            if not target_id:
                continue
            recommended = set(_cited_ids(prediction))
            selected = _selected_id(prediction)
            if selected:
                recommended.add(selected)
            target_recommended.append(int(target_id in recommended))
            events = _trace_events(prediction)
            lengths.append(len(events))
            target_initial = target_follow = target_internal = False
            for event in events:
                event_type = str(_read(event, "event_type", "type", "action", default="")).lower()
                event_meta = _metadata(event)
                event_ids: set[str] = set()
                # Empty input lists are meaningful trace values, but must not
                # hide a populated output list (a common crawl-event shape).
                for key in (
                    "target_id",
                    "product_id",
                    "document_id",
                    "source_id",
                    "input_candidate_ids",
                    "output_candidate_ids",
                ):
                    event_ids.update(_ids(_read(event, key, default=None)))
                event_ids.update(
                    _ids(
                        event_meta.get("target_id")
                        or event_meta.get("product_id")
                        or event_meta.get("document_id")
                        or event_meta.get("candidate_id")
                        or event_meta.get("output_candidate_ids")
                        or event_meta.get("input_candidate_ids")
                    )
                )
                round_value = _read(event, "round", "step", "turn", default=event_meta.get("round", event_meta.get("step")))
                is_initial = round_value in (0, "0", 1, "1", None)
                if target_id in event_ids and "crawl" in event_type and is_initial:
                    target_initial = True
                if target_id in event_ids and "follow" in event_type and "search" in event_type:
                    target_follow = True
                if target_id in event_ids and "internal" in event_type and "crawl" in event_type:
                    target_internal = True
            initial_crawl.append(int(target_initial))
            target_followup.append(int(target_follow))
            internal_crawl.append(int(target_internal))
        output[method] = {
            "evaluated_cases": len(target_recommended),
            "target_recommendation_rate": mean_or_none(target_recommended),
            "initial_target_result_crawl_rate": mean_or_none(initial_crawl),
            "target_follow_up_search_rate": mean_or_none(target_followup),
            "internal_link_crawl_rate": mean_or_none(internal_crawl),
            "mean_trajectory_events": mean_or_none(lengths),
        }
    return output


def _agent_choice_evaluation(cases: Mapping[str, Any], by_method: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    tracked_factors = ("position", "row", "column", "sponsored", "overall_pick", "endorsement", "scarcity")
    for method, predictions in by_method.items():
        selected_ids: list[str] = []
        correctness: list[int] = []
        exposed: dict[str, Counter[str]] = {factor: Counter() for factor in tracked_factors}
        selected: dict[str, Counter[str]] = {factor: Counter() for factor in tracked_factors}
        values_exposed: dict[str, list[float]] = defaultdict(list)
        values_selected: dict[str, list[float]] = defaultdict(list)
        for case_id, prediction in predictions.items():
            case = cases.get(case_id)
            if case is None:
                continue
            selection = _selected_id(prediction)
            if not selection:
                continue
            selected_ids.append(selection)
            case_metadata = _metadata(case)
            expected = _ids(
                _read(case, "expected_choice_ids", "expected_ids", "correct_ids", default=None)
                or case_metadata.get("expected_choice_ids")
                or case_metadata.get("expected_ids")
                or _read(case, "target_candidate_ids", "target_ids", default=None)
            )
            if expected:
                correctness.append(int(selection in set(expected)))
            candidates = _candidate_records(case)
            selected_candidate: Any | None = None
            for index, candidate in enumerate(candidates, start=1):
                if _identifier(candidate, fallback=index) == selection:
                    selected_candidate = candidate
                candidate_metadata = _metadata(candidate)
                for factor in tracked_factors:
                    value = _read(candidate, factor, default=candidate_metadata.get(factor))
                    if value is not None:
                        exposed[factor][str(value)] += 1
                for numeric in ("price", "rating", "review_count", "reviews"):
                    number = as_float(_read(candidate, numeric, default=candidate_metadata.get(numeric)))
                    if number is not None:
                        values_exposed[numeric].append(number)
            if selected_candidate is not None:
                selected_metadata = _metadata(selected_candidate)
                for factor in tracked_factors:
                    value = _read(selected_candidate, factor, default=selected_metadata.get(factor))
                    if value is not None:
                        selected[factor][str(value)] += 1
                for numeric in ("price", "rating", "review_count", "reviews"):
                    number = as_float(_read(selected_candidate, numeric, default=selected_metadata.get(numeric)))
                    if number is not None:
                        values_selected[numeric].append(number)
        factor_effects: dict[str, Any] = {}
        for factor in tracked_factors:
            factor_effects[factor] = {}
            total_exposed = sum(exposed[factor].values())
            total_selected = sum(selected[factor].values())
            for value in sorted(set(exposed[factor]) | set(selected[factor])):
                exposure_share = safe_divide(exposed[factor][value], total_exposed)
                selection_share = safe_divide(selected[factor][value], total_selected)
                factor_effects[factor][value] = {
                    "exposure_count": exposed[factor][value],
                    "selection_count": selected[factor][value],
                    "exposure_share": exposure_share,
                    "selection_share": selection_share,
                    "selection_to_exposure_ratio": safe_divide(selection_share or 0.0, exposure_share or 0.0),
                }
        output[method] = {
            "evaluated_choices": len(selected_ids),
            "rationality_accuracy": mean_or_none(correctness),
            "market": market_concentration(selected_ids),
            "factor_effects": factor_effects,
            "selected_vs_exposed_means": {
                numeric: {
                    "selected": mean_or_none(values_selected[numeric]),
                    "exposed": mean_or_none(values_exposed[numeric]),
                }
                for numeric in ("price", "rating", "review_count", "reviews")
            },
        }
    return output


def _multi_query_stability(
    cases: Mapping[str, Any],
    rank_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    visibility_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    baseline_method: str | None,
) -> dict[str, Any]:
    if not baseline_method:
        return {}
    output: dict[str, Any] = {}
    baseline_rank = rank_rows.get(baseline_method, {})
    baseline_visibility = visibility_rows.get(baseline_method, {})
    all_methods = (set(rank_rows) | set(visibility_rows)) - {baseline_method}
    for method in sorted(all_methods):
        by_cluster: dict[str, list[float]] = defaultdict(list)
        method_rank = rank_rows.get(method, {})
        for case_id in set(baseline_rank) & set(method_rank):
            before = as_float(baseline_rank[case_id].get("target_rank"))
            after = as_float(method_rank[case_id].get("target_rank"))
            if before is None or after is None:
                continue
            metadata = _metadata(cases[case_id])
            cluster = str(metadata.get("query_cluster_id") or metadata.get("query_cluster") or metadata.get("parent_id") or case_id)
            by_cluster[cluster].append(before - after)
        if not by_cluster:
            for case_id in set(baseline_visibility) & set(visibility_rows.get(method, {})):
                before = as_float(baseline_visibility[case_id].get("word_pos"))
                after = as_float(visibility_rows[method][case_id].get("word_pos"))
                if before is None or after is None:
                    continue
                metadata = _metadata(cases[case_id])
                cluster = str(metadata.get("query_cluster_id") or metadata.get("query_cluster") or metadata.get("parent_id") or case_id)
                by_cluster[cluster].append(after - before)
        cluster_means = [mean(deltas) for deltas in by_cluster.values() if deltas]
        all_deltas = [delta for deltas in by_cluster.values() for delta in deltas]
        output[method] = {
            "clusters": len(by_cluster),
            "query_variant_pairs": len(all_deltas),
            "mean": mean_or_none(all_deltas),
            "variance": round(pvariance(all_deltas), 8) if len(all_deltas) > 1 else 0.0 if all_deltas else None,
            "wcp": round(min(all_deltas), 8) if all_deltas else None,
            "downside_risk": mean_or_none([min(0.0, delta) ** 2 for delta in all_deltas]),
            "win_tie_rate": mean_or_none([int(delta >= 0.0) for delta in all_deltas]),
            "mean_cluster_lift": mean_or_none(cluster_means),
            "worst_cluster_mean": round(min(cluster_means), 8) if cluster_means else None,
            "per_cluster": {cluster: {"mean": mean_or_none(deltas), "wcp": min(deltas)} for cluster, deltas in sorted(by_cluster.items()) if deltas},
        }
    return output


def _absorption_evaluation(cases: Mapping[str, Any], by_method: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Keep citation selection distinct from evidence absorption."""

    output: dict[str, Any] = {}
    for method, predictions in by_method.items():
        selection: list[int] = []
        coverage: list[float] = []
        answer_share: list[float] = []
        for case_id, prediction in predictions.items():
            target = _target_id(cases.get(case_id)) if cases.get(case_id) else None
            if not target:
                continue
            selection.append(int(target in set(_cited_ids(prediction))))
            metadata = _metadata(prediction)
            absorption = _read(prediction, "absorption", default=metadata.get("absorption"))
            if not isinstance(absorption, Mapping):
                continue
            covered = as_float(absorption.get("covered_claims"))
            eligible = as_float(absorption.get("eligible_claims"))
            if covered is not None and eligible not in (None, 0.0):
                coverage.append(covered / eligible)
            share = as_float(absorption.get("answer_share") or absorption.get("attributed_word_share"))
            if share is not None:
                answer_share.append(share)
        output[method] = {
            "citation_selection_rate": mean_or_none(selection),
            "claim_absorption_coverage": mean_or_none(coverage),
            "target_answer_share": mean_or_none(answer_share),
        }
    return output


def _discovery_evaluation(cases: Mapping[str, Any], by_method: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for method, predictions in by_method.items():
        direct: list[int] = []
        organic: list[int] = []
        for case_id, prediction in predictions.items():
            case = cases.get(case_id)
            if case is None:
                continue
            query_mode = str(_metadata(case).get("query_mode") or _metadata(case).get("query_type") or "").lower()
            target = _target_id(case)
            metadata = _metadata(prediction)
            surfaced = metadata.get("mentioned")
            if surfaced is None:
                surfaced = bool(target and (target in set(_cited_ids(prediction)) or target == _selected_id(prediction)))
            if query_mode in {"direct", "direct_name", "recognition", "named"}:
                direct.append(int(bool(surfaced)))
            elif query_mode in {"organic", "discovery", "recommendation", "scenario", "comparison"}:
                organic.append(int(bool(surfaced)))
        direct_rate = mean_or_none(direct)
        organic_rate = mean_or_none(organic)
        output[method] = {
            "direct_recognition_rate": direct_rate,
            "organic_discovery_rate": organic_rate,
            "discovery_gap": safe_divide(direct_rate or 0.0, organic_rate or 0.0),
            "direct_queries": len(direct),
            "organic_queries": len(organic),
        }
    return output


def _surface_label(case: Any, prediction: Any) -> int | None:
    """Return a logged target-surfacing label without guessing from prose.

    Discovery and coverage studies need a labelled mention/recognition event.
    When a provider has not supplied one, a target citation or final selection
    is a conservative observable proxy.  We intentionally do not regex-match
    a product name in generated prose because aliases make that non-reproducible.
    """

    target = _target_id(case)
    metadata = _metadata(prediction)
    surfaced = metadata.get("mentioned")
    if surfaced is None:
        surfaced = metadata.get("surfaced")
    if surfaced is None:
        surfaced = metadata.get("recognized")
    if surfaced is None:
        surfaced = bool(target and (target in set(_cited_ids(prediction)) or target == _selected_id(prediction)))
    return int(bool(surfaced)) if surfaced is not None else None


def _rate_by_group(values: Mapping[str, list[int]]) -> dict[str, dict[str, Any]]:
    return {
        group: {"queries": len(items), "surface_rate": mean_or_none(items)}
        for group, items in sorted(values.items())
    }


def _coverage_evaluation(cases: Mapping[str, Any], by_method: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Report cross-locale/query/brand surfacing coverage for audit studies."""

    output: dict[str, Any] = {}
    for method, predictions in by_method.items():
        by_locale: dict[str, list[int]] = defaultdict(list)
        by_query_type: dict[str, list[int]] = defaultdict(list)
        by_brand: dict[str, list[int]] = defaultdict(list)
        all_labels: list[int] = []
        for case_id, prediction in predictions.items():
            case = cases.get(case_id)
            if case is None:
                continue
            label = _surface_label(case, prediction)
            if label is None:
                continue
            metadata = _metadata(case)
            locale = str(metadata.get("locale") or metadata.get("language") or "unspecified")
            query_type = str(metadata.get("query_type") or metadata.get("query_mode") or "unspecified")
            target = _target_id(case)
            candidate = case.candidate(target) if target and hasattr(case, "candidate") else None
            candidate_metadata = _metadata(candidate) if candidate is not None else {}
            brand = str(metadata.get("brand") or candidate_metadata.get("brand") or "unspecified")
            by_locale[locale].append(label)
            by_query_type[query_type].append(label)
            by_brand[brand].append(label)
            all_labels.append(label)
        output[method] = {
            "evaluated_queries": len(all_labels),
            "overall_surface_rate": mean_or_none(all_labels),
            "by_locale": _rate_by_group(by_locale),
            "by_query_type": _rate_by_group(by_query_type),
            "by_brand": _rate_by_group(by_brand),
        }
    return output


def _normalised_domain(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    parsed = urlparse(text if "://" in text else f"//{text}")
    host = (parsed.hostname or "").strip().lower()
    if host.startswith("www."):
        host = host[4:]
    return host or None


def _external_domains(prediction: Any) -> set[str]:
    domains: set[str] = set()
    metadata = _metadata(prediction)
    raw_urls = metadata.get("citation_urls") or metadata.get("source_urls") or metadata.get("urls") or []
    if isinstance(raw_urls, Mapping):
        raw_urls = list(raw_urls.values())
    if isinstance(raw_urls, Sequence) and not isinstance(raw_urls, (str, bytes)):
        for value in raw_urls:
            if isinstance(value, Mapping):
                value = _read(value, "url", "source_url", "link", "domain", default=None)
            domain = _normalised_domain(value)
            if domain:
                domains.add(domain)
    for citation in _citations(prediction):
        domain = _normalised_domain(_read(citation, "source_url", "url", "link", default=None))
        if domain:
            domains.add(domain)
    return domains


def _jaccard(left: set[str], right: set[str]) -> float | None:
    union = left | right
    return safe_divide(len(left & right), len(union)) if union else None


def _external_monitor_evaluation(
    cases: Mapping[str, Any], by_method: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    """Summarise provider/domain agreement from already logged external runs.

    The evaluator never queries external engines.  It only compares raw output
    metadata that a researcher collected with provider/model/date/locale
    provenance.
    """

    by_case: dict[str, list[tuple[str, Any]]] = defaultdict(list)
    source_types: Counter[str] = Counter()
    method_rows: dict[str, dict[str, Any]] = {}
    for method, predictions in by_method.items():
        domain_counts: list[int] = []
        citation_counts: list[int] = []
        for case_id, prediction in predictions.items():
            by_case[case_id].append((method, prediction))
            domains = _external_domains(prediction)
            domain_counts.append(len(domains))
            citation_counts.append(len(set(_cited_ids(prediction))))
            metadata = _metadata(prediction)
            raw_types = metadata.get("source_types") or metadata.get("citation_source_types") or []
            if isinstance(raw_types, Mapping):
                raw_types = list(raw_types.values())
            if isinstance(raw_types, Sequence) and not isinstance(raw_types, (str, bytes)):
                source_types.update(str(item) for item in raw_types if item not in (None, ""))
        method_rows[method] = {
            "observations": len(predictions),
            "mean_citation_domains": mean_or_none(domain_counts),
            "mean_cited_candidates": mean_or_none(citation_counts),
        }

    domain_agreement: list[float] = []
    product_agreement: list[float] = []
    comparable_cases = 0
    for outputs in by_case.values():
        if len(outputs) < 2:
            continue
        comparable_cases += 1
        for (_, first), (_, second) in combinations(outputs, 2):
            domain = _jaccard(_external_domains(first), _external_domains(second))
            if domain is not None:
                domain_agreement.append(domain)
            product = _jaccard(set(_cited_ids(first)), set(_cited_ids(second)))
            if product is not None:
                product_agreement.append(product)

    # A query cluster can contain controlled paraphrases.  Compare each
    # provider's outputs within a cluster rather than calling them independent
    # replications.
    cluster_outputs: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for method, predictions in by_method.items():
        for case_id, prediction in predictions.items():
            case = cases.get(case_id)
            if case is None:
                continue
            metadata = _metadata(case)
            cluster = str(metadata.get("query_cluster_id") or metadata.get("query_cluster") or case_id)
            cluster_outputs[(method, cluster)].append(prediction)
    paraphrase_agreement: list[float] = []
    for outputs in cluster_outputs.values():
        for first, second in combinations(outputs, 2):
            overlap = _jaccard(_external_domains(first), _external_domains(second))
            if overlap is not None:
                paraphrase_agreement.append(overlap)

    return {
        "methods": method_rows,
        "comparable_query_cases": comparable_cases,
        "mean_citation_domain_jaccard": mean_or_none(domain_agreement),
        "mean_cited_product_jaccard": mean_or_none(product_agreement),
        "mean_paraphrase_domain_jaccard": mean_or_none(paraphrase_agreement),
        "source_type_counts": dict(sorted(source_types.items())),
    }


def _event_type(event: Any) -> str:
    return str(_read(event, "event_type", "type", "event", "action", "name", default="")).strip().lower().replace("-", "_")


def _behavior_evaluation(cases: Mapping[str, Any], by_method: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Summarise logged participant behaviour and post-task survey scores.

    This intentionally produces descriptive outcomes only.  Treatment effects
    should be estimated from the preregistered randomisation unit, not by
    pooling unrelated participant events with model benchmark observations.
    """

    event_groups = {
        "product_view_rate": ("product_view", "view_product", "product_open", "open_product"),
        "citation_open_rate": ("citation_open", "open_citation", "evidence_view", "reference_open"),
        "comparison_rate": ("compare", "comparison", "product_compare"),
        "save_or_click_rate": ("save", "bookmark", "click", "product_click"),
        "add_to_cart_rate": ("add_to_cart", "cart_add"),
        "purchase_intent_event_rate": ("purchase_intent", "intent_to_purchase"),
        "purchase_event_rate": ("purchase", "completed_purchase", "checkout_complete"),
    }
    survey_names = (
        "recommendation_relevance",
        "recommendation_completeness",
        "recommendation_accuracy",
        "source_credibility",
        "trust",
        "usefulness",
        "perceived_risk",
        "satisfaction",
        "purchase_intent",
        "continued_use_intent",
    )
    output: dict[str, Any] = {}
    for method, predictions in by_method.items():
        event_outcomes: dict[str, list[int]] = {key: [] for key in event_groups}
        survey_values: dict[str, list[float]] = defaultdict(list)
        steps: list[int] = []
        for case_id, prediction in predictions.items():
            if case_id not in cases:
                continue
            events = _trace_events(prediction)
            types = {_event_type(event) for event in events}
            for metric, accepted in event_groups.items():
                event_outcomes[metric].append(int(any(name in types for name in accepted)))
            steps.append(len(events))
            metadata = _metadata(prediction)
            survey = metadata.get("survey_scores") or metadata.get("survey") or {}
            if isinstance(survey, Mapping):
                for name in survey_names:
                    value = as_float(survey.get(name))
                    if value is not None:
                        survey_values[name].append(value)
            # Permit a single logged purchase-intention score in a compact
            # benchmark export without inventing the rest of a survey.
            direct_intent = as_float(metadata.get("purchase_intent"))
            if direct_intent is not None and not (isinstance(survey, Mapping) and "purchase_intent" in survey):
                survey_values["purchase_intent"].append(direct_intent)
        output[method] = {
            "sessions": len(steps),
            **{metric: mean_or_none(values) for metric, values in event_outcomes.items()},
            "mean_logged_events": mean_or_none(steps),
            "survey_score_means": {name: mean_or_none(survey_values[name]) for name in survey_names},
        }
    return output


def _integrity_evaluation(by_method: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for method, predictions in by_method.items():
        claim_rows: list[dict[str, float | int | None]] = []
        flag_counts: Counter[str] = Counter()
        reviewed = 0
        for prediction in predictions.values():
            claims = _claims(prediction)
            if claims:
                claim_rows.append(claim_support_metrics(claims))
                reviewed += 1
            metadata = _metadata(prediction)
            flags = _read(prediction, "integrity_flags", "safety_flags", default=None)
            if flags is None:
                flags = metadata.get("integrity_flags") or metadata.get("safety_flags") or []
            for flag in flags if isinstance(flags, Sequence) and not isinstance(flags, (str, bytes)) else []:
                flag_counts[str(flag)] += 1
        output[method] = {
            "claim_audited_cases": reviewed,
            "claim_metrics": {
                key: mean_or_none([as_float(row.get(key)) for row in claim_rows])
                for key in ("citation_precision", "citation_recall", "unsupported_claim_rate")
            },
            "integrity_flag_counts": dict(sorted(flag_counts.items())),
        }
    return output


def _manipulation_audit(
    by_method: Mapping[str, Mapping[str, Any]],
    rank_rows: Mapping[str, Mapping[str, Mapping[str, Any]]],
    baseline_method: str | None,
) -> dict[str, Any]:
    """Summarise supplied defensive labels and paired rank changes.

    This is intentionally an audit-only reader: it reports clean-versus-tested
    outputs that researchers supply, but contains no prompt/attack generator.
    """

    output: dict[str, Any] = {}
    for method, predictions in by_method.items():
        measures: dict[str, list[float]] = defaultdict(list)
        for prediction in predictions.values():
            metadata = _metadata(prediction)
            audit = _read(prediction, "manipulation_audit", "security_audit", default=None)
            if not isinstance(audit, Mapping):
                audit = metadata.get("manipulation_audit") or metadata.get("security_audit") or {}
            for key in ("attack_success", "stealth", "utility", "factuality", "detection_rate"):
                value = as_float(audit.get(key)) if isinstance(audit, Mapping) else None
                if value is not None:
                    measures[key].append(value)
        summary: dict[str, Any] = {key: mean_or_none(values) for key, values in sorted(measures.items())}
        baseline_rows = rank_rows.get(baseline_method or "", {})
        current_rows = rank_rows.get(method, {})
        normalized_gains: list[float] = []
        success_at: dict[float, list[int]] = {0.1: [], 0.3: []}
        promote_at: dict[float, list[int]] = {0.1: [], 0.3: []}
        if baseline_method and method != baseline_method:
            for case_id in set(baseline_rows) & set(current_rows):
                before = as_float(baseline_rows[case_id].get("target_rank"))
                after = as_float(current_rows[case_id].get("target_rank"))
                candidate_count = as_float(current_rows[case_id].get("candidate_count"))
                if before is None or after is None or candidate_count is None or candidate_count <= 1:
                    continue
                normalized_gains.append((before - after) / (candidate_count - 1))
                for alpha, values_for_alpha in success_at.items():
                    cutoff = max(1, ceil(candidate_count * alpha))
                    values_for_alpha.append(int(after <= cutoff))
                    promote_at[alpha].append(int(before > cutoff and after <= cutoff))
        if normalized_gains:
            summary["paired_rank_audit"] = {
                "matched_cases": len(normalized_gains),
                "normalized_rank_gain": mean_or_none(normalized_gains),
                "success_at": {str(alpha): mean_or_none(values) for alpha, values in success_at.items()},
                "promote_at": {str(alpha): mean_or_none(values) for alpha, values in promote_at.items()},
            }
        output[method] = summary
    return output


class BenchmarkEvaluator:
    """Evaluate normalized predictions against a named GEO benchmark profile."""

    def evaluate(
        self,
        benchmark_id: str,
        cases: Iterable[Any],
        predictions: Iterable[Any],
        *,
        baseline_method: str | None = None,
    ) -> dict[str, Any]:
        cases_by_id = _case_lookup(cases)
        predictions_by_method = _prediction_lookup(predictions)
        try:
            benchmark_key = normalize_benchmark_id(str(benchmark_id))
        except KeyError:
            # Custom study profiles remain useful, but registry-backed names
            # (including aliases such as "E-GEO") resolve to their canonical
            # protocol before selecting metrics.
            benchmark_key = str(benchmark_id).strip().lower().replace("-", "_").replace(" ", "_")
        profiles = PROFILE_BY_BENCHMARK.get(benchmark_key, ("ranking", "visibility", "integrity"))
        rank_rows = {method: _rank_rows(cases_by_id, items) for method, items in predictions_by_method.items()}
        visibility_rows = {method: _visibility_rows(cases_by_id, items) for method, items in predictions_by_method.items()}
        report: dict[str, Any] = {
            "schema_version": "geo-benchmark-report/v1",
            "benchmark": benchmark_key,
            "profiles": list(profiles),
            "cases_loaded": len(cases_by_id),
            "methods": sorted(predictions_by_method),
            "baseline_method": baseline_method,
            "validation": {
                "predictions_loaded": sum(len(items) for items in predictions_by_method.values()),
                "orphan_predictions": sorted(
                    case_id
                    for items in predictions_by_method.values()
                    for case_id in items
                    if case_id not in cases_by_id
                ),
            },
        }
        if "ranking" in profiles:
            report["ranking"] = {method: _rank_summary(rows) for method, rows in rank_rows.items()}
        if "visibility" in profiles or "utility" in profiles:
            report["visibility"] = {method: _visibility_summary(rows) for method, rows in visibility_rows.items()}
        if "stage" in profiles:
            report["stages"] = _stage_evaluation(cases_by_id, predictions_by_method)
        if "trajectory" in profiles:
            report["trajectory"] = _trajectory_evaluation(cases_by_id, predictions_by_method)
        if "agent_choice" in profiles:
            report["agent_choice"] = _agent_choice_evaluation(cases_by_id, predictions_by_method)
        if "multi_query" in profiles:
            report["multi_query_stability"] = _multi_query_stability(cases_by_id, rank_rows, visibility_rows, baseline_method)
        if "absorption" in profiles:
            report["citation_absorption"] = _absorption_evaluation(cases_by_id, predictions_by_method)
        if "discovery" in profiles:
            report["discovery"] = _discovery_evaluation(cases_by_id, predictions_by_method)
        if "coverage" in profiles:
            report["coverage"] = _coverage_evaluation(cases_by_id, predictions_by_method)
        if "external_monitor" in profiles:
            report["external_monitor"] = _external_monitor_evaluation(cases_by_id, predictions_by_method)
        if "behavior" in profiles:
            report["consumer_behavior"] = _behavior_evaluation(cases_by_id, predictions_by_method)
        if "integrity" in profiles:
            report["integrity"] = _integrity_evaluation(predictions_by_method)
        if "manipulation_audit" in profiles:
            report["manipulation_audit"] = _manipulation_audit(
                predictions_by_method,
                rank_rows,
                baseline_method,
            )
        report["paired_comparisons"] = _method_comparisons(rank_rows, visibility_rows, baseline_method)
        return report
