"""Experiment assignment, evaluation, and descriptive metric utilities.

These functions deliberately return transparent descriptive statistics and
clearly label approximate confidence intervals.  They are useful for a
preregistered pilot/dashboard, but do not replace a planned mixed-effects or
paired analysis for a full DBA study.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import math
from typing import Any

from .text import normalize_whitespace, safe_float, to_mapping


CONTROL = "CONTROL"
GEO_OPTIMIZED = "GEO_OPTIMIZED"
CONDITIONS = (CONTROL, GEO_OPTIMIZED)

CONDITION_ALIASES = {
    "CONTROL": CONTROL,
    "BASELINE": CONTROL,
    "UNOPTIMIZED": CONTROL,
    "GEO": GEO_OPTIMIZED,
    "TREATMENT": GEO_OPTIMIZED,
    "OPTIMIZED": GEO_OPTIMIZED,
    "GEO_OPTIMIZED": GEO_OPTIMIZED,
}

DEFAULT_SCALE_MAP: dict[str, tuple[str, ...]] = {
    "recommendation_quality": ("rq1", "rq2", "rq3"),
    "citation_credibility": ("sc1", "sc2", "sc3"),
    "trust": ("tr1", "tr2", "tr3"),
    "usefulness": ("pu1", "pu2", "pu3"),
    "risk": ("pr1", "pr2", "pr3"),
    "purchase_intention": ("pi1", "pi2", "pi3"),
}


def normalize_condition(value: Any) -> str | None:
    cleaned = normalize_whitespace(value).upper().replace(" ", "_").replace("-", "_")
    return CONDITION_ALIASES.get(cleaned)


def _stable_parity(value: str) -> int:
    # Avoid Python's randomised hash() so assignments are reproducible across
    # processes/machines.  This is a deterministic allocation, not participant
    # randomisation; record the scheme/seed in a preregistration.
    return sum(ord(character) for character in value) % 2


@dataclass(frozen=True)
class AssignmentResult:
    products: list[dict[str, Any]]
    report: dict[str, Any]


def assign_balanced_conditions(
    products: Iterable[Any],
    *,
    seed: str = "geo-study-v1",
    respect_existing: bool = True,
) -> AssignmentResult:
    """Assign product records 1:1 within paired groups where possible.

    Existing valid assignments are retained by default.  Records sharing a
    ``pair_id`` are alternated first; unpaired records are alternated within
    category.  The function returns copies and never changes caller-owned
    records in place.
    """

    copied = [dict(to_mapping(product)) for product in products]
    paired: dict[str, list[dict[str, Any]]] = defaultdict(list)
    unpaired: dict[str, list[dict[str, Any]]] = defaultdict(list)
    invalid_existing: list[str] = []
    for product in copied:
        raw_condition = product.get("condition")
        normalized = normalize_condition(raw_condition)
        if raw_condition not in (None, "") and normalized is None:
            invalid_existing.append(str(product.get("id") or product.get("title") or "unknown"))
        if respect_existing and normalized:
            product["condition"] = normalized
            continue
        if product.get("pair_id"):
            paired[normalize_whitespace(product.get("pair_id"))].append(product)
        else:
            unpaired[normalize_whitespace(product.get("category")) or "Uncategorised"].append(product)

    assignment_groups: list[dict[str, Any]] = []
    for group_type, groups in (("pair", paired), ("category", unpaired)):
        for group_key, members in sorted(groups.items()):
            sorted_members = sorted(
                members,
                key=lambda item: (
                    normalize_whitespace(item.get("title")).lower(),
                    normalize_whitespace(item.get("id")).lower(),
                ),
            )
            parity = _stable_parity(f"{seed}:{group_type}:{group_key}")
            for index, product in enumerate(sorted_members):
                product["condition"] = CONDITIONS[(parity + index) % 2]
            counts = _condition_counts(sorted_members)
            assignment_groups.append(
                {
                    "group_type": group_type,
                    "group_key": group_key,
                    "n": len(sorted_members),
                    "control": counts[CONTROL],
                    "geo_optimized": counts[GEO_OPTIMIZED],
                }
            )

    counts = _condition_counts(copied)
    report = {
        "assignment_scheme": "deterministic alternation by pair_id, then category",
        "seed": seed,
        "respect_existing": respect_existing,
        "assigned_products": len(copied),
        "control": counts[CONTROL],
        "geo_optimized": counts[GEO_OPTIMIZED],
        "imbalance": abs(counts[CONTROL] - counts[GEO_OPTIMIZED]),
        "invalid_existing_conditions": invalid_existing,
        "groups": assignment_groups,
        "warning": (
            "This allocation is reproducible but is not a substitute for a preregistered randomized exposure scheme. "
            "Use same-SKU counterfactual page variants where feasible."
        ),
    }
    return AssignmentResult(products=copied, report=report)


def _condition_counts(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = {CONTROL: 0, GEO_OPTIMIZED: 0}
    for row in rows:
        condition = normalize_condition(row.get("condition"))
        if condition in counts:
            counts[condition] += 1
    return counts


def catalog_balance(products: Iterable[Any]) -> dict[str, Any]:
    """Summarise observed covariate balance without conducting a significance test."""

    arms: dict[str, list[dict[str, Any]]] = {CONTROL: [], GEO_OPTIMIZED: []}
    unknown = 0
    for raw in products:
        product = to_mapping(raw)
        condition = normalize_condition(product.get("condition"))
        if condition in arms:
            arms[condition].append(product)
        else:
            unknown += 1
    by_category: dict[str, dict[str, int]] = defaultdict(lambda: {CONTROL: 0, GEO_OPTIMIZED: 0})
    for condition, records in arms.items():
        for product in records:
            by_category[normalize_whitespace(product.get("category")) or "Uncategorised"][condition] += 1

    def numeric_summary(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
        values = [safe_float(record.get(field)) for record in records]
        observed = [value for value in values if value is not None]
        return {
            "n_observed": len(observed),
            "n_missing": len(records) - len(observed),
            "mean": round(sum(observed) / len(observed), 4) if observed else None,
            "min": min(observed) if observed else None,
            "max": max(observed) if observed else None,
        }

    return {
        "n_products": sum(len(records) for records in arms.values()),
        "n_unknown_condition": unknown,
        "by_condition": {
            condition: {
                "n": len(records),
                "price": numeric_summary(records, "price"),
                "rating": numeric_summary(records, "rating"),
                "review_count": numeric_summary(records, "review_count"),
            }
            for condition, records in arms.items()
        },
        "by_category": [
            {"category": category, CONTROL: counts[CONTROL], GEO_OPTIMIZED: counts[GEO_OPTIMIZED]}
            for category, counts in sorted(by_category.items())
        ],
        "interpretation": (
            "Check category, price, rating, reviews, brand familiarity, and matched-pair coverage before recruitment. "
            "Observed balance does not prove exchangeability."
        ),
    }


def scale_scores(
    answers: Mapping[str, Any],
    *,
    scale_map: Mapping[str, Iterable[str]] = DEFAULT_SCALE_MAP,
    lower: float = 1.0,
    upper: float = 7.0,
) -> dict[str, float | None]:
    """Compute transparent arithmetic means for completed Likert constructs."""

    result: dict[str, float | None] = {}
    for construct, items in scale_map.items():
        values = [safe_float(answers.get(item)) for item in items]
        valid = [value for value in values if value is not None and lower <= value <= upper]
        result[str(construct)] = round(sum(valid) / len(valid), 3) if valid else None
    return result


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return normalize_whitespace(value).lower() in {"1", "true", "yes", "y", "cited", "retrieved"}


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _mean(values: Iterable[float | None]) -> float | None:
    observed = [value for value in values if value is not None]
    return round(sum(observed) / len(observed), 6) if observed else None


def _arm_metrics(rows: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    n = len(rows)
    cited = sum(_bool(row.get("cited")) for row in rows)
    retrieved = sum(_bool(row.get("retrieved")) for row in rows)
    top_k_count = sum(
        _bool(row.get("retrieved")) and (safe_float(row.get("rank_position")) or math.inf) <= top_k for row in rows
    )
    first_cited = sum(
        _bool(row.get("cited"))
        and (_bool(row.get("first_citation")) or (safe_float(row.get("citation_position")) or math.inf) == 1)
        for row in rows
    )
    entailment_values = [row.get("citation_entails") for row in rows if row.get("citation_entails") is not None]
    integrity_flags = sum(_bool(row.get("integrity_flag")) for row in rows)
    score_values = [safe_float(row.get("retrieval_score", row.get("score"))) for row in rows]
    evidence_values = [safe_float(row.get("evidence_score")) for row in rows]
    return {
        "candidate_opportunities": n,
        "citation_events": cited,
        "citation_rate": _rate(cited, n),
        "retrieved_events": retrieved,
        "retrieval_rate": _rate(retrieved, n),
        "citation_given_retrieved": _rate(cited, retrieved),
        f"retrieved_at_{top_k}": top_k_count,
        f"retrieved_at_{top_k}_rate": _rate(top_k_count, n),
        "first_citation_events": first_cited,
        "first_citation_rate": _rate(first_cited, n),
        "mean_retrieval_score": _mean(score_values),
        "mean_evidence_score": _mean(evidence_values),
        "citation_entailment_rate": _rate(sum(_bool(value) for value in entailment_values), len(entailment_values)),
        "citation_entailment_n": len(entailment_values),
        "integrity_flags": integrity_flags,
        "integrity_flag_rate": _rate(integrity_flags, n),
    }


def risk_effect(
    treatment_events: int,
    treatment_total: int,
    control_events: int,
    control_total: int,
) -> dict[str, Any]:
    """Return an interpretable treatment-control risk contrast with caveats."""

    if not treatment_total or not control_total:
        return {
            "estimable": False,
            "reason": "Both treatment and control need one or more candidate opportunities.",
        }
    p_t = treatment_events / treatment_total
    p_c = control_events / control_total
    difference = p_t - p_c
    se = math.sqrt((p_t * (1 - p_t) / treatment_total) + (p_c * (1 - p_c) / control_total))
    ci = (difference - 1.96 * se, difference + 1.96 * se)
    result: dict[str, Any] = {
        "estimable": True,
        "treatment_rate": round(p_t, 6),
        "control_rate": round(p_c, 6),
        "risk_difference": round(difference, 6),
        "risk_difference_95ci_wald": [round(ci[0], 6), round(ci[1], 6)],
        "risk_ratio": round(p_t / p_c, 6) if p_c else None,
        "interpretation": "Positive risk difference favours GEO_OPTIMIZED on this descriptive outcome.",
        "caution": (
            "The Wald interval assumes independent opportunities and can be poor for sparse outcomes. "
            "Use preregistered pair/query/participant clustered or mixed-effects analysis for confirmatory inference."
        ),
    }
    # Haldane-Anscombe correction makes an odds ratio displayable when a cell is
    # zero; the raw rates above remain the primary transparent quantities.
    a, b = treatment_events + 0.5, treatment_total - treatment_events + 0.5
    c, d = control_events + 0.5, control_total - control_events + 0.5
    result["odds_ratio_haldane_anscombe"] = round((a * d) / (b * c), 6)
    return result


def evaluate_candidates(
    candidate_rows: Iterable[Any],
    *,
    top_k: int = 3,
) -> dict[str, Any]:
    """Evaluate candidate-level retrieval/citation data by experimental arm.

    Each row should represent one product-query-run opportunity and include at
    least ``condition`` and preferably ``retrieved``, ``rank_position``, and
    ``cited``.  Rows with unknown conditions are retained in quality counts but
    never silently allocated to an arm.
    """

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    arms: dict[str, list[dict[str, Any]]] = {CONTROL: [], GEO_OPTIMIZED: []}
    unknown: list[dict[str, Any]] = []
    for raw in candidate_rows:
        row = to_mapping(raw)
        condition = normalize_condition(row.get("condition"))
        if condition in arms:
            arms[condition].append(row)
        else:
            unknown.append(row)
    metrics = {condition: _arm_metrics(rows, top_k=top_k) for condition, rows in arms.items()}
    effect = risk_effect(
        metrics[GEO_OPTIMIZED]["citation_events"],
        metrics[GEO_OPTIMIZED]["candidate_opportunities"],
        metrics[CONTROL]["citation_events"],
        metrics[CONTROL]["candidate_opportunities"],
    )
    return {
        "outcome": "citation",
        "denominator": "all logged product-query-run candidate opportunities",
        "top_k": top_k,
        "by_condition": metrics,
        "citation_effect_geo_minus_control": effect,
        "data_quality": {
            "rows_received": sum(len(rows) for rows in arms.values()) + len(unknown),
            "unknown_condition_rows": len(unknown),
            "missing_retrieval_flags": sum("retrieved" not in row for rows in arms.values() for row in rows),
            "missing_citation_flags": sum("cited" not in row for rows in arms.values() for row in rows),
        },
        "analysis_note": (
            "Report end-to-end citation, retrieval, and citation-given-retrieval separately. "
            "Do not pool controlled probes with participant behaviour or external-engine replication data."
        ),
    }


def consumer_event_summary(events: Iterable[Any]) -> dict[str, Any]:
    """Describe logged participant interactions without inferring causal trust."""

    by_condition: dict[str, dict[str, int]] = {CONTROL: defaultdict(int), GEO_OPTIMIZED: defaultdict(int)}
    unknown = 0
    for raw in events:
        row = to_mapping(raw)
        condition = normalize_condition(row.get("condition"))
        event_type = normalize_whitespace(row.get("event_type")) or "unknown"
        if condition in by_condition:
            by_condition[condition][event_type] += 1
        else:
            unknown += 1
    return {
        "by_condition": {condition: dict(sorted(counts.items())) for condition, counts in by_condition.items()},
        "unknown_condition_events": unknown,
        "interpretation": "Engagement events are descriptive behavioural indicators; pair them with preregistered survey and task outcomes.",
    }
