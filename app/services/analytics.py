"""Privacy-conscious analytics for the researcher dashboard.

The study tables retain detailed, auditable event data for approved analysis.
This module deliberately projects only consented observations into the live
researcher dashboard: participant codes, countries, raw query text, raw survey
answers, and event metadata never leave this reporting boundary.

All SQL in this module uses SQLAlchemy core expressions supported by both the
SQLite development database and PostgreSQL study deployment.  Calendar
timeline bucketing and JSON score summarisation are done in Python because
database-specific JSON/date functions would make the report non-portable.
"""

from __future__ import annotations

import csv
import hashlib
import io
import math
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import case, func, select
from sqlalchemy.orm import Session as DBSession

from app.core.config import CONDITIONS
from app.core.experiment import SCALE_MAP, utc_now
from app.models import Event, Product, Query, QueryCandidate, Session, SurveyResponse


# These event types have an interpretable relationship to a product-condition
# outcome.  ``survey_open`` is deliberately excluded from engagement so the
# funnel does not mistake questionnaire navigation for product engagement.
ENGAGEMENT_EVENT_TYPES = ("citation_open", "product_open", "comparison_add", "purchase_intent")
CONSTRUCTS = tuple(SCALE_MAP)
RAW_FIELDS_EXCLUDED = (
    "participant_code",
    "country",
    "ai_familiarity",
    "raw_query_text",
    "raw_survey_answers",
    "event_metadata",
    "session_id",
)


def _safe_number(value: Any, *, minimum: float = 1.0, maximum: float = 7.0) -> float | None:
    """Return a finite in-range questionnaire score, otherwise ``None``."""

    if isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numeric) or numeric < minimum or numeric > maximum:
        return None
    return numeric


def _number(value: Any, *, digits: int = 3) -> float:
    return round(float(value or 0), digits)


def _rate(numerator: int, denominator: int, *, digits: int = 4) -> float | None:
    return round(numerator / denominator, digits) if denominator else None


def _utc_iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc).isoformat()
    return value.astimezone(timezone.utc).isoformat()


def _utc_day(value: datetime | None) -> str | None:
    timestamp = _utc_iso(value)
    return timestamp[:10] if timestamp else None


def _respondent_id(session_id: str) -> str:
    """Create a stable dashboard-only pseudonym from an internal session ID."""

    digest = hashlib.sha256(f"geo-analytics-v1:{session_id}".encode("utf-8")).hexdigest()[:12].upper()
    return f"R-{digest}"


def _condition_order(values: set[str]) -> list[str]:
    # Keep the experiment's two planned arms visible even before the first
    # participant interaction, then retain any legacy/imported arm labels.
    known = list(CONDITIONS)
    extras = sorted(value for value in values if value not in CONDITIONS)
    return known + extras


def _survey_summary(db: DBSession) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    values_by_construct: dict[str, list[float]] = {construct: [] for construct in CONSTRUCTS}
    rows = db.scalars(
        select(SurveyResponse.scale_scores_json)
        .select_from(SurveyResponse)
        .join(Session, SurveyResponse.session_id == Session.id)
        .where(Session.consent.is_(True))
    ).all()
    for score_json in rows:
        if not isinstance(score_json, dict):
            continue
        for construct in CONSTRUCTS:
            score = _safe_number(score_json.get(construct))
            if score is not None:
                values_by_construct[construct].append(score)

    constructs: list[dict[str, Any]] = []
    for construct in CONSTRUCTS:
        values = values_by_construct[construct]
        bins = {score: 0 for score in range(1, 8)}
        for value in values:
            # Construct scores can be item means (for example, 5.667).  A
            # nearest-integer bin supports compact dashboard histograms while
            # the precise mean/min/max remain available alongside it.
            bins[min(7, max(1, int(math.floor(value + 0.5))))] += 1
        constructs.append(
            {
                "construct": construct,
                "n": len(values),
                "mean": round(sum(values) / len(values), 3) if values else None,
                "minimum": round(min(values), 3) if values else None,
                "maximum": round(max(values), 3) if values else None,
                "distribution": [{"score": score, "count": bins[score]} for score in range(1, 8)],
            }
        )
    return {
        "score_range": {"minimum": 1, "maximum": 7},
        "distribution_basis": "Nearest-integer bins of construct means; means retain their original precision.",
        "constructs": constructs,
    }, constructs


def _condition_metrics(db: DBSession) -> list[dict[str, Any]]:
    candidate_rows = db.execute(
        select(
            QueryCandidate.condition,
            func.count().label("candidate_opportunities"),
            func.coalesce(func.sum(case((QueryCandidate.cited.is_(True), 1.0), else_=0.0)), 0.0).label("citation_events"),
            func.coalesce(
                func.sum(case((QueryCandidate.rank_position <= 3, 1.0), else_=0.0)), 0.0
            ).label("top3_events"),
            func.coalesce(
                func.sum(
                    case(
                        ((QueryCandidate.cited.is_(True)) & (QueryCandidate.rank_position <= 3), 1.0),
                        else_=0.0,
                    )
                ),
                0.0,
            ).label("top3_citation_events"),
            func.avg(QueryCandidate.retrieval_score).label("mean_retrieval_score"),
            func.avg(QueryCandidate.evidence_score).label("mean_evidence_score"),
        )
        .select_from(QueryCandidate)
        .join(Query, QueryCandidate.query_id == Query.id)
        .join(Session, Query.session_id == Session.id)
        .where(Session.consent.is_(True))
        .group_by(QueryCandidate.condition)
    ).all()
    candidates = {str(row.condition): row for row in candidate_rows}

    event_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    event_rows = db.execute(
        select(Product.condition, Event.event_type, func.count().label("events"))
        .select_from(Event)
        .join(Session, Event.session_id == Session.id)
        .join(Product, Event.product_id == Product.id)
        .where(Session.consent.is_(True), Event.event_type.in_(ENGAGEMENT_EVENT_TYPES))
        .group_by(Product.condition, Event.event_type)
    ).all()
    for condition, event_type, events in event_rows:
        event_counts[str(condition)][str(event_type)] = int(events or 0)

    observed_conditions = set(candidates) | set(event_counts)
    metrics: list[dict[str, Any]] = []
    for condition in _condition_order(observed_conditions):
        row = candidates.get(condition)
        opportunities = int(row.candidate_opportunities) if row else 0
        cited = int(row.citation_events) if row else 0
        top3 = int(row.top3_events) if row else 0
        top3_cited = int(row.top3_citation_events) if row else 0
        counts = event_counts[condition]
        engagement = sum(counts[event_type] for event_type in ENGAGEMENT_EVENT_TYPES)
        metrics.append(
            {
                "condition": condition,
                "candidate_opportunities": opportunities,
                "citation_events": cited,
                "citation_rate": _rate(cited, opportunities),
                "top3_events": top3,
                "top3_rate": _rate(top3, opportunities),
                "top3_citation_events": top3_cited,
                "top3_citation_rate": _rate(top3_cited, opportunities),
                "mean_retrieval_score": _number(row.mean_retrieval_score) if row else 0.0,
                "mean_evidence_score": _number(row.mean_evidence_score) if row else 0.0,
                "citation_open_events": counts["citation_open"],
                "product_open_events": counts["product_open"],
                "comparison_events": counts["comparison_add"],
                "purchase_intent_events": counts["purchase_intent"],
                "engagement_events": engagement,
                "engagement_per_citation": _rate(engagement, cited),
            }
        )
    return metrics


def _category_effects(db: DBSession) -> list[dict[str, Any]]:
    candidate_rows = db.execute(
        select(
            Product.category,
            QueryCandidate.condition,
            func.count().label("candidate_opportunities"),
            func.coalesce(func.sum(case((QueryCandidate.cited.is_(True), 1.0), else_=0.0)), 0.0).label("citation_events"),
            func.avg(QueryCandidate.retrieval_score).label("mean_retrieval_score"),
            func.avg(QueryCandidate.evidence_score).label("mean_evidence_score"),
        )
        .select_from(QueryCandidate)
        .join(Query, QueryCandidate.query_id == Query.id)
        .join(Session, Query.session_id == Session.id)
        .join(Product, QueryCandidate.product_id == Product.id)
        .where(Session.consent.is_(True))
        .group_by(Product.category, QueryCandidate.condition)
    ).all()

    event_rows = db.execute(
        select(Product.category, Product.condition, func.count().label("engagement_events"))
        .select_from(Event)
        .join(Session, Event.session_id == Session.id)
        .join(Product, Event.product_id == Product.id)
        .where(Session.consent.is_(True), Event.event_type.in_(ENGAGEMENT_EVENT_TYPES))
        .group_by(Product.category, Product.condition)
    ).all()
    engagement_by_key = {
        ((category or "Uncategorised"), str(condition)): int(events or 0)
        for category, condition, events in event_rows
    }

    output: list[dict[str, Any]] = []
    for row in candidate_rows:
        category = row.category or "Uncategorised"
        condition = str(row.condition)
        opportunities = int(row.candidate_opportunities or 0)
        cited = int(row.citation_events or 0)
        engagement = engagement_by_key.get((category, condition), 0)
        output.append(
            {
                "category": category,
                "condition": condition,
                "candidate_opportunities": opportunities,
                "citation_events": cited,
                "citation_rate": _rate(cited, opportunities),
                "engagement_events": engagement,
                "engagement_per_citation": _rate(engagement, cited),
                "mean_retrieval_score": _number(row.mean_retrieval_score),
                "mean_evidence_score": _number(row.mean_evidence_score),
            }
        )

    # A product can receive an interaction even if the corresponding category
    # has no currently retained candidate opportunity. Keep that outcome visible
    # rather than silently dropping it from the dashboard.
    existing_keys = {(row["category"], row["condition"]) for row in output}
    for (category, condition), engagement in engagement_by_key.items():
        if (category, condition) not in existing_keys:
            output.append(
                {
                    "category": category,
                    "condition": condition,
                    "candidate_opportunities": 0,
                    "citation_events": 0,
                    "citation_rate": None,
                    "engagement_events": engagement,
                    "engagement_per_citation": None,
                    "mean_retrieval_score": 0.0,
                    "mean_evidence_score": 0.0,
                }
            )
    return sorted(output, key=lambda item: (item["category"].lower(), item["condition"]))


def _timeline(db: DBSession) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}

    def bucket_for(timestamp: datetime | None) -> dict[str, Any] | None:
        day = _utc_day(timestamp)
        if day is None:
            return None
        return buckets.setdefault(
            day,
            {"date": day, "queries": 0, "events": 0, "surveys_completed": 0, "_sessions": set()},
        )

    query_rows = db.execute(
        select(Query.created_at, Query.session_id)
        .select_from(Query)
        .join(Session, Query.session_id == Session.id)
        .where(Session.consent.is_(True))
    ).all()
    for created_at, session_id in query_rows:
        bucket = bucket_for(created_at)
        if bucket is not None:
            bucket["queries"] += 1
            bucket["_sessions"].add(session_id)

    event_rows = db.execute(
        select(Event.created_at, Event.session_id)
        .select_from(Event)
        .join(Session, Event.session_id == Session.id)
        .where(Session.consent.is_(True))
    ).all()
    for created_at, session_id in event_rows:
        bucket = bucket_for(created_at)
        if bucket is not None:
            bucket["events"] += 1
            bucket["_sessions"].add(session_id)

    survey_rows = db.execute(
        select(SurveyResponse.completed_at, SurveyResponse.session_id)
        .select_from(SurveyResponse)
        .join(Session, SurveyResponse.session_id == Session.id)
        .where(Session.consent.is_(True))
    ).all()
    for completed_at, session_id in survey_rows:
        bucket = bucket_for(completed_at)
        if bucket is not None:
            bucket["surveys_completed"] += 1
            bucket["_sessions"].add(session_id)

    return [
        {
            "date": bucket["date"],
            "queries": bucket["queries"],
            "events": bucket["events"],
            "surveys_completed": bucket["surveys_completed"],
            "active_sessions": len(bucket["_sessions"]),
        }
        for _, bucket in sorted(buckets.items())
    ]


def _respondent_rows(
    db: DBSession,
    *,
    limit: int,
    offset: int,
) -> dict[str, Any]:
    total = int(db.scalar(select(func.count()).select_from(Session).where(Session.consent.is_(True))) or 0)
    sessions = db.scalars(
        select(Session)
        .where(Session.consent.is_(True))
        .order_by(Session.started_at.desc(), Session.id)
        .offset(offset)
        .limit(limit)
    ).all()
    session_ids = [record.id for record in sessions]
    if not session_ids:
        return {"items": [], "total": total, "limit": limit, "offset": offset, "has_more": False}

    surveys: dict[str, tuple[dict[str, Any], datetime | None]] = {}
    for session_id, scores, completed_at in db.execute(
        select(SurveyResponse.session_id, SurveyResponse.scale_scores_json, SurveyResponse.completed_at).where(
            SurveyResponse.session_id.in_(session_ids)
        )
    ).all():
        surveys[str(session_id)] = (scores if isinstance(scores, dict) else {}, completed_at)

    query_counts = {
        str(session_id): int(count or 0)
        for session_id, count in db.execute(
            select(Query.session_id, func.count().label("queries"))
            .where(Query.session_id.in_(session_ids))
            .group_by(Query.session_id)
        ).all()
    }
    event_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for session_id, event_type, count in db.execute(
        select(Event.session_id, Event.event_type, func.count().label("events"))
        .where(Event.session_id.in_(session_ids))
        .group_by(Event.session_id, Event.event_type)
    ).all():
        event_counts[str(session_id)][str(event_type)] = int(count or 0)

    candidate_counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"candidate_opportunities": 0, "citation_events": 0})
    )
    for session_id, condition, opportunities, cited in db.execute(
        select(
            Query.session_id,
            QueryCandidate.condition,
            func.count().label("candidate_opportunities"),
            func.coalesce(func.sum(case((QueryCandidate.cited.is_(True), 1.0), else_=0.0)), 0.0).label("citation_events"),
        )
        .select_from(QueryCandidate)
        .join(Query, QueryCandidate.query_id == Query.id)
        .where(Query.session_id.in_(session_ids))
        .group_by(Query.session_id, QueryCandidate.condition)
    ).all():
        candidate_counts[str(session_id)][str(condition)] = {
            "candidate_opportunities": int(opportunities or 0),
            "citation_events": int(cited or 0),
        }

    engagement_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for session_id, condition, count in db.execute(
        select(Event.session_id, Product.condition, func.count().label("engagement_events"))
        .select_from(Event)
        .join(Product, Event.product_id == Product.id)
        .where(Event.session_id.in_(session_ids), Event.event_type.in_(ENGAGEMENT_EVENT_TYPES))
        .group_by(Event.session_id, Product.condition)
    ).all():
        engagement_counts[str(session_id)][str(condition)] = int(count or 0)

    items: list[dict[str, Any]] = []
    for session in sessions:
        session_id = session.id
        raw_scores, survey_completed_at = surveys.get(session_id, ({}, None))
        scores = {construct: _safe_number(raw_scores.get(construct)) for construct in CONSTRUCTS}
        by_condition: dict[str, dict[str, Any]] = {}
        conditions = set(CONDITIONS) | set(candidate_counts[session_id]) | set(engagement_counts[session_id])
        for condition in _condition_order(conditions):
            outcome = candidate_counts[session_id][condition]
            cited = outcome["citation_events"]
            opportunities = outcome["candidate_opportunities"]
            engagement = engagement_counts[session_id][condition]
            by_condition[condition] = {
                "candidate_opportunities": opportunities,
                "citation_events": cited,
                "citation_rate": _rate(cited, opportunities),
                "engagement_events": engagement,
                "engagement_per_citation": _rate(engagement, cited),
            }
        counts = event_counts[session_id]
        items.append(
            {
                "respondent_id": _respondent_id(session_id),
                "study_cohort": session.study_cohort or "main",
                # Calendar dates rather than exact login times reduce the
                # chance of matching dashboard rows to external activity.
                "started_at": _utc_day(session.started_at),
                "completed_at": _utc_day(survey_completed_at or session.completed_at),
                "query_count": query_counts.get(session_id, 0),
                "event_count": sum(counts.values()),
                "product_open_events": counts["product_open"],
                "citation_open_events": counts["citation_open"],
                "comparison_events": counts["comparison_add"],
                "purchase_intent_events": counts["purchase_intent"],
                "survey_completed": session_id in surveys,
                "survey_scores": scores,
                "conditions": by_condition,
            }
        )
    return {
        "items": items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + len(items) < total,
    }


def build_analytics_report(
    db: DBSession,
    *,
    respondent_limit: int = 200,
    respondent_offset: int = 0,
) -> dict[str, Any]:
    """Build the researcher-safe, consented analytics report.

    This returns plain JSON-safe values so it can be returned directly by
    FastAPI or used to make a participant-safe metrics download.
    """

    consented_sessions = int(
        db.scalar(select(func.count()).select_from(Session).where(Session.consent.is_(True))) or 0
    )
    completed_surveys = int(
        db.scalar(
            select(func.count())
            .select_from(SurveyResponse)
            .join(Session, SurveyResponse.session_id == Session.id)
            .where(Session.consent.is_(True))
        )
        or 0
    )
    queries = int(
        db.scalar(
            select(func.count())
            .select_from(Query)
            .join(Session, Query.session_id == Session.id)
            .where(Session.consent.is_(True))
        )
        or 0
    )
    sessions_with_queries = int(
        db.scalar(
            select(func.count(func.distinct(Query.session_id)))
            .select_from(Query)
            .join(Session, Query.session_id == Session.id)
            .where(Session.consent.is_(True))
        )
        or 0
    )
    events = int(
        db.scalar(
            select(func.count())
            .select_from(Event)
            .join(Session, Event.session_id == Session.id)
            .where(Session.consent.is_(True))
        )
        or 0
    )
    sessions_with_engagement = int(
        db.scalar(
            select(func.count(func.distinct(Event.session_id)))
            .select_from(Event)
            .join(Session, Event.session_id == Session.id)
            .where(Session.consent.is_(True), Event.event_type.in_(ENGAGEMENT_EVENT_TYPES))
        )
        or 0
    )
    sessions_with_purchase_intent = int(
        db.scalar(
            select(func.count(func.distinct(Event.session_id)))
            .select_from(Event)
            .join(Session, Event.session_id == Session.id)
            .where(Session.consent.is_(True), Event.event_type == "purchase_intent")
        )
        or 0
    )

    survey, survey_distributions = _survey_summary(db)
    funnel_counts = (
        ("Consented sessions", consented_sessions),
        ("Searched with assistant", sessions_with_queries),
        ("Product engagement", sessions_with_engagement),
        ("Purchase intent", sessions_with_purchase_intent),
        ("Completed survey", completed_surveys),
    )
    return {
        "generated_at": _utc_iso(utc_now()),
        "privacy": {
            "consented_sessions_only": True,
            "respondent_rows_are_pseudonymous": True,
            "raw_fields_excluded": list(RAW_FIELDS_EXCLUDED),
        },
        "totals": {
            "consented_sessions": consented_sessions,
            "completed_surveys": completed_surveys,
            "survey_completion_rate": _rate(completed_surveys, consented_sessions),
            "queries": queries,
            "sessions_with_queries": sessions_with_queries,
            "events": events,
        },
        "survey": survey,
        # A top-level alias avoids UI consumers having to know the survey
        # container shape when drawing one chart per construct.
        "survey_distributions": survey_distributions,
        "timeline": _timeline(db),
        "funnel": [
            {"stage": stage, "count": count, "rate": _rate(count, consented_sessions)}
            for stage, count in funnel_counts
        ],
        "conditions": _condition_metrics(db),
        "category_effects": _category_effects(db),
        "respondents": _respondent_rows(db, limit=respondent_limit, offset=respondent_offset),
        "notes": [
            "All metrics exclude sessions without recorded active consent.",
            "Citation rate is cited product-query opportunities divided by all logged product-query candidates.",
            "Engagement is the total of citation opens, product opens, comparison adds, and purchase-intent events linked to a product condition.",
            "Use pseudonymous respondent rows for dashboard inspection; use approved governed exports for any raw-data analysis.",
        ],
    }


def analytics_report_csv(report: dict[str, Any], *, section: str = "overview") -> bytes:
    """Serialize a non-sensitive report section as a UTF-8 CSV download."""

    aliases = {"metrics": "overview", "categories": "category_effects", "respondent": "respondents"}
    section = aliases.get(section, section)
    valid_sections = {"overview", "conditions", "category_effects", "survey", "timeline", "respondents"}
    if section not in valid_sections:
        raise ValueError("Unknown analytics export section.")

    rows: list[dict[str, Any]]
    if section == "overview":
        rows = [{"metric": key, "value": value} for key, value in report["totals"].items()]
        rows.extend(
            {
                "metric": f"funnel:{item['stage']}",
                "value": item["count"],
                "rate": item["rate"],
            }
            for item in report["funnel"]
        )
    elif section == "conditions":
        rows = list(report["conditions"])
    elif section == "category_effects":
        rows = list(report["category_effects"])
    elif section == "timeline":
        rows = list(report["timeline"])
    elif section == "respondents":
        rows = []
        for item in report["respondents"]["items"]:
            row = {key: value for key, value in item.items() if key not in {"survey_scores", "conditions"}}
            row["survey_scores"] = item["survey_scores"]
            row["condition_outcomes"] = item["conditions"]
            rows.append(row)
    else:  # survey
        rows = []
        for construct in report["survey_distributions"]:
            for bucket in construct["distribution"]:
                rows.append(
                    {
                        "construct": construct["construct"],
                        "n": construct["n"],
                        "mean": construct["mean"],
                        "minimum": construct["minimum"],
                        "maximum": construct["maximum"],
                        "score": bucket["score"],
                        "count": bucket["count"],
                    }
                )

    columns = sorted({key for row in rows for key in row}) or ["metric", "value"]
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                key: (
                    ""
                    if value is None
                    else json_value(value)
                    if isinstance(value, (dict, list))
                    else value
                )
                for key, value in row.items()
            }
        )
    return stream.getvalue().encode("utf-8-sig")


def json_value(value: Any) -> str:
    """Keep nested CSV fields compact while preserving the download contract."""

    import json

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
