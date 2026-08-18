import csv
import io
import json
from collections import defaultdict
from typing import Any, Type

from fastapi import APIRouter, Depends, HTTPException, Query as QueryParam
from fastapi.responses import Response
from sqlalchemy import case, func, select
from sqlalchemy.orm import Session as DBSession

from app.core.config import CONDITIONS
from app.core.experiment import SCALE_MAP
from app.db.session import get_db
from app.models import (
    Event,
    GEOOptimizationApplication,
    GEOOptimizationConfig,
    ProbeCandidate,
    ProbeRun,
    Product,
    Query,
    QueryCandidate,
    Session,
    SurveyResponse,
)
from app.services.analytics import analytics_report_csv, build_analytics_report

router = APIRouter(tags=["GEO study"])

def _bad_request(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail=message)

def _dashboard(db: DBSession) -> dict[str, Any]:
    totals = {
        "products": int(db.scalar(select(func.count()).select_from(Product)) or 0),
        "sessions": int(db.scalar(select(func.count()).select_from(Session).where(Session.consent.is_(True))) or 0),
        "queries": int(db.scalar(select(func.count()).select_from(Query)) or 0),
        "completed_surveys": int(db.scalar(select(func.count()).select_from(SurveyResponse)) or 0),
    }
    rows = db.execute(
        select(
            QueryCandidate.condition,
            func.count().label("candidate_opportunities"),
            func.coalesce(func.sum(case((QueryCandidate.cited.is_(True), 1.0), else_=0.0)), 0.0).label("citation_events"),
            func.avg(QueryCandidate.retrieval_score).label("mean_retrieval_score"),
            func.avg(QueryCandidate.evidence_score).label("mean_evidence_score"),
            func.coalesce(func.sum(case((QueryCandidate.rank_position <= 3, 1.0), else_=0.0)), 0.0).label("top3_events"),
        )
        .group_by(QueryCandidate.condition)
        .order_by(QueryCandidate.condition)
    ).all()
    clicks = dict(
        db.execute(
            select(Product.condition, func.count().label("clicks"))
            .join(Event, Event.product_id == Product.id)
            .where(Event.event_type.in_(("citation_open", "product_open")))
            .group_by(Product.condition)
        ).all()
    )
    purchases = dict(
        db.execute(
            select(Product.condition, func.count().label("purchases"))
            .join(Event, Event.product_id == Product.id)
            .where(Event.event_type == "purchase_intent")
            .group_by(Product.condition)
        ).all()
    )
    metrics: list[dict[str, Any]] = []
    metrics_by_condition = {row.condition: row for row in rows}
    for condition in CONDITIONS:
        row = metrics_by_condition.get(condition)
        opportunities = int(row.candidate_opportunities) if row else 0
        cited = int(row.citation_events) if row else 0
        top3 = int(row.top3_events) if row else 0
        metrics.append(
            {
                "condition": condition,
                "candidate_opportunities": opportunities,
                "citation_events": cited,
                "citation_rate": round(cited / opportunities, 4) if opportunities else None,
                "top3_rate": round(top3 / opportunities, 4) if opportunities else None,
                "mean_retrieval_score": round(float(row.mean_retrieval_score or 0), 3) if row else 0.0,
                "mean_evidence_score": round(float(row.mean_evidence_score or 0), 3) if row else 0.0,
                "engagement_events": int(clicks.get(condition, 0)),
                "engagement_per_citation": round(int(clicks.get(condition, 0)) / cited, 4) if cited else None,
                "purchase_events": int(purchases.get(condition, 0)),
                "purchases_per_citation": round(int(purchases.get(condition, 0)) / cited, 4) if cited else None,
            }
        )
    constructs: dict[str, list[float]] = defaultdict(list)
    for response in db.scalars(select(SurveyResponse)).all():
        for construct, value in (response.scale_scores_json or {}).items():
            if value is not None and construct in SCALE_MAP:
                constructs[construct].append(float(value))
    survey_means = {
        construct: round(sum(values) / len(values), 3)
        for construct, values in constructs.items()
        if values
    }
    balance_cat = func.coalesce(Product.main_category, Product.category)
    balance = [
        {"condition": condition, "category": category, "products": int(products)}
        for condition, category, products in db.execute(
            select(Product.condition, balance_cat.label("cat"), func.count().label("products"))
            .group_by(Product.condition, balance_cat)
            .order_by(balance_cat, Product.condition)
        ).all()
    ]

    query_submit_events = db.scalars(select(Event).where(Event.event_type == "query_submit")).all()
    query_metrics = {"typed": 0, "suggested": 0}
    for ev in query_submit_events:
        if ev.metadata_json.get("is_suggested"):
            query_metrics["suggested"] += 1
        else:
            query_metrics["typed"] += 1

    return {
        "totals": totals,
        "by_condition": metrics,
        "survey_means": survey_means,
        "catalogue_balance": balance,
        "query_metrics": query_metrics,
        "interpretation_note": (
            "Citation rate is cited product-query opportunities divided by all logged product-query candidates. "
            "Use the exported candidate-level data for preregistered mixed-effects or fixed-effects models."
        ),
    }

@router.get("/dashboard")
def dashboard(db: DBSession = Depends(get_db)) -> dict[str, Any]:
    return _dashboard(db)

@router.get("/admin/analytics/report")
def analytics_report(
    response: Response,
    respondent_limit: int = QueryParam(default=200, ge=1, le=500),
    respondent_offset: int = QueryParam(default=0, ge=0),
    db: DBSession = Depends(get_db),
) -> dict[str, Any]:
    response.headers["Cache-Control"] = "no-store"
    return build_analytics_report(
        db,
        respondent_limit=respondent_limit,
        respondent_offset=respondent_offset,
    )

@router.get("/admin/analytics/download")
def analytics_download(
    format: str = QueryParam(default="csv", max_length=10),
    section: str = QueryParam(default="overview", max_length=40),
    respondent_limit: int = QueryParam(default=500, ge=1, le=500),
    respondent_offset: int = QueryParam(default=0, ge=0),
    db: DBSession = Depends(get_db),
) -> Response:
    normalized_format = format.strip().lower()
    normalized_section = section.strip().lower()
    if normalized_format not in {"csv", "json"}:
        raise _bad_request("Analytics download format must be 'csv' or 'json'.")
    try:
        report = build_analytics_report(
            db,
            respondent_limit=respondent_limit,
            respondent_offset=respondent_offset,
        )
        if normalized_format == "csv":
            content = analytics_report_csv(report, section=normalized_section)
        else:
            content = json.dumps(report, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    except ValueError as error:
        raise _bad_request(str(error)) from error

    suffix = "csv" if normalized_format == "csv" else "json"
    media_type = "text/csv; charset=utf-8" if normalized_format == "csv" else "application/json"
    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="geo-study-analytics-{normalized_section}.{suffix}"',
            "Cache-Control": "no-store",
        },
    )

EXPORT_MODELS: dict[str, Type[Any]] = {
    "products": Product,
    "sessions": Session,
    "queries": Query,
    "candidates": QueryCandidate,
    "probes": ProbeRun,
    "probe_candidates": ProbeCandidate,
    "events": Event,
    "surveys": SurveyResponse,
    "geo_optimization_configs": GEOOptimizationConfig,
    "geo_optimization_applications": GEOOptimizationApplication,
}

def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, bool):
        return int(value)
    return value

@router.get("/export/{name}")
def export(name: str, db: DBSession = Depends(get_db)) -> Response:
    model = EXPORT_MODELS.get(name)
    if model is None:
        raise _bad_request("Unknown export requested.")
    table = model.__table__
    columns = [column.name for column in table.columns]
    statement = select(model)
    if name == "products":
        statement = statement.order_by(Product.category, Product.title)
    elif name == "sessions":
        statement = statement.order_by(Session.started_at)
    elif name == "queries":
        statement = statement.order_by(Query.created_at)
    elif name == "candidates":
        statement = statement.order_by(QueryCandidate.query_id, QueryCandidate.rank_position)
    elif name == "probes":
        statement = statement.order_by(ProbeRun.created_at, ProbeRun.repetition)
    elif name == "probe_candidates":
        statement = statement.order_by(ProbeCandidate.probe_run_id, ProbeCandidate.rank_position)
    elif name == "events":
        statement = statement.order_by(Event.created_at)
    elif name == "surveys":
        statement = statement.order_by(SurveyResponse.completed_at)
    elif name == "geo_optimization_configs":
        statement = statement.order_by(GEOOptimizationConfig.revision)
    elif name == "geo_optimization_applications":
        statement = statement.order_by(GEOOptimizationApplication.created_at)
    rows = db.scalars(statement).all()
    stream = io.StringIO()
    if rows:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _csv_value(getattr(row, column)) for column in columns})
    content = stream.getvalue().encode("utf-8-sig")
    return Response(
        content=content,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="geo-study-{name}.csv"', "Cache-Control": "no-store"},
    )
