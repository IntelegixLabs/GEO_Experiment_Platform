"""Researcher analytics API coverage."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.db import session as database
from app.main import app


@pytest.fixture()
def client(tmp_path: Path):
    database.configure_database(f"sqlite:///{(tmp_path / 'study.sqlite3').as_posix()}")
    with TestClient(app) as test_client:
        yield test_client
    database.engine.dispose()


def test_consent_safe_researcher_analytics_report_and_downloads(client: TestClient) -> None:
    first = client.post(
        "/api/sessions",
        json={
            "consent": True,
            "participant_code": "P-ALPHA-PRIVATE",
            "country": "Private Country",
            "ai_familiarity": "daily",
            "study_cohort": "pilot-a",
        },
    )
    assert first.status_code == 201
    session_id = first.json()["session_id"]
    second = client.post("/api/sessions", json={"consent": True, "participant_code": "P-BETA-PRIVATE"})
    assert second.status_code == 201

    query = client.post(
        "/api/assistant/query",
        json={"session_id": session_id, "query": "wireless earbuds under 70", "category_filter": "Electronics"},
    )
    assert query.status_code == 201
    cited = query.json()["citations"][0]["product"]["id"]
    assert client.post(
        "/api/events",
        json={"session_id": session_id, "query_id": query.json()["query_id"], "product_id": cited, "event_type": "citation_open"},
    ).status_code == 201
    assert client.post(
        "/api/events",
        json={"session_id": session_id, "product_id": cited, "event_type": "purchase_intent"},
    ).status_code == 201
    assert client.post(
        "/api/surveys",
        json={
            "session_id": session_id,
            "answers": {"rq1": 6, "rq2": 5, "rq3": 7, "sc1": 5, "tr1": 6, "pi1": 7},
        },
    ).status_code == 201

    report_response = client.get("/api/admin/analytics/report?respondent_limit=1")
    assert report_response.status_code == 200
    assert report_response.headers["cache-control"] == "no-store"
    report = report_response.json()
    assert report["privacy"]["consented_sessions_only"] is True
    assert report["totals"]["consented_sessions"] == 2
    assert report["totals"]["queries"] == 1
    assert report["totals"]["completed_surveys"] == 1
    assert report["respondents"]["limit"] == 1
    assert report["respondents"]["total"] == 2
    assert report["respondents"]["has_more"] is True
    assert report["respondents"]["items"][0]["respondent_id"].startswith("R-")
    assert report["respondents"]["items"][0]["respondent_id"] != session_id
    assert "recommendation_quality" in report["respondents"]["items"][0]["survey_scores"]
    assert {"CONTROL", "GEO_OPTIMIZED"}.issubset(report["respondents"]["items"][0]["conditions"])
    assert any(row["category"] == "Electronics" for row in report["category_effects"])
    assert any(row["stage"] == "Completed survey" and row["count"] == 1 for row in report["funnel"])
    assert report["timeline"]
    assert all(len(construct["distribution"]) == 7 for construct in report["survey_distributions"])

    rendered = json.dumps(report)
    assert "P-ALPHA-PRIVATE" not in rendered
    assert "Private Country" not in rendered
    assert "wireless earbuds under 70" not in rendered
    assert "raw_survey_answers" in report["privacy"]["raw_fields_excluded"]

    csv_download = client.get("/api/admin/analytics/download?format=csv&section=conditions")
    assert csv_download.status_code == 200
    assert csv_download.headers["content-type"].startswith("text/csv")
    assert "candidate_opportunities" in csv_download.text
    assert "GEO_OPTIMIZED" in csv_download.text

    json_download = client.get("/api/admin/analytics/download?format=json&section=overview")
    assert json_download.status_code == 200
    assert json_download.headers["content-type"].startswith("application/json")
    assert json_download.json()["totals"]["queries"] == 1


def test_analytics_download_rejects_unknown_format_and_section(client: TestClient) -> None:
    invalid_format = client.get("/api/admin/analytics/download?format=xlsx")
    assert invalid_format.status_code == 400
    invalid_section = client.get("/api/admin/analytics/download?section=private")
    assert invalid_section.status_code == 400
