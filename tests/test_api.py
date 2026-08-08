"""Contract tests for the FastAPI study backend."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.db import session as database
from app.main import app
from app.models import ProbeCandidate, ProbeRun, QueryCandidate


@pytest.fixture()
def client(tmp_path: Path):
    database.configure_database(f"sqlite:///{(tmp_path / 'study.sqlite3').as_posix()}")
    with TestClient(app) as test_client:
        with database.SessionLocal() as db:
            from app.services.vector_db import get_indexed_products_page
            from app.models import Product
            from app.core.experiment import utc_now
            res = get_indexed_products_page(page=1, limit=1000)
            for prod in res.get("items", []):
                if not db.get(Product, prod["id"]):
                    db.add(Product(
                        id=prod["id"],
                        title=prod.get("title", "Product"),
                        category=prod.get("category", "Uncategorized"),
                        condition="CONTROL",
                        created_at=utc_now(),
                    ))
            db.commit()
        yield test_client
    database.engine.dispose()


def test_participant_flow_is_blinded_and_logs_candidates(client: TestClient) -> None:
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["ok"] is True
    assert health.json()["environment"]

    product_page = client.get("/api/products").json()
    products = product_page["products"]
    assert len(products) == 12
    assert product_page["total"] == 12
    assert product_page["offset"] == 0
    assert "condition" not in products[0]
    assert "geo_bundle" not in products[0]

    invalid = client.post("/api/sessions", json={"consent": False})
    assert invalid.status_code == 400
    assert "error" in invalid.json()
    session = client.post("/api/sessions", json={"consent": True, "participant_code": "PILOT-1"})
    assert session.status_code == 201
    session_id = session.json()["session_id"]

    response = client.post(
        "/api/assistant/query",
        json={"session_id": session_id, "query": "wireless earbuds under 70"},
    )
    assert response.status_code == 201
    payload = response.json()
    assert payload["citations"]
    assert all("condition" not in item["product"] for item in payload["citations"])

    with database.SessionLocal() as db:
        candidates = db.scalar(select(func.count()).select_from(QueryCandidate))
    # The SQL retrieval stage logs only the lexical candidate pool, rather
    # than loading and logging an entire production-scale catalog.
    assert candidates == 2

    event = client.post(
        "/api/events",
        json={
            "session_id": session_id,
            "query_id": payload["query_id"],
            "product_id": payload["citations"][0]["product"]["id"],
            "event_type": "citation_open",
        },
    )
    assert event.status_code == 201
    survey = client.post(
        "/api/surveys",
        json={"session_id": session_id, "answers": {"rq1": 6, "rq2": 5, "rq3": 7, "tr1": 5}},
    )
    assert survey.status_code == 201
    assert survey.json()["scale_scores"]["recommendation_quality"] == 6.0

    dashboard = client.get("/api/dashboard").json()
    assert dashboard["totals"]["queries"] == 1
    assert dashboard["totals"]["completed_surveys"] == 1


def test_research_probe_import_and_export_contract(client: TestClient) -> None:
    probe = client.post(
        "/api/admin/probes",
        json={"query": "water bottle for commuting", "probe_set": "pilot", "repetitions": 2},
    )
    assert probe.status_code == 201
    assert len(probe.json()["runs"]) == 2
    with database.SessionLocal() as db:
        assert db.scalar(select(func.count()).select_from(ProbeRun)) == 2
        assert db.scalar(select(func.count()).select_from(ProbeCandidate)) == 4

    imported = client.post(
        "/api/admin/products/import",
        json={
            "csv": (
                "id,title,category,pair_id,price,key_features\n"
                "TEST-01,Test Travel Mug,Home & Kitchen,test-mug,20,insulated|leak-resistant\n"
                "TEST-02,Test Commute Mug,Home & Kitchen,test-mug,21,insulated|leak-resistant\n"
            )
        },
    )
    assert imported.status_code == 201
    assert imported.json()["control"] == 1
    assert imported.json()["geo_optimized"] == 1

    researcher_products = client.get("/api/research/products")
    assert researcher_products.status_code == 200
    test_rows = [row for row in researcher_products.json()["products"] if row["id"].startswith("TEST-")]
    assert {row["condition"] for row in test_rows} == {"CONTROL", "GEO_OPTIMIZED"}

    export = client.get("/api/export/probe_candidates")
    assert export.status_code == 200
    assert export.headers["content-type"].startswith("text/csv")
    assert "probe_run_id" in export.text


def test_product_catalog_supports_bounded_pagination_and_sql_text_filter(client: TestClient) -> None:
    first_page = client.get("/api/products?limit=2&offset=0")
    assert first_page.status_code == 200
    assert first_page.json()["total"] == 12
    assert len(first_page.json()["products"]) == 2

    second_page = client.get("/api/products?limit=2&offset=2")
    assert second_page.status_code == 200
    assert second_page.json()["offset"] == 2
    assert {row["id"] for row in first_page.json()["products"]}.isdisjoint(
        {row["id"] for row in second_page.json()["products"]}
    )

    filtered = client.get("/api/products?q=earbuds")
    assert filtered.status_code == 200
    assert filtered.json()["total"] == 2
    assert all("earbuds" in row["title"].lower() for row in filtered.json()["products"])


def test_admin_respondents_and_activity_endpoint(client: TestClient) -> None:
    session = client.post(
        "/api/sessions",
        json={"consent": True, "email": "test@example.com", "age": 28, "country": "United States"},
    )
    assert session.status_code == 201
    sid = session.json()["session_id"]

    res = client.get("/api/admin/respondents")
    assert res.status_code == 200
    items = res.json()["items"]
    assert any(item["session_id"] == sid and item["email"] == "test@example.com" for item in items)

    activity = client.get(f"/api/admin/respondents/{sid}/activity")
    assert activity.status_code == 200
    assert activity.json()["session"]["email"] == "test@example.com"
    assert activity.json()["session"]["age"] == 28

