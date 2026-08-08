"""API coverage for the persistent researcher GEO optimization panel."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

import os
from app.db import session as database
from app.main import app


@pytest.fixture()
def client(tmp_path: Path):
    os.environ["CHROMA_DB_PATH"] = str(tmp_path / "test_chroma_db")
    database.configure_database(f"sqlite:///{(tmp_path / 'study.sqlite3').as_posix()}")
    with TestClient(app) as test_client:
        yield test_client
    database.engine.dispose()


def test_geo_optimization_panel_defaults_dry_run_and_persisted_apply(client: TestClient) -> None:
    initial = client.get("/api/admin/geo-optimization/config")
    assert initial.status_code == 200
    initial_payload = initial.json()
    assert initial_payload["config"]["id"] == "draft"
    assert initial_payload["scope_summary"]["catalog_total"] == 12
    assert initial_payload["options"]["parameter_weights"]["title_weight"]["step"] == 0.1

    dry_run = client.post(
        "/api/admin/geo-optimization/apply",
        json={
            "scope": {"type": "all_catalog", "categories": [], "product_ids": [], "pair_ids": []},
            "treatment_percentage": 50,
            "dry_run": True,
        },
    )
    assert dry_run.status_code == 201
    assert dry_run.json()["application"]["updated_products"] == 0
    assert dry_run.json()["application"]["control_products"] == 6
    assert dry_run.json()["application"]["geo_optimized_products"] == 6
    assert dry_run.json()["application"]["integrity_failures"] == 0

    applied = client.post(
        "/api/admin/geo-optimization/apply",
        json={
            "name": "Earbud matched-pair pilot",
            "optimization_target": "citation_visibility",
            "assignment_strategy": "matched_pairs",
            "treatment_percentage": 50,
            "feature_toggles": {
                "factual_summary": False,
                "structured_specifications": True,
                "claim_evidence_links": True,
                "factual_faq": False,
                "offer_details": False,
                "agent_readable_provenance": True,
            },
            "scope": {"type": "pair_ids", "categories": [], "product_ids": [], "pair_ids": ["electronics-earbuds"]},
        },
    )
    assert applied.status_code == 201
    result = applied.json()
    assert result["config"]["id"].startswith("GOC-")
    assert result["application"]["run_id"].startswith("GOA-")
    assert result["application"]["selected_products"] == 2
    assert result["application"]["control_products"] == 1
    assert result["application"]["geo_optimized_products"] == 1
    assert result["application"]["integrity_failures"] == 0

    rows = client.get("/api/research/products?category=Electronics").json()["products"]
    treatment_row = next(row for row in rows if row["pair_id"] == "electronics-earbuds" and row["condition"] == "GEO_OPTIMIZED")
    assert treatment_row["geo_bundle"]["summary"] == ""
    assert treatment_row["geo_bundle"]["faq"] == []
    assert "Listed price" not in treatment_row["geo_bundle"]["specifications"]
    assert treatment_row["geo_bundle"]["feature_vector"]["factual_faq"] == 0

    persisted = client.get("/api/admin/geo-optimization/config").json()
    assert persisted["config"]["id"] == result["config"]["id"]
    assert persisted["config"]["feature_toggles"]["factual_summary"] is False
    assert persisted["config"]["feature_toggles"]["factual_faq"] is False


def test_geo_optimization_rejects_invalid_matched_pair_split(client: TestClient) -> None:
    invalid = client.post(
        "/api/admin/geo-optimization/apply",
        json={
            "assignment_strategy": "matched_pairs",
            "treatment_percentage": 60,
            "scope": {"type": "pair_ids", "categories": [], "product_ids": [], "pair_ids": ["electronics-earbuds"]},
        },
    )
    assert invalid.status_code == 422
    assert "50%" in invalid.json()["error"]
