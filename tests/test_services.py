"""Unit tests for deterministic GEO treatment/retrieval/agent services."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import unittest


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.geo_service import GEOService  # noqa: E402


def bottle_product() -> dict[str, object]:
    return {
        "id": "BOTTLE-1",
        "sku": "BTL-001",
        "title": "Aster Trail Insulated Water Bottle",
        "brand": "Aster",
        "category": "Home & Kitchen",
        "price": 24.0,
        "currency": "USD",
        "description": "A 750 ml steel bottle for everyday travel.",
        "key_features": ["750 ml capacity", "double-wall steel", "leak-resistant cap"],
        "rating": 4.5,
        "review_count": 218,
        "availability": "In stock",
        "shipping": "Dispatches in 2 business days",
        "return_policy": "30-day returns",
        "source_url": "https://catalog.example.test/bottle-1",
        "source_timestamp": "2026-07-19T00:00:00Z",
        "condition": "GEO_OPTIMIZED",
        "pair_id": "bottle-pair",
    }


def mat_product() -> dict[str, object]:
    return {
        "id": "MAT-1",
        "sku": "MAT-001",
        "title": "Beacon Balance Yoga Mat",
        "brand": "Beacon",
        "category": "Fitness",
        "price": 29.0,
        "currency": "USD",
        "description": "A non-slip exercise mat for home yoga.",
        "key_features": ["6 mm cushioning", "non-slip surface"],
        "rating": 4.6,
        "review_count": 120,
        "availability": "In stock",
        "shipping": "Dispatches next business day",
        "return_policy": "30-day returns",
        "source_url": "https://catalog.example.test/mat-1",
        "source_timestamp": "2026-07-19T00:00:00Z",
        "condition": "CONTROL",
        "pair_id": "mat-pair",
    }


class GEOServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = GEOService()

    def test_factual_treatment_is_deterministic_and_valid(self) -> None:
        product = bottle_product()
        first = self.service.build_treatment(product)
        second = self.service.build_treatment(product)
        self.assertEqual(first["geo_bundle"], second["geo_bundle"])
        self.assertTrue(first["integrity"]["valid"])
        self.assertIn("double-wall steel", first["geo_bundle"]["summary"])
        self.assertNotIn("best", first["geo_bundle"]["summary"].lower())

    def test_integrity_gate_rejects_tampered_unsupported_bundle(self) -> None:
        product = bottle_product()
        bundle = self.service.build_treatment(product)["geo_bundle"]
        tampered = copy.deepcopy(bundle)
        tampered["summary"] = "The best guaranteed bottle."
        report = self.service.validate_treatment(product, tampered)
        self.assertFalse(report["valid"])
        self.assertTrue(any(issue["code"] == "content_hash_mismatch" for issue in report["errors"]))
        self.assertTrue(any(issue["code"] == "unsupported_persuasion_term" for issue in report["errors"]))

    def test_agent_cites_only_relevant_product_and_hides_condition(self) -> None:
        bottle = bottle_product()
        bottle["geo_bundle"] = self.service.build_treatment(bottle)["geo_bundle"]
        answer = self.service.answer_shopping_query(
            [bottle, mat_product()],
            "Compare insulated water bottles under $30 for commuting",
        )
        self.assertEqual(answer["cited_ids"], ["BOTTLE-1"])
        self.assertNotIn("condition", answer["citations"][0]["product"])
        self.assertGreater(answer["candidates"][0]["retrieval_score"], answer["candidates"][1]["retrieval_score"])

    def test_agent_returns_no_citation_for_no_material_catalog_match(self) -> None:
        answer = self.service.answer_shopping_query([bottle_product(), mat_product()], "Find a vacuum cleaner with HEPA filter")
        self.assertEqual(answer["cited_ids"], [])
        self.assertIn("could not find", answer["answer"].lower())

    def test_assignment_and_candidate_metrics_are_transparent(self) -> None:
        records = [
            {"id": "A", "title": "One", "category": "Office", "pair_id": "pair-1"},
            {"id": "B", "title": "Two", "category": "Office", "pair_id": "pair-1"},
        ]
        assigned = self.service.assign_conditions(records)
        self.assertEqual({item["condition"] for item in assigned["products"]}, {"CONTROL", "GEO_OPTIMIZED"})
        self.assertNotIn("condition", records[0])
        metrics = self.service.evaluate_candidates(
            [
                {"condition": "CONTROL", "retrieved": True, "rank_position": 1, "cited": False, "retrieval_score": 4},
                {"condition": "CONTROL", "retrieved": True, "rank_position": 2, "cited": True, "retrieval_score": 3},
                {"condition": "GEO_OPTIMIZED", "retrieved": True, "rank_position": 1, "cited": True, "retrieval_score": 5},
                {"condition": "GEO_OPTIMIZED", "retrieved": False, "rank_position": 4, "cited": False, "retrieval_score": 0},
            ]
        )
        self.assertEqual(metrics["by_condition"]["CONTROL"]["citation_rate"], 0.5)
        self.assertEqual(metrics["by_condition"]["GEO_OPTIMIZED"]["citation_rate"], 0.5)
        self.assertEqual(metrics["citation_effect_geo_minus_control"]["risk_difference"], 0.0)


if __name__ == "__main__":
    unittest.main()
