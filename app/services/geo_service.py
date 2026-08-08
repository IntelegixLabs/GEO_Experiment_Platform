"""High-level, framework-independent service facade for GEO study routes.

FastAPI handlers can import :class:`GEOService` without coupling route code to
the treatment builder, retrieval implementation, or agent internals.  Every
method accepts dictionaries, Pydantic v1/v2 models, dataclasses, or simple ORM
objects and returns JSON-safe primitive structures.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from .experiments import (
    AssignmentResult,
    assign_balanced_conditions,
    catalog_balance,
    consumer_event_summary,
    evaluate_candidates as _evaluate_candidates,
    scale_scores,
)
from .geo_treatment import FactualGEOBuilder, LLMGEOBuilder, GEOIntegrityValidator, TreatmentIntegrityError
from .retrieval import RetrievalConfig, SemanticSimilarityAdapter, TransparentRetriever
from .shopping_agent import GroundedGenerationAdapter, ShoppingResearchAgent
from .text import to_mapping


class GEOService:
    """Compose deterministic GEO treatment, retrieval, agent, and metrics tools.

    Parameters are dependency-injection points, not automatic external calls.
    Passing an embedding/LLM adapter is opt-in; record its name and version in
    the experiment register before using it in a participant-facing wave.
    """

    def __init__(
        self,
        *,
        retrieval_config: RetrievalConfig | None = None,
        semantic_adapter: SemanticSimilarityAdapter | None = None,
        generation_adapter: GroundedGenerationAdapter | None = None,
        allow_external_generation: bool = False,
    ) -> None:
        if allow_external_generation and generation_adapter:
            self.builder = LLMGEOBuilder(generation_adapter)
        else:
            self.builder = FactualGEOBuilder()
        self.validator = GEOIntegrityValidator(self.builder)
        self.retriever = TransparentRetriever(retrieval_config, semantic_adapter)
        self.agent = ShoppingResearchAgent(
            self.retriever,
            generation_adapter,
            allow_external_generation=allow_external_generation,
        )

    def build_treatment(self, product: Any, *, enforce_integrity: bool = True) -> dict[str, Any]:
        """Create a factual GEO bundle and immediately run its integrity gate."""

        record = to_mapping(product)
        bundle = self.builder.build(record)
        report = self.validator.validate(record, bundle)
        if enforce_integrity and not report.valid:
            messages = "; ".join(issue.message for issue in report.errors)
            raise TreatmentIntegrityError(messages)
        return {"geo_bundle": bundle, "integrity": report.as_dict()}

    def validate_treatment(self, product: Any, bundle: Any | None = None) -> dict[str, Any]:
        """Validate a generated/imported GEO bundle without changing data."""

        return self.validator.validate(product, bundle).as_dict()

    def prepare_product(
        self,
        product: Any,
        *,
        condition: str | None = None,
        enforce_integrity: bool = True,
    ) -> dict[str, Any]:
        """Return a product copy with a GEO bundle only for the treatment arm."""

        record = dict(to_mapping(product))
        active_condition = (condition or record.get("condition") or "").upper().replace(" ", "_")
        if active_condition in {"GEO", "TREATMENT", "OPTIMIZED"}:
            active_condition = "GEO_OPTIMIZED"
        if active_condition in {"CONTROL", "BASELINE"}:
            active_condition = "CONTROL"
        if active_condition == "GEO_OPTIMIZED":
            treatment = self.build_treatment(record, enforce_integrity=enforce_integrity)
            record["condition"] = active_condition
            record["geo_bundle"] = treatment["geo_bundle"]
            record["geo_integrity"] = treatment["integrity"]
        elif active_condition == "CONTROL":
            record["condition"] = active_condition
            record.pop("geo_bundle", None)
            record.pop("treatment_bundle", None)
        else:
            raise ValueError("Product condition must be CONTROL or GEO_OPTIMIZED.")
        return record

    def search_catalog(
        self,
        products: Iterable[Any],
        query: Any,
        *,
        category_filter: Any = None,
        limit: int | None = None,
        include_product: bool = False,
    ) -> dict[str, Any]:
        """Run deterministic lexical/semantic-proxy retrieval and reranking.

        ``results`` and ``ranked`` are aliases for the same JSON-safe candidate
        rows so route code can use either conventional response shape.
        """

        payload = self.retriever.search(
            products,
            query,
            category_filter=category_filter,
            limit=limit,
            include_product=include_product,
        )
        payload["ranked"] = payload["results"]
        return payload

    def answer_shopping_query(
        self,
        products: Iterable[Any],
        query: Any,
        *,
        category_filter: Any = None,
        include_candidates: bool = True,
    ) -> dict[str, Any]:
        """Return a relevance-gated, cited shopping answer and candidate log."""

        payload = self.agent.answer(
            products,
            query,
            category_filter=category_filter,
            include_candidates=include_candidates,
        )
        payload["ranked"] = payload.get("candidates", [])
        payload["results"] = payload.get("candidates", [])
        return payload

    def assign_conditions(
        self,
        products: Iterable[Any],
        *,
        seed: str = "geo-study-v1",
        respect_existing: bool = True,
    ) -> dict[str, Any]:
        assignment: AssignmentResult = assign_balanced_conditions(
            products,
            seed=seed,
            respect_existing=respect_existing,
        )
        return {"products": assignment.products, "assignment": assignment.report}

    def catalog_balance(self, products: Iterable[Any]) -> dict[str, Any]:
        return catalog_balance(products)

    def evaluate_candidates(self, candidate_rows: Iterable[Any], *, top_k: int = 3) -> dict[str, Any]:
        return _evaluate_candidates(candidate_rows, top_k=top_k)

    def evaluate_experiment(
        self,
        candidate_rows: Iterable[Any],
        *,
        products: Iterable[Any] | None = None,
        events: Iterable[Any] | None = None,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Bundle visibility metrics with optional balance/engagement summaries."""

        output: dict[str, Any] = {"visibility": self.evaluate_candidates(candidate_rows, top_k=top_k)}
        if products is not None:
            output["catalog_balance"] = self.catalog_balance(products)
        if events is not None:
            output["consumer_events"] = consumer_event_summary(events)
        return output

    def scale_scores(self, answers: dict[str, Any]) -> dict[str, float | None]:
        return scale_scores(answers)


# Functional aliases are useful for small FastAPI route handlers and tests.
def build_treatment(product: Any, *, enforce_integrity: bool = True) -> dict[str, Any]:
    return GEOService().build_treatment(product, enforce_integrity=enforce_integrity)


def validate_treatment(product: Any, bundle: Any | None = None) -> dict[str, Any]:
    return GEOService().validate_treatment(product, bundle)


def search_catalog(
    products: Iterable[Any],
    query: Any,
    *,
    category_filter: Any = None,
    limit: int | None = None,
    include_product: bool = False,
) -> dict[str, Any]:
    return GEOService().search_catalog(
        products,
        query,
        category_filter=category_filter,
        limit=limit,
        include_product=include_product,
    )


def answer_shopping_query(
    products: Iterable[Any],
    query: Any,
    *,
    category_filter: Any = None,
    include_candidates: bool = True,
) -> dict[str, Any]:
    return GEOService().answer_shopping_query(
        products,
        query,
        category_filter=category_filter,
        include_candidates=include_candidates,
    )


def assign_conditions(
    products: Iterable[Any],
    *,
    seed: str = "geo-study-v1",
    respect_existing: bool = True,
) -> dict[str, Any]:
    return GEOService().assign_conditions(products, seed=seed, respect_existing=respect_existing)


def evaluate_candidates(candidate_rows: Iterable[Any], *, top_k: int = 3) -> dict[str, Any]:
    return GEOService().evaluate_candidates(candidate_rows, top_k=top_k)
