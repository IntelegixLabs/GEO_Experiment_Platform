"""Deterministic GEO treatment, retrieval, agent, and evaluation services."""

from .experiments import (
    CONDITIONS,
    CONTROL,
    GEO_OPTIMIZED,
    assign_balanced_conditions,
    catalog_balance,
    scale_scores,
)
from .geo_service import (
    GEOService,
    answer_shopping_query,
    assign_conditions,
    build_treatment,
    evaluate_candidates,
    search_catalog,
    validate_treatment,
)
from .geo_treatment import (
    FactualGEOBuilder,
    GEOIntegrityValidator,
    TreatmentIntegrityError,
    factual_geo_bundle,
    validate_geo_bundle,
)
from .retrieval import RetrievalConfig, TransparentRetriever, analyse_query, search_catalog as transparent_search_catalog
from .shopping_agent import ShoppingResearchAgent, answer_shopping_query as deterministic_shopping_answer

__all__ = [
    "CONDITIONS",
    "CONTROL",
    "GEO_OPTIMIZED",
    "FactualGEOBuilder",
    "GEOIntegrityValidator",
    "GEOService",
    "RetrievalConfig",
    "ShoppingResearchAgent",
    "TreatmentIntegrityError",
    "TransparentRetriever",
    "analyse_query",
    "answer_shopping_query",
    "assign_conditions",
    "assign_balanced_conditions",
    "build_treatment",
    "catalog_balance",
    "evaluate_candidates",
    "factual_geo_bundle",
    "scale_scores",
    "search_catalog",
    "transparent_search_catalog",
    "deterministic_shopping_answer",
    "validate_treatment",
    "validate_geo_bundle",
]
