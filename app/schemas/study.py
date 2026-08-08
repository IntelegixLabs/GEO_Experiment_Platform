"""Input contracts shared by participant and researcher endpoints."""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class APIModel(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)


class SessionCreate(APIModel):
    consent: bool
    participant_code: str | None = Field(default=None, max_length=120)
    email: str | None = Field(default=None, max_length=160)
    age: int | None = Field(default=None, ge=18, le=120)
    country: str | None = Field(default=None, max_length=80)
    ai_familiarity: str | None = Field(default=None, max_length=80)
    study_cohort: str | None = Field(default="main", max_length=60)


class AssistantQueryCreate(APIModel):
    session_id: str = Field(min_length=1, max_length=80)
    query: str = Field(min_length=1, max_length=500)
    category_filter: str | None = Field(default=None, max_length=250)


class EventCreate(APIModel):
    session_id: str = Field(min_length=1, max_length=80)
    event_type: str = Field(min_length=1, max_length=60)
    query_id: str | None = Field(default=None, max_length=80)
    product_id: str | None = Field(default=None, max_length=80)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SurveyCreate(APIModel):
    session_id: str = Field(min_length=1, max_length=80)
    answers: dict[str, Any]


class ProbeCreate(APIModel):
    query: str = Field(min_length=1, max_length=500)
    repetitions: int = Field(default=1, ge=1, le=10)
    category_filter: str | None = Field(default=None, max_length=250)
    probe_set: str | None = Field(default="ad_hoc", max_length=80)
    engine_name: str | None = Field(default="Controlled catalog retrieval", max_length=120)
    model_version: str | None = Field(default="GEO-Study-Retriever/1.0", max_length=120)
    locale: str | None = Field(default=None, max_length=60)


class CatalogImportCreate(APIModel):
    csv: str = Field(min_length=1, max_length=2_000_000)


# GEO optimization-panel contracts.  These limits intentionally keep a browser
# panel from submitting an unbounded catalog list or arbitrary scorer settings.
GEO_MODEL_PROFILES = ("factual_structure", "evidence_first", "balanced_retrieval", "custom")
GEO_OPTIMIZATION_TARGETS = ("citation_visibility", "balanced_visibility", "decision_quality")
GEO_ASSIGNMENT_STRATEGIES = ("stratified_split", "matched_pairs")
GEO_SCOPE_TYPES = ("all_catalog", "categories", "product_ids", "pair_ids")
GEO_PARAMETER_WEIGHT_DEFAULTS: dict[str, float] = {
    "title_weight": 4.5,
    "category_weight": 3.2,
    "feature_weight": 2.8,
    "description_weight": 1.5,
    "offer_weight": 1.2,
    "semantic_weight": 1.8,
    "evidence_match_weight": 1.4,
    "max_structural_evidence_bonus": 2.14,
}
GEO_FEATURE_TOGGLE_DEFAULTS: dict[str, bool] = {
    "factual_summary": True,
    "structured_specifications": True,
    "claim_evidence_links": True,
    "factual_faq": True,
    "offer_details": True,
    "agent_readable_provenance": True,
}


def _clean_unique(values: list[str]) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        text = str(value).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


class GEOOptimizationScope(APIModel):
    type: Literal["all_catalog", "categories", "product_ids", "pair_ids"] = "all_catalog"
    categories: list[str] = Field(default_factory=list, max_length=100)
    product_ids: list[str] = Field(default_factory=list, max_length=1_000)
    pair_ids: list[str] = Field(default_factory=list, max_length=1_000)
    limit: int | None = Field(default=None, ge=1)

    @field_validator("categories", "product_ids", "pair_ids")
    @classmethod
    def clean_scope_values(cls, values: list[str]) -> list[str]:
        return _clean_unique(values)

    @model_validator(mode="after")
    def validate_selected_scope(self) -> "GEOOptimizationScope":
        required = {
            "categories": self.categories,
            "product_ids": self.product_ids,
            "pair_ids": self.pair_ids,
        }
        expected_field = {
            "categories": "categories",
            "product_ids": "product_ids",
            "pair_ids": "pair_ids",
        }.get(self.type)
        if expected_field and not required[expected_field]:
            raise ValueError(f"Scope type '{self.type}' requires one or more {expected_field}.")
        return self


class GEOOptimizationApply(APIModel):
    name: str = Field(default="GEO optimization wave", min_length=1, max_length=160)
    optimization_target: Literal["citation_visibility", "balanced_visibility", "decision_quality"] = "citation_visibility"
    model_profile: Literal["factual_structure", "evidence_first", "balanced_retrieval", "custom"] = "balanced_retrieval"
    # These identify a researcher-selected profile; they do not activate or
    # imply an external LLM. The service remains transparent and local.
    model_name: str = Field(default="Transparent factual GEO generator", min_length=1, max_length=160)
    model_version: str = Field(default="GEO-v2-factual-structure", min_length=1, max_length=120)
    treatment_percentage: int = Field(default=50, ge=1, le=99)
    assignment_strategy: Literal["stratified_split", "matched_pairs"] = "stratified_split"
    random_seed: str = Field(default="geo-study-v1", min_length=1, max_length=160)
    parameter_weights: dict[str, float] = Field(default_factory=lambda: dict(GEO_PARAMETER_WEIGHT_DEFAULTS))
    feature_toggles: dict[str, bool] = Field(default_factory=lambda: dict(GEO_FEATURE_TOGGLE_DEFAULTS))
    scope: GEOOptimizationScope = Field(default_factory=GEOOptimizationScope)
    dry_run: bool = False
    # A reconfiguration after response data is collected must be a deliberate
    # researcher decision. The route returns a 409 until this is affirmed.
    confirm_reconfiguration: bool = False

    @field_validator("parameter_weights")
    @classmethod
    def validate_parameter_weights(cls, values: dict[str, float]) -> dict[str, float]:
        unknown = sorted(set(values) - set(GEO_PARAMETER_WEIGHT_DEFAULTS))
        if unknown:
            raise ValueError(f"Unsupported GEO parameter weight(s): {', '.join(unknown)}.")
        merged = dict(GEO_PARAMETER_WEIGHT_DEFAULTS)
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"GEO parameter '{key}' must be a finite number.")
            numeric = float(value)
            if numeric < 0 or numeric > 20:
                raise ValueError(f"GEO parameter '{key}' must be between 0 and 20.")
            merged[key] = numeric
        return merged

    @field_validator("feature_toggles")
    @classmethod
    def validate_feature_toggles(cls, values: dict[str, bool]) -> dict[str, bool]:
        unknown = sorted(set(values) - set(GEO_FEATURE_TOGGLE_DEFAULTS))
        if unknown:
            raise ValueError(f"Unsupported GEO feature toggle(s): {', '.join(unknown)}.")
        if any(not isinstance(value, bool) for value in values.values()):
            raise ValueError("GEO feature toggles must be true or false.")
        merged = dict(GEO_FEATURE_TOGGLE_DEFAULTS)
        merged.update(values)
        rendered_sections = ("factual_summary", "structured_specifications", "claim_evidence_links", "factual_faq")
        if not any(merged[name] for name in rendered_sections):
            raise ValueError("Enable at least one factual GEO content component.")
        return merged

    @model_validator(mode="after")
    def validate_assignment_strategy(self) -> "GEOOptimizationApply":
        if self.assignment_strategy == "matched_pairs" and self.treatment_percentage != 50:
            raise ValueError("Matched-pair assignment requires a 50% GEO treatment split.")
        return self
