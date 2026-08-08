"""Persistent study records.

The table names intentionally match the original local prototype so exports and
the data dictionary retain the same conceptual row units.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Product(Base):
    __tablename__ = "products"
    __table_args__ = (
        CheckConstraint("condition IN ('CONTROL', 'GEO_OPTIMIZED')", name="ck_product_condition"),
        Index("idx_products_category", "category"),
        Index("idx_products_main_category", "main_category"),
        Index("idx_products_sub_category", "sub_category"),
        Index("idx_products_title", "title"),
        Index("idx_products_source_product_key", "source_product_key", unique=True),
    )

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    sku: Mapped[str | None] = mapped_column(String(160), nullable=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    brand: Mapped[str | None] = mapped_column(String(250), nullable=True)
    category: Mapped[str] = mapped_column(String(250), nullable=False)
    # ``category`` remains the participant-facing primary category.  The
    # source fields retain the Amazon hierarchy without flattening it away.
    main_category: Mapped[str | None] = mapped_column(String(250), nullable=True)
    sub_category: Mapped[str | None] = mapped_column(String(250), nullable=True)
    price: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    discount_price: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    actual_price: Mapped[float | None] = mapped_column(Numeric(14, 2), nullable=True)
    currency: Mapped[str] = mapped_column(String(12), nullable=False, default="INR")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    key_features: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    rating: Mapped[float | None] = mapped_column(Float, nullable=True)
    review_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    availability: Mapped[str | None] = mapped_column(String(250), nullable=True)
    shipping: Mapped[str | None] = mapped_column(String(500), nullable=True)
    return_policy: Mapped[str | None] = mapped_column(String(500), nullable=True)
    source_url: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    image_url: Mapped[str | None] = mapped_column(String(2000), nullable=True)
    condition: Mapped[str] = mapped_column(String(20), nullable=False)
    pair_id: Mapped[str | None] = mapped_column(String(160), nullable=True)
    # Import provenance is retained with the study-facing product so every
    # experimental row can be traced back to a source record without storing
    # scraped facts that were not present in the supplied dataset.
    source_dataset: Mapped[str | None] = mapped_column(String(250), nullable=True)
    source_row_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    source_product_key: Mapped[str | None] = mapped_column(String(100), nullable=True)
    imported_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    data_quality_flags: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    # Persist no treatment as SQL NULL, not JSON ``null``.  That keeps
    # researcher queries and PostgreSQL partial indexes semantically correct.
    geo_bundle: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


from pgvector.sqlalchemy import Vector

class ProductVector(Base):
    __tablename__ = "product_vectors"
    
    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    product_id: Mapped[str] = mapped_column(ForeignKey("products.id"), nullable=False)
    embedding = mapped_column(Vector(384), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)



class GEOOptimizationConfig(Base):
    """A versioned researcher configuration for a controlled GEO wave.

    Configuration rows are immutable in normal use: every applied change gets
    a new revision, while ``is_active`` identifies the configuration that the
    researcher most recently applied.  This keeps the parameter choices and
    their scope available for preregistration and later audit.
    """

    __tablename__ = "geo_optimization_configs"
    __table_args__ = (
        CheckConstraint("treatment_percentage >= 1 AND treatment_percentage <= 99", name="ck_geo_config_split"),
        Index("idx_geo_config_active", "is_active"),
        Index("idx_geo_config_created", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    optimization_target: Mapped[str] = mapped_column(String(80), nullable=False)
    model_profile: Mapped[str] = mapped_column(String(80), nullable=False)
    model_name: Mapped[str] = mapped_column(String(160), nullable=False)
    model_version: Mapped[str] = mapped_column(String(120), nullable=False)
    treatment_percentage: Mapped[int] = mapped_column(Integer, nullable=False)
    assignment_strategy: Mapped[str] = mapped_column(String(80), nullable=False)
    random_seed: Mapped[str] = mapped_column(String(160), nullable=False)
    parameter_weights_json: Mapped[dict[str, float]] = mapped_column(JSON, nullable=False, default=dict)
    feature_toggles_json: Mapped[dict[str, bool]] = mapped_column(JSON, nullable=False, default=dict)
    scope_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    last_applied_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class GEOOptimizationApplication(Base):
    """Immutable audit event for a configuration application to catalog rows."""

    __tablename__ = "geo_optimization_applications"
    __table_args__ = (Index("idx_geo_application_config", "config_id"), Index("idx_geo_application_created", "created_at"))

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    config_id: Mapped[str] = mapped_column(ForeignKey("geo_optimization_configs.id"), nullable=False)
    config_snapshot_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    scope_summary_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    previous_assignment_json: Mapped[dict[str, int]] = mapped_column(JSON, nullable=False, default=dict)
    application_summary_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    safety_notes_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class GEOOptimizedProduct(Base):
    """An optimized or control product record for a specific GEO wave run.
    
    This keeps the canonical products table untouched so multiple experiments
    can be run cleanly without overwriting catalog data.
    """

    __tablename__ = "geo_optimized_products"
    __table_args__ = (
        Index("idx_gop_run_id", "run_id"),
        Index("idx_gop_product_id", "product_id"),
        Index("idx_gop_condition", "condition"),
    )

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(80), nullable=False)
    product_id: Mapped[str] = mapped_column(String(80), nullable=False)
    condition: Mapped[str] = mapped_column(String(20), nullable=False)
    geo_bundle: Mapped[dict[str, Any] | None] = mapped_column(JSON(none_as_null=True), nullable=True)
    product_snapshot_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    participant_code: Mapped[str] = mapped_column(String(60), nullable=False)
    email: Mapped[str | None] = mapped_column(String(160), nullable=True)
    age: Mapped[int | None] = mapped_column(Integer, nullable=True)
    consent: Mapped[bool] = mapped_column(Boolean, nullable=False)
    country: Mapped[str | None] = mapped_column(String(80), nullable=True)
    ai_familiarity: Mapped[str | None] = mapped_column(String(80), nullable=True)
    study_cohort: Mapped[str | None] = mapped_column(String(60), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Query(Base):
    __tablename__ = "queries"
    __table_args__ = (Index("idx_queries_session", "session_id"),)

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), nullable=False)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    category_filter: Mapped[str | None] = mapped_column(String(250), nullable=True)
    intent_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    response_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class QueryCandidate(Base):
    __tablename__ = "query_candidates"
    __table_args__ = (
        Index("idx_candidates_condition", "condition"),
        UniqueConstraint("query_id", "product_id", name="uq_query_candidate"),
    )

    query_id: Mapped[str] = mapped_column(ForeignKey("queries.id"), primary_key=True)
    product_id: Mapped[str] = mapped_column(ForeignKey("products.id"), primary_key=True)
    condition: Mapped[str] = mapped_column(String(20), nullable=False)
    rank_position: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieval_score: Mapped[float] = mapped_column(Float, nullable=False)
    lexical_score: Mapped[float] = mapped_column(Float, nullable=False)
    evidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    cited: Mapped[bool] = mapped_column(Boolean, nullable=False)


class ProbeRun(Base):
    __tablename__ = "probe_runs"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    probe_set: Mapped[str] = mapped_column(String(80), nullable=False)
    repetition: Mapped[int] = mapped_column(Integer, nullable=False)
    engine_name: Mapped[str] = mapped_column(String(120), nullable=False)
    model_version: Mapped[str] = mapped_column(String(120), nullable=False)
    locale: Mapped[str | None] = mapped_column(String(60), nullable=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    category_filter: Mapped[str | None] = mapped_column(String(250), nullable=True)
    intent_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    response_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ProbeCandidate(Base):
    __tablename__ = "probe_candidates"
    __table_args__ = (
        Index("idx_probe_candidates_condition", "condition"),
        UniqueConstraint("probe_run_id", "product_id", name="uq_probe_candidate"),
    )

    probe_run_id: Mapped[str] = mapped_column(ForeignKey("probe_runs.id"), primary_key=True)
    product_id: Mapped[str] = mapped_column(ForeignKey("products.id"), primary_key=True)
    condition: Mapped[str] = mapped_column(String(20), nullable=False)
    rank_position: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieval_score: Mapped[float] = mapped_column(Float, nullable=False)
    lexical_score: Mapped[float] = mapped_column(Float, nullable=False)
    evidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    cited: Mapped[bool] = mapped_column(Boolean, nullable=False)


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (Index("idx_events_session", "session_id"),)

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), nullable=False)
    query_id: Mapped[str | None] = mapped_column(ForeignKey("queries.id"), nullable=True)
    product_id: Mapped[str | None] = mapped_column(ForeignKey("products.id"), nullable=True)
    event_type: Mapped[str] = mapped_column(String(60), nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class SurveyResponse(Base):
    __tablename__ = "survey_responses"

    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.id"), nullable=False, unique=True)
    answers_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    scale_scores_json: Mapped[dict[str, float | None]] = mapped_column(JSON, nullable=False)
    completed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
