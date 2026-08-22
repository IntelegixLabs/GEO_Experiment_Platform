import hashlib
import json
import math
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session as DBSession, aliased

from app.core.catalog import parse_catalog_csv, product_from_import, product_record
from app.core.experiment import build_geo_bundle, utc_now, assign_conditions
from app.db.session import get_db
from app.models import Event, GEOOptimizationApplication, GEOOptimizationConfig, GEOOptimizedProduct, Product, Query, \
    Session, SurveyResponse, QueryCandidate
from app.schemas import CatalogImportCreate, GEOOptimizationApply
from app.schemas.study import (
    GEO_ASSIGNMENT_STRATEGIES,
    GEO_FEATURE_TOGGLE_DEFAULTS,
    GEO_MODEL_PROFILES,
    GEO_OPTIMIZATION_TARGETS,
    GEO_PARAMETER_WEIGHT_DEFAULTS,
    GEO_SCOPE_TYPES,
)
from app.services.text import stable_json

router = APIRouter(tags=["GEO study"])
GEO_OPTIMIZATION_BATCH_SIZE = 250
MAX_GEO_OPTIMIZATION_APPLY_PRODUCTS = 1_000_000


def _bad_request(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail=message)


def _geo_scope_filters(scope: dict[str, Any]) -> list[Any]:
    scope_type = scope.get("type", "all_catalog")
    if scope_type == "categories":
        categories = [str(value).strip().lower() for value in scope.get("categories", []) if str(value).strip()]
        return [func.lower(Product.category).in_(categories)]
    if scope_type == "product_ids":
        product_ids = [str(value).strip() for value in scope.get("product_ids", []) if str(value).strip()]
        return [Product.id.in_(product_ids)]
    if scope_type == "pair_ids":
        pair_ids = [str(value).strip() for value in scope.get("pair_ids", []) if str(value).strip()]
        return [Product.pair_id.in_(pair_ids)]
    return []


def _geo_scope_statement(scope: dict[str, Any]):
    statement = select(Product)
    filters = _geo_scope_filters(scope)
    if filters:
        statement = statement.where(*filters)
    limit = scope.get("limit")
    if limit is not None and scope.get("type") not in ("all_catalog", "categories"):
        statement = statement.limit(limit)
    return statement


def _geo_scope_summary(db: DBSession, scope: dict[str, Any]) -> dict[str, Any]:
    filters = _geo_scope_filters(scope)
    count_statement = select(func.count()).select_from(Product)
    category_statement = select(Product.category, func.count().label("products")).group_by(Product.category)
    if filters:
        count_statement = count_statement.where(*filters)
        category_statement = category_statement.where(*filters)
    raw_categories = [
        {"category": category or "Uncategorised", "products": int(products)}
        for category, products in db.execute(category_statement.order_by(Product.category)).all()
    ]
    limit = scope.get("limit")
    if limit is not None and scope.get("type") in ("all_catalog", "categories"):
        num_categories = len(raw_categories)
        if num_categories > 0:
            limit_per_cat = math.ceil(limit / num_categories)
            for c in raw_categories:
                c["products"] = min(c["products"], limit_per_cat)
            selected_count = sum(c["products"] for c in raw_categories)
        else:
            selected_count = 0
        categories = raw_categories
    else:
        selected_count = int(db.scalar(count_statement) or 0)
        categories = raw_categories

    if limit is not None:
        selected_count = min(selected_count, limit)

    return {
        "scope": scope,
        "selected_products": selected_count,
        "catalog_total": int(db.scalar(select(func.count()).select_from(Product)) or 0),
        "categories": categories,
        "apply_limit": MAX_GEO_OPTIMIZATION_APPLY_PRODUCTS,
        "requires_narrower_scope": selected_count > MAX_GEO_OPTIMIZATION_APPLY_PRODUCTS,
    }


def _geo_config_response(record: GEOOptimizationConfig | None = None, payload: GEOOptimizationApply | None = None) -> \
dict[str, Any]:
    if record is not None:
        return {
            "id": record.id,
            "revision": record.revision,
            "name": record.name,
            "optimization_target": record.optimization_target,
            "model_profile": record.model_profile,
            "model_name": record.model_name,
            "model_version": record.model_version,
            "treatment_percentage": record.treatment_percentage,
            "assignment_strategy": record.assignment_strategy,
            "random_seed": record.random_seed,
            "parameter_weights": dict(record.parameter_weights_json or {}),
            "feature_toggles": dict(record.feature_toggles_json or {}),
            "scope": dict(record.scope_json or {}),
            "is_active": record.is_active,
            "last_applied_at": record.last_applied_at.isoformat() if record.last_applied_at else None,
        }
    requested = payload or GEOOptimizationApply()
    return {
        "id": "draft",
        "revision": 0,
        "name": requested.name,
        "optimization_target": requested.optimization_target,
        "model_profile": requested.model_profile,
        "model_name": requested.model_name,
        "model_version": requested.model_version,
        "treatment_percentage": requested.treatment_percentage,
        "assignment_strategy": requested.assignment_strategy,
        "random_seed": requested.random_seed,
        "parameter_weights": dict(requested.parameter_weights),
        "feature_toggles": dict(requested.feature_toggles),
        "scope": requested.scope.model_dump(),
        "is_active": False,
        "last_applied_at": None,
    }


def _geo_options() -> dict[str, Any]:
    return {
        "model_profiles": list(GEO_MODEL_PROFILES),
        "optimization_targets": list(GEO_OPTIMIZATION_TARGETS),
        "assignment_strategies": list(GEO_ASSIGNMENT_STRATEGIES),
        "scope_types": list(GEO_SCOPE_TYPES),
        "parameter_weights": {
            key: {"default": value, "min": 0, "max": 20, "step": 0.1}
            for key, value in GEO_PARAMETER_WEIGHT_DEFAULTS.items()
        },
        "feature_toggles": dict(GEO_FEATURE_TOGGLE_DEFAULTS),
        "apply_limit": MAX_GEO_OPTIMIZATION_APPLY_PRODUCTS,
        "model_note": (
            "The current optimizer is a transparent factual generator and controlled retriever, not an external LLM. "
            "Weights and targets are persisted for reproducibility; the factual bundle builder remains integrity-gated."
        ),
    }


def _apply_geo_feature_toggles(bundle: dict[str, Any], toggles: dict[str, bool]) -> dict[str, Any]:
    configured = json.loads(json.dumps(bundle, default=str))
    if not toggles["factual_summary"]:
        configured["summary"] = ""
    if not toggles["structured_specifications"]:
        configured["specifications"] = {}
    if not toggles["claim_evidence_links"]:
        configured["claim_blocks"] = []
    if not toggles["factual_faq"]:
        configured["faq"] = []
    if not toggles["offer_details"]:
        configured["specifications"] = {
            key: value
            for key, value in dict(configured.get("specifications") or {}).items()
            if key not in {"Listed price", "Availability", "Shipping", "Returns"}
        }
        configured["claim_blocks"] = [
            item for item in configured.get("claim_blocks", [])
            if not (isinstance(item, dict) and item.get("claim_id") == "offer")
        ]
        configured["faq"] = [
            item for item in configured.get("faq", [])
            if not (isinstance(item, dict) and "delivery and return" in str(item.get("question", "")).lower())
        ]
        configured["evidence_markers"] = [
            marker for marker in configured.get("evidence_markers", [])
            if "offer" not in str(marker).lower() and "availability" not in str(marker).lower()
        ]
    if not toggles["agent_readable_provenance"]:
        configured["evidence_markers"] = [
            marker for marker in configured.get("evidence_markers", [])
            if "evidence" not in str(marker).lower() and "fact-linked" not in str(marker).lower()
        ]
    configured["feature_vector"] = {
        **dict(configured.get("feature_vector") or {}),
        "factual_summary": int(toggles["factual_summary"]),
        "structured_specifications": int(toggles["structured_specifications"]),
        "claim_evidence_links": int(toggles["claim_evidence_links"]),
        "factual_faq": int(toggles["factual_faq"]),
        "offer_details": int(toggles["offer_details"]),
        "agent_readable_provenance": int(toggles["agent_readable_provenance"]),
    }
    configured.pop("content_hash", None)
    configured["content_hash"] = hashlib.sha256(stable_json(configured).encode("utf-8")).hexdigest()
    return configured


def _geo_target_count(count: int, percentage: int) -> int:
    if count <= 0:
        return 0
    if count == 1:
        return int(percentage >= 50)
    target = int(math.floor((count * percentage / 100) + 0.5))
    return min(count - 1, max(1, target))


def _geo_modular_assignment(index: int, count: int, percentage: int, seed: str, category: str) -> str:
    target = _geo_target_count(count, percentage)
    if not target:
        return "CONTROL"
    if target == count:
        return "GEO_OPTIMIZED"
    digest = hashlib.sha256(f"{seed}:{category}".encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], "big") % count
    step = (int.from_bytes(digest[8:16], "big") % max(1, count - 1)) + 1
    while math.gcd(step, count) != 1:
        step = (step % count) + 1
    return "GEO_OPTIMIZED" if ((index * step + offset) % count) < target else "CONTROL"


def _geo_assignment_stream(db: DBSession, scope: dict[str, Any], payload: GEOOptimizationApply):
    statement = _geo_scope_statement(scope)
    if payload.assignment_strategy == "matched_pairs":
        filters = _geo_scope_filters(scope)
        pair_statement = select(Product.pair_id, func.count().label("products")).group_by(Product.pair_id)
        if filters:
            pair_statement = pair_statement.where(*filters)
        invalid_pairs = [
            pair_id
            for pair_id, products in db.execute(pair_statement).all()
            if not pair_id or int(products) != 2
        ]
        if invalid_pairs:
            raise _bad_request(
                "Matched-pair assignment requires every selected pair to contain exactly two products. "
                "Use pair_ids or choose stratified_split for unpaired catalog rows."
            )
        current_pair: str | None = None
        pair_rows: list[Product] = []
        for product in db.scalars(statement.order_by(Product.pair_id, Product.id)).all():
            if current_pair is not None and product.pair_id != current_pair:
                parity = int(hashlib.sha256(f"{payload.random_seed}:{current_pair}".encode("utf-8")).hexdigest(),
                             16) % 2
                yield pair_rows[0], ("GEO_OPTIMIZED" if parity else "CONTROL")
                yield pair_rows[1], ("CONTROL" if parity else "GEO_OPTIMIZED")
                pair_rows = []
            current_pair = product.pair_id
            pair_rows.append(product)
        if pair_rows:
            parity = int(hashlib.sha256(f"{payload.random_seed}:{current_pair}".encode("utf-8")).hexdigest(), 16) % 2
            yield pair_rows[0], ("GEO_OPTIMIZED" if parity else "CONTROL")
            yield pair_rows[1], ("CONTROL" if parity else "GEO_OPTIMIZED")
        return

    from collections import defaultdict
    filters = _geo_scope_filters(scope)
    counts_statement = select(Product.category, func.count().label("products")).group_by(Product.category)
    if filters:
        counts_statement = counts_statement.where(*filters)
    counts = {str(category or ""): int(products) for category, products in db.execute(counts_statement).all()}

    limit = scope.get("limit")
    if limit is not None and scope.get("type") in ("all_catalog", "categories"):
        num_categories = len(counts)
        if num_categories > 0:
            limit_per_category = math.ceil(limit / num_categories)
            subq = select(
                Product,
                func.row_number().over(partition_by=Product.category, order_by=Product.id).label("rn")
            )
            if filters:
                subq = subq.where(*filters)
            subq = subq.subquery()
            ProductAlias = aliased(Product, subq)
            statement = select(ProductAlias).where(subq.c.rn <= limit_per_category)
            order_by_args = (ProductAlias.category, ProductAlias.id)
        else:
            order_by_args = (Product.category, Product.id)
    else:
        order_by_args = (Product.category, Product.id)

    total_seen = 0
    for product in db.scalars(statement.order_by(*order_by_args)).all():
        if limit is not None and total_seen >= limit:
            break
        condition = "GEO_OPTIMIZED" if ((
                                                    total_seen * payload.treatment_percentage) % 100) < payload.treatment_percentage else "CONTROL"
        total_seen += 1
        yield product, condition


def _geo_assignment_prediction(db: DBSession, scope: dict[str, Any], payload: GEOOptimizationApply) -> tuple[int, int]:
    filters = _geo_scope_filters(scope)
    if payload.assignment_strategy == "matched_pairs":
        statement = select(Product.pair_id, func.count().label("products")).group_by(Product.pair_id)
        if filters:
            statement = statement.where(*filters)
        rows = db.execute(statement).all()
        if any(not pair_id or int(products) != 2 for pair_id, products in rows):
            raise _bad_request(
                "Matched-pair assignment requires every selected pair to contain exactly two products. "
                "Use pair_ids or choose stratified_split for unpaired catalog rows."
            )
        geo_products = len(rows)
        return geo_products, geo_products
    selected = int(db.scalar(select(func.count()).select_from(Product).where(*filters)) or 0) if filters else int(
        db.scalar(select(func.count()).select_from(Product)) or 0
    )
    if scope.get("limit"):
        selected = min(selected, int(scope.get("limit")))
    geo_products = int(round(selected * (payload.treatment_percentage / 100.0)))
    return selected - geo_products, geo_products


@router.get("/admin/geo-optimization/config")
def geo_optimization_config(db: DBSession = Depends(get_db)) -> dict[str, Any]:
    active = db.scalar(
        select(GEOOptimizationConfig).where(GEOOptimizationConfig.is_active.is_(True)).order_by(
            GEOOptimizationConfig.revision.desc())
    )
    config_payload = _geo_config_response(active)
    return {
        "config": config_payload,
        "options": _geo_options(),
        "scope_summary": _geo_scope_summary(db, config_payload["scope"]),
    }


def _apply_geo_optimization_sync(payload: GEOOptimizationApply, db: DBSession) -> dict[str, Any]:
    scope = payload.scope.model_dump()
    scope_summary = _geo_scope_summary(db, scope)
    selected_products = scope_summary["selected_products"]
    if selected_products == 0:
        raise _bad_request("The selected scope contains no products.")
    if scope_summary.get("requires_narrower_scope"):
        raise _bad_request(
            f"This scope contains {selected_products:,} products. Apply is limited to {MAX_GEO_OPTIMIZATION_APPLY_PRODUCTS:,} products per wave; select categories, product IDs, or pair IDs. A dry run remains available for this scope.")
    outcome_counts = {
        name: int(db.scalar(select(func.count()).select_from(model)) or 0)
        for name, model in (("events", Event), ("queries", Query), ("surveys", SurveyResponse))
    }
    previous_assignments = {
        "geo_optimized": int(
            db.scalar(select(func.count()).select_from(Product).where(Product.condition == "GEO_OPTIMIZED")) or 0),
        "control": int(db.scalar(select(func.count()).select_from(Product).where(Product.condition == "CONTROL")) or 0)
    }
    control_products = 0
    geo_optimized_products = 0
    run_id = f"GOA-{uuid.uuid4().hex}"
    updated_products = 0
    integrity_failures: list[dict[str, Any]] = []
    skipped_product_ids: list[str] = []
    updated_records_for_index: list[dict[str, Any]] = []
    for product, condition in _geo_assignment_stream(db, scope, payload):
        assigned_condition = condition
        bundle = None
        if condition == "GEO_OPTIMIZED":
            try:
                bundle = build_geo_bundle(product_record(product))
                bundle = _apply_geo_feature_toggles(bundle, payload.feature_toggles)
                bundle["run_id"] = run_id
                geo_optimized_products += 1
            except Exception as e:
                integrity_failures.append({"product_id": product.id, "error": str(e)})
                assigned_condition = "CONTROL"
                bundle = {"run_id": run_id, "status": "control_fallback"}
                control_products += 1
        else:
            bundle = {"run_id": run_id, "status": "control"}
            control_products += 1

        if not payload.dry_run:
            product.condition = assigned_condition
            product.geo_bundle = bundle
            snapshot = product_record(product)
            snapshot["condition"] = assigned_condition
            snapshot["geo_bundle"] = bundle or {}
            snapshot["run_id"] = run_id
            gop = GEOOptimizedProduct(
                id=f"GOP-{uuid.uuid4().hex}",
                run_id=run_id,
                product_id=product.id,
                condition=assigned_condition,
                geo_bundle=bundle,
                product_snapshot_json=snapshot,
                created_at=utc_now(),
            )
            db.add(gop)
            updated_products += 1
            updated_records_for_index.append(snapshot)

    now = utc_now()
    if not payload.dry_run:
        revision = int(db.scalar(select(func.coalesce(func.max(GEOOptimizationConfig.revision), 0))) or 0) + 1
        db.execute(
            update(GEOOptimizationConfig).where(GEOOptimizationConfig.is_active.is_(True)).values(is_active=False))
        config_record = GEOOptimizationConfig(
            id=f"GOC-{uuid.uuid4().hex}", revision=revision, name=payload.name,
            optimization_target=payload.optimization_target, model_profile=payload.model_profile,
            model_name=payload.model_name, model_version=payload.model_version,
            treatment_percentage=payload.treatment_percentage, assignment_strategy=payload.assignment_strategy,
            random_seed=payload.random_seed, parameter_weights_json=payload.parameter_weights,
            feature_toggles_json=payload.feature_toggles, scope_json=scope, is_active=True,
            created_at=now, updated_at=now, last_applied_at=now,
        )
        db.add(config_record)
        db.flush()
        config_snapshot = _geo_config_response(config_record)
        application = GEOOptimizationApplication(
            id=run_id, config_id=config_record.id, config_snapshot_json=config_snapshot,
            scope_summary_json=scope_summary, previous_assignment_json=previous_assignments,
            application_summary_json={"selected_products": selected_products, "updated_products": updated_products,
                                      "control_products": control_products,
                                      "geo_optimized_products": geo_optimized_products,
                                      "integrity_failure_count": len(integrity_failures)},
            safety_notes_json=[
                "Factual GEO bundles are generated from catalog facts only.",
                "Parameter weights and optimization target are persisted study settings; they do not invoke an external LLM.",
                *(
                    ["This configuration was applied after outcome data existed; do not pool the changed wave with prior observations."] if any(
                        outcome_counts.values()) else []),
            ], created_at=now,
        )
        db.add(application)
        db.commit()
        if updated_records_for_index:
            try:
                from app.services.vector_db import clear_collection, index_products
                clear_collection()
                index_products(updated_records_for_index)
            except Exception as e:
                print(f"Vector DB indexing error: {e}")
    else:
        config_record = GEOOptimizationConfig(
            id=f"GOC-{uuid.uuid4().hex}", revision=0, name=payload.name,
            optimization_target=payload.optimization_target, model_profile=payload.model_profile,
            model_name=payload.model_name, model_version=payload.model_version,
            treatment_percentage=payload.treatment_percentage, assignment_strategy=payload.assignment_strategy,
            random_seed=payload.random_seed, parameter_weights_json=payload.parameter_weights,
            feature_toggles_json=payload.feature_toggles, scope_json=scope, is_active=False,
            created_at=now, updated_at=now, last_applied_at=now,
        )

    return {
        "config": _geo_config_response(config_record),
        "application": {
            "run_id": run_id,
            "config_id": config_record.id,
            "version": config_record.revision,
            "dry_run": payload.dry_run,
            "selected_products": selected_products,
            "updated_products": updated_products,
            "control_products": control_products,
            "geo_optimized_products": geo_optimized_products,
            "actual_treatment_percentage": round(geo_optimized_products * 100 / max(1, selected_products), 2),
            "scope": scope_summary,
            "assignment_strategy": payload.assignment_strategy,
            "integrity_failures": len(integrity_failures),
            "integrity_failure_details": integrity_failures,
            "skipped_product_ids": skipped_product_ids,
            "warning": "",
        },
    }


def _apply_geo_optimization_stream(payload: GEOOptimizationApply, db: DBSession):
    try:
        scope = payload.scope.model_dump()
        scope_summary = _geo_scope_summary(db, scope)
        selected_products = scope_summary["selected_products"]
        if selected_products == 0:
            yield json.dumps({"status": "error", "message": "The selected scope contains no products."}) + "\n"
            return
        if scope_summary.get("requires_narrower_scope"):
            yield json.dumps({"status": "error",
                              "message": f"This scope contains {selected_products:,} products. Apply is limited to {MAX_GEO_OPTIMIZATION_APPLY_PRODUCTS:,} products per wave."}) + "\n"
            return
        outcome_counts = {
            name: int(db.scalar(select(func.count()).select_from(model)) or 0)
            for name, model in (("events", Event), ("queries", Query), ("surveys", SurveyResponse))
        }
        previous_assignments = {
            "geo_optimized": int(
                db.scalar(select(func.count()).select_from(Product).where(Product.condition == "GEO_OPTIMIZED")) or 0),
            "control": int(
                db.scalar(select(func.count()).select_from(Product).where(Product.condition == "CONTROL")) or 0)
        }
        control_products = 0
        geo_optimized_products = 0
        run_id = f"GOA-{uuid.uuid4().hex}"
        updated_products = 0
        integrity_failures: list[dict[str, Any]] = []
        skipped_product_ids: list[str] = []
        updated_records_for_index = []
        batch_count = 0
        for product, condition in _geo_assignment_stream(db, scope, payload):
            bundle = None
            assigned_condition = condition
            if condition == "GEO_OPTIMIZED":
                try:
                    bundle = build_geo_bundle(product_record(product))
                    bundle = _apply_geo_feature_toggles(bundle, payload.feature_toggles)
                    bundle["run_id"] = run_id
                    geo_optimized_products += 1
                except Exception as e:
                    integrity_failures.append({"product_id": product.id, "error": str(e)})
                    assigned_condition = "CONTROL"
                    bundle = {"run_id": run_id, "status": "control_fallback"}
                    control_products += 1
            else:
                bundle = {"run_id": run_id, "status": "control"}
                control_products += 1
            product.condition = assigned_condition
            product.geo_bundle = bundle
            snapshot = product_record(product)
            snapshot["condition"] = assigned_condition
            snapshot["geo_bundle"] = bundle or {}
            snapshot["run_id"] = run_id
            gop = GEOOptimizedProduct(
                id=f"GOP-{uuid.uuid4().hex}",
                run_id=run_id,
                product_id=product.id,
                condition=assigned_condition,
                geo_bundle=bundle,
                product_snapshot_json=snapshot,
                created_at=utc_now(),
            )
            db.add(gop)
            updated_products += 1
            batch_count += 1
            updated_records_for_index.append(snapshot)
            if batch_count >= GEO_OPTIMIZATION_BATCH_SIZE:
                db.flush()
                batch_count = 0
            yield json.dumps({
                "status": "progress",
                "processed": updated_products,
                "total": selected_products,
                "latest_product": product.title,
                "condition": assigned_condition,
                "geo_bundle": bundle,
                "original_record": snapshot
            }) + "\n"
        db.flush()
        now = utc_now()
        revision = int(db.scalar(select(func.coalesce(func.max(GEOOptimizationConfig.revision), 0))) or 0) + 1
        db.execute(
            update(GEOOptimizationConfig).where(GEOOptimizationConfig.is_active.is_(True)).values(is_active=False))
        config_record = GEOOptimizationConfig(
            id=f"GOC-{uuid.uuid4().hex}", revision=revision, name=payload.name,
            optimization_target=payload.optimization_target, model_profile=payload.model_profile,
            model_name=payload.model_name, model_version=payload.model_version,
            treatment_percentage=payload.treatment_percentage, assignment_strategy=payload.assignment_strategy,
            random_seed=payload.random_seed, parameter_weights_json=payload.parameter_weights,
            feature_toggles_json=payload.feature_toggles, scope_json=scope, is_active=True,
            created_at=now, updated_at=now, last_applied_at=now,
        )
        db.add(config_record)
        db.flush()
        config_snapshot = _geo_config_response(config_record)
        application = GEOOptimizationApplication(
            id=run_id, config_id=config_record.id, config_snapshot_json=config_snapshot,
            scope_summary_json=scope_summary, previous_assignment_json=previous_assignments,
            application_summary_json={"selected_products": selected_products, "updated_products": updated_products,
                                      "control_products": control_products,
                                      "geo_optimized_products": geo_optimized_products,
                                      "integrity_failure_count": len(integrity_failures)},
            safety_notes_json=[
                "Factual GEO bundles are generated from catalog facts only.",
                "Parameter weights and optimization target are persisted study settings; they do not invoke an external LLM.",
                *(
                    ["This configuration was applied after outcome data existed; do not pool the changed wave with prior observations."] if any(
                        outcome_counts.values()) else []),
            ], created_at=now,
        )
        db.add(application)
        try:
            db.commit()
        except Exception as e:
            db.rollback()
            yield json.dumps({"status": "error", "message": str(e)}) + "\n"
            return
        if updated_records_for_index:
            try:
                from app.services.vector_db import index_products
                index_products(updated_records_for_index)
            except Exception as e:
                print(f"Vector DB indexing error: {e}")
        yield json.dumps({
            "status": "complete",
            "config": _geo_config_response(config_record),
            "application": {
                "run_id": application.id,
                "config_id": config_record.id,
                "version": config_record.revision,
                "dry_run": False,
                "selected_products": selected_products,
                "updated_products": updated_products,
                "control_products": control_products,
                "geo_optimized_products": geo_optimized_products,
                "actual_treatment_percentage": round(geo_optimized_products * 100 / max(1, selected_products), 2),
                "scope": scope_summary,
                "assignment_strategy": payload.assignment_strategy,
                "integrity_failures": len(integrity_failures),
                "integrity_failure_details": integrity_failures,
                "skipped_product_ids": skipped_product_ids,
                "warning": (
                    "Configuration and audit snapshot were saved. The factual generator is integrity-gated; no external LLM was invoked."
                    if not any(outcome_counts.values()) else
                    "Configuration was saved after existing outcomes. Keep the resulting wave separate from prior observations."
                ),
            },
        }) + "\n"
    except Exception as e:
        db.rollback()
        yield json.dumps({"status": "error", "message": str(e)}) + "\n"


@router.post("/admin/geo-optimization/apply", status_code=201, response_model=None)
def apply_geo_optimization(payload: GEOOptimizationApply, stream: bool = False, db: DBSession = Depends(get_db)):
    if not stream or payload.dry_run:
        return _apply_geo_optimization_sync(payload, db)
    return StreamingResponse(_apply_geo_optimization_stream(payload, db), media_type="application/x-ndjson")


@router.delete("/admin/vector-db/clear", status_code=200)
def clear_vector_db(db: DBSession = Depends(get_db)):
    from app.services.vector_db import clear_collection
    clear_collection()
    return {"status": "success", "message": "Vector DB cleared"}


@router.post("/admin/products/import", status_code=201)
def import_catalog(payload: CatalogImportCreate, db: DBSession = Depends(get_db)) -> dict[str, Any]:
    try:
        records = parse_catalog_csv(payload.csv)
        assign_conditions(records)
        now = utc_now()
        for record in records:
            db.add(product_from_import(record, created_at=now))
        db.commit()
    except (ValueError, IntegrityError) as error:
        db.rollback()
        message = str(error.orig) if isinstance(error, IntegrityError) else str(error)
        raise _bad_request(message) from error
    return {
        "imported": len(records),
        "control": sum(record["condition"] == "CONTROL" for record in records),
        "geo_optimized": sum(record["condition"] == "GEO_OPTIMIZED" for record in records),
        "message": "Products were stratified by pair ID or category. Verify balance before recruiting participants.",
    }


@router.get("/admin/geo-optimization/runs")
def list_geo_optimization_runs(db: DBSession = Depends(get_db)) -> list[dict[str, Any]]:
    runs = db.scalars(select(GEOOptimizationApplication).order_by(GEOOptimizationApplication.created_at.desc())).all()
    result = []
    for run in runs:
        summary = run.application_summary_json or {}
        result.append({
            "run_id": run.id,
            "config_id": run.config_id,
            "created_at": run.created_at.isoformat() if run.created_at else None,
            "selected_products": summary.get("selected_products", 0),
            "updated_products": summary.get("updated_products", 0),
            "geo_optimized_products": summary.get("geo_optimized_products", 0),
            "control_products": summary.get("control_products", 0),
            "scope_summary": run.scope_summary_json or {},
        })
    return result


@router.get("/admin/geo-optimization/runs/{run_id}/products")
def get_run_optimized_products(run_id: str, db: DBSession = Depends(get_db)) -> list[dict[str, Any]]:
    records = db.scalars(select(GEOOptimizedProduct).where(GEOOptimizedProduct.run_id == run_id)).all()
    return [r.product_snapshot_json for r in records]


@router.get("/admin/vectordb/status")
def vectordb_status() -> dict[str, Any]:
    from app.services.vector_db import get_vectordb_status
    return get_vectordb_status()


@router.get("/admin/vectordb/products")
def vectordb_products(
        page: int = 1,
        limit: int = 20,
        query: str | None = None,
        category_filter: str | None = None,
) -> dict[str, Any]:
    from app.services.vector_db import get_indexed_products_page
    return get_indexed_products_page(
        page=page,
        limit=limit,
        query=query or None,
        category_filter=category_filter or None,
    )


@router.get("/admin/respondents/demographics")
def admin_respondents_demographics(db: DBSession = Depends(get_db)) -> dict[str, Any]:
    rows = db.execute(
        select(Session.country, func.count().label("count"))
        .where(Session.country.is_not(None))
        .where(Session.country != "")
        .group_by(Session.country)
        .order_by(func.count().desc())
    ).all()

    return {
        "demographics": [{"country": row.country, "count": int(row.count)} for row in rows]
    }


@router.get("/admin/respondents")
def admin_list_respondents(
        page: int = 1,
        limit: int = 20,
        query: str | None = None,
        cohort: str | None = None,
        db: DBSession = Depends(get_db),
) -> dict[str, Any]:
    page = max(1, page)
    limit = max(1, min(100, limit))
    offset = (page - 1) * limit

    statement = select(Session)
    if query and query.strip():
        qstr = f"%{query.strip().lower()}%"
        statement = statement.where(
            func.lower(Session.participant_code).like(qstr)
            | func.lower(Session.id).like(qstr)
            | func.lower(func.coalesce(Session.email, "")).like(qstr)
            | func.lower(func.coalesce(Session.country, "")).like(qstr)
        )
    if cohort and cohort.strip():
        statement = statement.where(Session.study_cohort == cohort.strip())

    total = db.scalar(select(func.count()).select_from(statement.subquery())) or 0
    sessions = db.scalars(statement.order_by(Session.started_at.desc()).offset(offset).limit(limit)).all()

    session_ids = [s.id for s in sessions]
    queries_by_session: dict[str, int] = {}
    events_by_session: dict[str, int] = {}
    surveys_by_session: dict[str, bool] = {}

    if session_ids:
        q_rows = db.execute(
            select(Query.session_id, func.count()).where(Query.session_id.in_(session_ids)).group_by(Query.session_id)
        ).all()
        for sid, cnt in q_rows:
            queries_by_session[sid] = cnt

        e_rows = db.execute(
            select(Event.session_id, func.count()).where(Event.session_id.in_(session_ids)).group_by(Event.session_id)
        ).all()
        for sid, cnt in e_rows:
            events_by_session[sid] = cnt

        s_rows = db.execute(
            select(SurveyResponse.session_id).where(SurveyResponse.session_id.in_(session_ids))
        ).all()
        for (sid,) in s_rows:
            surveys_by_session[sid] = True

    items = []
    for s in sessions:
        items.append(
            {
                "session_id": s.id,
                "participant_code": s.participant_code,
                "email": s.email,
                "age": s.age,
                "country": s.country,
                "ai_familiarity": s.ai_familiarity,
                "study_cohort": s.study_cohort,
                "consent": s.consent,
                "started_at": s.started_at.isoformat() if s.started_at else None,
                "completed_at": s.completed_at.isoformat() if s.completed_at else None,
                "queries_count": queries_by_session.get(s.id, 0),
                "interactions_count": events_by_session.get(s.id, 0),
                "survey_status": "Completed" if s.id in surveys_by_session else "Pending",
            }
        )

    return {
        "items": items,
        "total": total,
        "page": page,
        "limit": limit,
        "pages": max(1, math.ceil(total / limit)) if total > 0 else 1,
    }


@router.get("/admin/respondents/{session_id}/activity")
def admin_respondent_activity(session_id: str, db: DBSession = Depends(get_db)) -> dict[str, Any]:
    session = db.get(Session, session_id)
    if not session:
        raise _bad_request("Respondent session not found.")

    queries = db.scalars(select(Query).where(Query.session_id == session_id).order_by(Query.created_at.asc())).all()
    events = db.scalars(select(Event).where(Event.session_id == session_id).order_by(Event.created_at.asc())).all()
    surveys = db.scalars(select(SurveyResponse).where(SurveyResponse.session_id == session_id)).all()

    return {
        "session": {
            "session_id": session.id,
            "participant_code": session.participant_code,
            "email": session.email,
            "age": session.age,
            "country": session.country,
            "ai_familiarity": session.ai_familiarity,
            "study_cohort": session.study_cohort,
            "consent": session.consent,
            "started_at": session.started_at.isoformat() if session.started_at else None,
            "completed_at": session.completed_at.isoformat() if session.completed_at else None,
        },
        "queries": [
            {
                "id": q.id,
                "query_text": q.query_text,
                "category_filter": q.category_filter,
                "created_at": q.created_at.isoformat() if q.created_at else None,
            }
            for q in queries
        ],
        "interactions": [
            {
                "id": e.id,
                "event_type": e.event_type,
                "product_id": e.product_id,
                "metadata": e.metadata_json,
                "created_at": e.created_at.isoformat() if e.created_at else None,
            }
            for e in events
        ],
        "surveys": [
            {
                "id": s.id,
                "answers": s.answers_json,
                "scale_scores": s.scale_scores_json,
                "completed_at": s.completed_at.isoformat() if s.completed_at else None,
            }
            for s in surveys
        ],
    }


@router.delete("/admin/respondents/{session_id}")
def admin_delete_respondent(session_id: str, db: DBSession = Depends(get_db)) -> dict[str, Any]:
    session = db.get(Session, session_id)
    if not session:
        raise _bad_request("Respondent session not found.")

    try:
        # Delete associated queries, events, and surveys
        # We delete manually in case foreign keys don't cascade on delete in the DB setup
        db.execute(QueryCandidate.__table__.delete().where(
            QueryCandidate.query_id.in_(select(Query.id).where(Query.session_id == session_id))
        ))
        db.execute(Event.__table__.delete().where(Event.session_id == session_id))
        db.execute(Query.__table__.delete().where(Query.session_id == session_id))
        db.execute(SurveyResponse.__table__.delete().where(SurveyResponse.session_id == session_id))

        # Finally delete the session
        db.delete(session)
        db.commit()
        return {"status": "success", "message": f"Respondent {session_id} and all activity deleted."}
    except Exception as e:
        db.rollback()
        raise _bad_request(f"Failed to delete respondent: {str(e)}")


class LoginRequest(BaseModel):
    username: str
    password: str


@router.post("/verify")
def verify_admin(request: LoginRequest) -> dict[str, bool]:
    from app.core.config import get_settings
    settings = get_settings()

    # If no credentials are required by the environment, allow access
    if not settings.admin_user and not settings.admin_password:
        return {"success": True}

    if request.username == settings.admin_user and request.password == settings.admin_password:
        return {"success": True}

    raise HTTPException(status_code=401, detail="Invalid username or password")
