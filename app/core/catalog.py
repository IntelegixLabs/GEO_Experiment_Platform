"""Catalog parsing and public/researcher serialization helpers."""

from __future__ import annotations

import csv
import io
import uuid
from decimal import Decimal
from typing import Any

from app.core.config import CONDITIONS
from app.core.experiment import as_list, build_geo_bundle
from app.models import Product


def _response_number(value: Any) -> float | None:
    """Return database numeric values in a JSON-friendly form.

    PostgreSQL returns ``Numeric`` columns as :class:`~decimal.Decimal`; using
    an explicit conversion keeps participant and export responses consistent
    with the existing float-shaped API contract.
    """

    if value is None or value == "":
        return None
    if isinstance(value, Decimal):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def product_record(product: Product) -> dict[str, Any]:
    return {
        "id": product.id,
        "sku": product.sku,
        "title": product.title,
        "brand": product.brand,
        "category": product.category,
        "main_category": product.main_category,
        "sub_category": product.sub_category,
        "price": _response_number(product.price),
        "discount_price": _response_number(product.discount_price),
        "actual_price": _response_number(product.actual_price),
        "currency": product.currency,
        "description": product.description or "",
        "key_features": list(product.key_features or []),
        "rating": product.rating,
        "review_count": product.review_count,
        "availability": product.availability or "",
        "shipping": product.shipping or "",
        "return_policy": product.return_policy or "",
        "source_url": product.source_url or "",
        "image_url": product.image_url or "",
        "condition": product.condition,
        "pair_id": product.pair_id,
        "source_dataset": product.source_dataset,
        "source_row_number": product.source_row_number,
        "source_product_key": product.source_product_key,
        "imported_at": product.imported_at.isoformat() if product.imported_at else None,
        "data_quality_flags": list(product.data_quality_flags or []),
        "geo_bundle": product.geo_bundle or {},
        "created_at": product.created_at.isoformat() if product.created_at else None,
    }


def public_product(product: Product | dict[str, Any]) -> dict[str, Any]:
    """Return only participant-safe fields and the actual page presentation."""

    record = product_record(product) if isinstance(product, Product) else dict(product)
    features = as_list(record.get("key_features"))
    bundle = record.get("geo_bundle") or {}
    return {
        "id": record.get("id"),
        "sku": record.get("sku"),
        "title": record.get("title"),
        "brand": record.get("brand"),
        "category": record.get("category"),
        "main_category": record.get("main_category"),
        "sub_category": record.get("sub_category"),
        "price": record.get("price"),
        "discount_price": record.get("discount_price"),
        "actual_price": record.get("actual_price"),
        "currency": record.get("currency") or "INR",
        "description": record.get("description") or "",
        "key_features": features,
        "rating": record.get("rating"),
        "review_count": record.get("review_count"),
        "availability": record.get("availability") or "",
        "shipping": record.get("shipping") or "",
        "return_policy": record.get("return_policy") or "",
        "source_url": record.get("source_url") or "",
        "image_url": record.get("image_url") or "",
        "product_page": {
            "description": record.get("description") or "",
            "key_features": features,
            "specifications": bundle.get("specifications", {}),
            "faq": bundle.get("faq", []),
            "claim_blocks": bundle.get("claim_blocks", []),
        },
    }


def researcher_product(product: Product, citations: int = 0, opportunities: int = 0) -> dict[str, Any]:
    return {**product_record(product), "citations": citations, "opportunities": opportunities}


def _number(value: Any, integer: bool = False) -> float | int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value)) if integer else float(value)
    except (TypeError, ValueError):
        return None


def _normalise_condition(value: Any) -> str | None:
    condition = str(value or "").upper().replace(" ", "_")
    if condition in {"GEO", "TREATMENT", "OPTIMIZED"}:
        return "GEO_OPTIMIZED"
    if condition in {"BASELINE", "CONTROL"}:
        return "CONTROL"
    return condition if condition in CONDITIONS else None


def parse_catalog_csv(csv_text: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(csv_text.strip()))
    if not reader.fieldnames:
        raise ValueError("The CSV needs a header row.")
    products: list[dict[str, Any]] = []
    for raw in reader:
        row = {
            str(key).strip().lower(): value.strip() if isinstance(value, str) else value
            for key, value in raw.items()
            if key
        }
        title = row.get("title") or row.get("product_name") or row.get("name")
        category = row.get("category")
        if not title or not category:
            raise ValueError("Each CSV row needs title (or product_name) and category.")
        products.append(
            {
                "id": row.get("id") or row.get("product_id") or f"P-{uuid.uuid4().hex[:10].upper()}",
                "sku": row.get("sku"),
                "title": title,
                "brand": row.get("brand"),
                "category": category,
                "price": _number(row.get("price")),
                "currency": row.get("currency") or "INR",
                "description": row.get("description") or "",
                "key_features": as_list(row.get("key_features") or row.get("features") or ""),
                "rating": _number(row.get("rating")),
                "review_count": _number(row.get("review_count"), integer=True),
                "availability": row.get("availability") or "",
                "shipping": row.get("shipping") or "",
                "return_policy": row.get("return_policy") or row.get("returns") or "",
                "source_url": row.get("source_url") or row.get("url") or "",
                "image_url": row.get("image_url") or "",
                "pair_id": row.get("pair_id") or "",
                "condition": _normalise_condition(row.get("condition")),
            }
        )
    if not products:
        raise ValueError("No product rows were found in the CSV.")
    return products


def product_from_import(record: dict[str, Any], *, created_at: Any) -> Product:
    values = dict(record)
    if values["condition"] == "GEO_OPTIMIZED":
        values["geo_bundle"] = build_geo_bundle(values)
    else:
        values["geo_bundle"] = None
    return Product(**values, created_at=created_at)
