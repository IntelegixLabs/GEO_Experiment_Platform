"""Deterministic fictional catalog used only for local development and demos."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.catalog import parse_catalog_csv, product_from_import
from app.core.config import get_settings
from app.core.experiment import assign_conditions, build_geo_bundle, utc_now
from app.models import Product


DEMO_PRODUCTS: list[dict[str, object]] = [
    {
        "id": "EL-001", "sku": "DEMO-EL-001", "title": "Aster Wave Wireless Earbuds", "brand": "Aster",
        "category": "Electronics", "price": 59.0, "description": "Compact Bluetooth earbuds for calls, music, and commuting.",
        "key_features": ["Bluetooth 5.3", "30-hour case battery", "IPX4 splash resistance", "USB-C charging"],
        "rating": 4.3, "review_count": 182, "availability": "In stock", "shipping": "Dispatches in 2 business days",
        "return_policy": "30-day returns", "pair_id": "electronics-earbuds",
    },
    {
        "id": "EL-002", "sku": "DEMO-EL-002", "title": "Beacon Sound Wireless Earbuds", "brand": "Beacon",
        "category": "Electronics", "price": 61.0, "description": "Compact Bluetooth earbuds for calls, music, and commuting.",
        "key_features": ["Bluetooth 5.3", "30-hour case battery", "IPX4 splash resistance", "USB-C charging"],
        "rating": 4.3, "review_count": 179, "availability": "In stock", "shipping": "Dispatches in 2 business days",
        "return_policy": "30-day returns", "pair_id": "electronics-earbuds",
    },
    {
        "id": "EL-003", "sku": "DEMO-EL-003", "title": "Aster Rise Adjustable Laptop Stand", "brand": "Aster",
        "category": "Electronics", "price": 32.0, "description": "Aluminum laptop stand designed for a more comfortable desk setup.",
        "key_features": ["Fits 11-17 inch laptops", "six height settings", "foldable aluminum frame", "non-slip pads"],
        "rating": 4.4, "review_count": 96, "availability": "In stock", "shipping": "Dispatches next business day",
        "return_policy": "30-day returns", "pair_id": "electronics-stand",
    },
    {
        "id": "EL-004", "sku": "DEMO-EL-004", "title": "Beacon Lift Adjustable Laptop Stand", "brand": "Beacon",
        "category": "Electronics", "price": 33.0, "description": "Aluminum laptop stand designed for a more comfortable desk setup.",
        "key_features": ["Fits 11-17 inch laptops", "six height settings", "foldable aluminum frame", "non-slip pads"],
        "rating": 4.4, "review_count": 94, "availability": "In stock", "shipping": "Dispatches next business day",
        "return_policy": "30-day returns", "pair_id": "electronics-stand",
    },
    {
        "id": "HM-001", "sku": "DEMO-HM-001", "title": "Aster Trail Insulated Water Bottle", "brand": "Aster",
        "category": "Home & Kitchen", "price": 24.0, "description": "750 ml stainless-steel bottle for everyday cold and hot drinks.",
        "key_features": ["750 ml capacity", "double-wall steel", "leak-resistant cap", "fits standard cup holders"],
        "rating": 4.5, "review_count": 218, "availability": "In stock", "shipping": "Dispatches in 2 business days",
        "return_policy": "30-day returns", "pair_id": "kitchen-bottle",
    },
    {
        "id": "HM-002", "sku": "DEMO-HM-002", "title": "Beacon Roam Insulated Water Bottle", "brand": "Beacon",
        "category": "Home & Kitchen", "price": 25.0, "description": "750 ml stainless-steel bottle for everyday cold and hot drinks.",
        "key_features": ["750 ml capacity", "double-wall steel", "leak-resistant cap", "fits standard cup holders"],
        "rating": 4.5, "review_count": 214, "availability": "In stock", "shipping": "Dispatches in 2 business days",
        "return_policy": "30-day returns", "pair_id": "kitchen-bottle",
    },
    {
        "id": "HM-003", "sku": "DEMO-HM-003", "title": "Aster Prep Glass Storage Set", "brand": "Aster",
        "category": "Home & Kitchen", "price": 38.0, "description": "Six-piece glass food-storage set for meal preparation and leftovers.",
        "key_features": ["six glass containers", "locking lids", "microwave-safe glass", "stackable design"],
        "rating": 4.2, "review_count": 121, "availability": "Limited stock", "shipping": "Dispatches in 3 business days",
        "return_policy": "30-day returns", "pair_id": "kitchen-storage",
    },
    {
        "id": "HM-004", "sku": "DEMO-HM-004", "title": "Beacon Keep Glass Storage Set", "brand": "Beacon",
        "category": "Home & Kitchen", "price": 39.0, "description": "Six-piece glass food-storage set for meal preparation and leftovers.",
        "key_features": ["six glass containers", "locking lids", "microwave-safe glass", "stackable design"],
        "rating": 4.2, "review_count": 118, "availability": "Limited stock", "shipping": "Dispatches in 3 business days",
        "return_policy": "30-day returns", "pair_id": "kitchen-storage",
    },
    {
        "id": "FT-001", "sku": "DEMO-FT-001", "title": "Aster Align Yoga Mat", "brand": "Aster",
        "category": "Fitness", "price": 29.0, "description": "Non-slip exercise mat for home yoga, stretching, and floor routines.",
        "key_features": ["6 mm cushioning", "non-slip surface", "carrying strap", "183 cm length"],
        "rating": 4.6, "review_count": 165, "availability": "In stock", "shipping": "Dispatches next business day",
        "return_policy": "30-day returns", "pair_id": "fitness-mat",
    },
    {
        "id": "FT-002", "sku": "DEMO-FT-002", "title": "Beacon Balance Yoga Mat", "brand": "Beacon",
        "category": "Fitness", "price": 30.0, "description": "Non-slip exercise mat for home yoga, stretching, and floor routines.",
        "key_features": ["6 mm cushioning", "non-slip surface", "carrying strap", "183 cm length"],
        "rating": 4.6, "review_count": 163, "availability": "In stock", "shipping": "Dispatches next business day",
        "return_policy": "30-day returns", "pair_id": "fitness-mat",
    },
    {
        "id": "PC-001", "sku": "DEMO-PC-001", "title": "Aster Calm Facial Cleanser", "brand": "Aster",
        "category": "Personal Care", "price": 18.0, "description": "Fragrance-free gel cleanser for a simple daily face-care routine.",
        "key_features": ["150 ml bottle", "fragrance-free", "gel formula", "pump dispenser"],
        "rating": 4.1, "review_count": 87, "availability": "In stock", "shipping": "Dispatches in 2 business days",
        "return_policy": "30-day returns", "pair_id": "care-cleanser",
    },
    {
        "id": "PC-002", "sku": "DEMO-PC-002", "title": "Beacon Gentle Facial Cleanser", "brand": "Beacon",
        "category": "Personal Care", "price": 19.0, "description": "Fragrance-free gel cleanser for a simple daily face-care routine.",
        "key_features": ["150 ml bottle", "fragrance-free", "gel formula", "pump dispenser"],
        "rating": 4.1, "review_count": 85, "availability": "In stock", "shipping": "Dispatches in 2 business days",
        "return_policy": "30-day returns", "pair_id": "care-cleanser",
    },
]


def seed_demo_catalog(db: Session) -> int:
    """Seed an empty database from an opt-in CSV or the fictional demo catalog.

    ``GEO_SEED_CATALOG_CSV`` is intentionally an explicit local configuration
    switch. It makes a verified example or an approved study catalog available
    on first startup, while never replacing a catalog that already has data.
    """

    if db.scalar(select(1).select_from(Product).limit(1)):
        return 0
    settings = get_settings()
    if settings.seed_catalog_csv is not None:
        catalog_path = settings.seed_catalog_csv
        if not catalog_path.is_file():
            raise RuntimeError(
                f"GEO_SEED_CATALOG_CSV points to a missing CSV file: {catalog_path}. "
                "Correct the local .env value or remove it to use the fictional demo catalog."
            )
        records = parse_catalog_csv(catalog_path.read_text(encoding="utf-8-sig"))
        assign_conditions(records)
        for record in records:
            if record.get("condition") == "GEO_OPTIMIZED":
                record["geo_bundle"] = build_geo_bundle(record)
            db.add(product_from_import(record, created_at=utc_now()))
        db.commit()
        
        # Index into Vector DB
        try:
            from app.services.vector_db import index_products
            index_products(records)
        except Exception as e:
            print("Failed to index products:", e)
            
        return len(records)

    records = [dict(product) for product in DEMO_PRODUCTS]
    assign_conditions(records)
    for record in records:
        record["currency"] = str(record.get("currency") or "INR")
        if record.get("condition") == "GEO_OPTIMIZED":
            record["geo_bundle"] = build_geo_bundle(record)
        db.add(Product(**record, created_at=utc_now()))
    db.commit()
    
    # Index into Vector DB
    try:
        from app.services.vector_db import index_products
        index_products(records)
    except Exception as e:
        print("Failed to index products:", e)
        
    return len(records)
