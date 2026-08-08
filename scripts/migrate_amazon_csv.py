"""Migrate Amazon-Products.csv into CockroachDB.

Usage:
    cd backend
    python -m scripts.migrate_amazon_csv

Environment: reads SQLALCHEMY_DATABASE_URL from .env (via app.core.config).
"""

import csv
import os
import re
import sys
import time
from pathlib import Path

# Ensure backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, select

from app.core.config import get_settings
from app.core.experiment import assign_conditions, build_geo_bundle, utc_now
from app.db.session import SessionLocal
from app.models.study import Product

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BATCH_SIZE = int(os.environ.get("MIGRATE_BATCH_SIZE", "5000"))  # Increased batch size for bulk insert
VECTOR_INDEX_BATCH = int(os.environ.get("VECTOR_INDEX_BATCH", "500"))

settings = get_settings()
CSV_PATH = Path(settings.amazon_catalog_csv) if getattr(settings, "amazon_catalog_csv", None) else None
if CSV_PATH and not CSV_PATH.is_absolute():
    CSV_PATH = (Path(__file__).resolve().parent.parent / CSV_PATH).resolve()


def _clean_price(raw: str | None) -> float | None:
    """Remove currency symbols, commas, spaces and convert to float."""
    if not raw or not raw.strip():
        return None
    cleaned = re.sub(r"[₹,\s]", "", raw.strip())
    try:
        return float(cleaned)
    except ValueError:
        return None


def _clean_ratings_count(raw: str | None) -> int | None:
    """Remove commas from ratings count like '2,255'."""
    if not raw or not raw.strip():
        return None
    cleaned = raw.replace(",", "").strip()
    try:
        return int(float(cleaned))
    except ValueError:
        return None


def _clean_rating(raw: str | None) -> float | None:
    if not raw or not raw.strip():
        return None
    try:
        return float(raw.strip())
    except ValueError:
        return None


def parse_amazon_row(row: dict, row_number: int) -> dict | None:
    """Map an Amazon CSV row to our Product schema."""
    name = (row.get("name") or "").strip()
    main_cat = (row.get("main_category") or "").strip()
    sub_cat = (row.get("sub_category") or "").strip()

    if not name or not main_cat:
        return None

    # Use sub_category as the participant-facing category, fallback to main_category
    category = sub_cat or main_cat

    product_id = f"AMZ-{row_number:06d}"

    discount_price = _clean_price(row.get("discount_price"))
    actual_price = _clean_price(row.get("actual_price"))

    return {
        "id": product_id,
        "sku": None,
        "title": name[:500],
        "brand": None,
        "category": category[:250],
        "main_category": main_cat[:250] if main_cat else None,
        "sub_category": sub_cat[:250] if sub_cat else None,
        "price": discount_price,
        "discount_price": discount_price,
        "actual_price": actual_price,
        "currency": "INR",
        "description": "",
        "key_features": [],
        "rating": _clean_rating(row.get("ratings")),
        "review_count": _clean_ratings_count(row.get("no_of_ratings")),
        "availability": "In stock",
        "shipping": "",
        "return_policy": "",
        "source_url": (row.get("link") or "")[:2000],
        "image_url": (row.get("image") or "")[:2000],
        "condition": None,
        "pair_id": None,
        "source_dataset": "Amazon-Products.csv",
        "source_row_number": row_number,
        "source_product_key": f"amazon-{row_number}",
        "data_quality_flags": [],
    }


def main():
    if not CSV_PATH or not CSV_PATH.is_file():
        print(f"ERROR: CSV not found at {CSV_PATH}")
        print("Set GEO_AMAZON_CATALOG_CSV in your .env file.")
        sys.exit(1)

    print(f"CSV path: {CSV_PATH}")
    print(f"Batch size: {BATCH_SIZE}")

    # Check existing products
    with SessionLocal() as db:
        existing_count = db.execute(select(func.count()).select_from(Product)).scalar_one()
        amazon_count = db.execute(
            select(func.count()).select_from(Product).where(Product.source_dataset == "Amazon-Products.csv")
        ).scalar_one()
        print(f"Existing products in DB: {existing_count} (Amazon: {amazon_count})")

    if amazon_count > 0:
        resp = input(f"Found {amazon_count} Amazon products already. Delete and re-import? (y/N): ")
        if resp.strip().lower() != "y":
            print("Aborted.")
            sys.exit(0)
        print("Deleting existing Amazon products...")
        with SessionLocal() as db:
            db.execute(Product.__table__.delete().where(Product.source_dataset == "Amazon-Products.csv"))
            db.commit()
        print("Deleted.")

    # Read and insert in batches
    t0 = time.time()
    total_inserted = 0
    total_skipped = 0
    batch = []

    print(f"Reading {CSV_PATH.name}...")
    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_number, raw_row in enumerate(reader, start=1):
            if row_number % 10000 == 0:
                print(f"  ... Parsed {row_number:,} rows from CSV")
            
            record = parse_amazon_row(raw_row, row_number)
            if record is None:
                total_skipped += 1
                continue
            batch.append(record)

            if len(batch) >= BATCH_SIZE:
                try:
                    _insert_batch(batch)
                except Exception as e:
                    print(f"ERROR inserting batch: {e}")
                    raise
                total_inserted += len(batch)
                batch = []

                elapsed = time.time() - t0
                rate = total_inserted / elapsed if elapsed > 0 else 0
                print(f"  ... {total_inserted:,} inserted ({rate:.0f} rows/s)")

    # Final batch
    if batch:
        _insert_batch(batch)
        total_inserted += len(batch)
        elapsed = time.time() - t0
        rate = total_inserted / elapsed if elapsed > 0 else 0
        print(f"  ... {total_inserted:,} inserted ({rate:.0f} rows/s)")

    elapsed = time.time() - t0
    print(f"\nDone! Inserted {total_inserted:,} products in {elapsed:.1f}s")
    print(f"Skipped {total_skipped:,} rows (missing name/category)")
    print("Migration complete!")


BATCH_SIZE = int(os.environ.get("MIGRATE_BATCH_SIZE", "500"))

from app.core.experiment import factual_geo_bundle
from app.core.config import CONDITIONS
from collections import defaultdict

def _local_assign_conditions(products: list[dict]):
    """Assign conditions deterministically without invoking LLM."""
    paired = defaultdict(list)
    remainder = defaultdict(list)
    for product in products:
        if product.get("condition") in CONDITIONS:
            continue
        if product.get("pair_id"):
            paired[str(product["pair_id"])].append(product)
        else:
            remainder[str(product.get("category") or "Uncategorised")].append(product)
    
    for group_key, group in [*paired.items(), *remainder.items()]:
        for index, product in enumerate(sorted(group, key=lambda item: (str(item.get("title", "")).lower(), str(item.get("id", ""))))):
            parity = sum(ord(char) for char in group_key) % 2
            product["condition"] = CONDITIONS[(index + parity) % 2]


def _insert_batch(batch: list[dict]):
    """Insert a batch of product records using bulk_insert_mappings."""
    t_start = time.time()
    _local_assign_conditions(batch)
    now = utc_now()
    
    # Prepare dictionaries for bulk insert
    insert_data = []
    for record in batch:
        if record.get("condition") == "GEO_OPTIMIZED":
            geo_bundle = factual_geo_bundle(record)
        else:
            geo_bundle = None

        insert_data.append({
            "id": record["id"],
            "sku": record.get("sku"),
            "title": record["title"],
            "brand": record.get("brand"),
            "category": record["category"],
            "main_category": record.get("main_category"),
            "sub_category": record.get("sub_category"),
            "price": record.get("price"),
            "discount_price": record.get("discount_price"),
            "actual_price": record.get("actual_price"),
            "currency": record.get("currency", "INR"),
            "description": record.get("description", ""),
            "key_features": record.get("key_features", []),
            "rating": record.get("rating"),
            "review_count": record.get("review_count"),
            "availability": record.get("availability"),
            "shipping": record.get("shipping"),
            "return_policy": record.get("return_policy"),
            "source_url": record.get("source_url"),
            "image_url": record.get("image_url"),
            "condition": record["condition"],
            "pair_id": record.get("pair_id"),
            "source_dataset": record.get("source_dataset"),
            "source_row_number": record.get("source_row_number"),
            "source_product_key": record.get("source_product_key"),
            "data_quality_flags": record.get("data_quality_flags", []),
            "geo_bundle": geo_bundle,
            "created_at": now,
            "imported_at": now,
        })

    with SessionLocal() as db:
        db.bulk_insert_mappings(Product, insert_data)
        db.commit()
    t_end = time.time()
    print(f"      [db bulk insert of {len(batch)} rows took {t_end - t_start:.2f}s]")


if __name__ == "__main__":
    main()
