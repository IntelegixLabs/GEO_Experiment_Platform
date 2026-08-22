import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

# Setup sys path for absolute imports
backend_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(backend_dir))

from app.core.experiment import build_geo_bundle
from app.db.session import SessionLocal
from app.models.study import Product, ProductVector
from app.services.vector_db import index_products
from sqlalchemy import select, func, delete
from app.core.experiment import utc_now

def _clean_price(raw: str | None) -> float | None:
    if not raw or not raw.strip():
        return None
    cleaned = raw.replace(",", "").replace("?", "").replace("₹", "").replace("$", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None

def _clean_ratings_count(raw: str | None) -> int | None:
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

def parse_row(row: dict, row_number: int, dataset_name: str) -> dict | None:
    name = (row.get("name") or "").strip()
    main_cat = (row.get("main_category") or "").strip()
    sub_cat = (row.get("sub_category") or "").strip()

    if not name or not main_cat:
        return None

    category = sub_cat or main_cat
    
    # We create a stable ID based on dataset name and row number to avoid collisions
    prefix = "".join([c for c in dataset_name if c.isalpha()]).upper()[:5]
    product_id = f"TRN-{prefix}-{row_number:06d}"

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
        "condition": "GEO_OPTIMIZED", # We want all to be trained
        "pair_id": None,
        "source_dataset": dataset_name,
        "source_row_number": row_number,
        "source_product_key": f"train-{dataset_name}-{row_number}",
        "data_quality_flags": [],
    }

def load_csv_products(csv_path: Path, limit: int = None):
    dataset_name = csv_path.name
    print(f"Parsing CSV {csv_path}...")
    products = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row_number, raw_row in enumerate(reader, start=1):
            record = parse_row(raw_row, row_number, dataset_name)
            if record:
                products.append(record)

    if limit:
        products = products[:limit]
        print(f"  Limited {dataset_name} to {limit} products.")
        
    return products, dataset_name

def process_all_products(all_products: list, dataset_names: list, treatment_percentage: int, cache_dir: Path):
    print(f"\nTotal valid products to train: {len(all_products)}")
    
    # 1. Exact Split Calculation
    num_treated = int(len(all_products) * (treatment_percentage / 100))
    print(f"Applying exact split: {num_treated} GEO_OPTIMIZED, {len(all_products) - num_treated} CONTROL")
    
    # Sort deterministically by ID, then assign condition
    all_products.sort(key=lambda p: p["id"])
    for i, prod in enumerate(all_products):
        if i < num_treated:
            prod["condition"] = "GEO_OPTIMIZED"
        else:
            prod["condition"] = "CONTROL"
    
    # Optional: shuffle back or leave sorted. Let's leave them sorted for consistency.
    
    # 2. Load caches for all datasets
    caches = {}
    for d_name in dataset_names:
        cache_path = cache_dir / f"{Path(d_name).stem}_trained_cache.jsonl"
        cache = {}
        if cache_path.is_file():
            with open(cache_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    cache[data["id"]] = data["geo_bundle"]
        caches[d_name] = {"path": cache_path, "data": cache}
        print(f"Loaded {len(cache)} optimized products from cache for {d_name}.")

    # 3. Process Products
    optimized_count = 0
    control_count = 0
    
    # We will append to cache files as we go, so open them in append mode
    file_handles = {d_name: open(caches[d_name]["path"], "a", encoding="utf-8") for d_name in dataset_names}
    
    try:
        from app.core.experiment import factual_geo_bundle
        
        for i, prod in enumerate(all_products, 1):
            pid = prod["id"]
            d_name = prod["source_dataset"]
            cache = caches[d_name]["data"]
            f_out = file_handles[d_name]
            
            if prod["condition"] == "CONTROL":
                prod["geo_bundle"] = factual_geo_bundle(prod)
                control_count += 1
                continue
                
            if pid in cache:
                prod["geo_bundle"] = cache[pid]
                optimized_count += 1
            else:
                print(f"[{i}/{len(all_products)}] Optimizing {pid}: {prod['title'][:50]}...")
                try:
                    bundle = build_geo_bundle(prod)
                    prod["geo_bundle"] = bundle
                    f_out.write(json.dumps({"id": pid, "geo_bundle": bundle}) + "\n")
                    f_out.flush()
                    optimized_count += 1
                except Exception as e:
                    print(f"  -> Error optimizing {pid}: {e}")
                    prod["geo_bundle"] = None
                    
                time.sleep(1)
    finally:
        for f in file_handles.values():
            f.close()

    print(f"\nProcessing complete. {optimized_count} optimized, {control_count} control.")
    
    # 4. DB Insertion (Single Run)
    valid_products = [p for p in all_products if p.get("geo_bundle")]
    if not valid_products:
        print("No optimized products to insert into DB.")
        return

    print(f"Inserting {len(valid_products)} products into Database...")
    now = utc_now()
    import uuid
    from app.models.study import GEOOptimizationApplication, GEOOptimizedProduct, GEOOptimizationConfig
    from sqlalchemy.exc import SQLAlchemyError
    
    run_id = f"TRN-{uuid.uuid4().hex[:12]}"
    max_retries = 5
    
    for attempt in range(max_retries):
        try:
            with SessionLocal() as db:
                config_id = "cfg-chatbot-training-run"
                if not db.scalars(select(GEOOptimizationConfig).where(GEOOptimizationConfig.id == config_id)).first():
                    dummy_config = GEOOptimizationConfig(
                        id=config_id, revision=1, name="Chatbot CSV Training Config",
                        optimization_target="Persuasion", model_profile="Aggressive",
                        model_name="gemini-3.1-pro-preview", model_version="1.0",
                        treatment_percentage=treatment_percentage, assignment_strategy="exact_split",
                        random_seed="csv-training", parameter_weights_json={}, feature_toggles_json={},
                        scope_json={"type": "all_catalog", "categories": [], "product_ids": [], "pair_ids": []},
                        is_active=True, created_at=now, updated_at=now, last_applied_at=now
                    )
                    db.add(dummy_config)
                    db.commit()

                application = GEOOptimizationApplication(
                    id=run_id, config_id=config_id, 
                    config_snapshot_json={"note": "Batch CSV training run"},
                    scope_summary_json={"categories": dataset_names, "total_products": len(valid_products)},
                    previous_assignment_json={},
                    application_summary_json={
                        "status": "completed", 
                        "selected_products": len(valid_products), 
                        "updated_products": len(valid_products),
                        "control_products": control_count,
                        "geo_optimized_products": optimized_count,
                        "integrity_failures": 0
                    },
                    safety_notes_json=["Batch training across all CSVs."], created_at=now,
                )
                db.add(application)
                
                print("Cleaning up old DB entries for target datasets...")
                db.execute(ProductVector.__table__.delete().where(
                    ProductVector.product_id.in_(select(Product.id).where(Product.source_dataset.in_(dataset_names)))
                ))
                db.execute(GEOOptimizedProduct.__table__.delete().where(
                    GEOOptimizedProduct.product_id.in_(select(Product.id).where(Product.source_dataset.in_(dataset_names)))
                ))
                db.execute(Product.__table__.delete().where(Product.source_dataset.in_(dataset_names)))
                db.flush()

                insert_data = []
                geo_insert_data = []
                for p in valid_products:
                    p_copy = dict(p)
                    p_copy["created_at"] = now
                    p_copy["imported_at"] = now
                    insert_data.append(p_copy)
                    geo_insert_data.append({
                        "id": f"GOP-{uuid.uuid4().hex}", "run_id": run_id, "product_id": p["id"],
                        "condition": p.get("condition", "CONTROL"), "geo_bundle": p["geo_bundle"],
                        "product_snapshot_json": p, "integrity_flags_json": [],
                        "applied_at": now, "created_at": now,
                    })

                print(f"Bulk inserting {len(insert_data)} products...")
                db.bulk_insert_mappings(Product, insert_data)
                db.bulk_insert_mappings(GEOOptimizedProduct, geo_insert_data)
                db.commit()
            break
        except SQLAlchemyError as e:
            print(f"Database connection error on attempt {attempt+1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                print("Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"Failed to save to database after {max_retries} attempts.")
                raise

    print("Indexing products for the Chatbot Vector DB (RAG)...")
    index_products(valid_products)

def main():
    parser = argparse.ArgumentParser(description="Train Chatbot from CSV Data")
    parser.add_argument("--csv", help="Path to a single training CSV file")
    parser.add_argument("--dir", help="Path to a directory containing CSV files to process")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to process per file")
    parser.add_argument("--split", type=int, default=100, help="Percentage of products to run generative GEO optimization on (e.g. 50 for 50/50 split). The rest will remain as CONTROL.")
    args = parser.parse_args()

    if not args.csv and not args.dir:
        print("Error: Must provide either --csv or --dir")
        sys.exit(1)

    all_products = []
    dataset_names = []
    cache_dir = None

    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.is_file():
            print(f"Error: CSV file not found at {csv_path}")
            sys.exit(1)
        cache_dir = csv_path.parent
        prods, d_name = load_csv_products(csv_path, args.limit)
        all_products.extend(prods)
        dataset_names.append(d_name)

    if args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            print(f"Error: Directory not found at {dir_path}")
            sys.exit(1)
        
        cache_dir = dir_path
        csv_files = list(dir_path.glob("*.csv"))
        print(f"Found {len(csv_files)} CSV files in {dir_path}. Reading data...\n")
        for csv_file in csv_files:
            if csv_file.name == "Cameras_over_10000.csv":
                continue
            prods, d_name = load_csv_products(csv_file, args.limit)
            all_products.extend(prods)
            dataset_names.append(d_name)
            
    if all_products:
        process_all_products(all_products, dataset_names, args.split, cache_dir)
        print("\nAll Training Complete!")
        print("The Chatbot is now ready to retrieve and serve these optimized products.")
    else:
        print("No products found to process.")

if __name__ == "__main__":
    main()
