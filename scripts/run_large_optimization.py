"""Background Script for Large Scale GEO Optimization.

Usage:
    cd backend
    python -m scripts.run_large_optimization --limit 600000 --treatment-percentage 50

This script safely executes a massive optimization run in the background. 
It batches database commits and vector indexing to ensure progress is saved 
periodically without crashing or losing data if interrupted.
"""

import argparse
import sys
import time
import uuid
from pathlib import Path
import json

# Ensure backend package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, select, update

from app.core.config import get_settings
from app.core.catalog import product_record
from app.core.experiment import build_geo_bundle, utc_now
from app.db.session import SessionLocal
from app.models.study import Product, GEOOptimizedProduct, GEOOptimizationConfig, GEOOptimizationApplication, Event, Query, SurveyResponse
from app.schemas.study import GEOOptimizationApply, GEOOptimizationScope
from app.api.routes.admin import _geo_scope_summary, _geo_assignment_stream, _geo_config_response, _apply_geo_feature_toggles
from app.services.vector_db import index_products

BATCH_SIZE = 500

def main():
    parser = argparse.ArgumentParser(description="Run large scale GEO optimization")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum number of products to process")
    parser.add_argument("--treatment-percentage", type=int, default=50, help="Treatment percentage (1-99)")
    parser.add_argument("--seed", type=str, default="geo-study-v1", help="Randomization seed")
    parser.add_argument("--dry-run", action="store_true", help="Run without calling LLM or saving to DB")
    args = parser.parse_args()

    payload = GEOOptimizationApply(
        name="Large Scale Background Run",
        treatment_percentage=args.treatment_percentage,
        random_seed=args.seed,
        dry_run=args.dry_run,
        scope=GEOOptimizationScope(
            type="all_catalog",
            limit=args.limit
        )
    )

    print(f"Starting Large Scale Optimization (Limit: {args.limit}, Split: {args.treatment_percentage}%)")
    
    with SessionLocal() as db:
        scope = payload.scope.model_dump()
        scope_summary = _geo_scope_summary(db, scope)
        selected_products = scope_summary["selected_products"]
        
        if selected_products == 0:
            print("ERROR: Scope contains no products.")
            return

        print(f"Found {selected_products} products in scope.")
        
        outcome_counts = {
            name: int(db.scalar(select(func.count()).select_from(model)) or 0)
            for name, model in (("events", Event), ("queries", SurveyResponse), ("surveys", SurveyResponse))
        }
        
        previous_assignments = {
            "geo_optimized": int(db.scalar(select(func.count()).select_from(Product).where(Product.condition == "GEO_OPTIMIZED")) or 0),
            "control": int(db.scalar(select(func.count()).select_from(Product).where(Product.condition == "CONTROL")) or 0)
        }
        
        run_id = f"GOA-{uuid.uuid4().hex}"
        now = utc_now()
        
        # 1. Create the Config and Application records first so they exist if we crash
        if not payload.dry_run:
            revision = int(db.scalar(select(func.coalesce(func.max(GEOOptimizationConfig.revision), 0))) or 0) + 1
            db.execute(update(GEOOptimizationConfig).where(GEOOptimizationConfig.is_active.is_(True)).values(is_active=False))
            
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
                application_summary_json={"status": "running", "selected_products": selected_products, "updated_products": 0},
                safety_notes_json=["Background large scale run."],
                created_at=now,
            )
            db.add(application)
            db.commit()
            print(f"Created Application Run ID: {run_id}")

        control_products = 0
        geo_optimized_products = 0
        updated_products = 0
        integrity_failures = []
        batch_records_for_index = []
        
        t0 = time.time()
        
        try:
            # 2. Iterate and process
            for product, condition in _geo_assignment_stream(db, scope, payload):
                assigned_condition = condition
                bundle = None
                
                if condition == "GEO_OPTIMIZED":
                    try:
                        if not payload.dry_run:
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
                    batch_records_for_index.append(snapshot)
                
                updated_products += 1
                
                if updated_products % 10 == 0:
                    elapsed = time.time() - t0
                    rate = updated_products / elapsed if elapsed > 0 else 0
                    print(f"Progress: {updated_products:,} / {selected_products:,} ({(updated_products/selected_products)*100:.1f}%) - Rate: {rate:.1f} items/s")
                
                # Batch Commit & Index
                if not payload.dry_run and len(batch_records_for_index) >= BATCH_SIZE:
                    db.commit()
                    print(f"  -> Committed batch of {BATCH_SIZE} to SQL DB.")
                    
                    try:
                        index_products(batch_records_for_index)
                        print(f"  -> Indexed {len(batch_records_for_index)} products into Vector DB.")
                    except Exception as e:
                        print(f"  -> Vector DB indexing error: {e}")
                    
                    batch_records_for_index = []
                    
                    # Update application summary status
                    application.application_summary_json = {
                        "status": "running",
                        "selected_products": selected_products,
                        "updated_products": updated_products,
                        "control_products": control_products,
                        "geo_optimized_products": geo_optimized_products,
                        "integrity_failure_count": len(integrity_failures)
                    }
                    db.commit()
            
            # Final Batch Commit
            if not payload.dry_run:
                if batch_records_for_index:
                    db.commit()
                    try:
                        index_products(batch_records_for_index)
                        print(f"  -> Indexed final {len(batch_records_for_index)} products into Vector DB.")
                    except Exception as e:
                        print(f"  -> Vector DB indexing error: {e}")
                
                # Finalize application summary
                application.application_summary_json = {
                    "status": "complete",
                    "selected_products": selected_products,
                    "updated_products": updated_products,
                    "control_products": control_products,
                    "geo_optimized_products": geo_optimized_products,
                    "integrity_failure_count": len(integrity_failures)
                }
                db.commit()
                
            print(f"\nOptimization completed successfully!")
            print(f"Total time: {time.time() - t0:.1f} seconds")
            print(f"Products updated: {updated_products:,}")
            print(f"GEO Optimized: {geo_optimized_products:,}")
            print(f"Control: {control_products:,}")
            
        except KeyboardInterrupt:
            print("\nScript manually interrupted. Progress up to the last batch was saved.")
        except Exception as e:
            print(f"\nFatal error during processing: {e}")
            if not payload.dry_run:
                db.rollback()

if __name__ == "__main__":
    main()
