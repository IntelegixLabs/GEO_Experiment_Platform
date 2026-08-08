"""Integration-style tests for the resumable Amazon catalog import command."""

from __future__ import annotations

from pathlib import Path
import sys

from sqlalchemy import func, select


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.db import session as database
from app.models import Product
from app.scripts.import_amazon_catalog import import_catalog, plan_category_assignments


def _write_catalog(path: Path) -> None:
    path.write_text(
        ",name,main_category,sub_category,image,link,ratings,no_of_ratings,discount_price,actual_price\n"
        "0,Travel Kettle,home,Kettles,https://images.example/k1.jpg,https://www.amazon.in/dp/B012345678,4.4,20,₹1299,₹1999\n"
        "1,Desk Kettle,home,Kettles,https://images.example/k2.jpg,https://www.amazon.in/dp/B012345679,4.2,19,₹1399,₹2099\n"
        "2,Compact Kettle,home,Kettles,https://images.example/k3.jpg,https://www.amazon.in/dp/B012345670,4.1,18,₹1199,₹1899\n"
        "3,Travel Headphones,electronics,Audio,https://images.example/h1.jpg,https://www.amazon.in/dp/B012345671,4.0,17,₹2299,₹2999\n"
        "4,Desk Headphones,electronics,Audio,https://images.example/h2.jpg,https://www.amazon.in/dp/B012345672,3.9,16,₹2399,₹3099\n",
        encoding="utf-8",
    )


def test_amazon_import_is_batched_idempotent_and_balanced(tmp_path: Path) -> None:
    source = tmp_path / "Amazon-Products.csv"
    _write_catalog(source)
    database.configure_database(f"sqlite:///{(tmp_path / 'catalog.sqlite3').as_posix()}")
    try:
        # A planning pass keeps only small category counters and makes the
        # overall source frame as balanced as possible (2 CONTROL / 3 GEO for
        # this odd-sized five-row fixture).
        from app.scripts.import_amazon_catalog import ImportStats

        planning_stats = ImportStats(source=str(source))
        starts, total = plan_category_assignments(source, limit=0, stats=planning_stats)
        assert total == 5
        assert set(starts) == {"home", "electronics"}

        first = import_catalog(source, batch_size=2)
        assert first.planned_records == 5
        assert first.imported == 5
        assert first.skipped_existing == 0
        assert first.control == 2
        assert first.geo_optimized == 3

        with database.SessionLocal() as db:
            assert db.scalar(select(func.count()).select_from(Product)) == 5
            assert set(db.scalars(select(Product.currency)).all()) == {"INR"}
            assert db.scalar(select(func.count()).select_from(Product).where(Product.geo_bundle.is_not(None))) == 3
            assert all(product.source_product_key for product in db.scalars(select(Product)).all())

        second = import_catalog(source, batch_size=3)
        assert second.imported == 0
        assert second.skipped_existing == 5
    finally:
        database.engine.dispose()
