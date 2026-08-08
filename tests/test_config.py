"""Tests for local dotenv configuration and opt-in catalog seeding."""

from __future__ import annotations

from pathlib import Path
import sys

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core import config
from app.core.catalog import parse_catalog_csv
from app.db.base import Base
from app.db.seed import seed_demo_catalog
from app.models import Product


def test_dotenv_loader_respects_existing_process_values(tmp_path: Path, monkeypatch) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("GEO_TEST_VALUE=from-file\nGEO_TEST_QUOTED='two words'\n", encoding="utf-8")
    monkeypatch.delenv("GEO_TEST_VALUE", raising=False)
    monkeypatch.delenv("GEO_TEST_QUOTED", raising=False)

    assert config.load_env_file(dotenv) is True
    assert config.os.getenv("GEO_TEST_VALUE") == "from-file"
    assert config.os.getenv("GEO_TEST_QUOTED") == "two words"

    monkeypatch.setenv("GEO_TEST_VALUE", "from-process")
    dotenv.write_text("GEO_TEST_VALUE=from-second-file\n", encoding="utf-8")
    config.load_env_file(dotenv)
    assert config.os.getenv("GEO_TEST_VALUE") == "from-process"


def test_sqlalchemy_database_url_is_authoritative_and_uses_psycopg(monkeypatch) -> None:
    monkeypatch.setenv("SQLALCHEMY_DATABASE_URL", "postgresql://researcher:secret@db.example:5434/geo")
    monkeypatch.setenv("GEO_DATABASE_URL", "sqlite:///ignored.sqlite3")
    config.get_settings.cache_clear()
    try:
        settings = config.get_settings()
        assert settings.database_url == "postgresql+psycopg://researcher:secret@db.example:5434/geo"
    finally:
        config.get_settings.cache_clear()


def test_database_url_normalizer_preserves_explicit_drivers() -> None:
    assert config.normalize_database_url("postgres://host/geo") == "postgresql+psycopg://host/geo"
    assert (
        config.normalize_database_url("postgresql+psycopg://host/geo")
        == "postgresql+psycopg://host/geo"
    )
    assert config.normalize_database_url("sqlite:///study.sqlite3") == "sqlite:///study.sqlite3"


def test_amazon_catalog_settings_use_a_resolved_path_and_safe_batch_size(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GEO_AMAZON_CATALOG_CSV", "catalog/Amazon-Products.csv")
    monkeypatch.setenv("GEO_CATALOG_IMPORT_BATCH_SIZE", "not-a-number")
    config.get_settings.cache_clear()
    try:
        settings = config.get_settings()
        assert settings.amazon_catalog_csv == (settings.backend_dir / "catalog" / "Amazon-Products.csv").resolve()
        assert settings.catalog_import_batch_size == 1000
    finally:
        config.get_settings.cache_clear()


def test_configured_csv_seeds_only_an_empty_database(tmp_path: Path, monkeypatch) -> None:
    catalog = tmp_path / "catalog.csv"
    catalog.write_text(
        "id,title,brand,category,pair_id,description,key_features,source_url\n"
        "REAL-001,Verified Bottle,Example,Home & Kitchen,real-bottle,Insulated bottle,leakproof cap|dishwasher safe,https://example.org/a\n"
        "REAL-002,Verified Flask,Example,Home & Kitchen,real-bottle,Insulated flask,leakproof cap|dishwasher safe,https://example.org/b\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("GEO_SEED_CATALOG_CSV", str(catalog))
    config.get_settings.cache_clear()
    engine = create_engine("sqlite://")
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine)
    try:
        with session_factory() as db:
            assert seed_demo_catalog(db) == 2
            assert {product.id for product in db.scalars(select(Product)).all()} == {"REAL-001", "REAL-002"}
            assert seed_demo_catalog(db) == 0
    finally:
        engine.dispose()
        config.get_settings.cache_clear()


def test_real_world_example_catalog_is_parseable_and_does_not_invent_volatile_fields() -> None:
    example = BACKEND_ROOT.parent / "examples" / "real_world_commuter_bottles" / "commuter_bottles_catalog.csv"
    products = parse_catalog_csv(example.read_text(encoding="utf-8"))
    assert {product["id"] for product in products} == {"HYDRO-FLASK-32-WM-FSC", "YETI-RAMBLER-26-CHUG"}
    assert all(product["price"] is None for product in products)
    assert all(product["rating"] is None for product in products)
    assert all(product["review_count"] is None for product in products)
    assert all(product["condition"] is None for product in products)
    assert all(product["source_url"].startswith("https://") for product in products)
