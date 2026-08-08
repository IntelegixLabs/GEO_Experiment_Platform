"""Focused engine-construction checks that do not require a running server."""

from __future__ import annotations

from pathlib import Path
import sys


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.db.session import _build_engine


def test_postgres_engine_uses_psycopg_for_a_standard_postgresql_url() -> None:
    engine = _build_engine("postgresql://researcher:example@localhost:5434/geo")
    try:
        assert engine.url.drivername == "postgresql+psycopg"
    finally:
        engine.dispose()

