"""SQLAlchemy engine and request-session dependency."""

from __future__ import annotations

import os
import time
from collections.abc import Generator
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings, normalize_database_url


def _build_engine(database_url: str) -> Engine:
    """Create a durable SQLAlchemy engine for the configured database."""

    connect_args: dict[str, object] = {}
    engine_kwargs: dict[str, object] = {"future": True}

    normalized_url = normalize_database_url(database_url)

    is_sqlite = normalized_url.startswith("sqlite")
    is_postgres = normalized_url.startswith("postgresql")
    is_cockroachdb = normalized_url.startswith("cockroachdb")

    if is_sqlite:
        connect_args["check_same_thread"] = False

        if (
            normalized_url.startswith("sqlite:///")
            and not normalized_url.endswith(":memory:")
        ):
            raw_path = normalized_url.removeprefix("sqlite:///")

            if raw_path:
                Path(raw_path).expanduser().parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

    elif is_postgres or is_cockroachdb:
        engine_kwargs.update(
            pool_pre_ping=True,
            pool_recycle=600,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
        )

        # CockroachDB Cloud SSL
        connect_args["sslrootcert"] = os.path.join(
            os.path.dirname(__file__),
            "certs",
            "cc-ca.crt",
        )
        connect_args["sslmode"] = "verify-full"

    engine = create_engine(
        normalized_url,
        connect_args=connect_args,
        **engine_kwargs,
    )

    if is_sqlite:
        event.listen(
            engine,
            "connect",
            _enable_sqlite_foreign_keys,
        )

    return engine


def _enable_sqlite_foreign_keys(
    dbapi_connection: object,
    _: object,
) -> None:
    cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


engine = _build_engine(
    get_settings().database_url
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


def warmup_pool() -> None:
    """Pre-open one connection so the first HTTP request doesn't pay the TLS cost."""

    t0 = time.time()

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))

        print(
            f"DB pool warmed up in {time.time() - t0:.2f}s"
        )

    except Exception as e:
        print(
            f"DB warmup failed after {time.time() - t0:.2f}s: {e}"
        )


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


def configure_database(database_url: str) -> None:
    """Swap the engine for test suites or an explicitly configured app instance."""

    global engine, SessionLocal

    engine.dispose()

    engine = _build_engine(database_url)

    SessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )