"""Small, dependency-free application configuration layer."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


STUDY_NAME = (
    "Generative Engine Optimization in E-Commerce: An Empirical Study of "
    "Product Page Optimization and AI-Assisted Consumer Shopping Behavior"
)
CONDITIONS = ("CONTROL", "GEO_OPTIMIZED")
ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _unquote_env_value(value: str) -> str:
    """Parse the small, predictable subset of dotenv syntax we need locally."""

    text = value.strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
        text = text[1:-1]
    elif " #" in text:
        # Preserve hashes in URLs/tokens unless they start a conventional
        # whitespace-separated comment.
        text = text.split(" #", 1)[0].rstrip()
    return text.replace("\\n", "\n") if "\\n" in text else text


def load_env_file(path: Path, *, override: bool = False) -> bool:
    """Load a local ``.env`` file without adding a third-party dependency.

    Existing process environment variables win by default, which keeps Docker,
    CI, and institution-managed secrets authoritative over a developer's local
    file. Malformed lines are ignored rather than being interpreted as shell
    syntax; dotenv files are configuration, never executable code.
    """

    if not path.is_file():
        return False
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, raw_value = line.split("=", 1)
        key = key.strip()
        if not ENV_KEY.match(key):
            continue
        if override or key not in os.environ:
            os.environ[key] = _unquote_env_value(raw_value)
    return True


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def normalize_database_url(database_url: str) -> str:
    """Return a SQLAlchemy URL that uses the installed PostgreSQL driver.

    ``postgresql://`` is the common libpq-style spelling, but SQLAlchemy maps
    that bare dialect to psycopg2 by default.  This project installs psycopg
    v3, so normalize legacy PostgreSQL URLs to the explicit psycopg dialect.
    URLs that already select a driver (for example ``postgresql+psycopg://``)
    and all non-PostgreSQL URLs are deliberately left untouched.
    """

    value = database_url.strip()
    lowercase = value.lower()
    if lowercase.startswith("postgresql://"):
        return f"postgresql+psycopg://{value[len('postgresql://') :]}"
    if lowercase.startswith("postgres://"):
        return f"postgresql+psycopg://{value[len('postgres://') :]}"
    return value


def _database_url(default: str) -> str:
    """Resolve the database URL with the standard SQLAlchemy name first.

    ``GEO_DATABASE_URL`` remains supported for existing local installations,
    but an explicitly supplied ``SQLALCHEMY_DATABASE_URL`` is authoritative.
    """

    raw_url = _env("SQLALCHEMY_DATABASE_URL", "") or _env("GEO_DATABASE_URL", default)
    return normalize_database_url(raw_url)


def _optional_path(value: str, *, base_dir: Path) -> Path | None:
    text = value.strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    return candidate if candidate.is_absolute() else (base_dir / candidate).resolve()


def _csv_env(name: str, default: str) -> tuple[str, ...]:
    value = _env(name, default)
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _positive_int_env(name: str, default: int) -> int:
    """Read a positive bounded configuration integer without crashing startup."""

    try:
        value = int(_env(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


@dataclass(frozen=True)
class Settings:
    """Runtime settings read once at startup.

    A dataclass is deliberately used instead of an additional settings package
    so the backend has a compact installation footprint for local research.
    """

    backend_dir: Path
    database_url: str
    cors_origins: tuple[str, ...]
    environment: str
    seed_catalog_csv: Path | None
    amazon_catalog_csv: Path | None
    catalog_import_batch_size: int
    api_prefix: str = "/api"
    llm_provider: str = "openai"
    llm_model_name: str = "gpt-4o-mini"
    openai_api_key: str | None = None
    gemini_api_key: str | None = None
    openai_model_name: str = "gpt-4o-mini" # deprecated, use llm_model_name
    admin_user: str | None = None
    admin_password: str | None = None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    backend_dir = Path(__file__).resolve().parents[2]
    load_env_file(backend_dir / ".env")
    default_db = (backend_dir / "data" / "geo_study.sqlite3").resolve().as_posix()

    # Fallback logic for model name
    llm_provider = _env("LLM", _env("LLM_PROVIDER", "openai")).lower()

    if llm_provider == "gemini":
        model_name = _env("GEMINI_MODEL_NAME", "gemini-2.5-flash")
    else:
        legacy_model = _env("OPENAI_MODEL_NAME", "gpt-4o-mini")
        model_name = _env("LLM_MODEL_NAME", legacy_model)

    return Settings(
        backend_dir=backend_dir,
        database_url=_database_url(f"sqlite:///{default_db}"),
        cors_origins=_csv_env(
            "GEO_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ),
        environment=_env("GEO_APP_ENV", "development"),
        seed_catalog_csv=_optional_path(_env("GEO_SEED_CATALOG_CSV", ""), base_dir=backend_dir),
        amazon_catalog_csv=_optional_path(_env("GEO_AMAZON_CATALOG_CSV", ""), base_dir=backend_dir),
        catalog_import_batch_size=_positive_int_env("GEO_CATALOG_IMPORT_BATCH_SIZE", 1000),
        llm_provider=llm_provider,
        llm_model_name=model_name,
        openai_api_key=_env("OPENAI_API_KEY", "") or None,
        gemini_api_key=_env("GEMINI_API_KEY", "") or None,
        openai_model_name=model_name,
        admin_user=_env("ADMIN_USER", "") or None,
        admin_password=_env("ADMIN_PASSWORD", "") or None,
    )
