"""Small deterministic text helpers shared by the study services.

The helpers deliberately avoid model downloads, remote services, and opaque
language processing.  They make the local experiment reproducible and allow a
researcher to inspect exactly why a term was considered a match.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
import json
import re
from typing import Any


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[._/+\-][A-Za-z0-9]+)*")

# These are deliberately conservative: removing them only prevents common
# function words from counting as evidence; it does not create a new claim.
STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "please",
        "show",
        "that",
        "the",
        "to",
        "want",
        "what",
        "which",
        "with",
        "would",
        "you",
    }
)


def to_mapping(value: Any) -> dict[str, Any]:
    """Convert a Pydantic/dataclass/mapping record to a plain dictionary.

    Routes can pass ORM/Pydantic values without making this service layer depend
    on either FastAPI, Pydantic, SQLAlchemy, or a particular database model.
    """

    if value is None:
        return {}
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="python")
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    dict_method = getattr(value, "dict", None)
    if callable(dict_method):  # Pydantic v1 compatibility.
        dumped = dict_method()
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    if is_dataclass(value) and not isinstance(value, type):
        dumped = asdict(value)
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    try:
        return dict(vars(value))
    except TypeError:
        return {}


def as_list(value: Any) -> list[str]:
    """Normalize catalogue feature values into non-empty strings."""

    if value is None:
        return []
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return []
        # Imported DB values may contain a JSON array; handle it first.
        try:
            parsed = json.loads(cleaned)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = None
        if isinstance(parsed, list):
            return as_list(parsed)
        return [piece.strip() for piece in re.split(r"[|;,]", cleaned) if piece.strip()]
    if isinstance(value, Mapping):
        return [str(item).strip() for item in value.values() if str(item).strip()]
    if isinstance(value, Iterable):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def normalize_whitespace(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def tokenize(value: Any, *, include_stop_words: bool = False) -> list[str]:
    """Return stable lower-case lexical tokens.

    Hyphenated product terms remain one token where possible, so model numbers
    and terms such as ``USB-C`` are not accidentally discarded.
    """

    tokens = [token.lower() for token in TOKEN_RE.findall(normalize_whitespace(value))]
    if include_stop_words:
        return tokens
    return [token for token in tokens if len(token) > 1 and token not in STOP_WORDS]


def stable_json(value: Any) -> str:
    """Encode an object predictably for hashing, logs, and regression tests."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
