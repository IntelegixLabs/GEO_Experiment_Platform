"""Streaming mapper for the Kaggle Amazon Products catalog.

This module is deliberately database-agnostic: it reads one CSV record at a
time and returns compact, validated values that a caller can upsert with
SQLAlchemy in batches.  It neither downloads data nor creates experiment
conditions itself.

The source data has no durable product primary key.  We therefore use a
canonical source URL hash as the durable row identity.  This matters because
the supplied catalog contains distinct URLs that can share an Amazon ASIN.
When an ASIN is present, it is retained in ``pair_id`` for later matched-variant
work, but it is never used as the unique source-row key.

No descriptions, features, brands, availability claims, or shipping claims
are inferred from a product name.  Missing source fields remain empty or
``None`` and are documented with ``data_quality_flags``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import hashlib
import math
from pathlib import Path
import re
from typing import Any, TextIO, TypeVar
from urllib.parse import parse_qsl, urlsplit, urlunsplit


AMAZON_DATASET_NAME = "kaggle/lokeshparab/amazon-products-dataset"
REQUIRED_AMAZON_COLUMNS = frozenset(
    {
        "name",
        "main_category",
        "sub_category",
        "image",
        "link",
        "ratings",
        "no_of_ratings",
        "discount_price",
        "actual_price",
    }
)

# Values used by pandas/CSV exports to mean a missing field.  Product names
# and categories matching these values are not valid catalogue values.
MISSING_VALUES = frozenset({"", "-", "--", "n/a", "na", "nan", "none", "null"})
MAX_TITLE_LENGTH = 500
MAX_CATEGORY_LENGTH = 250
MAX_URL_LENGTH = 2_000

_HEADER_SEPARATOR = re.compile(r"[^a-z0-9]+")
_NUMBER = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?")
_COUNT_WITH_SUFFIX = re.compile(
    r"^\s*([+-]?(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d+)?)\s*([km])?\s*$",
    re.IGNORECASE,
)
_ASIN_IN_PATH = re.compile(
    r"/(?:dp|gp/product|gp/aw/d|gp/offer-listing)/([a-z0-9]{10})(?:[/?]|$)",
    re.IGNORECASE,
)
_ASIN_IN_QUERY = re.compile(r"^[a-z0-9]{10}$", re.IGNORECASE)

_HEADER_ALIASES = {
    "no_of_rating": "no_of_ratings",
    "number_of_ratings": "no_of_ratings",
    "number_of_rating": "no_of_ratings",
    "rating": "ratings",
    "discounted_price": "discount_price",
    "list_price": "actual_price",
    "mrp": "actual_price",
    "url": "link",
    "image_url": "image",
}
_VALID_CONDITIONS = frozenset({"CONTROL", "GEO_OPTIMIZED"})
T = TypeVar("T")


class AmazonCatalogError(ValueError):
    """Base class for an invalid Amazon catalog source."""


class AmazonCatalogHeaderError(AmazonCatalogError):
    """Raised when a CSV does not expose the expected Kaggle columns."""


class AmazonCatalogRowError(AmazonCatalogError):
    """Raised when one source row cannot safely become a product record."""

    def __init__(self, row_number: int, message: str) -> None:
        self.row_number = row_number
        super().__init__(f"Amazon catalog row {row_number}: {message}")


@dataclass(frozen=True)
class AmazonCatalogRecord:
    """A source-faithful product record ready for a SQLAlchemy bulk upsert.

    ``source_product_key`` is the durable idempotency key.  ``source_product_id``
    is a short database-safe identifier derived from it.  These values are not
    based on the source row number, so reordering the CSV does not change them.
    """

    source_product_id: str
    source_product_key: str
    pair_id: str
    source_row_number: int
    name: str
    main_category: str | None
    sub_category: str | None
    image_url: str | None
    source_url: str | None
    rating: float | None
    review_count: int | None
    discount_price: float | None
    actual_price: float | None
    data_quality_flags: tuple[str, ...] = ()

    @property
    def category(self) -> str:
        """Return the source main category, falling back to sub-category."""

        # ``map_amazon_row`` ensures at least one source category is present.
        return self.main_category or self.sub_category or "Uncategorized"

    @property
    def price(self) -> float | None:
        """Use the advertised discounted price, otherwise the listed MRP."""

        return self.discount_price if self.discount_price is not None else self.actual_price

    def catalog_product_values(
        self,
        *,
        condition: str | None,
        product_id: str | None = None,
    ) -> dict[str, Any]:
        """Return fields for the project's persistent ``Product`` table.

        The importer supplies ``condition`` explicitly so the study's assignment
        policy remains visible and reproducible outside this source mapper.
        ``product_id`` may be overridden when a caller materialises matched
        experimental variants from one source listing.
        """

        normalised_condition = _normalise_condition(condition)
        if normalised_condition not in _VALID_CONDITIONS:
            raise ValueError("condition must be CONTROL or GEO_OPTIMIZED")

        return {
            "id": product_id or self.source_product_id,
            # The Kaggle source does not provide a merchant SKU.  The stable
            # internal source ID is retained separately rather than displayed
            # as though it were a seller-supplied SKU.
            "sku": None,
            "title": self.name,
            "category": self.category,
            "main_category": self.main_category,
            "sub_category": self.sub_category,
            "price": self.price,
            "discount_price": self.discount_price,
            "actual_price": self.actual_price,
            "currency": "INR",
            # The dataset contains no description/features/brand fields.  Do
            # not generate claims that would contaminate a GEO treatment.
            "description": "",
            "key_features": [],
            "rating": self.rating,
            "review_count": self.review_count,
            "source_url": self.source_url,
            "image_url": self.image_url,
            "condition": normalised_condition,
            "pair_id": self.pair_id,
            "source_dataset": AMAZON_DATASET_NAME,
            "source_row_number": self.source_row_number,
            "source_product_key": self.source_product_key,
            "data_quality_flags": list(self.data_quality_flags),
        }


def parse_inr_amount(value: object) -> float | None:
    """Parse a single INR amount without guessing from ambiguous text.

    The Kaggle export contains values such as ``₹1,29,999`` and, on systems
    where UTF-8 was decoded incorrectly, ``â‚¹32,999``.  A value containing
    multiple amounts (for example a price range) deliberately returns ``None``
    rather than selecting one amount arbitrarily.
    """

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return float(value) if value.is_finite() else None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None

    text = _optional_text(value)
    if text is None:
        return None

    # The amount expression intentionally ignores currency glyphs, including
    # mojibake, but rejects ranges/compound values by requiring one match.
    matches = _NUMBER.findall(text.replace("\u00a0", " "))
    if len(matches) != 1:
        return None
    try:
        amount = Decimal(matches[0].replace(",", ""))
    except InvalidOperation:
        return None
    return float(amount) if amount.is_finite() else None


def parse_review_count(value: object) -> int | None:
    """Parse an Amazon review-count field, including grouped and K/M values."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float):
        return int(value) if math.isfinite(value) and value >= 0 and value.is_integer() else None

    text = _optional_text(value)
    if text is None:
        return None
    matched = _COUNT_WITH_SUFFIX.match(text)
    if not matched:
        return None
    try:
        count = Decimal(matched.group(1).replace(",", ""))
    except InvalidOperation:
        return None
    suffix = (matched.group(2) or "").lower()
    if suffix == "k":
        count *= 1_000
    elif suffix == "m":
        count *= 1_000_000
    if not count.is_finite() or count < 0 or count != count.to_integral_value():
        return None
    return int(count)


def parse_rating(value: object) -> float | None:
    """Return a finite 0–5 rating, or ``None`` for missing/invalid input."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
    else:
        text = _optional_text(value)
        if text is None:
            return None
        try:
            numeric = float(text.replace(",", ""))
        except ValueError:
            return None
    return numeric if math.isfinite(numeric) and 0 <= numeric <= 5 else None


def map_amazon_row(raw_row: Mapping[str, object], *, source_row_number: int) -> AmazonCatalogRecord:
    """Map and validate one Kaggle Amazon Products row.

    Required source facts are a non-empty ``name`` and at least one category.
    Non-critical missing or malformed source fields produce data-quality flags
    instead of fabricated replacement claims.
    """

    row = _normalise_row(raw_row)
    name = _required_text(row.get("name"), field="name", row_number=source_row_number, max_length=MAX_TITLE_LENGTH)
    main_category = _validated_optional_text(
        row.get("main_category"),
        field="main_category",
        row_number=source_row_number,
        max_length=MAX_CATEGORY_LENGTH,
    )
    sub_category = _validated_optional_text(
        row.get("sub_category"),
        field="sub_category",
        row_number=source_row_number,
        max_length=MAX_CATEGORY_LENGTH,
    )
    if main_category is None and sub_category is None:
        raise AmazonCatalogRowError(source_row_number, "requires main_category or sub_category")

    flags: list[str] = []
    discount_price = parse_inr_amount(row.get("discount_price"))
    actual_price = parse_inr_amount(row.get("actual_price"))
    _append_number_quality_flags(
        flags,
        raw_value=row.get("discount_price"),
        parsed=discount_price,
        field="discount_price",
    )
    _append_number_quality_flags(
        flags,
        raw_value=row.get("actual_price"),
        parsed=actual_price,
        field="actual_price",
    )
    if discount_price is None and actual_price is None:
        flags.append("missing_price")

    rating = parse_rating(row.get("ratings"))
    _append_number_quality_flags(flags, raw_value=row.get("ratings"), parsed=rating, field="ratings")
    review_count = parse_review_count(row.get("no_of_ratings"))
    _append_number_quality_flags(
        flags,
        raw_value=row.get("no_of_ratings"),
        parsed=review_count,
        field="no_of_ratings",
    )

    source_url = _safe_url(row.get("link"), field="link", flags=flags)
    image_url = _safe_url(row.get("image"), field="image", flags=flags)
    source_product_key, source_product_id, pair_id = stable_source_identifiers(
        name=name,
        main_category=main_category,
        sub_category=sub_category,
        source_url=source_url,
        image_url=image_url,
    )
    return AmazonCatalogRecord(
        source_product_id=source_product_id,
        source_product_key=source_product_key,
        pair_id=pair_id,
        source_row_number=source_row_number,
        name=name,
        main_category=main_category,
        sub_category=sub_category,
        image_url=image_url,
        source_url=source_url,
        rating=rating,
        review_count=review_count,
        discount_price=discount_price,
        actual_price=actual_price,
        data_quality_flags=tuple(sorted(set(flags))),
    )


def stable_source_identifiers(
    *,
    name: str,
    main_category: str | None,
    sub_category: str | None,
    source_url: str | None,
    image_url: str | None,
) -> tuple[str, str, str]:
    """Return ``(source_product_key, product_id, pair_id)`` deterministically.

    The source product key hashes a canonical product URL whenever one is
    present.  An ASIN alone is intentionally not unique enough for this
    dataset: one ASIN can occur under multiple distinct source URLs.  The ASIN
    is used only as a stable grouping ``pair_id``.  When no usable URL exists,
    the mapper falls back to a conservative source fingerprint.  It never uses
    row position.
    """

    asin = extract_amazon_asin(source_url)
    canonical_url = _canonical_url_for_identity(source_url)
    if canonical_url:
        identity_material = f"url\x1f{canonical_url}"
        source_product_key = f"amazon-url-sha256:{_digest(identity_material)}"
    else:
        # A fingerprint is a last resort.  It intentionally excludes prices,
        # ratings, and review counts because those fields can change without
        # making the underlying listing a new product.
        identity_material = "\x1f".join(
            (
                "fingerprint",
                _identity_text(name),
                _identity_text(main_category),
                _identity_text(sub_category),
                _canonical_url_for_identity(image_url) or "",
            )
        )
        source_product_key = f"amazon-fingerprint-sha256:{_digest(identity_material)}"

    source_product_id = f"AMZ-H-{_digest(source_product_key)[:24].upper()}"
    pair_id = f"AMZ-PAIR-{asin}" if asin else f"AMZ-PAIR-{source_product_id.removeprefix('AMZ-H-')}"
    return source_product_key, source_product_id, pair_id


def experimental_variant_id(source_product_id: str, condition: str) -> str:
    """Return a short deterministic ID for a matched source-listing variant."""

    normalised_condition = _normalise_condition(condition)
    suffixes = {"CONTROL": "CTRL", "GEO_OPTIMIZED": "GEO"}
    try:
        suffix = suffixes[normalised_condition]
    except KeyError as exc:
        raise ValueError("condition must be CONTROL or GEO_OPTIMIZED") from exc
    return f"{source_product_id}-{suffix}"


def iter_amazon_catalog_records(
    source: str | Path | TextIO,
    *,
    on_error: Callable[[AmazonCatalogRowError], None] | None = None,
) -> Iterator[AmazonCatalogRecord]:
    """Yield valid source records one at a time without loading the CSV.

    If ``on_error`` is omitted, the first bad row raises immediately to avoid
    silently losing data.  Supplying a callback makes the iterator tolerant:
    each invalid row is reported to that callback and skipped.
    """

    with _open_csv_source(source) as stream:
        reader = csv.DictReader(stream)
        _validate_headers(reader.fieldnames)
        for raw_row in reader:
            try:
                yield map_amazon_row(raw_row, source_row_number=reader.line_num)
            except AmazonCatalogRowError as error:
                if on_error is None:
                    raise
                on_error(error)


def iter_batches(items: Iterable[T], *, batch_size: int = 1_000) -> Iterator[list[T]]:
    """Group a stream into bounded lists suitable for SQLAlchemy bulk writes."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def extract_amazon_asin(source_url: str | None) -> str | None:
    """Extract a normalized ASIN from common Amazon product URL forms."""

    canonical = _canonical_url_for_identity(source_url)
    if canonical is None:
        return None
    split = urlsplit(canonical)
    matched = _ASIN_IN_PATH.search(split.path)
    if matched:
        return matched.group(1).upper()
    for key, value in parse_qsl(split.query, keep_blank_values=False):
        if key.casefold() == "asin" and _ASIN_IN_QUERY.fullmatch(value):
            return value.upper()
    return None


@contextmanager
def _open_csv_source(source: str | Path | TextIO) -> Iterator[TextIO]:
    if hasattr(source, "read"):
        yield source  # type: ignore[misc]
        return
    with Path(source).open("r", encoding="utf-8-sig", errors="replace", newline="") as stream:
        yield stream


def _validate_headers(fieldnames: list[str] | None) -> None:
    if not fieldnames:
        raise AmazonCatalogHeaderError("CSV needs a header row.")
    canonical_headers = [_canonical_header(header) for header in fieldnames if header is not None]
    seen = [header for header in canonical_headers if header]
    duplicates = sorted({header for header in seen if seen.count(header) > 1})
    if duplicates:
        raise AmazonCatalogHeaderError(f"CSV has duplicate normalized columns: {', '.join(duplicates)}")
    missing = sorted(REQUIRED_AMAZON_COLUMNS.difference(seen))
    if missing:
        raise AmazonCatalogHeaderError(f"CSV is missing required columns: {', '.join(missing)}")


def _normalise_row(raw_row: Mapping[str, object]) -> dict[str, object]:
    row: dict[str, object] = {}
    for key, value in raw_row.items():
        if key is None:
            continue
        canonical = _canonical_header(key)
        if canonical:
            row[canonical] = value
    return row


def _canonical_header(value: object) -> str:
    text = str(value or "").lstrip("\ufeff").strip().casefold()
    normalised = _HEADER_SEPARATOR.sub("_", text).strip("_")
    return _HEADER_ALIASES.get(normalised, normalised)


def _required_text(value: object, *, field: str, row_number: int, max_length: int) -> str:
    text = _validated_optional_text(value, field=field, row_number=row_number, max_length=max_length)
    if text is None:
        raise AmazonCatalogRowError(row_number, f"requires a non-empty {field}")
    return text


def _validated_optional_text(
    value: object,
    *,
    field: str,
    row_number: int,
    max_length: int,
) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    if "\x00" in text:
        raise AmazonCatalogRowError(row_number, f"{field} contains a NUL character")
    if len(text) > max_length:
        raise AmazonCatalogRowError(row_number, f"{field} exceeds {max_length} characters")
    return text


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if text.casefold() in MISSING_VALUES else text


def _safe_url(value: object, *, field: str, flags: list[str]) -> str | None:
    text = _optional_text(value)
    if text is None:
        flags.append(f"missing_{field}")
        return None
    if len(text) > MAX_URL_LENGTH or "\x00" in text:
        flags.append(f"invalid_{field}")
        return None
    try:
        split = urlsplit(text)
        # Accessing ``port`` validates malformed netloc values such as
        # ``https://example.org:invalid`` that urlsplit itself accepts.
        _ = split.port
    except ValueError:
        flags.append(f"invalid_{field}")
        return None
    if split.scheme.casefold() not in {"http", "https"} or not split.hostname or split.username or split.password:
        flags.append(f"invalid_{field}")
        return None
    return text


def _append_number_quality_flags(
    flags: list[str],
    *,
    raw_value: object,
    parsed: int | float | None,
    field: str,
) -> None:
    if parsed is not None:
        return
    if _optional_text(raw_value) is None:
        flags.append(f"missing_{field}")
    else:
        flags.append(f"invalid_{field}")


def _canonical_url_for_identity(value: str | None) -> str | None:
    if not value:
        return None
    try:
        split = urlsplit(value)
        port = split.port
    except ValueError:
        return None
    if split.scheme.casefold() not in {"http", "https"} or not split.netloc:
        return None
    hostname = (split.hostname or "").casefold()
    if not hostname:
        return None
    # Do not include fragments.  Query terms are retained here because ASIN can
    # occur in a query-only URL; Amazon tracking terms do not matter when an
    # ASIN is present, which is the overwhelmingly common dataset path.
    netloc = hostname
    if port and not ((split.scheme.casefold() == "http" and port == 80) or (split.scheme.casefold() == "https" and port == 443)):
        netloc = f"{hostname}:{port}"
    path = split.path.rstrip("/") or "/"
    return urlunsplit((split.scheme.casefold(), netloc, path, split.query, ""))


def _identity_text(value: str | None) -> str:
    return " ".join(str(value or "").casefold().split())


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalise_condition(value: str | None) -> str:
    return str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
