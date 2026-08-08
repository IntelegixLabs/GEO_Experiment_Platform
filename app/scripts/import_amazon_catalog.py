"""Import the supplied Kaggle Amazon Products catalog into PostgreSQL.

Run from ``backend`` (or with that directory on ``PYTHONPATH``)::

    python -m app.scripts.import_amazon_catalog

The command is deliberately append-only and idempotent: source-product keys
already present in ``products`` are retained, so re-running after an
interruption fills only missing rows and never deletes study data.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import CONDITIONS, get_settings
from app.core.experiment import utc_now
from app.db import session as database
from app.db.base import Base
from app.models import Product
from app.services.amazon_catalog import (
    AMAZON_DATASET_NAME,
    AmazonCatalogRecord,
    AmazonCatalogRowError,
    iter_amazon_catalog_records,
    iter_batches,
)
from app.services.geo_treatment import FactualGEOBuilder, GEOIntegrityValidator


DEFAULT_LIMIT = 0


@dataclass
class ImportStats:
    """Small, JSON-serialisable audit record printed at the end of an import."""

    source: str
    source_dataset: str = AMAZON_DATASET_NAME
    requested_limit: int = DEFAULT_LIMIT
    planned_records: int = 0
    imported: int = 0
    skipped_existing: int = 0
    skipped_invalid: int = 0
    control: int = 0
    geo_optimized: int = 0
    quality_flag_counts: Counter[str] = field(default_factory=Counter)
    invalid_examples: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_dataset": self.source_dataset,
            "requested_limit": self.requested_limit or None,
            "planned_records": self.planned_records,
            "imported": self.imported,
            "skipped_existing": self.skipped_existing,
            "skipped_invalid": self.skipped_invalid,
            "control": self.control,
            "geo_optimized": self.geo_optimized,
            "quality_flag_counts": dict(sorted(self.quality_flag_counts.items())),
            "invalid_examples": self.invalid_examples,
            "assignment_scheme": (
                "Deterministic alternating assignment within source main category; "
                "the complete planned catalog is exactly balanced overall."
            ),
            "write_mode": "append-only idempotent insert by source_product_key",
        }


def _category_key(record: AmazonCatalogRecord) -> str:
    return " ".join(record.category.casefold().split()) or "uncategorized"


def _iter_limited_records(
    source: Path,
    *,
    limit: int,
    stats: ImportStats,
) -> Iterator[AmazonCatalogRecord]:
    """Yield valid records while recording malformed source rows for audit."""

    def on_error(error: AmazonCatalogRowError) -> None:
        stats.skipped_invalid += 1
        if len(stats.invalid_examples) < 10:
            stats.invalid_examples.append(str(error))

    emitted = 0
    for record in iter_amazon_catalog_records(source, on_error=on_error):
        if limit and emitted >= limit:
            break
        emitted += 1
        stats.quality_flag_counts.update(record.data_quality_flags)
        yield record


def plan_category_assignments(
    source: Path,
    *,
    limit: int,
    stats: ImportStats,
) -> tuple[dict[str, str], int]:
    """Plan an exactly balanced, category-stratified two-arm assignment.

    The source is scanned once for category counts, requiring only a handful
    of counters.  A second streaming scan performs the insert.  For each
    category the two conditions alternate; category start arms are chosen so
    the total number of CONTROL records is ``floor(n / 2)``.
    """

    category_counts: Counter[str] = Counter()
    total = 0
    for record in _iter_limited_records(source, limit=limit, stats=stats):
        category_counts[_category_key(record)] += 1
        total += 1

    # Every category contributes floor(n / 2) controls regardless of its start
    # arm. Allocate the required extra controls among odd-sized categories in
    # a stable sorted order to make the complete catalog exactly balanced.
    baseline_controls = sum(count // 2 for count in category_counts.values())
    required_extra_controls = (total // 2) - baseline_controls
    odd_categories = sorted(category for category, count in category_counts.items() if count % 2)
    control_start_for_odd = set(odd_categories[:required_extra_controls])
    starts: dict[str, str] = {}
    for category, count in category_counts.items():
        if count % 2:
            starts[category] = "CONTROL" if category in control_start_for_odd else "GEO_OPTIMIZED"
        else:
            # The start arm does not affect even-sized category balance.  A
            # stable character parity merely varies the ordering by category.
            starts[category] = CONDITIONS[sum(ord(char) for char in category) % 2]
    return starts, total


def _condition_for_record(record: AmazonCatalogRecord, ordinal: int, starts: dict[str, str]) -> str:
    start = starts[_category_key(record)]
    if ordinal % 2 == 0:
        return start
    return "GEO_OPTIMIZED" if start == "CONTROL" else "CONTROL"


def _product_values(
    record: AmazonCatalogRecord,
    *,
    condition: str,
    created_at: Any,
    builder: FactualGEOBuilder,
    validator: GEOIntegrityValidator,
) -> dict[str, Any]:
    values = record.catalog_product_values(condition=condition)
    values["created_at"] = created_at
    values["imported_at"] = created_at
    if condition == "GEO_OPTIMIZED":
        bundle = builder.build(values)
        report = validator.validate(values, bundle)
        if not report.valid:
            errors = "; ".join(issue.message for issue in report.errors)
            raise ValueError(f"GEO integrity validation failed for {record.source_product_id}: {errors}")
        values["geo_bundle"] = bundle
    else:
        values["geo_bundle"] = None
    return values


def _insert_batch(db: Session, values: list[dict[str, Any]]) -> tuple[int, int]:
    """Insert unseen source records and preserve existing study rows intact."""

    unique_by_key = {str(value["source_product_key"]): value for value in values}
    source_keys = list(unique_by_key)
    existing = set(
        db.scalars(select(Product.source_product_key).where(Product.source_product_key.in_(source_keys))).all()
    )
    pending = [value for key, value in unique_by_key.items() if key not in existing]
    if not pending:
        return 0, len(values)
    try:
        db.bulk_insert_mappings(Product, pending)
        db.commit()
    except IntegrityError:
        # A concurrent/import-resume race can surface as a unique key error.
        # Do not delete or overwrite research data; roll back and let a retry
        # safely detect the rows as existing.
        db.rollback()
        existing_after_rollback = set(
            db.scalars(select(Product.source_product_key).where(Product.source_product_key.in_(source_keys))).all()
        )
        retry = [value for key, value in unique_by_key.items() if key not in existing_after_rollback]
        if not retry:
            return 0, len(values)
        db.bulk_insert_mappings(Product, retry)
        db.commit()
        return len(retry), len(values) - len(retry)
    return len(pending), len(values) - len(pending)


def _existing_source_keys(db: Session, records: list[AmazonCatalogRecord]) -> set[str]:
    """Return only keys already durable in the target database for one batch."""

    source_keys = list({record.source_product_key for record in records})
    if not source_keys:
        return set()
    return set(
        db.scalars(select(Product.source_product_key).where(Product.source_product_key.in_(source_keys))).all()
    )


def import_catalog(
    source: Path,
    *,
    batch_size: int,
    limit: int = DEFAULT_LIMIT,
    dry_run: bool = False,
    unmodified: bool = False,
) -> ImportStats:
    """Create schema if needed and stream the source into persistent storage."""

    if not source.is_file():
        raise FileNotFoundError(f"Amazon catalog CSV was not found: {source}")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if limit < 0:
        raise ValueError("limit must be zero (all records) or a positive integer")

    stats = ImportStats(source=str(source), requested_limit=limit)
    starts, planned = plan_category_assignments(source, limit=limit, stats=stats)
    stats.planned_records = planned
    if dry_run:
        stats.control = planned if unmodified else planned // 2
        stats.geo_optimized = 0 if unmodified else planned - stats.control
        return stats

    # Models are imported above before metadata creation; create_all is
    # non-destructive and lets an empty PostgreSQL research database bootstrap
    # without a separate local setup command.
    Base.metadata.create_all(bind=database.engine)
    category_ordinals: Counter[str] = Counter()
    builder = FactualGEOBuilder()
    validator = GEOIntegrityValidator(builder)
    now = utc_now()

    # The second scan must not double-count audit flags/errors from the plan.
    write_stats = ImportStats(source=str(source), requested_limit=limit)
    with database.SessionLocal() as db:
        for records in iter_batches(
            _iter_limited_records(source, limit=limit, stats=write_stats), batch_size=batch_size
        ):
            # Resume cheaply: establish the persisted keys before creating
            # treatment JSON.  Otherwise a restarted run would regenerate
            # hundreds of thousands of GEO bundles only to skip them.
            existing_source_keys = _existing_source_keys(db, records)
            pending: list[tuple[AmazonCatalogRecord, str]] = []
            for record in records:
                category = _category_key(record)
                condition = "CONTROL" if unmodified else _condition_for_record(record, category_ordinals[category], starts)
                category_ordinals[category] += 1
                if record.source_product_key in existing_source_keys:
                    stats.skipped_existing += 1
                else:
                    pending.append((record, condition))
                if condition == "CONTROL":
                    stats.control += 1
                else:
                    stats.geo_optimized += 1

            values = [
                _product_values(
                    record,
                    condition=condition,
                    created_at=now,
                    builder=builder,
                    validator=validator,
                )
                for record, condition in pending
            ]
            if not values:
                continue
            inserted, existing = _insert_batch(db, values)
            stats.imported += inserted
            stats.skipped_existing += existing

    # Row validity and quality flags were observed twice (once per streaming
    # scan), but report each source record once using the write pass.
    stats.skipped_invalid = write_stats.skipped_invalid
    stats.invalid_examples = write_stats.invalid_examples
    stats.quality_flag_counts = write_stats.quality_flag_counts
    return stats


def _parser() -> argparse.ArgumentParser:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=settings.amazon_catalog_csv,
        help="Path to Amazon-Products.csv (defaults to GEO_AMAZON_CATALOG_CSV).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=settings.catalog_import_batch_size,
        help="Rows committed per transaction (default: GEO_CATALOG_IMPORT_BATCH_SIZE).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
        help="Import at most this many valid rows; zero imports all rows.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and plan assignment without creating tables or writing rows.",
    )
    parser.add_argument(
        "--unmodified",
        "--all-control",
        action="store_true",
        dest="unmodified",
        help="Import all products cleanly as CONTROL without applying any GEO modification/treatment.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.source is None:
        raise SystemExit("Set GEO_AMAZON_CATALOG_CSV or pass --source PATH.")
    result = import_catalog(
        Path(args.source).expanduser().resolve(),
        batch_size=args.batch_size,
        limit=args.limit,
        dry_run=args.dry_run,
        unmodified=args.unmodified,
    )
    print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
