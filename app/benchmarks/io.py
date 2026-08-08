"""Portable JSON, JSONL, and CSV input/output for normalised benchmark data.

This module uses only the Python standard library.  It accepts the common
shapes seen in research exports while keeping source-specific columns in model
metadata.  No loader assumes that a named external benchmark is present on the
local machine.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, TextIO

from .models import (
    BenchmarkCase,
    BenchmarkDataset,
    Candidate,
    Prediction,
    TargetSelection,
    TraceEvent,
)


PathOrStream = str | Path | TextIO


class BenchmarkIOError(ValueError):
    """A parse/serialisation error with record or line context where available."""


_FORMAT_ALIASES = {
    "json": "json",
    ".json": "json",
    "jsonl": "jsonl",
    ".jsonl": "jsonl",
    "ndjson": "jsonl",
    ".ndjson": "jsonl",
    "csv": "csv",
    ".csv": "csv",
}


def _format_for(value: PathOrStream, file_format: str | None = None) -> str:
    candidate = file_format
    if candidate is None and isinstance(value, (str, Path)):
        candidate = Path(value).suffix
    if candidate is None or not str(candidate).strip():
        # JSON is the least surprising default for in-memory text streams.
        return "json"
    key = str(candidate).strip().lower()
    try:
        return _FORMAT_ALIASES[key]
    except KeyError as exc:
        allowed = ", ".join(sorted(set(_FORMAT_ALIASES.values())))
        raise BenchmarkIOError(f"Unsupported benchmark file format '{candidate}'. Use one of: {allowed}.") from exc


def _read_text(source: PathOrStream, *, encoding: str) -> str:
    if hasattr(source, "read"):
        content = source.read()  # type: ignore[union-attr]
    else:
        content = Path(source).read_text(encoding=encoding)
    if isinstance(content, bytes):
        return content.decode(encoding)
    return str(content)


def _write_text(destination: PathOrStream, content: str, *, encoding: str) -> None:
    if hasattr(destination, "write"):
        destination.write(content)  # type: ignore[union-attr]
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding=encoding, newline="")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    return str(value)


def _as_record(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise BenchmarkIOError(f"{context} must be a JSON object / mapping, not {type(value).__name__}.")
    return {str(key): item for key, item in value.items()}


def _collection_records(value: Any, *, context: str) -> list[dict[str, Any]]:
    """Accept an array or an id-keyed mapping and always return record dicts."""

    if isinstance(value, Mapping):
        # Some data releases use a mapping keyed by case id.  Preserve that id
        # only when the record does not already provide one of the normal keys.
        if any(key in value for key in ("case_id", "query_id", "candidate_id", "product_id", "id", "query")):
            return [_as_record(value, context=context)]
        records: list[dict[str, Any]] = []
        for identifier, item in value.items():
            record = _as_record(item, context=f"{context}[{identifier!r}]")
            if not any(key in record for key in ("case_id", "query_id", "candidate_id", "product_id", "id")):
                record["id"] = str(identifier)
            records.append(record)
        return records
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise BenchmarkIOError(f"{context} must contain a list of records.")
    return [_as_record(item, context=f"{context}[{index}]") for index, item in enumerate(value, start=1)]


def _parse_json_payload(text: str, *, source_label: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise BenchmarkIOError(
            f"Invalid JSON in {source_label} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc


def _parse_jsonl_records(text: str, *, source_label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchmarkIOError(
                f"Invalid JSONL in {source_label} at line {line_number}, column {exc.colno}: {exc.msg}"
            ) from exc
        if isinstance(decoded, list):
            records.extend(_collection_records(decoded, context=f"{source_label} line {line_number}"))
        else:
            records.append(_as_record(decoded, context=f"{source_label} line {line_number}"))
    return records


def _parse_csv_cell(value: str | None) -> Any:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return ""
    # Nested structures written by write_records round-trip from CSV.  Leave
    # numbers as strings because field meaning (id vs. score) is model-specific.
    if (stripped.startswith("{") and stripped.endswith("}")) or (
        stripped.startswith("[") and stripped.endswith("]")
    ):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass
    if stripped.lower() == "null":
        return None
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False
    return stripped


def _parse_csv_records(text: str, *, source_label: str) -> list[dict[str, Any]]:
    try:
        # StringIO (rather than splitlines) preserves legal embedded newlines
        # inside quoted CSV fields such as product descriptions or answers.
        from io import StringIO

        reader = csv.DictReader(StringIO(text, newline=""))
        if not reader.fieldnames:
            raise BenchmarkIOError(f"CSV source {source_label} needs a header row.")
        records = []
        for line_number, row in enumerate(reader, start=2):
            if None in row:
                raise BenchmarkIOError(
                    f"CSV source {source_label} has more values than headers on line {line_number}."
                )
            if all(value in (None, "") for value in row.values()):
                continue
            records.append(
                {
                    str(key).strip(): _parse_csv_cell(value)
                    for key, value in row.items()
                    if key is not None and str(key).strip()
                }
            )
        return records
    except csv.Error as exc:
        raise BenchmarkIOError(f"Invalid CSV in {source_label}: {exc}") from exc


def _source_label(source: PathOrStream) -> str:
    if isinstance(source, (str, Path)):
        return str(source)
    return getattr(source, "name", "in-memory stream")


def _records_from_payload(payload: Any, *, collection_keys: Sequence[str], source_label: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return _collection_records(payload, context=source_label)
    if not isinstance(payload, Mapping):
        raise BenchmarkIOError(f"JSON source {source_label} must contain an object or list of objects.")
    for key in collection_keys:
        if key in payload:
            return _collection_records(payload[key], context=f"{source_label}.{key}")
    return [_as_record(payload, context=source_label)]


def read_records(
    source: PathOrStream,
    *,
    format: str | None = None,
    encoding: str = "utf-8-sig",
    collection_keys: Sequence[str] = ("records", "data", "items"),
) -> list[dict[str, Any]]:
    """Read generic mapping records from JSON, JSONL/NDJSON, or CSV.

    ``collection_keys`` lets typed loaders prioritise a top-level collection
    such as ``cases`` or ``predictions`` without hard-coding external schemas.
    """

    file_format = _format_for(source, format)
    source_label = _source_label(source)
    text = _read_text(source, encoding=encoding)
    if file_format == "json":
        return _records_from_payload(
            _parse_json_payload(text, source_label=source_label),
            collection_keys=collection_keys,
            source_label=source_label,
        )
    if file_format == "jsonl":
        return _parse_jsonl_records(text, source_label=source_label)
    return _parse_csv_records(text, source_label=source_label)


def _record_from_value(value: Any) -> dict[str, Any]:
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return _as_record(_json_safe(value), context="record")


def _csv_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(_json_safe(value), ensure_ascii=False, sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def write_records(
    destination: PathOrStream,
    records: Iterable[Any],
    *,
    format: str | None = None,
    encoding: str = "utf-8",
    fieldnames: Sequence[str] | None = None,
    indent: int = 2,
) -> None:
    """Write mappings or normalised model objects in an interoperable format."""

    rows = [_record_from_value(record) for record in records]
    file_format = _format_for(destination, format)
    if file_format == "json":
        _write_text(destination, json.dumps(rows, ensure_ascii=False, indent=indent, sort_keys=True) + "\n", encoding=encoding)
        return
    if file_format == "jsonl":
        content = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
        _write_text(destination, content, encoding=encoding)
        return

    ordered_fields: list[str] = [str(name) for name in fieldnames or ()]
    seen = set(ordered_fields)
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                ordered_fields.append(key)
    if not ordered_fields:
        raise BenchmarkIOError("Cannot write an empty CSV without explicit fieldnames.")
    # csv.DictWriter needs a file-like object; writing to StringIO keeps the
    # implementation identical for real paths and user-provided text streams.
    from io import StringIO

    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=ordered_fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: _csv_cell(row.get(key)) for key in ordered_fields})
    _write_text(destination, buffer.getvalue(), encoding=encoding)


def read_cases(
    source: PathOrStream,
    *,
    benchmark: str | None = None,
    format: str | None = None,
    encoding: str = "utf-8-sig",
) -> list[BenchmarkCase]:
    """Read and normalise benchmark query/candidate records."""

    records = read_records(
        source,
        format=format,
        encoding=encoding,
        collection_keys=("cases", "queries", "records", "data", "items"),
    )
    cases: list[BenchmarkCase] = []
    for index, record in enumerate(records, start=1):
        try:
            cases.append(
                BenchmarkCase.from_mapping(
                    record,
                    benchmark=benchmark,
                    fallback_case_id=f"case-{index:06d}",
                )
            )
        except (TypeError, ValueError) as exc:
            raise BenchmarkIOError(f"Invalid benchmark case at record {index}: {exc}") from exc
    return cases


def write_cases(
    destination: PathOrStream,
    cases: Iterable[BenchmarkCase | Mapping[str, Any]],
    *,
    format: str | None = None,
    encoding: str = "utf-8",
) -> None:
    write_records(destination, cases, format=format, encoding=encoding)


def read_predictions(
    source: PathOrStream,
    *,
    format: str | None = None,
    encoding: str = "utf-8-sig",
) -> list[Prediction]:
    """Read normalised or common provider-style ranking/answer outputs."""

    records = read_records(
        source,
        format=format,
        encoding=encoding,
        collection_keys=("predictions", "runs", "results", "records", "data", "items"),
    )
    predictions: list[Prediction] = []
    for index, record in enumerate(records, start=1):
        try:
            predictions.append(Prediction.from_mapping(record))
        except (TypeError, ValueError) as exc:
            raise BenchmarkIOError(f"Invalid prediction at record {index}: {exc}") from exc
    return predictions


def write_predictions(
    destination: PathOrStream,
    predictions: Iterable[Prediction | Mapping[str, Any]],
    *,
    format: str | None = None,
    encoding: str = "utf-8",
) -> None:
    write_records(destination, predictions, format=format, encoding=encoding)


def read_trace_events(
    source: PathOrStream,
    *,
    format: str | None = None,
    encoding: str = "utf-8-sig",
) -> list[TraceEvent]:
    """Read agent/sandbox event traces without interpreting their semantics."""

    records = read_records(
        source,
        format=format,
        encoding=encoding,
        collection_keys=("traces", "events", "trace_events", "records", "data", "items"),
    )
    events: list[TraceEvent] = []
    for index, record in enumerate(records, start=1):
        try:
            events.append(TraceEvent.from_mapping(record))
        except (TypeError, ValueError) as exc:
            raise BenchmarkIOError(f"Invalid trace event at record {index}: {exc}") from exc
    return events


def write_trace_events(
    destination: PathOrStream,
    events: Iterable[TraceEvent | Mapping[str, Any]],
    *,
    format: str | None = None,
    encoding: str = "utf-8",
) -> None:
    write_records(destination, events, format=format, encoding=encoding)


def read_target_selections(
    source: PathOrStream,
    *,
    format: str | None = None,
    encoding: str = "utf-8-sig",
) -> list[TargetSelection]:
    """Read a separate target/relevance selection table."""

    records = read_records(
        source,
        format=format,
        encoding=encoding,
        collection_keys=("target_selections", "targets", "relevance", "labels", "records", "data", "items"),
    )
    selections: list[TargetSelection] = []
    for index, record in enumerate(records, start=1):
        try:
            selections.append(TargetSelection.from_mapping(record))
        except (TypeError, ValueError) as exc:
            raise BenchmarkIOError(f"Invalid target selection at record {index}: {exc}") from exc
    return selections


def write_target_selections(
    destination: PathOrStream,
    selections: Iterable[TargetSelection | Mapping[str, Any]],
    *,
    format: str | None = None,
    encoding: str = "utf-8",
) -> None:
    write_records(destination, selections, format=format, encoding=encoding)


def merge_target_selections(
    cases: Iterable[BenchmarkCase | Mapping[str, Any]],
    selections: Iterable[TargetSelection | Mapping[str, Any]],
    *,
    on_missing_case: str = "keep",
    on_missing_candidate: str = "preserve",
) -> list[BenchmarkCase]:
    """Merge external target annotations into normalised benchmark cases.

    ``on_missing_case`` is ``"keep"`` (ignore unmatched selections) or
    ``"error"``.  ``on_missing_candidate`` is ``"preserve"`` (retain the
    external id in ``target_candidate_ids``), ``"placeholder"`` (append a
    minimal target candidate), or ``"error"``.  ``preserve`` is useful for
    trajectory benchmarks whose target product is intentionally not in a
    static local corpus.
    """

    if on_missing_case not in {"keep", "error"}:
        raise ValueError("on_missing_case must be 'keep' or 'error'.")
    if on_missing_candidate not in {"preserve", "placeholder", "error"}:
        raise ValueError("on_missing_candidate must be 'preserve', 'placeholder', or 'error'.")

    normalised_cases = [
        case if isinstance(case, BenchmarkCase) else BenchmarkCase.from_mapping(case)
        for case in cases
    ]
    selection_index: dict[str, list[TargetSelection]] = {}
    for item in selections:
        selection = item if isinstance(item, TargetSelection) else TargetSelection.from_mapping(item)
        selection_index.setdefault(selection.case_id, []).append(selection)

    case_ids = {case.case_id for case in normalised_cases}
    unmatched = [case_id for case_id in selection_index if case_id not in case_ids]
    if unmatched and on_missing_case == "error":
        preview = ", ".join(sorted(unmatched)[:5])
        suffix = "..." if len(unmatched) > 5 else ""
        raise BenchmarkIOError(f"Target selections reference missing case ids: {preview}{suffix}")

    merged: list[BenchmarkCase] = []
    for case in normalised_cases:
        case_selections = selection_index.get(case.case_id, [])
        if not case_selections:
            merged.append(case)
            continue

        selected_targets = list(case.target_candidate_ids)
        target_set = set(selected_targets)
        candidate_by_id = {candidate.candidate_id: candidate for candidate in case.candidates}
        ordered_candidates = list(case.candidates)

        for selection in case_selections:
            candidate = candidate_by_id.get(selection.candidate_id)
            if candidate is None:
                if on_missing_candidate == "error":
                    raise BenchmarkIOError(
                        f"Target selection for case '{case.case_id}' references unknown candidate "
                        f"'{selection.candidate_id}'."
                    )
                if on_missing_candidate == "placeholder":
                    candidate = Candidate(
                        candidate_id=selection.candidate_id,
                        is_target=selection.selected,
                        relevance=selection.relevance,
                        relevance_label=selection.relevance_label,
                        source=selection.source,
                        protocol=selection.protocol,
                        metadata={"target_selection_placeholder": True, **selection.metadata},
                    )
                    candidate_by_id[candidate.candidate_id] = candidate
                    ordered_candidates.append(candidate)
            else:
                # Keep product/document content but update only gold-label fields.
                updated_metadata = dict(candidate.metadata)
                updated_metadata.setdefault("target_selection_sources", [])
                source_record = {
                    "source": selection.source,
                    "protocol": selection.protocol,
                }
                if source_record not in updated_metadata["target_selection_sources"]:
                    updated_metadata["target_selection_sources"].append(source_record)
                candidate = replace(
                    candidate,
                    is_target=selection.selected,
                    relevance=selection.relevance if selection.relevance is not None else candidate.relevance,
                    relevance_label=selection.relevance_label or candidate.relevance_label,
                    source=selection.source or candidate.source,
                    protocol=selection.protocol or candidate.protocol,
                    metadata=updated_metadata,
                )
                candidate_by_id[candidate.candidate_id] = candidate
                for index, current in enumerate(ordered_candidates):
                    if current.candidate_id == candidate.candidate_id:
                        ordered_candidates[index] = candidate
                        break

            if selection.selected:
                if selection.candidate_id not in target_set:
                    target_set.add(selection.candidate_id)
                    selected_targets.append(selection.candidate_id)
            else:
                target_set.discard(selection.candidate_id)
                selected_targets = [identifier for identifier in selected_targets if identifier != selection.candidate_id]

        case_metadata = dict(case.metadata)
        provenance = list(case_metadata.get("target_selection_provenance", []))
        for selection in case_selections:
            provenance.append(
                {
                    "candidate_id": selection.candidate_id,
                    "selected": selection.selected,
                    "source": selection.source,
                    "protocol": selection.protocol,
                }
            )
        case_metadata["target_selection_provenance"] = provenance
        merged.append(
            replace(
                case,
                candidates=tuple(ordered_candidates),
                target_candidate_ids=tuple(selected_targets),
                metadata=case_metadata,
            )
        )
    return merged


def read_benchmark_dataset(
    source: PathOrStream,
    *,
    benchmark: str | None = None,
    format: str | None = None,
    encoding: str = "utf-8-sig",
) -> BenchmarkDataset:
    """Read cases and, for JSON manifests, retain dataset-level provenance."""

    file_format = _format_for(source, format)
    dataset_metadata: dict[str, Any] = {}
    manifest: Mapping[str, Any] | None = None
    if file_format == "json":
        text = _read_text(source, encoding=encoding)
        payload = _parse_json_payload(text, source_label=_source_label(source))
        if isinstance(payload, Mapping) and "cases" in payload:
            manifest = payload
            case_records = _collection_records(payload["cases"], context=f"{_source_label(source)}.cases")
            cases = [
                BenchmarkCase.from_mapping(record, benchmark=benchmark or payload.get("benchmark"), fallback_case_id=f"case-{index:06d}")
                for index, record in enumerate(case_records, start=1)
            ]
            dataset_metadata = dict(payload.get("metadata") or {}) if isinstance(payload.get("metadata"), Mapping) else {}
        else:
            # Re-read via a memory-free direct conversion rather than passing a
            # potentially exhausted user stream back to read_cases.
            case_records = _records_from_payload(payload, collection_keys=("cases", "queries", "records", "data", "items"), source_label=_source_label(source))
            cases = [
                BenchmarkCase.from_mapping(record, benchmark=benchmark, fallback_case_id=f"case-{index:06d}")
                for index, record in enumerate(case_records, start=1)
            ]
    else:
        cases = read_cases(source, benchmark=benchmark, format=file_format, encoding=encoding)

    resolved_benchmark = benchmark or (manifest.get("benchmark") if manifest else None)
    if not resolved_benchmark and cases:
        resolved_benchmark = cases[0].benchmark
    return BenchmarkDataset(
        benchmark=str(resolved_benchmark or "unspecified"),
        cases=tuple(cases),
        source=(manifest.get("source") if manifest else None),
        protocol=(manifest.get("protocol") if manifest else None),
        version=(manifest.get("version") if manifest else None),
        split=(manifest.get("split") if manifest else None),
        metadata=dataset_metadata,
    )


def write_benchmark_dataset(
    destination: PathOrStream,
    dataset: BenchmarkDataset,
    *,
    format: str | None = None,
    encoding: str = "utf-8",
    indent: int = 2,
) -> None:
    """Write a full JSON dataset manifest or case rows in JSONL/CSV."""

    file_format = _format_for(destination, format)
    if file_format == "json":
        _write_text(
            destination,
            json.dumps(dataset.to_dict(), ensure_ascii=False, indent=indent, sort_keys=True) + "\n",
            encoding=encoding,
        )
        return
    # JSONL/CSV have no canonical dataset-manifest container.  Cases remain
    # portable, and their own benchmark/source/protocol fields are retained.
    write_cases(destination, dataset.cases, format=file_format, encoding=encoding)


# Short, discoverable aliases for callers that prefer load/save terminology.
load_cases = read_cases
load_predictions = read_predictions
load_trace_events = read_trace_events
load_target_selections = read_target_selections
save_cases = write_cases
save_predictions = write_predictions
save_trace_events = write_trace_events
save_target_selections = write_target_selections


__all__ = [
    "BenchmarkIOError",
    "PathOrStream",
    "load_cases",
    "load_predictions",
    "load_target_selections",
    "load_trace_events",
    "merge_target_selections",
    "read_benchmark_dataset",
    "read_cases",
    "read_predictions",
    "read_records",
    "read_target_selections",
    "read_trace_events",
    "save_cases",
    "save_predictions",
    "save_target_selections",
    "save_trace_events",
    "write_benchmark_dataset",
    "write_cases",
    "write_predictions",
    "write_records",
    "write_target_selections",
    "write_trace_events",
]
