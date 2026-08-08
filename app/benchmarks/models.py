"""Normalised, dependency-free records for GEO benchmark evaluation.

Benchmark releases use different names for nearly identical concepts (for
example, ``query_id`` vs. ``prompt_id`` and ``product_id`` vs. ``document``).
The classes in this module deliberately provide one small, typed interchange
format.  They do not claim that a given external dataset is installed or that
all benchmark protocols use the same relevance labels.

All ``from_mapping`` constructors preserve unfamiliar fields in ``metadata``.
This is important for research reproducibility: a loader should not silently
discard benchmark-specific annotations merely because the common evaluator
does not currently use them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
import json
from typing import Any, TypeAlias


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]


def _text(value: Any, default: str = "") -> str:
    """Return a stable, whitespace-trimmed string without turning None into 'None'."""

    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Get the first present, non-empty field from a heterogeneous record."""

    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return default


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return {}
            if isinstance(parsed, Mapping):
                return {str(key): item for key, item in parsed.items()}
    return {}


def _as_sequence(value: Any) -> list[Any]:
    """Make JSON arrays, comma-separated CSV cells, and scalar values iterable."""

    if value is None or value == "":
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if (stripped.startswith("[") and stripped.endswith("]")) or (
            stripped.startswith("{") and stripped.endswith("}")
        ):
            try:
                decoded = json.loads(stripped)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, Mapping):
                return list(decoded.values())
            if isinstance(decoded, list):
                return decoded
        # A simple comma-separated cell is common in exported benchmark CSVs.
        return [part.strip() for part in stripped.split(",") if part.strip()]
    if isinstance(value, Mapping):
        return list(value.values())
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return list(value)
    return [value]


def _dedupe_text(values: Iterable[Any]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        item = _text(value)
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return tuple(result)


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _bool_or_none(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = _text(value).lower()
    if lowered in {"1", "true", "t", "yes", "y", "selected", "target", "relevant", "cited"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "not_selected", "non_target", "irrelevant", "uncited"}:
        return False
    return None


def _json_value(value: Any) -> JSONValue:
    """Convert arbitrary common Python values to a JSON-safe, copy-owned value."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return [_json_value(item) for item in value]
    return str(value)


def _metadata(
    mapping: Mapping[str, Any],
    *,
    known: Iterable[str],
    explicit: Any = None,
) -> dict[str, JSONValue]:
    """Merge an explicit metadata object with unknown source columns."""

    output: dict[str, JSONValue] = {}
    if isinstance(explicit, Mapping):
        output.update({str(key): _json_value(value) for key, value in explicit.items()})
    known_keys = set(known)
    for key, value in mapping.items():
        if key not in known_keys:
            output[str(key)] = _json_value(value)
    return output


def _normalise_timestamp(value: Any) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    text = _text(value)
    if not text:
        return None
    # Preserve an unparseable provider timestamp rather than losing it.  ISO
    # timestamps are normalized where possible only.
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return text
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.isoformat()


@dataclass(frozen=True, slots=True)
class CitationSpan:
    """A cited source/product span emitted by an answer engine or agent."""

    candidate_id: str
    start: int | None = None
    end: int | None = None
    text: str | None = None
    claim: str | None = None
    source_url: str | None = None
    citation_id: str | None = None
    entails: bool | None = None
    source: str | None = None
    protocol: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        candidate_id = _text(self.candidate_id)
        if not candidate_id:
            raise ValueError("CitationSpan.candidate_id is required.")
        start = _int_or_none(self.start)
        end = _int_or_none(self.end)
        if start is not None and start < 0:
            raise ValueError("CitationSpan.start cannot be negative.")
        if end is not None and end < 0:
            raise ValueError("CitationSpan.end cannot be negative.")
        if start is not None and end is not None and end < start:
            raise ValueError("CitationSpan.end cannot precede CitationSpan.start.")
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)
        object.__setattr__(self, "text", _text(self.text) or None)
        object.__setattr__(self, "claim", _text(self.claim) or None)
        object.__setattr__(self, "source_url", _text(self.source_url) or None)
        object.__setattr__(self, "citation_id", _text(self.citation_id) or None)
        object.__setattr__(self, "entails", _bool_or_none(self.entails))
        object.__setattr__(self, "source", _text(self.source) or None)
        object.__setattr__(self, "protocol", _text(self.protocol) or None)
        object.__setattr__(self, "metadata", _json_value(self.metadata))

    @property
    def product_id(self) -> str:
        """Compatibility alias for product-centric e-commerce releases."""

        return self.candidate_id

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> "CitationSpan":
        known = {
            "candidate_id", "product_id", "document_id", "doc_id", "source_id", "id", "citation_id",
            "start", "start_char", "start_index", "end", "end_char", "end_index", "text", "span",
            "citation_text", "claim", "claim_text", "source_url", "url", "link", "entails",
            "citation_entails", "is_entailed", "source", "protocol", "metadata",
        }
        return cls(
            candidate_id=_first(record, "candidate_id", "product_id", "document_id", "doc_id", "source_id", "id"),
            start=_first(record, "start", "start_char", "start_index"),
            end=_first(record, "end", "end_char", "end_index"),
            text=_first(record, "text", "span", "citation_text"),
            claim=_first(record, "claim", "claim_text"),
            source_url=_first(record, "source_url", "url", "link"),
            citation_id=_first(record, "citation_id", "id"),
            entails=_first(record, "entails", "citation_entails", "is_entailed"),
            source=_first(record, "source"),
            protocol=_first(record, "protocol"),
            metadata=_metadata(record, known=known, explicit=_as_mapping(record.get("metadata"))),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "candidate_id": self.candidate_id,
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "claim": self.claim,
            "source_url": self.source_url,
            "citation_id": self.citation_id,
            "entails": self.entails,
            "source": self.source,
            "protocol": self.protocol,
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class Candidate:
    """A product, web document, or other recommendation candidate."""

    candidate_id: str
    title: str | None = None
    text: str | None = None
    source_url: str | None = None
    rank: int | None = None
    score: float | None = None
    relevance: float | None = None
    relevance_label: str | None = None
    is_target: bool | None = None
    condition: str | None = None
    source: str | None = None
    protocol: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        candidate_id = _text(self.candidate_id)
        if not candidate_id:
            raise ValueError("Candidate.candidate_id is required.")
        rank = _int_or_none(self.rank)
        if rank is not None and rank < 1:
            raise ValueError("Candidate.rank must be at least 1 when supplied.")
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "title", _text(self.title) or None)
        object.__setattr__(self, "text", _text(self.text) or None)
        object.__setattr__(self, "source_url", _text(self.source_url) or None)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "score", _float_or_none(self.score))
        object.__setattr__(self, "relevance", _float_or_none(self.relevance))
        object.__setattr__(self, "relevance_label", _text(self.relevance_label) or None)
        object.__setattr__(self, "is_target", _bool_or_none(self.is_target))
        object.__setattr__(self, "condition", _text(self.condition) or None)
        object.__setattr__(self, "source", _text(self.source) or None)
        object.__setattr__(self, "protocol", _text(self.protocol) or None)
        object.__setattr__(self, "metadata", _json_value(self.metadata))

    @property
    def id(self) -> str:
        """Short alias useful for generic ranking code."""

        return self.candidate_id

    @property
    def product_id(self) -> str:
        """Compatibility alias for product-centric benchmark releases."""

        return self.candidate_id

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> "Candidate":
        known = {
            "candidate_id", "id", "product_id", "document_id", "doc_id", "item_id", "sku", "url",
            "source_url", "link", "title", "name", "product_name", "document_title", "text", "content",
            "description", "snippet", "body", "rank", "position", "rank_position", "score",
            "relevance", "relevance_score", "grade", "label", "relevance_label", "is_target", "target",
            "selected", "condition", "source", "protocol", "metadata",
        }
        candidate_id = _first(
            record,
            "candidate_id",
            "id",
            "product_id",
            "document_id",
            "doc_id",
            "item_id",
            "sku",
            "url",
        )
        return cls(
            candidate_id=candidate_id,
            title=_first(record, "title", "name", "product_name", "document_title"),
            text=_first(record, "text", "content", "description", "snippet", "body"),
            source_url=_first(record, "source_url", "url", "link"),
            rank=_first(record, "rank", "position", "rank_position"),
            score=_first(record, "score"),
            relevance=_first(record, "relevance", "relevance_score", "grade"),
            relevance_label=_first(record, "relevance_label", "label"),
            is_target=_first(record, "is_target", "target", "selected"),
            condition=_first(record, "condition"),
            source=_first(record, "source"),
            protocol=_first(record, "protocol"),
            metadata=_metadata(record, known=known, explicit=_as_mapping(record.get("metadata"))),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "candidate_id": self.candidate_id,
            "title": self.title,
            "text": self.text,
            "source_url": self.source_url,
            "rank": self.rank,
            "score": self.score,
            "relevance": self.relevance,
            "relevance_label": self.relevance_label,
            "is_target": self.is_target,
            "condition": self.condition,
            "source": self.source,
            "protocol": self.protocol,
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class TargetSelection:
    """An externally supplied relevance or target-product selection.

    Several GEO datasets keep target selections in a separate table.  Keeping
    the selection source and protocol alongside the target avoids treating a
    benchmark author's annotation as if it were a system prediction.
    """

    case_id: str
    candidate_id: str
    selected: bool = True
    relevance: float | None = None
    relevance_label: str | None = None
    source: str | None = None
    protocol: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        case_id = _text(self.case_id)
        candidate_id = _text(self.candidate_id)
        if not case_id:
            raise ValueError("TargetSelection.case_id is required.")
        if not candidate_id:
            raise ValueError("TargetSelection.candidate_id is required.")
        selected = _bool_or_none(self.selected)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "selected", True if selected is None else selected)
        object.__setattr__(self, "relevance", _float_or_none(self.relevance))
        object.__setattr__(self, "relevance_label", _text(self.relevance_label) or None)
        object.__setattr__(self, "source", _text(self.source) or None)
        object.__setattr__(self, "protocol", _text(self.protocol) or None)
        object.__setattr__(self, "metadata", _json_value(self.metadata))

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any]) -> "TargetSelection":
        known = {
            "case_id", "query_id", "qid", "id", "prompt_id", "candidate_id", "product_id", "target_id",
            "document_id", "doc_id", "item_id", "selected", "is_target", "target", "relevance",
            "relevance_score", "grade", "relevance_label", "label", "source", "protocol", "metadata",
        }
        return cls(
            case_id=_first(record, "case_id", "query_id", "qid", "prompt_id", "id"),
            candidate_id=_first(record, "candidate_id", "product_id", "target_id", "document_id", "doc_id", "item_id"),
            selected=_first(record, "selected", "is_target", "target", default=True),
            relevance=_first(record, "relevance", "relevance_score", "grade"),
            relevance_label=_first(record, "relevance_label", "label"),
            source=_first(record, "source"),
            protocol=_first(record, "protocol"),
            metadata=_metadata(record, known=known, explicit=_as_mapping(record.get("metadata"))),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "case_id": self.case_id,
            "candidate_id": self.candidate_id,
            "selected": self.selected,
            "relevance": self.relevance,
            "relevance_label": self.relevance_label,
            "source": self.source,
            "protocol": self.protocol,
            "metadata": _json_value(self.metadata),
        }


def _candidate_records(value: Any) -> list[Mapping[str, Any]]:
    """Coerce candidate lists and id-keyed candidate dictionaries to records."""

    if isinstance(value, Mapping):
        # A single candidate object has an explicit identifier/title.  An
        # id-keyed corpus does not, so carry the mapping key into the record.
        if any(key in value for key in ("candidate_id", "id", "product_id", "document_id", "title", "name")):
            return [value]
        output: list[Mapping[str, Any]] = []
        for key, item in value.items():
            if isinstance(item, Mapping):
                copied = dict(item)
                copied.setdefault("candidate_id", str(key))
                output.append(copied)
            else:
                output.append({"candidate_id": str(key), "text": item})
        return output
    output = []
    for item in _as_sequence(value):
        if isinstance(item, Mapping):
            output.append(item)
        else:
            output.append({"candidate_id": item})
    return output


def _target_ids(value: Any) -> tuple[str, ...]:
    if isinstance(value, Mapping):
        # Support {candidate_id: true/false} as well as a record-like object.
        if any(key in value for key in ("candidate_id", "product_id", "target_id", "id")):
            value = [value]
        else:
            return _dedupe_text(key for key, selected in value.items() if _bool_or_none(selected) is not False)
    identifiers: list[Any] = []
    for item in _as_sequence(value):
        if isinstance(item, Mapping):
            identifier = _first(item, "candidate_id", "product_id", "target_id", "document_id", "doc_id", "id")
            selected = _bool_or_none(_first(item, "selected", "is_target", "target", default=True))
            if selected is not False:
                identifiers.append(identifier)
        else:
            identifiers.append(item)
    return _dedupe_text(identifiers)


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    """A benchmark query, its candidate pool, and optional gold targets."""

    case_id: str
    query: str
    benchmark: str = "unspecified"
    candidates: tuple[Candidate, ...] = ()
    target_candidate_ids: tuple[str, ...] = ()
    split: str | None = None
    query_context: str | None = None
    constraints: dict[str, JSONValue] = field(default_factory=dict)
    source: str | None = None
    protocol: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        case_id = _text(self.case_id)
        query = _text(self.query)
        benchmark = _text(self.benchmark, "unspecified") or "unspecified"
        if not case_id:
            raise ValueError("BenchmarkCase.case_id is required.")
        if not query:
            raise ValueError("BenchmarkCase.query is required.")
        candidates = tuple(
            candidate if isinstance(candidate, Candidate) else Candidate.from_mapping(candidate)
            for candidate in self.candidates
        )
        identifiers = [candidate.candidate_id for candidate in candidates]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError(f"BenchmarkCase '{case_id}' contains duplicate candidate_id values.")
        target_ids = _dedupe_text(self.target_candidate_ids)
        # Candidate-level target flags are valid gold annotations too.  Retain
        # the explicit target order first, then append candidate-marked targets.
        target_ids = _dedupe_text(
            [*target_ids, *(candidate.candidate_id for candidate in candidates if candidate.is_target is True)]
        )
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "benchmark", benchmark)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "target_candidate_ids", target_ids)
        object.__setattr__(self, "split", _text(self.split) or None)
        object.__setattr__(self, "query_context", _text(self.query_context) or None)
        object.__setattr__(self, "constraints", _json_value(self.constraints))
        object.__setattr__(self, "source", _text(self.source) or None)
        object.__setattr__(self, "protocol", _text(self.protocol) or None)
        object.__setattr__(self, "metadata", _json_value(self.metadata))

    @property
    def id(self) -> str:
        return self.case_id

    @property
    def query_id(self) -> str:
        return self.case_id

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self.candidates)

    @property
    def target_ids(self) -> tuple[str, ...]:
        return self.target_candidate_ids

    def candidate(self, candidate_id: str) -> Candidate | None:
        target = _text(candidate_id)
        return next((item for item in self.candidates if item.candidate_id == target), None)

    @classmethod
    def from_mapping(
        cls,
        record: Mapping[str, Any],
        *,
        benchmark: str | None = None,
        fallback_case_id: str | None = None,
    ) -> "BenchmarkCase":
        known = {
            "case_id", "id", "query_id", "qid", "prompt_id", "request_id", "query", "question", "prompt",
            "user_query", "text", "benchmark", "benchmark_id", "dataset", "suite", "candidates", "products",
            "documents", "items", "candidate_products", "candidate_documents", "corpus", "target_candidate_ids",
            "target_ids", "target_product_ids", "target_document_ids", "relevant_ids", "gold_ids", "answer_ids",
            "targets", "split", "partition", "set", "query_context", "context", "user_context", "scenario",
            "constraints", "filters", "source", "protocol", "metadata",
        }
        case_id = _first(record, "case_id", "id", "query_id", "qid", "prompt_id", "request_id", default=fallback_case_id)
        candidates_value = _first(
            record,
            "candidates",
            "products",
            "documents",
            "items",
            "candidate_products",
            "candidate_documents",
            "corpus",
            default=[],
        )
        candidates = tuple(Candidate.from_mapping(item) for item in _candidate_records(candidates_value))
        targets = _target_ids(
            _first(
                record,
                "target_candidate_ids",
                "target_ids",
                "target_product_ids",
                "target_document_ids",
                "relevant_ids",
                "gold_ids",
                "answer_ids",
                "targets",
                default=[],
            )
        )
        constraints = _as_mapping(_first(record, "constraints", "filters", default={}))
        return cls(
            case_id=case_id,
            query=_first(record, "query", "question", "prompt", "user_query", "text"),
            benchmark=benchmark or _first(record, "benchmark", "benchmark_id", "dataset", "suite", default="unspecified"),
            candidates=candidates,
            target_candidate_ids=targets,
            split=_first(record, "split", "partition", "set"),
            query_context=_first(record, "query_context", "context", "user_context", "scenario"),
            constraints=constraints,
            source=_first(record, "source"),
            protocol=_first(record, "protocol"),
            metadata=_metadata(record, known=known, explicit=_as_mapping(record.get("metadata"))),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "case_id": self.case_id,
            "query": self.query,
            "benchmark": self.benchmark,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "target_candidate_ids": list(self.target_candidate_ids),
            "split": self.split,
            "query_context": self.query_context,
            "constraints": _json_value(self.constraints),
            "source": self.source,
            "protocol": self.protocol,
            "metadata": _json_value(self.metadata),
        }


def _ranked_ids_and_scores(value: Any) -> tuple[tuple[str, ...], dict[str, float]]:
    """Normalise ranking lists, ranking dictionaries, and candidate records."""

    if isinstance(value, Mapping):
        # A record-like mapping represents one result; an id->score/rank map
        # represents a whole ranking.
        if any(key in value for key in ("candidate_id", "product_id", "document_id", "id")):
            items: list[Any] = [value]
        else:
            sortable: list[tuple[int, str, Any]] = []
            for index, (candidate_id, item) in enumerate(value.items()):
                if isinstance(item, Mapping):
                    rank = _int_or_none(_first(item, "rank", "position", "rank_position"))
                    sortable.append((rank if rank is not None else index + 1, str(candidate_id), item))
                elif isinstance(item, (int, float)):
                    sortable.append((index + 1, str(candidate_id), {"score": item}))
                else:
                    sortable.append((index + 1, str(candidate_id), {}))
            sortable.sort(key=lambda item: (item[0], item[1]))
            items = [dict(item, candidate_id=candidate_id) if isinstance(item, Mapping) else {"candidate_id": candidate_id} for _, candidate_id, item in sortable]
    else:
        items = _as_sequence(value)

    ranked: list[str] = []
    scores: dict[str, float] = {}
    for item in items:
        if isinstance(item, Mapping):
            candidate_id = _text(_first(item, "candidate_id", "product_id", "document_id", "doc_id", "id", "item_id"))
            score = _float_or_none(_first(item, "score", "retrieval_score", "ranking_score"))
        else:
            candidate_id = _text(item)
            score = None
        if candidate_id and candidate_id not in ranked:
            ranked.append(candidate_id)
            if score is not None:
                scores[candidate_id] = score
    return tuple(ranked), scores


@dataclass(frozen=True, slots=True)
class Prediction:
    """One system/agent output for one benchmark case and run."""

    case_id: str
    ranked_candidate_ids: tuple[str, ...] = ()
    run_id: str = "default"
    scores: dict[str, float] = field(default_factory=dict)
    cited_candidate_ids: tuple[str, ...] = ()
    citations: tuple[CitationSpan, ...] = ()
    answer: str | None = None
    model_name: str | None = None
    model_version: str | None = None
    condition: str | None = None
    trace_id: str | None = None
    latency_ms: float | None = None
    source: str | None = None
    protocol: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        case_id = _text(self.case_id)
        if not case_id:
            raise ValueError("Prediction.case_id is required.")
        citations = tuple(
            citation if isinstance(citation, CitationSpan) else CitationSpan.from_mapping(citation)
            for citation in self.citations
        )
        cited_ids = _dedupe_text([*self.cited_candidate_ids, *(citation.candidate_id for citation in citations)])
        score_values: dict[str, float] = {}
        for key, value in self.scores.items():
            score = _float_or_none(value)
            identifier = _text(key)
            if identifier and score is not None:
                score_values[identifier] = score
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "ranked_candidate_ids", _dedupe_text(self.ranked_candidate_ids))
        object.__setattr__(self, "run_id", _text(self.run_id, "default") or "default")
        object.__setattr__(self, "scores", score_values)
        object.__setattr__(self, "cited_candidate_ids", cited_ids)
        object.__setattr__(self, "citations", citations)
        object.__setattr__(self, "answer", _text(self.answer) or None)
        object.__setattr__(self, "model_name", _text(self.model_name) or None)
        object.__setattr__(self, "model_version", _text(self.model_version) or None)
        object.__setattr__(self, "condition", _text(self.condition) or None)
        object.__setattr__(self, "trace_id", _text(self.trace_id) or None)
        object.__setattr__(self, "latency_ms", _float_or_none(self.latency_ms))
        object.__setattr__(self, "source", _text(self.source) or None)
        object.__setattr__(self, "protocol", _text(self.protocol) or None)
        object.__setattr__(self, "metadata", _json_value(self.metadata))

    @property
    def query_id(self) -> str:
        return self.case_id

    @property
    def ranked_ids(self) -> tuple[str, ...]:
        return self.ranked_candidate_ids

    @property
    def cited_ids(self) -> tuple[str, ...]:
        return self.cited_candidate_ids

    def rank_of(self, candidate_id: str) -> int | None:
        target = _text(candidate_id)
        try:
            return self.ranked_candidate_ids.index(target) + 1
        except ValueError:
            return None

    def cites(self, candidate_id: str) -> bool:
        return _text(candidate_id) in self.cited_candidate_ids

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any], *, fallback_case_id: str | None = None) -> "Prediction":
        known = {
            "case_id", "query_id", "qid", "prompt_id", "id", "run_id", "run", "trial_id", "ranking",
            "ranked_candidate_ids", "ranked_ids", "ranked_results", "recommendations", "results", "candidates",
            "scores", "score_map", "cited_candidate_ids", "cited_ids", "citation_ids", "citations",
            "citation_spans", "evidence", "sources", "answer", "response", "output", "generated_answer",
            "model_name", "model", "engine", "agent", "model_version", "version", "condition", "trace_id",
            "latency_ms", "latency", "source", "protocol", "metadata",
        }
        ranking_value = _first(
            record,
            "ranked_candidate_ids",
            "ranked_ids",
            "ranking",
            "ranked_results",
            "recommendations",
            "results",
            "candidates",
            default=[],
        )
        ranked_ids, inferred_scores = _ranked_ids_and_scores(ranking_value)
        explicit_scores = _as_mapping(_first(record, "scores", "score_map", default={}))
        for key, value in explicit_scores.items():
            score = _float_or_none(value)
            if score is not None:
                inferred_scores[str(key)] = score
        citation_value = _first(record, "citations", "citation_spans", "evidence", "sources", default=[])
        citations = tuple(
            CitationSpan.from_mapping(item)
            for item in _as_sequence(citation_value)
            if isinstance(item, Mapping)
        )
        cited_ids = _target_ids(_first(record, "cited_candidate_ids", "cited_ids", "citation_ids", default=[]))
        return cls(
            case_id=_first(record, "case_id", "query_id", "qid", "prompt_id", "id", default=fallback_case_id),
            ranked_candidate_ids=ranked_ids,
            run_id=_first(record, "run_id", "run", "trial_id", default="default"),
            scores=inferred_scores,
            cited_candidate_ids=cited_ids,
            citations=citations,
            answer=_first(record, "answer", "response", "output", "generated_answer"),
            model_name=_first(record, "model_name", "model", "engine", "agent"),
            model_version=_first(record, "model_version", "version"),
            condition=_first(record, "condition"),
            trace_id=_first(record, "trace_id"),
            latency_ms=_first(record, "latency_ms", "latency"),
            source=_first(record, "source"),
            protocol=_first(record, "protocol"),
            metadata=_metadata(record, known=known, explicit=_as_mapping(record.get("metadata"))),
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "case_id": self.case_id,
            "ranked_candidate_ids": list(self.ranked_candidate_ids),
            "run_id": self.run_id,
            "scores": dict(self.scores),
            "cited_candidate_ids": list(self.cited_candidate_ids),
            "citations": [citation.to_dict() for citation in self.citations],
            "answer": self.answer,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "condition": self.condition,
            "trace_id": self.trace_id,
            "latency_ms": self.latency_ms,
            "source": self.source,
            "protocol": self.protocol,
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class TraceEvent:
    """A protocol-level observation from an agentic, multi-step run."""

    case_id: str
    event_type: str
    run_id: str = "default"
    event_id: str | None = None
    timestamp: str | None = None
    step: int | None = None
    actor: str | None = None
    input_candidate_ids: tuple[str, ...] = ()
    output_candidate_ids: tuple[str, ...] = ()
    source: str | None = None
    protocol: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        case_id = _text(self.case_id)
        event_type = _text(self.event_type)
        if not case_id:
            raise ValueError("TraceEvent.case_id is required.")
        if not event_type:
            raise ValueError("TraceEvent.event_type is required.")
        step = _int_or_none(self.step)
        if step is not None and step < 0:
            raise ValueError("TraceEvent.step cannot be negative.")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "event_type", event_type)
        object.__setattr__(self, "run_id", _text(self.run_id, "default") or "default")
        object.__setattr__(self, "event_id", _text(self.event_id) or None)
        object.__setattr__(self, "timestamp", _normalise_timestamp(self.timestamp))
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "actor", _text(self.actor) or None)
        object.__setattr__(self, "input_candidate_ids", _dedupe_text(self.input_candidate_ids))
        object.__setattr__(self, "output_candidate_ids", _dedupe_text(self.output_candidate_ids))
        object.__setattr__(self, "source", _text(self.source) or None)
        object.__setattr__(self, "protocol", _text(self.protocol) or None)
        object.__setattr__(self, "metadata", _json_value(self.metadata))

    @property
    def query_id(self) -> str:
        return self.case_id

    @classmethod
    def from_mapping(cls, record: Mapping[str, Any], *, fallback_case_id: str | None = None) -> "TraceEvent":
        known = {
            "case_id", "query_id", "qid", "prompt_id", "event_type", "type", "event", "action", "name",
            "run_id", "run", "trial_id", "event_id", "id", "timestamp", "time", "created_at", "step",
            "sequence", "turn", "actor", "agent", "input_candidate_ids", "input_ids", "input_products",
            "input_documents", "output_candidate_ids", "output_ids", "output_products", "output_documents",
            "source", "protocol", "metadata", "payload",
        }
        metadata = _metadata(record, known=known, explicit=_as_mapping(record.get("metadata")))
        payload = _as_mapping(record.get("payload"))
        if payload:
            metadata.setdefault("payload", _json_value(payload))
        return cls(
            case_id=_first(record, "case_id", "query_id", "qid", "prompt_id", default=fallback_case_id),
            event_type=_first(record, "event_type", "type", "event", "action", "name"),
            run_id=_first(record, "run_id", "run", "trial_id", default="default"),
            event_id=_first(record, "event_id", "id"),
            timestamp=_first(record, "timestamp", "time", "created_at"),
            step=_first(record, "step", "sequence", "turn"),
            actor=_first(record, "actor", "agent"),
            input_candidate_ids=_target_ids(
                _first(record, "input_candidate_ids", "input_ids", "input_products", "input_documents", default=[])
            ),
            output_candidate_ids=_target_ids(
                _first(record, "output_candidate_ids", "output_ids", "output_products", "output_documents", default=[])
            ),
            source=_first(record, "source"),
            protocol=_first(record, "protocol"),
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "case_id": self.case_id,
            "event_type": self.event_type,
            "run_id": self.run_id,
            "event_id": self.event_id,
            "timestamp": self.timestamp,
            "step": self.step,
            "actor": self.actor,
            "input_candidate_ids": list(self.input_candidate_ids),
            "output_candidate_ids": list(self.output_candidate_ids),
            "source": self.source,
            "protocol": self.protocol,
            "metadata": _json_value(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class BenchmarkDataset:
    """A self-describing collection used by readers and evaluator entry points."""

    benchmark: str
    cases: tuple[BenchmarkCase, ...]
    source: str | None = None
    protocol: str | None = None
    version: str | None = None
    split: str | None = None
    metadata: dict[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        benchmark = _text(self.benchmark, "unspecified") or "unspecified"
        cases = tuple(case if isinstance(case, BenchmarkCase) else BenchmarkCase.from_mapping(case, benchmark=benchmark) for case in self.cases)
        identifiers = [case.case_id for case in cases]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("BenchmarkDataset contains duplicate case_id values.")
        object.__setattr__(self, "benchmark", benchmark)
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "source", _text(self.source) or None)
        object.__setattr__(self, "protocol", _text(self.protocol) or None)
        object.__setattr__(self, "version", _text(self.version) or None)
        object.__setattr__(self, "split", _text(self.split) or None)
        object.__setattr__(self, "metadata", _json_value(self.metadata))

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "benchmark": self.benchmark,
            "cases": [case.to_dict() for case in self.cases],
            "source": self.source,
            "protocol": self.protocol,
            "version": self.version,
            "split": self.split,
            "metadata": _json_value(self.metadata),
        }


__all__ = [
    "BenchmarkCase",
    "BenchmarkDataset",
    "Candidate",
    "CitationSpan",
    "JSONScalar",
    "JSONValue",
    "Prediction",
    "TargetSelection",
    "TraceEvent",
]
