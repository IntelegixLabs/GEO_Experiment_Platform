"""Reusable, dependency-free metrics for GEO benchmark evaluation.

The module deliberately keeps calculated metrics separate from LLM judging.
For example, it implements the objective Word and position-adjusted Word
metrics from GEO-Bench, while subjective ``Overall``/quality scores must be
supplied by a registered, versioned judge.  This prevents a local benchmark
run from silently claiming that it reproduced a proprietary judge model.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from math import exp, sqrt
from random import Random
from statistics import mean, median, stdev
from typing import Any


def safe_divide(numerator: float | int, denominator: float | int) -> float | None:
    """Return a rounded ratio, or ``None`` for an undefined denominator."""

    if not denominator:
        return None
    return round(float(numerator) / float(denominator), 8)


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def mean_or_none(values: Iterable[float | int | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return round(mean(usable), 8) if usable else None


def median_or_none(values: Iterable[float | int | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return round(median(usable), 8) if usable else None


def sample_standard_deviation(values: Iterable[float | int | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    return round(stdev(usable), 8) if len(usable) >= 2 else None


def bootstrap_mean_interval(
    values: Sequence[float | int | None],
    *,
    iterations: int = 2_000,
    confidence: float = 0.95,
    seed: int = 20260719,
) -> dict[str, float | int | None]:
    """Deterministic percentile bootstrap confidence interval for a mean.

    This is intentionally a *query-level* bootstrap.  Callers must first
    compute one value per matched query/product pair rather than pool repeated
    stochastic runs into one pseudo-observation.
    """

    usable = [float(value) for value in values if value is not None]
    if not usable:
        return {"mean": None, "lower": None, "upper": None, "n": 0, "iterations": 0}
    if len(usable) == 1:
        only = round(usable[0], 8)
        return {"mean": only, "lower": only, "upper": only, "n": 1, "iterations": 0}
    total_iterations = max(100, int(iterations))
    generator = Random(seed)
    sample_size = len(usable)
    boot_means = sorted(
        sum(usable[generator.randrange(sample_size)] for _ in range(sample_size)) / sample_size
        for _ in range(total_iterations)
    )
    alpha = max(0.0, min(1.0, 1.0 - float(confidence))) / 2
    lower_index = min(total_iterations - 1, max(0, int(alpha * total_iterations)))
    upper_index = min(total_iterations - 1, max(0, int((1.0 - alpha) * total_iterations) - 1))
    return {
        "mean": round(mean(usable), 8),
        "lower": round(boot_means[lower_index], 8),
        "upper": round(boot_means[upper_index], 8),
        "n": sample_size,
        "iterations": total_iterations,
    }


def rank_of(ranked_ids: Sequence[Any], target_id: Any, *, candidate_count: int | None = None) -> int | None:
    """Return one-indexed rank; use ``candidate_count + 1`` for a miss.

    SAGEO-style stage evaluation explicitly treats a target omitted from a
    finite candidate list as rank ``k + 1``.  ``candidate_count`` therefore
    controls the missing-rank convention rather than assuming a rank of zero.
    """

    if target_id in (None, ""):
        return None
    target = str(target_id)
    for position, item_id in enumerate(ranked_ids, start=1):
        if str(item_id) == target:
            return position
    count = candidate_count if candidate_count is not None else len(ranked_ids)
    return max(0, int(count)) + 1


def reciprocal_rank(rank: int | None) -> float | None:
    return round(1.0 / rank, 8) if rank and rank > 0 else None


def hit_at(rank: int | None, cutoff: int) -> int | None:
    if rank is None:
        return None
    return int(rank <= max(1, int(cutoff)))


def paired_effects(
    baseline_by_case: Mapping[str, float | int | None],
    treatment_by_case: Mapping[str, float | int | None],
    *,
    higher_is_better: bool = True,
    bootstrap_seed: int = 20260719,
) -> dict[str, Any]:
    """Summarise matched baseline/treatment effects without pooling cases."""

    shared = sorted(set(baseline_by_case) & set(treatment_by_case))
    differences: list[float] = []
    absolute_baseline: list[float] = []
    absolute_treatment: list[float] = []
    wins = ties = losses = 0
    binary_pairs: list[tuple[int, int]] = []
    for case_id in shared:
        baseline = as_float(baseline_by_case[case_id])
        treatment = as_float(treatment_by_case[case_id])
        if baseline is None or treatment is None:
            continue
        delta = (treatment - baseline) if higher_is_better else (baseline - treatment)
        differences.append(delta)
        absolute_baseline.append(baseline)
        absolute_treatment.append(treatment)
        if delta > 0:
            wins += 1
        elif delta < 0:
            losses += 1
        else:
            ties += 1
        if baseline in (0.0, 1.0) and treatment in (0.0, 1.0):
            binary_pairs.append((int(baseline), int(treatment)))
    n = len(differences)
    relative_lift = None
    baseline_mean = mean_or_none(absolute_baseline)
    treatment_mean = mean_or_none(absolute_treatment)
    if baseline_mean not in (None, 0.0) and treatment_mean is not None:
        raw_change = treatment_mean - baseline_mean if higher_is_better else baseline_mean - treatment_mean
        relative_lift = round(raw_change / abs(baseline_mean), 8)
    effect_size = None
    deviation = sample_standard_deviation(differences)
    mean_delta = mean_or_none(differences)
    if deviation not in (None, 0.0) and mean_delta is not None:
        effect_size = round(mean_delta / deviation, 8)
    output: dict[str, Any] = {
        "matched_cases": n,
        "baseline_mean": baseline_mean,
        "treatment_mean": treatment_mean,
        "absolute_lift": mean_delta,
        "relative_lift": relative_lift,
        "median_lift": median_or_none(differences),
        "standard_deviation": deviation,
        "cohen_d_paired": effect_size,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": safe_divide(wins, n),
        "win_tie_rate": safe_divide(wins + ties, n),
        "downside_risk": mean_or_none([min(0.0, delta) ** 2 for delta in differences]),
        "bootstrap_95_ci": bootstrap_mean_interval(differences, seed=bootstrap_seed),
    }
    if binary_pairs:
        output["mcnemar_table"] = mcnemar_table(binary_pairs)
    return output


def mcnemar_table(pairs: Iterable[tuple[int, int]]) -> dict[str, int]:
    """Return the paired binary contingency table for a later exact test."""

    table = Counter((int(bool(before)), int(bool(after))) for before, after in pairs)
    return {
        "both_negative": table[(0, 0)],
        "baseline_only": table[(1, 0)],
        "treatment_only": table[(0, 1)],
        "both_positive": table[(1, 1)],
        "discordant": table[(1, 0)] + table[(0, 1)],
    }


def sentence_segments(text: str) -> list[str]:
    """A transparent sentence splitter sufficient for citation-span scoring."""

    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return []
    segments: list[str] = []
    current: list[str] = []
    for character in cleaned:
        current.append(character)
        if character in ".!?":
            candidate = "".join(current).strip()
            if candidate:
                segments.append(candidate)
            current = []
    tail = "".join(current).strip()
    if tail:
        segments.append(tail)
    return segments or [cleaned]


def word_count(text: str) -> int:
    return len([piece for piece in str(text or "").replace("\n", " ").split() if piece])


def _record_value(record: Any, *keys: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        for key in keys:
            if key in record and record[key] is not None:
                return record[key]
        return default
    for key in keys:
        value = getattr(record, key, None)
        if value is not None:
            return value
    # Normalised benchmark records preserve provider-specific columns in
    # metadata.  CitationSpan intentionally has a small typed core, so fields
    # such as ``sentence_index`` and ``attributed_words`` can arrive there.
    metadata = getattr(record, "metadata", None)
    if isinstance(metadata, Mapping):
        for key in keys:
            if key in metadata and metadata[key] is not None:
                return metadata[key]
    return default


def _citation_source_ids(citation: Any) -> list[str]:
    value = _record_value(
        citation,
        "source_ids",
        "candidate_ids",
        "document_ids",
        "product_ids",
        default=None,
    )
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item not in (None, "")]
    # Canonical ``CitationSpan`` objects expose candidate_id.  Prefer it to a
    # citation-row identifier so visibility is attributed to the cited source.
    singular = _record_value(
        citation,
        "candidate_id",
        "source_id",
        "document_id",
        "product_id",
        "citation_id",
        "id",
        default=None,
    )
    return [str(singular)] if singular not in (None, "") else []


def citation_visibility(
    answer: str,
    citations: Iterable[Any],
    source_id: str | None,
) -> dict[str, float | int | bool | None]:
    """Calculate GEO-Bench Word and position-adjusted Word for one source.

    Citation spans must identify sentence indexes or provide attributed words.
    When a sentence cites multiple sources, its credit is divided equally, as
    specified by the original GEO-Bench Word Count definition.
    """

    normalized_source = str(source_id) if source_id not in (None, "") else None
    segments = sentence_segments(answer)
    total_words = sum(word_count(segment) for segment in segments)
    records = list(citations)
    attributed_words = 0.0
    position_adjusted_words = 0.0
    first_position: int | None = None
    cited = False
    sentence_count = max(1, len(segments))
    for fallback_order, citation in enumerate(records):
        source_ids = _citation_source_ids(citation)
        if normalized_source is None or normalized_source not in source_ids:
            continue
        cited = True
        raw_index = _record_value(citation, "sentence_index", "position", "citation_position", default=fallback_order)
        try:
            sentence_index = max(0, int(raw_index))
        except (TypeError, ValueError):
            sentence_index = fallback_order
        if first_position is None or sentence_index < first_position:
            first_position = sentence_index
        explicit_words = as_float(_record_value(citation, "attributed_words", "word_count", default=None))
        sentence_words = explicit_words
        if sentence_words is None:
            sentence_words = float(word_count(segments[sentence_index])) if sentence_index < len(segments) else 0.0
        source_count = max(1, len(source_ids))
        contribution = sentence_words / source_count
        attributed_words += contribution
        position_adjusted_words += contribution * exp(-sentence_index / sentence_count)
    return {
        "cited": cited,
        "citation_rate": int(cited),
        "attributed_words": round(attributed_words, 8),
        "word": round(attributed_words / total_words, 8) if total_words else None,
        "pos": round(position_adjusted_words / total_words, 8) if total_words else None,
        "word_pos": round(position_adjusted_words / total_words, 8) if total_words else None,
        "first_citation_sentence": first_position,
        "answer_words": total_words,
    }


def claim_support_metrics(claims: Iterable[Any]) -> dict[str, float | int | None]:
    """Summarise citation precision, recall and unsupported-claim rate.

    Human or independently-modelled claim labels are inputs; this function does
    not infer factual entailment from superficial lexical overlap.
    """

    total = cited = supported_cited = required = cited_required = unsupported = 0
    for claim in claims:
        total += 1
        is_cited = bool(_record_value(claim, "cited", "has_citation", default=False))
        is_supported = _record_value(claim, "supported", "entailed", default=None)
        needs_citation = bool(_record_value(claim, "requires_citation", "factual", default=True))
        if is_cited:
            cited += 1
        if is_cited and is_supported is True:
            supported_cited += 1
        if needs_citation:
            required += 1
            if is_cited:
                cited_required += 1
        if is_supported is False:
            unsupported += 1
    return {
        "claims": total,
        "citation_precision": safe_divide(supported_cited, cited),
        "citation_recall": safe_divide(cited_required, required),
        "unsupported_claim_rate": safe_divide(unsupported, total),
    }


def jaccard_similarity(left: str, right: str) -> float | None:
    left_tokens = {token.lower() for token in str(left or "").split() if token}
    right_tokens = {token.lower() for token in str(right or "").split() if token}
    if not left_tokens and not right_tokens:
        return None
    return round(len(left_tokens & right_tokens) / len(left_tokens | right_tokens), 8)


def structural_features(content: str) -> dict[str, float | int]:
    """Extract transparent text-structure diagnostics for GEO-SFE/FeatGEO.

    These are descriptive features, not claims that a specific threshold causes
    visibility.  They make treatment/ablation reports auditable.
    """

    lines = [line.strip() for line in str(content or "").splitlines() if line.strip()]
    paragraphs = [chunk.strip() for chunk in str(content or "").split("\n\n") if chunk.strip()]
    words = word_count(content)
    headings = sum(1 for line in lines if line.startswith("#") or line.endswith(":"))
    bullets = sum(1 for line in lines if line.startswith(("- ", "* ", "\u2022 ")))
    numbered = sum(1 for line in lines if len(line) > 2 and line[0].isdigit() and line[1] in ".)")
    sentences = sentence_segments(content)
    return {
        "word_count": words,
        "sentence_count": len(sentences),
        "paragraph_count": len(paragraphs),
        "heading_count": headings,
        "bullet_count": bullets,
        "numbered_list_count": numbered,
        "list_density": round((bullets + numbered) / max(1, len(lines)), 8),
        "heading_density": round(headings / max(1, len(lines)), 8),
        "mean_sentence_words": round(words / max(1, len(sentences)), 8),
    }


def market_concentration(selected_ids: Iterable[str]) -> dict[str, float | int | dict[str, int]]:
    selections = [str(item) for item in selected_ids if item not in (None, "")]
    counts = Counter(selections)
    total = len(selections)
    shares = {item: count / total for item, count in sorted(counts.items())} if total else {}
    hhi = sum(share * share for share in shares.values()) if shares else None
    return {
        "choices": total,
        "unique_selected_products": len(counts),
        "hhi": round(hhi, 8) if hhi is not None else None,
        "selection_counts": dict(sorted(counts.items())),
    }


def standard_error(values: Iterable[float | int | None]) -> float | None:
    usable = [float(value) for value in values if value is not None]
    if len(usable) < 2:
        return None
    return round(stdev(usable) / sqrt(len(usable)), 8)
