"""Deterministic research logic shared by FastAPI routes.

This fallback is intentionally transparent and conservative.  It gives the
platform a runnable controlled retrieval condition without claiming to model a
commercial answer engine.  If the optional GEO agent service is installed, it
is used opportunistically and its output is normalised into the same audit
record schema.
"""

from __future__ import annotations

import inspect
import re
from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Callable, Iterable

from app.core.config import CONDITIONS


TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.-][a-z0-9]+)?", re.IGNORECASE)
SCALE_MAP = {
    "recommendation_quality": ["rq1", "rq2", "rq3", "rq4"],
    "relevance_accuracy": ["ra1", "ra2", "ra3"],
    "usefulness": ["pu1", "pu2", "pu3", "pu4"],
    "trust": ["tr1", "tr2", "tr3", "tr4"],
    "decision_satisfaction": ["ds1", "ds2", "ds3"],
    "agent_satisfaction": ["rs1", "rs2", "rs3"],
    "purchase_intention": ["pi1", "pi2", "pi3"],
    "reuse_intention": ["ri1", "ri2", "ri3"],
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return as_list(list(value))
    if isinstance(value, str):
        return [piece.strip() for piece in re.split(r"[|;,]", value) if piece.strip()]
    return [str(value).strip()]


def tokenize(value: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(value or "") if len(token) > 1]


def trim_text(value: Any, max_length: int) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()[:max_length]


def factual_geo_bundle(product: dict[str, Any]) -> dict[str, Any]:
    """Build a factual and auditable GEO treatment from canonical fields only."""

    features = as_list(product.get("key_features"))
    title = str(product.get("title") or "This product")
    currency = str(product.get("currency") or "INR")
    price = product.get("price")
    price_text = f"₹{float(price):,.2f}" if price not in (None, "") else "See current offer"
    feature_sentence = ", ".join(features) if features else "the listed product features"
    rating = product.get("rating")
    review_count = product.get("review_count")
    rating_text = (
        f"Rated {float(rating):.1f}/5 from {int(float(review_count or 0)):,} listed reviews"
        if rating not in (None, "")
        else "No rating summary supplied"
    )
    return {
        "treatment_version": "GEO-v1-factual-structure",
        "summary": f"{title}: {feature_sentence}.",
        "specifications": {
            "Category": product.get("category") or "Not supplied",
            "Brand": product.get("brand") or "Not supplied",
            "Listed price": price_text,
            "Availability": product.get("availability") or "Not supplied",
            "Shipping": product.get("shipping") or "Not supplied",
            "Returns": product.get("return_policy") or "Not supplied",
            "Rating evidence": rating_text,
        },
        "claim_blocks": [
            {
                "claim": f"Listed features: {feature_sentence}.",
                "evidence": "Product feature fields supplied to the study catalog.",
            },
            {
                "claim": f"Offer information: {price_text}; {product.get('availability') or 'availability not supplied'}.",
                "evidence": "Product offer fields supplied to the study catalog.",
            },
            {
                "claim": rating_text + ".",
                "evidence": "Rating and review-count fields supplied to the study catalog.",
            },
        ],
        "faq": [
            {
                "question": f"What is {title} intended to support?",
                "answer": f"It is listed in {product.get('category') or 'this'} category with {feature_sentence}.",
            },
            {
                "question": "What are the delivery and return details?",
                "answer": (
                    f"Availability: {product.get('availability') or 'not supplied'}. "
                    f"Shipping: {product.get('shipping') or 'not supplied'}. "
                    f"Returns: {product.get('return_policy') or 'not supplied'}."
                ),
            },
        ],
        "evidence_markers": [
            "structured specifications",
            "fact-linked claim blocks",
            "product FAQ",
            "offer and availability details",
        ],
    }


def _call_optional(function: Callable[..., Any], **kwargs: Any) -> Any:
    """Call an agent method with only the keyword arguments it accepts."""

    try:
        signature = inspect.signature(function)
    except (TypeError, ValueError):
        return function(**kwargs)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return function(**kwargs)
    allowed = {name: value for name, value in kwargs.items() if name in signature.parameters}
    return function(**allowed)


@lru_cache(maxsize=1)
def _geo_service() -> Any | None:
    """Load optional ML/agent support without making it a runtime dependency."""

    try:
        from app.services.geo_service import GEOService  # type: ignore[import-not-found]
        from app.core.config import get_settings

        settings = get_settings()
        generation_adapter = None
        if settings.openai_api_key:
            from app.services.openai_adapter import OpenAIGenerationAdapter
            generation_adapter = OpenAIGenerationAdapter(
                model_name=settings.openai_model_name,
                api_key=settings.openai_api_key,
            )

        return GEOService(
            generation_adapter=generation_adapter,
            allow_external_generation=bool(generation_adapter)
        )
    except (ImportError, AttributeError, TypeError):
        return None


def build_geo_bundle(product: dict[str, Any]) -> dict[str, Any]:
    service = _geo_service()
    method = getattr(service, "build_treatment", None) if service else None
    if callable(method):
        try:
            result = _call_optional(method, product=product, canonical_product=product)
            if isinstance(result, dict):
                return result.get("geo_bundle", result.get("bundle", result))
        except (TypeError, ValueError, KeyError):
            pass
    return factual_geo_bundle(product)


def choose_condition(index: int, group_key: str) -> str:
    """Stable 1:1 assignment without exposing the assignment to participants."""

    parity = sum(ord(char) for char in group_key) % 2
    return CONDITIONS[(index + parity) % 2]


def assign_conditions(products: list[dict[str, Any]]) -> None:
    """Assign unlabelled products deterministically by pair, then category."""

    service = _geo_service()
    method = getattr(service, "assign_conditions", None) if service else None
    if callable(method):
        try:
            result = _call_optional(method, products=products, catalog=products)
            if isinstance(result, dict) and isinstance(result.get("products"), list):
                products[:] = result["products"]
            if isinstance(result, list):
                products[:] = result
            if all(product.get("condition") in CONDITIONS for product in products):
                return
        except (TypeError, ValueError, KeyError):
            pass

    paired: dict[str, list[dict[str, Any]]] = defaultdict(list)
    remainder: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for product in products:
        if product.get("condition") in CONDITIONS:
            continue
        if product.get("pair_id"):
            paired[str(product["pair_id"])].append(product)
        else:
            remainder[str(product.get("category") or "Uncategorised")].append(product)
    for group_key, group in [*paired.items(), *remainder.items()]:
        for index, product in enumerate(sorted(group, key=lambda item: (str(item.get("title", "")).lower(), str(item.get("id", ""))))):
            product["condition"] = choose_condition(index, group_key)


def query_intent(query: str, category_filter: str | None = None) -> dict[str, Any]:
    tokens = tokenize(query)
    lower_query = query.lower()
    price_match = re.search(r"(?:under|below|less than|budget of?)\s*\$?\s*(\d+(?:\.\d+)?)", lower_query)
    return {
        "tokens": tokens,
        "category_filter": category_filter or "",
        "max_price": float(price_match.group(1)) if price_match else None,
        "comparison": any(word in tokens for word in ("compare", "versus", "vs", "alternative", "alternatives")),
        "decision": any(word in tokens for word in ("best", "recommend", "choose", "should", "buy", "right")),
        "evidence_need": any(word in tokens for word in ("review", "reviews", "rating", "return", "shipping", "available", "price")),
    }


def _fallback_score_product(product: dict[str, Any], intent: dict[str, Any]) -> dict[str, Any]:
    tokens = set(intent["tokens"])
    features = as_list(product.get("key_features"))
    base_fields = {
        "title": f"{product.get('title', '')} {product.get('brand', '')}",
        "category": str(product.get("category", "")),
        "description": str(product.get("description", "")),
        "features": " ".join(features),
        "offer": " ".join(str(product.get(key) or "") for key in ("availability", "shipping", "return_policy", "price")),
    }
    field_tokens = {name: set(tokenize(value)) for name, value in base_fields.items()}
    overlap = {name: len(tokens & value) for name, value in field_tokens.items()}
    lexical = (
        overlap["title"] * 4.5
        + overlap["category"] * 3.2
        + overlap["features"] * 2.8
        + overlap["description"] * 1.5
        + overlap["offer"] * 1.2
    )
    category_filter_match = bool(
        intent["category_filter"]
        and intent["category_filter"].lower() == str(product.get("category", "")).lower()
    )
    if category_filter_match:
        lexical += 7.0
    if intent["max_price"] is not None and product.get("price") not in (None, ""):
        lexical += 3.0 if float(product["price"]) <= intent["max_price"] else -2.0

    bundle = product.get("geo_bundle") or {}
    evidence_text = " ".join(
        [
            str(bundle.get("summary", "")),
            " ".join(str(value) for value in bundle.get("specifications", {}).values()),
            " ".join(str(block.get("claim", "")) for block in bundle.get("claim_blocks", [])),
            " ".join(f"{faq.get('question', '')} {faq.get('answer', '')}" for faq in bundle.get("faq", [])),
        ]
    )
    evidence_matches = len(tokens & set(tokenize(evidence_text)))
    complete_evidence = 0.0
    relevance_hits = sum(overlap.values()) + evidence_matches + int(category_filter_match)
    if bundle and relevance_hits:
        complete_evidence += min(len(bundle.get("specifications", {})), 7) * 0.18
        complete_evidence += min(len(bundle.get("claim_blocks", [])), 3) * 0.30
        complete_evidence += min(len(bundle.get("faq", [])), 2) * 0.22
        complete_evidence *= min(1.0, relevance_hits / 2)
    evidence_score = evidence_matches * 1.4 + complete_evidence
    reasons: list[str] = []
    if overlap["category"] or intent["category_filter"]:
        reasons.append(f"category: {product.get('category')}")
    if overlap["features"]:
        matching = [feature for feature in features if tokens & set(tokenize(feature))]
        if matching:
            reasons.append("feature match: " + matching[0])
    if intent["max_price"] is not None and product.get("price") not in (None, "") and float(product["price"]) <= intent["max_price"]:
        reasons.append(f"within stated budget ({product.get('currency', 'USD')} {float(product['price']):,.2f})")
    if intent["evidence_need"] and bundle:
        reasons.append("contains structured offer and evidence details")
    if not reasons:
        reasons.append("best available catalog term match")
    return {
        "score": round(lexical + evidence_score, 3),
        "retrieval_score": round(lexical + evidence_score, 3),
        "lexical_score": round(lexical, 3),
        "evidence_score": round(evidence_score, 3),
        "reasons": reasons,
        "evidence_markers": bundle.get("evidence_markers", []),
    }


def _normalize_ranked(raw_ranked: Any, catalog: list[dict[str, Any]], intent: dict[str, Any]) -> list[dict[str, Any]] | None:
    if isinstance(raw_ranked, dict):
        raw_ranked = raw_ranked.get("ranked_products") or raw_ranked.get("ranked") or raw_ranked.get("candidates")
    if not isinstance(raw_ranked, Iterable) or isinstance(raw_ranked, (str, bytes, dict)):
        return None
    catalog_by_id = {str(product["id"]): product for product in catalog}
    ranked: list[dict[str, Any]] = []
    for raw in raw_ranked:
        candidate = raw.model_dump() if hasattr(raw, "model_dump") else raw
        if not isinstance(candidate, dict):
            return None
        product_id = str(candidate.get("id") or candidate.get("product_id") or "")
        base = catalog_by_id.get(product_id)
        if not base:
            continue
        fallback = _fallback_score_product(base, intent)
        score = candidate.get("score", candidate.get("retrieval_score", fallback["score"]))
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = fallback["score"]
        ranked.append(
            {
                **base,
                "score": round(score, 3),
                "retrieval_score": round(float(candidate.get("retrieval_score", score)), 3),
                "lexical_score": round(float(candidate.get("lexical_score", fallback["lexical_score"])), 3),
                "evidence_score": round(float(candidate.get("evidence_score", fallback["evidence_score"])), 3),
                "reasons": candidate.get("reasons") or fallback["reasons"],
                "evidence_markers": candidate.get("evidence_markers") or fallback["evidence_markers"],
            }
        )
    if not ranked:
        return None
    ranked.sort(key=lambda product: (-product["score"], product["title"].lower(), product["id"]))
    return ranked


def rank_catalog(
    catalog: list[dict[str, Any]],
    query: str,
    category_filter: str | None,
    intent: dict[str, Any],
    service: Any | None = None,
) -> list[dict[str, Any]]:
    """Rank a complete catalog and retain every candidate for the denominator."""

    filtered = [
        product for product in catalog
        if not category_filter or str(product.get("category", "")).lower() == category_filter.lower()
    ]
    service = service or _geo_service()
    method = getattr(service, "search_catalog", None) if service else None
    if callable(method):
        try:
            service_result = _call_optional(
                method,
                products=filtered,
                catalog=filtered,
                query=query,
                category_filter=category_filter or "",
                intent=intent,
            )
            normalised = _normalize_ranked(service_result, filtered, intent)
            if normalised is not None:
                return normalised
        except (TypeError, ValueError, KeyError, AttributeError):
            pass

    ranked = [{**product, **_fallback_score_product(product, intent)} for product in filtered]
    ranked.sort(key=lambda product: (-product["score"], product["title"].lower(), product["id"]))
    return ranked


def make_answer(
    query: str,
    ranked: list[dict[str, Any]],
    intent: dict[str, Any],
    service: Any | None = None,
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Generate a factual response, a bounded citation list, and suggested follow-ups."""

    if not ranked:
        return "I could not find a matching product in this study catalog.", [], []

    service = service or _geo_service()
    method = getattr(service, "answer_shopping_query", None) if service else None
    if callable(method):
        try:
            result = _call_optional(
                method,
                products=ranked,
                catalog=ranked,
                query=query,
                category_filter=intent.get("category_filter", ""),
                ranked_products=ranked,
                ranked=ranked,
                intent=intent,
            )
            result = result.model_dump() if hasattr(result, "model_dump") else result
            if isinstance(result, dict):
                answer = str(result.get("answer") or result.get("response") or "").strip()
                raw_cited = result.get("cited_products") or result.get("citations") or result.get("cited_ids") or []
                cited_ids = {
                    str(item.get("id") or item.get("product_id")) if isinstance(item, dict) else str(item)
                    for item in raw_cited
                }
                cited = [product for product in ranked if product["id"] in cited_ids][:3]
                if answer and cited:
                    suggestions = result.get("suggestions", [])
                    return answer, cited, suggestions
        except (TypeError, ValueError, KeyError, AttributeError):
            pass

    relevance_cutoff = max(2.0, ranked[0]["score"] * 0.40)
    cited = [product for product in ranked if product["score"] >= relevance_cutoff][:3] or [ranked[0]]
    primary = cited[0]
    names = ", ".join(item["title"] for item in cited)
    qualifier = " for comparison" if intent["comparison"] else ""
    answer = (
        f"I found {len(cited)} catalog item{'s' if len(cited) != 1 else ''}{qualifier}. "
        f"The strongest evidence match is {primary['title']}. I also referenced {names}. "
        "The linked references show the product information used for this response; please verify current price, "
        "availability, and suitability before making a real purchase."
    )
    return answer, cited, []


def scale_scores(answers: dict[str, Any]) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for construct, fields in SCALE_MAP.items():
        values: list[float] = []
        for field in fields:
            try:
                value = float(answers.get(field))
            except (TypeError, ValueError):
                continue
            if 1 <= value <= 5:
                values.append(value)
        scores[construct] = round(sum(values) / len(values), 3) if values else None
    return scores
