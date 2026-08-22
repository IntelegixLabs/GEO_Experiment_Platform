"""Transparent retrieval and reranking for the controlled shopping study.

This is deliberately *not* a simulation of a commercial answer engine.  The
default scorer is deterministic, inspectable, and local.  It reports lexical,
semantic-proxy, and factual-evidence components separately so a researcher can
log and analyse the retrieval/citation pipeline without treating a hidden model
score as an outcome.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import re
from typing import Any, Protocol

from .text import as_list, normalize_whitespace, safe_float, tokenize, to_mapping


# A small explicit concept map supports transparent, repeatable semantic-like
# matching in a no-dependency prototype.  It is not an embedding model and its
# contents should be frozen/versioned for each experimental wave.
DEFAULT_CONCEPTS: dict[str, frozenset[str]] = {
    "audio": frozenset({"earbuds", "earphones", "headphones", "headset", "speaker"}),
    "wireless": frozenset({"bluetooth", "cordless"}),
    "portable": frozenset({"compact", "lightweight", "travel", "commuting", "commute", "foldable"}),
    "bottle": frozenset({"flask", "tumbler", "drinkware"}),
    "insulated": frozenset({"thermal", "double-wall", "doublewall"}),
    "laptop": frozenset({"notebook", "computer"}),
    "water_resistant": frozenset({"waterproof", "splash-resistant", "ipx4", "ipx5", "ipx6"}),
    "exercise": frozenset({"fitness", "workout", "yoga", "stretching", "training"}),
    "storage": frozenset({"container", "containers", "organizer", "organiser"}),
    "cleanser": frozenset({"cleaning", "facewash", "face-wash", "wash"}),
    "shipping": frozenset({"delivery", "dispatch", "arrival"}),
    "returns": frozenset({"return", "refund", "exchange"}),
}


@dataclass(frozen=True)
class SearchIntent:
    query: str
    tokens: tuple[str, ...]
    category_filter: str = ""
    max_price: float | None = None
    min_rating: float | None = None
    comparison: bool = False
    decision: bool = False
    evidence_need: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "tokens": list(self.tokens),
            "category_filter": self.category_filter,
            "max_price": self.max_price,
            "min_rating": self.min_rating,
            "comparison": self.comparison,
            "decision": self.decision,
            "evidence_need": self.evidence_need,
        }


@dataclass(frozen=True)
class SemanticMatch:
    score: float
    matched_concepts: tuple[str, ...] = ()
    matched_terms: tuple[str, ...] = ()


class SemanticSimilarityAdapter(Protocol):
    """Optional interface for a research-approved semantic model adapter.

    An application can inject an adapter based on a locally hosted embedding
    model or an approved external service.  The default remains local and
    deterministic; callers should log adapter name/version when overriding it.
    """

    name: str
    version: str

    def score(self, query_tokens: set[str], document_tokens: set[str]) -> SemanticMatch:
        """Return a bounded 0..1 similarity and transparent diagnostic data."""


class ConceptSemanticSimilarity:
    """Explicit concept-overlap semantic proxy with no network/API dependency."""

    name = "concept-overlap"
    version = "1"

    def __init__(self, concepts: Mapping[str, Iterable[str]] | None = None) -> None:
        raw = concepts or DEFAULT_CONCEPTS
        self.concepts = {
            normalize_whitespace(name).lower(): frozenset(tokenize(" ".join(as_list(values)), include_stop_words=True))
            for name, values in raw.items()
        }

    def score(self, query_tokens: set[str], document_tokens: set[str]) -> SemanticMatch:
        if not query_tokens or not document_tokens:
            return SemanticMatch(0.0)
        matching_concepts: list[str] = []
        matched_terms: set[str] = set()
        for concept, terms in self.concepts.items():
            query_terms = query_tokens & terms
            document_terms = document_tokens & terms
            if query_terms and document_terms:
                matching_concepts.append(concept)
                matched_terms.update(query_terms | document_terms)
        # Normalise by query signal so a long document cannot inflate a score.
        concept_score = len(matching_concepts) / max(1, min(len(query_tokens), 4))
        # Prefix/stem matching handles transparent variants such as "commute"
        # vs "commuting" without pretending to infer a latent meaning.
        stem_hits = {
            query_term
            for query_term in query_tokens
            if len(query_term) >= 5
            and any(document_term.startswith(query_term[:5]) or query_term.startswith(document_term[:5]) for document_term in document_tokens if len(document_term) >= 5)
        }
        stem_score = len(stem_hits) / max(1, len(query_tokens))
        score = min(1.0, (0.75 * concept_score) + (0.25 * stem_score))
        return SemanticMatch(
            score=round(score, 6),
            matched_concepts=tuple(sorted(matching_concepts)),
            matched_terms=tuple(sorted(matched_terms | stem_hits)),
        )


@dataclass(frozen=True)
class RetrievalConfig:
    """Frozen scoring configuration for an experimental wave."""

    title_weight: float = 4.5
    category_weight: float = 3.2
    feature_weight: float = 2.8
    description_weight: float = 1.5
    offer_weight: float = 1.2
    semantic_weight: float = 1.8
    evidence_match_weight: float = 1.4
    max_structural_evidence_bonus: float = 2.14
    category_filter_bonus: float = 7.0
    within_budget_bonus: float = 3.0
    over_budget_penalty: float = 2.0
    min_relevance_score: float = 1.0
    candidate_limit: int = 25
    version: str = "controlled-retriever-v2"

    def as_dict(self) -> dict[str, Any]:
        return {
            "title_weight": self.title_weight,
            "category_weight": self.category_weight,
            "feature_weight": self.feature_weight,
            "description_weight": self.description_weight,
            "offer_weight": self.offer_weight,
            "semantic_weight": self.semantic_weight,
            "evidence_match_weight": self.evidence_match_weight,
            "max_structural_evidence_bonus": self.max_structural_evidence_bonus,
            "category_filter_bonus": self.category_filter_bonus,
            "within_budget_bonus": self.within_budget_bonus,
            "over_budget_penalty": self.over_budget_penalty,
            "min_relevance_score": self.min_relevance_score,
            "candidate_limit": self.candidate_limit,
            "version": self.version,
        }


@dataclass
class RankedCandidate:
    product: dict[str, Any]
    rank_position: int
    retrieval_score: float
    lexical_score: float
    semantic_score: float
    evidence_score: float
    is_relevant: bool
    retrieved: bool
    reasons: list[str] = field(default_factory=list)
    evidence_markers: list[str] = field(default_factory=list)
    matched_terms: dict[str, list[str]] = field(default_factory=dict)
    semantic_concepts: list[str] = field(default_factory=list)

    @property
    def id(self) -> str:
        return str(self.product.get("id") or self.product.get("product_id") or "")

    def as_dict(self, *, include_product: bool = False) -> dict[str, Any]:
        output: dict[str, Any] = {
            "id": self.id,
            "product_id": self.id,
            "title": normalize_whitespace(self.product.get("title")),
            "brand": normalize_whitespace(self.product.get("brand")),
            "category": normalize_whitespace(self.product.get("category")),
            "source_url": normalize_whitespace(self.product.get("source_url")),
            "rank_position": self.rank_position,
            "score": self.retrieval_score,
            "retrieval_score": self.retrieval_score,
            "lexical_score": self.lexical_score,
            "semantic_score": self.semantic_score,
            "evidence_score": self.evidence_score,
            "is_relevant": self.is_relevant,
            "retrieved": self.retrieved,
            "reasons": list(self.reasons),
            "evidence_markers": list(self.evidence_markers),
            "matched_terms": {field: list(values) for field, values in self.matched_terms.items()},
            "semantic_concepts": list(self.semantic_concepts),
        }
        if include_product:
            # Avoid leaking experimental labels to participant-facing callers.
            output["product"] = public_product(self.product)
        return output


def public_product(product: Any) -> dict[str, Any]:
    """Return product facts safe to display to a study participant.

    The actual structured GEO blocks remain visible where they belong on a
    product page, but experimental assignment and pairing fields never leave
    the service through this function.
    """

    record = to_mapping(product)
    # Keep the bundle's displayable content without internal hashes or source
    # bookkeeping.  A route may choose a more narrowly shaped response.
    bundle = to_mapping(record.get("geo_bundle") or record.get("treatment_bundle"))
    # Use an allow-list rather than removing only known internal fields.  This
    # prevents a new database/model field (for example, ``condition`` or a
    # reviewer note) from accidentally unblinding participants.
    visible_fields = (
        "id",
        "sku",
        "title",
        "brand",
        "category",
        "price",
        "currency",
        "description",
        "rating",
        "review_count",
        "availability",
        "shipping",
        "return_policy",
        "source_url",
        "image_url",
        "model_number",
        "gtin",
        "warranty",
    )
    record = {field: record.get(field) for field in visible_fields if field in record}
    if bundle:
        record["product_page"] = {
            "summary": normalize_whitespace(bundle.get("summary")),
            "specifications": to_mapping(bundle.get("specifications")),
            "claim_blocks": [
                {"claim": normalize_whitespace(item.get("claim")), "evidence": normalize_whitespace(item.get("evidence"))}
                for item in bundle.get("claim_blocks", [])
                if isinstance(item, Mapping)
            ],
            "faq": [
                {"question": normalize_whitespace(item.get("question")), "answer": normalize_whitespace(item.get("answer"))}
                for item in bundle.get("faq", [])
                if isinstance(item, Mapping)
            ],
        }
    record["key_features"] = as_list(to_mapping(product).get("key_features"))
    return record


def analyse_query(query: Any, category_filter: Any = None) -> SearchIntent:
    """Extract only explicit shopper constraints; no user profile inference."""

    cleaned_query = normalize_whitespace(query)
    lower = cleaned_query.lower()
    # Deliberately only accepts clear budget wording so product numbers/model
    # numbers do not become accidental price filters.
    price_match = re.search(
        r"\b(?:under|below|less\s+than|up\s+to|max(?:imum)?\s+(?:budget|price)?(?:\s+of)?)\s*(?:[$€£₹]\s*)?(\d+(?:\.\d+)?)\b",
        lower,
    )
    rating_match = re.search(r"\b(?:rated|rating(?:\s+of)?|at\s+least)\s*(\d(?:\.\d+)?)\s*(?:/\s*5|stars?)?\b", lower)
    query_tokens = tuple(tokenize(cleaned_query))
    return SearchIntent(
        query=cleaned_query,
        tokens=query_tokens,
        category_filter=normalize_whitespace(category_filter),
        max_price=float(price_match.group(1)) if price_match else None,
        min_rating=float(rating_match.group(1)) if rating_match else None,
        comparison=any(token in {"compare", "versus", "vs", "alternative", "alternatives", "difference"} for token in query_tokens),
        decision=any(token in {"best", "recommend", "recommendation", "choose", "should", "buy", "right"} for token in query_tokens),
        evidence_need=any(token in {"review", "reviews", "rating", "return", "returns", "shipping", "available", "availability", "price", "warranty"} for token in query_tokens),
    )


class TransparentRetriever:
    """A two-stage lexical/semantic-proxy scorer with an explicit relevance gate."""

    def __init__(
        self,
        config: RetrievalConfig | None = None,
        semantic_adapter: SemanticSimilarityAdapter | None = None,
    ) -> None:
        self.config = config or RetrievalConfig()
        self.semantic_adapter = semantic_adapter or ConceptSemanticSimilarity()

    def rank(
        self,
        products: Iterable[Any],
        query: Any,
        *,
        category_filter: Any = None,
        limit: int | None = None,
    ) -> tuple[SearchIntent, list[RankedCandidate]]:
        intent = analyse_query(query, category_filter)
        records = [to_mapping(product) for product in products]
        if intent.category_filter:
            filtered = [
                product
                for product in records
                if normalize_whitespace(product.get("category")).lower() == intent.category_filter.lower()
            ]
            # A bad category filter should produce no result instead of silently
            # changing the experimental retrieval pool.
            records = filtered
        scored = [self._score_product(record, intent) for record in records]
        scored.sort(
            key=lambda item: (
                -item.retrieval_score,
                normalize_whitespace(item.product.get("title")).lower(),
                item.id,
            )
        )
        candidate_limit = max(1, limit if limit is not None else self.config.candidate_limit)
        ranked: list[RankedCandidate] = []
        for position, candidate in enumerate(scored, start=1):
            candidate.rank_position = position
            candidate.retrieved = candidate.is_relevant and position <= candidate_limit
            ranked.append(candidate)
        return intent, ranked

    def search(
        self,
        products: Iterable[Any],
        query: Any,
        *,
        category_filter: Any = None,
        limit: int | None = None,
        include_product: bool = False,
    ) -> dict[str, Any]:
        intent, ranked = self.rank(products, query, category_filter=category_filter, limit=limit)

        return {
            "intent": intent.as_dict(),
            "retriever": {
                "name": "geo-retriever",
                "version": "1.1",
                "semantic_adapter": "pgvector",
                "semantic_adapter_version": "1.0",
                "config": self.config.as_dict(),
            },
            "results": [candidate.as_dict(include_product=include_product) for candidate in ranked],
            "ranked": [candidate.as_dict(include_product=include_product) for candidate in ranked],
        }

    def _score_product(self, product: dict[str, Any], intent: SearchIntent) -> RankedCandidate:
        query_tokens = set(intent.tokens)
        features = as_list(product.get("key_features"))
        fields = {
            "title": f"{product.get('title', '')} {product.get('brand', '')}",
            "category": product.get("category", ""),
            "features": " ".join(features),
            "description": product.get("description", ""),
            "offer": " ".join(str(product.get(name) or "") for name in ("price", "availability", "shipping", "return_policy", "warranty")),
        }
        field_tokens = {name: set(tokenize(value)) for name, value in fields.items()}
        overlap = {name: sorted(query_tokens & tokens) for name, tokens in field_tokens.items()}
        lexical_score = (
            len(overlap["title"]) * self.config.title_weight
            + len(overlap["category"]) * self.config.category_weight
            + len(overlap["features"]) * self.config.feature_weight
            + len(overlap["description"]) * self.config.description_weight
            + len(overlap["offer"]) * self.config.offer_weight
        )
        category_match = bool(intent.category_filter) and normalize_whitespace(product.get("category")).lower() == intent.category_filter.lower()
        if category_match:
            lexical_score += self.config.category_filter_bonus

        budget_reason: str | None = None
        price = safe_float(product.get("price"))
        if intent.max_price is not None and price is not None:
            if price <= intent.max_price:
                lexical_score += self.config.within_budget_bonus
                budget_reason = f"within stated budget (₹{price:,.2f})"
            else:
                lexical_score -= self.config.over_budget_penalty
                budget_reason = f"above stated budget (₹{price:,.2f})"

        rating = safe_float(product.get("rating"))
        rating_match = intent.min_rating is not None and rating is not None and rating >= intent.min_rating
        if rating_match:
            lexical_score += 1.0

        bundle = to_mapping(product.get("geo_bundle") or product.get("treatment_bundle"))
        evidence_text = _evidence_text(bundle)
        evidence_tokens = set(tokenize(evidence_text))
        evidence_matches = sorted(query_tokens & evidence_tokens)
        document_tokens = set().union(*field_tokens.values(), evidence_tokens)

        precomputed_score = product.get("score")
        if precomputed_score is not None:
            semantic = SemanticMatch(score=float(precomputed_score), matched_concepts=("vector_search_match",))
        else:
            semantic = self.semantic_adapter.score(query_tokens, document_tokens)

        base_relevance_hits = sum(len(items) for items in overlap.values()) + len(evidence_matches) + int(category_match) + int(semantic.score > 0)
        structural_bonus = 0.0
        if bundle and base_relevance_hits:
            # Structure may clarify an already relevant record; it can never
            # manufacture relevance for an unrelated product.
            specifications = to_mapping(bundle.get("specifications"))
            claims = bundle.get("claim_blocks") if isinstance(bundle.get("claim_blocks"), list) else []
            faqs = bundle.get("faq") if isinstance(bundle.get("faq"), list) else []
            structural_bonus = min(
                self.config.max_structural_evidence_bonus,
                (min(len(specifications), 8) * 0.18) + (min(len(claims), 3) * 0.30) + (min(len(faqs), 2) * 0.22),
            ) * min(1.0, base_relevance_hits / 2.0)
        evidence_score = (len(evidence_matches) * self.config.evidence_match_weight) + structural_bonus
        semantic_weighted = semantic.score * self.config.semantic_weight
        total = lexical_score + semantic_weighted + evidence_score

        explicit_signal = bool(sum(len(items) for items in overlap.values()) or evidence_matches or category_match or semantic.score >= 0.20)
        # A price/rating rule only constrains a product after it matches a
        # topical signal; it never serves as the sole basis for recommendation.
        is_relevant = explicit_signal and total >= self.config.min_relevance_score
        reasons = self._reasons(
            product,
            intent,
            overlap,
            semantic,
            evidence_matches,
            category_match,
            budget_reason,
            rating_match,
            bool(bundle),
        )
        return RankedCandidate(
            product=product,
            rank_position=0,
            retrieval_score=round(total, 6),
            lexical_score=round(lexical_score, 6),
            semantic_score=round(semantic_weighted, 6),
            evidence_score=round(evidence_score, 6),
            is_relevant=is_relevant,
            retrieved=False,
            reasons=reasons,
            evidence_markers=[str(value) for value in bundle.get("evidence_markers", [])] if bundle else [],
            matched_terms={name: terms for name, terms in overlap.items() if terms} | ({"evidence": evidence_matches} if evidence_matches else {}),
            semantic_concepts=list(semantic.matched_concepts),
        )

    @staticmethod
    def _reasons(
        product: dict[str, Any],
        intent: SearchIntent,
        overlap: dict[str, list[str]],
        semantic: SemanticMatch,
        evidence_matches: list[str],
        category_match: bool,
        budget_reason: str | None,
        rating_match: bool,
        has_bundle: bool,
    ) -> list[str]:
        reasons: list[str] = []
        if category_match or overlap.get("category"):
            reasons.append(f"category match: {normalize_whitespace(product.get('category'))}")
        if overlap.get("features"):
            matched_features = [
                feature for feature in as_list(product.get("key_features")) if set(tokenize(feature)) & set(overlap["features"])
            ]
            if matched_features:
                reasons.append(f"feature match: {matched_features[0]}")
        if overlap.get("title"):
            reasons.append(f"title/brand term match: {', '.join(overlap['title'][:3])}")
        if semantic.matched_concepts:
            reasons.append(f"transparent semantic concept match: {', '.join(semantic.matched_concepts[:2])}")
        if budget_reason:
            reasons.append(budget_reason)
        if rating_match:
            reasons.append("meets stated rating threshold")
        if intent.evidence_need and has_bundle and evidence_matches:
            reasons.append("matching structured factual evidence is available")
        if not reasons:
            reasons.append("no material catalog evidence matched the query")
        return reasons


def _evidence_text(bundle: dict[str, Any]) -> str:
    chunks = [normalize_whitespace(bundle.get("summary"))]
    specs = to_mapping(bundle.get("specifications"))
    chunks.extend(normalize_whitespace(value) for value in specs.values())
    for claim in bundle.get("claim_blocks", []) if isinstance(bundle.get("claim_blocks"), list) else []:
        if isinstance(claim, Mapping):
            chunks.append(normalize_whitespace(claim.get("claim")))
    for faq in bundle.get("faq", []) if isinstance(bundle.get("faq"), list) else []:
        if isinstance(faq, Mapping):
            chunks.extend((normalize_whitespace(faq.get("question")), normalize_whitespace(faq.get("answer"))))
    return " ".join(chunk for chunk in chunks if chunk)


def search_catalog(
    products: Iterable[Any],
    query: Any,
    *,
    category_filter: Any = None,
    limit: int | None = None,
) -> dict[str, Any]:
    """Convenience API returning JSON-safe transparent search results."""

    return TransparentRetriever().search(products, query, category_filter=category_filter, limit=limit)
