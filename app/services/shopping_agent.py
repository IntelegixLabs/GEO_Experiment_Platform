"""Grounded shopping research agent for the controlled GEO experiment.

The agent plans a retrieval step, selects only relevance-gated catalog records,
and generates a factual answer with explicit product citations.  The default is
deterministic template generation; an external LLM is never contacted unless a
researcher explicitly supplies and enables a reviewed adapter.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from .retrieval import RankedCandidate, SearchIntent, TransparentRetriever, public_product
from .text import as_list, normalize_whitespace, safe_float, tokenize


@dataclass(frozen=True)
class GroundedClaim:
    """A response claim with the catalog fields that support it."""

    text: str
    citation_id: str
    source_fields: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "citation_id": self.citation_id,
            "source_fields": list(self.source_fields),
        }


@dataclass(frozen=True)
class Citation:
    citation_id: str
    product_id: str
    rank_position: int
    retrieval_score: float
    title: str
    source_url: str
    reasons: tuple[str, ...]
    evidence: tuple[dict[str, Any], ...]
    product: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "product_id": self.product_id,
            "rank_position": self.rank_position,
            "recommendation_score": self.retrieval_score,
            "title": self.title,
            "source_url": self.source_url,
            "reasons": list(self.reasons),
            "evidence": [dict(item) for item in self.evidence],
            "product": dict(self.product),
        }


@dataclass(frozen=True)
class GroundedGenerationContext:
    """Immutable, citation-limited context for an optional response adapter."""

    query: str
    intent: dict[str, Any]
    citations: tuple[Citation, ...]
    allowed_claims: tuple[GroundedClaim, ...]


@dataclass(frozen=True)
class AdapterGeneration:
    """A response from an optional reviewed adapter, auditable before release."""

    answer: str
    claims: tuple[GroundedClaim, ...]
    suggested_follow_ups: tuple[str, ...] = ()


class GroundedGenerationAdapter(Protocol):
    """Interface for an opt-in, institutionally reviewed LLM/agent adapter."""

    name: str
    version: str

    def generate(self, context: GroundedGenerationContext) -> AdapterGeneration:
        """Generate only from supplied citations and source-field claims."""


class CitationGroundingError(ValueError):
    """Raised when an optional adapter returns ungrounded citation metadata."""


class ShoppingResearchAgent:
    """Tool-using but deliberately narrow shopping-research agent.

    Its planning sequence is fixed: understand explicit constraints -> retrieve
    and rerank -> gate relevance -> cite source facts -> generate a qualified
    answer.  It does not browse, purchase, infer personal traits, or claim that
    a product is objectively best.
    """

    name = "grounded-shopping-research-agent"
    version = "2"

    def __init__(
        self,
        retriever: TransparentRetriever | None = None,
        generation_adapter: GroundedGenerationAdapter | None = None,
        *,
        max_citations: int = 3,
        allow_external_generation: bool = False,
    ) -> None:
        self.retriever = retriever or TransparentRetriever()
        self.generation_adapter = generation_adapter
        self.max_citations = max(1, min(int(max_citations), 5))
        self.allow_external_generation = bool(allow_external_generation)

    def answer(
        self,
        products: Iterable[Any],
        query: Any,
        *,
        category_filter: Any = None,
        include_candidates: bool = True,
    ) -> dict[str, Any]:
        intent, ranked = self.retriever.rank(products, query, category_filter=category_filter)
        citations = self._select_citations(ranked)
        allowed_claims = tuple(claim for citation in citations for claim in self._claims_for_citation(citation, intent))
        context = GroundedGenerationContext(
            query=intent.query,
            intent=intent.as_dict(),
            citations=tuple(citations),
            allowed_claims=allowed_claims,
        )
        answer, response_claims, generation_info = self._generate(context, intent)
        output: dict[str, Any] = {
            "answer": answer,
            "citations": [citation.as_dict() for citation in citations],
            "cited_ids": [citation.product_id for citation in citations],
            "intent": intent.as_dict(),
            "agent": {
                "name": self.name,
                "version": self.version,
                "generation": generation_info,
                "citation_policy": {
                    "max_citations": self.max_citations,
                    "requires_relevance_gate": True,
                    "external_generation_enabled": bool(self.generation_adapter and self.allow_external_generation),
                },
            },
            "grounded_claims": [claim.as_dict() for claim in response_claims],
            "suggestions": generation_info.get("suggested_follow_ups", []),
        }
        if include_candidates:
            output["candidates"] = [candidate.as_dict(include_product=False) for candidate in ranked]
        return output

    def _select_citations(self, ranked: list[RankedCandidate]) -> list[Citation]:
        relevant = [candidate for candidate in ranked if candidate.retrieved and candidate.is_relevant]
        if not relevant:
            return []
        # A relative cut-off avoids citing every weak tail record merely because
        # it happens to be present in a small test catalog.  Candidate rows are
        # still returned/loggable, including non-citations.
        top_score = relevant[0].retrieval_score
        cutoff = max(self.retriever.config.min_relevance_score, top_score * 0.40)
        selected = [candidate for candidate in relevant if candidate.retrieval_score >= cutoff][: self.max_citations]
        return [self._make_citation(candidate, index + 1) for index, candidate in enumerate(selected)]

    def _make_citation(self, candidate: RankedCandidate, citation_number: int) -> Citation:
        product = candidate.product
        evidence = tuple(self._citation_evidence(candidate))
        return Citation(
            citation_id=f"C{citation_number}",
            product_id=candidate.id,
            rank_position=candidate.rank_position,
            retrieval_score=candidate.retrieval_score,
            title=normalize_whitespace(product.get("title")) or "Unnamed catalog product",
            source_url=normalize_whitespace(product.get("source_url")),
            reasons=tuple(candidate.reasons),
            evidence=evidence,
            product=public_product(product),
        )

    @staticmethod
    def _citation_evidence(candidate: RankedCandidate) -> list[dict[str, Any]]:
        """Expose only exact supplied facts that can support a citation card."""

        product = candidate.product
        fields: list[tuple[str, str, Any]] = [
            ("title", "Product", product.get("title")),
            ("brand", "Brand", product.get("brand")),
            ("category", "Category", product.get("category")),
        ]
        for feature in as_list(product.get("key_features")):
            feature_tokens = set(tokenize(feature))
            matched = set().union(*(set(values) for values in candidate.matched_terms.values())) if candidate.matched_terms else set()
            if feature_tokens & matched:
                fields.append(("key_features", "Feature", feature))
        if safe_float(product.get("price")) is not None:
            price_val = safe_float(product.get("price"))
            fields.append(("price", "Listed price", f"₹{price_val:,.2f}"))
        for field_name, label in (("availability", "Availability"), ("shipping", "Shipping"), ("return_policy", "Returns")):
            value = normalize_whitespace(product.get(field_name))
            if value:
                fields.append((field_name, label, value))
        rating = safe_float(product.get("rating"))
        if rating is not None:
            fields.append(("rating", "Listed rating", f"{rating:.1f}/5"))
        seen: set[tuple[str, str]] = set()
        evidence: list[dict[str, Any]] = []
        for source_field, label, value in fields:
            clean_value = normalize_whitespace(value)
            key = (source_field, clean_value)
            if clean_value and key not in seen:
                evidence.append({"source_field": source_field, "label": label, "value": clean_value})
                seen.add(key)
        return evidence

    def _claims_for_citation(self, citation: Citation, intent: SearchIntent) -> list[GroundedClaim]:
        claims: list[GroundedClaim] = []
        product = citation.product
        title = citation.title
        matched_features = [
            evidence["value"]
            for evidence in citation.evidence
            if evidence["source_field"] == "key_features"
        ]
        if matched_features:
            claims.append(
                GroundedClaim(
                    text=f"{title} lists {matched_features[0]}.",
                    citation_id=citation.citation_id,
                    source_fields=("title", "key_features"),
                )
            )
        if intent.max_price is not None and safe_float(product.get("price")) is not None:
            price = safe_float(product.get("price"))
            qualifier = "is within" if price is not None and price <= intent.max_price else "is above"
            claims.append(
                GroundedClaim(
                    text=f"{title} has a listed price of ₹{price:,.2f}, which {qualifier} the stated budget.",
                    citation_id=citation.citation_id,
                    source_fields=("title", "price", "currency"),
                )
            )
        if intent.evidence_need:
            detail = normalize_whitespace(product.get("return_policy")) or normalize_whitespace(product.get("shipping"))
            if detail:
                source = "return_policy" if normalize_whitespace(product.get("return_policy")) else "shipping"
                claims.append(
                    GroundedClaim(
                        text=f"{title} lists: {detail}.",
                        citation_id=citation.citation_id,
                        source_fields=("title", source),
                    )
                )
        if not claims:
            category = normalize_whitespace(product.get("category"))
            claims.append(
                GroundedClaim(
                    text=f"{title} is listed in the {category or 'general'} category.",
                    citation_id=citation.citation_id,
                    source_fields=("title", "category"),
                )
            )
        return claims

    def _generate(
        self,
        context: GroundedGenerationContext,
        intent: SearchIntent,
    ) -> tuple[str, list[GroundedClaim], dict[str, Any]]:
        if not context.citations:
            return (
                "I could not find a catalog item with enough matching factual evidence for that request. "
                "Try a more specific product type, feature, category, or budget.",
                [],
                {"mode": "deterministic", "adapter": None, "fallback_used": False},
            )
        if self.generation_adapter and self.allow_external_generation:
            try:
                generation = self.generation_adapter.generate(context)
                self._validate_adapter_generation(generation, context)
                return (
                    normalize_whitespace(generation.answer),
                    list(generation.claims),
                    {
                        "mode": "approved_adapter",
                        "adapter": getattr(self.generation_adapter, "name", type(self.generation_adapter).__name__),
                        "adapter_version": getattr(self.generation_adapter, "version", "unknown"),
                        "fallback_used": False,
                        "suggested_follow_ups": list(generation.suggested_follow_ups),
                    },
                )
            except Exception as ex:
                # Log the error but silently fall back to the deterministic baseline
                import traceback, sys
                traceback.print_exc(file=sys.stderr)
                answer, claims = self._deterministic_generation(context, intent)
                return (
                    answer,
                    claims,
                    {
                        "mode": "deterministic",
                        "adapter": getattr(self.generation_adapter, "name", type(self.generation_adapter).__name__),
                        "fallback_used": True,
                        "fallback_reason": type(ex).__name__,
                        "error": str(ex),
                    },
                )
        answer, claims = self._deterministic_generation(context, intent)
        return answer, claims, {"mode": "deterministic", "adapter": None, "fallback_used": False}

    @staticmethod
    def _validate_adapter_generation(generation: AdapterGeneration, context: GroundedGenerationContext) -> None:
        if not isinstance(generation, AdapterGeneration):
            raise CitationGroundingError("Adapter must return AdapterGeneration.")
        if not normalize_whitespace(generation.answer):
            raise CitationGroundingError("Adapter returned an empty answer.")
        allowed = {(claim.citation_id, claim.text, claim.source_fields) for claim in context.allowed_claims}
        for claim in generation.claims:
            if (claim.citation_id, claim.text, claim.source_fields) not in allowed:
                raise CitationGroundingError("Adapter introduced a claim outside the approved grounded claim set.")

    @staticmethod
    def _deterministic_generation(context: GroundedGenerationContext, intent: SearchIntent) -> tuple[str, list[GroundedClaim]]:
        citations = list(context.citations)
        claims = list(context.allowed_claims)
        first = citations[0]
        lead = (
            f"I found {len(citations)} catalog item{'s' if len(citations) != 1 else ''} with matching factual evidence. "
            f"The closest match in this controlled catalog is {first.title} [{first.citation_id}]."
        )
        # Keep answer claims bounded to the factual ledger; present one claim per
        # citation so a short answer does not bury the participant in details.
        selected_claims: list[GroundedClaim] = []
        seen_citations: set[str] = set()
        for claim in claims:
            if claim.citation_id not in seen_citations:
                selected_claims.append(claim)
                seen_citations.add(claim.citation_id)
        body = " ".join(f"{claim.text} [{claim.citation_id}]" for claim in selected_claims)
        comparison = " You can open the cited items to compare the listed facts." if len(citations) > 1 or intent.comparison else ""
        caution = " Verify current price, availability, compatibility, and suitability before making a real purchase."
        return f"{lead} {body}{comparison}{caution}".strip(), selected_claims


def answer_shopping_query(
    products: Iterable[Any],
    query: Any,
    *,
    category_filter: Any = None,
) -> dict[str, Any]:
    """Convenience API for a grounded deterministic shopping answer."""

    return ShoppingResearchAgent().answer(products, query, category_filter=category_filter)
