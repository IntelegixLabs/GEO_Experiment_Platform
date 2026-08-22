"""Factual, auditable GEO treatment generation and integrity validation.

This module intentionally implements a conservative treatment.  It adds
structure around catalogued facts; it never calls an LLM, writes sales copy, or
tries to influence a model with hidden instructions.  That makes treatment
assignment reproducible and lets the study distinguish visibility from
misleading persuasion.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import re
from typing import Any

from .text import as_list, normalize_whitespace, safe_float, safe_int, stable_json, to_mapping


GEO_TREATMENT_VERSION = "GEO-v2-factual-structure"
LLM_GEO_TREATMENT_VERSION = "GEO-v3-llm-optimized"

BUNDLE_KEYS = frozenset(
    {
        "treatment_version",
        "summary",
        "specifications",
        "claim_blocks",
        "faq",
        "evidence_markers",
        "source_record_hash",
        "feature_vector",
        "content_hash",
    }
)
CLAIM_BLOCK_KEYS = frozenset({"claim_id", "claim", "evidence", "source_fields"})
FAQ_KEYS = frozenset({"question", "answer", "source_fields"})

# Fields which a generated claim is permitted to cite.  A study can add fields
# (e.g. GTIN, warranty, manual URL) by subclassing the builder and extending
# this explicit allow-list.
CANONICAL_FACTUAL_FIELDS = frozenset(
    {
        "id",
        "sku",
        "title",
        "brand",
        "category",
        "price",
        "currency",
        "description",
        "key_features",
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
        "source_timestamp",
        "factual_record_version",
    }
)

PRESERVATION_FIELDS = (
    "id",
    "sku",
    "title",
    "brand",
    "category",
    "price",
    "currency",
    "rating",
    "review_count",
    "availability",
    "shipping",
    "return_policy",
    "model_number",
    "gtin",
    "warranty",
    "source_url",
    "source_timestamp",
)

# Terms can be legitimate in the original record (for example, "limited stock"
# may be supplied).  The validator only errors when a treatment introduces one
# that is not present in the factual source; sourced instances are warnings for
# a human reviewer to assess before recruitment.
UNSUPPORTED_PERSUASION_TERMS = frozenset(
    {
        "best",
        "leading",
        "only",
        "guaranteed",
        "limited",
        "urgent",
        "newest",
        "exclusive",
        "unbeatable",
        "must-have",
        "award-winning",
    }
)

INJECTION_PATTERNS = (
    re.compile(r"\bignore\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions?|rules?)\b", re.I),
    re.compile(r"\b(?:system|developer)\s+message\b", re.I),
    re.compile(r"\b(?:you are|act as)\s+(?:an?\s+)?(?:ai|assistant|chatbot)\b", re.I),
    re.compile(r"\bdo not cite (?:other|competitor)\b", re.I),
    re.compile(r"\b(?:hidden|invisible)\s+(?:text|instruction)\b", re.I),
)


class TreatmentIntegrityError(ValueError):
    """Raised only when a caller asks to enforce a failed integrity report."""


@dataclass(frozen=True)
class IntegrityIssue:
    severity: str
    code: str
    message: str
    path: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "path": self.path,
        }


@dataclass
class IntegrityReport:
    product_id: str | None
    treatment_version: str | None
    errors: list[IntegrityIssue] = field(default_factory=list)
    warnings: list[IntegrityIssue] = field(default_factory=list)
    canonical_hash: str | None = None
    content_hash: str | None = None

    @property
    def valid(self) -> bool:
        return not self.errors

    def add(self, severity: str, code: str, message: str, path: str = "") -> None:
        issue = IntegrityIssue(severity=severity, code=code, message=message, path=path)
        (self.errors if severity == "error" else self.warnings).append(issue)

    def as_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "product_id": self.product_id,
            "treatment_version": self.treatment_version,
            "canonical_hash": self.canonical_hash,
            "content_hash": self.content_hash,
            "errors": [issue.as_dict() for issue in self.errors],
            "warnings": [issue.as_dict() for issue in self.warnings],
        }


def _canonical_value(value: Any) -> Any:
    """Make values hashable/serializable without changing their factual meaning."""

    if isinstance(value, (list, tuple, set)):
        return [normalize_whitespace(item) for item in value if normalize_whitespace(item)]
    if isinstance(value, dict):
        return {str(key): _canonical_value(item) for key, item in sorted(value.items())}
    if isinstance(value, str):
        return normalize_whitespace(value)
    return value


def canonical_snapshot(product: Any) -> dict[str, Any]:
    """Return the stable subset of factual source fields used by treatment code."""

    record = to_mapping(product)
    snapshot = {field: _canonical_value(record.get(field)) for field in sorted(CANONICAL_FACTUAL_FIELDS) if field in record}
    # Feature strings and serialized database lists normalize to the same
    # canonical representation, avoiding accidental version differences.
    snapshot["key_features"] = as_list(record.get("key_features"))
    return snapshot


def canonical_hash(product: Any) -> str:
    return sha256(stable_json(canonical_snapshot(product)).encode("utf-8")).hexdigest()


def _price_text(product: dict[str, Any]) -> str:
    price = safe_float(product.get("price"))
    return f"₹{price:,.2f}" if price is not None else "Not supplied"


def _rating_text(product: dict[str, Any]) -> str:
    rating = safe_float(product.get("rating"))
    reviews = safe_float(product.get("review_count"))
    if rating is None and reviews is None:
        return "Not supplied"
    rating_part = f"{rating:.1f} out of 5" if rating is not None else "Unrated"
    review_part = f" across {int(reviews):,} reviews" if reviews is not None else ""
    return f"{rating_part}{review_part}"


def _clean_field(product: dict[str, Any], key: str) -> str:
    return normalize_whitespace(product.get(key)) or "Not supplied"


class FactualGEOBuilder:
    """Build a structured, factual GEO bundle from study-catalog record fields.

    The returned ``source_fields`` ledger is intentionally machine-readable. It
    allows the integrity gate and later citation analysis to identify what
    factual record supported each rendered evidence block.
    """

    version = GEO_TREATMENT_VERSION

    def build(self, product: Any) -> dict[str, Any]:
        record = to_mapping(product)
        title = _clean_field(record, "title")
        category = _clean_field(record, "category")
        features = as_list(record.get("key_features"))
        feature_text = "; ".join(features) if features else "No product features supplied"
        price_text = _price_text(record)
        rating_text = _rating_text(record)
        actual_price_val = safe_float(record.get("actual_price"))
        discount_price_val = safe_float(record.get("discount_price"))
        actual_price_text = f"₹{actual_price_val:,.2f}" if actual_price_val is not None else "Not supplied"
        discount_price_text = f"₹{discount_price_val:,.2f}" if discount_price_val is not None else "Not supplied"

        specifications = {
            "Product ID": _clean_field(record, "id"),
            "SKU": _clean_field(record, "sku"),
            "Name": title,
            "Category": category,
            "Main Category": _clean_field(record, "main_category"),
            "Sub Category": _clean_field(record, "sub_category"),
            "Brand": _clean_field(record, "brand"),
            "Listed price": price_text,
            "Actual Price": actual_price_text,
            "Discount Price": discount_price_text,
            "Ratings": _clean_field(record, "rating"),
            "Number of Ratings": _clean_field(record, "review_count"),
            "Image": _clean_field(record, "image_url"),
            "Link": _clean_field(record, "source_url"),
            "Availability": _clean_field(record, "availability"),
            "Shipping": _clean_field(record, "shipping"),
            "Returns": _clean_field(record, "return_policy"),
            "Rating evidence": rating_text,
        }
        optional_specs = {
            "Model number": _clean_field(record, "model_number"),
            "GTIN": _clean_field(record, "gtin"),
            "Warranty": _clean_field(record, "warranty"),
        }
        for label, value in optional_specs.items():
            if value != "Not supplied":
                specifications[label] = value

        claims = [
            {
                "claim_id": "features",
                "claim": f"Listed features: {feature_text}.",
                "evidence": "Study-catalog product feature fields.",
                "source_fields": ["key_features"],
            },
            {
                "claim_id": "offer",
                "claim": f"Listed offer: {price_text}; discount: {discount_price_text}; actual price: {actual_price_text}; availability: {_clean_field(record, 'availability')}.",
                "evidence": "Study-catalog price, discount, actual price in INR, and availability fields.",
                "source_fields": ["price", "discount_price", "actual_price", "currency", "availability"],
            },
            {
                "claim_id": "rating",
                "claim": f"Listed rating evidence: {rating_text}.",
                "evidence": "Study-catalog rating and review-count fields.",
                "source_fields": ["rating", "review_count"],
            },
            {
                "claim_id": "metadata",
                "claim": f"Product link: {_clean_field(record, 'source_url')}; Image: {_clean_field(record, 'image_url')}; Main category: {_clean_field(record, 'main_category')}; Sub category: {_clean_field(record, 'sub_category')}.",
                "evidence": "Study-catalog URL, image, and category hierarchy fields.",
                "source_fields": ["source_url", "image_url", "main_category", "sub_category"],
            },
        ]
        faqs = [
            {
                "question": f"What is {title} listed for?",
                "answer": f"It is listed in the {category} category with these supplied features: {feature_text}.",
                "source_fields": ["title", "category", "key_features"],
            },
            {
                "question": "What are the listed delivery and return details?",
                "answer": (
                    f"Availability: {_clean_field(record, 'availability')}. "
                    f"Shipping: {_clean_field(record, 'shipping')}. "
                    f"Returns: {_clean_field(record, 'return_policy')}."
                ),
                "source_fields": ["availability", "shipping", "return_policy"],
            },
        ]
        # Build the content hash before adding it to avoid a self-reference.
        bundle: dict[str, Any] = {
            "treatment_version": self.version,
            "summary": f"{title}: {feature_text}.",
            "specifications": specifications,
            "claim_blocks": claims,
            "faq": faqs,
            "evidence_markers": [
                "structured specifications",
                "fact-linked claim blocks",
                "product FAQ",
                "offer and availability details",
            ],
            "source_record_hash": canonical_hash(record),
            "feature_vector": {
                "factual_summary": 1,
                "structured_specifications": 1,
                "claim_evidence_links": 1,
                "factual_faq": 1,
                "offer_details": 1,
                "agent_readable_provenance": 1,
            },
        }
        bundle["content_hash"] = sha256(stable_json(bundle).encode("utf-8")).hexdigest()
        return bundle


class LLMGEOBuilder(FactualGEOBuilder):
    """Build a truly generative GEO treatment using an LLM to optimize persuasion."""
    
    version = LLM_GEO_TREATMENT_VERSION
    
    def __init__(self, generation_adapter: Any) -> None:
        super().__init__()
        self.client = generation_adapter.client
        self.model = generation_adapter.model
        self.adapter_name = generation_adapter.name

    def build(self, product: Any) -> dict[str, Any]:
        record = to_mapping(product)
        # First build the factual baseline so we have the same structure and specifications
        bundle = super().build(product)
        bundle["treatment_version"] = self.version

        # Use LLM to rewrite the summary and claims to be highly persuasive
        system_prompt = (
            "You are an expert E-Commerce Generative Engine Optimizer. "
            "Your task is to take the provided product details and rewrite them into a highly relevant, fluent, "
            "and authoritative product summary that will rank well in AI search engines. "
            "Apply the following top-performing GEO strategies:\n"
            "1. Fluency Optimization: Improve readability and text flow.\n"
            "2. Statistics Addition: Emphasize quantitative statistics (like ratings, price) over qualitative discussion.\n"
            "3. Authoritative: Use a persuasive and authoritative tone.\n\n"
            "Do NOT use keyword stuffing. \n"
            "Output your response as a valid JSON object matching this schema:\n"
            '{\n'
            '  "summary": "Your fluent, authoritative product summary paragraph",\n'
            '  "claim_blocks": [\n'
            '     {"claim": "Relevant claim incorporating statistics or features", "evidence": "Factual evidence", "source_fields": ["title", "key_features"]},\n'
            '     {"claim": "Relevant claim about offer/rating", "evidence": "Factual evidence", "source_fields": ["price", "rating"]}\n'
            '  ],\n'
            '  "faq": [\n'
            '     {"question": "Highly relevant user question based on product?", "answer": "Clear, authoritative answer.", "source_fields": ["description", "category"]}\n'
            '  ]\n'
            '}'
        )

        product_json = stable_json({
            "title": _clean_field(record, "title"),
            "category": _clean_field(record, "category"),
            "key_features": as_list(record.get("key_features")),
            "description": _clean_field(record, "description"),
            "price": _price_text(record),
            "rating": _rating_text(record)
        })

        try:
            is_gemini = getattr(self, 'adapter_name', '') == 'gemini-adapter'
            import json

            if is_gemini:
                from google.genai import types
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=f"Optimize this product:\n{product_json}",
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        response_mime_type="application/json",
                        temperature=0.7
                    )
                )
                content = response.text
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Optimize this product:\n{product_json}"}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7
                )
                content = response.choices[0].message.content

            if content:
                content = content.strip()
                if content.startswith("```json"):
                    content = content[7:]
                if content.endswith("```"):
                    content = content[:-3]
                content = content.strip()
                parsed = json.loads(content)
                if "summary" in parsed:
                    bundle["summary"] = parsed["summary"]
                if "claim_blocks" in parsed:
                    bundle["claim_blocks"] = parsed["claim_blocks"]
                if "faq" in parsed:
                    bundle["faq"] = parsed["faq"]
                bundle["evidence_markers"] = ["LLM optimized summary", "Generative claims", "Persuasive FAQ"]
        except Exception as e:
            # Do not fallback to factual if LLM fails, let the error bubble up
            raise e

        without_hash = dict(bundle)
        without_hash.pop("content_hash", None)
        bundle["content_hash"] = sha256(stable_json(without_hash).encode("utf-8")).hexdigest()
        return bundle


class GEOIntegrityValidator:
    """Validate provenance, factual preservation, and manipulation safeguards."""

    def __init__(self, builder: FactualGEOBuilder | None = None) -> None:
        self.builder = builder or FactualGEOBuilder()

    def validate(self, product: Any, bundle: Any | None = None) -> IntegrityReport:
        record = to_mapping(product)
        candidate = to_mapping(bundle) if bundle is not None else self.builder.build(record)
        report = IntegrityReport(
            product_id=normalize_whitespace(record.get("id")) or None,
            treatment_version=normalize_whitespace(candidate.get("treatment_version")) or None,
            canonical_hash=canonical_hash(record),
            content_hash=normalize_whitespace(candidate.get("content_hash")) or None,
        )
        if candidate.get("treatment_version") == LLM_GEO_TREATMENT_VERSION:
            # Only check for hallucinations / wrong claims for LLM treatments
            self._validate_provenance(record, candidate, report)
            return report

        self._validate_required_record_fields(record, report)
        self._validate_version_and_hashes(record, candidate, report)
        self._validate_provenance(record, candidate, report)
        self._validate_generated_language(record, candidate, report)
        self._validate_strict_template(record, candidate, report)
        return report

    @staticmethod
    def enforce(product: Any, bundle: Any | None = None) -> dict[str, Any]:
        validator = GEOIntegrityValidator()
        report = validator.validate(product, bundle)
        if not report.valid:
            messages = "; ".join(issue.message for issue in report.errors)
            raise TreatmentIntegrityError(messages)
        return report.as_dict()

    def _validate_required_record_fields(self, record: dict[str, Any], report: IntegrityReport) -> None:
        for field_name in ("title", "category"):
            if not normalize_whitespace(record.get(field_name)):
                report.add("error", "missing_required_fact", f"Canonical product field '{field_name}' is required.", field_name)
        if not normalize_whitespace(record.get("id")):
            report.add("warning", "missing_product_identifier", "Product ID is absent; use a stable ID before a production study.", "id")
        if not normalize_whitespace(record.get("source_url")):
            report.add("warning", "missing_source_url", "No source URL is recorded; add provenance before a production release.", "source_url")
        if not normalize_whitespace(record.get("source_timestamp")):
            report.add("warning", "missing_source_timestamp", "No source timestamp is recorded; record freshness for live offers.", "source_timestamp")

    def _validate_version_and_hashes(self, record: dict[str, Any], bundle: dict[str, Any], report: IntegrityReport) -> None:
        for unexpected_key in sorted(set(bundle) - BUNDLE_KEYS):
            report.add(
                "error",
                "unexpected_bundle_field",
                f"Unexpected GEO bundle field '{unexpected_key}' is not allowed in the preregistered treatment.",
                unexpected_key,
            )
        if bundle.get("treatment_version") != self.builder.version:
            report.add(
                "error",
                "unknown_treatment_version",
                f"Expected treatment version '{self.builder.version}'.",
                "treatment_version",
            )
        expected_source_hash = canonical_hash(record)
        if bundle.get("source_record_hash") != expected_source_hash:
            report.add(
                "error",
                "source_record_hash_mismatch",
                "The bundle does not match the canonical product record used for this release.",
                "source_record_hash",
            )
        without_hash = dict(bundle)
        supplied_hash = without_hash.pop("content_hash", None)
        computed_hash = sha256(stable_json(without_hash).encode("utf-8")).hexdigest()
        if supplied_hash != computed_hash:
            report.add("error", "content_hash_mismatch", "Bundle content hash does not match its visible content.", "content_hash")

    def _validate_provenance(self, record: dict[str, Any], bundle: dict[str, Any], report: IntegrityReport) -> None:
        claims = bundle.get("claim_blocks")
        if not isinstance(claims, list) or not claims:
            report.add("error", "missing_claim_ledger", "GEO treatment needs one or more fact-linked claim blocks.", "claim_blocks")
            claims = []
        faqs = bundle.get("faq")
        if not isinstance(faqs, list) or not faqs:
            report.add("error", "missing_faq_ledger", "GEO treatment needs factual FAQ provenance.", "faq")
            faqs = []
        for section_name, entries in (("claim_blocks", claims), ("faq", faqs)):
            for index, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    report.add("error", "invalid_evidence_entry", "Evidence entry must be an object.", f"{section_name}[{index}]")
                    continue
                allowed_keys = CLAIM_BLOCK_KEYS if section_name == "claim_blocks" else FAQ_KEYS
                for unexpected_key in sorted(set(entry) - allowed_keys):
                    report.add(
                        "error",
                        "unexpected_evidence_field",
                        f"Unexpected field '{unexpected_key}' is not allowed in a factual {section_name} entry.",
                        f"{section_name}[{index}].{unexpected_key}",
                    )
                sources = entry.get("source_fields")
                if not isinstance(sources, list) or not sources:
                    report.add("error", "missing_claim_source", "Each generated claim/FAQ needs source_fields.", f"{section_name}[{index}]")
                    continue
                for source in sources:
                    if source not in CANONICAL_FACTUAL_FIELDS:
                        report.add(
                            "error",
                            "unapproved_source_field",
                            f"'{source}' is not an allowed canonical factual source field.",
                            f"{section_name}[{index}].source_fields",
                        )
                    elif source not in record:
                        report.add(
                            "warning",
                            "missing_claim_source_value",
                            f"Claim cites '{source}', which is not present in this product record.",
                            f"{section_name}[{index}].source_fields",
                        )

    def _validate_generated_language(self, record: dict[str, Any], bundle: dict[str, Any], report: IntegrityReport) -> None:
        if bundle.get("treatment_version") == LLM_GEO_TREATMENT_VERSION:
            return

        source_text = " ".join(
            normalize_whitespace(value)
            for key, value in record.items()
            if key in CANONICAL_FACTUAL_FIELDS and not isinstance(value, (list, tuple, dict))
        ) + " " + " ".join(as_list(record.get("key_features")))
        source_tokens = set(re.findall(r"[a-z0-9-]+", source_text.lower()))
        generated_text = _bundle_text(bundle)
        generated_tokens = set(re.findall(r"[a-z0-9-]+", generated_text.lower()))
        for term in sorted(UNSUPPORTED_PERSUASION_TERMS & generated_tokens):
            if term in source_tokens:
                report.add(
                    "warning",
                    "persuasion_term_in_source",
                    f"'{term}' is sourced from the record but needs human review for condition cues or marketing language.",
                    "generated_content",
                )
            else:
                report.add(
                    "error",
                    "unsupported_persuasion_term",
                    f"Generated GEO content introduced unsupported persuasive term '{term}'.",
                    "generated_content",
                )
        for pattern in INJECTION_PATTERNS:
            if pattern.search(generated_text):
                report.add(
                    "error",
                    "prompt_injection_pattern",
                    "Generated GEO content contains an AI-targeting or prompt-injection pattern.",
                    "generated_content",
                )
                break

    def _validate_strict_template(self, record: dict[str, Any], bundle: dict[str, Any], report: IntegrityReport) -> None:
        """Ensure an altered bundle cannot silently bypass fact-preservation rules.

        This strict comparison is appropriate for the preregistered default
        treatment.  A later experiment with a different preregistered builder
        should use a new builder/version and validate it separately.
        """
        if self.builder.version != FactualGEOBuilder.version:
            return


        expected = self.builder.build(record)
        comparable_keys = (
            "summary",
            "specifications",
            "claim_blocks",
            "faq",
            "evidence_markers",
            "feature_vector",
            "source_record_hash",
        )
        for key in comparable_keys:
            if bundle.get(key) != expected.get(key):
                report.add(
                    "error",
                    "factual_bundle_mismatch",
                    f"'{key}' differs from the deterministic factual treatment generated from the canonical record.",
                    key,
                )


def _bundle_text(bundle: dict[str, Any]) -> str:
    values: list[str] = [normalize_whitespace(bundle.get("summary"))]
    specifications = bundle.get("specifications")
    if isinstance(specifications, dict):
        values.extend(normalize_whitespace(value) for value in specifications.values())
    for block in bundle.get("claim_blocks") or []:
        if isinstance(block, dict):
            values.extend((normalize_whitespace(block.get("claim")), normalize_whitespace(block.get("evidence"))))
    for faq in bundle.get("faq") or []:
        if isinstance(faq, dict):
            values.extend((normalize_whitespace(faq.get("question")), normalize_whitespace(faq.get("answer"))))
    return " ".join(value for value in values if value)


def factual_geo_bundle(product: Any) -> dict[str, Any]:
    """Convenience function retained for simple route/service integration."""

    return FactualGEOBuilder().build(product)


def validate_geo_bundle(product: Any, bundle: Any | None = None) -> dict[str, Any]:
    """Return a JSON-safe integrity report without raising on invalid content."""

    return GEOIntegrityValidator().validate(product, bundle).as_dict()
