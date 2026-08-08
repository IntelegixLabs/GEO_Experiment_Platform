"""Declarative registry for GEO, e-commerce, and agentic benchmark protocols.

The registry is intentionally metadata-only.  It neither downloads benchmark
data nor represents restricted/copyrighted releases as bundled with this
application.  Researchers supply an authorised export through the IO layer,
then record the exact source, version, and protocol in the run manifest.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Mapping


def _canonicalise(value: str) -> str:
    """Make a permissive lookup key without changing a stored canonical id."""

    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


@dataclass(frozen=True, slots=True)
class BenchmarkSpec:
    """Protocol metadata used to label a benchmark run, not a data loader."""

    benchmark_id: str
    display_name: str
    family: str
    primary_task: str
    default_source: str
    default_protocol: str
    record_kinds: tuple[str, ...] = ("cases", "predictions")
    aliases: tuple[str, ...] = ()
    citation: str | None = None
    data_access: str = "bring_your_own_authorised_export"
    notes: str = ""
    required_case_fields: tuple[str, ...] = ("case_id", "query")
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        benchmark_id = _canonicalise(self.benchmark_id)
        if not benchmark_id:
            raise ValueError("BenchmarkSpec.benchmark_id is required.")
        if not self.display_name.strip():
            raise ValueError("BenchmarkSpec.display_name is required.")
        object.__setattr__(self, "benchmark_id", benchmark_id)
        object.__setattr__(self, "family", self.family.strip())
        object.__setattr__(self, "primary_task", self.primary_task.strip())
        object.__setattr__(self, "default_source", self.default_source.strip())
        object.__setattr__(self, "default_protocol", self.default_protocol.strip())
        object.__setattr__(self, "record_kinds", tuple(dict.fromkeys(item.strip() for item in self.record_kinds if item.strip())))
        object.__setattr__(self, "aliases", tuple(dict.fromkeys(alias.strip() for alias in self.aliases if alias.strip())))
        object.__setattr__(self, "required_case_fields", tuple(dict.fromkeys(item.strip() for item in self.required_case_fields if item.strip())))
        object.__setattr__(self, "metadata", {str(key): str(value) for key, value in self.metadata.items()})

    def as_dict(self) -> dict[str, object]:
        return {
            "benchmark_id": self.benchmark_id,
            "display_name": self.display_name,
            "family": self.family,
            "primary_task": self.primary_task,
            "default_source": self.default_source,
            "default_protocol": self.default_protocol,
            "record_kinds": list(self.record_kinds),
            "aliases": list(self.aliases),
            "citation": self.citation,
            "data_access": self.data_access,
            "notes": self.notes,
            "required_case_fields": list(self.required_case_fields),
            "metadata": dict(self.metadata),
        }


# Keep descriptions deliberately conservative.  Exact corpus versions,
# licences, and annotation schemas must come from the researcher's authorised
# release rather than being inferred from a benchmark name.
_SPECS: tuple[BenchmarkSpec, ...] = (
    BenchmarkSpec(
        benchmark_id="egeo",
        display_name="E-GEO",
        family="E-commerce and product recommendation",
        primary_task="Context-rich product retrieval, recommendation, and citation visibility",
        default_source="Author-provided E-GEO export supplied by the researcher",
        default_protocol=(
            "Fixed ten-candidate e-commerce ranking task; report target rank lift, recommendation, and grounded citation outcomes"
        ),
        record_kinds=("cases", "predictions", "targets"),
        aliases=("e-geo", "e_geo", "eg eo"),
        citation="Bagga et al. (2025)",
        notes=(
            "E-GEO uses fixed candidate sets of ten products per query in its benchmark protocol. "
            "Use release-provided query, product, split, and relevance fields when available. "
            "Do not substitute scraped marketplace listings without recording the provenance."
        ),
    ),
    BenchmarkSpec(
        benchmark_id="autogeo_ecommerce",
        display_name="AutoGEO E-commerce",
        family="E-commerce and product recommendation",
        primary_task="Commercial-intent generative search and product recommendation",
        default_source="Author-provided AutoGEO e-commerce subset supplied by the researcher",
        default_protocol=(
            "Five-candidate commercial-query evaluation with GEO and GEU outcome reporting under the release protocol"
        ),
        record_kinds=("cases", "predictions", "targets"),
        aliases=("autogeo", "autogeo_e_commerce", "ecommerce_autogeo", "e_commerce_from_autogeo"),
        citation="Wu et al. (2025)",
        notes=(
            "The e-commerce setting uses five candidates and reports the GEO/GEU measures defined by AutoGEO. "
            "Preserve train/test membership and the original commercial-intent filtering decisions. "
            "Use a documented split rather than treating a dataset-level count as a protocol."
        ),
    ),
    BenchmarkSpec(
        benchmark_id="geo_bench",
        display_name="GEO-Bench",
        family="Generative engine optimisation",
        primary_task="Generative-engine visibility and response-level evaluation",
        default_source="Author-provided GEO-Bench export supplied by the researcher",
        default_protocol="Objective citation selection, visibility, and response-level metric evaluation",
        record_kinds=("cases", "predictions", "targets", "traces"),
        aliases=("geo-bench", "geobench", "geo_benchmark"),
        citation="GEO-Bench research release",
        notes=(
            "The name GEO-Bench is used by more than one emerging evaluation context. "
            "Use the release's objective citation metrics rather than subjective proxy labels; record the paper, release hash, and protocol."
        ),
    ),
    BenchmarkSpec(
        benchmark_id="researchy_geo",
        display_name="Researchy-GEO",
        family="Generative engine optimisation",
        primary_task="Role- and intent-aware generative search optimisation",
        default_source="Researchy-GEO or role-augmented protocol export supplied by the researcher",
        default_protocol="Role-conditioned, intent-conditioned multi-query evaluation",
        record_kinds=("cases", "predictions", "targets", "traces"),
        aliases=("researchy-geo", "researchygeo", "role_augmented_geo", "role_augmented_intent_driven_geo"),
        citation="Role-Augmented Intent-Driven Generative Search Engine Optimization research",
        notes=(
            "Store role, intent, and conflict-resolution annotations as case metadata. "
            "Do not collapse heterogeneous role conditions into a single unlabelled score."
        ),
    ),
    BenchmarkSpec(
        benchmark_id="opr_bench",
        display_name="OPR-Bench",
        family="E-commerce and product recommendation",
        primary_task="Open-ended product recommendation in web-enabled agent trajectories",
        default_source="Author-provided OPR-Bench export supplied by the researcher",
        default_protocol=(
            "Local controlled trajectory evaluation over the release's 3,124 query-product pairs; assess whether the target is surfaced"
        ),
        record_kinds=("cases", "predictions", "targets", "traces"),
        aliases=("opr-bench", "oprbench", "open_ended_product_recommendation"),
        citation="Ye et al. / EcoGEO (2026)",
        notes=(
            "OPR-Bench contains 3,124 query-product pairs in a locally controlled trajectory protocol. "
            "Keep the query-source stratum and target construction metadata. "
            "Targets may be fictional or absent from a local candidate corpus; the normalised model supports this."
        ),
    ),
    BenchmarkSpec(
        benchmark_id="aces",
        display_name="ACES",
        family="Agentic e-commerce",
        primary_task="Causal auditing of autonomous shopping-agent choice behaviour and biases",
        default_source="Author-provided ACES sandbox traces supplied by the researcher",
        default_protocol="Randomized causal choice audit of shopping-agent rationality and position/badge interventions",
        record_kinds=("cases", "predictions", "traces"),
        aliases=("agentic_ecommerce_simulator", "agentic_e_commerce_simulator", "aces_benchmark"),
        citation="Allouah et al. (2025)",
        notes=(
            "ACES is a causal randomized choice audit rather than a static relevance benchmark. "
            "Evaluate baseline rationality, position/badge interventions, and choice traces separately. "
            "Do not infer causal effects from observational ranking logs alone."
        ),
    ),
    BenchmarkSpec(
        benchmark_id="sageo",
        display_name="SAGEO Arena",
        family="Generative engine optimisation",
        primary_task="Realistic evaluation of generative search optimisation agents",
        default_source="Author-provided SAGEO Arena export supplied by the researcher",
        default_protocol="Stage-pipeline environment/episode evaluation with stage-labelled agent traces",
        record_kinds=("cases", "predictions", "traces"),
        aliases=("sageo_arena", "sageo-arena"),
        citation="SAGEO Arena research release",
        notes=(
            "SAGEO evaluates a staged pipeline; preserve stage name/order, episode seed, environment version, and intervention budget."
        ),
    ),
    BenchmarkSpec(
        benchmark_id="ifgeo",
        display_name="IF-GEO",
        family="Generative engine optimisation",
        primary_task="Conflict-aware instruction fusion for multi-query GEO",
        default_source="Author-provided IF-GEO export supplied by the researcher",
        default_protocol="Multi-query conflict/fusion evaluation with stability outcomes reported across instruction bundles",
        record_kinds=("cases", "predictions", "traces"),
        aliases=("if-geo", "if_geo", "instruction_fusion_geo"),
        citation="IF-GEO research release",
        notes=(
            "Store query bundles, instruction priorities, conflict labels, and repeated-run stability evidence in case or trace metadata."
        ),
    ),
    BenchmarkSpec(
        benchmark_id="citation_absorption",
        display_name="Citation Absorption",
        family="Citation quality and generative engine optimisation",
        primary_task="Citation selection, attribution, and absorption into generated answers",
        default_source="Author-provided citation-absorption evaluation export supplied by the researcher",
        default_protocol="Citation selection, entailment, placement, and absorption into the generated answer evaluation",
        record_kinds=("cases", "predictions", "targets", "traces"),
        aliases=("from_citation_selection_to_citation_absorption", "citation-absorption", "citation_absorption_bench"),
        citation="From Citation Selection to Citation Absorption research",
        notes=(
            "Use CitationSpan fields for source identity, answer offsets, and entailment labels where provided."
        ),
    ),
    BenchmarkSpec(
        benchmark_id="geo_bench_manipulation",
        display_name="GEO-Bench Manipulation",
        family="Security and manipulation resistance",
        primary_task="Ranking-manipulation effectiveness and robustness in generative engines",
        default_source="Author-provided GEO-Bench manipulation export supplied by the researcher",
        default_protocol="Defensive paired clean/manipulated ranking, citation, and safety audit",
        record_kinds=("cases", "predictions", "targets", "traces"),
        aliases=("geo_bench_ranking_manipulation", "geo-bench-manipulation", "geobench_manipulation"),
        citation="GEO-BENCH: Benchmarking Ranking Manipulation research",
        notes=(
            "Record attack condition, defensive setting, and clean counterfactual in metadata. "
            "Use this profile only for defensive robustness auditing; never label an observed change as successful manipulation without the paired protocol."
        ),
    ),
    BenchmarkSpec(
        benchmark_id="study_framework",
        display_name="DBA GEO E-commerce Study Framework",
        family="Controlled empirical study",
        primary_task="Blinded control-versus-factual-GEO product-page and consumer-shopping evaluation",
        default_source="Local study platform export with preregistered study metadata",
        default_protocol="Matched product/query comparison with controlled probes and participant behaviour analysed separately",
        record_kinds=("cases", "predictions", "targets", "traces"),
        aliases=("framework", "local_study", "dba_study", "geo_ecommerce_study"),
        citation="Generative Engine Optimization in E-Commerce: empirical study framework",
        notes=(
            "Keep CONTROL and GEO_OPTIMIZED allocation, treatment integrity, provenance, and participant consent separate "
            "from external-engine replication. The framework does not fabricate third-party citations or claims."
        ),
    ),
    BenchmarkSpec(
        benchmark_id="external_engine_monitor",
        display_name="External Engine Monitor",
        family="Generative search comparison",
        primary_task="Query-level monitoring of product discovery, citation, and answer behaviour across external engines",
        default_source="Researcher-collected, authorised external-engine run logs",
        default_protocol="Time-stamped repeated query runs with engine/model/version, locale, and observable citation logging",
        record_kinds=("cases", "predictions", "traces"),
        aliases=("chatgpt_google_comparison", "chatgpt_vs_google", "external_engine", "engine_monitor"),
        citation="ChatGPT vs. Google comparative search-performance study",
        notes=(
            "This profile evaluates observed outputs from external systems, not a controlled simulator. "
            "Keep engine access conditions, date/time, locale, query wording, and unavailable citations in provenance."
        ),
    ),
    BenchmarkSpec(
        benchmark_id="discovery_gap",
        display_name="Discovery Gap",
        family="LLM product discoverability",
        primary_task="Product/startup existence, coverage, and recommendation visibility in discovery queries",
        default_source="Author-provided discovery-gap query/product annotations or documented researcher replication export",
        default_protocol="Stratified discovery-query coverage and target-surfacing evaluation",
        record_kinds=("cases", "predictions", "targets"),
        aliases=("the_discovery_gap", "startup_discovery", "llm_discovery_queries"),
        citation="The Discovery Gap: How Product Startups Disappear in LLM Discovery Queries",
        notes=(
            "Report target coverage and absence separately from rank among surfaced candidates. "
            "Retain company/product age, category, and query-stratum annotations if provided."
        ),
    ),
    BenchmarkSpec(
        benchmark_id="cultural_coverage",
        display_name="Cultural Coverage / Existence Gap",
        family="Representation and coverage",
        primary_task="Culturally stratified existence, coverage, and response-quality evaluation",
        default_source="Author-provided cultural encoding/existence-gap annotations or authorised replication export",
        default_protocol="Predefined cultural-stratum coverage and output-comparison evaluation",
        record_kinds=("cases", "predictions", "targets"),
        aliases=("cultural_encoding", "existence_gap", "cultural_existence_gap"),
        citation="Cultural Encoding in Large Language Models: The Existence Gap",
        notes=(
            "Keep cultural/linguistic strata, annotator protocol, and missingness distinct from ordinary relevance labels. "
            "Do not derive sensitive attributes from product text without an approved protocol."
        ),
    ),
    BenchmarkSpec(
        benchmark_id="multimodal_geo",
        display_name="Multimodal GEO",
        family="Multimodal generative engine optimisation",
        primary_task="Visual-language and agentic product/content discovery evaluation",
        default_source="Author-provided multimodal GEO export with authorised image and text assets",
        default_protocol="Paired visual/text candidate evaluation with model, image preprocessing, and agent trace logging",
        record_kinds=("cases", "predictions", "targets", "traces"),
        aliases=("vlm_geo", "pinterest_geo", "visual_language_geo", "multimodal"),
        citation="Generative Engine Optimization: A VLM and Agent Framework (Pinterest)",
        notes=(
            "Record image identifiers, transformations, vision-model version, and modality availability. "
            "The common Candidate model stores asset references in metadata rather than embedding media."
        ),
    ),
    BenchmarkSpec(
        benchmark_id="human_shopping_experiment",
        display_name="Human AI-Assisted Shopping Experiment",
        family="Consumer behaviour",
        primary_task="Consumer product search, comparison, trust, and purchase-intention outcomes with AI assistance",
        default_source="Consent-approved study-platform exports and preregistered survey/task instruments",
        default_protocol="Blinded or explicitly labelled experimental conditions with participant events and survey outcomes analysed separately",
        record_kinds=("cases", "predictions", "traces"),
        aliases=("consumer_behaviour", "consumer_behavior", "shopping_assistant_experiment", "human_study"),
        citation="AI shopping-assistant and consumer trust/purchase-intention research",
        notes=(
            "Never infer participant consent, demographics, or purchase from model traces. "
            "Store de-identified session IDs, consent/version, condition, and survey measurement metadata outside generated answers."
        ),
    ),
    BenchmarkSpec(
        benchmark_id="competitive_citation",
        display_name="Competitive Citation GEO",
        family="Citation quality and generative engine optimisation",
        primary_task="Competitive source/product citation selection, prominence, and grounded recommendation evaluation",
        default_source="Author-provided competitive-citation export supplied by the researcher",
        default_protocol="Same-query competing-candidate citation, placement, and evidence-quality evaluation",
        record_kinds=("cases", "predictions", "targets", "traces"),
        aliases=("what_gets_cited", "competitive_geo", "competitive_citation_geo"),
        citation="What Gets Cited: Competitive GEO in AI Answer Engines",
        notes=(
            "Use CitationSpan provenance and entailment labels when available. "
            "Compare candidate opportunity, retrieval, citation, and first-citation prominence rather than only aggregate mentions."
        ),
    ),
)


BENCHMARK_REGISTRY: dict[str, BenchmarkSpec] = {spec.benchmark_id: spec for spec in _SPECS}

_ALIASES: dict[str, str] = {}
for _spec in _SPECS:
    for _alias in (_spec.benchmark_id, _spec.display_name, *_spec.aliases):
        _ALIASES[_canonicalise(_alias)] = _spec.benchmark_id


def normalize_benchmark_id(value: str) -> str:
    """Resolve a canonical registry id or raise a helpful KeyError."""

    key = _canonicalise(value)
    try:
        return _ALIASES[key]
    except KeyError as exc:
        available = ", ".join(sorted(BENCHMARK_REGISTRY))
        raise KeyError(f"Unknown benchmark '{value}'. Available ids: {available}.") from exc


def get_benchmark_spec(value: str) -> BenchmarkSpec:
    """Return registry metadata for a canonical id, display name, or alias."""

    return BENCHMARK_REGISTRY[normalize_benchmark_id(value)]


def list_benchmarks(*, family: str | None = None) -> tuple[BenchmarkSpec, ...]:
    """List specs in a deterministic order, optionally filtering by family."""

    if family is None:
        return tuple(sorted(BENCHMARK_REGISTRY.values(), key=lambda spec: spec.benchmark_id))
    expected = _canonicalise(family)
    return tuple(
        spec
        for spec in sorted(BENCHMARK_REGISTRY.values(), key=lambda item: item.benchmark_id)
        if _canonicalise(spec.family) == expected
    )


__all__ = [
    "BENCHMARK_REGISTRY",
    "BenchmarkSpec",
    "get_benchmark_spec",
    "list_benchmarks",
    "normalize_benchmark_id",
]
