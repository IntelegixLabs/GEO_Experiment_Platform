import secrets
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, or_, select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.orm import Session as DBSession

from app.core.catalog import product_record, public_product
from app.core.config import get_settings
from app.core.experiment import make_answer, query_intent, rank_catalog, tokenize, trim_text, utc_now
from app.db.session import get_db
from app.models import GEOOptimizationConfig, ProbeCandidate, ProbeRun, Product, Query, QueryCandidate, Session
from app.schemas import AssistantQueryCreate, ProbeCreate
from app.schemas.study import GEO_PARAMETER_WEIGHT_DEFAULTS
from app.services.geo_service import GEOService
from app.services.retrieval import RetrievalConfig

router = APIRouter(tags=["GEO study"])
RETRIEVAL_POOL_SIZE = 250


def _bad_request(message: str) -> HTTPException:
    return HTTPException(status_code=400, detail=message)


def _active_session(db: DBSession, session_id: str) -> Session:
    if not session_id or not str(session_id).strip():
        raise _bad_request("A session ID is required.")
    session_id = str(session_id).strip()
    session = db.get(Session, session_id)
    if session is None:
        session = Session(
            id=session_id,
            participant_code=f"P-{secrets.token_hex(3).upper()}",
            consent=True,
            study_cohort="auto-recovered",
            started_at=utc_now(),
        )
        db.add(session)
        db.commit()
    elif not session.consent:
        session.consent = True
        db.add(session)
        db.commit()
    return session


def _search_terms(value: str | None, *, maximum: int = 8) -> list[str]:
    terms: list[str] = []
    for token in tokenize(value or ""):
        if len(token) < 2 or token in terms:
            continue
        terms.append(token[:80])
        if len(terms) >= maximum:
            break
    return terms


def _product_text_predicate(terms: list[str]):
    if not terms:
        return None
    fields = (Product.title, Product.category, Product.main_category, Product.sub_category)
    return or_(
        *[
            func.lower(field).like(f"%{term.lower()}%")
            for term in terms
            for field in fields
        ]
    )


def _catalog_records(
        db: DBSession,
        category_filter: str | None = None,
        query_text: str | None = None,
        *,
        limit: int = RETRIEVAL_POOL_SIZE,
) -> list[dict[str, Any]]:
    # Search ONLY ChromaDB Vector DB using semantic similarity matching
    try:
        from app.services.vector_db import search_products
        ranked_candidates = search_products(query_text or "", limit=limit, category_filter=category_filter)
        results = []
        for c in ranked_candidates:
            prod_dict = c.product
            pid = prod_dict.get("id")
            if pid and not db.get(Product, pid):
                try:
                    db.add(
                        Product(
                            id=pid,
                            title=str(prod_dict.get("title") or "Product"),
                            category=str(prod_dict.get("category") or "Uncategorized"),
                            condition=str(prod_dict.get("condition") or "CONTROL"),
                            price=float(prod_dict.get("price", 0)) if prod_dict.get("price") is not None else None,
                            description=str(prod_dict.get("description") or ""),
                            key_features=prod_dict.get("key_features") if isinstance(prod_dict.get("key_features"),
                                                                                     list) else [],
                            created_at=utc_now(),
                        )
                    )
                    db.commit()
                except Exception:
                    db.rollback()
            results.append(prod_dict)
        return results
    except Exception as e:
        print(f"Vector DB retrieval error in assistant_query: {e}")
        return []


def _candidate_payload(product: dict[str, Any], position: int) -> dict[str, Any]:
    return {
        "rank_position": position,
        "product_id": product["id"],
        "condition": product["condition"],
        "retrieval_score": float(product.get("retrieval_score", product.get("score", 0.0))),
        "lexical_score": float(product.get("lexical_score", 0.0)),
        "evidence_score": float(product.get("evidence_score", 0.0)),
    }


def _active_geo_service(db: DBSession) -> GEOService | None:
    active = db.scalar(
        select(GEOOptimizationConfig).where(GEOOptimizationConfig.is_active.is_(True)).order_by(
            GEOOptimizationConfig.revision.desc())
    )
    if active is None:
        return None
    weights = dict(GEO_PARAMETER_WEIGHT_DEFAULTS)
    weights.update(
        {key: float(value) for key, value in (active.parameter_weights_json or {}).items() if key in weights})
    settings = get_settings()
    generation_adapter = None
    if settings.openai_api_key:
        from app.services.openai_adapter import OpenAIGenerationAdapter
        generation_adapter = OpenAIGenerationAdapter(
            model_name=settings.openai_model_name,
            api_key=settings.openai_api_key,
        )

    return GEOService(
        retrieval_config=RetrievalConfig(**weights, version=f"geo-config-{active.revision}"),
        generation_adapter=generation_adapter,
        allow_external_generation=bool(generation_adapter),
    )


@router.post("/assistant/query", status_code=201)
def assistant_query(payload: AssistantQueryCreate, db: DBSession = Depends(get_db)) -> dict[str, Any]:
    query_text = trim_text(payload.query, 500)
    category_filter = trim_text(payload.category_filter, 250)
    if not query_text:
        raise _bad_request("A session ID and a product-search question are required.")
    _active_session(db, payload.session_id)
    intent = query_intent(query_text, category_filter)
    catalog = _catalog_records(db, category_filter or None, query_text)
    configured_service = _active_geo_service(db)
    ranked = rank_catalog(catalog, query_text, category_filter or None, intent, service=configured_service)
    answer, cited, suggestions = make_answer(query_text, ranked, intent, service=configured_service)
    cited_ids = {product["id"] for product in cited}
    query_id = f"Q-{uuid.uuid4().hex}"
    citations = [
        {
            "product": public_product(product),
            "rank_position": position,
            "recommendation_score": product["score"],
            "reasons": product.get("reasons", []),
            "evidence_markers": product.get("evidence_markers", []),
        }
        for position, product in enumerate(cited, start=1)
    ]
    response_payload = {"answer": answer, "citations": citations, "intent": intent, "suggestions": suggestions}
    record = Query(
        id=query_id,
        session_id=payload.session_id,
        query_text=query_text,
        category_filter=category_filter or None,
        intent_json=intent,
        response_json=response_payload,
        created_at=utc_now(),
    )
    candidates = [
        QueryCandidate(query_id=query_id, cited=product["id"] in cited_ids, **_candidate_payload(product, position))
        for position, product in enumerate(ranked, start=1)
    ]

    max_retries = 3
    for attempt in range(max_retries):
        try:
            db.add(record)
            db.add_all(candidates)
            db.commit()
            break
        except OperationalError as error:
            db.rollback()
            if getattr(error.orig, "sqlstate", None) == "40001" and attempt < max_retries - 1:
                import time
                time.sleep(0.1 * (2 ** attempt))
                continue
            raise
        except IntegrityError as error:
            db.rollback()
            raise _bad_request(f"Database constraint: {error.orig}") from error
    return {"query_id": query_id, **response_payload}


@router.post("/admin/probes", status_code=201)
def run_probes(payload: ProbeCreate, db: DBSession = Depends(get_db)) -> dict[str, Any]:
    query_text = trim_text(payload.query, 500)
    category_filter = trim_text(payload.category_filter, 250)
    if not query_text:
        raise _bad_request("A probe query is required.")
    intent = query_intent(query_text, category_filter)
    catalog = _catalog_records(db, category_filter or None, query_text)
    runs: list[dict[str, Any]] = []
    for repetition in range(1, payload.repetitions + 1):
        configured_service = _active_geo_service(db)
        ranked = rank_catalog(catalog, query_text, category_filter or None, intent, service=configured_service)
        answer, cited, _ = make_answer(query_text, ranked, intent, service=configured_service)
        cited_ids = {product["id"] for product in cited}
        probe_id = f"PR-{uuid.uuid4().hex}"
        response_payload = {
            "answer": answer,
            "cited_product_ids": [product["id"] for product in cited],
            "intent": intent,
        }
        db.add(
            ProbeRun(
                id=probe_id,
                probe_set=trim_text(payload.probe_set, 80) or "ad_hoc",
                repetition=repetition,
                engine_name=trim_text(payload.engine_name, 120) or "Controlled catalog retrieval",
                model_version=trim_text(payload.model_version, 120) or "GEO-Study-Retriever/1.0",
                locale=trim_text(payload.locale, 60) or None,
                query_text=query_text,
                category_filter=category_filter or None,
                intent_json=intent,
                response_json=response_payload,
                created_at=utc_now(),
            )
        )
        db.flush()
        for position, product in enumerate(ranked, start=1):
            candidate = _candidate_payload(product, position)
            db.add(ProbeCandidate(probe_run_id=probe_id, cited=product["id"] in cited_ids, **candidate))
        candidates_detail = []
        for pos, product in enumerate(ranked, start=1):
            if pos <= 10 or product["id"] in cited_ids:
                candidates_detail.append({
                    "rank_position": pos,
                    "product_id": product["id"],
                    "title": product.get("title", ""),
                    "category": product.get("category", ""),
                    "condition": product.get("condition", "UNKNOWN"),
                    "retrieval_score": round(float(product.get("retrieval_score", product.get("score", 0.0))), 3),
                    "lexical_score": round(float(product.get("lexical_score", 0.0)), 3),
                    "semantic_score": round(float(product.get("semantic_score", 0.0)), 3),
                    "evidence_score": round(float(product.get("evidence_score", 0.0)), 3),
                    "cited": product["id"] in cited_ids,
                })
        runs.append(
            {
                "probe_run_id": probe_id,
                "repetition": repetition,
                "cited_products": [product["title"] for product in cited],
                "cited_conditions": [product["condition"] for product in cited],
                "answer": answer,
                "candidates": candidates_detail,
            }
        )
    db.commit()
    return {
        "ok": True,
        "query": query_text,
        "probe_set": trim_text(payload.probe_set, 80) or "ad_hoc",
        "engine_name": trim_text(payload.engine_name, 120) or "Controlled catalog retrieval",
        "model_version": trim_text(payload.model_version, 120) or "GEO-Study-Retriever/1.0",
        "runs": runs,
        "message": "Controlled probes logged. Export probe and probe-candidate records for analysis; do not pool them with participant behavior records.",
    }
