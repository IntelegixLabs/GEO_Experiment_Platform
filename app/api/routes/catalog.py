from typing import Any

from fastapi import APIRouter, Depends, Query as QueryParam
from sqlalchemy import case, func, or_, select
from sqlalchemy.orm import Session as DBSession

from app.core.catalog import public_product, researcher_product
from app.core.config import CONDITIONS, STUDY_NAME, get_settings
from app.core.experiment import tokenize
from app.db.session import get_db
from app.models import Product, QueryCandidate

router = APIRouter(tags=["GEO study"])
DEFAULT_PRODUCT_PAGE_SIZE = 48
MAX_PRODUCT_PAGE_SIZE = 96

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

@router.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "study": STUDY_NAME, "environment": get_settings().environment}

@router.get("/config")
def config(db: DBSession = Depends(get_db)) -> dict[str, Any]:
    categories = db.scalars(
        select(Product.category).where(Product.category.is_not(None)).distinct().order_by(Product.category)
    ).all()
    settings = get_settings()
    return {
        "study_name": STUDY_NAME,
        "conditions": list(CONDITIONS),
        "categories": categories,
        "environment": settings.environment,
        "catalog_seed_mode": "configured_csv" if settings.seed_catalog_csv else "database_catalog",
        "participant_notice": "This is a research catalog. No real purchases are processed.",
    }

@router.get("/products")
def products(
    category: str | None = QueryParam(default=None, max_length=250),
    q: str | None = QueryParam(default=None, max_length=500),
    limit: int = QueryParam(default=DEFAULT_PRODUCT_PAGE_SIZE, ge=1, le=MAX_PRODUCT_PAGE_SIZE),
    offset: int = QueryParam(default=0, ge=0),
    db: DBSession = Depends(get_db),
) -> dict[str, Any]:
    filters = []
    if category:
        filters.append(func.lower(Product.category) == category.strip().lower())
    predicate = _product_text_predicate(_search_terms(q))
    if predicate is not None:
        filters.append(predicate)
    statement = select(Product)
    count_statement = select(func.count()).select_from(Product)
    if filters:
        statement = statement.where(*filters)
        count_statement = count_statement.where(*filters)
    total = int(db.scalar(count_statement) or 0)
    result = db.scalars(
        statement.order_by(Product.category, Product.title, Product.id).offset(offset).limit(limit)
    ).all()
    return {
        "products": [public_product(product) for product in result],
        "total": total,
        "limit": limit,
        "offset": offset,
    }

@router.get("/research/products")
def research_products(
    category: str | None = QueryParam(default=None, max_length=250),
    limit: int = QueryParam(default=100, ge=1, le=500),
    offset: int = QueryParam(default=0, ge=0),
    db: DBSession = Depends(get_db),
) -> dict[str, Any]:
    citation_count = func.coalesce(func.sum(case((QueryCandidate.cited.is_(True), 1.0), else_=0.0)), 0.0).label("citations")
    opportunities = func.count(QueryCandidate.product_id).label("opportunities")
    statement = (
        select(Product, citation_count, opportunities)
        .outerjoin(QueryCandidate, QueryCandidate.product_id == Product.id)
        .group_by(Product.id)
        .order_by(Product.category, Product.title, Product.id)
    )
    if category:
        statement = statement.where(func.lower(Product.category) == category.strip().lower())
    total_statement = select(func.count()).select_from(Product)
    if category:
        total_statement = total_statement.where(func.lower(Product.category) == category.strip().lower())
    total = int(db.scalar(total_statement) or 0)
    return {
        "products": [
            researcher_product(product, int(citations or 0), int(opportunities or 0))
            for product, citations, opportunities in db.execute(statement.offset(offset).limit(limit)).all()
        ],
        "total": total,
        "limit": limit,
        "offset": offset,
    }
