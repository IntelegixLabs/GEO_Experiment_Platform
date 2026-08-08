import os
import json
from typing import Any, Iterable

from sqlalchemy import select, delete, func, text, desc
from chromadb.utils import embedding_functions

from app.db.session import SessionLocal
from app.models.study import ProductVector, Product
from app.services.retrieval import RankedCandidate
from app.services.text import to_mapping, normalize_whitespace

_ef = None

def get_embedding_function():
    global _ef
    if _ef is None:
        _ef = embedding_functions.DefaultEmbeddingFunction()
    return _ef

def clear_collection():
    try:
        with SessionLocal() as db:
            db.execute(delete(ProductVector))
            db.commit()
    except Exception as e:
        print(f"Failed to clear ProductVector collection: {e}")

def index_products(products: Iterable[Any]):
    ef = get_embedding_function()
    
    ids = []
    documents = []
    metadatas = []
    product_ids = []
    
    for product in products:
        record = to_mapping(product)
        product_id = str(record.get("id"))
        
        # Combine fields for embedding
        features = " ".join(record.get("key_features") or [])
        bundle = record.get("geo_bundle") or record.get("treatment_bundle")
        evidence_text = ""
        if bundle:
            evidence_text = " ".join([
                str(bundle.get("summary", "")),
                " ".join(str(v) for v in bundle.get("specifications", {}).values()),
                " ".join(str(block.get("claim", "")) for block in bundle.get("claim_blocks", [])),
            ])
            
        document = f"{record.get('title', '')} {record.get('brand', '')} {record.get('category', '')} {record.get('description', '')} {features} {evidence_text}"
        documents.append(normalize_whitespace(document))
        ids.append(product_id)
        product_ids.append(product_id)
        
        safe_record = dict(record)
        metadatas.append(safe_record)
        
    if documents:
        # Generate embeddings in memory using Chroma's default embedding function
        embeddings = ef(documents)
        
        with SessionLocal() as db:
            for i, p_id in enumerate(ids):
                # Check if it exists
                existing = db.execute(select(ProductVector).where(ProductVector.product_id == p_id)).scalar_one_or_none()
                if existing:
                    existing.embedding = embeddings[i]
                    existing.metadata_json = metadatas[i]
                else:
                    vec = ProductVector(
                        id=p_id,
                        product_id=p_id,
                        embedding=embeddings[i],
                        metadata_json=metadatas[i]
                    )
                    db.add(vec)
            db.commit()

def search_products(query: str, limit: int = 10, category_filter: str = None) -> list[RankedCandidate]:
    ef = get_embedding_function()
    query_embedding = ef([query])[0]
    
    with SessionLocal() as db:
        count = db.execute(select(func.count()).select_from(ProductVector)).scalar_one()
        if count == 0:
            return []
            
        # We need a list of vectors. We will order by L2 distance `<->`
        # Because CockroachDB supports pgvector's `<->` operator
        stmt = select(ProductVector, ProductVector.embedding.l2_distance(query_embedding).label("distance"))
        
        if category_filter:
            # Join with Product to filter by category
            stmt = stmt.join(Product, ProductVector.product_id == Product.id)
            stmt = stmt.where(func.lower(Product.category) == normalize_whitespace(category_filter).lower())
            
        stmt = stmt.order_by("distance").limit(limit)
        results = db.execute(stmt).all()
        
        ranked = []
        for idx, row in enumerate(results):
            pv = row.ProductVector
            distance = row.distance
            score = max(0.0, 1.0 - (float(distance) / 2.0))
            
            product = pv.metadata_json
            product["score"] = round(score, 6)
            product["retrieval_score"] = round(score, 6)
            
            candidate = RankedCandidate(
                product=product,
                rank_position=idx + 1,
                retrieval_score=round(score, 6),
                lexical_score=0.0,
                semantic_score=round(score, 6),
                evidence_score=0.0,
                is_relevant=True,
                retrieved=True,
                reasons=["Semantic vector match"],
                evidence_markers=product.get("geo_bundle", {}).get("evidence_markers", []) if product.get("geo_bundle") else [],
            )
            ranked.append(candidate)
            
        return ranked

def get_vectordb_status() -> dict[str, Any]:
    try:
        with SessionLocal() as db:
            count = db.execute(select(func.count()).select_from(ProductVector)).scalar_one()
        return {
            "status": "online",
            "count": count,
            "collection_name": "product_vectors",
            "embedding_function": "DefaultEmbeddingFunction (CockroachDB Native)",
            "path": "cockroachdb",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "count": 0,
            "collection_name": "product_vectors",
            "embedding_function": "DefaultEmbeddingFunction",
            "path": "cockroachdb",
        }

def get_indexed_products_page(page: int = 1, limit: int = 20, query: str = None, category_filter: str = None) -> dict[str, Any]:
    with SessionLocal() as db:
        total_count = db.execute(select(func.count()).select_from(ProductVector)).scalar_one()
        if total_count == 0:
            return {"items": [], "total": 0, "page": page, "limit": limit, "pages": 0, "collection_total": 0}
            
        if query and query.strip():
            candidates = search_products(query.strip(), limit=max(250, total_count), category_filter=category_filter)
            all_items = []
            for idx, c in enumerate(candidates):
                prod = c.product
                prod["rank_position"] = idx + 1
                prod["retrieval_score"] = c.retrieval_score
                all_items.append(prod)
        else:
            # Query just the metadata_json instead of the whole model (which includes heavy vectors)
            stmt = select(ProductVector.metadata_json)
            if category_filter and category_filter.strip():
                stmt = stmt.join(Product, ProductVector.product_id == Product.id)
                stmt = stmt.where(func.lower(Product.category) == category_filter.strip().lower())
                
            # Paginate directly in the database
            total_filtered = db.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()
            
            pages = (total_filtered + limit - 1) // limit if limit > 0 else 1
            page = max(1, min(page, pages)) if pages > 0 else 1
            offset = (page - 1) * limit
            
            stmt = stmt.offset(offset).limit(limit)
            results = db.execute(stmt).scalars().all()
            
            all_items = []
            for metadata in results:
                if metadata:
                    all_items.append(metadata)
            
            # Since we only fetched a page, sort the page, or ideally rely on DB sorting.
            all_items.sort(key=lambda p: (str(p.get("category", "")), str(p.get("title", "")), str(p.get("id", ""))))
            
            return {
                "items": all_items,
                "total": total_filtered,
                "page": page,
                "limit": limit,
                "pages": pages,
                "collection_total": total_count,
            }
