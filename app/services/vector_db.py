import os
import json
import functools
from typing import Any, Iterable

from sqlalchemy import select, delete, func, text, desc
from openai import OpenAI

from app.db.session import SessionLocal
from app.models.study import ProductVector, Product
from app.services.retrieval import RankedCandidate
from app.services.text import to_mapping, normalize_whitespace

_openai_client = None

import functools


@functools.lru_cache(maxsize=1024)
def get_cached_embedding(text: str) -> list[float]:
    return generate_embeddings([text])[0]


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI()
    embeddings = []
    chunk_size = 200
    total_chunks = (len(texts) + chunk_size - 1) // chunk_size
    for i in range(0, len(texts), chunk_size):
        current_chunk = (i // chunk_size) + 1
        print(
            f"  -> Generating embeddings: batch {current_chunk}/{total_chunks} ({min(i + chunk_size, len(texts))} of {len(texts)} products)...")
        chunk = texts[i:i + chunk_size]
        response = _openai_client.embeddings.create(
            input=chunk,
            model="text-embedding-3-small",
            dimensions=384
        )
        embeddings.extend([data.embedding for data in response.data])
    return embeddings


def clear_collection():
    try:
        with SessionLocal() as db:
            db.execute(delete(ProductVector))
            db.commit()
    except Exception as e:
        print(f"Failed to clear ProductVector collection: {e}")


def index_products(products: Iterable[Any]):
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
        # Generate embeddings in memory using OpenAI
        embeddings = generate_embeddings(documents)

        batch_size = 200
        total_batches = (len(ids) + batch_size - 1) // batch_size
        for batch_start in range(0, len(ids), batch_size):
            batch_end = min(batch_start + batch_size, len(ids))
            current_batch = (batch_start // batch_size) + 1
            print(
                f"  -> Saving vectors to DB: batch {current_batch}/{total_batches} ({batch_end} of {len(ids)} products)...")
            max_retries = 10
            for attempt in range(max_retries):
                try:
                    with SessionLocal() as db:
                        for i in range(batch_start, batch_end):
                            p_id = ids[i]
                            # Check if it exists
                            existing = db.execute(
                                select(ProductVector).where(ProductVector.product_id == p_id)).scalar_one_or_none()
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
                    break  # Success, exit retry loop
                except Exception as e:
                    import time
                    if attempt < max_retries - 1:
                        print(
                            f"    [Retry {attempt + 1}/{max_retries}] Serialization error committing batch {current_batch}, retrying in {2 * (attempt + 1)}s...")
                        time.sleep(2 * (attempt + 1))
                    else:
                        print(f"Error committing batch {batch_start}-{batch_end} after {max_retries} attempts: {e}")
                        raise


def search_products(query: str, limit: int = 10, category_filter: str = None) -> list[RankedCandidate]:
    query_embedding = get_cached_embedding(query)

    with SessionLocal() as db:
        # We need a list of vectors. We will order by L2 distance `<->`
        # Because CockroachDB supports pgvector's `<->` operator
        stmt = select(
            ProductVector.id,
            ProductVector.metadata_json,
            ProductVector.embedding.l2_distance(query_embedding).label("distance")
        )

        if category_filter:
            # Join with Product to filter by category
            stmt = stmt.join(Product, ProductVector.product_id == Product.id)
            stmt = stmt.where(func.lower(Product.category) == normalize_whitespace(category_filter).lower())

        stmt = stmt.order_by("distance").limit(limit)
        results = db.execute(stmt).all()

        ranked = []
        for idx, row in enumerate(results):
            distance = row.distance
            score = max(0.0, 1.0 - (float(distance) / 2.0))

            product = row.metadata_json
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
                evidence_markers=product.get("geo_bundle", {}).get("evidence_markers", []) if product.get(
                    "geo_bundle") else [],
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
            "embedding_function": "OpenAI text-embedding-3-small (384d)",
            "path": "cockroachdb",
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "count": 0,
            "collection_name": "product_vectors",
            "embedding_function": "OpenAI text-embedding-3-small (384d)",
            "path": "cockroachdb",
        }


def get_indexed_products_page(page: int = 1, limit: int = 20, query: str = None, category_filter: str = None) -> dict[
    str, Any]:
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

            # Order by id deterministically, then Paginate directly in the database
            total_filtered = db.execute(select(func.count()).select_from(stmt.subquery())).scalar_one()

            pages = (total_filtered + limit - 1) // limit if limit > 0 else 1
            page = max(1, min(page, pages)) if pages > 0 else 1
            offset = (page - 1) * limit

            # Sort by ProductVector.id which is predictable
            stmt = stmt.order_by(ProductVector.id.asc()).offset(offset).limit(limit)
            results = db.execute(stmt).scalars().all()

            all_items = []
            for metadata in results:
                if metadata:
                    all_items.append(metadata)

            return {
                "items": all_items,
                "total": total_filtered,
                "page": page,
                "limit": limit,
                "pages": pages,
                "collection_total": total_count,
            }
