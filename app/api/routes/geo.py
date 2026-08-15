from typing import Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.agentic_geo import AgenticGEOOptimizer
from app.services.geobench_eval import GeoBenchEvaluator
from app.core.config import get_settings

settings = get_settings()

router = APIRouter(prefix="/geo", tags=["geo"])

class ProductRequest(BaseModel):
    product: dict[str, Any]

class RewriteRequest(BaseModel):
    product: dict[str, Any]
    strategies: List[str]

class EvaluateRequest(BaseModel):
    query: str
    original_product: dict[str, Any]
    optimized_product: dict[str, Any]
    competitors: List[dict[str, Any]]

@router.post("/plan")
def plan_geo_strategies(req: ProductRequest):
    optimizer = AgenticGEOOptimizer(api_key=settings.openai_api_key)
    try:
        strategies = optimizer.plan_strategies(req.product)
        return {"strategies": strategies}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/rewrite")
def rewrite_product(req: RewriteRequest):
    optimizer = AgenticGEOOptimizer(api_key=settings.openai_api_key)
    try:
        optimized = optimizer.rewrite_product(req.product, req.strategies)
        return {"optimized_product": optimized}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate")
def evaluate_product(req: EvaluateRequest):
    evaluator = GeoBenchEvaluator(api_key=settings.openai_api_key)
    try:
        results = evaluator.evaluate_optimization(
            query=req.query,
            original_product=req.original_product,
            optimized_product=req.optimized_product,
            competitors=req.competitors
        )
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
