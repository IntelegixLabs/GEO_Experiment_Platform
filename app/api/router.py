"""Top-level API router."""

from fastapi import APIRouter

from app.api.routes.admin import router as admin_router
from app.api.routes.analytics import router as analytics_router
from app.api.routes.assistant import router as assistant_router
from app.api.routes.catalog import router as catalog_router
from app.api.routes.sessions import router as sessions_router

api_router = APIRouter()
api_router.include_router(admin_router)
api_router.include_router(analytics_router)
api_router.include_router(assistant_router)
api_router.include_router(catalog_router)
api_router.include_router(sessions_router)
