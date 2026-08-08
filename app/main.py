"""ASGI entry point for the GEO Shopping Lab FastAPI backend."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.router import api_router
from app.core.config import STUDY_NAME, get_settings
from app.db.base import Base
from app.db.seed import seed_demo_catalog
from app.db import session as database

# Register ORM model metadata before create_all.
import app.models  # noqa: F401


@asynccontextmanager
async def lifespan(_: FastAPI):
    import time
    t0 = time.time()
    print("[STARTUP] Warming up DB pool...")
    database.warmup_pool()
    print(f"[STARTUP] Pool warm in {time.time()-t0:.2f}s. Seeding catalog...")
    Base.metadata.create_all(bind=database.engine)
    with database.SessionLocal() as db:
        seed_demo_catalog(db)
    print(f"[STARTUP] Catalog seed done in {time.time()-t0:.2f}s total. Ready!")
    yield


settings = get_settings()
app = FastAPI(
    title="GEO Shopping Lab API",
    description="Controlled e-commerce GEO experiment backend.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins),
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)
app.include_router(api_router, prefix=settings.api_prefix)


@app.exception_handler(HTTPException)
async def http_error(_: Request, error: HTTPException) -> JSONResponse:
    """Keep FastAPI failures compatible with the original browser client."""

    message = error.detail if isinstance(error.detail, str) else "The study server could not process that request."
    return JSONResponse(status_code=error.status_code, content={"error": message})


@app.exception_handler(RequestValidationError)
async def validation_error(_: Request, error: RequestValidationError) -> JSONResponse:
    first = error.errors()[0] if error.errors() else {}
    message = str(first.get("msg") or "The request data is invalid.")
    return JSONResponse(status_code=422, content={"error": message})
