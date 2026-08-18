import secrets
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session as DBSession

from app.core.experiment import scale_scores, trim_text, utc_now
from app.db.session import get_db
from app.models import Event, Product, Query, Session, SurveyResponse
from app.schemas import EventCreate, SessionCreate, SurveyCreate

router = APIRouter(tags=["GEO study"])
ALLOWED_EVENT_TYPES = {"product_open", "citation_open", "comparison_add", "purchase_intent", "survey_open", "query_submit"}

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

@router.post("/sessions", status_code=201)
def create_session(payload: SessionCreate, db: DBSession = Depends(get_db)) -> dict[str, Any]:
    if payload.consent is not True:
        raise _bad_request("Active consent is required before a study session can begin.")
    participant_code = trim_text(payload.participant_code, 120) or f"P-{secrets.token_hex(3).upper()}"
    study_cohort = trim_text(payload.study_cohort, 60) or "main"
    record = Session(
        id=f"S-{uuid.uuid4().hex}",
        participant_code=participant_code,
        email=trim_text(payload.email, 160) if payload.email else None,
        age=payload.age,
        consent=True,
        country=trim_text(payload.country, 80) or None,
        ai_familiarity=trim_text(payload.ai_familiarity, 80) or None,
        study_cohort=study_cohort,
        started_at=utc_now(),
    )
    db.add(record)
    db.commit()
    return {
        "session_id": record.id,
        "participant_code": record.participant_code,
        "email": record.email,
        "age": record.age,
        "country": record.country,
        "study_cohort": record.study_cohort,
    }

@router.post("/events", status_code=201)
def record_event(payload: EventCreate, db: DBSession = Depends(get_db)) -> dict[str, bool]:
    event_type = trim_text(payload.event_type, 60)
    if event_type not in ALLOWED_EVENT_TYPES:
        raise _bad_request("Unknown event type.")
    _active_session(db, payload.session_id)
    if payload.product_id and db.get(Product, payload.product_id) is None:
        raise _bad_request("The selected product does not exist.")
    if payload.query_id:
        query = db.get(Query, payload.query_id)
        if query is None or query.session_id != payload.session_id:
            raise _bad_request("The referenced query does not belong to this study session.")
    db.add(
        Event(
            id=f"E-{uuid.uuid4().hex}",
            session_id=payload.session_id,
            query_id=payload.query_id or None,
            product_id=payload.product_id or None,
            event_type=event_type,
            metadata_json=payload.metadata if isinstance(payload.metadata, dict) else {},
            created_at=utc_now(),
        )
    )
    db.commit()
    return {"ok": True}

@router.post("/surveys", status_code=201)
def record_survey(payload: SurveyCreate, db: DBSession = Depends(get_db)) -> dict[str, Any]:
    if not isinstance(payload.answers, dict):
        raise _bad_request("A session ID and questionnaire answers are required.")
    scores = scale_scores(payload.answers)
    if not any(value is not None for value in scores.values()):
        raise _bad_request("Please complete at least one questionnaire construct.")
    session = _active_session(db, payload.session_id)
    now = utc_now()
    existing = db.scalar(select(SurveyResponse).where(SurveyResponse.session_id == payload.session_id))
    if existing:
        existing.answers_json = payload.answers
        existing.scale_scores_json = scores
        existing.completed_at = now
    else:
        db.add(
            SurveyResponse(
                id=f"SR-{uuid.uuid4().hex}",
                session_id=payload.session_id,
                answers_json=payload.answers,
                scale_scores_json=scores,
                completed_at=now,
            )
        )
    session.completed_at = now
    db.commit()
    return {"ok": True, "scale_scores": scores}
