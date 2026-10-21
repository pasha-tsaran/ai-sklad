from typing import Annotated

from fastapi import APIRouter, Depends, Header, HTTPException, Query

from app.api.dependencies import current_user, get_session
from app.core.config import get_settings
from app.models import Job
from app.schemas.contracts import ChatRequest, EscalationDetails, FeedbackRequest
from app.services.chat.service import accept_chat, accept_feedback, get_owned, idempotent, incident_view
from app.services.feedback.service import solution_stats
from app.services.rag.retrieval import LocalRetriever
from app.services.ticketing.service import collect_details

router = APIRouter(prefix="/api")
Key = Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=100)]


@router.post("/chat", status_code=202)
def chat(body: ChatRequest, key: Key, user_id=Depends(current_user), session=Depends(get_session)):
    return idempotent(
        session, user_id, key, "chat", body.model_dump(), lambda: accept_chat(session, user_id, body)
    )


@router.post("/incidents/{incident_id}/feedback", status_code=202)
def feedback(
    incident_id: str,
    body: FeedbackRequest,
    key: Key,
    user_id=Depends(current_user),
    session=Depends(get_session),
):
    return idempotent(
        session,
        user_id,
        key,
        f"feedback:{incident_id}",
        body.model_dump(),
        lambda: accept_feedback(session, user_id, incident_id, body),
    )


@router.post("/incidents/{incident_id}/escalation-details", status_code=202)
def escalation_details(
    incident_id: str,
    body: EscalationDetails,
    key: Key,
    user_id=Depends(current_user),
    session=Depends(get_session),
):
    return idempotent(
        session,
        user_id,
        key,
        f"escalation:{incident_id}",
        body.model_dump(),
        lambda: collect_details(session, user_id, incident_id, body, get_settings()),
    )


@router.get("/incidents/{incident_id}")
def incident(incident_id: str, user_id=Depends(current_user), session=Depends(get_session)):
    return incident_view(session, get_owned(session, incident_id, user_id))


@router.get("/operations/{operation_id}")
def operation(operation_id: str, user_id=Depends(current_user), session=Depends(get_session)):
    job = session.get(Job, operation_id)
    if not job:
        raise HTTPException(404, "Operation not found")
    get_owned(session, job.incident_id, user_id)
    return {"id": job.id, "status": job.status, "tries": job.tries, "error_code": job.error_code}


@router.get("/knowledge/search")
def search(
    q: str = Query(min_length=1, max_length=1000), user_id=Depends(current_user), session=Depends(get_session)
):
    return {
        "backend": "demo-hash-v1/lexical",
        "results": [
            {
                "solution_id": item.solution_id,
                "title": item.title,
                "score": item.score,
                "source_chunk_id": item.chunk_id,
            }
            for item in LocalRetriever(session, get_settings().rag_min_score).search(q, set())
        ],
    }


@router.get("/admin/solutions/stats")
def statistics(user_id=Depends(current_user), session=Depends(get_session)):
    # The mock principal is also the local demo administrator. Real RBAC comes later.
    return {"solutions": solution_stats(session)}
