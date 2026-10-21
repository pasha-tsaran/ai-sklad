from fastapi import FastAPI, Request
from sqlalchemy import text

from app.api.routes import router
from app.api.talk import router as talk_router
from app.core.config import get_settings
from app.core.logging import configure_logging, event
from app.db import get_engine

configure_logging()
app = FastAPI(title="ai-support", version="0.2.0", description="Open WebUI knowledge, Talk bot and SMTP")
app.include_router(router)
app.include_router(talk_router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    response = await call_next(request)
    # Do not log body, query parameters, tokens or personal data.
    route = request.scope.get("route")
    event(
        "http.request",
        method=request.method,
        route=getattr(route, "path", "unknown"),
        status=response.status_code,
    )
    return response


@app.get("/health/live")
def live():
    return {"status": "ok"}


@app.get("/health/ready")
def ready():
    with get_engine().connect() as connection:
        connection.execute(text("SELECT 1 FROM alembic_version"))
    settings = get_settings()
    return {
        "status": "ok",
        "llm": settings.llm_backend,
        "nextcloud": settings.nextcloud_backend,
        "mail": settings.mail_backend,
        "rag": settings.knowledge_backend,
        "knowledge_configured": bool(settings.openwebui_knowledge_ids)
        if settings.knowledge_backend == "openwebui"
        else True,
        "demo_knowledge": settings.seed_demo_knowledge,
    }
