from fastapi import FastAPI, HTTPException, Request
from sqlalchemy import text

from app.api.routes import router
from app.api.talk import router as talk_router
from app.api.device import router as device_router
from app.core.config import get_settings
from app.core.logging import configure_logging, event
from app.db import get_engine

configure_logging()
app = FastAPI(title="ai-support", version="0.4.4", description="Grounded support, personal Talk chats and mail drafts")
app.include_router(router)
app.include_router(talk_router)
app.include_router(device_router)


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
        revision = connection.execute(text("SELECT version_num FROM alembic_version")).scalar()
        if revision != "0011":
            raise HTTPException(503, {"error_code": "DatabaseMigrationRequired", "expected_revision": "0011"})
    settings = get_settings()
    return {
        "status": "ok",
        "app_version": app.version,
        "database_revision": revision,
        "llm": settings.llm_backend,
        "nextcloud": settings.nextcloud_backend,
        "mail": settings.mail_backend,
        "rag": settings.knowledge_backend,
        "knowledge_configured": bool(settings.openwebui_knowledge_ids)
        if settings.knowledge_backend == "openwebui"
        else True,
        "demo_knowledge": settings.seed_demo_knowledge,
        "talk_images_configured": (bool(settings.nextcloud_username)
                                   and bool(settings.nextcloud_app_password.get_secret_value()))
        if settings.nextcloud_backend == "account" else
        (bool(settings.nextcloud_files_username) and bool(settings.nextcloud_files_app_password.get_secret_value())),
        "talk_account_configured": settings.nextcloud_backend == "account" and bool(settings.nextcloud_username)
        and bool(settings.nextcloud_app_password.get_secret_value()),
        "talk_private_only": settings.nextcloud_backend == "account",
        "device_collection_configured": bool(settings.device_collection_base_url),
        "talk_confirm_understanding": settings.talk_confirm_understanding,
    }
