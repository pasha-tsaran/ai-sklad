import hashlib
import hmac
import json
import re
from uuid import NAMESPACE_URL, uuid5

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.api.dependencies import get_session
from app.core.config import get_settings
from app.models import Conversation, Incident, User
from app.schemas.contracts import ChatRequest, EscalationDetails, FeedbackRequest
from app.services.chat.policies import FORM_ACTION, TERMINAL_STATES
from app.services.chat.service import accept_chat, accept_feedback, idempotent, messenger
from app.services.ticketing.service import collect_details

router = APIRouter()
FORM_HELP = "Отправьте одним сообщением:\nФИО: ...\nЛогин: ...\nКомпьютер: ...\nТема: ...\nОписание: ..."


def handle_event(session, payload, settings):
    actor, obj, target = payload.get("actor", {}), payload.get("object", {}), payload.get("target", {})
    if (
        payload.get("type") != "Create"
        or actor.get("type") != "Person"
        or not str(actor.get("id", "")).startswith("users/")
        or obj.get("type") != "Note"
        or obj.get("name") != "message"
    ):
        return {"ignored": True}
    room = target.get("id")
    if room not in settings.nextcloud_allowed_rooms:
        raise HTTPException(403, "Room not allowed")
    message_id = str(obj.get("id", ""))
    if not message_id or len(message_id) > 100:
        raise HTTPException(400, "Missing message id")
    content = json.loads(obj["content"])
    request = ChatRequest(message=content["message"])
    identity = settings.nextcloud_base_url.rstrip("/") + "/" + room + "/" + actor["id"]
    user_id = str(uuid5(NAMESPACE_URL, identity))
    if session.get(User, user_id) is None:
        try:
            with session.begin_nested():
                session.add(
                    User(
                        id=user_id,
                        external_id="talk:" + user_id,
                        display_name=str(actor.get("name", ""))[:255],
                    )
                )
                session.flush()
        except IntegrityError:
            pass  # Concurrent first webhook; the unique user row serializes subsequent work.

    def operation():
        conversation = session.scalar(select(Conversation).where(Conversation.user_id == user_id))
        if conversation is None:
            conversation = Conversation(user_id=user_id, channel="nextcloud_talk", room_token=room)
            session.add(conversation)
            session.flush()
        incident = session.scalar(
            select(Incident)
            .where(Incident.conversation_id == conversation.id, Incident.state.not_in(TERMINAL_STATES))
            .with_for_update()
        )
        try:
            with session.begin_nested():
                match = re.fullmatch(r"(не помогло|помогло)\s+([a-fA-F0-9-]{36})", request.message, re.I)
                if match and not incident:
                    return {"accepted": False}
                if match and incident:
                    return accept_feedback(
                        session,
                        user_id,
                        incident.id,
                        FeedbackRequest(attempt_id=match[2], helped=match[1].lower() == "помогло"),
                    )
                if incident and incident.pending_action == FORM_ACTION:
                    names = {
                        "фио": "account_holder_name",
                        "логин": "account_login",
                        "компьютер": "computer_name",
                        "тема": "subject",
                        "описание": "additional_info",
                    }
                    values = {}
                    for line in request.message.splitlines():
                        key, separator, value = line.partition(":")
                        if separator and key.strip().lower() in names:
                            values[names[key.strip().lower()]] = value.strip()
                    return collect_details(
                        session, user_id, incident.id, EscalationDetails(**values), settings
                    )
                return accept_chat(session, user_id, request)
        except (HTTPException, ValidationError):
            if incident:
                text = (
                    FORM_HELP
                    if incident.pending_action == FORM_ACTION
                    else (
                        "Сообщение не принято. Дождитесь ответа; для оценки скопируйте команду "
                        "«Помогло <код>» или «Не помогло <код>» из ответа бота."
                    )
                )
                messenger.send(session, incident.id, text)
            return {"accepted": False}

    key = hashlib.sha256((room + ":" + message_id).encode()).hexdigest()
    return idempotent(session, user_id, key, "talk", payload, operation)


@router.post("/api/integrations/nextcloud/talk")
async def webhook(request: Request, session=Depends(get_session)):
    settings = get_settings()
    if settings.nextcloud_backend != "talk":
        raise HTTPException(404)
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > 65536:
            raise HTTPException(413)
    nonce = request.headers.get("X-Nextcloud-Talk-Random", "")
    signature = request.headers.get("X-Nextcloud-Talk-Signature", "").lower()
    backend = request.headers.get("X-Nextcloud-Talk-Backend", "").rstrip("/")
    if len(nonce) != 64 or backend != settings.nextcloud_base_url.rstrip("/"):
        raise HTTPException(401)
    expected = hmac.new(
        settings.nextcloud_bot_secret.get_secret_value().encode(), nonce.encode() + body, hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(signature, expected):
        raise HTTPException(401)
    try:
        payload = json.loads(body)
        if not isinstance(payload, dict):
            raise ValueError()
        return handle_event(session, payload, settings)
    except (KeyError, TypeError, ValueError, AttributeError):
        raise HTTPException(400, "Invalid Talk event") from None
