import hashlib
import hmac
import json
from types import SimpleNamespace

import httpx
import pytest
from sqlalchemy import func, select

from app.core.config import Settings, get_settings
from app.models import Delivery, Escalation, Incident, Job, Message, SolutionAttempt
from app.services.mail.smtp import SMTPMail
from app.services.nextcloud.delivery import TalkClient, deliver_one
from app.services.rag.openwebui import OpenWebUIKnowledge


def remote_settings(**kwargs):
    return Settings(_env_file=None, openwebui_api_key="test", openwebui_knowledge_ids=["approved"], **kwargs)


def knowledge_transport(selected="file-1", content="Проверьте подключение USB.", revoked=False):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        assert request.headers["Authorization"] == "Bearer test"
        if request.url.path.endswith("/knowledge/approved"):
            files = [] if revoked and len(calls) > 2 else [{"id": "file-1", "filename": "Zebra USB"}]
            return httpx.Response(200, json={"files": files})
        if request.url.path.endswith("/chat/completions"):
            body = json.loads(request.content)
            assert body["files"] == [{"type": "collection", "id": "approved"}]
            assert body["tool_ids"] == []
            return httpx.Response(
                200, json={"choices": [{"message": {"content": json.dumps({"file_id": selected})}}]}
            )
        assert request.url.path == "/api/v1/files/file-1/data/content"
        return httpx.Response(200, json={"content": content})

    return httpx.MockTransport(handler), calls


def test_knowledge_exact_source_and_stable_exclusion():
    transport, calls = knowledge_transport()
    provider = OpenWebUIKnowledge(remote_settings(), transport)
    candidate = provider.search("Zebra USB не печатает", set())[0]
    assert candidate.content == "Проверьте подключение USB."
    assert candidate.source["file_id"] == "file-1"
    assert candidate.source["sha256"] == hashlib.sha256(candidate.content.encode()).hexdigest()
    before = len(calls)
    assert provider.search("повтор", {candidate.solution_id}) == []
    assert len(calls) == before + 1  # No model call or changed-file retry.


@pytest.mark.parametrize("selected,revoked", [(None, False), ("file-1", True)])
def test_knowledge_missing_or_revoked(selected, revoked):
    transport, _ = knowledge_transport(selected=selected, revoked=revoked)
    assert OpenWebUIKnowledge(remote_settings(), transport).search("принтер", set()) == []


@pytest.mark.parametrize("selected,content", [("invented", "text"), ("file-1", ""), ("file-1", "a" * 20001)])
def test_knowledge_rejects_invalid_sources(selected, content):
    transport, _ = knowledge_transport(selected, content)
    with pytest.raises(ValueError):
        OpenWebUIKnowledge(remote_settings(), transport).search("принтер", set())


def test_remote_worker_keeps_snapshot_and_no_local_fallback(env, monkeypatch):
    settings = get_settings()
    settings.knowledge_backend = "openwebui"
    transport, _ = knowledge_transport()
    provider = OpenWebUIKnowledge(
        remote_settings(knowledge_backend="openwebui", seed_demo_knowledge=False), transport
    )
    monkeypatch.setattr("app.workers.handlers.OpenWebUIKnowledge", lambda _: provider)
    response = env.post("/api/chat", {"message": "Принтер Zebra не печатает"}).json()
    env.drain()
    with env.factory() as session:
        attempt = session.scalar(select(SolutionAttempt))
        assert attempt.evidence["source"]["file_id"] == "file-1"
        assert session.get(Message, attempt.message_id).content.count("Проверьте подключение USB.") == 1
        attempt_id = attempt.id
    env.post(
        f"/api/incidents/{response['incident_id']}/feedback", {"attempt_id": attempt_id, "helped": False}
    )
    env.drain()
    with env.factory() as session:
        assert session.scalar(select(func.count()).select_from(SolutionAttempt)) == 1
        assert session.get(Incident, response["incident_id"]).pending_action == "COLLECT_ESCALATION_DETAILS"


def setup_talk(monkeypatch):
    monkeypatch.setenv("NEXTCLOUD_BACKEND", "talk")
    monkeypatch.setenv("NEXTCLOUD_BASE_URL", "https://talk.internal")
    monkeypatch.setenv("NEXTCLOUD_BOT_SECRET", "s" * 32)
    monkeypatch.setenv("NEXTCLOUD_ALLOWED_ROOMS", '["room-a","room-b"]')
    get_settings.cache_clear()


def post_talk(env, text, message_id="1", user="ada", room="room-a", tamper=False):
    payload = {
        "type": "Create",
        "actor": {"type": "Person", "id": "users/" + user, "name": user},
        "object": {
            "type": "Note",
            "id": message_id,
            "name": "message",
            "content": json.dumps({"message": text, "parameters": {}}),
        },
        "target": {"id": room},
    }
    raw = json.dumps(payload).encode()
    nonce = "a" * 64
    signature = hmac.new(b"s" * 32, nonce.encode() + raw, hashlib.sha256).hexdigest()
    return env.client.post(
        "/api/integrations/nextcloud/talk",
        content=raw + (b" " if tamper else b""),
        headers={
            "X-Nextcloud-Talk-Random": nonce,
            "X-Nextcloud-Talk-Signature": signature,
            "X-Nextcloud-Talk-Backend": "https://talk.internal",
        },
    )


def test_talk_signature_dedup_and_room_scope(env, monkeypatch):
    setup_talk(monkeypatch)
    assert post_talk(env, "Принтер не печатает", tamper=True).status_code == 401
    assert post_talk(env, "Принтер не печатает", room="other").status_code == 403
    first = post_talk(env, "Принтер не печатает").json()
    assert post_talk(env, "Принтер не печатает").json() == first
    second = post_talk(env, "Принтер не печатает", room="room-b").json()
    assert second["incident_id"] != first["incident_id"]
    env.drain()
    with env.factory() as session:
        assert session.scalar(select(func.count()).select_from(SolutionAttempt)) == 2
        assert session.scalar(select(func.count()).select_from(Delivery)) == 2
        attempt = session.scalar(
            select(SolutionAttempt).where(SolutionAttempt.incident_id == first["incident_id"])
        )
        attempt_id = attempt.id
    # A user in another room cannot assess this attempt.
    assert post_talk(env, "Не помогло " + attempt_id, "2", room="room-b").json() == {"accepted": False}
    assert post_talk(env, "Не помогло " + attempt_id, "2").status_code == 200
    env.drain()
    # Late duplicate with a new webhook id cannot fail the next attempt.
    post_talk(env, "Не помогло " + attempt_id, "3")
    with env.factory() as session:
        assert session.get(Incident, first["incident_id"]).failed_attempts == 1


def test_talk_form_and_unknown_delivery_no_automatic_retry(env, monkeypatch):
    setup_talk(monkeypatch)
    response = post_talk(env, "оператор").json()
    assert post_talk(env, "неполная форма", "2").json() == {"accepted": False}
    form = "ФИО: Иванов Иван\nЛогин: ivanov\nКомпьютер: WS-1\nТема: Принтер\nОписание: USB"
    post_talk(env, form, "3")
    env.drain()
    with env.factory() as session:
        ticket = session.scalar(select(Escalation))
        assert ticket.incident_id == response["incident_id"]
        assert ticket.status == "SENT_MOCK"
    calls = []

    def fail(*args):
        calls.append(args)
        raise TimeoutError()

    assert deliver_one(env.factory, get_settings(), SimpleNamespace(send=fail))
    assert not deliver_one(env.factory, get_settings(), SimpleNamespace(send=fail))
    assert len(calls) == 1


def test_late_talk_feedback_does_not_open_new_incident(env, monkeypatch):
    setup_talk(monkeypatch)
    post_talk(env, "Принтер не печатает")
    env.drain()
    with env.factory() as session:
        attempt_id = session.scalar(select(SolutionAttempt.id))
    post_talk(env, "Помогло " + attempt_id, "2")
    post_talk(env, "Помогло " + attempt_id, "3")
    with env.factory() as session:
        assert session.scalar(select(func.count()).select_from(Incident)) == 1
        assert session.scalar(select(Incident.state)) == "RESOLVED"


def test_talk_outbound_signature():
    settings = Settings(
        _env_file=None,
        nextcloud_backend="talk",
        nextcloud_base_url="https://talk.internal",
        nextcloud_bot_secret="s" * 32,
        nextcloud_allowed_rooms=["room-a"],
    )

    def handler(request):
        body = json.loads(request.content)
        nonce = request.headers["X-Nextcloud-Talk-Bot-Random"]
        assert (
            request.headers["X-Nextcloud-Talk-Bot-Signature"]
            == hmac.new(b"s" * 32, (nonce + body["message"]).encode(), hashlib.sha256).hexdigest()
        )
        return httpx.Response(201, json={"ocs": {"meta": {"statuscode": 201}, "data": {"id": 12}}})

    assert TalkClient(settings, httpx.MockTransport(handler)).send("room-a", "Ответ", "uuid") == "12"


def test_smtp_tls_fixed_recipient(monkeypatch):
    calls = []

    class FakeSMTP:
        def __init__(self, *args, **kwargs):
            calls.append("connect")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def ehlo(self):
            calls.append("ehlo")

        def starttls(self, context):
            assert context.check_hostname
            calls.append("tls")

        def login(self, username, password):
            calls.append("login")

        def send_message(self, message, from_addr, to_addrs):
            assert to_addrs == ["support@internal.test"]
            assert message["To"] == "support@internal.test"
            assert "История" in message.get_content()
            calls.append("send")
            return {}

    monkeypatch.setattr("app.services.mail.smtp.smtplib.SMTP", FakeSMTP)
    settings = Settings(
        _env_file=None,
        mail_backend="smtp",
        smtp_host="smtp.internal",
        smtp_from="bot@internal.test",
        support_email="support@internal.test",
        smtp_username="bot",
        smtp_password="test",
    )
    assert (
        SMTPMail(settings).send(
            {
                "to": "attacker@example.org",
                "subject": "Заявка",
                "message_id": "<id@internal>",
                "body": "История",
            }
        )
        == "SENT"
    )
    assert calls == ["connect", "ehlo", "tls", "ehlo", "login", "send"]


def test_smtp_timeout_is_not_retried_by_worker(env, monkeypatch):
    response = env.post("/api/chat", {"message": "оператор"}).json()
    env.post(
        f"/api/incidents/{response['incident_id']}/escalation-details",
        {
            "account_holder_name": "Иванов Иван",
            "account_login": "ivanov",
            "computer_name": "WS-1",
            "subject": "Принтер",
        },
    )
    get_settings().mail_backend = "smtp"
    calls = []

    def fail(self, email):
        calls.append(email)
        raise TimeoutError()

    monkeypatch.setattr(SMTPMail, "send", fail)
    env.drain()
    with env.factory() as session, session.begin():
        ticket = session.scalar(select(Escalation))
        assert ticket.status == "UNKNOWN"
        session.add(Job(incident_id=ticket.incident_id, kind="SEND_MAIL", expected_version=0))
    env.drain()
    assert len(calls) == 1
