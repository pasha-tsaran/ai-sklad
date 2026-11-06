"""Live regression: read-only knowledge doctor and employee-controlled mail drafts."""

import json
import re
from email import policy
from email.parser import BytesParser
from urllib.parse import parse_qs, urlsplit

import pytest
from sqlalchemy import func, select

from app import talk_worker
from app.core.config import Settings, get_settings
from app.models import Delivery, Escalation, Job, MessageAttachment
from app.services.rag.talk import TalkKnowledge
from app.services.mail.draft import make_mailto
from app.workers.handlers import run_one
from tests.test_talk_dialogue import QUESTION, Knowledge, talk as talk
from tests.test_workflow import DETAILS


@pytest.mark.parametrize("status,exit_code", [("ok", 0), ("error", 1)])
def test_knowledge_doctor_never_runs_database_polling_model_or_delivery(monkeypatch, capsys, status, exit_code):
    result = {"status": status, "read_only": True, "instruction_found": status == "ok"}
    monkeypatch.setattr(talk_worker, "get_settings", lambda: Settings(_env_file=None))
    monkeypatch.setattr(TalkKnowledge, "check", lambda self: result)
    monkeypatch.setattr(talk_worker, "get_engine", lambda: pytest.fail("Doctor must not open a database"))
    monkeypatch.setattr(talk_worker, "run", lambda *a, **kw: pytest.fail("Doctor must not poll or send"))
    assert talk_worker.main(["--check-knowledge"]) == exit_code
    assert json.loads(capsys.readouterr().out) == result


def test_mailto_configuration_needs_only_real_configured_recipient_and_no_smtp_credentials():
    settings = Settings(_env_file=None, mail_backend="mailto", support_email="operator@example.net")
    assert settings.smtp_host == "" and settings.smtp_password.get_secret_value() == ""
    for recipient in ["support@example.invalid", "support@EXAMPLE.INVALID", "one@example.net,two@example.net",
                      "one@example.net\r\nBcc: bad@example.net"]:
        with pytest.raises(ValueError):
            Settings(_env_file=None, mail_backend="mailto", support_email=recipient)
    with pytest.raises(ValueError, match="account-mode"):
        Settings(_env_file=None, mail_backend="mailto", support_email="operator@example.net", nextcloud_backend="talk",
                 nextcloud_base_url="https://cloud.example.net", nextcloud_bot_secret="x" * 40,
                 nextcloud_allowed_rooms=["roomone"])


@pytest.mark.parametrize("extra", ["Проблема со сканером", "Все введённые сведения: " + "я" * 1600],
                         ids=["short-fields", "long-fields"])
def test_complete_private_form_creates_exact_mail_draft_and_never_queues_smtp(talk, extra):
    talk.settings.mail_backend, talk.settings.support_email = "mailto", "operator@example.net"
    knowledge = Knowledge()
    for sequence, content in enumerate([
        QUESTION, "оператор", "1",
        "ФИО: Иванов Иван Иванович\nЛогин: ivanov\nКомпьютер: WS-001\n"
        "Тема: Сканер не работает\nОписание: " + extra,
    ], 1):
        talk.incoming(content, sequence)
        assert talk.process(knowledge)
    with talk.env.factory() as session:
        escalation = session.scalar(select(Escalation))
        assert escalation.status == "DRAFT" and escalation.sent_at is None
        assert session.scalar(select(func.count()).select_from(Job).where(Job.kind == "SEND_MAIL")) == 0
        drafts = session.scalars(select(MessageAttachment).where(MessageAttachment.content_type == "message/rfc822")).all()
        assert len(drafts) == 1
        parsed = BytesParser(policy=policy.default).parsebytes(drafts[0].data)
        assert parsed["To"] == "operator@example.net" and parsed["X-Unsent"] == "1"
        assert parsed["From"] is None and "Сканер не работает" in str(parsed["Subject"])
        body = parsed.get_content()
        for expected in ["Иванов Иван Иванович", "ivanov", "WS-001", extra, QUESTION]:
            assert expected in body
        delivery = session.scalar(select(Delivery).where(Delivery.message_id == drafts[0].message_id))
        assert delivery.account_id == talk.account_id and delivery.provenance is None
        expected_link = make_mailto(escalation.email, "operator@example.net")
    replies = [text for role, text, photo in talk.snapshot()["messages"] if role == "assistant"]
    assert any("Подготовлено письмо для технической поддержки." in text and "Кому: operator@example.net" in text for text in replies)
    assert not any("Письмо поставлено в очередь" in text for text in replies)
    text = next(text for text in replies if "Подготовлено письмо для технической поддержки." in text)
    link = re.search(r"\[Открыть письмо в почте\]\((mailto:[^)]+)\)", text)
    if expected_link:
        assert link is not None
        uri = urlsplit(link.group(1))
        assert uri.path == "operator@example.net"
        fields = parse_qs(uri.query)
        for expected in ["Иванов Иван Иванович", "ivanov", "WS-001", extra]:
            assert expected in fields["body"][0]
        assert QUESTION in fields["body"][0]
    else:
        assert link is None  # Long form stays complete in EML instead of being truncated into a URI.


def test_cancel_before_form_does_not_create_any_mail_artifact_or_send_job(talk):
    talk.settings.mail_backend, talk.settings.support_email = "mailto", "operator@example.net"
    for sequence, content in enumerate([QUESTION, "оператор", "1", "Отмена"], 1):
        talk.incoming(content, sequence)
        assert talk.process(Knowledge())
    with talk.env.factory() as session:
        assert session.scalar(select(Escalation)) is None
        assert session.scalar(select(Job).where(Job.kind == "SEND_MAIL")) is None
        assert session.scalar(select(MessageAttachment).where(MessageAttachment.content_type == "message/rfc822")) is None


def test_switch_to_drafts_pauses_an_old_send_mail_job_without_model_or_mail_calls(env, monkeypatch):
    response = env.post("/api/chat", {"message": "оператор"})
    incident_id = response.json()["incident_id"]
    env.drain()
    assert env.post(f"/api/incidents/{incident_id}/escalation-details", DETAILS).status_code == 202
    monkeypatch.setenv("MAIL_BACKEND", "mailto")
    monkeypatch.setenv("SUPPORT_EMAIL", "operator@example.net")
    get_settings.cache_clear()
    assert run_one(env.factory, object(), get_settings()) is False
    with env.factory() as session:
        queued = session.scalar(select(Job).where(Job.kind == "SEND_MAIL"))
        assert queued.status == "PENDING" and queued.tries == 0
        assert session.scalar(select(Escalation)).status == "QUEUED"
