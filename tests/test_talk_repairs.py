"""Live regression: read-only knowledge doctor and employee-controlled mail drafts."""

import json
import re
from urllib.parse import parse_qs, urlsplit

import pytest
from sqlalchemy import func, select

from app import talk_worker
from app.core.config import Settings, get_settings
from app.models import Delivery, Escalation, Job, Message, MessageAttachment
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
    for sequence, content in enumerate([QUESTION, "оператор", "1"], 1):
        talk.incoming(content, sequence)
        assert talk.process(knowledge)
    with talk.env.factory() as session:
        original_photos = {row.id: (row.data, row.content_type, row.caption, row.sha256)
                           for row in session.scalars(select(MessageAttachment))}
        source_tags = {row.message_id: (row.account_id, row.provenance)
                       for row in session.scalars(select(Delivery))}
    assert original_photos
    talk.incoming("ФИО: Иванов Иван Иванович\nЛогин: ivanov\nКомпьютер: WS-001\n"
                  "Тема: Сканер не работает\nОписание: " + extra, 4)
    assert talk.process(knowledge)
    with talk.env.factory() as session:
        escalation = session.scalar(select(Escalation))
        assert escalation.status == "DRAFT" and escalation.sent_at is None
        assert escalation.email["subject"] == "[AI-Support] Сканер не работает"
        assert escalation.incident_id not in escalation.email["subject"]
        assert escalation.id not in escalation.email["subject"]
        assert session.scalar(select(func.count()).select_from(Job).where(Job.kind == "SEND_MAIL")) == 0
        drafts = session.scalars(select(MessageAttachment).where(MessageAttachment.content_type == "message/rfc822")).all()
        assert drafts == []
        assert {row.id: (row.data, row.content_type, row.caption, row.sha256)
                for row in session.scalars(select(MessageAttachment))} == original_photos
        deliveries = {row.message_id: row for row in session.scalars(select(Delivery))}
        assert all((deliveries[key].account_id, deliveries[key].provenance) == value
                   for key, value in source_tags.items())
        body = escalation.email["body"]
        for expected in ["Иванов Иван Иванович", "WS-001", extra]:
            assert expected in body
        assert body == extra + "\n\nФИО: Иванов Иван Иванович\nКомпьютер: WS-001\n"
        assert QUESTION not in body and "Исходная проблема:" not in body
        assert "Тема:" not in body and "Описание:" not in body
        assert "Логин:" not in body and "account_login" not in body and "ivanov" not in body
        preview_message = session.scalar(select(Message).where(
            Message.content.startswith("Подготовлено письмо для технической поддержки."),
        ))
        delivery = deliveries[preview_message.id]
        assert delivery.account_id == talk.account_id and delivery.provenance is None
        expected_link = make_mailto(escalation.email, "operator@example.net")
    replies = [text for role, text, photo in talk.snapshot()["messages"] if role == "assistant"]
    assert any("Подготовлено письмо для технической поддержки." in text and "Кому: operator@example.net" in text for text in replies)
    assert not any("Письмо поставлено в очередь" in text for text in replies)
    text = next(text for text in replies if "Подготовлено письмо для технической поддержки." in text)
    rendered = re.search(r"(`{3,})\n(.*?)\n\1", text, re.S)
    assert rendered and rendered.group(2) + "\n" == (
        "Кому: operator@example.net\nТема: [AI-Support] Сканер не работает\n\n" + body
    )
    link = re.search(r"\[[^\]]+\]\((mailto:[^)]+)\)", text)
    assert link is not None
    uri = urlsplit(link.group(1))
    assert uri.path == "operator@example.net"
    fields = parse_qs(uri.query, keep_blank_values=True)
    assert fields["subject"] == ["[AI-Support] Сканер не работает"]
    if expected_link:
        assert link.group(1) == expected_link
        for expected in ["Иванов Иван Иванович", "WS-001", extra]:
            assert expected in fields["body"][0]
        assert "Логин:" not in fields["body"][0] and "ivanov" not in fields["body"][0]
        assert fields["body"][0].replace("\r\n", "\n") == body and QUESTION not in fields["body"][0]
    else:
        assert fields["body"] == [""]
        assert "Письмо длинное: скопируйте текст письма выше" in text


def test_cancel_before_form_does_not_create_any_mail_artifact_or_send_job(talk):
    talk.settings.mail_backend, talk.settings.support_email = "mailto", "operator@example.net"
    for sequence, content in enumerate([QUESTION, "оператор", "1", "Отмена"], 1):
        talk.incoming(content, sequence)
        assert talk.process(Knowledge())
    with talk.env.factory() as session:
        assert session.scalar(select(Escalation)) is None
        assert session.scalar(select(Job).where(Job.kind == "SEND_MAIL")) is None
        assert session.scalar(select(MessageAttachment).where(MessageAttachment.content_type == "message/rfc822")) is None


def test_guided_talk_letter_asks_explicit_topic_then_main_text_without_automatic_report_fallback(talk):
    talk.settings.mail_backend, talk.settings.support_email = "mailto", "operator@example.net"
    for sequence, content in enumerate([
        "оператор", "1", "ФИО: Павел Царан\nКомпьютер: WS-001", "Уменьшить громкость сканера",
    ], 1):
        talk.incoming(content, sequence)
        assert talk.process(Knowledge())
    with talk.env.factory() as session:
        assert session.scalar(select(Escalation)) is None
    state = talk.snapshot()
    assert state["flow"]["escalation"]["intake_requested_field"] == "additional_info"
    description = "После сканирования громкий звуковой сигнал.\nНужно сделать звук тише."
    talk.incoming(description, 5)
    assert talk.process(Knowledge())
    with talk.env.factory() as session:
        escalation = session.scalar(select(Escalation))
        assert escalation.status == "DRAFT" and escalation.sent_at is None
        assert escalation.email["subject"] == "[AI-Support] Уменьшить громкость сканера"
        assert escalation.email["body"] == description + "\n\nФИО: Павел Царан\nКомпьютер: WS-001\n"
        assert "оператор" not in escalation.email["body"] and "Исходная проблема:" not in escalation.email["body"]
        assert "Тема:" not in escalation.email["body"] and "Описание:" not in escalation.email["body"]
        assert session.scalar(select(MessageAttachment)) is None
        assert session.scalar(select(Job).where(Job.kind == "SEND_MAIL")) is None
        preview = session.scalar(select(Message.content).where(Message.content.startswith("Подготовлено письмо")))
        assert "своего почтового ящика" in preview
        link = re.search(r"\[Открыть письмо в почте\]\((mailto:[^)]+)\)", preview)
        assert link is not None
        parsed = urlsplit(link.group(1))
        fields = parse_qs(parsed.query)
        assert parsed.path == "operator@example.net" and fields["subject"] == [escalation.email["subject"]]
        assert fields["body"][0].replace("\r\n", "\n") == escalation.email["body"]


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
