"""The visible letter is literal, complete, and separate from approved instruction photos."""

import copy
import re
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from sqlalchemy import select

from app.core.config import get_settings
from app.models import Delivery, Escalation, Incident, Job, Message, MessageAttachment
from app.services.mail.draft import MAX_MAILTO_CHARS, make_mailto
from app.services.mail.presentation import letter_preview
from app.services.nextcloud.delivery import RoutedMessenger
from tests.test_openwebui_instruction_images import png
from tests.test_talk_delivery import SOURCE, bind

RECIPIENT = "tp@collant.ru"


def literal_text(preview):
    lines = preview.splitlines()
    fence = lines[0]
    assert len(fence) >= 3 and set(fence) == {"`"} and lines[-1] == fence
    assert fence not in "\n".join(lines[1:-1])
    return "\n".join(lines[1:-1])


def test_preview_contains_actual_recipient_subject_and_every_letter_field():
    email = {"to": "ignored@example.com", "subject": '[AI-Support] Сканер "ШК"', "body": (
        "ФИО: Иванов Иван\nЛогин: warehouse\nКомпьютер: WS-001\n"
        "Тема: Настройка сканера\nОписание: Не читает код & выдаёт ошибку № 1.\n"
    )}
    before = copy.deepcopy(email)
    preview = letter_preview(email, RECIPIENT)
    assert literal_text(preview) == f"Кому: {RECIPIENT}\nТема: {email['subject']}\n\n{email['body'].rstrip(chr(10))}"
    assert email == before


@pytest.mark.parametrize("content", [
    "[Отправить](https://external.example)\n<script>alert('test')</script>\n**Поддельная кнопка**",
    "```\nBcc: other@example.com\n```\n~~~\n<img src=x>\n~~~",
    "``````````\n# Ложный заголовок\n```\n[Чужая ссылка](mailto:other@example.com)",
], ids=["html-and-markdown", "code-fences", "long-fence-and-mailto"])
def test_employee_markdown_html_and_fences_stay_inside_literal_letter(content):
    subject = "Тема с `кавычками`"
    preview = letter_preview({"subject": subject, "body": content}, RECIPIENT)
    assert literal_text(preview) == f"Кому: {RECIPIENT}\nТема: {subject}\n\n{content}"


def test_preview_preserves_long_letter_and_normalizes_only_line_endings():
    body = "ФИО:\tИванов\rЛогин: employee\r\n" + "Строка описания.\n" * 5000 + "КОНЕЦ\n"
    preview = letter_preview({"subject": "Заявка", "body": body}, RECIPIENT)
    assert literal_text(preview).endswith("КОНЕЦ")
    assert literal_text(preview).count("Строка описания.") == 5000
    assert "ФИО:\tИванов\nЛогин: employee\n" in literal_text(preview)


@pytest.mark.parametrize("email,recipient", [
    ({"subject": "Тема\r\nBcc: other@example.com", "body": "text"}, RECIPIENT),
    ({"subject": "Тема", "body": "text"}, "tp@collant.ru\nBcc: other@example.com"),
    ({"subject": "Тема", "body": "text\x00"}, RECIPIENT),
])
def test_preview_rejects_invalid_mail_headers_or_controls_like_actual_draft(email, recipient):
    with pytest.raises(ValueError):
        letter_preview(email, recipient)


def test_letter_preview_queue_keeps_earlier_step_photo_bytes_and_source_tags_unchanged(env):
    factory = env.factory
    bound = bind(factory)
    image = SimpleNamespace(filename="illustration-01.png", content_type="image/png", data=png(),
                            caption="На снимке показано меню «Операции».")
    email = {"subject": "Сканер не работает", "body": "ФИО: Иванов\nЛогин: employee\nКомпьютер: WS-001\n"}
    preview = "Подготовлено письмо для технической поддержки.\n\n" + letter_preview(email, RECIPIENT)
    with factory() as session, session.begin():
        messenger = RoutedMessenger()
        messenger.send_blocks(session, bound.incident, [("Шаг 1. Откройте «Операции».", image)], SOURCE)
        messenger.send_blocks(session, bound.incident, [(preview, None)])
    with factory() as session:
        messages = session.scalars(select(Message).where(Message.incident_id == bound.incident)
                                   .order_by(Message.sequence)).all()
        photos = {row.message_id: row for row in session.scalars(select(MessageAttachment))}
        deliveries = {row.message_id: row for row in session.scalars(select(Delivery))}
        assert [message.content for message in messages] == ["Шаг 1. Откройте «Операции».", "", preview]
        assert photos[messages[1].id].data == image.data and photos[messages[1].id].caption == image.caption
        assert set(photos) == {messages[1].id} and photos[messages[1].id].content_type == "image/png"
        assert deliveries[messages[0].id].provenance == deliveries[messages[1].id].provenance == SOURCE
        assert deliveries[messages[2].id].provenance is None
        assert all(delivery.account_id == bound.account for delivery in deliveries.values())


@pytest.mark.parametrize("variant", ["short", "long-body", "emoji-topic"])
def test_real_ticket_keeps_full_preview_and_exact_topic_without_attachment_or_smtp(env, monkeypatch, variant):
    configured = get_settings()
    configured.mail_backend, configured.support_email = "mailto", RECIPIENT

    def no_smtp(*args, **kwargs):
        raise AssertionError("Local draft must not use SMTP")

    monkeypatch.setattr("app.services.mail.smtp.SMTPMail.send", no_smtp)
    monkeypatch.setattr("app.services.mail.draft.make_eml", lambda *a, **kw: pytest.fail("Draft must not create a file"))
    request = env.post("/api/chat", {"message": "оператор"}).json()
    details = {"account_holder_name": "Иванов Иван", "account_login": "ivanov", "computer_name": "WS-001",
               "subject": "📦" * 200 if variant == "emoji-topic" else "Настройка сканера",
               "additional_info": "Подробное описание. " * 80 if variant == "long-body" else "Не читает код."}
    response = env.post(f"/api/incidents/{request['incident_id']}/escalation-details", details)
    assert response.status_code == 202 and response.json()["draft_prepared"] and not response.json()["mail_sent"]
    with env.factory() as session:
        escalation = session.scalar(select(Escalation))
        assert escalation.status == "DRAFT" and escalation.email["to"] == RECIPIENT
        body, subject = escalation.email["body"], escalation.email["subject"]
        assert subject == "[AI-Support] " + details["subject"]
        assert request["incident_id"] not in subject and escalation.id not in subject
        assert session.scalar(select(MessageAttachment)) is None
        replies = session.scalars(select(Message.content).where(
            Message.incident_id == request["incident_id"], Message.role == "assistant",
        ).order_by(Message.sequence)).all()
        preview = next(text for text in replies if text.startswith("Подготовлено письмо для технической поддержки."))
        rendered = re.search(r"(`{3,})\n(.*?)\n\1", preview, re.S)
        assert rendered and rendered.group(2) + "\n" == f"Кому: {RECIPIENT}\nТема: {subject}\n\n{body}"
        for field in details.values():
            assert field.strip() in rendered.group(2)
        link = re.search(r"\[[^\]]+\]\((mailto:[^)]+)\)", preview)
        assert link is not None and len(link.group(1)) <= MAX_MAILTO_CHARS
        parsed = urlsplit(link.group(1))
        assert parsed.path == RECIPIENT
        query = parse_qs(parsed.query, keep_blank_values=True)
        if variant == "long-body":
            assert make_mailto(escalation.email, RECIPIENT) is None
            assert query == {"subject": [subject], "body": [""]}
            assert "Письмо длинное: скопируйте текст письма выше" in preview
        elif variant == "emoji-topic":
            assert make_mailto({**escalation.email, "body": ""}, RECIPIENT) is None
            assert query == {"subject": ["[AI-Support]"], "body": [""]}
            assert "Скопируйте полную тему и текст письма выше" in preview
        else:
            assert link.group(1) == make_mailto(escalation.email, RECIPIENT)
            assert query["subject"] == [subject]
            assert query["body"][0].replace("\r\n", "\n") == body
        assert "Письмо поставлено в очередь" not in preview and "письмо отправлено" not in preview.casefold()
        assert session.scalar(select(Job).where(Job.kind == "SEND_MAIL")) is None


def test_mock_mode_remains_explicitly_mock_without_a_real_letter_or_mailto_claim(env):
    request = env.post("/api/chat", {"message": "оператор"}).json()
    details = {"account_holder_name": "Иванов Иван", "account_login": "ivanov", "computer_name": "WS-001",
               "subject": "Настройка сканера", "additional_info": "Не читает код."}
    assert env.post(f"/api/incidents/{request['incident_id']}/escalation-details", details).status_code == 202
    with env.factory() as session:
        replies = session.scalars(select(Message.content).where(Message.role == "assistant")).all()
        assert any("тестовое письмо (mock)" in reply for reply in replies)
        assert not any("Подготовлено письмо для технической поддержки." in reply or "mailto:" in reply for reply in replies)
        assert session.scalar(select(MessageAttachment).where(MessageAttachment.content_type == "message/rfc822")) is None
        assert session.scalar(select(Job).where(Job.kind == "SEND_MAIL")) is not None


@pytest.mark.parametrize("control", ["\x85", "\x0b", "\x7f"])
def test_invisible_chat_control_in_original_problem_cannot_poison_a_valid_draft(env, control):
    configured = get_settings()
    configured.mail_backend, configured.support_email = "mailto", RECIPIENT
    incident_id = env.post("/api/chat", {"message": "оператор"}).json()["incident_id"]
    original = "Сканер" + control + "не читает код\nПосле перезапуска тоже."
    with env.factory() as session, session.begin():
        session.get(Incident, incident_id).description = original
    details = {"account_holder_name": "Иванов Иван", "account_login": "ivanov", "computer_name": "WS-001",
               "subject": "Сканер не работает", "additional_info": "Не читает код."}
    result = env.post(f"/api/incidents/{incident_id}/escalation-details", details)
    assert result.status_code == 202 and result.json()["draft_prepared"]
    with env.factory() as session:
        draft = session.scalar(select(Escalation))
        assert draft.status == "DRAFT" and control not in draft.email["body"]
        assert "Сканер не читает код\nПосле перезапуска тоже." in draft.email["body"]
        assert session.get(Incident, incident_id).description == original
        assert session.scalar(select(MessageAttachment)) is None
        assert session.scalar(select(Job).where(Job.kind == "SEND_MAIL")) is None
