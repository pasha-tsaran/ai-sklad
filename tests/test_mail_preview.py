"""The visible letter is literal, complete, and separate from approved instruction photos."""

import copy
import re
from email import policy
from email.parser import BytesParser
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit

import pytest
from sqlalchemy import select

from app.core.config import get_settings
from app.models import Delivery, Escalation, Job, Message, MessageAttachment
from app.services.mail.draft import make_eml
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
    draft = BytesParser(policy=policy.default).parsebytes(make_eml(email, RECIPIENT))
    assert str(draft["To"]) == RECIPIENT and str(draft["Subject"]) == email["subject"]
    assert draft.get_content().replace("\r\n", "\n") == email["body"]
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
    eml = make_eml(email, RECIPIENT)
    attachment = SimpleNamespace(filename="support-request.eml", content_type="message/rfc822", data=eml,
                                 caption="Письмо для технической поддержки")
    preview = "Подготовлено письмо для технической поддержки.\n\n" + letter_preview(email, RECIPIENT)
    with factory() as session, session.begin():
        messenger = RoutedMessenger()
        messenger.send_blocks(session, bound.incident, [("Шаг 1. Откройте «Операции».", image)], SOURCE)
        messenger.send_blocks(session, bound.incident, [(preview, attachment)])
    with factory() as session:
        messages = session.scalars(select(Message).where(Message.incident_id == bound.incident)
                                   .order_by(Message.sequence)).all()
        photos = {row.message_id: row for row in session.scalars(select(MessageAttachment))}
        deliveries = {row.message_id: row for row in session.scalars(select(Delivery))}
        assert [message.content for message in messages] == ["Шаг 1. Откройте «Операции».", "", preview, ""]
        assert photos[messages[1].id].data == image.data and photos[messages[1].id].caption == image.caption
        assert photos[messages[3].id].data == eml and photos[messages[3].id].content_type == "message/rfc822"
        assert deliveries[messages[0].id].provenance == deliveries[messages[1].id].provenance == SOURCE
        assert deliveries[messages[2].id].provenance is deliveries[messages[3].id].provenance is None
        assert all(delivery.account_id == bound.account for delivery in deliveries.values())


@pytest.mark.parametrize("long_body", [False, True], ids=["clickable-full-letter", "long-eml-fallback"])
def test_real_ticket_preview_and_eml_match_stored_letter_without_shortening_mailto(env, monkeypatch, long_body):
    configured = get_settings()
    configured.mail_backend, configured.support_email = "mailto", RECIPIENT

    def no_smtp(*args, **kwargs):
        raise AssertionError("Local draft must not use SMTP")

    monkeypatch.setattr("app.services.mail.smtp.SMTPMail.send", no_smtp)
    request = env.post("/api/chat", {"message": "оператор"}).json()
    details = {"account_holder_name": "Иванов Иван", "account_login": "ivanov", "computer_name": "WS-001",
               "subject": "Настройка сканера", "additional_info": "Подробное описание. " * 80 if long_body else "Не читает код."}
    response = env.post(f"/api/incidents/{request['incident_id']}/escalation-details", details)
    assert response.status_code == 202 and response.json()["draft_prepared"] and not response.json()["mail_sent"]
    with env.factory() as session:
        escalation = session.scalar(select(Escalation))
        assert escalation.status == "DRAFT" and escalation.email["to"] == RECIPIENT
        body, subject = escalation.email["body"], escalation.email["subject"]
        attachment = session.scalar(select(MessageAttachment).where(MessageAttachment.content_type == "message/rfc822"))
        assert attachment is not None
        eml = BytesParser(policy=policy.default).parsebytes(attachment.data)
        assert eml["X-Unsent"] == "1" and str(eml["To"]) == RECIPIENT and str(eml["Subject"]) == subject
        assert eml.get_content().replace("\r\n", "\n") == body
        replies = session.scalars(select(Message.content).where(
            Message.incident_id == request["incident_id"], Message.role == "assistant",
        ).order_by(Message.sequence)).all()
        preview = next(text for text in replies if text.startswith("Подготовлено письмо для технической поддержки."))
        rendered = re.search(r"(`{3,})\n(.*?)\n\1", preview, re.S)
        assert rendered and rendered.group(2) + "\n" == f"Кому: {RECIPIENT}\nТема: {subject}\n\n{body}"
        link = re.search(r"\[[^\]]+\]\((mailto:[^)]+)\)", preview)
        if long_body:
            assert link is None and details["additional_info"].strip() in rendered.group(2)
        else:
            assert link is not None
            parsed = urlsplit(link.group(1))
            assert parsed.path == RECIPIENT
            query = parse_qs(parsed.query)
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
