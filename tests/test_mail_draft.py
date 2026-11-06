"""Local email hand-off keeps routing fixed and preserves user-entered ticket text."""

import smtplib
import socket
from email import policy
from email.parser import BytesParser
from urllib.parse import parse_qs, unquote, urlsplit

import pytest

from app.services.mail.draft import MAX_MAILTO_CHARS, make_eml, make_mailto, validate_recipient

RECIPIENT = "tp@collant.ru"
SUBJECT = '[AI-Support 123] Сканер "ШК" — настройка № 1'
BODY = ("ФИО: Иванов Иван \"Иванович\"\nЛогин: user+warehouse\nКомпьютер: WS-001\n"
        "Тема: Сканер не читает штрихкоды\nОписание: Нажал «ОК» & #1? Нет результата = 100%.\n")


def ticket(**changes):
    return {"subject": SUBJECT, "body": BODY} | changes


def normalize(value):
    return value.replace("\r\n", "\n").replace("\r", "\n")


def test_mailto_keeps_complete_utf8_fields_and_encodes_reserved_query_characters():
    uri = make_mailto(ticket(), RECIPIENT)
    assert uri and len(uri) <= MAX_MAILTO_CHARS
    parsed = urlsplit(uri)
    assert parsed.scheme == "mailto" and unquote(parsed.path) == RECIPIENT and parsed.fragment == ""
    query = parse_qs(parsed.query, strict_parsing=True)
    assert query == {"subject": [SUBJECT], "body": [BODY.replace("\n", "\r\n")]}
    assert "%D0" in uri and "%0D%0A" in uri and "%26" in uri and "%23" in uri and "+" not in parsed.query


def test_only_configured_recipient_is_used_and_no_sender_is_selected():
    hostile = ticket(to="other@example.com", cc="extra@example.com", bcc="hidden@example.com",
                     sender="attacker@example.com", **{"from": "attacker@example.com"})
    uri = make_mailto(hostile, RECIPIENT)
    assert set(parse_qs(urlsplit(uri).query)) == {"subject", "body"}
    draft = BytesParser(policy=policy.default).parsebytes(make_eml(hostile, RECIPIENT))
    assert str(draft["To"]) == RECIPIENT and draft["X-Unsent"] == "1"
    assert draft["From"] is None and draft["Cc"] is None and draft["Bcc"] is None
    assert str(draft["Subject"]) == SUBJECT and normalize(draft.get_content()) == BODY
    assert not draft.is_multipart() and draft.get_content_type() == "text/plain"


def test_body_header_looking_text_stays_inside_body_and_does_not_add_recipients():
    body = BODY + "\r\nBcc: attacker@example.com\r\nFrom: user@example.com\r\n\r\n--boundary\r\n"
    uri = make_mailto(ticket(body=body), RECIPIENT)
    assert set(parse_qs(urlsplit(uri).query)) == {"subject", "body"}
    draft = BytesParser(policy=policy.default).parsebytes(make_eml(ticket(body=body), RECIPIENT))
    assert draft["Bcc"] is None and draft["From"] is None
    assert normalize(draft.get_content()) == normalize(body)


@pytest.mark.parametrize("recipient", ["tp@example.com\r\nBcc: attacker@example.com", "tp@example.com\n",
                                      "Two <tp@example.com>", "a@example.com,b@example.com",
                                      "a@example.com;b@example.com", "tp@example.com?bcc=attacker@example.com",
                                      "tp@example.com#fragment", " tp@example.com", "tp@example.com ",
                                      "@example.com", "user@", "user@local", ".user@example.com",
                                      "user..two@example.com", "user@example..com", "user@-example.com",
                                      "user@example-.com", "a" * 65 + "@example.com", "user\x00@example.com",
                                      "пользователь@example.com"],
                         ids=["crlf", "newline", "display-name", "comma-list", "semicolon-list", "query",
                              "fragment", "leading-space", "trailing-space", "empty-local", "empty-domain",
                              "local-domain", "leading-dot", "double-local-dot", "double-domain-dot",
                              "leading-domain-hyphen", "trailing-domain-hyphen", "long-local", "null", "unicode-local"])
def test_invalid_or_multiple_mailboxes_cannot_become_draft_headers(recipient):
    for operation in (make_mailto, make_eml):
        with pytest.raises(ValueError, match="InvalidDraftRecipient"):
            operation(ticket(), recipient)
    with pytest.raises(ValueError, match="InvalidDraftRecipient"):
        validate_recipient(recipient)


def test_plus_address_and_case_are_preserved_without_splitting_uri_recipient():
    recipient = "Warehouse+Help@Example.COM"
    assert validate_recipient(recipient) == recipient
    uri = make_mailto(ticket(), recipient)
    assert unquote(urlsplit(uri).path) == recipient and "%2B" in uri
    draft = BytesParser(policy=policy.default).parsebytes(make_eml(ticket(), recipient))
    assert str(draft["To"]) == recipient


@pytest.mark.parametrize("subject", ["Тема\r\nBcc: attacker@example.com", "Тема\nСкрытый заголовок",
                                    "Тема\tBcc:", "Тема\x00", "Тема\x7f", "Тема\x85"])
def test_header_controls_are_rejected_instead_of_folding_or_sanitizing_subject(subject):
    for operation in (make_mailto, make_eml):
        with pytest.raises(ValueError, match="InvalidDraftContent"):
            operation(ticket(subject=subject), RECIPIENT)


@pytest.mark.parametrize("control", ["\x00", "\x08", "\x0b", "\x0c", "\x1b", "\x7f", "\x85"])
def test_unsafe_body_controls_are_rejected_without_silently_changing_the_request(control):
    for operation in (make_mailto, make_eml):
        with pytest.raises(ValueError, match="InvalidDraftContent"):
            operation(ticket(body=BODY + control), RECIPIENT)


def test_tabs_and_mixed_real_linebreaks_remain_plain_body_text():
    body = "ФИО:\tИванов\rЛогин: user\r\nКомпьютер: WS-1\n"
    uri = make_mailto(ticket(body=body), RECIPIENT)
    assert parse_qs(urlsplit(uri).query)["body"] == [normalize(body).replace("\n", "\r\n")]
    draft = BytesParser(policy=policy.default).parsebytes(make_eml(ticket(body=body), RECIPIENT))
    assert normalize(draft.get_content()) == normalize(body)


def test_mailto_exact_length_boundary_falls_back_without_truncation():
    minimal = {"subject": "S", "body": ""}
    prefix_length = len(make_mailto(minimal, RECIPIENT))
    content = "a" * (MAX_MAILTO_CHARS - prefix_length)
    assert len(make_mailto(minimal | {"body": content}, RECIPIENT)) == MAX_MAILTO_CHARS
    assert make_mailto(minimal | {"body": content + "a"}, RECIPIENT) is None
    draft = BytesParser(policy=policy.default).parsebytes(make_eml(minimal | {"body": content + "a"}, RECIPIENT))
    assert normalize(draft.get_content()) == content + "a\n"


def test_long_transcript_is_complete_in_eml_and_never_partially_inserted_into_mailto():
    body = BODY + ("Пользователь: Настройка не помогла.\nПомощник: Проверенный шаг.\n" * 5000) + "КОНЕЦ ИСТОРИИ\n"
    assert make_mailto(ticket(body=body), RECIPIENT) is None
    raw = make_eml(ticket(body=body), RECIPIENT)
    assert b"X-Unsent: 1\r\n" in raw and b"MIME-Version: 1.0\r\n" in raw
    draft = BytesParser(policy=policy.default).parsebytes(raw)
    assert normalize(draft.get_content()) == body and str(draft["Subject"]) == SUBJECT


def test_draft_preparation_has_no_smtp_or_network_side_effects(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Draft preparation must not contact a mail server")

    monkeypatch.setattr(smtplib, "SMTP", forbidden)
    monkeypatch.setattr(smtplib, "SMTP_SSL", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    assert make_mailto(ticket(), RECIPIENT)
    assert make_eml(ticket(), RECIPIENT)
