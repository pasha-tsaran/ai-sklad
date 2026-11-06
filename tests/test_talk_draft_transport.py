"""A local email draft is shared as immutable private EML, never sent as email."""

import hashlib
import json
from email import policy
from email.parser import BytesParser
from types import SimpleNamespace
from urllib.parse import parse_qs

import httpx
import pytest

from app.services.mail.draft import make_eml
from app.services.nextcloud.account import AccountClient, AccountError
from tests.test_talk_account import EMPLOYEE, ROOM, SERVICE, Server, configuration, participant


def draft():
    raw = make_eml({"subject": "Настройка сканера ШК", "body": (
        "ФИО: Иванов Иван\nЛогин: warehouse\nКомпьютер: WS-001\n"
        "Тема: Сканер не читает код\nОписание: Уже нажал «ОК».\n"
    )}, "tp@collant.ru")
    return SimpleNamespace(data=raw, filename="../../support-request.eml", caption="Черновик письма в поддержку",
                           content_type="message/rfc822", sha256=hashlib.sha256(raw).hexdigest())


def test_eml_upload_preserves_exact_draft_bytes_and_shares_only_with_expected_employee():
    server = Server()
    original = draft()
    assert server.client().send_file(ROOM, "", "saved-draft", original, expected_employee=EMPLOYEE) == "501"
    writes = [request for request in server.requests if request.method != "GET"]
    assert [request.method for request in writes] == ["MKCOL", "PUT", "POST"]
    uploaded = writes[1]
    assert uploaded.content == original.data and uploaded.headers["Content-Type"] == "message/rfc822"
    assert f"/dav/files/{SERVICE}/AI-Support-" in uploaded.url.path
    assert uploaded.url.path.endswith(".eml") and "/../" not in uploaded.url.path
    filename = uploaded.url.path.rsplit("/", 1)[1]
    assert filename == "support-request-" + hashlib.sha256(b"saved-draft").hexdigest() + ".eml"
    shared = parse_qs(writes[2].content.decode())
    assert shared["shareType"] == ["10"] and shared["shareWith"] == [ROOM]
    assert shared["path"][0].endswith("/" + filename)
    assert shared["referenceId"] == [hashlib.sha256(b"saved-draft").hexdigest()]
    assert json.loads(shared["talkMetaData"][0]) == {"caption": ""}
    parsed = BytesParser(policy=policy.default).parsebytes(uploaded.content)
    assert parsed["X-Unsent"] == "1" and parsed["To"] == "tp@collant.ru" and parsed["From"] is None
    roster_checks = [request for request in server.requests if request.url.path.endswith("/participants")]
    assert len(roster_checks) == 2


@pytest.mark.parametrize("changed", ["hash", "mime", "extension", "oversize", "empty", "mutable"])
def test_invalid_or_changed_draft_snapshot_is_rejected_before_any_write(changed):
    server = Server()
    attachment = draft()
    if changed == "hash":
        attachment.data += b"tampering"
    elif changed == "mime":
        attachment.content_type = "text/html"
    elif changed == "extension":
        attachment.filename = "request.eml.exe"
    elif changed == "oversize":
        attachment.data = b"x" * 2_000_001
        attachment.sha256 = hashlib.sha256(attachment.data).hexdigest()
    elif changed == "empty":
        attachment.data = b""
    else:
        attachment.data = bytearray(attachment.data)
    with pytest.raises(AccountError, match="invalid_mail_draft_snapshot"):
        server.client().send_file(ROOM, "", "id", attachment, expected_employee=EMPLOYEE)
    assert not any(request.method != "GET" for request in server.requests)


@pytest.mark.parametrize("changed", ["group", "public", "extra_member", "wrong_employee", "remote"])
def test_draft_cannot_be_uploaded_or_disclosed_outside_exact_local_personal_chat(changed):
    server = Server()
    if changed in {"group", "public"}:
        server.conversations[ROOM]["type"] = 2 if changed == "group" else 3
    elif changed == "extra_member":
        server.participants[ROOM].append(participant("another-employee"))
    elif changed == "wrong_employee":
        server.participants[ROOM][1] = participant("another-employee")
    else:
        server.conversations[ROOM]["remoteServer"] = "elsewhere.example"
    with pytest.raises(AccountError):
        server.client().send_file(ROOM, "", "id", draft(), expected_employee=EMPLOYEE)
    assert not any(request.method != "GET" for request in server.requests)


def test_draft_revalidates_employee_after_upload_before_room_share():
    server = Server()
    server.after_upload = lambda: server.participants[ROOM].__setitem__(1, participant("another-employee"))
    with pytest.raises(AccountError, match="room_employee_changed"):
        server.client().send_file(ROOM, "", "id", draft(), expected_employee=EMPLOYEE)
    writes = [request for request in server.requests if request.method != "GET"]
    assert [request.method for request in writes] == ["MKCOL", "PUT"]


def test_ambiguous_share_post_is_not_retried_and_never_contacts_smtp():
    server = Server()

    def handler(request):
        if request.url.path.endswith("/shares"):
            server.requests.append(request)
            raise httpx.ReadTimeout("Ambiguous room share", request=request)
        return server.request(request)

    client = AccountClient(configuration(), httpx.MockTransport(handler))
    with pytest.raises(httpx.ReadTimeout):
        client.send_file(ROOM, "", "id", draft(), expected_employee=EMPLOYEE)
    assert len([request for request in server.requests if request.method == "POST"]) == 1
    assert all(request.url.host == "nextcloud.example" and request.url.scheme == "https"
               for request in server.requests)
