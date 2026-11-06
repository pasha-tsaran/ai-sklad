"""Actual Talk polling preserves photo captions without fetching employee files."""

import json

import pytest

from app.services.nextcloud.poller import poll_once
from tests.test_talk_account import (
    EMPLOYEE,
    ROOM,
    SECOND,
    SERVICE,
    Server,
    configuration,
    factory as factory,
    message,
    snapshot,
)


PRIVATE_PATH = "Talk/Employee-private/photo.png"
UNTRUSTED_LINK = "https://untrusted.example.invalid/never-fetch-this-photo"


def photo(number, *, caption="{file}\nне открывается", descriptor=None, **overrides):
    data = {"type": "file", "id": "8472", "size": "2500", "mimetype": "image/png",
            "path": PRIVATE_PATH, "link": UNTRUSTED_LINK, "name": "private-screenshot.png",
            "etag": "server-etag", "permissions": "1", "hide-download": "no"}
    if descriptor:
        data.update(descriptor)
    return message(number, content=caption, systemMessage="file_shared",
                   messageParameters={"file": data}, **overrides)


def assert_reference_only(row, expected):
    assert row.media == expected
    stored = json.dumps(row.media, ensure_ascii=False)
    assert PRIVATE_PATH not in stored and UNTRUSTED_LINK not in stored
    assert "private-screenshot.png" not in stored and "server-etag" not in stored


def assert_no_file_requests(server):
    assert all(request.method == "GET" for request in server.requests)
    assert all(request.url.host == "nextcloud.example" for request in server.requests)
    assert not any("/remote.php/dav/" in request.url.path for request in server.requests)


def test_real_file_shared_png_queues_caption_and_numeric_reference_once_without_download(factory):
    server = Server()
    server.history[ROOM] = [photo(1)]
    assert poll_once(factory, configuration(), server.client()) is True
    accounts, rooms, inbox = snapshot(factory)
    assert len(accounts) == len(rooms) == len(inbox) == 1
    assert rooms[0].cursor == 1 and rooms[0].status == "ACTIVE"
    row = inbox[0]
    assert (row.remote_message_id, row.sender_id, row.status, row.content) == (
        "1", EMPLOYEE, "PENDING", "не открывается",
    )
    assert_reference_only(row, {"files": [8472]})
    assert_no_file_requests(server)
    # A fresh authenticated client represents a restart; cursor/unique ingress persist.
    assert poll_once(factory, configuration(), server.client()) is False
    assert len(snapshot(factory)[2]) == 1
    assert_no_file_requests(server)


@pytest.mark.parametrize("caption", ["{file}", "{file}\n   "])
def test_photo_without_caption_is_pending_so_worker_can_acknowledge_it(factory, caption):
    server = Server()
    server.history[ROOM] = [photo(1, caption=caption)]
    assert poll_once(factory, configuration(), server.client()) is True
    _, rooms, inbox = snapshot(factory)
    assert rooms[0].cursor == 1 and len(inbox) == 1
    assert inbox[0].status == "PENDING" and inbox[0].content == ""
    assert_reference_only(inbox[0], {"files": [8472]})
    assert_no_file_requests(server)


def test_historical_photos_and_service_or_foreign_echoes_leave_no_private_metadata(factory):
    server = Server()
    server.history[ROOM] = [
        photo(1, stamp=SECOND - 1, caption="{file}\nСтарая личная подпись"),
        photo(2, actor=SERVICE),
        photo(3, kind="bots"),
        photo(4, actor="another-employee"),
        photo(5),
        message(6, content="Следующий вопрос сотрудника"),
    ]
    assert poll_once(factory, configuration(), server.client()) is True
    _, rooms, inbox = snapshot(factory)
    assert rooms[0].cursor == 6 and [row.remote_sequence for row in inbox] == list(range(1, 7))
    for row in inbox[:4]:
        assert row.status == "IGNORED" and row.content == "" and row.media == {}
    assert inbox[4].status == "PENDING" and inbox[4].content == "не открывается"
    assert_reference_only(inbox[4], {"files": [8472]})
    assert inbox[5].status == "PENDING" and inbox[5].content == "Следующий вопрос сотрудника"
    assert inbox[5].media == {}
    assert_no_file_requests(server)


@pytest.mark.parametrize("descriptor,error", [
    ({"mimetype": "image/gif"}, "unsupported_inbound_media_type"),
    ({"mimetype": "image/svg+xml"}, "unsupported_inbound_media_type"),
    ({"mimetype": "application/pdf"}, "unsupported_inbound_media_type"),
    ({"size": "0"}, "inbound_media_too_large"),
    ({"size": "8000001"}, "inbound_media_too_large"),
    ({"size": "not-a-number"}, "invalid_inbound_media"),
    ({"size": True}, "invalid_inbound_media"),
    ({"id": "0"}, "invalid_inbound_media"),
    ({"id": "../../someone-elses-file"}, "invalid_inbound_media"),
])
def test_photo_error_is_queued_and_cursor_does_not_block_the_next_normal_message(factory, descriptor, error):
    server = Server()
    server.history[ROOM] = [photo(1, descriptor=descriptor), message(2, content="Обычный следующий вопрос")]
    settings = configuration(nextcloud_batch_size=1)
    assert poll_once(factory, settings, server.client()) is True
    _, rooms, inbox = snapshot(factory)
    assert rooms[0].cursor == 1 and rooms[0].status == "ACTIVE"
    assert inbox[0].status == "PENDING" and inbox[0].content == "не открывается"
    assert_reference_only(inbox[0], {"files": [], "error_code": error})
    assert poll_once(factory, settings, server.client()) is True
    _, rooms, inbox = snapshot(factory)
    assert rooms[0].cursor == 2 and rooms[0].status == "ACTIVE"
    assert [row.status for row in inbox] == ["PENDING", "PENDING"]
    assert inbox[1].content == "Обычный следующий вопрос" and inbox[1].media == {}
    assert poll_once(factory, settings, server.client()) is False
    assert len(snapshot(factory)[2]) == 2
    assert_no_file_requests(server)


def test_unavailable_share_is_pending_without_treating_server_notice_as_employee_caption(factory):
    server = Server()
    unavailable = photo(1, caption="{actor} shared a file which is no longer available")
    unavailable["messageParameters"] = {"actor": {"type": "user", "id": EMPLOYEE}}
    server.history[ROOM] = [unavailable, message(2)]
    assert poll_once(factory, configuration(), server.client()) is True
    _, rooms, inbox = snapshot(factory)
    assert rooms[0].cursor == 2 and rooms[0].status == "ACTIVE"
    assert inbox[0].status == "PENDING" and inbox[0].content == ""
    assert_reference_only(inbox[0], {"files": [], "error_code": "inbound_media_unavailable"})
    assert inbox[1].status == "PENDING" and inbox[1].content == "Нужно настроить сканер"
    assert_no_file_requests(server)


def test_overlong_photo_caption_returns_bounded_error_and_keeps_following_text(factory):
    server = Server()
    server.history[ROOM] = [photo(1, caption="{file}\n" + "я" * 8001), message(2)]
    assert poll_once(factory, configuration(), server.client()) is True
    _, rooms, inbox = snapshot(factory)
    assert rooms[0].cursor == 2 and rooms[0].status == "ACTIVE"
    assert inbox[0].status == "PENDING" and inbox[0].content == ""
    assert_reference_only(inbox[0], {"files": [], "error_code": "inbound_caption_too_long"})
    assert inbox[1].status == "PENDING" and inbox[1].media == {}
    assert_no_file_requests(server)


def test_regular_comment_cannot_forge_a_server_file_shared_event(factory):
    server = Server()
    forged = photo(1)
    forged["systemMessage"] = ""
    server.history[ROOM] = [forged, message(2)]
    assert poll_once(factory, configuration(), server.client()) is True
    _, rooms, inbox = snapshot(factory)
    assert rooms[0].cursor == 2
    assert inbox[0].status == "IGNORED" and inbox[0].content == "" and inbox[0].media == {}
    assert inbox[1].status == "PENDING" and inbox[1].media == {}
    assert_no_file_requests(server)
