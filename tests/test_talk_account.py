"""Personal-account Talk privacy and durable ingress, without network or real secrets."""

import base64
import copy
import hashlib
import importlib.util
import io
import json
from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs

import httpx
import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.operations import Operations
from PIL import Image
from pydantic import SecretStr
from sqlalchemy import create_engine, event, select

from app.db import session_factory
from app.models import Base
from app.models.talk import TalkAccount, TalkInbox, TalkRoom
from app.services.nextcloud.account import AccountClient, AccountError
from app.services.nextcloud.poller import poll_once

STAMP = datetime(2026, 11, 6, 10, tzinfo=timezone.utc)
SECOND = int(STAMP.timestamp())
SERVICE = "actual-service-uid"
EMPLOYEE = "employee-one"
ROOM = "roomone"


def configuration(**overrides):
    values = dict(nextcloud_backend="account", nextcloud_base_url="https://nextcloud.example/office",
                  nextcloud_username="SkladTech-bot", nextcloud_app_password=SecretStr("test-only-app-password"),
                  nextcloud_allowed_rooms=[], nextcloud_batch_size=100, nextcloud_poll_interval_seconds=3,
                  integration_timeout_seconds=20)
    return SimpleNamespace(**(values | overrides))


def participant(uid, kind="users"):
    return {"actorType": kind, "actorId": uid, "participantType": 3}


def message(number, *, actor=EMPLOYEE, kind="users", stamp=SECOND, content="Нужно настроить сканер", **extras):
    return {"id": number, "token": ROOM, "actorType": kind, "actorId": actor, "timestamp": stamp,
            "messageType": "comment", "systemMessage": "", "message": content, "messageParameters": {}} | extras


class Server:
    def __init__(self):
        self.uid = SERVICE
        self.conversations = {ROOM: {"token": ROOM, "type": 1}}
        self.participants = {ROOM: [participant(SERVICE), participant(EMPLOYEE)]}
        self.history = {ROOM: []}
        self.requests = []
        self.after_upload = None
        self.post_timeout = False
        self.raw_page = None
        self.transport = httpx.MockTransport(self.request)

    def ocs(self, data, *, status=200, headers=None):
        return httpx.Response(status, json={"ocs": {"meta": {"statuscode": status}, "data": data}},
                              headers={"Date": format_datetime(STAMP, usegmt=True)} | (headers or {}))

    def request(self, request):
        self.requests.append(request)
        path = request.url.path.removeprefix("/office")
        if path == "/ocs/v2.php/cloud/user":
            return self.ocs({"id": self.uid})
        if path == "/ocs/v2.php/apps/spreed/api/v4/room":
            return self.ocs(copy.deepcopy(list(self.conversations.values())))
        room_prefix = "/ocs/v2.php/apps/spreed/api/v4/room/"
        if path.startswith(room_prefix):
            token, _, suffix = path.removeprefix(room_prefix).partition("/")
            if token not in self.conversations:
                return httpx.Response(404)
            return self.ocs(copy.deepcopy(self.participants[token] if suffix == "participants"
                                          else self.conversations[token]))
        chat_prefix = "/ocs/v2.php/apps/spreed/api/v1/chat/"
        if path.startswith(chat_prefix):
            token = path.removeprefix(chat_prefix)
            if request.method == "POST":
                if self.post_timeout:
                    raise httpx.ReadTimeout("simulated ambiguous POST", request=request)
                return self.ocs({"id": 901}, status=201)
            if self.raw_page is not None:
                return self.ocs(self.raw_page, headers={"X-Chat-Last-Given": "99"})
            cursor, limit = int(request.url.params["lastKnownMessageId"]), int(request.url.params["limit"])
            page = [item for item in self.history[token] if item["id"] > cursor][:limit]
            if not page:
                return httpx.Response(304, headers={"X-Chat-Last-Given": str(cursor)})
            return self.ocs(copy.deepcopy(page), headers={"X-Chat-Last-Given": str(page[-1]["id"])})
        if path.startswith("/remote.php/dav/files/"):
            if request.method == "PUT" and self.after_upload:
                self.after_upload()
            return httpx.Response(201)
        if path == "/ocs/v2.php/apps/files_sharing/api/v1/shares":
            return self.ocs({"id": 501}, status=201)
        raise AssertionError(f"Unexpected endpoint: {request.method} {path}")

    def client(self, **kwargs):
        return AccountClient(configuration(**kwargs), transport=self.transport)


@pytest.fixture
def factory(tmp_path):
    engine = create_engine("sqlite:///" + str(tmp_path / "account.db"))
    Base.metadata.create_all(engine)
    yield session_factory(engine)
    engine.dispose()


def snapshot(factory):
    with factory() as session:
        accounts = session.scalars(select(TalkAccount)).all()
        rooms = session.scalars(select(TalkRoom)).all()
        inbox = session.scalars(select(TalkInbox).order_by(TalkInbox.remote_sequence)).all()
        return accounts, rooms, inbox


def picture():
    output = io.BytesIO()
    Image.new("RGB", (12, 8), "red").save(output, format="PNG")
    raw = output.getvalue()
    return SimpleNamespace(data=raw, sha256=hashlib.sha256(raw).hexdigest(), content_type="image/png",
                           filename="../../never-use-this-name.png")


def test_actual_uid_and_app_password_authenticate_api_without_webhook_or_admin_commands():
    server = Server()
    client = server.client()
    assert client.identity() == SERVICE
    assert client.server_time == STAMP
    assert client.validate_room(ROOM, expected_employee=EMPLOYEE) == {
        "token": ROOM, "type": 1, "service_id": SERVICE, "employee_id": EMPLOYEE,
    }
    expected = "Basic " + base64.b64encode(b"SkladTech-bot:test-only-app-password").decode()
    assert all(request.headers["Authorization"] == expected for request in server.requests)
    assert all(request.headers["OCS-APIRequest"] == "true" for request in server.requests)
    assert all("test-only-app-password" not in str(request.url) and not request.content for request in server.requests)


@pytest.mark.parametrize("case", ["group", "public", "former", "self", "foreign", "guest", "deleted",
                                 "three", "duplicate", "wrong_service", "wrong_employee", "remote_actor"])
def test_outgoing_text_never_posts_outside_exact_local_service_employee_dm(case):
    server = Server()
    if case in {"group", "public", "former", "self"}:
        server.conversations[ROOM]["type"] = {"group": 2, "public": 3, "former": 6, "self": 7}[case]
    elif case == "foreign":
        server.conversations[ROOM]["remoteServer"] = "elsewhere.example"
    elif case == "guest":
        server.participants[ROOM][1] = participant(EMPLOYEE, "guests")
    elif case == "deleted":
        server.participants[ROOM][1] = participant(EMPLOYEE, "deleted_users")
    elif case == "three":
        server.participants[ROOM].append(participant("another-employee"))
    elif case == "duplicate":
        server.participants[ROOM] = [participant(SERVICE), participant(SERVICE)]
    elif case == "wrong_service":
        server.participants[ROOM][0] = participant("another-account")
    elif case == "wrong_employee":
        server.participants[ROOM][1] = participant("another-employee")
    elif case == "remote_actor":
        server.participants[ROOM][1]["remoteServer"] = "remote.example"
    with pytest.raises(AccountError):
        server.client().send(ROOM, "Проверенный ответ", "local-message", expected_employee=EMPLOYEE)
    assert not any(request.method != "GET" for request in server.requests)


def test_room_allowlist_and_unsafe_tokens_fail_before_network():
    server = Server()
    with pytest.raises(AccountError, match="room_not_allowed"):
        server.client(nextcloud_allowed_rooms=["otherroom"]).send(ROOM, "Ответ", "id")
    for token in ["../../other", "room?query", "https://remote.example", "x"]:
        with pytest.raises(AccountError, match="invalid_room_token"):
            server.client().send(token, "Ответ", "id")
    assert not server.requests


def test_text_reference_id_is_stable_and_ambiguous_post_has_no_automatic_retry():
    server = Server()
    assert server.client().send(ROOM, "Нажмите «ОК».", "saved-id", expected_employee=EMPLOYEE) == "901"
    posted = [request for request in server.requests if request.method == "POST"]
    assert len(posted) == 1 and json.loads(posted[0].content) == {
        "message": "Нажмите «ОК».", "referenceId": hashlib.sha256(b"saved-id").hexdigest(),
    }
    server.requests.clear()
    server.post_timeout = True
    with pytest.raises(httpx.ReadTimeout):
        server.client().send(ROOM, "Ответ", "saved-id", expected_employee=EMPLOYEE)
    assert len([request for request in server.requests if request.method == "POST"]) == 1


def test_photo_upload_uses_actual_uid_and_private_room_share_with_source_caption():
    server = Server()
    attachment = picture()
    assert server.client().send_image(ROOM, "На фото нажмите «ОК».", "saved-photo", attachment,
                                      expected_employee=EMPLOYEE) == "501"
    writes = [request for request in server.requests if request.method != "GET"]
    assert [request.method for request in writes] == ["MKCOL", "PUT", "POST"]
    assert f"/dav/files/{SERVICE}/" in writes[0].url.path
    assert writes[1].content == attachment.data and writes[1].headers["Content-Type"] == "image/png"
    shared = parse_qs(writes[2].content.decode())
    assert shared["shareType"] == ["10"] and shared["shareWith"] == [ROOM]
    assert json.loads(shared["talkMetaData"][0]) == {"caption": "На фото нажмите «ОК»."}
    assert "never-use-this-name" not in shared["path"][0] and ".." not in shared["path"][0]
    assert len([request for request in server.requests if request.url.path.endswith("/participants")]) == 2


def test_photo_rechecks_employee_after_upload_before_disclosing_file_to_room():
    server = Server()
    server.after_upload = lambda: server.participants[ROOM].append(participant("another-employee"))
    with pytest.raises(AccountError, match="room_not_private"):
        server.client().send_image(ROOM, "Фото", "snapshot", picture(), expected_employee=EMPLOYEE)
    assert not any(request.method == "POST" for request in server.requests)
    assert len([request for request in server.requests if request.method == "PUT"]) == 1


@pytest.mark.parametrize("changed", ["bytes", "mime", "oversize"])
def test_only_bounded_immutable_png_snapshots_are_uploaded(changed):
    server = Server()
    attachment = picture()
    if changed == "bytes":
        attachment.data += b"tampered"
    elif changed == "mime":
        attachment.content_type = "image/svg+xml"
    else:
        attachment.data = b"x" * 8_000_001
    with pytest.raises(AccountError, match="invalid_illustration_snapshot"):
        server.client().send_image(ROOM, "Фото", "id", attachment, expected_employee=EMPLOYEE)
    assert not any(request.method != "GET" for request in server.requests)


def test_forward_poll_does_not_change_read_markers_or_follow_remote_redirects():
    server = Server()
    server.history[ROOM] = [message(12)]
    page = server.client().messages(ROOM, 10)
    assert page == {"messages": server.history[ROOM], "cursor": 12}
    params = server.requests[-1].url.params
    assert params["lookIntoFuture"] == "1" and params["timeout"] == "0"
    assert params["setReadMarker"] == "0" and params["markNotificationsAsRead"] == "0"
    calls = []

    def redirect(request):
        calls.append(request)
        return httpx.Response(302, headers={"Location": "https://foreign.example/steal"})

    with pytest.raises(AccountError, match="nextcloud_http_error"):
        AccountClient(configuration(), httpx.MockTransport(redirect)).identity()
    assert len(calls) == 1


def test_api_response_limits_and_auth_errors_do_not_echo_body_or_credentials():
    def huge(request):
        return httpx.Response(200, content=b"x" * 4_000_001)

    with pytest.raises(AccountError, match="response_too_large"):
        AccountClient(configuration(), httpx.MockTransport(huge)).identity()

    def denied(request):
        return httpx.Response(401, text="private upstream error and test-only-app-password")

    with pytest.raises(AccountError) as exc:
        AccountClient(configuration(), httpx.MockTransport(denied)).identity()
    assert str(exc.value) == "access_denied"


def test_activation_ignores_old_unread_history_but_keeps_first_new_question_and_own_echoes(factory):
    server = Server()
    server.history[ROOM] = [message(1, stamp=SECOND - 30, content="Old unread private text"), message(2),
                            message(3, actor=SERVICE), message(4, systemMessage="conversation_created"),
                            message(5, messageParameters={"object": {"type": "file", "link": "https://external"}}),
                            message(6, kind="bots"), message(7, actor="other-employee")]
    assert poll_once(factory, configuration(), server.client()) is True
    accounts, rooms, inbox = snapshot(factory)
    assert len(accounts) == len(rooms) == 1 and accounts[0].username == SERVICE
    assert accounts[0].activated_at.replace(tzinfo=timezone.utc) == STAMP
    assert rooms[0].cursor == 7 and rooms[0].employee_id == EMPLOYEE
    assert [(row.remote_sequence, row.status) for row in inbox] == [
        (1, "IGNORED"), (2, "PENDING"), (3, "IGNORED"), (4, "IGNORED"),
        (5, "IGNORED"), (6, "IGNORED"), (7, "IGNORED"),
    ]
    assert inbox[1].content == "Нужно настроить сканер"
    assert all(not row.content for row in inbox if row.status == "IGNORED")


def test_new_personal_chat_after_activation_does_not_drop_its_first_question(factory):
    server = Server()
    server.conversations.clear()
    assert poll_once(factory, configuration(), server.client()) is False
    server.conversations[ROOM] = {"token": ROOM, "type": 1}
    server.history[ROOM] = [message(1, stamp=SECOND + 1)]
    assert poll_once(factory, configuration(), server.client()) is True
    _, rooms, inbox = snapshot(factory)
    assert rooms[0].cursor == 1 and len(inbox) == 1 and inbox[0].status == "PENDING"


def test_restart_scans_old_pages_then_queues_new_messages_in_numeric_order_once(factory):
    server = Server()
    server.history[ROOM] = [message(1, stamp=SECOND - 10), message(3), message(10, stamp=SECOND + 1)]
    settings = configuration(nextcloud_batch_size=1)
    assert poll_once(factory, settings, server.client()) is False
    assert snapshot(factory)[1][0].cursor == 1
    assert poll_once(factory, settings, server.client()) is True
    assert poll_once(factory, settings, server.client()) is True
    assert poll_once(factory, settings, server.client()) is False
    _, rooms, inbox = snapshot(factory)
    assert rooms[0].cursor == 10 and [row.remote_sequence for row in inbox] == [1, 3, 10]
    assert [row.remote_message_id for row in inbox if row.status == "PENDING"] == ["3", "10"]
    # A replay after a stale cursor cannot insert duplicate work.
    with factory() as session, session.begin():
        session.get(TalkRoom, rooms[0].id).cursor = 0
    assert poll_once(factory, settings, server.client()) is False
    assert len(snapshot(factory)[2]) == 3


def test_cursor_and_inbox_rollback_together_before_retrying_failed_page(factory):
    server = Server()
    server.history[ROOM] = [message(1), message(2)]
    fail = [True]

    def unavailable(session, context, instances):
        if fail[0] and any(isinstance(row, TalkInbox) and row.remote_sequence == 2 for row in session.new):
            fail[0] = False
            raise OSError("Simulated DB failure")

    event.listen(factory.class_, "before_flush", unavailable)
    try:
        assert poll_once(factory, configuration(), server.client()) is False
        _, rooms, inbox = snapshot(factory)
        assert rooms[0].cursor == 0 and not inbox and rooms[0].status == "UNAVAILABLE"
        assert poll_once(factory, configuration(), server.client()) is True
        assert [row.remote_sequence for row in snapshot(factory)[2]] == [1, 2]
    finally:
        event.remove(factory.class_, "before_flush", unavailable)


def test_malformed_page_cannot_advance_cursor_or_hide_an_earlier_message(factory):
    server = Server()
    server.raw_page = [message(2), message(1)]
    assert poll_once(factory, configuration(), server.client()) is False
    _, rooms, inbox = snapshot(factory)
    assert not inbox and rooms[0].cursor == 0
    server.raw_page = None
    server.history[ROOM] = [message(1), message(2)]
    assert poll_once(factory, configuration(), server.client()) is True
    assert [row.remote_sequence for row in snapshot(factory)[2]] == [1, 2]


def test_removed_or_changed_room_is_disabled_without_reassigning_employee(factory):
    server = Server()
    server.history[ROOM] = [message(1)]
    poll_once(factory, configuration(), server.client())
    server.participants[ROOM][1] = participant("another-employee")
    server.history[ROOM].append(message(2, actor="another-employee"))
    assert poll_once(factory, configuration(), server.client()) is False
    _, rooms, inbox = snapshot(factory)
    assert not rooms[0].enabled and rooms[0].employee_id == EMPLOYEE and rooms[0].cursor == 1
    assert len(inbox) == 1 and rooms[0].status == "BLOCKED"
    server.conversations.clear()
    poll_once(factory, configuration(), server.client())
    assert snapshot(factory)[1][0].status == "NOT_PRESENT"


@pytest.mark.parametrize("kind", [2, 3, 6])
def test_discovery_never_reads_history_or_creates_work_for_nonpersonal_rooms(factory, kind):
    server = Server()
    server.conversations[ROOM]["type"] = kind
    server.history[ROOM] = [message(1)]
    assert poll_once(factory, configuration(), server.client()) is False
    assert not snapshot(factory)[1] and not snapshot(factory)[2]
    assert not any("/chat/" in request.url.path for request in server.requests)


def test_changing_authenticated_account_disables_old_queues_without_forwarding_them(factory):
    server = Server()
    server.history[ROOM] = [message(1)]
    assert poll_once(factory, configuration(), server.client()) is True
    server.uid = "new-service-uid"
    server.participants[ROOM][0] = participant(server.uid)
    server.history[ROOM] = []
    assert poll_once(factory, configuration(), server.client()) is False
    accounts, rooms, inbox = snapshot(factory)
    old = next(account for account in accounts if account.username == SERVICE)
    new = next(account for account in accounts if account.username == server.uid)
    assert not old.enabled and new.enabled
    old_room = next(room for room in rooms if room.account_id == old.id)
    assert not old_room.enabled and old_room.status == "ACCOUNT_CHANGED"
    assert len(inbox) == 1 and inbox[0].room_id == old_room.id and inbox[0].status == "PENDING"


def test_activation_persists_even_when_initial_room_discovery_fails(factory):
    server = Server()
    client = server.client()

    def unavailable():
        raise httpx.ConnectTimeout("discovery unavailable")

    client.rooms = unavailable
    with pytest.raises(httpx.ConnectTimeout):
        poll_once(factory, configuration(), client)
    accounts, _, _ = snapshot(factory)
    assert accounts[0].activated_at.replace(tzinfo=timezone.utc) == STAMP
    server.history[ROOM] = [message(1, stamp=SECOND + 1)]
    assert poll_once(factory, configuration(), server.client()) is True
    assert snapshot(factory)[2][0].status == "PENDING"


def test_talk_account_migration_matches_models_and_downgrades_cleanly(tmp_path):
    path = Path(__file__).resolve().parents[1] / "migrations/versions/0005_talk_account.py"
    spec = importlib.util.spec_from_file_location("talk_account_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    profile_spec = importlib.util.spec_from_file_location(
        "talk_profile_progress_migration", path.with_name("0006_talk_profile_progress.py"),
    )
    profile_migration = importlib.util.module_from_spec(profile_spec)
    profile_spec.loader.exec_module(profile_migration)
    device_spec = importlib.util.spec_from_file_location(
        "device_collection_migration", path.with_name("0007_device_collection.py"),
    )
    device_migration = importlib.util.module_from_spec(device_spec)
    device_spec.loader.exec_module(device_migration)
    photo_spec = importlib.util.spec_from_file_location(
        "inbound_photo_metadata_migration", path.with_name("0008_inbound_photo_metadata.py"),
    )
    photo_migration = importlib.util.module_from_spec(photo_spec)
    photo_spec.loader.exec_module(photo_migration)
    engine = create_engine("sqlite:///" + str(tmp_path / "migration.db"))
    old_tables = [table for table in Base.metadata.sorted_tables
                  if not table.name.startswith("talk_") and table.name != "device_collections"]
    with engine.begin() as connection:
        Base.metadata.create_all(connection, tables=old_tables)
        # Reproduce the actual 0004 schema, before the account outbox tags exist.
        connection.exec_driver_sql("DROP INDEX ix_deliveries_account_id")
        connection.exec_driver_sql("ALTER TABLE deliveries DROP COLUMN provenance")
        connection.exec_driver_sql("ALTER TABLE deliveries DROP COLUMN account_id")
        connection.exec_driver_sql("ALTER TABLE incidents DROP COLUMN has_user_photos")
        context = MigrationContext.configure(connection)
        with Operations.context(context):
            migration.upgrade()
            profile_migration.upgrade()
            device_migration.upgrade()
            photo_migration.upgrade()
        assert compare_metadata(context, Base.metadata) == []
        with Operations.context(context):
            photo_migration.downgrade()
            device_migration.downgrade()
            profile_migration.downgrade()
            migration.downgrade()
        assert all(not name.startswith("talk_") for name in connection.dialect.get_table_names(connection))
        assert "device_collections" not in connection.dialect.get_table_names(connection)
        assert "has_user_photos" not in {column["name"] for column in connection.dialect.get_columns(connection, "incidents")}
        assert not {"provenance", "account_id"}.intersection(
            column["name"] for column in connection.dialect.get_columns(connection, "deliveries")
        )
    engine.dispose()
