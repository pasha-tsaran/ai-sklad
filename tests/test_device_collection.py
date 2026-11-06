"""Optional device capture is scoped, short-lived, one-use and free of network writes."""

import copy
import hashlib
import importlib.util
import re
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from uuid import NAMESPACE_URL, uuid5

import pytest
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from alembic.operations import Operations
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event, select

from app.api.dependencies import get_session
from app.api.device import router
from app.core.config import get_settings
from app.db import session_factory
from app.models import Base, Conversation, Delivery, DeviceCollection, Incident, Message, TalkAccount, TalkInbox, TalkRoom, User, now
from app.services.chat.policies import FORM_ACTION
from app.services.nextcloud.device_collection import TTL_SECONDS, issue_collection

BASE = "https://nextcloud.example.invalid"
PUBLIC = "https://support.example.invalid/assistant"
SERVICE, EMPLOYEE = "actual-service-uid", "employee-one"


@pytest.fixture
def device(tmp_path, monkeypatch):
    settings = SimpleNamespace(nextcloud_backend="account", nextcloud_base_url=BASE,
                               device_collection_base_url=PUBLIC)
    engine = create_engine("sqlite:///" + str(tmp_path / "device.db"), connect_args={"check_same_thread": False})
    @event.listens_for(engine, "connect")
    def foreign_keys(connection, record):
        connection.execute("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(engine)
    factory = session_factory(engine)
    with factory() as session, session.begin():
        account = TalkAccount(server_url=BASE, username=SERVICE, activated_at=now())
        session.add(account)
        session.flush()
        principal = str(uuid5(NAMESPACE_URL, BASE + "/" + SERVICE + "/privateroom/" + EMPLOYEE))
        user = User(id=principal, external_id="talk-account:" + principal, display_name="Павел Царан")
        session.add(user)
        session.flush()
        conversation = Conversation(user_id=principal, channel="nextcloud_talk", room_token="privateroom")
        session.add(conversation)
        session.flush()
        incident = Incident(conversation_id=conversation.id, state="DIAGNOSING", subject="Запрос в ТП",
                            description="", pending_action=FORM_ACTION)
        session.add(incident)
        session.flush()
        room = TalkRoom(account_id=account.id, room_token="privateroom", employee_id=EMPLOYEE,
                        incident_id=incident.id, flow={"escalation": {"stage": "FORM"}},
                        profile={"nextcloud_user_id": EMPLOYEE, "nextcloud_display_name": "Павел Царан",
                                 "identity_source": "nextcloud_participants"})
        session.add(room)
        session.flush()
        ids = dict(account_id=account.id, room_id=room.id, incident_id=incident.id, conversation_id=conversation.id)
    def sessions():
        with factory() as session, session.begin():
            yield session
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_session] = sessions
    app.dependency_overrides[get_settings] = lambda: settings
    monkeypatch.setattr("app.services.nextcloud.account.AccountClient.send", lambda *a, **kw: pytest.fail("No Talk network"))
    def issue():
        with factory() as session, session.begin():
            room, incident = session.get(TalkRoom, ids["room_id"]), session.get(Incident, ids["incident_id"])
            collection_id, link = issue_collection(session, settings, room, incident)
            room.flow = {**room.flow, "escalation": {
                "stage": "DEVICE", "collection_id": collection_id, "intake_fields": {},
            }}
        return collection_id, link, parse_qs(urlsplit(link).fragment)["token"][0]
    with TestClient(app) as client:
        yield SimpleNamespace(settings=settings, factory=factory, client=client, issue=issue, **ids)
    engine.dispose()


def callback(device, token, computer="WS-001", **kwargs):
    return device.client.post("/api/device-collection", json={"computer_name": computer},
                              headers={"Authorization": "Bearer " + token}, **kwargs)


def snapshot(device):
    with device.factory() as session:
        room = session.get(TalkRoom, device.room_id)
        incident = session.get(Incident, device.incident_id)
        rows = session.scalars(select(DeviceCollection).order_by(DeviceCollection.created_at)).all()
        return {"profile": copy.deepcopy(room.profile), "flow": copy.deepcopy(room.flow), "version": incident.version,
                "messages": [(row.id, row.content) for row in session.scalars(select(Message))],
                "deliveries": [(row.id, row.account_id) for row in session.scalars(select(Delivery))],
                "collections": [(row.id, row.computer_name, row.consumed_at) for row in rows]}


def test_issue_stores_only_sha256_with_fixed_expiry_and_fragment_link(device):
    collection_id, link, token = device.issue()
    assert re.fullmatch(r"[A-Za-z0-9_-]{43}", token)
    assert link == PUBLIC + "/device/collect#token=" + token and urlsplit(link).query == ""
    with device.factory() as session:
        row = session.get(DeviceCollection, collection_id)
        assert row.token_hash == hashlib.sha256(token.encode()).hexdigest()
        assert row.service_uid == SERVICE and row.requester_uid == EMPLOYEE
        assert row.account_id == device.account_id and row.room_id == device.room_id and row.incident_id == device.incident_id
        assert (row.expires_at - row.created_at).total_seconds() == TTL_SECONDS == 300
        assert token not in str(row.__dict__) and row.computer_name is row.consumed_at is None


def test_capture_updates_only_bound_incident_and_queues_private_profile_card_without_sending(device):
    collection_id, _, token = device.issue()
    result = callback(device, token)
    assert result.status_code == 200 and result.json() == {"status": "complete"}
    assert result.headers["cache-control"] == "no-store" and result.headers["referrer-policy"] == "no-referrer"
    with device.factory() as session:
        row = session.get(DeviceCollection, collection_id)
        room, incident = session.get(TalkRoom, device.room_id), session.get(Incident, device.incident_id)
        assert row.consumed_at is not None and row.computer_name == "WS-001" and incident.version == 1
        assert room.profile["nextcloud_display_name"] == "Павел Царан"
        assert room.profile["device_incident_id"] == device.incident_id
        assert room.profile["device_name"] == "WS-001" and room.profile["device_source"] == "windows_device_collector"
        assert room.flow["escalation"]["stage"] == "FORM"
        assert room.flow["escalation"]["intake_fields"]["computer_name"] == "WS-001"
        assert room.flow["escalation"]["intake_requested_field"] == "subject"
        replies = session.scalars(select(Message)).all()
        assert len(replies) == 1 and "Павел Царан" in replies[0].content and "WS-001" in replies[0].content
        assert token not in replies[0].content
        delivery = session.scalar(select(Delivery))
        assert delivery.account_id == device.account_id and delivery.room_token == "privateroom"
        assert delivery.provenance is None and delivery.status == "PENDING"


def test_same_consumed_computer_is_idempotent_but_different_computer_cannot_replace_it(device):
    _, _, token = device.issue()
    assert callback(device, token).status_code == 200
    before = snapshot(device)
    assert callback(device, token).json() == {"status": "complete"}
    assert snapshot(device) == before
    assert callback(device, token, "OTHER-PC").status_code == 409
    assert snapshot(device) == before


@pytest.mark.parametrize("scope", ["account-disabled", "room-disabled", "requester", "service", "server",
                                  "incident", "stage", "collection", "pending", "conversation", "malformed-flow"])
def test_stale_or_foreign_collection_cannot_mutate_any_profile_or_queue_a_card(device, scope):
    collection_id, _, token = device.issue()
    with device.factory() as session, session.begin():
        room, account = session.get(TalkRoom, device.room_id), session.get(TalkAccount, device.account_id)
        row, incident = session.get(DeviceCollection, collection_id), session.get(Incident, device.incident_id)
        if scope == "account-disabled":
            account.enabled = False
        elif scope == "room-disabled":
            room.enabled = False
        elif scope == "requester":
            row.requester_uid = "other-employee"
        elif scope == "service":
            row.service_uid = "other-service"
        elif scope == "server":
            account.server_url = "https://other.example.invalid"
        elif scope == "incident":
            room.incident_id = None
        elif scope == "stage":
            room.flow = {"escalation": {"stage": "FORM", "collection_id": row.id}}
        elif scope == "collection":
            room.flow = {"escalation": {"stage": "DEVICE", "collection_id": "newer-collection"}}
        elif scope == "pending":
            incident.pending_action = "PROCESSING"
        elif scope == "conversation":
            session.get(Conversation, device.conversation_id).room_token = "otherroom"
        else:
            room.flow = []
    before = snapshot(device)
    result = callback(device, token)
    assert result.status_code == 400 and result.json() == {"detail": "Device collection unavailable"}
    assert snapshot(device) == before


def test_reissue_and_cancel_make_previous_unconsumed_challenge_unusable(device):
    _, _, old = device.issue()
    _, _, current = device.issue()
    assert callback(device, old).status_code == 400
    with device.factory() as session, session.begin():
        session.get(TalkRoom, device.room_id).flow = {"escalation": {"stage": "CHOICE"}}
    before = snapshot(device)
    assert callback(device, current).status_code == 400 and snapshot(device) == before


@pytest.mark.parametrize("content,status", [("MANUAL-PC", "PENDING"), ("Отмена", "PENDING"),
                                           ("Компьютер: MANUAL-PC", "RUNNING")])
def test_queued_employee_input_wins_before_device_callback_advances_the_next_question(device, content, status):
    _, _, token = device.issue()
    with device.factory() as session, session.begin():
        session.add(TalkInbox(room_id=device.room_id, remote_message_id="new-message", remote_sequence=1,
                              sender_id=EMPLOYEE, content=content, occurred_at=now(), status=status))
    before = snapshot(device)
    assert callback(device, token, "AUTO-PC").status_code == 400
    assert snapshot(device) == before


def test_queued_service_echo_is_not_treated_as_employee_cancel_or_hostname(device):
    _, _, token = device.issue()
    with device.factory() as session, session.begin():
        session.add(TalkInbox(room_id=device.room_id, remote_message_id="own-echo", remote_sequence=1,
                              sender_id=SERVICE, content="our previous card", occurred_at=now(), status="PENDING"))
    assert callback(device, token).status_code == 200


@pytest.mark.parametrize("computer", [None, 3, "", "---", "...", "WS 001", "../pc", "WS\r\n001", "x" * 64],
                         ids=["null", "number", "empty", "dashes", "dots", "space", "path", "newline", "long"])
def test_invalid_hostnames_never_mutate_state(device, computer):
    _, _, token = device.issue()
    before = snapshot(device)
    assert callback(device, token, computer).status_code == 400
    assert snapshot(device) == before


@pytest.mark.parametrize("extra", [{"requester_uid": "boss"}, {"account_holder_name": "Администратор"},
                                  {"api_token": "must-not-be-used"}, {"endpoint": "https://elsewhere.example"}])
def test_device_payload_can_supply_only_computer_name_never_identity_credentials_or_target(device, extra):
    _, _, token = device.issue()
    before = snapshot(device)
    result = device.client.post("/api/device-collection", json={"computer_name": "WS-001", **extra},
                                headers={"Authorization": "Bearer " + token})
    assert result.status_code == 400 and snapshot(device) == before
    assert all(value not in result.text for value in extra.values())


@pytest.mark.parametrize("raw", ['{"computer_name":"FIRST-PC","computer_name":"SECOND-PC"}',
                                 '{"computer_name":"WS-001",', '{"computer_name":"' + "A" * 2100 + '"}'],
                         ids=["duplicate-key", "broken-json", "oversize"])
def test_malformed_ambiguous_or_oversized_json_has_generic_error_without_state_change(device, raw):
    _, _, token = device.issue()
    before = snapshot(device)
    result = device.client.post("/api/device-collection", content=raw,
                                headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
    assert result.status_code == 400 and result.json() == {"detail": "Device collection unavailable"}
    assert snapshot(device) == before and token not in result.text


def test_expired_unknown_malformed_and_missing_tokens_have_generic_no_store_failure(device):
    collection_id, _, token = device.issue()
    with device.factory() as session, session.begin():
        session.get(DeviceCollection, collection_id).expires_at = now() - timedelta(seconds=1)
    before = snapshot(device)
    attempts = [callback(device, token), callback(device, "x" * 43), callback(device, "../unsafe"),
                device.client.post("/api/device-collection", json={"computer_name": "WS-001"})]
    assert all(result.status_code == 400 for result in attempts)
    assert all(result.json() == {"detail": "Device collection unavailable"} for result in attempts)
    assert all(result.headers["cache-control"] == "no-store" for result in attempts)
    assert snapshot(device) == before


def test_status_is_read_only_contains_no_identifiers_and_rejects_query_auth(device):
    _, _, token = device.issue()
    before = snapshot(device)
    headers = {"Authorization": "Bearer " + token}
    result = device.client.get("/api/device-collection", headers=headers)
    assert result.status_code == 200 and result.json() == {"status": "pending"}
    assert result.headers["cache-control"] == "no-store" and snapshot(device) == before
    assert device.client.get("/api/device-collection?token=" + token).status_code == 400
    assert callback(device, token).status_code == 200
    result = device.client.get("/api/device-collection", headers=headers)
    assert result.json() == {"status": "complete"} and "WS-001" not in result.text and EMPLOYEE not in result.text


def test_landing_uses_fragment_and_one_permission_respecting_attempt_without_external_assets(device):
    result = device.client.get("/device/collect")
    assert result.status_code == 200 and result.headers["cache-control"] == "no-store"
    assert result.headers["referrer-policy"] == "no-referrer"
    assert "frame-ancestors 'none'" in result.headers["content-security-policy"]
    nonce = re.search(r"script-src 'nonce-([^']+)'", result.headers["content-security-policy"])[1]
    assert '<script nonce="' + nonce + '">' in result.text and "__NONCE__" not in result.text
    assert "location.hash" in result.text and "history.replaceState" in result.text
    assert '"ai-support://collect/"+token' in result.text and "attempted=true" in result.text
    assert 'Authorization:"Bearer "+token' in result.text and 'redirect:"error"' in result.text
    assert "разрешение" in result.text and "только имя этого компьютера" in result.text
    assert "<script src=" not in result.text and "<img" not in result.text and "console." not in result.text


def test_optional_collection_is_disabled_until_valid_https_receiver_is_configured(device):
    device.settings.device_collection_base_url = ""
    assert device.client.get("/device/collect").status_code == 404
    with pytest.raises(HTTPException) as error:
        device.issue()
    assert error.value.status_code == 404
    with device.factory() as session:
        assert session.scalar(select(DeviceCollection)) is None


def test_device_migration_is_additive_preserves_existing_rows_and_downgrades_only_its_table(tmp_path):
    path = Path(__file__).resolve().parents[1] / "migrations/versions/0007_device_collection.py"
    spec = importlib.util.spec_from_file_location("device_additive_migration", path)
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    engine = create_engine("sqlite:///" + str(tmp_path / "migration.db"))
    old = [table for table in Base.metadata.sorted_tables if table.name != "device_collections"]
    with engine.begin() as connection:
        Base.metadata.create_all(connection, tables=old)
        connection.execute(User.__table__.insert().values(id="existing-user", external_id="prior-user", display_name="Имя"))
        connection.execute(TalkAccount.__table__.insert().values(id="existing-account", server_url=BASE,
                                                                username=SERVICE, activated_at=now()))
        connection.execute(TalkRoom.__table__.insert().values(id="existing-room", account_id="existing-account",
                          room_token="roomold", employee_id=EMPLOYEE, flow={"step": 6}, profile={"device_name": "OLD-PC"}))
        tables = [User.__table__, TalkAccount.__table__, TalkRoom.__table__]
        before = {table.name: [dict(row._mapping) for row in connection.execute(select(table))] for table in tables}
        context = MigrationContext.configure(connection)
        with Operations.context(context):
            migration.upgrade()
        assert compare_metadata(context, Base.metadata) == []
        assert {table.name: [dict(row._mapping) for row in connection.execute(select(table))] for table in tables} == before
        with Operations.context(context):
            migration.downgrade()
        assert "device_collections" not in connection.dialect.get_table_names(connection)
        assert {table.name: [dict(row._mapping) for row in connection.execute(select(table))] for table in tables} == before
    engine.dispose()
