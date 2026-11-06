"""Runner checks never send, and restart recovery belongs to the actual service UID."""

import json
from types import SimpleNamespace

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import create_engine, event, select

from app import talk_worker as runner
from app.db import session_factory
from app.models import Base, Conversation, Delivery, Incident, Message, TalkAccount, TalkInbox, TalkRoom, User, now
from app.services.nextcloud.account import AccountClient, AccountError

BASE = "https://nextcloud.example.invalid/office"
ACTUAL_UID = "service-actual-uid"
LOGIN = "SkladTech-bot"


def settings(**changes):
    return SimpleNamespace(**{
        "nextcloud_backend": "account", "nextcloud_base_url": BASE + "/",
        "nextcloud_username": LOGIN, "nextcloud_app_password": SecretStr("fake-test-only-app-password"),
        "nextcloud_allowed_rooms": [], "integration_timeout_seconds": 20,
    } | changes)


def test_connection_check_uses_only_readonly_identity_room_and_participant_requests(monkeypatch):
    requests = []

    def request_handler(request):
        requests.append(request)
        assert request.method == "GET"
        path = request.url.path.removeprefix("/office")
        if path == "/ocs/v2.php/cloud/user":
            data = {"id": ACTUAL_UID}
        elif path == "/ocs/v2.php/apps/spreed/api/v4/room":
            assert request.url.params["noStatusUpdate"] == "1"
            assert request.url.params["includeStatus"] == "false"
            data = [{"type": 1, "token": "privateone"}, {"type": 2, "token": "groupone"},
                    {"type": 1, "token": "changedone"}]
        elif path.endswith("/participants"):
            assert request.url.params["includeStatus"] == "false"
            people = [ACTUAL_UID, "employee"]
            if "changedone" in path:
                people.append("third-person")
            data = [{"actorType": "users", "actorId": uid, "participantType": 3} for uid in people]
        elif path.endswith(("/privateone", "/changedone")):
            data = {"type": 1, "token": path.rsplit("/", 1)[1]}
        else:
            pytest.fail("Unexpected endpoint in a read-only connection check")
        return httpx.Response(200, json={"ocs": {"meta": {"statuscode": 200}, "data": data}})

    monkeypatch.setattr(runner, "get_engine", lambda: pytest.fail("Connection check must not open the DB"))
    monkeypatch.setattr(runner, "run", lambda *args, **kwargs: pytest.fail("Connection check must not run the worker"))
    client = AccountClient(settings(), transport=httpx.MockTransport(request_handler))
    result = runner.check_connection(settings(), client)
    assert result == {"status": "ok", "account": "authenticated", "private_only": True,
                      "private_conversations": 1, "messages_sent": 0, "files_uploaded": 0}
    assert len(requests) == 6
    assert all(request.method == "GET" and not request.content for request in requests)
    assert not any("/chat/" in request.url.path or "/files/" in request.url.path for request in requests)
    serialized = json.dumps(result)
    assert ACTUAL_UID not in serialized and "employee" not in serialized and "privateone" not in serialized


@pytest.mark.parametrize("backend", ["mock", "talk"])
def test_connection_check_rejects_non_account_mode_before_any_request(backend):
    client = SimpleNamespace(identity=lambda: pytest.fail("Non-account check must not authenticate"))
    with pytest.raises(ValueError, match="TalkAccountModeRequired"):
        runner.check_connection(settings(nextcloud_backend=backend), client)


def test_connection_check_does_not_hide_authentication_failure():
    requests = []

    def denied(request):
        requests.append(request)
        return httpx.Response(401)

    client = AccountClient(settings(), transport=httpx.MockTransport(denied))
    with pytest.raises(AccountError, match="access_denied"):
        runner.check_connection(settings(), client)
    assert len(requests) == 1 and requests[0].method == "GET"


def test_cli_check_reports_only_safe_readonly_result(monkeypatch, capsys):
    monkeypatch.setattr(runner, "configure_logging", lambda: None)
    monkeypatch.setattr(runner, "get_settings", settings)
    calls = []

    def connection_check(configuration):
        calls.append(configuration.nextcloud_username)
        return {"status": "ok", "private_conversations": 0, "messages_sent": 0, "files_uploaded": 0}

    monkeypatch.setattr(runner, "check_connection", connection_check)
    monkeypatch.setattr(runner, "run", lambda *args, **kwargs: pytest.fail("--check must not poll or deliver"))
    assert runner.main(["--check"]) == 0
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert result["messages_sent"] == 0 and result["files_uploaded"] == 0
    assert calls == [LOGIN]
    assert "fake-test-only-app-password" not in output.out + output.err
    assert LOGIN not in output.out + output.err


def test_cli_check_error_never_serializes_http_body_or_credentials(monkeypatch, capsys):
    monkeypatch.setattr(runner, "configure_logging", lambda: None)
    monkeypatch.setattr(runner, "get_settings", settings)

    def denied(configuration):
        raise RuntimeError("fake-password-and-private-response-body")

    monkeypatch.setattr(runner, "check_connection", denied)
    assert runner.main(["--check"]) == 1
    output = capsys.readouterr()
    assert json.loads(output.out) == {"status": "error", "error_code": "RuntimeError"}
    assert "fake-password" not in output.out + output.err
    assert "private-response-body" not in output.out + output.err


@pytest.fixture
def factory():
    engine = create_engine("sqlite://")

    @event.listens_for(engine, "connect")
    def foreign_keys(connection, record):
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    yield session_factory(engine)
    engine.dispose()


def add_account(session, label, *, server=BASE, uid=ACTUAL_UID, enabled=True):
    account = TalkAccount(server_url=server, username=uid, enabled=enabled)
    user = User(external_id=label, display_name="Fixture employee")
    session.add_all([account, user])
    session.flush()
    conversation = Conversation(user_id=user.id, channel="nextcloud_talk", room_token="sameroom")
    session.add(conversation)
    session.flush()
    incident = Incident(conversation_id=conversation.id, subject="Fixture", description="Fixture")
    session.add(incident)
    session.flush()
    room = TalkRoom(account_id=account.id, room_token="sameroom", employee_id="employee-" + label,
                    cursor=42, flow={"position": "keep-" + label}, incident_id=incident.id)
    session.add(room)
    session.flush()
    return account, room, incident


def add_inbox(session, room, sequence, status, tries):
    inbox = TalkInbox(room_id=room.id, remote_message_id=str(sequence), remote_sequence=sequence,
                      sender_id=room.employee_id, content="Fixture question", occurred_at=now(),
                      status=status, tries=tries)
    session.add(inbox)
    session.flush()
    return inbox.id


def add_delivery(session, account_id, incident, sequence, status):
    message = Message(incident_id=incident.id, sequence=sequence, role="assistant", content="Fixture reply",
                      delivery_status="SENDING" if status == "SENDING" else "QUEUED")
    session.add(message)
    session.flush()
    delivery = Delivery(account_id=account_id, room_token="sameroom", message_id=message.id,
                        status=status, tries=1)
    session.add(delivery)
    session.flush()
    return delivery.id, message.id


def test_recovery_is_scoped_to_actual_uid_and_server_not_login_or_shared_room_token(factory):
    with factory() as session, session.begin():
        current, room, incident = add_account(session, "current")
        inbox = {
            "retry": add_inbox(session, room, 1, "RUNNING", 2),
            "exhausted": add_inbox(session, room, 2, "RUNNING", 3),
            "pending": add_inbox(session, room, 3, "PENDING", 0),
            "done": add_inbox(session, room, 4, "DONE", 1),
        }
        deliveries = {status: add_delivery(session, current.id, incident, i, status)
                      for i, status in enumerate(["SENDING", "UNKNOWN", "PENDING", "SENT"], 1)}
        legacy = add_delivery(session, None, incident, 5, "SENDING")
        others = []
        for label, server, uid in [("different-user", BASE, "other-uid"),
                                   ("different-server", "https://another.example.invalid", ACTUAL_UID),
                                   ("login-alias", BASE, LOGIN)]:
            account, other_room, other_incident = add_account(session, label, server=server, uid=uid)
            other_inbox = add_inbox(session, other_room, 1, "RUNNING", 3)
            other_delivery = add_delivery(session, account.id, other_incident, 1, "SENDING")
            others.append((other_inbox, *other_delivery))
        room_id, current_id = room.id, current.id

    runner.recover(factory, settings(), ACTUAL_UID)
    runner.recover(factory, settings(), ACTUAL_UID)  # Recovery is repeatable, not a resend.
    with factory() as session:
        assert session.get(TalkInbox, inbox["retry"]).status == "PENDING"
        exhausted = session.get(TalkInbox, inbox["exhausted"])
        assert (exhausted.status, exhausted.error_code, exhausted.tries) == ("FAILED", "InterruptedGeneration", 3)
        assert session.get(TalkInbox, inbox["pending"]).status == "PENDING"
        assert session.get(TalkInbox, inbox["done"]).status == "DONE"
        for prior, (delivery_id, message_id) in deliveries.items():
            delivery = session.get(Delivery, delivery_id)
            assert delivery.status == ("UNKNOWN" if prior in {"SENDING", "UNKNOWN"} else prior)
            assert delivery.tries == 1
            if prior in {"SENDING", "UNKNOWN"}:
                assert session.get(Message, message_id).delivery_status == "UNKNOWN"
        for inbox_id, delivery_id, message_id in others:
            assert session.get(TalkInbox, inbox_id).status == "RUNNING"
            assert session.get(Delivery, delivery_id).status == "SENDING"
            assert session.get(Message, message_id).delivery_status == "SENDING"
        assert session.get(Delivery, legacy[0]).status == "SENDING"
        assert session.get(Message, legacy[1]).delivery_status == "SENDING"
        assert (session.get(TalkRoom, room_id).cursor, session.get(TalkRoom, room_id).flow) == (
            42, {"position": "keep-current"})
        assert session.get(TalkAccount, current_id).enabled is True


def test_current_account_requires_enabled_actual_uid_at_the_current_server(factory):
    with factory() as session, session.begin():
        current, _, _ = add_account(session, "current")
        add_account(session, "alias", uid=LOGIN)
        current_id = current.id
    assert runner.current_account(factory, settings(), ACTUAL_UID) == current_id
    assert runner.current_account(factory, settings(), "unknown-uid") is None
    with factory() as session, session.begin():
        session.get(TalkAccount, current_id).enabled = False
    assert runner.current_account(factory, settings(), ACTUAL_UID) is None
    with factory() as session:
        assert session.scalar(select(TalkAccount.enabled).where(TalkAccount.id == current_id)) is False
