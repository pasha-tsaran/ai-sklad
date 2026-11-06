"""Native signaling never becomes a visible message or changes support delivery."""

import copy
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, func, select

from app.db import session_factory
from app.models import Base, Delivery, Message, TalkAccount, TalkInbox, TalkRoom, now
from app.services.nextcloud import progress

BASE = "https://nextcloud.example.invalid"
UID = "actual-service-uid"


def settings(**changes):
    return SimpleNamespace(**({"nextcloud_backend": "account", "nextcloud_base_url": BASE + "/"} | changes))


@pytest.fixture
def factory():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    yield session_factory(engine)
    engine.dispose()


def incoming(factory, *, status="RUNNING", state="WAITING", mode="native", uid=UID,
             base=BASE, enabled=True, room_enabled=True, sender="employee"):
    with factory() as session, session.begin():
        account = TalkAccount(server_url=base, username=uid, enabled=enabled)
        session.add(account)
        session.flush()
        room = TalkRoom(account_id=account.id, room_token="privateroom", employee_id="employee",
                        enabled=room_enabled, flow={"approved": "source", "position": 2})
        session.add(room)
        session.flush()
        inbox = TalkInbox(room_id=room.id, remote_message_id="10", remote_sequence=10, sender_id=sender,
                           content="Private fixture question", occurred_at=now(), status=status,
                           progress={"mode": mode, "state": state, "started_at": 100.0, "refreshed_at": 100.0})
        session.add(inbox)
        session.flush()
        return inbox.id, room.id, account.id


def snapshot(factory, inbox_id):
    with factory() as session:
        inbox = session.get(TalkInbox, inbox_id)
        room = session.get(TalkRoom, inbox.room_id)
        assert session.scalar(select(func.count()).select_from(Message)) == 0
        assert session.scalar(select(func.count()).select_from(Delivery)) == 0
        assert room.flow == {"approved": "source", "position": 2}
        return copy.deepcopy(inbox.progress), inbox.status


class Client:
    def __init__(self, success=True, error=None, after_signal=None):
        self.success, self.error, self.after_signal = success, error, after_signal
        self.calls = []

    def identity(self):
        return UID

    def set_typing(self, room, *, expected_employee, active):
        self.calls.append((room, expected_employee, active))
        if self.after_signal:
            self.after_signal(active)
        if self.error:
            raise self.error
        return self.success

    def __getattr__(self, name):
        if name in {"send", "delete_message", "edit_message", "send_file", "send_image"}:
            pytest.fail("Typing must never use chat messages, files or deletion")
        raise AttributeError(name)


def test_new_claim_records_only_local_native_intent(monkeypatch):
    monkeypatch.setattr(progress.time, "time", lambda: 100)
    inbox = SimpleNamespace(progress={"state": "SHOWN", "remote_id": "old-chat-message"})
    progress.start_progress(inbox)
    assert inbox.progress == {"mode": "native", "state": "WAITING", "started_at": 100}


@pytest.mark.parametrize("state", sorted(progress.ACTIVE_STATES))
def test_retry_preserves_a_native_session_until_it_is_stopped(state):
    initial = {"mode": "native", "state": state, "started_at": 100, "operation": "existing"}
    inbox = SimpleNamespace(progress=copy.deepcopy(initial))
    progress.start_progress(inbox)
    assert inbox.progress == initial


def test_delayed_native_start_refresh_and_finish_never_create_a_chat_message(factory):
    inbox_id, _, _ = incoming(factory)
    client = Client()
    assert not progress.tick_progress(factory, settings(), client, clock=lambda: 100.999)
    assert client.calls == []
    assert progress.tick_progress(factory, settings(), client, clock=lambda: 101)
    assert client.calls == [("privateroom", "employee", True)]
    assert snapshot(factory, inbox_id)[0]["state"] == "ACTIVE"
    assert not progress.tick_progress(factory, settings(), client, clock=lambda: 103.99)
    assert progress.tick_progress(factory, settings(), client, clock=lambda: 104)
    assert client.calls == [("privateroom", "employee", True)] * 2
    with factory() as session, session.begin():
        session.get(TalkInbox, inbox_id).status = "DONE"
    assert progress.tick_progress(factory, settings(), client, clock=lambda: 104.1)
    assert client.calls[-1] == ("privateroom", "employee", False)
    assert snapshot(factory, inbox_id) == (
        {"mode": "native", "state": "STOPPED", "started_at": 100.0, "refreshed_at": 104.1}, "DONE",
    )
    assert not progress.tick_progress(factory, settings(), client, clock=lambda: 110)


@pytest.mark.parametrize("status", ["DONE", "FAILED", "IGNORED", "PENDING"])
def test_fast_or_no_longer_running_turn_has_no_native_start(factory, status):
    inbox_id, _, _ = incoming(factory, status=status)
    client = Client()
    assert progress.tick_progress(factory, settings(), client, clock=lambda: 110)
    assert snapshot(factory, inbox_id)[0]["state"] == "STOPPED" and client.calls == []


@pytest.mark.parametrize("status", ["DONE", "FAILED", "IGNORED", "PENDING"])
def test_terminal_or_retrying_turn_stops_an_existing_native_indicator(factory, status):
    inbox_id, _, _ = incoming(factory, status=status, state="ACTIVE")
    client = Client()
    assert progress.tick_progress(factory, settings(), client, clock=lambda: 100.1)
    assert client.calls == [("privateroom", "employee", False)]
    assert snapshot(factory, inbox_id)[0]["state"] == "STOPPED"


@pytest.mark.parametrize("changes", [
    {"uid": "another-service"}, {"base": "https://another.example.invalid"},
])
def test_different_account_or_server_is_not_used_for_native_typing(factory, changes):
    inbox_id, _, _ = incoming(factory, **changes)
    client = Client()
    assert not progress.tick_progress(factory, settings(), client, clock=lambda: 110)
    assert snapshot(factory, inbox_id)[0]["state"] == "WAITING" and client.calls == []


@pytest.mark.parametrize("changes", [{"enabled": False}, {"room_enabled": False}, {"sender": "another-employee"}])
def test_disabled_room_or_unbound_employee_never_starts_typing(factory, changes):
    inbox_id, _, _ = incoming(factory, **changes)
    client = Client()
    assert progress.tick_progress(factory, settings(), client, clock=lambda: 110)
    assert snapshot(factory, inbox_id)[0]["state"] == "STOPPED" and client.calls == []


@pytest.mark.parametrize("revoke", ["room", "account", "finish"])
def test_completion_or_revocation_during_setup_immediately_stops_native_indicator(factory, revoke):
    inbox_id, room_id, account_id = incoming(factory)

    def change(active):
        if not active:
            return
        with factory() as session, session.begin():
            if revoke == "finish":
                session.get(TalkInbox, inbox_id).status = "DONE"
            else:
                row = session.get(TalkRoom, room_id) if revoke == "room" else session.get(TalkAccount, account_id)
                row.enabled = False

    client = Client(after_signal=change)
    assert progress.tick_progress(factory, settings(), client, clock=lambda: 110)
    assert client.calls == [("privateroom", "employee", True), ("privateroom", "employee", False)]
    assert snapshot(factory, inbox_id)[0]["state"] == "STOPPED"


def test_unsupported_native_transport_never_falls_back_to_chat_or_retries_the_turn(factory):
    inbox_id, _, _ = incoming(factory)
    client = Client(success=False)
    assert progress.tick_progress(factory, settings(), client, clock=lambda: 110)
    assert snapshot(factory, inbox_id)[0]["state"] == "UNAVAILABLE"
    assert not progress.tick_progress(factory, settings(), client, clock=lambda: 120)
    assert client.calls == [("privateroom", "employee", True)]


def test_failed_optional_transport_releases_claim_and_never_logs_private_exception_text(factory, monkeypatch):
    inbox_id, _, _ = incoming(factory)
    events = []
    monkeypatch.setattr(progress, "event", lambda name, **fields: events.append((name, fields)))
    client = Client(error=RuntimeError("private-fixture-password-and-message"))
    assert not progress.tick_progress(factory, settings(), client, clock=lambda: 110)
    assert snapshot(factory, inbox_id)[0]["state"] == "UNAVAILABLE"
    assert client.calls[-1] == ("privateroom", "employee", False)
    assert events == [("talk.typing_unavailable", {"error_code": "RuntimeError"})]


@pytest.mark.parametrize("prior_mode,state,status,expected", [
    ("legacy", "SHOWN", "RUNNING", "STOPPED"), ("legacy", "WAITING", "DONE", "STOPPED"),
    ("native", "STARTING", "PENDING", "WAITING"), ("native", "ACTIVE", "RUNNING", "WAITING"),
    ("native", "STOPPING", "DONE", "STOPPED"),
])
def test_restart_retires_old_placeholders_without_editing_or_deleting_history(factory, prior_mode, state, status, expected):
    inbox_id, _, _ = incoming(factory, mode=prior_mode, state=state, status=status)
    progress.recover_progress(factory, settings(), UID)
    assert snapshot(factory, inbox_id)[0]["state"] == expected
    assert "remote_id" not in snapshot(factory, inbox_id)[0]


def test_legacy_placeholder_is_retired_even_without_worker_recovery(factory):
    inbox_id, _, _ = incoming(factory, mode="legacy", state="WAITING")
    client = Client()
    assert progress.tick_progress(factory, settings(), client, clock=lambda: 110)
    assert snapshot(factory, inbox_id)[0] == {"mode": "native", "state": "STOPPED"}
    assert client.calls == []


def test_other_backend_never_authenticates_or_signals(factory):
    class Forbidden(Client):
        def identity(self):
            pytest.fail("Other backends cannot activate Talk typing")

    assert not progress.tick_progress(factory, settings(nextcloud_backend="mock"), Forbidden())
