"""Optional delayed Talk progress stays private, durable and outside support history."""

import copy
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, func, select

from app.db import session_factory
from app.models import Base, Delivery, Message, TalkAccount, TalkInbox, TalkRoom, now
from app.services.nextcloud import progress
from app.services.nextcloud.account import PROGRESS_MESSAGE


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


def incoming(factory, *, inbox_status="RUNNING", state="WAITING", started=100.0,
             uid=UID, base=BASE, enabled=True, room_enabled=True, sender="employee", remote_id=None):
    with factory() as session, session.begin():
        account = TalkAccount(server_url=base, username=uid, enabled=enabled)
        session.add(account)
        session.flush()
        room = TalkRoom(account_id=account.id, room_token="privateroom", employee_id="employee",
                        enabled=room_enabled, flow={"source": "fixture-source", "step": 2})
        session.add(room)
        session.flush()
        record = TalkInbox(room_id=room.id, remote_message_id="10", remote_sequence=10, sender_id=sender,
                           content="Fixture private question", occurred_at=now(), status=inbox_status,
                           progress={"state": state, "started_at": started,
                                     **({"remote_id": remote_id} if remote_id is not None else {})})
        session.add(record)
        session.flush()
        return record.id, room.id, account.id


def snapshot(factory, inbox_id):
    with factory() as session:
        inbox = session.get(TalkInbox, inbox_id)
        room = session.get(TalkRoom, inbox.room_id)
        return copy.deepcopy(inbox.progress), inbox.status, copy.deepcopy(room.flow)


class Client:
    def __init__(self, *, uid=UID, support=True, send_error=None, deleted=True, edited=False, after_capabilities=None):
        self.uid, self.support, self.send_error = uid, support, send_error
        self.deleted, self.edited, self.after_capabilities = deleted, edited, after_capabilities
        self.calls = []

    def identity(self):
        return self.uid

    def supports_progress_cleanup(self):
        self.calls.append(("capabilities",))
        if self.after_capabilities:
            self.after_capabilities()
        return self.support

    def send(self, room, content, message_id, *, expected_employee):
        self.calls.append(("send", room, content, message_id, expected_employee))
        if self.send_error:
            raise self.send_error
        return "901"

    def delete_message(self, room, remote_id, *, expected_employee):
        self.calls.append(("delete", room, remote_id, expected_employee))
        return self.deleted

    def edit_message(self, room, remote_id, content, *, expected_employee):
        self.calls.append(("edit", room, remote_id, content, expected_employee))
        return self.edited


@pytest.mark.parametrize("state", ["SENDING", "SHOWN", "UNKNOWN", "CLEANING"])
def test_claim_does_not_reset_an_active_or_uncertain_progress_record(state, monkeypatch):
    original = {"state": state, "remote_id": "901", "started_at": 10}
    inbox = SimpleNamespace(progress=copy.deepcopy(original))
    monkeypatch.setattr(progress.time, "time", lambda: 100.0)
    progress.start_progress(inbox)
    assert inbox.progress == original


def test_new_claim_starts_a_delayed_record_without_network(monkeypatch):
    inbox = SimpleNamespace(progress={})
    monkeypatch.setattr(progress.time, "time", lambda: 100.0)
    progress.start_progress(inbox)
    assert inbox.progress == {"state": "WAITING", "started_at": 100.0}


def test_no_marker_before_two_seconds_then_exactly_one_private_marker_is_sent(factory):
    inbox_id, _, _ = incoming(factory)
    client = Client()
    assert progress.tick_progress(factory, settings(), client, clock=lambda: 101.999) is False
    assert client.calls == []
    assert progress.tick_progress(factory, settings(), client, clock=lambda: 102.0) is True
    assert client.calls == [("capabilities",), ("send", "privateroom", PROGRESS_MESSAGE, "progress:" + inbox_id, "employee")]
    state, inbox_status, flow = snapshot(factory, inbox_id)
    assert state == {"state": "SHOWN", "started_at": 100.0, "remote_id": "901"}
    assert inbox_status == "RUNNING" and flow == {"source": "fixture-source", "step": 2}
    assert progress.tick_progress(factory, settings(), client, clock=lambda: 110.0) is False
    assert len([call for call in client.calls if call[0] == "send"]) == 1
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(Message)) == 0
        assert session.scalar(select(func.count()).select_from(Delivery)) == 0


@pytest.mark.parametrize("inbox_status", ["DONE", "FAILED", "IGNORED"])
def test_completed_fast_turn_has_no_extra_progress_message(factory, inbox_status):
    inbox_id, _, _ = incoming(factory, inbox_status=inbox_status)
    client = Client()
    assert progress.tick_progress(factory, settings(), client, clock=lambda: 110) is True
    assert snapshot(factory, inbox_id)[0]["state"] == "CLEANED" and client.calls == []


def test_completion_during_the_capability_request_does_not_send_a_late_marker(factory):
    inbox_id, _, _ = incoming(factory)

    def finish():
        with factory() as session, session.begin():
            session.get(TalkInbox, inbox_id).status = "DONE"

    client = Client(after_capabilities=finish)
    assert progress.tick_progress(factory, settings(), client, clock=lambda: 110) is True
    assert snapshot(factory, inbox_id)[0]["state"] == "CLEANED"
    assert client.calls == [("capabilities",)]


def test_no_optional_marker_is_posted_when_server_cannot_remove_or_edit_it(factory):
    inbox_id, _, _ = incoming(factory)
    client = Client(support=False)
    assert progress.tick_progress(factory, settings(), client, clock=lambda: 110) is True
    assert snapshot(factory, inbox_id)[0]["state"] == "UNSUPPORTED"
    assert client.calls == [("capabilities",)]


def test_ambiguous_progress_post_is_never_automatically_repeated(factory):
    inbox_id, _, _ = incoming(factory)
    client = Client(send_error=TimeoutError("fixture-sensitive-network-body"))
    assert progress.tick_progress(factory, settings(), client, clock=lambda: 110) is True
    assert snapshot(factory, inbox_id)[0]["state"] == "UNKNOWN"
    progress.recover_progress(factory, settings(), UID)
    assert progress.tick_progress(factory, settings(), client, clock=lambda: 120) is False
    with factory() as session, session.begin():
        session.get(TalkInbox, inbox_id).status = "DONE"
    assert progress.tick_progress(factory, settings(), client, clock=lambda: 130) is True
    assert snapshot(factory, inbox_id)[0]["state"] == "ABANDONED"
    assert len([call for call in client.calls if call[0] == "send"]) == 1
    assert not any(call[0] in {"delete", "edit"} for call in client.calls)


@pytest.mark.parametrize("deleted,edited,expected", [(True, False, "CLEANED"), (False, True, "CLEANED"),
                                                     (False, False, "CLEANUP_FAILED")])
def test_cleanup_is_best_effort_and_preserves_the_employee_reply_and_flow(factory, deleted, edited, expected):
    inbox_id, _, _ = incoming(factory, inbox_status="DONE", state="SHOWN", remote_id="901")
    client = Client(deleted=deleted, edited=edited)
    assert progress.tick_progress(factory, settings(), client, clock=lambda: 110) is True
    state, inbox_status, flow = snapshot(factory, inbox_id)
    assert state["state"] == expected and inbox_status == "DONE"
    assert flow == {"source": "fixture-source", "step": 2}
    assert client.calls[0] == ("delete", "privateroom", "901", "employee")
    if deleted:
        assert len(client.calls) == 1
    else:
        assert client.calls[1] == ("edit", "privateroom", "901", "Обработка завершена.", "employee")


@pytest.mark.parametrize("changes", [
    {"uid": "other-service-uid"}, {"base": "https://another.example.invalid"},
    {"enabled": False}, {"room_enabled": False},
])
def test_other_or_disabled_account_room_is_not_used_for_progress(factory, changes):
    inbox_id, _, _ = incoming(factory, **changes)
    client = Client()
    assert progress.tick_progress(factory, settings(), client, clock=lambda: 110) is False
    assert snapshot(factory, inbox_id)[0]["state"] == "WAITING" and client.calls == []


def test_mismatching_employee_is_abandoned_before_optional_post(factory):
    inbox_id, _, _ = incoming(factory, sender="another-employee")
    client = Client()
    assert progress.tick_progress(factory, settings(), client, clock=lambda: 110) is True
    assert snapshot(factory, inbox_id)[0]["state"] == "ABANDONED" and client.calls == []


@pytest.mark.parametrize("revoke", ["room", "account"])
def test_local_revocation_during_capability_check_prevents_a_progress_post(factory, revoke):
    inbox_id, room_id, account_id = incoming(factory)

    def disable():
        with factory() as session, session.begin():
            target = session.get(TalkRoom, room_id) if revoke == "room" else session.get(TalkAccount, account_id)
            target.enabled = False

    client = Client(after_capabilities=disable)
    progress.tick_progress(factory, settings(), client, clock=lambda: 110)
    assert not any(call[0] == "send" for call in client.calls)
    assert snapshot(factory, inbox_id)[0]["state"] != "SHOWN"


def test_restart_recovery_changes_only_the_current_actual_account_records(factory):
    current, _, _ = incoming(factory, state="SENDING")
    # The account+room schema requires different accounts for independent fixtures.
    with factory() as session, session.begin():
        account = session.scalar(select(TalkAccount).where(TalkAccount.username == UID))
        room = session.scalar(select(TalkRoom).where(TalkRoom.account_id == account.id))
        cleaning = TalkInbox(room_id=room.id, remote_message_id="11", remote_sequence=11, sender_id="employee",
                             content="Fixture", occurred_at=now(), status="DONE",
                             progress={"state": "CLEANING", "started_at": 100.0, "remote_id": "902"})
        session.add(cleaning)
        session.flush()
        cleaning_id = cleaning.id
    other_user, _, _ = incoming(factory, state="SENDING", uid="other-service")
    other_server, _, _ = incoming(factory, state="CLEANING", base="https://another.example.invalid", remote_id="903")
    progress.recover_progress(factory, settings(), UID)
    progress.recover_progress(factory, settings(), UID)
    assert snapshot(factory, current)[0]["state"] == "UNKNOWN"
    assert snapshot(factory, cleaning_id)[0] == {"state": "SHOWN", "started_at": 100.0, "remote_id": "902"}
    assert snapshot(factory, other_user)[0]["state"] == "SENDING"
    assert snapshot(factory, other_server)[0]["state"] == "CLEANING"


def test_optional_identity_failure_logs_only_safe_code_and_never_changes_main_processing(factory, monkeypatch):
    inbox_id, _, _ = incoming(factory)
    events = []
    monkeypatch.setattr(progress, "event", lambda name, **fields: events.append((name, fields)))

    class Denied(Client):
        def identity(self):
            raise ValueError("fixture-private-response-and-password")

    assert progress.tick_progress(factory, settings(), Denied(), clock=lambda: 110) is False
    assert events == [("talk.progress_unavailable", {"error_code": "ValueError"})]
    state, inbox_status, flow = snapshot(factory, inbox_id)
    assert state["state"] == "WAITING" and inbox_status == "RUNNING" and flow["step"] == 2


def test_non_account_mode_never_authenticates_or_posts_progress(factory):
    class Forbidden(Client):
        def identity(self):
            pytest.fail("Other modes must not call the optional Talk progress transport")

    assert progress.tick_progress(factory, settings(nextcloud_backend="mock"), Forbidden()) is False
