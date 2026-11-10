"""Ordered account outbox delivery cannot cross a source, employee or account boundary."""

import copy
import hashlib
import hmac
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import NAMESPACE_URL, uuid5

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import create_engine, event, select

from app.db import session_factory
from app.models import (Base, Conversation, Delivery, Incident, Message, MessageAttachment,
                        TalkAccount, TalkInbox, TalkRoom, User)
from app.services.nextcloud.delivery import RoutedMessenger, TalkClient, deliver_one
from app.talk_worker import check_connection, recover

BASE = "https://nextcloud.example"
ROOM = "roomone"
UID = "service-uid"
SOURCE = {"file_id": "00000000-0000-4000-8000-000000000009", "filename": "Сканер.docx",
          "knowledge_id": "85cda3aa-b854-4c26-9948-12da742277ef", "sha256": "a" * 64}


def settings(**overrides):
    return SimpleNamespace(**(dict(nextcloud_backend="account", nextcloud_base_url=BASE,
                                   nextcloud_username="SkladTech-bot", nextcloud_allowed_rooms=[ROOM],
                                   nextcloud_bot_secret=SecretStr("test-secret" * 6), integration_timeout_seconds=20,
                                   openwebui_knowledge_ids=[SOURCE["knowledge_id"]]) | overrides))


@pytest.fixture
def factory(tmp_path):
    engine = create_engine("sqlite:///" + str(tmp_path / "delivery.db"))

    @event.listens_for(engine, "connect")
    def foreign_keys(connection, record):
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    yield session_factory(engine)
    engine.dispose()


def bind(factory, *, uid=UID, employee="employee-one", token=ROOM, base=BASE):
    with factory() as session, session.begin():
        account = session.scalar(select(TalkAccount).where(TalkAccount.server_url == base, TalkAccount.username == uid))
        if account is None:
            account = TalkAccount(server_url=base, username=uid)
            session.add(account)
            session.flush()
        principal = str(uuid5(NAMESPACE_URL, base + "/" + uid + "/" + token + "/" + employee))
        user = User(id=principal, external_id="talk:" + principal, display_name=employee)
        session.add(user)
        session.flush()
        conversation = Conversation(user_id=user.id, channel="nextcloud_talk", room_token=token)
        session.add(conversation)
        session.flush()
        incident = Incident(conversation_id=conversation.id, subject="Сканер", description="Настройка сканера")
        session.add(incident)
        session.flush()
        room = TalkRoom(account_id=account.id, room_token=token, employee_id=employee, incident_id=incident.id)
        session.add(room)
        session.flush()
        return SimpleNamespace(account=account.id, room=room.id, incident=incident.id)


def enqueue(factory, bound, blocks=None, source=None):
    blocks = blocks if blocks is not None else [("Проверенный ответ", None)]
    with factory() as session, session.begin():
        RoutedMessenger().send_blocks(session, bound.incident, blocks, source=source)
        return [row.id for row in session.scalars(select(Delivery).join(Message).where(
            Message.incident_id == bound.incident,
        ).order_by(Message.sequence))]


def state(factory):
    with factory() as session:
        return [(copy.deepcopy(row), copy.deepcopy(session.get(Message, row.message_id)))
                for row in session.scalars(select(Delivery).join(Message).order_by(Message.created_at, Message.sequence))]


def photograph(number=1):
    raw = b"immutable-png-snapshot-" + str(number).encode()
    return SimpleNamespace(data=raw, filename=f"illustration-{number:02d}.png", content_type="image/png",
                           caption=f"На фотографии {number} нажмите нужную кнопку.")


class Sender:
    def __init__(self, uid=UID, *, timeline=None, failure=False, changed_identity=False):
        self.uid, self.failure, self.changed_identity = uid, failure, changed_identity
        self.identity_calls = 0
        self.writes = []
        self.timeline = timeline if timeline is not None else []

    def identity(self):
        self.identity_calls += 1
        return "different-service-uid" if self.changed_identity and self.identity_calls > 1 else self.uid

    def send(self, room, content, message_id, *, expected_employee=None):
        self.writes.append(("text", room, content, message_id, expected_employee))
        self.timeline.append("send")
        if self.failure:
            raise httpx.ReadTimeout("ambiguous write")
        return "text-remote-id"

    def send_image(self, room, content, message_id, attachment, *, expected_employee=None):
        self.writes.append(("photo", room, content, message_id, expected_employee, attachment.data,
                            attachment.caption, attachment.sha256))
        self.timeline.append("photo")
        return "photo-remote-id"


def approve(monkeypatch, *, current=True, timeline=None):
    checks = []

    class Knowledge:
        def __init__(self, configured):
            assert configured.openwebui_knowledge_ids == [SOURCE["knowledge_id"]]

        def is_current(self, source):
            checks.append(copy.deepcopy(source))
            if timeline is not None:
                timeline.append("source")
            return current

    monkeypatch.setattr("app.services.rag.talk.TalkKnowledge", Knowledge)
    return checks


def test_blocks_keep_exact_step_text_before_each_photo_and_tag_same_source_account(factory):
    bound = bind(factory)
    first, second = photograph(1), photograph(2)
    text_one, text_two = "Шаг 1. Нажмите «Операции».\nНа фото показано меню.", "Шаг 2. Выберите «Настройка ТО»."
    source = dict(SOURCE)
    ids = enqueue(factory, bound, [(text_one, first), (text_two, second)], source)
    source["sha256"] = "b" * 64
    rows = state(factory)
    assert len(ids) == len(rows) == 4
    assert [message.content for _, message in rows] == [text_one, "", text_two, ""]
    assert [message.sequence for _, message in rows] == [1, 2, 3, 4]
    assert all(delivery.provenance == SOURCE and delivery.account_id == bound.account for delivery, _ in rows)
    with factory() as session:
        pictures = {photo.message_id: photo for photo in session.scalars(select(MessageAttachment))}
        assert set(pictures) == {rows[1][1].id, rows[3][1].id}
        for index, original in [(1, first), (3, second)]:
            saved = pictures[rows[index][1].id]
            assert saved.data == original.data and saved.caption == original.caption
            assert saved.sha256 == hashlib.sha256(original.data).hexdigest()


def test_delivery_checks_current_source_then_sends_text_photo_in_order_with_employee_binding(factory, monkeypatch):
    bound = bind(factory)
    original = photograph()
    enqueue(factory, bound, [("Нажмите кнопку на фотографии.", original)], SOURCE)
    timeline = []
    checks = approve(monkeypatch, timeline=timeline)
    sender = Sender(timeline=timeline)
    assert deliver_one(factory, settings(), sender)
    assert deliver_one(factory, settings(), sender)
    assert not deliver_one(factory, settings(), sender)
    assert [write[0] for write in sender.writes] == ["text", "photo"]
    assert [write[4] for write in sender.writes] == ["employee-one", "employee-one"]
    assert sender.writes[0][2] == "Нажмите кнопку на фотографии."
    assert sender.writes[1][5:7] == (original.data, original.caption)
    assert checks == [SOURCE, SOURCE] and timeline == ["source", "send", "source", "photo"]
    assert all(delivery.status == "SENT" and delivery.tries == 1 and message.delivery_status == "DELIVERED_TALK"
               for delivery, message in state(factory))


@pytest.mark.parametrize("problem", ["source_changed", "room_missing", "room_disabled", "wrong_identity"])
def test_stale_source_or_missing_private_binding_never_writes_and_does_not_retry(factory, monkeypatch, problem):
    bound = bind(factory)
    enqueue(factory, bound, [("Утверждённый текст.", photograph())], SOURCE)
    approve(monkeypatch, current=problem != "source_changed")
    sender = Sender(changed_identity=problem == "wrong_identity")
    with factory() as session, session.begin():
        if problem == "room_missing":
            session.delete(session.get(TalkRoom, bound.room))
        elif problem == "room_disabled":
            session.get(TalkRoom, bound.room).enabled = False
    assert deliver_one(factory, settings(), sender)
    assert not sender.writes
    assert not deliver_one(factory, settings(), sender)
    rows = state(factory)
    assert rows[0][0].status == "UNKNOWN" and rows[0][0].tries == 1
    assert rows[0][1].delivery_status == "UNKNOWN" and rows[1][0].status == "PENDING"


def test_ambiguous_write_blocks_later_text_and_photos_only_for_its_room(factory):
    first = bind(factory)
    second = bind(factory, token="roomtwo", employee="employee-two")
    enqueue(factory, first, [("Первая попытка", photograph()), ("Не отправлять повторно", None)])
    enqueue(factory, second, [("Другой сотрудник", None)])
    failing = Sender(failure=True)
    assert deliver_one(factory, settings(), failing)
    sender = Sender()
    assert deliver_one(factory, settings(), sender)
    assert sender.writes[0][1:3] == ("roomtwo", "Другой сотрудник")
    assert not deliver_one(factory, settings(), sender)
    rows = state(factory)
    # Different rooms can have the same clock tick. Only the per-room order
    # is contractual; sorting both rooms by their local sequence is ambiguous.
    assert [delivery.status for delivery, _ in rows if delivery.room_token == ROOM] == ["UNKNOWN", "PENDING", "PENDING"]
    assert [delivery.status for delivery, _ in rows if delivery.room_token == "roomtwo"] == ["SENT"]
    assert len(failing.writes) == len(sender.writes) == 1


def test_same_room_token_in_different_accounts_does_not_share_blocking_or_delivery(factory):
    first = bind(factory, uid="old-service", employee="old-employee")
    second = bind(factory, uid="new-service", employee="new-employee")
    first_id = enqueue(factory, first, [("OLD_ACCOUNT_PRIVATE_TEXT", None)])[0]
    enqueue(factory, second, [("New account reply", None)])
    with factory() as session, session.begin():
        session.get(Delivery, first_id).status = "UNKNOWN"
    sender = Sender(uid="new-service")
    assert deliver_one(factory, settings(), sender)
    assert [write[2] for write in sender.writes] == ["New account reply"]
    assert sender.writes[0][4] == "new-employee"
    assert not deliver_one(factory, settings(), sender)


def test_incorrect_account_tag_cannot_deliver_another_incidents_private_text(factory):
    first = bind(factory, uid="first-service", employee="first-employee")
    second = bind(factory, uid="second-service", employee="second-employee")
    delivery_id = enqueue(factory, second, [("SECOND_EMPLOYEE_PRIVATE_TEXT", None)])[0]
    with factory() as session, session.begin():
        session.get(Delivery, delivery_id).account_id = first.account
    sender = Sender(uid="first-service")
    assert deliver_one(factory, settings(), sender)
    assert not sender.writes and state(factory)[0][0].status == "UNKNOWN"


def test_late_previous_incident_ticket_acknowledgement_still_binds_same_private_conversation(factory):
    bound = bind(factory)
    with factory() as session, session.begin():
        previous = session.get(Incident, bound.incident)
        previous.state = "ESCALATED"
        current = Incident(conversation_id=previous.conversation_id, subject="Другая проблема",
                           description="Новый вопрос", state="DIAGNOSING")
        session.add(current)
        session.flush()
        session.get(TalkRoom, bound.room).incident_id = current.id
        acknowledgement = RoutedMessenger().send(session, previous.id, "Заявка принята почтовым сервером.")
        delivery = session.scalar(select(Delivery).where(Delivery.message_id == acknowledgement.id))
        assert delivery.account_id == bound.account
    sender = Sender()
    assert deliver_one(factory, settings(), sender)
    assert sender.writes[0][2] == "Заявка принята почтовым сервером."
    assert sender.writes[0][4] == "employee-one"
    assert state(factory)[0][0].status == "SENT"


def test_legacy_webhook_mode_never_drains_account_outbox_even_when_room_token_is_allowed(factory):
    bound = bind(factory)
    enqueue(factory, bound, [("ACCOUNT_PRIVATE_TEXT", None)])
    sender = Sender()
    assert not deliver_one(factory, settings(nextcloud_backend="talk"), sender)
    assert not sender.writes and state(factory)[0][0].status == "PENDING"


@pytest.mark.parametrize("empty", [None, []])
def test_legacy_bot_201_empty_acknowledgement_is_a_successful_reference_receipt(empty):
    configured = settings(nextcloud_backend="talk")
    requests = []

    def accepted(request):
        requests.append(request)
        content = json.loads(request.content)
        nonce = request.headers["X-Nextcloud-Talk-Bot-Random"]
        expected = hmac.new(configured.nextcloud_bot_secret.get_secret_value().encode(),
                            (nonce + content["message"]).encode(), hashlib.sha256).hexdigest()
        assert request.headers["X-Nextcloud-Talk-Bot-Signature"] == expected
        return httpx.Response(201, json={"ocs": {"meta": {"statuscode": 100}, "data": empty}})

    receipt = TalkClient(configured, httpx.MockTransport(accepted)).send(ROOM, "Обычный ответ", "local-id")
    assert receipt == "reference:" + hashlib.sha256(b"local-id").hexdigest() and len(requests) == 1


def test_runner_recovery_is_scoped_to_current_account_and_never_replays_external_send(factory):
    first = bind(factory)
    second = bind(factory, uid="other-service", employee="other-employee")
    first_delivery = enqueue(factory, first)[0]
    other_delivery = enqueue(factory, second)[0]
    with factory() as session, session.begin():
        for delivery_id in [first_delivery, other_delivery]:
            session.get(Delivery, delivery_id).status = "SENDING"
        session.add_all([
            TalkInbox(room_id=first.room, remote_message_id="1", remote_sequence=1, sender_id="employee-one",
                      content="Первая", occurred_at=datetime.now(timezone.utc), status="RUNNING", tries=1),
            TalkInbox(room_id=first.room, remote_message_id="2", remote_sequence=2, sender_id="employee-one",
                      content="Вторая", occurred_at=datetime.now(timezone.utc), status="RUNNING", tries=3),
            TalkInbox(room_id=second.room, remote_message_id="3", remote_sequence=3, sender_id="other-employee",
                      content="Другая", occurred_at=datetime.now(timezone.utc), status="RUNNING", tries=1),
        ])
    recover(factory, settings(), UID)
    with factory() as session:
        inbox = session.scalars(select(TalkInbox).order_by(TalkInbox.remote_sequence)).all()
        assert [row.status for row in inbox] == ["PENDING", "FAILED", "RUNNING"]
        assert session.get(Delivery, first_delivery).status == "UNKNOWN"
        assert session.get(Delivery, other_delivery).status == "SENDING"
        assert session.get(Message, session.get(Delivery, first_delivery).message_id).delivery_status == "UNKNOWN"


def test_connection_check_does_not_send_or_change_read_markers():
    calls = []

    class Client:
        def identity(self):
            calls.append("identity")
            return UID

        def rooms(self):
            calls.append("rooms")
            return [{"type": 1, "token": ROOM}, {"type": 2, "token": "groupchat"}]

        def validate_room(self, token):
            calls.append(token)
            return {"type": 1, "employee_id": "employee-one"}

    result = check_connection(settings(), Client())
    assert result["status"] == "ok" and result["private_conversations"] == 1
    assert result["messages_sent"] == result["files_uploaded"] == 0
    assert calls == ["identity", "rooms", ROOM]


@pytest.mark.parametrize("content", ["Строка инструкции.\n" * 5000, "x" * 70_003, "слово " * 12_000],
                         ids=["many-lines", "long-word", "long-line"])
def test_long_instruction_is_queued_without_losing_or_reordering_source_characters(factory, content):
    bound = bind(factory)
    enqueue(factory, bound, [(content, photograph())], SOURCE)
    rows = state(factory)
    with factory() as session:
        photo_messages = {photo.message_id for photo in session.scalars(select(MessageAttachment))}
    text = [message.content for _, message in rows if message.id not in photo_messages]
    assert "".join(text) == content and all(0 < len(page) <= 32_000 for page in text)
    assert rows[-1][1].id in photo_messages and len(photo_messages) == 1
    assert all(delivery.provenance == SOURCE and delivery.account_id == bound.account for delivery, _ in rows)


def test_large_text_only_reply_uses_same_lossless_message_limit_without_photos(factory):
    bound = bind(factory)
    text = "Точный исходный текст.\n" * 4000
    with factory() as session, session.begin():
        first = RoutedMessenger().send(session, bound.incident, text)
        assert first.content and first.sequence == 1
    rows = state(factory)
    assert "".join(message.content for _, message in rows) == text
    assert all(0 < len(message.content) <= 32_000 for _, message in rows)
    assert all(delivery.account_id == bound.account for delivery, _ in rows)
    with factory() as session:
        assert session.scalar(select(MessageAttachment)) is None
