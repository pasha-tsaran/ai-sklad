"""New input interrupts output without losing, repeating or reordering photos."""

import copy

import pytest
from sqlalchemy import func, select

from app.models import Delivery, Feedback, Incident, Message, SuccessfulRetrieval, now
from app.models.talk import TalkInbox, TalkRoom
from app.services.chat.talk_dialogue import CHANGED, Turn
from app.services.nextcloud.delivery import deliver_one
from app.services.nextcloud.interruption import ACTIVE, PAUSED, is_resume_request, latest_paused, resume_response
from app.talk_worker import recover
from tests.test_talk_delivery import Sender, bind, enqueue
from tests.test_talk_dialogue import DOC, QUESTION, Knowledge, approved
from tests.test_talk_dialogue import talk as talk


def _settings(talk):
    return talk.settings.model_copy(update={"nextcloud_backend": "account"})


def _rows(talk):
    with talk.env.factory() as session:
        return [(copy.deepcopy(row), copy.deepcopy(session.get(Message, row.message_id)))
                for row in session.scalars(select(Delivery).join(Message).order_by(Message.sequence))]


def _approve(monkeypatch, knowledge):
    monkeypatch.setattr("app.services.rag.talk.TalkKnowledge", lambda configured: knowledge)


def _drain(talk, sender):
    for _ in range(100):
        if not deliver_one(talk.env.factory, _settings(talk), sender):
            return
    raise AssertionError("Outbox did not finish")


def test_pause_between_step_text_and_photo_resumes_exact_unsent_rows_after_new_reply(talk, monkeypatch):
    knowledge = Knowledge([approved(8)])
    _approve(monkeypatch, knowledge)
    talk.incoming(QUESTION)
    talk.process(knowledge)
    original_ids = [row.id for row, _ in _rows(talk)]
    original_flow = talk.snapshot()["flow"]
    sender = Sender(uid="bot-uid")
    assert deliver_one(talk.env.factory, _settings(talk), sender)
    assert deliver_one(talk.env.factory, _settings(talk), sender)
    assert [item[0] for item in sender.writes] == ["text", "text"]

    talk.incoming("Другой вопрос", 2, previous_delivered=False)
    assert not deliver_one(talk.env.factory, _settings(talk), sender)
    assert [row.status for row, _ in _rows(talk)] == ["SENT", "SENT"] + ["PAUSED"] * (len(original_ids) - 2)
    import app.services.chat.talk_dialogue as dialogue

    original_turn = dialogue._turn
    monkeypatch.setattr(dialogue, "_turn", lambda *args, **kwargs: Turn("REPLY", [("Ответ на новый вопрос", None)], flow={}))
    assert talk.process(knowledge)
    _drain(talk, sender)
    assert sender.writes[-1][2] == "Ответ на новый вопрос"
    assert len(sender.writes) == 3
    monkeypatch.setattr(dialogue, "_turn", original_turn)

    talk.incoming("Закончи предыдущий ответ", 3, previous_delivered=False)
    assert talk.process(knowledge)
    assert talk.snapshot()["flow"] == original_flow
    _drain(talk, sender)
    wire_ids = [item[3] for item in sender.writes]
    original_message_ids = [message.id for row, message in _rows(talk) if row.id in original_ids]
    assert [value for value in wire_ids if value in original_message_ids] == original_message_ids
    assert sender.writes[3][0] == "photo"  # The step text already reached the employee.
    assert len(wire_ids) == len(set(wire_ids)) == len(original_ids) + 1
    assert all(row.status == "SENT" for row, _ in _rows(talk))
    talk.incoming("Готово", 4, previous_delivered=False)
    assert talk.process(knowledge)
    assert knowledge.reads == [DOC]  # Resume did not regenerate or read from step one.
    assert talk.snapshot()["flow"]["next_offset"] is None


def test_later_input_arriving_during_generation_pauses_all_stale_output_before_delivery(talk, monkeypatch):
    import app.services.chat.talk_dialogue as dialogue

    original_turn = dialogue._turn
    def interrupted(message, *args, **kwargs):
        response = original_turn(message, *args, **kwargs)
        talk.incoming("Сканер уже подключен, нужна другая настройка", 2, previous_delivered=False)
        return response

    monkeypatch.setattr(dialogue, "_turn", interrupted)
    first = talk.incoming(QUESTION)
    assert talk.process(Knowledge([approved(8)]))
    rows = _rows(talk)
    assert rows and all(row.status == "PAUSED" for row, _ in rows)
    assert all(row.tries == 0 for row, _ in rows)
    with talk.env.factory() as session:
        assert session.get(TalkInbox, first).status == "DONE"
        assert len(session.get(TalkRoom, talk.room_id).profile[PAUSED]) == 1


def test_restart_keeps_paused_tail_and_resumes_without_repeating_sent_items(talk, monkeypatch):
    knowledge = Knowledge([approved(2)])
    _approve(monkeypatch, knowledge)
    talk.incoming(QUESTION)
    talk.process(knowledge)
    sender = Sender(uid="bot-uid")
    assert deliver_one(talk.env.factory, _settings(talk), sender)
    talk.incoming("Привет", 2, previous_delivered=False)
    assert not deliver_one(talk.env.factory, _settings(talk), sender)
    assert talk.process(knowledge)
    _drain(talk, sender)
    recover(talk.env.factory, talk.settings, "bot-uid")
    before = list(sender.writes)
    talk.incoming("продолжай", 3, previous_delivered=False)
    assert talk.process(knowledge)
    _drain(talk, sender)
    assert sender.writes[:len(before)] == before
    assert len({item[3] for item in sender.writes}) == len(sender.writes)
    assert all(row.status == "SENT" for row, _ in _rows(talk))


def test_upgrade_recognizes_unambiguous_previous_version_guide_tail(talk, monkeypatch):
    knowledge = Knowledge([approved(2)])
    _approve(monkeypatch, knowledge)
    talk.incoming(QUESTION)
    talk.process(knowledge)
    with talk.env.factory() as session, session.begin():
        room = session.get(TalkRoom, talk.room_id)
        room.profile = {key: value for key, value in room.profile.items() if key != ACTIVE}
    sender = Sender(uid="bot-uid")
    assert deliver_one(talk.env.factory, _settings(talk), sender)
    talk.incoming("Привет", 2, previous_delivered=False)
    assert not deliver_one(talk.env.factory, _settings(talk), sender)
    assert talk.process(knowledge)
    _drain(talk, sender)
    talk.incoming("продолжай", 3, previous_delivered=False)
    assert talk.process(knowledge)
    _drain(talk, sender)
    assert len({item[3] for item in sender.writes}) == len(sender.writes)
    assert all(row.status == "SENT" for row, _ in _rows(talk))


def test_uncertain_send_remains_a_barrier_and_is_not_paused_or_replayed(talk, monkeypatch):
    knowledge = Knowledge([approved(2)])
    _approve(monkeypatch, knowledge)
    talk.incoming(QUESTION)
    talk.process(knowledge)
    sender = Sender(uid="bot-uid", failure=True)
    assert deliver_one(talk.env.factory, _settings(talk), sender)
    second = talk.incoming("Другой вопрос", 2, previous_delivered=False)
    assert not talk.process(knowledge)
    assert not deliver_one(talk.env.factory, _settings(talk), sender)
    with talk.env.factory() as session:
        assert session.get(TalkInbox, second).status == "PENDING"
    rows = _rows(talk)
    assert rows[0][0].status == "UNKNOWN"
    assert all(row.status == "PAUSED" for row, _ in rows[1:])
    assert len(sender.writes) == 1


def test_changed_source_blocks_resume_and_does_not_requeue_old_photographs(talk, monkeypatch):
    knowledge = Knowledge([approved(2)])
    _approve(monkeypatch, knowledge)
    talk.incoming(QUESTION)
    talk.process(knowledge)
    talk.incoming("Привет", 2, previous_delivered=False)
    talk.process(knowledge)
    sender = Sender(uid="bot-uid")
    _drain(talk, sender)
    knowledge.current = False
    talk.incoming("Закончи предыдущий ответ", 3, previous_delivered=False)
    assert talk.process(knowledge)
    _drain(talk, sender)
    assert sender.writes[-1][2] == CHANGED
    assert all(item[0] != "photo" for item in sender.writes)
    assert any(row.status == "PAUSED" for row, _ in _rows(talk))


def test_paused_state_is_bounded(talk, monkeypatch):
    import app.services.chat.talk_dialogue as dialogue

    monkeypatch.setattr(dialogue, "_turn", lambda *args, **kwargs: Turn("REPLY", [("Ответ", None)], flow={}))
    for index in range(1, 6):
        talk.incoming("Новый вопрос", index, previous_delivered=False)
        assert talk.process()
    with talk.env.factory() as session:
        room = session.get(TalkRoom, talk.room_id)
        assert len(room.profile[PAUSED]) == 3 and room.profile[ACTIVE]
    assert sum(row.status == "CANCELLED" for row, _ in _rows(talk)) == 1


@pytest.mark.parametrize("other_uid", ["bot-uid", "different-service"])
def test_tampered_paused_ids_cannot_release_another_room_or_account(talk, other_uid):
    talk.incoming(QUESTION)
    talk.process(Knowledge([approved(2)]))
    other = bind(talk.env.factory, uid=other_uid, employee="different-employee", token="otherroom",
                 base=talk.settings.nextcloud_base_url)
    foreign_id = enqueue(talk.env.factory, other, [("PRIVATE_OTHER_EMPLOYEE", None)])[0]
    with talk.env.factory() as session, session.begin():
        room = session.get(TalkRoom, talk.room_id)
        malicious = {**room.profile[ACTIVE], "delivery_ids": [foreign_id]}
        room.profile = {**room.profile, PAUSED: [malicious]}
        session.get(Delivery, foreign_id).status = "PAUSED"
        assert latest_paused(session, room) is None
        inbox = session.scalar(select(TalkInbox).where(TalkInbox.room_id == room.id))
        assert not resume_response(session, room, session.get(Incident, room.incident_id), inbox, malicious)
        assert session.get(Delivery, foreign_id).status == "PAUSED"


class RefreshSender(Sender):
    def __init__(self, pages, *, failure=False):
        super().__init__(uid="bot-uid")
        self.pages, self.read_failure, self.reads = list(pages), failure, []

    def validate_room(self, token, *, expected_employee=None):
        return {"token": token, "type": 1, "employee_id": expected_employee, "service_id": self.uid}

    def messages(self, token, cursor, limit):
        self.reads.append((token, cursor, limit))
        if self.read_failure:
            raise TimeoutError("private content must not be logged")
        data = self.pages.pop(0) if self.pages else []
        return {"messages": data, "cursor": data[-1]["id"] if data else cursor}


def _remote(talk, number, *, own=False):
    return {"id": number, "token": "private-room", "actorId": "bot-uid" if own else "employee",
            "actorType": "users", "messageType": "comment", "systemMessage": "",
            "timestamp": int(now().timestamp()), "message": "Сканер уже подключён. Нужна другая настройка."}


def test_fresh_read_sees_correction_before_background_poll_and_pauses_before_next_post(talk):
    talk.incoming(QUESTION)
    talk.process(Knowledge([approved(2)]))
    sender = RefreshSender([[_remote(talk, 2)]])
    assert not deliver_one(talk.env.factory, _settings(talk), sender)
    assert not sender.writes and len(sender.reads) == 1
    assert all(row.status == "PAUSED" for row, _ in _rows(talk))
    with talk.env.factory() as session:
        inbox = session.scalar(select(TalkInbox).where(TalkInbox.remote_sequence == 2))
        assert inbox.sender_id == "employee" and inbox.status == "PENDING"


def test_full_read_page_does_not_send_before_the_later_correction_is_fetched(talk):
    talk.incoming(QUESTION)
    talk.process(Knowledge([approved(2)]))
    configured = _settings(talk).model_copy(update={"nextcloud_batch_size": 2})
    sender = RefreshSender([[_remote(talk, 2, own=True), _remote(talk, 3, own=True)], [_remote(talk, 4)]])
    assert not deliver_one(talk.env.factory, configured, sender)
    assert not sender.writes and all(row.status == "PENDING" for row, _ in _rows(talk))
    assert not deliver_one(talk.env.factory, configured, sender)
    assert not sender.writes and all(row.status == "PAUSED" for row, _ in _rows(talk))
    assert [read[1] for read in sender.reads] == [0, 3]


def test_failed_fresh_read_keeps_output_unstarted_and_does_not_make_post_uncertain(talk):
    talk.incoming(QUESTION)
    talk.process(Knowledge([approved(2)]))
    sender = RefreshSender([], failure=True)
    assert not deliver_one(talk.env.factory, _settings(talk), sender)
    assert not sender.writes
    assert all(row.status == "PENDING" and row.tries == 0 for row, _ in _rows(talk))


def test_backlog_room_does_not_starve_another_employee_reply(talk):
    talk.incoming(QUESTION)
    talk.process(Knowledge([approved(2)]))
    other = bind(talk.env.factory, uid="bot-uid", employee="different-employee", token="otherroom",
                 base=talk.settings.nextcloud_base_url)
    other_id = enqueue(talk.env.factory, other, [("Ответ другому сотруднику", None)])[0]
    configured = _settings(talk).model_copy(update={"nextcloud_batch_size": 2})
    sender = RefreshSender([[_remote(talk, 2, own=True), _remote(talk, 3, own=True)], []])
    assert not deliver_one(talk.env.factory, configured, sender)
    assert deliver_one(talk.env.factory, configured, sender)
    assert sender.writes[0][1:3] == ("otherroom", "Ответ другому сотруднику")
    with talk.env.factory() as session:
        assert session.get(Delivery, other_id).status == "SENT"


def test_explicit_success_can_resolve_early_but_unseen_guide_does_not_seed_cache(talk):
    talk.incoming(QUESTION)
    talk.process(Knowledge([approved(8)]))
    talk.incoming("Помогло", 2, previous_delivered=False)
    assert talk.process()
    snapshot = talk.snapshot()
    assert snapshot["incident"].state == "RESOLVED"
    assert snapshot["attempts"][0].status == "SUCCEEDED"
    with talk.env.factory() as session:
        assert session.scalar(select(func.count()).select_from(Feedback)) == 1
        assert session.scalar(select(func.count()).select_from(SuccessfulRetrieval)) == 0
        assert any(row.status == "PAUSED" for row in session.scalars(select(Delivery)))


@pytest.mark.parametrize("message", ["продолжай", "продолжи предыдущий ответ", "Закончи предыдущий ответ",
                                    "пожалуйста, допиши предыдущую инструкцию", "вернись к предыдущему ответу"])
def test_explicit_resume_phrases(message):
    assert is_resume_request(message)


@pytest.mark.parametrize("message", ["Готово", "не продолжай", "продолжить с ботом", "продолжи без фото"])
def test_resume_does_not_capture_feedback_or_modified_requests(message):
    assert not is_resume_request(message)
