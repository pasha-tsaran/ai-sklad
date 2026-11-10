"""A failed generation is terminal work, not a permanent private-room mute."""

from types import SimpleNamespace

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.models import AuditEvent, Delivery, Feedback, Incident, Message, SolutionAttempt, now
from app.models.talk import TalkInbox, TalkRoom
from app.services.chat.talk_dialogue import CHANGED, _safe_database_error
from tests.test_talk_dialogue import DOC, OTHER, QUESTION, Knowledge, approved, assistants
from tests.test_talk_dialogue import talk as talk


def test_revisiting_unassessed_approved_guide_reuses_attempt_instead_of_duplicate_unique_insert(talk):
    knowledge = Knowledge([approved(1, DOC)])
    talk.incoming(QUESTION)
    talk.process(knowledge)
    first_attempt = talk.snapshot()["attempts"][0]
    knowledge.documents = [approved(1, OTHER)]
    talk.incoming(QUESTION + " Настройка на другом рабочем месте.", 2)
    talk.process(knowledge)
    knowledge.documents = [approved(1, DOC)]
    last = talk.incoming(QUESTION, 3)
    assert talk.process(knowledge)
    with talk.env.factory() as session:
        assert session.get(TalkInbox, last).status == "DONE"
        attempts = session.scalars(select(SolutionAttempt).order_by(SolutionAttempt.ordinal)).all()
        assert len(attempts) == 2
        assert attempts[0].id == first_attempt.id and attempts[0].status == "WAITING_FEEDBACK"
        assert attempts[1].status == "UNASSESSED"
        assert session.scalar(select(func.count()).select_from(Feedback)) == 0
        assert session.get(Incident, attempts[0].incident_id).failed_attempts == 0


def test_failed_generation_reports_once_and_next_rapid_message_can_run(talk, monkeypatch):
    import app.services.chat.talk_dialogue as dialogue

    first = talk.incoming(QUESTION, 1, previous_delivered=False)
    second = talk.incoming("Привет", 2, previous_delivered=False)
    original = dialogue._turn

    def failing(message, *args, **kwargs):
        if message == QUESTION:
            raise RuntimeError("private employee details must never appear in diagnostics")
        return original(message, *args, **kwargs)

    monkeypatch.setattr(dialogue, "_turn", failing)
    for _ in range(3):
        assert talk.process()
    with talk.env.factory() as session:
        assert session.get(TalkInbox, first).status == "FAILED"
        assert session.get(TalkInbox, second).status == "PENDING"
    assert talk.process()  # Durable failure notice, without replaying the failed generation.
    notice = talk.snapshot()
    assert len(assistants(notice)) == 1 and "не удалось" in assistants(notice)[0][0].casefold()
    assert not talk.process()  # Deliver that notice before processing the next turn.
    talk.incoming("Спасибо", 3)  # The fixture acknowledges all pending deliveries.
    assert talk.process()
    with talk.env.factory() as session:
        assert session.get(TalkInbox, second).status == "DONE"
        assert session.get(TalkInbox, first).tries == 3
        events = session.scalars(select(AuditEvent).where(AuditEvent.event == "talk.turn_failure_reported")).all()
        assert len(events) == 1
        assert "private employee" not in str([row.details for row in events])


def test_failed_attempt_is_not_reinserted_or_given_duplicate_feedback(talk):
    knowledge = Knowledge([approved(1, DOC)])
    talk.incoming(QUESTION)
    talk.process(knowledge)
    knowledge.missing = True
    talk.incoming("Не помогло", 2)
    talk.process(knowledge)
    talk.incoming("Отмена", 3)
    talk.process(knowledge)
    knowledge.missing = False
    # Defence in depth: even a misbehaving retriever returning an excluded
    # solution must not create a duplicate attempt or duplicate its Feedback.
    knowledge.search = lambda *args, **kwargs: knowledge.documents[0]
    current = talk.incoming(QUESTION, 4)
    assert talk.process(knowledge)
    with talk.env.factory() as session:
        assert session.get(TalkInbox, current).status == "DONE"
        assert session.scalar(select(func.count()).select_from(SolutionAttempt)) == 1
        assert session.scalar(select(func.count()).select_from(Feedback)) == 1
        assert session.scalar(select(SolutionAttempt)).status == "FAILED"


def test_restart_exhausted_running_turn_recovers_previous_flow_and_notifies_once(talk):
    from app.services.chat.talk_dialogue import _claim
    from app.talk_worker import recover

    talk.incoming(QUESTION)
    talk.process(Knowledge([approved(8)]))
    saved = talk.snapshot()
    first = talk.incoming("Готово", 2)
    claimed = _claim(talk.env.factory, talk.settings, talk.account_id)
    with talk.env.factory() as session, session.begin():
        session.get(TalkInbox, first).tries = 3
    recover(talk.env.factory, talk.settings, "bot-uid")
    talk.incoming("Привет", 3, previous_delivered=False)
    assert talk.process()
    after = talk.snapshot()
    assert after["flow"] == saved["flow"]
    assert after["incident"].pending_action is None
    assert after["attempts"][0].id == saved["attempts"][0].id
    assert len(after["attempts"]) == 1
    assert len(assistants(after)) == len(assistants(saved)) + 1
    with talk.env.factory() as session:
        assert session.get(TalkInbox, first).status == "FAILED"
        assert claimed["inbox_id"] == first
        assert session.get(TalkRoom, talk.room_id).flow.get("_processing") is None


def test_changed_historical_guide_never_reactivates_old_attempt_or_posts_new_bytes(talk):
    knowledge = Knowledge([approved(1, DOC)])
    talk.incoming(QUESTION)
    talk.process(knowledge)
    knowledge.documents = [approved(1, OTHER)]
    talk.incoming(QUESTION + " Настройка на другом рабочем месте.", 2)
    talk.process(knowledge)
    before = talk.snapshot()
    changed = approved(1, DOC)
    changed.document["source"]["sha256"] = "b" * 64
    knowledge.documents = [changed]
    last = talk.incoming(QUESTION, 3)
    assert talk.process(knowledge)
    after = talk.snapshot()
    assert after["flow"] == before["flow"]
    assert assistants(after)[-1] == (CHANGED, None)
    assert len(after["attempts"]) == 2
    assert after["attempts"][1].status == "WAITING_FEEDBACK"
    with talk.env.factory() as session:
        assert session.get(TalkInbox, last).status == "DONE"


def test_two_rapid_messages_pause_first_photo_batch_before_second_reply(talk):
    first = talk.incoming(QUESTION, 1, previous_delivered=False)
    second = talk.incoming("Привет", 2, previous_delivered=False)
    assert talk.process(Knowledge([approved(8)]))
    before = talk.snapshot()
    assert talk.process()
    with talk.env.factory() as session:
        assert session.get(TalkInbox, first).status == "DONE"
        assert session.get(TalkInbox, second).status == "DONE"
        deliveries = session.scalars(select(Delivery).join(Message).order_by(Message.sequence)).all()
        assert len(deliveries) > 6 and all(row.status == "PAUSED" for row in deliveries[:-1])
        assert deliveries[-1].status == "PENDING"
    talk.incoming("Спасибо", 3)
    assert talk.process()
    after = talk.snapshot()
    assert after["flow"] == before["flow"]
    def wire(blocks):
        return [(text, (picture.id, picture.sha256, picture.data) if picture else None) for text, picture in blocks]

    assert wire(assistants(after)[:len(assistants(before))]) == wire(assistants(before))
    assert [text for text, _ in assistants(after)[len(assistants(before)):]] == [
        "Здравствуйте! Я помощник технической поддержки. Чем могу помочь?",
        "Пожалуйста! Если нужна помощь, напишите.",
    ]
    assert len(after["attempts"]) == 1
    with talk.env.factory() as session:
        assert session.get(TalkInbox, second).status == "DONE"
        deliveries = session.scalars(select(Delivery).join(Message).order_by(Message.sequence)).all()
        assert all(row.status == "PAUSED" for row in deliveries[:-2])
        assert [row.status for row in deliveries[-2:]] == ["SENT", "PENDING"]


def test_failed_room_with_pending_notice_cannot_starve_another_private_room(talk, monkeypatch):
    import app.services.chat.talk_dialogue as dialogue

    talk.incoming(QUESTION)
    original = dialogue._turn
    monkeypatch.setattr(dialogue, "_turn", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("temporary")))
    for _ in range(3):
        talk.process()
    assert talk.process()  # Report the failed first turn.
    monkeypatch.setattr(dialogue, "_turn", original)
    with talk.env.factory() as session, session.begin():
        room = TalkRoom(account_id=talk.account_id, room_token="another-private-room", employee_id="second-employee")
        session.add(room)
        session.flush()
        other = TalkInbox(room_id=room.id, remote_message_id="20", remote_sequence=20,
                          sender_id="second-employee", content="Привет", occurred_at=now())
        session.add(other)
        session.flush()
        other_id = other.id
    talk.incoming("Привет", 2, previous_delivered=False)
    assert talk.process()
    with talk.env.factory() as session:
        assert session.get(TalkInbox, other_id).status == "DONE"


def test_safe_integrity_diagnostics_keep_identifiers_but_never_sql_or_bindings():
    class DriverError(Exception):
        sqlstate = "23505"
        diag = SimpleNamespace(constraint_name="solution_attempts_incident_id_solution_id_key")

    error = IntegrityError("INSERT SECRET_SQL", {"password": "SECRET_VALUE"}, DriverError("PRIVATE_NAME"))
    assert _safe_database_error(error) == {
        "db_sqlstate": "23505", "db_constraint": "solution_attempts_incident_id_solution_id_key",
    }
    error.orig.diag = SimpleNamespace(constraint_name="PRIVATE_EMPLOYEE_DATA")
    error.orig.sqlstate = "PRIVATE_PASSWORD"
    assert _safe_database_error(error) == {}


def test_knowledge_constructor_failure_is_retried_not_left_running_forever(talk, monkeypatch):
    import app.services.chat.talk_dialogue as dialogue

    def fail(settings):
        raise ValueError("temporary constructor error")

    monkeypatch.setattr(dialogue, "TalkKnowledge", fail)
    inbox = talk.incoming(QUESTION)
    assert dialogue.process_one(talk.env.factory, talk.settings, talk.env.llm, account_id=talk.account_id)
    with talk.env.factory() as session:
        assert session.get(TalkInbox, inbox).status == "PENDING"
        assert session.get(Incident, session.get(TalkRoom, talk.room_id).incident_id).pending_action is None
