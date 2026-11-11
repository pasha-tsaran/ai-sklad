"""Withdrawing a known source cancels only sends which have not started."""

import copy

import pytest
from sqlalchemy import select

from app.models import Delivery, SolutionAttempt, TalkInbox, TalkRoom
from app.services.nextcloud.delivery import deliver_one
from tests.test_talk_delivery import (
    SOURCE, Sender, bind, enqueue, factory as factory, photograph, settings, state,
)
from tests.test_talk_dialogue import QUESTION, Knowledge, approved, talk as talk

WITHDRAWN = "33_Сканер штрихкодов в базе АДРЕС.docx"


def retired_source():
    return {**SOURCE, "filename": WITHDRAWN}


def test_pending_retired_text_and_photo_are_cancelled_before_source_check_or_send(factory, monkeypatch):
    bound = bind(factory)
    enqueue(factory, bound, [("Отозванная инструкция", photograph())], retired_source())
    enqueue(factory, bound, [("Ответ на новый вопрос", None)])

    def unexpected(*args, **kwargs):
        raise AssertionError("A known withdrawal requires no request to the source server")

    monkeypatch.setattr("app.services.rag.talk.TalkKnowledge", unexpected)
    sender = Sender()
    assert deliver_one(factory, settings(), sender)
    assert [(row.status, row.tries) for row, _ in state(factory)] == [
        ("CANCELLED", 0), ("CANCELLED", 0), ("SENT", 1),
    ]
    assert [message.delivery_status for _, message in state(factory)] == ["CANCELLED", "CANCELLED", "DELIVERED_TALK"]
    assert [write[2] for write in sender.writes] == ["Ответ на новый вопрос"]


@pytest.mark.parametrize("uncertain", ["UNKNOWN", "SENDING"])
def test_retired_source_does_not_clear_an_uncertain_external_write(factory, uncertain):
    bound = bind(factory)
    ids = enqueue(factory, bound, [("Возможно уже отправлено", None), ("Ещё не отправлено", None)], retired_source())
    enqueue(factory, bound, [("Поздний ответ", None)])
    with factory() as session, session.begin():
        session.get(Delivery, ids[0]).status = uncertain
    sender = Sender()
    assert not deliver_one(factory, settings(), sender)
    assert [row.status for row, _ in state(factory)] == [uncertain, "CANCELLED", "PENDING"]
    assert not sender.writes


def test_retirement_cancellation_is_scoped_to_the_active_service_account(factory):
    first = bind(factory, uid="old-account", token="old-room")
    second = bind(factory, uid="new-account", token="new-room", employee="employee-two")
    old_ids = enqueue(factory, first, [("Старый аккаунт", None)], retired_source())
    new_ids = enqueue(factory, second, [("Текущий аккаунт", None)], retired_source())
    sender = Sender(uid="new-account")
    assert not deliver_one(factory, settings(), sender)
    with factory() as session:
        assert session.get(Delivery, old_ids[0]).status == "PENDING"
        assert session.get(Delivery, new_ids[0]).status == "CANCELLED"
    assert not sender.writes


def test_old_queued_guide_cannot_mute_the_next_employee_message(talk):
    knowledge = Knowledge([approved(2)])
    talk.incoming(QUESTION)
    assert talk.process(knowledge)
    # Recreate a previously persisted guide before source 33 was withdrawn.
    with talk.env.factory() as session, session.begin():
        room = session.get(TalkRoom, talk.room_id)
        source = {**room.flow["source"], "filename": WITHDRAWN}
        room.flow = {**copy.deepcopy(room.flow), "source": source}
        for row in session.scalars(select(Delivery).where(Delivery.status == "PENDING")):
            row.provenance = dict(source)
        for attempt in session.scalars(select(SolutionAttempt)):
            attempt.evidence = {**attempt.evidence, "talk_source": source}
    sender = Sender(uid="bot-uid")
    configured = talk.settings.model_copy(update={"nextcloud_backend": "account"})
    assert not deliver_one(talk.env.factory, configured, sender)
    assert not sender.writes
    incoming = talk.incoming("Привет", 2, previous_delivered=False)
    assert talk.process(knowledge)
    with talk.env.factory() as session:
        assert session.get(TalkInbox, incoming).status == "DONE"
        assert not session.scalar(select(Delivery.id).where(Delivery.status == "UNKNOWN"))
