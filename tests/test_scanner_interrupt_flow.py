"""The actual scanner correction interrupts delivery without restarting its guide."""

from sqlalchemy import select

from app.models import Delivery, Message
from app.services.nextcloud.delivery import deliver_one
from tests.test_talk_delivery import Sender
from tests.test_talk_dialogue import QUESTION, Knowledge, NoGeneration, approved
from tests.test_talk_dialogue import talk as talk


def test_connected_scanner_correction_pauses_real_guide_and_resumes_exact_photo_tail(talk, monkeypatch):
    knowledge, model = Knowledge([approved(8)]), NoGeneration()
    monkeypatch.setattr("app.services.rag.talk.TalkKnowledge", lambda settings: knowledge)
    settings = talk.settings.model_copy(update={"nextcloud_backend": "account"})
    sender = Sender(uid="bot-uid")
    talk.incoming(QUESTION)
    assert talk.process(knowledge, model)
    original_flow = talk.snapshot()["flow"]
    with talk.env.factory() as session:
        original_ids = session.scalars(select(Message.id).join(Delivery).order_by(Message.sequence)).all()
    assert deliver_one(talk.env.factory, settings, sender)
    assert deliver_one(talk.env.factory, settings, sender)
    assert [item[0] for item in sender.writes] == ["text", "text"]

    talk.incoming("Сканер уже подключен. Мне надо его настроить", 2, previous_delivered=False)
    assert not deliver_one(talk.env.factory, settings, sender)
    assert talk.process(knowledge, model)
    for _ in range(10):
        if not deliver_one(talk.env.factory, settings, sender):
            break
    assert len(sender.writes) == 3 and sender.writes[-1][0] == "text"
    assert "что" in sender.writes[-1][2].casefold() and "настроить" in sender.writes[-1][2].casefold()
    assert "Операции" not in sender.writes[-1][2]
    assert len(knowledge.searches) == 1 and not model.calls

    talk.incoming("Закончи предыдущий ответ", 3, previous_delivered=False)
    assert talk.process(knowledge, model)
    assert talk.snapshot()["flow"] == original_flow
    for _ in range(50):
        if not deliver_one(talk.env.factory, settings, sender):
            break
    sent_original_ids = [item[3] for item in sender.writes if item[3] in original_ids]
    assert sent_original_ids == original_ids
    assert sender.writes[3][0] == "photo"
    assert len({item[3] for item in sender.writes}) == len(sender.writes)
    assert len(talk.snapshot()["attempts"]) == 1 and len(knowledge.searches) == 1
