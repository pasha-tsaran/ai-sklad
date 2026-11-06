"""Employee photo metadata is acknowledged without vision or approved-KB authority."""

import copy

import pytest
from sqlalchemy import func, select

from app.models import Escalation, Message, MessageAttachment, TalkInbox, TalkRoom
from app.schemas.contracts import DialogueResult
from app.services.chat.talk_dialogue import INTAKE_QUESTIONS, PHOTO_NOTICE
from tests.test_talk_dialogue import (
    QUESTION, Knowledge, NoGeneration, _profile, assistants, talk as talk,
)


def incoming_photo(talk, caption="", sequence=1, *, media=None):
    inbox_id = talk.incoming(caption, sequence)
    with talk.env.factory() as session, session.begin():
        session.get(TalkInbox, inbox_id).media = copy.deepcopy(media if media is not None else {"files": [101]})
    return inbox_id


def waiting_description(talk, model, knowledge):
    _profile(talk)
    talk.settings.mail_backend, talk.settings.support_email = "mailto", "operator@example.net"
    for sequence, content in enumerate(["оператор", "1", "terminal76-sk", "Громкость сканера"], 1):
        talk.incoming(content, sequence)
        assert talk.process(knowledge, model)
    state = talk.snapshot()
    assert state["flow"]["escalation"]["intake_requested_field"] == "additional_info"
    return state


def test_photo_without_caption_is_a_local_reply_and_never_becomes_problem_text_or_knowledge(talk):
    model, knowledge = NoGeneration(), Knowledge()
    inbox_id = incoming_photo(talk)
    assert talk.process(knowledge, model) and not talk.process(knowledge, model)
    state = talk.snapshot()
    assert assistants(state)[-1][0].startswith(PHOTO_NOTICE)
    assert "Опишите словами" in assistants(state)[-1][0]
    assert not model.calls and not knowledge.searches and not knowledge.reads
    assert not state["attempts"] and state["flow"] == {} and state["incident"].description == ""
    assert state["incident"].has_user_photos and state["incident"].failed_attempts == 0
    with talk.env.factory() as session:
        row = session.get(TalkInbox, inbox_id)
        assert row.status == "DONE" and row.tries == 1 and row.media == {"files": [101]}
        assert session.scalar(select(func.count()).select_from(Message)) == 2
        assert session.scalar(select(MessageAttachment)) is None
        assert session.scalar(select(Escalation)) is None


def test_photo_only_does_not_advance_or_replace_a_pending_approved_guide(talk):
    model, knowledge = NoGeneration(), Knowledge()
    talk.incoming(QUESTION)
    talk.process(knowledge, model)
    before = talk.snapshot()
    incoming_photo(talk, sequence=2)
    talk.process(knowledge, model)
    after = talk.snapshot()
    assert after["flow"] == before["flow"] and after["incident"].state == "WAITING_FEEDBACK"
    assert after["attempts"][0].id == before["attempts"][0].id and after["attempts"][0].status == "WAITING_FEEDBACK"
    assert len(assistants(after)) == len(assistants(before)) + 1
    assert "на каком шаге" in assistants(after)[-1][0] and assistants(after)[-1][1] is None
    assert len(knowledge.searches) == 1 and not knowledge.reads and not model.calls


def test_technical_photo_caption_follows_the_same_approved_steps_and_original_photo_order(talk):
    model, knowledge = NoGeneration(), Knowledge()
    incoming_photo(talk, QUESTION)
    talk.process(knowledge, model)
    replies = assistants(talk.snapshot())
    assert replies[0] == (PHOTO_NOTICE, None)
    photos = [photo for _, photo in replies if photo is not None]
    assert len(photos) == 6 and len(talk.snapshot()["attempts"]) == 1
    for number, photo in enumerate(photos, 1):
        assert photo.data == knowledge.documents[0].illustrations[number].data
        position = next(index for index, (_, candidate) in enumerate(replies) if candidate is photo)
        assert photo.caption in replies[position - 1][0]
    assert knowledge.searches[0][0] == QUESTION and not model.calls
    assert all(photo.filename != "101" for photo in photos)


def test_media_ids_and_extra_descriptor_data_never_enter_model_context(talk):
    class Social(NoGeneration):
        def dialogue(self, message, context):
            self.calls.append((message, copy.deepcopy(context)))
            return DialogueResult(route="SOCIAL", reply="Могу поддержать беседу.")

    model, knowledge = Social(), Knowledge()
    caption = "Расскажи что-нибудь весёлое"
    media = {"files": [987654321], "url": "https://never-follow.example.invalid/private.png",
             "caption": "This metadata string must not replace the actual caption"}
    incoming_photo(talk, caption, media=media)
    talk.process(knowledge, model)
    assert len(model.calls) == 1 and model.calls[0][0] == caption
    assert "987654321" not in str(model.calls) and "never-follow" not in str(model.calls)
    assert media["caption"] not in str(model.calls) and not knowledge.searches
    assert assistants(talk.snapshot()) == [(PHOTO_NOTICE, None), ("Могу поддержать беседу.", None)]


def test_photo_only_during_description_keeps_fields_and_waits_for_actual_text(talk):
    model, knowledge = NoGeneration(), Knowledge()
    before = waiting_description(talk, model, knowledge)
    incoming_photo(talk, sequence=5)
    talk.process(knowledge, model)
    state = talk.snapshot()
    assert state["flow"] == before["flow"] and state["incident"].pending_action == "COLLECT_ESCALATION_DETAILS"
    assert assistants(state)[-1][0] == PHOTO_NOTICE + "\n\n" + INTAKE_QUESTIONS["additional_info"]
    assert not model.calls and not knowledge.searches
    with talk.env.factory() as session:
        assert session.scalar(select(Escalation)) is None


def test_photo_only_does_not_cancel_a_pending_device_collector_link(talk):
    model, knowledge = NoGeneration(), Knowledge()
    _profile(talk)
    for sequence, content in enumerate(["оператор", "1"], 1):
        talk.incoming(content, sequence)
        talk.process(knowledge, model)
    with talk.env.factory() as session, session.begin():
        room = session.get(TalkRoom, talk.room_id)
        flow = copy.deepcopy(room.flow)
        flow["escalation"].update(stage="DEVICE", collection_id="existing-private-collection")
        room.flow = flow
    incoming_photo(talk, sequence=3)
    talk.process(knowledge, model)
    stage = talk.snapshot()["flow"]["escalation"]
    assert stage["stage"] == "DEVICE" and stage["collection_id"] == "existing-private-collection"
    assert stage["intake_requested_field"] == "computer_name" and "computer_name" not in stage["intake_fields"]
    assert INTAKE_QUESTIONS["computer_name"] in assistants(talk.snapshot())[-1][0]
    assert not model.calls and not knowledge.searches


def test_photo_caption_is_the_exact_independent_description_and_preview_requires_manual_attachment(talk):
    model, knowledge = NoGeneration(), Knowledge()
    waiting_description(talk, model, knowledge)
    caption = "Сканер считывает код.\n\nЗвуковой сигнал слишком громкий, нужна помощь с его настройкой."
    incoming_photo(talk, caption, 5)
    talk.process(knowledge, model)
    state = talk.snapshot()
    assert state["incident"].state == "ESCALATED" and not model.calls and not knowledge.searches
    with talk.env.factory() as session:
        draft = session.scalar(select(Escalation))
        assert draft.status == "DRAFT" and draft.details["additional_info"] == caption
        assert draft.details["subject"] == "Громкость сканера"
        assert session.scalar(select(MessageAttachment)) is None
    replies = [text for text, _ in assistants(state)]
    preview = next(text for text in replies if text.startswith("Подготовлено письмо для технической поддержки."))
    assert "Прикрепите нужные фотографии в своём почтовом ящике" in preview
    assert PHOTO_NOTICE not in replies  # No duplicate prelude when the preview already explains the limitation.


@pytest.mark.parametrize("error,expected", [
    ("unsupported_inbound_media_type", "Формат вложения"),
    ("inbound_media_too_large", "слишком большое"),
    ("inbound_media_too_many", "слишком много"),
    ("inbound_media_unavailable", "Вложение недоступно"),
    ("inbound_caption_too_long", "Подпись к вложению слишком длинная"),
])
def test_unsupported_or_missing_photo_is_an_honest_local_reply_without_poisoning_the_queue(talk, error, expected):
    model, knowledge = NoGeneration(), Knowledge()
    inbox_id = incoming_photo(talk, media={"files": [], "error_code": error})
    assert talk.process(knowledge, model)
    state = talk.snapshot()
    assert expected in assistants(state)[-1][0] and "вручную" in assistants(state)[-1][0]
    assert "Сообщение с фото получено" not in assistants(state)[-1][0]
    assert state["flow"] == {} and not state["attempts"] and not model.calls and not knowledge.searches
    with talk.env.factory() as session:
        assert session.get(TalkInbox, inbox_id).status == "DONE"
    talk.incoming(QUESTION, 2)
    assert talk.process(knowledge, model) and len(talk.snapshot()["attempts"]) == 1


def test_unsupported_file_does_not_discard_a_caption_that_completes_the_letter(talk):
    model, knowledge = NoGeneration(), Knowledge()
    waiting_description(talk, model, knowledge)
    caption = "Сканер не считывает штрихкод. Нужна помощь с настройкой устройства."
    incoming_photo(talk, caption, 5, media={"files": [], "error_code": "unsupported_inbound_media_type"})
    talk.process(knowledge, model)
    state = talk.snapshot()
    with talk.env.factory() as session:
        assert session.scalar(select(Escalation)).details["additional_info"] == caption
    replies = [text for text, _ in assistants(state)]
    notice = next(index for index, text in enumerate(replies) if "Формат вложения пока не поддерживается" in text)
    preview = next(index for index, text in enumerate(replies) if text.startswith("Подготовлено письмо для технической поддержки."))
    assert notice < preview and "вручную" in replies[notice]
    assert not model.calls and not knowledge.searches


def test_photo_caption_operator_keeps_its_normal_menu_action_and_notice(talk):
    model, knowledge = NoGeneration(), Knowledge()
    incoming_photo(talk, "оператор")
    talk.process(knowledge, model)
    replies = assistants(talk.snapshot())
    assert replies[0] == (PHOTO_NOTICE, None) and "1 —" in replies[1][0]
    assert talk.snapshot()["incident"].pending_action == "COLLECT_ESCALATION_DETAILS"
    assert not model.calls and not knowledge.searches


def test_photo_notice_and_user_caption_rollback_together_before_a_safe_retry(talk, monkeypatch):
    from app.services.chat.talk_dialogue import messenger

    original, calls = messenger.send_blocks, []

    def fail_once(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("Failure before transaction commit")
        return result

    monkeypatch.setattr(messenger, "send_blocks", fail_once)
    model, knowledge = NoGeneration(), Knowledge()
    inbox_id = incoming_photo(talk)
    assert talk.process(knowledge, model)
    with talk.env.factory() as session:
        assert session.get(TalkInbox, inbox_id).status == "PENDING"
        assert session.scalar(select(func.count()).select_from(Message)) == 0
    assert talk.process(knowledge, model)
    with talk.env.factory() as session:
        assert session.get(TalkInbox, inbox_id).status == "DONE"
        assert session.scalar(select(func.count()).select_from(Message)) == 2
    assert not model.calls and not knowledge.searches
