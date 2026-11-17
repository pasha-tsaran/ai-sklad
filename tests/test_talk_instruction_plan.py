"""Verified source plans run through the production controller and durable outbox."""

import copy
import io
import json
from zipfile import ZipFile

import httpx
import pytest
from sqlalchemy import select

from app.models import Delivery, Message, MessageAttachment, SuccessfulRetrieval
from app.services.chat.instruction_plan import restore_plan, serialize_plan
from app.services.chat.talk_dialogue import CHANGED
from app.services.rag.success import cache_key
from app.services.llm.openwebui import OpenWebUILLM
from app.services.nextcloud.delivery import deliver_one
from integrations.openwebui.approved_instruction_images import read_docx_images
from tests.test_instruction_planning import selected_part
from tests.test_openwebui_instruction_images import instruction, office, png
from tests.test_talk_dialogue import assistants
from tests.test_talk_dialogue import talk as talk
from tests.test_talk_delivery import Sender
from tests.test_talk_knowledge import DOC, OTHER
from tests.test_talk_model_dialogue import model_chat as model_chat


FIRST = "Измените яркость рабочего терминала с помощью регулятора на фотографии."
SECOND = "Измените размер шрифта приложения АльфаРегистр в окне на фотографии."
REQUEST = "Нужно изменить яркость терминала и размер шрифта в АльфаРегистр."


def planning_chat(model_chat, *, captions=None):
    captions = captions or [FIRST]
    originals = [instruction(captions, image_data=png("green")), instruction([SECOND], image_data=png("blue"))]

    def choose(payload):
        assert payload["instruction_planning"] is True
        return {"result": "PLAN", "parts": [selected_part(payload, OTHER, "g2", SECOND),
                                               selected_part(payload, DOC, "g1", FIRST)],
                "missing_goal_ids": []}

    chat = model_chat([choose], rows=[(DOC, "Терминал.docx", originals[0]),
                                      (OTHER, "АльфаРегистр.docx", originals[1])])
    chat.talk.settings.talk_confirm_understanding = True
    chat.model.supports_instruction_planning = True
    chat.model.supports_source_clarification = True
    chat.model.understand_task = lambda *args: {
        "facts": [{"key": "application", "value": "АльфаРегистр", "evidence": "АльфаРегистр"}],
        "goals": [{"text": "изменить яркость терминала", "evidence": "изменить яркость терминала"},
                  {"text": "размер шрифта в АльфаРегистр", "evidence": "размер шрифта в АльфаРегистр"}],
    }
    return chat, originals


def test_compound_plan_confirms_all_goals_and_preserves_each_original_photo_provenance(model_chat):
    chat, originals = planning_chat(model_chat)
    pending = chat.send(REQUEST)
    saved = pending["flow"]["understanding_confirmation"]["guide_flow"]["instruction_plan"]
    assert [part["source"]["file_id"] for part in saved["parts"]] == [DOC, OTHER]
    assert saved["cursor"] == 0 and not pending["attempts"]
    result = chat.send("Да, правильно", 2)
    photos = [photo for _, photo in assistants(result) if photo]
    assert [photo.data for photo in photos] == [read_docx_images(data)[0].data for data in originals]
    assert len(result["attempts"]) == 1 and not result["flow"]["has_more"]
    assert result["attempts"][0].evidence["instruction_plan"]["digest"] == saved["digest"]
    with chat.talk.env.factory() as session:
        deliveries = session.scalars(select(Delivery).join(Message, Delivery.message_id == Message.id).join(
            MessageAttachment, MessageAttachment.message_id == Message.id,
        ).order_by(Message.sequence)).all()
        assert [delivery.provenance["file_id"] for delivery in deliveries] == [DOC, OTHER]


def test_text_selected_procedure_delivers_its_separate_illustration_appendix_in_original_order(model_chat):
    action = "Для первого входа в СкладОкно откройте приложение и выберите рабочую учётную запись."
    captions = [f"В окне СкладОкно выполните действие номер {number}." for number in range(1, 7)]
    with ZipFile(io.BytesIO(instruction(captions))) as archive:
        parts = {item.filename: archive.read(item.filename) for item in archive.infolist()}
    cover = '<w:p><w:r><w:t>Утверждённая инструкция Acrobat</w:t></w:r></w:p>'
    introduction = (
        '<w:p><w:pPr><w:outlineLvl w:val="0"/></w:pPr>'
        '<w:r><w:t>Первый вход в рабочее приложение</w:t></w:r></w:p>'
        '<w:p><w:pPr><w:outlineLvl w:val="1"/></w:pPr>'
        '<w:r><w:t>Первый вход в СкладОкно</w:t></w:r></w:p>'
        f'<w:p><w:r><w:t>{action}</w:t></w:r></w:p>'
        '<w:p><w:pPr><w:outlineLvl w:val="1"/></w:pPr>'
        '<w:r><w:t>Фотографии и пояснения</w:t></w:r></w:p>'
    )
    document = parts["word/document.xml"].decode()
    assert cover in document
    parts["word/document.xml"] = document.replace(cover, introduction, 1)
    for number, color in enumerate(["red", "green", "blue", "yellow", "purple", "orange"], 1):
        parts[f"word/media/photo{number}.png"] = png(color)
    original = office(parts)

    def choose(payload):
        part = selected_part(payload, DOC, "g1", action)
        assert part["covered_goals"][0]["support_kind"] == "TEXT"
        assert not part["covered_goals"][0]["image_numbers"]
        return {"result": "MATCH", **part}

    chat = model_chat([choose], rows=[(DOC, "Первый вход в рабочее приложение.docx", original)])
    chat.talk.settings.talk_confirm_understanding = True
    chat.model.supports_instruction_planning = True
    chat.model.supports_source_clarification = True
    chat.model.understand_task = lambda *args: {
        "facts": [{"key": "application", "value": "СкладОкно", "evidence": "СкладОкно"}],
        "goals": [{"text": "Первый вход в СкладОкно", "evidence": "Первый вход в СкладОкно"}],
    }
    pending = chat.send("Первый вход в СкладОкно")
    assert pending["flow"].get("understanding_confirmation")
    current = chat.send("Да, правильно", 2)
    assert any(photo for _, photo in assistants(current))
    assert current["flow"]["has_more"]
    previous_cursor = current["flow"]["instruction_plan"]["cursor"]
    for sequence in range(3, 7):
        if not current["flow"]["has_more"]:
            break
        current = chat.send("Продолжить", sequence)
        assert current["flow"]["instruction_plan"]["cursor"] > previous_cursor
        previous_cursor = current["flow"]["instruction_plan"]["cursor"]
    assert not current["flow"]["has_more"]
    assert len(chat.model.calls) == 1 and len(current["attempts"]) == 1
    photographs = [photo.data for _, photo in assistants(current) if photo]
    assert photographs == [photo.data for photo in read_docx_images(original)]
    assert len(photographs) == 6


@pytest.mark.parametrize("changed", [DOC, OTHER])
def test_confirmation_rechecks_every_plan_original_and_blocks_changed_source(model_chat, changed):
    chat, _ = planning_chat(model_chat)
    chat.send(REQUEST)
    chat.state["originals"][changed] = instruction(["Источник обновлён, прежние действия больше не подтверждены."])
    result = chat.send("Да", 2)
    assert assistants(result)[-1] == (CHANGED, None)
    assert not result["attempts"] and not any(photo for _, photo in assistants(result))


def test_long_plan_keeps_cursor_on_help_and_continues_without_restarting_or_reselecting(model_chat):
    captions = [FIRST, *[f"Выполните действие номер {index} в окне рабочего терминала." for index in range(2, 8)]]
    chat, _ = planning_chat(model_chat, captions=captions)
    chat.send(REQUEST)
    first = chat.send("Да", 2)
    cursor = first["flow"]["instruction_plan"]["cursor"]
    assert cursor == 3 and first["flow"]["has_more"]
    chat.model.understand_task = lambda *args: {}
    helped = chat.send("Покажи фото к этому шагу", 3)
    assert helped["flow"]["instruction_plan"]["cursor"] == cursor
    assert len(chat.model.calls) == 1 and len(helped["attempts"]) == 1
    following = chat.send("Готово", 4)
    assert following["flow"]["instruction_plan"]["cursor"] > cursor
    assert len(chat.model.calls) == 1 and len(following["attempts"]) == 1
    assert FIRST not in "\n".join(text for text, _ in assistants(following)[len(assistants(helped)):])


def test_exact_successful_plan_is_reused_after_current_source_checks_without_another_selection(model_chat):
    chat, _ = planning_chat(model_chat)
    chat.send(REQUEST)
    chat.send("Да", 2)
    resolved = chat.send("Помогло, спасибо", 3)
    assert resolved["incident"].state == "RESOLVED"
    with chat.talk.env.factory() as session:
        rows = session.scalars(select(SuccessfulRetrieval)).all()
        assert len(rows) == 2 and all(row.source["kind"] == "verified_instruction_plan" for row in rows)
    repeated = chat.send(REQUEST, 4)
    assert repeated["flow"].get("understanding_confirmation")
    assert len(chat.model.calls) == 1
    sent = chat.send("Да", 5)
    assert len(sent["attempts"]) == 1 and len([photo for _, photo in assistants(sent) if photo]) == 2


def test_vague_successful_opening_recalls_complete_plan_conditions_for_confirmation(model_chat):
    chat, originals = planning_chat(model_chat)
    vague = "Настройка рабочего терминала"
    understand = chat.model.understand_task
    understanding_calls = []

    def task(message, *args):
        understanding_calls.append(message)
        if message == vague:
            return {"question": "Что именно нужно изменить на рабочем месте?", "question_field": "goal"}
        return understand(message, *args)

    chat.model.understand_task = task
    first = chat.send(vague)
    assert first["flow"].get("clarification_question") and not chat.model.calls
    chat.send(REQUEST, 2)
    chat.send("Да", 3)
    chat.send("Помогло", 4)
    recalled = chat.send(vague, 5)
    confirmation = recalled["flow"]["understanding_confirmation"]
    assert len(confirmation["guide_flow"]["instruction_plan"]["parts"]) == 2
    assert "яркость" in confirmation["summary"] and "шрифта" in confirmation["summary"]
    assert "АльфаРегистр" in confirmation["summary"] and "Сообщённые условия" in confirmation["summary"]
    assert len(understanding_calls) == 2 and len(chat.model.calls) == 1
    final = chat.send("Да, правильно", 6)
    assert [photo.data for _, photo in assistants(final) if photo] == [read_docx_images(data)[0].data for data in originals]


def test_rejected_personal_hint_does_not_make_past_conditions_facts_of_new_task(model_chat):
    chat, _ = planning_chat(model_chat)
    chat.send(REQUEST)
    chat.send("Да", 2)
    chat.send("Помогло", 3)
    recalled = chat.send(REQUEST, 4)
    assert recalled["flow"].get("remembered_interpretation")
    rejected = chat.send("Нет", 5)
    state = rejected["flow"]["task_state"]
    assert state["facts"] == [] and state["goals"] == {}
    assert state["messages"] == [REQUEST]


def test_task_signature_partitions_all_conditions_polarity_entities_goals_and_unknowns(model_chat):
    chat, _ = planning_chat(model_chat)
    state = {"facts": [{"key": "condition", "value": "один", "attribute": "число устройств", "entity": "зал"}],
             "goals": {"g1": {"text": "изменить яркость"}, "g2": {"text": "размер шрифта"}},
             "unknown_answers": []}
    original = cache_key(chat.talk.settings, REQUEST, {"task_state": state})
    for field, value in [("negated", True), ("entity", "офис"), ("attribute", "режим"), ("value", "два")]:
        changed = copy.deepcopy(state)
        changed["facts"][0][field] = value
        assert cache_key(chat.talk.settings, REQUEST, {"task_state": changed}) != original
    changed = copy.deepcopy(state)
    changed["goals"] = dict(reversed(list(changed["goals"].items())))
    assert cache_key(chat.talk.settings, REQUEST, {"task_state": changed}) != original
    changed = copy.deepcopy(state)
    changed["unknown_answers"] = [{"question": "Какой режим?", "answer": "Не знаю"}]
    assert cache_key(chat.talk.settings, REQUEST, {"task_state": changed}) != original
    state["facts"][0]["evidence"] = "личная цитата"
    state["messages"] = ["История содержит секрет"]
    assert cache_key(chat.talk.settings, REQUEST, {"task_state": state}) == original


def test_full_task_signature_ignores_condition_arrival_order_and_raw_conversational_wording(model_chat):
    chat, _ = planning_chat(model_chat)
    state = {"facts": [{"key": "connection", "value": "Wi-Fi"},
                       {"key": "condition", "attribute": "число устройств", "value": "два"}],
             "goals": {"g1": {"text": "Подключить устройства"}}}
    first = cache_key(chat.talk.settings, "Два устройства по Wi-Fi", {"task_state": state})
    state["facts"].reverse()
    assert cache_key(chat.talk.settings, "Wi-Fi, устройств два", {"task_state": state}) == first


def test_success_before_all_plan_groups_are_shown_never_trains_a_complete_plan(model_chat):
    captions = [FIRST, *[f"Выполните действие номер {index} в окне рабочего терминала." for index in range(2, 8)]]
    chat, _ = planning_chat(model_chat, captions=captions)
    chat.send(REQUEST)
    partial = chat.send("Да", 2)
    assert partial["flow"]["has_more"]
    resolved = chat.send("Помогло", 3)
    assert resolved["incident"].state == "RESOLVED"
    with chat.talk.env.factory() as session:
        assert not session.scalars(select(SuccessfulRetrieval)).all()


def test_serialized_plan_rejects_changed_sections_and_cursor_does_not_change_identity(model_chat):
    chat, _ = planning_chat(model_chat)
    saved = chat.send(REQUEST)["flow"]["understanding_confirmation"]["guide_flow"]["instruction_plan"]
    knowledge = chat.source_factory(chat.talk.settings)
    restored = restore_plan(knowledge, saved)
    assert serialize_plan(restored)["digest"] == saved["digest"]
    changed = copy.deepcopy(saved)
    changed["cursor"] = 1
    assert restore_plan(knowledge, changed)
    changed["parts"][0]["section_ids"] = ["section-invented"]
    with pytest.raises(Exception, match="InvalidSavedInstructionPlan"):
        restore_plan(knowledge, changed)


@pytest.mark.parametrize("confirmation_enabled", [False, True])
def test_real_reference_adapter_selects_multisource_plan_with_negative_conditions_and_original_photos(
    model_chat, confirmation_enabled,
):
    chat, originals = planning_chat(model_chat)
    chat.talk.settings.talk_confirm_understanding = confirmation_enabled
    opening = REQUEST + " Кабель не используется."
    calls = []

    def handler(request):
        payload = json.loads(json.loads(request.content)["messages"][-1]["content"])
        calls.append(copy.deepcopy(payload))
        if "message" in payload:
            response = {"new_task": False, "facts": [
                {"key": "application", "value": "АльфаРегистр", "evidence": "АльфаРегистр"},
                {"key": "connection", "value": "кабель", "entity": "терминал", "negated": True,
                 "evidence": "Кабель не используется"},
            ], "goals": [
                {"text": "изменить яркость терминала", "evidence": "изменить яркость терминала", "goal_id": None},
                {"text": "размер шрифта в АльфаРегистр", "evidence": "размер шрифта в АльфаРегистр", "goal_id": None},
            ], "question": "Какой протокол использует приложение?", "question_field": "connection"}
        else:
            assert payload["instruction_planning"] is True and payload["evidence_references"] is True
            negative = next(fact for fact in payload["context"]["task_state"]["facts"] if fact.get("negated"))
            assert negative["entity"] == "терминал" and negative["value"] == "кабель"
            parts = []
            for goal_id, needle in [("g1", "яркость"), ("g2", "шрифта")]:
                file_id, reference, proof = next(
                    (file_id, key, item) for file_id, registry in payload["candidates"].items()
                    for key, item in registry.items() if needle in item["text"] and item["support_kind"] == "IMAGE"
                )
                section = next(key for key, value in payload["candidate_sections"][file_id].items()
                               if reference in value["evidence_ids"])
                parts.append({"file_id": file_id, "section_ids": [section],
                              "covered_goals": [{"goal_id": goal_id, "evidence_id": reference}]})
                assert "text" in proof and "evidence" not in parts[-1]["covered_goals"][0]
            response = {"result": "PLAN", "parts": parts, "missing_goal_ids": []}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(response)},
                                                      "finish_reason": "stop"}]})

    adapter = OpenWebUILLM(chat.talk.settings, httpx.MockTransport(handler))
    chat.model.understand_task, chat.model.select_instruction = adapter.understand_task, adapter.select_instruction
    first = chat.send(opening)
    assert len(calls) == 2
    assert bool(first["flow"].get("understanding_confirmation")) == confirmation_enabled
    if confirmation_enabled:
        result = chat.send("Да", 2)
    else:
        result = first
        assert result["flow"]["task_state"]["goals"]
        assert result["flow"]["instruction_plan"]["parts"]
    assert len(calls) == 2 and len(result["attempts"]) == 1
    assert [photo.data for _, photo in assistants(result) if photo] == [read_docx_images(data)[0].data for data in originals]


def test_changed_original_invalidates_successful_plan_and_opening_hint_before_reselection(model_chat):
    chat, _ = planning_chat(model_chat)
    first = chat.send(REQUEST)
    previous_digest = first["flow"]["understanding_confirmation"]["guide_flow"]["instruction_plan"]["digest"]
    chat.send("Да", 2)
    chat.send("Помогло", 3)
    chat.state["originals"][OTHER] = instruction([SECOND, "После настройки сохраните новый размер шрифта."])
    chat.model.responses.append(lambda payload: {"result": "PLAN", "missing_goal_ids": [], "parts": [
        selected_part(payload, DOC, "g1", FIRST), selected_part(payload, OTHER, "g2", SECOND),
    ]})
    repeated = chat.send(REQUEST, 4)
    assert len(chat.model.calls) == 2
    current = repeated["flow"]["understanding_confirmation"]["guide_flow"]["instruction_plan"]
    assert current["digest"] != previous_digest


def test_new_uploaded_instruction_is_found_without_controller_or_question_changes(model_chat):
    chat, _ = planning_chat(model_chat)
    chat.send(REQUEST)
    chat.send("Да", 2)
    chat.send("Помогло", 3)
    new_id = "00000000-0000-4000-8000-000000000099"
    caption = "Для экспорта отчёта в приложении СкладПоток выберите CSV в окне на фотографии."
    original = instruction([caption], image_data=png("purple"))
    chat.state["files"].append({"id": new_id, "filename": "Новый экспорт рабочего отчёта.docx"})
    chat.state["originals"][new_id] = original
    chat.model.understand_task = lambda *args: {
        "facts": [{"key": "application", "value": "СкладПоток", "evidence": "СкладПоток"}],
        "goals": [{"text": "Экспорт отчёта в СкладПоток", "evidence": "Экспорт отчёта в СкладПоток"}],
    }
    chat.model.responses.append(lambda payload: {"result": "MATCH", **selected_part(payload, new_id, "g1", caption)})
    pending = chat.send("Экспорт отчёта в СкладПоток", 4)
    plan = pending["flow"]["understanding_confirmation"]["guide_flow"]["instruction_plan"]
    assert plan["parts"][0]["source"]["file_id"] == new_id
    result = chat.send("Да", 5)
    assert [photo.data for _, photo in assistants(result) if photo] == [read_docx_images(original)[0].data]


def test_same_original_caption_in_purpose_and_illustration_is_delivered_once(model_chat):
    chat, _ = planning_chat(model_chat)
    repeated_text = f"<w:p><w:r><w:t>{FIRST}</w:t></w:r></w:p>"
    chat.state["originals"][DOC] = instruction([FIRST], extra_paragraph=repeated_text)
    pending = chat.send(REQUEST)
    result = chat.send("Да", 2)
    replies = assistants(result)[len(assistants(pending)):]
    assert "\n".join(text for text, _ in replies).count(FIRST) == 1
    assert len([photo for _, photo in replies if photo]) == 2


def test_changed_active_plan_keeps_task_state_then_reselects_current_original(model_chat):
    chat, _ = planning_chat(model_chat)
    chat.send(REQUEST)
    original = chat.send("Да", 2)
    chat.state["originals"][OTHER] = instruction([SECOND, "После настройки сохраните новое значение шрифта."])
    chat.model.understand_task = lambda *args: {}
    changed = chat.send("Покажи фото к этому шагу", 3)
    assert assistants(changed)[-1] == (CHANGED, None)
    assert changed["flow"]["task_state"]["goals"] == original["flow"]["task_state"]["goals"]
    assert "source" not in changed["flow"] and "instruction_plan" not in changed["flow"]
    chat.model.responses.append(lambda payload: {"result": "PLAN", "missing_goal_ids": [], "parts": [
        selected_part(payload, DOC, "g1", FIRST), selected_part(payload, OTHER, "g2", SECOND),
    ]})
    current = chat.send("Продолжить", 4)
    assert current["flow"].get("understanding_confirmation") and len(chat.model.calls) == 2


def test_multiline_original_caption_stays_before_one_original_image(model_chat):
    chat, _ = planning_chat(model_chat)
    caption = FIRST + "\nСледующая строка исходной подписи с условиями настройки."
    chat.state["originals"][DOC] = instruction([caption], image_data=png("green"))
    pending = chat.send(REQUEST)
    result = chat.send("Да", 2)
    replies = assistants(result)[len(assistants(pending)):]
    text = "\n".join(content for content, _ in replies)
    assert text.count(FIRST) == 1 and text.count("Следующая строка исходной подписи") == 1
    assert len([photo for _, photo in replies if photo]) == 2


def test_whitespace_equivalent_caption_does_not_repeat_the_preceding_action_paragraphs(model_chat, monkeypatch):
    chat, _ = planning_chat(model_chat)
    final_action = "Подтвердите выбранный уровень яркости на терминале."
    caption = FIRST + " " + final_action
    with ZipFile(io.BytesIO(instruction([caption], image_data=png("purple")))) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    cover = '<w:p><w:r><w:t>Утверждённая инструкция Acrobat</w:t></w:r></w:p>'
    body = f'<w:p><w:r><w:t>{FIRST}</w:t></w:r></w:p><w:p><w:r><w:t>{final_action}</w:t></w:r></w:p>'
    document = parts["word/document.xml"].decode()
    assert cover in document
    parts["word/document.xml"] = document.replace(cover, body, 1)
    original = office(parts)
    chat.state["originals"][DOC] = original
    pending = chat.send(REQUEST)
    result = chat.send("Да", 2)
    replies = assistants(result)[len(assistants(pending)):]
    text = "\n".join(content for content, _ in replies)
    assert text.count(FIRST) == 1 and text.count(final_action) == 1
    assert [photo.data for _, photo in replies if photo] == [
        read_docx_images(original)[0].data, read_docx_images(chat.state["originals"][OTHER])[0].data,
    ]

    chat.talk.settings.nextcloud_backend = "account"
    monkeypatch.setattr("app.services.rag.talk.TalkKnowledge", chat.source_factory)
    sender = Sender(uid="bot-uid")
    while deliver_one(chat.talk.env.factory, chat.talk.settings, sender):
        pass
    delivered_text = "\n".join(write[2] for write in sender.writes)
    assert delivered_text.count(FIRST) == 1 and delivered_text.count(final_action) == 1
    assert [write[5] for write in sender.writes if write[0] == "photo"] == [
        read_docx_images(original)[0].data, read_docx_images(chat.state["originals"][OTHER])[0].data,
    ]


def goal_order_chat(model_chat, *, confirmation_enabled=True):
    chat, _ = planning_chat(model_chat)
    third = "Установите часовой пояс рабочего терминала по исходной фотографии."
    chat.state["originals"][DOC] = instruction([FIRST, third], image_data=png("green"))
    chat.talk.settings.talk_confirm_understanding = confirmation_enabled
    opening = REQUEST + " Затем установить часовой пояс терминала."
    chat.model.understand_task = lambda *args: {"goals": [
        {"text": "изменить яркость терминала", "evidence": "изменить яркость терминала"},
        {"text": "размер шрифта в АльфаРегистр", "evidence": "размер шрифта в АльфаРегистр"},
        {"text": "установить часовой пояс терминала", "evidence": "установить часовой пояс терминала"},
    ]}

    def choose(payload):
        first = selected_part(payload, DOC, "g1", FIRST)
        final = selected_part(payload, DOC, "g3", third)
        first["covered_goals"] += final["covered_goals"]
        first["section_ids"] += [key for key in final["section_ids"] if key not in first["section_ids"]]
        return {"result": "PLAN", "missing_goal_ids": [], "parts": [
            first, selected_part(payload, OTHER, "g2", SECOND),
        ]}

    chat.model.responses = [choose, choose]
    return chat, opening


@pytest.mark.parametrize("confirmation_enabled", [False, True])
def test_indivisible_goal_order_requires_explicit_consent_then_reselects_once(model_chat, confirmation_enabled):
    chat, opening = goal_order_chat(model_chat, confirmation_enabled=confirmation_enabled)
    offered = chat.send(opening)
    proposal = offered["flow"]["goal_order_offer"]
    assert proposal["goal_order"] == ["g1", "g3", "g2"]
    assert all(goal["text"] in proposal["prompt"] for goal in offered["flow"]["task_state"]["goals"].values())
    assert not offered["attempts"] and not any(photo for _, photo in assistants(offered))
    agreed = chat.send("Да, правильно", 2)
    assert len(chat.model.calls) == 2
    assert list(agreed["flow"]["task_state"]["goals"]) == ["g1", "g3", "g2"]
    assert bool(agreed["flow"].get("understanding_confirmation")) == confirmation_enabled
    if confirmation_enabled:
        assert not agreed["attempts"]
        final = chat.send("Да", 3)
    else:
        final = agreed
    assert len(final["attempts"]) == 1 and len([photo for _, photo in assistants(final) if photo]) == 3
    assert "reselect_source" not in final["flow"]


@pytest.mark.parametrize("answer", ["Нет", "Не знаю"])
def test_declined_or_unknown_goal_reordering_keeps_original_goals_without_repeating_offer(model_chat, answer):
    chat, opening = goal_order_chat(model_chat)
    offered = chat.send(opening)
    declined = chat.send(answer, 2)
    assert declined["flow"]["task_state"]["goals"] == offered["flow"]["task_state"]["goals"]
    assert "goal_order_offer" not in declined["flow"] and not declined["attempts"]
    assert declined["flow"]["rejected_goal_orders"] == [["g1", "g3", "g2"]]
    subsequent = chat.send("Продолжить", 3)
    assert len(chat.model.calls) == 2 and "goal_order_offer" not in subsequent["flow"]
    assert not subsequent["attempts"] and not any(photo for _, photo in assistants(subsequent))


def test_order_consent_rechecks_sources_before_changing_goal_order(model_chat):
    chat, opening = goal_order_chat(model_chat)
    offered = chat.send(opening)
    chat.state["originals"][OTHER] = instruction(["Инструкция изменилась, нужно проверить новые условия."])
    result = chat.send("Да", 2)
    assert assistants(result)[-1] == (CHANGED, None)
    assert result["flow"]["task_state"]["goals"] == offered["flow"]["task_state"]["goals"]
    assert "goal_order_offer" not in result["flow"] and len(chat.model.calls) == 1
