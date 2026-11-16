"""Employee wording from a scanner conversation follows the durable Talk route."""

import pytest
from sqlalchemy import func, select

from app.models import SuccessfulRetrieval
from integrations.openwebui.approved_instruction_images import read_docx_images
from tests.test_openwebui_instruction_images import instruction, png
from tests.test_talk_dialogue import assistants
from tests.test_talk_dialogue import talk as talk
from tests.test_talk_knowledge import DOC, OTHER
from tests.test_talk_model_dialogue import decision
from tests.test_talk_model_dialogue import model_chat as model_chat


VOLUME = "Для увеличения громкости сканера используйте настроечный код максимальной громкости."
SUMMARY = "Увеличить громкость сканера штрихкодов Zebra DS2208"
SOFTWARE = "В программе Альфа настройка устройства выполняется в разделе подключения сканера штрихкодов."
RESET_VOLUME = ("Для сканера штрихкодов Zebra DS2208 предусмотрены заводской сброс и увеличение громкости. "
                "Используйте исходные настроечные коды нужных действий.")


def volume_match():
    return {**decision("MATCH", file_id=OTHER, evidence=VOLUME), "understanding": SUMMARY}


def configure(model_chat, responses):
    photo_document = instruction([VOLUME], image_data=png("blue"))
    chat = model_chat(responses, rows=[
        (DOC, "Настройка подключения сканера в программе Альфа.md", SOFTWARE.encode()),
        (OTHER, "Настройка громкости сканера штрихкодов.docx", photo_document),
    ])
    chat.talk.settings.talk_confirm_understanding = True
    return chat, photo_document


def new_replies(before, after):
    return assistants(after)[len(assistants(before)):]


def test_detailed_rejection_is_processed_even_if_the_optional_interpreter_cannot_classify_it(model_chat):
    software_match = {**decision("MATCH", file_id=DOC, evidence=SOFTWARE),
                      "understanding": "Добавить сканер штрихкодов в программе Альфа"}
    question = "Какой параметр самого сканера нужно изменить?"
    chat, _ = configure(model_chat, [software_match, decision("CLARIFY", question=question)])
    first = chat.send("Нужно добавить сканер штрихкодов в программе Альфа")
    assert first["flow"]["understanding_confirmation"]["source"]["file_id"] == DOC
    chat.model.interpret_understanding_reply = lambda message, understanding: {"result": "UNCLEAR"}
    correction = "нет. Сканер уже подключен к программе, надо сам сканер настроить, я его взял с коробки"
    corrected = chat.send(correction, 2)
    assert len(chat.model.calls) == 2
    assert correction in chat.model.calls[-1]["payload"]["actual_problem"]
    assert new_replies(first, corrected) == [(question, None)]
    assert not corrected["attempts"] and corrected["incident"].failed_attempts == 0


@pytest.mark.parametrize("answer", ["да, правильно", "да, правильно понял", "Да, всё верно понял"])
def test_common_confirmation_phrases_release_the_original_photo_without_another_model_call(model_chat, answer):
    chat, data = configure(model_chat, [volume_match()])
    first = chat.send("Увеличить громкость сканера штрихкодов Zebra DS2208")

    def unexpected_interpretation(*args):
        pytest.fail("A common unambiguous confirmation does not need model interpretation")

    chat.model.interpret_understanding_reply = unexpected_interpretation
    accepted = chat.send(answer, 2)
    photos = [photo for _, photo in new_replies(first, accepted) if photo]
    assert [photo.data for photo in photos] == [read_docx_images(data)[0].data]
    assert accepted["flow"]["source"]["file_id"] == OTHER
    assert len(chat.model.calls) == 1 and len(accepted["attempts"]) == 1


@pytest.mark.parametrize("message", [
    "Да, удалось, спасибо. Все правильно",
    "Да, удалось, спасибо. Всё правильно!",
    "Всё получилось, спасибо, теперь работает",
])
def test_natural_success_closes_the_incident_without_searching_the_knowledge_base_again(model_chat, message):
    chat, _ = configure(model_chat, [volume_match()])
    chat.send("Увеличить громкость сканера штрихкодов Zebra DS2208")
    accepted = chat.send("да", 2)
    calls_before = len(chat.model.calls)
    result = chat.send(message, 3)
    assert result["incident"].state == "RESOLVED"
    assert len(chat.model.calls) == calls_before
    assert result["attempts"][0].status == "SUCCEEDED"
    replies = new_replies(accepted, result)
    assert replies and all(photo is None for _, photo in replies)
    assert any("обратную связь" in text for text, _ in replies)


@pytest.mark.parametrize("answer", [
    "и то и другое надо настроить",
    "сначала надо сбросить к заводским настройкам, потом настроить громкость",
])
def test_answering_both_goals_does_not_repeat_the_same_exclusive_question(model_chat, answer):
    question = "Нужно настроить громкость сканера или выполнить заводской сброс?"
    data = instruction([RESET_VOLUME], image_data=png("blue"))
    full_match = {**decision("MATCH", file_id=OTHER, evidence=RESET_VOLUME),
                  "understanding": "Сбросить настройки сканера и увеличить громкость"}
    chat = model_chat([decision("CLARIFY", question=question)], rows=[
        (OTHER, "Заводской сброс и громкость сканера.docx", data),
    ])
    chat.talk.settings.talk_confirm_understanding = True
    first = chat.send("Нужно настроить сам сканер штрихкодов Zebra DS2208")
    previous_question = assistants(first)[-1][0]
    # A repeated model question must be repaired or replaced before reaching the employee.
    chat.model.responses = [decision("CLARIFY", question=previous_question), full_match]
    result = chat.send(answer, 2)
    replies = new_replies(first, result)
    assert replies and all(text != previous_question for text, _ in replies)
    assert not result["attempts"] and not any(photo for _, photo in replies)


def scanner_task_updates(chat):
    messages = [
        "Настройка сканера",
        "Zebra DS2208",
        "Другая, мне надо сам сканер настроить. Он уже подключен к 1С, я его взял с коробки",
        "Сначала надо сбросить к заводским настройкам, потом увеличить громкость",
    ]
    reset_goal = "Сбросить к заводским настройкам"
    volume_goal = "Увеличить громкость"
    updates = [
        {"facts": [{"key": "equipment_type", "value": "сканер", "evidence": "сканера"}],
         "question": "Это сканер штрихкодов или сканер документов?", "question_field": "equipment_type"},
        {"facts": [
            {"key": "equipment_type", "value": "сканер штрихкодов", "evidence": "Zebra DS2208"},
            {"key": "equipment_model", "value": "Zebra DS2208", "evidence": "Zebra DS2208"},
        ], "question": "Что нужно настроить у самого сканера?", "question_field": "goal"},
        {"facts": [
            {"key": "scope", "value": "hardware", "evidence": "сам сканер настроить"},
            {"key": "application", "value": "1С", "evidence": "1С"},
            {"key": "completed", "value": "Сканер уже подключен к 1С", "evidence": "Он уже подключен к 1С"},
        ], "question": "Какие параметры самого сканера нужно изменить?", "question_field": "goal"},
        {"goals": [
            {"text": reset_goal, "evidence": "сбросить к заводским настройкам"},
            {"text": volume_goal, "evidence": "увеличить громкость"},
        ]},
    ]
    calls = []

    def understand(message, previous, previous_question):
        index = len(calls)
        assert message == messages[index]
        calls.append({"message": message, "previous": previous, "previous_question": previous_question})
        return updates[index]

    chat.model.understand_task = understand
    return messages, calls, reset_goal, volume_goal


def test_compound_hardware_task_keeps_user_facts_and_sends_only_verified_partial_help(model_chat):
    def partial(payload):
        assert payload["requested_goals"] == {"g1": "Сбросить к заводским настройкам", "g2": "Увеличить громкость"}
        assert "Zebra DS2208" in payload["actual_problem"]
        assert "1С" not in payload["actual_problem"]
        return {
            "result": "PARTIAL", "file_id": OTHER,
            "covered_goals": [{"goal_id": "g2", "evidence": VOLUME,
                               "support_kind": "IMAGE", "image_numbers": [1]}],
            "missing_goal_ids": ["g1"],
        }

    chat, data = configure(model_chat, [partial])
    messages, task_calls, reset_goal, volume_goal = scanner_task_updates(chat)
    questions = [
        "Это сканер штрихкодов или сканер документов?",
        "Что нужно настроить у самого сканера?",
        "Какие параметры самого сканера нужно изменить?",
    ]
    previous = {"messages": []}
    for sequence, message in enumerate(messages[:-1], 1):
        progress = chat.send(message, sequence)
        assert new_replies(previous, progress) == [(questions[sequence - 1], None)]
        assert not progress["attempts"] and not any(photo for _, photo in assistants(progress))
        assert not chat.model.calls
        previous = progress
    pending = chat.send(messages[-1], 4)
    state = pending["flow"]["task_state"]
    assert state["goals"] == {
        "g1": {"text": reset_goal, "evidence": "сбросить к заводским настройкам"},
        "g2": {"text": volume_goal, "evidence": "увеличить громкость"},
    }
    assert any(fact["key"] == "scope" and fact["value"] == "hardware" for fact in state["facts"])
    assert any(fact["key"] == "completed" and "1С" in fact["value"] for fact in state["facts"])
    guide = pending["flow"]["understanding_confirmation"]["guide_flow"]
    assert guide["coverage_mode"] == "partial"
    assert list(guide["covered_goals"]) == [volume_goal]
    assert list(guide["missing_goals"]) == [reset_goal]
    assert not pending["attempts"] and not any(photo for _, photo in assistants(pending))
    confirmation_text = "\n".join(text for text, _ in assistants(pending)).casefold()
    assert "сброс" in confirmation_text and "громк" in confirmation_text
    accepted = chat.send("да, правильно", 5)
    replies = new_replies(pending, accepted)
    photos = [photo for _, photo in replies if photo]
    assert [photo.data for photo in photos] == [read_docx_images(data)[0].data]
    assert accepted["flow"]["coverage_mode"] == "partial"
    assert not accepted["flow"].get("retrieval_key")
    boundary_text = "\n".join(text for text, _ in replies).casefold()
    assert "сброс" in boundary_text and any(word in boundary_text for word in ("не найден", "не наш", "отсутств", "нет", "не подтвержд"))
    source_calls = len(chat.state["calls"])
    completed = chat.send("Да, удалось, спасибо. Все правильно", 6)
    assert completed["incident"].state != "RESOLVED"
    assert completed["attempts"][0].status == "SUCCEEDED"
    assert completed["flow"]["coverage_mode"] == "partial_completed"
    assert completed["flow"]["missing_goals"] == [reset_goal]
    assert completed["flow"]["task_state"]["goals"] == {
        "g1": {"text": reset_goal, "evidence": "сбросить к заводским настройкам"},
    }
    assert len(chat.state["calls"]) == source_calls
    completion_text = "\n".join(text for text, _ in new_replies(accepted, completed)).casefold()
    assert "сброс" in completion_text and "спасибо" in completion_text
    assert len(task_calls) == 4 and len(chat.model.calls) == 1
    with chat.talk.env.factory() as session:
        assert session.scalar(select(func.count()).select_from(SuccessfulRetrieval)) == 0
    thanked = chat.send("Спасибо, теперь получилось", 7)
    assert len(chat.state["calls"]) == source_calls
    assert thanked["incident"].state == "DIAGNOSING" and len(thanked["attempts"]) == 1
    assert thanked["flow"]["coverage_mode"] == "partial_completed"
    repeated = chat.send("покажи фото", 8)
    photos = [photo for _, photo in new_replies(thanked, repeated) if photo]
    assert [photo.data for photo in photos] == [read_docx_images(data)[0].data]
    assert len(repeated["attempts"]) == 1 and repeated["attempts"][0].status == "SUCCEEDED"
    assert repeated["incident"].state == "DIAGNOSING"
    assert repeated["incident"].pending_action is None
    assert repeated["flow"]["missing_goals"] == [reset_goal]
    assert len(task_calls) == 4 and len(chat.model.calls) == 1


def test_software_reference_without_a_photo_cannot_prove_hardware_programming_codes(model_chat):
    def wrong_match(payload):
        return {
            "result": "MATCH", "file_id": DOC,
            "covered_goals": [{"goal_id": goal_id, "evidence": SOFTWARE,
                               "support_kind": "IMAGE", "image_numbers": [1]}
                              for goal_id in payload["requested_goals"]],
        }

    chat, _ = configure(model_chat, [wrong_match])
    messages, _, _, _ = scanner_task_updates(chat)
    for sequence, message in enumerate(messages, 1):
        result = chat.send(message, sequence)
    assert not result["attempts"] and not any(photo for _, photo in assistants(result))
    assert "understanding_confirmation" not in result["flow"]
    assert not any(SOFTWARE in text for text, _ in assistants(result))


def test_known_hardware_goal_skips_an_unnecessary_application_question(model_chat):
    def full_match(payload):
        assert payload["requested_goals"] == {"g1": "Увеличить громкость"}
        return {
            "result": "MATCH", "file_id": OTHER,
            "covered_goals": [{"goal_id": "g1", "evidence": VOLUME,
                               "support_kind": "IMAGE", "image_numbers": [1]}],
        }

    chat, data = configure(model_chat, [full_match])
    unnecessary_question = "Для какой программы нужно настроить сканер?"
    chat.model.understand_task = lambda message, previous, previous_question: {
        "facts": [
            {"key": "equipment_type", "value": "сканер штрихкодов", "evidence": "сканера штрихкодов"},
            {"key": "scope", "value": "hardware", "evidence": "самого сканера"},
        ],
        "goals": [{"text": "Увеличить громкость", "evidence": "Увеличить громкость"}],
        "question": unnecessary_question, "question_field": "application",
    }
    pending = chat.send("Увеличить громкость самого сканера штрихкодов")
    assert "understanding_confirmation" in pending["flow"]
    assert not any(unnecessary_question in text for text, _ in assistants(pending))
    assert len(chat.model.calls) == 1 and not pending["attempts"]
    accepted = chat.send("да, правильно", 2)
    photos = [photo for _, photo in new_replies(pending, accepted) if photo]
    assert [photo.data for photo in photos] == [read_docx_images(data)[0].data]


def test_legacy_clarification_seeds_task_understanding_with_prior_user_messages_only(model_chat):
    original = "Настройка сканера штрихкодов"
    wrong_assumption = "Вам нужна настройка сканера в программе 1С?"

    def full_match(payload):
        assert "1С" not in payload["actual_problem"]
        return {
            "result": "MATCH", "file_id": OTHER,
            "covered_goals": [{"goal_id": "g1", "evidence": VOLUME,
                               "support_kind": "IMAGE", "image_numbers": [1]}],
        }

    chat, _ = configure(model_chat, [decision("CLARIFY", question=wrong_assumption), full_match])
    legacy = chat.send(original)
    assert "task_state" not in legacy["flow"]
    assert assistants(legacy) == [(wrong_assumption, None)]
    task_calls = []

    def understand(message, previous, previous_question):
        task_calls.append(previous)
        assert original in previous["messages"]
        assert all("1С" not in text for text in previous["messages"])
        assert previous_question == wrong_assumption
        return {
            "facts": [
                {"key": "equipment_type", "value": "сканер штрихкодов", "evidence": "сканера штрихкодов"},
                {"key": "scope", "value": "hardware", "evidence": "самого устройства"},
            ],
            "goals": [{"text": "Увеличить громкость", "evidence": "увеличить громкость"}],
        }

    chat.model.understand_task = understand
    corrected = chat.send("Нет, нужно увеличить громкость самого устройства", 2)
    assert len(task_calls) == 1 and len(chat.model.calls) == 2
    assert "understanding_confirmation" in corrected["flow"]
    assert not corrected["attempts"]
    assert all("1С" not in text for text in corrected["flow"]["task_state"]["messages"])


def test_selecting_a_related_source_rechecks_each_goal_before_partial_confirmation(model_chat):
    goals = {"g1": "Сбросить к заводским настройкам", "g2": "Увеличить громкость"}

    def offer(payload):
        assert payload["requested_goals"] == goals
        return {"result": "RELATED", "alternatives": [{"file_id": OTHER, "evidence": VOLUME}]}

    def partial(payload):
        assert payload["requested_goals"] == goals
        assert payload["context"]["verified_previous_source"]["file_id"] == OTHER
        return {
            "result": "PARTIAL", "file_id": OTHER,
            "covered_goals": [{"goal_id": "g2", "evidence": VOLUME,
                               "support_kind": "IMAGE", "image_numbers": [1]}],
            "missing_goal_ids": ["g1"],
        }

    chat, data = configure(model_chat, [offer, partial])
    messages, task_calls, _, _ = scanner_task_updates(chat)
    for sequence, message in enumerate(messages, 1):
        offered = chat.send(message, sequence)
    assert "instruction_offer" in offered["flow"]
    assert not offered["attempts"] and not any(photo for _, photo in assistants(offered))
    selected = chat.send("1", 5)
    guide = selected["flow"]["understanding_confirmation"]["guide_flow"]
    assert guide["coverage_mode"] == "partial"
    assert guide["missing_goals"] == [goals["g1"]]
    assert len(chat.model.calls) == 2 and len(task_calls) == 4
    assert not selected["attempts"] and not any(photo for _, photo in assistants(selected))
    accepted = chat.send("да, правильно", 6)
    photos = [photo for _, photo in new_replies(selected, accepted) if photo]
    assert [photo.data for photo in photos] == [read_docx_images(data)[0].data]
    assert accepted["flow"]["coverage_mode"] == "partial" and not accepted["flow"].get("retrieval_key")


def test_photo_request_with_a_new_technical_condition_is_understood_before_source_navigation(model_chat):
    hardware_photo = instruction([VOLUME], image_data=png("blue"))
    software_photo = instruction([SOFTWARE], image_data=png("green"))

    def hardware_match(payload):
        return {
            "result": "MATCH", "file_id": OTHER,
            "covered_goals": [{"goal_id": "g1", "evidence": VOLUME,
                               "support_kind": "IMAGE", "image_numbers": [1]}],
        }

    def software_match(payload):
        assert "Подключить сканер к другой программе" in payload["actual_problem"]
        assert "громкость" not in payload["actual_problem"]
        return {
            "result": "MATCH", "file_id": DOC,
            "covered_goals": [{"goal_id": "g1", "evidence": SOFTWARE,
                               "support_kind": "IMAGE", "image_numbers": [1]}],
        }

    chat = model_chat([hardware_match, software_match], rows=[
        (OTHER, "Настройка громкости сканера штрихкодов.docx", hardware_photo),
        (DOC, "Подключение сканера штрихкодов к программе Альфа.docx", software_photo),
    ])
    chat.talk.settings.talk_confirm_understanding = True
    task_calls = []

    def understand(message, previous, previous_question):
        task_calls.append(message)
        facts = [{"key": "equipment_type", "value": "сканер штрихкодов", "evidence": "сканера штрихкодов"}]
        if len(task_calls) == 1:
            return {
                "facts": facts + [{"key": "scope", "value": "hardware", "evidence": "самого сканера"}],
                "goals": [{"text": "Увеличить громкость", "evidence": "Увеличить громкость"}],
            }
        return {
            "new_task": True,
            "facts": facts + [{"key": "scope", "value": "application", "evidence": "другой программе"}],
            "goals": [{"text": "Подключить сканер к другой программе",
                       "evidence": "подключения сканера штрихкодов к другой программе"}],
        }

    chat.model.understand_task = understand
    chat.send("Увеличить громкость самого сканера штрихкодов")
    shown = chat.send("да", 2)
    assert shown["flow"]["source"]["file_id"] == OTHER
    next_message = "Покажи фото подключения сканера штрихкодов к другой программе"
    pending = chat.send(next_message, 3)
    assert task_calls[-1] == next_message and len(task_calls) == 2
    assert len(chat.model.calls) == 2
    confirmation = pending["flow"]["understanding_confirmation"]
    assert confirmation["source"]["file_id"] == DOC
    assert not any(photo for _, photo in new_replies(shown, pending))
    accepted = chat.send("да, правильно", 4)
    photos = [photo for _, photo in new_replies(pending, accepted) if photo]
    assert [photo.data for photo in photos] == [read_docx_images(software_photo)[0].data]
    assert accepted["flow"]["source"]["file_id"] == DOC
