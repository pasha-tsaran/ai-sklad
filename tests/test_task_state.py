"""Task understanding retains user facts and ordered goals independently of sources."""

import copy
import json

import httpx
import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.schemas.contracts import SupportTaskUnderstanding
from app.services.chat.task_state import (
    is_unknown_answer, remember_task_message, starts_new_task, task_query, update_task_state,
)
from app.services.llm.openwebui import OpenWebUILLM
from app.services.llm.prompts import SUPPORT_TASK_TASK


class UnderstandingModel:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def understand_task(self, message, previous_state, previous_question):
        self.calls.append((message, copy.deepcopy(previous_state), previous_question))
        return self.responses.pop(0)


def goal(text, evidence=None, goal_id=None):
    return {"text": text, "evidence": evidence or text, "goal_id": goal_id}


def fact(key, value, evidence=None):
    return {"key": key, "value": value, "evidence": evidence or value}


@pytest.mark.parametrize("opening,answer,target", [
    ("Настройка сканера", "Уменьшить громкость сканера штрихкодов", "Уменьшить громкость сканера штрихкодов"),
    ("Нужно настроить принтер", "Настроить печать этикеток на принтере", "Настроить печать этикеток на принтере"),
])
def test_initial_goal_answer_keeps_its_pending_frame_even_if_the_model_marks_it_new(opening, answer, target):
    question = "Какой результат настройки вам нужен?"
    model = UnderstandingModel({"new_task": True, "question": question, "question_field": "goal"},
                               {"new_task": True, "goals": [goal(target, answer)]})
    previous = update_task_state(opening, None, "", model)
    result = update_task_state(answer, previous, question, model)
    assert result["new_task"] is False and result["messages"] == [opening, answer]
    assert result["goals"]["g1"]["text"] == target


@pytest.mark.parametrize("answer", ["Мне нужен первый вход в Outlook", "Другая задача. Уменьшить громкость сканера"])
def test_a_different_application_or_explicit_new_task_can_replace_an_empty_goal_frame(answer):
    question = "Какой результат настройки сканера вам нужен?"
    model = UnderstandingModel({"question": question, "question_field": "goal"},
                               {"new_task": True, "goals": [goal(answer)]})
    previous = update_task_state("Настройка сканера", None, "", model)
    result = update_task_state(answer, previous, question, model)
    assert result["new_task"] is True and result["messages"] == [answer]


def test_a_shared_computer_does_not_turn_a_new_outlook_topic_into_a_scanner_goal_answer():
    opening = "Нужно настроить сканер на компьютере"
    answer = "Первый вход в Outlook на этом компьютере"
    question = "Какой результат настройки сканера нужен?"
    model = UnderstandingModel({"question": question, "question_field": "goal"},
                               {"new_task": True, "goals": [goal(answer)]})
    previous = update_task_state(opening, None, "", model)
    result = update_task_state(answer, previous, question, model)
    assert result["new_task"] is True and result["messages"] == [answer]


def compound_task():
    message = "Zebra DS2208: сначала сбросить настройки, затем увеличить громкость"
    model = UnderstandingModel({
        "facts": [fact("equipment_model", "Zebra DS2208")],
        "goals": [goal("сбросить настройки"), goal("увеличить громкость")],
    })
    return update_task_state(message, None, "", model)


def test_explicit_goal_kind_and_discovery_aliases_survive_state_roundtrip_without_becoming_query_facts():
    message = "Нужен внутренний номер руководства склада"
    aliases = ["внутренний телефон руководство", "warehouse management extension number"]
    state = update_task_state(message, None, "", UnderstandingModel({"goals": [{
        **goal("Найти телефон руководства", message), "kind": "LOOKUP", "search_queries": aliases,
    }]}))
    assert state["goals"]["g1"]["kind"] == "LOOKUP"
    assert state["goals"]["g1"]["search_queries"] == aliases
    query = task_query(state)
    assert query == "Найти телефон руководства" and aliases[1] not in query
    continued = remember_task_message("Ещё нужен номер", state)
    assert continued["goals"] == state["goals"]
    updated = update_task_state("Руководство именно склада", continued, "", UnderstandingModel({
        "goals": [goal("Найти телефон руководства склада", "Руководство именно склада", "g1")],
    }))
    assert updated["goals"]["g1"]["kind"] == "LOOKUP" and updated["goals"]["g1"]["search_queries"] == aliases


def test_explicit_action_kind_stays_distinguishable_from_a_legacy_unspecified_goal():
    message = "Открыть рабочую почту"
    legacy = update_task_state(message, None, "", UnderstandingModel({"goals": [goal(message)]}))
    explicit = update_task_state(message, None, "", UnderstandingModel({"goals": [{
        **goal(message), "kind": "ACTION", "search_queries": [],
    }]}))
    assert legacy["goals"]["g1"] == {"text": message, "evidence": message}
    assert explicit["goals"]["g1"] == {
        "text": message, "evidence": message, "kind": "ACTION", "search_queries": [],
    }
    assert remember_task_message("Не получилось", explicit)["goals"] == explicit["goals"]


def test_explicit_goal_kind_update_replaces_previous_lookup_metadata():
    first = "Узнать громкость"
    state = update_task_state(first, None, "", UnderstandingModel({"goals": [{
        **goal(first), "kind": "LOOKUP", "search_queries": ["beeper volume"],
    }]}))
    later = "Теперь хочу уменьшить громкость"
    state = update_task_state(later, state, "", UnderstandingModel({"goals": [{
        **goal("Уменьшить громкость", later, "g1"), "kind": "ACTION", "search_queries": ["decrease volume"],
    }]}))
    assert state["goals"]["g1"]["kind"] == "ACTION"
    assert state["goals"]["g1"]["search_queries"] == ["decrease volume"]


@pytest.mark.parametrize("queries", [["x"] * 4, ["x" * 161], [""], ["one\ntwo"]])
def test_goal_discovery_aliases_are_bounded_and_single_line(queries):
    with pytest.raises(ValidationError):
        SupportTaskUnderstanding.model_validate({"goals": [{
            **goal("Найти телефон"), "kind": "LOOKUP", "search_queries": queries,
        }]})


def test_guide_context_is_passed_only_to_explicitly_capable_providers_and_not_to_new_tasks():
    guide = {"sources": [{"sections": [{"text": "Откройте приложение."}]}]}
    legacy = UnderstandingModel({})
    state = update_task_state("На этом шаге ошибка", None, "", legacy, guide_context=guide)
    assert len(legacy.calls) == 1 and not state["facts"]

    class GuideModel(UnderstandingModel):
        supports_guide_context = True

        def understand_task(self, message, previous_state, previous_question, *, guide_context=None):
            self.guide = guide_context
            return super().understand_task(message, previous_state, previous_question)

    capable = GuideModel({})
    update_task_state("На этом шаге ошибка", state, "", capable, guide_context=guide)
    assert capable.guide == guide
    new_task = GuideModel({})
    update_task_state("Другая задача. Открыть почту", state, "", new_task, guide_context=guide)
    assert new_task.guide is None


@pytest.mark.parametrize("message", [
    "Другая задача. Нужно открыть программу.", "Новая проблема: нужно открыть программу.",
    "Следующая тема — открыть программу", "Теперь другой вопрос", "А теперь новая задача",
    "У меня новый вопрос", "Пожалуйста, сейчас другая проблема",
    "У меня теперь другой вопрос", "Теперь у меня новая задача",
    "Перейдём к другой задаче", "Давайте переключимся на новую тему",
    "Забудь предыдущую задачу. Нужно открыть программу.",
    "Отмените текущую проблему, теперь нужно открыть программу.",
])
def test_explicit_topic_boundaries_are_general_conversation_requests(message):
    assert starts_new_task(message)


@pytest.mark.parametrize("message", [
    "Другая, мне надо сам сканер настроить", "Это не новая задача",
    "Не другая задача, продолжаем прежнюю", "На четвёртом шаге ошибка",
    "А теперь увеличить громкость", "Нужно и то и другое", "не знаю", "да, правильно", None,
])
def test_topic_boundaries_do_not_swallow_ordinary_followups_or_negation(message):
    assert not starts_new_task(message)


@pytest.mark.parametrize("reported_new_task", [None, False])
def test_explicit_new_task_uses_fresh_model_input_even_when_model_omits_the_transition(reported_new_task):
    previous = compound_task()
    previous.update(question="Какой режим подключения используется?", question_field="connection")
    message = "Другая задача. Хочу открыть рабочую почту."
    response = {"goals": [goal("Открыть рабочую почту", "открыть рабочую почту", "g1")]}
    if reported_new_task is not None:
        response["new_task"] = reported_new_task
    model = UnderstandingModel(response)
    state = update_task_state(message, previous, previous["question"], model)
    _, model_previous, model_question = model.calls[0]
    assert model_previous["facts"] == [] and model_previous["goals"] == {}
    assert model_previous["messages"] == [] and model_question == ""
    assert state["new_task"] is True and state["facts"] == []
    assert state["messages"] == [message]
    assert list(state["goals"].values()) == [{"text": "Открыть рабочую почту", "evidence": "открыть рабочую почту"}]


def test_explicit_boundaries_keep_outlook_scanner_and_printer_as_separate_tasks():
    model = UnderstandingModel(
        {"facts": [fact("application", "Outlook")], "goals": [goal("Открыть почту", "открыть почту")]},
        {"facts": [fact("equipment_type", "сканер", "сканера")],
         "question": "Какой тип сканера требуется настроить?", "question_field": "equipment_type"},
        {"facts": [fact("equipment_type", "принтер", "принтеру")],
         "goals": [goal("Подключить ТСД к принтеру TSC по Wi-Fi", "Подключить ТСД к принтеру TSC по Wi-Fi")]},
    )
    state = update_task_state("Хочу открыть почту в Outlook", None, "", model)
    state = update_task_state("Другая задача. Настройка сканера", state, "", model)
    assert not state["goals"] and "Outlook" not in task_query(state)
    state = update_task_state("Другая задача. Подключить ТСД к принтеру TSC по Wi-Fi", state, state["question"], model)
    assert [item["text"] for item in state["goals"].values()] == ["Подключить ТСД к принтеру TSC по Wi-Fi"]
    assert "Outlook" not in task_query(state) and "сканер" not in task_query(state)
    assert all(call[1]["goals"] == {} and call[2] == "" for call in model.calls[1:])


def test_explicit_topic_boundary_does_not_authorize_a_repeated_old_user_fact():
    previous = compound_task()
    model = UnderstandingModel({"facts": [fact("equipment_model", "Zebra DS2208")]})
    with pytest.raises(ValueError, match="InvalidTaskEvidence"):
        update_task_state("Новая задача. Нужно открыть рабочую почту.", previous, "", model)


def test_vague_request_retains_a_single_type_question_without_inventing_a_goal():
    model = UnderstandingModel({
        "question": "Нужно настроить сканер штрихкодов или сканер документов?",
        "question_field": "equipment_type",
    })
    state = update_task_state("Настройка сканера", None, "", model)
    assert state["facts"] == [] and state["goals"] == {}
    assert state["question_field"] == "equipment_type"
    assert state["messages"] == ["Настройка сканера"]
    assert len(model.calls) == 1 and state["new_task"] is False


def test_known_model_and_hardware_scope_do_not_turn_completed_1c_setup_into_a_goal():
    model = UnderstandingModel(
        {"facts": [fact("equipment_model", "Zebra DS2208"),
                   fact("equipment_type", "сканер штрихкодов", "Zebra DS2208")],
         "question": "Что нужно настроить у самого сканера?", "question_field": "goal"},
        {"facts": [fact("scope", "hardware", "сам сканер настроить"),
                   fact("application", "1С"),
                   fact("completed", "Подключён к 1С", "Сканер уже подключен к 1С")],
         "question": "Какой результат настройки вам нужен?", "question_field": "goal"},
        {"goals": [goal("сбросить к заводским настройкам"), goal("увеличить громкость")]},
    )
    state = update_task_state("Zebra DS2208", None, "Какая программа: 1С или другая?", model)
    state = update_task_state(
        "Сканер уже подключен к 1С, надо сам сканер настроить", state, state["question"], model,
    )
    state = update_task_state(
        "сначала сбросить к заводским настройкам, потом увеличить громкость", state, state["question"], model,
    )
    assert [item["text"] for item in state["goals"].values()] == [
        "сбросить к заводским настройкам", "увеличить громкость",
    ]
    query = task_query(state)
    assert "1С" not in query and "Подключён" not in query and "hardware" not in query
    assert "Zebra DS2208" in query and "сканер штрихкодов" in query
    assert state["question"] == "" and state["question_field"] is None


def test_question_text_cannot_supply_evidence_for_a_user_fact():
    model = UnderstandingModel({"facts": [fact("application", "1С")]})
    with pytest.raises(ValueError, match="InvalidTaskEvidence"):
        update_task_state("Zebra DS2208", None, "В программе 1С или другой?", model)


@pytest.mark.parametrize("message", ["оба", "Оба!", "и то и другое", "и то и другое надо настроить"])
def test_both_goals_are_retained_without_another_model_call(message):
    previous = compound_task()
    unchanged = copy.deepcopy(previous)
    model = UnderstandingModel()
    state = update_task_state(message, previous, "Громкость или сброс?", model)
    assert state["goals"] == previous["goals"] and model.calls == []
    assert state["messages"][-1] == message and state["question"] == ""
    assert previous == unchanged


@pytest.mark.parametrize("message", [
    "не знаю", "Не знаю!", "я не знаю", "не разбираюсь", "Я в этом не разбираюсь",
    "не могу определить", "не могу сказать", "без понятия", "понятия не имею",
    "не уверен", "не уверена", "затрудняюсь ответить", "не помню",
    "не понимаю вопрос", "Не знаю, помогите", "не знаю, что это значит",
    "Не знаю, я в этом не разбираюсь.", "Я не знаю и не могу определить.",
    "Ну, я не знаю. Не понимаю вопрос.", "К сожалению, не помню, честно.",
])
def test_unknown_answer_keeps_every_goal_and_fact_without_a_model_call(message):
    previous = compound_task()
    question = "Какой способ подключения используется?"
    previous.update(question=question, question_field="connection")
    unchanged = copy.deepcopy(previous)
    model = UnderstandingModel()
    state = update_task_state(message, previous, question, model)
    assert is_unknown_answer(message)
    assert model.calls == [] and state["goals"] == previous["goals"]
    assert state["facts"] == previous["facts"] and state["messages"][-1] == message
    assert state["question"] == "" and state["question_field"] is None
    assert state["unknown_answers"] == [{"question": question, "answer": message, "field": "connection"}]
    assert previous == unchanged and state["new_task"] is False


def test_unknown_answer_to_a_legacy_question_does_not_copy_an_unrelated_old_field():
    previous = compound_task()
    previous.update(question="Какой результат нужен?", question_field="goal")
    question = "Какой параметр программы используется?"
    state = update_task_state("не знаю", previous, question, UnderstandingModel())
    assert state["unknown_answers"] == [{"question": question, "answer": "не знаю", "field": None}]
    assert state["goals"] == previous["goals"]


def test_unknown_without_an_asked_question_still_uses_normal_task_understanding():
    model = UnderstandingModel({})
    state = update_task_state("не знаю", None, "", model)
    assert len(model.calls) == 1 and state["unknown_answers"] == []
    assert state["facts"] == [] and state["goals"] == {}


@pytest.mark.parametrize("message", [
    "не знаю, но подключено по USB", "не знаю, теперь другая задача",
    "не понимаю, почему не работает", "не знаю название, устройство пищит",
    "Не знаю, я в этом не разбираюсь, но подключено по USB.",
    "Не знаю, и теперь другая задача: хочу подключить почту.",
    "Не знаю пароль", "Не уверен, можно ли сбрасывать сканер",
    "Exchange", "да", "нет", "сначала сброс, потом громкость", None,
])
def test_unknown_shortcut_does_not_swallow_additional_conditions_or_another_task(message):
    assert is_unknown_answer(message) is False


def test_unknown_message_with_a_concrete_condition_is_interpreted_and_keeps_the_condition():
    previous = compound_task()
    question = "Как подключено устройство?"
    previous.update(question=question, question_field="connection")
    model = UnderstandingModel({"facts": [fact("connection", "USB")]})
    state = update_task_state("не знаю, но подключено по USB", previous, question, model)
    assert len(model.calls) == 1 and state["unknown_answers"] == []
    assert any(item["key"] == "connection" and item["value"] == "USB" for item in state["facts"])
    assert state["goals"] == previous["goals"]


def test_unknown_records_are_bounded_deduplicated_and_preserve_the_latest_answer():
    state = compound_task()
    for index in range(12):
        question = f"Какой параметр {index} используется?"
        state.update(question=question, question_field="other")
        state = update_task_state("не знаю", state, question, UnderstandingModel())
        assert len(state["unknown_answers"]) <= 8
    assert len(state["unknown_answers"]) == 8
    assert state["unknown_answers"][0]["question"] == "Какой параметр 4 используется?"
    state = update_task_state("Я не знаю", state, "Какой параметр 11 используется?", UnderstandingModel())
    assert len(state["unknown_answers"]) == 8
    assert state["unknown_answers"][-1]["answer"] == "Я не знаю"
    assert state["unknown_answers"][-1]["field"] == "other"


def test_a_later_reported_condition_replaces_unknown_status_for_that_field():
    previous = compound_task()
    previous.update(question="Как подключено устройство?", question_field="connection")
    state = update_task_state("не знаю", previous, previous["question"], UnderstandingModel())
    state = update_task_state("Посмотрел, USB", state, previous["question"], UnderstandingModel({
        "facts": [fact("connection", "USB")],
    }))
    assert state["unknown_answers"] == []
    assert state["goals"] == previous["goals"]
    assert any(item["key"] == "connection" and item["value"] == "USB" for item in state["facts"])


def test_a_different_question_in_the_same_coarse_field_keeps_the_prior_unknown_answer():
    previous = compound_task()
    question = "Какой режим обмена выбран в программе?"
    previous.update(question=question, question_field="connection")
    unknown = update_task_state("не знаю", previous, question, UnderstandingModel())
    state = update_task_state("Вижу провод USB", unknown, "Что видно на кабеле?", UnderstandingModel({
        "facts": [fact("connection", "USB")],
    }))
    assert state["unknown_answers"] == unknown["unknown_answers"]
    assert any(item["key"] == "connection" and item["value"] == "USB" for item in state["facts"])
    assert "USB" in task_query(state)


def test_reemitting_an_old_fact_does_not_cancel_a_reported_unknown_answer():
    previous = update_task_state("USB подключение", None, "", UnderstandingModel({
        "facts": [fact("connection", "USB")],
    }))
    previous.update(question="Какое соединение используется сейчас?", question_field="connection")
    unknown = update_task_state("не помню", previous, previous["question"], UnderstandingModel())
    state = update_task_state("так и есть", unknown, "", UnderstandingModel({
        "facts": [fact("connection", "USB")],
    }))
    assert state["unknown_answers"] == unknown["unknown_answers"]


def test_a_new_task_clears_unknown_conditions_from_the_previous_task():
    previous = compound_task()
    previous.update(question="Какой параметр используется?", question_field="other")
    previous = update_task_state("не знаю", previous, previous["question"], UnderstandingModel())
    state = update_task_state("Нужно открыть файл в новой программе Бета", previous, "", UnderstandingModel({
        "new_task": True,
        "facts": [fact("application", "Бета")],
        "goals": [goal("Открыть файл в программе Бета", "открыть файл")],
    }))
    assert state["unknown_answers"] == [] and state["new_task"] is True
    assert "параметр" not in task_query(state)


def test_new_application_names_are_preserved_without_a_catalogue_of_programs():
    message = "В новой программе Гамма хочу получить список всех поставок"
    state = update_task_state(message, None, "", UnderstandingModel({
        "facts": [fact("application", "Гамма"), fact("scope", "application", "В новой программе Гамма")],
        "goals": [goal("Получить список всех поставок", "получить список всех поставок")],
    }))
    assert state["question"] == "" and state["unknown_answers"] == []
    assert "Гамма" in task_query(state) and "Получить список" in task_query(state)


@pytest.mark.parametrize("unknown", [
    {}, None, [None], [{"question": ""}], [{"question": "x" * 301}],
    [{"question": "Вопрос?", "field": "administrator"}],
    [{"question": "Вопрос?", "answer": "x" * 301}],
    [{"question": f"Вопрос {index}?"} for index in range(9)],
])
def test_corrupt_unknown_metadata_is_rejected(unknown):
    with pytest.raises(ValueError):
        update_task_state("ответ", {"unknown_answers": unknown}, "", UnderstandingModel())


def test_old_state_without_unknown_metadata_remains_compatible():
    previous = {"facts": [], "goals": {"g1": {"text": "Открыть файл", "evidence": "Открыть файл"}},
                "messages": ["Открыть файл"]}
    state = remember_task_message("в программе", previous)
    assert state["unknown_answers"] == [] and state["goals"] == previous["goals"]


def test_free_form_inability_to_answer_is_retained_by_a_user_evidenced_model_delta():
    previous = compound_task()
    question = "Какой параметр используется?"
    previous.update(question=question, question_field="connection")
    message = "Эти сокращения мне ни о чём не говорят, я обычный сотрудник"
    model = UnderstandingModel({"unknown_answer_evidence": "Эти сокращения мне ни о чём не говорят",
                                "question": question, "question_field": "connection"})
    state = update_task_state(message, previous, question, model)
    assert len(model.calls) == 1 and not is_unknown_answer(message)
    assert state["goals"] == previous["goals"] and state["facts"] == previous["facts"]
    assert state["unknown_answers"] == [{"question": question, "field": "connection",
                                        "answer": "Эти сокращения мне ни о чём не говорят"}]
    assert state["question"] == "" and state["question_field"] is None
    assert state["messages"][-1] == message and state["new_task"] is False


def test_free_form_unknown_answer_keeps_a_useful_supplement_without_inventing_the_missing_setting():
    previous = compound_task()
    question = "Какой режим соединения настроен в программе?"
    previous.update(question=question, question_field="connection")
    message = "В этих настройках я совершенно не понимаю, вижу только кабель USB"
    state = update_task_state(message, previous, question, UnderstandingModel({
        "unknown_answer_evidence": "В этих настройках я совершенно не понимаю",
        "facts": [fact("connection", "USB")],
    }))
    assert state["unknown_answers"] == [{"question": question, "field": "connection",
                                        "answer": "В этих настройках я совершенно не понимаю"}]
    assert any(item["key"] == "connection" and item["value"] == "USB" for item in state["facts"])
    assert state["goals"] == previous["goals"] and "режим" not in task_query(state)


def test_semantic_unknown_answer_can_keep_a_reported_goal_refinement():
    previous = compound_task()
    question = "Какой режим используется?"
    previous.update(question=question, question_field="connection")
    state = update_task_state("С режимом не разберусь, громкость нужна максимальная", previous, question,
                              UnderstandingModel({
                                  "unknown_answer_evidence": "С режимом не разберусь",
                                  "goals": [goal("максимальная громкость", "максимальная", "g2")],
                              }))
    assert list(state["goals"]) == ["g1", "g2"]
    assert state["goals"]["g1"] == previous["goals"]["g1"]
    assert state["goals"]["g2"]["text"] == "максимальная громкость"
    assert len(state["unknown_answers"]) == 1


@pytest.mark.parametrize("question,update", [
    ("Какой режим?", {"unknown_answer_evidence": "не знаю"}),
    ("", {"unknown_answer_evidence": "не умею определить"}),
    (None, {"unknown_answer_evidence": "не умею определить"}),
    ("Какой режим?", {"unknown_answer_evidence": "Какой режим?"}),
    ("Какой режим?", {"new_task": True, "unknown_answer_evidence": "не умею определить"}),
])
def test_invalid_semantic_unknown_delta_does_not_change_the_previous_task(question, update):
    previous = compound_task()
    unchanged = copy.deepcopy(previous)
    with pytest.raises(ValueError, match="InvalidTaskUnknownAnswer"):
        update_task_state("Сам не умею определить", previous, question, UnderstandingModel(update))
    assert previous == unchanged


def test_semantic_unknown_evidence_cannot_be_reused_from_an_old_message():
    previous = compound_task()
    previous["messages"].append("Эти настройки мне непонятны")
    with pytest.raises(ValueError, match="InvalidTaskUnknownAnswer"):
        update_task_state("да", previous, "Какой режим?", UnderstandingModel({
            "unknown_answer_evidence": "Эти настройки мне непонятны",
        }))


def test_explicit_new_task_after_an_unknown_answer_clears_old_question_and_values():
    previous = compound_task()
    previous.update(question="Какой режим?", question_field="connection")
    previous = update_task_state("В этих параметрах не разберусь", previous, previous["question"], UnderstandingModel({
        "unknown_answer_evidence": "В этих параметрах не разберусь",
    }))
    state = update_task_state("Теперь нужно открыть файл в программе Дельта", previous, "Какой режим?",
                              UnderstandingModel({"new_task": True, "facts": [fact("application", "Дельта")],
                                                  "goals": [goal("Открыть файл", "открыть файл")]}))
    assert state["unknown_answers"] == [] and state["new_task"] is True
    assert state["messages"] == ["Теперь нужно открыть файл в программе Дельта"]


def test_updating_one_goal_preserves_the_other_goal_and_its_order():
    previous = compound_task()
    unchanged = copy.deepcopy(previous)
    model = UnderstandingModel({"goals": [goal("максимальная громкость", "максимальная", "g2")]})
    state = update_task_state("пусть будет максимальная", previous, "Какой уровень громкости?", model)
    assert list(state["goals"]) == ["g1", "g2"]
    assert state["goals"]["g1"] == previous["goals"]["g1"]
    assert state["goals"]["g2"]["text"] == "максимальная громкость"
    assert previous == unchanged


def test_repeating_the_same_goal_without_an_id_preserves_its_stable_id():
    previous = compound_task()
    model = UnderstandingModel({"goals": [goal("увеличить громкость")]})
    state = update_task_state("да, увеличить громкость", previous, "", model)
    assert list(state["goals"]) == ["g1", "g2"]


@pytest.mark.parametrize("include_ids", [True, False])
def test_explicit_full_goal_sequence_updates_order_while_retaining_stable_ids(include_ids):
    previous = compound_task()
    previous["goals"] = {key: previous["goals"][key] for key in ("g2", "g1")}
    model = UnderstandingModel({"goals": [
        goal("сбросить настройки", goal_id="g1" if include_ids else None),
        goal("увеличить громкость", goal_id="g2" if include_ids else None),
    ]})
    state = update_task_state("сначала сбросить настройки, потом увеличить громкость", previous, "", model)
    assert list(state["goals"]) == ["g1", "g2"]
    assert task_query(state).endswith("сбросить настройки\nувеличить громкость")
    assert list(previous["goals"]) == ["g2", "g1"]


def test_incremental_goal_update_retains_the_existing_sequence():
    previous = compound_task()
    previous["goals"] = {key: previous["goals"][key] for key in ("g2", "g1")}
    state = update_task_state("сбросить настройки", previous, "", UnderstandingModel({
        "goals": [goal("сбросить настройки", goal_id="g1")],
    }))
    assert list(state["goals"]) == ["g2", "g1"]


def test_explicit_replacement_keeps_retained_goal_ids_and_removes_only_cancelled_goals():
    previous = compound_task()
    model = UnderstandingModel({
        "replace_goals": True, "replacement_evidence": "Сброс больше не нужен",
        "goals": [goal("увеличить громкость", goal_id="g2")],
    })
    state = update_task_state("Сброс больше не нужен, увеличить громкость", previous, "", model)
    assert list(state["goals"]) == ["g2"]


@pytest.mark.parametrize("reply,error", [
    ({"replace_goals": True, "replacement_evidence": "пользователь отменил"}, "InvalidTaskReplacement"),
    ({"replacement_evidence": "да"}, "UnexpectedTaskReplacement"),
    ({"goals": [goal("увеличить громкость", goal_id="g9")]}, "UnknownTaskGoal"),
    ({"facts": [fact("application", "Несуществующая программа")]}, "InvalidTaskEvidence"),
])
def test_invalid_deltas_leave_prior_state_unchanged(reply, error):
    previous = compound_task()
    unchanged = copy.deepcopy(previous)
    with pytest.raises(ValueError, match=error):
        update_task_state("да", previous, "", UnderstandingModel(reply))
    assert previous == unchanged


def test_a_new_task_resets_facts_goals_and_user_history():
    state = update_task_state("Не печатает PDF", compound_task(), "", UnderstandingModel({
        "new_task": True, "goals": [goal("Не печатает PDF")],
    }))
    assert state["new_task"] is True and state["facts"] == []
    assert state["goals"] == {"g1": {"text": "Не печатает PDF", "evidence": "Не печатает PDF"}}
    assert state["messages"] == ["Не печатает PDF"]


def test_a_new_task_cannot_reuse_old_message_evidence():
    with pytest.raises(ValueError, match="InvalidTaskEvidence"):
        update_task_state("Не печатает PDF", compound_task(), "", UnderstandingModel({
            "new_task": True, "facts": [fact("equipment_model", "Zebra DS2208")],
        }))


@pytest.mark.parametrize("returned_ids", [("g1", "g2"), ("g2", "g1"), ("g9", "g9")])
def test_a_new_task_assigns_fresh_ids_independently_of_the_model_returned_identifiers(returned_ids):
    previous = compound_task()
    previous.update(question="Какой режим используется?", question_field="connection")
    previous = update_task_state("не знаю", previous, previous["question"], UnderstandingModel())
    previous["source"] = {"file_id": "untrusted-old-source"}
    unchanged = copy.deepcopy(previous)
    message = "В программе Вектор хочу открыть корпоративную почту и дождаться первых писем"
    state = update_task_state(message, previous, "Какой режим используется?", UnderstandingModel({
        "new_task": True,
        "facts": [fact("application", "Вектор")],
        "goals": [goal("Открыть корпоративную почту", "открыть корпоративную почту", returned_ids[0]),
                  goal("Дождаться первых писем", "дождаться первых писем", returned_ids[1])],
    }))
    assert state["goals"] == {
        "g1": {"text": "Открыть корпоративную почту", "evidence": "открыть корпоративную почту"},
        "g2": {"text": "Дождаться первых писем", "evidence": "дождаться первых писем"},
    }
    assert state["facts"] == [fact("application", "Вектор")]
    assert state["messages"] == [message] and state["unknown_answers"] == []
    assert state["new_task"] is True and "source" not in state
    assert previous == unchanged


@pytest.mark.parametrize("reply", [
    {"facts": [fact("equipment_model", "Zebra DS2208")]},
    {"goals": [goal("Открыть корпоративную почту", "Настроить почту", "g1")]},
])
def test_new_task_id_normalization_keeps_strict_current_message_evidence_validation(reply):
    previous = compound_task()
    unchanged = copy.deepcopy(previous)
    with pytest.raises(ValueError, match="InvalidTaskEvidence"):
        update_task_state("Хочу открыть корпоративную почту", previous, "", UnderstandingModel({
            "new_task": True, **reply,
        }))
    assert previous == unchanged


def test_current_task_still_rejects_a_goal_id_that_was_never_assigned():
    previous = compound_task()
    with pytest.raises(ValueError, match="UnknownTaskGoal"):
        update_task_state("увеличить громкость", previous, "", UnderstandingModel({
            "goals": [goal("увеличить громкость", "увеличить громкость", "g9")],
        }))


def test_real_task_adapter_can_switch_tasks_when_the_model_reuses_a_prior_goal_identifier():
    previous = compound_task()
    message = "Хочу открыть корпоративную почту в новой программе"
    observed = []

    def handler(request):
        body = json.loads(request.content)
        allowed = body["response_format"]["json_schema"]["schema"]["$defs"]["SupportTaskGoal"]["properties"]["goal_id"]
        assert "g2" in allowed["anyOf"][0]["enum"]
        observed.append(json.loads(body["messages"][-1]["content"]))
        reply = {"new_task": True, "facts": [], "question": "", "question_field": None,
                 "goals": [{"goal_id": "g2", "text": "Открыть корпоративную почту",
                           "evidence": "открыть корпоративную почту", "kind": "ACTION", "search_queries": []}]}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(reply)},
                                                      "finish_reason": "stop"}]})

    model = OpenWebUILLM(model_settings(), httpx.MockTransport(handler))
    state = update_task_state(message, previous, "", model)
    assert len(observed) == 1
    assert state["goals"] == {"g1": {"text": "Открыть корпоративную почту", "evidence": "открыть корпоративную почту",
                                      "kind": "ACTION", "search_queries": []}}
    assert state["facts"] == [] and state["messages"] == [message]


def test_history_is_bounded_and_retains_current_message():
    state = None
    for number in range(20):
        message = f"Сообщение {number} " + "я" * 600
        state = update_task_state(message, state, "", UnderstandingModel({}))
        assert len(state["messages"]) <= 12 and sum(map(len, state["messages"])) <= 6000
        assert state["messages"][-1] == message
    state = update_task_state("я" * 6000, state, "", UnderstandingModel({}))
    assert state["messages"] == ["я" * 6000]


def test_failure_memory_preserves_employee_wording_without_changing_facts_or_goals():
    previous = compound_task()
    unchanged = copy.deepcopy(previous)
    state = remember_task_message("На 4 шаге неверный пароль", previous)
    assert state["messages"] == [*previous["messages"], "На 4 шаге неверный пароль"]
    assert state["facts"] == previous["facts"] and state["goals"] == previous["goals"]
    assert previous == unchanged and state["new_task"] is False
    continued = update_task_state("причем в браузере зашел нормально", state, "", UnderstandingModel({
        "facts": [fact("error_text", "неверный пароль"),
                   fact("completed", "В браузере вход выполнен", "в браузере зашел нормально")],
    }))
    assert any(item["key"] == "error_text" for item in continued["facts"])
    assert continued["goals"] == previous["goals"]


def test_failure_memory_is_bounded_and_accepts_an_empty_previous_state():
    state = None
    for number in range(20):
        message = f"Ошибка {number} " + "я" * 600
        state = remember_task_message(message, state)
        assert len(state["messages"]) <= 12 and sum(map(len, state["messages"])) <= 6000
        assert state["messages"][-1] == message
        assert state["facts"] == [] and state["goals"] == {}


@pytest.mark.parametrize("message", [None, "", "  ", "я" * 6001], ids=["null", "empty", "blank", "oversized"])
def test_failure_memory_rejects_invalid_messages(message):
    with pytest.raises(ValueError, match="InvalidTaskMessage"):
        remember_task_message(message)


def test_validated_fact_evidence_survives_pruned_message_history():
    previous = compound_task()
    previous["messages"] = ["Более позднее уточнение"]
    state = update_task_state("да", previous, "", UnderstandingModel({
        "facts": [fact("equipment_model", "Zebra DS2208")],
    }))
    assert state["facts"][0]["value"] == "Zebra DS2208"


def test_background_application_is_included_only_when_the_goal_itself_requires_it():
    state = update_task_state("В 1С настроить обмен", None, "", UnderstandingModel({
        "facts": [fact("application", "1С")], "goals": [goal("В 1С настроить обмен")],
    }))
    assert task_query(state) == "В 1С настроить обмен"


def test_connections_of_distinct_devices_and_negative_conditions_survive_followups():
    message = "Первый терминал по Wi-Fi, второй по USB. Bluetooth не используется. Нужен обмен."
    first = update_task_state(message, None, "", UnderstandingModel({
        "facts": [
            {**fact("connection", "Wi-Fi"), "entity": "Первый терминал"},
            {**fact("connection", "USB"), "entity": "второй"},
            {**fact("connection", "Bluetooth", "Bluetooth не используется"), "negated": True},
        ], "goals": [goal("Нужен обмен")],
    }))
    later = update_task_state("На первом теперь кабель USB", first, "", UnderstandingModel({
        "facts": [{**fact("connection", "USB", "кабель USB"), "entity": "Первый терминал"}],
    }))
    assert [(item.get("entity", ""), item["value"], item.get("negated", False))
            for item in later["facts"]] == [("второй", "USB", False), ("", "Bluetooth", True),
                                           ("Первый терминал", "USB", False)]
    query = task_query(later)
    assert "Не Bluetooth" in query and "Первый терминал" in query and "второй" in query
    assert later["goals"] == first["goals"]


def test_future_properties_preserve_independent_attributes_and_explicit_polarity_changes():
    state = update_task_state("Папка личная, архив отключён. Нужно открыть отчёт.", None, "", UnderstandingModel({
        "facts": [{**fact("condition", "личная"), "attribute": "область папки"},
                  {**fact("condition", "архив", "архив отключён"), "attribute": "режим", "negated": True}],
        "goals": [goal("открыть отчёт")],
    }))
    changed = update_task_state("Теперь архив включён", state, "", UnderstandingModel({
        "facts": [{**fact("condition", "архив", "архив включён"), "attribute": "режим"}],
    }))
    assert changed["facts"] == [state["facts"][0],
                                 {"key": "condition", "value": "архив", "evidence": "архив включён",
                                  "attribute": "режим"}]
    assert "область папки" in task_query(changed) and "Не архив" not in task_query(changed)


def test_default_fact_extensions_preserve_preexisting_serialized_shape():
    state = update_task_state("USB. Открыть папку", None, "", UnderstandingModel({
        "facts": [fact("connection", "USB")], "goals": [goal("Открыть папку")],
    }))
    assert state["facts"] == [{"key": "connection", "value": "USB", "evidence": "USB"}]


def test_error_and_followup_messages_retain_the_current_goal_step_and_scope():
    messages = [
        "Нужно загрузить файл в Nextcloud",
        "На 4 шаге выдает ошибку",
        "неверный пароль",
        "причем в браузере зашел нормально",
        "Я же писал уже тебе что нужно сделать, все то же самое",
    ]
    model = UnderstandingModel(
        {"facts": [fact("application", "Nextcloud"), fact("scope", "application", "загрузить файл в Nextcloud")],
         "goals": [goal("Загрузить файл в Nextcloud", "загрузить файл в Nextcloud")]},
        {"facts": [fact("current_step", "Шаг 4", "4 шаге"),
                   fact("symptoms", "Ошибка при загрузке файла", messages[1])],
         "question": "Какой текст ошибки появляется на этом шаге?", "question_field": "error_text"},
        {"facts": [fact("error_text", "неверный пароль")]},
        {"facts": [fact("completed", "В браузере вход выполнен", "в браузере зашел нормально")]},
        {},
    )
    state = None
    for message in messages:
        state = update_task_state(message, state, state["question"] if state else "", model)
        assert state["new_task"] is False
    assert state["goals"] == {
        "g1": {"text": "Загрузить файл в Nextcloud", "evidence": "загрузить файл в Nextcloud"},
    }
    values = {item["key"]: item["value"] for item in state["facts"]}
    assert values["scope"] == "application"
    assert values["current_step"] == "Шаг 4" and values["error_text"] == "неверный пароль"
    assert values["completed"] == "В браузере вход выполнен"
    query = task_query(state)
    assert "Nextcloud" in query and "Загрузить файл" in query
    assert "Шаг 4" in query and "неверный пароль" in query
    assert "Ошибка при загрузке файла" in query and "В браузере вход выполнен" not in query
    assert state["messages"] == messages


@pytest.mark.parametrize("scope,include_application", [("application", True), ("software", True), ("hardware", False)])
def test_query_uses_application_only_for_software_scope(scope, include_application):
    state = update_task_state("В программе Альфа нужно изменить громкость", None, "", UnderstandingModel({
        "facts": [fact("application", "Альфа"), fact("scope", scope, "изменить громкость")],
        "goals": [goal("изменить громкость")],
    }))
    assert ("Альфа" in task_query(state)) is include_application


def test_error_and_current_step_are_replaced_only_by_new_reported_values():
    state = update_task_state("На шаге 4 ошибка доступа", None, "", UnderstandingModel({
        "facts": [fact("current_step", "шаг 4", "шаге 4"), fact("error_text", "ошибка доступа")],
    }))
    state = update_task_state("Теперь шаг 5, файл не найден", state, "", UnderstandingModel({
        "facts": [fact("current_step", "шаг 5"), fact("error_text", "файл не найден")],
    }))
    assert [item["key"] for item in state["facts"]] == ["current_step", "error_text"]
    assert "ошибка доступа" not in task_query(state) and "шаг 4" not in task_query(state)
    assert "файл не найден" in task_query(state) and "шаг 5" in task_query(state)


def test_assistant_question_cannot_supply_an_unreported_error():
    with pytest.raises(ValueError, match="InvalidTaskEvidence"):
        update_task_state("Не получилось", None, "Ошибка «неверный пароль»?", UnderstandingModel({
            "facts": [fact("error_text", "неверный пароль")],
        }))


def test_latest_error_can_replace_old_completed_history_when_fact_budget_is_full():
    previous = {"facts": [fact("application", "Nextcloud"), fact("scope", "application"),
                           fact("current_step", "Шаг 4")]
                + [fact("completed", f"Проверка {number} выполнена") for number in range(9)],
                "goals": {"g1": {"text": "Загрузить файл", "evidence": "Загрузить файл"}},
                "messages": ["Загрузить файл"]}
    unchanged = copy.deepcopy(previous)
    state = update_task_state("неверный пароль", previous, "Какой текст ошибки?", UnderstandingModel({
        "facts": [fact("error_text", "неверный пароль")],
    }))
    assert len(state["facts"]) == 12
    assert any(item["key"] == "error_text" and item["value"] == "неверный пароль" for item in state["facts"])
    assert not any(item["value"] == "Проверка 0 выполнена" for item in state["facts"])
    assert state["goals"] == previous["goals"] and previous == unchanged


@pytest.mark.parametrize("message", [None, "", "  ", "я" * 6001], ids=["null", "empty", "blank", "oversized"])
def test_invalid_user_message_does_not_call_the_model(message):
    model = UnderstandingModel()
    with pytest.raises(ValueError, match="InvalidTaskMessage"):
        update_task_state(message, None, "", model)
    assert model.calls == []


@pytest.mark.parametrize("previous", [{"goals": []}, {"facts": "facts"}, {"goals": {"g1": "bad"}},
                                      {"messages": [None]}, ["bad"]])
def test_corrupt_previous_state_is_rejected(previous):
    with pytest.raises(ValueError, match="InvalidTaskState"):
        update_task_state("вопрос", previous, "", UnderstandingModel())


@pytest.mark.parametrize("update", [
    {"new_task": "false"}, {"question_field": "equipment_model"},
    {"facts": [fact("administrator", "true")]},
    {"goals": [goal("x", goal_id="unknown")]},
    {"instruction": "Перезапустите компьютер"},
])
def test_task_schema_rejects_unrecognized_fields_types_and_model_questions(update):
    with pytest.raises(ValidationError):
        SupportTaskUnderstanding.model_validate(update)


def test_task_prompt_keeps_user_intent_independent_of_retrieved_sources():
    assert "Здесь нет документов базы знаний" in SUPPORT_TASK_TASK
    assert "Уже выполненное подключение к 1С" in SUPPORT_TASK_TASK
    assert "Не проси модель, бренд" in SUPPORT_TASK_TASK
    assert "Не теряй вторую цель" in SUPPORT_TASK_TASK


def test_task_prompt_defers_configuration_questions_to_the_actual_sources():
    assert "Если желаемый результат уже понятен, question=\"\"" in SUPPORT_TASK_TASK
    assert "до чтения базы знаний не запрашивай" in SUPPORT_TASK_TASK
    assert "Настройка, которую предписывает инструкция" in SUPPORT_TASK_TASK
    assert "previous_state.unknown_answers" in SUPPORT_TASK_TASK
    assert "Не повторяй вопрос\nиз unknown_answers" in SUPPORT_TASK_TASK
    assert "IMAP" not in SUPPORT_TASK_TASK and "Outlook" not in SUPPORT_TASK_TASK


def test_task_prompt_distinguishes_static_facts_from_workflows_and_preserves_object_identity():
    assert "подачи обращения — ACTION" in SUPPORT_TASK_TASK
    assert "Готовое значение, перечень требований" in SUPPORT_TASK_TASK
    assert "Просьба настроек\nдля начала конкретной работы задаёт ACTION" in SUPPORT_TASK_TASK
    assert "пока сотрудник не назвал второй" in SUPPORT_TASK_TASK
    assert "первый и второй принтер" in SUPPORT_TASK_TASK
    assert "ширина и высота не заменяются одной\nшириной" in SUPPORT_TASK_TASK


def model_settings():
    return Settings(_env_file=None, llm_backend="openwebui", openwebui_api_key="test-key",
                    openwebui_model="local-qwen")


def test_real_adapter_task_stage_has_its_own_user_only_payload_and_bounded_budget():
    previous = compound_task()

    def handler(request):
        body = json.loads(request.content)
        assert body["max_tokens"] == 4096
        assert body["params"] == {"think": False, "max_tokens": 4096, "temperature": 0}
        assert body["tools"] == [] and body["tool_ids"] == []
        assert body["features"]["web_search"] is False
        schema = body["response_format"]["json_schema"]["schema"]
        assert schema["title"] == "SupportTaskUnderstanding"
        assert schema["$defs"]["SupportTaskGoal"]["properties"]["goal_id"]["anyOf"][0]["enum"] == ["g1", "g2"]
        payload = json.loads(body["messages"][-1]["content"])
        assert {key: payload[key] for key in ("message", "previous_state", "previous_question")} == {
            "message": "Сначала сброс", "previous_state": previous,
            "previous_question": "Что сделать сначала?",
        }
        assert payload["user_evidence_references"] is True
        assert "Сначала сброс" in payload["user_evidence"].values()
        assert payload["current_user_evidence_ids"] == [
            key for key, text in payload["user_evidence"].items() if text in "Сначала сброс"
        ]
        assert "catalogue" not in payload and "candidates" not in payload
        assert "JSON Schema: " not in body["messages"][0]["content"]
        response = {"new_task": False, "facts": [], "goals": [], "question": "", "question_field": None}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(response)},
                                                     "finish_reason": "stop"}]})

    result = OpenWebUILLM(model_settings(), httpx.MockTransport(handler)).understand_task(
        "Сначала сброс", previous, "Что сделать сначала?",
    )
    assert result == SupportTaskUnderstanding()


def test_explicit_task_boundary_removes_old_goal_identifiers_from_the_real_model_grammar():
    previous = compound_task()
    message = "Другой вопрос. Хочу открыть рабочую почту."
    calls = []

    def handler(request):
        body = json.loads(request.content)
        schema = body["response_format"]["json_schema"]["schema"]
        assert schema["$defs"]["SupportTaskGoal"]["properties"]["goal_id"]["type"] == "null"
        payload = json.loads(body["messages"][-1]["content"])
        calls.append(payload)
        assert payload["previous_state"]["facts"] == [] and payload["previous_state"]["goals"] == {}
        assert payload["previous_state"]["messages"] == [] and payload["previous_question"] == ""
        response = {"new_task": False, "facts": [], "question": "", "question_field": None,
                    "goals": [{"text": "Открыть рабочую почту", "evidence": "открыть рабочую почту",
                               "goal_id": None, "kind": "ACTION", "search_queries": []}]}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(response)},
                                                     "finish_reason": "stop"}]})

    model = OpenWebUILLM(model_settings(), httpx.MockTransport(handler))
    state = update_task_state(message, previous, "Какое действие выполнить сначала?", model)
    assert len(calls) == 1 and state["new_task"] is True
    assert list(state["goals"].values()) == [{"text": "Открыть рабочую почту", "evidence": "открыть рабочую почту",
                                           "kind": "ACTION", "search_queries": []}]


@pytest.mark.parametrize("budget", [False, 0, 127, 4097, 2.5, "512"])
def test_fast_output_budget_rejects_invalid_or_unbounded_values(budget):
    model = OpenWebUILLM(model_settings())
    with pytest.raises(ValueError, match="InvalidOutputTokenBudget"):
        model._call(SupportTaskUnderstanding, "task", {}, fast_selection=True, max_output_tokens=budget)


@pytest.mark.parametrize("goal_count,budget", [(1, 768), (2, 768), (3, 1024), (4, 1024), (5, 1536), (8, 1536)])
def test_compound_selection_scales_its_json_budget_and_aliases_image_inventory(goal_count, budget):
    file_id = "00000000-0000-4000-8000-000000000037"
    caption = "Исходный текст подтверждённой инструкции настройки оборудования."
    payload = {"catalogue": {file_id: "Настройки.docx"}, "candidates": {file_id: caption},
               "requested_goals": {f"g{i}": f"Цель {i}" for i in range(1, goal_count + 1)},
               "candidate_image_numbers": {file_id: [1, 2]},
               "candidate_image_captions": {file_id: {1: caption}}, "allow_clarification": False}
    unchanged = copy.deepcopy(payload)

    def handler(request):
        body = json.loads(request.content)
        assert body["max_tokens"] == budget and body["params"]["max_tokens"] == budget
        schema = body["response_format"]["json_schema"]["schema"]
        variants = {variant["properties"]["result"]["const"]: variant for variant in schema["anyOf"]}
        assert "CLARIFY" not in variants and ("PARTIAL" in variants) is (goal_count > 1)
        matched = variants["MATCH"]
        assert matched["additionalProperties"] is False
        assert matched["properties"]["file_id"] == {"type": "string", "const": "s1"}
        assert matched["properties"]["covered_goals"]["minItems"] == goal_count
        assert matched["properties"]["covered_goals"]["maxItems"] == goal_count
        assert "missing_goal_ids" not in matched["properties"]
        proof = schema["$defs"]["GoalReference_s1"]["properties"]
        assert proof["goal_id"]["enum"] == list(payload["requested_goals"])
        assert proof["evidence_id"]["enum"] == ["i1_1"]
        assert variants["NO_KNOWLEDGE"]["properties"]["missing_goal_ids"]["const"] == list(payload["requested_goals"])
        wire = json.loads(body["messages"][-1]["content"])
        assert "candidate_image_numbers" not in wire and "candidate_image_captions" not in wire
        assert wire["candidates"] == {"s1": {"i1_1": {
            "text": caption, "support_kind": "IMAGE", "image_numbers": [1],
        }}}
        assert wire["requested_goals"] == payload["requested_goals"]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
            "result": "NO_KNOWLEDGE", "missing_goal_ids": list(payload["requested_goals"]),
        })}}]})

    result = OpenWebUILLM(model_settings(), httpx.MockTransport(handler)).select_instruction(payload)
    assert result.missing_goal_ids == list(payload["requested_goals"])
    assert payload == unchanged


def test_image_inventory_rejects_sources_outside_the_current_candidates():
    model = OpenWebUILLM(model_settings())
    with pytest.raises(ValueError, match="UnknownInstructionImageCandidate"):
        model.select_instruction({"catalogue": {"outside": "doc"}, "candidates": {},
                                  "candidate_image_numbers": {"outside": [1]}})
