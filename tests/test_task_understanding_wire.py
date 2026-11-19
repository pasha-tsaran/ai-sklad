"""The task adapter requires explicit transitions without changing stored defaults."""

import copy
import json

import httpx
import pytest

from app.core.config import Settings
from app.schemas.contracts import SupportTaskUnderstanding
from app.services.chat.task_evidence import (
    MAX_FRAGMENT_CHARS, MAX_REFERENCE_CHARS, MAX_REFERENCES, build_user_evidence,
)
from app.services.llm.openwebui import OpenWebUILLM
from app.services.llm.prompts import SUPPORT_TASK_TASK


REQUIRED_FIELDS = {"new_task", "facts", "goals", "question", "question_field"}


def complete_reply(**values):
    reply = {"new_task": False, "facts": [], "goals": [], "question": "", "question_field": None, **values}
    reply["goals"] = [{"goal_id": None, "kind": "ACTION", "search_queries": [], **goal}
                      for goal in reply["goals"]]
    return reply


def model_with_replies(replies, requests):
    def handler(request):
        requests.append(json.loads(request.content))
        response = replies.pop(0)
        return httpx.Response(200, json={"choices": [{
            "message": {"content": json.dumps(response)}, "finish_reason": "stop",
        }]})

    settings = Settings(_env_file=None, llm_backend="openwebui", openwebui_api_key="test-api-key",
                        openwebui_model="local-test-model")
    return OpenWebUILLM(settings, httpx.MockTransport(handler))


@pytest.mark.parametrize("previous_goals", [{}, {"g1": {"text": "Первая цель"}, "g3": {"text": "Другая цель"}}])
def test_task_response_schema_requires_every_transition_field_and_current_goal_ids(previous_goals):
    requests = []
    model = model_with_replies([complete_reply()], requests)
    previous = {"facts": [], "goals": previous_goals, "messages": []}
    unchanged = copy.deepcopy(previous)
    reply = model.understand_task("Новый вопрос", previous, "")
    assert REQUIRED_FIELDS <= reply.model_fields_set
    assert previous == unchanged and len(requests) == 1
    request = requests[0]
    assert request["max_tokens"] == 4096
    schema = request["response_format"]["json_schema"]["schema"]
    assert REQUIRED_FIELDS <= set(schema["required"])
    goal_definition = schema["$defs"]["SupportTaskGoal"]
    assert {"text", "evidence", "goal_id", "kind", "search_queries"} <= set(goal_definition["required"])
    goal_ids = goal_definition["properties"]["goal_id"]
    if previous_goals:
        assert goal_ids == {"anyOf": [{"type": "string", "enum": list(previous_goals)},
                                     {"type": "null"}], "default": None}
    else:
        assert goal_ids == {"type": "null", "default": None}
    assert json.loads(request["messages"][-1]["content"]) == {
        "message": "Новый вопрос", "previous_state": previous, "previous_question": "",
        "user_evidence_references": True, "user_evidence": {"u1": "Новый вопрос"},
        "current_user_evidence_ids": ["u1"],
    }


def referenced_fact(**values):
    return {"key": "condition", "value": "одно устройство", "evidence": "u1",
            "entity": "", "attribute": "", "negated": False, **values}


def test_user_reference_restores_original_inflection_without_copying_a_quote():
    message = "Хочу подключить терминал к одному принтеру по вай фай."
    response = complete_reply(new_task=True,
                              facts=[referenced_fact(value="один принтер", entity="принтер", attribute="количество")],
                              goals=[{"text": "Подключить один принтер", "evidence": "u1"}])
    requests = []
    reply = model_with_replies([response], requests).understand_task(message, {}, "")
    assert reply.facts[0].value == "один принтер" and reply.facts[0].evidence == message
    assert reply.goals[0].evidence == message and len(requests) == 1
    schema = requests[0]["response_format"]["json_schema"]["schema"]
    for definition in ("SupportTaskFact", "SupportTaskGoal"):
        assert schema["$defs"][definition]["properties"]["evidence"] == {"type": "string", "enum": ["u1"]}
    assert {"negated", "entity", "attribute"} <= set(schema["$defs"]["SupportTaskFact"]["required"])


def test_lookup_goal_keeps_bounded_search_aliases_outside_the_user_evidence_registry():
    message = "Какой добавочный номер у руководства склада?"
    aliases = ["внутренний телефон руководство склада", "warehouse management extension number"]
    response = complete_reply(goals=[{
        "text": "Узнать внутренний телефон руководства склада", "evidence": "u1",
        "kind": "LOOKUP", "search_queries": aliases,
    }])
    requests = []
    reply = model_with_replies([response], requests).understand_task(message, {}, "")
    assert reply.goals[0].kind == "LOOKUP" and reply.goals[0].search_queries == aliases
    payload = json.loads(requests[0]["messages"][-1]["content"])
    assert payload["user_evidence"] == {"u1": message}
    schema = requests[0]["response_format"]["json_schema"]["schema"]["$defs"]["SupportTaskGoal"]
    assert schema["properties"]["search_queries"]["maxItems"] == 3
    assert schema["properties"]["search_queries"]["items"]["maxLength"] == 160


def test_mixed_value_lookup_and_change_request_keeps_two_independent_ordered_kinds():
    message = "Какая стандартная ширина этикетки, и как мне изменить её?"
    reply = complete_reply(goals=[
        {"text": "Узнать стандартную ширину этикетки", "evidence": "u1", "kind": "LOOKUP",
         "search_queries": ["default label width"]},
        {"text": "Изменить ширину этикетки", "evidence": "u1", "kind": "ACTION",
         "search_queries": ["change label width"]},
    ])
    requests = []
    result = model_with_replies([reply], requests).understand_task(message, {}, "")
    assert [goal.kind for goal in result.goals] == ["LOOKUP", "ACTION"]
    assert [goal.evidence for goal in result.goals] == [message, message]
    prompt = requests[0]["messages"][0]["content"]
    assert prompt.index("У каждой цели ровно один kind") < prompt.index("Сначала отдели общую тему настройки")


def test_eight_task_goals_have_their_own_output_budget_without_changing_other_stages():
    message = "Нужны восемь отдельных результатов для рабочего устройства и программы"
    reply = complete_reply(goals=[{
        "text": f"Отдельный результат {index}", "evidence": "u1", "kind": "ACTION",
        "search_queries": [f"рабочая задача {index}", f"work task {index}"],
    } for index in range(1, 9)])
    requests = []
    model = model_with_replies([reply], requests)
    result = model.understand_task(message, {}, "")
    assert len(result.goals) == 8 and requests[0]["max_tokens"] == 4096
    assert requests[0]["params"]["max_tokens"] == 4096
    from app.schemas.contracts import DialogueResult

    with pytest.raises(ValueError, match="InvalidOutputTokenBudget"):
        model._call(DialogueResult, "stage", {}, fast_selection=True, max_output_tokens=1537)


@pytest.mark.parametrize("message", [
    "Как позвонить руководству нашего склада по внутреннему телефону?",
    "Как связаться с отделом кадров?",
    "Где найти контакт ответственного за документы?",
])
def test_contact_request_can_use_a_lookup_goal_without_inventing_phone_configuration(message):
    response = complete_reply(goals=[{
        "text": "Узнать контакт получателя", "evidence": "u1", "kind": "LOOKUP",
        "search_queries": ["контакт получатель справочник", "contact directory"],
    }])
    requests = []
    reply = model_with_replies([response], requests).understand_task(message, {}, "")
    assert reply.goals[0].kind == "LOOKUP" and reply.goals[0].evidence == message
    task = requests[0]["messages"][0]["content"]
    assert task.index("Контактный LOOKUP отвечает на просьбу") < task.index("kind=LOOKUP означает")
    assert "Слова «как» и «сделать» отдельно не определяют kind" in task


@pytest.mark.parametrize("message", [
    "Выдали настольный принтер этикеток. Покажите, куда у него подключается питание.",
    "Где находится нужный разъём на схеме устройства?",
    "Покажите, где в окне расположена кнопка просмотра.",
])
def test_request_to_locate_an_existing_element_keeps_lookup_intent(message):
    response = complete_reply(goals=[{
        "text": "Найти расположение нужного элемента", "evidence": "u1", "kind": "LOOKUP",
        "search_queries": ["расположение элемента схема", "element location diagram"],
    }])
    requests = []
    result = model_with_replies([response], requests).understand_task(message, {}, "")
    assert result.goals[0].kind == "LOOKUP" and result.goals[0].evidence == message
    assert "Указание места существующего элемента также LOOKUP" in requests[0]["messages"][0]["content"]


@pytest.mark.parametrize("field", ["kind", "search_queries"])
def test_missing_goal_discovery_metadata_gets_one_explicit_contract_repair(field):
    message = "Нужно найти внутренний номер руководства"
    complete = complete_reply(goals=[{"text": "Узнать телефон руководства", "evidence": "u1", "kind": "LOOKUP"}])
    incomplete = copy.deepcopy(complete)
    incomplete["goals"][0].pop(field)
    requests = []
    reply = model_with_replies([incomplete, complete], requests).understand_task(message, {}, "")
    assert reply.goals[0].kind == "LOOKUP" and len(requests) == 2
    repair = json.loads(requests[1]["messages"][-1]["content"])
    assert repair["missing_fields"] == ["goals.0." + field]


@pytest.mark.parametrize("field", ["kind", "search_queries"])
def test_repeated_missing_goal_discovery_metadata_cannot_silently_default(field):
    incomplete = complete_reply(goals=[{"text": "Узнать телефон", "evidence": "u1"}])
    incomplete["goals"][0].pop(field)
    with pytest.raises(ValueError, match="^IncompleteTaskUnderstanding$"):
        model_with_replies([incomplete, incomplete], []).understand_task("Нужен телефон", {}, "")


def test_search_alias_cannot_supply_evidence_for_an_invented_user_fact():
    message = "Нужен телефон руководства"
    invalid = complete_reply(
        facts=[referenced_fact(value="Warehouse", evidence="warehouse management extension number")],
        goals=[{"text": "Найти телефон", "evidence": "u1", "kind": "LOOKUP",
                "search_queries": ["warehouse management extension number"]}],
    )
    with pytest.raises(ValueError, match="^InvalidTaskEvidence$"):
        model_with_replies([invalid, invalid], []).understand_task(message, {}, "")


def test_verified_guide_context_is_read_only_and_never_enters_user_evidence():
    message = "На этом шаге пишет, что пароль неверный"
    guide = {"sources": [{"sections": [{"id": "step-1", "text": "Откройте рабочую программу."}]}],
             "current_progress": {"cursor": 1}}
    response = complete_reply(facts=[referenced_fact(key="error_text", value="пароль неверный")])
    requests = []
    reply = model_with_replies([response], requests).understand_task(message, {}, "", guide_context=guide)
    payload = json.loads(requests[0]["messages"][-1]["content"])
    assert payload["guide_context"] == guide and payload["user_evidence"] == {"u1": message}
    assert reply.facts[0].evidence == message
    invalid = complete_reply(facts=[referenced_fact(key="completed", value="Рабочая программа открыта",
                                                   evidence="Откройте рабочую программу.")])
    with pytest.raises(ValueError, match="^InvalidTaskEvidence$"):
        model_with_replies([invalid, invalid], []).understand_task(message, {}, "", guide_context=guide)


def test_observational_step_error_keeps_existing_goal_text_identity_kind_and_aliases():
    from app.services.chat.task_state import update_task_state

    original = "Нужно открыть рабочую почту"
    previous = {"facts": [], "goals": {"g1": {
        "text": "Open work mail", "evidence": original, "kind": "ACTION", "search_queries": ["first mail login"],
    }}, "messages": [original], "question": "", "question_field": None}
    message = "На 4 шаге неверный пароль, но в браузере вошёл нормально"
    guide = {"sources": [{"sections": [{"id": "step-4", "text": "Введите свои рабочие учётные данные."}]}]}
    reply = complete_reply(facts=[
        referenced_fact(key="current_step", value="4"),
        referenced_fact(key="error_text", value="неверный пароль"),
        referenced_fact(key="completed", value="В браузере вход выполнен"),
    ])
    requests = []
    state = update_task_state(message, previous, "", model_with_replies([reply], requests), guide_context=guide)
    assert state["goals"] == previous["goals"]
    assert {fact["key"] for fact in state["facts"]} == {"current_step", "error_text", "completed"}
    payload = json.loads(requests[0]["messages"][-1]["content"])
    assert payload["guide_context"] == guide
    assert "и не меняет требуемый результат" in requests[0]["messages"][0]["content"]


def test_user_reference_preserves_spaces_newlines_and_message_edges():
    message = "  Рабочая  почта\n\nне открывается.  "
    response = complete_reply(goals=[{"text": "Открыть рабочую почту", "evidence": "u1"}])
    requests = []
    reply = model_with_replies([response], requests).understand_task(message, {}, "")
    assert reply.goals[0].evidence == message
    payload = json.loads(requests[0]["messages"][-1]["content"])
    assert payload["user_evidence"] == {"u1": message}


@pytest.mark.parametrize(("message", "goal"), [
    ("На экране есть изображение, а на распечатке оно пропадает.",
     "Печатать документ вместе с изображением"),
    ("В рабочую почту больше не приходят письма, хотя интернет работает.",
     "Восстановить получение рабочих писем"),
    ("При считывании товара код не появляется в открытой программе.",
     "Передавать считанный код товара в программу"),
])
def test_concrete_symptom_can_ground_a_minimal_goal_without_an_explicit_request(message, goal):
    response = complete_reply(new_task=True,
                              facts=[referenced_fact(key="symptoms", value=message)],
                              goals=[{"text": goal, "evidence": "u1"}])
    requests = []
    reply = model_with_replies([response], requests).understand_task(message, {}, "")
    assert reply.goals[0].text == goal and reply.goals[0].evidence == message
    assert reply.facts[0].key == "symptoms" and reply.facts[0].evidence == message
    assert reply.question == "" and reply.question_field is None and len(requests) == 1
    task = requests[0]["messages"][0]["content"]
    assert "Конкретное нежелательное проявление уже задаёт цель" in task
    assert "не спрашивай повторно" in task and "способ\nремонта" in task


@pytest.mark.parametrize("message", [
    "Настройка сканера",
    "Другая задача. Настройка сканера",
    "Помогите с настройкой рабочей программы",
])
def test_general_setup_topic_keeps_empty_goals_before_a_concrete_outcome_is_known(message):
    response = complete_reply(new_task=message.startswith("Другая задача"),
                              question="Что хотите изменить или что получается неправильно?",
                              question_field="goal")
    requests = []
    reply = model_with_replies([response], requests).understand_task(message, {}, "")
    assert not reply.goals and reply.question and reply.question_field == "goal"
    task = requests[0]["messages"][0]["content"]
    assert task.index("Сначала отдели общую тему настройки") < task.index("Конкретное нежелательное проявление")
    assert "не\nявляются конкретными целями" in task


@pytest.mark.parametrize("message,new_task,goal", [
    ("Уменьшить звуковой сигнал сканера штрихкодов", False, "Уменьшить звуковой сигнал сканера"),
    ("Хочу впервые открыть почту в установленной программе", True, "Впервые открыть рабочую почту"),
])
def test_pending_goal_question_keeps_its_frame_without_forcing_a_different_topic_to_continue(
        message, new_task, goal):
    question = "Какой результат настройки сканера нужен?"
    previous = {"facts": [], "goals": {}, "messages": ["Настройка сканера"],
                "question": question, "question_field": "goal"}
    response = complete_reply(new_task=new_task, goals=[{"text": goal, "evidence": "u1"}])
    requests = []
    reply = model_with_replies([response], requests).understand_task(message, previous, question)
    assert reply.new_task is new_task and reply.goals[0].evidence == message
    payload = json.loads(requests[0]["messages"][-1]["content"])
    assert payload["previous_state"] == previous and payload["previous_question"] == question
    policy = requests[0]["messages"][0]["content"]
    assert "при previous_state.goals={}" in policy
    assert "Новый самостоятельный предмет или другая рабочая задача получает new_task=true" in policy


def test_long_message_references_are_exact_bounded_spans_including_the_last_line():
    message = "  начало\n" + "условие " * 620 + "\nпоследняя строка  "
    references = build_user_evidence([message])
    reference = next(key for key, text in references.items() if "последняя строка" in text)
    response = complete_reply(goals=[{"text": "Проверить последнее условие", "evidence": reference}])
    requests = []
    reply = model_with_replies([response], requests).understand_task(message, {}, "")
    assert reply.goals[0].evidence == references[reference]
    assert all(0 < len(text) <= MAX_FRAGMENT_CHARS and text in message for text in references.values())
    assert references["u1"].startswith("  начало\n")
    assert any(text.endswith("последняя строка  ") for text in references.values())


def test_user_reference_registry_has_bounded_size_and_current_message_priority():
    current = "текущее сообщение\n" + "первое условие " * 350 + "последнее текущее условие"
    previous = [f"Предыдущая реплика {index}. " + "длинное условие " * 100 for index in range(30)]
    references = build_user_evidence([current, *previous])
    assert len(references) <= MAX_REFERENCES
    assert sum(map(len, references.values())) <= MAX_REFERENCE_CHARS
    assert references["u1"].startswith("текущее сообщение\n")
    assert any("последнее текущее условие" in text for text in references.values())


def test_followup_can_resolve_a_reference_to_previously_verified_user_text():
    original = "Хочу открыть рабочую папку"
    message = "Теперь появляется ошибка"
    previous = {"messages": [original], "facts": [], "goals": {"g1": {"text": "Открыть папку"}}}
    response = complete_reply(goals=[{"goal_id": "g1", "text": "Открыть папку", "evidence": "u2"}])
    requests = []
    reply = model_with_replies([response], requests).understand_task(message, previous, "")
    assert reply.goals[0].evidence == original
    payload = json.loads(requests[0]["messages"][-1]["content"])
    assert payload["current_user_evidence_ids"] == ["u1"]


@pytest.mark.parametrize("explicit_boundary", [False, True])
def test_new_task_cannot_resolve_a_reference_from_the_previous_task(explicit_boundary):
    message = "Хочу открыть почту"
    if explicit_boundary:
        message = "Другая задача. " + message
    previous = {"messages": ["Старая задача с рабочей папкой"], "facts": [], "goals": {}}
    invalid = complete_reply(new_task=not explicit_boundary,
                             goals=[{"text": "Открыть рабочую папку", "evidence": "u2"}])
    requests = []
    with pytest.raises(ValueError, match="^InvalidTaskEvidence$"):
        model_with_replies([invalid, invalid], requests).understand_task(message, previous, "")
    assert len(requests) == 2


@pytest.mark.parametrize("field", ["replacement_evidence", "unknown_answer_evidence"])
def test_control_references_are_constrained_and_resolved_to_the_current_message(field):
    message = "Я не знаю, что это значит"
    previous = {"messages": ["Хочу открыть рабочую папку"], "facts": [], "goals": {}}
    requests = []
    valid = complete_reply(**{field: "u1"})
    reply = model_with_replies([valid], requests).understand_task(message, previous, "Какое подключение?")
    assert getattr(reply, field) == message
    schema = requests[0]["response_format"]["json_schema"]["schema"]
    assert schema["properties"][field]["enum"] == ["", "u1"]
    invalid = complete_reply(**{field: "u2"})
    with pytest.raises(ValueError, match="^InvalidTaskEvidence$"):
        model_with_replies([invalid, invalid], []).understand_task(message, previous, "Какое подключение?")


def test_unknown_user_reference_exhausts_only_the_shared_repair_budget():
    invalid = complete_reply(goals=[{"text": "Изменить громкость", "evidence": "u9999"}])
    requests = []
    with pytest.raises(ValueError, match="^InvalidTaskEvidence$"):
        model_with_replies([invalid, invalid], requests).understand_task("Хочу сделать звук тише", {}, "")
    assert len(requests) == 2
    repair = json.loads(requests[1]["messages"][-1]["content"])
    assert repair["invalid_evidence_fields"] == ["goals.0.evidence"]


def test_referenced_negative_fact_preserves_device_identity_and_explicit_negation():
    message = "Оба принтера по Wi-Fi, блютуз не используется"
    response = complete_reply(facts=[referenced_fact(key="connection", value="Bluetooth", entity="оба принтера",
                                                    attribute="соединение", negated=True)])
    reply = model_with_replies([response], []).understand_task(message, {}, "")
    assert reply.facts[0].negated is True and reply.facts[0].entity == "оба принтера"
    assert reply.facts[0].attribute == "соединение" and reply.facts[0].evidence == message


@pytest.mark.parametrize("field", ["negated", "entity", "attribute"])
def test_referenced_fact_cannot_silently_default_its_condition_metadata(field):
    incomplete = complete_reply(facts=[referenced_fact()])
    incomplete["facts"][0].pop(field)
    requests = []
    corrected = complete_reply(facts=[referenced_fact()])
    reply = model_with_replies([incomplete, corrected], requests).understand_task("одно устройство", {}, "")
    assert reply.facts[0].evidence == "одно устройство" and len(requests) == 2
    repair = json.loads(requests[1]["messages"][-1]["content"])
    assert repair["missing_fields"] == ["facts.0." + field]


def test_repeated_missing_referenced_fact_metadata_fails_without_dropping_the_fact():
    incomplete = complete_reply(facts=[referenced_fact(negated=True)])
    incomplete["facts"][0].pop("negated")
    requests = []
    with pytest.raises(ValueError, match="^IncompleteTaskUnderstanding$"):
        model_with_replies([incomplete, incomplete], requests).understand_task("блютуз не используется", {}, "")
    assert len(requests) == 2


@pytest.mark.parametrize("missing", sorted(REQUIRED_FIELDS))
def test_one_incomplete_response_gets_one_explicit_contract_repair(missing):
    incomplete = complete_reply()
    incomplete.pop(missing)
    complete = complete_reply(new_task=True)
    requests = []
    model = model_with_replies([incomplete, complete], requests)
    reply = model.understand_task("Хочу выполнить другую работу", {"goals": {}}, "")
    assert reply.new_task is True and len(requests) == 2
    first, second = [json.loads(request["messages"][-1]["content"]) for request in requests]
    assert second == {**first, "contract_retry": True, "missing_fields": [missing]}
    assert REQUIRED_FIELDS <= set(requests[1]["response_format"]["json_schema"]["schema"]["required"])


def test_empty_provider_responses_cannot_silently_reuse_previous_task():
    requests = []
    model = model_with_replies([{}, {}], requests)
    with pytest.raises(ValueError, match="^IncompleteTaskUnderstanding$"):
        model.understand_task("Другой вопрос", {"goals": {"g1": {"text": "Прежняя цель"}}}, "")
    assert len(requests) == 2


def test_missing_goal_id_gets_one_repair_that_preserves_the_corrected_goal():
    message = "Хочу печатать на двух устройствах"
    goal = {"text": "Печатать на двух устройствах", "evidence": "печатать на двух устройствах"}
    incomplete = complete_reply(goals=[goal])
    incomplete["goals"][0].pop("goal_id")
    corrected = complete_reply(goals=[{**goal, "goal_id": "g1"}])
    previous = {"facts": [], "goals": {"g1": {"text": "Печатать на устройстве"}}, "messages": []}
    requests = []
    reply = model_with_replies([incomplete, corrected], requests).understand_task(message, previous, "")
    assert reply.goals[0].goal_id == "g1" and len(requests) == 2
    repair = json.loads(requests[1]["messages"][-1]["content"])
    assert repair["missing_fields"] == ["goals.0.goal_id"] and repair["contract_retry"] is True


def test_repeated_missing_goal_id_exhausts_the_shared_contract_retry():
    message = "Хочу открыть папку"
    incomplete = complete_reply(goals=[{"text": "Открыть папку", "evidence": "открыть папку"}])
    incomplete["goals"][0].pop("goal_id")
    requests = []
    with pytest.raises(ValueError, match="^IncompleteTaskUnderstanding$"):
        model_with_replies([incomplete, incomplete], requests).understand_task(message, {}, "")
    assert len(requests) == 2


def test_task_schema_defaults_remain_usable_by_existing_callers():
    legacy = SupportTaskUnderstanding.model_validate({})
    assert legacy.new_task is False and legacy.facts == [] and legacy.goals == []
    assert legacy.question == "" and legacy.question_field is None


def test_task_summary_reports_only_bounded_metadata(monkeypatch):
    diagnostics, requests = [], []
    monkeypatch.setattr("app.services.llm.openwebui.event", lambda name, **fields: diagnostics.append((name, fields)))
    reply = complete_reply(
        facts=[{"key": "application", "value": "Private application", "evidence": "private employee phrase"}],
        goals=[{"text": "Private requested outcome", "evidence": "private employee phrase", "goal_id": "g1"}],
        question="Private employee question?", question_field="goal",
    )
    model = model_with_replies([reply], requests)
    model.understand_task("private employee phrase", {"goals": {"g1": {"text": "Old goal"}}}, "")
    summary = next(fields for name, fields in diagnostics if name == "llm.task_understanding")
    assert summary == {
        "new_task": False, "replace_goals": False, "goal_count": 1, "goal_ids": ["g1"],
        "fact_keys": ["application"], "question_field": "goal", "has_question": True,
        "missing_fields": [], "invalid_evidence_fields": [], "contract_retry": False,
    }
    recorded = json.dumps(diagnostics)
    assert "Private application" not in recorded and "Private requested outcome" not in recorded
    assert "private employee phrase" not in recorded and "Private employee question" not in recorded
    assert "test-api-key" not in recorded


def test_task_diagnostics_failure_preserves_valid_reply(monkeypatch):
    def failed_logger(*args, **kwargs):
        raise RuntimeError("logger unavailable")

    monkeypatch.setattr("app.services.llm.openwebui.event", failed_logger)
    requests = []
    reply = model_with_replies([complete_reply(new_task=True)], requests).understand_task("Другая работа", {}, "")
    assert reply.new_task is True and len(requests) == 1


def test_task_transport_failure_is_not_a_contract_retry():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("service unavailable")

    settings = Settings(_env_file=None, llm_backend="openwebui", openwebui_api_key="test-key")
    model = OpenWebUILLM(settings, httpx.MockTransport(handler))
    with pytest.raises(httpx.ReadTimeout):
        model.understand_task("Вопрос", {}, "")
    assert len(calls) == 1


def test_invalid_task_field_is_not_a_missing_field_repair():
    requests = []
    model = model_with_replies([complete_reply(new_task="unknown")], requests)
    with pytest.raises(ValueError):
        model.understand_task("Вопрос", {}, "")
    assert len(requests) == 1


@pytest.mark.parametrize("kind", ["facts", "goals"])
def test_paraphrased_evidence_gets_one_repair_with_an_exact_original_quote(kind):
    message = "хочу изменить размер текста"
    item = ({"key": "condition", "value": "размер текста", "evidence": "Хочу изменить размер текста"}
            if kind == "facts" else {"text": "Изменить размер текста", "evidence": "Хочу изменить размер текста"})
    invalid = complete_reply(**{kind: [item]})
    valid = complete_reply(**{kind: [{**item, "evidence": message}]})
    requests = []
    reply = model_with_replies([invalid, valid], requests).understand_task(message, {}, "")
    assert getattr(reply, kind)[0].evidence == message and len(requests) == 2
    repair = json.loads(requests[1]["messages"][-1]["content"])
    assert repair["contract_retry"] is True
    assert repair["invalid_evidence_fields"] == [kind + ".0.evidence"]


def test_invalid_evidence_can_be_replaced_with_a_grounded_clarification():
    message = "Нужно настроить рабочее устройство"
    invalid = complete_reply(new_task=True, goals=[{
        "text": "Настроить подключение", "evidence": "требуется подключить оборудование",
    }])
    clarification = complete_reply(new_task=True, question="Какой результат настройки нужен?", question_field="goal")
    requests = []
    reply = model_with_replies([invalid, clarification], requests).understand_task(message, {}, "")
    assert reply.new_task and not reply.facts and not reply.goals
    assert reply.question_field == "goal" and len(requests) == 2


@pytest.mark.parametrize("new_task", [False, True])
def test_old_task_evidence_is_unavailable_to_a_new_standalone_task(new_task):
    previous = {"messages": ["Первая задача с отчётом"], "facts": [], "goals": {}}
    invalid = complete_reply(new_task=new_task, goals=[{
        "text": "Работа с отчётом", "evidence": "Первая задача с отчётом",
    }])
    requests = []
    with pytest.raises(ValueError, match="^InvalidTaskEvidence$"):
        model_with_replies([invalid, invalid], requests).understand_task("Другая задача. Открыть рабочую папку", previous, "")
    assert len(requests) == 2


def test_followup_can_quote_previously_verified_user_evidence():
    evidence = "Хочу открыть рабочую папку"
    previous = {"messages": [evidence], "facts": [], "goals": {}}
    response = complete_reply(goals=[{"text": "Открыть рабочую папку", "evidence": evidence}])
    requests = []
    reply = model_with_replies([response], requests).understand_task("В ней ошибка", previous, "")
    assert reply.goals[0].evidence == evidence and len(requests) == 1


def test_incomplete_and_invalid_evidence_repairs_share_one_retry_budget():
    invalid = complete_reply(goals=[{"text": "Открыть папку", "evidence": "несуществующая цитата"}])
    requests = []
    with pytest.raises(ValueError, match="^InvalidTaskEvidence$"):
        model_with_replies([{}, invalid], requests).understand_task("Хочу открыть папку", {}, "")
    assert len(requests) == 2


def test_rejected_evidence_is_not_written_to_diagnostics(monkeypatch):
    diagnostics, requests = [], []
    monkeypatch.setattr("app.services.llm.openwebui.event", lambda name, **fields: diagnostics.append((name, fields)))
    invalid = complete_reply(goals=[{"text": "private desired goal", "evidence": "private invalid quote"}])
    with pytest.raises(ValueError, match="^InvalidTaskEvidence$"):
        model_with_replies([invalid, invalid], requests).understand_task("private current employee text", {}, "")
    summaries = [fields for name, fields in diagnostics if name == "llm.task_understanding"]
    assert len(summaries) == 2 and summaries[0]["invalid_evidence_fields"] == ["goals.0.evidence"]
    text = json.dumps(diagnostics)
    assert "private desired goal" not in text and "private invalid quote" not in text
    assert "private current employee text" not in text


@pytest.mark.parametrize("field,message,options", [
    ("replacement_evidence", "Отменить первую цель", {"replace_goals": True}),
    ("unknown_answer_evidence", "Я не знаю этот вариант", {}),
])
def test_control_evidence_repairs_to_an_exact_current_message_quote(field, message, options):
    invalid = complete_reply(**options, **{field: "Пересказ фразы сотрудника"})
    corrected = complete_reply(**options, **{field: message})
    requests = []
    reply = model_with_replies([invalid, corrected], requests).understand_task(message, {}, "Какой вариант?")
    assert getattr(reply, field) == message and len(requests) == 2
    repair = json.loads(requests[1]["messages"][-1]["content"])
    assert repair["invalid_evidence_fields"] == [field]


def test_task_prompt_distinguishes_natural_new_requests_from_results_and_corrections():
    assert "Распознавай новую задачу и в сообщениях без слов «другая задача»" in SUPPORT_TASK_TASK
    assert "Как открыть общую сетевую папку?" in SUPPORT_TASK_TASK
    assert "Пишет отказано в доступе" in SUPPORT_TASK_TASK
    assert "исправляет условия той же задачи и сохраняет false" in SUPPORT_TASK_TASK
    assert "Если contract_retry=true" in SUPPORT_TASK_TASK
    assert "Передавай goal_id явно для каждой цели" in SUPPORT_TASK_TASK
    assert "способа соединения уточняет существующую цель и сохраняет её goal_id" in SUPPORT_TASK_TASK
