"""The task adapter requires explicit transitions without changing stored defaults."""

import copy
import json

import httpx
import pytest

from app.core.config import Settings
from app.schemas.contracts import SupportTaskUnderstanding
from app.services.llm.openwebui import OpenWebUILLM
from app.services.llm.prompts import SUPPORT_TASK_TASK


REQUIRED_FIELDS = {"new_task", "facts", "goals", "question", "question_field"}


def complete_reply(**values):
    reply = {"new_task": False, "facts": [], "goals": [], "question": "", "question_field": None, **values}
    reply["goals"] = [{"goal_id": None, **goal} for goal in reply["goals"]]
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
    assert request["max_tokens"] == 1536
    schema = request["response_format"]["json_schema"]["schema"]
    assert REQUIRED_FIELDS <= set(schema["required"])
    goal_definition = schema["$defs"]["SupportTaskGoal"]
    assert {"text", "evidence", "goal_id"} <= set(goal_definition["required"])
    goal_ids = goal_definition["properties"]["goal_id"]
    if previous_goals:
        assert goal_ids == {"anyOf": [{"type": "string", "enum": list(previous_goals)},
                                     {"type": "null"}], "default": None}
    else:
        assert goal_ids == {"type": "null", "default": None}
    assert json.loads(request["messages"][-1]["content"]) == {
        "message": "Новый вопрос", "previous_state": previous, "previous_question": "",
    }


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
