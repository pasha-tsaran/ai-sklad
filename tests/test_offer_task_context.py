"""An unaccepted document offer does not replace the employee's task frame."""

import copy
import json
from types import SimpleNamespace

import httpx
import pytest

from app.core.config import Settings
from app.schemas.contracts import DialogueResult
from app.services.chat import talk_dialogue
from app.services.chat.task_state import is_unknown_answer, update_task_state
from app.services.llm.openwebui import OpenWebUILLM
from tests.test_task_state import UnderstandingModel, compound_task


def native_model(response):
    calls = []

    def handler(request):
        body = json.loads(request.content)
        payload = json.loads(body["messages"][-1]["content"])
        calls.append(payload)
        reply = response(payload) if callable(response) else copy.deepcopy(response)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(reply)},
                                                     "finish_reason": "stop"}]})

    settings = Settings(_env_file=None, talk_dialogue_mode="legacy", llm_backend="openwebui", openwebui_api_key="test-key",
                        openwebui_model="local-test", nextcloud_backend="mock", mail_backend="mock")
    return OpenWebUILLM(settings, httpx.MockTransport(handler)), calls


def native_reply(*, new_task=False, evidence="u1", marker="u1", goal_id=None):
    return {"new_task": new_task, "facts": [], "goals": [{
        "text": "Открыть рабочую почту", "evidence": evidence, "goal_id": goal_id,
        "kind": "ACTION", "search_queries": ["открыть рабочую почту"],
    }], "unknown_answer_evidence": marker, "question": "", "question_field": None}


def offer_flow():
    return {"task_state": compound_task(), "task_query": "прежняя рабочая задача",
            "opening_query": "исходные слова сотрудника", "answered_questions": ["Старый вопрос?"],
            "rejected_scope_ids": ["previous-rejection"], "rejected_scope_entries": [[{"text": "старый отказ"}]],
            "ui_resume_progress": {"source": {"file_id": "paused-guide"}, "has_more": True},
            "instruction_offer": {"choices": [{"file_id": "unaccepted", "filename": "Другая программа.docx"}],
                                  "alternative_goals": {"unaccepted": {"g1": "Установить другую программу"}},
                                  "scope_entries": [{"text": "неподтверждённое условие альтернативы"}]}}


@pytest.mark.parametrize("interpreter", [True, False])
def test_nonchoice_offer_reply_keeps_user_frame_and_paused_rejections(monkeypatch, interpreter):
    flow = offer_flow()
    original = copy.deepcopy(flow)
    captured = []

    def text_turn(message, retained, *args):
        captured.append(copy.deepcopy(retained))
        retained["task_state"]["messages"].append(message)
        return talk_dialogue.Turn("REPLY", [("Уточняем текущую задачу.", None)], flow=retained)

    monkeypatch.setattr(talk_dialogue, "_text_turn", text_turn)
    model = SimpleNamespace()
    if interpreter:
        model.interpret_instruction_offer = lambda message, offered: {"result": "NEW_REQUEST"}
    turn = talk_dialogue._instruction_offer_turn(
        "Как подключить принтер по Bluetooth?", flow, {}, model, object(), SimpleNamespace())
    assert "instruction_offer" not in captured[0]
    for key in original.keys() - {"instruction_offer"}:
        assert captured[0][key] == original[key]
    assert "неподтверждённое условие" not in json.dumps(captured[0], ensure_ascii=False)
    assert turn.flow["task_state"]["goals"] == original["task_state"]["goals"]
    assert flow == original


def test_pure_unknown_without_pending_question_keeps_all_goals_without_inference():
    previous = compound_task()
    previous.update(question="", question_field=None)
    original = copy.deepcopy(previous)
    model = UnderstandingModel({"new_task": True, "goals": []})
    result = update_task_state("Не знаю, я в этом не разбираюсь.", previous, "", model)
    assert not model.calls and result["goals"] == original["goals"]
    assert result["facts"] == original["facts"] and result["unknown_answers"] == original["unknown_answers"]
    assert result["new_task"] is False and previous == original


@pytest.mark.parametrize("previous_question,field,stored_question,new_task,expected_marker", [
    ("", None, "", False, False),
    ("Какой вариант файла подходит?", None, "", False, False),
    ("Какой режим используется?", "connection", "Какой режим используется?", False, True),
    ("Какой режим используется?", "connection", "Другой прежний вопрос?", False, False),
    ("Какой режим используется?", "connection", "Какой режим используется?", True, False),
])
def test_native_mixed_goal_marker_only_attaches_to_a_real_pending_field(
        previous_question, field, stored_question, new_task, expected_marker):
    previous = compound_task()
    previous.update(question=stored_question, question_field=field)
    message = "Не знаю, какой файл подходит. Мне нужно открыть рабочую почту."
    model, calls = native_model(native_reply(new_task=new_task))
    update = model.understand_task(message, previous, previous_question)
    assert bool(update.unknown_answer_evidence) is expected_marker
    assert update.goals[0].evidence == message and len(calls) == 1


def test_same_task_goal_repeat_retains_existing_goals_and_exact_history(monkeypatch):
    flow = offer_flow()
    previous = copy.deepcopy(flow["task_state"])
    message = "Мне нужно открыть рабочую почту."
    model, calls = native_model(native_reply(goal_id="g1", marker=""))
    model.interpret_instruction_offer = lambda message, offered: {"result": "NEW_REQUEST"}
    model.dialogue = lambda message, context: DialogueResult(route="SUPPORT")
    monkeypatch.setattr(talk_dialogue, "_retrieve_support_turn", lambda *args: talk_dialogue.Turn(
        "REPLY", [("Ищем по уточнённой задаче.", None)], flow={}))
    settings = Settings(_env_file=None, talk_dialogue_mode="legacy")
    result = talk_dialogue._instruction_offer_turn(message, flow, {}, model, object(), settings)
    state = result.flow["task_state"]
    assert calls[0]["previous_state"] == previous
    assert set(state["goals"]) == set(previous["goals"])
    assert state["goals"]["g2"] == previous["goals"]["g2"]
    assert state["messages"] == [*previous["messages"], message]
    assert "instruction_offer" not in result.flow
    assert result.flow["rejected_scope_ids"] == flow["rejected_scope_ids"]
    assert result.flow["ui_resume_progress"] == flow["ui_resume_progress"]
    assert "Другая программа" not in json.dumps(calls[0], ensure_ascii=False)


@pytest.mark.parametrize("message", [
    "Другая задача. Мне нужно открыть рабочую почту.",
    "Не знаю. Другая задача — мне нужно открыть рабочую почту.",
])
def test_grounded_new_topic_with_mixed_unknown_has_no_stale_facts_or_goals(message, monkeypatch):
    previous = compound_task()
    previous.update(question="Какой режим используется?", question_field="connection")
    model, calls = native_model(native_reply(new_task=True))
    model.interpret_instruction_offer = lambda message, offered: {"result": "NEW_REQUEST"}
    model.dialogue = lambda message, context: DialogueResult(route="SUPPORT")
    monkeypatch.setattr(talk_dialogue, "_retrieve_support_turn", lambda *args: talk_dialogue.Turn(
        "REPLY", [("Ищем новую задачу.", None)], flow={}))
    flow = offer_flow()
    flow["task_state"] = previous
    turn = talk_dialogue._instruction_offer_turn(message, flow, {}, model, object(),
                                                Settings(_env_file=None, talk_dialogue_mode="legacy"))
    result = turn.flow["task_state"]
    assert not is_unknown_answer(message) and len(calls) == 1
    assert result["new_task"] is True and result["facts"] == []
    assert len(result["goals"]) == 1 and result["goals"]["g1"]["text"] == "Открыть рабочую почту"
    assert result["messages"] == [message] and result["unknown_answers"] == []
    assert not {"rejected_scope_ids", "rejected_scope_entries", "ui_resume_progress"} & turn.flow.keys()


def test_native_marker_cannot_be_discarded_to_hide_an_invalid_reference():
    model, calls = native_model(native_reply(marker="u999"))
    with pytest.raises(ValueError, match="InvalidTaskEvidence"):
        model.understand_task("Мне нужно открыть рабочую почту.", compound_task(), "")
    assert len(calls) == 2


def test_historical_goal_evidence_does_not_authorize_marker_normalization():
    previous = compound_task()
    message = "В этих названиях я не разбираюсь"

    def response(payload):
        old = next(alias for alias, text in payload["user_evidence"].items() if text not in message)
        return native_reply(evidence=old)

    model, calls = native_model(response)
    update = model.understand_task(message, previous, "")
    assert update.unknown_answer_evidence == message and len(calls) == 1


def test_pure_unknown_cannot_gain_a_goal_from_native_marker_normalization():
    model, calls = native_model(native_reply())
    update = model.understand_task("Не знаю", compound_task(), "")
    assert update.unknown_answer_evidence == "Не знаю" and len(calls) == 1
