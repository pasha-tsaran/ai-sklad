"""Difficulties preserve the active work and cannot restart generic intake."""

import copy
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.services.chat.talk_dialogue import _contextual_question, _restarts_task, _retrieve_support_turn, _support_turn
from app.services.rag.talk import InstructionDecision
from tests.test_talk_dialogue import Knowledge, approved
from tests.test_task_state import UnderstandingModel, compound_task


@pytest.mark.parametrize("message", ["Не получилось.", "Не разобрался", "Застряла"])
@pytest.mark.parametrize("legacy", [False, True])
def test_plain_difficulty_asks_observable_followup_without_replaying_or_reconfirming(message, legacy):
    flow = {"source": {"file_id": "existing"}, "query": "Текущая задача",
            "instruction_plan": {"parts": [{"source": "immutable"}]}, "next_offset": 3, "has_more": True}
    if not legacy:
        flow["task_state"] = compound_task()
    original = copy.deepcopy(flow)
    model = UnderstandingModel()
    knowledge = SimpleNamespace(settings=Settings(_env_file=None, talk_confirm_understanding=True))

    turn = _support_turn(message, flow, {}, model, knowledge)

    assert turn.action == "REPLY" and len(turn.blocks) == 1
    assert turn.blocks[0][0].count("?") == 1 and turn.blocks[0][1] is None
    assert "действии" in turn.blocks[0][0] and "происходит" in turn.blocks[0][0]
    assert not turn.flow.get("understanding_confirmation")
    assert turn.flow["instruction_plan"] == original["instruction_plan"]
    assert turn.flow["next_offset"] == 3 and turn.flow["has_more"]
    assert not model.calls and flow == original
    if not legacy:
        assert turn.flow["task_state"]["goals"] == original["task_state"]["goals"]
        assert turn.flow["task_state"]["question_field"] == "current_step"


def test_observable_fallback_does_not_invent_an_error_or_completed_instruction_step():
    question = _contextual_question({"task_state": {"facts": [{"key": "symptoms", "value": "На бумаге нет картинок"}]}})
    assert "ошибк" not in question and "шаг" not in question


def test_general_setup_fallback_asks_for_a_desired_result_without_assuming_a_failure():
    question = _contextual_question({"task_state": {"facts": [{"key": "equipment_type", "value": "устройство"}],
                                                   "goals": {}}})
    assert "результат" in question and "ошибк" not in question and "шаг" not in question


@pytest.mark.parametrize("question", [
    "Что именно нужно сделать? Опишите желаемый результат.",
    "Что нужно сделать?", "Какой желаемый результат вам нужен?",
])
def test_known_task_cannot_restart_with_an_unqualified_goal_question(question):
    assert _restarts_task(question)


def test_invalid_source_question_retries_selection_without_substituting_a_generic_intake_question():
    document = approved(count=1)

    class SelectingKnowledge(Knowledge):
        uses_llm_selection = True

        def __init__(self):
            super().__init__([document])
            self.settings = Settings(_env_file=None, talk_confirm_understanding=True)
            self.selections = []

        def search_with_llm(self, problem, **options):
            self.selections.append(options)
            if len(self.selections) == 1:
                return InstructionDecision("CLARIFY", question="Что именно нужно сделать?")
            return InstructionDecision("MATCH", approved=document, understanding="Изменить громкость")

    knowledge = SelectingKnowledge()
    flow = {"task_state": compound_task(), "task_query": "Изменить громкость"}
    turn = _retrieve_support_turn("Нужна громкость", flow, {}, UnderstandingModel(), knowledge)

    assert len(knowledge.selections) == 2
    assert knowledge.selections[1]["context"]["clarification_retry"] is True
    assert turn.flow.get("understanding_confirmation")
    assert "Что именно нужно сделать" not in turn.blocks[0][0]
