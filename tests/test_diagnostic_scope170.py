"""Different setup conditions must not replace a performed guide's diagnosis."""

import copy

import pytest

from app.services.chat.instruction_plan import _units, serialize_plan
from app.services.chat.support_followup import NO_DIAGNOSTIC_STEPS
from app.services.chat.talk_dialogue import _support_turn
from app.services.rag.catalog import instruction_sections
from app.services.rag.talk import InstructionDecision, InstructionPlanPart
from tests.conftest import env as env
from tests.test_openwebui_instruction_images import instruction
from tests.test_talk_dialogue import assistants
from tests.test_talk_dialogue import talk as talk
from tests.test_talk_knowledge import DOC, OTHER
from tests.test_talk_model_dialogue import model_chat as model_chat
from tests.test_task_state import fact, goal
from tests.test_verified_support_followup import DiagnosticKnowledge, GuideModel, active_flow, document


class RelatedKnowledge(DiagnosticKnowledge):
    def __init__(self, approved, *alternatives):
        super().__init__(approved)
        self.documents.extend(alternatives)
        self.alternatives = alternatives

    def search_with_llm(self, query, **options):
        self.diagnostics.append((query, copy.deepcopy(options)))
        return InstructionDecision("RELATED", alternatives=self.alternatives)


def related_document(text, file_id):
    approved = document(text)
    approved.document["source"]["file_id"] = file_id
    return approved


@pytest.mark.parametrize("opening,conditions,error,completed,other_setups", [
    (
        "Войти в корпоративное приложение на компьютере",
        [fact("equipment_type", "компьютер"), fact("equipment_model", "Windows"),
         fact("application", "установленное приложение")],
        "Неверный пароль",
        "В браузере с этими данными захожу нормально",
        ["Установка приложения на телефоне", "Смена пароля в удалённом рабочем столе", "Вход в почту"],
    ),
    (
        "Напечатать документ на рабочем принтере",
        [fact("equipment_type", "принтер"), fact("connection", "Ethernet")],
        "Задание осталось в очереди",
        "С другого компьютера этот принтер печатает",
        ["Первое подключение принтера по Wi-Fi", "Настройка печати по Bluetooth"],
    ),
])
def test_related_setups_after_step_error_keep_active_guide_through_later_observation(
    opening, conditions, error, completed, other_setups,
):
    approved = document("1. Откройте рабочее окно.\n\n4. Выполните указанное действие.")
    original = active_flow(approved)
    original["task_state"]["goals"]["g1"] = {"text": opening, "evidence": opening, "kind": "ACTION"}
    original["task_state"]["facts"] = conditions
    original["task_state"]["messages"] = [opening]
    sections = tuple(instruction_sections(approved.document))
    part = InstructionPlanPart(approved, ("g1",), (opening,), tuple(s["id"] for s in sections), sections)
    original["instruction_plan"] = serialize_plan([part])
    original["instruction_plan"]["cursor"] = len(_units([part]))
    original.update(query=opening, task_query=opening, opening_query=opening)
    alternatives = [related_document("1. " + text + ".", "other-" + str(index))
                    for index, text in enumerate(other_setups)]
    knowledge = RelatedKnowledge(approved, *alternatives)
    model = GuideModel(
        {"facts": [fact("current_step", "4"), fact("symptoms", "ошибка")]},
        {"facts": [fact("error_text", error)]},
        {"facts": [fact("completed", completed)]},
    )
    asked = _support_turn("На 4 шаге ошибка", original, {}, model, knowledge)
    assert "точный текст ошибки" in asked.blocks[0][0] and not knowledge.diagnostics
    password = _support_turn(error, asked.flow, {}, model, knowledge)
    continued = _support_turn(completed, password.flow, {}, model, knowledge)
    assert len(knowledge.diagnostics) == 2
    for turn in (password, continued):
        assert turn.action == "UNRESOLVED" and NO_DIAGNOSTIC_STEPS in turn.blocks[0][0]
        assert "instruction_offer" not in turn.flow and "understanding_confirmation" not in turn.flow
        assert turn.flow["source"] == original["source"]
        assert turn.flow["instruction_plan"] == original["instruction_plan"]
        assert turn.flow["task_state"]["goals"] == original["task_state"]["goals"]
        assert all(photo is None for _, photo in turn.blocks)
    facts = continued.flow["task_state"]["facts"]
    assert all(item in facts for item in conditions)
    assert any(item["key"] == "current_step" and item["value"] == "4" for item in facts)
    assert any(item["key"] == "error_text" and item["value"] == error for item in facts)
    assert any(item["key"] == "completed" and item["value"] == completed for item in facts)
    context = knowledge.diagnostics[-1][1]["context"]
    assert context["diagnostic_request"] is True
    assert context["active_guide_context"]["sources"][0]["source"] == original["source"]
    assert completed in str(knowledge.diagnostics[-1])
    assert all(item["value"] in str(context["task_state"]["facts"]) for item in conditions)


def test_explicit_new_task_after_a_guide_keeps_normal_related_offers_available():
    approved = document("1. Откройте приложение.")
    alternative = related_document("1. Подключите принтер по Wi-Fi.", "wifi-source")
    knowledge = RelatedKnowledge(approved, alternative)
    model = GuideModel({
        "new_task": True, "goals": [goal("Подключить принтер по Bluetooth")],
        "facts": [fact("equipment_type", "принтер")],
    })
    turn = _support_turn("Другая задача. Подключить принтер по Bluetooth", active_flow(approved), {}, model, knowledge)
    assert "instruction_offer" in turn.flow
    assert turn.flow["instruction_offer"]["choices"] == [alternative.document["source"]]
    assert "diagnostic_request" not in knowledge.diagnostics[-1][1]["context"]


def test_worker_keeps_one_attempt_after_error_and_browser_observation_instead_of_related_setup(model_chat):
    opening = "Войти в рабочее приложение на компьютере Windows."
    requested = "Войти в рабочее приложение"
    captions = ["Откройте рабочее приложение.", "Укажите рабочий сервер.",
                "Выберите рабочую учётную запись.", "Введите рабочие данные для входа."]
    alternative = "Для входа на телефоне откройте мобильное приложение."
    chat = model_chat([
        {"result": "MATCH", "file_id": DOC,
         "covered_goals": [{"goal_id": "g1", "evidence": captions[-1],
                            "support_kind": "IMAGE", "image_numbers": [4]}]},
        {"result": "RELATED", "alternatives": [{"file_id": OTHER, "evidence": alternative}]},
        {"result": "RELATED", "alternatives": [{"file_id": OTHER, "evidence": alternative}]},
    ], rows=[(DOC, "Вход в приложение на компьютере.docx", instruction(captions)),
             (OTHER, "Вход в приложение на телефоне.docx", instruction([alternative]))])
    chat.talk.settings.talk_confirm_understanding = True
    chat.model.supports_guide_context = True
    responses = [
        {"goals": [goal(requested)],
         "facts": [fact("equipment_type", "компьютер", "компьютере"), fact("equipment_model", "Windows")]},
        {"facts": [fact("current_step", "4"), fact("symptoms", "ошибка")]},
        {"facts": [fact("error_text", "неверный пароль")]},
        {"facts": [fact("completed", "В браузере с этими данными захожу нормально")]},
    ]

    def understand(message, previous, previous_question, *, guide_context=None):
        if previous and previous.get("goals"):
            assert guide_context["read_only"]
        return responses.pop(0)

    chat.model.understand_task = understand
    chat.send(opening)
    guide = chat.send("да, правильно", 2)
    assert len(guide["attempts"]) == 1
    asked = chat.send("На 4 шаге ошибка", 3)
    assert "точный текст ошибки" in assistants(asked)[-1][0]
    error = chat.send("Пишет неверный пароль", 4)
    later = chat.send("В браузере с этими данными захожу нормально", 5)
    assert not responses and len(chat.model.calls) == 3
    for result in (error, later):
        assert "instruction_offer" not in result["flow"]
        assert result["flow"]["source"] == guide["flow"]["source"]
        for key in ("instruction_plan", "selection", "next_offset", "shown", "has_more"):
            assert result["flow"].get(key) == guide["flow"].get(key)
        assert result["flow"]["task_state"]["goals"] == guide["flow"]["task_state"]["goals"]
        assert result["incident"].state == "WAITING_FEEDBACK" and result["incident"].failed_attempts == 0
        assert len(result["attempts"]) == 1 and result["incident"].pending_action is None
        assert NO_DIAGNOSTIC_STEPS in assistants(result)[-1][0]
    assert all(photo is None for _, photo in assistants(later)[len(assistants(guide)):])
    assert "браузере" in str(chat.model.calls[-1]["payload"]).casefold()
