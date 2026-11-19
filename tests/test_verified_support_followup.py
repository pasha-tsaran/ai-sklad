"""Guide observations keep their task and can only produce verified follow-ups."""

import copy
import hashlib

import httpx
import pytest

from app.core.config import Settings
from app.services.chat.instruction_plan import _units, serialize_plan
from app.services.chat.support_followup import diagnostic_excerpt, needs_error_text, observational_followup, verified_guide_context
from app.services.chat.talk_dialogue import (
    CHANGED, UNAVAILABLE, _guide_reply, _repeated_clarification, _retrieve_support_turn, _support_turn,
)
from app.services.chat.task_state import update_task_state
from app.services.rag.catalog import instruction_sections
from app.services.rag.interface import Candidate
from app.services.rag.talk import ApprovedTalkDocument, InstructionDecision, InstructionPlanPart
from tests.test_talk_dialogue import Knowledge
from tests.test_task_state import UnderstandingModel, fact, goal


OPENING = "Открыть рабочую программу"


def document(content):
    source = {"file_id": "00000000-0000-4000-8000-000000000099", "knowledge_id": "collection",
              "filename": "Рабочая программа.md", "sha256": hashlib.sha256(content.encode()).hexdigest()}
    candidate = Candidate("solution", source["filename"], 1, "original-chunk", content, [content], 1.0)
    return ApprovedTalkDocument({"source": source, "content": content, "images": []}, {}, candidate)


def active_flow(approved, *, sections=None, kind="ACTION"):
    state = update_task_state(OPENING, None, "", UnderstandingModel({"goals": [{**goal(OPENING), "kind": kind}]}))
    selected = tuple(sections if sections is not None else instruction_sections(approved.document))
    part = InstructionPlanPart(approved, ("g1",), (OPENING,), tuple(section["id"] for section in selected), selected)
    saved = serialize_plan([part])
    saved["cursor"] = len(_units([part]))
    return {"source": approved.document["source"], "query": OPENING, "task_query": OPENING,
            "opening_query": OPENING, "task_state": state, "instruction_plan": saved,
            "has_more": False, "selection": [], "shown": [], "understanding_confirmed": True}


class GuideModel:
    supports_guide_context = True
    supports_instruction_planning = True
    supports_source_clarification = True

    def __init__(self, *updates):
        self.updates = list(updates)
        self.contexts = []

    def understand_task(self, message, state, question, *, guide_context=None):
        self.contexts.append(copy.deepcopy(guide_context))
        return self.updates.pop(0)

    def followup(self, message, context, approved_source):
        return {"result": "NO_KNOWLEDGE"}


class DiagnosticKnowledge(Knowledge):
    uses_llm_selection = True

    def __init__(self, approved):
        super().__init__([approved])
        self.settings = Settings(_env_file=None, talk_confirm_understanding=True)
        self.diagnostics = []

    def search_with_llm(self, query, **options):
        self.diagnostics.append((query, copy.deepcopy(options)))
        return InstructionDecision("NO_KNOWLEDGE")


def test_step_error_then_error_text_then_browser_check_keep_goal_and_never_reconfirm_recipe():
    approved = document("# Вход в программу\n\n1. Откройте программу.\n\n4. Введите рабочие данные.")
    knowledge = DiagnosticKnowledge(approved)
    model = GuideModel(
        {"facts": [fact("current_step", "4"), fact("symptoms", "ошибка")]},
        {"facts": [fact("error_text", "Неверный пароль")]},
        {"facts": [fact("completed", "В браузере зашёл нормально")]},
    )
    original = active_flow(approved)
    error = _support_turn("На 4 шаге ошибка", original, {}, model, knowledge)
    assert "точный текст ошибки" in error.blocks[0][0] and not knowledge.diagnostics
    assert error.flow["instruction_plan"] == original["instruction_plan"]
    assert model.contexts[0]["read_only"] and model.contexts[0]["sources"][0]["sections"]
    password = _support_turn("Неверный пароль", error.flow, {}, model, knowledge)
    assert not password.flow.get("understanding_confirmation")
    assert "Оставить заявку" in password.blocks[0][0] and len(knowledge.diagnostics) == 1
    assert password.flow["task_state"]["goals"] == original["task_state"]["goals"]
    assert "Неверный пароль" in knowledge.diagnostics[0][0]
    assert knowledge.diagnostics[0][1]["excluded"] == [approved.document["source"]["file_id"]]
    browser = _support_turn("В браузере зашёл нормально", password.flow, {}, model, knowledge)
    assert "Оставить заявку" in browser.blocks[0][0] and len(knowledge.diagnostics) == 2
    assert browser.flow["task_state"]["goals"] == original["task_state"]["goals"]
    assert any(item["value"] == "Неверный пароль" for item in browser.flow["task_state"]["facts"])


def test_unknown_error_is_not_asked_repeatedly():
    approved = document("# Вход\n\n1. Откройте программу.")
    knowledge = DiagnosticKnowledge(approved)
    model = GuideModel({"facts": [fact("current_step", "4"), fact("symptoms", "ошибка")]})
    error = _support_turn("На 4 шаге ошибка", active_flow(approved), {}, model, knowledge)
    unknown = _support_turn("Не знаю", error.flow, {}, model, knowledge)
    assert "точный текст ошибки" not in unknown.blocks[0][0]
    assert unknown.flow["task_state"]["goals"] == error.flow["task_state"]["goals"]


def test_unknown_uses_the_saved_actual_question_without_another_understanding_call():
    approved = document("1. Откройте программу.")
    knowledge = DiagnosticKnowledge(approved)
    original = active_flow(approved)
    original["task_state"].update(question="Какой точный текст ошибки?", question_field="error_text")
    turn = _support_turn("Не знаю, я в этом не разбираюсь.", original, {}, GuideModel(), knowledge)
    assert turn.flow["task_state"]["goals"] == original["task_state"]["goals"]
    assert turn.flow["task_state"]["unknown_answers"][-1]["question"] == "Какой точный текст ошибки?"
    assert turn.flow["instruction_plan"] == original["instruction_plan"]


def test_unknown_after_an_answer_asks_observable_progress_and_keeps_the_active_guide():
    approved = document("1. Откройте программу.")
    original = active_flow(approved)
    turn = _support_turn("Не знаю", original, {}, GuideModel(), DiagnosticKnowledge(approved))
    assert "что сейчас происходит" in turn.blocks[0][0]
    assert turn.flow["instruction_plan"] == original["instruction_plan"]
    assert turn.flow["task_state"]["goals"] == original["task_state"]["goals"]


def test_an_exact_goal_clarification_retains_the_broad_opening_for_later_personal_memory():
    approved = document("Для громкости используйте исходный код настройки.")

    class MatchingKnowledge(DiagnosticKnowledge):
        def search_with_llm(self, query, **options):
            return InstructionDecision("MATCH", approved=approved, understanding=query)

    opening = "Настройка сканера"
    answer = "Уменьшить громкость сканера"
    question = "Какой результат настройки сканера нужен?"
    model = GuideModel({"new_task": True, "question": question, "question_field": "goal"},
                       {"new_task": True, "goals": [goal(answer)]})
    knowledge = MatchingKnowledge(approved)
    first = _support_turn(opening, {}, {}, model, knowledge)
    second = _support_turn(answer, first.flow, {}, model, knowledge)
    guide = second.flow["understanding_confirmation"]["guide_flow"]
    assert guide["opening_query"] == opening
    assert guide["task_state"]["messages"] == [opening, answer]


def test_independent_alternative_method_requires_fresh_scope_consent_and_excludes_sibling_guards():
    approved = document("# Рабочая программа\n\nВыберите один способ.\n\n"
                        "### Способ 1\n\nВнимание: сохраните исходное окно.\n\n1. Откройте исходное меню.\n\n"
                        "### Способ 2\n\nВнимание: сохраните изменяемый список.\n\n1. Откройте другой список.\n\n"
                        "2. Проверьте результат второго способа.")
    sections = instruction_sections(approved.document)
    choices = list(dict.fromkeys(section["exclusive_choice_section_id"] for section in sections
                                 if section.get("exclusive_choice_section_id")))
    original = active_flow(approved, sections=[section for section in sections
                                             if not section.get("exclusive_choice_section_id")
                                             or section["exclusive_choice_section_id"] == choices[0]])

    class Alternative(GuideModel):
        def followup(self, message, context, approved_source):
            index = next(index for index, text in enumerate(approved_source["excerpts"]) if "другой список" in text)
            return {"result": "EXCERPT", "source_chunk_ids": [approved_source["chunk_id"]], "step_indices": [index]}

    knowledge = DiagnosticKnowledge(approved)
    model = Alternative({"facts": [fact("symptoms", "Выполнил, результат не изменился")]})
    turn = _support_turn("Выполнил, результат не изменился", original, {}, model, knowledge)
    confirmation = turn.flow["understanding_confirmation"]
    assert not turn.block_sources
    assert confirmation["guide_flow"]["instruction_plan"]["cursor"] == 0
    chosen = confirmation["guide_flow"]["instruction_plan"]["parts"][0]["section_ids"]
    selected_text = "\n".join(section["text"] for section in sections if section["id"] in chosen)
    assert "сохраните изменяемый список" in selected_text and "результат второго способа" in selected_text
    assert "исходное меню" not in selected_text and "сохраните исходное окно" not in selected_text


@pytest.mark.parametrize("text,generic", [
    ("На четвёртом шаге появляется ошибка.", True),
    ("На 4 шаге выдало ошибку", True),
    ("Программа снова выдаёт ошибку", True),
    ("Неверный пароль", False),
    ("Ошибка 403", False),
    ("Invalid password", False),
])
def test_generic_error_reports_need_the_actual_visible_message(text, generic):
    assert needs_error_text({"facts": [fact("error_text", text)]}) is generic


@pytest.mark.parametrize("current,previous,repeated", [
    ("Вы пытаетесь войти в десктопное приложение Nextcloud Talk на компьютере Windows или в веб-браузер?",
     "Вы пытаетесь войти в приложение Nextcloud Talk на компьютере Windows или в веб-браузере?", True),
    ("Какой текст ошибки на 5 шаге?", "Какой текст ошибки на 4 шаге?", False),
    ("Что показано в окне браузера при ошибке?", "Что показано в окне приложения при ошибке?", False),
    ("Можете прислать фотографию экрана?", "Какой точный текст ошибки?", False),
])
def test_repeated_question_detection_keeps_new_steps_windows_and_alternative_observations(current, previous, repeated):
    assert _repeated_clarification(current, previous) is repeated


@pytest.mark.parametrize("failure,code", [("timeout", "RequestTimedOut"), ("invalid_json", "ValidationError")])
def test_followup_provider_failures_keep_the_observation_and_are_reported_at_the_right_stage(monkeypatch, failure, code):
    approved = document("1. Откройте программу.\n\n"
                        "Если пароль отклонён, отдельный способ: обратитесь к специалисту с текстом ошибки.")
    knowledge = DiagnosticKnowledge(approved)

    class UnavailableModel(GuideModel):
        def followup(self, message, context, approved_source):
            if failure == "timeout":
                raise httpx.ReadTimeout("Private transport detail")
            return {"result": "UNSUPPORTED"}

    failures = []
    monkeypatch.setattr("app.services.chat.talk_dialogue.report_failure", failures.append)
    original = active_flow(approved, sections=[section for section in instruction_sections(approved.document)
                                              if not section.get("parent_group_id")])
    model = UnavailableModel({"facts": [fact("error_text", "Неверный пароль")]})
    turn = _support_turn("Неверный пароль", original, {}, model, knowledge)
    assert turn.blocks == [(UNAVAILABLE, None)]
    assert turn.flow["instruction_plan"] == original["instruction_plan"]
    assert turn.flow["task_state"]["goals"] == original["task_state"]["goals"]
    assert any(item["value"] == "Неверный пароль" for item in turn.flow["task_state"]["facts"])
    assert [(error.stage, error.code) for error in failures] == [("instruction_followup", code)]
    assert not knowledge.diagnostics


def test_performed_recipe_sections_are_not_offered_as_new_diagnostic_excerpts():
    approved = document("1. Откройте программу.\n\n2. Введите рабочие данные.")
    knowledge = DiagnosticKnowledge(approved)

    class NoRepeatedRecipe(GuideModel):
        def followup(self, *args):
            raise AssertionError("The source has no additional diagnostic fragment")

    original = active_flow(approved)
    turn = _support_turn("Неверный пароль", original, {},
                         NoRepeatedRecipe({"facts": [fact("error_text", "Неверный пароль")]}), knowledge)
    assert turn.action == "UNRESOLVED" and "Оставить заявку" in turn.blocks[0][0]
    assert not turn.flow.get("understanding_confirmation")
    assert turn.flow["instruction_plan"] == original["instruction_plan"]


def test_observations_do_not_hide_a_different_connection_or_goal():
    approved = document("1. Настройте программу.")
    state = active_flow(approved)["task_state"]
    observed = copy.deepcopy(state)
    observed["facts"] = [fact("error_text", "Ошибка подключения")]
    assert observational_followup(state, observed)
    observed["facts"].append(fact("connection", "Bluetooth"))
    assert not observational_followup(state, observed)
    observed["facts"] = [fact("error_text", "Ошибка подключения")]
    observed["goals"]["g1"]["text"] = "Подключить другой принтер"
    assert not observational_followup(state, observed)


def test_followup_context_is_bounded_and_a_changed_source_is_discarded_without_losing_the_task():
    approved = document("# Программа\n\n" + "\n\n".join(f"{number}. Проверьте действие {'текст ' * 200}"
                                                       for number in range(1, 40)))
    knowledge = DiagnosticKnowledge(approved)
    flow = active_flow(approved)
    import json
    assert len(json.dumps(verified_guide_context(knowledge, flow), ensure_ascii=False)) <= 8000
    approved.document["source"] = {**approved.document["source"], "sha256": "b" * 64}
    turn = _support_turn("На 4 шаге ошибка", flow, {}, GuideModel(), knowledge)
    assert turn.blocks == [(CHANGED, None)] and "source" not in turn.flow
    assert turn.flow["task_state"]["goals"] == flow["task_state"]["goals"]


def test_displayed_status_is_scoped_to_each_original_in_a_multi_source_plan():
    first = document("1. Откройте приложение.")
    second = document("1. Откройте приложение.")
    second.document["source"] = {**second.document["source"], "file_id": "another-file"}
    parts = []
    for approved in [first, second]:
        sections = tuple(instruction_sections(approved.document))
        parts.append(InstructionPlanPart(approved, ("g1",), (OPENING,), tuple(section["id"] for section in sections), sections))
    saved = serialize_plan(parts)
    saved["cursor"] = 1
    knowledge = Knowledge([first, second])
    context = verified_guide_context(knowledge, {"instruction_plan": saved, "has_more": True})
    assert all(section["shown"] for section in context["sources"][0]["sections"])
    assert not any(section["shown"] for section in context["sources"][1]["sections"])


def test_ui_question_can_repeat_one_relevant_source_fragment_without_new_consent():
    approved = document("1. Нажмите кнопку «Печать».")
    knowledge = DiagnosticKnowledge(approved)

    class Explain(GuideModel):
        def followup(self, message, context, approved_source):
            return {"result": "EXCERPT", "source_chunk_ids": [approved_source["chunk_id"]], "step_indices": [0]}

    model = Explain({"facts": [fact("symptoms", "Не вижу кнопку")]})
    original = active_flow(approved)
    original["has_more"] = True
    turn = _support_turn("Не вижу кнопку", original, {}, model, knowledge)
    assert turn.action == "REPEAT_GUIDE" and not turn.flow.get("understanding_confirmation")
    assert "Нажмите кнопку «Печать»" in turn.blocks[0][0]
    assert turn.block_sources[0] == approved.document["source"] and not knowledge.diagnostics
    assert turn.flow["instruction_plan"] == original["instruction_plan"] and turn.flow["has_more"]
    assert "основной инструкции" in turn.blocks[-1][0]


def test_conditional_followup_keeps_its_complete_branch_and_does_not_repeat_the_parent():
    approved = document("# Программа\n\n1. Откройте исходное меню.\n\n"
                        "Если экран пустой, отдельный способ: нажмите F5.\n\n2. Проверьте появление окна.")
    sections = instruction_sections(approved.document)
    flow = active_flow(approved, sections=[section for section in sections if not section.get("parent_group_id")])
    knowledge = DiagnosticKnowledge(approved)

    class ChooseBranch(GuideModel):
        def followup(self, message, context, approved_source):
            index = next(index for index, text in enumerate(approved_source["excerpts"]) if text.startswith("Если экран"))
            return {"result": "EXCERPT", "source_chunk_ids": [approved_source["chunk_id"]], "step_indices": [index]}

    selected = diagnostic_excerpt(knowledge, flow, "Экран пустой", {}, ChooseBranch())
    body = "\n".join(text for text, _ in selected["blocks"])
    assert "Если экран пустой" in body and "Проверьте появление окна" in body
    assert "Откройте исходное меню" not in body


def test_current_source_fragment_does_not_claim_an_unanswered_goal_from_a_partial_plan():
    approved = document("1. Нажмите кнопку «Печать».")
    knowledge = DiagnosticKnowledge(approved)
    original = active_flow(approved)
    original["task_state"]["goals"]["g2"] = {"text": "Отдельная неподтверждённая задача", "evidence": "Отдельная задача"}
    original.update(coverage_mode="partial", covered_goals=[OPENING], missing_goals=["Отдельная неподтверждённая задача"])

    class Explain(GuideModel):
        def followup(self, message, context, approved_source):
            return {"result": "EXCERPT", "source_chunk_ids": [approved_source["chunk_id"]], "step_indices": [0]}

    selected = diagnostic_excerpt(knowledge, original, "Где кнопка печати?", {}, Explain(), allow_repeat=True)
    assert selected["plan"]["parts"][0]["goal_ids"] == ["g1"]
    assert selected["plan"]["parts"][0]["goals"] == [OPENING]
    assert original["missing_goals"] == ["Отдельная неподтверждённая задача"]


def test_a_different_diagnostic_source_requires_new_scope_consent_without_overwriting_original_goal():
    approved = document("# Рабочая программа\n\n1. Откройте приложение.")
    alternative = document("# Указанное сообщение об ошибке\n\n1. Выполните подтверждённую проверку.")
    alternative.document["source"] = {**alternative.document["source"], "file_id": "another-original"}

    class MoreKnowledge(DiagnosticKnowledge):
        def search_with_llm(self, query, **options):
            self.diagnostics.append((query, copy.deepcopy(options)))
            return InstructionDecision("MATCH", approved=alternative, understanding=query)

    knowledge = MoreKnowledge(approved)
    model = GuideModel({"facts": [fact("error_text", "Неверный пароль")]})
    original = active_flow(approved)
    turn = _support_turn("Неверный пароль", original, {}, model, knowledge)
    confirmation = turn.flow["understanding_confirmation"]
    assert confirmation["source"] == alternative.document["source"]
    assert "Неверный пароль" in confirmation["summary"]
    assert turn.flow["task_state"]["goals"] == original["task_state"]["goals"]
    assert confirmation["guide_flow"]["task_state"]["goals"] == original["task_state"]["goals"]
    assert all(picture is None for _, picture in turn.blocks)


def test_diagnostic_clarification_preserves_the_active_source_and_supplies_readonly_scope_to_search():
    approved = document("# Вход в установленную программу\n\n1. Откройте приложение.")

    class ClarifyingKnowledge(DiagnosticKnowledge):
        def search_with_llm(self, query, **options):
            self.diagnostics.append((query, copy.deepcopy(options)))
            return InstructionDecision("CLARIFY", question="Какой заголовок у окна с ошибкой?")

    knowledge = ClarifyingKnowledge(approved)
    original = active_flow(approved)
    model = GuideModel({"facts": [fact("error_text", "Неверный пароль")]})
    turn = _support_turn("Неверный пароль", original, {}, model, knowledge)
    assert turn.flow["source"] == original["source"]
    assert turn.flow["instruction_plan"] == original["instruction_plan"]
    assert turn.flow["clarification_question"] == "Какой заголовок у окна с ошибкой?"
    context = knowledge.diagnostics[0][1]["context"]
    assert context["diagnostic_request"] is True
    assert context["active_guide_context"]["read_only"] is True
    assert context["active_guide_context"]["sources"][0]["source"] == original["source"]


def test_diagnostic_source_preserves_an_unanswered_goal_and_its_partial_confirmation_boundary():
    approved = document("# Рабочая программа\n\n1. Откройте приложение.")
    alternative = document("# Проверка указанной ошибки\n\n1. Выполните подтверждённую проверку.")
    alternative.document["source"] = {**alternative.document["source"], "file_id": "another-original"}

    class MoreKnowledge(DiagnosticKnowledge):
        def __init__(self, original):
            super().__init__(original)
            self.documents.append(alternative)

        def search_with_llm(self, query, **options):
            self.diagnostics.append((query, copy.deepcopy(options)))
            return InstructionDecision("MATCH", approved=alternative, understanding=query)

    knowledge = MoreKnowledge(approved)
    initial = active_flow(approved)
    initial["task_state"]["goals"]["g2"] = {"text": "Отдельная неподтверждённая задача", "evidence": "Отдельная задача"}
    initial.update(coverage_mode="partial", covered_goals=[OPENING], missing_goals=["Отдельная неподтверждённая задача"])
    model = GuideModel({"facts": [fact("error_text", "Неверный пароль")]})
    turn = _support_turn("Неверный пароль", initial, {}, model, knowledge)
    confirmation = turn.flow["understanding_confirmation"]
    assert set(knowledge.diagnostics[0][1]["context"]["requested_goals"]) == {"g1"}
    assert confirmation["guide_flow"]["coverage_mode"] == "partial"
    assert confirmation["guide_flow"]["missing_goals"] == ["Отдельная неподтверждённая задача"]
    assert "Отдельная неподтверждённая задача" in turn.blocks[0][0]
    assert turn.flow["task_state"]["goals"] == initial["task_state"]["goals"]


def test_discovery_aliases_and_explicit_kinds_are_forwarded_without_rewriting_legacy_goals():
    approved = document("# Программа\n\n1. Откройте приложение.")
    knowledge = DiagnosticKnowledge(approved)
    state = active_flow(approved)["task_state"]
    state["goals"]["g1"]["search_queries"] = ["desktop application login"]
    state["goals"]["legacy"] = {"text": "Другая старая цель", "evidence": "Другая старая цель"}
    _retrieve_support_turn(OPENING, {"task_state": state, "task_query": OPENING}, {}, GuideModel(), knowledge)
    options = knowledge.diagnostics[0][1]["context"]
    assert options["requested_goal_kinds"] == {"g1": "ACTION"}
    assert options["goal_search_queries"] == {"g1": ["desktop application login"]}
    assert options["requested_goals"] == {"g1": OPENING, "legacy": "Другая старая цель"}


@pytest.mark.parametrize("kind,expected", [("LOOKUP", "Нашли нужную информацию?"),
                                          ("ACTION", "Удалось выполнить эти действия?"),
                                          (None, "Удалось выполнить эти действия?")])
def test_final_question_matches_explicit_goal_kind_and_preserves_legacy_action_wording(kind, expected):
    approved = document("Внутренний номер руководства — 126.")
    knowledge = DiagnosticKnowledge(approved)
    flow = active_flow(approved)
    flow["instruction_plan"]["cursor"] = 0
    if kind is None:
        flow["task_state"]["goals"]["g1"].pop("kind", None)
    else:
        flow["task_state"]["goals"]["g1"]["kind"] = kind
    turn = _guide_reply(None, [], flow, knowledge)
    assert turn.blocks[-1] == (expected, None)
