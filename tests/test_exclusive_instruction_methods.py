"""Source-declared alternative methods never become one sequential recipe."""

import copy

import pytest

from app.services.chat.instruction_plan import restore_plan, serialize_plan
from app.services.rag.catalog import instruction_sections
from tests.test_instruction_planning import context, selected_part
from tests.test_scanner_original_retrieval import scanner_provider
from tests.test_talk_knowledge import DOC


COMMON = "Перед началом сохраните рабочие данные."
CHOICE = "Выберите один способ, а не выполняйте все подряд."
FIRST = "Откройте синий регулятор и проверьте результат."
SECOND = "Удалите только временную запись и создайте её заново."
SECOND_GUARD = "Не удаляйте сохранённые рабочие документы."
THIRD = "Вручную проверьте временные каталоги текущего пользователя."


def knowledge_source(choice=CHOICE):
    text = "\n\n".join(["# Новый рабочий материал", "## Восстановление клиента", COMMON, choice,
                          "### Способ 1: регулятор", FIRST,
                          "### Способ 2: запись", SECOND, SECOND_GUARD,
                          "### Способ 3: каталоги", THIRD])
    return scanner_provider([(DOC, "Новый рабочий материал.md", text.encode())])[0]


def test_explicit_one_of_methods_have_stable_scopes_and_leave_the_original_unchanged():
    knowledge = knowledge_source()
    approved = knowledge.read(DOC)
    original = copy.deepcopy(approved.document)
    sections = instruction_sections(approved.document)
    methods = [section for section in sections if section["text"] in (FIRST, SECOND, THIRD)]
    assert len({section["group_id"] for section in methods}) == 3
    assert len({section["exclusive_set_id"] for section in methods}) == 1
    assert all(section["exclusive_parent_group_id"] == section["exclusive_set_id"] for section in methods)
    assert all(section["exclusive_choice_section_id"] in {item["id"] for item in sections} for section in methods)
    assert not any("exclusive_set_id" in section for section in sections if section["text"] in (COMMON, CHOICE))
    assert instruction_sections(approved.document) == sections and approved.document == original


@pytest.mark.parametrize("select_all", [False, True])
def test_first_method_keeps_common_prerequisites_without_sibling_actions_or_sibling_guards(select_all):
    knowledge = knowledge_source()

    def choose(payload):
        part = selected_part(payload, DOC, "g1", FIRST)
        if select_all:
            part["section_ids"] = list(payload["candidate_sections"][DOC])
        return {"result": "MATCH", **part}

    knowledge.selector = choose
    decision = knowledge.search_with_llm("Хочу восстановить клиент через синий регулятор.",
                                        context("Восстановить клиент через синий регулятор"))
    text = "\n".join(section["text"] for section in decision.plan[0].sections)
    assert COMMON in text and CHOICE in text and FIRST in text
    assert SECOND not in text and SECOND_GUARD not in text and THIRD not in text
    saved = serialize_plan(decision.plan)
    assert restore_plan(knowledge, saved)[0].sections == decision.plan[0].sections


@pytest.mark.parametrize("notice", ["Выполните способы последовательно.", "Выберите один цвет регулятора."])
def test_method_headings_without_an_explicit_one_of_method_rule_keep_the_existing_procedure(notice):
    sections = instruction_sections(knowledge_source(notice).read(DOC).document)
    methods = [section for section in sections if section["text"] in (FIRST, SECOND, THIRD)]
    assert len({section["group_id"] for section in methods}) == 1
    assert not any("exclusive_set_id" in section for section in sections)


def test_one_of_notice_does_not_split_ordinary_numbered_steps():
    approved = knowledge_source().read(DOC)
    approved.document["content"] = approved.document["content"].replace("### Способ", "### Шаг")
    sections = instruction_sections(approved.document)
    assert not any("exclusive_set_id" in section for section in sections)
