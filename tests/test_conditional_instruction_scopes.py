"""Explicit alternative remedies keep source conditions and separate photographs."""

import copy
import io
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape
from zipfile import ZipFile

import pytest

from app.services.rag.catalog import instruction_sections
from app.services.chat.instruction_plan import restore_plan, serialize_plan
from tests.test_instruction_planning import context, selected_part
from integrations.openwebui.approved_instruction_images import W
from tests.test_openwebui_instruction_images import instruction, office
from tests.test_scanner_original_retrieval import scanner_provider
from tests.test_talk_knowledge import DOC, KB


PREREQUISITE = "Перед настройкой сохраните текущие параметры."
STEPS = ["1. Откройте окно печати.", "2. Снимите «Размер из файла».", "3. Проверьте одну пробную страницу."]
ALTERNATIVE = ("Если после этого отсутствуют рисунки, отдельный способ: "
               "откройте дополнительные параметры и включите «Печатать как рисунок».")
CAPTIONS = ["Первое окно. Снимите «Размер из файла».",
            "Дополнительное окно. Включите «Печатать как рисунок»."]


def source(captions=CAPTIONS):
    body = "".join("<w:p><w:r><w:t>" + escape(text) + "</w:t></w:r></w:p>"
                   for text in [PREREQUISITE, *STEPS, ALTERNATIVE])
    return scanner_provider([(DOC, "Новый материал печати.docx", instruction(captions, extra_paragraph=body))])[0]


def test_explicit_conditional_method_has_source_scoped_parent_and_condition_ids():
    knowledge = source()
    approved = knowledge.read(DOC)
    sections = instruction_sections(approved.document)
    original = copy.deepcopy(approved.document)
    primary = next(section for section in sections if section["text"] == STEPS[1])
    branch = next(section for section in sections if section["text"] == ALTERNATIVE)
    pictures = {number: section for section in sections for number in section["image_numbers"]}

    assert branch["parent_group_id"] == primary["group_id"]
    assert branch["condition_section_id"] == branch["id"]
    assert branch["condition_text"] == "Если после этого отсутствуют рисунки"
    assert branch["condition_text"] in branch["text"]
    assert branch["group_id"] != primary["group_id"]
    assert pictures[1]["group_id"] == primary["group_id"]
    assert pictures[2]["group_id"] == branch["group_id"]
    assert pictures[2]["condition_section_id"] == branch["id"]
    assert pictures[2]["condition_text"] == branch["condition_text"]
    assert "parent_group_id" not in primary and "condition_section_id" not in primary
    assert instruction_sections(approved.document) == sections and approved.document == original
    changed = {**approved.document, "source": {**approved.document["source"], "sha256": "0" * 64}}
    changed_branch = next(section for section in instruction_sections(changed) if section["text"] == ALTERNATIVE)
    assert changed_branch["group_id"] != branch["group_id"]
    assert changed_branch["condition_section_id"] != branch["condition_section_id"]


@pytest.mark.parametrize("select_all_ids", [False, True])
def test_primary_proof_keeps_all_primary_steps_without_activating_unanchored_child(select_all_ids):
    knowledge = source()

    def select(payload):
        part = selected_part(payload, DOC, "g1", "Снимите «Размер из файла»")
        if select_all_ids:
            part["section_ids"] = list(payload["candidate_sections"][DOC])
        return {"result": "MATCH", **part}

    knowledge.selector = select
    decision = knowledge.search_with_llm("Неправильный размер бумаги, рисунки печатаются нормально.",
                                        context("Исправить только размер бумаги"))
    sections = decision.plan[0].sections
    text = "\n".join(section["text"] for section in sections)
    assert PREREQUISITE in text and all(step in text for step in STEPS)
    assert CAPTIONS[0] in text and CAPTIONS[1] not in text and ALTERNATIVE not in text
    assert [number for section in sections for number in section["image_numbers"]] == [1]


def test_child_proof_inherits_complete_parent_steps_prerequisite_and_photographs():
    knowledge = source()
    knowledge.selector = lambda payload: {
        "result": "MATCH", **selected_part(payload, DOC, "g1", "Включите «Печатать как рисунок»"),
    }
    decision = knowledge.search_with_llm("После проверки размера бумаги рисунки всё равно не печатаются.",
                                        context("Исправить печать рисунков после основной проверки"))
    sections = decision.plan[0].sections
    text = "\n".join(section["text"] for section in sections)
    assert PREREQUISITE in text and all(step in text for step in STEPS)
    assert ALTERNATIVE in text and all(caption in text for caption in CAPTIONS)
    assert [number for section in sections for number in section["image_numbers"]] == [1, 2]


@pytest.mark.parametrize("one_part", [False, True])
def test_parent_then_child_goals_merge_the_same_ancestor_steps_and_images_once(one_part):
    knowledge = source()

    def select(payload):
        primary = selected_part(payload, DOC, "g1", "Снимите «Размер из файла»")
        child = selected_part(payload, DOC, "g2", "Включите «Печатать как рисунок»")
        if one_part:
            primary["covered_goals"] += child["covered_goals"]
            primary["section_ids"] += child["section_ids"]
            return {"result": "MATCH", **primary}
        return {"result": "PLAN", "parts": [primary, child]}

    knowledge.selector = select
    result = knowledge.search_with_llm("Проверить размер, затем способ печати рисунков.",
                                      context("Исправить размер бумаги", "После проверки исправить рисунки"))
    assert result.result == "MATCH" and len(result.plan) == 1
    part = result.plan[0]
    assert part.goal_ids == ("g1", "g2")
    assert len(part.section_ids) == len(set(part.section_ids))
    assert [number for section in part.sections for number in section["image_numbers"]] == [1, 2]
    assert all(sum(section["text"] == step for section in part.sections) == 1 for step in STEPS)
    assert [section["ordinal"] for section in part.sections] == sorted(section["ordinal"] for section in part.sections)


@pytest.mark.parametrize("rejected", [False, True])
def test_child_before_requested_parent_requires_order_consent_without_repeating_rejected_offer(rejected):
    knowledge = source()
    knowledge.selector = lambda payload: {"result": "PLAN", "parts": [
        selected_part(payload, DOC, "g1", "Включите «Печатать как рисунок»"),
        selected_part(payload, DOC, "g2", "Снимите «Размер из файла»"),
    ]}
    request = context("Сначала исправить рисунки", "Потом исправить размер бумаги")
    if rejected:
        request["rejected_goal_orders"] = [["g2", "g1"]]
    result = knowledge.search_with_llm("Сначала рисунки, потом размер бумаги.", request)
    assert result.result == "CLARIFY" and not result.plan
    assert result.proposed_goal_order == (() if rejected else ("g2", "g1"))
    assert len(result.clarification_sources) == 1


def test_late_inline_parent_photo_does_not_keep_stale_child_metadata():
    knowledge = source()
    with knowledge._client() as client:
        original = knowledge._original(client, DOC)
    with ZipFile(io.BytesIO(original)) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    tree = ET.fromstring(parts["word/document.xml"])
    body = tree.find(W + "body")
    drawings = [paragraph for paragraph in body if paragraph.find(".//" + W + "drawing") is not None]
    for paragraph in drawings:
        body.remove(paragraph)
        body.append(paragraph)
    parts["word/document.xml"] = ET.tostring(tree, encoding="utf-8")
    knowledge, _ = scanner_provider([(DOC, "Иллюстрации после действий.docx", office(parts))])
    sections = instruction_sections(knowledge.read(DOC).document)
    primary = next(section for section in sections if section["text"] == STEPS[1])
    picture = next(section for section in sections if section["image_numbers"] == [1])
    assert picture["group_id"] == primary["group_id"]
    assert "parent_group_id" not in picture and "condition_section_id" not in picture
    assert "condition_text" not in picture


def test_unselected_ambiguous_composite_picture_keeps_source_data_and_is_omitted():
    caption = "Снимите «Размер из файла», затем включите «Печатать как рисунок»."
    knowledge = source([caption])
    parsed = instruction_sections(knowledge.read(DOC).document)
    image = next(section for section in parsed if section["image_numbers"])
    assert image["scope_ambiguous"] and len(image["scope_group_ids"]) == 2
    assert image["image_numbers"] == [1] and image["text"] == caption
    knowledge.selector = lambda payload: {
        "result": "MATCH", **selected_part(payload, DOC, "g1", "2. Снимите «Размер из файла»"),
    }
    result = knowledge.search_with_llm("Неправильный размер бумаги", context("Исправить размер бумаги"))
    assert result.result == "MATCH"
    delivered = [section for part in result.plan for section in part.sections]
    text = "\n".join(section["text"] for section in delivered)
    assert PREREQUISITE in text and all(step in text for step in STEPS)
    assert ALTERNATIVE not in text and caption not in text
    assert not any(section["image_numbers"] for section in delivered)
    assert instruction_sections(knowledge.read(DOC).document) == parsed


def test_composite_photo_is_safe_when_all_of_its_actual_procedure_owners_are_requested():
    caption = "Снимите «Размер из файла», затем включите «Печатать как рисунок»."
    knowledge = source([caption])

    def choose(payload):
        part = selected_part(payload, DOC, "g1", STEPS[1])
        child = selected_part(payload, DOC, "g2", ALTERNATIVE[:50])
        part["section_ids"] += child["section_ids"]
        part["covered_goals"] += child["covered_goals"]
        return {"result": "MATCH", **part}

    knowledge.selector = choose
    result = knowledge.search_with_llm("Исправить размер бумаги, затем проверить печать рисунков.",
                                      context("Исправить размер бумаги", "После проверки исправить рисунки"))
    assert result.result == "MATCH"
    assert [number for part in result.plan for section in part.sections for number in section["image_numbers"]] == [1]


def test_excerpt_text_does_not_activate_an_unselected_ambiguous_gallery_photo():
    knowledge = source(["Снимите «Размер из файла», затем включите «Печатать как рисунок»."])
    knowledge.selector = lambda payload: {
        "result": "MATCH", **selected_part(payload, DOC, "g1", STEPS[1]), "delivery_mode": "EXCERPT",
    }
    result = knowledge.search_with_llm("Как называется настройка размера бумаги?", context("Узнать настройку размера"))
    assert not any(section["image_numbers"] for section in result.plan[0].sections)
    assert ALTERNATIVE not in [section["text"] for section in result.plan[0].sections]


def test_excerpt_can_show_the_exact_original_composite_photo_without_expanding_its_owners():
    caption = "Снимите «Размер из файла», затем включите «Печатать как рисунок»."
    knowledge = source([caption])
    knowledge.selector = lambda payload: {
        "result": "MATCH", **selected_part(payload, DOC, "g1", caption), "delivery_mode": "EXCERPT",
    }
    result = knowledge.search_with_llm("Покажите исходную картинку настроек.", context("Найти исходную картинку"))
    assert [number for section in result.plan[0].sections for number in section["image_numbers"]] == [1]
    assert STEPS[1] not in [section["text"] for section in result.plan[0].sections]
    assert ALTERNATIVE not in [section["text"] for section in result.plan[0].sections]


def test_warning_in_an_unselected_photo_caption_does_not_force_that_photo_into_an_excerpt():
    caption = "Отдельная схема выключателя. Не отключайте питание во время настройки."
    body = "<w:p><w:r><w:t>" + escape(STEPS[1]) + "</w:t></w:r></w:p>"
    knowledge, _ = scanner_provider([(DOC, "Новый справочный материал.docx",
                                     instruction([caption], extra_paragraph=body))])
    knowledge.selector = lambda payload: {
        "result": "MATCH", **selected_part(payload, DOC, "g1", STEPS[1]), "delivery_mode": "EXCERPT",
    }
    result = knowledge.search_with_llm("Как называется настройка размера?", context("Узнать настройку размера"))
    assert not any(section["image_numbers"] for section in result.plan[0].sections)
    assert restore_plan(knowledge, serialize_plan(result.plan))[0].sections == result.plan[0].sections


@pytest.mark.parametrize("guard", [
    "Если не считывается код, проверьте устройство.",
    "Если принтер вообще не принимает задания, обратитесь в поддержку.",
    "Если не получается",
    "Если SMS не приходит, проверьте связь и обратитесь в поддержку.",
    "Если соединение не устанавливается, сообщите, какая сеть используется.",
    "Если письма не загружаются, передайте текст ошибки в поддержку.",
    "Если каталог очереди изменён, эти пути не подходят. Не удаляйте другие папки.",
    "В случае отсутствия предметов обратитесь в отдел обслуживания.",
])
def test_bare_conditions_and_safety_guards_remain_in_their_complete_procedure(guard):
    content = "Откройте программу.\n" + guard + "\nПроверьте результат."
    document = {"source": {"file_id": DOC, "knowledge_id": KB, "sha256": "1" * 64},
                "content": content, "images": []}
    sections = instruction_sections(document)
    assert len({section["group_id"] for section in sections}) == 1
    assert guard in [section["text"] for section in sections]
    assert not any("parent_group_id" in section or "scope_ambiguous" in section for section in sections)
