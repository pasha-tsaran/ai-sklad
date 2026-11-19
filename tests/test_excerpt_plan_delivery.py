"""Reference answers keep their values together through persistence and reuse."""

import copy
import hashlib
import json

import pytest

from app.services.chat.instruction_plan import render_plan, restore_plan, serialize_plan
from app.services.rag.catalog import instruction_sections
from app.services.rag.talk import InstructionPlanPart, KnowledgeUnavailable
from tests.test_instruction_planning import context, selected_part
from tests.test_openwebui_instruction_images import instruction
from tests.test_scanner_original_retrieval import scanner_provider
from tests.test_talk_dialogue import assistants
from tests.test_talk_dialogue import talk as talk
from tests.test_talk_knowledge import DOC
from tests.test_talk_model_dialogue import model_chat as model_chat


ROWS = ["Лист: Контакты", "Добавочные номера", "Должность | Добавочный", "Руководитель | 126"]
REQUEST = "Узнать добавочный руководителя"


def source_plan(mode="EXCERPT"):
    knowledge, _ = scanner_provider([(DOC, "Контакты.md", "\n".join(ROWS).encode())])
    approved = knowledge.read(DOC)
    sections = tuple(instruction_sections(approved.document))
    part = InstructionPlanPart(approved, ("g1",), (REQUEST,),
                               tuple(section["id"] for section in sections), sections, delivery_mode=mode)
    return knowledge, part


def digest(body):
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def test_four_short_reference_rows_deliver_the_value_in_one_block_without_continuation():
    _, part = source_plan()
    blocks, _, saved, more = render_plan([part], serialize_plan([part]))
    assert len(blocks) == 2 and blocks[0] == ("\n\n".join(ROWS), None)
    assert "126" in blocks[0][0] and not more and saved["cursor"] == 1
    assert not any("Продолжить" in text for text, _ in blocks)
    assert blocks[-1][0] == "Эта информация помогла?"


def test_excerpt_mode_round_trip_is_included_in_plan_identity():
    knowledge, part = source_plan()
    saved = serialize_plan([part])
    assert saved["parts"][0]["delivery_mode"] == "EXCERPT"
    restored = restore_plan(knowledge, saved)
    assert restored[0].delivery_mode == "EXCERPT" and serialize_plan(restored) == saved
    changed = copy.deepcopy(saved)
    changed["parts"][0].pop("delivery_mode")
    with pytest.raises(KnowledgeUnavailable, match="InvalidSavedInstructionPlan"):
        restore_plan(knowledge, changed)


def test_legacy_procedure_plan_format_hash_and_paging_remain_compatible():
    knowledge, part = source_plan("PROCEDURE")
    saved = serialize_plan([part], version=1)
    row = {"source": part.approved.document["source"], "goal_ids": list(part.goal_ids), "goals": list(part.goals),
           "section_ids": list(part.section_ids), "sections_sha256": digest(list(part.sections))}
    assert saved["parts"] == [row] and "delivery_mode" not in row
    assert saved["digest"] == digest({"version": 1, "parts": [row]})
    restored = restore_plan(knowledge, saved)
    assert restored[0].delivery_mode == "PROCEDURE"
    blocks, _, _, more = render_plan(restored, saved)
    assert more and "126" not in "\n".join(text for text, _ in blocks)


@pytest.mark.parametrize("mode", [None, "OTHER", [], {}])
def test_invalid_persisted_mode_is_rejected_even_with_a_recomputed_outer_digest(mode):
    knowledge, part = source_plan()
    saved = serialize_plan([part])
    saved["parts"][0]["delivery_mode"] = mode
    saved["digest"] = digest({"version": saved["version"], "parts": saved["parts"]})
    with pytest.raises(KnowledgeUnavailable, match="InvalidSavedInstructionPlan"):
        restore_plan(knowledge, saved)


def test_identical_action_and_lookup_scopes_remain_separate_plan_parts():
    text = "Проверьте значение параметра: 70 мм."
    knowledge, _ = scanner_provider([(DOC, "Параметр.md", text.encode())])
    knowledge.selector = lambda payload: {"result": "PLAN", "parts": [
        {**selected_part(payload, DOC, "g1", text), "delivery_mode": "EXCERPT"},
        {**selected_part(payload, DOC, "g2", text), "delivery_mode": "PROCEDURE"},
    ]}
    result = knowledge.search_with_llm("Какое значение и как его проверить?", {
        **context("Узнать значение", "Проверить значение"),
        "requested_goal_kinds": {"g1": "LOOKUP", "g2": "ACTION"},
    })
    assert len(result.plan) == 2 and result.plan[0].sections == result.plan[1].sections
    assert [part.delivery_mode for part in result.plan] == ["EXCERPT", "PROCEDURE"]
    rows = serialize_plan(result.plan)["parts"]
    assert rows[0]["delivery_mode"] == "EXCERPT" and "delivery_mode" not in rows[1]


def test_fresh_procedure_plans_coalesce_short_text_and_keep_legacy_midstream_cursors():
    knowledge, part = source_plan("PROCEDURE")
    fresh = serialize_plan([part])
    assert fresh["version"] == 3
    blocks, _, _, more = render_plan([part], fresh)
    assert not more and blocks[0][0] == "\n\n".join(ROWS)
    legacy = serialize_plan([part], version=1)
    legacy.update(cursor=3, shown=[0, 1, 2])
    restored = restore_plan(knowledge, legacy)
    next_blocks, _, updated, more = render_plan(restored, legacy)
    assert next_blocks[0][0] == ROWS[3] and not more and updated["cursor"] == 4
    repeated, _, _, _ = render_plan(restored, legacy, repeat=True)
    assert [text for text, _ in repeated[:-1]] == ROWS[:3]


@pytest.mark.parametrize("marker", ["```", "~~~~"])
def test_new_procedure_delivers_a_complete_code_fence_in_one_message(marker):
    lines = ["Откройте окно служебного сценария.", marker + "text",
             *[f"echo Исходная строка команды {number}" for number in range(8)], marker,
             "Проверьте результат выполнения."]
    knowledge, _ = scanner_provider([(DOC, "Новый сценарий.md", "\n".join(lines).encode())])
    approved = knowledge.read(DOC)
    sections = tuple(instruction_sections(approved.document))
    part = InstructionPlanPart(approved, ("g1",), ("Выполнить сценарий",),
                               tuple(section["id"] for section in sections), sections)
    original = copy.deepcopy(sections)
    blocks, _, _, more = render_plan([part], serialize_plan([part]))
    assert not more and len(blocks) == 2
    assert all(line in blocks[0][0] for line in lines)
    assert blocks[0][0].count(marker) == 2
    assert part.sections == original


def test_new_text_coalescing_keeps_distinct_plan_parts_separate():
    _, lookup = source_plan("EXCERPT")
    _, action = source_plan("PROCEDURE")
    blocks, _, saved, more = render_plan([lookup, action], serialize_plan([lookup, action]))
    assert not more and saved["cursor"] == 2
    assert [text for text, _ in blocks[:-1]] == ["\n\n".join(ROWS), "\n\n".join(ROWS)]


def test_oversize_multiline_code_has_balanced_message_chunks_with_all_original_commands():
    commands = ["echo " + str(number).zfill(4) + " " + "x" * 100 for number in range(700)]
    content = "Перед выполнением сохраните работу.\n```cmd\n" + "\n".join(commands) + "\n```\nПроверьте результат."
    knowledge, _ = scanner_provider([(DOC, "Большой сценарий.md", content.encode())])
    approved = knowledge.read(DOC)
    sections = tuple(instruction_sections(approved.document))
    part = InstructionPlanPart(approved, ("g1",), ("Выполнить сценарий",),
                               tuple(section["id"] for section in sections), sections)
    saved = serialize_plan([part])
    seen, sources = [], []
    more = True
    while more:
        blocks, provenance, saved, more = render_plan([part], saved)
        seen.extend(text for text, _ in blocks[:-1])
        sources.extend(provenance[:-1])
    fenced = [text for text in seen if "```" in text]
    assert len(fenced) > 1
    assert all(text.startswith("```cmd\n") and text.rstrip().endswith("```") for text in fenced)
    assert all(len(text) <= 24_000 for text in fenced)
    assert [line for text in seen for line in text.splitlines() if line.startswith("echo ")] == commands
    assert all(source == approved.document["source"] for source in sources)


def test_a_long_single_command_below_the_transport_limit_stays_in_one_original_fence():
    command = "echo " + "x" * 30_000
    content = "```cmd\n" + command + "\n```"
    knowledge, _ = scanner_provider([(DOC, "Длинная команда.md", content.encode())])
    approved = knowledge.read(DOC)
    sections = tuple(instruction_sections(approved.document))
    part = InstructionPlanPart(approved, ("g1",), ("Выполнить сценарий",),
                               tuple(section["id"] for section in sections), sections)
    blocks, _, _, more = render_plan([part], serialize_plan([part]))
    assert not more and len(blocks) == 2
    assert command in blocks[0][0] and len(blocks[0][0]) <= 32_000
    assert blocks[0][0].count("```cmd") == 1 and blocks[0][0].count("```") == 2


def test_excerpt_photographs_keep_the_existing_six_image_limit_and_order():
    captions = [f"Исходная иллюстрация справки номер {number}." for number in range(1, 8)]
    knowledge, _ = scanner_provider([(DOC, "Справочные иллюстрации.docx", instruction(captions))])
    approved = knowledge.read(DOC)
    sections = tuple(instruction_sections(approved.document))
    part = InstructionPlanPart(approved, ("g1",), ("Найти иллюстрации",),
                               tuple(section["id"] for section in sections), sections, delivery_mode="EXCERPT")
    first, _, saved, more = render_plan([part], serialize_plan([part]))
    assert more and [photo for _, photo in first if photo] == [approved.illustrations[number] for number in range(1, 7)]
    following, _, _, more = render_plan([part], saved)
    assert not more and [photo for _, photo in following if photo] == [approved.illustrations[7]]


def test_phone_value_arrives_after_confirmation_and_cached_reuse_without_header_only_paging(model_chat):
    def choose(payload):
        part = selected_part(payload, DOC, "g1", "Руководитель | 126")
        part["section_ids"] = list(payload["candidate_sections"][DOC])
        return {"result": "MATCH", **part, "delivery_mode": "EXCERPT"}

    chat = model_chat([choose], rows=[(DOC, "Контакты.md", "\n".join(ROWS).encode())])
    chat.talk.settings.talk_confirm_understanding = True
    chat.model.supports_instruction_planning = True
    chat.model.supports_source_clarification = True
    chat.model.understand_task = lambda *args: {
        "goals": [{"text": REQUEST, "evidence": REQUEST, "kind": "LOOKUP", "search_queries": []}],
    }
    pending = chat.send(REQUEST)
    plan = pending["flow"]["understanding_confirmation"]["guide_flow"]["instruction_plan"]
    assert plan["parts"][0]["delivery_mode"] == "EXCERPT"
    sent = chat.send("Да, правильно", 2)
    assert "126" in "\n".join(text for text, _ in assistants(sent))
    assert sent["flow"]["has_more"] is False
    assert not any("Продолжить" in text for text, _ in assistants(sent))
    resolved = chat.send("Помогло, спасибо", 3)
    assert resolved["incident"].state == "RESOLVED"
    repeated = chat.send(REQUEST, 4)
    cached = repeated["flow"]["understanding_confirmation"]["guide_flow"]["instruction_plan"]
    assert cached["parts"][0]["delivery_mode"] == "EXCERPT" and len(chat.model.calls) == 1
    final = chat.send("Да", 5)
    assert final["flow"]["has_more"] is False
    assert "126" in "\n".join(text for text, _ in assistants(final))
