"""Confirmation discloses applicability before delivering selected source content."""

import copy
from types import SimpleNamespace

import pytest

from app.services.chat.confirmation_scope import confirmation_scope_entries, confirmation_scope_identity
from app.services.chat.instruction_plan import serialize_plan
from app.services.chat.talk_dialogue import _confirmation_turn, _resume_turn, _understanding_turn
from app.services.rag.catalog import instruction_sections
from app.services.rag.talk import InstructionPlanPart
from scripts.evaluate_instruction_dialogues import check_visible_scopes
from tests.conftest import env as env
from tests.test_openwebui_instruction_images import instruction, png
from tests.test_talk_dialogue import Knowledge, assistants
from tests.test_talk_dialogue import talk as talk
from tests.test_talk_knowledge import DOC
from tests.test_talk_model_dialogue import model_chat as model_chat
from tests.test_talk_understanding import enabled_chat, match
from tests.test_verified_support_followup import document


COMMAND = 'del /F /Q "%SystemRoot%\\System32\\spool\\PRINTERS\\*.shd"'
GOAL = "Понять назначение присланного файла"


def planned(content, *, goal=GOAL, filename="Описание рабочего файла.md", kind="LOOKUP"):
    approved = document(content)
    approved.document["source"]["filename"] = filename
    sections = tuple(instruction_sections(approved.document))
    evidence = tuple({"section_id": section["id"], "text": section["text"], "support_kind": "TEXT",
                      "image_numbers": []} for section in sections if not section["is_heading"])
    part = InstructionPlanPart(approved, ("g1",), (goal,), tuple(section["id"] for section in sections), sections,
                               delivery_mode="EXCERPT" if kind == "LOOKUP" else "PROCEDURE",
                               confirmation_evidence=evidence)
    flow = {"source": copy.deepcopy(approved.document["source"]), "query": goal, "task_query": goal,
            "opening_query": goal, "selection": [], "shown": [], "has_more": False, "next_offset": 0,
            "instruction_plan": serialize_plan([part]), "excluded": [],
            "task_state": {"facts": [], "goals": {"g1": {"text": goal, "evidence": goal, "kind": kind}},
                           "messages": [goal], "question": "", "question_field": None, "unknown_answers": [],
                           "new_task": False}}
    return approved, (part,), flow


@pytest.mark.parametrize("content,forbidden", [
    ("# Рабочий инструмент\n\nПоложение регулятора определяет яркость подсветки.", "Положение регулятора"),
    ("# Управление заданиями\n\n" + COMMAND, COMMAND),
    ("# Справка\n\nШирина составляет 70 мм, высота составляет 40 мм.", "70 мм"),
    (COMMAND, COMMAND),
])
def test_source_body_quotes_and_declarative_titles_are_not_the_confirmation_answer(content, forbidden):
    approved, parts, flow = planned(content)
    untouched = copy.deepcopy(flow)
    turn = _confirmation_turn(approved, flow, GOAL, parts=parts)
    text = turn.blocks[0][0]
    assert GOAL in text and forbidden not in text
    if "del /F /Q" in forbidden:
        assert "del /F /Q" not in text
    assert text.count("?") == 1 and all(image is None for _, image in turn.blocks)
    confirmation = turn.flow["understanding_confirmation"]
    assert confirmation["guide_flow"] == untouched and flow == untouched
    assert confirmation["scope_entries"] == confirmation_scope_entries(approved, parts)
    assert confirmation["scope_identity"] == confirmation_scope_identity(confirmation["scope_entries"])
    assert check_visible_scopes(confirmation, [text], lambda file_id: approved) == 1


def test_summary_can_retain_a_command_already_quoted_by_the_employee():
    summary = "Понять назначение команды `sc sdset spooler`, которую мне прислали"
    approved, parts, flow = planned("# Назначение\n\n" + COMMAND, goal=summary)
    turn = _confirmation_turn(approved, flow, summary, parts=parts)
    assert "sc sdset spooler" in turn.blocks[0][0] and COMMAND not in turn.blocks[0][0]
    assert turn.flow["understanding_confirmation"]["summary"] == summary


def test_multi_goal_confirmation_lists_each_source_once_and_keeps_all_scope_proofs():
    first, first_parts, flow = planned("# Раздел A\n\nПервый источник сообщает исходное значение.",
                                       goal="Узнать первое значение", filename="Первый источник.md")
    second, second_parts, _ = planned("# Раздел B\n\nВторой источник сообщает второе значение.",
                                      goal="Узнать второе значение", filename="Второй источник.md")
    second.document["source"]["file_id"] = "second-source"
    parts = (*first_parts, *second_parts)
    summary = "Узнать первое значение, затем узнать второе значение"
    turn = _confirmation_turn(first, flow, summary, parts=parts)
    text = turn.blocks[0][0]
    assert text.count("Раздел A") == text.count("Раздел B") == 1
    assert "Первый источник сообщает" not in text and "Второй источник сообщает" not in text
    assert turn.flow["understanding_confirmation"]["scope_entries"] == confirmation_scope_entries(first, parts)
    repeated = _confirmation_turn(first, flow, summary, parts=(*first_parts, *first_parts))
    assert repeated.blocks[0][0].count("Раздел A") == 1


@pytest.mark.parametrize("legacy_without_entries", [False, True])
def test_resuming_saved_confirmation_cannot_redisplay_stored_source_quotes(legacy_without_entries):
    approved, parts, flow = planned("# Проверка\n\n" + COMMAND)
    pending = _confirmation_turn(approved, flow, GOAL, parts=parts).flow
    confirmation = pending["understanding_confirmation"]
    assert COMMAND not in confirmation["scope_description"]
    if legacy_without_entries:
        confirmation.pop("scope_entries")
    saved = copy.deepcopy(pending)
    resumed = _resume_turn({"escalation": {"resume": {"flow": pending}}}, {}, None, None)
    assert resumed.flow == saved
    assert COMMAND not in resumed.blocks[0][0] and "Проверка" in resumed.blocks[0][0]
    assert "del /F /Q" not in resumed.blocks[0][0]


def test_lookup_content_is_delivered_once_after_consent_without_changing_its_plan():
    approved, parts, flow = planned("# Назначение\n\nФайл меняет права службы, а не очищает очередь.")
    pending = _confirmation_turn(approved, flow, GOAL, parts=parts)
    assert "Файл меняет права" not in pending.blocks[0][0]
    accepted = _understanding_turn("Да", pending.flow, {}, SimpleNamespace(), Knowledge([approved]), SimpleNamespace())
    assert "Файл меняет права службы, а не очищает очередь." in "\n".join(text for text, _ in accepted.blocks)
    assert "Файл меняет права службы, а не очищает очередь." not in pending.blocks[0][0]
    assert accepted.flow["instruction_plan"]["parts"] == flow["instruction_plan"]["parts"]


def test_worker_sends_original_source_text_and_photo_once_only_after_explicit_consent(model_chat):
    caption = "Для проверки рабочего инструмента откройте подтверждённое окно."
    data = instruction([caption], image_data=png("blue"), extra_paragraph="<w:p><w:r><w:t>" + COMMAND + "</w:t></w:r></w:p>")
    chat = enabled_chat(model_chat, [match(evidence=caption, summary="Проверить рабочий инструмент")], rows=[
        (DOC, "Проверка рабочего инструмента.docx", data),
    ])
    pending = chat.send("Проверить рабочий инструмент")
    before = assistants(pending)
    assert not pending["attempts"] and not any(photo for _, photo in before)
    assert COMMAND not in "\n".join(text for text, _ in before) and caption not in "\n".join(text for text, _ in before)
    assert "del /F /Q" not in "\n".join(text for text, _ in before)
    accepted = chat.send("Да", 2)
    delivered = assistants(accepted)[len(before):]
    assert len([photo for _, photo in delivered if photo]) == 1
    assert "\n".join(text for text, _ in delivered).count(caption) == 1
    assert len(accepted["attempts"]) == 1 and len(chat.model.calls) == 1
