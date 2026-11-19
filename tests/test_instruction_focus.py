"""Checks for the newly added source-bound answer paths."""

from dataclasses import replace
from types import SimpleNamespace

from app.services.chat.instruction_plan import render_plan, restore_plan, serialize_plan
from app.services.rag.instruction_focus import (
    focus_covers_goals, focused_excerpt, focused_question,
)
from app.services.rag.catalog import instruction_sections
from app.services.rag.talk import InstructionPlanPart
from tests.test_verified_support_followup import document


def paragraph(text, *, kind="TEXT", images=()):
    return {"text": text, "kind": kind, "image_numbers": list(images), "group_id": "main"}


def test_one_linked_printer_requires_both_the_physical_diagram_and_application_settings():
    question = "Как настроить печать с ТСД на один принтер Urovo по блютузу?"
    steps = [paragraph("1. В LabelPrint выберите ZPL и TCP/IP."),
             paragraph("2. Включите Новый LabelPrint."),
             paragraph("3. Не переключайте LabelPrint на Bluetooth."),
             paragraph("4. Сделайте тестовую печать.")]
    diagram = paragraph("Схема ТСД → Urovo по Bluetooth. LabelPrint использует TCP/IP.", kind="IMAGE", images=(1,))
    kind = focused_question(question)
    assert kind == "linked_printer"
    assert focused_question(question.replace("один принтер", "два принтера")) is None
    assert focused_question(question + " Можно включить Bluetooth в LabelPrint?") is None
    assert focused_excerpt(kind, question, {"filename": "Печать.docx"}, steps) is None
    result = focused_excerpt(kind, question, {"filename": "Печать.docx"}, [*steps, diagram])
    assert result.kind == "ACTION" and result.sections[-1] == diagram


def test_default_dimensions_require_both_rows_in_the_defaults_table():
    question = "В руководстве TSC есть таблица исходных размеров этикетки?"
    spec = {"filename": "Справочник_TSC TE500.docx"}
    header = paragraph("Параметр             Стандартная настройка")
    width = paragraph("Label width     101,5 мм")
    height = paragraph("Label height     75,0 мм")
    limited = paragraph("Допустимые размеры от 20 до 300 мм")
    reset = paragraph("1. Переведите выключатель питания в положение Выключено.")
    focus = focused_question(question)
    assert focus == "default_dimensions"
    assert focused_excerpt(focus, question, spec, [header, width, reset, height]) is None
    matched = focused_excerpt(focus, question, spec, [limited, header, width, height, reset])
    assert matched.kind == "LOOKUP" and matched.sections == (header, width, height)


def test_lookup_about_preparation_quotes_warning_without_triggering_the_update():
    warning = "Не отключайте питание или USB во время записи."
    approved = document(f"# Подготовка\n\n6. Нажмите Download. {warning}")
    sections = tuple(instruction_sections(approved.document))
    ids = tuple(section["id"] for section in sections)
    regular = InstructionPlanPart(approved, ("g1",), ("Что подготовить",), ids, sections, delivery_mode="EXCERPT")
    knowledge = SimpleNamespace(read=lambda file_id: approved)

    for part in (regular, replace(regular, preparation_checklist=True)):
        saved = serialize_plan((part,))
        restored = restore_plan(knowledge, saved)
        assert restored == (part,)
        assert restored[0].sections == sections
        blocks, _, _, _ = render_plan(restored, saved)
        body = "\n".join(text for text, _ in blocks)
        assert warning in body
        assert ("Нажмите Download" in body) is not part.preparation_checklist


def test_contrast_match_requires_a_real_description_beside_the_command_name():
    question = "Этикетка с переносного принтера бледная. Есть ли регулировка темноты?"
    focus = focused_question(question)
    command = paragraph("[name] set contrast command")
    description = paragraph("Set the print contrast. The higher the contrast, the darker the print.")
    density = paragraph("Density (Насыщенность) 8")
    spec = {"filename": "Руководство поставщика.docx"}
    assert focused_excerpt(focus, question, spec, [command, density]) is None
    assert focused_excerpt(focus, question, spec, [command, description, density]).sections == (command, description)


def test_ribbon_procedure_does_not_cover_unrelated_goals():
    goals = ("Установить красящую ленту", "Проверить печать после её замены")
    assert focus_covers_goals("ribbon_test", goals)
    assert not focus_covers_goals("ribbon_test", (*goals, "Настроить почту"))
    assert not focus_covers_goals("ribbon_test", (goals[0], "Настроить почту"))
