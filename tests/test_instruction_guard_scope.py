"""Original safety context follows the selected manual chapter and its ancestors."""

from app.services.rag.catalog import instruction_scope_guard, instruction_scope_groups, instruction_sections
from tests.test_verified_support_followup import document


def sections():
    original = document("Внимание: работу выполняет ответственный специалист.\n\n"
                        "5.1 Общая подготовка\n\nВнимание: сохраните параметры первого раздела.\n\n"
                        "Сведения первого раздела.\n\n5.2 Подключение устройства\n\n"
                        "Внимание: сохраните настройки подключения.\n\n"
                        "5.2.1 Настройка интерфейса\n\nТребуется рабочий адрес подключения.\n\n"
                        "Сведения интерфейса.\n\n5.3 Размер этикетки\n\nШирина по умолчанию 101,5 мм.")
    return instruction_sections(original.document)


def test_manual_lookup_excludes_requirements_from_other_numbered_chapters():
    original = sections()
    selected = next(section for section in original if "Ширина по умолчанию" in section["text"])
    guards = [section["text"] for section in original if instruction_scope_guard(section, {selected["group_id"]})]
    assert guards == ["Внимание: работу выполняет ответственный специалист."]


def test_a_chapter_inherits_its_parent_guard_without_sibling_requirements():
    original = sections()
    selected = next(section for section in original if section["text"] == "Сведения интерфейса.")
    guards = [section["text"] for section in original if instruction_scope_guard(section, {selected["group_id"]})]
    assert "Внимание: работу выполняет ответственный специалист." in guards
    assert "Внимание: сохраните настройки подключения." in guards
    assert "Требуется рабочий адрес подключения." in guards
    assert "Внимание: сохраните параметры первого раздела." not in guards


def test_source_declared_exclusive_choice_guards_keep_their_original_boundary():
    original = document("5.2 Настройка устройства\n\nВыберите один способ.\n\n"
                        "### Способ 1\n\nВнимание: сохраните первый список.\n\n1. Откройте первый список.\n\n"
                        "### Способ 2\n\nВнимание: сохраните второй список.\n\n1. Откройте второй список.")
    parsed = instruction_sections(original.document)
    selected = next(section for section in parsed if "Откройте второй" in section["text"])
    groups = instruction_scope_groups(parsed, {selected["group_id"]})
    guards = [section["text"] for section in parsed if instruction_scope_guard(section, groups)]
    assert "Внимание: сохраните второй список." in guards
    assert "Внимание: сохраните первый список." not in guards


def test_a_wrapped_negative_requirement_cannot_become_an_affirmative_guard():
    original = document("Страница PDF 1\n\n5.2 Подключение\n\nUSB, в поле интерфейса не\n\nтребуется изменять\n\nнастройку.")
    parsed = instruction_sections(original.document)
    assert [section["text"] for section in parsed] == [
        "Страница PDF 1", "5.2 Подключение", "USB, в поле интерфейса не", "требуется изменять", "настройку.",
    ]
    assert not any(instruction_scope_guard(section, {parsed[-1]["group_id"]}) for section in parsed)


def test_a_wrapped_real_warning_keeps_all_original_fragments_mandatory():
    original = document("Страница PDF 1\n\n5.2 Обновление\n\nНе\n\nотключайте питание\n\nпринтера.")
    parsed = instruction_sections(original.document)
    guards = [section["text"] for section in parsed if instruction_scope_guard(section, {parsed[-1]["group_id"]})]
    assert guards == ["Не", "отключайте питание", "принтера."]


def test_a_wrapped_prerequisite_is_kept_complete_and_independent_uppercase_lines_remain_separate():
    original = document("Страница PDF 1\n\n5.2 Работа\n\nНачинаем настройку\n\nТребуется\n\nрабочий адрес\n\nподключения.")
    parsed = instruction_sections(original.document)
    guards = [section["text"] for section in parsed if instruction_scope_guard(section, {parsed[-1]["group_id"]})]
    assert guards == ["Требуется", "рабочий адрес", "подключения."]
    assert next(section for section in parsed if section["text"] == "Начинаем настройку")["kind"] == "TEXT"


def test_code_fences_and_original_illustration_boundaries_are_not_soft_wrapped():
    original = document("Страница PDF 1\n\n5.2 Справка\n\n```text\n\nне\n\nтребуется изменять\n\nнастройку.\n\n```\n\n"
                        "USB, в поле интерфейса не\n\nИллюстрация подключения.")
    original.document["images"] = [{"number": 1, "caption": "Иллюстрация подключения."}]
    parsed = instruction_sections(original.document)
    assert next(section for section in parsed if section["text"] == "Иллюстрация подключения.")["kind"] == "IMAGE"
    assert next(section for section in parsed if section["text"] == "требуется изменять")["kind"] == "PREREQUISITE"
    assert [section["text"] for section in parsed].count("```") == 1
