"""Instruction applicability is visible before a user consents to its actions."""

import copy
from types import SimpleNamespace

from app.services.chat.confirmation_scope import confirmation_scope_entries, render_confirmation_scopes
from app.services.chat.talk_dialogue import _confirmation_turn, _resume_turn, _understanding_turn
from app.services.rag.catalog import instruction_sections


def original(text, *, filename="Инструкция.docx", identity="a"):
    source = {"file_id": identity * 36, "filename": filename,
              "knowledge_id": "collection", "sha256": identity * 64}
    return SimpleNamespace(document={"source": source, "content": text, "images": []},
                           candidate=SimpleNamespace(title=filename))


def flow(approved):
    return {"query": "Открыть рабочую почту", "opening_query": "Как открыть почту?",
            "source": approved.document["source"], "shown": [], "excluded": []}


def test_browser_scope_uses_original_body_heading_when_filename_is_technical():
    approved = original("# Вход в почту webmail\n\n## Вход в корпоративную почту через браузер\n\n"
                        "1. Откройте сайт.\n2. Введите свой пароль.", filename="08_Вход в почту webmail.docx")
    entries = confirmation_scope_entries(approved)
    visible = render_confirmation_scopes(entries)
    assert "Вход в корпоративную почту через браузер" in visible
    assert "Введите" not in visible and "Откройте" not in visible
    assert entries[0]["title"] == "Вход в почту webmail"
    sections = {section["id"]: section for section in instruction_sections(approved.document)}
    assert entries[0]["context"][0]["text"] == sections[entries[0]["context"][0]["section_id"]]["text"].lstrip("# ")


def test_installed_application_scope_and_actual_two_device_scenario_are_visible():
    outlook = original("# Первый вход в Outlook\n\n## Первый вход в Outlook classic\n\n1. Откройте Outlook.")
    printer = original("# Печать с ТСД\n\n## Сценарий\n\n"
                       "Один ТСД и два принтера по Wi-Fi. Используется рабочая программа склада.\n\n"
                       "1. В LabelPrint выберите ZPL и TCP/IP.", identity="b")
    parts = [SimpleNamespace(approved=approved, sections=tuple(instruction_sections(approved.document)))
             for approved in [outlook, printer]]
    visible = render_confirmation_scopes(confirmation_scope_entries(outlook, parts))
    assert "Outlook classic" in visible and "два принтера по Wi-Fi" in visible
    assert "ZPL" not in visible and "TCP/IP" not in visible


def test_distinct_procedure_scopes_from_the_same_original_are_retained():
    approved = original("# Подключение принтеров\n\n## Два принтера по Wi-Fi\n\n1. Откройте приложение.\n\n"
                        "## Один принтер по Bluetooth\n\n1. Выберите устройство.")
    sections = instruction_sections(approved.document)
    parts = [SimpleNamespace(approved=approved, sections=tuple(section for section in sections
                                                              if phrase in section["title"]))
             for phrase in ["Два принтера", "Один принтер"]]
    entries = confirmation_scope_entries(approved, parts)
    assert len(entries) == 2
    assert [entry["context"][0]["text"] for entry in entries] == [
        "Два принтера по Wi-Fi", "Один принтер по Bluetooth",
    ]
    assert confirmation_scope_entries(approved, [parts[0], parts[0]]) == [entries[0]]


def test_confirmation_exposes_the_scope_before_the_only_consent_question():
    approved = original("# Настройка сканера в 1С\n\n1. Откройте настройки.")
    turn = _confirmation_turn(approved, flow(approved), "Подключить сканер к рабочей программе")
    body = turn.blocks[0][0]
    assert body.index("Настройка сканера в 1С") < body.index("Правильно я понял?")
    assert body.count("?") == 1 and "Откройте настройки" not in body
    assert turn.flow["understanding_confirmation"]["scope_entries"][0]["source"] == approved.document["source"]


def test_saved_confirmation_without_scope_requires_fresh_confirmation_once():
    approved = original("# Вход в корпоративную почту через браузер\n\n1. Откройте сайт.")
    pending = _confirmation_turn(approved, flow(approved), "Открыть почту").flow
    pending["understanding_confirmation"].pop("scope_entries")
    pending["understanding_confirmation"].pop("scope_description")
    pending["task_state"] = {"goals": {"g1": {"text": "Открыть почту"}}}
    knowledge = SimpleNamespace(read=lambda file_id: approved)
    refreshed = _understanding_turn("Да", pending, {}, SimpleNamespace(), knowledge, SimpleNamespace())
    assert refreshed.action == "REPLY" and "через браузер" in refreshed.blocks[0][0]
    assert refreshed.flow["understanding_confirmation"]["scope_entries"]
    assert not refreshed.flow["understanding_confirmation"]["guide_flow"].get("understanding_confirmed")
    assert refreshed.flow["task_state"] == pending["task_state"]


def test_resume_from_form_keeps_the_visible_applicability():
    approved = original("# Вход в корпоративную почту через браузер\n\n1. Откройте сайт.")
    pending = _confirmation_turn(approved, flow(approved), "Открыть почту").flow
    form = {"escalation": {"resume": {"flow": copy.deepcopy(pending)}}}
    resumed = _resume_turn(form, {}, SimpleNamespace(), SimpleNamespace())
    assert resumed.action == "RESUME" and "через браузер" in resumed.blocks[0][0]


def test_unstyled_browser_declaration_is_visible_before_actions():
    approved = original("Вход в корпоративную почту через браузер\n\nОткройте сайт почты.\n\nВведите свой пароль.")
    entry, = confirmation_scope_entries(approved)
    assert entry["title"] == "Вход в корпоративную почту через браузер"
    assert entry["title_section_id"]
    assert "Введите" not in render_confirmation_scopes([entry])


def test_unstyled_two_printer_applicability_is_retained():
    entry, = confirmation_scope_entries(original("Один ТСД и два стационарных принтера по Wi-Fi.\n\n"
                                               "1. Откройте рабочее приложение."))
    assert "два стационарных принтера" in entry["title"]


def test_warning_or_parameter_values_do_not_become_a_scope_title():
    entry, = confirmation_scope_entries(original("Внимание, не удаляйте существующую запись.\n\n"
                                               "ZPL и TCP/IP\n\nВход в установленный почтовый клиент.\n\n"
                                               "1. Нажмите кнопку."))
    assert entry["title"] == "Вход в установленный почтовый клиент."


def test_scope_does_not_come_from_procedure_or_later_caption():
    entry, = confirmation_scope_entries(original("Сначала откройте настройки приложения.\n\n"
                                               "Вход в корпоративную почту через браузер"))
    assert entry["title"] == "Инструкция.docx" and entry["title_section_id"] is None


def test_spreadsheet_metadata_does_not_replace_the_applicable_source_name():
    approved = original("Лист: Телефоны\nA1: Добавочные телефоны Часцы\n"
                        "A3: Место или сотрудник | B3: Добавочный\nA4: Руководство СК | B4: 126",
                        filename="Добавочные телефоны Часцы.xlsx")
    entry, = confirmation_scope_entries(approved)
    assert entry["title"] == "Добавочные телефоны Часцы.xlsx" and entry["title_section_id"] is None
    assert "Часцы" in render_confirmation_scopes([entry])
    assert "Лист:" not in render_confirmation_scopes([entry])


def test_excerpt_scope_comes_from_the_selected_fact_instead_of_an_unrelated_mandatory_section():
    approved = original("# Корпус устройства\n\n## Вид изнутри\n\n"
                        "Не отключайте питание до остановки механизма.\n\n"
                        "## Вид сзади\n\nКруглый разъём питания находится на задней панели.")
    sections = instruction_sections(approved.document)
    selected = tuple(section for section in sections if section["kind"] == "WARNING"
                     or "Круглый разъём" in section["text"])
    part = SimpleNamespace(approved=approved, sections=selected, delivery_mode="EXCERPT")
    entry, = confirmation_scope_entries(approved, [part])
    visible = render_confirmation_scopes([entry])
    assert "Вид сзади" in visible and "Вид изнутри" not in visible
    assert "Круглый разъём" not in visible and "отключайте" not in visible


def test_excerpt_photo_caption_with_a_warning_keeps_its_own_scope_over_a_bare_warning_label():
    caption = "Задняя панель и разъём питания. Штекер нельзя вставлять в разъём данных."
    approved = original("# Корпус устройства\n\n## Вид изнутри\n\nОСТОРОЖНО!\n\n"
                        "## Вид сзади\n\n" + caption)
    approved.document["images"] = [{"number": 1, "caption": caption}]
    selected = tuple(section for section in instruction_sections(approved.document)
                     if section["kind"] == "WARNING")
    assert any(section["image_numbers"] for section in selected)
    part = SimpleNamespace(approved=approved, sections=selected, delivery_mode="EXCERPT")
    visible = render_confirmation_scopes(confirmation_scope_entries(approved, [part]))
    assert "Вид сзади" in visible and "Вид изнутри" not in visible


def test_conditional_child_discloses_its_condition_without_action_breadcrumbs():
    approved = original("# Печать PDF\n\n## Проблема\n\nПрограмма выбирает неверный размер бумаги.\n\n"
                        "1. Проверьте выбранный лист.\n\nЕсли картинки всё равно не печатаются, "
                        "отдельный способ: «Дополнительно → Печатать как изображение → ОК».")
    sections = instruction_sections(approved.document)
    conditional = next(section for section in sections if section.get("condition_text"))
    part = SimpleNamespace(approved=approved, sections=tuple(sections))
    entries = confirmation_scope_entries(approved, [part])
    assert entries[0]["context"] == [{"section_id": conditional["condition_section_id"],
                                      "text": "Если картинки всё равно не печатаются"}]
    visible = render_confirmation_scopes(entries)
    assert "Если картинки всё равно не печатаются" in visible
    assert "Печатать как изображение" not in visible and "неверный размер бумаги" not in visible


def test_unstyled_conditional_method_cannot_become_an_action_bearing_title():
    approved = original("Если картинки всё равно не печатаются, отдельный способ: "
                        "«Дополнительно → Печатать как изображение → ОК».")
    part = SimpleNamespace(approved=approved, sections=tuple(instruction_sections(approved.document)))
    entries = confirmation_scope_entries(approved, [part])
    assert entries[0]["title"] == "Инструкция.docx" and entries[0]["title_section_id"] is None
    visible = render_confirmation_scopes(entries)
    assert "Если картинки всё равно не печатаются" in visible and "Печатать как изображение" not in visible


def test_formatted_conditional_and_step_headings_do_not_expose_actions_before_consent():
    conditional = "Если картинки всё равно не печатаются, отдельный способ: «Дополнительно → Печатать как изображение» ."
    approved = original(conditional)
    approved.document["headings"] = [{"text": conditional, "occurrence": 1, "level": 2}]
    part = SimpleNamespace(approved=approved, sections=tuple(instruction_sections(approved.document)))
    entries = confirmation_scope_entries(approved, [part])
    assert entries[0]["title"] == "Инструкция.docx"
    assert "Если картинки всё равно не печатаются" in render_confirmation_scopes(entries)
    assert "Печатать как изображение" not in render_confirmation_scopes(entries)
    step = original("# Шаг 1. Откройте настройки приложения")
    assert confirmation_scope_entries(step)[0]["title"] == "Инструкция.docx"
