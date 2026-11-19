"""Logical manual chapters preserve complete steps across source page markers."""

import hashlib
import json

import pytest

from app.services.rag.catalog import instruction_sections
from tests.test_knowledge_catalog import document


@pytest.mark.parametrize("word_page_headings", [False, True])
def test_numbered_chapter_retains_all_steps_across_pdf_page_navigation(word_page_headings):
    page1, page2 = "Страница PDF 27", "Страница PDF 28"
    content = "\n".join([
        "# Руководство рабочего устройства", page1,
        "3.3.2 Установка внешнего держателя", "Перед настройкой сохраните параметры.",
        "1. Закрепите внешний держатель.", "2. Поместите большой рулон на держатель.",
        "3. Протяните материал через задний вход.", page2,
        "4. Откалибруйте датчик после замены материала.",
        "Внимание: не отключайте устройство во время проверки.",
        "3.3.3 Установка режущего модуля", "1. Установите режущий модуль.",
    ])
    source = document(content)
    if word_page_headings:
        source["headings"] = [{"text": page, "occurrence": 1, "level": 2} for page in (page1, page2)]
    sections = instruction_sections(source)
    chapter = next(section for section in sections if section["text"].startswith("3.3.2"))
    selected = [section for section in sections if section["group_id"] == chapter["group_id"]]
    assert "4. Откалибруйте датчик после замены материала." in [section["text"] for section in selected]
    assert "1. Установите режущий модуль." not in [section["text"] for section in selected]
    assert next(section for section in sections if section["text"] == page2)["group_id"] == chapter["group_id"]
    assert next(section for section in selected if section["text"].startswith("4."))["title"] == chapter["text"]
    assert {"PREREQUISITE", "WARNING"} <= {section["kind"] for section in selected}
    membership = hashlib.sha256(json.dumps([section["id"] for section in selected],
                                         ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert all(section["group_membership_digest"] == membership for section in selected)


def test_decimal_measurements_and_single_number_steps_do_not_start_chapters():
    source = document("# Проверка устройства\n1. Откройте крышку.\n101.5 мм\n3.3 вольта\n"
                      "## 2. Проверка подключения\nНажмите кнопку проверки.\n3.2 Подключение модуля\nИная процедура.")
    sections = instruction_sections(source)
    group = sections[0]["group_id"]
    assert all(section["group_id"] == group for section in sections[:6])
    assert all(not section["is_heading"] for section in sections[1:4])
    assert sections[-2]["is_heading"] and sections[-2]["group_id"] != group


def gallery_document(paragraphs, captions):
    source = document("\n\n".join(paragraphs), images=[{"number": number, "caption": caption}
                                                     for number, caption in enumerate(captions, 1)])
    positions = {max(index for index, text in enumerate(paragraphs, 1) if text == caption): number
                 for number, caption in enumerate(captions, 1)}
    source["layout"] = [{"text": text, "paragraph": ordinal,
                         "image_numbers": [positions[ordinal]] if ordinal in positions else []}
                        for ordinal, text in enumerate(paragraphs, 1)]
    return source


def test_shared_manual_gallery_keeps_each_numbered_procedures_own_photos():
    captions = [
        "Большой внешний рулон расположен на держателе у заднего входа.",
        "Внешний держатель подаёт материал внутрь устройства.",
        "Режущий модуль установлен в переднем отверстии корпуса.",
        "Отделитель снимает подложку с готовой наклейки.",
    ]
    paragraphs = [
        "# Руководство устройства", "3.3.2 Установка внешнего держателя",
        "Закрепите внешний держатель и поместите большой рулон у заднего входа.",
        "Страница PDF 28", "Откалибруйте датчик после установки внешнего держателя.",
        "3.3.3 Установка режущего модуля", "Закрепите режущий модуль в переднем отверстии корпуса.",
        "3.3.4 Установка отделителя", "Отделитель снимает подложку с готовой наклейки.",
        "# Фотографии и пояснения",
    ]
    for number, caption in enumerate(captions, 1):
        paragraphs += [f"Иллюстрация {number}. {caption}", caption]
    sections = instruction_sections(gallery_document(paragraphs, captions))
    for label, expected in [("3.3.2", [1, 2]), ("3.3.3", [3]), ("3.3.4", [4])]:
        group = next(section["group_id"] for section in sections if section["text"].startswith(label))
        actual = [number for section in sections if section["group_id"] == group for number in section["image_numbers"]]
        assert actual == expected
        assert not any(section.get("scope_ambiguous") for section in sections if section["group_id"] == group)


def test_numbered_chapter_gallery_preserves_conditional_alternative_metadata_across_pages():
    primary = "Выберите «Основной режим устройства» в окне настроек."
    alternative = "Выберите «Альтернативный режим устройства» в дополнительном окне."
    condition = "Если основной режим недоступен, отдельный способ: откройте дополнительное окно."
    source = gallery_document([
        "# Руководство", "3.2 Настройка рабочего режима", primary, condition,
        "Страница PDF 18", alternative, "# Фотографии и пояснения", primary, alternative,
    ], [primary, alternative])
    sections = instruction_sections(source)
    root = next(section for section in sections if section["text"].startswith("3.2"))
    branch = next(section for section in sections if section["text"] == condition)
    assert branch["parent_group_id"] == root["group_id"]
    alternative_image = next(section for section in sections if section["image_numbers"] == [2])
    assert alternative_image["group_id"] == branch["group_id"]
    assert alternative_image["condition_section_id"] == branch["id"]
    assert alternative_image["condition_text"] == "Если основной режим недоступен"
    assert next(section for section in sections if section["image_numbers"] == [1])["group_id"] == root["group_id"]


def test_gallery_platform_context_keeps_shared_login_steps_in_the_correct_application_variant():
    captions = [
        "На компьютере открыта первая форма рабочего кабинета.",
        "Мобильная форма рабочего кабинета запрашивает данные сотрудника.",
        "На форме проверки введите код из SMS, затем нажмите «Продолжить».",
        "На телефоне показана следующая страница рабочего кабинета.",
        "Логотип компании является оформлением документа.",
    ]
    source = gallery_document([
        "# Вход на компьютере", "Откройте рабочий кабинет на компьютере.",
        "На форме проверки введите код из SMS, затем нажмите «Продолжить».",
        "# На телефоне", "Установите мобильное приложение и войдите в рабочий кабинет.",
        "# Фотографии и пояснения", *captions,
    ], captions)
    sections = instruction_sections(source)
    desktop = sections[0]["group_id"]
    mobile = next(section["group_id"] for section in sections if section["text"] == "# На телефоне")
    assert [number for section in sections if section["group_id"] == desktop
            for number in section["image_numbers"]] == [1]
    assert [number for section in sections if section["group_id"] == mobile
            for number in section["image_numbers"]] == [2, 3, 4]
    logo = next(section for section in sections if section["image_numbers"] == [5])
    assert logo["group_id"] not in {desktop, mobile} and not logo.get("scope_ambiguous")


def test_gallery_does_not_extend_platform_context_across_conflicting_explicit_anchors():
    captions = ["Мобильная форма открыта на телефоне.", "Общая форма подтверждения операции.",
                "На компьютере открыта форма рабочего кабинета."]
    source = gallery_document([
        "# На телефоне", "Рабочий кабинет используется на телефоне.",
        "# На компьютере", "Рабочий кабинет используется на компьютере.",
        "# Фотографии и пояснения", *captions,
    ], captions)
    sections = instruction_sections(source)
    middle = next(section for section in sections if section["image_numbers"] == [2])
    assert middle.get("scope_ambiguous")
