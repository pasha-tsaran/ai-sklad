"""Evidence references preserve the current original without model-made quotes."""

import copy
import re

import pytest

from app.services.rag.evidence import build_evidence_registry


def canonical(text):
    return " ".join(text.split())


def assert_exact_registry(content, captions, registry):
    original = canonical(content)
    for reference, item in registry.items():
        assert set(item) == {"text", "support_kind", "image_numbers"}
        assert 0 < len(item["text"]) <= 200 and item["text"] in original
        if item["support_kind"] == "IMAGE":
            number, = item["image_numbers"]
            assert number in captions and item["text"] in canonical(captions[number])
            assert re.fullmatch(rf"i{number}_[1-9][0-9]*", reference)
        else:
            assert item["support_kind"] == "TEXT" and item["image_numbers"] == []
            assert re.fullmatch(r"t[1-9][0-9]*", reference)


def test_unicode_and_punctuation_are_preserved_with_only_whitespace_normalization():
    text = "Заголовок\n\nВ окне «Печать» выберите A4 — это не A-4.\tСуффикс #13; имя файла: данные.xlsx."
    registry = build_evidence_registry(text)
    assert registry == {"t1": {"text": canonical(text), "support_kind": "TEXT", "image_numbers": []}}
    assert "«Печать»" in registry["t1"]["text"]
    assert "—" in registry["t1"]["text"] and "#13;" in registry["t1"]["text"]


@pytest.mark.parametrize("length", [1, 19, 20, 199, 200, 201, 219, 220, 399, 400, 401])
def test_unbroken_tokens_keep_bounds_and_the_last_short_window_overlaps_the_same_source(length):
    text = "Ж" * length
    registry = build_evidence_registry(text)
    assert_exact_registry(text, {}, registry)
    if length <= 200:
        assert [item["text"] for item in registry.values()] == [text]
    else:
        assert all(20 <= len(item["text"]) <= 200 for item in registry.values())
        assert list(registry.values())[-1]["text"] == text[-len(list(registry.values())[-1]["text"]):]


def test_short_header_and_tail_remain_separate_context_across_an_image_caption():
    caption = "На рисунке показано окно выбора рабочего устройства."
    text = f"Вход\n\n{caption}\n\nГотово"
    registry = build_evidence_registry(text, {1: caption})
    assert list(registry) == ["t1", "i1_1", "t2"]
    assert registry["t1"]["text"] == "Вход" and registry["t2"]["text"] == "Готово"
    assert all(item["text"] != "Вход Готово" for item in registry.values())
    assert_exact_registry(text, {1: caption}, registry)


def test_long_paragraphs_keep_all_words_and_every_original_paragraph():
    paragraphs = [f"Раздел {number}. " + " ".join(f"элемент{number}_{word}" for word in range(80))
                  for number in range(4)]
    text = "\n\n".join(paragraphs)
    registry = build_evidence_registry(text)
    assert len(registry) > 10
    for word in canonical(text).split():
        assert any(word in item["text"] for item in registry.values())
    assert_exact_registry(text, {}, registry)


def test_short_final_word_window_expands_using_only_its_own_body_span():
    caption = "Исходная фотография показывает другое окно настройки."
    first = "первый " * 28 + "край"
    second = "второй " * 28 + "хвост"
    text = first + caption + second
    registry = build_evidence_registry(text, {4: caption})
    assert_exact_registry(text, {4: caption}, registry)
    image_index = list(registry).index("i4_1")
    before = list(registry.values())[:image_index]
    after = list(registry.values())[image_index + 1:]
    assert before and after
    assert all(item["text"] in canonical(first) for item in before)
    assert all(item["text"] in canonical(second) for item in after)
    assert all(len(item["text"]) >= 20 for item in before + after)


def test_caption_windows_use_the_actual_number_and_cannot_become_body_text():
    caption = " ".join(f"параметр{number:03d}" for number in range(70))
    text = "Начало инструкции. " + caption + " Конец инструкции."
    registry = build_evidence_registry(text, {7: caption})
    images = [item for item in registry.values() if item["support_kind"] == "IMAGE"]
    assert len(images) > 1 and all(item["image_numbers"] == [7] for item in images)
    assert all(20 <= len(item["text"]) <= 200 for item in images)
    assert [key for key in registry if key.startswith("i")] == [f"i7_{number}" for number in range(1, len(images) + 1)]
    assert_exact_registry(text, {7: caption}, registry)


def test_longest_caption_wins_when_one_caption_is_contained_in_another():
    short = "Нажмите кнопку подтверждения."
    long = "После выбора устройства " + short + " Дождитесь результата проверки."
    registry = build_evidence_registry("Начало. " + long + " Конец.", {1: short, 2: long})
    images = {key: item for key, item in registry.items() if item["support_kind"] == "IMAGE"}
    assert list(images) == ["i2_1"]
    assert images["i2_1"]["text"] == long


def test_duplicate_caption_uses_the_lowest_original_image_number_deterministically():
    caption = "Выберите кнопку подтверждения в показанном окне."
    registry = build_evidence_registry(caption, {8: caption, 3: caption, 5: caption})
    assert registry == {"i3_1": {"text": caption, "support_kind": "IMAGE", "image_numbers": [3]}}
    assert registry == build_evidence_registry(caption, {5: caption, 8: caption, 3: caption})


def test_repeated_caption_is_registered_once_and_does_not_join_its_body_gaps():
    caption = "Иллюстрация показывает выбор папки для сохранения файла."
    text = "Первая часть. " + caption + " Вторая часть. " + caption + " Третья часть."
    registry = build_evidence_registry(text, {2: caption})
    assert list(registry) == ["t1", "i2_1", "t2", "t3"]
    assert [item["text"] for item in registry.values() if item["support_kind"] == "TEXT"] == [
        "Первая часть.", "Вторая часть.", "Третья часть.",
    ]
    assert_exact_registry(text, {2: caption}, registry)


def test_caption_matching_is_literal_and_whitespace_normalized_without_fuzzy_changes():
    caption = "Опция [A+B] (порт 1.2) — «Подключить»?"
    body = "Опция\n[A+B]   (порт 1.2) — «Подключить»?"
    registry = build_evidence_registry(body, {6: caption})
    assert list(registry) == ["i6_1"] and registry["i6_1"]["text"] == caption
    different = build_evidence_registry(body, {6: caption.replace("—", "-")})
    assert all(item["support_kind"] == "TEXT" for item in different.values())


def test_captions_missing_from_the_original_cannot_provide_image_evidence():
    content = "Откройте окно выбора документа и проверьте показанное имя."
    caption = "Неподтверждённая подпись, отсутствующая в текущем оригинале."
    registry = build_evidence_registry(content, {9: caption})
    assert registry == build_evidence_registry(content)
    assert all(item["support_kind"] == "TEXT" for item in registry.values())


def test_registry_is_deterministic_ordered_and_does_not_mutate_its_inputs():
    captions = {4: "На втором снимке показано окно завершения настройки.",
                1: "На первом снимке показано исходное окно программы."}
    content = "Краткое введение. " + captions[1] + " Промежуточный шаг. " + captions[4] + " Завершение."
    original = copy.deepcopy(captions)
    expected = build_evidence_registry(content, captions)
    assert list(expected) == ["t1", "i1_1", "t2", "i4_1", "t3"]
    for _ in range(3):
        assert build_evidence_registry(content, captions) == expected
    assert captions == original


@pytest.mark.parametrize("content,captions", [(None, None), (123, {}), ("Текст", []), ("Текст", "caption")])
def test_invalid_source_shapes_are_rejected(content, captions):
    with pytest.raises(ValueError, match="InvalidEvidenceSource"):
        build_evidence_registry(content, captions)


@pytest.mark.parametrize("captions", [{0: "Подпись"}, {-1: "Подпись"}, {True: "Подпись"},
                                     {"1": "Подпись"}, {1: None}, {1: "Подпись", "2": "Подпись"}])
def test_invalid_image_numbers_and_caption_shapes_are_rejected(captions):
    with pytest.raises(ValueError, match="InvalidEvidenceCaption"):
        build_evidence_registry("Исходный текст", captions)


@pytest.mark.parametrize("content", ["", " \n\t "])
def test_empty_originals_have_no_evidence_references(content):
    assert build_evidence_registry(content, {1: "Несуществующая подпись"}) == {}


def test_existing_source_and_image_limits_keep_every_caption_in_its_original():
    captions = {number: f"На иллюстрации {number:02d} показан исходный пункт {number:02d} настройки программы."
                for number in range(1, 61)}
    content = "\n".join(captions.values()) + "\n" + " ".join(f"слово{number:04d}" for number in range(2400))
    assert len(content) < 32_000
    registry = build_evidence_registry(content, captions)
    assert len(registry) < 300
    assert {item["image_numbers"][0] for item in registry.values() if item["support_kind"] == "IMAGE"} == set(captions)
    assert_exact_registry(content, captions, registry)
