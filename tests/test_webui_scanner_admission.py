"""Approved scanner setup remains usable through standalone Open WebUI entry points."""

import json
from types import SimpleNamespace

import pytest

from tests.test_openwebui_instruction_images import (
    FakeRuntime, SOURCE_ID, display, image_tool as image_tool, instruction, run, source_store,
)
from tests.test_openwebui_automatic_images import (
    DOC_ID, KNOWLEDGE_ID, RuntimeState, answer, apply_filter, automatic as automatic,
    chat_body, document, file_events, native_pair,
)

WITHDRAWN = "33_Сканер штрихкодов в базе АДРЕС.docx"
SETTINGS = "ИТ_37_Zebra DS2208 — лист настроечных штрихкодов.docx"
REQUEST = "Настроить луч сканера ШК и уменьшить громкость"
CAPTION = "Настройка подсветки и громкости сканера штрихкодов. Выберите нужный настроечный штрихкод."


@pytest.mark.parametrize("filename", [SETTINGS, "37_Zebra DS2208 — лист настроечных штрихкодов.docx",
                                     "ИТ_01_Настройка нового сканера штрихкодов в 1С.md",
                                     "СПРАВОЧНИК_Настройки сканера.pdf"])
def test_both_standalone_policies_admit_supported_scoped_sources_regardless_of_prefix(image_tool, automatic, filename):
    assert image_tool.allowed_source_filename(filename)
    assert automatic.allowed_source_filename(filename)


@pytest.mark.parametrize("filename", [WITHDRAWN, "33-СКАНЕР ШТРИХКОДОВ В БАЗЕ АДРЕС.docx", "Файл.exe", ""])
def test_both_standalone_policies_reject_withdrawn_or_unsupported_sources(image_tool, automatic, filename):
    assert not image_tool.allowed_source_filename(filename)
    assert not automatic.allowed_source_filename(filename)


def test_native_tool_returns_original_scanner_photo_for_it37(image_tool, monkeypatch):
    original = instruction((CAPTION,), extra_paragraph="Только для ИТ относится к установке драйвера.")
    runtime = FakeRuntime(image_tool, original)
    runtime.file.filename = SETTINGS
    result, events = display(image_tool, monkeypatch, runtime)
    assert result["status"] == "displayed" and result["source"]["filename"] == SETTINGS
    assert result["images"][0]["caption"] == CAPTION
    assert runtime.uploads[0][0].data == image_tool.read_docx_images(original)[0].data
    assert len([event for event in events if event["type"] == "files"]) == 1


def test_native_tool_rejects_withdrawn_scanner_original_before_read_or_upload(image_tool, monkeypatch):
    runtime = FakeRuntime(image_tool)
    runtime.file.filename = WITHDRAWN
    result, events = display(image_tool, monkeypatch, runtime)
    assert result["code"] == "withdrawn_instruction" and not result["displayed"]
    assert not runtime.uploads and not events


def test_native_source_authorization_does_not_return_the_withdrawn_document(image_tool, monkeypatch):
    source_store(image_tool, monkeypatch, filename=WITHDRAWN)
    with pytest.raises(image_tool.ToolError, match="исключена"):
        run(image_tool._OpenWebUI().source(SOURCE_ID, "employee", [KNOWLEDGE_ID]))


def test_approved_markdown_keeps_source_conditions_as_readable_text(image_tool, monkeypatch):
    runtime = FakeRuntime(image_tool)
    runtime.file.filename = "ИТ_01_Настройка сканера.md"
    runtime.file.data = {"content": "Только для ИТ. Установка системного драйвера. Остальные настройки описаны ниже."}
    monkeypatch.setattr(image_tool, "_OpenWebUI", lambda: runtime)
    result = json.loads(run(image_tool.Tools().read_approved_instruction(
        SOURCE_ID, __user__={"id": "employee"}, __request__=SimpleNamespace(),
    )))
    assert result["status"] == "ready" and result["content"] == runtime.file.data["content"]


def test_filter_search_finds_volume_and_light_by_content_without_model_in_question(automatic, monkeypatch):
    approved = document(1, filename=SETTINGS, captions=[CAPTION])
    approved["content"] = "Настройки сканера штрихкодов.\n\n1. Выберите нужный код громкости и подсветки."
    state = RuntimeState({DOC_ID: approved})
    state.search_result = {"status": "ok", "chunks": [
        {"file_id": DOC_ID, "source": SETTINGS, "content": approved["content"]},
    ]}
    body = chat_body(messages=[{"id": "u1", "role": "user", "content": REQUEST},
                               answer(read=False, content="Какой модели ваш сканер?")])
    result = apply_filter(automatic, monkeypatch, state, body)
    assert state.reads == [DOC_ID] and state.shows == [(DOC_ID, "1", 0)]
    assert CAPTION in result["messages"][-1]["content"]
    assert "модели" not in result["messages"][-1]["content"]
    assert len(file_events(state)) == 1


def test_scanner_connection_source_cannot_substitute_for_beeper_and_light_settings(automatic):
    chunks = [{"file_id": DOC_ID, "source": "01_Подключение сканера.docx",
               "content": "Подключите сканер штрихкодов в программе и проверьте считывание."}]
    assert automatic.choose_search_source(REQUEST, chunks) is None


def test_long_equipment_type_answer_retains_the_goal_from_the_previous_question(automatic):
    messages = [
        {"role": "user", "content": "Мне нужно настроить луч сканера и громкость"},
        {"role": "assistant", "content": "Вы имеете в виду сканер штрихкодов или сканер документов?"},
        {"role": "user", "content": "сканер для считывания штрих кодов"},
    ]
    query = automatic.problem_query(messages, [(0, messages[0]), (2, messages[2])])
    assert "луч сканера и громкость" in query
    assert "сканер для считывания штрих кодов" in query
    assert automatic.equipment_type_question(query) is None


@pytest.mark.parametrize("query,expected", [
    ("Не работает сканер", "сканер штрихкодов или сканер документов"),
    ("Настройка сканера ШК", "Что именно нужно настроить"),
    ("Не работает принтер", "принтер документов А4, стационарный термопринтер или мобильный термопринтер"),
    ("Настроить термопринтер", "Это стационарный термопринтер или мобильный термопринтер?"),
])
def test_filter_asks_one_equipment_type_or_goal_question_before_arbitrary_search(automatic, monkeypatch, query, expected):
    state = RuntimeState()
    body = chat_body(messages=[{"id": "u1", "role": "user", "content": query},
                               answer(read=False, content="Какая модель оборудования?")])
    result = apply_filter(automatic, monkeypatch, state, body)
    text = result["messages"][-1]["content"]
    assert expected in text and text.count("?") == 1 and "модел" not in text.casefold()
    assert not state.searches and not state.reads and not state.shows


def test_old_read_or_search_result_cannot_authorize_withdrawn_scanner_guide(automatic):
    message = answer(filename=WITHDRAWN)
    message["output"] += native_pair("search_approved_instructions", {"query": "сканер"}, {
        "status": "ok", "knowledge_ids": [KNOWLEDGE_ID], "chunks": [
            {"file_id": DOC_ID, "source": WITHDRAWN, "content": "Настройка сканера штрихкодов."},
        ],
    })
    assert automatic.selected_source(message) is None
    assert automatic.searched_source(message, "сканер штрихкодов в АДРЕС", [KNOWLEDGE_ID]) is None


def test_filter_refuses_retired_reader_response_even_when_runtime_returns_ready(automatic, monkeypatch):
    state = RuntimeState({DOC_ID: document(filename=WITHDRAWN)})
    body = chat_body(answer(filename="Актуальные операции.docx"))
    apply_filter(automatic, monkeypatch, state, body)
    assert state.reads == [DOC_ID] and not state.shows and not file_events(state)


def test_uncertain_retired_photo_result_is_neutralized_without_a_retry(automatic, monkeypatch):
    state = RuntimeState()
    message = answer(read=False, content="Повтор отозванной инструкции.")
    message["output"] += native_pair("show_instruction_images", {"file_id": DOC_ID}, {
        "status": "delivery_unknown", "displayed": False,
        "source": {"file_id": DOC_ID, "filename": WITHDRAWN, "knowledge_id": KNOWLEDGE_ID},
    })
    result = apply_filter(automatic, monkeypatch, state, chat_body(message))
    assert "исключена из базы знаний" in result["messages"][-1]["content"]
    assert "Повтор отозванной" not in result["messages"][-1]["content"]
    assert not state.searches and not state.reads and not state.shows and not file_events(state)


def test_prompt_uses_types_and_admits_instructions_by_collection():
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "docs/prompts/openwebui-system-prompt.txt").read_text(encoding="utf-8")
    assert "сканер штрихкодов или сканер документов/планшетный" in text
    assert "стационарный термопринтер и мобильный термопринтер" in text
    assert "Все загруженные инструкции разрешённой коллекции участвуют" in text
    assert "33_Сканер штрихкодов в базе АДРЕС.docx" in text
