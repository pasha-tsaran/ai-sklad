"""Flexible terms cannot relax equipment, transport, source scope or provenance."""

import json

import pytest

from app.services.rag.query import choose_source, normalized_query, retrieval_query, source_compatible
from tests.test_openwebui_instruction_images import instruction
from tests.test_talk_knowledge import DOC, OTHER, provider


def test_query_normalization_handles_inflections_abbreviation_and_obvious_typos_without_repair_advice():
    assert retrieval_query("Подскажите, как подключить принтр к ТСД по WI-FI?") == "подключение принтер к тсд по wifi"
    assert retrieval_query("Настройка считывателя штрих-кода на терминале") == "настройка сканер штрихкод на тсд"
    assert normalized_query("ZD420: НЕ печатает по Wi-Fi") != normalized_query("ZD410: печатает по Bluetooth")
    assert normalized_query("USB не работает") != normalized_query("USB работает")
    assert normalized_query("Bluetooth / 1.0.53") != normalized_query("Wi-Fi / 1.0.52")


def test_scoped_hit_can_be_relevant_without_a_literal_query_word_in_its_title():
    chunks = [{"file_id": DOC, "source": "44_Подключение оборудования.docx",
               "content": "Подключите мобильный принтер к терминалу сбора данных ТСД по Wi-Fi."}]
    assert choose_source("как подключить принтр к тсд по вай фай", chunks) == DOC


def test_generic_printing_overlap_cannot_choose_queue_cleanup_for_missing_pdf_images():
    problem = "Принтер печатает PDF без картинок, остальные документы печатает нормально"
    chunks = [{"file_id": DOC, "source": "Очистка очереди принтера.md",
               "content": "Если принтер не печатает, очистите очередь печати."}]
    assert choose_source(problem, chunks) is None
    assert not source_compatible(problem, chunks[0]["content"], complete=True)


@pytest.mark.parametrize("problem", [
    "Принтер печатает полосами. Что делать?",
    "Принтер печатает бледно, помогите",
    "При печати всё смазано, как исправить?",
    "Принтер печатает нечетко и с пятнами",
])
def test_print_quality_symptoms_cannot_authorize_a_generic_queue_cleanup(problem):
    original = "Если принтер не печатает, очистите очередь печати."
    chunks = [{"file_id": DOC, "source": "Очистка очереди принтера.md", "content": original}]
    assert choose_source(problem, chunks) is None
    assert not source_compatible(problem, chunks[0]["source"] + "\n" + original, complete=True)


@pytest.mark.parametrize("problem", ["Принтер не работает, помогите", "Как настроить сканер?"])
def test_only_generic_device_and_action_overlap_requires_semantic_confirmation(problem):
    chunks = [{"file_id": DOC, "source": "Настройка оборудования.md",
               "content": "Настройте принтер и сканер. Проверьте работу устройств."}]
    assert choose_source(problem, chunks) is None


def test_quality_document_can_explain_a_different_wording_of_print_defects():
    problem = "Принтер печатает бледно и с пятнами"
    original = "Качество печати принтера: проверьте состояние картриджа и уровень тонера."
    assert source_compatible(problem, original, complete=True)
    # There is no symptom-specific lexical overlap; a semantic choice may
    # confirm the document, but generic printer/printing is not enough alone.
    assert choose_source(problem, [{"file_id": DOC, "source": "Качество.md", "content": original}]) is None


def test_specific_application_and_connection_keep_fast_direct_search():
    scanner = [{"file_id": DOC, "source": "Настройка сканера 1С АДРЕС.docx",
                "content": "В 1С АДРЕС откройте параметры сканера штрихкодов."}]
    assert choose_source("Как настроить сканер штрихкодов в 1С АДРЕС?", scanner) == DOC
    mobile = [{"file_id": DOC, "source": "Подключение оборудования.docx",
               "content": "Подключите мобильный принтер к терминалу ТСД по Wi-Fi."}]
    assert choose_source("Как подключить принтер к ТСД по Wi-Fi?", mobile) == DOC


@pytest.mark.parametrize("problem,original", [
    ("Как подключить принтер к компьютеру?", "Если принтер не печатает, очистите очередь печати."),
    ("Сканер громко пищит после считывания", "Подключение сканера. Установите драйвер сканера."),
    ("Как очистить очередь печати?", "Настройте подключение принтера к компьютеру."),
])
def test_full_original_cannot_replace_the_requested_task_with_another_hardware_task(problem, original):
    assert not source_compatible(problem, original, complete=True)


def test_semantic_selection_with_real_quote_still_cannot_turn_queue_cleanup_into_pdf_image_fix():
    problem = "Принтер печатает PDF без картинок, остальные документы печатает нормально"
    wrong = "Если принтер не печатает, очистите очередь печати. Команда выполняется администратором."
    knowledge, state = provider(data=wrong.encode(), filename="Очистка очереди принтера.md", chunks=[(DOC, wrong)])
    calls = []
    knowledge.selector = lambda payload: calls.append(payload) or {"file_id": DOC, "evidence": wrong}
    assert knowledge.search(problem) is None
    assert len(calls) == 1  # The quoted source is real, but its task is inapplicable.


def test_task_guards_keep_direct_matches_from_source_content_instead_of_title_only():
    problem = "В принтере PDF без картинок, как исправить?"
    content = "В Acrobat включите параметр «Печатать как изображение», затем напечатайте PDF на принтере."
    chunks = [{"file_id": DOC, "source": "44_Утвержденная инструкция.md", "content": content}]
    assert choose_source(problem, chunks) == DOC
    assert source_compatible(problem, content, complete=True)
    assert source_compatible("Как подключить принтр к ТСД по Wi-Fi?",
                             "Настройка мобильного принтера на ТСД по Wi-Fi.", complete=True)


@pytest.mark.parametrize("query,text", [
    ("подключить принтер к ТСД по Wi-Fi", "Мобильный принтер подключается к ТСД через Bluetooth."),
    ("подключить принтер через Bluetooth", "Принтер подключается по Wi-Fi."),
    ("настроить сканер", "Откройте настройки принтера и напечатайте тест."),
    ("подключить принтер к ТСД", "Откройте настройки принтера на компьютере."),
])
def test_full_original_must_cover_explicit_device_and_connection(query, text):
    assert not source_compatible(query, text, complete=True)


@pytest.mark.parametrize("problem,source", [
    ("Zebra ZD420 не печатает", "Инструкция принтера Zebra ZD410."),
    ("Сканер в 1С АДРЕС версии 1.0.53", "Сканер в 1С АДРЕС версии 1.0.52."),
    ("Мобильный принтер ZQ-520", "Подключите мобильный принтер ZQ-620."),
])
def test_full_original_cannot_substitute_a_similar_model_or_software_version(problem, source):
    assert not source_compatible(problem, source, complete=True)


def test_unrelated_counts_are_not_hardware_identifiers_and_matching_version_is_accepted():
    assert source_compatible("Принтер в 192.168.10.198 печатает 3 страницы", "Настройка принтера.", complete=True)
    assert source_compatible("Zebra ZD420 в 1С АДРЕС версии 1.0.53",
                             "Zebra ZD420. Настройка 1С АДРЕС 1.0.53.", complete=True)


def test_semantic_selection_returns_only_verified_original_and_photos_with_literal_evidence():
    original = instruction(["Настройка мобильного принтера на терминале сбора данных ТСД."])
    excerpt = "Печать этикеток номенклатуры с терминала сбора данных. Выберите доступное печатающее устройство."
    knowledge, state = provider(data=original, filename="44_Label Mobile.docx", chunks=[(DOC, excerpt)])
    calls = []

    def select(payload):
        calls.append(payload)
        return {"file_id": DOC, "evidence": excerpt}

    knowledge.selector = select
    approved = knowledge.search("Нужен мобильный принтер для ценников на ТСД")
    assert len(calls) == 1
    assert calls[0]["actual_problem"] == "Нужен мобильный принтер для ценников на ТСД"
    assert "Настройка мобильного принтера на терминале сбора данных ТСД." in approved.document["content"]
    assert len(approved.illustrations) == 1
    assert state["data"] == original


@pytest.mark.parametrize("choice", [
    {"file_id": OTHER, "evidence": "Настройка подключения принтера к ТСД."},
    {"file_id": DOC, "evidence": "Выдуманная цитата про принтер и ТСД из внешнего источника."},
    {"file_id": DOC, "evidence": "Принтер", "answer": "Удалите все задания."},
    {"file_id": None, "evidence": ""},
])
def test_untrusted_semantic_selection_cannot_forge_source_evidence_or_add_answer(choice):
    knowledge, state = provider(chunks=[(DOC, "Устройства печати и передача документов.")])
    knowledge.selector = lambda payload: choice
    assert knowledge.search("Мобильный принтер для ценников на ТСД") is None
    assert not any("/files/" in request.url.path for request in state["calls"])


def test_generated_query_cannot_override_actual_wifi_problem_with_bluetooth_source():
    excerpt = "Мобильный принтер и терминал ТСД соединяются через Bluetooth."
    knowledge, state = provider(data=instruction([excerpt]), filename="44_Принтер ТСД Bluetooth.docx",
                                chunks=[(DOC, excerpt)])
    calls = []
    knowledge.selector = lambda payload: calls.append(payload) or {"file_id": DOC, "evidence": excerpt}
    assert knowledge.search("принтер ТСД Bluetooth", fallback_query="Подключить принтер к ТСД по Wi-Fi") is None
    assert calls == []
    assert not any("/files/" in request.url.path for request in state["calls"])


def test_model_failure_never_becomes_an_instruction_or_an_unbounded_retry():
    knowledge, state = provider(chunks=[(DOC, "Неполный фрагмент справки.")])
    calls = []

    def fail(payload):
        calls.append(payload)
        raise TimeoutError("private model response")

    knowledge.selector = fail
    assert knowledge.search("Мобильный принтер ТСД") is None
    assert len(calls) == 1
    assert len([r for r in state["calls"] if r.method == "POST"]) == 2


def test_normalized_fallback_searches_only_current_configured_collection():
    knowledge, state = provider(chunks=[])
    assert knowledge.search("Как подключить принтр к тсд по Wi-Fi?") is None
    payloads = [json.loads(request.content) for request in state["calls"] if request.method == "POST"]
    assert [payload["query"] for payload in payloads] == [
        "Как подключить принтр к тсд по Wi-Fi?", "подключение принтер к тсд по wifi",
    ]
    assert all(payload["collection_names"] == knowledge.settings.openwebui_knowledge_ids for payload in payloads)
