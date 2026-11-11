"""Clarification may name only equipment present in this bot's approved originals."""

import pytest

from app.services.rag.query import scanner_kinds, source_compatible
from app.services.rag.talk import KnowledgeUnavailable
from tests.test_talk_knowledge import DOC, provider


def test_catalogue_reads_originals_and_uses_source_backed_model_without_generation():
    text = "Сканер штрихкодов Zebra DS2208. Настройка громкости сканера."
    knowledge, state = provider(filename="Настройка сканера.md", data=text.encode())
    result = knowledge.scanner_context()
    assert result["models"] == ["Zebra DS2208"]
    assert result["categories"] == ["barcode"]
    source = result["sources"][0]
    assert source["file_id"] == DOC and source["sha256"]
    assert source["filename"] == "Настройка сканера.md"
    assert not any(request.method == "POST" for request in state["calls"])


def test_forged_model_in_filename_is_not_promoted_to_source_backed_hint():
    knowledge, _ = provider(filename="Zebra DS2208 сканер.md", data="Настройка сканера штрихкодов в 1С АДРЕС.".encode())
    result = knowledge.scanner_context()
    assert result["models"] == []
    assert result["sources"][0]["applications"] == ["1С АДРЕС"]


def test_withdrawn_instruction_does_not_supply_clarification_models():
    knowledge, state = provider(filename="33_Сканер штрихкодов в базе АДРЕС.docx")
    result = knowledge.scanner_context()
    assert result == {"status": "ok", "sources": [], "categories": [], "models": []}
    assert not any("/files/" in request.url.path for request in state["calls"])


def test_source_prefix_does_not_remove_scanner_model_facts_from_approved_knowledge():
    knowledge, _ = provider(filename="ИТ_37_Zebra DS2208 сканер.md", data="Сканер штрихкодов Zebra DS2208.".encode())
    assert knowledge.scanner_context()["models"] == ["Zebra DS2208"]


def test_revoked_catalogue_source_cannot_supply_an_equipment_hint():
    knowledge, _ = provider(filename="Настройка сканера.md", data="Сканер штрихкодов Zebra DS2208.".encode(), revoked=True)
    with pytest.raises(KnowledgeUnavailable, match="SourceMembershipChanged"):
        knowledge.scanner_context()


@pytest.mark.parametrize("problem,original", [
    ("Не работает сканер документов", "Настройка сканера штрихкодов Zebra DS2208."),
    ("Настроить сканер ШК", "Сканирование документов на сканере Epson."),
    ("Подключить сканер штрихкодов", "Подключение сканера документов."),
])
def test_similar_word_scanner_does_not_mix_different_device_categories(problem, original):
    assert not source_compatible(problem, original, complete=True)


def test_document_mention_is_not_mistaken_for_document_scanner_type():
    assert scanner_kinds("Документ: подключение сканера штрихкодов в 1С АДРЕС") == {"barcode"}
    assert scanner_kinds("Инструкция сканера штрихкодов, откройте документ") == {"barcode"}
    assert source_compatible("Сканер ШК в 1С АДРЕС", "Сканер штрихкодов в 1С АДРЕС", complete=True)
