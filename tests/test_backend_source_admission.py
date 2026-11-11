"""All backend paths share the collection's instruction admission policy."""

import pytest

from app.services.rag.openwebui import OpenWebUIKnowledge
from app.services.rag.success import _valid_source
from tests.test_talk_knowledge import DOC, KB, provider


@pytest.mark.parametrize("filename", [
    "ИТ_37_Zebra DS2208 — лист настроечных штрихкодов.md",
    "37_Zebra DS2208 — лист настроечных штрихкодов.md",
    "СПРАВОЧНИК_Сканер штрихкодов.md",
])
@pytest.mark.parametrize("paginated", [False, True])
def test_api_backend_and_success_cache_accept_scoped_instructions_with_any_prefix(filename, paginated):
    knowledge, _ = provider(filename=filename, data="Сканер штрихкодов: громкость и подсветка.".encode(), paginated=paginated)
    legacy = OpenWebUIKnowledge(knowledge.settings, transport=knowledge.transport)
    with knowledge._client() as client:
        assert DOC in legacy._files(client)
    source = {"file_id": DOC, "filename": filename, "knowledge_id": KB, "sha256": "a" * 64}
    assert _valid_source(knowledge.settings, source)


def test_retired_instruction_is_not_retrieved_or_restored_from_success_cache():
    filename = "33_Сканер штрихкодов в базе АДРЕС.docx"
    knowledge, _ = provider(filename=filename)
    legacy = OpenWebUIKnowledge(knowledge.settings, transport=knowledge.transport)
    with knowledge._client() as client:
        assert legacy._files(client) == {}
    source = {"file_id": DOC, "filename": filename, "knowledge_id": KB, "sha256": "a" * 64}
    assert not _valid_source(knowledge.settings, source)
