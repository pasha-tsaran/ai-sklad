"""Native Talk retrieval remains scoped to approved current original uploads."""

import hashlib
import json

import httpx
import pytest

from app.core.config import Settings
from app.services.rag.talk import KnowledgeUnavailable, TalkKnowledge
from integrations.openwebui.approved_instruction_images import read_docx_images, read_docx_text
from tests.test_openwebui_instruction_images import instruction, office

KB = "85cda3aa-b854-4c26-9948-12da742277ef"
DOC = "00000000-0000-4000-8000-000000000033"
OTHER = "00000000-0000-4000-8000-000000000099"
FILENAME = "05_Acrobat — PDF с картинками не печатается.docx"
QUERY = "Acrobat PDF с картинками не печатается"


def provider(*, data=None, filename=FILENAME, chunks=None, retrieval_status=200, revoked=False, query_results=None):
    data = data if data is not None else instruction(["Нажмите «Дополнительно» и включите «Печатать как изображение»."])
    state = {"data": data, "files": [{"id": DOC, "filename": filename}], "collections": 0, "calls": []}

    def handler(request):
        state["calls"].append(request)
        assert request.headers["Authorization"] == "Bearer test-key"
        assert request.url.host == "webui.example.invalid"
        if request.url.path == "/api/v1/knowledge/" + KB:
            state["collections"] += 1
            return httpx.Response(200, json={"files": [] if revoked and state["collections"] > 1 else state["files"]})
        if request.url.path == "/api/v1/retrieval/query/collection":
            payload = json.loads(request.content)
            assert payload["collection_names"] == [KB] and payload["k"] == 8
            if retrieval_status != 200:
                return httpx.Response(retrieval_status, json={"error": "temporary"})
            found = chunks if chunks is not None else [(DOC, "Acrobat PDF с картинками не печатается: печать как изображение.")]
            if query_results is not None:
                found = query_results.get(payload["query"], found)
            return httpx.Response(200, json={"documents": [[text for _, text in found]],
                                             "metadatas": [[{"file_id": file_id, "source": "forged-name.docx"}
                                                            for file_id, _ in found]]})
        if request.url.path == "/api/v1/files/" + DOC + "/content":
            return httpx.Response(200, content=state["data"])
        if request.url.path == "/api/v1/files/" + DOC + "/data/content":
            return httpx.Response(200, json={"content": state["data"].decode()})
        raise AssertionError(request.url)

    settings = Settings(_env_file=None, openwebui_base_url="https://webui.example.invalid",
                        openwebui_api_key="test-key", openwebui_knowledge_ids=[KB])
    return TalkKnowledge(settings, httpx.MockTransport(handler)), state


def test_native_scoped_retrieval_reads_the_same_docx_text_and_pixels_as_webui_tool():
    knowledge, state = provider()
    approved = knowledge.search(QUERY)
    assert approved.document["content"] == read_docx_text(state["data"])
    expected = read_docx_images(state["data"])[0]
    assert approved.illustrations[1].data == expected.data
    assert approved.illustrations[1].caption == expected.caption
    assert approved.document["source"] == {"file_id": DOC, "filename": FILENAME, "knowledge_id": KB,
                                            "sha256": hashlib.sha256(state["data"]).hexdigest()}
    assert approved.candidate.source["original_sha256"] == approved.document["source"]["sha256"]
    assert all("/chat/completions" not in str(request.url) for request in state["calls"])


def test_explicit_original_read_checks_approved_membership_before_and_after():
    knowledge, state = provider()
    approved = knowledge.read(DOC)
    assert approved.document["status"] == "ready" and state["collections"] == 2
    knowledge, _ = provider(revoked=True)
    with pytest.raises(KnowledgeUnavailable, match="SourceMembershipChanged"):
        knowledge.read(DOC)


def test_source_hash_revalidation_detects_original_changes_and_revocation():
    knowledge, state = provider()
    original = knowledge.read(DOC).document["source"]
    assert knowledge.is_current(original)
    state["data"] = instruction(["Другая иллюстрация."])
    assert not knowledge.is_current(original)
    state["files"] = []
    assert not knowledge.is_current(original)


def test_no_semantic_result_is_distinct_from_service_unavailability():
    knowledge, _ = provider(chunks=[])
    assert knowledge.search(QUERY) is None
    knowledge, _ = provider(retrieval_status=503)
    with pytest.raises(KnowledgeUnavailable):
        knowledge.search(QUERY)


def test_foreign_vector_metadata_never_authorizes_a_file_outside_allowed_collection():
    knowledge, state = provider(chunks=[(OTHER, "Acrobat PDF с картинками не печатается")])
    assert knowledge.search(QUERY) is None
    assert all("/files/" not in request.url.path for request in state["calls"])


@pytest.mark.parametrize("filename", ["ИТ_05_Acrobat.docx", "СПРАВОЧНИК_Acrobat.docx", "Acrobat.exe"])
def test_administrative_and_unsupported_uploads_are_excluded_before_retrieval(filename):
    knowledge, state = provider(filename=filename)
    assert knowledge.search(QUERY) is None
    assert len(state["calls"]) == 1


def test_concise_analysis_query_has_one_bounded_full_problem_fallback():
    knowledge, state = provider(query_results={"неясно": []})
    approved = knowledge.search("неясно", fallback_query=QUERY)
    searches = [json.loads(request.content)["query"] for request in state["calls"] if request.method == "POST"]
    assert searches == ["неясно", QUERY] and approved is not None


def test_generated_search_terms_cannot_authorize_an_unrelated_source_for_the_actual_problem():
    knowledge, state = provider(chunks=[(OTHER, "Телефон Samsung: настройка звонков")])
    state["files"].append({"id": OTHER, "filename": "Телефон Samsung звонки.docx"})
    assert knowledge.search("Телефон Samsung звонки", fallback_query=QUERY) is None
    assert all("/files/" not in request.url.path for request in state["calls"])


def test_excluded_files_are_not_repeated_as_another_attempt():
    knowledge, state = provider()
    assert knowledge.search(QUERY, excluded=[DOC]) is None
    assert len(state["calls"]) == 1


def test_invalid_scope_or_file_reference_never_expands_to_all_available_knowledge():
    knowledge, state = provider()
    knowledge.settings.openwebui_knowledge_ids = []
    with pytest.raises(KnowledgeUnavailable, match="KnowledgeScopeNotConfigured"):
        knowledge.search(QUERY)
    with pytest.raises(KnowledgeUnavailable, match="InvalidSourceId"):
        knowledge.read("https://untrusted.invalid/file.docx")
    assert not state["calls"]


def test_plain_markdown_instructions_are_read_as_original_utf8_without_invented_images():
    knowledge, _ = provider(filename="05_Acrobat PDF печать.md", data="Нажмите Ctrl+P.\nПроверьте печать PDF.".encode())
    approved = knowledge.search(QUERY)
    assert approved.document["images"] == [] and approved.illustrations == {}
    assert approved.document["content"] == "Нажмите Ctrl+P.\nПроверьте печать PDF."


def test_pdf_uses_complete_webui_extracted_text_and_revalidates_its_hash_without_invented_photos():
    knowledge, state = provider(filename="05_Acrobat PDF печать.pdf", data="Откройте PDF и нажмите Ctrl+P.".encode())
    approved = knowledge.search(QUERY)
    assert approved.document["content"] == state["data"].decode()
    assert approved.document["images"] == [] and approved.illustrations == {}
    assert approved.candidate.source["original_sha256"] is None
    assert knowledge.is_current(approved.document["source"])
    state["data"] = "Изменённая инструкция PDF.".encode()
    assert not knowledge.is_current(approved.document["source"])


def test_xlsx_tables_keep_all_original_cells_without_model_generated_prose():
    data = office({
        "xl/workbook.xml": '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
                           'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
                           '<sheets><sheet name="Acrobat" r:id="s1"/></sheets></workbook>',
        "xl/_rels/workbook.xml.rels": '<Relationships><Relationship Id="s1" Target="/xl/worksheets/sheet1.xml"/></Relationships>',
        "xl/worksheets/sheet1.xml": '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
                                   '<sheetData><row><c r="A1" t="inlineStr"><is><t>Печать PDF</t></is></c>'
                                   '<c r="B1"><v>141</v></c></row></sheetData></worksheet>',
    })
    knowledge, _ = provider(filename="Acrobat PDF печать.xlsx", data=data)
    approved = knowledge.search(QUERY)
    assert "A1: Печать PDF | B1: 141" in approved.document["content"]
    assert approved.document["images"] == []


@pytest.mark.parametrize("data", [b"not a zip", instruction([""]), instruction(external=True)])
def test_invalid_original_or_missing_photo_description_is_an_unavailable_source(data):
    knowledge, _ = provider(data=data)
    with pytest.raises(KnowledgeUnavailable):
        knowledge.read(DOC)
