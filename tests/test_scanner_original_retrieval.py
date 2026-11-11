"""Specific scanner hardware tasks can use current originals without vector or model latency."""

import json

import httpx
import pytest

from app.core.config import Settings
from app.services.rag.talk import MAX_SCANNER_ORIGINALS, KnowledgeUnavailable, TalkKnowledge
from integrations.openwebui.approved_instruction_images import read_docx_images
from tests.test_openwebui_instruction_images import instruction
from tests.test_talk_knowledge import DOC, KB, OTHER, provider

PROBLEM = "Настройка сканер ШК\nуменьшить громкость"
VOLUME = "Сканер штрихкодов. Для настройки громкости используйте настроечный штрихкод на листе."
FILENAME = "37_Zebra DS2208 — лист настроечных штрихкодов.docx"


def scanner_provider(rows, *, chunks=(), revoke_after=None):
    state = {"files": [{"id": key, "filename": name} for key, name, _ in rows],
             "originals": {key: data for key, _, data in rows}, "calls": [], "metadata_reads": 0}

    def handler(request):
        state["calls"].append(request)
        assert request.url.host == "webui.example.invalid"
        assert request.headers["Authorization"] == "Bearer test-key"
        if request.url.path == "/api/v1/knowledge/" + KB:
            state["metadata_reads"] += 1
            files = state["files"]
            if revoke_after is not None and state["metadata_reads"] > revoke_after:
                files = []
            return httpx.Response(200, json={"id": KB, "files": files})
        if request.url.path == "/api/v1/retrieval/query/collection":
            assert json.loads(request.content)["collection_names"] == [KB]
            return httpx.Response(200, json={"documents": [[text for _, text in chunks]],
                                             "metadatas": [[{"file_id": key} for key, _ in chunks]]})
        if request.url.path.startswith("/api/v1/files/") and request.url.path.endswith("/content"):
            key = request.url.path.split("/")[-2]
            assert key in state["originals"]
            return httpx.Response(200, content=state["originals"][key])
        raise AssertionError(request.url)

    settings = Settings(_env_file=None, openwebui_base_url="https://webui.example.invalid",
                        openwebui_api_key="test-key", openwebui_knowledge_ids=[KB])
    return TalkKnowledge(settings, httpx.MockTransport(handler)), state


def original_reads(state):
    return [request for request in state["calls"] if request.url.path.startswith("/api/v1/files/")]


@pytest.mark.parametrize("chunks", [(), ((DOC, "Сканер штрихкодов уже подключён в программе."),),
                                      ((OTHER, "Чужая инструкция о громкости."),)])
def test_instruction37_original_and_pixels_are_available_despite_missing_or_partial_vector_hits(chunks):
    data = instruction([VOLUME])
    knowledge, state = scanner_provider([(DOC, FILENAME, data)], chunks=chunks)
    selections = []
    knowledge.selector = lambda payload: selections.append(payload)
    approved = knowledge.search(PROBLEM, fallback_query=PROBLEM)
    assert approved.document["source"]["file_id"] == DOC
    assert approved.document["source"]["knowledge_id"] == KB
    assert VOLUME in approved.document["content"]
    assert approved.illustrations[1].data == read_docx_images(data)[0].data
    assert approved.illustrations[1].caption == VOLUME
    assert not selections and not any(request.method == "POST" for request in state["calls"])


@pytest.mark.parametrize("goal,body", [
    ("уменьшить громкость", VOLUME),
    ("приглушить звук", "Сканер штрихкодов. Настройка звука выполняется штрихкодом на листе."),
    ("настроить подсветку", "Сканер штрихкодов. Настройка подсветки по исходному листу."),
    ("изменить яркость луча", "Сканер штрихкодов. Яркость подсветки регулируется по листу."),
    ("настроить CapsLock", "Сканер штрихкодов. Caps Lock Override. Настройка Caps Lock по листу."),
    ("выполнить базовую настройку", "Сканер штрихкодов. Базовая настройка указана на листе."),
])
def test_original_fast_path_matches_requested_hardware_goal_without_fixed_filename(goal, body):
    knowledge, state = scanner_provider([(DOC, "Параметры сканера штрихкодов.md", body.encode())])
    approved = knowledge.search("Сканер ШК: " + goal)
    assert approved is not None and approved.document["content"] == body
    assert not any(request.method == "POST" for request in state["calls"])


@pytest.mark.parametrize("problem", ["Сканер ШК", "Как подключить сканер ШК в 1С?",
                                    "Уменьшить громкость", "Сканер документов: уменьшить громкость"])
def test_generic_or_other_scanner_tasks_retain_existing_retrieval(problem):
    knowledge, state = scanner_provider([(DOC, FILENAME, instruction([VOLUME]))])
    assert knowledge.search(problem) is None
    assert not original_reads(state)
    assert any(request.method == "POST" for request in state["calls"])


def test_two_equally_applicable_originals_do_not_authorize_arbitrary_first_scanner():
    knowledge, state = scanner_provider([
        (DOC, "Первый сканер штрихкодов.md", VOLUME.encode()),
        (OTHER, "Второй сканер штрихкодов.md", VOLUME.encode()),
    ])
    assert knowledge.search(PROBLEM) is None
    assert len(original_reads(state)) == 2
    assert any(request.method == "POST" for request in state["calls"])


def test_catalogue_larger_than_bound_cannot_appear_unique_after_truncation():
    rows = [(f"00000000-0000-4000-8000-{number:012d}", f"Сканер штрихкодов {number}.md", VOLUME.encode())
            for number in range(1, MAX_SCANNER_ORIGINALS + 2)]
    knowledge, state = scanner_provider(rows)
    assert knowledge.search(PROBLEM) is None
    assert not original_reads(state)
    assert any(request.method == "POST" for request in state["calls"])


def test_excluded_original_cannot_reappear_through_catalogue_or_native_metadata():
    knowledge, state = scanner_provider([(DOC, FILENAME, instruction([VOLUME]))], chunks=[(DOC, VOLUME)])
    assert knowledge.search(PROBLEM, excluded=[DOC]) is None
    assert not original_reads(state) and not any(request.method == "POST" for request in state["calls"])


def test_foreign_and_withdrawn_scanner_ids_never_authorize_original_reads():
    knowledge, state = scanner_provider([
        (DOC, "33_Сканер штрихкодов в базе АДРЕС.docx", instruction([VOLUME])),
    ], chunks=[(OTHER, VOLUME)])
    assert knowledge.search(PROBLEM) is None
    assert not original_reads(state) and not any(request.method == "POST" for request in state["calls"])


def test_revoked_scanner_original_is_service_failure_instead_of_no_knowledge():
    knowledge, _ = scanner_provider([(DOC, FILENAME, instruction([VOLUME]))], revoke_after=1)
    with pytest.raises(KnowledgeUnavailable, match="SourceMembershipChanged"):
        knowledge.search(PROBLEM)


def test_invalid_scanner_original_does_not_silently_become_missing_knowledge():
    knowledge, state = scanner_provider([(DOC, FILENAME, b"not an office document")])
    with pytest.raises(KnowledgeUnavailable) as captured:
        knowledge.search(PROBLEM)
    assert captured.value.stage == "parse_document"
    assert not any(request.method == "POST" for request in state["calls"])


@pytest.mark.parametrize("problem,body", [
    ("Сканер ШК: настроить CapsLock", "Сканер штрихкодов. Подключение в 1С."),
    ("Сканер ШК: базовая настройка", VOLUME),
    ("Сканер ШК: уменьшить громкость", "Сканер штрихкодов. Включение подсветки."),
    ("Сканер документов: уменьшить громкость", VOLUME),
    ("Сканер штрихкодов Zebra DS2278: уменьшить громкость", "Сканер штрихкодов Zebra DS2208. Громкость по листу."),
])
def test_original_must_cover_goal_type_and_voluntarily_named_model(problem, body):
    knowledge, _ = scanner_provider([(DOC, "Сканер штрихкодов CapsLock базовая настройка.md", body.encode())])
    assert knowledge.search(problem) is None


def test_original_lookup_logs_aggregate_counts_without_private_content(caplog):
    knowledge, _ = scanner_provider([(DOC, FILENAME, instruction([VOLUME]))])
    with caplog.at_level("INFO", logger="ai-support"):
        assert knowledge.search(PROBLEM) is not None
    fields = next(record.fields for record in caplog.records if record.message == "talk.knowledge_search")
    assert fields == {"stage": "scanner_originals", "outcome": "selected", "approved_files": 1,
                      "candidate_count": 1, "scanned_files": 1, "selected": True}
    assert PROBLEM not in str(fields) and FILENAME not in str(fields) and "test-key" not in str(fields)


@pytest.mark.parametrize("error,code,status", [
    (httpx.ReadTimeout("private model body"), "RequestTimedOut", None),
    (TimeoutError("private model body"), "RequestTimedOut", None),
    (httpx.ConnectError("private model body"), "RequestFailed", None),
    (httpx.HTTPStatusError("private model body", request=httpx.Request("POST", "https://webui.example.invalid"),
                           response=httpx.Response(503)), "HTTPError", 503),
])
def test_semantic_transport_failure_is_unavailability_with_safe_stage_and_no_repeat(error, code, status, caplog):
    knowledge, state = provider(chunks=[(DOC, "Неполный фрагмент справки.")])
    selections = []

    def fail(payload):
        selections.append(payload)
        raise error

    knowledge.selector = fail
    with caplog.at_level("INFO", logger="ai-support"):
        with pytest.raises(KnowledgeUnavailable) as captured:
            knowledge.search("Мобильный принтер ТСД")
    assert (captured.value.code, captured.value.stage, captured.value.http_status) == (code, "source_selection", status)
    assert len(selections) == 1
    assert not original_reads(state)
    fields = next(record.fields for record in caplog.records if record.message == "talk.knowledge_error")
    assert "private model body" not in str(fields) and "test-key" not in str(fields)
