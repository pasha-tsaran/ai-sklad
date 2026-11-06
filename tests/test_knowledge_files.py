"""Open WebUI 0.11's real file-list envelope, with finite exact-scope traversal."""

import json
from uuid import UUID

import httpx
import pytest

from app.services.rag import knowledge_files as listing


BASE = "https://webui.example.invalid"
KB = "85cda3aa-b854-4c26-9948-12da742277ef"
OTHER_KB = "00000000-0000-4000-8000-000000000099"


def item(number, **changes):
    return {
        "id": str(UUID(int=number)), "filename": f"{number:03}_Инструкция.docx",
        "user_id": "fixture-owner", "hash": "fixture-hash", "meta": {"size": 123},
        "created_at": 1, "updated_at": 2, "user": {"id": "fixture-owner"}, "data": None,
    } | changes


def run_pages(handler, *, knowledge_id=KB, follow_redirects=False):
    requests = []

    def request(request):
        requests.append(request)
        return handler(request)

    with httpx.Client(transport=httpx.MockTransport(request), follow_redirects=follow_redirects) as client:
        return listing.list_knowledge_files(client, BASE, knowledge_id), requests


def test_real_v011_items_envelope_reads_all_pages_without_root_directory_filter_or_payload_text():
    files = [item(i) for i in range(1, 66)]

    def page(request):
        assert request.method == "GET" and not request.content
        assert request.url.path == f"/api/v1/knowledge/{KB}/files"
        assert dict(request.url.params) == {
            "page": request.url.params["page"], "limit": "30", "order_by": "name", "direction": "asc"}
        assert "directory_id" not in request.url.params
        number = int(request.url.params["page"])
        return httpx.Response(200, json={
            "items": files[(number - 1) * 30:number * 30], "total": len(files),
            "directories": [{"id": "directory-only", "knowledge_id": KB}], "breadcrumbs": [],
            "next": "https://foreign.example.invalid/all-files",  # Never follow response links.
        })

    result, requests = run_pages(page)
    assert result == [{"id": file["id"], "filename": file["filename"]} for file in files]
    assert [request.url.params["page"] for request in requests] == ["1", "2", "3"]
    assert all(request.url.host == "webui.example.invalid" for request in requests)
    assert "fixture-owner" not in json.dumps(result) and "data" not in json.dumps(result)


def test_empty_collection_is_a_successful_single_page_not_an_unavailable_service():
    result, requests = run_pages(lambda _: httpx.Response(200, json={"items": [], "total": 0}))
    assert result == [] and len(requests) == 1


@pytest.mark.parametrize("knowledge_id", [None, "", "not-a-uuid", "../other", KB + "/../other", 1])
def test_invalid_collection_id_never_makes_a_request(knowledge_id):
    with pytest.raises(listing.KnowledgeFilesError, match="InvalidKnowledgeFileId"):
        run_pages(lambda _: pytest.fail("Invalid scope must not issue HTTP requests"), knowledge_id=knowledge_id)


def test_uppercase_uuid_is_canonicalized_in_the_fixed_collection_route():
    def page(request):
        assert request.url.path == f"/api/v1/knowledge/{KB}/files"
        return httpx.Response(200, json={"items": [], "total": 0})

    assert run_pages(page, knowledge_id=KB.upper())[0] == []


@pytest.mark.parametrize("payload", [
    {"items": [], "total": 0, "knowledge_id": OTHER_KB},
    {"items": [item(1, knowledge_id=OTHER_KB)], "total": 1},
])
def test_explicit_foreign_scope_in_response_never_authorizes_items(payload):
    with pytest.raises(listing.KnowledgeFilesError, match="KnowledgeFilesScopeMismatch"):
        run_pages(lambda _: httpx.Response(200, json=payload))


def test_foreign_redirect_is_not_followed_even_if_the_client_default_allows_it():
    requests = []

    def redirect(request):
        requests.append(request)
        return httpx.Response(302, headers={"Location": "https://foreign.example.invalid/all-files"})

    with pytest.raises(httpx.HTTPStatusError) as error:
        run_pages(redirect, follow_redirects=True)
    assert error.value.response.status_code == 302
    assert len(requests) == 1 and requests[0].url.host == "webui.example.invalid"


@pytest.mark.parametrize("payload", [
    [], {"files": [item(1)]}, {"items": None, "total": 0}, {"items": [], "total": True},
    {"items": [], "total": -1}, {"items": [], "total": "0"},
    {"items": [None], "total": 1}, {"items": [item(1, filename=None)], "total": 1},
    {"items": [item(1, filename=" ")], "total": 1},
])
def test_malformed_envelope_or_item_fails_without_partial_membership(payload):
    with pytest.raises(listing.KnowledgeFilesError, match="InvalidKnowledgeFilesResponse"):
        run_pages(lambda _: httpx.Response(200, json=payload))


def test_invalid_file_uuid_is_not_used_as_a_reference():
    with pytest.raises(listing.KnowledgeFilesError, match="InvalidKnowledgeFileId"):
        run_pages(lambda _: httpx.Response(200, json={"items": [item(1, id="../secret")], "total": 1}))


@pytest.mark.parametrize("status", [401, 403, 404, 500])
def test_http_acl_and_service_failures_propagate_with_their_status_for_safe_stage_diagnostics(status):
    with pytest.raises(httpx.HTTPStatusError) as error:
        run_pages(lambda _: httpx.Response(status, text="private-body-must-not-be-logged"))
    assert error.value.response.status_code == status


@pytest.mark.parametrize("change", ["empty", "total", "duplicate"])
def test_pagination_changes_fail_without_returning_a_partial_authorized_list(change):
    calls = []

    def page(request):
        calls.append(request)
        if request.url.params["page"] == "1":
            return httpx.Response(200, json={"items": [item(i) for i in range(1, 31)], "total": 31})
        payload = {"items": [] if change == "empty" else [item(1 if change == "duplicate" else 31)],
                   "total": 32 if change == "total" else 31}
        return httpx.Response(200, json=payload)

    with pytest.raises(listing.KnowledgeFilesError, match="IncompleteKnowledgeFilesPage|ChangedDuringPagination"):
        run_pages(page)
    assert len(calls) == 2


def test_duplicate_file_within_one_page_is_not_accepted():
    with pytest.raises(listing.KnowledgeFilesError, match="ChangedDuringPagination"):
        run_pages(lambda _: httpx.Response(200, json={"items": [item(1), item(1)], "total": 2}))


def test_excessive_total_stops_after_one_request_without_a_partial_result(monkeypatch):
    calls = []

    def page(request):
        calls.append(request)
        return httpx.Response(200, json={"items": [], "total": listing.MAX_FILES + 1})

    with pytest.raises(listing.KnowledgeFilesError, match="KnowledgeFilesLimitExceeded"):
        run_pages(page)
    assert len(calls) == 1


def test_page_cap_stops_finite_traversal_without_returning_truncated_membership(monkeypatch):
    monkeypatch.setattr(listing, "MAX_PAGES", 1)
    calls = []

    def page(request):
        calls.append(request)
        return httpx.Response(200, json={"items": [item(i) for i in range(1, 31)], "total": 31})

    with pytest.raises(listing.KnowledgeFilesError, match="KnowledgeFilesPaginationLimitExceeded"):
        run_pages(page)
    assert len(calls) == 1


def test_oversized_json_page_is_rejected_without_echoing_content(monkeypatch):
    monkeypatch.setattr(listing, "MAX_PAGE_BYTES", 40)
    with pytest.raises(listing.KnowledgeFilesError) as error:
        run_pages(lambda _: httpx.Response(200, text="private-body-" * 100))
    assert str(error.value) == "KnowledgeFilesPageTooLarge"


def test_invalid_json_page_has_only_a_fixed_safe_error_code():
    with pytest.raises(listing.KnowledgeFilesError) as error:
        run_pages(lambda _: httpx.Response(200, text="private body is not JSON"))
    assert str(error.value) == "InvalidKnowledgeFilesResponse"
