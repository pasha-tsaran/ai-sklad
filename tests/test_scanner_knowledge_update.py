"""Explicit scanner maintenance stays inside the configured knowledge collection."""

import json

import httpx
import pytest

from app.core.config import Settings
from scripts.update_scanner_knowledge import COLLECTION, RENAMES, WITHDRAWN, update_scanner_knowledge


def maintenance_provider():
    names = [WITHDRAWN, *RENAMES, "05_Acrobat.docx"]
    state = {"files": [{"id": f"00000000-0000-4000-8000-{index:012d}", "filename": name}
                       for index, name in enumerate(names, 1)], "calls": [], "fail_rename": None}

    def handler(request):
        state["calls"].append(request)
        assert request.url.host == "webui.example.invalid"
        assert request.headers["Authorization"] == "Bearer test-key"
        if request.url.path == "/api/v1/knowledge/" + COLLECTION:
            return httpx.Response(200, json={"id": COLLECTION, "files": None})
        if request.url.path == "/api/v1/knowledge/" + COLLECTION + "/files":
            return httpx.Response(200, json={"items": state["files"], "total": len(state["files"])})
        if request.url.path == "/api/v1/knowledge/" + COLLECTION + "/file/remove":
            assert request.url.params["delete_file"] == "false"
            target = json.loads(request.content)["file_id"]
            state["files"] = [row for row in state["files"] if row["id"] != target]
            return httpx.Response(200, json={"files": state["files"]})
        if request.url.path.endswith("/rename"):
            target = request.url.path.split("/")[-2]
            row = next(item for item in state["files"] if item["id"] == target)
            if row["filename"] == state["fail_rename"]:
                return httpx.Response(503, json={"detail": "temporary"})
            row["filename"] = json.loads(request.content)["filename"]
            return httpx.Response(200, json=row)
        raise AssertionError(request.url)

    settings = Settings(_env_file=None, openwebui_base_url="https://webui.example.invalid",
                        openwebui_api_key="test-key", openwebui_knowledge_ids=[COLLECTION])
    return settings, httpx.MockTransport(handler), state


def test_default_plan_is_read_only_and_contains_only_the_requested_scanner_changes():
    settings, transport, state = maintenance_provider()
    result = update_scanner_knowledge(settings, transport=transport)
    assert result["applied"] is False and len(result["changes"]) == 4
    assert all(request.method == "GET" for request in state["calls"])
    assert WITHDRAWN in {row["filename"] for row in state["files"]}


def test_apply_removes_wrong_instruction_and_renames_three_files_idempotently():
    settings, transport, state = maintenance_provider()
    original_ids = {row["filename"]: row["id"] for row in state["files"]}
    result = update_scanner_knowledge(settings, apply=True, transport=transport)
    assert result["verified"] and result["applied"]
    names = {row["filename"]: row["id"] for row in state["files"]}
    assert WITHDRAWN not in names and "05_Acrobat.docx" in names
    for old, new in RENAMES.items():
        assert names[new] == original_ids[old] and old not in names
    again = update_scanner_knowledge(settings, apply=True, transport=transport)
    assert again["verified"] and again["changes"] == []
    assert not any(request.method == "DELETE" for request in state["calls"])


def test_previously_uploaded_replacement_is_kept_instead_of_adding_duplicate_name():
    settings, transport, state = maintenance_provider()
    new_name = RENAMES["ИТ_37_Zebra DS2208 — лист настроечных штрихкодов.docx"]
    replacement = {"id": "00000000-0000-4000-8000-000000000099", "filename": new_name}
    state["files"].append(replacement)
    result = update_scanner_knowledge(settings, apply=True, transport=transport)
    assert result["verified"]
    assert [row for row in state["files"] if row["filename"] == new_name] == [replacement]


def test_partial_failure_can_resume_without_renaming_completed_files_again():
    settings, transport, state = maintenance_provider()
    state["fail_rename"] = "ИТ_37_Zebra DS2208 — лист настроечных штрихкодов.docx"
    with pytest.raises(httpx.HTTPStatusError):
        update_scanner_knowledge(settings, apply=True, transport=transport)
    assert WITHDRAWN not in {row["filename"] for row in state["files"]}
    state["fail_rename"] = None
    result = update_scanner_knowledge(settings, apply=True, transport=transport)
    assert result["verified"] and len(result["changes"]) == 2


def test_mismatched_collection_is_rejected_before_any_network_request():
    settings, transport, state = maintenance_provider()
    settings.openwebui_knowledge_ids = ["00000000-0000-4000-8000-000000000001"]
    with pytest.raises(ValueError, match="KnowledgeScopeMismatch"):
        update_scanner_knowledge(settings, apply=True, transport=transport)
    assert state["calls"] == []


def test_missing_sources_are_reported_without_claiming_all_instructions_available():
    settings, transport, state = maintenance_provider()
    state["files"] = []
    result = update_scanner_knowledge(settings, apply=True, transport=transport)
    assert result["verified"] is False
