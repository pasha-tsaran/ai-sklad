"""Historical instruction reuse requires current membership and identical source bytes."""

import hashlib

import httpx
import pytest

from app.core.config import Settings
from app.services.rag.openwebui import OpenWebUIKnowledge
from tests.test_illustrations import docx


def configured(**overrides):
    return Settings(_env_file=None, **{
        "openwebui_base_url": "https://webui.internal",
        "openwebui_api_key": "test-key",
        "openwebui_knowledge_ids": ["approved"],
        **overrides,
    })


def source_for(content="Проверьте кабель USB.", original=None, **overrides):
    return {
        "backend": "openwebui",
        "base_url": "https://webui.internal",
        "file_id": "instruction",
        "sha256": hashlib.sha256(content.encode()).hexdigest(),
        "original_sha256": hashlib.sha256(original).hexdigest() if original is not None else None,
        **overrides,
    }


def revalidation_transport(*, content="Проверьте кабель USB.", original=None, remove_on_check=None):
    calls = []
    membership_checks = 0

    def handler(request):
        nonlocal membership_checks
        calls.append(request.url.path)
        assert request.method == "GET"  # Revalidation must never invoke the LLM.
        assert request.headers["Authorization"] == "Bearer test-key"
        if request.url.path == "/api/v1/knowledge/approved":
            membership_checks += 1
            present = remove_on_check is None or membership_checks < remove_on_check
            return httpx.Response(200, json={"files": [
                {"id": "instruction", "filename": "procedure.docx" if original is not None else "procedure.md"}
            ] if present else []})
        if original is not None:
            assert request.url.path == "/api/v1/files/instruction/content"
            return httpx.Response(200, content=original)
        assert request.url.path == "/api/v1/files/instruction/data/content"
        return httpx.Response(200, json={"content": content})

    return httpx.MockTransport(handler), calls


def test_unchanged_markdown_requires_membership_before_and_after_source_fetch():
    transport, calls = revalidation_transport()
    assert OpenWebUIKnowledge(configured(), transport).is_current(source_for())
    assert calls == [
        "/api/v1/knowledge/approved",
        "/api/v1/files/instruction/data/content",
        "/api/v1/knowledge/approved",
    ]


def test_changed_markdown_is_not_a_current_historical_source():
    transport, calls = revalidation_transport(content="Новая редакция с другими шагами.")
    assert not OpenWebUIKnowledge(configured(), transport).is_current(source_for())
    assert "/api/v1/files/instruction/data/content" in calls


@pytest.mark.parametrize("remove_on_check", [1, 2])
def test_removed_membership_rejects_even_unchanged_source(remove_on_check):
    transport, calls = revalidation_transport(remove_on_check=remove_on_check)
    assert not OpenWebUIKnowledge(configured(), transport).is_current(source_for())
    assert calls.count("/api/v1/knowledge/approved") == remove_on_check
    assert ("/api/v1/files/instruction/data/content" in calls) == (remove_on_check == 2)


def test_unchanged_original_docx_is_current_without_relying_on_indexed_text():
    original = docx()
    transport, calls = revalidation_transport(original=original)
    assert OpenWebUIKnowledge(configured(), transport).is_current(source_for(original=original))
    assert "/api/v1/files/instruction/content" in calls
    assert "/api/v1/files/instruction/data/content" not in calls


def test_changed_docx_original_bytes_reject_historical_photos_and_text():
    previous = docx(caption="Кнопка Дополнительно находится сверху")
    changed = docx(caption="Новая фотография и новое пояснение")
    transport, _ = revalidation_transport(original=changed)
    assert not OpenWebUIKnowledge(configured(), transport).is_current(source_for(original=previous))


@pytest.mark.parametrize("case", ["wrong-base", "collections-disabled", "missing-file-id"])
def test_invalid_source_configuration_fails_without_network(case):
    def unexpected_request(request):
        raise AssertionError("A disabled or unrelated knowledge source must not cause a request")

    settings = configured(openwebui_knowledge_ids=[] if case == "collections-disabled" else ["approved"])
    source = source_for()
    if case == "wrong-base":
        source["base_url"] = "https://different-webui.internal"
    elif case == "missing-file-id":
        source.pop("file_id")
    assert not OpenWebUIKnowledge(settings, httpx.MockTransport(unexpected_request)).is_current(source)


def test_oversized_original_is_not_reused():
    previous = docx()
    transport, _ = revalidation_transport(original=previous + b"x" * 2000)
    provider = OpenWebUIKnowledge(configured(knowledge_max_file_bytes=1000), transport)
    assert not provider.is_current(source_for(original=previous))
