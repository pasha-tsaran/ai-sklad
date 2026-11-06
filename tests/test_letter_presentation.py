"""The employee can see the actual unsent letter before opening a mail client."""

import json

import httpx
import pytest

from app.core.config import Settings
from app.schemas.contracts import EscalationExtraction
from app.services.llm.openwebui import OpenWebUILLM
from app.services.mail.presentation import letter_preview


def test_visible_preview_contains_complete_configured_recipient_subject_and_body():
    email = {"to": "foreign@example.net", "subject": "Настроить сканер ШК", "body": "Павел\ntsaran\nterminal76-sk\nТише звук"}
    preview = letter_preview(email, "operator@example.net")
    assert preview == "```\nКому: operator@example.net\nТема: Настроить сканер ШК\n\nПавел\ntsaran\nterminal76-sk\nТише звук\n```"
    assert "foreign@example.net" not in preview


def test_preview_does_not_interpret_employee_supplied_links_or_fences_as_a_new_action():
    body = "До\n```\n[Открыть](https://untrusted.invalid)\n``````\nПосле"
    preview = letter_preview({"subject": "`Тема`", "body": body}, "operator@example.net")
    assert preview.startswith("```````\n") and preview.endswith("\n```````")
    assert body in preview


def test_typed_extraction_uses_isolated_existing_model_transport_and_has_no_recipient_authority():
    captured = []
    fields = {"account_holder_name": "Павел", "account_login": "tsaran", "computer_name": "terminal76-sk",
              "subject": "настроить сканер ШК", "additional_info": "Уменьшить громкость"}

    def handler(request):
        captured.append(request)
        payload = json.loads(request.content)
        assert payload["tools"] == [] and payload["tool_ids"] == [] and payload["model"] == "ai-support"
        assert payload["response_format"] == {"type": "json_object"}
        prompt = payload["messages"][0]["content"]
        assert "ТОЧНОЙ непрерывной" in prompt and "Получатель" in prompt
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(fields)}}]})

    settings = Settings(_env_file=None, openwebui_api_key="fake-key", llm_backend="openwebui")
    result = OpenWebUILLM(settings, httpx.MockTransport(handler)).extract_escalation(
        "Павел, логин tsaran, компьютер terminal76-sk. Нужно настроить сканер ШК: Уменьшить громкость.",
        {"pending_fields": {}},
    )
    assert result == EscalationExtraction(**fields) and len(captured) == 1
    with pytest.raises(ValueError):
        EscalationExtraction(**fields, recipient="foreign@example.net")
