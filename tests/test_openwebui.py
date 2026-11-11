import json

import httpx
import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.services.llm.openwebui import OpenWebUILLM


def settings():
    return Settings(
        llm_backend="openwebui", openwebui_api_key="test-key", openwebui_model="local-qwen", _env_file=None
    )


def test_real_adapter_http_contract():
    def handler(request):
        import json

        assert request.url.path == "/api/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        assert payload["model"] == "local-qwen"
        assert payload["stream"] is False
        assert payload["tools"] == [] and payload["tool_ids"] == []
        assert payload["features"]["web_search"] is False
        assert "params" not in payload
        assert payload["max_tokens"] == 1200
        assert payload["response_format"] == {"type": "json_object"}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "intent": "REPORT_PROBLEM",
                                    "problem_type": "printer_not_printing",
                                    "equipment_model": None,
                                    "search_query": "принтер не печатает",
                                }
                            )
                        }
                    }
                ]
            },
        )

    model = OpenWebUILLM(settings(), httpx.MockTransport(handler))
    assert model.analyze("принтер не печатает").problem_type == "printer_not_printing"


def test_real_adapter_rejects_invalid_json():
    model = OpenWebUILLM(
        settings(),
        httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={"choices": [{"message": {"content": "Не JSON"}}]},
            )
        ),
    )
    with pytest.raises(ValidationError):
        model.analyze("test")


def test_instruction_selection_has_a_short_structured_response_with_thinking_disabled(monkeypatch):
    from app.services.rag.talk import InstructionSelection

    diagnostics = []
    requests = []
    monkeypatch.setattr("app.services.llm.openwebui.event", lambda name, **fields: diagnostics.append((name, fields)))

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        assert body["model"] == "local-qwen"
        assert body["tools"] == [] and body["tool_ids"] == []
        assert body["features"]["web_search"] is False
        assert body["max_tokens"] == 512
        assert body["params"] == {"think": False, "max_tokens": 512, "temperature": 0}
        assert body["response_format"] == {"type": "json_schema", "json_schema": {
            "name": "instruction_selection", "schema": InstructionSelection.model_json_schema(),
        }}
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"result":"NO_KNOWLEDGE"}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 111, "completion_tokens": 12},
        })

    model = OpenWebUILLM(settings(), httpx.MockTransport(handler))
    result = model.select_instruction({"actual_problem": "private-example", "catalogue": {}, "candidates": {}})
    assert result.result == "NO_KNOWLEDGE" and len(requests) == 1
    name, fields = diagnostics[0]
    assert name == "llm.instruction_response"
    assert fields["finish_reason"] == "stop"
    assert fields["prompt_tokens"] == 111 and fields["completion_tokens"] == 12
    assert "private-example" not in json.dumps(diagnostics)
    assert "test-key" not in json.dumps(diagnostics)


def test_truncated_instruction_selection_is_rejected_even_with_a_valid_json_prefix():
    model = OpenWebUILLM(settings(), httpx.MockTransport(lambda request: httpx.Response(200, json={
        "choices": [{"message": {"content": '{"result":"NO_KNOWLEDGE"}'}, "finish_reason": "length"}],
    })))
    with pytest.raises(ValueError, match="IncompleteInstructionSelection"):
        model.select_instruction({"actual_problem": "example", "catalogue": {}, "candidates": {}})


def test_response_diagnostics_failure_does_not_discard_valid_selection(monkeypatch):
    def fail_log(*args, **kwargs):
        raise RuntimeError("unavailable logger")

    monkeypatch.setattr("app.services.llm.openwebui.event", fail_log)
    model = OpenWebUILLM(settings(), httpx.MockTransport(lambda request: httpx.Response(200, json={
        "choices": [{"message": {"content": '{"result":"NO_KNOWLEDGE"}'}, "finish_reason": "stop"}],
    })))
    assert model.select_instruction({"catalogue": {}, "candidates": {}}).result == "NO_KNOWLEDGE"
