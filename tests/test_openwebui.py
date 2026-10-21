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
        assert "tools" not in payload and "tool_ids" not in payload
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
