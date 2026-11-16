import json

import httpx
import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.schemas.contracts import UnderstandingReply
from app.services.llm.openwebui import OpenWebUILLM


def model_for(choice, inspect=None, finish_reason="stop"):
    def handler(request):
        if inspect is not None:
            inspect(request)
        return httpx.Response(200, json={"choices": [{
            "message": {"content": json.dumps(choice)}, "finish_reason": finish_reason,
        }]})

    settings = Settings(llm_backend="openwebui", openwebui_api_key="test-key",
                        openwebui_model="local-qwen", _env_file=None)
    return OpenWebUILLM(settings, httpx.MockTransport(handler))


@pytest.mark.parametrize("message,result,evidence", [
    ("Ага, вы меня правильно поняли", "ACCEPT", "вы меня правильно поняли"),
    ("Нет, хочу настроить подсветку", "CORRECT", "хочу настроить подсветку"),
    ("Как подключить ТСД к принтеру?", "NEW_REQUEST", "Как подключить ТСД к принтеру?"),
    ("Расскажи короткую добрую шутку", "NEW_REQUEST", "Расскажи короткую добрую шутку"),
    ("...", "UNCLEAR", ""),
])
def test_understanding_reply_uses_fast_strict_json_without_instruction_or_alias_fields(message, result, evidence):
    understanding = {"question": "Нужно уменьшить громкость сканера штрихкодов?", "problem": "Сканер громко пищит"}
    calls = []

    def inspect(request):
        body = json.loads(request.content)
        calls.append(body)
        assert request.url.path == "/api/chat/completions"
        assert request.headers["Authorization"] == "Bearer test-key"
        assert body["model"] == "local-qwen"
        assert body["tools"] == [] and body["tool_ids"] == []
        assert body["features"]["web_search"] is False
        assert body["params"] == {"think": False, "max_tokens": 512, "temperature": 0}
        schema = body["response_format"]["json_schema"]["schema"]
        assert schema["title"] == "UnderstandingReply"
        assert set(schema["properties"]) == {"result", "evidence"}
        assert schema["additionalProperties"] is False
        assert schema["properties"]["result"]["enum"] == ["ACCEPT", "CORRECT", "NEW_REQUEST", "UNCLEAR"]
        payload = json.loads(body["messages"][-1]["content"])
        assert payload == {"message": message, "understanding": understanding}
        assert "JSON Schema: " not in body["messages"][0]["content"]

    model = model_for({"result": result, "evidence": evidence}, inspect)
    reply = model.interpret_understanding_reply(message, understanding)
    assert reply.result == result and reply.evidence == evidence and len(calls) == 1


@pytest.mark.parametrize("result", ["ACCEPT", "CORRECT", "NEW_REQUEST"])
@pytest.mark.parametrize("evidence", ["", "Реплика другого сотрудника"])
def test_understanding_reply_rejects_missing_or_invented_current_message_evidence(result, evidence):
    model = model_for({"result": result, "evidence": evidence})
    with pytest.raises(ValueError, match="InvalidUnderstandingEvidence"):
        model.interpret_understanding_reply("Нет, мне нужна другая настройка", "Настроить звук?")


def test_unclear_understanding_reply_has_no_evidence():
    model = model_for({"result": "UNCLEAR", "evidence": "не знаю"})
    with pytest.raises(ValueError, match="UnexpectedUnderstandingEvidence"):
        model.interpret_understanding_reply("не знаю", "Подключить сканер?")


@pytest.mark.parametrize("choice", [
    {"result": "GUIDE", "evidence": "да"},
    {"result": "ACCEPT", "evidence": "да", "instructions": ["Сбросить устройство"]},
    {"result": "ACCEPT", "evidence": "x" * 301},
    {"result": "ACCEPT", "evidence": None},
])
def test_understanding_reply_rejects_invalid_schema_and_generated_instructions(choice):
    with pytest.raises(ValidationError):
        model_for(choice).interpret_understanding_reply("да", "Подключить сканер?")


def test_truncated_understanding_response_is_rejected():
    model = model_for({"result": "ACCEPT", "evidence": "Верно"}, finish_reason="length")
    with pytest.raises(ValueError, match="IncompleteInstructionSelection"):
        model.interpret_understanding_reply("Верно", "Подключить сканер?")


def test_understanding_schema_remains_separate_from_source_choices():
    reply = UnderstandingReply.model_validate({"result": "UNCLEAR"})
    assert reply.evidence == ""
    assert set(reply.model_dump()) == {"result", "evidence"}
