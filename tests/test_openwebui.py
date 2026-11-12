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
        assert "JSON Schema: " in payload["messages"][0]["content"]
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
        assert "JSON Schema: " not in body["messages"][0]["content"]
        expected_schema = InstructionSelection.model_json_schema()
        expected_schema["properties"]["file_id"] = {"type": "null", "default": None}
        expected_schema["properties"]["alternatives"]["maxItems"] = 0
        assert body["response_format"] == {"type": "json_schema", "json_schema": {
            "name": "instruction_selection", "schema": expected_schema,
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


def test_instruction_schema_enumerates_only_each_requests_current_source_aliases():
    from app.services.rag.talk import InstructionSelection

    first = "00000000-0000-4000-8000-000000000001"
    second = "00000000-0000-4000-8000-000000000002"
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        field = body["response_format"]["json_schema"]["schema"]["properties"]["file_id"]
        assert field["anyOf"] == [{"type": "string", "enum": ["s1"]}, {"type": "null"}]
        nested = body["response_format"]["json_schema"]["schema"]["$defs"]["InstructionAlternative"]
        assert nested["properties"]["file_id"]["enum"] == ["s1"]
        assert json.loads(body["messages"][-1]["content"])["catalogue"] == {"s1": "Описание устройства.md"}
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"result":"NO_KNOWLEDGE"}'}}]})

    model = OpenWebUILLM(settings(), httpx.MockTransport(handler))
    for file_id in (first, second):
        assert model.select_instruction({"catalogue": {file_id: "Описание устройства.md"}}).result == "NO_KNOWLEDGE"
    assert len(calls) == 2
    assert "enum" not in str(InstructionSelection.model_json_schema()["properties"]["file_id"])


def test_natural_offer_reply_uses_fast_json_and_quotes_the_employees_own_consent():
    def handler(request):
        body = json.loads(request.content)
        assert body["params"]["think"] is False
        schema = body["response_format"]["json_schema"]["schema"]
        assert schema["title"] == "InstructionOfferReply" and "file_id" not in schema["properties"]
        source_payload = json.loads(body["messages"][-1]["content"])
        assert source_payload == {"message": "Пожалуй, второй вариант подойдет", "offered": [
            {"number": 1, "filename": "Первая инструкция.md"},
            {"number": 2, "filename": "Вторая инструкция.md"},
        ]}
        assert "JSON Schema: " not in body["messages"][0]["content"]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
            "result": "ACCEPT", "choice": 2, "evidence": "второй вариант подойдет",
        })}}]})

    model = OpenWebUILLM(settings(), httpx.MockTransport(handler))
    result = model.interpret_instruction_offer("Пожалуй, второй вариант подойдет", [
        {"number": 1, "filename": "Первая инструкция.md"}, {"number": 2, "filename": "Вторая инструкция.md"},
    ])
    assert result.result == "ACCEPT" and result.choice == 2


@pytest.mark.parametrize("result_kind", ["MATCH", "READ", "RELATED"])
def test_selection_compacts_fifty_source_ids_and_returns_canonical_choices(result_kind):
    from copy import deepcopy

    ids = [f"00000000-0000-4000-8000-{number:012d}" for number in range(1, 51)]
    evidence = "Включите печать как изображение и нажмите «ОК»."
    content = "Полная инструкция. " + evidence + " Исходный UUID " + ids[-1] + ". Коды s1 и s50 — текст."
    problem = "Покажи PDF «с картинками», UUID " + ids[-1] + " не меняй."
    payload = {
        "actual_problem": problem,
        "catalogue": {file_id: f"Инструкция {number}.docx" for number, file_id in enumerate(ids, 1)},
        "candidates": {ids[0]: content, ids[-1]: content},
        "knowledge_ids": ["00000000-0000-4000-9000-000000000001"],
        "context": {
            "shown_source": {"file_id": ids[-1], "filename": "Инструкция 50.docx"},
            "verified_previous_source": {"file_id": ids[0]},
            "current_message": problem,
            "recent_messages": [{"role": "user", "content": ids[-1] + " и s50"}],
        },
    }
    unchanged = deepcopy(payload)

    def handler(request):
        body = json.loads(request.content)
        wire = json.loads(body["messages"][-1]["content"])
        assert wire["catalogue"] == {
            f"s{number}": f"Инструкция {number}.docx" for number in range(1, 51)
        }
        assert wire["candidates"] == {"s1": content, "s50": content}
        assert wire["actual_problem"] == problem
        assert wire["knowledge_ids"] == payload["knowledge_ids"]
        assert wire["context"]["shown_source"] == {"file_id": "s50", "filename": "Инструкция 50.docx"}
        assert wire["context"]["verified_previous_source"] == {"file_id": "s1"}
        assert wire["context"]["current_message"] == problem
        assert wire["context"]["recent_messages"] == payload["context"]["recent_messages"]
        schema = body["response_format"]["json_schema"]["schema"]
        aliases = [f"s{number}" for number in range(1, 51)]
        assert schema["properties"]["file_id"]["anyOf"][0]["enum"] == aliases
        assert schema["$defs"]["InstructionAlternative"]["properties"]["file_id"]["enum"] == aliases
        assert len(json.dumps(schema)) < 2_000
        if result_kind == "RELATED":
            choice = {"result": result_kind, "alternatives": [
                {"file_id": "s1", "evidence": evidence}, {"file_id": "s50", "evidence": evidence},
            ]}
        else:
            choice = {"result": result_kind, "file_id": "s50"}
            if result_kind == "MATCH":
                choice["evidence"] = evidence
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(choice)}}]})

    model = OpenWebUILLM(settings(), httpx.MockTransport(handler))
    result = model.select_instruction(payload)
    assert payload == unchanged
    assert result.result == result_kind
    if result_kind == "RELATED":
        assert [alternative.file_id for alternative in result.alternatives] == [ids[0], ids[-1]]
        assert all(alternative.evidence == evidence for alternative in result.alternatives)
    else:
        assert result.file_id == ids[-1]
        assert result.evidence == (evidence if result_kind == "MATCH" else "")


@pytest.mark.parametrize("result_kind", ["MATCH", "RELATED"])
@pytest.mark.parametrize("returned_id", ["s2", "S1", "00000000-0000-4000-8000-000000000001"])
def test_selection_rejects_unknown_aliases_and_canonical_ids_on_the_wire(result_kind, returned_id):
    file_id = "00000000-0000-4000-8000-000000000001"
    evidence = "Печатайте документ как изображение."
    choice = {"result": result_kind}
    if result_kind == "RELATED":
        choice["alternatives"] = [{"file_id": returned_id, "evidence": evidence}]
    else:
        choice.update(file_id=returned_id, evidence=evidence)
    model = OpenWebUILLM(settings(), httpx.MockTransport(lambda request: httpx.Response(200, json={
        "choices": [{"message": {"content": json.dumps(choice)}}],
    })))
    with pytest.raises(ValueError, match="UnknownInstructionSourceAlias"):
        model.select_instruction({"catalogue": {file_id: "Печать.docx"}, "candidates": {file_id: evidence}})


def test_selection_aliases_are_local_to_each_read_round_and_following_request():
    first = "00000000-0000-4000-8000-000000000001"
    second = "00000000-0000-4000-8000-000000000002"
    requests = []

    def handler(request):
        body = json.loads(request.content)
        wire = json.loads(body["messages"][-1]["content"])
        requests.append(wire)
        choice = {"result": "READ", "file_id": "s2"} if len(requests) == 1 else {
            "result": "MATCH", "file_id": "s1", "evidence": "Инструкция по печати документа.",
        }
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(choice)}}]})

    model = OpenWebUILLM(settings(), httpx.MockTransport(handler))
    read = model.select_instruction({"catalogue": {first: "Сканер.md", second: "Печать.md"},
                                     "candidates": {first: "Настройка сканера."}})
    matched = model.select_instruction({"catalogue": {second: "Печать.md", first: "Сканер.md"},
                                        "candidates": {second: "Инструкция по печати документа."}})
    assert read.result == "READ" and read.file_id == second
    assert matched.result == "MATCH" and matched.file_id == second
    assert requests[0]["catalogue"] == {"s1": "Сканер.md", "s2": "Печать.md"}
    assert requests[1]["catalogue"] == {"s1": "Печать.md", "s2": "Сканер.md"}


def test_selection_omits_source_context_outside_the_current_catalogue():
    allowed = "00000000-0000-4000-8000-000000000001"
    excluded = "00000000-0000-4000-8000-000000000002"
    context = {"shown_source": {"file_id": excluded, "filename": "Прежняя инструкция.md"},
               "verified_previous_source": {"file_id": excluded}, "current_message": "Новая проблема"}

    def handler(request):
        wire = json.loads(json.loads(request.content)["messages"][-1]["content"])
        assert wire["context"] == {"current_message": "Новая проблема"}
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"result":"NO_KNOWLEDGE"}'}}]})

    result = OpenWebUILLM(settings(), httpx.MockTransport(handler)).select_instruction({
        "catalogue": {allowed: "Текущая инструкция.md"}, "candidates": {}, "context": context,
    })
    assert result.result == "NO_KNOWLEDGE"
    assert context["shown_source"]["file_id"] == excluded


def test_selection_rejects_candidate_outside_the_catalogue_before_sending():
    def handler(request):
        pytest.fail("Invalid source scope must not reach the model")

    model = OpenWebUILLM(settings(), httpx.MockTransport(handler))
    with pytest.raises(ValueError, match="UnknownInstructionCandidate"):
        model.select_instruction({"catalogue": {}, "candidates": {"outside": "Чужой источник"}})


@pytest.mark.parametrize("value,expected", [(None, None), ("", None), ("16384", 16384)])
def test_optional_context_window_accepts_unset_compose_value_and_explicit_size(value, expected):
    configured = Settings(openwebui_num_ctx=value, _env_file=None)
    assert configured.openwebui_num_ctx == expected


@pytest.mark.parametrize("value", [2047, 131073, "invalid"])
def test_invalid_context_window_is_rejected(value):
    with pytest.raises(ValidationError):
        Settings(openwebui_num_ctx=value, _env_file=None)


@pytest.mark.parametrize("selection", [False, True])
def test_explicit_context_window_is_forwarded_to_both_service_routes(selection):
    configured = settings()
    configured.openwebui_num_ctx = 16384

    def handler(request):
        body = json.loads(request.content)
        assert body["params"]["num_ctx"] == 16384
        if selection:
            assert body["params"] == {"num_ctx": 16384, "think": False, "max_tokens": 512, "temperature": 0}
            choice = {"result": "NO_KNOWLEDGE"}
        else:
            assert body["params"] == {"num_ctx": 16384}
            choice = {"route": "SOCIAL", "reply": "Здравствуйте!"}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(choice)}}]})

    model = OpenWebUILLM(configured, httpx.MockTransport(handler))
    if selection:
        assert model.select_instruction({"catalogue": {}, "candidates": {}}).result == "NO_KNOWLEDGE"
    else:
        assert model.dialogue("Привет", {}).route == "SOCIAL"
