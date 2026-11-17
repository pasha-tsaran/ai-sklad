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
        expected_schema["properties"].pop("covered_goals")
        expected_schema["properties"].pop("missing_goal_ids")
        expected_schema["properties"].pop("clarification_kind")
        expected_schema["properties"].pop("clarification_sources")
        expected_schema["$defs"].pop("InstructionGoalCoverage")
        expected_schema["properties"]["result"]["enum"].remove("PARTIAL")
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


@pytest.mark.parametrize("with_goals", [False, True])
@pytest.mark.parametrize("availability", [None, True, False, "false", 0])
def test_instruction_schema_disallows_another_read_only_after_explicit_budget_exhaustion(with_goals, availability):
    from app.services.rag.talk import InstructionSelection

    file_id = "00000000-0000-4000-8000-000000000001"
    payload = {"catalogue": {file_id: "Материал.md"},
               "candidates": {file_id: "Исходная процедура настройки рабочей программы."}}
    if availability is not None:
        payload["read_available"] = availability
    if with_goals:
        payload["requested_goals"] = {"g1": "Сбросить настройки"}

    def handler(request):
        body = json.loads(request.content)
        schema = body["response_format"]["json_schema"]["schema"]
        if with_goals:
            variants = {variant["properties"]["result"]["const"]: variant for variant in schema["anyOf"]}
            outcomes = set(variants)
            assert "covered_goals" in variants["MATCH"]["required"]
            assert "missing_goal_ids" not in variants["MATCH"]["properties"]
            assert "missing_goal_ids" in variants["NO_KNOWLEDGE"]["required"]
        else:
            outcomes = schema["properties"]["result"]["enum"]
            assert "covered_goals" not in schema["properties"]
        assert ("READ" in outcomes) is (availability is not False)
        assert "PARTIAL" not in outcomes
        assert body["max_tokens"] == (768 if with_goals else 512)
        choice = {"result": "NO_KNOWLEDGE"}
        if with_goals:
            choice["missing_goal_ids"] = ["g1"]
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(choice)}}]})

    result = OpenWebUILLM(settings(), httpx.MockTransport(handler)).select_instruction(payload)
    assert result.result == "NO_KNOWLEDGE"
    assert "READ" in InstructionSelection.model_json_schema()["properties"]["result"]["enum"]


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


@pytest.mark.parametrize("required", [False, True])
def test_source_clarification_contract_is_explicit_and_preserves_legacy_compactness(required):
    file_id = "00000000-0000-4000-8000-000000000001"
    evidence = "При первом запуске программы используется корпоративное подключение."
    unknown = [{"question": "Какой служебный протокол?", "answer": "Не знаю"}]
    payload = {"catalogue": {file_id: "Вход в программу.md"}, "candidates": {file_id: evidence},
               "source_clarification_required": required, "unknown_answers": unknown}

    def handler(request):
        body = json.loads(request.content)
        schema = body["response_format"]["json_schema"]["schema"]
        assert ("clarification_kind" in schema["properties"]) is required
        assert ("clarification_sources" in schema["properties"]) is required
        assert ("clarification_sources" in schema.get("required", [])) is required
        assert ("clarification_kind" in schema.get("required", [])) is required
        wire = json.loads(body["messages"][-1]["content"])
        assert wire["unknown_answers"] == unknown
        if required:
            assert schema["$defs"]["InstructionAlternative"]["properties"]["file_id"]["enum"] == ["s1"]
            kind_schema = schema["properties"]["clarification_kind"]
            assert kind_schema["anyOf"][0]["enum"] == ["USER_GOAL", "VISIBLE_STATE", "PHYSICAL_SETUP", "WORK_HISTORY"]
            choice = {"result": "CLARIFY", "question": "Что сейчас видно в окне программы?",
                      "clarification_kind": "VISIBLE_STATE",
                      "clarification_sources": [{"file_id": "s1", "evidence": evidence}]}
        else:
            assert len(json.dumps(schema)) < 2000
            choice = {"result": "CLARIFY", "question": "Что сейчас видно в окне программы?"}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(choice)}}]})

    model = OpenWebUILLM(settings(), httpx.MockTransport(handler))
    assert model.supports_source_clarification is True
    result = model.select_instruction(payload)
    assert result.result == "CLARIFY"
    assert result.clarification_kind == ("VISIBLE_STATE" if required else None)
    assert [item.file_id for item in result.clarification_sources] == ([file_id] if required else [])


@pytest.mark.parametrize("source_id", ["s2", "S1", "00000000-0000-4000-8000-000000000001"])
def test_source_clarification_cannot_smuggle_unknown_or_canonical_wire_identifiers(source_id):
    file_id = "00000000-0000-4000-8000-000000000001"
    choice = {"result": "CLARIFY", "question": "Что сейчас видно в окне программы?",
              "clarification_kind": "VISIBLE_STATE",
              "clarification_sources": [{"file_id": source_id, "evidence": "Условие применимости из оригинала."}]}
    model = OpenWebUILLM(settings(), httpx.MockTransport(lambda request: httpx.Response(200, json={
        "choices": [{"message": {"content": json.dumps(choice)}}],
    })))
    with pytest.raises(ValueError, match="UnknownInstructionSourceAlias"):
        model.select_instruction({"source_clarification_required": True,
                                  "catalogue": {file_id: "Материал.md"}, "candidates": {file_id: "Текст"}})


@pytest.mark.parametrize("goal_count", [1, 3])
def test_goal_schema_uses_complete_closed_outcomes_and_exact_match_coverage(goal_count):
    from copy import deepcopy

    from app.services.llm.openwebui import _instruction_schema_variants
    from app.services.rag.talk import InstructionSelection

    base = InstructionSelection.model_json_schema()
    payload = {"requested_goals": {f"g{number}": f"Цель {number}" for number in range(1, goal_count + 1)},
               "catalogue": {"s1": "Рабочая программа.md", "s2": "Иной вариант.md"},
               "candidates": {"s1": "Оригинал инструкции по рабочей программе."},
               "source_clarification_required": True}
    original_base, original_payload = deepcopy(base), deepcopy(payload)
    schema = _instruction_schema_variants(base, payload)
    assert base == original_base and payload == original_payload
    assert "properties" not in schema and "allOf" not in schema
    variants = {variant["properties"]["result"]["const"]: variant for variant in schema["anyOf"]}
    for variant in variants.values():
        assert variant["type"] == "object" and variant["additionalProperties"] is False
        assert "result" in variant["required"]

    matched = variants["MATCH"]
    assert set(matched["required"]) == {"result", "file_id", "covered_goals"}
    assert matched["properties"]["file_id"] == {"type": "string", "enum": ["s1"]}
    covered = matched["properties"]["covered_goals"]
    assert covered["minItems"] == goal_count and covered["maxItems"] == goal_count
    assert set(matched["properties"]) == {"result", "file_id", "covered_goals", "understanding", "new_request"}
    # The two live malformed replies cannot satisfy this branch. Empty coverage,
    # a null or omitted file, and a simultaneous missing-goal list are excluded.
    assert covered["minItems"] > 0
    assert "anyOf" not in matched["properties"]["file_id"]
    assert "missing_goal_ids" not in matched["properties"]

    absent = variants["NO_KNOWLEDGE"]
    assert absent["required"] == ["result", "missing_goal_ids"]
    assert absent["properties"]["missing_goal_ids"]["const"] == list(payload["requested_goals"])
    assert set(absent["properties"]) == {"result", "missing_goal_ids", "new_request"}
    assert variants["READ"]["properties"]["file_id"] == {"type": "string", "enum": ["s1", "s2"]}

    clarify = variants["CLARIFY"]
    assert set(clarify["required"]) == {"result", "question", "clarification_kind", "clarification_sources"}
    assert clarify["properties"]["clarification_sources"]["minItems"] == 1
    assert "file_id" not in clarify["properties"] and "covered_goals" not in clarify["properties"]
    alternative = schema["$defs"]["InstructionAlternative"]["properties"]
    assert alternative["file_id"] == {"type": "string", "enum": ["s1"]}
    assert alternative["evidence"]["minLength"] == 20
    assert ("PARTIAL" in variants) is (goal_count > 1)
    if goal_count > 1:
        partial = variants["PARTIAL"]["properties"]
        for field in ("covered_goals", "missing_goal_ids"):
            assert partial[field]["minItems"] == 1 and partial[field]["maxItems"] == goal_count - 1


def test_goal_schema_distinguishes_text_and_original_image_support():
    from app.services.llm.openwebui import _instruction_schema_variants
    from app.services.rag.talk import InstructionSelection

    schema = _instruction_schema_variants(InstructionSelection.model_json_schema(), {
        "requested_goals": {"first": "Первое действие", "second": "Второе действие"},
        "catalogue": {"s1": "Рабочая программа.md"}, "candidates": {"s1": "Оригинал"},
    })
    variants = schema["$defs"]["InstructionGoalCoverage"]["anyOf"]
    text, image = variants
    assert text["properties"]["support_kind"]["const"] == "TEXT"
    assert text["properties"]["image_numbers"]["maxItems"] == 0
    assert image["properties"]["support_kind"]["const"] == "IMAGE"
    assert image["properties"]["image_numbers"]["minItems"] == 1
    assert "image_numbers" in image["required"]
    for variant in variants:
        assert variant["properties"]["goal_id"]["enum"] == ["first", "second"]
        assert variant["properties"]["evidence"]["minLength"] == 20
        assert variant["additionalProperties"] is False


@pytest.mark.parametrize("loaded", [False, True])
def test_goal_schema_needs_loaded_sources_and_respects_bounded_read_and_question_flags(loaded):
    from app.services.llm.openwebui import _instruction_schema_variants
    from app.services.rag.talk import InstructionSelection

    payload = {"requested_goals": {"g1": "Открыть рабочую программу"},
               "catalogue": {"s1": "Вход.md"}, "candidates": {"s1": "Оригинал"} if loaded else {},
               "source_clarification_required": True}
    base = InstructionSelection.model_json_schema()
    initial = _instruction_schema_variants(base, payload)
    outcomes = {variant["properties"]["result"]["const"] for variant in initial["anyOf"]}
    assert ("MATCH" in outcomes) is loaded
    assert ("RELATED" in outcomes) is loaded
    assert ("CLARIFY" in outcomes) is loaded
    assert "READ" in outcomes
    final = _instruction_schema_variants(base, {**payload, "read_available": False, "allow_clarification": False})
    final_outcomes = {variant["properties"]["result"]["const"] for variant in final["anyOf"]}
    assert "READ" not in final_outcomes and "CLARIFY" not in final_outcomes
    assert "NO_KNOWLEDGE" in final_outcomes


def test_goal_schema_leaves_no_goals_legacy_contract_unchanged():
    from app.services.llm.openwebui import _instruction_schema_variants
    from app.services.rag.talk import InstructionSelection

    base = InstructionSelection.model_json_schema()
    assert _instruction_schema_variants(base, {"catalogue": {}, "candidates": {}}) is base


@pytest.mark.parametrize("result_kind", ["MATCH", "NO_KNOWLEDGE", "READ", "CLARIFY"])
def test_goal_outcome_response_can_omit_irrelevant_fields_and_restores_canonical_defaults(result_kind):
    file_id = "00000000-0000-4000-8000-000000000001"
    evidence = "При первом входе откройте окно корпоративной программы."

    def handler(request):
        body = json.loads(request.content)
        schema = body["response_format"]["json_schema"]["schema"]
        variants = {variant["properties"]["result"]["const"]: variant for variant in schema["anyOf"]}
        assert result_kind in variants
        wire = json.loads(body["messages"][-1]["content"])
        assert wire["evidence_references"] is True
        assert wire["candidates"]["s1"]["t1"]["text"] == evidence
        choice = {"result": result_kind}
        if result_kind == "MATCH":
            choice.update(file_id="s1", covered_goals=[{
                "goal_id": "g1", "evidence_id": "t1",
            }])
        elif result_kind == "NO_KNOWLEDGE":
            choice["missing_goal_ids"] = ["g1"]
        elif result_kind == "READ":
            choice["file_id"] = "s1"
        else:
            choice.update(question="Какое окно программы вы сейчас видите?", clarification_kind="VISIBLE_STATE",
                          clarification_sources=[{"file_id": "s1", "evidence_id": "t1"}])
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(choice)}}]})

    result = OpenWebUILLM(settings(), httpx.MockTransport(handler)).select_instruction({
        "requested_goals": {"g1": "Открыть программу в первый раз"},
        "catalogue": {file_id: "Первый вход.docx"}, "candidates": {file_id: evidence},
        "source_clarification_required": True,
    })
    assert result.result == result_kind
    assert result.evidence == "" and result.alternatives == []
    assert result.file_id == (file_id if result_kind in {"MATCH", "READ"} else None)
    assert [item.goal_id for item in result.covered_goals] == (["g1"] if result_kind == "MATCH" else [])
    assert result.missing_goal_ids == (["g1"] if result_kind == "NO_KNOWLEDGE" else [])
    assert [item.file_id for item in result.clarification_sources] == ([file_id] if result_kind == "CLARIFY" else [])


@pytest.mark.parametrize("support_kind", ["TEXT", "IMAGE"])
def test_goal_selection_uses_precomputed_source_fragments_and_restores_exact_proof(support_kind):
    from copy import deepcopy

    file_id = "00000000-0000-4000-8000-000000000001"
    text = "Откройте окно рабочей программы и выберите корпоративное подключение."
    caption = "Снимок окна. Кнопка входа находится справа от названия подключения."
    payload = {"requested_goals": {"g1": "Открыть рабочую программу"},
               "catalogue": {file_id: "Первый вход.docx"}, "candidates": {file_id: text + "\n\n" + caption},
               "candidate_image_numbers": {file_id: [4]}, "candidate_image_captions": {file_id: {4: caption}},
               "source_clarification_required": True}
    unchanged = deepcopy(payload)
    selected = {}

    def handler(request):
        body = json.loads(request.content)
        wire = json.loads(body["messages"][-1]["content"])
        assert wire["catalogue"] == {"s1": "Первый вход.docx"}
        assert "candidate_image_captions" not in wire and "candidate_image_numbers" not in wire
        registry = wire["candidates"]["s1"]
        reference_id, proof = next((key, item) for key, item in registry.items()
                                   if item["support_kind"] == support_kind)
        selected.update(proof)
        assert proof["text"] == (text if support_kind == "TEXT" else caption)
        schema = body["response_format"]["json_schema"]["schema"]
        assert schema["title"] == "InstructionReferenceSelection"
        match = next(branch for branch in schema["anyOf"] if branch["properties"]["result"]["const"] == "MATCH")
        definition = match["properties"]["covered_goals"]["items"]["$ref"].split("/")[-1]
        proof_schema = schema["$defs"][definition]["properties"]
        assert "evidence" not in proof_schema and "support_kind" not in proof_schema
        assert reference_id in proof_schema["evidence_id"]["enum"]
        choice = {"result": "MATCH", "file_id": "s1", "covered_goals": [{
            "goal_id": "g1", "evidence_id": reference_id,
        }]}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(choice)}}]})

    result = OpenWebUILLM(settings(), httpx.MockTransport(handler)).select_instruction(payload)
    assert payload == unchanged and result.file_id == file_id
    proof = result.covered_goals[0]
    assert proof.evidence == selected["text"] and proof.support_kind == support_kind
    assert proof.image_numbers == ([4] if support_kind == "IMAGE" else [])


@pytest.mark.parametrize("reference_id", ["unknown", "i7_1", "s2:t1", "Точная цитата из другого файла"])
def test_goal_selection_rejects_unknown_and_other_file_reference_ids(reference_id):
    first = "00000000-0000-4000-8000-000000000001"
    second = "00000000-0000-4000-8000-000000000002"
    first_text = "Оригинальная последовательность действий первого документа."
    second_caption = "Иллюстрация 7. Последовательность действий другого документа."
    choice = {"result": "MATCH", "file_id": "s1", "covered_goals": [{"goal_id": "g1", "evidence_id": reference_id}]}
    model = OpenWebUILLM(settings(), httpx.MockTransport(lambda request: httpx.Response(200, json={
        "choices": [{"message": {"content": json.dumps(choice)}}],
    })))
    with pytest.raises(ValueError, match="UnknownInstructionEvidenceReference"):
        model.select_instruction({"requested_goals": {"g1": "Выполнить настройку"},
                                  "catalogue": {first: "Первый.md", second: "Другой.docx"},
                                  "candidates": {first: first_text, second: second_caption},
                                  "candidate_image_captions": {second: {7: second_caption}}})


@pytest.mark.parametrize("generated", [
    {"evidence": "Модель не должна придумывать или копировать эту цитату."},
    {"support_kind": "IMAGE"},
    {"image_numbers": [7]},
])
def test_goal_wire_forbids_model_generated_proof_metadata(generated):
    file_id = "00000000-0000-4000-8000-000000000001"
    choice = {"result": "MATCH", "file_id": "s1", "covered_goals": [{
        "goal_id": "g1", "evidence_id": "t1", **generated,
    }]}
    model = OpenWebUILLM(settings(), httpx.MockTransport(lambda request: httpx.Response(200, json={
        "choices": [{"message": {"content": json.dumps(choice)}}],
    })))
    with pytest.raises(ValidationError):
        model.select_instruction({"requested_goals": {"g1": "Открыть программу"},
                                  "catalogue": {file_id: "Вход.md"},
                                  "candidates": {file_id: "Оригинальная последовательность действий программы."}})


@pytest.mark.parametrize("field", ["alternatives", "clarification_sources"])
def test_source_reference_items_resolve_each_files_own_original(field):
    ids = ["00000000-0000-4000-8000-000000000001", "00000000-0000-4000-8000-000000000002"]
    texts = ["Если окно выбора уже видно, используйте сохранённое подключение.",
             "Если программа запускается впервые, создайте корпоративное подключение."]
    choice = {"result": "RELATED" if field == "alternatives" else "CLARIFY",
              field: [{"file_id": "s1", "evidence_id": "t1"}, {"file_id": "s2", "evidence_id": "t1"}]}
    if field == "clarification_sources":
        choice.update(question="Вы открываете эту программу впервые на компьютере?", clarification_kind="WORK_HISTORY")
    model = OpenWebUILLM(settings(), httpx.MockTransport(lambda request: httpx.Response(200, json={
        "choices": [{"message": {"content": json.dumps(choice)}}],
    })))
    result = model.select_instruction({"requested_goals": {"g1": "Открыть программу"},
                                       "catalogue": dict(zip(ids, ["Первый вход.md", "Повторный вход.md"], strict=True)),
                                       "candidates": dict(zip(ids, texts, strict=True)),
                                       "source_clarification_required": True})
    assert [item.file_id for item in getattr(result, field)] == ids
    assert [item.evidence for item in getattr(result, field)] == texts


def test_reference_schema_binds_each_file_and_filters_short_context_fragments():
    from app.services.llm.instruction_selection import InstructionReferenceSelection, reference_selection_schema

    first = {"t1": {"text": "Исходный текст первого документа.", "support_kind": "TEXT", "image_numbers": []},
             "t2": {"text": "Краткое название", "support_kind": "TEXT", "image_numbers": []}}
    second = {"i5_1": {"text": "Исходная подпись программирующего штрихкода.",
                       "support_kind": "IMAGE", "image_numbers": [5]}}
    payload = {"evidence_references": True, "requested_goals": {"g1": "Первое действие", "g2": "Второе действие"},
               "candidates": {"s1": first, "s2": second, "s3": {"t1": first["t2"]}},
               "catalogue": {"s1": "Первый.md", "s2": "Другой.docx", "s3": "Краткий.md", "s4": "Не прочитан.docx"},
               "source_clarification_required": True}
    schema = reference_selection_schema(InstructionReferenceSelection.model_json_schema(), payload)
    matches = [item for item in schema["anyOf"] if item["properties"]["result"]["const"] == "MATCH"]
    assert len(matches) == 2
    for branch in matches:
        file_id = branch["properties"]["file_id"]["const"]
        reference = branch["properties"]["covered_goals"]["items"]["$ref"].split("/")[-1]
        allowed = schema["$defs"][reference]["properties"]["evidence_id"]["enum"]
        assert allowed == (["t1"] if file_id == "s1" else ["i5_1"])
        assert branch["properties"]["covered_goals"]["minItems"] == 2
        assert branch["properties"]["covered_goals"]["maxItems"] == 2
        assert branch["additionalProperties"] is False
    for file_id, allowed in (("s1", ["t1"]), ("s2", ["i5_1"])):
        fields = schema["$defs"]["AlternativeReference_" + file_id]["properties"]
        assert fields["file_id"]["const"] == file_id and fields["evidence_id"]["enum"] == allowed
    assert "AlternativeReference_s3" not in schema["$defs"]
    partials = [item for item in schema["anyOf"] if item["properties"]["result"]["const"] == "PARTIAL"]
    assert len(partials) == 2
    for branch in partials:
        assert branch["properties"]["covered_goals"]["maxItems"] == 1
        assert branch["properties"]["missing_goal_ids"]["maxItems"] == 1
    read = next(item for item in schema["anyOf"] if item["properties"]["result"]["const"] == "READ")
    assert read["properties"]["file_id"]["enum"] == ["s1", "s2", "s3", "s4"]


@pytest.mark.parametrize("captions", [{"outside": {}}, {"s1": {"1": "Подпись"}}, {"s1": {0: "Подпись"}}])
def test_caption_inventory_cannot_leave_source_or_image_scope(captions):
    model = OpenWebUILLM(settings(), httpx.MockTransport(lambda request: pytest.fail("Invalid inventory reached model")))
    with pytest.raises(ValueError, match="UnknownInstructionImageCaption"):
        model.select_instruction({"requested_goals": {"g1": "Открыть программу"},
                                  "catalogue": {"s1": "Вход.md"},
                                  "candidates": {"s1": "Оригинальная инструкция входа в программу."},
                                  "candidate_image_captions": captions})


def test_actual_reference_payload_still_has_the_original_network_budget():
    from app.services.rag.talk import MAX_SELECTION_PAYLOAD_CHARS, KnowledgeUnavailable

    model = OpenWebUILLM(settings(), httpx.MockTransport(lambda request: pytest.fail("Oversized payload reached model")))
    payload = {"requested_goals": {"g1": "Открыть программу"},
               "catalogue": {"s1": "Вход.md"},
               "candidates": {"s1": "Оригинальная последовательность действий программы. " * 900}}
    assert len(json.dumps(payload, ensure_ascii=False)) < MAX_SELECTION_PAYLOAD_CHARS
    with pytest.raises(KnowledgeUnavailable, match="SelectionPayloadTooLarge"):
        model.select_instruction(payload)
