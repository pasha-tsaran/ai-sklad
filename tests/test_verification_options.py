"""Capability-gated thinking preserves the existing fast request contracts."""

import copy
import json

import httpx
import pytest

from app.core.config import Settings
from app.services.llm.instruction_reader import InstructionReading, SourceGoalReading
from app.services.llm.instruction_verification import (
    ILLUSTRATION_TASK, VERIFICATION_TASK, InstructionIllustrationSelection, InstructionVerification,
    prepare_illustrations, prepare_verification, resolve_illustrations, resolve_verification,
)
from app.services.llm.openwebui import OpenWebUILLM


NATIVE_SCHEMA = {"title": "PrivateRequestTest", "type": "object", "properties": {},
                 "additionalProperties": False}
MESSAGES = [{"role": "system", "content": "Return the supplied JSON schema."},
            {"role": "user", "content": "{}"}]


def settings(**changes):
    return Settings(_env_file=None, llm_backend="openwebui", openwebui_api_key="test-key",
                    openwebui_model="configured-capable-model", openwebui_verification_think=False, **changes)


def native_model(**changes):
    requests = []

    def respond(request):
        requests.append({"body": json.loads(request.content), "timeout": request.extensions["timeout"]})
        return httpx.Response(200, json={"choices": [{
            "message": {"content": "{}", "reasoning_content": "auxiliary model reasoning"}, "finish_reason": "stop",
        }]})

    return OpenWebUILLM(settings(**changes), httpx.MockTransport(respond)), requests


def source_payload():
    return {"source": {"file_id": "private-source", "filename": "Сведения об экране.md", "sha256": "a" * 64},
            "requested_goals": {"setup": "Настроить экран приложения"},
            "requested_goal_kinds": {"setup": "ACTION"},
            "context": {"current_message": "Нужно настроить главный экран приложения."},
            "candidate_evidence": {"step": {"text": "Откройте приложение и настройте главный экран.",
                                               "support_kind": "TEXT", "image_numbers": []}},
            "candidate_sections": {"section-step": {"title": "Настройка экрана", "group_id": "screen",
                                                         "kind": "TEXT", "evidence_ids": ["step"],
                                                         "image_numbers": []}}}


def initial_reading(status="SUPPORTED"):
    return InstructionReading(goals={"setup": SourceGoalReading(
        status=status, evidence_ids=["step"], delivery_mode="PROCEDURE" if status == "SUPPORTED" else None,
    )})


def verification_reply(**changes):
    return {"source_finding": "Исходный порядок полностью описывает настройку главного экрана.",
            "operation_match": "EXACT", "condition_match": "EXACT", "goal_kind": "ACTION",
            "required_material_kind": "TEXT", "material_state": "READY", "all_materials_available": True,
            "verdict": "PASS", "missing_observable_condition": "NONE", "clarification_kind": None,
            "question": "", "applicability_refs": ["e1"], "operation_refs": ["e1"], "material_refs": [],
            "condition_refs": [], "answer_refs": ["e1"], "illustration_refs": [], **changes}


def stub_verifier(*, enabled=False, reply=None):
    configured = settings().model_copy(update={"openwebui_verification_think": enabled})
    model = OpenWebUILLM(configured)
    calls = []

    def request(messages, *, selection_schema=None, **options):
        calls.append({"messages": messages, "wire": json.loads(messages[-1]["content"]),
                      "schema": selection_schema, "options": options})
        return json.dumps(verification_reply() if reply is None else reply, ensure_ascii=False)

    model._request = request
    return model, calls


def test_thinking_capability_flag_defaults_false_and_accepts_explicit_environment(monkeypatch):
    monkeypatch.delenv("OPENWEBUI_VERIFICATION_THINK", raising=False)
    assert Settings(_env_file=None).openwebui_verification_think is False
    monkeypatch.setenv("OPENWEBUI_VERIFICATION_THINK", "true")
    assert Settings(_env_file=None).openwebui_verification_think is True


@pytest.mark.parametrize("timeout,expected", [(60, 120), (180, 180)])
def test_native_thinking_request_has_fixed_budget_and_bounded_timeout(timeout, expected):
    model, requests = native_model(llm_timeout_seconds=timeout, openwebui_num_ctx=16384)
    assert model._request(MESSAGES, selection_schema=NATIVE_SCHEMA, max_output_tokens=8192, thinking=True) == "{}"
    assert len(requests) == 1
    body = requests[0]["body"]
    assert body["params"] == {"num_ctx": 16384, "think": True, "max_tokens": 8192,
                              "temperature": 0, "num_predict": 8192}
    assert body["max_tokens"] == 8192
    assert body["response_format"]["json_schema"]["schema"] == NATIVE_SCHEMA
    assert body["messages"] == MESSAGES
    assert requests[0]["timeout"] == {"connect": expected, "read": expected, "write": expected, "pool": expected}


@pytest.mark.parametrize("thinking,budget,schema", [
    (1, 8192, NATIVE_SCHEMA), (0, 512, NATIVE_SCHEMA), ("true", 8192, NATIVE_SCHEMA),
    (None, 8192, NATIVE_SCHEMA), (True, 512, NATIVE_SCHEMA), (True, 4096, NATIVE_SCHEMA),
    (True, 8192.0, NATIVE_SCHEMA), (True, None, NATIVE_SCHEMA), (True, 8192, None),
])
def test_invalid_thinking_options_fail_before_transport(thinking, budget, schema):
    model, requests = native_model()
    with pytest.raises(ValueError, match="^InvalidInstructionThinkingOptions$"):
        model._request(MESSAGES, selection_schema=schema, max_output_tokens=budget, thinking=thinking)
    assert requests == []


def test_native_fast_selection_keeps_its_existing_body_and_timeout():
    model, requests = native_model(llm_timeout_seconds=45)
    assert model._request(MESSAGES, selection_schema=NATIVE_SCHEMA, max_output_tokens=512) == "{}"
    body = requests[0]["body"]
    assert body["params"] == {"think": False, "max_tokens": 512, "temperature": 0}
    assert body["max_tokens"] == 512
    assert requests[0]["timeout"] == {"connect": 45, "read": 45, "write": 45, "pool": 45}


def test_native_plain_request_keeps_its_existing_json_object_contract():
    model, requests = native_model()
    assert model._request(MESSAGES) == "{}"
    body = requests[0]["body"]
    assert "params" not in body
    assert body["max_tokens"] == 1200
    assert body["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize("enabled,status,expected", [
    (False, "SUPPORTED", {"max_output_tokens": 2048}),
    (False, "RELATED", {"max_output_tokens": 2048}),
    (True, "RELATED", {"max_output_tokens": 2048}),
    (True, "SUPPORTED", {"max_output_tokens": 8192, "thinking": True}),
])
def test_independent_verifier_enables_thinking_only_for_supported_capable_requests(enabled, status, expected):
    model, calls = stub_verifier(enabled=enabled)
    payload = source_payload()
    original = initial_reading(status)
    assert model.verify_instruction(payload, original).goals["setup"].status == "SUPPORTED"
    assert len(calls) == 1 and calls[0]["options"] == expected
    assert calls[0]["messages"][0] == {"role": "system", "content": VERIFICATION_TASK}
    assert "goals" not in calls[0]["wire"] and "source_finding" not in calls[0]["wire"]
    assert original.goals["setup"].status == status


def test_thinking_option_preserves_independent_wire_for_different_prior_readings():
    supported_model, supported_calls = stub_verifier(enabled=True)
    related_model, related_calls = stub_verifier(enabled=True)
    supported_model.verify_instruction(source_payload(), initial_reading("SUPPORTED"))
    related_model.verify_instruction(source_payload(), initial_reading("RELATED"))
    assert supported_calls[0]["wire"] == related_calls[0]["wire"]
    assert supported_calls[0]["schema"] == related_calls[0]["schema"]


def test_clarification_stays_unverified_when_thinking_capability_enabled():
    model, calls = stub_verifier(enabled=True)
    original = InstructionReading(goals={"setup": SourceGoalReading(
        status="CLARIFY", evidence_ids=["step"], question="Какое приложение открыто?",
        clarification_kind="VISIBLE_STATE",
    )})
    assert model.verify_instruction(source_payload(), original) == original
    assert calls == []


def test_legacy_reader_callback_keeps_its_existing_identity_and_mode_contract():
    model, calls = stub_verifier(enabled=True)
    payload = source_payload()
    observed = []

    def verify(incoming, reading):
        observed.append((copy.deepcopy(incoming), reading.model_copy(deep=True)))
        return reading

    def request(messages, *, selection_schema=None, max_output_tokens=None):
        wire = json.loads(messages[-1]["content"])
        calls.append({"wire": wire, "schema": selection_schema})
        return json.dumps({"goals": {"setup": {"status": "SUPPORTED", "evidence_ids": ["e1"],
                                                "delivery_mode": "PROCEDURE", "question": "",
                                                "clarification_kind": None}}})

    model.verify_instruction = verify
    model._request = request
    result = model.read_instruction(payload)
    assert result.goals["setup"].evidence_ids == ["step"]
    assert len(calls) == 1 and len(observed) == 1
    assert observed[0][0] == payload
    assert "independently_review_goal_kind" not in calls[0]["wire"]


def illustration_payload():
    payload = source_payload()
    payload["context"]["employee_messages"] = ["На экране открыто корпоративное приложение."]
    payload["candidate_sections"]["section-step"]["text"] = (
        "Откройте приложение и настройте главный экран. Сохраните настройки после выбора нужного режима. "
        "После сохранения проверьте главный экран и сохранённое значение. "
        "Если значение изменилось, повторно откройте настройки и проверьте его до завершения работы."
    )
    payload["candidate_evidence"]["ui-picture"] = {
        "text": "Снимок главного экрана с расположением кнопки настройки.",
        "support_kind": "IMAGE", "image_numbers": [1],
    }
    payload["candidate_evidence"]["unrelated-picture"] = {
        "text": "Снимок сетевого подключения из другого порядка.",
        "support_kind": "IMAGE", "image_numbers": [2],
    }
    payload["candidate_sections"]["section-ui-picture"] = {
        "title": "Настройка экрана", "group_id": "screen", "kind": "IMAGE",
        "evidence_ids": ["ui-picture"], "image_numbers": [1],
        "text": "Снимок главного экрана с расположением кнопки настройки.",
    }
    payload["candidate_sections"]["section-unrelated-picture"] = {
        "title": "Сеть", "group_id": "network", "kind": "IMAGE",
        "evidence_ids": ["unrelated-picture"], "image_numbers": [2],
        "text": "Снимок сетевого подключения из другого порядка.",
    }
    return payload


def checked_answer(payload, **changes):
    original = initial_reading().goals["setup"]
    _, _, reverse = prepare_verification(payload, "setup", original)
    verification = InstructionVerification.model_validate(verification_reply(**changes))
    answer = resolve_verification(verification, payload, "setup", reverse, original)
    return verification, answer


def test_text_answer_picker_uses_full_selected_owner_text_context_and_only_actual_image_aliases():
    payload = illustration_payload()
    verification, answer = checked_answer(payload)
    wire, schema, reverse = prepare_illustrations(payload, "setup", verification, answer)
    encoded = json.dumps(wire, ensure_ascii=False)
    assert payload["candidate_sections"]["section-step"]["text"] in encoded
    assert payload["candidate_sections"]["section-ui-picture"]["text"] in encoded
    assert wire["context"] == payload["context"]
    assert wire["requested_goal"] == payload["requested_goals"]["setup"]
    assert set(reverse.values()) == {"ui-picture", "unrelated-picture"}
    assert schema["properties"]["illustration_refs"]["maxItems"] == 24
    assert schema["properties"]["illustration_refs"]["items"]["enum"] == list(reverse)
    selected_alias = next(alias for alias, reference in reverse.items() if reference == "ui-picture")
    result = resolve_illustrations(InstructionIllustrationSelection(illustration_refs=[selected_alias]),
                                  payload, "setup", reverse, answer)
    assert result._verified_scope["answer_refs"] == ("step", "ui-picture")
    assert answer._verified_scope["answer_refs"] == ("step",)
    assert result._verified_scope["material_refs"] == answer._verified_scope["material_refs"] == ()
    assert result._verified_scope["operation_refs"] == answer._verified_scope["operation_refs"] == ("step",)
    assert result.model_dump() == answer.model_dump()


def test_picker_does_not_duplicate_long_selected_owner_fragments_into_its_payload():
    from app.services.rag.evidence import build_evidence_registry

    payload = illustration_payload()
    long_text = " ".join("Настройте пункт " + str(number) + " в главном окне приложения."
                         for number in range(750))
    fragments = build_evidence_registry(long_text)
    texts = {"step" if index == 0 else "step-" + str(index): proof
             for index, proof in enumerate(fragments.values())}
    pictures = {reference: proof for reference, proof in payload["candidate_evidence"].items()
                if proof["support_kind"] == "IMAGE"}
    payload["candidate_evidence"] = {**texts, **pictures}
    payload["candidate_sections"]["section-step"].update(text=long_text, evidence_ids=list(texts))
    original_wire, _, _ = prepare_verification(payload, "setup", initial_reading().goals["setup"])
    assert len(json.dumps(original_wire, ensure_ascii=False)) < 50000
    verification, answer = checked_answer(payload, answer_refs=["e" + str(number) for number in range(1, 97)])
    wire, _, _ = prepare_illustrations(payload, "setup", verification, answer)
    assert len(json.dumps(wire, ensure_ascii=False)) < 50000
    selected_owner = wire["source_sections"]["section-step"]
    assert selected_owner["text"] == long_text
    assert selected_owner["answer_refs"] == wire["selected_answer_refs"]
    assert "evidence" not in selected_owner


def test_picker_empty_selection_preserves_text_answer_when_source_pictures_are_irrelevant():
    payload = illustration_payload()
    verification, answer = checked_answer(payload)
    _, _, reverse = prepare_illustrations(payload, "setup", verification, answer)
    result = resolve_illustrations(InstructionIllustrationSelection(), payload, "setup", reverse, answer)
    assert result._verified_scope == answer._verified_scope
    assert result.model_dump() == answer.model_dump()


def test_picker_is_skipped_when_original_has_no_images_or_answer_already_contains_an_image():
    payload = source_payload()
    verification, answer = checked_answer(payload)
    assert prepare_illustrations(payload, "setup", verification, answer) is None
    payload = illustration_payload()
    verification, answer = checked_answer(payload, answer_refs=["e1", "e2"])
    assert prepare_illustrations(payload, "setup", verification, answer) is None


@pytest.mark.parametrize("references", [["e1"], ["foreign-image"], ["e2", "e2"]])
def test_picker_rejects_text_foreign_and_duplicate_aliases(references):
    payload = illustration_payload()
    verification, answer = checked_answer(payload)
    _, _, reverse = prepare_illustrations(payload, "setup", verification, answer)
    selected = InstructionIllustrationSelection(illustration_refs=references)
    with pytest.raises(ValueError, match="^InvalidInstructionVerificationIllustration$"):
        resolve_illustrations(selected, payload, "setup", reverse, answer)
    assert answer._verified_scope["answer_refs"] == ("step",)


def test_picker_preserves_original_image_material_proof_gate_and_legacy_scope_behavior():
    payload = illustration_payload()
    with pytest.raises(ValueError, match="^MissingInstructionVerificationOriginalImage$"):
        checked_answer(payload, required_material_kind="ORIGINAL_IMAGE")
    verification, answer = checked_answer(payload, required_material_kind="ORIGINAL_IMAGE", material_refs=["e2"])
    assert prepare_illustrations(payload, "setup", verification, answer) is None
    assert answer._verified_scope["material_refs"] == ("ui-picture",)
    verification, answer = checked_answer(payload, operation_match=None, goal_kind=None)
    assert prepare_illustrations(payload, "setup", verification, answer) is None
    assert answer._verified_scope is None


def test_picker_enforces_combined_answer_limit_without_mutating_scope():
    payload = illustration_payload()
    for number in range(95):
        reference = "extra-" + str(number)
        payload["candidate_evidence"][reference] = {"text": "Дополнительный исходный шаг " + str(number) + ".",
                                                     "support_kind": "TEXT", "image_numbers": []}
        payload["candidate_sections"]["section-" + reference] = {
            "title": "Настройка экрана", "kind": "TEXT", "evidence_ids": [reference], "image_numbers": [],
        }
    _, _, reverse = prepare_verification(payload, "setup", initial_reading().goals["setup"])
    answer_refs = [alias for alias, reference in reverse.items()
                   if payload["candidate_evidence"][reference]["support_kind"] == "TEXT"]
    verification, answer = checked_answer(payload, answer_refs=answer_refs)
    _, _, image_reverse = prepare_illustrations(payload, "setup", verification, answer)
    assert len(answer._verified_scope["answer_refs"]) == 96
    selected = InstructionIllustrationSelection(illustration_refs=[next(iter(image_reverse))])
    with pytest.raises(ValueError, match="^InstructionVerificationAnswerTooLarge$"):
        resolve_illustrations(selected, payload, "setup", image_reverse, answer)
    assert len(answer._verified_scope["answer_refs"]) == 96


@pytest.mark.parametrize("second_valid", [True, False])
def test_picker_gets_one_bounded_repair_with_identical_source_and_known_context(second_valid):
    model, calls = stub_verifier(enabled=True)
    payload = illustration_payload()

    def request(messages, *, selection_schema=None, **options):
        wire = json.loads(messages[-1]["content"])
        calls.append({"messages": messages, "wire": wire, "schema": selection_schema, "options": options})
        if messages[0]["content"] == VERIFICATION_TASK:
            return json.dumps(verification_reply())
        assert messages[0]["content"] == ILLUSTRATION_TASK
        return json.dumps({"illustration_refs": ["e2" if second_valid and len(calls) == 3 else "e1"]})

    model._request = request
    if second_valid:
        result = model.verify_instruction(payload, initial_reading()).goals["setup"]
        assert result._verified_scope["answer_refs"] == ("step", "ui-picture")
    else:
        with pytest.raises(ValueError, match="^InvalidInstructionVerificationIllustration$"):
            model.verify_instruction(payload, initial_reading())
    assert len(calls) == 3
    assert calls[0]["options"] == {"max_output_tokens": 8192, "thinking": True}
    assert all(call["options"] == {"max_output_tokens": 512} for call in calls[1:])
    original_wire = copy.deepcopy(calls[1]["wire"])
    repaired_wire = copy.deepcopy(calls[2]["wire"])
    assert repaired_wire.pop("repair_reason").startswith("InvalidInstructionVerificationIllustration.")
    assert repaired_wire == original_wire and original_wire["context"] == payload["context"]


def test_non_delivery_verification_keeps_synthetic_alternative_fast_and_skips_picker():
    model, calls = stub_verifier(enabled=True)
    payload = illustration_payload()
    result = model.verify_instruction(payload, initial_reading(), for_delivery=False).goals["setup"]
    assert result._verified_scope["answer_refs"] == ("step",)
    assert len(calls) == 1 and calls[0]["options"] == {"max_output_tokens": 2048}
