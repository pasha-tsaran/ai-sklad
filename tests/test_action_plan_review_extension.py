"""Reader disagreement stays an optional review of a valid native answer."""

import copy
import json

import httpx
import pytest

from app.services.llm.instruction_verification import (
    ACTION_PLAN_REVIEW_TASK, VERIFICATION_TASK, InstructionVerification,
    prepare_action_plan_review, prepare_verification, resolve_verification,
)
from test_action_plan_review import (
    assert_initial_request, native_model, reading, source_payload, verification_reply,
)


def reader_with_main_steps():
    result = reading()
    result.goals["setup"].evidence_ids = ["open", "apply", "finish"]
    return result


def partial_text_reply(payload):
    return verification_reply(payload, ["apply", "finish", "warning", "image"], ["apply"])


def checked_answer(payload, original, reply):
    first = original.goals["setup"]
    _, _, reverse = prepare_verification(payload, "setup", first)
    verification = InstructionVerification.model_validate(reply)
    return resolve_verification(verification, payload, "setup", reverse, first)


def sequenced_model(replies):
    model, requests = native_model([])

    def respond(request):
        requests.append(json.loads(request.content))
        assert len(requests) <= len(replies), "Unexpected extra native request"
        reply = replies[len(requests) - 1]
        if isinstance(reply, Exception):
            raise reply
        content = reply if isinstance(reply, str) else json.dumps(reply, ensure_ascii=False)
        return httpx.Response(200, json={"choices": [{
            "message": {"content": content}, "finish_reason": "stop",
        }]})

    model.transport = httpx.MockTransport(respond)
    return model, requests


def test_plain_text_reader_cited_omission_gets_one_review_with_unchanged_first_request():
    payload = source_payload()
    before = copy.deepcopy(payload)
    original = reader_with_main_steps()
    partial = partial_text_reply(payload)
    complete = verification_reply(payload, ["open", "apply", "finish", "warning", "image"], ["apply"])
    model, requests = sequenced_model([partial, complete])

    result = model.verify_instruction(payload, original).goals["setup"]

    assert len(requests) == 2
    assert_initial_request(requests[0], payload, original)
    second_wire = json.loads(requests[1]["messages"][1]["content"])
    assert second_wire["action_plan_review"] == {"selected_refs": ["e3", "e5", "e6", "e8"],
                                                "omitted_step_refs": ["e2"]}
    assert requests[1]["messages"][0]["content"] == VERIFICATION_TASK + ACTION_PLAN_REVIEW_TASK
    assert requests[0]["response_format"] == requests[1]["response_format"]
    assert result._verified_scope["answer_refs"] == ("open", "apply", "finish", "warning", "image")
    assert payload == before


@pytest.mark.parametrize("case", ["uncited-owner", "different-procedure", "other-fragment-selected"])
def test_reader_disagreement_requires_an_omitted_owner_in_the_selected_procedure(case):
    payload = source_payload()
    original = reader_with_main_steps()
    reply = partial_text_reply(payload)
    if case == "uncited-owner":
        original.goals["setup"].evidence_ids = ["apply", "finish"]
    elif case == "different-procedure":
        payload["candidate_sections"]["section-open"]["group_id"] = "another-procedure"
    else:
        original.goals["setup"].evidence_ids = ["apply"]
        reply = verification_reply(payload, ["apply-tail", "finish", "image"], ["finish"])
    model, requests = sequenced_model([reply])

    result = model.verify_instruction(payload, original).goals["setup"]

    assert len(requests) == 1
    assert_initial_request(requests[0], payload, original)
    assert result.status == "SUPPORTED"


def test_existing_caption_trigger_does_not_require_the_reader_to_cite_a_main_step():
    payload = source_payload()
    original = reading()
    reply = verification_reply(payload)
    first = original.goals["setup"]
    _, _, reverse = prepare_verification(payload, "setup", first)
    verification = InstructionVerification.model_validate(reply)
    checked = resolve_verification(verification, payload, "setup", reverse, first)

    review = prepare_action_plan_review(payload, "setup", verification, checked, reverse,
                                        reader_evidence_ids=first.evidence_ids)

    assert review["omitted_step_refs"] == ["e2", "e3", "e5"]


def test_completed_reader_cited_action_can_leave_the_valid_text_plan_unchanged():
    payload = source_payload()
    payload["context"] = {"current_message": "Рабочее окно уже открыто. Нужны оставшиеся действия.",
                          "employee_messages": ["Рабочее окно уже открыто. Нужны оставшиеся действия."]}
    original = reader_with_main_steps()
    reply = partial_text_reply(payload)
    first = checked_answer(payload, original, reply)
    model, requests = sequenced_model([reply, reply])

    result = model.verify_instruction(payload, original).goals["setup"]

    assert len(requests) == 2
    assert result.model_dump() == first.model_dump()
    assert result._verified_scope == first._verified_scope


def test_a_sufficient_caption_can_cover_a_dropped_reader_cited_main_owner():
    payload = source_payload()
    payload["candidate_sections"]["section-caption"]["text"] = (
        "Иллюстрация 1. Откройте список записей и завершите настройку."
    )
    payload["candidate_evidence"]["caption"]["text"] = (
        payload["candidate_sections"]["section-caption"]["text"]
    )
    payload["candidate_sections"]["section-image"]["text"] = (
        "Откройте список записей и завершите настройку."
    )
    payload["candidate_evidence"]["image"]["text"] = payload["candidate_sections"]["section-image"]["text"]
    original = reader_with_main_steps()
    reply = verification_reply(payload, ["open", "apply", "caption", "image"], ["apply"])
    first = checked_answer(payload, original, reply)
    model, requests = sequenced_model([reply, reply])

    result = model.verify_instruction(payload, original).goals["setup"]

    assert len(requests) == 2
    assert result._verified_scope == first._verified_scope
    assert "finish" not in result._verified_scope["answer_refs"]


@pytest.mark.parametrize("failure", [
    httpx.ReadTimeout("Synthetic optional timeout"),
    httpx.ConnectError("Synthetic optional connection failure"),
    httpx.HTTPStatusError("Synthetic optional HTTP failure", request=httpx.Request("POST", "https://native.invalid"),
                          response=httpx.Response(503)),
    ValueError("IncompleteInstructionSelection"),
    "{invalid JSON",
    {"source_finding": "Synthetic incomplete native schema."},
], ids=["timeout", "connection", "http-status", "request-value-error", "invalid-json", "invalid-schema"])
def test_optional_review_failure_returns_the_first_valid_scope(failure):
    payload = source_payload()
    original = reader_with_main_steps()
    reply = partial_text_reply(payload)
    first = checked_answer(payload, original, reply)
    model, requests = sequenced_model([reply, failure])

    result = model.verify_instruction(payload, original).goals["setup"]

    assert len(requests) == 2
    assert_initial_request(requests[0], payload, original)
    assert result.model_dump() == first.model_dump()
    assert result._verified_scope == first._verified_scope


def test_optional_review_invalid_source_reference_returns_the_first_valid_scope():
    payload = source_payload()
    original = reader_with_main_steps()
    reply = partial_text_reply(payload)
    invalid = {**reply, "answer_refs": ["foreign-source-ref"]}
    first = checked_answer(payload, original, reply)
    model, requests = sequenced_model([reply, invalid])

    result = model.verify_instruction(payload, original).goals["setup"]

    assert len(requests) == 2
    assert result._verified_scope == first._verified_scope


@pytest.mark.parametrize("case", ["missing", "related", "clarify", "lookup", "original-image", "no-material",
                                 "legacy"])
def test_optional_review_cannot_replace_a_supported_action_text_contract(case):
    payload = source_payload()
    original = reader_with_main_steps()
    reply = partial_text_reply(payload)
    changes = {}
    if case == "missing":
        changes = {"operation_match": "DIFFERENT", "verdict": "MISSING", "answer_refs": []}
    elif case == "related":
        changes = {"condition_match": "DIFFERENT", "verdict": "RELATED", "answer_refs": []}
    elif case == "clarify":
        changes = {"condition_match": "UNKNOWN", "answer_refs": [], "missing_observable_condition": "VISIBLE_STATE",
                   "condition_refs": ["e2"], "question": "Рабочее окно уже открыто?",
                   "clarification_kind": "VISIBLE_STATE"}
    elif case == "lookup":
        changes = {"goal_kind": "LOOKUP"}
    elif case == "original-image":
        changes = {"required_material_kind": "ORIGINAL_IMAGE", "material_refs": ["e8"]}
    elif case == "no-material":
        changes = {"required_material_kind": "NONE"}
    elif case == "legacy":
        changes = {"operation_match": None, "goal_kind": None}
    second = {**reply, **changes}
    checked_answer(payload, original, second)
    first = checked_answer(payload, original, reply)
    model, requests = sequenced_model([reply, second])

    result = model.verify_instruction(payload, original).goals["setup"]

    assert len(requests) == 2
    assert result.model_dump() == first.model_dump()
    assert result._verified_scope == first._verified_scope


def test_existing_caption_review_also_falls_back_after_optional_timeout():
    payload = source_payload()
    original = reading()
    reply = verification_reply(payload)
    first = checked_answer(payload, original, reply)
    model, requests = sequenced_model([reply, httpx.ReadTimeout("Synthetic caption review timeout")])

    result = model.verify_instruction(payload, original).goals["setup"]

    assert len(requests) == 2
    assert result._verified_scope == first._verified_scope


def test_failed_ordinary_native_format_repair_still_raises_after_two_attempts():
    payload = source_payload()
    original = reader_with_main_steps()
    invalid = {**partial_text_reply(payload), "answer_refs": ["foreign-source-ref"]}
    model, requests = sequenced_model([invalid, "{invalid second JSON"])

    with pytest.raises(ValueError):
        model.verify_instruction(payload, original)

    assert len(requests) == 2
    second_wire = json.loads(requests[1]["messages"][1]["content"])
    assert second_wire["repair_reason"].startswith("UnknownInstructionVerificationEvidence.")
    assert "action_plan_review" not in second_wire
    assert requests[1]["messages"][0]["content"] == VERIFICATION_TASK


@pytest.mark.parametrize("failure", [httpx.ReadTimeout("Synthetic initial timeout"),
                                   ValueError("IncompleteInstructionSelection")])
def test_initial_request_failure_without_a_valid_plan_keeps_existing_behavior(failure):
    payload = source_payload()
    model, requests = sequenced_model([failure])

    with pytest.raises(type(failure)):
        model.verify_instruction(payload, reader_with_main_steps())

    assert len(requests) == 1


def test_optional_review_does_not_swallow_a_programming_error():
    payload = source_payload()
    model, requests = sequenced_model([partial_text_reply(payload), KeyError("synthetic programming error")])

    with pytest.raises(KeyError, match="synthetic programming error"):
        model.verify_instruction(payload, reader_with_main_steps())

    assert len(requests) == 2
