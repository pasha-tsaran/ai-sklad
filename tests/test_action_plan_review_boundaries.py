"""Optional completeness reviews stay fast and preserve the selected procedures."""

import copy
import json

import pytest

from app.services.llm.instruction_verification import (
    ACTION_PLAN_REVIEW_TASK, VERIFICATION_TASK, prepare_verification,
)
from test_action_plan_review import reading, source_payload, verification_reply
from test_action_plan_review_extension import checked_answer, sequenced_model


def with_section(payload, reference, group, *, kind="TEXT", images=(), **metadata):
    text = "Дополнительный исходный раздел " + reference + "."
    payload["candidate_evidence"][reference] = {
        "text": text, "support_kind": "IMAGE" if images else "TEXT", "image_numbers": list(images),
    }
    payload["candidate_sections"]["section-" + reference] = {
        "id": "section-" + reference, "title": "Дополнительный раздел", "text": text,
        "group_id": group, "kind": kind, "image_numbers": list(images), "is_heading": False,
        "evidence_ids": [reference], **metadata,
    }


def wire_alias(payload, reference):
    _, _, reverse = prepare_verification(payload, "setup", reading().goals["setup"])
    return next(alias for alias, canonical in reverse.items() if canonical == reference)


def configured_model(replies, enabled=True):
    model, requests = sequenced_model(replies)
    model.settings = model.settings.model_copy(update={"openwebui_verification_think": enabled})
    return model, requests


@pytest.mark.parametrize("enabled", [False, True])
def test_optional_quality_request_is_fast_while_primary_keeps_its_existing_options(enabled):
    payload = source_payload()
    first = verification_reply(payload)
    complete = verification_reply(payload, ["open", "apply", "finish", "warning", "image"])
    model, requests = configured_model([first, complete], enabled)

    result = model.verify_instruction(payload, reading()).goals["setup"]

    assert len(requests) == 2
    initial_budget = 8192 if enabled else 2048
    assert requests[0]["params"]["think"] is enabled
    assert requests[0]["params"]["max_tokens"] == requests[0]["max_tokens"] == initial_budget
    assert requests[1]["params"] == {"think": False, "max_tokens": 2048, "temperature": 0}
    assert requests[1]["max_tokens"] == 2048
    assert requests[0]["messages"][0]["content"] == VERIFICATION_TASK
    assert requests[1]["messages"][0]["content"] == VERIFICATION_TASK + ACTION_PLAN_REVIEW_TASK
    assert requests[0]["response_format"] == requests[1]["response_format"]
    expected_wire, expected_schema, _ = prepare_verification(payload, "setup", reading().goals["setup"])
    assert json.loads(requests[0]["messages"][1]["content"]) == expected_wire
    assert requests[0]["response_format"]["json_schema"]["schema"] == expected_schema
    assert result._verified_scope["answer_refs"] == ("open", "apply", "finish", "warning", "image")


def test_ordinary_format_repair_keeps_thinking_and_does_not_add_a_quality_request():
    payload = source_payload()
    invalid = {**verification_reply(payload), "answer_refs": ["unknown"]}
    complete = verification_reply(payload, ["open", "apply", "finish", "warning", "image"])
    model, requests = configured_model([invalid, complete])

    result = model.verify_instruction(payload, reading()).goals["setup"]

    assert len(requests) == 2
    assert all(request["params"]["think"] is True and request["max_tokens"] == 8192 for request in requests)
    assert all(request["messages"][0]["content"] == VERIFICATION_TASK for request in requests)
    assert json.loads(requests[1]["messages"][1]["content"])["repair_reason"].startswith(
        "UnknownInstructionVerificationEvidence.")
    assert result.status == "SUPPORTED"


def test_for_delivery_false_keeps_its_existing_fast_single_request():
    payload = source_payload()
    model, requests = configured_model([verification_reply(payload)])

    model.verify_instruction(payload, reading(), for_delivery=False)

    assert len(requests) == 1
    assert requests[0]["params"] == {"think": False, "max_tokens": 2048, "temperature": 0}


@pytest.mark.parametrize("role", ["answer_refs", "operation_refs", "material_refs", "illustration_refs"])
def test_valid_shaped_review_cannot_introduce_another_procedure(role):
    payload = source_payload()
    with_section(payload, "other", "another-procedure",
                 kind="IMAGE" if role == "illustration_refs" else "TEXT",
                 images=[2] if role == "illustration_refs" else [])
    first_reply = verification_reply(payload)
    second = copy.deepcopy(first_reply)
    second[role].append(wire_alias(payload, "other"))
    original = reading()
    baseline = checked_answer(payload, original, first_reply)
    # It passes the unchanged schema/proof resolver; the quality boundary adds the restriction.
    assert checked_answer(payload, original, second).status == "SUPPORTED"
    model, requests = configured_model([first_reply, second])

    result = model.verify_instruction(payload, original).goals["setup"]

    assert len(requests) == 2
    assert result.model_dump() == baseline.model_dump()
    assert result._verified_scope == baseline._verified_scope


def test_review_cannot_switch_to_a_competing_method_under_the_same_parent():
    payload = source_payload()
    for section in payload["candidate_sections"].values():
        section.update(group_id="method-one", exclusive_set_id="choice",
                       exclusive_parent_group_id="shared-parent")
    with_section(payload, "other-method", "method-two", exclusive_set_id="choice",
                 exclusive_parent_group_id="shared-parent")
    with_section(payload, "shared", "shared-parent")
    first_reply = verification_reply(payload)
    second = verification_reply(payload, ["other-method", "image"], ["other-method"])
    baseline = checked_answer(payload, reading(), first_reply)
    model, requests = configured_model([first_reply, second])

    result = model.verify_instruction(payload, reading()).goals["setup"]

    assert len(requests) == 2
    assert result._verified_scope == baseline._verified_scope


@pytest.mark.parametrize("dependency", ["parent", "condition", "exclusive-choice", "required-warning"])
def test_source_declared_required_context_can_be_added_without_leaving_the_plan(dependency):
    payload = source_payload()
    with_section(payload, "required", "source-context",
                 kind="WARNING" if dependency == "required-warning" else "TEXT")
    for name in ("caption", "image"):
        section = payload["candidate_sections"]["section-" + name]
        if dependency == "parent":
            section["parent_group_id"] = "source-context"
        elif dependency == "condition":
            section["condition_section_id"] = "section-required"
        elif dependency == "exclusive-choice":
            section["exclusive_choice_section_id"] = "section-required"
    first_reply = verification_reply(payload)
    second = verification_reply(payload, ["open", "apply", "finish", "warning", "image", "required"])
    expected = checked_answer(payload, reading(), second)
    model, requests = configured_model([first_reply, second])

    result = model.verify_instruction(payload, reading()).goals["setup"]

    assert len(requests) == 2
    assert result._verified_scope == expected._verified_scope
    assert "required" in result._verified_scope["answer_refs"]


def test_scoped_warning_for_a_different_procedure_cannot_be_added():
    payload = source_payload()
    with_section(payload, "other-warning", "another-procedure", kind="WARNING",
                 guard_scope_group_ids=["another-procedure"])
    first_reply = verification_reply(payload)
    second = verification_reply(payload, ["caption", "image", "other-warning"])
    baseline = checked_answer(payload, reading(), first_reply)
    model, requests = configured_model([first_reply, second])

    result = model.verify_instruction(payload, reading()).goals["setup"]

    assert len(requests) == 2
    assert result._verified_scope == baseline._verified_scope


def test_explicitly_completed_steps_can_keep_the_same_valid_review_plan():
    payload = source_payload()
    payload["context"] = {"current_message": "Действия выполнены, результат проверен. Нужна подпись кнопки.",
                          "employee_messages": ["Действия выполнены, результат проверен. Нужна подпись кнопки."]}
    first_reply = verification_reply(payload)
    baseline = checked_answer(payload, reading(), first_reply)
    model, requests = configured_model([first_reply, first_reply])

    result = model.verify_instruction(payload, reading()).goals["setup"]

    assert len(requests) == 2
    assert requests[1]["params"]["think"] is False
    assert result._verified_scope == baseline._verified_scope
