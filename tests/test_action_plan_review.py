"""A bounded completeness review preserves unrelated native request contracts."""

import copy
import json

import httpx
import pytest

from app.core.config import Settings
from app.services.llm.instruction_reader import InstructionReading, SourceGoalReading
from app.services.llm.instruction_verification import (
    ACTION_PLAN_REVIEW_TASK, ILLUSTRATION_TASK, VERIFICATION_TASK,
    InstructionVerification, prepare_action_plan_review, prepare_verification, resolve_verification,
)
from app.services.llm.openwebui import OpenWebUILLM


def source_payload():
    paragraphs = [
        ("heading", "Настройка рабочего окна", "TEXT", [], True),
        ("open", "1. Откройте рабочее окно.", "TEXT", [], False),
        ("apply", "2. Примените изменение. Проверьте сохранение изменения.", "TEXT", [], False),
        ("finish", "3. Откройте список записей и завершите настройку.", "TEXT", [], False),
        ("warning", "Сохраните исходные параметры до изменения.", "WARNING", [], False),
        ("caption", "Иллюстрация 1. Кнопка изменения находится в рабочем окне.", "TEXT", [], False),
        ("image", "Кнопка изменения находится в рабочем окне.", "IMAGE", [1], False),
    ]
    evidence, sections = {}, {}
    for reference, text, kind, images, heading in paragraphs:
        fragments = [text]
        if reference == "apply":
            fragments = ["2. Примените изменение.", "Проверьте сохранение изменения."]
        references = []
        for number, fragment in enumerate(fragments):
            proof_id = reference if number == 0 else reference + "-tail"
            evidence[proof_id] = {"text": fragment, "support_kind": "IMAGE" if images else "TEXT",
                                  "image_numbers": images}
            references.append(proof_id)
        sections["section-" + reference] = {
            "id": "section-" + reference, "title": "Рабочее окно", "text": text,
            "kind": kind, "group_id": "window", "image_numbers": images,
            "is_heading": heading, "evidence_ids": references,
        }
    return {"source": {"file_id": "synthetic-window", "filename": "Рабочее окно.md", "sha256": "a" * 64},
            "requested_goals": {"setup": "Полностью настроить рабочее окно"},
            "requested_goal_kinds": {"setup": "ACTION"},
            "context": {"current_message": "Полностью настройте рабочее окно.",
                        "employee_messages": ["Полностью настройте рабочее окно."]},
            "candidate_evidence": evidence, "candidate_sections": sections}


def reading(status="SUPPORTED", mode="PROCEDURE"):
    return InstructionReading(goals={"setup": SourceGoalReading(
        status=status, evidence_ids=["caption"], delivery_mode=mode if status == "SUPPORTED" else None,
    )})


def verification_reply(payload, references=None, operations=None, **changes):
    _, _, reverse = prepare_verification(payload, "setup", reading().goals["setup"])
    aliases = {reference: alias for alias, reference in reverse.items()}
    references = ["caption", "image"] if references is None else references
    operations = ["caption"] if operations is None else operations
    return {"source_finding": "Исходный порядок подтверждает настройку рабочего окна.",
            "operation_match": "EXACT", "condition_match": "EXACT", "goal_kind": "ACTION",
            "required_material_kind": "TEXT", "material_state": "READY", "all_materials_available": True,
            "verdict": "PASS", "missing_observable_condition": "NONE", "clarification_kind": None,
            "question": "", "applicability_refs": [aliases["heading"]],
            "operation_refs": [aliases[reference] for reference in operations], "material_refs": [],
            "condition_refs": [], "answer_refs": [aliases[reference] for reference in references],
            "illustration_refs": [], **changes}


def native_model(replies):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        assert len(requests) <= len(replies), "Unexpected extra native request"
        return httpx.Response(200, json={"choices": [{
            "message": {"content": json.dumps(replies[len(requests) - 1], ensure_ascii=False)},
            "finish_reason": "stop",
        }]})

    settings = Settings(_env_file=None, llm_backend="openwebui", openwebui_api_key="synthetic-key",
                        openwebui_base_url="https://native.invalid", openwebui_model="synthetic-model",
                        openwebui_verification_think=False)
    return OpenWebUILLM(settings, httpx.MockTransport(respond)), requests


def assert_initial_request(body, payload, original):
    wire, schema, _ = prepare_verification(payload, "setup", original.goals["setup"])
    assert body["messages"] == [
        {"role": "system", "content": VERIFICATION_TASK},
        {"role": "user", "content": json.dumps(wire, ensure_ascii=False)},
    ]
    assert body["response_format"]["json_schema"]["schema"] == schema
    assert body["params"] == {"think": False, "max_tokens": 2048, "temperature": 0}
    assert body["max_tokens"] == 2048
    assert "action_plan_review" not in wire


def checked_review(payload, reply):
    original = reading().goals["setup"]
    _, _, reverse = prepare_verification(payload, "setup", original)
    verification = InstructionVerification.model_validate(reply)
    checked = resolve_verification(verification, payload, "setup", reverse, original)
    return prepare_action_plan_review(payload, "setup", verification, checked, reverse)


def test_caption_plan_gets_one_review_and_can_add_the_missing_actions():
    payload = source_payload()
    before = copy.deepcopy(payload)
    original = reading()
    caption_plan = verification_reply(payload)
    complete_plan = verification_reply(payload, ["open", "apply", "finish", "warning", "image"])
    model, requests = native_model([caption_plan, complete_plan])
    result = model.verify_instruction(payload, original).goals["setup"]

    assert len(requests) == 2
    assert_initial_request(requests[0], payload, original)
    expected_second = copy.deepcopy(requests[0])
    first_wire = json.loads(requests[0]["messages"][1]["content"])
    review = checked_review(payload, caption_plan)
    assert review == {"selected_refs": caption_plan["answer_refs"],
                      "omitted_step_refs": ["e2", "e3", "e5"]}
    assert set(review) == {"selected_refs", "omitted_step_refs"}
    first_wire["action_plan_review"] = review
    expected_second["messages"] = [
        {"role": "system", "content": VERIFICATION_TASK + ACTION_PLAN_REVIEW_TASK},
        {"role": "user", "content": json.dumps(first_wire, ensure_ascii=False)},
    ]
    assert requests[1] == expected_second
    assert result._verified_scope["answer_refs"] == ("open", "apply", "finish", "warning", "image")
    assert result._verified_scope["material_refs"] == ()
    assert result._verified_scope["source"] == payload["source"]
    assert payload == before
    assert original.goals["setup"]._verified_scope is None


@pytest.mark.parametrize("with_warning", [True, False])
def test_complete_text_plans_keep_the_existing_request_and_illustration_path(with_warning):
    payload = source_payload()
    references = ["open", "apply", "apply-tail", "finish"]
    if with_warning:
        references.append("warning")
    complete_plan = verification_reply(payload, references, ["heading"])
    model, requests = native_model([complete_plan, {"illustration_refs": ["e8"]}])
    original = reading()
    result = model.verify_instruction(payload, original).goals["setup"]

    assert len(requests) == 2
    assert_initial_request(requests[0], payload, original)
    assert requests[1]["messages"][0] == {"role": "system", "content": ILLUSTRATION_TASK}
    assert requests[1]["max_tokens"] == 512
    assert all("action_plan_review" not in json.loads(body["messages"][1]["content"]) for body in requests)
    assert result._verified_scope["answer_refs"] == (*references, "image")


def test_completed_numbered_steps_allow_an_unchanged_caption_answer_after_review():
    payload = source_payload()
    payload["context"] = {"current_message": "Все пронумерованные действия уже выполнены. Покажите кнопку.",
                          "employee_messages": ["Все пронумерованные действия уже выполнены. Покажите кнопку."]}
    caption_plan = verification_reply(payload)
    model, requests = native_model([caption_plan, caption_plan])
    result = model.verify_instruction(payload, reading()).goals["setup"]
    assert len(requests) == 2
    assert result._verified_scope["answer_refs"] == ("caption", "image")
    assert result._verified_scope["operation_refs"] == ("caption",)
    assert result._verified_scope["material_refs"] == ()
    assert json.loads(requests[1]["messages"][1]["content"])["context"] == payload["context"]


@pytest.mark.parametrize("case", ["other-group", "no-numbered-steps", "ambiguous-image", "headings-only"])
def test_unrelated_or_absent_primary_steps_do_not_trigger_review(case):
    payload = source_payload()
    for reference in ["open", "apply", "finish"]:
        section = payload["candidate_sections"]["section-" + reference]
        if case == "other-group":
            section["group_id"] = "other-window"
        elif case == "no-numbered-steps":
            section["text"] = section["text"][3:]
            proof = payload["candidate_evidence"][reference]
            proof["text"] = proof["text"][3:]
        elif case == "headings-only":
            section["is_heading"] = True
    if case == "ambiguous-image":
        payload["candidate_sections"]["section-image"]["scope_ambiguous"] = True
    reply = verification_reply(payload)
    model, requests = native_model([reply])
    result = model.verify_instruction(payload, reading()).goals["setup"]
    assert len(requests) == 1
    assert result._verified_scope["answer_refs"] == ("caption", "image")


@pytest.mark.parametrize("case", ["lookup", "original-image", "legacy", "non-delivery", "related-initial",
                                 "related-result"])
def test_existing_non_action_or_non_delivery_contracts_skip_review(case):
    payload = source_payload()
    original = reading("RELATED" if case == "related-initial" else "SUPPORTED")
    changes = {}
    if case == "lookup":
        changes["goal_kind"] = "LOOKUP"
    elif case == "original-image":
        changes.update(required_material_kind="ORIGINAL_IMAGE", material_refs=["e8"])
    elif case == "legacy":
        changes.update(operation_match=None, goal_kind=None)
    elif case == "related-result":
        changes.update(condition_match="DIFFERENT", verdict="RELATED", answer_refs=[])
    reply = verification_reply(payload, **changes)
    model, requests = native_model([reply])
    result = model.verify_instruction(payload, original, for_delivery=case != "non-delivery").goals["setup"]
    assert len(requests) == 1
    assert_initial_request(requests[0], payload, original)
    assert result.status == ("RELATED" if case == "related-result" else "SUPPORTED")


def test_an_operation_reference_to_a_primary_step_prevents_caption_review():
    payload = source_payload()
    reply = verification_reply(payload, operations=["open"])
    model, requests = native_model([reply])
    result = model.verify_instruction(payload, reading()).goals["setup"]
    assert len(requests) == 1
    assert result._verified_scope["operation_refs"] == ("open",)


def test_selected_heading_does_not_hide_a_caption_plan_and_image_kind_can_be_warning():
    payload = source_payload()
    payload["candidate_sections"]["section-image"]["kind"] = "WARNING"
    reply = verification_reply(payload, operations=["heading"])
    review = checked_review(payload, reply)
    assert review["omitted_step_refs"] == ["e2", "e3", "e5"]
    assert review["selected_refs"] == ["e1", "e7", "e8"]


def test_format_repair_already_uses_the_two_attempt_budget():
    payload = source_payload()
    valid = verification_reply(payload)
    invalid = {**valid, "answer_refs": ["foreign-ref"]}
    model, requests = native_model([invalid, valid])
    result = model.verify_instruction(payload, reading()).goals["setup"]
    assert len(requests) == 2
    second_wire = json.loads(requests[1]["messages"][1]["content"])
    assert second_wire["repair_reason"].startswith("UnknownInstructionVerificationEvidence.")
    assert "action_plan_review" not in second_wire
    assert requests[1]["messages"][0] == {"role": "system", "content": VERIFICATION_TASK}
    assert result._verified_scope["answer_refs"] == ("caption", "image")


def test_optional_review_keeps_the_first_valid_plan_when_its_metadata_exceeds_the_budget(monkeypatch):
    from app.services.rag import talk

    payload = source_payload()
    original = reading()
    wire, _, _ = prepare_verification(payload, "setup", original.goals["setup"])
    budget = len(json.dumps(wire, ensure_ascii=False)) + 1
    monkeypatch.setattr(talk, "MAX_SELECTION_PAYLOAD_CHARS", budget)
    caption_plan = verification_reply(payload)
    review_wire = {**wire, "action_plan_review": checked_review(payload, caption_plan)}
    assert len(json.dumps(review_wire, ensure_ascii=False)) > budget
    model, requests = native_model([caption_plan])

    result = model.verify_instruction(payload, original).goals["setup"]

    assert len(requests) == 1
    assert_initial_request(requests[0], payload, original)
    assert result._verified_scope["answer_refs"] == ("caption", "image")
    assert result._verified_scope["operation_refs"] == ("caption",)
    assert result._verified_scope["material_refs"] == ()
