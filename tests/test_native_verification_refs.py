"""Native reference sets retain every distinct proof without weakening validation."""

import copy
import json

import pytest
from pydantic import ValidationError

from app.services.llm.instruction_reader import InstructionReading, SourceGoalReading
from app.services.llm.instruction_verification import (
    InstructionVerification, normalize_native_verification_refs, prepare_verification, resolve_verification,
)
from test_action_plan_review import native_model


ROLES = ("applicability_refs", "operation_refs", "material_refs", "condition_refs",
         "answer_refs", "illustration_refs")


def gallery_payload():
    rows = [
        ("title", "Исходные изображения установки", "TEXT", [], True),
        ("operation", "Оригинал показывает установку и её продолжение.", "TEXT", [], False),
        ("caption-one", "Иллюстрация 1. Начало установки.", "TEXT", [], False),
        ("picture-one-a", "Начало установки.", "IMAGE", [1], False),
        ("picture-one-b", "Первая часть пояснения.", "IMAGE", [1], False),
        ("caption-two", "Иллюстрация 2. Продолжение установки.", "TEXT", [], False),
        ("picture-two", "Продолжение установки.", "IMAGE", [2], False),
    ]
    registry, sections = {}, {}
    for reference, text, kind, images, heading in rows:
        registry[reference] = {"text": text, "support_kind": kind, "image_numbers": images}
        sections["section-" + reference] = {
            "id": "section-" + reference, "title": "Установка", "text": text,
            "kind": kind, "group_id": "installation", "is_heading": heading,
            "image_numbers": images, "evidence_ids": [reference],
        }
    return {"source": {"file_id": "synthetic-gallery", "filename": "Установка.docx", "sha256": "b" * 64},
            "requested_goals": {"gallery": "Покажите исходные изображения установки"},
            "requested_goal_kinds": {"gallery": "LOOKUP"},
            "context": {"current_message": "Покажите исходные изображения установки."},
            "candidate_evidence": registry, "candidate_sections": sections}


def gallery_reading():
    return InstructionReading(goals={"gallery": SourceGoalReading(
        status="SUPPORTED", evidence_ids=["operation"], delivery_mode="EXCERPT",
    )})


def gallery_reply(payload):
    _, _, reverse = prepare_verification(payload, "gallery", gallery_reading().goals["gallery"])
    aliases = {reference: alias for alias, reference in reverse.items()}
    images = [aliases[name] for name in ("picture-one-a", "picture-one-b", "picture-two")]
    return {
        "source_finding": "Оригинал содержит обе части установки и их исходные изображения.",
        "operation_match": "EXACT", "condition_match": "EXACT", "goal_kind": "LOOKUP",
        "required_material_kind": "ORIGINAL_IMAGE", "material_state": "READY",
        "all_materials_available": True, "verdict": "PASS", "missing_observable_condition": "NONE",
        "clarification_kind": None, "question": "", "applicability_refs": [aliases["title"]],
        "operation_refs": [aliases["operation"]], "material_refs": images,
        "condition_refs": [],
        "answer_refs": [aliases[name] for name in (
            "caption-one", "picture-one-a", "picture-one-b", "caption-two", "picture-two")],
        "illustration_refs": images,
    }


def checked(payload, reply):
    original = gallery_reading().goals["gallery"]
    _, _, reverse = prepare_verification(payload, "gallery", original)
    return resolve_verification(InstructionVerification.model_validate(reply), payload, "gallery", reverse, original)


def test_native_repeated_image_roles_preserve_both_original_images_in_one_request():
    payload = gallery_payload()
    unique = gallery_reply(payload)
    repeated = copy.deepcopy(unique)
    for role in ("material_refs", "answer_refs", "illustration_refs"):
        repeated[role] = unique[role] * 4
    model, requests = native_model([repeated])

    result = model.verify_instruction(payload, gallery_reading()).goals["gallery"]

    assert len(requests) == 1
    expected = checked(payload, unique)
    assert result.model_dump() == expected.model_dump()
    assert result._verified_scope == expected._verified_scope
    assert result._verified_scope["material_refs"] == ("picture-one-a", "picture-one-b", "picture-two")
    images = {number for ref in result._verified_scope["answer_refs"]
              for number in payload["candidate_evidence"][ref]["image_numbers"]}
    assert images == {1, 2}


def test_native_normalization_keeps_first_seen_order_in_all_six_lists_without_mutation():
    reply = gallery_reply(gallery_payload())
    for role in ROLES:
        reply[role] = ["e3", "e1", "e3", "e2", "e1"]
    parsed = InstructionVerification.model_validate_json(json.dumps(reply))

    normalized = normalize_native_verification_refs(parsed)

    assert normalized is not parsed
    for role in ROLES:
        assert getattr(normalized, role) == ["e3", "e1", "e2"]
        assert getattr(parsed, role) == reply[role]
    assert normalized.model_dump(exclude=set(ROLES)) == parsed.model_dump(exclude=set(ROLES))


@pytest.mark.parametrize("role", ROLES)
def test_unknown_native_alias_remains_invalid_after_deduplication(role):
    payload = gallery_payload()
    reply = gallery_reply(payload)
    reply[role] = ["unknown", "unknown"]
    model, requests = native_model([reply, reply])

    with pytest.raises(ValueError, match="UnknownInstructionVerificationEvidence"):
        model.verify_instruction(payload, gallery_reading())

    assert len(requests) == 2
    wire = json.loads(requests[1]["messages"][1]["content"])
    assert wire["repair_reason"].startswith("UnknownInstructionVerificationEvidence.")


@pytest.mark.parametrize("role,error", [
    ("material_refs", "MissingInstructionVerificationOriginalImage"),
    ("illustration_refs", "InvalidInstructionVerificationIllustration"),
])
def test_native_normalization_preserves_image_role_checks(role, error):
    payload = gallery_payload()
    reply = gallery_reply(payload)
    reply[role] = ["e2", "e2"]  # TEXT proof cannot serve as original image material.
    model, requests = native_model([reply, reply])

    with pytest.raises(ValueError, match=error):
        model.verify_instruction(payload, gallery_reading())

    assert len(requests) == 2


@pytest.mark.parametrize("role,maximum", [
    ("applicability_refs", 12), ("operation_refs", 12), ("material_refs", 12),
    ("condition_refs", 12), ("answer_refs", 96), ("illustration_refs", 24),
])
def test_raw_native_list_limits_are_checked_before_normalization(role, maximum, monkeypatch):
    payload = gallery_payload()
    reply = gallery_reply(payload)
    reply[role] = [reply["material_refs"][0]] * (maximum + 1)
    model, requests = native_model([reply, reply])

    def unexpected_normalization(verification):
        pytest.fail("Raw oversized list reached normalization")

    monkeypatch.setattr("app.services.llm.instruction_verification.normalize_native_verification_refs",
                        unexpected_normalization)
    with pytest.raises(ValidationError) as failure:
        model.verify_instruction(payload, gallery_reading())

    assert len(requests) == 2
    assert any(error["type"] == "too_long" and error["loc"] == (role,) for error in failure.value.errors())


def test_unique_native_result_and_first_request_remain_identical():
    payload = gallery_payload()
    reply = gallery_reply(payload)
    model, requests = native_model([reply])
    original = gallery_reading()
    expected = checked(payload, reply)

    result = model.verify_instruction(payload, original).goals["gallery"]

    assert len(requests) == 1
    wire, schema, _ = prepare_verification(payload, "gallery", original.goals["gallery"])
    from app.services.llm.instruction_verification import VERIFICATION_TASK

    assert requests[0]["messages"] == [
        {"role": "system", "content": VERIFICATION_TASK},
        {"role": "user", "content": json.dumps(wire, ensure_ascii=False)},
    ]
    assert requests[0]["response_format"]["json_schema"]["schema"] == schema
    assert result.model_dump() == expected.model_dump()
    assert result._verified_scope == expected._verified_scope


@pytest.mark.parametrize("role", ROLES)
def test_direct_resolver_still_rejects_duplicate_reference_contract(role):
    payload = gallery_payload()
    reply = gallery_reply(payload)
    reply[role] = ["e2", "e2"]

    with pytest.raises(ValueError, match="UnknownInstructionVerificationEvidence"):
        checked(payload, reply)
