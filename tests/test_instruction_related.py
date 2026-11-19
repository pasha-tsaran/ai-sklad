"""Nearby instructions need matching purpose and a separately verified actual goal."""

import copy

import pytest
from pydantic import ValidationError

from app.services.llm.instruction_reader import InstructionReading, SourceGoalReading
from app.services.llm.instruction_related import (
    RELATED_TASK, RelatedInstruction, RelatedInstructionSelection, prepare_related, related_source_scopes, resolve_related,
)
from tests.test_instruction_verification import model_verifier, payload, verdict


def proposal(**changes):
    return {"original_goal_id": "setup", "scope_ref": "scope1",
            "evidence_ids": ["e3"], "purpose_comparison": "Описана та же операция с другим условием подключения.",
            "relation_kind": "ALTERNATE_CONDITIONS", "status": "RELATED", **changes}


def test_related_wire_preserves_full_original_and_closed_original_goal_ids():
    data = payload()
    data["context"] = {"current_message": "Нужно подключить по Bluetooth, по Wi-Fi не подключали."}
    wire, schema, reverse = prepare_related(data)
    assert wire["context"] == data["context"]
    assert schema["properties"]["original_goal_id"]["enum"] == ["size", "setup", None]
    assert schema["additionalProperties"] is False
    assert schema["properties"]["evidence_ids"]["items"]["enum"] == list(reverse)
    assert schema["properties"]["evidence_ids"]["uniqueItems"] is True
    assert schema["properties"]["scope_ref"]["enum"] == ["scope1", None]
    assert "source_goal" not in schema["properties"]
    assert wire["approved_original"]["sections"][-1]["images"] == [3]
    assert "source_goal" not in wire


@pytest.mark.parametrize("changes", [
    {"original_goal_id": "foreign-goal"}, {"original_goal_id": None}, {"scope_ref": "foreign-scope"},
    {"evidence_ids": []}, {"evidence_ids": ["foreign-ref"]},
    {"evidence_ids": ["e1", "e1"]},
])
def test_bad_relationship_or_noncanonical_source_reference_is_rejected(changes):
    data = payload()
    _, _, reverse = prepare_related(data)
    with pytest.raises(ValueError):
        resolve_related(RelatedInstructionSelection.model_validate(proposal(**changes)), data, reverse)


@pytest.mark.parametrize("changes", [{"relation_kind": "DIFFERENT_OPERATION"}, {"relation_kind": "UNPROVEN"},
                                      {"status": "UNRELATED"}])
def test_explicit_denial_discards_unused_explanations_without_turning_it_into_service_error(changes):
    data = payload()
    _, _, reverse = prepare_related(data)
    result = resolve_related(RelatedInstructionSelection.model_validate(proposal(**changes)), data, reverse)
    assert result.status == "UNRELATED" and result.source_goal == "" and result.evidence_ids == []


@pytest.mark.parametrize("changes", [
    {"image_numbers": [99]}, {"source_goal": "x" * 251}, {"same_purpose": "true"},
    {"question": "Покажите настройки"}, {"filename": "another.docx"},
])
def test_relationship_provider_cannot_add_images_questions_or_untyped_claims(changes):
    with pytest.raises(ValidationError):
        RelatedInstructionSelection.model_validate(proposal(**changes))


def test_ready_alternative_is_verified_as_its_actual_goal_without_pretending_original_conditions_changed():
    data = payload()
    data["context"] = {"current_message": "Подключить устройство по Bluetooth."}
    unchanged = copy.deepcopy(data)
    model, calls = model_verifier(lambda wire, number: proposal())
    checks = []

    def verify(scoped, reading):
        checks.append((scoped, reading))
        return InstructionReading(goals={"setup": SourceGoalReading(
            status="SUPPORTED", evidence_ids=["picture", "height"], delivery_mode="PROCEDURE",
        )})

    model.verify_instruction = verify
    result = model.related_instruction(data)
    assert result.status == "RELATED" and result.evidence_ids == ["picture", "height"]
    assert result.original_goal_id == "setup" and len(checks) == 1 and data == unchanged
    scoped, reading = checks[0]
    assert scoped["requested_goals"] == {"setup": result.source_goal}
    assert scoped["requested_goal_kinds"] == {"setup": "ACTION"} and scoped["context"] == {}
    assert scoped["candidate_evidence"] == data["candidate_evidence"] and scoped["source"] == data["source"]
    assert reading.goals["setup"].evidence_ids == ["picture"]
    assert calls[0]["messages"][0]["content"] == RELATED_TASK


@pytest.mark.parametrize("status", ["MISSING", "RELATED", "CLARIFY"])
def test_related_model_claim_is_discarded_when_actual_alternative_goal_is_not_ready(status):
    model, _ = model_verifier(lambda wire, number: proposal())
    model.verify_instruction = lambda data, reading: InstructionReading(goals={"setup": SourceGoalReading(status=status)})
    result = model.related_instruction(payload())
    assert result.status == "UNRELATED" and result.evidence_ids == [] and result.source_goal == ""


def test_unrelated_source_never_gets_a_verification_call():
    model, calls = model_verifier(lambda wire, number: proposal(
        original_goal_id=None, scope_ref=None, evidence_ids=[], relation_kind="DIFFERENT_OPERATION", status="UNRELATED",
    ))
    model.verify_instruction = lambda *args: pytest.fail("Unrelated source must not be verified")
    assert model.related_instruction(payload()).status == "UNRELATED" and len(calls) == 1


def test_relationship_stage_repairs_closed_id_once_then_verifies_independently():
    model, calls = model_verifier(lambda wire, number: proposal(evidence_ids=["e999"] if number == 1 else ["e3"])
                                  if number < 3 else verdict(applicability_refs=["e3"],
                                                            required_material_kind="ORIGINAL_IMAGE"))
    result = model.related_instruction(payload())
    assert result.status == "RELATED" and result.evidence_ids == ["picture"]
    assert [call["schema"]["title"] for call in calls] == [
        "RelatedInstruction", "RelatedInstruction", "InstructionVerification",
    ]
    assert calls[1]["wire"]["repair_reason"]


def test_verification_failure_does_not_restart_the_related_stage():
    model, calls = model_verifier(lambda wire, number: proposal() if number == 1 else verdict(operation_refs=["e999"]))
    with pytest.raises(ValueError, match="UnknownInstructionVerificationEvidence"):
        model.related_instruction(payload())
    assert [call["schema"]["title"] for call in calls] == [
        "RelatedInstruction", "InstructionVerification", "InstructionVerification",
    ]


def test_complete_related_adapter_rejects_missing_programming_image_even_after_same_purpose_claim():
    model, calls = model_verifier(lambda wire, number: proposal(evidence_ids=["e1"]) if number == 1 else verdict(
        required_material_kind="ORIGINAL_IMAGE",
    ))
    result = model.related_instruction(payload())
    assert result.status == "UNRELATED" and len(calls) == 2


def source_scope_payload():
    data = payload()
    texts = ["ТСД TSC по Wi-Fi", "Сценарий",
             "ТСД и TSC по Wi-Fi. Настройки относятся к корпоративной связке 1С и LabelPrint.",
             "1. В LabelPrint выберите ZPL и TCP/IP."]
    data["candidate_evidence"] = {"r" + str(index): {"text": text, "support_kind": "TEXT", "image_numbers": []}
                                  for index, text in enumerate(texts)}
    data["candidate_sections"] = {"s" + str(index): {
        "text": text, "title": texts[0] if index == 0 else "Сценарий", "kind": "TEXT",
        "group_id": "title" if index == 0 else "scenario", "is_heading": index < 2,
        "evidence_ids": ["r" + str(index)],
    } for index, text in enumerate(texts)}
    data["context"] = {"current_message": "Мне нужен мобильный принтер по Bluetooth"}
    return data


def test_source_goal_is_exact_heading_and_scenario_without_invented_physical_conditions():
    data = source_scope_payload()
    scopes = related_source_scopes(data)
    assert scopes == {"scope1": "ТСД TSC по Wi-Fi. ТСД и TSC по Wi-Fi. "
                                "Настройки относятся к корпоративной связке 1С и LabelPrint."}
    assert "мобильн" not in scopes["scope1"] and "Bluetooth" not in scopes["scope1"]
    assert "выберите" not in scopes["scope1"] and "ZPL" not in scopes["scope1"]
    _, _, reverse = prepare_related(data)
    selected = RelatedInstructionSelection.model_validate(proposal(evidence_ids=["e1"]))
    result = resolve_related(selected, data, reverse)
    assert result.source_goal == scopes["scope1"] and len(result.source_goal) <= 250


def test_model_cannot_replace_backend_scope_with_generated_source_goal():
    with pytest.raises(ValidationError):
        RelatedInstructionSelection.model_validate(proposal(source_goal="Подключить мобильный принтер"))
    with pytest.raises(ValidationError):
        RelatedInstruction(original_goal_id="g1", source_goal="x" * 251, evidence_ids=["canonical"],
                           same_purpose=True, status="RELATED")


def test_scope_discards_operational_sentences_inside_a_purpose_paragraph():
    data = source_scope_payload()
    paragraph = "Настройка выполняется ИТ. Перед удалением существующей записи сохраните её параметры."
    data["candidate_sections"]["s2"]["text"] = paragraph
    scopes = related_source_scopes(data)
    assert "Настройка выполняется ИТ." in scopes["scope1"]
    assert "сохраните" not in scopes["scope1"]
