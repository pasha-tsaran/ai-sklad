"""The independent verifier may repair proof coverage, never source authority."""

import copy
import json

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.services.llm.instruction_reader import InstructionReading, SourceGoalReading
from app.services.llm.instruction_verification import (
    VERIFICATION_TASK, InstructionVerification, prepare_verification, resolve_verification,
)
from app.services.llm.openwebui import OpenWebUILLM
from app.services.chat.instruction_plan import render_plan, serialize_plan
from integrations.openwebui.approved_instruction_images import read_docx_images
from tests.test_full_instruction_reading import answer, read_context
from tests.test_instruction_reader import wire_evidence
from tests.test_openwebui_instruction_images import instruction
from tests.test_scanner_original_retrieval import scanner_provider
from tests.test_talk_knowledge import DOC, OTHER


def payload():
    return {
        "source": {"file_id": "private-file", "filename": "Новая инструкция.docx", "sha256": "a" * 64},
        "requested_goals": {"size": "Какой стандартный размер?", "setup": "Настроить устройство"},
        "requested_goal_kinds": {"size": "LOOKUP", "setup": "ACTION"},
        "context": {"current_message": "Какой размер и как настроить?"},
        "candidate_evidence": {
            "width": {"text": "Ширина 70 мм.", "support_kind": "TEXT", "image_numbers": []},
            "height": {"text": "Высота 40 мм.", "support_kind": "TEXT", "image_numbers": []},
            "picture": {"text": "Исходный настроечный код нужного режима.", "support_kind": "IMAGE",
                        "image_numbers": [3]},
        },
        "candidate_sections": {
            "s-width": {"title": "Размеры", "group_id": "sizes", "kind": "TEXT", "evidence_ids": ["width"]},
            "s-height": {"title": "Размеры", "group_id": "sizes", "kind": "TEXT", "evidence_ids": ["height"]},
            "s-picture": {"title": "Настройка", "group_id": "setup", "kind": "IMAGE",
                          "evidence_ids": ["picture"], "image_numbers": [3]},
        },
    }


def reading():
    return InstructionReading(goals={
        "size": SourceGoalReading(status="SUPPORTED", evidence_ids=["width"], delivery_mode="EXCERPT"),
        "setup": SourceGoalReading(status="SUPPORTED", evidence_ids=["picture"], delivery_mode="PROCEDURE"),
    })


def verdict(**changes):
    result = {"source_finding": "Область применения и все необходимые материалы подтверждены оригиналом.",
              "applicability_refs": ["e1"], "operation_refs": [], "condition_match": "EXACT",
              "required_material_kind": "TEXT", "material_refs": [], "material_state": "READY",
              "all_materials_available": True, "verdict": "PASS"}
    return {**result, **changes}


def model_verifier(responder):
    model = OpenWebUILLM(Settings(_env_file=None, llm_backend="openwebui", openwebui_api_key="test-key"))
    calls = []

    def request(messages, *, selection_schema=None, **options):
        wire = json.loads(messages[-1]["content"])
        calls.append({"wire": wire, "schema": selection_schema, "messages": messages, "options": options})
        return json.dumps(responder(wire, len(calls)))

    model._request = request
    return model, calls


def test_verification_schema_keeps_reasoned_conditions_and_materials_before_verdict():
    wire, schema, reverse = prepare_verification(payload(), "size", reading().goals["size"])
    assert schema["additionalProperties"] is False
    assert schema["required"] == [
        "source_finding", "operation_match", "condition_match", "goal_kind", "required_material_kind",
        "material_state", "all_materials_available", "verdict", "missing_observable_condition",
        "clarification_kind", "question", "applicability_refs", "operation_refs", "material_refs",
        "condition_refs", "answer_refs", "illustration_refs",
    ]
    for field in ("applicability_refs", "operation_refs", "material_refs"):
        assert schema["properties"][field]["items"] == {"$ref": "#/$defs/Evidence"}
        assert schema["$defs"]["Evidence"]["enum"] == list(reverse)
        assert schema["properties"][field]["uniqueItems"] is True
    assert "proposed_support_ids" not in wire
    assert wire["source_image_inventory"] == {"3": {"original_available": True, "evidence_ids": ["e3"]}}
    assert wire["approved_original"]["sections"][0]["evidence"] == {
        "e1": "Ширина 70 мм.", "e2": "Высота 40 мм.",
    }
    assert "private-file" not in json.dumps(wire) and "source_finding" not in wire


def test_independent_wire_is_identical_for_conflicting_prior_valid_hypotheses():
    data = payload()
    first = SourceGoalReading(status="SUPPORTED", evidence_ids=["width"], delivery_mode="EXCERPT",
                              source_finding="Предыдущая модель считает, что достаточно одной ширины.")
    second = SourceGoalReading(status="RELATED", evidence_ids=["picture"],
                               source_finding="Предыдущая модель предложила другую настройку устройства.")
    assert prepare_verification(data, "size", first) == prepare_verification(data, "size", second)
    second.evidence_ids = ["foreign-reference"]
    with pytest.raises(ValueError, match="UnknownInstructionVerificationEvidence"):
        prepare_verification(data, "size", second)


def test_union_all_roles_repairs_missing_second_property_without_role_array_requirement():
    data = payload()
    original = copy.deepcopy(data)
    answer = reading().goals["size"]
    _, _, reverse = prepare_verification(data, "size", answer)
    checked = InstructionVerification.model_validate(verdict(applicability_refs=["e1", "e2"], material_refs=["e2"]))
    result = resolve_verification(checked, data, "size", reverse, answer)
    assert result.status == "SUPPORTED" and result.evidence_ids == ["width", "height"]
    assert result.delivery_mode == "EXCERPT" and data == original and answer.evidence_ids == ["width"]


@pytest.mark.parametrize("changes", [
    {"condition_match": "DIFFERENT"}, {"condition_match": "UNKNOWN"},
    {"material_state": "MENTION_ONLY"}, {"material_state": "EXTERNAL_UNVERIFIED"},
    {"material_state": "UNVERIFIED_PURPOSE"}, {"material_state": "MISSING"},
    {"all_materials_available": False}, {"verdict": "MISSING"},
    {"applicability_refs": []}, {"required_material_kind": "ORIGINAL_IMAGE"},
])
def test_proclaimed_pass_cannot_override_unavailable_materials_or_wrong_conditions(changes):
    data = payload()
    answer = reading().goals["size"]
    _, _, reverse = prepare_verification(data, "size", answer)
    checked = InstructionVerification.model_validate(verdict(**changes))
    result = resolve_verification(checked, data, "size", reverse, answer)
    assert result.status == "MISSING" and result.evidence_ids == [] and result.delivery_mode is None


def test_programming_material_requires_actual_original_image_proof_in_any_role():
    data = payload()
    answer = reading().goals["setup"]
    _, _, reverse = prepare_verification(data, "setup", answer)
    checked = InstructionVerification.model_validate(verdict(
        required_material_kind="ORIGINAL_IMAGE", applicability_refs=["e3"],
    ))
    result = resolve_verification(checked, data, "setup", reverse, answer)
    assert result.status == "SUPPORTED" and result.evidence_ids == ["picture"]
    assert result.delivery_mode == "PROCEDURE"


def test_legacy_untyped_related_reading_can_be_upgraded_to_verified_full_procedure():
    data = payload()
    data["requested_goal_kinds"] = {}
    answer = SourceGoalReading(status="RELATED", evidence_ids=["width"])
    _, _, reverse = prepare_verification(data, "size", answer)
    checked = InstructionVerification.model_validate(verdict())
    result = resolve_verification(checked, data, "size", reverse, answer)
    assert result.status == "SUPPORTED" and result.delivery_mode == "PROCEDURE"


@pytest.mark.parametrize("changes,status", [
    ({"verdict": "RELATED", "condition_match": "DIFFERENT"}, "RELATED"),
    ({"verdict": "RELATED", "condition_match": "UNKNOWN"}, "MISSING"),
    ({"verdict": "RELATED", "condition_match": "DIFFERENT", "all_materials_available": False}, "MISSING"),
    ({"verdict": "RELATED", "condition_match": "DIFFERENT", "material_state": "MENTION_ONLY"}, "MISSING"),
])
def test_related_alternative_needs_its_own_available_source_material(changes, status):
    data = payload()
    answer = reading().goals["size"]
    _, _, reverse = prepare_verification(data, "size", answer)
    result = resolve_verification(InstructionVerification.model_validate(verdict(**changes)), data, "size", reverse, answer)
    assert result.status == status and result.delivery_mode is None


@pytest.mark.parametrize("changes", [
    {"operation_refs": ["e999"]}, {"material_refs": ["e2", "e2"]},
    {"applicability_refs": ["private-width"]},
])
def test_unknown_reference_is_rejected_even_in_an_unused_material_role(changes):
    data = payload()
    answer = reading().goals["size"]
    _, _, reverse = prepare_verification(data, "size", answer)
    with pytest.raises(ValueError, match="UnknownInstructionVerificationEvidence"):
        resolve_verification(InstructionVerification.model_validate(verdict(**changes)), data, "size", reverse, answer)


@pytest.mark.parametrize("changes", [
    {"all_materials_available": "true"}, {"image_numbers": [77]}, {"section_id": "foreign"},
    {"delivery_mode": "EXCERPT"}, {"answer_refs": "e1"}, {"source_finding": "x" * 401},
])
def test_verifier_cannot_forge_images_modes_questions_or_typed_readiness(changes):
    with pytest.raises(ValidationError):
        InstructionVerification.model_validate(verdict(**changes))


def test_adapter_preserves_missing_and_clarification_and_checks_only_proposed_goals():
    data = payload()
    data["requested_goals"].update(missing="Другая задача", unclear="Какая версия нужна?")
    result = reading()
    result.goals.update(missing=SourceGoalReading(status="MISSING", source_finding="Нет подходящего материала."),
                        unclear=SourceGoalReading(status="CLARIFY", evidence_ids=["width"],
                                                  question="Какой размер вы видите?", clarification_kind="VISIBLE_STATE"))
    unchanged = result.model_copy(deep=True)
    model, calls = model_verifier(lambda wire, number: verdict(
        applicability_refs=["e1", "e2"] if number == 1 else ["e3"],
        required_material_kind="TEXT" if number == 1 else "ORIGINAL_IMAGE",
    ))
    checked = model.verify_instruction(data, result)
    assert len(calls) == 2 and result == unchanged
    assert checked.goals["missing"] == result.goals["missing"] and checked.goals["unclear"] == result.goals["unclear"]
    assert checked.goals["size"].evidence_ids == ["width", "height"]
    assert all(call["messages"][0] == {"role": "system", "content": VERIFICATION_TASK} for call in calls)
    assert all(len(call["messages"]) == 2 and call["options"]["max_output_tokens"] == 2048 for call in calls)


def test_adapter_does_not_call_model_when_no_proposed_support_needs_verification():
    data = payload()
    result = InstructionReading(goals={key: SourceGoalReading(status="MISSING") for key in data["requested_goals"]})
    model, calls = model_verifier(lambda wire, number: pytest.fail("Unexpected verification call"))
    assert model.verify_instruction(data, result) == result and calls == []


def test_adapter_repairs_foreign_reference_once_without_changing_original_or_authority():
    data = payload()
    data["requested_goals"].pop("setup")
    data["requested_goal_kinds"].pop("setup")
    result = reading()
    result.goals.pop("setup")
    model, calls = model_verifier(lambda wire, number: verdict(operation_refs=["e999"] if number == 1 else ["e2"]))
    checked = model.verify_instruction(data, result)
    assert len(calls) == 2 and checked.goals["size"].evidence_ids == ["width", "height"]
    assert calls[0]["wire"]["approved_original"] == calls[1]["wire"]["approved_original"]
    assert calls[1]["wire"]["repair_reason"]


def test_adapter_two_invalid_verification_replies_never_authorize_original():
    data = payload()
    model, calls = model_verifier(lambda wire, number: verdict(operation_refs=["e999"]))
    with pytest.raises(ValueError, match="UnknownInstructionVerificationEvidence"):
        model.verify_instruction(data, reading())
    assert len(calls) == 2


def test_approved_acquisition_workflow_is_not_equated_with_missing_binary():
    assert "прошивку у ответственного" in VERIFICATION_TASK
    assert "наличие копии файла внутри базы знаний не требуется" in VERIFICATION_TASK
    assert "непроверенная внешняя страница" in VERIFICATION_TASK


def test_real_adapter_runs_reader_then_verifier_and_keeps_repaired_complete_facts():
    model, calls = model_verifier(lambda wire, number: {
        "goals": {"size": {"source_finding": "В оригинале найдено значение ширины.", "status": "SUPPORTED",
                            "evidence_ids": ["e1"], "delivery_mode": "EXCERPT", "question": "",
                            "clarification_kind": None},
                  "setup": {"source_finding": "В этом чтении второй порядок не выбран.", "status": "MISSING",
                             "evidence_ids": [], "delivery_mode": None, "question": "", "clarification_kind": None}}
    } if number == 1 else verdict(applicability_refs=["e1", "e2"]))
    result = model.read_instruction(payload())
    assert result.goals["size"].evidence_ids == ["width", "height"]
    assert result.goals["setup"].status == "MISSING" and len(calls) == 2
    assert [call["schema"]["title"] for call in calls] == ["InstructionReading", "InstructionVerification"]
    assert calls[1]["messages"][0]["content"] == VERIFICATION_TASK


def test_reading_supported_is_removed_when_independent_material_check_fails():
    model, calls = model_verifier(lambda wire, number: {
        "goals": {"size": {"source_finding": "Источник содержит упоминание нужной операции.", "status": "SUPPORTED",
                            "evidence_ids": ["e1"], "delivery_mode": "EXCERPT", "question": "",
                            "clarification_kind": None},
                  "setup": {"source_finding": "Для другой цели подтверждения пока нет.", "status": "MISSING",
                             "evidence_ids": [], "delivery_mode": None, "question": "", "clarification_kind": None}}
    } if number == 1 else verdict(required_material_kind="ORIGINAL_IMAGE"))
    result = model.read_instruction(payload())
    assert result.goals["size"].status == "MISSING" and result.goals["size"].evidence_ids == []
    assert len(calls) == 2


def test_verifier_format_repair_does_not_repeat_the_successful_reader_stage():
    def reply(wire, number):
        if number == 1:
            return {"goals": {key: {"source_finding": "Сведения об этом размере присутствуют в оригинале.",
                                    "status": "SUPPORTED", "evidence_ids": ["e1"], "delivery_mode": "EXCERPT",
                                    "question": "", "clarification_kind": None}
                              for key in wire["requested_goals"]}}
        return verdict(operation_refs=["e999"] if number == 2 else ["e2"])

    data = payload()
    data["requested_goals"].pop("setup")
    data["requested_goal_kinds"].pop("setup")
    model, calls = model_verifier(reply)
    result = model.read_instruction(data)
    assert result.goals["size"].evidence_ids == ["width", "height"]
    assert [call["schema"]["title"] for call in calls] == [
        "InstructionReading", "InstructionVerification", "InstructionVerification",
    ]


def test_full_reader_and_verifier_repaired_facts_reach_original_rendered_answer():
    width, height = "Ширина носителя составляет 70 мм.", "Высота носителя составляет 40 мм."
    original = ("# Стандартные значения\n" + width + "\n" + height).encode()
    knowledge, _ = scanner_provider([(DOC, "Новый материал.md", original)])
    knowledge.settings = knowledge.settings.model_copy(update={"llm_backend": "openwebui"})

    def reply(wire, number):
        if number == 1:
            return answer(wire, {"g1": width})
        return verdict(applicability_refs=wire_evidence(wire, width) + wire_evidence(wire, height))

    model, calls = model_verifier(reply)
    knowledge.reader = model.read_instruction
    result = knowledge.search_with_llm("Какой стандартный размер?", read_context({"g1": "Узнать стандартный размер"}))
    blocks, _, _, more = render_plan(result.plan, serialize_plan(result.plan))
    text = "\n".join(value for value, _ in blocks)
    assert width in text and height in text and not more
    assert "Область применения и все необходимые материалы" not in text
    assert len(calls) == 2 and result.plan[0].delivery_mode == "EXCERPT"


def test_verifier_canonical_image_reference_reaches_original_source_pixels():
    caption = "Настроечный код выбранного режима устройства."
    original = instruction([caption])
    knowledge, _ = scanner_provider([(DOC, "Совершенно новая настройка.docx", original)])
    knowledge.settings = knowledge.settings.model_copy(update={"llm_backend": "openwebui"})

    def reply(wire, number):
        if number == 1:
            return answer(wire, {"g1": caption})
        image_references = next(iter(wire["source_image_inventory"].values()))["evidence_ids"]
        return verdict(applicability_refs=[], material_refs=image_references, required_material_kind="ORIGINAL_IMAGE")

    model, _ = model_verifier(reply)
    knowledge.reader = model.read_instruction
    result = knowledge.search_with_llm("Настроить нужный режим.",
                                      read_context({"g1": "Настроить выбранный режим"}, {"g1": "ACTION"}))
    assert result.approved.illustrations[1].data == read_docx_images(original)[0].data
    assert 1 in [number for part in result.plan for section in part.sections for number in section["image_numbers"]]


def test_independent_rejection_of_first_hypothesis_allows_later_exact_original(monkeypatch):
    wrong = "Для устройства Альфа эта инструкция меняет только звук."
    exact = "Для устройства Альфа откройте окно проверки и прочитайте полученный результат."
    knowledge, _ = scanner_provider([(DOC, "Настройки.md", wrong.encode()), (OTHER, "Проверка.md", exact.encode())])
    knowledge.settings = knowledge.settings.model_copy(update={"llm_backend": "openwebui"})
    monkeypatch.setattr(knowledge, "_instruction_ranked_sources", lambda *args, **kwargs: [DOC, OTHER])

    def reply(wire, number):
        filename = wire["approved_original"]["filename"]
        text = wrong if filename == "Настройки.md" else exact
        if "requested_goals" in wire:
            return answer(wire, {"g1": text})
        return verdict(applicability_refs=wire_evidence(wire, text),
                       condition_match="DIFFERENT" if filename == "Настройки.md" else "EXACT")

    model, calls = model_verifier(reply)
    knowledge.reader = model.read_instruction
    result = knowledge.search_with_llm("Устройство Альфа не работает, хочу проверить результат.",
                                      read_context({"g1": "Проверить устройство Альфа"}, {"g1": "ACTION"}))
    assert result.approved.document["source"]["file_id"] == OTHER and result.result == "MATCH"
    assert len(calls) == 4
