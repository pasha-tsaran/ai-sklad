"""Native status correlations and private independently verified answer scope."""

import copy

import pytest
from pydantic import ValidationError

from app.services.llm.instruction_reader import (
    InstructionReading, SourceGoalReading, prepare_reading, resolve_reading, verified_reading_scope,
)
from app.services.llm.instruction_related import RelatedInstructionSelection, prepare_related, resolve_related
from app.services.llm.instruction_verification import InstructionVerification, prepare_verification, resolve_verification
from app.services.llm.openwebui import _instruction_contract_repair
from tests.test_instruction_related import proposal
from tests.test_instruction_verification import model_verifier, payload, reading, verdict


def modern(**changes):
    return verdict(operation_match="EXACT", goal_kind="ACTION", operation_refs=["e1"],
                   missing_observable_condition="NONE", condition_refs=[], question="",
                   clarification_kind=None, answer_refs=["e1", "e2"], **changes)


def checked(changes=None, *, data=None, goal="setup"):
    data = payload() if data is None else data
    answer = reading().goals[goal]
    _, _, reverse = prepare_verification(data, goal, answer)
    values = modern()
    values.update(changes or {})
    return resolve_verification(InstructionVerification.model_validate(values), data, goal, reverse, answer)


def test_native_reading_statuses_bind_every_correlated_field():
    _, schema, reverse = prepare_reading(payload())
    assert schema["$defs"]["Evidence"]["enum"] == list(reverse)
    for slot in schema["properties"]["goals"]["properties"].values():
        branches = {branch["properties"]["status"]["const"]: branch["properties"] for branch in slot["anyOf"]}
        assert branches["SUPPORTED"]["evidence_ids"]["minItems"] == 1
        assert branches["SUPPORTED"]["delivery_mode"]["type"] == "string"
        assert branches["MISSING"]["evidence_ids"]["maxItems"] == 0
        assert branches["RELATED"]["evidence_ids"]["minItems"] == 1
        assert branches["CLARIFY"]["question"]["minLength"] == 1
        for status, fields in branches.items():
            assert fields["evidence_ids"]["items"] == {"$ref": "#/$defs/Evidence"}
            if status != "CLARIFY":
                assert fields["question"] == {"const": ""}
                assert fields["clarification_kind"] == {"type": "null"}
            if status != "SUPPORTED":
                assert fields["delivery_mode"] == {"type": "null"}


def test_only_explicit_independent_reader_enables_both_provisional_modes():
    _, fixed, _ = prepare_reading(payload())
    _, reviewed, _ = prepare_reading(payload(), allow_mode_review=True)
    fixed_branch = fixed["properties"]["goals"]["properties"]["size"]["anyOf"][0]
    review_branch = reviewed["properties"]["goals"]["properties"]["size"]["anyOf"][0]
    assert fixed_branch["properties"]["delivery_mode"]["enum"] == ["EXCERPT"]
    assert review_branch["properties"]["delivery_mode"]["enum"] == ["PROCEDURE", "EXCERPT"]


def test_native_positive_programming_image_requires_actual_image_material_alias():
    _, schema, _ = prepare_verification(payload(), "setup", reading().goals["setup"])
    assert schema["required"] == list(schema["properties"])
    image_branches = [branch for branch in schema["anyOf"]
                      if branch["properties"]["required_material_kind"].get("const") == "ORIGINAL_IMAGE"]
    assert len(image_branches) == 3
    assert schema["$defs"]["OriginalImageEvidence"]["enum"] == ["e3"]
    for branch in image_branches:
        fields = branch["properties"]
        assert fields["material_refs"]["minItems"] == 1
        assert fields["material_refs"]["items"] == {"$ref": "#/$defs/OriginalImageEvidence"}
        assert fields["material_state"]["const"] == "READY"
        assert fields["all_materials_available"]["const"] is True


def test_no_image_registry_has_no_native_positive_image_branch():
    data = payload()
    del data["candidate_evidence"]["picture"]
    del data["candidate_sections"]["s-picture"]
    _, schema, _ = prepare_verification(data, "size", reading().goals["size"])
    assert "OriginalImageEvidence" not in schema["$defs"]
    assert all(branch["properties"]["required_material_kind"].get("const") != "ORIGINAL_IMAGE"
               for branch in schema["anyOf"])


def test_modern_scope_uses_actual_source_and_separate_complete_answer_plan():
    data = payload()
    original = copy.deepcopy(data)
    result = checked({"material_refs": ["e3"], "answer_refs": ["e1", "e2", "e3"]}, data=data)
    scope = verified_reading_scope(result, data, "setup")
    assert scope == {"source": data["source"], "goal_id": "setup", "goal_text": data["requested_goals"]["setup"],
                     "goal_kind": "ACTION", "delivery_mode": "PROCEDURE", "operation_refs": ("width",),
                     "material_refs": ("picture",), "answer_refs": ("width", "height", "picture"),
                     "confirmation_refs": ("width",)}
    assert result.evidence_ids == ["width", "picture"]
    assert result.model_copy(deep=True)._verified_scope == scope
    assert "_verified_scope" not in result.model_dump()
    assert "_verified_scope" not in SourceGoalReading.model_json_schema()["properties"]
    assert data == original and scope["source"] is not data["source"]


@pytest.mark.parametrize("claim", ["_verified_scope", "verified_scope", "source", "answer_refs"])
def test_reader_json_cannot_claim_backend_scope(claim):
    with pytest.raises(ValidationError):
        SourceGoalReading.model_validate({"status": "SUPPORTED", "evidence_ids": ["width"],
                                         "delivery_mode": "PROCEDURE", claim: {"goal_kind": "ACTION"}})


def test_legacy_verifier_retains_default_semantics_without_scope_privilege():
    data = payload()
    answer = reading().goals["setup"]
    _, _, reverse = prepare_verification(data, "setup", answer)
    result = resolve_verification(InstructionVerification.model_validate(verdict()), data, "setup", reverse, answer)
    assert result.status == "SUPPORTED" and result._verified_scope is None


def test_goal_kind_correction_requires_bound_private_scope_for_final_reader():
    data = payload()
    data["requested_goal_kinds"]["setup"] = "LOOKUP"
    supported = checked(data=data)
    identities = {reference: reference for reference in data["candidate_evidence"]}
    other = SourceGoalReading(status="MISSING")
    final = resolve_reading(InstructionReading(goals={"size": other, "setup": supported}), data, identities)
    assert final.goals["setup"].delivery_mode == "PROCEDURE"
    legacy = SourceGoalReading(status="SUPPORTED", evidence_ids=["width"], delivery_mode="PROCEDURE")
    with pytest.raises(ValueError, match="InvalidInstructionReadingMode"):
        resolve_reading(InstructionReading(goals={"size": other, "setup": legacy}), data, identities)
    supported._verified_scope["source"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="InvalidInstructionVerifiedScope"):
        resolve_reading(final, data, identities)


@pytest.mark.parametrize("field", ["operation_refs", "material_refs", "answer_refs", "condition_refs"])
def test_unknown_canonical_plan_or_proof_reference_is_rejected(field):
    with pytest.raises(ValueError, match="UnknownInstructionVerificationEvidence"):
        checked({field: ["e999"]})


def test_answer_plan_has_its_own_96_reference_bound_outside_32_proof_union():
    data = payload()
    for number in range(4, 54):
        reference = "proof" + str(number)
        data["candidate_evidence"][reference] = {"text": "Original section " + str(number),
                                                 "support_kind": "TEXT", "image_numbers": []}
        data["candidate_sections"]["section" + str(number)] = {"title": "Steps", "evidence_ids": [reference]}
    result = checked({"answer_refs": ["e" + str(number) for number in range(1, 54)]}, data=data)
    assert len(result._verified_scope["answer_refs"]) == 53
    assert result.evidence_ids == ["width"]


@pytest.mark.parametrize("changes", [{"answer_refs": []}, {"operation_refs": []}])
def test_supported_modern_plan_needs_operation_and_complete_answer_selection(changes):
    with pytest.raises(ValueError, match="IncompleteInstructionVerificationAnswerPlan"):
        checked(changes)


def test_programming_image_positive_contradiction_triggers_one_targeted_repair():
    data = payload()
    data["requested_goals"] = {"setup": data["requested_goals"]["setup"]}
    data["requested_goal_kinds"] = {"setup": "ACTION"}
    result = InstructionReading(goals={"setup": reading().goals["setup"]})
    def reply(wire, number):
        values = modern()
        values.update(required_material_kind="ORIGINAL_IMAGE", material_refs=[] if number == 1 else ["e3"],
                      answer_refs=["e1", "e3"])
        return values
    model, calls = model_verifier(reply)
    verified = model.verify_instruction(data, result).goals["setup"]
    assert verified.status == "SUPPORTED" and len(calls) == 2
    assert calls[1]["wire"]["repair_reason"].startswith("MissingInstructionVerificationOriginalImage.")
    assert calls[0]["wire"]["approved_original"] == calls[1]["wire"]["approved_original"]


def test_ordinary_ui_text_operation_does_not_require_original_barcode_image():
    assert checked().status == "SUPPORTED"


@pytest.mark.parametrize("state", ["UNVERIFIED_PURPOSE", "MISSING", "MENTION_ONLY", "EXTERNAL_UNVERIFIED"])
def test_unknown_reset_barcode_purpose_remains_missing_without_model_question(state):
    result = checked({"required_material_kind": "ORIGINAL_IMAGE", "material_state": state,
                      "all_materials_available": False, "verdict": "MISSING", "answer_refs": []})
    assert result.status == "MISSING" and not result.question and result._verified_scope is None


def test_same_operation_with_one_source_backed_unknown_observable_condition_clarifies():
    values = {"condition_match": "UNKNOWN", "missing_observable_condition": "APPLICATION_SURFACE",
              "condition_refs": ["e2"], "question": "Вы открываете почту в браузере или отдельной программе?",
              "clarification_kind": "VISIBLE_STATE", "answer_refs": []}
    result = checked(values)
    assert result.status == "CLARIFY" and result.evidence_ids == ["height"]
    assert result.delivery_mode is None and result.clarification_condition == "APPLICATION_SURFACE"
    assert result._verified_scope is None


@pytest.mark.parametrize("changes", [
    {"operation_match": "DIFFERENT"}, {"operation_match": "UNPROVEN"},
    {"question": "Какую модель устройства вы используете?"},
    {"condition_refs": []}, {"missing_observable_condition": "NONE"},
])
def test_unknown_condition_cannot_change_operation_or_ask_for_device_model(changes):
    values = {"condition_match": "UNKNOWN", "missing_observable_condition": "APPLICATION_SURFACE",
              "condition_refs": ["e2"], "question": "Вы открываете почту в браузере?",
              "clarification_kind": "VISIBLE_STATE", "answer_refs": [], **changes}
    with pytest.raises(ValueError):
        checked(values)


def test_previously_unanswered_observable_question_is_not_repeated():
    data = payload()
    data["context"]["unknown_answers"] = [{"field": "application", "clarification_condition": "APPLICATION_SURFACE",
                                               "goal_ids": ["setup"], "question": "Вы открываете ПОЧТУ в браузере ?"}]
    with pytest.raises(ValueError, match="InvalidInstructionVerificationQuestion"):
        checked({"condition_match": "UNKNOWN", "missing_observable_condition": "APPLICATION_SURFACE",
                 "condition_refs": ["e2"], "question": "Вы открываете почту в браузере?",
                 "clarification_kind": "VISIBLE_STATE", "answer_refs": []}, data=data)


@pytest.mark.parametrize("target", ["DIFFERENT_TARGET", "UNPROVEN"])
def test_related_different_service_or_function_is_rejected(target):
    data = payload()
    _, _, reverse = prepare_related(data)
    result = resolve_related(RelatedInstructionSelection.model_validate(proposal(target_relation=target)), data, reverse)
    assert result.status == "UNRELATED"


def test_native_related_requires_source_scope_and_same_functional_target():
    _, schema, _ = prepare_related(payload())
    related, unrelated = schema["anyOf"]
    assert related["properties"]["scope_ref"]["enum"] == ["scope1"]
    assert related["properties"]["evidence_ids"]["minItems"] == 1
    assert related["properties"]["target_relation"]["enum"] == ["SAME_FUNCTIONAL_TARGET", "ALTERNATIVE_ROUTE"]
    assert unrelated["properties"]["scope_ref"] == {"type": "null"}
    assert unrelated["properties"]["evidence_ids"]["maxItems"] == 0


def test_repair_reason_never_serializes_pydantic_input_or_arbitrary_error_data():
    secret = "sensitive-token-should-never-be-copied"
    try:
        InstructionVerification.model_validate(verdict(all_materials_available=secret))
    except ValidationError as error:
        reason = _instruction_contract_repair(error, "Verification")
    assert secret not in reason and reason.startswith("InvalidInstructionVerificationFormat.")
    assert secret not in _instruction_contract_repair(ValueError(secret), "Verification")


def test_distinct_observable_detail_can_follow_an_answer_with_same_category_and_source():
    data = payload()
    data["context"]["answered_conditions"] = [{
        "clarification_condition": "PHYSICAL_CONNECTION", "goal_ids": ["setup"],
        "question": "Первый принтер подключён кабелем?", "answer": "Да",
        "sources": [{"file_id": data["source"]["file_id"], "sha256": data["source"]["sha256"]}],
    }]
    result = checked({"condition_match": "UNKNOWN", "missing_observable_condition": "PHYSICAL_CONNECTION",
                      "condition_refs": ["e2"], "question": "Второй принтер подключён кабелем?",
                      "clarification_kind": "PHYSICAL_SETUP", "answer_refs": []}, data=data)
    assert result.status == "CLARIFY"


@pytest.mark.parametrize("verdict_value", ["MISSING", "RELATED"])
def test_modern_exact_ready_flags_cannot_hide_a_negative_verdict(verdict_value):
    with pytest.raises(ValueError, match="ContradictoryInstructionVerification"):
        checked({"verdict": verdict_value, "required_material_kind": "ORIGINAL_IMAGE",
                 "material_refs": [], "answer_refs": []})


def test_modern_negative_ready_contradiction_gets_the_existing_single_repair():
    data = payload()
    data["requested_goals"] = {"setup": data["requested_goals"]["setup"]}
    data["requested_goal_kinds"] = {"setup": "ACTION"}
    original = InstructionReading(goals={"setup": reading().goals["setup"]})
    def reply(wire, number):
        values = modern()
        values.update(required_material_kind="ORIGINAL_IMAGE", verdict="MISSING" if number == 1 else "PASS",
                      material_refs=[] if number == 1 else ["e3"], answer_refs=[] if number == 1 else ["e1", "e3"])
        return values
    model, calls = model_verifier(reply)
    result = model.verify_instruction(data, original).goals["setup"]
    assert result.status == "SUPPORTED" and len(calls) == 2
    assert calls[1]["wire"]["repair_reason"].startswith("ContradictoryInstructionVerification.")
    assert result._verified_scope["material_refs"] == ("picture",)
    assert result._verified_scope["answer_refs"] == ("width", "picture")


def test_native_decisions_precede_dependent_reference_lists():
    data = payload()
    _, reader_schema, _ = prepare_reading(data)
    for slot in reader_schema["properties"]["goals"]["properties"].values():
        for branch in slot["anyOf"]:
            assert list(branch["properties"])[:3] == ["source_finding", "status", "evidence_ids"]
    _, schema, _ = prepare_verification(data, "setup", reading().goals["setup"])
    order = ["source_finding", "operation_match", "condition_match", "goal_kind", "required_material_kind",
             "material_state", "all_materials_available", "verdict", "missing_observable_condition",
             "clarification_kind", "question", "applicability_refs", "operation_refs", "material_refs",
             "condition_refs", "answer_refs", "illustration_refs"]
    assert list(schema["properties"]) == order and schema["required"] == order
    assert all(list(branch["properties"]) == order for branch in schema["anyOf"])


def test_native_missing_branches_each_exclude_the_fully_positive_claims():
    _, schema, _ = prepare_verification(payload(), "setup", reading().goals["setup"])
    positive = {"operation_match": "EXACT", "condition_match": "EXACT",
                "material_state": "READY", "all_materials_available": True}
    branches = [branch["properties"] for branch in schema["anyOf"]
                if branch["properties"]["verdict"].get("const") == "MISSING"]
    assert len(branches) == 5
    for fields in branches:
        assert any(("const" in fields[name] and fields[name]["const"] != value)
                   or ("enum" in fields[name] and value not in fields[name]["enum"])
                   for name, value in positive.items())


@pytest.mark.parametrize("reason", [
    {"operation_match": "DIFFERENT"}, {"operation_match": "UNPROVEN"},
    {"condition_match": "DIFFERENT"}, {"material_state": "MISSING"},
    {"all_materials_available": False},
])
def test_modern_genuine_missing_reasons_remain_normal_source_misses(reason):
    result = checked({"verdict": "MISSING", "answer_refs": [], **reason})
    assert result.status == "MISSING" and not result.question


@pytest.mark.parametrize("allow", [False, True])
def test_unknown_condition_without_claimed_observable_question_is_a_normal_source_miss(allow):
    data = payload()
    data["allow_clarification"] = allow
    result = checked({"condition_match": "UNKNOWN", "verdict": "MISSING", "answer_refs": []}, data=data)
    assert result.status == "MISSING" and not result.question
    _, schema, _ = prepare_verification(data, "setup", reading().goals["setup"])
    branches = [branch["properties"] for branch in schema["anyOf"]]
    unknown = [fields for fields in branches if fields["condition_match"].get("const") == "UNKNOWN"]
    if not allow:
        assert len(unknown) == 1 and unknown[0]["verdict"] == {"const": "MISSING"}
        assert all(fields["question"].get("const") == "" for fields in branches)
    else:
        missing = [fields for fields in unknown if fields["verdict"].get("const") == "MISSING"]
        assert len(missing) == 1 and missing[0]["missing_observable_condition"] == {"const": "NONE"}


def test_native_related_requires_different_conditions_with_ready_exact_operation():
    _, schema, _ = prepare_verification(payload(), "setup", reading().goals["setup"])
    related = [branch["properties"] for branch in schema["anyOf"]
               if branch["properties"]["verdict"].get("const") == "RELATED"]
    assert len(related) == 2
    for fields in related:
        assert fields["operation_match"] == {"const": "EXACT"}
        assert fields["condition_match"] == {"const": "DIFFERENT"}
        assert fields["material_state"] == {"const": "READY"}
        assert fields["all_materials_available"] == {"const": True}
        assert fields["answer_refs"]["maxItems"] == 0


def test_text_procedure_illustration_role_adds_original_image_to_verified_answer_only():
    result = checked({"required_material_kind": "TEXT", "material_refs": [], "illustration_refs": ["e3"]})
    assert result.status == "SUPPORTED" and result.evidence_ids == ["width"]
    assert result._verified_scope["material_refs"] == ()
    assert result._verified_scope["answer_refs"] == ("width", "height", "picture")
    assert payload()["candidate_evidence"]["picture"]["image_numbers"] == [3]


@pytest.mark.parametrize("illustrations,expected", [([], ("width", "height")), (["e3"], ("width", "height", "picture"))])
def test_lookup_illustration_role_is_optional_and_preserves_excerpt(illustrations, expected):
    result = checked({"goal_kind": "LOOKUP", "illustration_refs": illustrations}, goal="size")
    assert result.status == "SUPPORTED" and result.delivery_mode == "EXCERPT"
    assert result._verified_scope["answer_refs"] == expected


def test_native_illustration_role_binds_only_actual_image_aliases_and_negative_arrays_are_empty():
    _, schema, _ = prepare_verification(payload(), "setup", reading().goals["setup"])
    role = schema["properties"]["illustration_refs"]
    assert role["maxItems"] == 24 and role["uniqueItems"] is True
    assert role["items"] == {"$ref": "#/$defs/OriginalImageEvidence"}
    assert schema["$defs"]["OriginalImageEvidence"]["enum"] == ["e3"]
    for branch in schema["anyOf"]:
        fields = branch["properties"]
        if fields["verdict"].get("const") != "PASS":
            assert fields["illustration_refs"]["maxItems"] == 0


def test_native_no_image_registry_uses_empty_illustrations_without_an_empty_enum():
    data = payload()
    del data["candidate_evidence"]["picture"]
    del data["candidate_sections"]["s-picture"]
    _, schema, _ = prepare_verification(data, "size", reading().goals["size"])
    role = schema["properties"]["illustration_refs"]
    assert role == {"type": "array", "items": {"type": "string"}, "maxItems": 0, "uniqueItems": True}
    assert "OriginalImageEvidence" not in schema["$defs"]


@pytest.mark.parametrize("references,code", [(["e999"], "UnknownInstructionVerificationEvidence"),
                                             (["e1"], "InvalidInstructionVerificationIllustration")])
def test_foreign_or_text_illustration_reference_is_rejected(references, code):
    with pytest.raises(ValueError, match=code):
        checked({"illustration_refs": references})


def test_original_programming_code_still_requires_its_material_proof_with_an_illustration():
    with pytest.raises(ValueError, match="MissingInstructionVerificationOriginalImage"):
        checked({"required_material_kind": "ORIGINAL_IMAGE", "material_refs": [], "illustration_refs": ["e3"]})


def test_legacy_verifier_illustration_default_preserves_previous_callback_semantics():
    verification = InstructionVerification.model_validate(verdict())
    assert verification.illustration_refs == []
    data = payload()
    answer = reading().goals["setup"]
    _, _, reverse = prepare_verification(data, "setup", answer)
    result = resolve_verification(verification, data, "setup", reverse, answer)
    assert result.status == "SUPPORTED" and result._verified_scope is None


def test_answer_and_illustration_union_is_deduplicated_and_bounded_at_96():
    data = payload()
    for number in range(4, 97):
        reference = "text" + str(number)
        data["candidate_evidence"][reference] = {"text": "Original text " + str(number),
                                                "support_kind": "TEXT", "image_numbers": []}
        data["candidate_sections"]["section" + str(number)] = {"title": "Original", "evidence_ids": [reference]}
    data["candidate_evidence"]["last-picture"] = {"text": "Another original illustration.",
                                                  "support_kind": "IMAGE", "image_numbers": [4]}
    data["candidate_sections"]["last-image"] = {"title": "Original", "kind": "IMAGE", "image_numbers": [4],
                                                "evidence_ids": ["last-picture"]}
    answers = ["e" + str(number) for number in range(1, 97)]
    result = checked({"answer_refs": answers, "illustration_refs": ["e3"]}, data=data)
    assert len(result._verified_scope["answer_refs"]) == 96
    assert result.evidence_ids == ["width"]
    with pytest.raises(ValueError, match="InstructionVerificationAnswerTooLarge"):
        checked({"answer_refs": answers, "illustration_refs": ["e97"]}, data=data)


def test_negative_outcome_cannot_carry_illustrations():
    with pytest.raises(ValueError, match="UnexpectedInstructionVerificationAnswerPlan"):
        checked({"operation_match": "DIFFERENT", "verdict": "MISSING", "answer_refs": [],
                 "illustration_refs": ["e3"]})
