"""Optional illustration refinement respects payload and canonical answer budgets."""

import copy
import json
import socket
from types import SimpleNamespace

import pytest

from app.core.config import Settings
from app.services.llm.instruction_reader import SourceGoalReading, prepare_reading
from app.services.llm.instruction_verification import (
    ILLUSTRATION_TASK, InstructionIllustrationSelection, InstructionVerification,
    compact_illustration_wire, prepare_illustrations, resolve_illustrations, resolve_verification,
)
from app.services.llm.openwebui import OpenWebUILLM
from app.services.rag.talk import MAX_SELECTION_PAYLOAD_CHARS


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("These tests use only synthetic in-memory source data")

    monkeypatch.setattr(socket.socket, "connect", blocked)


def example(*, refs=1, metadata_size=0, body_size=0, pictures=2):
    evidence, sections = {}, {}
    for index in range(refs):
        reference, identity = "text-" + str(index), "section-" + str(index)
        text = "Original source statement " + str(index)
        evidence[reference] = {"text": text, "support_kind": "TEXT", "image_numbers": []}
        sections[identity] = {
            "id": identity, "title": "Source purpose", "kind": "TEXT", "image_numbers": [],
            "text": text + (" original body" * body_size if index == 0 else ""), "evidence_ids": [reference],
            "ordinal": index + 1, "is_heading": False, "heading_level": None, "group_id": "main-procedure",
            "step_ids": ["technical-step-" + "x" * metadata_size], "image_ids": [],
        }
    for index in range(pictures):
        reference, identity = "picture-" + str(index), "picture-section-" + str(index)
        evidence[reference] = {"text": "Exact original caption " + str(index), "support_kind": "IMAGE",
                               "image_numbers": [index + 1]}
        sections[identity] = {
            "id": identity, "title": "Source purpose", "kind": "IMAGE", "image_numbers": [index + 1],
            "text": evidence[reference]["text"], "evidence_ids": [reference], "ordinal": refs + index + 1,
            "is_heading": False, "group_id": "main-procedure", "step_ids": ["picture-step"],
            "image_ids": ["original-picture-" + str(index)],
        }
    payload = {
        "source": {"file_id": "example-source", "filename": "Example.docx", "sha256": "a" * 64},
        "requested_goals": {"g1": "Find the relevant reference information"},
        "requested_goal_kinds": {"g1": "LOOKUP"}, "context": {"current_message": "Find this information"},
        "candidate_evidence": evidence, "candidate_sections": sections,
    }
    selected = tuple(reference for reference in evidence if reference.startswith("text-"))
    answer = SourceGoalReading(status="SUPPORTED", evidence_ids=list(selected[:1]), delivery_mode="EXCERPT")
    answer._verified_scope = {
        "source": copy.deepcopy(payload["source"]), "goal_id": "g1", "goal_text": payload["requested_goals"]["g1"],
        "goal_kind": "LOOKUP", "delivery_mode": "EXCERPT", "operation_refs": selected[:1],
        "material_refs": (), "answer_refs": selected, "confirmation_refs": selected[:1],
    }
    return payload, SimpleNamespace(required_material_kind="TEXT"), answer


def model_with_reply(reply=None):
    model = OpenWebUILLM(Settings(_env_file=None, llm_backend="openwebui", openwebui_api_key="test-key"))
    calls = []

    def request(messages, **options):
        calls.append({"messages": copy.deepcopy(messages), "options": copy.deepcopy(options)})
        return json.dumps(reply(json.loads(messages[-1]["content"]), options) if reply else {"illustration_refs": []})

    model._request = request
    return model, calls


def test_small_request_with_room_preserves_normal_preparation_exactly():
    payload, verification, answer = example()
    wire, schema, _ = prepare_illustrations(payload, "g1", verification, answer)
    model, calls = model_with_reply()
    result = model._instruction_illustrations(payload, "g1", verification, answer)
    assert calls == [{
        "messages": [{"role": "system", "content": ILLUSTRATION_TASK},
                     {"role": "user", "content": json.dumps(wire, ensure_ascii=False)}],
        "options": {"selection_schema": schema, "max_output_tokens": 512},
    }]
    assert result._verified_scope == answer._verified_scope


def test_small_full_answer_cap_keeps_valid_text_without_calling_picker():
    payload, verification, answer = example(refs=96)
    wire = prepare_illustrations(payload, "g1", verification, answer)[0]
    assert len(json.dumps(wire, ensure_ascii=False)) < MAX_SELECTION_PAYLOAD_CHARS
    before = copy.deepcopy(answer._verified_scope)
    model, calls = model_with_reply(lambda *args: pytest.fail("No answer-ref slot remains"))
    assert model._instruction_illustrations(payload, "g1", verification, answer) is answer
    assert calls == [] and answer._verified_scope == before


def test_small_one_slot_bounds_schema_without_changing_wire_or_candidates():
    payload, verification, answer = example(refs=95, pictures=3)
    wire = prepare_illustrations(payload, "g1", verification, answer)[0]
    assert len(json.dumps(wire, ensure_ascii=False)) < MAX_SELECTION_PAYLOAD_CHARS

    def choose(actual, options):
        assert actual == wire and len(actual["image_candidates"]) == 3
        assert options["selection_schema"]["properties"]["illustration_refs"]["maxItems"] == 1
        return {"illustration_refs": [next(iter(actual["image_candidates"]))]}

    model, calls = model_with_reply(choose)
    result = model._instruction_illustrations(payload, "g1", verification, answer)
    assert len(calls) == 1 and len(result._verified_scope["answer_refs"]) == 96
    assert result._verified_scope["source"] == payload["source"]
    assert result._verified_scope["answer_refs"][:-1] == answer._verified_scope["answer_refs"]


def test_overflow_compaction_preserves_text_captions_and_conditional_relationships():
    payload, verification, answer = example(refs=20, metadata_size=3000)
    step = payload["candidate_sections"]["section-0"]
    picture = payload["candidate_sections"]["picture-section-0"]
    conditional = {"parent_group_id": "parent-procedure", "condition_section_id": "condition-owner",
                   "condition_text": "If this original condition holds", "exclusive_set_id": "alternative-set",
                   "exclusive_parent_group_id": "parent-procedure", "exclusive_choice_section_id": "choice-owner",
                   "scope_ambiguous": True}
    step.update(conditional, scope_group_ids=["main-procedure", "parent-procedure"])
    picture.update(conditional)
    before_payload, before_scope = copy.deepcopy(payload), copy.deepcopy(answer._verified_scope)
    wire, _, reverse = prepare_illustrations(payload, "g1", verification, answer)
    compact = compact_illustration_wire(wire)
    assert len(json.dumps(wire, ensure_ascii=False)) > MAX_SELECTION_PAYLOAD_CHARS
    assert len(json.dumps(compact, ensure_ascii=False)) < MAX_SELECTION_PAYLOAD_CHARS
    assert payload == before_payload and answer._verified_scope == before_scope
    assert compact["selected_answer_refs"] == wire["selected_answer_refs"] and compact["context"] == wire["context"]
    assert [row.get("text") for row in compact["source_sections"].values()] == [
        row.get("text") for row in wire["source_sections"].values()
    ]
    assert {key: row["caption"] for key, row in compact["image_candidates"].items()} == {
        key: row["caption"] for key, row in wire["image_candidates"].items()
    }
    first = next(iter(compact["source_sections"].values()))["scope"]
    photo_id = compact["image_candidates"][next(iter(reverse))]["section_id"]
    photo = compact["source_sections"][photo_id]["scope"]
    for field in conditional:
        assert first[field] == photo[field]
    assert first["scope_group_ids"] == [first["group_id"], first["parent_group_id"]]
    assert first["exclusive_parent_group_id"] == first["parent_group_id"]
    assert not {"id", "evidence_ids", "step_ids", "image_ids", "ordinal"} & set(first)
    assert list(compact["image_candidates"]) == list(reverse)
    model, calls = model_with_reply()
    model._instruction_illustrations(payload, "g1", verification, answer)
    assert json.loads(calls[0]["messages"][-1]["content"]) == compact


def test_still_oversize_optional_text_preserves_the_same_valid_answer():
    payload, verification, answer = example(body_size=6000)
    wire = prepare_illustrations(payload, "g1", verification, answer)[0]
    assert len(json.dumps(compact_illustration_wire(wire), ensure_ascii=False)) > MAX_SELECTION_PAYLOAD_CHARS
    model, calls = model_with_reply(lambda *args: pytest.fail("Optional refinement does not fit"))
    assert model._instruction_illustrations(payload, "g1", verification, answer) is answer
    assert calls == []


def test_foreign_image_references_still_fail_after_only_one_repair():
    payload, verification, answer = example()
    model, calls = model_with_reply(lambda *args: {"illustration_refs": ["foreign-image-alias"]})
    with pytest.raises(ValueError, match="InvalidInstructionVerificationIllustration"):
        model._instruction_illustrations(payload, "g1", verification, answer)
    assert len(calls) == 2


def test_source_binding_is_checked_before_any_optional_fallback():
    payload, verification, answer = example(body_size=6000)
    answer._verified_scope["source"]["sha256"] = "b" * 64
    model, calls = model_with_reply()
    with pytest.raises(ValueError, match="InvalidInstructionVerifiedScope"):
        model._instruction_illustrations(payload, "g1", verification, answer)
    assert calls == []


def test_canonical_answer_cap_does_not_silently_drop_valid_image_references():
    payload, verification, answer = example(refs=95)
    reverse = prepare_illustrations(payload, "g1", verification, answer)[2]
    with pytest.raises(ValueError, match="InstructionVerificationAnswerTooLarge"):
        resolve_illustrations(InstructionIllustrationSelection(illustration_refs=list(reverse)),
                              payload, "g1", reverse, answer)


def test_original_image_material_is_outside_optional_text_refinement():
    payload, verification, answer = example(body_size=6000)
    verification.required_material_kind = "ORIGINAL_IMAGE"
    model, calls = model_with_reply(lambda *args: pytest.fail("Core material must not enter optional picker"))
    assert model._instruction_illustrations(payload, "g1", verification, answer) is answer
    assert calls == []
    core = InstructionVerification(
        source_finding="The operation requires an original graphical material.",
        applicability_refs=["e1"], operation_refs=["e1"], condition_match="EXACT",
        required_material_kind="ORIGINAL_IMAGE", material_refs=["e1"], material_state="READY",
        all_materials_available=True, verdict="PASS", operation_match="EXACT", goal_kind="LOOKUP", answer_refs=["e1"],
    )
    with pytest.raises(ValueError, match="MissingInstructionVerificationOriginalImage"):
        resolve_verification(core, payload, "g1", prepare_reading(payload)[2], answer)
