"""Wire compaction preserves every original fragment and canonical owner."""

import copy
import json

import pytest

from app.services.llm.instruction_reader import InstructionReading, prepare_reading, resolve_reading
from tests.test_instruction_reader import wire_goal


def payload_with_sections(count=3):
    registry = {}
    sections = {}
    for number in range(1, count + 1):
        reference = f"original-reference-{number}"
        registry[reference] = {"text": f"Исходная строка {number:04} " + "а" * 19,
                               "support_kind": "TEXT", "image_numbers": []}
        sections[f"original-section-{number}"] = {
            "title": "Сведения", "group_id": "original-procedure", "kind": "TEXT",
            "evidence_ids": [reference], "image_numbers": [],
        }
    return {"source": {"file_id": "original-source", "filename": "Новый справочник.docx"},
            "requested_goals": {"g1": "Узнать значение"}, "requested_goal_kinds": {"g1": "LOOKUP"},
            "context": {}, "candidate_evidence": registry, "candidate_sections": sections}


def flattened(wire):
    return [(reference, text) for row in wire["approved_original"]["sections"]
            for reference, text in row["evidence"].items()]


def test_adjacent_equivalent_rows_keep_every_reference_text_and_canonical_owner_in_order():
    payload = payload_with_sections()
    original = copy.deepcopy(payload)
    wire, schema, reverse = prepare_reading(payload)
    assert len(wire["approved_original"]["sections"]) == 1
    assert flattened(wire) == [(f"e{number}", proof["text"])
                               for number, proof in enumerate(payload["candidate_evidence"].values(), 1)]
    assert reverse == {f"e{number}": reference
                       for number, reference in enumerate(payload["candidate_evidence"], 1)}
    choice = InstructionReading.model_validate({"goals": {"g1": wire_goal("SUPPORTED", ["e2"], "EXCERPT")}})
    resolved = resolve_reading(choice, payload, reverse)
    assert resolved.goals["g1"].evidence_ids == ["original-reference-2"]
    assert payload["candidate_sections"]["original-section-2"]["evidence_ids"] == ["original-reference-2"]
    supported = schema["properties"]["goals"]["properties"]["g1"]["anyOf"][0]
    assert supported["properties"]["delivery_mode"]["enum"] == ["EXCERPT"]
    assert payload == original


@pytest.mark.parametrize("metadata", [
    {"group_id": "another-procedure"}, {"kind": "WARNING"},
    {"condition_text": "Если принтер подключён по Wi-Fi"},
    {"exclusive_set_id": "another-alternative-group"},
])
def test_different_scope_metadata_keeps_rows_separate(metadata):
    payload = payload_with_sections(2)
    payload["candidate_sections"]["original-section-2"].update(metadata)
    wire, _, _ = prepare_reading(payload)
    assert len(wire["approved_original"]["sections"]) == 2
    assert [reference for reference, _ in flattened(wire)] == ["e1", "e2"]


def image_payload(second_image):
    payload = payload_with_sections(2)
    for number, image in ((1, 1), (2, second_image)):
        payload["candidate_sections"][f"original-section-{number}"].update(kind="IMAGE", image_numbers=[image])
        payload["candidate_evidence"][f"original-reference-{number}"].update(support_kind="IMAGE", image_numbers=[image])
    return payload


def test_matching_image_rows_keep_pixels_identifiers_and_each_caption():
    payload = image_payload(1)
    original = copy.deepcopy(payload)
    wire, _, reverse = prepare_reading(payload)
    assert len(wire["approved_original"]["sections"]) == 1
    assert wire["approved_original"]["sections"][0]["images"] == [1]
    assert wire["approved_original"]["available_original_images"] == [1]
    assert len(flattened(wire)) == len(reverse) == 2 and payload == original


def test_different_original_images_are_never_combined():
    wire, _, _ = prepare_reading(image_payload(2))
    assert [row["images"] for row in wire["approved_original"]["sections"]] == [[1], [2]]
    assert wire["approved_original"]["available_original_images"] == [1, 2]


def test_nonadjacent_equivalent_rows_do_not_move_across_a_warning():
    payload = payload_with_sections()
    payload["candidate_sections"]["original-section-2"]["kind"] = "WARNING"
    wire, _, _ = prepare_reading(payload)
    assert [row["kind"] for row in wire["approved_original"]["sections"]] == ["TEXT", "WARNING", "TEXT"]
    assert [reference for reference, _ in flattened(wire)] == ["e1", "e2", "e3"]


def test_compaction_does_not_change_the_response_grammar_or_evidence_aliases():
    compact = payload_with_sections(2)
    separate = copy.deepcopy(compact)
    separate["candidate_sections"]["original-section-2"]["group_id"] = "another-procedure"
    compact_wire, compact_schema, compact_reverse = prepare_reading(compact)
    separate_wire, separate_schema, separate_reverse = prepare_reading(separate)
    assert compact_schema == separate_schema and compact_reverse == separate_reverse
    assert flattened(compact_wire) == flattened(separate_wire)


def test_a_line_wrapped_32k_original_fits_the_wire_budget_without_losing_text():
    payload = payload_with_sections(800)
    original = copy.deepcopy(payload)
    original_text = "".join(proof["text"] for proof in payload["candidate_evidence"].values())
    wire, _, reverse = prepare_reading(payload)
    assert 30000 <= len(original_text) <= 33000
    assert len(json.dumps(wire, ensure_ascii=False)) < 50000
    assert "".join(text for _, text in flattened(wire)) == original_text
    assert len(flattened(wire)) == len(reverse) == 800
    assert payload == original
