"""A focused reader selects closed, original-bound evidence for every user goal."""

import copy
import json

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.services.llm.instruction_reader import InstructionReading, prepare_reading, resolve_reading
from app.services.llm.openwebui import OpenWebUILLM


def reading_payload():
    return {
        "source": {"file_id": "source-private-id", "filename": "Новая инструкция.docx", "sha256": "a" * 64},
        "requested_goals": {"width": "Узнать ширину", "start": "Выполнить проверку"},
        "requested_goal_kinds": {"width": "LOOKUP", "start": "ACTION"},
        "context": {"current_message": "Какая ширина и как запустить проверку?"},
        "candidate_evidence": {
            "width-original-reference": {"text": "Ширина составляет 70 мм.", "support_kind": "TEXT", "image_numbers": []},
            "photo-original-reference": {"text": "Окно проверки рабочего устройства.", "support_kind": "IMAGE",
                                         "image_numbers": [2]},
        },
        "candidate_sections": {
            "width-private-section": {"title": "Размеры", "group_id": "private-dimensions", "kind": "TEXT",
                                      "evidence_ids": ["width-original-reference"], "image_numbers": []},
            "photo-private-section": {"title": "Проверка", "group_id": "private-test", "kind": "IMAGE",
                                      "evidence_ids": ["photo-original-reference"], "image_numbers": [2]},
        },
    }


def model_reader(responder):
    model = OpenWebUILLM(Settings(_env_file=None, llm_backend="openwebui", openwebui_api_key="test-key"))
    calls = []

    def request(messages, *, selection_schema=None, **options):
        wire = json.loads(messages[-1]["content"])
        calls.append({"wire": wire, "schema": selection_schema, "options": options})
        return json.dumps(responder(wire, len(calls)), ensure_ascii=False)

    model._request = request
    # Reader-format tests isolate this stage. The full two-stage adapter is
    # exercised by test_instruction_verification without this replacement.
    model.verify_instruction = lambda payload, reading: reading
    return model, calls


def wire_evidence(wire, needle, *, exact=False):
    return [identity for section in wire["approved_original"]["sections"]
            for identity, text in section["evidence"].items()
            if text == needle or not exact and needle in text]


def valid_reply():
    return {"goals": {"width": wire_goal("SUPPORTED", ["e1"], "EXCERPT"),
                      "start": wire_goal("SUPPORTED", ["e2"], "PROCEDURE")}}


def wire_goal(status, references=(), mode=None):
    return {"source_finding": "В оригинале проверены сведения для этой цели.",
            "evidence_ids": list(references), "status": status, "delivery_mode": mode,
            "question": "", "clarification_kind": None}


def test_reading_schema_has_one_closed_required_slot_for_each_exact_goal():
    wire, schema, reverse = prepare_reading(reading_payload())
    goal_object = schema["properties"]["goals"]
    assert set(goal_object["required"]) == {"width", "start"}
    assert goal_object["additionalProperties"] is False and schema["additionalProperties"] is False
    for goal, expected_mode in (("width", "EXCERPT"), ("start", "PROCEDURE")):
        slot = goal_object["properties"][goal]
        branches = {branch["properties"]["status"]["const"]: branch for branch in slot["anyOf"]}
        assert set(branches) == {"SUPPORTED", "MISSING", "RELATED", "CLARIFY"}
        for branch in branches.values():
            assert branch["additionalProperties"] is False
            assert branch["required"] == ["source_finding", "status", "evidence_ids", "delivery_mode", "question",
                                           "clarification_kind"]
        fields = branches["SUPPORTED"]["properties"]
        assert fields["delivery_mode"]["enum"] == [expected_mode]
        assert fields["evidence_ids"]["items"] == {"$ref": "#/$defs/Evidence"}
        assert schema["$defs"]["Evidence"]["enum"] == list(reverse)
        assert fields["evidence_ids"]["uniqueItems"] is True
        assert fields["evidence_ids"]["minItems"] == 1
        assert fields["source_finding"]["maxLength"] == 400
    serialized = json.dumps(wire)
    assert "source-private-id" not in serialized and "width-private-section" not in serialized
    assert "width-original-reference" not in serialized and "private-dimensions" not in serialized
    assert wire["approved_original"]["sections"][1]["images"] == [2]


def test_resolving_aliases_keeps_the_original_canonical_photo_and_text_bindings():
    payload = reading_payload()
    unchanged = copy.deepcopy(payload)
    wire, _, reverse = prepare_reading(payload)
    assert wire["approved_original"]["sections"][0]["evidence"] == {"e1": "Ширина составляет 70 мм."}
    resolved = resolve_reading(InstructionReading.model_validate(valid_reply()), payload, reverse)
    assert resolved.goals["width"].evidence_ids == ["width-original-reference"]
    assert resolved.goals["start"].evidence_ids == ["photo-original-reference"]
    assert payload == unchanged


@pytest.mark.parametrize("change", [
    lambda reply: reply["goals"].pop("start"),
    lambda reply: reply["goals"].update(unknown={"status": "MISSING"}),
    lambda reply: reply["goals"]["width"].update(evidence_ids=[]),
    lambda reply: reply["goals"]["width"].update(evidence_ids=["e999"]),
    lambda reply: reply["goals"]["width"].update(evidence_ids=["e1", "e1"]),
    lambda reply: reply["goals"]["width"].update(delivery_mode="PROCEDURE"),
    lambda reply: reply["goals"]["start"].update(delivery_mode="EXCERPT"),
    lambda reply: reply["goals"].update(width={"status": "MISSING", "evidence_ids": ["e1"]}),
    lambda reply: reply["goals"].update(width={"status": "RELATED"}),
    lambda reply: reply["goals"]["width"].update(question="Какое устройство?"),
    lambda reply: reply["goals"].update(width={"status": "CLARIFY", "evidence_ids": ["e1"],
                                             "clarification_kind": "VISIBLE_STATE"}),
    lambda reply: reply["goals"].update(width={"status": "CLARIFY", "evidence_ids": ["e1"],
                                             "question": "Какое окно видно?"}),
])
def test_invalid_per_goal_outcomes_and_unknown_references_are_rejected(change):
    payload = reading_payload()
    _, _, reverse = prepare_reading(payload)
    reply = valid_reply()
    change(reply)
    with pytest.raises(ValueError):
        resolve_reading(InstructionReading.model_validate(reply), payload, reverse)


@pytest.mark.parametrize("field,value", [("image_numbers", [99]), ("filename", "other.png"),
                                         ("evidence", "invented quote"), ("section_id", "other-section")])
def test_provider_cannot_supply_arbitrary_images_sections_or_quotes(field, value):
    reply = valid_reply()
    reply["goals"]["width"][field] = value
    with pytest.raises(ValidationError):
        InstructionReading.model_validate(reply)


def test_clarification_requires_original_grounding_and_can_be_disabled():
    payload = reading_payload()
    payload["allow_clarification"] = False
    _, schema, reverse = prepare_reading(payload)
    statuses = [branch["properties"]["status"]["const"]
                for branch in schema["properties"]["goals"]["properties"]["width"]["anyOf"]]
    assert "CLARIFY" not in statuses
    reply = valid_reply()
    reply["goals"]["width"] = {"status": "CLARIFY", "evidence_ids": ["e1"],
                                "question": "Какое окно вы видите?", "clarification_kind": "VISIBLE_STATE"}
    with pytest.raises(ValueError, match="InvalidInstructionReadingQuestion"):
        resolve_reading(InstructionReading.model_validate(reply), payload, reverse)


@pytest.mark.parametrize("mutate", [
    lambda payload: payload["candidate_sections"].pop("width-private-section"),
    lambda payload: payload["candidate_sections"]["photo-private-section"]["evidence_ids"].append(
        "width-original-reference"),
    lambda payload: payload["candidate_evidence"]["photo-original-reference"].update(image_numbers=[True]),
    lambda payload: payload["candidate_evidence"]["width-original-reference"].update(image_numbers=[1]),
])
def test_input_registry_rejects_unbound_duplicate_and_noncanonical_proofs(mutate):
    payload = reading_payload()
    mutate(payload)
    with pytest.raises(ValueError):
        prepare_reading(payload)


def test_adapter_reads_all_goals_through_short_aliases_and_preserves_every_chosen_reference():
    model, calls = model_reader(lambda wire, number: valid_reply())
    resolved = model.read_instruction(reading_payload())
    assert len(calls) == 1
    assert calls[0]["options"]["max_output_tokens"] == 1536
    assert resolved.goals["start"].evidence_ids == ["photo-original-reference"]
    assert calls[0]["wire"]["requested_goals"] == reading_payload()["requested_goals"]


def test_adapter_repairs_a_bad_reference_once_using_the_same_closed_original():
    def reply(wire, number):
        response = valid_reply()
        if number == 1:
            response["goals"]["width"]["evidence_ids"] = ["e999"]
        return response

    model, calls = model_reader(reply)
    result = model.read_instruction(reading_payload())
    assert len(calls) == 2 and "repair_reason" in calls[1]["wire"]
    assert calls[0]["wire"]["approved_original"] == calls[1]["wire"]["approved_original"]
    assert result.goals["width"].evidence_ids == ["width-original-reference"]


def test_adapter_stops_after_two_invalid_replies_without_authorizing_any_source():
    model, calls = model_reader(lambda wire, number: {"goals": {"width": {"status": "MISSING"}}})
    with pytest.raises(ValueError, match="InvalidInstructionReadingGoals"):
        model.read_instruction(reading_payload())
    assert len(calls) == 2


def test_source_finding_is_bounded_while_legacy_callbacks_can_leave_it_empty():
    reply = valid_reply()
    reply["goals"]["width"]["source_finding"] = "x" * 401
    with pytest.raises(ValidationError):
        InstructionReading.model_validate(reply)
    legacy = {"goals": {"width": {"status": "MISSING"}, "start": {"status": "MISSING"}}}
    assert InstructionReading.model_validate(legacy).goals["width"].source_finding == ""
