"""Catalogue nominations can request unread originals without supplying an answer."""

import copy
import json
from types import SimpleNamespace

import httpx
import pytest

from app.services.llm.instruction_discovery import (
    MAX_DISCOVERY_PAYLOAD_CHARS, InstructionNomination, prepare_discovery, resolve_discovery,
)
from tests.test_instruction_reader import model_reader


def catalogue_payload(count=3):
    files, entries = {}, {}
    for index in range(count):
        identity = "approved-file-" + str(index)
        filename = f"Руководство страницы {index}.docx"
        files[identity] = {"filename": filename}
        entries[identity] = SimpleNamespace(
            source={"file_id": identity, "filename": filename},
            sections=({"text": "Страница PDF 1", "is_heading": True},
                      {"text": "## Размеры по умолчанию", "is_heading": True},
                      {"text": "## Проверка связи", "is_heading": True}),
            registry={"p1": {"text": "Общее введение в работу устройства."},
                      "p2": {"text": "Стандартная ширина составляет 70 мм."},
                      "p3": {"text": "Стандартная высота составляет 40 мм."}},
        )
    return {"requested_goals": {"g1": "Узнать стандартные ширину и высоту"},
            "goal_search_queries": {"g1": ["label default width height"]},
            "files": files, "entries": entries, "eligible_file_ids": list(files)[1:],
            "context": {"current_message": "Мне нужны значения из справочника."}}


def test_nomination_wire_contains_only_eligible_aliases_all_chapters_and_ranked_original_fragments():
    payload = catalogue_payload()
    original = copy.deepcopy(payload)
    wire, schema, reverse = prepare_discovery(payload)
    assert reverse == {"d1": "approved-file-1", "d2": "approved-file-2"}
    assert set(wire["catalogue"]) == set(reverse)
    assert all(card["headings"] == ["Размеры по умолчанию", "Проверка связи"] for card in wire["catalogue"].values())
    assert all(len(card["fragments"]) == 2 for card in wire["catalogue"].values())
    assert all(any("ширина" in text or "высота" in text for text in card["fragments"])
               for card in wire["catalogue"].values())
    assert "approved-file-0" not in json.dumps(wire) and "approved-file-1" not in json.dumps(wire)
    assert schema["additionalProperties"] is False and schema["required"] == ["file_ids"]
    assert schema["properties"]["file_ids"]["items"]["enum"] == ["d1", "d2"]
    assert schema["properties"]["file_ids"]["uniqueItems"] is True
    assert payload == original


def test_unindexed_or_renamed_source_keeps_only_current_approved_filename():
    payload = catalogue_payload()
    payload["entries"].pop("approved-file-1")
    payload["entries"]["approved-file-2"].source["filename"] = "Старое имя.docx"
    wire, _, _ = prepare_discovery(payload)
    assert all(not card["indexed"] and not card["headings"] and not card["fragments"]
               for card in wire["catalogue"].values())
    assert [card["filename"] for card in wire["catalogue"].values()] == [
        payload["files"][identity]["filename"] for identity in payload["eligible_file_ids"]]


def test_all_fifty_eligible_sources_remain_visible_within_the_discovery_budget():
    payload = catalogue_payload(count=51)
    wire, _, reverse = prepare_discovery(payload)
    assert len(wire["catalogue"]) == len(reverse) == 50
    assert len(json.dumps(wire, ensure_ascii=False)) <= MAX_DISCOVERY_PAYLOAD_CHARS
    assert all(len(card["headings"]) == 2 and len(card["fragments"]) == 2 for card in wire["catalogue"].values())


def test_oversize_chapter_catalogue_is_reported_without_silently_omitting_sources():
    payload = catalogue_payload()
    payload["entries"]["approved-file-1"].sections = ({"is_heading": True, "text": "Название " + "x" * 50_000},)
    with pytest.raises(ValueError, match="InstructionDiscoveryPayloadTooLarge"):
        prepare_discovery(payload)


@pytest.mark.parametrize("eligible", [["unknown-file"], ["approved-file-1", "approved-file-1"]])
def test_unknown_or_duplicate_eligible_sources_are_rejected(eligible):
    payload = catalogue_payload()
    payload["eligible_file_ids"] = eligible
    with pytest.raises(ValueError, match="InvalidInstructionDiscoveryInput"):
        prepare_discovery(payload)


@pytest.mark.parametrize("identities", [["d1", "d1"], ["d999"], ["approved-file-1"]])
def test_nomination_cannot_return_duplicate_unknown_or_unaliased_source_ids(identities):
    _, _, reverse = prepare_discovery(catalogue_payload())
    with pytest.raises(ValueError, match="InvalidInstructionNomination"):
        resolve_discovery(InstructionNomination(file_ids=identities), reverse)


def test_nomination_has_no_answer_question_quote_or_image_authority():
    with pytest.raises(ValueError):
        InstructionNomination.model_validate({"file_ids": ["d1"], "answer": "Invented answer"})
    with pytest.raises(ValueError):
        InstructionNomination(file_ids=["d1", "d2", "d3", "d4"])


def test_adapter_returns_canonical_eligible_ids_with_a_256_token_discovery_request():
    model, calls = model_reader(lambda wire, number: {"file_ids": ["d2"]})
    result = model.nominate_instructions(catalogue_payload())
    assert result.file_ids == ["approved-file-2"] and len(calls) == 1
    assert calls[0]["options"]["max_output_tokens"] == 256
    assert "approved_original" not in calls[0]["wire"] and "catalogue" in calls[0]["wire"]


def test_empty_eligible_catalogue_avoids_a_model_call():
    payload = catalogue_payload()
    payload["eligible_file_ids"] = []
    model, calls = model_reader(lambda wire, number: pytest.fail("No eligible originals"))
    assert model.nominate_instructions(payload).file_ids == [] and not calls


def test_invalid_nomination_is_repaired_once_without_expanding_eligibility():
    model, calls = model_reader(lambda wire, number: {"file_ids": ["d999" if number == 1 else "d1"]})
    assert model.nominate_instructions(catalogue_payload()).file_ids == ["approved-file-1"]
    assert len(calls) == 2 and "repair_reason" in calls[1]["wire"]
    assert calls[1]["wire"]["catalogue"] == calls[0]["wire"]["catalogue"]


def competing_originals():
    payload = catalogue_payload()
    payload["requested_goal_kinds"] = {"g1": "LOOKUP"}
    for entry in payload["entries"].values():
        entry.content = "Значения параметров. " * 210
        entry.sections = ({"id": "s1", "kind": "TEXT", "text": "Таблица параметров", "is_heading": True,
                           "image_numbers": []},)
    return payload


@pytest.mark.parametrize("order, expected", [
    (["d2", "d1"], ["approved-file-2", "approved-file-1"]),
    (["d2", "d2"], ["approved-file-1", "approved-file-2"]),
    (["d2", "d999"], ["approved-file-1", "approved-file-2"]),
])
def test_reference_question_compares_complete_originals_with_only_ordering_authority(order, expected):
    payload = competing_originals()
    model, calls = model_reader(lambda wire, number: {"file_ids": ["d1", "d2"]} if number == 1
                               else {"file_order": order})
    assert model.nominate_instructions(payload).file_ids == expected
    assert len(calls) == 2
    comparison = calls[1]["wire"]
    assert comparison["user_evidence"] == {"u1": payload["context"]["current_message"]}
    assert comparison["candidates"]["d1"]["full_body"] == payload["entries"]["approved-file-1"].content
    assert set(comparison["candidates"]) == {"d1", "d2"}
    assert calls[1]["options"]["timeout_cap"] == 20


def test_comparison_timeout_keeps_nomination_without_retry():
    def reply(wire, number):
        if number == 2:
            raise httpx.ReadTimeout("Optional comparison timed out")
        return {"file_ids": ["d1", "d2"]}

    model, calls = model_reader(reply)
    assert model.nominate_instructions(competing_originals()).file_ids == ["approved-file-1", "approved-file-2"]
    assert len(calls) == 2


@pytest.mark.parametrize("skip_reason", ["action", "unindexed", "diagnostic", "oversize"])
def test_comparison_keeps_existing_nomination_when_inapplicable(skip_reason):
    payload = competing_originals()
    if skip_reason == "action":
        payload["requested_goal_kinds"] = {"g1": "ACTION"}
    elif skip_reason == "unindexed":
        payload["entries"].pop("approved-file-1")
    elif skip_reason == "diagnostic":
        payload["context"]["diagnostic_request"] = True
    else:
        payload["entries"]["approved-file-1"].content = "x" * 50_001
    model, calls = model_reader(lambda wire, number: {"file_ids": ["d1", "d2"]})
    assert model.nominate_instructions(payload).file_ids == ["approved-file-1", "approved-file-2"]
    assert len(calls) == 1
