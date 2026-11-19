"""Canonical image proof can cover its labeled caption in the real plan path."""

import copy
import io
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import pytest

from app.services.chat.instruction_plan import restore_plan, serialize_plan
from app.services.llm.instruction_reader import InstructionReading, SourceGoalReading
from app.services.rag.talk import KnowledgeUnavailable, _caption_proof_matches_image
from integrations.openwebui.approved_instruction_images import W, read_docx_images
from tests.test_openwebui_instruction_images import instruction, office
from tests.test_scanner_original_retrieval import scanner_provider
from tests.test_talk_knowledge import DOC


STATE = "Состояние индикатора: свет включён или ожидает данные."
CONTROL = "Кнопка переключает режим ожидания."
CAPTION = STATE + " " + CONTROL
GOAL = "Узнать состояние индикатора"


def gallery_source(caption_text=None, *, image_caption=CAPTION, second_gallery=False):
    captions = [image_caption, image_caption] if second_gallery else [image_caption]
    with ZipFile(io.BytesIO(instruction(captions))) as archive:
        parts = {name: archive.read(name) for name in archive.namelist()}
    root = ET.fromstring(parts["word/document.xml"])
    body = root.find(W + "body")
    drawings = [copy.deepcopy(paragraph) for paragraph in body if paragraph.find(".//" + W + "drawing") is not None]
    body.clear()
    first_caption = "Иллюстрация 1. " + image_caption if caption_text is None else caption_text
    for index, drawing in enumerate(drawings, 1):
        texts = ["# Профиль " + str(index), "## Состояние", STATE, "## Управление", CONTROL,
                 "## Фотографии и пояснения", first_caption if index == 1 else "Иллюстрация 2. " + image_caption]
        for text in texts:
            paragraph = ET.SubElement(body, W + "p")
            run = ET.SubElement(paragraph, W + "r")
            ET.SubElement(run, W + "t").text = text
        body.append(drawing)
    parts["word/document.xml"] = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    return office(parts)


def configured_source(data, caption_text, *, selected_image=1, select_caption=True):
    knowledge, state = scanner_provider([(DOC, "Справочник состояния.docx", data)])
    knowledge.settings = knowledge.settings.model_copy(update={"llm_backend": "openwebui"})
    captured = []

    def read(payload):
        captured.append(copy.deepcopy(payload))
        sections = payload["candidate_sections"]
        main = next(section for section in sections.values() if section["text"] == STATE)
        operation_refs = (main["evidence_ids"][0],)
        selected = list(operation_refs)
        if select_caption:
            caption = next(section for section in sections.values() if section["text"] == caption_text)
            selected.append(caption["evidence_ids"][0])
        if selected_image is not None:
            image = next(section for section in sections.values() if section["image_numbers"] == [selected_image])
            selected.append(image["evidence_ids"][0])
        answer = SourceGoalReading(status="SUPPORTED", evidence_ids=list(operation_refs), delivery_mode="EXCERPT")
        answer._verified_scope = {
            "source": copy.deepcopy(payload["source"]), "goal_id": "g1", "goal_text": GOAL,
            "goal_kind": "LOOKUP", "delivery_mode": "EXCERPT", "operation_refs": operation_refs,
            "material_refs": (), "answer_refs": tuple(selected), "confirmation_refs": operation_refs,
        }
        return InstructionReading(goals={"g1": answer})

    knowledge.reader = read
    return knowledge, state, captured


def find_information(knowledge):
    return knowledge.search_with_llm(GOAL, {"instruction_planning": True, "requested_goals": {"g1": GOAL},
                                           "requested_goal_kinds": {"g1": "LOOKUP"}})


@pytest.mark.parametrize("label", ["Иллюстрация 1. ", "Figure 1. "])
def test_prefixed_caption_with_the_same_scope_and_selected_original_image_builds_a_full_plan(label):
    caption_text = label + CAPTION
    data = gallery_source(caption_text)
    knowledge, state, captured = configured_source(data, caption_text)

    result = find_information(knowledge)

    caption = next(section for section in captured[0]["candidate_sections"].values()
                   if section["text"] == caption_text)
    image = next(section for section in captured[0]["candidate_sections"].values()
                 if section["image_numbers"] == [1])
    assert caption["scope_ambiguous"] and image["scope_ambiguous"]
    assert set(caption["scope_group_ids"]) == set(image["scope_group_ids"])
    assert result.result == "MATCH" and result.plan[0].delivery_mode == "EXCERPT"
    assert caption_text in [section["text"] for section in result.plan[0].sections]
    assert result.plan[0].approved.illustrations[1].data == read_docx_images(data)[0].data
    assert restore_plan(knowledge, serialize_plan(result.plan))[0].sections == result.plan[0].sections
    assert not any("/chat/completions" in request.url.path for request in state["calls"])


def test_different_full_caption_is_rejected_even_when_it_mentions_the_original_caption():
    caption_text = "Иллюстрация 1. " + CAPTION + " Дополнительное неподтверждённое действие."
    knowledge, _, captured = configured_source(gallery_source(caption_text), caption_text)

    with pytest.raises(KnowledgeUnavailable, match="AmbiguousInstructionImageScope"):
        find_information(knowledge)

    caption = next(section for section in captured[0]["candidate_sections"].values()
                   if section["text"] == caption_text)
    assert caption["scope_ambiguous"]


def test_full_caption_owners_match_when_their_evidence_is_split_into_fragments():
    long_caption = CAPTION + " " + (
        "Пояснение сохраняет исходные обозначения, условия чтения таблицы и различия описанных состояний. "
        "Пример относится к показанному справочному материалу и не добавляет новую процедуру."
    )
    caption_text = "Иллюстрация 1. " + long_caption
    knowledge, _, captured = configured_source(
        gallery_source(caption_text, image_caption=long_caption), caption_text,
    )

    result = find_information(knowledge)

    sections = captured[0]["candidate_sections"].values()
    caption = next(section for section in sections if section["text"] == caption_text)
    image = next(section for section in sections if section["image_numbers"] == [1])
    assert len(caption["evidence_ids"]) > 1 and len(image["evidence_ids"]) > 1
    assert caption_text in [section["text"] for section in result.plan[0].sections]


def test_identical_caption_on_an_image_with_other_scope_groups_is_rejected():
    caption_text = "Иллюстрация 1. " + CAPTION
    data = gallery_source(caption_text, second_gallery=True)
    knowledge, _, captured = configured_source(data, caption_text, selected_image=2)

    with pytest.raises(KnowledgeUnavailable, match="AmbiguousInstructionImageScope"):
        find_information(knowledge)

    sections = captured[0]["candidate_sections"].values()
    caption = next(section for section in sections if section["text"] == caption_text)
    image = next(section for section in sections if section["image_numbers"] == [2])
    assert caption["scope_ambiguous"] and image["scope_ambiguous"]
    assert set(caption["scope_group_ids"]) != set(image["scope_group_ids"])
    assert image["text"] == CAPTION


def test_a_matching_caption_without_a_selected_image_is_rejected():
    caption_text = "Иллюстрация 1. " + CAPTION
    knowledge, _, _ = configured_source(gallery_source(caption_text), caption_text, selected_image=None)

    with pytest.raises(KnowledgeUnavailable, match="AmbiguousInstructionImageScope"):
        find_information(knowledge)


def test_label_only_caption_is_rejected_with_an_actual_selected_image():
    caption_text = "Иллюстрация 1."
    # The numeric image tag also occurs in the label, so the parser associates
    # their scopes. Removing the label must not yield an empty universal match.
    knowledge, _, captured = configured_source(gallery_source(caption_text, image_caption="1"), caption_text)

    with pytest.raises(KnowledgeUnavailable, match="AmbiguousInstructionImageScope"):
        find_information(knowledge)

    caption = next(section for section in captured[0]["candidate_sections"].values()
                   if section["text"] == caption_text)
    assert caption["scope_ambiguous"]


def test_an_unprefixed_original_caption_keeps_the_existing_full_plan_path():
    knowledge, _, _ = configured_source(gallery_source(CAPTION), CAPTION, select_caption=False)

    result = find_information(knowledge)

    assert result.result == "MATCH"
    assert result.plan[0].approved.illustrations[1].caption == CAPTION


@pytest.mark.parametrize("text,expected", [
    ("свет включён или ожидает данные", True),
    ("Иллюстрация 1. " + CAPTION, True),
    ("Иллюстрация 1. " + STATE, False),
    ("Иллюстрация 1.", False),
    ("Иллюстрация 1. Другая таблица.", False),
    ("состояние индикатора: свет включён или ожидает данные.", False),
])
def test_caption_matching_keeps_exact_unprefixed_substrings_and_rejects_empty_or_loose_matches(text, expected):
    assert _caption_proof_matches_image(text, CAPTION) is expected
