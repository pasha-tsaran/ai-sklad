"""Shared original material is delivered once across goals without losing proof or progress."""

import copy
import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.services.chat.instruction_plan import _units, render_plan, restore_plan, serialize_plan
from app.services.rag.catalog import instruction_scope_guard, instruction_sections
from app.services.rag.documents import Illustration
from app.services.rag.interface import Candidate
from app.services.rag.talk import ApprovedTalkDocument, InstructionPlanPart, KnowledgeUnavailable


COMMON = "Сохраните общий параметр перед выполнением двух операций."
FIRST = "Выполните первую отдельную операцию до перехода к следующей."
SECOND = "Проверьте результат второй отдельной операции после первой."
WARNING = "Предупреждение: Не отключайте питание во время выполнения операций."
CAPTIONS = [f"Оригинальная схема этапа номер {number}, с его отдельными пояснениями." for number in range(1, 5)]
CONTINUE = "Когда выполните показанные действия, напишите «Продолжить» — покажу следующие шаги."


def parts_fixture(*, same_pixels=False, include_warning=False):
    paragraphs = ([WARNING] if include_warning else []) + [COMMON, FIRST, *CAPTIONS[:3], SECOND, CAPTIONS[3]]
    content = "\n\n".join(paragraphs)
    source = {"file_id": "same-file", "filename": "Две операции.md", "knowledge_id": "approved",
              "sha256": hashlib.sha256(content.encode()).hexdigest()}
    pictures = {number: Illustration(f"image{number}.png", caption, "image/png",
                                    b"same-pixels" if same_pixels else f"original-pixels-{number}".encode())
                for number, caption in enumerate(CAPTIONS, 1)}
    document = {"source": source, "content": content,
                "images": [{"number": number, "caption": picture.caption} for number, picture in pictures.items()]}
    approved = ApprovedTalkDocument(document, pictures, Candidate("solution", "Две операции", 1, "chunk", content, [], 1.0))
    sections = instruction_sections(document)

    def part(goal, texts):
        selected = tuple(section for section in sections if section["text"] in texts)
        return InstructionPlanPart(approved, (goal,), (goal + " requested",), tuple(section["id"] for section in selected), selected)

    shared = {COMMON, WARNING} if include_warning else {COMMON}
    first = part("g1", {*shared, FIRST, *CAPTIONS[:3]})
    second = part("g2", {*shared, SECOND, *CAPTIONS})
    return approved, (first, second)


def deliver_all(parts, saved):
    pages, more = [], True
    while more:
        blocks, sources, saved, more = render_plan(parts, saved)
        pages.append((blocks, sources, copy.deepcopy(saved)))
    return pages, saved


def test_two_goals_same_original_keep_unique_sections_photos_and_continue_after_restore():
    approved, parts = parts_fixture()
    original = copy.deepcopy(parts)
    saved = serialize_plan(parts)
    assert saved["version"] == 3
    assert [row["section_ids"] for row in saved["parts"]] == [list(part.section_ids) for part in parts]
    knowledge = SimpleNamespace(read=lambda file_id: approved)
    restored = restore_plan(knowledge, saved)
    first, sources, saved, more = render_plan(restored, saved)
    assert more and saved["cursor"] == 3
    assert [picture for _, picture in first if picture] == [approved.illustrations[number] for number in (1, 2, 3)]
    repeated, _, unchanged, _ = render_plan(restored, saved, repeat=True)
    assert repeated == first and unchanged == saved
    following, next_sources, saved, more = render_plan(restore_plan(knowledge, saved), saved)
    assert not more and saved["cursor"] == 4
    assert [picture for _, picture in following if picture] == [approved.illustrations[4]]
    all_blocks = first[:-1] + following[:-1]
    text = "\n".join(value for value, _ in all_blocks)
    assert all(text.count(value) == 1 for value in (COMMON, FIRST, SECOND, *CAPTIONS))
    assert all(source == approved.document["source"] for source in sources[:-1] + next_sources[:-1])
    assert parts == original
    assert serialize_plan(restored)["digest"] == serialize_plan(parts)["digest"]


@pytest.mark.parametrize("version", [1, 2])
def test_saved_plan_versions_keep_their_frozen_units_and_midstream_cursor(version):
    approved, parts = parts_fixture()
    # Frozen delivery units for the persisted formats: these include the three
    # repeated original images rather than adopting the new-plan dedup rule.
    expected_units = [
        (approved, "\n\n".join((COMMON, FIRST, CAPTIONS[0])), [1]),
        (approved, CAPTIONS[1], [2]),
        (approved, CAPTIONS[2], [3]),
        (approved, "\n\n".join((COMMON, CAPTIONS[0])), [1]),
        (approved, CAPTIONS[1], [2]),
        (approved, CAPTIONS[2], [3]),
        (approved, "\n\n".join((SECOND, CAPTIONS[3])), [4]),
    ]
    assert _units(parts, version=version) == expected_units
    saved = serialize_plan(parts, version=version)
    saved.update(cursor=3, shown=[0, 1, 2])
    restored = restore_plan(SimpleNamespace(read=lambda file_id: approved), saved)
    following, sources, updated, more = render_plan(restored, saved)
    assert following == [
        ("\n\n".join((COMMON, CAPTIONS[0])), None), ("", approved.illustrations[1]),
        (CAPTIONS[1], None), ("", approved.illustrations[2]),
        (CAPTIONS[2], None), ("", approved.illustrations[3]), (CONTINUE, None),
    ]
    assert sources == [approved.document["source"]] * 6 + [None]
    assert more and updated["cursor"] == 6 and updated["shown"] == [3, 4, 5]
    repeated, _, unchanged, more = render_plan(restored, saved, repeat=True)
    assert repeated == [
        ("\n\n".join((COMMON, FIRST, CAPTIONS[0])), None), ("", approved.illustrations[1]),
        (CAPTIONS[1], None), ("", approved.illustrations[2]),
        (CAPTIONS[2], None), ("", approved.illustrations[3]), (CONTINUE, None),
    ]
    assert more and unchanged == saved


def test_lookup_and_action_have_separate_copy_domains():
    _, parts = parts_fixture()
    lookup = replace(parts[0], delivery_mode="EXCERPT")
    action = replace(parts[0], goal_ids=("g2",), goals=("Perform",))
    rendered, _ = deliver_all([lookup, action], serialize_plan([lookup, action]))
    pictures = [picture for blocks, _, _ in rendered for _, picture in blocks if picture]
    assert len(pictures) == 6


def test_same_pixels_at_distinct_positions_inside_one_original_part_are_preserved():
    approved, parts = parts_fixture(same_pixels=True)
    rendered, _ = deliver_all([parts[0]], serialize_plan([parts[0]]))
    pictures = [picture for blocks, _, _ in rendered for _, picture in blocks if picture]
    assert pictures == [approved.illustrations[number] for number in (1, 2, 3)]


def test_identical_pixels_in_a_later_distinct_owner_keep_its_unique_caption_once():
    approved, parts = parts_fixture(same_pixels=True)
    rendered, _ = deliver_all(parts, serialize_plan(parts))
    pictures = [picture for blocks, _, _ in rendered for _, picture in blocks if picture]
    text = "\n".join(value for blocks, _, _ in rendered for value, _ in blocks[:-1])
    assert pictures == [approved.illustrations[number] for number in (1, 2, 3)]
    assert text.count(CAPTIONS[3]) == 1 and text.count(SECOND) == 1


def test_different_source_hash_preserves_the_other_original_and_its_provenance():
    approved, parts = parts_fixture()
    other_document = copy.deepcopy(approved.document)
    other_document["content"] += "\n\nДругой действующий оригинал."
    other_document["source"].update(file_id="other-file", sha256=hashlib.sha256(other_document["content"].encode()).hexdigest())
    other = replace(approved, document=other_document)
    later = replace(parts[1], approved=other)
    rendered, _ = deliver_all([parts[0], later], serialize_plan([parts[0], later]))
    pictures = [picture for blocks, _, _ in rendered for _, picture in blocks if picture]
    assert len(pictures) == 7
    assert any(source == other_document["source"] for _, sources, _ in rendered for source in sources)


def test_missing_original_image_is_still_rejected_before_duplicate_suppression():
    approved, parts = parts_fixture()
    missing = replace(approved, illustrations={number: image for number, image in approved.illustrations.items() if number != 1})
    later = replace(parts[1], approved=missing)
    with pytest.raises(KnowledgeUnavailable, match="InstructionPlanImageMissing"):
        render_plan([parts[0], later], serialize_plan([parts[0], later]))


def test_shared_mandatory_warning_stays_in_both_verified_scopes_and_is_delivered_once():
    approved, parts = parts_fixture(include_warning=True)
    warning = next(section for section in parts[0].sections if section["text"] == WARNING)
    assert warning["kind"] == "WARNING"
    assert all(instruction_scope_guard(warning, {section["group_id"] for section in part.sections}) for part in parts)
    saved = serialize_plan(parts)
    assert all(warning["id"] in row["section_ids"] for row in saved["parts"])
    restored = restore_plan(SimpleNamespace(read=lambda file_id: approved), saved)
    assert all(warning in part.sections for part in restored)
    rendered, _ = deliver_all(restored, saved)
    text = "\n".join(value for blocks, _, _ in rendered for value, _ in blocks[:-1])
    assert text.count(WARNING) == 1
    assert all(warning["id"] in row["section_ids"] for row in saved["parts"])


def test_same_source_bytes_with_different_file_metadata_keep_both_canonical_proofs():
    approved, parts = parts_fixture()
    other_document = copy.deepcopy(approved.document)
    other_document["source"].update(file_id="other-file", filename="Копия двух операций.md", knowledge_id="other-approved")
    other = replace(approved, document=other_document)
    selected_texts = {section["text"] for section in parts[1].sections}
    other_sections = tuple(section for section in instruction_sections(other_document) if section["text"] in selected_texts)
    later = replace(parts[1], approved=other, section_ids=tuple(section["id"] for section in other_sections), sections=other_sections)
    assert other_document["source"]["sha256"] == approved.document["source"]["sha256"]
    assert not set(later.section_ids).intersection(parts[0].section_ids)
    selected = [parts[0], later]
    saved = serialize_plan(selected)
    assert [row["source"] for row in saved["parts"]] == [approved.document["source"], other_document["source"]]
    originals = {item.document["source"]["file_id"]: item for item in (approved, other)}
    restored = restore_plan(SimpleNamespace(read=originals.__getitem__), saved)
    assert [part.section_ids for part in restored] == [part.section_ids for part in selected]
    assert serialize_plan(restored)["parts"] == saved["parts"]
    rendered, _ = deliver_all(restored, saved)
    delivered_photos = [(picture, source) for blocks, sources, _ in rendered
                        for (_, picture), source in zip(blocks, sources, strict=True) if picture]
    assert [picture for picture, _ in delivered_photos] == [approved.illustrations[number] for number in (1, 2, 3, 4)]
    assert [source["file_id"] for _, source in delivered_photos] == ["same-file"] * 3 + ["other-file"]
    assert any(source == other_document["source"] for _, sources, _ in rendered for source in sources)
