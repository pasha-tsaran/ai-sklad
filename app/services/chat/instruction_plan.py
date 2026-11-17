"""Immutable source plans and bounded delivery of their exact original sections."""

import copy
import hashlib
import json
import re
from uuid import NAMESPACE_URL, uuid5

from app.services.rag.catalog import instruction_sections
from app.services.rag.interface import Candidate
from app.services.rag.talk import ApprovedTalkDocument, InstructionPlanPart, KnowledgeUnavailable


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def immutable_plan(saved):
    return {"version": saved["version"], "parts": copy.deepcopy(saved["parts"]), "digest": saved["digest"]}


def serialize_plan(parts):
    if not parts or len(parts) > 8:
        raise KnowledgeUnavailable("InvalidSavedInstructionPlan")
    rows = [{"source": copy.deepcopy(part.approved.document["source"]),
             "goal_ids": list(part.goal_ids), "goals": list(part.goals),
             "section_ids": [section["id"] for section in part.sections],
             "sections_sha256": _digest(list(part.sections))} for part in parts]
    body = {"version": 1, "parts": rows}
    return {**body, "digest": _digest(body), "cursor": 0, "shown": []}


def restore_plan(knowledge, saved):
    """Re-read every original and resolve section IDs before any plan output."""
    if (not isinstance(saved, dict) or saved.get("version") != 1
            or not isinstance(saved.get("parts"), list) or not 1 <= len(saved["parts"]) <= 8
            or saved.get("digest") != _digest({"version": 1, "parts": saved["parts"]})):
        raise KnowledgeUnavailable("InvalidSavedInstructionPlan")
    parts = []
    for row in saved["parts"]:
        source = row.get("source", {})
        approved = knowledge.read(source.get("file_id"))
        if approved.document["source"] != source:
            raise KnowledgeUnavailable("InstructionPlanSourceChanged")
        sections = instruction_sections(approved.document)
        selected_ids = row.get("section_ids")
        if (not isinstance(selected_ids, list) or not selected_ids or len(selected_ids) > 10000
                or len(set(selected_ids)) != len(selected_ids)):
            raise KnowledgeUnavailable("InvalidSavedInstructionPlan")
        selected = tuple(section for section in sections if section["id"] in selected_ids)
        if ([section["id"] for section in selected] != selected_ids
                or _digest(list(selected)) != row.get("sections_sha256")
                or any(section["id"] not in selected_ids for section in sections
                       if section["kind"] in {"WARNING", "PREREQUISITE"})):
            raise KnowledgeUnavailable("InstructionPlanSourceChanged")
        parts.append(InstructionPlanPart(approved, tuple(row["goal_ids"]), tuple(row["goals"]),
                                         tuple(selected_ids), selected))
    return tuple(parts)


def composite_approved(parts, saved):
    """Give the whole verified plan its own immutable solution-attempt identity."""
    identity = "talk-instruction-plan:" + saved["digest"]
    content = "\n\n".join(section["text"] for part in parts for section in part.sections)
    candidate = Candidate(
        solution_id=str(uuid5(NAMESPACE_URL, identity)),
        chunk_id=str(uuid5(NAMESPACE_URL, identity + ":sections")), version=1,
        title=" / ".join(dict.fromkeys(part.approved.candidate.title for part in parts))[:255],
        content=content, steps=[section["text"] for part in parts for section in part.sections], score=1.0,
        source={"backend": "openwebui", "plan_digest": saved["digest"],
                "sources": [copy.deepcopy(part.approved.document["source"]) for part in parts]},
    )
    document = {"source": copy.deepcopy(parts[0].approved.document["source"]), "content": content, "images": []}
    return ApprovedTalkDocument(document, {}, candidate)


def _units(parts):
    units = []
    for part in parts:
        pending = []
        emitted_images = set()
        original_text = []
        last_image_section = {number: index for index, section in enumerate(part.sections)
                              for number in section.get("image_numbers", [])}
        captions = {" ".join(paragraph.split()) for image in part.approved.illustrations.values()
                    for paragraph in [image.caption, *image.caption.splitlines()] if paragraph.strip()}
        illustrated = any(section.get("image_numbers") for section in part.sections)
        for section_index, section in enumerate(part.sections):
            original_numbers = section.get("image_numbers", [])
            numbers = [number for number in original_numbers
                       if number not in emitted_images and last_image_section[number] == section_index]
            if any(number not in part.approved.illustrations for number in numbers):
                raise KnowledgeUnavailable("InstructionPlanImageMissing")
            text = section["text"]
            caption_body = re.sub(r"^Иллюстрация\s+\d+[.)]\s*", "", text, flags=re.I)
            normalized = " ".join(caption_body.split())
            repeated_caption = normalized in captions and any(normalized in earlier for earlier in original_text)
            if not repeated_caption and text.strip() != "Фотографии и пояснения":
                pending.append(text)
                original_text.append(" ".join(text.split()))
            emitted_images.update(numbers)
            if not pending and not numbers:
                continue
            if illustrated and not numbers:
                continue
            groups = [numbers[offset:offset + 6] for offset in range(0, len(numbers), 6)] or [[]]
            for index, group in enumerate(groups):
                units.append((part.approved, "\n\n".join(pending) if index == 0 else "", group))
            pending = []
        if pending:
            units.append((part.approved, "\n\n".join(pending), []))
    return units


def render_plan(parts, saved, *, repeat=False):
    """Deliver at most three illustrated action groups and six images in order."""
    units = _units(parts)
    cursor = saved.get("cursor", 0)
    shown = saved.get("shown", [])
    if (type(cursor) is not int or not 0 <= cursor <= len(units)
            or not isinstance(shown, list) or any(type(index) is not int or not 0 <= index < len(units)
                                                for index in shown)):
        raise KnowledgeUnavailable("InvalidSavedInstructionProgress")
    selection = shown if repeat else list(range(cursor, len(units)))
    selected, photo_count = [], 0
    for index in selection:
        number = len(units[index][2])
        if len(selected) >= 3 or photo_count + number > 6:
            break
        selected.append(index)
        photo_count += number
    blocks, sources = [], []
    for index in selected:
        approved, text, numbers = units[index]
        source = approved.document["source"]
        if text:
            blocks.append((text, None))
            sources.append(source)
        for number in numbers:
            picture = approved.illustrations[number]
            caption = picture.caption if picture.caption not in text else ""
            blocks.append((caption, picture))
            sources.append(source)
    updated = copy.deepcopy(saved)
    if not repeat:
        updated.update(cursor=cursor + len(selected), shown=selected)
    more = updated["cursor"] < len(units)
    prompt = ("Когда выполните показанные действия, напишите «Продолжить» — покажу следующие шаги."
              if more else "Удалось выполнить эти действия?")
    blocks.append((prompt, None))
    sources.append(None)
    return blocks, sources, updated, more


