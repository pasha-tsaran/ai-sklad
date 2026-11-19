"""Immutable source plans and bounded delivery of their exact original sections."""

import copy
import hashlib
import json
import re
from uuid import NAMESPACE_URL, uuid5
from dataclasses import replace

from app.services.rag.catalog import instruction_scope_guard, instruction_sections
from app.services.rag.interface import Candidate
from app.services.rag.instruction_focus import preparation_warning_excerpt
from app.services.rag.talk import ApprovedTalkDocument, InstructionPlanPart, KnowledgeUnavailable
from app.services.nextcloud.delivery import split_text


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def immutable_plan(saved):
    return {"version": saved["version"], "parts": copy.deepcopy(saved["parts"]), "digest": saved["digest"]}


def serialize_plan(parts, *, version=3):
    if (type(version) is not int or version not in (1, 2, 3) or not parts or len(parts) > 8
            or any(part.delivery_mode not in ("PROCEDURE", "EXCERPT")
                   or getattr(part, "preparation_checklist", False) and part.delivery_mode != "EXCERPT"
                   for part in parts)):
        raise KnowledgeUnavailable("InvalidSavedInstructionPlan")
    rows = [{"source": copy.deepcopy(part.approved.document["source"]),
             "goal_ids": list(part.goal_ids), "goals": list(part.goals),
             "section_ids": [section["id"] for section in part.sections],
             "sections_sha256": _digest(list(part.sections)),
             **({"preparation_checklist": True} if getattr(part, "preparation_checklist", False) else {}),
             **({"delivery_mode": "EXCERPT"} if part.delivery_mode == "EXCERPT" else {})} for part in parts]
    body = {"version": version, "parts": rows}
    return {**body, "digest": _digest(body), "cursor": 0, "shown": []}


def restore_plan(knowledge, saved):
    """Re-read every original and resolve section IDs before any plan output."""
    if (not isinstance(saved, dict) or type(saved.get("version")) is not int or saved["version"] not in (1, 2, 3)
            or not isinstance(saved.get("parts"), list) or not 1 <= len(saved["parts"]) <= 8
            or saved.get("digest") != _digest({"version": saved["version"], "parts": saved["parts"]})):
        raise KnowledgeUnavailable("InvalidSavedInstructionPlan")
    parts = []
    for row in saved["parts"]:
        mode = row.get("delivery_mode", "PROCEDURE")
        checklist = row.get("preparation_checklist", False)
        if mode not in ("PROCEDURE", "EXCERPT") or type(checklist) is not bool or checklist and mode != "EXCERPT":
            raise KnowledgeUnavailable("InvalidSavedInstructionPlan")
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
        selected_groups = {section["group_id"] for section in selected}
        if ([section["id"] for section in selected] != selected_ids
                or _digest(list(selected)) != row.get("sections_sha256")
                or any(section["id"] not in selected_ids for section in sections
                       if instruction_scope_guard(section, selected_groups))):
            raise KnowledgeUnavailable("InstructionPlanSourceChanged")
        parts.append(InstructionPlanPart(approved, tuple(row["goal_ids"]), tuple(row["goals"]),
                                         tuple(selected_ids), selected, delivery_mode=mode,
                                         preparation_checklist=checklist))
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


def _legacy_units(parts):
    units = []
    for part in parts:
        pending = []
        excerpt_images = []
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
            text = preparation_warning_excerpt(section) if getattr(part, "preparation_checklist", False) else section["text"]
            caption_body = re.sub(r"^Иллюстрация\s+\d+[.)]\s*", "", text, flags=re.I)
            normalized = " ".join(caption_body.split())
            repeated_caption = normalized in captions and normalized in " ".join(original_text)
            if not repeated_caption and text.strip() != "Фотографии и пояснения":
                pending.append(text)
                original_text.append(" ".join(text.split()))
            emitted_images.update(numbers)
            if part.delivery_mode == "EXCERPT":
                excerpt_images.extend(numbers)
                continue
            if not pending and not numbers:
                continue
            if illustrated and not numbers:
                continue
            groups = [numbers[offset:offset + 6] for offset in range(0, len(numbers), 6)] or [[]]
            for index, group in enumerate(groups):
                units.append((part.approved, "\n\n".join(pending) if index == 0 else "", group))
            pending = []
        if part.delivery_mode == "EXCERPT" and (pending or excerpt_images):
            groups = [excerpt_images[offset:offset + 6] for offset in range(0, len(excerpt_images), 6)] or [[]]
            for index, group in enumerate(groups):
                units.append((part.approved, "\n\n".join(pending) if index == 0 else "", group))
        elif pending:
            units.append((part.approved, "\n\n".join(pending), []))
    return units


def _fence_state(text, state=None):
    for match in re.finditer(r"(?m)^ {0,3}(?P<marks>`{3,}|~{3,})(?P<suffix>[^\n]*)$", text):
        marks = match.group("marks")
        if state is None:
            state = marks[0], len(marks)
        elif marks[0] == state[0] and len(marks) >= state[1] and not match.group("suffix").strip():
            state = None
    return state


def _append_unit(units, unit):
    approved, text, numbers = unit
    chunks = (_presentation_chunks(text) if text else []) or [""]
    for chunk in chunks[:-1]:
        units.append((approved, chunk, []))
    text = chunks[-1]
    if not numbers:
        units.append((approved, text, numbers))
        return
    for offset in range(0, len(numbers), 6):
        units.append((approved, text if offset == 0 else "", numbers[offset:offset + 6]))


def _fenced_chunks(lines, *, closed):
    original = "".join(lines)
    if len(original) <= 32_000:
        return [original]
    opening = lines[0].rstrip("\r\n")
    marker = re.match(r" {0,3}(`{3,}|~{3,})", opening).group(1)
    closing = lines[-1].rstrip("\r\n") if closed else marker
    body_lines = lines[1:-1] if closed else lines[1:]
    prefix = opening + "\n"
    overhead = len(prefix) + len(closing) + 1
    if overhead > 32_000:
        raise KnowledgeUnavailable("InstructionCodeLineTooLong")
    chunks, pending, size = [], [], 0
    for line in body_lines:
        if len(line) + overhead > 32_000:
            raise KnowledgeUnavailable("InstructionCodeLineTooLong")
        if pending and size + len(line) + overhead > 24_000:
            body = "".join(pending)
            chunks.append(prefix + body + ("" if body.endswith("\n") else "\n") + closing)
            pending, size = [], 0
        pending.append(line)
        size += len(line)
    if pending:
        body = "".join(pending)
        chunks.append(prefix + body + ("" if body.endswith("\n") else "\n") + closing)
    return chunks


def _presentation_chunks(text):
    if len(text) <= 32_000:
        return [text]
    chunks, pending, fence = [], [], None
    for line in text.splitlines(keepends=True):
        following = _fence_state(line, fence)
        if fence is None and following is not None:
            chunks.extend(chunk for chunk in split_text("".join(pending), limit=24_000) if chunk.strip())
            pending = [line]
        else:
            pending.append(line)
            if fence is not None and following is None:
                chunks.extend(_fenced_chunks(pending, closed=True))
                pending = []
        fence = following
    if fence is not None:
        chunks.extend(_fenced_chunks(pending, closed=False))
    else:
        chunks.extend(chunk for chunk in split_text("".join(pending), limit=24_000) if chunk.strip())
    return chunks


def _unique_part_sections(parts):
    """Suppress only copies already selected by earlier goals of the same source/mode.

    Persisted parts retain their complete verified scope. This view changes new
    plan presentation only, so source revalidation and old delivery cursors stay
    bound to their original data and unit algorithms.
    """
    seen_sections, seen_images = set(), set()
    for part in parts:
        source_sha = part.approved.document["source"].get("sha256")
        if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha):
            yield part
            continue
        domain = source_sha, part.delivery_mode
        sections, current_sections, current_images = [], set(), set()
        image_hashes = {}
        for section in part.sections:
            section_key = (*domain, section["id"], _digest(section))
            current_sections.add(section_key)
            numbers = []
            for number in section.get("image_numbers", []):
                if number not in part.approved.illustrations:
                    raise KnowledgeUnavailable("InstructionPlanImageMissing")
                image_sha = image_hashes.setdefault(number, hashlib.sha256(part.approved.illustrations[number].data).hexdigest())
                image_key = (*domain, image_sha)
                current_images.add(image_key)
                if image_key not in seen_images:
                    numbers.append(number)
            if section_key not in seen_sections:
                sections.append({**section, "image_numbers": numbers})
        # Identical bytes at distinct positions inside this original part retain
        # their original semantics. Only images delivered by earlier goals count.
        seen_sections.update(current_sections)
        seen_images.update(current_images)
        yield replace(part, sections=tuple(sections))


def _units(parts, *, version=2):
    """Keep old cursors stable while new plans deliver coherent text and whole code fences."""
    if type(version) is not int or version not in (1, 2, 3):
        raise KnowledgeUnavailable("InvalidSavedInstructionPlan")
    if version == 1:
        return _legacy_units(parts)
    if version == 3:
        parts = _unique_part_sections(parts)
    units = []
    for part in parts:
        pending, fence = None, None
        for approved, text, numbers in _legacy_units([part]):
            if (pending is not None and (fence is not None
                    or not pending[2] and not numbers and len(pending[1]) + len(text) + 2 <= 4000)):
                pending = approved, "\n\n".join(value for value in (pending[1], text) if value), pending[2] + numbers
                fence = _fence_state(text, fence)
                continue
            if pending is not None:
                _append_unit(units, pending)
            pending, fence = (approved, text, numbers), _fence_state(text)
        if pending is not None:
            _append_unit(units, pending)
    return units


def render_plan(parts, saved, *, repeat=False):
    """Deliver at most three illustrated action groups and six images in order."""
    units = _units(parts, version=saved.get("version", 1))
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
        normalized_text = " ".join(text.split())
        if text:
            blocks.append((text, None))
            sources.append(source)
        for number in numbers:
            picture = approved.illustrations[number]
            caption = picture.caption if " ".join(picture.caption.split()) not in normalized_text else ""
            blocks.append((caption, picture))
            sources.append(source)
    updated = copy.deepcopy(saved)
    if not repeat:
        updated.update(cursor=cursor + len(selected), shown=selected)
    more = updated["cursor"] < len(units)
    if all(part.delivery_mode == "EXCERPT" for part in parts):
        prompt = ("Напишите «Продолжить» — покажу оставшуюся информацию."
                  if more else "Эта информация помогла?")
    else:
        prompt = ("Когда выполните показанные действия, напишите «Продолжить» — покажу следующие шаги."
                  if more else "Удалось выполнить эти действия?")
    blocks.append((prompt, None))
    sources.append(None)
    return blocks, sources, updated, more


