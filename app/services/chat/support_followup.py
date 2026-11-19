"""Verified active-guide context and bounded, source-only troubleshooting excerpts."""

import copy
import json
import re

from app.schemas.contracts import FollowupResult
from app.services.chat.instruction_plan import _units, composite_approved, render_plan, restore_plan, serialize_plan
from app.services.rag.catalog import instruction_scope_groups, instruction_scope_guard, instruction_sections
from app.services.rag.query import _terms, normalized_query
from app.services.rag.talk import InstructionPlanPart, KnowledgeUnavailable, knowledge_stage


OBSERVATIONS = {"current_step", "error_text", "symptoms", "completed"}
NO_DIAGNOSTIC_STEPS = (
    "В доступной инструкции нет подтверждённых дополнительных действий для описанного результата. "
    "Сохраню контекст для поддержки. Чтобы подготовить заявку, напишите «Оставить заявку»."
)


def _verified_parts(knowledge, flow):
    if flow.get("instruction_plan"):
        return restore_plan(knowledge, flow["instruction_plan"])
    source = flow["source"]
    approved = knowledge.read(source["file_id"])
    if approved.document["source"] != source:
        raise KnowledgeUnavailable("InstructionPlanSourceChanged")
    sections = tuple(instruction_sections(approved.document))
    return (InstructionPlanPart(approved, (), (), tuple(section["id"] for section in sections), sections),)


def _displayed_originals(parts, saved):
    units = _units(parts, version=saved.get("version", 1))
    cursor = saved.get("cursor", 0)
    shown = units[:cursor] if type(cursor) is int and 0 <= cursor <= len(units) else []
    displayed_by_source, pictures_by_source = {}, {}
    for approved, text, numbers in shown:
        identity = approved.document["source"]["file_id"], approved.document["source"]["sha256"]
        displayed_by_source.setdefault(identity, []).append(" ".join(text.split()))
        pictures_by_source.setdefault(identity, set()).update(numbers)
    return cursor, displayed_by_source, pictures_by_source


def verified_guide_context(knowledge, flow):
    parts = _verified_parts(knowledge, flow)
    saved = flow.get("instruction_plan", {})
    cursor, displayed_by_source, pictures_by_source = _displayed_originals(parts, saved)
    sources = []
    for part in parts:
        identity = part.approved.document["source"]["file_id"], part.approved.document["source"]["sha256"]
        displayed = " ".join(displayed_by_source.get(identity, []))
        pictures = pictures_by_source.get(identity, set())
        steps = [{"section_id": section["id"], "text": section["text"][:500], "kind": section["kind"],
                  "image_numbers": section["image_numbers"],
                  "shown": " ".join(section["text"].split()) in displayed
                           or bool(set(section["image_numbers"]) & pictures)} for section in part.sections]
        steps.sort(key=lambda step: not step["shown"])
        sources.append({"source": copy.deepcopy(part.approved.document["source"]),
                        "procedure_ids": list(dict.fromkeys(section["group_id"] for section in part.sections)),
                        "sections": steps[:12]})
    result = {"read_only": True, "sources": sources,
              "current_progress": {"cursor": cursor, "shown": saved.get("shown", []),
                                   "has_more": bool(flow.get("has_more"))},
              "last_question": flow.get("clarification_question", "")[:300]}
    while len(json.dumps(result, ensure_ascii=False)) > 8000:
        largest = max(sources, key=lambda source: len(source["sections"]))
        if not largest["sections"]:
            raise KnowledgeUnavailable("GuideContextTooLarge")
        largest["sections"].pop()
    return result


def _fact_identity(fact):
    return tuple(normalized_query(str(fact.get(key, ""))) for key in ("key", "value", "entity", "attribute")) + (
        fact.get("negated", False),
    )


def observational_followup(previous, state, *, ui_question=False):
    if state.get("new_task") or not previous:
        return False
    old_goals = {key: normalized_query(goal["text"]) for key, goal in previous.get("goals", {}).items()}
    new_goals = {key: normalized_query(goal["text"]) for key, goal in state.get("goals", {}).items()}
    if old_goals != new_goals or not new_goals:
        return False
    if all(goal.get("kind") == "LOOKUP" for goal in state["goals"].values()):
        return False
    old_conditions = {_fact_identity(fact) for fact in previous.get("facts", []) if fact["key"] not in OBSERVATIONS}
    new_conditions = {_fact_identity(fact) for fact in state.get("facts", []) if fact["key"] not in OBSERVATIONS}
    old_observations = {_fact_identity(fact) for fact in previous.get("facts", []) if fact["key"] in OBSERVATIONS}
    new_observations = {_fact_identity(fact) for fact in state.get("facts", []) if fact["key"] in OBSERVATIONS}
    unknown_observation = any(answer.get("field") in OBSERVATIONS
                              for answer in state.get("unknown_answers", [])
                              if answer not in previous.get("unknown_answers", []))
    return old_conditions == new_conditions and (bool(new_observations - old_observations) or unknown_observation
                                                 or ui_question)


def needs_error_text(state):
    facts = [fact for fact in state.get("facts", []) if not fact.get("negated", False)]
    errors = [fact["value"] for fact in facts if fact["key"] == "error_text"]
    if errors:
        return all(_generic_error_report(text) for text in errors)
    return any(fact["key"] in {"current_step", "symptoms"}
               and re.search(r"ошибк\w*|error\b|fail\w*", fact["value"], re.I) for fact in facts)


def _generic_error_report(text):
    text = normalized_query(text)
    text = re.sub(r"\b(?:на|в|после)\s+\S+(?:\s+\S+)?\s+шаг\w*\b", " ", text)
    text = re.sub(r"\bшаг\w*\s+\d+(?:\s*[а-я]+)?\b", " ", text)
    words = re.findall(r"\w+", text)
    return bool(words) and all(re.fullmatch(
        r"ошиб\w*|error|fail\w*|выда\w*|появ\w*|возник\w*|показыва\w*|выскакива\w*|"
        r"сообщени\w*|программ\w*|пишет|опять|снова|это|какая|то|что|то|мне", word,
    ) for word in words)


def diagnostic_excerpt(knowledge, flow, message, context, llm, *, allow_repeat=False):
    """Select original help for a difficulty without replaying a performed recipe."""
    followup = getattr(llm, "followup", None)
    if not callable(followup):
        return None
    parts = _verified_parts(knowledge, flow)
    _, shown_text, shown_pictures = _displayed_originals(parts, flow.get("instruction_plan", {}))
    for part in parts:
        approved = part.approved
        sections = instruction_sections(approved.document)
        selected_ids = {section["id"] for section in part.sections}
        identity = approved.document["source"]["file_id"], approved.document["source"]["sha256"]
        displayed = " ".join(shown_text.get(identity, []))
        pictures = shown_pictures.get(identity, set())
        displayed_groups = {section["group_id"] for section in part.sections
                            if " ".join(section["text"].split()) in displayed
                            or bool(set(section["image_numbers"]) & pictures)}
        pending_conditionals = {section["id"] for section in part.sections
                                if flow.get("has_more") and flow.get("instruction_plan")
                                and section.get("parent_group_id") and section.get("condition_section_id")
                                and section["group_id"] not in displayed_groups}
        available = [section for section in sections if not section["is_heading"]
                     and (allow_repeat or section["id"] not in selected_ids or section["id"] in pending_conditionals)
                     and section["kind"] not in {"WARNING", "PREREQUISITE"}]
        if not available:
            continue
        terms = _terms(message)
        available.sort(key=lambda section: (-len(terms & _terms(section["text"])), section["ordinal"]))
        excerpts, size = [], 0
        for section in available[:32]:
            if size + len(section["text"]) <= 12000:
                excerpts.append(section)
                size += len(section["text"])
        if not excerpts:
            continue
        images = sorted(approved.illustrations)
        with knowledge_stage("instruction_followup"):
            answer = FollowupResult.model_validate(followup(message, context, {
                "chunk_id": approved.candidate.chunk_id, "title": approved.candidate.title,
                "excerpts": [section["text"] for section in excerpts],
                "illustrations": [{"index": index, "caption": approved.illustrations[number].caption}
                                  for index, number in enumerate(images)],
                "excerpt_image_indices": {str(index): [images.index(number) for number in section["image_numbers"]]
                                          for index, section in enumerate(excerpts)},
            }))
        if answer.result == "NO_KNOWLEDGE":
            continue
        valid = (answer.source_chunk_ids == [approved.candidate.chunk_id] and bool(answer.step_indices)
                 and len(answer.step_indices) == len(set(answer.step_indices))
                 and len(answer.illustration_indices) == len(set(answer.illustration_indices))
                 and all(0 <= index < len(excerpts) for index in answer.step_indices)
                 and all(0 <= index < len(images) for index in answer.illustration_indices))
        if not valid:
            raise KnowledgeUnavailable("InvalidDiagnosticExcerpt", "instruction_followup")
        selected = [excerpts[index] for index in answer.step_indices]
        selected_ids = {section["id"] for section in selected}
        condition_ids = {section["condition_section_id"] for section in selected if section.get("condition_section_id")}
        complete_groups = {section["group_id"] for section in selected
                           if section.get("parent_group_id") or section.get("exclusive_set_id")}
        action_groups = instruction_scope_groups(sections, complete_groups, include_parents=False)
        guard_groups = instruction_scope_groups(sections, {section["group_id"] for section in selected})
        chosen = [section for section in sections
                  if section["id"] in selected_ids | condition_ids or section["group_id"] in action_groups
                  or instruction_scope_guard(section, guard_groups)]
        previous_choices = {(section["exclusive_set_id"], section["exclusive_choice_section_id"])
                            for section in part.sections if section.get("exclusive_choice_section_id")}
        selected_choices = {(section["exclusive_set_id"], section["exclusive_choice_section_id"])
                            for section in chosen if section.get("exclusive_choice_section_id")}
        requires_confirmation = bool(selected_choices - previous_choices)
        old_actions = [" ".join(section["text"].split()) for section in part.sections
                       if not section["is_heading"] and section["kind"] == "TEXT"
                       and not re.match(r"^(?:Иллюстрация|Фотографии)\b", section["text"], re.I)]
        chosen_text = " ".join(" ".join(section["text"].split()) for section in chosen)
        if not allow_repeat and old_actions and all(action in chosen_text for action in old_actions):
            continue
        permitted_images = {number for section in chosen for number in section["image_numbers"]}
        selected_images = [images[index] for index in sorted(answer.illustration_indices)]
        if not set(selected_images) <= permitted_images:
            raise KnowledgeUnavailable("InvalidDiagnosticImage", "instruction_followup")
        if not knowledge.is_current(approved.document["source"]):
            raise KnowledgeUnavailable("InstructionPlanSourceChanged")
        goals = flow.get("task_state", {}).get("goals", {})
        goal_ids = part.goal_ids or tuple(goals)
        if not set(goal_ids) <= goals.keys():
            raise KnowledgeUnavailable("InvalidDiagnosticGoals", "instruction_followup")
        fragment = InstructionPlanPart(approved, goal_ids, tuple(goals[goal_id]["text"] for goal_id in goal_ids),
                                       tuple(section["id"] for section in chosen), tuple(chosen), delivery_mode="EXCERPT")
        saved = serialize_plan([fragment])
        blocks, sources, saved, more = render_plan([fragment], saved)
        return {"blocks": blocks, "block_sources": sources, "source": approved.document["source"],
                "section_ids": [section["id"] for section in chosen], "plan": saved, "has_more": more,
                "requires_scope_confirmation": requires_confirmation,
                "approved": composite_approved([fragment], saved)}
    return None
