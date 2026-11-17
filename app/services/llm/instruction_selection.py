"""Request-local evidence references for source-grounded instruction choices."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool


class GoalReference(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    goal_id: str = Field(min_length=1, max_length=64)
    evidence_id: str = Field(min_length=1, max_length=64)


class AlternativeReference(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    file_id: str
    evidence_id: str = Field(min_length=1, max_length=64)


class PartReference(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    file_id: str
    covered_goals: list[GoalReference] = Field(min_length=1, max_length=8)
    section_ids: list[str] = Field(min_length=1, max_length=32)


class InstructionReferenceSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    result: Literal["MATCH", "PARTIAL", "CLARIFY", "NO_KNOWLEDGE", "READ", "RELATED", "PLAN"]
    file_id: str | None = None
    question: str = Field(default="", max_length=300)
    alternatives: list[AlternativeReference] = Field(default_factory=list, max_length=3)
    understanding: str = Field(default="", max_length=300)
    new_request: StrictBool = False
    covered_goals: list[GoalReference] = Field(default_factory=list, max_length=8)
    missing_goal_ids: list[str] = Field(default_factory=list, max_length=8)
    clarification_kind: Literal["USER_GOAL", "VISIBLE_STATE", "PHYSICAL_SETUP", "WORK_HISTORY"] | None = None
    clarification_sources: list[AlternativeReference] = Field(default_factory=list, max_length=3)
    section_ids: list[str] = Field(default_factory=list, max_length=32)
    parts: list[PartReference] = Field(default_factory=list, max_length=8)


def _closed_object(fields, required):
    return {"type": "object", "properties": fields, "required": required, "additionalProperties": False}


def _eligible_references(registry, *, planning=False):
    return [reference_id for reference_id, evidence in registry.items()
            if (1 if planning else 20) <= len(evidence["text"]) <= 200]


def reference_selection_schema(base, payload):
    """Bind every selectable fragment to its loaded source inside the grammar."""
    properties = base["properties"]
    goals = list(payload["requested_goals"])
    goal_count = len(goals)
    candidates = payload["candidates"]
    planning = payload.get("instruction_planning") is True
    eligible = {file_id: references for file_id, registry in candidates.items()
                if (references := _eligible_references(registry, planning=planning))}
    definitions = {}
    branches = []

    def add(result, fields, required):
        branches.append(_closed_object({"result": {"type": "string", "const": result}, **fields},
                                       ["result", *required]))

    optional = {name: properties[name] for name in ("understanding", "new_request")}
    part_references = []
    for file_id, references in eligible.items():
        goal_definition = "GoalReference_" + file_id
        source_definition = "AlternativeReference_" + file_id
        evidence_id = {"type": "string", "enum": references}
        definitions[goal_definition] = _closed_object({
            "goal_id": {"type": "string", "enum": goals}, "evidence_id": evidence_id,
        }, ["goal_id", "evidence_id"])
        definitions[source_definition] = _closed_object({
            "file_id": {"type": "string", "const": file_id}, "evidence_id": evidence_id,
        }, ["file_id", "evidence_id"])
        covered = {"type": "array", "items": {"$ref": "#/$defs/" + goal_definition},
                   "minItems": goal_count, "maxItems": goal_count}
        source = {"type": "string", "const": file_id}
        section_ids = list(payload.get("candidate_sections", {}).get(file_id, {}))
        sections = {"type": "array", "items": {"type": "string", "enum": section_ids},
                    "minItems": 1, "maxItems": min(len(section_ids), 32), "uniqueItems": True}
        section_fields = {"section_ids": sections} if planning else {}
        section_required = ["section_ids"] if planning else []
        if planning and not section_ids:
            continue
        if planning:
            part_name = "PartReference_" + file_id
            definitions[part_name] = _closed_object({"file_id": source,
                "covered_goals": {**covered, "minItems": 1}, "section_ids": sections},
                ["file_id", "covered_goals", "section_ids"])
            part_references.append({"$ref": "#/$defs/" + part_name})
        add("MATCH", {"file_id": source, "covered_goals": covered, **section_fields, **optional},
            ["file_id", "covered_goals", *section_required])
        if goal_count > 1:
            missing = {"type": "array", "items": {"type": "string", "enum": goals},
                       "minItems": 1, "maxItems": goal_count - 1, "uniqueItems": True}
            add("PARTIAL", {"file_id": source,
                            "covered_goals": {**covered, "minItems": 1, "maxItems": goal_count - 1},
                            "missing_goal_ids": missing, **section_fields, **optional},
                ["file_id", "covered_goals", "missing_goal_ids", *section_required])

    if planning and goal_count > 1 and part_references:
        add("PLAN", {"parts": {"type": "array", "items": {"anyOf": part_references},
                                "minItems": 1, "maxItems": min(goal_count, 8)},
                     "missing_goal_ids": {"type": "array", "items": {"type": "string", "enum": goals},
                                          "maxItems": goal_count - 1, "uniqueItems": True}, **optional},
            ["parts", "missing_goal_ids"])

    source_references = {"type": "array", "minItems": 1, "maxItems": 3, "items": {"anyOf": [
        {"$ref": "#/$defs/AlternativeReference_" + file_id} for file_id in eligible
    ]}}
    new_request = {"new_request": properties["new_request"]}
    if eligible:
        add("RELATED", {"alternatives": source_references, **new_request}, ["alternatives"])
    add("NO_KNOWLEDGE", {"missing_goal_ids": {"type": "array", "const": goals,
                                            "items": {"type": "string", "enum": goals},
                                            "minItems": goal_count, "maxItems": goal_count}, **new_request},
        ["missing_goal_ids"])
    catalogue = [file_id for file_id in payload.get("catalogue", {})
                 if not planning or file_id not in candidates]
    if catalogue and payload.get("read_available") is not False:
        add("READ", {"file_id": {"type": "string", "enum": catalogue}, **new_request}, ["file_id"])
    strict_questions = payload.get("source_clarification_required") is True
    if payload.get("allow_clarification") is not False and (eligible or not strict_questions):
        question = {"question": {"type": "string", "minLength": 5, "maxLength": 300}, **new_request}
        required = ["question"]
        if strict_questions:
            question.update(
                clarification_kind={"type": "string", "enum": [
                    "USER_GOAL", "VISIBLE_STATE", "PHYSICAL_SETUP", "WORK_HISTORY",
                ]},
                clarification_sources=source_references,
            )
            required += ["clarification_kind", "clarification_sources"]
        add("CLARIFY", question, required)
    return {"title": base["title"], "$defs": definitions, "anyOf": branches}


def resolve_instruction_references(choice, registries, *, planning=False, sections=None):
    """Restore exact original-backed proof before ordinary canonical validation."""
    decoded = choice.model_dump()

    def resolve(file_id, reference_id):
        registry = registries.get(file_id, {})
        if reference_id not in registry or reference_id not in _eligible_references(registry, planning=planning):
            raise ValueError("UnknownInstructionEvidenceReference")
        return registry[reference_id]

    def coverage(file_id, goal):
        proof = resolve(file_id, goal.evidence_id)
        value = {"goal_id": goal.goal_id, "evidence": proof["text"],
                 "support_kind": proof["support_kind"], "image_numbers": proof["image_numbers"]}
        if planning:
            matches = [section_id for section_id, section in (sections or {}).get(file_id, {}).items()
                       if goal.evidence_id in section["evidence_ids"]]
            if len(matches) != 1:
                raise ValueError("UnknownInstructionEvidenceSection")
            value["section_id"] = matches[0]
        return value

    decoded["covered_goals"] = [coverage(choice.file_id, goal) for goal in choice.covered_goals]
    decoded["parts"] = [{"file_id": part.file_id, "section_ids": part.section_ids,
                          "covered_goals": [coverage(part.file_id, goal) for goal in part.covered_goals]}
                        for part in choice.parts]
    for field in ("alternatives", "clarification_sources"):
        decoded[field] = [{"file_id": source.file_id,
                           "evidence": resolve(source.file_id, source.evidence_id)["text"]}
                          for source in getattr(choice, field)]
    return decoded
