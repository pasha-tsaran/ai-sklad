"""Request-local references to exact, bounded employee-authored text spans."""

MAX_FRAGMENT_CHARS = 300
MAX_REFERENCES = 120
MAX_REFERENCE_CHARS = 12000
FRAGMENT_OVERLAP = 80


def _fragments(original):
    start = 0
    while start < len(original):
        end = min(start + MAX_FRAGMENT_CHARS, len(original))
        if end < len(original):
            boundary = original.rfind(" ", start + MAX_FRAGMENT_CHARS // 2, end)
            if boundary >= 0:
                end = boundary + 1
        fragment = original[start:end]
        if fragment.strip():
            yield fragment
        if end == len(original):
            break
        start = max(start + 1, end - FRAGMENT_OVERLAP)


def build_user_evidence(authority):
    """Keep current-message spans first and preserve every original character."""
    references, seen, size = {}, set(), 0
    for original in authority:
        if not isinstance(original, str):
            continue
        for fragment in _fragments(original):
            if fragment in seen:
                continue
            if len(references) >= MAX_REFERENCES or size + len(fragment) > MAX_REFERENCE_CHARS:
                return references
            references[f"u{len(references) + 1}"] = fragment
            seen.add(fragment)
            size += len(fragment)
    return references


def constrain_user_evidence(schema, references, current_message):
    """Make every evidence choice a backend-owned ID inside the response grammar."""
    definitions = schema.get("$defs", {})
    for name in ("SupportTaskFact", "SupportTaskGoal"):
        definitions[name]["properties"]["evidence"] = {"type": "string", "enum": list(references)}
    fact = definitions["SupportTaskFact"]
    fact["required"] = list(dict.fromkeys([
        *fact.get("required", []), "entity", "attribute", "negated",
    ]))
    current_ids = [reference for reference, text in references.items() if text in current_message]
    for name in ("replacement_evidence", "unknown_answer_evidence"):
        schema["properties"][name] = {"type": "string", "enum": ["", *current_ids], "default": ""}
    return schema


def resolve_user_evidence(value, references, authority):
    """Restore exact text and retain compatible, already-literal provider replies."""
    original = references.get(value, value)
    return original if any(original in text for text in authority) else None
