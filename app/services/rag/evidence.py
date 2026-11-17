"""Deterministic references to exact source text and original image captions.

Short spans remain available as context. Callers only offer references whose
text satisfies the existing evidence minimum of twenty characters.
"""

import re
import textwrap


def _fragments(span):
    span = span.strip()
    fragments = textwrap.wrap(span, width=200, expand_tabs=False, replace_whitespace=False,
                              break_long_words=True, break_on_hyphens=False)
    if len(fragments) > 1 and len(fragments[-1]) < 20:
        start = len(span) - len(fragments[-1])
        while start > 0 and len(span) - start < 20:
            start = span.rfind(" ", 0, start - 1) + 1
        if len(span) - start > 200:
            start = len(span) - 20
        fragments[-1] = span[start:]
    return fragments


def build_evidence_registry(content: str, image_captions: dict[int, str] | None = None):
    """Partition an already bounded original without joining across captions."""
    if not isinstance(content, str) or (image_captions is not None and not isinstance(image_captions, dict)):
        raise ValueError("InvalidEvidenceSource")
    captions = {}
    rows = image_captions or {}
    if any(type(number) is not int or number < 1 or not isinstance(caption, str)
           for number, caption in rows.items()):
        raise ValueError("InvalidEvidenceCaption")
    for number, caption in sorted(rows.items()):
        normalized = " ".join(caption.split())
        if normalized:
            captions.setdefault(normalized, number)
    canonical = " ".join(content.split())
    matches = []
    if captions:
        alternatives = sorted(captions, key=lambda caption: (-len(caption), captions[caption]))
        pattern = re.compile("|".join(re.escape(caption) for caption in alternatives))
        matches = pattern.finditer(canonical)
    spans = []
    position = 0
    for match in matches:
        if match.start() > position:
            spans.append(("TEXT", canonical[position:match.start()], None))
        spans.append(("IMAGE", match.group(), captions[match.group()]))
        position = match.end()
    if position < len(canonical):
        spans.append(("TEXT", canonical[position:], None))
    registry = {}
    text_number = 0
    seen_images = set()
    for kind, span, number in spans:
        if kind == "IMAGE":
            if number in seen_images:
                continue
            seen_images.add(number)
        for fragment_number, fragment in enumerate(_fragments(span), 1):
            if kind == "TEXT":
                text_number += 1
                reference = f"t{text_number}"
            else:
                reference = f"i{number}_{fragment_number}"
            registry[reference] = {"text": fragment, "support_kind": kind,
                                   "image_numbers": [number] if kind == "IMAGE" else []}
    return registry
