"""Expose original instruction applicability before an employee accepts a plan."""

import copy
import hashlib
import json
import re

from app.services.rag.catalog import instruction_sections
from app.services.rag.query import normalized_query


SCOPE_LABEL = re.compile(r"^(?:назначение|область применения|сценарий|проблема|когда использовать|"
                         r"purpose|scope|scenario)\s*[:.]?$", re.I)
GENERIC_HEADING = re.compile(r"^(?:назначение|область применения|сценарий|проблема|когда использовать|"
                             r"фотографии(?: и пояснения)?|иллюстрации|шаг\s*\d*|страница(?: PDF)?\s*\d*)$", re.I)
PROCEDURE_START = re.compile(r"^(?:\d+[.)]\s|(?:шаг|step|этап)\s*\d|"
                             r"(?:откройте|нажмите|выберите|введите|включите|выключите|снимите|"
                             r"установите|проверьте|перейдите|подключите|зайдите|заходим)\b)", re.I)
PARAMETER_VALUE = re.compile(r"\b(?:ZPL|CPCL|TCP/IP|USBHID|ScanOPOS|IMAP|POP3|SMTP|num_ctx)\b", re.I)
LEAD_ACTION = re.compile(r"^(?:(?:сначала|затем|далее|теперь|после этого)\s+)?"
                         r"(?:[а-яё-]+(?:йте|ите)\b|войти\b|открыть\b|нажать\b|выбрать\b)", re.I)
TABLE_METADATA = re.compile(r"^(?:Лист:\s|[A-Z]+[1-9]\d*:\s)", re.I)
COMMAND_TEXT = re.compile(r"(?:`|%[A-Za-z_][A-Za-z0-9_]*%|(?:^|\s)/[A-Za-z?](?:\s|$))")


def confirmation_scope_identity(entries):
    """Identify ordered original applicability statements, independently of the task."""
    canonical = []
    for entry in entries:
        source = entry["source"]
        canonical.append({"file_id": source["file_id"], "sha256": source["sha256"],
                          "title": entry["title"], "title_section_id": entry.get("title_section_id"),
                          "context": [{"section_id": item["section_id"], "text": item["text"]}
                                      for item in entry.get("context", [])]})
    return hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _plain(text):
    return " ".join(text.lstrip("# ").split()).rstrip(":")


def _short_original(text):
    text = _plain(text)
    if len(text) <= 220:
        return text
    boundaries = [match.end() for match in re.finditer(r"[.!?](?:\s|$)", text[:220])]
    return text[:boundaries[-1]].strip() if boundaries else ""


def _scope_heading(section):
    text = _plain(section["text"])
    return (section.get("is_heading") and not section.get("condition_section_id")
            and not GENERIC_HEADING.fullmatch(text) and not PROCEDURE_START.search(text)
            and not LEAD_ACTION.search(text) and not COMMAND_TEXT.search(text))


def _scope_sections(sections, selected):
    selected_titles = {_plain(section["title"]) for section in selected}
    selected_ids = {section["id"] for section in selected}
    first_action = next((index for index, section in enumerate(sections)
                         if not section.get("is_heading") and PROCEDURE_START.search(_plain(section["text"]))),
                        len(sections))
    headings, purpose = [], []
    for index, section in enumerate(sections):
        text = _plain(section["text"])
        if (_scope_heading(section)
                and (text in selected_titles or not selected and index < min(first_action, 3))):
            headings.append(section)
        if SCOPE_LABEL.fullmatch(text) and index + 1 < len(sections):
            following = sections[index + 1]
            if (not following.get("is_heading") and not PROCEDURE_START.search(_plain(following["text"]))
                    and (not selected or following["id"] in selected_ids or index < first_action)):
                purpose.append(following)
    return headings + purpose


def _declarative_lead(sections):
    for section in sections:
        text = _plain(section["text"])
        if section.get("condition_section_id") or PROCEDURE_START.search(text) or LEAD_ACTION.search(text):
            break
        if (section["kind"] != "WARNING" and not GENERIC_HEADING.fullmatch(text)
                and not PARAMETER_VALUE.search(text) and not TABLE_METADATA.search(text)
                and not COMMAND_TEXT.search(text) and _short_original(text)):
            return section
    return None


def _conditional_context(sections, selected):
    originals = {section["id"]: section for section in sections}
    context = {}
    for section in selected:
        reference = section.get("condition_section_id")
        condition = section.get("condition_text")
        if section.get("parent_group_id") and isinstance(condition, str) and reference in originals:
            text = _short_original(condition)
            if text and _plain(originals[reference]["text"]).startswith(text):
                context[reference] = {"section_id": reference, "text": text}
    return list(context.values())


def _excerpt_scopes(sections, selected):
    """Use the selected fact's logical section, excluding inherited safety context."""
    anchors = ([section for section in selected if section.get("image_numbers")]
               or [section for section in selected
                   if not section.get("is_heading") and section["kind"] not in {"WARNING", "PREREQUISITE"}]
               or [section for section in selected
                   if not section.get("is_heading") and len(_plain(section["text"])) >= 20])
    groups = {section["group_id"] for section in anchors or selected}
    headings = {}
    for section in sections:
        if section["group_id"] in groups and _scope_heading(section):
            headings.setdefault(section["group_id"], section)
    return list(headings.values())


def _operation_context(sections, evidence):
    """Expose literal descriptive operation proof, after checking original membership."""
    originals = {section["id"]: section for section in sections}
    context = []
    has_text_context = False
    ordered = sorted((item for item in evidence if isinstance(item, dict)),
                     key=lambda item: item.get("support_kind") != "TEXT")
    for item in ordered:
        if has_text_context and item.get("support_kind") != "TEXT":
            break
        section = originals.get(item.get("section_id"))
        quote = item.get("text")
        numbers = item.get("image_numbers", [])
        original_text = " ".join(section["text"].split()) if section else ""
        normalized_quote = " ".join(quote.split()) if isinstance(quote, str) else ""
        if (section is None or not isinstance(quote, str) or not quote.strip()
                or not normalized_quote or normalized_quote not in original_text
                or item.get("support_kind") not in {"TEXT", "CAPTION", "IMAGE", "IMAGE_CAPTION"}
                or not isinstance(numbers, (tuple, list)) or any(type(number) is not int for number in numbers)
                or not set(numbers) <= set(section.get("image_numbers", []))):
            continue
        start = original_text.index(normalized_quote)
        end = start + len(normalized_quote)
        sentences = []
        for match in re.finditer(r".+?(?:[.!?](?=\s|$)|$)", original_text):
            overlap = min(end, match.end()) - max(start, match.start())
            if overlap > 0:
                sentences.append((overlap, match.start(), match.group()))
        # Canonical evidence is a bounded fragment. Recover only complete
        # original sentences that overlap it, prioritizing its largest span.
        for _, _, sentence in sorted(sentences, key=lambda item: (-item[0], item[1])):
            text = _plain(sentence)
            if (not 5 <= len(text) <= 220 or PROCEDURE_START.search(text) or LEAD_ACTION.search(text)
                    or PARAMETER_VALUE.search(text) or TABLE_METADATA.search(text) or COMMAND_TEXT.search(text)
                    or re.search(r"[→<>`]|\b(?:сканируйте|отсканируйте|считайте|выполните|код\w*|"
                                 r"откройте|нажмите|выберите|введите|включите|выключите|снимите|"
                                 r"установите|проверьте|перейдите|подключите|зайдите)\b", text, re.I)
                    or GENERIC_HEADING.fullmatch(text)):
                continue
            entry = {"section_id": section["id"], "text": text}
            if entry not in context:
                context.append(entry)
                has_text_context = has_text_context or item.get("support_kind") == "TEXT"
            if len(context) >= 2:
                break
        if len(context) >= 2:
            break
    return context


def confirmation_scope_entries(approved, parts=()):
    """Keep source/section references for every distinct applicability statement."""
    documents = [(part.approved, part.sections, getattr(part, "delivery_mode", "PROCEDURE"),
                  getattr(part, "confirmation_evidence", ()))
                 for part in parts] or [(approved, (), "PROCEDURE", ())]
    entries, seen = [], set()
    for original, selected, mode, evidence in documents:
        sections = instruction_sections(original.document)
        heading = next((section for section in sections if _scope_heading(section)), None)
        if heading is None and mode != "EXCERPT":
            heading = _declarative_lead(sections)
        title = _plain(heading["text"] if heading else original.candidate.title)
        if len(title) > 220:
            title = title[:220].rsplit(" ", 1)[0]
        conditions = _conditional_context(sections, selected)
        operation = [] if mode == "EXCERPT" else [item for item in _operation_context(sections, evidence)
                                                  if normalized_query(item["text"]) != normalized_query(title)]
        context = conditions + operation
        used = {normalized_query(title), *(normalized_query(item["text"]) for item in context)}
        scopes = _excerpt_scopes(sections, selected) if mode == "EXCERPT" else _scope_sections(sections, selected)
        scope_count = 0
        for section in (() if conditions else scopes):
            if operation and section.get("image_numbers") and not section.get("is_heading"):
                continue
            text = _short_original(section["text"])
            key = normalized_query(text)
            if text and key not in used and not PARAMETER_VALUE.search(text) and not COMMAND_TEXT.search(text):
                context.append({"section_id": section["id"], "text": text})
                used.add(key)
                scope_count += 1
            if scope_count >= 2:
                break
        source = original.document["source"]
        identity = source["file_id"], source["sha256"], title, tuple(item["text"] for item in context)
        if identity not in seen:
            entries.append({"source": copy.deepcopy(source), "title": title,
                            "title_section_id": heading["id"] if heading else None, "context": context})
            seen.add(identity)
        if len(entries) >= 8:
            break
    return entries


def render_confirmation_scopes(entries):
    descriptions = []
    for entry in entries:
        descriptions.append("Нашлась инструкция «" + entry["title"] + "».")
        descriptions.extend("В ней описано «" + item["text"] + "»." for item in entry["context"])
    return "\n".join(descriptions)
