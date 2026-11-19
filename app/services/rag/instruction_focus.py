"""Bind narrowly recognized questions to complete evidence in approved originals."""

from dataclasses import dataclass
import re

from app.services.rag.query import _connections, _identifiers, normalized_query, printer_kinds


@dataclass(frozen=True)
class FocusedExcerpt:
    kind: str
    sections: tuple[dict, ...]


def _find(sections, pattern):
    return tuple(section for section in sections if re.search(pattern, section["text"], re.I))


def focused_question(message):
    """Recognize only explicit purposes and conditions. All other turns use regular search."""
    text = normalized_query(message)
    if ("bluetooth" in _connections(text) and re.search(r"\burovo\b", text)
            and re.search(r"\b(?:тсд|ручн\w* терминал\w*)\b", text)
            and re.search(r"\b(?:один|одного|одном) (?:принтер\w* )?urovo\b", text)
            and re.search(r"\b(?:настро\w*|печать)\b", text)
            and not re.search(r"\b(?:два|двум|двух|2|нескольк\w*|usb|ethernet|wifi)\b", text)
            and not re.search(r"bluetooth в labelprint|в labelprint.{0,25}bluetooth|включ\w+ bluetooth", text)):
        return "linked_printer"
    if (re.search(r"\bлент\w*\b", text) and re.search(r"провер\w+ печат\w+|пробн\w+ печат\w+", text)
            and re.search(r"\bзамен\w+\b", text) and _identifiers(text)[0]):
        return "ribbon_test"
    if (re.search(r"\bчто\b.{0,38}\bподготов\w+\b", text)
            and re.search(r"\b(?:обнов\w+|прошив\w+)\b", text)
            and not re.search(r"\bкак\s+(?:обнов\w+|прош\w+)|что нажимать", text)
            and printer_kinds(text) == {"thermal_mobile"}):
        return "firmware_preparation"
    if (re.search(r"\bприслал\w*\b.{0,25}\bфайл\w*\b", text)
            and re.search(r"\bправа? (?:служб\w+ )?печат\w*\b", text)
            and re.search(r"удал\w+.{0,26}(?:задани\w+|очеред\w+)", text)):
        return "file_purpose"
    if (re.search(r"\b(?:этикет\w*|накле\w*)\b", text)
            and (re.search(r"таблиц\w* (?:исходн\w*|заводск\w*) размер\w*", text)
                 or re.search(r"ширин\w* и высот\w* .{0,45}стандартн\w*", text)
                 and re.search(r"\bсбрасыва\w*\b.{0,35}\bне надо\b", text))):
        return "default_dimensions"
    if (printer_kinds(text) == {"thermal_mobile"} and re.search(r"бледн\w+|светл\w+", text)
            and re.search(r"(?:регулиров\w+ (?:темнот\w+|контраст\w+)|сделать печать темнее)", text)):
        return "print_contrast"
    return None


def focus_covers_goals(focus, goals):
    if len(goals) == 1:
        return True
    if focus != "ribbon_test" or len(goals) != 2:
        return False
    first, second = (normalized_query(goal) for goal in goals)
    ribbon = re.compile(r"\b(?:лент\w*|риббон\w*)\b")
    test_print = re.compile(r"(?:провер\w+|пробн\w+) печат\w+")
    return bool(ribbon.search(first) and test_print.search(second)
                or ribbon.search(second) and test_print.search(first))


def _default_dimensions(sections):
    result = []
    for position, section in enumerate(sections):
        if not re.search(r"^параметр\s+стандартная настройка$", section["text"], re.I):
            continue
        rows = []
        for row in sections[position + 1:position + 17]:
            if row.get("is_heading") or re.match(r"^\s*\d+[.)]", row["text"]):
                break
            if re.match(r"^Label (?:width|height)\b", row["text"], re.I) and re.search(
                    r"\d[\d,.]*\s*(?:мм|mm|дюйм|inch)", row["text"], re.I):
                rows.append(row)
        labels = {row["text"].split()[1].casefold() for row in rows}
        if len(rows) == 2 and labels == {"width", "height"}:
            result.extend((section, *rows))
    return tuple(result) if len(result) == 3 else ()


def focused_excerpt(focus, message, source, sections):
    """Return literal sections only when every relevant property is present together."""
    sections = tuple(sections)
    title = source["filename"]
    requested_models = _identifiers(message)[0]
    if requested_models and not requested_models <= _identifiers(title)[0]:
        return None

    kind = "LOOKUP"
    selected = ()
    if focus == "linked_printer":
        diagram = tuple(section for section in sections if section.get("image_numbers")
                        and re.search(r"ТСД\s*→\s*Urovo по Bluetooth", section["text"], re.I)
                        and re.search(r"LabelPrint.{0,65}TCP/IP", section["text"], re.I))
        steps = _find(sections, r"^\d+[.)]\s")
        if (len(diagram) == 1 and 3 <= len(steps) <= 8
                and _find(steps, r"^\d+\. В LabelPrint выберите .*TCP/IP")
                and _find(steps, r"Не переключайте LabelPrint на Bluetooth")
                and _find(steps, r"Новый LabelPrint") and _find(steps, r"тестов\w+ печать")):
            selected = (*steps, *diagram)
            kind = "ACTION"
    elif focus == "ribbon_test":
        steps = _find(sections, r"^\d+[.)]\s")
        if (3 <= len(steps) <= 8 and _find(steps, r"риббон по центру")
                and _find(steps, r"калибровк\w+ данн\w+ оборудовани\w+")
                and _find(steps, r"пробн\w+ печать.{0,75}этикетка TEST")):
            selected = steps
            kind = "ACTION"
    elif focus == "firmware_preparation":
        requirements = _find(sections, r"Нужны согласованный файл.{0,100}стабильное питание.{0,100}восстановить")
        setup = _find(sections, r"^\d+\..{0,70}\bUSB\b.{0,60}\bConsole\b")
        warning = _find(sections, r"\bНе отключайте питание или USB во время записи\.")
        if len(requirements) == len(setup) == len(warning) == 1:
            selected = (*requirements, *setup, *warning)
    elif focus == "file_purpose":
        selected = _find(sections, r"меняет права на служб\w+.{0,50}не чистит очередь")
    elif focus == "default_dimensions":
        if not re.search(r"\btsc\b", normalized_query(message)) or re.search(
                r"\btsc\b", normalized_query(title.replace("_", " "))):
            selected = _default_dimensions(sections)
    elif focus == "print_contrast":
        names = _find(sections, r"^\[name\]\s+set contrast command$")
        details = _find(sections, r"Set the print contrast\..{0,40}the darker the print")
        if len(names) == len(details) == 1 and names[0]["group_id"] == details[0]["group_id"]:
            selected = (*names, *details)
    if selected:
        return FocusedExcerpt(kind, selected)
    return None


def preparation_warning_excerpt(section):
    """Render the original safety sentence in a preparation-only answer."""
    text = section["text"]
    if section["kind"] == "WARNING":
        match = re.search(r"\bНе отключайте [^.\n]+ во время записи\.$", text)
        if match:
            return match[0]
    return text
