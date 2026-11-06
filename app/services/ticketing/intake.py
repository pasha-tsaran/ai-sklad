"""Ground employee form data in their own words, never in model guesses."""

from dataclasses import dataclass
import re
import unicodedata

from pydantic import ValidationError

from app.schemas.contracts import EscalationDetails, EscalationExtraction


MAX_MESSAGE_CHARS = 8000
REQUIRED_FIELDS = ("account_holder_name", "account_login", "computer_name", "subject")
FIELDS = (*REQUIRED_FIELDS, "additional_info")
_BASE = {"account_holder_name": "Иван", "account_login": "user", "computer_name": "PC01",
         "subject": "Проблема", "additional_info": ""}
_ALIASES = {
    "фио": "account_holder_name", "ф.и.о.": "account_holder_name", "имя": "account_holder_name",
    "фио пользователя рабочей станции": "account_holder_name", "имя пользователя": "account_holder_name",
    "логин": "account_login", "логин пользователя": "account_login",
    "компьютер": "computer_name", "имя компьютера": "computer_name", "пк": "computer_name",
    "имя пк": "computer_name", "рабочая станция": "computer_name",
    "тема": "subject", "тема проблемы": "subject", "проблема": "subject",
    "описание": "additional_info", "дополнительно": "additional_info",
    "дополнительное описание": "additional_info", "подробности": "additional_info",
    **{field: field for field in FIELDS},
}
_LOGIN = re.compile(r"^[A-Za-zА-Яа-яЁё0-9_][A-Za-zА-Яа-яЁё0-9_.@+\\-]*$")
_NAME_REJECT = re.compile(
    r"\b(?:я|мне|меня|мой|моя|мои|зовут|логин|компьютер|сканер|принтер|пароль|"
    r"sudo|curl|reboot|reset|rm|выполни\w*|удали\w*|игнорируй\w*|открой\w*|"
    r"перезагрузи\w*|настрой\w*|установи\w*|нужно|надо|помоги\w*|привет|здравствуй\w*|"
    r"спасибо|благодар\w*|пожалуйста|готово)\b", re.I,
)
_PROBLEM = re.compile(
    r"\b(?:нуж\w*|надо|необходим\w*|хочу|требуется|проблем\w*|ошиб\w*|неполад\w*|"
    r"настро\w*|сканер\w*|принтер\w*|печат\w*|громк\w*|звук\w*|доступ\w*|"
    r"работ\w*|запуск\w*|подключ\w*|установ\w*|сбой\w*|1с|1c)\b", re.I,
)
_PROBLEM_START = re.compile(
    r"\b(?:нужно|нужна|нужен|надо|необходимо|хочу|требуется|проблема|ошибка|"
    r"не\s+(?:работает|печатает|запускается|получается|могу))\b", re.I,
)
_EVIDENCE = {
    "account_holder_name": r"(?:меня\s+зовут|мо[её]\s+имя|фио(?:\s+пользователя)?|я)",
    "account_login": r"(?:логин(?:\s+пользователя)?|уч[её]тная\s+запись|username|account_login|"
                     r"(?:вхожу|захожу|авторизуюсь)\s+(?:как|под))",
    "computer_name": r"(?:компьютер(?:а)?|имя\s+(?:моего\s+)?компьютера|пк|имя\s+пк|"
                     r"рабочая\s+станция|терминал|хост|computer_name|(?:работаю|сижу)\s+на)",
    "subject": r"(?:тема(?:\s+проблемы)?|проблема|subject)",
    "additional_info": r"(?:описание|дополнительно|подробности|additional_info)",
}


@dataclass(frozen=True)
class IntakeResult:
    details: EscalationDetails | None
    fields: dict[str, str]
    missing: list[str]


def _valid(field, value):
    if field not in FIELDS or not isinstance(value, str):
        return None
    value = value.strip()
    if any(unicodedata.category(character) == "Cc" and (field != "additional_info" or character not in "\r\n\t")
           for character in value):
        return None
    if field == "account_holder_name":
        if (_NAME_REJECT.search(value) or len(value.split()) > 6
                or not all(character.isalpha() or character in " .'-’" for character in value)
                or not any(character.isalpha() for character in value)):
            return None
    if field == "account_login" and not _LOGIN.fullmatch(value):
        return None
    if field == "computer_name" and not any(character.isalnum() for character in value):
        return None
    try:
        return EscalationDetails.model_validate({**_BASE, field: value}).model_dump()[field]
    except ValidationError:
        return None


def _result(fields, invalid=(), *, allow_completion=True):
    missing = [field for field in REQUIRED_FIELDS if field not in fields]
    if "additional_info" in invalid:
        missing.append("additional_info")
    details = None
    if not missing and allow_completion:
        try:
            details = EscalationDetails.model_validate(fields)
        except ValidationError:
            pass
    return IntakeResult(details, dict(fields), missing)


def _labels(message):
    candidates, invalid, structured = {}, set(), True
    current = None
    for line in message.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        if not line.strip():
            if current == "additional_info":
                candidates[current] += "\n"
            continue
        label, separator, value = line.partition(":")
        key = _ALIASES.get(" ".join(label.strip().strip("* ").casefold().split())) if separator else None
        if key:
            value = value.strip()
            if key in candidates and candidates[key] != value:
                invalid.add(key)
            candidates[key] = value
            current = key
            if re.search(r"\b(?:логин|компьютер|фио|тема)\s*:", value, re.I):
                structured = False
        elif current == "additional_info":
            candidates[current] += "\n" + line
        else:
            structured = False
            current = None
    return candidates, invalid, bool(candidates) and structured


def _ordered_lines(message):
    lines = [line.strip() for line in message.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]
    if len(lines) not in {4, 5} or any(":" in line for line in lines[:3]):
        return None
    values = dict(zip(FIELDS, lines))
    if any(_valid(field, values[field]) is None for field in REQUIRED_FIELDS):
        return None
    # Ordinary prose split into lines is not an ordered form. Bare lower-case
    # words such as English "to" are not enough evidence of a workstation ID.
    computer = values["computer_name"]
    if not (any(character.isdigit() for character in computer) or any(c in computer for c in "-_.")
            or len(computer) >= 2 and computer.isupper()):
        return None
    return values


def _spans(message, value, field):
    for match in re.finditer(re.escape(value), message):
        before, after = match.start() - 1, match.end()
        if before >= 0 and (message[before].isalnum() or message[before] == "_"):
            continue
        if after < len(message) and (message[after].isalnum() or message[after] == "_"):
            continue
        if field in {"account_login", "computer_name"}:
            if after < len(message) and message[after] in "@+\\-":
                continue
            if after + 1 < len(message) and message[after] == "." and message[after + 1].isalnum():
                continue
        yield match


def _evidenced(field, value, message):
    if not value:
        return False
    for span in _spans(message, value, field):
        prefix = message[max(0, span.start() - 120):span.start()]
        if re.search(r"\b" + _EVIDENCE[field] + r"\s*(?::|=|—|-)?\s*(?:это\s+)?[«\"']?\s*$", prefix, re.I):
            return True
        if field in {"subject", "additional_info"} and _PROBLEM.search(value):
            if not any(re.search(r"\b" + _EVIDENCE[identity]
                                 + r"\s*(?::|=|—|-)?\s*[«\"']?\s*$", prefix, re.I)
                       for identity in REQUIRED_FIELDS[:3]):
                return True
    return False


def parse_intake(message, pending_fields, llm, *, requested_field=None):
    """Collect only observed employee spans; return one ordered list of missing fields."""
    fields = {}
    if isinstance(pending_fields, dict):
        for field in FIELDS:
            value = _valid(field, pending_fields.get(field))
            if value is not None:
                fields[field] = value
    if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE_CHARS:
        result = _result(fields, allow_completion=False)
        return result if result.missing else IntakeResult(None, fields, ["additional_info"])
    message = message.strip()
    labeled, invalid, structured = _labels(message)
    ordered = _ordered_lines(message) if not labeled else None
    observed = labeled if labeled else ordered or {}
    for field, raw in observed.items():
        value = _valid(field, raw)
        if value is None or field in invalid:
            invalid.add(field)
            fields.pop(field, None)
        else:
            fields[field] = value
    if structured or ordered is not None:
        return _result(fields, invalid)

    missing = [field for field in REQUIRED_FIELDS if field not in fields]
    target = requested_field if requested_field in FIELDS else missing[0] if len(missing) == 1 else None
    if target and "\n" not in message and len(message.split()) <= 30 and not labeled:
        value = _valid(target, message)
        if value is not None:
            fields[target] = value
            invalid.discard(target)
            return _result(fields, invalid)
    if not missing and "additional_info" not in invalid:
        return _result(fields)

    try:
        raw = llm.extract_escalation(message, {"pending_fields": dict(fields), "missing_fields": missing,
                                                "requested_field": requested_field if requested_field in FIELDS else None})
        extraction = EscalationExtraction.model_validate(raw).model_dump()
    except Exception:
        return _result(fields, invalid)
    for field, raw in extraction.items():
        value = _valid(field, raw)
        if value is not None and _evidenced(field, value, message) and field not in invalid:
            fields[field] = value
    # Keep the complete original problem, not a model's shortened paraphrase.
    # Explicit label/ordered forms already keep their original description above.
    if "subject" in fields and (start := _PROBLEM_START.search(message)):
        problem = message[start.start():].strip()
        if _valid("additional_info", problem) is not None:
            fields["additional_info"] = problem
        else:
            fields.pop("additional_info", None)
            invalid.add("additional_info")
    return _result(fields, invalid)
