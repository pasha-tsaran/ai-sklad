"""Ground employee form data in their own words, never in model guesses."""

from dataclasses import dataclass
import re
import unicodedata

from pydantic import ValidationError

from app.schemas.contracts import EscalationDetails, EscalationExtraction


MAX_MESSAGE_CHARS = 8000
REQUIRED_FIELDS = ("account_holder_name", "account_login", "subject")
IDENTITY_FIELDS = ("account_holder_name", "account_login", "computer_name")
FIELDS = (*IDENTITY_FIELDS, "subject", "additional_info")  # Order of existing unlabelled forms.
_BASE = {"account_holder_name": "Иван", "account_login": "user", "computer_name": "PC01",
         "subject": "Проблема", "additional_info": ""}
_ALIASES = {
    "фио": "account_holder_name", "ф.и.о.": "account_holder_name", "имя": "account_holder_name",
    "фио пользователя рабочей станции": "account_holder_name", "имя пользователя": "account_holder_name",
    "логин": "account_login", "логин пользователя": "account_login",
    "компьютер": "computer_name", "имя компьютера": "computer_name", "пк": "computer_name",
    "имя пк": "computer_name", "рабочая станция": "computer_name",
    "устройство": "computer_name", "имя устройства": "computer_name", "тсд": "computer_name",
    "принтер": "computer_name", "терминал": "computer_name",
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
    "computer_name": r"(?:устройство|имя\s+устройства|тсд|принтер|компьютер(?:а)?|имя\s+(?:моего\s+)?компьютера|пк|имя\s+пк|"
                     r"рабочая\s+станция|терминал|хост|computer_name|(?:работаю|сижу)\s+на)",
    "subject": r"(?:тема(?:\s+проблемы)?|проблема|subject)",
    "additional_info": r"(?:описание|дополнительно|подробности|additional_info)",
}


@dataclass(frozen=True)
class IntakeResult:
    details: EscalationDetails | None
    fields: dict[str, str]
    missing: list[str]


def skips_device(message):
    text = " ".join(message.strip().casefold().rstrip(".!").split())
    return (text in {"-", "—", "нет", "дальше", "далее", "пропустить", "пропусти", "пропускаю", "пропустим",
                     "пропустить устройство", "без устройства", "без имени", "не хочу указывать", "не указано"}
            or bool(re.fullmatch(r"(?:я )?не (?:знаю|помню)(?: (?:имя|название)(?: устройства)?|,? как называется)?",
                                 text)))


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


def _result(fields, invalid=(), *, allow_completion=True, require_description=False):
    missing = [field for field in REQUIRED_FIELDS if field not in fields]
    if "computer_name" in invalid:
        missing.append("computer_name")
    if "additional_info" in invalid or require_description and not fields.get("additional_info", "").strip():
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
    lines = [line.strip() for line in message.strip().splitlines()]
    empty_device = len(lines) in {4, 5} and (not lines[2] or skips_device(lines[2]))
    if not empty_device:
        lines = [line for line in lines if line]
    if len(lines) not in {4, 5} or any(":" in line for line in lines[:3]):
        return None
    values = dict(zip(FIELDS, lines))
    if any(_valid(field, values[field]) is None for field in REQUIRED_FIELDS):
        return None
    if empty_device:
        del values["computer_name"]
        return values
    # Ordinary prose split into lines is not an ordered form. Bare lower-case
    # words such as English "to" are not enough evidence of a workstation ID.
    computer = values["computer_name"]
    if not (_valid("computer_name", computer) and (any(character.isdigit() for character in computer)
            or any(c in computer for c in "-_.") or len(computer) >= 2 and computer.isupper())):
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
                       for identity in IDENTITY_FIELDS):
                return True
    return False


def parse_intake(message, pending_fields, llm, *, requested_field=None, require_description=False):
    """Collect observed spans; Talk requires an independent description, older callers need not."""
    fields = {}
    if isinstance(pending_fields, dict):
        for field in FIELDS:
            value = _valid(field, pending_fields.get(field))
            if value is not None:
                fields[field] = value
    if not isinstance(message, str) or not message.strip() or len(message) > MAX_MESSAGE_CHARS:
        result = _result(fields, allow_completion=False, require_description=require_description)
        return result if result.missing else IntakeResult(None, fields, ["additional_info"])
    message = message.strip()
    if requested_field == "computer_name" and skips_device(message):
        fields.pop("computer_name", None)
        return _result(fields, require_description=require_description)
    labeled, invalid, structured = _labels(message)
    # A reply to the description question is the body itself. Keep its complete
    # wording/line breaks; do not ask a model to summarize or infer its topic.
    if requested_field == "additional_info" and (not labeled or not structured or (
            set(labeled) == {"subject"} and message.partition(":")[0].strip().casefold() == "проблема")):
        value = _valid("additional_info", message)
        if value is not None and value:
            fields["additional_info"] = value
            return _result(fields, require_description=require_description)
        fields.pop("additional_info", None)
        return _result(fields, ["additional_info"], require_description=require_description)
    ordered = _ordered_lines(message) if not labeled else None
    observed = labeled if labeled else ordered or {}
    for field, raw in observed.items():
        if field == "computer_name" and field not in invalid and (not raw or skips_device(raw)):
            fields.pop(field, None)
            continue
        value = _valid(field, raw)
        if value is None or field in invalid:
            invalid.add(field)
            fields.pop(field, None)
        else:
            fields[field] = value
    if structured or ordered is not None:
        return _result(fields, invalid, require_description=require_description)

    missing = [field for field in REQUIRED_FIELDS if field not in fields]
    if require_description and not fields.get("additional_info", "").strip():
        missing.append("additional_info")
    target = requested_field if requested_field in FIELDS else None
    if target is None:
        # Older turns without a recorded question can infer a single unambiguous field.
        unanswered = missing if "computer_name" in fields else [*missing, "computer_name"]
        if len(unanswered) == 1:
            candidate = unanswered[0]
            if candidate != "computer_name" or re.fullmatch(r"[A-Za-z0-9_.-]+", message):
                target = candidate
    if require_description and target == "subject" and not labeled:
        value = _valid("subject", message)
        if value is not None:
            fields["subject"] = value
            return _result(fields, invalid, require_description=True)
        fields.pop("subject", None)
        return _result(fields, [*invalid, "subject"], require_description=True)
    if (target and "\n" not in message and len(message.split()) <= 30 and not labeled
            and not (target == "computer_name" and (_PROBLEM_START.search(message)
                     or re.search(r"\b(?:логин|фио|меня\s+зовут)\b", message, re.I)))):
        value = _valid(target, message)
        if value is not None:
            fields[target] = value
            invalid.discard(target)
            return _result(fields, invalid, require_description=require_description)
    if not missing and "additional_info" not in invalid:
        return _result(fields, require_description=require_description)

    try:
        raw = llm.extract_escalation(message, {"pending_fields": dict(fields), "missing_fields": missing,
                                                "requested_field": requested_field if requested_field in FIELDS else None})
        extraction = EscalationExtraction.model_validate(raw).model_dump()
    except Exception:
        return _result(fields, invalid, require_description=require_description)
    for field, raw in extraction.items():
        if require_description and field == "additional_info":
            continue  # Only a separate observed description completes Talk intake.
        value = _valid(field, raw)
        if value is not None and _evidenced(field, value, message) and field not in invalid:
            fields[field] = value
    # Keep the complete original problem, not a model's shortened paraphrase.
    # Explicit label/ordered forms already keep their original description above.
    if not require_description and "subject" in fields and (start := _PROBLEM_START.search(message)):
        problem = message[start.start():].strip()
        if _valid("additional_info", problem) is not None:
            fields["additional_info"] = problem
        else:
            fields.pop("additional_info", None)
            invalid.add("additional_info")
    return _result(fields, invalid, require_description=require_description)
