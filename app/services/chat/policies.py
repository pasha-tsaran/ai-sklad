import re


MAX_ATTEMPTS = 5
TERMINAL_STATES = {"RESOLVED", "ESCALATED"}
FORM_ACTION = "COLLECT_ESCALATION_DETAILS"

FORM_TEXT = """Для передачи обращения заполните форму:

| Поле | Значение |
| --- | --- |
| ФИО пользователя рабочей станции | |
| Логин пользователя рабочей станции | |
| Имя устройства (ПК, ТСД или принтер, необязательно) | |
| Тема проблемы | |
| Дополнительное описание (необязательно) | |

После заполнения обязательных полей обращение будет передано в поддержку."""

FORM_HELP = ("Отправьте данные одним сообщением. Строку «Устройство» можно пропустить.\n"
             "ФИО: ...\nЛогин: ...\nУстройство: ...\nТема: ...\nОписание: ...")


_REQUEST_OBJECT = r"(?:заявк\w*|обращени\w*|тикет\w*)"
_REQUEST_ACTION = (
    r"(?:остав(?:ить|лю|им|ьте|ь|лять)|оформ(?:ить|лю|им|ите|и|лять)|"
    r"созда(?:ть|м|дим|йте|й|вай)|под(?:ать|ам|адим|айте)|"
    r"состав(?:ить|лю|им|ьте|ь|лять)|завести|завед(?:у|ем|ите|и)|"
    r"отправ(?:ить|лю|им|ьте|ь|лять)|зарегистрир(?:овать|ую|уем|уйте))"
)
_SUPPORT_TARGET = r"(?:тп|тех\s*поддерж\w*|техническ\w*\s+поддерж\w*|поддержк\w*|ит[- ]отдел\w*)"
_REQUEST_FORM = re.compile(
    rf"\b(?:{_REQUEST_ACTION}\b(?:\s+\w+){{0,3}}\s+{_REQUEST_OBJECT}|"
    rf"{_REQUEST_OBJECT}\b(?:\s+\w+){{0,2}}\s+{_REQUEST_ACTION}|"
    rf"(?:хочу|нужн[оа]|надо|давайте)\s+{_REQUEST_OBJECT}|"
    rf"(?:написать|напиш(?:и|ите|ем|у)|писать|связаться)\b[^.!?;\n]{{0,45}}\b{_SUPPORT_TARGET}|"
    rf"письмо\b[^.!?;\n]{{0,25}}\b{_SUPPORT_TARGET})\b"
)
_REQUEST_NEGATION = (
    r"\b(?:не\s*(?:(?:хочу|хотел|хотела|буду|будем|надо|нужно|стоит|собираюсь|"
    r"могу|может|можем|могут|смог|смогла|смогли|смогу|получается|удается)"
    r"(?:\s+(?:сейчас|пока|больше|снова|еще|уже|вообще|пожалуйста|мне|нам)){0,2})?|ненадо|незачем)"
)
_NEGATED_REQUEST_PREFIX = re.compile(_REQUEST_NEGATION + r"\s*$")
_NEGATED_REQUEST_INTERNAL = re.compile(rf"(?:{_REQUEST_NEGATION}\s+{_REQUEST_ACTION}|\bне\s+{_REQUEST_OBJECT})\b")
_NEGATED_REQUEST_SUFFIX = re.compile(
    r"\s*(?:пока\s+)?не\s+(?:надо|нужно|хочу|буду|будем|стоит|могу|может|можем|могут|получается|удается)\b"
)
_CONDITIONAL_REQUEST_PREFIX = re.compile(r"\b(?:если|когда|вдруг)\b")
_QUOTED_TEXT = re.compile(r'«[^»]*»|“[^”]*”|"[^"\n]*"')

_LETTER_OBJECT = r"(?:заявк[ауи]|письм[оа]|обращени[ея]|тикет(?:а)?|сообщени[ея])"
_LETTER_PROCESS = rf"(?:отправку|оформление|создание)(?:\s+{_LETTER_OBJECT})?"
_DONT_SEND = r"не\s+(?:отправляй(?:те)?|посылай(?:те)?|оставляй(?:те)?|оформляй(?:те)?|создавай(?:те)?|надо|нужно)"
_LETTER_VERB = r"(?:оставлять|оставить|оформлять|оформить|отправлять|отправить|создавать|создать|подавать|подать)"
_LETTER_DECLINE = re.compile(
    rf"(?:(?:нет|пожалуйста|спасибо)[,\s]+)*(?:(?:я|давай(?:те)?)\s+)?(?:(?:уже|пока|лучше)\s+)?(?:"
    rf"отмен(?:а|и(?:те)?|ить|яй(?:те)?)(?:,\s*пожалуйста,?)?"
    rf"(?:\s+(?:{_LETTER_PROCESS}|{_LETTER_OBJECT}|е[её]|его|это))?|"
    rf"не\s+(?:(?:хочу|буду|будем|надо|нужно|стоит)\s+)?(?:ничего\s+)?{_LETTER_VERB}(?:\s+{_LETTER_OBJECT})?|"
    rf"не\s+(?:(?:хочу|будем|надо|нужно)\s+)?(?:писать|обращаться)\s+в\s+{_SUPPORT_TARGET}|"
    rf"не\s+(?:хочу|нужна|нужен|нужно)\s+{_LETTER_OBJECT}|"
    rf"{_DONT_SEND}(?:\s+(?:ничего|{_LETTER_OBJECT}))?|ничего\s+не\s+отправляй(?:те)?|"
    rf"{_LETTER_OBJECT}\s+(?:отмен(?:и(?:те)?|ить)|не\s+(?:надо|нужно)\s+{_LETTER_VERB}|"
    rf"(?:{_LETTER_VERB}\s+)?{_DONT_SEND}|не\s+нуж(?:на|но|ен))|"
    rf"без\s+(?:заявки|письма|обращения|тикета)|"
    rf"передумал(?:а|и)?(?:\s+{_LETTER_VERB}(?:\s+{_LETTER_OBJECT})?)?|отбой|не\s+сейчас|нет)"
    rf"(?:\s+в\s+{_SUPPORT_TARGET})?(?:[,\s]+(?:спасибо|пожалуйста|пока))?",
    re.IGNORECASE,
)
_NEXT_REQUEST = re.compile(
    r"\s+(?:(?:и|а|но)\s+)?(?=(?:теперь|лучше|расскажи(?:те)?|подскажи(?:те)?|помоги(?:те)?|"
    r"объясни(?:те)?|как|где|почему|что\s+делать)\b)", re.IGNORECASE,
)


def letter_decline_remainder(message: str) -> str | None:
    """Only a leading refusal of the pending letter is a command. Keep any next request verbatim."""
    text = message.strip()
    match = _LETTER_DECLINE.match(text)
    if not match:
        return None
    tail = text[match.end():]
    if not tail:
        return ""
    separator = re.match(r"\s*[,.;!:\n—–]+\s*", tail) or _NEXT_REQUEST.match(tail)
    if not separator:
        return None
    remainder = tail[separator.end():].strip()
    if remainder and match.group().casefold() == "нет" and not (
        "?" in remainder or _NEXT_REQUEST.match(" " + remainder)
    ):
        return None
    return remainder


_FAILED_OUTCOME = re.compile(
    r"\b(?:не\s+(?:помогло|сработало|заработало|работает)|"
    r"ничего\s+не\s+изменилось|проблема\s+(?:осталась|сохраняется)|"
    r"не\s+удалось\s+(?:решить|устранить)\s+проблему)\b"
)
_SUCCESSFUL_OUTCOME = re.compile(
    r"\b(?:помогло|сработало|заработало|все\s+работает|"
    r"(?:теперь|сейчас)\s+(?:все\s+)?работает|"
    r"проблема\s+(?:решена|устранена)|вопрос\s+решен|"
    r"удалось\s+(?:решить|устранить)\s+проблему)\b"
)
_UNCERTAIN_OUTCOME = re.compile(
    r"\b(?:если|вдруг|надеюсь|возможно|кажется|похоже|скажи|напиши|ответь|"
    r"не\s+уверен\w*)\b"
)
_PARTIAL_OUTCOME = re.compile(
    r"\b(?:но|однако|частично|кроме|остал(?:ось|ся|ась|ись)|"
    r"еще\s+(?:нужно|надо)|не\s+полностью|не\s+до\s+конца|не\s+все)\b"
)
_COMPLETION_ACKNOWLEDGEMENTS = {
    "да", "ну", "ура", "спасибо", "большое", "благодарю", "вас", "вам", "за", "помощь",
    "отлично", "хорошо", "вот", "все", "правильно", "верно", "теперь", "уже", "наконец", "то",
    "получилось", "удалось",
}


def resolution_feedback(message: str) -> str | None:
    """Recognize a reported overall outcome after an active instruction.

    Goal consent and completion of an individual step remain separate dialogue
    actions. The caller must check that the employee has received a guide.
    """
    text = _QUOTED_TEXT.sub(" ", message.casefold().replace("ё", "е"))
    if "?" in text:
        return None
    normalized = " ".join(re.sub(r"[^\w\s]", " ", text).split())
    if _UNCERTAIN_OUTCOME.search(normalized):
        return None
    failed = bool(_FAILED_OUTCOME.search(normalized))
    remaining = _FAILED_OUTCOME.sub(" ", normalized)
    succeeded = bool(_SUCCESSFUL_OUTCOME.search(remaining))
    if failed:
        progress = bool(re.search(r"(?<!не )\b(?:удалось|получилось)\b", remaining))
        return None if succeeded or progress else "FAILED"
    if _PARTIAL_OUTCOME.search(normalized):
        return None
    if succeeded:
        return "SUCCESS"
    words = set(normalized.split())
    has_completion = bool(words & {"удалось", "получилось"})
    has_closure = bool(words & {"спасибо", "благодарю", "все"})
    if has_completion and has_closure and words <= _COMPLETION_ACKNOWLEDGEMENTS:
        return "SUCCESS"
    return None


def requests_support_form(message: str) -> bool:
    """Recognize a current request to prepare support mail in ordinary wording."""
    normalized = " ".join(message.casefold().replace("ё", "е").split())
    normalized = _QUOTED_TEXT.sub("", normalized)
    for clause in re.split(r"[.!?;]+|,\s*(?:но|а)\s+", normalized):
        clause = clause.replace(",", " ")
        for request in _REQUEST_FORM.finditer(clause):
            prefix = clause[:request.start()]
            if (_NEGATED_REQUEST_PREFIX.search(prefix) or _CONDITIONAL_REQUEST_PREFIX.search(prefix)
                    or _NEGATED_REQUEST_INTERNAL.search(request.group())
                    or _NEGATED_REQUEST_SUFFIX.match(clause[request.end():])):
                continue
            return True
    return False


def direct_intent(message: str) -> str | None:
    normalized = message.strip().lower().rstrip(".!?")
    if normalized in {"оператор", "специалист", "позвать специалиста"} or requests_support_form(message):
        return "HUMAN"
    if normalized in {"помогло", "не помогло"}:
        return "FEEDBACK"
    return None
