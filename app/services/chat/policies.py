import re


MAX_ATTEMPTS = 5
TERMINAL_STATES = {"RESOLVED", "ESCALATED"}
FORM_ACTION = "COLLECT_ESCALATION_DETAILS"

FORM_TEXT = """Для передачи обращения заполните форму:

| Поле | Значение |
| --- | --- |
| ФИО пользователя рабочей станции | |
| Логин пользователя рабочей станции | |
| Имя компьютера | |
| Тема проблемы | |
| Дополнительное описание (необязательно) | |

После заполнения обязательных полей обращение будет передано в поддержку."""


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
