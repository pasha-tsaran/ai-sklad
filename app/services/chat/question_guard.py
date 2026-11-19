"""Keep clarification questions observational and understandable to the employee."""

import re


_INTERNAL_ID = re.compile(r"\b(?:device|printer|scanner|terminal|entity|asset|host|equipment|goal|condition)_\d+\b", re.I)
_KEY_CHORD = re.compile(r"\b(?:ctrl|control|alt|shift|win|cmd|command)\s*(?:[+\-]\s*[a-z0-9]+\s*)+", re.I)
_TECHNICAL_TERM = re.compile(r"\b(?:zpl|cpcl|ipsec|opos|snmp|lpr|imap|pop3|tcp\s*/\s*ip)\b", re.I)
_TECHNICAL_COMMAND = re.compile(r"\b(?:gpupdate|regedit|bcdedit|powershell|cmd\.exe|net\s+(?:stop|start)|"
                                r"sc\s+(?:sdset|stop|start)|sfc\s*/|dism\s*/)\b", re.I)
_NEW_OPERATION = re.compile(r"\b(?:перезагруз(?:или|ите)|выключ(?:или|ите)|"
                            r"отключ(?:али|или|ите)|удал(?:яли|или|ите)|"
                            r"сброс(?:или|ьте)|переустанов(?:или|ите))\b", re.I)
_TRIAL_ACTION = re.compile(r"\b(?:пробовал\w*|пытал\w*)\s+(?:ли\s+|уже\s+)?"
                          r"(?:нажать|нажимать|запустить|открыть|выполнить|изменить|установить)\b", re.I)
_TECHNICAL_PARAMETER = re.compile(
    r"\b(?:какой|какие|укажите|назовите)\s+(?:именно\s+)?(?:ip[ -]?(?:адрес\w*)?|"
    r"порт\w*|драйвер\w*|протокол\w*|адрес\s+(?:(?:или\s+)?имя\s+)?(?:сервера|шлюза)|"
    r"имя\s+сервера|(?:тип|метод)\s+аутентификации)\b", re.I,
)
_ACKNOWLEDGEMENTS = {"понял", "поняла", "спасибо", "хорошо", "уточню", "давайте уточним"}
_MODEL_IDENTIFICATION_REQUEST = re.compile(
    r"\b(?:какая|какую|какие|какой|какого|кто|назовите|укажите|напишите|сообщите|уточните|посмотрите)"
    r"[\s,:]*(?:(?:именно|точная|точную|полная|полную|пожалуйста)\s*[,\s]*){0,3}"
    r"(?:модел\w*|производител\w*|маркировк\w*|бренд\w*|марка|марку)\b|"
    r"\bчто\s+написано\s+на\s+(?:шильдик\w*|наклейк\w*\s+(?:с|на)\s+корпус\w*)\b", re.I,
)


def _normalized(value):
    return " ".join(value.casefold().replace("ё", "е").split())


def asks_model_identification(question):
    return bool(_MODEL_IDENTIFICATION_REQUEST.search(question))


def unsupported_question_content(question, reported_messages=()):
    """Questions cannot introduce a technical test, action or unexplained identifier."""
    reports = [_normalized(text) for text in reported_messages if isinstance(text, str)]
    for expression in (_INTERNAL_ID, _TECHNICAL_TERM, _TECHNICAL_COMMAND, _NEW_OPERATION):
        for match in expression.finditer(question):
            token = _normalized(match.group())
            if not any(token in text for text in reports):
                return True
    for match in _KEY_CHORD.finditer(question):
        chord = re.sub(r"\s+", "", match.group()).casefold().replace("-", "+")
        if not any(chord in re.sub(r"\s+", "", text).replace("-", "+") for text in reports):
            return True
    if _TRIAL_ACTION.search(question) or _TECHNICAL_PARAMETER.search(question) or asks_model_identification(question):
        return True
    first_sentence = re.match(r"(.+?)[.!]\s+\S", question)
    if first_sentence and _normalized(first_sentence[1]) not in _ACKNOWLEDGEMENTS:
        return True
    return False
