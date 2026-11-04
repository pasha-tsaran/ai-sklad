"""Bounded conversational routing; technical actions still require approved knowledge.

These checks complement capability restrictions and model instructions. They do not
claim to recognize every possible prompt injection or confidential value.
"""

import re

from app.services.chat.memory import _redact


REFUSAL = (
    "Пароли, ключи доступа, чужая переписка и служебные настройки защищены. "
    "Изменение внутренних данных выполняется администратором через предусмотренные средства управления. "
    "Могу помочь с рабочей проблемой или обычным вопросом."
)

_SECRET_REQUEST = re.compile(
    r"(?:покажи|выдай|пришли|раскрой|сообщи|напиши|скопируй|выведи|прочитай|"
    r"предоставь|расскажи|верни|закодируй|show|reveal|print|give|dump|expose|encode).{0,100}"
    r"(?:парол\w*|password\w*|api[ _-]?(?:key|ключ)\w*|токен\w*|secret\w*|"
    r"секретн\w*|секрет\s+(?:системы|компании|пользователя)|конфиденциальн\w*|"
    r"чуж\w*\s+(?:переписк\w*|данн\w*)|переписк\w*\s+(?:друг\w*|чуж\w*)|"
    r"системн\w*\s+промпт\w*|system\s+prompt|"
    r"(?:openwebui|postgres|smtp)[_ -](?:api[_ -]key|password|secret))",
    re.I | re.S,
)
_INTERNAL_CHANGE = re.compile(
    r"(?:удали|сотри|очисти|уничтожь|обнули|забудь|delete|erase|wipe|drop|forget).{0,100}"
    r"(?:все\s+данные|все\s+правила|все\s+инструкции|всю\s+память|всю\s+историю|"
    r"из\s+себя|внутри\s+себя|тво\w*\s+(?:данн\w*|памят\w*|истори\w*|правил\w*)|"
    r"сво\w*\s+(?:данн\w*|памят\w*|истори\w*|правил\w*)|баз\w*\s+(?:знани\w*|данн\w*)|"
    r"all\s+(?:data|rules|instructions|memory)|your\s+(?:data|memory|history))",
    re.I | re.S,
)
_TECHNICAL = re.compile(
    r"\b(?:принтер\w*|сканер\w*|компьютер\w*|ноутбук\w*|клавиатур\w*|мыш[ьи]\w*|"
    r"монитор\w*|драйвер\w*|устройств\w*|штрихкод\w*|перепрош\w*|прошив\w*|"
    r"настрой\w*|установ\w*|удал\w*|перезагруз\w*|подключ\w*|сброс\w*|"
    r"кнопк\w*|галочк\w*|пункт\w*\s+меню|операции|торгов\w*\s+оборудован\w*|"
    r"ошибк\w*|неисправн\w*|печата\w*|1с|1c|pdf|acrobat|windows|linux|docker|"
    r"nextcloud|openwebui|zebra|supermag|usb|wifi|bios|uart|sql|powershell|cmd|"
    r"ready|готово|получилось|помогло|дальше|продолжай|оператор)\b",
    re.I,
)
_TECHNICAL_OUTPUT = re.compile(
    r"```|`[^`]+`|https?://|/api/|\b(?:нажмите|откройте|выберите|введите|удалите|"
    r"снимите|перейдите|включите|перезапустите|запустите|выполните|sudo|rm\s+-|"
    r"drop\s+table|net\s+stop|systemctl|docker\s+(?:exec|run))\b",
    re.I,
)


_OPERATION_CLAIM = re.compile(
    r"\b(?:заявк\w*|обращени\w*|письм\w*|ticket\w*|e[- ]?mail\w*)\b.{0,60}"
    r"\b(?:создан\w*|сформирован\w*|отправлен\w*|передан\w*|зарегистрирован\w*|"
    r"created|sent|submitted|forwarded)\b|"
    r"\b(?:данн\w*|памят\w*|истори\w*|баз\w*\s+данн\w*|data|memory|history|database)\b.{0,60}"
    r"\b(?:удал[её]н\w*|ст[её]рт\w*|очищен\w*|сброшен\w*|deleted|erased|cleared|wiped)\b|"
    r"\b(?:создал\w*|отправил\w*|передал\w*|ст[её]р\w*|очистил\w*|удалил\w*|"
    r"сбросил\w*|сотру|удалю|отправлю|создам|created|sent|submitted|forwarded|deleted|erased|cleared|wiped)\b.{0,60}"
    r"\b(?:заявк\w*|обращени\w*|письм\w*|данн\w*|памят\w*|истори\w*|"
    r"ticket\w*|e[- ]?mail\w*|data|memory|history|database)\b|"
    r"\b(?:проблем\w*|неисправност\w*|issue|problem)\b.{0,40}"
    r"\b(?:решен\w*|реш[её]н\w*|устран[её]н\w*|исправлен\w*|resolved|fixed|solved)\b",
    re.I | re.S,
)

_UI_FOLLOWUP = re.compile(
    r"\b(?:где\s+(?:это|этот|эта|меню|кнопка|пункт)|не\s+(?:могу\s+)?"
    r"(?:найти|нахожу|вижу)|что\s+нажать)\b",
    re.I,
)

def refusal_reply(message):
    normalized = str(message).casefold().replace("ё", "е")
    return REFUSAL if _SECRET_REQUEST.search(normalized) or _INTERNAL_CHANGE.search(normalized) else None


def requires_support(message):
    return bool(_TECHNICAL.search(str(message)) or _UI_FOLLOWUP.search(str(message)))


def safe_social_reply(reply):
    """Reject generated operational/secret content instead of weakening the KB path."""
    if not isinstance(reply, str) or not reply.strip() or len(reply) > 800:
        return None
    reply = reply.strip()
    if (requires_support(reply) or _TECHNICAL_OUTPUT.search(reply) or _OPERATION_CLAIM.search(reply)
            or _redact(reply) != reply or refusal_reply(reply) or reply.count("?") > 1):
        return None
    return reply
