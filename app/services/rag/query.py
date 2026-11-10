"""Conservative Talk-only search terms; original instructions remain untouched."""

import re
import unicodedata
from difflib import get_close_matches


_NOISE = frozenset("как что где для при или мой мне меня есть его это вот нужно нужен хочу могу помогите "
                   "подскажите пожалуйста работать работаю версия версии настроить настройка подключение "
                   "настройки инструкция инструкции документ документы фото фотографии показать покажи "
                   "уже моего моем рабочем рабочего месте места с чего начать плохо разбираюсь "
                   "база базе знаний docx xlsx pdf md txt файл файла".split())
_KEYWORDS = ("принтер", "сканер", "терминал", "подключить", "настроить", "штрихкод", "bluetooth")
_CONNECTIONS = frozenset({"wifi", "bluetooth", "usb", "ethernet"})
# Equipment plus a generic action says very little about applicability. Such
# hits need semantic confirmation; named transports, applications and symptoms
# can still support an inexpensive, specific direct match.
_GENERIC_TERMS = frozenset({
    "принт", "печат", "скане", "тсд", "настр", "подкл", "устро", "обору",
    "работ", "делат", "испра", "помог", "помоч", "нужно", "нужен", "пробл", "вопро",
})
_QUALITY_SYMPTOMS = (
    "полос", "пятн", "блед", "туск", "размыт", "нечет", "смаз", "развод", "неравномер",
    "искаж", "зеркал", "клякс", "двоит", "раздво", "зернист", "streak", "faded", "blur",
)
_QUALITY_EVIDENCE = _QUALITY_SYMPTOMS + (
    "качеств", "тонер", "картридж", "чернил", "головк", "quality", "cartridge", "ink",
)

# Closed aliases of named products/transports. Do not transliterate arbitrary
# Russian words, expand generic 'толк', or conflate related vendor products.
_ALIAS_SEPARATOR = r"[\s‐‑‒–—-]*"
_WIFI_ALIAS = rf"(?:wi{_ALIAS_SEPARATOR}fi|(?:вай|ви){_ALIAS_SEPARATOR}(?:фай|фи))"
_BLUETOOTH_ALIAS = rf"(?:blue{_ALIAS_SEPARATOR}tooth|бл[юу]ту[зсф](?:а|у|ом|е)?|(?:блю|блу){_ALIAS_SEPARATOR}ту[зс](?:а|у|ом|е)?)"
_NEXTCLOUD_ALIAS = rf"(?:next{_ALIAS_SEPARATOR}cloud|некст{_ALIAS_SEPARATOR}кл[ао]уд)"
_PRODUCT_ALIASES = (
    (rf"{_NEXTCLOUD_ALIAS}{_ALIAS_SEPARATOR}(?:talk|т[оа]лк)(?:а|у|ом|е)?", "nextcloud talk"),
    (_NEXTCLOUD_ALIAS, "nextcloud"),
    (rf"(?:forti{_ALIAS_SEPARATOR}gate|форт[ие]?{_ALIAS_SEPARATOR}г[еэ]й?т(?:а|у|ом|е|ы|ов)?)", "fortigate"),
    (rf"(?:forti{_ALIAS_SEPARATOR}client|форт[ие]?{_ALIAS_SEPARATOR}клиент(?:а|у|ом|е|ы|ов)?)", "forticlient"),
    (_WIFI_ALIAS, "wifi"),
    (_BLUETOOTH_ALIAS, "bluetooth"),
)


def _normalize_product_aliases(text):
    # Users often omit the separator in 'поблютуз'. Split only a known
    # preposition immediately followed by a complete known transport alias.
    text = re.sub(rf"\b(по|через)(?=(?:{_WIFI_ALIAS}|{_BLUETOOTH_ALIAS})\b)", r"\1 ", text)
    for alias, canonical in _PRODUCT_ALIASES:
        text = re.sub(rf"\b(?:{alias})\b", canonical, text)
    return text


def normalized_query(value):
    """A full, ordered fingerprint: preserve negations, model numbers and symptoms."""
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
    text = _normalize_product_aliases(text)
    return " ".join(re.findall(r"[a-zа-я0-9]+(?:[._-][a-zа-я0-9]+)*", text))


def _word(word):
    # Correct only obvious near-misses of these equipment/search words. Never
    # fuzzy-match a product model, number, symptom or an arbitrary employee name.
    if word not in _KEYWORDS and word.isalpha() and len(word) >= 5:
        matches = get_close_matches(word, _KEYWORDS, n=1, cutoff=0.87)
        if matches and word[:2] == matches[0][:2]:
            word = matches[0]
    if word.startswith(("принтер", "принтр")):
        return "принтер"
    if word.startswith(("сканер", "считыват")):
        return "сканер"
    if word.startswith("терминал") or word in {"тсд", "tsd"}:
        return "тсд"
    if word.startswith(("подключ", "падключ")):
        return "подключение"
    if word.startswith("настро"):
        return "настройка"
    if word.startswith(("штрих", "баркод")) or word == "шк":
        return "штрихкод"
    if word.startswith(("изображ", "картин", "рисунк")):
        return "изображение"
    if word.startswith(("акробат", "acrobat")):
        return "acrobat"
    if word.startswith(("печат", "печет", "напечата", "распечата")):
        return "печать"
    return word


def retrieval_query(value):
    """Compact inflected/abbreviated input, without introducing a repair method."""
    words = [_word(word) for word in normalized_query(value).split()]
    words = [word for word in words if word not in _NOISE or word in {"настройка", "подключение"}]
    return " ".join(dict.fromkeys(words))[:1000]


def _terms(value):
    words = [_word(word) for word in normalized_query(value).split()]
    return {word if word in _CONNECTIONS else word[:5] for word in words
            if word not in _NOISE and not word.isdigit() and (len(word) >= 3 or word == "1с")}


def _connections(value):
    words = normalized_query(value).split()
    return {word for index, word in enumerate(words) if word in _CONNECTIONS
            and "не" not in words[max(0, index - 2):index] and "без" not in words[max(0, index - 2):index]}


def _identifiers(value):
    text = normalized_query(value)
    # Preserve explicit product identifiers and software versions. Counts and
    # isolated numbers aren't models; an employee's IPv4 address isn't a version.
    models = {re.sub(r"[-_]", "", word) for word in text.split()
              if re.fullmatch(r"[a-zа-я][a-zа-я0-9_-]{2,}", word)
              and re.search(r"\d", word)}
    versions = set()
    for word in text.split():
        if re.fullmatch(r"\d+(?:\.\d+){1,3}", word):
            parts = word.split(".")
            if len(parts) == 4 and all(int(part) <= 255 for part in parts):
                continue
            versions.add(word)
    return models, versions


def scanner_kinds(value):
    """Distinguish a document scanner from a barcode reader using explicit evidence."""
    text = normalized_query(value)
    kinds = set()
    if re.search(r"\b(?:штрих\w*|баркод\w*|barcode|шк)\b", text):
        kinds.add("barcode")
    if re.search(r"\b(?:сканер\w*|сканирован\w*)\s+(?:(?:бумажн\w*|обычн\w*)\s+)?"
                 r"(?:документ\w*|бумаг\w*|лист\w*)\b", text):
        kinds.add("document")
    return kinds


def _task_covered(problem, text):
    """Require the named task/feature, not just generic equipment overlap.

    These are evidence anchors, not repair suggestions. A short retrieval hit
    without them can still go through semantic selection and a complete-source
    check; a real original lacking them cannot authorize the requested task.
    """
    problem_words = [_word(word) for word in normalized_query(problem).split()]
    source_words = [_word(word) for word in normalized_query(text).split()]

    def mentions(words, prefixes):
        return any(word.startswith(prefixes) for word in words)

    required = []
    if mentions(problem_words, ("pdf", "acrobat")):
        required.append(("pdf", "acrobat"))
    if mentions(problem_words, ("изображение",)):
        required.append(("изображение",))
    if mentions(problem_words, ("очеред",)):
        required.append(("очеред",))
    if mentions(problem_words, ("громк", "звук", "пищ", "писк", "зуммер")):
        required.append(("громк", "звук", "пищ", "писк", "зуммер", "сигнал", "beep", "sound", "volume"))
    # Printing with visible defects is not the same task as a queue which does
    # not print at all. Require evidence about print quality/materials rather
    # than permitting a generic 'printer/printing' match to authorize cleanup.
    if (mentions(problem_words, ("принтер", "печать"))
            and mentions(problem_words, _QUALITY_SYMPTOMS)):
        required.append(_QUALITY_EVIDENCE)
    # A request to connect is different from a symptom report which merely
    # states that the device is already connected.
    if re.search(r"\b(?:подключ(?:ить|ите|и|ение\w*)|соединить|соединени\w*|сопрячь|сопряжени\w*)\b",
                 normalized_query(problem)):
        required.append(("подключ", "настрой", "соедин", "сопряж", "подсоедин"))
    return all(mentions(source_words, prefixes) for prefixes in required)


def source_compatible(problem, text, *, complete=False):
    """Don't replace named hardware/transport with a different similar procedure.

    Excerpts may omit a prerequisite, so absence is checked on the complete
    original. An explicitly contradictory excerpt is rejected before selection.
    """
    wanted, supplied = _connections(problem), _connections(text)
    if wanted and (supplied and not wanted.issubset(supplied) or complete and not wanted.issubset(supplied)):
        return False
    wanted_scanners, supplied_scanners = scanner_kinds(problem), scanner_kinds(text)
    if wanted_scanners and (supplied_scanners and not wanted_scanners.issubset(supplied_scanners)
                            or complete and not wanted_scanners.issubset(supplied_scanners)):
        return False
    required = _terms(problem) & {"принт", "скане", "тсд"}
    available = _terms(text)
    if "печат" in available:
        available.add("принт")
    if complete and not required.issubset(available):
        return False
    if complete:
        requested_words = set(normalized_query(problem).split())
        source_words = set(normalized_query(text).split())
        products = requested_words & {"fortigate", "forticlient", "nextcloud"}
        if not products.issubset(source_words):
            return False
        if {"nextcloud", "talk"}.issubset(requested_words) and "talk" not in source_words:
            return False
        if not _task_covered(problem, text):
            return False
        wanted_models, wanted_versions = _identifiers(problem)
        models, versions = _identifiers(text)
        if not wanted_models.issubset(models) or not wanted_versions.issubset(versions):
            return False
    return True


def choose_source(problem, chunks):
    """Rank scoped hits by informative overlap, including evidence outside titles."""
    terms = _terms(problem)
    if not terms:
        return None
    grouped = {}
    for chunk in chunks[:32]:
        entry = grouped.setdefault(chunk["file_id"], {"title": "", "content": []})
        entry["title"] = chunk["source"]
        entry["content"].append(chunk["content"])
    ranked = []
    for file_id, entry in grouped.items():
        text = entry["title"] + "\n" + "\n".join(entry["content"])
        if not source_compatible(problem, text) or not _task_covered(problem, text):
            continue
        title_hits = terms & _terms(entry["title"])
        all_hits = terms & _terms(text)
        # A single generic word such as 'принтер' isn't enough to diagnose a
        # problem. Semantic fallback can disambiguate it using actual excerpts.
        if len(all_hits) < 2 or not (all_hits - _GENERIC_TERMS):
            continue
        ranked.append((len(title_hits) * 2 + len(all_hits), file_id))
    ranked.sort(reverse=True)
    if not ranked or len(ranked) > 1 and ranked[0][0] - ranked[1][0] < 2:
        return None
    return ranked[0][1]
