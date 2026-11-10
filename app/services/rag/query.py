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


def normalized_query(value):
    """A full, ordered fingerprint: preserve negations, model numbers and symptoms."""
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
    text = re.sub(r"\b(?:wi[\s‐‑–—-]*fi|вай[\s-]*фай|вайфай|вифи)\b", " wifi ", text)
    text = re.sub(r"\b(?:блютуз|блютус|блютуф|blue[\s-]*tooth)\b", " bluetooth ", text)
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
    required = _terms(problem) & {"принт", "скане", "тсд"}
    available = _terms(text)
    if "печат" in available:
        available.add("принт")
    if complete and not required.issubset(available):
        return False
    if complete:
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
