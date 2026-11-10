"""Small, source-backed equipment hints for questions, never repair instructions."""

import re

from app.services.rag.query import scanner_kinds

# A vendor alone is not a model. Accept only a named product with a model
# identifier actually present in an approved scanner original, never an LLM guess.
_MODEL = re.compile(
    r"\b(?:Zebra|Honeywell|CipherLab|Datalogic|Newland|Symbol|Metrologic|Mindeo|АТОЛ)"
    r"[ \t]+[A-Za-zА-Яа-я]{0,12}[- ]?\d{2,6}[A-Za-zА-Яа-я0-9/-]{0,15}\b", re.I,
)
_SCANNER_NAME = re.compile(r"сканер|штрих|barcode|scanner|ds2208", re.I)


def scanner_filename(filename):
    return isinstance(filename, str) and bool(_SCANNER_NAME.search(filename))


def scanner_facts(approved):
    """Every suggested name has an immutable original plus approved membership."""
    content = approved.document["content"]
    categories = sorted(scanner_kinds(content))
    models = list(dict.fromkeys(match.group() for match in _MODEL.finditer(content)))[:6]
    applications = ["1С АДРЕС"] if re.search(r"\b1[СсCc]\s+АДРЕС\b", content, re.I) else []
    return {**approved.document["source"], "categories": categories,
            "models": models, "applications": applications}
