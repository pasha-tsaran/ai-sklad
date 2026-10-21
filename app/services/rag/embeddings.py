"""Offline lexical baseline, NOT a semantic embedding model. No network access."""

import hashlib
import math
import re
from typing import Protocol


class Embeddings(Protocol):
    def embed(self, text: str) -> list[float]: ...


def tokens(text: str) -> set[str]:
    result = set()
    for word in re.findall(r"[\w]+", text.lower()):
        if word in {"не", "и", "на", "в", "с", "у", "меня", "the", "a", "is", "not"}:
            continue
        if word.startswith(("принтер", "printer")):
            word = "printer"
        elif word.startswith(("печат", "печать", "printing", "print")):
            word = "printing"
        result.add(word)
    return result


class DemoHashEmbeddings:
    model_id = "demo-hash-v1"

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * 128
        for word in tokens(text):
            digest = hashlib.sha256(word.encode()).digest()
            vector[int.from_bytes(digest[:4], "big") % 128] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector
