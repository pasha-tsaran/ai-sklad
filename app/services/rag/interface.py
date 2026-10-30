from dataclasses import dataclass, field
from typing import Protocol

from app.services.rag.documents import Illustration


@dataclass(frozen=True)
class Candidate:
    solution_id: str
    title: str
    version: int
    chunk_id: str
    content: str
    steps: list[str]
    score: float
    source: dict = field(default_factory=dict)
    illustrations: list[Illustration] = field(default_factory=list)


class Retriever(Protocol):
    def search(self, query: str, excluded: set[str]) -> list[Candidate]: ...
