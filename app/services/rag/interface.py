from dataclasses import dataclass, field
from typing import Protocol


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


class Retriever(Protocol):
    def search(self, query: str, excluded: set[str]) -> list[Candidate]: ...
