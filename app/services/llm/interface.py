from typing import Protocol

from app.schemas.contracts import AnalysisResult, AnswerResult, SummaryResult
from app.services.rag.interface import Candidate


class LanguageModel(Protocol):
    name: str

    def analyze(self, message: str) -> AnalysisResult: ...
    def compose(self, candidate: Candidate) -> AnswerResult: ...
    def summarize(self, transcript: str) -> SummaryResult: ...
