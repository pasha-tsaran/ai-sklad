from app.schemas.contracts import AnalysisResult, AnswerResult, SummaryResult
from app.services.rag.embeddings import tokens


class MockLLM:
    name = "mock-v1"

    def analyze(self, message):
        return AnalysisResult(
            intent="REPORT_PROBLEM",
            search_query=message[:1000],
            problem_type="printer_not_printing" if "printer" in tokens(message) else None,
        )

    def compose(self, candidate):
        return AnswerResult(
            result="ANSWER", solution_id=candidate.solution_id, source_chunk_ids=[candidate.chunk_id]
        )

    def summarize(self, transcript):
        return SummaryResult(summary="Автоматические решения не помогли. См. историю обращения.")
