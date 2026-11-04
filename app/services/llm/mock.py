from app.schemas.contracts import AnalysisResult, AnswerResult, DialogueResult, FollowupResult, SummaryResult
from app.services.rag.embeddings import tokens


class MockLLM:
    name = "mock-v1"

    def dialogue(self, message, context):
        normalized = message.strip().casefold().rstrip(".!?")
        if normalized in {"привет", "здравствуйте", "добрый день", "спасибо", "как дела"}:
            return DialogueResult(route="SOCIAL", reply="Здравствуйте! Рад помочь. Как проходит ваш день?")
        return DialogueResult(route="SUPPORT")

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

    def analyze_context(self, message, context):
        return self.analyze(message)

    def followup(self, question, context, approved_source):
        # Deterministic demo must not pretend to understand an arbitrary follow-up.
        return FollowupResult(result="NO_KNOWLEDGE")

    def summarize(self, transcript):
        return SummaryResult(summary="Автоматические решения не помогли. См. историю обращения.")
