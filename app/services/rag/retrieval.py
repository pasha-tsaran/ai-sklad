import math

from sqlalchemy import select

from app.models import KnowledgeChunk, KnowledgeDocument, Solution
from app.services.feedback.service import solution_stats
from app.services.rag.embeddings import DemoHashEmbeddings, tokens
from app.services.rag.interface import Candidate


class LocalRetriever:
    def __init__(self, session, min_score: float = 0.30):
        self.session = session
        self.min_score = min_score
        self.embeddings = DemoHashEmbeddings()

    def search(self, query: str, excluded: set[str]) -> list[Candidate]:
        query_tokens = tokens(query)
        if not query_tokens:
            return []
        vector = self.embeddings.embed(query)
        statement = (
            select(Solution, KnowledgeChunk)
            .join(KnowledgeChunk, Solution.chunk_id == KnowledgeChunk.id)
            .join(KnowledgeDocument, KnowledgeChunk.document_id == KnowledgeDocument.id)
            .where(
                Solution.active.is_(True),
                KnowledgeDocument.approved.is_(True),
                KnowledgeChunk.embedding_model == self.embeddings.model_id,
            )
        )
        if excluded:
            statement = statement.where(Solution.id.not_in(excluded))
        if self.session.bind.dialect.name == "postgresql":
            statement = statement.order_by(KnowledgeChunk.embedding.cosine_distance(vector))
        # Tiny curated corpus in this slice: exact lexical gate prevents hash collisions
        # from being mistaken for evidence. Replace retriever for a larger corpus.
        statistics = {row["solution_id"]: row for row in solution_stats(self.session)}
        ranked = []
        for solution, chunk in self.session.execute(statement):
            if not solution.steps or not all(
                isinstance(step, str) and step.strip() for step in solution.steps
            ):
                continue
            source_tokens = tokens(chunk.search_text)
            overlap = len(query_tokens & source_tokens)
            score = overlap / math.sqrt(len(query_tokens) * len(source_tokens)) if source_tokens else 0
            if score < self.min_score:
                continue
            candidate = Candidate(
                solution.id,
                solution.title,
                solution.version,
                chunk.id,
                chunk.content,
                list(solution.steps),
                score,
            )
            success = statistics[solution.id]["smoothed_success_rate"]
            # Relevance first; effectiveness breaks ties between equally relevant cards.
            ranked.append((score, success, -solution.priority, solution.id, candidate))
        ranked.sort(key=lambda entry: entry[:4], reverse=True)
        return [entry[4] for entry in ranked]
