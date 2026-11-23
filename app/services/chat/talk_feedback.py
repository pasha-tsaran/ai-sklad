"""Aggregate explicit Talk feedback for the same question and document version."""

import hashlib
import re
from datetime import timedelta

from sqlalchemy import select

from app.models import Conversation, Delivery, Feedback, Incident, SolutionAttempt, now
from app.services.chat.memory import _redact
from app.services.rag.query import normalized_query


_OUTCOME_PREFIX = re.compile(r"^(?:(?:да|нет|спасибо|благодарю|отлично|ура) )+")
_NEXT_QUESTION = re.compile(
    r"(?:[.!\n]+\s*|[,;]\s*(?=(?:(?:а\s+)?теперь|как|где|почему|что|когда)\b)|\s+(?=(?:а\s+)?теперь\b))"
    r"(?:(?:а\s+)?теперь(?:\s+смотри(?:те)?)?|другой вопрос|новый вопрос|еще вопрос)?[,:]?\s*",
    re.IGNORECASE,
)
_HELPED = {
    "помогло", "помог", "помогла", "это помогло", "очень помогло", "совет помог", "инструкция помогла",
    "помогла инструкция", "решилось",
    "сработало", "получилось", "все получилось", "все работает", "работает", "заработало",
    "все заработало", "проблема решена", "вопрос решен", "удалось", "удалось решить проблему",
}
_UNHELPFUL = {
    "не помогло", "не помог", "не помогла", "это не помогло", "не сработало", "не получилось",
    "не удалось", "не работает", "все еще не работает", "проблема осталась", "ничего не изменилось",
    "ответ не подходит", "ответ не подошел", "ответ не понравился", "не понравилось", "инструкция не подходит",
}
_LIKED = {"ответ подошел", "ответ понравился", "понравилось", "ответ полезен", "инструкция подошла"}


def _rating(message, awaiting):
    if "?" in message:
        return None
    text = normalized_query(message)
    if awaiting and text in {"да", "ага", "угу", "верно", "точно", "конечно", "да спасибо"}:
        return "AGENT_RESOLVED"
    if awaiting and text in {"нет", "неа", "нет спасибо", "еще нет", "пока нет"}:
        return "AGENT_NEGATIVE"
    text = re.sub(r" (?:спасибо|спасибо большое|благодарю)$", "", _OUTCOME_PREFIX.sub("", text))
    text = re.sub(r"^(?:теперь|сейчас|уже) ", "", text)
    if not awaiting and text in {"получилось", "удалось", "работает"}:
        return None
    if text in _HELPED:
        return "AGENT_RESOLVED"
    if text in _UNHELPFUL or message.strip() == "👎":
        return "AGENT_NEGATIVE"
    if text in _LIKED or message.strip() == "👍":
        return "AGENT_RATED"
    return None


def split_agent_feedback(message, *, awaiting=False):
    """Return a clear rating and any separate request that follows it."""
    rating = _rating(message, awaiting)
    if rating:
        return rating, None
    for boundary in _NEXT_QUESTION.finditer(message):
        rating = _rating(message[:boundary.start()], awaiting)
        question = message[boundary.end():].strip()
        normalized = normalized_query(question)
        if rating == "AGENT_NEGATIVE" and normalized in {
            "что делать", "что дальше", "что теперь делать", "как быть", "что еще попробовать",
        }:
            return rating, None
        if rating and normalized and not normalized.startswith(("но ", "однако ", "частично ")):
            return rating, question
    return None, None


def question_key(message):
    normalized = normalized_query(_redact(message))
    if not normalized or len(normalized) > 2000:
        return None
    return hashlib.sha256(normalized.encode()).hexdigest()


def source_key(source):
    fields = [source.get(field) for field in ("file_id", "knowledge_id", "sha256")]
    return ":".join(fields) if all(isinstance(value, str) and value for value in fields) else None


def learned_feedback(session, account_id, message):
    key = question_key(message)
    if key is None:
        return {}
    rows = session.execute(
        select(Conversation.user_id, SolutionAttempt.evidence, Feedback.helped)
        .select_from(Feedback)
        .join(SolutionAttempt, SolutionAttempt.id == Feedback.attempt_id)
        .join(Incident, Incident.id == SolutionAttempt.incident_id)
        .join(Conversation, Conversation.id == Incident.conversation_id)
        .join(Delivery, Delivery.message_id == SolutionAttempt.message_id)
        .where(Conversation.channel == "nextcloud_talk", Delivery.account_id == account_id,
               Delivery.status == "SENT", Feedback.created_at >= now() - timedelta(days=90),
               SolutionAttempt.evidence["feedback_question"].as_string() == key)
        .order_by(Feedback.created_at.desc(), Feedback.id.desc()).limit(250)
    )
    results, seen = {}, set()
    for user_id, evidence, helped in rows:
        sources = evidence.get("agent_sources", [])
        if len(sources) != 1:
            continue
        source = sources[0]
        identity = source_key(source)
        if identity is None or (user_id, identity) in seen:
            continue
        seen.add((user_id, identity))
        score = results.setdefault(identity, {"helpful": 0, "unhelpful": 0})
        score["helpful" if helped else "unhelpful"] += 1
    return results


def reorder_candidates(result, feedback):
    candidates = result["candidates"]
    if not feedback or not candidates:
        return result
    ranked = []
    for position, candidate in enumerate(candidates):
        item = dict(candidate)
        votes = feedback.get(source_key(item))
        adjustment = 0
        if votes:
            item["feedback"] = votes
            adjustment = max(-2, min(2, votes["helpful"] - votes["unhelpful"]))
        ranked.append((position - adjustment, -adjustment, position, item))
    return {**result, "candidates": [item for _, _, _, item in sorted(ranked)]}
