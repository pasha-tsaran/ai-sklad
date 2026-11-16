"""An owned successful dialogue can suggest a past interpretation for confirmation."""

import copy
from datetime import timedelta

import pytest
from sqlalchemy import func, select

from app.models import Feedback, Incident, Message, SolutionAttempt, SuccessfulRetrieval, now
from app.services.rag.success import remember_success
from tests.test_openwebui_instruction_images import instruction
from tests.test_talk_knowledge import DOC, OTHER, QUERY
from tests.test_talk_success_cache import cached as cached


OPENING = "настройка печати"


def add_opening(cached, *, text=OPENING, role="user", after_answer=False):
    with cached.factory() as session, session.begin():
        answer = session.scalar(select(Message).where(Message.incident_id == cached.ids["incident"]))
        answer.sequence = 2
        session.flush()
        session.add(Message(incident_id=cached.ids["incident"], sequence=3 if after_answer else 1,
                            role=role, content=text))


def hint_rows(cached):
    with cached.factory() as session:
        return session.scalars(select(SuccessfulRetrieval).where(
            SuccessfulRetrieval.context_hash != cached.key["context_hash"],
        )).all()


def test_successful_clarified_report_can_be_recalled_after_wrapper_restart(cached):
    add_opening(cached)
    cached.remember(opening_query=OPENING)
    assert len(hint_rows(cached)) == 1
    cached.state["calls"].clear()
    wrapper = cached.wrapper()
    hint = wrapper.confirmation_hint("  НАСТРОЙКА   ПЕЧАТИ! ")
    assert hint == {"problem": QUERY, "source": cached.approved.document["source"]}
    assert not wrapper.cache_hit and wrapper.last_cache_key is None
    assert all(request.method == "GET" for request in cached.state["calls"])
    assert sum(request.url.path.endswith("/content") for request in cached.state["calls"]) == 1
    assert cached.wrapper().confirmation_hint(OPENING) == hint
    assert len(hint_rows(cached)) == 1


@pytest.mark.parametrize("change", ["negative", "unassessed", "not-resolved", "wrong-principal", "no-key"])
def test_interpretation_confirmation_alone_cannot_train_a_hint(cached, change):
    add_opening(cached)
    with cached.factory() as session, session.begin():
        if change == "negative":
            session.get(Feedback, cached.ids["feedback"]).helped = False
        elif change == "unassessed":
            session.get(SolutionAttempt, cached.ids["attempt"]).status = "WAITING_FEEDBACK"
        elif change == "not-resolved":
            session.get(Incident, cached.ids["incident"]).state = "WAITING_FEEDBACK"
    overrides = {"principal": cached.ids["other"]} if change == "wrong-principal" else {}
    if change == "no-key":
        overrides["key"] = None
    assert cached.remember(opening_query=OPENING, **overrides) is None
    assert not hint_rows(cached)
    assert cached.wrapper().confirmation_hint(OPENING) is None


@pytest.mark.parametrize("change", ["missing", "assistant", "after-answer", "different"])
def test_opening_must_be_an_actual_user_report_before_the_successful_answer(cached, change):
    if change != "missing":
        add_opening(cached, role="assistant" if change == "assistant" else "user",
                    after_answer=change == "after-answer", text="Другой вопрос" if change == "different" else OPENING)
    assert cached.remember(opening_query=OPENING) is not None
    assert not hint_rows(cached)
    assert cached.wrapper().confirmation_hint(OPENING) is None


def test_hint_has_exact_query_scope_and_user_boundaries(cached):
    add_opening(cached)
    cached.remember(opening_query=OPENING)
    assert cached.wrapper().confirmation_hint("настройка печати не работает") is None
    assert cached.wrapper().confirmation_hint("настройка сканера") is None
    assert cached.wrapper().confirmation_hint(OPENING, excluded=[DOC]) is None
    assert cached.wrapper(principal=cached.ids["other"]).confirmation_hint(OPENING) is None
    settings = cached.settings.model_copy(update={"openwebui_base_url": "https://other.invalid"})
    assert cached.wrapper(settings=settings).confirmation_hint(OPENING) is None


@pytest.mark.parametrize("change", ["changed-original", "revoked", "expired", "deleted-feedback", "forged-owner"])
def test_stale_or_unproven_hint_is_not_reused(cached, change):
    add_opening(cached)
    cached.remember(opening_query=OPENING)
    if change == "changed-original":
        cached.state["data"] = instruction(["Другая актуальная настройка печати."])
    elif change == "revoked":
        cached.state["files"] = []
    else:
        with cached.factory() as session, session.begin():
            row = session.scalar(select(SuccessfulRetrieval).where(
                SuccessfulRetrieval.context_hash != cached.key["context_hash"],
            ))
            if change == "expired":
                row.expires_at = now() - timedelta(seconds=1)
            elif change == "deleted-feedback":
                session.delete(session.get(Feedback, cached.ids["feedback"]))
            else:
                row.user_id = cached.ids["other"]
    wrapper = cached.wrapper(principal=cached.ids["other"]) if change == "forged-owner" else cached.wrapper()
    assert wrapper.confirmation_hint(OPENING) is None
    assert not hint_rows(cached)


def test_hint_does_not_copy_transcript_or_secrets_into_pointer_storage(cached):
    add_opening(cached)
    with cached.factory() as session, session.begin():
        session.get(Incident, cached.ids["incident"]).description = QUERY + "\nПароль: private-secret"
    cached.remember(opening_query=OPENING)
    hint = cached.wrapper().confirmation_hint(OPENING)
    assert "private-secret" not in hint["problem"] and "[REDACTED]" in hint["problem"]
    row = hint_rows(cached)[0]
    assert row.source == cached.approved.document["source"]
    assert OPENING not in str(vars(row)) and "private-secret" not in str(vars(row))


def test_duplicate_recording_keeps_one_hint_and_original_expiry(cached):
    add_opening(cached)
    cached.remember(opening_query=OPENING)
    first = hint_rows(cached)[0]
    cached.remember(opening_query=OPENING)
    second = hint_rows(cached)[0]
    assert first.id == second.id and first.expires_at == second.expires_at
    with cached.factory() as session:
        assert session.scalar(select(func.count()).select_from(SuccessfulRetrieval)) == 2


@pytest.mark.parametrize("different_source,different_goal", [
    (True, True), (False, True), (False, False),
], ids=["different-source", "same-source-different-goal", "same-source-same-goal"])
def test_past_interpretations_merge_only_when_source_and_goal_agree(cached, monkeypatch,
                                                                 different_source, different_goal):
    add_opening(cached)
    cached.remember(opening_query=OPENING)
    another = copy.deepcopy(cached.approved)
    if different_source:
        another.document["source"]["file_id"] = OTHER
        another.document["source"]["filename"] = "Другой способ настройки печати.docx"
    source = another.document["source"]
    with cached.factory() as session, session.begin():
        previous = session.get(Incident, cached.ids["incident"])
        problem = "Настроить другой способ печати" if different_goal else QUERY
        incident = Incident(conversation_id=previous.conversation_id, state="RESOLVED",
                            subject=OPENING, description=problem, closed_at=now())
        session.add(incident)
        session.flush()
        report = Message(incident_id=incident.id, sequence=1, role="user", content=OPENING)
        answer = Message(incident_id=incident.id, sequence=2, role="assistant", content="Другой исходный документ")
        session.add_all([report, answer])
        session.flush()
        attempt = SolutionAttempt(incident_id=incident.id, solution_id=cached.approved.candidate.solution_id,
                                  ordinal=1, status="SUCCEEDED", message_id=answer.id,
                                  evidence={"talk_source": source}, retrieval_score=1, llm_backend="test")
        session.add(attempt)
        session.flush()
        session.add(Feedback(attempt_id=attempt.id, helped=True))
        session.flush()
        remember_success(session, cached.settings, cached.ids["principal"], incident.id,
                         cached.key, source, opening_query=OPENING)
    original_read = cached.knowledge.read
    monkeypatch.setattr(cached.knowledge, "read", lambda file_id: another if file_id == OTHER else original_read(file_id))
    assert len(hint_rows(cached)) == 2
    recalled = cached.wrapper().confirmation_hint(OPENING)
    if different_source or different_goal:
        assert recalled is None
    else:
        assert recalled == {"problem": QUERY, "source": source}
    remaining = cached.wrapper().confirmation_hint(OPENING, excluded=[DOC])
    if different_source:
        assert remaining["source"] == source
    else:
        assert remaining is None


def test_ordinary_success_pointer_does_not_become_an_opening_hint(cached):
    cached.remember()
    assert cached.wrapper().confirmation_hint(QUERY) is None
