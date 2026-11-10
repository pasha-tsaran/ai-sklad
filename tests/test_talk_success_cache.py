"""Only explicit private success can shorten a later, still-current source lookup."""

import copy
from datetime import timedelta

import pytest
from sqlalchemy import create_engine, event, func, select

from app.db import session_factory
from app.models import Base, Conversation, Feedback, Incident, Message, SolutionAttempt, SuccessfulRetrieval, User, now
from app.services.rag.openwebui import save_snapshot
from app.services.rag.success import SuccessfulTalkKnowledge, cache_key, remember_success
from tests.test_openwebui_instruction_images import instruction
from tests.test_talk_knowledge import DOC, KB, QUERY, provider


@pytest.fixture
def cached(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'success.db'}")

    @event.listens_for(engine, "connect")
    def foreign_keys(connection, _):
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    factory = session_factory(engine)
    knowledge, state = provider()
    approved = knowledge.read(DOC)
    settings = knowledge.settings
    context = {"incident": {"equipment_model": None, "problem_type": None},
               "channel_profile": {"display_name": "PRIVATE", "computer_name": "PRIVATE-PC"}}
    with factory() as session, session.begin():
        user = User(external_id="private-principal", display_name="Employee")
        other = User(external_id="another-principal", display_name="Another employee")
        session.add_all([user, other])
        session.flush()
        conversation = Conversation(user_id=user.id)
        session.add(conversation)
        session.flush()
        incident = Incident(conversation_id=conversation.id, state="RESOLVED", subject=QUERY,
                            description=QUERY, closed_at=now())
        session.add(incident)
        session.flush()
        message = Message(incident_id=incident.id, sequence=1, role="assistant", content="Original instruction")
        session.add(message)
        save_snapshot(session, approved.candidate)
        session.flush()
        attempt = SolutionAttempt(incident_id=incident.id, solution_id=approved.candidate.solution_id,
                                  ordinal=1, status="SUCCEEDED", message_id=message.id,
                                  evidence={"talk_source": approved.document["source"]}, retrieval_score=1,
                                  llm_backend="test")
        session.add(attempt)
        session.flush()
        feedback = Feedback(attempt_id=attempt.id, helped=True)
        session.add(feedback)
        session.flush()
        ids = {"principal": user.id, "other": other.id, "incident": incident.id,
               "attempt": attempt.id, "feedback": feedback.id}
    key = cache_key(settings, QUERY, context)

    def remember(**overrides):
        args = {"principal": ids["principal"], "incident_id": ids["incident"], "key": key,
                "source": approved.document["source"], **overrides}
        with factory() as session, session.begin():
            return remember_success(session, settings, **args)

    def wrapper(**overrides):
        return SuccessfulTalkKnowledge(knowledge, factory, overrides.get("settings", settings),
                                       overrides.get("principal", ids["principal"]), overrides.get("context", context))

    yield type("CacheFixture", (), {"factory": factory, "knowledge": knowledge, "state": state,
                                     "approved": approved, "settings": settings, "context": context,
                                     "ids": ids, "key": key, "remember": staticmethod(remember),
                                     "wrapper": staticmethod(wrapper)})
    engine.dispose()


def count(cached):
    with cached.factory() as session:
        return session.scalar(select(func.count()).select_from(SuccessfulRetrieval))


def test_explicit_success_survives_wrapper_restart_and_rechecks_original_without_vector_or_model(cached):
    assert cached.remember() is not None
    cached.state["calls"].clear()
    first = cached.wrapper()
    approved = first.search(QUERY)
    assert first.cache_hit is True and first.last_cache_key == cached.key
    assert approved.document == cached.approved.document
    assert approved.illustrations[1].data == cached.approved.illustrations[1].data
    assert not any(request.method == "POST" for request in cached.state["calls"])
    assert sum(request.url.path.endswith("/content") for request in cached.state["calls"]) == 1
    assert sum("/knowledge/" in request.url.path for request in cached.state["calls"]) == 2
    assert cached.wrapper().search(QUERY).document == approved.document
    assert count(cached) == 1


def test_search_alone_never_trains_or_caches_an_unconfirmed_answer(cached):
    assert cached.wrapper().search(QUERY) is not None
    assert count(cached) == 0


@pytest.mark.parametrize("change", ["negative", "unassessed", "not-resolved", "wrong-user", "wrong-source"])
def test_cache_requires_owned_positive_feedback_and_exact_original_source(cached, change):
    with cached.factory() as session, session.begin():
        if change == "negative":
            session.get(Feedback, cached.ids["feedback"]).helped = False
        elif change == "unassessed":
            session.get(SolutionAttempt, cached.ids["attempt"]).status = "UNASSESSED"
        elif change == "not-resolved":
            session.get(Incident, cached.ids["incident"]).state = "WAITING_FEEDBACK"
    args = {}
    if change == "wrong-user":
        args["principal"] = cached.ids["other"]
    if change == "wrong-source":
        args["source"] = {**cached.approved.document["source"], "sha256": "0" * 64}
    assert cached.remember(**args) is None
    assert count(cached) == 0


def test_cache_key_is_private_hashes_and_ignores_profile_intake_and_chat_history(cached):
    original = cached.key
    context = copy.deepcopy(cached.context)
    context["channel_profile"] = {"display_name": "Different user", "computer_name": "SECRET-PC"}
    context["short_term"] = {"messages": [{"content": "PRIVATE PASSWORD"}]}
    assert cache_key(cached.settings, QUERY, context) == original
    assert set(original) == {"scope_hash", "query_hash", "context_hash"}
    assert all(len(value) == 64 for value in original.values())
    cached.remember()
    with cached.factory() as session:
        row = session.scalar(select(SuccessfulRetrieval))
        assert row.source == cached.approved.document["source"]
        assert not any(term in str(vars(row)) for term in (QUERY, "PRIVATE-PC", "PRIVATE PASSWORD"))


@pytest.mark.parametrize("query", [
    "Acrobat PDF с картинками печатается", "Acrobat PDF без картинок не печатается",
    "Zebra ZD420 не печатает", "Zebra ZD410 не печатает",
    "Подключить принтер к ТСД по Wi-Fi", "Подключить принтер к ТСД через Bluetooth",
])
def test_distinct_negation_symptoms_models_and_connections_have_distinct_keys(cached, query):
    assert cache_key(cached.settings, query, cached.context) != cached.key


def test_every_explicit_model_and_connection_in_problem_context_partitions_cache(cached):
    for model in ("Zebra ZD420 Wi-Fi", "Zebra ZD410 Bluetooth"):
        context = {"incident": {"equipment_model": model, "problem_type": "Не печатает"}}
        assert cache_key(cached.settings, QUERY, context)["context_hash"] != cached.key["context_hash"]


def test_cache_never_crosses_principals_knowledge_scope_or_webui_server(cached):
    cached.remember()
    for overrides in ({"principal": cached.ids["other"]},
                      {"settings": cached.settings.model_copy(update={"openwebui_base_url": "https://another.invalid"})},
                      {"settings": cached.settings.model_copy(update={"openwebui_knowledge_ids": [
                          KB, "00000000-0000-4000-8000-000000000088"]})}):
        wrapper = cached.wrapper(**overrides)
        wrapper.search(QUERY)
        assert not wrapper.cache_hit


def test_generated_retrieval_query_never_replaces_actual_problem_cache_key(cached):
    cached.remember()
    wrapper = cached.wrapper()
    assert wrapper.search("Модель придумала другой запрос", fallback_query=QUERY) is not None
    assert wrapper.cache_hit and wrapper.last_cache_key == cached.key


def test_excluded_source_is_not_retried_even_when_previously_successful(cached):
    cached.remember()
    cached.state["calls"].clear()
    wrapper = cached.wrapper()
    assert wrapper.search(QUERY, excluded=[DOC]) is None
    assert not wrapper.cache_hit
    assert not any("/files/" in request.url.path for request in cached.state["calls"])


@pytest.mark.parametrize("change", ["changed-original", "revoked-membership", "expired", "deleted-feedback"])
def test_changed_revoked_or_unproven_cache_is_invalidated_before_fresh_scoped_search(cached, change):
    cached.remember()
    if change == "changed-original":
        cached.state["data"] = instruction(["Другая актуальная инструкция Acrobat PDF."])
    elif change == "revoked-membership":
        cached.state["files"] = []
    else:
        with cached.factory() as session, session.begin():
            if change == "expired":
                session.scalar(select(SuccessfulRetrieval)).expires_at = now() - timedelta(seconds=1)
            else:
                session.delete(session.get(Feedback, cached.ids["feedback"]))
    wrapper = cached.wrapper()
    found = wrapper.search(QUERY)
    assert not wrapper.cache_hit and count(cached) == 0
    if change == "revoked-membership":
        assert found is None
    else:
        assert found is not None
    if change == "changed-original":
        assert found.document["source"] != cached.approved.document["source"]


def test_duplicate_confirmation_cannot_extend_retention_from_original_resolution(cached):
    first = cached.remember()
    second = cached.remember()
    assert first.id == second.id and first.expires_at == second.expires_at
    assert count(cached) == 1
