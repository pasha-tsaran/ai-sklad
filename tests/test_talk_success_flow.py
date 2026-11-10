"""Positive feedback teaches a source pointer through the real durable dialogue."""

import copy

import pytest
from sqlalchemy import select

from app.models import SuccessfulRetrieval
from tests.test_talk_dialogue import QUESTION, Knowledge, NoGeneration, assistants, talk as talk
from tests.test_talk_knowledge import QUERY, provider


def test_successful_dialogue_reuses_source_on_next_incident_without_repeating_search_or_model(talk):
    knowledge, state = provider()
    model = NoGeneration()
    talk.incoming(QUERY)
    assert talk.process(knowledge, model)
    first = talk.snapshot()
    first_blocks = [(text, picture.data if picture else None) for text, picture in assistants(first)]
    assert sum(request.method == "POST" for request in state["calls"]) == 1
    assert "retrieval_key" in first["flow"]
    talk.incoming("Да, всё заработало, спасибо большое!", 2)
    assert talk.process(knowledge, model)
    assert talk.snapshot()["incident"].state == "RESOLVED"
    with talk.env.factory() as session:
        saved = session.scalar(select(SuccessfulRetrieval))
        assert saved is not None and saved.incident_id == first["incident"].id
        assert saved.source == first["flow"]["source"]
    # A new process_one invocation builds a fresh wrapper and a new incident.
    talk.incoming(QUERY, 3)
    assert talk.process(knowledge, model)
    after = talk.snapshot()
    assert after["incident"].id != first["incident"].id
    assert sum(request.method == "POST" for request in state["calls"]) == 1
    assert sum(request.url.path.endswith("/content") for request in state["calls"]) >= 2
    assert [(text, picture.data if picture else None) for text, picture in assistants(after)] == first_blocks
    assert not model.calls and len(after["attempts"]) == 1


@pytest.mark.parametrize("feedback", ["Готово", "Да", "получилось", "Не помогло"])
def test_step_confirmation_or_failure_never_trains_a_successful_answer(talk, feedback):
    knowledge, model = Knowledge(), NoGeneration()
    talk.incoming(QUESTION)
    talk.process(knowledge, model)
    talk.incoming(feedback, 2)
    talk.process(knowledge, model)
    with talk.env.factory() as session:
        assert session.scalar(select(SuccessfulRetrieval)) is None


def test_changed_source_on_repeat_invalidates_success_pointer_and_uses_normal_search(talk):
    knowledge, model = Knowledge(), NoGeneration()
    for sequence, text in enumerate([QUESTION, "Помогло"], 1):
        talk.incoming(text, sequence)
        talk.process(knowledge, model)
    changed = copy.deepcopy(knowledge.documents[0])
    changed.document["source"]["sha256"] = "b" * 64
    knowledge.documents = [changed]
    talk.incoming(QUESTION, 3)
    talk.process(knowledge, model)
    assert len(knowledge.searches) == 2
    assert talk.snapshot()["flow"]["source"]["sha256"] == "b" * 64
    with talk.env.factory() as session:
        assert session.scalar(select(SuccessfulRetrieval)) is None
