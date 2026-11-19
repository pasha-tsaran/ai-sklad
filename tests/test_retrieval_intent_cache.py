"""Learned delivery scope distinguishes procedural work from factual lookup."""

import copy

from app.core.config import Settings
from app.services.rag.success import _digest, _hint_key, cache_key


def settings():
    return Settings(_env_file=None, openwebui_knowledge_ids=["11111111-1111-4111-8111-111111111111"],
                    openwebui_api_key="test-only-key", seed_demo_knowledge=False)


def task(kind=None):
    goal = {"text": "Размер этикетки", "evidence": "Размер этикетки"}
    if kind is not None:
        goal["kind"] = kind
    return {"task_state": {"facts": [], "goals": {"g1": goal}}}


def test_lookup_does_not_reuse_a_complete_procedure_with_the_same_goal_words():
    action = cache_key(settings(), "Размер этикетки", task("ACTION"))
    lookup = cache_key(settings(), "Размер этикетки", task("LOOKUP"))
    assert action["scope_hash"] == lookup["scope_hash"]
    assert action["query_hash"] != lookup["query_hash"]
    assert action["context_hash"] != lookup["context_hash"]


def test_discovery_aliases_never_change_a_proven_intent_cache_key():
    context = task("LOOKUP")
    original = copy.deepcopy(context)
    expected = cache_key(settings(), "Размер этикетки", context)
    context["task_state"]["goals"]["g1"]["search_queries"] = ["factory default label width height"]
    assert cache_key(settings(), "Размер этикетки", context) == expected
    assert original["task_state"]["goals"]["g1"] == {"text": "Размер этикетки", "evidence": "Размер этикетки",
                                                     "kind": "LOOKUP"}


def test_legacy_goal_shape_keeps_its_existing_cache_identity_and_cannot_alias_new_typed_scope():
    legacy = cache_key(settings(), "Размер этикетки", task())
    assert legacy["query_hash"] == _digest({"purpose": "typed_task_v2", "goals": ["размер этикетки"]})
    assert legacy != cache_key(settings(), "Размер этикетки", task("ACTION"))
    assert cache_key(settings(), "Размер этикетки", task("invalid")) is None


def test_old_untyped_clarification_hints_are_rechecked_after_the_intent_protocol_upgrade():
    key = _hint_key(settings(), "Настройка устройства", "old-incident")
    old = _digest({"purpose": "clarification_hint_v1", "incident_id": "old-incident"})
    assert key["context_hash"] != old
