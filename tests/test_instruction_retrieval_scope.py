"""Scope-preserving retrieval before the model chooses verified source sections."""

import copy
from types import SimpleNamespace

import pytest

from app.services.rag.catalog import bm25_rank
from app.services.rag.query import normalized_query
from app.services.rag.talk import MAX_SELECTION_CANDIDATES, TalkKnowledge
from tests.test_instruction_planning import MemoryCatalog, context
from tests.test_scanner_original_retrieval import scanner_provider


@pytest.mark.parametrize(("problem", "leading", "relevant", "filename"), [
    (
        "Впервые открыть корпоративную почту в программе Маяк на компьютере",
        "Откройте корпоративную почту в программе Маяк через браузер на компьютере.",
        "Для первого входа в установленный клиент Маяк добавьте корпоративную почту.",
        "Почту Маяк компьютере.md",
    ),
    (
        "Подключить терминал к прибору по Wi-Fi",
        "Подключить терминал к прибору по Wi-Fi можно для одного прибора.",
        "Подключить два прибора по Wi-Fi к одному терминалу.",
        "Терминал прибору Wi-Fi.md",
    ),
    (
        "Уменьшить громкость считывателя на компьютере",
        "Уменьшить громкость считывателя на компьютере в настройках приложения.",
        "Уменьшить громкость считывателя можно тихим режимом самого устройства.",
        "Громкость считывателя компьютере.md",
    ),
])
def test_second_content_hit_survives_competing_vector_and_filename_votes(problem, leading, relevant, filename):
    rows = [
        ("a", "Материал А.md", leading),
        ("b", "Материал Б.md", relevant),
        ("c", filename, "Справка о правилах доступа. " * 100),
        ("v", "Материал В.md", "Справка о замене оборудования."),
        ("d", "Материал Г.md", "Внутренний рабочий справочник."),
    ]
    entries = {identity: SimpleNamespace(card={"title": title}, content=text) for identity, title, text in rows}
    files = {identity: {"filename": title} for identity, title, _ in rows}
    assert bm25_rank(problem, entries)[:2] == ["a", "b"]

    ranked = TalkKnowledge._instruction_ranked_sources(problem, {"g1": problem}, entries, files, ["v"])

    assert ranked[:2] == ["a", "b"]
    assert {"a", "b", "v"} <= set(ranked[:4])


def test_latest_original_correction_has_its_own_protected_candidate(monkeypatch):
    queries = []
    latest = "Уточняю: устройств два. Нужен обычный экспорт таблицы, не печать."
    files = {key: {"filename": f"Материал {key}.md"} for key in "abcdef"}

    def rank(query, entries):
        queries.append(query)
        return ["f", "e", "d", "c", "b", "a"] if "два" in query else list("abcdef")

    monkeypatch.setattr("app.services.rag.catalog.bm25_rank", rank)
    original = {"current_message": latest, "task_state": {"facts": []}}
    unchanged = copy.deepcopy(original)
    ranked = TalkKnowledge._instruction_ranked_sources(
        "Настроить устройство", {"g1": "Настроить устройство"}, dict.fromkeys(files, object()), files, ["e"],
        context=original,
    )

    assert ranked[:4] == ["a", "b", "f", "e"]
    assert normalized_query(latest) in queries
    assert original == unchanged


@pytest.mark.parametrize("another_device_uses_bluetooth", [False, True])
@pytest.mark.parametrize(("fact_key", "attribute"), [("connection", "connection_type"),
                                                     ("condition", "transport"), ("condition", "link_method")])
def test_negated_transport_does_not_gain_lexical_weight_or_hide_other_devices(
        monkeypatch, another_device_uses_bluetooth, fact_key, attribute):
    queries = []
    files = {"a": {"filename": "Материал.md"}}

    def rank(query, entries):
        queries.append(query)
        return ["a"]

    monkeypatch.setattr("app.services.rag.catalog.bm25_rank", rank)
    facts = [
        {"key": fact_key, "attribute": attribute, "value": "Wi-Fi", "entity": "first", "negated": False},
        {"key": fact_key, "attribute": attribute, "value": "Bluetooth", "entity": "first", "negated": True},
    ]
    if another_device_uses_bluetooth:
        facts.append({"key": "condition", "attribute": "connection_type", "value": "Bluetooth",
                      "entity": "second", "negated": False})
    state = {"facts": facts}
    original = copy.deepcopy(state)
    TalkKnowledge._instruction_ranked_sources(
        "Первое устройство по Wi-Fi. Bluetooth не используется.", {"g1": "Подключить устройство"},
        {"a": object()}, files, [], context={"task_state": state, "current_message": "Первое по ви фи, без блютуза."},
    )

    assert queries and all("wifi" in query for query in queries)
    assert all(("bluetooth" in query) is another_device_uses_bluetooth for query in queries)
    assert state == original


def test_all_current_goals_retain_independent_lexical_rankings(monkeypatch):
    queries = []
    files = {key: {"filename": f"Материал {key}.md"} for key in "abcdef"}

    def rank(query, entries):
        queries.append(query)
        return list("cdefab") if "экспорт" in query else list("abcdef")

    monkeypatch.setattr("app.services.rag.catalog.bm25_rank", rank)
    goals = {"g1": "Изменить громкость устройства", "g2": "Экспорт таблицы"}
    TalkKnowledge._instruction_ranked_sources("Две задачи", goals, dict.fromkeys(files, object()), files, [])

    assert normalized_query(goals["g1"]) in queries
    assert normalized_query(goals["g2"]) in queries


def test_string_and_legacy_object_goals_survive_selector_context_compaction():
    state = {"goals": {"g1": "Открыть клиент", "g2": {"text": "Сохранить таблицу", "evidence": "длинное " * 400}},
             "facts": [], "messages": ["история " * 400]}
    unchanged = copy.deepcopy(state)
    result = TalkKnowledge._instruction_context({"task_state": state})

    assert result["task_state"]["goals"] == {"g1": "Открыть клиент", "g2": {"text": "Сохранить таблицу"}}
    assert state == unchanged


def test_expanded_read_window_stays_inside_existing_eight_candidate_budget():
    ids = [f"00000000-0000-4000-8000-{number:012}" for number in range(1, 10)]
    knowledge, _ = scanner_provider([
        (identity, f"Материал {index}.md", f"Подтверждённая рабочая процедура номер {index}.".encode())
        for index, identity in enumerate(ids)
    ])
    knowledge.set_catalog(MemoryCatalog([knowledge.read(identity) for identity in ids]))
    calls = []

    def select(payload):
        calls.append(copy.deepcopy(payload))
        return {"result": "NO_KNOWLEDGE", "missing_goal_ids": ["g1"]}

    knowledge.selector = select
    decision = knowledge.search_with_llm("Иная задача", context("Открыть новый сервис"))

    assert decision.result == "NO_KNOWLEDGE"
    assert [len(payload["candidates"]) for payload in calls] == [4, MAX_SELECTION_CANDIDATES]
    assert calls[0]["read_available"] is True and calls[-1]["read_available"] is False
