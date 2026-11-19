"""Related fallback runs after exact search and retains fresh source authority."""

import hashlib
from uuid import NAMESPACE_URL, uuid5

import pytest

from app.services.llm.instruction_reader import InstructionReading, SourceGoalReading
from app.services.llm.instruction_related import unrelated_instruction
from app.services.llm.openwebui import OpenWebUILLM
from app.services.rag.talk import KnowledgeUnavailable
from tests.test_full_instruction_reading import answer, configure_reader, read_context
from tests.test_scanner_original_retrieval import scanner_provider
from tests.test_talk_knowledge import DOC, OTHER


GOAL = "Подключить одно устройство по Bluetooth"
ALTERNATIVE = "Подключить одно устройство по Wi-Fi"
ORIGINAL = "Готовый порядок подключения одного устройства по Wi-Fi."


def related_answer(payload, **changes):
    return {"original_goal_id": "g1", "source_goal": ALTERNATIVE,
            "evidence_ids": [next(iter(payload["candidate_evidence"]))], "same_purpose": True,
            "status": "RELATED", **changes}


def test_first_exact_solution_never_triggers_related_fallback(monkeypatch):
    exact = "Готовый порядок подключения одного устройства по Bluetooth."
    knowledge, _, calls = configure_reader([(DOC, "Точная инструкция.md", exact.encode())],
                                           lambda wire, number: answer(wire, {"g1": exact}), monkeypatch)
    knowledge.related_reader = lambda data: pytest.fail("Exact solution was already found")
    result = knowledge.search_with_llm(GOAL, read_context({"g1": GOAL}, {"g1": "ACTION"}))
    assert result.result == "MATCH" and len(calls) == 1 and result.alternative_goals == {}


def test_no_exact_solution_offers_verified_source_goal_and_fresh_original(monkeypatch):
    original = ORIGINAL.encode()
    knowledge, state, calls = configure_reader([(DOC, "Готовая альтернатива.md", original)],
                                               lambda wire, number: answer(wire), monkeypatch)
    related_calls = []

    def related(data):
        related_calls.append(data)
        return related_answer(data)

    knowledge.related_reader = related
    result = knowledge.search_with_llm(GOAL, read_context({"g1": GOAL}, {"g1": "ACTION"}))
    assert result.result == "RELATED" and len(calls) == 1 and len(related_calls) == 1
    assert result.alternative_goals == {DOC: {"g1": ALTERNATIVE}}
    assert result.alternatives[0].document["source"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert result.alternatives[0].document["source"]["file_id"] == DOC
    assert related_calls[0]["allow_clarification"] is False
    assert related_calls[0]["source"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert sum(request.url.path.endswith("/content") for request in state["calls"]) >= 2


@pytest.mark.parametrize("changes", [
    {"original_goal_id": "foreign-goal"}, {"evidence_ids": ["foreign-evidence"]},
    {"source_goal": "x" * 251}, {"same_purpose": False}, {"source_goal": ""}, {"evidence_ids": []},
])
def test_callback_cannot_offer_unknown_goal_proof_or_incompatible_task_scope(monkeypatch, changes):
    knowledge, _, _ = configure_reader([(DOC, "Готовая альтернатива.md", ORIGINAL.encode())],
                                       lambda wire, number: answer(wire), monkeypatch)
    knowledge.related_reader = lambda data: related_answer(data, **changes)
    with pytest.raises(KnowledgeUnavailable) as error:
        knowledge.search_with_llm(GOAL, read_context({"g1": GOAL}, {"g1": "ACTION"}))
    assert error.value.code in {"InvalidRelatedInstruction", "ValidationError"}


def test_source_changed_during_related_assessment_cannot_be_offered(monkeypatch):
    knowledge, state, _ = configure_reader([(DOC, "Готовая альтернатива.md", ORIGINAL.encode())],
                                           lambda wire, number: answer(wire), monkeypatch)

    def related(data):
        result = related_answer(data)
        state["originals"][DOC] = "После редактирования описана другая операция.".encode()
        return result

    knowledge.related_reader = related
    with pytest.raises(KnowledgeUnavailable, match="SourceChangedDuringSelection"):
        knowledge.search_with_llm(GOAL, read_context({"g1": GOAL}, {"g1": "ACTION"}))


def test_related_fallback_reads_only_three_already_examined_ranked_originals(monkeypatch):
    identities = [str(uuid5(NAMESPACE_URL, "related-budget/" + str(number))) for number in range(5)]
    rows = [(identity, f"Материал {number}.md", f"Другой готовый материал номер {number}.".encode())
            for number, identity in enumerate(identities)]
    knowledge, _, calls = configure_reader(rows, lambda wire, number: answer(wire), monkeypatch, order=identities)
    inspected = []

    def related(data):
        inspected.append(data["source"]["file_id"])
        return unrelated_instruction()

    knowledge.related_reader = related
    result = knowledge.search_with_llm(GOAL, read_context({"g1": GOAL}, {"g1": "ACTION"}))
    assert result.result == "NO_KNOWLEDGE" and len(calls) == 5 and inspected == identities[:3]


def test_injected_reader_does_not_enable_automatic_related_model_network(monkeypatch):
    knowledge, _ = scanner_provider([(DOC, "Другая тема.md", ORIGINAL.encode())])
    knowledge.settings = knowledge.settings.model_copy(update={"llm_backend": "openwebui"})
    knowledge.reader = lambda data: InstructionReading(goals={"g1": SourceGoalReading(status="MISSING")})
    monkeypatch.setattr(OpenWebUILLM, "related_instruction", lambda *args: pytest.fail("Unexpected model network"))
    result = knowledge.search_with_llm(GOAL, read_context({"g1": GOAL}, {"g1": "ACTION"}))
    assert result.result == "NO_KNOWLEDGE" and result.alternative_goals == {}


def test_later_exact_source_supersedes_earlier_native_related_and_skips_fallback(monkeypatch):
    exact = "Готовый порядок подключения одного устройства по Bluetooth."

    def read(wire, number):
        if wire["approved_original"]["filename"] == "Альтернатива.md":
            return answer(wire, {"g1": ORIGINAL}, status="RELATED")
        return answer(wire, {"g1": exact})

    knowledge, _, _ = configure_reader([(DOC, "Альтернатива.md", ORIGINAL.encode()),
                                        (OTHER, "Точный вариант.md", exact.encode())], read, monkeypatch,
                                       order=[DOC, OTHER])
    knowledge.related_reader = lambda data: pytest.fail("Later exact source already solves this task")
    result = knowledge.search_with_llm(GOAL, read_context({"g1": GOAL}, {"g1": "ACTION"}))
    assert result.result == "MATCH" and result.approved.document["source"]["file_id"] == OTHER


def test_modern_related_reassessment_can_discard_native_same_equipment_hint(monkeypatch):
    knowledge, _, _ = configure_reader([(DOC, "Соседняя функция.md", ORIGINAL.encode())],
                                       lambda wire, number: answer(wire, {"g1": ORIGINAL}, status="RELATED"), monkeypatch)
    knowledge.related_reader = lambda data: unrelated_instruction()
    result = knowledge.search_with_llm(GOAL, read_context({"g1": GOAL}, {"g1": "ACTION"}))
    assert result.result == "NO_KNOWLEDGE" and result.alternative_goals == {}


def test_legacy_injected_reader_native_related_remains_available_without_new_callback(monkeypatch):
    knowledge, _, _ = configure_reader([(DOC, "Старая альтернатива.md", ORIGINAL.encode())],
                                       lambda wire, number: answer(wire, {"g1": ORIGINAL}, status="RELATED"), monkeypatch)
    result = knowledge.search_with_llm(GOAL, read_context({"g1": GOAL}, {"g1": "ACTION"}))
    assert result.result == "RELATED" and result.alternative_goals == {}
    assert result.alternatives[0].document["source"]["file_id"] == DOC
