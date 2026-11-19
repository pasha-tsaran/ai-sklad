"""Verified modern scope stays bound to fresh original paragraphs and proof roles."""

import copy
from uuid import NAMESPACE_URL, uuid5

import pytest
from pydantic import ValidationError

from app.services.chat.confirmation_scope import confirmation_scope_entries, confirmation_scope_identity
from app.services.chat.instruction_plan import restore_plan, serialize_plan
from app.services.llm.instruction_discovery import InstructionNomination
from app.services.llm.instruction_reader import InstructionReading, SourceGoalReading
from app.services.rag.talk import InstructionPartSelection, KnowledgeUnavailable, TalkKnowledge
from tests.test_conditional_instruction_scopes import ALTERNATIVE, CAPTIONS, PREREQUISITE, STEPS, source
from tests.test_exclusive_instruction_methods import CHOICE, COMMON, FIRST, SECOND, knowledge_source
from tests.test_full_instruction_reading import answer, configure_reader, read_context
from tests.test_instruction_planning import selected_part
from tests.test_instruction_reader import wire_evidence
from tests.test_instruction_verification import model_verifier
from tests.test_openwebui_instruction_images import instruction
from tests.test_scanner_original_retrieval import scanner_provider
from tests.test_talk_knowledge import DOC, OTHER


def scope_reader(knowledge, operation, complete, *, materials=(), kind="ACTION", mutate=None):
    knowledge.settings = knowledge.settings.model_copy(update={"llm_backend": "openwebui"})
    captured = []

    def read(payload):
        captured.append(copy.deepcopy(payload))
        registry = payload["candidate_evidence"]

        def refs(needles):
            return tuple(dict.fromkeys(reference for needle in needles for reference, proof in registry.items()
                                       if needle in proof["text"]))

        operation_refs, answer_refs = refs(operation), refs(complete)
        assert operation_refs and answer_refs
        goal_id, goal_text = next(iter(payload["requested_goals"].items()))
        mode = "PROCEDURE" if kind == "ACTION" else "EXCERPT"
        reading = SourceGoalReading(status="SUPPORTED", evidence_ids=list(operation_refs), delivery_mode=mode)
        reading._verified_scope = {
            "source": copy.deepcopy(payload["source"]), "goal_id": goal_id, "goal_text": goal_text,
            "goal_kind": kind, "delivery_mode": mode, "operation_refs": operation_refs,
            "material_refs": refs(materials), "answer_refs": answer_refs, "confirmation_refs": operation_refs,
        }
        if mutate:
            mutate(reading._verified_scope)
        return InstructionReading(goals={goal_id: reading})

    knowledge.reader = read
    return captured


def test_modern_scope_keeps_complete_original_paragraphs_and_required_guards_without_unasked_suffix():
    knowledge, _ = scanner_provider([(DOC, "Полный порядок.md", (
        "# Проверка\nВнимание: сохраните рабочие данные.\n"
        "1. Откройте тест.\n2. Проверьте результат. Значение должно появиться в окне.\n"
        "3. Удалите запись только для другой задачи.").encode())])
    scope_reader(knowledge, ["Откройте тест"], ["Откройте тест", "Проверьте результат"])
    result = knowledge.search_with_llm("Проверить устройство", read_context({"g1": "Проверить устройство"}, {"g1": "ACTION"}))
    text = "\n".join(section["text"] for section in result.plan[0].sections)
    assert "сохраните рабочие данные" in text and "Значение должно появиться в окне." in text
    assert "Удалите запись" not in text
    assert result.goal_kind_overrides == {"g1": "ACTION"}
    assert result.plan[0].confirmation_evidence[0]["text"] == "1. Откройте тест."
    assert restore_plan(knowledge, serialize_plan(result.plan))[0].sections == result.plan[0].sections


def test_operation_text_anchor_and_shared_original_photo_do_not_activate_other_method():
    caption = "Снимите «Размер из файла», затем включите «Печатать как рисунок»."
    knowledge = source([caption])
    scope_reader(knowledge, [STEPS[1]], [*STEPS, caption], materials=[caption])
    result = knowledge.search_with_llm("Исправить только размер", read_context({"g1": "Исправить размер"}, {"g1": "ACTION"}))
    text = "\n".join(section["text"] for section in result.plan[0].sections)
    assert all(step in text for step in STEPS) and PREREQUISITE in text
    assert ALTERNATIVE not in text
    assert [number for section in result.plan[0].sections for number in section["image_numbers"]] == [1]
    assert result.plan[0].confirmation_evidence[0]["support_kind"] == "TEXT"


def test_modern_remainder_does_not_expand_to_completed_prefix_or_old_photo():
    knowledge = source()
    scope_reader(knowledge, [ALTERNATIVE[:60]], [ALTERNATIVE[:60], CAPTIONS[1]], materials=[CAPTIONS[1]])
    result = knowledge.search_with_llm("Основная проверка уже выполнена", read_context({"g1": "Дополнительная проверка"}, {"g1": "ACTION"}))
    text = "\n".join(section["text"] for section in result.plan[0].sections)
    assert ALTERNATIVE in text and PREREQUISITE in text
    assert not any(step in text for step in STEPS)
    assert CAPTIONS[0] not in text
    assert [number for section in result.plan[0].sections for number in section["image_numbers"]] == [2]
    assert restore_plan(knowledge, serialize_plan(result.plan))[0].sections == result.plan[0].sections


@pytest.mark.parametrize("field,value", [("source", {}), ("goal_id", "g2"), ("goal_text", "Другая цель"),
                                        ("answer_refs", ("unknown",)), ("operation_refs", ("unknown",)),
                                        ("confirmation_refs", ()), ("goal_kind", "OTHER")])
def test_forged_private_scope_bindings_fail_closed(field, value):
    knowledge, _ = scanner_provider([(DOC, "Проверка.md", "Откройте проверку.".encode())])
    scope_reader(knowledge, ["Откройте"], ["Откройте"], mutate=lambda scope: scope.update({field: value}))
    with pytest.raises(KnowledgeUnavailable, match="InvalidInstructionReading"):
        knowledge.search_with_llm("Проверить", read_context({"g1": "Проверить"}, {"g1": "ACTION"}))


def test_public_json_cannot_claim_verified_scope_and_model_copy_preserves_private_backend_scope():
    with pytest.raises(ValidationError):
        SourceGoalReading.model_validate({"status": "SUPPORTED", "_verified_scope": {}})
    part = InstructionPartSelection(file_id=DOC, section_ids=["section-a"], covered_goals=[{
        "goal_id": "g1", "evidence": "Откройте окно", "support_kind": "TEXT"}])
    part._verified_scope = {"source": {"file_id": DOC}}
    assert part.model_copy(deep=True)._verified_scope == part._verified_scope
    assert "_verified_scope" not in part.model_dump()
    with pytest.raises(ValidationError):
        InstructionPartSelection.model_validate({**part.model_dump(), "_verified_scope": part._verified_scope})


def test_verified_kind_review_is_bound_to_same_goal_without_rewriting_intent():
    knowledge, _ = scanner_provider([(DOC, "Свойства.md", "Ширина устройства: 70 мм.\nВысота устройства: 40 мм.".encode())])
    scope_reader(knowledge, ["Ширина"], ["Ширина", "Высота"], kind="LOOKUP")
    goal = "Узнать ширину и высоту устройства"
    result = knowledge.search_with_llm(goal, read_context({"g1": goal}, {"g1": "ACTION"}))
    assert result.goal_kind_overrides == {"g1": "LOOKUP"}
    assert result.plan[0].goals == (goal,) and result.plan[0].delivery_mode == "EXCERPT"
    assert "Высота" in "\n".join(section["text"] for section in result.plan[0].sections)


def test_modern_one_of_keeps_actual_choice_guard_and_common_prerequisite():
    knowledge = knowledge_source()
    scope_reader(knowledge, [FIRST], [FIRST])
    result = knowledge.search_with_llm("Восстановить синим регулятором", read_context({"g1": "Восстановить"}, {"g1": "ACTION"}))
    text = "\n".join(section["text"] for section in result.plan[0].sections)
    assert FIRST in text and CHOICE in text and COMMON in text and SECOND not in text


def test_competing_one_of_paths_for_one_action_are_rejected():
    knowledge = knowledge_source()
    scope_reader(knowledge, [FIRST, SECOND], [FIRST, SECOND])
    with pytest.raises(KnowledgeUnavailable, match="CompetingInstructionMethods"):
        knowledge.search_with_llm("Восстановить", read_context({"g1": "Восстановить"}, {"g1": "ACTION"}))


def test_verified_plan_can_retain_more_than_32_complete_section_owners():
    steps = [f"{index}. Проверьте исходный параметр номер {index}." for index in range(1, 61)]
    knowledge, _ = scanner_provider([(DOC, "Большой порядок.md", "\n".join(steps).encode())])
    scope_reader(knowledge, [steps[0]], steps)
    result = knowledge.search_with_llm("Все параметры", read_context({"g1": "Проверить все параметры"}, {"g1": "ACTION"}))
    assert len(result.plan[0].section_ids) == 60
    assert restore_plan(knowledge, serialize_plan(result.plan))[0].sections == result.plan[0].sections


def test_controller_short_term_history_forwards_only_actual_employee_messages():
    context = {"current_message": "Да", "task_state": {"facts": []}, "short_term": {"messages": [
        {"role": "user", "content": "У меня один терминал и один принтер TSC по Wi-Fi."},
        {"role": "assistant", "content": "Два принтера?"}, {"role": "user", "content": "Один."}]},
        "answered_questions": ["Как подключён принтер?"], "unknown_answers": [{"question": "Модель?", "answer": "Не знаю"}]}
    result = TalkKnowledge._reading_context(context)
    assert result["employee_messages"] == [context["short_term"]["messages"][0]["content"], "Один."]
    assert result["current_message"] == "Да" and result["unknown_answers"] == context["unknown_answers"]


def test_explicit_nominator_reads_the_most_specific_original_before_lexical_candidates(monkeypatch):
    identities = [str(uuid5(NAMESPACE_URL, "early-scope/" + str(index))) for index in range(9)]
    value = "Полезное значение: 170 мм."
    rows = [(identity, f"Источник {index}.md", (value if index == 8 else "Другая тема.").encode())
            for index, identity in enumerate(identities)]
    knowledge, _, calls = configure_reader(rows, lambda wire, count: answer(wire, {"g1": value})
        if wire["approved_original"]["filename"] == "Источник 8.md" else answer(wire), monkeypatch, order=identities)
    nominations = []
    knowledge.nominator = lambda payload: nominations.append(payload) or InstructionNomination(file_ids=[identities[8]])
    result = knowledge.search_with_llm("Нужно значение", {**read_context({"g1": "Узнать значение"}), "current_message": "Нужен один принтер"})
    assert result.result == "MATCH" and len(calls) == 1
    assert nominations[0]["eligible_file_ids"] == identities
    assert nominations[0]["context"]["current_message"] == "Нужен один принтер"


def test_empty_nomination_falls_back_to_actual_ranked_originals(monkeypatch):
    value = "Нужное значение: 42."
    knowledge, _, calls = configure_reader([(DOC, "Первый.md", "Другая тема.".encode()), (OTHER, "Второй.md", value.encode())],
        lambda wire, count: answer(wire, {"g1": value}) if wire["approved_original"]["filename"] == "Второй.md" else answer(wire),
        monkeypatch, order=[DOC, OTHER])
    knowledge.nominator = lambda payload: InstructionNomination(file_ids=[])
    result = knowledge.search_with_llm("Значение", read_context({"g1": "Узнать значение"}))
    assert result.result == "MATCH" and len(calls) == 2


def test_custom_reader_without_nominator_uses_no_unexpected_native_provider(monkeypatch):
    value = "Нужное значение: 42."
    knowledge, _, calls = configure_reader([(DOC, "Свойства.md", value.encode())],
                                           lambda wire, count: answer(wire, {"g1": value}), monkeypatch)
    monkeypatch.setattr("app.services.llm.openwebui.OpenWebUILLM.nominate_instructions",
                        lambda *args: pytest.fail("Unexpected native nomination"))
    assert knowledge.search_with_llm("Значение", read_context({"g1": "Узнать значение"})).result == "MATCH"
    assert len(calls) == 1


def test_explicit_ambiguous_image_remains_strict_for_legacy_selector():
    caption = "Снимите «Размер из файла», затем включите «Печатать как рисунок»."
    knowledge = source([caption])
    knowledge.selector = lambda payload: {"result": "MATCH", **selected_part(payload, DOC, "g1", caption)}
    with pytest.raises(KnowledgeUnavailable, match="AmbiguousInstructionImageScope"):
        knowledge.search_with_llm("Исправить размер", read_context({"g1": "Исправить размер"}, {"g1": "ACTION"}))


def test_rejected_actual_scope_is_skipped_before_consent_and_search_continues_other_original(monkeypatch):
    value = "Параметр исходного устройства: 42."
    knowledge, _, calls = configure_reader([(DOC, "Первый метод.md", value.encode()),
                                           (OTHER, "Второй метод.md", value.encode())],
        lambda wire, count: answer(wire, {"g1": value}), monkeypatch, order=[DOC, OTHER])
    context = read_context({"g1": "Узнать параметр"})
    first = knowledge.search_with_llm("Какой параметр?", context)
    scopes = confirmation_scope_entries(first.approved, first.plan)
    rejected = {**context, "rejected_scope_entries": [scopes], "rejected_scope_ids": [confirmation_scope_identity(scopes)]}
    second = knowledge.search_with_llm("Уточняю тот же параметр", rejected)
    assert second.result == "MATCH" and second.approved.document["source"]["file_id"] == OTHER
    assert len(calls) == 3


def test_goal_paraphrase_does_not_bypass_rejected_actual_scope():
    value = "Параметр устройства: 42."
    knowledge, _ = scanner_provider([(DOC, "Параметр.md", value.encode())])
    scope_reader(knowledge, [value], [value], kind="LOOKUP")
    first = knowledge.search_with_llm("Какой параметр?", read_context({"g1": "Узнать параметр"}))
    scopes = confirmation_scope_entries(first.approved, first.plan)
    context = {**read_context({"g1": "Посмотреть тот же параметр"}), "rejected_scope_entries": [scopes]}
    second = knowledge.search_with_llm("Это тот же вопрос", context)
    assert second.result == "NO_KNOWLEDGE" and not second.plan


def test_other_actual_method_in_same_file_can_follow_rejected_scope():
    knowledge = knowledge_source()
    scope_reader(knowledge, [FIRST], [FIRST])
    first = knowledge.search_with_llm("Первый способ", read_context({"g1": "Восстановить"}, {"g1": "ACTION"}))
    scopes = confirmation_scope_entries(first.approved, first.plan)
    scope_reader(knowledge, [SECOND], [SECOND])
    context = {**read_context({"g1": "Восстановить другим способом"}, {"g1": "ACTION"}),
               "rejected_scope_entries": [scopes]}
    second = knowledge.search_with_llm("Другим способом", context)
    assert second.result == "MATCH" and second.approved.document["source"]["file_id"] == DOC
    assert SECOND in "\n".join(section["text"] for section in second.plan[0].sections)


def test_independent_critic_complete_plan_restores_middle_step_absent_from_reader_proofs():
    steps = ["1. Откройте форму проверки.", "2. Введите выданное значение.", "3. Проверьте завершение."]
    knowledge, _ = scanner_provider([(DOC, "Три шага.md", "\n".join(steps).encode())])
    knowledge.settings = knowledge.settings.model_copy(update={"llm_backend": "openwebui"})

    def verify(wire, count):
        refs = [wire_evidence(wire, step)[0] for step in steps]
        assert len(set(refs)) == 3
        return {"source_finding": "В оригинале три последовательных шага.", "applicability_refs": [],
                "operation_refs": [refs[0]], "condition_match": "EXACT", "required_material_kind": "TEXT",
                "material_refs": [], "material_state": "READY", "all_materials_available": True, "verdict": "PASS",
                "operation_match": "EXACT", "goal_kind": "ACTION", "missing_observable_condition": "NONE",
                "condition_refs": [], "question": "", "clarification_kind": None, "answer_refs": refs}

    model, calls = model_verifier(verify)

    def read(payload):
        refs = [reference for reference, proof in payload["candidate_evidence"].items() if proof["text"] == steps[0]]
        proposal = InstructionReading(goals={"g1": SourceGoalReading(status="SUPPORTED", evidence_ids=refs,
                                                                     delivery_mode="PROCEDURE")})
        return model.verify_instruction(payload, proposal)

    knowledge.reader = read
    result = knowledge.search_with_llm("Выполнить проверку", read_context({"g1": "Проверить"}, {"g1": "ACTION"}))
    assert [section["text"] for section in result.plan[0].sections] == steps and len(calls) == 1


def test_same_original_picture_can_support_a_different_parameter_after_scope_rejection():
    lighting = "Настройка подсветки сканера описана на исходном листе."
    volume = "Настройка громкости сканера описана на исходном листе."
    caption = lighting + " " + volume
    body = "<w:p><w:r><w:t>" + lighting + "</w:t></w:r></w:p><w:p><w:r><w:t>" + volume + "</w:t></w:r></w:p>"
    knowledge, _ = scanner_provider([(DOC, "Параметры сканера.docx", instruction([caption], extra_paragraph=body))])
    scope_reader(knowledge, [lighting], [lighting, caption], materials=[caption])
    first = knowledge.search_with_llm("Подсветка", read_context({"g1": "Изменить подсветку"}, {"g1": "ACTION"}))
    first_scopes = confirmation_scope_entries(first.approved, first.plan)
    scope_reader(knowledge, [volume], [volume, caption], materials=[caption])
    context = {**read_context({"g1": "Изменить громкость"}, {"g1": "ACTION"}), "rejected_scope_entries": [first_scopes]}
    second = knowledge.search_with_llm("Громкость", context)
    second_scopes = confirmation_scope_entries(second.approved, second.plan)
    assert second.result == "MATCH" and second.approved.document["source"]["file_id"] == DOC
    assert confirmation_scope_identity(first_scopes) != confirmation_scope_identity(second_scopes)
    assert any(volume in item["text"] for item in second_scopes[0]["context"])
    assert [number for section in second.plan[0].sections for number in section["image_numbers"]] == [1]


def test_private_scope_registry_forgery_is_rechecked_against_fresh_original():
    knowledge, _ = scanner_provider([(DOC, "Два шага.md", "Откройте окно.\nПроверьте результат.".encode())])
    scope_reader(knowledge, ["Откройте окно"], ["Откройте окно", "Проверьте результат"])
    original_reader = knowledge.reader

    def forged(payload):
        reading = original_reader(payload)
        reference = reading.goals["g1"]._verified_scope["answer_refs"][-1]
        payload["candidate_evidence"][reference]["text"] = "Посторонний выдуманный шаг."
        return reading

    knowledge.reader = forged
    with pytest.raises(KnowledgeUnavailable, match="InvalidInstructionVerifiedScope"):
        knowledge.search_with_llm("Проверка", read_context({"g1": "Проверить"}, {"g1": "ACTION"}))


def test_contract_telemetry_logs_stable_cause_and_source_identity_without_source_or_user_text(monkeypatch):
    secret_text = "Секретное содержание исходного материала."
    knowledge, _ = scanner_provider([(DOC, "Материал.md", secret_text.encode())])
    knowledge.settings = knowledge.settings.model_copy(update={"llm_backend": "openwebui"})
    knowledge.reader = lambda payload: InstructionReading(goals={"g1": SourceGoalReading(
        status="SUPPORTED", evidence_ids=["unknown-reference"], delivery_mode="PROCEDURE")})
    events = []
    monkeypatch.setattr("app.services.rag.talk._search_event", lambda stage, **fields: events.append((stage, fields)))
    with pytest.raises(KnowledgeUnavailable, match="InvalidInstructionReading"):
        knowledge.search_with_llm("Личная реплика сотрудника", read_context({"g1": "Проверить"}, {"g1": "ACTION"}))
    failure = next(fields for stage, fields in events if stage == "instruction_reading_contract_error")
    assert failure["file_id"] == DOC and len(failure["source_sha256"]) == 64
    assert failure["contract_cause"] == "UnknownInstructionReadingEvidence"
    assert secret_text not in str(events) and "Личная реплика сотрудника" not in str(events)


def test_native_default_reader_nominates_before_reading_first_original(monkeypatch):
    value = "Значение параметра: 42."
    knowledge, state = scanner_provider([(DOC, "Общий материал.md", "Общие сведения.".encode()),
                                         (OTHER, "Точный материал.md", value.encode())])
    knowledge.settings = knowledge.settings.model_copy(update={"llm_backend": "openwebui"})
    calls = []

    def nominate(model, payload):
        assert not any(request.url.path.endswith("/content") for request in state["calls"])
        assert payload["eligible_file_ids"] == [DOC, OTHER]
        calls.append("nominate")
        return InstructionNomination(file_ids=[OTHER])

    def read(model, payload):
        calls.append("read")
        assert payload["source"]["file_id"] == OTHER
        reference = next(reference for reference, proof in payload["candidate_evidence"].items() if value in proof["text"])
        return InstructionReading(goals={"g1": SourceGoalReading(status="SUPPORTED", evidence_ids=[reference],
                                                                 delivery_mode="EXCERPT")})

    monkeypatch.setattr("app.services.llm.openwebui.OpenWebUILLM.nominate_instructions", nominate)
    monkeypatch.setattr("app.services.llm.openwebui.OpenWebUILLM.read_instruction", read)
    result = knowledge.search_with_llm("Значение", read_context({"g1": "Узнать значение"}))
    assert result.result == "MATCH" and calls == ["nominate", "read"]
