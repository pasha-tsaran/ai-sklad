"""The model selects meaning; the backend proves current scoped source ownership."""

import copy
import json

import httpx
import pytest

from app.services.rag.query import source_compatible
from app.services.rag.talk import (
    MAX_SELECTION_CATALOGUE_CHARS, MAX_SELECTION_CONTEXT_CHARS, MAX_SELECTION_SOURCE_CHARS,
    INSTRUCTION_SELECTION_TASK, InstructionSelection, KnowledgeUnavailable,
)
from integrations.openwebui.approved_instruction_images import read_docx_images
from tests.test_openwebui_instruction_images import instruction
from tests.test_scanner_original_retrieval import scanner_provider
from tests.test_talk_knowledge import DOC, KB, OTHER, provider


EVIDENCE = "Уровень звукового уведомления задаётся кодом на исходной фотографии."


def match(file_id=DOC, evidence=EVIDENCE):
    return {"result": "MATCH", "file_id": file_id, "evidence": evidence}


def selection_store():
    knowledge, state = provider(filename="Параметры устройства.docx", data=instruction([EVIDENCE]), chunks=[])
    decisions = []

    def select(payload):
        decisions.append(copy.deepcopy(payload))
        return match()

    knowledge.selector = select
    return knowledge, state, decisions


def test_single_source_still_requires_model_selection_and_keeps_original_photo():
    knowledge, state, decisions = selection_store()
    problem = "Уменьшить громкость сканера штрихкодов"
    assert not source_compatible(problem, "Параметры устройства.docx\n" + EVIDENCE, complete=True)
    result = knowledge.search_with_llm(problem, {"incident": {"problem_type": "звуковой сигнал"}})
    assert result.result == "MATCH"
    approved = result.approved
    assert approved.document["source"]["file_id"] == DOC
    assert approved.document["source"]["knowledge_id"] == KB
    original_photo = read_docx_images(state["data"])[0]
    assert approved.illustrations[1].data == original_photo.data
    assert approved.illustrations[1].caption == EVIDENCE
    assert len(decisions) == 1
    assert decisions[0]["actual_problem"] == problem
    assert decisions[0]["candidates"][DOC] == approved.document["content"]
    assert not any(request.method == "POST" for request in state["calls"])
    assert sum(request.url.path.endswith("/content") for request in state["calls"]) == 2


def test_default_selector_uses_real_model_transport_with_scoped_catalogue_and_no_tools():
    knowledge, state, _ = selection_store()
    knowledge.selector = None
    original_transport = knowledge.transport
    model_calls = []

    def handler(request):
        if request.url.path != "/api/chat/completions":
            return original_transport.handle_request(request)
        assert request.url.host == "webui.example.invalid"
        assert request.headers["Authorization"] == "Bearer test-key"
        payload = json.loads(request.content)
        model_calls.append(payload)
        assert payload["model"] == knowledge.settings.openwebui_model
        assert payload["tools"] == [] and payload["tool_ids"] == []
        assert payload["features"] == {"web_search": False}
        assert payload["temperature"] == 0 and payload["stream"] is False
        assert payload["response_format"]["type"] == "json_schema"
        assert payload["params"]["think"] is False
        assert payload["params"]["max_tokens"] == payload["max_tokens"] == 512
        system = payload["messages"][0]["content"]
        assert INSTRUCTION_SELECTION_TASK in system
        assert '"title": "InstructionSelection"' in system
        assert '"additionalProperties": false' in system
        source_payload = json.loads(payload["messages"][-1]["content"])
        assert source_payload["actual_problem"] == "The scanner is too loud."
        assert source_payload["knowledge_ids"] == [KB]
        assert source_payload["catalogue"] == {DOC: state["files"][0]["filename"]}
        assert EVIDENCE in source_payload["candidates"][DOC]
        return httpx.Response(200, json={
            "choices": [{"message": {"content": json.dumps(match())}}],
        })

    knowledge.transport = httpx.MockTransport(handler)
    result = knowledge.search_with_llm("The scanner is too loud.")
    assert result.result == "MATCH" and result.approved.illustrations[1].data
    assert len(model_calls) == 1
    assert sum(request.url.path.endswith("/content") for request in state["calls"]) == 2


def test_instruction37_without_native_index_hit_is_selected_by_model():
    filename = "37_Zebra DS2208 — лист настроечных штрихкодов.docx"
    knowledge, state = scanner_provider([(DOC, filename, instruction([EVIDENCE]))], chunks=[])
    calls = []
    knowledge.selector = lambda payload: calls.append(payload) or match()
    result = knowledge.search_with_llm("Настройка сканер ШК\nуменьшить громкость")
    assert result.result == "MATCH" and len(result.approved.illustrations) == 1
    assert len(calls) == 1
    assert calls[0]["knowledge_ids"] == [KB]
    assert calls[0]["catalogue"] == {DOC: filename}
    assert not any(request.method == "POST" for request in state["calls"])


@pytest.mark.parametrize("answer,expected", [
    ({"result": "CLARIFY", "question": "Это сканер штрихкодов или сканер документов?"}, "CLARIFY"),
    ({"result": "NO_KNOWLEDGE"}, "NO_KNOWLEDGE"),
])
def test_explicit_clarification_and_absence_are_typed_model_decisions(answer, expected):
    knowledge, _, calls = selection_store()
    knowledge.selector = lambda payload: calls.append(payload) or answer
    result = knowledge.search_with_llm("Помогите со сканером")
    assert result.result == expected and result.approved is None
    assert result.question == answer.get("question", "")
    assert len(calls) == 1


@pytest.mark.parametrize("answer", [
    {"result": "MATCH", "file_id": OTHER, "evidence": EVIDENCE},
    {"result": "MATCH", "file_id": DOC, "evidence": "Такой цитаты в выбранном документе вообще нет."},
    {"result": "MATCH", "file_id": DOC, "evidence": EVIDENCE, "answer": "Удалите данные."},
    {"result": "MATCH", "file_id": DOC, "evidence": EVIDENCE, "question": "Сделать сброс?"},
    {"result": "CLARIFY", "question": "Что случилось?", "file_id": DOC},
    {"result": "CLARIFY", "question": ""},
    {"result": "NO_KNOWLEDGE", "file_id": DOC},
    {"result": "NO_KNOWLEDGE", "evidence": EVIDENCE},
    {"file_id": DOC, "evidence": EVIDENCE},
])
def test_untrusted_choice_cannot_forge_ids_quotes_fields_or_technical_answers(answer):
    knowledge, _, calls = selection_store()
    knowledge.selector = lambda payload: calls.append(payload) or answer
    with pytest.raises(KnowledgeUnavailable):
        knowledge.search_with_llm("Нужна помощь")
    assert len(calls) == 1


def test_evidence_may_normalize_whitespace_but_preserves_actual_words():
    knowledge, _, _ = selection_store()
    knowledge.selector = lambda payload: match(evidence=EVIDENCE.replace(" ", "\n"))
    assert knowledge.search_with_llm("Нужна помощь").result == "MATCH"
    knowledge.selector = lambda payload: match(evidence="Уровень  " + " " * 30 + "звука")
    with pytest.raises(KnowledgeUnavailable, match="InstructionEvidenceMismatch"):
        knowledge.search_with_llm("Нужна помощь")


@pytest.mark.parametrize("change", ["bytes", "membership"])
def test_source_changes_during_model_call_prevent_delivery(change):
    knowledge, state, calls = selection_store()

    def select(payload):
        calls.append(payload)
        if change == "bytes":
            state["data"] = instruction([EVIDENCE + " Дополнение изменило оригинал."])
        else:
            state["files"] = []
        return match()

    knowledge.selector = select
    with pytest.raises(KnowledgeUnavailable):
        knowledge.search_with_llm("Нужна помощь")
    assert len(calls) == 1


def large_catalogue():
    rows = [(DOC, "Материал номер один.md", "Общая справка по работе устройства.".encode())]
    rows += [(f"00000000-0000-4000-8000-{index:012d}", f"Материал номер {index}.md", b"General guidance.")
             for index in range(101, 109)]
    rows.append((OTHER, "Другой документ.md", EVIDENCE.encode()))
    return scanner_provider(rows, chunks=[(DOC, "Общая справка по работе устройства.")])


def test_model_can_request_another_authorized_original_and_then_confirm_it():
    knowledge, state = large_catalogue()
    calls = []

    def select(payload):
        calls.append(copy.deepcopy(payload))
        return {"result": "READ", "file_id": OTHER} if len(calls) == 1 else match(OTHER)

    knowledge.selector = select
    result = knowledge.search_with_llm("Прибор разбудил весь склад писком")
    assert result.result == "MATCH" and result.approved.document["source"]["file_id"] == OTHER
    assert len(calls) == 2
    assert set(calls[0]["candidates"]) == {DOC}
    assert set(calls[1]["candidates"]) == {DOC, OTHER}
    assert len(calls[0]["catalogue"]) == 10 and calls[1]["read_available"] is False
    assert all(request.url.host == "webui.example.invalid" for request in state["calls"])


@pytest.mark.parametrize("first,second", [
    ({"result": "READ", "file_id": "00000000-0000-4000-8000-000000000777"}, None),
    ({"result": "READ", "file_id": DOC}, None),
    ({"result": "READ", "file_id": OTHER}, {"result": "READ", "file_id": DOC}),
])
def test_read_cannot_leave_scope_repeat_a_loaded_file_or_trigger_unbounded_calls(first, second):
    knowledge, state = large_catalogue()
    calls = []
    knowledge.selector = lambda payload: calls.append(payload) or (first if len(calls) == 1 else second)
    with pytest.raises(KnowledgeUnavailable, match="InvalidInstructionRead"):
        knowledge.search_with_llm("Прибор разбудил весь склад писком")
    assert len(calls) == (2 if second is not None else 1)
    assert not any("000000000777" in request.url.path for request in state["calls"])


@pytest.mark.parametrize("filename,excluded", [
    ("33_Сканер штрихкодов в базе АДРЕС.docx", ()),
    ("Параметры устройства.docx", (DOC,)),
])
def test_withdrawn_and_excluded_sources_are_absent_from_model_catalogue(filename, excluded):
    knowledge, state = provider(filename=filename, data=instruction([EVIDENCE]))
    payloads = []
    knowledge.selector = lambda payload: payloads.append(payload) or {"result": "NO_KNOWLEDGE"}
    result = knowledge.search_with_llm("Нужна помощь", excluded=excluded)
    assert result.result == "NO_KNOWLEDGE" and payloads[0]["catalogue"] == {}
    assert not any("/files/" in request.url.path for request in state["calls"])


def test_foreign_vector_ids_and_contents_never_reach_the_model_or_original_reader():
    foreign = "00000000-0000-4000-8000-000000000777"
    rows = [(f"00000000-0000-4000-8000-{index:012d}", f"Материал {index}.md", b"General guidance.")
            for index in range(101, 110)]
    knowledge, state = scanner_provider(rows, chunks=[(foreign, "Private foreign source contents.")])
    payloads = []
    knowledge.selector = lambda payload: payloads.append(payload) or {"result": "NO_KNOWLEDGE"}
    assert knowledge.search_with_llm("Прибор разбудил склад писком").result == "NO_KNOWLEDGE"
    assert payloads[0]["candidates"] == {}
    assert foreign not in str(payloads) and "Private foreign" not in str(payloads)
    assert not any("/api/v1/files/" in request.url.path for request in state["calls"])


@pytest.mark.parametrize("failure", [httpx.ReadTimeout("private details"), TimeoutError("private details"),
                                     httpx.ConnectError("private details"), ValueError("private details")])
def test_model_failure_never_becomes_a_missing_knowledge_claim_or_retry(failure, caplog):
    knowledge, _, calls = selection_store()

    def select(payload):
        calls.append(payload)
        raise failure

    knowledge.selector = select
    with caplog.at_level("INFO", logger="ai-support"):
        with pytest.raises(KnowledgeUnavailable) as captured:
            knowledge.search_with_llm("Нужна помощь")
    assert captured.value.stage == "instruction_selection" and len(calls) == 1
    fields = next(record.fields for record in caplog.records if record.message == "talk.knowledge_error")
    assert "private details" not in str(fields) and "test-key" not in str(fields)


def test_oversized_original_cannot_turn_incomplete_context_into_no_knowledge():
    content = EVIDENCE + "x" * MAX_SELECTION_SOURCE_CHARS
    knowledge, _ = scanner_provider([(DOC, "Описание устройства.md", content.encode())])
    payloads = []
    knowledge.selector = lambda payload: payloads.append(payload) or {"result": "NO_KNOWLEDGE"}
    with pytest.raises(KnowledgeUnavailable, match="SelectionBudgetIncomplete"):
        knowledge.search_with_llm("Нужна помощь")
    assert payloads[0]["candidate_budget_limited"] is True and payloads[0]["candidates"] == {}


def test_large_catalogue_is_rejected_before_any_partial_listing_reaches_model():
    rows = [(f"00000000-0000-4000-8000-{index:012d}", "Имя" * 150 + f"{index}.md", EVIDENCE.encode())
            for index in range(1, 31)]
    knowledge, state = scanner_provider(rows)
    calls = []
    knowledge.selector = lambda payload: calls.append(payload)
    with pytest.raises(KnowledgeUnavailable, match="SelectionCatalogueTooLarge"):
        knowledge.search_with_llm("Нужна помощь")
    assert MAX_SELECTION_CATALOGUE_CHARS < sum(len(row[1]) for row in rows)
    assert not calls and not any("/files/" in request.url.path for request in state["calls"])


def test_context_is_bounded_while_actual_user_problem_remains_complete():
    knowledge, _, payloads = selection_store()
    problem = "Нужна помощь с тихим сигналом"
    latest = "У сканера штрихкодов надо уменьшить громкость"
    conversation = [{"role": "assistant", "content": "old" * 10_000}]
    conversation += [{"role": "user", "content": "short recent remark " * 30} for _ in range(3)]
    conversation.append({"role": "user", "content": latest})
    result = knowledge.search_with_llm(problem, {"short_term": {"messages": conversation},
                                               "current_message": latest})
    context = payloads[0]["context"]
    assert result.result == "MATCH" and payloads[0]["actual_problem"] == problem
    assert len(json.dumps(context, ensure_ascii=False)) <= MAX_SELECTION_CONTEXT_CHARS
    assert context["current_message"] == latest
    assert context["recent_messages"][-1] == {"role": "user", "content": latest}
    assert all(len(message["content"]) <= 400 for message in context["recent_messages"])
    assert "old" not in str(context) and "context_excerpt" not in context


def test_compact_catalogue_keeps_all_fifty_ids_names_and_complete_source_conditions(caplog):
    filename = "37_Zebra DS2208 — лист настроечных штрихкодов.docx"
    conditions = EVIDENCE + " Применяется только к указанному устройству. Не сканируйте все коды подряд."
    source_data = instruction([conditions])
    rows = [(DOC, filename, source_data)]
    rows += [(f"00000000-0000-4000-8000-{index:012d}", f"Материал по работе программы {index}.md",
              b"Other approved guidance.") for index in range(101, 150)]
    knowledge, _ = scanner_provider(rows, chunks=[])
    payloads = []
    knowledge.selector = lambda payload: payloads.append(copy.deepcopy(payload)) or match()
    with caplog.at_level("INFO", logger="ai-support"):
        decision = knowledge.search_with_llm("Уменьшить громкость сканера")
    payload = payloads[0]
    assert decision.result == "MATCH" and payload["knowledge_ids"] == [KB]
    assert payload["catalogue"] == {file_id: name for file_id, name, _ in rows}
    assert payload["candidates"] == {DOC: decision.approved.document["content"]}
    assert conditions in payload["candidates"][DOC]
    assert decision.approved.illustrations[1].data == read_docx_images(source_data)[0].data
    expanded = {**payload, "catalogue": [
        {"file_id": file_id, "filename": name, "knowledge_id": KB} for file_id, name, _ in rows
    ], "candidates": [{"file_id": DOC, "source": filename, "knowledge_id": KB,
                       "content": payload["candidates"][DOC]}]}
    assert len(json.dumps(payload, ensure_ascii=False)) < len(json.dumps(expanded, ensure_ascii=False)) * 0.85
    fields = next(record.fields for record in caplog.records if record.message == "talk.knowledge_search"
                  and record.fields.get("stage") == "instruction_selection_request")
    assert fields["approved_files"] == 50 and fields["candidate_count"] == fields["call_number"] == 1
    assert fields["payload_chars"] == len(json.dumps(payload, ensure_ascii=False))
    assert fields["source_chars"] == len(payload["candidates"][DOC])
    assert all(isinstance(value, int) for name, value in fields.items() if name != "stage")
    assert DOC not in str(fields) and filename not in str(fields) and EVIDENCE not in str(fields)


def test_selection_context_keeps_technical_facts_and_recent_exchange_without_profile_or_record_ids():
    knowledge, _, payloads = selection_store()
    context = {
        "authority": "PRIVATE_AUTHORITY",
        "incident": {"id": "PRIVATE_INCIDENT", "state": "PRIVATE_STATE", "failed_attempts": 42,
                     "subject": "Настройка сканера ШК", "equipment_model": "Сообщённая пользователем модель",
                     "problem_type": "громкость"},
        "channel_profile": {"account_holder_name": "PRIVATE_PROFILE", "computer_name": "PRIVATE_COMPUTER"},
        "short_term": {"summary": "PRIVATE_OLD_SUMMARY", "attempts": [{"comment": "PRIVATE_ATTEMPT"}],
                       "messages": [{"role": "user", "sequence": "PRIVATE_SEQUENCE", "content": "Сканер ШК"},
                                    {"role": "assistant", "content": "Что нужно изменить?"},
                                    {"role": "user", "content": "уменьшить громкость"}]},
        "long_term": [{"incident_id": "PRIVATE_HISTORY", "resolved_at": "PRIVATE_TIMESTAMP"}],
        "verified_previous_source": {"file_id": DOC, "sha256": "PRIVATE_SOURCE_HASH"},
        "previous_question": "Что нужно изменить?", "current_message": "уменьшить громкость",
    }
    assert knowledge.search_with_llm("Настройка сканера ШК\nуменьшить громкость", context).result == "MATCH"
    selected = payloads[0]["context"]
    assert selected["technical_facts"] == {key: context["incident"][key]
                                           for key in ("subject", "equipment_model", "problem_type")}
    assert selected["current_message"] == "уменьшить громкость"
    assert selected["previous_question"] == "Что нужно изменить?"
    assert selected["recent_messages"] == [
        {"role": "user", "content": "Сканер ШК"},
        {"role": "assistant", "content": "Что нужно изменить?"},
        {"role": "user", "content": "уменьшить громкость"},
    ]
    assert selected["verified_previous_source"] == {"file_id": DOC}
    assert "PRIVATE_" not in json.dumps(selected)
    assert len(json.dumps(selected, ensure_ascii=False)) <= MAX_SELECTION_CONTEXT_CHARS


def test_oversized_optional_context_does_not_splice_fields_or_lose_actual_problem():
    knowledge, _, payloads = selection_store()
    problem = "Сканер\n" + "уточнение " * 500
    context = {"current_message": "\\\"" * 3000, "previous_question": "Что нужно изменить?",
               "incident": {"problem_type": "громкость"},
               "short_term": {"messages": [{"role": "user", "content": "уменьшить громкость"}]}}
    assert knowledge.search_with_llm(problem, context).result == "MATCH"
    selected = payloads[0]["context"]
    assert payloads[0]["actual_problem"] == problem
    assert selected["previous_question"] == "Что нужно изменить?"
    assert selected["technical_facts"] == {"problem_type": "громкость"}
    assert "current_message" not in selected
    assert selected["recent_messages"] == [{"role": "user", "content": "уменьшить громкость"}]
    assert len(json.dumps(selected, ensure_ascii=False)) <= MAX_SELECTION_CONTEXT_CHARS


def test_verified_previous_source_is_a_hint_and_still_requires_current_model_choice():
    knowledge, state = large_catalogue()
    calls = []
    knowledge.selector = lambda payload: calls.append(payload) or match(OTHER)
    result = knowledge.search_with_llm("Прибор разбудил склад", {"verified_previous_source": {"file_id": OTHER}})
    assert result.result == "MATCH" and len(calls) == 1
    assert set(calls[0]["candidates"]) == {DOC, OTHER}


def test_verified_previous_source_does_not_hide_a_better_instruction_in_small_catalogue():
    knowledge, _ = scanner_provider([
        (DOC, "Первый документ.md", "Ранее подходившая инструкция с исходными условиями.".encode()),
        (OTHER, "Другой документ.md", EVIDENCE.encode()),
    ])
    payloads = []
    knowledge.selector = lambda payload: payloads.append(payload) or match(OTHER)
    decision = knowledge.search_with_llm("Новая проблема", {"verified_previous_source": {"file_id": DOC}})
    assert decision.result == "MATCH" and decision.approved.document["source"]["file_id"] == OTHER
    assert list(payloads[0]["candidates"]) == [DOC, OTHER]


@pytest.mark.parametrize("answer", ["MATCH", "NO_KNOWLEDGE"])
def test_model_can_read_catalogue_original_when_optional_vector_discovery_is_unavailable(answer):
    knowledge, state = large_catalogue()
    original = knowledge._instruction_vector_ids

    def fail(client, problem, files):
        raise KnowledgeUnavailable("RequestTimedOut", "retrieval")

    knowledge._instruction_vector_ids = fail
    calls = []

    def select(payload):
        calls.append(copy.deepcopy(payload))
        if answer == "NO_KNOWLEDGE":
            return {"result": "NO_KNOWLEDGE"}
        return {"result": "READ", "file_id": OTHER} if len(calls) == 1 else match(OTHER)

    knowledge.selector = select
    if answer == "MATCH":
        decision = knowledge.search_with_llm("Прибор разбудил склад писком")
        assert decision.result == "MATCH" and decision.approved.document["source"]["file_id"] == OTHER
        assert len(calls) == 2
    else:
        with pytest.raises(KnowledgeUnavailable, match="RequestTimedOut"):
            knowledge.search_with_llm("Прибор разбудил склад писком")
        assert len(calls) == 1
    assert calls[0]["retrieval_available"] is False
    assert all(request.method == "GET" for request in state["calls"])
    knowledge._instruction_vector_ids = original


def test_selection_mode_is_explicitly_available_only_for_real_model_backend():
    knowledge, _, _ = selection_store()
    assert knowledge.uses_llm_selection is False
    knowledge.settings.llm_backend = "openwebui"
    assert knowledge.uses_llm_selection is True
    assert InstructionSelection.model_validate(match()).result == "MATCH"
