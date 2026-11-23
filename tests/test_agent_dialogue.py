import copy
import json

import httpx
import pytest
from sqlalchemy import select

from app.core.config import Settings
from app.models import Delivery, Feedback, ResolutionMemory
from app.services.chat.agent_dialogue import (
    MODEL_UNAVAILABLE, OUTLOOK_LARGE_MAIL_FILE, OUTLOOK_LARGE_MAIL_SHA256, PAGE_CHARS, WINDOWS_SCREEN_FILE,
    WINDOWS_SCREEN_SHA256, _PRINT_SCHEMES, run_agent,
)
from app.services.chat.talk_dialogue import AGENT_OUTCOME_QUESTION, UNAVAILABLE, _resume_turn, _text_turn
from app.services.chat.talk_feedback import learned_feedback, reorder_candidates, source_key, split_agent_feedback
from app.services.llm.openwebui import OpenWebUILLM
from app.services.llm.support_agent import AgentAction
from app.services.rag.talk import KnowledgeUnavailable
from tests.conftest import env as env
from tests import test_talk_dialogue as legacy
from tests.test_talk_dialogue import Knowledge, approved, assistants
from tests.test_talk_knowledge import provider

legacy_talk = legacy.talk


@pytest.fixture
def talk(legacy_talk):
    legacy_talk.settings.talk_dialogue_mode = "agent"
    return legacy_talk


class Agent:
    name = "scripted-agent"

    def __init__(self, *actions):
        self.actions = iter(actions)
        self.payloads = []

    def agent_action(self, payload):
        self.payloads.append(copy.deepcopy(payload))
        return next(self.actions)


class Sources(Knowledge):
    def search_candidates(self, query):
        self.searches.append(query)
        return {"candidates": [{"file_id": d.document["source"]["file_id"], "title": d.candidate.title}
                               for d in self.documents], "search_degraded": False}


def search():
    return {"action": "search_knowledge", "query": "принтер Bluetooth Wi-Fi"}


def read(doc, offset=0):
    return {"action": "read_instruction", "file_id": doc.document["source"]["file_id"], "offset": offset}


def answer(doc, number=1, *, text=None):
    return {"action": "reply", "kind": "instruction", "blocks": [{
        "text": text or f"Выполните шаг {number}. Напишите, когда будете готовы.",
        "references": [{"file_id": doc.document["source"]["file_id"], "quote": doc.illustrations[number].caption}],
        "images": [{"file_id": doc.document["source"]["file_id"], "number": number}],
    }], "memory": {"goal": "Настроить печать", "current_step": str(number), "next_step": str(number + 1),
                     "expected_reply": "Готово или продолжить"}}


def settings(**kwargs):
    return Settings(_env_file=None, **kwargs)


def test_search_read_reply_then_continue_without_search_or_confirmation():
    doc = approved()
    knowledge = Sources([doc])
    agent = Agent(search(), read(doc), answer(doc))
    first = _text_turn("Настроить печать", {}, {}, None, agent, knowledge, settings())
    assert first.action == "AGENT_REPLY"
    assert first.blocks[1][1].data == doc.illustrations[1].data
    assert first.flow["agent"]["memory"]["current_step"] == "1"
    # Fresh agent instance simulates another worker/process. Only persisted state is reused.
    following = Agent(read(doc), answer(doc, 2))
    second = _text_turn("продолжить", first.flow, {}, None, following, knowledge, settings())
    assert len(knowledge.searches) == 1
    assert second.flow["agent"]["memory"]["current_step"] == "2"
    assert following.payloads[0]["memory"]["next_step"] == "2"
    assert not any("Правильно" in text or "Нашлась инструкция" in text for text, _ in second.blocks)
    tool_read = agent.payloads[-1]["tool_results"][-1]["result"]
    assert tool_read["content"] == doc.document["content"]
    assert tool_read["images"][0]["description"] == doc.illustrations[1].caption
    assert "data" not in tool_read["images"][0]


def test_changed_configuration_is_visible_to_agent_and_can_choose_other_source():
    single, dual = approved(), approved(file_id="00000000-0000-4000-8000-000000000055")
    knowledge = Sources([single, dual])
    first = run_agent("ТСД и Urovo по Bluetooth", {}, {}, Agent(search(), read(single), answer(single)), knowledge, settings())
    agent = Agent(search(), read(dual), answer(dual))
    change = "а теперь один принтер по блютуз, а другой по wi-fi"
    second = run_agent(change, first.flow, {"short_term": {"messages": [
        {"role": "user", "content": "ТСД и Urovo по Bluetooth"}]}}, agent, knowledge, settings())
    assert agent.payloads[0]["message"] == change
    assert "Urovo" in agent.payloads[0]["context"]["short_term"]["messages"][0]["content"]
    assert second.flow["source"] == dual.document["source"]


def test_new_topic_drops_old_source_and_keeps_only_new_history_on_followup():
    doc = approved()
    clarification = {"action": "reply", "kind": "clarification", "blocks": [{"text": "Какая программа?"}],
                     "memory": {"goal": "Почта не открывается", "expected_reply": "Название программы"}}
    old = {"agent": {"memory": {"goal": "Печать"}, "sources": [doc.document["source"]]},
           "source": doc.document["source"]}
    messages = [{"sequence": 9, "role": "assistant", "content": "Настройте LabelPrint."}]
    agent = Agent(read(doc), clarification)
    agent.classify_task_relation = lambda *_: "new"
    result = run_agent("Не открывается почта", old, {"short_term": {"messages": messages}},
                       agent, Sources([doc]), settings())
    assert agent.payloads[0]["active_sources"] == []
    assert agent.payloads[0]["context"]["short_term"]["messages"] == []
    assert agent.payloads[1]["tool_results"][-1]["error"] == "SearchBeforeReadingUnknownSource"
    assert "source" not in result.flow
    messages.append({"sequence": 10, "role": "user", "content": "Не открывается почта"})
    followup = Agent(clarification)
    followup.classify_task_relation = lambda *_: "current"
    run_agent("Outlook", result.flow, {"short_term": {"messages": messages}}, followup, Sources(), settings())
    assert followup.payloads[0]["context"]["short_term"]["messages"] == messages[1:]


def test_model_timeout_is_not_reported_as_knowledge_outage():
    class TimedOutAgent:
        def agent_action(self, payload):
            raise httpx.ReadTimeout("private provider detail")

    assert run_agent("печать", {}, {}, TimedOutAgent(), Sources(), settings()).blocks == [(MODEL_UNAVAILABLE, None)]


def test_routing_timeout_cannot_retain_previous_answer_as_source_of_new_clarification():
    old_doc = approved()
    new_doc = approved(file_id="00000000-0000-4000-8000-000000000032")
    knowledge = Sources([new_doc])
    knowledge.prefetch_candidates = knowledge.search_candidates
    saved = {"source": old_doc.document["source"], "agent": {
        "memory": {"goal": "Печать", "facts": ["предыдущий принтер"]}, "sources": [old_doc.document["source"]]}}
    agent = Agent({"action": "reply", "kind": "clarification", "blocks": [{"text": "Что видно при входе?"}]})

    def timeout(*args):
        raise httpx.ReadTimeout("timed out")

    agent.classify_task_relation = timeout
    turn = run_agent("Почта перестала открываться", saved, {"short_term": {"messages": [
        {"sequence": 7, "role": "assistant", "content": "Настройте принтер."}]}}, agent, knowledge, settings())
    payload = agent.payloads[0]
    assert payload["active_sources"] == [] and payload["context"]["short_term"]["messages"] == []
    assert not payload["memory"]["goal"] and not payload["memory"]["facts"]
    assert turn.flow["agent"]["sources"] == [] and "source" not in turn.flow
    assert turn.flow["agent"]["start_sequence"] == 8


def test_explicit_continuation_uses_verified_progress_without_requesting_topic_classification():
    doc = approved()
    knowledge = Sources([doc])
    knowledge.prefetch_candidates = knowledge.search_candidates
    flow = {"agent": {"memory": {"goal": "Настроить печать", "next_step": "2"},
                     "sources": [doc.document["source"]]}}
    agent = Agent(read(doc), answer(doc, 2))
    agent.classify_task_relation = lambda *_: pytest.fail("Explicit continuation was routed again")
    result = run_agent("Продолжить", flow, {}, agent, knowledge, settings())
    assert result.blocks[1][1] == doc.illustrations[2] and not knowledge.searches


def test_large_outlook_mail_uses_current_approved_original_without_model():
    doc = approved(file_id=OUTLOOK_LARGE_MAIL_FILE)
    doc.document["source"]["sha256"] = OUTLOOK_LARGE_MAIL_SHA256
    doc.document["content"] = "\n".join(
        f"{number}. Настройка папки поиска 1024 КБ" for number in range(1, 6))
    question = "Надо отследить большой размер писем, как это сделать?"
    knowledge = Sources([doc])
    old = {"agent": {"memory": {"goal": "Настроить печать"}}}
    result = run_agent(question, old, {}, Agent(), knowledge, settings())
    assert result.action == "AGENT_REPLY" and len(result.blocks) == 8
    assert result.agent_documents[0].document["source"] == doc.document["source"]
    assert "1024 КБ" in result.blocks[0][0]
    assert [picture for _, picture in result.blocks[1::2]] == [doc.illustrations[n] for n in (2, 3, 4, 5)]
    assert [text.split(". Настройка")[0][-1] for text, _ in result.blocks[2::2]] == ["3", "4", "5"]
    assert len(knowledge.checks) == 1

    no_source = {"action": "reply", "kind": "clarification", "blocks": [{"text": "Уточните программу."}]}
    for query, digest, expect_read in (
        (question, "changed", True),
        ("Как найти большие письма в Thunderbird?", OUTLOOK_LARGE_MAIL_SHA256, False),
        ("Как найти большие письма, и как настроить принтер?", OUTLOOK_LARGE_MAIL_SHA256, False),
    ):
        doc.document["source"]["sha256"] = digest
        knowledge = Sources([doc])
        result = run_agent(query, {}, {}, Agent(no_source), knowledge, settings())
        assert result.blocks == [("Уточните программу.", None)]
        assert bool(knowledge.reads) is expect_read


def test_failed_knowledge_search_cannot_be_reported_as_missing_instruction():
    class UnavailableKnowledge(Sources):
        def search_candidates(self, query):
            raise KnowledgeUnavailable("ReadTimeout")

    no_knowledge = {"action": "reply", "kind": "no_knowledge", "blocks": [{"text": "Инструкции нет."}]}
    turn = run_agent("печать", {}, {}, Agent(search(), no_knowledge), UnavailableKnowledge(),
                     settings(talk_agent_max_calls=2))
    assert turn.blocks == [(UNAVAILABLE, None)]


def test_initial_search_runs_before_model_and_supplies_candidates_for_reading():
    doc = approved()
    knowledge = Sources([doc])
    knowledge.prefetch_candidates = knowledge.search_candidates
    question = "Как настроить печать PDF в Acrobat?"
    agent = Agent(read(doc), answer(doc))
    turn = run_agent(question, {}, {}, agent, knowledge, settings())
    assert turn.action == "AGENT_REPLY"
    first = agent.payloads[0]["tool_results"][0]
    assert first["initial"] and first["query"] == question
    assert knowledge.searches == [question]


def test_found_document_requires_evidence_even_if_model_labels_technical_reply_as_conversation():
    doc = approved()
    knowledge = Sources([doc])
    knowledge.prefetch_candidates = knowledge.search_candidates
    agent = Agent({"action": "reply", "blocks": [{"text": "Используйте прежний добавочный номер."}]},
                  read(doc), answer(doc))
    turn = run_agent("Какой добавочный у администрации?", {}, {}, agent, knowledge, settings())
    assert agent.payloads[1]["tool_results"][-1]["error"] == "UseInstructionWithReferencesOrAskClarification"
    assert turn.agent_documents == [doc] and all("прежний" not in text for text, _ in turn.blocks)


def test_empty_initial_search_requires_reformulating_before_admitting_no_instruction():
    class EmptySources(Sources):
        def search_candidates(self, query):
            self.searches.append(query)
            return {"candidates": [], "search_degraded": False}

        prefetch_candidates = search_candidates

    knowledge = EmptySources()
    missing = {"action": "reply", "kind": "no_knowledge", "blocks": [{"text": "Инструкция не найдена."}]}
    agent = Agent(missing, search(), missing)
    result = run_agent("Как исправить печать PDF в Acrobat?", {}, {}, agent, knowledge, settings())
    assert result.blocks == [("Инструкция не найдена.", None)]
    assert len(knowledge.searches) == 2
    assert agent.payloads[1]["tool_results"][-1]["error"] == "TrySearchByDifferentWordsOrActionsBeforeNoKnowledge"


def test_scanner_requires_one_simple_clarification_even_if_model_is_down():
    knowledge = Sources([approved()])
    knowledge.prefetch_candidates = knowledge.search_candidates
    agent = Agent()
    result = run_agent("Как настроить сканер?", {}, {}, agent, knowledge, settings())
    assert knowledge.searches == ["Как настроить сканер?"]
    assert not agent.payloads
    assert result.blocks == [("Какой у вас сканер — ручной для штрихкодов или для бумажных документов?", None)]
    assert run_agent("для штрихкодов", result.flow, {}, agent, knowledge, settings()).flow[
        "clarification_keys"][-1] == "SCANNER_GOAL"


def test_verified_printer_procedures_follow_setup_changes_without_model():
    docs = {}
    for name, (file_id, digest, _) in _PRINT_SCHEMES.items():
        doc = approved(1, file_id=file_id)
        doc.document["source"]["sha256"] = digest
        doc.document["content"] = "\n\n".join(f"{i}. Настройка номер {i} для {name}." for i in range(1, 5))
        docs[name] = doc
    knowledge = Sources(list(docs.values()))
    knowledge.prefetch_candidates = knowledge.search_candidates
    flow = {}
    for name, question in [
        ("urovo", "Как настроить печать с ТСД на один принтер Urovo по Bluetooth?"),
        ("mixed", "А теперь мне нужно подключить один принтер по блютуз, а другой по Wi-Fi"),
        ("two", "А теперь надо подключить 2 принтера по Wi-Fi к ТСД"),
        ("tsc", "Как подключить к ТСД TSC по Wi-Fi?"),
    ]:
        turn = run_agent(question, flow, {}, Agent(), knowledge, settings())
        flow = turn.flow
        assert knowledge.searches[-1] == (_PRINT_SCHEMES[name][2] if name in {"mixed", "two"} else question)
        assert turn.agent_documents == [docs[name]] and flow["source"] == docs[name].document["source"]
        before, photo, after = turn.blocks
        assert all(step in before[0] for step in docs[name].document["content"].split("\n\n")[:3])
        assert photo[1] == docs[name].illustrations[1]
        assert after[0] == docs[name].document["content"].split("\n\n")[3]


@pytest.mark.parametrize("question,digest,expect_read", [
    ("Как настроить печать с ТСД на один принтер Urovo по Bluetooth?", "changed", True),
    ("Как настроить печать с ТСД на один принтер Urovo по Bluetooth, если печатает три этикетки?", None, False),
    ("А теперь один принтер по Bluetooth, другой по Wi-Fi", None, False),
])
def test_fast_printer_path_requires_reviewed_revision_and_exact_setup(question, digest, expect_read):
    file_id, current, _ = _PRINT_SCHEMES["urovo"]
    doc = approved(1, file_id=file_id)
    doc.document["source"]["sha256"] = digest or current
    knowledge = Sources([doc])
    knowledge.prefetch_candidates = knowledge.search_candidates
    agent = Agent({"action": "reply", "kind": "clarification", "blocks": [{"text": "Какая у вас модель?"}]})
    turn = run_agent(question, {}, {}, agent, knowledge, settings())
    assert not turn.agent_documents and bool(knowledge.reads) is expect_read


def test_agent_rejects_real_quote_from_source_with_wrong_connection():
    doc = approved()
    doc.document["content"] += "\nПодключение по Wi-Fi."
    knowledge = Sources([doc])
    knowledge.prefetch_candidates = knowledge.search_candidates
    clarify = {"action": "reply", "kind": "clarification", "blocks": [{"text": "Уточните модель принтера."}]}
    agent = Agent(read(doc), answer(doc), clarify)
    result = run_agent("Как подключить принтер по Bluetooth?", {}, {}, agent, knowledge, settings())
    assert agent.payloads[-1]["tool_results"][-1]["error"] == "SourceConflictsWithCurrentRequest"
    assert result.blocks == [("Уточните модель принтера.", None)]


def test_model_retries_transient_gateway_error_and_respects_turn_budget():
    from app.services.chat.turn_budget import turn_budget

    attempts = []

    def handler(request):
        attempts.append(request.extensions["timeout"]["read"])
        if len(attempts) == 1:
            return httpx.Response(503)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(search())}}]})

    model = OpenWebUILLM(settings(openwebui_api_key="test-key"), httpx.MockTransport(handler))
    with turn_budget(5):
        assert model.agent_action({"message": "печать"}).action == "search_knowledge"
    assert len(attempts) == 2 and 0 < attempts[1] < attempts[0] <= 5


@pytest.mark.parametrize("invalid,error", [
    ({"action": "read_instruction", "file_id": "outside"}, "SearchBeforeReadingUnknownSource"),
    ({"action": "reply", "kind": "instruction", "blocks": [{"text": "Нажмите кнопку"}]}, "ReadFullInstructionBeforeReply"),
    ({"action": "reply", "kind": "resolved", "blocks": [{"text": "Решено"}]}, "ResolutionRequiresExplicitUserFeedback"),
])
def test_invalid_actions_are_returned_to_agent_without_side_effects(invalid, error):
    agent = Agent(invalid, {"action": "reply", "kind": "clarification", "blocks": [{"text": "Что нужно настроить?"}]})
    result = run_agent("готово", {}, {}, agent, Sources(), settings())
    assert agent.payloads[-1]["tool_results"][-1]["error"] == error
    assert result.blocks == [("Что нужно настроить?", None)]


@pytest.mark.parametrize("fault,error", [("image", "UnknownOriginalImage"), ("quote", "QuoteNotInOriginal")])
def test_fabricated_image_or_quote_is_not_delivered(fault, error):
    doc = approved()
    bad = answer(doc)
    if fault == "image":
        bad["blocks"][0]["images"][0]["number"] = 999
    else:
        bad["blocks"][0]["references"][0]["quote"] = "Несуществующие сведения в инструкции"
    agent = Agent(search(), read(doc), bad, answer(doc))
    result = run_agent("печать", {}, {}, agent, Sources([doc]), settings())
    assert agent.payloads[-1]["tool_results"][-1]["error"] == error
    assert result.blocks[1][1] == doc.illustrations[1]


def test_entire_document_must_be_read_before_answer():
    doc = approved()
    doc.document["content"] += "x" * PAGE_CHARS
    agent = Agent(search(), read(doc), answer(doc), read(doc, PAGE_CHARS), answer(doc))
    result = run_agent("печать", {}, {}, agent, Sources([doc]), settings())
    assert agent.payloads[3]["tool_results"][-1]["error"] == "ReadFullInstructionBeforeReply"
    assert result.action == "AGENT_REPLY"
    pages = [row["result"]["content"] for row in agent.payloads[-1]["tool_results"] if "result" in row
             and row["tool"] == "read_instruction"]
    assert "".join(pages) == doc.document["content"]


def test_revoked_source_and_exhaustion_preserve_progress():
    doc = approved()
    knowledge = Sources([doc])
    knowledge.current = False
    flow = {"agent": {"memory": {"current_step": "1"}, "sources": [doc.document["source"]]}}
    result = run_agent("продолжить", flow, {}, Agent(read(doc), answer(doc)), knowledge,
                       settings(talk_agent_max_calls=2))
    assert result.action == "REPLY" and result.flow == flow
    assert all(image is None for _, image in result.blocks)


def test_question_mid_instruction_and_social_reply_keep_the_next_step():
    doc = approved()
    knowledge = Sources([doc])
    first = run_agent("печать", {}, {}, Agent(search(), read(doc), answer(doc)), knowledge, settings())
    clarification = answer(doc, text="Панель показана на этой фотографии.")
    clarification.pop("memory")
    explained = run_agent("где эта панель?", first.flow, {}, Agent(read(doc), clarification), knowledge, settings())
    assert explained.flow["agent"]["memory"] == first.flow["agent"]["memory"]
    social = run_agent("спасибо", explained.flow, {}, Agent({"action": "reply", "blocks": [{"text": "Пожалуйста."}]}),
                       knowledge, settings())
    assert social.flow == explained.flow


def test_legacy_progress_migrates_without_reentering_confirmation():
    doc = approved()
    old = {"source": doc.document["source"], "shown": [1, 2], "next_offset": 2, "query": "Настроить печать"}
    model = Agent({"action": "reply", "blocks": [{"text": "Пожалуйста."}]})
    result = run_agent("спасибо", old, {}, model, Sources([doc]), settings())
    assert result.flow["agent"]["sources"] == [doc.document["source"]]
    assert result.flow["agent"]["memory"]["next_step"] == "2"
    assert "understanding_confirmation" not in result.flow


def test_cancel_letter_keeps_agent_memory_even_before_a_source_was_found():
    saved = {"agent": {"memory": {"goal": "Печать", "expected_reply": "Модель принтера"}, "sources": []}}
    result = _resume_turn({"escalation": {"resume": {"flow": saved}}}, {}, Agent(), Sources())
    assert result.action == "RESUME" and result.flow == saved


def test_agent_can_prepare_letter_after_user_accepts_its_offer():
    model = Agent({"action": "request_support", "consent_evidence": "да"})
    context = {"short_term": {"messages": [{"role": "assistant", "content": "Составить письмо в поддержку?"}]}}
    result = run_agent("да", {}, context, model, Sources(), settings())
    assert result.action == "FORM_REQUEST"


def test_agent_cannot_open_letter_on_unrelated_yes():
    model = Agent({"action": "request_support", "consent_evidence": "да"},
                  {"action": "reply", "blocks": [{"text": "Продолжаем."}]})
    context = {"short_term": {"messages": [{"role": "assistant", "content": "Принтер включён?"}]}}
    result = run_agent("да", {}, context, model, Sources(), settings())
    assert result.action == "AGENT_REPLY"
    assert model.payloads[-1]["tool_results"][-1]["error"] == "SupportRequestNeedsUserConsent"


def test_agent_can_start_an_independent_task_without_reusing_old_source(talk):
    doc = approved()
    talk.incoming("Настроить печать")
    talk.process(Sources([doc]), Agent(search(), read(doc), answer(doc)))
    talk.incoming("Почта перестала открываться", 2)
    talk.process(Sources([doc]), Agent({"action": "reply", "kind": "clarification", "new_task": True,
                                       "memory": {"goal": "Открыть почту", "expected_reply": "Название программы"},
                                       "blocks": [{"text": "В какой программе открываете почту?"}]}))
    result = talk.snapshot()
    assert result["flow"]["agent"]["sources"] == []
    assert "source" not in result["flow"]
    assert result["attempts"][0].status == "UNASSESSED"
    assert result["incident"].state == "DIAGNOSING"


@pytest.mark.parametrize("rating,helped", [("Ответ понравился", True), ("Не помогло", False)])
def test_talk_learns_only_from_delivered_answers_for_the_same_question_and_version(talk, rating, helped):
    doc = approved()
    source = doc.document["source"]
    question = "Как настроить Urovo по Bluetooth?"
    knowledge = Sources([doc])
    talk.incoming(question)
    assert talk.process(knowledge, Agent(search(), read(doc), answer(doc)))
    with talk.env.factory() as session, session.begin():
        for delivery in session.scalars(select(Delivery)):
            delivery.status = "SENT"
            account_id = delivery.account_id
    talk.incoming(rating, 2)
    assert talk.process(knowledge, Agent())  # Explicit ratings are handled without the model.
    with talk.env.factory() as session:
        assert session.scalar(select(Feedback)).helped is helped
        assert learned_feedback(session, account_id, "Как настроить Urovo по Wi-Fi?") == {}
        assert learned_feedback(session, "other-account", question) == {}
        votes = learned_feedback(session, account_id, question)
    assert votes == {source_key(source): {"helpful": int(helped), "unhelpful": int(not helped)}}
    other = {**source, "file_id": "alternative"}
    candidates = [other, source] if helped else [source, other]
    ranked = reorder_candidates({"candidates": candidates}, votes)["candidates"]
    assert ranked[0 if helped else 1]["file_id"] == source["file_id"]
    changed = {**source, "sha256": "changed"}
    assert reorder_candidates({"candidates": [changed]}, votes)["candidates"] == [changed]


def test_search_candidates_does_not_read_or_call_a_selector():
    knowledge, state = provider()
    knowledge.selector = lambda _: pytest.fail("Search must not ask a second model to select")
    result = knowledge.search_candidates("Acrobat PDF")
    assert result["candidates"] and not result["search_degraded"]
    assert not any("/files/" in str(request.url) for request in state["calls"])


def test_candidate_search_reports_partial_failure_and_filters_foreign_ids():
    knowledge, _ = provider(retrieval_status=503)
    assert knowledge.search_candidates("Acrobat")["search_degraded"]
    knowledge, _ = provider(chunks=[("outside", "Acrobat")])
    assert all(item["file_id"] != "outside" for item in knowledge.search_candidates("Acrobat")["candidates"])


def test_agent_wire_is_text_only_and_disables_provider_tools():
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append(body)
        assert request.extensions["timeout"]["read"] == 180
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(search())}}]})

    model = OpenWebUILLM(settings(openwebui_api_key="test-key"), httpx.MockTransport(handler))
    assert model.agent_action({"message": "печать"}).action == "search_knowledge"
    assert seen[0]["tools"] == [] and seen[0]["tool_ids"] == []
    assert all(isinstance(message["content"], str) for message in seen[0]["messages"])
    assert seen[0]["response_format"]["json_schema"]["schema"] == AgentAction.model_json_schema()


def test_combined_success_and_screen_question_rates_previous_answer_and_asks_only_device(talk):
    doc = approved()
    knowledge = Sources([doc])
    talk.incoming("Как настроить печать?")
    assert talk.process(knowledge, Agent(search(), read(doc), answer(doc)))
    knowledge.prefetch_candidates = knowledge.search_candidates
    talk.incoming("Спасибо, помогло.\nТеперь смотри у меня гаснет экран через какое-то время, что делать?", 2)
    agent = Agent()
    assert talk.process(knowledge, agent)
    result = talk.snapshot()
    assert not agent.payloads
    assert result["attempts"][0].status == "SUCCEEDED"
    assert knowledge.searches[-1].startswith("у меня гаснет экран")
    assert result["flow"]["agent"]["memory"]["expected_reply"] == "Тип устройства"
    assert result["flow"]["agent"]["sources"] == []
    assert "на компьютере, на сканере или на телефоне" in assistants(result)[-1][0]
    with talk.env.factory() as session:
        assert session.scalar(select(Feedback)).helped is True

    windows = approved(file_id=WINDOWS_SCREEN_FILE)
    windows.document["source"]["sha256"] = WINDOWS_SCREEN_SHA256
    windows.document["content"] = ("\n".join(f"{number}. Диагностический шаг" for number in range(1, 5)) +
                                   "\nТолько для ИТ\n5. Не выдавать администратора")
    talk.incoming("На компьютере", 3)
    assert talk.process(Sources([windows]), Agent())
    blocks = assistants(talk.snapshot())[-5:]
    assert "1. Диагностический шаг\n2. Диагностический шаг" in blocks[0][0] and blocks[0][1] is None
    assert "3. Диагностический шаг" in blocks[1][0]
    assert blocks[2][1].data == windows.illustrations[1].data
    assert blocks[3][0] == "4. Диагностический шаг"
    assert blocks[4][0] == AGENT_OUTCOME_QUESTION


@pytest.mark.parametrize("message,helped", [("Ага", True), ("Неа", False),
                                            ("Да, помог. Теперь гаснет экран, что делать?", True),
                                            ("Нет, не получилось. Теперь гаснет экран, что делать?", False)])
def test_feedback_after_delivered_question_and_optional_new_task(talk, message, helped):
    doc = approved()
    first = answer(doc)
    first["memory"]["next_step"] = ""
    talk.incoming("Как настроить печать?")
    assert talk.process(Sources([doc]), Agent(search(), read(doc), first))
    assert assistants(talk.snapshot())[-1][0] == AGENT_OUTCOME_QUESTION
    with talk.env.factory() as session, session.begin():
        for delivery in session.scalars(select(Delivery)):
            delivery.status = "SENT"
    talk.incoming(message, 2)
    assert talk.process(Sources([doc]), Agent())
    with talk.env.factory() as session:
        assert session.scalar(select(Feedback)).helped is helped
    if "экран" in message:
        assert "на сканере или на телефоне" in assistants(talk.snapshot())[-1][0]


def test_feedback_does_not_treat_unsolicited_confirmation_or_symptom_as_rating():
    assert split_agent_feedback("Да") == (None, None)
    assert split_agent_feedback("Нет") == (None, None)
    assert split_agent_feedback("Не работает почта") == (None, None)
    assert split_agent_feedback("Спасибо, помогла инструкция.") == ("AGENT_RESOLVED", None)
    assert split_agent_feedback("Нет, не получилось. Что теперь делать?") == ("AGENT_NEGATIVE", None)


def test_talk_persists_agent_progress_photos_and_explicit_resolution(talk):
    doc = approved()
    knowledge = Sources([doc])
    talk.incoming("Настроить печать")
    assert talk.process(knowledge, Agent(search(), read(doc), answer(doc)))
    initial = talk.snapshot()
    assert initial["flow"]["agent"]["memory"]["current_step"] == "1"
    assert len(initial["attempts"]) == 1
    assert assistants(initial)[1][1].data == doc.illustrations[1].data
    talk.incoming("продолжить", 2)
    assert talk.process(knowledge, Agent(read(doc), answer(doc, 2)))
    assert talk.snapshot()["flow"]["agent"]["memory"]["current_step"] == "2"
    assert len(talk.snapshot()["attempts"]) == 1
    talk.incoming("Все работает, спасибо", 3)
    assert talk.process(knowledge, Agent({"action": "reply", "kind": "resolved",
                                         "blocks": [{"text": "Отлично, рад помочь."}]}))
    final = talk.snapshot()
    assert final["incident"].state == "RESOLVED"
    assert final["flow"] == {} and final["attempts"][0].status == "SUCCEEDED"
    with talk.env.factory() as session:
        assert session.scalar(select(ResolutionMemory)).facts["problem_type"] == "Настроить печать"


def test_multisource_reply_keeps_each_outbox_item_bound_to_its_original(talk):
    first, second = approved(), approved(file_id="00000000-0000-4000-8000-000000000055")
    response = answer(first)
    response["blocks"].extend(answer(second)["blocks"])
    talk.incoming("Настроить оба принтера")
    assert talk.process(Sources([first, second]), Agent(search(), read(first), read(second), response))
    with talk.env.factory() as session:
        deliveries = list(session.scalars(select(Delivery).order_by(Delivery.created_at)))
        assert len(deliveries) == 4
        assert [item.provenance for item in deliveries] == [first.document["source"]] * 2 + [second.document["source"]] * 2
    assert len(talk.snapshot()["flow"]["agent"]["sources"]) == 2
    assert not talk.process(Sources([first, second]), Agent())  # Inbox retry cannot duplicate the reply.
