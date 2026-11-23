import copy
import json

import httpx
import pytest
from sqlalchemy import select

from app.core.config import Settings
from app.models import Delivery, Feedback, ResolutionMemory
from app.services.chat.agent_dialogue import MODEL_UNAVAILABLE, PAGE_CHARS, run_agent
from app.services.chat.talk_dialogue import UNAVAILABLE, _resume_turn, _text_turn
from app.services.chat.talk_feedback import learned_feedback, reorder_candidates, source_key
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


def test_failed_knowledge_search_cannot_be_reported_as_missing_instruction():
    class UnavailableKnowledge(Sources):
        def search_candidates(self, query):
            raise KnowledgeUnavailable("ReadTimeout")

    no_knowledge = {"action": "reply", "kind": "no_knowledge", "blocks": [{"text": "Инструкции нет."}]}
    turn = run_agent("печать", {}, {}, Agent(search(), no_knowledge), UnavailableKnowledge(),
                     settings(talk_agent_max_calls=2))
    assert turn.blocks == [(UNAVAILABLE, None)]


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
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(search())}}]})

    model = OpenWebUILLM(settings(openwebui_api_key="test-key"), httpx.MockTransport(handler))
    assert model.agent_action({"message": "печать"}).action == "search_knowledge"
    assert seen[0]["tools"] == [] and seen[0]["tool_ids"] == []
    assert all(isinstance(message["content"], str) for message in seen[0]["messages"])
    assert seen[0]["response_format"]["json_schema"]["schema"] == AgentAction.model_json_schema()


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
