"""A verified alternative changes approved intent without manufacturing physical facts."""

import copy
import json
from types import SimpleNamespace

import httpx
import pytest

from app.core.config import Settings
from app.services.chat.offer_adoption import adopt_instruction_offer
from app.services.chat.talk_dialogue import CHANGED, _instruction_offer_turn, _retrieve_support_turn
from app.services.chat.task_state import update_task_state
from app.services.llm.openwebui import OpenWebUILLM
from app.services.rag.talk import InstructionDecision, KnowledgeUnavailable
from tests.test_task_state import UnderstandingModel, fact, goal
from tests.test_verified_support_followup import DiagnosticKnowledge, GuideModel, document


BEFORE = "Подключить ТСД к одному принтеру по Bluetooth"
AFTER = "Подключить ТСД к одному принтеру по Wi-Fi"


def prior(*, observed=False):
    observation = "Принтер уже подключён по Bluetooth" if observed else "Нужно подключить принтер по Bluetooth"
    facts = [fact("connection", "Bluetooth", observation), fact("equipment_type", "принтер", observation),
             {**fact("condition", "один принтер", observation), "attribute": "count"}]
    state = update_task_state(observation, None, "", UnderstandingModel({
        "facts": facts, "goals": [{**goal(BEFORE, observation), "kind": "ACTION", "search_queries": ["старый запрос"]},
                                  {**goal("Увеличить громкость", observation), "kind": "ACTION",
                                   "search_queries": ["громкость"]}],
    }))
    return state


class Classifier:
    def __init__(self, *, role="REQUESTED", retired=None):
        self.calls = []
        self.role = role
        self.retired = retired

    def adopt_instruction_offer(self, payload):
        self.calls.append(copy.deepcopy(payload))
        ids = list(payload["candidate_facts"])
        return {"fact_roles": {reference: self.role for reference in ids},
                "superseded_requested_fact_ids": ids if self.retired is None else self.retired}


@pytest.mark.parametrize("message", ["1", "да", "Выбираю Wi-Fi, Bluetooth больше не нужен"])
def test_accepted_alternative_retargets_only_the_selected_goal_and_retires_requested_connection(message):
    state = prior()
    model = Classifier()
    updated, retired = adopt_instruction_offer(message, message, state, {"filename": "Подключение по Wi-Fi"}, {"g1": AFTER}, model)
    assert updated["goals"]["g1"] == {"text": AFTER, "evidence": message, "kind": "ACTION"}
    assert updated["goals"]["g2"] == state["goals"]["g2"] and list(updated["goals"]) == ["g1", "g2"]
    assert updated["facts"] == state["facts"][1:]
    assert updated["messages"][-1] == message and updated["new_task"] is False
    assert list(model.calls[0]["candidate_facts"]) == ["f1"]
    assert retired == ["f1"]
    assert model.calls[0]["preserved_goals"] == {"g2": "Увеличить громкость"}
    assert state["facts"][0]["value"] == "Bluetooth"


def test_an_observed_connection_is_preserved_and_cannot_be_retired_by_a_choice():
    state = prior(observed=True)
    updated, retired = adopt_instruction_offer("1", "1", state, {}, {"g1": AFTER}, Classifier(role="OBSERVED", retired=[]))
    assert updated["facts"] == state["facts"]
    assert retired == []
    with pytest.raises(KnowledgeUnavailable, match="InvalidInstructionOfferIntent"):
        adopt_instruction_offer("1", "1", state, {}, {"g1": AFTER}, Classifier(role="OBSERVED", retired=["f1"]))


@pytest.mark.parametrize("changes", [
    {"fact_roles": {"other": "REQUESTED"}, "superseded_requested_fact_ids": ["other"]},
    {"fact_roles": {"f1": "REQUESTED"}, "superseded_requested_fact_ids": ["f1", "f1"]},
    {"fact_roles": {"f1": "REQUESTED"}, "superseded_requested_fact_ids": ["f2"]},
])
def test_adoption_cannot_change_an_unknown_or_protected_fact(changes):
    model = Classifier()
    model.adopt_instruction_offer = lambda payload: changes
    with pytest.raises(KnowledgeUnavailable, match="InvalidInstructionOfferIntent"):
        adopt_instruction_offer("1", "1", prior(), {}, {"g1": AFTER}, model)


@pytest.mark.parametrize("answer", ["1", "да", "Выбираю Wi-Fi, Bluetooth больше не нужен"])
def test_controller_accepts_a_clearly_shown_alternative_and_requires_fresh_scope_consent(answer):
    approved = document("# Подключение по Wi-Fi\n\n1. Выполните показанные настройки сети.")

    class Knowledge(DiagnosticKnowledge):
        def search_with_llm(self, query, **options):
            self.diagnostics.append((query, copy.deepcopy(options)))
            assert options["context"]["requested_goals"]["g1"] == AFTER
            assert AFTER in query and "Bluetooth" not in query and options["fallback_query"] == query
            accepted = options["context"]["accepted_instruction_offer"]
            assert accepted["accepted_goals"] == {"g1": AFTER} and accepted["choice_evidence"] == answer
            assert not any(item["key"] == "connection" for item in options["context"]["task_state"]["facts"])
            return InstructionDecision("MATCH", approved=approved, understanding=query)

    source = approved.document["source"]
    flow = {"task_state": prior(), "task_query": BEFORE,
            "instruction_offer": {"problem": BEFORE, "choices": [source], "alternative_goals": {source["file_id"]: {"g1": AFTER}}}}
    knowledge = Knowledge(approved)
    model = GuideModel({})
    model.interpret_instruction_offer = lambda message, choices: {"result": "ACCEPT", "choice": 1, "evidence": message}
    classifier = Classifier()
    model.adopt_instruction_offer = classifier.adopt_instruction_offer
    turn = _instruction_offer_turn(answer, flow, {}, model, knowledge, knowledge.settings)
    confirmation = turn.flow["understanding_confirmation"]
    assert confirmation["guide_flow"]["task_state"]["goals"]["g1"]["text"] == AFTER
    assert confirmation["guide_flow"]["task_state"]["goals"]["g2"] == flow["task_state"]["goals"]["g2"]
    assert confirmation["guide_flow"]["accepted_instruction_offer"]["superseded_requested_fact_ids"] == ["f1"]
    assert len(classifier.calls) == 1 and not any(image for _, image in turn.blocks)
    assert len(model.contexts) == int("Wi-Fi" in answer)


def test_verified_alternative_intent_is_visible_before_the_employee_selects_it():
    approved = document("# Подключение по Wi-Fi\n\n1. Выполните показанные настройки сети.")
    source = approved.document["source"]

    class RelatedKnowledge(DiagnosticKnowledge):
        def search_with_llm(self, query, **options):
            return SimpleNamespace(result="RELATED", alternatives=(approved,), proposed_goal_order=(),
                                   alternative_goals={source["file_id"]: {"g1": AFTER}})

    state = prior()
    turn = _retrieve_support_turn(BEFORE, {"task_state": state, "task_query": BEFORE}, {}, GuideModel(), RelatedKnowledge(approved))
    assert AFTER in turn.blocks[0][0]
    assert turn.flow["instruction_offer"]["alternative_goals"] == {source["file_id"]: {"g1": AFTER}}
    assert not any(photo for _, photo in turn.blocks)


def test_closed_model_schema_limits_roles_and_retired_ids_to_eligible_original_user_facts():
    requests = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps({
            "fact_roles": {"f1": "REQUESTED"}, "superseded_requested_fact_ids": ["f1"],
        })}, "finish_reason": "stop"}]})

    model = OpenWebUILLM(Settings(_env_file=None, openwebui_base_url="https://model.invalid", openwebui_api_key="test"),
                         transport=httpx.MockTransport(respond))
    updated, _ = adopt_instruction_offer("1", "1", prior(), {}, {"g1": AFTER}, model)
    assert updated["goals"]["g1"]["text"] == AFTER
    schema = requests[0]["response_format"]["json_schema"]["schema"]
    assert schema["properties"]["fact_roles"]["required"] == ["f1"]
    assert schema["properties"]["superseded_requested_fact_ids"]["items"]["enum"] == ["f1"]
    assert requests[0]["params"]["think"] is False and requests[0]["max_tokens"] == 512


def test_requested_connection_type_condition_remains_eligible_without_exposing_hardware_type():
    state = prior()
    state["facts"][0].update(key="condition", attribute="connection_type")
    updated, retired = adopt_instruction_offer("1", "1", state, {}, {"g1": AFTER}, Classifier())
    assert retired == ["f1"] and updated["facts"] == state["facts"][1:]


def test_embedded_equipment_count_cannot_be_removed_with_a_requested_connection():
    state = prior()
    state["facts"][0]["value"] = "два принтера по Bluetooth"
    model = Classifier()
    updated, retired = adopt_instruction_offer("1", "1", state, {}, {"g1": AFTER}, model)
    assert updated["facts"] == state["facts"] and retired == [] and not model.calls


def test_an_explicit_requested_numeric_parameter_is_not_mistaken_for_an_equipment_count():
    state = prior()
    state["facts"][0].update(key="condition", attribute="port", value="9100", evidence="Хочу подключиться через порт 9100")
    updated, retired = adopt_instruction_offer("1", "1", state, {}, {"g1": AFTER}, Classifier())
    assert retired == ["f1"] and updated["facts"] == state["facts"][1:]


def test_changed_alternative_retains_the_working_task_and_discards_only_the_stale_choice():
    approved = document("Исходный вариант подключения.")
    original = copy.deepcopy(approved.document["source"])
    approved.document["source"]["sha256"] = "c" * 64
    state = prior()
    flow = {"task_state": state, "task_query": BEFORE, "opening_query": "Настроить рабочую печать",
            "required_source": original, "instruction_offer": {"problem": BEFORE, "choices": [original]}}
    turn = _instruction_offer_turn("1", flow, {}, GuideModel(), DiagnosticKnowledge(approved), Settings(_env_file=None))
    assert turn.blocks == [(CHANGED, None)]
    assert turn.flow["task_state"] == state and turn.flow["opening_query"] == flow["opening_query"]
    assert "instruction_offer" not in turn.flow and "required_source" not in turn.flow
    assert turn.flow["reselect_source"] is True


@pytest.mark.parametrize("unavailable", [False, True])
def test_a_changed_or_withdrawn_original_keeps_the_selected_user_intent_without_any_technical_answer(unavailable):
    approved = document("Исходное подключение по Wi-Fi.")
    original = copy.deepcopy(approved.document["source"])
    approved.document["source"]["sha256"] = "d" * 64

    class Knowledge(DiagnosticKnowledge):
        def read(self, file_id):
            if unavailable:
                raise KnowledgeUnavailable("SourceNotInApprovedKnowledge")
            return super().read(file_id)

    state = prior()
    flow = {"task_state": state, "task_query": BEFORE, "opening_query": "Настройка рабочей печати",
            "instruction_offer": {"problem": BEFORE, "choices": [original],
                                  "alternative_goals": {original["file_id"]: {"g1": AFTER}}}}
    model = GuideModel()
    model.adopt_instruction_offer = Classifier().adopt_instruction_offer
    turn = _instruction_offer_turn("1", flow, {}, model, Knowledge(approved), Settings(_env_file=None))
    assert turn.blocks == [(CHANGED, None)] and not turn.block_sources
    assert turn.flow["task_state"]["goals"]["g1"]["text"] == AFTER
    assert turn.flow["task_state"]["goals"]["g2"] == state["goals"]["g2"]
    assert turn.flow["task_state"]["facts"] == state["facts"][1:]
    assert turn.flow["task_state"]["messages"][-1] == "1"
    assert turn.flow["opening_query"] == flow["opening_query"] and "instruction_offer" not in turn.flow


def test_clear_future_transport_request_is_superseded_even_if_the_classifier_calls_it_observed():
    state = prior()
    updated, retired = adopt_instruction_offer("1", "1", state, {}, {"g1": AFTER}, Classifier(role="OBSERVED", retired=[]))
    assert retired == ["f1"] and updated["facts"] == state["facts"][1:]


def test_an_explicit_already_connected_transport_is_kept_even_if_the_classifier_calls_it_requested():
    state = prior(observed=True)
    updated, retired = adopt_instruction_offer("1", "1", state, {}, {"g1": AFTER}, Classifier())
    assert retired == [] and updated["facts"] == state["facts"]


def test_a_compound_future_request_does_not_erase_an_already_observed_connection():
    state = prior()
    state["facts"][0]["evidence"] = "Нужно настроить печать. Принтер уже подключил по Bluetooth."
    updated, retired = adopt_instruction_offer("1", "1", state, {}, {"g1": AFTER}, Classifier())
    assert retired == [] and updated["facts"] == state["facts"]


def test_a_substantive_acceptance_reads_new_user_conditions_and_searches_all_sources():
    old = document("# Подключение одного принтера\n\n1. Настройте выбранный принтер.")
    new = document("# Подключение двух принтеров\n\n1. Настройте оба принтера.")
    new.document["source"]["file_id"] = "two-printer-source"
    source = old.document["source"]
    message = "Выбираю 1, но принтеров два"
    target = "Подключить ТСД к двум принтерам по Wi-Fi"

    class Knowledge(DiagnosticKnowledge):
        def __init__(self):
            super().__init__(old)
            self.documents.append(new)

        def search_with_llm(self, query, **options):
            assert "verified_previous_source" not in options["context"]
            assert options["context"]["requested_goals"]["g1"] == target
            assert any(item["key"] == "condition" and item["value"] == "два принтера"
                       for item in options["context"]["task_state"]["facts"])
            return InstructionDecision("MATCH", approved=new, understanding=query)

    state = prior()
    flow = {"task_state": state, "task_query": BEFORE, "instruction_offer": {
        "problem": BEFORE, "choices": [source], "alternative_goals": {source["file_id"]: {"g1": AFTER}},
    }}
    model = GuideModel({"facts": [{**fact("condition", "два принтера", "принтеров два"), "attribute": "count"}],
                        "goals": [goal(target, "принтеров два", "g1")]})
    model.adopt_instruction_offer = Classifier().adopt_instruction_offer
    model.interpret_instruction_offer = lambda text, choices: {"result": "ACCEPT", "choice": 1, "evidence": "Выбираю 1"}
    knowledge = Knowledge()
    turn = _instruction_offer_turn(message, flow, {}, model, knowledge, knowledge.settings)
    guide = turn.flow["understanding_confirmation"]["guide_flow"]
    assert guide["source"] == new.document["source"]
    assert guide["task_state"]["messages"].count(message) == 1
    assert guide["task_state"]["goals"]["g2"] == state["goals"]["g2"]
    assert model.contexts[0]["accepted_instruction_offer"]["choice_evidence"] == "Выбираю 1"
