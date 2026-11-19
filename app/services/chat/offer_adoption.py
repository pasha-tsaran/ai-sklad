"""Bind an actual employee choice to verified alternative intent, retaining physical facts."""

import copy
import re

from app.schemas.contracts import SupportTaskGoal
from app.services.chat.task_state import remember_task_message
from app.services.llm.offer_adoption import InstructionOfferAdoption
from app.services.rag.query import _CONNECTIONS, _connections, normalized_query
from app.services.rag.talk import KnowledgeUnavailable, knowledge_stage


_EQUIPMENT_COUNT = re.compile(
    r"\b(?:\d+|один|одного|одной|одна|одно|одном|одним|два|две|двух|двум|три|тр[её]х|четыре|пять|несколько)"
    r"\s+(?:(?:ручн\w*|мобильн\w*|стационарн\w*|рабоч\w*|этикеточн\w*|термо\w*)\s+){0,2}"
    r"(?:принтер\w*|терминал\w*|тсд|сканер\w*|устройств\w*|компьютер\w*|пк|экземпляр\w*)\b", re.I,
)
_PHYSICAL_ATTRIBUTE = re.compile(
    r"^(?:type|тип|os|app)$|(?:^|[\s_-])(?:count|quantity|model|application|platform|equipment|"
    r"device|hardware|колич\w*|числ\w*|модел\w*|прилож\w*|оборуд\w*|платформ\w*)(?:$|[\s_-])", re.I,
)
_FUTURE_REQUEST = re.compile(
    r"^\s*(?:мне\s+)?(?:нужно|надо|хочу|требуется|планирую|собираюсь|помогите)\b"
    r"[^.!?\n]{0,150}\b(?:подключить|соединить|настроить|организовать|использовать|печатать|работать)\b", re.I,
)
_OBSERVED_STATE = re.compile(r"\b(?:уже|сейчас|подключ[её]н\w*|соедин[её]н\w*|работает|работало|"
                              r"подключил\w*|соединил\w*|работал\w*|использую|используется|"
                              r"имеет|имеют|имеется|имею|поддерживает|фактически|наблюдаю)\b", re.I)


def _explicit_changed_transport_request(fact, previous, replacements):
    transport = normalized_query(fact["value"])
    evidence = fact["evidence"]
    return (transport in _CONNECTIONS and transport in _connections(evidence)
            and not re.search(r"[,;:\n]|[.!?]\s*\S", evidence)
            and bool(_FUTURE_REQUEST.search(evidence)) and not _OBSERVED_STATE.search(evidence)
            and any(transport in _connections(previous["goals"][goal_id]["text"])
                    and _connections(text) and transport not in _connections(text)
                    for goal_id, text in replacements.items()))


def _explicit_observed_transport(fact):
    transport = normalized_query(fact["value"])
    evidence = fact["evidence"]
    return (transport in _CONNECTIONS and transport in _connections(evidence)
            and bool(_OBSERVED_STATE.search(evidence)))


def adopt_instruction_offer(message, evidence, previous, source, alternative_goals, llm):
    if (not isinstance(alternative_goals, dict) or not alternative_goals
            or not set(alternative_goals) <= previous.get("goals", {}).keys()
            or not evidence or evidence not in message or len(evidence) > 300):
        raise KnowledgeUnavailable("InvalidInstructionOfferIntent", "offer_adoption")
    for goal_id, text in alternative_goals.items():
        SupportTaskGoal.model_validate({"goal_id": goal_id, "text": text, "evidence": evidence})
    facts = previous.get("facts", [])
    eligible = {f"f{index + 1}": copy.deepcopy(fact) for index, fact in enumerate(facts)
                if fact["key"] in {"connection", "condition"}
                and (fact["key"] == "connection" or not _PHYSICAL_ATTRIBUTE.search(fact.get("attribute", "")))
                and not _EQUIPMENT_COUNT.search(fact["value"])}
    retired = set()
    if eligible:
        classify = getattr(llm, "adopt_instruction_offer", None)
        if not callable(classify):
            raise KnowledgeUnavailable("OfferAdoptionUnavailable", "offer_adoption")
        with knowledge_stage("offer_adoption"):
            choice = InstructionOfferAdoption.model_validate(classify({
                "actual_user_reply": message, "consent_evidence": evidence, "selected_source": source,
                "accepted_goals": {goal_id: {"previous": previous["goals"][goal_id]["text"], "selected": text}
                                   for goal_id, text in alternative_goals.items()},
                "preserved_goals": {key: goal["text"] for key, goal in previous["goals"].items()
                                    if key not in alternative_goals},
                "candidate_facts": eligible,
            }))
        retired = set(choice.superseded_requested_fact_ids)
        if (set(choice.fact_roles) != set(eligible) or len(retired) != len(choice.superseded_requested_fact_ids)
                or any(key not in eligible or choice.fact_roles[key] != "REQUESTED" for key in retired)):
            raise KnowledgeUnavailable("InvalidInstructionOfferIntent", "offer_adoption")
        retired.update(key for key, fact in eligible.items()
                       if _explicit_changed_transport_request(fact, previous, alternative_goals))
        retired.difference_update(key for key, fact in eligible.items() if _explicit_observed_transport(fact))
    state = remember_task_message(message, previous)
    state["facts"] = [fact for index, fact in enumerate(state["facts"]) if f"f{index + 1}" not in retired]
    for goal_id, text in alternative_goals.items():
        state["goals"][goal_id].update(text=text, evidence=evidence)
        state["goals"][goal_id].pop("search_queries", None)
    state.update(new_task=False, question="", question_field=None)
    return state, sorted(retired)
