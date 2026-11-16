"""User-evidenced task facts and ordered goals before document selection."""

import copy
import re

from app.schemas.contracts import SupportTaskFact, SupportTaskGoal, SupportTaskUnderstanding


_BOTH = {
    "оба", "обе", "оба варианта", "и то и другое", "то и другое", "и то и другое надо настроить",
    "все перечисленное", "все эти действия",
}
_MULTIPLE_FACTS = {"completed"}


def _normalized(value):
    return " ".join(re.sub(r"[^\w\s]", " ", value.casefold().replace("ё", "е")).split())


def _previous_state(previous):
    if not previous:
        return {"facts": [], "goals": {}, "messages": [], "question": "", "question_field": None,
                "new_task": False}
    if not isinstance(previous, dict):
        raise ValueError("InvalidTaskState")
    raw_facts = previous.get("facts", [])
    raw_goals = previous.get("goals", {})
    if not isinstance(raw_facts, list) or not isinstance(raw_goals, dict):
        raise ValueError("InvalidTaskState")
    facts = [SupportTaskFact.model_validate(item).model_dump() for item in raw_facts]
    goals = {}
    for goal_id, item in raw_goals.items():
        if not isinstance(item, dict):
            raise ValueError("InvalidTaskState")
        goal = SupportTaskGoal.model_validate({**item, "goal_id": goal_id})
        goals[goal_id] = {"text": goal.text, "evidence": goal.evidence}
    messages = previous.get("messages", [])
    if (len(facts) > 12 or len(goals) > 8 or not isinstance(messages, list)
            or any(not isinstance(message, str) for message in messages)):
        raise ValueError("InvalidTaskState")
    return {"facts": facts, "goals": goals, "messages": messages,
            "question": previous.get("question", ""), "question_field": previous.get("question_field"),
            "new_task": False}


def _history(messages, message):
    retained = [*messages[-11:], message]
    while len(retained) > 1 and sum(map(len, retained)) > 6000:
        retained.pop(0)
    return retained


def _evidenced(evidence, authority):
    return bool(evidence) and any(evidence in message for message in authority)


def update_task_state(message, previous, previous_question, llm):
    """Merge validated deltas; documents and assistant questions cannot supply facts."""
    if not isinstance(message, str) or not message.strip() or len(message) > 6000:
        raise ValueError("InvalidTaskMessage")
    prior = _previous_state(previous)
    if len(prior["goals"]) >= 2 and _normalized(message) in _BOTH:
        return {**copy.deepcopy(prior), "messages": _history(prior["messages"], message),
                "question": "", "question_field": None, "new_task": False}
    raw = llm.understand_task(message, copy.deepcopy(prior), previous_question)
    update = SupportTaskUnderstanding.model_validate(raw)
    state = _previous_state(None) if update.new_task else copy.deepcopy(prior)
    authority = [message]
    if not update.new_task:
        authority += prior["messages"]
        authority += [fact["evidence"] for fact in prior["facts"]]
        authority += [goal["evidence"] for goal in prior["goals"].values()]
    if any(not _evidenced(item.evidence, authority) for item in [*update.facts, *update.goals]):
        raise ValueError("InvalidTaskEvidence")
    if update.replace_goals and not _evidenced(update.replacement_evidence, [message]):
        raise ValueError("InvalidTaskReplacement")
    if not update.replace_goals and update.replacement_evidence:
        raise ValueError("UnexpectedTaskReplacement")
    for fact in update.facts:
        if fact.key not in _MULTIPLE_FACTS:
            state["facts"] = [item for item in state["facts"] if item["key"] != fact.key]
        state["facts"] = [item for item in state["facts"]
                          if (item["key"], _normalized(item["value"])) != (fact.key, _normalized(fact.value))]
        state["facts"].append(fact.model_dump())
    existing = state["goals"]
    next_id = max((int(key[1:]) for key in existing), default=0) + 1
    goals = {} if update.replace_goals else copy.deepcopy(existing)
    updated_goal_ids = []
    for goal in update.goals:
        goal_id = goal.goal_id
        if goal_id is not None and goal_id not in existing:
            raise ValueError("UnknownTaskGoal")
        if goal_id is None:
            goal_id = next((key for key, item in existing.items()
                            if _normalized(item["text"]) == _normalized(goal.text)), None)
        if goal_id is None:
            goal_id = f"g{next_id}"
            next_id += 1
        goals[goal_id] = {"text": goal.text, "evidence": goal.evidence}
        updated_goal_ids.append(goal_id)
    if existing.keys() <= set(updated_goal_ids):
        goals = {goal_id: goals[goal_id] for goal_id in updated_goal_ids}
    if len(state["facts"]) > 12 or len(goals) > 8:
        raise ValueError("TaskStateTooLarge")
    state.update({"goals": goals, "messages": _history(state["messages"], message),
                  "question": update.question, "question_field": update.question_field,
                  "new_task": update.new_task})
    return state


def task_query(state):
    """Retrieve the desired work, excluding already completed setup and background apps."""
    task = _previous_state(state)
    parts = [fact["value"] for fact in task["facts"] if fact["key"] in {"equipment_type", "equipment_model"}]
    parts += [goal["text"] for goal in task["goals"].values()]
    return "\n".join(dict.fromkeys(parts))
