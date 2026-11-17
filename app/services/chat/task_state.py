"""User-evidenced task facts and ordered goals before document selection."""

import copy
import re

from app.schemas.contracts import SupportTaskFact, SupportTaskGoal, SupportTaskUnderstanding, SupportUnknownAnswer


_BOTH = {
    "оба", "обе", "оба варианта", "и то и другое", "то и другое", "и то и другое надо настроить",
    "все перечисленное", "все эти действия",
}
_MULTIPLE_FACTS = {"completed"}
_UNKNOWN_ANSWERS = {
    "не знаю", "я не знаю", "не знаю что это", "я не знаю что это", "не знаю что это значит",
    "не разбираюсь", "я не разбираюсь", "не разбираюсь в этом", "я в этом не разбираюсь",
    "не могу определить", "я не могу определить", "не могу сказать", "затрудняюсь ответить",
    "без понятия", "понятия не имею", "я понятия не имею", "не уверен", "не уверена",
    "не помню", "я не помню", "не понимаю вопрос", "я не понимаю вопрос",
    "не понимаю что вы спрашиваете", "я не понимаю что вы спрашиваете",
}


def _normalized(value):
    return " ".join(re.sub(r"[^\w\s]", " ", value.casefold().replace("ё", "е")).split())


def is_unknown_answer(message):
    if not isinstance(message, str):
        return False
    text = _normalized(message)
    text = re.sub(r"\s+(?:пожалуйста|подскажите|помоги|помогите)$", "", text)
    return text in _UNKNOWN_ANSWERS


def _previous_state(previous):
    if not previous:
        return {"facts": [], "goals": {}, "messages": [], "question": "", "question_field": None,
                "unknown_answers": [], "new_task": False}
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
    unknown = previous.get("unknown_answers", [])
    if (len(facts) > 12 or len(goals) > 8 or not isinstance(messages, list)
            or any(not isinstance(message, str) for message in messages)
            or not isinstance(unknown, list) or len(unknown) > 8):
        raise ValueError("InvalidTaskState")
    unknown = [SupportUnknownAnswer.model_validate(item).model_dump() for item in unknown]
    return {"facts": facts, "goals": goals, "messages": messages,
            "question": previous.get("question", ""), "question_field": previous.get("question_field"),
            "unknown_answers": unknown, "new_task": False}


def _history(messages, message):
    retained = [*messages[-11:], message]
    while len(retained) > 1 and sum(map(len, retained)) > 6000:
        retained.pop(0)
    return retained


def _evidenced(evidence, authority):
    return bool(evidence) and any(evidence in message for message in authority)


def _unknown_answer(prior, question, answer):
    field = prior["question_field"] if prior["question"] == question else None
    if field is None:
        field = next((item["field"] for item in reversed(prior["unknown_answers"])
                      if _normalized(item["question"]) == _normalized(question)), None)
    unknown = SupportUnknownAnswer(question=question, answer=answer, field=field).model_dump()
    retained = [item for item in prior["unknown_answers"]
                if _normalized(item["question"]) != _normalized(unknown["question"])]
    return [*retained, unknown][-8:]


def remember_task_message(message, previous=None):
    """Retain employee wording after a failed model call without accepting new facts."""
    if not isinstance(message, str) or not message.strip() or len(message) > 6000:
        raise ValueError("InvalidTaskMessage")
    state = _previous_state(previous)
    state["messages"] = _history(state["messages"], message)
    return state


def update_task_state(message, previous, previous_question, llm):
    """Merge validated deltas; documents and assistant questions cannot supply facts."""
    if not isinstance(message, str) or not message.strip() or len(message) > 6000:
        raise ValueError("InvalidTaskMessage")
    prior = _previous_state(previous)
    if is_unknown_answer(message) and isinstance(previous_question, str) and previous_question.strip():
        return {**copy.deepcopy(prior), "messages": _history(prior["messages"], message),
                "question": "", "question_field": None,
                "unknown_answers": _unknown_answer(prior, previous_question, message),
                "new_task": False}
    if len(prior["goals"]) >= 2 and _normalized(message) in _BOTH:
        return {**copy.deepcopy(prior), "messages": _history(prior["messages"], message),
                "question": "", "question_field": None, "new_task": False}
    raw = llm.understand_task(message, copy.deepcopy(prior), previous_question)
    update = SupportTaskUnderstanding.model_validate(raw)
    if update.unknown_answer_evidence and (
        not _evidenced(update.unknown_answer_evidence, [message])
        or not isinstance(previous_question, str) or not previous_question.strip() or update.new_task
    ):
        raise ValueError("InvalidTaskUnknownAnswer")
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
    while len(state["facts"]) > 12:
        completed = next((index for index, item in enumerate(state["facts"]) if item["key"] == "completed"), None)
        if completed is None:
            raise ValueError("TaskStateTooLarge")
        state["facts"].pop(completed)
    if len(goals) > 8:
        raise ValueError("TaskStateTooLarge")
    answered_fields = {fact.key for fact in update.facts if _evidenced(fact.evidence, [message])}
    if any(_evidenced(goal.evidence, [message]) for goal in update.goals):
        answered_fields.add("goal")
    state["unknown_answers"] = [item for item in state["unknown_answers"]
                                if not (item["field"] in answered_fields
                                        and isinstance(previous_question, str)
                                        and _normalized(item["question"]) == _normalized(previous_question))]
    question, question_field = update.question, update.question_field
    if update.unknown_answer_evidence:
        state["unknown_answers"] = _unknown_answer(
            {**prior, "unknown_answers": state["unknown_answers"]},
            previous_question, update.unknown_answer_evidence,
        )
        question, question_field = "", None
    state.update({"goals": goals, "messages": _history(state["messages"], message),
                  "question": question, "question_field": question_field,
                  "new_task": update.new_task})
    return state


def task_query(state):
    """Retrieve current work and reported conditions without completed setup steps."""
    task = _previous_state(state)
    parts = [fact["value"] for fact in task["facts"] if fact["key"] in {"equipment_type", "equipment_model"}]
    software = any(fact["key"] == "scope" and fact["value"] in {"application", "software"}
                   for fact in task["facts"])
    if software:
        parts += [fact["value"] for fact in task["facts"] if fact["key"] == "application"]
    parts += [goal["text"] for goal in task["goals"].values()]
    parts += [fact["value"] for fact in task["facts"]
              if fact["key"] in {"connection", "current_step", "error_text", "symptoms"}]
    return "\n".join(dict.fromkeys(parts))
