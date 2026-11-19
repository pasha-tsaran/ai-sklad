"""Recognize an initial goal answer inside the employee's pending task frame."""

from app.services.rag.query import _terms


_GENERIC = {"настр", "подкл", "работ", "помог", "сдела", "измен", "нужно", "хочу", "парам", "устро", "обору",
            "прогр", "прило", "компь", "телеф"}


def continues_pending_goal(message, prior, question, update):
    if (not update.new_task or not update.goals or prior.get("goals") or not question
            or question != prior.get("question") or prior.get("question_field") not in (None, "goal")):
        return False
    previous = " ".join(prior.get("messages", []))
    previous += " " + " ".join(fact.get("evidence", "") for fact in prior.get("facts", []))
    current = message + " " + " ".join(goal.text for goal in update.goals)
    return bool((_terms(previous) & _terms(current)) - _GENERIC)
