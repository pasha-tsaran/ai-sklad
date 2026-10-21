from sqlalchemy import select

from app.models import Feedback, Solution, SolutionAttempt


def solution_stats(session):
    solutions = session.scalars(select(Solution).order_by(Solution.priority, Solution.id)).all()
    counts = {s.id: {"successes": 0, "failures": 0, "attempts": 0} for s in solutions}
    rows = session.execute(
        select(SolutionAttempt.solution_id, Feedback.helped).outerjoin(
            Feedback, Feedback.attempt_id == SolutionAttempt.id
        )
    )
    for solution_id, helped in rows:
        count = counts[solution_id]
        count["attempts"] += 1
        if helped is not None:
            count["successes" if helped else "failures"] += 1
    result = []
    for solution in solutions:
        count = counts[solution.id]
        assessed = count["successes"] + count["failures"]
        result.append(
            {
                "solution_id": solution.id,
                "title": solution.title,
                "version": solution.version,
                **count,
                "assessed": assessed,
                "success_rate": count["successes"] / assessed if assessed else None,
                "smoothed_success_rate": (count["successes"] + 1) / (assessed + 2),
                "feedback_rate": assessed / count["attempts"] if count["attempts"] else None,
            }
        )
    return result
