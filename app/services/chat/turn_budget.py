"""Share a monotonic network budget across the read-only stages of one turn."""

from contextlib import contextmanager
from contextvars import ContextVar
from time import monotonic

import httpx

from app.core.logging import turn_id_context


_turn_deadline = ContextVar("talk_turn_deadline", default=None)


@contextmanager
def turn_budget(seconds, *, turn_id=None):
    token = _turn_deadline.set(monotonic() + seconds)
    log_token = turn_id_context.set(turn_id)
    try:
        yield
    finally:
        _turn_deadline.reset(token)
        turn_id_context.reset(log_token)


def remaining_seconds():
    deadline = _turn_deadline.get()
    return None if deadline is None else deadline - monotonic()


def limit_timeout(seconds):
    remaining = remaining_seconds()
    if remaining is None:
        return seconds
    if remaining <= 0:
        raise httpx.ReadTimeout("TalkTurnBudgetExceeded")
    return min(seconds, remaining)


def cap_request_timeout(request):
    values = request.extensions.get("timeout", {})
    if remaining_seconds() is not None:
        request.extensions["timeout"] = {
            key: limit_timeout(value if value is not None else float("inf")) for key, value in values.items()
        }
