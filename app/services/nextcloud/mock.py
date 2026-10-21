from typing import Protocol

from app.models import Message
from app.services.chat.messages import append_message


class Messenger(Protocol):
    def send(self, session, incident_id: str, content: str) -> Message: ...


class MockNextcloud:
    """A durable mock inbox: inspect assistant messages through GET /incidents/{id}."""

    def send(self, session, incident_id, content):
        return append_message(session, incident_id, "assistant", content, "DELIVERED_MOCK")
