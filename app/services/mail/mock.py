from typing import Protocol


class Mailer(Protocol):
    def send(self, email: dict) -> str: ...


class MockMail:
    """Never opens a network connection. The immutable email is stored in Escalation."""

    def send(self, email):
        if not email.get("message_id") or not email.get("to"):
            raise ValueError("Invalid email envelope")
        return "SENT_MOCK"
