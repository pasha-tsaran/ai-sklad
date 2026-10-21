from sqlalchemy import func, select

from app.models import Message


def append_message(session, incident_id, role, content, delivery_status):
    """Caller holds the incident row lock (or has just inserted the incident)."""
    sequence = (
        session.scalar(
            select(func.coalesce(func.max(Message.sequence), 0)).where(Message.incident_id == incident_id)
        )
        + 1
    )
    message = Message(
        incident_id=incident_id,
        sequence=sequence,
        role=role,
        content=content,
        delivery_status=delivery_status,
    )
    session.add(message)
    session.flush()
    return message
