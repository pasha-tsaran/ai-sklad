from app.core.logging import event
from app.models import AuditEvent


def audit(session, incident_id, name, **details):
    session.add(AuditEvent(incident_id=incident_id, event=name, details=details))
    event(name, incident_id=incident_id, **details)
