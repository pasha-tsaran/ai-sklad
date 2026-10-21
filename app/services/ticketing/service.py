import json
from datetime import timezone

from fastapi import HTTPException
from sqlalchemy import select

from app.models import Escalation, Message, SolutionAttempt, new_id, now
from app.services.chat.audit import audit
from app.services.chat.policies import FORM_ACTION
from app.services.chat.service import add_user_message, enqueue, get_owned, messenger


def collect_details(session, user_id, incident_id, details, settings):
    incident = get_owned(session, incident_id, user_id, lock=True)
    existing = session.scalar(select(Escalation).where(Escalation.incident_id == incident.id))
    data = details.model_dump()
    if existing:
        if existing.details != data:
            raise HTTPException(409, "Заявка уже сформирована с другими данными")
        return {"incident_id": incident.id, "escalation_id": existing.id, "duplicate": True}
    if incident.pending_action != FORM_ACTION:
        raise HTTPException(409, "Обращение не ожидает форму эскалации")
    add_user_message(session, incident, "Данные заявки: " + json.dumps(data, ensure_ascii=False))
    messages = session.scalars(
        select(Message).where(Message.incident_id == incident.id).order_by(Message.sequence)
    ).all()
    attempts = session.scalars(
        select(SolutionAttempt)
        .where(SolutionAttempt.incident_id == incident.id)
        .order_by(SolutionAttempt.ordinal)
    ).all()
    transcript = "\n\n".join(
        f"{'Пользователь' if m.role == 'user' else 'Помощник (бот)'}: {m.content}" for m in messages
    )
    escalation_id = new_id()
    created_at = incident.created_at
    if created_at.tzinfo is None:  # SQLite tests
        created_at = created_at.replace(tzinfo=timezone.utc)
    body = (
        f"Обращение: {incident.id}\n"
        f"ФИО пользователя рабочей станции: {details.account_holder_name}\n"
        f"Логин: {details.account_login}\nКомпьютер: {details.computer_name}\n"
        f"Время обращения (UTC): {created_at.isoformat()}\n"
        f"Дополнительно: {details.additional_info}\n\n"
        "Испробованные решения:\n"
        + "\n".join(f"{a.ordinal}. {a.evidence['title']} — {a.status}" for a in attempts)
        + "\n\nИстория разговора:\n"
        + transcript
    )
    escalation = Escalation(
        id=escalation_id,
        incident_id=incident.id,
        reason=incident.escalation_reason,
        details=data,
        email={
            "to": settings.support_email,
            "subject": f"[AI-Support {incident.id}] {details.subject}",
            "message_id": f"<{escalation_id}@ai-support.local>",
            "body": body,
        },
    )
    session.add(escalation)
    incident.subject = details.subject
    incident.state = "ESCALATED"
    incident.pending_action = None
    incident.result = "ESCALATE"
    incident.closed_at = now()
    incident.version += 1
    job = enqueue(session, incident, "SEND_MAIL")
    messenger.send(
        session,
        incident.id,
        "Заявка сохранена. "
        + (
            "Письмо поставлено в очередь отправки в поддержку."
            if settings.mail_backend == "smtp"
            else "Подготовлено тестовое письмо (mock), реальной отправки не будет."
        ),
    )
    audit(session, incident.id, "escalation.queued", escalation_id=escalation.id)
    return {"incident_id": incident.id, "escalation_id": escalation.id, "operation_id": job.id}
