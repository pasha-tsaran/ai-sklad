import json
from datetime import timezone

from fastapi import HTTPException
from sqlalchemy import select

from app.models import Escalation, Message, SolutionAttempt, new_id, now
from app.services.chat.audit import audit
from app.services.chat.policies import FORM_ACTION
from app.services.chat.service import add_user_message, enqueue, get_owned, messenger
from app.services.mail.draft import make_eml, make_mailto
from app.services.rag.documents import Illustration


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
    if settings.mail_backend == "mailto":
        # Complete user-entered fields go to the employee's own mail client.
        # This mode never queues SMTP or claims the employee pressed Send.
        entered_fields = (
            f"ФИО: {details.account_holder_name}\nЛогин: {details.account_login}\n"
            f"Компьютер: {details.computer_name}\nТема: {details.subject}\n"
            f"Описание: {details.additional_info}\n"
        )
        draft = {**escalation.email, "body": (
            entered_fields
            + (f"\nИсходная проблема:\n{incident.description}\n" if incident.description else "")
        )}
        link = make_mailto({**draft, "body": entered_fields}, settings.support_email)
        explanation = (
            "Черновик письма подготовлен. Получатель: " + settings.support_email + ".\n"
            "Откройте письмо в своей почтовой программе, проверьте данные и нажмите «Отправить». "
            "Пока письмо не отправлено."
        )
        if link:
            explanation += "\n\n[Открыть письмо](" + link + ")"
        explanation += (
            "\n\nТакже подготовлен файл support-request-….eml со всеми данными и исходной проблемой для отдельного сообщения. "
            "Если ссылка не открывает почту, откройте этот файл в своей почтовой программе.\n"
            "Чтобы вернуться к боту, напишите «Продолжить с ботом»."
        )
        messenger.send_blocks(session, incident.id, [
            (explanation, Illustration("support-request.eml", "Черновик письма в поддержку", "message/rfc822",
                                       make_eml(draft, settings.support_email))),
        ])
        escalation.status = "DRAFT"
        audit(session, incident.id, "escalation.draft_prepared", escalation_id=escalation.id)
        return {"incident_id": incident.id, "escalation_id": escalation.id, "operation_id": None,
                "draft_prepared": True, "mail_sent": False}
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
