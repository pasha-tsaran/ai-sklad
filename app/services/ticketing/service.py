import json
import re
from datetime import timezone

from fastapi import HTTPException
from sqlalchemy import select

from app.models import Escalation, Message, SolutionAttempt, new_id, now
from app.services.chat.audit import audit
from app.services.chat.memory import redact_device_tokens
from app.services.chat.policies import FORM_ACTION
from app.services.chat.service import add_user_message, enqueue, get_owned, messenger
from app.services.mail.draft import make_mailto
from app.services.mail.presentation import letter_preview


def _letter_text(content):
    """Omit the retired login field from letter copies, keeping private history intact."""
    content = redact_device_tokens(content)
    prefix = "Данные заявки: "
    if content.startswith(prefix):
        try:
            fields = json.loads(content[len(prefix):])
        except (TypeError, ValueError):
            pass
        else:
            if isinstance(fields, dict):
                fields.pop("account_login", None)
                content = prefix + json.dumps(fields, ensure_ascii=False)
    return re.sub(
        r"^[ \t]*(?:логин(?:\s+пользователя(?:\s+рабочей\s+станции)?)?|account_login)\s*:[^\r\n]*(?:\r?\n|$)"
        r"|^[ \t]*Данные профиля Nextcloud[ \t]*(?:\r?\n|$)", "", content, flags=re.I | re.M,
    )


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
        f"{'Пользователь' if m.role == 'user' else 'Помощник (бот)'}: {_letter_text(m.content)}" for m in messages
    )
    escalation_id = new_id()
    created_at = incident.created_at
    if created_at.tzinfo is None:  # SQLite tests
        created_at = created_at.replace(tzinfo=timezone.utc)
    body = (
        f"Обращение: {incident.id}\n"
        f"ФИО пользователя рабочей станции: {details.account_holder_name}\n"
        f"Компьютер: {details.computer_name}\n"
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
            "subject": f"[AI-Support] {details.subject}",
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
        # The employee's explicit description is the main letter text. The topic
        # belongs only to Subject; an old chat report isn't a substitute for it.
        footer = f"ФИО: {details.account_holder_name}\nКомпьютер: {details.computer_name}\n"
        draft = {**escalation.email, "body": (
            (details.additional_info + "\n\n" if details.additional_info else "") + footer
        )}
        escalation.email = draft
        link = make_mailto(draft, settings.support_email)
        explanation = (
            "Подготовлено письмо для технической поддержки.\n\n"
            + letter_preview(draft, settings.support_email)
        )
        if incident.has_user_photos:
            explanation += ("\n\nФотографии из чата не добавлены в письмо автоматически. "
                            "Прикрепите нужные фотографии в своём почтовом ящике перед отправкой.")
        if link:
            explanation += "\n\n[Открыть письмо в почте](" + link + ")"
        else:
            # Some mail clients reject long URIs. Preserve the whole letter in
            # the visible preview rather than silently shortening it or sending a file.
            link = make_mailto({**draft, "body": ""}, settings.support_email)
            if link:
                explanation += ("\n\n[Открыть почту](" + link + ")\n"
                                "Письмо длинное: скопируйте текст письма выше в открывшееся окно почты.")
            else:
                link = make_mailto({"subject": "[AI-Support]", "body": ""}, settings.support_email)
                explanation += ("\n\n[Открыть почту](" + link + ")\n"
                                "Скопируйте полную тему и текст письма выше в открывшееся окно почты.")
        explanation += (
            "\n\nНажмите ссылку, проверьте письмо и отправьте его из своего почтового ящика.\n"
            "Чтобы вернуться к боту, напишите «Продолжить с ботом»."
        )
        messenger.send_blocks(session, incident.id, [(explanation, None)])
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
