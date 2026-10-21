import hashlib
import json

from fastapi import HTTPException
from sqlalchemy import select

from app.models import (
    Conversation,
    Escalation,
    Feedback,
    Incident,
    Job,
    Message,
    RequestReceipt,
    SolutionAttempt,
    User,
    now,
)
from app.services.chat.audit import audit
from app.services.chat.messages import append_message
from app.services.chat.policies import FORM_ACTION, FORM_TEXT, MAX_ATTEMPTS, TERMINAL_STATES, direct_intent
from app.services.nextcloud.delivery import RoutedMessenger

messenger = RoutedMessenger()


def get_owned(session, incident_id, user_id, lock=False):
    statement = (
        select(Incident)
        .join(Conversation, Incident.conversation_id == Conversation.id)
        .where(Incident.id == incident_id, Conversation.user_id == user_id)
    )
    if lock:
        statement = statement.with_for_update(of=Incident)
    incident = session.scalar(statement)
    if incident is None:
        raise HTTPException(404, "Обращение не найдено")
    return incident


def idempotent(session, user_id, key, action, payload, operation):
    # Serialize mutations for the single authenticated principal, including deduplication.
    session.scalar(select(User).where(User.id == user_id).with_for_update())
    fingerprint = hashlib.sha256(
        json.dumps(
            {"action": action, "payload": payload},
            sort_keys=True,
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    receipt = session.scalar(
        select(RequestReceipt).where(
            RequestReceipt.user_id == user_id,
            RequestReceipt.key == key,
        )
    )
    if receipt:
        if receipt.fingerprint != fingerprint:
            raise HTTPException(409, "Idempotency-Key уже использован для другого запроса")
        return receipt.response
    response = operation()
    session.add(RequestReceipt(user_id=user_id, key=key, fingerprint=fingerprint, response=response))
    session.flush()
    return response


def add_user_message(session, incident, content):
    return append_message(session, incident.id, "user", content, "RECEIVED")


def enqueue(session, incident, kind):
    job = Job(incident_id=incident.id, kind=kind, expected_version=incident.version)
    session.add(job)
    session.flush()
    audit(session, incident.id, "job.queued", job_id=job.id, kind=kind)
    return job


def ask_escalation(session, incident, reason):
    incident.state = "DIAGNOSING"
    incident.pending_action = FORM_ACTION
    incident.result = "NO_KNOWLEDGE" if reason == "NO_KNOWLEDGE" else "ESCALATE"
    incident.escalation_reason = reason
    prefix = {
        "ATTEMPTS_EXHAUSTED": "Пять решений не помогли. Новые решения больше не предлагаются.",
        "NO_KNOWLEDGE": "В утверждённой базе нет достаточных сведений для технических инструкций.",
        "NO_MORE_SOLUTIONS": "Других подходящих утверждённых решений нет.",
    }.get(reason, "Перейдём к оформлению обращения в поддержку.")
    messenger.send(session, incident.id, prefix + "\n\n" + FORM_TEXT)
    conversation = session.get(Conversation, incident.conversation_id)
    if conversation.channel == "nextcloud_talk":
        messenger.send(
            session,
            incident.id,
            "Отправьте одним сообщением:\nФИО: ...\nЛогин: ...\nКомпьютер: ...\nТема: ...\nОписание: ...",
        )
    audit(session, incident.id, "incident.form_requested", reason=reason)


def accept_chat(session, user_id, request):
    conversation = session.scalar(select(Conversation).where(Conversation.user_id == user_id))
    if not conversation:
        conversation = Conversation(user_id=user_id)
        session.add(conversation)
        session.flush()
    if request.incident_id:
        incident = get_owned(session, request.incident_id, user_id, lock=True)
    else:
        incident = session.scalar(
            select(Incident)
            .where(
                Incident.conversation_id == conversation.id,
                Incident.state.not_in(TERMINAL_STATES),
            )
            .with_for_update()
        )
    if incident and incident.state in TERMINAL_STATES:
        raise HTTPException(409, "Обращение закрыто; отправьте сообщение без incident_id для нового")
    if incident and incident.pending_action == "PROCESSING":
        raise HTTPException(409, "Предыдущее сообщение обрабатывается")
    intent = direct_intent(request.message)
    if intent == "FEEDBACK":
        raise HTTPException(409, "Передайте attempt_id и helped через endpoint feedback")
    if incident and incident.pending_action == FORM_ACTION:
        raise HTTPException(409, "Заполните escalation-details")
    if incident and incident.state == "WAITING_FEEDBACK" and intent != "HUMAN":
        raise HTTPException(409, "Сначала оцените текущее решение через endpoint feedback")
    if not incident:
        incident = Incident(
            conversation_id=conversation.id, subject=request.message[:200], description=request.message
        )
        session.add(incident)
        session.flush()
        audit(session, incident.id, "incident.created")
    else:
        incident.description = (incident.description + "\n" + request.message)[-8000:]
    message = add_user_message(session, incident, request.message)
    incident.version += 1
    if intent == "HUMAN":
        pending = session.scalar(
            select(SolutionAttempt).where(
                SolutionAttempt.incident_id == incident.id,
                SolutionAttempt.status == "WAITING_FEEDBACK",
            )
        )
        if pending:
            pending.status = "UNASSESSED"
        ask_escalation(session, incident, "USER_REQUEST")
        operation_id = None
    else:
        incident.state = "DIAGNOSING"
        incident.pending_action = "PROCESSING"
        incident.result = None
        operation_id = enqueue(session, incident, "ADVANCE").id
    audit(session, incident.id, "message.accepted", message_id=message.id)
    return {"incident_id": incident.id, "message_id": message.id, "operation_id": operation_id}


def accept_feedback(session, user_id, incident_id, request):
    incident = get_owned(session, incident_id, user_id, lock=True)
    attempt = session.scalar(
        select(SolutionAttempt).where(
            SolutionAttempt.id == request.attempt_id,
            SolutionAttempt.incident_id == incident.id,
        )
    )
    if not attempt:
        raise HTTPException(404, "Попытка не найдена")
    existing = session.scalar(select(Feedback).where(Feedback.attempt_id == attempt.id))
    if existing:
        if existing.helped != request.helped or existing.comment != request.comment:
            raise HTTPException(409, "Результат попытки уже зафиксирован")
        return {"incident_id": incident.id, "feedback_id": existing.id, "duplicate": True}
    if incident.state != "WAITING_FEEDBACK" or attempt.status != "WAITING_FEEDBACK":
        raise HTTPException(409, "Эта попытка не ожидает обратной связи")
    feedback = Feedback(attempt_id=attempt.id, helped=request.helped, comment=request.comment)
    session.add(feedback)
    session.flush()
    add_user_message(
        session,
        incident,
        ("Помогло" if request.helped else "Не помогло") + (": " + request.comment if request.comment else ""),
    )
    attempt.status = "SUCCEEDED" if request.helped else "FAILED"
    incident.version += 1
    operation_id = None
    if request.helped:
        incident.state = "RESOLVED"
        incident.pending_action = None
        incident.closed_at = now()
        messenger.send(session, incident.id, "Спасибо за обратную связь. Обращение решено.")
    else:
        incident.failed_attempts += 1
        if incident.failed_attempts >= MAX_ATTEMPTS:
            ask_escalation(session, incident, "ATTEMPTS_EXHAUSTED")
        else:
            incident.state = "DIAGNOSING"
            incident.pending_action = "PROCESSING"
            incident.result = None
            operation_id = enqueue(session, incident, "ADVANCE").id
    audit(
        session,
        incident.id,
        "feedback.recorded",
        attempt_id=attempt.id,
        helped=request.helped,
        state=incident.state,
        failed_attempts=incident.failed_attempts,
    )
    return {"incident_id": incident.id, "feedback_id": feedback.id, "operation_id": operation_id}


def incident_view(session, incident):
    messages = session.scalars(
        select(Message).where(Message.incident_id == incident.id).order_by(Message.sequence)
    ).all()
    attempts = session.scalars(
        select(SolutionAttempt)
        .where(SolutionAttempt.incident_id == incident.id)
        .order_by(SolutionAttempt.ordinal)
    ).all()
    escalation = session.scalar(select(Escalation).where(Escalation.incident_id == incident.id))
    return {
        "id": incident.id,
        "state": incident.state,
        "result": incident.result,
        "subject": incident.subject,
        "problem_type": incident.problem_type,
        "equipment_model": incident.equipment_model,
        "failed_attempts": incident.failed_attempts,
        "max_attempts": MAX_ATTEMPTS,
        "pending_action": incident.pending_action,
        "created_at": incident.created_at,
        "messages": [
            {
                "id": m.id,
                "sequence": m.sequence,
                "role": m.role,
                "content": m.content,
                "delivery_status": m.delivery_status,
                "created_at": m.created_at,
            }
            for m in messages
        ],
        "attempts": [
            {
                "id": a.id,
                "solution_id": a.solution_id,
                "ordinal": a.ordinal,
                "status": a.status,
                "evidence": a.evidence,
            }
            for a in attempts
        ],
        "escalation": {
            "id": escalation.id,
            "status": escalation.status,
            "reason": escalation.reason,
            "mock_email": escalation.email,
            "email": escalation.email,
        }
        if escalation
        else None,
    }
