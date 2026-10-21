from sqlalchemy import select

from app.models import (
    Escalation,
    Incident,
    Job,
    KnowledgeChunk,
    KnowledgeDocument,
    Solution,
    SolutionAttempt,
    now,
)
from app.schemas.contracts import AnalysisResult, AnswerResult, SummaryResult
from app.services.chat.audit import audit
from app.services.chat.policies import MAX_ATTEMPTS
from app.services.chat.service import ask_escalation, messenger
from app.services.mail.mock import MockMail
from app.services.rag.embeddings import tokens
from app.services.rag.retrieval import LocalRetriever
from app.services.rag.openwebui import OpenWebUIKnowledge, save_snapshot
from app.services.mail.smtp import SMTPMail
from app.models import Conversation


def advance(factory, job_id, llm, settings):
    with factory() as session:
        job = session.get(Job, job_id)
        incident = session.get(Incident, job.incident_id)
        if incident.pending_action != "PROCESSING" or incident.version != job.expected_version:
            return
        incident_id, version, description = incident.id, incident.version, incident.description
        attempts = session.scalars(
            select(SolutionAttempt).where(
                SolutionAttempt.incident_id == incident_id,
            )
        ).all()
        excluded = {a.solution_id for a in attempts}
        failures = incident.failed_attempts
    # All model calls are outside database transactions.
    analysis = None
    candidate = None
    result = None
    reason = None
    invalid_answer = False
    if failures >= MAX_ATTEMPTS or len(excluded) >= MAX_ATTEMPTS:
        reason = "ATTEMPTS_EXHAUSTED"
    elif len(tokens(description)) < 2:
        result = "NEED_MORE_INFO"
    else:
        analysis = AnalysisResult.model_validate(llm.analyze(description))
        if settings.knowledge_backend == "openwebui":
            candidates = OpenWebUIKnowledge(settings).search(description, excluded)
        else:
            with factory() as session:
                retriever = LocalRetriever(session, settings.rag_min_score)
                candidates = retriever.search(analysis.search_query, excluded)
                grounded = {c.solution_id for c in retriever.search(description, excluded)}
                candidates = [c for c in candidates if c.solution_id in grounded]
        if not candidates:
            reason = "NO_MORE_SOLUTIONS" if excluded else "NO_KNOWLEDGE"
        else:
            candidate = candidates[0]
            try:
                answer = (
                    AnswerResult(
                        result="ANSWER",
                        solution_id=candidate.solution_id,
                        source_chunk_ids=[candidate.chunk_id],
                    )
                    if settings.knowledge_backend == "openwebui"
                    else AnswerResult.model_validate(llm.compose(candidate))
                )
                valid = (
                    answer.result == "ANSWER"
                    and answer.solution_id == candidate.solution_id
                    and answer.source_chunk_ids == [candidate.chunk_id]
                )
            except Exception:
                # A validated, approved card remains safe even when generation fails.
                valid = False
                answer = None
            if not valid:
                invalid_answer = True
            # Model suggestions cannot initiate escalation or alter attempts. Render
            # the selected approved card even if the model output was invalid.
            result = "ANSWER"
    with factory() as session, session.begin():
        incident = session.scalar(select(Incident).where(Incident.id == incident_id).with_for_update())
        if incident.version != version or incident.pending_action != "PROCESSING":
            audit(session, incident_id, "job.stale_result_ignored", job_id=job_id)
            return
        if analysis:
            incident.problem_type = analysis.problem_type
            incident.equipment_model = analysis.equipment_model
        if candidate:
            if settings.knowledge_backend == "openwebui":
                save_snapshot(session, candidate)
            current = session.execute(
                select(Solution, KnowledgeDocument)
                .join(KnowledgeChunk, Solution.chunk_id == KnowledgeChunk.id)
                .join(KnowledgeDocument, KnowledgeChunk.document_id == KnowledgeDocument.id)
                .where(Solution.id == candidate.solution_id)
                .with_for_update()
            ).first()
            if (
                not current
                or not current[0].active
                or not current[1].approved
                or current[0].version != candidate.version
                or current[0].steps != candidate.steps
            ):
                reason = "NO_KNOWLEDGE"
                audit(session, incident_id, "solution.revoked_before_delivery")
        if reason:
            ask_escalation(session, incident, reason)
        elif result == "NEED_MORE_INFO":
            incident.result = result
            incident.pending_action = None
            messenger.send(session, incident_id, "Уточните устройство и опишите, что именно не работает.")
        else:
            # Deliberately never use generated technical prose, commands or attachment URLs.
            intro = "Попробуйте шаги из утверждённой инструкции."
            if not invalid_answer and answer.intro_key == "CHECK_APPROVED_STEPS":
                intro = "Проверьте следующие пункты утверждённой инструкции."
            content = (
                intro
                + "\n\n"
                + candidate.title
                + "\n"
                + (
                    candidate.content
                    if candidate.source
                    else "\n".join(f"{i}. {step}" for i, step in enumerate(candidate.steps, 1))
                )
                + (
                    f"\n\nИсточник: {candidate.title}."
                    if candidate.source
                    else f"\n\nИсточник: {candidate.chunk_id}, версия {candidate.version}."
                )
                + "\nПомогло решение?"
            )
            incident.state = "SOLUTION_PROPOSED"
            audit(session, incident_id, "incident.state_changed", state=incident.state)
            message = messenger.send(session, incident_id, content)
            attempt = SolutionAttempt(
                incident_id=incident_id,
                solution_id=candidate.solution_id,
                ordinal=len(excluded) + 1,
                message_id=message.id,
                evidence={
                    "title": candidate.title,
                    "version": candidate.version,
                    "chunk_id": candidate.chunk_id,
                    "content": candidate.content,
                    "steps": candidate.steps,
                    "prompt_version": "v1",
                    "source": candidate.source,
                },
                retrieval_score=candidate.score,
                llm_backend=llm.name,
            )
            session.add(attempt)
            session.flush()
            conversation = session.get(Conversation, incident.conversation_id)
            if conversation.channel == "nextcloud_talk":
                message.content += (
                    f"\n\nДля оценки отправьте:\nПомогло {attempt.id}\nили\nНе помогло {attempt.id}"
                )
            incident.state = "WAITING_FEEDBACK"
            incident.pending_action = None
            incident.result = "ANSWER"
            audit(
                session,
                incident_id,
                "solution.delivered",
                solution_id=candidate.solution_id,
                ordinal=attempt.ordinal,
                state=incident.state,
                llm_fallback=invalid_answer,
            )
        incident.version += 1


def send_mail(factory, job_id, llm, settings):
    with factory() as session:
        job = session.get(Job, job_id)
        escalation = session.scalar(select(Escalation).where(Escalation.incident_id == job.incident_id))
        if escalation.status != "QUEUED":
            return
        escalation_id = escalation.id
        body = escalation.email.get("original_body", escalation.email["body"])
    try:
        summary = SummaryResult.model_validate(llm.summarize(body[-16000:])).summary
    except Exception:
        summary = "Summary недоступно. Полная история обращения приведена ниже."
    with factory() as session, session.begin():
        escalation = session.scalar(
            select(Escalation).where(Escalation.id == escalation_id).with_for_update()
        )
        if escalation.status != "QUEUED":
            return
        email = {**escalation.email, "summary": summary, "original_body": body}
        # Preserve the full transcript. Summary is explicitly a non-authoritative addition.
        email["body"] = "Краткое описание (LLM, справочно):\n" + summary + "\n\n" + body
        escalation.email = email
        escalation.status = "SENDING"
    # Commit intent before contacting SMTP. Ambiguous delivery must never be retried automatically.
    try:
        status = (SMTPMail(settings) if settings.mail_backend == "smtp" else MockMail()).send(email)
        error_code = None
    except Exception as exc:
        status, error_code = "UNKNOWN", type(exc).__name__
    with factory() as session, session.begin():
        escalation = session.get(Escalation, escalation_id)
        escalation.status = status
        if status in {"SENT", "SENT_MOCK"}:
            escalation.sent_at = now()
        audit(
            session,
            escalation.incident_id,
            "mail." + status.lower(),
            escalation_id=escalation.id,
            error_code=error_code,
        )
        if status == "SENT":
            messenger.send(
                session, escalation.incident_id, "Письмо с заявкой принято почтовым сервером поддержки."
            )
        elif status == "UNKNOWN":
            messenger.send(
                session,
                escalation.incident_id,
                "Заявка сохранена, но отправку письма подтвердить не удалось. Требуется проверка администратором.",
            )


def run_one(factory, llm, settings):
    """One job; caller must hold the singleton worker lock in production."""
    with factory() as session, session.begin():
        job = session.scalar(
            select(Job)
            .where(Job.status == "PENDING")
            .order_by(Job.created_at, Job.id)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        if not job:
            return False
        job.status = "RUNNING"
        job.tries += 1
        job_id, kind = job.id, job.kind
    try:
        {"ADVANCE": advance, "SEND_MAIL": send_mail}[kind](factory, job_id, llm, settings)
    except Exception as error:
        with factory() as session, session.begin():
            job = session.get(Job, job_id)
            job.error_code = type(error).__name__  # No response bodies, tokens or PII in logs.
            job.status = "PENDING" if job.tries < 3 else "FAILED"
            audit(
                session,
                job.incident_id,
                "job.error",
                job_id=job.id,
                error_code=job.error_code,
                tries=job.tries,
            )
            if job.status == "FAILED":
                incident = session.scalar(
                    select(Incident)
                    .where(
                        Incident.id == job.incident_id,
                    )
                    .with_for_update()
                )
                if (
                    kind == "ADVANCE"
                    and incident.pending_action == "PROCESSING"
                    and incident.version == job.expected_version
                ):
                    incident.pending_action = None
                    incident.result = "NEED_MORE_INFO"
                    incident.version += 1
                    messenger.send(
                        session,
                        incident.id,
                        "Сервис временно недоступен. Повторите сообщение или напишите «оператор».",
                    )
                elif kind == "SEND_MAIL":
                    escalation = session.scalar(
                        select(Escalation).where(Escalation.incident_id == incident.id)
                    )
                    escalation.status = "FAILED"
        return True
    with factory() as session, session.begin():
        job = session.get(Job, job_id)
        job.status = "DONE"
        job.error_code = None
        audit(session, job.incident_id, "job.completed", job_id=job.id)
    return True
