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
        with factory() as session:
            retriever = LocalRetriever(session, settings.rag_min_score)
            candidates = retriever.search(analysis.search_query, excluded)
            # For this lexical baseline a reformulation must also be grounded in the
            # original user text, rather than inventing a completely different issue.
            grounded = {c.solution_id for c in retriever.search(description, excluded)}
            candidates = [c for c in candidates if c.solution_id in grounded]
        if not candidates:
            reason = "NO_MORE_SOLUTIONS" if excluded else "NO_KNOWLEDGE"
        else:
            candidate = candidates[0]
            try:
                answer = AnswerResult.model_validate(llm.compose(candidate))
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
                + "\n".join(f"{i}. {step}" for i, step in enumerate(candidate.steps, 1))
                + f"\n\nИсточник: {candidate.chunk_id}, версия {candidate.version}.\nПомогло решение?"
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
                },
                retrieval_score=candidate.score,
                llm_backend=llm.name,
            )
            session.add(attempt)
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
        if escalation.status == "SENT_MOCK":
            return
        escalation_id, body = escalation.id, escalation.email["body"]
    try:
        summary = SummaryResult.model_validate(llm.summarize(body[-16000:])).summary
    except Exception:
        summary = "Summary недоступно. Полная история обращения приведена ниже."
    with factory() as session, session.begin():
        escalation = session.scalar(
            select(Escalation).where(Escalation.id == escalation_id).with_for_update()
        )
        if escalation.status == "SENT_MOCK":
            return
        email = {**escalation.email, "summary": summary}
        # Preserve the full transcript. Summary is explicitly a non-authoritative addition.
        email["body"] = "Краткое описание (LLM, справочно):\n" + summary + "\n\n" + body
        escalation.email = email
        escalation.status = MockMail().send(email)
        escalation.sent_at = now()
        audit(session, escalation.incident_id, "mail.sent_mock", escalation_id=escalation.id)


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
