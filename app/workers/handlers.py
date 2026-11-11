from sqlalchemy import select

from app.models import (
    AuditEvent,
    Escalation,
    Incident,
    Job,
    KnowledgeChunk,
    KnowledgeDocument,
    Message,
    MessageAttachment,
    Solution,
    SolutionAttempt,
    now,
)
from app.schemas.contracts import AnalysisResult, AnswerResult, DialogueResult, FollowupResult, SummaryResult
from app.services.chat.audit import audit
from app.services.chat.dialogue import REFUSAL, refusal_reply, requires_support, safe_social_reply
from app.services.chat.memory import load_context
from app.services.chat.policies import MAX_ATTEMPTS
from app.services.chat.service import ask_escalation, messenger
from app.services.mail.mock import MockMail
from app.services.rag.embeddings import tokens
from app.services.rag.retrieval import LocalRetriever
from app.services.rag.openwebui import OpenWebUIKnowledge, save_snapshot
from app.services.mail.smtp import SMTPMail
from app.models import Conversation
from app.services.rag.documents import Illustration

CLARIFY_QUESTIONS = {
    "DEVICE_TYPE": "О каком типе оборудования идёт речь — сканер штрихкодов, сканер документов, принтер документов А4 или термопринтер?",
    "DEVICE_MODEL": "О каком типе оборудования идёт речь — сканер штрихкодов, сканер документов, принтер документов А4 или термопринтер?",
    "CONNECTION": "Как устройство подключено: кабелем USB к компьютеру, сетевым кабелем или по Wi-Fi?",
    "ERROR_TEXT": "Какой точный текст ошибки вы видите? Перепишите его, пожалуйста.",
    "SYMPTOMS": "Опишите, что именно не работает и что появляется на экране.",
}
NO_FOLLOWUP_ANSWER = (
    "В переданной инструкции базы знаний нет ответа на этот вопрос. "
    "Недостаточно сведений для ответа на ваше уточнение. "
    "Требуется помощь специалиста технической поддержки. "
    "Для перехода к оформлению заявки напишите «оператор». Обращение ещё не передано."
)
FOLLOWUP_UNAVAILABLE = (
    "Не удалось проверить уточнение: сервис временно недоступен или вернул некорректный ответ. "
    "Повторите вопрос или напишите «оператор». Текущая попытка сохранена."
)


def converse_or_support(factory, job_id, llm, settings, handler):
    """Route ordinary chat without spending or resolving a technical attempt."""
    with factory() as session, session.begin():
        job = session.get(Job, job_id)
        incident = session.get(Incident, job.incident_id)
        if incident.pending_action != "PROCESSING" or incident.version != job.expected_version:
            return
        incident_id, version = incident.id, incident.version
        message = session.scalar(select(Message.content).where(
            Message.incident_id == incident.id, Message.role == "user",
        ).order_by(Message.sequence.desc()).limit(1)) or ""
        response = refusal_reply(message)
        technical = requires_support(message)
        context = load_context(session, incident, settings) if not response and not technical else {}
    if not response and not technical:
        try:
            routed = DialogueResult.model_validate(llm.dialogue(message, context))
            if routed.route == "REFUSE":
                response = REFUSAL
            elif routed.route == "SOCIAL":
                response = safe_social_reply(routed.reply)
        except Exception:
            # Invalid free text cannot bypass approved-source rendering.
            response = None
    if not response:
        with factory() as session, session.begin():
            incident = session.scalar(select(Incident).where(Incident.id == incident_id).with_for_update())
            if incident.version != version or incident.pending_action != "PROCESSING":
                audit(session, incident_id, "job.stale_result_ignored", job_id=job_id)
                return
            accepted = session.scalar(select(AuditEvent).where(
                AuditEvent.incident_id == incident.id, AuditEvent.event == "message.accepted",
                AuditEvent.details["operation_id"].as_string() == job_id,
            ))
            # The explicit acceptance marker distinguishes new deferred reports
            # from legacy jobs and feedback, whose report is already populated.
            if (accepted and accepted.details.get("technical_description_deferred")
                    and not accepted.details.get("technical_description_applied")):
                if not incident.description:
                    incident.subject = message[:200]
                incident.description = (incident.description + "\n" + message).strip()[-8000:]
                accepted.details = {**accepted.details, "technical_description_applied": True}
        return handler(factory, job_id, llm, settings)
    with factory() as session, session.begin():
        incident = session.scalar(select(Incident).where(Incident.id == incident_id).with_for_update())
        if incident.version != version or incident.pending_action != "PROCESSING":
            audit(session, incident_id, "job.stale_result_ignored", job_id=job_id)
            return
        # Keep the full transcript. Deferred conversational text never enters
        # the technical report used to derive structured long-term facts.
        if not incident.description:
            incident.subject = "Диалог с поддержкой"
        accepted = session.scalar(select(AuditEvent).where(
            AuditEvent.incident_id == incident.id, AuditEvent.event == "message.accepted",
            AuditEvent.details["operation_id"].as_string() == job_id,
        ))
        if accepted and accepted.details.get("technical_description_deferred"):
            incident.result = accepted.details.get("previous_result")
        messenger.send(session, incident_id, response)
        incident.pending_action = None
        incident.version += 1
        audit(session, incident_id, "dialogue.replied", refused=response == REFUSAL)


def advance(factory, job_id, llm, settings):
    with factory() as session, session.begin():
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
        context = load_context(session, incident, settings)
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
        analysis = AnalysisResult.model_validate(llm.analyze_context(description, context))
        if analysis.clarification_key:
            result = "NEED_MORE_INFO"
            candidates = None
        elif settings.knowledge_backend == "openwebui":
            candidates = OpenWebUIKnowledge(settings).search(description, excluded, context=context)
        else:
            with factory() as session:
                retriever = LocalRetriever(session, settings.rag_min_score)
                candidates = retriever.search(analysis.search_query, excluded)
                grounded = {c.solution_id for c in retriever.search(description, excluded)}
                candidates = [c for c in candidates if c.solution_id in grounded]
        if candidates is None:
            pass
        elif not candidates:
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
            messenger.send(session, incident_id, CLARIFY_QUESTIONS.get(
                analysis.clarification_key if analysis else None,
                "Уточните устройство и опишите, что именно не работает.",
            ))
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
            message = messenger.send(session, incident_id, content, candidate.illustrations)
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
                    "prompt_version": "v2-grounded-memory",
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


def current_source(session, attempt):
    row = session.execute(
        select(Solution, KnowledgeDocument)
        .join(KnowledgeChunk, Solution.chunk_id == KnowledgeChunk.id)
        .join(KnowledgeDocument, KnowledgeChunk.document_id == KnowledgeDocument.id)
        .where(Solution.id == attempt.solution_id)
        .with_for_update()
    ).first()
    return bool(
        row and row[0].active and row[1].approved
        and row[0].chunk_id == attempt.evidence["chunk_id"]
        and row[0].version == attempt.evidence["version"]
        and row[0].steps == attempt.evidence["steps"]
    )


def clarify(factory, job_id, llm, settings):
    """A question about a pending instruction never creates/counts another attempt."""
    with factory() as session, session.begin():
        job = session.get(Job, job_id)
        incident = session.get(Incident, job.incident_id)
        if (
            incident.pending_action != "PROCESSING" or incident.version != job.expected_version
            or incident.state != "WAITING_FEEDBACK"
        ):
            return
        attempt = session.scalar(select(SolutionAttempt).where(
            SolutionAttempt.incident_id == incident.id, SolutionAttempt.status == "WAITING_FEEDBACK",
        ))
        if not attempt:
            raise ValueError("Missing pending attempt")
        incident_id, version, attempt_id = incident.id, incident.version, attempt.id
        evidence = dict(attempt.evidence)
        context = load_context(session, incident, settings)
        question = session.scalar(select(Message.content).where(
            Message.incident_id == incident.id, Message.role == "user",
        ).order_by(Message.sequence.desc()).limit(1))
        # Associate images with this attempt's original response, not a different solution.
        first = session.get(Message, attempt.message_id)
        following_user = session.scalar(select(Message.sequence).where(
            Message.incident_id == incident.id, Message.role == "user", Message.sequence > first.sequence,
        ).order_by(Message.sequence).limit(1))
        photos = session.scalars(select(MessageAttachment).join(Message).where(
            Message.incident_id == incident.id, Message.sequence > first.sequence,
            Message.sequence < following_user,
        ).order_by(Message.sequence)).all() if following_user is not None else []
        illustrations = [Illustration(p.filename, p.caption, p.content_type, p.data) for p in photos]
        approved = current_source(session, attempt)
    excerpts = [p for p in evidence["content"].split("\n\n") if p.strip()]
    # Never truncate a procedure mid-sentence. The complete source remains in the response below.
    if len(excerpts) > 256:
        excerpts = [evidence["content"]]
    answer = None
    model_failed = False
    source = evidence.get("source", {})
    if approved and source.get("backend") == "openwebui":
        approved = OpenWebUIKnowledge(settings).is_current(source)
    if approved:
        try:
            answer = FollowupResult.model_validate(llm.followup(question, context, {
                "chunk_id": evidence["chunk_id"], "title": evidence["title"], "excerpts": excerpts,
                "illustrations": [{"index": i, "caption": p.caption} for i, p in enumerate(illustrations)],
            }))
            valid = (
                answer.result == "EXCERPT" and answer.source_chunk_ids == [evidence["chunk_id"]]
                and bool(answer.step_indices)
                and len(set(answer.step_indices)) == len(answer.step_indices)
                and len(set(answer.illustration_indices)) == len(answer.illustration_indices)
                and all(0 <= i < len(excerpts) for i in answer.step_indices)
                and all(0 <= i < len(illustrations) for i in answer.illustration_indices)
            )
        except Exception:
            valid = False
            model_failed = True
    else:
        valid = False
    # Remote membership/content may change during the model call, just as local approval can.
    if valid and source.get("backend") == "openwebui":
        valid = OpenWebUIKnowledge(settings).is_current(source)
    with factory() as session, session.begin():
        incident = session.scalar(select(Incident).where(Incident.id == incident_id).with_for_update())
        if incident.version != version or incident.pending_action != "PROCESSING":
            audit(session, incident_id, "job.stale_result_ignored", job_id=job_id)
            return
        attempt = session.get(SolutionAttempt, attempt_id)
        valid = valid and attempt.status == "WAITING_FEEDBACK" and current_source(session, attempt)
        if valid:
            content = (
                "В утверждённой инструкции есть такой фрагмент:\n\n"
                + "\n\n".join(excerpts[i] for i in sorted(answer.step_indices))
            )
            if len(answer.step_indices) < len(excerpts):
                content += "\n\nПолная инструкция с условиями и предупреждениями:\n" + evidence["content"]
            content += "\n\nИсточник: " + evidence["title"] + ".\nУдалось выполнить этот пункт?"
            images = [illustrations[i] for i in sorted(answer.illustration_indices)]
        else:
            content = FOLLOWUP_UNAVAILABLE if model_failed else NO_FOLLOWUP_ANSWER
            images = []
        conversation = session.get(Conversation, incident.conversation_id)
        if conversation.channel == "nextcloud_talk":
            content += f"\nДля оценки решения:\nПомогло {attempt.id}\nили\nНе помогло {attempt.id}"
        messenger.send(session, incident_id, content, images)
        incident.pending_action = None
        incident.result = "ANSWER" if valid else ("NEED_MORE_INFO" if model_failed else "NO_KNOWLEDGE")
        incident.version += 1
        audit(session, incident_id, "instruction.clarified", attempt_id=attempt_id, source_answer=bool(valid))


def send_mail(factory, job_id, llm, settings):
    if settings.mail_backend == "mailto":
        return  # This mode never submits old queued mail, even through a direct caller.
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
        query = select(Job).where(Job.status == "PENDING")
        if settings.mail_backend == "mailto":
            query = query.where(Job.kind != "SEND_MAIL")
        job = session.scalar(
            query
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
        if kind in {"ADVANCE", "CLARIFY"}:
            converse_or_support(factory, job_id, llm, settings, advance if kind == "ADVANCE" else clarify)
        else:
            {"SEND_MAIL": send_mail}[kind](factory, job_id, llm, settings)
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
                    kind in {"ADVANCE", "CLARIFY"}
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
