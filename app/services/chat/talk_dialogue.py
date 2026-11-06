"""Durable private Talk dialogue using the same approved steps/photos as WebUI."""

import copy
import re
from dataclasses import dataclass, field
from datetime import timezone
from uuid import NAMESPACE_URL, uuid5

from pydantic import ValidationError
from sqlalchemy import select

from app.core.logging import event
from app.models import (
    Conversation, Delivery, Feedback, Incident, Message, SolutionAttempt, User, now,
)
from app.models.talk import TalkAccount, TalkInbox, TalkRoom
from app.schemas.contracts import AnalysisResult, DialogueResult, EscalationDetails
from app.services.chat.audit import audit
from app.services.chat.memory import load_context, remember_resolution
from app.services.chat.messages import append_message
from app.services.chat.policies import FORM_ACTION, MAX_ATTEMPTS, TERMINAL_STATES, direct_intent
from app.services.chat.service import messenger
from app.services.rag.openwebui import save_snapshot
from app.services.rag.talk import ApprovedTalkDocument, KnowledgeUnavailable, TalkKnowledge
from app.services.ticketing.service import collect_details
from integrations.openwebui.automatic_instruction_images import (
    TIMING_MARKER, _clock, _measure, _milliseconds,
    applicable_images, clarification_images, instruction_blocks, is_confirmation,
    is_ui_question, knowledge_scope_reply, refusal_reply,
    requires_support, safe_social_reply,
)

UNAVAILABLE = "Сейчас не удалось проверить базу знаний. Повторите вопрос немного позже или напишите «оператор»."
CHANGED = "Инструкция изменилась или доступ к ней недоступен. Продолжение приостановлено до проверки актуального источника."
FORM_HELP = ("Отправьте одним сообщением:\nФИО: ...\nЛогин: ...\nКомпьютер: ...\nТема: ...\n"
             "Описание: ...\n\nЧтобы продолжить с ботом без отправки заявки, напишите «Отмена» или «2».")
ESCALATION_MENU = ("Как продолжим?\n1 — оформить письмо в техническую поддержку.\n"
                   "2 — продолжить диалог с ботом.\n\nМожно ответить цифрой или словами. Заявка пока не отправлена.")
RESUME_GUIDE = ("Продолжим с ботом с того же шага. Заявка не отправлена. "
                "Напишите, что не получается, или «Готово», если выполнили показанные действия.")
BACK_TO_BOT = "Продолжим с ботом. Опишите, с каким вопросом нужна помощь."
FORM_FIELDS = {"фио": "account_holder_name", "логин": "account_login", "компьютер": "computer_name",
               "тема": "subject", "описание": "additional_info"}
CLARIFY_QUESTIONS = {
    "DEVICE_MODEL": "Какое устройство не работает? Напишите его название и модель, если она указана на корпусе.",
    "CONNECTION": "Как устройство подключено: кабелем USB к компьютеру, сетевым кабелем или по Wi-Fi?",
    "ERROR_TEXT": "Какой точный текст ошибки вы видите? Перепишите его, пожалуйста.",
    "SYMPTOMS": "Опишите, что именно не работает и что появляется на экране.",
}
_PHOTO_REQUEST = re.compile(
    r"\b(?:покажи(?:те)?|повтори(?:те)?|пришли(?:те)?|отправь(?:те)?)\s+"
    r"(?:(?:мне|ещ[её]|эту?|предыдущую|текущую|следующие|следующую)\s+){0,3}"
    r"(?:фото(?:графи[юияй]*)?|изображени[ея]|картинк[ауи]|скриншот\w*)\b", re.I,
)
_NEXT_PHOTO = re.compile(r"\bследующ\w*\s+(?:фото\w*|изображени\w*|картин\w*|скриншот\w*)\b", re.I)


@dataclass
class Turn:
    action: str
    blocks: list = field(default_factory=list)
    approved: ApprovedTalkDocument | None = None
    flow: dict | None = None
    analysis: AnalysisResult | None = None
    details: EscalationDetails | None = None
    resume: dict | None = None


class _TimedProxy:
    def __init__(self, target, metadata, stages):
        self.target, self.metadata, self.stages = target, metadata, stages

    def __getattr__(self, name):
        value = getattr(self.target, name)
        stage = self.stages.get(name)
        if stage is None:
            return value

        def invoke(*args, **kwargs):
            with _measure(self.metadata, stage):
                return value(*args, **kwargs)
        return invoke


def _report_timing(metadata, started, route, block_count, error_code=None):
    try:
        finished = _clock()
        total = _milliseconds((finished - started) * 1000) if finished is not None and started is not None else None
        state = metadata[TIMING_MARKER]
        event("talk.turn_failed" if error_code else "talk.turn_completed", total_ms=total, route=route,
              timings_ms=dict(state.get("stages_ms", {})), stage_calls=dict(state.get("stage_calls", {})),
              block_count=block_count, error_code=error_code)
    except Exception:
        pass  # Diagnostics never change the stored employee reply.


def _form(message):
    values = {}
    for line in message.splitlines():
        name, separator, value = line.partition(":")
        if separator and name.strip().casefold() in FORM_FIELDS:
            values[FORM_FIELDS[name.strip().casefold()]] = value.strip()
    try:
        return EscalationDetails(**values)
    except ValidationError:
        return None


def _normalized(message):
    return " ".join(re.sub(r"[^\w\s]", " ", message.casefold(), flags=re.UNICODE).split())


def _simple_social(message):
    # Full-message matches keep a greeting plus a technical question on the support route.
    normalized = _normalized(message)
    if normalized in {"привет", "здравствуйте", "здравствуй", "приветствую", "добрый день",
                      "день добрый", "доброе утро", "добрый вечер", "доброй ночи"}:
        return "Здравствуйте! Я помощник технической поддержки. Чем могу помочь?"
    if normalized in {"спасибо", "большое спасибо", "спасибо большое", "благодарю", "благодарю вас",
                      "спасибо за помощь", "спасибо за ответ"}:
        return "Пожалуйста! Если нужна помощь, напишите."
    if normalized in {"как дела", "как твои дела", "как у тебя дела"}:
        return "Спасибо, всё хорошо. Готов помочь с рабочим вопросом."
    return None


def _escalation_choice(message):
    normalized = _normalized(message)
    if normalized in {"2", "отмена", "отменить", "отменить заявку", "назад", "вернуться к боту",
                      "продолжить с ботом", "продолжить диалог с ботом", "продолжить разговор с ботом",
                      "не хочу отправлять", "не хочу отправлять помоги ты мне", "помоги ты мне",
                      "не надо отправлять", "не отправлять заявку", "продолжить", "дальше с ботом"}:
        return "RESUME"
    if normalized in {"1", "оставить заявку", "оставить заявку в тп", "оставить заявку в поддержку",
                      "оформить заявку", "оформить обращение", "заявка в поддержку", "передать в поддержку",
                      "отправить заявку", "отправить заявку в поддержку", "техническая поддержка",
                      "оформить письмо", "оформить письмо в поддержку", "письмо в поддержку"}:
        return "FORM_REQUEST"
    return None


def _return_to_bot(message):
    # Bare «Продолжить» is intentionally excluded: it advances an active guide.
    return _normalized(message) in {
        "продолжить с ботом", "продолжить диалог с ботом", "продолжить разговор с ботом",
        "вернуться к боту", "назад к боту", "дальше с ботом", "2", "отмена",
    }


def _last_problem(flow, context):
    def control(text):
        return direct_intent(text) is not None or is_confirmation(text) or _escalation_choice(text) is not None or _simple_social(text) is not None

    for value in (flow.get("query"), flow.get("clarification_query")):
        if isinstance(value, str) and value.strip() and not control(value):
            return value.strip()[:6000]
    for item in reversed(context.get("short_term", {}).get("messages", [])):
        text = item.get("content", "")
        if item.get("role") == "user" and requires_support(text) and not control(text) and not _form_fields(text):
            return text[:6000]
    subject = context.get("incident", {}).get("subject")
    return subject if isinstance(subject, str) and requires_support(subject) and not control(subject) else ""


def _form_fields(message):
    return any(line.partition(":")[0].strip().casefold() in FORM_FIELDS and ":" in line
               for line in message.splitlines())


def _problem_query(message, flow):
    previous = flow.get("clarification_query")
    if previous and message != previous and len(message.split()) <= 8:
        return (previous + "\n" + message)[:6000]
    return message[:6000]


def _resume_turn(flow, context, llm, knowledge):
    saved = flow.get("escalation", {}).get("resume", {})
    restored = copy.deepcopy(saved.get("flow", {key: value for key, value in flow.items() if key != "escalation"}))
    query = _last_problem(restored, context)
    if restored.get("source"):
        try:
            approved = knowledge.read(restored["source"]["file_id"])
            valid = approved.document["source"] == restored["source"] and knowledge.is_current(restored["source"])
        except KnowledgeUnavailable:
            valid = False
            reply = UNAVAILABLE
        else:
            reply = CHANGED
        if valid:
            return Turn("RESUME", [(RESUME_GUIDE, None)], flow=restored, resume=saved)
        restored = {"clarification_query": query, "excluded": restored.get("excluded", [])} if query else {}
        return Turn("RESUME", [("Заявка не отправлена. " + reply, None)], flow=restored, resume={})
    if query:
        try:
            following = _support_turn(query, restored, context, llm, knowledge)
        except KnowledgeUnavailable:
            following = Turn("REPLY", [(UNAVAILABLE, None)], flow={"clarification_query": query,
                                                                  "excluded": restored.get("excluded", [])})
        if following.action == "GUIDE":
            return following
        if following.action == "REPLY":
            following.action, following.resume = "RESUME", {}
            following.flow = following.flow if following.flow is not None else restored
            return following
        restored = following.flow or {"clarification_query": query, "excluded": restored.get("excluded", [])}
        return Turn("RESUME", [("Заявка не отправлена. В базе знаний пока нет подходящей инструкции. "
                                "Можем продолжить разбираться: напишите, что именно не получается.", None)],
                    flow=restored, resume={})
    return Turn("RESUME", [("Продолжим с ботом. Заявка не отправлена. Опишите, с чем нужна помощь.", None)],
                flow=restored, resume={})


def _photo_blocks(approved, selected, has_more, clarification):
    document = approved.document
    images = [image for image in document["images"] if image["number"] in selected]
    if len(images) != len(selected) or any(image["number"] not in approved.illustrations for image in images):
        raise KnowledgeUnavailable("IllustrationsDoNotMatchSource")
    blocks = []
    for text, image in instruction_blocks(document, images, has_more, clarification):
        if image is None:
            blocks.append((text, None))
            continue
        picture = approved.illustrations[image["number"]]
        if picture.caption != image["caption"]:
            raise KnowledgeUnavailable("IllustrationCaptionMismatch")
        blocks.append((text, picture))
    return blocks


def _support_turn(message, flow, context, llm, knowledge):
    has_guide = bool(flow.get("source"))
    photo_request = bool(_PHOTO_REQUEST.search(message) or _NEXT_PHOTO.search(message))
    next_photo = bool(photo_request and _NEXT_PHOTO.search(message))
    clarification = bool(has_guide and (is_ui_question(message) or photo_request and not next_photo))
    continuation = bool(has_guide and (is_confirmation(message) or next_photo))
    if continuation and not flow.get("has_more"):
        return Turn("REPLY", [("Удалось решить проблему? Напишите «Помогло» или «Не помогло».", None)])
    analysis = None
    if clarification or continuation:
        approved = knowledge.read(flow["source"]["file_id"])
        if approved.document["source"] != flow["source"]:
            return Turn("REPLY", [(CHANGED, None)])
        eligible = flow["selection"]
        if clarification:
            selected = [number for number in clarification_images(
                approved.document, message, flow.get("shown", []) or eligible
            ) if number in eligible][:6]
            next_flow = copy.deepcopy(flow)
        else:
            offset = flow.get("next_offset")
            if type(offset) is not int or offset < 0:
                raise KnowledgeUnavailable("InvalidSavedProgress")
            selected = eligible[offset:offset + 6]
            next_flow = {**flow, "has_more": offset + len(selected) < len(eligible),
                         "next_offset": offset + len(selected) if offset + len(selected) < len(eligible) else None,
                         "shown": selected}
    else:
        problem = _problem_query(message, flow)
        excluded = flow.get("excluded", [])
        # Retrieval has the actual report already. Avoid a generation round trip
        # whenever an approved matching instruction can be found directly.
        approved = knowledge.search(problem, excluded=excluded, fallback_query=problem)
        if approved is None:
            try:
                analysis = AnalysisResult.model_validate(llm.analyze_context(problem, context))
                if analysis.intent == "REQUEST_HUMAN":
                    return Turn("HUMAN", flow={"clarification_query": problem, "excluded": excluded})
                query = analysis.search_query or problem
                if _normalized(query) != _normalized(problem):
                    approved = knowledge.search(query, excluded=excluded, fallback_query=problem)
            except KnowledgeUnavailable:
                raise
            except Exception:
                # Generated repair advice is never required for source-based replies.
                analysis = None
        if approved is None:
            asked = flow.get("clarification_keys", [])
            if analysis and analysis.clarification_key and analysis.clarification_key not in asked:
                return Turn("REPLY", [(CLARIFY_QUESTIONS[analysis.clarification_key], None)], flow={
                    "clarification_query": problem, "clarification_keys": asked + [analysis.clarification_key],
                    "excluded": flow.get("excluded", []),
                })
            return Turn("NO_KNOWLEDGE", flow={"clarification_query": problem,
                                              "clarification_keys": asked, "excluded": excluded})
        if (flow.get("source", {}).get("file_id") == approved.document["source"]["file_id"]
                and flow["source"] != approved.document["source"]):
            return Turn("REPLY", [(CHANGED, None)])
        eligible = applicable_images(approved.document, problem)
        selected = eligible[:6]
        next_flow = {"source": approved.document["source"], "selection": eligible, "shown": selected,
                     "query": problem, "has_more": len(eligible) > 6,
                     "next_offset": 6 if len(eligible) > 6 else None,
                     "excluded": flow.get("excluded", []) if flow else []}
    if not knowledge.is_current(approved.document["source"]):
        return Turn("REPLY", [(CHANGED, None)])
    if not selected:
        text = approved.document["content"].split("Фотографии и пояснения", 1)[0].strip()
        return Turn("GUIDE", [(text + "\n\nУдалось выполнить эти действия?", None)],
                    approved, next_flow, analysis)
    return Turn("GUIDE", _photo_blocks(approved, selected, next_flow["has_more"], clarification),
                approved, next_flow, analysis)


def _turn(message, flow, context, previous_pending, llm, knowledge, settings):
    protected = (knowledge_scope_reply(message, settings.openwebui_knowledge_ids)
                 or refusal_reply(message))
    if protected:
        return Turn("REPLY", [(protected, None)])
    intent = direct_intent(message)
    if previous_pending == FORM_ACTION:
        choice = _escalation_choice(message)
        if choice == "RESUME":
            return _resume_turn(flow, context, llm, knowledge)
        if choice == "FORM_REQUEST" or intent == "HUMAN":
            return Turn("FORM_REQUEST", [(FORM_HELP, None)])
        if intent == "FEEDBACK" or is_confirmation(message):
            return Turn("REPLY", [(FORM_HELP, None)])
        if _form_fields(message):
            details = _form(message)
            return Turn("FORM", details=details) if details else Turn("REPLY", [(FORM_HELP, None)])
    if _return_to_bot(message):
        if flow.get("source"):
            return Turn("REPLY", [("Продолжим с ботом. Напишите, что не получается на текущем шаге, "
                                    "или «Готово», если выполнили показанные действия.", None)])
        # A completed draft may already have been sent by the employee. Do not
        # infer its delivery status or retrieve knowledge for the menu command.
        return Turn("REPLY", [(BACK_TO_BOT, None)], flow={})
    if intent == "HUMAN":
        return Turn("HUMAN")
    if reply := _simple_social(message):
        return Turn("REPLY", [(reply, None)])
    if intent == "FEEDBACK":
        if message.strip().casefold().rstrip(".!?") == "помогло":
            return Turn("SUCCESS")
        if flow.get("source") and context.get("incident", {}).get("failed_attempts", 0) < MAX_ATTEMPTS - 1:
            search_flow = {"excluded": list(dict.fromkeys(flow.get("excluded", []) + [flow["source"]["file_id"]]))}
            try:
                following = _support_turn(flow.get("query") or message, search_flow, context, llm, knowledge)
            except KnowledgeUnavailable:
                return Turn("REPLY", [(UNAVAILABLE, None)])
            if following.action == "GUIDE":
                following.action = "RETRY_GUIDE"
                return following
        return Turn("FAILED")
    if is_confirmation(message) and not flow:
        return Turn("REPLY", [("Опишите, пожалуйста, с какой рабочей проблемой нужна помощь.", None)])
    pinned = (flow.get("source") and (is_confirmation(message) or is_ui_question(message)
                                      or _PHOTO_REQUEST.search(message) or _NEXT_PHOTO.search(message))) or (
        flow.get("clarification_query") and len(message.split()) <= 8
    )
    if not requires_support(message) and not pinned:
        try:
            routed = DialogueResult.model_validate(llm.dialogue(message, context))
            if routed.route == "REFUSE":
                from app.services.chat.dialogue import REFUSAL
                return Turn("REPLY", [(REFUSAL, None)])
            if routed.route == "SOCIAL" and (reply := safe_social_reply(routed.reply)):
                return Turn("REPLY", [(reply, None)])
        except Exception:
            pass
    if previous_pending == FORM_ACTION:
        details = _form(message)
        return Turn("FORM", details=details) if details else Turn("REPLY", [(FORM_HELP, None)])
    try:
        return _support_turn(message, flow, context, llm, knowledge)
    except KnowledgeUnavailable:
        problem_flow = None if flow.get("source") else {**flow, "clarification_query": _problem_query(message, flow)}
        return Turn("REPLY", [(UNAVAILABLE, None)], flow=problem_flow)


def _claim(factory, settings, account_id):
    with factory() as session, session.begin():
        statement = select(TalkInbox).join(TalkRoom).join(TalkAccount).where(
            TalkInbox.status.in_(["PENDING", "RUNNING", "FAILED"]), TalkRoom.enabled.is_(True),
            TalkAccount.enabled.is_(True),
            TalkAccount.server_url == settings.nextcloud_base_url.rstrip("/"),
        )
        statement = statement.where(TalkAccount.id == account_id) if account_id else statement.where(
            TalkAccount.username == getattr(settings, "nextcloud_username", "")
        )
        queued = session.scalars(statement.order_by(
            TalkInbox.remote_sequence, TalkInbox.created_at, TalkInbox.id
        ).with_for_update(skip_locked=True)).all()
        blocked = set()
        inbox = None
        for item in queued:
            if item.room_id not in blocked and item.status == "PENDING":
                room = session.get(TalkRoom, item.room_id)
                account = session.get(TalkAccount, room.account_id)
                principal = str(uuid5(NAMESPACE_URL, account.server_url + "/" + account.username + "/"
                                      + room.room_token + "/" + room.employee_id))
                uncertain = session.scalar(select(Delivery.id).join(Message).join(Incident).join(Conversation).where(
                    Delivery.room_token == room.room_token, Delivery.status != "SENT",
                    Conversation.user_id == principal,
                ).limit(1))
                if uncertain is None:
                    inbox = item
                    break
            blocked.add(item.room_id)
        if inbox is None:
            return None
        room = session.get(TalkRoom, inbox.room_id)
        account = session.get(TalkAccount, room.account_id)
        occurred = inbox.occurred_at.replace(tzinfo=timezone.utc) if inbox.occurred_at.tzinfo is None else inbox.occurred_at
        activated = account.activated_at.replace(tzinfo=timezone.utc) if account.activated_at.tzinfo is None else account.activated_at
        if inbox.sender_id != room.employee_id or inbox.sender_id == account.username or occurred < activated:
            inbox.status = "IGNORED"
            return {"ignored": True}
        principal = str(uuid5(NAMESPACE_URL, account.server_url + "/" + account.username + "/"
                              + room.room_token + "/" + room.employee_id))
        user = session.get(User, principal)
        if user is None:
            session.add(User(id=principal, external_id="talk-account:" + principal, display_name=room.employee_id[:255]))
            session.flush()
        conversation = session.scalar(select(Conversation).where(Conversation.user_id == principal))
        if conversation is None:
            conversation = Conversation(user_id=principal, channel="nextcloud_talk", room_token=room.room_token)
            session.add(conversation)
            session.flush()
        incident = session.get(Incident, room.incident_id) if room.incident_id else None
        if incident and incident.conversation_id != conversation.id:
            inbox.status, inbox.error_code = "FAILED", "IncidentOwnershipMismatch"
            return {"ignored": True}
        if incident is None or incident.state in TERMINAL_STATES:
            incident = Incident(conversation_id=conversation.id, state="DIAGNOSING", subject=inbox.content[:200], description="")
            session.add(incident)
            session.flush()
            room.incident_id, room.flow = incident.id, {}
        if incident.pending_action == "PROCESSING":
            interrupted = (room.flow or {}).get("_processing", {})
            if interrupted.get("inbox_id") != inbox.id:
                return None
            previous = interrupted.get("previous", {})
            incident.pending_action = previous.get("pending")
            incident.result = previous.get("result")
            room.flow = {key: value for key, value in (room.flow or {}).items() if key != "_processing"}
        previous = {"pending": incident.pending_action, "result": incident.result, "state": incident.state}
        inbox.status, inbox.tries, inbox.error_code = "RUNNING", inbox.tries + 1, None
        incident.pending_action, incident.version = "PROCESSING", incident.version + 1
        context = load_context(session, incident, settings)
        flow = copy.deepcopy(room.flow or {})
        room.flow = {**flow, "_processing": {"inbox_id": inbox.id, "tries": inbox.tries, "previous": previous}}
        return {"inbox_id": inbox.id, "room_id": room.id, "incident_id": incident.id, "version": incident.version,
                "message": inbox.content, "flow": flow, "previous": previous,
                "context": context, "principal": principal, "account_id": account.id, "tries": inbox.tries}


def _attempt(session, incident):
    return session.scalar(select(SolutionAttempt).where(
        SolutionAttempt.incident_id == incident.id, SolutionAttempt.status == "WAITING_FEEDBACK",
    ))


def _deliver(session, incident, blocks, source=None):
    # Root integration supplies this method so one captioned original photo stays
    # paired with its exact text in the durable Talk outbox.
    return messenger.send_blocks(session, incident.id, blocks, source=source)


def _offer_escalation(session, incident, reason, flow, context):
    pending = _attempt(session, incident)
    saved_flow = copy.deepcopy(flow)
    existing = saved_flow.pop("escalation", None)
    if existing and existing.get("resume"):
        resume = existing["resume"]
    else:
        if pending is None:
            # A failed/exhausted guide must not become a fresh waiting attempt
            # simply because the employee backs out of the support menu.
            problem = _last_problem(saved_flow, context)
            saved_flow = {"clarification_query": problem, "excluded": saved_flow.get("excluded", [])} if problem else {}
        resume = {"flow": saved_flow, "state": incident.state if pending else "DIAGNOSING",
                  "result": incident.result if pending else None}
    problem = _last_problem(resume["flow"], context)
    if problem:
        incident.description = problem[-8000:]
    incident.state, incident.pending_action = "DIAGNOSING", FORM_ACTION
    incident.result = "NO_KNOWLEDGE" if reason == "NO_KNOWLEDGE" else "ESCALATE"
    incident.escalation_reason = reason
    prefix = {
        "ATTEMPTS_EXHAUSTED": "Пять решений не помогли. Новые решения больше не предлагаются.",
        "NO_KNOWLEDGE": ("В базе знаний нет ответа на этот вопрос. Для этой проблемы у меня пока нет "
                         "утверждённой инструкции. Требуется помощь специалиста технической поддержки."),
        "NO_MORE_SOLUTIONS": "Других подходящих утверждённых решений нет.",
    }.get(reason)
    _deliver(session, incident, [((prefix + "\n\n" if prefix else "") + ESCALATION_MENU, None)])
    audit(session, incident.id, "incident.escalation_choice_requested", reason=reason)
    return {**flow, "escalation": {"stage": "CHOICE", "resume": resume}}


def _finish(factory, turn, claimed, llm, settings):
    with factory() as session, session.begin():
        inbox = session.scalar(select(TalkInbox).where(
            TalkInbox.id == claimed["inbox_id"],
        ).with_for_update())
        room = session.scalar(select(TalkRoom).where(TalkRoom.id == claimed["room_id"]).with_for_update())
        incident = session.scalar(select(Incident).where(Incident.id == claimed["incident_id"]).with_for_update())
        account = session.get(TalkAccount, room.account_id)
        if inbox.status != "RUNNING" or inbox.tries != claimed["tries"]:
            return
        if incident.version != claimed["version"] or incident.pending_action != "PROCESSING":
            inbox.status, inbox.error_code = "IGNORED", "StaleTurn"
            return
        if (not room.enabled or not account.enabled or room.account_id != claimed["account_id"]
                or account.server_url != settings.nextcloud_base_url.rstrip("/")
                or inbox.sender_id != room.employee_id):
            incident.pending_action = claimed["previous"]["pending"]
            room.flow = claimed["flow"]
            inbox.status, inbox.error_code = "IGNORED", "RoomScopeChanged"
            return
        incident.pending_action = claimed["previous"]["pending"]
        incident.result = claimed["previous"]["result"]
        room.flow = claimed["flow"]
        append_message(session, incident.id, "user", claimed["message"], "RECEIVED")
        pending = _attempt(session, incident)
        if turn.action in {"GUIDE", "RETRY_GUIDE"}:
            if turn.action == "RETRY_GUIDE" and pending:
                pending.status = "FAILED"
                session.add(Feedback(attempt_id=pending.id, helped=False, comment=""))
                incident.failed_attempts += 1
                session.flush()
                pending = None
            file_id = turn.approved.document["source"]["file_id"]
            same = bool(pending and pending.evidence.get("talk_source", {}).get("file_id") == file_id)
            if not same and pending:
                pending.status = "UNASSESSED"
                session.flush()
            attempts = session.scalars(select(SolutionAttempt).where(SolutionAttempt.incident_id == incident.id)).all()
            if not same and len(attempts) >= MAX_ATTEMPTS:
                room.flow = _offer_escalation(session, incident, "ATTEMPTS_EXHAUSTED", turn.flow, claimed["context"])
            else:
                save_snapshot(session, turn.approved.candidate)
                first = _deliver(session, incident, turn.blocks, source=turn.approved.document["source"])
                if not same:
                    candidate = turn.approved.candidate
                    session.add(SolutionAttempt(
                        incident_id=incident.id, solution_id=candidate.solution_id, ordinal=len(attempts) + 1,
                        message_id=first.id, evidence={"title": candidate.title, "version": candidate.version,
                            "chunk_id": candidate.chunk_id, "content": candidate.content, "steps": candidate.steps,
                            "source": candidate.source, "talk_source": turn.approved.document["source"]},
                        retrieval_score=candidate.score, llm_backend=llm.name,
                    ))
                incident.state, incident.pending_action, incident.result = "WAITING_FEEDBACK", None, "ANSWER"
                incident.escalation_reason = None
                if turn.analysis:
                    incident.problem_type, incident.equipment_model = turn.analysis.problem_type, turn.analysis.equipment_model
                if not same:
                    if claimed["flow"].get("clarification_query"):
                        # This report was already retained while retrieval was
                        # unavailable or a factual clarification was pending.
                        incident.description = turn.flow["query"][-8000:]
                    else:
                        incident.description = (incident.description + "\n" + turn.flow["query"]).strip()[-8000:]
                room.flow = turn.flow
        elif turn.action in {"SUCCESS", "FAILED"}:
            if not pending:
                _deliver(session, incident, [("Сначала опишите, что не работает, чтобы я мог подобрать инструкцию.", None)])
            else:
                helped = turn.action == "SUCCESS"
                pending.status = "SUCCEEDED" if helped else "FAILED"
                session.add(Feedback(attempt_id=pending.id, helped=helped, comment=""))
                if helped:
                    incident.state, incident.pending_action, incident.closed_at = "RESOLVED", None, now()
                    session.flush()
                    remember_resolution(session, incident, settings)
                    room.flow = {}
                    _deliver(session, incident, [("Спасибо за обратную связь. Рад, что удалось решить проблему.", None)])
                else:
                    incident.failed_attempts += 1
                    excluded = list(dict.fromkeys((room.flow or {}).get("excluded", []) + [pending.evidence["talk_source"]["file_id"]]))
                    room.flow = {**(room.flow or {}), "excluded": excluded, "has_more": False}
                    room.flow = _offer_escalation(session, incident,
                        "ATTEMPTS_EXHAUSTED" if incident.failed_attempts >= MAX_ATTEMPTS else "NO_MORE_SOLUTIONS",
                        room.flow, claimed["context"])
        elif turn.action == "HUMAN":
            room.flow = _offer_escalation(session, incident, "USER_REQUEST", turn.flow or room.flow, claimed["context"])
        elif turn.action == "NO_KNOWLEDGE":
            room.flow = _offer_escalation(session, incident, "NO_KNOWLEDGE", turn.flow or room.flow, claimed["context"])
        elif turn.action == "FORM_REQUEST":
            flow = copy.deepcopy(room.flow)
            escalation = flow.get("escalation", {})
            flow["escalation"] = {**escalation, "stage": "FORM"}
            room.flow = flow
            _deliver(session, incident, turn.blocks)
        elif turn.action == "RESUME":
            room.flow = turn.flow or {}
            incident.pending_action, incident.escalation_reason = None, None
            if pending and room.flow.get("source"):
                incident.state, incident.result = "WAITING_FEEDBACK", (turn.resume or {}).get("result") or "ANSWER"
            else:
                if pending:
                    pending.status = "UNASSESSED"
                incident.state, incident.result = "DIAGNOSING", None
            _deliver(session, incident, turn.blocks)
            audit(session, incident.id, "incident.escalation_declined")
        elif turn.action == "FORM":
            if pending:
                pending.status = "UNASSESSED"
            collect_details(session, claimed["principal"], incident.id, turn.details, settings)
            room.flow = {}
        else:
            _deliver(session, incident, turn.blocks)
            if turn.flow is not None:
                room.flow = turn.flow
                if problem := turn.flow.get("clarification_query"):
                    incident.description = problem[-8000:]
        incident.version += 1
        inbox.status, inbox.error_code = "DONE", None
        audit(session, incident.id, "talk.turn_completed", route=turn.action, block_count=len(turn.blocks))
        return True


def process_one(factory, settings, llm, knowledge=None, *, account_id=None):
    """Claim one private-room message, do reads outside transactions, commit one reply."""
    started = _clock()
    claimed = _claim(factory, settings, account_id)
    if claimed is None:
        return False
    if claimed.get("ignored"):
        return True
    metadata = {"message_id": claimed["inbox_id"],
                TIMING_MARKER: {"message_id": claimed["inbox_id"], "stages_ms": {}, "stage_calls": {}}}
    timed_llm = _TimedProxy(llm, metadata, {"dialogue": "model_api", "analyze_context": "model_api"})
    timed_knowledge = _TimedProxy(knowledge or TalkKnowledge(settings), metadata,
                                  {"search": "search", "read": "read", "is_current": "verify"})
    try:
        with _measure(metadata, "route"):
            turn = _turn(claimed["message"], claimed["flow"], claimed["context"],
                         claimed["previous"]["pending"], timed_llm, timed_knowledge, settings)
        with _measure(metadata, "persist"):
            finished = _finish(factory, turn, claimed, llm, settings)
    except Exception as error:
        with factory() as session, session.begin():
            inbox = session.get(TalkInbox, claimed["inbox_id"])
            incident = session.get(Incident, claimed["incident_id"])
            if (inbox.status == "RUNNING" and inbox.tries == claimed["tries"]
                    and incident.version == claimed["version"]):
                inbox.status = "PENDING" if inbox.tries < 3 else "FAILED"
                inbox.error_code = type(error).__name__
                incident.pending_action = claimed["previous"]["pending"]
                session.get(TalkRoom, claimed["room_id"]).flow = claimed["flow"]
        _report_timing(metadata, started, "FAILED", 0, type(error).__name__)
        return True
    _report_timing(metadata, started, turn.action if finished else "IGNORED", len(turn.blocks))
    return True
