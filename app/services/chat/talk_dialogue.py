"""Durable private Talk dialogue using the same approved steps/photos as WebUI."""

import copy
import hashlib
import re
from dataclasses import dataclass, field
from datetime import timezone
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy import func, or_, select
from sqlalchemy.exc import DBAPIError

from app.core.logging import event
from app.models import (
    AuditEvent, Conversation, Delivery, Feedback, Incident, Message, SolutionAttempt, User, now,
)
from app.models.talk import TalkAccount, TalkInbox, TalkRoom
from app.schemas.contracts import AnalysisResult, DialogueResult, EscalationDetails, InstructionOfferReply
from app.services.chat.audit import audit
from app.services.chat.channel_profile import channel_profile, remember_computer
from app.services.chat.dialogue import _OPERATION_CLAIM, _TECHNICAL_OUTPUT
from app.services.nextcloud.progress import start_progress
from app.services.nextcloud.interruption import (
    TERMINAL_DELIVERIES, has_unfinished_guide, is_resume_request, latest_paused, pause_due_to_input,
    record_response, resume_response,
)
from app.services.chat.memory import _redact, load_context, remember_resolution
from app.services.chat.messages import append_message
from app.services.chat.policies import FORM_ACTION, MAX_ATTEMPTS, TERMINAL_STATES, direct_intent
from app.services.chat.service import messenger
from app.services.rag.openwebui import save_snapshot
from app.services.rag.query import normalized_query, printer_kinds, scanner_kinds
from app.services.rag.talk import (
    ApprovedTalkDocument, INSTRUCTION_SELECTION_TASK, InstructionSelection, KnowledgeUnavailable,
    SourceSelection, TalkKnowledge, _SELECTION_TASK, report_failure,
)
from app.services.rag.success import SuccessfulTalkKnowledge, remember_success
from app.services.ticketing.intake import _ordered_lines, parse_intake
from app.services.ticketing.service import collect_details
from integrations.openwebui.automatic_instruction_images import (
    TIMING_MARKER, _clock, _measure, _milliseconds,
    applicable_images, clarification_images, instruction_blocks, is_confirmation,
    is_ui_question, knowledge_scope_reply, refusal_reply,
    requires_support, safe_social_reply,
)

UNAVAILABLE = "Сейчас не удалось проверить базу знаний. Повторите вопрос немного позже или напишите «оператор»."
CHANGED = "Инструкция изменилась или доступ к ней недоступен. Продолжение приостановлено до проверки актуального источника."
ESCALATION_MENU = ("Как продолжим?\n1 — оформить письмо в техническую поддержку.\n"
                   "2 — продолжить диалог с ботом.\n\nМожно ответить цифрой или словами. Заявка пока не отправлена.")
OFFER_CHOICE = "Какой вариант подходит? Напишите его номер или «нет»."
RESUME_GUIDE = ("Продолжим с ботом с того же шага. Заявка не отправлена. "
                "Напишите, что не получается, или «Готово», если выполнили показанные действия.")
BACK_TO_BOT = "Продолжим с ботом. Опишите, с каким вопросом нужна помощь."
PHOTO_NOTICE = ("Сообщение с фото получено. Распознавание изображения ещё не подключено; "
                "я учитываю только текст. Фото к письму пока нужно прикрепить вручную.")
MEDIA_ERRORS = {
    "unsupported_inbound_media_type": "Формат вложения пока не поддерживается.",
    "inbound_media_too_large": "Вложение слишком большое для обработки.",
    "inbound_media_too_many": "В одном сообщении слишком много вложений.",
    "inbound_caption_too_long": "Подпись к вложению слишком длинная. Отправьте описание отдельным текстом.",
}
INTAKE_QUESTIONS = {
    "account_holder_name": "Как вас зовут?",
    "computer_name": "Как называется ваш рабочий компьютер?",
    "subject": "Укажите тему письма.",
    "additional_info": "Опишите проблему: что происходит и какая помощь нужна.",
}
CLARIFY_QUESTIONS = {
    "DEVICE_TYPE": "О каком типе оборудования идёт речь — сканер штрихкодов, сканер документов, принтер документов А4 или термопринтер?",
    "DEVICE_MODEL": "О каком типе оборудования идёт речь — сканер штрихкодов, сканер документов, принтер документов А4 или термопринтер?",
    "CONNECTION": "Как устройство подключено: кабелем USB к компьютеру, сетевым кабелем или по Wi-Fi?",
    "ERROR_TEXT": "Какой точный текст ошибки вы видите? Перепишите его, пожалуйста.",
    "SYMPTOMS": "Опишите, что именно не работает и что появляется на экране.",
    "SCANNER_KIND": "Вы имеете в виду сканер штрихкодов или сканер документов (например, планшетный)?",
    "SCANNER_GOAL": "Что именно нужно настроить у сканера — работу в программе, громкость, подсветку или другое? Напишите желаемый результат своими словами.",
    "SCANNER_SYMPTOMS": "Что происходит при сканировании: код не считывается, считывается неверно или не появляется в нужной программе?",
    "PRINTER_KIND": "Это принтер документов А4, стационарный термопринтер для этикеток или мобильный термопринтер?",
    "PRINTER_GOAL": "Что нужно сделать с принтером — подключить его, изменить настройку печати или решить конкретную проблему?",
    "PRINTER_SYMPTOMS": "Что происходит при печати — задание не печатается, печатается неверно или нужна другая настройка?",
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
    confirmed_computer: str | None = None
    media_notice: str | None = None


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
              block_count=block_count, error_code=error_code, cache_hit=bool(metadata.get("cache_hit")))
    except Exception:
        pass  # Diagnostics never change the stored employee reply.


def _prefill_intake(flow, context):
    updated = copy.deepcopy(flow)
    escalation = updated.get("escalation", {})
    fields = copy.deepcopy(escalation.get("intake_fields", {}))
    if not escalation.get("profile_prefilled"):
        profile = context.get("channel_profile", {})
        for key in ("account_holder_name", "computer_name"):
            value = profile.get(key)
            if key not in fields and isinstance(value, str) and value.strip():
                fields[key] = value.strip()
        escalation = {**escalation, "profile_prefilled": True}
        if not fields.get("computer_name"):
            escalation["intake_requested_field"] = "computer_name"
    if not fields.get("account_login"):
        if login := _internal_login(context):
            fields["account_login"] = login
    updated["escalation"] = {**escalation, "intake_fields": fields}
    return updated


def _internal_login(context):
    """Keep the old required schema field internal; never infer a Windows login."""
    uid = context.get("channel_profile", {}).get("account_login")
    if not isinstance(uid, str) or not uid:
        return None
    if len(uid) <= 100 and re.fullmatch(r"[A-Za-zА-Яа-яЁё0-9_][A-Za-zА-Яа-яЁё0-9_.@+\\-]*", uid):
        return uid
    return "nc_" + hashlib.sha256(uid.encode("utf-8")).hexdigest()


def _profile_form_request(flow, context):
    updated = _prefill_intake(flow, context)
    fields = updated["escalation"]["intake_fields"]
    prompted = _intake_prompt(updated)
    card = ("ФИО: " + fields.get("account_holder_name", "Не определено") + "\n"
            + "Компьютер: " + fields.get("computer_name", "Не определён"))
    text = (card + "\n\n" + prompted.blocks[0][0]
            + "\n\nЕсли данные неверны, напишите исправление. Чтобы вернуться к боту, напишите «Отмена» или «2».")
    return Turn("FORM_REQUEST", [(text, None)], flow=prompted.flow)


def _intake_turn(message, flow, llm, context):
    updated = _prefill_intake(flow, context)
    escalation = updated["escalation"]
    fields = copy.deepcopy(escalation["intake_fields"])
    requested = escalation.get("intake_requested_field")
    if fields.get("subject") and not fields.get("additional_info", "").strip() and requested in {None, "subject"}:
        requested = "additional_info"  # Upgrade a partially collected pre-0.3.5 form.
    result = parse_intake(message, fields, llm, requested_field=requested, require_description=True)
    collected, missing_fields, details = dict(result.fields), list(result.missing), result.details
    if "account_login" not in collected and (login := _internal_login(context)):
        collected["account_login"] = login
        missing_fields = [key for key in missing_fields if key != "account_login"]
        if not missing_fields:
            details = EscalationDetails.model_validate(collected)
    computer = collected.get("computer_name")
    current_computer = context.get("channel_profile", {}).get("computer_name")
    confirmed = computer if computer and (computer != fields.get("computer_name") or computer != current_computer) else None
    if details is not None and details.additional_info.strip():
        return Turn("FORM", details=details, confirmed_computer=confirmed)
    if not collected.get("additional_info", "").strip() and "additional_info" not in missing_fields:
        missing_fields.append("additional_info")
    missing = next((key for key in missing_fields if key in INTAKE_QUESTIONS), None)
    updated["escalation"] = {**escalation, "stage": "FORM", "intake_fields": collected,
                             "intake_requested_field": missing}
    question = INTAKE_QUESTIONS[missing] if missing else "В чём нужна помощь? Кратко опишите проблему."
    if missing == "additional_info" and len(message.strip()) > 2000:
        question += " Описание должно быть не длиннее 2000 символов."
    return Turn("REPLY", [(question, None)], flow=updated, confirmed_computer=confirmed)


def _intake_prompt(flow):
    updated = copy.deepcopy(flow)
    escalation = updated.get("escalation", {})
    fields = escalation.get("intake_fields", {})
    missing = escalation.get("intake_requested_field")
    if missing not in INTAKE_QUESTIONS or fields.get(missing):
        missing = next((key for key in INTAKE_QUESTIONS if not fields.get(key, "").strip()), "additional_info")
    updated["escalation"] = {**escalation, "stage": "FORM", "intake_requested_field": missing}
    return Turn("REPLY", [(INTAKE_QUESTIONS[missing], None)], flow=updated)


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


def _feedback_kind(message, flow=None):
    """Accept an explicit reported outcome, never infer it from a short 'yes'."""
    if "?" in message:
        return None
    normalized = _normalized(message).replace("ё", "е")
    normalized = re.sub(r"^(?:(?:да|ну|ура|спасибо|благодарю|отлично|вот)\s+)+", "", normalized)
    normalized = re.sub(r"\s+(?:спасибо(?:\s+большое)?|благодарю)(?:\s+вам)?$", "", normalized)
    positive = {
        "помогло", "да помогло", "помогло спасибо", "спасибо помогло", "заработало",
        "все заработало", "ура все заработало", "спасибо все заработало", "все работает",
        "теперь все работает", "теперь работает", "проблема решена", "проблема устранена",
        "получилось все работает",
    }
    negative = {
        "не помогло", "нет не помогло", "не сработало", "пока не работает", "все еще не работает",
        "ничего не изменилось", "проблема осталась", "все равно не работает", "не заработало",
        "сделал но не помогло", "сделала но не помогло", "попробовал не помогло",
        "попробовала не помогло",
    }
    if normalized in positive:
        return "SUCCESS"
    if normalized in negative:
        return "FAILED"
    problem = str((flow or {}).get("query", "")).casefold()
    printer = bool(re.search(r"\b(?:принтер\w*|печат\w*|pdf|acrobat)\b", problem))
    scanner = bool(re.search(r"\b(?:сканер\w*|сканир\w*|штрихкод\w*)\b", problem))
    if normalized in {"теперь печатает", "принтер теперь печатает", "теперь принтер печатает"}:
        return "SUCCESS" if printer and not scanner else None
    if normalized in {"теперь сканирует", "сканер теперь работает", "теперь сканер работает"}:
        return "SUCCESS" if scanner and not printer else None
    if normalized in {"принтер все равно не печатает", "все равно не печатает", "принтер по прежнему не печатает"}:
        return "FAILED" if printer and not scanner else None
    if normalized in {"сканер все равно не работает", "сканер по прежнему не работает", "все равно не сканирует"}:
        return "FAILED" if scanner and not printer else None
    return None


def _step_confirmed(message):
    normalized = _normalized(message).replace("ё", "е")
    return is_confirmation(message) or normalized in {
        "этот шаг сделал", "этот шаг сделала", "с этим шагом закончил", "с этим шагом закончила",
        "сделал что дальше", "сделала что дальше", "сделал это что дальше", "сделала это что дальше",
        "выполнил что дальше", "выполнила что дальше", "можно дальше", "можем продолжать",
        "давайте дальше", "готов можно дальше", "готово что дальше", "все сделал что дальше",
    }


def _current_step_question(message):
    """Conversational difficulty points back to source text/photos, not invented alternatives."""
    return is_ui_question(message) or _normalized(message) in {
        "не получилось", "не получается", "не смог", "не смогла", "не разобрался", "не разобралась",
        "застрял", "застряла", "я застрял", "я застряла",
    } or bool(re.search(
        r"(?:застрял(?:а)?\s+(?:на\s+)?(?:этом|первом|втором|текущем)\b|"
        r"не\s+(?:разобрался|разобралась|понял(?:а)?|наш[её]л(?:а)?)|куда\s+(?:здесь\s+|тут\s+)?нажать|"
        r"(?:объясни(?:те)?|расскажи(?:те)?)\s+(?:это\s+|пожалуйста\s+)?(?:проще|попроще|понятнее|подробнее)|"
        r"(?:можно|нужно)\s+подробнее|(?:такого|такой|этого|этой)\s+(?:пункта|кнопки|галочки|меню)\s+нет|"
        r"(?:пункт|кнопка|галочка)\s+(?:у\s+меня\s+)?(?:серая|недоступна|неактивна)|"
        r"нету\s+(?:такого|такой|этого|этой|кнопки|пункта|меню))", message.casefold(),
    ))


def _technical_message(message):
    canonical = normalized_query(message)
    return bool(requires_support(message) or requires_support(canonical)
                or re.search(r"\b(?:fortigate|forticlient|bluetooth|термопринтер\w*)\b", canonical))


def _free_support_request(message):
    # The CHOICE menu is not the form yet. A fresh work question is a request
    # to keep talking, while explicitly labelled ticket fields stay data.
    if (re.search(r"(?:^|\n)\s*(?:фио|логин|компьютер|тема|описание)\s*:", message, re.I)
            or re.search(r"\b(?:меня\s+зовут|мой\s+логин|логин|фио)\b", message, re.I)
            or _ordered_lines(message) is not None):
        return False
    return _technical_message(message) and bool(re.search(
        r"(?:^|[.!?]\s*)(?:как\b|почему\b|где\b|что\b|помоги\w*\b|подскажи\w*\b)|"
        r"\b(?:не\s+(?:работает|печатает|сканирует|подключается)|нужно\s+(?:подключить|настроить))\b",
        message.casefold(),
    ))


def _structured_letter(message):
    identity_reply = (not re.match(r"\s*(?:как|почему|где|зачем|когда|кто|что|а\s+если|другой\s+вопрос)\b", message, re.I)
                      and bool(re.search(r"\b(?:меня\s+зовут|(?:мой|рабочий)\s+логин)\b", message, re.I)))
    return (_ordered_lines(message) is not None or bool(re.search(
        r"(?:^|\n)\s*(?:фио|логин|компьютер|тема|описание)\s*:", message, re.I,
    )) or identity_reply)


def _positive_connections(message):
    words = normalized_query(message).split()
    return {word for index, word in enumerate(words) if word in {"wifi", "bluetooth", "usb", "ethernet"}
            and not {"не", "без"}.intersection(words[max(0, index - 3):index])}


def _without_previous_connection(previous, message):
    if not _positive_connections(message) or not _positive_connections(previous):
        return previous
    words = normalized_query(previous).split()
    return " ".join(word for index, word in enumerate(words)
                    if word not in {"wifi", "bluetooth", "usb", "ethernet"}
                    or {"не", "без"}.intersection(words[max(0, index - 3):index]))


def _form_interrupted(message, flow):
    if _structured_letter(message):
        return False
    text = normalized_query(message)
    if re.match(r"^(?:а\s+если|а\s+как|а\s+можно|другой\s+вопрос|другая\s+проблема|"
                r"подожди\w*|погоди\w*|кстати|не\s+хочу\s+оформлять|"
                r"давай\w*\s+(?:пока\s+)?(?:без\s+заявки|продолжим))\b", text):
        return True
    requested = flow.get("escalation", {}).get("intake_requested_field")
    if requested in {"subject", "additional_info"}:
        # The employee can deliberately write a question as the subject/body.
        return False
    return bool(_free_support_request(message) or _positive_connections(message)
                or re.match(r"^(?:как|почему|где|зачем|когда|кто|что|расскажи\w*|подскажи\w*)\b", text))


def _continue_from_letter(message, flow, context, llm, knowledge, settings):
    """A work question resumes support; unfinished personal form fields stay out of retrieval."""
    saved = flow.get("escalation", {}).get("resume", {})
    restored = copy.deepcopy(saved.get("flow", {key: value for key, value in flow.items() if key != "escalation"}))
    text = normalized_query(message)
    if text in {"да", "нет", "готово", "понятно", "ок", "хорошо", "не знаю"}:
        return Turn("RESUME", [(BACK_TO_BOT, None)], flow=restored, resume=saved)
    if not (restored.get("source") and _current_step_question(message)):
        prior = _last_problem(restored, context)
        refinement = bool(_positive_connections(message)) and bool(re.match(
            r"^(?:а\s+(?:если|по|через|как)|по\b|через\b|подключ\w*\s+по\b)", text,
        ))
        if prior and refinement:
            # A later explicit transport supersedes the previous choice. Never
            # require Bluetooth and Wi-Fi simultaneously after "а если по Wi-Fi".
            prior = _without_previous_connection(prior, message)
            restored = {"clarification_query": prior, "excluded": []}
        else:
            restored = {}
    try:
        following = _text_turn(message, restored, context, None, llm, knowledge, settings)
    except KnowledgeUnavailable:
        following = Turn("REPLY", [(UNAVAILABLE, None)], flow={
            "clarification_query": _problem_query(message, restored), "excluded": restored.get("excluded", []),
        })
    if following.action in {"NO_KNOWLEDGE", "HUMAN"}:
        return Turn("RESUME", [("В базе знаний пока нет подходящей утверждённой инструкции для этого вопроса. "
                                "Можем продолжить разбираться; письмо в поддержку не оформляется.", None)],
                    flow=following.flow or {"clarification_query": _problem_query(message, restored)}, resume={})
    if following.action == "REPLY":
        following.action, following.resume = "RESUME", {}
        following.flow = following.flow if following.flow is not None else restored
    return following


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
                      "оформить письмо", "оформить письмо в поддержку", "письмо в поддержку",
                      "хочу написать тп", "хочу написать в тп", "хочу написать в поддержку", "написать в тп"}:
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
    # Current reports are kept explicitly in flow. Recent conversation messages
    # can be intake data, so they are not a fallback source for retrieval queries.
    subject = context.get("incident", {}).get("subject")
    return subject if isinstance(subject, str) and _technical_message(subject) and not control(subject) else ""


def _problem_query(message, flow):
    previous = flow.get("clarification_query")
    if re.match(r"\s*(?:как|почему|зачем|когда|кто|что\s+делать|другой\s+вопрос|другая\s+проблема)\b", message, re.I):
        return message[:6000]
    answer = normalized_query(message)
    keys = flow.get("clarification_keys", [])
    if "PRINTER_KIND" in keys and len(answer.split()) <= 3:
        if re.match(r"^(?:мобильн\w*|переносн\w*|портативн\w*)$", answer):
            message += "\nМобильный термопринтер"
        elif re.match(r"^(?:стационарн\w*|настольн\w*)$", answer):
            message += "\nСтационарный термопринтер"
        elif answer in {"а4", "a4", "документов", "для документов"}:
            message += "\nПринтер документов А4"
    if "SCANNER_KIND" in keys and answer in {"документов", "документы", "планшетный", "планшетный сканер"}:
        message += "\nСканер документов"
    if flow.get("clarification_keys") and any(key.startswith("SCANNER_") for key in flow["clarification_keys"]):
        latest = normalized_query(message)
        if (re.search(r"\b(?:принтер\w*|тсд|nextcloud|fortigate|forticlient|outlook|почт\w*)\b", latest)
                and not re.search(r"\b(?:сканер\w*|штрих\w*|баркод\w*|шк)\b", latest)):
            return message[:6000]
    if flow.get("clarification_keys") and any(key.startswith("PRINTER_") for key in flow["clarification_keys"]):
        latest = normalized_query(message)
        if (re.search(r"\b(?:сканер\w*|штрих\w*|баркод\w*|шк|nextcloud|fortigate|forticlient|outlook|почт\w*)\b", latest)
                and not re.search(r"\b(?:(?:термо)?принтер\w*|печат\w*)\b", latest)):
            return message[:6000]
    if previous:
        previous = _without_previous_connection(previous, message)
    factual_answer = (flow.get("clarification_keys") and len(message) <= 2000
                      and not re.match(r"\s*(?:как|почему|зачем|что\s+делать|другая\s+проблема)\b", message, re.I))
    if previous and message != previous and (len(message.split()) <= 8 or factual_answer):
        return (previous + "\n" + message)[:6000]
    return message[:6000]


def _resume_turn(flow, context, llm, knowledge):
    saved = flow.get("escalation", {}).get("resume", {})
    restored = copy.deepcopy(saved.get("flow", {key: value for key, value in flow.items() if key != "escalation"}))
    if restored.get("declined_instruction_offer"):
        return Turn("RESUME", [(BACK_TO_BOT, None)], flow={}, resume={})
    # An explicit empty problem is significant: personal data collected after
    # the menu opened must never become a knowledge-search query on cancellation.
    if "query" in saved:
        query = saved["query"]
    elif restored.get("query") or restored.get("clarification_query"):
        query = _last_problem(restored, context)
    else:
        # Older pending forms have no resume query. Their original subject is
        # safer than recent messages that may already contain intake data.
        query = _last_problem({}, {"incident": context.get("incident", {})})
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


def _scanner_inventory(knowledge):
    try:
        inventory = getattr(knowledge, "scanner_context", lambda: {})()
    except Exception:
        return []
    if not isinstance(inventory, dict) or inventory.get("status") != "ok":
        return []
    sources = inventory.get("sources")
    return sources[:30] if isinstance(sources, list) else []


def _scanner_model_kind(text, sources):
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("models"), list):
            continue
        for model in source["models"][:20]:
            if not isinstance(model, str) or not 2 <= len(model) <= 80:
                continue
            normalized = normalized_query(model)
            # A short code must be an entire token, never a substring of a
            # different model or a document reference.
            identifiers = [normalized] + [word for word in normalized.split()
                                          if re.search(r"[a-zа-я]", word) and re.search(r"\d", word)]
            if any(value and re.search(r"(?<!\w)" + re.escape(value) + r"(?!\w)", text) for value in identifiers):
                kinds = source.get("categories", [])
                if isinstance(kinds, list) and len(kinds) == 1 and kinds[0] in {"barcode", "document"}:
                    return kinds[0]
    return None


def _scanner_clarification(message, flow, knowledge):
    """A related file is not proof that the employee wants its entire procedure.

    Ask one decision at a time before retrieval can pin a generic scanner report
    to a program-specific guide. Device names never come from model knowledge.
    """
    latest = normalized_query(message)
    problem = _problem_query(message, flow)
    correction = bool(re.search(r"\b(?:уже|давно)\s+подключ\w*\b", latest)
                      and re.search(r"\b(?:настро\w*|настрой\w*)\b", latest))
    if (correction and not re.search(r"\b(?:сканер\w*|штрих\w*|баркод\w*|шк)\b", latest)
            and not re.search(r"\b(?:принтер\w*|тсд|nextcloud|fortigate|forticlient)\b", latest)):
        prior = normalized_query(flow.get("query", ""))
        if re.search(r"\b(?:сканер\w*|штрих\w*|баркод\w*|шк)\b", prior):
            device = "Сканер штрихкодов" if re.search(r"\b(?:штрих\w*|баркод\w*|шк)\b", prior) else "Сканер"
            problem = device + "\n" + message
    text = normalized_query(problem)
    scanner = bool(re.search(r"\b(?:сканер\w*|штрих\w*|баркод\w*|шк)\b", text))
    if not scanner:
        return None
    barcode = bool(re.search(r"\b(?:штрих\w*|баркод\w*|шк)\b", text))
    document = bool(re.search(r"\b(?:документ\w*|мфу|планшетн\w*)\b", text))
    program = bool(re.search(r"\b(?:1[сc]|адрес)\b", text))
    concrete = bool(re.search(
        r"\b(?:громк\w*|звук\w*|звуков\w*|сигнал\w*|суффикс\w*|префикс\w*|"
        r"раскладк\w*|кодировк\w*|подсвет\w*|луч\w*|ярк\w*|базов\w*|enter|перенос\w*|красн\w*|мига\w*|морга\w*|"
        r"не\s+(?:читает|считывает|сканирует|передает|появляется)|неверн\w*|ошибк\w*)\b", text,
    ))
    if not barcode and not document and not program and re.search(r"\b[a-zа-я]+[-_]?\d+\w*\b", text):
        kind = _scanner_model_kind(text, _scanner_inventory(knowledge))
        barcode, document = kind == "barcode", kind == "document"
    # A correction is about a different desired result, even if the old source
    # happened to be a guide to adding the scanner in 1C.
    if correction and not re.search(
        r"\b(?:1[сc]|адрес|громк\w*|звук\w*|подсвет\w*|луч\w*|ярк\w*|базов\w*|суффикс\w*|префикс\w*|раскладк\w*|кодировк\w*)\b", latest,
    ):
        key = "SCANNER_GOAL"
    elif not barcode and not document and not program:
        key = "SCANNER_KIND"
    elif barcode and not program and not concrete:
        key = "SCANNER_GOAL" if re.search(r"\b(?:настро\w*|настрой\w*)\b", text) else "SCANNER_SYMPTOMS"
    else:
        return None
    asked = list(flow.get("clarification_keys", []))
    if key in asked:
        # Repeating an uninformative reply does not authorize an arbitrary
        # source, and must not cause the same broad question to loop forever.
        question = "Коротко опишите, что хотите изменить или что происходит при работе со сканером. Можно привести пример."
    else:
        question = CLARIFY_QUESTIONS[key]
        asked.append(key)
    return Turn("REPLY", [(question, None)], flow={
        "clarification_query": problem, "clarification_keys": asked,
        "excluded": flow.get("excluded", []),
    })


def _printer_clarification(message, flow):
    problem = _problem_query(message, flow)
    text = normalized_query(problem)
    if not re.search(r"\b(?:(?:термо)?принтер\w*|печат\w*)\b", text):
        return None
    # A named application, model, or precise operation already constrains the
    # source. Keep those established paths fast and ask only for a missing type.
    concrete = re.search(
        r"\b(?:pdf|acrobat|outlook|очеред\w*|калибр\w*|риббон\w*|картридж\w*|"
        r"полос\w*|блед\w*|тонер\w*|зажев\w*|замят\w*|застря\w*|bluetooth|wifi|usb|ethernet|тсд|"
        r"[a-zа-я]+[-_]?\d{2,}[a-zа-я0-9_-]*)\b", text,
    )
    kind = printer_kinds(problem)
    if concrete:
        return None
    asked = list(flow.get("clarification_keys", []))
    if kind:
        if re.search(r"\b(?:настро\w*|настрой\w*)\b", text):
            key = "PRINTER_GOAL"
        elif re.search(r"\bне\s+работает\b", text):
            key = "PRINTER_SYMPTOMS"
        else:
            return None
    elif "PRINTER_KIND" in asked:
        key = "PRINTER_SYMPTOMS"
    else:
        key = "PRINTER_KIND"
    if key in asked:
        return None
    question = CLARIFY_QUESTIONS[key]
    if key == "PRINTER_KIND" and re.search(r"\bтермопринтер\w*\b", text):
        question = "Это стационарный термопринтер для этикеток или мобильный термопринтер?"
    return Turn("REPLY", [(question, None)], flow={
        "clarification_query": problem, "clarification_keys": asked + [key],
        "excluded": flow.get("excluded", []),
    })


def _analysis_clarification(analysis, problem, flow):
    if analysis is None or analysis.clarification_key is None:
        return None
    key = "DEVICE_TYPE" if analysis.clarification_key == "DEVICE_MODEL" else analysis.clarification_key
    asked = list(flow.get("clarification_keys", []))
    already_asked = key in asked or key == "DEVICE_TYPE" and any(
        known in asked for known in ("DEVICE_MODEL", "SCANNER_KIND", "PRINTER_KIND")
    )
    typed = bool(scanner_kinds(problem) or printer_kinds(problem))
    scanner_program = bool(re.search(r"\bсканер\w*\b", normalized_query(problem))
                           and re.search(r"\b(?:1[сc]|адрес)\b", normalized_query(problem)))
    if already_asked or key == "DEVICE_TYPE" and (typed or scanner_program):
        return None
    return Turn("REPLY", [(CLARIFY_QUESTIONS[key], None)], flow={
        "clarification_query": problem, "clarification_keys": asked + [key],
        "excluded": flow.get("excluded", []),
    })


def _different_instruction(message):
    return bool(re.search(
        r"(?:\b(?:инструкци\w*|решени\w*|ответ)\s+(?:мне\s+)?не\s+подход\w*\b|"
        r"\b(?:неправильн\w*|неподходящ\w*)\s+инструкци\w*\b|"
        r"\bинструкци\w*\s+(?:неправильн\w*|неподходящ\w*)\b|"
        r"\b(?:не\s+(?:та|эта|подходящая)\s+инструкци\w*|(?:нужна|нужен|покажи\w*|дай\w*)\s+"
        r"друг\w*\s+(?:инструкци\w*|решени\w*|ответ))\b)", message.casefold(),
    ))


def _replace_guide_context(flow):
    previous = flow.get("query", "")
    excluded = list(dict.fromkeys(flow.get("excluded", []) + [flow["source"]["file_id"]]))
    if "barcode" in scanner_kinds(previous):
        context = "Настройка сканера штрихкодов"
    elif "document" in scanner_kinds(previous):
        context = "Настройка сканера документов"
    elif re.search(r"\bсканер\w*\b", normalized_query(previous)):
        context = "Настройка сканера"
    elif printer_kinds(previous) == {"thermal_mobile"}:
        context = "Настройка мобильного термопринтера"
    elif printer_kinds(previous) == {"thermal_stationary"}:
        context = "Настройка стационарного термопринтера"
    elif printer_kinds(previous) == {"document"}:
        context = "Настройка принтера документов А4"
    elif re.search(r"\b(?:(?:термо)?принтер\w*|печат\w*)\b", normalized_query(previous)):
        context = "Настройка принтера"
    else:
        context = ""
    return {"clarification_query": context, "excluded": excluded}


def _model_clarification(question):
    """Only a short informational question can leave the source-selection task."""
    question = question.strip() if isinstance(question, str) else ""
    if (not 5 <= len(question) <= 300 or question.count("?") != 1 or not question.endswith("?")
            or re.search(r"[\r\n<>]|(?:mailto:|file:|data:|javascript:|www\.)", question, re.I)
            or _TECHNICAL_OUTPUT.search(question) or _OPERATION_CLAIM.search(question)
            or _redact(question) != question or refusal_reply(question)
            or re.search(r"\b(?:в\s+(?:нашей\s+)?базе\s+знан\w*|наш[её]л\w*|"
                         r"найден\w*\s+инструкц\w*|есть\s+(?:похож\w*\s+)?инструкц\w*)\b", question, re.I)
            or re.search(r"\b(?:парол\w*|password\w*|токен\w*|api[ _-]?key|secret\w*|"
                         r"модел\w*|маркиров\w*|шильдик\w*|производител\w*|"
                         r"отключите|сбросьте|переустановите)\b", question, re.I)):
        error = KnowledgeUnavailable("InvalidClarification", "instruction_selection")
        report_failure(error)
        raise error
    return question


def _support_turn(message, flow, context, llm, knowledge):
    has_guide = bool(flow.get("source"))
    if has_guide and _different_instruction(message):
        replacement = _replace_guide_context(flow)
        message = "\n".join(part for part in (replacement["clarification_query"], message) if part)
        flow = {"excluded": replacement["excluded"]}
        has_guide = False
    photo_request = bool(_PHOTO_REQUEST.search(message) or _NEXT_PHOTO.search(message))
    next_photo = bool(photo_request and _NEXT_PHOTO.search(message))
    clarification = bool(has_guide and (_current_step_question(message) or photo_request and not next_photo))
    continuation = bool(has_guide and (_step_confirmed(message) or next_photo))
    model_selection = bool(getattr(knowledge, "uses_llm_selection", False))
    if not clarification and not continuation and not model_selection:
        if followup := _scanner_clarification(message, flow, knowledge):
            return followup
        if followup := _printer_clarification(message, flow):
            return followup
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
        if model_selection:
            selection_context = {**context, "current_message": message,
                                 "previous_question": flow.get("clarification_question", "")}
            source = flow.get("source")
            if (has_guide and isinstance(source, dict)
                    and all(isinstance(source.get(name), str) and source[name]
                            for name in ("file_id", "filename"))):
                selection_context["active_task"] = flow.get("query", "")
                selection_context["shown_source"] = {name: source[name] for name in ("file_id", "filename")}
            decision = knowledge.search_with_llm(problem, context=selection_context,
                                                 excluded=excluded, fallback_query=problem)
            if decision.result == "CLARIFY":
                question = _model_clarification(decision.question)
                return Turn("REPLY", [(question, None)], flow={
                    "clarification_query": problem, "clarification_keys": ["MODEL"],
                    "clarification_question": question, "excluded": excluded,
                })
            if decision.result == "NO_KNOWLEDGE":
                return Turn("NO_KNOWLEDGE", flow={"clarification_query": problem,
                                                  "clarification_keys": flow.get("clarification_keys", []),
                                                  "excluded": excluded})
            if decision.result == "RELATED":
                choices = [copy.deepcopy(item.document["source"]) for item in decision.alternatives]
                if not choices or not all(knowledge.is_current(source) for source in choices):
                    raise KnowledgeUnavailable("InvalidInstructionOffer", "instruction_selection")
                text = "В нашей базе есть похожие инструкции. Возможно, вы имели в виду один из этих вариантов?\n\n"
                text += "\n".join(f"{index}. {source['filename']}" for index, source in enumerate(choices, 1))
                text += "\n\nНапишите номер подходящего варианта или «нет», если они вам не подходят."
                return Turn("REPLY", [(text, None)], flow={
                    "instruction_offer": {"problem": problem, "choices": choices},
                    "clarification_query": problem, "excluded": excluded,
                })
            if decision.result != "MATCH" or decision.approved is None:
                raise KnowledgeUnavailable("InvalidInstructionDecision", "instruction_selection")
            approved = decision.approved
        else:
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
                    analysis = None
        if approved is None:
            asked = flow.get("clarification_keys", [])
            if followup := _analysis_clarification(analysis, problem, flow):
                return followup
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
        if retrieval_key := getattr(knowledge, "last_cache_key", None):
            next_flow["retrieval_key"] = retrieval_key
    return _guide_reply(approved, selected, next_flow, knowledge, clarification, analysis)


def _guide_reply(approved, selected, next_flow, knowledge, clarification=False, analysis=None):
    if not knowledge.is_current(approved.document["source"]):
        return Turn("REPLY", [(CHANGED, None)])
    if not selected:
        text = approved.document["content"].split("Фотографии и пояснения", 1)[0].strip()
        return Turn("GUIDE", [(text + "\n\nУдалось выполнить эти действия?", None)],
                    approved, next_flow, analysis)
    return Turn("GUIDE", _photo_blocks(approved, selected, next_flow["has_more"], clarification),
                approved, next_flow, analysis)


def _instruction_offer_turn(message, flow, context, llm, knowledge, settings):
    offer = flow["instruction_offer"]
    choices = offer.get("choices", [])
    if not isinstance(choices, list) or not 1 <= len(choices) <= 3:
        raise KnowledgeUnavailable("InvalidSavedInstructionOffer", "instruction_selection")
    normalized = _normalized(message).replace("ё", "е")
    if normalized in {"отмена", "отменить", "вернуться к боту", "продолжить с ботом", "продолжить диалог с ботом"}:
        return Turn("REPLY", [(BACK_TO_BOT, None)], flow={})
    index = None
    rejected = normalized in {"нет", "не подходит", "не подходят", "ни один", "ни один не подходит",
                              "нет не подходит", "не то", "это не то", "ничего не подходит",
                              "ни один вариант не подходит"}
    if "?" not in message and normalized in {"1", "2", "3", "первый", "второй", "третий"}:
        index = {"1": 1, "2": 2, "3": 3, "первый": 1, "второй": 2, "третий": 3}[normalized]
    elif "?" not in message and normalized in {"да", "да подходит", "подходит", "этот подходит", "давайте", "да именно это"}:
        index = 1 if len(choices) == 1 else None
    elif not rejected and not _PHOTO_REQUEST.search(message):
        interpret = getattr(llm, "interpret_instruction_offer", None)
        if callable(interpret):
            offered = [{"number": number, "filename": source["filename"]}
                       for number, source in enumerate(choices, 1)]
            try:
                result = InstructionOfferReply.model_validate(interpret(message, offered))
            except Exception as error:
                report_failure(KnowledgeUnavailable(type(error).__name__, "offer_interpretation"))
                return Turn("REPLY", [(OFFER_CHOICE, None)])
            if result.result in {"ACCEPT", "REJECT"}:
                if not result.evidence or result.evidence not in message:
                    report_failure(KnowledgeUnavailable("InvalidOfferConsent", "offer_interpretation"))
                    return Turn("REPLY", [(OFFER_CHOICE, None)])
                if result.result == "ACCEPT":
                    index = result.choice
                    if index is None:
                        raise KnowledgeUnavailable("InvalidOfferChoice", "instruction_selection")
                else:
                    if result.choice is not None:
                        raise KnowledgeUnavailable("InvalidOfferChoice", "instruction_selection")
                    rejected = True
            elif result.choice is not None or result.evidence:
                raise KnowledgeUnavailable("InvalidOfferReply", "instruction_selection")
            elif result.result == "NEW_REQUEST":
                turn = _text_turn(message, {}, context, None, llm, knowledge, settings)
                turn.flow = turn.flow if turn.flow is not None else flow
                return turn
        elif _technical_message(message):
            turn = _text_turn(message, {}, context, None, llm, knowledge, settings)
            turn.flow = turn.flow if turn.flow is not None else flow
            return turn
    if rejected:
        return Turn("NO_KNOWLEDGE", flow={"clarification_query": offer.get("problem", ""),
                                          "declined_instruction_offer": True,
                                          "excluded": flow.get("excluded", [])})
    if index is None or not 1 <= index <= len(choices):
        return Turn("REPLY", [(OFFER_CHOICE, None)])
    source = choices[index - 1]
    approved = knowledge.read(source["file_id"])
    if approved.document["source"] != source:
        return Turn("REPLY", [(CHANGED, None)], flow={"clarification_query": offer.get("problem", "")})
    selected = applicable_images(approved.document, source["filename"])
    query = f"Пользователь выбрал инструкцию «{source['filename']}»."
    next_flow = {"source": source, "selection": selected, "shown": selected[:6], "query": query,
                 "has_more": len(selected) > 6, "next_offset": 6 if len(selected) > 6 else None,
                 "excluded": flow.get("excluded", [])}
    # Confirmation of an alternative cannot train a shortcut for the original query.
    return _guide_reply(approved, selected[:6], next_flow, knowledge)


def _media_failure(error_code):
    return (MEDIA_ERRORS.get(error_code, "Вложение недоступно или его не удалось обработать.")
            + " Я учитываю только текст сообщения; нужный файл пока нужно прикрепить к письму вручную.")


def _turn(message, flow, context, previous_pending, llm, knowledge, settings, *, inbound_media=None):
    """Employee images remain evidence metadata; only the actual caption follows the text route."""
    media = inbound_media if isinstance(inbound_media, dict) else {}
    has_photo = isinstance(media.get("files"), list) and bool(media["files"])
    error = media.get("error_code")
    error = error if isinstance(error, str) and error else None
    if not has_photo and error is None:
        return _text_turn(message, flow, context, previous_pending, llm, knowledge, settings)
    notice = _media_failure(error) if error else PHOTO_NOTICE
    if not message.strip():
        next_flow = None
        if previous_pending == FORM_ACTION:
            prompted = _intake_prompt(_prefill_intake(flow, context))
            prompt, next_flow = prompted.blocks[0][0], prompted.flow
            # A photo is not a manual hostname reply or a new menu choice.
            # Keep an outstanding collector link and the intake stage intact.
            prior_stage = flow.get("escalation", {}).get("stage")
            if prior_stage:
                next_flow["escalation"]["stage"] = prior_stage
        elif flow.get("source"):
            prompt = "Опишите словами, что происходит и на каком шаге возникла трудность."
        elif flow.get("clarification_keys") and flow["clarification_keys"][-1] in CLARIFY_QUESTIONS:
            prompt = CLARIFY_QUESTIONS[flow["clarification_keys"][-1]]
        else:
            prompt = "Опишите словами проблему и то, какая помощь нужна."
        return Turn("REPLY", [(notice + "\n\n" + prompt, None)], flow=next_flow)
    turn = _text_turn(message, flow, context, previous_pending, llm, knowledge, settings)
    # A valid-photo FORM already gets the manual-attachment warning from the
    # letter presenter. Media failures must still be visible before that preview.
    if turn.action != "FORM" or error:
        turn.media_notice = notice
    return turn


def _text_turn(message, flow, context, previous_pending, llm, knowledge, settings):
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
            if flow.get("escalation", {}).get("profile_prefilled"):
                return _intake_prompt(_prefill_intake(flow, context))
            return _profile_form_request(flow, context)
        if reply := _simple_social(message):
            return Turn("REPLY", [(reply, None)])
        stage = flow.get("escalation", {}).get("stage")
        if ((stage == "CHOICE" and not _structured_letter(message)) or _form_interrupted(message, flow)):
            return _continue_from_letter(message, flow, context, llm, knowledge, settings)
        if intent == "FEEDBACK" or is_confirmation(message):
            return _intake_prompt(_prefill_intake(flow, context))
        # Everything else here is employee data for the pending letter. It must
        # not be classified as chat or used to search for another instruction.
        return _intake_turn(message, flow, llm, context)
    if flow.get("instruction_offer"):
        if intent == "HUMAN":
            return Turn("HUMAN")
        if reply := _simple_social(message):
            return Turn("REPLY", [(reply, None)])
        try:
            return _instruction_offer_turn(message, flow, context, llm, knowledge, settings)
        except Exception as error:
            if isinstance(error, KnowledgeUnavailable):
                report_failure(error)
            return Turn("REPLY", [(UNAVAILABLE, None)])
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
    if feedback := _feedback_kind(message, flow):
        if feedback == "SUCCESS":
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
    if _step_confirmed(message) and not flow:
        return Turn("REPLY", [("Опишите, пожалуйста, с какой рабочей проблемой нужна помощь.", None)])
    pinned = (flow.get("source") and (_step_confirmed(message) or _current_step_question(message) or _different_instruction(message)
                                      or _PHOTO_REQUEST.search(message) or _NEXT_PHOTO.search(message))) or (
        flow.get("clarification_query") and len(message.split()) <= 8
    )
    if not _technical_message(message) and not pinned:
        try:
            routed = DialogueResult.model_validate(llm.dialogue(message, context))
            if routed.route == "REFUSE":
                from app.services.chat.dialogue import REFUSAL
                return Turn("REPLY", [(REFUSAL, None)])
            if routed.route == "SOCIAL" and (reply := safe_social_reply(routed.reply)):
                return Turn("REPLY", [(reply, None)])
        except Exception:
            pass
    try:
        return _support_turn(message, flow, context, llm, knowledge)
    except KnowledgeUnavailable:
        problem_flow = None if flow.get("source") else {**flow, "clarification_query": _problem_query(message, flow)}
        return Turn("REPLY", [(UNAVAILABLE, None)], flow=problem_flow)


def _report_failed_inbox(session, inbox, room, principal):
    """A terminal generation failure gets one durable reply, never an endless room barrier."""
    incident = session.scalar(select(Incident).where(Incident.id == room.incident_id).with_for_update())
    conversation = session.get(Conversation, incident.conversation_id) if incident else None
    if conversation is None or conversation.user_id != principal:
        # Corrupt/obsolete ownership must not produce an external private reply.
        session.add(AuditEvent(event="talk.turn_failure_reported", details={
            "inbox_id": inbox.id, "error_code": "RoomScopeChanged", "reply_queued": False,
        }))
        return
    processing = (room.flow or {}).get("_processing", {})
    if processing.get("inbox_id") == inbox.id:
        previous = processing.get("previous", {})
        incident.pending_action, incident.result = previous.get("pending"), previous.get("result")
        room.flow = {key: value for key, value in (room.flow or {}).items() if key != "_processing"}
    append_message(session, incident.id, "user", inbox.content, "RECEIVED")
    _deliver(session, incident, [(
        "Не удалось обработать предыдущее сообщение из-за временной ошибки. "
        "Текущий шаг сохранён. Напишите вопрос ещё раз или «оператор», чтобы оформить письмо в поддержку.", None,
    )])
    incident.version += 1
    audit(session, incident.id, "talk.turn_failure_reported", inbox_id=inbox.id,
          error_code=inbox.error_code, reply_queued=True)


def _claim(factory, settings, account_id):
    with factory() as session, session.begin():
        reported_failure = select(AuditEvent.id).where(
            AuditEvent.event == "talk.turn_failure_reported",
            AuditEvent.details["inbox_id"].as_string() == TalkInbox.id,
        ).exists()
        statement = select(TalkInbox).join(TalkRoom).join(TalkAccount).where(
            or_(TalkInbox.status.in_(["PENDING", "RUNNING"]),
                (TalkInbox.status == "FAILED") & ~reported_failure), TalkRoom.enabled.is_(True),
            TalkAccount.enabled.is_(True),
            TalkAccount.server_url == settings.nextcloud_base_url.rstrip("/"),
        )
        statement = statement.where(TalkAccount.id == account_id) if account_id else statement.where(
            TalkAccount.username == getattr(settings, "nextcloud_username", "")
        )
        queued = session.scalars(statement.order_by(
            TalkInbox.remote_sequence, TalkInbox.created_at, TalkInbox.id
        ).with_for_update(skip_locked=True)).all()
        # A failed turn has no live generation. RUNNING remains a barrier even
        # if a later poll inserted an older remote sequence behind it.
        blocked = {item.room_id for item in queued if item.status == "RUNNING"}
        inbox = None
        for item in queued:
            if item.room_id not in blocked and item.status in {"PENDING", "FAILED"}:
                room = session.scalar(select(TalkRoom).where(TalkRoom.id == item.room_id).with_for_update())
                account = session.get(TalkAccount, room.account_id)
                principal = str(uuid5(NAMESPACE_URL, account.server_url + "/" + account.username + "/"
                                      + room.room_token + "/" + room.employee_id))
                pause_due_to_input(session, room, item)
                uncertain = session.scalar(select(Delivery.id).join(Message).join(Incident).join(Conversation).where(
                    Delivery.room_token == room.room_token, Delivery.status.not_in(TERMINAL_DELIVERIES),
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
        if inbox.status == "FAILED":
            _report_failed_inbox(session, inbox, room, principal)
            return {"ignored": True}
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
        media = copy.deepcopy(inbox.media or {})
        if media.get("files"):
            incident.has_user_photos = True
        start_progress(inbox)
        incident.pending_action, incident.version = "PROCESSING", incident.version + 1
        context = load_context(session, incident, settings)
        context["channel_profile"] = channel_profile(room, incident.id)
        flow = copy.deepcopy(room.flow or {})
        room.flow = {**flow, "_processing": {"inbox_id": inbox.id, "tries": inbox.tries, "previous": previous}}
        return {"inbox_id": inbox.id, "room_id": room.id, "incident_id": incident.id, "version": incident.version,
                "message": inbox.content, "flow": flow, "previous": previous,
                "context": context, "principal": principal, "account_id": account.id, "tries": inbox.tries,
                "media": media}


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
            if flow.get("declined_instruction_offer"):
                saved_flow["declined_instruction_offer"] = True
        resume = {"flow": saved_flow, "state": incident.state if pending else "DIAGNOSING",
                  "result": incident.result if pending else None}
    problem = resume["query"] if "query" in resume else _last_problem(resume["flow"], context)
    resume["query"] = problem
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
    if flow.get("declined_instruction_offer"):
        prefix = "Для описанной ситуации подходящая утверждённая инструкция не найдена. Предложенные варианты вам не подошли."
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
        first_sequence = session.scalar(select(func.coalesce(func.max(Message.sequence), 0)).where(
            Message.incident_id == incident.id,
        ))
        append_message(session, incident.id, "user", claimed["message"], "RECEIVED")
        if turn.confirmed_computer:
            remember_computer(room, incident.id, turn.confirmed_computer)
        if turn.media_notice:
            _deliver(session, incident, [(turn.media_notice, None)])
        pending = _attempt(session, incident)
        if turn.action == "DELIVERY_RESUME":
            if not resume_response(session, room, incident, inbox, turn.resume or {}):
                _deliver(session, incident, [("Предыдущий ответ уже завершён или его нельзя безопасно продолжить. "
                                                "Напишите, с чем нужна помощь.", None)])
        elif turn.action in {"GUIDE", "RETRY_GUIDE"}:
            if turn.action == "RETRY_GUIDE" and pending:
                pending.status = "FAILED"
                session.add(Feedback(attempt_id=pending.id, helped=False, comment=""))
                incident.failed_attempts += 1
                session.flush()
                pending = None
            source = turn.approved.document["source"]
            existing = session.scalar(select(SolutionAttempt).where(
                SolutionAttempt.incident_id == incident.id,
                SolutionAttempt.solution_id == turn.approved.candidate.solution_id,
            ))
            same = bool(existing is not None and pending is not None and existing.id == pending.id)
            attempts = session.scalars(select(SolutionAttempt).where(SolutionAttempt.incident_id == incident.id)).all()
            if existing and existing.evidence.get("talk_source") != source:
                _deliver(session, incident, [(CHANGED, None)])
            elif existing and existing.status not in {"UNASSESSED", "WAITING_FEEDBACK"}:
                # A failed solution already owns its unique feedback record.
                # Do not repeat it as a new attempt or silently reset the result.
                room.flow = _offer_escalation(session, incident, "NO_MORE_SOLUTIONS", room.flow, claimed["context"])
            elif existing is None and len(attempts) >= MAX_ATTEMPTS:
                room.flow = _offer_escalation(session, incident, "ATTEMPTS_EXHAUSTED", turn.flow, claimed["context"])
            else:
                if not same and pending:
                    pending.status = "UNASSESSED"
                    session.flush()  # Release uq_pending_attempt before reactivating another saved attempt.
                save_snapshot(session, turn.approved.candidate)
                first = _deliver(session, incident, turn.blocks, source=source)
                if not same:
                    if existing is not None:
                        existing.status, existing.message_id = "WAITING_FEEDBACK", first.id
                    else:
                        candidate = turn.approved.candidate
                        session.add(SolutionAttempt(
                            incident_id=incident.id, solution_id=candidate.solution_id, ordinal=len(attempts) + 1,
                            message_id=first.id, evidence={"title": candidate.title, "version": candidate.version,
                                "chunk_id": candidate.chunk_id, "content": candidate.content, "steps": candidate.steps,
                                "source": candidate.source, "talk_source": source},
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
                    if not has_unfinished_guide(session, room, pending):
                        remember_success(session, settings, claimed["principal"], incident.id,
                                         (room.flow or {}).get("retrieval_key"), pending.evidence.get("talk_source"))
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
            flow = copy.deepcopy(turn.flow if turn.flow is not None else room.flow)
            escalation = flow.get("escalation", {})
            flow["escalation"] = {**escalation, "stage": "FORM"}
            room.flow = flow
            if (getattr(settings, "device_collection_base_url", "")
                    and not escalation.get("intake_fields", {}).get("computer_name")):
                from app.services.nextcloud.device_collection import issue_collection

                collection_id, link = issue_collection(session, settings, room, incident)
                room.flow = {**flow, "escalation": {**flow["escalation"], "stage": "DEVICE",
                                                  "collection_id": collection_id}}
                _deliver(session, incident, [(
                    "Определите текущий компьютер для письма в поддержку:\n\n"
                    + "[Определить текущий компьютер](" + link + ")\n\n"
                    + "Будет передано только имя компьютера. ФИО берётся из вашего профиля. "
                    + "Ссылка действует 5 минут.\n"
                    + "Если автоматический сбор недоступен, напишите имя компьютера здесь. "
                    + "Чтобы вернуться к боту, напишите «Отмена» или «2».", None,
                )])
            else:
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
        if turn.action != "DELIVERY_RESUME":
            record_response(session, room, incident, inbox, first_sequence)
        pause_due_to_input(session, room)
        # The public completion event is emitted by process_one after commit.
        # Recording it through audit() here previously logged a false success
        # when a deferred UNIQUE constraint failed on transaction exit.
        session.add(AuditEvent(incident_id=incident.id, event="talk.turn_completed",
                               details={"route": turn.action, "block_count": len(turn.blocks)}))
        return True


def _safe_database_error(error):
    """Only known machine identifiers, never SQL, driver messages or bind values."""
    if not isinstance(error, DBAPIError):
        return {}
    identifiers = {}
    original = error.orig
    sqlstate = getattr(original, "sqlstate", None) or getattr(original, "pgcode", None)
    if isinstance(sqlstate, str) and re.fullmatch(r"[A-Z0-9]{5}", sqlstate):
        identifiers["db_sqlstate"] = sqlstate
    constraint = getattr(getattr(original, "diag", None), "constraint_name", None)
    if constraint in {
        "uq_pending_attempt", "uq_active_incident", "solution_attempts_incident_id_solution_id_key",
        "solution_attempts_incident_id_ordinal_key", "feedback_attempt_id_key",
        "messages_incident_id_sequence_key", "knowledge_documents_pkey", "knowledge_chunks_pkey", "solutions_pkey",
        "talk_inbox_room_id_remote_message_id_key",
    }:
        identifiers["db_constraint"] = constraint
    sqlite_code = getattr(original, "sqlite_errorname", None)
    if isinstance(sqlite_code, str) and re.fullmatch(r"SQLITE_(?:CONSTRAINT|BUSY|LOCKED)(?:_[A-Z]+)?", sqlite_code):
        identifiers["db_sqlite_code"] = sqlite_code
    return identifiers


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
    try:
        timed_llm = _TimedProxy(llm, metadata, {"dialogue": "model_api", "analyze_context": "model_api",
                                               "extract_escalation": "model_api", "interpret_instruction_offer": "model_api"})
        def select_source(payload):
            with _measure(metadata, "model_api"):
                call = getattr(llm, "_call", None)
                if settings.llm_backend == "openwebui":
                    select_instruction = getattr(llm, "select_instruction", None)
                    if callable(select_instruction):
                        return select_instruction(payload)
                    if not callable(call):
                        raise KnowledgeUnavailable("ModelUnavailable", "instruction_selection")
                    return call(InstructionSelection, INSTRUCTION_SELECTION_TASK, payload)
                return call(SourceSelection, _SELECTION_TASK, payload) if callable(call) else {"file_id": None}

        cached_knowledge = SuccessfulTalkKnowledge(
            knowledge or TalkKnowledge(settings, selector=select_source), factory, settings,
            claimed["principal"], claimed["context"],
        )
        timed_knowledge = _TimedProxy(cached_knowledge, metadata,
                                      {"search": "search", "search_with_llm": "search",
                                       "read": "read", "is_current": "verify"})
        with _measure(metadata, "route"):
            paused = None
            if is_resume_request(claimed["message"]) and not claimed["media"]:
                with factory() as session:
                    room = session.get(TalkRoom, claimed["room_id"])
                    paused = latest_paused(session, room)
            if paused is not None:
                source = paused.get("flow", {}).get("source")
                if source and not timed_knowledge.is_current(source):
                    turn = Turn("REPLY", [(CHANGED, None)])
                else:
                    turn = Turn("DELIVERY_RESUME", resume=paused)
            else:
                turn = _turn(claimed["message"], claimed["flow"], claimed["context"],
                             claimed["previous"]["pending"], timed_llm, timed_knowledge, settings,
                             inbound_media=claimed["media"])
        metadata["cache_hit"] = cached_knowledge.cache_hit
        with _measure(metadata, "persist"):
            finished = _finish(factory, turn, claimed, llm, settings)
    except Exception as error:
        with factory() as session, session.begin():
            inbox = session.scalar(select(TalkInbox).where(TalkInbox.id == claimed["inbox_id"]).with_for_update())
            room = session.scalar(select(TalkRoom).where(TalkRoom.id == claimed["room_id"]).with_for_update())
            incident = session.scalar(select(Incident).where(Incident.id == claimed["incident_id"]).with_for_update())
            if (inbox.status == "RUNNING" and inbox.tries == claimed["tries"]
                    and incident.version == claimed["version"]):
                inbox.status = "PENDING" if inbox.tries < 3 else "FAILED"
                inbox.error_code = type(error).__name__
                incident.pending_action = claimed["previous"]["pending"]
                incident.result = claimed["previous"]["result"]
                room.flow = claimed["flow"]
        event("talk.turn_error", error_code=type(error).__name__, **_safe_database_error(error))
        _report_timing(metadata, started, "FAILED", 0, type(error).__name__)
        return True
    _report_timing(metadata, started, turn.action if finished else "IGNORED", len(turn.blocks))
    return True
