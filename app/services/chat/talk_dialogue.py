"""Durable private Talk dialogue using the same approved steps/photos as WebUI."""

import copy
import hashlib
import json
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
from app.schemas.contracts import AnalysisResult, DialogueResult, EscalationDetails, InstructionOfferReply, UnderstandingReply
from app.services.chat.audit import audit
from app.services.chat.channel_profile import channel_profile, remember_computer
from app.services.chat.confirmation_scope import (
    confirmation_scope_entries, confirmation_scope_identity, render_confirmation_scopes,
)
from app.services.chat.dialogue import _OPERATION_CLAIM, _TECHNICAL_OUTPUT
from app.services.nextcloud.progress import start_progress
from app.services.nextcloud.interruption import (
    TERMINAL_DELIVERIES, has_unfinished_guide, is_resume_request, latest_paused, pause_due_to_input,
    record_response, resume_response,
)
from app.services.chat.memory import _redact, load_context, remember_resolution
from app.services.chat.offer_adoption import adopt_instruction_offer
from app.services.chat.messages import append_message
from app.services.chat.instruction_plan import composite_approved, immutable_plan, render_plan, restore_plan, serialize_plan
from app.services.chat.policies import (
    FORM_ACTION, TERMINAL_STATES, direct_intent, requests_support_form, resolution_feedback,
)
from app.services.chat.service import messenger
from app.services.chat.question_guard import unsupported_question_content
from app.services.chat.support_followup import (
    NO_DIAGNOSTIC_STEPS, diagnostic_excerpt, needs_error_text, observational_followup, verified_guide_context,
)
from app.services.chat.task_state import (
    is_unknown_answer, remember_task_message, starts_new_task, task_evidence_authority, task_query, update_task_state,
)
from app.services.rag.openwebui import save_snapshot
from app.services.rag.query import _terms, normalized_query, printer_kinds, scanner_kinds
from app.services.rag.talk import (
    ApprovedTalkDocument, INSTRUCTION_SELECTION_TASK, InstructionSelection, KnowledgeUnavailable,
    SourceSelection, TalkKnowledge, _SELECTION_TASK, knowledge_stage, report_failure,
)
from app.services.rag.success import SuccessfulTalkKnowledge, cache_key, remember_success
from app.services.ticketing.intake import _ordered_lines, parse_intake
from app.services.ticketing.service import collect_details
from integrations.openwebui.automatic_instruction_images import (
    TIMING_MARKER, _clock, _measure, _milliseconds,
    applicable_images, clarification_images, instruction_blocks, is_confirmation,
    is_ui_question, knowledge_scope_reply, refusal_reply,
    requires_support, safe_social_reply,
)

UNAVAILABLE = "Сейчас не удалось проверить базу знаний. Повторите вопрос немного позже."
SELECTION_UNCLEAR = ("Пока не удалось выбрать подтверждённый следующий шаг. "
                     "Можно описать, что сейчас видно на экране.")
INDEX_UPDATING = ("База знаний ещё обрабатывает новые материалы. Ваш вопрос сохранён. "
                  "Повторите его немного позже.")
CHANGED = "Инструкция изменилась или доступ к ней недоступен. Продолжение приостановлено до проверки актуального источника."
ESCALATION_MENU = ("Как продолжим?\n1 — оформить письмо в техническую поддержку.\n"
                   "2 — продолжить диалог с ботом.\n\nМожно ответить цифрой или словами. Заявка пока не отправлена.")
SUPPORT_OFFER = "Составить письмо для отправки в техническую поддержку?"
SUPPORT_OFFER_INTERVAL = 10
SUPPORT_OFFER_KEY = "_support_letter_offer"
NO_INSTRUCTION = "По вашему описанию подходящую утверждённую инструкцию пока не нашёл."
NO_OTHER_SOLUTION = "Других подходящих утверждённых решений пока не нашёл."
OFFER_CHOICE = "Какой вариант подходит? Напишите его номер или «нет»."
UNDERSTANDING_CHOICE = "Правильно ли я понял вашу задачу? Подтвердите или коротко поправьте меня."
UNDERSTANDING_CORRECTION = "Что именно нужно сделать? Коротко поправьте моё понимание задачи."
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
_UNSPECIFIED_DIFFICULTY = frozenset({
    "не получилось", "не получается", "не смог", "не смогла", "не разобрался", "не разобралась",
    "застрял", "застряла", "я застрял", "я застряла",
})


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
    block_sources: list | None = None
    failed_previous: bool = False


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
    if not updated["escalation"].get("resume"):
        updated["escalation"]["resume"] = {
            "flow": copy.deepcopy({key: value for key, value in flow.items() if key != "escalation"}),
            "state": context.get("incident", {}).get("state", "DIAGNOSING"), "result": None,
        }
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
    if (flow or {}).get("source") and (reported := resolution_feedback(message)):
        return reported
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
    return is_ui_question(message) or _normalized(message) in _UNSPECIFIED_DIFFICULTY or bool(re.search(
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
    if (not restored.get("task_state", {}).get("goals")
            and not (restored.get("source") and _current_step_question(message))):
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
        latest = message[:6000]
        previous = previous[:max(0, 6000 - len(latest) - 1)]
        return (previous + "\n" + latest).lstrip("\n")
    return message[:6000]


def _resume_turn(flow, context, llm, knowledge):
    saved = flow.get("escalation", {}).get("resume", {})
    restored = copy.deepcopy(saved.get("flow", {key: value for key, value in flow.items() if key != "escalation"}))
    if restored.get("declined_instruction_offer"):
        return Turn("RESUME", [(BACK_TO_BOT, None)], flow={}, resume={})
    if isinstance(restored.get("understanding_confirmation"), dict):
        confirmation = restored["understanding_confirmation"]
        summary = confirmation.get("summary", "")
        scope = confirmation.get("scope_description", "")
        scope = re.sub(r"([\\`*_{}\[\]()<>])", r"\\\1", scope)
        text = f"Я понял задачу так — {summary}.\n\n" + (scope + "\n\n" if scope else "") + UNDERSTANDING_CHOICE
        return Turn("RESUME", [(text, None)], flow=restored, resume=saved)
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
            if restored.get("instruction_plan"):
                restore_plan(knowledge, restored["instruction_plan"])
                valid = True
            else:
                approved = knowledge.read(restored["source"]["file_id"])
                valid = approved.document["source"] == restored["source"] and knowledge.is_current(restored["source"])
        except KnowledgeUnavailable:
            valid = False
            reply = UNAVAILABLE
        else:
            reply = CHANGED
        if valid:
            return Turn("RESUME", [(RESUME_GUIDE, None)], flow=restored, resume=saved)
        retained = {key: copy.deepcopy(restored[key]) for key in (
            "task_state", "task_query", "opening_query", "answered_questions",
        ) if key in restored}
        restored = {**retained, "clarification_query": query, "excluded": restored.get("excluded", [])} if query else retained
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


def _model_clarification(question, reported_messages=()):
    """Only a short informational question can leave the source-selection task."""
    question = question.strip() if isinstance(question, str) else ""
    checked = question
    for quoted in re.finditer(r"«([^»]+)»|\"([^\"]+)\"|“([^”]+)”", question):
        label = next(group for group in quoted.groups() if group is not None)
        if (len(label) <= 160 and re.search(r"\b(?:неверн\w*|неправильн\w*|ошибк\w*|invalid|incorrect|wrong)\b", label, re.I)
                and any(label.casefold() in text.casefold() for text in reported_messages if isinstance(text, str))):
            checked = checked.replace(quoted.group(), "сообщение об ошибке")
    if (not 5 <= len(question) <= 300 or question.count("?") != 1 or not question.endswith("?")
            or re.search(r"[\r\n<>]|(?:mailto:|file:|data:|javascript:|www\.)", question, re.I)
            or _TECHNICAL_OUTPUT.search(question) or _OPERATION_CLAIM.search(question)
            or _redact(question) != question or refusal_reply(checked)
            or unsupported_question_content(checked, reported_messages)
            or re.search(r"\b(?:в\s+(?:нашей\s+)?базе\s+знан\w*|наш[её]л\w*|"
                         r"найден\w*\s+инструкц\w*|есть\s+(?:похож\w*\s+)?инструкц\w*)\b", question, re.I)
            or re.search(r"\b(?:парол\w*|password\w*|токен\w*|api[ _-]?key|secret\w*|"
                         r"отключите|сбросьте|переустановите)\b", checked, re.I)):
        error = KnowledgeUnavailable("InvalidClarification", "instruction_selection")
        report_failure(error)
        raise error
    return question


def _confirmation_enabled(knowledge):
    return bool(getattr(getattr(knowledge, "settings", None), "talk_confirm_understanding", False))


def _corrected_known_fact(previous, state, message):
    for fact in state.get("facts", []):
        if (fact.get("negated") or fact.get("key") in {"completed", "current_step", "symptoms", "error_text"}
                or not fact.get("evidence") or fact["evidence"] not in message):
            continue
        for prior in previous.get("facts", []):
            if (not prior.get("negated") and all(_normalized(fact.get(key, "")) ==
                                                _normalized(prior.get(key, ""))
                                                for key in ("key", "entity", "attribute"))
                    and _normalized(fact["value"]) != _normalized(prior["value"])):
                return True
    return False


def _apply_goal_kind_overrides(decision, flow):
    overrides = getattr(decision, "goal_kind_overrides", {})
    goals = flow.get("task_state", {}).get("goals", {})
    if (not isinstance(overrides, dict) or not overrides.keys() <= goals.keys()
            or any(not isinstance(kind, str) or kind not in {"ACTION", "LOOKUP"}
                   for kind in overrides.values())):
        raise KnowledgeUnavailable("InvalidGoalKindOverride", "instruction_selection")
    for goal_id, kind in overrides.items():
        goals[goal_id]["kind"] = kind


def _source_condition(decision, flow, question):
    condition = getattr(decision, "clarification_condition", None)
    if condition is None:
        return None
    if condition not in {"APPLICATION_NAME", "APPLICATION_SURFACE", "EQUIPMENT_TYPE", "DEVICE_COUNT",
                          "PHYSICAL_CONNECTION", "VISIBLE_STATE", "WORK_HISTORY", "DESIRED_RESULT"}:
        raise KnowledgeUnavailable("InvalidSourceClarification", "instruction_selection")
    sources = [{key: item.document["source"][key] for key in ("file_id", "sha256")}
               for item in getattr(decision, "clarification_sources", ())]
    return {"question": question, "clarification_condition": condition,
            "goal_ids": list(flow.get("task_state", {}).get("goals", {})), "sources": sources}


def _legacy_confirmation_identity(guide_flow, scopes):
    goals = guide_flow.get("task_state", {}).get("goals", {})
    body = {"parts": guide_flow.get("instruction_plan", {}).get("parts"),
            "scopes": scopes, "goals": {key: {name: goal[name] for name in ("text", "kind") if name in goal}
                                         for key, goal in goals.items()},
            "query": (guide_flow.get("understanding") or guide_flow.get("query", "")) if not goals else ""}
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _confirmation_identity(guide_flow, scopes):
    if not guide_flow.get("task_state", {}).get("goals"):
        return _legacy_confirmation_identity(guide_flow, scopes)
    return confirmation_scope_identity(scopes)


def _confirmation_turn(approved, guide_flow, summary, *, remembered=False, parts=()):
    summary = " ".join(summary.split()).replace("?", ".").strip(".! ")
    summary = _redact(summary)[:6000]
    visible_summary = re.sub(r"([\\`*_{}\[\]()<>])", r"\\\1", summary)
    scopes = confirmation_scope_entries(approved, parts)
    identity = _confirmation_identity(guide_flow, scopes)
    if (identity in guide_flow.get("rejected_scope_ids", [])
            or _legacy_confirmation_identity(guide_flow, scopes) in guide_flow.get("rejected_scope_ids", [])
            or any(confirmation_scope_identity(entries) == identity
                   for entries in guide_flow.get("rejected_scope_entries", []))):
        retained = {key: copy.deepcopy(guide_flow[key]) for key in (
            "task_state", "task_query", "opening_query", "answered_questions", "excluded", "rejected_scope_ids",
            "rejected_scope_entries", "accepted_instruction_offer", "answered_conditions",
        ) if key in guide_flow}
        retained.update(clarification_query=guide_flow["query"], reselect_source=True)
        return Turn("NO_KNOWLEDGE", flow=retained)
    scope_description = _redact(render_confirmation_scopes(scopes))
    visible_scope = re.sub(r"([\\`*_{}\[\]()<>])", r"\\\1", scope_description)
    intro = "В прошлый раз с таким вопросом вы решали задачу" if remembered else "Я понял задачу так"
    text = (f"{intro} — {visible_summary}.\n\n{visible_scope}\n\n"
            "Правильно я понял? Подтвердите или коротко поправьте меня.")
    if guide_flow.get("coverage_mode") == "partial":
        text = (f"Я понял задачу так — {visible_summary}.\n\n"
                + visible_scope + "\n\n"
                + _partial_boundary(guide_flow)
                + "\n\nПоказать подтверждённую часть инструкции с фотографиями?")
    confirmation_flow = {
        "understanding_confirmation": {"source": copy.deepcopy(approved.document["source"]),
                                      "guide_flow": copy.deepcopy(guide_flow), "summary": summary,
                                      "scope_identity": identity,
                                      "scope_entries": scopes, "scope_description": scope_description},
        "clarification_query": guide_flow["query"],
        "opening_query": guide_flow.get("opening_query", guide_flow["query"]),
        "excluded": guide_flow.get("excluded", []),
        "rejected_scope_ids": guide_flow.get("rejected_scope_ids", []),
        "rejected_scope_entries": copy.deepcopy(guide_flow.get("rejected_scope_entries", [])),
    }
    if remembered:
        confirmation_flow["remembered_interpretation"] = True
    return Turn("REPLY", [(text, None)], flow=confirmation_flow)


def _partial_boundary(flow):
    covered = ", ".join(flow["covered_goals"])
    missing = ", ".join(flow["missing_goals"])
    return (f"В найденной инструкции подтверждено — {covered}.\n"
            f"Для действий «{missing}» подтверждённых шагов в найденных материалах пока нет.")


def _guide_flow(approved, problem, opening_query, excluded):
    eligible = applicable_images(approved.document, problem)
    return {"source": copy.deepcopy(approved.document["source"]), "selection": eligible, "shown": eligible[:6],
            "query": problem, "opening_query": opening_query, "has_more": len(eligible) > 6,
            "next_offset": 6 if len(eligible) > 6 else None, "excluded": excluded}


def _understanding_turn(message, flow, context, llm, knowledge, settings):
    confirmation = flow["understanding_confirmation"]
    if (not isinstance(confirmation, dict) or not isinstance(confirmation.get("source"), dict)
            or not isinstance(confirmation.get("guide_flow"), dict)
            or not isinstance(confirmation.get("summary"), str)):
        raise KnowledgeUnavailable("InvalidSavedUnderstanding", "understanding_reply")
    normalized = _normalized(message).replace("ё", "е")
    if _return_to_bot(message):
        return Turn("REPLY", [(BACK_TO_BOT, None)], flow={})
    accepted = normalized in {"да", "верно", "правильно", "все верно", "все правильно", "да верно",
                              "да все верно", "да все правильно", "именно", "именно так", "подтверждаю"}
    affirmation_words = set(normalized.split())
    if (affirmation_words and affirmation_words <= {
            "да", "ага", "верно", "правильно", "все", "так", "именно", "понял", "поняли",
            "вы", "меня", "подтверждаю", "совершенно", "абсолютно",
    } and affirmation_words.intersection({"да", "ага", "верно", "правильно", "подтверждаю"})):
        accepted = True
    scope_entries = confirmation.get("scope_entries", [])
    guide = confirmation["guide_flow"]
    if (len(scope_entries) == 1 and len((guide.get("task_state") or {}).get("goals", {})) <= 1
            and guide.get("coverage_mode") != "partial" and "?" not in message):
        title = normalized_query(scope_entries[0].get("title"))
        accepted = accepted or bool(title and normalized_query(message) == title)
    rejected = normalized in {"нет", "неверно", "неправильно", "не то", "это не то", "не так"}
    user_flow = {**confirmation["guide_flow"], **flow}
    correction = {key: copy.deepcopy(user_flow[key]) for key in (
        "task_state", "task_query", "opening_query", "answered_questions", "excluded",
        "rejected_scope_ids", "rejected_scope_entries", "accepted_instruction_offer", "answered_conditions",
    ) if key in user_flow}
    correction.update(clarification_query=flow["clarification_query"], clarification_keys=["MODEL"],
                      reselect_source=True)
    correction.setdefault("opening_query", flow.get("clarification_query", ""))
    rejected_scopes = list(correction.get("rejected_scope_ids", []))
    scope_identity = confirmation.get("scope_identity") or _confirmation_identity(
        confirmation["guide_flow"], scope_entries,
    )
    correction["rejected_scope_ids"] = list(dict.fromkeys([*rejected_scopes, scope_identity]))[-8:]
    records = list(correction.get("rejected_scope_entries", []))
    typed_goals = user_flow.get("task_state", {}).get("goals")
    if (typed_goals and scope_entries
            and not any(confirmation_scope_identity(entries) == confirmation_scope_identity(scope_entries)
                        for entries in records)):
        records.append(copy.deepcopy(scope_entries))
    if records:
        correction["rejected_scope_entries"] = records[-8:]
    if not typed_goals and not flow.get("remembered_interpretation"):
        correction["opening_query"] = ""
    if flow.get("remembered_interpretation"):
        opening = correction["opening_query"]
        correction["task_state"] = remember_task_message(opening) if opening else None
    if rejected:
        correction["clarification_question"] = UNDERSTANDING_CORRECTION
        return Turn("REPLY", [(UNDERSTANDING_CORRECTION, None)], flow=correction)
    if re.match(r"\s*(?:нет|неправильно|неверно)\b[.!,:\s]+\S", message, re.I) and _technical_message(message):
        return _support_turn(message, correction, context, llm, knowledge)
    if not accepted:
        if _PHOTO_REQUEST.search(message):
            return Turn("REPLY", [(UNDERSTANDING_CHOICE, None)])
        interpret = getattr(llm, "interpret_understanding_reply", None)
        if callable(interpret):
            try:
                accepted_meaning = confirmation["summary"] + "\n" + confirmation.get("scope_description", "")
                reply = UnderstandingReply.model_validate(interpret(message, accepted_meaning))
                if reply.result != "UNCLEAR" and (not reply.evidence or reply.evidence not in message):
                    raise ValueError("InvalidUnderstandingConsent")
            except Exception as error:
                report_failure(KnowledgeUnavailable(type(error).__name__, "understanding_reply"))
                return Turn("REPLY", [(UNDERSTANDING_CHOICE, None)])
            if reply.result == "NEW_REQUEST":
                turn = _text_turn(message, {}, context, None, llm, knowledge, settings)
                turn.flow = turn.flow if turn.flow is not None else flow
                return turn
            if reply.result == "CORRECT":
                return _support_turn(message, correction, context, llm, knowledge)
            accepted = reply.result == "ACCEPT"
        elif re.match(r"\s*(?:как|почему|где|другой\s+вопрос|другая\s+проблема)\b", message, re.I):
            return _text_turn(message, {}, context, None, llm, knowledge, settings)
        elif re.match(r"\s*нет\b", message, re.I) or _technical_message(message):
            return _support_turn(message, correction, context, llm, knowledge)
        else:
            try:
                reply = DialogueResult.model_validate(llm.dialogue(message, context))
                if reply.route == "SOCIAL" and (text := safe_social_reply(reply.reply)):
                    return Turn("REPLY", [(text, None)])
            except Exception:
                pass
        if not accepted:
            return Turn("REPLY", [(UNDERSTANDING_CHOICE, None)])
    source = confirmation["source"]
    guide_flow = copy.deepcopy(confirmation["guide_flow"])
    if not confirmation.get("scope_entries"):
        parts = restore_plan(knowledge, guide_flow["instruction_plan"]) if guide_flow.get("instruction_plan") else ()
        approved = composite_approved(parts, guide_flow["instruction_plan"]) if parts else knowledge.read(source["file_id"])
        if approved.document["source"] != source:
            return Turn("REPLY", [(CHANGED, None)], flow=correction)
        refreshed = _confirmation_turn(approved, guide_flow, confirmation["summary"], parts=parts)
        refreshed.flow.update({key: copy.deepcopy(flow[key]) for key in ("task_state", "task_query", "answered_questions")
                               if key in flow})
        return refreshed
    guide_flow["understanding_confirmed"] = True
    if guide_flow.get("instruction_plan"):
        try:
            return _guide_reply(None, [], guide_flow, knowledge)
        except KnowledgeUnavailable:
            return Turn("REPLY", [(CHANGED, None)], flow=correction)
    else:
        approved = knowledge.read(source["file_id"])
        if approved.document["source"] != source:
            return Turn("REPLY", [(CHANGED, None)], flow=correction)
    return _guide_reply(approved, guide_flow["shown"], guide_flow, knowledge)


def _goal_order_turn(message, flow, context, llm, knowledge, settings):
    offer = flow["goal_order_offer"]
    state = copy.deepcopy(flow.get("task_state", {}))
    order = offer.get("goal_order", [])
    if (not state.get("goals") or state["goals"] != offer.get("goals")
            or not isinstance(order, list) or len(order) != len(state["goals"])
            or len(set(order)) != len(order) or set(order) != set(state["goals"])):
        raise KnowledgeUnavailable("InvalidSavedGoalOrder")
    restored = {key: copy.deepcopy(value) for key, value in flow.items()
                if key not in {"goal_order_offer", "clarification_question", "source", "instruction_plan"}}
    text = _normalized(message).replace("ё", "е")
    words = set(text.split())
    accepted = bool(words and words <= {
        "да", "ага", "верно", "правильно", "все", "так", "именно", "понял", "поняли", "вы", "меня",
        "подтверждаю", "согласен", "согласна", "можно", "хорошо", "давайте", "согласовать", "порядок",
    } and words.intersection({"да", "ага", "верно", "правильно", "подтверждаю", "согласен", "согласна", "можно"}))
    rejected = bool(re.match(r"^(?:нет|не\s+(?:согласен|согласна|надо|меняйте|менять))\b", text))
    unknown = is_unknown_answer(message)
    if not accepted and not rejected and not unknown:
        interpret = getattr(llm, "interpret_understanding_reply", None)
        if callable(interpret):
            try:
                reply = UnderstandingReply.model_validate(interpret(message, offer.get("prompt", offer["question"])))
            except Exception as error:
                report_failure(KnowledgeUnavailable(type(error).__name__, "goal_order_reply"))
                return Turn("REPLY", [("Ответьте «да», «нет» или коротко поправьте задачу.", None)])
            if reply.result != "UNCLEAR" and (not reply.evidence or reply.evidence not in message):
                raise KnowledgeUnavailable("InvalidGoalOrderConsent")
            accepted = reply.result == "ACCEPT"
            if reply.result == "NEW_REQUEST":
                return _text_turn(message, {}, context, None, llm, knowledge, settings)
            if reply.result == "CORRECT":
                return _support_turn(message, restored, context, llm, knowledge)
        elif _technical_message(message):
            return _support_turn(message, restored, context, llm, knowledge)
    if accepted:
        sources = offer.get("sources", [])
        if not sources or not all(knowledge.is_current(source) for source in sources):
            return Turn("REPLY", [(CHANGED, None)], flow=restored)
        state = remember_task_message(message, state)
        state["goals"] = {goal_id: state["goals"][goal_id] for goal_id in order}
        answered = [*restored.get("answered_questions", []), offer["question"]]
        restored.update(task_state=state, task_query=task_query(state), reselect_source=True,
                        answered_questions=list(dict.fromkeys(answered))[-8:])
        turn = _retrieve_support_turn(message, restored, context, llm, knowledge)
        retained = {key: restored[key] for key in (
            "task_state", "task_query", "opening_query", "answered_questions", "rejected_goal_orders", "excluded",
        ) if key in restored}
        turn.flow = {**retained, **(turn.flow or {})}
        return turn
    if rejected or unknown:
        rejected_orders = list(restored.get("rejected_goal_orders", []))
        if order not in rejected_orders:
            rejected_orders.append(order)
        state = remember_task_message(message, state)
        restored.update(task_state=state, rejected_goal_orders=rejected_orders[-8:])
        return Turn("UNRESOLVED", [("Сохраняю указанный вами порядок действий. "
                               "Для него пока не удалось составить подтверждённую последовательность.", None)],
                    flow=restored)
    return Turn("REPLY", [("Можно изменить порядок действий, как предложено выше? "
                           "Ответьте «да», «нет» или поправьте задачу.", None)])


def _support_turn(message, flow, context, llm, knowledge):
    """Understand employee facts before retrieval, without document-driven guesses."""
    explicit_new_task = starts_new_task(message)
    if explicit_new_task:
        flow = {}
    typed_dialogue = _confirmation_enabled(knowledge) or bool(getattr(llm, "supports_instruction_planning", False))
    if not typed_dialogue or not callable(getattr(llm, "understand_task", None)):
        return _retrieve_support_turn(message, flow, context, llm, knowledge)
    if (flow.get("source") and is_unknown_answer(message)
            and not (flow.get("clarification_question") or flow.get("task_state", {}).get("question"))):
        retained = copy.deepcopy(flow)
        question = "На каком действии возникла трудность и что сейчас происходит?"
        if retained.get("task_state"):
            retained["task_state"] = remember_task_message(message, retained["task_state"])
            retained["task_state"].update(question=question, question_field="current_step")
        retained.update(clarification_question=question, clarification_query=retained.get("task_query", ""))
        return Turn("REPLY", [(question, None)], flow=retained)
    if flow.get("source") and _normalized(message) in _UNSPECIFIED_DIFFICULTY:
        retained = copy.deepcopy(flow)
        question = "На каком действии возникла трудность и что сейчас происходит?"
        if retained.get("task_state"):
            state = remember_task_message(message, retained["task_state"])
            state["question"], state["question_field"] = question, "current_step"
            retained["task_state"] = state
        retained.update(clarification_question=question,
                        clarification_query=retained.get("task_query") or retained.get("query", ""))
        return Turn("REPLY", [(question, None)], flow=retained)
    photo_navigation = (_PHOTO_REQUEST.search(message) or _NEXT_PHOTO.search(message)) and set(
        _normalized(message).split()
    ) <= {"покажи", "покажите", "пришли", "пришлите", "отправь", "отправьте", "повтори", "повторите",
          "фото", "фотографию", "фотографии", "картинку", "картинки", "изображение", "изображения",
          "скриншот", "мне", "пожалуйста", "еще", "раз", "следующее", "следующую", "следующие",
          "этот", "это", "эту", "предыдущее", "предыдущую", "предыдущие", "текущую", "к", "этому", "шагу"}
    if flow.get("source") and (_step_confirmed(message) or photo_navigation):
        return _retrieve_support_turn(message, flow, context, llm, knowledge)
    reminder = bool(re.search(
        r"(?:я\s+(?:же\s+)?(?:уже\s+)?писал\w*|вс[её]\s+(?:то\s+)?же\s+самое|"
        r"задач[ау]\s+не\s+менял\w*|(?:тот|та)\s+же\s+(?:вопрос|проблема))", message, re.I,
    )) and set(_normalized(message).split()) <= {
        "я", "же", "уже", "тебе", "вам", "выше", "это", "об", "этом", "что", "нужно", "надо",
        "сделать", "написал", "написала", "писал", "писала", "говорил", "говорила", "повторяю",
        "все", "то", "тоже", "самое", "тот", "та", "вопрос", "проблема", "задача", "задачу",
        "не", "менял", "меняла", "менялось", "изменилась",
    }
    if (reminder and flow.get("task_state", {}).get("goals") and flow.get("clarification_question")
            and not _restarts_task(flow["clarification_question"])):
        next_flow = copy.deepcopy(flow)
        previous_question = flow.get("clarification_question", "")
        question = _contextual_question(flow)
        if previous_question and not _restarts_task(previous_question):
            try:
                question = _model_clarification(previous_question, flow["task_state"]["messages"])
            except KnowledgeUnavailable:
                pass
        goals = ", ".join(goal["text"] for goal in flow["task_state"]["goals"].values())
        return Turn("REPLY", [(f"Продолжаем задачу «{goals}».\n\n{question}", None)], flow=next_flow)
    if not flow and not explicit_new_task and _confirmation_enabled(knowledge):
        remembered = _remembered_support_turn(message, context, knowledge)
        if remembered:
            return remembered
    previous_question = flow.get("clarification_question") or flow.get("task_state", {}).get("question", "")
    previous_state = flow.get("task_state")
    if previous_state is None and flow:
        short_term = context.get("short_term", {})
        messages = [row["content"] for row in short_term.get("messages", [])
                    if row.get("role") == "user" and isinstance(row.get("content"), str)]
        messages = messages[-11:]
        while messages and sum(map(len, messages)) > 6000:
            messages.pop(0)
        if messages:
            previous_state = {"facts": [], "goals": {}, "messages": messages}
    guide_context = None
    if flow.get("source") and getattr(llm, "supports_guide_context", False) is True:
        try:
            guide_context = verified_guide_context(knowledge, flow)
        except KnowledgeUnavailable as error:
            report_failure(error)
            retained = {key: copy.deepcopy(flow[key]) for key in (
                "task_state", "task_query", "opening_query", "answered_questions", "excluded",
            ) if key in flow}
            return Turn("REPLY", [(CHANGED, None)], flow=retained)
    elif flow.get("accepted_instruction_offer") and getattr(llm, "supports_guide_context", False) is True:
        guide_context = {"read_only": True, "accepted_instruction_offer": copy.deepcopy(flow["accepted_instruction_offer"])}
    try:
        state = update_task_state(message[:6000], previous_state, previous_question, llm, guide_context=guide_context)
    except Exception as error:
        code = type(error).__name__
        if type(error) is ValueError and str(error) in {
            "InvalidTaskState", "InvalidTaskMessage", "InvalidTaskUnknownAnswer", "InvalidTaskEvidence",
            "InvalidTaskReplacement", "UnexpectedTaskReplacement", "UnknownTaskGoal", "TaskStateTooLarge",
            "IncompleteTaskUnderstanding",
        }:
            code = str(error)
        failure = KnowledgeUnavailable(code, "task_understanding")
        report_failure(failure)
        try:
            saved = remember_task_message(message[:6000], previous_state)
        except ValueError:
            saved = remember_task_message(message[:6000])
        failed_flow = {**flow, "task_state": saved}
        if explicit_new_task:
            saved["new_task"] = True
            failed_flow.update(clarification_query=message[:6000], opening_query=message[:6000])
        return Turn("REPLY", [(UNAVAILABLE, None)], flow=failed_flow)
    next_flow = copy.deepcopy(flow)
    if state["new_task"]:
        next_flow = {}
    answered = list(next_flow.get("answered_questions", []))
    if previous_question and not state["new_task"]:
        answered.append(previous_question)
    next_flow.update(task_state=state, answered_questions=list(dict.fromkeys(answered))[-8:])
    if pending := next_flow.pop("pending_clarification", None):
        next_flow["answered_conditions"] = [*next_flow.get("answered_conditions", []),
                                             {**pending, "answer": message[:6000]}][-8:]
    previous = flow.get("task_state", {})
    if state.get("goal_replacement_evidence") or _corrected_known_fact(previous, state, message):
        next_flow.pop("rejected_scope_ids", None)
        next_flow.pop("rejected_scope_entries", None)
    if state.get("goal_replacement_evidence"):
        next_flow["opening_query"] = message[:6000]
    accepted_offer = next_flow.get("accepted_instruction_offer")
    if accepted_offer:
        if state.get("goal_replacement_evidence") or not set(accepted_offer["accepted_goals"]) <= state["goals"].keys():
            next_flow.pop("accepted_instruction_offer")
        else:
            accepted_offer["accepted_goals"] = {key: state["goals"][key]["text"]
                                                  for key in accepted_offer["accepted_goals"]}
    if (flow.get("source") and getattr(llm, "supports_guide_context", False) is True
            and observational_followup(previous, state, ui_question=is_ui_question(message))):
        return _diagnostic_followup_turn(message, next_flow, context, llm, knowledge)
    def fact_identity(item):
        return (item["key"], _normalized(item["value"]), _normalized(item.get("entity", "")),
                _normalized(item.get("attribute", "")), item.get("negated", False))

    facts = {fact_identity(item) for item in state["facts"]}
    previous_facts = {fact_identity(item) for item in previous.get("facts", [])}
    goals = [(key, _normalized(goal["text"])) for key, goal in state["goals"].items()]
    previous_goals = [(key, _normalized(goal["text"])) for key, goal in previous.get("goals", {}).items()]
    same_task = facts == previous_facts and goals == previous_goals
    if not same_task:
        next_flow.pop("rejected_goal_orders", None)
    if next_flow.get("source") and not same_task:
        next_flow["reselect_source"] = True
    question = state.get("question", "")
    field_name = state.get("question_field")
    known = {fact["key"] for fact in state["facts"] if not fact.get("negated", False)}
    generic_equipment = {"сканер", "принтер", "устройство", "оборудование", "техника"}
    if any(fact["key"] == "equipment_type" and not fact.get("negated", False)
           and _normalized(fact["value"]) in generic_equipment
           for fact in state["facts"]):
        known.discard("equipment_type")
    hardware = any(fact["key"] == "scope" and fact["value"] == "hardware" and not fact.get("negated", False)
                   for fact in state["facts"])
    software = any(fact["key"] == "scope" and fact["value"] in {"application", "software"}
                   and not fact.get("negated", False) for fact in state["facts"])
    answered_field = (field_name in known or field_name == "goal" and (
                          len(state["goals"]) > 1 or next_flow.get("source") and bool(state["goals"]))
                      or field_name == "application" and hardware and bool(state["goals"]))
    source_questions = bool(getattr(llm, "supports_source_clarification", False))
    observable_error = (field_name in {"error_text", "current_step", "symptoms"}
                        and (next_flow.get("source") or known.intersection({"error_text", "current_step", "symptoms"})))
    coarse_question = (field_name == "goal" and not state["goals"]
                       or field_name == "equipment_type" and "equipment_type" not in known
                       and "application" not in known and not software)
    legacy_followup = next_flow.get("source") and not source_questions and field_name == "other"
    ask_before_search = bool(coarse_question or observable_error or legacy_followup)
    if question and not ask_before_search:
        event("talk.question_deferred", question_field=field_name, reason="source_lookup_required")
    if question and not answered_field and ask_before_search:
        try:
            question = _model_clarification(question, task_evidence_authority(message, state, False))
        except KnowledgeUnavailable:
            question = _contextual_question(next_flow)
        repeated = any(_normalized(question) == _normalized(item) for item in answered)
        if not repeated:
            next_flow.update(clarification_query=task_query(state) or message[:6000],
                             clarification_keys=["MODEL"], clarification_question=question,
                             opening_query=next_flow.get("opening_query", message))
            state["question"] = question
            return Turn("REPLY", [(question, None)], flow=next_flow)
    state["question"], state["question_field"] = "", None
    problem = task_query(state)
    if not state["goals"]:
        question = "Что именно нужно сделать? Опишите желаемый результат своими словами."
        next_flow.update(clarification_query=problem or message[:6000], clarification_keys=["MODEL"],
                         clarification_question=question, opening_query=next_flow.get("opening_query", message))
        return Turn("REPLY", [(question, None)], flow=next_flow)
    next_flow["task_query"] = problem
    next_flow.pop("required_source", None)
    if is_unknown_answer(message) and previous_question:
        event("talk.unknown_answer", unknown_question_count=len(state.get("unknown_answers", [])))
    # Step difficulties can retain verified source navigation. New facts/goals
    # and corrections must select against the entire updated task again.
    if not (same_task and _current_step_question(message)):
        next_flow["reselect_source"] = True
    try:
        turn = _retrieve_support_turn(message, next_flow, context, llm, knowledge)
    except KnowledgeUnavailable as error:
        if error.code in {"InstructionPlanSourceChanged", "SourceNotInApprovedKnowledge",
                          "InstructionPlanImageMissing", "InvalidSavedInstructionPlan",
                          "InvalidSavedInstructionProgress"}:
            retained = {key: copy.deepcopy(next_flow[key]) for key in (
                "task_state", "task_query", "opening_query", "answered_questions", "excluded",
            ) if key in next_flow}
            return Turn("REPLY", [(CHANGED, None)], flow={**retained, "clarification_query": problem})
        reply = SELECTION_UNCLEAR if error.code in {
            "InvalidInstructionCoverage", "InvalidSourceClarification", "RepeatedInstructionClarification",
        } else UNAVAILABLE
        if error.code == "SelectionIndexIncomplete":
            reply = INDEX_UPDATING
        return Turn("REPLY", [(reply, None)], flow=next_flow)
    retained = turn.flow if turn.flow is not None else next_flow
    if turn.action in {"REPLY", "NO_KNOWLEDGE"} and not retained.get("understanding_confirmation"):
        retained = {**next_flow, **retained}
    retained.update(task_state=state, answered_questions=next_flow["answered_questions"], task_query=problem)
    if retained.get("clarification_question") and not retained.get("understanding_confirmation"):
        state["question"] = retained["clarification_question"]
        state["question_field"] = None
    if confirmation := retained.get("understanding_confirmation"):
        confirmation["guide_flow"].update(task_state=copy.deepcopy(state), task_query=problem,
                                          answered_questions=next_flow["answered_questions"])
    turn.flow = retained
    return turn


def _diagnostic_followup_turn(message, flow, context, llm, knowledge):
    retained = copy.deepcopy(flow)
    state = retained["task_state"]
    step = "|".join(fact["value"] for fact in state["facts"] if fact["key"] == "current_step") or "current"
    unknown_error = any(answer.get("field") == "error_text" for answer in state.get("unknown_answers", []))
    if needs_error_text(state) and not unknown_error and retained.get("diagnostic_error_requested_for") != step:
        question = "Какой точный текст ошибки появляется на этом шаге?"
        state["question"], state["question_field"] = question, "error_text"
        retained.update(clarification_question=question, diagnostic_error_requested_for=step,
                        clarification_query=retained.get("task_query") or retained.get("query", ""))
        return Turn("REPLY", [(question, None)], flow=retained)
    state["question"], state["question_field"] = "", None
    retained.pop("clarification_question", None)
    retained.pop("reselect_source", None)
    try:
        ui_question = is_ui_question(message)
        answer = diagnostic_excerpt(knowledge, retained, message, {
            **context, "task_state": state, "followup_mode": "ui_explanation" if ui_question else "diagnostic",
        }, llm, allow_repeat=ui_question)
    except KnowledgeUnavailable as error:
        report_failure(error)
        text = CHANGED if error.code == "InstructionPlanSourceChanged" else UNAVAILABLE
        if text == CHANGED:
            retained = {key: retained[key] for key in (
                "task_state", "task_query", "opening_query", "answered_questions", "excluded",
            ) if key in retained}
        return Turn("REPLY", [(text, None)], flow=retained)
    if answer is None:
        return _search_diagnostic_followup(message, retained, context, llm, knowledge)
    if answer["requires_scope_confirmation"]:
        saved = copy.deepcopy(answer["plan"])
        saved.update(cursor=0, shown=[])
        retained.update(instruction_plan=saved, source=copy.deepcopy(answer["source"]), has_more=False,
                        next_offset=0, diagnostic_followup={"source": copy.deepcopy(answer["source"]),
                                                           "section_ids": answer["section_ids"], "new_method": True})
        retained.pop("understanding_confirmed", None)
        parts = restore_plan(knowledge, saved)
        summary = ", затем ".join(goal["text"] for goal in state["goals"].values())
        turn = _confirmation_turn(answer["approved"], retained, summary, parts=parts)
        turn.flow["task_state"] = copy.deepcopy(state)
        return turn
    if ui_question and answer["has_more"]:
        retained["ui_resume_progress"] = {key: copy.deepcopy(retained[key]) for key in (
            "instruction_plan", "source", "has_more", "next_offset", "selection", "shown",
        ) if key in retained}
    if not ui_question:
        retained["diagnostic_followup"] = {"source": copy.deepcopy(answer["source"]), "section_ids": answer["section_ids"]}
    if not ui_question or answer["has_more"]:
        retained.update(instruction_plan=answer["plan"], source=copy.deepcopy(answer["source"]),
                        has_more=answer["has_more"], next_offset=answer["plan"]["cursor"], understanding_confirmed=True)
    elif retained.get("has_more"):
        answer["blocks"][-1] = ("Если пункт найден, напишите «Продолжить» — покажу следующие шаги основной инструкции.", None)
    return Turn("REPEAT_GUIDE", answer["blocks"], approved=answer["approved"], flow=retained,
                block_sources=answer["block_sources"])


def _search_diagnostic_followup(message, flow, context, llm, knowledge):
    search = getattr(knowledge, "search_with_llm", None)
    if not callable(search):
        return Turn("UNRESOLVED", [(NO_DIAGNOSTIC_STEPS, None)], flow=flow)
    original_state = copy.deepcopy(flow["task_state"])
    goals = original_state["goals"]
    observations = [fact["value"] for fact in original_state["facts"]
                    if fact["key"] in {"error_text", "symptoms", "completed"} and not fact.get("negated", False)]
    diagnostic = copy.deepcopy(flow)
    covered_ids = list(dict.fromkeys(goal_id for part in flow.get("instruction_plan", {}).get("parts", [])
                                     for goal_id in part["goal_ids"] if goal_id in goals)) or list(goals)
    diagnostic_goals = {}
    for goal_id in covered_ids:
        query = (goals[goal_id]["text"] + ". Уточнение после выданной инструкции — "
                 + ". ".join(observations or [message]))[:1000]
        diagnostic_goals[goal_id] = {**goals[goal_id], "text": query, "kind": "ACTION", "search_queries": [query[:160]]}
    diagnostic["task_state"]["goals"] = diagnostic_goals
    query = "\n".join(goal["text"] for goal in diagnostic_goals.values())
    previous_sources = [part["source"]["file_id"] for part in flow.get("instruction_plan", {}).get("parts", [])]
    previous_sources.append(flow["source"]["file_id"])
    diagnostic.update(task_query=query, opening_query=message[:6000], reselect_source=True,
                      excluded=list(dict.fromkeys(flow.get("excluded", []) + previous_sources)))
    diagnostic.pop("understanding_confirmed", None)
    diagnostic.pop("diagnostic_followup", None)
    try:
        diagnostic_context = {**context, "diagnostic_request": True,
                              "active_guide_context": verified_guide_context(knowledge, flow)}
        turn = _retrieve_support_turn(message, diagnostic, diagnostic_context, llm, knowledge)
    except KnowledgeUnavailable as error:
        report_failure(error)
        return Turn("REPLY", [(UNAVAILABLE, None)], flow=flow)
    if turn.action == "NO_KNOWLEDGE":
        observation = re.sub(r"([\\`*_{}\[\]()<>])", r"\\\1", _redact(message[:300]))
        notice = "Учёл уточнение «" + observation + "».\n\n" + NO_DIAGNOSTIC_STEPS
        return Turn("UNRESOLVED", [(notice, None)], flow=flow)
    retained = turn.flow if turn.flow is not None else diagnostic
    if not retained.get("understanding_confirmation"):
        retained = {**copy.deepcopy(flow), **retained}
    retained["task_state"] = original_state
    if confirmation := retained.get("understanding_confirmation"):
        guide = confirmation["guide_flow"]
        guide["task_state"] = copy.deepcopy(original_state)
        original_text = {goal["text"]: goals[goal_id]["text"] for goal_id, goal in diagnostic_goals.items()}
        for name in ("covered_goals", "missing_goals"):
            if name in guide:
                guide[name] = [original_text.get(text, text) for text in guide[name]]
        previous_missing = flow.get("missing_goals", []) if flow.get("coverage_mode") == "partial" else []
        if previous_missing:
            guide["missing_goals"] = list(dict.fromkeys(guide.get("missing_goals", []) + previous_missing))
            guide.setdefault("covered_goals", [goals[goal_id]["text"] for goal_id in covered_ids])
            guide["coverage_mode"] = "partial"
            if guide.get("instruction_plan"):
                parts = restore_plan(knowledge, guide["instruction_plan"])
                approved = composite_approved(parts, guide["instruction_plan"])
            else:
                parts = ()
                approved = knowledge.read(confirmation["source"]["file_id"])
            refreshed = _confirmation_turn(approved, guide, confirmation["summary"], parts=parts)
            refreshed.flow["task_state"] = original_state
            return refreshed
    turn.flow = retained
    return turn


def _contextual_question(flow):
    facts = {fact["key"]: fact["value"] for fact in flow.get("task_state", {}).get("facts", [])
             if not fact.get("negated", False)}
    if facts.get("error_text"):
        return "В каком окне появляется это сообщение об ошибке?"
    if flow.get("source") or facts.get("current_step"):
        return "На каком действии возникла трудность и что сейчас происходит?"
    if not flow.get("task_state", {}).get("goals") and not facts.get("symptoms"):
        return "Какой результат вы хотите получить?"
    return "Что сейчас видно на экране или как ведёт себя устройство?"


def _repeated_clarification(question, previous):
    if _normalized(question) == _normalized(previous):
        return True
    numbers = re.findall(r"\d+", question)
    prior_numbers = re.findall(r"\d+", previous)
    if numbers != prior_numbers:
        return False
    current_terms, previous_terms = _terms(question), _terms(previous)
    common = current_terms & previous_terms
    return (len(common) >= 3 and len(common) >= 0.8 * min(len(current_terms), len(previous_terms))
            and len(common) >= 0.6 * len(current_terms | previous_terms))


def _restarts_task(question):
    return bool(re.search(
        r"(?:что\s+именно\s+нужно\s+сделать.{0,40}с\s+каким\s+типом|"
        r"(?:опишите|какая)\s+(?:ваш[уа]\s+)?(?:задач[ау]|проблем[ау])|"
        r"^что\s+(?:именно\s+)?нужно\s+сделать\s*\?|"
        r"^какой\s+(?:именно\s+)?(?:желаемый\s+)?результат\s+(?:вам\s+)?нужен\s*\?)", question, re.I,
    ))


def _remembered_support_turn(message, context, knowledge, excluded=()):
    hint_method = getattr(knowledge, "confirmation_hint", None)
    hint = hint_method(message, excluded=excluded) if callable(hint_method) else None
    if hint:
        if hint.get("plan"):
            parts = restore_plan(knowledge, hint["plan"])
            approved = composite_approved(parts, hint["plan"])
            state = copy.deepcopy(hint["task_state"])
            state["messages"] = [message]
            next_flow = _guide_flow(approved, hint["problem"], message, list(excluded))
            next_flow.update(instruction_plan=hint["plan"], task_state=state, selection=[], shown=[],
                             has_more=True, next_offset=0,
                             retrieval_key=cache_key(knowledge.settings, hint["problem"], {"task_state": state}))
            conditions = []
            for fact in state["facts"]:
                if fact["key"] == "scope":
                    continue
                value = " — ".join(filter(None, [fact.get("entity"), fact.get("attribute"), fact["value"]]))
                conditions.append("Не " + value if fact.get("negated", False) else value)
            summary = ", затем ".join(goal["text"] for goal in state["goals"].values())
            if conditions:
                summary += ". Сообщённые условия — " + ", ".join(conditions)
            turn = _confirmation_turn(approved, next_flow, summary, remembered=True, parts=parts)
            turn.flow["task_state"] = state
            return turn
        approved = knowledge.read(hint["source"]["file_id"])
        if approved.document["source"] == hint["source"]:
            next_flow = _guide_flow(approved, hint["problem"], message, list(excluded))
            next_flow["retrieval_key"] = cache_key(knowledge.settings, hint["problem"], context)
            return _confirmation_turn(approved, next_flow, hint["problem"], remembered=True)
    return None


def _retrieve_support_turn(message, flow, context, llm, knowledge):
    has_guide = bool(flow.get("source"))
    if has_guide and _different_instruction(message):
        replacement = _replace_guide_context(flow)
        message = "\n".join(part for part in (replacement["clarification_query"], message) if part)
        flow = {**{key: copy.deepcopy(flow[key]) for key in (
            "task_state", "task_query", "answered_questions", "opening_query",
        ) if key in flow}, "excluded": replacement["excluded"]}
        has_guide = False
    photo_request = bool(_PHOTO_REQUEST.search(message) or _NEXT_PHOTO.search(message))
    next_photo = bool(photo_request and _NEXT_PHOTO.search(message))
    clarification = bool(has_guide and not flow.get("reselect_source")
                         and (_current_step_question(message) or photo_request and not next_photo))
    continuation = bool(has_guide and not flow.get("reselect_source") and (_step_confirmed(message) or next_photo))
    if flow.get("instruction_plan") and (clarification or continuation):
        try:
            return _guide_reply(None, [], copy.deepcopy(flow), knowledge, clarification)
        except KnowledgeUnavailable as error:
            if error.code not in {"InstructionPlanSourceChanged", "SourceNotInApprovedKnowledge",
                                  "InstructionPlanImageMissing", "InvalidSavedInstructionPlan",
                                  "InvalidSavedInstructionProgress"}:
                raise
            retained = {key: copy.deepcopy(flow[key]) for key in (
                "task_state", "task_query", "opening_query", "answered_questions", "excluded",
            ) if key in flow}
            return Turn("REPLY", [(CHANGED, None)], flow={
                **retained, "clarification_query": flow.get("task_query") or flow.get("query", ""),
            })
    model_selection = bool(getattr(knowledge, "uses_llm_selection", False))
    if _confirmation_enabled(knowledge) and not flow.get("source") and not flow.get("clarification_query"):
        if not flow.get("task_state") and (remembered := _remembered_support_turn(
                message, context, knowledge, flow.get("excluded", []))):
            return remembered
    if not clarification and not continuation and not model_selection:
        if followup := _scanner_clarification(message, flow, knowledge):
            return followup
        if followup := _printer_clarification(message, flow):
            return followup
    if continuation and not flow.get("has_more"):
        return Turn("REPLY", [("Удалось решить проблему? Напишите «Помогло» или «Не помогло».", None)])
    analysis = None
    confirmation_parts = ()
    if clarification or continuation:
        approved = knowledge.read(flow["source"]["file_id"])
        if approved.document["source"] != flow["source"]:
            return Turn("REPLY", [(CHANGED, None)])
        if photo_request and not approved.illustrations:
            return Turn("REPLY", [("В этой инструкции нет фотографий. Напишите, какой шаг нужно пояснить.", None)])
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
        problem = flow.get("task_query") or _problem_query(message, flow)
        excluded = flow.get("excluded", [])
        opening_query = flow.get("opening_query", message)
        if (has_guide and not flow.get("reselect_source")
                or not flow.get("task_state") and problem == message[:6000]
                and problem != flow.get("clarification_query")):
            opening_query = message
        understanding = ""
        if model_selection:
            selection_context = {**context, "current_message": message,
                                 "previous_question": flow.get("clarification_question", ""),
                                 "rejected_scope_ids": flow.get("rejected_scope_ids", []),
                                 "rejected_scope_entries": flow.get("rejected_scope_entries", []),
                                 "pending_clarification": flow.get("pending_clarification"),
                                 "answered_conditions": flow.get("answered_conditions", [])}
            if state := flow.get("task_state"):
                selection_context.update(
                    incident={},
                    short_term={"messages": [{"role": "user", "content": text}
                                             for text in state.get("messages", [])]},
                    requested_goals={key: goal["text"] for key, goal in state["goals"].items()},
                    requested_goal_kinds={key: goal["kind"] for key, goal in state["goals"].items() if "kind" in goal},
                    goal_search_queries={key: goal["search_queries"] for key, goal in state["goals"].items()
                                         if "search_queries" in goal},
                    task_state={"facts": [{key: value for key, value in fact.items() if key != "evidence"}
                                          for fact in state["facts"]],
                                "goals": {key: goal["text"] for key, goal in state["goals"].items()}},
                    answered_questions=flow.get("answered_questions", []),
                    unknown_answers=state.get("unknown_answers", []),
                    source_clarification_required=bool(getattr(llm, "supports_source_clarification", False)),
                    instruction_planning=bool(getattr(llm, "supports_instruction_planning", False)),
                    rejected_goal_orders=flow.get("rejected_goal_orders", []),
                    accepted_instruction_offer=flow.get("accepted_instruction_offer"),
                )
            if chosen := flow.get("required_source"):
                selection_context["verified_previous_source"] = {
                    "file_id": chosen["file_id"], "sha256": chosen["sha256"],
                }
            source = flow.get("source")
            if (has_guide and isinstance(source, dict)
                    and all(isinstance(source.get(name), str) and source[name]
                            for name in ("file_id", "filename"))):
                selection_context["active_task"] = flow.get("query", "")
                selection_context["shown_source"] = {name: source[name] for name in ("file_id", "filename")}
            decision = knowledge.search_with_llm(problem, context=selection_context,
                                                 excluded=excluded, fallback_query=problem)
            question = ""
            if decision.result == "CLARIFY":
                try:
                    reports = task_evidence_authority(message, flow.get("task_state"), False)
                    backend_order_question = (bool(getattr(decision, "proposed_goal_order", ()))
                                              and decision.question == "Связанные действия входят в общие разделы инструкций. "
                                              "Можно выполнить их в предложенном порядке?")
                    question = decision.question if backend_order_question else _model_clarification(
                        decision.question, reports,
                    )
                    if flow.get("task_state", {}).get("goals") and _restarts_task(question):
                        raise KnowledgeUnavailable("RepeatedInstructionClarification", "instruction_selection")
                except KnowledgeUnavailable:
                    if not _confirmation_enabled(knowledge):
                        raise
                    decision = knowledge.search_with_llm(
                        problem, context={**selection_context, "clarification_retry": True},
                        excluded=excluded, fallback_query=problem,
                    )
                    if decision.result == "CLARIFY":
                        raise KnowledgeUnavailable("RepeatedInstructionClarification", "instruction_selection")
                    question = ""
            if proposed_order := getattr(decision, "proposed_goal_order", ()):
                goals = flow.get("task_state", {}).get("goals", {})
                sources = [copy.deepcopy(item.document["source"])
                           for item in decision.clarification_sources]
                if (decision.result != "CLARIFY" or not goals or len(proposed_order) != len(goals)
                        or len(set(proposed_order)) != len(proposed_order) or set(proposed_order) != set(goals)
                        or list(proposed_order) == list(goals) or not sources
                        or not all(knowledge.is_current(source) for source in sources)):
                    raise KnowledgeUnavailable("InvalidGoalOrderProposal", "instruction_selection")
                if list(proposed_order) in flow.get("rejected_goal_orders", []):
                    return Turn("UNRESOLVED", [("Сохраняю ваш порядок действий. Для него подтверждённая "
                        "последовательность пока не найдена.", None)],
                        flow=copy.deepcopy(flow))
                prompt = question + "\n\nПредлагаемый порядок действий\n" + "\n".join(
                    f"{index}. {goals[goal_id]['text']}" for index, goal_id in enumerate(proposed_order, 1)
                ) + "\n\nВы согласны с таким порядком?"
                return Turn("REPLY", [(prompt, None)], flow={
                    **copy.deepcopy(flow), "clarification_query": problem, "clarification_question": question,
                    "opening_query": opening_query,
                    "goal_order_offer": {"goal_order": list(proposed_order), "goals": copy.deepcopy(goals),
                                         "sources": sources, "question": question, "prompt": prompt},
                })
            previous_questions = [flow.get("clarification_question", ""), *flow.get("answered_questions", [])]
            if (decision.result == "CLARIFY" and any(
                    (_normalized(question) == _normalized(previous_question)
                     if getattr(decision, "clarification_condition", None)
                     else _repeated_clarification(question, previous_question))
                    for previous_question in previous_questions if previous_question)):
                decision = knowledge.search_with_llm(
                    problem, context={**selection_context, "clarification_retry": True},
                    excluded=excluded, fallback_query=problem,
                )
                if decision.result == "CLARIFY":
                    raise KnowledgeUnavailable("RepeatedInstructionClarification", "instruction_selection")
            if getattr(decision, "new_request", False) and not flow.get("task_state"):
                problem, opening_query = message[:6000], message
            _apply_goal_kind_overrides(decision, flow)
            if decision.result == "CLARIFY":
                pending = _source_condition(decision, flow, question)
                return Turn("REPLY", [(question, None)], flow={
                    "clarification_query": problem, "clarification_keys": ["MODEL"],
                    "clarification_question": question, "excluded": excluded, "opening_query": opening_query,
                    **({"pending_clarification": pending} if pending else {}),
                })
            if decision.result == "NO_KNOWLEDGE":
                return Turn("NO_KNOWLEDGE", flow={"clarification_query": problem,
                                                  "clarification_keys": flow.get("clarification_keys", []),
                                                  "excluded": excluded, "opening_query": opening_query})
            if decision.result == "RELATED":
                # An alternate setup is not further diagnosis of the active
                # performed guide. Preserve that task when no exact help exists.
                if flow.get("required_source") or context.get("diagnostic_request") is True:
                    return Turn("NO_KNOWLEDGE", flow={"clarification_query": problem})
                choices = [copy.deepcopy(item.document["source"]) for item in decision.alternatives]
                if not choices or not all(knowledge.is_current(source) for source in choices):
                    raise KnowledgeUnavailable("InvalidInstructionOffer", "instruction_selection")
                alternative_goals = copy.deepcopy(getattr(decision, "alternative_goals", {}))
                goals = flow.get("task_state", {}).get("goals", {})
                if (not isinstance(alternative_goals, dict)
                        or set(alternative_goals) - {source["file_id"] for source in choices}
                        or any(not isinstance(mapping, dict) or not mapping or not set(mapping) <= goals.keys()
                               or any(not isinstance(text, str) or not text.strip() or len(text) > 250
                                      for text in mapping.values()) for mapping in alternative_goals.values())):
                    raise KnowledgeUnavailable("InvalidInstructionOfferIntent", "instruction_selection")
                text = "В нашей базе есть похожие инструкции. Возможно, вы имели в виду один из этих вариантов?\n\n"
                descriptions = []
                for index, source in enumerate(choices, 1):
                    description = f"{index}. {source['filename']}"
                    if selected_goals := alternative_goals.get(source["file_id"]):
                        description += "\nПредлагаемый вариант — " + ", затем ".join(selected_goals.values()) + "."
                    descriptions.append(description)
                text += "\n\n".join(descriptions)
                text += "\n\nНапишите номер подходящего варианта или «нет», если они вам не подходят."
                return Turn("REPLY", [(text, None)], flow={
                    "instruction_offer": {"problem": problem, "choices": choices, "alternative_goals": alternative_goals},
                    "clarification_query": problem, "excluded": excluded, "opening_query": opening_query,
                })
            if decision.result not in {"MATCH", "PARTIAL"} or decision.approved is None:
                raise KnowledgeUnavailable("InvalidInstructionDecision", "instruction_selection")
            approved = decision.approved
            if (flow.get("required_source")
                    and approved.document["source"] != flow["required_source"]):
                raise KnowledgeUnavailable("InstructionOfferSourceChanged", "instruction_selection")
            understanding = getattr(decision, "understanding", "")
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
            if not flow.get("task_state") and not flow.get("instruction_plan"):
                return Turn("REPLY", [(CHANGED, None)])
            retained = {key: copy.deepcopy(flow[key]) for key in (
                "task_state", "task_query", "opening_query", "answered_questions", "excluded",
            ) if key in flow}
            return Turn("REPLY", [(CHANGED, None)], flow={**retained, "clarification_query": problem})
        next_flow = _guide_flow(approved, problem, opening_query, excluded)
        for name in ("rejected_scope_ids", "rejected_scope_entries", "accepted_instruction_offer",
                     "answered_questions", "answered_conditions"):
            if flow.get(name):
                next_flow[name] = copy.deepcopy(flow[name])
        if model_selection and getattr(decision, "plan", ()):
            confirmation_parts = decision.plan
            saved_plan = serialize_plan(decision.plan)
            approved = composite_approved(decision.plan, saved_plan)
            next_flow.update(instruction_plan=saved_plan, selection=[], shown=[], has_more=True, next_offset=0)
        if state := flow.get("task_state"):
            next_flow["task_state"] = copy.deepcopy(state)
            next_flow["understanding"] = ", затем ".join(goal["text"] for goal in state["goals"].values())
        if model_selection and decision.result == "PARTIAL":
            next_flow.update(coverage_mode="partial", covered_goals=list(decision.covered_goals),
                             missing_goals=list(decision.missing_goals))
        selected = next_flow["shown"]
        if understanding and not flow.get("task_state"):
            next_flow["understanding"] = understanding
        retrieval_key = getattr(knowledge, "last_cache_key", None)
        if model_selection and getattr(decision, "new_request", False) and not flow.get("task_state"):
            retrieval_key = cache_key(knowledge.settings, problem, context)
        if retrieval_key and next_flow.get("coverage_mode") != "partial":
            next_flow["retrieval_key"] = retrieval_key
    return _guide_reply(approved, selected, next_flow, knowledge, clarification, analysis,
                        confirmation_parts=confirmation_parts)


def _guide_reply(approved, selected, next_flow, knowledge, clarification=False, analysis=None, *,
                 confirmation_parts=()):
    goals = next_flow.get("task_state", {}).get("goals", {})
    lookup_only = bool(goals) and all(goal.get("kind") == "LOOKUP" for goal in goals.values())
    lookup_only = lookup_only and next_flow.get("coverage_mode") not in {"partial", "partial_completed"}
    if saved_plan := next_flow.get("instruction_plan"):
        parts = restore_plan(knowledge, saved_plan)
        approved = composite_approved(parts, saved_plan)
        if _confirmation_enabled(knowledge) and not next_flow.get("understanding_confirmed"):
            summary = next_flow.get("understanding") or next_flow["query"]
            return _confirmation_turn(approved, next_flow, summary, parts=confirmation_parts or parts)
        blocks, sources, saved_plan, more = render_plan(parts, saved_plan, repeat=clarification)
        if lookup_only and not more:
            blocks[-1] = ("Нашли нужную информацию?", None)
        next_flow.update(instruction_plan=saved_plan, has_more=more, next_offset=saved_plan["cursor"])
        ui_navigation = next_flow.get("ui_resume_progress")
        if ui_navigation and not more:
            if ui_navigation.get("instruction_plan"):
                restore_plan(knowledge, ui_navigation["instruction_plan"])
            next_flow.update(copy.deepcopy(ui_navigation))
            next_flow.pop("ui_resume_progress", None)
            if next_flow.get("has_more"):
                blocks[-1] = ("Если пункт найден, напишите «Продолжить» — покажу следующие шаги основной инструкции.", None)
        if next_flow.get("coverage_mode") in {"partial", "partial_completed"}:
            blocks.insert(0, (_partial_boundary(next_flow), None))
            sources.insert(0, None)
        diagnostic = next_flow.get("diagnostic_followup", {})
        action = "REPEAT_GUIDE" if (next_flow.get("coverage_mode") == "partial_completed"
                                    or diagnostic and not diagnostic.get("new_method") or ui_navigation) else "GUIDE"
        return Turn(action, blocks, approved, next_flow, analysis, block_sources=sources)
    if not knowledge.is_current(approved.document["source"]):
        return Turn("REPLY", [(CHANGED, None)])
    if _confirmation_enabled(knowledge) and not next_flow.get("understanding_confirmed"):
        summary = next_flow.get("understanding") or next_flow["query"]
        return _confirmation_turn(approved, next_flow, summary)
    partial = next_flow.get("coverage_mode") in {"partial", "partial_completed"}
    action = "REPEAT_GUIDE" if next_flow.get("coverage_mode") == "partial_completed" else "GUIDE"
    if not selected:
        text = approved.document["content"].split("Фотографии и пояснения", 1)[0].strip()
        if partial:
            text = _partial_boundary(next_flow) + "\n\n" + text
        question = "Нашли нужную информацию?" if lookup_only else "Удалось выполнить эти действия?"
        return Turn(action, [(text + "\n\n" + question, None)],
                    approved, next_flow, analysis)
    blocks = _photo_blocks(approved, selected, next_flow["has_more"], clarification)
    if lookup_only and not next_flow["has_more"]:
        blocks[-1] = ("Нашли нужную информацию?", None)
    if partial:
        blocks.insert(0, (_partial_boundary(next_flow), None))
    return Turn(action, blocks,
                approved, next_flow, analysis)


def _offer_purpose_turn(message, flow, context, llm, knowledge):
    retained = copy.deepcopy(flow)
    offer = retained["instruction_offer"]
    if offer.get("purpose_explained"):
        retained.pop("instruction_offer")
        retained["clarification_question"] = offer.get("purpose_question", OFFER_CHOICE)
        return _support_turn(message, retained, context, llm, knowledge)
    descriptions, entries = [], []
    for number, source in enumerate(offer["choices"], 1):
        approved = knowledge.read(source["file_id"])
        if approved.document["source"] != source or not knowledge.is_current(source):
            return Turn("REPLY", [(CHANGED, None)], flow=retained)
        scopes = confirmation_scope_entries(approved)
        entries.extend(scopes)
        title = source["filename"]
        paragraphs = [item["text"] for entry in scopes for item in entry["context"]]
        description = f"{number}. «{title}»"
        if paragraphs:
            description += "\n" + "\n".join("«" + text + "»" for text in paragraphs)
        descriptions.append(description)
    question = "Какой из описанных результатов вам нужен? Можно ответить своими словами."
    offer.update(purpose_explained=True, purpose_question=question, scope_entries=entries)
    retained["clarification_question"] = question
    return Turn("REPLY", [("\n\n".join(descriptions) + "\n\n" + question, None)], flow=retained)


def _instruction_offer_turn(message, flow, context, llm, knowledge, settings):
    offer = flow["instruction_offer"]
    choices = offer.get("choices", [])
    if not isinstance(choices, list) or not 1 <= len(choices) <= 3:
        raise KnowledgeUnavailable("InvalidSavedInstructionOffer", "instruction_selection")
    normalized = _normalized(message).replace("ё", "е")
    if normalized in {"отмена", "отменить", "вернуться к боту", "продолжить с ботом", "продолжить диалог с ботом"}:
        return Turn("REPLY", [(BACK_TO_BOT, None)], flow={})
    if is_unknown_answer(message) or re.search(r"разниц\w*|отлич\w*|для\s+чего|что\s+(?:означ\w*|дела\w*)", message, re.I):
        return _offer_purpose_turn(message, flow, context, llm, knowledge)
    index = None
    consent_evidence = message.strip()
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
                    consent_evidence = result.evidence
                    if index is None:
                        raise KnowledgeUnavailable("InvalidOfferChoice", "instruction_selection")
                else:
                    if result.choice is not None:
                        raise KnowledgeUnavailable("InvalidOfferChoice", "instruction_selection")
                    rejected = True
            elif result.choice is not None or result.evidence:
                raise KnowledgeUnavailable("InvalidOfferReply", "instruction_selection")
            elif result.result == "NEW_REQUEST":
                retained = copy.deepcopy(flow)
                retained.pop("instruction_offer")
                if retained.get("clarification_query") == offer.get("problem"):
                    retained.pop("clarification_query", None)
                turn = _text_turn(message, retained, context, None, llm, knowledge, settings)
                turn.flow = turn.flow if turn.flow is not None else flow
                return turn
        elif _technical_message(message):
            retained = copy.deepcopy(flow)
            retained.pop("instruction_offer")
            if retained.get("clarification_query") == offer.get("problem"):
                retained.pop("clarification_query", None)
            turn = _text_turn(message, retained, context, None, llm, knowledge, settings)
            turn.flow = turn.flow if turn.flow is not None else flow
            return turn
    if rejected:
        retained = {key: copy.deepcopy(value) for key, value in flow.items() if key != "instruction_offer"}
        retained.update(clarification_query=offer.get("problem", ""), declined_instruction_offer=True)
        return Turn("NO_KNOWLEDGE", flow=retained)
    if index is None or not 1 <= index <= len(choices):
        return _offer_purpose_turn(message, flow, context, llm, knowledge)
    source = choices[index - 1]
    recheck = None
    if flow.get("task_state", {}).get("goals"):
        # Consent chooses a source to check, without proving that it supports
        # every outstanding action or changing the employee's equipment facts.
        recheck = {key: copy.deepcopy(flow[key]) for key in (
            "task_state", "task_query", "opening_query", "answered_questions", "excluded",
        ) if key in flow}
        selected_goals = offer.get("alternative_goals", {}).get(source["file_id"])
        if selected_goals:
            try:
                with knowledge_stage("offer_adoption"):
                    state, retired = adopt_instruction_offer(message, consent_evidence, recheck["task_state"], source,
                                                             selected_goals, llm)
            except KnowledgeUnavailable as error:
                report_failure(error)
                return Turn("REPLY", [(UNAVAILABLE, None)], flow=copy.deepcopy(flow))
            recheck.update(task_state=state, task_query=task_query(state))
            recheck["accepted_instruction_offer"] = {
                "source": copy.deepcopy(source), "accepted_goals": copy.deepcopy(selected_goals),
                "choice_evidence": consent_evidence, "superseded_requested_fact_ids": retired,
            }
            recheck.pop("answered_questions", None)
            simple_choice = set(normalized.split()) <= {
                "1", "2", "3", "да", "ага", "хорошо", "подходит", "выбираю", "выбрал", "выбрала", "первый", "второй",
                "третий", "этот", "тот", "вариант", "инструкцию", "способ", "пожалуйста", "давайте", "согласен",
                "согласна", "именно", "это", "он", "мне", "нам", "будем", "попробуем", "пожалуй",
            }
            if not simple_choice:
                # Adoption records the choice once. The normal parser now reads
                # the same utterance for additional employee-provided conditions.
                recheck["task_state"]["messages"] = copy.deepcopy(flow["task_state"].get("messages", []))
                recheck.pop("required_source", None)
                return _support_turn(message, recheck, context, llm, knowledge)
    try:
        approved = knowledge.read(source["file_id"])
    except KnowledgeUnavailable as error:
        report_failure(error)
        retained = {key: copy.deepcopy(value) for key, value in (recheck or flow).items()
                    if key not in {"instruction_offer", "required_source"}}
        retained.update(clarification_query=retained.get("task_query") or offer.get("problem", ""), reselect_source=True)
        notice = CHANGED if error.code == "SourceNotInApprovedKnowledge" else UNAVAILABLE
        return Turn("REPLY", [(notice, None)], flow=retained)
    if approved.document["source"] != source:
        retained = {key: copy.deepcopy(value) for key, value in (recheck or flow).items()
                    if key not in {"instruction_offer", "required_source"}}
        retained.update(clarification_query=retained.get("task_query") or offer.get("problem", ""), reselect_source=True)
        return Turn("REPLY", [(CHANGED, None)], flow=retained)
    if recheck is not None:
        recheck.update(required_source=copy.deepcopy(source), task_query=recheck.get("task_query") or offer["problem"])
        turn = _retrieve_support_turn(message, recheck, context, llm, knowledge)
        turn.flow = {**recheck, **(turn.flow or {})}
        return turn
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
    explicit_new_task = starts_new_task(message)
    if explicit_new_task and previous_pending == FORM_ACTION and not requests_support_form(message):
        return _continue_from_letter(message, {}, context, llm, knowledge, settings)
    if explicit_new_task:
        flow, previous_pending = {}, None
    form_request = requests_support_form(message)
    filling_form = previous_pending == FORM_ACTION and flow.get("escalation", {}).get("stage") != "CHOICE"
    if form_request and not filling_form:
        if flow.get("escalation", {}).get("profile_prefilled"):
            return _intake_prompt(_prefill_intake(flow, context))
        return _profile_form_request(flow, context)
    if previous_pending != FORM_ACTION and context.get("support_offer_pending"):
        reply = _normalized(message)
        if "?" not in message and reply in {"да", "давайте", "да составь", "да составьте", "составь", "составьте"}:
            return _profile_form_request(flow, context)
        if "?" not in message and reply in {"нет", "не надо", "нет не надо", "не нужно", "нет не нужно",
                                           "не надо спасибо", "нет спасибо", "не сейчас", "письмо не нужно"}:
            return Turn("SUPPORT_DECLINED", [("Хорошо, продолжим.", None)])
    intent = direct_intent(message)
    if previous_pending == FORM_ACTION:
        choice = _escalation_choice(message)
        if choice == "RESUME":
            return _resume_turn(flow, context, llm, knowledge)
        if choice == "FORM_REQUEST" or (intent == "HUMAN" and not form_request):
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
    if flow.get("understanding_confirmation"):
        if intent == "HUMAN":
            return Turn("HUMAN")
        if reply := _simple_social(message):
            return Turn("REPLY", [(reply, None)])
        try:
            return _understanding_turn(message, flow, context, llm, knowledge, settings)
        except KnowledgeUnavailable as error:
            report_failure(error)
            return Turn("REPLY", [(UNAVAILABLE, None)])
    if flow.get("goal_order_offer"):
        if intent == "HUMAN":
            return Turn("HUMAN")
        try:
            return _goal_order_turn(message, flow, context, llm, knowledge, settings)
        except KnowledgeUnavailable as error:
            report_failure(error)
            return Turn("REPLY", [(UNAVAILABLE, None)])
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
    completed_part = flow.get("coverage_mode") == "partial_completed"
    if completed_part and resolution_feedback(message) == "SUCCESS":
        return Turn("REPLY", [("Спасибо, результат этой части уже учтён. " + _partial_boundary(flow), None)])
    feedback = None if completed_part else _feedback_kind(message, flow)
    if feedback:
        if feedback == "SUCCESS":
            return Turn("PARTIAL_SUCCESS" if flow.get("coverage_mode") == "partial" else "SUCCESS")
        if flow.get("source"):
            search_flow = {key: copy.deepcopy(flow[key]) for key in (
                "task_state", "task_query", "opening_query", "answered_questions",
            ) if key in flow}
            search_flow["excluded"] = list(dict.fromkeys(flow.get("excluded", []) + [flow["source"]["file_id"]]))
            try:
                following = _support_turn(flow.get("query") or message, search_flow, context, llm, knowledge)
            except KnowledgeUnavailable:
                return Turn("REPLY", [(UNAVAILABLE, None)])
            if following.action == "GUIDE":
                following.action = "RETRY_GUIDE"
                return following
            if following.action == "REPLY" and following.flow and any(following.flow.get(key) for key in (
                "clarification_question", "understanding_confirmation", "instruction_offer", "goal_order_offer",
            )):
                following.failed_previous = True
                return following
        return Turn("FAILED")
    if _step_confirmed(message) and not flow:
        return Turn("REPLY", [("Опишите, пожалуйста, с какой рабочей проблемой нужна помощь.", None)])
    pinned = (flow.get("source") and (_step_confirmed(message) or _current_step_question(message) or _different_instruction(message)
                                      or _PHOTO_REQUEST.search(message) or _NEXT_PHOTO.search(message))) or (
        flow.get("clarification_query") and len(message.split()) <= 8
    )
    if not explicit_new_task and not _technical_message(message) and not pinned:
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
        "Текущий шаг сохранён. Напишите вопрос ещё раз.", None,
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
        offer = (room.profile or {}).get(SUPPORT_OFFER_KEY, {})
        if offered_message := offer.get("message_id"):
            context["support_offer_pending"] = bool(session.scalar(select(Delivery.id).where(
                Delivery.message_id == offered_message, Delivery.room_token == room.room_token,
                Delivery.account_id == room.account_id, Delivery.status == "SENT",
            ).limit(1)))
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


def _deliver(session, incident, blocks, source=None, block_sources=None):
    # Root integration supplies this method so one captioned original photo stays
    # paired with its exact text in the durable Talk outbox.
    if block_sources is None:
        return messenger.send_blocks(session, incident.id, blocks, source=source)
    if len(block_sources) != len(blocks):
        raise ValueError("InvalidBlockSources")
    first = None
    for block, provenance in zip(blocks, block_sources, strict=True):
        sent = messenger.send_blocks(session, incident.id, [block], source=provenance)
        first = first or sent
    return first


def _offer_escalation(session, incident, reason, flow, context):
    pending = _attempt(session, incident)
    saved_flow = copy.deepcopy(flow)
    existing = saved_flow.pop("escalation", None)
    if existing and existing.get("resume"):
        resume = existing["resume"]
    else:
        if pending is None and not saved_flow.get("understanding_confirmation"):
            # A failed/exhausted guide must not become a fresh waiting attempt
            # simply because the employee backs out of the support menu.
            problem = _last_problem(saved_flow, context)
            retained = {key: copy.deepcopy(saved_flow[key]) for key in (
                "task_state", "task_query", "opening_query", "answered_questions",
            ) if key in saved_flow}
            saved_flow = {**retained, "clarification_query": problem, "excluded": saved_flow.get("excluded", [])} if problem else retained
            if flow.get("declined_instruction_offer"):
                saved_flow["declined_instruction_offer"] = True
        resume = {"flow": saved_flow, "state": incident.state if pending else "DIAGNOSING",
                  "result": incident.result if pending else None}
    problem = resume["query"] if "query" in resume else _last_problem(resume["flow"], context)
    resume["query"] = problem
    if problem:
        incident.description = problem[-8000:]
    incident.state, incident.pending_action = "DIAGNOSING", FORM_ACTION
    incident.result = "ESCALATE"
    incident.escalation_reason = reason
    _deliver(session, incident, [(ESCALATION_MENU, None)])
    audit(session, incident.id, "incident.escalation_choice_requested", reason=reason)
    return {**flow, "escalation": {"stage": "CHOICE", "resume": resume}}


def _advance_support_offer(room, visible, declined):
    profile = room.profile or {}
    if offer := profile.get(SUPPORT_OFFER_KEY):
        remaining = 0 if offer.get("message_id") and not visible else max(0, offer["remaining"] - 1)
        if declined:
            remaining = SUPPORT_OFFER_INTERVAL
        room.profile = {**profile, SUPPORT_OFFER_KEY: {"remaining": remaining}}


def _deliver_unresolved(session, room, incident, text):
    profile = room.profile or {}
    if profile.get(SUPPORT_OFFER_KEY, {}).get("remaining", 0):
        _deliver(session, incident, [(text, None)])
        return
    message = _deliver(session, incident, [(text + "\n\n" + SUPPORT_OFFER, None)])
    room.profile = {**profile, SUPPORT_OFFER_KEY: {"remaining": SUPPORT_OFFER_INTERVAL, "message_id": message.id}}
    audit(session, incident.id, "talk.support_letter_offered")


def _no_solution_flow(flow, context):
    retained = {key: copy.deepcopy(flow[key]) for key in (
        "task_state", "task_query", "opening_query", "answered_questions", "excluded",
        "rejected_scope_ids", "rejected_scope_entries", "answered_conditions",
    ) if key in flow}
    if problem := _last_problem(flow, context):
        retained["clarification_query"] = problem
    return retained


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
        _advance_support_offer(room, claimed["context"].get("support_offer_pending"), turn.action == "SUPPORT_DECLINED")
        first_sequence = session.scalar(select(func.coalesce(func.max(Message.sequence), 0)).where(
            Message.incident_id == incident.id,
        ))
        append_message(session, incident.id, "user", claimed["message"], "RECEIVED")
        if turn.confirmed_computer:
            remember_computer(room, incident.id, turn.confirmed_computer)
        if turn.media_notice:
            _deliver(session, incident, [(turn.media_notice, None)])
        pending = _attempt(session, incident)
        if pending and (turn.action == "RETRY_GUIDE" or turn.failed_previous):
            pending.status = "FAILED"
            session.add(Feedback(attempt_id=pending.id, helped=False, comment=""))
            incident.failed_attempts += 1
            if turn.failed_previous:
                incident.state, incident.result = "DIAGNOSING", None
            session.flush()
            pending = None
        if turn.action == "DELIVERY_RESUME":
            if not resume_response(session, room, incident, inbox, turn.resume or {}):
                _deliver(session, incident, [("Предыдущий ответ уже завершён или его нельзя безопасно продолжить. "
                                                "Напишите, с чем нужна помощь.", None)])
        elif turn.action == "REPEAT_GUIDE":
            _deliver(session, incident, turn.blocks, source=turn.approved.document["source"],
                     block_sources=turn.block_sources)
            room.flow = turn.flow
        elif turn.action in {"GUIDE", "RETRY_GUIDE"}:
            source = turn.approved.document["source"]
            existing = session.scalar(select(SolutionAttempt).where(
                SolutionAttempt.incident_id == incident.id,
                SolutionAttempt.solution_id == turn.approved.candidate.solution_id,
            ))
            same = bool(existing is not None and pending is not None and existing.id == pending.id)
            if existing and existing.evidence.get("talk_source") != source:
                _deliver(session, incident, [(CHANGED, None)])
            elif existing and existing.status not in {"UNASSESSED", "WAITING_FEEDBACK"}:
                # A failed solution already owns its unique feedback record.
                # Do not repeat it as a new attempt or silently reset the result.
                room.flow = _no_solution_flow(turn.flow, claimed["context"])
                incident.state, incident.result, incident.escalation_reason = "DIAGNOSING", "NO_KNOWLEDGE", None
                if pending:
                    pending.status = "UNASSESSED"
                _deliver_unresolved(session, room, incident, NO_OTHER_SOLUTION)
            else:
                if not same and pending:
                    pending.status = "UNASSESSED"
                    session.flush()  # Release uq_pending_attempt before reactivating another saved attempt.
                save_snapshot(session, turn.approved.candidate)
                first = _deliver(session, incident, turn.blocks, source=source, block_sources=turn.block_sources)
                if not same:
                    if existing is not None:
                        existing.status, existing.message_id = "WAITING_FEEDBACK", first.id
                    else:
                        candidate = turn.approved.candidate
                        evidence = {"title": candidate.title, "version": candidate.version,
                                    "chunk_id": candidate.chunk_id, "content": candidate.content,
                                    "steps": candidate.steps, "source": candidate.source, "talk_source": source}
                        if saved_plan := turn.flow.get("instruction_plan"):
                            evidence["instruction_plan"] = immutable_plan(saved_plan)
                        previous_ordinal = session.scalar(select(func.max(SolutionAttempt.ordinal)).where(
                            SolutionAttempt.incident_id == incident.id,
                        )) or 0
                        session.add(SolutionAttempt(
                            incident_id=incident.id, solution_id=candidate.solution_id, ordinal=previous_ordinal + 1,
                            message_id=first.id, evidence=evidence,
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
        elif turn.action in {"SUCCESS", "PARTIAL_SUCCESS", "FAILED"}:
            if not pending:
                _deliver(session, incident, [("Сначала опишите, что не работает, чтобы я мог подобрать инструкцию.", None)])
            else:
                helped = turn.action in {"SUCCESS", "PARTIAL_SUCCESS"}
                pending.status = "SUCCEEDED" if helped else "FAILED"
                session.add(Feedback(attempt_id=pending.id, helped=helped, comment=""))
                if turn.action == "PARTIAL_SUCCESS":
                    remaining = ", ".join((room.flow or {}).get("missing_goals", []))
                    saved = copy.deepcopy(room.flow)
                    saved["opening_query"] = ""
                    state = saved.get("task_state")
                    if state:
                        state["goals"] = {key: goal for key, goal in state["goals"].items()
                                          if goal["text"] in saved["missing_goals"]}
                        state["question"], state["question_field"] = "", None
                        saved["task_query"] = task_query(state)
                    room.flow = {**saved, "coverage_mode": "partial_completed"}
                    incident.state, incident.pending_action, incident.result = "DIAGNOSING", None, "PARTIAL"
                    _deliver_unresolved(session, room, incident, "Спасибо за обратную связь. Подтверждённая часть выполнена. "
                        f"Для действий «{remaining}» подтверждённых шагов пока нет.")
                elif helped:
                    incident.state, incident.pending_action, incident.closed_at = "RESOLVED", None, now()
                    session.flush()
                    remember_resolution(session, incident, settings)
                    if (not has_unfinished_guide(session, room, pending) and not (room.flow or {}).get("has_more")
                            and not (room.flow or {}).get("diagnostic_followup")):
                        remember_success(session, settings, claimed["principal"], incident.id,
                                         (room.flow or {}).get("retrieval_key"), pending.evidence.get("talk_source"),
                                         opening_query=(room.flow or {}).get("opening_query")
                                         if settings.talk_confirm_understanding else None,
                                         plan=(room.flow or {}).get("instruction_plan"),
                                         task_state=(room.flow or {}).get("task_state"))
                    room.flow = {}
                    _deliver(session, incident, [("Спасибо за обратную связь. Рад, что удалось решить проблему.", None)])
                else:
                    incident.failed_attempts += 1
                    excluded = list(dict.fromkeys((room.flow or {}).get("excluded", []) + [pending.evidence["talk_source"]["file_id"]]))
                    room.flow = _no_solution_flow({**(room.flow or {}), "excluded": excluded}, claimed["context"])
                    incident.state, incident.result, incident.escalation_reason = "DIAGNOSING", "NO_KNOWLEDGE", None
                    _deliver_unresolved(session, room, incident, NO_OTHER_SOLUTION)
        elif turn.action == "HUMAN":
            room.flow = _offer_escalation(session, incident, "USER_REQUEST", turn.flow or room.flow, claimed["context"])
        elif turn.action == "NO_KNOWLEDGE":
            previous = turn.flow or room.flow
            same_guide = pending is not None and previous.get("source") == pending.evidence.get("talk_source")
            if same_guide:
                room.flow = previous
                incident.state = "WAITING_FEEDBACK"
            else:
                room.flow = _no_solution_flow(previous, claimed["context"])
                if pending:
                    pending.status = "UNASSESSED"
                incident.state = "DIAGNOSING"
            incident.pending_action, incident.result = None, "NO_KNOWLEDGE"
            incident.escalation_reason = None
            if problem := room.flow.get("clarification_query"):
                incident.description = problem[-8000:]
            text = NO_INSTRUCTION
            if previous.get("declined_instruction_offer"):
                text += " Предложенные варианты вам не подошли."
            _deliver_unresolved(session, room, incident, text)
        elif turn.action == "UNRESOLVED":
            room.flow = turn.flow if turn.flow is not None else room.flow
            if incident.pending_action == FORM_ACTION:
                incident.pending_action, incident.escalation_reason = None, None
            _deliver_unresolved(session, room, incident, turn.blocks[0][0])
        elif turn.action == "FORM_REQUEST":
            incident.pending_action = FORM_ACTION
            incident.escalation_reason = incident.escalation_reason or "USER_REQUEST"
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
            _deliver(session, incident, turn.blocks, block_sources=turn.block_sources)
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
                                               "understand_task": "model_api", "followup": "model_api",
                                               "extract_escalation": "model_api", "interpret_instruction_offer": "model_api",
                                               "adopt_instruction_offer": "model_api",
                                               "interpret_understanding_reply": "model_api"})
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

        read_instruction = getattr(llm, "read_instruction", None)

        def read_source(payload):
            with _measure(metadata, "model_api"):
                return read_instruction(payload)

        knowledge_options = {"selector": select_source}
        if settings.llm_backend == "openwebui" and callable(read_instruction):
            knowledge_options["reader"] = read_source
            nominate_instructions = getattr(llm, "nominate_instructions", None)
            if callable(nominate_instructions):
                def nominate_sources(payload):
                    with _measure(metadata, "model_api"):
                        return nominate_instructions(payload)

                knowledge_options["nominator"] = nominate_sources
            related_instruction = getattr(llm, "related_instruction", None)
            if callable(related_instruction):
                def read_related(payload):
                    with _measure(metadata, "model_api"):
                        return related_instruction(payload)

                knowledge_options["related_reader"] = read_related
        raw_knowledge = knowledge or TalkKnowledge(settings, **knowledge_options)
        if callable(getattr(raw_knowledge, "set_catalog", None)):
            from app.services.rag.catalog import KnowledgeCatalog

            raw_knowledge.set_catalog(KnowledgeCatalog(settings, factory))
        cached_knowledge = SuccessfulTalkKnowledge(
            raw_knowledge, factory, settings,
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
                saved_flow = paused.get("flow", {})
                source = saved_flow.get("source")
                current = True
                if saved_flow.get("instruction_plan"):
                    try:
                        restore_plan(timed_knowledge, saved_flow["instruction_plan"])
                    except KnowledgeUnavailable:
                        current = False
                elif source:
                    current = timed_knowledge.is_current(source)
                if not current:
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
