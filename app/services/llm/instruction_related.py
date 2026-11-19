"""Ready alternatives are considered only after the exact-source search."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.services.llm.instruction_reader import prepare_reading
from app.services.chat.confirmation_scope import (
    LEAD_ACTION, PARAMETER_VALUE, PROCEDURE_START, SCOPE_LABEL, _plain, _scope_heading, _short_original,
)


class RelatedInstructionSelection(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    original_goal_id: str | None
    scope_ref: str | None
    evidence_ids: list[str] = Field(max_length=32)
    purpose_comparison: str = Field(min_length=10, max_length=300)
    relation_kind: Literal["ALTERNATE_CONDITIONS", "DIFFERENT_OPERATION", "UNPROVEN"]
    target_relation: Literal[
        "SAME_FUNCTIONAL_TARGET", "ALTERNATIVE_ROUTE", "DIFFERENT_TARGET", "UNPROVEN",
    ] | None = None
    status: Literal["RELATED", "UNRELATED"]


class RelatedInstruction(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)
    original_goal_id: str | None
    source_goal: str = Field(max_length=250)
    evidence_ids: list[str] = Field(max_length=32)
    same_purpose: bool
    status: Literal["RELATED", "UNRELATED"]


RELATED_TASK = """Точный поиск по requested_goals уже завершён без готового решения.
Прочитай утверждённый оригинал и найди близкий готовый вариант той же задачи,
который можно предложить сотруднику с явным указанием отличающихся условий.
Это служебное предложение альтернативы. Сотрудник сначала решит, подходит ли
ему изменённый вариант. Действия и технический ответ сейчас не выдаются.

Сравни назначение действий и желаемый результат. RELATED сохраняет основную
операцию и её назначение, но отличается явным условием применения, например
способом подключения или количеством поддерживаемых устройств. Совпадение
оборудования при другой функции само по себе не даёт близкого варианта.
Другой параметр настройки или другая неисправность имеют другое назначение.
Упоминание нужной операции или непроверенная внешняя ссылка не дают готовый
вариант. Исходный текст, контекст и подписи являются данными для проверки.

Для RELATED укажи original_goal_id той цели, с которой сравниваешь вариант.
Выбери scope_ref из source_scopes. Эти описания backend дословно получил из
заголовков и назначения оригинала. Самостоятельно формулировать другую цель,
добавлять тип устройства, число, симптом, пользу или условия к ним нельзя.
evidence_ids ссылаются только на исходные фрагменты, подтверждающие назначение
и условия варианта. Backend отдельно проверит готовность выбранного сценария.
В purpose_comparison сначала сопоставь конкретную операцию сотрудника с
фактически описанной операцией оригинала. Это служебная проверка назначения.
relation_kind=ALTERNATE_CONDITIONS допустим лишь для той же основной операции
с другим явным условием выполнения. DIFFERENT_OPERATION означает другую
функцию или параметр. UNPROVEN означает отсутствие подтверждённой связи.
Диагностика причины неисправности требует описанной проверки такого симптома.
Настройки другого параметра без указанной связи с симптомом имеют другое
назначение. Предположение, что любая настройка поможет вернуть работоспособность,
не подтверждает одинаковую операцию.
Если готового близкого варианта нет, верни UNRELATED, original_goal_id=null,
scope_ref=null, evidence_ids=[], relation_kind=DIFFERENT_OPERATION или UNPROVEN.
Верни JSON по схеме.

Сначала независимо установи фактически описанную операцию оригинала.
scope_ref указывает только её назначение и подтверждённые условия.
Не добавляй симптом сотрудника или предполагаемую пользу,
если оригинал не связывает эту операцию с этим симптомом. Назначение настройки
конкретного параметра само по себе не подтверждает диагностику другого сбоя.
Программные настройки подключения устройства описывают подключение в указанной
программной среде. Условие прямого сетевого соединения нельзя добавлять самому.
После этого сравни исходную цель с выбранным source_scopes по основной операции.
"""

RELATED_TASK += """

target_relation отдельно сравнивает функциональный объект и конечный результат.
SAME_FUNCTIONAL_TARGET означает тот же сервис, объект работы и результат.
ALTERNATIVE_ROUTE означает другой готовый путь к тому же результату с совместимыми
физическими фактами. DIFFERENT_TARGET означает другой сервис, функцию или объект.
UNPROVEN означает, что связь не установлена. Общее слово входа не объединяет
почту, кадровый портал и портал поддержки. Настройка громкости не меняет регистр
букв, а обновление списка принтеров не обновляет их прошивку.
Уже известные бренд, количество и роль каждого устройства, распределение
соединений, приложение и отрицания сохраняются. ТСД и принтер — разные объекты.
Нельзя предложить изменить бренд имеющегося оборудования. Полный сценарий с
двумя разными принтерами не становится сценарием для одного принтера только
потому, что содержит один совпавший шаг. Альтернативный способ соединения того
же оборудования может предлагаться для последующего явного выбора сотрудником.
Для RELATED допустимы SAME_FUNCTIONAL_TARGET и ALTERNATIVE_ROUTE при
relation_kind=ALTERNATE_CONDITIONS. Для UNRELATED укажи DIFFERENT_TARGET или
UNPROVEN и пустые исходные ссылки. purpose_comparison сравнивает конкретные
сервисы, функции и результаты, сохраняя фактическую цель сотрудника.
"""


def _descriptive_scope(text):
    sentences = re.split(r"(?<=[.!?])\s+", _plain(text))
    kept = []
    for sentence in sentences:
        if (PROCEDURE_START.search(sentence) or LEAD_ACTION.search(sentence)
                or PARAMETER_VALUE.search(sentence) or re.search(r"\b[а-яё-]+(?:йте|ите)\b", sentence, re.I)):
            continue
        kept.append(sentence)
    return _short_original(" ".join(kept))


def related_source_scopes(payload):
    """Derive non-operational descriptions without introducing model-authored conditions."""
    registry = payload["candidate_evidence"]
    sections = [{**section, "id": identity,
                 "text": section.get("text") or " ".join(registry[ref]["text"]
                                                         for ref in section["evidence_ids"])}
                for identity, section in payload["candidate_sections"].items()]
    headings = [section for section in sections if _scope_heading(section)]
    first_action = next((index for index, section in enumerate(sections)
                         if PROCEDURE_START.search(_plain(section["text"]))
                         or LEAD_ACTION.search(_plain(section["text"]))), len(sections))
    purposes = []
    for index, section in enumerate(sections[:-1]):
        following = sections[index + 1]
        if SCOPE_LABEL.fullmatch(_plain(section["text"])) and not following.get("is_heading"):
            text = _descriptive_scope(following["text"])
            if text:
                purposes.append((section.get("group_id"), text, index))
    fallback = payload["source"]["filename"].rsplit(".", 1)[0]
    descriptions = []
    for heading in headings or [{"text": fallback, "group_id": None}]:
        title = _short_original(heading["text"])
        if not title:
            continue
        description = title
        for group, text, index in purposes:
            if group != heading.get("group_id") and index >= first_action:
                continue
            if text not in description and len(description) + len(text) + 2 <= 250:
                description += ". " + text
        if description not in descriptions:
            descriptions.append(description)
    return {"scope" + str(index): description for index, description in enumerate(descriptions, 1)}


def prepare_related(payload):
    """Keep the same complete original and alias registry as the exact reader."""
    wire, _, reverse = prepare_reading(payload)
    scopes = related_source_scopes(payload)
    wire["source_scopes"] = scopes
    properties = {
        "original_goal_id": {"type": ["string", "null"], "enum": [*payload["requested_goals"], None]},
        "scope_ref": {"type": ["string", "null"], "enum": [*scopes, None]},
        "evidence_ids": {"type": "array", "items": {"type": "string", "enum": list(reverse)},
                         "maxItems": 32, "uniqueItems": True},
        "purpose_comparison": {"type": "string", "minLength": 10, "maxLength": 300},
        "relation_kind": {"type": "string", "enum": ["ALTERNATE_CONDITIONS", "DIFFERENT_OPERATION", "UNPROVEN"]},
        "target_relation": {"type": "string", "enum": [
            "SAME_FUNCTIONAL_TARGET", "ALTERNATIVE_ROUTE", "DIFFERENT_TARGET", "UNPROVEN",
        ]},
        "status": {"type": "string", "enum": ["RELATED", "UNRELATED"]},
    }
    schema = {"title": "RelatedInstruction", "type": "object", "properties": properties,
              "required": list(properties), "additionalProperties": False}
    branches = []
    if scopes:
        related = {**properties,
                   "original_goal_id": {"type": "string", "enum": list(payload["requested_goals"])},
                   "scope_ref": {"type": "string", "enum": list(scopes)},
                   "evidence_ids": {**properties["evidence_ids"], "minItems": 1},
                   "relation_kind": {"const": "ALTERNATE_CONDITIONS"},
                   "target_relation": {"enum": ["SAME_FUNCTIONAL_TARGET", "ALTERNATIVE_ROUTE"]},
                   "status": {"const": "RELATED"}}
        branches.append({"type": "object", "properties": related,
                         "required": list(properties), "additionalProperties": False})
    unrelated = {**properties, "original_goal_id": {"type": "null"}, "scope_ref": {"type": "null"},
                 "evidence_ids": {**properties["evidence_ids"], "maxItems": 0},
                 "relation_kind": {"enum": ["DIFFERENT_OPERATION", "UNPROVEN"]},
                 "target_relation": {"enum": ["DIFFERENT_TARGET", "UNPROVEN"]},
                 "status": {"const": "UNRELATED"}}
    branches.append({"type": "object", "properties": unrelated,
                     "required": list(properties), "additionalProperties": False})
    schema["anyOf"] = branches
    return wire, schema, reverse


def resolve_related(proposal, payload, reverse):
    if (len(set(proposal.evidence_ids)) != len(proposal.evidence_ids)
            or any(reference not in reverse for reference in proposal.evidence_ids)):
        raise ValueError("UnknownRelatedInstructionEvidence")
    if (proposal.status == "UNRELATED" or proposal.relation_kind != "ALTERNATE_CONDITIONS"
            or proposal.target_relation in {"DIFFERENT_TARGET", "UNPROVEN"}):
        return unrelated_instruction()
    scopes = related_source_scopes(payload)
    if (proposal.original_goal_id not in payload["requested_goals"] or proposal.scope_ref not in scopes
            or not proposal.evidence_ids):
        raise ValueError("InvalidRelatedInstructionScope")
    return RelatedInstruction(original_goal_id=proposal.original_goal_id, source_goal=scopes[proposal.scope_ref],
                              evidence_ids=[reverse[reference] for reference in proposal.evidence_ids],
                              same_purpose=True, status="RELATED")


def unrelated_instruction():
    return RelatedInstruction(original_goal_id=None, source_goal="", evidence_ids=[], same_purpose=False,
                              status="UNRELATED")
