"""Classify old requested conditions after an employee chooses a verified alternative."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.services.llm.prompts import stage_messages


OFFER_ADOPTION_TASK = """Сотрудник выбрал показанный ему проверенный вариант инструкции.
accepted_goals содержит прежнюю цель и точную новую цель выбранного варианта.
actual_user_reply и consent_evidence подтверждены сервисом как выбор этого варианта.
Определи роль каждого candidate_facts по его исходному пользовательскому evidence.
REQUESTED — желаемое условие будущей работы, например способ подключения, который
сотрудник просил организовать. OBSERVED — уже существующее подключение, доступное
оборудование, ограничение среды или наблюдаемый факт. При сомнении выбирай OBSERVED.
Модальность определяется исходным evidence самого факта. «Нужно соединить терминал
по кабелю» описывает REQUESTED. «Терминал уже соединён по кабелю» описывает OBSERVED.
Последующий выбор другого варианта не превращает прежнее пожелание в наблюдение.
attribute=connection_type обозначает способ связи и не является типом оборудования.
superseded_requested_fact_ids содержит только REQUESTED, которые заменены условиями
выбранной новой цели. Условия других preserved_goals остаются в силе.
Факт об уже существующем оборудовании, соединении или ограничении среды сохраняется,
даже если он мешает применить выбранный вариант. Его нельзя скрывать согласием.
Новых фактов, значений, целей и технического ответа не создавай. Верни только JSON.
"""


class InstructionOfferAdoption(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fact_roles: dict[str, Literal["REQUESTED", "OBSERVED"]]
    superseded_requested_fact_ids: list[str] = Field(max_length=12)


def classify_offer_adoption(llm, payload):
    facts = payload.get("candidate_facts")
    if not isinstance(facts, dict) or not 1 <= len(facts) <= 12:
        raise ValueError("InvalidOfferAdoptionFacts")
    schema = InstructionOfferAdoption.model_json_schema()
    schema["properties"]["fact_roles"] = {
        "type": "object", "properties": {key: {"type": "string", "enum": ["REQUESTED", "OBSERVED"]} for key in facts},
        "required": list(facts), "additionalProperties": False,
    }
    schema["properties"]["superseded_requested_fact_ids"]["items"] = {"type": "string", "enum": list(facts)}
    schema["properties"]["superseded_requested_fact_ids"]["uniqueItems"] = True
    messages = stage_messages(InstructionOfferAdoption, OFFER_ADOPTION_TASK, payload, include_schema=False)
    result = InstructionOfferAdoption.model_validate_json(llm._request(
        messages, selection_schema=schema, max_output_tokens=512,
    ))
    selected = result.superseded_requested_fact_ids
    if (set(result.fact_roles) != set(facts) or len(set(selected)) != len(selected)
            or any(reference not in facts or result.fact_roles[reference] != "REQUESTED" for reference in selected)):
        raise ValueError("InvalidOfferAdoptionFacts")
    return result
