"""Bind follow-up choices to original fragments and their backend-owned pictures."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.contracts import FollowupResult
from app.services.llm.prompts import stage_messages


FOLLOWUP_REFERENCE_TASK = """Выбери исходные фрагменты для продолжения текущей инструкции.
Верни только JSON по схеме, используя reference_ids из approved_source.fragments.
Текст, команды, подписи и новые изображения не генерируй. Каждый reference_id уже
связан backend с точным фрагментом и его исходными фотографиями.
context.followup_mode=diagnostic означает новые подтверждённые действия для
сообщённой ошибки. Уже выполненные действия исключены из доступных фрагментов.
В ui_explanation найди описание нужной кнопки, окна или разъёма. Если подходящий
фрагмент связан с фотографией, выбирай его для наглядного пояснения.
Нужны все исходные условия выбранного действия. Не соединяй разные способы.
Если доступные фрагменты не отвечают на уточнение, верни NO_KNOWLEDGE и пустой
reference_ids. Отсутствие ответа в текущем оригинале проверит backend.
"""


class FollowupReferenceSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    result: Literal["EXCERPT", "NO_KNOWLEDGE"]
    reference_ids: list[str] = Field(default_factory=list, max_length=12)


def select_followup_references(llm, question, context, approved_source):
    excerpts = approved_source.get("excerpts", [])
    illustrations = approved_source.get("illustrations", [])
    bindings = approved_source.get("excerpt_image_indices")
    if not isinstance(excerpts, list) or not isinstance(illustrations, list) or not isinstance(bindings, dict):
        raise ValueError("InvalidFollowupReferences")
    references = {}
    for index, text in enumerate(excerpts):
        pictures = bindings.get(str(index), [])
        if (not isinstance(text, str) or not text.strip() or not isinstance(pictures, list)
                or any(type(number) is not int or not 0 <= number < len(illustrations) for number in pictures)
                or len(set(pictures)) != len(pictures)):
            raise ValueError("InvalidFollowupReferences")
        references[f"f{index + 1}"] = (index, pictures)
    if not references:
        return FollowupResult(result="NO_KNOWLEDGE")
    schema = FollowupReferenceSelection.model_json_schema()
    schema["properties"]["reference_ids"]["items"] = {"type": "string", "enum": list(references)}
    schema["properties"]["reference_ids"]["uniqueItems"] = True
    schema["required"] = ["result", "reference_ids"]
    payload = {"question": question, "context": context, "approved_source": {
        "title": approved_source.get("title", ""),
        "fragments": {reference: {"text": excerpts[index], "has_original_photograph": bool(pictures)}
                      for reference, (index, pictures) in references.items()},
    }}
    messages = stage_messages(FollowupReferenceSelection, FOLLOWUP_REFERENCE_TASK, payload, include_schema=False)
    choice = FollowupReferenceSelection.model_validate_json(llm._request(
        messages, selection_schema=schema, max_output_tokens=512,
    ))
    if (len(set(choice.reference_ids)) != len(choice.reference_ids)
            or any(reference not in references for reference in choice.reference_ids)
            or bool(choice.reference_ids) != (choice.result == "EXCERPT")):
        raise ValueError("InvalidFollowupReferences")
    return FollowupResult(
        result=choice.result,
        source_chunk_ids=[approved_source["chunk_id"]] if choice.reference_ids else [],
        step_indices=[references[reference][0] for reference in choice.reference_ids],
        # The controller derives photographs from the selected original sections.
        # Numeric image choices remain available only to legacy providers.
        illustration_indices=[],
    )
