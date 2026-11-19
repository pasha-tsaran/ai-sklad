"""Follow-up references bind each original fragment to its original picture."""

import hashlib
import json

import httpx
import pytest

from app.core.config import Settings
from app.schemas.contracts import FollowupResult
from app.services.chat.support_followup import diagnostic_excerpt
from app.services.llm.openwebui import OpenWebUILLM
from tests.test_talk_dialogue import Knowledge, approved
from tests.test_verified_support_followup import active_flow


def model(response, requests):
    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        answer = response(body) if callable(response) else response
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(answer)},
                                                     "finish_reason": "stop"}]})

    return OpenWebUILLM(Settings(_env_file=None, openwebui_base_url="https://model.invalid",
                                 openwebui_api_key="test"), transport=httpx.MockTransport(respond))


def test_ui_reference_returns_the_bound_original_photo_without_model_image_indices():
    original = approved(count=2)
    original.document["content"] += "\n\n" + "\n\n".join(image.caption for image in original.illustrations.values())
    original.document["source"]["sha256"] = hashlib.sha256(original.document["content"].encode()).hexdigest()
    requests = []

    def choose(body):
        payload = json.loads(body["messages"][-1]["content"])
        reference = next(key for key, fragment in payload["approved_source"]["fragments"].items()
                         if "Бета" in fragment["text"] and fragment["has_original_photograph"])
        return {"result": "EXCERPT", "reference_ids": [reference]}

    answer = diagnostic_excerpt(Knowledge([original]), active_flow(original), "Где панель Бета?",
                                {"followup_mode": "ui_explanation"}, model(choose, requests), allow_repeat=True)
    photos = [photo for _, photo in answer["blocks"] if photo]
    assert len(photos) == 1 and photos[0].data == original.illustrations[2].data
    schema = requests[0]["response_format"]["json_schema"]["schema"]
    assert set(schema["properties"]) == {"result", "reference_ids"}
    assert schema["properties"]["reference_ids"]["items"]["enum"]
    assert requests[0]["params"]["think"] is False and requests[0]["max_tokens"] == 512


@pytest.mark.parametrize("choice", [
    {"result": "EXCERPT", "reference_ids": ["not-an-original"]},
    {"result": "EXCERPT", "reference_ids": ["f1", "f1"]},
    {"result": "EXCERPT", "reference_ids": []},
    {"result": "NO_KNOWLEDGE", "reference_ids": ["f1"]},
    {"result": "EXCERPT", "reference_ids": ["f1"], "illustration_indices": [0]},
])
def test_unbound_conflicting_or_extra_choices_are_rejected(choice):
    backend = model(choice, [])
    with pytest.raises(ValueError):
        backend.followup("Где кнопка?", {}, {"chunk_id": "current-original", "title": "Инструкция",
            "excerpts": ["Исходный фрагмент с кнопкой."], "illustrations": [{"caption": "Кнопка"}],
            "excerpt_image_indices": {"0": [0]}})


def test_legacy_followup_provider_payload_retains_its_existing_result_contract(monkeypatch):
    backend = model({}, [])
    calls = []

    def legacy(schema, task, payload, **options):
        calls.append((schema, payload, options))
        return FollowupResult(result="EXCERPT", source_chunk_ids=["legacy"], step_indices=[0], illustration_indices=[0])

    monkeypatch.setattr(backend, "_call", legacy)
    result = backend.followup("Где кнопка?", {}, {"chunk_id": "legacy", "excerpts": ["Кнопка"]})
    assert result.illustration_indices == [0] and calls[0][0] is FollowupResult
    assert calls[0][2] == {"fast_selection": True, "max_output_tokens": 512}
