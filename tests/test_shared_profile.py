import asyncio
import copy
import json
import sys
from types import ModuleType, SimpleNamespace

import httpx
import pytest

from app.core.config import Settings
from app.services.llm.openwebui import OpenWebUILLM
from app.services.llm.profile import ProfileUnavailable, apply_profile, profile_settings
from tests.test_openwebui_automatic_images import automatic as automatic
from tests.test_openwebui_instruction_images import image_tool as image_tool
from tests.test_talk_delivery import SOURCE, Sender, bind, enqueue, factory as factory, state

KB = '85cda3aa-b854-4c26-9948-12da742277ef'


def settings():
    return Settings(_env_file=None, llm_backend='openwebui', openwebui_api_key='test-key',
                    openwebui_sync_profile=True)


def profile():
    return dict(id='ai-support', base_model_id='qwen3.6:35b', is_active=True,
                params={'system': 'Общие правила тестового помощника', 'temperature': 0.2,
                        'num_ctx': 32768},
                meta={'knowledge': [{'id': KB, 'type': 'collection'}]})


def test_snapshot_changes_prompt_model_and_scope_without_mutating_previous_turn():
    config, data = settings(), profile()
    first = apply_profile(config, data)
    data['params']['system'] = 'Новые правила'
    data['base_model_id'] = 'other-model'
    data['meta']['knowledge'] = []
    second = apply_profile(config, data)
    assert config.openwebui_model == 'ai-support'
    assert first.openwebui_system_prompt == 'Общие правила тестового помощника'
    assert first.openwebui_knowledge_ids == [KB]
    assert second.openwebui_model == 'other-model'
    assert second.openwebui_knowledge_ids == []
    assert first.openwebui_profile_revision != second.openwebui_profile_revision


def test_profile_context_overrides_local_fallback_only_when_set():
    config, data = settings(), profile()
    config.openwebui_num_ctx = 16384
    assert apply_profile(config, data).openwebui_num_ctx == 32768
    del data['params']['num_ctx']
    assert apply_profile(config, data).openwebui_num_ctx == 16384
    assert apply_profile(settings(), data).openwebui_num_ctx is None


@pytest.mark.parametrize('change', [
    {'is_active': False}, {'id': 'foreign'}, {'base_model_id': 'ai-support'},
    {'params': {}}, {'params': {'system': 'ok', 'temperature': float('nan')}},
    {'meta': {'knowledge': [{'type': 'file', 'id': KB}]}},
    {'meta': {'knowledge': [{'type': 'collection', 'id': 'invalid'}]}},
    {'meta': {'knowledge': {}}},
])
def test_invalid_or_inaccessible_settings_fail_closed(change):
    data = profile()
    data.update(change)
    with pytest.raises(ProfileUnavailable):
        apply_profile(settings(), data)


def test_api_reads_only_profile_then_calls_base_model_with_own_context():
    requests = []

    def handler(request):
        requests.append(request)
        if request.method == 'GET':
            assert request.url.path == '/api/v1/models/model'
            assert request.url.params['id'] == 'ai-support'
            return httpx.Response(200, json=profile())
        body = json.loads(request.content)
        assert request.url.path == '/api/chat/completions'
        assert body['model'] == 'qwen3.6:35b'
        assert body['tools'] == [] and body['tool_ids'] == []
        assert 'chat_id' not in body and 'session_id' not in body
        assert 'Общие правила тестового помощника' in body['messages'][0]['content']
        assert 'Nextcloud Talk' in body['messages'][0]['content']
        assert body['temperature'] == 0.2
        assert body['params']['num_ctx'] == 32768
        return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {
            'content': '{"action":"reply","blocks":[{"text":"Здравствуйте"}]}'}}]})

    transport = httpx.MockTransport(handler)
    snapshot = profile_settings(settings(), transport=transport)
    reply = OpenWebUILLM(snapshot, transport=transport).agent_action({'message': 'Привет'})
    assert reply.blocks[0].text == 'Здравствуйте'
    assert len(requests) == 2


def test_no_stale_fallback_when_profile_permission_revoked():
    responses = iter([httpx.Response(200, json=profile()), httpx.Response(403)])
    transport = httpx.MockTransport(lambda request: next(responses))
    snapshot = profile_settings(settings(), transport=transport)
    with pytest.raises(httpx.HTTPStatusError):
        profile_settings(snapshot, transport=transport)


def test_native_filter_leaves_dialogue_and_photos_to_model(automatic):
    instance = automatic.Filter()
    instance.valves.dialogue_mode = 'agent'
    body = {'model': 'ai-support', 'messages': [{'role': 'user', 'content': 'продолжить'}],
            'tools': [{'name': 'show_instruction_images'}]}
    original = copy.deepcopy(body)
    assert asyncio.run(instance.inlet(body)) == original
    assert asyncio.run(instance.outlet(body)) == original


def test_ui_tool_uses_saved_profile_not_chat_scope(image_tool, monkeypatch):
    module = ModuleType('open_webui.models.models')
    data = profile()
    calls = []

    async def get_model(model_id):
        calls.append(model_id)
        return SimpleNamespace(is_active=data['is_active'], meta=SimpleNamespace(**data['meta']))

    module.Models = SimpleNamespace(get_model_by_id=get_model)
    monkeypatch.setitem(sys.modules, 'open_webui.models.models', module)
    tool = image_tool.Tools()
    tool.valves.sync_model_profile = True
    request = SimpleNamespace(knowledge_ids=['untrusted'])
    assert asyncio.run(tool._resolved_ids({'id': 'user'}, request)) == [KB]
    data['meta']['knowledge'] = []
    with pytest.raises(image_tool.ToolError):
        asyncio.run(tool._resolved_ids({'id': 'user'}, request))
    assert calls == ['ai-support', 'ai-support']


def test_delivery_rechecks_current_profile_before_any_external_send(factory, monkeypatch):
    from app.services.nextcloud.delivery import deliver_one
    from tests.test_talk_delivery import settings as delivery_settings

    bound = bind(factory)
    enqueue(factory, bound, [('Технический ответ из очереди', None)], SOURCE)
    configured = delivery_settings()
    snapshot = copy.copy(configured)
    snapshot.openwebui_knowledge_ids = []
    calls = []

    def current_profile(value):
        assert value is configured
        calls.append('profile')
        return snapshot

    def knowledge(value):
        assert value is snapshot
        calls.append('knowledge')
        return SimpleNamespace(is_current=lambda source: False)

    monkeypatch.setattr('app.services.llm.profile.profile_settings', current_profile)
    monkeypatch.setattr('app.services.rag.talk.TalkKnowledge', knowledge)
    sender = Sender()
    assert deliver_one(factory, configured, sender)
    assert calls == ['profile', 'knowledge']
    assert not sender.writes
    assert state(factory)[0][0].status != 'SENT'


def test_catalog_refresh_uses_profile_scope(monkeypatch):
    from app.services.rag.catalog_worker import synchronize_knowledge

    config = settings()
    snapshot = apply_profile(config, profile())
    calls = []

    class Catalog:
        def __init__(self, value, factory):
            assert value is snapshot

        def sync(self, knowledge, *, refresh):
            return {'status': 'ok', 'complete': True}

    def knowledge(value, *, catalog):
        calls.append(value.openwebui_knowledge_ids)
        return object()

    monkeypatch.setattr('app.services.llm.profile.profile_settings', lambda value: snapshot)
    monkeypatch.setattr('app.services.rag.catalog_worker.KnowledgeCatalog', Catalog)
    monkeypatch.setattr('app.services.rag.catalog_worker.TalkKnowledge', knowledge)
    assert synchronize_knowledge(None, config)['complete']
    assert calls == [[KB]]
