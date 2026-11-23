"""Read shared configuration, never Open WebUI conversations or user memory."""

import hashlib
import json
from uuid import UUID

import httpx


class ProfileUnavailable(ValueError):
    pass


def profile_settings(settings, *, transport=None):
    """One immutable snapshot per turn. No stale fallback after access/config changes."""
    if not getattr(settings, 'openwebui_sync_profile', False):
        return settings
    with httpx.Client(timeout=settings.integration_timeout_seconds, transport=transport,
                      follow_redirects=False) as client:
        response = client.get(
            settings.openwebui_base_url.rstrip('/') + '/api/v1/models/model',
            params={'id': settings.openwebui_profile_id},
            headers={'Authorization': 'Bearer ' + settings.openwebui_api_key.get_secret_value()},
        )
        response.raise_for_status()
        return apply_profile(settings, response.json())


def apply_profile(settings, profile):
    if (not isinstance(profile, dict) or profile.get('id') != settings.openwebui_profile_id
            or profile.get('is_active') is not True):
        raise ProfileUnavailable('InvalidOrDisabledProfile')
    params, meta = profile.get('params'), profile.get('meta')
    base = profile.get('base_model_id')
    if (not isinstance(params, dict) or not isinstance(meta, dict)
            or not isinstance(base, str) or not base.strip() or len(base) > 256
            or base == settings.openwebui_profile_id):
        raise ProfileUnavailable('InvalidProfileConfiguration')
    prompt = params.get('system')
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 60000:
        raise ProfileUnavailable('ProfileSystemPromptUnavailable')
    items = meta.get('knowledge')
    if items is None:
        items = []
    if not isinstance(items, list) or len(items) > 20:
        raise ProfileUnavailable('InvalidProfileKnowledge')
    ids = []
    for item in items:
        # Individual attachments are not approved collections. Never silently widen scope.
        if not isinstance(item, dict) or item.get('type') != 'collection':
            raise ProfileUnavailable('ProfileRequiresKnowledgeCollections')
        try:
            ids.append(str(UUID(item['id'])))
        except (KeyError, ValueError, TypeError, AttributeError):
            raise ProfileUnavailable('InvalidProfileKnowledge') from None
    generation = {}
    for key, low, high in [('temperature', 0, 2), ('top_p', 0, 1),
                           ('top_k', 0, 1000), ('repeat_penalty', 0, 2),
                           ('seed', -1, 2147483647), ('num_ctx', 2048, 131072)]:
        value = params.get(key)
        if value is None or value == '':
            continue
        if type(value) not in (int, float) or not low <= value <= high:
            raise ProfileUnavailable('InvalidProfileGenerationParameters')
        if key in {'top_k', 'seed', 'num_ctx'} and type(value) is not int:
            raise ProfileUnavailable('InvalidProfileGenerationParameters')
        generation[key] = value
    snapshot = dict(base=base, prompt=prompt, ids=sorted(set(ids)), generation=generation)
    digest = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
    return settings.model_copy(update={
        'openwebui_model': base, 'openwebui_knowledge_ids': list(dict.fromkeys(ids)),
        'openwebui_system_prompt': prompt, 'openwebui_generation_params': generation,
        'openwebui_profile_revision': digest,
        'openwebui_num_ctx': generation.get('num_ctx', settings.openwebui_num_ctx),
    })
