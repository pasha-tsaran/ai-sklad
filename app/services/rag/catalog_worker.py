"""Refresh the instruction catalogue independently of dialogue and delivery."""

from app.services.rag.catalog import KnowledgeCatalog
from app.services.rag.success import expire_retrievals
from app.services.rag.talk import TalkKnowledge


def synchronize_knowledge(factory, settings, *, refresh=False):
    expired = expire_retrievals(factory)
    if getattr(settings, "openwebui_sync_profile", False):
        from app.services.llm.profile import profile_settings

        settings = profile_settings(settings)
    if settings.knowledge_backend != "openwebui" or not settings.openwebui_knowledge_ids:
        return {"status": "disabled", "ready": 0, "complete": False, "expired_retrievals": expired}
    catalog = KnowledgeCatalog(settings, factory)
    knowledge = TalkKnowledge(settings, catalog=catalog)
    return {**catalog.sync(knowledge, refresh=refresh), "expired_retrievals": expired}
