"""Refresh the instruction catalogue independently of dialogue and delivery."""

from app.services.rag.catalog import KnowledgeCatalog
from app.services.rag.talk import TalkKnowledge


def synchronize_knowledge(factory, settings, *, refresh=False):
    if settings.knowledge_backend != "openwebui" or not settings.openwebui_knowledge_ids:
        return {"status": "disabled", "ready": 0, "complete": False}
    catalog = KnowledgeCatalog(settings, factory)
    knowledge = TalkKnowledge(settings, catalog=catalog)
    return catalog.sync(knowledge, refresh=refresh)
