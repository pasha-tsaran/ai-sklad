"""Index refresh uses the shared store and stays separate from chat delivery."""

from types import SimpleNamespace

from app.services.rag.catalog_worker import synchronize_knowledge
from scripts import sync_knowledge


def test_unconfigured_index_refresh_does_not_open_a_database_or_network():
    settings = SimpleNamespace(knowledge_backend="openwebui", openwebui_knowledge_ids=[])
    assert synchronize_knowledge(None, settings) == {"status": "disabled", "ready": 0, "complete": False}


def test_refresh_uses_the_injected_database_and_current_scope(monkeypatch):
    settings = SimpleNamespace(knowledge_backend="openwebui", openwebui_knowledge_ids=["scope"])
    factory = object()
    calls = []

    class Catalog:
        def __init__(self, configured, store):
            assert configured is settings and store is factory

        def sync(self, knowledge, *, refresh):
            calls.append((knowledge, refresh))
            return {"status": "ok", "complete": True, "ready": 50}

    monkeypatch.setattr("app.services.rag.catalog_worker.KnowledgeCatalog", Catalog)
    monkeypatch.setattr("app.services.rag.catalog_worker.TalkKnowledge", lambda configured, *, catalog: catalog)
    assert synchronize_knowledge(factory, settings, refresh=True)["ready"] == 50
    assert len(calls) == 1 and isinstance(calls[0][0], Catalog) and calls[0][1] is True


def test_index_command_returns_failure_for_an_incomplete_catalogue(monkeypatch, capsys):
    monkeypatch.setattr(sync_knowledge, "get_settings", lambda: object())
    monkeypatch.setattr(sync_knowledge, "session_factory", lambda: object())
    monkeypatch.setattr(sync_knowledge, "synchronize_knowledge", lambda *args, **kwargs: {
        "status": "partial", "complete": False, "ready": 49,
    })
    assert sync_knowledge.main([]) == 1
    assert '"complete": false' in capsys.readouterr().out


def test_index_command_keeps_exception_bodies_out_of_the_report(monkeypatch, capsys):
    monkeypatch.setattr(sync_knowledge, "get_settings", lambda: object())
    monkeypatch.setattr(sync_knowledge, "session_factory", lambda: object())

    def unavailable(*args, **kwargs):
        raise RuntimeError("private source text and API key")

    monkeypatch.setattr(sync_knowledge, "synchronize_knowledge", unavailable)
    assert sync_knowledge.main([]) == 1
    output = capsys.readouterr().out
    assert "RuntimeError" in output and "private" not in output and "API key" not in output
