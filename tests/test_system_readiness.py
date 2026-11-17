"""A stale database must be visible before enabling the indexed dialogue system."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app import main
from app.core.config import Settings


@pytest.mark.parametrize("revision", [None, "0008", "0009"])
def test_readiness_requires_the_schema_used_by_the_instruction_index(monkeypatch, revision):
    connection = SimpleNamespace(execute=lambda statement: SimpleNamespace(scalar=lambda: revision))
    monkeypatch.setattr(main, "get_engine", lambda: SimpleNamespace(connect=lambda: nullcontext(connection)))
    with pytest.raises(HTTPException) as error:
        main.ready()
    assert error.value.status_code == 503
    assert error.value.detail == {"error_code": "DatabaseMigrationRequired", "expected_revision": "0010"}


def test_ready_response_identifies_the_actual_app_and_schema(monkeypatch):
    connection = SimpleNamespace(execute=lambda statement: SimpleNamespace(scalar=lambda: "0010"))
    monkeypatch.setattr(main, "get_engine", lambda: SimpleNamespace(connect=lambda: nullcontext(connection)))
    monkeypatch.setattr(main, "get_settings", lambda: Settings(_env_file=None, knowledge_backend="demo"))
    result = main.ready()
    assert result["status"] == "ok" and result["app_version"] == "0.4.0"
    assert result["database_revision"] == "0010"
