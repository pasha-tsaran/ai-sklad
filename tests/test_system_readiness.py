"""A stale database must be visible before enabling the indexed dialogue system."""

import tomllib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from app import main
from app.core.config import Settings


@pytest.mark.parametrize("revision", [None, "0008", "0009", "0010"])
def test_readiness_requires_the_schema_used_by_the_instruction_index(monkeypatch, revision):
    connection = SimpleNamespace(execute=lambda statement: SimpleNamespace(scalar=lambda: revision))
    monkeypatch.setattr(main, "get_engine", lambda: SimpleNamespace(connect=lambda: nullcontext(connection)))
    with pytest.raises(HTTPException) as error:
        main.ready()
    assert error.value.status_code == 503
    assert error.value.detail == {"error_code": "DatabaseMigrationRequired", "expected_revision": "0011"}


def test_ready_response_identifies_the_actual_app_and_schema(monkeypatch):
    connection = SimpleNamespace(execute=lambda statement: SimpleNamespace(scalar=lambda: "0011"))
    monkeypatch.setattr(main, "get_engine", lambda: SimpleNamespace(connect=lambda: nullcontext(connection)))
    monkeypatch.setattr(main, "get_settings", lambda: Settings(_env_file=None, knowledge_backend="demo"))
    result = main.ready()
    with (Path(__file__).resolve().parents[1] / "pyproject.toml").open("rb") as project:
        version = tomllib.load(project)["project"]["version"]
    assert result["status"] == "ok" and result["app_version"] == version
    assert result["database_revision"] == "0011"
