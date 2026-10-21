from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event

from app.api.dependencies import get_session
from app.bootstrap import seed
from app.core.config import get_settings
from app.db import session_factory
from app.main import app
from app.models import Base
from app.services.llm.mock import MockLLM
from app.workers.handlers import run_one


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("API_TOKEN", "test-token")
    monkeypatch.setenv("LLM_BACKEND", "mock")
    get_settings.cache_clear()
    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}", connect_args={"check_same_thread": False})

    @event.listens_for(engine, "connect")
    def foreign_keys(connection, record):
        connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    factory = session_factory(engine)
    with factory() as session, session.begin():
        seed(session)

    def session_override():
        with factory() as session, session.begin():
            yield session

    app.dependency_overrides[get_session] = session_override
    model = MockLLM()
    counter = 0
    with TestClient(app, headers={"Authorization": "Bearer test-token"}) as client:

        def post(path, body, key=None):
            nonlocal counter
            counter += 1
            return client.post(path, json=body, headers={"Idempotency-Key": key or f"test-{counter}"})

        def drain(llm=None):
            for _ in range(20):
                if not run_one(factory, llm or model, get_settings()):
                    return
            raise AssertionError("Worker did not drain")

        yield SimpleNamespace(client=client, post=post, drain=drain, factory=factory, llm=model)
    app.dependency_overrides.clear()
    engine.dispose()
    get_settings.cache_clear()
