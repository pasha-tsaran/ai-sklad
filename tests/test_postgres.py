"""Opt-in integration checks against the migrated Compose PostgreSQL database."""

import os

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import select, text

from app.db import get_engine, session_factory
from app.models import KnowledgeChunk
from app.services.rag.embeddings import DemoHashEmbeddings
from app.services.rag.retrieval import LocalRetriever

pytestmark = pytest.mark.skipif(os.environ.get("TEST_POSTGRES") != "1", reason="Requires Compose PostgreSQL")


def test_migrations_match_models():
    command.check(Config("alembic.ini"))


def test_pgvector_and_approved_retrieval():
    with session_factory()() as session:
        assert session.scalar(text("SELECT extname FROM pg_extension WHERE extname = 'vector'")) == "vector"
        vector = DemoHashEmbeddings().embed("Принтер не печатает")
        nearest = session.scalar(
            select(KnowledgeChunk).order_by(KnowledgeChunk.embedding.cosine_distance(vector))
        )
        assert nearest is not None
        results = LocalRetriever(session).search("Принтер не печатает", set())
        assert len(results) == 5
        assert not LocalRetriever(session).search("космический телескоп", set())


def test_worker_singleton_lock():
    # Separate lock key so the check never interferes with the running worker.
    engine = get_engine()
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as first:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as second:
            assert first.scalar(text("SELECT pg_try_advisory_lock(73412062)"))
            try:
                assert not second.scalar(text("SELECT pg_try_advisory_lock(73412062)"))
            finally:
                first.execute(text("SELECT pg_advisory_unlock(73412062)"))
