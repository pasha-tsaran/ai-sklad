import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from alembic import command
from alembic.config import Config
from sqlalchemy import select

from app.core.config import get_settings
from app.db import session_factory
from app.models import KnowledgeChunk, KnowledgeDocument, Solution, User
from app.services.rag.embeddings import DemoHashEmbeddings

ROOT = Path(__file__).resolve().parent.parent


def stable_id(name):
    return str(uuid5(NAMESPACE_URL, "ai-support/demo/v1/" + name))


def seed(session, with_knowledge=True):
    if not session.scalar(select(User).where(User.external_id == "mock-employee")):
        session.add(User(id=stable_id("user"), external_id="mock-employee", display_name="Демо сотрудник"))
    document_id = stable_id("printer-document")
    document = session.get(KnowledgeDocument, document_id)
    # Switching the demo off must also disable previously seeded knowledge.
    if document:
        document.approved = with_knowledge
        return
    if not with_knowledge:
        return
    data = json.loads((ROOT / "sample_knowledge/printer.json").read_text(encoding="utf-8"))
    session.add(KnowledgeDocument(id=document_id, title=data["title"], approved=True, is_demo=True))
    session.flush()
    for ordinal, item in enumerate(data["solutions"], 1):
        chunk_id = stable_id(f"printer-chunk-{ordinal}")
        session.add(
            KnowledgeChunk(
                id=chunk_id,
                document_id=document_id,
                content=item["title"] + "\n" + "\n".join(item["steps"]),
                search_text=data["search_text"],
                embedding=DemoHashEmbeddings().embed(data["search_text"]),
            )
        )
        session.flush()
        session.add(
            Solution(
                id=stable_id(f"printer-solution-{ordinal}"),
                chunk_id=chunk_id,
                title=item["title"],
                steps=item["steps"],
                problem_type=data["problem_type"],
                priority=ordinal,
            )
        )


def main():
    command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
    with session_factory()() as session, session.begin():
        seed(session, get_settings().seed_demo_knowledge)


if __name__ == "__main__":
    main()
