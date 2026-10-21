import time

from sqlalchemy import select, text, update

from app.core.config import get_settings
from app.core.logging import configure_logging, event
from app.db import get_engine, session_factory
from app.models import Job, Delivery, Escalation, Message
from app.services.nextcloud.delivery import deliver_one
from app.services.llm.mock import MockLLM
from app.services.llm.openwebui import OpenWebUILLM
from app.workers.handlers import run_one


def main():
    configure_logging()
    settings = get_settings()
    engine = get_engine()
    factory = session_factory(engine)
    llm = MockLLM() if settings.llm_backend == "mock" else OpenWebUILLM(settings)
    # A session-level advisory lock ensures exactly one active worker in this MVP.
    # The autocommit connection holds no transaction during network calls.
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as guard:
        acquired = guard.scalar(text("SELECT pg_try_advisory_lock(73412061)"))
        if not acquired:
            raise RuntimeError("Only one worker is supported in this MVP")
        with factory() as session, session.begin():
            session.execute(update(Job).where(Job.status == "RUNNING").values(status="PENDING"))
            session.execute(update(Delivery).where(Delivery.status == "SENDING").values(status="UNKNOWN"))
            session.execute(update(Escalation).where(Escalation.status == "SENDING").values(status="UNKNOWN"))
            session.execute(
                update(Message)
                .where(Message.id.in_(select(Delivery.message_id).where(Delivery.status == "UNKNOWN")))
                .values(delivery_status="UNKNOWN")
            )
        event(
            "worker.started", llm=llm.name, mail=settings.mail_backend, nextcloud=settings.nextcloud_backend
        )
        while True:
            # Stop on connection loss instead of silently continuing without the lock.
            guard.execute(text("SELECT 1"))
            run_one(factory, llm, settings)
            deliver_one(factory, settings)
            time.sleep(1)


if __name__ == "__main__":
    main()
