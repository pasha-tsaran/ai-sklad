"""Operator-only local CLI. Inspect mail/Talk ambiguities before an explicit retry."""

import argparse

from sqlalchemy import select

from app.db import session_factory
from app.models import Delivery, Escalation, Job, Message, now
from app.services.chat.audit import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["mail", "talk"])
    parser.add_argument("--id")
    parser.add_argument("--action", choices=["retry", "mark-sent"])
    args = parser.parse_args()
    if bool(args.id) != bool(args.action):
        parser.error("--id and --action are required together")
    model = Escalation if args.kind == "mail" else Delivery
    with session_factory()() as session, session.begin():
        if not args.id:
            for row in session.scalars(select(model).where(model.status.in_(["UNKNOWN", "FAILED"]))):
                print(row.id, row.status)
            return
        row = session.scalar(select(model).where(model.id == args.id).with_for_update())
        if not row or row.status not in {"UNKNOWN", "FAILED"}:
            raise ValueError("Only UNKNOWN/FAILED deliveries can be resolved")
        if args.kind == "mail":
            incident_id = row.incident_id
            if args.action == "retry":
                row.status = "QUEUED"
                session.add(Job(incident_id=incident_id, kind="SEND_MAIL", expected_version=0))
            else:
                row.status, row.sent_at = "SENT", now()
        else:
            message = session.get(Message, row.message_id)
            incident_id = message.incident_id
            row.status = "PENDING" if args.action == "retry" else "SENT"
            message.delivery_status = "QUEUED" if args.action == "retry" else "DELIVERED_TALK"
        audit(
            session,
            incident_id,
            "delivery.operator_resolution",
            kind=args.kind,
            delivery_id=row.id,
            action=args.action,
        )
        print(row.id, row.status)


if __name__ == "__main__":
    main()
