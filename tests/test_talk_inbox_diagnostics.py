"""Operator inspection exposes a failed inbox without transcript text or writes."""

import pytest

from app import deliveries
from app.models import TalkInbox
from tests.test_talk_dialogue import talk as talk


@pytest.mark.parametrize("code,expected", [
    ("IntegrityError", "IntegrityError"),
    ("InterruptedGeneration", "InterruptedGeneration"),
    ("private query\npassword=secret", "UnknownError"),
    (None, "UnknownError"),
])
def test_failed_inbox_listing_is_read_only_and_does_not_show_private_message(talk, monkeypatch, capsys, code, expected):
    failed = talk.incoming("PRIVATE EMPLOYEE PROBLEM", sequence=1)
    pending = talk.incoming("ANOTHER PRIVATE QUESTION", sequence=2)
    with talk.env.factory() as session, session.begin():
        row = session.get(TalkInbox, failed)
        row.status, row.error_code, row.tries = "FAILED", code, 3
    monkeypatch.setattr(deliveries, "session_factory", lambda: talk.env.factory)
    monkeypatch.setattr("sys.argv", ["deliveries", "talk"])
    deliveries.main()
    output = capsys.readouterr().out
    assert output.strip() == f"inbox {failed} FAILED {expected}"
    assert pending not in output and "PRIVATE" not in output and "password" not in output
    with talk.env.factory() as session:
        row = session.get(TalkInbox, failed)
        assert (row.status, row.tries, row.error_code) == ("FAILED", 3, code)
        assert session.get(TalkInbox, pending).status == "PENDING"


def test_mail_inspection_does_not_mix_failed_talk_inbox(talk, monkeypatch, capsys):
    failed = talk.incoming("PRIVATE")
    with talk.env.factory() as session, session.begin():
        session.get(TalkInbox, failed).status = "FAILED"
    monkeypatch.setattr(deliveries, "session_factory", lambda: talk.env.factory)
    monkeypatch.setattr("sys.argv", ["deliveries", "mail"])
    deliveries.main()
    assert capsys.readouterr().out == ""
