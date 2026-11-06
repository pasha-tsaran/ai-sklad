"""Readable, copyable letter preview without interpreting employee text as Markdown."""

import re

from app.services.mail.draft import _content


def letter_preview(email, recipient):
    recipient, subject, body = _content(email, recipient)
    text = f"Кому: {recipient}\nТема: {subject}\n\n{body}"
    longest = max((len(match.group()) for match in re.finditer(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return fence + "\n" + text + ("" if text.endswith("\n") else "\n") + fence
