"""Prepare reviewable local-mail drafts; this module never opens a connection."""

import re
import unicodedata
from email import policy
from email.message import EmailMessage
from urllib.parse import quote, urlencode

MAX_MAILTO_CHARS = 1900
_LOCAL = re.compile(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+")
_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?")


def validate_recipient(value: str) -> str:
    """Accept one basic configured mailbox, never a display-name or address list."""
    if not isinstance(value, str) or not value.isascii() or len(value) > 254 or value.count("@") != 1:
        raise ValueError("InvalidDraftRecipient")
    local, domain = value.split("@")
    labels = domain.split(".")
    if (not local or len(local) > 64 or _LOCAL.fullmatch(local) is None or local.startswith(".")
            or local.endswith(".") or ".." in local or len(labels) < 2
            or any(_LABEL.fullmatch(label) is None for label in labels)):
        raise ValueError("InvalidDraftRecipient")
    return value


def _content(email, recipient):
    recipient = validate_recipient(recipient)
    if not isinstance(email, dict):
        raise ValueError("InvalidDraftContent")
    subject, body = email.get("subject"), email.get("body")
    if not isinstance(subject, str) or not subject.strip() or not isinstance(body, str):
        raise ValueError("InvalidDraftContent")
    if (any(unicodedata.category(char) == "Cc" for char in subject)
            or any(unicodedata.category(char) == "Cc" and char not in "\r\n\t" for char in body)):
        raise ValueError("InvalidDraftContent")
    # All line breaks stay in the text body, never in generated mail headers.
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    return recipient, subject, body


def make_mailto(email: dict, recipient: str) -> str | None:
    """Return a complete UTF-8 mailto URI, or None when an EML file is required.

    Only the configured recipient argument is used. A caller may supply a compact
    body here and keep the full transcript for make_eml; nothing is truncated.
    RFC 6068 body line breaks are encoded as CRLF. URI length is conservatively
    bounded for mail clients that cannot open long links from Talk.
    """
    recipient, subject, body = _content(email, recipient)
    query = urlencode({"subject": subject, "body": body.replace("\n", "\r\n")}, quote_via=quote, safe="")
    uri = "mailto:" + quote(recipient, safe="@") + "?" + query
    return uri if len(uri) <= MAX_MAILTO_CHARS else None


def make_eml(email: dict, recipient: str) -> bytes:
    """Return an unsent MIME draft with the whole body and no chosen sender."""
    recipient, subject, body = _content(email, recipient)
    message = EmailMessage(policy=policy.SMTP)
    message["X-Unsent"] = "1"
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    return message.as_bytes()
