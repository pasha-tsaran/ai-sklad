import hashlib
import hmac
import secrets
from urllib.parse import quote

import httpx
from sqlalchemy import select

from app.models import Conversation, Delivery, Incident, Message
from app.services.chat.messages import append_message


class RoutedMessenger:
    def send(self, session, incident_id, content):
        incident = session.get(Incident, incident_id)
        conversation = session.get(Conversation, incident.conversation_id)
        talk = conversation.channel == "nextcloud_talk"
        message = append_message(
            session, incident_id, "assistant", content, "QUEUED" if talk else "DELIVERED_MOCK"
        )
        if talk:
            session.add(Delivery(message_id=message.id, room_token=conversation.room_token))
        return message


class TalkClient:
    def __init__(self, settings, transport=None):
        self.settings, self.transport = settings, transport

    def send(self, room, content, message_id):
        if room not in self.settings.nextcloud_allowed_rooms:
            raise ValueError("Room not allowed")
        nonce = secrets.token_hex(32)
        signature = hmac.new(
            self.settings.nextcloud_bot_secret.get_secret_value().encode(),
            (nonce + content).encode(),
            hashlib.sha256,
        ).hexdigest()
        with httpx.Client(
            timeout=self.settings.integration_timeout_seconds,
            transport=self.transport,
            follow_redirects=False,
        ) as client:
            response = client.post(
                self.settings.nextcloud_base_url.rstrip("/")
                + "/ocs/v2.php/apps/spreed/api/v1/bot/"
                + quote(room, safe="")
                + "/message",
                headers={
                    "OCS-APIRequest": "true",
                    "Accept": "application/json",
                    "X-Nextcloud-Talk-Bot-Random": nonce,
                    "X-Nextcloud-Talk-Bot-Signature": signature,
                },
                json={"message": content, "referenceId": hashlib.sha256(message_id.encode()).hexdigest()},
            )
            response.raise_for_status()
            data = response.json()["ocs"]
            if data["meta"]["statuscode"] not in (100, 200, 201):
                raise ValueError("Talk rejected message")
            return str(data["data"]["id"])


def deliver_one(factory, settings, client=None):
    if settings.nextcloud_backend != "talk":
        return False
    with factory() as session, session.begin():
        # An ambiguous/failed delivery blocks later messages in that room, not other rooms.
        rows = session.scalars(
            select(Delivery)
            .join(Message, Delivery.message_id == Message.id)
            .where(Delivery.status != "SENT")
            .order_by(Message.created_at, Message.incident_id, Message.sequence)
        ).all()
        blocked = set()
        delivery = None
        for row in rows:
            if row.room_token not in blocked and row.status == "PENDING":
                delivery = row
                break
            blocked.add(row.room_token)
        if delivery is None:
            return False
        delivery.status = "SENDING"
        delivery.tries += 1
        message = session.get(Message, delivery.message_id)
        delivery_id, room, content, message_id = delivery.id, delivery.room_token, message.content, message.id
    try:
        remote_id = (client or TalkClient(settings)).send(room, content, message_id)
        status, error = "SENT", None
    except Exception as exc:
        # A timeout can occur after the server accepted a message. Never blindly resend.
        status, error, remote_id = "UNKNOWN", type(exc).__name__, None
    with factory() as session, session.begin():
        delivery = session.get(Delivery, delivery_id)
        delivery.status, delivery.error_code, delivery.remote_id = status, error, remote_id
        session.get(Message, message_id).delivery_status = "DELIVERED_TALK" if status == "SENT" else status
    return True
