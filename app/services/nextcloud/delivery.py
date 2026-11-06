import hashlib
import hmac
import secrets
import json
from urllib.parse import quote

import httpx
from sqlalchemy import select, update
from sqlalchemy.orm import aliased

from app.models import Conversation, Delivery, Incident, Message, MessageAttachment, TalkAccount, TalkRoom
from app.services.chat.messages import append_message


def split_text(content, limit=32_000):
    """Fit Talk messages without dropping characters or truncating instructions."""
    while len(content) > limit:
        boundary = content.rfind("\n", 0, limit)
        if boundary < limit // 2:
            boundary = max(content.rfind(" ", 0, limit), content.rfind("\t", 0, limit))
        end = boundary + 1 if boundary > 0 else limit
        yield content[:end]
        content = content[end:]
    if content:
        yield content


class RoutedMessenger:
    @staticmethod
    def _account_id(session, incident_id):
        # SMTP may confirm an older escalated incident after the employee starts
        # another issue. Both belong to the same immutable private conversation.
        target = session.get(Incident, incident_id)
        return session.scalar(select(TalkRoom.account_id).join(Incident, TalkRoom.incident_id == Incident.id).where(
            Incident.conversation_id == target.conversation_id,
        ))

    def send_blocks(self, session, incident_id, blocks, source=None):
        """Persist source text before its original photo, independently of caption support."""
        incident = session.get(Incident, incident_id)
        conversation = session.get(Conversation, incident.conversation_id)
        talk = conversation.channel == "nextcloud_talk"
        account_id = self._account_id(session, incident_id) if talk else None
        first = None
        for content, picture in blocks:
            for part in split_text(content):
                text_message = append_message(session, incident_id, "assistant", part,
                                              "QUEUED" if talk else "DELIVERED_MOCK")
                first = first or text_message
                if talk:
                    session.add(Delivery(message_id=text_message.id, room_token=conversation.room_token,
                                         provenance=dict(source) if source else None, account_id=account_id))
            if picture is not None:
                photo = append_message(session, incident_id, "assistant", "", "QUEUED" if talk else "DELIVERED_MOCK")
                first = first or photo
                session.add(MessageAttachment(
                    message_id=photo.id, filename=picture.filename, caption=picture.caption,
                    content_type=picture.content_type, sha256=hashlib.sha256(picture.data).hexdigest(), data=picture.data,
                ))
                if talk:
                    session.add(Delivery(message_id=photo.id, room_token=conversation.room_token,
                                         provenance=dict(source) if source else None, account_id=account_id))
        if first is None:
            raise ValueError("Empty support response")
        return first

    def send(self, session, incident_id, content, illustrations=()):
        incident = session.get(Incident, incident_id)
        conversation = session.get(Conversation, incident.conversation_id)
        talk = conversation.channel == "nextcloud_talk"
        account_id = self._account_id(session, incident_id) if talk else None
        message = None
        for part in split_text(content) if talk else [content]:
            current = append_message(
                session, incident_id, "assistant", part, "QUEUED" if talk else "DELIVERED_MOCK"
            )
            message = message or current
            if talk:
                session.add(Delivery(message_id=current.id, room_token=conversation.room_token, account_id=account_id))
        if message is None:
            raise ValueError("Empty support response")
        for number, picture in enumerate(illustrations, 1):
            photo = append_message(
                session, incident_id, "assistant", f"Иллюстрация {number}. {picture.caption}",
                "QUEUED" if talk else "DELIVERED_MOCK",
            )
            session.add(MessageAttachment(
                message_id=photo.id, filename=picture.filename, caption=picture.caption,
                content_type=picture.content_type, sha256=hashlib.sha256(picture.data).hexdigest(),
                data=picture.data,
            ))
            if talk:
                session.add(Delivery(message_id=photo.id, room_token=conversation.room_token, account_id=account_id))
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
            result = data.get("data")
            if isinstance(result, dict) and result.get("id") is not None:
                return str(result["id"])
            if response.status_code == 201 and (result is None or result == []):
                # Bot API acknowledges delivery without a chat ID (unlike the user Chat API).
                return "reference:" + hashlib.sha256(message_id.encode()).hexdigest()
            raise ValueError("Invalid Talk acknowledgement")

    def send_image(self, room, content, message_id, attachment):
        """WebDAV upload + authenticated room-only share; bot secret is not a Files credential."""
        if room not in self.settings.nextcloud_allowed_rooms:
            raise ValueError("Room not allowed")
        username = self.settings.nextcloud_files_username
        password = self.settings.nextcloud_files_app_password.get_secret_value()
        if not username or not password:
            raise ValueError("NextcloudFilesCredentialsMissing")
        # Only server-created immutable snapshots are uploaded. Paths contain no user filenames.
        base = self.settings.nextcloud_base_url.rstrip("/")
        folder = "AI-Support-" + hashlib.sha256(room.encode()).hexdigest()[:24]
        filename = hashlib.sha256(message_id.encode()).hexdigest() + ".png"
        dav = base + "/remote.php/dav/files/" + quote(username, safe="") + "/" + folder
        with httpx.Client(
            auth=(username, password), timeout=self.settings.integration_timeout_seconds,
            transport=self.transport, follow_redirects=False,
        ) as client:
            created = client.request("MKCOL", dav)
            if created.status_code != 405:  # Already existing collection is expected on later photos.
                created.raise_for_status()
            uploaded = client.put(dav + "/" + filename, content=attachment.data,
                                  headers={"Content-Type": attachment.content_type})
            uploaded.raise_for_status()
            response = client.post(
                base + "/ocs/v2.php/apps/files_sharing/api/v1/shares",
                headers={"OCS-APIRequest": "true", "Accept": "application/json"},
                data={
                    "shareType": "10", "shareWith": room, "path": "/" + folder + "/" + filename,
                    "referenceId": hashlib.sha256(message_id.encode()).hexdigest(),
                    "talkMetaData": json.dumps({"caption": content}, ensure_ascii=False),
                },
            )
            response.raise_for_status()
            body = response.json()["ocs"]
            if body["meta"]["statuscode"] not in (100, 200, 201):
                raise ValueError("Talk rejected illustration")
            return str(body["data"]["id"])


def deliver_one(factory, settings, client=None):
    if settings.nextcloud_backend not in {"talk", "account"}:
        return False
    current_account_id = None
    if settings.nextcloud_backend == "account":
        from app.services.nextcloud.account import AccountClient

        client = client or AccountClient(settings)
        service_id = client.identity()
        with factory() as session:
            current_account_id = session.scalar(select(TalkAccount.id).where(
                TalkAccount.server_url == settings.nextcloud_base_url.rstrip("/"),
                TalkAccount.username == service_id, TalkAccount.enabled.is_(True),
            ))
        if current_account_id is None:
            return False
    with factory() as session, session.begin():
        # An ambiguous/failed delivery blocks later messages in that room, not other rooms.
        query = (select(Delivery)
            .join(Message, Delivery.message_id == Message.id)
            .where(Delivery.status != "SENT")
            .order_by(Message.created_at, Message.incident_id, Message.sequence))
        if current_account_id:
            query = query.where(Delivery.account_id == current_account_id)
        else:
            query = query.where(Delivery.account_id.is_(None))
        rows = session.scalars(query).all()
        blocked = set()
        delivery = None
        for row in rows:
            if row.room_token not in blocked and row.status == "PENDING":
                delivery = row
                break
            blocked.add(row.room_token)
        if delivery is None:
            return False
        claimed = session.execute(update(Delivery).where(
            Delivery.id == delivery.id, Delivery.status == "PENDING",
        ).values(status="SENDING", tries=Delivery.tries + 1))
        if claimed.rowcount != 1:
            return False  # A concurrent runner claimed this same oldest room item.
        message = session.get(Message, delivery.message_id)
        attachment = session.scalar(select(MessageAttachment).where(MessageAttachment.message_id == message.id))
        if attachment:
            session.expunge(attachment)
        delivery_id, room, content, message_id = delivery.id, delivery.room_token, message.content, message.id
        expected_employee, expected_service_id = None, None
        provenance = dict(delivery.provenance) if delivery.provenance else None
        if settings.nextcloud_backend == "account":
            current_incident = aliased(Incident)
            conversation_id = session.get(Incident, message.incident_id).conversation_id
            binding = session.scalar(select(TalkRoom).join(TalkAccount)
                .join(current_incident, TalkRoom.incident_id == current_incident.id).where(
                TalkRoom.room_token == room,
                current_incident.conversation_id == conversation_id,
                TalkAccount.server_url == settings.nextcloud_base_url.rstrip("/"),
                TalkAccount.id == current_account_id,
                TalkRoom.enabled.is_(True), TalkAccount.enabled.is_(True),
            ))
            if binding is not None:
                expected_employee = binding.employee_id
                expected_service_id = session.get(TalkAccount, binding.account_id).username
    try:
        if settings.nextcloud_backend == "account":
            from app.services.nextcloud.account import AccountClient

            if expected_employee is None:
                raise ValueError("PrivateRoomBindingMissing")
            sender = client or AccountClient(settings)
            if sender.identity() != expected_service_id:
                raise ValueError("TalkServiceAccountChanged")
            if provenance:
                from app.services.rag.talk import TalkKnowledge

                if not TalkKnowledge(settings).is_current(provenance):
                    raise ValueError("ApprovedInstructionChanged")
            remote_id = (sender.send_image(room, content, message_id, attachment, expected_employee=expected_employee)
                         if attachment else sender.send(room, content, message_id, expected_employee=expected_employee))
        else:
            sender = client or TalkClient(settings)
            remote_id = (sender.send_image(room, content, message_id, attachment) if attachment
                         else sender.send(room, content, message_id))
        status, error = "SENT", None
    except Exception as exc:
        # A timeout can occur after the server accepted a message. Never blindly resend.
        status, error, remote_id = "UNKNOWN", type(exc).__name__, None
    with factory() as session, session.begin():
        delivery = session.get(Delivery, delivery_id)
        delivery.status, delivery.error_code, delivery.remote_id = status, error, remote_id
        session.get(Message, message_id).delivery_status = "DELIVERED_TALK" if status == "SENT" else status
    return True
