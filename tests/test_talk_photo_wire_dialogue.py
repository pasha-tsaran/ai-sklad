"""The serialized Talk REST photo travels from polling to a real durable reply."""

import pytest
from sqlalchemy import select

from app.core.config import Settings
from app.models import Delivery, Incident, Message, MessageAttachment, TalkInbox, TalkRoom
from app.services.chat.talk_dialogue import PHOTO_NOTICE, process_one
from app.services.nextcloud.poller import poll_once
from tests.test_talk_account import ROOM, Server, message
from tests.test_talk_dialogue import KB, QUESTION, Knowledge, NoGeneration


@pytest.mark.parametrize("caption", ["", QUESTION], ids=["photo-only", "technical-caption"])
def test_actual_rest_photo_reaches_a_durable_reply_and_preserves_original_guide_images(env, caption):
    server = Server()
    # The Talk serializer clears systemMessage after converting file_shared to
    # comment. The rich file parameter remains, and caption replaces {file}.
    server.history[ROOM] = [message(1, content=caption or "{file}", systemMessage="", messageParameters={
        "file": {"type": "file", "id": "456", "name": "employee-screenshot.png",
                 "mimetype": "image/png", "size": "2500", "path": "Talk/private.png",
                 "link": "https://never-follow.example.invalid/photo"},
    })]
    settings = Settings(_env_file=None, nextcloud_backend="account",
                        nextcloud_base_url="https://nextcloud.example/office", nextcloud_username="SkladTech-bot",
                        nextcloud_app_password="test-only-app-password", knowledge_backend="openwebui",
                        openwebui_api_key="test", openwebui_knowledge_ids=[KB], seed_demo_knowledge=False)
    assert poll_once(env.factory, settings, server.client())
    with env.factory() as session:
        inbox = session.scalar(select(TalkInbox))
        assert inbox.status == "PENDING" and inbox.content == caption and inbox.media == {"files": [456]}
        account_id = session.get(TalkRoom, inbox.room_id).account_id
    knowledge, model = Knowledge(), NoGeneration()
    assert process_one(env.factory, settings, model, knowledge, account_id=account_id)
    with env.factory() as session:
        inbox = session.scalar(select(TalkInbox))
        assert inbox.status == "DONE" and inbox.tries == 1
        incident = session.scalar(select(Incident))
        assert incident.has_user_photos
        replies = session.scalars(select(Message).where(Message.role == "assistant").order_by(Message.sequence)).all()
        images = session.scalars(select(MessageAttachment).join(Message).order_by(Message.sequence)).all()
        if caption:
            assert replies[0].content == PHOTO_NOTICE
            assert incident.state == "WAITING_FEEDBACK" and len(images) == 6
            for number, image in enumerate(images, 1):
                assert image.data == knowledge.documents[0].illustrations[number].data
                position = next(index for index, reply in enumerate(replies) if reply.id == image.message_id)
                assert image.caption in replies[position - 1].content
            assert len(knowledge.searches) == 1
        else:
            assert len(replies) == 1 and replies[0].content.startswith(PHOTO_NOTICE)
            assert "Опишите словами" in replies[0].content
            assert not images and not knowledge.searches and incident.pending_action is None
        assert session.scalar(select(Delivery)) is not None
        assert not model.calls
    # Ingress reads only metadata. It never fetches user images or posts to Talk.
    assert all(request.method == "GET" for request in server.requests)
    assert not any("/remote.php/" in request.url.path for request in server.requests)
