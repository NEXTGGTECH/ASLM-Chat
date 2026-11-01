# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import pytest
from django.test import TestCase
from Apps.Data.models import Chat, Message, MessageAttachment, MessageAttachmentKind, MessageImage, MessageRole


pytestmark = [pytest.mark.component]


# Chat and attachment model tests.
# Verify compact model helpers used by admin and API serialization.
class ChatMessageModelTests(TestCase):
    """Verify string helpers on persisted chat data."""

    # Test chat and message string values stay readable.
    def test_chat_and_message_string_representations_are_readable(self):
        chat = Chat.objects.create(title="Planning")
        message = Message.objects.create(
            chat=chat,
            role=MessageRole.USER,
            content="x" * 60,
        )

        self.assertEqual(str(chat), "Planning")
        self.assertEqual(str(message), f"user: {'x' * 50}")

# Attachment model tests.
# Verify helper behavior for legacy image records.
class MessageImageTests(TestCase):
    """Verify helper serialization on stored message images."""

    # Ensure data URLs include the expected prefix.
    def test_data_url_builds_valid_prefix(self):
        chat = Chat.objects.create(title="Test")
        message = Message.objects.create(chat=chat, role=MessageRole.USER, content="Hello")
        image = MessageImage.objects.create(
            message=message,
            mime_type="image/png",
            data="abc123",
        )

        self.assertEqual(image.data_url(), "data:image/png;base64,abc123")

# Attachment model tests.
# Verify normalized attachment helpers and ordering.
class MessageAttachmentTests(TestCase):
    """Verify helper behavior for normalized message attachments."""

    # Set up a message for attachment records.
    def setUp(self):
        super().setUp()
        self.chat = Chat.objects.create(title="Test")
        self.message = Message.objects.create(
            chat=self.chat,
            role=MessageRole.USER,
            content="Hello",
        )

    # Ensure data URLs and image detection use stored metadata.
    def test_data_url_and_image_detection_use_stored_metadata(self):
        image = MessageAttachment.objects.create(
            message=self.message,
            kind=MessageAttachmentKind.FILE,
            name="diagram.svg",
            mime_type="image/svg+xml",
            data="PHN2Zz4=",
            size_bytes=6,
        )
        file_attachment = MessageAttachment.objects.create(
            message=self.message,
            kind=MessageAttachmentKind.FILE,
            name="note.txt",
            mime_type="text/plain",
            data="SGVsbG8=",
            size_bytes=5,
        )

        self.assertEqual(image.data_url(), "data:image/svg+xml;base64,PHN2Zz4=")
        self.assertTrue(image.is_image)
        self.assertFalse(file_attachment.is_image)

    # Ensure attachment query ordering is stable for the UI.
    def test_attachment_ordering_uses_order_then_id(self):
        last = MessageAttachment.objects.create(
            message=self.message,
            name="last.txt",
            data="MQ==",
            order=2,
        )
        first = MessageAttachment.objects.create(
            message=self.message,
            name="first.txt",
            data="Mg==",
            order=1,
        )
        second = MessageAttachment.objects.create(
            message=self.message,
            name="second.txt",
            data="Mw==",
            order=1,
        )

        self.assertEqual(list(self.message.attachments.all()), [first, second, last])
        self.assertEqual(str(first), f"first.txt for message {self.message.id}")
