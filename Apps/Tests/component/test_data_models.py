# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import pytest
from django.test import TestCase
from Apps.Data.models import Chat, Message, MessageAttachment, MessageAttachmentKind, MessageImage, MessageRole


pytestmark = [pytest.mark.component]


# Cover the string helpers of the chat and message models, which admin and API serialization rely on.
class ChatMessageModelTests(TestCase):
    # Verify that the chat string is its title and the message string is its role plus the first 50 characters.
    def test_chat_and_message_string_representations_are_readable(self):
        chat = Chat.objects.create(title="Planning")
        message = Message.objects.create(
            chat=chat,
            role=MessageRole.USER,
            content="x" * 60,
        )

        self.assertEqual(str(chat), "Planning")
        self.assertEqual(str(message), f"user: {'x' * 50}")


# Cover the helper serialization of legacy image-only message records.
class MessageImageTests(TestCase):
    # Verify that the data URL of a stored image starts with its MIME type and the base64 marker.
    def test_data_url_builds_valid_prefix(self):
        chat = Chat.objects.create(title="Test")
        message = Message.objects.create(chat=chat, role=MessageRole.USER, content="Hello")
        image = MessageImage.objects.create(
            message=message,
            mime_type="image/png",
            data="abc123",
        )

        self.assertEqual(image.data_url(), "data:image/png;base64,abc123")


# Cover the data URL, image detection, ordering and string helpers of normalized message attachments.
class MessageAttachmentTests(TestCase):
    # Create the chat and message that own the attachment records.
    def setUp(self):
        super().setUp()
        self.chat = Chat.objects.create(title="Test")
        self.message = Message.objects.create(
            chat=self.chat,
            role=MessageRole.USER,
            content="Hello",
        )

    # Verify that the data URL uses the stored MIME type and that image detection follows the MIME type
    # even when the attachment kind is a file.
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

    # Verify that attachments are ordered by their order field, then by creation (id), and that the
    # string form names the file and the owning message.
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
