# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from Apps.Data.models import (
    Chat,
    Message,
    MessageAttachment,
    MessageAttachmentKind,
    MessageRole,
    OllamaPreset,
)


# Create a chat with sensible defaults.
def make_chat(**overrides) -> Chat:
    params: dict = {"title": "Test chat"}
    params.update(overrides)
    return Chat.objects.create(**params)


# Create one message inside a chat.
def make_message(chat: Chat, *, role: str = MessageRole.USER, content: str = "Hello", **overrides) -> Message:
    params: dict = {"chat": chat, "role": role, "content": content}
    params.update(overrides)
    return Message.objects.create(**params)


# Create a user/assistant turn pair inside a chat.
def make_exchange(chat: Chat, *, user_text: str = "Question", assistant_text: str = "Answer") -> tuple[Message, Message]:
    user_message = make_message(chat, role=MessageRole.USER, content=user_text)
    assistant_message = make_message(chat, role=MessageRole.ASSISTANT, content=assistant_text)
    return user_message, assistant_message


# Create one attachment record on a message.
def make_attachment(
    message: Message,
    *,
    kind: str = MessageAttachmentKind.FILE,
    name: str = "note.txt",
    mime_type: str = "text/plain",
    data: str = "SGVsbG8=",
    **overrides,
) -> MessageAttachment:
    params: dict = {
        "message": message,
        "kind": kind,
        "name": name,
        "mime_type": mime_type,
        "data": data,
        "size_bytes": len(data),
    }
    params.update(overrides)
    return MessageAttachment.objects.create(**params)


# Create one Ollama preset row.
def make_ollama_preset(
    model_name: str = "llama3",
    *,
    name: str = "Default",
    config: dict | None = None,
    is_default: bool = False,
    is_active: bool = False,
    **overrides,
) -> OllamaPreset:
    params: dict = {
        "model_name": model_name,
        "name": name,
        "config": config if config is not None else {"num_ctx": 32768},
        "is_default": is_default,
        "is_active": is_active,
    }
    params.update(overrides)
    return OllamaPreset.objects.create(**params)
