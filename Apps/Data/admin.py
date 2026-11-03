# Copyright NEXTGGTECH. Elastic License 2.0.

from django.contrib import admin

from Apps.Data.models import Chat, ChatBranch, Message, MessageImage, OllamaPreset


# Register chat records in the admin.
@admin.register(Chat)
class ChatAdmin(admin.ModelAdmin):
    list_display = ("title", "active_tool_slug", "created_at", "updated_at")
    search_fields = ("title", "active_tool_slug")


# Register chat messages in the admin.
@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("chat", "role", "created_at")
    list_filter = ("role",)
    search_fields = ("content",)


# Register chat branch links (source chat/message to child chat/message) in the admin.
@admin.register(ChatBranch)
class ChatBranchAdmin(admin.ModelAdmin):
    list_display = ("source_chat", "source_message", "child_chat", "child_message", "created_at")
    search_fields = ("source_chat__title", "child_chat__title")


# Register legacy stored images in the admin.
@admin.register(MessageImage)
class MessageImageAdmin(admin.ModelAdmin):
    list_display = ("message", "mime_type", "order")
    list_filter = ("mime_type",)


# Register Ollama presets in the admin.
@admin.register(OllamaPreset)
class OllamaPresetAdmin(admin.ModelAdmin):
    list_display = ("model_name", "name", "is_default", "is_active", "updated_at")
    list_filter = ("is_default", "is_active")
    search_fields = ("model_name", "name")
