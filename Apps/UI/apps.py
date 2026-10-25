# Copyright NEXTGGTECH. Elastic License 2.0.

from django.apps import AppConfig


# Configure the UI application.
class UiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "Apps.UI"
