# Copyright NEXTGGTECH. Elastic License 2.0.

from django.apps import AppConfig


# Django application container for the backend test suite.
class TestsConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "Apps.Tests"
    label = "aslm_tests"
