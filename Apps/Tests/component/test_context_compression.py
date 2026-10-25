# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import pytest
from django.test import SimpleTestCase
from Apps.API.views import _resolve_history_char_budget


pytestmark = [pytest.mark.component]


# Cover context compression threshold math.
class ContextCompressionBudgetTests(SimpleTestCase):
    # Verify history budget uses same model token estimator as usage ui.
    def test_history_budget_uses_same_model_token_estimator_as_usage_ui(self):
        payload = {"context_length": 10000}

        self.assertEqual(
            _resolve_history_char_budget(payload, active_engine="lms", active_model="qwen3"),
            20000,
        )
        self.assertEqual(
            _resolve_history_char_budget(payload, active_engine="ollama-service", active_model="llama3"),
            27000,
        )

    # Verify history budget blends observed token ratio.
    def test_history_budget_blends_observed_token_ratio(self):
        payload = {"context_length": 10000}

        self.assertEqual(
            _resolve_history_char_budget(
                payload,
                active_engine="lms",
                active_model="qwen3",
                observed_chars_per_token=1.6,
            ),
            17400,
        )
