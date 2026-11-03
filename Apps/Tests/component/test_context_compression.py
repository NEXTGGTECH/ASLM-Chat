# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import pytest
from django.test import SimpleTestCase
from Apps.API.views import _resolve_history_char_budget


pytestmark = [pytest.mark.component]


# Cover the character budget that bounds chat history before context compression.
class ContextCompressionBudgetTests(SimpleTestCase):
    # Verify that the history budget is derived from the context length with the per-engine
    # characters-per-token estimate, the same one the usage UI relies on (20000 for LM Studio and
    # 27000 for Ollama at 10000 tokens).
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

    # Verify that an observed characters-per-token ratio is blended into the estimate, giving a
    # 17400-character budget for 10000 tokens at an observed ratio of 1.6.
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
