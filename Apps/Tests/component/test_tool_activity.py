# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import pytest
from django.test import SimpleTestCase
from Apps.API.views import _serialize_tool_activity_marker


pytestmark = [pytest.mark.component]


class ToolActivityStreamTests(SimpleTestCase):
    def test_activity_marker_keeps_structured_realtime_event(self):
        activity = {
            "alias": "deep_research__research",
            "server_id": "deep_research",
            "tool_id": "research",
            "event": {
                "sequence": 7,
                "phase": "research",
                "iteration": 2,
                "type": "model_output_delta",
                "data": {"channel": "reasoning", "content": "Checking sources"},
            },
        }

        marker = _serialize_tool_activity_marker(activity)

        self.assertTrue(marker.startswith("<tool_activity>"))
        self.assertTrue(marker.endswith("</tool_activity>"))
        payload = json.loads(marker.removeprefix("<tool_activity>").removesuffix("</tool_activity>"))
        self.assertEqual(payload, activity)
