# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch
from django.test import Client, SimpleTestCase
from django.urls import reverse
from Apps.UI.views import _is_active_browser_portal_state


pytestmark = [pytest.mark.component]


# Verify wait-for-user portal timing and finish signaling.
class BrowserPortalApiTests(SimpleTestCase):
    # Verify active browser portal state uses deadline when available.
    def test_active_browser_portal_state_uses_deadline_when_available(self):
        with patch("Apps.UI.views.time.time", return_value=1000.0):
            self.assertFalse(
                _is_active_browser_portal_state(
                    {
                        "status": "waiting",
                        "updated_at": 999.0,
                        "timeout_seconds": 45,
                        "deadline_at": 980.0,
                    }
                )
            )
            self.assertTrue(
                _is_active_browser_portal_state(
                    {
                        "status": "waiting",
                        "updated_at": 500.0,
                        "timeout_seconds": 45,
                        "deadline_at": 1005.0,
                    }
                )
            )

    # Verify finish event response reports done and queues event.
    def test_finish_event_response_reports_done_and_queues_event(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "browser_portal"
            events_dir = root / "events"
            events_dir.mkdir(parents=True)
            (root / "state.json").write_text(
                json.dumps(
                    {
                        "ok": True,
                        "status": "waiting",
                        "session_id": "session-a",
                        "updated_at": 999.0,
                        "timeout_seconds": 45,
                        "deadline_at": 1045.0,
                    }
                ),
                encoding="utf-8",
            )

            with patch("Apps.UI.views._browser_portal_roots", return_value=[root]):
                with patch("Apps.UI.views.time.time", return_value=1000.0):
                    response = Client().post(
                        reverse("browser_portal_event_api"),
                        data=json.dumps({"type": "finish", "session_id": "session-a"}),
                        content_type="application/json",
                    )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["queued"])
            self.assertEqual(payload["status"], "done")
            queued_events = list(events_dir.glob("event_*.json"))
            self.assertEqual(len(queued_events), 1)
            queued_payload = json.loads(queued_events[0].read_text(encoding="utf-8"))
            self.assertEqual(queued_payload["type"], "finish")
            self.assertEqual(queued_payload["session_id"], "session-a")
