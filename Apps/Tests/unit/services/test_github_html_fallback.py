# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from django.test import SimpleTestCase

from Services import github_html_fallback, user_mcp_client


pytestmark = [pytest.mark.unit]


class GitHubHtmlFallbackTests(SimpleTestCase):
    def test_build_url_preserves_query_scope_and_sort(self):
        url = github_html_fallback.build_github_html_search_url(
            "search_issues",
            {
                "query": "rate limit",
                "owner": "openai",
                "repo": "codex",
                "sort": "updated",
                "order": "desc",
                "page": 2,
            },
        )

        self.assertIn("q=rate+limit+repo%3Aopenai%2Fcodex+is%3Aissue", url)
        self.assertIn("type=issues", url)
        self.assertIn("s=updated", url)
        self.assertIn("o=desc", url)
        self.assertIn("p=2", url)

    def test_parse_repository_results_from_public_html(self):
        html = """
        <div data-testid="results-list">
          <div class="Result-module__Result__abc">
            <a href="/openai/codex">openai/codex</a>
            <span>Lightweight coding agent</span>
          </div>
          <div class="Result-module__Result__def">
            <a href="/django/django">django/django</a>
            <span>The Web framework</span>
          </div>
        </div>
        """

        results = github_html_fallback.parse_github_html_search(
            html,
            search_type="repositories",
        )

        self.assertEqual([item["url"] for item in results], [
            "https://github.com/openai/codex",
            "https://github.com/django/django",
        ])
        self.assertIn("Lightweight coding agent", results[0]["snippet"])

    @patch.object(github_html_fallback, "search_github_html")
    @patch.object(user_mcp_client._session_manager, "request")
    def test_github_mcp_search_uses_html_only_after_rate_limit(
        self, request_mock, fallback_mock
    ):
        request_mock.return_value = SimpleNamespace(
            isError=False,
            structuredContent={
                "message": "API rate limit exceeded for 192.0.2.1",
                "documentation_url": "https://docs.github.com/rest/using-the-rest-api/rate-limits-for-the-rest-api",
            },
            content=[],
        )
        fallback_mock.return_value = "GITHUB_API_RATE_LIMIT_FALLBACK: HTML results"
        entry = SimpleNamespace(server_id="github")

        result = user_mcp_client.call_user_mcp_tool(
            entry,
            "search_repositories",
            {"query": "deep research"},
        )

        self.assertEqual(result, "GITHUB_API_RATE_LIMIT_FALLBACK: HTML results")
        fallback_mock.assert_called_once_with(
            "search_repositories",
            {"query": "deep research"},
        )

    @patch.object(github_html_fallback, "search_github_html")
    @patch.object(user_mcp_client._session_manager, "request")
    def test_non_search_github_call_does_not_use_html_fallback(
        self, request_mock, fallback_mock
    ):
        request_mock.return_value = SimpleNamespace(
            isError=False,
            structuredContent={"message": "API rate limit exceeded"},
            content=[],
        )
        entry = SimpleNamespace(server_id="github")

        result = user_mcp_client.call_user_mcp_tool(
            entry,
            "create_issue",
            {"owner": "o", "repo": "r"},
        )

        self.assertIn("API rate limit exceeded", result)
        fallback_mock.assert_not_called()

