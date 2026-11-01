# Copyright NEXTGGTECH. Elastic License 2.0.

"""Fail-open, model-facing answer cards (Wikidata facts, accepted answers, repo/package status)."""

from .service import RichCardService, insert_cards_context

__all__ = ["RichCardService", "insert_cards_context"]
