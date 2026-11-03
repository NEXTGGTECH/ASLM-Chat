# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


# One vetted onion service from the registry seed. `onion` is the last-known address
# (seed/fallback); the resolver refreshes it from `clearnet_anchor`'s Onion-Location.
# The category is one of media, infosec, rights, privacy_mail, whistleblow or archive, and
# clearnet_anchor is the TLS https URL that authoritatively publishes the onion address.
@dataclass(frozen=True, slots=True)
class OnionService:
    name: str
    category: str
    clearnet_anchor: str
    onion: str

    # Return the lowercase .onion host of the seeded address (for logging and source_domain).
    @property
    def onion_host(self) -> str:
        return (urlparse(self.onion).netloc or "").lower()
