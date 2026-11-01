# Copyright NEXTGGTECH. Elastic License 2.0.

from .brave import BraveParser
from .duckduckgo import DuckDuckGoParser
from .google import GoogleHtmlParser
from .google_cse import GoogleParser
from .models import EngineParseResult, EngineRequest, ParseStatus, SearchResult
from .qwant import QwantParser
from .startpage import StartpageParser
from .yandex import LegacyYandexParser, YandexParser

__all__ = [
    "BraveParser",
    "DuckDuckGoParser",
    "EngineParseResult",
    "EngineRequest",
    "GoogleParser",
    "GoogleHtmlParser",
    "ParseStatus",
    "QwantParser",
    "SearchResult",
    "StartpageParser",
    "LegacyYandexParser",
    "YandexParser",
]
