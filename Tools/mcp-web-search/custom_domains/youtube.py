# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import logging
import re
from urllib.parse import urlparse

from core.fetch.thread_pool import io_pool as _io_pool

from custom_domains.base import FetchContext, PageResult

logger = logging.getLogger("custom_domains.youtube")

# Hosts (without the www./m. prefix) of YouTube watch and short links.
_YT_HOSTS = ("youtube.com", "youtu.be")


# URL helpers

# Return the lowercase host of the URL without the www./m. prefix.
def _host(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.").removeprefix("m.")


# Return True for YouTube watch/short URLs.
def is_youtube(url: str) -> bool:
    return _host(url) in _YT_HOSTS


# Parse the 11-char YouTube video id from a URL (watch, youtu.be, /v/, /embed/ and /shorts/ forms).
def youtube_video_id(url: str) -> str | None:
    m = re.search(r"(?:v=|/v/|youtu\.be/|/embed/|/shorts/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None


# Transcript fetching

# Fetch a YouTube transcript: youtube-transcript-api first, then yt-dlp fallback. Return the
# transcript text, or an "Error: ..." string when the id cannot be parsed or no transcript exists.
async def fetch_youtube_transcript(url: str) -> str:
    video_id = youtube_video_id(url)
    if not video_id:
        return f"Error: Could not extract video ID from: {url}"

    loop = asyncio.get_running_loop()

    # Attempt 1: youtube-transcript-api (one API call, no file download).
    # Returns transcript text, "ERR:<message>" on known errors, or None to continue.
    def _yta_try() -> str | None:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
        except ImportError:
            # The library is optional; continue with the yt-dlp fallback.
            return None

        # Transcript languages in order of preference.
        PREFERRED = ["ru", "en", "uk", "de", "fr"]

        # List the available transcripts, with a 15 second timeout on every HTTP request where the
        # library accepts a custom http_client. Newer library versions expose list() on the
        # instance, older ones a static list_transcripts().
        try:
            import requests as _req

            # requests session that applies a default timeout to every request.
            class _TimedSession(_req.Session):
                # Send the request with a default 15 second timeout unless the caller set one.
                def request(self, method, url, **kwargs):  # type: ignore[override]
                    kwargs.setdefault("timeout", 15)
                    return super().request(method, url, **kwargs)

            try:
                api = YouTubeTranscriptApi(http_client=_TimedSession())
            except TypeError:
                api = YouTubeTranscriptApi()
            if hasattr(api, "list"):
                transcript_list = api.list(video_id)
            else:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        except Exception as e:
            err = str(e)
            if "unavailable" in err.lower() or "no longer available" in err.lower():
                return f"ERR:Video unavailable: {url}"
            logger.debug("youtube_transcript_api list() failed for %s: %s", url, e)
            return None

        # Choose a transcript: a manually created one in the first available preferred language,
        # else an auto-generated one, else any transcript (translated to English when its language
        # is not a preferred one).
        fetched = None
        for lang in PREFERRED:
            try:
                fetched = transcript_list.find_manually_created_transcript([lang])
                break
            except Exception:
                logger.debug("Manual transcript unavailable for %s lang=%s", url, lang)
                continue

        if fetched is None:
            for lang in PREFERRED:
                try:
                    fetched = transcript_list.find_generated_transcript([lang])
                    break
                except Exception:
                    logger.debug("Generated transcript unavailable for %s lang=%s", url, lang)
                    continue

        if fetched is None:
            try:
                fetched = next(iter(transcript_list))
                if fetched.language_code not in PREFERRED:
                    try:
                        fetched = fetched.translate("en")
                    except Exception:
                        logger.debug("Transcript translation failed for %s from %s", url, fetched.language_code)
                        pass
            except StopIteration:
                logger.debug("No transcripts available via youtube_transcript_api for %s", url)
                return None

        # Download the chosen transcript and join its snippets into one text.
        try:
            data = fetched.fetch()
            parts = [
                e.get("text", "") if isinstance(e, dict) else getattr(e, "text", "")
                for e in data
            ]
            text = " ".join(p for p in parts if p)
            if text:
                lang_code = getattr(fetched, "language_code", "?")
                return f"YouTube transcript [{lang_code}]\nVideo: {url}\n\n{text}"
        except Exception:
            logger.debug("youtube_transcript_api fetch() failed for %s", url, exc_info=True)
        return None

    # Run attempt 1 on the I/O pool; "ERR:" marks a known terminal error (e.g. unavailable video).
    result = await loop.run_in_executor(_io_pool, _yta_try)
    if result and result.startswith("ERR:"):
        return f"Error: {result[4:]}"
    if result:
        return result

    # Attempt 2: yt-dlp (heavier — downloads subtitle files).
    # Returns the transcript text, or None when no subtitles could be obtained.
    def _yt_dlp_try() -> str | None:
        try:
            import glob as _glob
            import os
            import tempfile

            import yt_dlp

            # yt-dlp logger that discards every message.
            class _Silent:
                # Discard a debug message.
                def debug(self, msg: str) -> None: pass

                # Discard an info message.
                def info(self, msg: str) -> None: pass

                # Discard a warning message.
                def warning(self, msg: str) -> None: pass

                # Discard an error message.
                def error(self, msg: str) -> None: pass

            # Download only the subtitle files (manual and automatic, VTT) into a temp directory.
            with tempfile.TemporaryDirectory() as tmpdir:
                opts = {
                    "skip_download": True,
                    "writesubtitles": True,
                    "writeautomaticsub": True,
                    "subtitleslangs": ["ru", "en", "uk", "de", "fr", "all"],
                    "subtitlesformat": "vtt",
                    "outtmpl": os.path.join(tmpdir, "sub"),
                    "quiet": True,
                    "no_warnings": True,
                    "logger": _Silent(),
                    "socket_timeout": 15,
                    "retries": 1,
                }
                try:
                    with yt_dlp.YoutubeDL(opts) as ydl:
                        ydl.download([url])
                except Exception:
                    logger.debug("yt-dlp subtitle download failed for %s", url, exc_info=True)
                    return None
                vtt_files = _glob.glob(os.path.join(tmpdir, "*.vtt"))
                if not vtt_files:
                    return None

                # Pick the subtitle file of the first preferred language, else the first file found.
                PREFERRED_LANGS = ["ru", "en", "uk", "de", "fr"]
                chosen = vtt_files[0]
                for lang in PREFERRED_LANGS:
                    candidates = [f for f in vtt_files if f".{lang}." in os.path.basename(f)]
                    if candidates:
                        chosen = candidates[0]
                        break

                # Strip the VTT headers, cue timings, cue numbers and markup, dropping repeated lines.
                raw = open(chosen, encoding="utf-8", errors="replace").read()
                lines, seen = [], set()
                for line in raw.splitlines():
                    line = line.strip()
                    if not line or line.startswith("WEBVTT") or "-->" in line or re.match(r"^\d+$", line):
                        continue
                    clean = re.sub(r"<[^>]+>", "", line).strip()
                    if clean and clean not in seen:
                        seen.add(clean)
                        lines.append(clean)
                text = " ".join(lines)
                return f"YouTube transcript (yt-dlp)\nVideo: {url}\n\n{text}" if text else None
        except Exception:
            logger.debug("yt-dlp transcript fallback crashed for %s", url, exc_info=True)
            return None

    # Run attempt 2 on the I/O pool and report an error when it found nothing either.
    result = await loop.run_in_executor(_io_pool, _yt_dlp_try)
    return result or f"Error: No transcript available for: {url}"


# Handler

# Unified handler: return a YouTube video transcript as markdown text.
class YouTubeHandler:
    name = "youtube"
    fallback_to_generic = False

    # read_page only; web_search keeps YouTube links as snippets instead of fetching transcripts inline.
    scope = "read_page"

    # Return True for YouTube watch/short URLs.
    def matches(self, url: str) -> bool:
        return is_youtube(url)

    # Fetch the transcript; errors are returned verbatim (terminal, no fallback).
    async def read(self, url: str, ctx: FetchContext) -> PageResult:
        markdown = await fetch_youtube_transcript(url)
        ok = bool(markdown) and not markdown.startswith("Error:")
        return PageResult(
            markdown=markdown,
            ok=ok,
            method="youtube_transcript",
            error="" if ok else markdown,
        )


# Handler instance registered in custom_domains/__init__.py.
HANDLER = YouTubeHandler()
