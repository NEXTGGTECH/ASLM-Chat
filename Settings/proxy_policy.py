# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import os
import sys
import urllib.request
from collections.abc import Mapping, MutableMapping
from typing import Any
from urllib.parse import urlsplit


# Loopback spellings that must always bypass a proxy.
LOOPBACK_PROXY_BYPASS_HOSTS = ("localhost", "127.0.0.1", "::1")

# Proxy environment families (HTTP_PROXY, HTTPS_PROXY, ALL_PROXY) handled when merging proxy settings.
PROXY_SCHEMES = ("http", "https", "all")


# Return every non-empty value for one case-insensitive environment key.
def _environment_values(environ: Mapping[str, str], key: str) -> list[str]:
    normalized_key = key.casefold()
    return [
        str(value).strip()
        for env_key, value in environ.items()
        if str(env_key).casefold() == normalized_key and str(value).strip()
    ]


# Return whether an environment key is present, including an explicit empty value.
def _environment_has_key(environ: Mapping[str, str], key: str) -> bool:
    normalized_key = key.casefold()
    return any(str(env_key).casefold() == normalized_key for env_key in environ)


# Split comma- and Windows-style semicolon-delimited proxy bypass values.
def _split_bypass_values(values: list[Any]) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()

    for value in values:
        for raw_token in str(value or "").replace(";", ",").split(","):
            token = raw_token.strip()
            if not token:
                continue

            # HTTPX expects a bare IPv6 literal in NO_PROXY and adds URL
            # brackets itself. Preserving [::1] makes HTTPX parse ':1]' as a port.
            if token.casefold() == "[::1]" or token.casefold().startswith("[::1]:"):
                token = "::1"
            normalized = token.casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            tokens.append(token)

    return tokens


# Read Windows proxy exclusions before NO_PROXY makes urllib prefer the environment.
def _read_windows_proxy_override() -> str:
    if os.name != "nt":
        return ""

    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
        ) as internet_settings:
            proxy_enabled = bool(winreg.QueryValueEx(internet_settings, "ProxyEnable")[0])
            if not proxy_enabled:
                return ""
            return str(winreg.QueryValueEx(internet_settings, "ProxyOverride")[0] or "")
    except (ImportError, OSError, TypeError, ValueError):
        return ""


# Capture environment and platform proxy settings before installing NO_PROXY.
def _discover_system_proxy_settings() -> tuple[dict[str, str], list[str]]:
    discovered: dict[str, str] = {
        str(key).lower(): str(value)
        for key, value in urllib.request.getproxies().items()
        if value
    }
    bypass_values: list[str] = [discovered.get("no", "")]

    # urllib stops consulting the Windows registry or macOS System Configuration
    # as soon as any *_PROXY variable exists. Read the native source as well so
    # adding NO_PROXY does not accidentally disable the proxy for external URLs.
    native_getter = None
    if os.name == "nt":
        native_getter = getattr(urllib.request, "getproxies_registry", None)
        bypass_values.append(_read_windows_proxy_override())
    elif sys.platform == "darwin":
        native_getter = getattr(urllib.request, "getproxies_macosx_sysconf", None)

    if callable(native_getter):
        try:
            native_proxies = native_getter()
        except (OSError, TypeError, ValueError):
            native_proxies = {}
        if isinstance(native_proxies, dict):
            for key, value in native_proxies.items():
                normalized_key = str(key).lower()
                if value and normalized_key not in discovered:
                    discovered[normalized_key] = str(value)
            bypass_values.append(str(native_proxies.get("no", "") or ""))

    return discovered, bypass_values


# Merge loopback exclusions into one environment without changing explicit proxies.
def apply_loopback_proxy_bypass(
    environ: MutableMapping[str, str] | None = None,
    *,
    system_proxies: Mapping[str, str] | None = None,
    system_bypass: list[str] | tuple[str, ...] | None = None,
) -> str:
    target = os.environ if environ is None else environ

    if system_proxies is None or system_bypass is None:
        discovered_proxies, discovered_bypass = _discover_system_proxy_settings()
        if system_proxies is None:
            system_proxies = discovered_proxies
        if system_bypass is None:
            system_bypass = discovered_bypass

    # Preserve the effective platform proxy for external traffic. Explicit
    # environment values, including empty values used to disable a proxy, win.
    for scheme in PROXY_SCHEMES:
        proxy_key = f"{scheme}_proxy"
        if _environment_has_key(target, proxy_key):
            continue
        proxy_value = str((system_proxies or {}).get(scheme, "") or "").strip()
        if not proxy_value:
            continue
        target[proxy_key.upper()] = proxy_value
        target[proxy_key] = proxy_value

    bypass_values = [
        *_environment_values(target, "NO_PROXY"),
        *(system_bypass or []),
        *LOOPBACK_PROXY_BYPASS_HOSTS,
    ]
    bypass_tokens = _split_bypass_values(bypass_values)
    merged_bypass = "*" if "*" in bypass_tokens else ",".join(bypass_tokens)

    # Some clients prefer lowercase while others document uppercase. Keep both
    # synchronized; os.environ itself folds them together on Windows.
    target["NO_PROXY"] = merged_bypass
    target["no_proxy"] = merged_bypass
    return merged_bypass


# Build the restricted proxy-related overlay passed to user-owned subprocesses.
def build_proxy_environment_overlay(
    extra_environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    extra = {str(key): str(value) for key, value in (extra_environment or {}).items()}
    overlay = dict(extra)

    # MCP intentionally inherits only an allow-list. Add proxy families without
    # leaking unrelated parent variables, while keeping explicit MCP values authoritative.
    for scheme in PROXY_SCHEMES:
        proxy_key = f"{scheme}_proxy"
        if _environment_has_key(extra, proxy_key):
            continue
        for env_key, value in os.environ.items():
            if str(env_key).casefold() == proxy_key:
                overlay[str(env_key)] = str(value)

    parent_bypass = _environment_values(os.environ, "NO_PROXY")
    apply_loopback_proxy_bypass(
        overlay,
        system_proxies={},
        system_bypass=parent_bypass,
    )
    return overlay


# Open one HTTP request without consulting any proxy configuration.
def urlopen_direct(request: Any, *, timeout: float):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    return opener.open(request, timeout=timeout)


# Return whether a request URL targets one of the supported loopback spellings.
def is_loopback_url(request: Any) -> bool:
    url = getattr(request, "full_url", request)
    try:
        hostname = urlsplit(str(url or "")).hostname
    except ValueError:
        return False
    return str(hostname or "").strip().casefold() in {
        "localhost",
        "127.0.0.1",
        "::1",
    }


# Use a direct opener only for loopback and preserve normal proxy behavior elsewhere.
def urlopen_with_loopback_bypass(request: Any, *, timeout: float):
    if is_loopback_url(request):
        return urlopen_direct(request, timeout=timeout)
    return urllib.request.urlopen(request, timeout=timeout)
