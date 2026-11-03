# Copyright NEXTGGTECH. Elastic License 2.0.

import asyncio
import concurrent.futures
import importlib.util
import json
import logging
import re
import sys
import threading
from pathlib import Path
from typing import Any

from camoufox import DefaultAddons
from camoufox.async_api import AsyncCamoufox
from mcp.types import TextContent
from playwright.async_api import Browser, BrowserContext, Page

# This server's folder and the path of the sibling config module.
SERVER_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = SERVER_ROOT / "config.py"


# Load the sibling config module without relying on global sys.path order.
def _load_local_config():
    spec = importlib.util.spec_from_file_location("mcp_browser_agent_config", CONFIG_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load browser-agent config from {CONFIG_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Config module loaded once at import time.
_config = _load_local_config()

# Browser window, download folder and snapshot limits re-exported from the config module.
AUTO_TEXT_PREVIEW_LEN = _config.AUTO_TEXT_PREVIEW_LEN
BROWSER_HEIGHT = _config.BROWSER_HEIGHT
BROWSER_HEADLESS = _config.BROWSER_HEADLESS
BROWSER_WIDTH = _config.BROWSER_WIDTH
DOWNLOADS_DIR = _config.DOWNLOADS_DIR
MAX_A11Y_DEPTH = _config.MAX_A11Y_DEPTH
MAX_ELEMENTS = _config.MAX_ELEMENTS
MAX_MAIN_INTERACTIVE = _config.MAX_MAIN_INTERACTIVE


# Build deterministic Camoufox launch settings. The bundled uBlock addon is
# optional and can be left half-extracted by an interrupted download.
def _camoufox_launch_options() -> dict[str, Any]:
    return {
        "headless": BROWSER_HEADLESS,
        "window": (BROWSER_WIDTH, BROWSER_HEIGHT),
        "exclude_addons": [DefaultAddons.UBO],
    }


# Keep one dedicated event loop alive for all browser operations. The loop runs on its own daemon thread
# so that browser coroutines stay on a single loop while callers submit work from any thread or loop.
class BrowserRuntime:
    # Initialize empty loop/thread handles for the dedicated browser event loop.
    def __init__(self) -> None:
        # The loop and the thread running it; both are None while the runtime is stopped.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

        # Set once the thread has created its loop, and a lock guarding start and stop.
        self._ready = threading.Event()
        self._lock = threading.Lock()

    # Run the dedicated asyncio event loop on a background thread.
    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            loop.close()

    # Start the background loop thread when it is not already running.
    def ensure_started(self) -> None:
        if self._thread is not None and self._thread.is_alive() and self._loop is not None:
            return

        with self._lock:
            # Check again under the lock: another thread may have started the loop meanwhile.
            if self._thread is not None and self._thread.is_alive() and self._loop is not None:
                return

            self._ready.clear()
            self._thread = threading.Thread(
                target=self._thread_main,
                name="mcp-browser-agent-loop",
                daemon=True,
            )
            self._thread.start()
            self._ready.wait()

    # Schedule a coroutine on the dedicated browser loop from any thread.
    def submit(self, coro) -> concurrent.futures.Future:
        self.ensure_started()
        assert self._loop is not None
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    # Stop the dedicated browser loop after browser resources are closed.
    def close(self) -> None:
        # Detach the loop and thread handles under the lock so no new work is submitted to them.
        with self._lock:
            loop = self._loop
            thread = self._thread
            self._loop = None
            self._thread = None
            self._ready.clear()

        # Stop the loop and wait briefly for its thread (unless close() runs on that thread itself).
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)

        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=5.0)


# Process-wide runtime that owns the dedicated browser event loop.
_browser_runtime = BrowserRuntime()


# Stop the shared browser runtime loop during worker shutdown.
def close_browser_runtime() -> None:
    _browser_runtime.close()


# Run a browser coroutine on the dedicated browser loop with optional keepalive logs. While the coroutine
# runs, a debug log message is sent to the MCP session every interval seconds (until one send fails), and
# cancelling the caller cancels the underlying coroutine too.
async def run_in_browser_loop(
    coro,
    session=None,
    interval: float = 3.0,
    message: str = "working...",
):
    # Submit to the browser loop and await the result from the caller's loop.
    future = _browser_runtime.submit(coro)
    wrapped = asyncio.wrap_future(future)

    try:
        while True:
            done, _pending = await asyncio.wait({wrapped}, timeout=interval)
            if wrapped in done:
                return wrapped.result()

            # Still running: send a keepalive so the client does not time out.
            if session is not None:
                try:
                    send_log_message = getattr(session, "send_log_message", None)
                    if send_log_message is not None:
                        await send_log_message(level="debug", data=message, logger="browser-agent")
                except BaseException:
                    # A failing session stops further keepalives; the browser work itself continues.
                    session = None
                    pass
    except asyncio.CancelledError:
        future.cancel()
        raise


# On Windows, patch the destructors of asyncio transports at import time: Playwright/Camoufox may leave
# subprocess transports pending on shutdown, and guarding their destructors keeps tool calls from emitting
# false crash traces.
if sys.platform.startswith("win"):
    try:
        import asyncio.base_subprocess as _base_subprocess
        import asyncio.proactor_events as _proactor_events

        _orig_pipe_del = _proactor_events._ProactorBasePipeTransport.__del__
        _orig_subproc_del = _base_subprocess.BaseSubprocessTransport.__del__

        # Swallow destructor errors from orphaned Playwright pipe transports on Windows.
        def _safe_pipe_del(self):
            try:
                _orig_pipe_del(self)
            except Exception:
                # A destructor failure on an orphaned transport is harmless noise at shutdown.
                pass

        # Swallow destructor errors from orphaned subprocess transports on Windows.
        def _safe_subproc_del(self):
            try:
                _orig_subproc_del(self)
            except Exception:
                # A destructor failure on an orphaned transport is harmless noise at shutdown.
                pass

        # Replace the original destructors with the guarded versions.
        _proactor_events._ProactorBasePipeTransport.__del__ = _safe_pipe_del
        _base_subprocess.BaseSubprocessTransport.__del__ = _safe_subproc_del
    except Exception:
        # The asyncio internals differ between Python versions; skipping the patch only costs log noise.
        pass


# Log to stderr, leaving stdout free for the protocol streams, under the shared "browser-agent" logger.
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    stream=sys.stderr,
)
log = logging.getLogger("browser-agent")


# ARIA roles of controls the model can act on.
INTERACTIVE_ROLES = {
    "button",
    "link",
    "textbox",
    "searchbox",
    "combobox",
    "checkbox",
    "radio",
    "switch",
    "tab",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "option",
    "slider",
    "spinbutton", "treeitem",
}

# Roles of controls that accept typed text.
TEXT_ENTRY_ROLES = {
    "textbox",
    "searchbox",
    "combobox",
    "spinbutton",
}

# Roles of controls that are clicked or toggled rather than typed into or followed as links.
ACTION_CONTROL_ROLES = {
    "button",
    "checkbox",
    "radio",
    "switch",
    "tab",
    "menuitem",
    "menuitemcheckbox",
    "menuitemradio",
    "option",
    "slider",
    "treeitem",
}

# Roles that are kept in the snapshot even without an accessible name: interactive controls plus structure.
SEMANTIC_ROLES = INTERACTIVE_ROLES | {
    "heading",
    "img",
    "navigation",
    "main",
    "banner",
    "contentinfo",
    "complementary",
    "form",
    "region",
    "list",
    "listitem",
    "table",
    "row",
    "cell",
    "dialog",
    "alert",
    "status",
    "progressbar",
    "separator",
    "toolbar",
}

# Page landmark roles tracked while walking the tree; each element is tagged with its enclosing landmark.
LANDMARK_ROLES = {
    "banner",
    "main",
    "contentinfo",
    "navigation",
    "complementary",
    "form",
    "region",
}

# Presentational roles that are dropped when they have no accessible name.
SKIP_ROLES = {
    "none",
    "presentation",
    "generic",
    "paragraph",
    "text",
    "strong",
    "emphasis",
    "code",
}


# Accessible names of decorative or boilerplate controls (avatars, copy and reply buttons, social and
# footer links, in English and Russian) that are hidden from the compact snapshot.
_NOISE_NAME_PATTERNS = re.compile(
    r"^(User avatar|icon|Скопировать код|Показать полностью|"
    r"Показать обсуждения|Самые популярные|Оставить комментарий|"
    r"принципы сообщества|отдельный форум|Посмотреть \d+ отв|"
    r"Ответить|Не нравится|VKontakte|Telegram|Get it on|Skolkovo|cc-wiki|"
    r"использования|конфиденциальности|Тарифы|Прессе|Команда|Помощь|"
    r"О нас|Контакты|Каталог)$",
    re.IGNORECASE,
)

# Matches names made only of digits, such as comment counters on buttons.
_COMMENT_BUTTON_RE = re.compile(r"^\d+$")


# Patterns for parsing one line of the aria snapshot: the line (indent, role, optional quoted name and
# the trailing attributes) and a single "[key]" or "[key=value]" attribute.
_LINE_RE = re.compile(
    r'^(?P<indent>\s*)'
    r'- (?P<role>[A-Za-z][A-Za-z0-9_-]*)'
    r'(?::(?=\s*$))?'
    r'(?:\s+"(?P<name>[^"]*)")?'
    r'(?P<attrs>.*)?$'
)
_ATTR_RE = re.compile(r'\[(?P<key>\w+)(?:=(?P<val>[^\]]+))?\]')


# Accessibility tree extraction

# Extract the current accessibility tree and assign stable element refs. Return the list of element dicts
# (ref, role, name, landmark, depth and state flags) and a readable indented text tree. Refs are numbered
# "e0", "e1", ... in document order; the element cap is doubled for a full snapshot.
async def get_accessibility_tree(page: Page, full: bool = False) -> tuple[list[dict], str]:
    # Take the aria snapshot of the page body; an unreadable page yields an empty tree.
    try:
        yaml_text = await page.locator("body").aria_snapshot(timeout=10000)
    except Exception as exc:
        log.warning(f"aria_snapshot failed: {exc}")
        return [], "(accessibility tree unavailable)"

    elements: list[dict] = []
    lines: list[str] = []
    ref_counter = 0
    max_elements = MAX_ELEMENTS * 2 if full else MAX_ELEMENTS

    # Keep the active landmark in sync with snapshot indentation.
    landmark_stack: list[tuple[int, str]] = []
    current_landmark = "unknown"

    # Walk the YAML-like aria snapshot line by line.
    for raw_line in yaml_text.splitlines():
        stripped = raw_line.strip()
        if not stripped or not stripped.startswith("-"):
            continue

        match = _LINE_RE.match(raw_line)
        if not match:
            continue

        # Split the line into depth (two spaces per level), role, name and attribute text.
        indent_str = match.group("indent") or ""
        depth = len(indent_str) // 2
        role = (match.group("role") or "").strip().rstrip(":")
        name = match.group("name") or ""
        attrs_str = match.group("attrs") or ""

        # Skip lines that are too deep, beyond the element cap, or carry no useful information.
        if depth > MAX_A11Y_DEPTH or ref_counter >= max_elements:
            continue

        if role in SKIP_ROLES and not name:
            continue

        is_interactive = role in INTERACTIVE_ROLES
        is_semantic = role in SEMANTIC_ROLES

        if not (is_semantic or name):
            continue

        # Drop landmarks that are no longer active at the current depth, then work out which landmark
        # encloses this element (a landmark element becomes the current one itself).
        while landmark_stack and landmark_stack[-1][0] >= depth:
            landmark_stack.pop()

        if role in LANDMARK_ROLES:
            landmark_stack.append((depth, role))
            current_landmark = role
        elif landmark_stack:
            current_landmark = landmark_stack[-1][1]
        else:
            current_landmark = "unknown"

        # Convert bracket attributes into plain Python values: flags and "true"/"false" become booleans,
        # numeric values become integers and everything else stays a string.
        attrs: dict[str, Any] = {}
        for attr_match in _ATTR_RE.finditer(attrs_str):
            key = attr_match.group("key")
            val = attr_match.group("val")
            if val is None:
                attrs[key] = True
            else:
                val = val.strip('"\'')
                if val == "true":
                    attrs[key] = True
                elif val == "false":
                    attrs[key] = False
                else:
                    try:
                        attrs[key] = int(val)
                    except ValueError:
                        attrs[key] = val

        # Assign the next ref and record the element with the attributes that matter to the model.
        ref = f"e{ref_counter}"
        ref_counter += 1

        elem: dict[str, Any] = {
            "ref": ref, "role": role, "name": name,
            "landmark": current_landmark, "depth": depth,
        }
        if is_interactive:
            elem["interactive"] = True
        if "value" in attrs:
            elem["value"] = str(attrs["value"])
        if "level" in attrs:
            elem["level"] = attrs["level"]
        if "checked" in attrs:
            elem["checked"] = attrs["checked"]
        if "expanded" in attrs:
            elem["expanded"] = attrs["expanded"]
        if attrs.get("disabled"):
            elem["disabled"] = True
        if attrs.get("required"):
            elem["required"] = True

        elements.append(elem)

        # Keep the readable tree focused on content-bearing regions.
        show_in_tree = full or current_landmark in ("main", "dialog", "form", "region", "unknown", "complementary")

        # Render the element as one indented line of the readable tree.
        if show_in_tree:
            indent = "  " * depth
            label_parts = [f"[{ref}]", role]
            if name:
                label_parts.append(f'"{name}"')
            if elem.get("value"):
                label_parts.append(f'value="{elem["value"]}"')
            if elem.get("level"):
                label_parts.append(f'level={elem["level"]}')
            if elem.get("checked") is not None:
                label_parts.append(f'checked={elem["checked"]}')
            if elem.get("disabled"):
                label_parts.append("(disabled)")
            lines.append(f"{indent}{' '.join(label_parts)}")

    tree_text = "\n".join(lines)
    return elements, tree_text


# Accessibility tree filtering

# Return whether an element should be hidden from model-facing snapshot output.
def _is_noise_element(elem: dict) -> bool:
    name = elem.get("name", "")
    role = elem.get("role", "")
    landmark = elem.get("landmark", "")

    # Keep content-region controls visible even if they look generic.
    if landmark in ("main", "dialog", "form"):
        return False

    if not name and role in ("link", "button", "img"):
        return True

    if name and _NOISE_NAME_PATTERNS.match(name):
        return True

    if role == "button" and _COMMENT_BUTTON_RE.match(name) and landmark != "main":
        return True

    if "avatar" in name.lower() or "User avatar" in name:
        return True

    return False


# Return the stable, model-facing subset of one accessibility element (the landmark is reported as "region",
# and text-entry roles are marked editable).
def _snapshot_element_payload(el: dict) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ref": el.get("ref"),
        "role": el.get("role"),
        "name": el.get("name") or "",
        "region": el.get("landmark") or "unknown",
    }

    if el.get("role") in TEXT_ENTRY_ROLES:
        payload["editable"] = True
    if el.get("value"):
        payload["value"] = el.get("value")
    if el.get("checked") is not None:
        payload["checked"] = el.get("checked")
    if el.get("expanded") is not None:
        payload["expanded"] = el.get("expanded")
    if el.get("disabled"):
        payload["disabled"] = True
    if el.get("level") is not None:
        payload["level"] = el.get("level")

    return payload


# Format one element as a compact markdown action line.
def _format_control_line(el: dict) -> str:
    payload = _snapshot_element_payload(el)
    parts = [f'[{payload["ref"]}]', str(payload["role"])]
    if payload.get("name"):
        parts.append(json.dumps(str(payload["name"]), ensure_ascii=False))
    if payload.get("value"):
        parts.append(f'value={json.dumps(str(payload["value"]), ensure_ascii=False)}')
    if payload.get("editable"):
        parts.append("editable")
    if payload.get("disabled"):
        parts.append("disabled")
    if "checked" in payload:
        parts.append(f'checked={str(payload["checked"]).lower()}')
    if "expanded" in payload:
        parts.append(f'expanded={str(payload["expanded"]).lower()}')
    if payload.get("region") and payload["region"] != "unknown":
        parts.append(f'region={payload["region"]}')
    return "- " + " ".join(parts)


# Pick interactive controls that belong in the default snapshot.
def _filter_snapshot_controls(
    elements: list[dict],
    *,
    full: bool,
) -> list[dict]:
    controls = [e for e in elements if e.get("interactive")]
    if not full:
        controls = [e for e in controls if not _is_noise_element(e)]

    return controls


# Group interactive elements by the operation a model can perform.
def _group_controls(elements: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {
        "text_inputs": [],
        "buttons": [],
        "links": [],
        "other_controls": [],
    }

    for el in elements:
        role = el.get("role")
        if role in TEXT_ENTRY_ROLES:
            groups["text_inputs"].append(el)
        elif role in ACTION_CONTROL_ROLES:
            groups["buttons"].append(el)
        elif role == "link":
            groups["links"].append(el)
        else:
            groups["other_controls"].append(el)

    return groups


# Format controls in a predictable grouped order for snapshot markdown. At most max_items controls are
# listed across all groups; a final line reports how many were left out and how to see them.
def _format_parsed_controls(
    elements: list[dict],
    *,
    full: bool,
    max_items: int,
) -> list[str]:
    selected = _filter_snapshot_controls(elements, full=full)
    groups = _group_controls(selected)
    lines: list[str] = []
    shown = 0

    # Group headings in display order, paired with the keys of the groups.
    labels = [
        ("text_inputs", "Text inputs"),
        ("buttons", "Buttons and controls"),
        ("links", "Links"),
        ("other_controls", "Other interactive elements"),
    ]

    # List each non-empty group under its heading until the item cap is reached.
    for key, label in labels:
        items = groups[key]
        if not items:
            continue
        section_lines: list[str] = []
        for el in items:
            if shown >= max_items:
                break
            section_lines.append(_format_control_line(el))
            shown += 1
        if section_lines:
            lines.append(f"\n### {label}")
            lines.extend(section_lines)
        if shown >= max_items:
            break

    # Tell the model about controls cut by the cap, or that the page has none.
    remaining = max(0, len(selected) - shown)
    if remaining:
        hint = "browser_snapshot(full=true)" if not full else "scroll or refine the request"
        lines.append(f"\n... {remaining} more controls hidden; use {hint}.")

    if not lines:
        lines.append("\n### Controls\nNo interactive controls found.")

    return lines


# Build a compact structured state block for models that prefer JSON.
def _build_parsed_state(elements: list[dict], *, full: bool, max_items: int) -> dict[str, Any]:
    selected = _filter_snapshot_controls(elements, full=full)
    groups = _group_controls(selected)

    return {
        "mode": "full" if full else "controls",
        "ref_rule": "Refs are fresh for this snapshot. After click/key/text/scroll, use the returned refs.",
        "counts": {
            "total_elements": len(elements),
            "visible_controls": len(selected),
            "listed_controls": min(len(selected), max_items),
            "text_inputs": len(groups["text_inputs"]),
            "buttons": len(groups["buttons"]),
            "links": len(groups["links"]),
        },
    }


# Snapshot text extraction

# Extract a short text preview from the main content area for full snapshots. The in-page script prefers
# the main landmark, temporarily hides navigation, footer and comment elements, reads the visible text,
# restores the page, then collapses whitespace and truncates to max_chars. Failures yield an empty string.
async def _extract_brief_text(page: Page, max_chars: int = AUTO_TEXT_PREVIEW_LEN) -> str:
    try:
        # Run the extraction script in the page; max_chars is passed in as its argument.
        text = await page.evaluate("""(maxChars) => {
            // Try main landmark first
            let el = document.querySelector('main, [role="main"]');
            if (!el || !el.innerText || el.innerText.trim().length < 50) {
                el = document.body;
            }
            
            // Strip common noise elements in-place temporarily
            const NOISE = 'nav, footer, header, aside, .comments, [class*="comment"], ' +
                          '[class*="sidebar"], .cookie, [role="banner"], [role="navigation"], ' +
                          '[role="contentinfo"], script, style, [class*="footer"]';
            const hidden = [];
            el.querySelectorAll(NOISE).forEach(n => {
                if (window.getComputedStyle(n).display !== 'none') {
                    hidden.push([n, n.style.display]);
                    n.style.display = 'none';
                }
            });
            
            let text = (el.innerText || el.textContent || '').trim();
            
            // Restore hidden
            hidden.forEach(([n, orig]) => { n.style.display = orig; });
            
            // Collapse whitespace
            text = text.replace(/\\n{3,}/g, '\\n\\n').replace(/[ \\t]{2,}/g, ' ');
            
            if (text.length > maxChars) {
                text = text.substring(0, maxChars) + '\\n[...]';
            }
            return text;
        }""", max_chars)
        return text or ""
    except Exception as exc:
        # The text preview is optional; log the failure and let the snapshot go out without it.
        log.debug(f"Brief text extraction failed: {exc}")
        return ""


# Rendering wait helpers

# Wait until SPA main content becomes readable without fixed sleeps. Polls up to six times (about 300 ms
# apart) for a non-busy main landmark with more than 100 characters of text, then waits one last window.
async def _wait_for_spa_content(page: Page, timeout_ms: int = 5000):
    # Poll the main landmark until meaningful text appears.
    for _ in range(6):
        try:
            has_content = await page.evaluate("""() => {
                const main = document.querySelector('main, [role="main"]');
                if (!main) return false;
                const isVisible = element => {
                    const style = window.getComputedStyle(element);
                    return style.display !== 'none' && style.visibility !== 'hidden' &&
                           element.getClientRects().length > 0;
                };
                const busy = main.matches('[aria-busy="true"], [data-loading="true"], [data-state="loading"]') ||
                    Array.from(main.querySelectorAll('[aria-busy="true"]')).some(isVisible);
                if (busy) return false;
                const text = main.innerText || '';
                return text.trim().length > 100;
            }""")
            if has_content:
                return
        except Exception:
            # The page may be navigating or closing; treat it as "not ready yet" and keep polling.
            pass

        # Use an in-page timer to keep the Playwright transport active.
        try:
            await page.evaluate("() => new Promise(r => setTimeout(r, 300))")
        except Exception:
            await asyncio.sleep(0.3)

    # Give the page one last short rendering window.
    try:
        await page.evaluate("() => new Promise(r => setTimeout(r, 300))")
    except Exception:
        await asyncio.sleep(0.3)


# Overlay and page-state detection

# Selectors of close buttons that are common on generic modal and popup windows.
_KNOWN_CLOSE_SELECTORS = [
    "[data-dismiss='modal']", "[data-action*='close' i]",
    "[data-action*='dismiss' i]", "[data-testid*='close' i]",
    "[data-testid*='dismiss' i]", "[name*='close' i]",
    ".modal-close", ".popup-close", ".dialog-close",
    ".close-button", ".btn-close", ".close-btn",
    "button.close", "[class*='close'][role='button']",
]

# Selectors of "accept cookies" buttons of common consent platforms and generic accept attributes.
_KNOWN_ACCEPT_SELECTORS = [
    "#onetrust-accept-btn-handler",
    "#accept-cookie-consent",
    ".cc-accept", ".cc-btn.cc-allow",
    "[data-testid='cookie-accept']", "[data-testid='accept-cookies']",
    "#CybotCookiebotDialogBodyLevelButtonLevelOptinAllowAll",
    ".js-accept-cookies", "#js-accept-cookies",
    "[data-action*='accept' i]", "[data-consent-action*='accept' i]",
    "[data-testid*='accept' i]", "[name*='accept' i]",
]


# Try to dismiss common cookie banners and blocking popups before snapshotting. Four steps, from safest
# to most heuristic: click a known accept selector, otherwise let an in-page script rank the buttons of
# banner-like containers; then click a known close selector, or a script-found close control; finally press
# Escape when a dialog is still open. Return labels of the actions taken (empty when nothing was dismissed).
async def _auto_dismiss_overlays(page: Page) -> list[str]:
    dismissed = []

    # Make a few passes because some banners re-render after the first click.
    for _ in range(3):
        closed_this_pass = False

        # Prefer known selectors before using broad DOM heuristics.
        for sel in _KNOWN_ACCEPT_SELECTORS:
            try:
                el = page.locator(sel).first
                if await el.is_visible(timeout=150):
                    await el.click(timeout=2000)
                    await page.wait_for_timeout(600)
                    dismissed.append(f"selector:{sel}")
                    closed_this_pass = True
                    break
            except Exception:
                # The selector matched nothing clickable on this page; try the next one.
                pass

        if closed_this_pass:
            continue

        # No known selector matched: score the buttons of banner-like containers in the page and click
        # the best accept-style one, never reject/manage ones.
        try:
            clicked = await page.evaluate("""() => {
                const containers = Array.from(document.querySelectorAll(
                    '[id*="cookie"], [id*="consent"], [id*="gdpr"], [id*="banner"], [class*="cookie"], [class*="consent"], [class*="gdpr"], [class*="banner"], [class*="overlay"], [class*="modal"], [role="dialog"], [role="alertdialog"]'
                )).filter(el => {
                    const style = window.getComputedStyle(el);
                    if (style.display === 'none' || style.visibility === 'hidden') return false;
                    const pos = style.position;
                    if (pos === 'fixed' || pos === 'sticky') return true;
                    return el.offsetParent !== null;
                });

                const identity = element => [
                    element.id,
                    typeof element.className === 'string' ? element.className : '',
                    element.getAttribute('name'),
                    element.getAttribute('data-testid'),
                    element.getAttribute('data-action'),
                    element.getAttribute('data-consent-action')
                ].filter(Boolean).join(' ').toLowerCase();

                const score = element => {
                    const value = identity(element);
                    if (/reject|decline|deny|manage|setting|preference/.test(value)) return -100;
                    if (/accept.?all|allow.?all|accept.?cookie|cookie.?accept/.test(value)) return 20;
                    if (/accept|agree|allow|consent|confirm/.test(value)) return 10;
                    return /primary/.test(value) ? 1 : 0;
                };

                for (const container of containers) {
                    const btns = Array.from(container.querySelectorAll(
                        'button, [role="button"], a[href="#"], input[type="button"], input[type="submit"]'
                    )).filter(b => b.offsetParent !== null);

                    const ranked = btns.map(btn => ({ btn, score: score(btn) }))
                        .filter(candidate => candidate.score >= 10)
                        .sort((left, right) => right.score - left.score);
                    if (ranked.length) {
                        ranked[0].btn.click();
                        return identity(ranked[0].btn).slice(0, 160) || 'structured-accept-control';
                    }
                }

                return null;
            }""")

            if clicked:
                await page.wait_for_timeout(600)
                dismissed.append(f"js-heuristic:'{clicked}'")
                closed_this_pass = True
        except Exception as e:
            log.debug(f"Overlay JS heuristic error: {e}")

        if not closed_this_pass:
            break

    # Try explicit close buttons on generic modal windows.
    for sel in _KNOWN_CLOSE_SELECTORS:
        try:
            el = page.locator(sel).first
            if await el.is_visible(timeout=100):
                await el.click(timeout=1500)
                await page.wait_for_timeout(400)
                dismissed.append(f"close-selector:{sel}")
                break
        except Exception:
            # The selector matched nothing clickable on this page; try the next one.
            pass

    # Fall back to markup heuristics if explicit selectors are missing.
    if not any(d.startswith("close-") for d in dismissed):
        try:
            clicked = await page.evaluate("""() => {
                const containers = Array.from(document.querySelectorAll(
                    '[role="dialog"], [role="alertdialog"], [aria-modal="true"], [class*="modal"], [class*="popup"], [class*="overlay"], [class*="lightbox"]'
                )).filter(el => {
                    const style = window.getComputedStyle(el);
                    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
                    const z = parseInt(style.zIndex) || 0;
                    return el.offsetParent !== null || style.position === 'fixed' || z > 100;
                });

                const fixedHigh = Array.from(document.querySelectorAll('*')).filter(el => {
                    const style = window.getComputedStyle(el);
                    const z = parseInt(style.zIndex) || 0;
                    return style.position === 'fixed' && z > 100 &&
                           style.display !== 'none' && style.visibility !== 'hidden' &&
                           el.offsetWidth > 100 && el.offsetHeight > 50;
                });

                const allContainers = [...new Set([...containers, ...fixedHigh])];
                const identity = element => [
                    element.id,
                    typeof element.className === 'string' ? element.className : '',
                    element.getAttribute('name'),
                    element.getAttribute('data-testid'),
                    element.getAttribute('data-action'),
                    element.getAttribute('data-dismiss')
                ].filter(Boolean).join(' ').toLowerCase();

                for (const container of allContainers) {
                    const btns = Array.from(container.querySelectorAll(
                        'button, [role="button"], a'
                    )).filter(b => b.offsetParent !== null || window.getComputedStyle(b).position === 'fixed');

                    for (const btn of btns) {
                        const label = String(btn.innerText || '').trim().toLowerCase();
                        if (/^[×✕✖x]$/.test(label)) {
                            btn.click();
                            return `icon:${label}`;
                        }
                        const value = identity(btn);
                        if (/close|dismiss|cancel|skip|later/.test(value) && !/accept|confirm|submit/.test(value)) {
                            btn.click();
                            return value.slice(0, 160) || 'structured-close-control';
                        }
                    }
                }
                return null;
            }""")

            if clicked:
                await page.wait_for_timeout(500)
                dismissed.append(f"popup-close:'{clicked}'")
        except Exception as e:
            log.debug(f"Popup close heuristic error: {e}")

    # Use Escape as the last low-risk fallback for modal dialogs.
    try:
        has_modal = await page.evaluate("""() => {
            const el = document.querySelector('[role="dialog"],[role="alertdialog"],[aria-modal="true"]');
            if (!el) return false;
            const style = window.getComputedStyle(el);
            return style.display !== 'none' && style.visibility !== 'hidden';
        }""")
        if has_modal:
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(400)
            dismissed.append("escape-key")
    except Exception as e:
        log.debug(f"Escape key dismiss error: {e}")

    return dismissed


# Detect CAPTCHA, login, cookie banner, and error-page situations for warnings. An in-page script checks
# for known widgets and markup; the findings are turned into warning lines that tell the model what to
# do next. Detection failures are logged and give no warnings.
async def _detect_page_situation(page: Page) -> list[str]:
    warnings: list[str] = []

    # Collect the raw findings from the page.
    try:
        situation = await page.evaluate("""() => {
            const r = {
                captcha: false, captcha_type: null,
                has_login: false,
                has_cookie_banner: false,
                is_error_page: false, error_type: null,
            };

            if (document.querySelector('iframe[src*="recaptcha"]') ||
                document.querySelector('.g-recaptcha') ||
                document.querySelector('[data-sitekey]') ||
                window.grecaptcha !== undefined) {
                r.captcha = true; r.captcha_type = 'reCAPTCHA';
            }
            if (document.querySelector('iframe[src*="hcaptcha"]') ||
                document.querySelector('.h-captcha') ||
                window.hcaptcha !== undefined) {
                r.captcha = true; r.captcha_type = 'hCaptcha';
            }
            if (document.querySelector('#challenge-form') ||
                document.title.includes('Just a moment') ||
                document.title.includes('Checking your browser')) {
                r.captcha = true; r.captcha_type = 'Cloudflare';
            }
            if (document.querySelector('.cf-turnstile') ||
                document.querySelector('iframe[src*="challenges.cloudflare"]')) {
                r.captcha = true; r.captcha_type = 'Turnstile';
            }

            const pwd = document.querySelector('input[type="password"]');
            if (pwd && pwd.offsetParent !== null) r.has_login = true;

            const cookieSelectors = [
                '#cookie-banner', '#cookie-notice', '#cookie-consent',
                '#cookieConsent', '#cookieBanner', '.cookie-banner',
                '.cookie-notice', '.cookie-consent', '.cookie-bar',
                '[id*="gdpr"]', '[class*="gdpr"]',
                '[aria-label*="cookie"]', '[aria-label*="Cookie"]',
                '#CybotCookiebotDialog', '#onetrust-banner-sdk', '.cc-banner',
            ];
            for (const sel of cookieSelectors) {
                try {
                    const el = document.querySelector(sel);
                    if (el && el.offsetParent !== null) { r.has_cookie_banner = true; break; }
                } catch(e) {}
            }

            const url = window.location.href;
            const title = document.title.toLowerCase();
            const h1 = document.querySelector('h1');
            const h1text = h1 ? h1.textContent.toLowerCase() : '';
            if (url.includes('/404') || title.includes('404') ||
                h1text.includes('not found') || h1text.includes('page not found')) {
                r.is_error_page = true; r.error_type = '404';
            } else if (url.includes('/403') || title.includes('403') ||
                       title.includes('forbidden') || title.includes('access denied')) {
                r.is_error_page = true; r.error_type = '403';
            } else if (title.includes('500') || title.includes('server error') ||
                       h1text.includes('server error')) {
                r.is_error_page = true; r.error_type = '500';
            }

            return r;
        }""")
    except Exception as exc:
        log.warning(f"Situation detection failed: {exc}")
        return warnings

    # Turn each finding into a warning line for the model.
    if situation.get("captcha"):
        ctype = situation.get("captcha_type", "unknown")
        warnings.append(
            f"⚠️ CAPTCHA detected ({ctype}) — "
            f"call browser_wait_for_user to let the user solve it manually."
        )
    if situation.get("has_login"):
        warnings.append(
            "⚠️ Login form detected (password field present) — "
            "call browser_wait_for_user if credentials are needed."
        )
    if situation.get("has_cookie_banner"):
        warnings.append(
            "⚠️ Cookie consent banner detected — "
            "consider clicking the accept button before proceeding."
        )
    if situation.get("is_error_page"):
        etype = situation.get("error_type", "unknown")
        warnings.append(
            f"⚠️ Error page detected (HTTP {etype}) — "
            f"the requested resource may be unavailable."
        )

    return warnings


# Describe a blocking overlay that remains visible after auto-dismiss attempts. An in-page script looks
# for a large, high z-index fixed or absolute element covering the viewport centre and classifies it
# (registration, newsletter, login wall, cookie banner, ...) from its id, classes, role and text. Return
# "kind" or 'kind: "text"', or None when no such overlay exists or detection fails.
async def _detect_undismissable_overlay(page: Page) -> str | None:
    # Run the detection script in the page.
    try:
        result = await page.evaluate("""() => {
            const vw = window.innerWidth;
            const vh = window.innerHeight;
            const cx = vw / 2;
            const cy = vh / 2;
            const minArea = vw * vh * 0.15;

            const candidates = Array.from(document.querySelectorAll('*')).filter(el => {
                const style = window.getComputedStyle(el);
                if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') return false;
                const pos = style.position;
                if (pos !== 'fixed' && pos !== 'absolute') return false;
                const z = parseInt(style.zIndex) || 0;
                if (z < 200) return false;
                const rect = el.getBoundingClientRect();
                if (rect.width * rect.height < minArea) return false;
                if (rect.left > cx || rect.right < cx || rect.top > cy || rect.bottom < cy) return false;
                return true;
            });

            if (candidates.length === 0) return null;

            const top = candidates.sort((a, b) => {
                const za = parseInt(window.getComputedStyle(a).zIndex) || 0;
                const zb = parseInt(window.getComputedStyle(b).zIndex) || 0;
                return zb - za;
            })[0];

            const id = top.id ? `#${top.id}` : '';
            const cls = top.className && typeof top.className === 'string'
                ? top.className.trim().split(/\\s+/).slice(0, 3).map(c => `.${c}`).join('') : '';
            const role = top.getAttribute('role') || top.tagName.toLowerCase();
            const text = (top.innerText || '').trim().slice(0, 80);

            const sig = `${id} ${cls} ${role} ${text}`.toLowerCase();
            let kind = 'overlay';
            if (/sign.?up|register|creat.*account|join/.test(sig)) kind = 'registration popup';
            else if (/subscri|newsletter|email/.test(sig)) kind = 'newsletter popup';
            else if (/log.?in|sign.?in|auth/.test(sig)) kind = 'login wall';
            else if (/cookie|consent|gdpr|privacy/.test(sig)) kind = 'cookie banner';
            else if (/age|verify|18/.test(sig)) kind = 'age gate';
            else if (/survey|feedback|rate/.test(sig)) kind = 'survey popup';
            else if (/promo|discount|offer|sale/.test(sig)) kind = 'promo popup';
            else if (/paywall|subscri|premium/.test(sig)) kind = 'paywall';

            return { kind, id, cls, role, text };
        }""")
    except Exception as exc:
        log.debug(f"Overlay detection error: {exc}")
        return None

    # Format the classification as a short description for the model.
    if result is None:
        return None

    kind = result.get("kind", "overlay")
    detail = result.get("text", "")[:60]
    return f"{kind}" + (f': "{detail}"' if detail else "")


# Browser lifecycle

# Lower-case fragments of exception text that mean the browser, context or driver connection is gone.
_BROWSER_CLOSED_MARKERS = (
    "targetclosederror",
    "target page, context or browser has been closed",
    "browser has been closed",
    "browser closed",
    "context has been closed",
    "connection closed",
    "connection closed while reading from the driver",
    "playwright connection closed",
)


# Return whether an exception indicates the browser process or session died.
def is_browser_closed_error(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _BROWSER_CLOSED_MARKERS)


# Return the most recent non-blank, non-about:blank URL from the navigation history buffer.
def last_known_url() -> str:
    for url in reversed(_nav_history):
        value = str(url or "").strip()
        if value and value not in {"about:blank", ""}:
            return value
    return ""


# Store the shared Camoufox/Playwright session used by MCP browser tools.
class BrowserState:
    # Initialize empty browser, context, page, and tool context handles.
    def __init__(self):
        # The Camoufox context manager that owns the launched browser, and the Playwright objects created
        # from it; all are None until ensure_open() launches the browser.
        self._camoufox_cm: AsyncCamoufox | None = None
        self.browser: Browser | None = None
        self.context: BrowserContext | None = None
        self.page: Page | None = None

        # Context of the current tool call (module and project folders, sandbox flags, MCP session).
        self.tool_context: dict[str, Any] = {}

    # Return the active downloads directory for the current tool context: the sandbox downloads folder when
    # sandbox mode is on and the module folder is known, otherwise the configured DOWNLOADS_DIR.
    def downloads_dir(self) -> Path:
        safe_context = self.tool_context or {}
        module_dir = str(safe_context.get("module_dir") or safe_context.get("project_dir") or "").strip()
        selected = safe_context.get("selected_tool_server_ids")
        sandbox_enabled = bool(safe_context.get("sandbox_enabled"))
        if isinstance(selected, list):
            sandbox_enabled = sandbox_enabled or any(str(item) == "sandbox" for item in selected)
        if sandbox_enabled and module_dir:
            return Path(module_dir) / "Tools" / "mcp-sandbox" / "_sandbox" / "downloads"
        return DOWNLOADS_DIR

    # Return whether the stored Playwright page can still receive commands. A page that is closed, times
    # out on a trivial evaluate, or fails with a browser-closed error is dead; any other evaluate error is
    # treated as non-fatal and the page is still considered alive.
    async def _has_live_page(self) -> bool:
        if self.page is None:
            return False

        try:
            if self.page.is_closed():
                return False
        except Exception:
            # Even asking the page about its state failed, so it is unusable.
            return False

        # Health check: run a trivial script in the page.
        try:
            await asyncio.wait_for(self.page.evaluate("() => true"), timeout=2.0)
            return True
        except asyncio.TimeoutError:
            log.warning("Browser page health check timed out; relaunching browser session.")
            return False
        except Exception as exc:
            if is_browser_closed_error(exc):
                return False
            log.debug("Browser page health check failed but did not look fatal: %s", exc)
            return True

    # Launch the browser lazily on first use; return True when a new session started. A half-dead session
    # is closed first, a failed launch cleans up everything and raises a RuntimeError, and a new page gets
    # handlers that save downloads and auto-accept dialogs.
    async def ensure_open(self) -> bool:
        if await self._has_live_page():
            return False

        # Discard leftovers of a previous session that is no longer usable.
        if (
            self.page is not None
            or self.context is not None
            or self.browser is not None
            or self._camoufox_cm is not None
        ):
            log.warning("Stored Camoufox page is no longer usable; relaunching browser session.")
            await self.close()

        # Launch Camoufox and open a context and page of the configured size.
        log.info("Launching camoufox browser (Firefox stealth, headless=%s)...", BROWSER_HEADLESS)

        try:
            self._camoufox_cm = AsyncCamoufox(**_camoufox_launch_options())
            browser = await self._camoufox_cm.__aenter__()
            self.browser = browser

            self.context = await browser.new_context(
                viewport={"width": BROWSER_WIDTH, "height": BROWSER_HEIGHT},
                accept_downloads=True,
            )
            self.page = await self.context.new_page()
        except Exception as exc:
            # Roll back the partial launch and report a readable error.
            self.page = None
            self.context = None
            self.browser = None
            if self._camoufox_cm is not None:
                try:
                    await self._camoufox_cm.__aexit__(None, None, None)
                except Exception:
                    # Cleanup after a failed launch is best-effort; the launch error is raised below.
                    pass
            self._camoufox_cm = None
            raise RuntimeError(
                "Browser launch failed. Check that Playwright browsers and Camoufox are installed "
                f"and that the current runtime is allowed to spawn browser subprocesses. Original error: {exc}"
            ) from exc

        # Save downloaded files into the active workspace directory.
        async def handle_download(download):
            log.info(f"Download started: {download.suggested_filename}")
            try:
                downloads_dir = self.downloads_dir()
                downloads_dir.mkdir(parents=True, exist_ok=True)
                file_path = downloads_dir / download.suggested_filename
                await download.save_as(file_path)
                log.info(f"File saved to: {file_path}")
            except Exception as exc:
                log.error(f"Download failed: {exc}")

        self.page.on("download", handle_download)

        # Accept browser dialogs with a deterministic response so automation does not hang.
        async def handle_dialog(dialog):
            log.info(f"Dialog: {dialog.type} - {dialog.message}")
            try:
                if dialog.type == "prompt":
                    await dialog.accept("accepted")
                else:
                    await dialog.accept()
            except Exception as exc:
                log.error(f"Dialog handling failed: {exc}")

        self.page.on("dialog", handle_dialog)
        log.info("Camoufox browser ready.")
        return True

    # Close all shared browser resources and clear stored references. Page, context, browser and the
    # Camoufox manager are closed in that order, each independently, so one failure does not skip the rest.
    async def close(self):
        # Detach the references first so the state is clean even if closing fails.
        page = self.page
        context = self.context
        browser = self.browser
        camoufox_cm = self._camoufox_cm
        self.page = None
        self.context = None
        self.browser = None
        self._camoufox_cm = None

        # Close from the innermost resource outwards.
        if page:
            try:
                if not page.is_closed():
                    await page.close()
            except Exception:
                # The browser may already be dead; shutdown must continue with the other resources.
                pass
        if context:
            try:
                await context.close()
            except Exception:
                # The browser may already be dead; shutdown must continue with the other resources.
                pass
        if browser:
            try:
                await browser.close()
            except Exception:
                # The browser may already be dead; shutdown must continue with the other resources.
                pass
        if camoufox_cm:
            try:
                await camoufox_cm.__aexit__(None, None, None)
            except Exception:
                # The browser may already be dead; there is nothing more to release.
                pass


# Shared runtime state: the browser session, the elements of the last snapshot (refs resolve against them),
# whether a wait_for_user call is pending, and the URLs visited so far.
state = BrowserState()
_last_elements: list[dict] = []
_waiting_for_user: bool = False
_nav_history: list[str] = []

# How often (at most) to capture the a11y bundle during active portal polling.
# The portal captures a JPEG on every publish; we only rebuild the a11y tree
# every PORTAL_A11Y_CAPTURE_EVERY publishes to avoid excessive Playwright calls.
PORTAL_A11Y_CAPTURE_EVERY = 4

# Compact a11y bundle for the live portal panel and its publish counter. The bundle is kept as the
# last-known value so the Django frame response always has something to return even between rebuilds.
_portal_a11y_bundle: dict | None = None
_portal_a11y_capture_counter: int = 0


# Build a compact a11y bundle for the live portal panel (throttled Playwright calls). Only every
# PORTAL_A11Y_CAPTURE_EVERY-th call rebuilds the bundle; the others return the last-known one (which is
# None before the first rebuild).
async def capture_portal_a11y_bundle(page: Any, *, max_controls: int = 60) -> dict | None:
    global _last_elements, _portal_a11y_bundle, _portal_a11y_capture_counter

    # Throttle: rebuild on the first call and then once per PORTAL_A11Y_CAPTURE_EVERY calls.
    _portal_a11y_capture_counter += 1
    if _portal_a11y_capture_counter % PORTAL_A11Y_CAPTURE_EVERY != 1:
        # Return the last-known bundle without re-querying Playwright.
        return _portal_a11y_bundle

    if page is None:
        return _portal_a11y_bundle

    try:
        elements, _ = await get_accessibility_tree(page, full=False)
    except Exception:
        # A failed snapshot keeps the last-known bundle rather than breaking the portal frame.
        return _portal_a11y_bundle

    # Sync the shared element cache so _click_by_role_and_name works with ref
    # IDs from this bundle (same as _take_compact_snapshot does).
    _last_elements = elements

    controls = _filter_snapshot_controls(elements, full=False)[:max_controls]
    items = [_snapshot_element_payload(el) for el in controls]

    # Page address and title are optional extras for the panel.
    try:
        url = page.url
        title = await page.title()
    except Exception:
        url = ""
        title = ""

    bundle = {
        "url": url,
        "title": title,
        "controls": items,
        "total_controls": len(controls),
        "truncated": len(controls) > max_controls,
    }
    _portal_a11y_bundle = bundle
    return bundle


# Reset portal a11y counters and bundle when a new portal session starts.
def reset_portal_a11y_state() -> None:
    global _portal_a11y_bundle, _portal_a11y_capture_counter
    _portal_a11y_bundle = None
    _portal_a11y_capture_counter = 0


# Snapshot builders

# Build the detailed page snapshot returned after navigation or major page changes. The default snapshot
# lists controls only; full mode also adds the page text and the accessibility tree. Side effects: the
# element cache used to resolve refs is replaced and the navigation trail is updated.
async def _take_snapshot(
    action_context: str | None = None,
    run_dismiss: bool = False,
    include_text: bool = True,
    full: bool = False,
) -> list[TextContent]:
    global _last_elements, _nav_history

    # Clear low-value overlays before collecting snapshot data.
    if run_dismiss:
        dismissed = await _auto_dismiss_overlays(state.page)
        if dismissed:
            log.info(f"Auto-dismissed overlays: {dismissed}")

    # Surface blockers that still remain after dismiss attempts.
    blocking = await _detect_undismissable_overlay(state.page)
    overlay_warning = None
    if blocking:
        log.warning(f"Undismissable overlay detected: {blocking}")
        overlay_warning = (
            f"⚠️ OVERLAY: the page has a {blocking} that could not be dismissed automatically. "
            f"Consider calling browser_wait_for_user so the user can close it manually."
        )

    # Refresh the in-memory element cache for later click and type actions.
    elements, tree_text = await get_accessibility_tree(state.page, full=full)
    _last_elements = elements

    # Read the page address and title and check for CAPTCHA, login, cookie and error-page situations.
    url = state.page.url
    title = await state.page.title()
    warnings = await _detect_page_situation(state.page)

    # Keep a short navigation trail (at most 20 entries) to help with backtracking.
    if not _nav_history or _nav_history[-1] != url:
        _nav_history.append(url)
        if len(_nav_history) > 20:
            _nav_history.pop(0)

    # Build the textual snapshot in model-friendly sections.
    parts = [
        "## Browser snapshot",
        f"**URL:** {url}",
        f"**Title:** {title}",
        f"**Mode:** {'full page' if full else 'controls only'}",
        "**Ref rule:** use refs from this snapshot only; after an action, use the returned snapshot refs.",
    ]
    if len(_nav_history) >= 2:
        parts.append(f"**Back URL:** {_nav_history[-2]}")
    if action_context:
        parts.append(f"**Action:** {action_context}")

    main_count = sum(1 for e in elements if e.get("landmark") == "main")
    controls = _filter_snapshot_controls(elements, full=full)
    parts.append(f"**Elements:** {len(elements)} total, {main_count} in main area, {len(controls)} controls")

    # Warnings come first, with the unremovable-overlay warning on top.
    all_warnings = list(warnings)
    if overlay_warning:
        all_warnings.insert(0, overlay_warning)

    if all_warnings:
        parts.append("\n### ⚠️ Page Situation")
        parts.extend(all_warnings)

    # Machine-readable counts, using a larger control cap for full snapshots.
    max_items = MAX_MAIN_INTERACTIVE * 3 if full else MAX_MAIN_INTERACTIVE
    parts.append("\n### Parsed state")
    parts.append("```json")
    parts.append(json.dumps(_build_parsed_state(elements, full=full, max_items=max_items), ensure_ascii=False, indent=2))
    parts.append("```")

    # The default snapshot stops after the grouped controls and a hint about the full mode.
    if not full:
        parts.extend(_format_parsed_controls(elements, full=full, max_items=max_items))
        parts.append(
            "\n### Hint\nDefault snapshot hides page text and low-value links. "
            "Call browser_snapshot(full=true) when you need full page text, navigation, footer, or hidden/missing controls."
        )
        return [TextContent(type="text", text="\n".join(parts))]

    # Full mode includes page text and the accessibility tree for broad orientation.
    if include_text:
        brief_text = await _extract_brief_text(state.page, max_chars=max(AUTO_TEXT_PREVIEW_LEN * 8, 12000))
        if brief_text and len(brief_text.strip()) > 30:
            parts.append(f"\n### Page text\n{brief_text}")
        else:
            parts.append("\n### Page text\n(no text content detected — page may still be loading)")

    parts.extend(_format_parsed_controls(elements, full=full, max_items=max_items))

    # Finish with the readable accessibility tree.
    if tree_text.strip():
        parts.append(f"\n### Accessibility tree\n```\n{tree_text}\n```")
    else:
        parts.append("\n### Accessibility tree\n(empty — page may not have rendered yet)")

    return [TextContent(type="text", text="\n".join(parts))]


# Build the smaller controls-only snapshot used after in-page actions. Like the detailed snapshot it
# refreshes the element cache and the navigation trail, but it skips overlay and situation detection.
async def _take_compact_snapshot(action_context: str | None = None) -> list[TextContent]:
    global _last_elements, _nav_history

    # Refresh the element cache and read the page address and title.
    elements, _ = await get_accessibility_tree(state.page)
    _last_elements = elements

    url = state.page.url
    title = await state.page.title()

    # Record the page in the navigation trail (at most 20 entries).
    if not _nav_history or _nav_history[-1] != url:
        _nav_history.append(url)
        if len(_nav_history) > 20:
            _nav_history.pop(0)

    # Keep compact snapshots focused on interactive controls.
    main_count = sum(1 for e in elements if e.get("landmark") == "main")
    controls = _filter_snapshot_controls(elements, full=False)
    parts = [
        "## Browser snapshot",
        f"**URL:** {url}",
        f"**Title:** {title}",
        "**Mode:** controls only",
        "**Ref rule:** use refs from this snapshot only; after an action, use the returned snapshot refs.",
    ]
    if action_context:
        parts.append(f"**Action:** {action_context}")
    parts.append(f"**Elements:** {len(elements)} total, {main_count} in main area, {len(controls)} controls")
    parts.append("\n### Parsed state")
    parts.append("```json")
    parts.append(json.dumps(_build_parsed_state(elements, full=False, max_items=MAX_MAIN_INTERACTIVE), ensure_ascii=False, indent=2))
    parts.append("```")
    parts.extend(_format_parsed_controls(elements, full=False, max_items=MAX_MAIN_INTERACTIVE))
    parts.append("\n### Hint\nCall browser_snapshot(full=true) if the needed text or control is not listed.")

    return [TextContent(type="text", text="\n".join(parts))]


# Element lookup and locator resolution

# Return the cached accessibility element matching the given ref.
def _find_element(ref: str) -> dict | None:
    for el in _last_elements:
        if el["ref"] == ref:
            return el
    return None


# Resolve the best Playwright locator for a role and accessible name pair. Try role lookup (exact name
# first, then substring), then label, then visible text; with several matches the first visible one among
# the first six wins. Return None when nothing matches.
async def _resolve_locator(role: str, name: str):
    # Prefer semantic role lookups because they match the accessibility tree.
    for exact in (True, False):
        try:
            loc = state.page.get_by_role(role, name=name, exact=exact)
            count = await loc.count()
            if count == 0:
                continue
            if count == 1:
                return loc
            for i in range(min(count, 6)):
                try:
                    if await loc.nth(i).is_visible():
                        return loc.nth(i)
                except Exception:
                    # This candidate went stale or detached; check the next one.
                    continue
            return loc.first
        except Exception:
            # This lookup strategy failed; fall through to the next one.
            continue

    if not name:
        return None

    # Labels are a strong fallback for form fields and controls.
    try:
        loc = state.page.get_by_label(name, exact=False)
        if await loc.count() > 0:
            return loc.first
    except Exception:
        # The label lookup failed; fall back to text matching.
        pass

    # Text matching is the broadest fallback when semantic lookup fails.
    for exact in (True, False):
        try:
            loc = state.page.get_by_text(name, exact=exact)
            count = await loc.count()
            if count == 0:
                continue
            for i in range(min(count, 6)):
                try:
                    if await loc.nth(i).is_visible():
                        return loc.nth(i)
                except Exception:
                    # This candidate went stale or detached; check the next one.
                    continue
        except Exception:
            # This lookup strategy failed; fall through to the next one.
            continue

    return None


# Click actions

# Click a cached accessibility element using role/name resolution and fallbacks. Strategies in order:
# normal click, check for checkable roles, forced click, DOM click. Raise ValueError when the ref is
# unknown, the element cannot be located, or every strategy fails.
async def _click_by_role_and_name(ref: str):
    elem = _find_element(ref)
    if not elem:
        raise ValueError(
            f"Element ref='{ref}' not found. "
            f"Use browser_snapshot to refresh elements."
        )

    role = elem["role"]
    name = elem.get("name", "")
    is_checkable = role in ("checkbox", "radio", "switch", "menuitemcheckbox", "menuitemradio")

    loc = await _resolve_locator(role, name)
    if loc is None:
        raise ValueError(
            f"Could not locate element ref='{ref}' (role={role}, name='{name}') "
            f"on the page. Try browser_snapshot to get fresh elements."
        )

    last_err: Exception | None = None

    # Try the normal interaction path first.
    try:
        await loc.click(timeout=5000)
        return
    except Exception as e:
        last_err = e

    # Some controls are better handled through Playwright's check helper.
    if is_checkable:
        try:
            await loc.check(force=True, timeout=5000)
            return
        except Exception as e:
            last_err = e

    # Force-click can still work when Playwright thinks the element is obscured.
    try:
        await loc.click(force=True, timeout=5000)
        return
    except Exception as e:
        last_err = e

    # Fall back to DOM click as a last resort.
    try:
        await loc.evaluate("el => el.click()")
        return
    except Exception as e:
        last_err = e

    raise ValueError(
        f"Could not click element ref='{ref}' (role={role}, name='{name}'). "
        f"All click strategies failed. Last error: {last_err}"
    )
