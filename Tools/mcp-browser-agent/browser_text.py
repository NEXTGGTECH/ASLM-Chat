# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from typing import Any, Callable

# In-page script behind browser_text. It finds the target editor (the supplied element, else the focused
# element, else the first visible editor or text input), wraps CodeMirror, Ace, plain fields and
# contenteditable elements in a common get/set adapter, optionally writes payload.text when
# payload.mode is "set", and returns the editor kind, value, length, line count and label.
TEXT_EDITOR_SCRIPT = r"""(payload) => {
    const supplied = payload && payload.element ? payload.element : null;

    const visible = el => {
        if (!el) return false;
        const style = window.getComputedStyle(el);
        const rect = el.getBoundingClientRect();
        return style.display !== 'none' && style.visibility !== 'hidden' && rect.width > 0 && rect.height > 0;
    };

    const editorRoot = el => {
        if (!el) return null;
        return el.closest('.CodeMirror, .ace_editor, textarea, input, [contenteditable="true"]') ||
            el.querySelector?.('.CodeMirror, .ace_editor, textarea, input, [contenteditable="true"]') ||
            el;
    };

    let target = editorRoot(supplied);
    if (!target || !visible(target)) {
        const active = editorRoot(document.activeElement);
        if (active && visible(active)) target = active;
    }
    if (!target || !visible(target)) {
        target = Array.from(document.querySelectorAll(
            '.CodeMirror, .ace_editor, textarea, input[type="text"], input[type="search"], input[type="email"], input[type="url"], input[type="tel"], input[type="password"], [contenteditable="true"]'
        )).find(el => visible(el));
    }
    if (!target) {
        return { ok: false, error: 'No visible editor or text input found.' };
    }

    const fire = el => {
        el.dispatchEvent(new Event('input', { bubbles: true }));
        el.dispatchEvent(new Event('change', { bubbles: true }));
    };

    const getAdapter = el => {
        const cmRoot = el.classList?.contains('CodeMirror') ? el : el.querySelector?.('.CodeMirror');
        if (cmRoot && cmRoot.CodeMirror) {
            return {
                element: cmRoot,
                kind: 'codemirror',
                get: () => cmRoot.CodeMirror.getValue(),
                set: value => {
                    cmRoot.CodeMirror.setValue(value);
                    cmRoot.CodeMirror.focus();
                    cmRoot.CodeMirror.refresh();
                },
            };
        }

        const aceRoot = el.classList?.contains('ace_editor') ? el : el.querySelector?.('.ace_editor');
        if (aceRoot && aceRoot.env && aceRoot.env.editor) {
            return {
                element: aceRoot,
                kind: 'ace',
                get: () => aceRoot.env.editor.getValue(),
                set: value => {
                    aceRoot.env.editor.setValue(value, -1);
                    aceRoot.env.editor.focus();
                },
            };
        }

        const field = ['TEXTAREA', 'INPUT'].includes(el.tagName)
            ? el
            : el.querySelector?.('textarea, input');
        if (field) {
            return {
                element: field,
                kind: field.tagName.toLowerCase(),
                get: () => field.value || '',
                set: value => {
                    field.focus();
                    field.value = value;
                    fire(field);
                },
            };
        }

        const editable = el.isContentEditable ? el : el.querySelector?.('[contenteditable="true"]');
        if (editable) {
            return {
                element: editable,
                kind: 'contenteditable',
                get: () => editable.innerText || '',
                set: value => {
                    editable.focus();
                    editable.innerText = value;
                    fire(editable);
                },
            };
        }

        return {
            element: el,
            kind: 'unknown',
            get: () => el.innerText || '',
            set: value => {
                el.focus?.();
                el.innerText = value;
                fire(el);
            },
        };
    };

    const adapter = getAdapter(target);
    if (payload.mode === 'set') {
        adapter.set(String(payload.text || ''));
    }
    const value = adapter.get() || '';
    const label = adapter.element.getAttribute?.('aria-label') ||
        adapter.element.getAttribute?.('placeholder') ||
        adapter.element.getAttribute?.('name') ||
        adapter.element.id ||
        '';
        const editable = ['codemirror', 'ace', 'textarea', 'input', 'contenteditable'].includes(adapter.kind);
        return {
            ok: true,
            kind: adapter.kind,
            editable,
            value,
            length: value.length,
            line_count: value ? value.split(/\r\n|\r|\n/).length : 0,
        label,
    };
}"""


# Parse a 1-based line range string into start, end, and insert-mode flag. The range is "N" (one line) or
# "N:M" (inclusive). An end before the start (for example "5:4") means an insert before line 5, allowed up
# to one line past the end of the text. Raise ValueError for an empty, non-positive or out-of-text range.
def _parse_line_range(raw_range: Any, total_lines: int) -> tuple[int, int, bool]:
    raw = str(raw_range or "").strip()
    if not raw:
        raise ValueError("range is required.")
    left, right = raw.split(":", 1) if ":" in raw else (raw, raw)
    start = int(left)
    end = int(right)
    if start < 1 or end < 0:
        raise ValueError("range must use positive 1-based line numbers.")
    if end < start:
        if start > total_lines + 1:
            raise ValueError(f"insert start line {start} is beyond end of text.")
        return start, end, True
    if start > total_lines or end > total_lines:
        raise ValueError(f"range {start}:{end} is outside text with {total_lines} lines.")
    return start, end, False


# Replace or insert lines in editor text using a 1-based line range. Line endings are normalized to "\n"
# and a trailing newline of the original text is preserved.
def replace_line_range(current: str, raw_range: Any, replacement: str) -> str:
    # Split the current text into lines, ignoring the empty element after a trailing newline.
    normalized = current.replace("\r\n", "\n").replace("\r", "\n")
    had_trailing_newline = normalized.endswith("\n")
    lines = [] if normalized == "" else normalized.split("\n")
    if had_trailing_newline:
        lines = lines[:-1]

    # Validate the range against the text, then split the replacement into lines the same way.
    start, end, is_insert = _parse_line_range(raw_range, len(lines))
    replacement_normalized = str(replacement or "").replace("\r\n", "\n").replace("\r", "\n")
    replacement_lines = [] if replacement_normalized == "" else replacement_normalized.split("\n")
    if replacement_lines and replacement_normalized.endswith("\n"):
        replacement_lines = replacement_lines[:-1]

    # Insert before the start line, or replace the lines start..end.
    if is_insert:
        updated = lines[: start - 1] + replacement_lines + lines[start - 1 :]
    else:
        updated = lines[: start - 1] + replacement_lines + lines[end:]

    result = "\n".join(updated)
    if had_trailing_newline and result:
        result += "\n"
    return result


# Read or set text in the focused editor or in the element identified by ref. Without a ref the editor
# script finds its own target; with a ref the cached element is resolved to a locator and passed to the
# script. Return the script's result dict; raise ValueError when the ref is unknown or cannot be located.
async def _read_or_set_text_state(page: Any, ref: str, mode: str, text: str) -> dict[str, Any]:
    payload = {"mode": mode, "text": text}
    if not ref:
        return await page.evaluate(TEXT_EDITOR_SCRIPT, payload)

    # Imported lazily so that loading this module does not import the browser module (Camoufox, Playwright).
    from browser import _find_element, _resolve_locator

    elem = _find_element(ref)
    if not elem:
        raise ValueError(f"Element ref='{ref}' not found. Use browser_snapshot to refresh elements.")
    locator = await _resolve_text_locator(page, elem)
    if locator is None:
        raise ValueError(f"Could not locate element ref='{ref}'. Use browser_snapshot to refresh.")

    wrapper = "(el, payload) => { payload.element = el; return (" + TEXT_EDITOR_SCRIPT + ")(payload); }"
    return await locator.evaluate(wrapper, payload)


# Return the first visible locator match (among the first eight) from a multi-match locator, or None.
async def _first_visible(locator: Any) -> Any | None:
    try:
        count = await locator.count()
    except Exception:
        # A locator that cannot be counted (detached page, bad selector) simply has no match.
        return None
    for index in range(min(count, 8)):
        candidate = locator.nth(index)
        try:
            if await candidate.is_visible(timeout=100):
                return candidate
        except Exception:
            # This candidate went stale or detached; check the next one.
            continue
    return None


# Resolve text refs using exact DOM attributes before broad role fallback.
async def _resolve_text_locator(page: Any, elem: dict[str, Any]) -> Any | None:
    role = str(elem.get("role") or "")
    name = str(elem.get("name") or "")

    # An element's accessible name often comes from aria-label, placeholder, name or title, so match
    # those attributes exactly first.
    if name:
        for selector in (
            f"[aria-label={_css_string(name)}]",
            f"[placeholder={_css_string(name)}]",
            f"[name={_css_string(name)}]",
            f"[title={_css_string(name)}]",
        ):
            found = await _first_visible(page.locator(selector))
            if found is not None:
                return found

    # Fall back to the shared role/label/text resolution (lazy import, as in _read_or_set_text_state).
    from browser import _resolve_locator

    return await _resolve_locator(role, name)


# Escape a string for use inside a CSS attribute selector.
def _css_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'


# Infer read/set/replace/delete from explicit action or argument shape: old_text/new_text means replace,
# text means set, all or range means delete, and anything else is a read.
def _infer_action(args: dict[str, Any]) -> str:
    raw_action = str(args.get("action", "") or "").strip().lower()
    if raw_action:
        return raw_action
    if "old_text" in args or "new_text" in args:
        return "replace"
    if "text" in args:
        return "set"
    if bool(args.get("all", False)) or args.get("range"):
        return "delete"
    return "read"


# Build a user-facing error when the text target is not editable.
def _non_editable_error(ref: str, target: dict[str, Any], action: str) -> str:
    kind = str(target.get("kind") or "unknown")
    label = str(target.get("label") or "").strip()
    target_name = f" ({label})" if label else ""
    if action == "read":
        return f"Error: text target is not editable: {kind}{target_name}. Click it first if it opens a search/input dialog, then call browser_snapshot."
    return (
        f"Error: cannot write text into non-editable target: {kind}{target_name}. "
        "If this is a search button or menu trigger, call browser_click on it, then browser_snapshot, then browser_text on the input ref."
    )


# Compute the next editor value for set, replace, or delete actions. Replace and delete target either a
# line range or an old_text match; delete can also clear everything with all=true. Raise ValueError for
# missing arguments or an unknown action.
def _next_text(action: str, args: dict[str, Any], current: str) -> str:
    if action == "set":
        return str(args.get("text", ""))

    if action == "replace":
        if args.get("range"):
            return replace_line_range(current, args.get("range"), str(args.get("text", "")))
        old_text = str(args.get("old_text", ""))
        if old_text == "":
            raise ValueError("old_text or range is required for action='replace'.")
        replacement = str(args.get("new_text", args.get("text", "")))
        return _replace_match(current, old_text, replacement, bool(args.get("replace_all", False)))

    if action == "delete":
        if bool(args.get("all", False)):
            return ""
        if args.get("range"):
            return replace_line_range(current, args.get("range"), "")
        old_text = str(args.get("old_text", ""))
        if old_text == "":
            raise ValueError("old_text, range, or all=true is required for action='delete'.")
        return _replace_match(current, old_text, "", bool(args.get("replace_all", False)))

    raise ValueError("action must be one of read, set, replace, delete.")


# Replace one or all occurrences of old_text, with ambiguity checks: raise ValueError when old_text is
# missing, or occurs several times without replace_all.
def _replace_match(current: str, old_text: str, replacement: str, replace_all: bool) -> str:
    match_count = current.count(old_text)
    if match_count == 0:
        raise ValueError("old_text not found in editor.")
    if match_count > 1 and not replace_all:
        raise ValueError(f"old_text found {match_count} times. Use replace_all=true or provide more context.")
    limit = match_count if replace_all else 1
    return current.replace(old_text, replacement, limit)


# Execute browser_text: read, set, replace, or delete editor content and return a snapshot. A read returns
# the editor text (first 4000 characters); the other actions write the new value, optionally press Enter,
# and return a status header plus a fresh snapshot. Problems are returned as "Error: ..." strings.
async def handle_browser_text(
    args: dict[str, Any],
    *,
    page: Any,
    keyboard: Any,
    take_snapshot: Callable[..., Any],
    flatten_content: Callable[[Any], str],
) -> str:
    action = _infer_action(args)
    ref = str(args.get("ref", "") or "").strip()

    # Read the current editor state and refuse targets that cannot be edited.
    before = await _read_or_set_text_state(page, ref, "read", "")
    if not before.get("ok"):
        return f"Error: {before.get('error', 'Could not read editor text.')}"
    if not bool(before.get("editable", False)):
        return _non_editable_error(ref, before, action)

    # A read action returns a bounded preview of the text.
    current = str(before.get("value", ""))
    if action == "read":
        preview = current if len(current) <= 4000 else current[:4000] + "\n[...]"
        label = f" ({before.get('label')})" if before.get("label") else ""
        return (
            f"Text target: {before.get('kind', 'editor')}{label}\n"
            f"Length: {len(current)} chars, {before.get('line_count', 0)} lines\n\n"
            f"{preview}"
        )

    # Compute the new text, write it, and read back the editor value to verify the write.
    try:
        next_value = _next_text(action, args, current)
    except ValueError as exc:
        return f"Error: {exc}"

    after = await _read_or_set_text_state(page, ref, "set", next_value)
    if not after.get("ok"):
        return f"Error: {after.get('error', 'Could not write editor text.')}"

    # Optionally submit with Enter, then give the page a moment to react.
    actual = str(after.get("value", ""))
    verified = actual == next_value
    if bool(args.get("press_enter", False)):
        await keyboard.press("Enter")
        await page.wait_for_timeout(500)
    else:
        await page.wait_for_timeout(400)

    # Report the outcome followed by a fresh snapshot of the page.
    snapshot = flatten_content(
        await take_snapshot(
            action_context=(
                f"Text {action} on {ref or 'focused/visible editor'} "
                f"({'verified' if verified else 'not fully verified'})"
            )
        )
    )
    status = (
        f"Text {action}: {'ok' if verified else 'warning: final value differs'}\n"
        f"Target: {after.get('kind', before.get('kind', 'editor'))}\n"
        f"Before: {len(current)} chars\n"
        f"After: {len(actual)} chars, {after.get('line_count', 0)} lines\n\n"
    )
    return status + snapshot
