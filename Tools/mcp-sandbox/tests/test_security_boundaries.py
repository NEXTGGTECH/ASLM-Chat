# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from pathlib import Path

import pytest

from sandbox import config
from sandbox import workspace


# Redirect the workspace module to a temporary host workspace.
@pytest.fixture
def isolated_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "sandbox-workspace"
    (root / config.DEFAULT_TASK_DIR).mkdir(parents=True)
    monkeypatch.setattr(workspace, "HOST_WORKSPACE", root)
    return root


# Path validation

# Verify that absolute, drive-qualified and UNC-style paths are rejected with ValueError.
@pytest.mark.parametrize(
    "path",
    [
        "/etc/passwd",
        "C:\\Windows\\system32",
        "C:/Users/dimap",
        "//server/share",
    ],
)
def test_validate_model_path_rejects_absolute_paths(path: str) -> None:
    with pytest.raises(ValueError):
        workspace.validate_model_path(path)


# Verify that a path starting with "task/" is accepted as an ordinary relative directory.
def test_task_prefix_is_an_ordinary_relative_directory() -> None:
    workspace.validate_model_path("task/secret.txt")


# Verify that Windows separators in a relative path are normalized to POSIX separators.
def test_task_alias_is_normalized_to_posix_relative_path() -> None:
    assert workspace.normalize_model_relative_path(r"task\pages\example.json") == "task/pages/example.json"


# Traversal confinement

# Verify that paths with traversal sequences are rejected (ValueError or OSError) instead of resolving
# outside the task root.
@pytest.mark.parametrize(
    "path",
    [
        "../secret",
        "../../etc/passwd",
        "../../../Windows/System32/cmd.exe",
        "a/../../secret",
        "a/../../../b",
        ".//../secret",
        "..\\/etc/passwd",
    ],
)
def test_get_secure_task_path_blocks_traversal(path: str, isolated_workspace: Path) -> None:
    with pytest.raises((ValueError, OSError)):
        workspace.get_secure_task_path(path)


# Verify that a path containing a null byte is rejected.
def test_null_byte_in_path_is_rejected(isolated_workspace: Path) -> None:
    with pytest.raises((ValueError, OSError)):
        workspace.get_secure_task_path("file\x00../etc/passwd")


# Verify that "......" (six dots) is a filename, not a traversal sequence: whichever way the normalizer
# treats it, the resolved path stays inside the task root.
def test_six_dot_name_stays_inside_task_root(isolated_workspace: Path) -> None:
    resolved = workspace.get_secure_task_path("......")

    assert resolved.is_relative_to(isolated_workspace / config.DEFAULT_TASK_DIR)


# Verify that a percent-encoded traversal is not decoded: the filesystem does not decode %2F, so the path
# resolves to a literal filename inside the task root.
def test_percent_encoded_name_is_a_literal_filename(isolated_workspace: Path) -> None:
    resolved = workspace.get_secure_task_path("..%2Fetc%2Fpasswd")

    assert resolved == isolated_workspace / config.DEFAULT_TASK_DIR / "..%2Fetc%2Fpasswd"


# Verify that a 4000-character path name still resolves inside the task root.
def test_very_long_path_stays_inside_task_root(isolated_workspace: Path) -> None:
    resolved = workspace.get_secure_task_path("a" * 4000)

    assert resolved.is_relative_to(isolated_workspace / config.DEFAULT_TASK_DIR)


# Workspace writes

# Verify that writes whose path escapes the task root are rejected (ValueError or OSError).
@pytest.mark.parametrize(
    "path",
    [
        "../escape.txt",
        "../../escape.txt",
        "sub/../../escape.txt",
    ],
)
def test_write_rejects_escape_outside_task_root(path: str, isolated_workspace: Path) -> None:
    with pytest.raises((ValueError, OSError)):
        workspace.write(path, "pwned")


# Verify that writing a bare filename creates the file inside the task root and reports the written bytes
# and that it was created.
def test_write_to_bare_filename_lands_inside_task_root(isolated_workspace: Path) -> None:
    result = workspace.write("canary_test_delete_me.txt", "ok")

    canary = isolated_workspace / config.DEFAULT_TASK_DIR / "canary_test_delete_me.txt"
    try:
        assert canary.is_file()
        assert result["result"]["bytes_written"] == 2
        assert result["result"]["created"] is True
    finally:
        canary.unlink(missing_ok=True)


# Host imports

# Verify that host system paths (Windows System32 and /etc) are denied for import without raising.
@pytest.mark.parametrize(
    "path",
    [
        Path("C:/Windows/System32/cmd.exe"),
        Path("/etc/passwd"),
    ],
)
def test_is_allowed_host_import_denies_system_paths(path: Path) -> None:
    assert workspace.is_allowed_host_import(path) is False


# Verify that user profile secrets (~/.ssh keys and files on the desktop) are denied for import without
# raising.
def test_is_allowed_host_import_denies_user_profile_secrets() -> None:
    assert workspace.is_allowed_host_import(Path.home() / ".ssh" / "id_rsa") is False
    assert workspace.is_allowed_host_import(Path.home() / "Desktop" / "secret.txt") is False


# Shared file descriptions

# Verify that describe() rejects share paths shaped like traversals (ValueError or OSError).
@pytest.mark.parametrize(
    "path",
    [
        "../secret.txt",
        "../../outside.txt",
        "sub/../../escape.txt",
    ],
)
def test_describe_blocks_traversal(path: str, isolated_workspace: Path) -> None:
    with pytest.raises((ValueError, OSError)):
        workspace.describe(path)
