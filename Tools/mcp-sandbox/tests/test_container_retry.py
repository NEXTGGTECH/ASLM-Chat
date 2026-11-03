# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Package root of mcp-sandbox and its sandbox source directory; the latter is put on sys.path at import
# time so the sandbox package can be imported.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import os
import shutil
# Point the sandbox at a throwaway workspace at import time unless the environment already sets one.
os.environ.setdefault("SANDBOX_HOST_WORKSPACE", str(ROOT / ".test_workspace"))

import sandbox.docker_host as docker_host_mod  # noqa: E402

# Alias of the docker host module that the tests use when calling the container management functions.
container_mod = docker_host_mod


# Build a fake subprocess result with returncode 0.
def _ok(stdout: str = "", stderr: str = "") -> MagicMock:
    m = MagicMock()
    m.returncode = 0
    m.stdout = stdout
    m.stderr = stderr
    return m


# Build a fake subprocess result with non-zero returncode.
def _fail(stderr: str = "error", returncode: int = 1) -> MagicMock:
    m = MagicMock()
    m.returncode = returncode
    m.stdout = ""
    m.stderr = stderr
    return m


# Container inspection

# Cover _inspect_container: missing container, running container, volume mismatch and invalid output.
class TestInspectContainer(unittest.TestCase):
    # Verify that a failing docker inspect (no such container) yields None.
    def test_returns_none_when_container_missing(self) -> None:
        with patch.object(docker_host_mod, "_run_command", return_value=_fail("No such container", 1)):
            result = container_mod._inspect_container()
        self.assertIsNone(result)

    # Verify that a running container whose mount source matches the expected task directory is reported
    # as existing, running and with a correct volume.
    def test_returns_running_true_when_up(self) -> None:
        import json
        from sandbox.config import HOST_WORKSPACE, DEFAULT_TASK_DIR
        import os as os_mod
        mount = os_mod.path.normpath(os_mod.path.join(HOST_WORKSPACE, DEFAULT_TASK_DIR))
        payload = json.dumps([{
            "State": {"Running": True, "Status": "running"},
            "Mounts": [{"Source": mount}],
        }])
        with patch.object(docker_host_mod, "_run_command", return_value=_ok(payload)):
            result = container_mod._inspect_container()
        self.assertIsNotNone(result)
        self.assertTrue(result["running"])
        self.assertTrue(result["exists"])
        self.assertTrue(result["volume_ok"])

    # Verify that a running container mounting an unexpected source path is reported with volume_ok False.
    def test_detects_volume_mismatch(self) -> None:
        import json
        payload = json.dumps([{"State": {"Running": True, "Status": "running"}, "Mounts": [{"Source": "/wrong/path"}]}])
        with patch.object(docker_host_mod, "_run_command", return_value=_ok(payload)):
            result = container_mod._inspect_container()
        self.assertIsNotNone(result)
        self.assertFalse(result["volume_ok"], "Volume mismatch should be detected")

    # Verify that docker inspect output that is not valid JSON yields None.
    def test_returns_none_on_invalid_json(self) -> None:
        with patch.object(docker_host_mod, "_run_command", return_value=_ok("not json at all")):
            result = container_mod._inspect_container()
        self.assertIsNone(result)


# Forced container removal

# Cover _force_remove: success, an already removed container and a real docker error.
class TestForceRemove(unittest.TestCase):
    # Verify that a successful docker rm reports success.
    def test_success(self) -> None:
        with patch.object(docker_host_mod, "_run_command", return_value=_ok()):
            ok, msg = container_mod._force_remove()
        self.assertTrue(ok)

    # Verify that a "No such container" error is treated as success because the container is already gone.
    def test_already_gone_is_ok(self) -> None:
        with patch.object(docker_host_mod, "_run_command", return_value=_fail("No such container")):
            ok, msg = container_mod._force_remove()
        self.assertTrue(ok, "Already-gone container should be treated as success")

    # Verify that any other docker rm error (daemon not running) is reported as a failure.
    def test_real_error_returns_false(self) -> None:
        with patch.object(docker_host_mod, "_run_command", return_value=_fail("daemon not running")):
            ok, msg = container_mod._force_remove()
        self.assertFalse(ok)


# Starting an existing container

# Cover _start_existing: successful start, failing start and a container that exits right after starting.
class TestStartExisting(unittest.TestCase):
    # Verify that the start succeeds when docker start succeeds and the following inspect shows the
    # container running with the correct volume.
    def test_returns_true_when_container_running_after_start(self) -> None:
        import json, os as os_mod
        from sandbox.config import HOST_WORKSPACE, DEFAULT_TASK_DIR
        mount = os_mod.path.normpath(os_mod.path.join(HOST_WORKSPACE, DEFAULT_TASK_DIR))
        running_payload = json.dumps([{
            "State": {"Running": True, "Status": "running"},
            "Mounts": [{"Source": mount}],
        }])

        # docker start → OK; docker inspect → running
        with patch.object(docker_host_mod, "_run_command",
                          side_effect=[_ok(), _ok(running_payload)]):
            ok, msg = container_mod._start_existing()
        self.assertTrue(ok)

    # Verify that a failing docker start (permission denied) is reported as a failure.
    def test_returns_false_when_start_fails(self) -> None:
        with patch.object(docker_host_mod, "_run_command", return_value=_fail("permission denied")):
            ok, msg = container_mod._start_existing()
        self.assertFalse(ok)

    # Verify that the start fails when docker start returns 0 but the container is not running afterwards.
    def test_returns_false_when_not_running_after_start(self) -> None:
        import json

        # start OK, then inspect shows not running
        dead_payload = json.dumps([{"State": {"Running": False, "Status": "exited"}, "Mounts": []}])
        with patch.object(docker_host_mod, "_run_command",
                          side_effect=[_ok(), _ok(dead_payload)]):
            ok, msg = container_mod._start_existing()
        self.assertFalse(ok)


# Run command construction

# Cover _build_run_command: the task root mount, the dev-bind mounts of the supervisor source and the venv,
# and the rejection of a Windows venv.
class TestBuildRunCommand(unittest.TestCase):
    # Verify that the task root is mounted at the container workspace path, which is also the working
    # directory (-w), and that no --security-opt is added.
    def test_mounts_task_root_under_workspace_sandbox(self) -> None:
        from sandbox.config import DEFAULT_TASK_DIR, MODEL_WORKSPACE_CONTAINER

        cmd = container_mod._build_run_command("image:test", include_storage_limit=False)

        task_host_path = os.path.join(container_mod.HOST_WORKSPACE, DEFAULT_TASK_DIR)
        self.assertIn(f"{task_host_path}:{MODEL_WORKSPACE_CONTAINER}", cmd)
        self.assertIn("-w", cmd)
        self.assertEqual(cmd[cmd.index("-w") + 1], MODEL_WORKSPACE_CONTAINER)
        self.assertNotIn("--security-opt", cmd)

    # Verify that without SANDBOX_DEV_BIND=1 the supervisor source tree is not bind-mounted.
    def test_no_supervisor_source_mount_by_default(self) -> None:
        with patch.object(docker_host_mod, "DEV_BIND", False):
            cmd = container_mod._build_run_command("image:test", include_storage_limit=False)

        # The path appears as an -e env var, but must NOT appear as a -v bind mount.
        bind_entry = f":{container_mod.SUPERVISOR_SRC}:ro"
        self.assertNotIn(bind_entry, cmd)

    # Verify that with SANDBOX_DEV_BIND=1 the supervisor source tree is bind-mounted read-only.
    def test_mounts_supervisor_source_read_only_when_dev_bind_enabled(self) -> None:
        src_dir = ROOT / "src"
        with patch.object(docker_host_mod, "DEV_BIND", True), \
             patch.object(docker_host_mod, "SUPERVISOR_SRC_HOST", str(src_dir)):
            cmd = container_mod._build_run_command("image:test", include_storage_limit=False)
        self.assertIn(f"{src_dir}:{container_mod.SUPERVISOR_SRC}:ro", cmd)

    # Verify that with SANDBOX_DEV_BIND=1 a Linux-style venv is bind-mounted read-only.
    def test_adds_read_only_linux_venv_mount_when_dev_bind_enabled(self) -> None:
        temp_dir = ROOT / ".test_workspace" / "__venv_mount_test"
        shutil.rmtree(temp_dir, ignore_errors=True)
        try:
            # Build a fake Linux venv (bin/python and a pyvenv.cfg pointing at a Unix home).
            venv = temp_dir / "venv"
            (venv / "bin").mkdir(parents=True)
            (venv / "bin" / "python").write_text("#!/usr/bin/env python\n", encoding="utf-8")
            (venv / "pyvenv.cfg").write_text("home = /usr/local/bin\n", encoding="utf-8")

            with patch.object(docker_host_mod, "DEV_BIND", True), \
                 patch.object(docker_host_mod, "SUPERVISOR_VENV_HOST", str(venv)):
                cmd = container_mod._build_run_command("image:test", include_storage_limit=False)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertIn(f"{venv}:{container_mod.SUPERVISOR_VENV}:ro", cmd)

    # Verify that without SANDBOX_DEV_BIND=1 a Linux-style venv is not bind-mounted.
    def test_no_venv_mount_without_dev_bind(self) -> None:
        temp_dir = ROOT / ".test_workspace" / "__venv_mount_test2"
        shutil.rmtree(temp_dir, ignore_errors=True)
        try:
            # Build a fake Linux venv (bin/python and a pyvenv.cfg pointing at a Unix home).
            venv = temp_dir / "venv"
            (venv / "bin").mkdir(parents=True)
            (venv / "bin" / "python").write_text("#!/usr/bin/env python\n", encoding="utf-8")
            (venv / "pyvenv.cfg").write_text("home = /usr/local/bin\n", encoding="utf-8")

            with patch.object(docker_host_mod, "DEV_BIND", False), \
                 patch.object(docker_host_mod, "SUPERVISOR_VENV_HOST", str(venv)):
                cmd = container_mod._build_run_command("image:test", include_storage_limit=False)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        self.assertNotIn(f"{venv}:{container_mod.SUPERVISOR_VENV}:ro", cmd)

    # Verify that building the run command raises ValueError when the configured venv is a Windows venv
    # (Scripts/python.exe, Windows home in pyvenv.cfg).
    def test_rejects_windows_venv_mount(self) -> None:
        temp_dir = ROOT / ".test_workspace" / "__venv_windows_test"
        shutil.rmtree(temp_dir, ignore_errors=True)
        try:
            # Build a fake Windows venv (Scripts/python.exe and a pyvenv.cfg pointing at a Windows home).
            venv = temp_dir / "venv"
            (venv / "Scripts").mkdir(parents=True)
            (venv / "Scripts" / "python.exe").write_text("", encoding="utf-8")
            (venv / "pyvenv.cfg").write_text("home = C:\\Python\n", encoding="utf-8")

            with patch.object(docker_host_mod, "SUPERVISOR_VENV_HOST", str(venv)):
                with self.assertRaises(ValueError):
                    container_mod._build_run_command("image:test", include_storage_limit=False)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# Image preparation

# Cover _ensure_image: reuse of a current image and rebuild of a stale one.
class TestEnsureImage(unittest.TestCase):
    # Verify that an existing image carrying the required runtime label is reused with a single docker
    # command and no rebuild.
    def test_existing_image_with_runtime_label_is_reused(self) -> None:
        import json

        payload = json.dumps([
            {
                "Config": {
                    "Labels": {
                        container_mod.REQUIRED_IMAGE_LABEL: container_mod.REQUIRED_IMAGE_LABEL_VALUE,
                    }
                }
            }
        ])

        with patch.object(docker_host_mod, "_run_command", return_value=_ok(payload)) as run_mock:
            ok, message = container_mod._ensure_image()

        self.assertTrue(ok)
        self.assertIn("already exists", message)
        run_mock.assert_called_once()

    # Verify that an existing image without the runtime label is considered stale and rebuilt exactly once
    # through the setup subprocess.
    def test_existing_stale_image_without_runtime_label_is_rebuilt(self) -> None:
        import json

        stale_payload = json.dumps([{"Config": {"Labels": {}}}])
        setup_result = MagicMock(returncode=0)

        # Answer docker image inspect with the label-less image and fail any other docker command.
        def fake_run(args, timeout=30, **kwargs):
            if args[:3] == ["docker", "image", "inspect"]:
                return _ok(stale_payload)
            return _fail(f"unexpected command: {args}")

        with patch.object(docker_host_mod, "_run_command", side_effect=fake_run), \
             patch.object(docker_host_mod.subprocess, "run", return_value=setup_result) as setup_mock:
            ok, message = container_mod._ensure_image()

        self.assertTrue(ok)
        self.assertIn("ready", message)
        setup_mock.assert_called_once()


# Ensuring a running container

# Cover _ensure_container_running: reuse, recreation on a volume mismatch, retry on a name conflict and
# serialization of concurrent calls.
class TestEnsureContainerRunning(unittest.TestCase):
    # Build a mock for _ensure_docker_running that reports a running Docker daemon.
    def _mock_docker_ok(self) -> MagicMock:
        m = MagicMock(return_value=(True, "Docker daemon is running."))
        return m

    # Verify that a container that is already running with the correct volume is reused and reported as running.
    def test_returns_true_when_already_running_correct_volume(self) -> None:
        import json, os as os_mod
        from sandbox.config import HOST_WORKSPACE, DEFAULT_TASK_DIR
        mount = os_mod.path.normpath(os_mod.path.join(HOST_WORKSPACE, DEFAULT_TASK_DIR))
        payload = json.dumps([{
            "State": {"Running": True, "Status": "running"},
            "Mounts": [{"Source": mount}],
        }])

        with patch.object(docker_host_mod, "_ensure_docker_running",
                          return_value=(True, "OK")), \
             patch.object(docker_host_mod, "_run_command", return_value=_ok(payload)):
            ok, msg = container_mod._ensure_container_running()

        self.assertTrue(ok)
        self.assertIn("running", msg.lower())

    # Verify that a running container with a wrong volume is removed and a new one is created, ending in
    # success.
    def test_recreates_container_on_volume_mismatch(self) -> None:
        import json

        # First inspect: running but wrong volume
        bad_payload = json.dumps([{"State": {"Running": True, "Status": "running"}, "Mounts": [{"Source": "/wrong"}]}])

        # After rm, no container; after docker run, OK
        call_count = {"n": 0}
        payloads = [_ok(bad_payload), _ok(), _fail("No such container", 1), _ok()]

        # Return the scripted docker results in call order, then success for any further call.
        def fake_run(args, timeout=10, **kw):
            idx = call_count["n"]
            call_count["n"] += 1
            if idx < len(payloads):
                return payloads[idx]
            return _ok()

        with patch.object(docker_host_mod, "_ensure_docker_running", return_value=(True, "OK")), \
             patch.object(docker_host_mod, "_ensure_image", return_value=(True, "OK")), \
             patch.object(docker_host_mod, "_wait_for_container_running", return_value=(True, "running")), \
             patch.object(docker_host_mod, "_run_command", side_effect=fake_run):
            ok, msg = container_mod._ensure_container_running()

        self.assertTrue(ok)

    # Verify that a "name already in use" error while creating the container triggers a retry that ends in
    # success once the container is found running.
    def test_retries_on_name_conflict(self) -> None:
        import json, os as os_mod
        from sandbox.config import HOST_WORKSPACE, DEFAULT_TASK_DIR
        mount = os_mod.path.normpath(os_mod.path.join(HOST_WORKSPACE, DEFAULT_TASK_DIR))
        running_payload = json.dumps([{
            "State": {"Running": True, "Status": "running"},
            "Mounts": [{"Source": mount}],
        }])

        call_count = {"n": 0}

        # Script the docker calls by position: missing container, create conflict, then a running container.
        def fake_run(args, timeout=10, **kw):
            n = call_count["n"]
            call_count["n"] += 1

            # First inspect: no container.
            if "inspect" in args and n == 0:
                return _fail("No such object", 1)

            # Create fails with a name conflict.
            if "run" in args and n == 1:
                return _fail("name already in use", 125)

            # Second inspect: the container is now running.
            if "inspect" in args and n == 2:
                return _ok(running_payload)
            return _ok()

        with patch.object(docker_host_mod, "_ensure_docker_running", return_value=(True, "OK")), \
             patch.object(docker_host_mod, "_ensure_image", return_value=(True, "OK")), \
             patch.object(docker_host_mod, "_wait_for_container_running", return_value=(True, "running")), \
             patch.object(docker_host_mod, "_run_command", side_effect=fake_run):
            ok, msg = container_mod._ensure_container_running(retry_delay=0)

        self.assertTrue(ok, f"Expected success after retry, got: {msg}")

    # Verify that four concurrent _ensure_container_running calls all succeed without errors and never run
    # _inspect_container at the same time (the production lock serializes them).
    def test_concurrent_calls_serialized(self) -> None:
        results: list[tuple[bool, str]] = []
        errors: list[Exception] = []
        active_inspects = 0
        max_active_inspects = 0
        active_lock = threading.Lock()

        # Report a healthy container while tracking how many inspect calls overlap.
        def fake_inspect():
            nonlocal active_inspects, max_active_inspects
            with active_lock:
                active_inspects += 1
                max_active_inspects = max(max_active_inspects, active_inspects)
            try:
                # Give other threads a chance to enter if the production lock is broken.
                import time
                time.sleep(0.01)
                return {
                    "exists": True,
                    "running": True,
                    "status": "running",
                    "volume_ok": True,
                }
            finally:
                with active_lock:
                    active_inspects -= 1

        # Call _ensure_container_running once and collect its result or the exception it raised.
        def worker():
            try:
                r = container_mod._ensure_container_running()
                results.append(r)
            except Exception as e:
                errors.append(e)

        with patch.object(docker_host_mod, "_ensure_docker_running", return_value=(True, "OK")), \
             patch.object(docker_host_mod, "_inspect_container", side_effect=fake_inspect):
            threads = [threading.Thread(target=worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join(timeout=5)

        self.assertFalse(errors, f"Concurrent calls raised exceptions: {errors}")
        self.assertEqual(len(results), 4)
        self.assertTrue(all(ok for ok, _ in results), f"Some calls failed: {results}")
        self.assertEqual(max_active_inspects, 1, "_ensure_container_running did not serialize inspect calls")


# Allow running this file directly with the unittest runner.
if __name__ == "__main__":
    unittest.main(verbosity=2)
