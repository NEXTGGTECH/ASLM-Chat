# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sandbox.config import MAX_OUTPUT_BYTES, OUTPUT_HEAD_RATIO
from sandbox.responses import SandboxToolError


# Prefix of every background job id.
JOB_ID_PREFIX = "bg_"


# Record for a single background bash job (native or docker runtime). runtime is "native" or "docker",
# started_at is a wall-clock timestamp, process is the live Popen handle of a native job, and the two
# job directories hold the spooled output on the host and inside the container. status goes from
# "running" to "done" or "killed"; the offsets count the output bytes already returned by incremental
# reads.
@dataclass
class BackgroundJob:
    job_id: str
    command: str
    cwd: str
    runtime: str
    started_at: float
    pid: int | None = None
    process: Any | None = None
    host_job_dir: Path | None = None
    container_job_dir: str | None = None
    status: str = "running"
    exit_code: int | None = None
    stdout_offset: int = 0
    stderr_offset: int = 0

    # Return the job as a JSON-friendly dict, with the elapsed time computed at call time.
    def to_result(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "command": self.command,
            "cwd": self.cwd,
            "runtime": self.runtime,
            "pid": self.pid,
            "status": self.status,
            "exit_code": self.exit_code,
            "elapsed_s": round(time.time() - self.started_at, 3),
            "host_job_dir": str(self.host_job_dir) if self.host_job_dir else None,
            "container_job_dir": self.container_job_dir,
        }


# Thread-safe registry of background jobs and bounded incremental output reads.
class JobRegistry:
    # Start with no jobs; the lock guards the job table against concurrent tool calls.
    def __init__(self) -> None:
        self._jobs: dict[str, BackgroundJob] = {}
        self._lock = threading.Lock()

    # Job table

    # Register a new job and return it. runtime must be "native" or "docker"; a job id is generated
    # when none is given.
    def create(
        self,
        *,
        command: str,
        cwd: str,
        runtime: str,
        pid: int | None = None,
        host_job_dir: Path | str | None = None,
        container_job_dir: str | None = None,
        process: Any | None = None,
        job_id: str | None = None,
    ) -> BackgroundJob:
        if runtime not in {"native", "docker"}:
            raise ValueError("runtime must be 'native' or 'docker'.")

        # Build the record; the id is generated before the lock is taken because the generator locks too.
        normalized_id = job_id or self._new_job_id()
        job = BackgroundJob(
            job_id=normalized_id,
            command=command,
            cwd=cwd,
            runtime=runtime,
            started_at=time.time(),
            pid=pid,
            process=process,
            host_job_dir=Path(host_job_dir) if host_job_dir is not None else None,
            container_job_dir=container_job_dir,
        )

        # Publish the job in the registry.
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    # Return the job with this id, or raise SandboxToolError("job_not_found").
    def get(self, job_id: str) -> BackgroundJob:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise SandboxToolError("job_not_found", f"Background job not found: {job_id}")
        return job

    # Return every known job as a result dict.
    def list_jobs(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        return [job.to_result() for job in jobs]

    # Drop a job from the registry and return it (None if unknown); with cleanup=True its host job
    # directory is deleted as well.
    def remove(self, job_id: str, *, cleanup: bool = False) -> BackgroundJob | None:
        import shutil

        with self._lock:
            job = self._jobs.pop(job_id, None)
        if cleanup and job and job.host_job_dir and job.host_job_dir.exists():
            shutil.rmtree(job.host_job_dir, ignore_errors=True)
        return job

    # Output reading

    # Read a slice of a spool file with head/tail truncation when over MAX_OUTPUT_BYTES. Return the text
    # and the file size, which becomes the next incremental read offset.
    def _read_bounded_text(self, path: Path, *, start: int = 0) -> tuple[str, int]:
        size = path.stat().st_size
        start = max(0, min(start, size))
        remaining = size - start

        # Small enough: return everything from the offset.
        if remaining <= MAX_OUTPUT_BYTES:
            with path.open("rb") as fh:
                fh.seek(start)
                return fh.read().decode("utf-8", errors="replace").replace("\r\n", "\n"), size

        # Too large: keep the first head bytes after the offset and the last tail bytes of the file,
        # separated by a marker that reports the cut.
        head_ratio = min(0.9, max(0.1, OUTPUT_HEAD_RATIO))
        head_bytes = max(1, int(MAX_OUTPUT_BYTES * head_ratio))
        tail_bytes = max(1, MAX_OUTPUT_BYTES - head_bytes)
        with path.open("rb") as fh:
            fh.seek(start)
            head = fh.read(head_bytes)
            fh.seek(max(0, size - tail_bytes))
            tail = fh.read(tail_bytes)
        marker = (
            "\n\n"
            f"[output truncated while reading job spool: showed first {head_bytes} bytes "
            f"and last {tail_bytes} bytes of {remaining} new bytes]\n\n"
        )
        return (
            head.decode("utf-8", errors="replace")
            + marker
            + tail.decode("utf-8", errors="replace")
        ).replace("\r\n", "\n"), size

    # Read the spooled stdout or stderr of a job. With incremental=True only the output written since
    # the previous read is returned and the job's offset advances; otherwise the output is read from
    # the start.
    def read_output(self, job_id: str, stream: str = "stdout", *, incremental: bool = True) -> str:
        if stream not in {"stdout", "stderr"}:
            raise ValueError("stream must be 'stdout' or 'stderr'.")

        job = self.get(job_id)
        if job.host_job_dir is None:
            raise SandboxToolError(
                "job_output_unavailable",
                f"Job output is not available on host filesystem: {job_id}",
            )

        output_path = job.host_job_dir / stream
        if not output_path.exists():
            return ""

        if not incremental:
            content, _ = self._read_bounded_text(output_path, start=0)
            return content

        offset_attr = f"{stream}_offset"
        offset = int(getattr(job, offset_attr))
        new_content, new_offset = self._read_bounded_text(output_path, start=offset)
        setattr(job, offset_attr, new_offset)
        return new_content

    # Job lifecycle

    # Mark a job as finished with the given exit code.
    def mark_done(self, job_id: str, exit_code: int | None) -> BackgroundJob:
        job = self.get(job_id)
        job.status = "done"
        job.exit_code = exit_code
        return job

    # Mark a job as killed; an unset exit code becomes -9.
    def mark_killed(self, job_id: str) -> BackgroundJob:
        job = self.get(job_id)
        job.status = "killed"
        if job.exit_code is None:
            job.exit_code = -9
        return job

    # Remove finished jobs started more than ttl_seconds ago, with their job directories; return the
    # number removed.
    def purge_stale(self, ttl_seconds: float = 3600.0) -> int:
        import shutil
        cutoff = time.time() - ttl_seconds
        with self._lock:
            stale_ids = [
                job_id
                for job_id, job in self._jobs.items()
                if job.status in ("done", "killed") and job.started_at < cutoff
            ]
            removed = [self._jobs.pop(job_id) for job_id in stale_ids]
        for job in removed:
            if job.host_job_dir and job.host_job_dir.exists():
                shutil.rmtree(job.host_job_dir, ignore_errors=True)
        return len(removed)

    # Forget all jobs (their directories are left on disk).
    def reset(self) -> None:
        with self._lock:
            self._jobs.clear()

    # Return an unused job id, purging stale jobs first.
    def _new_job_id(self) -> str:
        self.purge_stale()
        while True:
            candidate = f"{JOB_ID_PREFIX}{uuid.uuid4().hex[:8]}"
            with self._lock:
                if candidate not in self._jobs:
                    return candidate


# Process-wide registry shared by all tool calls.
JOB_REGISTRY = JobRegistry()
