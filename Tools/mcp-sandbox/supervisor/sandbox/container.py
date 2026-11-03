# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from sandbox.exec import (
    _exec_bash_native as exec_bash,  # noqa: F401
    _job_files_result,
    _kill_process_group,
)
from sandbox.jobs import JOB_REGISTRY


# List background jobs; refresh native running jobs whose process has exited.
def list_background_jobs() -> dict:
    for listed in JOB_REGISTRY.list_jobs():
        if listed.get("runtime") == "native" and listed.get("status") == "running":
            try:
                job = JOB_REGISTRY.get(listed["job_id"])
                process = job.process
                if process is not None:
                    returncode = process.poll()
                    if returncode is not None:
                        JOB_REGISTRY.mark_done(job.job_id, returncode)
            except Exception:
                # A job that vanished or cannot be polled is simply reported with its old status.
                pass
    return {"jobs": JOB_REGISTRY.list_jobs()}


# Poll a background job and return incremental stdout/stderr since last read.
def foreground_background_job(job_id: str) -> dict:
    job = JOB_REGISTRY.get(job_id)
    process = job.process
    if process is not None and job.status == "running":
        returncode = process.poll()
        if returncode is not None:
            JOB_REGISTRY.mark_done(job.job_id, returncode)
    stdout, stderr, truncated = _job_files_result(job, incremental=True)
    return {
        **job.to_result(),
        "new_stdout": stdout,
        "new_stderr": stderr,
        "truncated": truncated,
    }


# Terminate a running background job's process group, then mark it killed.
def kill_background_job(job_id: str) -> dict:
    job = JOB_REGISTRY.get(job_id)
    process = job.process
    if process is not None:
        returncode = process.poll()
        if returncode is not None:
            JOB_REGISTRY.mark_done(job.job_id, returncode)
            return job.to_result()
        _kill_process_group(process)
        try:
            process.wait(timeout=2)
        except Exception:
            # The process may ignore the wait timeout or already be gone; the job is marked killed anyway.
            pass
    JOB_REGISTRY.mark_killed(job.job_id)
    return job.to_result()
