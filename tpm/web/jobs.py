"""
Background Job Runner
-----------------------
Minimal in-memory job tracking for long-running pipeline operations
(currently: building a dataset's baseline analysis — Layer 1 profiling
and Layer 2 causal discovery scale with run count and run inline in a
background thread here, not a distributed job queue). A plain
threading.Thread + a locked dict is appropriate for a single-process
app — no Celery/Redis needed.

Job state does not survive a server restart; pipeline_service also
writes a completion marker to disk so a dataset that finished building
is still recognized as ready after a restart, independent of this
in-memory registry.
"""

import threading
import traceback
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class JobStatus:
    id: str
    status: str = "queued"  # queued | running | done | error
    step: str = ""
    current: int = 0
    total: int = 0
    message: str = ""
    error: Optional[str] = None
    result: Any = None


_JOBS: Dict[str, JobStatus] = {}
_LOCK = threading.Lock()


def get_job(job_id: str) -> Optional[JobStatus]:
    with _LOCK:
        return _JOBS.get(job_id)


def start_job(fn: Callable[[Callable[..., None]], Any]) -> str:
    """
    Run `fn(progress)` in a background thread, where `progress(step,
    current=0, total=0, message="")` updates the job's status as `fn`
    runs. Returns the job id immediately (non-blocking).
    """
    job_id = uuid.uuid4().hex[:12]
    job = JobStatus(id=job_id)
    with _LOCK:
        _JOBS[job_id] = job

    def progress(step: str, current: int = 0, total: int = 0, message: str = ""):
        with _LOCK:
            job.status = "running"
            job.step = step
            job.current = current
            job.total = total
            job.message = message

    def run():
        try:
            progress("starting")
            result = fn(progress)
            with _LOCK:
                job.status = "done"
                job.result = result
        except Exception as exc:
            with _LOCK:
                job.status = "error"
                job.error = str(exc)
            traceback.print_exc()

    threading.Thread(target=run, daemon=True).start()
    return job_id
