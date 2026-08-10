"""A tiny in-process job registry for the probes that must not block the event loop.

WHY THIS EXISTS. `falsify_scan` and `calibration` spawn a Node re-roll per seed per arm; a 20-loss
scan is minutes, not milliseconds. Running one inside a request handler would wedge the whole
server — uvicorn serves every request on one event loop, so a single blocking call stalls the page
that is polling for the result. So the handler SUBMITS and returns a job id, the work runs on a
thread, and the page polls.

Deliberately not Celery/RQ/a database: the prober is a single-user forensic tool pointed at a
directory of traces. State lives for the life of the process, and a restart losing a queued scan is
correct behaviour, not data loss.

Two properties the callers rely on:

* **A job never raises into the request.** A failure is captured as `status="error"` with the
  exception text, because a probe failing on a trace that lacks a `reconstruction.json` sibling is
  an ordinary outcome that the page must render, not a 500.
* **`max_workers` is small and shared.** Each concurrent job spawns Node processes of its own, and
  the box normally carries a live training run — see the contention notes in the root CLAUDE.md.
  Two heavy probes at once is already generous.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

_TERMINAL = ("done", "error")


class Job:
    """One submitted probe. Mutated only under the registry lock."""

    def __init__(self, job_id: str, kind: str, params: dict) -> None:
        self.id = job_id
        self.kind = kind
        self.params = params
        self.status = "pending"       # pending -> running -> done | error
        self.result = None
        self.error = None
        self.submitted = time.time()
        self.started = None
        self.finished = None

    def as_dict(self, *, with_result: bool = True) -> dict:
        out = {
            "id": self.id, "kind": self.kind, "params": self.params, "status": self.status,
            "error": self.error, "submitted": self.submitted,
            "started": self.started, "finished": self.finished,
            "elapsed_sec": round((self.finished or time.time()) - (self.started or self.submitted), 1),
            "done": self.status in _TERMINAL,
        }
        if with_result:
            out["result"] = self.result
        return out


class JobRegistry:
    """Submit work to a small thread pool and read back its state by id."""

    def __init__(self, max_workers: int = 2) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers,
                                        thread_name_prefix="prober-web-job")
        self._jobs: "dict[str, Job]" = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, params: dict, fn) -> Job:
        job = Job(uuid.uuid4().hex[:12], kind, params)
        with self._lock:
            self._jobs[job.id] = job
        self._pool.submit(self._run, job, fn)
        return job

    def _run(self, job: Job, fn) -> None:
        with self._lock:
            job.status, job.started = "running", time.time()
        try:
            result = fn()
        except Exception as exc:                       # noqa: BLE001 — a failure IS a job state
            with self._lock:
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.finished = time.time()
            # The text goes to the page; the traceback goes to the server log, where it is useful.
            traceback.print_exc()
            return
        with self._lock:
            job.status, job.result, job.finished = "done", result, time.time()

    def get(self, job_id: str) -> "Job | None":
        with self._lock:
            return self._jobs.get(job_id)

    def latest(self, kind: str) -> "Job | None":
        """The most recently submitted job of a kind — what a page shows when it is re-opened
        without a job id in the URL, so a finished scan survives a page reload."""
        with self._lock:
            jobs = [j for j in self._jobs.values() if j.kind == kind]
        return max(jobs, key=lambda j: j.submitted) if jobs else None

    def list(self) -> "list[dict]":
        with self._lock:
            jobs = sorted(self._jobs.values(), key=lambda j: j.submitted, reverse=True)
        return [j.as_dict(with_result=False) for j in jobs]

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
