"""The launcher's half of a GPU WINDOW (T46): trigger, pause, GO, wait, resume — or HOLD.

The supervisor (``run._supervise``) owns the child; this class owns the window's files and its
waiting. The supervisor asks :meth:`WindowController.due` while the child runs; a due request makes
it SIGTERM the child exactly as a quit or an ``r`` restart does (the child saves
``final_model_interrupted.zip`` at its next safe point and exits ``INTERRUPTED``). Once the child is
gone the supervisor calls :meth:`WindowController.run_window`, which returns ``"resume"`` (the
supervisor then re-launches through its ordinary same-run RESTART path: same pin, same command, the
RESUME role) or ``"quit"``.

Nothing here touches the GPU lease: the launcher process stays alive through the window, so a lease
whose ``--watch-pid`` is the launcher stays with its holder, and the guest's GPU commands run on the
holder's token (``utils.gpu_lease``).
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Callable, List, Optional

from main.exit_codes import TrainExitCode
from main.launcher.gpu_window import protocol as P

#: How often the launcher re-reads ``request.json`` while the child runs, and polls while it waits.
REQUEST_POLL_S = 2.0
WAIT_POLL_S = 1.0
#: How long the child's processes may take to leave the GPU after it exits; past it the window HOLDS.
GPU_RELEASE_TIMEOUT_S = 600.0
#: A checkpoint at most this much OLDER than the pause's SIGTERM still counts as the pause's save (the
#: trainer writes its interrupted save after the SIGTERM; the slack covers filesystem mtime rounding).
CHECKPOINT_SLACK_S = 2.0


class GpuQueryFailed(RuntimeError):
    """``nvidia-smi`` is installed but could not answer (non-zero exit, timeout, OSError). NOT "free": a
    hung or failing ``nvidia-smi`` is exactly when the card's state is least known (review 2026-10-20)."""


def gpu_compute_pids() -> "Optional[List[int]]":
    """PIDs of every process holding a CUDA context (``nvidia-smi``); None only when ``nvidia-smi`` is
    ABSENT (a CPU box). Raises :class:`GpuQueryFailed` when it is present but fails: the window then
    keeps waiting and HOLDS at its bound, never GOes on an unanswered query.

    The whole box, not the child's tree: under the lease only its holder uses the GPU, so "no compute
    process" is what "the GPU is free for the guest" means (the hand-made windows' check, 2026-10-19)."""
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return None
    try:
        out = subprocess.run([exe, "--query-compute-apps=pid", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        raise GpuQueryFailed(f"nvidia-smi did not answer: {type(e).__name__}") from e
    if out.returncode != 0:
        raise GpuQueryFailed(f"nvidia-smi exited rc {out.returncode}: {out.stderr.strip()[:200]}")
    return [int(t) for t in out.stdout.split() if t.strip().isdigit()]


class WindowController:
    """One launcher session's window state. ``run_dir_fn`` returns the run dir NOW (it can move on a
    restart); ``add_event`` / ``set_status`` reach the launcher's event log and its status line."""

    def __init__(self, run_dir_fn: Callable[[], str], *, add_event: Callable[[str], None],
                 set_status: Callable[[Optional[str]], None],
                 gpu_pids: Callable[[], "Optional[List[int]]"] = gpu_compute_pids,
                 find_checkpoint: Optional[Callable[[str], Optional[str]]] = None,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                 request_poll_s: float = REQUEST_POLL_S, wait_poll_s: float = WAIT_POLL_S,
                 gpu_release_timeout_s: float = GPU_RELEASE_TIMEOUT_S) -> None:
        self._run_dir_fn = run_dir_fn
        self._event = add_event
        self._status = set_status
        self._gpu_pids = gpu_pids
        if find_checkpoint is None:
            from main.launcher.checkpoint import find_latest_checkpoint

            def find_checkpoint(rd: str) -> Optional[str]:
                return find_latest_checkpoint(run_dir=rd)
        self._find_checkpoint = find_checkpoint
        self._clock = clock
        self._sleep = sleep
        self._request_poll_s = request_poll_s
        self._wait_poll_s = wait_poll_s
        self._gpu_release_timeout_s = gpu_release_timeout_s
        self._last_poll = float("-inf")
        self._identity = P.process_identity(os.getpid()) or {"pid": os.getpid(), "start_ticks": -1}
        self._phase_record: dict = {}

    # ── identity + status ─────────────────────────────────────────────────────────────────────
    @property
    def root(self):
        return P.window_root(self._run_dir_fn())

    def announce(self) -> None:
        """Record that THIS launcher honours windows for the run (``arm`` refuses without it).

        A window left OPEN by a launcher that is GONE (it died or was killed mid-window) is closed
        here as ABANDONED: otherwise every later ``arm`` refuses on it forever, and its ``resume``
        writes a file no launcher reads. Nothing is resumed over it — this launcher was started by
        hand, from whatever checkpoint its operator chose."""
        cur = P.read_current(self._run_dir_fn())
        if (cur is not None and cur.get("phase") in P.OPEN_PHASES
                and not P.identity_alive(cur.get("launcher"))):
            P.write_json_atomic(self.root / P.CURRENT_FILE, {
                **cur, "phase": P.ABANDONED, "since_epoch": self._clock(), "launcher": self._identity,
                "reason": f"its launcher {cur.get('launcher')} is gone (left it {str(cur.get('phase')).upper()}: "
                          f"{cur.get('reason', '')})", "stale_launcher": cur.get("launcher")})
            self._event(f"⚠️  GPU window {cur.get('window_id')}: its launcher is gone and it was left "
                        f"{str(cur.get('phase')).upper()} — closed as ABANDONED (check the run's checkpoint)")
        P.write_json_atomic(self.root / P.LAUNCHER_FILE, {
            "launcher": self._identity, "started_epoch": self._clock(), "run_dir": self._run_dir_fn()})
        self._event(f"🪟 GPU window: ENABLED — arm one with `python -m main.launcher.gpu_window arm "
                    f"{self._run_dir_fn()} …` (requests read from {self.root})")

    def _set_phase(self, req: P.Request, phase: str, reason: str, **extra) -> None:
        rec = {"window_id": req.window_id, "phase": phase, "reason": reason, "since_epoch": self._clock(),
               "launcher": self._identity, "note": req.note, **extra}
        self._phase_record = rec
        P.write_json_atomic(self.root / P.CURRENT_FILE, rec)
        icon = {P.HOLD: "🛑", P.PAUSING: "⏸", P.GO: "▶", P.CLAIMED: "🏃"}.get(phase)
        self._status(None if icon is None else f"{icon} GPU window {req.window_id}: {phase.upper()} — {reason}")

    # ── the trigger (polled while the child runs) ────────────────────────────────────────────
    def due(self) -> Optional[P.Request]:
        """The request to act on NOW, consumed; else None. Throttled to one read per ``request_poll_s``.

        A request addressed to another launcher (a dead session's leftover), or one that breaks a
        bound, is REFUSED: moved aside and reported, never acted on."""
        now = self._clock()
        if now - self._last_poll < self._request_poll_s:
            return None
        self._last_poll = now
        path = self.root / P.REQUEST_FILE
        if not path.exists():
            return None
        raw = P.read_json(path)
        try:
            req = P.Request.from_json(raw)
        except P.RequestInvalid as e:
            self._refuse(path, raw, f"invalid request: {e}")
            return None
        if (int(req.launcher.get("pid", -1)), int(req.launcher.get("start_ticks", -2))) != (
                int(self._identity["pid"]), int(self._identity["start_ticks"])):
            self._refuse(path, raw, f"addressed to launcher {req.launcher}, not this one {self._identity} "
                                    "(armed for an earlier launcher session?)")
            return None
        if now < req.not_before_epoch:
            return None
        wdir = P.window_dir(self._run_dir_fn(), req.window_id)
        wdir.mkdir(parents=True, exist_ok=True)
        try:
            os.rename(path, wdir / P.REQUEST_FILE)         # CONSUME: a `cancel` racing this loses or wins whole
        except FileNotFoundError:
            return None
        self._set_phase(req, P.PAUSING, "stopping the run at its next safe point (SIGTERM → checkpoint)")
        self._event(f"⏸ GPU window {req.window_id}: trigger epoch {P.describe_epoch(req.not_before_epoch)} "
                    f"reached — pausing the run at its next safe point")
        return req

    def _refuse(self, path, raw: Optional[dict], why: str) -> None:
        wid = (raw or {}).get("window_id") if isinstance((raw or {}).get("window_id"), str) else None
        dest = self.root / f"refused_{wid or P.new_window_id(self._clock())}.json"
        try:
            os.rename(path, dest)
        except FileNotFoundError:
            return
        P.write_json_atomic(self.root / P.CURRENT_FILE, {
            "window_id": wid, "phase": P.REFUSED, "reason": why, "since_epoch": self._clock(),
            "launcher": self._identity})
        self._event(f"⚠️  GPU window request REFUSED ({why}) — moved to {dest.name}; the run continues")

    # ── the window (after the paused child exited) ───────────────────────────────────────────
    def abandon(self, req: P.Request, why: str) -> None:
        self._set_phase(req, P.ABANDONED, why)
        self._event(f"🪟 GPU window {req.window_id} ABANDONED: {why}")

    def run_window(self, req: P.Request, rc: int, sigterm_epoch: float,
                   should_quit: Callable[[], bool]) -> str:
        """Drive one window from the child's exit to its end. Returns ``"resume"`` or ``"quit"``.

        Any step that cannot finish cleanly HOLDS (TRAINING_RUN_SOP §3 rule 3): the run stays paused,
        the reason is in ``current.json`` and the events, and only an operator's ``resume`` (or a quit)
        ends it. A guest that never STARTS is not a failure — the launcher claims the window MISSED and
        resumes, because nothing ran on the GPU."""
        wdir = P.window_dir(self._run_dir_fn(), req.window_id)
        if rc != int(TrainExitCode.INTERRUPTED):
            return self._hold(req, wdir, f"the paused child exited rc {rc}, not INTERRUPTED "
                                         f"({int(TrainExitCode.INTERRUPTED)}): its stop is not a clean pause",
                              should_quit)
        ckpt = self._find_checkpoint(self._run_dir_fn())
        if ckpt is None or os.path.getmtime(ckpt) < sigterm_epoch - CHECKPOINT_SLACK_S:
            return self._hold(req, wdir, f"no checkpoint saved by the pause (latest: {ckpt})", should_quit)

        # The GPU must be EMPTY before GO: the child's tree can outlive it for a moment.
        out = self._await_empty_gpu(req, wdir, "the run stopped", should_quit)
        if out is not None:
            return out

        go_epoch = self._clock()
        P.write_json_atomic(wdir / P.GO_FILE, {"window_id": req.window_id, "go_epoch": go_epoch,
                                               "checkpoint": ckpt, "launcher": self._identity})
        self._set_phase(req, P.GO, f"run checkpointed ({os.path.basename(ckpt)}) and off the GPU — waiting "
                                   f"≤ {req.guest_start_timeout_s:.0f} s for the guest to claim",
                        go_epoch=go_epoch, checkpoint=ckpt)
        self._event(f"▶ GPU window {req.window_id}: GO — {os.path.basename(ckpt)} saved, GPU free")

        claim = None
        while claim is None:
            claim = P.read_json(wdir / P.CLAIM_FILE)
            if claim is not None:
                break
            if should_quit():
                return self._quit(req)
            if self._clock() - go_epoch > req.guest_start_timeout_s:
                if P.create_json_exclusive(wdir / P.CLAIM_FILE, {"by": "launcher", "window_id": req.window_id,
                                                                 "missed_epoch": self._clock()}):
                    self._set_phase(req, P.MISSED, f"no guest claimed within {req.guest_start_timeout_s:.0f} s "
                                                   "of GO — resuming (nothing ran on the GPU)")
                    self._event(f"⏭ GPU window {req.window_id}: MISSED (no guest within "
                                f"{req.guest_start_timeout_s:.0f} s) — resuming the run")
                    return "resume"
                continue                                   # the guest won the race: read its claim
            self._sleep(self._wait_poll_s)

        guest = claim.get("guest") or {}
        job = claim.get("job")
        self._set_phase(req, P.CLAIMED, f"guest pid {guest.get('pid')} running (window ends by "
                                        f"{P.describe_epoch(go_epoch + req.max_window_s)})",
                        go_epoch=go_epoch, guest=guest)
        self._event(f"🏃 GPU window {req.window_id}: guest pid {guest.get('pid')} claimed the GPU")
        while True:
            if not job:
                # the guest records its command's process group in the claim right after it starts the command
                job = (P.read_json(wdir / P.CLAIM_FILE) or {}).get("job")
                if job:
                    self._set_phase(req, P.CLAIMED, f"guest pid {guest.get('pid')} running its job "
                                                    f"(pgid {job.get('pgid')}; window ends by "
                                                    f"{P.describe_epoch(go_epoch + req.max_window_s)})",
                                    go_epoch=go_epoch, guest=guest, job=job)
                    self._event(f"🏃 GPU window {req.window_id}: the guest's job is process group {job.get('pgid')}")
            done = P.read_json(wdir / P.DONE_FILE)
            if done is not None and done.get("window_id") == req.window_id:
                rc_guest = done.get("rc")
                if rc_guest == 0:
                    # The guest's own processes must have LEFT the GPU before the run comes back (a
                    # guest command that returns while a worker it spawned still holds a context
                    # would put two jobs on the card: the trainer's first update then OOMs).
                    out = self._await_empty_gpu(req, wdir, "the guest exited 0", should_quit)
                    if out is not None:
                        return out
                    self._set_phase(req, P.DONE, "guest exited 0 — resuming the run", guest_rc=0)
                    self._event(f"✅ GPU window {req.window_id}: guest DONE (rc 0) after "
                                f"{(self._clock() - go_epoch) / 60:.1f} min — resuming the run")
                    return "resume"
                return self._hold(req, wdir, f"the guest exited rc {rc_guest}", should_quit)
            if should_quit():
                return self._quit(req)
            if not P.identity_alive(guest):
                if P.read_json(wdir / P.DONE_FILE) is not None:
                    continue                               # it wrote DONE as it exited: read it
                job = job or (P.read_json(wdir / P.CLAIM_FILE) or {}).get("job")
                return self._hold(req, wdir, f"the guest (pid {guest.get('pid')}) died without writing DONE; "
                                             f"{P.job_note(job)}", should_quit)
            if self._clock() - go_epoch > req.max_window_s:
                job = job or (P.read_json(wdir / P.CLAIM_FILE) or {}).get("job")
                return self._hold(req, wdir, f"the guest overran the window ({req.max_window_s / 60:.0f} min "
                                             f"after GO) and may still hold the GPU; {P.job_note(job)}",
                                  should_quit)
            self._sleep(self._wait_poll_s)

    def _wait_gpu_empty(self, should_quit: Callable[[], bool],
                        on_wait: Optional[Callable[[Optional[List[int]], Optional[str]], None]] = None
                        ) -> "tuple[str, str]":
        """Wait until ``nvidia-smi`` lists no compute process. Returns ``("clean", "")``; ``("absent", "")`` when
        ``nvidia-smi`` is ABSENT (a CPU box: warned, not checked); ``("quit", "")``; or ``("timeout", what)`` once
        ``gpu_release_timeout_s`` has passed with the GPU still busy (``what`` names the pids) or ``nvidia-smi``
        still failing. ``on_wait(pids, failed)`` is told the state at every poll that is not clean."""
        t0 = self._clock()
        while True:
            failed = None
            try:
                pids = self._gpu_pids()
            except GpuQueryFailed as e:
                pids, failed = None, str(e)
            if failed is None:
                if pids is None:
                    self._event("⚠️  GPU window: nvidia-smi absent — the GPU's release is NOT checked")
                    return "absent", ""
                if not pids:
                    return "clean", ""
            if on_wait is not None:
                on_wait(pids, failed)
            if should_quit():
                return "quit", ""
            if self._clock() - t0 > self._gpu_release_timeout_s:
                what = (f"nvidia-smi still fails ({failed})" if failed is not None
                        else f"the GPU still lists compute pid(s) {pids}")
                return "timeout", what
            self._sleep(self._wait_poll_s)

    def _await_empty_gpu(self, req: P.Request, wdir, after: str,
                         should_quit: Callable[[], bool]) -> Optional[str]:
        """None once the GPU is empty (or ``nvidia-smi`` is ABSENT); else the window's outcome — ``"quit"``, or the
        operator's ``"resume"`` that ends the HOLD this enters when the GPU is still busy, or ``nvidia-smi`` still
        fails, ``gpu_release_timeout_s`` after ``after``."""
        verdict, what = self._wait_gpu_empty(should_quit)
        if verdict in ("clean", "absent"):
            return None
        if verdict == "quit":
            return self._quit(req)
        return self._hold(req, wdir, f"{what} {self._gpu_release_timeout_s:.0f} s after {after}", should_quit)

    def _quit(self, req: P.Request) -> str:
        self.abandon(req, "the launcher is quitting")
        return "quit"

    def _hold(self, req: P.Request, wdir, why: str, should_quit: Callable[[], bool]) -> str:
        """Stay paused until an operator writes ``RESUME.json`` for THIS window (or the launcher quits).

        A ``resume`` is an INTENT, not a go: the trainer re-launches only onto an EMPTY GPU, the check the window
        made at its entry (:meth:`_wait_gpu_empty`). While the GPU lists a compute pid (a guest command that
        outlived its wrapper, the pids are named) or ``nvidia-smi`` fails, the HOLD stays, its reason says what
        it waits on, and the launcher waits up to ``gpu_release_timeout_s`` for the card to empty. Past that the
        resume is REFUSED: ``RESUME.json`` is set aside (``RESUME.refused_<epoch>.json``), the HOLD stays with the
        reason, and a new ``resume`` is needed. A re-launch onto a busy card OOMs at its first update and burns the
        3 rapid-crash restarts."""
        self._set_phase(req, P.HOLD, why)
        self._event(f"🛑 GPU WINDOW HOLD {req.window_id}: {why}. The run stays PAUSED until "
                    f"`python -m main.launcher.gpu_window resume {self._run_dir_fn()} --reason …` "
                    "(and the GPU is empty)")
        while True:
            res = P.read_json(wdir / P.RESUME_FILE)
            if res is not None and res.get("window_id") == req.window_id:
                waiting: list = [None]

                def on_wait(pids: Optional[List[int]], failed: Optional[str]) -> None:
                    state = failed if failed is not None else tuple(pids or ())
                    if state == waiting[0]:
                        return
                    waiting[0] = state
                    what = (f"nvidia-smi fails ({failed})" if failed is not None
                            else f"the GPU still lists compute pid(s) {pids}")
                    self._set_phase(req, P.HOLD, f"{why} | resume requested, waiting up to "
                                                 f"{self._gpu_release_timeout_s:.0f} s: {what}", hold_reason=why)
                    self._event(f"⏳ GPU window {req.window_id}: resume requested but {what} - the run re-launches "
                                f"only onto an empty GPU (waiting up to {self._gpu_release_timeout_s:.0f} s)")
                verdict, what = self._wait_gpu_empty(should_quit, on_wait)
                if verdict == "quit":
                    return self._quit(req)
                if verdict == "timeout":
                    try:
                        os.rename(wdir / P.RESUME_FILE, wdir / f"RESUME.refused_{int(self._clock())}.json")
                    except FileNotFoundError:
                        pass
                    reason = (f"resume REFUSED: {what} {self._gpu_release_timeout_s:.0f} s after the operator's "
                              f"resume; free the GPU, then `resume` again (HOLD was: {why})")
                    self._set_phase(req, P.HOLD, reason, hold_reason=why)
                    self._event(f"🛑 GPU WINDOW HOLD {req.window_id}: {reason}")
                    continue
                self._set_phase(req, P.RESUMED, f"operator resume after HOLD: {res.get('reason', '')}",
                                hold_reason=why)
                self._event(f"▶ GPU window {req.window_id}: HOLD ended by `resume` ({res.get('reason', '')}) "
                            f"— {'GPU empty' if verdict == 'clean' else 'nvidia-smi absent, GPU not checked'}, "
                            "resuming the run")
                return "resume"
            if should_quit():
                return self._quit(req)
            self._sleep(self._wait_poll_s)

    def mark_resumed(self, req: P.Request, checkpoint: str) -> None:
        """The supervisor re-launched the run: close the window's record."""
        prev = dict(self._phase_record)
        self._set_phase(req, P.RESUMED, f"run re-launched from {os.path.basename(checkpoint)}",
                        outcome=prev.get("phase"), outcome_reason=prev.get("reason"))
