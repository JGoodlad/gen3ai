"""The GPU WINDOW's preconditions, refusals and HOLDs, one at a time (T46). No child process; the full cycle
with a real tiny CPU child is ``window_cycle_test.py``.

Each test names the rule it pins: ``arm`` checks EVERY precondition before it writes anything; a trigger is
an ABSOLUTE epoch with a sanity bound; a request is honoured only by the launcher it was addressed to; every
step that cannot finish cleanly HOLDS; the window is OFF unless the launch asked for it.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from main.exit_codes import TrainExitCode
from main.launcher.checkpoint import _strip_launcher_args
from main.launcher.gpu_window import cli
from main.launcher.gpu_window import protocol as P
from main.launcher.gpu_window import controller as C
from main.launcher.gpu_window.controller import WindowController
from main.launcher.run import build_launcher_parser


@pytest.fixture(autouse=True)
def _private_gpu_lock(tmp_path, monkeypatch):
    """A lock of the test's own: the box's real lease (a live run's) must never decide a test's verdict."""
    monkeypatch.setenv("GEN3AI_GPU_LOCK", str(tmp_path / "gpu.lock"))


@pytest.fixture
def run_dir(tmp_path) -> Path:
    d = tmp_path / "models" / "rb_toy"
    d.mkdir(parents=True)
    return d


def _controller(run_dir: Path, events: list, *, gpu_pids=lambda: [], ckpt=None, **kw) -> WindowController:
    return WindowController(lambda: str(run_dir), add_event=events.append, set_status=lambda s: None,
                            gpu_pids=gpu_pids, find_checkpoint=lambda rd: ckpt, request_poll_s=0.0,
                            wait_poll_s=0.01, **kw)


def _arm(run_dir: Path, *extra: str) -> int:
    return cli.main(["arm", str(run_dir), "--no-precheck", "test", "--no-lease-check", "test", *extra])


# ── arm: every precondition before anything is written ─────────────────────────────────────────

def test_arm_refuses_without_a_live_window_launcher(run_dir, capsys):
    assert _arm(run_dir) == cli.EXIT_REFUSED
    assert "--gpu-window" in capsys.readouterr().err
    # a launcher.json whose launcher is gone (pid reused / dead) is no launcher
    P.write_json_atomic(P.window_root(run_dir) / P.LAUNCHER_FILE,
                        {"launcher": {"pid": os.getpid(), "start_ticks": -7}})
    assert _arm(run_dir) == cli.EXIT_REFUSED
    assert not (P.window_root(run_dir) / P.REQUEST_FILE).exists()


@pytest.mark.parametrize("when, fragment", [
    (["--at-epoch", str(time.time() - 3600)], "BEFORE the arming"),
    (["--in-minutes", str(49 * 60)], "more than 48 h"),
    (["--max-window-min", "0"], "max window"),
    (["--guest-start-timeout-s", "-1"], "guest start timeout"),
    (["--horizon-hours", "1000"], "horizon"),
])
def test_arm_refuses_an_out_of_bounds_trigger(run_dir, capsys, when, fragment):
    _controller(run_dir, []).announce()
    assert _arm(run_dir, *when) == cli.EXIT_REFUSED
    assert fragment in capsys.readouterr().err
    assert not (P.window_root(run_dir) / P.REQUEST_FILE).exists()


def test_arm_runs_the_precheck_with_the_gpu_hidden_and_refuses_on_failure(run_dir, capsys):
    _controller(run_dir, []).announce()
    rc = cli.main(["arm", str(run_dir), "--no-lease-check", "test",
                   "--precheck", "echo the-tool-refused; exit 9"])
    assert rc == cli.EXIT_REFUSED
    err = capsys.readouterr().err
    assert "rc 9" in err and "the-tool-refused" in err
    assert not (P.window_root(run_dir) / P.REQUEST_FILE).exists(), "a failed precheck must arm NOTHING"
    # the precheck sees CUDA hidden (an empty CUDA_VISIBLE_DEVICES), whatever the caller's environment
    rc = cli.main(["arm", str(run_dir), "--no-lease-check", "test",
                   "--precheck", '[ "${CUDA_VISIBLE_DEVICES-unset}" = "" ]'])
    assert rc == 0
    req = P.read_json(P.window_root(run_dir) / P.REQUEST_FILE)
    assert req["precheck"]["rc"] == 0


def test_arm_requires_the_caller_to_hold_the_gpu_lease(run_dir, capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("GEN3AI_GPU_LOCK", str(tmp_path / "gpu.lock"))
    monkeypatch.delenv("GEN3AI_GPU_LEASE_TOKEN", raising=False)
    monkeypatch.delenv("GEN3AI_GPU_LEASE_TOKEN_FILE", raising=False)
    _controller(run_dir, []).announce()
    assert cli.main(["arm", str(run_dir), "--no-precheck", "test"]) == cli.EXIT_REFUSED
    assert "does not hold the GPU lease" in capsys.readouterr().err


def test_one_window_at_a_time_and_cancel(run_dir):
    events: list = []
    ctl = _controller(run_dir, events)
    ctl.announce()
    assert _arm(run_dir, "--in-minutes", "30") == 0
    assert _arm(run_dir) == cli.EXIT_REFUSED, "a second arm while one is armed"
    assert cli.main(["cancel", str(run_dir)]) == 0
    assert ctl.due() is None
    assert cli.main(["cancel", str(run_dir)]) == cli.EXIT_REFUSED, "nothing left to cancel"


# ── the trigger: absolute epoch, addressed launcher ────────────────────────────────────────────

def test_a_future_trigger_waits_and_a_due_one_is_consumed(run_dir):
    events: list = []
    ctl = _controller(run_dir, events)
    ctl.announce()
    assert _arm(run_dir, "--in-minutes", "30") == 0
    assert ctl.due() is None and (P.window_root(run_dir) / P.REQUEST_FILE).exists()
    req = P.Request.from_json(P.read_json(P.window_root(run_dir) / P.REQUEST_FILE))
    later = WindowController(lambda: str(run_dir), add_event=events.append, set_status=lambda s: None,
                             clock=lambda: req.not_before_epoch + 1, request_poll_s=0.0)
    got = later.due()
    assert got is not None and got.window_id == req.window_id
    assert not (P.window_root(run_dir) / P.REQUEST_FILE).exists()
    assert (P.window_dir(run_dir, req.window_id) / P.REQUEST_FILE).exists()
    assert P.read_current(run_dir)["phase"] == P.PAUSING


def test_a_wider_horizon_armed_is_the_horizon_the_launcher_judges(run_dir):
    """The launcher re-checks a request's bounds against the request's OWN horizon, not the default:
    an arm the CLI accepted with --horizon-hours 72 must not be refused by the launcher."""
    ctl = _controller(run_dir, [])
    ctl.announce()
    assert _arm(run_dir, "--horizon-hours", "72", "--in-minutes", str(60 * 60)) == 0
    req = P.Request.from_json(P.read_json(P.window_root(run_dir) / P.REQUEST_FILE))
    assert req.horizon_h == 72
    assert ctl.due() is None and P.read_current(run_dir) is None, "refused instead of waiting for its epoch"


def test_a_request_for_another_launcher_is_refused_not_acted_on(run_dir):
    """A leftover from an EARLIER launcher session (or a hand-written file) must never pause this run."""
    events: list = []
    ctl = _controller(run_dir, events)
    ctl.announce()
    req = P.Request(window_id=P.new_window_id(), launcher={"pid": os.getpid(), "start_ticks": -3},
                    armed_epoch=time.time(), not_before_epoch=time.time(), guest_start_timeout_s=60,
                    max_window_s=600)
    P.write_json_atomic(P.window_root(run_dir) / P.REQUEST_FILE, req.to_json())
    assert ctl.due() is None
    assert P.read_current(run_dir)["phase"] == P.REFUSED
    assert not (P.window_root(run_dir) / P.REQUEST_FILE).exists()
    assert any("REFUSED" in e for e in events)
    P.write_json_atomic(P.window_root(run_dir) / P.REQUEST_FILE, {"window_id": "x"})
    assert ctl.due() is None and P.read_current(run_dir)["phase"] == P.REFUSED


# ── run_window: every unclean step HOLDS ───────────────────────────────────────────────────────

def _request(run_dir: Path, ctl: WindowController, **kw) -> P.Request:
    ctl.announce()
    assert _arm(run_dir, *[str(x) for kv in kw.items() for x in (f"--{kv[0].replace('_', '-')}", kv[1])]) == 0
    req = ctl.due()
    assert req is not None
    return req


def _hold_reason(run_dir: Path, ctl: WindowController, req: P.Request, rc: int, **kw) -> str:
    """Run the window until it HOLDS, then quit; return the HOLD's reason."""
    seen: list = []

    def should_quit() -> bool:
        cur = P.read_current(run_dir) or {}
        if cur.get("phase") == P.HOLD:
            seen.append(cur["reason"])
            return True
        return False
    assert ctl.run_window(req, rc, kw.get("sigterm_epoch", time.time() - 1), should_quit) == "quit"
    assert seen, "the window ended without a HOLD"
    return seen[0]


def test_a_child_that_did_not_stop_cleanly_holds(run_dir, tmp_path):
    ck = tmp_path / "c.zip"
    ck.write_text("x")
    ctl = _controller(run_dir, [], ckpt=str(ck))
    req = _request(run_dir, ctl)
    assert "rc 1" in _hold_reason(run_dir, ctl, req, int(TrainExitCode.CRASH))


def test_no_checkpoint_from_the_pause_holds(run_dir, tmp_path):
    old = tmp_path / "old.zip"
    old.write_text("x")
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    for ckpt in (None, str(old)):
        ctl = _controller(run_dir, [], ckpt=ckpt)
        req = _request(run_dir, ctl)
        reason = _hold_reason(run_dir, ctl, req, int(TrainExitCode.INTERRUPTED))
        assert "no checkpoint saved by the pause" in reason
        P.write_json_atomic(P.window_root(run_dir) / P.CURRENT_FILE, {"phase": P.ABANDONED})


def test_a_gpu_that_is_not_released_holds_and_no_go_is_written(run_dir, tmp_path):
    ck = tmp_path / "c.zip"
    ck.write_text("x")
    ctl = _controller(run_dir, [], ckpt=str(ck), gpu_pids=lambda: [4242], gpu_release_timeout_s=0.05)
    req = _request(run_dir, ctl)
    assert "4242" in _hold_reason(run_dir, ctl, req, int(TrainExitCode.INTERRUPTED))
    assert not (P.window_dir(run_dir, req.window_id) / P.GO_FILE).exists()


def test_a_guest_that_dies_without_done_or_overruns_holds(run_dir, tmp_path):
    ck = tmp_path / "c.zip"
    ck.write_text("x")
    ctl = _controller(run_dir, [], ckpt=str(ck))
    req = _request(run_dir, ctl)
    wdir = P.window_dir(run_dir, req.window_id)
    # a "guest" whose identity is not a live process: claimed, then gone
    assert P.create_json_exclusive(wdir / P.CLAIM_FILE, {"by": "guest", "window_id": req.window_id,
                                                         "guest": {"pid": os.getpid(), "start_ticks": -1}})
    assert "died without writing DONE" in _hold_reason(run_dir, ctl, req, int(TrainExitCode.INTERRUPTED))

    P.write_json_atomic(P.window_root(run_dir) / P.CURRENT_FILE, {"phase": P.ABANDONED})
    ctl2 = _controller(run_dir, [], ckpt=str(ck))
    req2 = _request(run_dir, ctl2, max_window_min=0.001)
    w2 = P.window_dir(run_dir, req2.window_id)
    assert P.create_json_exclusive(w2 / P.CLAIM_FILE, {"by": "guest", "window_id": req2.window_id,
                                                       "guest": P.process_identity(os.getpid())})
    assert "overran" in _hold_reason(run_dir, ctl2, req2, int(TrainExitCode.INTERRUPTED))


def test_exactly_one_side_wins_the_claim(tmp_path):
    p = tmp_path / "w" / P.CLAIM_FILE
    assert P.create_json_exclusive(p, {"by": "guest"})
    assert not P.create_json_exclusive(p, {"by": "launcher"})
    assert P.read_json(p) == {"by": "guest"}


# ── OFF unless asked for ───────────────────────────────────────────────────────────────────────

def test_gpu_window_is_off_by_default_and_never_reaches_the_child():
    p = build_launcher_parser()
    assert p.parse_known_args([])[0].gpu_window is False
    assert p.parse_known_args(["--gpu-window"])[0].gpu_window is True
    assert _strip_launcher_args(["--steps", "5", "--gpu-window", "--device", "cuda"]) == \
        ["--steps", "5", "--device", "cuda"]


# ── review 2026-10-20: the GPU check never fails open, a dead launcher's window never blocks ─────

def test_a_failing_nvidia_smi_is_not_a_free_gpu(run_dir, tmp_path, monkeypatch):
    """A present nvidia-smi that fails (rc != 0, a hang) used to read as "absent" and GO unchecked."""
    fake = tmp_path / "nvidia-smi"
    fake.write_text("#!/bin/sh\necho 'Unable to determine the device handle' >&2\nexit 15\n")
    fake.chmod(0o755)
    monkeypatch.setattr(C.shutil, "which", lambda name: str(fake))
    with pytest.raises(C.GpuQueryFailed, match="rc 15"):
        C.gpu_compute_pids()
    monkeypatch.setattr(C.shutil, "which", lambda name: None)
    assert C.gpu_compute_pids() is None, "ABSENT (a CPU box) is the only unchecked case"


def test_a_window_holds_without_go_while_nvidia_smi_fails(run_dir, tmp_path):
    ck = tmp_path / "c.zip"
    ck.write_text("x")

    def failing():
        raise C.GpuQueryFailed("nvidia-smi exited rc 15")
    ctl = _controller(run_dir, [], ckpt=str(ck), gpu_pids=failing, gpu_release_timeout_s=0.05)
    req = _request(run_dir, ctl)
    assert "nvidia-smi still fails" in _hold_reason(run_dir, ctl, req, int(TrainExitCode.INTERRUPTED))
    assert not (P.window_dir(run_dir, req.window_id) / P.GO_FILE).exists()


def test_the_run_does_not_resume_while_the_guests_processes_hold_the_gpu(run_dir, tmp_path):
    """DONE rc 0 is not "the GPU is free": a worker the guest spawned can outlive it."""
    ck = tmp_path / "c.zip"
    ck.write_text("x")
    calls: list = []

    def pids():
        calls.append(1)
        return [] if len(calls) == 1 else [777]          # free at GO, busy after the guest
    ctl = _controller(run_dir, [], ckpt=str(ck), gpu_pids=pids, gpu_release_timeout_s=0.05)
    req = _request(run_dir, ctl)
    wdir = P.window_dir(run_dir, req.window_id)
    assert P.create_json_exclusive(wdir / P.CLAIM_FILE, {"by": "guest", "window_id": req.window_id,
                                                         "guest": P.process_identity(os.getpid())})
    P.write_json_atomic(wdir / P.DONE_FILE, {"window_id": req.window_id, "rc": 0})
    reason = _hold_reason(run_dir, ctl, req, int(TrainExitCode.INTERRUPTED))
    assert "777" in reason and "the guest exited 0" in reason


def test_a_dead_launchers_open_window_is_closed_by_the_next_launcher(run_dir, capsys):
    """A launcher that died mid-window left current.json OPEN: every later `arm` refused forever."""
    stale = {"window_id": "w_old", "phase": P.HOLD, "reason": "the guest exited rc 3",
             "launcher": {"pid": os.getpid(), "start_ticks": -5}}
    P.write_json_atomic(P.window_root(run_dir) / P.CURRENT_FILE, stale)
    events: list = []
    _controller(run_dir, events).announce()
    cur = P.read_current(run_dir)
    assert cur["phase"] == P.ABANDONED and cur["stale_launcher"] == stale["launcher"], cur
    assert any("its launcher is gone" in e for e in events)
    assert _arm(run_dir) == 0, capsys.readouterr().err
    # a window whose launcher is ALIVE is never touched
    ctl = _controller(run_dir, [])
    live = {**stale, "launcher": P.process_identity(os.getpid())}
    P.write_json_atomic(P.window_root(run_dir) / P.CURRENT_FILE, live)
    ctl.announce()
    assert P.read_current(run_dir)["phase"] == P.HOLD


def test_no_lease_check_is_refused_while_a_lease_is_held(run_dir, capsys, monkeypatch):
    """The opt-out is for a box with no lease; with one held it let any caller pause the holder's run."""
    import utils.gpu_lease as L

    class _Held:
        def describe(self):
            return "owner 'training' pid 1"
    monkeypatch.setattr(L, "read_lease", lambda lock: _Held())
    _controller(run_dir, []).announce()
    assert _arm(run_dir) == cli.EXIT_REFUSED
    assert "refused while a GPU lease is held" in capsys.readouterr().err
    assert not (P.window_root(run_dir) / P.REQUEST_FILE).exists()


# ── 2026-10-25: an operator's `resume` re-checks the GPU (T46 adversarial review) ──────────────────

def _resume_cmd(run_dir: Path) -> int:
    return cli.main(["resume", str(run_dir), "--reason", "test: the failure is understood"])


def _hold_then(run_dir: Path, ctl: WindowController, req: P.Request, script) -> str:
    """Run the window (entering a HOLD at once: the paused child 'crashed'); ``script(cur, step)`` plays the operator
    and returns True to quit. Returns the window's outcome."""
    step = [0]

    def should_quit() -> bool:
        cur = P.read_current(run_dir) or {}
        step[0] += 1
        return bool(script(cur, step[0]))
    return ctl.run_window(req, int(TrainExitCode.CRASH), time.time() - 1, should_quit)


def test_a_resume_waits_for_the_gpu_to_empty_and_names_the_pids_it_waits_on(run_dir):
    """The HOLD of an overrun or a dead guest is exactly when a guest process may still hold the card: the run must
    not re-launch onto it (OOM at the first update, 3 rapid-crash restarts burnt)."""
    polls: list = []

    def pids():
        polls.append(1)
        return [4242] if len(polls) <= 5 else []
    events: list = []
    ctl = _controller(run_dir, events, gpu_pids=pids, gpu_release_timeout_s=60.0)
    req = _request(run_dir, ctl)
    resumed, reasons = [], []

    def operator(cur, step):
        if cur.get("phase") == P.HOLD and not resumed:
            assert _resume_cmd(run_dir) == 0
            resumed.append(1)
        elif resumed:
            reasons.append(cur.get("reason", ""))
        return False
    assert _hold_then(run_dir, ctl, req, operator) == "resume"
    assert len(polls) == 6, "polled until the card was empty, then re-launched"
    assert any("4242" in r and "resume requested" in r for r in reasons), reasons
    assert any("4242" in e and "resume requested" in e for e in events), events
    assert P.read_current(run_dir)["phase"] == P.RESUMED


def test_a_resume_onto_a_busy_gpu_is_refused_and_the_hold_stays_until_a_new_resume_finds_it_empty(run_dir):
    state = {"pids": [4242]}
    events: list = []
    ctl = _controller(run_dir, events, gpu_pids=lambda: state["pids"], gpu_release_timeout_s=0.05)
    req = _request(run_dir, ctl)
    wdir = P.window_dir(run_dir, req.window_id)
    seen: dict = {}

    def operator(cur, step):
        reason = cur.get("reason", "")
        if cur.get("phase") == P.HOLD and "seen_first" not in seen:
            seen["seen_first"] = reason
            assert _resume_cmd(run_dir) == 0
        elif "resume REFUSED" in reason and "refused" not in seen:
            seen["refused"] = reason
            seen["phase"] = cur["phase"]
            seen["files"] = sorted(p.name for p in wdir.glob("RESUME*"))
            state["pids"] = []                              # the operator freed the card ...
            assert _resume_cmd(run_dir) == 0                # ... and resumes again
        return False
    assert _hold_then(run_dir, ctl, req, operator) == "resume"
    assert seen["phase"] == P.HOLD and "4242" in seen["refused"], seen
    assert len(seen["files"]) == 1 and seen["files"][0].startswith("RESUME.refused_"), \
        f"the refused RESUME must be set aside, not left to fire later: {seen['files']}"
    assert any("resume REFUSED" in e for e in events)


def test_a_resume_is_not_honoured_while_nvidia_smi_fails(run_dir):
    def failing():
        raise C.GpuQueryFailed("nvidia-smi exited rc 15")
    ctl = _controller(run_dir, [], gpu_pids=failing, gpu_release_timeout_s=0.05)
    req = _request(run_dir, ctl)
    seen: dict = {}

    def operator(cur, step):
        if cur.get("phase") == P.HOLD and "resumed" not in seen:
            seen["resumed"] = True
            assert _resume_cmd(run_dir) == 0
        return "resume REFUSED" in cur.get("reason", "")        # quit once the refusal is on record
    assert _hold_then(run_dir, ctl, req, operator) == "quit", "a failing nvidia-smi must not read as a free GPU"
    assert P.read_current(run_dir)["phase"] == P.ABANDONED


def test_a_resume_on_a_box_without_nvidia_smi_proceeds_and_says_it_did_not_check(run_dir):
    events: list = []
    ctl = _controller(run_dir, events, gpu_pids=lambda: None)
    req = _request(run_dir, ctl)

    def operator(cur, step):
        if cur.get("phase") == P.HOLD and step == 1:
            assert _resume_cmd(run_dir) == 0
        return False
    assert _hold_then(run_dir, ctl, req, operator) == "resume"
    assert any("nvidia-smi absent" in e for e in events)


def test_quitting_while_a_resume_waits_for_the_gpu_ends_the_window(run_dir):
    ctl = _controller(run_dir, [], gpu_pids=lambda: [4242], gpu_release_timeout_s=60.0)
    req = _request(run_dir, ctl)

    def operator(cur, step):
        if cur.get("phase") == P.HOLD and step == 1:
            assert _resume_cmd(run_dir) == 0
            return False
        return "resume requested" in cur.get("reason", "")       # the launcher is quitting mid-wait
    assert _hold_then(run_dir, ctl, req, operator) == "quit"
    assert P.read_current(run_dir)["phase"] == P.ABANDONED
