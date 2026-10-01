"""`gen3_supply_guard_v1` — a live cf coefficient can never again train on a supply that is absent.

The class: `ai_v12_12_ladder_cflabels` ran 10M steps at `--cf-winprob-coef 0.5` with ZERO labels —
the producer is a separate program nobody started, and nothing thresholded the counters that
could have said so. Every test below fails on revert of the part it names:

* STARTUP — the trainer STARTS the producer (a real `cf_producer` process, holding the run's lock,
  bound to its parent), or REFUSES: no `cf_records/` ring, an `external` supply nobody runs, a second
  producer on the lock;
* PARENT DEATH — a trainer that dies by `os._exit` takes its spawned producer with it (real
  processes, PR_SET_PDEATHSIG);
* IN FLIGHT — zero ingest across N cycles (and T minutes) raises the typed FATAL, which maps to exit
  5, which the launcher does not restart; a producer that exits is FATAL at once; a HEALTHY producer
  (a real writer process shipping real rows through the real buffer) passes;
* END OF RUN — zero arrivals print LOUD.
"""
import argparse
import os
import signal
import subprocess
import sys
import textwrap
import time
from types import SimpleNamespace

import numpy as np
import pytest

from agents.training import cf_supply
from agents.training.cf_label_buffer import CfLabelBuffer
from agents.training.cf_supply import (CF_CONSUMER_COEFS, CfLabelSupplyError, CfProducerSupply,
                                       CfSupplyConfigError, CfSupplyGuard, acquire_producer_lock,
                                       live_cf_consumers, live_producer_pid,
                                       preflight_cf_label_supply, producer_args,
                                       start_cf_label_supply)
from agents.training.cf_supply_callback import CfSupplyCallback
from main.exit_codes import SupplyStarvedError, TrainExitCode, exit_code_for
from utils.contention import scale_timeout
from utils.paths import src_root

_OBS_DIM = 8


def _args(**kw):
    base = {c: 0.0 for c in CF_CONSUMER_COEFS}
    base.update(cf_label_supply="producer", cf_producer_args=None, cf_records=True)
    base.update(kw)
    return argparse.Namespace(**base)


def _row_file(labels_dir, name, n=1, *, policy_step=1000, q=False, fill=1.0):
    """Real v1 label rows through the producer's own batch writer (tmp + rename)."""
    import base64
    import hashlib
    import json
    os.makedirs(labels_dir, exist_ok=True)
    tmp = os.path.join(labels_dir, name + ".tmp")
    with open(tmp, "w") as f:
        for i in range(n):
            raw = np.full(_OBS_DIM, fill + i, dtype=np.float32).tobytes()
            row = {"schema": 1, "kind": "mc_winprob", "battle": "b", "decision_idx": i,
                   "obs_sha1": hashlib.sha1(raw).hexdigest(), "obs_npz": None,
                   "obs_inline": base64.b64encode(raw).decode(), "label": 0.5, "n_rollouts": 8,
                   "wilson_lo": 0.2, "wilson_hi": 0.8, "policy_step": policy_step,
                   "opponent": "self_current", "created_unix": 1.0}
            if q:
                row.update(q_labels=[{"action": 1, "label": 0.5, "n_rollouts": 4}],
                           taken_action=1, outcome_label=1.0)
            f.write(json.dumps(row) + "\n")
    os.replace(tmp, os.path.join(labels_dir, name))


class _Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _wait(pred, timeout_s, what):
    deadline = time.monotonic() + scale_timeout(timeout_s)
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.2)
    raise AssertionError(f"timed out after {scale_timeout(timeout_s):.0f}s waiting for {what}")


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # A zombie still answers kill(0); read its state.
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(") ", 1)[1][0] != "Z"
    except OSError:
        return False


def _lock_holders(rd):
    """Every process with the run's lock file open — the diagnosis when a lock outlives its pid."""
    out = []
    target = os.path.realpath(os.path.join(rd, cf_supply.LOCK_NAME))
    for pid in (p for p in os.listdir("/proc") if p.isdigit()):
        try:
            for fd in os.listdir(f"/proc/{pid}/fd"):
                if os.path.realpath(f"/proc/{pid}/fd/{fd}") == target:
                    with open(f"/proc/{pid}/cmdline", "rb") as f:
                        out.append((int(pid), f.read().replace(b"\0", b" ")[:160].decode()))
        except OSError:
            continue
    return out


# ── which coefficients need a supply ─────────────────────────────────────────

def test_every_cf_buffer_consumer_is_one_list_and_the_parser_check_agrees():
    from main.train import combination_checks
    assert combination_checks._CF_CONSUMER_COEFS == CF_CONSUMER_COEFS
    assert set(cf_supply.STREAM_OF) == set(CF_CONSUMER_COEFS)
    assert live_cf_consumers(_args()) == []
    assert live_cf_consumers(_args(cf_winprob_coef=0.5, q_winprob_onpolicy_coef=0.1)) == [
        "cf_winprob_coef", "q_winprob_onpolicy_coef"]


def test_a_live_q_coefficient_asks_the_producer_for_the_q_stream():
    assert producer_args(_args(cf_winprob_coef=0.5)) == []
    assert producer_args(_args(q_winprob_coef=0.2)) == ["--q-labels"]
    assert producer_args(_args(q_winprob_onpolicy_coef=0.2,
                               cf_producer_args="--rollouts 16 --top-n 4")) == [
        "--q-labels", "--rollouts", "16", "--top-n", "4"]


def test_the_parser_refuses_a_live_coefficient_whose_producer_would_have_no_ring():
    """The launch-path CombinationCheck: FATAL_CONFIG, naming the supplier and the fix."""
    from main.train.combination_checks import COMBINATION_CHECKS
    chk = next(c for c in COMBINATION_CHECKS if c.name == "cf_consumer_needs_label_supply")
    assert chk.exit_style == "fatal_config"
    bad = _args(cf_winprob_coef=0.5, cf_records=False)
    assert chk.predicate(bad)
    assert "--cf-records" in chk.text(bad) and "cf_winprob_coef" in chk.text(bad)
    assert not chk.predicate(_args(cf_winprob_coef=0.5, cf_records=True))
    assert not chk.predicate(_args(cf_records=False))                    # nothing live
    assert not chk.predicate(_args(cf_evidential_coef=0.05, cf_records=False,
                                   cf_label_supply="external"))      # checked at the lock instead


# ── the exit code ────────────────────────────────────────────────────────────

def test_a_starved_supply_maps_to_its_own_fatal_code_and_the_launcher_stops():
    from main.launcher.run import _fatal_config_reason
    assert int(TrainExitCode.FATAL_SUPPLY) == 5
    assert len({int(c) for c in TrainExitCode}) == len(TrainExitCode)
    assert issubclass(CfLabelSupplyError, SupplyStarvedError)
    assert exit_code_for(CfLabelSupplyError("starved")) == 5
    try:
        try:
            raise CfLabelSupplyError("starved")
        except CfLabelSupplyError as inner:
            raise RuntimeError("callback wrapper") from inner
    except RuntimeError as outer:
        assert exit_code_for(outer) == 5
    reason = _fatal_config_reason(5, ["x", "agents.training.cf_supply.CfLabelSupplyError: "
                                           "[SUPPLY] FATAL: cf_winprob_coef is live"])
    assert reason and "supply" in reason[0] and "[SUPPLY]" in reason[-1]


def test_the_launcher_never_restarts_a_starved_supply(tmp_path):
    from main.launcher.nonfinite_exit_test import _drive
    code, spawned, events = _drive(tmp_path, int(TrainExitCode.FATAL_SUPPLY))
    assert code == 5 and spawned == 1, (code, spawned)
    assert any("Starved supply — will NOT restart" in e for e in events), events


def test_a_fresh_runs_learn_crash_exits_through_the_mapping():
    """`build_and_train`'s FRESH-path handler used to `os._exit(1)` — a FATAL raised inside a fresh
    run's learn() (this guard's, or the non-finite learner's) read as a restartable CRASH."""
    import ast
    from utils.paths import src_path
    tree = ast.parse(src_path("main", "train", "model_build.py").read_text())
    # The handlers around `model.learn(...)` — the ones that catch whatever learn() raised.
    def _calls_learn(nodes):
        return any(isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute)
                   and c.func.attr == "learn" for n in nodes for c in ast.walk(n))
    handlers = [h for t in ast.walk(tree) if isinstance(t, ast.Try) and _calls_learn(t.body)
                for h in t.handlers]
    assert len(handlers) == 2, "the fresh and the resume learn() handlers"
    exits = [c for h in handlers for c in ast.walk(h) if isinstance(c, ast.Call)
             and isinstance(c.func, ast.Attribute) and c.func.attr == "_exit"]
    assert exits
    for c in exits:
        arg = c.args[0]
        assert isinstance(arg, ast.Call) and getattr(arg.func, "id", "") == "exit_code_for", (
            f"model_build exits {ast.unparse(arg)} — a supply FATAL would be restarted")


# ── the in-flight guard ──────────────────────────────────────────────────────

def _guard(consumers=("cf_winprob_coef",), cycles=3, minutes=10.0, supply=None):
    clk = _Clock()
    return CfSupplyGuard(consumers, starve_cycles=cycles, starve_minutes=minutes, supply=supply,
                         clock=clk), clk


def test_zero_ingest_across_n_cycles_raises_the_typed_fatal(tmp_path):
    buf = CfLabelBuffer(tmp_path / "cf_labels", obs_dim=_OBS_DIM)
    g, clk = _guard()
    g.observe(buf, checkpoint_present=True)                  # arms
    for i in range(2):
        clk.t += 600
        buf.poll(1000)
        g.observe(buf, checkpoint_present=True)
    clk.t += 600
    with pytest.raises(CfLabelSupplyError) as ei:
        g.observe(buf, checkpoint_present=True)
    msg = str(ei.value)
    assert "cf_winprob_coef" in msg and "NONE for 3 consecutive" in msg
    assert "NO usable row arrived" in msg
    assert exit_code_for(ei.value) == 5


def test_the_guard_is_disarmed_until_a_checkpoint_exists(tmp_path):
    """The producer stamps each row with the newest checkpoint's step, so before one exists there
    is nothing it CAN deliver — starving then is not a supply failure."""
    buf = CfLabelBuffer(tmp_path / "cf_labels", obs_dim=_OBS_DIM)
    g, clk = _guard(cycles=1, minutes=0.0)
    for _ in range(50):
        clk.t += 3600
        g.observe(buf, checkpoint_present=False)
    assert not g.armed


def test_both_floors_must_be_exceeded(tmp_path):
    """Cycles alone would trip a fast-iterating run before a healthy producer's first labels."""
    buf = CfLabelBuffer(tmp_path / "cf_labels", obs_dim=_OBS_DIM)
    g, clk = _guard(cycles=3, minutes=10.0)
    g.observe(buf, checkpoint_present=True)
    for _ in range(20):                                       # many cycles, little wall
        clk.t += 1
        g.observe(buf, checkpoint_present=True)
    clk.t += 600
    with pytest.raises(CfLabelSupplyError):
        g.observe(buf, checkpoint_present=True)


def test_an_arrival_resets_the_starvation_clock(tmp_path):
    labels = str(tmp_path / "cf_labels")
    buf = CfLabelBuffer(labels, obs_dim=_OBS_DIM)
    g, clk = _guard(cycles=3, minutes=10.0)
    g.observe(buf, checkpoint_present=True)
    for k in range(10):
        clk.t += 600
        if k % 2 == 0:
            _row_file(labels, f"labels_x_{k}.jsonl", fill=float(k * 10))
        buf.poll(1000)
        g.observe(buf, checkpoint_present=True)               # never 3 dry cycles in a row
    assert g.starved_cycles() <= 1 and buf.ingested_total == 5


def test_rows_that_all_expire_are_starvation_and_the_message_says_why(tmp_path):
    labels = str(tmp_path / "cf_labels")
    buf = CfLabelBuffer(labels, obs_dim=_OBS_DIM, lag_bound=100)
    g, clk = _guard(cycles=2, minutes=0.0)
    g.observe(buf, checkpoint_present=True)
    with pytest.raises(CfLabelSupplyError) as ei:
        for k in range(3):
            _row_file(labels, f"labels_old_{k}.jsonl", policy_step=0, fill=float(k * 10))
            buf.poll(1_000_000)
            clk.t += 60
            g.observe(buf, checkpoint_present=True)
    assert buf.expired_total > 0 and "EXPIRED at ingest" in str(ei.value)


def test_a_q_coefficient_is_guarded_on_its_own_stream(tmp_path):
    """Rows without `q_labels` keep `ingested_total` rising while the Q head folds nothing."""
    labels = str(tmp_path / "cf_labels")
    buf = CfLabelBuffer(labels, obs_dim=_OBS_DIM)
    g, clk = _guard(consumers=("cf_winprob_coef", "q_winprob_coef"), cycles=2, minutes=0.0)
    g.observe(buf, checkpoint_present=True)
    with pytest.raises(CfLabelSupplyError) as ei:
        for k in range(3):
            _row_file(labels, f"labels_noq_{k}.jsonl", fill=float(k * 10))
            buf.poll(1000)
            clk.t += 60
            g.observe(buf, checkpoint_present=True)
    msg = str(ei.value)
    assert "q_winprob_coef" in msg and "q_ingested_total" in msg and "--q-labels" in msg
    # ...and the same shape WITH q rows passes.
    labels2 = str(tmp_path / "cf_labels2")
    buf2 = CfLabelBuffer(labels2, obs_dim=_OBS_DIM)
    g2, clk2 = _guard(consumers=("q_winprob_coef", "q_winprob_onpolicy_coef"), cycles=2,
                      minutes=0.0)
    g2.observe(buf2, checkpoint_present=True)
    for k in range(6):
        _row_file(labels2, f"labels_q_{k}.jsonl", q=True, fill=float(k * 10))
        buf2.poll(1000)
        clk2.t += 60
        g2.observe(buf2, checkpoint_present=True)
    assert buf2.q_ingested_total == 6 and buf2.onpolicy_ingested_total == 6


def test_zero_disables_the_guard_but_the_summary_is_still_loud(tmp_path):
    buf = CfLabelBuffer(tmp_path / "cf_labels", obs_dim=_OBS_DIM)
    g, clk = _guard(cycles=0, minutes=0.0)
    for _ in range(10):
        clk.t += 3600
        g.observe(buf, checkpoint_present=True)
    lines = g.summary(buf)
    assert len(lines) == 1 and "ZERO cf labels ARRIVED" in lines[0] and "enabled=False" in lines[0]


def test_a_spawned_producer_that_exits_is_fatal_at_once(tmp_path):
    """A REAL child that dies: the next observe raises with the tail of its log."""
    log = tmp_path / "cf_producer.log"
    with open(log, "wb") as f:
        proc = subprocess.Popen([sys.executable, "-c",
                                 "import sys; print('anchor refused: boom'); sys.exit(3)"],
                                stdout=f, stderr=subprocess.STDOUT)
    proc.wait(timeout=scale_timeout(60))
    sup = CfProducerSupply(run_dir=str(tmp_path), mode="producer",
                           consumers=("cf_winprob_coef",), proc=proc, log_path=str(log))
    g, _clk = _guard(supply=sup)
    buf = CfLabelBuffer(tmp_path / "cf_labels", obs_dim=_OBS_DIM)
    with pytest.raises(CfLabelSupplyError) as ei:
        g.observe(buf, checkpoint_present=False)             # even before arming
    assert "EXITED with code 3" in str(ei.value) and "anchor refused: boom" in str(ei.value)


# ── the callback (what learn() calls) ────────────────────────────────────────

class _Logger:
    def __init__(self):
        self.rec = {}

    def record(self, k, v):
        self.rec[k] = v


def test_the_callback_judges_each_completed_train_and_raises_out_of_learn(tmp_path):
    buf = CfLabelBuffer(tmp_path / "cf_labels", obs_dim=_OBS_DIM)
    model = SimpleNamespace(_cf_buffer=buf, logger=_Logger(), num_timesteps=0)
    g, clk = _guard(cycles=2, minutes=0.0)
    said = []
    cb = CfSupplyCallback(g, run_dir=str(tmp_path), checkpoint_probe=lambda _d: True,
                          emit=said.append)
    cb.init_callback(model)
    cb.on_training_start({}, {})
    cb.on_rollout_start()                        # before any train(): nothing judged
    assert not g.armed
    cb.on_rollout_start()                        # arms
    cb.on_rollout_start()
    assert model.logger.rec["cf/supply_starved_cycles"] == 1.0
    with pytest.raises(CfLabelSupplyError):
        cb.on_rollout_start()
    cb.on_training_end()
    assert any("ZERO cf labels ARRIVED" in s for s in said)


# ── real processes: the producer as a declared startup resource ─────────────

def _run_dir(tmp_path):
    rd = tmp_path / "run"
    (rd / "cf_records").mkdir(parents=True)
    return str(rd)


@pytest.mark.integration
def test_the_trainer_starts_a_real_producer_that_holds_the_lock(tmp_path):
    """`producer` mode: a REAL `cf_producer` process, bound to us, holding the run's lock, logging
    to the run dir — and a second producer on the same run refuses (exit 4)."""
    rd = _run_dir(tmp_path)
    said = []
    sup = start_cf_label_supply(_args(cf_winprob_coef=0.5), rd, emit=said.append)
    try:
        assert sup is not None and sup.proc is not None and sup.mode == "producer"
        assert "--parent-pid" in sup.argv and str(os.getpid()) in sup.argv
        _wait(lambda: live_producer_pid(rd) == sup.proc.pid or sup.proc.poll() is not None,
              120, "the producer to take its lock")
        assert sup.proc.poll() is None, open(sup.log_path).read()[-2000:]
        assert live_producer_pid(rd) == sup.proc.pid
        sup.check_alive()
        second = subprocess.run(
            [sys.executable, "-m", "agents.training.cf_producer", rd, "--cycles", "1"],
            capture_output=True, text=True, timeout=scale_timeout(180),
            env={**os.environ, "PYTHONPATH": str(src_root())})
        assert second.returncode == cf_supply.PRODUCER_EXIT_LOCK_HELD, second.stderr[-2000:]
        assert f"pid {sup.proc.pid}" in second.stderr
        # A trainer relaunched while this one runs refuses rather than start a second producer.
        with pytest.raises(CfSupplyConfigError, match="already holds"):
            start_cf_label_supply(_args(cf_winprob_coef=0.5), rd, lock_wait_s=0.5)
        # `external` is SATISFIED by a live lock holder.
        ext = start_cf_label_supply(_args(cf_winprob_coef=0.5, cf_label_supply="external"), rd)
        assert ext is not None and ext.external_pid == sup.proc.pid and ext.proc is None
    finally:
        sup.stop()
    assert sup.proc.returncode is not None
    assert live_producer_pid(rd) is None                     # the kernel dropped the flock
    assert "spawned by trainer pid" in open(sup.log_path).read()
    assert any("[SUPPLY]" in s for s in said)


def test_startup_refusals_name_the_supplier_and_the_command(tmp_path):
    # no ring → nothing to label
    rd = tmp_path / "noring"
    rd.mkdir()
    with pytest.raises(CfSupplyConfigError, match="cf_records"):
        start_cf_label_supply(_args(cf_winprob_coef=0.5), str(rd))
    # external, nobody running → refused BEFORE the dir need exist, with the exact command
    missing = tmp_path / "not_yet_created"
    with pytest.raises(CfSupplyConfigError) as ei:
        preflight_cf_label_supply(_args(cf_twin_coef=0.1, q_winprob_coef=0.2,
                                        cf_label_supply="external"), str(missing))
    msg = str(ei.value)
    assert "agents.training.cf_producer" in msg and "--q-labels" in msg and str(missing) in msg
    assert not missing.exists()
    # nothing live → no supply, no process, no files
    rd2 = tmp_path / "off"
    rd2.mkdir()
    assert start_cf_label_supply(_args(), str(rd2)) is None
    assert os.listdir(rd2) == []


def test_a_stale_lock_file_without_a_holder_is_not_a_producer(tmp_path):
    rd = _run_dir(tmp_path)
    (tmp_path / "run" / cf_supply.LOCK_NAME).write_text("12345\n")   # a dead producer's file
    assert live_producer_pid(rd) is None
    fd = acquire_producer_lock(rd)
    try:
        assert fd is not None and live_producer_pid(rd) == os.getpid()
    finally:
        os.close(fd)
    assert live_producer_pid(rd) is None


_PARENT = textwrap.dedent("""
    import argparse, os, sys, time
    from agents.training.cf_supply import start_cf_label_supply, CF_CONSUMER_COEFS
    a = argparse.Namespace(**{c: 0.0 for c in CF_CONSUMER_COEFS})
    a.cf_winprob_coef = 0.5; a.cf_label_supply = "producer"; a.cf_producer_args = None
    sup = start_cf_label_supply(a, sys.argv[1], emit=lambda s: None)
    print(sup.proc.pid, flush=True)
    sys.stdin.readline()          # the test says when
    os._exit(1)                   # die the way a crashing trainer does: no atexit, no finally
""")


@pytest.mark.integration
def test_a_producer_dies_with_the_trainer_that_spawned_it(tmp_path):
    """PR_SET_PDEATHSIG: a trainer that `os._exit`s (its fail-fast path) leaves no producer behind
    — otherwise the next launcher segment would find the lock held, or two producers would share
    one state file."""
    rd = _run_dir(tmp_path)
    parent = subprocess.Popen([sys.executable, "-c", _PARENT, rd], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, text=True,
                              env={**os.environ, "PYTHONPATH": str(src_root())})
    try:
        child_pid = int(parent.stdout.readline().strip())
        _wait(lambda: live_producer_pid(rd) == child_pid or not _pid_alive(child_pid), 120,
              "the spawned producer to take its lock")
        assert _pid_alive(child_pid), open(os.path.join(rd, "cf_producer.log")).read()[-2000:]
        parent.stdin.write("go\n")
        parent.stdin.flush()
        parent.wait(timeout=scale_timeout(60))
        _wait(lambda: not _pid_alive(child_pid), 60, "the orphaned producer to die")
        _wait(lambda: live_producer_pid(rd) is None, 30, "the kernel to drop the dead producer's flock")
        assert live_producer_pid(rd) is None, (child_pid, _lock_holders(rd))
    finally:
        if parent.poll() is None:
            parent.kill()
        try:
            if _pid_alive(child_pid):
                os.kill(child_pid, signal.SIGKILL)
        except (NameError, ProcessLookupError):
            pass


_HEALTHY = textwrap.dedent("""
    import base64, hashlib, json, os, sys, time
    import numpy as np
    from agents.training.cf_producer_labels import write_label_batch
    labels, n = sys.argv[1], int(sys.argv[2])
    for seq in range(n):
        raw = np.full(8, float(seq), dtype=np.float32).tobytes()
        row = {"schema": 1, "kind": "mc_winprob", "battle": "b", "decision_idx": seq,
               "obs_sha1": hashlib.sha1(raw).hexdigest(), "obs_npz": None,
               "obs_inline": base64.b64encode(raw).decode(), "label": 0.5, "n_rollouts": 8,
               "wilson_lo": 0.2, "wilson_hi": 0.8, "policy_step": 1000,
               "opponent": "self_current", "created_unix": 1.0}
        write_label_batch(labels, [row], step=1000, seq=seq)
        print(seq, flush=True)
        sys.stdin.readline()      # one batch per trainer cycle
""")


@pytest.mark.integration
def test_a_healthy_producer_passes(tmp_path):
    """A real writer PROCESS shipping one real batch per cycle through the producer's own writer:
    the guard at its tightest floor (1 cycle, 0 min) never trips, and the summary reports it."""
    labels = str(tmp_path / "cf_labels")
    n = 8
    writer = subprocess.Popen([sys.executable, "-c", _HEALTHY, labels, str(n)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                              env={**os.environ, "PYTHONPATH": str(src_root())})
    try:
        buf = CfLabelBuffer(labels, obs_dim=_OBS_DIM)
        g, clk = _guard(cycles=1, minutes=0.0)
        g.observe(buf, checkpoint_present=True)
        for _ in range(n):
            assert writer.stdout.readline().strip() != ""
            buf.poll(1000)
            clk.t += 600
            g.observe(buf, checkpoint_present=True)
            writer.stdin.write("\n")
            writer.stdin.flush()
        writer.wait(timeout=scale_timeout(60))
        assert buf.ingested_total == n and g.starved_cycles() == 0
        assert "rows accepted this segment" in g.summary(buf)[0]
    finally:
        if writer.poll() is None:
            writer.kill()
