"""X14 A/B CELL (staged 2026-10-20): ONE arm of the `--overlap-collect` throughput read as a battery cell. The arm is a
FORK of X26 (`rb_x26_s2601` @ 380,000,195), 15 updates, run through the real launcher in this cell's OWN process
tree, at the battery's PIN. The cell is the 2026-10-20 battery's own GPU-window shape: `gpu_lock.sh mem_cap.sh timeout
python x14_cell.py ...`, one arm per cell, a wall cap each.

    python x14_cell.py --arm off|on --pin <sha> --out cells/x14_<arm>.json --battery-out <dir>      # the window
    python x14_cell.py --arm off|on --pin <sha> --check --scratch <dir> [--prove]                    # CPU, no GPU

**The recipe** (X14 README 3.1, `~/gen3ai_archive/x14_ab_runs/rb_x14ab_off_s2601`'s recorded command with the two
changes the 2026-10-20 deferral asked for -- the FP32_ALT pin and ``--behaviour-check warn``):

    --pin-commit <pin> --model <X26 @ 380,000,195> --steps 381,474,755 --device cuda --run-name <rb_x14ab2_{off,on}_s2601>
    --restart-interval-hours 24 --eval-freq 1000000000 --snapshot-ladder-games 0 --behaviour-check warn
    [--overlap-collect]                                                                  # the ON arm only

``--steps`` is ABSOLUTE: 380,000,195 + 15 x 98,304. Eval and ladder games are off. ``GEN3AI_MODELS_DIR`` is the scratch
archive (default ``~/gen3ai_archive/x14_ab_runs_2026-10-21``), never ``models/``. A run dir of the arm's name already
there (an earlier attempt of THIS battery) is MOVED ASIDE to ``<name>.prev-<time>``, never deleted.

**The pin** is the battery's (main HEAD when its first cell started, `battery_status.json`), recorded once in
``cells/x14_pin.json`` so the second arm cannot run at another commit. It MUST contain both commits the read depends on:
d4af9f1e (FP32_ALT: the learner update is ~42 s, not T62's ~97 s) and 02cac1a9 (the decoupled update normalises the RAW
advantage; ``--overlap-collect`` is a recorded recipe row).

**After the launcher exits** the cell reads the run (`x14_read.summarise`, read-only, CPU): the burn-in cycles, the
wall-fps, the overlap tags, and the K9(b) section-7 ceiling read (judged max >= 1e-3, p99 >= 1e-4, any non-finite value
=> a FLAG; ``warn`` mode means report, not abort: a flag does NOT fail the cell). The cell FAILS (the battery HOLDs) when
the launcher exits non-zero (11), or the run is not the arm that was specified (12: it stopped short of the target, the
ON arm never overlapped, the OFF arm did, or its recorded pin is not the battery's).

Exit: 0 ok · 3 a precondition (a required commit missing from the pin, no checkpoint, a bad archive dir, the arm's argv
does not launch) · 11 the launcher exited non-zero · 12 the run is not the specified arm · 143 stopped by SIGTERM.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import x14_read as XR

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SRC = ROOT / "src"

ARMS = ("off", "on")
PARENT_RUN = "rb_x26_s2601"
PARENT_CKPT = f"checkpoint_{XR.PARENT_STEP}_steps.zip"
RUN_NAMES = {"off": "rb_x14ab2_off_s2601", "on": "rb_x14ab2_on_s2601"}
DEFAULT_MODELS_DIR = Path.home() / "gen3ai_archive" / "x14_ab_runs_2026-10-21"
#: (what it carries, the full sha). The pin must contain both.
REQUIRED_COMMITS = (("FP32_ALT (gen3_tau_fp32_alt_v1)", "d4af9f1e26061681644473d22e2022dda2b43655"),
                    ("X14 norm-then-weight (gen3_overlap_norm_then_weight_v1)", "02cac1a98b8bf58bf26c2901c4588090a6d61e94"))
MIN_FREE_GIB = 14.0                       # the launcher's own DiskGuard says ~9 GiB REQUIRED at 15 updates; this leaves room
HEARTBEAT_S = 300
#: After the cell forwarded a SIGTERM, how long the launcher gets to save and tear its child down before the cell kills
#: its process group (the launcher's own child escalation is 150 s). The battery gives the cell `--kill-after=240`.
ESCALATE_S = float(os.environ.get("X14_ESCALATE_S", "200"))


def _py() -> str:
    return os.environ.get("PY") or os.environ.get("GEN3AI_PYTHON") or "/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3"


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=str(ROOT), capture_output=True, text=True, check=False)


def _main_checkout() -> Path:
    """The MAIN checkout (where `models/` lives), from git's common dir; this checkout if git cannot say."""
    q = _git("rev-parse", "--path-format=absolute", "--git-common-dir")
    return Path(q.stdout.strip()).parent if q.returncode == 0 and q.stdout.strip() else ROOT


def resolve_pin(spec: str) -> str:
    q = _git("rev-parse", "--verify", f"{spec}^{{commit}}")
    if q.returncode != 0 or not q.stdout.strip():
        raise ValueError(f"the pin {spec!r} does not resolve to a commit in {ROOT}: {q.stderr.strip()[:200]}")
    return q.stdout.strip()


def argv_for(arm: str, pin: str, ckpt: Path) -> list[str]:
    a = ["--pin-commit", pin, "--model", str(ckpt), "--steps", str(XR.TARGET_STEPS), "--device", "cuda",
         "--run-name", RUN_NAMES[arm], "--restart-interval-hours", "24", "--eval-freq", "1000000000",
         "--snapshot-ladder-games", "0", "--behaviour-check", "warn"]
    return a + (["--overlap-collect"] if arm == "on" else [])


def parent_checkpoint() -> Path | None:
    _src = str(SRC)
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from utils.paths import main_models_dir

    md = main_models_dir()
    if md is None:
        return None
    p = Path(md) / PARENT_RUN / "checkpoints" / PARENT_CKPT
    return p if p.is_file() else None


def archive_problem(models_dir: Path) -> str | None:
    """Why ``models_dir`` may not hold these runs (None = fine): never under any ``models/`` or ``data/``."""
    _src = str(SRC)
    if _src not in sys.path:
        sys.path.insert(0, _src)
    from utils.paths import main_models_dir

    md = Path(os.path.realpath(models_dir))
    bases = [ROOT, _main_checkout()]
    main = main_models_dir()
    if main is not None:
        bases.append(Path(main).parent)
    for base in bases:
        for bad in ("models", "data"):
            b = (base / bad).resolve()
            if md == b or b in md.parents:
                return f"{md} is under {b} (never write under models/ or data/)"
    if str(md).startswith(("/proc", "/sys", "/dev")):
        return f"{md} is not a data directory"
    return None


def preconditions(arm: str, pin_spec: str, models_dir: Path, battery_cells: Path | None) -> tuple[str, Path]:
    """Cheap, pure-git/filesystem preconditions; raises ``ValueError`` with the reason. Returns ``(pin, checkpoint)``."""
    pin = resolve_pin(pin_spec)
    for what, sha in REQUIRED_COMMITS:
        if _git("cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
            raise ValueError(f"commit {sha[:8]} ({what}) is not in this repository")
        if _git("merge-base", "--is-ancestor", sha, pin).returncode != 0:
            raise ValueError(f"the pin {pin[:8]} does not contain {sha[:8]} ({what}): a read at it is not the read "
                             "the deferral asked for")
    ckpt = parent_checkpoint()
    if ckpt is None:
        raise ValueError(f"the parent checkpoint {PARENT_RUN}/checkpoints/{PARENT_CKPT} is not in the main checkout's "
                         "models/ (the A/B forks it; it is read-only here)")
    bad = archive_problem(models_dir)
    if bad:
        raise ValueError(bad)
    if battery_cells is not None:
        rec = battery_cells / "x14_pin.json"
        if rec.is_file():
            have = json.loads(rec.read_text()).get("pin")
            if have != pin:
                raise ValueError(f"cells/x14_pin.json records pin {have}, this arm was given {pin}: the two arms of "
                                 "an A/B must run at ONE commit")
    return pin, ckpt


def _free_gib(path: Path) -> float:
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(p).free / 2**30


# ------------------------------------------------------------------------------------------------- --check

def check(arm: str, pin_spec: str, models_dir: Path, scratch: Path, battery_cells: Path | None,
          prove: bool = False) -> str:
    """CPU only (the caller exports CUDA_VISIBLE_DEVICES=""). Nothing is launched and nothing is created under the
    archive: the launcher's own --dry-run runs against an EMPTY SCRATCH dir (it refuses a $GEN3AI_MODELS_DIR that does
    not exist, and the real one is created by the run)."""
    pin, ckpt = preconditions(arm, pin_spec, models_dir, battery_cells)
    if _free_gib(models_dir) < MIN_FREE_GIB:
        raise ValueError(f"only {_free_gib(models_dir):.1f} GiB free at {models_dir} (< {MIN_FREE_GIB} GiB)")
    argv = argv_for(arm, pin, ckpt)
    env = {**os.environ, "PYTHONPATH": str(SRC), "CUDA_VISIBLE_DEVICES": ""}
    ca = subprocess.run([_py(), "-m", "main.checkargs", "--argv", shlex.join(argv)], env=env, cwd=str(ROOT),
                        capture_output=True, text=True, check=False, timeout=600)
    if ca.returncode != 0 or "this command still launches" not in ca.stdout:
        raise ValueError(f"checkargs refused the {arm} argv (rc {ca.returncode}): {(ca.stdout + ca.stderr)[-900:]}")
    scratch.mkdir(parents=True, exist_ok=True)
    dr = subprocess.run([_py(), "-m", "main.launcher", "--dry-run", *argv], cwd=str(ROOT), capture_output=True,
                        text=True, check=False, timeout=600, env={**env, "GEN3AI_MODELS_DIR": str(scratch)})
    if dr.returncode != 0 or "this command would launch" not in dr.stdout:
        raise ValueError(f"launcher --dry-run: the {arm} command would NOT launch (rc {dr.returncode}): "
                         f"{(dr.stdout + dr.stderr)[-1200:]}")
    st = XR.selftest()
    if prove:
        st += "; " + selftest_run(scratch / "selftest_run")
    prior = models_dir / RUN_NAMES[arm]
    moved = f"; the existing {prior.name} will be moved aside" if prior.exists() else ""
    return (f"pin {pin[:9]} carries {', '.join(sha[:8] for _w, sha in REQUIRED_COMMITS)}; checkpoint {ckpt.name}; "
            f"archive {models_dir}{moved}; checkargs ok; launcher --dry-run: would launch ({RUN_NAMES[arm]}, "
            f"{'--overlap-collect' if arm == 'on' else 'serial'}, --steps {XR.TARGET_STEPS:,}, "
            f"--behaviour-check warn, eval + ladder off); {st}")


# ------------------------------------------------------------------------------------------------- the run

def _write(out: Path, d: dict[str, Any]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=1, default=str))
    tmp.replace(out)


def run(a: argparse.Namespace, launcher: list[str] | None = None) -> int:
    """Run the arm. ``launcher`` is the command that is the launcher: None = the real one
    (``python -m main.launcher``); only `selftest_run` passes another (a FAKE that writes a synthetic run dir)."""
    launcher = launcher or [_py(), "-m", "main.launcher"]
    out = Path(a.out)
    models_dir = Path(a.models_dir).expanduser()
    cells = Path(a.battery_out) / "cells" if a.battery_out else out.parent
    try:
        pin, ckpt = preconditions(a.arm, a.pin, models_dir, cells)
    except ValueError as e:
        print(f"PRECONDITION FAILED: {e}", file=sys.stderr)
        return 3
    rec = cells / "x14_pin.json"
    if not rec.is_file():
        _write(rec, {"pin": pin, "first_arm": a.arm, "recorded": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "required": {w: sha for w, sha in REQUIRED_COMMITS}})
    models_dir.mkdir(parents=True, exist_ok=True)
    run_dir = models_dir / RUN_NAMES[a.arm]
    moved_to = None
    if run_dir.exists():
        moved_to = models_dir / f"{RUN_NAMES[a.arm]}.prev-{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir.rename(moved_to)
        print(f"[x14 {a.arm}] moved the existing run dir aside: {moved_to}", flush=True)
    argv = argv_for(a.arm, pin, ckpt)
    env = {**os.environ, "PYTHONPATH": str(SRC), "GEN3AI_MODELS_DIR": str(models_dir)}
    result: dict[str, Any] = {"cell": f"x14_{a.arm}", "arm": a.arm, "status": "running", "pin": pin,
                              "argv": argv, "models_dir": str(models_dir), "run_dir": str(run_dir),
                              "moved_aside": str(moved_to) if moved_to else None,
                              "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    _write(out, result)
    print(f"[x14 {a.arm}] launcher: {shlex.join([*launcher, *argv])}", flush=True)
    t0 = time.time()
    held: list[subprocess.Popen[bytes]] = []
    stopped: list[int] = []

    def forward(signum: int, _frame: Any) -> None:
        # `timeout` (the wall cap) / mem_cap / the operator signalled THIS cell: pass one SIGTERM to the launcher it
        # started, by its explicit PID (it saves and tears its child down); the cell then reports and exits 143.
        if not stopped:
            stopped.append(signum)
            if held:
                try:
                    os.kill(held[0].pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    for sg in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):          # BEFORE the launcher exists
        signal.signal(sg, forward)
    if stopped:                                                        # signalled during the setup above
        result.update(status="fail", reason=f"stopped by signal {stopped[0]} before the launcher started")
        _write(out, result)
        return 143
    proc = subprocess.Popen([*launcher, *argv], env=env, cwd=str(ROOT), stdin=subprocess.DEVNULL,
                            start_new_session=True)            # stdin off a TTY => the launcher runs headless
    held.append(proc)
    if stopped:                                                        # a signal landed between Popen and here
        os.kill(proc.pid, signal.SIGTERM)
    stop_t: list[float] = []
    killed = False
    last_beat = time.time()
    while True:
        try:
            rc = proc.wait(timeout=2)         # short, so a forwarded signal is acted on within seconds
            break
        except subprocess.TimeoutExpired:
            now = time.time()
            if stopped:
                stop_t = stop_t or [now]
                if not killed and now - stop_t[0] > ESCALATE_S:
                    killed = True             # the launcher ignored SIGTERM: its own process group, by explicit PGID
                    print(f"[x14 {a.arm}] the launcher did not exit {ESCALATE_S:.0f} s after SIGTERM: SIGKILL to its "
                          f"process group {proc.pid}", flush=True)
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            elif now - last_beat >= HEARTBEAT_S:
                last_beat = now
                print(f"[x14 {a.arm}] running {(now - t0) / 60:.0f} min (launcher pid {proc.pid})", flush=True)
    wall = (time.time() - t0) / 60
    result.update(launcher_rc=rc, wall_min=round(wall, 1), finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"[x14 {a.arm}] launcher exited rc {rc} after {wall:.1f} min; reading {run_dir}", flush=True)
    problems: list[str] = []
    try:
        summ = XR.summarise(str(run_dir), a.arm)
        result["summary"] = summ
        problems = XR.arm_problems(summ, a.arm, pin)
        result["k9b"] = summ["k9b"]
        if summ["k9b"].get("flag"):
            print(f"[x14 {a.arm}] FLAG: K9(b) section-7 ceiling read is {summ['k9b']['verdict']} "
                  f"(fired: {summ['k9b'].get('fired_families')}; warn mode: reported, not aborted)", flush=True)
    except Exception as exc:  # noqa: BLE001 -- the read is reported; the launcher's own outcome decides below
        result["read_error"] = f"{type(exc).__name__}: {exc}"
        problems.append(f"the post-run read raised {exc!r}")
    if stopped:
        result.update(status="fail", reason=f"stopped by signal {stopped[0]} (the wall cap)")
        _write(out, result)
        return 143
    if rc != 0:
        result.update(status="fail", reason=f"the launcher exited {rc}")
        _write(out, result)
        print(result["reason"], file=sys.stderr)
        return 11
    if problems:
        result.update(status="fail", reason="not the specified arm: " + "; ".join(problems))
        _write(out, result)
        print(result["reason"], file=sys.stderr)
        return 12
    result["status"] = "ok"
    _write(out, result)
    return 0


# ------------------------------------------------------------------------------------------------- --prove

FAKE_LAUNCHER = r'''
import os, signal, sys, time
from pathlib import Path
sys.path.insert(0, os.environ["X14_HERE"])
import x14_read as XR
mode = os.environ["X14_FAKE_MODE"]
a = sys.argv[1:]
get = lambda k: a[a.index(k) + 1]
name, pin = get("--run-name"), get("--pin-commit")
overlap = "--overlap-collect" in a
assert get("--behaviour-check") == "warn" and get("--steps") == str(XR.TARGET_STEPS) and get("--snapshot-ladder-games") == "0"
assert os.environ["GEN3AI_MODELS_DIR"] and os.environ["PYTHONPATH"]
if mode == "sleep":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    time.sleep(120)
if mode == "hang":
    signal.signal(signal.SIGTERM, signal.SIG_IGN)            # a launcher that will not stop: the cell must escalate
    time.sleep(300)
n = 6 if mode == "short" else 15
extra = {"overlap/join_wait_ms": 1.0, "overlap/play_ms": 40000.0} if (overlap and mode != "vacuous") or mode == "leak" else {}
dumps, t = [], 1000.0
for k in range(1, n + 1):
    t += 300.0 if k <= 3 else (80.0 if overlap else 100.0)
    tags = {"rollout/collect_ms": 30000.0, "train/train_ms": 40000.0, "time/fps": 400.0 + k,
            "lifecycle/cuda_update_peak_reserved_mib": 8300.0, **extra,
            "behaviour/max_abs_dlogp_judged": 2e-3 if mode == "flag" else 5e-5, "behaviour/p99_abs_dlogp_current": 1e-6}
    if overlap:
        tags.update({"behaviour_snapshot/max_abs_dlogp_judged": 6e-5, "behaviour_snapshot/p99_abs_dlogp_current": 2e-6})
    dumps.append((XR.PARENT_STEP + k * XR.ROWS_PER_UPDATE, t, tags))
XR._write_run(Path(os.environ["GEN3AI_MODELS_DIR"]), "x", [dumps], pin=("0" * 40 if mode == "wrongpin" else pin), name=name)
sys.exit(3 if mode == "rc3" else 0)
'''


def selftest_run(scratch: Path) -> str:
    """CPU PROOF of the cell's RUN machinery, with a FAKE launcher (a script that writes a synthetic run dir; nothing
    real is launched, nothing touches the GPU or models/): the launcher's argv and ``GEN3AI_MODELS_DIR`` arrive, an
    existing run dir is moved aside, exit codes 0 / 11 / 12 / 143, a K9(b) flag does not fail the cell, a second arm at
    another pin is refused (3). Each scenario is its own ``x14_cell.py`` process (the real cell is one process per arm)."""
    import shutil as _sh

    scratch.mkdir(parents=True, exist_ok=True)
    fake = scratch / "fake_launcher.py"
    fake.write_text(FAKE_LAUNCHER)
    got = []
    for scenario, arm, mode, want_rc, want_status in (
            ("complete OFF arm", "off", "ok", 0, "ok"), ("complete ON arm", "on", "ok", 0, "ok"),
            ("launcher exits 3", "off", "rc3", 11, "fail"), ("stops short of the target", "off", "short", 12, "fail"),
            ("ON arm that never overlaps", "on", "vacuous", 12, "fail"), ("OFF arm that overlaps", "off", "leak", 12, "fail"),
            ("run recorded another pin", "on", "wrongpin", 12, "fail"),
            ("K9(b) ceiling fires (warn: reported)", "off", "flag", 0, "ok"),
            ("wall cap (SIGTERM forwarded)", "off", "sleep", 143, "fail"),
            ("wall cap, a launcher that ignores SIGTERM", "off", "hang", 143, "fail")):
        d = scratch / scenario.split(" (")[0].replace(" ", "_").replace("'", "")
        _sh.rmtree(d, ignore_errors=True)
        (d / "cells").mkdir(parents=True)
        models = d / "archive"
        (models / RUN_NAMES[arm]).mkdir(parents=True)                    # a prior attempt: must be moved aside
        (models / RUN_NAMES[arm] / "marker").write_text("prior")
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), "--arm", arm, "--pin", "HEAD", "--out",
               str(d / "cells" / f"x14_{arm}.json"), "--battery-out", str(d), "--models-dir", str(models),
               "--launcher-cmd", shlex.join([sys.executable, str(fake)])]
        env = {**os.environ, "PYTHONPATH": str(SRC), "CUDA_VISIBLE_DEVICES": "", "X14_FAKE_MODE": mode,
               "X14_HERE": str(HERE), "X14_ESCALATE_S": "6"}
        p = subprocess.Popen(cmd, env=env, cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if mode in ("sleep", "hang"):
            time.sleep(8)
            os.kill(p.pid, signal.SIGTERM)                               # this selftest's own child, by PID
        try:
            text, _ = p.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            p.kill()
            raise
        res = json.loads((d / "cells" / f"x14_{arm}.json").read_text())
        assert p.returncode == want_rc and res["status"] == want_status, (scenario, p.returncode, res.get("status"),
                                                                           res.get("reason"), text[-1500:])
        assert (models / RUN_NAMES[arm]).is_dir() or mode in ("sleep", "hang"), scenario
        if mode == "hang":
            assert "SIGKILL to its process group" in text, text[-800:]
        assert res["moved_aside"] and Path(res["moved_aside"], "marker").read_text() == "prior", scenario
        if mode == "flag":
            assert res["k9b"]["flag"] and res["k9b"]["verdict"] == "VOID", res["k9b"]
        if mode == "ok":
            assert not res["k9b"]["flag"] and res["summary"]["n_cycles"] == 12, res["summary"]["n_cycles"]
            assert json.loads((d / "cells" / "x14_pin.json").read_text())["pin"] == resolve_pin("HEAD")
        got.append(f"{scenario} -> exit {p.returncode}")
    # the second arm at ANOTHER pin is refused before anything runs (the pin record is the first arm's)
    d = scratch / "pin_record"
    _sh.rmtree(d, ignore_errors=True)
    (d / "cells").mkdir(parents=True)
    (d / "cells" / "x14_pin.json").write_text(json.dumps({"pin": "d" * 40}))
    q = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--arm", "on", "--pin", "HEAD", "--out",
                        str(d / "cells" / "x14_on.json"), "--battery-out", str(d), "--models-dir", str(d / "archive"),
                        "--launcher-cmd", shlex.join([sys.executable, str(fake)])],
                       env={**os.environ, "PYTHONPATH": str(SRC), "CUDA_VISIBLE_DEVICES": "", "X14_FAKE_MODE": "ok",
                            "X14_HERE": str(HERE)}, cwd=str(ROOT), capture_output=True, text=True, check=False)
    assert q.returncode == 3 and "ONE commit" in q.stderr and not (d / "archive").exists(), (q.returncode, q.stderr)
    got.append("a second arm at another pin -> exit 3, nothing created")
    return "cell run machinery ok on a FAKE launcher: " + "; ".join(got)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--pin", required=True, help="the battery's pin (a commit-ish; resolved to a full sha)")
    ap.add_argument("--out", help="the cell result json (cells/x14_<arm>.json)")
    ap.add_argument("--battery-out", help="the battery's --out dir (cells/x14_pin.json lives under it)")
    ap.add_argument("--models-dir", default=os.environ.get("GEN3AI_X14_MODELS_DIR", str(DEFAULT_MODELS_DIR)))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--prove", action="store_true",
                    help="with --check: also run the run machinery against a FAKE launcher (~1-2 min on the CPU); the "
                         "battery's --dry-run does, the window's own check does not")
    ap.add_argument("--scratch", help="--check: an empty scratch dir for the launcher's --dry-run")
    ap.add_argument("--launcher-cmd", default=None, help=argparse.SUPPRESS)      # test seam: the FAKE launcher only
    a = ap.parse_args(argv)
    if a.check:
        if not a.scratch:
            ap.error("--check needs --scratch")
        cells = Path(a.battery_out) / "cells" if a.battery_out else None
        try:
            print("ok: " + check(a.arm, a.pin, Path(a.models_dir).expanduser(), Path(a.scratch), cells, a.prove))
            return 0
        except (ValueError, AssertionError, subprocess.TimeoutExpired) as e:
            print(f"PRECONDITION FAILED: {e}", file=sys.stderr)
            return 3
    if not a.out:
        ap.error("--out is required (or --check)")
    return run(a, shlex.split(a.launcher_cmd) if a.launcher_cmd else None)


if __name__ == "__main__":
    sys.exit(main())
