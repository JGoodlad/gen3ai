"""THE FP32_ALT GPU-WINDOW BATTERY (staged 2026-10-20): FIRST FP32_ALT's tau on the card (``tau_cuda_parity``: do the
compiled CUDA rollout and the CUDA learner land on one tau, REVIEW.md F2; a miss HOLDS the battery), then FP32_ALT's GPU
cost against the pre-T62 naive fp32 τ solve, then the X14 overlap A/B (two forks of X26 at the FP32_ALT pin), one cell at a time, in a short window before X35's
production arms. README.md has the order, the costs and the decision rules. A copy of the 2026-10-19 battery's driver
(`../gpu_window_battery_2026-10-19/battery.py`) with its cells changed, plus: the learner cells' FAIRNESS rule
(`learner_path.py`: both τ arms run the identical compiled path or the cell fails; ``--learner-gate off`` makes it
identical by construction) and the battery PIN (main HEAD when the window opened; ``--pin`` overrides) the X14 arms run at.

    # the training session (it holds the GPU lease), from the MAIN checkout, the run PAUSED and the GPU clear:
    export GEN3AI_GPU_LEASE_TOKEN_FILE=<the lease token file>     # or GEN3AI_GPU_LEASE_TOKEN=<token>
    designs/research_state/measurements/gpu_window_battery_2026-10-20/run_battery.sh --out <dir> [--budget-min N]
    designs/research_state/measurements/gpu_window_battery_2026-10-20/run_battery.sh --out <dir> --dry-run

INCREMENTAL + RESUMABLE: a cell whose ``cells/<name>.state.json`` reads ``ok`` (or ``skip``) is skipped on a re-run.
``--only a,b`` runs named cells; ``--budget-min N`` stops CLEANLY (status ``BUDGET_STOP``, exit 10) before a cell whose
ETA would overrun N minutes since the start -- re-run to resume.

HOLDS on any failure (TRAINING_RUN_SOP §3 window-script rule 3): a cell that exits non-zero, times out, is OOM-killed
at its cap, writes no ``ok`` result, or finds another process on the GPU -> ``battery_status.json`` reads ``HOLD`` with
the reason, the log line is ``HOLD: ...``, and the battery exits 1 WITHOUT running the next cell. It never stops,
resumes or signals a training run: the caller owns the pause / resume.

Every GPU cell runs as ``gpu_lock.sh mem_cap.sh --name <cell> <GB> timeout --kill-after=60 <cap> <cmd>``.
``--dry-run`` runs every cell's precondition check on the CPU with ``CUDA_VISIBLE_DEVICES=""`` and prints every
command; nothing touches the GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import learner_path as LP  # noqa: E402

ROOT = HERE.parents[3]          # designs/research_state/measurements/<this> -> the checkout
OPS = ROOT / "scripts" / "ops"
SRC = ROOT / "src"

#: The learner pairs, run ABBA (old, new, new, old) so a slow drift over the window cancels in the mean and shows in
#: the two pairs' disagreement, which the reader turns into the decision band. old = fp32_naive, new = fp32_alt.
TAU_UPDATE_CELLS = (("tau_u_naive_a", "fp32_naive"), ("tau_u_alt_a", "fp32_alt"),
                    ("tau_u_alt_b", "fp32_alt"), ("tau_u_naive_b", "fp32_naive"))


class Cell:
    """One measurement: its command, its caps and its precondition (pure description; `battery` runs it)."""

    def __init__(self, name: str, what: str, mem_gb: int, wall_min: int, eta_min: int,
                 cmd: Callable[[Path], list[str]], check: Callable[[Path], tuple[str, str]],
                 result: Callable[[Path], tuple[bool, str]], prepare: Callable[[Path], None] | None = None,
                 dry_check: Callable[[Path], tuple[str, str]] | None = None, gate_mode: str | None = None,
                 kill_after_s: int = 60):
        self.name, self.what, self.mem_gb, self.wall_min, self.eta_min = name, what, mem_gb, wall_min, eta_min
        self.cmd, self.check, self.result, self.prepare = cmd, check, result, prepare
        #: the heavier check ONLY the dry run makes (the learner cells' CPU compiled-path proof, ~5 min): the window's
        #: own check is the light one, so the GPU is not idle for it. None = the same check.
        self.dry_check = dry_check or check
        #: a learner cell's ``--learner-gate`` mode; a recorded ``ok`` of another mode is NOT reusable
        self.gate_mode = gate_mode
        #: `timeout --kill-after`: the grace between the wall cap's SIGTERM and SIGKILL. The X14 cells get 240 s: the
        #: cell forwards the TERM to the launcher, which saves and tears its child down (its own escalation is 150 s)
        self.kill_after_s = kill_after_s


#: A pytest cell run from the MAIN checkout must not let the conftest's slow-tier recorder dirty main mid-window.
#: The dry run asserts it for every pytest cell (``t2_guard``).
NO_SLOW_RECORD = "GEN3AI_SKIP_SLOW_STATUS_RECORD=1"


def _py() -> str:
    return os.environ.get("PY") or os.environ.get("GEN3AI_PYTHON") or "/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3"


def _cpu_env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(SRC), "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}


_CHECKS: dict[tuple[str, ...], tuple[str, str]] = {}


def _run_check(argv: list[str], timeout: int = 900) -> tuple[str, str]:
    """``("ok" | "fail", detail)`` of a CPU check subprocess (no GPU: ``CUDA_VISIBLE_DEVICES=""``). The same argv is
    run once per battery invocation (the four learner cells share one check)."""
    key = tuple(argv)
    if key in _CHECKS:
        st, d = _CHECKS[key]
        return st, d + " (same check as an earlier cell)"
    try:
        q = subprocess.run(argv, env=_cpu_env(), cwd=str(ROOT), capture_output=True, text=True, timeout=timeout,
                           check=False)
    except subprocess.TimeoutExpired:
        return "fail", f"check timed out after {timeout} s"
    tail = (q.stdout.strip().splitlines() or [""])[-1] if q.returncode == 0 else (q.stderr or q.stdout).strip()[-800:]
    _CHECKS[key] = ("ok" if q.returncode == 0 else "fail"), tail
    return _CHECKS[key]


def _json_status_ok(p: Path) -> tuple[bool, str]:
    if not p.is_file():
        return False, f"no result file {p}"
    try:
        st = json.loads(p.read_text()).get("status")
    except (OSError, ValueError) as e:
        return False, f"unreadable result {p}: {e!r}"
    return st == "ok", f"result status {st!r}"


# ------------------------------------------------------------------------------------------------- the cells

def _update_cfg(out: Path, name: str) -> Path:
    return out / "cells" / f"{name}.cfg.json"


def _update_prepare(name: str, gate: str) -> Callable[[Path], None]:
    def prep(out: Path) -> None:
        sys.path.insert(0, str(SRC))
        from agents.training.gpu_bench_suite import phase_config
        c = phase_config("update_bf16", False, "cuda", str(out / "cells"))
        # the suite's update_bf16 cell with ONE field changed: its name / output. Everything else (X26's argv, the K9
        # learner, the fixture rows, micro-batch 2,048 x accumulation 32, epochs (1, 3) -> the 10-epoch estimate, the
        # suite's reps 2) is the frozen suite's. (The 2026-10-19 t62_u_* cells ran reps 4: README "Noise".)
        # ``gate``: "on" = the launch's own sentinel (a non-PASS verdict FAILS the cell: learner_path.py); "off" = skipped by
        # config in all four cells (identical by construction). The suite's own cell default is "on" as well.
        c.update(name=name, out=str(out / "cells" / f"{name}.json"), gate=gate)
        _update_cfg(out, name).write_text(json.dumps(c, indent=1))
    return prep


def _upd_cmd(py: str, upd: str, name: str, arm: str) -> Callable[[Path], list[str]]:
    return lambda out: [py, "-u", upd, "--arm", arm, str(_update_cfg(out, name))]


def _json_result(name: str) -> Callable[[Path], tuple[bool, str]]:
    return lambda out: _json_status_ok(out / "cells" / f"{name}.json")


_PIN_OVERRIDE: str | None = None


def battery_pin(out: Path) -> str:
    """The battery's PIN (full sha): ``--pin`` if typed, else the one recorded when the window opened
    (``battery_status.json``), else this checkout's HEAD. The X14 arms run AT it (``--pin-commit``)."""
    st = out / "battery_status.json"
    stored = None
    if st.is_file():
        try:
            stored = json.loads(st.read_text()).get("pin")
        except (OSError, ValueError):
            stored = None
    spec = _PIN_OVERRIDE or stored or "HEAD"
    q = subprocess.run(["git", "rev-parse", "--verify", f"{spec}^{{commit}}"], cwd=str(ROOT), capture_output=True,
                       text=True, check=False)
    return q.stdout.strip() or spec


def cells(learner_gate: str = LP.GATE_ON) -> list[Cell]:
    py = _py()
    cs: list[Cell] = []
    # 0. FIRST: tau on the CUDA production pair (cheap; it gates what the cost read MEANS). A cell that is not PASS holds.
    par = str(HERE / "tau_cuda_parity_cell.py")
    cs.append(Cell(
        "tau_cuda_parity", "FP32_ALT on CUDA (REVIEW F2): compiled Triton tau vs eager on synthetic + X26 violator + 915 "
                           "fixture rows, and the compiled rollout vs the CUDA learner's taken-action log pi",
        mem_gb=16, wall_min=25, eta_min=9,
        cmd=lambda out: [py, "-u", par, "--out", str(out / "cells" / "tau_cuda_parity.json")],
        check=lambda out: _run_check([py, par, "--check"]),
        dry_check=lambda out: _run_check([py, par, "--check", "--prove"], timeout=1800),
        result=lambda out: _json_status_ok(out / "cells" / "tau_cuda_parity.json")))
    # 1. the rollout decision forward (naive vs alt in one process)
    dec = str(HERE / "tau_decision_cell.py")
    cs.append(Cell(
        "tau_dec", "FP32_ALT: T2 decision forward, compiled + CUDA graph, buckets 64 / 256, fp32_naive vs fp32_alt "
                   "tau (ABAB)",
        mem_gb=16, wall_min=45, eta_min=16,
        cmd=lambda out: [py, "-u", dec, "--out", str(out / "cells" / "tau_dec.json")],
        check=lambda out: _run_check([py, dec, "--check"]),
        result=lambda out: _json_status_ok(out / "cells" / "tau_dec.json")))
    # 2. the learner (four suite update cells, ABBA)
    upd = str(HERE / "tau_update_cell.py")
    for name, arm in TAU_UPDATE_CELLS:
        cs.append(Cell(
            name, f"FP32_ALT: learner compiled bf16 train() at production shape, tau solve {arm} "
                  f"(startup gate {learner_gate}: both arms the identical compiled path)",
            mem_gb=24, wall_min=40, eta_min=6,
            cmd=_upd_cmd(py, upd, name, arm),
            check=lambda out: _run_check([py, upd, "--check"]),
            dry_check=lambda out: _run_check([py, upd, "--check", "--prove"], timeout=1800),
            result=_json_result(name),
            prepare=_update_prepare(name, learner_gate), gate_mode=learner_gate))
    # 3. the X14 overlap A/B: two forks of X26 @ 380,000,195, 15 updates each, at the battery's pin
    x14 = str(HERE / "x14_cell.py")
    for arm, eta, wall in (("off", 45, 75), ("on", 25, 55)):
        cs.append(Cell(
            f"x14_{arm}", f"X14 A/B, the {'--overlap-collect ON' if arm == 'on' else 'serial OFF'} arm: a fork of X26 @ "
                          "380,000,195, 15 updates, --behaviour-check warn, eval + ladder off, at the battery's pin",
            mem_gb=40, wall_min=wall, eta_min=eta, kill_after_s=240,
            cmd=(lambda out, arm=arm: [py, "-u", x14, "--arm", arm, "--pin", battery_pin(out), "--battery-out", str(out),
                                       "--out", str(out / "cells" / f"x14_{arm}.json")]),
            check=(lambda out, arm=arm: _run_check([py, x14, "--arm", arm, "--pin", battery_pin(out), "--battery-out",
                                                    str(out), "--check", "--scratch", str(out / "x14_dry")],
                                                   timeout=1800)),
            dry_check=(lambda out, arm=arm: _run_check([py, x14, "--arm", arm, "--pin", battery_pin(out), "--battery-out",
                                                        str(out), "--check", "--prove", "--scratch", str(out / "x14_dry")],
                                                       timeout=1800)),
            result=(lambda out, arm=arm: _json_status_ok(out / "cells" / f"x14_{arm}.json"))))
    # 4. T2's frozen guard on the card (added 2026-10-20 with gen3_t2_stream_attribution_v1; LAST, so it never delays
    #    the FP32_ALT read): the 2026-10-19 arming check re-run on its fixed bar (gen3_expandable_floor_v1), and the
    #    overlap attribution proof -- T2's own growth still seen, a learner growth in a worker flush NOT booked to T2
    junit = "t2_guard.junit.xml"
    cs.append(Cell(
        "t2_guard", "T2 frozen guard: expandable steady state (eager + graph) + overlap stream attribution "
                    "(expandable + default allocator)",
        mem_gb=16, wall_min=45, eta_min=12,
        cmd=lambda out: ["env", "GEN3AI_TEST_ALLOW_GPU=1", NO_SLOW_RECORD, py, "-m", "pytest", *T2_GUARD_NODES, "-q",
                         "-rs", "-s", "-p", "no:cacheprovider", f"--junitxml={out / 'cells' / junit}"],
        check=lambda out: _check_t2_guard(py),
        result=lambda out: _junit_all_passed(out / "cells" / junit, expect=4)))
    return cs


#: The t2_guard cell's tests: 2 parametrised arming checks + 2 parametrised attribution proofs.
_T2_CUDA_TEST = "src/agents/inference/service/service_cuda_test.py"
T2_GUARD_NODES = (
    f"{_T2_CUDA_TEST}::test_expandable_segments_steady_state_reserves_nothing_and_a_new_mapping_is_seen",
    f"{_T2_CUDA_TEST}::test_overlap_attribution_books_only_t2s_own_streams")


def _check_t2_guard(py: str) -> tuple[str, str]:
    """SKIP at a commit without the attribution test; else the 4 tests must collect (they SKIP without
    ``GEN3AI_TEST_ALLOW_GPU=1``; a SKIP on the GPU is a HOLD: `_junit_all_passed` counts skips as a failure)."""
    src = ROOT / _T2_CUDA_TEST
    if not src.is_file() or "test_overlap_attribution_books_only_t2s_own_streams" not in src.read_text():
        return "skip", "gen3_t2_stream_attribution_v1 not at this commit (its card test is absent)"
    st, d = _run_check([py, "-m", "pytest", *[str(ROOT / n) for n in T2_GUARD_NODES], "--collect-only", "-q",
                        "-p", "no:cacheprovider"])
    if st != "ok":
        return st, d
    return "ok", "4 parametrised tests collect (a SKIP on the GPU is a HOLD)"


def _junit_all_passed(p: Path, expect: int) -> tuple[bool, str]:
    if not p.is_file():
        return False, f"no junit {p}"
    import xml.etree.ElementTree as ET
    r = ET.parse(p).getroot()
    s = r if r.tag == "testsuite" else r.find("testsuite")
    if s is None:
        return False, "junit has no testsuite"
    n, f, e, sk = (int(s.get(k, 0)) for k in ("tests", "failures", "errors", "skipped"))
    ok = n == expect and f == 0 and e == 0 and sk == 0
    return ok, f"tests {n} failures {f} errors {e} skipped {sk} (a SKIP is NOT a pass)"


# ------------------------------------------------------------------------------------------------- the driver

def gpu_cmd(c: Cell, out: Path) -> list[str]:
    return [str(OPS / "gpu_lock.sh"), str(OPS / "mem_cap.sh"), "--name", f"battery-{c.name}", str(c.mem_gb),
            "timeout", f"--kill-after={c.kill_after_s}", f"{c.wall_min * 60}"] + c.cmd(out)


def _gpu_others() -> list[str] | None:
    try:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if q.returncode != 0:
        return None
    return [ln.strip() for ln in q.stdout.splitlines() if ln.strip()]


class Status:
    def __init__(self, out: Path):
        self.path, self.log = out / "battery_status.json", out / "battery.log"
        self.d: dict[str, Any] = json.loads(self.path.read_text()) if self.path.is_file() else {"cells": {}}

    def line(self, msg: str) -> None:
        s = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(s, flush=True)
        with self.log.open("a") as fh:
            fh.write(s + "\n")

    def save(self, **kw: Any) -> None:
        self.d.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.d, indent=1))
        tmp.replace(self.path)


def _state_path(out: Path, c: Cell) -> Path:
    return out / "cells" / f"{c.name}.state.json"


def _write_state(out: Path, c: Cell, **kw: Any) -> None:
    extra = {"gate_mode": c.gate_mode} if c.gate_mode else {}
    _state_path(out, c).write_text(json.dumps({"cell": c.name, **extra, **kw}, indent=1))


def _done(out: Path, c: Cell) -> str | None:
    p = _state_path(out, c)
    if not p.is_file():
        return None
    d = json.loads(p.read_text())
    st = d.get("status")
    if st == "ok" and c.gate_mode and d.get("gate_mode") != c.gate_mode:
        return None            # a learner cell recorded under another --learner-gate is not part of THIS pair set
    return st if st in ("ok", "skip") else None


def _main_checkout() -> Path:
    """The MAIN checkout (where `models/` lives), from git's common dir; this checkout if git cannot say."""
    q = subprocess.run(["git", "rev-parse", "--path-format=absolute", "--git-common-dir"], cwd=str(ROOT),
                       capture_output=True, text=True, check=False)
    return Path(q.stdout.strip()).parent if q.returncode == 0 and q.stdout.strip() else ROOT


def _refuse_out(out: Path) -> str | None:
    for bad in ("models", "data"):
        for base in (ROOT, _main_checkout()):
            b = (base / bad).resolve()
            if out == b or b in out.parents:
                return f"--out {out} is under {b} (never write under models/ or data/)"
    return None


def dry_run(out: Path, sel: list[Cell]) -> int:
    print(f"[battery DRY-RUN] checkout {ROOT} @ {_git_head()}; out {out}; python {_py()}; NOTHING touches the GPU")
    bad = 0
    for tool in ("gpu_lock.sh", "mem_cap.sh"):
        ok = os.access(OPS / tool, os.X_OK)
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} {OPS / tool} executable")
    tok = os.environ.get("GEN3AI_GPU_LEASE_TOKEN") or os.environ.get("GEN3AI_GPU_LEASE_TOKEN_FILE")
    print(f"  {'ok  ' if tok else 'WARN'} lease token {'set' if tok else 'NOT set (the real run refuses without it)'}")
    for c in sel:                                  # every pytest cell must not record into main's status file
        argv = c.cmd(out)
        if "pytest" in argv:
            ok = NO_SLOW_RECORD in argv
            bad += not ok
            print(f"  {'ok  ' if ok else 'FAIL'} {c.name}: pytest cell exports {NO_SLOW_RECORD}")
    st_r, d_r = _run_check([_py(), str(HERE / "read_battery.py"), "--selftest"], timeout=600)
    bad += st_r == "fail"
    print(f"  {'ok  ' if st_r == 'ok' else 'FAIL'} reader self-test: {d_r}")
    print(f"  battery pin (the X14 arms run AT it): {battery_pin(out)}")
    total_eta = total_cap = 0
    for c in sel:
        st, detail = c.dry_check(out)
        if st == "ok" and c.prepare:                  # the cell's config is written (pure, CPU) and read back
            try:
                c.prepare(out)
                cfg = json.loads(_update_cfg(out, c.name).read_text())
                detail += (f"; config written ({_update_cfg(out, c.name).name}: reps {cfg.get('reps')}, "
                           f"epochs {cfg.get('epochs')}, trunk {cfg.get('trunk_precision')}, gate {cfg.get('gate')})")
            except Exception as e:  # noqa: BLE001 -- a dry-run FAIL, reported
                st, detail = "fail", f"prepare raised {e!r}"
        bad += st == "fail"
        if st != "skip":
            total_eta += c.eta_min
            total_cap += c.wall_min
        print(f"\n  [{st.upper():4}] {c.name}: {c.what}\n         mem cap {c.mem_gb} GB, wall cap {c.wall_min} min, "
              f"ETA {c.eta_min} min\n         check: {detail}\n         cmd: {shlex.join(gpu_cmd(c, out))}")
    print(f"\n[battery DRY-RUN] runnable cells: ETA {total_eta} min ({total_eta / 60:.1f} h), worst case "
          f"(every wall cap) {total_cap} min; {'FAIL' if bad else 'all checks ok'}")
    return 1 if bad else 0


def _git_head() -> str:
    q = subprocess.run(["git", "rev-parse", "--short=9", "HEAD"], cwd=str(ROOT), capture_output=True, text=True,
                       check=False)
    return q.stdout.strip() or "?"


def run(out: Path, sel: list[Cell], budget_min: float | None) -> int:
    S = Status(out)
    if not (os.environ.get("GEN3AI_GPU_LEASE_TOKEN") or os.environ.get("GEN3AI_GPU_LEASE_TOKEN_FILE")):
        S.line("HOLD: no GPU lease token in the environment (GEN3AI_GPU_LEASE_TOKEN[_FILE]) -- no lease, no GPU")
        S.save(status="HOLD", reason="no lease token")
        return 1
    pin = battery_pin(out)                # --pin if typed, else the one recorded at the window's opening, else HEAD
    gates = sorted({c.gate_mode for c in sel if c.gate_mode})
    S.save(status="RUNNING", commit=_git_head(), pin=pin, learner_gate=gates[0] if len(gates) == 1 else None,
           started=S.d.get("started") or time.strftime("%Y-%m-%d %H:%M:%S"))
    S.line(f"battery pin {pin[:12]} (the X14 arms run AT it); checkout HEAD {_git_head()}; learner gate {gates}")
    t_start = time.time()
    for c in sel:
        prev = _done(out, c)
        if prev:
            S.line(f"{c.name}: {prev} already -- skipped")
            continue
        st, detail = c.check(out)
        if st == "skip":
            S.line(f"{c.name}: SKIP ({detail})")
            _write_state(out, c, status="skip", reason=detail, commit=_git_head())
            S.d["cells"][c.name] = "skip"
            S.save()
            continue
        if st != "ok":
            S.line(f"HOLD: {c.name} precondition failed: {detail}")
            S.save(status="HOLD", reason=f"{c.name} precondition: {detail}")
            return 1
        elapsed = (time.time() - t_start) / 60
        if budget_min is not None and elapsed + c.eta_min > budget_min:
            S.line(f"BUDGET_STOP before {c.name} ({elapsed:.0f} + ETA {c.eta_min} > {budget_min:.0f} min); re-run to resume")
            S.save(status="BUDGET_STOP", reason=f"before {c.name}")
            return 10
        others = _gpu_others()
        if others is None or others:
            S.line(f"HOLD: before {c.name} the GPU is not clear: {others if others else 'nvidia-smi unreadable'}")
            S.save(status="HOLD", reason=f"GPU not clear before {c.name}: {others}")
            return 1
        if c.prepare:
            c.prepare(out)
        argv = gpu_cmd(c, out)
        logf = out / "logs" / f"{c.name}.log"
        S.line(f"{c.name}: START (mem cap {c.mem_gb} GB, wall cap {c.wall_min} min, ETA {c.eta_min} min)")
        t0 = time.time()
        with logf.open("w") as fh:
            fh.write(f"# {shlex.join(argv)}\n")
            fh.flush()
            rc = subprocess.run(argv, env={**os.environ, "PYTHONPATH": str(SRC)}, cwd=str(ROOT),
                                stdout=fh, stderr=subprocess.STDOUT, check=False).returncode
        wall = (time.time() - t0) / 60
        ok, why = c.result(out)
        if rc != 0 or not ok:
            kind = {124: "WALL TIMEOUT", 137: "KILLED after timeout", 86: "OOM at its mem cap", 85: "mem cap not in force",
                    5: "GPU busy (one-off holder)", 6: "GPU leased to someone else",
                    4: "tau forcing violated (wrapper replaced, or an arm never reached it)",
                    9: "tau_cuda_parity: a PASS BAR was missed (the compiled rollout and the learner do not land on one tau)",
                    8: "learner path UNFAIR (this arm did not run the production compiled path; see --learner-gate off)",
                    11: "the X14 launcher exited non-zero", 12: "the X14 run is not the specified arm",
                    143: "stopped by SIGTERM (the wall cap)"}.get(rc, f"exit {rc}")
            _write_state(out, c, status="fail", rc=rc, wall_min=round(wall, 1), reason=f"{kind}; {why}")
            S.d["cells"][c.name] = "fail"
            S.line(f"HOLD: {c.name} FAILED ({kind}; {why}) after {wall:.1f} min -- see {logf}")
            S.save(status="HOLD", reason=f"{c.name}: {kind}; {why}")
            return 1
        _write_state(out, c, status="ok", rc=rc, wall_min=round(wall, 1), commit=_git_head(), detail=why)
        S.d["cells"][c.name] = "ok"
        S.save()
        S.line(f"{c.name}: ok in {wall:.1f} min")
    S.save(status="DONE")
    S.line(f"DONE; read: {_py()} {HERE / 'read_battery.py'} {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, help="the battery's output dir (never under models/ or data/)")
    ap.add_argument("--dry-run", action="store_true", help="check every cell on the CPU, print the commands")
    ap.add_argument("--only", default="", help="comma-separated cell names (default: all, in order)")
    ap.add_argument("--budget-min", type=float, default=None, help="stop cleanly before a cell that would overrun")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--learner-gate", choices=LP.GATE_MODES, default=LP.GATE_ON,
                    help="the four learner cells' startup parity gate: `on` (default, the launch's own path; a non-PASS "
                         "verdict FAILS the cell) or `off` (skipped by config in ALL FOUR cells: identical by "
                         "construction). A recorded ok of the other mode is re-run")
    ap.add_argument("--pin", default=None, help="the commit the X14 arms run AT (default: the pin recorded when the "
                                               "window opened, else this checkout's HEAD)")
    a = ap.parse_args(argv)
    global _PIN_OVERRIDE
    _PIN_OVERRIDE = a.pin
    out = Path(a.out).resolve()
    why = _refuse_out(out)
    if why:
        print(f"REFUSED: {why}", file=sys.stderr)
        return 2
    allc = cells(a.learner_gate)
    if a.list:
        for c in allc:
            print(f"{c.name:14s} mem {c.mem_gb:3d} GB  cap {c.wall_min:3d} min  ETA {c.eta_min:3d} min  {c.what}")
        return 0
    names = [n for n in a.only.split(",") if n]
    unknown = set(names) - {c.name for c in allc}
    if unknown:
        print(f"REFUSED: unknown cells {sorted(unknown)}", file=sys.stderr)
        return 2
    sel = [c for c in allc if not names or c.name in names]
    (out / "cells").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    return dry_run(out, sel) if a.dry_run else run(out, sel, a.budget_min)


if __name__ == "__main__":
    sys.exit(main())
