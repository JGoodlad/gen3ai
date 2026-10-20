"""THE GPU-WINDOW BATTERY (staged 2026-10-19): every pending GPU-only measurement, one cell at a time, for the next
GPU window (X34's 24 h restart or the X34 -> X35 gap). README.md has the order, the costs and the decision rules.

    # the training session (it holds the GPU lease), from the MAIN checkout, the run PAUSED and the GPU clear:
    export GEN3AI_GPU_LEASE_TOKEN_FILE=<the lease token file>     # or GEN3AI_GPU_LEASE_TOKEN=<token>
    designs/research_state/measurements/gpu_window_battery_2026-10-19/run_battery.sh --out <dir> [--budget-min N]
    designs/research_state/measurements/gpu_window_battery_2026-10-19/run_battery.sh --out <dir> --dry-run

INCREMENTAL + RESUMABLE: a cell whose ``cells/<name>.state.json`` reads ``ok`` (or ``skip``: its precondition says it
cannot run at this commit) is skipped on a re-run. ``--only a,b`` runs named cells; ``--budget-min N`` stops CLEANLY
(status ``BUDGET_STOP``, exit 10) before a cell whose ETA would overrun N minutes since the start -- re-run to resume.

HOLDS on any failure (TRAINING_RUN_SOP §3 window-script rule 3): a cell that exits non-zero, times out, is OOM-killed
at its cap, writes no ``ok`` result, or finds another process on the GPU -> ``battery_status.json`` reads ``HOLD`` with
the reason, the log line is ``HOLD: ...``, and the battery exits 1 WITHOUT running the next cell. It never stops,
resumes or signals a training run: the caller owns the pause / resume.

Every GPU cell runs as ``gpu_lock.sh mem_cap.sh --name <cell> <GB> timeout --kill-after=60 <cap> <cmd>`` -- the
leaseholder's token passes the lock (anyone else is refused at once), the cap OOM-kills only the cell, the wall cap
kills a hung cell (a timeout is a HOLD, never a result). ``--dry-run`` runs every cell's precondition check on the
CPU with ``CUDA_VISIBLE_DEVICES=""`` and prints every command; nothing touches the GPU.
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
ROOT = HERE.parents[3]          # designs/research_state/measurements/<this> -> the checkout
OPS = ROOT / "scripts" / "ops"
SRC = ROOT / "src"

#: The T62 learner pairs, run ABBA (old, new, new, old) so a slow drift over the window cancels in the mean and
#: shows in the two pairs' disagreement, which the reader turns into the decision band.
T62_UPDATE_CELLS = (("t62_u_old_a", "float32"), ("t62_u_new_a", "float64"),
                    ("t62_u_new_b", "float64"), ("t62_u_old_b", "float32"))
FROZEN_GUARD_NODE = ("src/agents/inference/service/service_cuda_test.py::"
                     "test_expandable_segments_steady_state_reserves_nothing_and_a_new_mapping_is_seen")
X45_BENCH = "src/agents/inference/service/grouped_benchmark.py"
X45_FLAG = "--t2-grouped-opponents"
X14_HOOK = HERE / "x14_cell.sh"


class Cell:
    """One measurement: its command, its caps and its precondition (pure description; `battery` runs it)."""

    def __init__(self, name: str, what: str, mem_gb: int, wall_min: int, eta_min: int,
                 cmd: Callable[[Path], list[str]], check: Callable[[Path], tuple[str, str]],
                 result: Callable[[Path], tuple[bool, str]], prepare: Callable[[Path], None] | None = None):
        self.name, self.what, self.mem_gb, self.wall_min, self.eta_min = name, what, mem_gb, wall_min, eta_min
        self.cmd, self.check, self.result, self.prepare = cmd, check, result, prepare


#: Every pytest cell runs from the MAIN checkout, where the conftest's slow-tier recorder would rewrite main's
#: committed `designs/ops/slow_tier_status.json` and leave main dirty mid-window (standing rule 4). This is the
#: conftest's own escape hatch; the dry run asserts every pytest cell's command carries it.
NO_SLOW_RECORD = "GEN3AI_SKIP_SLOW_STATUS_RECORD=1"


def _py() -> str:
    return os.environ.get("PY") or os.environ.get("GEN3AI_PYTHON") or "/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3"


def _cpu_env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": str(SRC), "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1"}


def _run_check(argv: list[str], timeout: int = 900) -> tuple[str, str]:
    """``("ok" | "fail", detail)`` of a CPU check subprocess (no GPU: ``CUDA_VISIBLE_DEVICES=""``)."""
    try:
        q = subprocess.run(argv, env=_cpu_env(), cwd=str(ROOT), capture_output=True, text=True, timeout=timeout,
                           check=False)
    except subprocess.TimeoutExpired:
        return "fail", f"check timed out after {timeout} s"
    tail = (q.stdout.strip().splitlines() or [""])[-1] if q.returncode == 0 else (q.stderr or q.stdout).strip()[-800:]
    return ("ok" if q.returncode == 0 else "fail"), tail


def _json_status_ok(p: Path) -> tuple[bool, str]:
    if not p.is_file():
        return False, f"no result file {p}"
    try:
        st = json.loads(p.read_text()).get("status")
    except (OSError, ValueError) as e:
        return False, f"unreadable result {p}: {e!r}"
    return st == "ok", f"result status {st!r}"


# ------------------------------------------------------------------------------------------------- the cells

def _t62_update_cfg(out: Path, name: str) -> Path:
    return out / "cells" / f"{name}.cfg.json"


def _t62_update_prepare(name: str) -> Callable[[Path], None]:
    def prep(out: Path) -> None:
        sys.path.insert(0, str(SRC))
        from agents.training.gpu_bench_suite import phase_config
        c = phase_config("update_bf16", False, "cuda", str(out / "cells"))
        # the suite's update_bf16 cell with TWO fields changed: its name / output, and reps 2 -> 4 (+8 timed epochs,
        # ~35 s: a tighter median for a 3 % gate). Everything else (X26's argv, the K9 learner, the fixture rows,
        # micro-batch 2,048 x accumulation 32, epochs (1, 3) -> the 10-epoch estimate) is the frozen suite's.
        c.update(name=name, reps=4, out=str(out / "cells" / f"{name}.json"))
        _t62_update_cfg(out, name).write_text(json.dumps(c, indent=1))
    return prep


def _upd_cmd(py: str, upd: str, name: str, tau: str) -> Callable[[Path], list[str]]:
    return lambda out: [py, "-u", upd, "--tau", tau, str(_t62_update_cfg(out, name))]


X45_CUDA_TEST = "src/agents/inference/service/grouped_cuda_test.py"
X45_POOLS = (20, 5)
X45_BUCKETS = ("8,64,256", "8,16,32,64,256")


def x45_name(pool: int, buckets: str) -> str:
    return f"x45_p{pool}_b{len(buckets.split(','))}"


def _x45_cmd(py: str, pool: int, buckets: str, name: str) -> Callable[[Path], list[str]]:
    return lambda out: [py, "-u", str(ROOT / X45_BENCH), "--opp-slots", str(pool), "--buckets", buckets,
                        "--json", str(out / "cells" / f"{name}.json")]


def _json_result(name: str) -> Callable[[Path], tuple[bool, str]]:
    return lambda out: _json_status_ok(out / "cells" / f"{name}.json")


def _written_result(name: str) -> Callable[[Path], tuple[bool, str]]:
    return lambda out: _file_written(out / "cells" / f"{name}.json")


def cells() -> list[Cell]:
    py = _py()
    cs: list[Cell] = []
    # 1. T62, the rollout decision forward (old vs new in one process)
    dec = str(HERE / "t62_decision_cell.py")
    cs.append(Cell(
        "t62_dec", "T62: T2 decision forward, compiled + CUDA graph, buckets 64 / 256, fp32 vs fp64 tau (ABAB)",
        mem_gb=16, wall_min=45, eta_min=20,
        cmd=lambda out: [py, "-u", dec, "--out", str(out / "cells" / "t62_dec.json")],
        check=lambda out: _run_check([py, dec, "--check"]),
        result=lambda out: _json_status_ok(out / "cells" / "t62_dec.json")))
    # 2. the frozen_guard arming check (bf4eb59d)
    junit = "frozen_guard.junit.xml"
    cs.append(Cell(
        "frozen_guard", "T2 frozen guard under expandable_segments: a healthy flush reserves 0 B (eager + graph)",
        mem_gb=16, wall_min=65, eta_min=10,
        cmd=lambda out: ["env", "GEN3AI_TEST_ALLOW_GPU=1", NO_SLOW_RECORD, py, "-m", "pytest", FROZEN_GUARD_NODE, "-q", "-rs",
                         "-p", "no:cacheprovider", f"--junitxml={out / 'cells' / junit}"],
        check=lambda out: _check_frozen_guard(py),
        result=lambda out: _junit_all_passed(out / "cells" / junit, expect=2)))
    # 3. T62, the learner (four suite update cells, ABBA)
    upd = str(HERE / "t62_update_cell.py")
    for name, tau in T62_UPDATE_CELLS:
        cs.append(Cell(
            name, f"T62: learner compiled bf16 train() at production shape, tau solve {tau}",
            mem_gb=24, wall_min=40, eta_min=12,
            cmd=_upd_cmd(py, upd, name, tau),
            check=lambda out: _run_check([py, upd, "--check"]),
            result=_json_result(name),
            prepare=_t62_update_prepare(name)))
    # 4. X45 (landed ad821b615): the CUDA test first (it must PASS, not skip), then the sweep
    xjunit = "x45_cuda_test.junit.xml"
    cs.append(Cell(
        "x45_cuda_test", "X45: grouped_cuda_test.py -- graph-backend parity, weight swap, frozen steady state (GPU)",
        mem_gb=24, wall_min=30, eta_min=5,
        cmd=lambda out: ["env", "GEN3AI_TEST_ALLOW_GPU=1", NO_SLOW_RECORD, py, "-m", "pytest", str(ROOT / X45_CUDA_TEST), "-q", "-rs",
                         "-p", "no:cacheprovider", f"--junitxml={out / 'cells' / xjunit}"],
        check=lambda out: _check_x45(py, cuda_test=True),
        result=lambda out: _junit_all_passed(out / "cells" / xjunit, expect=1)))
    # X45's own command (design_compute_efficiency.md 1.6): grouped_benchmark.py --opp-slots P --buckets B --json
    for pool in X45_POOLS:
        for buckets in X45_BUCKETS:
            nm = x45_name(pool, buckets)
            cs.append(Cell(
                nm, f"X45: grouped (vmapped) vs per-slot opponent forward, {pool} active pool slots, buckets {buckets} "
                    "(OFF/ON interleaved)",
                mem_gb=24, wall_min=30, eta_min=10 if buckets.count(",") > 2 else 7,
                cmd=_x45_cmd(py, pool, buckets, nm),
                check=lambda out: _check_x45(py),
                result=_written_result(nm)))
    # 5. X14 (optional, in build): a hook file the X14 unit supplies
    cs.append(Cell(
        "x14_ab", "X14 (optional): off-policy overlap GPU A/B -- the X14 unit supplies x14_cell.sh <out-json>",
        mem_gb=24, wall_min=60, eta_min=30,
        cmd=lambda out: ["bash", str(X14_HOOK), str(out / "cells" / "x14_ab.json")],
        check=lambda out: _check_x14(),
        result=lambda out: _json_status_ok(out / "cells" / "x14_ab.json")))
    return cs


def _check_frozen_guard(py: str) -> tuple[str, str]:
    q = subprocess.run(["git", "merge-base", "--is-ancestor", "bf4eb59d", "HEAD"], cwd=str(ROOT), check=False)
    if q.returncode != 0:
        return "fail", "bf4eb59d (gen3_t2_reserved_after_freeze_v1) is not in this checkout's history"
    st, d = _run_check([py, "-m", "pytest", FROZEN_GUARD_NODE, "--collect-only", "-q", "-p", "no:cacheprovider"])
    if st != "ok":
        return st, d
    return "ok", "2 parametrised tests collect (they SKIP without GEN3AI_TEST_ALLOW_GPU=1; a SKIP on the GPU is a HOLD)"


def _check_x45(py: str, cuda_test: bool = False) -> tuple[str, str]:
    """SKIP at an older commit (no benchmark / flag); the CUDA test cell collects its test, the others run --help."""
    bench = ROOT / X45_BENCH
    parser_src = SRC / "main" / "train" / "parser" / "env_core.py"
    if not bench.is_file():
        return "skip", f"X45 not landed at this commit: {X45_BENCH} absent"
    if not parser_src.is_file() or X45_FLAG not in parser_src.read_text():
        return "skip", f"X45 not landed at this commit: {X45_FLAG} absent from the trainer parser"
    if cuda_test:
        if not (ROOT / X45_CUDA_TEST).is_file():
            return "skip", f"X45 not landed at this commit: {X45_CUDA_TEST} absent"
        st, d = _run_check([py, "-m", "pytest", str(ROOT / X45_CUDA_TEST), "--collect-only", "-q", "-p", "no:cacheprovider"])
        return (st, d) if st != "ok" else ("ok", "1 test collects (it SKIPs without GEN3AI_TEST_ALLOW_GPU=1; a SKIP on the GPU is a HOLD)")
    return _run_check([py, str(bench), "--help"])


def _check_x14() -> tuple[str, str]:
    if not X14_HOOK.is_file():
        return "skip", f"X14 not staged: {X14_HOOK.name} absent (the X14 unit adds it with its own precondition)"
    q = subprocess.run(["bash", str(X14_HOOK), "--check"], env=_cpu_env(), cwd=str(ROOT),
                       capture_output=True, text=True, timeout=900, check=False)
    if q.returncode == 3:
        return "skip", (q.stdout or q.stderr).strip()[-400:] or "x14_cell.sh --check: not at this commit"
    return ("ok" if q.returncode == 0 else "fail"), (q.stdout or q.stderr).strip()[-400:]


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


def _file_written(p: Path) -> tuple[bool, str]:
    return (p.is_file() and p.stat().st_size > 0), f"result {p} {'written' if p.is_file() else 'absent'}"


# ------------------------------------------------------------------------------------------------- the driver

def gpu_cmd(c: Cell, out: Path) -> list[str]:
    return [str(OPS / "gpu_lock.sh"), str(OPS / "mem_cap.sh"), "--name", f"battery-{c.name}", str(c.mem_gb),
            "timeout", "--kill-after=60", f"{c.wall_min * 60}"] + c.cmd(out)


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
    _state_path(out, c).write_text(json.dumps({"cell": c.name, **kw}, indent=1))


def _done(out: Path, c: Cell) -> str | None:
    p = _state_path(out, c)
    if not p.is_file():
        return None
    st = json.loads(p.read_text()).get("status")
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
    total_eta = total_cap = 0
    for c in sel:
        st, detail = c.check(out)
        if st == "ok" and c.prepare:                  # the cell's config is written (pure, CPU) and read back
            try:
                c.prepare(out)
                detail += f"; config written ({_t62_update_cfg(out, c.name).name})"
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
    S.save(status="RUNNING", commit=_git_head(), started=S.d.get("started") or time.strftime("%Y-%m-%d %H:%M:%S"))
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
                    5: "GPU busy (one-off holder)", 6: "GPU leased to someone else"}.get(rc, f"exit {rc}")
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
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    why = _refuse_out(out)
    if why:
        print(f"REFUSED: {why}", file=sys.stderr)
        return 2
    allc = cells()
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
