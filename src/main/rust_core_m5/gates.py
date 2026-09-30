"""THE LANE-GATE VERDICT TABLE (M5 Lane J): every declared lane's own gate, run (or read), folded
into one row per lane — PASS / FAIL / NOT BUILT / INCONCLUSIVE / NOT RUN / UNRECORDED.

Two ways to fill a row, never mixed silently:

* **RUN** (``--tier commit|milestone``): ONE pytest session over the union of every BUILT row's
  tests at the tier's markers (a node shared by two rows — the Rust cargo suite — runs once and
  counts for both), verdicts read back per node id by ``outcomes_plugin``. GPU tests run in a
  second session under ``flock /home/goodlad/.claude/jobs/gpu.lock`` with
  ``GEN3AI_TEST_ALLOW_GPU=1`` — only with ``--gpu``; otherwise the row's GPU part reads NOT RUN.
* **RECORDED** (``--from-status``): the MILESTONE (``slow``) verdicts the slow tier already banked
  in ``designs/ops/slow_tier_status.json``, with the commit each was recorded at — nothing runs.

The fold (``fold``): NOT BUILT rows never run and never pass; any ``fail`` ⇒ FAIL; nothing passed ⇒
INCONCLUSIVE (a skip-only or empty row is not a pass); an ``inconclusive`` node (a timeout) ⇒
INCONCLUSIVE unless something failed.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from main.rust_core_m5 import lanes as L

#: The box's GPU lock (every GPU use goes through ``flock <it>``); ``$GEN3AI_GPU_LOCK`` overrides.
GPU_LOCK = os.environ.get("GEN3AI_GPU_LOCK") or str(Path.home() / ".claude" / "jobs" / "gpu.lock")

PASS, FAIL, NOT_BUILT, INCONCLUSIVE, NOT_RUN, UNRECORDED = (
    "PASS", "FAIL", "NOT BUILT", "INCONCLUSIVE", "NOT RUN", "UNRECORDED")


@dataclass
class RowVerdict:
    lane: str
    title: str
    m5_gate: bool
    verdict: str
    counts: Dict[str, int] = field(default_factory=dict)
    gpu: str = ""                 # the GPU part's verdict ("" when the row has none)
    source: str = ""              # "run:<tier>" / "recorded"
    detail: str = ""


def _belongs(nodeid: str, tests: Iterable[str]) -> bool:
    for t in tests:
        if "::" in t:
            if nodeid == t or nodeid.startswith(t + "[") or nodeid.startswith(t + "::"):
                return True
        elif nodeid == t or nodeid.startswith(t + "::"):
            return True
    return False


def _count(outcomes: Dict[str, str]) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for v in outcomes.values():
        c[v] = c.get(v, 0) + 1
    return c


def _verdict_of(counts: Dict[str, int]) -> str:
    if counts.get("fail"):
        return FAIL
    if counts.get("inconclusive"):
        return INCONCLUSIVE
    if not counts.get("pass"):
        return INCONCLUSIVE
    return PASS


def fold(row: L.LaneGate, outcomes: Dict[str, str], *, source: str, gpu_ran: bool) -> RowVerdict:
    """One row's verdict from ``{nodeid: pass|fail|skip|inconclusive}`` (every node of the session;
    the row picks its own)."""
    if not row.built:
        return RowVerdict(row.lane, row.title, row.m5_gate, NOT_BUILT, source="", detail=row.pending)
    mine = {k: v for k, v in outcomes.items() if _belongs(k, row.tests)}
    gpu = {k: v for k, v in mine.items() if _belongs(k, row.gpu_tests)}
    cpu = {k: v for k, v in mine.items() if k not in gpu}
    counts = _count(cpu)
    v = _verdict_of(counts)
    detail = "" if counts.get("pass") else "no test of the row passed — nothing was measured"
    gv = ""
    if row.gpu_tests:
        gcounts = _count(gpu)
        if not gpu_ran or not gpu or set(gcounts) == {"skip"}:
            gv = NOT_RUN
        else:
            gv = _verdict_of(gcounts)
            if not cpu:                   # the row's only nodes here are GPU nodes (a recorded read)
                v, detail = gv, ""
            counts = {k: counts.get(k, 0) + gcounts.get(k, 0) for k in set(counts) | set(gcounts)}
            if gv == FAIL:
                v = FAIL
    return RowVerdict(row.lane, row.title, row.m5_gate, v, counts=counts, gpu=gv, source=source, detail=detail)


# ---------------------------------------------------------------------------------------- RUN

def bank_without_skip_clobber(scratch: Path, target: Path) -> List[str]:
    """Merge the slow-tier rows a session wrote into ``scratch`` (a copy of ``target``) back into
    ``target`` — every changed row EXCEPT a ``skip`` that would replace an existing verdict. A GPU
    test skips without ``GEN3AI_TEST_ALLOW_GPU``, and the slow tier's merge replaces rows outright,
    so a plain milestone run would overwrite a banked GPU PASS with a SKIP (F-LJ-5): a skip is not a
    measurement. Returns the node ids banked."""
    from utils.slow_tier_status import record_results

    if not scratch.is_file():
        return []
    new = json.loads(scratch.read_text()).get("tests", {})
    old = json.loads(target.read_text()).get("tests", {}) if target.is_file() else {}
    keep = {k: v for k, v in new.items()
            if v != old.get(k) and not (v.get("status") == "skip" and k in old)}
    if keep:
        record_results(keep, target)
    return sorted(keep)


def _pytest(tests: Sequence[str], markers: str, workers: int, *, gpu: bool, log: Path) -> Dict[str, str]:
    import shutil

    from utils.paths import repo_root, src_root
    from utils.slow_tier_status import status_path

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        out = Path(f.name)
    target = status_path()
    scratch = Path(tempfile.mkdtemp(prefix="m5J_slow_")) / "slow_tier_status.json"
    if target.is_file():
        shutil.copyfile(target, scratch)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{src_root()}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["GEN3AI_M5J_OUTCOMES"] = str(out)
    env["GEN3AI_SLOW_STATUS_FILE"] = str(scratch)
    argv = [sys.executable, "-m", "pytest", *tests, "-m", markers, "-q", "-p", "main.rust_core_m5.outcomes_plugin"]
    if workers > 1:
        argv += ["-n", str(workers)]
    if gpu:
        env["GEN3AI_TEST_ALLOW_GPU"] = "1"
        argv = ["flock", GPU_LOCK, *argv]
    with open(log, "a") as lf:
        lf.write(f"\n$ {' '.join(argv)}\n")
        lf.flush()
        p = subprocess.run(argv, cwd=str(repo_root()), env=env, stdout=lf, stderr=subprocess.STDOUT, check=False)
        banked = bank_without_skip_clobber(scratch, target)
        lf.write(f"\n[m5J] banked {len(banked)} slow-tier row(s) into {target} (skips never replace a verdict)\n")
    shutil.rmtree(scratch.parent, ignore_errors=True)
    try:
        data = json.loads(out.read_text())
    except (OSError, ValueError):
        return {"<session>": "fail"} if p.returncode not in (0, 5) else {}
    finally:
        out.unlink(missing_ok=True)
    return dict(data["outcomes"])


def run(tier: str, *, gpu: bool = False, workers: int = 2, lanes: Optional[Sequence[str]] = None,
        log: Optional[Path] = None) -> List[RowVerdict]:
    rows = [r for r in L.LANES if lanes is None or r.lane in lanes]
    markers = L.COMMIT_MARKERS if tier == "commit" else L.MILESTONE_MARKERS
    log = log or Path(tempfile.gettempdir()) / "m5J_gates.log"
    tests = sorted({t for r in rows if r.built for t in r.tests})
    outcomes = _pytest(tests, markers, workers, gpu=False, log=log) if tests else {}
    gpu_tests = sorted({t for r in rows if r.built for t in r.gpu_tests})
    if gpu and gpu_tests:
        gpu_out = _pytest(gpu_tests, L.MILESTONE_MARKERS, 1, gpu=True, log=log)
        outcomes.update({k: v for k, v in gpu_out.items() if _belongs(k, gpu_tests)})
    session = {"<session>": outcomes.pop("<session>")} if "<session>" in outcomes else {}
    verdicts = [fold(r, outcomes, source=f"run:{tier}", gpu_ran=gpu) for r in rows]
    if session:
        for v in verdicts:
            if v.verdict != NOT_BUILT:
                v.verdict, v.detail = FAIL, f"the pytest session produced no outcomes (see {log})"
    return verdicts


# ------------------------------------------------------------------------------------ RECORDED

def recorded(status: Optional[dict] = None, *, head: Optional[str] = None) -> List[RowVerdict]:
    """Each row's MILESTONE (``slow``) verdicts as the slow tier banked them. Rows with no banked
    slow test read UNRECORDED (their gate may be routine-only: run ``--tier commit``)."""
    from utils.slow_tier_status import status_path

    if status is None:
        status = json.loads(status_path().read_text())
    tests: Dict[str, dict] = status.get("tests", {})
    out = []
    for row in L.LANES:
        if not row.built:
            out.append(fold(row, {}, source="recorded", gpu_ran=False))
            continue
        mine = {k: v for k, v in tests.items() if _belongs(k, row.tests)}
        if not mine:
            out.append(RowVerdict(row.lane, row.title, row.m5_gate, UNRECORDED, source="recorded",
                                  detail="no slow test of this row is banked in slow_tier_status.json"))
            continue
        outcomes = {k: str(v.get("status")) for k, v in mine.items()}
        gpu_ran = any(_belongs(k, row.gpu_tests) and v == "pass" for k, v in outcomes.items())
        rv = fold(row, outcomes, source="recorded", gpu_ran=gpu_ran)
        commits = sorted({str(v.get("commit", ""))[:8] for v in mine.values()})
        stale = [c for c in commits if head is not None and c != head[:8]]
        ats = sorted(str(v.get("at", "")) for v in mine.values())
        rv.detail = f"recorded at {', '.join(commits)} ({ats[0][:16]} … {ats[-1][:16]})" + (
            "; STALE vs HEAD — a slow test that broke since reads green (run --tier milestone)" if stale else "")
        out.append(rv)
    return out


# -------------------------------------------------------------------------------------- render

def render(verdicts: Sequence[RowVerdict]) -> str:
    lines = ["| lane | what it is | verdict | GPU part | tests (pass/fail/skip/inconcl.) | source | note |",
             "|---|---|---|---|---|---|---|"]
    for v in verdicts:
        c = v.counts
        tally = (f"{c.get('pass', 0)}/{c.get('fail', 0)}/{c.get('skip', 0)}/{c.get('inconclusive', 0)}"
                 if c else "—")
        scope = "" if v.m5_gate else " *(beside M5)*"
        lines.append(f"| {v.lane} | {v.title}{scope} | **{v.verdict}** | {v.gpu or '—'} | {tally} | "
                     f"{v.source or '—'} | {v.detail} |")
    return "\n".join(lines)


def to_json(verdicts: Sequence[RowVerdict]) -> List[dict]:
    return [asdict(v) for v in verdicts]
