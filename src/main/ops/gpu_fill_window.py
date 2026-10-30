"""THE GPU FILL WINDOW (2026-10-29): arm-chain window items (``designs/ops/arm_chain.md`` §2,
``main.ops.arm_chain.window``) for an idle card between training arms, each its own time box behind ITS OWN preempt
marker (`PREEMPT_MARKERS`; write every one to stop them all). ``emit-chain --windows`` picks which (`WINDOWS`).

1. **``canary-stage2``**: canary recalibration stage 2's leased read
   (``designs/training/canary_recalibration_2026-10-28.md`` §7, its PREREG and FINDING 5): the canary's own check
   (``main.ops.canary_dump_replay --anchor``) on each of the W0 census's 20 X26-lineage checkpoints with the rows
   relocated at the stage-2 cap 1.2 (``--canary-shift-max 0.3``), then ``canary-verdict`` on CPU, which applies the
   registered rule (:func:`stage2_verdict`) and writes ``verdict.json``.
2. **``slice2-replay``** (2026-10-29): the slice-2-only canary refusals' discrimination read
   (``main.ops.canary_slice_read``, ``designs/research_state/measurements/canary_slice2_2026-10-29/``): one GPU step per
   canary state in `SLICE2_STATES` (the canary on slices 0 / 2 / 1 and on the carrier row sets, then the dither read),
   then ``canary_slice_read verdict`` on CPU (hypothesis (i), a real compiled-numerics difference: LIVE / RULED OUT /
   INCONCLUSIVE). ``on_fail: hold``.
3. **``slow-gpu``**: every CUDA-only ``slow`` test id (:func:`cuda_slow_ids`), ONE pytest process per node id with
   its own wall bound (:func:`node_timeout`); the root conftest banks each verdict into
   ``GEN3AI_SLOW_STATUS_FILE`` = ``<out-dir>/slow_gpu/slow_tier_status.json``, OUTSIDE every checkout (W0's compile
   cell, run in the main checkout, would have rewritten main's committed file). ``merge-rows`` later folds those
   rows into a ship worktree's ``designs/ops/slow_tier_status.json`` (``slow_tier_status.record_results``: a skip
   never replaces a measurement), and that worktree ships.

Each window starts with a CPU ``preflight`` step: on a failed precondition it WRITES ITS OWN window's preempt marker
(with the reason) and exits 3, so no GPU step of that window runs. What a failure means is `ON_FAIL`'s: the two
canary windows HOLD the chain; ``slow-gpu`` is ``continue`` (its failure, a preflight's, a step's or a budget
overrun's, ends it DONE_FAILED and the chain hands off), so a test node that runs long never HOLDs a training arm
behind it. THE SAFE ORDER: the HOLD windows first, ``slow-gpu`` LAST in its chain file (`WINDOWS` order).

THE BUDGET is computed, never typed (`window_budget_min`): the sum of each GPU step's MEASURED minutes (`STEP_MIN`) x
`BUDGET_HEADROOM`, plus `CPU_STEPS_MIN`. The 10-29 fill window's fixed 90 min assumed ~2 min per replay step; each
took 5 (2.5x), so its last three steps were NOT RUN and the HOLD stopped the queue behind it. The
window runner (not this module) checks the marker before every step, the lease token before every GPU step, the GPU
idle and the worktree clean and at the required commit, and wraps every GPU step in ``gpu_lock.sh mem_cap.sh <GB>
timeout <s>``; nothing here wraps itself (arm_chain.md §2).

    python -m main.ops.gpu_fill_window emit-chain --base CHAIN --name NAME --worktree WT --require-commit SHA \\
        --out-dir D [--windows slice2-replay,slow-gpu] [--write FILE]   # the chain file (JSON), CPU only (~1 min)
    python -m main.ops.gpu_fill_window canary-verdict --out-dir D    # the registered stage-2 rule (CPU)
    python -m main.ops.gpu_fill_window merge-rows --rows D/slow_gpu/slow_tier_status.json --into WT2/designs/ops/...

Nothing here touches the GPU, writes under ``models/`` or a checkout (``merge-rows`` writes only into a NON-main
worktree), or edits an existing chain file.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import pytest

#: The W0 CUDA census's checkpoints (``canary_recalibration_2026-10-28.md`` "The data"; the exact zips its
#: ``census/read_*.json`` name), in ``w0_window.census_lists()`` order (``gpu_fill_window_test`` pins the pairing).
CENSUS: Tuple[Tuple[str, str, str], ...] = (
    ("rb_x26_s2601", "100M", "checkpoint_100000037_steps.zip"),
    ("rb_x26_s2601", "180M", "checkpoint_180000055_steps.zip"),
    ("rb_x26_s2601", "230M", "checkpoint_230000074_steps.zip"),
    ("rb_x26_s2601", "280M", "checkpoint_280000170_steps.zip"),
    ("rb_x26_s2601", "330M", "checkpoint_330000202_steps.zip"),
    ("rb_x26_s2601", "380M", "checkpoint_380000195_steps.zip"),
    ("rb_x26c_s2601", "390M", "checkpoint_390000173_steps.zip"),
    ("rb_x26c_s2601", "400M", "checkpoint_400000161_steps.zip"),
    ("rb_x26c_s2601", "410M", "checkpoint_410000226_steps.zip"),
    ("rb_x26c_s2601", "420M", "checkpoint_420000146_steps.zip"),
    ("rb_x26c_s2601", "430M", "checkpoint_430000083_steps.zip"),
    ("rb_x26c_s2601", "440M", "checkpoint_440000037_steps.zip"),
    ("rb_x26c_s2601", "450M", "checkpoint_450000287_steps.zip"),
    ("rb_x26c_s2601", "460M", "checkpoint_460000104_steps.zip"),
    ("rb_x26c_s2601", "470M", "checkpoint_470000292_steps.zip"),
    ("rb_x26c_s2601", "480M", "checkpoint_480000085_steps.zip"),
    ("rb_x26c_s2601", "481M", "checkpoint_481000168_steps.zip"),
    ("rb_x26c_s2601", "482M", "checkpoint_482000079_steps.zip"),
    ("x26c_u1000_run", "478M", "checkpoint_478000208_steps.zip"),
    ("x26c_u1000_run", "479M", "checkpoint_479000039_steps.zip"),
)
#: the u1000 neighbours' scratch run dir W0 built (``w0_window prep-u1000``): copies, outside every checkout
U1000_RUN_DIR = "~/.claude/jobs/x35/w0_2026-10-28/census/x26c_u1000_run"
#: 4 x 0.3 = 1.2, the stage-2 cap C (``compile_regions.R1_BAR_K`` x the shift)
STAGE2_SHIFT_MAX = 0.3
STAGE2_MARGIN = 1.2
CANARY_BATCH = 2048
CANARY_REPEATS = 3
#: a per-parameter ratio within this RELATIVE distance of K is a boundary read (standing rule 8), never a pass
BOUNDARY_REL = 1e-3

#: CUDA-only slow tests whose skip is decided INSIDE the test body (invisible to the collection-time skip marks):
#: each must still be collected, or :func:`cuda_slow_ids` refuses (a stale entry is named, never dropped silently).
RUNTIME_CUDA_IDS: Tuple[str, ...] = (
    "src/agents/training/ridealong_update_test.py::test_PREALLOCATED_adam_steps_bit_identically_to_lazy_init[True]",
)
#: a collection-time skip reason that means "no CUDA here": the root conftest's hidden-GPU reason, or one naming CUDA
_CUDA_REASON = re.compile(r"cuda|hides the gpu", re.IGNORECASE)
_ENV_IDS_OUT = "GEN3AI_CUDA_IDS_OUT"
STATUS_REL = ("designs", "ops", "slow_tier_status.json")
#: THE WINDOWS, in their SAFE chain order (module docstring): every HOLD window before the ``continue`` one
WINDOWS: Tuple[str, ...] = ("canary-stage2", "slice2-replay", "slow-gpu")
DEFAULT_WINDOWS: Tuple[str, ...] = ("slice2-replay", "slow-gpu")
#: each window's OWN preempt marker (in the job dir): a failed preflight stops only its window
PREEMPT_MARKERS: Dict[str, str] = {"canary-stage2": "NO_GO_GPUFILL", "slice2-replay": "NO_GO_SLICE2",
                                   "slow-gpu": "NO_GO_SLOWGPU"}
#: what each window's failure means to the chain (`arm_chain.md` §2): the slow tier never HOLDs what follows it
ON_FAIL: Dict[str, str] = {"canary-stage2": "hold", "slice2-replay": "hold", "slow-gpu": "continue"}
#: MEASURED minutes per GPU step, the budget's basis. ``replay``: one `canary_dump_replay` process (build + the
#: ~4.3-min compile + 3 slices x 3 repeats), 5 min each over the 20 stage-2 steps of 2026-10-29 (job log
#: `x35-gpufill-oct29`, 15:50-17:21 for 17 steps + the by-hand three). ``slice_read``: one `canary_slice_read` process
#: (the same build and compile, 2 repeats, the row sets, a 16-seed dither read on 3 slices), **UNVERIFIED**: the
#: replay's 5 + ~3 for the dither read (CPU-timed 48 arm passes), + 1 for the larger state; its first run corrects it.
STEP_MIN: Dict[str, float] = {"replay": 5.0, "slice_read": 9.0}
BUDGET_HEADROOM = 1.5
#: the CPU steps' allowance (a preflight and a verdict: under a minute each, measured 0 min on 10-29)
CPU_STEPS_MIN = 5.0


def window_budget_min(step_minutes: Sequence[float]) -> float:
    """A window's budget: its GPU steps' measured minutes x `BUDGET_HEADROOM` + `CPU_STEPS_MIN`, whole minutes up."""
    return float(math.ceil(sum(float(m) for m in step_minutes) * BUDGET_HEADROOM + CPU_STEPS_MIN))


#: THE SLICE-2 STATES (2026-10-29): label, mode, the dump / checkpoint (``~`` and ``{models}`` expanded), the run dir,
#: the refusing parameters (the rule's targets) and the carrier row sets (the CPU read's carriers:
#: ``designs/research_state/measurements/canary_slice2_2026-10-29/README.md``), the anchor's shift (None: the run's own)
_ARCH = "~/gen3ai_archive"
SLICE2_STATES: Tuple[Dict[str, Any], ...] = (
    {"label": "a_x48kffn_s3502_u200", "mode": "dump", "run": "rb_x35_x48kffn_s3502",
     "path": f"{_ARCH}/x48kffn_s3502_canary_u200_unconfirmed_2026-10-28/canary_u200_unconfirmed_policy.pt",
     "watch": ("features_extractor.edge_bias.s1_map.weight", "features_extractor.edge_bias.c2_map.weight"),
     "row_sets": ("s2_minus_18_22=0-17,19-21,23-31", "s1_plus_18_22=18,22,32-63")},
    {"label": "b_n1p_s3505_u100", "mode": "dump", "run": "rb_x35_n1p_s3505",
     "path": f"{_ARCH}/n1p_s3505_canary_u100_unconfirmed_2026-10-29/canary_u100_unconfirmed_policy.pt",
     "watch": ("features_extractor.flat_intent_head.hidden.weight", "features_extractor.flat_intent_head.hidden.bias"),
     "row_sets": ("s2_minus_17=0-16,18-31",)},
    {"label": "c_n1p_s3501_u200", "mode": "dump", "run": "rb_x35_n1p_s3501",
     "path": f"{_ARCH}/n1p_canary_u200_2026-10-26/canary_u200_unconfirmed_policy.pt",
     "watch": ("features_extractor.hp_type_belief_head.norm.bias",),
     "row_sets": ("s2_minus_24=0-23,25-31", "s1_plus_24=24,32-63")},
    {"label": "d_x26c_482M_m1.2", "mode": "anchor", "run": "rb_x26c_s2601",
     "path": "{models}/rb_x26c_s2601/checkpoints/checkpoint_482000079_steps.zip", "shift_max": 0.3,
     "watch": ("features_extractor.spread_belief.nature_head.weight",),
     "row_sets": ("s2_minus_9=0-8,10-31", "s1_plus_9=9,32-63")},
    {"label": "e_a1arch_s3503_u200", "mode": "dump", "run": "rb_x35_a1arch_s3503",
     "path": f"{_ARCH}/a1arch_s3503_canary_u200_unconfirmed_2026-10-22/canary_u200_unconfirmed_policy.pt",
     "watch": ("features_extractor.edge_bias.s1_map.weight", "features_extractor.edge_bias.c2_map.weight"),
     "row_sets": ("s2_minus_18_22=0-17,19-21,23-31", "s1_plus_18_22=18,22,32-63")},
)
SLICE2_DITHER = 16
SLICE2_REPEATS = 2


def slice2_paths(state: Mapping[str, Any], models_dir: str) -> Tuple[str, str]:
    """``(the dump or checkpoint, the run dir)`` of one `SLICE2_STATES` entry."""
    path = os.path.expanduser(str(state["path"]).replace("{models}", models_dir))
    return path, os.path.join(models_dir, str(state["run"]))


class FillError(ValueError):
    """A precondition of the window does not hold (named)."""


# ------------------------------------------------------------------------------------------------ the CUDA ids
@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(session: Any, config: Any, items: List[Any]) -> None:
    """Loaded as a plugin (``-p main.ops.gpu_fill_window``) with ``GEN3AI_CUDA_IDS_OUT`` set: every collected item's
    id and the reason its collection-time skip marks give in THIS process (CUDA hidden), as JSON."""
    out = os.environ.get(_ENV_IDS_OUT)
    if not out:
        return
    from _pytest.skipping import evaluate_skip_marks
    rows = []
    for it in items:
        sk = evaluate_skip_marks(it)
        rows.append({"id": it.nodeid, "skip": sk.reason if sk is not None else None})
    Path(out).write_text(json.dumps(rows))


def select_cuda_ids(rows: Sequence[Mapping[str, Any]], runtime: Sequence[str] = RUNTIME_CUDA_IDS) -> List[str]:
    """The ids whose collection-time skip names CUDA / the GPU, plus the declared body-skip ids (each must be
    collected), in collection order."""
    collected = [str(r["id"]) for r in rows]
    missing = [i for i in runtime if i not in collected]
    if missing:
        raise FillError(f"RUNTIME_CUDA_IDS names test(s) no longer collected as slow: {missing}")
    want = {str(r["id"]) for r in rows if r.get("skip") and _CUDA_REASON.search(str(r["skip"]))} | set(runtime)
    return [i for i in collected if i in want]


def cuda_slow_ids(worktree: str, python: str, scratch: str, timeout_s: int = 900,
                  runtime: Sequence[str] = RUNTIME_CUDA_IDS) -> List[str]:
    """Collect ``src -m slow`` in ``worktree`` on the CPU (CUDA hidden, no GPU allowance) and select the CUDA-only
    ids. One collection, ~40 s on an agent core; it imports test modules, never runs a test."""
    out = Path(scratch) / "slow_collect.json"
    env = {k: v for k, v in os.environ.items() if k not in ("GEN3AI_TEST_ALLOW_GPU",)}
    env.update({"CUDA_VISIBLE_DEVICES": "", _ENV_IDS_OUT: str(out),
                "PYTHONPATH": os.pathsep.join([os.path.join(worktree, "src"), str(Path(__file__).resolve().parents[2])])})
    r = subprocess.run([python, "-m", "pytest", "src", "-m", "slow", "--collect-only", "-q", "-p", "no:cacheprovider",
                        "-p", "main.ops.gpu_fill_window"], cwd=worktree, env=env, capture_output=True, text=True,
                       timeout=timeout_s)
    if r.returncode != 0 or not out.is_file():
        raise FillError(f"the slow collection in {worktree} failed (rc {r.returncode}): {(r.stdout + r.stderr)[-600:]}")
    return select_cuda_ids(json.loads(out.read_text()), runtime)


def node_timeout(duration_s: Optional[float]) -> int:
    """A node's wall bound: 3 x its last banked duration, at least 600 s, rounded up to a minute; 1,800 s with no
    banked duration. (grouped_cuda's 836 s gets 2,520 s; its flat 600 s cut it on 10-24.) A timeout banks
    INCONCLUSIVE, never a semantic outcome."""
    if duration_s is None:
        return 1800
    return max(600, int(math.ceil(3.0 * float(duration_s) / 60.0)) * 60)


def banked_durations(status_file: Path) -> Dict[str, float]:
    doc = json.loads(status_file.read_text())
    return {k: float(v["duration_s"]) for k, v in doc.get("tests", {}).items()
            if isinstance(v, dict) and v.get("duration_s") is not None}


# ------------------------------------------------------------------------------------------------ the steps
def _gpu(name: str, argv: List[str], mem_gb: float, timeout_s: int) -> Dict[str, Any]:
    return {"name": name, "argv": argv, "gpu": True, "mem_gb": mem_gb, "timeout_s": timeout_s, "cwd": "worktree"}


def _cpu(name: str, argv: List[str], timeout_s: int = 600) -> Dict[str, Any]:
    return {"name": name, "argv": argv, "gpu": False, "mem_gb": 4, "timeout_s": timeout_s, "cwd": "worktree"}


def census_anchors(models_dir: str, u1000_run_dir: str) -> List[Tuple[str, str, str]]:
    """``(label, run_dir, zip)`` per census checkpoint."""
    out = []
    for run, at, z in CENSUS:
        rd = os.path.expanduser(u1000_run_dir) if run == "x26c_u1000_run" else os.path.join(models_dir, run)
        out.append((f"{run}_{at}", rd, os.path.join(rd, "checkpoints", z)))
    return out


def canary_steps(*, python: str, models_dir: str, u1000_run_dir: str, out_dir: str, marker: str) -> List[Dict[str, Any]]:
    root = os.path.join(out_dir, "canary_stage2")
    steps = [_cpu("preflight", [python, "-m", "main.ops.gpu_fill_window", "preflight-canary", "--out-dir", out_dir,
                                "--models-dir", models_dir, "--u1000-run-dir", u1000_run_dir, "--marker", marker])]
    for label, rd, z in census_anchors(models_dir, u1000_run_dir):
        steps.append(_gpu(f"stage2_{label}", [python, "-m", "main.ops.canary_dump_replay", "--anchor", z, "--run-dir", rd,
                                               "--out", os.path.join(root, label), "--device", "cuda",
                                               "--repeats", str(CANARY_REPEATS), "--batch", str(CANARY_BATCH),
                                               "--canary-shift-max", str(STAGE2_SHIFT_MAX)], 24, 900))
    steps.append(_cpu("verdict", [python, "-m", "main.ops.gpu_fill_window", "canary-verdict", "--out-dir", out_dir]))
    return steps


def slice2_steps(*, python: str, models_dir: str, out_dir: str, marker: str) -> List[Dict[str, Any]]:
    root = os.path.join(out_dir, "slice2_replay")
    steps = [_cpu("preflight", [python, "-m", "main.ops.gpu_fill_window", "preflight-slice2", "--out-dir", out_dir,
                                "--models-dir", models_dir, "--marker", marker])]
    for st in SLICE2_STATES:
        path, rd = slice2_paths(st, models_dir)
        argv = [python, "-m", "main.ops.canary_slice_read", f"--{st['mode']}", path, "--run-dir", rd,
                "--out", os.path.join(root, str(st["label"])), "--device", "cuda", "--watch", ",".join(st["watch"]),
                "--repeats", str(SLICE2_REPEATS), "--dither", str(SLICE2_DITHER)]
        for rs in st["row_sets"]:
            argv += ["--row-set", rs]
        if st.get("shift_max") is not None:
            argv += ["--canary-shift-max", str(st["shift_max"])]
        steps.append(_gpu(f"slice2_{st['label']}", argv, 24, 1800))
    roots = ",".join(os.path.join(root, str(st["label"])) for st in SLICE2_STATES)
    steps.append(_cpu("verdict", [python, "-m", "main.ops.canary_slice_read", "verdict", "--root", roots,
                                  "--verdict-out", os.path.join(root, "verdict.json")]))
    return steps


def slow_steps(*, python: str, ids: Sequence[str], durations: Mapping[str, float], out_dir: str,
               marker: str, worktree: str) -> List[Dict[str, Any]]:
    """One GPU step per node id, shortest banked duration first (a preempt banks the most ids); unknown last."""
    status = os.path.join(out_dir, "slow_gpu", "slow_tier_status.json")
    order = sorted(ids, key=lambda i: (i not in durations, durations.get(i, 0.0), i))
    steps = [_cpu("preflight", [python, "-m", "main.ops.gpu_fill_window", "preflight-slow", "--out-dir", out_dir,
                                "--worktree", worktree, "--marker", marker, *[a for i in order for a in ("--id", i)]])]
    for n, i in enumerate(order):
        steps.append(_gpu(f"slow_{n:02d}", ["env", f"GEN3AI_SLOW_STATUS_FILE={status}", python, "-m", "pytest", i,
                                            "-m", "slow", "-q", "-p", "no:xdist", "-p", "no:cacheprovider", "-rfEs"],
                          24, node_timeout(durations.get(i))))
    return steps


def _outside(path: str, roots: Iterable[str]) -> None:
    if not os.path.isabs(path):
        raise FillError(f"{path!r} must be absolute")
    p = os.path.realpath(path)
    for root in roots:
        r = os.path.realpath(root)
        if os.path.commonpath([p, r]) == r:
            raise FillError(f"{path} is inside {root}: the window writes nothing under a checkout or models/")
    for parent in [Path(p), *Path(p).parents]:
        if (parent / ".git").exists():
            raise FillError(f"{path} is inside the git checkout {parent}")


def window_items(*, worktree: str, require_commit: str, python: str, out_dir: str, models_dir: str, job_dir: str,
                 ids: Sequence[str], durations: Mapping[str, float], u1000_run_dir: str = U1000_RUN_DIR,
                 windows: Sequence[str] = DEFAULT_WINDOWS) -> List[Dict[str, Any]]:
    """The chain items of ``windows``, in `WINDOWS` order (the safe order), each with its own marker, `ON_FAIL` and
    computed budget (`window_budget_min`)."""
    _outside(out_dir, (worktree, models_dir))
    bad = [w for w in windows if w not in WINDOWS]
    if bad or not windows or len(set(windows)) != len(windows):
        raise FillError(f"--windows {list(windows)}: each of {list(WINDOWS)}, once")
    u1000_run_dir = os.path.expanduser(u1000_run_dir)
    items: List[Dict[str, Any]] = []
    for name in [w for w in WINDOWS if w in windows]:
        marker = os.path.join(job_dir, PREEMPT_MARKERS[name])
        common = {"kind": "window", "name": name, "worktree": worktree, "require_commit": require_commit,
                  "on_fail": ON_FAIL[name], "preempt_marker": marker}
        if name == "canary-stage2":
            steps = canary_steps(python=python, models_dir=models_dir, u1000_run_dir=u1000_run_dir, out_dir=out_dir,
                                 marker=marker)
            mins = [STEP_MIN["replay"]] * len(CENSUS)
            reg = ("canary_recalibration_2026-10-28.md §7 PREREG (stage 2): cap 1.2 ADOPTED iff verdict.json reads ADOPT; "
                   f"outputs {out_dir}/canary_stage2")
        elif name == "slice2-replay":
            steps = slice2_steps(python=python, models_dir=models_dir, out_dir=out_dir, marker=marker)
            mins = [STEP_MIN["slice_read"]] * len(SLICE2_STATES)
            reg = ("canary_slice2_2026-10-29 README (the pre-declared rule in main.ops.canary_slice_read): hypothesis (i) "
                   f"LIVE / RULED OUT / INCONCLUSIVE in {out_dir}/slice2_replay/verdict.json")
        else:
            steps = slow_steps(python=python, ids=ids, durations=durations, out_dir=out_dir, marker=marker,
                               worktree=worktree)
            mins = [float(durations.get(i, 1800.0)) / 60.0 for i in ids]
            reg = (f"GPU slow tier (TECH_DEBT 'GPU slow-tier filler'): rows in {out_dir}/slow_gpu/slow_tier_status.json, "
                   "merged and shipped afterwards (merge-rows); on_fail continue: never HOLDs the chain")
        items.append({**common, "budget_min": window_budget_min(mins), "x_registration": reg, "steps": steps})
    return items


#: the chain header keys copied from the base chain file (the queue's own job dir, backend and lease)
HEADER_KEYS = ("schema", "job_dir", "repo", "python", "status_file", "models_dir", "backend", "records")


def chain_doc(base: Mapping[str, Any], name: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    missing = [k for k in HEADER_KEYS if k not in base and k not in ("models_dir",)]
    if missing:
        raise FillError(f"the base chain lacks {missing}")
    return {**{k: base[k] for k in HEADER_KEYS if k in base}, "name": name, "items": items}


# ------------------------------------------------------------------------------------------------ preflights
def _fail(marker: str, why: str) -> int:
    Path(marker).write_text(f"written by gpu_fill_window preflight: {why}\n")
    print(f"[gpufill] PREFLIGHT FAILED: {why} -> wrote the preempt marker {marker}: no later step runs", flush=True)
    return 3


def preflight_canary(out_dir: str, models_dir: str, u1000_run_dir: str, marker: str) -> int:
    probs = []
    for label, rd, z in census_anchors(models_dir, u1000_run_dir):
        for f in (z, os.path.join(rd, "model_config.json"), os.path.join(rd, "metadata.json")):
            if not os.path.isfile(f):
                probs.append(f"{label}: missing {f}")
        o = os.path.join(out_dir, "canary_stage2", label)
        if os.path.isdir(o) and os.listdir(o):
            probs.append(f"{label}: {o} is not empty (a replay writes into a fresh dir)")
    if probs:
        return _fail(marker, "; ".join(probs[:6]) + (f" (+{len(probs) - 6})" if len(probs) > 6 else ""))
    print(f"[gpufill] preflight-canary PASS: {len(CENSUS)} checkpoints, outputs {out_dir}/canary_stage2", flush=True)
    return 0


def preflight_slow(out_dir: str, worktree: str, marker: str, ids: Sequence[str]) -> int:
    probs = []
    try:
        _outside(os.path.join(out_dir, "slow_gpu"), (worktree,))
    except FillError as exc:
        probs.append(str(exc))
    for i in ids:
        if not os.path.isfile(os.path.join(worktree, i.split("::")[0])):
            probs.append(f"no test file for {i} in {worktree}")
    if probs:
        return _fail(marker, "; ".join(probs[:6]))
    os.makedirs(os.path.join(out_dir, "slow_gpu"), exist_ok=True)
    print(f"[gpufill] preflight-slow PASS: {len(ids)} node ids; rows -> {out_dir}/slow_gpu/slow_tier_status.json",
          flush=True)
    return 0


def preflight_slice2(out_dir: str, models_dir: str, marker: str) -> int:
    """Every state's dump (its sha256 against the SHA256SUMS beside it) or checkpoint, its run's config and metadata,
    and an empty output dir; a failure WRITES the window's marker (exit 3)."""
    from main.ops.canary_dump_replay import Refusal, check_dump_sha
    probs = []
    for st in SLICE2_STATES:
        path, rd = slice2_paths(st, models_dir)
        try:
            if st["mode"] == "dump":
                check_dump_sha(Path(path))
            elif not os.path.isfile(path):
                probs.append(f"{st['label']}: missing {path}")
        except Refusal as exc:
            probs.append(f"{st['label']}: {exc}")
        for f in ("model_config.json", "metadata.json"):
            if not os.path.isfile(os.path.join(rd, f)):
                probs.append(f"{st['label']}: missing {os.path.join(rd, f)}")
        o = os.path.join(out_dir, "slice2_replay", str(st["label"]))
        if os.path.isdir(o) and os.listdir(o):
            probs.append(f"{st['label']}: {o} is not empty (a read writes into a fresh dir)")
    if probs:
        return _fail(marker, "; ".join(probs[:6]) + (f" (+{len(probs) - 6})" if len(probs) > 6 else ""))
    print(f"[gpufill] preflight-slice2 PASS: {len(SLICE2_STATES)} states, outputs {out_dir}/slice2_replay", flush=True)
    return 0


# ------------------------------------------------------------------------------------------------ the stage-2 rule
def _read_checkpoint(d: Path, repeats: int) -> Dict[str, Any]:
    """One checkpoint's reads: per repeat x slice, the canary arm's refusal and whether a judged ratio sits on K."""
    from agents.model.compile_regions import BF16_ENVELOPE_K
    k = float(BF16_ENVELOPE_K["param"])
    s = d / "summary.json"
    if not s.is_file():
        return {"status": "missing", "why": f"no {s}"}
    summ = json.loads(s.read_text())
    if not summ.get("complete") or int(summ.get("repeats", 0)) != repeats:
        return {"status": "incomplete", "why": f"complete {summ.get('complete')}, repeats {summ.get('repeats')}"}
    if not (summ.get("validity") or {}).get("valid"):
        return {"status": "invalid", "why": "self-validity failed"}
    reads: List[Dict[str, Any]] = []
    for r in range(repeats):
        rep = json.loads((d / f"repeat_{r:02d}.json").read_text())
        for sl, v in rep["slices"].items():
            c = v.get("canary")
            if not c:
                return {"status": "invalid", "why": f"repeat {r} slice {sl} has no canary arm (not a CUDA read)"}
            margin = float((c["watch"].get("clip_margin") or {}).get("margin", float("nan")))
            ratios = [float(p["ratio"]) for p in c.get("all_params", {}).values() if p.get("ratio") is not None]
            reads.append({"repeat": r, "slice": int(sl), "refused": bool(c["refused"]), "margin": margin,
                          "boundary": any(math.isfinite(x) and abs(x / k - 1.0) <= BOUNDARY_REL for x in ratios),
                          "max_ratio": max((x for x in ratios if math.isfinite(x)), default=None),
                          "text": c.get("text", "")[:300] if c["refused"] else ""})
    if any(abs(x["margin"] - STAGE2_MARGIN) > 1e-9 for x in reads):
        return {"status": "invalid", "why": "a read's clip margin is not 1.2", "reads": reads}
    return {"status": "read", "reads": reads}


STAGE2_RULE = ("REJECT iff some census checkpoint's canary refuses its scheduled slice (0) in >= ceil(N/2) of N repeats; "
               "else ADOPT iff all 20 checkpoints read complete, self-valid, at margin 1.2, with zero refusals on any "
               "slice and no per-parameter ratio within 0.1 % of K; else INCONCLUSIVE (cap stays 0.72)")


def stage2_verdict(per_ckpt: Mapping[str, Mapping[str, Any]], repeats: int, n_expected: int) -> Dict[str, Any]:
    """THE REGISTERED RULE (canary_recalibration §7 PREREG, before the read): REJECT iff some checkpoint's canary
    refuses its scheduled slice (0) in >= ceil(N/2) repeats; else ADOPT iff all ``n_expected`` checkpoints read
    (complete, self-valid, margin 1.2) with zero refusals on any slice and zero boundary reads; else INCONCLUSIVE."""
    need = math.ceil(repeats / 2)
    rejecting, dirty, unread = [], [], []
    for label, c in per_ckpt.items():
        if c.get("status") != "read":
            unread.append(f"{label}: {c.get('status')} ({c.get('why')})")
            continue
        sched = sum(1 for x in c["reads"] if x["slice"] == 0 and x["refused"])
        if sched >= need:
            rejecting.append(label)
        if any(x["refused"] or x["boundary"] for x in c["reads"]):
            dirty.append(label)
    if rejecting:
        verdict = "REJECT"
    elif not dirty and not unread and len(per_ckpt) == n_expected:
        verdict = "ADOPT"
    else:
        verdict = "INCONCLUSIVE"
    return {"verdict": verdict, "rule": STAGE2_RULE,
            "repeats": repeats, "need": need, "rejecting": rejecting, "refused_or_boundary": dirty, "unread": unread,
            "checkpoints": len(per_ckpt), "expected": n_expected}


def canary_verdict(out_dir: str, repeats: int = CANARY_REPEATS) -> Dict[str, Any]:
    root = Path(out_dir) / "canary_stage2"
    per = {f"{run}_{at}": _read_checkpoint(root / f"{run}_{at}", repeats) for run, at, _ in CENSUS}
    v = stage2_verdict(per, repeats, len(CENSUS))
    v["per_checkpoint"] = {k: {kk: vv for kk, vv in c.items() if kk != "reads"} |
                           ({"refusals": sum(x["refused"] for x in c["reads"]),
                             "boundary": sum(x["boundary"] for x in c["reads"]),
                             "max_ratio": max((x["max_ratio"] for x in c["reads"] if x["max_ratio"] is not None),
                                              default=None),
                             "refusal_text": [x["text"] for x in c["reads"] if x["refused"]][:2]}
                            if c.get("reads") else {}) for k, c in per.items()}
    root.mkdir(parents=True, exist_ok=True)
    (root / "verdict.json").write_text(json.dumps(v, indent=1))
    return v


# ------------------------------------------------------------------------------------------------ shipping the rows
def merge_rows(rows: Path, into: Path, main_root: Optional[str] = None) -> int:
    """Fold a scratch status file's rows into ``into`` (a worktree's committed file) by the writer's own merge;
    REFUSED when ``into`` is in the main checkout (main is never dirty) or is not an existing status file."""
    from utils.slow_tier_status import record_results
    if main_root is None:
        from utils.git import get_main_repo_root
        main_root = get_main_repo_root(cwd=str(into.parent))
    tgt = into.resolve()
    if tgt == (Path(main_root).resolve().joinpath(*STATUS_REL)):
        raise FillError(f"--into {into} is the MAIN checkout's file: merge into a worktree, then ship it")
    if not tgt.is_file():
        raise FillError(f"--into {into} is not an existing slow_tier_status.json")
    tests = json.loads(rows.read_text()).get("tests") or {}
    if not tests:
        raise FillError(f"{rows} holds no rows")
    record_results(tests, path=tgt)
    return len(tests)


# ------------------------------------------------------------------------------------------------ CLI
def _env_python() -> str:
    return str(Path.home() / "miniconda3" / "envs" / "gen3ai_torch28" / "bin" / "python3")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.ops.gpu_fill_window", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit-chain", help="the chain file (JSON) holding the two windows")
    e.add_argument("--base", required=True, help="the queue's chain file: job_dir, backend, lease and models_dir")
    e.add_argument("--name", required=True)
    e.add_argument("--worktree", required=True, help="a clean, bootstrapped SCRATCH worktree (never main)")
    e.add_argument("--require-commit", required=True)
    e.add_argument("--out-dir", required=True, help="absolute; outside every checkout and models/")
    e.add_argument("--python", default=_env_python())
    e.add_argument("--u1000-run-dir", default=U1000_RUN_DIR)
    e.add_argument("--write", default="", help="write the chain here (refuses an existing file); else stdout")
    e.add_argument("--windows", default=",".join(DEFAULT_WINDOWS), help=f"comma list from {list(WINDOWS)}")
    pc = sub.add_parser("preflight-canary")
    pc.add_argument("--out-dir", required=True)
    pc.add_argument("--models-dir", required=True)
    pc.add_argument("--u1000-run-dir", required=True)
    pc.add_argument("--marker", required=True)
    p2 = sub.add_parser("preflight-slice2")
    p2.add_argument("--out-dir", required=True)
    p2.add_argument("--models-dir", required=True)
    p2.add_argument("--marker", required=True)
    ps = sub.add_parser("preflight-slow")
    ps.add_argument("--out-dir", required=True)
    ps.add_argument("--worktree", required=True)
    ps.add_argument("--marker", required=True)
    ps.add_argument("--id", action="append", default=[])
    cv = sub.add_parser("canary-verdict")
    cv.add_argument("--out-dir", required=True)
    cv.add_argument("--repeats", type=int, default=CANARY_REPEATS)
    mr = sub.add_parser("merge-rows")
    mr.add_argument("--rows", required=True)
    mr.add_argument("--into", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "emit-chain":
            base = json.loads(Path(a.base).read_text())
            wt = os.path.realpath(a.worktree)
            from utils.git import get_main_repo_root
            if wt == os.path.realpath(get_main_repo_root(cwd=wt)):
                raise FillError(f"--worktree {a.worktree} is the MAIN checkout: the window runs in a scratch worktree")
            models_dir = str(base.get("models_dir") or os.path.join(base["repo"], "models"))
            _outside(a.out_dir, (wt, models_dir, base["repo"]))
            os.makedirs(a.out_dir, exist_ok=True)
            windows = [w for w in a.windows.split(",") if w.strip()]
            ids = cuda_slow_ids(wt, a.python, a.out_dir) if "slow-gpu" in windows else []
            durations = banked_durations(Path(wt).joinpath(*STATUS_REL))
            items = window_items(worktree=wt, require_commit=a.require_commit, python=a.python, out_dir=a.out_dir,
                                 models_dir=models_dir, job_dir=base["job_dir"], ids=ids, durations=durations,
                                 u1000_run_dir=a.u1000_run_dir, windows=windows)
            text = json.dumps(chain_doc(base, a.name, items), indent=1) + "\n"
            if a.write:
                if os.path.exists(a.write):
                    raise FillError(f"--write {a.write} exists: a chain file is never overwritten")
                Path(a.write).write_text(text)
                print(f"[gpufill] wrote {a.write}: " + "; ".join(
                    f"{it['name']} ({len(it['steps'])} steps, budget {it['budget_min']:g} min, on_fail {it['on_fail']})"
                    for it in items))
            else:
                sys.stdout.write(text)
            return 0
        if a.cmd == "preflight-canary":
            return preflight_canary(a.out_dir, a.models_dir, a.u1000_run_dir, a.marker)
        if a.cmd == "preflight-slice2":
            return preflight_slice2(a.out_dir, a.models_dir, a.marker)
        if a.cmd == "preflight-slow":
            return preflight_slow(a.out_dir, a.worktree, a.marker, a.id)
        if a.cmd == "canary-verdict":
            v = canary_verdict(a.out_dir, a.repeats)
            print(f"[gpufill] STAGE-2 VERDICT {v['verdict']}: {v['checkpoints']} of {v['expected']} checkpoints; "
                  f"rejecting {v['rejecting']}; refused/boundary {v['refused_or_boundary']}; unread {v['unread'][:3]} "
                  f"-> {a.out_dir}/canary_stage2/verdict.json", flush=True)
            return 0
        n = merge_rows(Path(a.rows), Path(a.into))
        print(f"[gpufill] merged {n} row(s) from {a.rows} into {a.into}")
        return 0
    except FillError as exc:
        print(f"[gpufill] REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
