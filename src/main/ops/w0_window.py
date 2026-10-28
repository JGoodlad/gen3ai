"""W0, the housekeeping GPU window of 2026-10-28 (``designs/ops/gpu_plan_2026-10-27.md`` §3 item 2), as an arm-chain
window item (``designs/ops/arm_chain.md`` §2, ``main.ops.arm_chain.window``). Three parts, one process per step:

1. **the B = 8 compile cell** (``extractor_compiles_test.py::test_production_arch_compiles_on_cuda``, the CUDA
   compile row of TECH_DEBT; ~5 min);
2. **the X48 canary-dump replay** (``main.ops.canary_dump_replay``): the archived u200 UNCONFIRMED dump of
   ``rb_x35_x48kffn_s3502``, compiled vs eager bf16 backward on edge_bias, a few repeats;
3. **the CUDA canary census** (``canary_row_null_design_2026-10-25``): ``census_read.py`` once per checkpoint of the
   X26 lineage, then ``null_cost.py``; ``derive_k.py`` afterwards on CPU.

**THE NESTED-LOCK QUESTION** (gpu_plan FINDING 5), answered by reading the code. A window step marked ``"gpu": true``
runs as ``gpu_lock.sh mem_cap.sh <GB> timeout <s> <argv>`` with the chain's lease token file in its environment.
``gpu_census.sh`` wraps each checkpoint in ``gpu_lock.sh mem_cap.sh 24 timeout 1200`` itself. The inner ``gpu_lock``
would be a no-op (``utils.gpu_lock.gpu_lock``: a verified ancestor holder, or the caller's token matching the lease,
yields without a second flock), and the inner ``mem_cap`` would start a SIBLING systemd scope, so it would run. But
the script swallows each checkpoint's exit code and always exits 0, so the window would read PASS with every read
failed, and the window's per-step lease re-check (CB-34) and budget would see one 45-75-min step. So W0 does NOT call
``gpu_census.sh``: each checkpoint is its own ``"gpu": true`` step running ``census_read.py`` directly (the same argv
the script runs inside its wrappers), the u1000 neighbours' scratch run dir is a CPU step (``prep-u1000``), and the
cost read is the last step. ``census_lists`` must equal the script's own lists (``w0_window_test``).

    python -m main.ops.w0_window emit --repo R --require-commit SHA --out-dir D   # the window item (JSON)
    python -m main.ops.w0_window prep-u1000 --out-dir D/census                     # the CPU step (the window runs it)

Nothing here touches the GPU itself, writes under ``models/`` or a checkout; every output goes under ``--out-dir``.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

CENSUS_DIR = "designs/research_state/measurements/canary_row_null_design_2026-10-25"
#: gpu_census.sh's checkpoint lists (``w0_window_test`` parses the script and asserts these are its lists).
X26_AT = ("100M", "180M", "230M", "280M", "330M", "380M")
X26C_AT = ("390M", "400M", "410M", "420M", "430M", "440M", "450M", "460M", "470M", "480M", "481M", "482M")
U1000_AT = ("478M", "479M")
X26_RUN = "rb_x26_s2601"
X26C_RUN = "rb_x26c_s2601"
#: the archived u1000 neighbours of the X26 continuation (pruned from the run dir; gpu_census.sh's ARC)
U1000_ARCHIVE = "gen3ai_archive/x26c_canary_u1000_2026-10-25"
U1000_RUN = "x26c_u1000_run"
#: the archived X48 dump (b5cd22862's canary row); the replay checks its sha256 against SHA256SUMS beside it
X48_DUMP_DIR = "gen3ai_archive/x48kffn_s3502_canary_u200_unconfirmed_2026-10-28"
X48_DUMP = "canary_u200_unconfirmed_policy.pt"
X48_RUN = "rb_x35_x48kffn_s3502"
COMPILE_CELL = "src/agents/model/extractor_compiles_test.py::test_production_arch_compiles_on_cuda"
_CKPT = re.compile(r"checkpoint_(\d+)_steps\.zip$")


class W0Error(ValueError):
    """A precondition of the window does not hold (named)."""


def census_lists() -> Dict[str, Tuple[str, ...]]:
    return {X26_RUN: X26_AT, X26C_RUN: X26C_AT, U1000_RUN: U1000_AT}


def last_checkpoint(run_dir: Path) -> Path:
    """gpu_census.sh's ``LAST`` (the newest checkpoint zip by step)."""
    ck = sorted(((int(m.group(1)), p) for p in (run_dir / "checkpoints").glob("checkpoint_*_steps.zip")
                 if (m := _CKPT.search(p.name))), key=lambda t: t[0])
    if not ck:
        raise W0Error(f"{run_dir} has no checkpoint zip")
    return ck[-1][1]


def _gpu(name: str, argv: List[str], mem_gb: float, timeout_s: int) -> Dict[str, Any]:
    return {"name": name, "argv": argv, "gpu": True, "mem_gb": mem_gb, "timeout_s": timeout_s, "cwd": "worktree"}


def steps(*, worktree: str, python: str, out_dir: str, models_dir: str, home: str,
          replay_repeats: int = 5) -> List[Dict[str, Any]]:
    """The window's steps, in order: the compile cell, the X48 replay, the u1000 prep (CPU), the census, the cost."""
    here = os.path.join(worktree, CENSUS_DIR)
    census_out = os.path.join(out_dir, "census")
    out: List[Dict[str, Any]] = [
        _gpu("compile_cell_b8", [python, "-m", "pytest", COMPILE_CELL, "-m", "slow", "-q", "-p", "no:xdist",
                                 "-p", "no:cacheprovider", "-rfEs"], 16, 900),
        _gpu("x48_dump_replay", [python, "-m", "main.ops.canary_dump_replay",
                                 "--dump", os.path.join(home, X48_DUMP_DIR, X48_DUMP),
                                 "--run-dir", os.path.join(models_dir, X48_RUN),
                                 "--out", os.path.join(out_dir, "x48_replay"),
                                 "--device", "cuda", "--repeats", str(replay_repeats)], 24, 1200),
    ]
    have_u1000 = os.path.isfile(os.path.join(home, U1000_ARCHIVE, "checkpoint_478000208_steps.zip"))
    if have_u1000:
        out.append({"name": "census_prep_u1000", "gpu": False, "mem_gb": 4, "timeout_s": 600, "cwd": "worktree",
                    "argv": [python, "-m", "main.ops.w0_window", "prep-u1000", "--out-dir", census_out,
                             "--models-dir", models_dir, "--archive", os.path.join(home, U1000_ARCHIVE)]})
    runs = [(os.path.join(models_dir, X26_RUN), X26_AT), (os.path.join(models_dir, X26C_RUN), X26C_AT)]
    if have_u1000:
        runs.append((os.path.join(census_out, U1000_RUN), U1000_AT))
    for run_dir, ats in runs:
        for at in ats:
            out.append(_gpu(f"census_{os.path.basename(run_dir)}_{at}",
                            [python, os.path.join(here, "census_read.py"), "--run-dir", run_dir, "--at", at,
                             "--out", census_out, "--device", "cuda", "--dither", "32", "--vf-coef", "0.5"],
                            24, 1200))
    last = last_checkpoint(Path(models_dir) / X26C_RUN)
    out.append(_gpu("census_cost", [python, os.path.join(here, "null_cost.py"), "--ckpt", str(last),
                                    "--out", os.path.join(census_out, "cost_cuda.json"), "--device", "cuda",
                                    "--k", "16", "--vf-coef", "0.5"], 24, 900))
    return out


def window_item(*, worktree: str, require_commit: str, python: str, out_dir: str, models_dir: str, home: str,
                budget_min: float = 150, on_fail: str = "continue", preempt_marker: str = "NO_GO_W0",
                replay_repeats: int = 5) -> Dict[str, Any]:
    if not os.path.isabs(out_dir):
        raise W0Error(f"--out-dir must be absolute, got {out_dir!r}")
    for root in (worktree, models_dir):
        if os.path.commonpath([os.path.abspath(out_dir), os.path.abspath(root)]) == os.path.abspath(root):
            raise W0Error(f"--out-dir {out_dir} is inside {root}: the window writes nothing under a checkout or models/")
    return {"kind": "window", "name": "w0", "worktree": worktree, "require_commit": require_commit,
            "budget_min": budget_min, "on_fail": on_fail, "preempt_marker": preempt_marker,
            "steps": steps(worktree=worktree, python=python, out_dir=out_dir, models_dir=models_dir, home=home,
                           replay_repeats=replay_repeats)}


def prep_u1000(out_dir: str, models_dir: str, archive: str) -> List[str]:
    """gpu_census.sh's u1000 block: a scratch run dir under ``out_dir`` holding COPIES of the two archived
    neighbours and the live run's config + metadata (read only; a symlink would resolve to the archive dir, which has
    no metadata.json). Never overwrites (``cp -n``)."""
    u = Path(out_dir) / U1000_RUN
    (u / "checkpoints").mkdir(parents=True, exist_ok=True)
    done: List[str] = []
    src_run = Path(models_dir) / X26C_RUN
    pairs = [(src_run / f, u / f) for f in ("model_config.json", "metadata.json")]
    zips = sorted(Path(archive).glob("checkpoint_47[89]*_steps.zip"))
    if len(zips) != 2:
        raise W0Error(f"{archive}: expected the 478M and 479M checkpoints, found {[z.name for z in zips]}")
    pairs += [(z, u / "checkpoints" / z.name) for z in zips]
    for s, d in pairs:
        if not s.is_file():
            raise W0Error(f"{s} is missing")
        if d.exists():
            done.append(f"{d.name}: present")
            continue
        shutil.copy2(s, d)
        done.append(f"{d.name}: copied")
    return done


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.ops.w0_window", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("emit", help="print the W0 window item (JSON) for a chain file")
    e.add_argument("--repo", default="/home/goodlad/dev/gen3ai", help="the window's checkout (clean, >= the commit)")
    e.add_argument("--require-commit", required=True)
    e.add_argument("--python", default="/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3")
    e.add_argument("--out-dir", required=True, help="absolute; outside every checkout and models/")
    e.add_argument("--models-dir", default=None, help="default: <repo>/models")
    e.add_argument("--home", default=os.path.expanduser("~"))
    e.add_argument("--budget-min", type=float, default=150)
    e.add_argument("--repeats", type=int, default=5)
    p = sub.add_parser("prep-u1000", help="the census's u1000 scratch run dir (CPU; copies only)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--models-dir", required=True)
    p.add_argument("--archive", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "emit":
            md = a.models_dir or os.path.join(a.repo, "models")
            print(json.dumps(window_item(worktree=a.repo, require_commit=a.require_commit, python=a.python,
                                         out_dir=a.out_dir, models_dir=md, home=a.home, budget_min=a.budget_min,
                                         replay_repeats=a.repeats), indent=1))
            return 0
        for line in prep_u1000(a.out_dir, a.models_dir, a.archive):
            print(f"[w0 prep-u1000] {line}")
        return 0
    except W0Error as exc:
        print(f"[w0] REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
