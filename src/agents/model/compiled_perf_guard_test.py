"""K8's MILESTONE performance guard (`gen3_no_silent_eager_v1`; GPU tier, `slow`): a production
update through the K8 regions still runs at the acceptance read's speed and compiled share.

Owner, 2026-10-01: "I just most care that we don't have silent performance regressions from it not
being fully compiled." The PRIMARY defences are in the run itself — the dispatchers' typed FATAL on a
region that ran eager, the inventory assertion at startup, the per-update `lifecycle/*` counters and
the update-wall drift warning (`compile_control.TrainMsWatch`). This test is the MILESTONE check:
the slow tier runs at major points (after the cutover, at a new era or lineage start) or when an agent
judges it warranted (owner, 2026-10-01: no nightly slow tier) — its banked verdict then fails
everyone's routine gate until fixed.

It runs `python -m main.compile_inventory run --stage time` (torch 2.8, CUDA, the prewarm and the
lock kept, one un-bracketed update) on arm C's final checkpoint and the learner benchmark's pinned
buffer — the SAME tool, checkpoint, buffer and shape the baseline was banked on — against the fp32
'highest' baseline `perf_baseline.json` holds (fp32 is the ONLY precision: TF32 was retired, deletion
pass K2, and the K8 acceptance's TF32 36.32 s is kept there as history, never compared against). It
asserts the run's recorded precision, rows, micro-batch and epochs equal the baseline's, the regions
installed, the update wall within the baseline's tolerance, and the compiled share of the update wall
>= 0.80. SKIPS, naming the reason, where the box lacks the checkpoint or the buffer — and while NO fp32
baseline is banked (`baseline: null`: the time stage did not reuse the pinned buffer under the Rust env
core when K2 tried to bank one, 2026-10-02; the file says why), so the slow-tier status reads this guard
as unmeasured, never as a pass."""
from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import pytest
import torch

from agents.model.extractor_compiles_test import _cuda_skip_reason
from utils.paths import main_models_dir, repo_path, src_path

_skip_cuda = pytest.mark.skipif(_cuda_skip_reason() is not None, reason=_cuda_skip_reason() or "")
BASELINE = repo_path("designs", "research_state", "measurements", "k6_k8", "acceptance",
                     "perf_baseline.json")
BUFFER = Path(os.environ.get("GEN3AI_PERF_GUARD_BUFFER", str(
    Path.home() / "gen3ai_archive" / "learner_bench" / "20260928_135948_cuda" / "rollout_buffer.pkl")))


def _checkpoint() -> Optional[Path]:
    m = main_models_dir()
    p = None if m is None else m / "ai_v14_02_lbat_ctrl" / "final_model.zip"
    return p if p is not None and p.is_file() else None


_BASELINE = json.loads(BASELINE.read_text()).get("baseline")
#: The shape the baselines were banked at (the pinned buffer under arm C's own flags).
_SHAPE = {"rows": 98304, "micro_batch": 2048, "grad_accum_steps": 32, "n_epochs": 10}


@_skip_cuda
@pytest.mark.skipif(not torch.__version__.startswith("2.8"), reason="the K8 regions are torch 2.8")
@pytest.mark.skipif(_checkpoint() is None or not BUFFER.is_file(),
                    reason="arm C's checkpoint or the pinned learner buffer is not on this box")
@pytest.mark.skipif(_BASELINE is None, reason="no fp32 'highest' baseline is banked in perf_baseline.json "
                    "(TF32 retired, K2; the file says why none could be banked) — NO measurement is taken")
@pytest.mark.slow
def test_a_production_update_keeps_the_acceptance_speed_and_compiled_share():
    precision = "highest"
    root = json.loads(BASELINE.read_text())
    base = root["baseline"]
    with tempfile.TemporaryDirectory(prefix="perf_guard_") as out:
        cmd = [sys.executable, "-m", "main.compile_inventory", "run", "--stage", "time",
               "--device", "cuda", "--buffer", str(BUFFER),
               "--model", str(_checkpoint()), "--keep-prewarm", "--unbracketed",
               "--worker-timeout-min", "25", "--out-root", out]
        env = {**os.environ, "PYTHONPATH": str(src_path())}
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=40 * 60)
        if proc.returncode != 0 and "REFUSED: the GPU is not idle" in proc.stdout + proc.stderr:
            # the time stage is a MEASUREMENT and refuses a box running another trainer: a precondition,
            # never a verdict on the code (benchmarks warn, never stretch)
            msg = "the time stage refused: another trainer process is running on this box"
            print(f"\n⚠️  [PerfGuard] SKIPPED — {msg}. NO measurement was taken; the slow-tier status "
                  f"keeps reporting this guard as unmeasured until a run on an idle box.",
                  file=sys.stderr, flush=True)
            pytest.skip(msg)
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
        res = json.loads(Path(glob.glob(os.path.join(out, "*", "time_result.json"))[-1]).read_text())
        ana = json.loads(Path(glob.glob(os.path.join(out, "*", "time_analysis.json"))[-1]).read_text())
    # the run must be the baseline's own configuration — else the comparison is meaningless
    assert res["matmul_precision"] == precision, (res["matmul_precision"], precision)
    geo = res["geometry"]
    assert {k: geo[k] for k in _SHAPE} == _SHAPE, geo
    assert res.get("compiled_regions") is True, "the K8 regions were not installed"
    wall = float(res["unbracketed"]["train_ms"]) / 1000.0
    bar = float(base["update_wall_s"]) * (1.0 + float(base["wall_tolerance"]))
    assert wall <= bar, (f"one production update took {wall:.2f} s at {precision!r}, over {bar:.2f} s "
                         f"(the banked {base['update_wall_s']} s + {base['wall_tolerance']:.0%}) — a "
                         f"silent performance regression (load? check the run's lifecycle/eager_share)")
    share = float(ana["profiles"][0]["totals"]["compiled_share_of_train_wall"])
    assert share >= float(root["min_compiled_share"]), (
        f"compiled share of the update wall {share:.3f} < {root['min_compiled_share']} at "
        f"{precision!r}: part of the learner is no longer compiled")
