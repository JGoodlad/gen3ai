"""K8's MILESTONE performance guard (`gen3_no_silent_eager_v1`; GPU tier, `slow`): a production
update through the K8 regions still runs at the acceptance read's speed and compiled share.

Owner, 2026-10-01: "I just most care that we don't have silent performance regressions from it not
being fully compiled." The PRIMARY defences are in the run itself — the dispatchers' typed FATAL on a
region that ran eager, the inventory assertion at startup, the per-update `lifecycle/*` counters and
the update-wall drift warning (`compile_control.TrainMsWatch`). This test is the MILESTONE check:
the slow tier runs at major points (after the cutover, at a new era or lineage start) or when an agent
judges it warranted (owner, 2026-10-01: no nightly slow tier) — its banked verdict then fails
everyone's routine gate until fixed.

It runs `python -m main.compile_inventory run --stage time` (torch 2.8, CUDA, TF32, the prewarm and
the lock kept, one un-bracketed update) on arm C's final checkpoint and the learner benchmark's pinned
buffer, and holds it to `designs/research_state/measurements/k6_k8/acceptance/perf_baseline.json`:
the regions installed, the update wall within the declared tolerance of the banked 36.32 s, and the
compiled share of the update wall >= 0.80. SKIPS where the box lacks the checkpoint or the buffer."""
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


@_skip_cuda
@pytest.mark.skipif(not torch.__version__.startswith("2.8"), reason="the K8 regions are torch 2.8")
@pytest.mark.skipif(_checkpoint() is None or not BUFFER.is_file(),
                    reason="arm C's checkpoint or the pinned learner buffer is not on this box")
@pytest.mark.slow
def test_a_production_update_keeps_the_acceptance_speed_and_compiled_share():
    base = json.loads(BASELINE.read_text())
    with tempfile.TemporaryDirectory(prefix="perf_guard_") as out:
        cmd = [sys.executable, "-m", "main.compile_inventory", "run", "--stage", "time",
               "--device", "cuda", "--matmul-precision", "high", "--buffer", str(BUFFER),
               "--model", str(_checkpoint()), "--keep-prewarm", "--unbracketed",
               "--worker-timeout-min", "25", "--out-root", out]
        env = {**os.environ, "PYTHONPATH": str(src_path())}
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=40 * 60)
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
        res = json.loads(Path(glob.glob(os.path.join(out, "*", "time_result.json"))[-1]).read_text())
        ana = json.loads(Path(glob.glob(os.path.join(out, "*", "time_analysis.json"))[-1]).read_text())
    assert res.get("compiled_regions") is True, "the K8 regions were not installed"
    wall = float(res["unbracketed"]["train_ms"]) / 1000.0
    bar = float(base["update_wall_s"]) * (1.0 + float(base["wall_tolerance"]))
    assert wall <= bar, (f"one production update took {wall:.2f} s, over {bar:.2f} s (the banked "
                         f"{base['update_wall_s']} s + {base['wall_tolerance']:.0%}) — a silent "
                         f"performance regression (load? check the run's lifecycle/eager_share)")
    share = float(ana["profiles"][0]["totals"]["compiled_share_of_train_wall"])
    assert share >= float(base["min_compiled_share"]), (
        f"compiled share of the update wall {share:.3f} < {base['min_compiled_share']}: part of the "
        f"learner is no longer compiled")
