"""The frozen GPU benchmark suite's CPU CODE PATH, end to end, as real subprocesses (no GPU: `CUDA_VISIBLE_DEVICES=""`).

* `test_update_cell_quick` (integration, ~15 s): one update phase at the quick geometry — the seeded learner, the
  fixture rollout, the schedule, the two-point estimate — writes a result the schema accepts;
* `test_quick_suite_end_to_end` (integration + slow, ~2-3 min on 3 busy cores): every phase, including the
  collection worker running the REAL trainer in process against the fixture pool. It pins the 2026-10-18 bug class:
  fixture snapshots built off the trainee's spaces were REFUSED by T2 (`SlotArchMismatch`) at the first opponent
  load, so the collect phase must finish with the opponents actually served by the pool (`p2_policy_share` > 0)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agents.training import gpu_bench_suite as S

pytestmark = pytest.mark.integration


def _env() -> dict:
    from utils.paths import src_root
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1")
    env["PYTHONPATH"] = str(src_root()) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


@pytest.mark.slow  # ship gate (2026-10-19): 17 s beside a live run
def test_update_cell_quick(tmp_path: Path) -> None:
    (tmp_path / "phases").mkdir()
    cfg = S.phase_config("update_fp32", True, "cpu", str(tmp_path))
    cfg_path = tmp_path / "phases" / "update_fp32.cfg.json"
    cfg_path.write_text(json.dumps(cfg))
    r = subprocess.run([sys.executable, "-m", "agents.training.gpu_bench_suite", "_phase", "update_fp32",
                        str(cfg_path)], env=_env(), capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    body = json.loads(Path(cfg["out"]).read_text())
    assert body["status"] == "ok"
    assert body["shape"]["rows"] == 32 and body["shape"]["batch_size"] == 16 and body["shape"]["grad_accum"] == 2
    kinds = [c["kind"] for c in body["calls"]]
    assert kinds == ["warm", "timed", "timed", "diag"]
    assert body["train"]["update_s_est"] > 0 and body["gate"]["verdict"] == "SKIPPED"
    assert "fixture_source" in body


@pytest.mark.slow
def test_quick_suite_end_to_end(tmp_path: Path) -> None:
    out = tmp_path / "suite"
    r = subprocess.run([sys.executable, "-m", "agents.training.gpu_bench_suite", "run", "--out", str(out), "--quick",
                        "--tag", "pytest", "--meta", "dph=0.0"], env=_env(), capture_output=True, text=True,
                       timeout=2400)
    suite = json.loads((out / "gpu_bench_suite.json").read_text())
    logs = {p.name: p.read_text()[-2000:] for p in (out / "logs").glob("*.log")}
    assert r.returncode == 0, (r.stdout[-3000:], suite.get("validation"), logs.get("collect.log"))
    assert S.validate(suite) == []
    col = suite["phases"]["collect"]
    assert col["pool"]["n_snapshots"] == 2
    assert col["pool"]["expected_self_play_fraction"] == pytest.approx(0.9)
    assert col["buffer_shape"]["rows"] == 32
    assert col["summary"]["p2_policy_share_median"] > 0.0      # the opponents were served from the fixture pool
    assert col["scratch_removed"] is True
    assert suite["meta"]["dph"] == 0.0
    row = S.card_row(suite)
    assert row["update_bf16_s"] and row["env_steps_per_s"]
