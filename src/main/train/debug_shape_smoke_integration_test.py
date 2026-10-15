"""`--arch production --debug --steps 10000` REACHES ITS UPDATES and exits 0 (`main.train.debug_shape`).

Before the override the production recipe's 98,304-row update target was kept under `--debug`'s ONE
CPU env, so this smoke never reached its first update — it ran for hours (three agents, 2026-10-07).
Measured with the override (2026-10-07, CPU, beside other agents' jobs): 5 updates, exit 0, ~6 min
wall. On a revert the bounded wait below times out instead (or exits with no update).

The trainer is started through `runpy` with its module name split, so no argv on this box carries the
trainer's literal name (watchers key on it). ``slow``: minutes of CPU. It runs at ONE torch / BLAS / OpenMP thread
(`debug_shape.SMOKE_THREAD_ENV`, pinned by `debug_shape_test.py`) so that beside a live run it is slow, not hung — and
under the STRICT metric-producer guard (`PYTEST_CURRENT_TEST` reaches the child, and `--debug` arms it itself):
a metric key with two producers fails the smoke (`gen3_metric_producer_guard_v1`).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

from main.train.debug_shape import SMOKE_THREAD_ENV
from utils.contention import scale_timeout
from utils.paths import repo_root, src_root

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_RUNNER = ("import runpy, sys; sys.argv = ['trainer'] + sys.argv[1:]; "
           "runpy.run_module('main.train_rl' '_agent', run_name='__main__', alter_sys=True)")


@pytest.mark.parametrize("arch", ["production", "endstate"])
def test_arch_debug_smoke_reaches_an_update_and_exits_0(tmp_path, run_archive, arch):
    """`production` and the NAMED ARM `endstate` (gen3_endstate_facts_v1, an alias of production since the 2026-10-12
    adoption; `main.train.arch_arms`) — an arm's run records every lever in its `model_config.json`. (`static_recovery`
    was deleted by the legacy deletion's U0.)"""
    log = tmp_path / "trainer.log"
    with open(log, "wb") as fh:
        proc = subprocess.Popen(
            [sys.executable, "-c", _RUNNER, "--arch", arch, "--debug", "--steps", "10000"],
            stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=str(repo_root()),
            env={**os.environ, "PYTHONPATH": str(src_root()), "CUDA_VISIBLE_DEVICES": "", **SMOKE_THREAD_ENV})
        try:
            # 1800 s: `endstate` (a 3-round trunk + the exact P(KO)) ran 884-1039 s on a quiet box, within 13 % of a
            # 1200 s bound (timed out beside the routine gate, 2026-10-09); a timeout is never the semantic outcome.
            rc = proc.wait(timeout=scale_timeout(1800))
        except subprocess.TimeoutExpired:
            rc = None
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=60)
    text = log.read_text(errors="replace")
    assert rc == 0, (rc, text[-3000:])
    assert "[DEBUG SHAPE]" in text and "update at >= 2,304 completed-game rows" in text, text[-3000:]
    updates = [int(m) for m in re.findall(r"\|\s+n_updates\s+\|\s+(\d+)", text)]
    assert updates and max(updates) >= 1, text[-3000:]
    assert "Training complete." in text
    # gen3_metric_key_collision_v1: >= 2 `dump_logs` tables, the post-update one carrying the production network's
    # `cell/*` liveness rows — the 2026-10-15 crash (every production run at a commit >= 65b71f6a died at its FIRST
    # dump holding them: '   move_resolution_restored_move_weight_norm' and its `grad_norm` twin cut alike, and sb3's
    # table raised). Both rows now render as two distinct rows (in full: the column is 48). Fails on revert (the run
    # crashes at that dump).
    from agents.training.train_logger import truncate_label
    assert text.count("total_timesteps") >= 2, text[-3000:]
    for half in ("weight_norm", "grad_norm"):
        label = truncate_label(f"   move_resolution_restored_move_{half}")
        assert re.search(re.escape(f"| {label}") + r" *\|", text), (label, text[-3000:])
    # the bf16 trunk is the production RECIPE (2026-10-14): `--arch <umbrella>` applied it, the trainer announced it
    # and the run recorded it (the CPU path runs the bf16 autocast rounds: this box's CPU bf16 update is as fast as fp32)
    assert "[TRUNK PRECISION] bf16" in text, text[-3000:]
    import json
    cfgs = sorted(run_archive.rglob("model_config.json"))
    assert len(cfgs) == 1, cfgs
    rec = json.loads(cfgs[0].read_text())
    assert rec.get("trunk_precision") == "bf16", rec.get("trunk_precision")
    if arch != "production":
        from main.train.arch_arms import arm_overlay
        for key, value in arm_overlay(arch).items():
            assert rec.get(key) == value, (key, rec.get(key), value)
        assert str(rec.get("arch_source", "")).startswith(f"{arch}@"), rec.get("arch_source")
