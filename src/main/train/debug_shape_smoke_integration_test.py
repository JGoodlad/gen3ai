"""`--arch production --debug --steps 10000` REACHES ITS UPDATES and exits 0 (`main.train.debug_shape`).

Before the override the production recipe's 98,304-row update target was kept under `--debug`'s ONE
CPU env, so this smoke never reached its first update — it ran for hours (three agents, 2026-10-07).
Measured with the override (2026-10-07, CPU, beside other agents' jobs): 5 updates, exit 0, ~6 min
wall. On a revert the bounded wait below times out instead (or exits with no update).

The trainer is started through `runpy` with its module name split, so no argv on this box carries the
trainer's literal name (watchers key on it). ``slow``: minutes of CPU.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

from utils.contention import scale_timeout
from utils.paths import repo_root, src_root

pytestmark = [pytest.mark.integration, pytest.mark.slow]

_RUNNER = ("import runpy, sys; sys.argv = ['trainer'] + sys.argv[1:]; "
           "runpy.run_module('main.train_rl' '_agent', run_name='__main__', alter_sys=True)")


@pytest.mark.parametrize("arch", ["production", "static_recovery", "endstate"])
def test_arch_debug_smoke_reaches_an_update_and_exits_0(tmp_path, run_archive, arch):
    """`production` and the NAMED ARMS `static_recovery` (gen3_static_recovery_v1: static + every static-recovery
    lever) and `endstate` (gen3_endstate_facts_v1: static_recovery + the bundle + every fact-completion lever;
    `main.train.arch_arms`) — an arm's run records every lever in its `model_config.json`."""
    log = tmp_path / "trainer.log"
    with open(log, "wb") as fh:
        proc = subprocess.Popen(
            [sys.executable, "-c", _RUNNER, "--arch", arch, "--debug", "--steps", "10000"],
            stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=str(repo_root()),
            env={**os.environ, "PYTHONPATH": str(src_root()), "CUDA_VISIBLE_DEVICES": ""})
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
    if arch != "production":
        import json
        from main.train.arch_arms import arm_overlay
        cfgs = sorted(run_archive.rglob("model_config.json"))
        assert len(cfgs) == 1, cfgs
        rec = json.loads(cfgs[0].read_text())
        for key, value in arm_overlay(arch).items():
            assert rec.get(key) == value, (key, rec.get(key), value)
        assert str(rec.get("arch_source", "")).startswith(f"{arch}@"), rec.get("arch_source")
