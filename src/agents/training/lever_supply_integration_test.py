"""`gen3_supply_guard_v2` with REAL processes — the trainer's own exit code.

`lever_supply_test.py` pins each guard; this pins what a launch actually DOES:

* (slow, sim) a `--self-play` run whose seeding gate it cannot reach COMPLETES (exit 0) and says so on every
  cycle — a fresh run below the gate is not a starved supply (`gen3_supply_guard_v3`, 2026-10-19; before, the
  same argv exited ``FATAL_SUPPLY`` (5)). The FATAL itself — a pool that stays empty after the gate opened, or an
  eval cycle that fails — is pinned in `lever_supply_test.py` through the real `_collect_pending` and
  `exit_code_for` (a real trainer cannot be made to fail its seeding step from an argv).

The trainer is started through `runpy` with its module name split, so no argv on this box carries
the trainer's literal name (watchers key on it). Every run dir is under `tmp_path`.
"""
from __future__ import annotations


import pytest

from main.exit_codes import TrainExitCode
from main.train.fatal_config_exits_integration_test import _trainer

pytestmark = [pytest.mark.integration, pytest.mark.slow, pytest.mark.sim]


def test_a_self_play_run_whose_seeding_gate_it_cannot_reach_completes_and_says_so(tmp_path, run_archive):
    """The `ai_v12_27` shape end to end: `--self-play` with a seeding gate the run cannot reach
    (`--self-play-start-wr 1.01`), so every eval cycle leaves the pool EMPTY. Since `gen3_supply_guard_v3` that is a
    fresh run below its gate, not a starved supply: the trainer EXITS 0 (it used to exit 5 here) with a loud line per
    cycle and the end-of-segment ZERO summary, even with the floor typed at 2. Takes the `run_archive` fixture: the
    trainer child inherits `$GEN3AI_MODELS_DIR` from it — without it the sealed test archive refuses the child's run
    directory (`RunArchiveError`, exit 3 = FATAL_CONFIG)."""
    rc, text = _trainer(tmp_path, "--debug-eval", "--self-play", "--self-play-start-wr", "1.01",
                        "--steps", "60000", "--supply-starve-cycles", "self_play_pool=2",
                        "--run-dir", str(tmp_path / "run"), timeout=2400)
    # 2400 s (was 1200): measured 2026-10-21 beside a live run, the 60000-step debug run takes 1312 s and COMPLETES
    # (exit 0, 15 loud lines, one ZERO summary, no FATAL) -- the old bound sat under the measured cost, so the test
    # banked INCONCLUSIVE (a timeout, never a semantic outcome) instead of the verdict.
    assert rc == int(TrainExitCode.COMPLETE), (rc, text[-4000:])
    assert "[SUPPLY] FATAL" not in text
    assert text.count("is below the seeding gate --self-play-start-wr 1.01") >= 2, "one loud line per gated cycle"
    assert "🚨🚨 [SUPPLY] ZERO" in text
    assert "Training complete" in text
