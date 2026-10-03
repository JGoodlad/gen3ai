"""The Rust eval core's SAFE POINTS (P10-A2, `main.train.deferred_abort`) on a REAL cycle: `RustEvalCore.run_cycle`
calls the run's `safe_point` at the top of EVERY host step, so a stop signal / forced checkpoint that lands
during the in-process eval waits one host step, never the whole cycle (9.6-16.3 s at the production roster).

The decision is a count: one call per host step, every call before that step's core step. A revert of the
executor's call reads 0. (The signal-level path — the real handlers, the save, exit 15 — is
`main/train/deferred_abort_test`'s `test_a_signal_during_an_eval_cycle_is_honoured_at_the_cycles_safe_point`;
the end to end is the `--debug-eval` smoke SIGTERMed during a cycle.)"""
from __future__ import annotations

from typing import List

import pytest

from agents.training.rust_eval import parity as PAR

pytestmark = [pytest.mark.sim, pytest.mark.integration]


def test_the_eval_core_calls_the_safe_point_at_every_host_step(built, tmp_path):
    from agents.training.eval_callback import eval_opponent_names

    trainee, sentinels, mcfg = PAR.build_models(tmp_path / "models", n_sentinels=0)
    md = tmp_path / "run"
    md.mkdir()
    md.joinpath("model_config.json").write_text(open(mcfg).read())
    calls: List[str] = []
    out = PAR.run_rust(run_dir=md / ".eval_runs" / "step_1000", model_dir=md, trainee=trainee, sentinels=sentinels,
                       items=PAR._items(list(eval_opponent_names())[:1], [], 2), shard_games=2, step=1000,
                       cycle_seed=20261003, quota={"win": 0, "loss": 0, "draw": 0}, device="cpu",
                       backend="eager", n_envs=2, safe_point=calls.append)
    host_steps = int(out["stats"]["host_steps"])
    assert out["stats"]["games"] == 2 and host_steps >= 2, out["stats"]
    assert len(calls) == host_steps, (len(calls), host_steps)
    assert set(calls) == {"eval cycle (step 1,000)"}, set(calls)
