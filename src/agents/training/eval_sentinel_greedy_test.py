"""The ``--eval-sentinel-greedy`` regime on the Rust eval executor (``RustEvalCore``): under the default a pool
sentinel plays GREEDY (the argmax of its served log-probs) and draws **the trainee's own teams**; under
``--no-eval-sentinel-greedy`` it SAMPLES (the keyed draw, keyed by the game) and draws the flat pool's.

🚨 THE TEAM HALF IS AS LOAD-BEARING AS THE GREEDY HALF (gen3_eval_sentinel_greedy_default_v1,
2026-09-07). The trainee draws from ``Gen3Teambuilder(all_teams, bias_teams=sample_teams,
bias_prob=0.1)`` while the sentinel used to draw from the flat ``Gen3Teambuilder(all_teams)`` — a
10% tilt toward the curated sample teams that the trainee got and its opponent did not. Together
with the temperature handicap that made an eval sentinel edge a DIFFERENT experiment from the
dense ladder's edge for the same frozen pair, worth +8.9 pp [+7.0, +10.7] to the trainee. One
switch moves both, because the pair of them is exactly the condition the ladder reuses a pair
under.

(These guards were the Python eval worker's ``_play_unit``'s until it was deleted — poke-env retirement P6 slice
6c; the executor is the one eval path.)"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agents.training.eval_sharding import BOT, FIXED, SENTINEL
from agents.training.rust_eval.executor import RustEvalCore


def _core():
    core = RustEvalCore.__new__(RustEvalCore)
    core.trainee_builder, core.opp_builder = object(), object()
    core.fixed_builders = {"ext_a": object()}
    return core


def _item(kind, key="sentinel_0"):
    return SimpleNamespace(kind=kind, key=key)


def test_a_greedy_sentinel_draws_the_TRAINEES_teams():
    """The symmetry half: greedy ⇒ the sentinel gets the trainee's own (sample-biased) builder, so the eval edge
    and the dense ladder's edge for that frozen pair are the SAME experiment."""
    core = _core()
    assert core.opponent_builder(_item(SENTINEL), sentinel_greedy=True) is core.trainee_builder


def test_a_sampled_sentinel_keeps_the_flat_pool_builder():
    """``--no-eval-sentinel-greedy`` is the pre-2026-09-07 behaviour on the team axis."""
    core = _core()
    assert core.opponent_builder(_item(SENTINEL), sentinel_greedy=False) is core.opp_builder


def test_the_switch_moves_only_the_sentinels_teams():
    """A bot draws from the flat pool and a fixed opponent from its pinned teams under EITHER regime."""
    core = _core()
    for greedy in (True, False):
        assert core.opponent_builder(_item(BOT, "heuristic"), greedy) is core.opp_builder
        assert core.opponent_builder(_item(FIXED, "ext_a"), greedy) is core.fixed_builders["ext_a"]


@pytest.fixture(scope="module")
def sentinel_games(tmp_path_factory):
    """One tiny real cycle per regime — 6 games against one sentinel, the same seed — on the emission self-check
    build (a MODULE fixture: the tier budget is per test call)."""
    from agents.training.rust_eval import offline as OFF
    from agents.training.rust_rollout.testkit import build_selfcheck

    build_selfcheck()
    root = tmp_path_factory.mktemp("sentinel_regime")
    trainee, sentinels, cfg = OFF.build_models(root / "models", n_sentinels=1)
    out = {}
    for greedy in (True, False):
        md = root / f"run_{greedy}"
        md.mkdir()
        md.joinpath("model_config.json").write_text(open(cfg).read())
        out[greedy] = OFF.run_rust(
            run_dir=md / ".eval_runs" / "step_2000", model_dir=md, trainee=trainee, sentinels=sentinels,
            items=OFF.plan_items([], sentinels, 6), shard_games=6, step=2000, cycle_seed=20261008,
            quota={"win": 0, "loss": 0, "draw": 0}, device="cpu", backend="eager", n_envs=2,
            sentinel_greedy=greedy, self_play_temp=1.0)["games"]
    return out



@pytest.mark.sim
@pytest.mark.integration
def test_a_greedy_sentinel_plays_the_argmax_and_a_sampled_one_does_not(sentinel_games):
    """The play half, on real games: every greedy sentinel decision is its argmax (``opp`` rows ``[dec, action,
    argmax, margin, turn]``); the sampled regime draws off the argmax somewhere in the same plan and seed."""
    greedy = [r for g in sentinel_games[True] for r in g["opp"]]
    sampled = [r for g in sentinel_games[False] for r in g["opp"]]
    assert greedy and sampled, "the sentinel made no decision"
    assert all(int(r[1]) == int(r[2]) for r in greedy)
    assert sum(int(r[1]) != int(r[2]) for r in sampled) > 0
