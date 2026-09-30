"""Eval-worker unit tests — the trainee-teambuilder selection (the specialist eval-alignment fix).

The worker used to HARDCODE the default full-pool trainee teambuilder, so a `--trainee-team` run's
eval (win rates / ELO / vs-ext verdicts) measured the model piloting RANDOM teams it never trained
on — the ai_v7_05–08 "plateau" was this out-of-distribution measurement, not the training. These
pin the fix: `cfg['trainee_team_str']` → the pinned single-team builder; absent → the default,
byte-identical to the old behavior."""

import pytest

from main.eval_worker import _build_trainee_tb, _fixed_opponent_tb
from utils.team_loader import TeamLoader


@pytest.fixture(scope="module")
def teams():
    loader = TeamLoader()
    return loader.get_all_teams(), loader.get_sample_teams()


@pytest.fixture(scope="module")
def tss_str():
    with open("data/teams/specialist/tss_starmie.txt", encoding="utf-8") as f:
        return f.read()


def test_pinned_team_builds_single_team_builder(teams, tss_str):
    all_teams, sample_teams = teams
    tb = _build_trainee_tb({"trainee_team_str": tss_str}, all_teams, sample_teams)
    assert len(tb.packed_teams) == 1              # eval pilots EXACTLY the trained team
    assert not tb.bias_packed_teams
    # every draw is that team
    assert len({tb.yield_team() for _ in range(5)}) == 1


def test_absent_key_is_the_default_pool(teams):
    all_teams, sample_teams = teams
    tb = _build_trainee_tb({}, all_teams, sample_teams)
    assert len(tb.packed_teams) == len(all_teams)  # the full pool (old behavior, byte-identical)
    assert tb.bias_prob == pytest.approx(0.1)


def test_none_value_is_the_default_pool(teams):
    """The callbacks always send the key (None when no pin) — None must mean default, not crash."""
    all_teams, sample_teams = teams
    tb = _build_trainee_tb({"trainee_team_str": None}, all_teams, sample_teams)
    assert len(tb.packed_teams) == len(all_teams)


# ── fold-back: a FIXED opponent's own pinned team ─────────────────────────────

class _Item:
    def __init__(self, team_str=None):
        self.team_str = team_str


def test_pinned_fixed_opponent_pilots_its_own_team(tss_str):
    tb = _fixed_opponent_tb(_Item(team_str=tss_str), opp_tb="POOL")
    assert len(tb.packed_teams) == 1
    assert len({tb.yield_team() for _ in range(5)}) == 1


def test_unpinned_fixed_opponent_keeps_the_pool_builder():
    pool = object()
    assert _fixed_opponent_tb(_Item(team_str=None), opp_tb=pool) is pool
    assert _fixed_opponent_tb(object(), opp_tb=pool) is pool   # pre-team_str item (no attr)


def test_a_multi_team_fixed_opponent_from_the_callbacks_samples_all_its_teams(teams):
    """Both eval callbacks build a FIXED item with ``EvalItem.fixed_from_cfg`` from the entry's
    ``to_cfg()`` — which carries ``team_strs`` — so a multi-team specialist is measured among ALL its
    teams (as the Rust eval core pins it), never its first alone (the pre-2026-09-30 callbacks)."""
    from agents.training.eval_sharding import EvalItem
    from utils.paths import src_path

    _, sample = teams
    cfg = {"label": "ext_multi", "path": "/m/x.zip", "config_path": "/m/c.json", "team_str": sample[0],
           "team_strs": [sample[0], sample[1], sample[2]]}
    item = EvalItem.from_dict(EvalItem.fixed_from_cfg(cfg, 4).to_dict())   # through plan.json
    assert item.team_strs == cfg["team_strs"] and item.team_str == sample[0]
    assert len(_fixed_opponent_tb(item, opp_tb="POOL").packed_teams) == 3
    for mod in ("agents/training/eval_callback.py", "agents/training/selfplay_callback.py"):
        text = src_path(*mod.split("/")).read_text()
        assert "EvalItem.fixed_from_cfg(f, n_games)" in text and "FIXED, n_games" not in text, mod


def test_the_keyed_sampler_draws_the_rust_eval_cores_key():
    """``install_opponent_log(keyed=True)`` (per_game + the sampled sentinel regime): decision d of a game
    draws the keyed draw at (the game's sample seed, the opponent stream, env 0, episode 0, d) — the
    executor's key — and a greedy opponent's decision is logged with its top-2 margin."""
    from types import SimpleNamespace

    import numpy as np
    import torch

    from agents.training import keyed_draw as KD
    from main.eval_worker import install_opponent_log

    logits = torch.tensor([[0.3, -1.0, 0.9, 0.1, 0.0, 0.2, -0.4, 0.5, 0.05, -0.2, 0.7]])
    mask = torch.tensor([[1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1]], dtype=torch.float32)
    ml = logits + (mask - 1.0) * 1e9

    def make(keyed):
        pl = SimpleNamespace()

        def predict(battle, stochastic=False, need_aux=True, temperature=1.0):
            pl._last_masked_logits = ml
            idx = pl._action_sampler(ml, temperature) if keyed else int(torch.argmax(ml).item())
            return idx, None, None

        pl._predict_best_action = predict
        pl.choose_move = lambda battle: pl._predict_best_action(battle, stochastic=keyed, temperature=0.8)[0]
        install_opponent_log(pl, keyed=keyed)
        return pl

    pl = make(True)
    with pytest.raises(RuntimeError, match="no game key"):
        pl.choose_move(None)
    pl._keyed_seed, pl._keyed_log = 123456789, []
    got = [pl.choose_move(None) for _ in range(40)]
    lp = torch.log_softmax(logits.masked_fill(mask == 0, float("-inf")), -1).numpy().astype(np.float32)
    want, _u, margins = KD.keyed_sample(np.repeat(lp, 40, 0), seed=123456789, stream=KD.STREAM_OPPONENT, env=0,
                                        episode=0, decision=np.arange(40), temperature=0.8)
    assert got == [int(a) for a in want] and len(set(got)) > 1
    assert [r[0] for r in pl._keyed_log] == list(range(40)) and all(mask[0, a] == 1 for a in got)
    assert all(r[1] == a and r[2] == 10 for r, a in zip(pl._keyed_log, got))
    assert all(abs(r[3] - float(m)) < 1e-12 for r, m in zip(pl._keyed_log, margins))
    g = make(False)
    g._keyed_seed = 1
    assert [g.choose_move(None) for _ in range(3)] == [10, 10, 10]
    assert [r[:3] for r in g._keyed_log] == [[0, 10, 10], [1, 10, 10], [2, 10, 10]]
    assert abs(g._keyed_log[0][3] - float(lp[0, 10] - lp[0, 7])) < 1e-6    # top-2 legal: 0.7 vs 0.5
