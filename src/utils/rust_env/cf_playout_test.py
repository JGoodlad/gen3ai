"""Pure unit tests for the Rust-core counterfactual (``utils.rust_env.counterfactual``,
``gen3_cf_core_playout_v1``): the side policies' arithmetic (``RLPlayer._predict_best_action``'s), the
mapping of a play-out's branches onto rollouts, the request the play-out is handed, and the narrated
play-by-play. No core, no model: the play-out and the policy forward are faked. The core's own semantics
(the divergence root, the in-core bot, the stall order, the text) are pinned by
``src/rust_env/tests/search_playout_cf_test.rs``; the old-vs-new identity by
``designs/research_state/measurements/pokeenv_p6_replay_cf_2026-10-07/``."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from utils.rust_env import counterfactual as CF
from utils.rust_env import successors as S
from utils.rust_env.counterfactual import ModelPolicy, summarize_trajectory


def _model(logits):
    """A fake SB3 policy whose distribution's logits are ``logits`` (per call, one row)."""
    def get_distribution(obs):
        assert obs["observation"].shape[0] == 1 and obs["action_mask"].shape == (1, 11)
        return SimpleNamespace(distribution=SimpleNamespace(logits=torch.tensor([logits], dtype=torch.float32)))

    return SimpleNamespace(device="cpu", policy=SimpleNamespace(get_distribution=get_distribution))


_ROW = np.zeros(4, np.float32)


def test_greedy_is_the_masked_argmax():
    logits = [0.0] * 11
    logits[3], logits[7] = 5.0, 2.0
    mask = np.zeros(11, np.uint8)
    mask[[1, 7]] = 1
    assert ModelPolicy(_model(logits)).decide(_ROW, mask, 0) == 7     # 3 is higher but masked
    mask[3] = 1
    assert ModelPolicy(_model(logits)).decide(_ROW, mask, 0) == 3


def test_a_stochastic_policy_needs_seeds_and_draws_per_rollout_like_the_rlplayer():
    with pytest.raises(ValueError, match="seeds"):
        ModelPolicy(_model([0.0] * 11), stochastic=True)
    logits = [0.3, -0.2, 0.9, 0.1, 0.0, 0.5, 0.0, 0.2, -1.0, 0.4, 0.0]
    mask = np.ones(11, np.uint8)
    mask[4] = 0
    pol = ModelPolicy(_model(logits), stochastic=True, seeds=[11, 22])
    seq0 = [pol.decide(_ROW, mask, 0) for _ in range(6)]
    # the RLPlayer's seeded draw: torch.multinomial over the masked categorical's probs, its own generator
    gen = torch.Generator(device="cpu")
    gen.manual_seed(11)
    masked = torch.tensor([logits]) + (torch.from_numpy(mask[None].astype(np.int8)) - 1.0) * 1e9
    probs = torch.distributions.Categorical(logits=masked).probs
    want = [int(torch.multinomial(probs, 1, True, generator=gen).item()) for _ in range(6)]
    assert seq0 == want
    assert 4 not in seq0
    # a rollout's draws are a function of ITS seed alone, whatever the interleaving with other rollouts
    pol2 = ModelPolicy(_model(logits), stochastic=True, seeds=[11, 22])
    inter = []
    for _ in range(6):
        inter.append(pol2.decide(_ROW, mask, 0))
        pol2.decide(_ROW, mask, 1)
    assert inter == seq0


def _record(side="p1"):
    return SimpleNamespace(trainee_username="me", side_of=lambda name: side)


def _result(branches, **kw):
    return S.PlayoutResult(side="p1", at=7, branches=branches, root={"tokens": {"0": "move 1"}, "other_recorded": "move 2"},
                           prefix_text=["|turn|1", "|move|p1a: A|Tackle|p2a: B"], **kw)


def _branch(winner, *, forfeit=False, truncated=False, turn=30, text=None):
    return {"action": 0, "seed": 0, "reseed": None, "decisions": [3, 4], "cmds": ["CHOOSE p1 move 1"],
            "end": {"winner": winner, "forfeit": forfeit, "truncated": truncated, "turn": turn},
            "text": text or ["|turn|2", "|win|x"]}


def test_the_play_out_request_and_the_rollout_mapping(monkeypatch):
    seen = {}

    def fake_play_out(log, at, side, **kw):
        seen.update(log=log, at=at, side=side, **kw)
        return _result([_branch(1), _branch(0, forfeit=True, turn=250), _branch(None), _branch(None, truncated=True)])

    monkeypatch.setattr(S, "play_out", fake_play_out)
    monkeypatch.setattr(S, "record_to_log", lambda rec: {"log": "L"})
    out = CF.replay_counterfactual(_record("p2"), divergence_turn=9, substitute_action=0,
                                   our_policy=ModelPolicy(_model([0.0] * 11)), opp_bot={"name": "staller", "seed": 5},
                                   post_t_seeds=[None, "1,2,3,4", "5,6,7,8", "9,9,9,9"], stall_sides=["p2"], text=True)
    assert seen["at"] == {"turn": 9, "other": "recorded"} and seen["side"] == "p2" and seen["actions"] == [0]
    assert seen["bot"] == {"side": "p1", "name": "staller", "seed": 5}
    assert seen["text"] == "p2" and seen["seeds"] == [None, "1,2,3,4", "5,6,7,8", "9,9,9,9"]
    assert seen["stall"]["side"] == "p2" and seen["stall"]["turn_limit"] == 250
    assert [r.outcome for r in out["rollouts"]] == ["win", "loss", "tie", "unfinished"]   # we are p2 = winner 1
    assert [r.capped for r in out["rollouts"]] == [False, True, False, False]
    assert out["rollouts"][0].text == ["|turn|1", "|move|p1a: A|Tackle|p2a: B", "|turn|2", "|win|x"]
    assert out["root"]["other_recorded"] == "move 2" and out["schema"] == CF.CF_SCHEMA


def test_exactly_one_opponent_and_both_stall_sides(monkeypatch):
    seen = {}
    monkeypatch.setattr(S, "play_out", lambda log, at, side, **kw: seen.update(kw) or _result([_branch(0)]))
    monkeypatch.setattr(S, "record_to_log", lambda rec: {})
    pol = ModelPolicy(_model([0.0] * 11))
    with pytest.raises(ValueError, match="exactly one"):
        CF.replay_counterfactual(_record(), divergence_turn=3, substitute_action=0, our_policy=pol)
    with pytest.raises(ValueError, match="exactly one"):
        CF.replay_counterfactual(_record(), divergence_turn=3, substitute_action=0, our_policy=pol, opp_policy=pol,
                                 opp_bot={"name": "random", "seed": 1})
    out = CF.replay_counterfactual(_record(), divergence_turn=3, substitute_action=0, our_policy=pol, opp_policy=pol,
                                   stall_sides=["p2", "p1"])
    assert seen["stall"]["sides"] == ["p1", "p2"] and seen["bot"] is None and seen["text"] is None
    assert out["rollouts"][0].outcome == "win" and out["rollouts"][0].text is None


def test_the_policy_dispatches_by_side_and_refuses_a_pending_bot_side(monkeypatch):
    def fake_play_out(log, at, side, *, policy, **kw):
        rows = np.zeros((3, 4), np.float32)
        masks = np.ones((3, 11), np.uint8)
        got = policy(rows, masks, np.array([0, 1, 2], np.uint32))      # branch 0 p1, branch 0 p2, branch 1 p1
        seen["got"] = list(got)
        return _result([_branch(0)])

    seen = {}
    monkeypatch.setattr(S, "play_out", fake_play_out)
    monkeypatch.setattr(S, "record_to_log", lambda rec: {})
    ours = _model([0.0, 9.0] + [0.0] * 9)
    theirs = _model([0.0, 0.0, 9.0] + [0.0] * 8)
    CF.replay_counterfactual(_record("p1"), divergence_turn=3, substitute_action=0, our_policy=ModelPolicy(ours),
                             opp_policy=ModelPolicy(theirs))
    assert seen["got"] == [1, 2, 1]
    with pytest.raises(S.SuccessorsError, match="pending"):
        CF.replay_counterfactual(_record("p1"), divergence_turn=3, substitute_action=0, our_policy=ModelPolicy(ours),
                                 opp_bot={"name": "random", "seed": 1})


def test_summarize_trajectory_parses_protocol():
    side = "p1"   # trainee is p1; opp is p2
    chunks = [
        ("p1", "|turn|5\n|switch|p1a: Gengar|Gengar, M|100/100\n"
               "|move|p2a: Swampert|Earthquake|p1a: Gengar\n|-immune|p1a: Gengar\n"),
        ("p2", "|turn|5\n|move|p2a: Swampert|Earthquake|p1a: Gengar\n"),   # opp-side chunk → must be IGNORED
        ("p1", "|turn|6\n|move|p1a: Gengar|Ice Beam|p2a: Swampert\n|-supereffective|p2a: Swampert\n"
               "|-crit|p2a: Swampert\n|-damage|p2a: Swampert|0 fnt\n|faint|p2a: Swampert\n|win|TraineeName\n"),
    ]
    turns = {t["turn"]: t["events"] for t in summarize_trajectory(side, chunks)}
    assert set(turns) == {5, 6}
    # turn 5: we switch to Gengar; opp Earthquakes; Gengar immune (no double-count from the p2 chunk).
    assert any("we sent in Gengar" in e for e in turns[5])
    assert sum("opp used Earthquake" in e for e in turns[5]) == 1   # the p2-side chunk was ignored
    assert any("immune" in e for e in turns[5])
    # turn 6: we Ice Beam → super-effective crit → Swampert faints → we win.
    assert any("we used Ice Beam" in e for e in turns[6])
    assert any("super-effective" in e for e in turns[6])
    assert any("crit" in e for e in turns[6])
    assert any("Swampert FAINTED" in e for e in turns[6])
    assert any("WINS" in e for e in turns[6])


def test_summarize_trajectory_hp_fraction_and_malformed():
    side = "p2"   # trainee is p2 this time
    chunks = [("p2", "|turn|3\n|move|p2a: Milotic|Surf|p1a: Tyranitar\n"
                     "|-damage|p1a: Tyranitar|140/300\n|garbage line no pipe\n|\n")]
    turns = {t["turn"]: t["events"] for t in summarize_trajectory(side, chunks)}
    assert any("we used Surf" in e for e in turns[3])
    assert any("47% hp" in e for e in turns[3])   # 140/300 → 47%
