"""T17 MIRRORED TEAM PAIRS on an external-anchor read (`main.anchors --mirrored-pairs`).

The front end pairs the SEEDS, the planner pairs the TEAMS, and the scorer refuses any pair whose two
battles did not in fact share both — each pinned here, each failing if its code is reverted.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from main.anchors import mirrored as M
from utils.bridge.ws_frontend import ShowdownFrontEnd, canonical_team, team_pair_key

_A = "Nick|Tyranitar|Leftovers|Sand Stream|rockslide,earthquake||252,,,,,|M|||||]Skarmory||Leftovers|Keen Eye|spikes,whirlwind||||||"
_A2 = "Tyranitar||leftovers|sandstream|earthquake,rockslide||||||]Skarmory||leftovers|keeneye|whirlwind,spikes||||||"
_B = "Zapdos||Leftovers|Pressure|thunderbolt,hiddenpowerice||||||]Celebi||Leftovers|Natural Cure|psychic,recover||||||"
_C = "Starmie||Leftovers|Natural Cure|surf,recover||||||]Jirachi||Leftovers|Serene Grace|psychic,wish||||||"


def test_one_team_packed_two_ways_is_one_team_and_the_pair_key_is_unordered():
    assert canonical_team(_A) == canonical_team(_A2)          # nickname / id spelling / move order
    assert team_pair_key(_A, _B) == team_pair_key(_B, _A2)
    assert team_pair_key(_A, _B) != team_pair_key(_A, _C)


def test_the_front_end_gives_a_mirrored_pair_ONE_seed_and_the_next_pair_a_new_one(tmp_path):
    log = tmp_path / "pair.jsonl"
    fe = ShowdownFrontEnd(seed_base=7, pair_seeds=True, pair_log=str(log), validate_teams=False)
    s_ab = fe._pair_seed("t1", _A, _B)
    s_c = fe._pair_seed("t2", _A, _C)                          # another battle runs in between
    s_ba = fe._pair_seed("t3", _B, _A2)                        # the mirror: teams handed over
    s_ab2 = fe._pair_seed("t4", _A, _B)                        # the SAME pairing drawn again later
    assert s_ab == s_ba and s_c != s_ab and s_ab2 != s_ab
    assert M.read_pair_log(log) == {"t1": s_ab, "t2": s_c, "t3": s_ba, "t4": s_ab2}
    with pytest.raises(ValueError):
        ShowdownFrontEnd(pair_seeds=True, validate_teams=False)  # a pair needs a seed base


def _plan(tmp_path, n=4):
    ours = [("o1", _A), ("o2", _C)]
    theirs = [("t1", _B)]
    return M.plan_half("ours_challenge", n, ours, theirs, seed=3, out_dir=tmp_path, validate=False)


def test_the_plan_hands_the_teams_over_in_the_second_battle(tmp_path):
    hm = _plan(tmp_path)
    assert hm.n_games == 4
    for k in range(2):
        (a_lab, a_txt), (b_lab, b_txt) = hm.ours[2 * k], hm.ours[2 * k + 1]
        assert hm.theirs[2 * k].name == b_lab and hm.theirs[2 * k + 1].name == a_lab
        assert hm.theirs[2 * k].read_text().strip() == b_txt.strip()
        assert hm.theirs[2 * k + 1].read_text().strip() == a_txt.strip()
    assert json.loads(hm.sequence_path.read_text()) == [str(p) for p in hm.theirs]
    assert [x[0] for x in _plan(tmp_path / "again").ours] == [x[0] for x in hm.ours]   # deterministic
    with pytest.raises(M.MirrorError):
        _plan(tmp_path / "odd", n=3)


def _rec(tag, result, finished=True, cap=False):
    return SimpleNamespace(battle_tag=tag, result=result, finished=finished, hit_forfeit_limit=cap)


def test_a_pair_counts_only_when_teams_order_and_seed_all_check_out(tmp_path):
    hm = _plan(tmp_path, n=8)
    ours = [lab for lab, _ in hm.ours]
    theirs = [p.name for p in hm.theirs]
    recs = [_rec(f"b{i}", r) for i, r in enumerate(["win", "loss", "win", "win", "loss", "tie", "win", "loss"])]
    seeds = {f"b{i}": [i // 2, 0, 0, 1] for i in range(8)}
    seeds["b7"] = [99, 0, 0, 1]                                 # pair 3's second battle got other dice
    out = M.score_half(hm, recs, ours, theirs, seeds)
    assert out["pair_counts"] == [0, 1, 1, 0, 1]                # (W,L)=2, (W,W)=4, (L,T)=1
    assert out["voided"] == {"seed_not_shared": 1}
    bad = list(theirs)
    bad[0], bad[1] = bad[1], bad[0]                             # the peer played pair 0 unswapped
    assert M.score_half(hm, recs, ours, bad, seeds)["voided"]["peer_team_off_plan"] == 1


def test_a_turn_limit_forfeit_is_a_DRAW_as_in_training():
    assert M.game_points(_rec("x", "loss", cap=True)) == 1
    assert M.game_points(_rec("x", "win")) == 2 and M.game_points(_rec("x", "loss", finished=False)) is None


def test_our_side_plays_the_sequence_in_order_and_logs_each_label():
    from main.anchors.session import build_team_source

    from utils.team_loader import TeamLoader

    t_a, t_b = TeamLoader().get_all_teams()[:2]                 # real Showdown exports
    log: list = []
    tb = build_team_source({"kind": "sequence", "items": [("p0_ours", t_a), ("p0_theirs", t_b)]}, 1, log)
    t0, t1 = tb.yield_team(), tb.yield_team()
    assert log == ["p0_ours", "p0_theirs"] and t0 != t1


def test_the_cli_refuses_a_mirrored_read_it_cannot_honour():
    from main.anchors.cli import build_parser, build_plan
    from main.anchors.config import load_config

    cfg = load_config()
    for argv in (["--opponent", "foulplay"], ["--games", "10"], ["--server", "node"]):
        with pytest.raises(SystemExit, match="--mirrored-pairs needs"):
            build_plan(build_parser().parse_args(["--model", "", "--mirrored-pairs", "--dry-run", *argv]), cfg)
    plan = build_plan(build_parser().parse_args(["--model", "", "--mirrored-pairs", "--dry-run",
                                                 "--games", "8"]), cfg)
    assert plan.mirrored_pairs and plan.seed_base is not None and plan.pair_log is not None


def test_every_row_and_the_summary_carry_the_regime():
    from main.anchors.results import REQUIRED_ROW_FIELDS, render

    assert "mirrored_pairs" in REQUIRED_ROW_FIELDS
    summary = {"cell": {"opponent": "m", "opponent_version": "", "opponent_commit": "", "our_regime": "greedy",
                        "their_regime": "greedy", "regime_matched": True, "teamset": "home",
                        "our_team_count": 1, "their_team_count": 1, "model_zip": "", "mirrored_pairs": True},
               "n": 4, "wilson95": [0.1, 0.9], "status": "OK", "win_rate": 0.5, "wins": 2, "losses": 2, "ties": 0,
               "mirrored_pairs": {"score": 0.5, "score_ci95": [0.5, 0.5], "n_pairs": 2, "n_voided": 0}}
    text = render(summary)
    assert "PAIR SCORE  0.500" in text and "NOT the unit" in text


def test_the_peers_copy_of_our_team_spells_out_the_hidden_power_ivs():
    """The first real mirrored read DIED here: the peer re-parsed our export without our builder's
    gen-3 HP IV derivation and the front end rejected the team. Every file we hand it now states them."""
    from utils.bridge.team_validator import validate_teams_locally
    from utils.team_loader import TeamLoader

    team = next(t for t in TeamLoader().get_all_teams() if "Hidden Power" in t and "IVs:" not in t)
    out = M.with_hp_ivs(team)
    assert "IVs:" in out and validate_teams_locally("gen3ou", [out])[0].get("valid")
    assert M.with_hp_ivs(out) == out                            # idempotent: stated IVs are left alone
