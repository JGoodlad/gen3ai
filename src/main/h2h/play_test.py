"""``main.h2h.play`` — the parts that need NO engine: the seed rule, the batch plan, the scoring of a batch into
counts, the resume rule, the team-source check and the row assembly (against the ledger's own schema).

THE MIRROR, WITHOUT AN ENGINE. ``score_games`` is where a pairing bug would live: games ``2k`` / ``2k+1`` must
be one team pairing handed over, scored as one pair. The toy engine below is SEAT-SYMMETRIC by construction
(the winner is the team with the larger strength, a tie goes to the lower id), so a player against itself
scores EXACTLY 1/2 in every pair — if the pairing, the hand-over or the scoring were wrong, it would not.
(The REAL engine is not exactly seat-symmetric — speed ties and RNG order — which is why the real-engine
claim is in ``play_integration_test.py`` and is stated as what it is.)"""
from __future__ import annotations

import json
import types

import pytest

from agents.training import eval_ledger as L
from agents.training import mirrored_pairs as MP
from agents.training.trace_result import DRAW, LOSS, WIN
from main.h2h import play as PL
from utils.rust_env import episode as EP

TL = EP.stall_threshold()

TEAMS = [f"packed team {i}" for i in range(12)]
STRENGTH = {i: (i * 7) % 12 for i in range(12)}          # distinct: no ties


def toy_games(n_pairs, player_skill=0, opp_skill=0, near=None):
    """The executor's rows for ``n_pairs`` mirrored pairs on a seat-symmetric toy engine: the PLAYER wins iff
    its (team strength + skill) beats the opponent's. Pair ``k`` is teams ``(a, b)`` then ``(b, a)``."""
    rows = []
    for k in range(n_pairs):
        a, b = (3 * k) % 12, (5 * k + 1) % 12
        if a == b:
            b = (b + 1) % 12
        for g, (tp, to) in ((2 * k, (a, b)), (2 * k + 1, (b, a))):
            sp, so = STRENGTH[tp] + player_skill, STRENGTH[to] + opp_skill
            res = WIN if sp > so else LOSS if sp < so else DRAW
            rows.append({"game": g, "result": res, "swapped": g % 2 == 1, "teams": [tp, to], "near_ties": 0,
                         "near_ties_wide": 0, **(near or {}).get(g, {})})
    return rows


def test_a_player_against_itself_cancels_exactly_in_every_pair():
    sc = PL.score_games(toy_games(25), TEAMS, 25)
    assert sc.pair_counts == [0, 0, 25, 0, 0] and (sc.w, sc.l, sc.d) == (25, 25, 0)
    assert MP.pair_score(sc.pair_counts) == 0.5
    assert sc.clean_pairs() == 25 and sc.clean_pairs_off_center() == 0


def test_a_stronger_player_wins_both_games_of_a_pair_and_the_pair_is_the_unit():
    sc = PL.score_games(toy_games(25, player_skill=20), TEAMS, 25)
    assert sc.pair_counts == [0, 0, 0, 0, 25] and (sc.w, sc.l) == (50, 0)
    assert MP.pair_score(sc.pair_counts) == 1.0


def test_team_counters_balance_and_name_teams_by_content():
    sc = PL.score_games(toy_games(25), TEAMS, 25)
    games = 2 * 25
    assert sum(v["p"][0] for v in sc.teams.values()) == games == sum(v["o"][0] for v in sc.teams.values())
    assert sum(v["p"][1] for v in sc.teams.values()) == sc.w and sum(v["o"][1] for v in sc.teams.values()) == sc.l
    assert set(sc.teams) <= {L.team_id(t) for t in TEAMS}


def test_a_draw_is_half_a_point_and_is_not_a_win():
    g = toy_games(1)
    g[0]["result"], g[1]["result"] = DRAW, DRAW
    sc = PL.score_games(g, TEAMS, 1)
    assert (sc.w, sc.l, sc.d) == (0, 0, 2) and sc.pair_counts == [0, 0, 1, 0, 0]


@pytest.mark.parametrize("name,mutate,match", [
    ("a missing game", lambda g: g.pop(), "off the plan"),
    ("a repeated game", lambda g: g.append(dict(g[0])), "twice"),
    ("both games unswapped", lambda g: g[1].update(swapped=False), "not .unswapped, swapped"),
    ("teams not handed over", lambda g: g[1].update(teams=list(g[0]["teams"])), "handed over"),
    ("an unknown result", lambda g: g[0].update(result="FORFEIT"), "unknown result"),
])
def test_a_pair_whose_games_do_not_satisfy_the_mirror_is_refused_not_scored(name, mutate, match):
    games = toy_games(3)
    mutate(games)
    with pytest.raises(PL.H2HError, match=match):
        PL.score_games(games, TEAMS, 3)


def test_clean_pairs_exclude_near_tie_decisions_and_count_the_off_center_ones():
    games = toy_games(25)
    # break pair 3 off-centre (the player wins both) and put a near-tie decision in pair 7's second game
    games[6]["result"], games[7]["result"] = WIN, WIN
    games[15]["near_ties"] = 1
    games[15]["near_ties_wide"] = 1
    sc = PL.score_games(games, TEAMS, 25)
    assert sc.pair_counts[4] == 1 and sc.near_tie_decisions == 1 and sc.near_tie_games == 1
    assert sc.clean_pairs() == 24 and sc.clean_pairs_off_center() == 1
    # the off-centre pair is excluded from the exactness claim when it carries a near-tie decision
    games[6]["near_ties"] = 1
    assert PL.score_games(games, TEAMS, 25).clean_pairs_off_center() == 0


def test_the_game_sink_keeps_what_scoring_needs_and_counts_near_ties_on_both_sides():
    s = PL._GameSink()
    s.append({"game": 4, "result": WIN, "swapped": False, "teams": [1, 2], "winner": 1, "end_turn": 30,
              "forfeit": 0, "script": "x" * 100000, "actions": [1, 2], "logp": [-1.0],
              "margins": [1.0, 1e-6, 5e-4, float("inf")],
              "opp": [[0, 3, 3, 1e-7, 1], [1, 3, 3, 0.5, 2]]})
    r = s[0]
    assert "script" not in r and "actions" not in r and r["game"] == 4
    assert r["near_ties"] == 2 and r["near_ties_wide"] == 3        # 1e-6, 1e-7  |  + 5e-4


def test_schedule_key_is_order_independent_and_cycle_seeds_are_disjoint_per_batch_and_namespace():
    a = PL.PlayerRef("a", "/a.zip", "/c", "/r", "ra", "explicit_zip", 1, "a" * 64)
    b = PL.PlayerRef("b", "/b.zip", "/c", "/r", "rb", "explicit_zip", 2, "b" * 64)
    assert PL.schedule_key_of(a, b) == PL.schedule_key_of(b, a)
    assert PL.schedule_key_of(a, a) != PL.schedule_key_of(a, b)
    key = PL.schedule_key_of(a, b)
    seeds = {PL.cycle_seed(0, key, k) for k in range(50)} | {PL.cycle_seed(1, key, 0)}
    assert len(seeds) == 51
    from agents.training.rust_eval.launch import cycle_seed as in_loop
    from agents.training.sprt_promotion import sprt_seed

    assert PL.cycle_seed(0, key, 0) not in {in_loop(0, 0), sprt_seed(0, 0, 0)}
    assert PL.cycle_seed(5, key, 3) == PL.cycle_seed(5, key, 3)


def test_batch_plan_splits_pairs_into_whole_batches_with_a_remainder():
    assert PL.batch_plan(1000, 500) == [500, 500]
    assert PL.batch_plan(1100, 500) == [500, 500, 100]
    assert PL.batch_plan(7, 500) == [7]
    with pytest.raises(PL.H2HError):
        PL.batch_plan(0, 500)


def test_team_source_check_refuses_a_pinned_team_run(tmp_path):
    run = tmp_path / "run_x"
    (run / "checkpoints").mkdir(parents=True)
    zip_path = run / "checkpoints" / "c.zip"
    zip_path.write_bytes(b"z")
    ref = PL.PlayerRef("s", str(zip_path), str(run / "model_config.json"), str(run), "run_x", "explicit_zip", 1, "c" * 64)
    assert "unverified" in PL.check_team_source(ref)                       # no metadata.json: said, not assumed
    spec = {"eval_trainee_teams": {"kind": "default_biased", "bias_prob": 0.1, "pin_file": None}}
    (run / "metadata.json").write_text(json.dumps({"matchup_history": [{"hash": "h1", "spec": spec}]}))
    assert "default_biased" in PL.check_team_source(ref)
    spec["eval_trainee_teams"] = {"kind": "pinned", "pin_file": "t.txt", "bias_prob": 0.0}
    (run / "metadata.json").write_text(json.dumps({"matchup_history": [{"hash": "h2", "spec": spec}]}))
    with pytest.raises(PL.H2HError, match="pinned-team"):
        PL.check_team_source(ref)


def _row(writer, a, b, batch, n, key="h2h:k", seed=0, regime=None):
    sc = PL.score_games(toy_games(n), TEAMS, n)

    eng = types.SimpleNamespace(compute=PL.Compute(device="cpu"),    # only what build_row reads
                                regime=regime or PL.regime_for(TL), torch_version="t",
                                core_stamp="s", team_check={"a": "x"}, historical={"a": []})
    return PL.build_row(writer=writer, run_label="study", commit="c0ffee", a=a, b=b, eng=eng, purpose="audit",
                        key=key, sched_seed=seed, batch=batch, cseed=PL.cycle_seed(seed, key, batch), score=sc,
                        t_start=L.utc_now(), t_end=L.utc_now(), wall_s=2.0,
                        load={"start": [0, 0, 0], "end": [0, 0, 0], "contention": 1.0}, executor={})


def _players(tmp_path):
    cfg = tmp_path / "model_config.json"
    cfg.write_text(json.dumps({"progress_decision_tense": False, "progress_switch_freeze": False}))
    mk = lambda c: PL.PlayerRef(c, f"/{c}.zip", str(cfg), "/r", f"run_{c}", "explicit_zip", 10, c * 64)  # noqa: E731
    return mk("a"), mk("b")


def test_a_built_row_validates_and_resume_finds_its_batches(tmp_path):
    a, b = _players(tmp_path)
    w = L.LedgerWriter(tmp_path, writer_id="t1")
    row = _row(w, a, b, 0, 10)
    assert L.validate_row(row) == []
    w.append(row)
    w.append(_row(w, a, b, 1, 10))
    rid = PL.regime_for(TL)["regime_id"]
    rows = L.read_rows(tmp_path)
    assert PL.completed_batches(rows, a, b, rid, "h2h:k", 0) == {0: 10, 1: 10}
    assert PL.completed_batches(rows, b, a, rid, "h2h:k", 0) == {}, "the reversed direction is another edge"
    assert PL.completed_batches(rows, a, b, rid, "h2h:other", 0) == {}, "another schedule is not the same games"
    assert PL.completed_batches(rows, a, b, rid, "h2h:k", 1) == {}, "another schedule seed neither"
    other = PL.regime_for(TL + 1)["regime_id"]
    assert PL.completed_batches(rows, a, b, other, "h2h:k", 0) == {}, "nor another regime"


def test_a_resume_with_another_batch_size_over_existing_rows_is_refused(tmp_path, monkeypatch):
    a, b = _players(tmp_path)
    w = L.LedgerWriter(tmp_path, writer_id="t1")
    w.append(_row(w, a, b, 0, 10))
    with pytest.raises(PL.H2HError, match="different batch size"):
        PL.play_edge(str(tmp_path), a, b, pairs=40, batch_pairs=20, schedule_seed=0, schedule_key="h2h:k",
                     run_label="study", compute=PL.Compute())


def test_play_edge_refuses_models_and_a_purpose_outside_the_closed_list(tmp_path, monkeypatch):
    a, b = _players(tmp_path)
    archive = tmp_path / "archive"
    archive.mkdir()
    monkeypatch.setenv("GEN3AI_MODELS_DIR", str(archive))
    with pytest.raises(L.LedgerPathError):
        PL.play_edge(str(archive / "out"), a, b, pairs=2, run_label="s", compute=PL.Compute())
    with pytest.raises(PL.H2HError, match="purpose"):
        PL.play_edge(str(tmp_path / "out"), a, b, pairs=2, run_label="s", compute=PL.Compute(), purpose="study")


def test_a_player_that_recorded_a_deleted_core_variant_on_is_refused(tmp_path):
    cfg = tmp_path / "model_config.json"
    ref = PL.PlayerRef("s", "/s.zip", str(cfg), "/r", "run_x", "explicit_zip", 1, "c" * 64)
    cfg.write_text(json.dumps({"progress_decision_tense": False, "progress_switch_freeze": False}))
    PL.check_core_flags(ref)
    cfg.write_text(json.dumps({"progress_decision_tense": False}))                  # absent = off
    PL.check_core_flags(ref)
    for k in ("progress_decision_tense", "progress_switch_freeze"):
        cfg.write_text(json.dumps({k: True}))
        with pytest.raises(PL.H2HError, match="P11d"):
            PL.check_core_flags(ref)
