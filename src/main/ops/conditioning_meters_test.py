"""``main.ops.conditioning_meters`` — the SPREAD IDENTITY, the decodes from ``V``, and every
refusal, on a synthetic trace tree with a KNOWN answer.

The meters answer one question — *does the critic's opinion move with who it is playing and whose
team it is holding?* — and the whole point of the identity is that a calibrated head reads 1.0 and
a marginal head reads 0. So the tree is planted twice from the same skeleton: once with a ``V``
that reproduces each opponent's true win rate (a SEPARATING head) and once with a ``V`` that is
the same number everywhere (a MARGINAL head). Recovering ~1 and ~0 respectively, with intervals,
is the test that the arithmetic measures what it claims.

What else is planted, because each of these has cost a wrong reading somewhere in this tree:

* the CLAMPED ratio can sit BELOW its own interval (head-refit hazard 1) — asserted, not
  worked around, and the unclamped companion is asserted to be finite and positive where the
  clamped one is exactly zero;
* the leave-one-battle-out label must not contain the outcome of the battle it labels;
* an UNANCHORED snapshot-ladder refit is REFUSED, never silently used as a strength axis
  (mixture-diagnostic hazard 1: ``fit_ladder`` returns ratings near 1000 from the wrong cwd);
* a cycle with no manifest ``selection`` block is SELECTION UNKNOWN and REFUSES (rule 17).

Pure unit, unmarked (runs in every tier): no ``models/``, no subprocess, no model forward.
"""
from __future__ import annotations

import json
from typing import Dict, Optional

import numpy as np
import pytest

from main.ops import calibration_slope as CS
from main.ops import conditioning_meters as CM
from main.ops import team_conditioning as TC

_TURNS = (1, 2, 3, 6, 12, 30)


def _plant(tmp_path, *, wr_by_opp: Dict[str, float], v_of: "callable",
           n_battles: int = 20, n_teams: int = 5, step: int = 100,
           seed: int = 0, schema: Optional[int] = 1):
    """A trace cycle with per-opponent win rates EXACTLY ``wr_by_opp`` and a planted ``V``.

    Capture rates are 1.0 on both outcome classes, so the traced sample IS the population and the
    manifest's ``battles_won / battles_played`` is reproduced by the rows — that keeps the test
    about the spread arithmetic rather than about the reweighting, which has its own test.
    """
    rng = np.random.default_rng(seed)
    cyc = tmp_path / "eval_traces" / f"step_{step}"
    sel: Dict[str, dict] = {}
    for opp, wr in wr_by_opp.items():
        odir = cyc / opp
        odir.mkdir(parents=True)
        wins = int(round(wr * n_battles))
        for b in range(n_battles):
            y = 1.0 if b < wins else 0.0
            v = float(np.clip(v_of(opp, wr, rng), 0.001, 0.999))
            n = len(_TURNS)
            np.savez(odir / f"b{b}_states.npz",
                     values=np.full(n, v), win_probs=np.full(n, v),
                     has_state=np.ones(n, dtype=np.int64))
            (odir / f"b{b}_summary.json").write_text(json.dumps({
                "meta": {"result": "WIN" if y else "LOSS", "step": step},
                "teams": {"ours": [{"species": f"mon{(b % n_teams) * 6 + k}"} for k in range(6)]},
                "invocations": [{"i": i, "turn": t} for i, t in enumerate(_TURNS)]}))
        sel[opp] = {"battles_played": n_battles, "battles_won": wins, "battles_drawn": 0,
                    "traces_written": n_battles, "traces_won": wins,
                    "capture_rate_win": 1.0, "capture_rate_loss": 1.0}
    man = {"step": step, "selection_schema": schema, "opponents": sorted(wr_by_opp),
           "selection": ({"opponents": sel} if schema is not None else None)}
    (cyc / "eval_manifest.json").write_text(json.dumps(man))
    return str(cyc)


_WRS = {"heuristic": 0.9, "staller": 0.7, "aggressive": 0.5,
        "sentinel_0": 0.3, "sentinel_1": 0.1}


def _separating(opp, wr, rng):
    """A head that says the truth about each opponent, plus per-battle noise."""
    return wr + rng.normal(0, 0.05)


def _marginal(opp, wr, rng):
    """A head that emits ONE number regardless of opponent — the defect under test."""
    return 0.6 + rng.normal(0, 0.05)


def _ratio(trace_dir, bucket="t1_3", boot=300, seed=7):
    arr, meta = CM.extract_cycle(trace_dir)
    b = CM.rollup(arr)
    opps, cid = CM.cell_index(b)
    n_cells = len(opps)
    true_wr = np.full(n_cells, np.nan)
    n_games = np.full(n_cells, np.nan)
    for o, wr, g in zip(b["opponent"].tolist(), b["true_wr"].tolist(), b["n_games"].tolist()):
        true_wr[opps.index(o)], n_games[opps.index(o)] = wr, g
    sel0 = np.arange(b["y"].size)
    st = CM.cell_stats(b, sel0, cid, n_cells, bucket, true_wr)
    return CM.spread_corrected(st, st["n_battles"] > 0, true_wr, n_games), meta


# --------------------------------------------------------------------------- the identity

def test_a_separating_head_recovers_a_spread_ratio_of_one(tmp_path) -> None:
    sc, meta = _ratio(_plant(tmp_path, wr_by_opp=_WRS, v_of=_separating, seed=1))
    assert meta["n_battles"] == 100 and meta["n_opponents"] == 5
    assert 0.80 < sc["ratio"] < 1.30, sc
    assert abs(sc["delta"]) < 0.06


def test_a_marginal_head_recovers_a_spread_ratio_of_zero(tmp_path) -> None:
    sc, _ = _ratio(_plant(tmp_path, wr_by_opp=_WRS, v_of=_marginal, seed=2))
    assert sc["ratio"] < 0.15, sc
    assert sc["delta"] < -0.20          # sd(V) far below sd(outcome)
    assert sc["sd_y"] > 0.20


def test_the_two_heads_are_separated_by_the_meter_by_a_wide_margin(tmp_path) -> None:
    sep, _ = _ratio(_plant(tmp_path / "sep", wr_by_opp=_WRS, v_of=_separating, seed=3))
    mar, _ = _ratio(_plant(tmp_path / "mar", wr_by_opp=_WRS, v_of=_marginal, seed=3))
    assert sep["ratio"] - mar["ratio"] > 0.6


def test_the_clamped_ratio_can_be_exactly_zero_while_the_raw_companion_is_not(tmp_path) -> None:
    """🚨 head-refit hazard 1. The corrected ratio floors a negative variance at zero, so a
    marginal head reads EXACTLY 0.000 while the unclamped companion is small but positive. A
    reader who takes the 0.000 as "no spread at all" has read a clamp, not a measurement."""
    sc, _ = _ratio(_plant(tmp_path, wr_by_opp=_WRS, v_of=_marginal, seed=4))
    assert sc["ratio"] == pytest.approx(0.0, abs=1e-9)
    assert 0.0 < sc["ratio_raw"] < 0.4
    assert np.isfinite(sc["noise_V"]) and sc["noise_V"] > 0


def test_the_interval_of_a_clamped_zero_sits_above_it(tmp_path) -> None:
    """The same hazard, through the whole block: the point is exactly 0.000, it sits AT OR BELOW
    its own interval's lower bound rather than inside it, and the interval reaches well above —
    which is why the render prints both the clamped and the raw ratio and calls the INTERVAL the
    read. A reader who takes the 0.000 at face value concludes "no spread at all"."""
    _plant(tmp_path, wr_by_opp=_WRS, v_of=_marginal, seed=4)
    blk = CM.conditioning_block(str(tmp_path), 100, boot=600, ladder="off", seed=4)
    pt = blk["points"]["cond.spread_ratio.t1_3"]
    lo, hi = np.percentile(blk["_draws"]["cond.spread_ratio.t1_3"], [2.5, 97.5])
    assert pt == pytest.approx(0.0, abs=1e-9)
    assert pt <= lo and hi > 0.02, (pt, lo, hi)   # never an interior point of its own interval
    assert blk["points"]["cond.spread_ratio_raw.t1_3"] > 0.0


def test_the_block_recovers_both_regimes_with_intervals(tmp_path) -> None:
    _plant(tmp_path / "sep", wr_by_opp=_WRS, v_of=_separating, seed=6)
    _plant(tmp_path / "mar", wr_by_opp=_WRS, v_of=_marginal, seed=6)
    sep = CM.conditioning_block(str(tmp_path / "sep"), 100, boot=400, ladder="off", seed=6)
    mar = CM.conditioning_block(str(tmp_path / "mar"), 100, boot=400, ladder="off", seed=6)
    s_lo = np.percentile(sep["_draws"]["cond.spread_ratio.t1_3"], 2.5)
    m_hi = np.percentile(mar["_draws"]["cond.spread_ratio.t1_3"], 97.5)
    assert sep["points"]["cond.spread_ratio.t1_3"] > 0.8
    assert mar["points"]["cond.spread_ratio.t1_3"] < 0.15
    assert s_lo > m_hi, (s_lo, m_hi)          # the two regimes' intervals do not overlap
    assert set(sep["_draws"]) >= {"cond.spread_ratio.all", "cond.opp_class_auc.t1"}


def test_the_elo_slope_row_is_omitted_with_a_reason_when_the_axis_is_off(tmp_path) -> None:
    _plant(tmp_path, wr_by_opp=_WRS, v_of=_marginal, seed=7)
    blk = CM.conditioning_block(str(tmp_path), 100, boot=100, ladder="off", seed=7)
    assert "cond.elo_slope" not in blk["points"]
    assert "DISABLED" in blk["omitted"]["cond.elo_slope"]


def test_the_opponent_class_auc_separates_a_head_that_ranks_the_pool(tmp_path) -> None:
    """`sentinel_*` is the POOL class. A head whose V tracks the true win rate ranks the two
    classes apart (the sentinels here are the strong ones); a marginal head cannot."""
    _plant(tmp_path / "sep", wr_by_opp=_WRS, v_of=_separating, seed=8)
    _plant(tmp_path / "mar", wr_by_opp=_WRS, v_of=_marginal, seed=8)
    sep = CM.conditioning_block(str(tmp_path / "sep"), 100, boot=200, ladder="off", seed=8)
    mar = CM.conditioning_block(str(tmp_path / "mar"), 100, boot=200, ladder="off", seed=8)
    assert sep["points"]["cond.opp_class_auc.t1"] > 0.90
    assert abs(mar["points"]["cond.opp_class_auc.t1"] - 0.5) < 0.20


# --------------------------------------------------------------------------- the own-team target

def test_the_leave_one_out_label_excludes_the_battle_it_labels() -> None:
    """Without the leave-one-out the label carries the outcome of its own battle, and any decoder
    that sees that battle's state scores against a target it was handed."""
    team = np.array(["A"] * 5 + ["B"] * 5)
    y = np.array([1.0, 1, 1, 1, 0, 0, 0, 0, 0, 1])
    w = np.ones(10)
    loo = CM.loo_team_wr(team, y, w, min_battles=4)
    assert loo[0] == pytest.approx(3 / 4)       # A's other four: 1,1,1,0
    assert loo[4] == pytest.approx(4 / 4)       # A's other four: 1,1,1,1
    assert loo[9] == pytest.approx(0.0)         # B's other four: 0,0,0,0


def test_a_team_under_the_minimum_gets_no_label() -> None:
    loo = CM.loo_team_wr(np.array(["A", "A", "B"]), np.array([1.0, 0, 1]), np.ones(3),
                         min_battles=4)
    assert np.isnan(loo).all()


def test_the_out_of_fold_decode_never_trains_and_tests_on_the_same_battle() -> None:
    """A grouped fold map must put every state of a battle on the same side. Broken, a decoder
    memorises the battle and every R² on this page is a memorisation score."""
    groups = np.repeat(np.arange(30), 4)
    folds = CM._fold_of(groups, 5, seed=0)
    for g in range(30):
        assert np.unique(folds[groups == g]).size == 1


def test_a_constant_feature_column_predicts_the_mean_rather_than_dividing_by_zero() -> None:
    x = np.full(200, 0.5)
    y = np.linspace(0, 1, 200)
    p = CM.grouped_oof_scalar(x, y, np.ones(200), np.repeat(np.arange(50), 4),
                              task="r2", seed=0)
    assert np.isfinite(p).all()
    assert CM.score(y, p, np.ones(200), "r2") < 0.05


def test_a_perfectly_informative_scalar_decodes_and_an_uninformative_one_does_not() -> None:
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(80), 4)
    y = rng.normal(size=80)[groups]
    good = CM.grouped_oof_scalar(y + rng.normal(0, 0.05, y.size), y, np.ones(y.size), groups,
                                 task="r2", seed=1)
    junk = CM.grouped_oof_scalar(rng.normal(size=y.size), y, np.ones(y.size), groups,
                                 task="r2", seed=1)
    assert CM.score(y, good, np.ones(y.size), "r2") > 0.95
    assert CM.score(y, junk, np.ones(y.size), "r2") < 0.10


# --------------------------------------------------------------------------- refusals

def test_a_cycle_without_a_manifest_refuses(tmp_path) -> None:
    (tmp_path / "eval_traces" / "step_100").mkdir(parents=True)
    with pytest.raises(CM.ConditioningRefusal, match="SELECTION UNKNOWN"):
        CM.extract_cycle(str(tmp_path / "eval_traces" / "step_100"))


def test_a_cycle_whose_manifest_has_no_selection_block_refuses(tmp_path) -> None:
    d = _plant(tmp_path, wr_by_opp={"heuristic": 0.5}, v_of=_marginal, schema=None, seed=9)
    with pytest.raises(CM.ConditioningRefusal, match="LOSS-ENRICHED"):
        CM.extract_cycle(d)


def test_a_battle_whose_outcome_class_has_no_capture_rate_is_dropped_never_weighted_one(
        tmp_path) -> None:
    """Rule 17's exact failure mode: mixing corrected rows with uncorrected ones and calling the
    result corrected. The row is dropped and the drop is REPORTED."""
    d = _plant(tmp_path, wr_by_opp={"heuristic": 0.5}, v_of=_marginal, n_battles=10, seed=10)
    man = json.loads(open(f"{d}/eval_manifest.json").read())
    man["selection"]["opponents"]["heuristic"]["capture_rate_loss"] = 0.0
    open(f"{d}/eval_manifest.json", "w").write(json.dumps(man))
    arr, meta = CM.extract_cycle(d)
    assert (arr["y"] == 1.0).all()
    assert sum("no capture rate" in r for r in meta["refusals"]) == 5


def test_an_unanchored_ladder_refit_is_refused_and_the_row_omitted(tmp_path, monkeypatch) -> None:
    """🚨 mixture-diagnostic hazard 1. `fit_ladder` silently returns an UNANCHORED ladder when it
    cannot find the bot pins — ratings near 1000 instead of near 2000, `anchored_to_bots` false,
    no error. Mixing those with the bot anchors puts one strength axis on two scales."""
    import agents.training.snapshot_ladder as SL

    d = _plant(tmp_path, wr_by_opp=_WRS, v_of=_marginal, seed=11)
    (tmp_path / "eval_results.jsonl").write_text(json.dumps(
        {"step": 100, "sentinels": [{"step": 50, "win_rate": 0.3},
                                    {"step": 60, "win_rate": 0.1}]}) + "\n")
    monkeypatch.setattr(SL, "load_games", lambda run: [(50, 60)])
    monkeypatch.setattr(SL, "fit_ladder",
                        lambda run, steps=None, write=False: {
                            "ratings": {"50": 1001.0, "60": 999.0}, "anchored_to_bots": False})
    per_opp = {o: {"true_win_rate": w} for o, w in _WRS.items()}
    axis, why = CM.strength_axis(str(tmp_path), 100, per_opp)
    assert axis is None
    assert "NOT anchored to the bot pins" in why and "OMITTED" in why
    assert d


def test_an_anchored_refit_with_a_verified_sentinel_map_produces_the_axis(
        tmp_path, monkeypatch) -> None:
    import agents.training.snapshot_ladder as SL

    _plant(tmp_path, wr_by_opp=_WRS, v_of=_separating, seed=12)
    (tmp_path / "eval_results.jsonl").write_text(json.dumps(
        {"step": 100, "sentinels": [{"step": 50, "win_rate": 0.3},
                                    {"step": 60, "win_rate": 0.1}]}) + "\n")
    monkeypatch.setattr(SL, "load_games", lambda run: [(50, 60)])
    monkeypatch.setattr(SL, "fit_ladder",
                        lambda run, steps=None, write=False: {
                            "ratings": {"50": 1907.0, "60": 2004.0}, "anchored_to_bots": True})
    per_opp = {o: {"true_win_rate": w} for o, w in _WRS.items()}
    axis, why = CM.strength_axis(str(tmp_path), 100, per_opp)
    assert axis is not None
    assert axis["sentinel_0"] == 1907.0 and axis["sentinel_1"] == 2004.0
    assert axis["heuristic"] > 1000.0            # from the committed bot anchors
    assert "verified" in why


def test_a_mismatched_sentinel_map_refuses_rather_than_guesses(tmp_path, monkeypatch) -> None:
    """The `sentinel_k` -> snapshot map is POSITIONAL and is verified against the manifest's own
    win counts. A mismatch means the positions have moved; guessing would silently label every
    sentinel cell with another snapshot's rating."""
    import agents.training.snapshot_ladder as SL

    _plant(tmp_path, wr_by_opp=_WRS, v_of=_separating, seed=13)
    (tmp_path / "eval_results.jsonl").write_text(json.dumps(
        {"step": 100, "sentinels": [{"step": 50, "win_rate": 0.77},
                                    {"step": 60, "win_rate": 0.1}]}) + "\n")
    monkeypatch.setattr(SL, "load_games", lambda run: [(50, 60)])
    monkeypatch.setattr(SL, "fit_ladder",
                        lambda run, steps=None, write=False: {
                            "ratings": {"50": 1907.0, "60": 2004.0}, "anchored_to_bots": True})
    axis, why = CM.strength_axis(str(tmp_path), 100,
                                 {o: {"true_win_rate": w} for o, w in _WRS.items()})
    assert axis is None and "MISMATCHES" in why


def test_the_frame_records_the_qc_that_v_is_the_win_prob_column(tmp_path) -> None:
    _plant(tmp_path, wr_by_opp=_WRS, v_of=_separating, seed=14)
    blk = CM.conditioning_block(str(tmp_path), 100, boot=100, ladder="off", seed=14)
    assert blk["frame"]["max_abs_values_minus_winprobs"] == 0.0
    assert blk["frame"]["selection_schema"] == 1
    assert "never pools cycles" in blk["frame"]["recorded_v_note"]


def test_every_declared_meter_is_either_reported_or_omitted_with_a_reason(tmp_path) -> None:
    _plant(tmp_path, wr_by_opp=_WRS, v_of=_separating, seed=15)
    blk = CM.conditioning_block(str(tmp_path), 100, boot=100, ladder="off", seed=15)
    for key in CM.METER_KEYS:
        if key in CM.PAIR_LEVEL_KEYS:
            # a PAIR-level row (the common-support slope) is fitted by `main.ops.critic_read` once
            # both sides exist; a single-run block neither reports nor omits it, because an
            # omission reason here would still be sitting there after the pair filled it in.
            assert key not in blk["points"] and key not in blk["omitted"], key
            continue
        assert (key in blk["points"]) ^ (key in blk["omitted"]), key


# ------------------------------------------------- (A) CONDITIONING vs (B) SUBSTITUTION
#
# Arm 8's ``V`` decodes its OWN TEAM at turn 1 where every control reads ~0, and the existing rows
# cannot say WHICH of two things that is: the critic CONDITIONING on its team (teams differ in
# strength; inside a team it still reads the board) or SUBSTITUTING team identity for board state
# (right per team, blind inside one). The rows below decompose it. Each regime is PLANTED with a
# known answer and recovered:
#
#   (a) team-mean-only    -> within-team resolution ~0, team spread ~1, late team R2 == t1
#   (b) team AND board    -> within-team resolution HIGH, team spread ~1, late team R2 << t1
#   (c) marginal-only     -> every component ~0
#
# (a) and (b) are the decisive pair: they are IDENTICAL between teams and opposite within one, so
# any row that separates them is measuring the within-team part and nothing else.

_TEAM_RATES = (0.20, 0.35, 0.45, 0.55, 0.65, 0.80)
_TURNS_LONG = (1, 2, 3, 8, 14, 22, 30, 38)
_TEAM_OPPS = ("heuristic", "staller", "aggressive", "sentinel_0")


def _plant_teams(tmp_path, *, v_of, team_rates=_TEAM_RATES, per_team: int = 30,
                 step: int = 100, seed: int = 0):
    """A cycle whose TEAMS differ in strength, with ``V`` planted as a function of the team's win
    rate, the battle's outcome and the TURN.

    Each team's win count is exact (``round(rate * per_team)``) and its battles are dealt
    round-robin across the opponents, so the team axis carries all the structure and the opponent
    axis carries none — the reverse of :func:`_plant`, and what makes these rows testable
    independently of the opponent identity.
    """
    rng = np.random.default_rng(seed)
    cyc = tmp_path / "eval_traces" / f"step_{step}"
    tally = {o: [0, 0] for o in _TEAM_OPPS}
    for o in _TEAM_OPPS:
        (cyc / o).mkdir(parents=True, exist_ok=True)
    for ti, p in enumerate(team_rates):
        wins = int(round(p * per_team))
        for k in range(per_team):
            opp = _TEAM_OPPS[(ti * per_team + k) % len(_TEAM_OPPS)]
            y = 1.0 if k < wins else 0.0
            tally[opp][0] += 1
            tally[opp][1] += int(y)
            vs = [float(np.clip(v_of(p, y, t, rng), 0.001, 0.999)) for t in _TURNS_LONG]
            base = f"t{ti}_b{k}"
            np.savez(cyc / opp / f"{base}_states.npz", values=np.array(vs),
                     win_probs=np.array(vs), has_state=np.ones(len(vs), dtype=np.int64))
            (cyc / opp / f"{base}_summary.json").write_text(json.dumps({
                "meta": {"result": "WIN" if y else "LOSS", "step": step},
                "teams": {"ours": [{"species": f"mon{ti * 6 + j}"} for j in range(6)]},
                "invocations": [{"i": i, "turn": t} for i, t in enumerate(_TURNS_LONG)]}))
    sel = {o: {"battles_played": tally[o][0], "battles_won": tally[o][1], "battles_drawn": 0,
               "traces_written": tally[o][0], "traces_won": tally[o][1],
               "capture_rate_win": 1.0, "capture_rate_loss": 1.0} for o in _TEAM_OPPS}
    (cyc / "eval_manifest.json").write_text(json.dumps(
        {"step": step, "selection_schema": 1, "opponents": list(_TEAM_OPPS),
         "selection": {"opponents": sel}}))
    return str(tmp_path)


def _board_weight(turn: int) -> float:
    """How much of ``V`` the BOARD carries at this turn — small at turn 1, dominant late. This is
    what a critic that reads the board is supposed to do, and it is what makes a conditioning
    critic's own-team decode FALL from turn 1 to late."""
    return 0.15 + 0.65 * min(1.0, (turn - 1) / 30.0)


def _team_mean_only(p, y, turn, rng):
    """(B) SUBSTITUTION planted exactly: one number per TEAM, whatever the board says."""
    return p


def _team_and_board(p, y, turn, rng):
    """(A) CONDITIONING planted exactly: the team's rate, plus a board term that is mean-zero
    GIVEN the team (wins move up by ``d(1-p)``, losses down by ``dp``) and grows with the clock.
    ``E[V | team] = p`` identically, so the two regimes share a between-team spread by
    construction and differ ONLY within a team."""
    return p + _board_weight(turn) * ((1 - p) if y else -p)


def _team_marginal(p, y, turn, rng):
    return 0.5 + rng.normal(0, 0.02)


def _ab(tmp_path, v_of, *, boot: int = 200, seed: int = 5, **kw):
    _plant_teams(tmp_path, v_of=v_of, seed=seed, **kw)
    return CM.conditioning_block(str(tmp_path), 100, boot=boot, ladder="off", seed=seed)


def test_a_team_mean_only_head_has_no_within_team_resolution(tmp_path) -> None:
    """(B) SUBSTITUTION. `V` is one number per team, so inside a team it separates nothing —
    both within-cell rows read ~0 — while the BETWEEN-team spread is a perfect 1.0 and the
    own-team decode is just as strong late as at turn 1."""
    blk = _ab(tmp_path, _team_mean_only)
    p = blk["points"]
    assert p["cond.within_team_resolution.all"] < 0.01, p
    assert p["cond.within_stratum_resolution.all"] < 0.01, p
    assert 0.80 < p["cond.team_spread_ratio.t1_3"] < 1.30, p
    assert p["cond.own_team_r2.t1"] > 0.8 and p["cond.own_team_r2.late"] > 0.8, p
    assert abs(p[CM.OWN_TEAM_R2_DIFF]) < 0.05, p


def test_a_team_conditioned_board_discriminating_head_resolves_within_the_team(tmp_path) -> None:
    """(A) CONDITIONING. The same between-team spread, but `V` also moves with the board, so the
    within-team resolution is high AND the own-team decode FALLS from turn 1 to late as board
    information takes over."""
    blk = _ab(tmp_path, _team_and_board)
    p = blk["points"]
    assert p["cond.within_team_resolution.all"] > 0.15, p
    assert p["cond.within_stratum_resolution.all"] > 0.15, p
    assert 0.80 < p["cond.team_spread_ratio.t1_3"] < 1.30, p
    assert p["cond.own_team_r2.late"] < p["cond.own_team_r2.t1"] - 0.30, p
    assert p[CM.OWN_TEAM_R2_DIFF] > 0.30, p


def test_a_marginal_head_moves_neither_component(tmp_path) -> None:
    """(c) NEITHER. One number for everything: no within-team resolution, no between-team spread,
    no own-team decode — the shape every control on the ladder reads."""
    blk = _ab(tmp_path, _team_marginal)
    p = blk["points"]
    assert p["cond.within_team_resolution.all"] < 0.02, p
    assert p["cond.within_stratum_resolution.all"] < 0.02, p
    assert p["cond.team_spread_ratio.t1_3"] < 0.15, p
    assert p["cond.own_team_r2.t1"] < 0.05 and p["cond.own_team_r2.late"] < 0.05, p


def test_the_two_readings_are_separated_by_the_within_team_row_alone(tmp_path) -> None:
    """🚨 THE DECISIVE CONTRAST. (a) and (b) are planted with the SAME per-team mean `V`, so their
    between-team spreads agree — and any separation between them is the within-team component and
    nothing else. That is the whole reason the row exists: the own-team R² row alone reads (a) and
    (b) as the same finding."""
    sub = _ab(tmp_path / "sub", _team_mean_only)["points"]
    cond = _ab(tmp_path / "cond", _team_and_board)["points"]
    assert abs(sub["cond.team_spread_ratio.t1_3"] - cond["cond.team_spread_ratio.t1_3"]) < 0.05
    assert cond["cond.within_team_resolution.all"] - sub["cond.within_team_resolution.all"] > 0.15
    assert (cond["cond.within_stratum_resolution.all"]
            - sub["cond.within_stratum_resolution.all"]) > 0.15
    assert cond[CM.OWN_TEAM_R2_DIFF] - sub[CM.OWN_TEAM_R2_DIFF] > 0.30


def test_the_new_rows_carry_battle_clustered_draws_and_an_interval(tmp_path) -> None:
    blk = _ab(tmp_path, _team_and_board, boot=200)
    for key in ("cond.within_team_resolution.all", "cond.within_stratum_resolution.all",
                "cond.team_spread_ratio.t1_3", "cond.team_spread_ratio_raw.t1_3",
                "cond.own_team_r2.late", CM.OWN_TEAM_R2_DIFF):
        d = blk["_draws"][key]
        assert d.size > 150 and np.isfinite(d).all(), key


def test_the_contrast_row_is_the_difference_of_its_two_scores(tmp_path) -> None:
    """The point is exactly `t1 − late`, and its draws are PAIRED — the two scores come from the
    same resampled battles, so the interval carries their covariance instead of pretending they
    are independent draws."""
    blk = _ab(tmp_path, _team_and_board, boot=120)
    p = blk["points"]
    assert p[CM.OWN_TEAM_R2_DIFF] == pytest.approx(
        p["cond.own_team_r2.t1"] - p["cond.own_team_r2.late"])
    d = blk["_draws"]
    assert d[CM.OWN_TEAM_R2_DIFF].size == d["cond.own_team_r2.t1"].size
    assert np.allclose(d[CM.OWN_TEAM_R2_DIFF],
                       d["cond.own_team_r2.t1"] - d["cond.own_team_r2.late"])


def test_the_cell_census_is_reported_beside_every_within_cell_row(tmp_path) -> None:
    """🚨 A within-cell resolution is not readable without its census: on cells of a handful of
    episodes the number is largely the binning's own (positively biased) noise. The counts are
    part of the block, not an appendix."""
    blk = _ab(tmp_path, _team_and_board, boot=60)
    cf = blk["frame"]["cell_frames"]
    for key in ("cond.within_team_resolution.all", "cond.within_stratum_resolution.all"):
        c = cf[key]
        assert c["n_cells"] > 0 and c["n_battles"] == 180 and c["n_states"] > 0
        assert c["median_battles_per_cell"] > 0 and c["median_states_per_cell"] > 0
    assert cf["cond.within_team_resolution.all"]["n_cells"] == 6
    assert cf["cond.within_stratum_resolution.all"]["n_cells"] == TC.TEAM_STRATA
    assert blk["frame"]["n_team_cells"] == 6 and blk["frame"]["team_strata"] == TC.TEAM_STRATA
    assert blk["frame"]["team_spread_frame"]["n_cells"] == 6


def test_a_team_under_the_minimum_is_never_a_cell(tmp_path) -> None:
    """The same threshold the leave-one-battle-out label uses. A team with too few battles has no
    trustworthy win rate, so it is not a cell, is not in a stratum, and is in no census count."""
    blk = _ab(tmp_path, _team_and_board, per_team=CM.MIN_TEAM_BATTLES - 1, boot=0,
              team_rates=(0.3, 0.5, 0.7, 0.9))
    assert blk["frame"]["n_team_cells"] == 0
    for key in ("cond.within_team_resolution.all", "cond.within_stratum_resolution.all",
                "cond.team_spread_ratio.t1_3"):
        assert key in blk["omitted"], key
        assert key not in blk["points"], key


def test_the_within_cell_resolution_reduces_to_the_ordinary_murphy_resolution(tmp_path) -> None:
    """ONE cell means the within-cell construction IS Murphy's resolution — checked against the
    ladder's own implementation in `critic_readouts`, so the two cannot drift apart."""
    from main.ops import critic_readouts as CRO

    rng = np.random.default_rng(3)
    v = rng.random(400)
    y = (rng.random(400) < v).astype(float)
    w = rng.random(400) + 0.5
    got = TC.cellwise_resolution(v, y, w, np.zeros(400, dtype=int), np.array([1.0]), 1)
    want = CRO.murphy_arrays(v, y, np.ones(400), w, TC.RESOLUTION_BINS)["resolution"]
    assert got == pytest.approx(want, abs=1e-12)


def test_the_strata_are_cut_at_equal_battle_mass_not_equal_team_count() -> None:
    """A stratum of five rarely-drawn teams has the same small-cell problem the per-team row has,
    so the cut is on BATTLES. One very heavy weak team must not swallow two strata."""
    rate = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    # EQUAL battle counts: five strata of two teams each, which equal-team-count cutting would
    # also produce — the case where the two rules agree.
    st = TC.strata_of(rate, np.full(10, 10.0), np.ones(10, dtype=bool), 5)
    assert st.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    # ONE VERY HEAVY TEAM: equal-team-count cutting would put it in a stratum with a much lighter
    # neighbour; cutting on battle mass leaves it in a stratum of its own, and no team is ever
    # split across two.
    nb = np.array([100.0, 4, 4, 4, 4, 4, 4, 4, 4, 4])
    st = TC.strata_of(rate, nb, np.ones(10, dtype=bool), 5)
    assert (st >= 0).all() and (np.diff(st) >= 0).all()    # monotone in strength
    assert (st == st[0]).sum() == 1                        # the heavy team is alone
    assert TC.strata_of(rate, nb, np.zeros(10, dtype=bool), 5).tolist() == [-1] * 10


def test_the_reading_helper_names_the_reading_each_sign_pattern_supports() -> None:
    """The (A)/(B) sentence is a reading of three SIGNS — never a verdict, which comes from the
    registered label machinery."""
    assert TC.reading_of(-0.1, -0.1, +0.2).startswith("(B) SUBSTITUTION")
    assert TC.reading_of(+0.1, +0.1, +0.2).startswith("(A) CONDITIONING")
    assert TC.reading_of(0.0, 0.0, +0.2).startswith("(A) CONDITIONING")
    assert TC.reading_of(0.0, 0.0, 0.0).startswith("neither")
    assert TC.reading_of(None, None, None).startswith("neither")


def test_the_provisional_contrast_row_declares_why_it_can_have_no_floor() -> None:
    """The controls read own-team R² ~0 at turn 1, so there is nothing for the contrast to FALL
    from and no two-draw replicate floor can be formed for it. The row therefore carries no
    verdict at all — which is a property of the METER, declared beside it."""
    m = CM.METER_BY_KEY[CM.OWN_TEAM_R2_DIFF]
    assert m.provisional is True and m.frame_sensitive is True
    assert "floor" in m.provisional_why.lower()
    assert CM.PROVISIONAL_KEYS == (CM.OWN_TEAM_R2_DIFF,)


# --------------------------------------------------------- the CALIBRATION SLOPE (SHRINKAGE)
#
# Arm 8 reads "alignment up, amplitude down": its ``V`` is better ORDERED by own-team strength
# than every control while its between-team SPREAD is smaller. One account of that pairing is
# SHRINKAGE — a bootstrapped (λ-return) target blends the critic's own ``V`` into the label, so
# the fitted target is compressed toward the base rate, and fitting a compressed target is a
# shrinkage estimator: better rank order, smaller amplitude. The sharp signature is the
# CALIBRATION SLOPE, and the tests below plant all three regimes with a known answer:
#
#   V == the true probability     -> slope ~ 1   (correctly dispersed)
#   V shrunk toward the base rate -> slope > 1   (UNDER-dispersed: 0.7 is followed by MORE than 0.7)
#   V stretched away from it      -> slope < 1   (over-dispersed)
#
# The middle row is the whole hypothesis, and the third is here because a test that only separates
# "1" from "big" cannot tell a slope estimator from a magnitude one.

_CAL_TURNS = (1, 2, 3, 9, 18, 30)
_CAL_OPPS = ("heuristic", "staller", "aggressive", "sentinel_0")


def _true(p):
    """A forecast that IS the probability — the calibrated reference, slope 1."""
    return p


def _shrunk(p, k: float = 0.5):
    """(C) SHRINKAGE planted exactly: every forecast pulled a fraction ``k`` of the way to 0.5.
    The rank ORDER is untouched (the map is monotone), only the AMPLITUDE is — which is precisely
    why a scale-invariant decode cannot see it and the slope can."""
    return 0.5 + k * (np.asarray(p, dtype=float) - 0.5)


def _stretched(p, k: float = 1.8):
    """The opposite defect: opinions more extreme than the evidence behind them, slope < 1."""
    x = np.log(np.asarray(p, dtype=float)) - np.log1p(-np.asarray(p, dtype=float))
    return 1.0 / (1.0 + np.exp(-k * x))


def _draw(n: int, seed: int, lo: float = 0.08, hi: float = 0.92):
    """``(p, y, w)`` — true probabilities, outcomes drawn from them, and unit weights."""
    rng = np.random.default_rng(seed)
    p = rng.uniform(lo, hi, n)
    return p, (rng.random(n) < p).astype(float), np.ones(n)


def test_a_forecast_that_is_the_true_probability_reads_a_slope_of_one() -> None:
    p, y, w = _draw(40000, 3)
    fit = CS.slope_intercept(_true(p), y, w)
    assert abs(fit["slope"] - 1.0) < 0.06, fit
    assert abs(fit["intercept"]) < 0.06, fit


def test_a_shrunk_forecast_reads_a_slope_ABOVE_one() -> None:
    """🚨 THE HYPOTHESIS. A forecast squeezed toward the base rate is UNDER-dispersed: where it
    says 0.7 the realized rate is above 0.7. The slope is the reciprocal of the squeeze on the
    logit scale, so it is well above 1 and the sign is not a matter of interpretation."""
    p, y, w = _draw(40000, 4)
    fit = CS.slope_intercept(_shrunk(p), y, w)
    assert fit["slope"] > 1.5, fit
    assert abs(fit["intercept"]) < 0.06, fit


def test_an_over_dispersed_forecast_reads_a_slope_BELOW_one() -> None:
    p, y, w = _draw(40000, 5)
    fit = CS.slope_intercept(_stretched(p), y, w)
    assert fit["slope"] < 0.75, fit


def test_the_three_regimes_are_ordered_and_only_the_slope_separates_them() -> None:
    """The decisive contrast: all three forecasts carry the SAME rank order (each map is strictly
    monotone in ``p``), so every scale-invariant statistic reads them identically. The slope does
    not — which is the entire reason this row was added beside the out-of-fold decodes."""
    p, y, w = _draw(40000, 6)
    slopes = [CS.slope_intercept(f(p), y, w)["slope"] for f in (_stretched, _true, _shrunk)]
    assert slopes[0] < slopes[1] < slopes[2], slopes
    aucs = [CM.w_auc(y, f(p), w) for f in (_stretched, _true, _shrunk)]
    assert max(aucs) - min(aucs) < 1e-9, aucs


def test_a_separated_fit_returns_nan_rather_than_a_diverging_coefficient() -> None:
    """Perfect separation drives the coefficient to infinity. A diverging slope reported as a
    number is a wrong reading with no tell, so the fit REFUSES."""
    p = np.linspace(0.05, 0.95, 400)
    y = (p > 0.5).astype(float)
    assert not np.isfinite(CS.slope_intercept(p, y, np.ones(p.size))["slope"])


def test_one_outcome_class_or_too_few_states_returns_nan() -> None:
    p, _y, w = _draw(200, 7)
    assert not np.isfinite(CS.slope_intercept(p, np.ones(200), w)["slope"])
    assert not np.isfinite(CS.slope_intercept(p[:3], np.array([1.0, 0.0, 1.0]), w[:3])["slope"])


def test_the_weights_are_honoured_by_the_fit() -> None:
    """The rows are Horvitz-Thompson reweighted, so a fit that ignored ``w`` would read the
    LOSS-ENRICHED tree. Duplicating a subset must equal doubling its weight, exactly."""
    p, y, w = _draw(3000, 8)
    half = np.arange(0, 3000, 2)
    dup = CS.slope_intercept(np.concatenate([p, p[half]]), np.concatenate([y, y[half]]),
                             np.ones(3000 + half.size))
    wt = w.copy()
    wt[half] = 2.0
    assert abs(dup["slope"] - CS.slope_intercept(p, y, wt)["slope"]) < 1e-6


# ---- the WITHIN-STRATUM companion

def test_the_within_stratum_slope_absorbs_a_between_stratum_offset() -> None:
    """Shrinkage ACROSS cells and shrinkage INSIDE one are different statements. Plant a forecast
    that is correctly dispersed INSIDE every stratum but whose per-stratum level is compressed
    between them: the pooled slope reads high and the within-stratum slope reads ~1."""
    rng = np.random.default_rng(9)
    n_s, per = 5, 9000
    strat = np.repeat(np.arange(n_s), per)
    centre = np.linspace(-1.4, 1.4, n_s)[strat]
    x = centre + rng.normal(0, 0.35, n_s * per)          # true logit
    p = 1.0 / (1.0 + np.exp(-x))
    y = (rng.random(x.size) < p).astype(float)
    v = 1.0 / (1.0 + np.exp(-(0.4 * centre + (x - centre))))   # levels squeezed, within intact
    w = np.ones(x.size)
    assert CS.slope_intercept(v, y, w)["slope"] > 1.3
    assert abs(CS.within_stratum_slope(v, y, w, strat) - 1.0) < 0.12


def test_a_stratum_with_no_outcome_variation_is_dropped_rather_than_diverging() -> None:
    """A stratum that is all wins would diverge on its OWN dummy and take the shared slope's
    convergence with it — the fit would return NaN for a reason with nothing to do with the
    slope. It is dropped, and the surviving strata still produce the slope."""
    p, y, w = _draw(9000, 10)
    strat = np.zeros(p.size, dtype=int)
    strat[: p.size // 3] = 1
    y[: p.size // 3] = 1.0                                # stratum 1 is all wins
    s = CS.within_stratum_slope(p, y, w, strat)
    assert np.isfinite(s) and abs(s - 1.0) < 0.15, s
    assert not np.isfinite(CS.within_stratum_slope(p, np.ones(p.size), w, strat))


def test_states_outside_every_stratum_are_excluded() -> None:
    p, y, w = _draw(6000, 11)
    strat = np.full(p.size, -1, dtype=int)
    strat[: 3000] = 0
    assert np.isfinite(CS.within_stratum_slope(p, y, w, strat))
    assert not np.isfinite(CS.within_stratum_slope(p, y, w, np.full(p.size, -1)))


# ---- the LEVER ARM and the COMMON SUPPORT

def test_the_support_stats_report_a_shorter_lever_arm_for_the_shrunk_forecast() -> None:
    """🚨 The conservative bias that must never be silent: the slope's SE scales as
    1/sd(logit V), and the shrunk arm's sd is smaller BY CONSTRUCTION. The number that says so is
    printed beside every slope row."""
    p, _y, w = _draw(20000, 12)
    a = CS.support_stats(_shrunk(p), w)
    c = CS.support_stats(_true(p), w)
    assert a["sd_logit_V"] < c["sd_logit_V"] * 0.75, (a, c)
    assert a["sd_V"] < c["sd_V"]
    assert a["q_lo"] > c["q_lo"] and a["q_hi"] < c["q_hi"]


def test_the_shrunk_arms_interval_is_the_wider_one_at_equal_n() -> None:
    """The bias stated as an outcome, not as an argument: the same battles, the same outcomes, a
    monotone re-map of the forecast — and the compressed side gets the wider interval."""
    rng = np.random.default_rng(13)
    p, y, w = _draw(4000, 14)

    def width(f):
        d = [CS.slope_intercept(f(p[i]), y[i], w[i])["slope"]
             for i in (rng.integers(0, p.size, p.size) for _ in range(120))]
        lo, hi = np.percentile([x for x in d if np.isfinite(x)], [2.5, 97.5])
        return hi - lo

    assert width(_shrunk) > width(_true)


def test_the_common_window_is_the_intersection_and_a_disjoint_pair_is_refused() -> None:
    a = {"q_lo": 0.2, "q_hi": 0.8}
    c = {"q_lo": 0.3, "q_hi": 0.9}
    assert CS.common_window(a, c) == (0.3, 0.8)
    assert CS.common_window({"q_lo": 0.1, "q_hi": 0.2}, {"q_lo": 0.5, "q_hi": 0.9}) is None
    assert CS.common_window({"q_lo": float("nan"), "q_hi": 0.8}, c) is None


def test_the_clip_is_reported_as_a_share_of_the_column() -> None:
    v = np.concatenate([np.full(90, 0.5), np.full(10, 0.0)])
    st = CS.support_stats(v, np.ones(v.size))
    assert abs(st["clipped_share"] - 0.10) < 1e-9
    assert np.isfinite(st["sd_logit_V"])


# ---- the BLOCK, on a planted trace tree

def _plant_calib(tmp_path, *, distort, n_teams: int = 12, per_team: int = 40, step: int = 100,
                 seed: int = 0):
    """A cycle whose per-battle TRUE win probability is known and whose ``V`` is a planted
    distortion of it. Teams differ in strength (so the strata are real) and the probability varies
    INSIDE a team (so the within-stratum slope has something to regress on)."""
    rng = np.random.default_rng(seed)
    cyc = tmp_path / "eval_traces" / f"step_{step}"
    tally = {o: [0, 0] for o in _CAL_OPPS}
    for o in _CAL_OPPS:
        (cyc / o).mkdir(parents=True, exist_ok=True)
    bases = np.linspace(0.28, 0.72, n_teams)
    for ti in range(n_teams):
        for k in range(per_team):
            p = float(np.clip(bases[ti] + rng.normal(0, 0.14), 0.05, 0.95))
            y = 1.0 if rng.random() < p else 0.0
            v = float(np.clip(distort(p), 1e-4, 1 - 1e-4))
            opp = _CAL_OPPS[(ti * per_team + k) % len(_CAL_OPPS)]
            tally[opp][0] += 1
            tally[opp][1] += int(y)
            base = f"t{ti}_b{k}"
            n = len(_CAL_TURNS)
            np.savez(cyc / opp / f"{base}_states.npz", values=np.full(n, v),
                     win_probs=np.full(n, v), has_state=np.ones(n, dtype=np.int64))
            (cyc / opp / f"{base}_summary.json").write_text(json.dumps({
                "meta": {"result": "WIN" if y else "LOSS", "step": step},
                "teams": {"ours": [{"species": f"mon{ti * 6 + j}"} for j in range(6)]},
                "invocations": [{"i": i, "turn": t} for i, t in enumerate(_CAL_TURNS)]}))
    sel = {o: {"battles_played": tally[o][0], "battles_won": tally[o][1], "battles_drawn": 0,
               "traces_written": tally[o][0], "traces_won": tally[o][1],
               "capture_rate_win": 1.0, "capture_rate_loss": 1.0} for o in _CAL_OPPS}
    (cyc / "eval_manifest.json").write_text(json.dumps(
        {"step": step, "selection_schema": 1, "opponents": list(_CAL_OPPS),
         "selection": {"opponents": sel}}))
    return str(tmp_path)


def _cal(tmp_path, distort, *, boot: int = 120, seed: int = 5, **kw):
    _plant_calib(tmp_path, distort=distort, seed=seed, **kw)
    return CM.conditioning_block(str(tmp_path), 100, boot=boot, ladder="off", seed=seed)


def test_the_block_recovers_a_calibrated_head_at_a_slope_of_one(tmp_path) -> None:
    blk = _cal(tmp_path, _true)
    p = blk["points"]
    assert 0.7 < p[CM.CALIB_SLOPE_ALL] < 1.4, p[CM.CALIB_SLOPE_ALL]
    assert abs(p[CM.CALIB_INTERCEPT_ALL]) < 0.3, p[CM.CALIB_INTERCEPT_ALL]


def test_the_block_recovers_a_shrunk_head_ABOVE_one_and_a_stretched_head_BELOW(tmp_path) -> None:
    """The three planted regimes through the whole pipeline — extraction, HT weights, the per
    battle state cap and the fit — not just through the estimator."""
    shrunk = _cal(tmp_path / "s", _shrunk)["points"]
    true = _cal(tmp_path / "t", _true)["points"]
    stretched = _cal(tmp_path / "o", _stretched)["points"]
    assert shrunk[CM.CALIB_SLOPE_ALL] > 1.5, shrunk[CM.CALIB_SLOPE_ALL]
    assert stretched[CM.CALIB_SLOPE_ALL] < 0.8, stretched[CM.CALIB_SLOPE_ALL]
    assert (stretched[CM.CALIB_SLOPE_ALL] < true[CM.CALIB_SLOPE_ALL]
            < shrunk[CM.CALIB_SLOPE_ALL])


def test_every_calibration_row_carries_battle_clustered_draws_and_an_interval(tmp_path) -> None:
    blk = _cal(tmp_path, _shrunk, boot=200)
    for key in (CM.CALIB_SLOPE_ALL, CM.CALIB_SLOPE_T13, CM.CALIB_SLOPE_WITHIN,
                CM.CALIB_INTERCEPT_ALL, CM.CALIB_INTERCEPT_T13):
        d = blk["_draws"][key]
        assert d.size > 100, key
        lo, hi = np.percentile(d, [2.5, 97.5])
        assert lo < blk["points"][key] < hi, key
    lo, _hi = np.percentile(blk["_draws"][CM.CALIB_SLOPE_ALL], [2.5, 97.5])
    assert lo > 1.0, "a planted shrunk head's interval must sit clear of the calibrated 1.0"


def test_a_calibrated_head_interval_covers_one_and_a_shrunk_one_does_not(tmp_path) -> None:
    """The interval is what a verdict is read from, so the SEPARATION has to be at the interval
    and not only at the point."""
    cal = _cal(tmp_path / "c", _true, boot=200)
    shr = _cal(tmp_path / "s", _shrunk, boot=200)
    clo, chi = np.percentile(cal["_draws"][CM.CALIB_SLOPE_ALL], [2.5, 97.5])
    slo, _shi = np.percentile(shr["_draws"][CM.CALIB_SLOPE_ALL], [2.5, 97.5])
    assert clo < 1.0 < chi, (clo, chi)
    assert slo > chi, (slo, chi)


def test_the_lever_arm_is_reported_in_the_frame_beside_every_slope_row(tmp_path) -> None:
    """🚨 The support is not an appendix. Without sd(logit V) on the page a wider interval on the
    shrunk side reads as a null instead of as the conservative bias it is."""
    shr = _cal(tmp_path / "s", _shrunk)["frame"]["calibration_support"]
    cal = _cal(tmp_path / "c", _true)["frame"]["calibration_support"]
    for bucket in CS.BUCKETS:
        for k in ("sd_V", "sd_logit_V", "q_lo", "q_hi", "clipped_share", "n_states",
                  "n_battles"):
            assert k in shr[bucket], (bucket, k)
    assert shr["all"]["sd_logit_V"] < cal["all"]["sd_logit_V"]


def test_the_within_stratum_row_is_reported_beside_the_pooled_one(tmp_path) -> None:
    blk = _cal(tmp_path, _shrunk)
    assert np.isfinite(blk["points"][CM.CALIB_SLOPE_WITHIN])
    assert blk["points"][CM.CALIB_SLOPE_WITHIN] > 1.2, blk["points"]


def test_the_common_support_rows_are_never_emitted_by_a_single_run(tmp_path) -> None:
    """A PAIR-level row has no single-run value: its window is the intersection of two sides."""
    blk = _cal(tmp_path, _true)
    for key in CM.PAIR_LEVEL_KEYS:
        assert key not in blk["points"] and key not in blk["_draws"]


def test_the_common_support_fit_equalises_the_lever_arm(tmp_path) -> None:
    """Re-fitting both sides inside their shared window removes the lever-arm difference BY
    CONSTRUCTION — which is the whole point of the companion row, and is asserted rather than
    argued."""
    a = _cal(tmp_path / "s", _shrunk, boot=40)
    c = _cal(tmp_path / "c", _true, boot=40)
    sa = a["frame"]["calibration_support"]["all"]
    sc = c["frame"]["calibration_support"]["all"]
    win = CS.common_window(sa, sc)
    assert win is not None
    blocks = {r: CS.block(b["_calib"], names=CM.CALIB_COMMON_NAMES, window=win, boot=40, seed=1)
              for r, b in (("arm", a), ("control", c))}
    wide = [blocks[r]["support"]["all"] for r in ("arm", "control")]
    assert max(s["q_hi"] for s in wide) <= win[1] + 1e-9
    assert min(s["q_lo"] for s in wide) >= win[0] - 1e-9
    assert blocks["arm"]["points"][CM.CALIB_SLOPE_COMMON] > \
        blocks["control"]["points"][CM.CALIB_SLOPE_COMMON]


def test_the_calibration_payload_round_trips_through_json(tmp_path) -> None:
    """`main.ops.critic_read` CACHES these columns so the common-support companion costs no second
    extraction. A round trip that lost a column would silently disable the row on every reuse."""
    blk = _cal(tmp_path, _true, boot=20)
    back = CS.from_json(json.loads(json.dumps(CS.to_json(blk["_calib"]))))
    assert back["n_battles"] == blk["_calib"]["n_battles"]
    for bucket in blk["_calib"]["buckets"]:
        for col, v in blk["_calib"]["buckets"][bucket].items():
            assert np.allclose(np.asarray(v, dtype=float), back["buckets"][bucket][col]), col
    assert back["buckets"]["all"]["stratum"].dtype.kind == "i"
