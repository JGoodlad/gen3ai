"""``main.ops.quota_match`` — the frame equalisation, on planted trees where the answer is known.

The module exists because of one measured failure: a conditioning row whose estimator is a FIT on
the frame read **DETECTED** between an arm traced at 40/40/10 and a control at 5/10/5, and **NOT
DETECTED** once the two frames were the same size (ledger 2026-09-09 · RETRACTION). So the tests
plant exactly that shape and assert the tool's whole job:

1. **The planted artefact is caught.** Two trees whose critics are the SAME function are traced at
   different quotas, and the extra frame alone buys the richer side a higher own-team R². Unmatched
   the delta is DETECTED; matched it is not. This is the test that would have failed before the
   fix, and it fails again on a revert.
2. **A symmetric pair is untouched, bit for bit.** Equal realized caps ⇒ no subsample, and
   ``compute_deltas`` returns rows byte-identical to the ones it returns with matching disabled.
   The fix must not move a number it was not meant to move.
3. **The opt-out prints a MARKER, never a label.** ``--no-quota-match`` on an unequal pair leaves
   every frame-sensitive row carrying ``UNMATCHED — not a reading`` and neither DETECTED nor NOT
   DETECTED — on an unequal frame both are claims the report may not make.
4. **The realized profile is read off the DISK, including the shard rounding.** A nominal 5/10/5
   lands as 8/12 per opponent; matching the NOMINAL numbers over-shrinks the richer side by ~35%.
   A manifest that CLAIMS a different count than the tree holds is reported and the disk wins.
5. **Symmetry.** A richer CONTROL is subsampled, not the arm — the rule is about frames, not roles.

Pure unit, unmarked (runs in every tier): no ``models/``, no subprocess, no model forward.
"""
from __future__ import annotations

import json
from typing import Dict, Optional

import numpy as np
import pytest

from main.ops import conditioning_meters as CM
from main.ops import critic_read as CR
from main.ops import quota_match as QM

STEP = 500
_TURNS = (1, 2, 3, 8, 20)


def _plant(root, *, quota_w: int, quota_l: int, n_battles: int = 60, n_teams: int = 12,
           seed: int = 0, team_signal: float = 0.0, claim: Optional[Dict[str, tuple]] = None):
    """A trace cycle at a given per-opponent capture quota, with a per-TEAM signal in ``V``.

    Every opponent plays ``n_battles`` games; the first ``quota_w`` wins and ``quota_l`` losses of
    each are TRACED, which is what the eval quota does. ``team_signal`` scales how much of ``V``
    is a function of the battle's own team, so the own-team decoder has something real to find and
    the only thing separating two trees can be how many battles their decoders were fit on.
    """
    rng = np.random.default_rng(seed)
    cyc = root / "eval_traces" / f"step_{STEP}"
    sel: Dict[str, dict] = {}
    for oi, (opp, wr) in enumerate((("heuristic", 0.8), ("staller", 0.6), ("aggressive", 0.5),
                                    ("sentinel_0", 0.35), ("sentinel_1", 0.2))):
        odir = cyc / opp
        odir.mkdir(parents=True)
        team_bias = {t: float(rng.normal(0, 0.12)) for t in range(n_teams)}
        wins = int(round(wr * n_battles))
        w_written = l_written = 0
        for b in range(n_battles):
            y = 1.0 if b < wins else 0.0
            if y and w_written >= quota_w:
                continue
            if not y and l_written >= quota_l:
                continue
            w_written += int(y)
            l_written += int(not y)
            team = (b * 7 + oi * 3) % n_teams
            v = float(np.clip(wr + team_signal * team_bias[team] + rng.normal(0, 0.04),
                              0.01, 0.99))
            n = len(_TURNS)
            np.savez(odir / f"b{b}_states.npz", values=np.full(n, v), win_probs=np.full(n, v),
                     has_state=np.ones(n, dtype=np.int64))
            (odir / f"b{b}_summary.json").write_text(json.dumps({
                "meta": {"result": "WIN" if y else "LOSS", "step": STEP},
                "teams": {"ours": [{"species": f"m{team}_{k}"} for k in range(6)]},
                "invocations": [{"i": i, "turn": t} for i, t in enumerate(_TURNS)]}))
        cw, cl = (w_written, l_written) if claim is None else claim.get(opp, (w_written,
                                                                             l_written))
        sel[opp] = {"battles_played": n_battles, "battles_won": wins, "battles_drawn": 0,
                    "traces_written": cw + cl, "traces_won": cw, "traces_drawn": 0,
                    "capture_rate_win": w_written / wins,
                    "capture_rate_loss": l_written / (n_battles - wins)}
    (cyc / "eval_manifest.json").write_text(json.dumps(
        {"step": STEP, "selection_schema": 1, "opponents": sorted(sel),
         "selection": {"opponents": sel, "win_quota": quota_w, "loss_quota": quota_l}}))
    return str(cyc)


def _block(root, trace_dir, *, boot: int = 400):
    return CM.conditioning_block(str(root), STEP, boot=boot, seed=0, ladder="off")


# --------------------------------------------------------------------------- the profile

def test_the_realized_profile_is_read_off_the_disk_not_the_nominal_quota(tmp_path) -> None:
    """🚨 A nominal quota is not a realized one. The cap this tool matches on is what the tree
    HOLDS, per opponent, per outcome class — the counted files, not the manifest's quota fields."""
    td = _plant(tmp_path, quota_w=8, quota_l=12)
    p = QM.realized_profile("r", STEP, td)
    assert p.cap == (8, 12, 0)
    assert p.n_traced == sum(o["wins"] + o["losses"] for o in p.opponents.values())
    # the cap is a MAXIMUM over opponents: `sentinel_1` wins only 20% of 60, so it cannot fill an
    # 8-win quota... it can (12 wins available), but `heuristic` loses only 12 of 60 and fills the
    # loss quota exactly. Every opponent's own counts are carried, not just the cap.
    assert set(p.opponents) == {"heuristic", "staller", "aggressive", "sentinel_0", "sentinel_1"}
    assert p.disk_mismatches == []


def test_a_manifest_that_claims_a_count_the_tree_does_not_hold_is_reported_and_the_disk_wins(
        tmp_path) -> None:
    """The manifest records what the trainer INTENDED to write; the disk is what a read consumes.
    A disagreement is named in the profile rather than silently resolved, because a profile that
    trusted the manifest would cut the richer side to a frame that does not exist."""
    td = _plant(tmp_path, quota_w=8, quota_l=12, claim={"staller": (99, 99)})
    p = QM.realized_profile("r", STEP, td)
    assert any("staller" in m and "DISK" in m for m in p.disk_mismatches), p.disk_mismatches
    assert p.opponents["staller"]["wins"] == 8 and p.opponents["staller"]["losses"] == 12
    assert p.cap[0] == 8                      # the 99 never reaches the cap


def test_equal_caps_are_a_symmetric_pair_and_nothing_is_subsampled(tmp_path) -> None:
    a = QM.realized_profile("a", STEP, _plant(tmp_path / "a", quota_w=8, quota_l=12, seed=1))
    c = QM.realized_profile("c", STEP, _plant(tmp_path / "c", quota_w=8, quota_l=12, seed=2))
    pl = QM.plan(a, c)
    assert pl["needed"] is False and pl["sides"] == []
    assert "SAME realized cap" in pl["why"]


def test_the_richer_side_is_the_one_cut_whichever_role_it_holds(tmp_path) -> None:
    """🚨 Symmetry. A control re-traced at the HIGHER quota is the side that gets subsampled; the
    rule is about frames, not about which run is called the arm."""
    poor = QM.realized_profile("poor", STEP, _plant(tmp_path / "p", quota_w=8, quota_l=12, seed=1))
    rich = QM.realized_profile("rich", STEP, _plant(tmp_path / "r", quota_w=30, quota_l=40, seed=2))
    assert QM.plan(rich, poor)["sides"] == ["arm"]
    assert QM.plan(poor, rich)["sides"] == ["control"]
    assert QM.plan(poor, rich)["caps"][:2] == (8, 12)


# --------------------------------------------------------------------------- the subsample

def test_the_subsample_recomputes_the_capture_rates_for_the_view_it_produces(tmp_path) -> None:
    """Rule 17 in the other direction: a subsampled view reweighted by the FULL tree's capture
    rates describes a tree that is no longer there. The weights must fall out of the kept counts
    against the manifest's own denominators."""
    td = _plant(tmp_path, quota_w=30, quota_l=40)
    arr, meta = CM.extract_cycle(td)
    sub, smeta = QM.subsample(arr, meta, (8, 12, 0), seed=0)
    for opp, rec in smeta["opponents"].items():
        kept_w = int(np.unique(sub["battle"][(sub["opponent"] == opp)
                                             & (sub["y"] == 1.0)]).size)
        assert kept_w <= 8
        assert rec["capture_rate_win"] == pytest.approx(kept_w / rec["battles_won"])
        w = sub["w"][(sub["opponent"] == opp) & (sub["y"] == 1.0)]
        if w.size:
            assert w[0] == pytest.approx(1.0 / rec["capture_rate_win"])
    assert smeta["n_battles"] < meta["n_battles"]
    assert smeta["subsample"]["caps"] == [8, 12, 0]


def test_the_subsample_is_seeded_reproducible_and_seed_dependent(tmp_path) -> None:
    arr, meta = CM.extract_cycle(_plant(tmp_path, quota_w=30, quota_l=40))
    a1, _ = QM.subsample(arr, meta, (8, 12, 0), seed=3)
    a2, _ = QM.subsample(arr, meta, (8, 12, 0), seed=3)
    b1, _ = QM.subsample(arr, meta, (8, 12, 0), seed=4)
    assert np.array_equal(np.unique(a1["battle"]), np.unique(a2["battle"]))
    assert not np.array_equal(np.unique(a1["battle"]), np.unique(b1["battle"]))


def test_nothing_is_written_anywhere_by_a_subsample(tmp_path) -> None:
    """The 2026-09-09 measurement materialised SYMLINK TREES; this does not, and the run archive
    is opened read-only. A subsample that touched the tree it reads would be a write under
    models/ in production."""
    td = _plant(tmp_path, quota_w=30, quota_l=40)
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    arr, meta = CM.extract_cycle(td)
    for s in range(5):
        QM.subsample(arr, meta, (8, 12, 0), seed=s)
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before


def test_the_decoder_matched_rung_gives_the_richer_side_the_poorer_sides_decoder_frame(
        tmp_path) -> None:
    """⚠️ Battle-matching is not decoder-matching. ``MIN_TEAM_BATTLES`` makes the fitted frame a
    nonlinear function of team diversity, so equal battle counts can leave the richer side's
    decoder with FEWER battles than the poorer side's."""
    poor_td = _plant(tmp_path / "p", quota_w=8, quota_l=12, seed=1, n_teams=6)
    rich_td = _plant(tmp_path / "r", quota_w=30, quota_l=40, seed=2, n_teams=24)
    poor_arr, _ = CM.extract_cycle(poor_td)
    rich_arr, rich_meta = CM.extract_cycle(rich_td)
    target = QM.decoder_battles(poor_arr)
    at_battle_match = QM.decoder_battles(QM.subsample(rich_arr, rich_meta, (8, 12, 0), seed=0)[0])
    caps, got = QM.decoder_matched_caps(rich_arr, rich_meta, (8, 12, 0), target)
    assert caps[0] >= 8 and caps[1] >= 12
    assert abs(got - target) <= abs(at_battle_match - target)


# --------------------------------------------------------------------------- the artefact

def _delta_rows(arm_block, ctl_block, qm) -> Dict[str, dict]:
    """``compute_deltas``' conditioning rows, from two conditioning blocks."""
    def side(blk):
        return {"conditioning": {"points": blk["points"]}, "_cond_draws": blk["_draws"],
                "_gate_draws": {}, "_identity_draws": {},
                "_gate_points": {"strata": {}},
                "_identity_points": {"strata": {}, "murphy": {"ci": {}}, "turn": {}}}
    rows = CR.compute_deltas(side(arm_block), side(ctl_block), None, seed=0, qm=qm)
    return {r["key"]: r for r in rows}


@pytest.fixture(scope="module")
def artefact(tmp_path_factory):
    """The planted artefact: ONE critic, TWO quotas.

    Both trees are drawn from the same generative process — the same per-opponent win rates and the
    same team-signal strength — so any difference in a decoder-based row between them is bought by
    frame SIZE and nothing else. That is the null the tool must restore.
    """
    root = tmp_path_factory.mktemp("artefact")
    rich_root, poor_root = root / "rich", root / "poor"
    rich_td = _plant(rich_root, quota_w=40, quota_l=40, n_battles=120, n_teams=10,
                     seed=11, team_signal=1.0)
    poor_td = _plant(poor_root, quota_w=5, quota_l=8, n_battles=120, n_teams=10,
                     seed=12, team_signal=1.0)
    return {"rich_root": rich_root, "poor_root": poor_root,
            "rich_td": rich_td, "poor_td": poor_td,
            "rich": _block(rich_root, rich_td), "poor": _block(poor_root, poor_td)}


def _qm(artefact, *, enabled=True, seeds=9, boot=400):
    a = QM.realized_profile("rich", STEP, artefact["rich_td"])
    c = QM.realized_profile("poor", STEP, artefact["poor_td"])
    if not enabled:
        return {"plan": QM.plan(a, c), "rungs": {}, "enabled": False}
    frames = {"arm": CM.extract_cycle(artefact["rich_td"]),
              "control": CM.extract_cycle(artefact["poor_td"])}
    return QM.match(arm_profile=a, control_profile=c, load_frame=lambda s: frames[s],
                    run_dirs={"arm": str(artefact["rich_root"]),
                              "control": str(artefact["poor_root"])},
                    steps={"arm": STEP, "control": STEP},
                    seeds=seeds, boot=boot, block_seed=0)


def test_the_planted_artefact_is_DETECTED_unmatched_and_NOT_DETECTED_matched(artefact) -> None:
    """🚨 THE REGRESSION. One critic, two quotas: unmatched, the extra frame alone buys the richer
    side a DETECTED own-team R²; matched, it does not. A revert of the fix fails HERE."""
    key = "cond.own_team_r2.t1"
    plain = _delta_rows(artefact["rich"], artefact["poor"], None)[key]
    assert plain["label"] == "DETECTED", plain
    assert plain["delta"] > 0

    matched = _delta_rows(artefact["rich"], artefact["poor"], _qm(artefact))[key]
    assert matched["quota_match"]["status"] == "MATCHED"
    assert matched["label"] == "NOT DETECTED", matched
    assert abs(matched["delta"]) < abs(plain["delta"])
    # the as-traced number rides along, and is never labelled
    assert matched["quota_match"]["unmatched"]["delta"] == pytest.approx(plain["delta"])
    assert "label" not in matched["quota_match"]["unmatched"]


def test_the_matched_row_reports_the_across_seed_spread_and_the_median_seeds_own_interval(
        artefact) -> None:
    qm = _qm(artefact)
    row = _delta_rows(artefact["rich"], artefact["poor"], qm)["cond.own_team_r2.t1"]
    v = row["quota_match"]["variants"]["battle"]
    side = v["sides"]["arm"]
    lo, hi = side["spread"]
    assert lo <= v["arm"] <= hi                       # the point is inside its own spread
    assert side["n_seeds"] == 9
    assert 0 <= side["median_seed"] < 9
    assert row["ci"][0] < row["delta"] < row["ci"][1]
    assert "decoder" in row["quota_match"]["variants"]


def test_the_richer_sides_matched_frame_is_the_size_of_the_poorer_sides(artefact) -> None:
    qm = _qm(artefact)
    battle = qm["rungs"]["arm"]["battle"]
    poor_arr, _ = CM.extract_cycle(artefact["poor_td"])
    poor_battles = int(np.unique(poor_arr["battle"]).size)
    assert abs(battle["median_battles"] - poor_battles) <= 0.15 * poor_battles


def test_the_opt_out_prints_a_marker_and_never_a_label(artefact) -> None:
    """🚨 On an unequal frame neither DETECTED nor NOT DETECTED is a claim this report may make."""
    rows = _delta_rows(artefact["rich"], artefact["poor"], _qm(artefact, enabled=False))
    for key in CM.FRAME_SENSITIVE_KEYS:
        r = rows[key]
        assert r["label"] == QM.UNMATCHED_LABEL, key
        assert r["clears_zero"] is False
        assert r["quota_match"]["status"] == "UNMATCHED"
        assert "--no-quota-match" in r["qualifier"]
    for key in (k for k in CM.METER_KEYS if k not in CM.FRAME_SENSITIVE_KEYS):
        if key in rows:
            assert rows[key]["label"] in ("DETECTED", "NOT DETECTED", "WITHIN FLOOR")


def test_only_the_frame_sensitive_rows_are_touched(artefact) -> None:
    """The noise-corrected spreads, the spread delta and the Elo slope are weighted statistics
    whose expectation does not move with frame size — matching them would cost power and remove no
    bias, so they must come out of a matched read bit-identical to an unmatched one."""
    plain = _delta_rows(artefact["rich"], artefact["poor"], None)
    matched = _delta_rows(artefact["rich"], artefact["poor"], _qm(artefact))
    for key in CM.METER_KEYS:
        if key in CM.FRAME_SENSITIVE_KEYS or key not in plain:
            continue
        assert {k: v for k, v in plain[key].items() if k != "quota_match"} == \
               {k: v for k, v in matched[key].items() if k != "quota_match"}, key


# --------------------------------------------------------------------------- the symmetric pair

def test_a_symmetric_pair_produces_rows_identical_to_matching_disabled(tmp_path) -> None:
    """🚨 BIT-FOR-BIT. Equal realized caps ⇒ nothing is subsampled, and every delta row is exactly
    what the tool produced before quota matching existed. A fix that moves a number it was not
    meant to move is not a fix."""
    a_root, c_root = tmp_path / "a", tmp_path / "c"
    a_td = _plant(a_root, quota_w=8, quota_l=12, seed=21, team_signal=1.0)
    c_td = _plant(c_root, quota_w=8, quota_l=12, seed=22, team_signal=1.0)
    a_blk, c_blk = _block(a_root, a_td), _block(c_root, c_td)
    pa, pc = QM.realized_profile("a", STEP, a_td), QM.realized_profile("c", STEP, c_td)
    qm = QM.match(arm_profile=pa, control_profile=pc,
                  load_frame=lambda s: (_ for _ in ()).throw(
                      AssertionError("a symmetric pair must not extract a frame")),
                  run_dirs={"arm": str(a_root), "control": str(c_root)},
                  steps={"arm": STEP, "control": STEP}, seeds=9, boot=200, block_seed=0)
    assert qm["plan"]["needed"] is False and qm["rungs"] == {}
    off = _delta_rows(a_blk, c_blk, None)
    on = _delta_rows(a_blk, c_blk, qm)
    assert set(off) == set(on)
    for key in off:
        assert off[key] == {k: v for k, v in on[key].items() if k != "quota_match"}, key
        if key in CM.FRAME_SENSITIVE_KEYS:
            assert on[key]["quota_match"]["status"] == "SYMMETRIC"


# --------------------------------------------------------------------------- the declaration

def test_every_meter_declares_its_frame_sensitivity_with_a_reason() -> None:
    """The dispatch is on a DECLARED flag, never on a key's name — a new meter that forgets to
    declare cannot exist, and a renamed one cannot quietly fall out of the matched set."""
    assert len(CM.METER_SPECS) == len(CM.METERS) == len(CM.METER_KEYS)
    for m in CM.METER_SPECS:
        assert isinstance(m.frame_sensitive, bool)
        # a reason, or an explicit cross-reference to the sibling row that carries it — never
        # blank, because the flag is the thing a future meter has to think about
        assert len(m.why) > 40 or any(f"`{o.key}`" in m.why for o in CM.METER_SPECS
                                      if o.key != m.key), m.key
    assert set(CM.FRAME_SENSITIVE_KEYS) == {
        "cond.spread_ratio_raw.t1_3", "cond.spread_ratio_raw.all",
        "cond.own_team_r2.t1", "cond.own_team_r2.all", "cond.opp_class_auc.t1"}


def test_every_row_that_goes_through_the_out_of_fold_decoder_is_frame_sensitive() -> None:
    """The three rows :func:`conditioning_block` fits a decoder for are exactly the three fitted
    rows in the spec table — read off the code, so adding a fourth decode without declaring it
    fails here rather than shipping an unmatched fit."""
    src = __import__("inspect").getsource(CM.conditioning_block)
    fitted = [m.key for m in CM.METER_SPECS
              if f'("{m.key}", "r2"' in src or f'("{m.key}", "auc"' in src]
    assert set(fitted) == {"cond.own_team_r2.t1", "cond.own_team_r2.all",
                           "cond.opp_class_auc.t1"}
    assert all(CM.METER_BY_KEY[k].frame_sensitive for k in fitted)


def test_the_legacy_three_tuple_view_still_unpacks() -> None:
    """The committed matched-quota measurement imports ``CM.METERS`` and unpacks three fields; a
    committed measurement must stay reproducible from the artifacts beside it."""
    for key, quantity, stratum in CM.METERS:
        assert isinstance(key, str) and isinstance(quantity, str) and isinstance(stratum, str)


def test_the_default_seed_count_is_odd_and_at_least_twenty() -> None:
    """>= 20 is the standing consequence's own number; ODD makes the reported median an exact
    order statistic, so the point and the interval beside it describe the same draw."""
    assert QM.DEFAULT_SEEDS >= 20 and QM.DEFAULT_SEEDS % 2 == 1


def test_the_report_renders_the_profiles_and_the_matched_detail(artefact) -> None:
    from main.ops import critic_read_render as RENDER

    qm = _qm(artefact)
    rows = CR.compute_deltas(
        {"conditioning": {"points": artefact["rich"]["points"]},
         "_cond_draws": artefact["rich"]["_draws"], "_gate_draws": {}, "_identity_draws": {},
         "_gate_points": {"strata": {}},
         "_identity_points": {"strata": {}, "murphy": {"ci": {}}, "turn": {}}},
        {"conditioning": {"points": artefact["poor"]["points"]},
         "_cond_draws": artefact["poor"]["_draws"], "_gate_draws": {}, "_identity_draws": {},
         "_gate_points": {"strata": {}},
         "_identity_points": {"strata": {}, "murphy": {"ci": {}}, "turn": {}}},
        None, seed=0, qm=qm)
    doc = {"deltas": rows, "quota_match": QM.serialisable(qm)}
    head = RENDER._profiles_block(doc)
    assert "REALIZED capture profile" in head and "UNEQUAL FRAMES — MATCHED" in head
    detail = RENDER._matched_detail(doc)
    assert "MATCHED · battle" in detail and "MATCHED · decoder" in detail
    assert "UNMATCHED (as traced)" in detail and "no label — not a reading" in detail
    assert QM.UNMATCHED_LABEL not in detail          # a MATCHED read never prints the marker


def test_the_ledger_line_says_which_frame_it_quotes(artefact) -> None:
    """🚨 A matched number and an as-traced one are different measurements. The 2026-09-09
    retraction is what happens when a quoted line does not say which one it is."""
    from main.ops import critic_read_render as RENDER

    rows = _delta_rows(artefact["rich"], artefact["poor"], _qm(artefact))
    doc = {"deltas": list(rows.values()),
           "arm": {"run": "rich", "step": STEP}, "control": {"run": "poor", "step": STEP}}
    assert "[QUOTA-MATCHED]" in RENDER.ledger_line(doc)
    rows_off = _delta_rows(artefact["rich"], artefact["poor"], _qm(artefact, enabled=False))
    doc["deltas"] = list(rows_off.values())
    assert "[FRAMES UNMATCHED]" in RENDER.ledger_line(doc)


def test_a_pair_matched_control_vs_control_is_the_same_operation(tmp_path) -> None:
    """The replicate FLOOR is a control-vs-control read, and it carries the same asymmetry with
    the same sign. Nothing in the matching depends on a side being called the arm."""
    rich_root, poor_root = tmp_path / "b", tmp_path / "a"
    rich_td = _plant(rich_root, quota_w=40, quota_l=40, n_battles=120, seed=31, team_signal=1.0)
    poor_td = _plant(poor_root, quota_w=5, quota_l=8, n_battles=120, seed=32, team_signal=1.0)
    frames = {"arm": CM.extract_cycle(poor_td), "control": CM.extract_cycle(rich_td)}
    qm = QM.match(arm_profile=QM.realized_profile("a", STEP, poor_td),
                  control_profile=QM.realized_profile("b", STEP, rich_td),
                  load_frame=lambda s: frames[s],
                  run_dirs={"arm": str(poor_root), "control": str(rich_root)},
                  steps={"arm": STEP, "control": STEP}, seeds=9, boot=300, block_seed=0)
    assert qm["plan"]["sides"] == ["control"]
    rows = _delta_rows(_block(poor_root, poor_td), _block(rich_root, rich_td), qm)
    row = rows["cond.own_team_r2.t1"]
    assert row["quota_match"]["status"] == "MATCHED"
    assert "control" in row["quota_match"]["variants"]["battle"]["sides"]
    assert row["label"] in ("DETECTED", "NOT DETECTED")


def test_a_floor_is_applied_to_the_MATCHED_delta_not_the_as_traced_one(artefact, tmp_path) -> None:
    """A replicate floor is a control-vs-control magnitude, and it must bite the number the row
    actually reports. A floor applied to an as-traced delta while the label was decided on a
    matched one would be two different measurements wearing one bar."""
    key = "cond.own_team_r2.t1"
    qm = _qm(artefact)
    rows = _delta_rows(artefact["rich"], artefact["poor"], qm)
    matched_delta = rows[key]["delta"]
    unmatched_delta = rows[key]["quota_match"]["unmatched"]["delta"]
    assert abs(unmatched_delta) > abs(matched_delta)

    # a floor that swallows the MATCHED delta but not the as-traced one
    floor = (abs(matched_delta) + abs(unmatched_delta)) / 2
    fp = tmp_path / "floor.json"
    fp.write_text(json.dumps({"floors": {key: floor}, "provenance": "planted"}))
    floors = CR.load_floors(str(fp))["floors"]
    assert floors[key] == pytest.approx(floor)

    with_floor = CR.compute_deltas(
        {"conditioning": {"points": artefact["rich"]["points"]},
         "_cond_draws": artefact["rich"]["_draws"], "_gate_draws": {}, "_identity_draws": {},
         "_gate_points": {"strata": {}},
         "_identity_points": {"strata": {}, "murphy": {"ci": {}}, "turn": {}}},
        {"conditioning": {"points": artefact["poor"]["points"]},
         "_cond_draws": artefact["poor"]["_draws"], "_gate_draws": {}, "_identity_draws": {},
         "_gate_points": {"strata": {}},
         "_identity_points": {"strata": {}, "murphy": {"ci": {}}, "turn": {}}},
        floors, seed=0, qm=qm)
    row = {r["key"]: r for r in with_floor}[key]
    assert row["floor"] == pytest.approx(floor)
    assert row["label"] == "WITHIN FLOOR"          # decided on the MATCHED delta
    assert row["delta"] == pytest.approx(matched_delta)
