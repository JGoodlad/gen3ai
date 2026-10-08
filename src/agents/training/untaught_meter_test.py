"""Unit tests for the untaught meter's PURE half — no models, no battles, no torch.

The aggregation is what every fold verdict is read off, so each property it must have gets its own
named test that FAILS on revert: the bootstrap is PAIRED (one shared index set), a floor is a
max-pairwise MAGNITUDE, ``WITHIN FLOOR`` outranks a CI that excludes zero, a run whose timeouts
clear 25% reports no verdict at all, and the TRANSPORT boundary (P2: the Rust eval core vs the
pre-boundary poke-env bridge) is refused by every reader that puts two reads side by side.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from agents.training import untaught_meter as um


def _cells(rates_by_label, team_keys, games=100, attempted=None):
    """Synthetic cells with exact win counts — ``rate * games`` must be an integer."""
    out = {}
    for lab, rates in rates_by_label.items():
        out[lab] = {}
        for k, r in zip(team_keys, rates):
            wins = int(round(r * games))
            out[lab][k] = um.Cell(wins=wins, ties=0, losses=games - wins, finished=games,
                                  attempted=attempted or games)
    return out


TEAMS = [f"U_{i}" for i in range(8)]


# ---------------------------------------------------------------------------------------------
# The bootstrap is PAIRED
# ---------------------------------------------------------------------------------------------

def test_the_bootstrap_index_set_is_shared_so_a_ref_vs_ref_delta_is_paired():
    """A and B differing by a CONSTANT on every team must give a ZERO-WIDTH delta interval.

    That is only true if both arms are resampled with the SAME index set. Bootstrapping each arm
    independently and differencing the means would give a wide interval here — which is exactly the
    unpaired reading this meter exists to avoid.
    """
    base = [0.50, 0.55, 0.60, 0.45, 0.62, 0.58, 0.41, 0.53]
    arm = [b + 0.10 for b in base]
    cells = _cells({"ARM": arm, "BASE": base}, TEAMS)
    res = um.aggregate(cells, TEAMS, ref_labels=["ARM"], baseline_label="BASE", draws=2000)
    d = res["contrasts"][0]["vs_baseline"]
    assert d["delta_pp"] == pytest.approx(10.0, abs=1e-9)
    assert d["ci95_pp"][0] == pytest.approx(10.0, abs=1e-9)
    assert d["ci95_pp"][1] == pytest.approx(10.0, abs=1e-9)


def test_an_unpaired_bootstrap_would_be_wide_here_so_the_test_above_can_fail():
    """The negative control for the pairing test: independent resamples DO spread."""
    base = np.array([0.50, 0.55, 0.60, 0.45, 0.62, 0.58, 0.41, 0.53])
    arm = base + 0.10
    ia = um.bootstrap_index(8, 2000, 1)
    ib = um.bootstrap_index(8, 2000, 2)          # a DIFFERENT index set = unpaired
    spread = arm[ia].mean(axis=1) - base[ib].mean(axis=1)
    assert spread.std() > 0.01


def test_the_same_index_set_serves_every_contrast_in_one_call():
    """Two refs against one baseline: their delta-of-deltas must equal the direct A−B delta."""
    cells = _cells({"A": [0.6] * 4 + [0.7] * 4, "B": [0.5] * 8,
                    "BASE": [0.55, 0.45, 0.6, 0.5, 0.52, 0.61, 0.49, 0.58]}, TEAMS)
    res = um.aggregate(cells, TEAMS, ref_labels=["A", "B"], baseline_label="BASE", draws=1000)
    a = res["contrasts"][0]["vs_baseline"]["delta_pp"]
    b = res["contrasts"][1]["vs_baseline"]["delta_pp"]
    direct = (np.array([0.6] * 4 + [0.7] * 4) - 0.5).mean() * 100
    assert a - b == pytest.approx(direct, abs=1e-9)


# ---------------------------------------------------------------------------------------------
# Floors
# ---------------------------------------------------------------------------------------------

def test_the_control_floor_is_the_max_pairwise_magnitude_not_the_mean():
    arms = [np.zeros(8), np.full(8, 0.02), np.full(8, 0.05)]
    floor, pairs = um.replicate_floor(arms)
    assert floor == pytest.approx(5.0)                    # |0.00 − 0.05| = 5pp, the LARGEST
    assert len(pairs) == 3
    assert sorted(round(p["abs"], 6) for p in pairs) == [2.0, 3.0, 5.0]


def test_one_control_arm_yields_no_floor_rather_than_a_fabricated_zero():
    assert um.replicate_floor([np.zeros(8)]) is None
    cells = _cells({"A": [0.6] * 8, "BASE": [0.5] * 8, "C": [0.55] * 8}, TEAMS)
    res = um.aggregate(cells, TEAMS, ref_labels=["A"], baseline_label="BASE",
                       control_labels=["C"], draws=500)
    assert res["control"]["replicate_floor_pp"] is None
    assert "NO floor" in res["control"]["floor_note"]
    # …and with no floor the control column still reads, it just cannot say WITHIN FLOOR.
    assert res["contrasts"][0]["vs_control"]["verdict"] in ("SIGNIFICANT", "NOT DETECTED")


def test_a_pooled_control_is_the_equal_weight_mean_of_its_arms():
    cells = _cells({"A": [0.6] * 8, "BASE": [0.5] * 8,
                    "C1": [0.52] * 8, "C2": [0.58] * 8}, TEAMS)
    res = um.aggregate(cells, TEAMS, ref_labels=["A"], baseline_label="BASE",
                       control_labels=["C1", "C2"], draws=500)
    assert res["control"]["pooled_cluster_mean_pp"] == pytest.approx(55.0)
    assert res["control"]["replicate_floor_pp"] == pytest.approx(6.0)
    assert res["contrasts"][0]["vs_control"]["delta_pp"] == pytest.approx(5.0)
    # 5pp against a 6pp replicate floor — the continuation column is WITHIN FLOOR even though the
    # frozen-baseline column reads +10pp.
    assert res["contrasts"][0]["vs_control"]["verdict"] == "WITHIN FLOOR"
    assert res["contrasts"][0]["vs_baseline"]["delta_pp"] == pytest.approx(10.0)


# ---------------------------------------------------------------------------------------------
# Verdict vocabulary
# ---------------------------------------------------------------------------------------------

def test_within_floor_outranks_a_ci_that_excludes_zero():
    """A delta smaller than the floor says the GAMES are consistent, not that the arm differs."""
    assert um.verdict(1.0, 0.5, 1.5, floor=2.0) == "WITHIN FLOOR"
    assert um.verdict(1.0, 0.5, 1.5, floor=None) == "SIGNIFICANT"


def test_not_detected_is_ci_spanning_zero_above_the_floor():
    assert um.verdict(5.0, -1.0, 11.0, floor=2.0) == "NOT DETECTED"
    assert um.verdict(-5.0, -11.0, 1.0, floor=2.0) == "NOT DETECTED"


def test_significant_needs_both_halves():
    assert um.verdict(5.0, 1.0, 9.0, floor=2.0) == "SIGNIFICANT"
    assert um.verdict(-5.0, -9.0, -1.0, floor=2.0) == "SIGNIFICANT"


# ---------------------------------------------------------------------------------------------
# Timeouts
# ---------------------------------------------------------------------------------------------

def test_timeouts_are_their_own_bucket_and_never_a_loss():
    c = um.Cell(wins=3, ties=0, losses=2, finished=5, attempted=8)
    assert c.timeouts == 3
    assert c.win_rate == pytest.approx(0.6)          # 3/5 FINISHED, not 3/8


def test_over_25pc_timeouts_reports_inconclusive_and_no_verdict():
    cells = _cells({"A": [0.6] * 8, "BASE": [0.5] * 8}, TEAMS, games=70, attempted=100)
    res = um.aggregate(cells, TEAMS, ref_labels=["A"], baseline_label="BASE", draws=200)
    assert res["timeouts"]["fraction"] == pytest.approx(0.30)
    assert res["timeouts"]["inconclusive"] is True
    assert res["contrasts"][0]["vs_baseline"]["verdict"] == "INCONCLUSIVE"


def test_exactly_25pc_timeouts_is_not_inconclusive():
    cells = _cells({"A": [0.6] * 8, "BASE": [0.5] * 8}, TEAMS, games=75, attempted=100)
    res = um.aggregate(cells, TEAMS, ref_labels=["A"], baseline_label="BASE", draws=200)
    assert res["timeouts"]["fraction"] == pytest.approx(0.25)
    assert res["timeouts"]["inconclusive"] is False
    assert res["contrasts"][0]["vs_baseline"]["verdict"] != "INCONCLUSIVE"


# ---------------------------------------------------------------------------------------------
# Levels are CLUSTER means, never game-weighted pools
# ---------------------------------------------------------------------------------------------

def test_the_level_is_the_equal_weight_team_mean_not_the_game_weighted_pool():
    """Measured trap: a game-weighted pool can disagree in SIGN with the clustered mean."""
    cells = {"A": {"U_0": um.Cell(wins=90, losses=10, finished=100, attempted=100),
                   "U_1": um.Cell(wins=0, losses=10, finished=10, attempted=10)}}
    res = um.aggregate(cells, ["U_0", "U_1"], ref_labels=["A"], baseline_label=None, draws=200)
    assert res["levels"]["A"]["cluster_mean_pp"] == pytest.approx(45.0)   # (0.9 + 0.0)/2
    pooled = 90 / 110 * 100
    assert abs(pooled - 45.0) > 30                                        # the pool would say 81.8


# ---------------------------------------------------------------------------------------------
# Concurrency refusal
# ---------------------------------------------------------------------------------------------

def test_concurrency_above_one_is_refused_as_a_RETIRED_knob():
    um.check_concurrency(1)
    with pytest.raises(um.MeterError, match="REFUSING concurrency=3: the knob is RETIRED"):
        um.check_concurrency(3)


def test_the_concurrency_refusal_has_no_override_any_more(monkeypatch):
    """The old `GEN3AI_UNTAUGHT_METER_ALLOW_CONCURRENCY=1` escape is gone with the poke-env path."""
    monkeypatch.setenv("GEN3AI_UNTAUGHT_METER_ALLOW_CONCURRENCY", "1")
    with pytest.raises(um.MeterError):
        um.check_concurrency(3)
    assert not hasattr(um, "ALLOW_CONCURRENCY_ENV")


def test_impl_node_is_refused_with_the_reason():
    um.check_impl("rust")
    with pytest.raises(um.MeterError, match="Rust EVAL CORE"):
        um.check_impl("node")


# ---------------------------------------------------------------------------------------------
# THE TRANSPORT BOUNDARY (poke-env retirement P2)
# ---------------------------------------------------------------------------------------------

def _stamped(cells, transport):
    for t in cells.values():
        for c in t.values():
            c.transport = transport
    return cells


def test_a_cell_without_a_stamp_is_the_pre_boundary_python_bridge_and_round_trips_the_stamp():
    old = um.Cell(wins=1, finished=2, attempted=2)
    assert old.transport_name == um.TRANSPORT_LEGACY == "python_bridge"
    assert "transport" not in old.to_json()                       # a pre-boundary cell's JSON is unchanged
    new = um.Cell(wins=1, ties=1, finished=2, attempted=2, transport=um.TRANSPORT_RUST, turn_limit_draws=1)
    back = um.cell_from_json(new.to_json())
    assert (back.transport, back.turn_limit_draws) == ("rust_eval", 1)
    assert um.cell_from_json(old.to_json(), "rust_eval").transport == "rust_eval"   # the artifact-level stamp


def test_aggregate_REFUSES_a_mix_of_transports_and_reads_it_only_with_consent():
    cells = _cells({"A": [0.6] * 8, "BASE": [0.5] * 8}, TEAMS)
    _stamped({"A": cells["A"]}, um.TRANSPORT_RUST)                # A on the Rust core, BASE pre-boundary
    with pytest.raises(um.TransportMixError, match="DIFFERENT transports"):
        um.aggregate(cells, TEAMS, ref_labels=["A"], baseline_label="BASE", draws=200)
    res = um.aggregate(cells, TEAMS, ref_labels=["A"], baseline_label="BASE", draws=200,
                       allow_transport_mix=True)
    assert res["transport"] == "MIXED:python_bridge+rust_eval"
    assert "TRANSPORT MIX" in um.render_markdown({"result": res})


def test_aggregate_stamps_the_one_transport_of_an_unmixed_read():
    cells = _stamped(_cells({"A": [0.6] * 8, "BASE": [0.5] * 8}, TEAMS), um.TRANSPORT_RUST)
    res = um.aggregate(cells, TEAMS, ref_labels=["A"], baseline_label="BASE", draws=200)
    assert res["transport"] == "rust_eval"
    legacy = um.aggregate(_cells({"A": [0.6] * 8}, TEAMS), TEAMS, ref_labels=["A"], baseline_label=None, draws=200)
    assert legacy["transport"] == "python_bridge"


def test_merge_cells_refuses_a_shard_set_that_mixes_transports_and_reads_a_shards_meta_stamp():
    rust_shard = {um.SHARD_META: {"transport": "rust_eval"},
                  "A": {"U_0": um.Cell(wins=1, finished=2, attempted=2).to_json()}}
    merged = um.merge_cells([rust_shard])
    assert merged["A"]["U_0"].transport == "rust_eval" and um.SHARD_META not in merged
    legacy_shard = {"A": {"U_1": um.Cell(wins=2, finished=2, attempted=2).to_json()}}
    with pytest.raises(um.TransportMixError):
        um.merge_cells([rust_shard, legacy_shard])


def test_a_rows_artifact_takes_its_meta_transport_stamp(tmp_path):
    art = tmp_path / "rust.json"
    art.write_text(json.dumps({"_meta": {"transport": "rust_eval"}, "U_a": {"wins": 5, "games": 10}}))
    assert um.cells_from_rows_artifact(str(art))["U_a"].transport == "rust_eval"
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"_meta": {}, "U_a": {"wins": 5, "games": 10}}))
    assert um.cells_from_rows_artifact(str(old))["U_a"].transport_name == "python_bridge"


def test_the_rust_cycle_seed_depends_on_the_seed_and_the_team_never_the_ref():
    """CRN: one cell's cycle seed is a function of (--seed, team index) ONLY — no ref goes in."""
    import inspect

    from agents.training import untaught_rust as UR

    assert list(inspect.signature(UR.cycle_seed).parameters) == ["seed", "team_index"]
    seeds = {UR.cycle_seed(s, t) for s in (0, 1) for t in range(8)}
    assert len(seeds) == 16 and UR.cycle_seed(0, 3) == UR.cycle_seed(0, 3)


# ---------------------------------------------------------------------------------------------
# Ingesting committed artifacts
# ---------------------------------------------------------------------------------------------

def test_the_pooled_summary_row_is_not_counted_as_a_team(tmp_path):
    art = tmp_path / "untaught_X_end.json"
    art.write_text(json.dumps({
        "_meta": {"tag": "X"},
        "U_a": {"wins": 5, "games": 10}, "U_b": {"wins": 6, "games": 10},
        "POOLED": {"wins": 11, "games": 20}}))
    cells = um.cells_from_rows_artifact(str(art))
    assert sorted(cells) == ["U_a", "U_b"]         # 2 clusters, not 3


def test_an_artifact_with_no_team_rows_refuses(tmp_path):
    art = tmp_path / "empty.json"
    art.write_text(json.dumps({"_meta": {}}))
    with pytest.raises(um.MeterError, match="no per-team rows"):
        um.cells_from_rows_artifact(str(art))


# ---------------------------------------------------------------------------------------------
# Team manifests
# ---------------------------------------------------------------------------------------------

def _manifest(tmp_path, rels):
    p = tmp_path / "teams.json"
    p.write_text(json.dumps({"untaught": rels}))
    return str(p)


def test_a_manifest_preserves_ORDER_because_the_order_is_the_seed(tmp_path):
    files = []
    for name in ("zzz.txt", "aaa.txt", "mmm.txt"):
        f = tmp_path / name
        f.write_text(f"team {name}\n")
        files.append(str(f))
    slices = um.load_team_manifest(_manifest(tmp_path, files))
    assert [s.index for s in slices] == [0, 1, 2]
    assert [os.path.basename(s.path) for s in slices] == ["zzz.txt", "aaa.txt", "mmm.txt"]


def test_a_missing_team_file_refuses_and_names_it(tmp_path):
    good = tmp_path / "a.txt"
    good.write_text("x\n")
    with pytest.raises(um.MeterError, match="nope.txt"):
        um.load_team_manifest(_manifest(tmp_path, [str(good), str(tmp_path / "nope.txt")]))


def test_the_two_team_sha_conventions_are_both_recorded_and_differ_on_a_trailing_newline(tmp_path):
    f = tmp_path / "t.txt"
    f.write_text("Snorlax @ Leftovers\n")            # a trailing newline: pin_sha != team_sha
    s = um.load_team_manifest(_manifest(tmp_path, [str(f)]))[0]
    assert len(s.pin_sha) == 10 and len(s.team_sha) == 10
    assert s.pin_sha != s.team_sha
    assert s.sha1.startswith(s.pin_sha)


def test_the_committed_untaught_manifest_is_the_eight_teams_in_seed_order():
    slices = um.load_team_manifest(str(um.DEFAULT_TEAMS_MANIFEST))
    assert len(slices) == 8
    assert slices[0].key == "U_61590463"
    assert slices[-1].key == "U_dbf81d8e"


def test_the_committed_taught_manifest_is_sixteen_and_disjoint_from_the_untaught_eight():
    taught = um.load_team_manifest(str(um.DEFAULT_TAUGHT_MANIFEST), prefix="T")
    untaught = um.load_team_manifest(str(um.DEFAULT_TEAMS_MANIFEST))
    assert len(taught) == 16
    assert not {t.sha1 for t in taught} & {u.sha1 for u in untaught}


# ---------------------------------------------------------------------------------------------
# Merging shards
# ---------------------------------------------------------------------------------------------

def test_merging_shards_reassembles_the_grid():
    a = {"A": {"U_0": um.Cell(wins=1, finished=2, attempted=2).to_json()}}
    b = {"A": {"U_1": um.Cell(wins=2, finished=2, attempted=2).to_json()}}
    merged = um.merge_cells([a, b])
    assert sorted(merged["A"]) == ["U_0", "U_1"]


def test_an_overlapping_shard_split_is_a_refusal_not_a_silent_overwrite():
    a = {"A": {"U_0": um.Cell(wins=1, finished=2, attempted=2).to_json()}}
    with pytest.raises(um.MeterError, match="shard overlap"):
        um.merge_cells([a, a])


# ---------------------------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------------------------

def test_a_report_with_no_control_carries_the_rebasing_warning():
    cells = _cells({"A": [0.6] * 8, "BASE": [0.5] * 8}, TEAMS)
    res = um.aggregate(cells, TEAMS, ref_labels=["A"], baseline_label="BASE", draws=200)
    md = um.render_markdown({"result": res})
    assert "No continuation control" in md
    assert "+3.45pp" in md                    # the measured size of what is being ignored


def test_a_report_with_a_control_renders_both_delta_columns():
    cells = _cells({"A": [0.6] * 8, "BASE": [0.5] * 8, "C1": [0.52] * 8, "C2": [0.58] * 8}, TEAMS)
    res = um.aggregate(cells, TEAMS, ref_labels=["A"], baseline_label="BASE",
                       control_labels=["C1", "C2"], draws=200)
    md = um.render_markdown({"result": res})
    assert "Δ vs continuation control" in md
    assert "No continuation control" not in md
