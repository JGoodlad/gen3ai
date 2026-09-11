"""``main.ops.critic_read`` — the DELTA arithmetic, the REFUSALS and the CACHE, on synthetic input.

The tool composes two expensive readers (``cf_audit`` and ``main.critic_gate``) over a run
archive, and ``models/`` is not committed — so nothing here reads it. What IS tested is
everything that could quietly produce a wrong sentence:

1. **The three-way LABEL.** Each of DETECTED / WITHIN FLOOR / NOT DETECTED is planted, in both
   directions, with and without a floor. The label is the thing a ledger line quotes, and it is
   the one place where "the CI clears zero" and "the CI clears the FLOOR" must not be conflated.
2. **The delta is the difference of INDEPENDENT bootstraps.** An arm read against ITSELF must
   produce exactly 0.0 with an interval that contains 0 — the self-consistency plant the tool
   ships with — and two well-separated samples must produce an interval that excludes it.
3. **Every REFUSAL.** No manifest, a manifest whose cycle has not collected, npz with no
   ``win_probs``, an anchor rate under the label-trust gate, a draw/timeout share over the cap,
   and a malformed floor file. Each must exit 2 naming the cause; a partial table would read as
   a complete one.
4. **The CACHE reuses on a matching fingerprint and only then.** A reused control that is
   silently stale would compare an arm against a different cycle than the report names.

Pure unit, unmarked (runs in every tier): no subprocess, no models/, no traces.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from main.ops import critic_read as CR
from main.ops import critic_readouts as R


# --------------------------------------------------------------------------- the label

def test_detected_needs_the_ci_to_clear_the_floor_not_only_zero() -> None:
    """A CI that clears ZERO but sits inside the floor is NOT DETECTED — the registration's
    whole point. Planted: delta +0.02, CI [+0.01, +0.03], floor 0.05."""
    v = R.label_delta(0.02, [0.01, 0.03], 0.05)
    assert v["label"] == "WITHIN FLOOR"
    assert v["clears_zero"] is True and v["clears_floor"] is False


def test_detected_when_the_ci_clears_both() -> None:
    v = R.label_delta(0.20, [0.12, 0.28], 0.05)
    assert v["label"] == "DETECTED" and v["clears_floor"] is True


def test_detected_in_the_negative_direction_too() -> None:
    v = R.label_delta(-0.20, [-0.28, -0.12], 0.05)
    assert v["label"] == "DETECTED"


def test_within_floor_when_the_point_sits_inside_it() -> None:
    v = R.label_delta(0.01, [-0.04, 0.06], 0.05)
    assert v["label"] == "WITHIN FLOOR"


def test_not_detected_when_the_ci_covers_zero_and_the_point_is_outside_the_floor() -> None:
    v = R.label_delta(0.09, [-0.02, 0.20], 0.05)
    assert v["label"] == "NOT DETECTED" and v["qualifier"] == "CI covers zero"


def test_not_detected_when_the_ci_clears_zero_but_straddles_the_floor() -> None:
    v = R.label_delta(0.09, [0.01, 0.20], 0.05)
    assert v["label"] == "NOT DETECTED"
    assert "does not clear the floor" in v["qualifier"]


def test_no_floor_detects_against_zero_and_says_so() -> None:
    v = R.label_delta(0.20, [0.12, 0.28], None)
    assert v["label"] == "DETECTED" and v["qualifier"] == "vs ZERO — NO FLOOR"
    assert v["floor"] is None and v["clears_floor"] is False


def test_no_interval_is_never_a_detection() -> None:
    v = R.label_delta(9.0, [float("nan"), float("nan")], None)
    assert v["label"] == "NOT DETECTED"


# --------------------------------------------------------------------------- the delta

def _rows(n_battles: int, per: int, bias: float, seed: int) -> list:
    """A planted identity sample that exercises EVERY reported stratum.

    Turns span all three buckets and both opponent classes appear, so a headline keyed on
    `late (turn>=25)` or on `pool` is actually produced; `mc` and the rollout wins vary, so no
    Murphy term degenerates to a NaN that would hide a non-zero delta in the self-plant."""
    rng = np.random.default_rng(seed)
    out = []
    for b in range(n_battles):
        outcome = "win" if b % 2 else "loss"
        cls = "pool" if b % 5 == 0 else "bot"
        for j in range(per):
            v = float(np.clip(0.5 + bias + rng.normal(0, 0.15), 0.02, 0.98))
            wins = int(rng.integers(0, 9))
            out.append({"battle_key": f"opp/b{b}", "turn": 2 + 11 * j, "opponent":
                        "sentinel_0" if cls == "pool" else "heuristic",
                        "opp_class": cls, "outcome": outcome, "v": v, "value": v,
                        "mc": wins / 8.0, "wins": wins, "n": 8})
    return out


def test_an_arm_against_itself_is_exactly_zero_with_an_interval_covering_zero() -> None:
    rows = _rows(40, 5, 0.10, seed=1)
    w = np.ones(len(rows))
    pt, draws = R.boot_draws(rows, R.bias_stat(rows), w=w, draws=500, seed=7)
    d = R.independent_delta(pt, draws, pt, draws, seed=3)
    assert d["delta"] == 0.0
    assert d["ci"][0] < 0.0 < d["ci"][1]
    assert R.label_delta(d["delta"], d["ci"], None)["label"] == "NOT DETECTED"


def test_two_well_separated_samples_give_an_interval_that_excludes_zero() -> None:
    a, b = _rows(40, 5, 0.30, seed=1), _rows(40, 5, -0.30, seed=2)
    pa, da = R.boot_draws(a, R.bias_stat(a), w=np.ones(len(a)), draws=500, seed=7)
    pb, db = R.boot_draws(b, R.bias_stat(b), w=np.ones(len(b)), draws=500, seed=8)
    d = R.independent_delta(pa, da, pb, db, seed=3)
    assert d["delta"] > 0.5
    assert d["ci"][0] > 0.0
    assert R.label_delta(d["delta"], d["ci"], None)["label"] == "DETECTED"


def test_the_delta_interval_is_wider_than_either_arms_own() -> None:
    """Two independent samples' difference cannot be tighter than one arm's interval; a delta CI
    narrower than its parts is the signature of pairing two unrelated bootstraps."""
    a, b = _rows(40, 5, 0.30, seed=1), _rows(40, 5, 0.10, seed=2)
    pa, da = R.boot_draws(a, R.bias_stat(a), w=np.ones(len(a)), draws=800, seed=7)
    pb, db = R.boot_draws(b, R.bias_stat(b), w=np.ones(len(b)), draws=800, seed=8)
    d = R.independent_delta(pa, da, pb, db, seed=3)
    own = min(np.percentile(da, 97.5) - np.percentile(da, 2.5),
              np.percentile(db, 97.5) - np.percentile(db, 2.5))
    assert (d["ci"][1] - d["ci"][0]) > own


def test_a_bootstrap_with_one_battle_yields_no_interval_and_no_detection() -> None:
    rows = _rows(1, 5, 0.2, seed=1)
    pt, draws = R.boot_draws(rows, R.bias_stat(rows), w=np.ones(len(rows)), draws=200, seed=1)
    assert draws.size == 0
    d = R.independent_delta(pt, draws, pt, draws)
    assert R.label_delta(d["delta"], d["ci"], None)["label"] == "NOT DETECTED"


# --------------------------------------------------------------------------- weights

def test_capture_weights_are_one_over_the_recorded_rate(tmp_path) -> None:
    run = tmp_path / "run"
    d = run / "eval_traces" / "step_100"
    d.mkdir(parents=True)
    (d / "eval_manifest.json").write_text(json.dumps({"selection": {"opponents": {
        "heuristic": {"capture_rate_win": 0.1, "capture_rate_loss": 0.5}}}}))
    cap = R.capture_weights(str(run), 100)
    assert cap == {("heuristic", 1): 10.0, ("heuristic", 0): 2.0}


def test_a_row_with_no_recorded_rate_gets_weight_zero_never_one() -> None:
    rows = [{"opponent": "heuristic", "outcome": "win"},
            {"opponent": "mystery", "outcome": "win"}]
    w, covered = R.apply_capture(rows, np.ones(2), {("heuristic", 1): 4.0})
    assert covered == 0.5
    assert w[1] == 0.0 and w[0] > 0.0


def test_no_manifest_selection_means_no_capture_weights(tmp_path) -> None:
    run = tmp_path / "run"
    d = run / "eval_traces" / "step_100"
    d.mkdir(parents=True)
    (d / "eval_manifest.json").write_text(json.dumps({"selection": None}))
    assert R.capture_weights(str(run), 100) is None


def test_pop_weights_have_mean_one() -> None:
    rows = _rows(10, 4, 0.1, seed=4)
    cells = {f"{min(9, int(r['v'] * 10))}|{r['outcome']}|{r['opp_class']}|"
             f"{R.turn_bucket(r['turn'])}": 3 for r in rows}
    w, cover = R.pop_weights(rows, cells)
    assert abs(float(w.mean()) - 1.0) < 1e-9
    assert 0.0 <= cover <= 1.0


# --------------------------------------------------------------------------- statistics

def test_murphy_decomposition_closes_on_planted_data() -> None:
    rows = _rows(30, 5, 0.2, seed=9)
    m = R.murphy(rows, np.ones(len(rows)))
    # the identity is EXACT only when states are grouped by distinct forecast value; binning
    # makes it approximate, and the residual has to stay far below the terms it splits.
    assert abs(m["residual"]) < 1e-3
    assert abs(m["residual"]) < 0.1 * m["reliability"]
    assert 0.0 <= m["resolution_cap_share"] <= 1.0


def test_turn_contrast_is_positive_when_v_rises_and_mc_falls_with_the_clock() -> None:
    rows = []
    for b in range(20):
        for j in range(6):
            rows.append({"battle_key": f"o/b{b}", "turn": 2 + 5 * j, "opponent": "heuristic",
                         "opp_class": "bot", "outcome": "loss",
                         "v": 0.4 + 0.02 * j, "value": 0.0,
                         "mc": 0.6 - 0.05 * j, "wins": 4, "n": 8})
    f = R.turn_contrast_stat(rows)
    assert f(np.arange(len(rows)), np.ones(len(rows))) > 1.5


# --------------------------------------------------------------------------- cycle selection

def _cycle(tmp_path, step: int, *, manifest: "dict | None", npz_winprob: bool = True,
           n_npz: int = 2):
    d = tmp_path / "eval_traces" / f"step_{step}"
    (d / "heuristic").mkdir(parents=True)
    for i in range(n_npz):
        cols = {"values": np.zeros(3), "has_state": np.ones(3, bool)}
        if npz_winprob:
            cols["win_probs"] = np.full(3, 0.5)
        np.savez(d / "heuristic" / f"loss_{i}_states.npz", **cols)
    if manifest is not None:
        (d / "eval_manifest.json").write_text(json.dumps(manifest))
    return d


_OK_MANIFEST = {"saved_at": "x", "selection": {"opponents": {
    "heuristic": {"battles_played": 100, "battles_drawn": 2,
                  "capture_rate_win": 0.1, "capture_rate_loss": 0.6},
    "sentinel_0": {"battles_played": 100, "battles_drawn": 1,
                   "capture_rate_win": 0.2, "capture_rate_loss": 0.7}}}}


def test_cycle_without_a_manifest_is_incomplete(tmp_path) -> None:
    d = _cycle(tmp_path, 100, manifest=None)
    ok, why, _ = CR.cycle_status(str(d))
    assert not ok and "no eval_manifest.json" in why


def test_cycle_whose_manifest_has_not_collected_is_incomplete(tmp_path) -> None:
    d = _cycle(tmp_path, 100, manifest={"selection": None})
    ok, why, _ = CR.cycle_status(str(d))
    assert not ok and "has not COLLECTED" in why


def test_pick_cycle_takes_the_last_complete_one_and_skips_the_uncollected_newest(tmp_path) -> None:
    _cycle(tmp_path, 100, manifest=_OK_MANIFEST)
    _cycle(tmp_path, 200, manifest=_OK_MANIFEST)
    _cycle(tmp_path, 300, manifest={"selection": None})
    got = CR.pick_cycle(tmp_path, on_live="skip-newest", step=None)
    assert got["step"] == 200


def test_pick_cycle_refuses_when_no_cycle_is_complete(tmp_path) -> None:
    _cycle(tmp_path, 100, manifest={"selection": None})
    with pytest.raises(SystemExit) as exc:
        CR.pick_cycle(tmp_path, on_live="skip-newest", step=None)
    assert exc.value.code == 2


def test_pick_cycle_refuses_a_run_with_no_traces_at_all(tmp_path) -> None:
    with pytest.raises(SystemExit) as exc:
        CR.pick_cycle(tmp_path, on_live="skip-newest", step=None)
    assert exc.value.code == 2


def test_draw_share_is_none_when_the_schema_does_not_record_draws() -> None:
    assert CR.draw_share({"selection": {"opponents": {
        "heuristic": {"battles_played": 100}}}}) is None


def test_draw_share_is_the_population_rate() -> None:
    assert CR.draw_share(_OK_MANIFEST) == pytest.approx(3 / 200)


def test_winprob_coverage_counts_the_npz_without_the_channel(tmp_path) -> None:
    d = _cycle(tmp_path, 100, manifest=_OK_MANIFEST, npz_winprob=False, n_npz=3)
    cov = CR.winprob_coverage(str(d))
    assert cov == {"n_npz": 3, "n_with_winprob": 0, "n_missing": 3}
    d2 = _cycle(tmp_path, 200, manifest=_OK_MANIFEST, npz_winprob=True, n_npz=3)
    assert CR.winprob_coverage(str(d2))["n_missing"] == 0


def test_live_pids_never_match_this_process(tmp_path) -> None:
    """The reader must not find ITSELF — a `pgrep -f`-shaped match on your own argv is the
    failure this function was written around."""
    assert CR.live_pids_for_run("definitely_not_a_run_name_12345") == []


# --------------------------------------------------------------------------- floors

def test_a_floor_file_is_read_as_magnitudes(tmp_path) -> None:
    p = tmp_path / "floor.json"
    p.write_text(json.dumps({"floors": {"gate.resolution.bot": -0.012},
                             "provenance": "ctrl replicate pair"}))
    got = CR.load_floors(str(p))
    assert got["floors"] == {"gate.resolution.bot": 0.012}
    assert got["provenance"] == "ctrl replicate pair"


def test_a_bare_mapping_is_accepted_as_a_floor_file(tmp_path) -> None:
    p = tmp_path / "floor.json"
    p.write_text(json.dumps({"gate.skill.bot": 0.03}))
    assert CR.load_floors(str(p))["floors"] == {"gate.skill.bot": 0.03}


def test_a_malformed_floor_file_refuses(tmp_path) -> None:
    p = tmp_path / "floor.json"
    p.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(SystemExit) as exc:
        CR.load_floors(str(p))
    assert exc.value.code == 2


def test_no_floor_file_yields_the_registered_no_floor_note() -> None:
    got = CR.load_floors(None)
    assert got["floors"] is None and "NO REPLICATE FLOOR" in got["provenance"]


# --------------------------------------------------------------------------- the read + cache

class _Args:
    """The CLI namespace `read_run` reads, with the shipped defaults."""
    states, anchors, rollouts, impl, seed = 24, 12, 8, "rust", 0
    anchor_tolerance, bins, boot, gate_boot = 0.90, 10, 60, 40
    deadline_min, nice, no_cache = 0, 0, False
    on_live, max_draw_share, allow_missing_winprob = "skip-newest", 0.25, 0.0
    parent, famine_comparator = "v9_fold_parent", "off"
    allow_gate_refusal = False
    # The synthetic tree has npz but no `_summary.json`, so it carries no CONDITIONING frame at
    # all; the default here keeps the cache/delta tests about the halves they are testing, and
    # the conditioning path has its own tests below and in `conditioning_meters_test.py`.
    no_conditioning, allow_conditioning_refusal = True, False
    cond_boot, cond_ladder = 50, "off"


def _plant_run(tmp_path, name="arm", *, anchor_rate=1.0, manifest=None, npz_winprob=True):
    run = tmp_path / name
    _cycle(run, 100, manifest=manifest if manifest is not None else _OK_MANIFEST,
           npz_winprob=npz_winprob)
    return run


def _install_stubs(monkeypatch, calls, *, anchor_rate=1.0):
    """Stand in for the two SUBPROCESSES and the two heavy readers. Everything the test asserts
    on — the refusals, the cache, the delta — is the tool's own code, untouched."""
    def fake_run(argv, log_path, *, nice, cwd=None):
        calls.append(argv[2] if len(argv) > 2 else "?")
        out = argv[argv.index("--out") + 1] if "--out" in argv else None
        if out:  # cf_audit
            import os
            os.makedirs(os.path.join(out, "cf_labels"), exist_ok=True)
            with open(os.path.join(out, "bias_map.json"), "w") as fh:
                json.dump({"anchor_rate": anchor_rate, "anchors_issued": 12,
                           "anchors_reproduced": int(round(anchor_rate * 12)),
                           "anchor_errors": 0}, fh)
            open(os.path.join(out, "cf_labels", "labels_x_100.jsonl"), "w").close()
        if "--json" in argv:  # critic_gate
            import os
            p = argv[argv.index("--json") + 1]
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as fh:
                json.dump({"calibration": {"verdict": {"G1": False}, "artifact": "baseline",
                                           "checkpoints": []}}, fh)
        return 0

    rows = _rows(24, 4, 0.1, seed=5)
    payload = {"rows": rows, "frame_cells": {
        f"{min(9, int(r['v'] * 10))}|{r['outcome']}|{r['opp_class']}|"
        f"{R.turn_bucket(r['turn'])}": 3 for r in rows}}

    def fake_arrays(run_dir, step, *, seed=0, say=None):
        rng = np.random.default_rng(abs(hash(str(run_dir))) % 1000)
        n = 200
        arr = {"p": rng.uniform(0.05, 0.95, n), "y": (rng.uniform(size=n) > 0.4).astype(float),
               "battles": np.array([f"b{i // 5}" for i in range(n)]), "w": np.ones(n)}
        return {"all": arr, "bot": arr}, {"n_traces_read": n}

    monkeypatch.setattr(CR, "_run", fake_run)
    monkeypatch.setattr(R, "build_identity_payload", lambda *_a, **_k: payload)
    monkeypatch.setattr(R, "gauge_arrays", fake_arrays)


def test_read_run_reuses_a_cached_readout_at_the_same_cycle(tmp_path, monkeypatch) -> None:
    run = _plant_run(tmp_path)
    calls: list = []
    _install_stubs(monkeypatch, calls)
    cache = tmp_path / "cache"
    CR.read_run(run, cache, _Args(), say=lambda _m: None)
    assert calls, "the first read must actually run the two tools"
    n_first = len(calls)
    doc = CR.read_run(run, cache, _Args(), say=lambda _m: None)
    assert len(calls) == n_first, "the second read must reuse, not re-run"
    assert doc["reused"] is True


def test_the_cache_is_not_reused_when_a_parameter_changes(tmp_path, monkeypatch) -> None:
    run = _plant_run(tmp_path)
    calls: list = []
    _install_stubs(monkeypatch, calls)
    cache = tmp_path / "cache"
    CR.read_run(run, cache, _Args(), say=lambda _m: None)
    n_first = len(calls)

    class Other(_Args):
        states = 999
    doc = CR.read_run(run, cache, Other(), say=lambda _m: None)
    assert len(calls) > n_first and doc["reused"] is False


def test_read_run_refuses_npz_without_win_probs(tmp_path, monkeypatch) -> None:
    run = _plant_run(tmp_path, npz_winprob=False)
    _install_stubs(monkeypatch, [])
    with pytest.raises(SystemExit) as exc:
        CR.read_run(run, tmp_path / "c", _Args(), say=lambda _m: None)
    assert exc.value.code == 2


def test_read_run_refuses_a_cycle_that_drew_too_often(tmp_path, monkeypatch) -> None:
    man = json.loads(json.dumps(_OK_MANIFEST))
    for rec in man["selection"]["opponents"].values():
        rec["battles_drawn"] = 40
    run = _plant_run(tmp_path, manifest=man)
    _install_stubs(monkeypatch, [])
    with pytest.raises(SystemExit) as exc:
        CR.read_run(run, tmp_path / "c", _Args(), say=lambda _m: None)
    assert exc.value.code == 2


def test_read_run_refuses_below_the_label_trust_gate(tmp_path, monkeypatch) -> None:
    run = _plant_run(tmp_path)
    _install_stubs(monkeypatch, [], anchor_rate=0.5)
    with pytest.raises(SystemExit) as exc:
        CR.read_run(run, tmp_path / "c", _Args(), say=lambda _m: None)
    assert exc.value.code == 2


def test_the_self_plant_is_exactly_zero_on_every_delta(tmp_path, monkeypatch) -> None:
    """The shipped self-consistency plant: an arm read against ITSELF. Every delta must be
    exactly 0.0 and every label NOT DETECTED — if any row is non-zero the two readouts are not
    the same computation and no delta from this tool means anything."""
    run = _plant_run(tmp_path)
    _install_stubs(monkeypatch, [])
    doc = CR.read_run(run, tmp_path / "c", _Args(), say=lambda _m: None)
    deltas = CR.compute_deltas(doc, doc, None, seed=0)
    assert deltas, "the plant must produce rows"
    assert all(d["delta"] == 0.0 for d in deltas), [d for d in deltas if d["delta"]]
    assert {d["label"] for d in deltas} == {"NOT DETECTED"}


def test_the_report_renders_and_the_ledger_line_is_the_registered_form(
        tmp_path, monkeypatch) -> None:
    run = _plant_run(tmp_path)
    _install_stubs(monkeypatch, [])
    doc = CR.read_run(run, tmp_path / "c", _Args(), say=lambda _m: None)
    full = {"arm": doc, "control": doc, "generated_at": "now", "out": str(tmp_path),
            "floor": CR.load_floors(None), "invocation": "python -m main.ops.critic_read …",
            "params": {"boot": 60, "gate_boot": 40, "seed": 0},
            "deltas": CR.compute_deltas(doc, doc, None, seed=0)}
    line = CR.ledger_line(full)
    assert line.startswith(f"{run.name} vs {run.name} at ")
    assert "G1 bot" in line and "turn-contrast" in line
    md = CR.render_md(full)
    assert "SUMMARY — the headline deltas" in md
    assert "NO REPLICATE FLOOR" in md


def test_the_report_renders_the_ab_separation_with_its_cell_census(tmp_path, monkeypatch) -> None:
    """🚨 The (A)/(B) block and the CELL CENSUS are one object: a within-cell resolution read
    without the counts beside it is a number, not a measurement."""
    from main.ops import conditioning_meters as CM

    run = _plant_run(tmp_path)
    _install_stubs(monkeypatch, [])
    doc = _with_conditioning(CR.read_run(run, tmp_path / "c", _Args(), say=lambda _m: None))
    doc["conditioning"]["frame"] = {
        "n_teams_seen": 40, "team_strata": 5,
        "cell_frames": {"cond.within_team_resolution.all": {
            "n_cells": 12, "n_battles": 96, "n_states": 180,
            "median_battles_per_cell": 7.0, "median_states_per_cell": 14.0,
            "min_battles_per_cell": 4.0, "max_battles_per_cell": 21.0}},
        "team_spread_frame": {"n_cells": 12.0, "n_battles": 96.0,
                              "median_battles_per_cell": 7.0}}
    full = {"arm": doc, "control": doc, "generated_at": "now", "out": str(tmp_path),
            "floor": CR.load_floors(None), "invocation": "python -m main.ops.critic_read …",
            "params": {"boot": 60, "gate_boot": 40, "seed": 0},
            "deltas": CR.compute_deltas(doc, doc, None, seed=0)}
    md = CR.render_md(full)
    assert "(A) CONDITIONING or (B) SUBSTITUTION" in md
    assert "OWN-TEAM CELL CENSUS" in md
    assert "| arm · team | 12 | 40 | 96 | 180 | 7.0 | 14.0 | 4–21 |" in md
    assert CM.PROVISIONAL_LABEL in md
    assert "Reading of the three signs:" in md


def test_a_provisional_row_never_carries_a_verdict_however_large_its_delta(
        tmp_path, monkeypatch) -> None:
    """🚨 The turn-1-vs-late contrast has NO floor and the controls cannot supply one — they read
    own-team R² ~0 at turn 1, so there is nothing for it to fall from. A huge, tight delta must
    still print PROVISIONAL: a large move is informative, a small one is not, and neither is a
    detection."""
    from main.ops import conditioning_meters as CM

    run = _plant_run(tmp_path)
    _install_stubs(monkeypatch, [])
    arm = _with_conditioning(CR.read_run(run, tmp_path / "c", _Args(), say=lambda _m: None))
    ctl = _with_conditioning(CR.read_run(run, tmp_path / "c", _Args(), say=lambda _m: None))
    rng = np.random.default_rng(1)
    arm["conditioning"]["points"][CM.OWN_TEAM_R2_DIFF] = 0.9
    arm["_cond_draws"][CM.OWN_TEAM_R2_DIFF] = 0.9 + rng.normal(0, 0.01, 400)
    rows = {r["key"]: r for r in CR.compute_deltas(arm, ctl, {CM.OWN_TEAM_R2_DIFF: 0.001},
                                                   seed=0)}
    r = rows[CM.OWN_TEAM_R2_DIFF]
    assert r["delta"] > 0.5 and r["ci"][0] > 0.0      # it would otherwise be DETECTED
    assert r["label"] == CM.PROVISIONAL_LABEL
    assert r["provisional"] is True and r["floor"] is None and r["clears_floor"] is False
    # every OTHER conditioning row still gets its registered verdict
    assert rows["cond.own_team_r2.t1"]["label"] in ("DETECTED", "NOT DETECTED", "WITHIN FLOOR")


def _with_conditioning(doc: dict) -> dict:
    """Graft a synthetic CONDITIONING block onto a readout, so the delta engine's conditioning
    family can be exercised without a trace tree."""
    from main.ops import conditioning_meters as CM

    rng = np.random.default_rng(0)
    pts = {k: 0.3 for k in CM.METER_KEYS}
    doc = dict(doc)
    doc["conditioning"] = {"points": pts, "ci": {}, "omitted": {}, "frame": {}, "refusal": None}
    doc["_cond_draws"] = {k: 0.3 + rng.normal(0, 0.05, 400) for k in CM.METER_KEYS}
    return doc


def test_every_headline_key_is_produced_by_compute_deltas(tmp_path, monkeypatch) -> None:
    """A headline the delta engine never emits would render as `NOT COMPUTED` forever; the key
    names are strings and nothing else checks that they still line up."""
    run = _plant_run(tmp_path)
    _install_stubs(monkeypatch, [])
    doc = _with_conditioning(CR.read_run(run, tmp_path / "c", _Args(), say=lambda _m: None))
    keys = {d["key"] for d in CR.compute_deltas(doc, doc, None, seed=0)}
    assert set(CR.HEADLINES) <= keys, set(CR.HEADLINES) - keys


def test_the_conditioning_family_is_a_zero_delta_against_itself(tmp_path, monkeypatch) -> None:
    """The self-consistency plant, extended to the new family: an arm read against ITSELF must
    produce exactly 0.0 on every conditioning row, with an interval covering zero."""
    from main.ops import conditioning_meters as CM

    run = _plant_run(tmp_path)
    _install_stubs(monkeypatch, [])
    doc = _with_conditioning(CR.read_run(run, tmp_path / "c", _Args(), say=lambda _m: None))
    rows = [r for r in CR.compute_deltas(doc, doc, None, seed=0)
            if r["family"] == "conditioning"]
    assert {r["key"] for r in rows} == set(CM.METER_KEYS)
    for r in rows:
        assert r["delta"] == 0.0
        assert r["ci"][0] < 0.0 < r["ci"][1]
        # a PROVISIONAL row carries no verdict at all — not even the zero-delta one.
        assert r["label"] == (CM.PROVISIONAL_LABEL if r["key"] in CM.PROVISIONAL_KEYS
                              else "NOT DETECTED")


def test_a_conditioning_meter_missing_from_one_side_is_dropped_not_compared(tmp_path,
                                                                           monkeypatch) -> None:
    """An arm whose cycle supports the Elo slope and a control whose cycle does not must not be
    compared on it — a delta against a row that does not exist is not a measurement."""
    run = _plant_run(tmp_path)
    _install_stubs(monkeypatch, [])
    base = CR.read_run(run, tmp_path / "c", _Args(), say=lambda _m: None)
    arm, ctl = _with_conditioning(base), _with_conditioning(base)
    ctl["conditioning"] = dict(ctl["conditioning"])
    ctl["conditioning"]["points"] = {k: v for k, v in ctl["conditioning"]["points"].items()
                                     if k != "cond.elo_slope"}
    keys = {r["key"] for r in CR.compute_deltas(arm, ctl, None, seed=0)
            if r["family"] == "conditioning"}
    assert "cond.elo_slope" not in keys
    assert "cond.spread_ratio.t1_3" in keys


def test_the_parser_requires_a_control_and_an_out() -> None:
    ap = CR.build_parser()
    with pytest.raises(SystemExit):
        ap.parse_args(["arm"])
    ns = ap.parse_args(["arm", "--control", "ctl", "--out", "/tmp/x"])
    assert (ns.states, ns.anchors, ns.impl, ns.anchor_tolerance) == (800, 150, "rust", 0.90)


def test_a_stratum_with_no_recorded_capture_rate_reports_no_number(tmp_path, monkeypatch) -> None:
    """Rule 17 has a corollary: a stratum the manifest cannot weight is NOT reported at weight 1
    beside corrected ones. It comes through with zero coverage and a NOT DETECTED label, never as
    a confident raw number wearing a `population` heading."""
    man = json.loads(json.dumps(_OK_MANIFEST))
    del man["selection"]["opponents"]["sentinel_0"]
    run = _plant_run(tmp_path, manifest=man)
    _install_stubs(monkeypatch, [])
    doc = CR.read_run(run, tmp_path / "c", _Args(), say=lambda _m: None)
    assert doc["identity"]["strata"]["pool"]["coverage"]["ipw"] == 0.0
    row = next(d for d in CR.compute_deltas(doc, doc, None, seed=0)
               if d["key"] == "identity.bias.pool")
    assert row["label"] == "NOT DETECTED"


def test_a_stamp_makes_this_arm_the_next_pairs_free_control(tmp_path, monkeypatch) -> None:
    """The arm's artifacts live under the PAIR's `--out`; the per-run stamp points at them. A
    later pair naming that run as its CONTROL must reuse those artifacts rather than pay for a
    second `cf_audit` on the same cycle — which is the whole reason the ladder's control is read
    once and not six times."""
    run = _plant_run(tmp_path)
    calls: list = []
    _install_stubs(monkeypatch, calls)
    pair_a, per_run = tmp_path / "pairA", tmp_path / "byrun"
    doc = CR.read_run(run, pair_a, _Args(), say=lambda _m: None)
    CR._stamp(per_run, doc)
    n_first = len(calls)
    assert (per_run / "run_readout.json").exists()

    again = CR.read_run(run, tmp_path / "pairB", _Args(), say=lambda _m: None,
                        alt_dirs=[per_run])
    assert len(calls) == n_first, "the second pair must reuse the first's artifacts"
    assert again["reused"] is True
    assert again["artifact_dir"] == str(pair_a)


def test_a_stamp_beside_its_own_artifacts_is_not_written_twice(tmp_path, monkeypatch) -> None:
    run = _plant_run(tmp_path)
    _install_stubs(monkeypatch, [])
    d = tmp_path / "same"
    doc = CR.read_run(run, d, _Args(), say=lambda _m: None)
    CR._stamp(d, doc)          # same directory: a no-op, never a self-referential rewrite
    assert json.loads((d / "run_readout.json").read_text())["artifact_dir"] == str(d)


# --------------------------------------------------------------------------- the --step pin

def test_step_pins_the_cycle_even_when_a_newer_complete_one_exists(tmp_path) -> None:
    """🚨 backlog 2026-09-09. `--on-live skip-newest` is the default and it DROPS the newest
    cycle when any process still names the run, so a finished 10M arm whose launcher had not yet
    exited is read at 8M. `--step` is the pin, and it must beat both the recency rule and the
    live-drop rule."""
    for s in (100, 200, 300):
        _cycle(tmp_path, s, manifest=_OK_MANIFEST)
    assert CR.pick_cycle(tmp_path, on_live="skip-newest", step=None)["step"] == 300
    assert CR.pick_cycle(tmp_path, on_live="skip-newest", step=100)["step"] == 100
    assert CR.pick_cycle(tmp_path, on_live="use", step=200)["step"] == 200


def test_step_refuses_a_cycle_the_run_does_not_have_and_names_the_ones_it_does(tmp_path) -> None:
    _cycle(tmp_path, 100, manifest=_OK_MANIFEST)
    _cycle(tmp_path, 200, manifest=_OK_MANIFEST)
    with pytest.raises(SystemExit) as exc:
        CR.pick_cycle(tmp_path, on_live="skip-newest", step=999)
    assert exc.value.code == 2


def test_a_pinned_step_must_still_be_a_COMPLETE_cycle(tmp_path) -> None:
    """A pin is not a licence to read a cycle that has not COLLECTED — traces may still be
    arriving into it, and a partially-written cycle is not a measurement."""
    _cycle(tmp_path, 100, manifest=_OK_MANIFEST)
    _cycle(tmp_path, 200, manifest={"selection": None})
    with pytest.raises(SystemExit) as exc:
        CR.pick_cycle(tmp_path, on_live="skip-newest", step=200)
    assert exc.value.code == 2


def test_the_why_sentence_names_the_pin(tmp_path) -> None:
    _cycle(tmp_path, 100, manifest=_OK_MANIFEST)
    c = CR.pick_cycle(tmp_path, on_live="skip-newest", step=100)
    assert CR._cycle_why(c, 100) == "PINNED by --step 100"


def test_the_why_sentence_says_when_the_newest_cycle_was_DROPPED(tmp_path) -> None:
    """The failure that motivated all of this was SILENT. The sentence that would have caught it
    has to name the drop, the pids and the fix."""
    why = CR._cycle_why({"live_pids": [4242], "on_live": "skip-newest",
                         "dropped_newest_because_live": True,
                         "steps_on_disk": [100, 200], "why": "manifest + selection recorded"},
                        None)
    assert "DROPPING THE NEWEST" in why and "4242" in why and "--step" in why


def test_the_why_sentence_says_when_the_run_is_simply_not_live(tmp_path) -> None:
    why = CR._cycle_why({"live_pids": [], "on_live": "skip-newest",
                         "dropped_newest_because_live": False, "steps_on_disk": [100],
                         "why": "manifest + selection recorded"}, None)
    assert "not live" in why


# --------------------------------------------------------------------------- the two caches

def test_the_readout_fingerprint_is_decoupled_from_the_report_version() -> None:
    """🚨 The identity half costs a ~25-minute `cf_audit`. Folding the REPORT's version into its
    cache key threw every readout on disk away the first time a new SECTION was added — which
    changes no number that key covers. Caught while regenerating the two committed ladder reads."""
    from main.ops import critic_read_render as RR

    assert CR.READOUT_FINGERPRINT_VERSION == 1
    assert RR.TOOL_VERSION > CR.READOUT_FINGERPRINT_VERSION


def test_the_conditioning_meters_are_all_in_the_delta_table_or_none_of_them_are() -> None:
    """The conditioning family is emitted from `CM.METERS`, so a meter added to the library and
    forgotten in the report is the failure this asserts against."""
    from main.ops import conditioning_meters as CM
    from main.ops import critic_read_render as RR

    assert {k for k, _q, _s in CM.METERS} == set(CM.METER_KEYS)
    assert set(RR.HEADLINES) & set(CM.METER_KEYS) == {"cond.spread_ratio.t1_3",
                                                      "cond.own_team_r2.t1",
                                                      CM.CALIB_SLOPE_ALL}
    # every calibration row the report's own section names is a declared meter — the section is
    # written from `CALIB_ROWS`, and a row listed there but never declared renders as a blank line
    assert set(RR.CALIB_ROWS) <= set(CM.METER_KEYS)


def test_a_cycle_with_no_usable_conditioning_frame_refuses_by_default(tmp_path,
                                                                     monkeypatch) -> None:
    """A trace tree whose npz carry no battle summaries yields no conditioning rows. The default
    is to REFUSE naming the cause — a report missing its headline conditioning rows would read
    exactly like one whose arm did not move them."""
    run = _plant_run(tmp_path)
    _install_stubs(monkeypatch, [])

    class WithCond(_Args):
        no_conditioning = False

    with pytest.raises(SystemExit) as exc:
        CR.read_run(run, tmp_path / "c1", WithCond(), say=lambda _m: None)
    assert exc.value.code == 2


def test_allow_conditioning_refusal_emits_the_report_saying_what_is_missing(tmp_path,
                                                                           monkeypatch) -> None:
    run = _plant_run(tmp_path)
    _install_stubs(monkeypatch, [])

    class WithCond(_Args):
        no_conditioning, allow_conditioning_refusal = False, True

    doc = CR.read_run(run, tmp_path / "c2", WithCond(), say=lambda _m: None)
    assert doc["conditioning"]["points"] == {}
    assert "0 usable states" in doc["conditioning"]["refusal"]


def test_no_conditioning_produces_no_conditioning_deltas(tmp_path, monkeypatch) -> None:
    run = _plant_run(tmp_path)
    _install_stubs(monkeypatch, [])
    doc = CR.read_run(run, tmp_path / "c3", _Args(), say=lambda _m: None)
    rows = CR.compute_deltas(doc, doc, None, seed=0)
    assert not [r for r in rows if r["family"] == "conditioning"]


# ------------------------------------------------- the COMMON-SUPPORT pass (the lever-arm fix)
#
# The calibration slope's standard error scales as 1/sd(logit V), and the SHRINKAGE hypothesis the
# row exists to test predicts the arm's sd is the smaller one — so the arm is handed a wider
# interval by the very effect under test. The common-support companion re-fits both sides inside
# the intersection of their central 95% of V, where the lever arm cannot differ. It is a PAIR-level
# pass because that window names both sides, and these tests hold it to that.

def _calib_side(*, distort, n_battles: int = 400, seed: int = 0):
    """A per-state calibration payload with a known true probability per battle, `V` a planted
    distortion of it, and one state per battle in each bucket."""
    from main.ops import calibration_slope as CS

    rng = np.random.default_rng(seed)
    p = rng.uniform(0.1, 0.9, n_battles)
    y = (rng.random(n_battles) < p).astype(float)
    v = np.clip(distort(p), 1e-4, 1 - 1e-4)
    b = np.arange(n_battles)
    col = {"v": v, "y": y, "w": np.ones(n_battles), "battle": b,
           "stratum": (b % 5).astype(int)}
    return CS.payload(n_battles=n_battles, opp_of_battle=(b % 4).astype(int),
                      buckets={"all": dict(col), "t1_3": dict(col)})


def _side_doc(payload):
    """A minimal readout doc carrying a calibration payload and the support the pass reads."""
    from main.ops import calibration_slope as CS

    sup = {b: dict(CS.support_stats(payload["buckets"][b]["v"], payload["buckets"][b]["w"]),
                   n_battles=payload["n_battles"]) for b in CS.BUCKETS}
    return {"run": "r", "step": 1, "_calib": payload, "_cond_draws": {},
            "conditioning": {"points": {}, "ci": {}, "omitted": {},
                             "frame": {"calibration_support": sup}, "refusal": None}}


def test_the_common_support_pass_fits_both_sides_inside_one_window() -> None:
    """🚨 The lever arm is equalised BY CONSTRUCTION, and the row lands on both readouts with its
    own interval — a companion that only reported the window would not be a reading."""
    from main.ops import conditioning_meters as CM

    arm = _side_doc(_calib_side(distort=lambda p: 0.5 + 0.5 * (p - 0.5), seed=1))
    ctl = _side_doc(_calib_side(distort=lambda p: p, seed=2))
    doc = CR.apply_common_support(arm, ctl, boot=120, seed=0)
    assert doc["status"] == "FITTED"
    lo, hi = doc["window"]
    assert lo < hi
    for side in (arm, ctl):
        for key in CM.PAIR_LEVEL_KEYS:
            assert key in side["conditioning"]["points"], key
            assert len(side["conditioning"]["ci"][key]) == 2
            assert side["_cond_draws"][key].size > 50
    # the shrunk side still reads the higher slope INSIDE the shared window
    assert (arm["conditioning"]["points"][CM.CALIB_SLOPE_COMMON]
            > ctl["conditioning"]["points"][CM.CALIB_SLOPE_COMMON])
    # and the supports the row was fitted on are now the same range
    for role in ("arm", "control"):
        s = doc["sides"][role]["support"]
        assert s["q_lo"] >= lo - 1e-9 and s["q_hi"] <= hi + 1e-9, role


def test_the_common_support_pass_names_the_reason_when_a_side_has_no_payload() -> None:
    """A cache written before this row existed carries no columns. That must produce a REASON, not
    a missing row a reader would read as a null."""
    arm = _side_doc(_calib_side(distort=lambda p: p, seed=3))
    ctl = _side_doc(_calib_side(distort=lambda p: p, seed=4))
    ctl["_calib"] = None
    doc = CR.apply_common_support(arm, ctl, boot=20, seed=0)
    assert doc["status"] == "NOT COMPUTED"
    assert "control" in doc["why"] and "--no-cache" in doc["why"]
    assert not arm["conditioning"]["points"]


def test_a_disjoint_pair_is_refused_rather_than_extrapolated() -> None:
    """Two heads whose central masses do not meet have NO common support. Widening the window to
    make a number appear would compare two extrapolations."""
    arm = _side_doc(_calib_side(distort=lambda p: 0.02 + 0.06 * p, seed=5))
    ctl = _side_doc(_calib_side(distort=lambda p: 0.80 + 0.15 * p, seed=6))
    doc = CR.apply_common_support(arm, ctl, boot=20, seed=0)
    assert doc["status"] == "NO OVERLAP"
    assert "extrapolation" in doc["why"]
    assert not arm["conditioning"]["points"] and not ctl["conditioning"]["points"]


def test_the_report_renders_the_calibration_slope_with_its_lever_arm(tmp_path,
                                                                    monkeypatch) -> None:
    """🚨 The slope and the support it was fitted on are ONE object in the report: a wider
    interval on the compressed side reads as a null unless sd(logit V) is on the same page."""
    from main.ops import calibration_slope as CS
    from main.ops import conditioning_meters as CM

    run = _plant_run(tmp_path)
    _install_stubs(monkeypatch, [])
    doc = _with_conditioning(CR.read_run(run, tmp_path / "c", _Args(), say=lambda _m: None))
    doc["conditioning"]["frame"] = {"calibration_support": {
        b: {"n_states": 900, "n_battles": 450, "mean_V": 0.5, "sd_V": 0.11,
            "sd_logit_V": 0.44, "q_lo": 0.31, "q_hi": 0.69, "clipped_share": 0.0}
        for b in CS.BUCKETS}}
    full = {"arm": doc, "control": doc, "generated_at": "now", "out": str(tmp_path),
            "floor": CR.load_floors(None), "invocation": "python -m main.ops.critic_read …",
            "params": {"boot": 60, "gate_boot": 40, "seed": 0},
            "common_support": {"status": "FITTED", "window": [0.31, 0.69], "sides": {}},
            "deltas": CR.compute_deltas(doc, doc, None, seed=0)}
    md = CR.render_md(full)
    assert "The CALIBRATION SLOPE" in md
    assert "SHRUNK" in md and "UNDER-dispersed" in md
    assert "The LEVER ARM these slopes are fitted on" in md
    assert "sd(logit V)" in md
    assert "0.4400" in md, "the lever arm must be printed, not merely mentioned"
    assert "COMMON-SUPPORT window is `V ∈ [0.3100, 0.6900]`" in md
    assert "replicate floor" in md
    assert CM.CALIB_SLOPE_ALL in [r["key"] for r in full["deltas"]]


def test_the_report_says_so_when_the_common_support_row_could_not_be_fitted(tmp_path,
                                                                           monkeypatch) -> None:
    run = _plant_run(tmp_path)
    _install_stubs(monkeypatch, [])
    doc = _with_conditioning(CR.read_run(run, tmp_path / "c", _Args(), say=lambda _m: None))
    full = {"arm": doc, "control": doc, "generated_at": "now", "out": str(tmp_path),
            "floor": CR.load_floors(None), "invocation": "python -m main.ops.critic_read …",
            "params": {"boot": 60, "gate_boot": 40, "seed": 0},
            "common_support": {"status": "NO OVERLAP", "window": None, "sides": {},
                               "why": "the two sides' central 95% of V do not overlap"},
            "deltas": CR.compute_deltas(doc, doc, None, seed=0)}
    md = CR.render_md(full)
    assert "The COMMON-SUPPORT row is NO OVERLAP" in md
    assert "do not overlap" in md


# ------------------------------------------------- v6: the late windows + the optimal reference

def test_a_new_v6_row_reads_vs_ZERO_NO_FLOOR_until_a_floor_json_carries_it(
        tmp_path, monkeypatch) -> None:
    """🚨 The hp800 floor json was built from the v5 rows and carries none of the v6 ones. A new
    row must therefore say in print that its detection is against ZERO — the registration's label
    is a claim about a replicate floor and quoting it without one is a stronger sentence than the
    number supports."""
    from main.ops import conditioning_meters as CM

    run = _plant_run(tmp_path)
    _install_stubs(monkeypatch, [])
    arm = _with_conditioning(CR.read_run(run, tmp_path / "c", _Args(), say=lambda _m: None))
    ctl = _with_conditioning(CR.read_run(run, tmp_path / "c", _Args(), say=lambda _m: None))
    rng = np.random.default_rng(3)
    arm["conditioning"]["points"][CM.OPP_CLASS_AUC_T4_10] = 0.8
    arm["_cond_draws"][CM.OPP_CLASS_AUC_T4_10] = 0.8 + rng.normal(0, 0.01, 400)
    # a v5 floor file: every pre-v6 key, and nothing new
    v5_floor = {k: 0.001 for k in CM.METER_KEYS
                if k not in (CM.OPP_CLASS_AUC_T1_3, CM.OPP_CLASS_AUC_T4_10,
                             CM.SPREAD_RATIO_T4_10, CM.SPREAD_RATIO_T11_24,
                             *CM.OPT_RATIO.values())}
    rows = {r["key"]: r for r in CR.compute_deltas(arm, ctl, v5_floor, seed=0)}
    new = rows[CM.OPP_CLASS_AUC_T4_10]
    assert new["label"] == "DETECTED" and new["qualifier"] == "vs ZERO — NO FLOOR"
    assert new["floor"] is None
    # ...while a row the file DOES carry is judged against its floor, unchanged
    assert rows["cond.opp_class_auc.t1"]["floor"] == 0.001


def test_the_report_prints_the_opponent_identity_split_with_both_ratios(
        tmp_path, monkeypatch) -> None:
    """A spread ratio has only two reference points without this block — 1.0 and the control — and
    1.0 is the WRONG one at early turns. The table must carry `V`'s ratio and its opponent-
    decodable part in the same row, and must say IN PRINT that the second is not a bound on the
    first — the first real read had `V` above it at every window."""
    from main.ops import conditioning_meters as CM

    run = _plant_run(tmp_path)
    _install_stubs(monkeypatch, [])
    doc = _with_conditioning(CR.read_run(run, tmp_path / "c", _Args(), say=lambda _m: None))
    doc["conditioning"]["frame"] = {"optimal_reference": {
        w: {"ratio_V": 0.30, "ratio_optimal": 0.05, "n_bins": CM.OPT_BINS,
            "n_folds": CM.OPT_FOLDS, "sd_V": 0.01, "sd_y": 0.2}
        for w, _row in CM.OPT_WINDOWS}}
    full = {"arm": doc, "control": doc, "generated_at": "now", "out": str(tmp_path),
            "floor": CR.load_floors(None), "invocation": "python -m main.ops.critic_read …",
            "params": {"boot": 60, "gate_boot": 40, "seed": 0},
            "deltas": CR.compute_deltas(doc, doc, None, seed=0)}
    md = CR.render_md(full)
    assert "HOW MUCH OF THE SPREAD IS OPPONENT IDENTITY" in md
    assert "| `t4_10` | +0.3000 | +0.0500 | 6.000 |" in md
    # 🚨 the first read of this row had V ABOVE the "ceiling" at every window; the report must
    # say in print that it is a decomposition and not a bound, or the next reader repeats it.
    assert "IT IS NOT AN UPPER BOUND, AND `V` ROUTINELY EXCEEDS IT" in md
    assert "never a pass/fail bar" in md


def test_the_optimal_block_is_absent_rather_than_empty_when_the_frame_has_no_reference(
        tmp_path, monkeypatch) -> None:
    """An old cached readout has no `optimal_reference`. The block must vanish, not render an
    empty table that reads like a measured zero."""
    run = _plant_run(tmp_path)
    _install_stubs(monkeypatch, [])
    doc = _with_conditioning(CR.read_run(run, tmp_path / "c", _Args(), say=lambda _m: None))
    full = {"arm": doc, "control": doc, "generated_at": "now", "out": str(tmp_path),
            "floor": CR.load_floors(None), "invocation": "x",
            "params": {"boot": 60, "gate_boot": 40, "seed": 0},
            "deltas": CR.compute_deltas(doc, doc, None, seed=0)}
    assert "HOW MUCH OF THE SPREAD IS OPPONENT IDENTITY" not in CR.render_md(full)


def test_the_conditioning_cache_key_moves_when_the_meter_set_does() -> None:
    """🚨 The conditioning block is CACHED on its own fingerprint. v6 adds rows, so a v5 cache
    must MISS — otherwise a re-read of a landed pair would serve the old block and silently report
    the new rows as absent. The identity half's key is deliberately NOT bumped: it costs a ~25-min
    `cf_audit` and none of its numbers changed."""
    assert CR.READOUT_FINGERPRINT_VERSION == 1
    cycle = {"run_dir": "/r", "read_root": "/r", "step": 10, "manifest": {"saved_at": "t"}}
    fp = CR._cond_fingerprint(cycle, _Args())
    assert fp["block_version"] == 3
    from main.ops import conditioning_meters as CM
    for key in (CM.SPREAD_RATIO_T4_10, CM.OPP_CLASS_AUC_T4_10, *CM.OPT_RATIO.values()):
        assert key in fp["meters"], key
