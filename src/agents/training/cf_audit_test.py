"""Pure unit tests for cf_audit's statistics and bookkeeping — no battles, no bridge, no model.

The estimator tests are the load-bearing ones. ``sd_true_excess`` is the program's PRIMARY
meter, and an estimator that reports structure where there is none would license a lever on
noise; so it is validated **at zero true effect** (a synthetic cell whose entire spread IS the
binomial floor must return ~0) as well as at a known nonzero one.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from agents.training.cf_audit import (CONVICTION_DECILE, Decision, LabelRow, SCHEMA_VERSION,
                                      cluster_bootstrap_ci, obs_digest, sd_true_excess,
                                      stratified_sample, stratum_of, turn_tercile_edges,
                                      wilson_ci, write_labels)


def _dec(**kw):
    base = dict(battle="/t/b1", short="heuristic/b1", opponent="heuristic", opp_class="bot",
                outcome="loss", inv=3, turn=10, win_prob=0.5, value=1.0, action=6,
                move_rank=1, n_moves=8)
    base.update(kw)
    return Decision(**base)


def test_the_conviction_class_CI_brackets_its_own_gap():
    """End-to-end through `bias_map`: whatever the arms' sizes, `loss_minus_win_ci` must contain
    `loss_minus_win_gap`. That is the invariant the shipped report violated."""
    from agents.training.cf_audit import bias_map
    labels = []
    # 24 conviction battles (lost) at gap +0.30, 8 control battles (won) at gap +0.10.
    for tag, n_batt, wp, mc, oc in (("L", 24, 0.90, 0.60, "loss"), ("W", 8, 0.90, 0.80, "win")):
        for b in range(n_batt):
            for k in range(6):
                labels.append({"battle": f"/t/{tag}{b}", "opponent": "heuristic",
                               "opp_class": "bot", "outcome": oc, "turn": 10 + k,
                               "win_prob": wp, "mc": mc})
    frame = [_dec(win_prob=r["win_prob"], outcome=r["outcome"], battle=r["battle"], turn=r["turn"])
             for r in labels]
    design = {"turn_tercile_edges": [12.0, 14.0], "sampler_version": "test", "seed": 0}
    cc = bias_map(labels, frame, n_rollouts=8, design=design, accounting={})["conviction_class"]
    gap, (lo, hi) = cc["loss_minus_win_gap"], cc["loss_minus_win_ci"]
    assert gap == pytest.approx(0.20, abs=1e-9)              # (0.90−0.60) − (0.90−0.80)
    assert lo <= gap <= hi, f"CI [{lo}, {hi}] excludes its own estimate {gap}"


# ----------------------------------------------------------------- stratifier

def test_stratum_is_confidence_decile_by_outcome_by_turn_tercile():
    edges = (10.0, 23.0)
    assert stratum_of(_dec(win_prob=0.0, turn=5), edges) == (0, "loss", 0)
    assert stratum_of(_dec(win_prob=0.99, turn=30, outcome="win"), edges) == (9, "win", 2)
    assert stratum_of(_dec(win_prob=1.0, turn=15), edges) == (9, "loss", 1), "1.0 must not overflow"


def test_turn_terciles_are_read_off_the_data():
    frame = [_dec(turn=t) for t in range(1, 31)]
    lo, hi = turn_tercile_edges(frame)
    assert 9 <= lo <= 12 and 19 <= hi <= 22


def test_stratifier_oversamples_the_high_confidence_loss_region():
    """The conviction class is the population R1 supervises, and it is RARE — so it is
    deliberately over-sampled relative to its frame share (and corrected for at aggregation)."""
    frame = ([_dec(battle=f"/b{i}", win_prob=0.9, outcome="loss", inv=i) for i in range(200)]
             + [_dec(battle=f"/c{i}", win_prob=0.2, outcome="win", inv=i) for i in range(200)])
    sample, design = stratified_sample(frame, 200, seed=5, max_per_battle=99)
    conv_frame = 0.5
    conv_sample = np.mean([d.win_prob >= 0.75 and d.outcome == "loss" for d in sample])
    assert conv_sample > conv_frame * 1.4, conv_sample
    assert design["conviction_decile_floor"] == CONVICTION_DECILE
    assert design["sampler_version"] and design["seed"] == 5


def test_stratifier_caps_per_battle_so_one_long_game_cannot_carry_a_stratum():
    frame = [_dec(battle="/only", inv=i, win_prob=0.5) for i in range(400)]
    sample, _ = stratified_sample(frame, 200, seed=5, max_per_battle=12)
    assert len(sample) == 12


def test_stratifier_is_deterministic_under_a_seed():
    frame = [_dec(battle=f"/b{i % 20}", inv=i, win_prob=(i % 10) / 10) for i in range(300)]
    a, _ = stratified_sample(frame, 50, seed=7)
    b, _ = stratified_sample(frame, 50, seed=7)
    c, _ = stratified_sample(frame, 50, seed=8)
    assert [d.inv for d in a] == [d.inv for d in b]
    assert [d.inv for d in a] != [d.inv for d in c]


def test_stratifier_design_records_every_populated_cell():
    frame = [_dec(battle=f"/b{i}", inv=i, win_prob=(i % 10) / 10,
                  outcome="win" if i % 2 else "loss", turn=1 + i % 40) for i in range(400)]
    _, design = stratified_sample(frame, 100, seed=1)
    assert design["cells"] and sum(c["frame_n"] for c in design["cells"]) == len(frame)


# --------------------------------------------------------------- schema writer

def test_label_row_serializes_the_v1_schema_exactly(tmp_path):
    obs = np.arange(8, dtype=np.float32)
    row = LabelRow(battle="/t/b_reconstruction.json", decision_idx=4, obs_sha1=obs_digest(obs),
                   label=0.625, n_rollouts=8, wilson_lo=0.3, wilson_hi=0.86,
                   policy_step=24000000, opponent="heuristic", obs_npz="/t/b_states.npz::obs")
    path = write_labels([row], str(tmp_path), producer="cf_audit", seq=24000000)
    assert os.path.basename(path) == "labels_cf_audit_24000000.jsonl"
    assert os.path.basename(os.path.dirname(path)) == "cf_labels"
    with open(path) as f:
        d = json.loads(f.readline())
    assert set(d) == {"schema", "kind", "battle", "decision_idx", "obs_sha1", "obs_npz",
                      "obs_inline", "label", "n_rollouts", "wilson_lo", "wilson_hi",
                      "policy_step", "opponent", "created_unix"}
    assert d["schema"] == SCHEMA_VERSION and d["kind"] == "mc_winprob"
    assert d["obs_inline"] is None and d["obs_npz"].endswith("::obs")
    assert 0.0 <= d["label"] <= 1.0 and d["wilson_lo"] <= d["label"] <= d["wilson_hi"]


def test_obs_digest_is_over_the_float32_BYTES_so_a_consumer_can_verify_its_row():
    a = np.arange(8, dtype=np.float32)
    assert obs_digest(a) == obs_digest(a.astype(np.float64))     # cast is normalized
    b = a.copy()
    b[3] += np.float32(1e-6)
    assert obs_digest(a) != obs_digest(b), "a one-ulp change must change the digest"


def test_obs_digest_matches_a_hand_computed_sha1():
    import hashlib
    a = np.array([1.0, 2.0], dtype=np.float32)
    assert obs_digest(a) == hashlib.sha1(a.tobytes()).hexdigest()


# ---------------------------------------------------------- the EVIDENTIAL read
#
# The pre-registered success meter for `--cf-evidential` is not the loss — it is whether the
# confessed Beta WIDTH tracks the measured `sd_true_excess` per stratum. Nothing computed that
# correlation until this batch, which meant the experiment had a declared meter with no reader.

def _evid_labels(width_of, *, n_per_decile=16, n_battles=8, rng_seed=0):
    """Synthetic labels whose per-decile MC spread is controlled, plus a width the caller chooses
    as a function of that decile — so a test can construct a head that tracks the blur and one
    that does not, and demand opposite verdicts."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for dec in range(10):
        wp = dec / 10 + 0.05
        # spread GROWS with the decile: the ground truth the width is supposed to discover
        spread = 0.02 + 0.04 * dec
        for i in range(n_per_decile):
            mc = float(np.clip(rng.normal(wp, spread), 0.0, 1.0))
            rows.append({"battle": f"/t/b{i % n_battles}", "inv": i, "turn": 10 + i,
                         "opponent": "heuristic", "opp_class": "bot",
                         "outcome": "win" if i % 2 else "loss", "win_prob": wp, "value": 0.0,
                         "mc": mc, "wins": int(mc * 8), "n": 8, "opponent_source": "bot",
                         "evid_width": float(width_of(dec)), "evid_precision": 20.0})
    return rows


def _frame_mass(labels):
    from collections import Counter
    c = Counter()
    for r in labels:
        c[(min(9, int(r["win_prob"] * 10)), r["outcome"])] += 1
    return c


def test_a_head_whose_width_TRACKS_the_blur_scores_positive():
    labels = _evid_labels(lambda dec: 0.05 + 0.03 * dec)
    from agents.training.cf_audit import evidential_read
    ev = evidential_read(labels, _frame_mass(labels), 8, draws=200)
    assert ev["n_strata"] >= 8
    assert ev["width_vs_blur_spearman"] > 0.7, ev
    lo, hi = ev["width_vs_blur_ci"]
    assert lo is not None and lo > 0.0, "the CI does not exclude 'no relation'"


def test_a_head_that_is_WIDE_EVERYWHERE_scores_the_null_and_says_so():
    """The failure mode the meter exists to catch: a confessed width that is a constant carries no
    information about which states the critic cannot separate."""
    labels = _evid_labels(lambda dec: 0.11)
    from agents.training.cf_audit import evidential_read
    ev = evidential_read(labels, _frame_mass(labels), 8, draws=50)
    assert ev["width_vs_blur_spearman"] is None
    assert ev["width_vs_blur_ci"] == [None, None]
    assert ev["evid_width_mean"] == pytest.approx(0.11)   # …the LEVEL is still reported


def test_a_head_whose_width_ANTI_tracks_the_blur_scores_negative():
    labels = _evid_labels(lambda dec: 0.5 - 0.03 * dec)
    from agents.training.cf_audit import evidential_read
    ev = evidential_read(labels, _frame_mass(labels), 8, draws=200)
    assert ev["width_vs_blur_spearman"] < -0.7, ev


def test_the_bias_map_carries_the_evidential_block_ONLY_when_the_head_was_read():
    """Absent, never zero: 'this checkpoint has no head' and 'this head claims no uncertainty' are
    opposite findings, and a column of zeros renders them identically."""
    from agents.training.cf_audit import bias_map
    labels = _evid_labels(lambda dec: 0.05 + 0.03 * dec)
    frame = [_dec(win_prob=r["win_prob"], outcome=r["outcome"], battle=r["battle"], turn=r["turn"])
             for r in labels]
    design = {"turn_tercile_edges": [12.0, 18.0], "sampler_version": "test", "seed": 0}
    with_head = bias_map(labels, frame, n_rollouts=8, design=design, accounting={})
    assert with_head["evidential"] is not None
    assert any(c.get("evid_width_mean") is not None for c in with_head["resolution"])

    headless = [{k: v for k, v in r.items() if not k.startswith("evid_")} for r in labels]
    without = bias_map(headless, frame, n_rollouts=8, design=design, accounting={})
    assert without["evidential"] is None
    assert all("evid_width_mean" not in c for c in without["resolution"])


# `test_the_markdown_says_ABSENT_rather_than_rendering_zeros` moved to
# `cf_audit_render_test.py` with `render_markdown` (2026-09-06). It imports `_evid_labels` from
# here, so the renderer is still tested against the bias map this file builds.


def test_attach_evidential_is_a_NO_OP_when_the_checkpoint_has_no_head(capsys):
    """A model that will not load, or one without the head, must cost the audit its evidential
    columns and NOTHING ELSE — the labels and the bias map are the product."""
    from agents.training.cf_audit import attach_evidential

    class _NoHead:
        def cf_evidential_batch(self, obs, masks=None):
            return None

    class _Session:
        def probe_model(self, battle_id):
            return _NoHead(), None

    labels = [{"battle": "/t/b1", "inv": 0}]
    cache = {"/t/b1_states.npz": np.zeros((3, 4), dtype=np.float32)}
    assert attach_evidential(_Session(), labels, cache) == 0
    assert "evid_width" not in labels[0]
    assert "no cf_evid_head" in capsys.readouterr().out

    class _Broken:
        def probe_model(self, battle_id):
            raise FileNotFoundError("no checkpoint at this architecture")

    assert attach_evidential(_Broken(), labels, cache) == 0
    assert "no model" in capsys.readouterr().out
    # a session with no such method at all (an injected test double) is silently fine
    assert attach_evidential(object(), labels, cache) == 0


def test_attach_evidential_writes_the_width_and_precision_it_was_given():
    from agents.training.cf_audit import attach_evidential

    class _Head:
        def cf_evidential_batch(self, obs, masks=None):
            n = len(obs)
            return np.full(n, 2.0), np.full(n, 6.0)      # Beta(2, 6)

    class _Session:
        def probe_model(self, battle_id):
            return _Head(), None

    labels = [{"battle": "/t/b1", "inv": i} for i in range(3)]
    cache = {"/t/b1_states.npz": np.zeros((3, 4), dtype=np.float32)}
    assert attach_evidential(_Session(), labels, cache) == 3
    # std of Beta(2,6) = sqrt(2*6 / (8^2 * 9)) = sqrt(12/576)
    assert labels[0]["evid_width"] == float(np.sqrt(12 / 576))
    assert labels[0]["evid_precision"] == 8.0


# --------------------------------------------------------- TWIN HEADS + SHADOW (v99)
# gen3_cf_twin_heads_v1. The section moved to `cf_audit_twin_test.py` with the functions it
# covers (2026-09-06); the golden below still calls them through this module's re-exports, which
# is what proves the move changed nothing.


# ══════════════════════════════════════════════════ the EXTRACTION parity golden ══
#
# `wilson_ci`, `spearman`, `cluster_bootstrap_ci`, `cluster_bootstrap_diff_ci` and
# `sd_true_excess` moved to `agents/training/stats.py` on 2026-09-06; `render_markdown` moved to
# `cf_audit_render.py` and the twin-head / shadow-critic block to `cf_audit_twin.py` the same day.
# Every one of them is still CALLED HERE through `cf_audit`'s re-exports, which is the point: this
# golden is what makes each move an EXTRACTION rather than a rewrite, and it has not been
# regenerated for any of them. Each claimed to be pure refactoring, and this is the evidence for
# that claim rather than a promise of it: every public
# readout of the audit, run on one synthetic fixture, JSON-serialised canonically and pinned by
# digest — the golden below was captured from the tree BEFORE the extraction and reproduced
# byte-for-byte after it.
#
# The named values beside the digest are not decoration. A bare hash cannot say WHICH readout
# moved when it fails, and a test whose only assertion is an opaque digest is one refactor away
# from being regenerated on autopilot.
#
# ⚠️ Regenerating: only ever after establishing WHY it changed. A numpy upgrade can legitimately
# move the bootstrap percentiles (`default_rng` streams are stable, `np.percentile` tie handling
# is not guaranteed forever); a change in this module's arithmetic is not legitimate without a
# stated reason. Print the current digest with::
#
#     python -c "import hashlib; from agents.training import cf_audit_test as t; \
#                print(hashlib.sha256(t._parity_blob().encode()).hexdigest())"

_PARITY_SHA256 = "cf2971d505ef96faf18bce4bc50608ec53831f052055ea4ccc9e4c15b06afd79"


def _parity_labels():
    """A deterministic label set that reaches every branch of the bias map: both outcomes, a
    spread of deciles and turns, two opponents, the conviction class AND its control, and the
    optional evidential / twin-head / shadow-critic columns."""
    rng = np.random.default_rng(20260906)
    rows = []
    for b in range(40):
        outcome = "loss" if b % 3 else "win"
        opp = "heuristic" if b % 2 else "staller"
        for k in range(7):
            wp = float(np.clip(0.05 + ((b * 7 + k) % 19) / 20.0, 0.0, 0.999))
            mc = float(np.clip(wp - 0.12 + rng.normal(0, 0.18), 0.0, 1.0))
            rows.append({
                "battle": f"/t/{opp}/b{b}", "short": f"{opp}/b{b}", "opponent": opp,
                "opp_class": "bot" if b % 2 else "sentinel",
                "outcome": outcome, "inv": k, "turn": 2 + (b + 3 * k) % 40,
                "win_prob": wp, "mc": mc,
                "evid_width": float(0.05 + 0.4 * abs(wp - 0.5) + 0.01 * k),
                "evid_precision": float(3.0 + k),
                "twin_b_pred": float(np.clip(wp - 0.03 + rng.normal(0, 0.02), 0, 1)),
                "twin_c_pred": float(np.clip(mc + rng.normal(0, 0.05), 0, 1)),
                "shadow_value": float(rng.normal(0.2, 0.5)),
                "live_v": float(rng.normal(0.35, 0.5)),
            })
    return rows


def _parity_readouts() -> dict:
    """Every public entry point of `cf_audit`, on the fixture above."""
    from collections import Counter

    from agents.training.cf_audit import (SAMPLER_VERSION, bias_map, cluster_bootstrap_diff_ci,
                                          evidential_read, paired_head_read, render_markdown,
                                          resolution_cells, shadow_read, spearman,
                                          twin_resolution_read)

    labels = _parity_labels()
    frame = [_dec(battle=r["battle"], short=r["short"], opponent=r["opponent"],
                  opp_class=r["opp_class"], outcome=r["outcome"], inv=r["inv"], turn=r["turn"],
                  win_prob=r["win_prob"], value=0.1, action=6, move_rank=r["inv"], n_moves=7)
             for r in labels]
    frame_mass: Counter = Counter()
    for d in frame:
        frame_mass[(min(9, int(d.win_prob * 10)), d.outcome)] += 1
    design = {"turn_tercile_edges": list(turn_tercile_edges(frame)),
              "sampler_version": SAMPLER_VERSION, "seed": 3}
    bm = bias_map(labels, frame, n_rollouts=8, design=design, accounting={"anchor_ok": 1.0})
    sample, sdesign = stratified_sample(frame, 60, seed=3)
    loss = [r for r in labels if r["outcome"] == "loss"]
    win = [r for r in labels if r["outcome"] == "win"]
    return {
        # the statistics that moved, called directly
        "wilson": [list(wilson_ci(w, n)) for w, n in
                   ((0, 0), (0, 8), (4, 8), (8, 8), (2.5, 7), (400, 800))],
        "spearman": [spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 6]),
                     spearman([1, 1, 1, 1], [1, 2, 3, 4]),
                     spearman([1, 2], [1, 2])],
        "cluster_ci": list(cluster_bootstrap_ci([r["mc"] for r in labels],
                                                [r["battle"] for r in labels],
                                                draws=500, seed=7)),
        "cluster_diff_ci": list(cluster_bootstrap_diff_ci(
            [r["mc"] for r in loss], [r["battle"] for r in loss],
            [r["mc"] for r in win], [r["battle"] for r in win], draws=500, seed=7)),
        "sd_true_excess": sd_true_excess([r["mc"] for r in labels], 8),
        "sd_true_excess_weighted": sd_true_excess(
            [r["mc"] for r in labels], 8,
            weights=[1.0 + (i % 5) for i in range(len(labels))]),
        # the audit's own readouts, which consume them
        "bias_map": bm,
        "resolution_cells": resolution_cells(labels, frame_mass, 8),
        "evidential_read": evidential_read(labels, frame_mass, 8, draws=120, seed=11),
        "paired_head_read": paired_head_read(labels, draws=300, seed=13),
        "twin_resolution_read": twin_resolution_read(labels, 8),
        "shadow_read": shadow_read(labels, draws=300, seed=17),
        "markdown": render_markdown(bm, run_dir="/t/run", step=1234, ckpt="/t/ck.zip"),
        "turn_tercile_edges": list(turn_tercile_edges(frame)),
        "stratified_sample": [[d.battle, d.inv] for d in sample],
        "stratified_design_cells": sdesign["cells"],
        "obs_digest": obs_digest(np.arange(8, dtype=np.float32)),
    }


def _parity_blob() -> str:
    return json.dumps(_parity_readouts(), sort_keys=True, separators=(",", ":"), default=repr)


def test_every_public_readout_matches_the_pre_extraction_golden():
    """The statistics live in `agents.training.stats` now; nothing they compute may have moved."""
    import hashlib
    blob = _parity_blob()
    assert '"' in blob and len(blob) > 10_000, "the fixture produced nothing to compare"
    got = hashlib.sha256(blob.encode()).hexdigest()
    assert got == _PARITY_SHA256, (
        f"cf_audit's readouts changed: {got} != {_PARITY_SHA256}. Read the named-value test "
        f"below to find which one, and do not regenerate the digest without a stated reason.")


def test_the_named_golden_values_are_what_the_digest_stands_for():
    """The digest says 'nothing moved'; these say WHAT it is that did not move — so a failure
    names the readout instead of a hash, and so the golden cannot be regenerated blind."""
    r = _parity_readouts()
    h = r["bias_map"]["headline"]
    assert h["n_labels"] == 280 and h["n_battles"] == 40
    assert h["population_weighted_gap"] == pytest.approx(0.10380581687311953, abs=1e-12)
    assert h["population_weighted_sd_true_excess"] == pytest.approx(0.04689387559950886, abs=1e-12)
    cc = r["bias_map"]["conviction_class"]
    assert cc["loss_minus_win_gap"] == pytest.approx(-0.0649106018083974, abs=1e-12)
    assert cc["loss_minus_win_ci"] == pytest.approx([-0.13981247151137305, 0.020275190848265102],
                                                   abs=1e-12)
    assert cc["loss_minus_win_ci"][0] <= cc["loss_minus_win_gap"] <= cc["loss_minus_win_ci"][1]
    assert r["sd_true_excess"]["sd_true_excess"] == pytest.approx(0.23816673899899182, abs=1e-12)
    assert r["sd_true_excess"]["sd_binomial_floor"] == pytest.approx(0.1506020869266234, abs=1e-12)
    assert r["evidential_read"]["width_vs_blur_spearman"] == pytest.approx(0.22510994381306007,
                                                                          abs=1e-12)
    assert r["evidential_read"]["width_vs_blur_ci"] == pytest.approx(
        [-0.26547633223229367, 0.5753922836133598], abs=1e-12)
    assert r["cluster_ci"] == pytest.approx([0.3389425853277456, 0.43906035315049646], abs=1e-12)
    assert r["cluster_diff_ci"] == pytest.approx(
        [0.05065245683999575, -0.049559904570685806, 0.1554283053841466], abs=1e-12)
    first = r["paired_head_read"]["contrasts"][0]
    assert first["contrast"] == "B_minus_A"
    assert first["brier"] == pytest.approx(-0.005088422875601386, abs=1e-12)
    assert first["brier_ci"] == pytest.approx(
        [-0.006624555553282069, -0.0038394225362491496], abs=1e-12)
    sh = r["shadow_read"]
    assert sh["shadow_vs_live_v"] == pytest.approx(-0.1067942550072824, abs=1e-12)
    assert sh["shadow_vs_live_v_ci"] == pytest.approx(
        [-0.18986503240932356, -0.01902852911531018], abs=1e-12)
    assert len(r["resolution_cells"]) == 10 and len(r["markdown"]) == 6276
