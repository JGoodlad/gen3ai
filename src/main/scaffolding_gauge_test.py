"""`python -m main.scaffolding_gauge` — folded over a CONSTRUCTED trace tree with known answers.

A fixture run is written from scratch here rather than borrowed, because the property under test
is a RELATIONSHIP between two recorded columns and the only way to know the right answer is to put
it there: one step whose ``values`` and ``win_probs`` agree by construction, one where they are
inverted, one where V is flat. The tool must recover exactly those three readings and must refuse —
loudly, with a reason — on the case that actually occurs in the wild: a run with no win-prob head,
whose ``win_probs`` column is all NaN.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from main.scaffolding_gauge import (
    SelectionWeightError, build_reliability, build_report, collect_slices, main, opponent_class,
    render, selection_weights, true_win_rates, write_plot,
)

_OBS = 4


def _battle(run: str, step: int, opponent: str, name: str, values, win_probs) -> None:
    """One trace pair: the `_summary.json` discovery walks and the `_states.npz` the gauge reads."""
    d = os.path.join(run, "eval_traces", f"step_{step}", opponent)
    os.makedirs(d, exist_ok=True)
    n = len(values)
    with open(os.path.join(d, f"{name}_summary.json"), "w") as fh:
        json.dump({"meta": {"step": step, "result": name.split("_")[0].upper()},
                   "invocations": [{"i": i + 1, "turn": i + 1} for i in range(n)]}, fh)
    np.savez(os.path.join(d, f"{name}_states.npz"),
             obs=np.zeros((n, _OBS), dtype=np.float32),
             has_state=np.ones(n, dtype=np.int8),
             values=np.asarray(values, dtype=np.float32),
             win_probs=np.asarray(win_probs, dtype=np.float32))


def _build_run(root: str) -> str:
    """Three steps, three regimes, twelve battles each so the cluster bootstrap has clusters.

    step 1_000_000 — AGREEING: win_prob is a strictly increasing function of V ⇒ rank gauge 0.
    step 2_000_000 — INVERTED: win_prob decreases in V ⇒ rank gauge 1.
    step 3_000_000 — FLAT V (the PBRS constancy endpoint) ⇒ rank gauge NaN, v_std 0.
    """
    run = os.path.join(root, "run_gauge_fixture")
    rng = np.random.default_rng(0)
    for b in range(12):
        won = b % 2 == 0
        name = f"{'win' if won else 'loss'}_{b:03d}"
        base = 1.0 if won else -1.0
        v = base + rng.normal(scale=0.4, size=8)
        p = 1.0 / (1.0 + np.exp(-v))                     # monotone in V
        _battle(run, 1_000_000, "aggressive", name, v, p)
        _battle(run, 2_000_000, "aggressive", name, v, 1.0 - p)
        _battle(run, 3_000_000, "aggressive", name, np.full(8, 2.5), p)
    with open(os.path.join(run, "metadata.json"), "w") as fh:
        json.dump({"git_hash": "cafebabe"}, fh)
    return run


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    run = _build_run(str(tmp_path_factory.mktemp("gauge")))
    slices, coverage = collect_slices(run)
    return build_report(slices, coverage, run_dir=run, n_boot=200, seed=0)


def _row(report, step):
    return next(r for r in report["curve"] if r["step"] == step)


# ══ 1. The three constructed regimes come back exactly ═══════════════════════


def test_the_agreeing_step_reads_zero_divergence(report):
    r = _row(report, 1_000_000)
    assert r["n_battles"] == 12 and r["n_states"] == 96
    assert r["rank_rho"] == pytest.approx(1.0)
    assert r["rank_gauge"] == pytest.approx(0.0)
    assert r["rank_ci_lo"] == pytest.approx(0.0) and r["rank_ci_hi"] == pytest.approx(0.0)


def test_the_inverted_step_reads_full_divergence(report):
    r = _row(report, 2_000_000)
    assert r["rank_rho"] == pytest.approx(-1.0)
    assert r["rank_gauge"] == pytest.approx(1.0)


def test_the_flat_V_step_reads_NaN_rank_and_zero_spread_not_a_fabricated_number(report):
    """The PBRS constancy endpoint, end to end. The rank gauge has nothing to rank with and must
    say so; the constancy row is what still carries information there."""
    r = _row(report, 3_000_000)
    assert math.isnan(r["rank_rho"]) and math.isnan(r["rank_gauge"])
    assert r["const_v_std"] == pytest.approx(0.0)
    assert r["affine_v_constant"] == 1.0
    assert r["affine_a"] == pytest.approx(0.0)


def test_the_affine_gauge_is_reported_in_probability_units_with_its_disclaimer_column(report):
    r = _row(report, 1_000_000)
    assert 0.0 <= r["affine_rms"] <= 1.0
    for key in ("affine_brier_head", "affine_brier_v_affine", "affine_brier_base",
                "affine_readout_penalty", "affine_base_rate"):
        assert key in r, key
    assert r["affine_base_rate"] == pytest.approx(0.5)     # six wins, six losses
    assert r["affine_readout_penalty"] == pytest.approx(
        r["affine_brier_v_affine"] - r["affine_brier_head"])


def test_the_constancy_row_splits_within_from_between_battle(report):
    r = _row(report, 1_000_000)
    assert r["const_n_groups"] == 12.0
    assert 0.0 < r["const_within_frac"] < 1.0              # V varies both inside and across battles
    assert r["const_between_std"] > 0.0


def test_the_trend_block_covers_all_three_curves(report):
    t = report["trend"]
    assert set(t) == {"rank_gauge", "affine_rms", "v_std"}
    # The rank slope skips the NaN step, leaving two points — below the 3-point floor, so NaN.
    assert math.isnan(t["rank_gauge"]["slope_per_Mstep"]) and t["rank_gauge"]["n_points"] == 2.0
    assert t["v_std"]["n_points"] == 3.0


# ══ 2. The units honesty ships INSIDE the artifact ═══════════════════════════


def test_every_gauge_carries_a_what_and_a_CANNOT(report):
    """A number quotable without its caveat will be quoted without it. Both halves are structural."""
    assert set(report["units"]) == {"recorded_V", "rank_gauge", "affine_gauge", "constancy",
                                    "curve"}
    for name, block in report["units"].items():
        assert block["what"] and block["cannot"], name
    assert "PopArt" in report["units"]["recorded_V"]["what"]
    assert "cannot be converted" in report["units"]["recorded_V"]["cannot"]
    assert "per-checkpoint FIT" in report["units"]["affine_gauge"]["cannot"]
    assert "readout_penalty" in report["units"]["affine_gauge"]["cannot"]


def test_the_rendered_table_states_the_units_and_the_cluster_bootstrap(report):
    txt = render(report)
    assert "SCAFFOLDING GAUGE" in txt
    assert "PopArt-normalized SHAPED return" in txt
    assert "CLUSTER bootstrap over BATTLES" in txt
    assert "UNIT-FREE" in txt and "PROBABILITY units" in txt


def test_the_constancy_only_view_is_the_quotable_one_line_check(report):
    txt = render(report, constancy_only=True)
    assert "CONSTANCY SANITY ROW" in txt and "db9bb5c" in txt
    assert "v_std trend" in txt
    assert "CONFIRMS the theory" in txt
    assert "RANK GAUGE" not in txt                       # it really is only the one block


# ══ 3. Refusals and edge cases ═══════════════════════════════════════════════


def test_a_run_with_NO_win_prob_head_REFUSES_instead_of_curving_zeros(tmp_path):
    """The case that actually occurs: `--win-prob-mode none` records an all-NaN column. A gauge of
    zeros there would read as "the two readouts agree perfectly", which is the exact opposite of
    "there is no second readout"."""
    run = str(tmp_path / "headless")
    _battle(run, 1_000_000, "aggressive", "win_000",
            [1.0, 2.0, 3.0], [float("nan")] * 3)
    with pytest.raises(SystemExit) as exc:
        collect_slices(run)
    msg = str(exc.value)
    assert "--win-prob-mode" in msg and "REFUSAL, not a zero" in msg


def test_an_empty_or_absent_trace_tree_refuses_with_the_path_it_looked_at(tmp_path):
    with pytest.raises(SystemExit) as exc:
        collect_slices(str(tmp_path / "nothing"))
    assert "no eval traces" in str(exc.value)


def test_rows_without_has_state_or_with_a_NaN_are_dropped_not_propagated(tmp_path):
    run = str(tmp_path / "partial")
    d = os.path.join(run, "eval_traces", "step_1000000", "aggressive")
    os.makedirs(d)
    with open(os.path.join(d, "win_000_summary.json"), "w") as fh:
        json.dump({"meta": {}, "invocations": []}, fh)
    np.savez(os.path.join(d, "win_000_states.npz"),
             obs=np.zeros((4, _OBS), dtype=np.float32),
             has_state=np.array([1, 0, 1, 1], dtype=np.int8),
             values=np.array([1.0, 9.0, 2.0, np.nan], dtype=np.float32),
             win_probs=np.array([0.1, 0.9, 0.2, 0.4], dtype=np.float32))
    _battle(run, 1_000_000, "aggressive", "loss_001", [3.0, 4.0], [0.3, 0.4])
    slices, cov = collect_slices(run)
    assert slices[1_000_000]["values"].size == 4          # 2 kept from the first + 2 from the second
    assert cov["n_traces_read"] == 2


def test_the_opponent_filter_and_the_seeded_battle_cap_keep_clusters_intact(tmp_path):
    run = str(tmp_path / "multi")
    for b in range(6):
        _battle(run, 1_000_000, "aggressive", f"win_{b:03d}", [1.0, 2.0], [0.4, 0.6])
        _battle(run, 1_000_000, "staller", f"loss_{b:03d}", [1.0, 2.0], [0.4, 0.6])
    only, _ = collect_slices(run, opponent="staller")
    assert set(only[1_000_000]["opponents"].tolist()) == {"staller"}

    capped, _ = collect_slices(run, max_battles_per_step=3, seed=1)
    ids = capped[1_000_000]["battles"]
    assert len(set(ids.tolist())) == 3                    # whole battles, never partial ones
    assert ids.size == 6                                  # every kept battle kept ALL its rows
    again, _ = collect_slices(run, max_battles_per_step=3, seed=1)
    assert set(again[1_000_000]["battles"].tolist()) == set(ids.tolist())   # seeded ⇒ reproducible


def test_a_corrupt_npz_is_skipped_and_counted_rather_than_crashing_the_run(tmp_path):
    run = str(tmp_path / "corrupt")
    _battle(run, 1_000_000, "aggressive", "win_000", [1.0, 2.0], [0.4, 0.6])
    d = os.path.join(run, "eval_traces", "step_1000000", "aggressive")
    with open(os.path.join(d, "loss_001_summary.json"), "w") as fh:
        json.dump({"meta": {}, "invocations": []}, fh)
    with open(os.path.join(d, "loss_001_states.npz"), "w") as fh:
        fh.write("not an npz")
    _slices, cov = collect_slices(run)
    assert cov["n_traces_read"] == 1 and cov["n_traces_no_npz"] == 1


# ══ 4. The CLI ═══════════════════════════════════════════════════════════════


def test_the_cli_writes_the_json_beside_the_run_and_prints_the_table(tmp_path, capsys):
    run = _build_run(str(tmp_path))
    assert main([run, "--boot", "50", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "SCAFFOLDING GAUGE" in out
    doc = json.loads(open(os.path.join(run, "scaffolding_gauge.json")).read())
    assert doc["tool"] == "scaffolding_gauge"
    assert len(doc["curve"]) == 3 and doc["meta"]["n_boot"] == 50
    assert doc["meta"]["coverage"]["per_opponent"] == {"aggressive": 36}


def test_the_cli_constancy_flag_prints_only_that_row_but_the_json_is_unchanged(tmp_path, capsys):
    run = _build_run(str(tmp_path))
    out_json = str(tmp_path / "c.json")
    assert main([run, "--boot", "50", "--constancy", "--out", out_json, "--quiet"]) == 0
    assert "RANK GAUGE" not in capsys.readouterr().out
    doc = json.loads(open(out_json).read())
    assert "rank_gauge" in doc["curve"][0]                # the full fold is still recorded


def test_the_plot_writes_a_png_or_says_why_it_did_not(tmp_path, report):
    png = str(tmp_path / "gauge.png")
    written = write_plot(report, png)
    if written is None:                                   # matplotlib absent — an allowed outcome
        assert not os.path.exists(png)
    else:
        assert os.path.getsize(png) > 1000


# ══ the RELIABILITY block — opt-in, stratified, and OFF by default ════════════


def _two_class_run(root: str) -> str:
    """One step, two opponent CLASSES, opposite calibration errors.

    `aggressive` (a BOT) is confidently right; `sentinel_0` (the POOL) says the same thing and is
    wrong every time. Pooling the two averages a good forecaster with a terrible one, which is
    precisely the ecology confound the stratification exists to prevent — so the fixture is built
    to make the pooled row uninformative and the two class rows decisive.
    """
    run = os.path.join(root, "run_reliability_fixture")
    for b in range(12):
        won = b % 2 == 0
        name = f"{'win' if won else 'loss'}_{b:03d}"
        p = np.full(8, 0.9 if won else 0.1)
        _battle(run, 5_000_000, "aggressive", name, np.linspace(-1, 1, 8), p)
        # the pool side: the SAME forecasts, the OPPOSITE outcomes.
        flip = f"{'loss' if won else 'win'}_{b:03d}"
        _battle(run, 5_000_000, "sentinel_0", flip, np.linspace(-1, 1, 8), p)
    with open(os.path.join(run, "metadata.json"), "w") as fh:
        json.dump({"git_hash": "cafebabe"}, fh)
    return run


def test_the_reliability_block_is_ABSENT_unless_asked_for(tmp_path, capsys):
    """The default JSON and the default render must not move — every existing consumer of this
    artifact predates the block."""
    run = _build_run(str(tmp_path))
    assert main([run, "--boot", "20", "--out", str(tmp_path / "a.json"), "--quiet"]) == 0
    doc = json.loads(open(str(tmp_path / "a.json")).read())
    assert "reliability" not in doc
    assert "RELIABILITY" not in capsys.readouterr().out


def test_opponent_class_splits_sentinels_from_bots_by_name():
    assert opponent_class("sentinel_0") == "pool"
    assert opponent_class("sentinel_4") == "pool"
    assert opponent_class("aggressive") == "bot"
    assert opponent_class("staller_v2") == "bot"
    assert opponent_class("random") == "bot"


def test_the_two_opponent_CLASSES_are_scored_separately_and_the_pooled_row_hides_both(tmp_path):
    """The stratification's reason for existing, as a measurement: same forecasts, opposite
    outcomes ⇒ one class scores strongly POSITIVE skill and the other strongly negative, while the
    pooled row lands between them and describes NEITHER. Quoted alone it would report a head that
    is near-perfect on bots as a failure — the ecology confound (`win_prob_decomposition.md` axis
    3) in its starkest form."""
    run = _two_class_run(str(tmp_path))
    slices, _ = collect_slices(run)
    blocks = build_reliability(slices, n_boot=60, seed=0)
    assert len(blocks) == 1
    rows = {(r["kind"], r["name"]): r for r in blocks[0]["strata"]}
    bot, pool, pooled = (rows[("class", "bot")]["skill"], rows[("class", "pool")]["skill"],
                         rows[("all", "all")]["skill"])
    assert bot > 0.5 and pool < -1.0
    assert pool < pooled < bot                             # the pooled row describes neither
    assert pooled < 0.0 < bot                              # and it INVERTS the bot verdict
    assert rows[("class", "bot")]["n_battles"] == 12
    assert set(rows) >= {("opponent", "aggressive"), ("opponent", "sentinel_0")}


def test_every_stratum_carries_a_battle_clustered_interval_and_a_battle_count(tmp_path):
    run = _two_class_run(str(tmp_path))
    slices, _ = collect_slices(run)
    for row in build_reliability(slices, n_boot=60, seed=0)[0]["strata"]:
        assert row["n_battles"] >= 1
        assert row["n"] == row["n_battles"] * 8            # 8 decisions per fixture battle
        assert math.isfinite(row["brier_ci_lo"]) and row["brier_ci_lo"] <= row["brier_ci_hi"]


def test_the_cli_flag_adds_the_block_the_units_entry_and_the_rendered_section(tmp_path, capsys):
    run = _two_class_run(str(tmp_path))
    out_json = str(tmp_path / "r.json")
    assert main([run, "--boot", "20", "--reliability", "--reliability-bins", "5",
                 "--out", out_json, "--quiet"]) == 0
    text = capsys.readouterr().out
    assert "(4) RELIABILITY" in text and "SELECTION" in text
    doc = json.loads(open(out_json).read())
    assert doc["reliability"][0]["bins"] == 5
    assert len(doc["reliability"][0]["strata"][0]["table"]) == 5
    # the standing rule: a published number ships what it CANNOT claim.
    assert "cannot" in doc["units"]["reliability"]
    assert "quota" in doc["units"]["reliability"]["cannot"]


def test_constancy_and_reliability_together_keep_the_constancy_only_render(tmp_path, capsys):
    """`--constancy` means 'print only that row'; asking for both must not reintroduce the tables
    it exists to suppress, while the JSON still carries everything computed."""
    run = _two_class_run(str(tmp_path))
    out_json = str(tmp_path / "c.json")
    assert main([run, "--boot", "20", "--constancy", "--reliability",
                 "--out", out_json, "--quiet"]) == 0
    text = capsys.readouterr().out
    assert "RANK GAUGE" not in text and "(4) RELIABILITY" not in text
    assert "reliability" in json.loads(open(out_json).read())


# ══ the SELECTION reweighting — the capture quota is not the deployed population ══


def _quota_run(root: str, *, true_win_rate: float = 0.9, kept_wins: int = 4,
               kept_losses: int = 8, with_results: bool = True) -> str:
    """A run whose eval CYCLE won at `true_win_rate` but whose recorder kept a loss-enriched slice.

    The head says 0.9 on every decision — right for the cycle, and badly wrong for the capture.
    That is the measured shape on `ai_v9_59_R2ACTION_0827` (captured 0.46 vs a recorded 0.90)
    reduced to a fixture with a known answer.
    """
    run = os.path.join(root, "run_quota_fixture")
    for b in range(kept_wins):
        _battle(run, 7_000_000, "heuristic", f"win_{b:03d}", np.zeros(6), np.full(6, 0.9))
    for b in range(kept_losses):
        _battle(run, 7_000_000, "heuristic", f"loss_{b:03d}", np.zeros(6), np.full(6, 0.9))
    if with_results:
        with open(os.path.join(run, "eval_results.jsonl"), "w") as fh:
            fh.write(json.dumps({"step": 7_000_000, "bots": {"heuristic": true_win_rate},
                                 "sentinels": []}) + "\n")
    with open(os.path.join(run, "metadata.json"), "w") as fh:
        json.dump({"git_hash": "cafebabe"}, fh)
    return run


def test_true_win_rates_reads_the_runs_own_recorded_per_opponent_rates(tmp_path):
    run = _quota_run(str(tmp_path))
    assert true_win_rates(run) == {7_000_000: {"heuristic": 0.9}}


def test_a_run_with_no_eval_results_REFUSES_rather_than_silently_going_unweighted(tmp_path):
    """The refusal is the point: an unweighted table looks identical and answers a different
    question, so a fall-back here would publish the wrong population under the right label."""
    run = _quota_run(str(tmp_path), with_results=False)
    with pytest.raises(SelectionWeightError, match="eval_results.jsonl"):
        true_win_rates(run)


def test_the_reweighting_moves_the_base_rate_onto_the_cycles_and_fixes_the_verdict(tmp_path):
    run = _quota_run(str(tmp_path))
    slices, _ = collect_slices(run)
    raw = build_reliability(slices, n_boot=40, seed=0)[0]
    rw = build_reliability(slices, n_boot=40, seed=0, reweight=true_win_rates(run))[0]
    raw_all = next(r for r in raw["strata"] if r["kind"] == "all")
    rw_all = next(r for r in rw["strata"] if r["kind"] == "all")

    assert raw["reweighted"] is False and rw["reweighted"] is True
    assert raw_all["base_rate"] == pytest.approx(4 / 12)          # the capture quota's
    assert rw_all["base_rate"] == pytest.approx(0.9)              # the eval cycle's
    # The verdict flips on RELIABILITY, which is the meter that answers "is this head
    # calibrated": ~0.32 on the quota, ~0 on the population. Brier does NOT go to zero — a 0.9
    # forecaster on a 0.9 population carries an irreducible 0.09 — and asserting that it would
    # is the trap this comment exists to mark.
    assert raw_all["reliability"] > 0.3
    assert rw_all["reliability"] == pytest.approx(0.0, abs=1e-12)
    assert rw_all["brier"] == pytest.approx(0.09)          # = the slice's own uncertainty
    assert rw_all["skill"] == pytest.approx(0.0, abs=1e-12)
    assert rw_all["ess"] < rw_all["n"]                            # the correction costs sample


def test_weights_are_CONSTANT_within_a_battle_so_the_cluster_bootstrap_stays_valid(tmp_path):
    """The reweighting must not break the clustering the intervals depend on."""
    run = _quota_run(str(tmp_path))
    slices, _ = collect_slices(run)
    s = slices[7_000_000]
    w, _ = selection_weights(s["outcomes"], s["battles"], s["opponents"], {"heuristic": 0.9})
    for b in set(s["battles"].tolist()):
        assert len(set(np.round(w[s["battles"] == b], 12).tolist())) == 1


def test_an_opponent_with_only_one_captured_OUTCOME_CLASS_is_zeroed_and_NAMED(tmp_path):
    """Nothing to reweight ⇒ weight 0 and a named reason. Leaving it at weight 1 would mix a
    corrected population with an uncorrected one and label the whole thing corrected."""
    run = _quota_run(str(tmp_path), kept_losses=0)
    slices, _ = collect_slices(run)
    s = slices[7_000_000]
    with pytest.raises(SelectionWeightError, match="no usable opponent"):
        selection_weights(s["outcomes"], s["battles"], s["opponents"], {"heuristic": 0.9})


def test_an_opponent_with_no_recorded_true_rate_is_zeroed_and_named_not_silently_kept(tmp_path):
    run = _quota_run(str(tmp_path))
    slices, _ = collect_slices(run)
    s = slices[7_000_000]
    with pytest.raises(SelectionWeightError, match="no usable opponent"):
        selection_weights(s["outcomes"], s["battles"], s["opponents"], {})


def test_the_cli_reweight_flag_labels_the_render_and_records_the_selection_block(tmp_path, capsys):
    run = _quota_run(str(tmp_path))
    out_json = str(tmp_path / "rw.json")
    assert main([run, "--boot", "20", "--reliability", "--reliability-reweight",
                 "--out", out_json, "--quiet"]) == 0
    text = capsys.readouterr().out
    assert "SELECTION-REWEIGHTED" in text and "RAW capture quota" not in text
    doc = json.loads(open(out_json).read())
    sel = doc["reliability"][0]["selection"]["per_opponent"]["heuristic"]
    assert sel["true_win_rate"] == 0.9
    assert sel["captured_win_rate"] == pytest.approx(4 / 12)
    assert sel["n_battles"] == 12


def test_WITHOUT_the_reweight_flag_the_render_says_so_in_the_header(tmp_path, capsys):
    """The label is not decoration — the two tables answer different questions and a reader who
    cannot tell them apart will quote the wrong one."""
    run = _quota_run(str(tmp_path))
    assert main([run, "--boot", "20", "--reliability",
                 "--out", str(tmp_path / "raw.json"), "--quiet"]) == 0
    assert "RAW capture quota — NOT the deployed population" in capsys.readouterr().out
