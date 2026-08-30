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

from main.scaffolding_gauge import build_report, collect_slices, main, render, write_plot

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
