"""main.endofrun — the pre-registered verdict rules, pinned on constructed inputs.

The runner exists so the runbook rules get applied MECHANICALLY; these tests are what keep the
mechanical application equal to the registered rules (tail-K reading, the §5 non-inferiority
margins, the §2 route-deletion ratio, the family-alive bar, the coverage-confound handling)."""
import pytest

from main.endofrun import (
    awareness_verdicts, family_verdicts, non_inferiority, render_markdown, route_verdicts,
    tail_rating,
)


def _curve(*elos, se=8.0):
    return [(i * 1000, e, se) for i, e in enumerate(elos)]


def test_tail_rating_reads_the_last_k_and_refuses_short_curves():
    t = tail_rating(_curve(2000, 2010, 2020, 2030, 2040, 2050))
    assert t["steps"] == [2000, 3000, 4000, 5000]
    assert t["elo"] == pytest.approx((2020 + 2030 + 2040 + 2050) / 4)
    assert tail_rating(_curve(2000, 2010)) is None       # under-sampled ⇒ no mid-run reading


def test_non_inferiority_three_verdicts():
    cur = {"elo": 2100.0, "se": 5.0}
    assert non_inferiority(cur, {"elo": 2105.0, "se": 5.0})["verdict"] == "NON_INFERIOR"
    # a big deficit with a tight CI entirely below the margin
    assert non_inferiority({"elo": 2040.0, "se": 3.0},
                           {"elo": 2105.0, "se": 3.0})["verdict"] == "INFERIOR"
    # wide CI straddling the hard bound
    wide = non_inferiority({"elo": 2085.0, "se": 20.0}, {"elo": 2105.0, "se": 20.0})
    assert wide["verdict"] == "INCONCLUSIVE"
    assert non_inferiority(None, cur)["verdict"] == "UNAVAILABLE"


def test_route_verdicts_ratio_rule_and_confound():
    arms = {
        "all_off": {"kl_mean": 0.1, "kl_p95": 0.3, "flip_rate": 0.05, "dv_mean": 1.00},
        "seed": {"kl_mean": 0.0, "kl_p95": 0.0, "flip_rate": 0.001, "dv_mean": 0.05},   # dead
        "threat": {"kl_mean": 0.0, "kl_p95": 0.0, "flip_rate": 0.001, "dv_mean": 0.60}, # live
        "hidden_opp_vf": {"kl_mean": 0.0, "kl_p95": 0.0, "flip_rate": 0.05,             # flips
                          "dv_mean": 0.05},
        "nmr": {"kl_mean": 0.02, "kl_p95": 0.1, "flip_rate": 0.01, "dv_mean": 0.2},
    }
    out = route_verdicts(arms)
    assert out["per_route"]["seed"]["verdict"] == "DELETION_CANDIDATE"
    assert out["per_route"]["threat"]["verdict"] == "KEEP"          # |dV| ratio too high
    assert out["per_route"]["hidden_opp_vf"]["verdict"] == "KEEP"   # flips too high
    assert out["per_route"]["nmr"]["verdict"] == "READ"             # informational arm
    # 0.05+0.6+0.05 = 0.70 ≥ 60% of the joint ⇒ no confound note
    assert not out["notes"]
    arms2 = dict(arms, threat={"kl_mean": 0, "kl_p95": 0, "flip_rate": 0.0, "dv_mean": 0.1})
    out2 = route_verdicts(arms2)
    assert any("CONFOUND" in n for n in out2["notes"])              # shared-content warning


def test_family_verdicts_half_median_bar():
    fams = {
        "d1": {"kl_mean": 0.01, "flip_rate": 0.01, "dv_mean": 0.40},
        "d3": {"kl_mean": 0.01, "flip_rate": 0.01, "dv_mean": 0.20},
        "v": {"kl_mean": 0.01, "flip_rate": 0.01, "dv_mean": 0.30},
        "h": {"kl_mean": 0.001, "flip_rate": 0.0, "dv_mean": 0.16},
    }
    out = family_verdicts(fams)
    assert out["h"]["verdict"] == "ALIVE"      # 0.16 >= 0.5 * median(0.4, 0.2, 0.3)=0.15
    fams["h"]["dv_mean"] = 0.10
    assert family_verdicts(fams)["h"]["verdict"] == "NULL"
    assert family_verdicts({"d1": {"dv_mean": 1.0}}, targets=("h",))["h"]["verdict"] == "ABSENT"


def test_awareness_verdict_directions():
    loss = {"blind_loss_fraction": 0.05, "median_lead_time": 9.0,
            "cap_aware_ge_bar_fraction": 0.7}
    allo = {"quantile_coverage": {"coverage80": 0.61, "pit_mean": 0.47}}
    v = awareness_verdicts(loss, allo)
    assert v["blind_loss_fraction"]["verdict"] == "IMPROVED"    # fewer blind losses
    assert v["median_lead_time"]["verdict"] == "IMPROVED"       # more warning
    assert v["coverage80"]["verdict"] == "IMPROVED"             # toward 0.80
    assert v["pit_mean"]["verdict"] == "IMPROVED"               # toward 0.5
    worse = awareness_verdicts({"blind_loss_fraction": 0.2, "median_lead_time": 3.0,
                                "cap_aware_ge_bar_fraction": 0.2},
                               {"quantile_coverage": {"coverage80": 0.30, "pit_mean": 0.30}})
    assert all(row["verdict"] == "WORSE" for row in worse.values())


def test_report_renders_every_step_status():
    report = {
        "run": "test_run", "run_dir": "/x", "ref": None,
        "steps": {
            "elo": {"status": "ok",
                    "current_tail": {"elo": 2100.0, "se": 5.0, "k": 4, "steps": []},
                    "reference_tail": None,
                    "non_inferiority": {"verdict": "UNAVAILABLE", "why": "no --ref"},
                    "curve_len": 9},
            "audits": {"status": "needs_pinned_tree", "why": "arch drift",
                       "how": ["git worktree add ..."]},
            "awareness": {"status": "ok", "loss_aggregate": {}, "all_aggregate": {},
                          "verdicts": {"coverage80": {"current": 0.5,
                                                      "gen10_baseline": 0.44,
                                                      "verdict": "IMPROVED"}}},
        },
    }
    md = render_markdown(report)
    assert "2100.0" in md and "UNAVAILABLE" in md
    assert "needs_pinned_tree" in md and "git worktree add" in md
    assert "coverage80" in md and "IMPROVED" in md
