"""`maturity_rules`: R-MATURITY and the one-network gate as code, on planted effects. Pure numpy."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from main.hidden_info_study import maturity_rules as R
from utils.paths import repo_path

T1 = repo_path("designs", "research_state", "measurements", "hidden_info_tier1_2026-10-13")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"{name}_for_rules_test", T1 / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _bank(n_battles=150, per=2):
    return [f"b{i:03d}" for i in range(n_battles) for _ in range(per)]


def test_boot_reproduces_tier1_draws_exactly():
    ST = _load("t1_stats")
    battles = _bank(97, 3)[:-1]
    x = np.random.default_rng(1).normal(size=len(battles))
    x[5] = np.nan
    sel = np.arange(len(x)) % 4 != 0
    mine, theirs = R.Boot(battles), ST.Boot(battles)
    assert np.array_equal(mine.C, theirs.C)
    e1, v1 = mine.mean(x, sel)
    e2, v2 = theirs.mean(x, sel)
    assert e1 == e2 and np.array_equal(v1, v2)
    assert R.ci(v1) == ST.ci(v1) and R.ci(v1, R.BONF_LEVEL) == ST.ci(v1, 99.6)


def test_eq_p_matches_tier1():
    grid = np.round(np.linspace(0.0, 1.0, 21), 2)
    curve = np.linspace(0.01, 0.08, 21)
    assert R.eq_p(curve, 0.0, grid) == 0.0
    assert R.eq_p(curve, 0.09, grid) is None
    p = R.eq_p(curve, 0.045, grid)
    assert p == pytest.approx(0.5)
    flat = np.array([0.0] * 10 + [0.05] * 11)
    assert R.eq_p(flat, 0.05, grid) == pytest.approx(0.5)
    src = (T1 / "unblind_t1.py").read_text()
    assert "return 0.0" in src and "if target <= curve[0]:" in src      # the same convention, by construction


def test_today_clamps_to_unit_interval():
    grid = np.array([0.0, 0.5, 1.0])
    curves = np.array([[0.0, 0.1, 0.2]] * 3)
    pt, vec = R.today_clamped(curves[0], 0.05, curves, np.array([-1.0, 0.5, np.nan]), grid)
    assert pt == pytest.approx(0.25)
    assert vec[0] == 0.0 and vec[1] == 1.0 and np.isnan(vec[2])


def _read(label, step, boot, g_by_state, today=None, gates=None):
    return R.CheckpointRead(label=label, step=step, g={"all": boot.mean(g_by_state)}, today=today, gates=gates or {})


def test_gap_shrinks_grows_and_null_on_planted_series():
    battles = _bank()
    bt = R.Boot(battles)
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.18, size=len(battles))
    g0 = 0.07 + noise
    shrink = R.r_maturity([_read("a", 25, bt, g0), _read("b", 225, bt, g0 - 0.05 + rng.normal(0, 0.05, len(g0)))])
    assert shrink["GAP"]["all"]["verdict"] == "GAP-SHRINKS"
    grow = R.r_maturity([_read("a", 25, bt, g0), _read("b", 225, bt, g0 + 0.05)])
    assert grow["GAP"]["all"]["verdict"] == "GAP-GROWS"
    same = R.r_maturity([_read("a", 25, bt, g0), _read("b", 100, bt, g0 + 1e-4), _read("c", 225, bt, g0)])
    assert same["GAP"]["all"]["verdict"] == "NOT DETECTED"
    assert same["CONVERGED"]["verdict"] == "CONVERGED" and same["CONVERGED"]["between"] == ["b", "c"]
    assert same["first"] == "a" and same["last"] == "c"
    # order of the input does not matter (sorted by step)
    assert R.r_maturity([_read("c", 225, bt, g0), _read("a", 25, bt, g0 - 0.05)])["GAP"]["all"]["verdict"] \
        == "GAP-GROWS"


def test_gap_change_boundary_is_not_clear():
    vec = np.full(R.B, -0.0004)               # an upper bound inside the ±0.0005 band
    assert R.gap_change((0.0, np.zeros(R.B)), (-0.0004, vec))["verdict"] == "NOT DETECTED"
    vec = np.full(R.B, -0.0006)
    assert R.gap_change((0.0, np.zeros(R.B)), (-0.0006, vec))["verdict"] == "GAP-SHRINKS"


def test_belief_rises_only_when_clear():
    base = (0.02, np.full(R.B, 0.02))
    assert R.belief_change(base, (0.1, np.full(R.B, 0.1)))["verdict"] == "BELIEF-RISES"
    assert R.belief_change(base, (0.0204, np.full(R.B, 0.0204)))["verdict"] == "NOT DETECTED"
    assert R.belief_change(base, (0.0, np.zeros(R.B)))["verdict"] == "NOT DETECTED"


def test_gate_status_thresholds():
    strong = np.full(R.B, 0.03)
    weak = np.concatenate([np.full(R.B - 10, 0.03), np.full(10, -0.01)])     # 95 % lower clear, 99.6 % lower not
    assert R.gate_status(160, strong, strong) == "GATE-OPEN"
    assert R.gate_status(149, strong, strong) == "GATE-WEAK"                 # n below 150 can never open
    assert R.gate_status(400, weak, strong) == "GATE-WEAK"
    assert R.gate_status(400, np.zeros(R.B), strong) == "GATE-CLOSED"


def test_frontier_moves_recedes_mixed_static():
    closed = {ax: {b: "GATE-CLOSED" for b in bs} for ax, (_f, bs) in R.AXES.items()}

    def with_open(**kw):
        g = {ax: dict(v) for ax, v in closed.items()}
        for ax, b in kw.items():
            g[ax][b] = "GATE-OPEN"
        return g

    assert R.frontier_change(closed, closed)["verdict"] == "FRONTIER-STATIC"
    assert R.frontier_change(closed, with_open(k="6"))["verdict"] == "FRONTIER-MOVES-EARLIER"   # from none
    assert R.frontier_change(with_open(k="6"), with_open(k="5"))["verdict"] == "FRONTIER-MOVES-EARLIER"
    assert R.frontier_change(with_open(k="5"), with_open(k="6"))["verdict"] == "FRONTIER-RECEDES"
    # material: MORE material is earlier in the game
    assert R.frontier_change(with_open(material="<=7"), with_open(material="12"))["verdict"] \
        == "FRONTIER-MOVES-EARLIER"
    assert R.frontier_change(with_open(k="6", turn="8-14"), with_open(k="4", turn=">=22"))["verdict"] \
        == "FRONTIER-MIXED"
    weak = {ax: dict(v) for ax, v in closed.items()}
    weak["k"]["6"] = "GATE-WEAK"
    assert R.frontier_change(closed, weak)["verdict"] == "FRONTIER-STATIC"     # weak is not open


def test_r_maturity_needs_two_distinct_anchors():
    bt = R.Boot(_bank(30))
    g = np.zeros(60)
    with pytest.raises(ValueError):
        R.r_maturity([_read("a", 25, bt, g)])
    with pytest.raises(ValueError):
        R.r_maturity([_read("a", 25, bt, g), _read("b", 25, bt, g)])
    out = R.r_maturity([_read("a", 25, bt, g), _read("b", 50, bt, g)])
    assert out["BELIEF"]["verdict"] == "NOT READ" and out["FRONTIER"]["verdict"] == "NOT READ"


# ------------------------------------------------------------------------------------------------
# amendment A1 (2026-10-19): CONVERGED NOT READ on one bank, STILL-MOVING, R3 at one anchor, BANK-DEPENDENT
# ------------------------------------------------------------------------------------------------

def test_a1_converged_not_read_on_one_bank_and_still_moving():
    battles = _bank()
    bt = R.Boot(battles)
    g0 = 0.07 + np.random.default_rng(7).normal(0, 0.18, size=len(battles))
    reads = [_read("a", 25, bt, g0), _read("b", 100, bt, g0 + 1e-4), _read("c", 225, bt, g0)]
    same = R.r_maturity(reads, converged_on_this_bank=False)
    assert same["CONVERGED"]["verdict"].startswith("NOT READ (amendment A1")       # the registered rule would say CONVERGED
    assert R.r_maturity(reads)["CONVERGED"]["verdict"] == "CONVERGED"
    assert same["STILL-MOVING"]["verdict"] == "NOT DETECTED" and same["STILL-MOVING"]["between"] == ["b", "c"]
    moving = R.r_maturity([_read("a", 25, bt, g0), _read("b", 100, bt, g0), _read("c", 225, bt, g0 - 0.05)],
                          converged_on_this_bank=False)
    assert moving["STILL-MOVING"]["verdict"] == "STILL-SHRINKING"
    up = R.r_maturity([_read("b", 100, bt, g0), _read("c", 225, bt, g0 + 0.05)])
    assert up["STILL-MOVING"]["verdict"] == "STILL-GROWING"


def _r3_imps(n, rng, u_rev, u_hide):
    """Per-state IMPs with a planted reveal-only (Δs) and hide-only (Δn) U margin."""
    f = rng.normal(0.0, 0.1, n)
    o = f + 0.07 + rng.normal(0, 0.02, n)
    return {"O": o, "F": f, "F+M": f + 0.01 + rng.normal(0, 0.01, n), "F+U": f + 0.01 + u_rev + rng.normal(0, 0.01, n),
            "O-U": o - 0.03 - u_hide + rng.normal(0, 0.01, n), "O-M": o - 0.03 + rng.normal(0, 0.01, n)}


def test_r3_read_verdicts_on_planted_margins():
    battles = _bank(150, 2)
    bt = R.Boot(battles)
    rng = np.random.default_rng(3)
    assert R.r3_read(bt, _r3_imps(len(battles), rng, 0.02, 0.02))["verdict"] == "U-FIRST"
    assert R.r3_read(bt, _r3_imps(len(battles), rng, -0.02, -0.02))["verdict"] == "M-FIRST"
    shared = R.r3_read(bt, _r3_imps(len(battles), rng, 0.02, -0.02))
    assert shared["verdict"] == "SHARED" and shared["label"] == "SHARED (SIDES-DISAGREE)"
    x = _r3_imps(len(battles), rng, 0.0, 0.0)
    x = {k: (v if k in ("O", "F") else x["F"] if k.startswith("F") else x["O"]) for k, v in x.items()}
    assert R.r3_read(bt, x)["verdict"] == "NOT DETECTED"                 # no class detected, no margin
    sel = np.arange(len(battles)) % 2 == 0
    assert R.r3_read(bt, _r3_imps(len(battles), rng, 0.02, 0.02), sel)["reveal_only Δs = IMP(F+U) - IMP(F+M)"]["n"] \
        == int(sel.sum())


def test_r3_read_equals_e2s_registered_read():
    """The plateau battery's R3 is E2's: on the same per-state IMPs, E2's ``read_frame`` (one checkpoint, Tier 1's
    unstratified Boot) gives the same verdict, qualifiers and numerators."""
    import sys

    e2 = repo_path("designs", "research_state", "measurements", "hidden_info_e2_competent_2026-10-16")
    sys.path.insert(0, str(e2))
    try:
        spec = importlib.util.spec_from_file_location("read_e2_for_rules_test", e2 / "read_e2.py")
        RE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(RE)
    finally:
        sys.path.remove(str(e2))
    battles = _bank(150, 2)
    rng = np.random.default_rng(11)
    for u_rev, u_hide in ((0.02, 0.02), (0.012, -0.002), (-0.015, 0.0)):
        x = _r3_imps(len(battles), rng, u_rev, u_hide)
        x["O-M"][4] = np.nan                                              # a hide-only arm not ok on one state
        recs = [{"battle": b, "cp": "c", "src": {s: ({"status": "ok", "imp": {"Z": float(v[i])}}
                                                     if np.isfinite(v[i]) else {"status": "excluded"})
                                                 for s, v in x.items()}} for i, b in enumerate(battles)]
        e2r = RE.read_frame(recs, False, "Z")
        ours = R.r3_read(R.Boot(battles), x)
        assert ours["verdict"] == e2r["verdict"] and ours["label"] == e2r["label"]
        assert ours["qualifiers"] == e2r["qualifiers"]
        for a, b in (("reveal_only Δs = IMP(F+U) - IMP(F+M)", "reveal_only Δs = IMP(F+U) − IMP(F+M)"),
                     ("hide_only Δn = IMP(O-M) - IMP(O-U)", "hide_only Δn = IMP(O−M) − IMP(O−U)")):
            assert ours[a]["mean"] == pytest.approx(e2r[b]["mean"], abs=1e-5)
            assert ours[a]["ci"] == pytest.approx(e2r[b]["ci"], abs=1e-5)


def test_bank_dependent_is_the_registered_disagreement():
    assert R.bank_dependent("GAP", "GAP-SHRINKS", "GAP-GROWS")
    assert not R.bank_dependent("GAP", "GAP-SHRINKS", "NOT DETECTED")
    assert not R.bank_dependent("GAP", "GAP-SHRINKS", "GAP-SHRINKS")
    assert R.bank_dependent("BELIEF", "BELIEF-RISES", "NOT DETECTED")
    assert R.bank_dependent("BELIEF", "NOT DETECTED", "BELIEF-RISES")
    assert not R.bank_dependent("BELIEF", "NOT DETECTED", "NOT DETECTED")
    assert R.bank_dependent("R3", "U-FIRST", "M-FIRST") and not R.bank_dependent("R3", "U-FIRST", "SHARED")
    assert R.bank_dependent("FRONTIER", "FRONTIER-MOVES-EARLIER", "FRONTIER-RECEDES")
    assert not R.bank_dependent("FRONTIER", "NOT READ (LEAN)", "FRONTIER-RECEDES")
    with pytest.raises(ValueError):
        R.bank_dependent("TODAY", "a", "b")
