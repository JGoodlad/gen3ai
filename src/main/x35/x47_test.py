"""X47's readout (``main.x35.x47``, B-X47-READ, AM-4): effective learn time along the surviving path, the s_eq node rule
with rule 8, ρ's coincidence, and every verdict class. Constructed rows only (the loader on a tiny written events
file); every constructed number sits well clear of its boundary except the deliberate rule-8 cases."""
from __future__ import annotations

import json

import pytest

from main.x35 import readout as R
from main.x35 import x47 as X

FORK = 1_000
IT = 100          # steps per iteration in the constructed runs


def _seg(start: int, n: int, collect_ms: float = 1000.0, update_s: float = 1.0):
    """One process: collect rows at start+IT .. start+n*IT; update rows lag one dump (from the 2nd row)."""
    c = [(start + IT * (i + 1), collect_ms) for i in range(n)]
    u = [(start + IT * (i + 1), update_s) for i in range(1, n)]
    return c, u


def _run(name: str, segs, nodes=(), fork: int = FORK) -> X.LearnTime:
    col, upd = X.surviving_path(segs, fork)
    return X.LearnTime(name, fork, tuple(col), tuple(upd), tuple(nodes))


# ------------------------------------------------------------------------------------ effective learn time
def test_a_single_process_sums_both_tags_over_their_own_rows():
    r = _run("c0", [_seg(FORK, 10)])
    assert r.final_step == FORK + 10 * IT and r.sigma() == pytest.approx(10 * 1.0 + 9 * 1.0, abs=1e-12)
    assert r.sigma(FORK + 5 * IT) == pytest.approx(5 + 4, abs=1e-12)


def test_a_restart_does_not_count_the_redone_span_twice():
    # process 1 runs to FORK+10*IT; it is restarted from a checkpoint at FORK+6*IT and runs to FORK+12*IT
    p1 = _seg(FORK, 10)
    p2 = _seg(FORK + 6 * IT, 6)
    r = _run("c0", [p1, p2])
    steps = [s for s, _ in r.collect]
    assert steps == sorted(set(steps)) and len(steps) == 12               # every iteration counted ONCE
    assert r.sigma() == pytest.approx(12 * 1.0 + (5 + 5) * 1.0, abs=1e-12)   # p1's updates <= 6*IT (5) + p2's (5)
    # reverting the supersession would count p1's rows at 7..10 too
    naive = sum(v for s, v in p1[0]) / 1000 + sum(v for s, v in p1[1]) + sum(v for s, v in p2[0]) / 1000 + \
        sum(v for s, v in p2[1])
    assert naive - r.sigma() > 5.0


def test_rows_at_or_below_the_fork_step_are_excluded():
    inherited = ([(FORK - 50, 9e6), (FORK, 9e6)], [(FORK, 9e3)])
    r = _run("c0", [inherited, _seg(FORK, 4)])
    assert r.sigma() == pytest.approx(4 + 3, abs=1e-12)


def test_a_missing_instrument_is_inconclusive_never_zero():
    c, _ = _seg(FORK, 6)
    with pytest.raises(X.X47InputError, match="more than the one-row dump lag"):
        X.surviving_path([(c, [])], FORK)
    with pytest.raises(X.X47InputError, match="no rollout/collect_ms row"):
        X.surviving_path([([], [(FORK + IT, 1.0)])], FORK)
    with pytest.raises(X.X47InputError, match="non-finite"):
        X.surviving_path([([(FORK + IT, float("nan")), (FORK + 2 * IT, 1.0)], [(FORK + 2 * IT, 1.0)])], FORK)


# --------------------------------------------------------------------------------------------- the node rule
def test_s_eq_is_the_largest_node_under_the_budget_and_rule8_takes_the_next_lower():
    r = _run("x47", [_seg(FORK, 20)], nodes=[FORK + 5 * IT, FORK + 10 * IT, FORK + 15 * IT])
    sig10 = r.sigma(FORK + 10 * IT)                                      # 10 + 9 = 19 s
    assert X.s_eq(r, sig10 + 1e-6).node == FORK + 10 * IT
    tie = X.s_eq(r, sig10 + 1e-12)                                      # within 1e-9 of B: NOT taken
    assert tie.node == FORK + 5 * IT and tie.skipped_rule8 == (FORK + 10 * IT,)
    assert X.s_eq(r, sig10 - 1e-6).node == FORK + 5 * IT
    assert X.s_eq(r, 1.0).node is None                                   # no node qualifies
    fin = X.s_eq(r, r.sigma() + 5.0)
    assert fin.node == r.final_step and fin.is_final


def test_rho_coincidence_reads_the_finals_and_rule8_at_1_02():
    c0 = [_run("c0", [_seg(FORK, 20)])]
    slow = [_run("x47", [_seg(FORK, 20, collect_ms=1400.0, update_s=1.4)], nodes=[FORK + 10 * IT])]
    comp = X.compare(c0, slow)
    assert comp.rho == pytest.approx(1.4, abs=1e-12) and not comp.coincide
    assert comp.seq[0].node == FORK + 10 * IT and not comp.seq[0].is_final
    fast = [_run("x47", [_seg(FORK, 20, collect_ms=1010.0, update_s=1.01)], nodes=[FORK + 10 * IT])]
    comp = X.compare(c0, fast)
    assert comp.coincide and comp.seq[0].is_final                         # ρ 1.01 <= 1.02
    # ρ within 1e-9 of 1.02 counts as ABOVE: no coincidence, the node rule runs
    near = [_run("x47", [_seg(FORK, 20, collect_ms=1020.0, update_s=1.02)], nodes=[FORK + 10 * IT])]
    comp = X.compare(c0, near)
    assert comp.near_rho and not comp.coincide
    assert set(comp.as_dict()["reported_only"]) == {"rho_collect", "rho_update"}


def test_the_loader_reads_the_runs_own_events_and_skips_the_inherited_file(tmp_path):
    from torch.utils.tensorboard import SummaryWriter

    run = tmp_path / "rb_x35t_x47d4_s3601"
    tb = run / "tb"
    inh = tmp_path / "inh"
    w = SummaryWriter(log_dir=str(inh))
    w.add_scalar(X.COLLECT_TAG, 9e9, FORK)
    w.add_scalar(X.UPDATE_TAG, 9e9, FORK)
    w.close()
    tb.mkdir(parents=True)
    next(inh.glob("*tfevents*")).rename(tb / "events.out.tfevents.0000000000.inherited.0.0")
    (tb / "INHERITED_FROM.json").write_text(json.dumps({"fork_step": FORK}))
    w = SummaryWriter(log_dir=str(tb))
    c, u = _seg(FORK, 5)
    for s, v in c:
        w.add_scalar(X.COLLECT_TAG, v, s)
    for s, v in u:
        w.add_scalar(X.UPDATE_TAG, v, s)
    w.close()
    (run / "checkpoints").mkdir()
    for n in (FORK - 100, FORK + 300):
        (run / "checkpoints" / f"checkpoint_{n}_steps.zip").write_bytes(b"z")
    r = X.load_run(str(run))
    assert r.fork_step == FORK and r.nodes == (FORK + 300,) and r.sigma() == pytest.approx(5 + 4, abs=1e-9)


# ---------------------------------------------------------------------------------------------- the verdict
def _s(xs, base: int = 3601):
    return {str(base + i): x for i, x in enumerate(xs)}


OFF = [50.0, 50.4, 49.6]
GAIN_ON = [60.0, 60.4, 59.6]                       # Δ̂ +10, tight: GAIN
NULL_ON = [50.1, 50.5, 49.7]                       # Δ̂ +0.1, tight: NULL
WIDE_POS = [53.0, 58.0, 49.0]                      # Δ̂ > 0, wide: UNDECIDED -> CONTINUE at look 1
LOSS_ON = [40.0, 40.4, 39.6]


def _doc(look=1, s_eq=GAIN_ON, final=GAIN_ON, off=OFF, **extra):
    n = len(off)
    d = {"look": look, "c0_final": _s(off), "x47_s_eq": _s(s_eq), "x47_final": _s(final) if final is not None else None,
         "g_f": {"h_pp": [[50.0 + ai - bj for bj in (0.5, -0.5, 0.0, 0.3, -0.3)[:n]]
                          for ai in (1.0, -1.0, 0.0, 0.6, -0.6)[:n]]},     # additive, level: G-F PASSes
         "g_c": {"b1": [0.0, -0.02, 0.02], "b2": [0.0, -0.01, 0.01]}}
    d.update(extra)
    return d


def test_each_verdict_class():
    assert X.read(_doc()).verdict == R.ADOPT
    assert X.read(_doc(g_c={"b1": [-0.2, -0.25, -0.15], "b2": [0.0, -0.01, 0.01]})).verdict == R.TRADE
    assert X.read(_doc(g_c=None)).verdict == R.INCONCLUSIVE                   # a required guard not read
    assert X.read(_doc(s_eq=NULL_ON, final=GAIN_ON)).verdict == X.PER_STEP_ONLY
    assert X.read(_doc(s_eq=NULL_ON, final=NULL_ON)).verdict == R.NULL
    assert X.read(_doc(s_eq=LOSS_ON, final=GAIN_ON)).verdict == R.LOSS
    v = X.read(_doc(s_eq=WIDE_POS, final=GAIN_ON))
    assert v.verdict == R.CONTINUE and v.as_dict()["arm"] == "X47" and v.as_dict()["look"] == 1
    assert X.read(_doc(s_eq={"3601": 1.0})).verdict == R.INCONCLUSIVE        # wrong seed count
    assert X.read(_doc(s_eq=NULL_ON, final=None)).verdict == R.INCONCLUSIVE  # PER-STEP ONLY needs the secondary


def test_look_2_reads_five_seeds_and_is_final():
    off5 = OFF + [50.2, 49.8]
    v = X.read(_doc(look=2, s_eq=WIDE_POS + [51.0, 55.0], final=WIDE_POS + [51.0, 55.0], off=off5))
    assert v.verdict in (R.UNDECIDED, R.GAIN) and v.primary.final
    assert X.read(_doc(look=2, s_eq=GAIN_ON + [60.2, 59.8], final=GAIN_ON + [60.2, 59.8], off=off5)).verdict == R.ADOPT
    assert X.read(_doc(look=2)).verdict == R.INCONCLUSIVE                    # 3 seeds at look 2


def test_the_cli_writes_the_verdict(tmp_path, capsys):
    f = tmp_path / "read.json"
    f.write_text(json.dumps(_doc(s_eq=NULL_ON, final=GAIN_ON)))
    assert X._main(["decide", str(f)]) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == X.PER_STEP_ONLY
