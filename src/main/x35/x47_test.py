"""X47's readout (``main.x35.x47``, B-X47-READ, AM-4): effective learn time along the surviving path, the s_eq node rule
with rule 8, ρ's coincidence, and every verdict class. Constructed rows only (the loader on a tiny written events
file); every constructed number sits well clear of its boundary except the deliberate rule-8 cases."""
from __future__ import annotations

import json

import pytest

from main.x35 import gc as GC
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
    assert set(comp.as_dict()["reported_only"]) == {"rho_collect", "rho_update", "eval_share"}


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


def _gc(n: int, b1: float = 0.0, b2: float = 0.0):
    """A seed-pooled G-C record (AM-X47b-1) over the toy seeds 3601.. of constructed per-turn deltas."""
    return GC.constructed_record("toy", [3601 + i for i in range(n)], [b1] * n, [b2] * n)


def _doc(look=1, s_eq=GAIN_ON, final=GAIN_ON, off=OFF, **extra):
    n = len(off)
    d = {"look": look, "c0_final": _s(off), "x47_s_eq": _s(s_eq), "x47_final": _s(final) if final is not None else None,
         "g_f": {"h_pp": [[50.0 + ai - bj for bj in (0.5, -0.5, 0.0, 0.3, -0.3)[:n]]
                          for ai in (1.0, -1.0, 0.0, 0.6, -0.6)[:n]]},     # additive, level: G-F PASSes
         "g_c": _gc(n)}
    d.update(extra)
    return d


def test_each_verdict_class():
    assert X.read(_doc()).verdict == R.ADOPT
    assert X.read(_doc(g_c=_gc(3, b1=-0.2))).verdict == R.TRADE
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


# ------------------------------------------------------------------ the AM-X47 amendments (2026-10-20, before any read)
from main.x35 import x47_report as XR  # noqa: E402

TARGET = FORK + 10 * IT


def _finished_dir(tmp_path, name: str, *, final_step: int, steps: int = TARGET, k9b: str = "PASS",
                  final: bool = True, nt=None, every: int = IT):
    d = tmp_path / name
    (d / "checkpoints").mkdir(parents=True)
    if final:
        (d / "final_model.zip").write_bytes(b"z")
    (d / "metadata.json").write_text(json.dumps({"num_timesteps": final_step if nt is None else nt,
                                                 "cli_args": {"steps": steps, "checkpoint_every_steps": every}}))
    stage = tmp_path / "stage"
    (stage / "k9b").mkdir(parents=True, exist_ok=True)
    if k9b is not None:
        (stage / "k9b" / f"{name}.json").write_text(json.dumps({"schema": "x35_k9b_arm_end_v1", "run": name,
                                                                "verdict": k9b}))
    return str(d), str(stage)


def _lt(name: str, run_dir: str, n: int = 10, nodes=(), **kw) -> X.LearnTime:
    col, upd = X.surviving_path([_seg(FORK, n, **kw)], FORK)
    return X.LearnTime(name, FORK, tuple(col), tuple(upd), tuple(nodes), run_dir=run_dir)


def test_am_x47_3_seq_reads_only_finished_k9b_pass_runs(tmp_path):
    d, stage = _finished_dir(tmp_path, "ok", final_step=TARGET)
    assert X.check_finished(_lt("ok", d), stage) == []
    d, stage = _finished_dir(tmp_path, "live", final_step=TARGET, final=False)
    assert any("no final_model.zip" in w for w in X.check_finished(_lt("live", d), stage))
    d, stage = _finished_dir(tmp_path, "stale", final_step=TARGET, nt=FORK)          # metadata not the last row
    assert any("num_timesteps" in w for w in X.check_finished(_lt("stale", d), stage))
    d, stage = _finished_dir(tmp_path, "short", final_step=TARGET, steps=TARGET + IT)
    assert any("< the run's --steps" in w for w in X.check_finished(_lt("short", d), stage))
    d, stage = _finished_dir(tmp_path, "void", final_step=TARGET, k9b="VOID")
    assert any("not PASS" in w for w in X.check_finished(_lt("void", d), stage))
    d, stage = _finished_dir(tmp_path, "unread", final_step=TARGET, k9b=None)
    assert any("K9(b) arm-end record" in w for w in X.check_finished(_lt("unread", d), stage))


def test_am_x47_12_a_thinned_x47_run_is_refused():
    dense = _lt("x47", "", 20, nodes=[FORK + IT * i for i in range(1, 20)])
    assert X.check_nodes_dense(dense, IT) == []
    thinned = _lt("x47", "", 20, nodes=[FORK + 10 * IT])
    assert X.check_nodes_dense(thinned, IT) and "thinned" in X.check_nodes_dense(thinned, IT)[0]
    assert X.check_nodes_dense(dense, None)                                       # unknown grid: refused


def test_am_x47_3_and_8_the_seq_cli_refuses_and_reads(tmp_path, monkeypatch, capsys):
    runs = {}
    for i in range(3):
        for arm, kw in (("c0", {}), ("x47", {"collect_ms": 1300.0, "update_s": 1.3})):
            name = f"{arm}_s{3601 + i}"
            d, stage = _finished_dir(tmp_path, name, final_step=FORK + 20 * IT, steps=FORK + 20 * IT)
            runs[d] = _lt(name, d, 20, nodes=[FORK + IT * k for k in range(1, 20)], **kw)
    monkeypatch.setattr(X, "load_run", lambda d: runs[d])
    c0 = [d for d in runs if "/c0_" in d]
    x47 = [d for d in runs if "/x47_" in d]
    args = ["seq", "--stage", stage] + [a for d in c0 for a in ("--c0-run", d)] + [a for d in x47 for a in ("--x47-run", d)]
    assert X._main(["seq", "--look", "2"] + args[1:]) == 2                       # look 2 needs 5 + 5
    assert "exactly 5 C0 + 5 X47" in capsys.readouterr().err
    assert X._main(["seq", "--look", "1"] + args[1:]) == 0
    out = json.loads(capsys.readouterr().out)
    one = out["s_eq"]["x47_s3601"]
    assert one["file"] == f"checkpoints/checkpoint_{one['s_eq']}_steps.zip" and not one["is_final"]
    assert out["c0_final"]["c0_s3601"]["file"] == "final_model.zip" and out["look"] == 1
    (tmp_path / "stage" / "k9b" / "x47_s3602.json").write_text(json.dumps(
        {"schema": "x35_k9b_arm_end_v1", "run": "x47_s3602", "verdict": "VOID"}))
    assert X._main(["seq", "--look", "1"] + args[1:]) == 2
    assert "x47_s3602: K9(b) arm end reads 'VOID'" in capsys.readouterr().err


def test_am_x47_8_look_2_records_which_nodes_moved():
    c0 = [_run("c0", [_seg(FORK, 20)])]
    x47 = [_run("x47a", [_seg(FORK, 20, collect_ms=1400.0, update_s=1.4)], nodes=[FORK + IT * k for k in range(1, 20)]),
           _run("x47b", [_seg(FORK, 20, collect_ms=1400.0, update_s=1.4)], nodes=[FORK + IT * k for k in range(1, 20)])]
    comp = X.compare(c0, x47)
    node = comp.seq[0].node
    moved = X.moved_nodes({"s_eq": {"x47a": {"s_eq": node}, "x47b": {"s_eq": node - IT}}}, comp)
    assert moved["x47a"] == {"look1_node": node, "look2_node": node, "file": comp.seq[0].file, "reread": False}
    assert moved["x47b"]["reread"]                                                # its node moved: S is re-read
    assert X.moved_nodes({"s_eq": {}}, comp)["x47a"]["reread"]                     # a new seed: read


def test_am_x47_4_the_eval_share_is_reported_on_the_surviving_path():
    p1 = _seg(FORK, 10)
    p2 = _seg(FORK + 6 * IT, 6)
    ev1 = [(FORK + 5 * IT, 2.0), (FORK + 10 * IT, 2.0)]                          # the second is superseded
    ev2 = [(FORK + 10 * IT, 3.0)]
    ev = X.surviving_extra([p1, p2], [ev1, ev2], FORK)
    assert ev == [(FORK + 5 * IT, 2.0), (FORK + 10 * IT, 3.0)]
    col, upd = X.surviving_path([p1, p2], FORK)
    lt = X.LearnTime("c0", FORK, tuple(col), tuple(upd), (), eval_s=5.0)
    assert lt.eval_share() == pytest.approx(5.0 / lt.sigma(), abs=1e-12)
    comp = X.compare([lt], [lt])
    assert comp.as_dict()["reported_only"]["eval_share"]["c0"] == pytest.approx(5.0 / lt.sigma(), abs=1e-12)
    doc = " ".join(X.__doc__.split())
    assert "INCLUDED (AM-X47-4" in doc and "checkpoint writes and in-learn eval" not in doc   # the withdrawn claim


def test_am_x47_5_per_step_only_carries_its_gap_and_label():
    clear = X.read(_doc(s_eq=[50.3, 50.5, 49.5], final=GAIN_ON))          # D = 9.7, 9.9, 10.1
    gap = clear.reported["per_step_gap"]
    assert gap["label"] == XR.THROUGHPUT and gap["ci95_pp"][0] > 0
    assert XR.THROUGHPUT in clear.reasons[0]
    noisy = XR.per_step_gap({"1": 3.0, "2": -1.0, "3": 1.0}, {"1": 0.0, "2": 0.0, "3": 0.0})
    assert noisy["label"] == XR.BOUNDARY and noisy["ci95_pp"][0] < 0
    flat = XR.per_step_gap({"1": 1.0, "2": 1.0}, {"1": 0.0, "2": 0.0})
    assert flat["label"] == XR.BOUNDARY and flat["ci95_pp"] is None              # zero variance: no interval
    v = X.decide(1, clear.primary, clear.secondary, clear.guards, reported={"per_step_gap": noisy})
    assert v.verdict == X.PER_STEP_ONLY and XR.BOUNDARY in v.reasons[0]


def test_am_x47_5_a_lower_bound_at_zero_is_not_above_it(monkeypatch):
    # D = [1, 2, 3]: mean 2, sd 1; a quantile of 2*sqrt(3) puts the lower bound exactly at 0
    monkeypatch.setattr(R, "t_quantile", lambda q, df: 2.0 * 3 ** 0.5)
    g = XR.per_step_gap({"1": 1.0, "2": 2.0, "3": 3.0}, {"1": 0.0, "2": 0.0, "3": 0.0})
    assert abs(g["ci95_pp"][0]) < 1e-9 and g["label"] == XR.BOUNDARY and g["near_boundary"]


def _members(base: float, pstar: float):
    return {"toy_a": base, "toy_b": base, XR.OUT_OF_LINEAGE: pstar}


def test_am_x47_7_per_member_and_the_in_lineage_label():
    on = {s: _members(x + 2.0, x - 4.0) for s, x in _s(GAIN_ON).items()}           # mean = S, P* member level-low
    off = {s: _members(x + 1.5, x - 3.0) for s, x in _s(OFF).items()}
    v = X.read(_doc(per_member={"x47_s_eq": on, "c0_final": off}))
    assert v.verdict == R.ADOPT
    pm = v.reported["per_member"]
    assert set(pm["members"]) == {"toy_a", "toy_b", XR.OUT_OF_LINEAGE} and not pm["mismatch"]
    assert pm["out_of_lineage"][XR.OUT_OF_LINEAGE]["delta_pp"] == pytest.approx(10.0 - 1.0, abs=1e-9)
    assert v.reported.get("lineage_label") is None                               # P* moved too
    on_flat = {s: {"toy_a": x + 6.0, "toy_b": x + 6.0, XR.OUT_OF_LINEAGE: x - 12.0} for s, x in _s(GAIN_ON).items()}
    off_flat = {s: {"toy_a": x - 1.0, "toy_b": x - 1.0, XR.OUT_OF_LINEAGE: x + 2.0} for s, x in _s(OFF).items()}
    v = X.read(_doc(per_member={"x47_s_eq": on_flat, "c0_final": off_flat}))
    assert v.verdict == R.ADOPT and v.reported["lineage_label"] == XR.IN_LINEAGE   # the GAIN is in-lineage only
    bad = {s: _members(x, x) for s, x in _s(GAIN_ON).items()}
    bad["3601"] = _members(0.0, 0.0)
    assert X.read(_doc(per_member={"x47_s_eq": bad, "c0_final": off})).reported["per_member"]["mismatch"]


def test_am_x47_2_every_verdict_says_what_it_does_to_x47b():
    assert X.read(_doc()).as_dict()["x47b"].startswith("X47b RUNS")
    off5 = [50.0, 50.4, 49.6, 50.2, 49.8]
    und = X.read(_doc(look=2, s_eq=[50.0, 56.0, 48.0, 53.0, 52.0], final=[50.0, 56.0, 48.0, 53.0, 52.0], off=off5))
    assert und.verdict == R.UNDECIDED and und.primary.stat.delta > 0
    line = und.as_dict()["x47b"]
    assert "keeps its rank-9 conditional slot" in line and "no effect >= 4.6 pp detected" in line
    neg = X.read(_doc(look=2, s_eq=[50.0, 44.0, 52.0, 47.0, 48.0], final=[50.0, 44.0, 52.0, 47.0, 48.0], off=off5))
    assert neg.verdict == R.UNDECIDED
    neg_line = neg.as_dict()["x47b"]                          # AM-X47b-4: ruled, no longer "NOT RULED"
    assert "NOT RULED" not in neg_line and "DEPRIORITISED" in neg_line
    assert "no effect >= 4.6 pp detected; point estimate <= 0" in neg_line
    assert X.read(_doc(s_eq=LOSS_ON, final=GAIN_ON)).as_dict()["x47b"] == X.X47B_ACTION[R.LOSS]
    assert X.read(_doc(s_eq=NULL_ON, final=NULL_ON)).as_dict()["x47b"].startswith("X47b drops in rank, not out")


def test_am_x47_10_the_descriptor_reads_the_new_rounds():
    import torch

    def state(scale: float):
        sd = {}
        for pre in ("features_extractor.team_transformer.", "pi_features_extractor.team_transformer."):
            for j in range(2):
                sd[f"{pre}transformer_layers.{j}.out_proj.weight"] = torch.full((4, 4), 1.0)        # norm 4
                sd[f"{pre}transformer_layers.{j}.linear2.weight"] = torch.full((4, 4), 2.0)         # norm 8
            for i in range(2):
                sd[f"{pre}extra_rounds.{i}.out_proj.weight"] = torch.full((4, 4), scale)
                sd[f"{pre}extra_rounds.{i}.linear2.weight"] = torch.full((4, 4), 2.0 * scale)
        return sd
    n = XR.round_norms(state(0.5))
    row = n["features_extractor.team_transformer."]["extra_rounds"]["0"]
    assert row["out_proj"] == pytest.approx(2.0) and row["out_proj_ratio"] == pytest.approx(0.5)
    assert XR.reading(n) == {"0": XR.ENGAGED, "1": XR.ENGAGED}
    assert XR.reading(XR.round_norms(state(0.001))) == {"0": XR.NEAR_IDENTITY, "1": XR.NEAR_IDENTITY}
    with pytest.raises(ValueError):
        XR.round_norms({"policy.x": torch.zeros(1)})


def test_am_x47_10_run_norms_reads_every_checkpoint_and_the_final(tmp_path):
    import torch

    run = tmp_path / "rb_x35t_x47d4_s3601"
    (run / "checkpoints").mkdir(parents=True)
    for n in (FORK + IT, FORK + 2 * IT):
        (run / "checkpoints" / f"checkpoint_{n}_steps.zip").write_bytes(b"z")
    (run / "final_model.zip").write_bytes(b"z")
    pre = "features_extractor.team_transformer."
    sd = {f"{pre}transformer_layers.0.out_proj.weight": torch.ones(2, 2), f"{pre}transformer_layers.0.linear2.weight":
          torch.ones(2, 2), f"{pre}extra_rounds.0.out_proj.weight": torch.zeros(2, 2),
          f"{pre}extra_rounds.0.linear2.weight": torch.zeros(2, 2)}
    out = XR.run_norms(str(run), loader=lambda f: sd)
    assert [s["step"] for s in out["series"]] == [FORK + IT, FORK + 2 * IT, "final"]
    assert out["last_node"] == "final" and out["reading"] == {"0": XR.NEAR_IDENTITY}


# ------------------------------------------------------------------ AM-X47b-3: the policy-difference descriptor
def test_am_x47b_3_zeroing_the_new_rounds_makes_them_the_identity():
    import torch

    from agents.model.trunk_depth import build_extra_rounds

    class Trunk(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.extra_rounds = build_extra_rounds(4)

    t = Trunk()
    with torch.no_grad():                                     # a "trained" round: its outputs are no longer zero
        for p in t.parameters():
            p.add_(0.05 * torch.randn_like(p))
    x = torch.randn(2, 5, t.extra_rounds[0].norm1.normalized_shape[0])
    assert not torch.equal(t.extra_rounds[1](t.extra_rounds[0](x)), x)
    assert XR.zero_extra_rounds(t) == 2
    assert torch.equal(t.extra_rounds[1](t.extra_rounds[0](x)), x)
    assert XR.zero_extra_rounds(torch.nn.Linear(2, 2)) == 0


def test_am_x47b_3_policy_diff_is_the_mean_kl_by_hand():
    import math

    p = [[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]]
    q = [[0.25, 0.75, 0.0], [1.0, 0.0, 0.0]]
    kl0 = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
    out = XR.policy_diff(p, q, free=[True, False])
    assert out["kl_mean"] == pytest.approx(kl0 / 2, abs=1e-15) and out["kl_max"] == pytest.approx(kl0, abs=1e-15)
    assert out["kl_mean_free"] == pytest.approx(kl0, abs=1e-15)
    assert out["argmax_moved_share"] == 0.5                   # row 0: argmax 0 -> 1
    assert XR.policy_diff(p, p)["kl_mean"] == 0.0
    with pytest.raises(ValueError, match="infinite"):
        XR.policy_diff([[0.5, 0.5]], [[1.0, 0.0]])
