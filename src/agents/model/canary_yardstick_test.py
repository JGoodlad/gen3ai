"""THE YARDSTICK IS NOT THE COMPILED ARM (`gen3_r1_yardstick_margin_v1`, 2026-10-19; the X34 u1700 UNCONFIRMED,
`designs/research_state/measurements/x34_canary_u1700_2026-10-19/`).

At X34 update 1700 the canary logged "compiled DISAGREES with eager" and counted an UNCONFIRMED toward its FATAL
ladder, but the arm that broke the clip margin (M 0.72) was EAGER bf16 at 0.856 — the canary's own yardstick; compiled
read 0.52 with 0 clip switches. Here, on CPU, with the rung's arms planted (no learner, no compile):

* the u1700 shape (eager 0.856, compiled 0.52, M 0.72) reads PASS with a recorded yardstick event, the rung re-run
  once at M' = 1.25 x 0.856 = 1.07 and judged there (the first read's verdict discarded);
* a COMPILED arm out of margin is refused as before, and through the canary it confirms on both halves and is FATAL;
* the compiled arm's own shift never sets M' (a miscompile cannot widen its own bar);
* every record names each arm's worst row (batch and fixture row) and the same-precision compiled-vs-eager gap; an
  unconfirmed / FATAL verdict dumps the policy weights, at most `DUMP_WEIGHTS_MAX` per process;
* the log says "compiled DISAGREES with eager" only when the COMPILED arm failed;
* a record with no yardstick event carries no yardstick key (the existing records are unchanged).

The real learner's end-to-end read is `compile_canary_test.test_a_u1700_shaped_EAGER_tail_on_the_real_learner_*`."""
from __future__ import annotations

import json
import math
import os
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest
import torch

from agents.model import compile_canary as CC
from agents.model import compile_regions as cr
from agents.model import compile_trainer as ct
from agents.model import trunk_precision as TP

CLIP = 0.15
M = cr.R1_CLIP_MARGIN
ROWS = 64
U1700 = {"eager": 0.856, "compiled": 0.52}            # the u1700 scheduled read's max |dlogp| per arm


def _fixture() -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(1700)
    ref = -torch.rand(ROWS, generator=g, dtype=torch.float64) * 3 - 0.5
    log_ratio = (torch.rand(ROWS, generator=g, dtype=torch.float64) - 0.5) * 4          # rho 0.14 .. 7.4
    adv = torch.randn(ROWS, generator=g, dtype=torch.float64)
    return {"ref": ref, "old": (ref - log_ratio).float(), "adv": adv}


def _arm(ref: torch.Tensor, shift: float, row: int) -> Dict[str, torch.Tensor]:
    lp = ref.clone()
    lp[row] += shift
    return {"log_prob": lp.float(), "loss": torch.zeros(1)}


class _Rung:
    """`_r1_judge`'s collaborators, planted: the relocation (the real `clip_relocation` on the fixture, at the margin it
    is asked for), the arms (their shifts per call), the fp32 reference, and the bf16 verdict (recorded; optionally
    refusing on the FIRST read, as a 0/1 clip-switch lottery would)."""

    def __init__(self, mp: pytest.MonkeyPatch, shifts: List[Dict[str, float]], rows: Dict[str, int],
                 first_verdict_refuses: bool = False) -> None:
        self.fx = _fixture()
        self.shifts, self.rows = shifts, rows
        self.margins: List[Optional[float]] = []
        self.verdicts = 0
        self.first_refuses = first_verdict_refuses
        mp.setattr(TP, "trunk_precision_of", lambda _p: "bf16")
        mp.setattr(cr, "relocate_clip_boundary", self.relocate)
        mp.setattr(cr, "_r1_pair", self.pair)
        mp.setattr(cr, "_r1_reference", lambda _m, _a: {"log_prob": self.fx["ref"].float(), "loss": torch.zeros(1)})
        mp.setattr(cr, "golden_row_losses", lambda *_a, **_k: {"ref": 0.0, "eager": 0.0, "comp": 0.0})
        mp.setattr(cr, "_r1_verdict_bf16", self.verdict)
        mp.setattr(cr, "release_gate_memory", lambda _m: 0)

    def relocate(self, _model: Any, args: Any, margin: Optional[float] = None) -> Any:
        self.margins.append(margin)
        return args, cr.clip_relocation(self.fx["ref"], self.fx["old"], self.fx["adv"], CLIP,
                                        M if margin is None else margin)

    def pair(self, _model: Any, _args: Any, _fn: Any = None) -> Any:
        s = self.shifts[min(len(self.margins) - 1, len(self.shifts) - 1)]
        ref = self.fx["ref"]
        return _arm(ref, s["compiled"], self.rows["compiled"]), _arm(ref, s["eager"], self.rows["eager"])

    def verdict(self, *_a: Any, **_k: Any) -> str:
        self.verdicts += 1
        if self.first_refuses and self.verdicts == 1:
            raise ct.CompileTrainerError("planted: a 0/1 clip-switch lottery on the first read")
        return "bf16 rule: planted pass"


def _judge(watch: Optional[cr.GradWatch] = None, row_bars: Any = None) -> str:
    model = SimpleNamespace(policy=None)                           # every collaborator that reads it is planted
    return cr._r1_judge(model, ("args",), ["pointer_head.w"], "trained", watch=watch, row_bars=row_bars)[2]


# ------------------------------------------------------------------------------------- the rung
def test_the_u1700_record_reads_PASS_with_a_YARDSTICK_event_rerun_once_at_M_prime(monkeypatch):
    """Fails on revert: the old rule refused eager's 0.856 > 0.72 as a disagreement."""
    rung = _Rung(monkeypatch, [U1700], {"eager": 4, "compiled": 20}, first_verdict_refuses=True)
    w = cr.GradWatch(("pointer_head.",))
    line = _judge(w)
    m2 = cr.YARDSTICK_HEADROOM * U1700["eager"]
    assert m2 == pytest.approx(1.07)
    assert rung.margins == [None, pytest.approx(m2)]                # ONE re-run, at M' from the YARDSTICK's shift
    assert rung.verdicts == 2                                       # the first read's verdict was discarded
    assert "YARDSTICK out of margin, NOT counted" in line and "eager 0.856 (row 4)" in line, line
    assert "0 clip-branch switches" in line and "M 1.070" in line, line
    rec = w.record()
    (ev,) = rec["yardstick_out_of_margin"]
    assert ev["margin"] == pytest.approx(M) and ev["rerun_margin"] == pytest.approx(m2, rel=1e-3)
    assert set(ev["out_of_margin"]) == {"eager"} and ev["out_of_margin"]["eager"]["row"] == 4
    assert ev["compiled_shift"] == pytest.approx(0.52, rel=1e-3)
    cm = rec["clip_margin"]                                         # the JUDGED read is the re-run's
    assert cm["margin"] == pytest.approx(m2, rel=1e-3) and cm["shift_max_source"].startswith("yardstick re-run")
    assert cm["worst_row"] == {"compiled": 20, "eager": 4, "fp32 reference": cm["worst_row"]["fp32 reference"]}
    sp = cm["same_precision"]
    assert sp["arms"] == "compiled - eager" and sp["bar"] is None and sp["row"] in (4, 20)
    assert sp["max_abs_dlogp"] == pytest.approx(0.856, rel=1e-3)


def test_a_COMPILED_only_breach_still_REFUSES_as_the_compiled_arm_with_no_rerun(monkeypatch):
    rung = _Rung(monkeypatch, [{"eager": 0.10, "compiled": 0.90}], {"eager": 4, "compiled": 20})
    with pytest.raises(ct.CompileTrainerError, match=r"the compiled arm moved a fixture row's log-prob by up to "
                                                     r"0\.9 \(row 20\)") as e:
        _judge()
    assert not isinstance(e.value, cr.YardstickRefusal)
    assert rung.margins == [None]


def test_the_COMPILED_shift_never_widens_the_margin(monkeypatch):
    """Compiled 1.2 beside an eager tail of 0.8: M' = max(M, 1.25 x 0.8) = 1.0, set by the YARDSTICK alone — and the
    compiled arm is refused at it. With compiled at 5.0, M' is still 1.0, never 1.25 x 5."""
    rung = _Rung(monkeypatch, [{"eager": 0.80, "compiled": 1.20}], {"eager": 4, "compiled": 20})
    with pytest.raises(ct.CompileTrainerError, match=r"the compiled arm .* margin is 1 = ") as e:
        _judge()
    assert not isinstance(e.value, cr.YardstickRefusal)
    assert rung.margins == [None, pytest.approx(1.0)]
    fx = _fixture()
    fix = cr.clip_relocation(fx["ref"], fx["old"], fx["adv"], CLIP)
    arms = {"compiled": _arm(fx["ref"], 5.0, 1), "eager": _arm(fx["ref"], 0.8, 2)}
    with pytest.raises(cr._YardstickOutOfMargin) as y:
        cr._clip_margin_rule(fix, arms, rerun=True)
    assert y.value.event["rerun_margin"] == pytest.approx(1.0) and y.value.event["compiled_shift"] == pytest.approx(5.0)


def test_a_yardstick_STILL_out_at_M_prime_is_a_YARDSTICK_refusal_not_a_compiled_one(monkeypatch):
    """The re-run reads the same forwards; a yardstick that moves past its own M' did not reproduce itself: refused,
    typed as the yardstick (the canary words it as itself)."""
    _Rung(monkeypatch, [U1700, {"eager": 1.5, "compiled": 0.52}], {"eager": 4, "compiled": 20})
    with pytest.raises(cr.YardstickRefusal, match=r"YARDSTICK eager arm — NOT the compiled one.*did not reproduce"):
        _judge()


def test_with_NO_yardstick_event_the_record_carries_no_yardstick_key(monkeypatch):
    rung = _Rung(monkeypatch, [{"eager": 0.30, "compiled": 0.25}], {"eager": 4, "compiled": 20})
    w = cr.GradWatch(("pointer_head.",))
    line = _judge(w)
    assert rung.margins == [None] and "YARDSTICK" not in line
    rec = w.record()
    assert "yardstick_out_of_margin" not in rec
    assert rec["clip_margin"]["margin"] == pytest.approx(M) and rec["clip_margin"]["shift_max_source"] == "default"


def test_the_bf16_DRIFT_refusal_is_typed_as_the_yardstick():
    names = ["pointer_head.w", "trunk.w"]
    sizes = torch.tensor([2, 2])
    ref = {"loss": torch.tensor([1.0]), "grad": torch.tensor([1.0, 1.0, 1.0, 1.0]), "grad_sizes": sizes}
    eager = dict(ref, loss=torch.tensor([1.5]))
    with pytest.raises(cr.YardstickRefusal, match="DRIFTED"):
        cr._r1_verdict_bf16(eager, dict(ref), ref, names, "trained")


# ------------------------------------------------------------------------------------ the canary
@pytest.fixture
def canary_world(monkeypatch, tmp_path):
    """A `CompileCanary` whose check runs the REAL `_r1_judge` and margin rule on planted arms (per check: the shifts
    the test lists), recording into the canary's own watch with the fixture rows a B = 2 x ROWS batch tiles."""
    monkeypatch.setattr(CC, "_WEIGHT_DUMPS", 0)
    policy = torch.nn.Linear(3, 2)
    model = type("M", (), {})()
    model.policy, model.num_timesteps, model.behaviour_dump_dir = policy, 267_205_580, str(tmp_path)
    plan: Dict[str, Any] = {"checks": [], "rows": {"eager": 4, "compiled": 20}}
    said: List[str] = []

    def check(self, slice_, out, watch=None, row_bars=None):
        shifts = plan["checks"].pop(0)
        # no learner, so the canary's null cannot bind: the per-row bars are the FALLBACK, every row at the cap = M
        assert row_bars is None or not row_bars.live
        bb = None if row_bars is None else row_bars.for_batch(range(ROWS))      # the planted fixture's rows
        if watch is not None:
            watch.fixture_rows = [int(x) for x in ct.fixture_index(2 * ROWS, ROWS, slice_)]
        with monkeypatch.context() as mp:
            _Rung(mp, shifts, plan["rows"])
            return [_judge(watch, bb)]
    monkeypatch.setattr(CC.CompileCanary, "_check", check)
    can = CC.CompileCanary(model, batch_size=2 * ROWS, every=1, saver=lambda _p: None, emit=said.append)
    return can, plan, tmp_path, said


def _records(tmp) -> List[Dict[str, Any]]:
    return [json.loads(x) for x in (tmp / CC.VERDICTS_FILE).read_text().splitlines()]


def test_the_canary_reads_the_u1700_shape_as_PASS_with_the_event_recorded_and_NOTHING_counted(canary_world):
    """Fails on revert: X34 logged "compiled DISAGREES with eager" here and counted an UNCONFIRMED."""
    can, plan, tmp, said = canary_world
    plan["checks"] = [[U1700]]
    out = can.after_update()
    assert out["compile/canary_ok"] == 1.0 and out["compile/canary_unconfirmed_disagreements"] == 0.0
    assert out["compile/canary_yardstick_events"] == 1.0 and not can.prev_disagreed
    (rec,) = _records(tmp)
    assert rec["verdict"] == "pass" and "policy_pt" not in rec
    w = rec["watch"]["scheduled"]
    (ev,) = w["yardstick_out_of_margin"]
    assert ev["out_of_margin"]["eager"]["row"] == 4 and ev["out_of_margin"]["eager"]["fixture_row"] == 4
    assert w["clip_margin"]["worst_fixture_row"]["compiled"] == 20
    assert not any("DISAGREES" in s for s in said) and any("YARDSTICK out of margin, NOT counted" in s for s in said)


def test_a_COMPILED_breach_still_CONFIRMS_and_is_FATAL_with_the_rows_and_weights_recorded(canary_world):
    can, plan, tmp, said = canary_world
    breach = {"eager": 0.10, "compiled": 0.90}
    plan["checks"] = [[breach]] * 3                              # the scheduled check and both disjoint halves
    with pytest.raises(CC.CompileCanaryError, match="CONFIRMED"):
        can.after_update()
    assert any("compiled DISAGREES with eager" in s for s in said), said
    (rec,) = _records(tmp)
    assert rec["verdict"] == "fatal" and rec["policy_pt"] == "canary_u1_s267205580_fatal_policy.pt"
    sd = torch.load(tmp / rec["policy_pt"])
    assert set(sd) == {"weight", "bias"}
    for name in ("scheduled", "confirm_slice2", "confirm_slice1"):
        cm = rec["watch"][name]["clip_margin"]
        assert cm["worst_row"]["compiled"] == 20 and cm["max_abs_dlogp"]["compiled"] == pytest.approx(0.9, rel=1e-3)
        assert "worst_fixture_row" in cm and "same_precision" in cm
    assert rec["watch"]["confirm_slice1"]["clip_margin"]["worst_fixture_row"]["compiled"] == ROWS // 2 + 20


def test_an_UNCONFIRMED_dumps_the_weights_and_a_YARDSTICK_refusal_is_never_worded_as_compiled(canary_world):
    can, plan, tmp, said = canary_world
    unreproducible = [U1700, {"eager": 1.5, "compiled": 0.52}]   # the yardstick moves past its own M'
    plan["checks"] = [unreproducible, [{"eager": 0.1, "compiled": 0.1}], unreproducible]
    out = can.after_update()
    assert out["compile/canary_ok"] == 0.0 and out["compile/canary_unconfirmed_disagreements"] == 1.0
    assert any("the canary's YARDSTICK refused" in s for s in said), said
    assert not any("DISAGREES" in s for s in said), said
    (rec,) = _records(tmp)
    assert rec["verdict"] == "unconfirmed" and rec["policy_pt"] == "canary_u1_s267205580_unconfirmed_policy.pt"
    assert (tmp / rec["policy_pt"]).exists()


def test_the_weight_dump_is_CAPPED_per_process(canary_world, tmp_path):
    can, *_ = canary_world
    got = [CC._dump_weights(can.model, str(tmp_path), u, "unconfirmed", 5) for u in range(CC.DUMP_WEIGHTS_MAX + 1)]
    assert got[:-1] == [f"canary_u{u}_s5_unconfirmed_policy.pt" for u in range(CC.DUMP_WEIGHTS_MAX)]
    assert got[-1] == "cap" and CC.DUMP_WEIGHTS_MAX == 3
    assert CC._dump_weights(can.model, None, 9, "fatal") is None


def test_a_post_restart_dump_never_overwrites_an_earlier_one(canary_world, tmp_path, monkeypatch):
    """0dcf0ea4 FINDING 3 (`gen3_canary_dump_no_clobber_v1`). Fails on revert: the dump was named by the update number
    alone, which restarts at 0 per process, so the second process's u1 dump overwrote the first's."""
    can, *_ = canary_world
    w = can.model.policy.weight
    with torch.no_grad():
        w.fill_(1.0)
    first = CC._dump_weights(can.model, str(tmp_path), 1, "unconfirmed", 100)
    monkeypatch.setattr(CC, "_WEIGHT_DUMPS", 0)                   # a new process: the per-process cap starts again
    with torch.no_grad():
        w.fill_(2.0)
    later = CC._dump_weights(can.model, str(tmp_path), 1, "unconfirmed", 200)       # a restart, a later step
    with torch.no_grad():
        w.fill_(3.0)
    again = CC._dump_weights(can.model, str(tmp_path), 1, "unconfirmed", 200)       # a resume from the SAME step
    assert (first, later, again) == ("canary_u1_s100_unconfirmed_policy.pt", "canary_u1_s200_unconfirmed_policy.pt",
                                     "canary_u1_s200_r2_unconfirmed_policy.pt")
    assert [float(torch.load(tmp_path / n)["weight"][0, 0]) for n in (first, later, again)] == [1.0, 2.0, 3.0]
    near = [CC._dump_near_margin(can.model, str(tmp_path), 1, 200) for _ in range(2)]
    assert near == ["canary_u1_s200_near_policy.pt", "canary_u1_s200_r2_near_policy.pt"]
    assert not [f for f in os.listdir(tmp_path) if f.endswith(".tmp")]


def test_the_headroom_is_its_declaration():
    assert cr.YARDSTICK_ARMS == ("eager", "fp32 reference") and cr.YARDSTICK_HEADROOM == 1.25
    assert math.isclose(cr.YARDSTICK_HEADROOM * 0.856, 1.07)
