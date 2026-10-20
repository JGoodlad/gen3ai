"""D1's readout (``main.x35.d1``) on CONSTRUCTED inputs: the D1 self-check suite passes check by check, the CLI
writes the verdict, and the edges the self-check does not name. Deterministic (the only draws are the self-check's
private seeded generator)."""
from __future__ import annotations

import json

import pytest

from main.x35 import d1 as D1
from main.x35 import readout as R
from main.x35 import selfcheck as SC


@pytest.mark.parametrize("check", SC.D1_CHECKS, ids=lambda c: c.__name__)
def test_every_d1_self_check_passes(check):
    name, ok, detail = check()
    assert ok, f"{name}: {detail}"


def test_the_self_check_runs_both_suites_and_each_alone(capsys):
    assert R._main(["self-check", "--suite", "d1"]) == 0
    out = capsys.readouterr().out
    assert f"self-check d1: {len(SC.D1_CHECKS)} / {len(SC.D1_CHECKS)} PASS" in out and "[x35]" not in out
    assert R._main(["self-check"]) == 0
    n = len(SC.CHECKS) + len(SC.D1_CHECKS)
    assert f"self-check: {n} / {n} PASS" in capsys.readouterr().out


def test_the_pooling_is_the_equal_weight_panel_mean_not_the_pooled_proportion():
    # p72 counted 400 games (one unit short), the others 420: the pooled proportion would weight p72 less
    s = D1.strat_score({"heldout72_hpfix": (210, 420), "forum72": (210, 420), "p72": (100, 400)})
    assert s.mean == pytest.approx((0.5 + 0.5 + 0.25) / 3, abs=1e-15)
    assert s.mean != pytest.approx(520 / 1240, abs=1e-6)


def test_the_unfamiliar_panels_are_the_registered_held_out_ones_and_the_kakuna_runners():
    from main import heldout_panels as HP
    from main.x35 import kakuna as KA

    assert D1.PANELS == KA.UNFAMILIAR
    for k in D1.PANELS:
        assert "training" in HP.panel_entry(k)["held_out_from"]


def test_a_look_mismatch_between_h_and_p_is_refused():
    h = D1.d1_primary(1, SC._flat(70), SC._flat(56))
    p = D1.d1_coprimary(2, SC._seeds([50.0] * 5), SC._seeds([50.0] * 5))
    with pytest.raises(ValueError, match="look"):
        D1.d1_verdict("X38", h, p, [])


def test_the_decide_d1_cli_writes_the_verdict_json(tmp_path, capsys):
    doc = {"arm": "X38", "look": 1, "h": {"on": SC._flat(56), "off": SC._flat(70)},
           "p": {"on": SC._seeds([50.1, 50.2, 50.0]), "off": SC._seeds([50.0, 50.1, 49.9])}}
    p = tmp_path / "read.json"
    p.write_text(json.dumps(doc))
    assert R._main(["decide-d1", str(p)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"] == D1.LOSS and out["h"]["class"] == D1.LOSS
    assert set(out["h"]["per_panel_reported"]) == set(D1.PANELS) and out["p"]["class"] == D1.NON_INFERIOR
    assert "own population" in out["h"]["per_panel_reported"]["forum72"]["role"]


def test_the_kakuna_d1h_output_is_decide_d1s_h_input(tmp_path):
    """The runner's ``d1h`` JSON drops straight into ``read_d1``'s ``h`` (the two halves agree on the shape)."""
    from main.x35 import kakuna as KA

    finals = [{"label": f"{arm.lower()}_s{s}", "arm": arm, "seed": s, "path": "/x"}
              for arm in ("X38", "C0") for s in (3501, 3502, 3503)]
    units = KA.plan_units(finals, hash_models=False, teamsets=KA.UNFAMILIAR)
    for u in units:
        d = tmp_path / "cells" / u.name
        d.mkdir(parents=True)
        d.joinpath("summary.json").write_text(json.dumps({
            "status": "OK", "n": 20, "regime_verified_decisions": True, "team_source_asymmetry": False,
            "peer_clean": True, "wins": 10 if u.arm == "X38" else 8, "ties": 0}))
    h = KA.d1h_input(KA.pool(tmp_path, units), "X38", "C0")
    v = D1.read_d1({"arm": "X38", "look": 1, "h": {"on": h["on"], "off": h["off"]}})
    assert v.h.cls == D1.GAIN and v.h.delta_pp == pytest.approx(10.0, abs=1e-9)
    assert v.verdict == D1.INCONCLUSIVE and "P is INCONCLUSIVE" in v.reasons[0]


# ------------------------------------------------------------------------------------- P's input from a ledger
class _Ledger:
    """A ledger root with one registered A/B family and a hand-moved clock (the ledger reader tests' pattern)."""

    def __init__(self, root):
        from agents.training import eval_ledger as L
        from agents.training.eval_ledger import testkit as K

        self.L, self.K = L, K
        self.root, self.clock = root, K.Clock()
        self.w = L.LedgerWriter(root, "h2h", clock=self.clock, host="testhost", pid=11, alive=lambda _p: True)
        self.w.register_family("x38_p", decision_kind="ab_verdict", rule="D1 §4.2 P", protocol=K.PROTO)
        self.req = self.w.open_request("look1", kind="ab_cell", purpose="ab", family="x38_p", spec={})

    def put(self, p, o, k_up, reg, batch=0):
        self.clock.advance(60)
        c = self.w.claim("look1", batch=batch, player=p, opponent=o, regime_id=reg["regime_id"], expected_wall_s=60)
        row = self.K.make_row(self.w.next_row_id(), request=self.req, batch=batch, p=p, o=o, reg=reg,
                              ts=self.clock.iso(), purpose="ab", pc=(0, 0, 500 - k_up, k_up, 0),
                              cycle_seed=hash((p, o, batch)) % 10 ** 9)
        self.w.append_row(row, c)


def _panel_reg(panel="home72"):
    from agents.training.eval_ledger import testkit as K
    from main.h2h import panel as PN

    return K.regime(team_set=PN.panel_team_set_id(panel), team_source=PN.panel_team_source(panel))


def test_p_input_scores_each_final_against_r_on_the_registered_panel(tmp_path):
    on, off, m1, m2 = "e" * 64, "f" * 64, "1" * 64, "2" * 64
    led = _Ledger(tmp_path / "ledger")
    reg = _panel_reg()
    for p, ks in ((on, (100, 200)), (off, (0, 100))):
        for o, k in zip((m1, m2), ks):
            led.put(p, o, k, reg)
    finals = [{"label": "x38_s3501", "arm": "X38", "seed": 3501, "sha256": on},
              {"label": "c0_s3501", "arm": "C0", "seed": 3501, "sha256": off}]
    out = D1.p_input(led.root, "x38_p", "look1", finals, "X38", "C0", {"m1": m1, "m2": m2})
    assert out["inconclusive"] == []
    assert out["on"]["3501"] == pytest.approx((55.0 + 60.0) / 2) and out["off"]["3501"] == pytest.approx(52.5)
    missing = D1.p_input(led.root, "x38_p", "look1", finals, "X38", "C0", {"m1": m1, "m3": "3" * 64})
    assert missing["on"]["3501"] is None and "missing cell" in missing["per_final"]["x38_s3501"]["inconclusive"][0]


def test_p_input_is_inconclusive_when_the_look_was_not_played_on_the_panel(tmp_path):
    """The ledger holds ONE regime per request, so the failure is a whole look played without ``--panel`` (the pool's
    team set) or on another panel; its population scores are computed but P does not decide."""
    from agents.training.eval_ledger import testkit as K

    on, off, m1 = "e" * 64, "f" * 64, "1" * 64
    led = _Ledger(tmp_path / "ledger")
    led.put(on, m1, 100, K.regime())                        # the POOL's team set
    led.put(off, m1, 100, K.regime())
    finals = [{"label": "a", "arm": "X38", "seed": 1, "sha256": on}, {"label": "b", "arm": "C0", "seed": 1,
                                                                     "sha256": off}]
    out = D1.p_input(led.root, "x38_p", "look1", finals, "X38", "C0", {"m1": m1})
    assert out["inconclusive"] and "not panel home72's" in out["inconclusive"][0]
    assert D1.p_input(led.root, "x38_p", "look1", finals, "X38", "C0", {"m1": m1}, panel="away")["inconclusive"]
    p = D1.d1_coprimary(1, out["on"], out["off"], inconclusive=out["inconclusive"])
    assert p.cls is None


def test_am5_widening_turns_a_seed_disagreement_gain_into_continue():
    """AM-5 (2026-10-19): the pooled-binomial interval reads GAIN on seeds at +10 / 0 / +20 pp; the overdispersion
    (φ̂ = 6.3 / .73 ≈ 8.63) widens it past 0. Fails on revert of the widening."""
    h = D1.d1_primary(1, SC._hc([(70, 70, 70), (56, 56, 56), (84, 84, 84)]), SC._flat(56))
    assert h.ci_unwidened_pp is not None and h.ci_unwidened_pp[0] > 2.0          # unwidened: lower clearly > 0
    assert h.ci_pp is not None and h.ci_pp[0] < -1.0                             # widened: clearly below 0
    assert (h.cls, h.outcome) == (D1.UNDECIDED, D1.CONTINUE)
    w = h.as_dict()
    assert w["overdispersion"]["df"] == 4 and w["overdispersion"]["half_width_factor"] == pytest.approx(
        (6.3 / 0.73) ** 0.5, rel=1e-12)


def test_am5_never_narrows_and_an_undefined_phi_is_inconclusive():
    h = D1.d1_primary(1, SC._flat(70), SC._flat(56))
    assert h.od is not None and h.od.factor == 1.0 and h.ci_pp == h.ci_unwidened_pp and h.cls == D1.GAIN
    with pytest.raises(R.ReadoutInputError, match="binomial expectation is 0"):
        D1.overdispersion(SC._hc([(0, 140, 0)] * 3), SC._flat(56))
