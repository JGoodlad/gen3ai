"""Gate ④'s at-scale readout: per-category dominated mass and starvation, paired steps, verdicts."""

from __future__ import annotations

import numpy as np
import pytest

from main.policy_spectrum import bank as B
from main.policy_spectrum import truth_report as R
from main.policy_spectrum.bank_test import BANK_V1


@pytest.fixture(scope="module")
def v1():
    return B.load_bank(BANK_V1)


def _setup(v1, n=40):
    """n free turns with a legal switch and a legal attack: switch near-best, attack dominated."""
    rows, ids, seen = [], [], set()
    for d in v1.decisions:
        if d["kind"] != "free" or d["battle"] in seen:      # one turn per battle: clusters
            continue
        sw = [int(a) for a, c in d["cats"].items() if c == "switch"]
        at = [int(a) for a, c in d["cats"].items() if c == "attack"]
        if not sw or not at:
            continue
        out = {int(a): [-1.0] * 4 for a in d["tokens"]}
        out[sw[0]] = [1.0] * 4
        rows.append({"id": d["id"], "ok": True, "seeds": ["s"] * 4,
                     "outcomes": {str(a): v for a, v in out.items()}})
        ids.append((d["id"], sw[0], at[0]))
        seen.add(d["battle"])
        if len(rows) == n:
            break
    return rows, ids


def test_sharpening_onto_the_best_is_healthy_and_starving_it_is_starvation(v1):
    rows, ids = _setup(v1)
    pos = {d["id"]: i for i, d in enumerate(v1.decisions)}
    a = np.zeros((len(v1.decisions), 11))
    good = a.copy()
    bad = a.copy()
    for did, sw, at in ids:
        i = pos[did]
        a[i, sw], a[i, at] = 0.5, 0.5          # half on the dominated attack
        good[i, sw], good[i, at] = 0.99, 0.01  # mass moved OFF the dominated action: healthy
        bad[i, sw], bad[i, at] = 0.005, 0.995  # the near-best switch starved
    table = R.turn_table(v1, rows)
    assert all(T["decisive"] for T in table)
    h = R.step(table, a, good, "attack")["dominated_mass"]
    assert h["delta"] == pytest.approx(-0.49) and R.verdict(h) == "−"
    s = R.step(table, a, bad, "switch")["starved"]
    assert s["delta"] == pytest.approx(1.0) and R.verdict(s) == "+"
    assert R.step(table, a, good, "switch")["starved"]["delta"] == 0.0
    lv = R.level(table, bad, "setup")
    assert lv["starved"]["n"] == 0 and lv["starved"]["mean"] is None


def test_analyse_and_render_agreement(v1):
    rows, ids = _setup(v1, 20)
    rows = [dict(r, seeds=["s"] * 4) for r in rows]
    pos = {d["id"]: i for i, d in enumerate(v1.decisions)}
    p = {}
    for lab, w in (("x", 0.5), ("y", 0.99)):
        q = np.zeros((len(v1.decisions), 11))
        for did, sw, at in ids:
            q[pos[did], sw], q[pos[did], at] = w, 1 - w
        p[lab] = q
    why = {r["id"]: "random" for r in rows}
    res = R.analyse(v1, {"c1": rows, "c2": rows}, p, ["x", "y"], why, nested=(2,))
    assert set(res["continuations"]["c1"]) == {"S4", "S2(nested)"}
    ag = res["verdict_agreement"]["x → y | attack | dominated_mass"]
    assert ag == {"c1": "−", "c2": "−"}
    md = R.render(res)
    assert "x → y" in md and "(−)" in md
