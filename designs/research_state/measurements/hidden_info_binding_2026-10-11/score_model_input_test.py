"""SYNTHETIC tests of the blinded scorer and the unblinder (no real output is read; the answers are known by
construction). Run with ``python -I -m pytest -q score_model_input_test.py`` from this folder (or as a script).

The fake dataset: 12 states in 6 battles (2 each), three legal actions with truth T = (1, 0, 1/2) (all '+' / all '-' /
all '0' over the 24 truth seeds), BASE root logits (2, 1, 0) (top1 = action 0), one-ply V = (0.9, 0.1, 0.5) with action
0's seed 7 TERMINAL (a win), so BASE's Qhat = (0.9125, 0.1, 0.5). Codes:
- A01 reverses everything: root (0, 2, 1) (top1 action 1), V (0.1, 0.9, 0.5) -> IMP_POLICY = IMP_CRITIC = -1, FLIP 1,
  PAIRWISE 0;
- A02 is BASE's bytes -> every metric equals BASE's;
- A03 picks action 2 on WIN states only (IMP_POLICY -1/2 there, 0 on losses): raw -0.25, re-weighted -1/3 (capture rate
  win 0.5 -> weight 2, loss 1.0 -> weight 1).
"""
import importlib
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
SM = importlib.import_module("score_model_input")
UB = importlib.import_module("unblind_model_input")

N_STATES, PER_BATTLE = 12, 2
TRUTH = ("+", "-", "0")
BASE_LG = [2.0, 1.0, 0.0] + [-9.0] * 8
V_BASE = (0.9, 0.1, 0.5)


def is_win(i):
    return (i // PER_BATTLE) % 2 == 0


def succ(v):
    """Branch order: action-major, seed-minor; action 0 seed 7 is terminal (no successor value)."""
    out = []
    for a in range(3):
        for s in range(8):
            if not (a == 0 and s == 7):
                out.append(float(v[a]))
    return out


def build(root: Path, cp="syn"):
    states, raw, p1 = {}, [], []
    for i in range(N_STATES):
        did = f"b{i // PER_BATTLE}#p1#{i}"
        bat = f"b{i // PER_BATTLE}"
        win = is_win(i)
        states[did] = {"battle": bat, "cycle_opp": "c/o", "logits": BASE_LG, "top": [0, 1, 2], "turn": 5,
                       "unrevealed": 0 if i // PER_BATTLE < 3 else 4, "outcome": "win" if win else "loss",
                       "quintile": 1, "n_legal": 3, "gap": 1.0, "v_root": 0.5}
        out = {str(a): TRUTH[a] * 40 for a in range(3)}
        trace = {str(a): [{"t": [], "v": [], "s": ""} for _ in range(40)] for a in range(3)}
        crit = {str(a): {"q": [0.0] * 8, "src": "V" * 8} for a in range(3)}
        p1.append({"id": did, "ok": True, "turn": 5, "out": out, "trace": trace, "crit": crit,
                   "cost": {"play": {"cpu_s": 1.0, "rows": 10, "playouts": 120}, "crit": {"cpu_s": 0.1}}})
        branches = [[a, s, "+", 1.0] if (a == 0 and s == 7) else [a, s, "V", None] for a in range(3) for s in range(8)]
        a3_root = [0.0, 0.5, 2.0] + [-9.0] * 8 if win else BASE_LG
        arms = {
            "A01": {"root": [0.0, 2.0, 1.0] + [-9.0] * 8 + [0.5], "succ": succ((0.1, 0.9, 0.5))},
            "A02": {"root": BASE_LG + [0.5], "succ": succ(V_BASE)},
            "A03": {"root": a3_root + [0.5], "succ": succ(V_BASE)},
            "A04": {"root": BASE_LG + [0.5], "succ": succ(V_BASE)},
        }
        for c in arms:
            arms[c].update({"root_stats": {"slots": 2, "moves": 5}, "succ_slots": 10, "succ_moves": 25})
        raw.append({"id": did, "cp": cp, "ok": True, "mask": [1, 1, 1] + [0] * 8, "branches": branches,
                    "base": {"root": BASE_LG + [0.5], "succ": succ(V_BASE)}, "arms": arms,
                    "n_true_target_moves": (i % 3) * 4})
    (root / "states").mkdir(exist_ok=True)
    (root / "states" / f"states_{cp}.json").write_text(json.dumps(
        {"states": states, "capture": {"c/o": {"capture_rate_win": 0.5, "capture_rate_loss": 1.0,
                                               "capture_rate_draw": 1.0}}}))
    (root / "raw" / cp / "raw").mkdir(parents=True)
    (root / "raw" / cp / "raw" / "raw_0_of_1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in raw))
    (root / "p1" / cp / "rows").mkdir(parents=True)
    (root / "p1" / cp / "rows" / "shard_0_of_1.jsonl").write_text("".join(json.dumps(r) + "\n" for r in p1))


def run(root: Path, extra=(), cps=("syn",), pool=""):
    out = root / "blinded.json"
    SM.main([*cps, "--out", str(out), "--raw-root", str(root / "raw"), "--p1-root", str(root / "p1"),
             "--states-dir", str(root / "states"), "--pool", pool, *extra])
    return json.loads(out.read_text())


def approx(x, y, tol=1e-4):
    return abs(x - y) <= tol


def softmax(x):
    z = np.exp(np.asarray(x) - max(x))
    return z / z.sum()


def with_data(fn):
    def wrapped():
        with tempfile.TemporaryDirectory(dir=HERE, prefix="_synthetic_") as d:
            root = Path(d)
            build(root)
            fn(root)
    wrapped.__name__ = fn.__name__
    return wrapped


@with_data
def test_metrics_known_answers(root):
    r = run(root)["cps"]["syn"]
    assert r["n_states"] == N_STATES and r["n_battles"] == N_STATES // PER_BATTLE
    assert r["codes"] == ["A01", "A02", "A03", "A04"]
    pc = r["per_code"]
    for c in ("BASE", "A02"):
        x = pc[c]["raw"]
        assert x["KL"]["mean"] == 0.0 and x["FLIP"]["mean"] == 0.0 and x["IMP_POLICY"]["mean"] == 0.0
        assert x["IMP_CRITIC"]["mean"] == 0.0 and x["PAIRWISE"]["mean"] == 1.0 and x["TOP12"]["mean"] == 1.0
        assert approx(x["ECE"]["mean"], (0.0875 + 0.1 + 0.0) / 3)
    a = pc["A01"]["raw"]
    pc_, p0 = softmax([0.0, 2.0, 1.0]), softmax([2.0, 1.0, 0.0])
    kl = float(np.sum(pc_ * np.log(pc_ / p0)))
    assert approx(a["KL"]["mean"], kl) and a["KL"]["ci"][0] == a["KL"]["ci"][1]
    assert a["FLIP"]["mean"] == 1.0 and a["IMP_POLICY"]["mean"] == -1.0 and a["IMP_CRITIC"]["mean"] == -1.0
    assert a["PAIRWISE"]["mean"] == 0.0 and a["TOP12"]["mean"] == 0.0
    assert approx(a["ECE"]["mean"], (0.7875 + 0.9 + 0.0) / 3)
    a3 = pc["A03"]
    assert approx(a3["raw"]["IMP_POLICY"]["mean"], -0.25) and approx(a3["reweighted"]["IMP_POLICY"]["mean"], -1 / 3)
    assert approx(a3["raw"]["FLIP"]["mean"], 0.5)
    lo, hi = a3["raw"]["IMP_POLICY"]["ci"]
    assert lo < -0.25 < hi                       # varies across battles -> a real interval
    c = r["contrasts"]
    assert c["IMP_POLICY"]["A01-A02"]["raw"] == {"mean": -1.0, "ci": [-1.0, -1.0], "n": N_STATES}
    assert approx(c["IMP_POLICY"]["A01-A03"]["raw"]["mean"], -0.75)
    assert c["IMP_CRITIC"]["A01-BASE"]["raw"]["mean"] == -1.0
    assert "A02-A01" not in c["IMP_POLICY"]


@with_data
def test_strata(root):
    r = run(root)["cps"]["syn"]
    s = r["strata"]
    assert s["unrev_0-1"]["n"] == 6 and s["unrev_2-3"]["n"] == 0 and s["unrev_4-5"]["n"] == 6
    assert s["unrev_2-3"]["per_code"]["A01"]["raw"]["IMP_POLICY"] == {"mean": None, "ci": None, "n": 0}
    # unrev_0-1 = battles 0..2 = win, loss, win -> 4 win states, 2 loss
    assert approx(s["unrev_0-1"]["per_code"]["A03"]["raw"]["IMP_POLICY"]["mean"], -4 * 0.5 / 6)
    assert [s[k]["n"] for k in ("tgt_0", "tgt_1-4", "tgt_5-8", "tgt_9+")] == [4, 4, 4, 0]
    assert s["tgt_5-8"]["contrasts"]["IMP_POLICY"]["A01-A02"]["raw"]["mean"] == -1.0


@with_data
def test_deterministic_and_partial(root):
    a = run(root)
    b = run(root)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    # drop one state's raw row: refused without --partial, read with it
    p = root / "raw" / "syn" / "raw" / "raw_0_of_1.jsonl"
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[1:]) + "\n")
    try:
        run(root)
        raise AssertionError("a missing state must be refused without --partial")
    except SystemExit:
        pass
    r = run(root, ["--partial"])
    assert r["cps"]["syn"]["n_states"] == N_STATES - 1 and r["cps"]["syn"]["missing"] == 1


def test_pair_acc_ties():
    q = np.array([0.5, 0.5, 0.2])
    tint = np.array([1, 0, 0])
    assert SM.pair_acc(q, tint, [(0, 1), (0, 2), (1, 2)]) == 0.75   # q tie = 1/2, T tie excluded
    assert math.isnan(SM.pair_acc(q, np.array([0, 0, 0]), [(0, 1)]))


def test_scorer_never_reads_the_key():
    src = (HERE / "score_model_input.py").read_text().replace(SM.__doc__, "")
    assert "arm_key" not in src and "hib_common" not in src and "label" not in src


@with_data
def test_unblind(root):
    run(root)
    key = {"A01": {"family": "PLACEBO", "p": 1.0, "label": "PLACEBO@1.00"},
           "A02": {"family": "TRUE", "p": 1.0, "label": "TRUE@1.00"},
           "A03": {"family": "SELFMAP", "p": 1.0, "label": "SELFMAP@1.00"},
           "A04": {"family": "TRUE", "p": 0.0, "label": "TRUE@0.00"}}
    (root / "key.json").write_text(json.dumps({"codes": key}))
    out = UB.main(["--key", str(root / "key.json"), "--blinded", str(root / "blinded.json"),
                   "--out", str(root / "unblinded.json")])
    u = out["cps"]["syn"]
    assert set(u["per_arm"]) == {"BASE", "TRUE@1.00", "PLACEBO@1.00", "SELFMAP@1.00", "TRUE@0.00"}
    assert u["per_arm"]["PLACEBO@1.00"]["raw"]["IMP_POLICY"]["mean"] == -1.0
    rc = u["registered_contrasts"]["IMP_POLICY"]
    # TRUE - PLACEBO = A02 - A01, stored as A01-A02 = -1 -> flipped to +1, CI [1, 1]
    assert rc["TRUE@1.00 - PLACEBO@1.00"]["raw"] == {"mean": 1.0, "ci": [1.0, 1.0], "n": N_STATES}
    # TRUE - SELFMAP = A02 - A03 = 0 - (-0.5 on wins) -> +0.25 raw, stored A02-A03 (no flip)
    assert approx(rc["TRUE@1.00 - SELFMAP@1.00"]["raw"]["mean"], 0.25)
    assert "PLACEBO@1.00 - TRUE@1.00" in u["contrasts"]["IMP_POLICY"]
    assert u["strata"]["unrev_0-1"]["registered_contrasts"]["IMP_POLICY"]["TRUE@1.00 - PLACEBO@1.00"]["raw"]["mean"] \
        == 1.0
    # p = 1.00: TRUE is BASE's bytes -> FLIP 0, KL 0 -> NET-IGNORES; the critic gains nothing -> NOT DETECTED
    v = u["verdicts"]["p=1.00 (secondary)"]
    assert v["NET"]["verdict"] == "NET-IGNORES" and v["CRITIC"]["verdict"] == "CRITIC-NOT-DETECTED"
    assert "error" in u["verdicts"]["p=0.95 (primary)"]          # no 0.95 arms in this key
    # a second key: the reversing arm A01 is TRUE@0.95 -> IMP -1 with FLIP 1 -> NET-MISUSES
    key2 = {"A01": {"family": "TRUE", "p": 0.95, "label": "TRUE@0.95"},
            "A02": {"family": "PLACEBO", "p": 0.95, "label": "PLACEBO@0.95"},
            "A03": {"family": "SELFMAP", "p": 0.95, "label": "SELFMAP@0.95"},
            "A04": {"family": "TRUE", "p": 0.0, "label": "TRUE@0.00"}}
    (root / "key2.json").write_text(json.dumps({"codes": key2}))
    out2 = UB.main(["--key", str(root / "key2.json"), "--blinded", str(root / "blinded.json"),
                    "--out", str(root / "unblinded2.json")])
    v2 = out2["cps"]["syn"]["verdicts"]["p=0.95 (primary)"]
    assert v2["NET"]["verdict"] == "NET-MISUSES" and v2["CRITIC"]["verdict"] == "CRITIC-NOT-DETECTED"
    assert out2["cps"]["syn"]["role"] == "other"
    assert out["cps"]["syn"]["identity_check"] == "PASS" and out2["cps"]["syn"]["identity_check"] == "PASS"
    # the identity teeth: a key naming the REVERSING arm A01 as TRUE@0.00 fails it -> the NET verdicts are void
    key3 = dict(key2)
    key3["A01"], key3["A04"] = ({"family": "TRUE", "p": 0.0, "label": "TRUE@0.00"},
                                {"family": "TRUE", "p": 0.95, "label": "TRUE@0.95"})
    (root / "key3.json").write_text(json.dumps({"codes": key3}))
    out3 = UB.main(["--key", str(root / "key3.json"), "--blinded", str(root / "blinded.json"),
                    "--out", str(root / "unblinded3.json")])
    u3 = out3["cps"]["syn"]
    assert u3["identity_check"] == "FAIL"
    v3 = u3["verdicts"]["p=0.95 (primary)"]
    assert v3["NET"]["verdict"] == "VOID (identity_check FAIL)" and v3["CRITIC"]["flag"] == "identity_check FAIL"
    assert u3["identity"]["TRUE@0.00"]["flip_count"] == N_STATES


@with_data
def test_top1_mismatch(root):
    p = root / "raw" / "syn" / "raw" / "raw_0_of_1.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    rows[0]["base"]["root"] = [1.0, 2.0, 0.0] + [-9.0] * 8 + [0.5]          # BASE now prefers action 1
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    r = run(root)["cps"]["syn"]
    assert r["top1_checked"] is True and r["top1_mismatch"] == [rows[0]["id"]]
    assert r["n_states"] == N_STATES                                         # the read continues
    # a checkpoint scored on ANOTHER checkpoint's states (as end2_s2002 on end2's) is not checked
    (root / "raw" / "syn3" / "raw").mkdir(parents=True)
    (root / "raw" / "syn3" / "raw" / "raw_0_of_1.jsonl").write_text(p.read_text())
    SM.STATES_OF["syn3"] = "syn"
    try:
        r3 = run(root, cps=("syn3",))["cps"]["syn3"]
    finally:
        del SM.STATES_OF["syn3"]
    assert r3["top1_checked"] is False and r3["top1_mismatch"] == []


@with_data
def test_pooled(root):
    build(root, cp="syn2")
    out = run(root, cps=("syn", "syn2"), pool="syn,syn2")
    po, one = out["pooled"], out["cps"]["syn"]
    assert po["n_states"] == 2 * N_STATES and po["n_battles"] == 2 * (N_STATES // PER_BATTLE)
    for c in ("A01", "A03", "BASE"):
        for k in ("IMP_POLICY", "IMP_CRITIC", "FLIP", "KL"):
            assert approx(po["per_code"][c]["raw"][k]["mean"], one["per_code"][c]["raw"][k]["mean"])
    w1 = np.diff(one["per_code"]["A03"]["raw"]["IMP_POLICY"]["ci"])[0]
    w2 = np.diff(po["per_code"]["A03"]["raw"]["IMP_POLICY"]["ci"])[0]
    assert w2 < w1                                # twice the battles -> a narrower interval
    assert po["contrasts"]["IMP_POLICY"]["A01-A02"]["raw"]["mean"] == -1.0
    key = {"A01": {"family": "TRUE", "p": 0.95, "label": "TRUE@0.95"},
           "A02": {"family": "PLACEBO", "p": 0.95, "label": "PLACEBO@0.95"},
           "A03": {"family": "SELFMAP", "p": 0.95, "label": "SELFMAP@0.95"},
           "A04": {"family": "TRUE", "p": 0.0, "label": "TRUE@0.00"}}
    (root / "key.json").write_text(json.dumps({"codes": key}))
    u = UB.main(["--key", str(root / "key.json"), "--blinded", str(root / "blinded.json"),
                 "--out", str(root / "unblinded.json")])
    assert u["pooled"]["identity_check"] == "PASS"
    assert u["pooled"]["verdicts"]["p=0.95 (primary)"]["NET"]["verdict"] == "NET-MISUSES"
    try:
        run(root, cps=("syn",), pool="syn,syn2")
        raise AssertionError("a pool naming an unscored checkpoint must be refused")
    except SystemExit:
        pass


# ------------------------------------------------------------------------------------------------
# the registered rules, one branch at a time (hand-built unblinded results at p = 0.95)
# ------------------------------------------------------------------------------------------------

def st(lo, hi):
    return {"mean": (lo + hi) / 2, "ci": [lo, hi], "n": 600}


def neg(lo, hi):
    return st(-hi, -lo)


def mk(imp, fl, kl, dP, dS, dZc=(0.0, 0.0), dPc=(0.0, 0.0)):
    """dS and dPc are STORED reversed (SELFMAP - TRUE, PLACEBO - TRUE) to exercise the orientation flip."""
    arms = {a: {"raw": {}} for a in ("PLACEBO@0.95", "SELFMAP@0.95", "TRUE@0.00")}
    arms["TRUE@0.95"] = {"raw": {"IMP_POLICY": st(*imp), "FLIP": {"mean": fl}, "KL": {"mean": kl}}}
    return {"per_arm": arms,
            "contrasts": {"IMP_POLICY": {"TRUE@0.95 - PLACEBO@0.95": {"raw": st(*dP)},
                                         "SELFMAP@0.95 - TRUE@0.95": {"raw": neg(*dS)}},
                          "IMP_CRITIC": {"TRUE@0.95 - TRUE@0.00": {"raw": st(*dZc)},
                                         "PLACEBO@0.95 - TRUE@0.95": {"raw": neg(*dPc)}}}}


def net(u):
    return UB.net_rule(u, 0.95)


def test_rule_net_uses():
    r = net(mk((0.01, 0.03), 0.10, 0.05, (0.002, 0.02), (0.001, 0.02)))
    assert r["verdict"] == "NET-USES" and r["held"] == ["NET-USES"]
    # a lower bound inside the rule-8 band (+0.0004 <= 0.0005) is not clear -> falls through
    r = net(mk((0.0004, 0.02), 0.10, 0.05, (0.002, 0.02), (0.001, 0.02)))
    assert r["verdict"] == "NET-NOT-DETECTED" and r["held"] == []
    # the paired contrast must clear too
    assert net(mk((0.01, 0.03), 0.10, 0.05, (0.0005, 0.02), (0.001, 0.02)))["verdict"] == "NET-NOT-DETECTED"


def test_rule_net_misuses_and_order():
    r = net(mk((-0.03, -0.001), 0.05, 0.05, (-0.004, 0.004), (-0.004, 0.004)))
    assert r["verdict"] == "NET-MISUSES"
    assert r["held"] == ["NET-MISUSES", "NET-CONFIDENCE-ONLY"]       # both hold; MISUSES comes first
    # an upper bound inside the band (-0.0004) is not clearly below 0
    assert net(mk((-0.03, -0.0004), 0.05, 0.05, (-0.01, 0.01), (-0.01, 0.01)))["verdict"] == "NET-NOT-DETECTED"
    # FLIP at the 0.02 boundary (inside the band) is neither >= nor <
    assert net(mk((-0.03, -0.001), 0.0204, 0.005, (-0.01, 0.01), (-0.01, 0.01)))["verdict"] == "NET-NOT-DETECTED"


def test_rule_net_confidence_only():
    r = net(mk((-0.01, 0.01), 0.05, 0.05, (-0.004, 0.004), (-0.003, 0.0044)))
    assert r["verdict"] == "NET-CONFIDENCE-ONLY"
    # an upper bound at +0.0046 is inside the band of +0.005 -> not clearly inside
    assert net(mk((-0.01, 0.01), 0.05, 0.05, (-0.004, 0.004), (-0.003, 0.0046)))["verdict"] == "NET-NOT-DETECTED"
    # both contrasts must be inside
    assert net(mk((-0.01, 0.01), 0.05, 0.05, (-0.006, 0.004), (-0.003, 0.004)))["verdict"] == "NET-NOT-DETECTED"


def test_rule_net_ignores():
    assert net(mk((-0.001, 0.001), 0.01, 0.005, (-0.002, 0.002), (-0.002, 0.002)))["verdict"] == "NET-IGNORES"
    assert net(mk((-0.001, 0.001), 0.0196, 0.005, (-0.002, 0.002), (-0.002, 0.002)))["verdict"] == "NET-NOT-DETECTED"
    assert net(mk((-0.001, 0.001), 0.01, 0.0097, (-0.002, 0.002), (-0.002, 0.002)))["verdict"] == "NET-NOT-DETECTED"


def test_rule_net_not_detected():
    assert net(mk((-0.01, 0.02), 0.05, 0.05, (-0.01, 0.02), (-0.01, 0.02)))["verdict"] == "NET-NOT-DETECTED"


def test_rule_critic():
    base = ((-0.001, 0.001), 0.01, 0.005, (-0.002, 0.002), (-0.002, 0.002))
    r = UB.critic_rule(mk(*base, dZc=(0.001, 0.01), dPc=(0.002, 0.02)), 0.95)
    assert r["verdict"] == "CRITIC-USES"
    assert r["evidence"]["IMP_CRIT(TRUE) - IMP_CRIT(PLACEBO)"]["ci"] == [0.002, 0.02]   # the stored reverse, flipped
    assert UB.critic_rule(mk(*base, dZc=(0.001, 0.01), dPc=(0.0003, 0.02)), 0.95)["verdict"] == "CRITIC-NOT-DETECTED"
    assert UB.critic_rule(mk(*base, dZc=(0.0, 0.01), dPc=(0.002, 0.02)), 0.95)["verdict"] == "CRITIC-NOT-DETECTED"


def test_verdicts_report_absent_arms():
    u = mk((0.01, 0.03), 0.10, 0.05, (0.002, 0.02), (0.001, 0.02))
    v = UB.verdicts(u)
    assert v["p=0.95 (primary)"]["NET"]["verdict"] == "NET-USES"
    assert "error" in v["p=1.00 (secondary)"]


def test_unblind_refuses_a_foreign_key():
    res = {"cp": "x", "codes": ["A01"]}
    try:
        UB.unblind_cp(res, {"A02": {"family": "TRUE", "p": 1.0, "label": "TRUE@1.00"}})
        raise AssertionError("a key with other codes must be refused")
    except SystemExit:
        pass


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
