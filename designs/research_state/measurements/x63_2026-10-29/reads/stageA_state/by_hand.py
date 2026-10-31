"""X63 Stage A by hand (FROZEN record, never a tool): PREREG §6-§7 from the counted totals in inputs.json, in plain
floats, beside the rule's decision.json. Run: python by_hand.py (from this directory)."""
import json, math
from pathlib import Path

H = Path(__file__).parent
inp = json.loads((H / "inputs.json").read_text())
dec = json.loads((H / "decision.json").read_text())
Z = 1.959964


def p(group, lab, ts):
    m = inp["groups"][group]["models"][lab][ts]
    return m["wins"] / m["n"], m["n"]


def kvk(group, ts):
    (pu, gu), (p1, g1), (p2, g2) = p(group, "x63u_s2601", ts), p(group, "x26c_s2601", ts), p(group, "x63c_s6302", ts)
    delta = pu - (p1 + p2) / 2
    b_on, b1, b2 = pu * (1 - pu) / gu, p1 * (1 - p1) / g1, p2 * (1 - p2) / g2
    B = b_on + (b1 + b2) / 4
    var = (p1 - p2) ** 2 / 2                       # ddof 1 over two points
    s2 = var - (b1 + b2) / 2
    sig = math.sqrt(max(0.0, s2))
    se = math.sqrt(B + max(0.01, sig) ** 2 * (1 + 1 / 2))
    sigma_cc = math.sqrt(max(0.0, (p2 - p1) ** 2 - b1 - b2) / 2)   # X63-C PREREG §3
    return {"delta": delta, "B": B, "sigma_hat": sig, "sigma_cc": sigma_cc, "se": se,
            "lb": delta - Z * se, "ub": delta + Z * se, "p": [pu, p1, p2]}


out = {"primary": kvk("primary", "forum72"), "home": kvk("home", "home72")}
f = out["primary"]
out["class"] = "CONTINUE" if f["delta"] >= 0.01 + 1e-9 else "NOT PURSUED"
rule = dec["primary"]
out["matches_rule"] = {k: abs(f[k2] - rule[k]) < 1e-12 for k, k2 in
                       (("delta", "delta"), ("se", "se"), ("lb", "lb"), ("ub", "ub"), ("sigma_hat", "sigma_hat"))}
out["matches_rule"]["class"] = out["class"] == dec["decision"]["class"]
out["matches_rule"]["sigma_cc"] = abs(f["sigma_cc"] - dec["decision"]["sigma_cc"]) < 1e-12
print(json.dumps(out, indent=1))
