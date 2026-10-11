"""Reviewer's POST-HOC read of banked CRN truth rows: the pairing VRF at the TERMINAL in S0 (today's
stream mode), the separable share, the acting-rule resolution, split-half agreement. Read-only.

Data: crn_label_refit_2026-10-06's committed held-out truth rows (rb_x5ab_blob_s1001 greedy on both
sides, Rust core, `sodium,` seeds shared by every action, 32 seeds) + the Lane S bank + the blob root
read `blob.probs.npz` (ARCHIVE only: ~/gen3ai_archive/crn_label_refit_2026-10-06/rows/reads/).

    python pairing_read.py <repo root> <archive rows dir>   # design_value_discrimination.md §12"""
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(sys.argv[1])
ARCH = Path(sys.argv[2])
M = REPO / "designs/research_state/measurements"
rows = [json.loads(line) for line in
        gzip.open(M / "crn_label_refit_2026-10-06/rows/truth_held_S32.compact.jsonl.gz", "rt")]
with gzip.open(M / "m5_laneS/bank_v1/decisions.jsonl.gz", "rt") as f:
    decs = [json.loads(x) for x in f]
dec = {d["id"]: d for d in decs}
z = np.load(ARCH / "reads/blob.probs.npz")
assert str(z["ids_sha256"]) == hashlib.sha256("\n".join(d["id"] for d in decs).encode()).hexdigest()
logit = {d["id"]: z["logits"][i].astype(np.float64) for i, d in enumerate(decs)}
DEC = {"+": 1.0, "-": -1.0, "0": 0.0}
sub = json.loads((M / "crn_label_refit_2026-10-06/rows/subset_train.json").read_text())["ids"] + \
    json.loads((M / "crn_label_refit_2026-10-06/rows/subset_held.json").read_text())["ids"]
gaps = {}
for i in sub:
    d = dec[i]
    if d["kind"] != "free" or not (2 <= d["turn"] <= 40) or d["n_legal"] < 3:
        continue
    m = np.array([c == "1" for c in d["mask"]])
    lg = np.sort(logit[i][m])[::-1]
    gaps[i] = lg[0] - lg[1]
thr = np.percentile(list(gaps.values()), 40)
cont = {i for i, g in gaps.items() if g < thr}


def cls(a, b):
    sw = (a < 6) + (b < 6)
    return {0: "move-move", 1: "move-switch", 2: "switch-switch"}[sw]


P = []
for r in rows:
    if not r.get("ok"):
        continue
    did = r["id"]
    d = dec[did]
    acts = sorted(int(a) for a in r["outcomes"])
    o = {a: np.array([DEC[c] for c in r["outcomes"][str(a)]]) for a in acts}
    m = np.array([c == "1" for c in d["mask"]])
    lg = logit[did].copy()
    lg[~m] = -np.inf
    order = list(np.argsort(-lg))
    top1, top2 = int(order[0]), int(order[1])
    for i, a in enumerate(acts):
        for b in acts[i + 1:]:
            x, y = o[a], o[b]
            dpair = x - y
            P.append(dict(battle=d["battle"], did=did, cont=did in cont, cls=cls(a, b),
                          t12=({a, b} == {top1, top2}),
                          s2a=x.var(ddof=1), s2b=y.var(ddof=1), s2p=dpair.var(ddof=1),
                          s2der=(x - np.roll(y, 1)).var(ddof=1),
                          mean=dpair.mean(), h1=dpair[:16].mean(), h2=dpair[16:].mean(), d=dpair))


def vrf(sel):
    su = sum(p["s2a"] + p["s2b"] for p in sel)
    sp = sum(p["s2p"] for p in sel)
    sd = sum(p["s2der"] for p in sel)
    return (su / sp if sp else float("nan")), (su / sd if sd else float("nan"))


def boot(sel, f, B=1000, seed=20261011):
    by = defaultdict(list)
    for p in sel:
        by[p["battle"]].append(p)
    keys = sorted(by)
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(B):
        ks = rng.choice(len(keys), len(keys))
        v.append(f([p for k in ks for p in by[keys[k]]]))
    return np.percentile(v, [2.5, 97.5]).round(3).tolist()


def report(name, sel):
    sel = [p for p in sel if p["s2a"] + p["s2b"] > 0]
    if not sel:
        return
    v, vd = vrf(sel)
    num = sum((p["s2a"] + p["s2b"] - p["s2p"]) / 2 for p in sel)
    den = sum(np.sqrt(p["s2a"] * p["s2b"]) for p in sel)
    se = np.array([np.sqrt(p["s2p"] / 32) for p in sel])
    mu = np.array([p["mean"] for p in sel])
    sep = np.mean((np.abs(mu) > 1.96 * se) & (se > 0))
    act = {}
    for R in (4, 8, 16, 32):
        ok = 0
        for p in sel:
            # the deleted production rule (search_dividend/playoff.py @ 8b5c7e4d^): |mean| >= 2 SE,
            # n >= MIN_PAIRS = 4, SE (ddof=1) floored at 0.5/n on the win scale = 1.0/n on +-1
            dd = p["d"][:R]
            s = max(dd.std(ddof=1) / np.sqrt(R), 1.0 / R)
            ok += bool(abs(dd.mean()) >= 2 * s)
        act[R] = round(ok / len(sel), 4)
    h = [(p["h1"], p["h2"]) for p in sel if p["h1"] != 0 and p["h2"] != 0]
    sha = np.mean([np.sign(a) == np.sign(b) for a, b in h]) if h else float("nan")
    print(json.dumps(dict(
        name=name, n_pairs=len(sel), n_battles=len({p["battle"] for p in sel}),
        VRF_terminal=round(v, 3), VRF_ci=boot(sel, lambda s: vrf(s)[0]), VRF_derangement=round(vd, 3),
        VRF_derangement_ci=boot(sel, lambda s: vrf(s)[1]),
        rho_pooled=round(num / den, 3), separable_M32=round(float(sep), 4), acting_rule_resolved=act,
        split_half_sign_agree=round(float(sha), 4), n_split_nontied=len(h),
        mean_abs_dhat=round(float(np.mean(np.abs(mu))), 4),
        sd_dhat=round(float(np.std(mu)), 4),
        mean_s2pair_single=round(float(np.mean([p["s2p"] for p in sel])), 4),
        # method of moments: var(true gap) = var(gap estimate) - mean paired noise / 32; WIN scale
        true_gap_sd_win=round(float(np.sqrt(max(np.var(mu) - np.mean([p["s2p"] for p in sel]) / 32, 0))) / 2, 4),
        frac_dhat_zero=round(float(np.mean(mu == 0)), 4))))


report("all held-out pairs", P)
report("contested, all pairs", [p for p in P if p["cont"]])
report("contested, top1|top2", [p for p in P if p["cont"] and p["t12"]])
for c in ("move-move", "move-switch", "switch-switch"):
    report(f"contested, {c}", [p for p in P if p["cont"] and p["cls"] == c])
print("thr", round(float(thr), 4), "n_cont_held", len({p["did"] for p in P if p["cont"]}),
      "n_held", len({p["did"] for p in P}))
