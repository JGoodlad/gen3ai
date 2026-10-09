"""Step 4: fold every capture + probe into arm-level tables. DESCRIPTIVE / exploratory; NOT the screen.

Arms are compared by their SEEDS (legacy n = 6, static n = 5 finished seeds): each number is the arm mean over
seeds, its seed sd, Δ = static − legacy and a Welch t over seeds (a description of the seed spread, never a
registered test). Stages: 2M / 5M / 10M (the nearest checkpoint; static s1003's is 9.6M) / final (15M).

Writes analysis.json beside this file and prints a digest.
"""
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARCH = Path("/home/goodlad/gen3ai_archive/static_diag_2026-10-09")
STAGES = ("2M", "5M", "10M", "final")
CATS = ("attack", "switch", "status", "setup", "hazard", "recovery")
QG = ("our_active", "their_active", "our_bench", "E3", "board")
KG = ("our_mons", "their_mons", "board", "E3", "E4_E5_other", "events")


def labels():
    out = defaultdict(dict)                      # (arm, stage) -> seed -> label
    for p in sorted((ARCH / "cap").glob("*.npz")):
        m = re.match(r"([LS])(\d)_(\w+)$", p.stem)
        arm = "legacy" if m.group(1) == "L" else "static"
        out[(arm, m.group(3))][int(m.group(2))] = p.stem
    return out


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return None
    se = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return round(float((b.mean() - a.mean()) / se), 2) if se > 0 else None


def cmp(L, S, nd=4):
    return {"legacy": round(float(np.mean(L)), nd), "legacy_sd": round(float(np.std(L, ddof=1)), nd),
            "static": round(float(np.mean(S)), nd), "static_sd": round(float(np.std(S, ddof=1)), nd),
            "delta": round(float(np.mean(S) - np.mean(L)), nd), "welch_t": welch(L, S)}


def main():
    LB = labels()
    meta = json.loads((ARCH / "bank_meta.json").read_text())
    N = len(meta)
    masks = np.load(ARCH / "bank_rows.npz")["masks"]
    out = {"schema": "static_diag_analysis_v1", "tag": "DESCRIPTIVE",
           "seeds": {f"{a}/{s}": sorted(v) for (a, s), v in LB.items()}}
    cache = {}

    def cap(label):
        if label not in cache:
            cache[label] = dict(np.load(ARCH / "cap" / f"{label}.npz"))
        return cache[label]

    # ---------------- A. probes
    pr = {}
    for st in STAGES:
        P = {a: [json.loads((ARCH / "probe" / f"{lab}.json").read_text())["probes"] for lab in LB[(a, st)].values()]
             for a in ("legacy", "static")}
        keys = sorted(set.intersection(*[set(p) for p in P["legacy"] + P["static"]]))
        pr[st] = {k: {d: cmp([p[k][d] for p in P["legacy"]], [p[k][d] for p in P["static"]], 3) for d in ("in", "L1", "L2")}
                  | {"kind": P["legacy"][0][k]["kind"]} for k in keys}
    out["probes"] = pr

    # ---------------- B. attention
    att = {}
    for st in STAGES:
        E = {a: np.stack([cap(l)["attn_ent"].astype(np.float32).mean(0) for l in LB[(a, st)].values()]) for a in ("legacy", "static")}
        M = {a: np.stack([cap(l)["attn_mass"].astype(np.float32).mean(0) for l in LB[(a, st)].values()]) for a in ("legacy", "static")}
        blk = {}
        for li in range(2):
            for qi, q in enumerate(QG):
                blk[f"entropy L{li + 1} {q} (head mean)"] = cmp(E["legacy"][:, li, :, qi].mean(1), E["static"][:, li, :, qi].mean(1), 3)
                for h in range(4):
                    blk[f"entropy L{li + 1} {q} h{h}"] = cmp(E["legacy"][:, li, h, qi], E["static"][:, li, h, qi], 3)
            for qn, qi in (("our_active", 0), ("their_active", 1)):
                for gi, g in enumerate(KG):
                    blk[f"mass L{li + 1} {qn}->{g} (head mean)"] = cmp(M["legacy"][:, li, :, qi, gi].mean(1),
                                                                     M["static"][:, li, :, qi, gi].mean(1), 3)
        att[st] = blk
    out["attention"] = att

    # ---------------- C. policy + critic per stratum (final and 10M)
    phase = np.array([d["phase"] for d in meta])
    kind = np.array([d["kind"] for d in meta])
    lb = np.array([d["legal_bucket"] for d in meta])
    oc = np.array([d["opp_class"] for d in meta])
    outc = np.array([d["outcome"] for d in meta])
    cat = np.full((N, 11), "", dtype=object)
    for i, d in enumerate(meta):
        for k, c in d["cats"].items():
            cat[i, int(k)] = c
    strata = {"all": np.ones(N, bool)}
    for name, arr in (("phase", phase), ("kind", kind), ("legal", lb), ("opp_class", oc), ("outcome", outc)):
        for v in sorted(set(arr)):
            strata[f"{name}={v}"] = arr == v
    free = kind == "free"
    for c in CATS[1:]:
        strata[f"free & {c} legal"] = free & (cat == c).any(1)
    y = np.where(outc == "win", 1.0, np.where(outc == "loss", 0.0, np.nan))

    pol = {}
    for st in ("10M", "final"):
        probs = {a: {s: cap(l)["probs"].astype(np.float64) for s, l in LB[(a, st)].items()} for a in ("legacy", "static")}
        winp = {a: {s: cap(l)["win_prob"].astype(np.float64) for s, l in LB[(a, st)].items()} for a in ("legacy", "static")}
        for a in probs:
            for s in probs[a]:
                p = probs[a][s] * masks
                probs[a][s] = p / p.sum(1, keepdims=True)
        Lseeds = sorted(probs["legacy"])
        Sseeds = sorted(probs["static"])

        def consensus(excl):
            return np.mean([probs["legacy"][s] for s in Lseeds if s != excl], 0).argmax(1)
        # leave-one-out legacy consensus: a legacy seed excludes itself; static seed i excludes legacy seed i
        cons = {("legacy", s): consensus(s) for s in Lseeds} | {("static", s): consensus(s) for s in Sseeds}
        per = {}
        for a in ("legacy", "static"):
            for s in probs[a]:
                p = probs[a][s]
                H = -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0)).sum(1)
                top1 = p.max(1)
                agree = (p.argmax(1) == cons[(a, s)]).astype(float)
                mcons = p[np.arange(N), cons[(a, s)]]
                w = winp[a][s]
                br = (w - y) ** 2
                ll = -(y * np.log(np.clip(w, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - w, 1e-6, 1)))
                cm = {c: (p * (cat == c)).sum(1) for c in CATS}
                per[(a, s)] = (H, top1, agree, mcons, br, ll, cm)
        blk = {}
        for name, m in strata.items():
            row = {"n": int(m.sum())}
            for fi, fn in enumerate(("entropy", "top1_mass", "agree_legacy_consensus", "mass_on_legacy_consensus")):
                row[fn] = cmp([per[("legacy", s)][fi][m].mean() for s in Lseeds],
                              [per[("static", s)][fi][m].mean() for s in Sseeds])
            mm = m & np.isfinite(y)
            row["critic_brier"] = cmp([per[("legacy", s)][4][mm].mean() for s in Lseeds], [per[("static", s)][4][mm].mean() for s in Sseeds])
            row["critic_logloss"] = cmp([per[("legacy", s)][5][mm].mean() for s in Lseeds], [per[("static", s)][5][mm].mean() for s in Sseeds])
            fm = m & free
            if fm.sum() > 50:
                row["free_cat_mass"] = {c: cmp([per[("legacy", s)][6][c][fm].mean() for s in Lseeds],
                                               [per[("static", s)][6][c][fm].mean() for s in Sseeds]) for c in CATS}
            blk[name] = row
        pol[st] = blk
    out["policy"] = pol
    (HERE / "analysis.json").write_text(json.dumps(out, indent=1) + "\n")

    # ---------------- digest
    def f(c):
        return f"L {c['legacy']:.3f}±{c['legacy_sd']:.3f} S {c['static']:.3f}±{c['static_sd']:.3f} Δ {c['delta']:+.3f} t {c['welch_t']}"
    print("== PROBES (final; in / L1 / L2 deltas)")
    for st in ("5M", "final"):
        print("-- stage", st)
        for k, v in pr[st].items():
            print(f"{k:34s} " + " | ".join(f"{d}: L {v[d]['legacy']:.3f} S {v[d]['static']:.3f} Δ {v[d]['delta']:+.3f} (t {v[d]['welch_t']})" for d in ("in", "L1", "L2")))
    print("== ATTENTION entropy (head mean) by stage")
    for li in (1, 2):
        for q in QG:
            print(f"L{li} {q:13s} " + " | ".join(f"{st}: L {att[st][f'entropy L{li} {q} (head mean)']['legacy']:.3f} S {att[st][f'entropy L{li} {q} (head mean)']['static']:.3f}" for st in STAGES))
    print("== ATTENTION mass from our_active (head mean), final")
    for li in (1, 2):
        for g in KG:
            print(f"L{li} OA->{g:12s} {f(att['final'][f'mass L{li} our_active->{g} (head mean)'])}")
    for st in ("10M", "final"):
        print(f"== POLICY / CRITIC @ {st}")
        for name, row in pol[st].items():
            print(f"{name:26s} n={row['n']:5d} agree {row['agree_legacy_consensus']['legacy']:.3f}/{row['agree_legacy_consensus']['static']:.3f} "
                  f"(Δ {row['agree_legacy_consensus']['delta']:+.3f} t {row['agree_legacy_consensus']['welch_t']}) "
                  f"H {row['entropy']['legacy']:.3f}/{row['entropy']['static']:.3f} (t {row['entropy']['welch_t']}) "
                  f"brier {row['critic_brier']['legacy']:.4f}/{row['critic_brier']['static']:.4f} (t {row['critic_brier']['welch_t']})")
        print("free cat mass (all):", {c: (v['legacy'], v['static'], v['welch_t']) for c, v in pol[st]["all"]["free_cat_mass"].items()})


if __name__ == "__main__":
    main()
