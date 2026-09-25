"""Analyse the belief win-rate A/B (PREDICTION.md §3-§5).

    python analyze.py report <rows_dir> <out_dir>                 # results.json + tables.md
    python analyze.py compare <rows_a> <rows_b> <fields> [cells]  # the registered repeat checks

Paired bootstrap over battle INDICES (10,000 resamples, drawn jointly across every cell). Each
quantity is a ratio of sums over the resampled battles, so no correlation across battles is computed.
"""
import glob
import json
import os
import sys

import numpy as np

PRIMARY = [("pool", "learned"), ("pool", "prior"), ("ladder", "learned"), ("ladder", "prior")]
SECONDARY = [("pool", "move"), ("ladder", "move")]
B = 10_000
SEED = 20260925


def load(rows_dir):
    rows = {}
    for f in sorted(glob.glob(os.path.join(rows_dir, "rows_w*.jsonl"))):
        for ln in open(f):
            r = json.loads(ln)
            rows[(r["col"], r["belief"], r["i"])] = r   # a resumed re-run keeps the latest line
    return rows


def score(r):
    return {1: 1.0, 0: 0.0, -1: 0.5}[r["winner"]]


def boot_weights(n, rng, chunk=500):
    for s in range(0, B, chunk):
        m = min(chunk, B - s)
        yield rng.multinomial(n, np.full(n, 1.0 / n), size=m).astype(np.float64)


def ci(samples):
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return [float(lo), float(hi)]


def tag(point, c):
    if c[0] > 0:
        return "POSITIVE"
    if c[1] < 0:
        return "NEGATIVE"
    return "NOT DETECTED"


def report(rows_dir, out_dir):
    rows = load(rows_dir)
    idx_all = sorted({k[2] for k in rows})
    cells = PRIMARY + SECONDARY
    attempted = {c: sum(1 for i in idx_all if (c[0], c[1], i) in rows) for c in cells}
    failed = {c: sum(1 for i in idx_all if (c[0], c[1], i) in rows and rows[(c[0], c[1], i)]["status"] != "ok")
              for c in cells}
    fail_share = {c: failed[c] / attempted[c] if attempted[c] else None for c in cells}
    inconclusive = [f"{c[0]}:{c[1]}" for c in PRIMARY if attempted[c] and fail_share[c] > 0.25]
    errs = {}
    for r in rows.values():
        if r["status"] != "ok":
            errs.setdefault(f"{r['col']}:{r['belief']}", []).append([r["i"], r["status"], r["err"]])

    def complete(cs):
        return [i for i in idx_all if all((c[0], c[1], i) in rows for c in cs)]

    def ok(cs):
        return [i for i in complete(cs) if all(rows[(c[0], c[1], i)]["status"] == "ok" for c in cs)]

    prim = ok(PRIMARY)
    sec = ok(PRIMARY + SECONDARY)
    res = {"n_indices_seen": len(idx_all), "attempted": {f"{c[0]}:{c[1]}": attempted[c] for c in cells},
           "failed": {f"{c[0]}:{c[1]}": failed[c] for c in cells},
           "fail_share": {f"{c[0]}:{c[1]}": fail_share[c] for c in cells},
           "primary_indices_complete": len(complete(PRIMARY)), "primary_indices_ok": len(prim),
           "primary_dropped": len(complete(PRIMARY)) - len(prim),
           "secondary_indices_complete": len(complete(PRIMARY + SECONDARY)), "secondary_indices_ok": len(sec),
           "INCONCLUSIVE_cells": inconclusive, "errors": errs}
    rng = np.random.default_rng(SEED)

    # ---------------- win rates, simple effects, DiD ----------------
    def wr_block(idx, cs, contrasts):
        Y = np.array([[score(rows[(c[0], c[1], i)]) for c in cs] for i in idx])        # [n, C]
        n = len(idx)
        point = Y.mean(0)
        samp = np.concatenate([w @ Y / n for w in boot_weights(n, rng)])                 # [B, C]
        out = {"n": n, "wr": {}, "contrasts": {}}
        for j, c in enumerate(cs):
            out["wr"][f"{c[0]}:{c[1]}"] = {"point": float(point[j]), "ci": ci(samp[:, j])}
        col = {f"{c[0]}:{c[1]}": j for j, c in enumerate(cs)}
        for name, terms in contrasts.items():
            p = sum(s * point[col[k]] for k, s in terms)
            sm = sum(s * samp[:, col[k]] for k, s in terms)
            c_ = ci(sm)
            out["contrasts"][name] = {"point": float(p), "ci": c_, "read": tag(p, c_)}
        # discordance: share of indices whose outcome differs between the belief rows
        for colname in ("pool", "ladder"):
            for alt in ("prior", "move"):
                if f"{colname}:{alt}" in col:
                    d = Y[:, col[f"{colname}:{alt}"]] != Y[:, col[f"{colname}:learned"]]
                    out.setdefault("discordant_share", {})[f"{colname}:{alt}"] = float(d.mean())
        return out

    res["primary"] = wr_block(prim, PRIMARY, {
        "E_pool = WR_prior - WR_learned (POOL)": [("pool:prior", 1), ("pool:learned", -1)],
        "E_ladder = WR_prior - WR_learned (LADDER)": [("ladder:prior", 1), ("ladder:learned", -1)],
        "DiD = E_ladder - E_pool": [("ladder:prior", 1), ("ladder:learned", -1),
                                   ("pool:prior", -1), ("pool:learned", 1)],
        "WR_learned ladder - pool": [("ladder:learned", 1), ("pool:learned", -1)],
    })
    if sec:
        res["secondary"] = wr_block(sec, PRIMARY + SECONDARY, {
            "E^move_pool = WR_move - WR_learned (POOL)": [("pool:move", 1), ("pool:learned", -1)],
            "E^move_ladder = WR_move - WR_learned (LADDER)": [("ladder:move", 1), ("ladder:learned", -1)],
            "DiD^move": [("ladder:move", 1), ("ladder:learned", -1), ("pool:move", -1), ("pool:learned", 1)],
        })

    # ---------------- side read (learned cells, primary indices) ----------------
    side = {}
    sanity_total = 0
    for colname in ("pool", "ladder"):
        cnt = np.zeros((len(prim), 7))
        ch = {"all": np.zeros((len(prim), 7)), "move": np.zeros((len(prim), 7))}
        for a, i in enumerate(prim):
            for k, s_off, s_all, s_mv in rows[(colname, "learned", i)]["side"]:
                cnt[a, k] += 1
                ch["all"][a, k] += s_all
                ch["move"][a, k] += s_mv
                sanity_total += s_off
        rng_s = np.random.default_rng(SEED + 1)
        ws = list(boot_weights(len(prim), rng_s))
        for scope in ("all", "move"):
            per = {}
            for kk in list(range(7)) + ["all"]:
                c_ = cnt.sum(1) if kk == "all" else cnt[:, kk]
                h_ = ch[scope].sum(1) if kk == "all" else ch[scope][:, kk]
                if c_.sum() == 0:
                    continue
                samp = np.concatenate([(w @ h_) / np.maximum(w @ c_, 1e-12) for w in ws])
                per[str(kk)] = {"n_decisions": int(c_.sum()), "rate": float(h_.sum() / c_.sum()), "ci": ci(samp)}
            side[f"{colname}:{scope}"] = per
    res["side_read"] = side
    res["side_read_sanity_off_mismatches"] = int(sanity_total)

    # ---------------- descriptive ----------------
    desc = {}
    for c in PRIMARY + SECONDARY:
        idx = prim if c in PRIMARY else sec
        if not idx:
            continue
        t = [rows[(c[0], c[1], i)]["turns"] for i in idx]
        d = [rows[(c[0], c[1], i)]["n_dec"] for i in idx]
        draws = sum(1 for i in idx if rows[(c[0], c[1], i)]["winner"] == -1)
        desc[f"{c[0]}:{c[1]}"] = {"mean_turns": float(np.mean(t)), "max_turns": int(np.max(t)),
                                   "mean_decisions": float(np.mean(d)), "draws": draws}
    res["descriptive"] = desc

    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, "results.json"), "w"), indent=1)
    render(res, os.path.join(out_dir, "tables.md"))
    print(open(os.path.join(out_dir, "tables.md")).read())


def pct(x):
    return f"{100 * x:+.1f}"


def render(res, path):
    L = ["# Belief win-rate A/B — generated tables (`scripts/analyze.py report`)", ""]
    L.append(f"Primary indices OK: **{res['primary_indices_ok']}** of {res['primary_indices_complete']} "
             f"complete (dropped {res['primary_dropped']}). Secondary indices OK: {res['secondary_indices_ok']}. "
             f"INCONCLUSIVE cells: {res['INCONCLUSIVE_cells'] or 'none'}.")
    L.append("")
    L.append("| cell | attempted | failed | share |")
    L.append("|---|---|---|---|")
    for k in res["attempted"]:
        fs = res["fail_share"][k]
        L.append(f"| {k} | {res['attempted'][k]} | {res['failed'][k]} | {'' if fs is None else f'{100*fs:.2f}%'} |")
    for blk in ("primary", "secondary"):
        if blk not in res:
            continue
        b = res[blk]
        L += ["", f"## {blk.upper()} (n = {b['n']} paired indices)", "", "| cell | WR | 95% CI |", "|---|---|---|"]
        for k, v in b["wr"].items():
            L.append(f"| {k} | {v['point']:.4f} | [{v['ci'][0]:.4f}, {v['ci'][1]:.4f}] |")
        L += ["", "| contrast | point (pp) | 95% CI (pp) | read |", "|---|---|---|---|"]
        for k, v in b["contrasts"].items():
            L.append(f"| {k} | {pct(v['point'])} | [{pct(v['ci'][0])}, {pct(v['ci'][1])}] | {v['read']} |")
        if "discordant_share" in b:
            L += ["", "Discordant-outcome share (index-level, vs LEARNED): " +
                  ", ".join(f"{k} {100*v:.1f}%" for k, v in b["discordant_share"].items())]
    L += ["", "## SIDE READ — greedy-action change rate along the LEARNED trajectories", "",
          f"Sanity (OFF re-forward ≠ committed action): {res['side_read_sanity_off_mismatches']} (must be 0).", ""]
    ks = [str(k) for k in range(7)] + ["all"]
    L.append("| column:scope | " + " | ".join(f"k={k}" for k in ks) + " |")
    L.append("|---|" + "---|" * len(ks))
    for key, per in res["side_read"].items():
        cells_ = []
        for k in ks:
            if k in per:
                v = per[k]
                cells_.append(f"{100*v['rate']:.1f}% [{100*v['ci'][0]:.1f}, {100*v['ci'][1]:.1f}] (n={v['n_decisions']})")
            else:
                cells_.append("—")
        L.append(f"| {key} | " + " | ".join(cells_) + " |")
    L += ["", "## Descriptive", "", "| cell | mean turns | max turns | mean decisions | draws |", "|---|---|---|---|---|"]
    for k, v in res["descriptive"].items():
        L.append(f"| {k} | {v['mean_turns']:.1f} | {v['max_turns']} | {v['mean_decisions']:.1f} | {v['draws']} |")
    if res["errors"]:
        L += ["", "## Failures", ""]
        for k, v in res["errors"].items():
            for e in v:
                L.append(f"* {k} i={e[0]} {e[1]}: {e[2]}")
    open(path, "w").write("\n".join(L) + "\n")


def compare(a, b, fields, cells=None):
    ra, rb = load(a), load(b)
    fields = fields.split(",")
    keys = sorted(set(ra) & set(rb))
    if cells:
        want = {tuple(c.split(":")) for c in cells.split(",")}
        keys = [k for k in keys if (k[0], k[1]) in want]
    mism = [k for k in keys if any(ra[k].get(f) != rb[k].get(f) for f in fields)]
    out = {"compared": len(keys), "fields": fields, "mismatches": len(mism), "examples": [list(k) for k in mism[:10]]}
    proof = {"n": 0, "off_eq_ref": 0, "on_all_eq_prior": 0, "on_move_eq_prior": 0, "on_all_logits_differ": 0,
             "n_fail_msgs": 0}
    for r in rb.values():
        p = r.get("proof")
        if p:
            for k in ("n", "off_eq_ref", "on_all_eq_prior", "on_move_eq_prior", "on_all_logits_differ"):
                proof[k] += p[k]
            proof["n_fail_msgs"] += len(p["fail"])
    out["proof_in_b"] = proof
    print(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    if sys.argv[1] == "report":
        report(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "compare":
        compare(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5] if len(sys.argv) > 5 else None)
