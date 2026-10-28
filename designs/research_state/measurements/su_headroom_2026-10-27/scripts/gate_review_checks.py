"""The S-U GATE's adversarial review (REVIEW_gate.md): checks over the gate's BANKED rows only, READ-ONLY on every input.

Inputs (the gate's state, ``~/gen3ai_archive/su_gate_2026-10-27/``, PREREG ef05d165c, pin 835c6b4b2): ``gate/selected.jsonl``
(the frozen draw, each with its kept true log), ``gate/scored.jsonl`` (S and the self-referee per state),
``gate/scored_default_threads_superseded.jsonl`` (15 states scored under the default thread pool, set aside), and
``decision.json`` (the operator's verdict). No game, search or playout is run.

Checks (all POST HOC unless noted; the registered verdict is the operator's):
1. ``reproduce``: the primary (pooled-U sign accuracy, game-clustered, rng 20261027, B 2,000) re-derived with a clone of
   ``sign_gate.cluster_ratio_boot``; must equal decision.json's digits (the clone's tooth).
2. ``clusters``: the same interval clustered by OUR team and by the OPPONENT's team instead of the game.
3. ``teams``: whose teams Kakuna plays in each panel (the referee's and S's opponent model is P* on those teams).
4. ``leak``: sign accuracy in leak-exposed states (a rewritten past opponent choice, or a seen mon's Speed pinned by
   the true seed, ``speed_flip.speed_pinned`` > 0) against leak-clean ones, per pooled U and home72, each marker alone,
   and the interaction (leaky - clean on U) - (leaky - clean on home72), game-clustered (rng 1, B 2,000).
5. ``robust``: ties counted as 1/2; every excluded state's expected countable deviations counted WRONG (a bound).
6. ``kinds``: accuracy by the deviation's kind (S's pick / pi's top a move or a switch).
7. ``threads``: the 15 superseded states against their final rows (picks, Q, referee strings).
8. ``attribution``: what the leak block's ``attribution_miss`` counts (opponent move commands absent from the true set).
9. ``live_power``: the z a ONE-game-per-action live check of the gate's deviations would reach IF Kakuna behaved
   exactly like the self-referee (per-seed split rate and net sign from the referee's common-dice outcome strings).

    python -I gate_review_checks.py   (CPU, < 1 min, < 1 GB; writes gate_review_results.json beside this folder)
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

import numpy as np

STATE = Path.home() / "gen3ai_archive" / "su_gate_2026-10-27"
GATE = STATE / "gate"
OUT = Path(__file__).resolve().parent.parent / "gate_review_results.json"
U = ["heldout72_hpfix", "forum72", "p72"]
HOME = "home72"
B, SEED = 2000, 20261027
OUTCOME = {"+": 1.0, "0": 0.5, "-": 0.0}


def jl(p: Path):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def boot(groups, panels, *, seed=SEED, b=B):
    """A clone of ``sign_gate.cluster_ratio_boot`` (same draw order)."""
    rng = np.random.default_rng(seed)
    s, n = {}, {}
    for p in panels:
        ks = sorted(groups.get(p, {}))
        arrs = [np.asarray([v for v in groups[p][k] if not np.isnan(v)], dtype=float) for k in ks]
        s[p] = np.array([a.sum() for a in arrs])
        n[p] = np.array([len(a) for a in arrs])

    def est(p, i=None):
        ss, nn = (s[p], n[p]) if i is None else (s[p][i], n[p][i])
        return float(ss.sum() / max(nn.sum(), 1))

    point = float(np.mean([est(p) for p in panels]))
    bs = [float(np.mean([est(p, rng.integers(0, len(s[p]), len(s[p]))) for p in panels])) for _ in range(b)]
    return {"est": round(point, 4), "lo": round(float(np.percentile(bs, 2.5)), 4),
            "hi": round(float(np.percentile(bs, 97.5)), 4), "n": int(sum(int(n[p].sum()) for p in panels)),
            "clusters": {p: int(len(s[p])) for p in panels}, "_raw": (point, float(np.percentile(bs, 2.5)),
                                                                      float(np.percentile(bs, 97.5)))}


def main() -> int:
    sel = {r["key"]: r for r in jl(GATE / "selected.jsonl")}
    rows = jl(GATE / "scored.jsonl")
    res = {}

    def me(r):
        return 0 if sel[r["key"]]["side"] == "p1" else 1

    def our(r):
        return sel[r["key"]]["log"]["teams"][me(r)]

    def opp(r):
        return sel[r["key"]]["log"]["teams"][1 - me(r)]

    def sign(r):
        s = r["search"]
        return float(r["referee"]["sign"]) if s.get("searched") and s.get("deviation") else float("nan")

    def rewrite(r):
        lk = r["search"].get("leak") or {}
        return bool(lk.get("rewritten_moves") or lk.get("rewritten_switches"))

    def pinned(r):
        return r["speed_flip"]["speed_pinned"] > 0

    def leaky(r):
        return rewrite(r) or pinned(r)

    def groups(cluster, keep=lambda r: True, value=sign):
        g = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in rows:
            g[r["panel"]][cluster(r)].append(value(r) if keep(r) else float("nan"))
        return g

    def clean(d):
        return {k: v for k, v in d.items() if k != "_raw"}

    game = lambda r: r["deal"]  # noqa: E731

    # 1. reproduce
    dec = json.loads((STATE / "decision.json").read_text())
    prim = boot(groups(game), U)
    p = dec["primary"]
    res["reproduce"] = {"mine": clean(prim), "decision": [p["est"], p["lo"], p["hi"], p["n"]],
                        "equal": prim["_raw"] == (p["est"], p["lo"], p["hi"]) and prim["n"] == p["n"]}

    # 2. clusters
    res["clusters"] = {"game_U": clean(prim), "our_team_U": clean(boot(groups(our), U)),
                       "opp_team_U": clean(boot(groups(opp), U)), "game_home72": clean(boot(groups(game), [HOME])),
                       "our_team_home72": clean(boot(groups(our), [HOME]))}

    # 3. teams
    t = {}
    for pn in U + [HOME]:
        pr = [r for r in sel.values() if r["panel"] == pn]
        o = {r["log"]["teams"][0 if r["side"] == "p1" else 1] for r in pr}
        k = {r["log"]["teams"][1 if r["side"] == "p1" else 0] for r in pr}
        t[pn] = {"our_teams": len(o), "opp_teams": len(k), "opp_also_ours_in_panel": len(o & k)}
    res["teams"] = t

    # 4. leak
    leak = {"share_leaky_of_searched": {pn: round(float(np.mean([leaky(r) for r in rows if r["panel"] == pn
                                                                  and r["search"].get("searched")])), 3)
                                        for pn in U + [HOME]}}
    for name, f in [("leaky", leaky), ("clean", lambda r: not leaky(r)), ("rewrite", rewrite),
                    ("no_rewrite", lambda r: not rewrite(r)), ("pinned", pinned), ("not_pinned", lambda r: not pinned(r)),
                    ("rewrite_not_pinned", lambda r: rewrite(r) and not pinned(r)),
                    ("pinned_no_rewrite", lambda r: pinned(r) and not rewrite(r))]:
        leak[name] = {"U": clean(boot(groups(game, f), U)), "home72": clean(boot(groups(game, f), [HOME]))}
    data = {}
    for pn in U + [HOME]:
        ga, gb = groups(game, leaky)[pn], groups(game, lambda r: not leaky(r))[pn]
        ks = sorted(ga)
        a = [np.array([v for v in ga[k] if v == v]) for k in ks]
        b = [np.array([v for v in gb[k] if v == v]) for k in ks]
        data[pn] = tuple(np.array(x) for x in ([v.sum() for v in a], [len(v) for v in a],
                                                [v.sum() for v in b], [len(v) for v in b]))

    def diff(pn, i=None):
        sa, na, sb, nb = data[pn] if i is None else tuple(x[i] for x in data[pn])
        return sa.sum() / na.sum() - sb.sum() / nb.sum()

    rng = np.random.default_rng(1)
    bu, bh, bi = [], [], []
    for _ in range(B):
        vu = float(np.mean([diff(pn, rng.integers(0, len(data[pn][0]), len(data[pn][0]))) for pn in U]))
        vh = diff(HOME, rng.integers(0, len(data[HOME][0]), len(data[HOME][0])))
        bu.append(vu), bh.append(vh), bi.append(vu - vh)
    pu, ph = float(np.mean([diff(pn) for pn in U])), diff(HOME)
    q = lambda x: [round(float(v), 3) for v in np.percentile(x, [2.5, 97.5])]  # noqa: E731
    leak["leaky_minus_clean"] = {"U": [round(pu, 3)] + q(bu), "home72": [round(ph, 3)] + q(bh),
                                 "interaction": [round(pu - ph, 3)] + q(bi)}
    ctx = {}
    for pn in U + [HOME]:
        for nm, f in [("leaky", leaky), ("clean", lambda r: not leaky(r))]:
            qs = [r for r in rows if r["panel"] == pn and r["search"].get("searched") and f(r)]
            dv = [r for r in qs if r["search"]["deviation"]]
            ctx[f"{pn}/{nm}"] = {"states": len(qs), "turn": round(float(np.mean([r["turn"] for r in qs])), 1),
                                 "dev_rate": round(len(dv) / len(qs), 3),
                                 "root_v": round(float(np.mean([r["search"]["root_v"] for r in qs])), 3)}
    leak["context"] = ctx
    res["leak"] = leak

    # 5. robust
    rob = {}
    for pn in U + [HOME]:
        pr = [r for r in rows if r["panel"] == pn]
        dv = [r for r in pr if r.get("referee")]
        ct = [r for r in dv if r["referee"]["sign"] == r["referee"]["sign"]]
        right = sum(r["referee"]["sign"] for r in ct)
        srch = [r for r in pr if r["search"].get("searched")]
        excl = sum(1 for r in pr if r["search"].get("why") == "excluded")
        add = excl * (len(dv) / len(srch)) * (len(ct) / max(len(dv), 1))
        rob[pn] = {"deviations": len(dv), "countable": len(ct), "ties": len(dv) - len(ct),
                   "acc": round(right / len(ct), 4), "ties_as_half": round((right + 0.5 * (len(dv) - len(ct))) / len(dv), 4),
                   "excluded": excl, "excluded_expected_countable": round(add, 1),
                   "acc_if_all_excluded_wrong": round(right / (len(ct) + add), 4),
                   "excluded_turn": round(float(np.mean([r["turn"] for r in pr if r["search"].get("why") == "excluded"])), 1),
                   "all_turn": round(float(np.mean([r["turn"] for r in pr])), 1)}
    for k in ("ties_as_half", "acc_if_all_excluded_wrong"):
        rob[f"pooled_U_{k}"] = round(float(np.mean([rob[pn][k] for pn in U])), 4)
    res["robust"] = rob

    # 6. kinds
    def kind(r, a):
        return "sw" if str(sel[r["key"]]["tokens"].get(str(a), "")).startswith("switch") else "mv"

    kd = {}
    for pn in U + [HOME]:
        c = collections.defaultdict(list)
        for r in rows:
            if r["panel"] == pn and r.get("referee") and r["referee"]["sign"] == r["referee"]["sign"]:
                c[f"{kind(r, r['search']['played'])}>{kind(r, r['pi_argmax'])}"].append(r["referee"]["sign"])
        kd[pn] = {k: [len(v), round(float(np.mean(v)), 3)] for k, v in sorted(c.items())}
    res["kinds_pick_over_top"] = kd

    # 7. threads
    fin = {r["key"]: r for r in rows}
    sup = jl(GATE / "scored_default_threads_superseded.jsonl")
    dq = [max(abs(r["search"]["q"][a] - fin[r["key"]]["search"]["q"][a]) for a in r["search"]["q"])
          for r in sup if r["search"].get("q")]
    res["threads"] = {"states": len(sup),
                      "same_pick": sum(r["search"].get("played") == fin[r["key"]]["search"].get("played") for r in sup),
                      "max_abs_dq": max(dq) if dq else None,
                      "referee_rows": sum(1 for r in sup if r.get("referee")),
                      "same_referee_out": sum(1 for r in sup if r.get("referee")
                                              and r["referee"]["out"] == (fin[r["key"]].get("referee") or {}).get("out"))}

    # 8. attribution
    toid = lambda s: re.sub(r"[^a-z0-9]", "", s.lower())  # noqa: E731
    att = collections.defaultdict(collections.Counter)
    for r in sel.values():
        oi = 1 if r["side"] == "p1" else 0
        ok = "p2" if r["side"] == "p1" else "p1"
        mv = set()
        for e in r["log"]["teams"][oi].split("]"):
            mv |= {toid(m) for m in e.split("|")[4].split(",")}
        for c in r["log"]["cmds"][: r["at"]]:
            if not c.startswith("CHOOSE "):
                continue
            _, s_, tk = c.split(" ", 2)
            if s_ == ok and tk.startswith("move ") and not tk[5:].strip().isdigit() and toid(tk[5:]) not in mv:
                att[r["panel"]][toid(tk[5:])] += 1
    res["attribution_miss_moves"] = {pn: dict(c) for pn, c in att.items()}

    # 9. live power
    lp = {}
    for name, panels in (("U", U), ("home72", [HOME])):
        dv = [r for r in rows if r["panel"] in panels and r.get("referee")]
        split, net = [], []
        for r in dv:
            o = r["referee"]["out"]
            d = np.array([OUTCOME[x] - OUTCOME[y] for x, y in zip(o[str(r["search"]["played"])], o[str(r["pi_argmax"])])])
            split.append(float(np.mean(d != 0)))
            net.append(float(np.mean(np.sign(d))))
        ps = float(np.mean(split))
        acc = 0.5 + float(np.mean(net)) / (2 * ps)
        ns = len(dv) * ps
        se = (acc * (1 - acc) / ns) ** 0.5
        lp[name] = {"deviations": len(dv), "mean_dT": round(float(np.mean(
            [r["referee"]["T_pick"] - r["referee"]["T_top"] for r in dv])), 4), "per_seed_split": round(ps, 3),
            "live_acc_given_split": round(acc, 3), "expected_splits": round(ns), "z": round((acc - 0.5) / se, 2)}
    res["live_power"] = lp

    OUT.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"reproduce": res["reproduce"]["equal"], "clean_U": res["leak"]["clean"]["U"],
                      "interaction": res["leak"]["leaky_minus_clean"]["interaction"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
