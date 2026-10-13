"""Render `by_reveal_count.json` and `belief_dose.json` as the markdown tables of this folder's README (`tables.md`).

    python -I render_tables.py   →  tables.md
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
KG = "G = O - A' (LEAF-K8)"
KH = "headroom O - C'"
KS = "seen moves CT' - C'"
KL = "search over critic, honest A' (LEAF-K8 - CRITIC)"
KT = "search over critic, honest A' (TRUNC-4-K8 - CRITIC)"
KPC = "C' LEAF-K8"
KPT = "CT' LEAF-K8"
R = json.loads((HERE / "by_reveal_count.json").read_text())
D = json.loads((HERE / "belief_dose.json").read_text())


def f(s, sign=True, ci=True, d=3):
    if s is None or s.get("mean") is None:
        return "n/a"
    fmt = f"{{:+.{d}f}}" if sign else f"{{:.{d}f}}"
    m = fmt.format(s["mean"])
    if ci and s.get("ci") and s["ci"][0] is not None:
        return f"{m} [{fmt.format(s['ci'][0])}, {fmt.format(s['ci'][1])}]"
    return m


def row(label, c):
    if "suppressed" in c or "imp" not in c:
        return f"| {label} | {c['n']} | (n < 30, suppressed) |" + " |" * 9
    k = c["contrast"]
    v = c["voi"]
    sh = c["share_of_headroom_over_C"]["seen_moves"]
    shs = "n/a" if sh["mean"] is None else f"{sh['mean']:.2f} [{sh['ci'][0]:.2f}, {sh['ci'][1]:.2f}]"
    return (f"| {label} | {c['n']} | {f(c['imp']['O|LEAF-K8'], ci=False)} | {f(c['imp']['A|LEAF-K8'], ci=False)} | "
            f"{f(k[KG])} | {f(k[KH], ci=False)} | "
            f"{f(k[KS], ci=False)} | {shs} | "
            f"{f(v['change_rate c (oracle pick != honest pick)'], sign=False, ci=False, d=2)} | "
            f"{f(v['stakes s (true-value gain per changed decision)'], sign=False, ci=False, d=3)} | "
            f"{f(k[KL], ci=False)} / "
            f"{f(k[KT], ci=False)} |")


HEAD = ("| stage | n | oracle IMP | honest IMP (A') | **G = O - A'** [95 %] | headroom O - C' | seen moves CT' - C' | "
        "seen share of headroom | change rate c | stakes s | honest search over critic (LEAF / TRUNC-4) |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|")


KORDER = ["k1-2", "k3", "k4", "k5", "k6"]


def ordered(axis, cells):
    e = R["validity"]["edges"]
    if axis == "turn":
        keys = e["turn_labels"]
    elif axis == "material":
        keys = e["material_labels"]
    elif axis in ("k", "k_single", "all"):
        keys = list(cells)
    else:
        sub = e["thalf_labels"] if axis == "k_x_turn_half" else (e["turn_labels"] if axis == "k_x_turn4" else e["material_labels"])
        keys = [f"{k}|{b}" for k in KORDER for b in sub]
    return {k: cells[k] for k in keys if k in cells}


def table(scope, axis, title):
    cells = ordered(axis, R["scopes"][scope][axis])
    return [f"**{title} ({scope})**", "", HEAD] + [row(k, c) for k, c in cells.items()] + [""]


def trend(scope):
    t = R["trend"][scope]
    out = [f"**{scope}: G on the stage axes (OLS, per +1 SD, battle-clustered 95 %)**; corr(k, turn) {t['corr_k_turn']}, "
           f"corr(k, material) {t['corr_k_material']}, corr(turn, material) {t['corr_turn_material']}", "",
           "| regression of g(s) = oracle - honest on | slope(s) per SD [95 %] | R^2 |", "|---|---|---|"]
    for key in ("on k (per +1 SD)", "on turn (per +1 SD)", "on material (per +1 SD)", "on k + turn", "on k + material",
                "on k + turn + material"):
        x = t["G"][key]
        parts = " ; ".join(f"{c:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]" for c, ci in list(zip(x["coef"], x["ci"]))[1:])
        out.append(f"| {key} | {parts} | {x['r2']:.3f} |")
    for key in ("G low(k<=3) - high(k>=5)", "oracle edge low - high", "honest A' low - high"):
        x = t[key]
        out.append(f"| {key} | {x['mean']:+.4f} [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}] | |")
    return out + [""]


def mechanism(scope):
    m = R["trend"][scope]["mechanism_contrasts"]
    out = [f"**Mechanism, G = c x s ({scope}): the EARLY group minus the LATE group, paired, 95 % CI**", "",
           "| contrast (first - second) | n | G | oracle edge IMP_O | honest IMP_A' | change rate c | stakes s | regret of honest pick | action-value spread |",
           "|---|---|---|---|---|---|---|---|---|"]
    for name, x in m.items():
        def g(key):
            v = x[key]
            return f"{v['mean']:+.3f} [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]"
        out.append(f"| {name} | {x['n'][0]} / {x['n'][1]} | {g('G (first - second)')} | {g('oracle edge IMP_O')} | {g(chr(104) + 'onest IMP_A' + chr(39))} | "
                   f"{g('change rate c')} | {g('stakes s (gain per changed decision)')} | {g('regret of the honest pick')} | {g('action-value spread')} |")
    return out + [""]


def dose(scope, cells):
    out = [f"**Belief-quality dose-response ({scope}): IMP of the K-world mixture search against p, the weight on the TRUE world**", "",
           "| stage | n | scorer | p = 0 | 0.1 | 0.2 | 0.35 | 0.5 | 0.75 | 1.0 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for lab in cells:
        c = D["scopes"][scope].get(lab)
        if c is None:
            continue
        for nm, short in (("M8|LEAF-K8", "leaf K8 (M8)"), ("M8|TRUNC-4-K8", "4-turn leaf (M8)"), ("M8|CRITIC", "one-ply critic (M8)"),
                          ("M5|LEAF-K8", "leaf, K=5 once (M5)"), ("M5W|LEAF-K8", "same, WRONG world (M5W)")):
            x = c[nm]
            out.append(f"| {lab} | {c['n']} | {short} | " + " | ".join(f"{v:+.3f}" for v in x["imp"]) + " |")
    out.append("")
    out += ["| stage | p-half | content effect M5 - M5W at p = 0.1 / 0.35 / 0.75 [95 %] (leaf) | p* leaf beats critic (same p): point / lower-95 | "
            "p* leaf beats the policy: point / lower-95 | p-equivalent of C' / CT' (value units) | action-value spread, true world / honest mean |",
            "|---|---|---|---|---|---|---|"]
    for lab in cells:
        c = D["scopes"][scope].get(lab)
        if c is None:
            continue
        m = c["M5-M5W|LEAF-K8"]
        eff = " / ".join(f"{m['diff'][i]:+.3f} [{m['ci'][i][0]:+.3f}, {m['ci'][i][1]:+.3f}]" for i in (1, 3, 5))
        b1 = c["break_even"]["LEAF beats the critic search at the same p"]
        b0 = c["break_even"]["LEAF beats the policy (IMP > edge)"]
        pe = c["p_equivalent_of_world_kinds"]
        def pf(b):
            return f"{b['p_star_point']} / {b['p_star_lower95']}"
        def pq(k):
            return f"{pe[k]['p_eq']:.3f} {pe[k]['interval']}"
        sc = c["signal_scale_across_actions"]["LEAF-K8"]
        scs = f"{sc['true world (8 seeds)']['mean']:.3f} / {sc[chr(104) + 'onest worlds' + chr(39) + ' mean (8 worlds)']['mean']:.3f} (x{sc['ratio of means']:.2f})"
        ph = c["p_half_leaf"]
        out.append(f"| {lab} | {ph['p_eq']:.2f} {ph['interval']} | {eff} | {pf(b1)} | {pf(b0)} | {pq(KPC)} / {pq(KPT)} | {scs} |")
    return out + [""]


def main():
    L = ["# Tables (generated by `render_tables.py` from `by_reveal_count.json` and `belief_dose.json`)", "",
         "All IMP are on the pilot's win scale, versus the policy's own pick, judged on the 24 truth seeds no scorer read. "
         "POST-HOC and DESCRIPTIVE: read after the original study's data existed (README, 'What this read may and may not do').", "",
         f"Validity: k agrees with the protocol-derived count on every state (mismatches {R['validity']['k_mismatch_vs_root_facts']}); "
         f"the common-state means reproduce the pilot's `result_moves.json` to {R['validity']['max_abs_diff_vs_pilot_result_moves']}; "
         f"bucket edges turn {R['validity']['edges']['turn_quartile_edges']}, material {R['validity']['edges']['material_quartile_edges']}.", ""]
    for scope in ("pooled", "end2", "prod"):
        L += table(scope, "k", "By reveal count k")
    for scope in ("pooled", "end2", "prod"):
        L += table(scope, "k_single", "By single k")
    L += table("pooled", "turn", "By turn bucket") + table("pooled", "material", "By material bucket (unfainted mons, ours + theirs)")
    L += table("end2", "turn", "By turn bucket") + table("prod", "turn", "By turn bucket")
    L += table("pooled", "k_x_turn_half", "Joint k x turn half") + table("pooled", "k_x_turn4", "Joint k x turn bucket")
    L += table("pooled", "k_x_material", "Joint k x material bucket")
    for scope in ("pooled", "end2", "prod"):
        L += trend(scope) + mechanism(scope)
    cells = ["all", "k1-2", "k3", "k4", "k5", "k6", "k1-3", "k4-6"] + R["validity"]["edges"]["turn_labels"] + R["validity"]["edges"]["material_labels"]
    for scope in ("pooled", "end2", "prod"):
        L += dose(scope, cells)
    (HERE / "tables.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
