"""Render `result_t1.json` (written by `unblind_t1.py`) as `result_t1.md`: the verdicts and the tables, no new number.

    python -I render_t1.py [result.json] [result.md]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


BIN_ORDER = {"k": ["1-2", "3", "4", "5", "6"], "turn": ["<=7", "8-14", "15-21", ">=22"],
             "material": ["12", "10-11", "8-9", "<=7"]}


def ordered(ax, d):
    """A stage axis' buckets in their natural order (the JSON sorts keys), anything else after."""
    want = BIN_ORDER.get(ax, [])
    return [(b, d[b]) for b in want if b in d] + [(b, v) for b, v in d.items() if b not in want]


def f(s, k=3):
    if s is None:
        return "—"
    if isinstance(s, dict):
        m, ci = s.get("mean"), s.get("ci", [None, None])
        if m is None:
            return "—"
        lo, hi = ci if ci else (None, None)
        return (f"{m:+.{k}f}" + (f" [{lo:+.{k}f}, {hi:+.{k}f}]" if lo is not None and hi is not None else ""))
    if isinstance(s, float):
        return f"{s:+.{k}f}"
    return str(s)


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "result_t1.json"
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / "result_t1.md"
    r = json.loads(src.read_text())
    out = ["# Tier 1 — the result (rendered from `result_t1.json`)", ""]
    mv = r["R-MOVES"]
    out += ["## R-MOVES (§12.3; registered 600 per checkpoint, pooled over end2 + prod, LEAF-K8)", "",
            f"**Verdict: {mv['verdict']}**", "", f"G = IMP_O − IMP_F: {f(mv['G'])} (n {mv['G'].get('n')})", ""]
    if "B" in mv:
        out += ["| condition | holds | read |", "|---|---|---|"]
        for k in ("B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"):
            if k in mv["B"]:
                b = mv["B"][k]
                rest = {x: y for x, y in b.items() if x != "holds"}

                def cell(y):
                    if isinstance(y, dict) and "holds" in y:          # a per-checkpoint block (B8)
                        return f"holds {y['holds']} (" + ", ".join(f"{q} {f(w)}" for q, w in y.items()
                                                                     if q != "holds") + ")"
                    return f(y) if isinstance(y, dict) else str(y)
                out.append(f"| {k} | {b.get('holds')} | " + "; ".join(f"{x} {cell(y)}" for x, y in rest.items())[:600]
                           + " |")
        out += ["", f"R1 {mv['R'].get('R1')} · R2 {mv['R'].get('R2')} · R3 {mv['R'].get('R3')}; "
                    f"NON-ADDITIVE {mv.get('NON-ADDITIVE', {}).get('flag')}", ""]
    out += ["### Shares (pooled)", "", "| class | s (sufficiency) | n (necessity) | σ = IMP_F+X − IMP_F | ν = IMP_O − IMP_O−X |",
            "|---|---|---|---|---|"]
    for X, d in mv.get("shares", {}).items():
        out.append(f"| {X} | {f(d['s'])} | {f(d['n'])} | {f(d['sigma'])} | {f(d['nu'])} |")
    out += ["", "Per checkpoint:", ""]
    for cp, d in r["per_cp_shares"].items():
        out.append(f"- {cp}: G {f(r['per_cp_G'][cp])}; " + "; ".join(
            f"s_{X} {f(v['s'])}, n_{X} {f(v['n'])}" for X, v in d.items()))
    u = r["R-RIVALS"]["U"]
    out += ["", f"## R-RIVALS U: **{u['verdict']}** (s_U {f(u['s_U'])}, n_U {f(u['n_U'])})", ""]
    rc = r["R-RECOVER"]
    out += [f"## R-RECOVER: **{rc['verdict']}**; PRIOR-MISMATCH {rc['PRIOR-MISMATCH']}; HONEST-SHARPENING-HELPS "
            f"{rc['HONEST-SHARPENING-HELPS']} (MAP100 − F {f(rc['MAP100_minus_F'])}); p½ {rc['p_half']}", "",
            "| arm | r = (IMP_X − IMP_F) / G |", "|---|---|"]
    out += [f"| {X} | {f(v)} |" for X, v in rc["r"].items()]
    ev = r["R-EVAL"]
    out += ["", f"## R-EVAL: **{ev['verdict_bound']}**; {ev['verdict_not_binding']}; INFORMATION-BOUND "
                f"{ev['INFORMATION-BOUND']['holds']} (g_HE {f(ev['INFORMATION-BOUND']['g_HE'])}, g_CRIT "
                f"{f(ev['INFORMATION-BOUND']['g_CRIT'])}, ratio {f(ev['INFORMATION-BOUND']['ratio'], 2)})", "",
            "| worlds | HE-TRUNC-4-K8 − TRUNC-4-K8 | HE-1PLY − CRITIC |", "|---|---|---|"]
    for W, d in ev["worlds"].items():
        out.append(f"| {W} | {f(d['HE-TRUNC-4-K8_minus_TRUNC-4-K8'])} | {f(d['HE-1PLY_minus_CRITIC'])} |")
    out += ["", "## The stage rules (§12.13.4; all Tier-1S states)", ""]
    for ax, d in r["R-STAGE-VOI"].items():
        out.append(f"- **R-STAGE-VOI {ax}: {', '.join(d['verdict'])}**; slope per SD {f(d['slope_per_sd'])}; G by bucket: "
                   + "; ".join(f"{b} {f(v)}" for b, v in ordered(ax, d["G"])))
    out.append("- **R-STAGE-AXIS**: " + "; ".join(f"{ax} {d['verdict']} ({f(d['partial_slope'])})"
                                                 for ax, d in r["R-STAGE-AXIS"].items()))
    for ax, d in r["R-STAGE-MECH"].items():
        out.append(f"- **R-STAGE-MECH {ax}: {d['verdict']}**; Δc {f(d['delta_c'])}, Δs {f(d['delta_s'])}; split "
                   f"Δc·s̄ {d['split']['delta_c_times_sbar']:+.4f}, c̄·Δs {d['split']['cbar_times_delta_s']:+.4f}")
    out += ["", "### R-STAGE-CLASS (M vs U) and the two-point Shapley credit", "",
            "| axis | bucket | n | G | σ_M | σ_U | ν_M | ν_U | φ_M | φ_U | verdict |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for ax, d in r["R-STAGE-CLASS"].items():
        for b, v in ordered(ax, d):
            if "n" not in v:
                out.append(f"| {ax} | {b} | | | | | | | | | {v['verdict']} |")
                continue
            out.append(f"| {ax} | {b} | {v['n']} | {f(v['G'])} | {f(v['sigma_M'])} | {f(v['sigma_U'])} | {f(v['nu_M'])} | "
                       f"{f(v['nu_U'])} | {v['phi_M']:+.3f} | {v['phi_U']:+.3f} | {v['verdict']} |")
    ss = r["R-STAGE-SEARCH"]
    out += ["", f"### R-STAGE-SEARCH: FRONTIER = {ss['FRONTIER'] or 'EMPTY'}", "",
            "| axis | bucket | n pooled | TRUNC-4-K8 − CRITIC (95 %) | TRUNC-4-K8 | end2 / prod hold | verdict |",
            "|---|---|---|---|---|---|---|"]
    for ax, d in ss["buckets"].items():
        for b, v in ordered(ax, d):
            pc = v["per_cp"]
            out.append(f"| {ax} | {b} | {v['pooled']['n']} | {f(v['pooled']['TRUNC4_minus_CRITIC'])} | "
                       f"{f(v['pooled']['TRUNC4'])} | {pc['end2']['holds']} / {pc['prod']['holds']} | {v['verdict']} |")
    dz = r["R-DOSE"]
    a = dz["all"]
    out += ["", "## R-DOSE (§6.9; M8 = the TRUE world at weight p, F's worlds at 1 − p; LEAF-K8)", "",
            f"Pooled: P-STAR {a['P-STAR']}; TODAY (FN) {a['TODAY']['p_eq']} (upper95 {a['TODAY']['upper95']}); "
            f"{a['REACH']}; SHAPE {a['SHAPE']['verdict']} (Δ² {f(a['SHAPE']['delta2'])}); p-half {a['p_half']}; "
            f"CONTENT-MATTERS {a['CONTENT-MATTERS']['holds']}; OVERCONFIDENCE-COSTS {a['OVERCONFIDENCE-COSTS']['holds']}; "
            f"CRITIC-INFO-LIMITED {a['CRITIC-INFO-LIMITED']['holds']}", "",
            "| p | IMP leaf | IMP critic | leaf − critic (same p) |", "|---|---|---|---|"]
    for p, v in a["curve"].items():
        out.append(f"| {p} | {f(v['leaf'])} | {f(v['crit'])} | {f(v['leaf_minus_crit'])} |")
    out += ["", "| axis | bucket | n | P-STAR (policy / critic same p / critic today) | TODAY [upper95] | REACH | SHAPE | Π |",
            "|---|---|---|---|---|---|---|---|"]
    for ax, d in dz["axes"].items():
        for b, v in ordered(ax, d["cells"]):
            if "P-STAR" not in v:
                continue
            ps = v["P-STAR"]
            out.append(f"| {ax} | {b} | {v['n']} | {ps['vs_policy']} / {ps['vs_critic_same_p']} / {ps['vs_critic_today']} | "
                       f"{v['TODAY']['p_eq']} [{v['TODAY']['upper95']}] | {v['REACH']} | {v['SHAPE']['verdict']} | "
                       f"{f(v.get('PI'))} |")
        out.append(f"| {ax} | DETERMINIZATION-FIRST | {d['DETERMINIZATION-FIRST'] or 'none'} | | | | | |")
    cd = r.get("CLASS-DOSE (Amendment 3)")
    if cd:
        out += ["", "## Amendment 3: per-class dose curves (F+X worlds at weight p, F worlds at 1 − p)", ""]
        for X, v in cd["all"].items():
            if "P-STAR" in v:
                out.append(f"- **{X}** pooled: P-STAR {v['P-STAR']}; SHAPE {v['SHAPE']['verdict']}; p-half {v['p_half']}; "
                           f"TODAY proxy {f(v['TODAY_proxy']['p_proxy'], 3)} (hit F {f(v['TODAY_proxy']['hit_F'])}, "
                           f"FN {f(v['TODAY_proxy']['hit_FN'])}); FN joint on this curve {v['FN_joint_p_eq_on_this_curve']['p_eq']}")
        out += ["", "| axis | bucket | M P-STAR (pol / crit) | U P-STAR (pol / crit) | M p-proxy | U p-proxy | CLASS-FIRST (from proxy; from 0) |",
                "|---|---|---|---|---|---|---|"]
        for ax, d in cd["axes"].items():
            for b, v in ordered(ax, d):
                m_, u_ = v.get("M", {}), v.get("U", {})
                cf = v.get("CLASS-FIRST", {})
                out.append(f"| {ax} | {b} | {m_.get('P-STAR', {}).get('vs_policy')} / {m_.get('P-STAR', {}).get('vs_critic_same_p')} | "
                           f"{u_.get('P-STAR', {}).get('vs_policy')} / {u_.get('P-STAR', {}).get('vs_critic_same_p')} | "
                           f"{f(m_.get('TODAY_proxy', {}).get('p_proxy'))} | {f(u_.get('TODAY_proxy', {}).get('p_proxy'))} | "
                           f"{cf.get('from_today_proxy', {}).get('verdict', '—')}; {cf.get('from_zero', {}).get('verdict', '—')} |")
    out += ["", "## Validity, exclusions, size rules", "", "```", json.dumps(r["validity"], indent=1)[:4000], "```", "",
            f"Size rules: {json.dumps(r['size_rules'])}", "", f"Natural mix: {json.dumps(r['natural_mix'])}", "",
            f"Root-choice sensitivity: {json.dumps(r['root_choice_sensitivity'])}", "",
            f"CONDITION (§10.4, partial): {r['CONDITION']['verdict']} {json.dumps(r['CONDITION']['conditions'])}"]
    dst.write_text("\n".join(out) + "\n")
    print("wrote", dst)


if __name__ == "__main__":
    sys.exit(main())
