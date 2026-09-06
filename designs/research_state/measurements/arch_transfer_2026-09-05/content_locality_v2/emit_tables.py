"""Emit the README's numeric tables straight from the artifacts, so no number is transcribed.

Run: python emit_tables.py            (reads the four v2 artifacts + the two v1 combined files)
"""
import json
import os

H = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(os.path.dirname(H), "content_locality")


def j(p):
    return json.load(open(p))


def ci(x):
    return f"[{x[0]:.4f}, {x[1]:.4f}]"


def main():
    c = {n: j(f"{H}/combined_v2_n{n}.json") for n in (3, 9)}
    o = {n: j(f"{V1}/combined_n{n}.json") for n in (3, 9)}

    print("### Headline — sibling-control R, v2 beside v1\n")
    print("| arm | reference | v1 R (n=9) | **v2 R (n=9)** | v2 R (n=3) |")
    print("|---|---|---|---|---|")
    rows = [("v8 (3 teachers)", None, "v8_all", "v8"),
            ("gen unfunded R5F (8)", "A", "gen_unfunded", ("refA", "unfunded")),
            ("gen funded R5FUND (8)", "A", "gen_funded", ("refA", "funded")),
            ("gen unfunded R5F (8)", "B", "gen_unfunded", ("refB", "unfunded")),
            ("gen funded R5FUND (8)", "B", "gen_funded", ("refB", "funded"))]
    for label, ref, v1key, v2key in rows:
        refname = {None: "parent = origin", "A": "REF-A fold parent",
                   "B": "REF-B true origin"}[ref]
        o9 = o[9]["sibling_R"][v1key]
        if isinstance(v2key, tuple):
            n9 = c[9][v2key[0]][v2key[1]]; n3 = c[3][v2key[0]][v2key[1]]
            g9, g3 = (n9["R"], n9["R_ci95"]), (n3["R"], n3["R_ci95"])
        else:
            n9 = c[9][v2key]; n3 = c[3][v2key]
            g9, g3 = (n9["R"], n9["ci95"]), (n3["R"], n3["ci95"])
        v1s = f"{o9['R']:.4f} {ci(o9['ci95'])}" if ref in (None, "A") else "—"
        print(f"| {label} | {refname} | {v1s} | **{g9[0]:.4f} {ci(g9[1])}** | "
              f"{g3[0]:.4f} {ci(g3[1])} |")

    print("\n### Absolute levels (n=9)\n")
    print("| half | reference | KL on own taught | KL on untaught 8 | raw L |")
    print("|---|---|---|---|---|")
    for ref in ("A", "B"):
        for half in ("unfunded", "funded"):
            b = c[9][f"ref{ref}"][half]
            print(f"| gen {half} | REF-{ref} | {b['kl_taught_mean']:.4f} "
                  f"{ci(b['kl_taught_ci95'])} | {b['kl_untaught_mean']:.4f} "
                  f"{ci(b['kl_untaught_ci95'])} | {b['L']:.4f} {ci(b['L_ci95'])} |")
    v8 = c[9]["v8"]
    print(f"| v8 (all 3) | parent = origin | {v8['kl_own_mean']:.4f} (own, sibling control) | "
          f"— | {v8['L']:.4f} {ci(v8['L_ci95'])} |")

    print("\n### Contrasts (n=9)\n")
    print("| contrast | reference | delta | CI95 | verdict |")
    print("|---|---|---|---|---|")
    for ref in ("A", "B"):
        for half in ("unfunded", "funded"):
            k = f"cross_era_v8_minus_gen_{half}_R"
            x = c[9][f"ref{ref}"][k]
            v = "SIGNIFICANT" if x["separates_from_zero"] else "NOT DETECTED"
            print(f"| v8 − gen {half} (R, unpaired) | REF-{ref} | {x['delta']:+.4f} | "
                  f"{ci(x['ci95'])} | {v} |")
        w = c[9][f"ref{ref}"]["within_gen_funded_minus_unfunded_R"]
        v = "SIGNIFICANT" if w["separates_from_zero"] else "NOT DETECTED"
        print(f"| gen funded − unfunded (R, paired) | REF-{ref} | {w['delta']:+.4f} | "
              f"{ci(w['ci95'])} | {v} |")

    print("\n### Matched-noise floor (n=9)\n")
    print("| era | pair | reference | KL untaught | KL taught | floor L |")
    print("|---|---|---|---|---|---|")
    for tag, f in c[9]["floor"]["gen"].items():
        t = f.get("taught16_mean", f.get("taught_mean"))
        print(f"| gen | `{tag}` | REF-{f.get('reference', 'A')} | {f['untaught_mean']:.4f} | "
              f"{t:.4f} | {t / f['untaught_mean']:.4f} |")
    for tag, f in c[9]["floor"]["v8"].items():
        t = f["taught_mean"]
        print(f"| v8 | `{tag}` | parent = origin | {f['untaught_mean']:.4f} | {t:.4f} | "
              f"{t / f['untaught_mean']:.4f} |")

    print("\n### v8 per-teacher, resolved vs what v1 scored (n=9)\n")
    g9 = j(f"{H}/v8_era_v2_n9.json")
    o9 = j(f"{V1}/v8_era_n9.json")
    print("| teacher | n taught | v1 KL untaught | **v2 KL untaught** | v1 L | **v2 L** |")
    print("|---|---|---|---|---|---|")
    for k in g9["primary_A_per_teacher"]:
        a = o9["primary_A_per_teacher"][k]; b = g9["primary_A_per_teacher"][k]
        print(f"| `{k}` | {b['n_taught']} | {a['kl_untaught']:.4f} | "
              f"**{b['kl_untaught']:.4f}** | {a['L']:.4f} | **{b['L']:.4f}** |")
    ea, eb = o9["primary_A_era"], g9["primary_A_era"]
    print(f"| **pooled** | {eb['n_cells']} cells | {ea['kl_untaught_mean']:.4f} | "
          f"**{eb['kl_untaught_mean']:.4f}** | {ea['L']:.4f} {ci(ea['L_ci95'])} | "
          f"**{eb['L']:.4f} {ci(eb['L_ci95'])}** |")

    print("\n### gen per-teacher untaught KL, v1 (final_model) → v2 (best_model), REF-A, n=9\n")
    gg = j(f"{H}/gen_era_v2_n9.json")
    og = j(f"{V1}/gen_era_n9.json")
    print("| teacher | v1 | **v2** | Δ |")
    print("|---|---|---|---|")
    for k in sorted(gg["primary_A_per_teacher_refA"]):
        a = og["primary_A_per_teacher"][k]["kl_untaught"]
        b = gg["primary_A_per_teacher_refA"][k]["kl_untaught"]
        print(f"| `{k}` | {a:.4f} | **{b:.4f}** | {b - a:+.4f} |")


if __name__ == "__main__":
    main()
