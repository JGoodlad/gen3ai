"""Every load-bearing number in README.md must appear in an artifact — checked, not trusted.

This project has been bitten by transcription often enough that a README quoting a measurement it
cannot re-derive is a defect. For each claim below the value is recomputed from the committed JSON
and its formatted string must occur literally in README.md.

Run: python verify_readme.py     (exit 1 on any miss)
"""
import json
import os
import sys

H = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.join(os.path.dirname(H), "content_locality")


def j(p):
    return json.load(open(p))


def main():
    # U+2212 MINUS SIGN reads better in prose; normalise before matching.
    md = open(f"{H}/README.md").read().replace("\u2212", "-")
    c9, c3 = j(f"{H}/combined_v2_n9.json"), j(f"{H}/combined_v2_n3.json")
    g9, v9 = j(f"{H}/gen_era_v2_n9.json"), j(f"{H}/v8_era_v2_n9.json")
    o9 = j(f"{V1}/combined_n9.json")
    res = j(f"{H}/resolved_teachers.json")

    claims = []

    def add(name, s):
        claims.append((name, s))

    # headline R
    add("v8 R n=9", f"{c9['v8']['R']:.4f} [{c9['v8']['ci95'][0]:.4f}, "
                    f"{c9['v8']['ci95'][1]:.4f}]")
    add("v8 R n=3", f"{c3['v8']['R']:.4f} [{c3['v8']['ci95'][0]:.4f}, "
                    f"{c3['v8']['ci95'][1]:.4f}]")
    for ref in ("A", "B"):
        for half in ("unfunded", "funded"):
            b = c9[f"ref{ref}"][half]
            add(f"gen {half} R REF-{ref} n=9",
                f"{b['R']:.4f} [{b['R_ci95'][0]:.4f}, {b['R_ci95'][1]:.4f}]")
    add("v1 v8 R", f"{o9['sibling_R']['v8_all']['R']:.4f}")

    # contrasts
    for ref in ("A", "B"):
        for half in ("unfunded", "funded"):
            x = c9[f"ref{ref}"][f"cross_era_v8_minus_gen_{half}_R"]
            add(f"cross-era {half} REF-{ref}",
                f"{x['delta']:+.4f} | [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]")
        w = c9[f"ref{ref}"]["within_gen_funded_minus_unfunded_R"]
        add(f"within-gen REF-{ref}",
            f"{w['delta']:+.4f} | [{w['ci95'][0]:+.4f}, {w['ci95'][1]:+.4f}]")

    # the reference gap and the resolver count
    add("reference gap", f"{g9['_meta']['ref_gap_kl_parent_given_origin']:.4f}")
    add("n_different teachers", f"{res['n_different']}/{len(res['teachers'])} teachers")

    # v8 semistall3, the thin cell
    add("semistall3 untaught",
        f"{v9['primary_A_per_teacher']['semistall3']['kl_untaught']:.4f}")

    # the paired funded-unfunded untaught gap, REF-A
    d = g9["primary_A_groups_refA"]["FUNDED_minus_UNFUNDED_kl_untaught"]
    add("FUNDED-UNFUNDED untaught REF-A",
        f"+{d['delta']:.4f} [+{d['ci95'][0]:.4f}, +{d['ci95'][1]:.4f}]")

    # states + wall clocks
    add("gen states", str(g9["_meta"]["n_states"]))
    add("v8 states", str(v9["_meta"]["n_states"]))

    bad = [(n, s) for n, s in claims if s not in md]
    for n, s in claims:
        print(f"  {'OK ' if s in md else 'MISS'}  {n:34s} {s}")
    if bad:
        print(f"\n  {len(bad)} claim(s) NOT found verbatim in README.md")
        sys.exit(1)
    print(f"\n  PASS — all {len(claims)} checked claims appear verbatim in README.md")


if __name__ == "__main__":
    main()
