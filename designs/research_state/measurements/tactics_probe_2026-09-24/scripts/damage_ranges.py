"""Measure the damage ranges that set the thresholds, in the real (Rust) simulator, from exact-HP lines.

  explosion: STD Gengar (Timid, 0 Atk EVs) Explosion -> the family-D Tyranitar (184 HP/104 Atk/220 Spe Adamant)
  doubleedge: the family-B Tyranitar (Adamant 112 Atk) Double-Edge -> the family-B Blissey (252 Def Modest), + recoil

    python damage_ranges.py --n 200 --out <dir>
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tlib  # noqa: E402
import teams as T  # noqa: E402


def exact_hp(lines, ident):
    """Every '|-damage|<ident>|a/b...' exact value in order."""
    out = []
    for ln in lines:
        p = ln.split("|")
        if len(p) > 3 and p[1] == "-damage" and p[2] == ident:
            m = re.match(r"(\d+)/(\d+)", p[3])
            if m:
                out.append((int(m.group(1)), int(m.group(2)), "|".join(p[4:])))
            elif p[3].startswith("0 fnt"):
                out.append((0, None, "|".join(p[4:])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ours_D, opp_D = tlib.pack(T.team_D_ours(lead="gengar")), tlib.pack(T.team_D_opp())
    ours_B, opp_B = tlib.pack(T.team_B_ours()), tlib.pack(T.team_B_opp())
    res = {"explosion_on_ttar": [], "de_on_blissey": [], "de_recoil": []}
    for i in range(a.n):
        seed = tlib.mint_seed(f"damage_ranges:{i}")
        r = tlib.run_scripted(ours_D, opp_D, ["move explosion", "switch Swampert"], ["move dragondance"], seed)
        d = exact_hp(tlib.side_lines(r["sink"], "p2"), "p2a: Tyranitar")
        crit = any(ln.startswith("|-crit|") for ln in tlib.turn_block(tlib.side_lines(r["sink"], "p2"), 1))
        if d:
            res["explosion_on_ttar"].append({"dmg": 387 - d[0][0], "crit": crit})
        r = tlib.run_scripted(ours_B, opp_B, ["move doubleedge"], ["move softboiled"], seed)
        lines = tlib.side_lines(r["sink"], "p1")
        crit = any(ln.startswith("|-crit|") for ln in tlib.turn_block(lines, 1))
        rec = exact_hp(lines, "p1a: Tyranitar")
        bl = exact_hp(tlib.side_lines(r["sink"], "p2"), "p2a: Blissey")
        if bl and rec:
            res["de_on_blissey"].append({"dmg": bl[0][1] - bl[0][0], "crit": crit})
            res["de_recoil"].append({"recoil": 387 - rec[0][0], "crit": crit})
    summ = {}
    for k, rows in res.items():
        key = "recoil" if k == "de_recoil" else "dmg"
        nc = sorted(r[key] for r in rows if not r["crit"])
        c = sorted(r[key] for r in rows if r["crit"])
        summ[k] = {"n": len(rows), "noncrit_min": nc[0] if nc else None, "noncrit_max": nc[-1] if nc else None,
                   "n_crit": len(c), "crit_min": c[0] if c else None, "crit_max": c[-1] if c else None}
    print(json.dumps(summ, indent=1))
    os.makedirs(a.out, exist_ok=True)
    json.dump({"summary": summ, "rows": res}, open(os.path.join(a.out, "damage_ranges.json"), "w"), indent=0)


if __name__ == "__main__":
    main()
