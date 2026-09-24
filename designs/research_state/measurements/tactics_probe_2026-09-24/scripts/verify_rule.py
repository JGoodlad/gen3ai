"""Verify the gen-3 TURN CUT in the Rust port against Node Showdown, on constructed battles.

Rule (Showdown sim/battle.ts faintMessages, gen <= 3 singles): any faint mid-turn cancels EVERY remaining
queued action of every active mon; the turn skips to residuals. So a faster self-KO (Explosion) or a
faster attacker's recoil suicide (Double-Edge) DENIES the slower opponent's queued move.

Cases (each run on impl=rust AND impl=node from the same seed; the omniscient-relevant per-side lines of
the probe turn are compared and printed):
  A  fast Gengar Explosion vs Tyranitar Dragon Dance       -> DD must NOT appear, TTar stays +0
  B  slow Gengar (TTar already +1) Explosion vs DD          -> DD resolves FIRST (+2), then Explosion
  C  low-HP Tyranitar Double-Edge (recoil KO) vs Soft-Boiled -> Soft-Boiled must NOT appear
  D  high-HP Tyranitar Double-Edge (no recoil KO) vs Soft-Boiled -> Soft-Boiled resolves
Also records Explosion's damage on Tyranitar and Double-Edge's damage/recoil (the thresholds).

    python verify_rule.py --out <dir>
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tlib  # noqa: E402
import teams as T  # noqa: E402


def cases():
    ours_D = tlib.pack(T.team_D_ours(lead="gengar"))
    opp_D = tlib.pack(T.team_D_opp())
    out = {}
    # A: turn 1 Gengar (308) Explosion vs TTar (213) DD.
    out["A_fast_explosion_vs_dd"] = dict(
        p1=ours_D, p2=opp_D, probe_turn=1,
        s1=["move explosion", "switch Swampert"], s2=["move dragondance"])
    # B: turn 1 Gengar Thunderbolt / TTar DD (+1 -> 319 > 308); turn 2 Explosion vs DD: DD first.
    out["B_slow_explosion_vs_dd"] = dict(
        p1=ours_D, p2=opp_D, probe_turn=2,
        s1=["move thunderbolt", "move explosion", "switch Swampert"],
        s2=["move dragondance", "move dragondance"])
    # C/D: our Tyranitar (205) vs their Blissey (153).
    ours_B = tlib.pack(T.team_B_ours())
    opp_B = tlib.pack(T.team_B_opp())
    # D: turn 1 Double-Edge at full HP vs Soft-Boiled.
    out["D_de_no_recoil_ko_vs_softboiled"] = dict(
        p1=ours_B, p2=opp_B, probe_turn=1,
        s1=["move doubleedge"], s2=["move softboiled"])
    # C: bring TTar low first: Blissey Ice Beams it while TTar uses Dragon Dance (non-damaging), then DE.
    # C: bring TTar low WITHOUT touching Blissey: their Salamence Dragon Claws it three times while our
    # TTar's Earthquake hits nothing (Salamence is Flying); Blissey comes in on turn 4 (our EQ hits it),
    # then turn 5 is Double-Edge vs Soft-Boiled. (Intimidate leaves our TTar at -1 Atk; recorded.)
    opp_B_mence = tlib.pack([T.SALAMENCE_e118] + [s for s in T.team_B_opp() if s != T.SALAMENCE_e118])
    out["C_de_recoil_ko_vs_softboiled"] = dict(
        p1=ours_B, p2=opp_B_mence, probe_turn=5,
        s1=["move earthquake", "move earthquake", "move earthquake", "move earthquake", "move doubleedge",
            "switch Zapdos"],
        s2=["move dragonclaw", "move dragonclaw", "move dragonclaw", "switch Blissey", "move softboiled"])
    return out


KEEP = ("|move|", "|-damage|", "|-heal|", "|faint|", "|-boost|", "|-unboost|", "|cant|", "|switch|",
        "|-fail|", "|-activate|", "|-status|", "|-miss|", "|-crit|", "|-resisted|", "|-supereffective|",
        "|-immune|", "|-ability|", "|win|", "|turn|", "|upkeep")


def norm(lines, names):
    out = []
    for ln in lines:
        if not ln.startswith(KEEP):
            continue
        for k, v in names.items():
            ln = ln.replace(k, v)
        out.append(ln)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--impls", default="rust,node")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    res = {}
    for name, c in cases().items():
        seed = tlib.mint_seed("verify_rule:" + name)
        per = {}
        for impl in args.impls.split(","):
            r = tlib.run_scripted(c["p1"], c["p2"], list(c["s1"]), list(c["s2"]), seed, impl=impl)
            names = {r["p1"].username: "P1", r["p2"].username: "P2"}
            lines = norm(tlib.side_lines(r["sink"], "p1"), names)
            per[impl] = {"probe_turn_lines": tlib.turn_block(lines, c["probe_turn"]),
                         "all_lines": lines,
                         "script_left": r["left"]}
        impls = list(per)
        same = all(per[i]["all_lines"] == per[impls[0]]["all_lines"] for i in impls)
        res[name] = {"seed": seed, "probe_turn": c["probe_turn"], "identical_across_impls": same,
                     **{i: {"probe_turn_lines": per[i]["probe_turn_lines"], "script_left": per[i]["script_left"]}
                        for i in impls}}
        print(f"=== {name}  (seed {seed}; identical across {impls}: {same})")
        for ln in per[impls[0]]["probe_turn_lines"]:
            print("   ", ln)
        if not same:
            for i in impls:
                print(f"  --- {i}")
                for ln in per[i]["probe_turn_lines"]:
                    print("     ", ln)
        with open(os.path.join(args.out, f"verify_{name}.lines.json"), "w") as fh:
            json.dump({i: per[i]["all_lines"] for i in impls}, fh, indent=0)
    with open(os.path.join(args.out, "verify_rule.json"), "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
