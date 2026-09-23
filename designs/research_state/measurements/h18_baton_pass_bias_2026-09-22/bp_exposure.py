"""H18 — count the EXPOSURE: how many battles actually contained a Baton Pass, and how many
contained a Baton Pass that carried a nonzero stat stage (the only kind the defect can lose).

A bias quoted per GAME is diluted by every game the mechanism never appeared in. This is what
lets it be quoted per EXPOSURE as well.

Reads `--capture-dir` records written by `main.anchors --server rust`. Each holds `chunks` as
`(slot, text)`; the p1 stream carries every `|switch|`, `|-boost|` and `|-unboost|` line for
BOTH sides, because stat stages are public information.

    python3 bp_exposure.py <capture_dir> [<capture_dir> ...] --out exposure.json
"""
import argparse
import glob
import json
import os
import re

STATS = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")

#: The `|-start|` effect names that RIDE a Baton Pass — the volatile half of the mechanism,
#: mirroring `BATON_PASS_COPIED_EFFECTS` in the fork. `copyVolatileFrom` copies every volatile
#: whose condition is not `noCopy`, and emits nothing, exactly as it does for the stat stages.
COPIED_VOLATILES = {
    "substitute", "leech seed", "confusion", "curse", "perish song", "ingrain",
    "focus energy", "charge", "bind", "clamp", "fire spin", "sand tomb", "whirlpool", "wrap",
}


def zero():
    return {s: 0 for s in STATS}


def slot_of(ident: str) -> str:
    # "p1a: Zapdos" -> "p1a"
    return ident.split(":")[0].strip()


def scan_battle(path: str) -> dict:
    blob = json.load(open(path))
    text = "\n".join(t for slot, t in blob["chunks"] if slot == "p1")
    boosts = {}
    vol = {}
    n_bp = 0
    n_bp_with_volatile = 0
    n_bp_with_either = 0
    n_bp_with_boost = 0
    n_bp_by_side = {"p1": 0, "p2": 0}
    n_bp_boost_by_side = {"p1": 0, "p2": 0}
    for line in text.split("\n"):
        if not line.startswith("|"):
            continue
        parts = line.split("|")
        if len(parts) < 2:
            continue
        kw = parts[1]
        if kw in ("switch", "drag"):
            sl = slot_of(parts[2])
            is_bp = any(t.startswith("[from]") and "baton pass" in t.lower()
                        for t in parts[5:])
            if is_bp:
                n_bp += 1
                side = sl[:2]
                n_bp_by_side[side] = n_bp_by_side.get(side, 0) + 1
                carried = any(v != 0 for v in boosts.get(sl, zero()).values())
                carried_vol = bool(vol.get(sl))
                if carried:
                    n_bp_with_boost += 1
                    n_bp_boost_by_side[side] = n_bp_boost_by_side.get(side, 0) + 1
                if carried_vol:
                    n_bp_with_volatile += 1
                if carried or carried_vol:
                    n_bp_with_either += 1
                # the entrant KEEPS the passer's table AND its copyable volatiles in the sim
            else:
                boosts[sl] = zero()
                vol[sl] = set()
        elif kw in ("-boost", "-unboost"):
            sl = slot_of(parts[2])
            stat, amt = parts[3], int(parts[4])
            b = boosts.setdefault(sl, zero())
            b[stat] = max(-6, min(6, b[stat] + (amt if kw == "-boost" else -amt)))
        elif kw == "-setboost":
            sl = slot_of(parts[2])
            boosts.setdefault(sl, zero())[parts[3]] = int(parts[4])
        elif kw == "-clearboost":
            boosts[slot_of(parts[2])] = zero()
        elif kw == "-clearallboost":
            for k in boosts:
                boosts[k] = zero()
        elif kw == "-invertboost":
            b = boosts.setdefault(slot_of(parts[2]), zero())
            for s in b:
                b[s] = -b[s]
        elif kw == "-clearnegativeboost":
            b = boosts.setdefault(slot_of(parts[2]), zero())
            for s in b:
                b[s] = max(0, b[s])
        elif kw == "-start" and len(parts) > 3:
            sl = slot_of(parts[2])
            eff = parts[3].replace("move: ", "").strip().lower()
            if eff in COPIED_VOLATILES:
                vol.setdefault(sl, set()).add(eff)
        elif kw == "-end" and len(parts) > 3:
            sl = slot_of(parts[2])
            vol.setdefault(sl, set()).discard(
                parts[3].replace("move: ", "").strip().lower())
        elif kw == "-copyboost":
            src, dst = slot_of(parts[3]), slot_of(parts[2])
            boosts[dst] = dict(boosts.get(src, zero()))
    return {
        "tag": blob["tag"],
        "n_baton_pass": n_bp,
        "n_baton_pass_with_boost": n_bp_with_boost,
        "n_baton_pass_with_volatile": n_bp_with_volatile,
        "n_baton_pass_with_either": n_bp_with_either,
        "by_side": n_bp_by_side,
        "with_boost_by_side": n_bp_boost_by_side,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per_battle = []
    for d in args.dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.json"))):
            per_battle.append(scan_battle(p))
    if not per_battle:
        raise SystemExit("no captures found — a zero from a scan that read nothing is not a zero")

    n = len(per_battle)
    games_any = sum(1 for r in per_battle if r["n_baton_pass"] > 0)
    games_boost = sum(1 for r in per_battle if r["n_baton_pass_with_boost"] > 0)
    out = {
        "n_battles_scanned": n,
        "baton_pass_events": sum(r["n_baton_pass"] for r in per_battle),
        "baton_pass_events_carrying_a_boost": sum(
            r["n_baton_pass_with_boost"] for r in per_battle),
        "games_with_any_baton_pass": games_any,
        "games_with_a_boost_carrying_pass": games_boost,
        "games_with_a_volatile_carrying_pass": sum(
            1 for r in per_battle if r["n_baton_pass_with_volatile"] > 0),
        "games_with_a_state_carrying_pass": sum(
            1 for r in per_battle if r["n_baton_pass_with_either"] > 0),
        "baton_pass_events_carrying_a_volatile": sum(
            r["n_baton_pass_with_volatile"] for r in per_battle),
        "baton_pass_events_carrying_any_state": sum(
            r["n_baton_pass_with_either"] for r in per_battle),
        "frac_games_with_any_baton_pass": games_any / n,
        "frac_games_with_a_boost_carrying_pass": games_boost / n,
        "frac_games_with_a_state_carrying_pass": sum(
            1 for r in per_battle if r["n_baton_pass_with_either"] > 0) / n,
        "per_battle": per_battle,
    }
    json.dump(out, open(args.out, "w"), indent=2)
    for k, v in out.items():
        if k != "per_battle":
            print(f"{k}: {v}")


if __name__ == "__main__":
    main()
