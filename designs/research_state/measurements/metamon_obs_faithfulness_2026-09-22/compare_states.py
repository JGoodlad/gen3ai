#!/usr/bin/env python3
"""Compare, per decision point, what METAMON read from our stream against what WE read.

Pure standard library — it never imports either ``poke_env``, so it can run anywhere.

THE THREE CLASSES, decided by WHICH BLOCK a field lives in
----------------------------------------------------------
* **(b) GIGO candidate** — the field is in ``UniversalState`` (``us``), i.e. Metamon's observation
  is built from it. A disagreement here means the opponent we measure against is not seeing the
  battle we are playing.
* **(c) our bug** — the field is one OUR read models (``live`` / ``legal``) expose and the two
  parsers disagree, but ``UniversalState`` never reads it. Metamon is fine; we are not.
* **(a) presentation** — the two agree on the fact and differ only in how it is spelled or which
  slice is kept. ``replay_common.canon`` already removes case/punctuation, so what remains here
  are DELIBERATE representation choices, listed explicitly below rather than silently tolerated.

THE DELIBERATE (a)-CLASS RULES, each stated so it cannot quietly absorb a real difference
------------------------------------------------------------------------------------------
1. **Side conditions.** ``UniversalState.universal_conditions`` keeps only the MOST RECENT
   condition name (``max(rep.keys(), key=rep.get)``); our ``LiveSide.side_conditions`` keeps the
   whole mapping. The check is therefore CONTAINMENT: Metamon's single name must be a key we also
   hold, and its ``noconditions`` must mean we hold none. A name we do not hold is class (b).
2. **Unknown item / ability.** poke-env stores ``unknown_item`` / an empty ability until revealed;
   Metamon renames those to ``unknownitem`` / ``unknownability`` and our ``LiveView`` renders them
   as ``None``. The three spellings are folded to one sentinel, so a REVEALED item that one side
   has and the other does not is still a mismatch.
3. **``opponents_remaining``.** Metamon computes ``6 - |{fainted in opponent_team}|``; our
   ``LiveSide.remaining`` counts non-fainted KNOWN mons, which is a different quantity (a lower
   bound before the team is revealed). We reconstruct Metamon's quantity from OUR raw
   ``opp_team`` and compare that — comparing the two as-written would be a category error.

🚨 The comparator asserts its own denominator: zero compared decision points is a FAILURE.
"""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import replay_common as rc  # noqa: E402

#: poke-env's pre-reveal sentinels, Metamon's renames of them, and our ``None`` — one fact.
_UNKNOWN = {"", "unknownitem", "unknownability", "unknown_item", "noitem", "noability"}


def sentinel(value: str) -> str:
    return "?" if value in _UNKNOWN or not value else value


def status(value: str) -> str:
    """Rule (a)4 — Metamon spells "no status" as ``nostatus``; our ``LiveView`` spells it ``None``."""
    return "" if value in ("", "nostatus") else value


def types(values) -> list:
    """Rule (a)5 — ``UniversalPokemon.universal_types`` pads a mono-type to two with ``notype``
    (``force_two=True``). The padding is a fixed-width representation choice, not a claim about
    the Pokémon, so it is dropped before comparison; a REAL second type is untouched."""
    return sorted(v for v in values if v and v != "notype")


def hp_move(move_id: str) -> str:
    """Rule (a)6 — Hidden Power, and it is OUR representation that is deliberate.

    In a live gen3 request the server re-keys a typed Hidden Power to the bare ``hiddenpower``.
    ``LegalActions.move_slots`` keeps that WIRE-TRUTH on purpose (``live_view.py``'s
    ``own_hp_typed_id`` exists precisely because the typed id is kept separately); upstream's
    ``available_moves`` reports the ``Move`` object's typed ``.id``. Both sides know the same
    move. Folded to the bare id, so a genuinely DIFFERENT move still mismatches."""
    return "hiddenpower" if move_id.startswith("hiddenpower") else move_id


def mon_fields(prefix, meta_mon, our_mon, our_raw_mon):
    """(field name, metamon value, our value) triples for one Pokémon."""
    if meta_mon is None or our_mon is None:
        return [(f"{prefix}.present", meta_mon is not None, our_mon is not None)]
    out = [
        (f"{prefix}.species", meta_mon["species"], our_mon["species"]),
        (f"{prefix}.hp", meta_mon["hp"], our_mon["hp"]),
        (f"{prefix}.status", status(meta_mon["status"]), status(our_mon["status"])),
        (f"{prefix}.boosts", meta_mon["boosts"], our_mon["boosts"]),
        (f"{prefix}.moves", meta_mon["moves"], our_mon["moves"]),
        (f"{prefix}.types", types(meta_mon["types"]), types(our_mon["types"])),
        (f"{prefix}.level", meta_mon["level"], our_mon["level"]),
    ]
    # item/ability: compare against OUR RAW, because our LiveView renders an unrevealed item as
    # None by design while Metamon renames the same sentinel — rule (a)2 above.
    if our_raw_mon is not None:
        out += [
            (f"{prefix}.item", sentinel(meta_mon["item"]), sentinel(our_raw_mon["item"])),
            (f"{prefix}.ability", sentinel(meta_mon["ability"]), sentinel(our_raw_mon["ability"])),
        ]
    return out


def compare_row(meta, ours):
    """Every (class, field, metamon, ours) disagreement at one decision point."""
    diffs = []
    if "error" in meta or "error" in ours:
        if meta.get("error") != ours.get("error"):
            diffs.append(("b", "parse_error", meta.get("error"), ours.get("error")))
        return diffs

    mus, mraw = meta["us"], meta["raw"]
    olive, olegal, oraw = ours["live"], ours["legal"], ours["raw"]

    # ---- class (b): the fields UniversalState is built from ------------------------------ #
    our_active_raw = oraw["our_team"].get(mus["active"]["species"])
    our_opp_raw = oraw["opp_team"].get(mus["opp_active"]["species"])
    pairs = []
    pairs += mon_fields("active", mus["active"], olive["active"], our_active_raw)
    pairs += mon_fields("opp_active", mus["opp_active"], olive["opp_active"], our_opp_raw)
    pairs += [
        ("turn", meta["turn"], ours["turn"]),
        ("switches", mus["switches"], olive["bench"]),
        ("switch_hp", mus["switch_hp"], olive["bench_hp"]),
        ("weather", mus["weather"] if mus["weather"] != "noweather" else "",
         olive["weather"]),
        ("forced_switch", mus["forced_switch"], olegal["force_switch"]),
        ("opponents_remaining", mus["opponents_remaining"],
         6 - sum(1 for m in oraw["opp_team"].values() if m["fainted"])),
        ("won", mus["won"], olive["won"]),
        ("lost", mus["lost"], olive["lost"]),
    ]
    for name, m, o in pairs:
        if m != o:
            diffs.append(("b", name, m, o))

    # rule (a)8 — VOLATILES, by CONTAINMENT. `UniversalPokemon.universal_effects` keeps only the
    # most recent effect; our LiveView keeps the whole map. A name Metamon holds that we do not is
    # a real disagreement; a name we hold that Metamon dropped is its own narrowing.
    for prefix, m_mon, o_mon in (("active", mus["active"], olive["active"]),
                                 ("opp_active", mus["opp_active"], olive["opp_active"])):
        m_eff = m_mon.get("effect", "noeffect")
        o_vol = set(o_mon.get("volatiles", ())) if o_mon else set()
        if m_eff in ("noeffect", ""):
            if o_vol:
                diffs.append(("b", f"{prefix}.effect", "noeffect", sorted(o_vol)))
        elif m_eff not in o_vol:
            diffs.append(("b", f"{prefix}.effect", m_eff, sorted(o_vol)))

    # rule (a)1 — side conditions, by CONTAINMENT
    for name, m_cond, o_map in (("our_conditions", mus["our_conditions"], olive["our_conditions"]),
                                ("opp_conditions", mus["opp_conditions"], olive["opp_conditions"])):
        if m_cond == "noconditions":
            if o_map:
                diffs.append(("b", name, "noconditions", sorted(o_map)))
        elif m_cond not in o_map:
            diffs.append(("b", name, m_cond, sorted(o_map)))

    # ---- class (b): the decision surface both parsers derive from the SAME request ------- #
    for name, m, o in (
        # 🚨 Rule (a)7 — ENABLED vs ALL SLOTS. ``LegalActions.move_ids`` is every slot the
        # request listed, disabled ones included (``LegalMove.disabled`` carries the server's own
        # word, so our masker filters downstream); upstream's ``available_moves`` is already the
        # enabled subset. Comparing the two as-written reports a Taunt or a Choice lock as a
        # parser disagreement. The enabled subsets are the comparable pair.
        ("legal.moves", sorted(map(hp_move, mraw["available_moves"])),
         sorted(map(hp_move, olegal["enabled_move_ids"]))),
        ("legal.moves.raw", sorted(map(hp_move, mraw["available_moves"])),
         sorted(map(hp_move, oraw["available_moves"]))),
        ("legal.switches", mraw["available_switches"], olegal["switches"]),
        ("legal.force_switch", mraw["force_switch"], olegal["force_switch"]),
        ("legal.trapped", mraw["trapped"], olegal["trapped"]),
    ):
        if m != o:
            diffs.append(("b", name, m, o))

    # ---- class (c): the revealed team — OUR read models expose it, UniversalState does not - #
    m_team, o_team = mraw["opp_team"], oraw["opp_team"]
    if sorted(m_team) != sorted(o_team):
        diffs.append(("c", "opp_team.roster", sorted(m_team), sorted(o_team)))
    for species in sorted(set(m_team) & set(o_team)):
        if status(m_team[species]["status"]) != status(o_team[species]["status"]):
            diffs.append(("c", "opp_team.status", {species: m_team[species]["status"]},
                          {species: o_team[species]["status"]}))
        if types(m_team[species]["types"]) != types(o_team[species]["types"]):
            diffs.append(("c", "opp_team.types", {species: m_team[species]["types"]},
                          {species: o_team[species]["types"]}))
        for field in ("hp", "fainted", "moves", "volatiles"):
            if m_team[species][field] != o_team[species][field]:
                diffs.append(("c", f"opp_team.{field}", {species: m_team[species][field]},
                              {species: o_team[species][field]}))
        for field in ("item", "ability"):
            if sentinel(m_team[species][field]) != sentinel(o_team[species][field]):
                diffs.append(("c", f"opp_team.{field}", {species: m_team[species][field]},
                              {species: o_team[species][field]}))
    m_own, o_own = mraw["our_team"], oraw["our_team"]
    if sorted(m_own) != sorted(o_own):
        diffs.append(("c", "our_team.roster", sorted(m_own), sorted(o_own)))
    for species in sorted(set(m_own) & set(o_own)):
        if status(m_own[species]["status"]) != status(o_own[species]["status"]):
            diffs.append(("c", "our_team.status", {species: m_own[species]["status"]},
                          {species: o_own[species]["status"]}))
        for field in ("hp", "fainted", "moves", "boosts", "volatiles"):
            if m_own[species][field] != o_own[species][field]:
                diffs.append(("c", f"our_team.{field}", {species: m_own[species][field]},
                              {species: o_own[species][field]}))
    return diffs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--meta-dir", required=True)
    ap.add_argument("--ours-dir", required=True)
    ap.add_argument("--out", required=True, help="JSON summary")
    ap.add_argument("--examples", type=int, default=3, help="repro rows kept per mismatch class")
    args = ap.parse_args(argv)

    tags = sorted(f[len("meta-"):-len(".jsonl")]
                  for f in os.listdir(args.meta_dir) if f.startswith("meta-"))
    counts, examples = Counter(), {}
    n_points = n_battles = 0
    align_failures = []

    for tag in tags:
        meta = rc.load(os.path.join(args.meta_dir, f"meta-{tag}.jsonl"))
        ours = rc.load(os.path.join(args.ours_dir, f"ours-{tag}.jsonl"))
        if len(meta) != len(ours):
            # 🚨 Not a mismatch to be counted and moved past: the two dumps no longer describe the
            # same decisions, so every per-row comparison after it is meaningless.
            align_failures.append({"tag": tag, "meta": len(meta), "ours": len(ours)})
            continue
        n_battles += 1
        for m_row, o_row in zip(meta, ours):
            assert m_row["seq"] == o_row["seq"], (tag, m_row["seq"], o_row["seq"])
            n_points += 1
            for klass, field, m_val, o_val in compare_row(m_row, o_row):
                key = f"{klass}:{field}"
                counts[key] += 1
                bucket = examples.setdefault(key, [])
                if len(bucket) < args.examples:
                    bucket.append({"tag": tag, "seq": m_row["seq"], "turn": m_row.get("turn"),
                                   "metamon": m_val, "ours": o_val})

    if n_points == 0:
        raise SystemExit("0 decision points compared — a comparator that compared nothing is a "
                         "FAILED check, not a passing one (PREDICTION.md, last bar)")

    summary = {
        "n_battles": n_battles, "n_decision_points": n_points,
        "n_alignment_failures": len(align_failures), "alignment_failures": align_failures,
        "mismatch_counts": dict(sorted(counts.items())),
        "class_b_total": sum(v for k, v in counts.items() if k.startswith("b:")),
        "class_c_total": sum(v for k, v in counts.items() if k.startswith("c:")),
        "examples": examples,
    }
    with open(args.out, "w") as handle:
        json.dump(summary, handle, indent=1, default=str)
    print(json.dumps({k: v for k, v in summary.items() if k != "examples"},
                     indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
