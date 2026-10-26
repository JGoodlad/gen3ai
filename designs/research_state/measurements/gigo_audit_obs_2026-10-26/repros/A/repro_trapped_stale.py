"""STALE 'trapped' volatile (investigator A): p1's Gengar uses Mean Look on p2's Snorlax, then p1 switches out.
Showdown removes the target's linked 'trapped' volatile SILENTLY when the trapper leaves (pokemon.ts clearVolatile ->
removeLinkedVolatiles; the 'trapped' condition has no onEnd line), so p2 is FREE. Read p1's row (the opp active context's
'trapped' volatile cell) and p2's own row (its own context's 'trapped' cell vs its request trapped bit / switch mask)."""
from __future__ import annotations

import numpy as np

import agents.observation.constants as K
from agents.observation.gen3_effects import _SLOT_INDEX
import scripted as S

P1 = ("Gengar||leftovers|levitate|meanlook,thunderbolt,icepunch,willowisp|Timid|,,,252,4,252||,0,,,,|||]"
      "Skarmory||leftovers|keeneye|spikes,roar,toxic,protect|Careful|252,,4,,252,||,0,,,,|||]"
      "Blissey||leftovers|naturalcure|softboiled,icebeam,toxic,seismictoss|Bold|252,,252,,4,||,0,,,,|||")
P2 = ("Snorlax||leftovers|immunity|curse,rest,bodyslam,earthquake|Careful|252,,4,,252,|||||]"
      "Starmie||leftovers|naturalcure|hydropump,icebeam,thunderbolt,rapidspin|Timid|4,,,252,,252||,0,,,,|||")

TRAP = K.BOOSTS_DIM + _SLOT_INDEX["trapped"]


def ctx(row, side):  # side 0 = the row owner's, 1 = its opponent's
    o = K.OFFSET_CONTEXT + side * K.ACTIVE_CONTEXT_DIM
    return row[o:o + K.ACTIVE_CONTEXT_DIM]


def own_active(row):
    for i in range(6):
        s = row[K.OFFSET_OUR_TEAM + i * K.POKEMON_FULL_DIM: K.OFFSET_OUR_TEAM + (i + 1) * K.POKEMON_FULL_DIM]
        if s[K.POKEMON_ACTIVE_OFFSET] == 1:
            return i, s
    return None, None


if __name__ == "__main__":
    # p1: Mean Look (move slot 1 = action 6), then switch to Skarmory (team slot 1), then Skarmory's move 1.
    out, script = S.play(P1, P2, p1_plan=[6, 1, 6, 6, 6], p2_plan=[6, 6, 6, 6, 6], max_steps=700)
    for k, (row, mask, turn) in enumerate(out[0][:5]):
        i, s = own_active(row)
        import math
        from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
        CLK = Gen3ObservationEncoder(load_mappings()).get_layout()["global_layout"]["clock"]["offset"]
        rt = math.exp(row[K.OFFSET_GLOBAL + CLK] * math.log(1 + K.MAX_TURNS)) - 1
        print(f"p1 dec {k} col-turn {turn} row-turn {rt:.3f}: our active slot {i} species {s[0] if s is not None else None}; "
              f"OPP ctx trapped={ctx(row, 1)[TRAP]}")
    for k, (row, mask, turn) in enumerate(out[1][:5]):
        i, s = own_active(row)
        print(f"p2 dec {k} turn {turn}: OWN ctx trapped={ctx(row, 0)[TRAP]}  request trapped bit="
              f"{s[K.POKEMON_TRAPPED_OFFSET]} maybe={s[K.POKEMON_MAYBE_TRAPPED_OFFSET]}  switch legal={bool(mask[:6].any())}")
    print("finished:", script is not None, "p1 decisions", len(out[0]))
    if script is None:
        raise SystemExit(0)
    t = S.trace_lines(script, 0)
    import json
    rec = t["records"]["p1"]
    lines = [l for l in rec.splitlines()]
    print("p1 record lines (first 60 with trapped / switch / move):")
    for l in lines:
        if any(x in l for x in ("trapped", "Mean Look", "\"switch\"", "|switch|", "|move|", "|turn|")):
            print("  ", l[:200])
