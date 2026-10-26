"""STALE single-turn / single-move volatiles (investigator A).

p2's Gengar: turn 1 Destiny Bond (`-singlemove`, lasts until its user's NEXT move: Showdown destinybond.condition
onBeforeMove -> removeVolatile, silent), turn 2 Endure (`-singleturn`, lasts that turn only: Showdown endure.condition
duration 1, no -end line), then Thunderbolt. p1's Blissey only uses Soft-Boiled / Seismic Toss (cannot KO quickly).
Read p1's opponent-context volatile cells for destinybond / endure at every decision, and p2's own."""
from __future__ import annotations

import agents.observation.constants as K
from agents.observation.gen3_effects import _SLOT_INDEX
import scripted as S

P1 = ("Blissey||leftovers|naturalcure|softboiled,seismictoss,toxic,icebeam|Bold|252,,252,,4,||,0,,,,|||]"
      "Skarmory||leftovers|keeneye|spikes,roar,toxic,protect|Careful|252,,4,,252,||,0,,,,|||")
P2 = ("Gengar||leftovers|levitate|destinybond,endure,thunderbolt,icepunch|Timid|,,,252,4,252||,0,,,,|||]"
      "Starmie||leftovers|naturalcure|hydropump,icebeam,thunderbolt,rapidspin|Timid|4,,,252,,252||,0,,,,|||")

CELLS = {name: K.BOOSTS_DIM + _SLOT_INDEX[name] for name in ("destinybond", "endure", "protect", "snatch", "rage", "grudge")}


def ctx(row, side):
    o = K.OFFSET_CONTEXT + side * K.ACTIVE_CONTEXT_DIM
    return row[o:o + K.ACTIVE_CONTEXT_DIM]


if __name__ == "__main__":
    # p1: Soft-Boiled (slot 1) every turn.  p2: DB (slot 1), Endure (slot 2), then Thunderbolt (slot 3) x4.
    out, script = S.play(P1, P2, p1_plan=[6] * 8, p2_plan=[6, 7, 8, 8, 8, 8], max_steps=8)
    for k, (row, mask, turn) in enumerate(out[0][:6]):
        c = ctx(row, 1)
        print(f"p1 dec {k} (before turn {k + 1}): OPP ctx " + " ".join(f"{n}={c[i]:.0f}" for n, i in CELLS.items()))
    for k, (row, mask, turn) in enumerate(out[1][:6]):
        c = ctx(row, 0)
        print(f"p2 dec {k} (before turn {k + 1}): OWN ctx " + " ".join(f"{n}={c[i]:.0f}" for n, i in CELLS.items()))
