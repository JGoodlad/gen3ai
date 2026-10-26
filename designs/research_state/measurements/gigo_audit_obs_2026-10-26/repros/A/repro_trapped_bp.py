"""MISSING 'trapped' after Baton Pass (investigator A): p1's Gengar Mean Looks p2's Celebi; p2 Baton Passes to Starmie.
Gen 3-4 copy 'trapped' through Baton Pass (data/mods/gen4/conditions.ts trapped.noCopy = false; gen3 inherits), so
Starmie stays trapped. Compare p1's opp-context 'trapped' cell and p2's own context cell with p2's request trapped bit."""
import agents.observation.constants as K
from agents.observation.gen3_effects import _SLOT_INDEX
import scripted as S

P1 = ("Gengar||leftovers|levitate|meanlook,thunderbolt,icepunch,willowisp|Timid|,,,252,4,252||,0,,,,|||]"
      "Blissey||leftovers|naturalcure|softboiled,icebeam,toxic,seismictoss|Bold|252,,252,,4,||,0,,,,|||")
P2 = ("Celebi||leftovers|naturalcure|batonpass,calmmind,psychic,recover|Bold|252,,252,,4,||,0,,,,|||]"
      "Starmie||leftovers|naturalcure|hydropump,icebeam,thunderbolt,rapidspin|Timid|4,,,252,,252||,0,,,,|||")
TRAP = K.BOOSTS_DIM + _SLOT_INDEX["trapped"]

def ctx(row, side):
    o = K.OFFSET_CONTEXT + side * K.ACTIVE_CONTEXT_DIM
    return row[o:o + K.ACTIVE_CONTEXT_DIM]

def act(row):
    for i in range(6):
        s = row[K.OFFSET_OUR_TEAM + i * K.POKEMON_FULL_DIM: K.OFFSET_OUR_TEAM + (i + 1) * K.POKEMON_FULL_DIM]
        if s[K.POKEMON_ACTIVE_OFFSET] == 1:
            return i, s
    return None, None

# p1: Mean Look then Soft-Boiled-free Gengar moves (Thunderbolt slot 2 would KO? use Will-O-Wisp slot 4 = action 9)
# p2: turn 1 Calm Mind (slot 2 = 7), turn 2 Baton Pass (slot 1 = 6) -> switch request: Starmie (team slot 1 = action 1), then Starmie move 1
out, script = S.play(P1, P2, p1_plan=[6, 9, 9, 9, 9, 9], p2_plan=[7, 6, 1, 6, 6, 6], max_steps=10)
for k, (row, mask, turn) in enumerate(out[0][:5]):
    print(f"p1 dec {k}: OPP ctx trapped={ctx(row, 1)[TRAP]:.0f}  opp spa boost cells={ctx(row,1)[4]:.3f}")
for k, (row, mask, turn) in enumerate(out[1][:6]):
    i, s = act(row)
    print(f"p2 dec {k}: active slot {i} OWN ctx trapped={ctx(row, 0)[TRAP]:.0f} request trapped={s[K.POKEMON_TRAPPED_OFFSET]:.0f} maybe={s[K.POKEMON_MAYBE_TRAPPED_OFFSET]:.0f} switch legal={bool(mask[:6].any())}")
