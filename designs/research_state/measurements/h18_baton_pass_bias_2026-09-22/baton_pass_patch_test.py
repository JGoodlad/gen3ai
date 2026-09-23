"""H18 — the ported Baton Pass carry-over test, run under the METAMON interpreter.

Ported from the gen3ai fork's `src/poke_env/battle/baton_pass_carryover_test.py` (2026-08-23).
It is a POSITIVE-CONTROL PAIR, not one test: run it against the installed upstream package and
it must FAIL the carry-over assertions; run it with the patched shadow copy first on
`PYTHONPATH` and it must pass all of them. A patch that patched nothing is indistinguishable
from a working one unless the unpatched half is shown to fail.

    <metamon python> baton_pass_patch_test.py            # expects UNPATCHED behaviour
    PYTHONPATH=<shadow> <metamon python> baton_pass_patch_test.py --expect-patched
"""
import sys

import poke_env
from poke_env.environment.battle import Battle
from poke_env.environment.effect import Effect

EXPECT_PATCHED = "--expect-patched" in sys.argv

print(f"[h18-test] poke_env.__file__ = {poke_env.__file__}")


def battle():
    b = Battle("tag", "player1", None, gen=3)
    b._player_role = "p1"
    return b


def feed(b, *lines):
    for line in lines:
        b.parse_message(line.split("|"))


def setup_passer(b):
    feed(
        b,
        "|switch|p1a: Celebi|Celebi|100/100",
        "|switch|p2a: Blissey|Blissey, F|100/100",
        "|-boost|p1a: Celebi|spa|2",
        "|-boost|p1a: Celebi|spd|2",
        "|-start|p1a: Celebi|Substitute",
    )


results = {}

# 1 — boosts ride the pass
b = battle()
setup_passer(b)
assert b.active_pokemon.boosts["spa"] == 2, "the passer never got its boosts"
feed(b, "|switch|p1a: Charizard|Charizard, M|100/100|[from] Baton Pass")
results["boosts_carried"] = (
    b.active_pokemon.species == "charizard"
    and b.active_pokemon.boosts["spa"] == 2
    and b.active_pokemon.boosts["spd"] == 2
)

# 2 — the copyable volatile rides the pass
b = battle()
setup_passer(b)
feed(b, "|switch|p1a: Charizard|Charizard, M|100/100|[from] Baton Pass")
results["substitute_carried"] = Effect.SUBSTITUTE in b.active_pokemon.effects

# 3 — a PLAIN switch still clears (the fix must not make every switch preserve state)
b = battle()
setup_passer(b)
feed(b, "|switch|p1a: Charizard|Charizard, M|100/100")
results["plain_switch_clears"] = (
    b.active_pokemon.boosts["spa"] == 0 and Effect.SUBSTITUTE not in b.active_pokemon.effects
)

# 4 — a phaze drag never carries
b = battle()
setup_passer(b)
feed(b, "|drag|p1a: Charizard|Charizard, M|100/100")
results["drag_clears"] = b.active_pokemon.boosts["spa"] == 0

# 5 — the OPPONENT's pass is tracked too (this is the half that biases the anchor)
b = battle()
feed(
    b,
    "|switch|p1a: Celebi|Celebi|100/100",
    "|switch|p2a: Jolteon|Jolteon, M|100/100",
    "|-boost|p2a: Jolteon|spe|2",
    "|switch|p2a: Snorlax|Snorlax, M|100/100|[from] Baton Pass",
)
results["opponent_pass_carried"] = b.opponent_active_pokemon.boosts["spe"] == 2

# 6 — negative stages ride the pass as well (the `-1` half of the measured defect)
b = battle()
feed(
    b,
    "|switch|p1a: Celebi|Celebi|100/100",
    "|switch|p2a: Blissey|Blissey, F|100/100",
    "|-boost|p1a: Celebi|spa|2",
    "|-unboost|p1a: Celebi|def|2",
    "|switch|p1a: Charizard|Charizard, M|100/100|[from] Baton Pass",
)
results["negative_stages_carried"] = (
    b.active_pokemon.boosts["spa"] == 2 and b.active_pokemon.boosts["def"] == -2
)

for k, v in results.items():
    print(f"[h18-test] {k}: {v}")

CARRY = ["boosts_carried", "substitute_carried", "opponent_pass_carried",
         "negative_stages_carried"]
INVARIANT = ["plain_switch_clears", "drag_clears"]

bad = [k for k in INVARIANT if not results[k]]
assert not bad, f"INVARIANT BROKEN (a plain switch must still clear): {bad}"

if EXPECT_PATCHED:
    missing = [k for k in CARRY if not results[k]]
    assert not missing, f"PATCH DID NOT TAKE — still losing: {missing}"
    print("[h18-test] PATCHED: all four carry-over assertions hold, both invariants hold")
else:
    carried = [k for k in CARRY if results[k]]
    assert not carried, f"UNPATCHED copy already carries {carried} — the arms are not distinct"
    print("[h18-test] UNPATCHED: all four carry-over assertions FAIL, as H18 says they must")
