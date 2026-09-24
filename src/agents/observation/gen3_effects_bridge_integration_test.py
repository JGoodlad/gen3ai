"""A REAL bridge battle that uses the one-shot effects, fully encoded at every decision.

``gen3_effects_test.py`` proves the classification from Showdown's SOURCE and from protocol lines
fed to a real ``Gen3Battle``. This proves it on the PROTOCOL a real gen3 battle emits: two
scripted players (fixed teams, fixed sim seed, concurrency 1, no randomness in either policy) play
one battle on the node bridge — the reference Showdown, which models every move below (the Rust
port's coverage filter excludes Beat Up) — and BOTH run the full ``embed_battle`` observation
encode at every decision. Before ``gen3_effects.NOT_A_VOLATILE`` existed, the first decision after
the Heal Bell raised ``UnknownVolatileError: 'healbell'`` (the belief-calibration crash of
2026-09-24; reproduced by this test on the pre-fix encoder). The script:

    T1  p1 Blissey Thunder Wave   p2 Vileplume Toxic          Vileplume paralysed, Blissey poisoned
    T2  p1 Blissey Heal Bell      p2 Vileplume Aromatherapy   -activate + -curestatus / -cureteam
    T3  p1 → Dugtrio              p2 Vileplume Toxic          Dugtrio poisoned (not a Beat Up ally)
    T4  p1 Dugtrio Beat Up        p2 Vileplume Sludge Bomb    NO -activate (Beat Up Nicknames Mod)
    T5  p1 Dugtrio Magnitude      p2 → Gengar                 -activate|…|move: Magnitude|N
    T6  p1 → Poliwrath            p2 Gengar Ice Punch
    T7  p1 Poliwrath Mind Reader  p2 Gengar Ice Punch         -activate|…|move: Mind Reader
    T8  p1 Poliwrath Brick Break  p2 Gengar Spite             -activate|…|move: Spite|Mind Reader|…
    T9  p1 Poliwrath Surf         p2 → Porygon2
    T10 p1 Poliwrath Surf         p2 Porygon2 Conversion      -start|…|typechange|…

after which both play their first legal damaging move until the battle ends. Each scripted
action is asserted to have HAPPENED (its protocol line is present) rather than branched on, so
a seed that stops producing one fails loudly instead of passing green.
"""
from __future__ import annotations

import asyncio
from typing import List, Optional, Set, Tuple

import numpy as np
import pytest

from poke_env import AccountConfiguration
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.battle.gen3_battle import Gen3Battle
from agents.inference.player import Gen3Player
from utils.bridge.local_battle_runner import run_local_battles

pytestmark = pytest.mark.sim

_P1_TEAM = """
Blissey @ Leftovers
Ability: Serene Grace
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Thunder Wave
- Heal Bell
- Seismic Toss
- Soft-Boiled

Dugtrio @ Leftovers
Ability: Sand Veil
EVs: 252 Atk / 4 SpD / 252 Spe
Jolly Nature
- Beat Up
- Magnitude
- Earthquake
- Rock Slide

Poliwrath @ Leftovers
Ability: Water Absorb
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Mind Reader
- Brick Break
- Surf
- Mist
"""

_P2_TEAM = """
Vileplume @ Leftovers
Ability: Chlorophyll
EVs: 252 HP / 252 SpA / 4 SpD
Modest Nature
- Toxic
- Aromatherapy
- Giga Drain
- Sludge Bomb

Gengar @ Leftovers
Ability: Levitate
EVs: 252 SpA / 4 SpD / 252 Spe
Timid Nature
- Spite
- Thunderbolt
- Ice Punch
- Psychic

Porygon2 @ Leftovers
Ability: Trace
EVs: 252 HP / 252 SpA / 4 SpD
Modest Nature
- Conversion
- Ice Beam
- Thunderbolt
- Recover
"""

# (kind, target): kind "move" names a move id, "switch" names a species id.
_P1_SCRIPT = [("move", "thunderwave"), ("move", "healbell"), ("switch", "dugtrio"),
              ("move", "beatup"), ("move", "magnitude"), ("switch", "poliwrath"),
              ("move", "mindreader"), ("move", "brickbreak"), ("move", "surf"), ("move", "surf")]
_P2_SCRIPT = [("move", "toxic"), ("move", "aromatherapy"), ("move", "toxic"),
              ("move", "sludgebomb"), ("switch", "gengar"), ("move", "icepunch"),
              ("move", "icepunch"), ("move", "spite"), ("switch", "porygon2"),
              ("move", "conversion")]
_SEED = [101, 202, 303, 404]


class _ScriptedEncodingPlayer(Gen3Player):
    """Plays a fixed script, then its first legal damaging move; ENCODES the full observation at
    every decision (the call that raised pre-fix) and records every volatile id it met."""

    def __init__(self, *, script, **kwargs):
        super().__init__(battle_class=Gen3Battle, **kwargs)
        self._script: List[Tuple[str, str]] = list(script)
        self.met: Set[str] = set()
        self.decisions = 0
        self.battle: Optional[Gen3Battle] = None
        self.snapshots: list = []

    def _pick(self, battle, mask) -> int:
        legal = [int(i) for i in np.flatnonzero(mask)]
        orders = [(i, self.action_to_order(i, battle).order) for i in legal]
        if self._script:
            kind, want = self._script[0]
            for i, o in orders:
                if ((kind == "move" and isinstance(o, Move) and o.id == want)
                        or (kind == "switch" and isinstance(o, Pokemon) and o.species == want)):
                    self._script.pop(0)
                    return i
        for i, o in orders:  # after the script / a forced switch: deterministic fallback
            if isinstance(o, Move) and o.base_power > 0:
                return i
        return legal[0]

    def choose_move(self, battle):
        self.battle = battle
        obs = self.embed_battle(battle)  # the full obs — raised UnknownVolatileError pre-fix
        assert np.all(np.isfinite(obs["observation"]))
        self.decisions += 1
        view = battle.live_view()
        self.snapshots.append(view)
        for side in (view.ours, view.opp):
            for mon in side.mons:
                self.met.update(mon.volatiles)
        mask = obs["action_mask"]
        if int(mask.sum()) == 0:
            return self.choose_default_move()
        idx = self._pick(battle, mask)
        self._get_tracker(battle).advance(idx)
        return self.action_to_order(idx, battle)


@pytest.fixture(scope="module")
def played():
    p1 = _ScriptedEncodingPlayer(
        script=_P1_SCRIPT, battle_format="gen3ou", team=_P1_TEAM,
        account_configuration=AccountConfiguration("HbHealer", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1)
    p2 = _ScriptedEncodingPlayer(
        script=_P2_SCRIPT, battle_format="gen3ou", team=_P2_TEAM,
        account_configuration=AccountConfiguration("HbAroma", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1)
    asyncio.run(run_local_battles(p1, p2, 1, seed=_SEED, impl="node", concurrency=1))
    return p1, p2


def _lines(player) -> List[str]:
    """The protocol this player's battle parsed (each event's verbatim split line)."""
    return ["|".join(e.raw) for e in player.battle.events_since(0) if e.raw]


def _first_view_with(player, side: str, species: str, vid: str):
    for view in player.snapshots:
        mon = (view.ours if side == "ours" else view.opp).get(species)
        if mon is not None and vid in mon.volatiles:
            return mon
    return None


def test_battle_went_as_scripted_with_every_decision_encoded(played):
    """The precondition of every other test, ASSERTED: the battle finished, both scripts were
    consumed, and each line the script exists to produce is in the protocol."""
    p1, p2 = played
    assert p1.battle is not None and p1.battle.finished and p2.battle.finished
    assert not p1._script and not p2._script, (
        f"a scripted action never became legal — the battle did not go as scripted: "
        f"p1 left {p1._script}, p2 left {p2._script}")
    log = "\n".join(_lines(p1))
    for needle in ("|-activate|p1a: Blissey|move: Heal Bell", "|-curestatus|p1a: Blissey|tox",
                   "|-cureteam|p2a: Vileplume|[from] move: Aromatherapy",
                   "|move|p1a: Dugtrio|Beat Up", "|-activate|p1a: Dugtrio|move: Magnitude",
                   "|-activate|p1a: Poliwrath|move: Mind Reader", "|-activate|p1a: Poliwrath|move: Spite",
                   "|-start|p2a: Porygon2|typechange"):
        assert needle in log, f"the scripted battle never produced {needle!r}:\n{log}"


def test_heal_bell_met_the_encoder_and_its_cure_arrived(played):
    """THE REGRESSION: `-activate|…|move: Heal Bell` puts `healbell` on the user; BOTH players'
    full encodes met it (ours and the opponent's view) without raising. Its consequence arrived
    on its own line, `-curestatus`: at that decision Blissey is no longer poisoned."""
    p1, p2 = played
    mine = _first_view_with(p1, "ours", "blissey", "healbell")
    theirs = _first_view_with(p2, "opp", "blissey", "healbell")
    assert mine is not None and theirs is not None
    assert mine.status is None and theirs.status is None


def test_aromatherapy_is_cureteam_in_gen3_and_never_a_volatile(played):
    """The suspected sibling is NOT one in gen3: the gen4 mod gen3 inherits emits
    `-cureteam|<user>|[from] move: Aromatherapy`, which poke-env folds into a team cure and never
    into `mon.effects` — measured here, so a Showdown change that brings back an `-activate`
    fails this test instead of a ladder game. The cure is real: Vileplume's paralysis is gone
    at p2's next decision."""
    p1, p2 = played
    assert "aromatherapy" not in p1.met | p2.met
    idx = next(i for i, v in enumerate(p2.snapshots)
               if v.ours.get("vileplume") is not None and v.ours.get("vileplume").status == "par")
    later = [v.ours.get("vileplume") for v in p2.snapshots[idx + 1:]]
    assert any(m is not None and m.status is None for m in later)


@pytest.mark.parametrize("vid", ["magnitude", "spite", "typechange"])
def test_each_new_classification_met_both_encoders(played, vid):
    """Every id this change classified that a turn-start decision can see reached BOTH players'
    full encodes on real protocol. (`mindreader` is classified too, but poke-env drops
    MIND_READER at the next `|turn|` — `ends_on_turn` — so only a mid-turn forced-switch decision
    can meet it; `gen3_effects_test` covers it on the protocol line.)"""
    p1, p2 = played
    assert vid in p1.met and vid in p2.met


def test_beat_up_announces_nothing_in_gen3ou(played):
    """Beat Up's `-activate|…|move: Beat Up|[of] <ally>` is gated by `Beat Up Nicknames Mod`,
    which gen3's `Standard AG` includes — so in gen3 OU it NEVER reaches `mon.effects`. Were the
    rule dropped, the line would land as `unknown` (poke-env has no Effect.BEAT_UP) and crash
    every encode: 11.4% of Metamon's gen3ou teams carry Beat Up."""
    p1, p2 = played
    assert not any("move: Beat Up" in ln for ln in _lines(p1) if ln.startswith("|-activate|"))
    assert not ({"beatup", "unknown"} & (p1.met | p2.met))


def test_conversion_changed_the_encoded_type(played):
    """`typechange` is NOT a volatile because the per-mon TYPE block carries it: poke-env's
    `start_effect(TYPECHANGE, details)` sets `_temporary_types`, which `type_1`/`type_2` (the obs
    TypeEncoder) and `LivePokemon.types` read. Porygon2 is pure Normal; gen3 Conversion picks the
    type of one of its moves that is not already its type (Ice Beam / Thunderbolt here)."""
    _, p2 = played
    mon = _first_view_with(p2, "ours", "porygon2", "typechange")
    assert mon is not None
    assert mon.types != ("normal",) and len(mon.types) == 1, mon.types
