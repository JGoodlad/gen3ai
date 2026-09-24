"""END-TO-END gates for the fork's `gen3_pe_reading_fixes_v1`: three CONSTRUCTED real battles, read
at the OBSERVATION.

`src/poke_env/battle/reading_fixes_test.py` pins each rule with hand-fed protocol lines. These refuse
to trust that the hand-fed lines are what the simulator emits: each plays a scripted battle through
the in-process bridge on the REAL Showdown (`impl="node"`, `deps/pokemon-showdown`), and asserts the
truth survives to the bytes the model is trained on — poke-env's `Pokemon`, our `LiveView`, and the
2501-dim observation (offsets from the declared layout, never literals). Each battle also asserts
its own SCENARIO from the raw protocol, so a battle that drifted off the shape proves nothing loudly.

* **PE-V10** — Snorlax Curses twice then Self-Destructs: at our replacement decision the corpse's
  active-context boosts are EMPTY (upstream poke-env kept +2 Atk / +2 Def / −2 Spe).
* **PE-R1b** — Snorlax is badly poisoned, pivots out, and re-enters as a post-faint replacement
  AFTER the residual. At every one of our decisions each badly-poisoned mon's count equals the
  residual `[from] psn` chips since its switch-in (0 benched), re-derived from the RAW protocol; the
  re-entry decision reads 0 (upstream read 1, one AHEAD).
* **PE-V16** — Houndoom (Flash Fire) absorbs a Flamethrower, then uses its own Flamethrower: the
  `flashfire` volatile slot stays set (upstream ended it at the holder's own Fire move).

Run directly (no server) or under pytest as a `sim` test:
    python src/agents/training/poke_env_gaps/pe_reading_fixes_obs_integration_test.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from typing import Any, Dict, List, Optional

import pytest

from agents.battle.gen3_battle import Gen3Battle
from agents.observation.constants import BOOSTS_DIM, OFFSET_CONTEXT
from agents.observation.gen3_effects import VOLATILE_SLOTS
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.training.poke_env_gaps.baton_pass_obs_integration_test import decode_obs_boosts
from poke_env import AccountConfiguration
from poke_env.battle.effect import Effect
from poke_env.battle.status import Status
from poke_env.player.player import Player
from utils.bridge.local_battle_runner import run_local_battles
from utils.teambuilder import Gen3Teambuilder

pytestmark = pytest.mark.sim

SEED = [1, 2, 3, 4]

SNORLAX_TEAM = """\
Snorlax @ Leftovers
Ability: Thick Fat
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Curse
- Self-Destruct
- Body Slam
- Earthquake

Magikarp
Ability: Swift Swim
EVs: 4 HP
Hardy Nature
- Splash
- Tackle

Skarmory @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 252 Def / 4 Spe
Impish Nature
- Drill Peck
- Protect
- Roar
- Spikes
"""

# Toxic first, then Seismic Toss — never anything that can KO a Snorlax quickly.
BLISSEY_TEAM = """\
Blissey @ Leftovers
Ability: Natural Cure
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Toxic
- Seismic Toss
- Soft-Boiled
- Protect

Umbreon @ Leftovers
Ability: Synchronize
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Toxic
- Mean Look
- Wish
- Protect

Skarmory @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 252 Def / 4 Spe
Impish Nature
- Drill Peck
- Protect
- Roar
- Spikes
"""

HOUNDOOM_TEAM = """\
Houndoom @ Leftovers
Ability: Flash Fire
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Flamethrower
- Crunch
- Hidden Power [Grass]
- Pursuit

Snorlax @ Leftovers
Ability: Thick Fat
EVs: 252 HP / 252 Atk / 4 SpD
Adamant Nature
- Body Slam
- Curse
- Rest
- Earthquake

Skarmory @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 252 Def / 4 Spe
Impish Nature
- Drill Peck
- Protect
- Roar
- Spikes
"""

FIRE_TEAM = """\
Charizard @ Leftovers
Ability: Blaze
EVs: 252 HP / 252 SpA / 4 Spe
Modest Nature
- Flamethrower
- Hidden Power [Grass]
- Substitute
- Focus Punch

Blissey @ Leftovers
Ability: Natural Cure
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Soft-Boiled
- Seismic Toss
- Toxic
- Protect

Skarmory @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 252 Def / 4 Spe
Impish Nature
- Drill Peck
- Protect
- Roar
- Spikes
"""

_ENC: Optional[Gen3ObservationEncoder] = None


def _encoder() -> Gen3ObservationEncoder:
    global _ENC
    if _ENC is None:
        _ENC = Gen3ObservationEncoder(mappings=load_mappings())
    return _ENC


def _first(player: Player, battle):
    """The DETERMINISTIC fallback (never `choose_random_move`): a collected test wants the same
    battle every run, so no choice may draw from the global RNG."""
    if battle.available_moves:
        return player.create_order(battle.available_moves[0])
    if battle.available_switches:
        return player.create_order(battle.available_switches[0])
    return player.choose_default_move()


def _pick(battle, move_id: str):
    for m in battle.available_moves:
        if m.id == move_id:
            return m
    return None


class _Recorder(Player):
    """A scripted player that keeps its own RAW protocol (the scenario oracle) and a row per
    decision (the three layers)."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.raw: List[List[str]] = []
        self.rows: List[Dict[str, Any]] = []

    async def _handle_battle_message(self, split_messages):
        for msg in split_messages[1:]:
            if len(msg) >= 2:
                self.raw.append(list(msg))
        await super()._handle_battle_message(split_messages)

    def _row(self, battle, **extra: Any) -> Dict[str, Any]:
        vec = _encoder().encode(battle)
        live = battle.strict_view().live
        row = {"turn": battle.turn, "raw_len": len(self.raw), "vec": vec, "live": live,
               "force_switch": bool(battle.force_switch)}
        row.update(extra)
        self.rows.append(row)
        return row


def _player(cls, name: str, team: str) -> Player:
    tag = f"{int(time.time() * 1000) % 1_000_000}"
    return cls(account_configuration=AccountConfiguration(f"{name}{tag}", "password"),
               battle_format="gen3ou", team=Gen3Teambuilder(team), battle_class=Gen3Battle,
               start_listening=False, max_concurrent_battles=1, log_level=40)


async def _play(p1: Player, p2: Player) -> None:
    await run_local_battles(p1, p2, 1, battle_format="gen3ou", seed=SEED, impl="node")


# ---------------------------------------------------------------------------------- PE-V10

class _CurseThenBoom(_Recorder):
    def choose_move(self, battle):
        active = battle.active_pokemon
        if battle.force_switch:
            self._row(battle, corpse=active.species if active else None,
                      pokeenv={k: v for k, v in (active.boosts if active else {}).items() if v})
            return _first(self, battle)
        self._row(battle)
        if active is not None and active.species == "snorlax":
            if active.boosts.get("atk", 0) < 2 and _pick(battle, "curse"):
                return self.create_order(_pick(battle, "curse"))
            if _pick(battle, "selfdestruct"):
                return self.create_order(_pick(battle, "selfdestruct"))
        return _first(self, battle)


class _Healer(Player):
    def choose_move(self, battle):
        mv = _pick(battle, "softboiled") or _pick(battle, "protect")
        return self.create_order(mv) if mv else _first(self, battle)


def test_pe_v10_a_fainted_active_holds_no_stages_at_the_replacement_decision():
    us, them = _player(_CurseThenBoom, "V10a", SNORLAX_TEAM), _player(_Healer, "V10b", BLISSEY_TEAM)
    asyncio.run(_play(us, them))
    raw = us.raw
    assert any(m[1] == "move" and m[2].startswith("p1a: Snorlax") and m[3] == "Self-Destruct"
               for m in raw), "scenario broke: Snorlax never Self-Destructed"
    assert any(m[1] == "faint" and m[2].startswith("p1a: Snorlax") for m in raw)
    cursed = [m for m in raw if m[1] == "-boost" and m[2].startswith("p1a: Snorlax") and m[3] == "atk"]
    assert len(cursed) >= 2, f"scenario broke: Snorlax boosted {len(cursed)} time(s)"
    before = [r for r in us.rows if not r["force_switch"] and r["live"].ours.active is not None
              and r["live"].ours.active.species == "snorlax"]
    assert decode_obs_boosts(before[-1]["vec"]).get("atk", 0) >= 2, "fixture: +2 Atk encoded alive"
    forced = [r for r in us.rows if r["force_switch"] and r["corpse"] == "snorlax"]
    assert forced, "scenario broke: no replacement decision with the Snorlax corpse in its slot"
    row = forced[0]
    assert row["pokeenv"] == {}, f"poke-env kept the corpse's stages: {row['pokeenv']}"
    active = row["live"].ours.active
    assert active is not None and active.fainted and dict(active.boosts) == {}, active
    assert decode_obs_boosts(row["vec"]) == {}, (
        f"the corpse's stages reached active_context: {decode_obs_boosts(row['vec'])}")


# ---------------------------------------------------------------------------------- PE-R1b

class _PoisonedPivot(_Recorder):
    """Snorlax takes Toxic, pivots to Magikarp, and comes back when Magikarp faints."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.pivoted = False

    def choose_move(self, battle):
        self._row(battle, counts={m.species: (m.status, m.status_counter) for m in battle.team.values()})
        active = battle.active_pokemon
        if battle.force_switch:
            for mon in battle.available_switches:
                if mon.species == "snorlax":
                    return self.create_order(mon)
            return _first(self, battle)
        if (active is not None and active.species == "snorlax" and active.status == Status.TOX
                and not self.pivoted):
            for mon in battle.available_switches:
                if mon.species == "magikarp":
                    self.pivoted = True
                    return self.create_order(mon)
        mv = _pick(battle, "splash") or _pick(battle, "curse")
        return self.create_order(mv) if mv else _first(self, battle)


class _Toxicer(Player):
    """Toxic whatever is unstatused; Seismic Toss a poisoned Snorlax; never attack a poisoned
    Magikarp — so Magikarp faints AT THE RESIDUAL and Snorlax re-enters AFTER it (a mid-turn KO's
    replacement enters before the residual in gen 3, which is not the AHEAD shape)."""

    def choose_move(self, battle):
        foe = battle.opponent_active_pokemon
        if foe is not None and foe.status is None and _pick(battle, "toxic"):
            return self.create_order(_pick(battle, "toxic"))
        if foe is not None and foe.species == "magikarp":
            mv = _pick(battle, "softboiled") or _pick(battle, "protect")
        else:
            mv = _pick(battle, "seismictoss")
        return self.create_order(mv) if mv else _first(self, battle)


def _chips_since_entry(raw: List[List[str]], name: str) -> Optional[int]:
    """The sim's toxic stage of OUR mon ``name``, re-derived from the raw protocol per the pinned
    Showdown `tox`: reset at each switch-in, +1 per non-KO `[from] psn` chip while badly poisoned
    (cap 15); ``None`` while it is benched (the effective stage is then 0)."""
    stage, on_field, tox = 0, False, False
    for m in raw:
        ident = m[2] if len(m) > 2 else ""
        if m[1] in ("switch", "drag"):
            if ident.startswith("p1a: "):
                on_field = ident == f"p1a: {name}"
                if on_field:
                    stage, tox = 0, m[4].endswith(" tox")
        elif ident == f"p1a: {name}":
            if m[1] == "-status":
                tox = m[3] == "tox"
                stage = 0 if tox else stage
            elif m[1] == "-curestatus":
                tox = False
            elif m[1] == "-damage" and "[from] psn" in m[4:] and tox and m[3] != "0 fnt":
                stage = min(stage + 1, 15)
    return stage if on_field else None


def test_pe_r1b_the_toxic_count_is_the_residual_stage_at_every_decision():
    us, them = _player(_PoisonedPivot, "R1ba", SNORLAX_TEAM), _player(_Toxicer, "R1bb", BLISSEY_TEAM)
    asyncio.run(_play(us, them))
    enc = _encoder()
    lay = enc.get_layout()
    our = lay["parts"]["our_team"]
    ctr_off = lay["pokemon"]["status_counters"]["offset"]
    assert us.pivoted, "scenario broke: Snorlax was never badly poisoned and pivoted out"
    checked, ahead_shape = 0, 0
    for row in us.rows:
        raw = us.raw[: row["raw_len"]]
        for i, lm in enumerate(row["live"].ours.mons):
            if lm.status != "tox":
                continue
            want = _chips_since_entry(raw, "Snorlax" if lm.species == "snorlax" else lm.species.capitalize())
            want = 0 if want is None else want
            assert lm.status_counter == want, (
                f"t{row['turn']} {lm.species}: reading {lm.status_counter}, the sim's stage {want}")
            slot = our["start"] + i * our["reshape"][1] + ctr_off + 1
            assert abs(float(row["vec"][slot]) - min(want, 8) / 8.0) < 1e-6, (
                f"t{row['turn']} {lm.species}: obs toxic slot {row['vec'][slot]} vs stage {want}")
            checked += 1
            if lm.species == "snorlax" and lm.active and want == 0:
                # the AHEAD shape: a `|turn|` has passed since this entry (upstream ticked it)
                last_in = max(k for k, m in enumerate(raw) if m[1] == "switch" and m[2] == "p1a: Snorlax")
                if any(m[1] == "turn" for m in raw[last_in:]) and last_in > 0:
                    ahead_shape += 1
    assert checked >= 3, f"only {checked} badly-poisoned decisions were checked"
    assert ahead_shape >= 1, ("scenario broke: no decision where Snorlax re-entered after the "
                              "residual and a |turn| passed with no chip — the PE-R1b AHEAD shape")


# ---------------------------------------------------------------------------------- PE-V16

class _Houndoom(_Recorder):
    def choose_move(self, battle):
        active = battle.active_pokemon
        if active is not None and active.species == "houndoom":
            self._row(battle, effects=set(active.effects))
        if battle.force_switch:
            return _first(self, battle)
        mv = _pick(battle, "flamethrower")
        return self.create_order(mv) if mv else _first(self, battle)


class _Flamer(Player):
    def choose_move(self, battle):
        mv = _pick(battle, "flamethrower")
        return self.create_order(mv) if mv else _first(self, battle)


def test_pe_v16_flash_fire_survives_its_holders_own_fire_move():
    us, them = _player(_Houndoom, "V16a", HOUNDOOM_TEAM), _player(_Flamer, "V16b", FIRE_TEAM)
    asyncio.run(_play(us, them))
    raw = us.raw
    started = [k for k, m in enumerate(raw)
               if m[1] == "-start" and m[2] == "p1a: Houndoom" and m[3] == "ability: Flash Fire"]
    assert started, "scenario broke: Flash Fire never activated"
    own_fire = [k for k, m in enumerate(raw)
                if k > started[0] and m[1] == "move" and m[2] == "p1a: Houndoom" and m[3] == "Flamethrower"]
    assert own_fire, "scenario broke: Houndoom never used its own Fire move after the activation"
    slot = OFFSET_CONTEXT + BOOSTS_DIM + VOLATILE_SLOTS.index("flashfire")
    after = [r for r in us.rows if r["raw_len"] > own_fire[0] and not r["live"].ours.active.fainted
             and not any(m[1] in ("switch", "drag") and m[2].startswith("p1a: ")
                         for m in raw[started[0]:r["raw_len"]])]
    assert after, "scenario broke: no Houndoom decision after its own Fire move"
    for row in after:
        assert Effect.FLASH_FIRE in row["effects"], f"t{row['turn']}: poke-env ended Flash Fire"
        assert "flashfire" in row["live"].ours.active.volatiles, row["live"].ours.active.volatiles
        assert float(row["vec"][slot]) == 1.0, f"t{row['turn']}: obs flashfire slot {row['vec'][slot]}"


if __name__ == "__main__":
    failed = 0
    for fn in (test_pe_v10_a_fainted_active_holds_no_stages_at_the_replacement_decision,
               test_pe_r1b_the_toxic_count_is_the_residual_stage_at_every_decision,
               test_pe_v16_flash_fire_survives_its_holders_own_fire_move):
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except AssertionError as exc:  # pragma: no cover - script path
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
    sys.exit(1 if failed else 0)
