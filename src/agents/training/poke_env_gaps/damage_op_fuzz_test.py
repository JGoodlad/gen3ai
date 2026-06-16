"""Bridge fuzz test for the differentiable DamageOperator's PHYSICS — confirms the op's predicted gen3
damage band CONTAINS Showdown's REALIZED damage, on thousands of real moves from random real battles.

  ⚠️ SECONDARY / BROAD-COVERAGE check. The AUTHORITATIVE physics gate is the CONSTRUCTED single-turn
  oracle `damage_op_probe_fuzz_test.py`, which drives the OMNISCIENT BattleStream and reads EXACT both-
  side HP + the sim's OWN stats (zero measurement confounds). THIS file scrapes damage from RANDOM games
  via the per-side protocol, where the defender's HP arrives in INTEGER PERCENT, `before`-HP goes stale
  on untracked chip, and overkill KOs cap the realized drop — so a handful of out-of-band cases are
  EXPECTED measurement noise, NOT op bugs (verified: cases that read "wrong" here land exactly in band
  in the exact-HP probe). Its value is broad random species/move coverage + a no-catastrophic-regression
  net; the band threshold is deliberately loose. Use the probe to adjudicate any real physics question.

Unlike the unit tests (synthetic inputs → exact ratios) and the op-vs-CPU parity test (op == the
Showdown-fuzz-validated `incoming_damage` reference), this validates the op's actual kernel + modifier
helpers END-TO-END against the SIM's realized damage. It is OK to use BOTH sides' exact stats here: we
are validating the PHYSICS LOGIC (formula + boosts/burn/weather/screens/STAB/type/fixed-damage), not the
belief — the model estimates hidden stats in the real game; the fuzz just needs to know the truth.

How it works (in-process via the local BattleStream bridge — no server):
  - Two players play real gen3ou battles. Each posts its OWN team's exact computed stats into a shared
    registry (keyed by species), so any (attacker, defender) hit can be resolved with TRUE stats.
  - `_handle_battle_message` intercepts the protocol: each `|move|` (actor + move) followed by a direct
    `|-damage|` on the target (no `[from]` residual/item tag) is one HIT; `|-crit|` flags a crit. The
    realized damage fraction is the target's HP drop on that line.
  - Per clean hit, the op's actual kernel (`DamageOperator._damage_rolls`) + modifier helpers
    (`_boost_mult` / `_weather_mult`, burn) are run with the TRUE both-side stats + the live board
    modifiers (weather / boosts / burn / screens), producing the [low_roll, high_roll, crit] band.
  - INVARIANT: the realized fraction sits inside the band — `low·(1−ε) ≤ realized ≤ (crit if crit else
    high)·(1+ε)`. A violation means the op's physics disagrees with Showdown on a real move.

Filtered out (not physics-validatable from the HP delta alone): multi-hit moves, moves whose damage is
confounded by same-line residuals, fixed-damage (validated separately), and unresolved species/moves.

Run directly (no `test_*` funcs → pytest collects nothing):
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/poke_env_gaps/damage_op_fuzz_test.py [n_battles]
"""
from __future__ import annotations

import asyncio
import sys
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from poke_env import AccountConfiguration
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents import gen3_data
from agents.model.features_extractor import (
    DamageOperator, _DMG_PARA_SPEED, _DMG_ROLL_MIN, _DMG_CHIP_CAP as _CHIP_CAP, _DMG_CRIT_CAP as _CRIT_CAP)
from agents.model.damage_tables import build_move_fixed_damage
from agents.observation.types import TypeEncoder
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder
from utils.bridge.local_battle_runner import run_local_battles

BATTLE_FORMAT = "gen3ou"
_T2I = TypeEncoder.TYPE_TO_IDX
_TOL = 0.10                # band tolerance (the op is a SMOOTH/un-floored belief vs the floored sim)
_PHYS_TYPES = {"NORMAL", "FIGHTING", "POISON", "GROUND", "FLYING", "BUG", "ROCK", "GHOST", "STEEL"}

# Shared across both players: (battle_tag, side, species) → its TRUE computed stats {hp,atk,def,spa,spd,
# spe}. Each player posts its OWN side (full info is fine for a physics check). Keyed by side so a MIRROR
# match (Suicune vs Suicune, common in gen3 OU) doesn't collide onto one spread.
_STATS_REGISTRY: Dict[tuple, Dict[str, int]] = {}
_DBG: Dict[str, object] = {}                  # last _op_band intermediates (for out-of-band diagnostics)
_OP = DamageOperator(Gen3ObservationEncoder(load_mappings()).get_layout())
_FIXED = build_move_fixed_damage(_OP.MOVE_BP.shape[0])


@dataclass
class Stats:
    name: str
    hits_checked: int = 0
    in_band: int = 0
    below: int = 0
    above: int = 0
    killed: int = 0          # overkill KOs — realized capped at remaining HP, so the lower bound is moot
    skipped: int = 0
    examples: List[dict] = field(default_factory=list)

    def record(self, ex: dict) -> None:
        if len(self.examples) < 20:
            self.examples.append(ex)


def _type_idx(name: Optional[str]) -> int:
    if not name:
        return 0
    return _T2I.get(str(name).upper(), 0)


# Moves whose damage isn't a flat-BP roll vs the registered stats (deferred op edges / variable power):
# Explosion/Self-Destruct halve the target Def (gen≤4); Return/Frustration/Flail/etc. are variable-power.
_SKIP_MOVES = {"explosion", "selfdestruct", "return", "frustration", "flail", "reversal",
               "lowkick", "magnitude", "psywave", "counter", "mirrorcoat", "bide", "hiddenpower",
               # Beat Up: one hit per non-fainted party member off ITS base Atk vs the target's base Def
               # (gen3) — not a flat-BP roll vs the active's stats. Triplekick/Rollout/Fury* multi-hit too.
               "beatup", "triplekick", "rollout", "iceball", "furyswipes", "furyattack", "doublekick",
               "bonemerang", "twineedle", "spikecannon", "barrage", "cometpunch", "doubleslap", "pinmissile"}

# Gen3 held items that SCALE damage. The op deliberately doesn't model items (unknown in-game), but the
# fuzz has full info on the ATTACKER's item (we only validate our OWN attacks — see _scan), so we apply
# the true multiplier to keep the band TIGHT rather than widening it (which would mask a real physics bug).
_TYPE_BOOST_ITEMS = {            # gen3 ×1.1 type-boosting held items, item_id → boosted TYPE name
    "silkscarf": "NORMAL", "charcoal": "FIRE", "mysticwater": "WATER", "magnet": "ELECTRIC",
    "miracleseed": "GRASS", "nevermeltice": "ICE", "blackbelt": "FIGHTING", "poisonbarb": "POISON",
    "softsand": "GROUND", "sharpbeak": "FLYING", "twistedspoon": "PSYCHIC", "silverpowder": "BUG",
    "hardstone": "ROCK", "spelltag": "GHOST", "dragonfang": "DRAGON", "blackglasses": "DARK",
    "metalcoat": "STEEL",
}


def _item_damage_mult(item: Optional[str], move_type_name: str, is_phys: bool, species: str) -> float:
    """Gen3 damage multiplier from the ATTACKER's held item (full-info fuzz). Stat-doublers (Choice Band,
    Thick Club, Light Ball, Soul Dew, Deep Sea Tooth) and the ×1.1 type-boost items all end up scaling
    final damage, so they fold into one multiplicative factor applied to both the normal + crit rolls."""
    if not item:
        return 1.0
    it = "".join(c for c in str(item).lower() if c.isalnum())
    if it == "choiceband" and is_phys:
        return 1.5
    if it == "souldew" and not is_phys and species in ("latios", "latias"):
        return 1.5
    if it == "thickclub" and is_phys and species in ("cubone", "marowak"):
        return 2.0
    if it == "lightball" and not is_phys and species == "pikachu":     # gen3 Light Ball = ×2 SpA only
        return 2.0
    if it == "deepseatooth" and not is_phys and species == "clamperl":
        return 2.0
    if it == "seaincense" and move_type_name == "WATER":               # gen3 Sea Incense = ×1.05
        return 1.05
    if _TYPE_BOOST_ITEMS.get(it) == move_type_name:
        return 1.1
    return 1.0


def _op_band(move_id, attacker, defender, weather, screens, tag, atk_side, def_side):
    """Run the op's ACTUAL kernel + modifier helpers with TRUE stats + live board → (low, high, crit) as
    a fraction of the defender's max HP. Returns None when not physics-validatable (fixed/0-BP/variable,
    no stats)."""
    md = gen3_data.moves.get(move_id)
    if md is None or md.base_power <= 0 or move_id in _SKIP_MOVES:
        return None
    a_stats = _STATS_REGISTRY.get((tag, atk_side, attacker["species"]))
    d_stats = _STATS_REGISTRY.get((tag, def_side, defender["species"]))
    if a_stats is None or d_stats is None:
        return None
    mty = _type_idx(md.type.name if md.type.name != "THREE_QUESTION_MARKS" else None)
    if mty == 0:
        return None
    is_phys = md.type.name in _PHYS_TYPES
    # offense + defence with stat-stage boosts (the op's OWN helper) + burn (½ phys).
    off = float(a_stats["atk"] if is_phys else a_stats["spa"])
    off_stage = attacker["boosts"].get("atk" if is_phys else "spa", 0)
    off *= _OP._boost_mult(torch.tensor([float(off_stage)])).item()
    if is_phys and attacker.get("burn"):
        off *= 0.5
    dfn = float(d_stats["def"] if is_phys else d_stats["spd"])
    def_stage = defender["boosts"].get("def" if is_phys else "spd", 0)
    dfn *= _OP._boost_mult(torch.tensor([float(def_stage)])).item()
    maxhp = float(d_stats["hp"])
    # STAB + type effectiveness (the op's CHART buffer) + screens + weather (op helper).
    at1, at2 = _type_idx(attacker.get("t1")), _type_idx(attacker.get("t2"))
    stab = 1.5 if mty in (at1, at2) else 1.0
    eff = _OP.CHART[_type_idx(defender.get("t1")), mty].item() * _OP.CHART[_type_idx(defender.get("t2")), mty].item()
    if eff <= 0:
        return (0.0, 0.0, 0.0)
    screen = 0.5 if ((is_phys and screens.get("reflect")) or (not is_phys and screens.get("lightscreen"))) else 1.0
    wf = torch.zeros(1, 7)
    if weather == "sun":  wf[0, 1] = 1.0
    if weather == "rain": wf[0, 2] = 1.0
    wmult = _OP._weather_mult(wf, torch.tensor([[1.0 if md.type.name == "WATER" else 0.0]]),
                              torch.tensor([[1.0 if md.type.name == "FIRE" else 0.0]]))[0, 0].item()
    # ATTACKER item (true, our own active → known): Choice Band / type-boost / etc. scale final damage.
    item_mult = _item_damage_mult(attacker.get("item"), md.type.name, is_phys, attacker["species"])
    core = 42.0 * md.base_power * off / (dfn + 1e-6) / 50.0 + 2.0
    # Mirror DamageOperator._rolls EXACTLY: dmg_ns = pre-screen MAX roll (×1.0, NOT a 0.925 mean);
    # high = post-screen max roll, low = 0.85·high, crit = ×2 the PRE-screen roll (ignores screens),
    # each clamped to the op's chip/crit caps.
    dmg_ns = core * stab * eff * wmult * item_mult / maxhp
    high = min(dmg_ns * screen, _CHIP_CAP)
    low = min(_DMG_ROLL_MIN * dmg_ns * screen, _CHIP_CAP)
    crit = min(2.0 * dmg_ns, _CRIT_CAP)
    _DBG.clear()
    _DBG.update(bp=md.base_power, phys=is_phys, off=round(off, 1), dfn=round(dfn, 1),
                maxhp=round(maxhp, 1), eff=eff, stab=stab, wmult=round(wmult, 3), im=item_mult)
    return (low, high, crit)


class DmgOpFuzzPlayer(Player):
    def __init__(self, scenario_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = Stats(name=scenario_name)
        self._hp: Dict[str, float] = {}          # mon ident → last-known HP fraction
        self._last_move = None                   # (actor_ident, move_id)
        self._crit = False

    def choose_move(self, battle):
        return self.choose_random_move(battle)         # physics fuzz — policy is irrelevant

    def _register_team(self, battle) -> None:
        role = battle.player_role            # 'p1' / 'p2' — our OWN side in this battle
        if role is None:
            return
        for mon in battle.team.values():
            sp = getattr(mon, "species", None)
            st = getattr(mon, "stats", None)
            if sp and st and all(st.get(k) for k in ("hp", "atk", "def", "spa", "spd", "spe")):
                _STATS_REGISTRY.setdefault(
                    (battle.battle_tag, role, sp),
                    {k: int(st[k]) for k in ("hp", "atk", "def", "spa", "spd", "spe")})

    def _mon_view(self, battle, ident: str) -> Optional[dict]:
        """Resolve a `p1a: Gengar`-style ident to a board snapshot. Returns None if the active mon's
        species no longer matches the ident (a SWITCH happened in the batch → the active's boosts/types
        would be the wrong mon's), so we only validate hits whose board state is unambiguous."""
        side_ours = ident.startswith(self._self_prefix(battle))
        mon = (battle.active_pokemon if side_ours else battle.opponent_active_pokemon)
        if mon is None or getattr(mon, "species", None) is None:
            return None
        ident_species = "".join(c for c in ident.split(":", 1)[1].lower() if c.isalnum()) if ":" in ident else ""
        if ident_species and mon.species != ident_species:    # switch confound → board state is stale
            return None
        st = getattr(mon, "status", None)
        types = [t.name for t in (mon.types or []) if t is not None]
        return {
            "species": mon.species,
            "t1": types[0] if len(types) > 0 else None,
            "t2": types[1] if len(types) > 1 else None,
            "boosts": dict(getattr(mon, "boosts", {}) or {}),
            "burn": (getattr(st, "name", "") == "BRN"),
            "item": getattr(mon, "item", None),     # true for our OWN active (we only check our attacks)
            "side_ours": side_ours,
        }

    def _self_prefix(self, battle) -> str:
        return "p1a:" if battle.player_role == "p1" else "p2a:"

    async def _handle_battle_message(self, split_messages):
        # poke-env updates the battle state in super(); we scan AFTER so HP/board reflect this batch.
        await super()._handle_battle_message(split_messages)
        try:
            self._scan(split_messages)
        except Exception:
            pass

    def _get_battle_tag(self, split_messages) -> Optional[str]:
        for m in split_messages:
            if m and m[0].startswith(">"):
                return m[0][1:]
        return None

    def _scan(self, split_messages) -> None:
        bt = self._get_battle_tag(split_messages)
        battle = self._battles.get(bt) if bt else None
        if battle is None:
            return
        self._register_team(battle)
        for msg in split_messages:
            if len(msg) < 2:
                continue
            tag = msg[1]
            if tag == "move" and len(msg) >= 4:
                move_id = "".join(c for c in msg[3].lower() if c.isalnum())   # "Hidden Power" → "hiddenpower"
                self._last_move = (msg[2].strip(), move_id)
                self._crit = False
            elif tag == "-crit":
                self._crit = True
            elif tag == "-damage" and len(msg) >= 4 and self._last_move is not None:
                target = msg[2].strip()
                # residual / item / ability damage carries a [from] tag → not direct move damage.
                if any(p.startswith("[from]") for p in msg[3:]):
                    self._record_hp(target, msg[3])
                    continue
                actor, move_id = self._last_move
                if target == actor:                  # recoil / self-hit
                    self._record_hp(target, msg[3])
                    continue
                before = self._hp.get(target)
                after = self._parse_hp(msg[3])
                self._record_hp(target, msg[3])
                if before is None or after is None or before <= after:
                    continue
                # Only validate OUR OWN attacks: the attacker's item (Choice Band etc.) + stats are then
                # truthfully known, and each hit is checked once (the OTHER player checks its own attacks).
                if not actor.startswith(self._self_prefix(battle)):
                    self._last_move = None
                    continue
                realized = before - after
                self._check(battle, actor, target, move_id, realized, before, after)
                self._last_move = None
            elif tag in ("switch", "drag", "-heal", "faint"):
                if len(msg) >= 4:
                    self._record_hp(msg[2].strip(), msg[3])

    def _parse_hp(self, field: str) -> Optional[float]:
        f = field.split()[0]
        if f in ("0", "0 fnt") or f.startswith("0 "):
            return 0.0
        if "/" in f:
            try:
                cur, mx = f.split("/")
                return float(cur) / float(mx)
            except Exception:
                return None
        return None

    def _record_hp(self, ident: str, field: str) -> None:
        v = self._parse_hp(field)
        if v is not None:
            self._hp[ident] = v

    def _check(self, battle, actor_ident, target_ident, move_id, realized, before=None, after=None) -> None:
        s = self.stats
        attacker = self._mon_view(battle, actor_ident)
        defender = self._mon_view(battle, target_ident)
        if attacker is None or defender is None:
            s.skipped += 1
            return
        weather = None
        w = getattr(battle, "weather", None)
        if w:
            wn = str(list(w)[0] if isinstance(w, dict) else w).lower()
            weather = "sun" if "sun" in wn else ("rain" if "rain" in wn else None)
        sc = battle.opponent_side_conditions if defender["side_ours"] is False else battle.side_conditions
        screens = {"reflect": any("REFLECT" in str(k) for k in (sc or {})),
                   "lightscreen": any("LIGHT_SCREEN" in str(k) for k in (sc or {}))}
        atk_side, def_side = actor_ident[:2], target_ident[:2]      # 'p1a: X' → 'p1'
        band = _op_band(move_id, attacker, defender, weather, screens, battle.battle_tag, atk_side, def_side)
        if band is None:
            s.skipped += 1
            return
        low, high, crit = band
        s.hits_checked += 1
        upper = (crit if self._crit else high) * (1.0 + _TOL)
        lower = low * (1.0 - _TOL)
        ex = {"move": move_id, "atk": attacker["species"], "def": defender["species"],
              "realized": round(realized, 3), "band": (round(low, 3), round(high, 3)), "crit": self._crit,
              "wx": weather, "scr": [k for k, v in screens.items() if v] or None,
              "item": attacker.get("item"), "ab": {k: v for k, v in attacker["boosts"].items() if v},
              "db": {k: v for k, v in defender["boosts"].items() if v},
              "hp": (round(before, 3) if before is not None else None,
                     round(after, 3) if after is not None else None), "dbg": dict(_DBG)}
        fainted = (after is not None and after <= 0.0)
        if realized > upper:
            # Valid even on a KO: realized ≤ remaining-HP ≤ true damage, so realized>upper ⇒ a genuine
            # under-read (the op's max roll couldn't reach the realized damage). The op physics is too weak.
            s.above += 1; s.record({**ex, "verdict": "ABOVE"})
        elif fainted:
            # Overkill: realized is CAPPED at the target's remaining HP, so it reads below the band by
            # construction — not a physics error. Can't validate the lower bound on a KO; count separately.
            s.killed += 1
        elif realized < lower and high > 0.02:    # ignore tiny chip (rounding on % HP)
            s.below += 1; s.record({**ex, "verdict": "BELOW"})
        else:
            s.in_band += 1


async def run_battles(n_battles: int) -> Stats:
    pool = TeamLoader().get_all_teams() or TeamLoader().get_sample_teams()
    tb = Gen3Teambuilder(pool)
    p1 = DmgOpFuzzPlayer("p1", account_configuration=AccountConfiguration("DmgOpFuzz1", None),
                         battle_format=BATTLE_FORMAT, team=tb,
                         server_configuration=LocalhostServerConfiguration, start_listening=False)
    p2 = DmgOpFuzzPlayer("p2", account_configuration=AccountConfiguration("DmgOpFuzz2", None),
                         battle_format=BATTLE_FORMAT, team=tb,
                         server_configuration=LocalhostServerConfiguration, start_listening=False)
    await run_local_battles(p1, p2, n_battles=n_battles)
    merged = Stats(name="merged")
    for p in (p1, p2):
        merged.hits_checked += p.stats.hits_checked
        merged.in_band += p.stats.in_band
        merged.below += p.stats.below
        merged.above += p.stats.above
        merged.killed += p.stats.killed
        merged.skipped += p.stats.skipped
        merged.examples += p.stats.examples
    return merged


def main(n_battles: int = 40) -> int:
    print(f"=== DamageOperator physics fuzz: {n_battles} bridge battles ===")
    s = asyncio.get_event_loop().run_until_complete(run_battles(n_battles))
    # `killed` (overkill KOs) and `skipped` (unresolved/variable/fixed) aren't full-band-validatable; the
    # validatable population is in_band+below+above (lower bound checkable ⇒ the target survived).
    validatable = s.in_band + s.below + s.above
    print(f"hits checked: {s.hits_checked} | IN-BAND: {s.in_band} | below: {s.below} | above: {s.above} "
          f"| killed(overkill): {s.killed} | skipped: {s.skipped}")
    rate = s.in_band / max(1, validatable)
    above_frac = s.above / max(1, validatable)
    print(f"in-band rate (of validatable): {rate:.3f} | under-read (ABOVE) frac: {above_frac:.3f}")
    bad = [e for e in s.examples if e.get("verdict") in ("ABOVE", "BELOW")]
    for e in bad[:14]:
        print("  OUT-OF-BAND:", e)
    # LOOSE secondary threshold (the probe is the authoritative gate — see the module docstring). The
    # in-percent / stale-HP / overkill confounds put a hard floor of ~10-15% on the out-of-band rate that
    # NO op fix can remove, so we only flag a CATASTROPHIC regression here (a real physics bug — a missing
    # weather/boost/screen/STAB mult — collapses the rate far below this). Adjudicate specifics in the probe.
    ok = validatable >= 50 and rate >= 0.75
    print("RESULT:", "PASS" if ok else "FAIL", "(loose net — adjudicate physics in damage_op_probe_fuzz_test.py)")
    return 0 if ok else 1


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    try:
        sys.exit(main(n))
    except Exception:
        traceback.print_exc()
        sys.exit(2)
